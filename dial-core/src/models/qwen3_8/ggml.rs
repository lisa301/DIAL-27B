//! Upstream GGML executor. DIAL retains topology, transport and sampling;
//! quantized weights never enter Candle. See backends/qwen38_ggml/adapter.h.
use super::TextConfig;
use anyhow::{Context as _, Result};
use candle_core::{CpuStorage, CustomOp1, DType, Device, Layout, Shape, Tensor};
use libloading::Library;
use std::{
    ffi::{c_char, c_void, CStr, CString},
    path::Path,
    sync::{Arc, Mutex},
};

#[repr(C)]
struct AbiConfig {
    hidden: i64,
    intermediate: i64,
    vocab: i64,
    layers: i64,
    q_heads: i64,
    kv_heads: i64,
    head_dim: i64,
    rotary_dim: i64,
    max_seq: i64,
    conv_kernel: i64,
    key_dim: i64,
    key_heads: i64,
    value_dim: i64,
    value_heads: i64,
    norm_eps: f32,
    rope_theta: f32,
}
type Open = unsafe extern "C" fn(
    *const c_char,
    *const AbiConfig,
    i32,
    i32,
    *mut c_char,
    usize,
) -> *mut c_void;
type Close = unsafe extern "C" fn(*mut c_void);
type Prepare = unsafe extern "C" fn(*mut c_void, i32, i32, *mut c_char, usize) -> i32;
type PrepareHead = unsafe extern "C" fn(*mut c_void, *mut c_char, usize) -> i32;
type NewState = unsafe extern "C" fn(*mut c_void, i32, *mut c_char, usize) -> *mut c_void;
type NewRange =
    unsafe extern "C" fn(*mut c_void, *const *mut c_void, i64, *mut c_char, usize) -> *mut c_void;
type Forward = unsafe extern "C" fn(
    *mut c_void,
    *mut c_void,
    *const f32,
    i64,
    i64,
    *mut f32,
    i32,
    *mut c_char,
    usize,
) -> i32;
type Embed =
    unsafe extern "C" fn(*mut c_void, *const i32, i64, *mut f32, i32, *mut c_char, usize) -> i32;
type Head = unsafe extern "C" fn(*mut c_void, *const f32, *mut f32, i32, *mut c_char, usize) -> i32;
type Stats = unsafe extern "C" fn(*mut c_void) -> u64;

pub struct GgmlEngine {
    _library: Library,
    handle: usize,
    // Serializes backend/graph allocation, execution and state destruction.
    gate: Mutex<()>,
    close: Close,
    prepare: Prepare,
    prepare_head: PrepareHead,
    new_state: NewState,
    free_state: Close,
    new_range: NewRange,
    free_range: Close,
    forward: Forward,
    forward_range: Forward,
    embed: Embed,
    head: Head,
    weight_bytes: Stats,
    weight_tensors: Stats,
    graph_calls: Stats,
    range_calls: Stats,
    cuda_graphs_available: Stats,
    hidden: usize,
    vocab: usize,
    layer_types: Vec<String>,
    device: Device,
    fused_shards: bool,
}
impl std::fmt::Debug for GgmlEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GgmlEngine")
            .field("hidden", &self.hidden)
            .field("device", &self.device)
            .finish_non_exhaustive()
    }
}
fn ffi_error(error: &[c_char]) -> candle_core::Error {
    let message = unsafe { CStr::from_ptr(error.as_ptr()) }.to_string_lossy();
    candle_core::Error::Msg(format!("Qwen3.8 GGML: {message}"))
}
impl GgmlEngine {
    pub fn open(
        library: &Path,
        gguf: &Path,
        text: &TextConfig,
        max_seq: usize,
        device: &Device,
    ) -> Result<Arc<Self>> {
        text.validate()?;
        anyhow::ensure!(
            device.is_cpu() || device.is_cuda(),
            "GGML adapter supports CPU/CUDA only"
        );
        let c = AbiConfig {
            hidden: text.hidden_size as i64,
            intermediate: text.intermediate_size as i64,
            vocab: text.vocab_size as i64,
            layers: text.num_hidden_layers as i64,
            q_heads: text.num_attention_heads as i64,
            kv_heads: text.num_key_value_heads as i64,
            head_dim: text.head_dim as i64,
            rotary_dim: text.rotary_dim() as i64,
            max_seq: max_seq as i64,
            conv_kernel: text.linear_conv_kernel_dim as i64,
            key_dim: text.linear_key_head_dim as i64,
            key_heads: text.linear_num_key_heads as i64,
            value_dim: text.linear_value_head_dim as i64,
            value_heads: text.linear_num_value_heads as i64,
            norm_eps: text.rms_norm_eps as f32,
            rope_theta: text.rope_theta(),
        };
        let path = CString::new(gguf.as_os_str().as_encoded_bytes())?;
        // Explicit opt-in; retain Library until all engines/states are destroyed.
        let lib = unsafe { Library::new(library) }.with_context(|| {
            format!(
                "can't load GGML adapter {} (build tools/build_qwen38_ggml.sh locally)",
                library.display()
            )
        })?;
        unsafe {
            let version = *lib.get::<unsafe extern "C" fn() -> u32>(b"dial_ggml_abi_version\0")?;
            anyhow::ensure!(version() == 2,
                "Unsupported DIAL GGML adapter ABI; rebuild tools/build_qwen38_ggml.sh on BOTH nodes (fused shards require ABI 2)");
            let open = *lib.get::<Open>(b"dial_ggml_open\0")?;
            let mut e = Self {
                close: *lib.get(b"dial_ggml_close\0")?,
                prepare: *lib.get(b"dial_ggml_prepare_layer\0")?,
                prepare_head: *lib.get(b"dial_ggml_prepare_head\0")?,
                new_state: *lib.get(b"dial_ggml_create_state\0")?,
                free_state: *lib.get(b"dial_ggml_destroy_state\0")?,
                new_range: *lib.get(b"dial_ggml_create_range\0")?,
                free_range: *lib.get(b"dial_ggml_destroy_range\0")?,
                forward: *lib.get(b"dial_ggml_forward\0")?,
                forward_range: *lib.get(b"dial_ggml_forward_range\0")?,
                embed: *lib.get(b"dial_ggml_embed\0")?,
                head: *lib.get(b"dial_ggml_head\0")?,
                weight_bytes: *lib.get(b"dial_ggml_weight_bytes\0")?,
                weight_tensors: *lib.get(b"dial_ggml_weight_tensors\0")?,
                graph_calls: *lib.get(b"dial_ggml_graph_calls\0")?,
                range_calls: *lib.get(b"dial_ggml_range_calls\0")?,
                cuda_graphs_available: *lib.get(b"dial_ggml_cuda_graphs_available\0")?,
                _library: lib,
                handle: 0,
                gate: Mutex::new(()),
                hidden: text.hidden_size,
                vocab: text.vocab_size,
                layer_types: text.layer_types.clone(),
                device: device.clone(),
                fused_shards: std::env::var("DIAL_GGML_FUSED").as_deref() != Ok("0"),
            };
            let mut error = [0; 2048];
            let dev = match device {
                Device::Cpu => -1,
                #[cfg(feature = "cuda")]
                Device::Cuda(d) => d.cuda_stream().context().ordinal() as i32,
                _ => anyhow::bail!("unsupported GGML device"),
            };
            e.handle = open(path.as_ptr(), &c, dev, 4, error.as_mut_ptr(), error.len()) as usize;
            if e.handle == 0 {
                return Err(ffi_error(&error).into());
            }
            log::info!(
                "Qwen3.8 GGML ready: device={device:?}, GGUF={}; fused_shards={} cuda_graphs_available={} active_kv_bucket=256; assigned weights loaded lazily",
                gguf.display(), e.fused_shards, (e.cuda_graphs_available)(e.ptr()) != 0
            );
            Ok(Arc::new(e))
        }
    }
    fn ptr(&self) -> *mut c_void {
        self.handle as *mut c_void
    }
    pub fn fused_shards_enabled(&self) -> bool {
        self.fused_shards
    }
    pub fn prepare_layer(&self, index: usize) -> Result<()> {
        let ty = self
            .layer_types
            .get(index)
            .ok_or_else(|| anyhow!("GGML layer index {index} out of range"))?;
        let _guard = self
            .gate
            .lock()
            .map_err(|_| anyhow!("GGML lock poisoned"))?;
        let mut error = [0; 2048];
        if unsafe {
            (self.prepare)(
                self.ptr(),
                index as i32,
                i32::from(ty == "linear_attention"),
                error.as_mut_ptr(),
                error.len(),
            )
        } != 0
        {
            return Err(ffi_error(&error).into());
        }
        Ok(())
    }
    pub fn prepare_head(&self) -> Result<()> {
        let _guard = self
            .gate
            .lock()
            .map_err(|_| anyhow!("GGML lock poisoned"))?;
        let mut error = [0; 2048];
        if unsafe { (self.prepare_head)(self.ptr(), error.as_mut_ptr(), error.len()) } != 0 {
            return Err(ffi_error(&error).into());
        }
        Ok(())
    }
    pub fn summary(&self) -> (u64, u64) {
        let _guard = self.gate.lock().unwrap_or_else(|e| e.into_inner());
        unsafe {
            (
                (self.weight_tensors)(self.ptr()),
                (self.weight_bytes)(self.ptr()),
            )
        }
    }
    /// Successful graph executions and fused-shard executions (not timings).
    pub fn execution_counts(&self) -> (u64, u64) {
        let _guard = self.gate.lock().unwrap_or_else(|e| e.into_inner());
        unsafe {
            (
                (self.graph_calls)(self.ptr()),
                (self.range_calls)(self.ptr()),
            )
        }
    }
    fn state(self: &Arc<Self>, index: usize) -> Result<Arc<GgmlState>> {
        let _guard = self
            .gate
            .lock()
            .map_err(|_| anyhow!("GGML lock poisoned"))?;
        let mut error = [0; 2048];
        let handle =
            unsafe { (self.new_state)(self.ptr(), index as i32, error.as_mut_ptr(), error.len()) }
                as usize;
        if handle == 0 {
            return Err(ffi_error(&error).into());
        }
        Ok(Arc::new(GgmlState {
            engine: self.clone(),
            handle,
        }))
    }
    pub fn embedding(self: &Arc<Self>, tokens: &Tensor) -> Result<Tensor> {
        let (batch, n) = tokens.dims2()?;
        anyhow::ensure!(batch == 1 && n > 0, "GGML currently supports batch size 1");
        let ids = tokens
            .flatten_all()?
            .to_vec1::<u32>()?
            .into_iter()
            .map(|id| i32::try_from(id).map_err(anyhow::Error::from))
            .collect::<Result<Vec<_>>>()?;
        let op = Op {
            engine: self.clone(),
            kind: Kind::Embed(ids),
            shape: Shape::from((1, n, self.hidden)),
        };
        Ok(Tensor::zeros((1,), DType::F32, &self.device)?.apply_op1_no_bwd(&op)?)
    }
    pub fn logits(self: &Arc<Self>, x: &Tensor) -> Result<Tensor> {
        anyhow::ensure!(
            x.elem_count() == self.hidden && x.device().same_device(&self.device),
            "GGML output head expects one row on its configured device"
        );
        let op = Op {
            engine: self.clone(),
            kind: Kind::Head,
            shape: Shape::from((1, self.vocab)),
        };
        Ok(x.to_dtype(DType::F32)?
            .contiguous()?
            .apply_op1_no_bwd(&op)?)
    }
    pub fn layer(
        self: &Arc<Self>,
        index: usize,
        position: usize,
        x: &Tensor,
        cache: &mut crate::models::llama3::Cache,
    ) -> Result<Tensor> {
        let (batch, n, hidden) = x.dims3()?;
        anyhow::ensure!(
            batch == 1 && n > 0 && hidden == self.hidden,
            "GGML expects [1, tokens, {}], got {:?}",
            self.hidden,
            x.dims()
        );
        anyhow::ensure!(
            x.device().same_device(&self.device),
            "GGML layer/Candle device mismatch"
        );
        let state = self.cached_state(index, cache)?;
        let op = Op {
            engine: self.clone(),
            kind: Kind::Layer { state, n, position },
            shape: x.shape().clone(),
        };
        // Device-local casts, not host roundtrips; retain F16 wire boundaries.
        Ok(x.to_dtype(DType::F32)?
            .contiguous()?
            .apply_op1_no_bwd(&op)?
            .to_dtype(x.dtype())?)
    }
    fn cached_state(
        self: &Arc<Self>,
        index: usize,
        cache: &mut crate::models::llama3::Cache,
    ) -> Result<Arc<GgmlState>> {
        if let Some(state) = cache.ggml_states.get(&index) {
            anyhow::ensure!(
                Arc::ptr_eq(&state.engine, self),
                "GGML cache belongs to another engine"
            );
            Ok(state.clone())
        } else {
            let state = self.state(index)?;
            cache.ggml_states.insert(index, state.clone());
            Ok(state)
        }
    }
    /// Execute a consecutive local shard as ONE GGML graph. Internal layer
    /// activations stay in GGML/F32; only the shard's F16 wire boundary is cast.
    pub fn range(
        self: &Arc<Self>,
        first: usize,
        count: usize,
        position: usize,
        x: &Tensor,
        cache: &mut crate::models::llama3::Cache,
    ) -> Result<Tensor> {
        let (batch, n, hidden) = x.dims3()?;
        anyhow::ensure!(
            batch == 1 && n > 0 && hidden == self.hidden,
            "GGML shard expects [1, tokens, {}], got {:?}",
            self.hidden,
            x.dims()
        );
        anyhow::ensure!(
            x.device().same_device(&self.device),
            "GGML shard/Candle device mismatch"
        );
        anyhow::ensure!(
            count > 0
                && first
                    .checked_add(count)
                    .is_some_and(|end| end <= self.layer_types.len()),
            "GGML shard layer range out of bounds"
        );
        let key = (first, count);
        let range = if let Some(range) = cache.ggml_ranges.get(&key) {
            anyhow::ensure!(
                Arc::ptr_eq(&range.engine, self),
                "GGML shard cache belongs to another engine"
            );
            range.clone()
        } else {
            let states = (first..first + count)
                .map(|index| self.cached_state(index, cache))
                .collect::<Result<Vec<_>>>()?;
            let handles = states
                .iter()
                .map(|s| s.handle as *mut c_void)
                .collect::<Vec<_>>();
            let range = {
                let _guard = self
                    .gate
                    .lock()
                    .map_err(|_| anyhow!("GGML lock poisoned"))?;
                let mut error = [0; 2048];
                let handle = unsafe {
                    (self.new_range)(
                        self.ptr(),
                        handles.as_ptr(),
                        count as i64,
                        error.as_mut_ptr(),
                        error.len(),
                    )
                } as usize;
                if handle == 0 {
                    return Err(ffi_error(&error).into());
                }
                Arc::new(GgmlRange {
                    engine: self.clone(),
                    handle,
                    _states: states,
                })
            };
            log::debug!(
                "GGML fused shard created: layers={}..{}",
                first,
                first + count - 1
            );
            cache.ggml_ranges.insert(key, range.clone());
            range
        };
        let op = Op {
            engine: self.clone(),
            kind: Kind::Range { range, n, position },
            shape: x.shape().clone(),
        };
        Ok(x.to_dtype(DType::F32)?
            .contiguous()?
            .apply_op1_no_bwd(&op)?
            .to_dtype(x.dtype())?)
    }
}
impl Drop for GgmlEngine {
    fn drop(&mut self) {
        if self.handle != 0 {
            unsafe { (self.close)(self.ptr()) };
        }
    }
}
#[derive(Debug)]
pub struct GgmlState {
    engine: Arc<GgmlEngine>,
    handle: usize,
}
impl Drop for GgmlState {
    fn drop(&mut self) {
        let _guard = self.engine.gate.lock().unwrap_or_else(|e| e.into_inner());
        unsafe { (self.engine.free_state)(self.handle as *mut c_void) };
    }
}
#[derive(Debug)]
pub struct GgmlRange {
    engine: Arc<GgmlEngine>,
    handle: usize,
    // Keep borrowed C++ states (and their engine) alive until range destruction.
    _states: Vec<Arc<GgmlState>>,
}
impl Drop for GgmlRange {
    fn drop(&mut self) {
        let _guard = self.engine.gate.lock().unwrap_or_else(|e| e.into_inner());
        unsafe { (self.engine.free_range)(self.handle as *mut c_void) };
    }
}
enum Kind {
    Embed(Vec<i32>),
    Head,
    Layer {
        state: Arc<GgmlState>,
        n: usize,
        position: usize,
    },
    Range {
        range: Arc<GgmlRange>,
        n: usize,
        position: usize,
    },
}
struct Op {
    engine: Arc<GgmlEngine>,
    kind: Kind,
    shape: Shape,
}
impl Op {
    // Pointers are valid contiguous F32 storage for this synchronous call.
    unsafe fn run(
        &self,
        input: *const f32,
        output: *mut f32,
        device_io: i32,
    ) -> candle_core::Result<()> {
        let e = &self.engine;
        let _guard = e
            .gate
            .lock()
            .map_err(|_| candle_core::Error::Msg("GGML lock poisoned".into()))?;
        let mut error = [0; 2048];
        let status = match &self.kind {
            Kind::Embed(ids) => (e.embed)(
                e.ptr(),
                ids.as_ptr(),
                ids.len() as i64,
                output,
                device_io,
                error.as_mut_ptr(),
                error.len(),
            ),
            Kind::Head => (e.head)(
                e.ptr(),
                input,
                output,
                device_io,
                error.as_mut_ptr(),
                error.len(),
            ),
            Kind::Layer { state, n, position } => (e.forward)(
                e.ptr(),
                state.handle as *mut c_void,
                input,
                *n as i64,
                *position as i64,
                output,
                device_io,
                error.as_mut_ptr(),
                error.len(),
            ),
            Kind::Range { range, n, position } => (e.forward_range)(
                e.ptr(),
                range.handle as *mut c_void,
                input,
                *n as i64,
                *position as i64,
                output,
                device_io,
                error.as_mut_ptr(),
                error.len(),
            ),
        };
        if status != 0 {
            return Err(ffi_error(&error));
        }
        Ok(())
    }
}
impl CustomOp1 for Op {
    fn name(&self) -> &'static str {
        "dial-qwen38-upstream-ggml"
    }
    fn cpu_fwd(
        &self,
        storage: &CpuStorage,
        layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        if !layout.is_contiguous() {
            candle_core::bail!("GGML input must be contiguous");
        }
        let values = storage.as_slice::<f32>()?;
        let input =
            &values[layout.start_offset()..layout.start_offset() + layout.shape().elem_count()];
        let mut out = vec![0f32; self.shape.elem_count()];
        unsafe {
            self.run(input.as_ptr(), out.as_mut_ptr(), 0)?;
        }
        Ok((CpuStorage::F32(out), self.shape.clone()))
    }
    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        storage: &candle_core::CudaStorage,
        layout: &Layout,
    ) -> candle_core::Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::{
            backend::BackendDevice,
            cuda::cudarc::driver::{DevicePtr, DevicePtrMut},
        };
        if !layout.is_contiguous() {
            candle_core::bail!("GGML CUDA input must be contiguous");
        }
        let values = storage.as_cuda_slice::<f32>()?;
        let input = values
            .slice(layout.start_offset()..layout.start_offset() + layout.shape().elem_count());
        let mut output = unsafe { storage.device.alloc::<f32>(self.shape.elem_count()) }
            .map_err(|e| candle_core::Error::Msg(format!("GGML output allocation failed: {e}")))?;
        let stream = storage.device.cuda_stream();
        let (input_ptr, read_guard) = input.device_ptr(&stream);
        let (output_ptr, write_guard) = output.device_ptr_mut(&stream);
        // Separate streams on the same CUDA primary context. The adapter
        // synchronizes GGML and D2D copies before returning to Candle.
        storage.device.synchronize()?;
        unsafe {
            self.run(input_ptr as *const f32, output_ptr as *mut f32, 1)?;
        }
        drop(write_guard);
        drop(read_guard);
        Ok((
            candle_core::CudaStorage::wrap_cuda_slice(output, storage.device.clone()),
            self.shape.clone(),
        ))
    }
}
