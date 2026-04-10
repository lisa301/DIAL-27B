use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::Result;
use candle_core::{DType, Device, Shape, Tensor};
use regex::Regex;
use rknpu2::{
    api::{runtime::RuntimeAPI, RknnInitFlags},
    bf16, f16,
    io::{
        buffer::{BufMutView, BufView},
        input::Input,
        output::{Output, OutputKind},
    },
    query::{InputAttr, InputOutputNum, OutputAttr, TensorAttrView},
    rknn::NpuCores,
    tensor::DataTypeKind,
    utils::find_rknn_library,
    RKNN,
};

use crate::models::llama3::{Cache, Config};

#[derive(Debug, Clone)]
struct LayerKv {
    past_len: usize,
    k: KvData,
    v: KvData,
}

#[derive(Debug, Clone, Copy)]
enum KvStorageDType {
    F32,
    F16,
    BF16,
}

impl KvStorageDType {
    fn from_rknn_dtype(dtype: &DataTypeKind) -> Result<Self> {
        match dtype {
            DataTypeKind::Float32(_) => Ok(Self::F32),
            DataTypeKind::Float16(_) => Ok(Self::F16),
            DataTypeKind::BFloat16(_) => Ok(Self::BF16),
            other => bail!("unsupported rknn kv dtype: {other:?}"),
        }
    }
}

#[derive(Debug, Clone)]
enum KvData {
    F32(Vec<f32>),
    F16(Vec<f16>),
    BF16(Vec<bf16>),
}

impl KvData {
    fn from_f32_slice(data: &[f32], dtype: KvStorageDType) -> Self {
        match dtype {
            KvStorageDType::F32 => Self::F32(data.to_vec()),
            KvStorageDType::F16 => Self::F16(data.iter().map(|v| f16::from_f32(*v)).collect()),
            KvStorageDType::BF16 => Self::BF16(data.iter().map(|v| bf16::from_f32(*v)).collect()),
        }
    }

    fn replace_from_f32_prefix(&mut self, data: &[f32], n: usize) -> Result<()> {
        if data.len() < n {
            bail!("replace_from_f32_prefix: data too short {} < {}", data.len(), n);
        }
        match self {
            Self::F32(buf) => {
                buf.clear();
                buf.extend_from_slice(&data[..n]);
            }
            Self::F16(buf) => {
                buf.clear();
                buf.reserve(n);
                for v in &data[..n] {
                    buf.push(f16::from_f32(*v));
                }
            }
            Self::BF16(buf) => {
                buf.clear();
                buf.reserve(n);
                for v in &data[..n] {
                    buf.push(bf16::from_f32(*v));
                }
            }
        }
        Ok(())
    }

    fn append_from_f32_prefix(&mut self, data: &[f32], n: usize) -> Result<()> {
        if data.len() < n {
            bail!("append_from_f32_prefix: data too short {} < {}", data.len(), n);
        }
        match self {
            Self::F32(buf) => buf.extend_from_slice(&data[..n]),
            Self::F16(buf) => {
                buf.reserve(n);
                for v in &data[..n] {
                    buf.push(f16::from_f32(*v));
                }
            }
            Self::BF16(buf) => {
                buf.reserve(n);
                for v in &data[..n] {
                    buf.push(bf16::from_f32(*v));
                }
            }
        }
        Ok(())
    }
}

struct ChunkModel {
    layer_start: usize,
    layer_end: usize,
    rknn: RKNN<RuntimeAPI>,
    input_attrs: Vec<InputAttr>,
    output_attrs: Vec<OutputAttr>,
}

impl ChunkModel {
    fn num_layers(&self) -> usize {
        self.layer_end - self.layer_start + 1
    }

    fn load(path: &Path, layer_start: usize, layer_end: usize, lib_path: &Path) -> Result<Self> {
        let mut model_data = std::fs::read(path)
            .map_err(|e| anyhow!("failed to read rknn model {}: {e}", path.display()))?;

        let rknn = RKNN::new_with_library(lib_path.to_path_buf(), &mut model_data, RknnInitFlags::builder())
            .map_err(|e| anyhow!("rknn init failed for {}: {e}", path.display()))?;

        rknn.set_core_mask(NpuCores::ALL)
            .map_err(|e| anyhow!("rknn set_core_mask failed for {}: {e}", path.display()))?;

        let io_num = rknn
            .query::<InputOutputNum>()
            .map_err(|e| anyhow!("rknn query io_num failed for {}: {e}", path.display()))?;

        let mut input_attrs = Vec::with_capacity(io_num.input_num() as usize);
        for i in 0..io_num.input_num() {
            input_attrs.push(
                rknn.query_with_input::<InputAttr>(i)
                    .map_err(|e| anyhow!("rknn query input attr[{i}] failed for {}: {e}", path.display()))?,
            );
        }

        let mut output_attrs = Vec::with_capacity(io_num.output_num() as usize);
        for i in 0..io_num.output_num() {
            output_attrs.push(
                rknn.query_with_input::<OutputAttr>(i)
                    .map_err(|e| anyhow!("rknn query output attr[{i}] failed for {}: {e}", path.display()))?,
            );
        }

        let n_layers = layer_end - layer_start + 1;
        let expected_inputs = 3 + 2 * n_layers;
        let expected_outputs = 1 + 2 * n_layers;
        if input_attrs.len() != expected_inputs {
            bail!(
                "{} input_num mismatch: expected {}, got {}",
                path.display(),
                expected_inputs,
                input_attrs.len()
            );
        }
        if output_attrs.len() != expected_outputs {
            bail!(
                "{} output_num mismatch: expected {}, got {}",
                path.display(),
                expected_outputs,
                output_attrs.len()
            );
        }

        Ok(Self {
            layer_start,
            layer_end,
            rknn,
            input_attrs,
            output_attrs,
        })
    }
}

enum OwnedInputBuffer<'a> {
    BorrowF32(&'a [f32]),
    BorrowF16(&'a [f16]),
    BorrowBF16(&'a [bf16]),
    OwnF16(Vec<f16>),
    OwnBF16(Vec<bf16>),
}

impl<'a> OwnedInputBuffer<'a> {
    fn from_f32(data: &'a [f32], dtype: DataTypeKind) -> Result<Self> {
        let out = match dtype {
            DataTypeKind::Float32(_) => Self::BorrowF32(data),
            DataTypeKind::Float16(_) => {
                let converted: Vec<f16> = data.iter().map(|v| f16::from_f32(*v)).collect();
                Self::OwnF16(converted)
            }
            DataTypeKind::BFloat16(_) => {
                let converted: Vec<bf16> = data.iter().map(|v| bf16::from_f32(*v)).collect();
                Self::OwnBF16(converted)
            }
            other => bail!("unsupported rknn input dtype: {other:?}"),
        };
        Ok(out)
    }

    fn from_kv_data(data: &'a KvData) -> Self {
        match data {
            KvData::F32(v) => Self::BorrowF32(v),
            KvData::F16(v) => Self::BorrowF16(v),
            KvData::BF16(v) => Self::BorrowBF16(v),
        }
    }

    fn as_buf_view(&'a self) -> BufView<'a> {
        match self {
            Self::BorrowF32(v) => BufView::F32(v),
            Self::BorrowF16(v) => BufView::F16(v),
            Self::BorrowBF16(v) => BufView::BF16(v),
            Self::OwnF16(v) => BufView::F16(v),
            Self::OwnBF16(v) => BufView::BF16(v),
        }
    }
}

fn build_rotary(position: usize, head_dim: usize, rope_theta: f64) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let mut freqs = vec![0f32; half];
    for (i, f) in freqs.iter_mut().enumerate() {
        let theta = 2 * i;
        let inv = 1f64 / rope_theta.powf(theta as f64 / head_dim as f64);
        *f = position as f32 * inv as f32;
    }

    let mut cos = vec![0f32; head_dim];
    let mut sin = vec![0f32; head_dim];
    for i in 0..half {
        let c = freqs[i].cos();
        let s = freqs[i].sin();
        cos[i] = c;
        cos[i + half] = c;
        sin[i] = s;
        sin[i + half] = s;
    }
    (cos, sin)
}

fn parse_chunk_path(path: &Path, re: &Regex) -> Option<(usize, usize)> {
    let name = path.file_name()?.to_str()?;
    let caps = re.captures(name)?;
    let s = caps.get(1)?.as_str().parse::<usize>().ok()?;
    let e = caps.get(2)?.as_str().parse::<usize>().ok()?;
    Some((s, e))
}

fn tensor_to_f32_vec3(x: &Tensor, expected_hidden: usize) -> Result<Vec<f32>> {
    let x = x
        .to_device(&Device::Cpu)
        .map_err(|e| anyhow!("x.to_cpu failed: {e}"))?
        .to_dtype(DType::F32)
        .map_err(|e| anyhow!("x.to_f32 failed: {e}"))?
        .contiguous()
        .map_err(|e| anyhow!("x.contiguous failed: {e}"))?;
    let (b, s, h) = x.dims3().map_err(|e| anyhow!("x.dims3 failed: {e}"))?;
    if (b, s, h) != (1, 1, expected_hidden) {
        bail!("text rknn expects x shape (1,1,{expected_hidden}), got ({b},{s},{h})");
    }
    x.flatten_all()
        .map_err(|e| anyhow!("x.flatten failed: {e}"))?
        .to_vec1::<f32>()
        .map_err(|e| anyhow!("x.to_vec1 failed: {e}"))
}

/// Run Qwen3-VL text decode via RKNN chunk models (e.g. 2 layers/chunk).
pub struct TextRknnRunner {
    chunks: Vec<ChunkModel>,
    hidden_size: usize,
    head_dim: usize,
    num_key_value_heads: usize,
    num_hidden_layers: usize,
    rope_theta: f64,
    layer_k_storage_dtypes: Vec<KvStorageDType>,
    layer_v_storage_dtypes: Vec<KvStorageDType>,
    kv: Vec<Option<LayerKv>>,
    synced_from_cache: bool,
}

impl TextRknnRunner {
    pub fn load(model_dir: &Path, lib_path: Option<&Path>, cfg: &Config) -> Result<Self> {
        if cfg.hidden_size == 0 || cfg.num_hidden_layers == 0 || cfg.num_key_value_heads == 0 {
            bail!("invalid text config for text rknn runner");
        }
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        let lib_path = match lib_path {
            Some(path) => path.to_path_buf(),
            None => find_rknn_library()
                .next()
                .ok_or_else(|| anyhow!("cannot find librknnrt.so (set --text-rknn-lib)"))?,
        };

        let mut files: Vec<PathBuf> = fs::read_dir(model_dir)
            .map_err(|e| anyhow!("read_dir {} failed: {e}", model_dir.display()))?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("rknn"))
            .collect();
        files.sort();
        if files.is_empty() {
            bail!("no .rknn files found in {}", model_dir.display());
        }

        let re = Regex::new(r"l(\d+)_l(\d+)").map_err(|e| anyhow!("regex build failed: {e}"))?;
        let mut parsed = Vec::new();
        for path in files {
            if let Some((s, e)) = parse_chunk_path(&path, &re) {
                parsed.push((s, e, path));
            }
        }
        if parsed.is_empty() {
            bail!(
                "no chunk filename matched pattern lXX_lYY under {}",
                model_dir.display()
            );
        }
        parsed.sort_by_key(|(s, _, _)| *s);

        let mut expected = 0usize;
        for (s, e, path) in &parsed {
            if *s != expected {
                bail!(
                    "chunk coverage gap: expected layer {}, got {} ({})",
                    expected,
                    s,
                    path.display()
                );
            }
            if e < s {
                bail!("invalid chunk range {}..{} in {}", s, e, path.display());
            }
            expected = e + 1;
        }
        if expected != cfg.num_hidden_layers {
            bail!(
                "chunk coverage mismatch: got 0..{}, expected 0..{}",
                expected.saturating_sub(1),
                cfg.num_hidden_layers.saturating_sub(1)
            );
        }

        let mut chunks = Vec::with_capacity(parsed.len());
        for (s, e, path) in parsed {
            let chunk = ChunkModel::load(&path, s, e, &lib_path)?;
            log::info!("text rknn chunk loaded: {} (layers {}-{})", path.display(), s, e);
            chunks.push(chunk);
        }

        let mut layer_k_storage_dtypes = vec![None; cfg.num_hidden_layers];
        let mut layer_v_storage_dtypes = vec![None; cfg.num_hidden_layers];
        for chunk in &chunks {
            for local_idx in 0..chunk.num_layers() {
                let layer = chunk.layer_start + local_idx;
                let k_attr_idx = 3 + 2 * local_idx;
                let v_attr_idx = 3 + 2 * local_idx + 1;
                let k_dtype = KvStorageDType::from_rknn_dtype(&chunk.input_attrs[k_attr_idx].dtype())?;
                let v_dtype = KvStorageDType::from_rknn_dtype(&chunk.input_attrs[v_attr_idx].dtype())?;
                layer_k_storage_dtypes[layer] = Some(k_dtype);
                layer_v_storage_dtypes[layer] = Some(v_dtype);
            }
        }
        let layer_k_storage_dtypes: Vec<KvStorageDType> = layer_k_storage_dtypes
            .into_iter()
            .enumerate()
            .map(|(layer, v)| v.ok_or_else(|| anyhow!("missing k dtype mapping for layer {layer}")))
            .collect::<Result<Vec<_>>>()?;
        let layer_v_storage_dtypes: Vec<KvStorageDType> = layer_v_storage_dtypes
            .into_iter()
            .enumerate()
            .map(|(layer, v)| v.ok_or_else(|| anyhow!("missing v dtype mapping for layer {layer}")))
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            chunks,
            hidden_size: cfg.hidden_size,
            head_dim,
            num_key_value_heads: cfg.num_key_value_heads,
            num_hidden_layers: cfg.num_hidden_layers,
            rope_theta: cfg.rope_theta as f64,
            layer_k_storage_dtypes,
            layer_v_storage_dtypes,
            kv: vec![None; cfg.num_hidden_layers],
            synced_from_cache: false,
        })
    }

    pub fn clear(&mut self) {
        self.kv = vec![None; self.num_hidden_layers];
        self.synced_from_cache = false;
    }

    fn sync_from_cache(&mut self, cache: &Cache) -> Result<()> {
        for layer_idx in 0..self.num_hidden_layers {
            let (k, v) = cache
                .kv_clone(layer_idx)
                .ok_or_else(|| anyhow!("cache missing kv for layer {layer_idx}; run prefill first"))?;

            let k = k
                .to_device(&Device::Cpu)
                .map_err(|e| anyhow!("layer {layer_idx} k.to_cpu failed: {e}"))?
                .to_dtype(DType::F32)
                .map_err(|e| anyhow!("layer {layer_idx} k.to_f32 failed: {e}"))?
                .contiguous()
                .map_err(|e| anyhow!("layer {layer_idx} k.contiguous failed: {e}"))?;
            let v = v
                .to_device(&Device::Cpu)
                .map_err(|e| anyhow!("layer {layer_idx} v.to_cpu failed: {e}"))?
                .to_dtype(DType::F32)
                .map_err(|e| anyhow!("layer {layer_idx} v.to_f32 failed: {e}"))?
                .contiguous()
                .map_err(|e| anyhow!("layer {layer_idx} v.contiguous failed: {e}"))?;

            let (b, h, seq, d) = k.dims4().map_err(|e| anyhow!("layer {layer_idx} k.dims4 failed: {e}"))?;
            if b != 1 || h != self.num_key_value_heads || d != self.head_dim {
                bail!(
                    "layer {layer_idx} k shape mismatch, expected (1,{},{},{}), got ({},{},{},{})",
                    self.num_key_value_heads,
                    "past_len",
                    self.head_dim,
                    b,
                    h,
                    seq,
                    d
                );
            }
            let (vb, vh, vseq, vd) =
                v.dims4().map_err(|e| anyhow!("layer {layer_idx} v.dims4 failed: {e}"))?;
            if (vb, vh, vseq, vd) != (b, h, seq, d) {
                bail!(
                    "layer {layer_idx} kv shape mismatch, k=({},{},{},{}) v=({},{},{},{})",
                    b,
                    h,
                    seq,
                    d,
                    vb,
                    vh,
                    vseq,
                    vd
                );
            }

            let k_vec = k
                .flatten_all()
                .map_err(|e| anyhow!("layer {layer_idx} k.flatten failed: {e}"))?
                .to_vec1::<f32>()
                .map_err(|e| anyhow!("layer {layer_idx} k.to_vec failed: {e}"))?;
            let v_vec = v
                .flatten_all()
                .map_err(|e| anyhow!("layer {layer_idx} v.flatten failed: {e}"))?
                .to_vec1::<f32>()
                .map_err(|e| anyhow!("layer {layer_idx} v.to_vec failed: {e}"))?;

            self.kv[layer_idx] = Some(LayerKv {
                past_len: seq,
                k: KvData::from_f32_slice(&k_vec, self.layer_k_storage_dtypes[layer_idx]),
                v: KvData::from_f32_slice(&v_vec, self.layer_v_storage_dtypes[layer_idx]),
            });
        }
        self.synced_from_cache = true;
        Ok(())
    }

    fn run_chunk(
        chunk: &ChunkModel,
        hidden: &[f32],
        cos: &[f32],
        sin: &[f32],
        kv: &mut [Option<LayerKv>],
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<Vec<f32>> {
        let mut input_buffers = Vec::with_capacity(chunk.input_attrs.len());
        input_buffers.push(OwnedInputBuffer::from_f32(hidden, chunk.input_attrs[0].dtype())?);
        input_buffers.push(OwnedInputBuffer::from_f32(cos, chunk.input_attrs[1].dtype())?);
        input_buffers.push(OwnedInputBuffer::from_f32(sin, chunk.input_attrs[2].dtype())?);

        for layer in chunk.layer_start..=chunk.layer_end {
            let layer_kv = kv[layer]
                .as_ref()
                .ok_or_else(|| anyhow!("missing kv for layer {} before chunk run", layer))?;
            input_buffers.push(OwnedInputBuffer::from_kv_data(&layer_kv.k));
            input_buffers.push(OwnedInputBuffer::from_kv_data(&layer_kv.v));
        }

        let mut inputs = Vec::with_capacity(input_buffers.len());
        for (idx, buf) in input_buffers.iter().enumerate() {
            inputs.push(Input::new(
                idx as u32,
                buf.as_buf_view(),
                true,
                chunk.input_attrs[idx].format(),
            ));
        }

        chunk
            .rknn
            .set_inputs(inputs)
            .map_err(|e| anyhow!("text chunk {}-{} set_inputs failed: {e}", chunk.layer_start, chunk.layer_end))?;
        chunk
            .rknn
            .run()
            .map_err(|e| anyhow!("text chunk {}-{} run failed: {e}", chunk.layer_start, chunk.layer_end))?;
        // Release immutable borrows of `kv` before mutating kv cache with present outputs.
        drop(input_buffers);

        let mut output_bufs: Vec<Vec<f32>> = chunk
            .output_attrs
            .iter()
            .enumerate()
            .map(|(i, attr)| {
                let n = attr.num_elements() as usize;
                if n == 0 {
                    bail!(
                        "text chunk {}-{} output[{}] has n_elems=0 (dynamic outputs must expose n_elems via RKNN metadata)",
                        chunk.layer_start,
                        chunk.layer_end,
                        i
                    );
                }
                Ok(vec![0f32; n])
            })
            .collect::<Result<Vec<_>>>()?;

        let mut outputs = Vec::with_capacity(output_bufs.len());
        for (i, out) in output_bufs.iter_mut().enumerate() {
            outputs.push(Output {
                index: i as u32,
                kind: OutputKind::Preallocated {
                    buf: BufMutView::F32(out),
                    want_float: true,
                },
            });
        }

        chunk.rknn.get_outputs(&mut outputs).map_err(|e| {
            anyhow!(
                "text chunk {}-{} get_outputs failed: {e}",
                chunk.layer_start,
                chunk.layer_end
            )
        })?;

        let hidden_out = output_bufs
            .first()
            .ok_or_else(|| anyhow!("text chunk {}-{} returned no outputs", chunk.layer_start, chunk.layer_end))?;
        if hidden_out.len() < hidden.len() {
            bail!(
                "text chunk {}-{} hidden_out too short: got {}, expected at least {}",
                chunk.layer_start,
                chunk.layer_end,
                hidden_out.len(),
                hidden.len()
            );
        }

        for local_idx in 0..chunk.num_layers() {
            let layer = chunk.layer_start + local_idx;
            let prev = kv[layer]
                .as_mut()
                .ok_or_else(|| anyhow!("missing kv for layer {} when updating outputs", layer))?;
            let present_len = prev.past_len + 1;
            let expected_full_elems = num_kv_heads * present_len * head_dim;
            let expected_delta_elems = num_kv_heads * head_dim;

            let k_idx = 1 + 2 * local_idx;
            let v_idx = 1 + 2 * local_idx + 1;
            let present_k = output_bufs.get(k_idx).ok_or_else(|| {
                anyhow!(
                    "missing present_k output for layer {} from chunk {}-{}",
                    layer,
                    chunk.layer_start,
                    chunk.layer_end
                )
            })?;
            let present_v = output_bufs.get(v_idx).ok_or_else(|| {
                anyhow!(
                    "missing present_v output for layer {} from chunk {}-{}",
                    layer,
                    chunk.layer_start,
                    chunk.layer_end
                )
            })?;

            // Preferred path: model outputs delta kv (seq=1) and we append locally.
            if present_k.len() >= expected_delta_elems
                && present_v.len() >= expected_delta_elems
                && (present_k.len() < expected_full_elems || present_v.len() < expected_full_elems)
            {
                prev.k
                    .append_from_f32_prefix(present_k, expected_delta_elems)
                    .map_err(|e| anyhow!("layer {} append delta k failed: {e}", layer))?;
                prev.v
                    .append_from_f32_prefix(present_v, expected_delta_elems)
                    .map_err(|e| anyhow!("layer {} append delta v failed: {e}", layer))?;
                prev.past_len = present_len;
                continue;
            }

            // Legacy path: model outputs full present kv (past+1) every token.
            if present_k.len() >= expected_full_elems && present_v.len() >= expected_full_elems {
                prev.k
                    .replace_from_f32_prefix(present_k, expected_full_elems)
                    .map_err(|e| anyhow!("layer {} replace full k failed: {e}", layer))?;
                prev.v
                    .replace_from_f32_prefix(present_v, expected_full_elems)
                    .map_err(|e| anyhow!("layer {} replace full v failed: {e}", layer))?;
                prev.past_len = present_len;
                continue;
            }

            {
                bail!(
                    "layer {} present kv too short: k={} v={} expected_full={} expected_delta={}",
                    layer,
                    present_k.len(),
                    present_v.len(),
                    expected_full_elems,
                    expected_delta_elems
                );
            }
        }

        Ok(hidden_out[..hidden.len()].to_vec())
    }

    pub fn forward_decode(
        &mut self,
        x: &Tensor,
        index_pos: usize,
        cache: &Cache,
        out_device: &Device,
        out_dtype: DType,
    ) -> Result<Tensor> {
        if !self.synced_from_cache {
            self.sync_from_cache(cache)?;
        }
        let mut hidden = tensor_to_f32_vec3(x, self.hidden_size)?;
        let (cos, sin) = build_rotary(index_pos, self.head_dim, self.rope_theta);

        for chunk in &self.chunks {
            hidden = Self::run_chunk(
                chunk,
                &hidden,
                &cos,
                &sin,
                &mut self.kv,
                self.num_key_value_heads,
                self.head_dim,
            )?;
        }

        Tensor::from_vec(hidden, Shape::from_dims(&[1, 1, self.hidden_size]), &Device::Cpu)
            .map_err(|e| anyhow!("hidden tensor build failed: {e}"))?
            .to_dtype(out_dtype)
            .map_err(|e| anyhow!("hidden to_dtype failed: {e}"))?
            .to_device(out_device)
            .map_err(|e| anyhow!("hidden to_device failed: {e}"))
    }
}
