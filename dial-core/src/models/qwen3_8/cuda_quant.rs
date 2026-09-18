#![cfg(feature = "cuda")]

use std::sync::{Arc, OnceLock};

use candle_core::{DType, Device, Tensor};
use half::f16;

const CUDA_SOURCE: &str = r#"
#include <cuda_fp16.h>

__device__ __forceinline__ float dial_e2m1_to_f32(unsigned char code) {
    const unsigned char magnitude = code & 7;
    float value;
    switch (magnitude) {
        case 0: value = 0.0f; break;
        case 1: value = 0.5f; break;
        case 2: value = 1.0f; break;
        case 3: value = 1.5f; break;
        case 4: value = 2.0f; break;
        case 5: value = 3.0f; break;
        case 6: value = 4.0f; break;
        default: value = 6.0f; break;
    }
    return (code & 8) ? -value : value;
}

__device__ __forceinline__ float dial_e4m3_to_f32(unsigned char code) {
    const unsigned int sign = code >> 7;
    const unsigned int exponent = (code >> 3) & 15;
    const unsigned int mantissa = code & 7;
    float value;
    if (exponent == 0) {
        value = (float)mantissa * 0.001953125f; // mantissa * 2^-9
    } else if (exponent == 15 && mantissa == 7) {
        value = 0.0f; // invalid/NaN scale or weight; checkpoints must not emit it
    } else {
        const unsigned int exponent_bits = (exponent + 120) << 23; // exp - 7 + 127
        value = __int_as_float((int)exponent_bits) * (1.0f + (float)mantissa * 0.125f);
    }
    return sign ? -value : value;
}

__device__ __forceinline__ float dial_nvfp4_weight(
    const unsigned char* packed,
    const unsigned char* scale,
    float global_scale,
    int out_idx,
    int k,
    int in_dim) {
    const long long packed_row = (long long)out_idx * (in_dim >> 1);
    const unsigned char byte = packed[packed_row + (k >> 1)];
    const unsigned char code = (k & 1) ? (byte >> 4) : (byte & 15);
    const long long scale_row = (long long)out_idx * (in_dim >> 4);
    const float block_scale = dial_e4m3_to_f32(scale[scale_row + (k >> 4)]);
    return dial_e2m1_to_f32(code) * block_scale * global_scale;
}

__device__ __forceinline__ float dial_fp8_weight(
    const unsigned char* weight,
    const float* scale,
    int out_idx,
    int k,
    int in_dim) {
    return dial_e4m3_to_f32(weight[(long long)out_idx * in_dim + k]) * scale[out_idx];
}

extern "C" __global__ void dial_nvfp4_decode_f16(
    const half* x,
    const unsigned char* packed,
    const unsigned char* scale,
    float global_scale,
    half* out,
    int rows,
    int in_dim,
    int out_dim) {
    const int out_idx = (int)blockIdx.x;
    const int row = (int)blockIdx.y;
    if (out_idx >= out_dim || row >= rows) return;
    const half* input = x + (long long)row * in_dim;
    float sum = 0.0f;
    for (int k = (int)threadIdx.x; k < in_dim; k += (int)blockDim.x) {
        sum += __half2float(input[k]) * dial_nvfp4_weight(
            packed, scale, global_scale, out_idx, k, in_dim);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    __shared__ float warp_sums[8];
    const int lane = (int)threadIdx.x & 31;
    const int warp = (int)threadIdx.x >> 5;
    if (lane == 0) warp_sums[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        sum = lane < ((int)blockDim.x >> 5) ? warp_sums[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (lane == 0) out[(long long)row * out_dim + out_idx] = __float2half_rn(sum);
    }
}

extern "C" __global__ void dial_fp8_decode_f16(
    const half* x,
    const unsigned char* weight,
    const float* scale,
    half* out,
    int rows,
    int in_dim,
    int out_dim) {
    const int out_idx = (int)blockIdx.x;
    const int row = (int)blockIdx.y;
    if (out_idx >= out_dim || row >= rows) return;
    const half* input = x + (long long)row * in_dim;
    float sum = 0.0f;
    for (int k = (int)threadIdx.x; k < in_dim; k += (int)blockDim.x) {
        sum += __half2float(input[k]) * dial_fp8_weight(weight, scale, out_idx, k, in_dim);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    __shared__ float warp_sums[8];
    const int lane = (int)threadIdx.x & 31;
    const int warp = (int)threadIdx.x >> 5;
    if (lane == 0) warp_sums[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        sum = lane < ((int)blockDim.x >> 5) ? warp_sums[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (lane == 0) out[(long long)row * out_dim + out_idx] = __float2half_rn(sum);
    }
}

extern "C" __global__ void dial_nvfp4_prefill_f16(
    const half* x,
    const unsigned char* packed,
    const unsigned char* scale,
    float global_scale,
    half* out,
    int rows,
    int in_dim,
    int out_dim) {
    const int tx = (int)threadIdx.x;
    const int ty = (int)threadIdx.y;
    const int out_idx = (int)blockIdx.x * 16 + tx;
    const int row = (int)blockIdx.y * 16 + ty;
    __shared__ half x_tile[16][16];
    __shared__ float w_tile[16][16];
    float sum = 0.0f;
    for (int k0 = 0; k0 < in_dim; k0 += 16) {
        const int k = k0 + tx;
        x_tile[ty][tx] = (row < rows && k < in_dim)
            ? x[(long long)row * in_dim + k]
            : __float2half(0.0f);
        const int load_out = (int)blockIdx.x * 16 + ty;
        w_tile[ty][tx] = (load_out < out_dim && k < in_dim)
            ? dial_nvfp4_weight(packed, scale, global_scale, load_out, k, in_dim)
            : 0.0f;
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < 16; ++kk) {
            sum += __half2float(x_tile[ty][kk]) * w_tile[tx][kk];
        }
        __syncthreads();
    }
    if (row < rows && out_idx < out_dim) {
        out[(long long)row * out_dim + out_idx] = __float2half_rn(sum);
    }
}

extern "C" __global__ void dial_fp8_prefill_f16(
    const half* x,
    const unsigned char* weight,
    const float* scale,
    half* out,
    int rows,
    int in_dim,
    int out_dim) {
    const int tx = (int)threadIdx.x;
    const int ty = (int)threadIdx.y;
    const int out_idx = (int)blockIdx.x * 16 + tx;
    const int row = (int)blockIdx.y * 16 + ty;
    __shared__ half x_tile[16][16];
    __shared__ float w_tile[16][16];
    float sum = 0.0f;
    for (int k0 = 0; k0 < in_dim; k0 += 16) {
        const int k = k0 + tx;
        x_tile[ty][tx] = (row < rows && k < in_dim)
            ? x[(long long)row * in_dim + k]
            : __float2half(0.0f);
        const int load_out = (int)blockIdx.x * 16 + ty;
        w_tile[ty][tx] = (load_out < out_dim && k < in_dim)
            ? dial_fp8_weight(weight, scale, load_out, k, in_dim)
            : 0.0f;
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < 16; ++kk) {
            sum += __half2float(x_tile[ty][kk]) * w_tile[tx][kk];
        }
        __syncthreads();
    }
    if (row < rows && out_idx < out_dim) {
        out[(long long)row * out_dim + out_idx] = __float2half_rn(sum);
    }
}
"#;

const MODULE: &str = "dial_qwen38_quant_linear";
const NVFP4_DECODE: &str = "dial_nvfp4_decode_f16";
const NVFP4_PREFILL: &str = "dial_nvfp4_prefill_f16";
const FP8_DECODE: &str = "dial_fp8_decode_f16";
const FP8_PREFILL: &str = "dial_fp8_prefill_f16";
static PTX: OnceLock<String> = OnceLock::new();

fn ensure_kernels(device: &candle_core::cuda::CudaDevice) -> candle_core::Result<()> {
    use candle_core::cuda::cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

    if let Some(ptx) = PTX.get() {
        for name in [NVFP4_DECODE, NVFP4_PREFILL, FP8_DECODE, FP8_PREFILL] {
            device.get_or_load_custom_func(name, MODULE, ptx)?;
        }
        return Ok(());
    }
    static COMPILE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _guard = COMPILE_LOCK
        .lock()
        .map_err(|_| candle_core::Error::Msg("Qwen3.8 QuantLinear compile lock poisoned".into()))?;
    if PTX.get().is_none() {
        let mut include_paths = Vec::new();
        for root in [
            std::env::var("CUDA_HOME").ok(),
            std::env::var("CUDA_PATH").ok(),
            Some("/usr/local/cuda".to_string()),
            Some("/opt/cuda".to_string()),
        ]
        .into_iter()
        .flatten()
        {
            let include = std::path::Path::new(&root).join("include");
            if include.join("cuda_fp16.h").is_file() {
                let include = include.to_string_lossy().into_owned();
                if !include_paths.contains(&include) {
                    include_paths.push(include);
                }
            }
        }
        let options = CompileOptions {
            use_fast_math: Some(true),
            include_paths,
            ..Default::default()
        };
        let ptx = compile_ptx_with_opts(CUDA_SOURCE, options)
            .map_err(|error| {
                candle_core::Error::Msg(format!("Qwen3.8 QuantLinear NVRTC failed: {error}"))
            })?
            .to_src();
        let _ = PTX.set(ptx);
    }
    let ptx = PTX
        .get()
        .ok_or_else(|| candle_core::Error::Msg("Qwen3.8 QuantLinear PTX is empty".into()))?;
    for name in [NVFP4_DECODE, NVFP4_PREFILL, FP8_DECODE, FP8_PREFILL] {
        device.get_or_load_custom_func(name, MODULE, ptx)?;
    }
    Ok(())
}

#[derive(Debug)]
enum QuantWeight {
    Nvfp4 {
        packed: candle_core::cuda::cudarc::driver::CudaSlice<u8>,
        scales: candle_core::cuda::cudarc::driver::CudaSlice<u8>,
        global_scale: f32,
    },
    Fp8 {
        weight: candle_core::cuda::cudarc::driver::CudaSlice<u8>,
        scales: candle_core::cuda::cudarc::driver::CudaSlice<f32>,
    },
}

#[derive(Debug)]
pub struct CudaQuantLinear {
    weight: QuantWeight,
    in_dim: usize,
    out_dim: usize,
}

impl CudaQuantLinear {
    pub fn nvfp4(
        packed: Vec<u8>,
        scales: Vec<u8>,
        global_scale: f32,
        in_dim: usize,
        out_dim: usize,
        device: &Device,
    ) -> candle_core::Result<Arc<Self>> {
        let Device::Cuda(cuda) = device else {
            candle_core::bail!("NVFP4 QuantLinear requires CUDA")
        };
        if in_dim % 16 != 0 {
            candle_core::bail!("NVFP4 QuantLinear input dimension {in_dim} is not divisible by 16")
        }
        if packed.len() != out_dim * in_dim / 2 {
            candle_core::bail!("invalid NVFP4 packed weight length {}", packed.len())
        }
        if scales.len() != out_dim * in_dim / 16 {
            candle_core::bail!("invalid NVFP4 scale length {}", scales.len())
        }
        ensure_kernels(cuda)?;
        let stream = cuda.cuda_stream();
        let packed = stream.clone_htod(&packed).map_err(|error| {
            candle_core::Error::Msg(format!("uploading NVFP4 weight failed: {error}"))
        })?;
        let scales = stream.clone_htod(&scales).map_err(|error| {
            candle_core::Error::Msg(format!("uploading NVFP4 scales failed: {error}"))
        })?;
        Ok(Arc::new(Self {
            weight: QuantWeight::Nvfp4 {
                packed,
                scales,
                global_scale,
            },
            in_dim,
            out_dim,
        }))
    }

    pub fn fp8(
        weight: Vec<u8>,
        scales: Vec<f32>,
        in_dim: usize,
        out_dim: usize,
        device: &Device,
    ) -> candle_core::Result<Arc<Self>> {
        let Device::Cuda(cuda) = device else {
            candle_core::bail!("FP8 QuantLinear requires CUDA")
        };
        if weight.len() != out_dim * in_dim {
            candle_core::bail!("invalid FP8 weight length {}", weight.len())
        }
        if scales.len() != out_dim {
            candle_core::bail!("invalid FP8 scale length {}", scales.len())
        }
        ensure_kernels(cuda)?;
        let stream = cuda.cuda_stream();
        let weight = stream.clone_htod(&weight).map_err(|error| {
            candle_core::Error::Msg(format!("uploading FP8 weight failed: {error}"))
        })?;
        let scales = stream.clone_htod(&scales).map_err(|error| {
            candle_core::Error::Msg(format!("uploading FP8 scales failed: {error}"))
        })?;
        Ok(Arc::new(Self {
            weight: QuantWeight::Fp8 { weight, scales },
            in_dim,
            out_dim,
        }))
    }

    pub fn resident_bytes(&self) -> usize {
        match &self.weight {
            QuantWeight::Nvfp4 { .. } => {
                self.out_dim * self.in_dim / 2
                    + self.out_dim * self.in_dim / 16
                    + std::mem::size_of::<f32>()
            }
            QuantWeight::Fp8 { .. } => {
                self.out_dim * self.in_dim + self.out_dim * std::mem::size_of::<f32>()
            }
        }
    }

    pub fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        if x.dtype() != DType::F16 {
            candle_core::bail!("Qwen3.8 QuantLinear expects F16 input, got {:?}", x.dtype())
        }
        if x.dims().last().copied() != Some(self.in_dim) {
            candle_core::bail!(
                "Qwen3.8 QuantLinear expected last dim {}, got {:?}",
                self.in_dim,
                x.dims()
            )
        }
        let x = if x.is_contiguous() {
            x.clone()
        } else {
            x.contiguous()?
        };
        x.apply_op1_no_bwd(self)
    }
}

impl candle_core::CustomOp1 for CudaQuantLinear {
    fn name(&self) -> &'static str {
        "dial-qwen38-quant-linear"
    }

    fn cpu_fwd(
        &self,
        _storage: &candle_core::CpuStorage,
        _layout: &candle_core::Layout,
    ) -> candle_core::Result<(candle_core::CpuStorage, candle_core::Shape)> {
        candle_core::bail!("Qwen3.8 QuantLinear is only available on CUDA")
    }

    fn cuda_fwd(
        &self,
        storage: &candle_core::CudaStorage,
        layout: &candle_core::Layout,
    ) -> candle_core::Result<(candle_core::CudaStorage, candle_core::Shape)> {
        use candle_core::cuda::cudarc::driver::{LaunchConfig, PushKernelArg};

        if !layout.is_contiguous() {
            candle_core::bail!("Qwen3.8 QuantLinear input must be contiguous")
        }
        let elem_count = layout.shape().elem_count();
        if elem_count % self.in_dim != 0 {
            candle_core::bail!("Qwen3.8 QuantLinear input shape is invalid")
        }
        let rows = elem_count / self.in_dim;
        let input_all = storage.as_cuda_slice::<f16>()?;
        let input = input_all.slice(layout.start_offset()..layout.start_offset() + elem_count);
        let mut output =
            unsafe { storage.device.alloc::<f16>(rows * self.out_dim) }.map_err(|error| {
                candle_core::Error::Msg(format!("allocating QuantLinear output: {error}"))
            })?;
        let is_decode = rows <= 4;
        let config = if is_decode {
            LaunchConfig {
                grid_dim: (self.out_dim as u32, rows as u32, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            }
        } else {
            LaunchConfig {
                grid_dim: (
                    self.out_dim.div_ceil(16) as u32,
                    rows.div_ceil(16) as u32,
                    1,
                ),
                block_dim: (16, 16, 1),
                shared_mem_bytes: 0,
            }
        };
        let ptx = PTX
            .get()
            .ok_or_else(|| candle_core::Error::Msg("QuantLinear PTX is not initialized".into()))?;
        let (kernel, is_nvfp4) = match &self.weight {
            QuantWeight::Nvfp4 { .. } => (
                if is_decode {
                    NVFP4_DECODE
                } else {
                    NVFP4_PREFILL
                },
                true,
            ),
            QuantWeight::Fp8 { .. } => (if is_decode { FP8_DECODE } else { FP8_PREFILL }, false),
        };
        let function = storage
            .device
            .get_or_load_custom_func(kernel, MODULE, ptx)?;
        let mut builder = function.builder();
        builder.arg(&input);
        match &self.weight {
            QuantWeight::Nvfp4 {
                packed,
                scales,
                global_scale,
            } => {
                builder.arg(packed);
                builder.arg(scales);
                builder.arg(global_scale);
            }
            QuantWeight::Fp8 { weight, scales } => {
                builder.arg(weight);
                builder.arg(scales);
            }
        }
        builder.arg(&mut output);
        let rows = rows as i32;
        let in_dim = self.in_dim as i32;
        let out_dim = self.out_dim as i32;
        builder.arg(&rows);
        builder.arg(&in_dim);
        builder.arg(&out_dim);
        unsafe { builder.launch(config) }.map_err(|error| {
            candle_core::Error::Msg(format!(
                "launching {} QuantLinear failed: {error}",
                if is_nvfp4 { "NVFP4" } else { "FP8" }
            ))
        })?;

        let mut dims = layout.shape().dims().to_vec();
        *dims.last_mut().expect("QuantLinear input rank checked") = self.out_dim;
        Ok((
            candle_core::CudaStorage::wrap_cuda_slice(output, storage.device.clone()),
            candle_core::Shape::from(dims),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::Module;

    #[test]
    #[ignore = "run on the target CUDA GPU"]
    fn quant_linears_match_dense_reference() -> candle_core::Result<()> {
        let device = Device::new_cuda(0)?;
        let in_dim = 32;
        let out_dim = 3;
        let scales = vec![0x38u8; out_dim * in_dim / 16]; // E4M3 1.0
        let packed = vec![0x22u8; out_dim * in_dim / 2]; // E2M1 1.0, 1.0
        let nvfp4 = CudaQuantLinear::nvfp4(packed, scales, 0.5, in_dim, out_dim, &device)?;
        let fp8 = CudaQuantLinear::fp8(
            vec![0x38u8; out_dim * in_dim],
            vec![0.5; out_dim],
            in_dim,
            out_dim,
            &device,
        )?;
        let dense = candle_nn::Linear::new(
            Tensor::from_vec(
                vec![f16::from_f32(0.5); out_dim * in_dim],
                (out_dim, in_dim),
                &device,
            )?,
            None,
        );
        for rows in [1usize, 7] {
            let x = Tensor::from_vec(
                (0..rows * in_dim)
                    .map(|index| f16::from_f32((index as f32 * 0.07).sin()))
                    .collect::<Vec<_>>(),
                (rows, in_dim),
                &device,
            )?;
            let expected = dense.forward(&x)?.to_dtype(DType::F32)?.flatten_all()?;
            for actual in [nvfp4.forward(&x)?, fp8.forward(&x)?] {
                let actual = actual.to_dtype(DType::F32)?.flatten_all()?;
                let max_abs = expected
                    .to_vec1::<f32>()?
                    .iter()
                    .zip(actual.to_vec1::<f32>()?)
                    .map(|(left, right)| (left - right).abs())
                    .fold(0.0f32, f32::max);
                assert!(max_abs < 0.02, "max_abs={max_abs}");
            }
        }
        Ok(())
    }
}
