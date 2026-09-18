use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};
#[cfg(feature = "cuda")]
use std::sync::Arc;

use candle_core::{DType, Device, Tensor};
use candle_nn::{Linear, Module, VarBuilder};
use half::{bf16, f16};
use rayon::prelude::*;

use crate::Qwen38QuantLinear;

#[cfg(feature = "cuda")]
use super::cuda_quant::CudaQuantLinear;

const NVFP4_GROUP_SIZE: usize = 16;
const E2M1_VALUES: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

const QUANT_DENSE: u8 = 0;
const QUANT_CUDA_AUTO: u8 = 1;
const QUANT_CUDA_REQUIRED: u8 = 2;
static QUANT_MODE: AtomicU8 = AtomicU8::new(QUANT_DENSE);
static QUANT_LINEAR_COUNT: AtomicUsize = AtomicUsize::new(0);
static QUANT_WEIGHT_BYTES: AtomicUsize = AtomicUsize::new(0);

pub fn configure_quant_linear(
    requested: Qwen38QuantLinear,
    device: &Device,
    dtype: DType,
) -> anyhow::Result<()> {
    let supported_runtime = device.is_cuda() && dtype == DType::F16;
    let mode = match requested {
        Qwen38QuantLinear::Dense => QUANT_DENSE,
        Qwen38QuantLinear::Auto if supported_runtime => QUANT_CUDA_AUTO,
        Qwen38QuantLinear::Auto => QUANT_DENSE,
        Qwen38QuantLinear::Cuda if !supported_runtime => anyhow::bail!(
            "--qwen38-quant-linear cuda requires a CUDA device and --dtype f16 (device={device:?}, dtype={dtype:?})"
        ),
        Qwen38QuantLinear::Cuda => QUANT_CUDA_REQUIRED,
    };
    #[cfg(not(feature = "cuda"))]
    if mode != QUANT_DENSE {
        anyhow::bail!("Qwen3.8 CUDA QuantLinear requires building with --features cuda");
    }
    QUANT_MODE.store(mode, Ordering::Relaxed);
    QUANT_LINEAR_COUNT.store(0, Ordering::Relaxed);
    QUANT_WEIGHT_BYTES.store(0, Ordering::Relaxed);
    match mode {
        QUANT_DENSE => log::info!(
            "Qwen3.8 QuantLinear: dense compatibility mode (quantized weights expand to {dtype:?})"
        ),
        QUANT_CUDA_AUTO => log::info!(
            "Qwen3.8 QuantLinear: CUDA auto W4A16/W8A16 mode (quantized weights remain device-resident)"
        ),
        QUANT_CUDA_REQUIRED => log::info!(
            "Qwen3.8 QuantLinear: CUDA W4A16/W8A16 required mode (startup fails instead of dense fallback)"
        ),
        _ => unreachable!(),
    }
    Ok(())
}

pub fn quant_linear_summary() -> Option<(usize, usize)> {
    (QUANT_MODE.load(Ordering::Relaxed) != QUANT_DENSE).then(|| {
        (
            QUANT_LINEAR_COUNT.load(Ordering::Relaxed),
            QUANT_WEIGHT_BYTES.load(Ordering::Relaxed),
        )
    })
}

#[derive(Debug, Clone)]
pub enum QuantLinear {
    Dense(Linear),
    #[cfg(feature = "cuda")]
    Cuda(Arc<CudaQuantLinear>),
}

#[cfg(feature = "cuda")]
#[doc(hidden)]
pub fn quant_linear_smoke_nvfp4(
    packed: Vec<u8>,
    scales: Vec<u8>,
    global_scale: f32,
    input_size: usize,
    output_size: usize,
    device: &Device,
) -> candle_core::Result<QuantLinear> {
    CudaQuantLinear::nvfp4(
        packed,
        scales,
        global_scale,
        input_size,
        output_size,
        device,
    )
    .map(QuantLinear::Cuda)
}

#[cfg(feature = "cuda")]
#[doc(hidden)]
pub fn quant_linear_smoke_fp8(
    weight: Vec<u8>,
    scales: Vec<f32>,
    input_size: usize,
    output_size: usize,
    device: &Device,
) -> candle_core::Result<QuantLinear> {
    CudaQuantLinear::fp8(weight, scales, input_size, output_size, device).map(QuantLinear::Cuda)
}

impl Module for QuantLinear {
    fn forward(&self, input: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Self::Dense(linear) => linear.forward(input),
            #[cfg(feature = "cuda")]
            Self::Cuda(linear) => linear.forward(input),
        }
    }
}

impl QuantLinear {
    pub fn is_quantized_cuda(&self) -> bool {
        match self {
            Self::Dense(_) => false,
            #[cfg(feature = "cuda")]
            Self::Cuda(_) => true,
        }
    }

    #[cfg(test)]
    fn dense_weight(&self) -> &Tensor {
        match self {
            Self::Dense(linear) => linear.weight(),
            #[cfg(feature = "cuda")]
            Self::Cuda(_) => panic!("test expected dense fallback"),
        }
    }
}

fn decode_nvfp4_values(
    packed: &[u8],
    scales: &[f32],
    global_scale: f32,
    input_size: usize,
    output_size: usize,
) -> candle_core::Result<Vec<f32>> {
    if input_size % NVFP4_GROUP_SIZE != 0 {
        candle_core::bail!(
            "NVFP4 input dimension {input_size} is not divisible by group size {NVFP4_GROUP_SIZE}"
        );
    }
    let expected_packed = output_size * input_size / 2;
    if packed.len() != expected_packed {
        candle_core::bail!(
            "NVFP4 packed weight has {} bytes, expected {expected_packed}",
            packed.len()
        );
    }
    let groups_per_row = input_size / NVFP4_GROUP_SIZE;
    let expected_scales = output_size * groups_per_row;
    if scales.len() != expected_scales {
        candle_core::bail!(
            "NVFP4 weight scale has {} values, expected {expected_scales}",
            scales.len()
        );
    }

    let mut decoded = vec![0.0f32; output_size * input_size];
    decoded
        .par_chunks_mut(input_size)
        .enumerate()
        .for_each(|(row, values)| {
            let packed_row = &packed[row * input_size / 2..(row + 1) * input_size / 2];
            let scale_row = &scales[row * groups_per_row..(row + 1) * groups_per_row];
            for (column, value) in values.iter_mut().enumerate() {
                let byte = packed_row[column / 2];
                let code = if column % 2 == 0 {
                    byte & 0x0f
                } else {
                    byte >> 4
                };
                *value = E2M1_VALUES[code as usize]
                    * scale_row[column / NVFP4_GROUP_SIZE]
                    * global_scale;
            }
        });
    Ok(decoded)
}

fn tensor_from_f32(
    values: Vec<f32>,
    shape: (usize, usize),
    dtype: DType,
    device: &Device,
) -> candle_core::Result<Tensor> {
    match dtype {
        DType::F16 => Tensor::from_vec(
            values
                .into_par_iter()
                .map(f16::from_f32)
                .collect::<Vec<_>>(),
            shape,
            device,
        ),
        DType::BF16 => Tensor::from_vec(
            values
                .into_par_iter()
                .map(bf16::from_f32)
                .collect::<Vec<_>>(),
            shape,
            device,
        ),
        DType::F32 => Tensor::from_vec(values, shape, device),
        other => candle_core::bail!("Qwen3.8 dequantization does not support {other:?}"),
    }
}

fn load_fp8_raw(
    vb: &VarBuilder,
    shape: impl Into<candle_core::Shape>,
    name: &str,
) -> candle_core::Result<Vec<u8>> {
    vb.clone()
        .set_device(Device::Cpu)
        .to_dtype(DType::F8E4M3)
        .get(shape, name)?
        .flatten_all()?
        .to_vec1::<float8::F8E4M3>()
        .map(|values| values.into_iter().map(|value| value.to_bits()).collect())
}

fn load_nvfp4(
    vb: &VarBuilder,
    input_size: usize,
    output_size: usize,
) -> candle_core::Result<Tensor> {
    let packed = vb
        .clone()
        .set_device(Device::Cpu)
        .to_dtype(DType::U8)
        .get((output_size, input_size / 2), "weight")?
        .flatten_all()?
        .to_vec1::<u8>()?;
    let scales = vb
        .clone()
        .set_device(Device::Cpu)
        .to_dtype(DType::F32)
        .get((output_size, input_size / NVFP4_GROUP_SIZE), "weight_scale")?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let global_scales = vb
        .clone()
        .set_device(Device::Cpu)
        .get_unchecked_dtype("weight_scale_2", DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    if global_scales.len() != 1 {
        candle_core::bail!(
            "NVFP4 weight_scale_2 for {} contains {} values, expected 1",
            vb.prefix(),
            global_scales.len()
        );
    }
    let decoded = decode_nvfp4_values(&packed, &scales, global_scales[0], input_size, output_size)?;
    tensor_from_f32(decoded, (output_size, input_size), vb.dtype(), vb.device())
}

fn load_compressed_tensors_nvfp4(
    vb: &VarBuilder,
    input_size: usize,
    output_size: usize,
) -> candle_core::Result<Tensor> {
    let cpu_vb = vb.clone().set_device(Device::Cpu);
    let packed = cpu_vb
        .to_dtype(DType::U8)
        .get((output_size, input_size / 2), "weight_packed")?
        .flatten_all()?
        .to_vec1::<u8>()?;
    let scales = cpu_vb
        .to_dtype(DType::F32)
        .get((output_size, input_size / NVFP4_GROUP_SIZE), "weight_scale")?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let global_divisors = cpu_vb
        .get_unchecked_dtype("weight_global_scale", DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    if global_divisors.len() != 1 {
        candle_core::bail!(
            "compressed-tensors weight_global_scale for {} contains {} values, expected 1",
            vb.prefix(),
            global_divisors.len()
        );
    }
    if global_divisors[0] == 0.0 {
        candle_core::bail!(
            "compressed-tensors weight_global_scale for {} is zero",
            vb.prefix()
        );
    }

    // compressed-tensors stores weight_global_scale as the quantization
    // divisor. Its dequantization multiplier is therefore the reciprocal.
    let decoded = decode_nvfp4_values(
        &packed,
        &scales,
        1.0 / global_divisors[0],
        input_size,
        output_size,
    )?;
    tensor_from_f32(decoded, (output_size, input_size), vb.dtype(), vb.device())
}

fn load_fp8(vb: &VarBuilder, input_size: usize, output_size: usize) -> candle_core::Result<Tensor> {
    // Candle 0.11 asks CUDA for `cast_f8e4m3_f32`, while its generated
    // kernel is exported as `cast_f8_e4m3_f32`. Decode on the host to avoid
    // that missing-symbol path, then copy the dense runtime weight to CUDA.
    let cpu_vb = vb.clone().set_device(Device::Cpu).to_dtype(DType::F32);
    let weight = cpu_vb.get((output_size, input_size), "weight")?;
    // `VarBuilder::get_unchecked()` uses the backend's original dtype rather
    // than the dtype selected by `to_dtype()`. Request f32 explicitly so a
    // runtime configured with `--dtype f16` does not produce F32 * F16 here.
    let scales = cpu_vb
        .get_unchecked_dtype("weight_scale", DType::F32)?
        .flatten_all()?;
    let scale = match scales.elem_count() {
        1 => scales.reshape(())?,
        n if n == output_size => {
            // ModelOpt exports per-output-channel FP8 scales for some
            // matrices (notably lm_head), flattened as [out].
            scales.reshape((output_size, 1))?
        }
        n => candle_core::bail!(
            "FP8 weight_scale for {} contains {n} values, expected 1 or {output_size}",
            vb.prefix()
        ),
    };
    weight
        .broadcast_mul(&scale)?
        .to_dtype(vb.dtype())
        .and_then(|weight| weight.to_device(vb.device()))
        .map(|weight| weight.detach())
}

/// Load an ordinary tensor on the host first. On some Thor CUDA stacks,
/// Candle's device-side BF16->F16 cast kernel cannot be resolved. Converting
/// on CPU and then copying the final tensor avoids that driver-specific path.
pub(crate) fn load_tensor<S: Into<candle_core::Shape>>(
    vb: VarBuilder,
    shape: S,
    name: &str,
) -> candle_core::Result<Tensor> {
    let target_device = vb.device().clone();
    vb.clone()
        .set_device(Device::Cpu)
        .get(shape, name)?
        .to_device(&target_device)
        .map(|tensor| tensor.detach())
}

#[cfg(feature = "cuda")]
fn try_cuda_nvfp4(
    input_size: usize,
    output_size: usize,
    vb: &VarBuilder,
) -> candle_core::Result<Arc<CudaQuantLinear>> {
    let cpu_vb = vb.clone().set_device(Device::Cpu);
    let (weight_name, global_name, invert_global) =
        if vb.contains_tensor("weight_packed") && vb.contains_tensor("weight_global_scale") {
            ("weight_packed", "weight_global_scale", true)
        } else if vb.contains_tensor("weight_scale_2") {
            ("weight", "weight_scale_2", false)
        } else {
            candle_core::bail!("{} is not an NVFP4 linear", vb.prefix())
        };
    let packed = cpu_vb
        .to_dtype(DType::U8)
        .get((output_size, input_size / 2), weight_name)?
        .flatten_all()?
        .to_vec1::<u8>()?;
    let scales = load_fp8_raw(
        &cpu_vb,
        (output_size, input_size / NVFP4_GROUP_SIZE),
        "weight_scale",
    )?;
    let global = cpu_vb
        .get_unchecked_dtype(global_name, DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    if global.len() != 1 || global[0] == 0.0 {
        candle_core::bail!(
            "{} {global_name} must contain one non-zero value, got {:?}",
            vb.prefix(),
            global
        )
    }
    let global = if invert_global {
        1.0 / global[0]
    } else {
        global[0]
    };
    CudaQuantLinear::nvfp4(packed, scales, global, input_size, output_size, vb.device())
}

#[cfg(feature = "cuda")]
fn try_cuda_fp8(
    input_size: usize,
    output_size: usize,
    vb: &VarBuilder,
) -> candle_core::Result<Arc<CudaQuantLinear>> {
    let weight = load_fp8_raw(vb, (output_size, input_size), "weight")?;
    let scales = vb
        .clone()
        .set_device(Device::Cpu)
        .get_unchecked_dtype("weight_scale", DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let scales = match scales.len() {
        1 => vec![scales[0]; output_size],
        n if n == output_size => scales,
        n => candle_core::bail!(
            "FP8 weight_scale for {} contains {n} values, expected 1 or {output_size}",
            vb.prefix()
        ),
    };
    CudaQuantLinear::fp8(weight, scales, input_size, output_size, vb.device())
}

#[cfg(feature = "cuda")]
fn register_cuda_quant_linear(name: &str, bytes: usize) {
    let count = QUANT_LINEAR_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
    let total = QUANT_WEIGHT_BYTES.fetch_add(bytes, Ordering::Relaxed) + bytes;
    if count == 1 || count % 32 == 0 {
        log::info!(
            "Qwen3.8 CUDA QuantLinear progress: last={name} linears={count} resident={:.1} MiB",
            total as f64 / 1048576.0
        );
    } else {
        log::debug!(
            "Qwen3.8 CUDA QuantLinear loaded: {name} resident={:.1} MiB",
            bytes as f64 / 1048576.0
        );
    }
}

pub fn linear_no_bias(
    input_size: usize,
    output_size: usize,
    vb: VarBuilder,
) -> candle_core::Result<QuantLinear> {
    let mode = QUANT_MODE.load(Ordering::Relaxed);
    let is_nvfp4 = (vb.contains_tensor("weight_packed")
        && vb.contains_tensor("weight_global_scale"))
        || vb.contains_tensor("weight_scale_2");
    let is_fp8 = !is_nvfp4 && vb.contains_tensor("weight_scale");
    if mode != QUANT_DENSE && (is_nvfp4 || is_fp8) {
        #[cfg(feature = "cuda")]
        {
            let loaded = if is_nvfp4 {
                try_cuda_nvfp4(input_size, output_size, &vb)
            } else {
                try_cuda_fp8(input_size, output_size, &vb)
            };
            match loaded {
                Ok(linear) => {
                    register_cuda_quant_linear(&vb.prefix(), linear.resident_bytes());
                    return Ok(QuantLinear::Cuda(linear));
                }
                Err(error) if mode == QUANT_CUDA_REQUIRED => return Err(error),
                Err(error) => {
                    QUANT_MODE.store(QUANT_DENSE, Ordering::Relaxed);
                    log::warn!(
                        "Qwen3.8 CUDA QuantLinear unavailable for {}: {error}; disabling it for this process and using dense fallback",
                        vb.prefix()
                    );
                }
            }
        }
        #[cfg(not(feature = "cuda"))]
        if mode == QUANT_CUDA_REQUIRED {
            candle_core::bail!("Qwen3.8 CUDA QuantLinear was required but cuda is not compiled")
        }
    }

    let weight = if vb.contains_tensor("weight_packed") && vb.contains_tensor("weight_global_scale")
    {
        log::debug!("loading {} as compressed-tensors NVFP4", vb.prefix());
        load_compressed_tensors_nvfp4(&vb, input_size, output_size)?
    } else if vb.contains_tensor("weight_scale_2") {
        log::debug!("loading {} as ModelOpt NVFP4", vb.prefix());
        load_nvfp4(&vb, input_size, output_size)?
    } else if vb.contains_tensor("weight_scale") {
        log::debug!("loading {} as ModelOpt FP8", vb.prefix());
        load_fp8(&vb, input_size, output_size)?
    } else {
        load_tensor(vb.clone(), (output_size, input_size), "weight")?
    };
    Ok(QuantLinear::Dense(Linear::new(weight, None)))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    #[test]
    fn decodes_low_nibble_before_high_nibble_and_applies_group_scales() {
        let mut packed = vec![0u8; 16];
        packed[0] = 0x92;
        packed[8] = 0x65;
        let decoded = decode_nvfp4_values(&packed, &[2.0, 0.5], 4.0, 32, 1).unwrap();
        assert_eq!(decoded[0], 8.0);
        assert_eq!(decoded[1], -4.0);
        assert_eq!(decoded[16], 6.0);
        assert_eq!(decoded[17], 8.0);
    }

    #[test]
    fn rejects_invalid_nvfp4_shapes() {
        let error = decode_nvfp4_values(&[0; 7], &[1.0], 1.0, 16, 1).unwrap_err();
        assert!(error.to_string().contains("expected 8"));
    }

    #[test]
    fn preserves_fp8_bit_patterns_when_loading_raw_weights() {
        let device = Device::Cpu;
        let values = [0x00u8, 0x38, 0xb8, 0x7e]
            .into_iter()
            .map(float8::F8E4M3::from_bits)
            .collect::<Vec<_>>();
        let mut tensors = HashMap::new();
        tensors.insert(
            "layer.weight".to_string(),
            Tensor::from_vec(values, (2, 2), &device).unwrap(),
        );
        let vb = VarBuilder::from_tensors(tensors, DType::F16, &device);
        assert_eq!(
            load_fp8_raw(&vb.pp("layer"), (2, 2), "weight").unwrap(),
            vec![0x00, 0x38, 0xb8, 0x7e]
        );
    }

    #[test]
    fn loads_modelopt_nvfp4_linear_from_safetensor_layout() {
        let device = Device::Cpu;
        let mut tensors = HashMap::new();
        tensors.insert(
            "layer.weight".to_string(),
            Tensor::from_vec(vec![0x92u8; 8], (1, 8), &device).unwrap(),
        );
        tensors.insert(
            "layer.weight_scale".to_string(),
            Tensor::from_vec(vec![2.0f32], (1, 1), &device).unwrap(),
        );
        tensors.insert(
            "layer.weight_scale_2".to_string(),
            Tensor::from_vec(vec![4.0f32], 1, &device).unwrap(),
        );
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
        let linear = linear_no_bias(16, 1, vb.pp("layer")).unwrap();
        assert_eq!(
            linear
                .dense_weight()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            vec![
                8.0, -4.0, 8.0, -4.0, 8.0, -4.0, 8.0, -4.0, 8.0, -4.0, 8.0, -4.0, 8.0, -4.0, 8.0,
                -4.0
            ]
        );
    }

    #[test]
    fn loads_fp8_style_linear_with_per_output_scale() {
        let device = Device::Cpu;
        let mut tensors = HashMap::new();
        tensors.insert(
            "layer.weight".to_string(),
            Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), &device).unwrap(),
        );
        tensors.insert(
            "layer.weight_scale".to_string(),
            Tensor::from_vec(vec![2.0f32, 0.5], 2, &device).unwrap(),
        );
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
        let linear = linear_no_bias(2, 2, vb.pp("layer")).unwrap();
        assert_eq!(
            linear
                .dense_weight()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            vec![2.0, 4.0, 1.5, 2.0]
        );
    }

    #[test]
    fn fp8_scales_stay_f32_when_runtime_dtype_is_f16() {
        let device = Device::Cpu;
        let mut tensors = HashMap::new();
        tensors.insert(
            "layer.weight".to_string(),
            Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), &device).unwrap(),
        );
        tensors.insert(
            "layer.weight_scale".to_string(),
            Tensor::from_vec(vec![2.0f32, 0.5], 2, &device).unwrap(),
        );
        let vb = VarBuilder::from_tensors(tensors, DType::F16, &device);
        let linear = linear_no_bias(2, 2, vb.pp("layer")).unwrap();
        assert_eq!(linear.dense_weight().dtype(), DType::F16);
        assert_eq!(
            linear
                .dense_weight()
                .flatten_all()
                .unwrap()
                .to_vec1::<f16>()
                .unwrap()
                .into_iter()
                .map(f16::to_f32)
                .collect::<Vec<_>>(),
            vec![2.0, 4.0, 1.5, 2.0]
        );
    }

    #[test]
    fn loads_compressed_tensors_nvfp4_with_reciprocal_global_scale() {
        let device = Device::Cpu;
        let mut tensors = HashMap::new();
        tensors.insert(
            "layer.weight_packed".to_string(),
            Tensor::from_vec(vec![0x92u8; 8], (1, 8), &device).unwrap(),
        );
        tensors.insert(
            "layer.weight_scale".to_string(),
            Tensor::from_vec(vec![2.0f32], (1, 1), &device).unwrap(),
        );
        tensors.insert(
            "layer.weight_global_scale".to_string(),
            Tensor::from_vec(vec![4.0f32], 1, &device).unwrap(),
        );
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
        let linear = linear_no_bias(16, 1, vb.pp("layer")).unwrap();
        assert_eq!(
            linear
                .dense_weight()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            vec![
                0.5, -0.25, 0.5, -0.25, 0.5, -0.25, 0.5, -0.25, 0.5, -0.25, 0.5, -0.25, 0.5, -0.25,
                0.5, -0.25
            ]
        );
    }
}
