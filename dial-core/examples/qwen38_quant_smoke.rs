#[cfg(feature = "cuda")]
fn main() -> anyhow::Result<()> {
    use candle_core::{DType, Device, Tensor};
    use candle_nn::Module;
    use float8::F8E4M3;
    use half::f16;

    let device = Device::new_cuda(0)?;
    let in_dim = 32usize;
    let out_dim = 3usize;
    let fp4_values = [
        0.0f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ];
    let packed_pattern = [0x92u8, 0x65, 0x38, 0x07, 0xab, 0xcd, 0xef, 0x14];
    let scale_pattern = [0x38u8, 0x30, 0x40, 0x28, 0x3c, 0x34];
    let packed = (0..out_dim * in_dim / 2)
        .map(|index| packed_pattern[index % packed_pattern.len()])
        .collect::<Vec<_>>();
    let nvfp4_scales = (0..out_dim * in_dim / 16)
        .map(|index| scale_pattern[index % scale_pattern.len()])
        .collect::<Vec<_>>();
    let nvfp4_global = 0.5f32;
    let nvfp4_dense = (0..out_dim * in_dim)
        .map(|index| {
            let row = index / in_dim;
            let column = index % in_dim;
            let byte = packed[row * in_dim / 2 + column / 2];
            let code = if column % 2 == 0 {
                byte & 0x0f
            } else {
                byte >> 4
            };
            let scale = nvfp4_scales[row * in_dim / 16 + column / 16];
            fp4_values[code as usize] * F8E4M3::from_bits(scale).to_f32() * nvfp4_global
        })
        .collect::<Vec<_>>();
    let nvfp4 = dial_core::models::qwen3_8::quant_linear_smoke_nvfp4(
        packed,
        nvfp4_scales,
        nvfp4_global,
        in_dim,
        out_dim,
        &device,
    )?;

    let fp8_pattern = [0x38u8, 0xb8, 0x30, 0x40, 0x01, 0x48, 0x34, 0xbc];
    let fp8_weight = (0..out_dim * in_dim)
        .map(|index| fp8_pattern[index % fp8_pattern.len()])
        .collect::<Vec<_>>();
    let fp8_scales = vec![0.5f32, 0.25, 0.75];
    let fp8_dense = fp8_weight
        .iter()
        .enumerate()
        .map(|(index, bits)| F8E4M3::from_bits(*bits).to_f32() * fp8_scales[index / in_dim])
        .collect::<Vec<_>>();
    let fp8 = dial_core::models::qwen3_8::quant_linear_smoke_fp8(
        fp8_weight, fp8_scales, in_dim, out_dim, &device,
    )?;

    for rows in [1usize, 7] {
        let x = Tensor::from_vec(
            (0..rows * in_dim)
                .map(|index| f16::from_f32((index as f32 * 0.07).sin()))
                .collect::<Vec<_>>(),
            (rows, in_dim),
            &device,
        )?;
        for (name, linear, dense_weight) in
            [("NVFP4", &nvfp4, &nvfp4_dense), ("FP8", &fp8, &fp8_dense)]
        {
            let dense = candle_nn::Linear::new(
                Tensor::from_vec(
                    dense_weight
                        .iter()
                        .map(|value| f16::from_f32(*value))
                        .collect::<Vec<_>>(),
                    (out_dim, in_dim),
                    &device,
                )?,
                None,
            );
            let expected = dense.forward(&x)?.to_dtype(DType::F32)?.flatten_all()?;
            let actual = linear.forward(&x)?.to_dtype(DType::F32)?.flatten_all()?;
            let expected_values = expected.to_vec1::<f32>()?;
            let actual_values = actual.to_vec1::<f32>()?;
            let max_abs = expected_values
                .iter()
                .zip(actual_values.iter())
                .map(|(left, right)| (left - right).abs())
                .fold(0.0f32, f32::max);
            println!("{name} rows={rows} max_abs={max_abs:.6}");
            if max_abs >= 0.06 {
                anyhow::bail!("{name} rows={rows} max_abs={max_abs} exceeds 0.06")
            }
        }
    }
    println!("Qwen3.8 CUDA QuantLinear smoke passed");
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("rebuild with --features cuda");
    std::process::exit(2);
}
