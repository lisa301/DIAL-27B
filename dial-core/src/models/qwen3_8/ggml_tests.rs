use super::{ggml::GgmlEngine, text::NativeTransformer, Qwen38Config, TextConfig};
use crate::{models::llama3::Cache, spm::Forwarder};
use candle_core::{
    quantized::{
        gguf_file::{self, Value},
        GgmlDType, QTensor,
    },
    DType, Device, IndexOp, Tensor,
};
use candle_nn::{Module, VarBuilder};
use std::{collections::HashMap, fs::File};

fn config() -> TextConfig {
    TextConfig {
        hidden_size: 256,
        intermediate_size: 512,
        vocab_size: 32,
        num_hidden_layers: 2,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: 64,
        rms_norm_eps: 1e-6,
        max_position_embeddings: 32,
        bos_token_id: None,
        eos_token_id: Some(1),
        layer_types: vec!["linear_attention".into(), "full_attention".into()],
        linear_conv_kernel_dim: 4,
        linear_key_head_dim: 128,
        linear_num_key_heads: 2,
        linear_num_value_heads: 4,
        linear_value_head_dim: 128,
        rope_parameters: None,
        partial_rotary_factor: 0.25,
    }
}
fn random(shape: impl Into<candle_core::Shape>, seed: usize, scale: f32) -> Tensor {
    let shape = shape.into();
    let values: Vec<f32> = (0..shape.elem_count())
        .map(|i| {
            let n = ((i + seed * 97) as u64)
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((n >> 32) % 10000) as f32 / 5000.0 - 1.0) * scale
        })
        .collect();
    Tensor::from_vec(values, shape, &Device::Cpu).unwrap()
}
struct Fixture {
    dir: std::path::PathBuf,
    gguf: std::path::PathBuf,
    tensors: HashMap<String, Tensor>,
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.gguf);
        let _ = std::fs::remove_file(self.dir.join("config.json"));
        let _ = std::fs::remove_file(self.dir.join("tokenizer.json"));
        let _ = std::fs::remove_file(self.dir.join("topology.yml"));
        let _ = std::fs::remove_dir(&self.dir);
    }
}
fn fixture() -> Fixture {
    fixture_for(&config())
}
fn fixture_for(text: &TextConfig) -> Fixture {
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("dial-ggml-test-{}-{unique}", std::process::id()));
    std::fs::create_dir(&dir).unwrap();
    let path = dir.join("tiny.gguf");
    let mut tensors = Vec::<(String, QTensor)>::new();
    let mut dequant = HashMap::new();
    let mut add = |name: String, tensor: Tensor, quant: bool| {
        let dtype = if quant && tensor.dims().last().unwrap() % 256 == 0 {
            GgmlDType::Q4K
        } else if quant {
            GgmlDType::Q8_0
        } else {
            GgmlDType::F32
        };
        let qt = QTensor::quantize(&tensor, dtype).unwrap();
        dequant.insert(name.clone(), qt.dequantize(&Device::Cpu).unwrap());
        tensors.push((name, qt));
    };
    add("token_embd.weight".into(), random((32, 256), 1, 0.15), true);
    add("output.weight".into(), random((32, 256), 2, 0.05), true);
    add(
        "output_norm.weight".into(),
        Tensor::ones(256, DType::F32, &Device::Cpu).unwrap(),
        false,
    );
    for index in 0..text.num_hidden_layers {
        let p = format!("blk.{index}.");
        let s = index * 100;
        for name in ["attn_norm.weight", "post_attention_norm.weight"] {
            add(
                format!("{p}{name}"),
                (random(256, s + 3, 0.1) + 1.0).unwrap(),
                false,
            );
        }
        add(
            format!("{p}ffn_gate.weight"),
            random((512, 256), s + 4, 0.03),
            true,
        );
        add(
            format!("{p}ffn_up.weight"),
            random((512, 256), s + 5, 0.03),
            true,
        );
        add(
            format!("{p}ffn_down.weight"),
            random((256, 512), s + 6, 0.03),
            true,
        );
        if text.layer_types[index] == "linear_attention" {
            add(
                format!("{p}attn_qkv.weight"),
                random((1024, 256), s + 7, 0.03),
                true,
            );
            add(
                format!("{p}attn_gate.weight"),
                random((512, 256), s + 8, 0.03),
                true,
            );
            add(
                format!("{p}ssm_beta.weight"),
                random((4, 256), s + 9, 0.03),
                true,
            );
            add(
                format!("{p}ssm_alpha.weight"),
                random((4, 256), s + 10, 0.03),
                true,
            );
            add(
                format!("{p}ssm_conv1d.weight"),
                random((1024, 4), s + 11, 0.2),
                false,
            );
            add(format!("{p}ssm_dt.bias"), random(4, s + 12, 0.2), false);
            add(
                format!("{p}ssm_a"),
                (random(4, s + 13, 0.2) - 1.0).unwrap(),
                false,
            );
            add(
                format!("{p}ssm_norm.weight"),
                (random(128, s + 14, 0.1) + 1.0).unwrap(),
                false,
            );
            add(
                format!("{p}ssm_out.weight"),
                random((256, 512), s + 15, 0.03),
                true,
            );
        } else {
            add(
                format!("{p}attn_q.weight"),
                random((256, 256), s + 7, 0.03),
                true,
            );
            add(
                format!("{p}attn_k.weight"),
                random((64, 256), s + 8, 0.03),
                true,
            );
            add(
                format!("{p}attn_v.weight"),
                random((64, 256), s + 9, 0.03),
                true,
            );
            add(
                format!("{p}attn_output.weight"),
                random((256, 128), s + 10, 0.03),
                true,
            );
            for name in ["attn_q_norm.weight", "attn_k_norm.weight"] {
                add(
                    format!("{p}{name}"),
                    (random(64, s + 11, 0.1) + 1.0).unwrap(),
                    false,
                );
            }
        }
    }
    let metadata = vec![
        ("general.architecture", Value::String("qwen35".into())),
        ("qwen35.embedding_length", Value::U32(256)),
        (
            "qwen35.block_count",
            Value::U32(text.num_hidden_layers as u32),
        ),
        ("qwen35.feed_forward_length", Value::U32(512)),
        ("qwen35.attention.head_count", Value::U32(2)),
        ("qwen35.attention.head_count_kv", Value::U32(1)),
        ("qwen35.rope.dimension_count", Value::U32(16)),
        ("qwen35.ssm.conv_kernel", Value::U32(4)),
        ("qwen35.ssm.inner_size", Value::U32(512)),
        ("qwen35.ssm.state_size", Value::U32(128)),
        ("qwen35.ssm.group_count", Value::U32(2)),
        ("qwen35.ssm.time_step_rank", Value::U32(4)),
        ("qwen35.rope.freq_base", Value::F32(10_000_000.0)),
        ("qwen35.attention.layer_norm_rms_epsilon", Value::F32(1e-6)),
    ];
    let refs: Vec<_> = metadata.iter().map(|(k, v)| (*k, v)).collect();
    let tensor_refs: Vec<_> = tensors.iter().map(|(k, v)| (k.as_str(), v)).collect();
    gguf_file::write(&mut File::create(&path).unwrap(), &refs, &tensor_refs).unwrap();
    Fixture {
        dir,
        gguf: path,
        tensors: dequant,
    }
}
// Invert the b10837 converter's [key_heads, repeat] -> [repeat, key_heads]
// reorder so the independent HF/Candle reference sees its original layout.
fn hf_heads(t: &Tensor, axis: usize, width: usize) -> Tensor {
    let mut dims = t.dims().to_vec();
    let original = dims.clone();
    dims.splice(axis..=axis, [2, 2, width]);
    t.reshape(dims)
        .unwrap()
        .transpose(axis, axis + 1)
        .unwrap()
        .contiguous()
        .unwrap()
        .reshape(original)
        .unwrap()
}
fn native_tensors(f: &Fixture, index: usize, device: &Device) -> HashMap<String, Tensor> {
    let p = format!("blk.{index}.");
    let hp = format!("model.language_model.layers.{index}.");
    let mut out = HashMap::new();
    let mut add = |from: &str, to: &str, norm: bool, head_axis: Option<(usize, usize)>| {
        let mut t = f.tensors[&format!("{p}{from}")].clone();
        if let Some((axis, width)) = head_axis {
            t = hf_heads(&t, axis, width);
        }
        if norm {
            t = (t - 1.0).unwrap();
        }
        out.insert(format!("{hp}{to}"), t.to_device(device).unwrap());
    };
    add("attn_norm.weight", "input_layernorm.weight", true, None);
    add(
        "post_attention_norm.weight",
        "post_attention_layernorm.weight",
        true,
        None,
    );
    for (a, b) in [
        ("ffn_gate", "gate_proj"),
        ("ffn_up", "up_proj"),
        ("ffn_down", "down_proj"),
    ] {
        add(
            &format!("{a}.weight"),
            &format!("mlp.{b}.weight"),
            false,
            None,
        );
    }
    if f.tensors.contains_key(&format!("{p}attn_qkv.weight")) {
        for (a, b, width) in [
            ("attn_gate.weight", "in_proj_z.weight", 128),
            ("ssm_beta.weight", "in_proj_b.weight", 1),
            ("ssm_alpha.weight", "in_proj_a.weight", 1),
            ("ssm_dt.bias", "dt_bias", 1),
        ] {
            add(a, &format!("linear_attn.{b}"), false, Some((0, width)));
        }
        add("ssm_norm.weight", "linear_attn.norm.weight", false, None);
        add(
            "ssm_out.weight",
            "linear_attn.out_proj.weight",
            false,
            Some((1, 128)),
        );
        drop(add);
        for (a, b) in [
            ("attn_qkv.weight", "in_proj_qkv.weight"),
            ("ssm_conv1d.weight", "conv1d.weight"),
        ] {
            let t = &f.tensors[&format!("{p}{a}")];
            let qk = t.narrow(0, 0, 512).unwrap();
            let v = hf_heads(&t.narrow(0, 512, 512).unwrap(), 0, 128);
            let mut t = Tensor::cat(&[qk, v], 0).unwrap();
            if a.starts_with("ssm_conv") {
                t = t.reshape((1024, 1, 4)).unwrap();
            }
            out.insert(format!("{hp}linear_attn.{b}"), t.to_device(device).unwrap());
        }
        let a = (&f.tensors[&format!("{p}ssm_a")] * -1.0)
            .unwrap()
            .log()
            .unwrap();
        out.insert(
            format!("{hp}linear_attn.A_log"),
            hf_heads(&a, 0, 1).to_device(device).unwrap(),
        );
    } else {
        for (a, b) in [
            ("attn_q", "q_proj"),
            ("attn_k", "k_proj"),
            ("attn_v", "v_proj"),
            ("attn_output", "o_proj"),
        ] {
            add(
                &format!("{a}.weight"),
                &format!("self_attn.{b}.weight"),
                false,
                None,
            );
        }
        add("attn_q_norm.weight", "self_attn.q_norm.weight", true, None);
        add("attn_k_norm.weight", "self_attn.k_norm.weight", true, None);
    }
    out
}
fn near(a: &Tensor, b: &Tensor, tolerance: f32) {
    let a = a
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let b = b
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    assert_eq!(a.len(), b.len());
    let error = a
        .iter()
        .zip(b)
        .map(|(x, y)| {
            assert!(x.is_finite() && y.is_finite());
            (x - y).abs()
        })
        .fold(0f32, f32::max);
    assert!(error <= tolerance, "max error {error} exceeds {tolerance}");
}

#[tokio::test]
#[ignore = "requires a built GGML adapter: set DIAL_GGML_TEST_LIB and run --ignored"]
async fn ggml_quantized_layer_prefill_decode_and_reset() {
    let library = std::env::var("DIAL_GGML_TEST_LIB").expect("set DIAL_GGML_TEST_LIB");
    let device = Device::Cpu;
    let f = fixture();
    let text = config();
    let full = Qwen38Config {
        model_type: "qwen3_5".into(),
        architectures: None,
        image_token_id: 29,
        video_token_id: None,
        vision_start_token_id: 30,
        vision_end_token_id: 31,
        text_config: text.clone(),
        quantization_config: None,
    };
    let cfg = full.generic_text_config();
    // A config-only HF directory must initialize without either a safetensors
    // index or any dense model shards when GGML execution is selected.
    let value = serde_json::json!({
        "model_type":"qwen3_5", "image_token_id":29,
        "vision_start_token_id":30, "vision_end_token_id":31,
        "text_config": {
            "hidden_size":text.hidden_size, "intermediate_size":text.intermediate_size,
            "vocab_size":text.vocab_size, "num_hidden_layers":text.num_hidden_layers,
            "num_attention_heads":text.num_attention_heads, "num_key_value_heads":text.num_key_value_heads,
            "head_dim":text.head_dim, "rms_norm_eps":text.rms_norm_eps,
            "max_position_embeddings":text.max_position_embeddings, "eos_token_id":1,
            "layer_types":text.layer_types, "linear_conv_kernel_dim":text.linear_conv_kernel_dim,
            "linear_key_head_dim":text.linear_key_head_dim, "linear_num_key_heads":text.linear_num_key_heads,
            "linear_num_value_heads":text.linear_num_value_heads, "linear_value_head_dim":text.linear_value_head_dim,
            "partial_rotary_factor":text.partial_rotary_factor
        }
    });
    std::fs::write(
        f.dir.join("config.json"),
        serde_json::to_vec(&value).unwrap(),
    )
    .unwrap();
    std::fs::write(f.dir.join("topology.yml"), b"{}\n").unwrap();
    use clap::Parser;
    let args = crate::Args::try_parse_from([
        "test",
        "--cpu",
        "--dtype",
        "f32",
        "--inference-backend",
        "qwen38-ggml",
        "--model",
        f.dir.to_str().unwrap(),
        "--qwen38-gguf",
        f.gguf.to_str().unwrap(),
        "--qwen38-ggml-lib",
        &library,
        "--topology",
        f.dir.join("topology.yml").to_str().unwrap(),
    ])
    .unwrap();
    let minimal = crate::spm::Context::from_args(args).unwrap();
    assert_eq!(
        minimal.config.qwen3_8_ggml.as_ref().unwrap().summary(),
        (0, 0)
    );
    drop(minimal);
    let e = GgmlEngine::open(std::path::Path::new(&library), &f.gguf, &text, 32, &device).unwrap();
    assert_eq!(e.summary(), (0, 0)); // no full-model load
    e.prepare_layer(0).unwrap();
    assert_eq!(e.summary().0, 14);
    // Resident quantized storage is materially smaller than its dense reference.
    let dense: usize = f
        .tensors
        .iter()
        .filter(|(k, _)| k.starts_with("blk.0."))
        .map(|(_, t)| t.elem_count() * 4)
        .sum();
    assert!(e.summary().1 < dense as u64 / 2);
    e.prepare_layer(1).unwrap();
    e.prepare_head().unwrap();
    let mut cache = Cache::new(true, DType::F32, &cfg, &device).unwrap();
    for index in 0..2 {
        let name = format!("model.language_model.layers.{index}");
        let native = NativeTransformer::load(
            name.clone(),
            VarBuilder::from_tensors(native_tensors(&f, index, &device), DType::F32, &device)
                .pp(&name),
            &cfg,
        )
        .unwrap();
        let mut native_cache = Cache::new(true, DType::F32, &cfg, &device).unwrap();
        let mut position = 0;
        for (step, n) in [3, 1, 1, 1].into_iter().enumerate() {
            let x = random((1, n, 256), 200 + step, 0.15);
            let y = e.layer(index, position, &x, &mut cache).unwrap();
            let golden = native
                .forward(&x, position, index, &mut native_cache)
                .await
                .unwrap();
            // GGML quantized matmuls also quantize activations to Q8, and FA
            // stores F16 KV. The comparison is not expected to be bit-exact.
            near(&y, &golden, 0.015);
            position += n;
        }
        let x = random((1, 3, 256), 900, 0.15);
        let y = e.layer(index, 0, &x, &mut cache).unwrap();
        let mut fresh = cache.as_new();
        near(&y, &e.layer(index, 0, &x, &mut fresh).unwrap(), 1e-6);
        // Compare chunked prefill+decode with one-shot causal evaluation.
        let x = random((1, 6, 256), 901, 0.15);
        let all = e.layer(index, 0, &x, &mut fresh).unwrap();
        let mut split = cache.as_new();
        let a = e
            .layer(index, 0, &x.narrow(1, 0, 3).unwrap(), &mut split)
            .unwrap();
        let b = e
            .layer(index, 3, &x.narrow(1, 3, 1).unwrap(), &mut split)
            .unwrap();
        let c = e
            .layer(index, 4, &x.narrow(1, 4, 2).unwrap(), &mut split)
            .unwrap();
        near(&all, &Tensor::cat(&[a, b, c], 1).unwrap(), 0.002);
        assert!(e
            .layer(index, 5, &random((1, 1, 256), 0, 0.1), &mut cache)
            .unwrap_err()
            .to_string()
            .contains("position mismatch"));
        assert!(e
            .layer(index, 31, &x, &mut cache)
            .unwrap_err()
            .to_string()
            .contains("context length"));
    }
    let ids = Tensor::new(&[[2u32, 3, 4]], &device).unwrap();
    let embedded = e.embedding(&ids).unwrap();
    let golden = f.tensors["token_embd.weight"]
        .index_select(&ids.flatten_all().unwrap(), 0)
        .unwrap()
        .reshape((1, 3, 256))
        .unwrap();
    near(&embedded, &golden, 1e-6);
    let row = embedded.i((.., 2, ..)).unwrap().contiguous().unwrap();
    let norm = candle_nn::RmsNorm::new(f.tensors["output_norm.weight"].clone(), 1e-6)
        .forward(&row)
        .unwrap();
    let logits = norm
        .matmul(&f.tensors["output.weight"].t().unwrap())
        .unwrap();
    near(&e.logits(&row).unwrap(), &logits, 0.03);
    cache.clear();
    assert!(cache.ggml_states.is_empty());
    assert!(cache.ggml_ranges.is_empty());
    let mut mismatch = text.clone();
    mismatch.hidden_size = 512;
    assert!(GgmlEngine::open(
        std::path::Path::new(&library),
        &f.gguf,
        &mismatch,
        32,
        &device
    )
    .unwrap_err()
    .to_string()
    .contains("mismatch"));

    // Exercise the real DIAL handshake/tensor protocol/Worker, not a mock
    // quantized API or a second external server process.
    use crate::spm::{ClientPool, Context, Mode, Node, Topology, Worker};
    let remote_engine =
        GgmlEngine::open(std::path::Path::new(&library), &f.gguf, &text, 32, &device).unwrap();
    let mut remote_cfg = cfg.clone();
    remote_cfg.qwen3_8_ggml = Some(remote_engine.clone());
    let mut topology = Topology::from_nodes(HashMap::new());
    let layer_name = "model.language_model.layers.0";
    topology.insert(
        "worker0".into(),
        Node {
            host: "127.0.0.1:0".into(),
            description: None,
            layers: vec![layer_name.into(), "model.language_model.layers.1".into()],
        },
    );
    let args = crate::Args {
        mode: Mode::Worker,
        name: Some("worker0".into()),
        address: "127.0.0.1:0".into(),
        ..Default::default()
    };
    let ctx = Context {
        args,
        dtype: DType::F32,
        topology,
        data_path: f.dir.clone(),
        device: device.clone(),
        config: remote_cfg,
        cache: cache.as_new(),
        var_builder: VarBuilder::zeros(DType::F32, &device),
    };
    let mut worker = Worker::<super::Qwen38>::new(ctx).await.unwrap();
    assert_eq!(remote_engine.summary().0, 25); // two assigned layers, no head
    let addr = worker.local_addr().unwrap().to_string();
    let task = tokio::spawn(async move { worker.run().await });
    let mut pool = ClientPool::default();
    let mut client = pool
        .client_for_layer(device.clone(), &addr, layer_name)
        .await
        .unwrap();
    let mut local_cache = cache.as_new();
    let mut client_cache = cache.as_new();
    for position in [0, 3, 4, 0] {
        let n = if position == 0 { 3 } else { 1 };
        let x = random((1, n, 256), 1234 + position, 0.15);
        let golden = e.layer(0, position, &x, &mut local_cache).unwrap();
        let golden = e.layer(1, position, &golden, &mut local_cache).unwrap();
        let actual = client
            .forward_batch(
                &x,
                vec![
                    (layer_name.into(), position, 0),
                    ("model.language_model.layers.1".into(), position, 1),
                ],
                &mut client_cache,
            )
            .await
            .unwrap();
        near(&golden, &actual, 1e-6);
    }
    // The actual TCP worker must have entered the fused path, not silently
    // fallen back to its per-layer implementation.
    assert!(remote_engine.fused_shards_enabled());
    assert_eq!(remote_engine.execution_counts(), (4, 4));
    // A second independent connection must have no recurrent state from the
    // previous connection, while the original connection remains usable.
    let mut pool2 = ClientPool::default();
    let mut client2 = pool2
        .client_for_layer(device.clone(), &addr, layer_name)
        .await
        .unwrap();
    let x = random((1, 3, 256), 4321, 0.15);
    let actual = client2
        .forward_mut(&x, 0, 0, &mut client_cache)
        .await
        .unwrap();
    near(&actual, &e.layer(0, 0, &x, &mut local_cache).unwrap(), 1e-6);
    drop(client2);
    drop(pool2);
    drop(client);
    drop(pool);
    task.abort();
    let _ = task.await;
}

#[tokio::test]
#[ignore = "requires a built ABI-2 adapter: set DIAL_GGML_TEST_LIB and run --ignored"]
async fn ggml_fused_shard_matches_layers_and_cache_lifecycle() {
    let library = std::env::var("DIAL_GGML_TEST_LIB").expect("set DIAL_GGML_TEST_LIB");
    let device = Device::Cpu;
    let mut text = config();
    text.num_hidden_layers = 4;
    text.layer_types = vec![
        "linear_attention".into(),
        "full_attention".into(),
        "linear_attention".into(),
        "full_attention".into(),
    ];
    let f = fixture_for(&text);
    let e = GgmlEngine::open(std::path::Path::new(&library), &f.gguf, &text, 32, &device).unwrap();
    for index in 0..4 {
        e.prepare_layer(index).unwrap();
    }
    let cfg = Qwen38Config {
        model_type: "qwen3_5".into(),
        architectures: None,
        image_token_id: 29,
        video_token_id: None,
        vision_start_token_id: 30,
        vision_end_token_id: 31,
        text_config: text,
        quantization_config: None,
    }
    .generic_text_config();
    let mut fused = Cache::new(true, DType::F32, &cfg, &device).unwrap();
    let mut separate = fused.as_new();
    for position in [0, 3, 4, 5, 0, 3] {
        let n = if position == 0 { 3 } else { 1 };
        let x = random((1, n, 256), 9000 + position, 0.15);
        let actual = e.range(0, 4, position, &x, &mut fused).unwrap();
        let mut expected = x;
        for index in 0..4 {
            expected = e.layer(index, position, &expected, &mut separate).unwrap();
        }
        near(&actual, &expected, 1e-6);
    }
    assert_eq!(fused.ggml_ranges.len(), 1);
    assert_eq!(fused.ggml_states.len(), 4);
    // A range and single-layer path share cache state, including transitions
    // between different range sizes on the same connection.
    let x = random((1, 1, 256), 9300, 0.15);
    let a = e.range(0, 2, 4, &x, &mut fused).unwrap();
    let a = e.range(2, 2, 4, &a, &mut fused).unwrap();
    let mut b = x;
    for index in 0..4 {
        b = e.layer(index, 4, &b, &mut separate).unwrap();
    }
    near(&a, &b, 1e-6);
    assert!(e
        .range(0, 4, 4, &b, &mut fused)
        .unwrap_err()
        .to_string()
        .contains("position mismatch"));
    assert!(e.range(0, 0, 0, &b, &mut fused).is_err());
    assert!(e.range(3, 2, 0, &b, &mut fused).is_err());
    assert!(e
        .range(0, 4, 32, &b, &mut fused)
        .unwrap_err()
        .to_string()
        .contains("context length"));
    let fresh = fused.as_new();
    assert!(fresh.ggml_ranges.is_empty() && fresh.ggml_states.is_empty());
    // Sequential request resets must preserve device allocations and warmed
    // graph handles while producing exactly the same result as fresh state.
    let retained_range = fused.ggml_ranges[&(0, 4)].clone();
    let retained_state = fused.ggml_states[&0].clone();
    for n in [3, 5, 1, 3] {
        fused.reuse_ggml_for_new_request();
        assert!(std::sync::Arc::ptr_eq(
            &retained_range,
            &fused.ggml_ranges[&(0, 4)]
        ));
        assert!(std::sync::Arc::ptr_eq(
            &retained_state,
            &fused.ggml_states[&0]
        ));
        let mut independent = fused.as_new();
        let x = random((1, n, 256), 9500 + n, 0.15);
        let actual = e.range(0, 4, 0, &x, &mut fused).unwrap();
        let expected = e.range(0, 4, 0, &x, &mut independent).unwrap();
        near(&actual, &expected, 1e-6);
        let x = random((1, 1, 256), 9600 + n, 0.15);
        near(
            &e.range(0, 4, n, &x, &mut fused).unwrap(),
            &e.range(0, 4, n, &x, &mut independent).unwrap(),
            1e-6,
        );
    }
    fused.clear();
    assert!(fused.ggml_ranges.is_empty() && fused.ggml_states.is_empty());
    // Independent ranges can never borrow another engine's states.
    let other = GgmlEngine::open(
        std::path::Path::new(&library),
        &f.gguf,
        cfg.qwen3_8.as_deref().unwrap(),
        32,
        &device,
    )
    .unwrap();
    for index in 0..4 {
        other.prepare_layer(index).unwrap();
    }
    e.range(0, 4, 0, &b, &mut fused).unwrap();
    assert!(other
        .range(0, 4, 0, &b, &mut fused)
        .unwrap_err()
        .to_string()
        .contains("another engine"));
}

#[tokio::test]
#[ignore = "requires a built ABI-2 adapter: set DIAL_GGML_TEST_LIB and run --ignored"]
async fn ggml_master_warmup_keeps_user_history_rng_and_request_state_clean() {
    use crate::models::{chat::Message, Generator};
    use clap::Parser;
    let library = std::env::var("DIAL_GGML_TEST_LIB").expect("set DIAL_GGML_TEST_LIB");
    let f = fixture();
    let text = config();
    let value = serde_json::json!({
        "model_type":"qwen3_5", "image_token_id":29,
        "vision_start_token_id":30, "vision_end_token_id":31,
        "text_config": {
            "hidden_size":text.hidden_size, "intermediate_size":text.intermediate_size,
            "vocab_size":text.vocab_size, "num_hidden_layers":text.num_hidden_layers,
            "num_attention_heads":text.num_attention_heads, "num_key_value_heads":text.num_key_value_heads,
            "head_dim":text.head_dim, "rms_norm_eps":text.rms_norm_eps,
            "max_position_embeddings":text.max_position_embeddings, "eos_token_id":1,
            "layer_types":text.layer_types, "linear_conv_kernel_dim":text.linear_conv_kernel_dim,
            "linear_key_head_dim":text.linear_key_head_dim, "linear_num_key_heads":text.linear_num_key_heads,
            "linear_num_value_heads":text.linear_num_value_heads, "linear_value_head_dim":text.linear_value_head_dim,
            "partial_rotary_factor":text.partial_rotary_factor
        }
    });
    std::fs::write(
        f.dir.join("config.json"),
        serde_json::to_vec(&value).unwrap(),
    )
    .unwrap();
    std::fs::write(f.dir.join("topology.yml"), b"{}\n").unwrap();
    let vocab = (0..32)
        .map(|id| {
            (
                if id == 0 {
                    "[UNK]".into()
                } else if id == 1 {
                    "<|im_end|>".into()
                } else {
                    format!("t{id}")
                },
                id,
            )
        })
        .collect();
    let wordlevel = tokenizers::models::wordlevel::WordLevel::builder()
        .vocab(vocab)
        .unk_token("[UNK]".into())
        .build()
        .unwrap();
    let mut tokenizer = tokenizers::Tokenizer::new(wordlevel);
    tokenizer.with_pre_tokenizer(Some(
        tokenizers::pre_tokenizers::whitespace::WhitespaceSplit,
    ));
    tokenizer.save(f.dir.join("tokenizer.json"), false).unwrap();
    let args = crate::Args::try_parse_from([
        "test",
        "--cpu",
        "--dtype",
        "f32",
        "--inference-backend",
        "qwen38-ggml",
        "--model",
        f.dir.to_str().unwrap(),
        "--qwen38-gguf",
        f.gguf.to_str().unwrap(),
        "--qwen38-ggml-lib",
        &library,
        "--topology",
        f.dir.join("topology.yml").to_str().unwrap(),
        "--system-prompt",
        "",
        "--qwen38-thinking",
        "false",
    ])
    .unwrap();
    let mut ctx = crate::spm::Context::from_args(args).unwrap();
    let engine = ctx.config.qwen3_8_ggml.as_ref().unwrap().clone();
    ctx.args.prompt = "t10 t11".into();
    let mut warmed = super::Qwen38::load(ctx.clone()).await.unwrap();
    // Five forwards, each embedding + fused shard + output head. Prewarming
    // really executed both attention types, rather than merely loading weights.
    assert_eq!(engine.execution_counts(), (15, 5));
    assert_eq!(warmed.generated_tokens(), 0);
    assert_eq!(
        crate::spm::snapshot_distributed_profile().remote_requests,
        0
    );
    // Warmup must not become a user message or advance the sampling RNG. Two
    // independent models with the same seed must produce the same sequence.
    ctx.args.prompt = "t20 t21 t22 t23".into();
    let mut independent = super::Qwen38::load(ctx).await.unwrap();
    warmed.reset().unwrap();
    independent.reset().unwrap();
    for model in [&mut warmed, &mut independent] {
        model.add_message(Message::user("t2 t3".into())).unwrap();
    }
    for index in 0..3 {
        let a = warmed.next_token(index).await.unwrap();
        let b = independent.next_token(index).await.unwrap();
        assert_eq!(
            (a.id, a.text, a.is_end_of_stream),
            (b.id, b.text, b.is_end_of_stream)
        );
    }
    // A second dialog with a different prompt length must also start clean.
    warmed.reset().unwrap();
    independent.reset().unwrap();
    for model in [&mut warmed, &mut independent] {
        model.add_message(Message::user("t4".into())).unwrap();
    }
    let a = warmed.next_token(0).await.unwrap();
    let b = independent.next_token(0).await.unwrap();
    assert_eq!(
        (a.id, a.text, a.is_end_of_stream),
        (b.id, b.text, b.is_end_of_stream)
    );
}

#[test]
#[ignore = "requires a built ABI-2 adapter: set DIAL_GGML_TEST_LIB and run --ignored"]
fn ggml_full_depth_shard_uses_one_graph_per_forward() {
    let library = std::env::var("DIAL_GGML_TEST_LIB").expect("set DIAL_GGML_TEST_LIB");
    let device = Device::Cpu;
    let mut text = config();
    text.num_hidden_layers = 64;
    text.layer_types = (0..64)
        .map(|i| {
            if i % 4 == 3 {
                "full_attention".into()
            } else {
                "linear_attention".into()
            }
        })
        .collect();
    let f = fixture_for(&text);
    let cfg = Qwen38Config {
        model_type: "qwen3_5".into(),
        architectures: None,
        image_token_id: 29,
        video_token_id: None,
        vision_start_token_id: 30,
        vision_end_token_id: 31,
        text_config: text.clone(),
        quantization_config: None,
    }
    .generic_text_config();
    let e = GgmlEngine::open(std::path::Path::new(&library), &f.gguf, &text, 32, &device).unwrap();
    for index in 0..64 {
        e.prepare_layer(index).unwrap();
    }
    assert_eq!(e.summary().0, 848); // only layer weights, no embedding/head
    let mut cache = Cache::new(true, DType::F32, &cfg, &device).unwrap();
    for (position, n) in [(0, 2), (2, 1)] {
        let x = random((1, n, 256), 11000 + position, 0.15);
        let y = e.range(0, 64, position, &x, &mut cache).unwrap();
        assert_eq!(y.dims(), x.dims());
        assert!(y
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .all(|v| v.is_finite()));
    }
    assert_eq!(e.execution_counts(), (2, 2));
}

#[test]
#[ignore = "requires a built ABI-2 adapter: set DIAL_GGML_TEST_LIB and run --ignored"]
fn ggml_active_kv_bucket_growth_matches_one_shot() {
    let library = std::env::var("DIAL_GGML_TEST_LIB").expect("set DIAL_GGML_TEST_LIB");
    let device = Device::Cpu;
    let mut text = config();
    text.max_position_embeddings = 513;
    text.layer_types = vec!["full_attention".into(), "full_attention".into()];
    let f = fixture_for(&text);
    let cfg = Qwen38Config {
        model_type: "qwen3_5".into(),
        architectures: None,
        image_token_id: 29,
        video_token_id: None,
        vision_start_token_id: 30,
        vision_end_token_id: 31,
        text_config: text.clone(),
        quantization_config: None,
    }
    .generic_text_config();
    let e = GgmlEngine::open(std::path::Path::new(&library), &f.gguf, &text, 513, &device).unwrap();
    for index in 0..2 {
        e.prepare_layer(index).unwrap();
    }
    let mut all_cache = Cache::new(true, DType::F32, &cfg, &device).unwrap();
    let mut split_cache = all_cache.as_new();
    let mut separate_cache = all_cache.as_new();
    let x = random((1, 258, 256), 10000, 0.15);
    let all = e.range(0, 2, 0, &x, &mut all_cache).unwrap();
    let ref_all = e.layer(0, 0, &x, &mut separate_cache).unwrap();
    let ref_all = e.layer(1, 0, &ref_all, &mut separate_cache).unwrap();
    near(&all, &ref_all, 1e-6);
    separate_cache.clear();
    let a = e
        .range(
            0,
            2,
            0,
            &x.narrow(1, 0, 255).unwrap().contiguous().unwrap(),
            &mut split_cache,
        )
        .unwrap();
    let mut parts = vec![a];
    let ref_a = e
        .layer(0, 0, &x.narrow(1, 0, 255).unwrap(), &mut separate_cache)
        .unwrap();
    let ref_a = e.layer(1, 0, &ref_a, &mut separate_cache).unwrap();
    near(&parts[0], &ref_a, 1e-6);
    // Decode within the 256 bucket, then rebuild in the 512 bucket without
    // resetting previous KV. Two FA layers share only mask/position inputs.
    for position in 255..258 {
        parts.push(
            e.range(
                0,
                2,
                position,
                &x.narrow(1, position, 1).unwrap().contiguous().unwrap(),
                &mut split_cache,
            )
            .unwrap(),
        );
        let ref_b = e
            .layer(
                0,
                position,
                &x.narrow(1, position, 1).unwrap(),
                &mut separate_cache,
            )
            .unwrap();
        let ref_b = e.layer(1, position, &ref_b, &mut separate_cache).unwrap();
        near(parts.last().unwrap(), &ref_b, 1e-6);
    }
    // Large prefill GEMM and one-token GEMV use different upstream activation
    // quantization paths. Fused vs per-layer results above must remain exact;
    // one-shot vs chunked evaluation uses the dense-reference tolerance.
    near(&all, &Tensor::cat(&parts, 1).unwrap(), 0.015);
    let reset = e
        .range(
            0,
            2,
            0,
            &x.narrow(1, 0, 3).unwrap().contiguous().unwrap(),
            &mut split_cache,
        )
        .unwrap();
    near(&reset, &all.narrow(1, 0, 3).unwrap(), 0.015);
}

#[cfg(feature = "cuda")]
#[tokio::test]
#[ignore = "requires the node's CUDA-built GGML adapter and a visible GPU"]
async fn ggml_cuda_device_interop_smoke() {
    let library = std::env::var("DIAL_GGML_TEST_LIB").expect("set DIAL_GGML_TEST_LIB");
    let ordinal = std::env::var("DIAL_GGML_TEST_DEVICE")
        .unwrap_or_else(|_| "0".into())
        .parse()
        .unwrap();
    let device = Device::new_cuda(ordinal).expect("CUDA device unavailable; run on Thor/Orin");
    let f = fixture();
    let text = config();
    let cfg = Qwen38Config {
        model_type: "qwen3_5".into(),
        architectures: None,
        image_token_id: 29,
        video_token_id: None,
        vision_start_token_id: 30,
        vision_end_token_id: 31,
        text_config: text.clone(),
        quantization_config: None,
    }
    .generic_text_config();
    let gpu =
        GgmlEngine::open(std::path::Path::new(&library), &f.gguf, &text, 32, &device).unwrap();
    let cpu = GgmlEngine::open(
        std::path::Path::new(&library),
        &f.gguf,
        &text,
        32,
        &Device::Cpu,
    )
    .unwrap();
    let mut gpu_cache = Cache::new(true, DType::F16, &cfg, &device).unwrap();
    let mut cpu_cache = Cache::new(true, DType::F32, &cfg, &Device::Cpu).unwrap();
    for index in 0..2 {
        gpu.prepare_layer(index).unwrap();
        cpu.prepare_layer(index).unwrap();
        let mut position = 0;
        for n in [3, 1, 1, 1] {
            let x = random((1, n, 256), 700 + position, 0.15)
                .to_dtype(DType::F16)
                .unwrap();
            let y = gpu
                .layer(
                    index,
                    position,
                    &x.to_device(&device).unwrap(),
                    &mut gpu_cache,
                )
                .unwrap();
            let golden = cpu.layer(index, position, &x, &mut cpu_cache).unwrap();
            near(&y, &golden, 0.025);
            position += n;
        }
        let x = random((1, 3, 256), 800, 0.15)
            .to_dtype(DType::F16)
            .unwrap()
            .to_device(&device)
            .unwrap();
        let y = gpu.layer(index, 0, &x, &mut gpu_cache).unwrap();
        near(
            &y,
            &gpu.layer(index, 0, &x, &mut gpu_cache.as_new()).unwrap(),
            1e-5,
        );
    }
    let before = gpu.execution_counts();
    let mut retained_range = None;
    for position in [0, 3, 4, 5, 6, 0, 3] {
        if position == 0 {
            gpu_cache.reuse_ggml_for_new_request();
            cpu_cache.reuse_ggml_for_new_request();
        }
        let n = if position == 0 { 3 } else { 1 };
        let x = random((1, n, 256), 1000 + position, 0.15)
            .to_dtype(DType::F16)
            .unwrap();
        let y = gpu
            .range(
                0,
                2,
                position,
                &x.to_device(&device).unwrap(),
                &mut gpu_cache,
            )
            .unwrap();
        let golden = cpu.range(0, 2, position, &x, &mut cpu_cache).unwrap();
        near(&y, &golden, 0.025);
        if let Some(retained) = &retained_range {
            assert!(std::sync::Arc::ptr_eq(
                retained,
                &gpu_cache.ggml_ranges[&(0, 2)]
            ));
        } else {
            retained_range = Some(gpu_cache.ggml_ranges[&(0, 2)].clone());
        }
    }
    let after = gpu.execution_counts();
    assert_eq!((after.0 - before.0, after.1 - before.1), (7, 7));
    gpu.prepare_head().unwrap();
    cpu.prepare_head().unwrap();
    assert_eq!(gpu.summary(), cpu.summary());
    let ids = Tensor::new(&[[2u32, 3, 4]], &Device::Cpu).unwrap();
    let embedded = gpu.embedding(&ids.to_device(&device).unwrap()).unwrap();
    near(&embedded, &cpu.embedding(&ids).unwrap(), 1e-6);
    let row = embedded.i((.., 2, ..)).unwrap().contiguous().unwrap();
    near(
        &gpu.logits(&row).unwrap(),
        &cpu.logits(&row.to_device(&Device::Cpu).unwrap()).unwrap(),
        0.03,
    );
    // Exercise CUDA graph warmup/replay, shared attention inputs and a KV
    // bucket rebuild on the actual device, not just on the CPU oracle.
    let mut grown_text = config();
    grown_text.max_position_embeddings = 513;
    grown_text.layer_types = vec!["full_attention".into(), "full_attention".into()];
    let grown_fixture = fixture_for(&grown_text);
    let mut grown_cfg = cfg.clone();
    grown_cfg.max_seq_len = 513;
    grown_cfg.qwen3_8 = Some(std::sync::Arc::new(grown_text.clone()));
    let grown_gpu = GgmlEngine::open(
        std::path::Path::new(&library),
        &grown_fixture.gguf,
        &grown_text,
        513,
        &device,
    )
    .unwrap();
    let grown_cpu = GgmlEngine::open(
        std::path::Path::new(&library),
        &grown_fixture.gguf,
        &grown_text,
        513,
        &Device::Cpu,
    )
    .unwrap();
    for index in 0..2 {
        grown_gpu.prepare_layer(index).unwrap();
        grown_cpu.prepare_layer(index).unwrap();
    }
    let mut grown_gpu_cache = Cache::new(true, DType::F16, &grown_cfg, &device).unwrap();
    let mut grown_cpu_cache = Cache::new(true, DType::F32, &grown_cfg, &Device::Cpu).unwrap();
    for (position, n) in [(0, 255), (255, 1), (256, 1), (257, 1), (258, 1), (0, 3)] {
        let x = random((1, n, 256), 12000 + position, 0.15)
            .to_dtype(DType::F16)
            .unwrap();
        let actual = grown_gpu
            .range(
                0,
                2,
                position,
                &x.to_device(&device).unwrap(),
                &mut grown_gpu_cache,
            )
            .unwrap();
        let expected = grown_cpu
            .range(0, 2, position, &x, &mut grown_cpu_cache)
            .unwrap();
        near(&actual, &expected, 0.025);
    }
    assert_eq!(grown_gpu.execution_counts(), (6, 6));
    device.synchronize().unwrap();
}
