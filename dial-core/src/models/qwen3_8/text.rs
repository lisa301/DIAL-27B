use anyhow::Result;
use async_trait::async_trait;
use candle_core::{DType, Tensor, D};
use candle_nn::{Module, VarBuilder};

use crate::{
    models::llama3::{Cache, Config},
    spm::Forwarder,
};

use super::{load_tensor, quantized_linear_no_bias, QuantLinear, TextConfig};

#[derive(Debug, Clone)]
pub struct ZeroCenteredRmsNorm {
    weight: Tensor,
    eps: f64,
}

impl ZeroCenteredRmsNorm {
    pub fn load(size: usize, eps: f64, vb: VarBuilder) -> candle_core::Result<Self> {
        Ok(Self {
            weight: load_tensor(vb, size, "weight")?,
            eps,
        })
    }

    pub fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let dtype = x.dtype();
        let x_f32 = x.to_dtype(DType::F32)?;
        let width = *x
            .dims()
            .last()
            .ok_or_else(|| candle_core::Error::Msg("RMSNorm input has no dimensions".into()))?;
        let variance = (x_f32.sqr()?.sum_keepdim(D::Minus1)? / width as f64)?;
        let normalized = x_f32.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        let weight = (self.weight.to_dtype(DType::F32)? + 1.0)?;
        normalized.broadcast_mul(&weight)?.to_dtype(dtype)
    }
}

#[derive(Debug, Clone)]
struct Mlp {
    gate: QuantLinear,
    up: QuantLinear,
    down: QuantLinear,
}

impl Mlp {
    fn load(vb: VarBuilder, cfg: &TextConfig) -> candle_core::Result<Self> {
        Ok(Self {
            gate: quantized_linear_no_bias(
                cfg.hidden_size,
                cfg.intermediate_size,
                vb.pp("gate_proj"),
            )?,
            up: quantized_linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("up_proj"))?,
            down: quantized_linear_no_bias(
                cfg.intermediate_size,
                cfg.hidden_size,
                vb.pp("down_proj"),
            )?,
        })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let gated = (candle_nn::ops::silu(&self.gate.forward(x)?)? * self.up.forward(x)?)?;
        self.down.forward(&gated)
    }
}

fn repeat_kv(x: Tensor, repetitions: usize) -> candle_core::Result<Tensor> {
    candle_transformers::utils::repeat_kv(x, repetitions)
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> candle_core::Result<Tensor> {
    let fill = Tensor::new(on_true, on_false.device())?
        .to_dtype(on_false.dtype())?
        .broadcast_as(mask.shape())?;
    mask.where_cond(&fill, on_false)
}

fn rotate_partial(
    x: &Tensor,
    rotary_dim: usize,
    index_pos: usize,
    cache: &Cache,
) -> candle_core::Result<Tensor> {
    let last_dim = *x
        .dims()
        .last()
        .ok_or_else(|| candle_core::Error::Msg("RoPE input has no dimensions".into()))?;
    let seq_len = x.dims()[2];
    let rotated = x.narrow(D::Minus1, 0, rotary_dim)?.contiguous()?;
    let rotated = candle_nn::rotary_emb::rope(
        &rotated,
        &cache.cosine(index_pos, seq_len)?,
        &cache.sine(index_pos, seq_len)?,
    )?;
    if rotary_dim == last_dim {
        Ok(rotated)
    } else {
        Tensor::cat(
            &[
                &rotated,
                &x.narrow(D::Minus1, rotary_dim, last_dim - rotary_dim)?,
            ],
            D::Minus1,
        )
    }
}

#[derive(Debug, Clone)]
struct FullAttention {
    q_proj: QuantLinear,
    k_proj: QuantLinear,
    v_proj: QuantLinear,
    o_proj: QuantLinear,
    q_norm: ZeroCenteredRmsNorm,
    k_norm: ZeroCenteredRmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    attn_f32: bool,
}

impl FullAttention {
    fn load(vb: VarBuilder, cfg: &TextConfig, attn_f32: bool) -> candle_core::Result<Self> {
        let q_size = cfg.num_attention_heads * cfg.head_dim;
        let kv_size = cfg.num_key_value_heads * cfg.head_dim;
        Ok(Self {
            q_proj: quantized_linear_no_bias(cfg.hidden_size, q_size * 2, vb.pp("q_proj"))?,
            k_proj: quantized_linear_no_bias(cfg.hidden_size, kv_size, vb.pp("k_proj"))?,
            v_proj: quantized_linear_no_bias(cfg.hidden_size, kv_size, vb.pp("v_proj"))?,
            o_proj: quantized_linear_no_bias(q_size, cfg.hidden_size, vb.pp("o_proj"))?,
            q_norm: ZeroCenteredRmsNorm::load(cfg.head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?,
            k_norm: ZeroCenteredRmsNorm::load(cfg.head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            rotary_dim: cfg.rotary_dim(),
            attn_f32,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        cache: &mut Cache,
    ) -> Result<Tensor> {
        let (batch, seq_len, _) = x.dims3()?;
        let q_size = self.num_heads * self.head_dim;
        // Qwen3.8 stores q and gate next to each other inside every head:
        // [head0.q, head0.gate, head1.q, head1.gate, ...]. Splitting the
        // flattened projection in half would silently mix different heads.
        let q_gate =
            self.q_proj
                .forward(x)?
                .reshape((batch, seq_len, self.num_heads, self.head_dim * 2))?;
        let q = q_gate.narrow(D::Minus1, 0, self.head_dim)?;
        let gate = q_gate
            .narrow(D::Minus1, self.head_dim, self.head_dim)?
            .reshape((batch, seq_len, q_size))?;
        let q = self.q_norm.forward(&q)?.transpose(1, 2)?.contiguous()?;
        let k = self
            .k_norm
            .forward(&self.k_proj.forward(x)?.reshape((
                batch,
                seq_len,
                self.num_kv_heads,
                self.head_dim,
            ))?)?
            .transpose(1, 2)?
            .contiguous()?;
        let v = self
            .v_proj
            .forward(x)?
            .reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let q = rotate_partial(&q, self.rotary_dim, index_pos, cache)?;
        let k = rotate_partial(&k, self.rotary_dim, index_pos, cache)?;
        let (k, v) = if batch == 1 && seq_len == 1 && index_pos > 0 {
            cache.process_kv_decode_in_place(block_idx, index_pos, k, v)?
        } else {
            cache.process_kv(block_idx, index_pos, k, v)?
        };
        let k = repeat_kv(k, self.num_heads / self.num_kv_heads)?;
        let v = repeat_kv(v, self.num_heads / self.num_kv_heads)?;

        let input_dtype = q.dtype();
        let compute_dtype = if self.attn_f32 {
            DType::F32
        } else {
            input_dtype
        };
        // RoPE, repeat_kv and cache views can all produce valid strided
        // tensors. Candle's CUDA matmul currently requires both operands to
        // be contiguous, including the transposed K operand used by QK^T.
        let q = q.to_dtype(compute_dtype)?.contiguous()?;
        let k_t = k.to_dtype(compute_dtype)?.t()?.contiguous()?;
        let v = v.to_dtype(compute_dtype)?.contiguous()?;
        let scores = (q.matmul(&k_t)? / (self.head_dim as f64).sqrt())?;
        let scores = if seq_len > 1 {
            let mask = cache.mask(seq_len)?.broadcast_as(scores.shape())?;
            masked_fill(&scores, &mask, f32::NEG_INFINITY)?
        } else {
            scores
        };
        // The reference path computes the softmax in f32 even when Q/K/V and
        // their matmuls use the selected lower precision.
        let probabilities = candle_nn::ops::softmax_last_dim(&scores.to_dtype(DType::F32)?)?
            .to_dtype(compute_dtype)?;
        let output = probabilities
            .matmul(&v)?
            .to_dtype(input_dtype)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch, seq_len, q_size))?;
        let output = output.broadcast_mul(&candle_nn::ops::sigmoid(&gate)?)?;
        Ok(self.o_proj.forward(&output)?)
    }
}

#[derive(Debug, Clone)]
struct GatedRmsNorm {
    weight: Tensor,
    eps: f64,
}

impl GatedRmsNorm {
    fn load(size: usize, eps: f64, vb: VarBuilder) -> candle_core::Result<Self> {
        Ok(Self {
            weight: vb.get(size, "weight")?,
            eps,
        })
    }

    fn forward(&self, x: &Tensor, gate: &Tensor) -> candle_core::Result<Tensor> {
        let dtype = x.dtype();
        let x = x.to_dtype(DType::F32)?;
        let width = *x.dims().last().unwrap_or(&1);
        let norm = (x.sqr()?.sum_keepdim(D::Minus1)? / width as f64)?;
        let x = x.broadcast_div(&(norm + self.eps)?.sqrt()?)?;
        let x = x.broadcast_mul(&self.weight.to_dtype(DType::F32)?)?;
        x.broadcast_mul(&candle_nn::ops::silu(&gate.to_dtype(DType::F32)?)?)?
            .to_dtype(dtype)
    }
}

#[derive(Debug, Clone)]
struct GatedDeltaNet {
    in_proj_qkv: QuantLinear,
    in_proj_z: QuantLinear,
    in_proj_b: QuantLinear,
    in_proj_a: QuantLinear,
    conv_weight: Tensor,
    dt_bias: Tensor,
    a_log: Tensor,
    norm: GatedRmsNorm,
    out_proj: QuantLinear,
    num_k_heads: usize,
    num_v_heads: usize,
    key_head_dim: usize,
    value_head_dim: usize,
    conv_kernel: usize,
    conv_dim: usize,
}

impl GatedDeltaNet {
    fn load(vb: VarBuilder, cfg: &TextConfig) -> candle_core::Result<Self> {
        let key_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim;
        let value_dim = cfg.linear_num_value_heads * cfg.linear_value_head_dim;
        let conv_dim = key_dim * 2 + value_dim;
        Ok(Self {
            in_proj_qkv: quantized_linear_no_bias(cfg.hidden_size, conv_dim, vb.pp("in_proj_qkv"))?,
            in_proj_z: quantized_linear_no_bias(cfg.hidden_size, value_dim, vb.pp("in_proj_z"))?,
            in_proj_b: quantized_linear_no_bias(
                cfg.hidden_size,
                cfg.linear_num_value_heads,
                vb.pp("in_proj_b"),
            )?,
            in_proj_a: quantized_linear_no_bias(
                cfg.hidden_size,
                cfg.linear_num_value_heads,
                vb.pp("in_proj_a"),
            )?,
            conv_weight: load_tensor(
                vb.clone(),
                (conv_dim, 1, cfg.linear_conv_kernel_dim),
                "conv1d.weight",
            )?,
            dt_bias: load_tensor(vb.clone(), cfg.linear_num_value_heads, "dt_bias")?,
            a_log: load_tensor(vb.clone(), cfg.linear_num_value_heads, "A_log")?,
            norm: GatedRmsNorm::load(cfg.linear_value_head_dim, cfg.rms_norm_eps, vb.pp("norm"))?,
            out_proj: quantized_linear_no_bias(value_dim, cfg.hidden_size, vb.pp("out_proj"))?,
            num_k_heads: cfg.linear_num_key_heads,
            num_v_heads: cfg.linear_num_value_heads,
            key_head_dim: cfg.linear_key_head_dim,
            value_head_dim: cfg.linear_value_head_dim,
            conv_kernel: cfg.linear_conv_kernel_dim,
            conv_dim,
        })
    }

    fn causal_conv(
        &self,
        mixed: &Tensor,
        previous: Option<&Tensor>,
    ) -> candle_core::Result<(Tensor, Tensor)> {
        let (batch, seq_len, channels) = mixed.dims3()?;
        let mixed = mixed.transpose(1, 2)?.contiguous()?;
        let history = self.conv_kernel.saturating_sub(1);
        let (input, padding) = if let Some(previous) = previous {
            (Tensor::cat(&[previous, &mixed], 2)?, 0)
        } else {
            (mixed.clone(), history)
        };
        let output = input.conv1d(&self.conv_weight, padding, 1, 1, self.conv_dim)?;
        let output = if previous.is_none() {
            output.narrow(2, 0, seq_len)?
        } else {
            output
        };
        let state_source = if previous.is_some() { input } else { mixed };
        let state_len = state_source.dims()[2];
        let state = if state_len >= history {
            state_source
                .narrow(2, state_len - history, history)?
                .contiguous()?
        } else {
            let zeros = Tensor::zeros(
                (batch, channels, history - state_len),
                state_source.dtype(),
                state_source.device(),
            )?;
            Tensor::cat(&[&zeros, &state_source], 2)?
        };
        Ok((candle_nn::ops::silu(&output)?.transpose(1, 2)?, state))
    }

    fn l2_normalize(x: &Tensor) -> candle_core::Result<Tensor> {
        let x = x.to_dtype(DType::F32)?;
        x.broadcast_div(&(x.sqr()?.sum_keepdim(D::Minus1)? + 1e-6)?.sqrt()?)
    }

    fn recurrent_scan(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        g: &Tensor,
        beta: &Tensor,
        mut recurrent: Tensor,
    ) -> candle_core::Result<(Tensor, Tensor)> {
        let seq_len = q.dims4()?.2;
        let mut outputs = Vec::with_capacity(seq_len);
        for token in 0..seq_len {
            let q_t = q.narrow(2, token, 1)?.squeeze(2)?;
            let k_t = k.narrow(2, token, 1)?.squeeze(2)?;
            let v_t = v.narrow(2, token, 1)?.squeeze(2)?;
            let beta_t = beta.narrow(2, token, 1)?.squeeze(2)?.unsqueeze(D::Minus1)?;
            let decay_t = g
                .narrow(2, token, 1)?
                .squeeze(2)?
                .exp()?
                .unsqueeze(D::Minus1)?
                .unsqueeze(D::Minus1)?;
            recurrent = recurrent.broadcast_mul(&decay_t)?;
            let prediction = recurrent
                .broadcast_mul(&k_t.unsqueeze(D::Minus1)?)?
                .sum(D::Minus2)?;
            let delta = v_t.broadcast_sub(&prediction)?.broadcast_mul(&beta_t)?;
            recurrent = (recurrent
                + k_t
                    .unsqueeze(D::Minus1)?
                    .broadcast_mul(&delta.unsqueeze(D::Minus2)?)?)?;
            outputs.push(
                recurrent
                    .broadcast_mul(&q_t.unsqueeze(D::Minus1)?)?
                    .sum(D::Minus2)?
                    .unsqueeze(2)?,
            );
        }
        Ok((
            Tensor::cat(&outputs.iter().collect::<Vec<_>>(), 2)?,
            recurrent,
        ))
    }

    fn forward(
        &self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        cache: &mut Cache,
    ) -> Result<Tensor> {
        let (batch, seq_len, _) = x.dims3()?;
        let activation_dtype = x.dtype();
        let previous = if index_pos == 0 {
            None
        } else {
            cache.linear_state(block_idx)
        };
        let mixed = self.in_proj_qkv.forward(x)?;
        let (mixed, conv_state) =
            self.causal_conv(&mixed, previous.as_ref().map(|state| &state.conv))?;
        let key_size = self.num_k_heads * self.key_head_dim;
        let value_size = self.num_v_heads * self.value_head_dim;
        let q = mixed
            .narrow(D::Minus1, 0, key_size)?
            .reshape((batch, seq_len, self.num_k_heads, self.key_head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = mixed
            .narrow(D::Minus1, key_size, key_size)?
            .reshape((batch, seq_len, self.num_k_heads, self.key_head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = mixed
            .narrow(D::Minus1, key_size * 2, value_size)?
            .reshape((batch, seq_len, self.num_v_heads, self.value_head_dim))?
            .transpose(1, 2)?
            .contiguous()?
            .to_dtype(DType::F32)?;
        let repetitions = self.num_v_heads / self.num_k_heads;
        let q =
            Self::l2_normalize(&repeat_kv(q, repetitions)?)? / (self.key_head_dim as f64).sqrt();
        let q = q?;
        let k = Self::l2_normalize(&repeat_kv(k, repetitions)?)?;
        let beta = candle_nn::ops::sigmoid(&self.in_proj_b.forward(x)?)?
            .transpose(1, 2)?
            .to_dtype(DType::F32)?;
        let a = self.in_proj_a.forward(x)?.to_dtype(DType::F32)?;
        let softplus = (a
            .broadcast_add(&self.dt_bias.to_dtype(DType::F32)?)?
            .exp()?
            + 1.0)?
            .log()?;
        let decay_rate = self.a_log.to_dtype(DType::F32)?.exp()?;
        let g = softplus
            .broadcast_mul(&decay_rate)?
            .neg()?
            .transpose(1, 2)?;

        let recurrent = previous
            .map(|state| state.recurrent)
            .unwrap_or(Tensor::zeros(
                (
                    batch,
                    self.num_v_heads,
                    self.key_head_dim,
                    self.value_head_dim,
                ),
                DType::F32,
                x.device(),
            )?);
        let (core, recurrent) = Self::recurrent_scan(&q, &k, &v, &g, &beta, recurrent)?;
        // The recurrent update is intentionally accumulated in f32. Match the
        // reference implementation by restoring the activation dtype before
        // gated normalization and the f16 output projection.
        let core = core
            .transpose(1, 2)?
            .contiguous()?
            .to_dtype(activation_dtype)?;
        let z = self.in_proj_z.forward(x)?.reshape((
            batch,
            seq_len,
            self.num_v_heads,
            self.value_head_dim,
        ))?;
        let core = self
            .norm
            .forward(&core, &z)?
            .reshape((batch, seq_len, value_size))?;
        cache.set_linear_state(block_idx, conv_state, recurrent)?;
        Ok(self.out_proj.forward(&core)?)
    }
}

#[derive(Debug, Clone)]
enum TokenMixer {
    Linear(GatedDeltaNet),
    Full(FullAttention),
}

#[derive(Debug, Clone)]
pub struct NativeTransformer {
    name: String,
    input_norm: ZeroCenteredRmsNorm,
    mixer: TokenMixer,
    post_norm: ZeroCenteredRmsNorm,
    mlp: Mlp,
}

impl std::fmt::Display for NativeTransformer {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{} (Qwen3.8 local)", self.name)
    }
}

fn layer_index(name: &str) -> Result<usize> {
    name.rsplit('.')
        .next()
        .ok_or_else(|| anyhow!("Qwen3.8 layer name has no index: {name}"))?
        .parse()
        .map_err(|error| anyhow!("invalid Qwen3.8 layer name {name}: {error}"))
}

#[async_trait]
impl Forwarder for NativeTransformer {
    fn load(name: String, vb: VarBuilder, cfg: &Config) -> Result<Box<Self>> {
        let text = cfg
            .qwen3_8
            .as_ref()
            .ok_or_else(|| anyhow!("Qwen3.8 layer loaded without Qwen3.8 configuration"))?;
        let index = layer_index(&name)?;
        let layer_type = text
            .layer_types
            .get(index)
            .ok_or_else(|| anyhow!("missing Qwen3.8 layer type for layer {index}"))?;
        let mixer = match layer_type.as_str() {
            "linear_attention" => {
                TokenMixer::Linear(GatedDeltaNet::load(vb.pp("linear_attn"), text)?)
            }
            "full_attention" => {
                TokenMixer::Full(FullAttention::load(vb.pp("self_attn"), text, cfg.attn_f32)?)
            }
            other => anyhow::bail!("unsupported Qwen3.8 layer type {other:?} at layer {index}"),
        };
        Ok(Box::new(Self {
            name,
            input_norm: ZeroCenteredRmsNorm::load(
                text.hidden_size,
                text.rms_norm_eps,
                vb.pp("input_layernorm"),
            )?,
            mixer,
            post_norm: ZeroCenteredRmsNorm::load(
                text.hidden_size,
                text.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
            mlp: Mlp::load(vb.pp("mlp"), text)?,
        }))
    }

    async fn forward(
        &self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        cache: &mut Cache,
    ) -> Result<Tensor> {
        let residual = x;
        let normalized = self.input_norm.forward(x)?;
        let mixed = match &self.mixer {
            TokenMixer::Linear(layer) => layer.forward(&normalized, index_pos, block_idx, cache)?,
            TokenMixer::Full(layer) => layer.forward(&normalized, index_pos, block_idx, cache)?,
        };
        let x = (residual + mixed)?;
        let residual = &x;
        let x = self.mlp.forward(&self.post_norm.forward(&x)?)?;
        Ok((residual + x)?)
    }

    async fn forward_mut(
        &mut self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        cache: &mut Cache,
    ) -> Result<Tensor> {
        self.forward(x, index_pos, block_idx, cache).await
    }

    fn layer_name(&self) -> &str {
        &self.name
    }
}

#[derive(Debug)]
pub enum Transformer {
    Native(Box<NativeTransformer>),
    Ggml {
        name: String,
        index: usize,
        engine: std::sync::Arc<super::ggml::GgmlEngine>,
    },
}
impl std::fmt::Display for Transformer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Native(layer) => layer.fmt(f),
            Self::Ggml { name, .. } => write!(f, "{name} (upstream GGML local)"),
        }
    }
}
#[async_trait]
impl Forwarder for Transformer {
    fn load(name: String, vb: VarBuilder, cfg: &Config) -> Result<Box<Self>> {
        if let Some(engine) = &cfg.qwen3_8_ggml {
            let index = layer_index(&name)?;
            engine.prepare_layer(index)?;
            Ok(Box::new(Self::Ggml {
                name,
                index,
                engine: engine.clone(),
            }))
        } else {
            Ok(Box::new(Self::Native(NativeTransformer::load(
                name, vb, cfg,
            )?)))
        }
    }
    async fn forward(
        &self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        cache: &mut Cache,
    ) -> Result<Tensor> {
        match self {
            Self::Native(layer) => layer.forward(x, index_pos, block_idx, cache).await,
            Self::Ggml { index, engine, .. } => {
                anyhow::ensure!(*index == block_idx, "GGML layer/index mismatch");
                engine.layer(*index, index_pos, x, cache)
            }
        }
    }
    async fn forward_mut(
        &mut self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        cache: &mut Cache,
    ) -> Result<Tensor> {
        self.forward(x, index_pos, block_idx, cache).await
    }
    fn layer_name(&self) -> &str {
        match self {
            Self::Native(layer) => layer.layer_name(),
            Self::Ggml { name, .. } => name,
        }
    }
    fn forward_local_batch(
        &self,
        x: &Tensor,
        batch: &[(String, usize, usize)],
        cache: &mut Cache,
    ) -> Option<Result<Tensor>> {
        let Self::Ggml { index, engine, .. } = self else {
            return None;
        };
        if !engine.fused_shards_enabled() || batch.is_empty() {
            return None;
        }
        let position = batch[0].1;
        // Non-consecutive/mixed-position legacy requests retain their existing
        // execution semantics. Normal DIAL shard batches take this fast path.
        if batch
            .iter()
            .enumerate()
            .any(|(offset, (name, pos, block))| {
                *pos != position
                    || index.checked_add(offset) != Some(*block)
                    || name != &format!("model.language_model.layers.{block}")
            })
        {
            return None;
        }
        Some(engine.range(*index, batch.len(), position, x, cache))
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, sync::Arc};

    use candle_core::Device;

    use super::*;

    fn text_config() -> TextConfig {
        TextConfig {
            hidden_size: 4,
            intermediate_size: 8,
            vocab_size: 16,
            num_hidden_layers: 1,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            head_dim: 4,
            rms_norm_eps: 1e-6,
            max_position_embeddings: 16,
            bos_token_id: None,
            eos_token_id: None,
            layer_types: vec!["linear_attention".to_string()],
            linear_conv_kernel_dim: 2,
            linear_key_head_dim: 2,
            linear_num_key_heads: 1,
            linear_num_value_heads: 1,
            linear_value_head_dim: 2,
            rope_parameters: None,
            partial_rotary_factor: 0.5,
        }
    }

    fn insert_zeros(
        tensors: &mut HashMap<String, Tensor>,
        name: &str,
        shape: impl Into<candle_core::Shape>,
        device: &Device,
    ) {
        tensors.insert(
            name.to_string(),
            Tensor::zeros(shape, DType::F16, device).unwrap(),
        );
    }

    #[tokio::test]
    async fn linear_attention_keeps_recurrent_state_in_dial_cache() -> Result<()> {
        let device = Device::Cpu;
        let text = text_config();
        let prefix = "model.language_model.layers.0";
        let mut tensors = HashMap::new();
        insert_zeros(
            &mut tensors,
            &format!("{prefix}.input_layernorm.weight"),
            4,
            &device,
        );
        insert_zeros(
            &mut tensors,
            &format!("{prefix}.post_attention_layernorm.weight"),
            4,
            &device,
        );
        insert_zeros(
            &mut tensors,
            &format!("{prefix}.mlp.gate_proj.weight"),
            (8, 4),
            &device,
        );
        insert_zeros(
            &mut tensors,
            &format!("{prefix}.mlp.up_proj.weight"),
            (8, 4),
            &device,
        );
        insert_zeros(
            &mut tensors,
            &format!("{prefix}.mlp.down_proj.weight"),
            (4, 8),
            &device,
        );
        insert_zeros(
            &mut tensors,
            &format!("{prefix}.linear_attn.in_proj_qkv.weight"),
            (6, 4),
            &device,
        );
        insert_zeros(
            &mut tensors,
            &format!("{prefix}.linear_attn.in_proj_z.weight"),
            (2, 4),
            &device,
        );
        insert_zeros(
            &mut tensors,
            &format!("{prefix}.linear_attn.in_proj_b.weight"),
            (1, 4),
            &device,
        );
        insert_zeros(
            &mut tensors,
            &format!("{prefix}.linear_attn.in_proj_a.weight"),
            (1, 4),
            &device,
        );
        insert_zeros(
            &mut tensors,
            &format!("{prefix}.linear_attn.conv1d.weight"),
            (6, 1, 2),
            &device,
        );
        insert_zeros(
            &mut tensors,
            &format!("{prefix}.linear_attn.dt_bias"),
            1,
            &device,
        );
        insert_zeros(
            &mut tensors,
            &format!("{prefix}.linear_attn.A_log"),
            1,
            &device,
        );
        insert_zeros(
            &mut tensors,
            &format!("{prefix}.linear_attn.norm.weight"),
            2,
            &device,
        );
        insert_zeros(
            &mut tensors,
            &format!("{prefix}.linear_attn.out_proj.weight"),
            (4, 2),
            &device,
        );

        let config = Config {
            hidden_size: 4,
            intermediate_size: 8,
            vocab_size: 16,
            num_hidden_layers: 1,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            bos_token_id: None,
            eos_token_id: None,
            max_seq_len: 16,
            attn_f32: false,
            qwen3_8: Some(Arc::new(text)),
            qwen3_8_ggml: None,
        };
        let vb = VarBuilder::from_tensors(tensors, DType::F16, &device);
        let layer = Transformer::load(prefix.to_string(), vb.pp(prefix), &config)?;
        let mut cache = Cache::new(true, DType::F16, &config, &device)?;
        let output = layer
            .forward(
                &Tensor::zeros((1, 3, 4), DType::F16, &device)?,
                0,
                0,
                &mut cache,
            )
            .await?;
        assert_eq!(output.dims(), [1, 3, 4]);
        assert_eq!(output.dtype(), DType::F16);
        let state = cache.linear_state(0).expect("linear state");
        assert_eq!(state.conv.dims(), [1, 6, 1]);
        assert_eq!(state.recurrent.dims(), [1, 1, 2, 2]);
        assert!(!state.conv.track_op());
        assert!(!state.recurrent.track_op());
        Ok(())
    }

    #[test]
    fn gated_delta_recurrence_matches_scalar_reference() -> Result<()> {
        let device = Device::Cpu;
        let q = Tensor::from_vec(vec![1.0f32, 1.0], (1, 1, 2, 1), &device)?;
        let k = Tensor::from_vec(vec![1.0f32, 1.0], (1, 1, 2, 1), &device)?;
        let v = Tensor::from_vec(vec![2.0f32, 4.0], (1, 1, 2, 1), &device)?;
        let g = Tensor::from_vec(vec![0.5f32.ln(), 0.5f32.ln()], (1, 1, 2), &device)?;
        let beta = Tensor::from_vec(vec![1.0f32, 0.5], (1, 1, 2), &device)?;
        let state = Tensor::zeros((1, 1, 1, 1), DType::F32, &device)?;
        let (output, state) = GatedDeltaNet::recurrent_scan(&q, &k, &v, &g, &beta, state)?;
        assert_eq!(output.flatten_all()?.to_vec1::<f32>()?, vec![2.0, 2.5]);
        assert_eq!(state.flatten_all()?.to_vec1::<f32>()?, vec![2.5]);
        Ok(())
    }
}
