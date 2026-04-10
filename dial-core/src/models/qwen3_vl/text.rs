//! Qwen3 text (language model) building blocks.

use anyhow::Result;
use async_trait::async_trait;
use candle_core::{DType, Tensor, D};
use candle_nn::{linear_no_bias as linear, Module, RmsNorm, VarBuilder};

use crate::spm::Forwarder;

use crate::models::llama3::{Cache, Config};

#[inline]
fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> candle_core::Result<Tensor> {
    let shape = mask.shape();
    let on_true = Tensor::new(on_true, on_false.device())?.broadcast_as(shape.dims())?;
    let m = mask.where_cond(&on_true, on_false)?;
    Ok(m)
}

#[derive(Debug, Clone)]
pub struct Mlp {
    gate_proj: candle_nn::Linear,
    up_proj: candle_nn::Linear,
    down_proj: candle_nn::Linear,
}

impl Mlp {
    pub fn load(vb: VarBuilder, cfg: &Config) -> candle_core::Result<Self> {
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        Ok(Self {
            gate_proj: linear(h, i, vb.pp("gate_proj"))?,
            up_proj: linear(h, i, vb.pp("up_proj"))?,
            down_proj: linear(i, h, vb.pp("down_proj"))?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let x = (candle_nn::ops::silu(&self.gate_proj.forward(x)?)? * self.up_proj.forward(x)?)?;
        self.down_proj.forward(&x)
    }
}

#[derive(Debug, Clone)]
pub struct QwenAttention {
    q_proj: candle_nn::Linear,
    k_proj: candle_nn::Linear,
    v_proj: candle_nn::Linear,
    o_proj: candle_nn::Linear,

    q_norm: RmsNorm,
    k_norm: RmsNorm,

    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
}

impl QwenAttention {
    fn apply_rotary_emb(&self, x: &Tensor, index_pos: usize, cache: &Cache) -> candle_core::Result<Tensor> {
        let (_batch_size, _, seq_len, _head_dim) = x.dims4()?;
        let cos = cache.cosine(index_pos, seq_len)?;
        let sin = cache.sine(index_pos, seq_len)?;
        candle_nn::rotary_emb::rope(x, &cos, &sin)
    }

    fn repeat_kv(&self, x: Tensor) -> candle_core::Result<Tensor> {
        candle_transformers::utils::repeat_kv(
            x,
            self.num_attention_heads / self.num_key_value_heads,
        )
    }

    pub fn load(vb: VarBuilder, cfg: &Config) -> candle_core::Result<Self> {
        let size_in = cfg.hidden_size;
        let size_q = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_attention_heads;
        let size_kv = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_key_value_heads;

        let head_dim = cfg.hidden_size / cfg.num_attention_heads;

        Ok(Self {
            q_proj: linear(size_in, size_q, vb.pp("q_proj"))?,
            k_proj: linear(size_in, size_kv, vb.pp("k_proj"))?,
            v_proj: linear(size_in, size_kv, vb.pp("v_proj"))?,
            o_proj: linear(size_q, size_in, vb.pp("o_proj"))?,

            q_norm: candle_nn::rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?,
            k_norm: candle_nn::rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?,

            num_attention_heads: cfg.num_attention_heads,
            num_key_value_heads: cfg.num_key_value_heads,
            head_dim,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        cache: &mut Cache,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, hidden_size) = x.dims3().map_err(|e| anyhow!("x.dims3 -> {e}"))?;

        let q = self.q_proj.forward(x).map_err(|e| anyhow!("q.forward -> {e}"))?;
        let k = self.k_proj.forward(x).map_err(|e| anyhow!("k.forward -> {e}"))?;
        let v = self.v_proj.forward(x).map_err(|e| anyhow!("v.forward -> {e}"))?;

        let q = q
            .reshape((b_sz, seq_len, self.num_attention_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // Apply q/k RMSNorm per head-dim (broadcast across batch/head/seq).
        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;

        let q = self.apply_rotary_emb(&q, index_pos, cache).map_err(|e| anyhow!("q.rope -> {e}"))?;
        let k = self.apply_rotary_emb(&k, index_pos, cache).map_err(|e| anyhow!("k.rope -> {e}"))?;

        // （新增）Qwen3-VL 的文本注意力同样复用 llama3::Cache（rope + KV-cache + mask）。
        // 这里传入 index_pos，用于在新请求 prefill 时覆盖旧 kv，避免第二次请求崩溃。
        let (k, v) = cache
            .process_kv(block_idx, index_pos, k, v)
            .map_err(|e| anyhow!("cache.process_kv(block={block_idx}) -> {e}"))?;

        let k = self.repeat_kv(k).map_err(|e| anyhow!("repeat_kv(k) -> {e}"))?;
        let v = self.repeat_kv(v).map_err(|e| anyhow!("repeat_kv(v) -> {e}"))?;

        let y = {
            let in_dtype = q.dtype();
            let q = q.to_dtype(DType::F32)?;
            let k = k.to_dtype(DType::F32)?;
            let v = v.to_dtype(DType::F32)?;
            let att = (q.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?;
            let att = if seq_len == 1 {
                att
            } else {
                let mask = cache
                    .mask(seq_len)
                    .map_err(|e| anyhow!("cache.mask({seq_len}) -> {e}"))?
                    .broadcast_as(att.shape())
                    .map_err(|e| anyhow!("mask.broadcast_as({:?}) -> {e}", att.shape()))?;
                masked_fill(&att, &mask, f32::NEG_INFINITY).map_err(|e| anyhow!("masked_fill -> {e}"))?
            };
            let att = candle_nn::ops::softmax(&att, D::Minus1)?;
            att.matmul(&v.contiguous()?)?.to_dtype(in_dtype)?
        };
        let y = y.transpose(1, 2)?.reshape(&[b_sz, seq_len, hidden_size])?;
        let y = self.o_proj.forward(&y)?;
        Ok(y)
    }
}

#[derive(Debug, Clone)]
pub struct Transformer {
    name: String,
    rms_1: RmsNorm,
    attn: QwenAttention,
    rms_2: RmsNorm,
    mlp: Mlp,
}

impl std::fmt::Display for Transformer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} (local)", &self.name)
    }
}

#[async_trait]
impl Forwarder for Transformer {
    fn load(name: String, vb: VarBuilder, cfg: &Config) -> Result<Box<Self>> {
        let attn = QwenAttention::load(vb.pp("self_attn"), cfg)?;
        let mlp = Mlp::load(vb.pp("mlp"), cfg)?;
        let rms_1 = candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let rms_2 = candle_nn::rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        Ok(Box::new(Self {
            name,
            rms_1,
            attn,
            rms_2,
            mlp,
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
        let x = self.rms_1.forward(x).map_err(|e| anyhow!("rms_1: {e}"))?;
        let x = (self
            .attn
            .forward(&x, index_pos, block_idx, cache)
            .map_err(|e| anyhow!("attention: {e}"))?
            + residual)
            .map_err(|e| anyhow!("residual: {e}"))?;
        let residual = &x;
        let x = self.rms_2.forward(&x).map_err(|e| anyhow!("rms_2: {e}"))?;
        let x = (self.mlp.forward(&x).map_err(|e| anyhow!("mlp: {e}"))? + residual)
            .map_err(|e| anyhow!("mlp residual: {e}"))?;
        Ok(x)
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
