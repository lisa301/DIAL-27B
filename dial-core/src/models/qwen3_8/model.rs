use anyhow::Result;
use async_trait::async_trait;
use candle_core::{DType, IndexOp, Tensor};
use candle_nn::{Embedding, Module};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use std::collections::HashSet;
use std::time::Instant;
use tokenizers::Tokenizer;

use crate::{
    models::{chat::Message, Generator, Token},
    spm::{Context, Forwarder},
};

use super::{
    load_tensor, quantized_linear_no_bias, QuantLinear, Qwen38Config, Transformer,
    ZeroCenteredRmsNorm,
};

const CHAT_EOS_TOKEN: &str = "<|im_end|>";
const TEXT_EOS_TOKEN: &str = "<|endoftext|>";
const GGML_WARMUP_PROMPT: &str = "只回答一个数字：1加1等于多少？";
const XHIGH_REASONING: &str = "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.";

pub struct Qwen38 {
    ctx: Context,
    tokenizer: Tokenizer,
    embedding: Option<Embedding>,
    blocks: Vec<Box<dyn Forwarder>>,
    final_norm: Option<ZeroCenteredRmsNorm>,
    lm_head: Option<QuantLinear>,
    eos_token_ids: HashSet<u32>,
    logits_processor: LogitsProcessor,
    history: Vec<Message>,
    tokens: Vec<u32>,
    index_pos: usize,
    generated: usize,
}

impl Qwen38 {
    fn clear_request_cache(&mut self) {
        if self.ctx.config.qwen3_8_ggml.is_some() {
            // API generation is serialized. Reuse this connection's storage;
            // the first forward at position 0 resets all participating layers.
            self.ctx.cache.reuse_ggml_for_new_request();
        } else {
            self.ctx.cache.clear();
        }
    }

    fn warmup_shape(prompt_tokens: usize, max_seq: usize) -> (usize, usize) {
        let decode_steps = max_seq.saturating_sub(1).min(3);
        (prompt_tokens.min(max_seq - decode_steps), decode_steps)
    }

    /// Allocate and warm the actual local/remote graph paths BEFORE opening the
    /// HTTP API. Do not sample, emit text or modify the user's sampling RNG.
    async fn warmup_ggml(&mut self) -> Result<()> {
        if self.ctx.config.qwen3_8_ggml.is_none()
            || std::env::var("DIAL_GGML_WARMUP").as_deref() == Ok("0")
        {
            return Ok(());
        }
        if !self.ctx.args.system_prompt.is_empty() {
            self.history
                .push(Message::system(self.ctx.args.system_prompt.clone()));
        }
        let prompt = std::env::var("DIAL_GGML_WARMUP_PROMPT").unwrap_or_else(|_| {
            if self.ctx.args.prompt.is_empty() {
                GGML_WARMUP_PROMPT.to_string()
            } else {
                self.ctx.args.prompt.clone()
            }
        });
        self.history.push(Message::user(prompt));
        let ids = self
            .tokenizer
            .encode(self.prompt()?, false)
            .map_err(anyhow::Error::msg)?
            .get_ids()
            .to_vec();
        self.history.clear();
        anyhow::ensure!(!ids.is_empty(), "GGML warmup prompt has no tokens");
        let (prefill_tokens, decode_steps) =
            Self::warmup_shape(ids.len(), self.ctx.config.max_seq_len);
        let start = Instant::now();
        log::info!(
            "Qwen3.8 GGML warmup starting: prefill_tokens={prefill_tokens} decode_steps={decode_steps}; API waits until complete"
        );
        let input = Tensor::new(&ids[..prefill_tokens], &self.ctx.device)?.unsqueeze(0)?;
        // Upstream CUDA graphs need two identical calls before capture. Keep
        // the same buffers and graph shape so subsequent requests can reuse it.
        for _ in 0..2 {
            self.forward(&input, 0)
                .await?
                .flatten_all()?
                .to_vec1::<f32>()?;
        }
        let input = Tensor::new(&[ids[prefill_tokens - 1]], &self.ctx.device)?.unsqueeze(0)?;
        // Warm one-token GEMV, capture the stable decode graph, then replay it.
        for offset in 0..decode_steps {
            self.forward(&input, prefill_tokens + offset)
                .await?
                .flatten_all()?
                .to_vec1::<f32>()?;
        }
        self.reset()?;
        crate::spm::reset_distributed_profile();
        log::info!(
            "Qwen3.8 GGML warmup complete in {:.3}s; request graphs retained, first request resets state at position 0",
            start.elapsed().as_secs_f64()
        );
        Ok(())
    }

    fn extend_eos_from_value(ids: &mut HashSet<u32>, value: &serde_json::Value) {
        match value.get("eos_token_id") {
            Some(serde_json::Value::Number(id)) => {
                if let Some(id) = id.as_u64().and_then(|id| u32::try_from(id).ok()) {
                    ids.insert(id);
                }
            }
            Some(serde_json::Value::Array(values)) => {
                ids.extend(
                    values
                        .iter()
                        .filter_map(|value| value.as_u64().and_then(|id| u32::try_from(id).ok())),
                );
            }
            _ => {}
        }
    }

    fn load_eos_token_ids(
        data_path: &std::path::Path,
        tokenizer: &Tokenizer,
        config_eos: Option<u32>,
    ) -> Result<HashSet<u32>> {
        let mut ids = HashSet::new();
        ids.extend(config_eos);
        ids.extend(tokenizer.token_to_id(CHAT_EOS_TOKEN));
        ids.extend(tokenizer.token_to_id(TEXT_EOS_TOKEN));

        // Transformers checkpoints may store one EOS id or a list in the
        // generation configuration. Qwen3.8 uses both chat-end and
        // end-of-text, so preserve every declared value.
        let generation_path = data_path.join("generation_config.json");
        if generation_path.exists() {
            let raw = std::fs::read(&generation_path)
                .map_err(|error| anyhow!("can't read {}: {error}", generation_path.display()))?;
            let value: serde_json::Value = serde_json::from_slice(&raw)
                .map_err(|error| anyhow!("can't parse {}: {error}", generation_path.display()))?;
            Self::extend_eos_from_value(&mut ids, &value);
        }
        if ids.is_empty() {
            anyhow::bail!("Qwen3.8 checkpoint does not define a usable EOS token");
        }
        Ok(ids)
    }

    fn create_logits_processor(ctx: &Context) -> LogitsProcessor {
        let sampling = if ctx.args.temperature <= 0.0 {
            Sampling::ArgMax
        } else {
            match (ctx.args.top_k, ctx.args.top_p) {
                (None, None) => Sampling::All {
                    temperature: ctx.args.temperature,
                },
                (Some(k), None) => Sampling::TopK {
                    k,
                    temperature: ctx.args.temperature,
                },
                (None, Some(p)) => Sampling::TopP {
                    p,
                    temperature: ctx.args.temperature,
                },
                (Some(k), Some(p)) => Sampling::TopKThenTopP {
                    k,
                    p,
                    temperature: ctx.args.temperature,
                },
            }
        };
        LogitsProcessor::from_sampling(ctx.args.seed, sampling)
    }

    fn prompt(&self) -> Result<String> {
        if self.history.is_empty() {
            anyhow::bail!("Qwen3.8 chat requires at least one message");
        }
        let mut prompt = String::new();
        let thinking = self.ctx.args.qwen38_thinking;
        let first_is_system = matches!(
            self.history.first().map(|message| &message.role),
            Some(crate::models::chat::MessageRole::System)
        );
        if thinking && !first_is_system {
            prompt.push_str("<|im_start|>system\n");
            prompt.push_str(XHIGH_REASONING);
            prompt.push_str("<|im_end|>\n");
        }
        for (index, message) in self.history.iter().enumerate() {
            let content = message.content.to_text_strict().map_err(|_| {
                anyhow!(
                    "native Qwen3.8 text path does not yet accept image/video parts; use text messages while the Qwen3.8 vision encoder is being integrated"
                )
            })?;
            prompt.push_str("<|im_start|>");
            prompt.push_str(&message.role.to_string());
            prompt.push('\n');
            if index == 0
                && thinking
                && matches!(message.role, crate::models::chat::MessageRole::System)
            {
                prompt.push_str(XHIGH_REASONING);
                prompt.push_str("\n\n");
            }
            if matches!(message.role, crate::models::chat::MessageRole::Assistant) {
                prompt.push_str("<think>\n\n</think>\n\n");
            }
            prompt.push_str(content.trim());
            prompt.push_str("<|im_end|>\n");
        }
        prompt.push_str("<|im_start|>assistant\n");
        if thinking {
            prompt.push_str("<think>\n");
        } else {
            prompt.push_str("<think>\n\n</think>\n\n");
        }
        Ok(prompt)
    }

    fn start_dialog(&mut self) -> Result<()> {
        self.clear_request_cache();
        self.index_pos = 0;
        self.tokens = self
            .tokenizer
            .encode(self.prompt()?, false)
            .map_err(anyhow::Error::msg)?
            .get_ids()
            .to_vec();
        if self.tokens.is_empty() {
            anyhow::bail!("Qwen3.8 tokenizer produced an empty prompt");
        }
        if self.tokens.len() > self.ctx.config.max_seq_len {
            anyhow::bail!(
                "Qwen3.8 prompt has {} tokens, exceeding configured limit {}",
                self.tokens.len(),
                self.ctx.config.max_seq_len
            );
        }
        Ok(())
    }

    async fn forward_blocks(&mut self, mut x: Tensor, index_pos: usize) -> Result<Tensor> {
        let mut block_idx = 0usize;
        while block_idx < self.blocks.len() {
            let ident = self.blocks[block_idx].ident().to_string();
            if ident == "local" {
                if let Some(engine) = self
                    .ctx
                    .config
                    .qwen3_8_ggml
                    .clone()
                    .filter(|e| e.fused_shards_enabled())
                {
                    let first = block_idx;
                    while block_idx < self.blocks.len() && self.blocks[block_idx].ident() == "local"
                    {
                        block_idx += 1;
                    }
                    let shard_start = Instant::now();
                    x = engine
                        .range(first, block_idx - first, index_pos, &x, &mut self.ctx.cache)
                        .map_err(|error| {
                            anyhow!(
                                "Qwen3.8 local shard {}..{} failed: {error}",
                                first,
                                block_idx - 1
                            )
                        })?;
                    if shard_start.elapsed().as_secs_f64() >= 1.0 {
                        log::warn!(
                            "[ggml latency] master_local_layers={}..{} position={index_pos} total_s={:.3}",
                            first, block_idx - 1, shard_start.elapsed().as_secs_f64()
                        );
                    }
                    continue;
                }
                x = self.blocks[block_idx]
                    .forward_mut(&x, index_pos, block_idx, &mut self.ctx.cache)
                    .await
                    .map_err(|error| anyhow!("Qwen3.8 local layer {block_idx} failed: {error}"))?;
                block_idx += 1;
            } else {
                let first = block_idx;
                let mut batch = Vec::new();
                while block_idx < self.blocks.len() && self.blocks[block_idx].ident() == ident {
                    batch.push((
                        self.blocks[block_idx].layer_name().to_string(),
                        index_pos,
                        block_idx,
                    ));
                    block_idx += 1;
                }
                x = self.blocks[first]
                    .forward_batch(&x, batch, &mut self.ctx.cache)
                    .await
                    .map_err(|error| {
                        anyhow!(
                            "Qwen3.8 remote layers {}..{} failed: {error}",
                            first,
                            block_idx.saturating_sub(1)
                        )
                    })?;
            }
        }
        Ok(x)
    }

    async fn forward(&mut self, input: &Tensor, index_pos: usize) -> Result<Tensor> {
        let start = Instant::now();
        let (_, seq_len) = input.dims2()?;
        let ggml = self.ctx.config.qwen3_8_ggml.clone();
        let x = if let Some(engine) = &ggml {
            engine.embedding(input)?.to_dtype(self.ctx.dtype)?
        } else {
            self.embedding
                .as_ref()
                .ok_or_else(|| anyhow!("missing native embedding"))?
                .forward(input)?
        };
        let embedding_s = start.elapsed().as_secs_f64();
        let blocks_start = Instant::now();
        let x = self.forward_blocks(x, index_pos).await?;
        let blocks_s = blocks_start.elapsed().as_secs_f64();
        let head_start = Instant::now();
        let logits = if let Some(engine) = &ggml {
            engine.logits(&x.i((.., seq_len - 1, ..))?.contiguous()?)
        } else {
            let x = self
                .final_norm
                .as_ref()
                .ok_or_else(|| anyhow!("missing native norm"))?
                .forward(&x)?;
            let x = x.i((.., seq_len - 1, ..))?.contiguous()?;
            Ok(self
                .lm_head
                .as_ref()
                .ok_or_else(|| anyhow!("missing native output head"))?
                .forward(&x)?
                .to_dtype(DType::F32)?)
        }?;
        if ggml.is_some() && start.elapsed().as_secs_f64() >= 1.0 {
            log::warn!(
                "[ggml latency] phase={} tokens={seq_len} position={index_pos} embedding_s={embedding_s:.3} blocks_s={blocks_s:.3} head_s={:.3} total_s={:.3}",
                if seq_len > 1 { "prefill" } else { "decode" },
                head_start.elapsed().as_secs_f64(), start.elapsed().as_secs_f64()
            );
        }
        Ok(logits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ggml_warmup_leaves_room_for_decode_without_exceeding_context() {
        assert_eq!(Qwen38::warmup_shape(35, 4096), (35, 3));
        assert_eq!(Qwen38::warmup_shape(35, 32), (29, 3));
        assert_eq!(Qwen38::warmup_shape(35, 2), (1, 1));
        assert_eq!(Qwen38::warmup_shape(35, 1), (1, 0));
    }

    #[test]
    fn qwen38_accepts_all_generation_eos_ids() {
        let mut ids = HashSet::new();
        Qwen38::extend_eos_from_value(
            &mut ids,
            &serde_json::json!({"eos_token_id": [248046, 248044]}),
        );
        assert!(ids.contains(&248046));
        assert!(ids.contains(&248044));
    }

    #[test]
    fn qwen38_accepts_scalar_generation_eos_id() {
        let mut ids = HashSet::new();
        Qwen38::extend_eos_from_value(&mut ids, &serde_json::json!({"eos_token_id": 248046}));
        assert_eq!(ids, HashSet::from([248046]));
    }
}

#[async_trait]
impl Generator for Qwen38 {
    type Shardable = Transformer;

    const MODEL_NAME: &'static str = "qwen3.8-27b";

    async fn load(ctx: Context) -> Result<Box<Self>> {
        let full_config = Qwen38Config::from_path(&ctx.data_path.join("config.json"))?;
        if full_config.is_quantized() && ctx.config.qwen3_8_ggml.is_none() {
            log::info!(
                "Qwen3.8 quantized checkpoint detected; DIAL will decode compressed-tensors/ModelOpt NVFP4 and FP8 weights to {:?} while loading this node's assigned layers",
                ctx.dtype
            );
        }
        let text = &full_config.text_config;
        let tokenizer = Tokenizer::from_file(ctx.data_path.join("tokenizer.json"))
            .map_err(anyhow::Error::msg)?;
        let (embedding, final_norm, lm_head) = if let Some(engine) = &ctx.config.qwen3_8_ggml {
            engine.prepare_head()?;
            (None, None, None)
        } else {
            log::info!("loading Qwen3.8 embedding ...");
            let embedding_weight = load_tensor(
                ctx.var_builder.pp("model.language_model.embed_tokens"),
                (text.vocab_size, text.hidden_size),
                "weight",
            )
            .map_err(|error| anyhow!("failed to load Qwen3.8 embedding: {error}"))?;
            let embedding = Embedding::new(embedding_weight, text.hidden_size);
            log::info!("loading Qwen3.8 final norm ...");
            let final_norm = ZeroCenteredRmsNorm::load(
                text.hidden_size,
                text.rms_norm_eps,
                ctx.var_builder.pp("model.language_model.norm"),
            )
            .map_err(|error| anyhow!("failed to load Qwen3.8 final norm: {error}"))?;
            log::info!("loading Qwen3.8 lm_head ...");
            let lm_head = quantized_linear_no_bias(
                text.hidden_size,
                text.vocab_size,
                ctx.var_builder.pp("lm_head"),
            )
            .map_err(|error| anyhow!("failed to load Qwen3.8 lm_head: {error}"))?;
            (Some(embedding), Some(final_norm), Some(lm_head))
        };

        let mut blocks: Vec<Box<dyn Forwarder>> = Vec::with_capacity(text.num_hidden_layers);
        let mut worker_connections = crate::spm::ClientPool::default();
        for index in 0..text.num_hidden_layers {
            let name = format!("model.language_model.layers.{index}");
            if let Some((node_name, node)) = ctx.topology.get_node_for_layer(&name) {
                log::info!("Qwen3.8 layer {index} -> {node_name}@{}", node.host);
                blocks.push(Box::new(
                    worker_connections
                        .client_for_layer(ctx.device.clone(), &node.host, &name)
                        .await?,
                ));
            } else {
                blocks.push(Transformer::load(
                    name.clone(),
                    ctx.var_builder.pp(&name),
                    &ctx.config,
                )?);
            }
        }
        let eos_token_ids =
            Self::load_eos_token_ids(&ctx.data_path, &tokenizer, text.eos_token_id)?;
        log::info!("Qwen3.8 EOS token ids: {:?}", eos_token_ids);
        let logits_processor = Self::create_logits_processor(&ctx);
        log::info!(
            "Qwen3.8 DIAL model loaded: layers={} hidden={} local/remote topology active",
            text.num_hidden_layers,
            text.hidden_size
        );
        if let Some(engine) = &ctx.config.qwen3_8_ggml {
            let (tensors, bytes) = engine.summary();
            log::info!("Qwen3.8 upstream GGML: resident_tensors={tensors} resident_weights={:.1} MiB (quantized storage; state/workspace excluded)", bytes as f64 / 1048576.0);
        }
        if let Some((linears, bytes)) = super::quant_linear_summary() {
            log::info!(
                "Qwen3.8 CUDA QuantLinear summary: linears={} resident_weights={:.1} MiB",
                linears,
                bytes as f64 / 1048576.0
            );
        }
        let mut model = Self {
            ctx,
            tokenizer,
            embedding,
            blocks,
            final_norm,
            lm_head,
            eos_token_ids,
            logits_processor,
            history: Vec::new(),
            tokens: Vec::new(),
            index_pos: 0,
            generated: 0,
        };
        model
            .warmup_ggml()
            .await
            .map_err(|error| anyhow!("Qwen3.8 GGML startup warmup failed: {error}"))?;
        Ok(Box::new(model))
    }

    fn add_message(&mut self, message: Message) -> Result<()> {
        if message.is_multimodal() {
            anyhow::bail!("native Qwen3.8 vision input is not implemented yet");
        }
        self.history.push(message);
        Ok(())
    }

    fn reset(&mut self) -> Result<()> {
        self.history.clear();
        self.tokens.clear();
        self.clear_request_cache();
        self.index_pos = 0;
        self.generated = 0;
        Ok(())
    }

    async fn next_token(&mut self, index: usize) -> Result<Token> {
        if self.generated == 0 {
            self.start_dialog()?;
        }
        let num_tokens = self.tokens.len();
        let (context_size, context_index) = if self.ctx.cache.with_kv_cache() && index > 0 {
            (1, self.index_pos)
        } else {
            (num_tokens, 0)
        };
        let context = &self.tokens[num_tokens.saturating_sub(context_size)..];
        let consumed = context.len();
        let input = Tensor::new(context, &self.ctx.device)?.unsqueeze(0)?;
        let logits = self.forward(&input, context_index).await?.squeeze(0)?;
        let logits = if self.ctx.args.repeat_penalty == 1.0 {
            logits
        } else {
            let start = num_tokens.saturating_sub(self.ctx.args.repeat_last_n);
            candle_transformers::utils::apply_repeat_penalty(
                &logits,
                self.ctx.args.repeat_penalty,
                &self.tokens[start..],
            )?
        };
        self.index_pos += consumed;
        let next = self.logits_processor.sample(&logits)?;
        self.tokens.push(next);
        self.generated += 1;
        let is_end_of_stream = self.eos_token_ids.contains(&next);
        // Never expose the stop marker as assistant content, even to a caller
        // that renders Token.text before checking is_end_of_stream.
        let text = if is_end_of_stream {
            None
        } else {
            self.tokenizer.decode(&[next], false).ok()
        };
        Ok(Token {
            id: next,
            text,
            is_end_of_stream,
        })
    }

    fn generated_tokens(&self) -> usize {
        self.generated
    }
}
