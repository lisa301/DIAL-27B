use anyhow::Result;
use std::path::Path;
use async_trait::async_trait;
use candle_core::{DType, IndexOp, Tensor};
use candle_nn::{linear_no_bias as linear, Embedding, Linear, Module, RmsNorm};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use tokenizers::Tokenizer;

use crate::{
    models::llama3::Config,
    models::{chat::Message, Generator, Token},
    spm::{Context, Forwarder},
};

use super::{
    image_to_tensor, ImageSpan, PromptEncoder, Qwen3VlConfig, Transformer, VisionConfig,
    TextRknnRunner, VisionEncoder, VisionOutputs, VisionRknn,
};

/// 定义一个字符串常量结束标志符.
const DEFAULT_EOS_TOKEN: &str = "<|im_end|>";
/// 从模型目录里把 tokenizer 文件加载出来
fn load_tokenizer(ctx: &Context) -> Result<Tokenizer> {
    let tokenizer_filename = ctx.data_path.join("tokenizer.json");
    log::info!("loading tokenizer from {}", tokenizer_filename.display());
    Tokenizer::from_file(tokenizer_filename).map_err(anyhow::Error::msg)
}
/// 把用户传入的采样参数翻译成实际可执行的采样策略
fn create_logits_processor(ctx: &Context) -> LogitsProcessor {
    let temperature = ctx.args.temperature;
    let sampling = if temperature <= 0. {
        Sampling::ArgMax
    } else {
        match (ctx.args.top_k, ctx.args.top_p) {
            (None, None) => Sampling::All { temperature },
            (Some(k), None) => Sampling::TopK { k, temperature },
            (None, Some(p)) => Sampling::TopP { p, temperature },
            (Some(k), Some(p)) => Sampling::TopKThenTopP { k, p, temperature },
        }
    };
    LogitsProcessor::from_sampling(ctx.args.seed, sampling)
}
/// 成功时返回图片的二进制字节数组，失败时返回错误
fn decode_base64_image(data: &str) -> Result<Vec<u8>> {
    // Accept raw base64 or data URLs: data:image/png;base64,....
    let payload = if let Some(idx) = data.find("base64,") {
        &data[idx + "base64,".len()..]
    } else {
        data
    };
    use base64::Engine;
    base64::engine::general_purpose::STANDARD
        .decode(payload.trim())
        .map_err(|e| anyhow!("invalid base64 image: {e}"))
}
/// 把已经解码出来的图片二进制数据加载成 RGB 图像，并返回原始像素数组、高度和宽度
fn load_image_rgb(bytes: &[u8]) -> Result<(Vec<u8>, usize, usize)> {
    let img = image::load_from_memory(bytes).map_err(|e| anyhow!("image decode failed: {e}"))?;
    let rgb = img.to_rgb8();
    let (w, h) = rgb.dimensions();
    Ok((rgb.into_raw(), h as usize, w as usize))
}


/// 把任意尺寸的 RGB 图像，缩放到符合模型输入要求的标准尺寸
fn smart_resize_like_pipeline(
    rgb: Vec<u8>,
    h: usize,
    w: usize,
    patch_size: usize,
    spatial_merge_size: usize,
    num_position_embeddings: usize,
    max_side_override: Option<u32>,
    allow_upscale: bool,
) -> Result<(Vec<u8>, usize, usize)> {
    let img =
        image::RgbImage::from_raw(w as u32, h as u32, rgb).ok_or_else(|| anyhow!("invalid rgb buffer"))?;

    // 这里用 factor=patch_size*spatial_merge_size（默认 32）保证：
    // - 像素尺寸能整除 patch_size（生成 patch grid）
    // - patch grid 能整除 spatial_merge_size（避免 encode() 再次裁剪）
    let factor = (patch_size * spatial_merge_size).max(1) as u32;

    // pipeline.py 里 image 的 min_pixels 传的是 4*32*32（即 4096），这里按同样思路取默认最小像素数。
    let min_pixels: u64 = 4u64 * factor as u64 * factor as u64;

    // 最大像素数由位置嵌入容量决定：最多 num_position_embeddings 个 patch token，
    // 每个 patch 对应 patch_size^2 像素。
    let max_pixels: u64 = num_position_embeddings as u64 * patch_size as u64 * patch_size as u64;

    let oh = h as u64;
    let ow = w as u64;
    let op = oh * ow;

    // 计算缩放比例，使得总像素落在 [min_pixels, max_pixels] 之间（保持宽高比）。
    let mut scale = 1.0f64;
    if op > max_pixels {
        scale = (max_pixels as f64 / op as f64).sqrt();
    } else if op < min_pixels && allow_upscale {
        scale = (min_pixels as f64 / op as f64).sqrt();
    }

    let mut th = ((h as f64) * scale).round().max(factor as f64) as u32;
    let mut tw = ((w as f64) * scale).round().max(factor as f64) as u32;

    // 额外约束：把 patch grid 的长宽都限制在 base_grid(48) 以内。
    // 原因：当前 Rust 版 VisionEncoder 的位置编码是固定表（48x48），不做插值。
    // 如果出现 hp>48 或 wp>48，会导致位置编码与网格不匹配，模型容易“看不懂图”。
    let base_side = (num_position_embeddings as f64).sqrt() as u32;
    if base_side * base_side != num_position_embeddings as u32 {
        bail!("num_position_embeddings {} is not a square", num_position_embeddings);
    }
    let default_max_side_pixels = base_side * patch_size as u32;
    let max_side_pixels = max_side_override
        .filter(|v| *v > 0)
        .map(|v| v.min(default_max_side_pixels))
        .unwrap_or(default_max_side_pixels);
    let max_dim = th.max(tw);
    if max_dim > max_side_pixels {
        let s = max_side_pixels as f64 / max_dim as f64;
        th = ((th as f64) * s).floor().max(factor as f64) as u32;
        tw = ((tw as f64) * s).floor().max(factor as f64) as u32;
    }

    // 对齐到 factor 的整数倍（不做 crop，而是直接 resize 到对齐后的尺寸）。
    th = (th / factor).max(1) * factor;
    tw = (tw / factor).max(1) * factor;

    // 再次确认不会超过 max_pixels：如果超过则向下对齐一档。
    let capped_max_pixels = (max_side_pixels as u64) * (max_side_pixels as u64);
    let max_pixels = max_pixels.min(capped_max_pixels);
    while (th as u64) * (tw as u64) > max_pixels && th > factor && tw > factor {
        th -= factor;
        tw -= factor;
    }

    let resized = image::imageops::resize(
        &img,
        tw.max(1),
        th.max(1),
        // 比 Triangle(双线性) 保细节一些，更适合图文场景（类似 bicubic）。
        image::imageops::FilterType::CatmullRom,
    );
    let (rw, rh) = resized.dimensions();
    Ok((resized.into_raw(), rh as usize, rw as usize))
}

/// Resize to a fixed square side length (static shape), keeping aspect ratio with letterbox padding.
///
/// This is intended for hardware backends requiring static input shapes (e.g. BM1684 bmodel).
/// Padding uses mid-gray (128) so that after `(x/255 - 0.5)/0.5` it is ~0.
/// 把任意尺寸的长方形图像 → 缩放到固定大小的正方形，保持原图比例不变形，多余区域用灰色居中填充
fn resize_to_fixed_square_letterbox(
    rgb: Vec<u8>,
    h: usize,
    w: usize,
    patch_size: usize,
    spatial_merge_size: usize,
    num_position_embeddings: usize,
    fixed_side: u32,
    allow_upscale: bool,
) -> Result<(Vec<u8>, usize, usize)> {
    let img =
        image::RgbImage::from_raw(w as u32, h as u32, rgb).ok_or_else(|| anyhow!("invalid rgb buffer"))?;

    // Align side to the grid factor so patch/merge grids are exact and encode() won't crop.
    let factor = (patch_size * spatial_merge_size).max(1) as u32;

    // Cap by the learned position embedding table capacity (no interpolation in this implementation).
    let base_side = (num_position_embeddings as f64).sqrt() as u32;
    if base_side * base_side != num_position_embeddings as u32 {
        bail!("num_position_embeddings {} is not a square", num_position_embeddings);
    }
    let max_side_pixels = base_side * patch_size as u32;

    let mut side = fixed_side.min(max_side_pixels).max(factor);
    side = (side / factor).max(1) * factor;

    // Fit into the square canvas.
    let max_dim = (h.max(w) as f64).max(1.0);
    let mut scale = side as f64 / max_dim;
    if !allow_upscale {
        scale = scale.min(1.0);
    }
    let new_h = ((h as f64) * scale).round().max(1.0) as u32;
    let new_w = ((w as f64) * scale).round().max(1.0) as u32;

    let resized = image::imageops::resize(
        &img,
        new_w.min(side).max(1),
        new_h.min(side).max(1),
        image::imageops::FilterType::CatmullRom,
    );
    let (rw, rh) = resized.dimensions();

    // Letterbox pad to square.
    let pad = image::Rgb([128u8, 128u8, 128u8]);
    let mut canvas = image::RgbImage::from_pixel(side, side, pad);
    let x0 = ((side - rw) / 2) as i64;
    let y0 = ((side - rh) / 2) as i64;
    image::imageops::overlay(&mut canvas, &resized, x0, y0);

    Ok((canvas.into_raw(), side as usize, side as usize))
}

/// Qwen3-VL 多模态大模型（文本 + 图像） 的核心数据结构定义.
pub struct Qwen3Vl {
    ctx: Context,

    // Language model pieces.
    tokenizer: Tokenizer,
    embedding: Embedding,
    ln_f: RmsNorm,
    lm_head: Linear,
    blocks: Vec<Box<dyn Forwarder>>,
    text_rknn: Option<TextRknnRunner>,

    // Vision model.
    vision_cfg: VisionConfig,
    vision: Option<VisionEncoder>,
    vision_rknn: Option<VisionRknn>,
    vision_rknn_side: Option<u32>,
    prompt_encoder: PromptEncoder,

    // Special tokens.
    image_token_id: u32,
    vision_start_token_id: u32,
    vision_end_token_id: u32,
    eos_token_id: Option<u32>,

    logits_processor: LogitsProcessor,

    // Chat state.
    history: Vec<Message>,
    tokens: Vec<u32>,
    image_spans: Vec<(ImageSpan, Tensor)>,
    deepstack_spans: Vec<Vec<(ImageSpan, Tensor)>>,

    index_pos: usize,
    generated: usize,
}

impl Qwen3Vl {
    async fn forward_embeds(&mut self, mut x: Tensor, idx: usize, inject_images: bool) -> Result<Tensor> {
        let (_batch_size, seq_len, _hidden) = x.dims3()?;

        if inject_images && !self.image_spans.is_empty() {
            x = self.inject_image_embeddings(&x)?;
        }

        let use_text_rknn_decode = self.text_rknn.is_some()
            && seq_len == 1
            && idx > 0
            && !inject_images
            && self.ctx.cache.with_kv_cache();
        if use_text_rknn_decode {
            let runner = self
                .text_rknn
                .as_mut()
                .ok_or_else(|| anyhow!("text rknn runner not initialized"))?;
            x = runner.forward_decode(&x, idx, &self.ctx.cache, &self.ctx.device, self.ctx.dtype)?;
        } else {
            let num_blocks = self.blocks.len();
            let mut block_idx = 0;

            while block_idx < num_blocks {
                let curr_block_id = self.blocks[block_idx].ident().to_owned();
                if inject_images {
                    x = self.inject_deepstack_embeddings(&x, block_idx)?;
                }
                if curr_block_id == "local" {
                    x = self.blocks[block_idx]
                        .forward_mut(&x, idx, block_idx, &mut self.ctx.cache)
                        .await
                        .map_err(|e| anyhow!("error in forward operation of local block {block_idx}: {e}"))?;
                    block_idx += 1;
                } else {
                    let mut batch = vec![];
                    let first = block_idx;
                    while block_idx < num_blocks && self.blocks[block_idx].ident() == curr_block_id {
                        batch.push((self.blocks[block_idx].layer_name().to_string(), idx, block_idx));
                        block_idx += 1;
                    }
                    x = self.blocks[first]
                        .forward_batch(&x, batch, &mut self.ctx.cache)
                        .await
                        .map_err(|e| anyhow!("error in forward batch operation for block {block_idx}: {e}"))?;
                }
            }
        }

        let x = self.ln_f.forward(&x).map_err(|e| anyhow!("ln_f.forward: {e}"))?;
        let x = x
            .i((.., seq_len - 1, ..))
            .map_err(|e| anyhow!("x.i: {e}"))?
            .contiguous()
            .map_err(|e| anyhow!("x.i.contiguous: {e}"))?;
        let logits = self.lm_head.forward(&x).map_err(|e| anyhow!("lm_head.forward: {e}"))?;
        logits.to_dtype(DType::F32).map_err(|e| anyhow!("logits.to_dtype: {e}"))
    }

    fn inject_image_embeddings(&self, x: &Tensor) -> Result<Tensor> {
        // x: (1, seq, hidden)
        let (_b, seq, _h) = x.dims3()?;
        let mut out = x.clone();
        // Apply spans in order (they are non-overlapping).
        // We rebuild the tensor with concatenation to "replace" the span.
        for (span, embeds) in &self.image_spans {
            let n = span.end - span.start;
            let embeds = embeds.unsqueeze(0)?; // (1, n, hidden)
            if embeds.dims3()?.1 != n {
                bail!("image embeds length mismatch, span {}..{} expects {}", span.start, span.end, n);
            }
            let before = out.narrow(1, 0, span.start)?;
            let after = out.narrow(1, span.end, seq - span.end)?;
            out = Tensor::cat(&[&before, &embeds, &after], 1)?;
        }
        Ok(out)
    }

    fn inject_deepstack_embeddings(&self, x: &Tensor, layer_idx: usize) -> Result<Tensor> {
        let Some(spans) = self.deepstack_spans.get(layer_idx) else {
            return Ok(x.clone());
        };
        if spans.is_empty() {
            return Ok(x.clone());
        }

        let (_b, seq, _h) = x.dims3()?;
        let mut out = x.clone();
        for (span, embeds) in spans {
            let n = span.end - span.start;
            let embeds = embeds.unsqueeze(0)?; // (1, n, hidden)
            if embeds.dims3()?.1 != n {
                bail!(
                    "deepstack embeds length mismatch for layer {} span {}..{} expects {}",
                    layer_idx,
                    span.start,
                    span.end,
                    n
                );
            }
            let before = out.narrow(1, 0, span.start)?;
            let local = out.narrow(1, span.start, n)?;
            let local = (local + &embeds)?;
            let after = out.narrow(1, span.end, seq - span.end)?;
            out = Tensor::cat(&[&before, &local, &after], 1)?;
        }
        Ok(out)
    }

    fn start_dialog_prompt(&mut self) -> Result<()> {
        self.tokens.clear();
        self.image_spans.clear();
        self.deepstack_spans.clear();
        self.ctx.cache.clear();
        if let Some(runner) = self.text_rknn.as_mut() {
            runner.clear();
        }
        self.index_pos = 0;

        let fixed_side = self.vision_rknn_side.or(self.ctx.args.vision_fixed_side);

        // 1) Encode images in-order, collecting their token counts.
        let mut image_embeds: Vec<Tensor> = vec![];
        let mut image_deepstack_embeds: Vec<Vec<Tensor>> = vec![];
        let mut image_token_counts: Vec<usize> = vec![];

        for msg in &self.history {
            match &msg.content {
                crate::models::chat::MessageContent::Text(_) => {}
                crate::models::chat::MessageContent::Parts(parts) => {
                    for part in parts {
                        match part {
                            crate::models::chat::ContentPart::Text { .. } => {}
                            crate::models::chat::ContentPart::ImageBase64 { data, .. } => {
                                let bytes = decode_base64_image(data)?;
                                let (rgb, h, w) = load_image_rgb(&bytes)?;
                                let (rgb, h, w) = if let Some(side) = fixed_side
                                {
                                    resize_to_fixed_square_letterbox(
                                        rgb,
                                        h,
                                        w,
                                        self.vision_cfg.patch_size,
                                        self.vision_cfg.spatial_merge_size,
                                        self.vision_cfg.num_position_embeddings,
                                        side,
                                        !self.ctx.args.vision_no_upscale,
                                    )?
                                } else {
                                    // （重要）按 pipeline.py 的思路做“网格对齐 + 像素上下限”缩放，尽量避免 crop 造成的信息丢失。
                                    smart_resize_like_pipeline(
                                        rgb,
                                        h,
                                        w,
                                        self.vision_cfg.patch_size,
                                        self.vision_cfg.spatial_merge_size,
                                        self.vision_cfg.num_position_embeddings,
                                        self.ctx.args.vision_max_side,
                                        !self.ctx.args.vision_no_upscale,
                                    )?
                                };
                                let img = image_to_tensor(&rgb, h, w, &self.ctx.device)?
                                    .to_dtype(self.ctx.dtype)?;
                                let VisionOutputs { embeds, deepstack } = if let Some(rknn) = &self.vision_rknn {
                                    rknn.encode(&img, &self.ctx.device, self.ctx.dtype)?
                                } else {
                                    self.vision
                                        .as_ref()
                                        .ok_or_else(|| anyhow!("vision encoder is not loaded"))?
                                        .encode(&img)?
                                };
                                let n = embeds.dims3()?.1;
                                image_token_counts.push(n);
                                image_embeds.push(embeds.squeeze(0)?); // (n, hidden)
                                let mut deepstack_embeds = Vec::with_capacity(deepstack.len());
                                for d in deepstack {
                                    deepstack_embeds.push(d.squeeze(0)?);
                                }
                                image_deepstack_embeds.push(deepstack_embeds);
                            }
                            crate::models::chat::ContentPart::ImageUrl { image_url } => {
                                // Support data URLs only (no network fetch).
                                if !image_url.url.starts_with("data:") {
                                    bail!("image_url is only supported as data URLs in this build");
                                }
                                let bytes = decode_base64_image(&image_url.url)?;
                                let (rgb, h, w) = load_image_rgb(&bytes)?;
                                let (rgb, h, w) = if let Some(side) = fixed_side
                                {
                                    resize_to_fixed_square_letterbox(
                                        rgb,
                                        h,
                                        w,
                                        self.vision_cfg.patch_size,
                                        self.vision_cfg.spatial_merge_size,
                                        self.vision_cfg.num_position_embeddings,
                                        side,
                                        !self.ctx.args.vision_no_upscale,
                                    )?
                                } else {
                                    smart_resize_like_pipeline(
                                        rgb,
                                        h,
                                        w,
                                        self.vision_cfg.patch_size,
                                        self.vision_cfg.spatial_merge_size,
                                        self.vision_cfg.num_position_embeddings,
                                        self.ctx.args.vision_max_side,
                                        !self.ctx.args.vision_no_upscale,
                                    )?
                                };
                                let img = image_to_tensor(&rgb, h, w, &self.ctx.device)?
                                    .to_dtype(self.ctx.dtype)?;
                                let VisionOutputs { embeds, deepstack } = if let Some(rknn) = &self.vision_rknn {
                                    rknn.encode(&img, &self.ctx.device, self.ctx.dtype)?
                                } else {
                                    self.vision
                                        .as_ref()
                                        .ok_or_else(|| anyhow!("vision encoder is not loaded"))?
                                        .encode(&img)?
                                };
                                let n = embeds.dims3()?.1;
                                image_token_counts.push(n);
                                image_embeds.push(embeds.squeeze(0)?);
                                let mut deepstack_embeds = Vec::with_capacity(deepstack.len());
                                for d in deepstack {
                                    deepstack_embeds.push(d.squeeze(0)?);
                                }
                                image_deepstack_embeds.push(deepstack_embeds);
                            }
                        }
                    }
                }
            }
        }

        // 2) Encode text + vision placeholders into token ids.
        let (ids, spans) = self
            .prompt_encoder
            .encode(&self.tokenizer, &self.history, &image_token_counts)?;
        self.tokens = ids;

        // 3) Map spans to embeds.
        if spans.len() != image_embeds.len() {
            bail!(
                "internal error: spans {} != image_embeds {}",
                spans.len(),
                image_embeds.len()
            );
        }
        self.image_spans = spans.clone().into_iter().zip(image_embeds).collect();

        let max_deepstack = image_deepstack_embeds.iter().map(Vec::len).max().unwrap_or(0);
        self.deepstack_spans = vec![vec![]; max_deepstack];
        for (span, deepstack_embeds) in spans.into_iter().zip(image_deepstack_embeds) {
            for (layer_idx, embeds) in deepstack_embeds.into_iter().enumerate() {
                self.deepstack_spans[layer_idx].push((span.clone(), embeds));
            }
        }

        Ok(())
    }
}

#[async_trait]
impl Generator for Qwen3Vl {
    type Shardable = Transformer;

    const MODEL_NAME: &'static str = "qwen3_vl";

    async fn load(ctx: Context) -> Result<Box<Self>> {
        let cfg_path = ctx.data_path.join("config.json");
        let full_cfg = Qwen3VlConfig::from_path(&cfg_path)?;
        let mut text_cfg: Config = full_cfg.text();
        text_cfg.attn_f32 = ctx.args.attn_f32;
        let tie_word_embeddings = full_cfg.tie_word_embeddings || full_cfg.text_config.tie_word_embeddings;

        let tokenizer = load_tokenizer(&ctx)?;
        let eos_token_id = text_cfg
            .eos_token_id
            .or_else(|| tokenizer.token_to_id(DEFAULT_EOS_TOKEN));

        let vision_cfg = full_cfg.vision_config.clone();

        let mut vision_rknn = None;
        let mut vision_rknn_side = None;
        if let Some(model_path) = ctx.args.vision_rknn.as_deref() {
            let lib_path = ctx.args.vision_rknn_lib.as_deref().map(Path::new);
            let rknn = VisionRknn::load(Path::new(model_path), lib_path)?;
            let side = rknn.expected_side().ok_or_else(|| {
                anyhow!(
                    "rknn input must be square NCHW/NHWC with 3 channels, got {}",
                    rknn.input_summary()
                )
            })?;
            if let Some(user_side) = ctx.args.vision_fixed_side {
                if user_side != side {
                    bail!(
                        "vision_fixed_side {} does not match rknn input side {}",
                        user_side,
                        side
                    );
                }
            }
            vision_rknn_side = Some(side);
            vision_rknn = Some(rknn);
            log::info!("vision rknn enabled (side={})", side);
        }

        let vision = if vision_rknn.is_some() {
            None
        } else {
            log::info!("loading vision encoder ...");
            Some(VisionEncoder::load(
                vision_cfg.clone(),
                ctx.var_builder.clone(),
                ctx.args.attn_f32,
            )?)
        };

        let text_rknn = if let Some(model_dir) = ctx.args.text_rknn_dir.as_deref() {
            let lib_path = ctx.args.text_rknn_lib.as_deref().map(Path::new);
            let runner = TextRknnRunner::load(Path::new(model_dir), lib_path, &text_cfg)?;
            log::info!("text rknn decode enabled from {}", model_dir);
            Some(runner)
        } else {
            None
        };

        let prompt_encoder = PromptEncoder::from_tokenizer(
            &tokenizer,
            full_cfg.image_token_id,
            full_cfg.vision_start_token_id,
            full_cfg.vision_end_token_id,
        )?;

        log::info!("loading language embeddings ...");
        let embedding: Embedding = candle_nn::embedding(
            text_cfg.vocab_size,
            text_cfg.hidden_size,
            ctx.var_builder.pp("model.language_model.embed_tokens"),
        )?;

        log::info!("loading language norm ...");
        let ln_f = candle_nn::rms_norm(
            text_cfg.hidden_size,
            text_cfg.rms_norm_eps,
            ctx.var_builder.pp("model.language_model.norm"),
        )?;

        log::info!("loading lm_head ...");
        let lm_head = match linear(
            text_cfg.hidden_size,
            text_cfg.vocab_size,
            ctx.var_builder.pp("lm_head"),
        ) {
            Ok(v) => v,
            Err(e1) => match linear(
                text_cfg.hidden_size,
                text_cfg.vocab_size,
                ctx.var_builder.pp("model.language_model.lm_head"),
            ) {
                Ok(v) => v,
                Err(e2) => {
                    if tie_word_embeddings {
                        log::warn!(
                            "lm_head.weight not found ({}; {}), using tied word embeddings from model.language_model.embed_tokens.weight",
                            e1,
                            e2
                        );
                        Linear::new(embedding.embeddings().clone(), None)
                    } else {
                        return Err(anyhow!(
                            "cannot find tensor lm_head.weight (tried `lm_head.weight` and `model.language_model.lm_head.weight`): {e1}; {e2}"
                        ));
                    }
                }
            },
        };

        log::info!("loading {} text blocks ...", text_cfg.num_hidden_layers);
        let mut blocks: Vec<Box<dyn Forwarder>> = vec![];
        for i in 0..text_cfg.num_hidden_layers {
            let block_layer_name = format!("model.language_model.layers.{i}");
            if let Some((node_name, node)) = ctx.topology.get_node_for_layer(&block_layer_name) {
                log::debug!("node {node_name} will serve {}", &block_layer_name);
                blocks.push(Box::new(
                    crate::spm::Client::new(ctx.device.clone(), &node.host, &block_layer_name)
                        .await?,
                ));
            } else {
                blocks.push(Transformer::load(
                    block_layer_name.clone(),
                    ctx.var_builder.pp(&block_layer_name),
                    &text_cfg,
                )?);
            }
        }
        for block in &blocks {
            log::info!("  {}", block)
        }

        let logits_processor = create_logits_processor(&ctx);

        Ok(Box::new(Self {
            tokenizer,
            ctx,
            embedding,
            ln_f,
            lm_head,
            blocks,
            text_rknn,
            vision_cfg,
            vision,
            vision_rknn,
            vision_rknn_side,
            prompt_encoder,
            image_token_id: full_cfg.image_token_id,
            vision_start_token_id: full_cfg.vision_start_token_id,
            vision_end_token_id: full_cfg.vision_end_token_id,
            eos_token_id,
            logits_processor,
            history: vec![],
            tokens: vec![],
            image_spans: vec![],
            deepstack_spans: vec![],
            index_pos: 0,
            generated: 0,
        }))
    }

    fn add_message(&mut self, message: Message) -> Result<()> {
        self.history.push(message);
        Ok(())
    }

    fn reset(&mut self) -> Result<()> {
        self.tokens.clear();
        self.history.clear();
        self.image_spans.clear();
        self.deepstack_spans.clear();
        self.ctx.cache.clear();
        if let Some(runner) = self.text_rknn.as_mut() {
            runner.clear();
        }
        self.index_pos = 0;
        self.generated = 0;
        Ok(())
    }

    async fn next_token(&mut self, index: usize) -> Result<Token> {
        if self.generated == 0 {
            self.start_dialog_prompt()?;
        }

        let num_tokens = self.tokens.len();
        let (context_size, context_index) = if self.ctx.cache.with_kv_cache() && index > 0 {
            (1, self.index_pos)
        } else {
            (num_tokens, 0)
        };

        let context_offset = num_tokens.saturating_sub(context_size);
        let context_tokens = &self.tokens[context_offset..];
        let num_context_tokens = context_tokens.len();

        let input_ids = Tensor::new(context_tokens, &self.ctx.device)?.unsqueeze(0)?;
        let x = self.embedding.forward(&input_ids)?;

        // Only inject images for the full "prefill" pass.
        let inject_images = self.index_pos == 0 && context_index == 0 && context_size == num_tokens;

        let logits = self
            .forward_embeds(x, context_index, inject_images)
            .await
            .map_err(|e| anyhow!("forward failed: {e}"))?;

        let logits = logits.squeeze(0)?;

        let logits = if self.ctx.args.repeat_penalty == 1. {
            logits
        } else {
            let start_at = num_tokens.saturating_sub(self.ctx.args.repeat_last_n);
            candle_transformers::utils::apply_repeat_penalty(
                &logits,
                self.ctx.args.repeat_penalty,
                &self.tokens[start_at..],
            )?
        };

        self.index_pos += num_context_tokens;

        let next_token = self
            .logits_processor
            .sample(&logits)
            .map_err(|e| anyhow!("error sampling logits {logits}: {e}"))?;

        self.generated += 1;
        self.tokens.push(next_token);

        Ok(Token {
            id: next_token,
            text: match self.tokenizer.decode(&[next_token], false) {
                Ok(s) => Some(s),
                Err(e) => {
                    log::error!("could not decode token {next_token}: {e}");
                    None
                }
            },
            is_end_of_stream: Some(next_token) == self.eos_token_id,
        })
    }

    fn generated_tokens(&self) -> usize {
        self.generated
    }
}
