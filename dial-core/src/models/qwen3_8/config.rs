use std::{path::Path, sync::Arc};

use anyhow::Result;
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct Qwen38Config {
    pub model_type: String,
    pub architectures: Option<Vec<String>>,
    pub image_token_id: u32,
    pub video_token_id: Option<u32>,
    pub vision_start_token_id: u32,
    pub vision_end_token_id: u32,
    pub text_config: TextConfig,
    #[serde(default)]
    pub quantization_config: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeParameters {
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default = "default_partial_rotary_factor")]
    pub partial_rotary_factor: f64,
}

fn default_rope_theta() -> f32 {
    10_000_000.0
}

fn default_partial_rotary_factor() -> f64 {
    0.25
}

#[derive(Debug, Clone, Deserialize)]
pub struct TextConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub max_position_embeddings: usize,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub layer_types: Vec<String>,
    pub linear_conv_kernel_dim: usize,
    pub linear_key_head_dim: usize,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_value_head_dim: usize,
    #[serde(default)]
    pub rope_parameters: Option<RopeParameters>,
    #[serde(default = "default_partial_rotary_factor")]
    pub partial_rotary_factor: f64,
}

impl TextConfig {
    pub fn rope_theta(&self) -> f32 {
        self.rope_parameters
            .as_ref()
            .map(|rope| rope.rope_theta)
            .unwrap_or_else(default_rope_theta)
    }

    pub fn rotary_dim(&self) -> usize {
        let factor = self
            .rope_parameters
            .as_ref()
            .map(|rope| rope.partial_rotary_factor)
            .unwrap_or(self.partial_rotary_factor);
        ((self.head_dim as f64 * factor) as usize).max(2)
    }

    pub fn validate(&self) -> Result<()> {
        if self.layer_types.len() != self.num_hidden_layers {
            anyhow::bail!(
                "Qwen3.8 layer_types has {} entries, expected {}",
                self.layer_types.len(),
                self.num_hidden_layers
            );
        }
        if self.rotary_dim() > self.head_dim || self.rotary_dim() % 2 != 0 {
            anyhow::bail!(
                "invalid Qwen3.8 rotary dimension {} for head dimension {}",
                self.rotary_dim(),
                self.head_dim
            );
        }
        if self.linear_num_value_heads % self.linear_num_key_heads != 0 {
            anyhow::bail!(
                "Qwen3.8 linear value heads {} must be divisible by key heads {}",
                self.linear_num_value_heads,
                self.linear_num_key_heads
            );
        }
        for (index, layer_type) in self.layer_types.iter().enumerate() {
            if layer_type != "linear_attention" && layer_type != "full_attention" {
                anyhow::bail!("unsupported Qwen3.8 layer type {layer_type:?} at layer {index}");
            }
        }
        Ok(())
    }
}

impl Qwen38Config {
    pub fn from_path(path: &Path) -> Result<Self> {
        log::info!("loading Qwen3.8 configuration from {}", path.display());
        let data = std::fs::read(path)
            .map_err(|error| anyhow!("can't read {}: {error}", path.display()))?;
        let config: Self = serde_json::from_slice(&data)
            .map_err(|error| anyhow!("can't parse {}: {error}", path.display()))?;
        if config.model_type != "qwen3_5" {
            anyhow::bail!(
                "Qwen3.8 backend requires model_type=qwen3_5, got {:?}",
                config.model_type
            );
        }
        config.text_config.validate()?;
        Ok(config)
    }

    pub fn is_quantized(&self) -> bool {
        self.quantization_config.is_some()
    }

    pub fn generic_text_config(&self) -> crate::models::llama3::Config {
        crate::models::llama3::Config {
            hidden_size: self.text_config.hidden_size,
            intermediate_size: self.text_config.intermediate_size,
            vocab_size: self.text_config.vocab_size,
            num_hidden_layers: self.text_config.num_hidden_layers,
            num_attention_heads: self.text_config.num_attention_heads,
            num_key_value_heads: self.text_config.num_key_value_heads,
            rms_norm_eps: self.text_config.rms_norm_eps,
            rope_theta: self.text_config.rope_theta(),
            bos_token_id: self.text_config.bos_token_id,
            eos_token_id: self.text_config.eos_token_id,
            max_seq_len: self.text_config.max_position_embeddings,
            attn_f32: false,
            qwen3_8: Some(Arc::new(self.text_config.clone())),
            qwen3_8_ggml: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_official_qwen38_shape() {
        let config: Qwen38Config = serde_json::from_value(serde_json::json!({
            "model_type": "qwen3_5",
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "image_token_id": 248056,
            "video_token_id": 248057,
            "vision_start_token_id": 248053,
            "vision_end_token_id": 248054,
            "text_config": {
                "hidden_size": 5120,
                "intermediate_size": 17408,
                "vocab_size": 248320,
                "num_hidden_layers": 4,
                "num_attention_heads": 24,
                "num_key_value_heads": 4,
                "head_dim": 256,
                "rms_norm_eps": 1e-6,
                "max_position_embeddings": 262144,
                "bos_token_id": 248044,
                "eos_token_id": 248044,
                "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
                "linear_conv_kernel_dim": 4,
                "linear_key_head_dim": 128,
                "linear_num_key_heads": 16,
                "linear_num_value_heads": 48,
                "linear_value_head_dim": 128,
                "partial_rotary_factor": 0.25,
                "rope_parameters": {"rope_theta": 10000000, "partial_rotary_factor": 0.25}
            }
        })).unwrap();
        config.text_config.validate().unwrap();
        assert_eq!(config.text_config.rotary_dim(), 64);
    }
}
