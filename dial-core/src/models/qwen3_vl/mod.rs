//! Qwen3-VL model implementation (vision + language).
//!
//! Note: This implementation focuses on end-to-end inference support for the
//! Qwen3-VL-8B-Instruct checkpoint layout (HF safetensors) and integrates with
//! the spm master/worker sharding framework (text layers can be sharded).

mod config;
mod history;
mod qwen3_vl;
mod text;
mod text_rknn;
mod vision;
mod vision_rknn;

pub use config::*;
pub use history::*;
pub use qwen3_vl::*;
pub use text::*;
pub use text_rknn::*;
pub use vision::*;
pub use vision_rknn::*;
