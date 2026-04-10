//! This is the core library where all spm logic is implemented.
#[macro_use]
extern crate anyhow;

use spm::Mode;

use clap::Parser;

pub mod models;
pub mod spm;
pub mod utils;

#[derive(Clone, Parser, Default, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// GPU device index.
    #[arg(long, default_value_t = 0)]
    pub device: usize,
    #[arg(long, default_value_t, value_enum)]
    pub mode: Mode,

    /// Worker name.
    #[arg(long, default_value = "worker0")]
    pub name: Option<String>,

    /// Binding address and port for workers.
    #[arg(long, default_value = "0.0.0.0:10128")]
    pub address: String,

    /// Enable OpenAI compatible chat completion API.
    #[arg(long)]
    pub api: Option<String>,

    /// （新增）作为“客户端”连接到一个已启动的 API 服务端（例如 http://127.0.0.1:8082）。
    ///
    /// 为什么要加：之前用 curl 需要手写 JSON，请求/解析都不方便；
    /// 加了这个参数后，`spm-cli` 可以直接在命令行里输入问题并打印模型回复（纯文本）。
    #[arg(long)]
    pub api_client: Option<String>,

    /// （新增）配合 `--api-client`：发送一条消息并退出（单次提问）。
    #[arg(long)]
    pub ask: Option<String>,

    /// （新增）配合 `--api-client`：启动交互式对话（REPL），像聊天一样连续提问。
    #[arg(long, default_value_t = false)]
    pub repl: bool,

    /// （新增）配合 `--api-client`：服务端流式输出（边生成边显示）。
    #[arg(long, default_value_t = true)]
    pub stream: bool,

    /// （新增）配合 `--api-client`：在命令行额外打印服务端返回的性能指标（如 ttft_s/total_s）。
    /// 默认不打印，避免影响“只要回复文本”的使用体验。
    #[arg(long, default_value_t = true)]
    pub metrics: bool,

    /// （新增）限制 KV-cache/rope 的最大上下文长度（默认 4096）。
    ///
    /// 为什么要加：在 GPU 上使用旋转 KV-cache 时需要预分配固定窗口，
    /// 过大的 max_len 会导致显存暴涨/ OOM；限制到 4K 既能加速 decode
    /// 也能避免占用过大显存。
    #[arg(long, default_value_t = 4096)]
    pub kv_cache_max_len: usize,

    /// （新增）限制多模态图片送入视觉编码器前的最大边长（像素）。
    ///
    /// 为什么要加：Qwen3-VL 的视觉编码（ViT）在 CPU 上对 token 数非常敏感（复杂度近似 O(N^2)）。
    /// 默认如果把图片缩放到 768（48x48 patches），TTFT 很容易到几十秒；把它降到 384（24x24）
    /// 往往能带来数量级的提速。
    #[arg(long)]
    pub vision_max_side: Option<u32>,

    /// （新增）禁止把小图放大（只缩小，不上采样）。
    ///
    /// 为什么要加：上采样不会增加细节，但会显著增加视觉 token 数与 TTFT。
    #[arg(long, default_value_t = false)]
    pub vision_no_upscale: bool,

    /// （新增）把多模态图片强制缩放/补边到固定的正方形边长（像素），用于需要静态输入 shape 的硬件后端（如 BM1684）。
    ///
    /// - 输出尺寸会对齐到 `patch_size * spatial_merge_size` 的整数倍，避免后续 encode() 再 crop。
    /// - 默认会保持宽高比并做 letterbox（用 0.5 灰填充，归一化后约为 0），不会裁剪内容。
    /// - 与 `--vision-max-side` 不同：它是“强制固定”，适合导出 ONNX/bmodel。
    #[arg(long)]
    pub vision_fixed_side: Option<u32>,

    /// （新增）使用 RKNN 运行视觉编码器（ViT+merger），指定 .rknn 模型路径。
    #[arg(long)]
    pub vision_rknn: Option<String>,

    /// （新增）RKNN runtime 库路径（librknnrt.so），不填则自动搜索。
    #[arg(long)]
    pub vision_rknn_lib: Option<String>,

    /// （新增）使用 RKNN chunk 运行 Qwen3-VL 文本 decode（目录下应包含按层切块的 .rknn）。
    ///
    /// 目前仅接管 decode（seq=1, index_pos>0）路径，prefill 仍走现有 candle/分布式 forward。
    #[arg(long)]
    pub text_rknn_dir: Option<String>,

    /// （新增）文本 RKNN chunk 的 runtime 库路径（librknnrt.so），不填则自动搜索。
    #[arg(long)]
    pub text_rknn_lib: Option<String>,

    /// （新增）配合 `--api-client`：附带本地图片文件，用于 Qwen3-VL 这类多模态模型的图片理解。
    /// 实现方式：客户端会把图片转成 base64，并按 OpenAI 风格的 `image_base64` part 发送。
    #[arg(long)]
    pub image: Option<String>,

    /// Llama3 model data path.
    #[arg(long, default_value = "/root/sdb/Qwen3-VL-8B-Instruct")]
    pub model: String,

    /// Topology file.
    #[arg(long, default_value = "/root/sdb/ljl/Spm_llama/topology.yml")]
    pub topology: String,

    /// The initial prompt.
    #[arg(long, default_value = "")]
    pub prompt: String,
    /// The system prompt.
    #[arg(long, default_value = "You are a helpful AI assistant.")]
    pub system_prompt: String,
    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    pub seed: u64,
    /// The length of the sample to generate (in tokens).
    #[arg(short = 'n', long, default_value_t = 2048)]
    pub sample_len: usize,
    /// The temperature used to generate samples.
    #[arg(long, default_value_t = 1.0)]
    pub temperature: f64,
    /// Nucleus sampling probability cutoff.
    #[arg(long)]
    pub top_p: Option<f64>,
    /// Only sample among the top K samples.
    #[arg(long)]
    pub top_k: Option<usize>,
    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.1)]
    pub repeat_penalty: f32,
    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 128)]
    pub repeat_last_n: usize,
    /// Use different dtype than f16
    #[arg(long)]
    pub dtype: Option<String>,
    /// Force attention matmul/softmax to use f32 (more stable, slower).
    #[arg(long, default_value_t = false)]
    pub attn_f32: bool,
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    pub cpu: bool,
}
