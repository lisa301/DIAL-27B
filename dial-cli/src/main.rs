//! This is the spm command line utility.

use dial_core::{
    spm::{Context, Master, Mode, Worker},
    Args, InferenceBackend, ModelSize,
};

use anyhow::{anyhow, Result};
use base64::Engine;
use clap::Parser;
use std::{
    fs,
    io::{Read, Write},
    path::Path,
    process::Stdio,
    time::Duration,
};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command};

fn fmt_metric(v: Option<f64>, unit: &str) -> String {
    v.map(|v| format!("{v:.3}{unit}"))
        .unwrap_or_else(|| "null".to_string())
}

fn print_metrics(
    ttft: Option<f64>,
    total: Option<f64>,
    tps: Option<f64>,
    decode_tps: Option<f64>,
    dist_overhead: Option<f64>,
    remote_compute: Option<f64>,
    remote_requests: Option<usize>,
) {
    eprintln!(
        "[metrics] ttft_s={} total_s={} tps={} decode_tps={} dist_overhead_s={} remote_compute_s={} remote_requests={}",
        fmt_metric(ttft,"s"),
        fmt_metric(total,"s"),
        fmt_metric(tps,"toks/s"),
        fmt_metric(decode_tps,"toks/s"),
        fmt_metric(dist_overhead,"s"),
        fmt_metric(remote_compute,"s"),
        remote_requests
            .map(|v| v.to_string())
            .unwrap_or_else(|| "null".to_string()),
    );
}

fn display_model_name(model_dir: &str) -> String {
    let name = Path::new(model_dir)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(model_dir)
        .to_ascii_lowercase();

    if name.contains("qwen3.8-27b") || name.contains("qwen38-27b") {
        "Qwen3.8-27B".to_string()
    } else if name.contains("qwen3-vl-8b") {
        "qwen3-vl-8B".to_string()
    } else if name.contains("qwen3-vl-2b") {
        "qwen3-vl-2B".to_string()
    } else if name.contains("llama-3-8b") || name.contains("llama3-8b") {
        "llama3-8B".to_string()
    } else {
        name
    }
}

fn print_model_header(args: &Args, server_model: Option<&str>) {
    let name = server_model.map(display_model_name).unwrap_or_else(|| {
        if matches!(
            args.inference_backend,
            InferenceBackend::Qwen38Native | InferenceBackend::Qwen38Ggml | InferenceBackend::Qwen38Rpc
        ) {
            "Qwen3.8-27B".to_string()
        } else {
            display_model_name(&args.model)
        }
    });
    println!("[choose_model]:{name}");
}

fn print_response_model_header(args: &Args, response: &reqwest::Response) {
    let server_model = response
        .headers()
        .get("x-dial-model")
        .and_then(|value| value.to_str().ok());
    print_model_header(args, server_model);
}

fn has_option<I, S>(raw_args: I, option: &str) -> bool
where
    I: IntoIterator<Item = S>,
    S: AsRef<std::ffi::OsStr>,
{
    raw_args.into_iter().any(|value| {
        value.as_ref().to_str().is_some_and(|value| {
            value == option
                || value
                    .strip_prefix(option)
                    .is_some_and(|rest| rest.starts_with('='))
        })
    })
}

fn select_model_size<I, S>(args: &mut Args, raw_args: I) -> Result<()>
where
    I: IntoIterator<Item = S>,
    S: AsRef<std::ffi::OsStr>,
{
    let raw_args = raw_args.into_iter().collect::<Vec<_>>();
    let qwen38_option_was_set = raw_args.iter().any(|value| {
        value
            .as_ref()
            .to_str()
            .is_some_and(|value| value.starts_with("--qwen38-"))
    });
    let backend_was_set = has_option(&raw_args, "--inference-backend");
    let gguf_was_set = has_option(&raw_args, "--qwen38-gguf");
    let ggml_lib_was_set = has_option(&raw_args, "--qwen38-ggml-lib");

    if backend_was_set
        && matches!(
            args.inference_backend,
            InferenceBackend::Native | InferenceBackend::Qwen38Native
        )
        && (gguf_was_set || ggml_lib_was_set)
    {
        return Err(anyhow!(
            "--qwen38-gguf/--qwen38-ggml-lib cannot be used with the native backend; use --inference-backend qwen38-ggml"
        ));
    }

    match args.model_size {
        Some(ModelSize::B8) => {
            if args.inference_backend != InferenceBackend::Native || qwen38_option_was_set {
                return Err(anyhow!(
                    "--model-size 8b cannot be combined with qwen38-rpc or --qwen38-* options"
                ));
            }
            args.inference_backend = InferenceBackend::Native;
        }
        Some(ModelSize::B27) => {
            if backend_was_set && args.inference_backend == InferenceBackend::Native {
                return Err(anyhow!(
                    "--model-size 27b cannot be combined with --inference-backend native"
                ));
            }
            if !backend_was_set {
                args.inference_backend = match (gguf_was_set, ggml_lib_was_set) {
                    (true, true) => InferenceBackend::Qwen38Ggml,
                    (true, false) => {
                        return Err(anyhow!(
                            "--qwen38-gguf requires --qwen38-ggml-lib when --inference-backend is omitted"
                        ));
                    }
                    (false, true) => {
                        return Err(anyhow!(
                            "--qwen38-ggml-lib requires --qwen38-gguf when --inference-backend is omitted"
                        ));
                    }
                    (false, false) => InferenceBackend::Qwen38Native,
                };
            }
        }
        None if args.inference_backend == InferenceBackend::Native && qwen38_option_was_set => {
            return Err(anyhow!(
                "--qwen38-* options require --model-size 27b (or --inference-backend qwen38-rpc)"
            ));
        }
        None => {}
    }

    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SelectedModel {
    Qwen3Vl,
    Qwen38,
    Llama3,
}

/// （新增）根据图片扩展名推断 mime 类型。
///
/// 为什么要加：Qwen3-VL 的 `image_base64` part 允许携带 `media_type`；
/// 这能帮助服务端/模型更准确地理解图片格式（png/jpg/webp...）。
fn guess_media_type(path: &str) -> Option<String> {
    let ext = Path::new(path)
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    let mime = match ext.as_str() {
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "webp" => "image/webp",
        "bmp" => "image/bmp",
        _ => return None,
    };
    Some(mime.to_string())
}

/// （新增）把本地图片读出来并转成 `image_base64` 这个多模态输入 part。
///
/// 为什么要加：用户希望“命令行直接问图”，而不是手写一大坨 JSON + base64；
/// 这个函数把“读文件 + base64 编码 + 组装 ContentPart”封装起来。
fn image_part_from_path(path: &str) -> Result<dial_core::models::chat::ContentPart> {
    let bytes = fs::read(path).map_err(|e| anyhow!("can't read image {}: {e}", path))?;
    let data = base64::engine::general_purpose::STANDARD.encode(bytes);
    Ok(dial_core::models::chat::ContentPart::ImageBase64 {
        media_type: guess_media_type(path),
        data,
    })
}

fn guess_video_media_type(path: &str) -> Option<String> {
    let ext = Path::new(path)
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    let mime = match ext.as_str() {
        "mp4" | "m4v" => "video/mp4",
        "webm" => "video/webm",
        "mov" => "video/quicktime",
        "mkv" => "video/x-matroska",
        "avi" => "video/x-msvideo",
        _ => return None,
    };
    Some(mime.to_string())
}

fn video_part_from_path(
    path: &str,
    max_bytes: usize,
) -> Result<dial_core::models::chat::ContentPart> {
    let metadata = fs::metadata(path).map_err(|e| anyhow!("can't stat video {}: {e}", path))?;
    if metadata.len() > max_bytes as u64 {
        return Err(anyhow!(
            "video {} is {} bytes, exceeding --video-max-bytes {}",
            path,
            metadata.len(),
            max_bytes
        ));
    }
    let bytes = fs::read(path).map_err(|e| anyhow!("can't read video {}: {e}", path))?;
    let data = base64::engine::general_purpose::STANDARD.encode(bytes);
    Ok(dial_core::models::chat::ContentPart::VideoBase64 {
        media_type: guess_video_media_type(path),
        data,
    })
}

/// （新增）生成一个 user message：既支持纯文本，也支持“文本 + 图片”的多模态 message。
///
/// 实现了什么：当传入 `image_path` 时，构造 OpenAI 风格的 `content: [ {text}, {image_base64} ]`；
/// 不传图片时则退化为普通文本消息，兼容纯文本模型/请求。
fn user_message_with_media(
    text: String,
    image_path: Option<&str>,
    video_path: Option<&str>,
    video_max_bytes: usize,
) -> Result<dial_core::models::chat::Message> {
    if image_path.is_some() && video_path.is_some() {
        return Err(anyhow!("--image and --video cannot be used together"));
    }
    if let Some(path) = image_path {
        // Qwen3-VL 的官方/常见用法是“先给图，再提问”，因此把 image part 放在 text 前面。
        let parts = vec![
            image_part_from_path(path)?,
            dial_core::models::chat::ContentPart::Text { text },
        ];
        Ok(dial_core::models::chat::Message {
            role: dial_core::models::chat::MessageRole::User,
            content: dial_core::models::chat::MessageContent::Parts(parts),
        })
    } else if let Some(path) = video_path {
        let parts = vec![
            video_part_from_path(path, video_max_bytes)?,
            dial_core::models::chat::ContentPart::Text { text },
        ];
        Ok(dial_core::models::chat::Message {
            role: dial_core::models::chat::MessageRole::User,
            content: dial_core::models::chat::MessageContent::Parts(parts),
        })
    } else {
        Ok(dial_core::models::chat::Message::user(text))
    }
}

/// （新增）把 `--api-client` 的输入统一规范成 base URL。
///
/// 为什么要加：用户可能传 `127.0.0.1:8082` / `http://127.0.0.1:8082/` 等各种形式；
/// 统一后拼接 `/api/v1/chat/completions` 更稳妥。
fn normalize_api_base(raw: &str) -> String {
    let mut s = raw.trim().to_string();
    if s.is_empty() {
        return s;
    }
    if !s.starts_with("http://") && !s.starts_with("https://") {
        s = format!("http://{s}");
    }
    s.trim_end_matches('/').to_string() //把末尾的/去掉
}

/// Read an OpenAI-style SSE stream and print delta tokens to stdout as they arrive.
/// Returns the full assistant content and optional metrics when provided by the server.
async fn consume_sse_stream(
    mut resp: reqwest::Response,
    print_deltas: bool,
) -> Result<(
    String,
    Option<f64>,
    Option<f64>,
    Option<f64>,
    Option<f64>,
    Option<f64>,
    Option<f64>,
    Option<usize>,
)> {
    let mut buf = String::new();
    let mut out = String::new();
    let mut ttft_s: Option<f64> = None;
    let mut total_s: Option<f64> = None;
    let mut tokens_per_second: Option<f64> = None;
    let mut decode_tokens_per_second: Option<f64> = None;
    let mut distributed_overhead_s: Option<f64> = None;
    let mut remote_compute_s: Option<f64> = None;
    let mut remote_requests: Option<usize> = None;

    loop {
        let chunk = resp.chunk().await?;
        let Some(chunk) = chunk else { break };
        let s = std::str::from_utf8(&chunk)
            .map_err(|e| anyhow!("invalid utf-8 in SSE response: {e}"))?;
        buf.push_str(s);

        while let Some(idx) = buf.find("\n\n") {
            let event = buf[..idx].to_string();
            buf.drain(..idx + 2);

            for line in event.lines() {
                let Some(data) = line.strip_prefix("data:") else {
                    continue;
                };
                let data = data.trim();

                if data == "[DONE]" {
                    if print_deltas {
                        println!();
                    }
                    return Ok((
                        out,
                        ttft_s,
                        total_s,
                        tokens_per_second,
                        decode_tokens_per_second,
                        distributed_overhead_s,
                        remote_compute_s,
                        remote_requests,
                    ));
                }

                // Normal OpenAI streaming chunk.
                if data.starts_with('{') {
                    let v: serde_json::Value =
                        serde_json::from_str(data).map_err(|e| anyhow!("bad SSE json: {e}"))?;

                    if let Some(err) = v.get("error").and_then(|e| e.as_str()) {
                        return Err(anyhow!("{err}"));
                    }

                    if let Some(c) = v["choices"][0]["delta"]["content"].as_str() {
                        if !c.is_empty() {
                            out.push_str(c);
                            if print_deltas {
                                print!("{c}");
                                std::io::stdout().flush().ok();
                            }
                        }
                    }

                    // Our server optionally attaches metrics on the final chunk.
                    if ttft_s.is_none() {
                        ttft_s = v.get("ttft_s").and_then(|x| x.as_f64());
                    }
                    if total_s.is_none() {
                        total_s = v.get("total_s").and_then(|x| x.as_f64());
                    }
                    if tokens_per_second.is_none() {
                        tokens_per_second = v.get("tokens_per_second").and_then(|x| x.as_f64());
                    }
                    if decode_tokens_per_second.is_none() {
                        decode_tokens_per_second =
                            v.get("decode_tokens_per_second").and_then(|x| x.as_f64());
                    }
                    if distributed_overhead_s.is_none() {
                        distributed_overhead_s =
                            v.get("distributed_overhead_s").and_then(|x| x.as_f64());
                    }
                    if remote_compute_s.is_none() {
                        remote_compute_s = v.get("remote_compute_s").and_then(|x| x.as_f64());
                    }
                    if remote_requests.is_none() {
                        remote_requests = v
                            .get("remote_requests")
                            .and_then(|x| x.as_u64())
                            .map(|v| v as usize);
                    }
                }
            }
        }
    }

    // Stream ended without a [DONE] marker (still return what we got).
    if print_deltas {
        println!();
    }
    Ok((
        out,
        ttft_s,
        total_s,
        tokens_per_second,
        decode_tokens_per_second,
        distributed_overhead_s,
        remote_compute_s,
        remote_requests,
    ))
}

/// （新增）`spm-cli` 的“简易客户端模式”：
/// - 连接到已运行的 `--api` 服务端
/// - 自动拼 JSON、发送请求
/// - 只在终端打印 assistant 的纯文本内容（不输出整段 JSON）
///
/// 为什么要加：让第二个端口（API 服务）用起来更像“聊天”，而不是每次写 curl + JSON。
async fn run_api_client(args: Args) -> Result<()> {
    let base = args
        .api_client
        .as_deref()
        .map(normalize_api_base)
        .unwrap_or_default();
    if base.is_empty() {
        return Err(anyhow!("--api-client is empty"));
    }

    let url = format!("{base}/api/v1/chat/completions");
    let http = reqwest::Client::new();

    let mut messages: Vec<dial_core::models::chat::Message> = vec![];
    if !args.system_prompt.is_empty() {
        // 把 system_prompt 放到会话历史里，服务端按 OpenAI 兼容格式处理。
        messages.push(dial_core::models::chat::Message::system(
            args.system_prompt.clone(),
        ));
    }

    // Single-shot mode: --ask, else fallback to --prompt, else read stdin once.
    if !args.repl {
        let ask = args
            .ask
            .clone()
            .or_else(|| (!args.prompt.is_empty()).then(|| args.prompt.clone()));

        let ask = match ask {
            Some(s) => s,
            None => {
                let mut buf = String::new();
                BufReader::new(tokio::io::stdin())
                    .read_line(&mut buf)
                    .await?;
                buf.trim().to_string()
            }
        };

        if ask.is_empty() {
            return Err(anyhow!(
                "no prompt provided; use --ask, --prompt, or pipe text into stdin"
            ));
        }

        // 单次提问：支持纯文本或“文本+图片”（由 --image 控制）。
        messages.push(user_message_with_media(
            ask,
            args.image.as_deref(),
            args.video.as_deref(),
            args.video_max_bytes,
        )?);

        if args.stream {
            let resp = http
                .post(url)
                .json(&serde_json::json!({ "messages": messages, "stream": true }))
                .send()
                .await?
                .error_for_status()?;
            print_response_model_header(&args, &resp);
            let (
                _content,
                ttft,
                total,
                tps,
                decode_tps,
                dist_overhead,
                remote_compute,
                remote_requests,
            ) = consume_sse_stream(resp, true).await?;
            if args.metrics {
                print_metrics(
                    ttft,
                    total,
                    tps,
                    decode_tps,
                    dist_overhead,
                    remote_compute,
                    remote_requests,
                );
            }
        }
        return Ok(());
    }

    // REPL mode.
    let mut stdin = BufReader::new(tokio::io::stdin());
    let mut line = String::new();

    if !args.prompt.is_empty() {
        // 可选：启动 REPL 前先发一条初始 prompt（也支持 --image）。
        messages.push(user_message_with_media(
            args.prompt.clone(),
            args.image.as_deref(),
            args.video.as_deref(),
            args.video_max_bytes,
        )?);
        if args.stream {
            let resp = http
                .post(&url)
                .json(&serde_json::json!({ "messages": messages, "stream": true }))
                .send()
                .await?
                .error_for_status()?;
            print_response_model_header(&args, &resp);
            let (
                content,
                ttft,
                total,
                tps,
                decode_tps,
                dist_overhead,
                remote_compute,
                remote_requests,
            ) = consume_sse_stream(resp, true).await?;
            messages.push(dial_core::models::chat::Message::assistant(content));
            if args.metrics {
                print_metrics(
                    ttft,
                    total,
                    tps,
                    decode_tps,
                    dist_overhead,
                    remote_compute,
                    remote_requests,
                );
            }
        }
    }

    loop {
        line.clear();
        // 交互提示符写到 stderr，方便把 stdout 重定向保存模型回复内容。
        eprint!("你> ");
        let n = stdin.read_line(&mut line).await?;
        if n == 0 {
            break;
        }
        let input = line.trim();
        if input.is_empty() {
            continue;
        }
        if matches!(input, "/q" | "/quit" | "/exit") {
            break;
        }

        // （新增）REPL 图片快捷指令：`/img path/to.png 你要问的问题`
        // 为什么要加：在 REPL 里临时换图片更方便，不需要退出重启进程/改 --image。
        let (user_text, img_path, video_path) = if let Some(rest) = input.strip_prefix("/img ") {
            let rest = rest.trim();
            let mut it = rest.splitn(2, char::is_whitespace);
            let path = it.next().unwrap_or("").trim();
            let question = it.next().unwrap_or("").trim();
            if path.is_empty() || question.is_empty() {
                println!("用法: /img <图片路径> <问题>");
                continue;
            }
            (question.to_string(), Some(path.to_string()), None)
        } else if let Some(rest) = input.strip_prefix("/video ") {
            let rest = rest.trim();
            let mut it = rest.splitn(2, char::is_whitespace);
            let path = it.next().unwrap_or("").trim();
            let question = it.next().unwrap_or("").trim();
            if path.is_empty() || question.is_empty() {
                println!("用法: /video <视频路径> <问题>");
                continue;
            }
            (question.to_string(), None, Some(path.to_string()))
        } else {
            (input.to_string(), None, None)
        };

        // 行内 /img 或 /video 优先；普通文本才复用启动参数中的媒体。
        let (selected_image, selected_video) = if img_path.is_some() || video_path.is_some() {
            (img_path.as_deref(), video_path.as_deref())
        } else {
            (args.image.as_deref(), args.video.as_deref())
        };
        messages.push(user_message_with_media(
            user_text,
            selected_image,
            selected_video,
            args.video_max_bytes,
        )?);
        if args.stream {
            let resp = http
                .post(&url)
                .json(&serde_json::json!({ "messages": messages, "stream": true }))
                .send()
                .await?
                .error_for_status()?;
            print_response_model_header(&args, &resp);
            let (
                content,
                ttft,
                total,
                tps,
                decode_tps,
                dist_overhead,
                remote_compute,
                remote_requests,
            ) = consume_sse_stream(resp, true).await?;
            messages.push(dial_core::models::chat::Message::assistant(content));
            if args.metrics {
                print_metrics(
                    ttft,
                    total,
                    tps,
                    decode_tps,
                    dist_overhead,
                    remote_compute,
                    remote_requests,
                );
            }
        }
    }

    Ok(())
}

fn detect_model_type(model_dir: &str) -> Result<SelectedModel> {
    let config_path = Path::new(model_dir).join("config.json");
    let raw =
        fs::read(&config_path).map_err(|e| anyhow!("can't read {}: {e}", config_path.display()))?;
    let v: serde_json::Value = serde_json::from_slice(&raw)
        .map_err(|e| anyhow!("can't parse {}: {e}", config_path.display()))?;
    Ok(match v.get("model_type") {
        Some(serde_json::Value::String(s)) if s == "qwen3_vl" => SelectedModel::Qwen3Vl,
        Some(serde_json::Value::String(s)) if s == "qwen3_5" => SelectedModel::Qwen38,
        _ => SelectedModel::Llama3,
    })
}

#[cfg(all(target_arch = "aarch64", target_os = "linux"))]
fn env_value_is_off(value: &str) -> bool {
    let value = value.trim().to_ascii_lowercase();
    matches!(
        value.as_str(),
        "0" | "false" | "no" | "off" | "disable" | "disabled"
    )
}

#[cfg(all(target_arch = "aarch64", target_os = "linux"))]
fn parse_cpu_affinity_spec(spec: &str) -> Vec<usize> {
    let mut cpus = Vec::new();
    for part in spec
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
    {
        if let Some((start, end)) = part.split_once('-') {
            let (Ok(start), Ok(end)) = (start.trim().parse::<usize>(), end.trim().parse::<usize>())
            else {
                continue;
            };
            if start <= end {
                cpus.extend(start..=end);
            }
        } else if let Ok(cpu) = part.parse::<usize>() {
            cpus.push(cpu);
        }
    }
    cpus.sort_unstable();
    cpus.dedup();
    cpus
}

#[cfg(all(target_arch = "aarch64", target_os = "linux"))]
fn default_cpu_affinity_spec() -> Option<String> {
    let cpus = std::thread::available_parallelism().ok()?.get();
    if cpus >= 8 {
        Some("4-7".to_string())
    } else {
        None
    }
}

#[cfg(all(target_arch = "aarch64", target_os = "linux"))]
fn set_thread_affinity(tid: libc::pid_t, cpus: &[usize]) -> std::io::Result<()> {
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut set);
        for cpu in cpus {
            libc::CPU_SET(*cpu, &mut set);
        }
        let rc = libc::sched_setaffinity(tid, std::mem::size_of::<libc::cpu_set_t>(), &set);
        if rc == 0 {
            Ok(())
        } else {
            Err(std::io::Error::last_os_error())
        }
    }
}

#[cfg(all(target_arch = "aarch64", target_os = "linux"))]
fn apply_cpu_affinity() {
    if std::env::var("DIAL_DISABLE_CPU_AFFINITY")
        .map(|value| !env_value_is_off(&value))
        .unwrap_or(false)
    {
        log::info!("cpu affinity disabled by DIAL_DISABLE_CPU_AFFINITY");
        return;
    }

    let spec = match std::env::var("DIAL_CPU_AFFINITY") {
        Ok(value) if env_value_is_off(&value) => {
            log::info!("cpu affinity disabled by DIAL_CPU_AFFINITY={value}");
            return;
        }
        Ok(value) => value,
        Err(_) => match default_cpu_affinity_spec() {
            Some(value) => value,
            None => return,
        },
    };

    let cpus = parse_cpu_affinity_spec(&spec);
    if cpus.is_empty() {
        log::warn!("invalid DIAL_CPU_AFFINITY={spec}; cpu affinity not changed");
        return;
    }

    let mut applied = 0usize;
    let mut last_error = None;
    if let Ok(entries) = fs::read_dir("/proc/self/task") {
        for entry in entries.flatten() {
            let Some(name) = entry.file_name().to_str().map(str::to_string) else {
                continue;
            };
            let Ok(tid) = name.parse::<libc::pid_t>() else {
                continue;
            };
            match set_thread_affinity(tid, &cpus) {
                Ok(()) => applied += 1,
                Err(e) => last_error = Some(e),
            }
        }
    } else if let Err(e) = set_thread_affinity(0, &cpus) {
        last_error = Some(e);
    } else {
        applied = 1;
    }

    if applied > 0 {
        log::info!("cpu affinity set to {spec} for {applied} thread(s)");
    } else if let Some(error) = last_error {
        log::warn!("failed to set cpu affinity {spec}: {error}");
    } else {
        log::warn!("failed to set cpu affinity {spec}: no target threads found");
    }
}

fn qwen38_gguf_path(args: &Args) -> Result<String> {
    let path = args.qwen38_gguf.as_deref().unwrap_or(&args.model);
    if !path.to_ascii_lowercase().ends_with(".gguf") {
        return Err(anyhow!(
            "qwen38-rpc requires a quantized GGUF file; pass --qwen38-gguf /path/Qwen3.8-27B-Q4_K_M.gguf (the 55.6 GB Hugging Face directory is not usable by this backend)"
        ));
    }
    if !Path::new(path).is_file() {
        return Err(anyhow!("Qwen3.8 GGUF file does not exist: {path}"));
    }
    let metadata = fs::metadata(path)?;
    if metadata.len() < 1024 * 1024 * 1024 {
        return Err(anyhow!(
            "Qwen3.8 GGUF file is unexpectedly small ({:.2} GiB): {path}; the download may be incomplete",
            metadata.len() as f64 / 1073741824.0
        ));
    }
    let mut header = [0u8; 4];
    fs::File::open(path)?.read_exact(&mut header)?;
    if &header != b"GGUF" {
        return Err(anyhow!(
            "Qwen3.8 model is not a GGUF file (bad header): {path}"
        ));
    }
    Ok(path.to_string())
}

fn split_bind_address(address: &str) -> Result<(String, String)> {
    let (host, port) = address
        .rsplit_once(':')
        .ok_or_else(|| anyhow!("invalid --address {address}; expected HOST:PORT"))?;
    let host = host.trim_matches(['[', ']']);
    if host.is_empty() || port.parse::<u16>().is_err() {
        return Err(anyhow!("invalid --address {address}; expected HOST:PORT"));
    }
    Ok((host.to_string(), port.to_string()))
}

async fn run_qwen38_rpc_worker(args: Args) -> Result<()> {
    if args.worker_quantized_gguf.is_some()
        || args.worker_gguf_fp16_prefill
        || args.worker_gguf_output_head
        || args.worker_gguf_sample_token
        || args.worker_w8a16
    {
        return Err(anyhow!(
            "qwen38-rpc worker does not load --worker-quantized-gguf or other native Qwen3-VL worker options; remove them and load Qwen3.8-27B on the master with --qwen38-gguf"
        ));
    }

    let (host, port) = split_bind_address(&args.address)?;
    let mut command = Command::new(&args.qwen38_rpc_server_bin);
    command
        .arg("--host")
        .arg(host)
        .arg("--port")
        .arg(port)
        .arg("--device")
        .arg(format!("CUDA{}", args.device))
        .stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());
    if args.qwen38_rpc_cache {
        command.arg("--cache");
    }

    log::warn!("Qwen3.8 GGML RPC has no authentication; bind it only to a trusted private network");
    log::info!(
        "starting Qwen3.8 CUDA RPC worker: {} on {}",
        args.qwen38_rpc_server_bin,
        args.address
    );
    let status = command
        .status()
        .await
        .map_err(|error| anyhow!("failed to start {}: {error}", args.qwen38_rpc_server_bin))?;
    if status.success() {
        Ok(())
    } else {
        Err(anyhow!("Qwen3.8 RPC worker exited with {status}"))
    }
}

async fn wait_for_llama_server(child: &mut Child, upstream: &str, timeout_s: u64) -> Result<()> {
    let health_url = format!("{}/health", upstream.trim_end_matches('/'));
    let http = reqwest::Client::new();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(timeout_s);
    loop {
        if let Some(status) = child.try_wait()? {
            return Err(anyhow!(
                "managed llama-server exited before becoming ready: {status}"
            ));
        }
        if let Ok(response) = http.get(&health_url).send().await {
            if response.status().is_success() {
                return Ok(());
            }
        }
        if tokio::time::Instant::now() >= deadline {
            let _ = child.kill().await;
            return Err(anyhow!(
                "llama-server did not become ready within {timeout_s}s; inspect its log above"
            ));
        }
        tokio::time::sleep(Duration::from_secs(2)).await;
    }
}

async fn run_qwen38_rpc_master(args: Args) -> Result<()> {
    if args.api.is_none() {
        return Err(anyhow!("qwen38-rpc master requires --api 0.0.0.0:8082"));
    }

    if let Some(upstream) = args.qwen38_upstream_url.clone() {
        log::info!("using external Qwen3.8 llama.cpp server at {upstream}");
        return dial_core::qwen3_8_rpc::start_proxy(args, &upstream).await;
    }

    let gguf = qwen38_gguf_path(&args)?;
    let upstream = format!("http://127.0.0.1:{}", args.qwen38_upstream_port);

    if let Some(workers) = args.qwen38_rpc_workers.as_deref() {
        for endpoint in workers
            .split(',')
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            tokio::time::timeout(
                Duration::from_secs(3),
                tokio::net::TcpStream::connect(endpoint),
            )
            .await
            .map_err(|_| anyhow!("timed out connecting to Qwen3.8 RPC worker {endpoint}"))?
            .map_err(|error| anyhow!("cannot connect to Qwen3.8 RPC worker {endpoint}: {error}"))?;
        }
    } else {
        log::warn!(
            "--qwen38-rpc-workers is empty; Qwen3.8 will use only the Master's local devices"
        );
    }

    let mut command = Command::new(&args.qwen38_llama_server_bin);
    command
        .arg("--model")
        .arg(&gguf)
        .arg("--host")
        .arg("127.0.0.1")
        .arg("--port")
        .arg(args.qwen38_upstream_port.to_string())
        .arg("--alias")
        .arg("Qwen3.8-27B")
        .arg("--jinja")
        .arg("--reasoning")
        .arg(if args.qwen38_thinking { "on" } else { "off" })
        .arg("--gpu-layers")
        .arg("all")
        .arg("--split-mode")
        .arg("layer")
        .arg("--fit")
        .arg("on")
        .arg("--fit-target")
        .arg(args.qwen38_fit_target_mib.to_string())
        .arg("--ctx-size")
        .arg(args.kv_cache_max_len.to_string())
        .arg("--parallel")
        .arg("1")
        .stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());

    if let Some(workers) = args
        .qwen38_rpc_workers
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        command.arg("--rpc").arg(workers);
    }
    if let Some(split) = args.qwen38_tensor_split.as_deref() {
        command.arg("--tensor-split").arg(split);
    }
    if let Some(mmproj) = args.qwen38_mmproj.as_deref() {
        if !Path::new(mmproj).is_file() {
            return Err(anyhow!("Qwen3.8 mmproj file does not exist: {mmproj}"));
        }
        command.arg("--mmproj").arg(mmproj);
    }
    command.args(&args.qwen38_llama_arg);

    log::info!(
        "starting managed Qwen3.8 llama.cpp backend: model={} rpc={} context={}",
        gguf,
        args.qwen38_rpc_workers.as_deref().unwrap_or("none"),
        args.kv_cache_max_len
    );
    let mut child = command
        .spawn()
        .map_err(|error| anyhow!("failed to start {}: {error}", args.qwen38_llama_server_bin))?;
    if let Err(error) =
        wait_for_llama_server(&mut child, &upstream, args.qwen38_startup_timeout_s).await
    {
        let _ = child.kill().await;
        return Err(error);
    }
    log::info!("Qwen3.8 llama.cpp backend is ready at {upstream}");

    let proxy_args = args.clone();
    tokio::select! {
        result = dial_core::qwen3_8_rpc::start_proxy(proxy_args, &upstream) => {
            let _ = child.kill().await;
            result
        }
        status = child.wait() => {
            Err(anyhow!("managed Qwen3.8 llama-server exited: {}", status?))
        }
    }
}

async fn run_qwen38_rpc(args: Args) -> Result<()> {
    match args.mode {
        Mode::Master => run_qwen38_rpc_master(args).await,
        Mode::Worker => run_qwen38_rpc_worker(args).await,
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // parse command line
    let raw_args = std::env::args_os().collect::<Vec<_>>();
    let mut args = Args::parse_from(&raw_args);
    select_model_size(&mut args, &raw_args)?;

    // （新增）客户端模式：不加载本地模型、不启动 master/worker，只负责调用 API 并打印纯文本回复。
    if args.api_client.is_some() {
        return run_api_client(args).await;
    }

    #[cfg(target_arch = "aarch64")]
    if std::env::var_os("RAYON_NUM_THREADS").is_none() {
        // RK3588 has 4 big Cortex-A76 cores plus 4 small A55 cores. Candle/GEMM defaults to
        // all CPUs, which often slows single-token decode by scheduling matmul work on A55 cores.
        std::env::set_var("RAYON_NUM_THREADS", "4");
    }

    // setup logging
    if std::env::var_os("RUST_LOG").is_none() {
        // set `RUST_LOG=debug` to see debug logs
        std::env::set_var("RUST_LOG", "info,tokenizers=error,actix_server=warn");
    }

    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_module_path(false)
        .format_target(false)
        .init();

    log::info!("selected inference backend: {:?}", args.inference_backend);
    if args.inference_backend == InferenceBackend::Qwen38Ggml {
        log::info!(
            "Qwen3.8 GGML inputs: gguf={} adapter={}",
            args.qwen38_gguf.as_deref().unwrap_or("missing"),
            args.qwen38_ggml_lib.as_deref().unwrap_or("missing")
        );
    }

    #[cfg(all(target_arch = "aarch64", target_os = "linux"))]
    apply_cpu_affinity();

    // The old llama.cpp integration remains an explicit compatibility backend.
    // Qwen3.8 native execution continues below through the normal DIAL context,
    // topology, Master and Worker paths.
    if args.inference_backend == InferenceBackend::Qwen38Rpc {
        return run_qwen38_rpc(args).await;
    }

    let selected_model = detect_model_type(&args.model)?;

    // setup context
    let ctx = Context::from_args(args)?;
    log::info!("selected model: {:?}", selected_model);

    // run either in master or worker mode depending on command line
    let ret = match (ctx.args.mode.clone(), selected_model) {
        (Mode::Master, SelectedModel::Qwen3Vl) => {
            Master::<dial_core::models::qwen3_vl::Qwen3Vl>::new(ctx)
                .await?
                .run()
                .await
        }
        (Mode::Worker, SelectedModel::Qwen3Vl) => {
            Worker::<dial_core::models::qwen3_vl::Qwen3Vl>::new(ctx)
                .await?
                .run()
                .await
        }
        (Mode::Master, SelectedModel::Qwen38) => {
            Master::<dial_core::models::qwen3_8::Qwen38>::new(ctx)
                .await?
                .run()
                .await
        }
        (Mode::Worker, SelectedModel::Qwen38) => {
            Worker::<dial_core::models::qwen3_8::Qwen38>::new(ctx)
                .await?
                .run()
                .await
        }
        (Mode::Master, SelectedModel::Llama3) => {
            Master::<dial_core::models::llama3::LLama>::new(ctx)
                .await?
                .run()
                .await
        }
        (Mode::Worker, SelectedModel::Llama3) => {
            Worker::<dial_core::models::llama3::LLama>::new(ctx)
                .await?
                .run()
                .await
        }
    };

    if ret.is_err() {
        // we were possibly streaming text, add a newline before reporting the error
        println!();
        return ret;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_size_27b_selects_native_qwen38_backend() {
        let raw_args = ["dial-cli", "--model-size", "27b", "--mode", "worker"];
        let mut args = Args::try_parse_from(raw_args).unwrap();

        select_model_size(&mut args, raw_args).unwrap();
        assert_eq!(args.inference_backend, InferenceBackend::Qwen38Native);
    }

    #[test]
    fn model_size_8b_keeps_native_backend() {
        let raw_args = ["dial-cli", "--model-size", "8b", "--mode", "worker"];
        let mut args = Args::try_parse_from(raw_args).unwrap();

        select_model_size(&mut args, raw_args).unwrap();
        assert_eq!(args.inference_backend, InferenceBackend::Native);
    }

    #[test]
    fn model_size_27b_preserves_explicit_ggml_backend() {
        let raw = ["dial-cli", "--model-size", "27b", "--inference-backend", "qwen38-ggml",
            "--qwen38-gguf", "/model/q4.gguf", "--qwen38-ggml-lib", "/lib/libdial_qwen38_ggml.so"];
        let mut args = Args::try_parse_from(raw).unwrap();
        select_model_size(&mut args, raw).unwrap();
        assert_eq!(args.inference_backend, InferenceBackend::Qwen38Ggml);
    }

    #[test]
    fn model_size_27b_with_gguf_selects_ggml_backend() {
        let raw = [
            "dial-cli",
            "--model-size",
            "27b",
            "--qwen38-gguf",
            "/model/q4.gguf",
            "--qwen38-ggml-lib",
            "/lib/libdial_qwen38_ggml.so",
        ];
        let mut args = Args::try_parse_from(raw).unwrap();

        select_model_size(&mut args, raw).unwrap();
        assert_eq!(args.inference_backend, InferenceBackend::Qwen38Ggml);
    }

    #[test]
    fn model_size_27b_rejects_incomplete_ggml_selection() {
        let raw = [
            "dial-cli",
            "--model-size",
            "27b",
            "--qwen38-gguf",
            "/model/q4.gguf",
        ];
        let mut args = Args::try_parse_from(raw).unwrap();

        let error = select_model_size(&mut args, raw).unwrap_err();
        assert!(error.to_string().contains("--qwen38-ggml-lib"));
    }

    #[test]
    fn native_backend_rejects_ignored_gguf() {
        let raw = [
            "dial-cli",
            "--model-size",
            "27b",
            "--inference-backend",
            "qwen38-native",
            "--qwen38-gguf",
            "/model/q4.gguf",
        ];
        let mut args = Args::try_parse_from(raw).unwrap();

        let error = select_model_size(&mut args, raw).unwrap_err();
        assert!(error.to_string().contains("cannot be used with the native backend"));
    }

    #[test]
    fn model_size_8b_rejects_ggml_backend() {
        let raw = ["dial-cli", "--model-size", "8b", "--inference-backend", "qwen38-ggml"];
        let mut args = Args::try_parse_from(raw).unwrap();
        assert!(select_model_size(&mut args, raw).is_err());
    }

    #[test]
    fn qwen38_options_require_explicit_27b_selection() {
        let raw_args = ["dial-cli", "--qwen38-thinking", "false"];
        let mut args = Args::try_parse_from(raw_args).unwrap();

        let error = select_model_size(&mut args, raw_args).unwrap_err();
        assert!(error.to_string().contains("--model-size 27b"));
    }

    #[test]
    fn model_size_and_backend_conflicts_are_rejected() {
        let raw_args = [
            "dial-cli",
            "--model-size",
            "8b",
            "--inference-backend",
            "qwen38-rpc",
        ];
        let mut args = Args::try_parse_from(raw_args).unwrap();

        assert!(select_model_size(&mut args, raw_args).is_err());
    }
}
