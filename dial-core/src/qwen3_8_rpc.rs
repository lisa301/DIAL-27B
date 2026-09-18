//! DIAL API adapter for the Qwen3.8 llama.cpp/GGML-RPC backend.
//!
//! Qwen3.8 uses the Qwen3.5 hybrid Gated-DeltaNet/full-attention architecture.
//! It is intentionally isolated from the existing native Qwen3-VL backend so
//! selecting this module cannot change the behavior of existing deployments.

use std::{sync::Arc, time::Duration};

use actix_web::{
    error::ErrorBadGateway, http::StatusCode, web, App, HttpResponse, HttpServer, Responder,
};
use serde_json::{json, Map, Value};
use tokio::net::TcpStream;
use tokio_stream::StreamExt;

use crate::Args;

#[derive(Clone)]
struct ProxyState {
    http: reqwest::Client,
    upstream: String,
    model_alias: String,
    rpc_workers: Vec<String>,
    sample_len: usize,
    temperature: f64,
    top_p: Option<f64>,
    top_k: Option<usize>,
    repeat_penalty: f32,
    thinking: bool,
}

fn normalize_base_url(raw: &str) -> String {
    let raw = raw.trim().trim_end_matches('/');
    if raw.starts_with("http://") || raw.starts_with("https://") {
        raw.to_string()
    } else {
        format!("http://{raw}")
    }
}

fn rpc_workers(raw: Option<&str>) -> Vec<String> {
    raw.unwrap_or_default()
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .collect()
}

fn convert_media_parts(body: &mut Value) {
    let Some(messages) = body.get_mut("messages").and_then(Value::as_array_mut) else {
        return;
    };

    for message in messages {
        let Some(parts) = message.get_mut("content").and_then(Value::as_array_mut) else {
            continue;
        };
        for part in parts {
            let Some(kind) = part.get("type").and_then(Value::as_str) else {
                continue;
            };
            match kind {
                "image_base64" => {
                    let media_type = part
                        .get("media_type")
                        .and_then(Value::as_str)
                        .unwrap_or("image/jpeg");
                    let Some(data) = part.get("data").and_then(Value::as_str) else {
                        continue;
                    };
                    *part = json!({
                        "type": "image_url",
                        "image_url": {"url": format!("data:{media_type};base64,{data}")}
                    });
                }
                "video_base64" => {
                    let media_type = part
                        .get("media_type")
                        .and_then(Value::as_str)
                        .unwrap_or("video/mp4");
                    let Some(data) = part.get("data").and_then(Value::as_str) else {
                        continue;
                    };
                    *part = json!({
                        "type": "video_url",
                        "video_url": {"url": format!("data:{media_type};base64,{data}")}
                    });
                }
                _ => {}
            }
        }
    }
}

fn set_default(map: &mut Map<String, Value>, name: &str, value: Value) {
    if !map.contains_key(name) {
        map.insert(name.to_string(), value);
    }
}

fn prepare_request(mut body: Value, state: &ProxyState) -> anyhow::Result<Value> {
    convert_media_parts(&mut body);
    let map = body
        .as_object_mut()
        .ok_or_else(|| anyhow!("chat request must be a JSON object"))?;

    set_default(map, "model", json!(state.model_alias));
    set_default(map, "max_tokens", json!(state.sample_len));
    set_default(map, "temperature", json!(state.temperature));
    set_default(map, "repeat_penalty", json!(state.repeat_penalty));
    if let Some(top_p) = state.top_p {
        set_default(map, "top_p", json!(top_p));
    }
    if let Some(top_k) = state.top_k {
        set_default(map, "top_k", json!(top_k));
    }

    let template = map
        .entry("chat_template_kwargs")
        .or_insert_with(|| Value::Object(Map::new()));
    if let Some(template) = template.as_object_mut() {
        set_default(template, "enable_thinking", json!(state.thinking));
    }

    Ok(body)
}

async fn chat(state: web::Data<Arc<ProxyState>>, body: web::Json<Value>) -> HttpResponse {
    let body = match prepare_request(body.into_inner(), &state) {
        Ok(body) => body,
        Err(error) => return HttpResponse::BadRequest().json(json!({"error": error.to_string()})),
    };
    let stream_requested = body.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let url = format!("{}/v1/chat/completions", state.upstream);
    let upstream = match state.http.post(&url).json(&body).send().await {
        Ok(response) => response,
        Err(error) => {
            log::error!("Qwen3.8 upstream request failed: {error}");
            return HttpResponse::BadGateway().json(json!({
                "error": format!("Qwen3.8 llama.cpp upstream unavailable: {error}")
            }));
        }
    };

    let status =
        StatusCode::from_u16(upstream.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    let content_type = upstream
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or(if stream_requested {
            "text/event-stream"
        } else {
            "application/json"
        })
        .to_string();

    if !status.is_success() || !stream_requested {
        return match upstream.bytes().await {
            Ok(bytes) => HttpResponse::build(status)
                .content_type(content_type)
                .body(bytes),
            Err(error) => HttpResponse::BadGateway().json(json!({
                "error": format!("failed reading Qwen3.8 upstream response: {error}")
            })),
        };
    }

    let stream = upstream
        .bytes_stream()
        .map(|result| result.map_err(ErrorBadGateway));
    HttpResponse::build(status)
        .insert_header(("Content-Type", content_type))
        .insert_header(("Cache-Control", "no-cache"))
        .insert_header(("X-Accel-Buffering", "no"))
        .streaming(stream)
}

async fn health(state: web::Data<Arc<ProxyState>>) -> HttpResponse {
    let url = format!("{}/health", state.upstream);
    match state.http.get(url).send().await {
        Ok(response) if response.status().is_success() => HttpResponse::Ok().json(json!({
            "status": "ok",
            "backend": "qwen38-rpc",
            "upstream": state.upstream,
        })),
        Ok(response) => HttpResponse::ServiceUnavailable().json(json!({
            "status": "loading",
            "upstream_status": response.status().as_u16(),
        })),
        Err(error) => HttpResponse::ServiceUnavailable().json(json!({
            "status": "unavailable",
            "error": error.to_string(),
        })),
    }
}

async fn models(state: web::Data<Arc<ProxyState>>) -> HttpResponse {
    let url = format!("{}/v1/models", state.upstream);
    match state.http.get(url).send().await {
        Ok(response) => {
            let status =
                StatusCode::from_u16(response.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
            match response.bytes().await {
                Ok(bytes) => HttpResponse::build(status)
                    .content_type("application/json")
                    .body(bytes),
                Err(error) => HttpResponse::BadGateway().json(json!({
                    "error": format!("failed reading Qwen3.8 model list: {error}")
                })),
            }
        }
        Err(error) => HttpResponse::BadGateway().json(json!({
            "error": format!("Qwen3.8 llama.cpp model list unavailable: {error}")
        })),
    }
}

async fn topology(state: web::Data<Arc<ProxyState>>) -> impl Responder {
    let mut workers = Vec::with_capacity(state.rpc_workers.len());
    for (index, endpoint) in state.rpc_workers.iter().enumerate() {
        let online = tokio::time::timeout(
            Duration::from_millis(900),
            TcpStream::connect(endpoint.as_str()),
        )
        .await
        .map(|result| result.is_ok())
        .unwrap_or(false);
        workers.push(json!({
            "role": "worker",
            "name": format!("qwen38-rpc-worker-{index}"),
            "host": endpoint,
            "description": "llama.cpp GGML RPC CUDA worker",
            "layers": [],
            "active": true,
            "online": online,
            "state": if online { "online" } else { "offline" },
            "device": "cuda",
        }));
    }

    let upstream_online = state
        .http
        .get(format!("{}/health", state.upstream))
        .send()
        .await
        .map(|response| response.status().is_success())
        .unwrap_or(false);
    HttpResponse::Ok().json(json!({
        "backend": "qwen38-rpc",
        "model": state.model_alias,
        "master_api": "DIAL API proxy",
        "master": {
            "role": "master",
            "name": "Master",
            "description": "DIAL Qwen3.8 llama.cpp/RPC coordinator",
            "active": true,
            "online": upstream_online,
            "state": if upstream_online { "online" } else { "loading" },
            "device": "cuda",
        },
        "upstream": state.upstream,
        "configured_workers": workers.len(),
        "active_workers": workers.len(),
        "online_workers": workers.iter().filter(|worker| worker["online"] == true).count(),
        "reload_required": false,
        "workers": workers,
    }))
}

async fn web_chat() -> HttpResponse {
    HttpResponse::Ok()
        .content_type("text/html; charset=utf-8")
        .body(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../web_chat/index.html"
        )))
}

async fn web_chat_styles() -> HttpResponse {
    HttpResponse::Ok()
        .content_type("text/css; charset=utf-8")
        .body(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../web_chat/styles.css"
        )))
}

async fn web_chat_script() -> HttpResponse {
    HttpResponse::Ok()
        .content_type("application/javascript; charset=utf-8")
        .body(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../web_chat/app.js"
        )))
}

async fn not_found() -> HttpResponse {
    HttpResponse::NotFound().body("nope")
}

/// Start the DIAL-compatible API in front of a local or externally managed
/// llama.cpp server. Both DIAL's historical route and the standard OpenAI route
/// are exposed so existing clients keep working.
pub async fn start_proxy(args: Args, upstream: &str) -> anyhow::Result<()> {
    let address = args
        .api
        .as_deref()
        .ok_or_else(|| anyhow!("qwen38-rpc master requires --api <host:port>"))?
        .to_string();
    let upstream = normalize_base_url(upstream);
    let json_limit = args
        .video_max_bytes
        .saturating_mul(4)
        .saturating_div(3)
        .saturating_add(8 * 1024 * 1024);
    let state = Arc::new(ProxyState {
        http: reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .build()?,
        upstream,
        model_alias: "Qwen3.8-27B".to_string(),
        rpc_workers: rpc_workers(args.qwen38_rpc_workers.as_deref()),
        sample_len: args.sample_len,
        temperature: args.temperature,
        top_p: args.top_p,
        top_k: args.top_k,
        repeat_penalty: args.repeat_penalty,
        thinking: args.qwen38_thinking,
    });

    log::info!(
        "starting DIAL Qwen3.8 API on http://{} -> {}",
        address,
        state.upstream
    );
    HttpServer::new(move || {
        App::new()
            .app_data(web::Data::new(state.clone()))
            .app_data(web::JsonConfig::default().limit(json_limit))
            .route("/", web::get().to(web_chat))
            .route("/chat", web::get().to(web_chat))
            .route("/web-chat", web::get().to(web_chat))
            .route("/web-chat/styles.css", web::get().to(web_chat_styles))
            .route("/web-chat/app.js", web::get().to(web_chat_script))
            .route("/health", web::get().to(health))
            .route("/v1/models", web::get().to(models))
            .route("/api/v1/topology", web::get().to(topology))
            .route("/api/v1/chat/completions", web::post().to(chat))
            .route("/v1/chat/completions", web::post().to(chat))
            .default_service(web::route().to(not_found))
    })
    .bind(&address)
    .map_err(|error| anyhow!(error))?
    .run()
    .await
    .map_err(|error| anyhow!(error))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state() -> ProxyState {
        ProxyState {
            http: reqwest::Client::new(),
            upstream: "http://127.0.0.1:18083".to_string(),
            model_alias: "Qwen3.8-27B".to_string(),
            rpc_workers: vec![],
            sample_len: 128,
            temperature: 0.7,
            top_p: Some(0.8),
            top_k: Some(20),
            repeat_penalty: 1.0,
            thinking: false,
        }
    }

    #[test]
    fn converts_dial_image_base64_to_openai_data_url() {
        let request = json!({
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_base64", "media_type": "image/png", "data": "AAAA"},
                    {"type": "text", "text": "describe it"}
                ]
            }],
            "stream": true
        });
        let prepared = prepare_request(request, &state()).unwrap();
        assert_eq!(
            prepared["messages"][0]["content"][0]["image_url"]["url"],
            "data:image/png;base64,AAAA"
        );
        assert_eq!(prepared["model"], "Qwen3.8-27B");
        assert_eq!(prepared["chat_template_kwargs"]["enable_thinking"], false);
    }

    #[test]
    fn caller_sampling_values_are_not_overwritten() {
        let request = json!({
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0.1,
            "max_tokens": 9,
            "chat_template_kwargs": {"enable_thinking": true}
        });
        let prepared = prepare_request(request, &state()).unwrap();
        assert_eq!(prepared["temperature"], 0.1);
        assert_eq!(prepared["max_tokens"], 9);
        assert_eq!(prepared["chat_template_kwargs"]["enable_thinking"], true);
    }
}
