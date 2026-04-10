use std::sync::Arc;
use std::time::Instant;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;

use actix_web::web;
use actix_web::App;
use actix_web::HttpRequest;
use actix_web::HttpResponse;
use actix_web::HttpServer;
use actix_web::Responder;
use serde::Deserialize;
use serde::Serialize;
use tokio::sync::mpsc;
use tokio::sync::RwLock;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tokio_stream::StreamExt;

use crate::models::chat::Message;
use crate::models::Generator;

use super::{snapshot_distributed_profile, Master};

#[derive(Deserialize)]
struct Request {
    pub messages: Vec<Message>,
    /// OpenAI-style streaming responses over SSE.
    #[serde(default)]
    pub stream: bool,
}

#[derive(Serialize)]
struct Choice {
    pub index: usize,
    pub message: Message,
}

#[derive(Serialize)]
struct Response {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    /// （新增）首 token 延迟（TTFT, time-to-first-token），单位秒。
    /// 说明：包含多模态图片编码、prefill，以及生成出第一个 token 的总耗时。
    pub ttft_s: Option<f64>,
    /// （新增）本次请求总耗时，单位秒（从开始生成到结束）。
    pub total_s: f64,
    /// （新增）平均生成速率（tokens/s），使用生成的 token 数 / total_s 计算。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokens_per_second: Option<f64>,
    /// （新增）解码阶段平均生成速率（tokens/s），使用 (生成token数-1) / (total_s-ttft_s) 计算。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decode_tokens_per_second: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub distributed_overhead_s: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub remote_compute_s: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub remote_requests: Option<usize>,
}

#[derive(Serialize)]
struct StreamDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

#[derive(Serialize)]
struct StreamChoice {
    pub index: usize,
    pub delta: StreamDelta,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

#[derive(Serialize)]
struct StreamResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<StreamChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ttft_s: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_s: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokens_per_second: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decode_tokens_per_second: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub distributed_overhead_s: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub remote_compute_s: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub remote_requests: Option<usize>,
}

impl Response {
    pub fn from_assistant_response(
        model: String,
        message: String,
        ttft_s: Option<f64>,
        total_s: f64,
        tokens_per_second: Option<f64>,
        decode_tokens_per_second: Option<f64>,
        distributed_overhead_s: Option<f64>,
        remote_compute_s: Option<f64>,
        remote_requests: Option<usize>,
    ) -> Self {
        let id = uuid::Uuid::new_v4().to_string();
        let object = String::from("chat.completion");
        let created = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let choices = vec![Choice {
            index: 0,
            message: Message::assistant(message),
        }];

        Self {
            id,
            object,
            created,
            model,
            choices,
            ttft_s,
            total_s,
            tokens_per_second,
            decode_tokens_per_second,
            distributed_overhead_s,
            remote_compute_s,
            remote_requests,
        }
    }
}

async fn chat<G>(
    state: web::Data<Arc<RwLock<Master<G>>>>,
    req: HttpRequest,
    messages: web::Json<Request>,
) -> impl Responder
where
    G: Generator + Send + Sync + 'static,
{
    let client = req
        .peer_addr()
        .map(|a| a.to_string())
        .unwrap_or_else(|| "<unknown>".to_string());

    log::info!("starting chat for {} ...", &client);

    let Request { messages, stream } = messages.into_inner();

    if !stream {
        let mut master = state.write().await;

        if let Err(e) = master.reset() {
            log::error!("reset failed for {}: {}", &client, &e);
            return HttpResponse::InternalServerError().body(format!("reset failed: {e}"));
        }

        for message in messages {
            if let Err(e) = master.model.add_message(message) {
                log::warn!("invalid message from {}: {}", &client, &e);
                return HttpResponse::BadRequest().body(format!("invalid message: {e}"));
            }
        }

        let mut resp = String::new();
        let start = Instant::now();
        let mut ttft_s: Option<f64> = None;

        let mut generated_tokens: usize = 0;

        if let Err(e) = master
            .generate(|data| {
                // 记录首 token 时间（只要收到第一个非空 chunk，就认为首 token 已产生）。
                if ttft_s.is_none() && !data.is_empty() {
                    ttft_s = Some(start.elapsed().as_secs_f64());
                }
                generated_tokens += 1;
                resp += data;
            })
            .await
        {
            log::error!("generation failed for {}: {}", &client, &e);
            return HttpResponse::InternalServerError().body(format!("generation failed: {e}"));
        }

        let total_s = start.elapsed().as_secs_f64();
        let tokens_per_second = if total_s > 0.0 {
            Some(generated_tokens as f64 / total_s)
        } else {
            None
        };
        let decode_tokens_per_second = match ttft_s {
            Some(ttft) if total_s > ttft && generated_tokens > 1 => {
                let decode_tokens = generated_tokens.saturating_sub(1) as f64;
                let decode_s = total_s - ttft;
                if decode_s > 0.0 {
                    Some(decode_tokens / decode_s)
                } else {
                    None
                }
            }
            _ => None,
        };

        let response = Response::from_assistant_response(
            G::MODEL_NAME.to_string(),
            resp,
            ttft_s,
            total_s,
            tokens_per_second,
            decode_tokens_per_second,
            Some(snapshot_distributed_profile().distributed_overhead_s()),
            Some(snapshot_distributed_profile().remote_compute_s),
            Some(snapshot_distributed_profile().remote_requests),
        );

        let dist = snapshot_distributed_profile();

        // （新增）服务端日志也打印一份，方便不看 JSON 的情况下观察首 token 与总耗时。
        log::info!(
            "metrics for {}: ttft_s={} total_s={:.3} tps={} decode_tps={} dist_overhead_s={:.3} remote_compute_s={:.3} remote_requests={}",
            &client,
            ttft_s
                .map(|v| format!("{v:.3}"))
                .unwrap_or_else(|| "null".to_string()),
            total_s,
            tokens_per_second
                .map(|v| format!("{v:.3}"))
                .unwrap_or_else(|| "null".to_string()),
            decode_tokens_per_second
                .map(|v| format!("{v:.3}"))
                .unwrap_or_else(|| "null".to_string()),
            dist.distributed_overhead_s(),
            dist.remote_compute_s,
            dist.remote_requests
        );

        return HttpResponse::Ok().json(response);
    }

    // Streaming mode (SSE). Server does not print tokens; client renders them as they arrive.
    let (tx, rx) = mpsc::unbounded_channel::<String>();
    let state = state.clone();
    let client_for_task = client.clone();
    let model = G::MODEL_NAME.to_string();

    tokio::spawn(async move {
        let send = |tx: &mpsc::UnboundedSender<String>, payload: &str| {
            let _ = tx.send(payload.to_string());
        };

        let id = uuid::Uuid::new_v4().to_string();
        let created = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let mut master = state.write().await;

        if let Err(e) = master.reset() {
            log::error!("reset failed for {}: {}", &client_for_task, &e);
            let j = serde_json::json!({ "error": format!("reset failed: {e}") }).to_string();
            send(&tx, &format!("data: {j}\n\n"));
            send(&tx, "data: [DONE]\n\n");
            return;
        }

        for message in messages {
            if let Err(e) = master.model.add_message(message) {
                log::warn!("invalid message from {}: {}", &client_for_task, &e);
                let j = serde_json::json!({ "error": format!("invalid message: {e}") }).to_string();
                send(&tx, &format!("data: {j}\n\n"));
                send(&tx, "data: [DONE]\n\n");
                return;
            }
        }

        let mut resp = String::new();
        let start = Instant::now();
        let mut ttft_s: Option<f64> = None;
        let mut generated_tokens: usize = 0;

        let gen = master
            .generate(|data| {
                // End-of-stream marker from Master::generate.
                if data.is_empty() {
                    return;
                }

                if ttft_s.is_none() {
                    ttft_s = Some(start.elapsed().as_secs_f64());
                }

                resp.push_str(data);
                generated_tokens += 1;

                let chunk = StreamResponse {
                    id: id.clone(),
                    object: "chat.completion.chunk".to_string(),
                    created,
                    model: model.clone(),
                    choices: vec![StreamChoice {
                        index: 0,
                        delta: StreamDelta {
                            content: Some(data.to_string()),
                        },
                        finish_reason: None,
                    }],
                    ttft_s: None,
                    total_s: None,
                    tokens_per_second: None,
                    decode_tokens_per_second: None,
                    distributed_overhead_s: None,
                    remote_compute_s: None,
                    remote_requests: None,
                };

                match serde_json::to_string(&chunk) {
                    Ok(j) => send(&tx, &format!("data: {j}\n\n")),
                    Err(e) => {
                        log::error!("failed to serialize stream chunk: {e}");
                        send(
                            &tx,
                            "data: {\"error\":\"internal serialization error\"}\n\n",
                        );
                    }
                }
            })
            .await;

        match gen {
            Ok(()) => {
                let total_s = start.elapsed().as_secs_f64();
                let tokens_per_second = if total_s > 0.0 {
                    Some(generated_tokens as f64 / total_s)
                } else {
                    None
                };
                let decode_tokens_per_second = match ttft_s {
                    Some(ttft) if total_s > ttft && generated_tokens > 1 => {
                        let decode_tokens = generated_tokens.saturating_sub(1) as f64;
                        let decode_s = total_s - ttft;
                        if decode_s > 0.0 {
                            Some(decode_tokens / decode_s)
                        } else {
                            None
                        }
                    }
                    _ => None,
                };
                let final_chunk = StreamResponse {
                    id: id.clone(),
                    object: "chat.completion.chunk".to_string(),
                    created,
                    model: model.clone(),
                    choices: vec![StreamChoice {
                        index: 0,
                        delta: StreamDelta { content: None },
                        finish_reason: Some("stop".to_string()),
                    }],
                    ttft_s,
                    total_s: Some(total_s),
                    tokens_per_second,
                    decode_tokens_per_second,
                    distributed_overhead_s: Some(snapshot_distributed_profile().distributed_overhead_s()),
                    remote_compute_s: Some(snapshot_distributed_profile().remote_compute_s),
                    remote_requests: Some(snapshot_distributed_profile().remote_requests),
                };
                let dist = snapshot_distributed_profile();
                if let Ok(j) = serde_json::to_string(&final_chunk) {
                    send(&tx, &format!("data: {j}\n\n"));
                }

                log::info!(
                    "metrics for {}: ttft_s={} total_s={:.3} tps={} decode_tps={} dist_overhead_s={:.3} remote_compute_s={:.3} remote_requests={}",
                    &client_for_task,
                    ttft_s
                        .map(|v| format!("{v:.3}"))
                        .unwrap_or_else(|| "null".to_string()),
                    total_s,
                    tokens_per_second
                        .map(|v| format!("{v:.3}"))
                        .unwrap_or_else(|| "null".to_string()),
                    decode_tokens_per_second
                        .map(|v| format!("{v:.3}"))
                        .unwrap_or_else(|| "null".to_string()),
                    dist.distributed_overhead_s(),
                    dist.remote_compute_s,
                    dist.remote_requests
                );
            }
            Err(e) => {
                log::error!("generation failed for {}: {}", &client_for_task, &e);
                let j =
                    serde_json::json!({ "error": format!("generation failed: {e}") }).to_string();
                send(&tx, &format!("data: {j}\n\n"));
            }
        }

        send(&tx, "data: [DONE]\n\n");
    });

    let body = UnboundedReceiverStream::new(rx)
        .map(|s| Ok::<web::Bytes, actix_web::Error>(web::Bytes::from(s)));

    HttpResponse::Ok()
        .insert_header(("Content-Type", "text/event-stream"))
        .insert_header(("Cache-Control", "no-cache"))
        .insert_header(("Connection", "keep-alive"))
        .streaming(body)
}

async fn not_found() -> actix_web::Result<HttpResponse> {
    Ok(HttpResponse::NotFound().body("nope"))
}

pub(crate) async fn start<G>(master: Master<G>) -> anyhow::Result<()>
where
    G: Generator + Send + Sync + 'static,
{
    let address = master.ctx.args.api.as_ref().unwrap().to_string();

    log::info!("starting api on http://{} ...", &address);

    let state = Arc::new(RwLock::new(master));

    HttpServer::new(
        move || {
            App::new()
                .app_data(web::Data::new(state.clone()))
                .route("/api/v1/chat/completions", web::post().to(chat::<G>))
                .default_service(web::route().to(not_found))
        }, //.wrap(actix_web::middleware::Logger::default()))
    )
    .bind(&address)
    .map_err(|e| anyhow!(e))?
    .run()
    .await
    .map_err(|e| anyhow!(e))
}
