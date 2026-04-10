use anyhow::Result;
use async_trait::async_trait;
use candle_core::{Device, Tensor};
use std::{
    sync::{Mutex, OnceLock},
    time::Instant,
};
use tokio::net::TcpStream;

use crate::models::llama3::{Cache, Config};

use super::{Message, WorkerInfo};

#[derive(Debug, Clone, Default)]
pub struct DistributedProfile {
    pub remote_requests: usize,
    pub remote_total_s: f64,
    pub remote_compute_s: f64,
}

// 记录分布式统计信息：远程请求数、远程总耗时、远程计算耗时（秒）。
impl DistributedProfile {
    pub fn distributed_overhead_s(&self) -> f64 {
        (self.remote_total_s - self.remote_compute_s).max(0.0)
    }
}
// 
fn distributed_profile() -> &'static Mutex<DistributedProfile> {
    static PROFILE: OnceLock<Mutex<DistributedProfile>> = OnceLock::new();
    PROFILE.get_or_init(|| Mutex::new(DistributedProfile::default()))
}

pub fn reset_distributed_profile() {
    if let Ok(mut profile) = distributed_profile().lock() {
        *profile = DistributedProfile::default();
    }
}

pub fn snapshot_distributed_profile() -> DistributedProfile {
    distributed_profile()
        .lock()
        .map(|profile| profile.clone())
        .unwrap_or_default()
}

/// A client object used by the master to connect and orchestrate the workers.
/// From the spm perspective, each worker is a server and the master uses
/// multiple Client instances to connect to them.
#[derive(Debug)]
pub struct Client {
    device: Device,
    address: String,
    layer_name: String,
    stream: TcpStream,
    info: WorkerInfo,
    request_seq: u64,
}

impl Client {
    fn transfer_trace_enabled() -> bool {
        matches!(
            std::env::var("SPM_TRACE_TRANSFER").ok().as_deref(),
            Some("1") | Some("true") | Some("TRUE") | Some("yes") | Some("YES")
        )
    }

    fn message_summary(message: &Message) -> String {
        match message {
            Message::Hello => "hello".to_string(),
            Message::WorkerInfo(_) => "worker_info".to_string(),
            Message::SingleOp {
                layer_name,
                x,
                index_pos,
                block_idx,
            } => format!(
                "single layer={} index_pos={} block_idx={} shape={:?} dtype={}",
                layer_name, index_pos, block_idx, x.shape, x.dtype
            ),
            Message::Batch { x, batch } => {
                let first = batch.first().map(|(name, _, _)| name.as_str()).unwrap_or("-");
                let last = batch.last().map(|(name, _, _)| name.as_str()).unwrap_or("-");
                format!(
                    "batch ops={} first={} last={} shape={:?} dtype={}",
                    batch.len(),
                    first,
                    last,
                    x.shape,
                    x.dtype
                )
            }
            Message::Tensor { x, compute_us } => format!(
                "tensor shape={:?} dtype={} compute={:.3}ms",
                x.shape,
                x.dtype,
                *compute_us as f64 / 1000.0
            ),
        }
    }

    /// Connects to the given worker address.
    /// NOTE: device and layer_name here are only passed for std::fmt::Display.
    pub async fn new(device: Device, address: &str, layer_name: &str) -> Result<Self> {
        let address = address.to_string();
        let layer_name = layer_name.to_string();
        let stream = TcpStream::connect(&address)
            .await
            .map_err(|e| anyhow!("can't connect to {address}: {e}"))?;
        let worker_info = WorkerInfo::default();

        let mut client = Self {
            address,
            device,
            stream,
            layer_name,
            info: worker_info,
            request_seq: 0,
        };

        let resp = client.request(Message::Hello).await?;
        client.info = if let Message::WorkerInfo(info) = resp {
            info
        } else {
            return Err(anyhow!("unexpected worker info message: {:?}", &resp));
        };

        Ok(client)
    }

    /// Send a Message to the worker and return a response.
    async fn request(&mut self, req: Message) -> Result<Message> {
        self.request_seq += 1;
        let req_id = self.request_seq;
        let req_summary = Self::message_summary(&req);
        let total_start = Instant::now();

        let write_start = Instant::now();
        let written = req
            .to_writer(&mut self.stream)
            .await
            .map_err(|e| anyhow!("error sending message {:?}: {}", req, e))?;
        let write_time = write_start.elapsed();

        let read_start = Instant::now();
        let (read_size, msg) = super::Message::from_reader(&mut self.stream)
            .await
            .map_err(|e| anyhow!("error receiving response for {:?}: {}", req, e))?;
        let read_time = read_start.elapsed();
        let total_time = total_start.elapsed();

        if let (Ok(mut profile), Message::Tensor { compute_us, .. }) =
            (distributed_profile().lock(), &msg)
        {
            profile.remote_requests += 1;
            profile.remote_total_s += total_time.as_secs_f64();
            profile.remote_compute_s += *compute_us as f64 / 1_000_000.0;
        }

        if Self::transfer_trace_enabled() {
            log::info!(
                "[transfer][client {}#{}] {} -> write={}B/{:.3}ms read={}B/{:.3}ms total={:.3}ms resp={}",
                self.address,
                req_id,
                req_summary,
                written,
                write_time.as_secs_f64() * 1000.0,
                read_size,
                read_time.as_secs_f64() * 1000.0,
                total_time.as_secs_f64() * 1000.0,
                Self::message_summary(&msg)
            );
        }
        Ok(msg)
    }

    async fn forward_request(&mut self, req: Message) -> Result<Tensor> {
        let resp = self.request(req).await?;
        match resp {
            Message::Tensor { x, .. } => Ok(x.to_tensor(&self.device)?),
            _ => Err(anyhow!("unexpected response {:?}", &resp)),
        }
    }
}

impl std::fmt::Display for Client {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}@{} [{}<{}> {}-{} latency={}ms]",
            &self.layer_name,
            &self.address,
            &self.info.device,
            &self.info.device_idx,
            &self.info.os,
            &self.info.arch,
            self.info.latency
        )
    }
}

#[async_trait]
impl super::Forwarder for Client {
    fn load(_: String, _: candle_nn::VarBuilder, _: &Config) -> Result<Box<Self>> {
        Err(anyhow!("load should never be called on spm::Client"))
    }

    async fn forward(&self, _: &Tensor, _: usize, _: usize, _: &mut Cache) -> Result<Tensor> {
        Err(anyhow!(
            "immutable forward should never be called on spm::Client"
        ))
    }

    /// Executes the worker's pipeline for this tensor.
    async fn forward_mut(
        &mut self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        _: &mut Cache,
    ) -> Result<Tensor> {
        self.forward_request(super::Message::single_op(
            &self.layer_name,
            x,
            index_pos,
            block_idx,
        ))
        .await
    }

    /// 一次性发送多个层的批量计算任务，给远端服务器发一批算子一起执行，减少网络往返。
    async fn forward_batch(
        &mut self,
        x: &Tensor,   //输入张量
        batch: Vec<(String, usize, usize)>,   //批量算子信息：层名、位置、块号
        _: &mut Cache,
    ) -> Result<Tensor> {
        // 打包成Batch信息->发送远程请求->返回结果
        self.forward_request(super::Message::from_batch(x, batch))
            .await
    }
    // 返回客户端的唯一身份标识=服务器地址
    fn ident(&self) -> &str {
        &self.address
    }
    // 返回整个客户端负责的模型层名字
    fn layer_name(&self) -> &str {
        &self.layer_name
    }
}
