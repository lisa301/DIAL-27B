use std::{
    collections::HashMap,
    net::SocketAddr,
    sync::Arc,
    time::{Duration, Instant},
};

use super::{Context, Forwarder, Message, WorkerInfo};
use crate::models::{llama3::Cache, Generator};

use anyhow::Result;
use candle_core::{DType, Device};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
};

/// 每执行5次推理/操作，就输出一次worker统计日志。
const NUM_OPS_TO_STATS: usize = 5;

/// 一个独立的工作节点.
#[derive(Clone)]=
struct WorkerContext<F> {
    device: Device,
    device_idx: usize,
    dtype: DType,
    blocks: Arc<HashMap<String, Box<F>>>,
    cache: Cache,
}
/// AI推理工作节点的方法，用于向master报告自己的状态和性能指标。
impl<F: Forwarder> WorkerContext<F> {
    ///创建WorkInfo结构，发送给主节点（master）.
    fn to_info(&self, latency: u128) -> WorkerInfo {
        WorkerInfo {
            version: env!("CARGO_PKG_VERSION").to_string(),
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
            device: if self.device.is_cuda() {
                "cuda".to_string()
            } else if self.device.is_metal() {
                "metal".to_string()
            } else {
                "cpu".to_string()
            },
            device_idx: self.device_idx,
            latency,
            dtype: format!("{:?}", self.dtype),
        }
    }

    /// Create a copy of self with new kv-cache.
    /// 复制自己+换新自己->给客户端使用
    fn get_client_context(&self) -> Self {
        WorkerContext {
            device: self.device.clone(),
            device_idx: self.device_idx,
            dtype: self.dtype,
            blocks: self.blocks.clone(),
            // each client loop gets a new cache
            /// 重点：每个客户端 = 新缓存！
            cache: self.cache.as_new(),
        }
    }
}

/// 定义结构体——工作节点.
pub struct Worker<G: Generator> {
    listener: TcpListener,
    context: WorkerContext<G::Shardable>,
}
/// 给Worker这个结构体实现方法
impl<G: Generator + 'static> Worker<G> {
    /// 判断是否开启传输跟踪日志
    fn transfer_trace_enabled() -> bool {
        matches!(
            std::env::var("SPM_TRACE_TRANSFER").ok().as_deref(),
            Some("1") | Some("true") | Some("TRUE") | Some("yes") | Some("YES")
        )
    }
    /// 生成操作统计摘要
    fn ops_summary(ops: &[(String, usize, usize)]) -> String {
        let first = ops.first().map(|(name, _, _)| name.as_str()).unwrap_or("-");
        let last = ops.last().map(|(name, _, _)| name.as_str()).unwrap_or("-");
        format!("ops={} first={} last={}", ops.len(), first, last)
    }

    /// Create a new Worker from the context.
    /// 整个AI worker最核心的初始化函数！
    /// 创建并启动一个AI推理工作节点的服务，加载模型、绑定端口、准备就绪等待客户端请求。
    pub async fn new(ctx: Context) -> Result<Self> {
        /// 获取工作节点名称
        let worker_name = if let Some(name) = &ctx.args.name {
            name.to_string()
        } else {
            return Err(anyhow!("no --name provided for worker"));
        };
        /// 获取节点的拓扑（模型层分配）
        let worker_topology = if let Some(node) = ctx.topology.get(&worker_name) {
            node
        } else if !ctx.topology.is_empty() {
            let first = ctx.topology.keys().next().unwrap();
            log::warn!(
                "topology for worker name '{}' not found, using '{}'",
                &worker_name,
                first
            );
            ctx.topology.get(first).unwrap()
        } else {
            return Err(anyhow!(
                "could not find topology for {worker_name} and topology file is empty"
            ));
        };
        /// 加载模型层（blocks)
        let mut blocks = HashMap::new();
        // 在这显示加载了哪些块
        for block_layer_name in &worker_topology.layers {
            log::info!("loading {} ...", &block_layer_name);

            let block = G::Shardable::load(
                block_layer_name.to_string(),
                ctx.var_builder.pp(block_layer_name),
                &ctx.config,
            )?;
            blocks.insert(block_layer_name.to_string(), block);
        }

        let blocks = Arc::new(blocks);
        /// 绑定TCP端口，启动监听
        let listener = TcpListener::bind(&ctx.args.address).await?;
        /// 打印启动日志
        log::info!(
            "listening on {} (mem:{}) ...",
            &ctx.args.address,
            human_bytes::human_bytes(memory_stats::memory_stats().unwrap().physical_mem as f64)
        );

        let cache = ctx.cache;
        let device = ctx.device;
        let dtype = ctx.dtype;
        let device_idx = ctx.args.device;
        /// 组装Worker 实例
        let context = WorkerContext {
            device,
            device_idx,
            dtype,
            blocks,
            cache,
        };
        /// 返回worker实例
        Ok(Self { listener, context })
    }

    /// Read a message from the socket and return elapsed time, message size and message.
    /// 异步网络工具函数，从socket读取消息并返回经过的时间、消息大小和消息。
    async fn read_message_timed<R>(mut socket: R) -> Result<(Duration, usize, Message)>
    where
        R: AsyncReadExt + Unpin,   /// 这个函数能处理任何异步可读的数据流（TCP、管道、文件等）
    {
        let start = Instant::now();
        let (size, message) = Message::from_reader(&mut socket).await?;  /// 异步读取消息
        let latency = start.elapsed();

        Ok((latency, size, message))   /// 返回结果
    }

    /// Write a message to the socket and return the elapsed time with written size.
    /// 异步网络工具函数，将消息写入socket并返回经过的时间与写入的大小。
    async fn write_message_timed<W>(mut socket: W, message: Message) -> Result<(Duration, usize)>
    where
        W: AsyncWriteExt + Unpin,
    {
        let start = Instant::now();
        let size = message.to_writer(&mut socket).await?;
        let latency = start.elapsed();

        Ok((latency, size))
    }

    /// Main loop handling communication with the master.
    /// 这个函数是整个Worker工作节点的开头，专门负责和Master建立连接、握手、验证身份
    /// 是分布式AI服务的安全＋通信入口
    async fn handle_master_client(
        mut socket: TcpStream,　　　/// TCP连接
        client: SocketAddr,        /// 主节点地址
        mut context: WorkerContext<G::Shardable>,
    ) -> Result<()> {
        // 读取Master发来的第一条消息，必须是Hello握手包
        let (latency, _size, hello) = Self::read_message_timed(&mut socket).await?;
        if !matches!(hello, Message::Hello) {
            return Err(anyhow!(
                "[{}] unpexpected message instead of hello: {:?}",
                &client,
                hello
            ));
        }

        // 发送worker信息给Master
        if let Err(e) = Self::write_message_timed(
            &mut socket,
            Message::WorkerInfo(context.to_info(latency.as_millis())),
        )
        .await   //异步发送
        {
            return Err(anyhow!("[{}] could not send worker info: {:?}", &client, e));
        }

        let mut msg_idx = 0;
        let mut avg_ops = 0;
        let mut avg_write = 0;
        let mut avg_read = 0;

        // 持续读取消息
        while let Ok((read_time, read_size, op_message)) =
            Self::read_message_timed(&mut socket).await
        {
            let req_start = Instant::now();   /// 记录请求开始时间
            let (x, ops) = match op_message {
                /// 单操作请求
                Message::SingleOp {
                    layer_name,
                    x,
                    index_pos,
                    block_idx,
                } => (x, vec![(layer_name, index_pos, block_idx)]),
                /// 批量操作请求
                Message::Batch { x, batch } => (x, batch),
                _ => {
                    return Err(anyhow!(
                        "[{}] unhandled message in loop: {:?}",
                        &client,
                        op_message
                    ));
                }
            };
            let ops_summary = Self::ops_summary(&ops);

            // （新增）这里避免使用 `unwrap()`：
            // 为什么要加：一旦出现协议/数据不一致或 shape 错误，`unwrap()` 会直接 panic 把 worker 进程干掉；
            // 改成返回带上下文的错误，让 master 能拿到错误信息并继续运行/重试。
            //
            // 解码张量并统计耗时
            let decode_start = Instant::now();
            let mut x = x
                .to_tensor(&context.device)
                .map_err(|e| anyhow!("[{}] could not decode tensor: {e}", &client))?;
            let decode_time = decode_start.elapsed();
            /// 本次要执行的模型层数
            let num_ops = ops.len();
            let start_ops = Instant::now();

            // 遍历所有要执行的模型层
            for (layer_name, index_pos, block_idx) in ops {
                // 根据模型层名获取模型层
                if let Some(block) = context.blocks.get(&layer_name) {
                    // （新增）同样避免 `unwrap()`：把 layer/index_pos/block_idx 打进错误里，方便定位是哪一层/哪一步出错。
                    // forward 前向传播
                    x = block
                        .forward(&x, index_pos, block_idx, &mut context.cache)
                        .await
                        .map_err(|e| {
                            anyhow!(
                                "[{}] forward failed for {} (index_pos={}, block_idx={}): {e}",
                                &client,
                                layer_name,
                                index_pos,
                                block_idx
                            )
                        })?;
                } else {
                    return Err(anyhow!("could not find layer {}", &layer_name));
                }
            }

            let elaps_ops = start_ops.elapsed();

            // 发送推理结果张量
            let compute_us = elaps_ops.as_micros().min(u128::from(u64::MAX)) as u64;
            /// 异步发送结果
            match Self::write_message_timed(
                &mut socket,
                Message::from_tensor_with_compute(&x, compute_us),
            )
            .await
            /// 处理发送结果
            {
                Ok((elaps_write, written)) => {
                    /// 发送成功：统计性能+打印日志
                    let ops_per_sec = (num_ops as f64 / elaps_ops.as_secs_f64()) as usize;
                    let write_bytes_per_sec = (written as f64 / elaps_write.as_secs_f64()) as usize;
                    let read_bytes_per_sec = (read_size as f64 / read_time.as_secs_f64()) as usize;
                    // 累计平均值
                    avg_ops += ops_per_sec;
                    avg_write += write_bytes_per_sec;
                    avg_read += read_bytes_per_sec;
                    /// 打印详细的日志
                    if Self::transfer_trace_enabled() {
                        log::info!(
                            "[transfer][worker {} msg={}] {} read={}B/{:.3}ms decode={:.3}ms compute={:.3}ms write={}B/{:.3}ms total={:.3}ms",
                            &client,
                            msg_idx,
                            ops_summary,
                            read_size,
                            read_time.as_secs_f64() * 1000.0,
                            decode_time.as_secs_f64() * 1000.0,
                            elaps_ops.as_secs_f64() * 1000.0,
                            written,
                            elaps_write.as_secs_f64() * 1000.0,
                            req_start.elapsed().as_secs_f64() * 1000.0
                        );
                    }
                }
                Err(e) => {
                    return Err(anyhow!(
                        "[{}] could not send response tensor: {:?}",
                        &client,
                        e
                    ));
                }
            }

            ///  每处理N条消息，计算并打印一次统计，避免刷屏stdout.
            if msg_idx % NUM_OPS_TO_STATS == 0 {
                log::info!(
                    "ops={}/s read={}/s write={}/s",
                    avg_ops / NUM_OPS_TO_STATS,  /// 平均推理速度
                    human_bytes::human_bytes(avg_read as f64 / NUM_OPS_TO_STATS as f64),  /// 平均读速度
                    human_bytes::human_bytes(avg_write as f64 / NUM_OPS_TO_STATS as f64)  /// 平均写速度
                );
                avg_ops = 0;
                avg_write = 0;
                avg_read = 0;
            }
            msg_idx += 1;
        }

        Ok(())
    }

    /// 运行工作节点服务器的accept循环，等待并处理来自主节点的连接请求。
    pub async fn run(&mut self) -> Result<()> {
        while let Ok((socket, client)) = self.listener.accept().await {
            log::info!("{} connected", &client);
            /// 给这个客户端复制一份独立上下文
            let context = self.context.get_client_context();
            tokio::spawn(async move {
                /// 处理连接，出错只打印日志，不崩溃整个服务
                if let Err(e) = Self::handle_master_client(socket, client, context).await {
                    log::error!("{}", e);
                }
            });
        }

        Ok(())
    }
}
