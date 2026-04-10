use std::str::FromStr;

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use safetensors::View;
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// 该结构体表示spm协议中的张量.
#[derive(Serialize, Debug, Deserialize)]
pub struct RawTensor {
    /// Tensor data.
    pub data: Vec<u8>,
    /// The data type as string.
    pub dtype: String,
    /// The tensor shape.
    pub shape: Vec<usize>,
}

impl RawTensor {
    /// 这里写RawTensor的方法。.
    pub fn from_tensor(x: &Tensor) -> Self {
        let data: Vec<u8> = x.data().to_vec();   /// 获取张量的原始二进制数据
        let dtype = x.dtype().as_str().to_string();  /// 获取张量的数据类型并转换为字符串
        let shape = x.shape().clone().into_dims();   /// 把类型消息变成可序列化的字符串
        Self { data, dtype, shape }   /// 用上面提取的3个字段构造RawTensor实例
    }

    /// 把接收到的RawTensor转换回Tensor，供后续计算使用。
    pub fn to_tensor(&self, device: &Device) -> Result<Tensor> {
        let dtype = DType::from_str(&self.dtype)?;
        Tensor::from_raw_buffer(&self.data, dtype, &self.shape, device).map_err(|e| anyhow!(e))
    }
}

/// 工作节点的诊断信息
#[derive(Serialize, Debug, Default, Deserialize)]
pub struct WorkerInfo {
    /// 通信协议版本.
    pub version: String,
    /// Tensor数据类型.
    pub dtype: String,
    /// 操作系统.
    pub os: String,
    /// 架构.
    pub arch: String,
    /// 设备.
    pub device: String,
    /// 多GPU时的卡号.
    pub device_idx: usize,
    /// 延迟.
    pub latency: u128,
}

/// 这是SPM分布式通信协议的消息类型.
#[derive(Serialize, Debug, Deserialize)]
/// 主从之间只能发送这几种消息，分别是Hello、WorkerInfo、SingleOp、Batch和Tensor。
pub enum Message {
    /// Hello握手消息.
    Hello,
    /// Worker把自己的信息发给Master.
    WorkerInfo(WorkerInfo),
    /// 单算子推理任务（发任务）.
    SingleOp {
        layer_name: String,  ///层名字
        x: RawTensor,   /// 张量数据
        index_pos: usize,  /// 位置
        block_idx: usize,  /// 共序号
    },
    /// 批量推理任务
    Batch {
        x: RawTensor,
        batch: Vec<(String, usize, usize)>,
    },
    /// Worker->Master回传计算结果.
    Tensor {
        x: RawTensor,
        compute_us: u64,
    },
}
/// 给Message这个枚举，实现两个实用函数。
impl Message {
    /// 创建任务消息Single_Op.
    pub fn single_op(layer_name: &str, x: &Tensor, index_pos: usize, block_idx: usize) -> Self {
        let layer_name = layer_name.to_owned();  /// 把&str转成string(所有权转移，网络传输必备)
        let x = RawTensor::from_tensor(x);   /// 把Tensor转换成RawTensor，方便序列化和网络传输
        /// 构建并返回Message::SingleOp
        Self::SingleOp {
            layer_name,
            x,
            index_pos,
            block_idx,
        }
    }

    /// 创建计算结果消息
    pub fn from_tensor(x: &Tensor) -> Self {
        Self::Tensor {
            x: RawTensor::from_tensor(x),
            compute_us: 0,
        }
    }

    /// 创建一个带计算耗时的Tensor结果消息.
    pub fn from_tensor_with_compute(x: &Tensor, compute_us: u64) -> Self {
        Self::Tensor {
            x: RawTensor::from_tensor(x),
            compute_us,
        }
    }

    /// 创建批量计算任务消息
    pub fn from_batch(x: &Tensor, batch: Vec<(String, usize, usize)>) -> Self {
        Self::Batch {
            x: RawTensor::from_tensor(x),
            batch,
        }
    }

    /// 把Message消息->二进制字节数组.
    fn to_bytes(&self) -> Result<Vec<u8>> {
        bitcode::serialize(self).map_err(|e| anyhow!(e))
    }

    /// 收到网络二进制->还原成Message消息.
    fn from_bytes(raw: &[u8]) -> Result<Self> {
        bitcode::deserialize(raw).map_err(|e| anyhow!(e))
    }

    /// 从网络流里读取一条完整的消息：先读校验码->读消息长度->读消息内容->反序列化成Message实例->返回.
    pub async fn from_reader<R>(reader: &mut R) -> Result<(usize, Self)>
    where
        R: AsyncReadExt + Unpin,
    {
        /// 读魔数，校验协议
        let magic = reader.read_u32().await?;
        if magic != super::PROTO_MAGIC {
            return Err(anyhow!("invalid magic value: {magic}"));
        }
        /// 读消息长度
        let req_size = reader.read_u32().await?;
        if req_size > super::MESSAGE_MAX_SIZE {
            return Err(anyhow!("request size {req_size} > MESSAGE_MAX_SIZE"));
        }
        /// 创建缓冲区，读取完整数据
        let mut req = vec![0_u8; req_size as usize];

        reader.read_exact(&mut req).await?;
        /// 解析成消息并返回
        Ok((req.len(), Self::from_bytes(&req)?))
    }

    /// 把消息序列化成二进制->按照[魔数+长度+数据]的协议格式->发送到网络流
    pub async fn to_writer<W>(&self, writer: &mut W) -> Result<usize>
    where
        W: AsyncWriteExt + Unpin,
    {
        /// 把消息序列化成二进制
        let req = self.to_bytes()?;
        let req_size = req.len() as u32;
        /// 安全校验
        if req_size > super::MESSAGE_MAX_SIZE {
            return Err(anyhow!("request size {req_size} > MESSAGE_MAX_SIZE"));
        }
        /// 写魔数、长度、数据，按协议格式发送
        writer.write_u32(super::PROTO_MAGIC).await?;
        writer.write_u32(req_size).await?;
        writer.write_all(&req).await?;
        /// 返回总字节数
        Ok(8 + req.len())
    }
}
