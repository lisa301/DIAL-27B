# Qwen3.8-27B 原生 DIAL 双节点部署

`--model-size 27b` 选择原生 `qwen38-native` 后端。模型的 64 个语言层继续由 DIAL 的 `topology.yml` 分配，Master/Worker 之间继续使用 DIAL 张量协议，性能接口继续统计远程请求、远程计算和通信开销。

当前实现包含：

- Qwen3.8 `qwen3_5` 配置解析；
- 48 个 Gated DeltaNet 线性注意力层；
- 16 个门控全注意力层；
- 每层 causal-conv、循环状态和 KV cache；
- zero-centered RMSNorm、门控注意力、MLP 和 ChatML 文本生成；
- 连续远程层合并为一个 DIAL batch 请求。

原生路径支持官方 BF16 safetensors，以及 ModelOpt 或 compressed-tensors 混合 NVFP4/FP8 safetensors。CUDA QuantLinear 会让量化权重保持 packed/raw 格式常驻 GPU，直接按 E2M1、FP8 分组 scale 和全局 scale 规则执行 W4A16/W8A16；不经过 GGUF、llama.cpp 或 vLLM。只有 `dense` 兼容模式会在加载时展开成 F16。

CUDA/F16 构建默认启用 `--qwen38-quant-linear auto`：NVFP4/FP8 权重以 packed/raw 格式常驻 GPU，由 DIAL 自定义 CUDA QuantLinear 直接计算，不再为量化矩阵创建 F16 副本。当前 kernel 是 correctness/reference CUDA kernel，主要先降低模型内存和权重带宽；`--qwen38-quant-linear dense` 可切回原先的 F16 展开路径做 A/B 对比，`cuda` 则在量化 kernel 不可用时直接退出。

图片和视频入口暂未接到 Qwen3.8 原生视觉编码器；当前先验证文本双节点推理。

## 节点规划

示例使用：

- `192.168.2.22`：Master，使用 `/home/nvidia/models/Qwen3.8-27B-NVFP4`，运行第 16-63 层、embedding、final norm 和 lm_head；
- `192.168.2.24`：Worker，使用 `/media/nvidia/Elements/Qwen3.8-27B` 的 BF16 权重，运行第 0-15 层；
- 两边的模型结构和 tokenizer 必须属于同一个 Qwen3.8-27B 基座，但 safetensors 的存储精度可以不同。

拓扑文件 `topology_qwen38.yml`：

```yaml
worker0:
  host: "192.168.2.24:10128"
  description: "Qwen3.8-27B native DIAL shard on the second NVIDIA node"
  layers:
    - "model.language_model.layers.0-15"
```

## 检查现有模型

不需要再复制或转换模型。分别检查两边的索引和分片是否齐全：

```bash
# 192.168.2.22：应看到 3 个 NVIDIA NVFP4 分片
find /home/nvidia/models/Qwen3.8-27B-NVFP4 -maxdepth 1 \
  -name 'model-*.safetensors' | wc -l
test -f /home/nvidia/models/Qwen3.8-27B-NVFP4/model.safetensors.index.json

# 192.168.2.24：应看到 18 个 BF16 分片
find /media/nvidia/Elements/Qwen3.8-27B -maxdepth 1 \
  -name 'model-*.safetensors' | wc -l
test -f /media/nvidia/Elements/Qwen3.8-27B/model.safetensors.index.json
```

## 编译 DIAL

将更新后的 DIAL 代码同步到两台设备。`.22` 根分区空间充足，可直接编译：

```bash
cd /home/nvidia/Documents/Dial_llama
cargo build --release --features cuda
```

`.24` 的根分区只剩约 1.8 GB，构建产物应放到移动硬盘：

```bash
cd /home/nvidia/Dial_llama
CARGO_TARGET_DIR=/media/nvidia/Elements/dial-target \
  cargo build --release --features cuda
```

帮助信息应包含：

```bash
/home/nvidia/Documents/Dial_llama/target/release/dial-cli --help \
  | grep -A8 inference-backend
```

其中应出现 `qwen38-native`。

在启动27B模型前，两台 CUDA 节点都先运行小矩阵自测：

```bash
cargo run --release --features cuda -p dial-core \
  --example qwen38_quant_smoke
```

必须同时看到 decode (`rows=1`) 和 prefill (`rows=7`) 的 `NVFP4`、`FP8` 误差小于 `0.06`，以及：

```text
Qwen3.8 CUDA QuantLinear smoke passed
```

如果自测失败，不要启动模型；保留完整错误日志。临时回退时在启动命令中改为 `--qwen38-quant-linear dense`。

## 启动 Worker（192.168.2.24）

```bash
cd /home/nvidia/Dial_llama

/media/nvidia/Elements/dial-target/release/dial-cli \
  --model-size 27b \
  --mode worker \
  --name worker0 \
  --address 0.0.0.0:10128 \
  --model /media/nvidia/Elements/Qwen3.8-27B \
  --topology /home/nvidia/Dial_llama/topology_qwen38.yml \
  --dtype f16 \
  --qwen38-quant-linear dense \
  --kv-cache-max-len 4096
```

## 启动 Master（192.168.2.22）

先检查 Worker：

```bash
nc -vz 192.168.2.24 10128
```

然后启动 Master：

```bash
cd /home/nvidia/Documents/Dial_llama

./target/release/dial-cli \
  --model-size 27b \
  --mode master \
  --api 0.0.0.0:8082 \
  --model /home/nvidia/models/Qwen3.8-27B-NVFP4 \
  --topology /home/nvidia/Documents/Dial_llama/topology_qwen38.yml \
  --dtype f16 \
  --qwen38-quant-linear cuda \
  --kv-cache-max-len 4096 \
  --qwen38-thinking false
```

## 请求和观测

```bash
./target/release/dial-cli \
  --api-client http://127.0.0.1:8082 \
  --ask "1+1等于多少？"
```

```bash
curl http://127.0.0.1:8082/api/v1/topology
```

Master 的响应和日志继续包含 `distributed_overhead_s`、`remote_compute_s` 和 `remote_requests`。启动日志中应看到 `Qwen3.8 quantized checkpoint detected`，以及各层指向 `worker0@192.168.2.24:10128`。这条路径不需要 llama.cpp、GGUF 或 vLLM。
