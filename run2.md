# 一、服务器端

## 1、执行指令：

```raw
//挂载硬盘
sudo mount /dev/sda1 /media/nvidia/Elements/

cargo clean #清除缓存
cargo build --release --features cuda  #有GPU＋cuda，没有去掉
QT_QUICK_BACKEND=software /home/firefly/iSure-build-board/iSure  #qt启动

RUST_LOG=info SPM_TRACE_TRANSFER=1 SPM_TRANSFER_LIMIT_MBPS=100 ./target/release/dial-cli  --mode worker
```

### 2、model地址

```raw
/// qwen3-vl-8B模型   ,transformers一共36层，需修改topology.yml
/media/nvidia/Elements/Qwen3-vl/Qwen3-VL-8B-Instruct


/// qwen3-vl-2B模型  ,transformers一共28层，需修改topology.yml
/media/nvidia/Elements/Qwen3-vl/Qwen3-VL-2B-Instruct
```

### 3、yml地址

```raw
/home/nvidia/Dial_llama/topology_qwen3vl.yml
```

# 二、客户端执行指令：

### 1、终端1：（与服务器连通）

```raw
cargo clean #清除缓存
cargo build --release  #有GPU＋cuda，没有去掉


./target/release/dial-cli --api 0.0.0.0:8082
```

```raw
/// vision-max-side限制图片所占的token数，如果太大的话，首token会很慢
./target/release/dial-cli \
  --mode master \
  --api 0.0.0.0:8082 \
  --vision-max-side 512 \
  --vision-no-upscale
```

### 2、终端2：（大模型问答推理）

```raw
./target/release/dial-cli  --api-client http://127.0.0.1:8082 --ask "1+1等于多少"


  ./target/release/dial-cli  --api-client http://127.0.0.1:8082 --image test3.png --ask "请你判断一下图片中是否有人摔倒？"
```

### 3、model地址

```raw
/// qwen3-vl-8B模型
/home/firefly/Documents/Qwen3-VL-8B-Instruct

/// qwen3-vl-2B模型
/userdata/Qwen3-VL-2B-Instruct
```

### 4、yml地址

```raw
/home/firefly/Documents/Dial_llama/topology_qwen3vl.yml
```

# 三、客户端使用NPU执行指令：

### 1、qwen3-vl-2B 单张图片(服务器端执行)

```raw
./target/release/dial-cli \
  --mode master \
  --api 0.0.0.0:8082 \
  --vision-rknn /home/firefly/Documents/Dial_llama/vision_448.rknn \
  --vision-fixed-side 448
```

### 2、qwen3-vl-8B 单张图片(服务器端执行)

```raw
./target/release/dial-cli \
  --mode master \
  --api 0.0.0.0:8082 \
  --vision-rknn /home/firefly/Documents/Dial_llama/qwen3_vl_8b_vision_448_deepstack_fp16.rknn \
  --vision-fixed-side 448

./target/release/dial-cli \
  --mode master \
  --api 0.0.0.0:8082 \
  --model /home/firefly/Documents/Qwen3-VL-8B-Instruct \
  --text-decode-mode cpu-gpu \
  --vision-rknn /home/firefly/Documents/Dial_llama/qwen3_vl_8b_vision_448_deepstack_fp16.rknn \
  --vision-fixed-side 448
```

### 3、qwen3-vl-8B 视频流(客户端执行)

```raw
  python3 /home/firefly/Documents/Dial_llama/tool/stream_video_client.py \
  --source "rtsp://172.16.30.113:8554/live/a" \
  --interval-sec 20 \
  --spm-cli /home/firefly/Documents/Dial_llama/target/release/dial-cli \
  --api-client http://127.0.0.1:8082 \
  --keep-frames \
  --prompt "请回答图片中是否有人摔倒"
```

# 四、查看板子资源状态常用命令

## 1、RK3588的NPU状态

```raw
watch -n 1 cat /sys/kernel/debug/rknpu/load
```

## 2、查看CPU和MEM状态

```raw
htop
```

## 3、查看ORIN的CPU、MEM和GPU状态

```raw
jtop
```

# 五、三个RK3588运行，两台worker一台mode

### 1、服务器端

```raw
./target/release/dial-cli  --mode worker  --name worker1  --address 0.0.0.0:10128
./target/release/dial-cli  --mode worker  --name worker2  --address 0.0.0.0:10128
```

### 2、客户端

```raw
./target/release/dial-cli \
  --mode master \
  --api 0.0.0.0:8082 \
  --vision-rknn /home/firefly/Documents/Dial_llama/qwen3_vl_8b_vision_448_deepstack_fp16.rknn \
  --vision-fixed-side 448
```

### 3、各个性能参数含义

| 参数             | 含义                                                                            |
| ---------------- | ------------------------------------------------------------------------------- |
| ttft_s           | 首 token 延迟，从请求开始到第一个非空输出。包含图片编码、prefill、首 token 解码 |
| total_s          | 本次请求总耗时                                                                  |
| tps              | 平均生成速率。	generated_tokens / total_s                                       |
| decode_tps       | 解码阶段速率（去掉首 token 阶段）                                               |
| dist_overhead_s  | 分布式“非计算”开销（网络+序列化+协议往返等）                                  |
| remote_compute_s | Worker 输入张量恢复、层执行含同步/转换、输出张量转换的累计时间，不是纯 GPU 时间 |
| remote_requests  | 本次请求中发往远端 worker 的请求次数（收到 Tensor 响应就 +1）                   |

### 4、使用Linux TC流量控制常用命令

#### 1、安装tc

```
sudo apt update && sudo apt install iproute2 -y
```

#### 2、查看tc

```
# 查看 tc 版本（确认工具存在）
tc -V
# 查看系统网络接口（找到你要配置的网卡名，比如 eth0、ens33、wlan0 等）
ip addr
```

#### 3、查看当前网络带宽

```
ethtool eth1
```

### 5、把conda环境打包并在另一个设备上使用

#### 1、解压

```raw
mkdir -p ~/envs/rknn9
tar -xzf rknn9.tar.gz -C ~/envs/rknn9
```

#### 2、激活

```raw
conda env create -f environment.yml
```

#### 3、第一次激活后，立刻修复路径

```raw
conda-unpack
```

#### 4、再检查

```raw
python --version
which python
```

#### 5、开机自启为conda环境

```raw
source /home/firefly/envs/rknn9/bin/activate
source ~/.bashrc
```

### 6.使用不同网段固定IP

### Qwen3.8-27B：Thor + Orin 使用 GGML 量化层后端

新增独立后端 `--inference-backend qwen38-ggml`，保留 DIAL 分层通信。
两端分别编译适配库，使用同一份从原始 BF16 转出的 Q4_K_M GGUF。
原来的 native/NVFP4 启动方式仍可用，不要把旧脚本的 `--qwen38-quant-linear cuda`
混到新后端命令中。

完整步骤见 [docs/qwen38_ggml.md](docs/qwen38_ggml.md)：先构建，再转换一次模型，
复制相同 GGUF 到另一端，随后运行 `bash run_27b_ggml.sh worker/master/client`。
脚本已保存各设备的路径默认值，不再需要每次 `export`。
Worker GGUF 使用你确认的 `/media/nvidia/Elements/Qwen3.8-27B-Q4_K_M.gguf`；
Master 路径沿用此前部署路径。路径变化时修改 `run_27b_ggml.sh` 顶部的
`case "$mode"`，或临时通过环境变量覆盖。不新增配置文件。

#### 当前部署 IP 与启动命令

Orin Worker：`192.168.2.88:10128`；Thor Master API：`192.168.2.101:8082`。
两端的 `topology_qwen38.yml` 必须一致，其中 Worker 地址为：

```yaml
worker0:
  host: "192.168.2.88:10128"
  description: "Qwen3.8-27B native DIAL shard on the second NVIDIA node"
  layers:
    - "model.language_model.layers.0-15"
```

IP 修改只涉及脚本和拓扑，不需要重新编译 DIAL 或 GGML。
停止原进程，先在 Orin 上启动 Worker：

```bash
cd /home/nvidia/Dial_llama
bash run_27b_ggml.sh worker
```

看到监听 `192.168.2.88:10128` 后，在 Thor 上启动 Master：

```bash
cd /home/nvidia/Documents/Dial_llama
bash run_27b_ggml.sh master
```

在 Thor 另开一个终端测试：

```bash
cd /home/nvidia/Documents/Dial_llama
bash run_27b_ggml.sh client
```

仅这次启动脚本默认值更新不需要重新编译。注意旧终端中已有的 `HF_MODEL_DIR`、
`GGUF_MODEL`、`DIAL_WORKER_BIND` 等环境变量仍优先于脚本默认值；若沿用了旧路径，
在该终端执行一次 `unset HF_MODEL_DIR GGUF_MODEL DIAL_WORKER_BIND DIAL_API_BIND`，
或者使用一个未设置这些变量的新终端，再启动。

### GGML 速度优化更新（ABI 2）

连续本地层改成一个 GGML 图执行，减少逐层 Candle/GGML 转换、D2D 拷贝和同步。
构建脚本显式开启 CUDA Graph（只单独引入 ggml/ 时上游默认 OFF），短上下文的
完整注意力按 256-position bucket 扩展，不再始终扫描 4096-position KV。

先在旧 Master/Worker 终端按 Ctrl+C 停止服务，同步更新源码到两端后，都要重编
适配库和程序；模型不用重新下载。

```bash
# Orin 192.168.2.88
cd /home/nvidia/Dial_llama
bash tools/build_qwen38_ggml.sh /media/nvidia/Elements/llama.cpp-b10837 87
cargo build --release --features cuda

# Thor 192.168.2.101（在 Thor 自己的终端执行）
cd /home/nvidia/Documents/Dial_llama
bash tools/build_qwen38_ggml.sh /home/nvidia/Documents/llama.cpp-b10837 110
cargo build --release --features cuda
```

重启后确认 `fused_shards=true cuda_graphs_available=true`。先按原 16/48 拓扑对照；再在两端统一设置
`DIAL_TOPOLOGY="$PWD/topology_qwen38_orin4.yml"`，重新启动 Worker/Master，
测试 Orin 4 层、Thor 60 层。另提供 `topology_qwen38_orin8.yml` 和
Thor 单机空拓扑 `topology_qwen38_thor.yml`。默认拓扑未被覆盖，分层仍可配置。
详细测速和诊断说明见 [docs/qwen38_ggml.md](docs/qwen38_ggml.md#41-速度回归逐层开销和-orin-分层比例分别测试)。

从其他设备测试时，用 `DIAL_API_URL=http://192.168.2.101:8082`。
`0.0.0.0` 只表示监听本机所有 IPv4 网卡，不是客户端连接目标。
DIAL Worker 没有认证，仅在可信内网开放 `10128`，不要暴露到公网。

### 首 token 延迟修复：只更新 Master Rust（沿用 ABI 2）

Master 新请求现在复用本地 GGML 状态、prefill/decode 图和工作区，不再每次销毁。
缓存内容由首轮 position=0 重置；新连接仍独立。API 监听前完成短 prefill、decode
和输出头预热，不输出预热文本、不推进采样 RNG，不污染用户性能指标。
本次不改 GGUF、拓扑或 `.so` ABI；已有 ABI 2 时只需将更新的
`dial-core/src/models/llama3/cache.rs`、`dial-core/src/models/qwen3_8/model.rs`
同步到 Master，停止旧 Master 后执行：

```bash
cd /home/nvidia/Documents/Dial_llama
cargo build --release --features cuda
bash run_27b_ggml.sh master
```

Worker 保持运行。等待 `Qwen3.8 GGML warmup complete` 和 API 监听，再另开终端：

```bash
cd /home/nvidia/Documents/Dial_llama
bash run_27b_ggml.sh client
```

启动预热会多花一段时间；不同 prompt 长度仍可能重建 prefill 图。
若单步依旧慢于 1 秒，Master 自动打印 `[ggml latency]` 各阶段耗时。
这不是修改 token 计数或隐藏用户请求内的耗时，不能提前保证板端达到某个速度。
