# Qwen3.8-27B 双 NVIDIA 节点部署

本部署模式通过新增的 `qwen38-rpc` 后端运行 Qwen3.8-27B：DIAL 负责兼容 API、网页和客户端入口，llama.cpp 负责 Qwen3.8/Qwen3.5 混合线性注意力、CUDA kernel、量化和跨节点 GGML RPC。原有 `native`（Llama3/Qwen3-VL）路径不受影响。

Qwen3.8 的原始 Transformers 权重约 55.6 GB，不能直接放进两块 16 GB 的 Jetson NX；应使用 Q4_K_M（约 19 GB）或 Q4_K_S（约 16.7 GB）GGUF，并让 llama.cpp 在本机 GPU 和远端 RPC GPU 之间切分。官方 GGUF 仓库同时提供 Q4_K_M、Q8_0、BF16 和视觉 `mmproj` 文件，且架构标记为 `qwen35`，这是 Qwen3.8 的正常兼容标识。

## 1. 两台设备准备 llama.cpp

在两台 NVIDIA 设备上执行（CUDA toolkit、驱动和编译器需已经安装）：

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
cmake -S . -B build -DGGML_CUDA=ON -DGGML_RPC=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
```

确认生成了以下程序：

```bash
test -x build/bin/llama-server
test -x build/bin/ggml-rpc-server
```

GGML RPC 是实验性协议，只应绑定在可信的专用网或 VPN 上，不能直接暴露到公网。

## 2. 准备模型

在 DIAL Master 所在设备准备量化模型和视觉投影器。以 Hugging Face 官方 GGUF 仓库为例：

```bash
mkdir -p /data/models/Qwen3.8-27B-GGUF
hf download ggml-org/Qwen3.8-27B-GGUF \
  Qwen3.8-27B-Q4_K_M.gguf \
  mmproj-Qwen3.8-27B-BF16.gguf \
  --local-dir /data/models/Qwen3.8-27B-GGUF
```

如果 NX 内存余量很小，可改用社区仓库 `bartowski/Qwen3.8-27B-GGUF` 中约 16.7 GB 的 `Qwen3.8-27B-Q4_K_S.gguf`。`--qwen38-mmproj` 只在需要图片输入时传入；纯文本可以省略。

两台板上构建 DIAL。这个后端的 CUDA 由 llama.cpp 提供，因此 DIAL 自身无需启用 Candle 的 `cuda` feature：

```bash
cd /path/to/Dial_llama
cargo build --release
```

原有 `native` 模块仍可按原来的方式用 `cargo build --release --features cuda` 构建。

## 3. 在第二台设备启动 CUDA RPC Worker

假设第二台设备 IP 是 `192.168.2.21`：

```bash
cd /path/to/Dial_llama
./target/release/dial-cli \
  --inference-backend qwen38-rpc \
  --mode worker \
  --address 192.168.2.21:50052 \
  --device 0 \
  --qwen38-rpc-server-bin /path/to/llama.cpp/build/bin/ggml-rpc-server
```

该 DIAL Worker 模式会启动等价的 `ggml-rpc-server --device CUDA0`。在 Master 上先检查端口：

```bash
nc -vz 192.168.2.21 50052
```

默认不启用 RPC 磁盘缓存，避免把 NX 系统盘写满。如果 Worker 有额外的高速盘且至少还有约 12 GB，可以这样启用：

```bash
LLAMA_CACHE=/data/llama-cache ./target/release/dial-cli \
  --inference-backend qwen38-rpc \
  --mode worker \
  --address 192.168.2.21:50052 \
  --qwen38-rpc-server-bin /path/to/llama.cpp/build/bin/ggml-rpc-server \
  --qwen38-rpc-cache true
```

## 4. 在第一台设备启动 DIAL Qwen3.8 Master

```bash
cd /home/seaway/sdb/ljl/Dial_llama

cargo build --release

./target/release/dial-cli \
  --inference-backend qwen38-rpc \
  --mode master \
  --api 0.0.0.0:8082 \
  --qwen38-llama-server-bin /path/to/llama.cpp/build/bin/llama-server \
  --qwen38-gguf /data/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf \
  --qwen38-mmproj /data/models/Qwen3.8-27B-GGUF/mmproj-Qwen3.8-27B-BF16.gguf \
  --qwen38-rpc-workers 192.168.2.21:50052 \
  --qwen38-tensor-split 1,1 \
  --kv-cache-max-len 4096 \
  --qwen38-thinking false
```

`--qwen38-thinking false` 会同时把托管的 llama.cpp 服务设为 `--reasoning off`，并给转发请求补充 `chat_template_kwargs.enable_thinking=false`；设为 `true` 则开启思考模式。

如果启动日志提示 llama.cpp 不认识某个参数，可以用重复参数把它传递给下游程序，例如：

```bash
--qwen38-llama-arg=--flash-attn \
--qwen38-llama-arg=on
```

DIAL 等待下游 `/health` 返回成功后才开放 API。网页地址为 `http://MASTER_IP:8082/`，原有 DIAL 客户端仍然使用：

```bash
./target/release/dial-cli \
  --inference-backend qwen38-rpc \
  --api-client http://MASTER_IP:8082 \
  --ask "请简要介绍一下你自己"
```

图片请求：

```bash
./target/release/dial-cli \
  --inference-backend qwen38-rpc \
  --api-client http://MASTER_IP:8082 \
  --image ./test.png \
  --ask "请描述这张图片"
```

## 5. 诊断

```bash
curl http://127.0.0.1:8082/health
curl http://127.0.0.1:8082/v1/models
curl http://127.0.0.1:8082/api/v1/topology
nvidia-smi
```

Jetson 上可另开终端查看内存和 GPU：

```bash
tegrastats
```

如果启动时 OOM，按以下顺序处理：

1. 将 `Q4_K_M` 换成 `Q4_K_S`；
2. 将 `--kv-cache-max-len` 降到 `2048`；
3. 保持 `--qwen38-tensor-split 1,1`，不要把全部权重压到单节点；
4. 关闭图片投影器（不传 `--qwen38-mmproj`）进行纯文本验证；
5. 确认两台机器的 `llama.cpp` commit 和 CUDA 构建选项一致。

## 后端选择

```bash
# 8B：继续使用 DIAL 原生 Qwen3-VL 路径
./target/release/dial-cli --model-size 8b ...

# 27B：Qwen3.8，经 llama.cpp + CUDA RPC
./target/release/dial-cli --model-size 27b ...
```

不传 `--model-size` 时仍默认使用原来的 8B 路径。高级用法可以继续使用
`--inference-backend native` 或 `--inference-backend qwen38-rpc`；模型大小与后端、
专用参数冲突时 CLI 会直接报错，不会静默加载另一个模型。

`qwen38-rpc` 不读取 `config.json`，因此不要求把 55.6 GB 的 Transformers safetensors 目录传给 DIAL；它使用 `--qwen38-gguf` 指定的量化文件。原有 Qwen3-VL 的 `--vision-rknn`、`--text-rknn-dir`、RKNN/RKLLM 转换文件仍只属于 `native` 后端，不能与 Qwen3.8 GGUF 混用。
