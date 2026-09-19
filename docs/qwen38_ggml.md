# Thor + Orin：GGUF/GGML 本地层执行，DIAL 分层部署

新增 `--inference-backend qwen38-ggml`。不是 `llama-server` API 代理，也不是
GGML RPC：拓扑、Master/Worker、跨节点激活张量、分词、采样和 HTTP API 都在 DIAL。
连续本地 Transformer 层使用 llama.cpp **b10837** 的一个 GGML 图执行，权重保持 GGUF
量化存储，矩阵乘法、SSM 卷积、Gated DeltaNet、FlashAttention 均调用上游算子。
原来的 `native`、`qwen38-native` 和 `qwen38-rpc` 没有被替换。

## 当前边界

- 首版仅 Qwen3.8/Qwen3.5 **dense 文本模型、batch=1**，不支持 MoE/图片/视频/MTP。
  转换脚本显式传 `--no-mtp`，只导出 HF 配置中的主干层；不要混用带额外 MTP 层的 GGUF。
- 两端用同一份、由 b10837 转换的 GGUF。建议从原始 BF16 checkpoint 生成 Q4_K_M；
  不自动把现有 NVFP4 checkpoint 二次量化成 Q4。不是两端使用不同量化。
- 默认继续使用 `topology_qwen38.yml`：Orin Worker 0–15，Thor Master 16–63；
  **分层不是写死在执行器里**，改拓扑即可。先不要同时更换拓扑和量化方式。
- Worker 只读入分配层的权重，不加载 embedding/lm_head。Master 只加载本地层和
  embedding/output head。不需要在 GGUF 模式 mmap 原 safetensors；`--model` 仍是
  **与 GGUF 相匹配的 HF config/tokenizer 目录**，不是 GGUF 文件。
- 每条 DIAL 连接有独立的 GGML KV/卷积/递归状态；position=0 重置新请求。
  超过 `--kv-cache-max-len` 明确报错，不实现滑动窗口/缓存分页。
- Master 的连续请求保留本地状态存储、prefill/decode 图和工作区，不再每次销毁重建；
  请求首轮 position=0 清零参与层的数据，新连接仍分配独立状态。
- Master 在 API 开始监听前执行两次短 prefill 和三次单 token decode，预热实际
  embedding、本地/远程 shard、输出头及 host 读取路径。预热不输出回答、不采样，
  不改变用户的 RNG 或性能指标。启动会多等一段时间，但不再把这些首次执行开销
  留给首个 API 请求；不同 prompt 长度仍可能触发有界的 prefill 图重建。
- GPU 交换边界是同步 D2D 拷贝和必要的 dtype cast。连续层内部激活留在 GGML/F32，
  不再逐层返回 Candle/F16。没有每个 Linear 的 CPU 权重反量化/host 激活来回拷贝；
  跨框架仍不是零拷贝。构建脚本显式开启 `GGML_CUDA_GRAPHS=ON`，兼容且稳定的
  decode 图由上游后端自动捕获/复用，不承诺每个算子或每次请求都能捕获。
- 完整注意力的 KV 存储仍按最大上下文分配，但计算范围按已用位置向上取整到 256。
  例如短请求只读 256 个位置，不再始终按 4096 建图。token 数或 KV bucket 变化时才重建图，
  保留已有 KV；同一个 shard 的完整注意力层共享一次位置/掩码上传。
- 本地 CPU 数值测试不能代替 Thor/Orin CUDA 正确性和速度测试。首次运行请先测短问答。
  跨节点串行 decode/网络开销仍存在，不能保证比 Thor 单机更快。

## 1. 两端分别编译适配库和 DIAL

把更新后的 DIAL 源码同步到两端。使用各设备已有的 `llama.cpp-b10837` **源码目录**，
不是其 `build/bin`。适配库在本机编译：Orin SM87，Thor SM110。
更新前先在旧 Master/Worker 终端按 Ctrl+C 停止服务，避免覆盖正在映射的适配库。
Thor 的 CUDA toolkit 必须支持 SM110；不要把 Thor 编出来的 `.so` 直接复制到 Orin。
脚本默认 `-j2`，首次 CUDA 编译仍需要时间。
本次优化使用适配库 **ABI 2**：必须同步更新并在两端重编 `.so` 和 Rust 程序，
不能只更新 `dial-cli`。开启 CUDA Graph 会使 CUDA 源文件重新编译，不需要重新下载模型。

Thor Master (192.168.2.101)，在 `/home/nvidia/Documents/Dial_llama`：

```bash
cd /home/nvidia/Documents/Dial_llama
bash tools/build_qwen38_ggml.sh /home/nvidia/Documents/llama.cpp-b10837 110
cargo build --release --features cuda
```

Orin Worker (192.168.2.88)，在 `/home/nvidia/Dial_llama`：

```bash
cd /home/nvidia/Dial_llama
bash tools/build_qwen38_ggml.sh /media/nvidia/Elements/llama.cpp-b10837 87
cargo build --release --features cuda
```

这些 llama.cpp 路径来自之前部署记录；如果源码在别处，用设备上的实际路径替换。
构建脚本检查上游头文件、Qwen 图和转换器的版本指纹，拒绝混用其它版本。
若 nvcc 不在 PATH，先把**本机** CUDA bin 目录加入 PATH。不要在 Orin 使用 Thor 的 CUDA 路径。

## 2. 转换一次 GGUF，再复制给另一个设备

原始 BF16 模型目前在 Orin 外接盘 `/media/nvidia/Elements/Qwen3.8-27B`。
转换/量化只需一次。下面就在有原始模型和足够磁盘空间的 Orin 做；也可把原始
模型复制到 Thor 后在那里转换。不要往 97% 满的根分区输出。

```bash
cd /home/nvidia/Dial_llama
python3 -m venv /media/nvidia/Elements/gguf-convert-venv
/media/nvidia/Elements/gguf-convert-venv/bin/python -m pip install \
  -r /media/nvidia/Elements/llama.cpp-b10837/requirements.txt
cmake --build /media/nvidia/Elements/llama.cpp-b10837/build --target llama-quantize -j2
DIAL_CONVERT_PYTHON=/media/nvidia/Elements/gguf-convert-venv/bin/python \
  bash tools/convert_qwen38_gguf.sh \
  /media/nvidia/Elements/llama.cpp-b10837 \
  /media/nvidia/Elements/Qwen3.8-27B \
  /media/nvidia/Elements/Qwen3.8-27B-GGUF
```

会保留 BF16 GGUF 和 Q4_K_M GGUF，不覆盖已有输出、不删除原始模型。
转换需要额外保存约 52GB BF16 GGUF 和十几 GB 的量化输出；精确大小以文件为准。
若上游转换失败，先看转换器报错，不要把错误的文件拿去部署。

把 `Qwen3.8-27B-Q4_K_M.gguf` 复制到 Thor，例如目标
`/home/nvidia/models/Qwen3.8-27B-GGUF/`（先确认这里有足够空间）。
两端执行 `sha256sum /实际路径/Qwen3.8-27B-Q4_K_M.gguf`，确认相同。
HF 配置、tokenizer 和 generation_config 必须来自同一模型；可只复制这些小文件，
GGML 模式无需复制 BF16 safetensors 到 Thor。

## 3. 启动 Worker，再启动 Master

先确认两端的 `topology_qwen38.yml` 内容一致，worker0 指向 Orin **实际 IP**。
当前 Worker 是 `192.168.2.88:10128`，Master 是 `192.168.2.101:8082`。
如果地址改变了，同步修改两端拓扑和 `DIAL_WORKER_BIND`。
监听地址和连接地址不同：Master 的 API 可以监听 `0.0.0.0:8082`，
远程客户端连接 `192.168.2.101:8082`，不能连接 `0.0.0.0`。
DIAL Worker 无认证，只绑定可信内网，不暴露到公网。

Orin Worker (192.168.2.88)：

启动脚本已有本机默认路径，`HF_MODEL_DIR` 为原 HF 模型目录，Worker 的
`GGUF_MODEL` 为你确认的 `/media/nvidia/Elements/Qwen3.8-27B-Q4_K_M.gguf`。
只需运行下列命令，不用每次 export；如果沿用第 2 节的 GGUF 子目录，修改脚本
顶部的 Worker 默认路径，或临时用 `GGUF_MODEL=/实际路径/...gguf bash ... worker`
覆盖。文件存在性检查仍保留，不会自动猜测或搜索模型。

```bash
cd /home/nvidia/Dial_llama
bash run_27b_ggml.sh worker
```

看到 `GGML worker: assigned_layers=16` 和 `listening ...10128` 后保留该终端。

Thor Master (192.168.2.101)，脚本里的 GGUF 默认路径是
`/home/nvidia/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf`，请先确认文件存在。
现有 NVFP4 模型目录这里只提供匹配的配置/分词器，**不会加载 NVFP4 权重**：

```bash
cd /home/nvidia/Documents/Dial_llama
bash run_27b_ggml.sh master
```

启动日志应包含 `Qwen3.8 GGML ready`、`upstream GGML ... resident_weights`、
`starting api ...8082`。若出现“decode NVFP4/FP8 weights to F16”则不是新 GGML 路径。
脚本会拒绝不识别新参数的旧 `dial-cli`，并检查所需路径；不会替你猜模型位置。
只调整默认值时，更新 `run_27b_ggml.sh` 即可，不需要重新编译。
现有环境变量优先于默认值；如果旧终端还设置着其它路径，先执行一次
`unset HF_MODEL_DIR GGUF_MODEL DIAL_WORKER_BIND DIAL_API_BIND`，或换一个干净终端。

### 首 token 延迟修复：Master 缓存复用与启动预热

本次 Rust 修复需要更新 Master 的 `dial-core/src/models/llama3/cache.rs` 和
`dial-core/src/models/qwen3_8/model.rs`；回归测试在
`dial-core/src/models/qwen3_8/ggml_tests.rs`。保留 ABI 2，不改适配库、模型和拓扑。
如果两端已经使用上一节的 ABI 2 `.so`，此次只需在 Master 停止旧服务、同步
源码、重编 Rust，再启动；Worker 保持运行，不需要再次编译 `.so` 或下载模型。

```bash
cd /home/nvidia/Documents/Dial_llama
cargo build --release --features cuda
bash run_27b_ggml.sh master
```

等日志出现 `Qwen3.8 GGML warmup complete` 和 API 监听后，再启动客户端。
预热期间可能显示 `[ggml latency]`，这是测到的首次执行耗时，不是额外一轮
用户回答。慢于 1 秒的 GGML forward 会自动记录 embedding、blocks、head 耗时，
慢本地 shard 另记录层范围；正常快路径不会逐 token 打印这些日志。
诊断时可用 `DIAL_GGML_WARMUP=0` 禁用启动预热，但正式运行不要设置该变量。
默认预热短算术问题，也可用 `DIAL_GGML_WARMUP_PROMPT` 指定常用短 prompt。
预热没有消除真实 Transformer 算量或 Thor/Orin 的串行执行，不承诺具体 TTFT/tps；
CPU 缓存生命周期/数值测试及 CUDA 编译检查不能替代板端测速。

## 4. 客户端测试与对照

在 Thor 另开一个终端：

```bash
cd /home/nvidia/Documents/Dial_llama
bash run_27b_ggml.sh client
DIAL_ASK='用一句话介绍你自己。' bash run_27b_ggml.sh client
```

远程客户端设置 `DIAL_API_URL=http://192.168.2.101:8082`。
先验证答案 `2` 且没有 `<|im_end|>` 泄漏，再测试较长回答和连续两个请求。
把客户端性能指标、Worker 出错/耗时日志一起保存。

比较 native/ggml 时固定相同 prompt、thinking=false、temperature=0、上下文和拓扑，
分别看 TTFT 与输出 token/s。GGUF 量化有精度损失，不能只看“一加一”判断整体精度。
再用空拓扑跑 Thor 单机对照，区别网络分层代价和本地算子速度；不承诺异构两机
比 Thor 单机更快。解码图在同一 active-KV bucket 内复用，4K 以外的上下文另测。

### 4.1 速度回归：逐层开销和 Orin 分层比例分别测试

当前优化默认开启，CUDA 启动日志应包含 `fused_shards=true`、
`cuda_graphs_available=true`、`active_kv_bucket=256`。如果 CUDA Graph 不可用，
检查构建缓存的 `GGML_CUDA_GRAPHS:BOOL=ON`，并确认没有设置
`GGML_CUDA_DISABLE_GRAPHS`（上游连 `GGML_CUDA_DISABLE_GRAPHS=0` 也会禁用）。
原来的 16/48 分层不变时，Transformer 部分每个 token 由 64 次跨框架调用减少为
2 次整段调用，embedding/head 仍独立执行；这不是把整个生成过程并行化。

先保持原拓扑，对照修改前后的相同 prompt、Q4_K_M 文件、上下文和输出限制。
连续测试三次，记录后两次；同时检查回答正确性和实际输出 token 数，不只看总耗时。
`DIAL_SAMPLE_LEN=128` 可控制 Master 的输出上限，但模型仍可提前遇到 EOS。
设置 `DIAL_GGML_FUSED=0` 并重启两端可隔离比较逐层与整段路径；此开关不会关闭
新的 active-KV 优化。若需隔离 CUDA Graph，在本机用
`DIAL_GGML_CUDA_GRAPHS=OFF`（或直接设置上游的 `GGML_CUDA_DISABLE_GRAPHS=1`
后重启进程）关闭它。前者需要用本机源码和 87/110 架构重编适配库；后者不用重编。
正式测试前恢复 ON，或 `unset GGML_CUDA_DISABLE_GRAPHS` 后重启。

Orin 的权重计算和统一内存访问仍可能成为瓶颈。一个 token 必须依次经过 Orin
远端层、Thor 本地层和输出头，两板不会自动让单请求 decode 的速度相加。
不要为了“用满两个板”强制平均分层。新增以下对照拓扑，不覆盖原文件：

单请求、batch=1 追求最低延迟时，直接在 Thor 上运行全部解码层：

```bash
# Thor：启动 Master，再另开终端运行 Client；Orin 不启动 Worker
bash run_27b_low_latency.sh master
bash run_27b_low_latency.sh client
```

`run_27b_low_latency.sh` 固定选择 Thor-only 拓扑。原来的
`run_27b_ggml.sh worker/master/client` 仍用于双板实验或多请求吞吐测试。

| 拓扑文件 | Orin 层 | Thor 层 |
|---|---|---|
| `topology_qwen38.yml` | 0–15 | 16–63 |
| `topology_qwen38_orin8.yml` | 0–7 | 8–63 |
| `topology_qwen38_orin4.yml` | 0–3 | 4–63 |
| `topology_qwen38_thor.yml` | 不执行 | 0–63 |

例如测试 4/60 分层，**两端**先设置下列环境变量，再按第 3 节启动，先 Worker 后 Master：

```bash
export DIAL_TOPOLOGY="$PWD/topology_qwen38_orin4.yml"
```

测试 Thor 单机时，只在 Thor 设置
`DIAL_TOPOLOGY="$PWD/topology_qwen38_thor.yml"` 后启动 Master，不启动 Worker。
这样单机和双机仍使用同一 DIAL/GGML 执行后端，避免把不同软件栈的速度混在一起。
更换拓扑必须先停旧进程，两端同步选择；恢复默认用 `unset DIAL_TOPOLOGY`。

性能指标中的 `remote_compute_s` 是 Worker 解码输入、执行层、转换输出的累计时间，
**不是纯 GPU kernel 时间**；`dist_overhead_s` 是统计到的请求往返时间减 Worker
耗时，不能覆盖 Master 所有 tensor staging/转换开销。可临时在 Worker 启用：

```bash
SPM_TRACE_TRANSFER=1 bash run_27b_ggml.sh worker
```

日志新增 `fused=true`、`execute=...ms`、`encode=...ms`，分别帮助确认整段路径、
层执行含同步/转换的耗时、结果 GPU→CPU 转换耗时。诊断日志会增加开销，正式测速
关闭它。没有 Thor/Orin 实测前，不承诺两板超过 Thor 单机的 token/s。

本次性能更新需同步的文件（不要用工作站编出的 `.so` 替换板端 `.so`）：

```text
backends/qwen38_ggml/adapter.h
backends/qwen38_ggml/adapter.cpp
backends/qwen38_ggml/CMakeLists.txt
dial-core/src/models/qwen3_8/ggml.rs
dial-core/src/models/qwen3_8/ggml_tests.rs
dial-core/src/models/qwen3_8/model.rs
dial-core/src/models/qwen3_8/text.rs
dial-core/src/models/llama3/cache.rs
dial-core/src/spm/mod.rs
dial-core/src/spm/worker.rs
tools/build_qwen38_ggml.sh
run_27b_ggml.sh
topology_qwen38_orin4.yml
topology_qwen38_orin8.yml
topology_qwen38_thor.yml
docs/qwen38_ggml.md
run2.md
```

## 开发验证

```bash
bash tools/build_qwen38_ggml.sh /实际路径/llama.cpp-b10837 cpu
DIAL_GGML_TEST_LIB="$PWD/build/qwen38-ggml/lib/libdial_qwen38_ggml.so" \
  cargo test -p dial-core ggml_ -- --ignored --nocapture
cargo test --workspace
CUDARC_CUDA_VERSION=12060 cargo check --workspace --features cuda
```

每台设备编译 CUDA 适配库后，先用不依赖大模型的 CUDA 互操作测试验证其本地 GPU：

```bash
DIAL_GGML_TEST_LIB="$PWD/build/qwen38-ggml/lib/libdial_qwen38_ggml.so" \
  cargo test -p dial-core --features cuda ggml_cuda_device_interop_smoke -- --ignored --nocapture
```

这项测试明确要求可见 GPU，不会偷偷改成 CPU；验证 F16 设备边界、同步、
SSM/DeltaNet、FlashAttention、embedding 和输出头，并与 CPU GGML 数值对照。

显式数值测试构造非零 Q4_K/Q8 GGUF，用 HF/Candle dense 独立路径对照线性注意力与
完整注意力，检查 prefill/单 token decode、分块等价、position=0 重置、连接缓存隔离、
embedding/lm_head、上下文越界和配置不匹配。新增多层 fused/per-layer 等价、
256→512 KV bucket 增长、64 层整图容量和实际 TCP Worker 整段执行计数验证。
正常测试不会自动下载/编译上游库。
