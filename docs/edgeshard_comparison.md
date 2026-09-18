# EdgeShard 对比实验

## 代码性质

本仓库实现的是可在 DIAL 现有 RK3588/Jetson Orin 运行时中执行的 EdgeShard
策略复现，不是 EdgeShard 作者发布的官方代码。实现依据
[EdgeShard 论文](https://arxiv.org/html/2405.14371)中的两个目标：

- `edgeshard-latency`：对应论文算法 1 的总延迟最小化目标；
- `edgeshard-throughput`：对应论文算法 2 的最慢流水段最小化目标；
- `dial`：本系统的 TTFT、TPOT、能耗和远程设备代价联合目标。

三种策略使用相同的设备 Profile、内存约束和连续层段搜索空间。每台设备最多
承载一个连续层段，设备可以不被选择。状态为：

```text
DP[end_layer][selected_device_mask][last_device]
```

对一段层 `[i, m]`，设完整请求的计算代价为：

```text
C(i,m,j) = prefill(i,m,j) + output_tokens * decode(i,m,j)
```

DIAL 当前协议让远程层段按 `Master -> Worker -> Master` 往返，因此远程段的完整
请求通信代价为：

```text
N(j) = prefill_roundtrip(j) + output_tokens * decode_roundtrip(j)
```

三个目标分别为：

```text
EdgeShard-Latency:    min sum(C + N)
EdgeShard-Throughput: min max(C_stage, N_stage)
DIAL:                 min weighted(TTFT, TPOT, energy, worker_count)
```

代码入口位于 `dial-core/src/spm/planner.rs`，运行参数为
`--auto-plan-algorithm`。Master 生成的 JSON 报告会记录算法名称、目标定义、预测
分数、层段和实现边界。

这套方法属于“离线测量 + 启动时规划”：Profile 离线生成，每次进程启动时计算
一次层切分，推理过程中不会迁移权重或在线重规划。

## 与原论文的边界

EdgeShard 原论文假设相邻设备直接传输激活，DIAL 当前使用以 Master 为中心的星型
往返。论文算法 1 可以逐层改变设备，本实现将搜索限制为 DIAL 可稳定部署的连续
层段。论文算法 2 的真实吞吐提升还依赖多请求流水并行和 EdgeShard No-bubbles
调度，当前 DIAL API 使用独占模型状态，尚未实现该调度。

因此：

- `edgeshard-latency` 可以作为主要的真实系统延迟基线；
- `edgeshard-throughput` 可以比较规划结果和瓶颈预测，但当前实测只能代表该布局
  在顺序推理下的性能，不能声称复现了 EdgeShard 的流水吞吐量；
- 论文中应称为 “EdgeShard objective adapted to the DIAL runtime”，并明确上述差异。

原论文的隐私约束要求输入所在源节点执行第一层。DIAL 的输入、tokenization 和
embedding 固定在 Master，只向 Worker 发送 hidden activation，因此本实现按“不发送
原始输入”的等价语义处理，不强制第 0 个 Transformer block 位于 Master。

## 实验前准备

复制并标定 Profile：

```bash
cp docs/auto_plan_4devices.example.yml /path/auto_plan_4devices.measured.yml
```

示例数字仅说明格式。必须使用实际实验后端逐层测量
`prefill_ms`、`decode_ms`、`layer_memory_mb`，并实测 Master 到各 Worker 的
`rtt_ms`、`bandwidth_mbps` 和 `protocol_ms`。Profile 中：

公平比较时增加一组`DIAL-Mean`消融（`risk_weight: 0`），再用相同Profile中的
P95/P05字段运行`DIAL-RP`。`edgeshard-latency`和`edgeshard-throughput`只使用均值
目标；这样可以分别回答“DIAL的相位/拓扑建模是否有效”和“尾部风险项是否有效”，
不能把两项改动混成一次对比。

```yaml
objective:
  prompt_tokens: 32
  output_tokens: 96
  max_devices: 4
```

这与 EdgeShard 论文的主要输入/输出长度一致。研究本系统时还应增加 128、512、
1024 输入 token 等工作负载。Profile 的 `output_tokens` 必须与 Master 的
`--sample-len` 一致。

在 Orin 上按实际 CUDA/GGUF 后端构建，在 RK3588 上使用同一提交对应的 CPU/RKNN
构建。所有实验组必须固定：

- 模型和权重精度；
- CPU/GPU/NPU 频率及功耗模式；
- `--sample-len`、seed、temperature、top-p 和 top-k；
- 视觉输入尺寸和图片；
- GGUF、W8A16、RKNN、lm_head 和远程采样开关；
- 网络连接和 `tc` 参数。

只比较规划器时，建议把 `--worker-gguf-output-head` 和
`--worker-gguf-sample-token` 都设为 `false`，避免某个算法因最后一层恰好在 Orin
而额外改变输出头后端。论文的系统最佳结果可以另设一组，统一开启完整优化。

## 启动一组实验

以下变量在四台设备上取同一个值，每次只运行其中一个：

```bash
ALGORITHM=edgeshard-latency
PROFILE=/path/auto_plan_4devices.measured.yml
```

也需要依次测试：

```text
dial
edgeshard-latency
edgeshard-throughput
```

两台 RK3588 Worker 分别启动：

```bash
RUST_LOG=info ./target/release/dial-cli \
  --mode worker \
  --name rk-worker1 \
  --address 0.0.0.0:10128 \
  --model /path/Qwen3-VL-8B-Instruct \
  --cpu \
  --auto-plan-profile "$PROFILE" \
  --auto-plan-algorithm "$ALGORITHM"
```

第二台把名称改为 `rk-worker2`。Orin Worker 使用与 Profile 标定一致的后端，例如：

```bash
RUST_LOG=info ./target/release/dial-cli \
  --mode worker \
  --name orin-worker \
  --address 0.0.0.0:10128 \
  --model /path/Qwen3-VL-8B-Instruct \
  --worker-quantized-gguf /path/qwen3-vl-8b-instruct-q4_K_M.gguf \
  --worker-gguf-fp16-prefill true \
  --worker-gguf-output-head false \
  --worker-gguf-sample-token false \
  --dtype f16 \
  --auto-plan-profile "$PROFILE" \
  --auto-plan-algorithm "$ALGORITHM"
```

最后在 RK3588 Master 启动 API：

```bash
RUST_LOG=info SPM_COMPACT_BATCH=1 ./target/release/dial-cli \
  --mode master \
  --api 0.0.0.0:8082 \
  --model /path/Qwen3-VL-8B-Instruct \
  --cpu \
  --sample-len 96 \
  --seed 299792458 \
  --temperature 0 \
  --auto-plan-profile "$PROFILE" \
  --auto-plan-algorithm "$ALGORITHM" \
  --auto-plan-output "result/${ALGORITHM}.plan.json"
```

若当前采样器不接受 `temperature=0`，三组都使用同一个正数并固定 seed。四个进程
必须读取内容相同的 Profile，并传入相同算法，否则各节点会得到不同层归属。

## 采集原始结果

Master 启动完成后，在 Master 上执行；也可以在其他能访问 API 的机器执行，但需要
先把对应的 plan JSON 放到该机器的同一路径：

```bash
python3 tools/run_planner_benchmark.py \
  --api http://MASTER_IP:8082 \
  --algorithm "$ALGORITHM" \
  --prompts-jsonl docs/edgeshard_prompts.example.jsonl \
  --warmup 5 \
  --runs 30 \
  --sample-len 96 \
  --plan-report "result/${ALGORITHM}.plan.json" \
  --tag model=qwen3-vl-8b \
  --tag network=1gbe \
  --output "result/${ALGORITHM}.csv"
```

`--algorithm` 只是结果标签，不会远程切换服务端。每换一种算法，都要停止四个旧
进程、用新算法重新启动并确认 Master 日志中的层段，然后再运行脚本。脚本默认拒绝
覆盖已有 CSV；需要继续写入时显式传 `--append`。

汇总三组结果：

```bash
python3 tools/summarize_planner_benchmarks.py \
  result/dial.csv \
  result/edgeshard-latency.csv \
  result/edgeshard-throughput.csv \
  --output result/planner_comparison.md
```

原始 CSV 保留每次请求的 TTFT、总时间、decode TPS、实际生成 token 数、分布式
开销、远端计算时间、规划层段和规划报告 SHA-256。论文表格至少报告 median、P95
和成功样本数，不应只报告一次运行。`--sample-len` 是上限；若某些请求提前 EOS，
应报告实际 token 数，并使用 TTFT/decode TPS 或等长样本进行公平比较。

## Solo 和 Static-Even

为了对应 EdgeShard 论文中的 Edge-Solo 和 Even 基线，仓库提供：

```text
docs/topology_solo.example.yml
docs/topology_even_4devices.example.yml
```

这两组不传 `--auto-plan-profile` 和 `--auto-plan-algorithm`，改用
`--topology`。Static-Even 中 Master 默认执行 0-8 层，三个 Worker 各执行 9 层；
运行前修改示例 IP。Solo 只需启动 Master。采集时分别使用：

```bash
--algorithm solo
--algorithm static-even
```

这两个值同样只是 CSV 标签。Static-Even 的四个进程必须读取同一静态拓扑。

为降低温度和运行顺序偏差，建议三种自动规划策略各做至少 3 次独立冷启动，按拉丁方
或随机顺序轮换策略，而不是始终按同一顺序测试。

## 建议的论文表格

主要表格使用真实板卡结果：

| 方法 | 使用设备 | 层切分 | TTFT median/P95 | TPOT 或 decode TPS | 总延迟 | 分布式开销 | 峰值内存 | 能耗 |
|---|---|---|---|---|---|---|---|---|
| Static-Even | | | | | | | | |
| EdgeShard-Latency adapted | | | | | | | | |
| DIAL | | | | | | | | |

补充实验报告 `edgeshard-throughput` 的预测瓶颈和顺序执行结果，并明确没有实现
No-bubbles。网络敏感性实验可以用 `tc` 固定 10、50、100、500、1000 Mbps；每个
带宽点重新测量网络 Profile、重新规划并运行，不能只修改 Profile 中的数字而不改变
真实网络。

不同硬件、模型和精度下的绝对数值不能与原论文表格直接比较；有效结论来自同一套
RK3588/Orin 环境中的方法间相对差异。
