# DIAL-RP 四设备风险感知自动层规划器

## 适用拓扑

当前实现面向一个主控和若干 Worker。对于三台 RK3588 和一台 Jetson Orin，建议角色为：

- `rk-master`：RK3588，负责 API、视觉链路、embedding、lm_head，以及规划后留在本地的文本层；
- `rk-worker1`：RK3588 Worker；
- `rk-worker2`：RK3588 Worker；
- `orin-worker`：Jetson Orin Worker。

自动规划不是负载均分。算法允许任意设备不承载 Transformer 层。如果 Orin 能容纳全部层且网络成本足够低，结果可能只让 Orin Worker 承载 Transformer 层；此时 RK3588 Master 仍然负责 API、视觉/输入处理、调度、采样和输出，并没有被关闭，也没有把 Master 角色迁移给 Orin。如果容量、能耗权重或网络条件不同，算法才会选择 RK3588 Master 或其他 RK3588 Worker 承载部分 Transformer 层。

## 核心创新

本文可以把该算法命名为 **DIAL-RP（Risk-aware Partitioning）**。它针对DIAL已有
运行时组合四项设计：

1. **设备子集、顺序和层边界联合搜索**：三台RK3588和一台Orin只是候选集，算法
   可以只选一台、两台或更多设备，不把“设备越多越快”作为前提；
2. **prefill/decode相位感知**：分别优化影响TTFT的prefill和影响交互体验的decode，
   不用一个总延迟掩盖二者差异；
3. **DIAL星型链路原生建模**：每个远程段都按实际的
   `Master -> Worker -> Master`往返计费，不借用Worker直连流水线的理想通信模型；
4. **尾延迟风险感知**：除均值外，使用设备逐层P95、网络RTT/协议P95和带宽P05
   构造保守场景，避免均值最快但热降频或网络抖动后很慢的设备主导计划。

与已有工作的区别必须按适用问题陈述，不能写成所有场景都优于它们：

| 方法 | 主要优化对象 | 与DIAL-RP的关键差异 |
|---|---|---|
| EdgeShard | 边缘设备间DNN切分的平均延迟/流水线瓶颈 | DIAL-RP额外区分LLM的TTFT/TPOT、KV内存、设备可选和尾部风险，并使用DIAL真实星型链路 |
| LinguaLinked | 多设备LLM流水线与负载均衡 | DIAL-RP不要求所有设备形成同一种流水线，允许Orin单独承担全部文本层或跳过慢节点 |
| H-PFC | 异构边缘DNN的预划分和网络变化下调度 | H-PFC的在线重调度更适合无状态DNN；DIAL-RP在启动时生成稳定计划，避免LLM权重和KV cache在线迁移，并显式优化decode |
| HeteroInfer | 单个移动SoC内部GPU/NPU协同 | 它启发了按prefill/decode分别画像；DIAL-RP解决的是多个独立设备间的层归属和通信 |

DIAL-RP的优势应通过实验建立，而不是直接宣称。主要待验证假设是P95 TTFT/TPOT
更低、网络或热状态改变时退化更小、达到相同性能时使用的设备更少。

## 算法

规划器使用位掩码动态规划，同时搜索：

1. 使用哪些设备；
2. 被选设备的先后顺序；
3. 每台设备负责的唯一连续层段；
4. 满足内存约束的最低代价方案。

状态为 `DP[end_layer][device_mask][last_device]`。每次转移选择一个尚未使用的设备，并把后续一个非空连续层段分配给它。最终在所有覆盖完整模型层的状态中选择最低分，因此没有要求四台设备全部出现。

复杂度为：

```text
O(2^D * D^2 * L^2)
```

四台设备、Qwen3-VL-8B 的 36 层规模很小，可以精确搜索，无需启发式或强化学习。

伪代码如下：

```text
for device d:
    for boundary e in 1..L:
        if segment(d, 0..e) satisfies weight+KV memory:
            DP[e][{d}][d] = segment_cost(d, 0..e)

for end, used_mask, last_device:
    for unused device d:
        for next_end in end+1..L:
            s = segment(d, end..next_end)
            if s satisfies memory:
                relax DP[next_end][used_mask U {d}][d]

return the minimum-score state with end=L and popcount(mask)<=max_devices
```

因为每个候选设备最多出现一次，生成的每个远程层段都能被当前DIAL批量RPC直接执行。
对于四设备36层，算法是精确解；若扩展到十几台以上，应改用束搜索或整数规划。

令`rho = risk_weight`，目标函数为：

```text
mean_latency = ttft_weight * mean_ttft_ms
             + tpot_weight * mean_tpot_ms

tail_latency = ttft_weight * tail_ttft_ms
             + tpot_weight * tail_tpot_ms

score = (1 - rho) * mean_latency
      + rho * tail_latency
      + energy_weight * estimated_request_energy_mj
      + remote_device_penalty * selected_remote_workers
```

`mean_*`使用逐层均值、平均RTT和平均有效带宽；`tail_*`使用逐层P95、RTT/协议
P95和带宽P05。各段P95相加是一个“所有已选段同时处于尾部状态”的保守场景，
不是对整条请求P95的无条件统计等式，论文中应称为`conservative tail estimate`。

估计的TTFT包括prefill、第一次decode，以及每个远程连续层段的一次发送和返回；
TPOT包括单token decode和远程层段往返。网络模型遵循当前DIAL的实际路径
`Master -> Worker -> Master`，没有假设尚未实现的Worker-to-Worker直连。

对设备`d`上的连续层段`[i,j)`，代码实际计算：

```text
Cpre(d,i,j) = sum(prefill_ms[d][l]),       l in [i,j)
Cdec(d,i,j) = sum(decode_ms[d][l]),        l in [i,j)

Bpre = 2 * prompt_tokens * hidden_size * dtype_bytes
Bdec = 2 * hidden_size * dtype_bytes

Npre(d) = rtt[d] + protocol[d] + 8*Bpre/(bandwidth[d]*1000)
Ndec(d) = rtt[d] + protocol[d] + 8*Bdec/(bandwidth[d]*1000)

TTFT(d,i,j) = Cpre + Cdec + Npre + Ndec
TPOT(d,i,j) = Cdec + Ndec
```

其中系数2表示hidden activation从Master发送到Worker并返回；Master本地段的`Npre`
和`Ndec`均为0。尾部版本把计算耗时换成逐层P95，把RTT和协议开销换成P95，
把带宽换成P05。完整计划的TTFT、TPOT和能耗是各连续段的对应代价之和。

每台设备的内存约束为：

```text
fixed_memory_mb
+ assigned_layer_memory_mb
+ assigned_layer_kv_cache_mb
<= usable_memory_mb
```

单层KV内存按下面的模型配置自动推导：

```text
kv_mb_per_layer = 2 * num_kv_heads * head_dim
                    * kv_context_tokens * dtype_bytes / 2^20
```

两个因子分别对应K和V。每台设备最多出现一次，因此该设备的固定内存只计一次，
层权重和KV内存也能在生成层段时直接判定。

### 精确性说明

在“每台设备至多一个连续段”的可执行约束下，状态已经记录未来决策所需的全部信息：
已覆盖到哪一层、哪些设备已经使用、最后一台设备是谁。未来层段的计算、通信、能耗和
内存可行性不依赖更早的切分细节；DIAL和EdgeShard-Latency代价按段相加，
EdgeShard-Throughput代价使用单调的`max`聚合。因此Bellman转移不会丢失更优后缀，
四设备场景返回的是该约束搜索空间内的全局最优解，而不是启发式近似。

## 可选启用

不传 `--auto-plan-profile` 时，系统继续读取 `--topology`，原有行为不变：

```bash
./target/release/dial-cli \
  --mode master \
  --topology topology_qwen3vl.yml
```

启用自动规划时，所有参与进程必须读取内容相同的 Profile：

```bash
--auto-plan-profile docs/auto_plan_4devices.example.yml
```

Master可以额外输出规划报告：

```bash
--auto-plan-output result/auto_plan.json
```

报告包含最终设备子集、层范围、均值及保守尾部TTFT/TPOT、内存、搜索状态数和
内存拒绝次数。

规划器支持三种目标：

```bash
--auto-plan-algorithm dial
--auto-plan-algorithm edgeshard-latency
--auto-plan-algorithm edgeshard-throughput
```

不传该参数时默认使用 `dial`。EdgeShard 对比实验的复现范围、启动命令和采集脚本见
`docs/edgeshard_comparison.md`。

## 启动顺序

先启动三个Worker。它们使用相同Profile，但各自使用对应名称：

```bash
# RK3588 Worker 1
./target/release/dial-cli \
  --mode worker \
  --name rk-worker1 \
  --address 0.0.0.0:10128 \
  --model /path/Qwen3-VL-8B-Instruct \
  --cpu \
  --auto-plan-profile /path/auto_plan_4devices.yml
```

```bash
# RK3588 Worker 2
./target/release/dial-cli \
  --mode worker \
  --name rk-worker2 \
  --address 0.0.0.0:10128 \
  --model /path/Qwen3-VL-8B-Instruct \
  --cpu \
  --auto-plan-profile /path/auto_plan_4devices.yml
```

```bash
# Orin Worker，按实际构建启用CUDA
./target/release/dial-cli \
  --mode worker \
  --name orin-worker \
  --address 0.0.0.0:10128 \
  --model /path/Qwen3-VL-8B-Instruct \
  --auto-plan-profile /path/auto_plan_4devices.yml
```

最后启动RK3588 Master：

```bash
SPM_COMPACT_BATCH=1 ./target/release/dial-cli \
  --mode master \
  --api 0.0.0.0:8082 \
  --model /path/Qwen3-VL-8B-Instruct \
  --cpu \
  --vision-rknn /path/vision_448.rknn \
  --vision-fixed-side 448 \
  --auto-plan-profile /path/auto_plan_4devices.yml \
  --auto-plan-output result/auto_plan.json
```

没有被选中的Worker会以空层集合启动，Master不会连接它。若将某设备设为 `enabled: false`，不要启动对应Worker；自动规划模式下，未知Worker名称会直接报错，不会沿用旧行为去错误加载其他Worker的层。

## Profile标定

示例文件中的数字仅用于展示格式，不能作为论文实验数据。每台设备需要在预热后
重复采样（建议每个配置至少100次），实测：

- 每层prefill耗时，测试目标prompt长度；
- 每层decode耗时，测试目标KV长度；
- 每层实际权重内存；
- 模型之外的固定内存；
- Master到Worker的`ping` RTT；
- Master到Worker的`iperf3`有效吞吐；
- 可选的每层prefill/decode能耗。

均值写入`prefill_ms`、`decode_ms`、`rtt_ms`、`bandwidth_mbps`；逐层P95写入
`prefill_p95_ms`和`decode_p95_ms`，RTT P95写入`rtt_p95_ms`，吞吐P05写入
`bandwidth_p05_mbps`，协议开销P95写入`protocol_p95_ms`。若省略尾部字段，规划器
会回退到对应均值，保持与version 1 Profile兼容，但`risk_weight`不会产生额外作用。

尾部样本应覆盖论文声明的目标环境，例如设备热稳态、并发视觉负载以及受控网络抖动；
不能只在空闲冷机上采样后声称算法具备热或网络鲁棒性。

`prefill_ms`、`decode_ms`和`layer_memory_mb`既可以填写一个统一数值，也可以填写与模型层数相同的数组。论文实验建议使用逐层数组。

Profile描述的执行后端必须与实际启动参数一致。例如Orin Profile来自CUDA+GGUF测量时，Worker也必须使用相同后端启动。当前规划器选择设备与层段，不会远程修改各进程的CLI后端参数。

## 当前边界

- 每台被选设备最多负责一个连续Transformer层段；
- prefill和decode使用相同的层归属，避免跨节点迁移KV cache；
- Profile代表节点当前配置好的执行后端；
- 视觉后端尚不参与搜索，其固定内存应计入Master的`fixed_memory_mb`；
- 规划是启动时确定的，不进行运行中的权重迁移；
- 当前输出一个预测最优方案，必须在真实四设备上运行验证，不能把预测值当作实验结果；
- 保守尾部估计不是运行时故障恢复，设备断线仍需要重新启动并生成新计划。

这些限制保证生成的计划与当前DIAL执行模型一致。后续可以在此基础上增加后端模式
候选、视觉分辨率候选和事件触发的安全重规划。

## 论文实验设计

至少报告以下组别，才能分别证明各创新点：

| 实验组 | 配置 | 证明内容 |
|---|---|---|
| Static-Even | 手工均分 | 自动规划是否优于朴素切分 |
| EdgeShard-Latency adapted | `--auto-plan-algorithm edgeshard-latency` | 星型运行时相同情况下，DIAL目标函数的收益 |
| DIAL-Mean | `dial`且`risk_weight: 0` | 设备子集、TTFT/TPOT和真实通信建模的收益 |
| DIAL-RP | `dial`且`risk_weight: 0.25`等 | 风险项对P95和扰动退化的收益 |
| DIAL-RP no subset | `min_devices: 4`且`max_devices: 4` | 自动跳过设备的收益 |

主表报告TTFT median/P95、TPOT median/P95、总延迟、decode tokens/s、峰值内存、
能耗和实际使用设备数。敏感性实验扫描`risk_weight = 0, 0.25, 0.5, 0.75, 1`，
并分别施加网络限速/延迟抖动和RK3588热稳态负载。所有策略都必须在同一DIAL提交、
同一模型精度、同一prompt/output长度和同一运行后端上执行。
