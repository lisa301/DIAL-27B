# DIAL 论文对比实验调研与推荐方案

调研日期：2026-07-30

## 1. 结论先行

DIAL 当前最适合定位为“面向 RK3588/Jetson 等异构边缘节点的多模态大模型协同部署系统”，而不是通用数据中心 LLM serving 系统。论文需要回答四个问题：

1. 单设备放不下模型时，DIAL 是否能让多个边缘设备完成正确推理？
2. 相同硬件和模型下，DIAL 的层段放置与聚合是否优于朴素切分？
3. CPU/GPU/NPU 混合执行是否降低 TTFT、提高 decode TPS，并保持模型质量？
4. 在什么带宽、时延、图像尺寸和节点数量下，协同推理有收益或失效？

最重要的主基线不是 vLLM 或 TensorRT-LLM，而是：单设备执行、平均层切分、按内存比例切分、按实测计算能力切分，以及 DIAL 完整方法。外部系统中，EdgeShard、LinguaLinked 和 AdaptLink 与研究问题最接近；但它们很难直接运行 Qwen3-VL + RKNN，因此应复现其核心放置策略，并明确标注为“strategy reimplementation”，不能声称运行了原系统。

## 2. 推荐对比对象

| 类别 | 对比对象 | 是否放主表 | 公平实现方式 | 目的 |
|---|---|---:|---|---|
| 单机下界 | 单 RK3588，CPU 软件路径 | 是 | 同一模型、精度、prompt、生成长度 | 设备内基础性能和内存可行性 |
| 单机强基线 | 单 Jetson Orin，GPU 软件路径 | 是 | 同一模型和生成参数 | 判断分布式是否真正加速；也是性能上界参照 |
| 异构单机 | RK3588 CPU、Vision NPU、CPU+NPU | 是 | 仅改变后端 | 验证异构辅助收益 |
| 朴素分布式 | EqualSplit：各节点平均分层 | 是 | 连续层，其他机制完全相同 | LinguaLinked 使用的基本基线 |
| 内存感知切分 | Memory-Proportional：按可用内存比分层 | 是 | 连续层，满足容量约束 | 对比只考虑“能装下”而不考虑速度的策略 |
| 性能感知切分 | Profile-Greedy：按逐层实测耗时分配 | 是 | 离线 profile 后平衡各节点计算量 | 证明 DIAL 预设层段的选择价值 |
| 近似最优切分 | Exhaustive/DP：枚举连续切点最小化预测延迟 | 强烈建议 | 2–3 节点、36 层时可枚举所有切点 | 对应 EdgeShard/LinguaLinked 的优化思路，给出 optimality gap |
| DIAL 消融 | SingleOp、Batch、CompactBatch/CompactRangeBatch | 是 | 同一拓扑，仅改协议模式 | 隔离层段聚合和紧凑协议收益 |
| 开源工程系统 | distributed-llama 或 exo | 附表/文本模型实验 | 换用双方共同支持的 Llama/Qwen 文本模型及同精度 | 展示与公开工程系统的关系，不强行用于 Qwen3-VL 主结论 |
| 数据中心系统 | vLLM、TensorRT-LLM | 否，相关工作即可 | 除非有相同设备可运行 | 面向并发 GPU serving，和当前单请求边缘问题不公平 |
| 互联网协作 | Petals | 否，相关工作即可 | 不建议实测 | 面向公网大模型协作，网络和硬件假设不同 |

外部资料：

- [EdgeShard](https://arxiv.org/abs/2405.14371)：联合设备选择与模型切分，用动态规划优化延迟或吞吐，是 DIAL 最直接的算法基线。
- [LinguaLinked](https://aclanthology.org/2024.acl-demos.16/)：移动设备分布式 LLM 推理，包含均匀切分、优化模型分配和运行时负载均衡。
- [AdaptLink](https://openreview.net/forum?id=qN95zN4zle)：异构边缘设备上的分布式多模态模型推理，是最接近 DIAL 多模态定位的工作。
- [HexGen](https://proceedings.mlr.press/v235/jiang24f.html)：异构 GPU 环境的非对称张量/流水线并行，适合相关工作和异构调度讨论，但不适合作为 RKNN 主基线。
- [Jupiter](https://arxiv.org/abs/2504.08242)：同时针对 prefill 和 decode 的边缘协同 LLM 推理，可作为强相关工作；其 speculative/pipeline 机制与 DIAL 串行层段执行不同。
- [distributed-llama](https://github.com/b4rtaz/distributed-llama)：可复现的多机 LLM 工程基线，支持 ARM/x86、CPU/Vulkan，但不是 Qwen3-VL/RKNN 同构实现。
- [exo](https://github.com/exo-explore/exo)：异构设备自动发现与拓扑感知切分，适合作为工程参照；主要后端和 DIAL 不同。

## 3. 必做实验（按论文价值排序）

### E1. 端到端主结果

模型分别用 Qwen3-VL-2B 和 8B；输入分为纯文本与单图问答。至少比较：

1. 单 RK3588 CPU；
2. 单 RK3588 CPU + Vision NPU；
3. 单 Orin GPU；
4. RK3588 + Orin EqualSplit；
5. RK3588 + Orin Profile-Greedy；
6. DIAL 完整配置；
7. 两台 RK3588 + Orin 的 DIAL 配置。

报告 TTFT、prefill latency、TPOT（ms/token）、decode TPS、端到端 latency、峰值内存、平均/峰值功率、单请求能耗和是否 OOM。当前系统已有 `ttft_s` 和 `decode_tps`，但还应单独暴露 vision、prefill、decode、序列化、网络等待的耗时。边缘交互式推理采用 TTFT 与后续 token 速率是合理做法，可参考 [MLPerf Client](https://mlcommons.org/benchmarks/client/) 的指标定义。

关键表述：若分布式慢于单 Orin，但能运行单 RK3588 无法容纳的 8B 模型，结论应写“resource pooling / deployment feasibility”，不能写“全面加速”。

### E2. 层切分策略对比

固定模型、后端、协议和带宽，只改变层的归属：EqualSplit、Memory-Proportional、Profile-Greedy、DP/Exhaustive、DIAL chosen placement。

对两节点枚举切点 `k=0..L`；三节点枚举 `(k1,k2)`。画出“切点—TTFT”和“切点—TPOT”曲线，同时给出各策略相对实测最优点的 gap。该实验比只测 0–11、0–17 两个点有说服力，因为它能证明层段选择不是人工挑出的偶然结果。

建议模型：2B 做全切点扫描，8B 只测 profile 预测最优点附近 ±2/±4 层及几个代表点，以控制实验时间。

### E3. 通信机制消融

在完全相同的层切分上比较：

- SingleOp：逐层一次请求；
- Batch：连续远端层一次请求；
- CompactBatch；
- CompactRangeBatch（若这是最终完整机制）。

报告每生成一个 token 的远程请求数、写/读字节数、序列化时间、socket 时间、远端计算时间、非计算开销、TPOT。理论请求数和实测值应互相校验。实验至少覆盖 1、6、12、18 个连续远端层，证明聚合收益随层段长度变化。

### E4. 网络适用边界

不仅限制带宽，还要同时扫描 RTT：

- 带宽：1、5、10、50、100、1000 Mbps；
- 额外 RTT：0、5、20、50 ms；
- prompt：128 和 1024 tokens；
- 阶段：prefill 与 decode 分开报告。

输出 heatmap，并报告 DIAL 相对单 RK3588 的 break-even boundary。现有 `test.xls` 已有 100 Kbps–100 Mbps 数据，可以保留，但每点需重复且加入置信区间。100 Kbps 可作为极端压力点，不宜占主图主要区域。

### E5. 异构后端与量化消融

分别比较：CPU-only、Vision-RKNN、Text-QKV-RKNN、Text-MLP-RKNN、NPU+CPU、NPU+CPU+GPU。每种配置同时报告性能、内存、功率与质量。

必须增加数值正确性：对每个 RKNN/RKLLM 子图，用 FP16/FP32 软件输出作参考，报告 cosine similarity、normalized RMSE、max absolute error；再对完整生成报告固定 greedy decoding 下的 token agreement rate。只报告速度不足以证明量化路径可用。

### E6. 多模态质量与图像尺度折中

推荐用官方支持的工具接入 DIAL 的 OpenAI 风格 API。Qwen3-VL 官方使用 [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) 和 [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval)。无需铺满所有榜单，选择三类互补任务：

- 综合感知/推理：MMBench 或 MMMU（资源紧张可固定抽样）；
- OCR/细节：TextVQA 或 DocVQA；
- 自有应用：摔倒检测数据集，报告 Precision、Recall、F1、误报率，而不是只给几个案例。

尺寸设为 336、384、448、512（仅使用各后端实际支持的尺寸），报告准确率/F1、视觉 token 数、vision latency、prefill latency、TTFT、能耗，画 Pareto frontier。

注意：当前代码只找到 `vision_max_side` 和 `vision_fixed_side`，未发现“给定 TTFT 约束自动选最大尺度”的在线控制器。因此当前只能声称“图像尺度限制/选择”，不能把“延迟约束自动策略”作为已实现贡献。若实现控制器，再比较 Fixed-336、Fixed-448、Fixed-512、Always-smallest、DIAL adaptive，并报告 deadline satisfaction ratio 和平均质量。

### E7. 扩展性和鲁棒性

节点数用 1、2、3；分别报告可承载最大模型、总内存利用率、TTFT、TPOT、speedup、parallel efficiency。由于自回归层流水线对单请求未必随节点数加速，应把“容量扩展”和“速度扩展”拆开。

鲁棒性至少做一个简单实验：worker 在请求前不可达时是否快速失败、错误是否明确；运行时断连是否能回退或终止。若没有恢复机制，就如实作为限制，不需要包装成容错贡献。

### E8. 视频帧复用（可作为应用实验，不宜混入核心主表）

现有结果显示帧复用能明显降低响应时间，但需要补质量约束：在不同相似度阈值下测真实推理比例、平均响应时间、F1/漏报率，尤其统计“场景从正常变为摔倒”时被错误复用的比例和检测延迟。否则仅展示 TTFT 降低容易被审稿人认为是跳过计算得到的平凡收益。

## 4. 统一实验协议

- 固定模型 checkpoint、tokenizer、图片预处理和生成参数；性能测试建议 greedy decoding，避免采样导致输出长度和路径波动。
- 输入长度设为 128、512、1024 tokens；输出固定 64 或 128 tokens。图文输入还要记录视觉 token 数。
- 每个配置预热 3–5 次，正式运行至少 20 次；报告 median、P95 和 95% bootstrap CI，而不是只报单次值。
- 性能与质量分开跑：性能使用固定输出长度；质量使用基准规定的生成配置。Qwen3-VL 官方评测配置可参考[官方 README](https://github.com/QwenLM/Qwen3-VL/blob/main/README.md)，不要混用性能测试的 greedy 设置来声称复现官方分数。
- 清楚区分 cold start、warm model、warm vision cache。主结果建议 warm model、cold request；另表报告模型加载时间。
- 所有网络实验记录真实 `iperf3` 吞吐和 `ping` RTT，不只记录 `tc` 配置值。
- 功耗按整机墙上功率最规范；若只能读取板载传感器，明确测量边界。报告 Joule/request 和 Joule/output-token。
- 每次运行保存原始 JSON/CSV、commit hash、拓扑文件、命令行、设备频率、温度及是否发生降频。

## 5. 建议论文主表与主图

控制正文篇幅时，保留以下 4 表 5 图即可：

- 表 1：硬件、软件、模型和网络环境；
- 表 2：端到端主结果（性能、内存、能耗、是否 OOM）；
- 表 3：与 EqualSplit、Memory-Proportional、Profile-Greedy、DP 的策略对比；
- 表 4：完整系统与各消融的质量保持结果；
- 图 1：不同切点的 TTFT/TPOT 曲线；
- 图 2：SingleOp → Batch → CompactRangeBatch 通信开销分解；
- 图 3：带宽×RTT 的 break-even heatmap；
- 图 4：图像尺度的质量—TTFT Pareto 曲线；
- 图 5：1/2/3 节点的最大可部署模型与性能扩展。

## 6. 对现有数据和草稿的具体判断

`result/test.xls` 中单 Orin 的 Qwen3-VL-8B 为 TTFT 8.916 s、TPS 6.59、decode TPS 7.89；RK3588+Orin 数据约为 TTFT 5.808–12.817 s、TPS 1.282–2.236、decode TPS 1.447–2.553。这个结果有两个可用结论：

1. 某些切分点能比单 Orin 获得更低 TTFT，但 decode 明显更慢，应拆分 prefill/vision/decode 解释原因；
2. 分布式的核心价值很可能是边缘资源池化和异构部署，不是所有指标均超越强 GPU 单机。

现有“限速测试”趋势合理，但缺重复次数、误差线、RTT 和通信字节；“FNN+DOWN 量化”“lm_head 量化”命名需要改为准确算子名称，并补精度结果；“视频帧复用”必须补漏检风险。

当前完整草稿最大缺口不是再加更多系统名称，而是缺少：公平的策略基线、质量保持、能耗、统计重复、自动尺度机制的真实实现，以及对负结果的诚实边界分析。完成 E1–E6，已经足以形成一篇结构完整的系统实验论文；E7–E8 可按篇幅和实现进度选做。
