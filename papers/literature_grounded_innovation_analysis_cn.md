# DIAL 可创新方向：基于已发表论文的证据分析

调研日期：2026-07-30

## 1. 调研口径

本文把“正式发表”限定为可核验的会议或期刊论文，优先引用 ACL/NAACL/EMNLP、ICML、CVPR/ICCV、WACV、SoCC、NSDI 等官方论文页面。只有预印本或 workshop 投稿的工作单列说明，不用它们证明某个方向已经形成正式共识。

判断一个方向是否适合 DIAL，不仅看它是否热门，还看三个条件：

1. 已有工作是否已经完整解决相同问题；
2. DIAL 的 RK3588、Jetson Orin、RKNN、Qwen3-VL 和层段通信是否形成不同研究约束；
3. 仓库中是否已有足够实现基础，能在合理工作量内做成可验证机制。

## 2. 已发表工作覆盖了什么

| 研究方向 | 代表性正式论文 | 已解决的问题 | 对 DIAL 的含义 |
|---|---|---|---|
| 移动设备分布式 LLM | [LinguaLinked, ACL Demo 2024](https://aclanthology.org/2024.acl-demos.16/) | 线性优化模型分配、结构化传输、运行时负载均衡 | “自动层切分”本身不能再作为主要创新 |
| 异构廉价设备流水线 | [HPipe, NAACL Industry 2024](https://aclanthology.org/2024.naacl-industry.1/) | 在异构设备和昂贵通信下，通过token维流水线处理长上下文 | 仅做异构连续分层也不够新；需加入多模态/NPU约束 |
| 异构GPU分布式推理 | [HexGen, ICML 2024](https://proceedings.mlr.press/v235/jiang24f.html) | 非对称tensor/pipeline并行及约束优化调度 | 通用异构调度已有强工作，DIAL不能泛称“首个异构调度” |
| 分布式稀疏通信 | [DISCO, WACV 2024](https://openaccess.thecvf.com/content/WACV2024/html/Qin_DISCO_Distributed_Inference_With_Sparse_Communications_WACV_2024_paper.html) | 训练得到层内稀疏通信，在CV网络上减少传输与计算 | 中间特征压缩不是空白，但其模型、并行粒度和训练要求与DIAL不同 |
| 多模态阶段解耦 | [ModServe, SoCC 2025](https://doi.org/10.1145/3772052.3772254) | 根据模态和推理阶段做资源解耦，面向生产级LMM serving | “视觉和语言分开部署”本身已有工作；边缘编译式NPU约束仍可形成差异 |
| 多模态KV压缩 | [MEDA, NAACL 2025](https://aclanthology.org/2025.naacl-long.125/) | 按层注意力熵动态分配多模态KV预算，最高报告72%内存降低 | 单做KV压缩不适合作为DIAL系统主创新，可作为组合机制 |
| 视觉token剪枝 | [TopV, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Yang_TopV_Compatible_Token_Pruning_with_Inference_Time_Optimization_for_Fast_CVPR_2025_paper.html)、[DivPrune, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Alvar_DivPrune_Diversity-based_Visual_Token_Pruning_for_Large_Multimodal_Models_CVPR_2025_paper.html)、[ATP-LLaVA, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Ye_ATP-LLaVA_Adaptive_Token_Pruning_for_Large_Vision_Language_Models_CVPR_2025_paper.html) | 训练无关、实例自适应或层自适应视觉token减少 | 已非常拥挤，简单删token不构成新贡献 |
| 视觉token剪枝反思 | [Wen et al., Findings ACL 2025](https://aclanthology.org/2025.findings-acl.802/) | 指出部分剪枝法甚至不如随机选择，且评测协议存在问题 | 若采用视觉token控制，必须与随机、固定比例和强剪枝基线公平比较 |
| 视频帧选择 | [M-LLM Frame Selection, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Hu_M-LLM_Based_Video_Frame_Selection_for_Efficient_Video_Understanding_CVPR_2025_paper.html)、[Flexible Frame Selection, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Buch_Flexible_Frame_Selection_for_Efficient_Video_Reasoning_CVPR_2025_paper.html)、[Q-Frame, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_Q-Frame_Query-aware_Frame_Selection_and_Multi-Resolution_Adaptation_for_Video-LLMs_ICCV_2025_paper.html) | 查询相关选帧、自适应帧数、多分辨率 | DIAL当前的相似帧结果复用只能算工程优化，不能声称首创 |
| 中间表示隐私 | [TextFusion, EMNLP 2022](https://aclanthology.org/2022.emnlp-main.572/)、[IR-AIA, Findings ACL 2026](https://aclanthology.org/2026.findings-acl.1172/) | 中间表示可被恢复或用于推断敏感属性，并提出隐私机制/攻击 | DIAL不能把“只传hidden state”直接写成保护隐私 |
| 边缘量化 | [Agile-Quant, AAAI 2024](https://ojs.aaai.org/index.php/AAAI/article/view/29860)、[AWQ, MLSys 2024](https://proceedings.mlsys.org/paper_files/paper/2024/file/42a452cbafa9dd64e9ba4aa95cc1ef21-Paper-Conference.pdf) | 面向边缘或端侧的权重量化和激活感知量化 | 单纯把权重量化成INT8/GGUF不是研究创新，需要联合系统决策或新误差机制 |

补充边界：EdgeShard目前最常引用的版本是[arXiv预印本](https://arxiv.org/abs/2405.14371)；AdaptLink是[AAAI 2025 Workshop投稿](https://openreview.net/forum?id=qN95zN4zle)。它们与DIAL高度相关，应该在相关工作和实验基线中讨论，但应准确标注发表状态。

## 3. 最推荐的创新方向

### 方向A：面向编译式NPU约束的多模态阶段—层段联合规划

**推荐等级：最高。**

已有工作的边界：

- LinguaLinked优化移动设备上的模型段分配，但研究对象是文本LLM，没有Qwen3-VL视觉编码、deepstack注入和RKNN固定shape约束。
- HPipe研究异构廉价设备的长上下文pipeline，但没有CPU/GPU/NPU子图编译约束。
- HexGen研究异构GPU和跨数据中心，不处理RKNN这类“只能运行预编译固定子图”的边缘NPU。
- ModServe已经提出模态和阶段感知的资源解耦，因此DIAL不能只说“视觉放NPU、文本放GPU”；必须把边缘后端可行性约束正式加入优化问题。

可成立的新问题是：

> 在视觉编码器、deepstack注入层、prefill、decode、KV cache、固定shape RKNN子图、设备内存和网络带宽共同约束下，联合选择图像尺度、后端、连续文本层段和节点位置，最小化TTFT/TPOT/能耗。

与普通自动切层相比，规划变量至少包括：

- 视觉后端：CPU、GPU或RKNN NPU；
- 图像输入bucket：336/384/448/512中硬件实际支持的集合；
- 文本层位置：Master、Worker0、Worker1；
- 文本子图：软件、QKV-RKNN、MLP-RKNN、RKLLM前缀；
- 阶段：prefill和decode允许使用不同后端策略；
- 约束：设备内存、KV cache、RKNN固定输入、deepstack必须在指定层注入、网络RTT/带宽；
- 目标：加权TTFT、TPOT、能耗，或者在质量下限下优化性能。

为什么适合当前代码：DIAL已经有静态拓扑、Vision RKNN、QKV/MLP RKNN、RKLLM前缀、CPU/GPU路径和分阶段性能指标，缺的是统一profiling、代价模型、求解器和自动生成拓扑。

要形成论文贡献，必须实现：

1. 各设备/后端的prefill、decode、内存、能耗profile；
2. 包含通信和编译后端可行性的代价模型；
3. 两节点枚举、三节点动态规划或整数规划；
4. 自动生成部署拓扑及后端配置；
5. 与EqualSplit、Memory-Proportional、LinguaLinked-style、仅文本层规划比较；
6. 预测与实测误差，以及网络变化时的决策变化。

这条创新最容易把现有工程提升为研究方法，也是最适合作为整篇论文主线的方向。

### 方向B：带宽与数值误差联合感知的中间hidden传输

**推荐等级：高，可与方向A组合。**

已有工作的边界：DISCO通过训练让CV模型的层内通信稀疏；已有tensor-parallel通信量化研究主要针对GPU集合通信。DIAL是层间串行协同，传输对象是Qwen3-VL文本hidden state，网络可能从1 Mbps变化到1 Gbps，并且Master/Worker后端精度不同。

可研究的问题是：

> 根据prefill/decode阶段、网络带宽和层敏感度，在FP16、INT8、INT4或稀疏传输间自动选择，最小化网络时间，同时约束最终生成质量下降。

建议机制：

- Prefill hidden较大，优先使用per-channel INT8或分组INT8；
- Decode每次hidden只有一个token，RTT可能比字节数更重要，低比特压缩未必有收益；
- 离线校准每个候选切点的量化敏感度；
- 在线根据 `RTT + bytes/bandwidth + encode/decode cost` 选择传输格式；
- 协议头显式携带dtype、scale、zero-point和压缩版本；
- 质量约束使用hidden cosine、token agreement和下游任务分数。

论文价值来自“系统发现”：在哪个带宽以下压缩才获益、prefill和decode是否应采用不同精度、哪些切层对量化最敏感。仅实现FP16转INT8而没有自适应策略和质量约束，不足以成为创新。

### 方向C：面向Qwen3-VL的质量约束视觉预算—硬件协同控制

**推荐等级：中高，但必须避开单纯视觉token剪枝。**

CVPR 2025已经有TopV、DivPrune、ATP-LLaVA、EfficientLLaVA等大量方法；Q-Frame还联合了帧选择与多分辨率。因此“根据图片复杂度调整尺寸”或“删视觉token”不能直接宣称新颖。

DIAL可以把问题改成部署系统问题：

> 在真实边缘CPU/NPU固定shape模型集合上，给定任务质量下限和TTFT deadline，联合选择图像分辨率、视觉token预算和执行后端。

与现有token剪枝论文的区别必须是：

- 决策针对真实RK3588 NPU编译模型bucket，而非高端GPU上的抽象FLOPs；
- 同时考虑视觉编码耗时、文本prefill、KV内存和网络传输；
- 使用deadline satisfaction ratio，而不只报告平均FLOPs；
- 针对摔倒检测、OCR和通用VQA分别学习/标定质量曲线；
- 与Fixed-336/448/512、Random Pruning、TopV/DivPrune式方法比较。

如果没有公开任务质量评测，这一方向很容易退化成调参，不建议单独作为主贡献。

### 方向D：多模态KV cache的分布式放置与压缩

**推荐等级：中。**

MEDA已正式研究多模态KV预算分配，后续多模态KV压缩工作也很多。因此单做KV剪枝不新。DIAL可以研究“KV跟随层部署”之外的边缘系统问题：

- 每个Worker只保存自己层的KV；
- 根据设备内存和上下文长度联合决定层位置与KV精度；
- 多轮对话中复用文本KV和视觉embedding；
- 节点内存不足时选择KV量化、淘汰或迁移；
- 规划时同时考虑权重内存和KV随上下文增长的内存。

真正的新点应是“层段放置—KV预算—设备内存联合优化”，而不是复现MEDA。它适合长上下文或多轮会话论文，但当前DIAL主要是单请求且每次API请求会reset，工作量较大。

### 方向E：隐私风险感知的多模态切分点选择

**推荐等级：中，研究新颖性较好但会改变论文主题。**

TextFusion和2026年的IR-AIA已经证明中间表示并不天然隐私。对分布式多模态系统，切点不仅影响性能，也影响图像/文本信息可恢复程度。

可研究：

> 对候选切层测量文本属性泄露或图像重建风险，在延迟、内存和隐私约束下联合选择切点及hidden保护强度。

可行机制包括中间表示量化、噪声、token融合或可信节点分组。但要成立，必须实现攻击者和定量隐私指标，不能只加TLS。TLS保护链路窃听，却不能防止参与计算的恶意Worker读取hidden。

这条方向有现实意义，但需要安全实验和攻击复现，可能使论文从系统优化转向隐私计算，不建议与其他五六个机制同时展开。

### 方向F：事件安全约束的视频连续推理复用

**推荐等级：中低，适合作为第二篇或应用章节。**

CVPR/ICCV 2025已经大量研究查询相关选帧和多分辨率。DIAL当前根据帧相似度复用上一次答案，速度收益本质上来自跳过推理；如果不测状态变化漏检，就不是可靠创新。

可转化为研究问题：

> 在摔倒检测等状态转换任务中，根据低成本视觉变化、历史置信度和最大跳帧预算决定是否复用，约束事件漏检率和检测延迟。

需要与Uniform Sampling、Scene Difference、CLIP Similarity、Q-Frame式查询相关选择比较，报告正常到摔倒转折点的检测延迟、漏报率和能耗。该方向有应用价值，但与DIAL核心分布式层推理联系较弱。

## 4. 不建议作为独立核心创新的内容

以下机制可以作为系统组件或消融，但已有工作充分，不应单独写成主要创新：

- 自动层切分：LinguaLinked、HPipe、HexGen等已经覆盖；
- 连续层分配：属于常见pipeline/model partition方式；
- OpenAI兼容API、SSE流式输出、YAML拓扑；
- 视觉编码器放NPU：需要联合决策才能形成研究贡献；
- 模型INT8/GGUF量化：已有AWQ、Agile-Quant等成熟工作；
- 固定图片到448或限制最大边长；
- 简单视觉token剪枝：CVPR 2025已有多个强基线；
- 相似帧直接复用答案；
- “只传hidden所以保护隐私”：已被正式隐私研究反驳；
- 只比较SingleOp和Batch：这是有价值的系统消融，但单独创新强度有限。

## 5. 三种可形成论文的组合方案

### 方案一：最稳妥的系统论文

主贡献：方向A“编译式NPU约束的多模态联合规划”。

辅助贡献：现有CompactRangeBatch通信聚合；Vision/Text NPU混合执行；完整可观测性。

实验主线：规划质量、端到端性能、内存可部署性、网络敏感性、质量保持、能耗。

优点：最贴合当前代码，新增工作集中，论文叙事统一。

### 方案二：网络与边缘协同论文

主贡献：方向B“阶段与带宽自适应hidden压缩”。

辅助贡献：方向A的简化自动切点；CompactRangeBatch。

实验主线：1 Mbps–1 Gbps、不同RTT、FP16/INT8/INT4、压缩开销、质量和break-even边界。

优点：现有项目已经有网络限速和传输字节统计，实验基础较好。

### 方案三：多模态实时应用论文

主贡献：方向C“质量/Deadline约束的视觉预算与硬件控制”。

辅助贡献：方向F“事件安全的视频复用”。

实验主线：MMBench/TextVQA/摔倒数据，分辨率与token预算，TTFT deadline、F1和能耗。

风险：视觉token与选帧方向竞争激烈，必须实现强质量评测和现有方法基线。

## 6. 最终建议

对于当前DIAL，优先级建议为：

1. **方向A：多模态阶段—层段—硬件联合规划**；
2. **方向B：阶段/带宽/误差联合感知的hidden传输**；
3. 方向C：质量约束视觉预算；
4. 方向D：分布式KV放置；
5. 方向E：隐私感知切点；
6. 方向F：事件安全视频复用。

最合理的论文不是把六个方向全部堆入系统，而是选A作为唯一主问题，B作为第二个关键机制。这样论文可以提出一个清楚的论断：

> 现有异构LLM规划没有同时处理多模态阶段依赖和编译式边缘NPU约束；DIAL通过阶段—层段—后端联合规划，并结合阶段自适应中间表示传输，在真实RK3588/Orin集群上改善模型可部署性与延迟—质量折中。

这句话只有在完成正式的联合代价模型、自动求解、强基线和质量实验后才能写入摘要。当前代码提供了实现基础，但尚未完成这一研究贡献。
