# Climber-Pilot: 迈向更优指令跟随的非短视生成式推荐模型


本文介绍了 Climber-Pilot: 迈向更优指令跟随的非短视生成式推荐模型。核心内容：


关键发现：

---


郭达 王诗佳 肖强\*
网易云音乐 网易云音乐 网易云音乐
中国杭州 中国杭州 中国杭州
> guoda@corp.netease.com wangshijia1@corp.netease.com hzxiaoqiang@corp.netease.com

任尹涛 李伟胜 徐松培
网易云音乐 网易云音乐 网易云音乐
中国杭州 中国杭州 中国杭州
renyintao@corp.netease.com liweisheng01@corp.netease.com xusongpei@corp.netease.com

岳明 黄斌 吴冠林
网易云音乐 网易云音乐 网易云音乐
中国杭州 中国杭州 中国杭州
yueming03@corp.netease.com huangbin02@corp.netease.com wuguanlin03@corp.netease.com

罗川江
网易云音乐
中国杭州
luochuanjiang03@corp.netease.com

## 摘要

生成式检索已成为推荐系统中一种有前景的范式，相较于传统的双塔架构提供了更优的序列建模能力。然而，在大规模工业场景中，此类模型常受固有限制——短视（myopia）：由于单步推理和严格的延迟约束，模型倾向于将多样化的用户意图坍缩到局部最优预测中，无法捕捉长期和多item的消费模式。此外，真实世界的检索系统必须遵循明确的检索指令，例如类别级控制和策略约束。将此类指令跟随行为融入生成式检索仍然具有挑战性，因为现有的条件化或事后过滤方法常常会损害相关性或效率。在这项工作中，我们提出了 Climber-Pilot，一个统一的生成式检索框架，旨在解决上述两个局限。首先，我们引入了时间感知多item预测（Time-Aware Multi-Item Prediction, TAMIP），这是一种新颖的训练范式，旨在缓解生成式检索中固有的短视问题。通过时间感知掩码将长期多item的前瞻性知识蒸馏到模型参数中，TAMIP 在保持高效单步推理的同时缓解了局部最优预测。其次，为了支持灵活的指令跟随检索，我们提出了条件引导稀疏注意力（Condition-Guided Sparse Attention, CGSA），通过稀疏注意力直接将业务约束融入生成过程，而无需引入额外的推理步骤。在网易云音乐（最大的音乐流媒体平台之一）上进行的大量离线实验和在线 A/B 测试表明，Climber-Pilot 持续优于最先进的基线方法，核心业务指标提升了 4.24%。

### CCS 概念

- 信息系统 $\to$ 推荐系统。

### 关键词

生成式推荐, 检索, 指令跟随, 推荐系统

### ACM 引用格式

DaGuo, ShijiaWang, QiangXiao, YintaoRen, WeishengLi, SongpeiXu, MingYue, BinHuang, GuanlinWu, and ChuanjiangLuo. 2026. Climber-Pilot: A Non-Myopic Generative Recommendation Model Towards Better Instruction-Following. In Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2 (KDD '26), August 09–13, 2026, Jeju Island, Republic of Korea. ACM, New York, NY, USA, 10 pages. https://doi.org/10.1145/3770855.3818340

## 1 引言

大规模推荐系统依赖检索模型来高效地将海量item库缩小为一小组候选item，供下游排序使用。在工业环境中，检索需要在严格的延迟约束下运行，同时每天服务数十亿次请求，因此效率和鲁棒性都是首要关注点。传统的检索系统主要构建在双塔架构 [3, 29, 39] 之上，这种架构独立地对用户和item进行编码，并通过向量相似性搜索进行候选选择。虽然计算效率高，但这种架构从根本上限制了检索模型的表达能力，因为丰富且异构的用户交互历史必须被压缩到单个静态表征中。尽管近期的工业努力已经探索了多模态特征融合 [38]、多任务学习 [34] 和跨域兴趣迁移 [23] 来丰富该范式下的表征，但底层的表达能力瓶颈依然存在。

生成式检索 [13, 16, 28] 近年来作为一种有前景的替代范式出现，将候选生成重新定义为一个条件序列建模问题。通过自回归地建模用户交互序列，生成式检索器能够捕捉细粒度的时序依赖关系和复杂的行为模式，而这些在基于嵌入的检索框架中难以表达。此外，生成式方法支持端到端优化，无需在推理时依赖近似最近邻搜索，提供了一个统一的检索建模框架。

尽管有诸多优势，生成式检索模型在大规模工业系统中部署时仍面临重大挑战。一个根本性的局限是我们称之为**固有短视**（inherent myopia）的问题 [8, 14, 21]。在实践中，延迟约束通常将生成式检索器限制为在服务时仅进行单步推理。因此，大多数现有方法采用下一item预测（Next-Item Prediction）目标进行训练，鼓励模型优化即时相关性而非长期用户意图 [15]。这种训练-推理不匹配常常导致局部最优预测，多样化的未来兴趣被坍缩到一小组高概率的item上，限制了探索性和覆盖率。

另一个同样关键的挑战来自于工业检索系统的运营需求。除了相关性，检索模型通常需要在候选生成过程中遵循明确的检索指令 [1, 37]，例如类别级约束、策略规则或业务驱动的控制信号。在传统检索流程中，这些需求通常通过启发式过滤或基于规则的后处理来解决。然而，此类方法不适合生成式检索模型。简单地以控制信号为条件进行生成往往精度不足，而事后过滤则可能降低相关性或引入额外的推理开销，从而削弱生成式检索的效率优势。

解决这些挑战需要重新思考生成式检索系统中**在哪里**以及**如何**处理复杂性。我们认为，关键不在于引入多步推理或依赖流程级启发式方法，而在于将复杂性从推理转移到训练，从事后约束处理转移到模型内部归纳偏置。遵循这一原则，我们提出了 Climber-Pilot，一个为大规模工业部署设计的统一生成式检索框架。Climber-Pilot 使生成式检索器能够捕捉长期、多item的用户意图 [6] 并遵循明确的检索指令，同时保持单步推理的效率。

Climber-Pilot 通过两个互补的设计选择实现了这一点。首先，它通过在训练期间显式建模基于批次的曝光和延迟消费模式，将长期、多item的前瞻性知识蒸馏到模型参数中。这种训练时蒸馏缓解了单步目标诱导的短视行为，使模型能够内化面向未来的信号，而不会增加推理复杂度。其次，Climber-Pilot 通过注意力级别的控制将检索指令直接嵌入生成过程，使约束能够以细粒度和高效的方式影响候选生成，而无需诉诸事后过滤。

我们通过大量的离线实验和在线 A/B 测试在大规模工业推荐系统中评估了 Climber-Pilot。结果表明，Climber-Pilot 在检索质量和可控性方面均持续优于最先进的生成式检索基线，在严格的延迟约束下为核心业务指标带来了一致的提升。我们的贡献概括如下：

- 我们识别了固有短视作为生成式检索在单步推理下的根本局限，并揭示了在批次曝光设置下的消费滞后（consumption lag）进一步加剧了工业系统中的这一问题。
- 我们提出了一种训练时蒸馏范式，使生成式检索器能够内化长期、多item的用户意图，而无需增加推理复杂度。
- 我们引入了一个注意力级别的条件框架，支持在生成过程中进行显式指令跟随，同时允许将延迟的消费信号纳入检索过程。
- 我们通过大量的离线实验和真实世界生产系统中的大规模在线部署验证了所提出的框架。

## 2 相关工作

以 SASRec [17] 和 BERT4Rec [30] 为代表的序列推荐模型，通过自注意力机制建立了捕捉用户行为中长期依赖关系的标准。因此，范式已经转向生成式推荐框架，如 P5 [9] 和 VQ-Rec [12]，它们将检索重新定义为序列到序列的生成任务。像 OneRec [5] 和 HSTU [40] 等工业适配方案进一步证明了该方法在高吞吐量环境中的可扩展性。除了架构上的进步，近期的研究还通过渐进式残差量化 [32, 35] 和层次化结构感知多模态建模 [24] 改进了语义标识符设计，为生成式检索提供了补充信号。

尽管取得了这些进展，生成式检索模型主要依赖于下一item预测（Next Item Prediction, NIP）目标。由于 NIP 优化固有的贪婪特性，模型预测倾向于坍缩到高频率的"下一个"item，这导致了多样性和长期参与度方面的问题。虽然传统的多兴趣模型如 MIND [19] 和 ComiRec [2] 通过静态胶囊路由解决了多样性问题，但它们缺乏在生成式框架内建模动态兴趣轨迹的能力。LUM [37] 和 PinRec [1] 利用生成式架构的上下文学习能力探索了条件检索。然而，它们的方法主要局限于基于粗粒度特征（如动作类型）调整检索结果。关键在于，它们未能将多样化和复杂的业务需求——如特定流派、语言或item新鲜度——直接纳入检索模型。这一局限表明缺乏真正的**指令跟随**能力，严重限制了它们在需要灵活、策略感知推荐的复杂工业场景中的适用性。

## 3 方法

### 3.1 框架概述

在本节中，我们提出了一个新颖的序列推荐框架 Climber-Pilot，它遵循两阶段预训练和微调范式。在预训练阶段，模型通过下次item预测目标学习捕捉用户的未来兴趣表征。为了缓解生成式架构中固有的短视问题，我们提出了时间感知多item预测（Time-Aware Multi-Item Prediction）方法。随后，在微调阶段，模型使用条件化下一item预测方法进行优化。通过引入条件信号（例如音乐流派），我们引导模型学习细粒度子类别中的兴趣表征，从而实现定向召回。整体训练流程如图 1 所示。

### 3.2 通用检索能力预训练

#### 3.2.1 序列编码

令 $S_n = \{i_1, i_2, ..., i_n\}$ 表示用户 $u$ 的历史交互序列，其中 $i_k \in \mathcal{I}$ 代表用户与之交互的第 $k$ 个item。为了得到每个item $i_k$ 的综合表征，我们将可学习的 ID 嵌入与其对应的类别特征嵌入拼接。我们对 ID 存储使用基于哈希的嵌入表。因此，原始交互序列 $S_n$ 被转换为密集嵌入序列 $E_n = \{e_1, e_2, ..., e_n\} \in \mathbb{R}^{n \times d}$ ，其中 $e_k \in \mathbb{R}^d$ 表示第 $k$ 个item的拼接嵌入。

嵌入序列 $E_n$ 被送入编码器。遵循 [36]，我们采用多层 Transformer 架构，结合了调整后的相对注意力偏置和因果掩码。这种设计防止了来自未来位置的信息泄露，并产生了一系列上下文相关的隐藏状态 $H_n = \{h_1, ..., h_n\} \in \mathbb{R}^{n \times d}$ 。

#### 3.2.2 时间感知多item预测（TAMIP）

标准的下一item预测范式存在固有短视问题 [21]。在 NLP 领域，缓解这种短视的一种常见策略是通过多token预测（Multi-Token Prediction, MTP）[10] 目标来扩展预测范围。然而，在工业环境中直接应用多item目标具有挑战性，因为批次服务（batch serving）与顺序日志记录之间存在错位。正如 [20] 所指出的，同一检索请求中的item没有请求内的因果依赖关系——因果关系仅存在于请求之间。然而，交互日志机械地将这些共同曝光的item序列化为密集序列。形式上，对于单个请求 $r$ ，有 $m$ 个item同时曝光给用户，但交互日志将它们记录为序列化的序列 $\{i_1, i_2, ..., i_m\}$ ，施加了一种不反映真实用户意图进展的人为排序。我们将此现象称为**消费滞后**（Consumption Lag）：在此类日志上训练的模型从本质上是并行的过程中学习到了虚假的序列模式。因此，模型过度拟合了人为的批次内模式。这些观察表明，标准的序列建模对于基于批次的检索是不够的。

为了从根本上缓解这一局限，我们引入了时间感知多item预测（TAMIP）。在结构上，TAMIP 由两个组成部分构成：一个用于扩展前瞻性的多分支预测主干，以及一个关键的时间感知掩码（Time-Aware Masking）机制，作为过滤虚假相关性的核心改进。

形式上，为了扩展模型的前瞻性，我们在共享用户表征 $h_n$ 之上构建 $K$ 个并行的投影分支。每个分支实现为一个具有独立（非共享）参数的单层 Transformer，将共享输入 $h_n$ 转换为步特定的表征 $h_n^{(k)}$ ，负责预测item $i_{n+k}$ 。在这种设置下， $K=1$ 的情况与标准 NIP 目标一致。

然而，朴素的多步预测容易受到上述消费滞后的影响。为了消除这种污染并迫使模型学习真正的长期依赖关系，我们引入了时间感知掩码机制（如图 1(a) 所示）。

为了与在线生产环境中使用的批次服务机制对齐，我们的模型在单次前向传播中预测目标序列 $\{i_{n+1}, ..., i_{n+K}\}$ 。因此，所有预测头的时间可见性锚定在第一个目标item的时间戳 $\tau_{n+1}$ 。因此，我们对主干编码器和所有 TAMIP 预测分支施加共享的时间约束。我们定义一个安全间隔 $\Delta\tau$ ，经验性地设置为已记录的服务时间间隔的平均值，在我们的生产系统中为 15 分钟。具体地，我们显式地掩码任何时间戳 $\tau_j$ 落在区间 $[\tau_{n+1} - \Delta\tau, \tau_{n+1}]$ 内的历史交互 $i_j$ 。形式上，给定带有时间戳 $\tau$ 的用户序列 $S$ ，时间感知注意力掩码 $M_{temp} \in \mathbb{R}^{n \times n}$ 定义为：

$$
$M_{temp}$(i,j) = \begin{cases}
-\infty & \text{if } \tau_j \in [\ta$u_{n+1}$ - \Delta\tau, \ta$u_{n+1}$] \\
0 & \text{otherwise}.
\end{cases}
\qquad (1)
$$

该掩码与标准因果掩码 $M_{causal}$ 一起被集成到自注意力机制中。令 $Q, K, V \in \mathbb{R}^{n \times d_k}$ 分别表示从隐藏状态线性投影得到的查询、键和值矩阵（ $d_k$ 是投影维度， $M_{causal}(i,j) = -\infty$ 如果 $j > i$ ，否则为 0）。最终的注意力计算公式为：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} + $M_{causal}$ + $M_{temp}$\right) V.
\qquad (2)
$$

注意，在 $M_{temp}$ 中使用 $\tau_{n+1}$ 并不会引入数据泄露：该时间戳仅用于掩码最近的交互，从而模拟推理时可用的信息范围。除了时间截止点之外，没有向模型暴露任何标签特定的特征。此外，时间感知掩码仅在训练期间使用。在服务时，由于消费滞后，最近的交互尚未被记录，因此同样的信息差距自然出现，无需使用时间戳。

通过严格禁止来自时间邻近token的信息流，TAMIP 防止了模型依赖短期行为突发或批次引入的伪影。相反，它迫使模型基于历史上下文推断未来意图，从而有效缓解固有短视并增强模型遵循长期用户兴趣轨迹的能力。

#### 3.2.3 TAMIP 损失公式

对于每个 TAMIP 分支 $k \in \{1, ..., K\}$ ，我们以增强表征 $h_n^{(k)} \in \mathbb{R}^d$ 为条件预测item $i_{n+k}$ 。由于在整个item空间上计算完整的 softmax 在计算上不可行，我们采用了跨所有预测头共享的采样 softmax 策略 [26]。第 $k$ 个预测头的损失定义为：

$$
\mathcal{L}_{\text{TAMIP}}^{(k)}($i_{n+k}$|S_n) = -\log \frac{\exp(\phi(h_n^{(k)}, $e_{n+k}$))}{\exp(\phi(h_n^{(k)}, $e_{n+k}$)) + \su$m_{j \in \mathcal{N}$_{n+k}} \exp(\phi(h_n^{(k)}, e_j))},
\qquad (3)
$$

其中 $S_n = \{i_1, ..., i_n\}$ 代表用户交互历史， $\phi(a, b) = a^\top b$ 表示内积评分函数， $e_{n+k} \in \mathbb{R}^d$ 是步骤 $n+k$ 处目标item的嵌入，而 $\mathcal{N}_{n+k}$ 表示共享的批次内负样本集合。

整体预训练目标 $\mathcal{L}_{PT}$ 汇总了所有 $K$ 个 TAMIP 分支的贡献：

$$
\mathcal{L}_{PT} = \su$m_{k=1}$^K \mathcal{L}_{\text{TAMIP}}^{(k)}.
\qquad (4)
$$

通过在大量用户行为语料上优化这一目标，我们进行了预训练阶段，使模型具备通用的检索能力。关键在于，所提出的 TAMIP 策略通过强制多步前瞻性，有效缓解了生成式架构中固有的短视问题。

### 3.3 通过 SFT 实现指令跟随

虽然基于 TAMIP 的预训练赋予了模型预测未来用户兴趣的能力，但缺乏显式的指令跟随能力阻碍了 Climber-Pilot 模型应对工业系统多样化和动态的需求。借鉴 LLM [11] 中已被进一步扩展到个性化推荐任务 [25, 33] 的 SFT 范式，我们将各种检索需求——如特定流派、语言或item新鲜度——重新构建为指令跟随任务。因此，我们将这些重新定义为条件检索需求作为不同的指令跟随任务。为了实现这一点，我们构建了条件特定的 SFT 数据集并提出了新颖的条件引导稀疏注意力机制。这个过程有效地将模型从无条件生成器转换为可控且指令感知的检索器。

通过以显式提示为条件对模型进行调节，我们实现了动态的、任务特定的候选生成。这种设计提供了一个显著的效率优势：一个统一的模型现在可以通过简单地改变提示来替代多个专门的检索流程。因此，这种整合极大地降低了部署复杂性和维护成本。

#### 3.3.1 条件特定检索数据集的构建

我们通过收集与特定检索条件一致的高质量在线用户交互日志来构建 SFT 数据集。这些条件涵盖了多种场景，例如从"新歌"候选集中检索item或定位具有特定属性（如摇滚标签）的item。形式上，我们将这些检索条件的集合记为 $\mathcal{C}$ 。对于 SFT 数据集中的每个目标item，其对应的条件记为 $c \in \mathcal{C}$ 。

首先，我们将任务形式化为一个指令跟随问题。在这种设置下，检索条件 $c$ 作为显式指令，引导模型基于用户序列 $\{i_1, ..., i_n\}$ 预测目标item $i_{n+1}$ 。注意，由于与每个条件关联的item池基数很大，该任务仍然非平凡，有效地缓解了数据泄露风险。

其次，我们在 SFT 中重新利用了 TAMIP 的多分支结构。在预训练阶段，TAMIP 使用单一上下文预测未来轨迹 $(i_{n+1}, ..., i_{n+K})$ 。然而在 SFT 阶段，我们改变了这一范式：我们使用多个截断的上下文窗口——具体来说是 $\{i_1, ..., i_{n-K+1}\}$ 到 $\{i_1, ..., i_n\}$ ——作为不同分支的输入。每个分支以其特定的上下文 $S_{n-k+1}$ （如图 1(c) 所示，对于 $k=1$ 到 $k=K$ 的分支，分别使用从 $S_{n-K+1}$ 到 $S_n$ 的上下文）以及目标指令 $c_{n+1}$ 为条件，将注意力引导到与指令相关的item $i_{n+1}$ 上。对于特定的分支 $k=K$ ，其上下文 $S_{n-K+1}$ 是最短的。

这种设计带来两个关键优势。首先，通过将单个训练样本扩展为 $K$ 个上下文目标对（全部以相同指令 $c_{n+1}$ 为条件），我们极大地提高了数据利用率和训练效率。其次，通过向模型暴露不同长度的截断上下文，迫使模型在不同长度的历史信息下遵循相同的指令。这提高了鲁棒性，因为服务时看到的历史长度也是变化的。同时，由于每个分支处理不同长度的上下文，模型保留了捕捉多范围依赖关系的能力——保持了预训练阶段建立的缓解短视的益处。

#### 3.3.2 条件引导稀疏注意力机制（CGSA）

用户行为本质上是多模态且带有噪声的。标准的多头注意力（MHA）全局地聚合信号，这通常会导致来自不相关兴趣的噪声积累，从而稀释了检索的相关性。为了解决这一问题，我们提出了**条件引导稀疏注意力**（Condition-Guided Sparse Attention, CGSA）机制。与标准 MHA 不同，CGSA 利用指令 $c$ 主动对上下文进行去噪。它引导模型仅关注与 $c$ 语义对齐的历史行为。

为了强制这种遵从，我们构建了一个稀疏掩码 $M_{sparse} \in \mathbb{R}^{n \times n}$ ，将注意力连接限制在相关item上。形式上，对于长度为 $n$ 的序列，位置 $(i, j)$ 处的注意力掩码定义为：

$$
$M_{sparse}$(i, j) = \begin{cases}
0, & \text{if } \mathcal{C}(\text{item}_j) = c \\
-\infty, & \text{otherwise},
\end{cases}
\qquad (5)
$$

其中 $\mathcal{C}(\cdot)$ 将item映射到其类别空间， $c$ 表示目标指令条件。

CGSA 机制通过将与目标指令不匹配的item上的注意力权重置零，显式地过滤长期用户历史，从而对行为序列进行去噪，使模型聚焦于条件相关的交互。

我们将 CGSA 仅应用于最终的 TAMIP 分支，同时在前 $L$ 层 Transformer 层中保留与预训练阶段相同的注意力掩码配置。这种选择主要出于计算效率的考虑：共享的 $L$ 层骨干网络可以一次计算并在多个条件特定分支间复用，从而实现了高效批量的多条件推理。在前 $L$ 层重用预训练注意力掩码的另一个好处是，它保留了预训练期间学到的丰富语义表征。

#### 3.3.3 SFT 优化

以预训练权重初始化允许模型保留通用检索能力，目标从无条件预测演变为指令条件检索。

在此阶段，所提出的模型被优化为在条件 $c_{n+1}$ 的约束下检索真实item $i_{n+1}$ 。关键在于，CGSA 机制在此阶段被激活，过滤噪声并引导 TAMIP 分支专注于用户交互序列中与指令相关的item。

为了缓解预训练中学到的通用偏好的灾难性遗忘，采用了降低的学习率。

形式上，损失函数定义为：

$$
\mathcal{L}_{SFT} = \su$m_{k=1}$^K \mathcal{L}_{\text{TAMIP}}^{(k)}($i_{n+1}$|$S_{n-k+1}$, $c_{n+1}$),
\qquad (6)
$$

其中 $S_{n-k+1}$ 表示第 $k$ 个 TAMIP 分支的截断历史窗口， $c_{n+1}$ 表示目标指令， $K$ 是 TAMIP 分支的数量。该目标强制模型将其预测与显式条件 $c_{n+1}$ 对齐。

### 3.4 部署细节

Climber-Pilot 的部署细节如图 2 所示。我们在一个由 8
$$
\times
$$
A100 GPU 组成的集群上使用 450 亿交互行为语料库训练模型，并通过 TensorRT [4] 框架部署推理服务。当用户发起请求时，系统同步获取行为序列和缓存的召回指令 $\{c_1, ..., c_P\}$ ，其中 $P$ 表示预计算指令的数量。

#### 3.4.1 预计算召回指令

为了实现高效的指令跟随推理，我们根据每个用户最近的交互历史预计算其 top- $P$ 兴趣类别。具体地，我们聚合用户在滑动窗口内的行为信号，并识别最频繁参与的类别（例如 Hip-Hop、新发行、英文歌曲）。这些类别标签随后被转换为召回指令 $\{c_1, ..., c_P\}$ 并缓存以供实时检索。这种设计将指令生成与在线服务路径解耦，确保模型能够动态适应用户偏好，而无需在推理时引入额外延迟。

#### 3.4.2 批量多条件推理

受 Climber [36] 启发，我们提出了批量多条件推理（Batched Multi-Condition Inference），以在严格的延迟约束下实现高吞吐量。由于 CGSA 机制仅影响最终的分支，我们可以将所有 $P$ 个条件合并到一次推理前向传播中。

计算分为两步进行。首先，初始的 $L$ 层共享 Transformer 块仅处理用户序列一次，并将得到的中间表征缓存。随后，缓存的表征在最后的所有 $P$ 个条件分支间复用。这种设计有效地避免了繁重骨干网络的冗余计算。

## 4 实验

### 4.1 实验设置

#### 4.1.1 数据集

我们在两个大规模数据集上进行实验：(1) **Amazon 基准数据集**，包括 Amazon-Sports、Amazon-Beauty 和 Amazon-Toys；(2) **工业数据集**（Industrial），一个来自网易云音乐平台真实用户交互日志的工业规模数据集。表 1 总结了两个数据集的主要统计信息。为了评估 Climber-Pilot 的指令跟随能力，我们进一步从在线日志中整理了一个专用的 SFT 数据集，该数据集与预训练语料无重叠。在所有关于可控生成的离线实验中，我们采用音乐流派和语言作为代表性的条件信号。

**表 1：处理后数据集的统计信息。"Avg. n"表示每个用户的平均序列长度。**

| 数据集 | #用户 | #item | Avg. n |
|--------|-------|-------|--------|
| Amazon-Sports | 18,357 | 35,598 | 8.32 |
| Amazon-Beauty | 22,363 | 12,101 | 8.87 |
| Amazon-Toys | 19,412 | 11,924 | 8.63 |
| Industrial | >40M | >6M | >100 |

#### 4.1.2 实现细节和指标

**模型架构。** 所提出的模型采用 3 层 Transformer [31] 骨干网络。我们对工业数据集设置嵌入维度 $d=128$ ，对 Amazon 数据集设置 $d=32$ 。由于额外的 TAMIP 分支在训练和推理过程中都会引入额外的计算开销，我们设置 $K=2$ 以平衡效率和验证 TAMIP 缓解短视有效性的核心目标。

**训练配置。** 所有模型使用 Adam 优化器 [18]，学习率为 $5 \times 10^{-4}$ ，权重衰减为 $1 \times 10^{-6}$ 。批次大小固定为 256。对于 SFT 阶段，我们采用降低的学习率 $1 \times 10^{-5}$ 以缓解预训练知识的灾难性遗忘。时间感知掩码间隔 $\Delta\tau$ 对于工业数据集设置为 15 分钟，对应于平均记录的请求间隔。

**评估指标。** 为了评估候选生成阶段的检索质量，我们采用 HitRate@K（HR@K）作为主要指标，其中 $K \in \{10, 20, 50\}$ 。具体地，我们将每个序列的最后一个item视为真实标签，并利用前面的交互历史生成用户兴趣表征。为了模拟大规模检索，我们使用 FAISS [7] 和 HNSW 算法 [22] 在整个item库上构建近似最近邻（ANN）索引，并检索 top- $K$ 候选。形式上，HR@K 定义为：

$$
HR@K = \frac{1}{|\mathcal{D}|} \su$m_{i=1}$^{|\mathcal{D}|} \mathbb{I}[\text{rank}(y_i) \leq K],
\qquad (7)
$$

其中 $\mathcal{D}$ 表示测试集， $y_i$ 表示第 $i$ 个实例的真实item， $\mathbb{I}[\cdot]$ 是指示函数。

### 4.2 预训练评估：通过 TAMIP 的通用检索

在工业数据集和三个公开 Amazon 数据集上，我们对 Climber-Pilot 与多个 SOTA 模型（包括序列推荐和生成式检索基线）进行了全面比较。为确保与标准基线的公平比较，我们仅使用 Climber-Pilot 的主要 TAMIP 分支（针对直接下一item，即 $i_{n+1}$ ）进行评估，排除了训练中使用的辅助长期 TAMIP 分支。

如表 2 所示，Climber-Pilot 在所有指标上 consistently 展现出优越的性能。我们将这一显著改进归因于所提出的 TAMIP 机制。传统 SOTA 模型依赖标准的 NIP 范式，由于只关注即时监督而常常受限于固有短视，而 Climber-Pilot 通过纳入多间隔监督来学习鲁棒的用户偏好，有效缓解了这一问题。

**表 2：预训练评估：在多个数据集上的整体性能比较。最优结果以粗体显示，次优结果以下划线标出。**

| 模型 | Industrial HR@50 | Industrial HR@20 | Industrial HR@10 | Amazon-Sports HR@50 | Amazon-Sports HR@20 | Amazon-Sports HR@10 | Amazon-Beauty HR@50 | Amazon-Beauty HR@20 | Amazon-Beauty HR@10 | Amazon-Toys HR@50 | Amazon-Toys HR@20 | Amazon-Toys HR@10 |
|--------|----------|----------|----------|-------------|-------------|-------------|-------------|-------------|-------------|----------|----------|----------|
| SASRec | 7.62% | 4.41% | 2.77% | 4.67% | 2.98% | 1.98% | 9.35% | 6.21% | 4.44% | 8.42% | 5.79% | 4.51% |
| TIGER | 8.34% | 5.15% | 3.82% | 7.14% | 4.40% | 3.05% | 11.43% | 8.39% | 6.10% | 10.33% | 7.11% | 4.97% |
| HSTU | 9.50% | 5.31% | 3.24% | 6.49% | 4.31% | 2.83% | 7.43% | 4.96% | 3.46% | 7.83% | 5.10% | 3.63% |
| PinRec | 8.78% | 5.00% | 3.07% | 6.63% | 4.30% | 2.98% | 12.08% | 8.53% | 6.32% | 8.52% | 5.15% | 3.66% |
| **Climber-Pilot** | **10.95%** | **6.40%** | **4.02%** | **7.20%** | **4.62%** | **3.28%** | **12.12%** | **8.95%** | **6.74%** | **11.20%** | **8.01%** | **6.13%** |

### 4.3 时间感知掩码的效果

#### 4.3.1 三方对比：NIP vs MIP vs TAMIP

我们通过将 TAMIP 与 NIP 和 MIP（无时间感知掩码的多item预测）在 Climber-Pilot 架构上进行对比，验证了 TAMIP 在缓解固有短视方面的有效性。如图 3 所示，我们使用时间步 $t$ 的检索结果来预测后续 10 步（ $t+1, ..., t+10$ ）的用户兴趣。

从结果中得出两个关键发现。首先，NIP 随着预测范围的延伸表现出明显的性能下降，确认了其贪婪的单步优化目标导致的固有短视问题。MIP 部分缓解了这一问题，在初始步骤的 HR@K 上略有提升且下降趋势较缓和，但仍受到明显的长期衰减影响。

其次，TAMIP 带来了双重优势：它显著提升了整体性能基线，同时保持了卓越的稳定性。我们将这一改进归因于时间感知掩码机制，它有效过滤了人为的批次内模式，迫使模型依赖真正的长期行为信号。

**图 3：TAMIP 缓解固有短视的有效性。我们在工业数据集上比较三种训练范式：NIP（红色）、MIP（橙色）和 TAMIP（蓝色）。TAMIP 曲线表现出卓越的长期稳定性。**
（注：原文包含图 3，展示 NIP/MIP/TAMIP 在 HR@50、HR@20、HR@10 上随预测步数的变化曲线）

#### 4.3.2 $\Delta\tau$ 的敏感性

上述比较显示了应用时间感知掩码的二值效果。如 3.2.2 节所述，生产设置基于平均记录的服务时间间隔使用 $\Delta\tau = 15$ 分钟。我们现在分析 TAMIP 对该值偏差的敏感性。由于 $\Delta\tau = 0$ 恢复为 MIP 变体，本研究也可视为对掩码强度的细粒度消融。我们在工业数据集上变化 $\Delta\tau \in \{0, 7.5, 15, 30\}$ 分钟。结果如表 3 所示。

**表 3：时间感知掩码窗口 $\Delta\tau$ 在工业数据集上的敏感性。**

| $\Delta\tau$ | HR@50 | HR@20 | HR@10 |
|------------|-------|-------|-------|
| 0 分钟     | 10.16% | 6.03% | 3.86% |
| 7.5 分钟   | 10.11% | 6.07% | 3.91% |
| 15 分钟    | 10.95% | 6.40% | 4.02% |
| 30 分钟    | 9.89% | 5.94% | 3.86% |

15 分钟设置在全部指标上最优。从 0 到 7.5 分钟，三个指标几乎持平。这表明窄于典型请求间间隔的窗口仍然使批次内item对模型可见，因此消费滞后导致的虚假序列模式未得到有效抑制。从 7.5 到 15 分钟，HR@50 从 10.11% 上升到 10.95%，相对提升约 8%，HR@20 和 HR@10 也呈现相同方向的变化。15 分钟设置与前面 3.2.2 节提到的系统统计数据匹配：每个请求曝光 3-4 首歌曲（每首约 4 分钟），15 分钟是批次消费与下一次请求之间的平均间隔，这也是消费滞后运作的时间尺度。从 15 到 30 分钟，性能回落到低于 MIP 变体（ $\Delta\tau = 0$ ）。远大于此间隔的窗口会掩码真正有信息量的较旧交互，从而减少可用于预测的信号并损害模型性能。因此， $\Delta\tau$ 的选择与系统级统计数据相关，而非自由超参数调优。

### 4.4 SFT 评估：通过 CGSA 的指令跟随

在本节中，我们在工业数据集上评估 Climber-Pilot 的指令跟随检索能力。我们将评估任务配置如下：给定用户交互序列 $S_n = \{i_1, ..., i_n\}$ ，我们利用真实下一item $i_{n+1}$ 的属性（例如流派或语言）作为显式指令信号 $c_{n+1}$ 。

我们的评估基于两个关键维度：(1) **推荐准确性**，通过 HR@K 衡量；(2) **指令遵从度**（Instruction Adherence），通过条件合规率（Condition Compliance Rate）量化。

条件合规率 CC@K 正式量化了模型将显式用户意图或业务约束转化为检索候选集的能力。令 $\mathcal{R}_K$ 表示通过上述 ANN 过程检索到的 top- $K$ item集合。我们将 CC@K 定义为检索到的item中满足指定条件 $c$ 的比例：

$$
CC@K = \frac{1}{K} \su$m_{j \in \mathcal{R}$_K} \mathbb{I}[\mathcal{C}(j) = c],
\qquad (8)
$$

其中 $\mathcal{C}(j)$ 表示与item $j$ 关联的类别。

为了验证所提出的 CGSA 模块的有效性，我们将我们的方法与三种条件生成范式进行比较：PinRec [1]、LUM [37]（两种最初为粗粒度动作类型设计的现有可控检索模型）以及 AdaLN-Zero [27]（一种来自扩散变换器 DiT 的条件机制）。为了公平比较，所有条件变体都从相同的预训练检查点进行微调。

表 4 总结了实验结果。我们观察到，所提出的 CGSA 在所有 HR@K 指标上均取得了最高性能，展示了卓越的推荐准确性。在指令遵从度方面，尽管 PinRec 表现出最高的 CC，但我们提出的方法保持了非常有竞争力的性能。虽然 PinRec 和 LUM 是为粗粒度动作类型条件设计的，但 CGSA 是针对推荐的序列建模特性量身定制的。通过从历史序列中过滤不相关item，CGSA 移除了与目标条件无关的行为噪声。这带来了两个好处：有竞争力的 CC 值表明 CGSA 满足了业务约束，而更高的 HR@K 则表明与用户偏好更好地对齐。在实践中，CGSA 不仅仅遵从指令——它同时优化了业务需求和用户满意度。

**表 4：SFT 评估：条件生成范式的性能比较。CGSA 方法在 HR@K 上达到最高，并在条件合规性方面表现稳健。**

| 模型 | HR@50 | HR@20 | HR@10 | CC@50 | CC@20 | CC@10 |
|--------|-------|-------|-------|-------|-------|-------|
| PinRec | 13.82% | 9.16% | 6.29% | 81.69% | 84.30% | 85.57% |
| LUM | 11.45% | 7.51% | 5.24% | 68.97% | 74.51% | 77.84% |
| AdaLN-Zero | 15.84% | 10.27% | 6.88% | 75.49% | 80.41% | 80.96% |
| **CGSA** | **18.53%** | **11.69%** | **7.73%** | 78.58% | 80.07% | 81.75% |

### 4.5 在线 A/B 测试

#### 4.5.1 整体性能对比

我们在网易云音乐上进行了为期两周的在线 A/B 测试，将 5% 的实时流量分配给每个模型变体，涵盖两个不同的生产场景：(1) **通用推荐**（General Recommendation），从整个item库中检索音乐推荐；(2) **特定流派推荐**（Genre-Specific Recommendation），满足用户寻找特定类别音乐的需求。我们以生产基线（基于 Transformer 的架构）以及包括 SASRec [17] 和 HSTU [40] 在内的 SOTA 模型作为基准，结果总结于表 5。

在无约束的通用场景中，Climber-Pilot（w/o SFT）已经优于强基线。这表明我们的预训练目标有效缓解了固有的短视预测偏差，从而捕捉长期用户兴趣以提升整体满意度。

在特定流派场景中，w/o SFT 变体以简单方式工作：模型检索候选时不带任何流派信号，系统随后仅保留那些流派与目标匹配的候选。这种方法仅带来边际改进。对相同 A/B 测试期间的流量日志分析显示，它平均仅保留约 40% 的匹配目标流派的检索候选，而 CGSA 将此比例提升至 77.9%。相比之下，通过将目标流派作为检索指令输入，完整的 Climber-Pilot 模型在 Like Rate（点赞率）上实现了 4.10% 的观测提升。这表明 SFT 有效地赋予了模型指令跟随能力，以满足特定的检索需求。

值得注意的是，SFT 的引入在通用场景中进一步放大了用户参与度，实现了 4.24% 的 Like Rate 提升——是未使用 SFT 时 2.03% 提升的两倍多。在此设置中，个性化指令按照 3.4.1 节的策略构建。这一实证证据表明，指令跟随不仅对特定流派场景有益；当与用户自适应的指令构建相结合时，SFT 有效增强了模型解读和匹配潜在用户意图的能力，从而带来更优的推荐质量。

关于推理效率，Climber-Pilot 实现了每个请求平均 7.26ms 的在线延迟，相比之下生产基线为 6.94ms——额外开销低于 5%，相对于显著的准确性提升可以忽略不计。

**表 5：生产环境中的在线 A/B 测试结果。改进以相对于生产基线的百分比提升报告。**

| 场景 | 模型 | Like Rate | 总收听时长 |
|--------|----------------------|-----------|----------------|
| 通用 | 基线 | – | – |
| 通用 | SASRec | −0.38% | +1.00% |
| 通用 | HSTU | +0.85% | +3.05% |
| 通用 | Climber-Pilot (w/o SFT) | +2.03% | +3.86% |
| 通用 | Climber-Pilot | **+4.24%** | +3.25% |
| 特定流派 | Climber-Pilot (w/o SFT) | +0.68% | −0.38% |
| 特定流派 | Climber-Pilot | **+4.10%** | +0.56% |

#### 4.5.2 参与度多样性分析

表 6 展示了 Climber-Pilot 带来的曝光多样性（流派和语言）的相对提升。用户活跃度级别根据内部业务标准分为三组（低、中、高）。一个显著的趋势是，多样性增益随着用户活跃度而增加。Climber-Pilot 显著增强了高活跃度用户的探索广度，多样性指标提升了超过 2%。这表明 Climber-Pilot 的指令跟随能力不仅提升了核心业务指标，还使系统能够更好地满足活跃用户的多模态兴趣。

**表 6：按用户活跃度级别划分的曝光多样性（流派和语言）的相对提升。**

| 用户活跃度级别 | 流派 | 语言 |
|----------------|-------|----------|
| 低 | +0.28% | +0.09% |
| 中 | +1.87% | +2.28% |
| 高 | +2.56% | +2.63% |

### 4.6 案例研究

为了评估 Climber-Pilot 的指令跟随能力，我们展示了一个对 Hip-Hop 和经典金曲（Classic Hits）具有混合兴趣的用户的案例研究，如图 4 所示。在无约束设置下，Climber-Pilot 检索的候选反映了两种流派。在注入流派指令后，模型生成的推荐严格限于目标流派。

关键在于，这种严格的遵从并不会损害个性化。指令引导的推荐与无约束设置的结果存在重叠，表明 Climber-Pilot 成功地找到了用户兴趣和指令约束的交集。

这些结果验证了我们的 CGSA 机制。通过过滤与指令无关的交互以确保准确性，同时选择性地关注上下文相关行为以保持个性化，Climber-Pilot 在单个统一模型中同时实现了准确性和个性化。

**图 4：展示 Climber-Pilot 指令跟随能力的案例研究。**
（注：原文包含图 4，展示用户序列、预计算召回指令（Hip-Hop、Classic Hits），以及无约束检索与指令引导检索的结果对比。指令引导下，Hip-Hop 指令输出 Travis Scott、Kendrick Lamar、Drake 等，Classic Hits 指令输出 Daniel Powter、Westlife、Robbie Williams 等。）

## 5 结论

在这项工作中，我们识别了固有短视和指令跟随效率低下作为生成式检索模型在工业推荐系统中单步推理和严格延迟约束下面临的根本挑战。为了解决这些问题，我们提出了 Climber-Pilot，一个统一的框架，在训练期间蒸馏长期多item用户意图，并将检索指令直接纳入生成过程。大量的离线实验和生产系统中的大规模在线 A/B 测试表明，Climber-Pilot 持续优于最先进的基线方法，在保持高效单步推理的同时为核心业务指标带来了一致的提升。这些结果突出了 Climber-Pilot 在真实世界、延迟敏感的推荐系统中部署生成式检索模型的实际有效性。

## 参考文献

[1] Prabhat Agarwal et al. 2025. Pinrec: Outcome-conditioned, multi-token generative retrieval for industry-scale recommendation systems. arXiv preprint arXiv:2504.10507 (2025).

[2] Yukuo Cen et al. 2020. Controllable multi-interest framework for recommendation. In KDD 2020. 2942–2951.

[3] Paul Covington, Jay Adams, and Emre Sargin. 2016. Deep neural networks for youtube recommendations. In RecSys 2016. 191–198.

[4] Pooya Davoodi et al. 2019. TensorRT Inference with TensorFlow. In GPU Technology Conference.

[5] Jiaxin Deng et al. 2025. OneRec: Unifying Retrieve and Rank with Generative Recommender and Iterative Preference Alignment. arXiv preprint arXiv:2502.18965 (2025).

[6] Tim Donkers, Benedikt Loepp, and Jürgen Ziegler. 2017. Sequential user-based recurrent neural network recommendations. In RecSys 2017. 152–160.

[7] Matthijs Douze et al. 2025. The faiss library. IEEE Transactions on Big Data (2025).

[8] Ningya Feng et al. 2024. Long-Sequence Recommendation Models Need Decoupled Embeddings. arXiv preprint arXiv:2410.02604 (2024).

[9] Shijie Geng et al. 2022. Recommendation as Language Processing (RLP): A Unified Pretrain, Personalized Prompt & Predict Paradigm (P5). In RecSys 2022. 299–315.

[10] Fabian Gloeckle et al. 2024. Better & faster large language models via multi-token prediction. arXiv preprint arXiv:2404.19737 (2024).

[11] Beliz Gunel et al. 2020. Supervised contrastive learning for pre-trained language model fine-tuning. arXiv preprint arXiv:2011.01403 (2020).

[12] Yupeng Hou et al. 2023. Learning vector-quantized item representation for transferable sequential recommenders. In WWW 2023. 1162–1171.

[13] Yupeng Hou et al. 2025. Generating long semantic ids in parallel for recommendation. In KDD 2025 V.2. 956–966.

[14] Hongtao Huang et al. 2025. Listwise Preference Diffusion Optimization for User Behavior Trajectories Prediction. arXiv preprint arXiv:2511.00530 (2025).

[15] Eugene Ie et al. 2019. SlateQ: A Tractable Decomposition for Reinforcement Learning with Recommendation Sets. In IJCAI, Vol. 19. 2592–2599.

[16] Clark Mingxuan Ju et al. 2025. Generative Recommendation with Semantic IDs: A Practitioner's Handbook. In CIKM 2025. 6420–6425.

[17] Wang-Cheng Kang and Julian McAuley. 2018. Self-attentive sequential recommendation. In ICDM 2018. IEEE, 197–206.

[18] Diederik P Kingma and Jimmy Ba. 2014. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980 (2014).

[19] Chao Li et al. 2019. Multi-interest network with dynamic routing for recommendation at Tmall. In CIKM 2019. 2615–2623.

[20] Zida Liang et al. 2025. TBGRecall: A Generative Retrieval Model for E-commerce Recommendation Scenarios. In CIKM 2025. 5863–5870.

[21] Xiaohao Liu et al. 2025. L-MTP: Leap Multi-Token Prediction Beyond Adjacent Context for Large Language Models. arXiv preprint arXiv:2505.17505 (2025).

[22] Yu A Malkov and Dmitry A Yashunin. 2018. Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs. IEEE TPAMI 42, 4 (2018), 824–836.

[23] Pingjun Pan et al. 2024. A User-State Based Interest Transfer Network for Cross-Domain Recommendation. In WWW 2024 Companion. 662–665.

[24] Pingjun Pan et al. 2026. Hi-SAM: A Hierarchical Structure-Aware Multi-modal Framework for Large-Scale Recommendation. arXiv preprint arXiv:2602.11799 (2026). Accepted at ACM KDD 2026 ADS.

[25] Pingjun Pan et al. 2026. L2Rec: Towards Dual-View Understanding of LLMs for Personalized Recommendation. In SIGIR 2026. To appear.

[26] Nikil Pancha et al. 2022. PinnerFormer: Sequence modeling for user representation at pinterest. In KDD 2022. 3702–3712.

[27] William Peebles and Saining Xie. 2023. Scalable diffusion models with transformers. In ICCV 2023. 4195–4205.

[28] Shashank Rajput et al. 2023. Recommender systems with generative retrieval. Advances in NeurIPS 36 (2023), 10299–10315.

[29] Yelong Shen et al. 2014. Learning semantic representations using convolutional neural networks for web search. In WWW 2014. 373–374.

[30] Fei Sun et al. 2019. BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer. In CIKM 2019. 1441–1450.

[31] Ashish Vaswani et al. 2017. Attention is all you need. Advances in NeurIPS 30 (2017).

[32] Shijia Wang et al. 2025. Progressive Semantic Residual Quantization for Multimodal-Joint Interest Modeling in Music Recommendation. In CIKM 2025. 6119–6127.

[33] Shijia Wang et al. 2025. Enhanced Emotion-aware Music Recommendation via Large Language Models. In KDD 2025. 4986–4994.

[34] Shijia Wang et al. 2024. Sparsity-Aware Personalized Pattern Extractor Network for Music Multi-task Learning. In DASFAA (LNCS). Springer, 352–363.

[35] Liwen Xiao et al. 2026. Beyond Residuals: A Progressive Semantic-Preserving Quantization Approach for Recommendation. In DASFAA (LNCS, Vol. 16540). Springer, 593–605.

[36] Songpei Xu et al. 2025. Climber: Towards efficient scaling laws for large recommendation models. In CIKM 2025. 6193–6200.

[37] Bencheng Yan et al. 2025. ... (incomplete)

[38] Qimeng Yang et al. 2024. Cascading Multimodal Feature Enhanced Contrast Learning for Music Recommendation. In ICDM 2024. IEEE, 905–910.

[39] Xinyang Yi et al. 2019. Sampling-bias-corrected neural modeling for large corpus item recommendations. In RecSys 2019. 269–277.

[40] Jiaqi Zhai et al. 2024. Actions speak louder than words: trillion-parameter sequential transducers for generative recommendations. In ICML 2024. 58484–58509.
