# TWIN: 用于快手 CTR 预测中终身用户行为建模的两阶段兴趣网络

> Jianxin Chang, Chenbin Zhang, Zhiyi Fu, Xiaoxue Zang, Lin Guan, Jing Lu, Yiqun Hui, Dewei Leng, Yanan Niu, Yang Song, Kun Gai | Kuaishou



本文提出两阶段兴趣网络（TWIN），**通过一致性保持GSU（CP-GSU）与精确搜索单元（ESU）共享完全相同的目标-行为相关性度量，解决了传统两阶段终身行为建模中GSU与ESU的不一致性问题**。

核心内容：

- 传统两阶段终身行为建模中，GSU使用的粗略相关性度量与ESU中的目标注意力（TA）不一致，导致遗漏高相关行为、检索出不相关行为
- 提出TWIN：CP-GSU采用与ESU完全相同的MHTA结构和参数，使两个阶段成为"双胞胎"
- 通过行为特征拆分（固有特征预计算缓存 + 交叉特征压缩为偏置项）突破TA计算瓶颈，将适用序列长度从$10^2$扩展到$10^4-10^5$
- 在快手460亿规模工业数据集上的离线和在线实验验证有效性

关键发现：

- TWIN在AUC上比最佳对比模型（SIM Soft）提升**+0.29%**，GAUC提升**+0.51%**
- 在线A/B测试中，相比SIM Soft观看时长提升**+1.4%~+2.8%**，相比SIM Hard提升**+3.7%~+6.2%**
- 预计算缓存策略将计算瓶颈降低**99.3%**，成功部署服务3.46亿日活用户
- GSU-ESU一致性从SIM Hard的40%命中率提升至TWIN的**94%命中率**

---



## 摘要

终身用户行为建模，即从用户数月甚至数年的丰富历史行为中提取隐藏兴趣，在现代 CTR 预测系统中发挥着核心作用。传统算法大多遵循两个级联阶段：一个简单的通用搜索单元（GSU），用于对数万个长期行为进行快速粗略搜索；一个精确搜索单元（ESU），用于对 GSU 筛选出的少量候选进行有效的目标注意力（TA）计算。尽管高效，现有算法大多存在一个关键缺陷：**GSU 和 ESU 之间目标-行为相关性度量不一致**。这导致 GSU 通常会遗漏高相关行为，却检索出 ESU 认为不相关的行为。在这种情况下，无论注意力如何分配，ESU 中的 TA 都会偏离用户的真实兴趣，从而降低整体 CTR 预测精度。为了解决这种不一致性，我们提出了**两阶段兴趣网络（TWIN）**，其中我们的**一致性保持 GSU（CP-GSU）** 采用与 ESU 中 TA **完全相同的**目标-行为相关性度量，使两个阶段如同"双胞胎"。具体而言，为了突破 TA 的计算瓶颈并将其从 ESU 扩展到 GSU（即从行为长度 $10^2$ 扩展到 $10^4-10^5$ ），我们通过行为特征拆分构建了一种新颖的注意力机制。对于行为的视频固有特征，我们通过高效的预计算和缓存策略来计算其线性投影。对于用户-item交叉特征，我们将其压缩为注意力分数计算中的一维偏置项，以节省计算成本。两个阶段之间的一致性，加上 CP-GSU 中有效的基于 TA 的相关性度量，带来了 CTR 预测的显著性能提升。在快手的 460 亿规模真实生产数据集上的离线实验和在线 A/B 测试表明，TWIN 优于所有对比的 SOTA 算法。通过优化在线基础设施，我们将计算瓶颈降低了 **99.3%**，这促成了 TWIN 在快手的成功部署，每天服务于数亿活跃用户的主要流量。

**CCS 概念**：信息系统 $\rightarrow$ 学习排序；推荐系统；计算方法 $\rightarrow$ 神经网络。

**关键词**：click-through rate prediction；user interest modeling；lifelong user behavior；recommender systems

**ACM 引用格式**：
Jianxin Chang, Chenbin Zhang, Zhiyi Fu, Xiaoxue Zang, Lin Guan, Jing Lu, Yiqun Hui, Dewei Leng, Yanan Niu, Yang Song, and Kun Gai. 2023. TWIN: Two-stage Interest Network for Lifelong User Behavior Modeling in CTR Prediction at Kuaishou. In *Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '23), August 6–10, 2023, Long Beach, CA, USA*. ACM, New York, NY, USA, 10 pages. https://doi.org/10.1145/3580305.3599922

---

## 1 引言

作为中国最受欢迎的短视频分享应用之一，快手高度依赖其强大的推荐系统（RS）。每天，RS 帮助数亿活跃用户过滤掉数以百万计不感兴趣的视频，找到他们感兴趣的内容，产生数百亿的点击日志。这些海量数据不仅为 RS 的训练提供养分，也推动了技术的革新，不断提升平台上的用户体验和商业效益。

在现代 RS 中，一项基本任务是**点击率（CTR）预测**，其目标是预测用户点击某个item/视频的概率 [2, 10, 32]。准确的 CTR 预测能够指导 RS 为每个用户提供其最喜欢的内容，并将每个视频推送给感兴趣的受众。为此，CTR 模型需要高度个性化，并充分利用稀缺的用户信息。因此，**终身用户行为建模**，即从丰富的长期历史行为中提取用户的隐藏兴趣，通常作为 CTR 模型的关键组成部分 [7, 16, 34–36]。

工业界终身行为建模算法大多遵循**两个级联阶段** [19]：（1）**通用搜索单元（GSU）**，对数万个长期行为进行快速粗略搜索，输出少量与目标最相关的行为；（2）**精确搜索单元（ESU）**，对 GSU 筛选出的少量候选执行有效的**目标注意力（TA）**。这种两阶段设计的原因有两方面。一方面，为了精确捕捉用户兴趣，TA 是强调目标相关行为、抑制目标无关行为的合适选择。另一方面，TA 昂贵的计算成本将其适用序列长度限制在最多几百个。因此，一个简单快速的 GSU 作为预过滤器对于截断工业规模的行为序列至关重要，这些序列在短短几个月内就可能达到 $10^4-10^5$ 。

近年来，涌现了大量关于两阶段终身行为建模的研究，其主要区别在于 GSU 的策略，即粗略选择与目标相关的行为。例如，SIM Hard [19] 简单地选择与目标item**同类别**的行为；SIM Soft [19] 通过预训练item嵌入的内积计算目标-行为相关性分数，并选择相关性最高的行为 [19]。ETA [3] 使用局部敏感哈希（LSH）和汉明距离 [3] 来近似相关性分数计算。SDIM [1] 通过多轮哈希碰撞采样与目标item具有相同哈希签名的行为 [1]，等等。尽管已被广泛研究，现有的两阶段终身行为建模算法仍然存在一个关键缺陷：**GSU 和 ESU 之间的不一致性**（如图 1 所示）。具体来说，GSU 中使用的目标-行为相关性度量既粗糙又与 ESU 中使用的 TA 不一致。结果是，GSU 很可能遗漏相关行为，却检索出 ESU 认为不相关的行为，浪费了 ESU 宝贵的计算资源。在这种情况下，ESU 中的 TA，无论注意力如何分配，大多会偏离用户的真实兴趣，从而降低整体 CTR 预测精度。

**图 1：传统两阶段算法中 GSU 和 ESU 之间的不一致性。** 假设一个"神谕"（Oracle，蓝色）能够负担得起对全部 $10^4-10^5$ 个行为使用与 ESU 中相同的相关性度量，即找出"真正的 top-100"。而 GSU（橙色）使用无效且不一致的粗略搜索。在 GSU 返回的 top-100 中（x 轴），只有 40 个命中了真正的 top-100（y 轴）。这种不一致性（灰色区域）表明了 TWIN 可以提升的潜在改进空间。

为了解决这种不一致性，我们提出了 **TWIN：用于终身用户行为建模的两阶段兴趣网络**，其中一致性保持 GSU（CP-GSU）采用与 ESU 中 TA 完全相同的目标-行为相关性度量，使两个阶段成为"双胞胎"。为了将昂贵的 TA 扩展到 CP-GSU，TWIN 通过有效的**行为特征拆分**、**简化的 TA 架构**和**高度优化的在线基础设施**，突破了 TA 的关键计算瓶颈，即所有行为的线性投影。具体来说：1）对于行为的视频固有特征（如视频 ID、作者、时长、话题），这些特征在不同用户/行为序列之间共享，我们通过高效的**预计算和缓存策略**加速其投影；2）对于行为的用户-视频交叉特征（如用户的点击时间戳、播放时长、评分），由于缓存不适用，我们通过将其投影压缩为**偏置项**来简化 TA 架构。通过优化的在线基础设施，我们成功地将 TA 的适用序列长度从 ESU 中的 $10^2$ 扩展到 CP-GSU 中的 $10^4-10^5$ 。两个阶段之间的一致性，加上 CP-GSU 中有效的基于 TA 的相关性度量，带来了 CTR 预测的显著性能提升。

总的来说，我们做出了以下贡献：
- 在我们提出的 TWIN 中，CP-GSU 精确且一致地检索出不仅与目标相关，而且被 ESU 认为重要的行为，最大化行为建模的检索效果。据我们所知，我们首次成功解决了两阶段终身行为建模问题中的不一致性。
- 我们通过在快手 460 亿规模工业数据集上的大量离线实验和在线 A/B 测试验证了 TWIN 的有效性。我们通过消融研究验证了我们的结论，并展示了 TWIN 带来的显著线上收益。
- 我们构建了高效的工业基础设施，将 TWIN 应用于真实的在线推荐系统。提出了有效的预计算和缓存策略，将 TWIN 的计算瓶颈（即 CP-GSU 中行为的线性投影）降低了 99.3%，满足了在线服务系统的低延迟要求。

TWIN 现已部署在快手的推荐系统上，每天服务 3.46 亿活跃用户的主要流量。

---

## 2 相关工作

我们的工作与两个活跃研究领域密切相关：CTR 预测和长期用户行为建模。

### 2.1 点击率预测

CTR 预测旨在预测用户的个性化兴趣，对当今的推荐系统至关重要。早期的 CTR 模型是浅层的，主要关注利用特征交互，如因子分解机（FM）[22] 和场感知因子分解机（FFM）[12]。随着深度学习的成功，深度 CTR 模型被广泛研究并成为主流选择。例如，Chen 等人 [2] 和 Zhang 等人 [33] 首先将深度模型应用于 CTR 任务。Wide & Deep [5] 结合了宽线性模型和深度模型，兼顾了特征交互的记忆和深度架构的泛化能力。DeepFM [10] 和 DCN [26, 27] 改进了 Wide & Deep 的宽部分以增强特征交互能力。xDeepFM [15] 和 AFM [29] 进一步利用卷积层和注意力机制来改进深度部分并提升模型性能。

随着 CTR 模型日益个性化，用户行为建模（即通过总结用户历史行为来捕捉隐藏兴趣）成为一个关键模块。受计算资源限制，早期算法大多以目标无关的方式运行，可以高效地进行离线预计算 [8, 23, 31]。为了更好地提取用户对特定item的兴趣，各种 TA 机制被采用。DIN [36] 通过对历史行为执行 TA 来表征用户兴趣，强调与目标相关的行为。DIEN [35] 进一步引入 ARGRU（一种基于注意力的经典 GRU [6] 变体）来建模行为的时间关系。DSIN [9] 将行为按会话拆分，并在每个会话内部进行自注意力以强调会话内关系。MIND [14] 和 DMIN [30] 使用多个向量表示用户兴趣。BST [4]、SASRec [13] 和 BERT4Rec [24] 也使用 Transformer 来提升模型性能和并行性。

### 2.2 长期用户行为建模

随着 TA 和兴趣建模在现代工业推荐系统中的有效性得到确认，研究者们开始建模越来越长的行为序列。Liu 和 Zamanian [16] 结合了 CTR 预测中的长期和短期兴趣。MIMN [18] 在用户兴趣中心（UIC）将用户行为存储为记忆矩阵，并在新用户行为到来时更新记忆。然而，MIMN 难以扩展到超过 $10^3$ 的序列，并且它对不同的候选item生成相同的记忆矩阵，携带了无用的噪声并削弱了 TA 的效果。

最近，SIM [19] 和 UBR4CTR [20, 21] 引入了两阶段级联框架来解决这些挑战，并在 CTR 预测中取得了 SOTA 性能。传统的两阶段算法通常包括：1）一个简单快速的 GSU，从数千个用户行为中检索出与目标item最"相关"的item；2）一个带注意力的 ESU，对 GSU 的候选执行 TA。UBR4CTR 在其第一阶段使用 BM25 作为相关性度量。在原始的 SIM 中，有两个具有不同 GSU 设计的实例。SIM Hard 的 GSU 从与目标item相同类别的item中选择相关item，而 SIM Soft 的 GSU 使用预训练item嵌入的内积作为相关性度量。尽管两阶段设计向前迈进了一大步，但原始的 GSU 仍然面临高计算负担，并且与 ESU 具有不同的检索度量，这导致了两个阶段之间的不一致性。

最近，ETA [3] 使用局部敏感哈希（LSH）对 ESU 训练的item嵌入进行编码，并通过汉明距离（HD）从长期行为中检索相关item。SDIM [1] 通过多轮哈希碰撞，采样与目标item具有相同哈希签名的行为item，然后 ESU 对这些采样的行为item进行线性聚合以获取用户兴趣。ETA 和 SDIM 采用**端到端训练**是积极的，即它们的两个阶段共享相同的嵌入。然而，检索策略（特别是网络结构和参数）仍然存在不一致性。

在本文中，我们提出将 TA 结构扩展到 GSU，并将嵌入和注意力参数从 ESU 同步到 GSU，保持端到端训练。结果，我们在网络结构和模型参数上都实现了一致性，相比于 ETA 和 SDIM 带来了显著的性能提升。我们在表 1 中详细列出了我们的模型与其他模型的差异。请注意，我们的工作不同于那些通过将行为映射到码本并查找距离来加速 Transformer 的索引算法（如 LISA [28]）。我们的工作以及许多其他两阶段算法，使用精确的距离计算，但通过 GSU 作为预过滤器来减少行为数量。

**表 1：SOTA 用户兴趣模型对比。** 底部列出了两阶段模型。Length 表示原始论文中用户行为序列的最大长度。

| 方法 | 长度 | GSU 策略 | 端到端 | 一致性 |
|------|------|----------|--------|--------|
| DIN [36] | ~ $10^3$ | N/A | N/A | N/A |
| DIEN [35] | ~ $10^2$ | N/A | N/A | N/A |
| MIMN [18] | ~ $10^3$ | N/A | N/A | N/A |
| UBR4CTR [20, 21] | ~ $10^2$ | BM25 | ✗ | ✗ |
| SIM Hard [19] | ~ $10^3$ | 类别过滤 | ✗ | ✗ |
| SIM Soft [19] | ~ $10^3$ | 内积 | ✗ | ✗ |
| ETA [3] | ~ $10^3$ | LSH & 汉明距离 | ✓ | ✗ |
| SDIM [1] | ~ $10^3$ | 哈希碰撞 | ✓ | ✗ |
| **TWIN（本文）** | **~ $10^5$ ** | **目标注意力** | **✓** | **✓** |

---

## 3 快手 CTR 预测中的 TWIN

首先，我们在第 3.1 节回顾 CTR 预测问题的一般预备知识。然后我们在第 3.2 节描述我们在快手的 CTR 预测系统的模型架构。我们在第 3.3 节进一步深入探讨我们提出的**一致性保持终身用户行为建模模块**，即**两阶段兴趣网络（TWIN）** 的细节。最后，我们在第 3.4 节介绍确保 TWIN 在快手主要流量上成功在线部署的关键加速策略。使用的符号总结在表 2 中。

**表 2：第 3 节中使用的重要符号**

| 符号 | 含义 | 符号 | 含义 |
|------|------|------|------|
| $f$ | 预测器 | $\sigma$ | Sigmoid 函数 |
| $\hat{y}$ | 预测 CTR | $D$ | 数据集 |
| $\ell$ | 损失 | $\mathbf{x}$ | 特征向量 |
| $y$ | 真实标签 | $\mathbf{R}$ | 实数集 |
| $E$ | 嵌入字典 | $d$ | 特征维度 |
| $\mathbf{x}_{\text{emb}}$ | 嵌入特征 | $\mathbf{x}_{\text{hot}}$ | One-hot/Multi-hot 编码 |
| $v$ | 词汇表大小 | $K$ | 行为特征 |
| $L$ | 行为长度 | $H$ | 固有特征维度 |
| $C$ | 交叉特征维度 | $K_h$ | 固有特征 |
| $K_c$ | 交叉特征 | $J$ | 交叉特征数量 |
| $a$ | 注意力头索引 | $d_k$ | 投影后固有特征的维度 |
| $d_{\text{out}}$ | 原始 MHTA 中投影维度 | $\boldsymbol{\beta}$ | 交叉特征权重 |
| $\boldsymbol{\alpha}$ | 注意力权重 | $W_h, W_c, W_v, W^o$ | 线性投影参数 |
| $\mathbf{w}_j^c$ | 投影参数 $W_c$ 的对角块 | | |

### 3.1 预备知识

CTR 预测的目标是预测用户在特定上下文中**点击**某个item的概率。准确的 CTR 预测不仅能通过提供偏好内容提升用户体验，还能通过触达感兴趣的受众来惠及内容生产者和平台的商业效益。因此，CTR 预测已成为各种工业推荐系统的核心组件，尤其是像快手这样的短视频推荐平台。

CTR 预测通常被形式化为一个二分类问题，目标是给定训练数据集 $D = \{(\mathbf{x}_1, y_1), \ldots, (\mathbf{x}_{|D|}, y_{|D|})\}$ ，学习一个预测函数 $f: \mathbb{R}^d \rightarrow \mathbb{R}$ 。具体来说， $\mathbf{x}_i \in \mathbb{R}^d$ 是第 $i$ 个训练样本的特征向量（即用户、item和上下文特征的拼接）， $y_i \in \{0,1\}$ 是真实标签，表示用户是否点击（1）该item。预测的 CTR 计算如下：
$$
\hat{y}_i = \sigma(f(\mathbf{x}_i)). \qquad (1)
$$
 $\sigma(\cdot)$ 是 Sigmoid 函数，将 $f$ 的输出缩放到 (0,1)。模型通过最小化负对数似然来训练：
$$
\ell(D) = -\frac{1}{|D|} \sum_{i=1}^{|D|} \left[ y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i) \right]. \qquad (2)
$$
为简洁起见，在以下各节中，当不会引起混淆时，我们将省略训练样本索引 $i$ 。

### 3.2 CTR 预测系统的架构

我们现在阐述我们在快手的 CTR 预测系统的架构。详情如图 2 所示。

**图 2：快手 CTR 预测系统中的 TWIN。** 与传统的两阶段行为建模算法不同，TWIN 在 CP-GSU 和 ESU 中采用**完全相同的**目标-行为相关性度量，不仅包括**相同的网络架构**（如左侧所示），还包括**相同的参数值**（如中下部所示）。这是具有挑战性的，因为 MHTA 设计上计算成本高，因此仅适用于 ESU（100 个行为），而不适用于 CP-GSU（ $10^4$ 个行为）。我们通过提出以下方案解决这一挑战：1）高效的特征拆分和投影策略，以不同方式处理item固有特征和用户-item交叉特征（如右下方所示）；2）简化的目标注意力架构，通过将交叉特征压缩为偏置项来加速目标注意力的效率（如左下方所示）。

#### 3.2.1 嵌入层

在底层，我们的模型从特征嵌入层开始，该层将训练样本的原始特征转换为嵌入向量。不失一般性，我们假设所有特征在经过必要的预处理后都处于类别形式。对于词汇表大小为 $v_A$ 的特征 $A$ ，我们首先将类别信息编码为 one-hot/multi-hot 编码 $\mathbf{x}_{A,\text{hot}} \in \{0,1\}^{v_A}$ 。例如：
- 星期几=周一 $\Rightarrow \mathbf{x}_{\text{WeekDay,hot}} = [1,0,0,0,0,0,0]^\top$ ，
- 话题={搞笑, 宠物} $\Rightarrow \mathbf{x}_{\text{Topic,hot}} = [\ldots,0,1,0,\ldots,0,1,0,\ldots]^\top$ 。

需要注意的是，在大多数工业系统中，词汇表大小（尤其是用户/作者/视频 ID）可以轻松达到数亿。因此，常见策略是将极高维的 one-hot 编码转换为低维嵌入：
$$
\mathbf{x}_{A,\text{emb}} = E_A \mathbf{x}_{A,\text{hot}}, \qquad (3)
$$
其中 $E_A \in \mathbb{R}^{d_A \times v_A}$ 是 $A$ 的嵌入字典， $d_A$ 是嵌入维度。在我们的系统中，对于具有大词汇表的 ID 特征，我们将嵌入维度设置为 64；对于其他特征（如视频话题、视频播放时间戳），设置为 8。

在所有上层中，我们将嵌入向量作为输入，为简洁起见省略下标"emb"。

#### 3.2.2 深度网络

我们 CTR 预测的整体架构如图 2 所示。上层模块由堆叠的神经网络和 ReLU 组成，作为一个混合器，学习三个中间模块输出之间的交互：

- **TWIN**，我们提出的**一致性保持终身用户行为建模模块**，通过两个级联的行为建模子模块提取用户兴趣：1）**一致性保持通用搜索单元（CP-GSU）**，从数万个长期历史行为中粗略搜索最相关的 100 个行为；2）**精确搜索单元（ESU）**，对 CP-GSU 的 100 个候选采用注意力机制以捕捉精确的用户兴趣。与传统算法通常包含一个"轻量"的 GSU 和一个"重量"的 ESU 不同，我们提出的 CP-GSU 遵循与 ESU 完全相同的相关性评估度量，使两个级联阶段成为"双胞胎"。因此，CP-GSU 一致地检索出 ESU 认为重要的item，最大化行为建模的有效性。
- **短期行为建模**，从最近的 50 个行为中提取用户兴趣。该模块关注用户最近几天的短期兴趣，作为 TWIN 的有力补充。
- **其他任务建模**。除了行为建模，我们还拼接了各种其他任务建模的输出，这些模块对用户的性别、年龄、职业、位置、视频的时长、话题、热度、质量以及上下文特征（如播放日期、时间戳、页面位置等）进行建模。

### 3.3 TWIN：两阶段兴趣网络

我们将提出的算法命名为 TWIN，以强调 CP-GSU 遵循与 ESU 完全相同的相关性评估度量。需要注意的是，这种一致性是**非平凡**的，因为：
- 有效的行为建模算法通常基于**多头目标注意力（MHTA）**[25]，它通过强调目标相关行为来精确捕捉用户兴趣。不幸的是，由于高计算复杂度，MHTA 的适用行为序列长度大多限制在几百个。
- 为了全面捕捉用户的长期兴趣，CP-GSU 需要覆盖过去几个月内的用户行为，这可以轻易达到数万个。考虑到在线系统的严格低延迟要求，这个序列长度远远超出了传统 MHTA 的能力。

本节旨在回答这个关键问题：**如何提高 MHTA 的效率，以便我们将其从 ESU 扩展到 CP-GSU，即从几百的序列长度扩展到至少数万的序列长度？**

#### 3.3.1 行为特征拆分与线性投影

遵循 MHTA [25] 的标准符号，我们将长度为 $L$ 的行为序列 $[s_1, s_2, \ldots, s_L]$ 的特征定义为矩阵 $K$ ，其中每一行表示一个行为的特征。在实践中，MHTA 注意力分数计算中 $K$ 的线性投影是关键的计算瓶颈，阻碍了 MHTA 在超长用户行为序列上的应用。因此，我们提出以下方法来降低其复杂度。

我们首先将行为特征矩阵 $K$ 拆分为两部分：
$$
K \triangleq [K_h \ K_c] \in \mathbb{R}^{L \times (H + C)}, \qquad (4)
$$
我们将 $K_h \in \mathbb{R}^{L \times H}$ 定义为行为item的**固有特征**（如视频 ID、作者、话题、时长），这些特征与特定用户/行为序列**无关**；将 $K_c \in \mathbb{R}^{L \times C}$ 定义为**用户-item交叉特征**（如用户点击时间戳、用户播放时长、点击页面位置、用户-视频交互）。这种拆分允许对以下线性投影 $K_h W_h$ 和 $K_c W_c$ 进行高效计算。

对于**固有特征 $K_h$ **，虽然维度 $H$ 较大（每个 ID 特征 64 维），但线性投影实际上并不昂贵。特定item的固有特征在不同用户/行为序列之间是共享的。通过必要的缓存策略， $K_h W_h$ 可以通过查找和收集过程高效"计算"。在线部署的细节将在第 3.4 节中介绍。

对于**用户-item交叉特征 $K_c$ **，缓存策略不适用，因为：1）交叉特征描述用户和视频之间的交互细节，因此不在不同用户的行为序列之间共享；2）每个用户最多观看一个视频一次，即在投影交叉特征时不存在重复计算。因此，我们通过简化线性投影权重来降低计算成本。

给定 $J$ 个交叉特征，每个特征的嵌入维度为 8（因为不是具有巨大词汇表的 ID 特征）。我们有 $C = 8J$ 。我们简化线性投影如下：
$$
K_c W_c \triangleq [K_{c,1} \mathbf{w}_1^c, \ldots, K_{c,J} \mathbf{w}_J^c], \qquad (5)
$$
其中 $K_{c,j} \in \mathbb{R}^{L \times 8}$ 是 $K_c$ 对应第 $j$ 个交叉特征的列切片， $\mathbf{w}_j^c \in \mathbb{R}^{8}$ 是其线性投影权重。使用这种简化的投影，我们将每个交叉特征压缩到一维，即 $K_c W_c \in \mathbb{R}^{L \times J}$ 。注意，这种简化投影等价于将 $W_c$ 限制为对角块矩阵。

#### 3.3.2 复杂度分析

在传统的 MHTA 中， $K$ 的线性投影（从维度 $L \times (H+C)$ 到 $L \times d_{\text{out}}$ ）的时间复杂度为 $O(L \times (H+C) \times d_{\text{out}})$ 。

而在 TWIN 的 MHTA 中，item固有特征 $K_h W_h$ 是**预计算**并通过高效收集在 $O(L)$ 内完成的，与维度 $H$ 无关。用户-item交叉特征 $K_c W_c$ 的计算减少为低维计算 $O(L \times C)$ 。

由于 $C \ll H$ 且 $C \ll d_{\text{out}}$ ，正是这种理论上的加速使得 MHTA 能够在 CP-GSU 和 ESU 中一致地实现。

#### 3.3.3 TWIN 中的目标注意力

基于行为特征 $K_h W_h$ 和 $K_c W_c$ 的线性投影，我们现在定义在 CP-GSU 和 ESU 中统一使用的目标-行为相关性度量。不失一般性，我们假设用户与目标item之间尚未发生交互，并将目标item的固有特征记为 $\mathbf{q} \in \mathbb{R}^H$ 。通过适当的线性投影 $W_q$ ，目标item与历史行为之间的相关性分数 $\boldsymbol{\alpha} \in \mathbb{R}^L$ 计算如下：
$$
\boldsymbol{\alpha} = \frac{(K_h W_h)(\mathbf{q}^\top W_q)^\top}{\sqrt{d_k}} + (K_c W_c)\boldsymbol{\beta}, \qquad (6)
$$
其中 $d_k$ 是投影后 query 和 key 的维度。这个相关性分数通过 query（即目标的固有特征）和 key（即行为的固有特征）之间的内积计算。另外，交叉特征由于被压缩至一维，作为偏置项。我们使用 $\boldsymbol{\beta} \in \mathbb{R}^J$ 作为可学习参数来表示交叉特征的相对重要性。

在 **CP-GSU** 中，这个相关性分数 $\boldsymbol{\alpha}$ 用于将 $L = 10^4$ 个长期历史行为截断到最相关的 100 个。在 **ESU** 中，我们对 100 个最终候选执行加权平均池化：
$$
\text{Attention}(\mathbf{q}^\top W_q, K_h W_h, K_c W_c, K W_v) = \text{Softmax}(\boldsymbol{\alpha})^\top K W_v, \qquad (7)
$$
其中 $W_v$ 是一个投影矩阵。我们略微滥用符号，将 $L$ 设为 100。这个投影 $K W_v$ 仅对 100 个行为执行，因此可以在线高效计算。我们不需要像计算 $10^4$ 个行为的 $\boldsymbol{\alpha}$ 那样对 $K$ 进行拆分。

为了联合关注来自不同表示子空间的信息，我们在 MHTA 中采用 4 个头。因此，TWIN 的最终输出定义为：
$$
\begin{aligned}
\text{TWIN} &= \text{Concat}(\text{head}_1, \ldots, \text{head}_4) W^o, \\
\text{head}_a &= \text{Attention}(\mathbf{q}^\top W_q^a, K_h W_h^a, K_c W_c^a, K W_v^a), \quad a \in \{1, \ldots, 4\},
\end{aligned} \qquad (8)
$$
 $W^o$ 是学习各个头之间相对重要性的投影矩阵。

### 3.4 系统部署

我们在快手的排序系统上部署了 TWIN，服务于 3.46 亿日活用户的主要流量。在本节中，我们介绍部署中的实践经验。我们的系统架构细节如图 3 所示。

**图 3：在线 CTR 预测系统中 TWIN 的部署。** 我们提出了必要的预计算和缓存策略来减少关键计算瓶颈，即 $10^4$ 个行为固有特征的线性投影。通过适当的频率控制，我们截断尾部视频并将候选视频池的大小限制为 80 亿。结果，固有特征投影仪可以每 15 分钟循环刷新所有候选视频的线性投影，最小化缓存的精度损失。而存储了 80 亿候选视频投影的嵌入服务器可以覆盖 97% 的在线请求，在有限的计算资源下取得了令人满意的效果。

#### 3.4.1 训练系统

我们的 TWIN 模块与整个 CTR 预测模型一起，在快手的大规模分布式近线学习系统上联合训练。每天，数亿用户访问快手，观看和互动短视频，每天产生 460 亿条观看和交互日志。每条日志在不到 8 分钟内被实时收集、预处理并用于模型训练。这个近线训练系统使用发生在不到 8 分钟前的用户-视频交互的最新知识，增量更新模型参数。

此外，我们的消息队列系统每 5 分钟将持续将最新的参数值从训练系统同步到离线推理和在线服务系统。这种同步确保了在线 CTR 预测服务始终基于最新的模型。

#### 3.4.2 离线推理

离线推理系统旨在通过提供查找服务来加速在线服务。收到查找键（一批视频 ID）后，该服务返回查找值，即对应的拼接后的投影固有特征 $K_h W_h^a$ （针对所有头 $a \in \{1, \ldots, 4\}$ ）。

具体来说，离线推理系统由两部分组成：1）**固有特征投影仪**，它使用从训练系统同步的最新嵌入和 TWIN 参数 $W_h^a$ ，循环预计算固有特征的线性投影。通过适当的频率控制，该投影仪可以每 15 分钟刷新一个 80 亿规模的候选视频池的投影固有特征，最小化缓存的精度损失。2）**嵌入服务器**，它将固有特征投影仪的结果存储为键值结构，并提供前述的键查找服务。通过截断尾部视频，80 亿个键可以覆盖 97% 的在线请求，平衡了效率和效果。

#### 3.4.3 在线服务

一旦收到请求，在线服务系统将查询离线推理系统以获取投影的固有特征 $K_h W_h^a$ ，并实时计算公式 6 的其他部分以得到用户-item相关性分数 $\boldsymbol{\alpha}$ 。然后我们选择 $\boldsymbol{\alpha}$ 中注意力分数最高的 100 个行为，将这 100 个行为输入 ESU。这种设计将 TWIN 的计算瓶颈（即具有 $10^4$ 行的 $K_h$ 的线性投影）在实践中降低了 99.3%。注意，只处理 100 个行为的 ESU 足够轻量，所有计算都可以使用从训练系统同步的最新参数实时进行。结果，ESU 计算的 $K_h W_h$ 比 CP-GSU 中的稍微更新，这进一步提升了我们 TA 机制的性能。

通过加速设计，TWIN 成功部署在快手的排序系统上，服务于 3.46 亿活跃用户的主要流量，峰值请求达到每秒 3000 万视频。

---

## 4 实验

在本节中，我们详细介绍了在真实工业数据上的离线与在线实验，以评估我们提出的方法，旨在回答以下四个研究问题（RQ）：
- **RQ1**：在终身用户行为建模中，TWIN 与其他 SOTA 方法相比在离线评估中表现如何？
- **RQ2**：TWIN 能达到多大程度的一致性？或者说，TWIN 为什么有效？
- **RQ3**：随着用户行为序列长度的增加，TWIN 的有效性如何变化？
- **RQ4**：所提方法的关键组件和不同实现的效果如何？
- **RQ5**：TWIN 在真实在线推荐系统中的表现如何？

### 4.1 数据集

为了在终身用户行为建模的真实场景中评估 TWIN，我们需要一个大规模的 CTR 预测数据集，具有丰富的用户历史行为，理想情况下每个用户可达到数万个行为。不幸的是，现有的公开数据集要么相对较小，要么缺乏足够的用户历史行为。例如，在广泛使用的 Amazon 数据集 [11, 13, 17] 中，每个用户平均不到 10 个历史行为。在淘宝数据集 [18, 19, 35–37] 中，用户行为的平均序列长度最多为 500。因此，我们从中国顶尖的短视频分享平台快手收集了一个工业数据集。

我们从每日用户日志中构建样本，以用户的点击行为作为标签。如表 3 所示，快手的日活跃用户规模约为 3.46 亿。每天有 4500 万个短视频被发布，这些视频总共被播放 460 亿次。平均而言，每个用户每天观看 133.7 个短视频。为了利用丰富的行为信息，我们收集了数月前旧日志中的完整用户历史行为。平均而言，每个用户在过去六个月中观看了 14,500 个视频，这为模型提供了肥沃的用户历史行为学习土壤。我们将最大用户行为序列长度截断为 100,000，这大约是重度用户每年的总观看次数。

**表 3：基于快手每日收集的用户日志构建的工业数据集统计信息**

| 数据字段 | 规模 |
|---------|------|
| 用户数 | 3.455 亿 |
| 视频数 | 4510 万 |
| 每日日志 | 样本数 | 462 亿 |
| | 平均用户行为数 | 133.7/天 |
| 历史行为 | 平均用户行为数 | 14,500 |
| | 最大用户行为数 | 100,000 |

### 4.2 基线方法

为了证明有效性，我们将 TWIN 与以下 SOTA 终身用户行为建模算法进行了比较。
- **Avg-Pooling**：用户终身行为的平均池化。
- **DIN [36]**：最广泛使用的短期行为建模方法，利用 TA 获取目标特定的兴趣。
- **SIM Hard [19]**：GSU 选择与目标item同类别的行为，ESU 采用 DIN 中的 TA。在我们的场景中，视频类别的总数为 37。
- **ETA [3]**：使用局部敏感哈希（LSH）为目标视频和行为生成哈希签名。然后 GSU 使用汉明距离作为目标-行为相关性度量。
- **SDIM [1]**：GSU 通过多轮哈希碰撞选择与目标视频具有相同哈希签名的行为。在原始论文中，ESU 对来自多轮的采样行为进行线性聚合以获取用户兴趣。在我们的实验中，为了公平比较，在 ESU 中采用了更强大的 TA。
- **SIM Cluster**：由于"类别"需要昂贵的人工标注，且在短视频场景中通常不可靠，我们将 SIM Cluster 实现为 SIM Hard 的改进变体。我们基于预训练嵌入将视频分组到 1,000 个簇中。GSU 检索与目标item同一簇中的行为。
- **SIM Cluster+**：SIM Cluster 的改进，簇的数量从 1,000 扩展到 10,000。
- **SIM Soft [19]**：GSU 使用视频预训练嵌入的内积分数来检索相关行为。内积是比汉明距离和哈希碰撞更精细的检索方法，但计算成本更高。

总之，ETA 和 SDIM 采用端到端训练方法，但使用粗略的检索方法来避免高复杂度计算。SIM Cluster(+) 和 SIM Soft 使用精细的检索方法，但代价是它们必须使用预训练嵌入并提前生成离线倒排索引。注意，SIM Soft 尚未被后续的工作 ETA 和 SDIM 击败。

我们不与 UBR4CTR [20, 21] 比较，因为其迭代训练不适合我们的流式场景。此外，UBR4CTR 已被证实性能劣于 SIM Hard 和 ETA [3]。

### 4.3 实验设置

我们使用一天中连续 23 小时的样本作为训练数据，接下来一小时的样本作为测试。我们在连续 5 天评估所有算法，并报告每天的平均性能。

对于离线评估，我们使用两个广泛采用的指标：**AUC** 和 **GAUC**。AUC 表示正样本得分高于负样本得分的概率，反映模型的排序能力。GAUC 对所有用户的 AUC 进行加权平均，权重设置为每个用户的样本数。GAUC 消除了用户之间的偏差，以更细粒度和公平的粒度评估模型性能。

为了公平比较，在所有算法中，我们使用相同的网络结构，包括嵌入层、上层深度网络、短期行为建模和其他任务建模，但长期行为建模模块除外。对于两阶段模型，我们使用最近的 10,000 个行为作为 GSU 的输入，并检索 100 个行为用于 ESU 中的 TA。对于 DIN，由于其在处理长序列时的瓶颈，我们使用最近的 100 个行为。尽管 TWIN 的 CP-GSU 在注意力分数计算中使用 4 个头，但我们递归地遍历 4 个头排序的item，直到收集到 100 个唯一的行为。对于所有模型，嵌入层使用 AdaGrad 优化器，学习率为 0.05。DNN 参数由 Adam 优化，学习率为 5.0e-06。批大小设置为 8192。

### 4.4 整体性能（RQ1）

表 4 展示了所有模型的性能。注意，由于我们数据集中大量的用户和样本，离线评估中 AUC 和 GAUC 提高 0.05% 就足以带来在线业务收益。

**表 4：与 SOTA 方法的离线比较（RQ1）。** 我们报告了连续 5 天的均值和标准差（std）。最佳和第二佳结果分别以**粗体**和_下划线_突出显示。注意，TWIN 在 AUC 上比最佳对比模型提高 0.29%，在 GAUC 上提高 0.51%。这些改进远大于 0.05%（足以带来在线收益的值）。

| 方法 | AUC（均值 $\pm$ 标准差） $\uparrow$ | GAUC（均值 $\pm$ 标准差） $\uparrow$ |
|------|-------------------|--------------------|
| Avg-Pooling | 0.7855 $\pm$ 0.00023 | 0.7168 $\pm$ 0.00019 |
| DIN | 0.7873 $\pm$ 0.00014 | 0.7191 $\pm$ 0.00012 |
| SIM Hard | 0.7901 $\pm$ 0.00016 | 0.7224 $\pm$ 0.00021 |
| ETA | 0.7910 $\pm$ 0.00004 | 0.7243 $\pm$ 0.00011 |
| SIM Cluster | 0.7915 $\pm$ 0.00017 | 0.7253 $\pm$ 0.00018 |
| SDIM | 0.7919 $\pm$ 0.00009 | 0.7267 $\pm$ 0.00006 |
| SIM Cluster+ | 0.7927 $\pm$ 0.00009 | 0.7275 $\pm$ 0.00011 |
| SIM Soft | 0.7939 $\pm$ 0.00014 | 0.7299 $\pm$ 0.00013 |
| **TWIN** | **0.7962 $\pm$ 0.00008** | **0.7336 $\pm$ 0.00011** |
| **改进幅度** | **+0.29%** | **+0.51%** |

**第一**，TWIN 显著优于所有基线，尤其是那些具有不一致 GSU 的两阶段 SOTA 方法。这验证了 TWIN 在终身行为建模中的关键优势，即 CP-GSU 中强大且一致的 TA。具体来说，CP-GSU 精确检索出 ESU 认为高相关的行为，为最重要的用户信息节省了 ESU 宝贵的计算资源。而在其他方法中，无效且不一致的 GSU 可能会遗漏重要行为并引入噪声，降低 TA 的性能。此外，从 Avg-Pooling 到 DIN 的增益显示了 TA 在检索有效信息方面的能力。从 DIN 到其他两阶段 SOTA 的增益验证了建模长期行为的必要性。这两者共同支持了我们的动机：将 TA 扩展到长序列。

**第二**，仅有端到端训练是不够的。我们观察到 TWIN 明显优于 ETA 和 SDIM（这两个强基线的 GSU 中的嵌入也是端到端训练的）。特别是，ETA 使用 LSH 和汉明距离，SDIM 使用多轮哈希碰撞。这两种 GSU 策略都比 TA 精度低，并且与它们 ESU 中使用的目标-行为相关性度量不一致。而 TWIN 中的 CP-GSU 不仅端到端训练，而且与 ESU 中的 TA 一致。这表明，精确的相关性度量对 GSU 至关重要，验证了我们相对于现有端到端算法的优势。

**第三**，以更细粒度建模终身行为是有效的。我们比较了 SIM 的变体：37 个类别的 SIM Hard、1,000/10,000 个簇的 SIM Cluster(+) 以及每个行为独立计算目标-行为相关性分数的 SIM Soft。我们观察到，随着 GSU 中使用的检索方法粒度越来越细，性能持续提升。这是因为当 GSU 能够更精细地捕捉视频之间的相关性分数时，它会更准确地检索行为。从这个角度来看，我们进一步将优于 SIM Soft 的优势归因于 TWIN 采用了更精确的相关性度量。

### 4.5 一致性分析（RQ2）

正如我们所声称的，我们优越的行为建模能力源于 CP-GSU 和 ESU 中一致的相关性度量。但我们实际达到了一致性有多大呢（图 4）？

**图 4：GSU 和 ESU 之间的一致性（RQ2）。** Oracle 使用 ESU 的相关性度量在 $10^4$ 个行为上进行排序，得到"真正的 top-100"。SIM Hard 的 GSU 返回的 100 个行为中，只有 40 个命中。TWIN 将命中数提升至 94。从理论上讲，TWIN 可以接近 100% 的命中率，因为我们采用一致的 GSU。然而，我们的实际性能受到部署约束的限制，即缓存中 15 分钟的刷新延迟。

对于每个训练好的两阶段模型，我们使用其 ESU 的参数作为其 Oracle，从 10,000 个行为中检索"真正的 top-100"。换句话说，这些真正的 top-100 是 ESU 认为真正重要的 ground-truth。然后，我们遍历所有对比算法从 10 到 $10^4$ 的 GSU 输出大小，检查有多少输出命中真正的 top-100。注意，每个对比算法都有其自己的 Oracle 和 top-100。但我们只绘制了一条 Oracle 曲线，因为所有 Oracle 都完美地命中了 ground-truth。

SIM Soft 得益于 GSU 中更精确的检索策略，检索一致性有所提升。进一步，TWIN 在返回 100 个行为时达到 94 个命中，这验证了我们在保持两阶段一致性方面的优势。注意，这是最值得注意的值，因为在推理时间和 TA 计算复杂度的约束下，100 是 ESU 输入的上界。由于在 3.4.2 节中描述的缓存刷新延迟，我们在实践中未达到理论上的 100% 一致性。

基于上述结果，我们推测 CP-GSU 比传统的 GSU 具有更强的将用户兴趣与目标视频匹配的能力。这归因于一致性。通过与 ESU 共享相同的结构和参数，CP-GSU 能够准确判断和检索具有相似内容的视频。此外，随着 CP-GSU 参数实时更新，模型能够捕捉用户动态演化的兴趣。

### 4.6 行为长度的影响（RQ3）

我们旨在测试 TWIN 在不同行为序列长度下的有效性，并进一步挖掘 TWIN 的潜力。注意，只有 GSU 的输入序列长度被改变，输出长度保持 100。结果如图 5 所示。

**图 5：GSU 采用不同行为序列长度时的性能（RQ3）。** 左图：GAUC 值。右图：相对于长度 = 10,000 的 GAUC 相对改进百分比。随着序列长度的增加，所有模型的性能都有所提升，TWIN 与基线之间的性能差距也在增大。

我们观察到：1）TWIN 始终表现最佳；2）TWIN 与其他方法之间的性能差距随着序列长度的增加而增大。这表明 TWIN 在建模**极长序列**方面具有更好的有效性。

### 4.7 消融研究（RQ4）

我们对 TWIN 应用不同的操作进行消融研究，以评估我们关键模型设计的贡献：1）两个阶段之间的一致性；2）高效的 MHTA。

**图 6：TWIN 中关键组件的效果（RQ4）。** 左图：TWIN w/o Para-Cons（一致网络结构 + 不一致参数）优于 SIM Soft（两者都不一致），但劣于 TWIN（两者一致）。右图：直接移除用户-item交叉特征（TWIN w/o Bias）节省的计算量很小，但与 TWIN 相比导致 GAUC 显著下降。与使用原始 MHTA 的 TWIN（TWIN w/Raw MHTA）相比，将item固有特征压缩为偏置以提升效率（TWIN）几乎不损害性能，但大大加快了推理时间。

TWIN 在两个层面保持了两阶段之间的一致性：网络结构和参数。为了研究各自的贡献，我们实现了一个变体 **TWIN w/o Para-Con**，它不保持参数一致性。具体来说，我们首先训练一个辅助模型 **TWIN-aux**，它使用与 TWIN 相同的网络结构和训练数据，但分别训练。然后我们将 TWIN-aux 的 GSU 参数同步到 TWIN w/o Para-Con。这是为了确保 TWIN w/o Para-Con 仍然实时更新，并且 TWIN 和 TWIN w/o Para-Con 之间的差异完全由参数不一致引起。

如图 6（左）所示，TWIN w/o Para-Con 显著优于 SIM Soft（结构和参数都不一致），但略差于 TWIN。这表明结构一致性和参数一致性都有益，但网络结构一致性的贡献更大。

为了高效计算 MHTA 以用于工业部署，我们拆分用户行为特征，并将每个用户-item交叉特征压缩为一维偏置项。为了研究这种修改的影响以及保留注意力计算中用户-item交叉特征的益处，我们实现了两个变体并比较了它们的性能和推理时间：使用原始 MHTA 的 TWIN（其中使用直接的线性投影 $KW$ ，不进行特征拆分）和不在 MHTA 中使用用户-item交叉特征的 TWIN，分别缩写为 **TWIN w/Raw MHTA** 和 **TWIN w/o Bias**。

如图 6（右）所示，TWIN 显著优于 TWIN w/o Bias，并且与 TWIN w/Raw MHTA 表现几乎相同，验证了我们提出的 MHTA 修改几乎不损害性能。关于计算成本，由于当用户-item交叉特征用于 $K$ 的线性投影时，缓存对 TWIN w/Raw MHTA 不适用（详见第 3.3.2 节），TWIN w/Raw MHTA 的推理时间显著增加。相反，移除用户-item交叉特征（TWIN w/o Bias）并不能节省太多计算，却损害了性能。

**表 5：TWIN 在在线 A/B 测试中相比 SIM Hard 和 SIM Soft 的相对观看时长改进（RQ5）。** 在快手的场景中，0.1% 的增长就足以成为一个显著的改进，带来巨大的商业效益。

| 场景 | 精选视频 Tab | 发现 Tab | 滑动 Tab |
|------|-------------|---------|---------|
| 对比 SIM Hard | +4.893% | +3.712% | +6.249% |
| 对比 SIM Soft | +2.778% | +1.374% | +2.705% |

### 4.8 在线结果（RQ5）

为了评估 TWIN 的在线性能，我们在快手的短视频推荐平台上进行了严格的在线 A/B 测试。表 5 比较了 TWIN 与 SIM Hard 和 SIM Soft 在快手的三个代表性业务场景（精选视频 Tab、发现 Tab 和滑动 Tab）中的性能。与电商中通常使用 CTR 和 GMV 作为在线评估指标不同，短视频推荐场景通常使用**观看时长**，即用户观看视频的总时间。如图所示，TWIN 在所有场景中都显著优于 SIM Hard 和 SIM Soft。鉴于 0.1% 的观看时长增长在快手就被视为有效的改进，TWIN 实现了显著的商业收益。

---

## 5 结论

为了解决传统终身行为建模算法的不一致性问题，我们提出了一种**一致性保持的两阶段兴趣模型**，成功地将有效但计算昂贵的 MHTA 从 ESU 扩展到 CP-GSU，即从 100 的序列长度扩展到 $10^4-10^5$ 。具体来说，我们设计了新颖的 MHTA 机制以及高效的工业基础设施，包括行为特征拆分与压缩、预计算与缓存、在线训练与参数同步。我们将计算瓶颈加速了 99.3%，这促成了 TWIN 在快手的成功部署，服务于数亿活跃用户的主要流量。两个阶段之间的一致性，加上 CP-GSU 中有效的基于 TA 的相关性度量，最大化了行为建模的检索效果，并显著提升了 CTR 预测的性能。据我们所知，TWIN 是首个在两阶段终身行为建模问题中实现一致性的方法。

---

## 参考文献

[1] Yue Cao, Xiao Jiang Zhou, Jiaqi Feng, Peihao Huang, Yao Xiao, Dayao Chen, and Sheng Chen. 2022. Sampling Is All You Need on Modeling Long-Term User Behaviors for CTR Prediction. *arXiv preprint arXiv:2205.10249* (2022).

[2] Junxuan Chen, Baigui Sun, Hao Li, Hongtao Lu, and Xian-Sheng Hua. 2016. Deep CTR prediction in display advertising. In *Proceedings of the 24th ACM international conference on Multimedia*. 811–820.

[3] Qiwei Chen, Changhua Pei, Shanshan Lv, Chao Li, Junfeng Ge, and Wenwu Ou. 2021. End-to-End User Behavior Retrieval in Click-Through Rate Prediction Model. *arXiv preprint arXiv:2108.04468* (2021).

[4] Qiwei Chen, Huan Zhao, Wei Li, Pipei Huang, and Wenwu Ou. 2019. Behavior sequence transformer for e-commerce recommendation in Alibaba. In *Proceedings of the 1st International Workshop on Deep Learning Practice for High-Dimensional Sparse Data*. 1–4.

[5] Heng-Tze Cheng, Levent Koc, Jeremiah Harmsen, Tal Shaked, Tushar Chandra, Hrishi Aradhye, Glen Anderson, Greg Corrado, Wei Chai, Mustafa Ispir, et al. 2016. Wide & deep learning for recommender systems. In *Proceedings of the 1st workshop on deep learning for recommender systems*. 7–10.

[6] Kyunghyun Cho, Bart Van Merriënboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. 2014. Learning phrase representations using RNN encoder-decoder for statistical machine translation. *arXiv preprint arXiv:1406.1078* (2014).

[7] Junyoung Chung, Caglar Gulcehre, KyungHyun Cho, and Yoshua Bengio. 2014. Empirical evaluation of gated recurrent neural networks on sequence modeling. *arXiv preprint arXiv:1412.3555* (2014).

[8] Paul Covington, Jay Adams, and Emre Sargin. 2016. Deep neural networks for YouTube recommendations. In *Proceedings of the 10th ACM conference on recommender systems*. 191–198.

[9] Yufei Feng, Fuyu Lv, Weichen Shen, Menghan Wang, Fei Sun, Yu Zhu, and Keping Yang. 2019. Deep session interest network for click-through rate prediction. *arXiv preprint arXiv:1905.06482* (2019).

[10] Huifeng Guo, Ruiming Tang, Yunming Ye, Zhenguo Li, and Xiuqiang He. 2017. DeepFM: a factorization-machine based neural network for CTR prediction. *arXiv preprint arXiv:1703.04247* (2017).

[11] Ruining He, Wang-Cheng Kang, and Julian McAuley. 2017. Translation-based recommendation. In *Proceedings of the eleventh ACM conference on recommender systems*. 161–169.

[12] Yuchin Juan, Yong Zhuang, Wei-Sheng Chin, and Chih-Jen Lin. 2016. Field-aware factorization machines for CTR prediction. In *Proceedings of the 10th ACM conference on recommender systems*. 43–50.

[13] Wang-Cheng Kang and Julian McAuley. 2018. Self-attentive sequential recommendation. In *2018 IEEE international conference on data mining (ICDM)*. IEEE, 197–206.

[14] Chao Li, Zhiyuan Liu, Mengmeng Wu, Yuchi Xu, Huan Zhao, Pipei Huang, Guoliang Kang, Qiwei Chen, Wei Li, and Dik Lun Lee. 2019. Multi-interest network with dynamic routing for recommendation at Tmall. In *Proceedings of the 28th ACM International Conference on Information and Knowledge Management*. 2615–2623.

[15] Jianxun Lian, Xiaohuan Zhou, Fuzheng Zhang, Zhongxia Chen, Xing Xie, and Guangzhong Sun. 2018. xDeepFM: Combining explicit and implicit feature interactions for recommender systems. In *Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining*. 1754–1763.

[16] Hongche Liu and MS Zamanian. 2007. Framework for selecting and delivering advertisements over a network based on combined short-term and long-term user behavioral interests. US Patent App. 11/225,238.

[17] Julian McAuley, Christopher Targett, Qinfeng Shi, and Anton Van Den Hengel. 2015. Image-based recommendations on styles and substitutes. In *Proceedings of the 38th international ACM SIGIR conference on research and development in information retrieval*. 43–52.

[18] Qi Pi, Weijie Bian, Guorui Zhou, Xiaoqiang Zhu, and Kun Gai. 2019. Practice on long sequential user behavior modeling for click-through rate prediction. In *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*. 2671–2679.

[19] Qi Pi, Guorui Zhou, Yujing Zhang, Zhe Wang, Lejian Ren, Ying Fan, Xiaoqiang Zhu, and Kun Gai. 2020. Search-based user interest modeling with lifelong sequential behavior data for click-through rate prediction. In *Proceedings of the 29th ACM International Conference on Information & Knowledge Management*. 2685–2692.

[20] Jiarui Qin, Weinan Zhang, Rong Su, Zhirong Liu, Weiwen Liu, Guangpeng Zhao, Hao Li, Ruiming Tang, Xiuqiang He, and Yong Yu. 2023. Learning to Retrieve User Behaviors for Click-Through Rate Estimation. *ACM Transactions on Information Systems* (2023).

[21] Jiarui Qin, Weinan Zhang, Xin Wu, Jiarui Jin, Yuchen Fang, and Yong Yu. 2020. User behavior retrieval for click-through rate prediction. In *Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval*. 2347–2356.

[22] Steffen Rendle. 2010. Factorization machines. In *2010 IEEE International conference on data mining*. IEEE, 995–1000.

[23] Yang Song, Ali Mamdouh Elkahky, and Xiaodong He. 2016. Multi-rate deep learning for temporal recommendation. In *Proceedings of the 39th International ACM SIGIR conference on Research and Development in Information Retrieval*. 909–912.

[24] Fei Sun, Jun Liu, Jian Wu, Changhua Pei, Xiao Lin, Wenwu Ou, and Peng Jiang. 2019. BERT4Rec: Sequential recommendation with bidirectional encoder representations from transformer. In *Proceedings of the 28th ACM international conference on information and knowledge management*. 1441–1450.

[25] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. *Advances in neural information processing systems* 30 (2017).

[26] Ruoxi Wang, Bin Fu, Gang Fu, and Mingliang Wang. 2017. Deep & cross network for ad click predictions. In *Proceedings of the ADKDD'17*. 1–7.

[27] Ruoxi Wang, Rakesh Shivanna, Derek Cheng, Sagar Jain, Dong Lin, Lichan Hong, and Ed Chi. 2021. DCN V2: Improved deep & cross network and practical lessons for web-scale learning to rank systems. In *Proceedings of the Web Conference 2021*. 1785–1797.

[28] Yongji Wu, Defu Lian, Neil Zhenqiang Gong, Lu Yin, Mingyang Yin, Jingren Zhou, and Hongxia Yang. 2021. Linear-time self attention with codeword histogram for efficient recommendation. In *Proceedings of the Web Conference 2021*. 1262–1273.

[29] Jun Xiao, Hao Ye, Xiangnan He, Hanwang Zhang, Fei Wu, and Tat-Seng Chua. 2017. Attentional factorization machines: Learning the weight of feature interactions via attention networks. *arXiv preprint arXiv:1708.04617* (2017).

[30] Zhibo Xiao, Luwei Yang, Wen Jiang, Yi Wei, Yi Hu, and Hao Wang. 2020. Deep multi-interest network for click-through rate prediction. In *Proceedings of the 29th ACM International Conference on Information & Knowledge Management*. 2265–2268.

[31] Feng Yu, Qiang Liu, Shu Wu, Liang Wang, and Tieniu Tan. 2016. A dynamic recurrent model for next basket recommendation. In *Proceedings of the 39th International ACM SIGIR conference on Research and Development in Information Retrieval*. 729–732.

[32] Li Zhang, Weichen Shen, Jianhang Huang, Shijian Li, and Gang Pan. 2019. Field-aware neural factorization machine for click-through rate prediction. *IEEE Access* 7 (2019), 75032–75040.

[33] Weinan Zhang, Tianming Du, and Jun Wang. 2016. Deep learning over multi-field categorical data. In *European conference on information retrieval*. Springer, 45–57.

[34] Chang Zhou, Jinze Bai, Junshuai Song, Xiaofei Liu, Zhengchao Zhao, Xiusi Chen, and Jun Gao. 2018. ATrank: An attention-based user behavior modeling framework for recommendation. In *Thirty-Second AAAI Conference on Artificial Intelligence*.

[35] Guorui Zhou, Na Mou, Ying Fan, Qi Pi, Weijie Bian, Chang Zhou, Xiaoqiang Zhu, and Kun Gai. 2019. Deep interest evolution network for click-through rate prediction. In *Proceedings of the AAAI conference on artificial intelligence*, Vol. 33. 5941–5948.

[36] Guorui Zhou, Xiaoqiang Zhu, Chenru Song, Ying Fan, Han Zhu, Xiao Ma, Yanghui Yan, Junqi Jin, Han Li, and Kun Gai. 2018. Deep interest network for click-through rate prediction. In *Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining*. 1059–1068.

[37] Han Zhu, Xiang Li, Pengye Zhang, Guozheng Li, Jie He, Han Li, and Kun Gai. 2018. Learning tree-based deep model for recommender systems. In *Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*. 1079–1088.
