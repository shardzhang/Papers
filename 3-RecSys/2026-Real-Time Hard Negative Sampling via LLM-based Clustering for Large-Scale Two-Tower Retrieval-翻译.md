# 基于LLM聚类的实时难负样本采样，用于大规模双塔检索

> Ivan Ji¹\*、Liuyi Hu¹\*、Harrison (Zihao) Zhao¹\*、Lei Huang¹、Qunshu Zhang¹、Max (Xiangjun) Fan¹、Aameek Singh¹ | Meta
>
> \*三位作者对本研究贡献相同。



本文提出了一种新的自监督难负样本采样技术：利用大语言模型（LLM）学习媒体表示，在训练过程中从同一簇（cluster）内生成难负样

本，并配套设计了一个可实时、可大规模上线的采样框架 GOOBS，**解决传统负采样"负样本太简单"和推荐系统"流行度偏差"两大问题**。

核心内容：

- 双塔（two-tower）模型是大规模推荐系统召回阶段的标配，行业标准训练采用 batch 内 和/或 batch 外负采样，但这些方法往往产生"简单负样本"，对模型的挑战不足
- 提出基于簇的自监督难负样本采样：从正样本所在簇内采样难负样本，让生成的负样本更有挑战性、信息量更大
- 提出 GOOBS（Global Out-of-Batch Sampling）实时采样框架，能无缝集成进生产模型，处理数十亿训练数据点而计算开销极小
- 在公开数据集和大型线上系统上的实验证明，该负采样技术优于行业广泛使用的方法，并能打破推荐中的反馈回路、显著缓解流行度偏差

关键发现：

- 在公开数据集（MovieLens-1M、Amazon Reviews）上，Cluster GOOBS 在所有指标上均取得最佳表现，相比 batch 内基线 HR@50 提升 +7.2% 到 +55.6%
- 线上 A/B 测试中，作为生产推荐系统的召回来源，相比原模型 CTR 提升 +53%，训练 QPS 仅回归 −1.4%
- Top 100 item 的曝光占比从 50% 降到 32%，显著缓解流行度偏差

---



## 摘要

双塔模型已被广泛用于大规模推荐系统，尤其是召回（retrieval）阶段。训练双塔模型的行业标准通常涉及 batch 内（in-batch）和/或 batch 外（out-of-batch）负采样。然而，这些方法往往产生模型能快速学会的简单负样本，无法充分挑战模型。为解决这一问题，本文提出一种新颖的自监督难负样本采样技术，利用大语言模型（LLM）在模型训练过程中从同一簇生成难负样本。通过利用 LLM 学习媒体表示，所提方法确保生成的负样本更具挑战性和信息量。该实时采样框架设计为可无缝集成进生产模型，能够以极小的计算复杂度处理数十亿训练数据点。在公开数据集上的实验以及在大型线上系统中的部署表明，所提负采样技术优于行业广泛使用的方法。此外，工业应用中的分析表明，该采样方法有助于打破推荐中固有的反馈回路，并显著减少流行度偏差。

日期：2026年7月7日
通讯作者：Harrison (Zihao) Zhao，邮箱 harrisonzhao@meta.com



## 1 引言

在大数据时代，大规模推荐系统在电子商务、社交媒体、娱乐等各种应用中变得越来越重要。这些系统旨在为用户提供满足其兴趣和偏好的个性化推荐。然而，由于可选候选item数量巨大，设计一个高效且有效的推荐系统是一项具有挑战性的任务。为应对这一挑战，现代推荐系统通常采用多阶段设计，包括召回（retrieval）和排序（ranking）阶段 [9,18,8]。召回阶段旨在从大规模语料库中召回一小部分相关候选，而排序阶段进一步精化选择，以提供最相关的推荐。双塔模型设计因其服务效率而被广泛用作大规模应用中的召回模型 [9,14]。

召回模型训练通常被表述为一个极端分类（extreme classification）问题，其中负样本起着关键作用。研究者提出了各种基于采样的技术来提升训练效率 [4,3,9]。广泛使用的采样技术包括 batch 内负采样 [12,11] 和 batch 外（OOB）负采样，以及混合负采样 [27,13]。然而，batch 内负样本受限于 mini-batch 大小，可能导致模型在训练中出现推荐偏差，且接触不到多样化的语料库。而 OOB 负样本对模型来说又太容易学习，尤其是当 OOB item 池非常大且多样时。

此外，召回模型通常使用用户行为（如点击）作为正样本。然而，用户是否点击取决于我们向用户展示了哪个 item，而这又由多阶段系统控制。强烈的反馈回路会导致推荐中的流行度偏差 [6,5,19,20]。



## 2 相关工作

### 2.1 多阶段系统

对于大规模推荐系统，有数百万个候选item符合选择条件。由于延迟约束，用复杂的排序模型对所有候选item进行排序是不可行的。为平衡推荐效率和效果，现代推荐系统通常采用多阶段设计：召回和排序。召回阶段旨在从大规模语料库中召回一小部分子集。然后部署若干排序阶段对召回的候选集进行重新排序。这种多阶段系统在工业界和学术界都得到了广泛应用 [9,18,8]。

### 2.2 双塔模型

双塔模型是神经网络架构的一种变体，自 [15] 提出以来，一直是推荐系统中的强大工具。它被广泛用于工业应用的召回阶段 [9,14]。其中一个塔通常用于建模用户交互，另一个塔用于建模 item 特征。用户和 item 由用户塔和 item 塔的嵌入（embedding）表示。然后召回问题被转化为嵌入空间中的最近邻（NN）搜索问题。该架构的最大优势是执行效率，因为整个语料库的 item 嵌入可以预先计算并建立索引，因此在线推理无需实时计算。

### 2.3 负采样

召回模型训练通常被表述为一个极端分类问题，研究者提出了各种基于采样的技术来提升训练效率 [4,3,9]。

#### 2.3.1 Batch 内负采样

它将同一 mini-batch 中其他用户的正item视为负item，而不是从语料库中选取。由于其时间和内存效率而得到普遍使用 [12,11]。然而，batch 内负样本本质上受限于 mini-batch 的大小。这可能导致模型遭受推荐偏差，且无法让模型接触多样化的候选item进行训练。为缓解这一问题，研究者提出了多种变体。例如，[22] 提出了跨 batch 负采样（cross-batch negative sampling），利用最近几个 mini-batch 编码好的 item 嵌入来提升模型训练。

#### 2.3.2 LogQ 修正

LogQ 修正是大规模推荐系统中使用的一项技术，用于修正训练期间采样 softmax 引入的偏差——流行 item 更可能被采样为负样本。该技术已被 Google [28,27]、ByteDance [26]、Kuaishou [17] 等公司广泛采用。它的做法是通过减去负item在训练 batch 中出现概率的对数来调整模型的 logits（原始输出分数），从而减少对流行 item 的过度惩罚，提高模型召回低频 item 的能力。这一修正有助于确保模型从真实偏好信号中学习，而不是从采样过程的统计假象中学习。

#### 2.3.3 Batch 外（OOB）负采样

Batch 外负样本是从当前处理的 mini-batch 中不存在的 item 池中选择的。通过从当前 mini-batch 之外的 item 中采样，它提供了更广泛的负样本范围，由于模型会接触到更多种类的可能被视为"不相关"的 item，因此可能带来更好的模型泛化能力。它也使训练与服务更一致，因为在服务阶段，模型需要从整个语料库而非仅 mini-batch 中的 item 召回相关 item。然而，与 batch 内采样相比，它在计算上可能更昂贵，因为它可能需要访问和处理当前 mini-batch 之外的数据。此外，随机的 OOB 负样本对模型来说是非常容易区分的简单负样本。为克服这一障碍，研究中提出了许多不同的难负样本方法，如动态负采样（Dynamic Negative Sampling） [31] 和自适应采样（Adaptive Sampling） [7,23]。但由于计算成本，它们并未被大规模工业应用广泛采用。

#### 2.3.4 混合负采样

混合负采样使用 batch 内和 batch 外采样负样本的混合，以解决隐式用户反馈的选择偏差。该技术也被广泛用于各种应用 [27,13]。

### 2.4 流行度偏差

推荐系统的理想状态是以个性化的方式帮助用户找到最相关的item。然而，在实际应用中，我们经常看到推荐中存在流行度偏差。流行度偏差是指推荐系统倾向于向用户推荐相当流行的item，而牺牲那些用户可能觉得相关的较小众item。过去十年，由于其实践重要性，推荐系统中的流行度偏差问题引起了研究者的广泛关注 [6,5,19,20]。

总体而言，推荐系统中的流行度偏差有以下负面影响：（1）使用户与有限数量的item交互，从而损害用户体验；（2）不探索长尾 item，导致无法学习其嵌入；（3）造成强烈的系统反馈回路且难以打破，因为已经流行的item会获得更多曝光，从而变得更加流行。

为缓解这一问题，有些方法直接在模型训练中纠正偏差 [1,21,24]，有些方法则通过用偏差因子调整模型预测来采取后处理策略 [2,32]。另一种典型方法是在损失函数中给 item 分配与 item 流行度成反比的权重，以平衡曝光数据中流行与不流行的 item [21]。



## 3 方法

### 3.1 问题表述

将带标签的样本记为 $\{x_i, y_i, r_i\}_{i=1}^n$，其中 $x_i$ 表示用户侧信息，$y_i$ 表示 item 侧信息，$r_i$ 表示第 $i$ 个样本对 $(x_i, y_i)$ 的标签。

召回模型通常被建模为极端多分类器。模型的目标是预测用户 $x_i$ 与 item $y_i$ 交互的概率：

$$
P(y_i|x_i; \theta) = \frac{e^{s(x_i, y_i|\theta)}}{\sum_{j \in I} e^{s(x_i, y_j|\theta)}}
$$

其中 $I$ 是符合条件的 item 集合，$s(x_i, y_i|\theta)$ 是基于 $x_i$ 和 $y_i$ 的某个函数，$\theta$ 是模型参数。例如，在基本双塔模型设置中，$s(x_i, y_i) = v_i^T u_i$，其中 $v_i$ 和 $u_i$ 分别是用户塔和 item 塔输出的嵌入。在工业应用中，item 集合 $I$ 的基数至少是百万级。

考虑到广泛采用的交叉熵损失，需要优化的损失函数为：

$$
L(\{x_i, y_i, r_i\}_{i=1}^n) = -\frac{1}{n} \sum_{i=1}^n r_i \cdot \log(P(y_i|x_i; \theta))
$$

### 3.2 自监督基于簇的负采样

为提升训练效率，所提方法依赖于一种自监督基于簇的负采样技术，从 item 池 $I$ 中采样难负样本。首先，对 item 池 $I$ 进行聚类，每个 $y_j$ 被分配一个簇 ID。聚类可以基于各种维度进行，例如 item 类别、item 主题等。然后，对于带标签样本中的每个样本对 $(x_i, y_i)$，从与 $y_i$ 相同的簇中采样 $y_{i1}^{-n}, y_{i2}^{-n}, \ldots, y_{iK}^{-n}$ 个负样本。$\{y_{ik}^{-n}\}$ 是 $(x_i, y_i)$ 的难负样本，因为它们在某种程度上与 $y_i$ 相似。

对于每个样本，针对真实标签和采样得到的负类别最小化交叉熵损失，即：

$$
L(\{x_i, y_i, r_i\}, \{x_i, y_{ik}^{-n}, 0\}_{k=1}^K) = -r_i \cdot \log\left( \frac{e^{s(x_i, y_i|\theta)}}{\sum_{j \in \{y_i, \{y_{i1}^{-n}\}_{k=1}^K\}} e^{s(x_i, y_j|\theta)}} \right)
$$

### 3.3 基于簇的负样本的理论依据

基于簇的负采样的有效性可以从对比学习理论和梯度方差分析的角度进行形式化理解。在使用 InfoNCE 或 softmax 交叉熵损失训练的标准双塔模型中，负样本 $x_j^{-}$ 相对于用户锚点 $x_u$ 的梯度贡献与其指数化相似度 $\exp(s(x_u, x_j^{-}))$ 成正比。均匀采样的负样本通常位于嵌入空间中离锚点较远的位置，导致相似度极小，从而梯度几乎为零，提供的学习信号微乎其微。

通过在正item所在的同一语义簇内采样负样本，所提方法确保 $x_j^{-}$ 与正item共享潜在特征，从而在完全收敛之前保证更高的相似度分数 $s(x_u, x_j^{-})$。这把负样本推向模型当前的决策边界附近。因此，梯度幅度显著增大，迫使模型学习簇内 item 之间的细粒度区别，而不是依赖琐碎的簇间差异。这一方法与近似最近邻负对比估计（ANCE） [25] 中的理论发现一致，该工作证明从查询局部邻域抽取的负样本能产生更高的梯度范数，并能更好地逼近 oracle 重要性采样分布。

### 3.4 基于LLM的簇生成

在基于簇的负采样过程中，选择合适的簇至关重要。传统上，媒体类型或类别被用作聚类选项，如第 5.1 节所讨论。然而，对于本文描述的工业应用，item 簇由自研的多模态内容嵌入模型导出，与传统基于类别的聚类方法相比具有显著优势。

内容嵌入模型由一组基于大语言模型（LLM）微调的编码器构成，旨在跨文本、图像和视频模态生成语义丰富的 item 表示。图 1 展示了工作流程，包括以下步骤：

- **预训练（Pre-training）**：利用预训练的 LLM 实现稳健的通用语言理解。这一基础步骤确保模型捕捉复杂的语义关系，这些关系往往被仅基于表面特征（如类型或类别）的传统聚类方法遗漏。
- **多模态编码器（Multimodal Encoder）**：该模块使用基于 transformer 的架构处理包括文本、图像和视频在内的多种输入类型，将它们编码为固定大小的向量表示。这种多模态能力使得对媒体内容的理解更细致、更全面，超越了通常依赖单模态数据的传统聚类的局限。
- **微调（Fine-tuning）**：然后对模型进行微调以适配特定任务，确保生成的簇高度相关且具有上下文感知能力。这一微调过程使模型能够捕捉用户兴趣中那些传统聚类方法可能忽略的细微差别和模式。

通过利用 LLM 骨干网络的先进能力，所提聚类方法不仅提升了媒体表示的准确性，还提高了负采样的质量。这带来更具挑战性和信息量的负样本，最终带来更好的模型性能。基于 LLM 的聚类方法提供了一个更动态、更灵活的框架，能够适应用户兴趣和媒体内容的不断演变，从而在精度和可扩展性上都优于传统聚类技术。

簇的粒度对性能起着关键作用。对于有效的负采样，选择粒度过细的簇是不可取的，因为这有助于降低选择假负样本的风险。

![goobs-fig1](/Users/dazhang/PycharmProject/Papers/3-RecSys/.picture/goobs-fig1.png)

图 1：基于LLM的簇生成



## 4 GOOBS：实时采样框架

GOOBS（Global Out-of-Batch Sampling，全局 batch 外采样）是一个实时的基于簇的负采样框架，旨在集成进处理数十亿训练数据且计算复杂度极低的生产模型。如图 2 所示，GOOBS 框架利用一个存储 OOB 样本张量的 item 池。item 的最大数量是预定义的，每个 item 被分配到一个"槽位"（slot），槽位是一组存储 item 特征的张量。

![goobs-fig2](/Users/dazhang/PycharmProject/Papers/3-RecSys/.picture/goobs-fig2.png)

图 2：GOOBS：实时基于簇的负采样框架

在训练期间，batch 内样本通过更新引擎（update engine），该引擎采用自定义哈希函数确定 item 应存储的槽位。并行地，采样引擎（sampling engine）以 batch 内 item 的簇 ID 为输入，从 item 池中采样相应的 OOB 样本。这些 OOB 样本随后与 batch 内样本合并用于训练。

为确保训练早期阶段有较高的采样命中率，OOB item 池预先加载了来自先前训练数据导出的 item 数据表中的 item 特征。尽管这些数据可能相比最新训练数据略有延迟，但 batch 内训练样本捕获了最新的可用 item，并持续更新 OOB item 池以保持其新鲜度。

GOOBS 基于簇的负采样框架的核心功能内嵌于更新引擎和采样引擎中。如图 3 和算法 1 所示，在更新过程中，item ID 和簇 ID 通过哈希函数确定 item 池中对应的簇分段（cluster segment）。每个簇分段由多个槽位组成以存储 item 特征，每个分段具有相同数量的槽位。

在采样过程中，如图 4 和算法 2 所示，使用 batch 内样本的簇 ID 来识别要采样的目标簇分段。在目标簇分段内，随机选择现有 item 及其特征作为 OOB 样本。

![goobs-fig3](/Users/dazhang/PycharmProject/Papers/3-RecSys/.picture/goobs-fig3.png)

图 3：GOOBS 基于簇的池更新

![goobs-fig4](/Users/dazhang/PycharmProject/Papers/3-RecSys/.picture/goobs-fig4.png)

图 4：GOOBS 基于簇的采样

**算法 1：更新引擎**

输入：第 $i$ 个训练 batch 中的 item ID $x_i = [x_{i1}, x_{i2}, \ldots, x_{iB}]$，batch 大小 $B$，簇大小 $S$

$$
\begin{aligned}
&\textbf{for } j = 1 \textbf{ to } B \textbf{ do} \\
&\quad 1. \text{获取 } x_{ij} \text{ 的簇 ID：} c_{ij} \\
&\quad 2. \text{在 item 池中将 } x_{ij} \text{ 哈希到索引 } c_{ij} \cdot S + x_{ij} \bmod S \\
&\textbf{end for}
\end{aligned}
$$

**算法 2：采样引擎**

输入：第 $i$ 个训练 batch 中的 item ID $x_i = [x_{i1}, x_{i2}, \ldots, x_{iB}]$，batch 大小 $B$，簇大小 $S$

$$
\begin{aligned}
&\textbf{for } j = 1 \textbf{ to } B \textbf{ do} \\
&\quad 1. \text{获取 batch 内样本的簇 ID：} c_{ij} \\
&\quad 2. \text{随机采样 } r_{ij} \in_{rand}(0, S) \\
&\quad 3. \text{从 item 池的索引 } c_{ij} \cdot S + r_{ij} \text{ 处采样 item } N_{ij} \\
&\textbf{end for}
\end{aligned}
$$

## 5 实验

### 5.1 公开数据集

#### 5.1.1 数据集

基于簇的负采样的性能首先在公开数据集 MovieLens-1M 和 Amazon Reviews（子集 Grocery、Electronics、Home）上进行评估，采用与先前推荐工作 [29,30] 类似的完整打乱（full-shuffle）。预处理采用基于时间戳的划分，按时间戳对数据排序，前 80% 用于训练，20% 用于评估。评分标签被转换为二值标签，评分 1-2 为负，3-5 为正。特征包括可直接从数据集获得的用户 ID、item ID 和簇 ID（如类型 ID、类别 ID）。每个用户的历史交互序列被提取作为用户特征。

需要特别指出，评估协议有意避免 k-core 过滤（例如，删除交互少于 5 或 10 次的用户或item），并将目标 item 与整个全局语料库进行精确评估，而非从 100 个随机负样本的小样本中评估。如 Krichene 和 Rendle [16] 所证明，采样评估指标往往与精确的全局排序不一致，且会人为夸大 Hit Rate 的绝对值。通过保留原始数据集的全部稀疏性并执行精确的全局排序，该实验设置更准确地反映了真实世界冷启动部署场景的难度。因此，虽然此处报告的 HR@50 和 HR@100 绝对值低于采用重度 k-core 过滤和采样指标的研究，但各采样策略之间的相对性能提升仍是稳健且无偏的。

表 1：公开数据集上负采样方法的比较。报告 HR@50 和 HR@100。加粗表示最佳性能。括号内为相对基线的提升。

| 数据集 | 指标 | Baseline | DNS | CBNS | ANCE | GOOBS | Cluster GOOBS (In-batch) |
|------|------|----------|-----|------|------|-------|--------------------------|
| Movielens-1M | hr@50 | .2253 | .2298 (+2.0%) | .2331 (+3.5%) | .2380 (+5.6%) | .2346 (+4.2%) | **.2415 (+7.2%)** |
| | hr@100 | .3588 | .3618 (+0.8%) | .3608 (+0.6%) | .3661 (+2.0%) | .3611 (+0.7%) | **.3682 (+2.7%)** |
| Amazon-Grocery | hr@50 | .0254 | .0261 (+2.8%) | .0271 (+6.7%) | .0288 (+13.4%) | .0279 (+10.1%) | **.0301 (+18.5%)** |
| | hr@100 | .0406 | .0419 (+3.2%) | .0432 (+6.4%) | .0451 (+11.1%) | .0440 (+8.3%) | **.0470 (+15.7%)** |
| Amazon-Electronics | hr@50 | .0084 | .0090 (+7.1%) | .0099 (+17.9%) | .0108 (+28.6%) | .0110 (+30.9%) | **.0131 (+55.6%)** |
| | hr@100 | .0154 | .0163 (+5.8%) | .0176 (+14.3%) | .0186 (+20.8%) | .0190 (+23.5%) | **.0201 (+30.2%)** |
| Amazon-Home | hr@50 | .0050 | .0054 (+8.0%) | .0061 (+22.0%) | .0065 (+30.0%) | .0067 (+34.2%) | **.0074 (+47.3%)** |
| | hr@100 | .0082 | .0088 (+7.3%) | .0094 (+14.6%) | .0099 (+20.7%) | .0102 (+23.9%) | **.0118 (+42.3%)** |

#### 5.1.2 模型与负采样策略

使用标准双塔模型作为骨干，在其上实现六种负采样策略，以便与最先进方法进行综合比较：

- **Baseline（batch 内负采样 + LogQ 修正）**：行业标准基线，使用带 LogQ 修正的 batch 内负样本，以减少对高频 item 的过度惩罚。应用假负样本掩码（false-negative mask）抑制已知正样本作为 batch 内负样本出现时的 logits。
- **DNS [31]**：动态负采样（Dynamic Negative Sampling），从当前模型检查点中选择预测相关度分数最高的负样本，即模型当前认为最相关但实际上为负的 item。这是推荐文献中经典的难负样本基线。
- **CBNS [10]**：跨 batch 负采样（Cross-Batch Negative Sampling），缓存最近 mini-batch 的 item 嵌入，并复用作当前 batch 的负样本。与 GOOBS 类似，CBNS 利用 batch 外的 item；但它根据出现的最近性而非语义簇成员来选择负样本。
- **ANCE [25]**：近似最近邻负对比估计（Approximate Nearest Neighbor Negative Contrastive Estimation），在完整 item 语料库上构建全局 ANN 索引，并在训练期间异步刷新。负样本被选择为嵌入空间中每个查询的近似最近邻，使其成为全局在几何上最难的负样本。
- **GOOBS（Global Out-of-Batch Sampling，全局 batch 外采样）**：在 Baseline 基础上扩展，从维护的 item 池中实时均匀采样随机 batch 外（OOB）样本。预训练池加载步骤确保训练一开始就有较高的 OOB 命中率。
- **Cluster GOOBS**：不再使用随机 OOB 样本，而是从与正item相同的语义簇中抽取负样本，随机与簇 OOB 样本的比例对 MovieLens-1M 为 1:15，对 Amazon Reviews 数据集为 1:31。

#### 5.1.3 结果

表 1 报告了全部四个数据集的 HR@50 和 HR@100。Cluster GOOBS 在每个数据集和每个指标上都取得最佳性能，相比 batch 内基线的提升从 +7.2%（Movielens-1M HR@50）到 +55.6%（Amazon-Electronics HR@50）不等，证实基于簇的难负样本在多样化的推荐领域提供了持续且显著的训练信号。

有几个趋势值得强调。首先，仅 GOOBS 本身就已体现出 batch 外采样的好处：随机 OOB 样本相比 batch 内基线带来 +4.2%–+34.2% 的 HR@50 提升，表明训练期间接触更广泛的 item 池能改善泛化，即便没有难负样本选择。其次，用基于簇的难负样本（Cluster GOOBS）替代随机 OOB 样本后，持续优于所有基线，包括需要全局 ANN 索引刷新的几何意义上最难的方法 ANCE。这在更稀疏的 Amazon 数据集上尤为明显：在 Amazon-Electronics 上，Cluster GOOBS 达到 +55.6% 的 HR@50 提升，而 ANCE 为 +28.6%；在 Amazon-Home 上，Cluster GOOBS 为 +47.3%，ANCE 为 +30.0%。第三，在所有数据集上，HR@50 的提升比 HR@100 更明显，表明基于簇的负样本尤其提升了模型将最相关 item 排在召回列表顶部的能力——这是对下游排序阶段最关键的区间。

总体而言，这些结果验证了语义簇成员关系提供了比最近性（CBNS）、基于动态分数选择（DNS）或近似最近邻几何（ANCE）更有效的难负样本信号，同时不需要全局索引，并且与实时服务兼容。

### 5.2 工业数据集

为评估 Cluster GOOBS 在真实世界应用中的效果，它被应用于大规模工业推荐系统，并通过在线 A/B 测试进行评估。

与公开数据集不同，真实世界数据集呈现更大的复杂性和挑战，通常表现出流行度偏差。例如，在所用的工业数据集中，约有 1800 万个 item 符合曝光条件；然而，Top 100 的 item 占据了总曝光的 50%。这表明现有推荐系统对长尾 item 的探索不足，导致用户只与有限数量的 item 交互，并形成强烈的系统反馈回路。

#### 5.2.1 设置

对照组和测试组分别随机选取 3% 的用户。在对照组中，用户由采用 GOOBS 的双塔模型架构服务。在测试组中，用户由采用 Cluster GOOBS 的相同双塔模型架构服务。

#### 5.2.2 簇选择

如第 3.4 节所讨论，在真实数据应用中，利用基于 LLM 的内容理解模型学习媒体表示并导出簇。总共有 300 个簇，其中 98% 的簇拥有 ≥10k 个 item。图 5 总结了不同簇的簇大小分布。

![goobs-fig5](/Users/dazhang/PycharmProject/Papers/3-RecSys/.picture/goobs-fig5.png)

图 5：簇大小分布

#### 5.2.3 结果

模型优化的是点击等用户行为。因此，使用 CTR（click-through-rate，点击率）来衡量在线性能。我们的系统包含多个召回来源。模型改进针对生产中一个主要来源。如表 2 所示，Cluster GOOBS 将 CTR 提升了 53%。注意这是来源层面的增益，而非系统级指标——总体影响取决于该来源的流量占比。同时，它仅带来轻微的训练 QPS（Queries Per Second，每秒查询数）回归 −1.4%（推理 QPS 无回归），展现出该算法强大的可扩展性和效率，使其易于在生产中采用。

进一步分析内容分布以衡量 Cluster GOOBS 的流行度去偏效果。过去 1 天拥有 ≥1K 次曝光的 item 作为目标 item 组群。在测试组中，该比例增加了约 50%，如表 3 所示。Top 100 item 的曝光贡献从 50% 下降到 32%，表明流行度偏差显著改善。

表 2：相似模型架构下不同采样技术的在线 CTR 对比。数字为相对提升。

| 模型 | Source CTR | Training QPS |
|------|-----------|--------------|
| GOOBS（对照组） | 0% | 0% |
| Cluster GOOBS（测试组） | +53% | −1.4% |

表 3：相似模型架构下不同采样技术的流行度去偏对比。数字为相对提升。

| 模型 | Imp. buckets ≥1k | Top 100 items' imp. contrib. |
|------|-----------------|------------------------------|
| GOOBS（对照组） | 0% | 50% |
| Cluster GOOBS（测试组） | +50% | 32% |



## 6 结论

本文介绍了 Cluster GOOBS（Cluster-based Global Out-of-Batch Sampling，基于簇的全局 batch 外采样），作为一种新颖的实时负采样框架，旨在增强大规模推荐系统中双塔模型的训练。通过利用大语言模型（LLM）从同一簇生成难负样本，所提方法解决了传统负采样方法的局限性，后者通常产生无法充分挑战模型的简单负样本。

GOOBS 实现利用 item 池高效地管理和更新 OOB 样本，在处理数十亿训练数据点的同时确保最小的计算复杂度。在更新引擎中集成自定义哈希函数和定向采样引擎，实现了精确有效的负采样，在整个训练过程中保持 item 池的新鲜度和相关性。

总体而言，GOOBS 框架为提升推荐模型性能提供了稳健且可扩展的解决方案，为生产环境提供了无缝集成路径。未来工作可探索进一步的优化和扩展，包括加权基于簇的负采样和基于用户查询的负采样。



## 参考文献

[1] Himan Abdollahpouri, Robin Burke, and Bamshad Mobasher. Controlling popularity bias in learning-to-rank recommendation. In Proceedings of the eleventh ACM conference on recommender systems, pages 42–46, 2017.

[2] Himan Abdollahpouri, Robin Burke, and Bamshad Mobasher. Managing popularity bias in recommender systems with personalized re-ranking. arXiv preprint arXiv:1901.07555, 2019.

[3] Yoshua Bengio and Jean-Sébastien Senécal. Quick training of probabilistic neural nets by importance sampling. In International Workshop on Artificial Intelligence and Statistics, pages 17–24. PMLR, 2003.

[4] Yoshua Bengio and Jean-Sébastien Senécal. Adaptive importance sampling to accelerate training of a neural probabilistic language model. IEEE Transactions on Neural Networks, 19(4):713–722, 2008.

[5] Rocío Cañamares and Pablo Castells. Should i follow the crowd? a probabilistic analysis of the effectiveness of popularity in recommender systems. In The 41st International ACM SIGIR Conference on Research & Development in Information Retrieval, pages 415–424, 2018.

[6] Jiawei Chen, Hande Dong, Xiang Wang, Fuli Feng, Meng Wang, and Xiangnan He. Bias and debias in recommender system: a survey and future directions (2020). arXiv preprint arXiv:2010.03240, 2020.

[7] Jin Chen, Defu Lian, Binbin Jin, Kai Zheng, and Enhong Chen. Learning recommenders for implicit feedback with importance resampling. In Proceedings of the ACM Web Conference 2022, pages 1997–2005, 2022.

[8] Ruey-Cheng Chen, Luke Gallagher, Roi Blanco, and J Shane Culpepper. Efficient cost-aware cascade ranking in multi-stage retrieval. In Proceedings of the 40th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 445–454, 2017.

[9] Paul Covington, Jay Adams, and Emre Sargin. Deep neural networks for youtube recommendations. In Proceedings of the 10th ACM conference on recommender systems, pages 191–198, 2016.

[10] Jingtao Ding, Yuhan Quan, Xiangnan He, Yong Li, and Depeng Jin. Simplify and robustify negative sampling for implicit collaborative filtering. In Advances in Neural Information Processing Systems, volume 33, pages 1094–1105, 2020.

[11] Daniel Gillick, Sayali Kulkarni, Larry Lansing, Alessandro Presta, Jason Baldridge, Eugene Ie, and Diego Garcia-Olano. Learning dense representations for entity retrieval. arXiv preprint arXiv:1909.10506, 2019.

[12] B Hidasi. Session-based recommendations with recurrent neural networks. arXiv preprint arXiv:1511.06939, 2015.

[13] Balázs Hidasi and Alexandros Karatzoglou. Recurrent neural networks with top-k gains for session-based recommendations. In Proceedings of the 27th ACM international conference on information and knowledge management, pages 843–852, 2018.

[14] Jui-Ting Huang, Ashish Sharma, Shuying Sun, Li Xia, David Zhang, Philip Pronin, Janani Padmanabhan, Giuseppe Ottaviano, and Linjun Yang. Embedding-based retrieval in facebook search. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pages 2553–2561, 2020.

[15] Po-Sen Huang, Xiaodong He, Jianfeng Gao, Li Deng, Alex Acero, and Larry Heck. Learning deep structured semantic models for web search using clickthrough data. In Proceedings of the 22nd ACM international conference on Information & Knowledge Management, pages 2333–2338, 2013.

[16] Walid Krichene and Steffen Rendle. On sampled metrics for item recommendation. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pages 1748–1757, 2020.

[17] Chi Liu, Jiangxia Cao, Rui Huang, Kai Zheng, Qiang Luo, Kun Gai, and Guorui Zhou. Kuaiformer: Transformer-based retrieval at kuaishou. arXiv preprint arXiv:2411.10057, 2024.

[18] Shichen Liu, Fei Xiao, Wenwu Ou, and Luo Si. Cascade ranking for operational e-commerce search. In Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pages 1557–1565, 2017.

[19] Marco Morik, Ashudeep Singh, Jessica Hong, and Thorsten Joachims. Controlling fairness and bias in dynamic learning-to-rank. In Proceedings of the 43rd international ACM SIGIR conference on research and development in information retrieval, pages 429–438, 2020.

[20] Harrie Oosterhuis. Computationally efficient optimization of plackett-luce ranking models for relevance and fairness. In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 1023–1032, 2021.

[21] Harald Steck. Item popularity and recommendation accuracy. In Proceedings of the fifth ACM conference on Recommender systems, pages 125–132, 2011.

[22] Jinpeng Wang, Jieming Zhu, and Xiuqiang He. Cross-batch negative sampling for training two-tower recommenders. In Proceedings of the 44th international ACM SIGIR conference on research and development in information retrieval, pages 1632–1636, 2021.

[23] Jun Wang, Lantao Yu, Weinan Zhang, Yu Gong, Yinghui Xu, Benyou Wang, Peng Zhang, and Dell Zhang. Irgan: A minimax game for unifying generative and discriminative information retrieval models. In Proceedings of the 40th International ACM SIGIR conference on Research and Development in Information Retrieval, pages 515–524, 2017.

[24] Tianxin Wei, Fuli Feng, Jiawei Chen, Ziwei Wu, Jinfeng Yi, and Xiangnan He. Model-agnostic counterfactual reasoning for eliminating popularity bias in recommender system. In Proceedings of the 27th ACM SIGKDD conference on knowledge discovery & data mining, pages 1791–1800, 2021.

[25] Lee Xiong, Chenyan Xiong, Ye Li, Kwok-Fung Tang, Jialin Liu, Paul Bennett, Junaid Ahmed, and Arnold Overwijk. Approximate nearest neighbor negative contrastive estimation for dense text retrieval. In International Conference on Learning Representations, 2021.

[26] Jing Yan, Liu Jiang, Jianfei Cui, Zhichen Zhao, Xingyan Bin, Feng Zhang, and Zuotao Liu. Trinity: Syncretizing multi-/long-tail/long-term interests all in one. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pages 6095–6104, 2024.

[27] Ji Yang, Xinyang Yi, Derek Zhiyuan Cheng, Lichan Hong, Yang Li, Simon Xiaoming Wang, Taibai Xu, and Ed H Chi. Mixed negative sampling for learning two-tower neural networks in recommendations. In Companion proceedings of the web conference 2020, pages 441–447, 2020.

[28] Xinyang Yi, Ji Yang, Lichan Hong, Derek Zhiyuan Cheng, Lukasz Heldt, Aditee Kumthekar, Zhe Zhao, Li Wei, and Ed Chi. Sampling-bias-corrected neural modeling for large corpus item recommendations. In Proceedings of the 13th ACM conference on recommender systems, pages 269–277, 2019.

[29] Jiaqi Zhai, Zhaojie Gong, Yueming Wang, Xiao Sun, Zheng Yan, Fu Li, and Xing Liu. Revisiting neural retrieval on accelerators. arXiv preprint arXiv:2306.04039, 2023.

[30] Jiaqi Zhai, Lucy Liao, Xing Liu, Yueming Wang, Rui Li, Xuan Cao, Leon Gao, Zhaojie Gong, Fangda Gu, Michael He, Yinghai Lu, and Yu Shi. Actions speak louder than words: Trillion-parameter sequential transducers for generative recommendations. arXiv preprint arXiv:2402.17152v3, 2024.

[31] Weinan Zhang, Tianqi Chen, Jun Wang, and Yong Yu. Optimizing top-n collaborative filtering via dynamic negative item sampling. In Proceedings of the 36th international ACM SIGIR conference on Research and development in information retrieval, pages 785–788, 2013.

[32] Ziwei Zhu, Yun He, Xing Zhao, Yin Zhang, Jianling Wang, and James Caverlee. Popularity-opportunity bias in collaborative filtering. In Proceedings of the 14th ACM International Conference on Web Search and Data Mining, pages 85–93, 2021.
