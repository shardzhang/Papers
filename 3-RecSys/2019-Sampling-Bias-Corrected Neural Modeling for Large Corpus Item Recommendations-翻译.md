# Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations：大规模语料库item推荐中采样偏差校正的神经建模

> Xinyang Yi, Ji Yang, Lichan Hong, Derek Zhiyuan Cheng, Lukasz Heldt, Aditee Kumthekar, Zhe Zhao, Li Wei, Ed Chi | Google, Inc.

本文介绍了采样偏差校正的神经建模（Sampling-Bias-Corrected Neural Modeling）方法。核心内容：

- 提出一种从流式数据中估计item频率的新算法，无需固定item词汇表，能产生无偏估计并适应item分布变化
- 将估计的item频率通过 logQ 校正纳入批量 softmax 的交叉熵损失，以消除批内负样本带来的采样偏差
- 将该建模范式扩展为 YouTube 大规模神经检索系统，包括顺序训练、索引（MIPS）与服务组件
- 在 Wikipedia 与 YouTube 两个真实世界数据集上验证有效性，并在 YouTube 进行在线 A/B 测试

关键发现：

- 校正后的批量 softmax（correct-sfx）在 Wikipedia 与 YouTube 离线实验中的 Recall@K 均显著优于未校正的 plain-sfx
- 基于批量 softmax 的方法整体优于 mse-gramian 基线
- YouTube 在线实验中 correct-sfx 的参与度提升（+0.37%）显著高于 plain-sfx（+0.20%）
- 流式频率估计算法能快速适应分布变化，且多重哈希在相同参数预算下可进一步降低估计误差

---

## 摘要

许多推荐系统从非常大的语料库中检索和评分item。处理数据稀疏性和幂律item分布的一个常见配方是从item的内容特征中学习item表示。除了许多基于矩阵分解的内容感知系统外，我们还考虑一个使用双塔神经网络的建模框架，其中一个塔（item塔）编码各种item内容特征。训练这种双塔模型的一般配方是优化从小批量中随机采样的item（即批内负样本）计算出的损失函数。然而，批内损失存在采样偏差，可能损害模型性能，特别是在高度倾斜的分布情况下。在本文中，我们提出了一种新颖的从流式数据中估计item频率的算法。通过理论分析和模拟，我们展示了所提出的算法可以在不要求固定item词汇表的情况下工作，并且能够产生无偏估计并适应item分布变化。然后，我们将采样偏差校正的建模方法应用于为 YouTube 推荐构建大规模神经检索系统。该系统已部署用于从数千万视频的语料库中检索个性化建议。我们通过在两个真实世界数据集上的离线实验证明了采样偏差校正的有效性。我们还进行了在线 A/B 测试，以展示神经检索系统为 YouTube 带来了改进的推荐质量。

**CCS概念**：信息系统 $\rightarrow$ 信息检索。

**关键词**：Recommender systems, Information Retrieval, Neural Networks

## 1 引言

推荐系统帮助用户在众多互联网服务中发现感兴趣的内容，包括视频推荐 [12, 18]、应用建议 [9] 和在线广告定向 [38]。在许多情况下，这些系统将数十亿用户连接到来自极大内容语料库的item，通常规模在数百万到数十亿，并受到严格的延迟要求。一个常见的做法是将推荐视为检索-排序问题，并设计一个两阶段系统 [9, 12]。也就是说，一个可扩展的检索模型首先从大型语料库中检索一小部分相关item，然后一个全面的排序模型基于一个或多个目标（如点击或用户评分）对检索到的item进行重新排序。在这项工作中，我们专注于构建一个真实世界的学习检索系统，用于个性化推荐，可扩展到数百万item。

给定{用户, 上下文, item}三元组，构建可扩展检索模型的常见解决方案是：1）分别学习{用户, 上下文}和{item}的查询和item表示；2）在查询和item表示之间使用简单的评分函数（如点积）来获得为查询量身定制的推荐。上下文通常代表具有动态性质的变量，如时间和用户使用的设备。表示学习问题通常在两个方面具有挑战性：1）对于许多工业级应用，item语料库可能非常大；2）从用户反馈中收集的训练数据对于大多数item来说非常稀疏，因此导致模型预测对长尾内容具有较大方差。面对众所周知的冷启动问题，真实世界系统需要适应数据分布变化，以更好地展示新鲜内容。

受 Netflix 奖 [32] 的启发，基于矩阵分解（MF，Matrix Factorization）的建模已被广泛采用，用于学习构建检索系统中的查询和item潜在因子。在 MF 框架下，一系列推荐研究（例如 [21, 34]）解决了从大型语料库中学习的上述挑战。其共同思想是利用查询和item的内容特征。内容特征可以粗略定义为描述item的超越item ID 的各种特征。例如，视频的内容特征可以是从视频帧中提取的视觉和音频特征。基于 MF 的模型通常只能捕获特征的二阶交互，因此在表示各种格式的特征集合时能力有限。

近年来，受深度学习在计算机视觉和自然语言处理中成功的推动，有大量工作将深度神经网络（DNN，Deep Neural Network）应用于推荐。深度表示非常适合在低维嵌入空间中编码复杂的用户状态和item内容特征。在本文中，我们探索了双塔DNN在构建检索模型中的应用。图1提供了双塔模型架构的示意图，其中左塔和右塔分别编码{用户, 上下文}和{item}。双塔DNN是从多类分类神经网络 [19]（一种多层感知机（MLP，Multi-Layer Perceptron）模型）推广而来的，其中图1的右塔被简化为带item嵌入的单层。因此，双塔模型架构能够建模标签具有结构或内容特征的情况。MLP模型通常使用从固定item词汇表中采样的许多负样本进行训练。相比之下，由于item内容特征和用于计算所有item嵌入的共享网络参数，深item塔通常很难高效地采样和训练大量负样本。

**图1：用于学习查询和候选表示的双塔DNN模型。**

我们考虑批量 softmax 优化——在一个随机批次中对所有item计算item概率——作为训练双塔DNN的一般配方。然而，正如我们的实验所示，批量 softmax 存在采样偏差，如果不加任何校正，可能严重限制模型性能。重要性采样和相应的偏差减少已在 MLP 模型 [4, 5] 中研究。受这些工作的启发，我们提出使用估计的item频率来校正批量 softmax 的采样偏差。与MLP模型（输出item词汇表是固定的）不同，我们针对流式数据情况，其词汇表和分布随时间变化。我们提出了一种新颖的算法，通过梯度下降来草图和估计item频率。此外，我们应用偏差校正的建模并将其扩展为构建用于 YouTube 推荐的个性化检索系统。我们还引入了一种顺序训练策略，旨在纳入流式数据，以及系统的索引和服务组件。

本文的主要贡献包括：

- **流式频率估计。** 我们提出了一种新颖的算法，从数据流中估计item频率，能够应对词汇表和分布变化。我们提供了分析结果来展示估计的方差和偏差。我们还提供了模拟来证明我们的方法在捕获数据动态方面的有效性。

- **建模范式。** 我们提供了一个用于构建大规模检索系统的通用建模范式。特别地，我们将估计的item频率纳入批量 softmax 的交叉熵损失中，以减少批内item的采样偏差。

- **YouTube 推荐。** 我们描述了如何应用建模范式为 YouTube 推荐构建大规模检索系统。我们介绍了端到端系统，包括训练、索引和服务组件。

- **离线和在线实验。** 我们在两个真实世界数据集上进行了离线实验，并展示了采样偏差校正的有效性。我们还展示了为 YouTube 构建的检索系统在在线实验中的参与度指标得到改善。

## 2 相关工作

在本节中，我们概述相关工作，并突出与我们的贡献之间的联系。

### 2.1 内容感知和神经推荐器

利用用户和item的内容特征对于改善泛化和缓解冷启动问题至关重要。有一系列研究专注于在经典矩阵分解框架中纳入内容特征 [23]。例如，广义矩阵分解模型如 SVDFeature [8] 和因子分解机 [33] 可以应用于纳入item内容特征。这些模型能够捕获最多双线性（即二阶）的特征交互。近年来，深度神经网络（DNN）已被证明在提高推荐精度方面有效。由于高度非线性的特性，与传统因子分解方法相比，DNN 为捕获复杂的特征交互提供了更大的容量 [6, 35]。

He 等人 [21] 直接应用协同过滤（CF，Collaborative Filtering）的直觉，并提出了一个神经CF（NCF，Neural Collaborative Filtering）架构来建模用户-item交互。在 NCF 框架中，用户和item嵌入被拼接并传递给多层神经网络，得到最终预测。我们的工作与 NCF 在两个方面的不同：1）我们利用双塔神经网络建模用户-item交互，以便推理可以对大型item语料库进行，时间复杂度为亚线性；2）NCF 的学习依赖逐点损失（如平方或对数损失），而我们引入了多类 softmax 损失并显式建模item频率。

在另一条工作线上，深度和循环模型（如 LSTM，Long Short-Term Memory，长短期记忆网络）被应用于在推荐中纳入时间信息和历史事件，例如 [12, 14]。除了学习独立的用户和item表示外，还有另一组工作专注于为学习排序系统设计神经网络。值得注意的是，多任务学习一直是优化复杂推荐器多个目标的核心技术 [27, 28]。Cheng 等人 [9] 引入了一个 wide-n-deep 框架，联合训练宽线性模型和深度神经网络。

### 2.2 极多类分类

Softmax 是设计预测高达数百万标签的大输出空间模型时最常用的函数之一。大量研究专注于训练大量类别的 softmax 分类模型，范围从语言任务 [5, 29] 到推荐器 [12]。当类别数量非常大时，一个广泛使用的加速训练的技术是采样类别的子集。Bengio 等人 [5] 表明一个好的采样分布应该适应模型的输出分布。为避免计算采样分布的复杂性，许多真实世界模型应用简单分布如 unigram 或 uniform 作为代理。最近，Blanc 等人 [7] 设计了一种高效的自适应基于核的采样方法。

尽管采样 softmax 在各个领域取得了成功，但它不适用于标签具有内容特征的情况。这种情况下的自适应采样仍然是一个未解决的问题。各种工作已经表明，基于树的标签结构（例如层次 softmax [30]）对于构建大规模分类模型很有用，同时显著减少推理时间。这些方法通常需要基于某些类别属性预定义树结构。因此，它们不适合纳入各种各样的输入特征。

### 2.3 双塔模型

构建具有两个塔的神经网络最近已成为几个自然语言任务中的流行方法，包括建模句子相似性 [31]、回复建议 [24] 和基于文本的信息检索 [17, 37]。我们的工作为这一研究线做出了贡献，特别是证明了双塔模型在构建大规模推荐器中的有效性。与上述文献中的许多语言任务相比，值得注意的是，我们关注的是大得多的语料库规模问题，这在我们的目标应用（如 YouTube）中很常见。通过在线实验，我们发现显式建模item频率对于在这种设置下提高检索精度至关重要。然而，这个问题在现有工作中没有得到很好的解决。

## 3 建模范式

我们考虑推荐问题的一个常见设置，其中我们有一组查询和item。查询和item分别用特征向量 $\{x_i\}_{i=1}^{N}$ 和 $\{y_j\}_{j=1}^{M}$ 表示。这里 $x_i \in X$、$y_j \in Y$ 都是各种各样的特征的混合（例如稀疏ID和稠密特征），并且可能处于非常高的维度空间中。目标是给定查询检索item的一个子集。在个性化场景中，我们假设用户和上下文完全包含在 $x_i$ 中。请注意，我们从有限数量的查询和item开始解释直觉。我们的建模范式在没有这种假设的情况下也能工作。

我们的目标是用两个参数化嵌入函数 $u: X \times \mathbb{R}^d \to \mathbb{R}^k$、$v: Y \times \mathbb{R}^d \to \mathbb{R}^k$ 构建模型，将模型参数 $\theta \in \mathbb{R}^d$ 以及查询和候选的特征映射到 $k$ 维嵌入空间。我们关注 $u$、$v$ 由图1所示的两个深度神经网络表示的情况。模型的输出是两个嵌入的内积，即：

$$
s(x, y) = \langle u(x, \theta), v(y, \theta) \rangle
$$

目标是从 T 个样本的训练数据集学习模型参数 $\theta$，记为：

$$
\mathcal{T} := \{(x_i, y_i, r_i)\}_{i=1}^{T}
$$

其中 $(x_i, y_i)$ 表示查询 $x_i$ 和item $y_i$ 的对，$r_i \in \mathbb{R}$ 是每对的关联奖励。

直观上，检索问题可以被视为带连续奖励的多类分类问题。在分类任务中，每个标签同等重要，所有正样本对的 $r_i = 1$。在推荐器中，$r_i$ 可以扩展为捕获用户对某个候选的各种参与度。例如，在新闻推荐中，$r_i$ 可以是用户在某篇文章上花费的时间。给定查询 $x$，从 M 个item $\{y_j\}_{j=1}^{M}$ 中选择候选 $y$ 的概率分布的常见选择基于 softmax 函数，即：

$$
P(y|x; \theta) = \frac{e^{s(x,y)}}{\sum_{j \in [M]} e^{s(x,y_j)}} \qquad (1)
$$

通过进一步纳入奖励 $r_i$，我们考虑以下加权对数似然作为损失函数：

$$
L_T(\theta) := -\frac{1}{T} \sum_{i \in [T]} r_i \cdot \log(P(y_i|x_i; \theta)) \qquad (2)
$$

当 M 非常大时，将所有候选示例纳入计算配分函数（即公式 (1) 中的分母）是不可行的。一个常见的思想是使用item的一个子集来构建配分函数。我们关注处理流式数据。因此，与训练MLP模型（负样本从固定语料库中采样）不同，我们只考虑使用同一批次中所有查询的批内item [22] 作为负样本。精确地说，给定 B 对的小批量 $\{(x_i, y_i, r_i)\}_{i=1}^{B}$，对于每个 $i \in [B]$，批量 softmax 为：

$$
P_B(y_i|x_i; \theta) = \frac{e^{s(x_i,y_i)}}{\sum_{j \in [B]} e^{s(x_i,y_j)}} \qquad (3)
$$

在我们的目标应用中，批内item通常从幂律分布中采样。因此，公式 (3) 引入了对完整 softmax 的很大偏差：热门item由于在批次中被包含的概率高而作为负样本被过度惩罚。受采样 softmax 模型 [5] 中使用的 logQ 校正的启发，我们通过以下方程校正每个 logit $s(x_i, y_j)$：

$$
s^c(x_i, y_j) = s(x_i, y_j) - \log(p_j)
$$

这里 $p_j$ 表示item $j$ 在一个随机批次中的采样概率。我们将 $p_j$ 的估计推迟到下一节。

有了校正，我们有：

$$
P_B^c(y_i|x_i; \theta) = \frac{e^{s^c(x_i,y_i)}}{e^{s^c(x_i,y_i)} + \sum_{j \in [B], j \neq i} e^{s^c(x_i,y_j)}}
$$

然后将上述项代入公式 (2) 得到：

$$
L_B(\theta) := -\frac{1}{B} \sum_{i \in [B]} r_i \cdot \log(P_B^c(y_i|x_i; \theta)) \qquad (4)
$$

这就是批量损失函数。使用学习率 $\gamma$ 运行 SGD（Stochastic Gradient Descent，随机梯度下降）得到模型参数更新：

$$
\theta \leftarrow \theta - \gamma \cdot \nabla L_B(\theta) \qquad (5)
$$

注意 $L_B$ 不要求固定的查询或候选集合。因此，公式 (5) 可以应用于分布随时间变化的流式训练数据。关于我们提出的方法的完整描述，见算法1。

**算法1：训练算法**

1. 输入：两个参数化嵌入函数 $u(\cdot, \theta)$、$v(\cdot, \theta)$，其中每个函数都通过神经网络将输入特征映射到嵌入空间。学习率 $\gamma$（固定或自适应）。
2. 重复
3. 从流中采样或接收一批训练数据 $\{(x_i, y_i, r_i)\}_{i=1}^{B}$。
4. 从算法2获取每个 $y_i$ 的估计采样概率 $p_i$。
5. 根据公式 (4) 构建损失 $L_B(\theta)$。
6. $\theta \leftarrow \theta - \gamma \cdot \nabla L_B(\theta)$。
7. 直到满足停止条件

**归一化和温度。** 根据经验，我们发现添加嵌入归一化，即 $u(x, \theta) \leftarrow u(x, \theta)/\lVert u(x, \theta) \rVert_2$、$v(y, \theta) \leftarrow v(y, \theta)/\lVert v(y, \theta) \rVert_2$，可以提高模型的可训练性，从而带来更好的检索质量。此外，为每个 logit 添加温度 $\tau$ 来锐化预测，即：

$$
s(x, y) = \langle u(x, \theta), v(y, \theta) \rangle / \tau
$$

在实践中，$\tau$ 是一个超参数，用于最大化召回率或精确率等检索指标。

## 4 流式频率估计

在本节中，我们详细阐述算法1中使用的流式频率估计。

考虑随机批次的流，其中每个批次包含一组item。问题是估计每个item $y$ 在一个批次中被命中的概率。一个关键的设计标准是有一个完全分布式的估计来支持存在多个训练任务（即worker）时的分布式训练。

在单机或分布式训练中，一个唯一的全局步骤（表示训练器消费的数据批次数）与每个采样批次相关联。在分布式设置中，全局步骤通常通过参数服务器在多个worker之间同步。我们可以利用全局步骤，将item频率 $p$ 的估计转化为 $\delta$ 的估计，$\delta$ 表示两次连续命中item之间的平均步数。例如，如果一个item每50步被采样一次，那么我们有 $p = 0.02$。使用全局步骤提供两个优势：1）多个worker通过读取和修改全局步骤在频率估计中隐式同步；2）$\delta$ 的估计可以通过简单的移动平均更新实现，该更新能适应分布变化。

由于使用固定item词汇表不实际，我们应用哈希数组来记录流式ID的采样信息。注意引入哈希可能导致潜在的哈希冲突。我们将在本节末尾重新讨论这个问题并提出改进算法。如算法2所示，我们保留两个大小为 $H$ 的数组 $A$ 和 $B$。假设 $h$ 是一个将任何item映射到 $[H]$ 中整数的哈希函数，映射可以基于ID或任何其他缩略特征值。然后对于给定的item $y$，$A[h(y)]$ 记录 $y$ 最近被采样的步数，$B[h(y)]$ 包含 $y$ 的估计 $\delta$。我们使用数组 $A$ 来帮助更新数组 $B$。一旦item $y$ 在步骤 $t$ 出现，我们按如下方式更新数组 $B$：

$$
B[h(y)] \leftarrow (1-\alpha) \cdot B[h(y)] + \alpha \cdot (t - A[h(y)]) \qquad (6)
$$

在 $B$ 更新后，我们将 $t$ 赋给 $A[h(y)]$。

对于每个item，假设两次连续命中之间的步数遵循由随机变量 $\Delta$ 表示的分布，均值为 $\delta = E(\Delta)$。这里我们的目标是从样本流中估计 $\delta$。每当一个item在步骤 $t$ 的批次中出现时，$t - A[h(y)]$ 是 $\Delta$ 的一个新样本。相应地，上述更新可以理解为以固定学习率 $\alpha$ 运行 SGD 来学习该随机变量的均值。形式上，在没有冲突的情况下，下一个结果显示了这种在线估计的偏差和方差。

**命题4.1。** 假设 $\{\Delta_1, \Delta_2, ..., \Delta_t\}$ 是随机变量 $\Delta$ 的独立同分布（i.i.d.）样本序列。设 $\delta = E[\Delta]$。考虑如下在线估计，其中对于 $i \in [t]$ 和 $\alpha \in (0, 1)$：

$$
\hat{\delta}_i = (1-\alpha) \cdot \hat{\delta}_{i-1} + \alpha \cdot \Delta_i
$$

估计的偏差由下式给出：

$$
E(\hat{\delta}_t) - \delta = (1-\alpha)^t \hat{\delta}_0 - (1-\alpha)^{t-1} \delta \qquad (7)
$$

对于方差，我们有：

$$
E[(\hat{\delta}_t - E[\hat{\delta}_t])^2] \leq (1-\alpha)^{2t}(\hat{\delta}_0 - \delta)^2 + \alpha \cdot E[(\Delta_1 - \delta)^2] \qquad (8)
$$

**证明。** 通过取期望，我们有 $E[\hat{\delta}_i] = (1-\alpha) \cdot E(\hat{\delta}_{i-1}) + \alpha \cdot \delta$。通过对 $t$ 归纳，得到 (7)。对于方差，我们有 $E[(\hat{\delta}_t - E[\hat{\delta}_t])^2] = E[(\hat{\delta}_t - \delta)^2] + 2 \cdot E[(\hat{\delta}_t - \delta)(\delta - E[\hat{\delta}_t])] + (E[\hat{\delta}_t] - \delta)^2 = E[(\hat{\delta}_t - \delta)^2] - (E[\hat{\delta}_t] - \delta)^2 \leq E[(\hat{\delta}_t - \delta)^2]$。对于最后一项，我们有 $E[(\hat{\delta}_i - \delta)^2] = (1-\alpha)^2 E[(\hat{\delta}_{i-1} - \delta)^2] + \alpha^2 E[(\Delta_t - \delta)^2] + 2(1-\alpha)\alpha E[(\hat{\delta}_{i-1} - \delta)(\Delta_i - \delta)]$。由于 $\hat{\delta}_{i-1}$ 和 $\Delta_i$ 是独立的，最后一项为零。然后通过对 $i$ 归纳，我们有 $E[(\hat{\delta}_t - \delta)^2] = (1-\alpha)^{2t}(\hat{\delta}_0 - \delta)^2 + \alpha^2 \cdot \frac{1-(1-\alpha)^{2t-2}}{1-(1-\alpha)^2} \cdot E[(\Delta_1 - \delta)^2] \leq (1-\alpha)^{2t}(\hat{\delta}_0 - \delta)^2 + \alpha E[(\Delta_1 - \delta)^2]$。$\square$

公式 (7) 表明偏差 $|E[\hat{\delta}_t] - \delta| \to 0$ 当 $t \to \infty$。此外，理想的初始化 $\hat{\delta}_0 = \delta/(1-\alpha)$ 可以在每一步产生无偏估计。公式 (8) 给出了估计方差的上界。学习率 $\alpha$ 在两个层面上影响方差：1）更高的学习率导致依赖于初始化误差的第一项更快地下降；2）更低的学习率减少依赖于 $\Delta$ 的方差且随时间不递减的第二项。

为了获得 $y$ 的估计采样概率 $\hat{p}$，我们可以简单地执行 $\hat{p} = 1/B[h(y)]$。

**算法2：流式频率估计**

1. 输入：学习率 $\alpha$。大小为 $H$ 的数组 $A$ 和 $B$。输出空间为 $[H]$ 的哈希函数 $h$。
2. （训练）
3. 对于步骤 $t = 1, 2, ...$：
4. 采样一个item批次 $\mathcal{B}$。对于每个 $y \in \mathcal{B}$，执行
5. $B[h(y)] \leftarrow (1-\alpha) \cdot B[h(y)] + \alpha \cdot (t - A[h(y)])$。
6. $A[h(y)] \leftarrow t$。
7. 直到满足停止条件
8. （推理）
9. 对于任何item $y$，采样概率 $\hat{p} = 1/B[h(y)]$。

**分布式更新。** 我们考虑 [13] 中提出的分布式训练框架，其中模型参数分布在一组称为参数服务器的服务器上，多个worker独立处理训练数据并与参数服务器通信以获取和更新模型参数。算法2可以扩展到这种设置。数组 $A$、$B$ 和全局步骤参数分布在参数服务器上。每个worker通过采样item小批量来执行第4行。详细来说，在步骤 $t$，从参数服务器获取 $A[h(y)]$、$B[h(y)]$。然后按所示更新 $A[h(y)]$、$B[h(y)]$ 并发回。因此，算法2中的更新可以与神经网络的异步随机梯度下降训练一起执行。

**多重哈希。** 受 count-min sketch [11] 中类似思想的启发，我们将算法2扩展为利用多个哈希函数来缓解由于冲突导致的item频率过度估计。改进后的估计在算法3中给出。更新每个数组 $A_i$、$B_i$ 遵循算法2中的相应步骤。$B$ 中的每个桶都可能是真实步数间隔的低估，因为它可能代表多个item的并集。因此，对于推理，我们取 $m$ 个估计中的最大值，表示两次连续命中之间的步数。

**算法3：多数组改进的频率估计**

1. 输入：学习率 $\alpha$。一组 $m$ 个大小为 $H$ 的数组 $\{A_i\}_{i=1}^{m}$ 和 $\{B_i\}_{i=1}^{m}$。一组输出空间为 $[H]$ 的独立哈希函数 $\{h_i\}_{i=1}^{m}$。
2. （训练）
3. 对于步骤 $t = 1, 2, ...$：
4. 采样一个item批次 $\mathcal{B}$。对于每个 $y \in \mathcal{B}$ 和 $i \in [m]$，执行
5. $B_i[h(y)] \leftarrow (1-\alpha) \cdot B_i[h(y)] + \alpha \cdot (t - A_i[h(y)])$。
6. $A_i[h(y)] \leftarrow t$。
7. 直到满足停止条件
8. （推理）
9. 对于任何item $y$，估计概率 $\hat{p} = 1/\max_i\{B_i[h(y)]\}$。

## 5 YouTube 神经检索系统

我们应用所提出的建模范式并将其扩展为在 YouTube 的某个特定产品上构建大规模神经检索系统。该产品根据用户正在观看的视频（称为种子视频）生成视频推荐。推荐系统由两个阶段组成：提名（又称检索）和排序。在提名阶段，多个提名人根据用户和种子视频的约束各自生成数百个视频推荐。这些视频随后由完整的神经网络排序模型打分和重新排序。在本节中，我们专注于在检索阶段构建一个额外的提名人，特别是从数据、模型架构、训练和服务这些角度。

### 5.1 建模概述

我们构建的 YouTube 神经检索模型由查询和候选网络组成。图2展示了通用的模型架构。在任何时候，用户正在观看的视频（即种子视频）都提供了关于用户当前兴趣的强信号。因此，我们利用大量的种子视频特征以及用户的观看历史。候选塔被构建为从候选视频特征中学习。

**图2：YouTube神经检索模型的示意图。**

**训练标签。** 视频点击被用作正标签。此外，对于每次点击，我们构建一个奖励 $r_i$ 来反映用户对视频的不同参与度。例如，观看时间短（little watch time）的被点击视频 $r_i = 0$。另一方面，$r_i = 1$ 表示整个视频被观看。该奖励如公式 (4) 所示用作样本权重。

**视频特征。** 我们使用的视频特征包括类别特征和稠密特征。类别特征的示例包括视频ID（Video Id）和频道ID（Channel Id）。对于这些实体中的每一个，我们创建嵌入层将每个类别特征映射到稠密向量。通常我们处理两种类别特征。有些特征（例如视频ID）每个视频严格有一个类别值，因此我们有一个嵌入向量来表示它。或者，一个特征（例如视频主题）可能是类别值的稀疏向量，表示该特征的最终嵌入将是稀疏向量中每个值的嵌入的加权和。为了处理词汇表外的实体，我们将它们随机分配到一组固定的哈希桶中，并为每个桶学习一个嵌入。哈希桶对于模型捕获 YouTube 中可用的新实体很重要，特别是当使用5.2节中描述的顺序训练时。

**用户特征。** 除了种子视频，我们使用用户的观看历史来捕获用户的兴趣。一个例子是用户最近观看的 $k$ 个视频ID的序列。我们将观看历史视为词袋（BOW，Bag of Words），并用视频ID嵌入的平均值表示它。在查询塔中，用户和种子视频特征在输入层融合，然后通过前馈神经网络传递。

对于相同类型的ID，嵌入在相关特征之间共享。例如，同一组视频ID嵌入用于种子、候选和用户过去的观看。我们尝试了非共享嵌入，但没有观察到显著的模型质量提升。

### 5.2 顺序训练

我们的模型在 TensorFlow [1] 中实现，并在多个worker和参数服务器上使用分布式梯度下降进行训练。在 YouTube，每天都会生成新的训练数据，训练数据集相应地按天组织。模型训练以下列方式利用这种顺序结构。训练器按顺序消费数据，从最古老的训练样本到最新的训练样本。一旦训练器赶上最新的训练数据日，它就等待下一天训练数据的到来。这样，模型能够跟上最新的数据分布变化。训练数据本质上是以流式方式被训练器消费的。我们应用算法2（如果使用多重哈希则应用算法3）来估计item频率。公式 (6) 的在线更新使模型能够适应新的频率分布。

### 5.3 索引与模型服务

检索系统中的索引流水线定期为在线服务创建 TensorFlow SavedModel。索引流水线分三个阶段构建：候选样本生成、嵌入推理和嵌入索引，如图3所示。在第一阶段，基于某些标准从 YouTube 语料库中选择一组视频。它们的特征被获取并添加到候选样本中。在第二阶段，应用图2的右塔从候选样本计算嵌入。在第三阶段，我们基于树和量化哈希技术（例如 [2, 10, 25]）训练一个基于 TensorFlow 的嵌入索引模型，用于近似最大内积搜索（MIPS，Maximum Inner Product Search）。具体来说，高维嵌入的紧凑表示通过量化 [20] 和粗量化器与乘积量化器的端到端学习 [36] 构建。我们略过细节，因为它们不是本文的重点。最后，通过拼接图2的查询塔和索引模型来创建用于服务的 SavedModel。

**图3：YouTube神经检索系统索引流水线的概览。**

## 6 实验

在本节中，我们展示实验结果以证明所提出的item频率估计和建模框架的有效性。

### 6.1 频率估计模拟

为了评估算法2和3的有效性，我们从模拟研究开始，首先将每个提出的算法应用于拟合固定的item分布，然后在某一步之后改变分布。更准确地说，在我们的设置中，我们使用一组固定的 M 个item，每个item根据概率 $q_i \propto i^2$（对于 $i \in [M]$，$\sum_i q_i = 1$）独立采样。我们在输入流上进行模拟，每一步采样一个item批次 $\mathcal{B}$。这里 $\mathcal{B}$ 中的每个item从 $q_i$ 中无放回采样。因此，我们旨在拟合的item采样概率是 $p_i = |\mathcal{B}| \times q_i$。我们在前 $t$ 步保持采样分布静态。然后在剩余步骤中将其切换为 $q_i \propto (M-1-i)^2$。为了评估估计精度，我们使用估计概率集 $\{\hat{p}_i\}_{i \in [M]}$ 和 $\{p_i\}_{i \in [M]}$ 之间的重新缩放 $L_1$ 距离，精确地说，$\frac{1}{2|\mathcal{B}|} \sum_i |\hat{p}_i - p_i|$，作为估计误差。它也可以被理解为估计 $\{\hat{q}_i\}_{i \in [M]}$ 和 $\{q_i\}_{i \in [M]}$ 之间的总变差。

具体来说，我们报告：1）学习率 $\alpha$ 的影响，以及 2）多重哈希的影响。

**学习率 $\alpha$ 的影响。** 我们设置 $M = 1000$，$B = 128$，并对 $A$ 和 $B$ 都使用数组大小 $H = 5000$。此外，我们将数组 $A$ 初始化为全零，$B$ 的每个条目初始化为100。分布在第 $t = 10000$ 步切换。我们使用一个哈希函数并运行算法2。图4显示了在给定一组学习率 $\alpha$ 下，随全局步数变化的估计误差。我们观察到所有三条曲线都收敛到一个误差水平，该水平来自哈希冲突和估计方差。学习率越高，算法对分布变化的适应性越强，但如命题4.1所示，最终方差越高。

**图4：不同学习率 $\alpha$ 下的频率估计误差。item分布在步骤10000处切换。**

**多重哈希的影响。** 对于第二个模拟，我们运行算法3并尝试各种数量的哈希函数 $m$。图5显示了 $m = 1, 2, 4$ 的估计误差曲线。我们为 $A$、$B$ 选择不同的数组大小 $H$，以便在这三种设置中保持哈希桶总数相同。图5证明了即使参数数量相同，使用多个哈希函数也可以减少估计误差。

**图5：不同哈希函数数量 $m$ 下的频率估计误差。**

### 6.2 Wikipedia 页面检索

在本节中，我们在 Wikipedia 数据集 [16] 上进行页面检索实验，以展示采样偏差校正批量损失（公式 (4)）的有效性。

**数据集。** 我们考虑预测 Wikipedia 页面之间站内链接的任务。对于给定的一对源页面和目标页面 $(x, y)$，如果从 $x$ 到 $y$ 存在链接，则标签为1，否则为0。每个页面由一组内容特征表示，包括页面URL、页面标题中n-gram集合的词袋表示，以及页面类别的词袋表示。我们在英文图上进行实验，该图由5.3M页面、430M链接、510K标题n-gram和403.4K唯一类别组成。

**模型。** 我们将链接预测视为一个检索问题：给定一个源页面，任务是从页面语料库中检索目标页面。因此，我们训练一个双塔神经网络，其中左塔和右塔分别映射源页面和目标页面的特征。输入特征嵌入在两个塔之间共享。每个塔有两个全连接 ReLU 层，维度为 [512, 128]。

**基线。** 我们将所提出的采样偏差校正批量 softmax（correct-sfx）与没有任何校正的批量 softmax（plain-sfx）进行比较，如公式 (3) 所示，以证明偏差校正的有效性。此外，我们考虑在推荐中建模隐式反馈时广泛采用的均方误差损失。该损失是观测对上 MSE 与正则项的组合，正则项将所有未见对推向通常选择为0的常数先验。在第3节介绍的框架下，该损失为：

$$
L = \frac{1}{|\Omega|} \sum_{(x_i, y_i) \in \Omega} (\langle u(x_i), v(y_i) \rangle - r_i)^2 + \lambda \cdot \frac{1}{|\Omega^c|} \sum_{(x_i, y_i) \in \Omega^c} \langle u(x_i), v(y_i) \rangle^2
$$

其中 $\Omega$ 和 $\Omega^c$ 分别表示观测对的集合及其补集，$\lambda$ 是正超参数。在矩阵分解中，这种损失通常通过交替最小二乘 [23] 或坐标下降方法 [3] 训练，通过将正则项写成两个格拉姆矩阵的矩阵内积，这对于线性嵌入可以高效计算。最近，Krichene 等人 [26] 通过 SGD 估计将格拉姆矩阵计算扩展到非线性场景。我们将这种方法称为 mse-gramian。

**训练和评估。** 对于所有方法，我们使用批次大小1024，模型使用 Adagrad [15] 和学习率0.01训练10M步。对于频率估计，我们使用 $m = 1$，$H = 40M$ 和 $\alpha = 0.01$。我们留出10%的链接用于评估。我们通过 Recall@K 评估模型性能，这本质上是将真实标签包含在针对完整页面语料库的 top k 候选中的平均概率。mse-gramian 中的参数 $\lambda$ 通过线搜索调整，我们在这里报告最佳结果。我们发现归一化输出层总是能提高模型性能和训练稳定性。我们只展示带归一化的结果。我们对 plain-sfx 和 correct-sfx 尝试了多个温度值 $\tau$。

结果汇总在表1中。对于每个温度值，correct-sfx 以很大的幅度优于相应的 plain-sfx。有趣的是看到温度对性能的影响，表明在应用归一化时，仔细调整这个参数是必要的。我们还注意到基于批量 softmax 的方法比 mse-gramian 表现更好。

**表1：从Wikipedia 5.3M页面语料库中检索目标页面的Recall@K。**

| 方法 | Recall@10 | Recall@50 | Recall@100 | Recall@300 |
|------|-----------|-----------|------------|------------|
| mse-gramian [26] | 0.0432 | 0.1326 | 0.2027 | 0.3530 |
| plain-sfx $\tau = 0.05$ | 0.0579 | 0.2259 | 0.3573 | 0.5931 |
| plain-sfx $\tau = 0.07$ | 0.0643 | 0.2423 | 0.3746 | 0.5991 |
| plain-sfx $\tau = 0.14$ | 0.0614 | 0.2216 | 0.3341 | 0.5200 |
| correct-sfx $\tau = 0.05$ | 0.0987 | 0.3202 | 0.4835 | 0.7413 |
| correct-sfx $\tau = 0.07$ | 0.1065 | 0.3079 | 0.4664 | 0.7234 |
| correct-sfx $\tau = 0.14$ | 0.0807 | 0.2411 | 0.3519 | 0.5529 |

### 6.3 YouTube 实验

我们基于第5节介绍的神经检索系统在 YouTube 上进行离线和在线实验。我们使用的 YouTube 训练数据包括每天数十亿次点击视频，跨越多天。

**设置。** 回想一下我们使用的模型结构如图2所示。如前所述，如果查询塔和候选塔都可获得输入特征嵌入，则它们之间共享。我们对两个塔都使用三层DNN，隐藏层大小为 [1024, 512, 128]。我们使用 Adagrad、学习率0.2和批次大小8192训练模型。对于频率估计，我们设置 $H = 50M$，$m = 1$，$\alpha = 0.01$。回想一下我们应用5.2节中介绍的顺序训练。对于本节中的实验，从 YouTube 语料库中选择的约1000万视频的索引每隔几小时定期构建。索引语料库会随时间变化，例如由于新视频的上传。但它通常覆盖超过90%的训练样本。

**离线实验。** 我们为所有被点击的视频分配 $r_i = 1$，并通过检索被点击视频时的召回率评估模型性能。我们简化离线实验的奖励函数，因为为连续奖励定义合适的离线指标并不明显。为了纳入顺序训练，我们在第 $d_0$ 天之后评估模型性能，此时训练器完成追赶阶段（设置为15天）并开始等待新数据。对于 $d_0$ 之后的每个新的一天，我们留出10%的数据用于评估。为了考虑每周模式，我们报告7天的平均离线结果，即从 $d_0 + 1$ 到 $d_0 + 7$。结果在表2中呈现。我们再次看到使用批量 softmax 相比 mse-gramian 有显著改进。此外，在具有不同 $\tau$ 的设置中，item频率校正的 softmax 显著优于普通 softmax。

**表2：从YouTube的1000万视频语料库中检索被点击视频的Recall@K。训练中所有被点击视频的奖励 $r_i$ 设为1。**

| 方法 | Recall@5 | Recall@10 | Recall@30 | Recall@50 |
|------|----------|-----------|-----------|-----------|
| mse-gramian [26] | 0.0554 | 0.0768 | 0.1149 | 0.1338 |
| plain-sfx $\tau = 0.1$ | 0.1916 | 0.2512 | 0.3658 | 0.4246 |
| plain-sfx $\tau = 0.025$ | 0.1958 | 0.2609 | 0.3839 | 0.4456 |
| plain-sfx $\tau = 0.05$ | 0.2069 | 0.2728 | 0.3964 | 0.4586 |
| correct-sfx $\tau = 0.1$ | 0.1957 | 0.2689 | 0.4125 | 0.4796 |
| correct-sfx $\tau = 0.025$ | 0.2014 | 0.2790 | 0.4314 | 0.5082 |
| correct-sfx $\tau = 0.05$ | 0.2150 | 0.2960 | 0.4537 | 0.5322 |

**在线实验。** 我们还在 YouTube 的 A/B 测试框架中进行在线实验。对于对照组中的用户，视频由生产系统推荐。对于处理组，用户看到的是将图2所示的神经检索系统的候选添加到提名阶段后的推荐。由于推荐用户喜欢点击的视频并不理想，对于在线实验，我们以反映用户与所点击视频真实参与度的方式训练模型。我们报告与此标签一致的参与度指标。结果如表3所示。可以看出，添加神经检索系统比之前的生产系统有显著的改进。此外，使用 correct-sfx 的模型比使用 plain-sfx 的基线显著更好，证明了采样偏差校正的有效性。

**表3：YouTube在线实验结果。被点击视频的奖励 $r_i$ 设为与所报告的参与度指标相关的某种用户反馈。**

| 方法 | 参与度指标改进 |
|------|----------------|
| plain-sfx $\tau = 0.05$ | +0.20% |
| correct-sfx $\tau = 0.05$ | +0.37% |

## 7 结论

在本文中，我们提出了一个通用的建模框架，用于为工业级应用构建大规模内容感知检索模型。我们提出了一种新颖的估计item频率的算法。理论分析和模拟证明了其正确性和有效性。我们将所提出的建模框架应用于为 YouTube 推荐构建神经检索系统。特别是，为了捕获 YouTube 的数据动态，我们提出了一种顺序训练策略，流式频率估计算法可以轻松集成到其中。在 Wikipedia 链接预测和 YouTube 视频检索上的离线实验表明，使用采样偏差校正有显著改进。YouTube 的在线实验也表明，用户对我们神经检索系统检索到的候选的参与度有所提升。

## 参考文献

[1] Martín Abadi, Ashish Agarwal, Paul Barham, Eugene Brevdo, Zhifeng Chen, Craig Citro, Greg S. Corrado, Andy Davis, Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Ian Goodfellow, Andrew Harp, Geoffrey Irving, Michael Isard, Yangqing Jia, Rafal Jozefowicz, Lukasz Kaiser, Manjunath Kudlur, Josh Levenberg, Dandelion Mané, Rajat Monga, Sherry Moore, Derek Murray, Chris Olah, Mike Schuster, Jonathon Shlens, Benoit Steiner, Ilya Sutskever, Kunal Talwar, Paul Tucker, Vincent Vanhoucke, Vijay Vasudevan, Fernanda Viégas, Oriol Vinyals, Pete Warden, Martin Wattenberg, Martin Wicke, Yuan Yu, and Xiaoqiang Zheng. 2015. TensorFlow: Large-Scale Machine Learning on Heterogeneous Systems. https://www.tensorflow.org/ Software available from tensorflow.org.

[2] Alexandr Andoni and Piotr Indyk. 2008. Near-optimal Hashing Algorithms for Approximate Nearest Neighbor in High Dimensions. Commun. ACM 51, 1 (Jan. 2008), 117–122. https://doi.org/10.1145/1327452.1327494

[3] Immanuel Bayer, Xiangnan He, Bhargav Kanagal, and Steffen Rendle. 2017. A Generic Coordinate Descent Framework for Learning from Implicit Feedback. In Proceedings of the 26th International Conference on World Wide Web (WWW '17). 1341–1350.

[4] Yoshua Bengio and Jean-Sébastien Sénécal. 2003. Quick Training of Probabilistic Neural Nets by Importance Sampling. In Proceedings of the conference on Artificial Intelligence and Statistics (AISTATS).

[5] Y. Bengio and J. S. Senecal. 2008. Adaptive Importance Sampling to Accelerate Training of a Neural Probabilistic Language Model. Trans. Neur. Netw. 19, 4 (April 2008), 713–722. https://doi.org/10.1109/TNN.2007.912312

[6] Alex Beutel, Paul Covington, Sagar Jain, Can Xu, Jia Li, Vince Gatto, and Ed H. Chi. 2018. Latent Cross: Making Use of Context in Recurrent Recommender Systems. In Proceedings of the Eleventh ACM International Conference on Web Search and Data Mining (WSDM '18). ACM, New York, NY, USA, 46–54. https://doi.org/10.1145/3159652.3159727

[7] Guy Blanc and Steffen Rendle. 2018. Adaptive Sampled Softmax with Kernel Based Sampling. In Proceedings of the 35th International Conference on Machine Learning, ICML 2018, Stockholmsmässan, Stockholm, Sweden, July 10-15, 2018. 589–598. http://proceedings.mlr.press/v80/blanc18a.html

[8] Tianqi Chen, Weinan Zhang, Qiuxia Lu, Kailong Chen, Zhao Zheng, and Yong Yu. 2012. SVDFeature: A Toolkit for Feature-based Collaborative Filtering. J. Mach. Learn. Res. 13, 1 (Dec. 2012), 3619–3622. http://dl.acm.org/citation.cfm?id=2503308.2503357

[9] Heng-Tze Cheng, Levent Koc, Jeremiah Harmsen, Tal Shaked, Tushar Chandra, Hrishi Aradhye, Glen Anderson, Greg Corrado, Wei Chai, Mustafa Ispir, Rohan Anil, Zakaria Haque, Lichan Hong, Vihan Jain, Xiaobing Liu, and Hemal Shah. 2016. Wide Deep Learning for Recommender Systems. arXiv:1606.07792 (2016). http://arxiv.org/abs/1606.07792

[10] Edith Cohen and David D. Lewis. 1997. Approximating Matrix Multiplication for Pattern Recognition Tasks. In Proceedings of the Eighth Annual ACM-SIAM Symposium on Discrete Algorithms (SODA '97). Society for Industrial and Applied Mathematics, Philadelphia, PA, USA, 682–691. http://dl.acm.org/citation.cfm?id=314161.314415

[11] Graham Cormode and S. Muthukrishnan. 2005. An Improved Data Stream Summary: The Count-min Sketch and Its Applications. J. Algorithms 55, 1 (April 2005), 58–75. https://doi.org/10.1016/j.jalgor.2003.12.001

[12] Paul Covington, Jay Adams, and Emre Sargin. 2016. Deep Neural Networks for YouTube Recommendations. In Proceedings of the 10th ACM Conference on Recommender Systems. New York, NY, USA.

[13] Jeffrey Dean, Greg S. Corrado, Rajat Monga, Kai Chen, Matthieu Devin, Quoc V. Le, Mark Z. Mao, Marc'Aurelio Ranzato, Andrew Senior, Paul Tucker, Ke Yang, and Andrew Y. Ng. 2012. Large Scale Distributed Deep Networks. In NIPS.

[14] Tim Donkers, Benedikt Loepp, and Jürgen Ziegler. 2017. Sequential User-based Recurrent Neural Network Recommendations. In Proceedings of the Eleventh ACM Conference on Recommender Systems (RecSys '17). ACM, New York, NY, USA, 152–160. https://doi.org/10.1145/3109859.3109877

[15] John Duchi, Elad Hazan, and Yoram Singer. 2011. Adaptive Subgradient Methods for Online Learning and Stochastic Optimization. J. Mach. Learn. Res. 12 (July 2011), 2121–2159. http://dl.acm.org/citation.cfm?id=1953048.2021068

[16] Wikimedia Foundation. [n.d.]. Wikimedia Downloads. https://dumps.wikimedia.org/

[17] Daniel Gillick, Alessandro Presta, and Gaurav Singh Tomar. 2018. End-to-End Retrieval in Continuous Space. CoRR abs/1811.08008 (2018). arXiv:1811.08008 http://arxiv.org/abs/1811.08008

[18] Carlos A. Gomez-Uribe and Neil Hunt. 2015. The Netflix Recommender System: Algorithms, Business Value, and Innovation. ACM Trans. Manage. Inf. Syst. 6, 4, Article 13 (Dec. 2015), 19 pages. https://doi.org/10.1145/2843948

[19] Ian Goodfellow, Yoshua Bengio, and Aaron Courville. 2016. Deep Learning. MIT Press. http://www.deeplearningbook.org.

[20] Ruiqi Guo, Sanjiv Kumar, Krzysztof Choromanski, and David Simcha. 2016. Quantization based Fast Inner Product Search. In Proceedings of the 19th International Conference on Artificial Intelligence and Statistics (Proceedings of Machine Learning Research), Arthur Gretton and Christian C. Robert (Eds.), Vol. 51. PMLR, Cadiz, Spain, 482–490. http://proceedings.mlr.press/v51/guo16a.html

[21] Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu, and Tat-Seng Chua. 2017. Neural Collaborative Filtering. In Proceedings of the 26th International Conference on World Wide Web (WWW '17). International World Wide Web Conferences Steering Committee, Republic and Canton of Geneva, Switzerland, 173–182. https://doi.org/10.1145/3038912.3052569

[22] Balázs Hidasi, Alexandros Karatzoglou, Linas Baltrunas, and Domonkos Tikk. 2016. Session-based Recommendations with Recurrent Neural Networks. In The International Conference on Learning Representations (ICLR 2016).

[23] Y. Hu, Y. Koren, and C. Volinsky. 2008. Collaborative Filtering for Implicit Feedback Datasets. In 2008 Eighth IEEE International Conference on Data Mining. 263–272. https://doi.org/10.1109/ICDM.2008.22

[24] Anjuli Kannan, Karol Kurach, Sujith Ravi, Tobias Kaufman, Balint Miklos, Greg Corrado, Andrew Tomkins, Laszlo Lukacs, Marina Ganea, Peter Young, and Vivek Ramavajjala. 2016. Smart Reply: Automated Response Suggestion for Email. In Proceedings of the ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD) (2016). https://arxiv.org/pdf/1606.04870v1.pdf

[25] Noam Koenigstein, Parikshit Ram, and Yuval Shavitt. 2012. Efficient Retrieval of Recommendations in a Matrix Factorization Framework. In Proceedings of the 21st ACM International Conference on Information and Knowledge Management (CIKM '12). ACM, New York, NY, USA, 535–544. https://doi.org/10.1145/2396761.2396831

[26] Walid Krichene, Nicolas Mayoraz, Steffen Rendle, Li Zhang, Xinyang Yi, Lichan Hong, Ed Chi, and John Anderson. 2019. Efficient Training on Very Large Corpora via Gramian Estimation. In 7th International Conference on Learning Representations.

[27] Jiaqi Ma, Zhe Zhao, Jilin Chen, Ang Li, Lichan Hong, and Ed H. Chi. 2019. SNR: Sub-Network Routing for Flexible Parameter Sharing in Multi-task Learning. In AAAI 2019. http://www.jiaqima.com/papers/SNR.pdf

[28] Jiaqi Ma, Zhe Zhao, Xinyang Yi, Jilin Chen, Lichan Hong, and Ed H. Chi. 2018. Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD '18). ACM, New York, NY, USA, 1930–1939. https://doi.org/10.1145/3219819.3220007

[29] Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, and Jeffrey Dean. 2013. Distributed Representations of Words and Phrases and Their Compositionality. In Proceedings of the 26th International Conference on Neural Information Processing Systems - Volume 2 (NIPS'13). Curran Associates Inc., USA, 3111–3119. http://dl.acm.org/citation.cfm?id=2999792.2999959

[30] Frederic Morin and Yoshua Bengio. 2005. Hierarchical probabilistic neural network language model. In AISTATS'05. 246–252.

[31] Paul Neculoiu, Maarten Versteegh, and Mihai Rotaru. 2016. Learning Text Similarity with Siamese Recurrent Networks. In Rep4NLP@ACL.

[32] The Netflix Prize. 2012. The Netflix Prize. http://www.netflixprize.com/.

[33] S. Rendle. 2010. Factorization Machines. In 2010 IEEE International Conference on Data Mining. 995–1000. https://doi.org/10.1109/ICDM.2010.127

[34] Maksims Volkovs, Guangwei Yu, and Tomi Poutanen. 2017. DropoutNet: Addressing Cold Start in Recommender Systems. In Advances in Neural Information Processing Systems 30, I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett (Eds.). Curran Associates, Inc., 4957–4966. http://papers.nips.cc/paper/7081-dropoutnet-addressing-cold-start-in-recommender-systems.pdf

[35] Ruoxi Wang, Bin Fu, Gang Fu, and Mingliang Wang. 2017. Deep & Cross Network for Ad Click Predictions. In Proceedings of the ADKDD'17 (ADKDD'17). ACM, New York, NY, USA, Article 12, 7 pages. https://doi.org/10.1145/3124749.3124754

[36] Xiang Wu, Ruiqi Guo, Ananda Theertha Suresh, Sanjiv Kumar, Daniel N Holtmann-Rice, David Simcha, and Felix X Yu. 2017. Multiscale Quantization for Fast Similarity Search. In Advances in Neural Information Processing Systems 30, I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett (Eds.). Curran Associates, Inc., 5749–5757. http://papers.nips.cc/paper/7157-multiscale-quantization-for-fast-similarity-search.pdf

[37] Yinfei Yang, Steve Yuan, Daniel Cer, Sheng-Yi Kong, Noah Constant, Petr Pilar, Heming Ge, Yun-hsuan Sung, Brian Strope, and Ray Kurzweil. 2018. Learning Semantic Textual Similarity from Conversations. In Proceedings of The Third Workshop on Representation Learning for NLP. Association for Computational Linguistics, Melbourne, Australia, 164–174. https://www.aclweb.org/anthology/W18-3022

[38] Han Zhu, Xiang Li, Pengye Zhang, Guozheng Li, Jie He, Han Li, and Kun Gai. 2018. Learning Tree-based Deep Model for Recommender Systems. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD '18). ACM, New York, NY, USA, 1079–1088. https://doi.org/10.1145/3219819.3219826
