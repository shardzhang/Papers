# Sampling Is All You Need on Modeling Long-Term User Behaviors for CTR Prediction

> Yue Cao, XiaoJiang Zhou, Jiaqi Feng, Peihao Huang, Yao Xiao, Dayao Chen, Sheng Chen | Meituan Inc.

---

本文介绍了 Sampling Is All You Need on Modeling Long-Term User Behaviors for CTR Prediction。核心内容：


关键发现：

---

## 摘要

丰富的用户行为数据已被证明对CTR（点击率）预测应用具有巨大价值，特别是在工业推荐系统、搜索系统或广告系统中。然而，由于在线服务时间的严格要求，现实世界系统很难充分利用长期用户行为。先前的大多数工作采用基于检索的策略，即首先检索少量用户行为，然后进行后续的注意力计算。然而，基于检索的方法并非最优，会导致信息损失，并且很难平衡检索算法的有效性和效率。

在本文中，我们提出SDIM（基于采样的深度兴趣建模，Sampling-based Deep Interest Modeling），这是一种简单而有效的基于采样的端到端方法，用于建模长期用户行为。我们通过多个哈希函数进行采样，为目标item和用户行为序列中的每个item生成哈希签名，并通过直接聚合与目标item具有相同哈希签名的行为item来获得用户兴趣。我们在理论上和实验上证明，所提出的方法在建模长期用户行为方面与基于标准注意力的模型性能相当，同时速度提升数倍。我们还介绍了SDIM在我们系统中的部署。具体来说，我们将最耗时的部分——行为序列哈希——从CTR模型中解耦出来，设计了一个名为BSE（行为序列编码，Behavior Sequence Encoding）的独立模块。BSE对CTR服务器而言是无延迟的，使我们能够建模极长的用户行为。我们进行了离线和在线实验来证明SDIM的有效性。SDIM目前已在美团APP的搜索系统中在线部署。

---

## CCS概念

- **信息系统** \rightarrow 个性化；推荐系统。

---

## 关键词

基于哈希的采样，长期用户行为建模，点击率预测

---

## ACM引用格式

Yue Cao, XiaoJiang Zhou, Jiaqi Feng, Peihao Huang, Yao Xiao, Dayao Chen, Sheng Chen. 2022. Sampling Is All You Need on Modeling Long-Term User Behaviors for CTR Prediction. In *CIKM '22: 31st ACM International Conference on Information and Knowledge Management, October 17-21, 2022, Georgia, USA*. ACM, New York, NY, USA, 10 pages. https://doi.org/10.1145/1122445.1122456

---

## 1 引言

点击率（CTR）预测是工业应用系统中的一项关键任务。用户兴趣建模旨在从用户历史行为数据中学习用户的隐式兴趣，已被广泛引入现实世界系统，并带来了显著的性能提升[4, 14, 18]。

研究人员提出了各种模型来建模用户兴趣[5, 14, 18]。其中，DIN[18]通过考虑给定目标item时历史行为的相关性来自适应地计算用户兴趣。DIN引入了一个名为"目标注意力"（target attention）的新注意力模块，其中目标item充当查询Q，用户历史行为充当键K和值V。由于其优越的性能，基于DIN的方法近年来已成为建模用户兴趣的主流解决方案。然而，在线服务时间的严格要求限制了可以使用的用户行为序列长度。因此，大多数工业系统会截断用户行为序列，仅使用最近50个行为[17, 18]进行用户兴趣建模，这导致了信息损失。

随着互联网的快速发展，用户在电商平台上积累了越来越多的行为数据。以淘宝¹为例，他们报告称23%的用户在六个月内拥有超过1,000个行为[13]。在美团APP中，超过60%的用户至少拥有1,000个行为，超过10%的用户在一年内至少拥有5,000个行为。如何有效利用更多的用户行为以进行更准确的用户兴趣估计，对工业系统来说变得越来越重要。

最近，一些方法被提出用于从长行为序列中建模用户的长期兴趣[3, 10–12]。MIMN[10]通过设计一个独立的用户兴趣中心（UIC）模块，将用户兴趣建模从整个模型中解耦出来。尽管UIC可以节省大量在线服务时间，但它使得CTR模型无法利用来自目标item的信息，而已有研究表明这对用户兴趣建模至关重要[3]。因此，MIMN只能建模浅层用户兴趣。SIM[11]和UBR4CTR[12]采用两阶段框架来处理长期用户行为。它们首先从序列中检索top-𝑘个相似item，然后将这些item输入后续的注意力模块[18]。正如[3]所指出的，这些方法的检索目标与CTR模型的目标存在分歧，且离线倒排索引中的预训练嵌入不适合在线学习系统。为了提高检索算法的质量，ETA[3]提出了一种基于LSH的方法，以端到端的方式从用户行为中检索top-𝑘个相似item。他们使用局部敏感哈希（LSH）将item转换为哈希签名，然后基于候选item与行为item之间的汉明距离检索top-𝑘个相似item。LSH的使用大大降低了计算item间相似度的成本，ETA取得了比SIM[11]和UBR4CTR[12]更好的结果。

SIM、UBR4CTR和ETA都是基于检索的方法。基于检索的方法存在以下缺点：从整个序列中检索top-𝑘个item是次优的，并且会对用户长期兴趣的估计产生偏差。当用户拥有丰富的行为时，检索到的top-𝑘个item可能都与候选item相似，从而使得估计的用户兴趣表示不准确。此外，很难平衡检索算法的有效性和效率。以SIM（hard）为例，他们使用简单的检索算法，因此其性能不如其他方法。相比之下，UBR4CTR借助复杂的检索模块取得了显著的改进，但其推理速度慢了4倍[12]，这阻碍了UBR4CTR的在线部署，特别是对于长期用户行为建模而言。

在本文中，我们提出了一种基于简单哈希采样的方法来建模长期用户行为。首先，我们通过多个哈希函数进行采样，为目标item和用户行为序列中的每个item生成哈希签名。我们不使用特定度量来检索top-𝑘个相似item，而是直接从整个序列中聚合与目标item具有相同哈希签名的行为item，以形成用户兴趣。我们方法背后的核心思想是用LSH碰撞概率来近似用户兴趣的softmax分布。我们在理论上证明，这种简单的基于采样的方法可以产生与基于softmax的目标注意力非常相似的注意力模式，并实现一致的模型性能，同时更加高效。因此，我们的方法就像直接在原始长序列上计算注意力一样，没有信息损失。我们将所提出的方法命名为SDIM（基于采样的深度兴趣建模）。

我们还介绍了在线部署SDIM的实践经验。具体来说，我们将框架解耦为两部分：（1）行为序列哈希（BSE）服务器和（2）CTR服务器，这两部分分别部署。行为序列哈希是整个算法中最耗时的部分，解耦这部分大大减少了服务时间。更多细节将在第4.4节中介绍。

我们在公开数据集和工业数据集上都进行了实验。实验结果表明，SDIM取得了与基于标准注意力的方法一致的结果，在建模长期用户行为方面优于所有竞争基线，并且速度提升显著。SDIM已部署在中国最大的生活服务平台——美团²的搜索系统中，带来了2.98%的CTR提升和2.69%的VBR提升，这对我们的业务来说非常显著。

总之，本文的主要贡献总结如下：

- 我们提出了SDIM，一种基于哈希采样的框架，用于CTR预测中的长期用户行为建模。我们证明了这种简单的基于采样的策略可以产生与目标注意力非常相似的注意力模式。
- 我们详细介绍在线部署SDIM的实践经验。我们相信这项工作将有助于推动社区的发展，特别是在长期用户行为建模方面。
- 我们在公开数据集和工业数据集上进行了大量实验，结果证明了SDIM在效率和有效性方面的优越性。SDIM已部署在美团搜索系统中，为业务带来了显著改进。

---

## 2 相关工作

### 2.1 点击率（CTR）预测

随着深度学习的快速发展，基于深度神经网络的方法在CTR预测中取得了显著性能。用户兴趣建模是CTR预测的关键问题。近年来，许多工作[4, 8, 17, 18]专注于从用户历史行为中学习更好的表示。DIN[18]引入了目标注意力，该机制针对不同的目标item学习不同的用户兴趣。[17]考虑到用户行为中的兴趣演化过程，提出了DIEN，其中包含一个兴趣演化层来捕捉关于目标item的动态兴趣。在DSIN[4]中，基于用户行为在每个会话内高度同质、在不同会话间高度异质的先验知识，提出了基于会话的模型。值得注意的是，DIN[18]中的目标注意力机制如今已被广泛应用于CTR模型中。

### 2.2 长期序列化用户行为建模

用户行为建模在工业应用中表现出色。探索CTR模型中更丰富的用户行为已引起广泛关注。然而，长期用户行为建模面临着复杂模型部署和系统延迟等挑战。

在MIMN[10]中，提出了具有基于记忆的架构设计的用户兴趣中心（UIC）模块来应对这一挑战。该模块通过用户行为事件更新，且只需存储有限的用户兴趣记忆。MIMN难以建模用户行为序列与目标item之间的交互，而已有研究表明这种交互在CTR建模中非常重要。在SIM[11]中，提出了一种两阶段方法来建模长期用户行为序列。首先，利用通用搜索单元（GSU）提取与目标item相关的行为。其次，提出了精确搜索单元（ESU）以端到端方式对相关行为进行建模。UBR4CTR[12]使用与SIM[11]类似的基于搜索的方法来应对挑战。最近，ETA[3]提出了一种端到端的目标注意力方法来建模长期用户行为序列。他们应用局部敏感哈希（LSH）来减少训练和推理的时间成本。

除了上述CTR预测领域的相关工作，自然语言处理（NLP）领域也有大量旨在提高自注意力效率的工作[6, 15, 16, 19]。这些方法可以将自注意力的时间复杂度从𝑂(𝐿²)降低到𝑂(𝐿 log 𝐿)，其中𝐿是序列长度，但不能应用于降低目标注意力𝑂(𝐿)的时间复杂度。

---

## 3 预备知识

### 3.1 任务定义

CTR预测是工业推荐系统、搜索系统和广告系统中的核心任务。CTR任务的目标是估计用户点击item的概率，定义如下：

$$
prob_i = P(y_i = 1 | x_i; \theta)   (1)
$$

在上述公式中，\theta代表CTR模型中的可训练参数。给定输入特征x_i，CTR模型通过最小化交叉熵损失进行训练，以预测点击概率prob_i：

$$
L = - (1/N) \Sigma_i [y_i log(prob_i) + (1 - y_i) log(1 - prob_i)]   (2)
$$

### 3.2 目标注意力

目标注意力的概念最早由DIN[18]提出，并被广泛应用于CTR任务中的用户兴趣建模[3, 4, 10, 11, 17]。目标注意力将目标item作为查询（query），用户行为序列中的每个item作为键（key）和值（value），并使用注意力操作从用户行为序列中软搜索相关部分。然后通过对用户行为序列进行加权求和来获得用户兴趣。

具体来说，将目标item表示为Q $\in$ ℝ^{B$×$d}，用户序列表示为S $\in$ ℝ^{B$×$L$×$d}，其中B是每次请求中由CTR模型评分的候选item数量，L是用户行为序列的长度，d是模型的隐藏层大小。令q_i为第i个目标item，目标注意力计算q_i与行为序列S中每个item的点积相似度，然后使用归一化后的相似度作为权重来获得用户兴趣表示：

$$
TA(q_i, S) = \Sigma_j [exp(q_i^T s_j / t) / \Sigma_k exp(q_i^T s_k / t)] · s_j   (3)
$$

公式3的矩阵形式可以写为：

$$
TA(Q, S) = softmax(Q^T S / t) · S   (4)
$$

缩放因子t用于避免内积值过大[3]。

计算TA(Q, S)的复杂度为𝑂(BLd)，其中B是每次请求的候选item数，L是用户行为序列长度，d是模型的隐藏层大小。在大规模工业系统中，B约为10³，d约为10²，因此在线系统中对长期用户行为建模部署目标注意力是不可行的[3, 10]。

### 3.3 局部敏感哈希（LSH）与SimHash

局部敏感哈希（LSH）[1]是一种在高维空间中高效寻找最近邻的算法技术。LSH满足位置保持特性：距离近的向量有很高的概率获得相同的哈希值，而距离远的向量则不会。借助这一特性，基于LSH的签名已被广泛应用于许多应用，如网络搜索系统[9]。随机投影方案（SimHash）[2, 7]是LSH的一种高效实现。SimHash的基本思想是采样一个随机投影（由一个法向单位向量r定义）作为哈希函数，将输入向量哈希到两个轴（+1或-1）上。具体来说，给定输入x和哈希函数r，对应的哈希计算如下：

$$
h(x, r) = sign(r^T x)   (5)
$$

其中r $\in$ ℝ^d, r_i ~ N(0, 1)。h(x, r) = ±1 取决于哈希输出位于哪一侧。对于两个向量x_1和x_2，我们说x_1与x_2碰撞当且仅当它们具有相同的哈希码值：

$$
p(r) = 1_{h(x_1, r) = h(x_2, r)}   (6)
$$

虽然单个哈希函数可以用来估计相似度，但输出可以是多个哈希函数的平均值，以降低估计误差。在实践中，通常采用(m, \tau)参数化的SimHash（多轮哈希）算法[7]，其中m是哈希函数的数量，\tau是宽度参数。第一步，SimHash随机采样m个哈希函数并为每个输入生成m个哈希码：

$$
h(x, R) = sign(Rx)   (7)
$$

其中R $\in$ ℝ^{m$×$d}, h(x, R) $\in$ ℝ^m。为了降低不相似item具有相同哈希码的概率，该算法将每\tau个码的结果合并，形成一个新的哈希签名。更多细节请参考[7]。在SimHash中，x_1与x_2碰撞当且仅当它们具有相同的哈希签名值，即它们在一批\tau个码中的哈希码全部相同：

$$
p̃(R) = p(r_1) & p(r_2) & ... & p(r_\tau) = AND_{k=1}^{\tau} p(r_k)   (8)
$$

其中"&"表示逻辑"与"运算符，r_k是第k个哈希函数。图1底部展示了一个使用4个哈希函数、2个哈希签名（每个签名合并2个哈希码）的SimHash示例。

---

## 4 方法论

在本节中，我们介绍在我们系统中实现SDIM的框架。高层概览如图1所示。该框架由两个独立的服务器组成：行为序列编码（BSE）服务器和CTR服务器，后续将详细介绍。

### 4.1 基于哈希采样的用户行为建模

#### 4.1.1 基于哈希采样的注意力

第一步，我们采样多个哈希函数，并生成用户行为和候选item的哈希码。与ETA[3]类似，我们使用固定的随机投影矩阵，这些矩阵从标准正态分布中采样，且在训练后保持固定。

在哈希之后，ETA计算行为item与候选item之间的汉明距离，作为用户兴趣分布的近似，以选择top-𝑘个相似item。此处我们提出一种更高效、更有效的方式来近似用户兴趣。

根据位置保持特性，相似的向量有很大概率落入相同的哈希桶中，因此用户行为item与候选item之间的相似度可以通过它们具有相同哈希码（签名）的频率，即碰撞概率来近似。这使我们假设哈希碰撞的概率可以成为用户兴趣的有效估计量。

基于这一观察，我们提出了一种使用LSH获取用户兴趣的新方法。哈希之后，我们直接对与候选itemq具有相同签名的行为items_j进行求和，形成用户兴趣。对于单个哈希函数r，所提出的用户兴趣估计方法可以计算如下：

$$
ℓ_2(P(r)S) = ℓ_2(\Sigma_j p_j(r) · s_j)   (9)
$$

其中S是用户行为序列，s_j是该序列中的第j个item，p_j(r) = {0, 1} 指定s_j是否对用户兴趣做出贡献。具体来说，如果s_j与候选itemq在哈希函数r下共享相同的哈希签名，则p_j(r) = 1，否则p_j(r) = 0：

$$
p_j(r) = 1_{h(q, r) = h(s_j, r)}   (10)
$$

其中h(·,·)在公式5中定义。公式9中的ℓ_2(·)指的是ℓ_2归一化，用于归一化兴趣分布，使得注意力权重之和为1³。

#### 4.1.2 多轮哈希

不相似的item始终有小概率与候选item共享相同的哈希码，从而给模型带来噪声。为了降低这种概率，我们使用第3.3节中描述的多轮哈希算法。具体来说，我们使用(m, \tau)参数化的SimHash算法。我们并行采样并执行m次哈希，将每\tau个哈希码的结果合并形成一个新的哈希签名。我们认为s_j与q碰撞当且仅当它们具有相同的聚合哈希签名值，即s_j在一批签名中的码应与q的全部相同：

$$
p̃_j(R_i) = AND_{k=1}^{\tau} p_j(r_{i,k}), 其中 p_j(r_{i,k}) = 1_{h(q, r_{i,k}) = h(s_j, r_{i,k})}   (11)
$$

m/\tau个哈希签名的输出取平均，以得到低方差的用户兴趣估计。可以表述为：

$$
Attn(q, S) = (1 / (m/\tau)) \Sigma_i ℓ_2(P̃(R_i)S)
           = (1 / (m/\tau)) \Sigma_i ℓ_2(\Sigma_j p̃_j(R_i) · s_j)   (12)
$$

### 4.2 注意力模式分析

#### 4.2.1 基于哈希采样的注意力的期望

在我们的方法中，随着s_j与q越来越相似，它们的碰撞概率变得更高，p̃_j的期望值也更高。可以证明p̃_j的期望是s_j与q在单位圆上的夹角[2]⁴：

$$
E[p̃_j] = (1 - arccos(q^T s_j) / \pi)^\tau   (13)
$$

因此，SDIM产生的用户兴趣表示的期望为：

$$
E[Attn(q, S)] = E[ℓ_2(\Sigma_j p̃_j(R_k) s_j)]
              = \Sigma_j [(1 - arccos(q^T s_j) / \pi)^\tau / \Sigma_k (1 - arccos(q^T s_k) / \pi)^\tau] · s_j   (14)
$$

随着哈希签名数量m/\tau的增加，p̃_j收敛到E[p̃_j]，Attn(q, S)收敛到E[Attn(q, S)]。实验结果表明，当m/\tau $\geq$ 16时，估计误差将非常小。在实践中，我们在在线模型中使用m=48和\tau=3。

图2绘制了SDIM产生的注意力权重(1 - arccos(q^T s_j)/\pi)^³。为了比较，我们还在同一图中绘制了目标注意力产生的注意力权重exp(q^T s_j / 0.5)。从图2中可以看出，SDIM的注意力权重与目标注意力中的softmax函数良好对齐，因此理论上，我们的方法可以获得非常接近目标注意力的输出。我们将所提出的方法命名为SDIM，即基于采样的深度兴趣建模。

#### 4.2.2 \tau的性质及与其他方法的关系

在本小节中，我们描述SDIM中的宽度参数\tau在控制模型对更相似item投入更多注意力的强度方面所起的作用。

记w_j为目标itemq对items_j的注意力权重，即：

$$
w_j = (1 - arccos(q^T s_j) / \pi)^\tau   (15)
$$

随着\tau的增加，注意力分布的熵H(w_j)严格减小，w的分布在较大的相似度区域变得更加尖锐，这促使模型对更相似的item投入更多注意力。

让我们考虑两个极端情况：

- 当\tau \rightarrow +$\infty$（可以通过分配一个大值来近似）时，算法只关注与目标item完全相同的item。这是一个非常严格的匹配算法，类似于SIM（hard）[11]。因此，我们的算法可以被看作是SIM（hard）的扩展，它也会考虑非常相似但具有不同类别ID的行为item。
- 当\tau = 0，或m = 1时，算法退化为平均池化，所有具有相同目标签名的行为item都将被聚合并视为具有相同权重。

因此，SDIM非常灵活，可以通过分配不同的\tau值来建模不同的注意力模式。

### 4.3 实现与复杂度分析

在本小节中，我们阐述SDIM比基于标准注意力的方法快数倍。

让我们回顾目标注意力中计算用户兴趣的公式（公式4）。该算法首先通过将目标向量与行为序列相乘得到注意力权重，然后使用注意力权重对行为序列进行加权求和。因此，目标注意力中的用户兴趣表示需要两次矩阵乘法，其计算复杂度为𝑂(BLd)。

SDIM将用户兴趣的计算分解为两部分：（1）行为序列哈希和（2）目标向量哈希。注意，用户行为序列是用户的固有特征，与候选item无关，这意味着无论候选item是什么，用户行为序列哈希的结果都保持不变。因此，对于每个用户，我们在每次请求中只需要计算一次用户行为序列的哈希变换。结果，SDIM在时间复杂度中将B减少为1，因此比标准目标注意力快数倍。哈希之后，选择与目标item具有相同哈希签名的序列item并求和，以形成用户兴趣。在TensorFlow中，这一步可以通过tf.batch_gather操作实现，这是TensorFlow的原子操作，时间成本可以忽略不计。

SDIM中最耗时的部分是行为序列的多轮哈希，即将d维矩阵转换为m维哈希码。该操作的时间复杂度为𝑂(Lmd)，可以使用近似随机投影算法[1]降低到𝑂(Lm log d)。由于m $≪$ B且log d $≪$ d，SDIM比基于标准注意力的方法快得多。在我们的场景中，SDIM在用户行为建模部分的训练速度提升了10倍到20倍。

### 4.4 整个系统的部署

我们在本小节介绍如何成功在线部署SDIM。如上所述，整个算法被解耦为两部分：（1）行为序列哈希和（2）目标向量哈希。行为序列哈希与候选item无关，这促使我们构建一个专门的服务器来维护每个用户的行为序列哈希。

我们将系统分为两部分：行为序列编码（BSE）服务器和CTR服务器，如图1所示。BSE服务器负责维护每个用户的行为序列哈希。当收到用户行为列表时，BSE服务器从多个哈希函数中进行采样，并为行为序列中的每个item生成哈希签名。然后根据签名将item分配到不同的桶中，每个桶对应一个哈希签名，如图1所示。哈希桶被传递给CTR服务器，以建模候选item感知的用户兴趣。

CTR服务器负责预测候选item的点击概率。当收到一批B个候选item时，CTR服务器将每个item哈希为签名，并从相应的桶中聚合item表示。用户兴趣特征与其他特征连接后，被送入一个复杂的CTR模型，以获得每个item的预测概率。

我们CTR模型的整体结构如图3所示。该模型以item特征、上下文特征、短期用户兴趣特征和长期用户兴趣特征作为输入，并使用多层感知机（MLP）来预测候选item的点击概率。

请注意，SDIM不需要改变CTR模型的结构，可以自然地插入到现有的CTR架构中。所提出的框架在训练阶段是端到端的：用户兴趣建模部分与CTR模型的其他部分联合训练，我们仅在线服务阶段将它们分开部署。

解耦BSE和CTR服务器后，BSE的计算对CTR服务器是无延迟的。在实践中，我们将BSE服务器放在CTR服务器之前，并与检索模块并行计算。

分开部署后，计算用户长期兴趣的时间成本仅在于候选item的哈希，其时间复杂度为𝑂(Bm log d)，与序列长度L无关，这意味着我们的框架理论上可以处理极长行为的用户兴趣建模。从CTR服务器的角度看，这个时间复杂度就像是添加一个普通特征。在表1中，我们比较了不同方法的时间复杂度⁵。

| 方法 | 训练 | 在线服务 |
|------|------|----------|
| DIN | 𝑂(BLd) | 𝑂(BLd) |
| SIM | 𝑂(B log M + Bkd) | 𝑂(B log M + Bkd) |
| ETA | 𝑂(Lm log d + BLm + Bkd) | 𝑂(BLm + Bkd) |
| SDIM | 𝑂(Lm log d + Bm log d) | 𝑂(Bm log d) |

**表1：不同方法在训练和在线服务阶段的时间复杂度。** B是每次请求的候选item数，m是哈希数量，L和k分别是原始和检索后的用户行为序列长度，M是SIM中属性倒排索引的大小，d是模型的隐藏层大小。在我们系统中，B约为10³，L=1024，m=48，d=128。

我们的服务系统与MIMN[10]有些相似。最大的区别在于，我们的系统可以建模用户的深层兴趣，而MIMN只能建模浅层兴趣。

#### 4.4.1 关于服务器间传输成本的说明

对于每次请求，我们需要将桶表示从BSE服务器传输到CTR服务器。注意，我们使用固定数量的哈希函数，因此无论用户的行为有多长，我们只需要向CTR服务器传输固定长度的桶表示向量。在我们的在线系统中，该向量的大小为8KB，传输成本约为1ms。

---

## 5 实验

### 5.1 数据集与评估指标

为了验证SDIM的有效性，我们在公开数据集和真实工业数据集上进行了实验。对于公开数据集，我们遵循先前工作[3, 10, 11]选择淘宝数据集⁶。对于工业数据集，我们使用从美团搜索系统收集的真实数据进行实验。

**淘宝数据集：** 淘宝数据集由[20]发布，并被先前工作广泛用于离线实验[11, 12]。该数据集中的每个样本包含五个特征字段：用户ID、itemID、类别ID、行为类型和时间戳。遵循[12]，我们根据时间戳额外引入"是否周末"特征以丰富上下文特征。我们采用与MIMN[10]和ETA[3]相同的预处理方式。具体来说，我们使用第1到第(L-1)个行为作为输入来预测第L个行为。我们按时间步将样本分为训练集（80%）、验证集（10%）和测试集（10%）。选择最近16个行为作为短期序列，最近256个行为作为长期序列。

**工业数据集：** 该数据集收集自移动端美团APP的搜索平台，这是中国最大的生活服务电商平台。我们选择连续14天的样本进行训练，接下来2天的样本用于评估和测试，训练样本数量约为100亿。选择最近50个行为作为短期序列，最近1024个行为作为长期序列。如果用户行为数量未达到此长度，我们使用默认值将序列填充到最大长度。除用户行为特征外，我们还额外引入约20个重要的ID特征以丰富输入。

**评估指标：** 对于离线实验，我们遵循先前工作，使用广泛使用的AUC（Area Under Curve）进行评估。我们还使用训练和推理速度（T&I Speed）作为补充指标，以展示各模型的效率。对于在线实验，我们使用CTR（点击率）和VBR（访问购买率）作为在线指标。

### 5.2 竞争模型

遵循先前工作[3, 11]，我们将SDIM与以下用于建模长期用户行为的主流工业模型进行比较：

- **DIN[18]：** DIN是在工业系统中建模用户兴趣最流行的模型之一。然而，由于其高时间复杂度，DIN无法在建模长期用户行为时部署。在此基线中，我们仅使用短期用户行为特征，不使用长期特征。
- **DIN（长序列）：** 对于离线实验，为了衡量长期用户行为序列的信息增益，我们为DIN配备长行为序列。我们在淘宝数据集上设置L=256，在工业数据集上设置L=1024。
- **DIN（平均池化长序列）：** 此基线由[3]和[11]提出，其中DIN用于建模短期用户兴趣，长期用户兴趣通过对长期行为进行平均池化操作获得。我们将此基线表示为DIN（Avg-Pooling）。
- **SIM[11]：** SIM首先通过类别ID从整个序列中检索top-𝑘个相似item，然后在top-𝑘个item上应用目标注意力以获得用户兴趣。我们遵循先前工作与SIM（hard）进行比较，因为他们在线上部署的是SIM（hard）且性能几乎相同。
- **UBR4CTR[12]：** UBR4CTR是一种两阶段方法。在第一阶段，他们设计一个特征选择模块来选择特征形成查询，并以倒排索引的方式存储用户行为。在第二阶段，检索到的行为被输入到基于目标注意力的模块中以获得用户兴趣。
- **ETA[3]：** ETA应用LSH将目标item和用户行为序列编码为二进制码，然后计算逐item的汉明距离以选择top-𝑘个相似item进行后续的目标注意力。

MIMN[10]由与SIM相同的团队提出。由于作者声称SIM击败了MIMN并且他们在线部署SIM，为节省篇幅，我们仅与SIM进行比较，省略MIMN基线。

对于所有基线和SDIM，我们使用相同的特征（包括时间信息特征）作为输入，并采用相同的模型结构，除长期用户行为建模模块外。所有模型使用相同长度的长期用户行为（淘宝T=256，工业数据集T=1024）。

---

## 6 结果与分析

### 6.1 淘宝数据集上的结果

不同模型在淘宝数据集上的总体结果如表2所示。我们可以得出以下结论：

| 方法 | AUC↑ | 训练速度↑ |
|------|------|-----------|
| DIN（长序列）[18] | 0.8848 | - |
| DIN[18] | 0.8627* | 7.4$×$ |
| DIN（平均池化）[3, 11] | 0.8669* | 2.6$×$ |
| SIM[11] | 0.8692* | 2.4$×$ |
| UBR4CTR[12] | 0.8752* | 0.8$×$ |
| ETA[3] | 0.8753* | 1.8$×$ |
| SDIM | **0.8854** | **5.0$×$** |

**表2：不同模型在淘宝数据集上的性能比较。** 训练速度提升是基于DIN（长序列）计算的。"*"表示SDIM相对于该基线的改进在配对Wilcoxon检验中p值<0.05，具有统计显著性。

（1）SDIM在建模长期用户行为方面与DIN（长序列）性能相当，同时速度快5倍。如上所述，SDIM可以模拟与目标注意力非常相似的注意力模式，因此SDIM可以匹配甚至超越DIN（长序列）的性能。

（2）SDIM的表现优于所有为建模长期用户行为而提出的基线模型。具体来说，SDIM比SIM高出1.62%，比UBR4CTR高出1.02%，比ETA高出1.01%。我们还注意到，DIN（长序列）相比DIN在AUC上提升了2.21%，这表明了建模长期用户行为对CTR预测的重要性。SIM、UBR4CTR和ETA的性能不如DIN（长序列），这是由于用户行为检索造成的信息损失。这些检索操作可能有助于从序列中去除噪声，但在没有足够信息性行为进行top-𝑘检索时是有害的。

（3）SDIM比DIN（长序列）、SIM和UBR4CTR高效得多。效率提升归因于将算法的时间复杂度降低到𝑂(Lm log d)。SDIM也比ETA[3]高效得多。ETA也使用LSH对目标item和行为序列进行哈希，哈希操作的时间复杂度与SDIM相同。哈希之后，ETA计算汉明距离并选择top-𝑘个item进行目标注意力，时间复杂度为𝑂(BL + Bkd)。而SDIM仅引入一个聚合操作，后接m/\tau个哈希的平均池化，因此比ETA高效得多。

### 6.2 工业数据集上的结果

不同模型在工业数据集上的总体结果如表3所示。与淘宝数据集的结果类似，SDIM优于所有竞争基线，并与DIN（长序列）性能相当。

| 方法 | AUC | 训练速度↑ |
|------|-----|-----------|
| DIN（长序列）[18] | 0.7049 | - |
| DIN[18] | 0.6652* | 13.5$×$ |
| DIN（平均池化）[3, 11] | 0.6749* | 11.0$×$ |
| SIM[11] | 0.6852* | 10.8$×$ |
| UBR4CTR[12] | 0.6836* | 1.2$×$ |
| ETA[3] | 0.6906* | 3.7$×$ |
| SDIM | **0.7044** | **11.4$×$** |

**表3：不同模型在工业数据集上的性能比较。** 训练速度提升是基于DIN（长序列）计算的。"*"表示SDIM相对于该基线的改进在配对Wilcoxon检验中p值<0.05，具有统计显著性。

我们的SDIM相比SIM、UBR4CTR和ETA分别取得了1.92%、2.08%和1.38%的提升，同时速度远快于这些方法。

由于工业数据集中的用户序列长度足够大，这对基于检索的方法是有利的，似乎它们应该能与DIN（长序列）性能相当。然而，表3的结果显示它们的性能与DIN（长序列）存在一定差距。我们认为这是因为用户兴趣通常是多样化的，且人们常常想购买新类别的商品，特别是在我们的餐饮搜索场景中。当面对一个新类别的候选item时，这些检索算法很难准确地从用户历史行为中挑选出最有价值的item。

与淘宝数据集相比，工业数据集每次请求包含更多的候选item，因此SDIM在该数据集上可以获得更大的速度提升。工业数据集还具有更长的用户行为序列（T=1024），因此基于检索的方法也能获得更多的加速。实验结果证明了SDIM的优越性。

### 6.3 超参数分析

SDIM中有两个重要的超参数：（1）哈希数量m，和（2）哈希签名宽度\tau。

#### 6.3.1 m的分析

哈希数量m影响所提出的基于哈希的注意力的估计误差。随着采样哈希函数数量的增加，估计的用户兴趣将更接近公式14中的E[Attn(q, S)]。

为了评估估计误差，我们测试了使用m个哈希的SDIM的性能，其中m $\in$ {24, 36, 48, 60, 72, 90, 120}。我们还实现了一个SDIM的变体，直接使用公式13中的期望碰撞概率E[p̃_j]来计算注意力权重。该基线模拟了当哈希签名数量趋于无穷时SDIM的行为。结果如图4所示。

![图4：工业数据集上当改变哈希数量m时SDIM的AUC结果](attachment: figure4)

从图4可以看出，当m > 48时，模型的性能几乎相同。出于效率考虑，我们在模型中使用m=48。

#### 6.3.2 \tau的分析

如第4.2.2节所述，\tau控制着模型对更相似item投入更多注意力的强度。我们通过在\tau $\in$ {1, 2, 3, 5, 10}范围内变化\tau来研究SDIM的不同注意力模式。结果如表4所示。

| 超参数 | AUC |
|--------|-----|
| \tau = 1 | 0.6911 |
| \tau = 2 | 0.7032 |
| \tau = 3 | **0.7044** |
| \tau = 5 | 0.7034 |
| \tau = 10 | 0.6923 |

**表4：工业数据集上当改变宽度参数\tau时SDIM的AUC结果。**

从表4可以看出，当2 $\leq$ \tau $\leq$ 5时，SDIM表现良好。为了平衡有效性和效率，我们在在线模型中使用\tau = 3。当\tau = 1时模型表现不佳，因为编码了太多噪声行为。相反，当\tau = 10时模型表现不佳，因为只有非常相似的item有机会对用户兴趣做出贡献，这对行为较少的用户不友好。

### 6.4 短期用户行为建模实验

我们还进行了额外实验，测试SDIM在建模短期用户行为上的性能。但请注意，SDIM主要是为了解决工业推荐系统中的长期用户兴趣建模问题而提出的，对于短序列可以直接使用完整的目标注意力或更复杂的模块。我们进行此实验仅为了展示模型在特殊情况下的性能。我们在淘宝数据集上进行此实验，结果如表5所示。

| 方法 | AUC | 训练速度↑ |
|------|-----|-----------|
| DIN (T=16) | 0.8627 | - |
| SDIM (T=16) | **0.8637** | **2.0$×$** |

**表5：不同模型在建模短期用户兴趣上的性能比较。**

结果表明，SDIM在建模短序列方面仍能取得与标准目标注意力模型相当的结果，同时更加高效。

### 6.5 在线A/B测试

我们还进行了严格的在线A/B测试，以验证SDIM的有效性。对于在线A/B测试，基线模型是美团搜索系统中之前的在线CTR模型，该模型仅使用短期用户行为序列。测试模型采用与基线模型相同的结构和特征，但在此基础上引入了一个使用用户最近1024或2000个行为的长期用户兴趣建模模块。我们使用所提出的SDIM来建模长期用户兴趣，并将这种测试模型表示为SDIM（T=1024）和SDIM（T=2000）。测试持续14天，每种模型分配10%的美团搜索流量。A/B测试结果如表6所示。

| 方法 | CTR | VBR | 推理时间 |
|------|-----|-----|----------|
| Base（无长序列） | - | - | \approx60ms |
| DIN（T=1024） | 无法部署 | 无法部署 | - |
| SDIM（T=1024） | +2.39% | +2.21% | +1ms |
| SDIM（T=2000） | +2.98% | +2.69% | +1ms |

**表6：在线A/B测试结果。**

SDIM（T=2000）相比于基线模型在CTR上取得了2.98%的提升（p值<0.001），在VBR上取得了2.69%的提升（p值<0.005），考虑到美团APP的巨大流量，这可以大幅增加在线收益。SDIM（T=2000）的推理时间相比Base（无长序列）仅增加了1ms。推理时间的增加主要是由BSE服务器和CTR服务器之间的传输时间引起的。

我们还尝试部署直接使用目标注意力来建模T=1024长期用户行为序列的模型。然而，其推理时间大幅增加了约50%（25-30ms），这对我们的系统来说是不可接受的。因此我们无法将该模型在线保留14天用于A/B测试。SDIM与该模型性能相当，但节省了95%的在线推理时间。SDIM目前已经在线部署，服务于美团首页搜索系统的主要流量。

---

## 7 结论

在本文中，我们提出了一种名为SDIM的基于哈希采样的方法，用于建模长期用户行为。我们不需要设计复杂的模块来从长期用户行为中检索，而是直接聚合与候选item具有相同哈希签名的行为item来形成用户兴趣。我们还提出了一种新的在线服务框架，将用户行为序列的哈希从整个模型中解耦出来，使其对CTR服务器无延迟。我们证明所提出的方法与DIN（长序列）性能相当，同时速度快数倍。SDIM已在美团APP中在线部署。

未来工作包括降低传输成本、探索更复杂的结构如多头哈希等。

---

## 参考文献

[1] Alexandr Andoni, Piotr Indyk, Thijs Laarhoven, Ilya P. Razenshteyn, and Ludwig Schmidt. 2015. Practical and Optimal LSH for Angular Distance. In *Advances in Neural Information Processing Systems 28: Annual Conference on Neural Information Processing Systems 2015, December 7-12, 2015, Montreal, Quebec, Canada*. 1225–1233.

[2] Moses Charikar. 2002. Similarity estimation techniques from rounding algorithms. In *Proceedings on 34th Annual ACM Symposium on Theory of Computing, May 19-21, 2002, Montréal, Québec, Canada*, John H. Reif (Ed.). ACM, 380–388.

[3] Qiwei Chen, Changhua Pei, Shanshan Lv, Chao Li, Junfeng Ge, and Wenwu Ou. 2021. End-to-End User Behavior Retrieval in Click-Through Rate Prediction Model. *CoRR* abs/2108.04468 (2021).

[4] Yufei Feng, Fuyu Lv, Weichen Shen, Menghan Wang, Fei Sun, Yu Zhu, and Keping Yang. 2019. Deep Session Interest Network for Click-Through Rate Prediction. In *IJCAI 2019, Macao, China, August 10-16, 2019*. ijcai.org, 2301–2307.

[5] Wang-Cheng Kang and Julian J. McAuley. 2018. Self-Attentive Sequential Recommendation. In *IEEE International Conference on Data Mining, ICDM 2018, Singapore, November 17-20, 2018*. IEEE Computer Society, 197–206.

[6] Nikita Kitaev, Łukasz Kaiser, and Anselm Levskaya. 2020. Reformer: The Efficient Transformer. In *8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020*. OpenReview.net.

[7] Jure Leskovec, Anand Rajaraman, and Jeffrey D. Ullman. 2014. *Mining of Massive Datasets, 2nd Ed.* Cambridge University Press.

[8] Chao Li, Zhiyuan Liu, Mengmeng Wu, Yuchi Xu, Huan Zhao, Pipei Huang, Guoliang Kang, Qiwei Chen, Wei Li, and Dik Lun Lee. 2019. Multi-Interest Network with Dynamic Routing for Recommendation at Tmall. In *Proceedings of the 28th ACM International Conference on Information and Knowledge Management, CIKM 2019, Beijing, China, November 3-7, 2019*. ACM, 2615–2623.

[9] Gurmeet Singh Manku, Arvind Jain, and Anish Das Sarma. 2007. Detecting near-duplicates for web crawling. In *WWW 2007, Banff, Alberta, Canada, May 8-12, 2007*. ACM, 141–150.

[10] Qi Pi, Weijie Bian, Guorui Zhou, Xiaoqiang Zhu, and Kun Gai. 2019. Practice on Long Sequential User Behavior Modeling for Click-Through Rate Prediction. In *KDD 2019, Anchorage, AK, USA, August 4-8, 2019*. ACM, 2671–2679.

[11] Qi Pi, Guorui Zhou, Yujing Zhang, Zhe Wang, Lejian Ren, Ying Fan, Xiaoqiang Zhu, and Kun Gai. 2020. Search-based User Interest Modeling with Lifelong Sequential Behavior Data for Click-Through Rate Prediction. In *CIKM '20: The 29th ACM International Conference on Information and Knowledge Management, Virtual Event, Ireland, October 19-23, 2020*. ACM, 2685–2692.

[12] Jiarui Qin, Weinan Zhang, Xin Wu, Jiarui Jin, Yuchen Fang, and Yong Yu. 2020. User Behavior Retrieval for Click-Through Rate Prediction. In *SIGIR 2020, Virtual Event, China, July 25-30, 2020*. ACM, 2347–2356.

[13] Kan Ren, Jiarui Qin, Yuchen Fang, Weinan Zhang, Lei Zheng, Weijie Bian, Guorui Zhou, Jian Xu, Yong Yu, Xiaoqiang Zhu, and Kun Gai. 2019. Lifelong Sequential Modeling with Personalized Memorization for User Response Prediction. In *SIGIR 2019, Paris, France, July 21-25, 2019*. ACM, 565–574.

[14] Jiaxi Tang and Ke Wang. 2018. Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding. In *WSDM 2018, Marina Del Rey, CA, USA, February 5-9, 2018*. ACM, 565–573.

[15] Sinong Wang, Belinda Z. Li, Madian Khabsa, Han Fang, and Hao Ma. 2020. Linformer: Self-Attention with Linear Complexity. *CoRR* abs/2006.04768 (2020).

[16] Zhanpeng Zeng, Yunyang Xiong, Sathya N. Ravi, Shailesh Acharya, Glenn Moo Fung, and Vikas Singh. 2021. You Only Sample (Almost) Once: Linear Cost Self-Attention Via Bernoulli Sampling. In *Proceedings of the 38th International Conference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event* (Proceedings of Machine Learning Research, Vol. 139), Marina Meila and Tong Zhang (Eds.). PMLR, 12321–12332.

[17] Guorui Zhou, Na Mou, Ying Fan, Qi Pi, Weijie Bian, Chang Zhou, Xiaoqiang Zhu, and Kun Gai. 2019. Deep Interest Evolution Network for Click-Through Rate Prediction. In *AAAI 2019, Honolulu, Hawaii, USA, January 27-February 1, 2019*. AAAI Press, 5941–5948.

[18] Guorui Zhou, Xiaoqiang Zhu, Chengru Song, Ying Fan, Han Zhu, Xiao Ma, Yanghui Yan, Junqi Jin, Han Li, and Kun Gai. 2018. Deep Interest Network for Click-Through Rate Prediction. In *KDD 2018, London, UK, August 19-23, 2018*. ACM, 1059–1068.

[19] Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai Zhang, Jianxin Li, Hui Xiong, and Wancai Zhang. 2021. Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting. In *AAAI 2021, Virtual Event, February 2-9, 2021*. AAAI Press, 11106–11115.

[20] Han Zhu, Xiang Li, Pengye Zhang, Guozheng Li, Jie He, Han Li, and Kun Gai. 2018. Learning Tree-based Deep Model for Recommender Systems. In *KDD 2018, London, UK, August 19-23, 2018*. ACM, 1079–1088.

---

> **翻译说明：** 本文为美团发表于 CIKM 2022 的论文《Sampling Is All You Need on Modeling Long-Term User Behaviors for CTR Prediction》的中文翻译。SDIM 核心思想是使用局部敏感哈希（LSH）碰撞概率来近似注意力 softmax 分布，以极低的计算成本实现对长期用户行为序列的建模，并在美团搜索系统中成功部署。部分 OCR 噪声内容已根据语义还原。
