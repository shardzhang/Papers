# Practice on Long Sequential User Behavior Modeling for Click-Through Rate Prediction

> Qi Pi, Weijie Bian, Guorui Zhou, Xiaoqiang Zhu, Kun Gai | Alibaba Group, Beijing, P.R.China
>
> KDD '19, August 4–8, 2019, Anchorage, AK, USA


本文介绍了 Practice on Long Sequential User Behavior Modeling for Click-Through Rate Prediction（长序列用户行为建模在点击率预测中的实践）。核心内容：

- 通过机器学习算法与在线服务系统协同设计（UIC + MIMN），从长序列用户行为数据中捕获用户兴趣
- 设计独立的UIC（用户兴趣中心）模块，将资源消耗最大的用户兴趣建模从整个CTR预测模型中解耦，对实时流量请求无延迟
- 提出基于记忆的MIMN（多通道用户兴趣记忆网络）架构，通过记忆利用正则化和记忆归纳单元改进NTM，并以增量方式与UIC一起实现

关键发现：

- UIC与MIMN的协同设计使系统理论上可处理无限长度的序列行为数据，实现用户兴趣建模
- 首批能够处理长度达数千的长序列用户行为数据的工业解决方案之一，已部署于阿里巴巴展示广告系统
- 在公共数据集和工业数据集上均优于最先进模型，在线A/B测试中CTR提升7.5%、RPM提升6%

---

## 摘要

CTR（Click-Through Rate，点击率）预测对于推荐系统和在线广告等工业应用至关重要。在实践中，通过从丰富的历史行为数据中挖掘用户兴趣，在这些应用的CTR建模中发挥着重要作用。在深度学习的推动下，具有精巧设计架构的深度CTR模型被提出用于用户兴趣建模，在离线指标上带来了显著的模型性能提升。然而，将这些复杂模型部署到在线服务系统进行实时推理面临巨大挑战，需要应对海量的流量请求。当涉及长序列用户行为数据时，情况变得更加困难，因为系统延迟和存储成本随着用户行为序列的长度近似线性增长。

在本文中，我们直面长序列用户行为建模的挑战，并介绍我们在CTR预测任务中通过机器学习算法和在线服务系统协同设计的实践。（i）从服务系统角度看，我们通过设计一个名为UIC（User Interest Center，用户兴趣中心）的独立模块，将用户兴趣建模中资源消耗最大的部分从整个模型中解耦出来。UIC维护每个用户的最新兴趣状态，其更新取决于实时用户行为触发事件，而不是流量请求。因此，UIC对于实时CTR预测是无延迟的。（ii）从机器学习算法角度看，我们提出了一种新颖的基于记忆的架构，名为MIMN（Multi-channel user Interest Memory Network，多通道用户兴趣记忆网络），用于从长序列行为数据中捕获用户兴趣，实现了优于现有最先进模型的性能。MIMN以增量方式与UIC模块一起实现。

理论上，UIC和MIMN的协同设计方案使我们能够处理无限长度的序列行为数据的用户兴趣建模。模型性能和系统效率的比较证明了所提出方案的有效性。据我们所知，这是首批能够处理长度达数千的长序列用户行为数据的工业解决方案之一。该方案现已部署在阿里巴巴的展示广告系统中。

*Q. Pi和W. Bian为共同第一作者。通讯作者为G. Zhou。

**CCS概念**：• 信息系统 $\rightarrow$ 推荐系统；在线广告。

**关键词**：Click-Through Rate Prediction; User Behavior Modeling

ACM引用格式：
Qi Pi, Weijie Bian, Guorui Zhou, Xiaoqiang Zhu, Kun Gai. 2019. Practice on Long Sequential User Behavior Modeling for Click-Through Rate Prediction. In The 25th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '19), August 4–8, 2019, Anchorage, AK, USA. ACM, New York, NY, USA, 9 pages. https://doi.org/10.1145/3292500.3330666

---

## 1 引言

日益发展的互联网将我们带入了一个具有个性化在线服务的数字世界。从在线系统中收集的海量用户行为数据为我们更好地理解用户偏好提供了巨大机会。从技术上来说，从丰富的用户行为数据中捕获用户兴趣至关重要，因为它为典型的现实应用（如推荐系统和在线广告）带来了显著的性能提升[8, 30, 31]。在本文中，我们将范围限定在点击率（CTR）预测建模任务上，这在在线服务中扮演着关键角色。本文讨论的解决方案也适用于许多相关任务，如转化率预测和用户偏好建模。

在深度学习的推动下，具有精巧设计架构的深度CTR模型被提出用于用户兴趣建模，并取得了最先进的性能。这些模型大致可以分为两类：（i）基于池化的架构[4, 8, 31]，将用户的历史行为视为独立信号，并使用sum/max/attention等池化操作来总结用户兴趣表示；（ii）序列建模架构[21, 30]，将用户行为视为序列信号，并使用LSTM（Long Short-Term Memory，长短期记忆）/GRU（Gated Recurrent Unit，门控循环单元）操作进行用户兴趣总结。

然而，在工业应用中，将这些复杂模型部署到在线服务系统进行实时推理需要巨大的努力，每天有数亿用户访问系统。当遇到极长的序列用户行为数据时，情况变得更加困难，因为上述所有模型都需要在在线服务系统中存储完整的用户行为序列（即特征），并在严格的延迟限制内获取这些序列来计算兴趣表示。这里的"长"意味着序列用户行为的长度达到1000或更多。实际上，系统延迟和存储成本随着用户行为序列的长度近似线性增长。如[30]中所述，部署仅处理最大长度为50的用户行为序列的序列模型就需要大量的工程工作。图1展示了阿里巴巴在线展示广告系统中用户行为序列的平均长度和相应的CTR模型性能。显然，解决长序列用户行为建模的挑战是值得的。

在本文中，我们介绍了机器学习算法和在线服务系统协同设计的实践。我们将用户行为建模模块从整个CTR预测系统中解耦出来，并相应地设计了具体的解决方案。（i）服务系统角度：我们设计了一个独立的UIC（用户兴趣中心）模块。UIC专注于用户行为建模的在线服务问题，维护每个用户的最新兴趣表示。UIC的一个关键点是其更新机制。用户状态的更新仅取决于实时用户行为触发事件，而不是流量请求。也就是说，UIC对于实时CTR预测是无延迟的。（ii）机器学习算法角度：仅解耦UIC模块无法解决存储问题，因为对于数亿用户，当用户行为序列长度达到1000时，存储和推理仍然相当困难。在此，我们借鉴了NTM（Neural Turing Machine，神经图灵机）[11]中的记忆网络思想，提出了一种新颖的架构，名为MIMN（多通道用户兴趣记忆网络）。MIMN以增量方式工作，并且可以容易地与UIC模块一起实现。这有助于解决存储挑战。此外，MIMN通过两种设计——记忆利用正则化和记忆归纳单元——对传统NTM进行了改进，使其在有限存储下建模用户行为序列更加高效，并带来了显著的模型性能提升。

理论上，结合UIC和MIMN为我们提供了一种处理无限长度序列行为数据的用户兴趣建模的解决方案。我们的实验显示了所提出方案在模型性能和系统效率方面的优越性。据我们所知，这是首批能够处理长度达数千的长序列用户行为数据的工业解决方案之一。

本工作的主要贡献总结如下：

* 我们介绍了CTR预测任务中学习算法和服务系统协同设计的实践。该方案已部署在全球领先的广告系统中，并使我们具备了处理长序列用户行为建模的能力。
* 我们设计了一个新颖的UIC模块，将繁重的用户兴趣计算从整个CTR预测过程中解耦出来。UIC对于流量请求是无延迟的，并允许任意复杂的模型计算（相对于实时推理，这些计算以离线模式运行）。
* 我们提出了一个新颖的MIMN模型，通过两种设计——记忆利用正则化和记忆归纳单元——改进了原始的NTM架构，使其更适合用户兴趣学习。MIMN易于与UIC服务器一起实现，以增量方式更新每个用户的兴趣表示。
* 我们在公共数据集和从阿里巴巴广告系统收集的工业数据集上进行了仔细的实验。我们还详细分享了部署所提出方案的实际问题经验。我们相信这将有助于推动社区进步。

**图1：阿里巴巴展示广告系统中序列用户行为数据的统计及相应的模型性能。**

---

## 2 相关工作

**深度CTR模型。** 随着深度学习的快速发展，我们在许多领域取得了进展，例如计算机视觉[15]、自然语言处理[2]。受这些成功的启发，一系列基于深度学习的CTR预测方法[4, 8, 24, 29]被提出。与传统的特征工程方法不同，这些方法利用神经网络来捕获特征交互。虽然这个想法看似简单，但这些工作在CTR预测任务的发展中迈出了一大步。此后，工业界更加关注模型架构设计，而不是通过穷举式特征工程来提升性能。除了学习特征交互之外，越来越多的方法被提出来从丰富的历史行为数据中捕获用户的洞察。DIN[31]指出用户的兴趣是多样化的，并且随item而变化。DIN引入了注意力机制来捕获用户的兴趣。DIEN[30]提出了辅助损失来从具体行为中捕获潜在兴趣，并改进了GRU[6]来建模兴趣的演化。

**长期用户兴趣。** [17]认为长期兴趣意味着一般性兴趣，它存在于用户的脑海中并对于个性化非常重要。[19]提出在类别级别上建模用户的长期兴趣。[5]以增量方式建模长期和短期用户画像分数以表达用户兴趣。所有这些方法都是通过特征工程而不是自适应的端到端学习来建模长期兴趣。TDSSM[25]提出联合建模长期和短期用户兴趣以提高推荐质量。不幸的是，这些基于深度学习的方法，如TDSSM、DIN、DIEN，在面对极长用户行为序列时很难部署到实时预测服务器中。存储和计算延迟的压力将随着用户行为序列的长度线性增长。在工业应用中，行为序列的长度通常较小，例如50，而淘宝上的活跃用户在两周内可能会留下超过1000个行为（如点击、转化等）。

**记忆网络。** 记忆网络[11, 26]被提出来通过外部记忆组件提取知识。这一思想已广泛应用于NLP领域，如问答系统。几项工作[3, 10, 16, 27]利用记忆网络进行用户兴趣建模。然而，这些方法忽略了长期兴趣建模和实际部署问题。

---

## 3 实时CTR预测系统

在现实世界的推荐或广告系统中，CTR预测模块是一个关键组件[8, 31]。通常它接收一组候选（如item或广告），并通过执行实时模型推理返回相应的预测概率分数。这个过程需要在严格的延迟限制下完成，实践中典型的延迟限制为10毫秒。

图2的A部分简要说明了我们在线展示广告系统中用于CTR任务的RTP（Real-Time Prediction，实时预测）系统。为了便于读者理解，我们假设RTP的请求输入仅包括用户和广告信息，省略了上下文或其他因素。

### 3.1 使用长序列用户行为数据服务的挑战

在工业应用中，例如电子商务行业的推荐系统[30, 31]，用户行为特征在特征集中贡献了最大的体量。例如，在我们的系统中，近90%的特征是用户行为特征，其余10%是用户人口统计特征和广告特征。这些行为数据包含丰富的信息，对用户兴趣建模非常有价值[5, 17, 19, 25]。图1显示了在我们的系统中在不同天数内收集的用户行为序列的平均长度，以及使用不同长度的用户行为特征训练的基础模型（Embedding&MLP[31]）的离线性能。在不做任何其他努力的情况下，使用长度为1000的序列的基础模型相比长度为100的序列在AUC（Area Under the Curve，曲线下面积）上获得了0.6%的提升。值得一提的是，仅0.3%的AUC提升对我们的业务来说就已经足够显著了。这一提升表明利用这些长序列用户行为数据具有重要价值。

然而，利用长序列行为数据带来了巨大的挑战。实际上，来自数亿用户的行为特征体量巨大。为了在推荐系统中保持低延迟和高吞吐量，行为特征通常存储在一个额外的分布式内存存储系统中，比如我们系统中的TAIR[9]。当收到流量请求时，这些特征被获取到预测服务器并参与实时推理计算。根据我们的实践经验，在我们的系统中实现DIEN[30]需要大量的工程工作。当用户行为序列长度为150时，延迟和吞吐量已经达到了RTP系统的性能极限，更不用说长度为1000的情况了。直接引入更多的用户行为数据相当困难，因为它面临几个挑战，其中两个最关键的包括：

* **存储约束。** 我们的系统中有超过6亿用户。每个用户的行为序列最大长度为150。这需要大约1TB的存储空间，存储的不仅是product_id，还有其他相关特征ID，如shop_id、brand_id等。当行为序列长度达到1000时，将消耗6TB的存储空间，并且这个数字随着用户行为序列的长度线性增长。如前所述，我们的系统使用高性能存储来保持低延迟和高吞吐量，维持如此巨大的存储成本过高。庞大的数据量也导致用户行为特征的相应计算和更新成本高昂。因此，相当长的行为序列意味着无法接受的存储消耗。

* **延迟约束。** 使用序列深度网络进行实时推理众所周知极具挑战性，尤其是在我们这样拥有海量请求的场景中。DIEN[30]在我们的系统中部署了几项技术来降低DIEN服务的延迟，达到了14ms，每个worker的QPS（Queries Per Second，每秒查询数）为500。然而，当用户行为长度达到1000时，DIEN在500 QPS下的延迟达到200ms。在我们的展示广告系统中，延迟限制为30ms（500 QPS），这是难以承受的。因此，在现有的系统架构下，无法获得长行为带来的收益。

**图2：用于CTR任务的实时预测（RTP）系统示意图。它通常由三个关键组件组成：特征管理模块、模型管理模块和预测服务器。（A）是我们RTP系统的上一个版本，（B）是使用所提出的UIC服务器更新后的版本。系统A和B之间的关键区别在于用户兴趣表示的计算：（i）在A中，它是相对于请求在预测服务器内执行的。（ii）在B中，它是相对于实时用户行为事件在UIC服务器中单独执行的。也就是说，它被解耦了，并且相对于流量请求是无延迟的。**

### 3.2 用户兴趣中心

为了解决上述长序列用户行为建模的挑战，我们提出了一种机器学习算法和服务系统协同设计的解决方案。由于用户行为建模是CTR预测系统中最具挑战性的部分，我们设计了一个UIC（用户兴趣中心）模块来处理它。

图2的B部分展示了新设计的带有UIC服务器的RTP系统。系统A和B之间的区别在于用户兴趣表示的计算。在B中，UIC服务器维护每个用户的最新兴趣表示。UIC的一个关键点是其更新机制。用户状态的更新仅取决于实时用户行为触发事件，而不是请求。也就是说，UIC对于实时CTR预测是无延迟的。在我们的系统中，UIC可以将DIEN模型在1000行为长度下的延迟从200ms降低到19ms（500 QPS）。

---

## 4 多通道用户兴趣记忆网络

在本节中，我们将详细介绍我们的用于长序列用户行为建模的机器学习算法。

### 4.1 从长序列用户行为数据中学习的挑战

从长序列数据中学习是众所周知的难题。简单的RNN（Recurrent Neural Network，循环神经网络）[28]、GRU[7]、LSTM[14]在面对相当长的序列时失败并不奇怪。注意力机制通过将序列数据的必要信息压缩成固定长度的张量来增强模型的表达能力[1, 31]。例如，在DIN[31]中，注意力机制通过软搜索与目标item相关的部分隐藏状态或源行为序列来工作。对于实时推理，它需要存储所有原始行为序列，这给在线系统带来了巨大的存储压力。此外，注意力的计算成本随行为序列长度线性增长，这对于长序列用户行为建模是不可接受的。实际上，RNN中的隐藏状态并非设计用来存储过去源序列的全部信息，而是更关注于预测目标。因此，最后的隐藏状态可能会遗忘长期信息。此外，存储所有隐藏状态也是冗余的。

最近，NTM（神经图灵机[11]）被提出来从源序列中捕获信息，并将其存储在固定大小的外部记忆中，在许多长序列数据建模任务中取得了比RNN模型显著的改进。借鉴NTM的思想，本文提出了一种基于记忆网络的模型，为我们处理长序列用户行为建模提供了新的解决方案。我们将该模型命名为MIMN（多通道用户兴趣记忆网络），如图3所示。UIC存储MIMN的外部记忆张量，并为用户的每个新行为更新它。通过这种方式，UIC以增量方式从用户的行为序列中捕获用户兴趣。尽管UIC存储的是固定长度的记忆张量而不是原始行为序列，但在考虑存储压力时，记忆张量的维度必须受到限制。在本文中，我们提出了记忆利用正则化，通过提高记忆的利用率来增强UIC中记忆张量的表达能力。

另一方面，由于用户兴趣会随时间变化和演化，我们提出了记忆归纳单元来帮助捕获高阶信息。

### 4.2 神经图灵机

MIMN遵循传统的Embedding&MLP范式[8, 30, 31]，更多细节请参阅[31]。MIMN的结构如图3所示。

标准的NTM通过记忆网络从序列数据中捕获和存储信息。在时间步 $t$，记忆参数表示为 $M_t$，由 $m$ 个记忆槽 $\{M_t(i)\}_{i=1}^{m}$ 组成。NTM的两个基本操作是记忆读取和记忆写入，它们通过控制器与记忆进行交互。

**记忆读取。** 输入第 $t$ 个行为嵌入向量后，控制器生成一个读取键 $k_t$ 来寻址记忆。它首先遍历所有记忆槽，生成权重向量 $w_t^r$：

$$
w_t^r(i) = \frac{\exp\left(K\left(k_t, M_t(i)\right)\right)}{\sum_j \exp\left(K\left(k_t, M_t(j)\right)\right)}, \quad \text{for } i = 1, 2, \ldots, m \qquad (1)
$$

其中
$$
K\left(k_t, M_t(i)\right) = \frac{k_t^T M_t(i)}{\left\|k_t\right\|\left\|M_t(i)\right\|}, \qquad (2)
$$

然后计算加权记忆汇总作为输出 $r_t$：

$$
r_t = \sum_i w_t^r(i) M_t(i), \qquad (3)
$$

**记忆写入。** 用于记忆写寻址的权重向量 $w_t^w$ 的生成与记忆读取操作类似，如公式（1）所示。控制器还会生成两个额外的键：加法向量 $a_t$ 和擦除向量 $e_t$，它们控制记忆的更新：

$$
M_t = (1 - E_t) \odot M_{t-1} + A_t, \qquad (4)
$$

其中 $E_t$ 是擦除矩阵，$A_t$ 是加法矩阵，$E_t = w_t^w \otimes e_t$，$A_t = w_t^w \otimes a_t$。这里 $\odot$ 和 $\otimes$ 分别表示点积和外积。

**图3：提出的MIMN模型的网络架构。MIMN由两个主要部分组成：（i）左侧子网络，专注于使用序列行为特征进行用户兴趣建模；（ii）右侧子网络，遵循传统的Embedding&MLP范式，将左侧子网络的输出和其他特征作为输入。MIMN的贡献在于左侧子网络，它受NTM模型启发，包含两个重要的记忆架构：a）基本的NTM记忆单元，具有标准的记忆读取和记忆写入操作；b）记忆归纳单元，基于先前学习的NTM记忆，使用多通道GRU捕获高阶信息。**

### 4.3 记忆利用正则化

在实践中，基本的NTM存在记忆利用不平衡的问题，尤其是在用户兴趣建模场景中。也就是说，热门item往往容易出现在用户行为数据序列中并主导记忆更新，导致记忆利用率低下。NLP领域的先前工作[22, 23]提出了使用LRU策略来平衡每个记忆的利用。由于LRU在处理过程中非常注重平衡每个短时间段内记忆的利用，LRU几乎不会在相邻时间步将信息写入同一个槽。然而，在我们的场景中，用户可能会与属于同一兴趣的几个行为进行交互，因此这些行为应该被写入同一个槽。LRU会打乱内容寻址，不适合我们的任务。在本文中，我们提出了一种名为记忆利用正则化的新策略，该策略被证明对用户兴趣建模有效。

**记忆利用正则化。** 记忆利用正则化策略背后的思想是规范化不同记忆槽之间写入权重的方差，推动记忆利用趋于平衡。设 $g_t = \sum_{c=1}^{t} \tilde{w}_c^w$ 为截至第 $t$ 时间步的累积更新权重，其中 $\tilde{w}_c^w$ 表示第 $c$ 时间步的重新平衡后的写入权重。重新平衡后的写入权重 $\tilde{w}_t^w$ 可以表示为：

$$
P_t = \text{softmax}(W_{\partial} g_t), \qquad (5)
$$
$$
\tilde{w}_t^w = w_t^w \odot P_t, \qquad (6)
$$

$w_t^w$ 是第4.2节中介绍的原始写入权重，$\tilde{w}_t^w$ 是用于记忆更新的新写入权重。权重转移矩阵 $P_t$ 取决于：（i）$g_t$，表示在第 $t$ 步每个记忆槽的累积利用率；（ii）参数矩阵 $W_{\partial}$，通过正则化损失学习：

$$
\tilde{w}^w = \sum_{t=1}^{T} \tilde{w}_t^w, \qquad (7)
$$
$$
L_{reg} = \lambda \sum_{i=1}^{m}\left(\tilde{w}^w(i) - \frac{1}{m}\sum_{i=1}^{m}\tilde{w}^w(i)\right)^2, \qquad (8)
$$

其中 $m$ 是记忆的槽数。$L_{reg}$ 有助于减少不同记忆槽之间更新权重的方差。用 $\tilde{w}_t^w$ 替换 $w_t^w$ 后，所有 $m$ 个槽的更新率趋于均衡。通过这种方式，所有记忆槽的利用率都得到提高，趋于平衡。利用正则化可以帮助记忆张量从源行为数据中存储更多信息。

### 4.4 记忆归纳单元

NTM中的记忆被设计为尽可能多地存储来自源数据的原始信息。美中不足的是，它可能无法捕获一些高阶信息，例如兴趣各部分的演化过程。为了进一步增强用户兴趣提取的能力，MIMN设计了一个记忆归纳单元（MIU，Memory Induction Unit）。MIU也包含一个内部记忆 $S$，其槽数为 $m$，与NTM相同。这里我们将每个记忆槽称为一个用户兴趣通道。在第 $t$ 时间步，MIU：（i）选择 $k$ 个通道，通道索引在集合 $\{i : w_t^r(i) \in \text{topk}(w_t^r)\}_{i=1}^{k}$ 中，其中 $w_t^r$ 是上述NTM的记忆读取权重向量，如公式（1）所示；（ii）对于第 $i$ 个选中的通道，根据公式（9）更新 $S_t(i)$：

$$
S_t(i) = \text{GRU}(S_{t-1}(i), M_t(i), e_t), \qquad (9)
$$

其中 $M_t(i)$ 是NTM的第 $i$ 个记忆槽，$e_t$ 是行为嵌入向量。公式（9）显示MIU从原始行为输入和NTM模块中记忆的信息中捕获信息。这起到了归纳过程的作用，如图4所示。多通道记忆的GRU参数是共享的，不会增加参数量。

**图4：多通道记忆归纳过程。**

### 4.5 在线服务实现

与[30, 31]应用注意力机制来获得以候选为中心的兴趣表示不同，MIMN学习为每个用户在外部记忆中显式地捕获和存储用户的多样化兴趣。这种基于记忆的架构不需要候选（例如我们系统中的目标广告，如图3所示）与用户行为序列之间的交互计算，并且可以增量执行，使其可扩展以处理长序列用户行为建模。

MIMN在在线服务中的实现是直接的。正如第3.2节所介绍的，我们将整个模型拆分并在两个服务器中实现：最重的计算部分——带有NTM和MIU的用户兴趣建模左侧子网络——在UIC服务器中实现，如图5所示，其余的右侧子网络在RTP服务器中实现。图3清晰地说明了这种实现。

NTM和MIU模块都享有增量计算的好处。最新的记忆状态表示用户兴趣，并被更新到TAIR中用于实时CTR预测。当接收到新的用户行为事件时，UIC计算并更新用户兴趣表示到TAIR。通过这种方式，用户行为数据不需要被存储。大量长期用户行为数据可以从6T减少到我们系统中的2.7T。

**图5：在UIC服务器中使用NTM和MIU实现用户兴趣建模子网络。**

**讨论。** UIC服务器和MIMN算法的协同设计使我们能够处理长度达数千的长序列用户行为数据。用户兴趣表示的UIC更新独立于整个模型计算，使其对实时CTR预测无延迟。MIMN提出以增量方式建模用户兴趣，无需像传统解决方案那样存储完整的用户行为序列。此外，MIMN设计了改进的记忆架构，实现了优越的模型性能。然而，它并非适用于所有情况。我们建议在以下应用中采用该方案：（i）丰富的用户行为数据，（ii）实时用户行为事件的流量规模不能显著超过实时CTR预测请求的流量规模。

---

## 5 实验

在本节中，实验分为两部分：（i）我们详细介绍了算法验证，包括数据集、实验设置、对比模型和相应的分析。公共数据集和实验代码均已公开¹。（ii）我们讨论并分享了在阿里巴巴展示广告系统中部署所提出方案的实际经验。

¹https://github.com/UIC-Paper/MIMN

### 5.1 数据集和实验设置

模型比较在三个数据集上进行：两个公共数据集和一个从阿里巴巴在线展示广告系统收集的工业数据集。表1显示了所有数据集的统计信息。

**Amazon数据集²** 由来自Amazon的产品评论和元数据组成[20]。我们使用Amazon数据集的Books子集。对于该数据集，我们将评论视为一种交互行为，并按时间对一个用户的评论进行排序。假设用户 $u$ 有 $T$ 个行为，我们的目标是使用前 $T-1$ 个行为来预测用户 $u$ 是否会撰写第 $T$ 个评论中所示的评论。为了聚焦于长序列用户行为预测，我们过滤掉行为序列长度小于20的样本，并将行为序列截断为长度100。

²http://jmcauley.ucsd.edu/data/amazon/

**Taobao数据集³** 是来自淘宝推荐系统的用户行为集合[12]。该数据集包含几种类型的用户行为，包括点击、购买等。它包含约一百万个用户的用户行为序列。我们取每个用户的点击行为，并按时间排序以构建行为序列。假设用户 $u$ 有 $T$ 个行为，我们使用前 $T-1$ 个点击的商品作为特征来预测用户是否会点击第 $T$ 个商品。行为序列被截断为长度200。

³https://tianchi.aliyun.com/dataset/dataDetail?dataId=649&userId=1

**工业数据集** 从阿里巴巴的在线展示广告系统收集。样本来自曝光日志，以"点击"或"未点击"作为标签。训练集由过去49天的样本组成，测试集是随后一天的样本，这是工业建模的经典设置。在该数据集中，每天样本中的用户行为特征包含前60天的历史行为序列，长度截断为1000。

**实验设置** 对于所有模型，我们使用Adam[18]求解器。我们应用指数衰减，初始学习率为0.001，衰减率为0.9。FCN（Fully Connected Network，全连接网络）的层数设置为 $200 \times 80 \times 2$。嵌入维度设置为16，与记忆槽的维度相同。MIU中GRU的隐藏维度设置为32。NTM和MIU中的记忆槽数是一个参数，在消融研究部分进行了仔细考察。我们采用AUC作为模型性能的衡量指标。

**表1：本文使用的数据集统计信息。**

| 数据集 | 用户数 | item数a | 类别数 | 样本数 |
|--------|--------|---------|--------|--------|
| Amazon（图书） | 75,053 | 358,367 | 1,583 | 150,016 |
| Taobao | 987,994 | 4,162,024 | 9,439 | 987,994 |
| 工业数据集 | 2.9亿 | 6亿 | 100,000 | 122亿 |

a 对于工业数据集，item指广告。

### 5.2 对比模型

我们将MIMN与长序列用户行为建模场景中最先进的CTR预测模型进行比较。

* **Embedding&MLP** 是用于CTR预测的基础深度学习模型。它采用求和池化操作来整合行为嵌入。
* **DIN [31]** 是用户行为建模的早期工作，提出了相对于候选的软搜索用户行为的方法。
* **GRU4Rec [13]** 基于RNN，是第一个使用循环单元建模序列用户行为的工作。
* **ARNN** 是GRU4Rec的一个变体，使用注意力机制对时间上的所有隐藏状态进行加权求和，以获得更好的用户序列表示。
* **RUM [3]** 使用外部记忆来存储用户的行为特征。它还利用软写入和注意力读取机制来与记忆交互。我们使用特征级别的RUM来存储序列信息。
* **DIEN [30]** 将GRU与以候选为中心的注意力技巧相结合，以捕获用户兴趣的演化趋势，并取得了最先进的性能。为了公平比较，我们省略了DIEN中用于更好嵌入学习的辅助损失技巧，否则上述所有模型都应实现该技巧。

### 5.3 公共数据集上的结果

表2展示了所有对比模型和MIMN的结果。每个实验重复3次。

所有其他模型都优于Embedding&MLP，这验证了用户行为建模网络架构设计的有效性。MIMN在AUC指标上显著优于所有模型。我们相信这是因为基于记忆的架构具有巨大容量，适合用户行为建模。如第4.1节所讨论的，长序列行为数据背后的用户兴趣是多样化的并随时间演化。MIMN通过两个方面使用多通道记忆学习捕获用户兴趣：（i）基本NTM中的记忆，具有平衡的利用，用于兴趣记忆；（ii）MIU中的记忆，通过在NTM记忆的基础上归纳兴趣的序列关系，进一步捕获高阶信息。

**表2：公共数据集上的模型性能（AUC）**

| 模型 | Taobao（均值 $\pm$ 标准差） | Amazon（均值 $\pm$ 标准差） |
|------|----------------------|----------------------|
| Embedding&MLP | 0.8709 $\pm$ 0.00184 | 0.7367 $\pm$ 0.00043 |
| DIN | 0.8833 $\pm$ 0.00220 | 0.7419 $\pm$ 0.00049 |
| GRU4REC | 0.9006 $\pm$ 0.00094 | 0.7411 $\pm$ 0.00185 |
| ARNN | 0.9066 $\pm$ 0.00420 | 0.7420 $\pm$ 0.00029 |
| RUM | 0.9018 $\pm$ 0.00253 | 0.7428 $\pm$ 0.00041 |
| DIEN | 0.9081 $\pm$ 0.00221 | 0.7481 $\pm$ 0.00102 |
| MIMN | 0.9179 $\pm$ 0.00325 | 0.7593 $\pm$ 0.00150 |

### 5.4 消融研究

在本节中，我们研究了MIMN中不同模块的效果。

**记忆的槽数。** 我们对具有不同记忆槽数的MIMN进行实验，槽数是人工设置的参数。为简化起见，我们仅使用基本NTM架构评估MIMN，省略了记忆利用正则化和记忆归纳单元的设计。表3显示了结果。

**表3：不同槽数下的模型性能**

| 模型 | Taobao（均值 $\pm$ 标准差） | Amazon（均值 $\pm$ 标准差） |
|------|----------------------|----------------------|
| MIMN 4槽 | 0.9046 $\pm$ 0.00135 | 0.7522 $\pm$ 0.00231 |
| MIMN 6槽 | 0.9052 $\pm$ 0.00202 | 0.7503 $\pm$ 0.00120 |
| MIMN 8槽 | 0.9070 $\pm$ 0.00186 | 0.7486 $\pm$ 0.00071 |

经验上，槽数影响模型性能。对于Amazon数据集，4槽时性能最佳；对于Taobao数据集，8槽时最佳。我们的分析结果表明，这与数据集中用户行为序列的长度有关。每个记忆槽都是随机初始化的。对于行为序列较长的数据集（如Taobao数据集），记忆有更多机会学习并获得稳定的表示。在行为序列较短的情况下（如Amazon数据集），较大记忆容量的模型性能会受到学习困难的影响。特别是当所有记忆槽的利用不均衡时，部分记忆向量可能没有得到充分的利用和更新，这意味着这些记忆向量仍保持在接近初始化的状态。这会损害模型性能。因此，我们提出了记忆利用正则化来缓解这个问题。

**记忆利用正则化。** 由于每个用户的兴趣强度不同以及记忆的随机初始化，基本NTM模型中存储的利用可能是不平衡的。这个问题会损害记忆的学习，使其无法充分利用有限的记忆存储。我们采用记忆利用正则化技巧来帮助解决这个问题。图6显示了记忆利用情况，验证了所提出正则化器的有效性。这种平衡效果也带来了模型性能的提升，如表4所示。

**图6：NTM中不同槽上的记忆利用率**

**表4：MIMN在有/无记忆利用正则化和记忆归纳单元时的模型性能（AUC）比较**

| 模型 | Taobao（均值 $\pm$ 标准差） | Amazon（均值 $\pm$ 标准差） |
|------|----------------------|----------------------|
| MIMN (无MURa和MIUb) | 0.9070 $\pm$ 0.00186 | 0.7486 $\pm$ 0.00071 |
| MIMN (有MUR) | 0.9112 $\pm$ 0.00267 | 0.7551 $\pm$ 0.00121 |
| MIMN (有MUR和MIU) | 0.9179 $\pm$ 0.00208 | 0.7593 $\pm$ 0.00296 |

a MUR代表记忆利用正则化（Memory Utilization Regularization）
b MIU代表记忆归纳单元（Memory Induction Unit）

**记忆归纳单元。** 通过从基本NTM进行归纳，带有记忆归纳单元的MIMN能够捕获高阶信息并带来更多改进，如表4所示。它增强了用户兴趣提取的能力，有助于从长序列行为数据中建模用户兴趣。

### 5.5 工业数据集上的结果

我们进一步在从阿里巴巴在线展示广告系统收集的数据集上进行了实验。我们将MIMN与DIEN模型进行了比较。表5显示了结果。MIMN在AUC上比DIEN提升了0.01，这对我们的业务来说是显著的。

除了离线模型性能之外，MIMN和DIEN模型在系统问题上也存在巨大差异。图7展示了在使用MIMN和DIEN模型服务时，实时CTR预测系统的系统性能。MIMN与UIC服务器的协同设计以较大优势超过了DIEN，保持了恒定的延迟和吞吐量特性。因此，MIMN可以在我们的系统中利用长度达数千的长序列用户行为数据，并享受模型性能的提升。相反，使用DIEN服务的系统在延迟和系统吞吐量方面都受到困扰。由于系统压力，DIEN中利用的用户行为序列长度（作为我们上一个产品模型）仅为50。这再次验证了我们提出的解决方案的优越性。

**表5：工业数据集上的模型性能（AUC）**

| 模型 | AUC |
|------|-----|
| DIEN | 0.6541 |
| MIMN | 0.6644 |
| MIMN (失步设置，一天内) | 0.6644 |
| MIMN (使用大促数据训练) | 0.6627 |

**图7：实时CTR预测系统在不同用户行为序列长度下的系统性能，分别使用MIMN和DIEN模型服务。MIMN模型使用UIC服务器设计实现。**

**在线A/B测试。** 我们已将提出的解决方案部署在阿里巴巴的展示广告系统中。从2019年3月30日到2019年5月10日，我们进行了严格的在线A/B测试实验来验证提出的MIMN模型。与DIEN（我们上一个产品模型）相比，MIMN实现了7.5%的CTR提升和6%的RPM（Revenue Per Mille，千次展示收入）提升。我们将此归因于所提出的协同设计方案能够从长序列行为数据中挖掘额外信息。

### 5.6 部署实践经验

在本节中，我们讨论了在我们的在线系统中部署所提出的UIC和MIMN解决方案的实践经验。

**UIC服务器和RTP服务器的同步。** 如第4.5节所述，MIMN由UIC和RTP服务器共同实现。因此，UIC和RTP服务器之间存在失步问题。两个服务器的异步参数更新可能会导致具有周期性模型部署的真实系统中出现错误的模型推理，这具有很大的风险。我们进行了实验来模拟失步情况。表5显示了结果。具有失步参数更新的MIMN在模型性能上几乎没有差异。注意，在这个实验中，失步更新的时间差距在一天之内，这是工业系统中的传统设置。实际上，在我们的真实系统中，模型部署被设计为每小时执行一次，进一步降低了风险。我们相信这是因为MIMN学习了稳定的用户兴趣表示，从而产生了良好的泛化性能。

**大促数据的影响。** 如今，许多电子商务网站采用大促活动来吸引用户在线消费，例如中国阿里巴巴著名的双11大促。在这种极端情况下，样本的分布以及用户行为与日常情况有很大不同。我们比较了MIMN在使用和不使用我们的系统中双11大促日收集的训练数据时的性能。结果显示在表5中。我们发现根据经验，移除大促数据效果更好。

**预热策略。** 虽然UIC被设计为增量更新，但从一开始就需要相当长的时间来稳定积累。在实践中，我们采用预热策略，用预先计算好的用户兴趣表示来初始化UIC。也就是说，我们为每个用户收集过去120天的历史行为（用户行为序列的平均长度为1000），并在离线模式下使用训练好的MIMN模型进行推理，然后将累积的记忆推送到UIC以进行进一步的增量更新。这一策略使得在尽可能早地部署所提出解决方案时能够获得合理的模型性能。

**回滚策略。** 为了防止意外问题，例如大规模在线作弊导致的训练样本污染，UIC服务器的增量更新机制可能会受到很大影响。一个麻烦的挑战是找到异常情况发生的时间点。为了抵御这种风险，我们设计了一个回滚策略，每天凌晨0点存储学习到的用户兴趣表示的副本，并记录过去7天的副本。

---

## 6 结论

在本文中，我们专注于通过机器学习算法和在线服务系统的协同设计来利用长序列用户行为数据。在计算方面，存储是从相当长的序列用户行为数据中捕获长期用户兴趣的主要瓶颈。我们介绍了我们的实践——一个新颖的解决方案：用于用户兴趣建模实时推理的解耦UIC服务器，以及一个可以增量实现并优于其他最先进模型的基于记忆的MIMN模型。

值得一提的是，深度学习为我们提供了强大的工具包，可以在工业应用中引入更多有价值的数据。我们相信这项工作通过使用极长序列用户行为数据进行建模，开辟了新的空间。未来，我们计划进一步推动研究，包括学习算法、训练系统以及在线服务系统。

---

## 致谢

作者感谢马国强、沈振中、刘浩宇、王为照、马驰、宋俊涛、易鹏涛在在线系统实现中付出的辛勤努力。

---

## 参考文献

[1] Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. 2014. Neural machine translation by jointly learning to align and translate. arXiv preprint arXiv:1409.0473 (2014).

[2] Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. 2015. Neural Machine Translation by Jointly Learning to Align and Translate. In Proceedings of the 3rd International Conference on Learning Representations.

[3] Xu Chen, Hongteng Xu, Yongfeng Zhang, Jiaxi Tang, Yixin Cao, Zheng Qin, and Hongyuan Zha. 2018. Sequential recommendation with user memory networks. In Proceedings of the Eleventh ACM International Conference on Web Search and Data Mining. ACM, 108–116.

[4] Heng-Tze Cheng and Levent Koc. 2016. Wide & deep learning for recommender systems. In Proceedings of the 1st Workshop on Deep Learning for Recommender Systems. ACM, 7–10.

[5] Christina Yip Chung, Abhinav Gupta, Joshua M Koran, Long-Ji Lin, and Hongfeng Yin. 2011. Incremental update of long-term and short-term user profile scores in a behavioral targeting system. US Patent 7,904,448.

[6] Junyoung Chung, Caglar Gulcehre, KyungHyun Cho, and Yoshua Bengio. 2014. Empirical evaluation of gated recurrent neural networks on sequence modeling. arXiv preprint arXiv:1412.3555 (2014).

[7] Junyoung Chung, Caglar Gulcehre, KyungHyun Cho, and Yoshua Bengio. 2014. Empirical evaluation of gated recurrent neural networks on sequence modeling. arXiv preprint arXiv:1412.3555 (2014).

[8] Paul Covington, Jay Adams, and Emre Sargin. 2016. Deep neural networks for youtube recommendations. In Proceedings of the 10th ACM Conference on Recommender Systems. ACM, 191–198.

[9] duolong, daoan, and fanggang. 2017. TAIR, A distributed key-value storage system developed by Alibaba Group. https://github.com/alibaba/tair.

[10] Travis Ebesu, Bin Shen, and Yi Fang. 2018. Collaborative Memory Network for Recommendation Systems. arXiv preprint arXiv:1804.10862 (2018).

[11] Alex Graves, Greg Wayne, and Ivo Danihelka. 2014. Neural turing machines. arXiv preprint arXiv:1410.5401 (2014).

[12] Zhu Han, Pengye Zhang, Guozheng Li, He Jie, and Kun Gai. 2018. Learning Tree-based Deep Model for Recommender Systems. (2018), 1079–1088.

[13] Balázs Hidasi, Alexandros Karatzoglou, Linas Baltrunas, and Domonkos Tikk. 2015. Session-based recommendations with recurrent neural networks. arXiv preprint arXiv:1511.06939 (2015).

[14] Sepp Hochreiter and Jürgen Schmidhuber. 1997. Long short-term memory. Neural computation 9, 8 (1997), 1735–1780.

[15] Gao Huang, Zhuang Liu, Laurens van der Maaten, and Kilian Q. Weinberger. [n.d.]. Densely connected convolutional networks.

[16] Jin Huang, Wayne Xin Zhao, Hongjian Dou, Ji-Rong Wen, and Edward Y Chang. 2018. Improving sequential recommendation with knowledge-enhanced memory networks. In The 41st International ACM SIGIR Conference on Research & Development in Information Retrieval. ACM, 505–514.

[17] Hyoung R Kim and Philip K Chan. 2003. Learning implicit user interest hierarchy for context in personalization. In Proceedings of the 8th international conference on Intelligent user interfaces. ACM, 101–108.

[18] Diederik P Kingma and Jimmy Ba. 2014. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980 (2014).

[19] Hongche Liu and MS Zamanian. 2007. Framework for selecting and delivering advertisements over a network based on combined short-term and long-term user behavioral interests. US Patent App. 11/225,238.

[20] Julian McAuley, Christopher Targett, Qinfeng Shi, and Anton Van Den Hengel. 2015. Image-based recommendations on styles and substitutes. In Proceedings of the 38th International ACM SIGIR Conference on Research and Development in Information Retrieval. ACM, 43–52.

[21] Massimo Quadrana, Alexandros Karatzoglou, Balázs Hidasi, and Paolo Cremonesi. 2017. Personalizing session-based recommendations with hierarchical recurrent neural networks. In Proceedings of the Eleventh ACM Conference on Recommender Systems. ACM, 130–137.

[22] Jack Rae, Jonathan J Hunt, Ivo Danihelka, Timothy Harley, Andrew W Senior, Gregory Wayne, Alex Graves, and Timothy Lillicrap. 2016. Scaling memory-augmented neural networks with sparse reads and writes. In Advances in Neural Information Processing Systems. 3621–3629.

[23] Adam Santoro, Sergey Bartunov, Matthew Botvinick, Daan Wierstra, and Timothy Lillicrap. 2016. One-shot learning with memory-augmented neural networks. arXiv preprint arXiv:1605.06065 (2016).

[24] Ying Shan, T Ryan Hoens, Jian Jiao, Haijing Wang, Dong Yu, and JC Mao. [n.d.]. Deep Crossing: Web-scale modeling without manually crafted combinatorial features.

[25] Yang Song, Ali Mamdouh Elkahky, and Xiaodong He. 2016. Multi-rate deep learning for temporal recommendation. In Proceedings of the 39th International ACM SIGIR conference on Research and Development in Information Retrieval. ACM, 909–912.

[26] Sainbayar Sukhbaatar, Jason Weston, Rob Fergus, et al. 2015. End-to-end memory networks. In Advances in neural information processing systems. 2440–2448.

[27] Qinyong Wang, Hongzhi Yin, Zhiting Hu, Defu Lian, Hao Wang, and Zi Huang. 2018. Neural memory streaming recommender networks with adversarial training. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 2467–2475.

[28] Ronald J Williams and David Zipser. 1989. A learning algorithm for continually running fully recurrent neural networks. Neural computation (1989), 270–280.

[29] Shuangfei Zhai, Keng-hao Chang, Ruofei Zhang, and Zhongfei Mark Zhang. 2016. Deepintent: Learning attentions for online advertising with recurrent neural networks. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. ACM, 1295–1304.

[30] Guorui Zhou, Na Mou, Ying Fan, Qi Pi, Weijie Bian, Chang Zhou, Xiaoqiang Zhu, and Kun Gai. 2019. Deep interest evolution network for click-through rate prediction. In Proceedings of the 33nd AAAI Conference on Artificial Intelligence. Honolulu, USA.

[31] Guorui Zhou, Xiaoqiang Zhu, Chenru Song, Ying Fan, Han Zhu, Xiao Ma, Yanghui Yan, Junqi Jin, Han Li, and Kun Gai. 2018. Deep interest network for click-through rate prediction. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 1059–1068.
