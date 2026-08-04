# Deep Interest Evolution Network for Click-Through Rate Prediction

> Guorui Zhou, Na Mou, Ying Fan, Qi Pi, Weijie Bian, Chang Zhou, Xiaoqiang Zhu, Kun Gai | Alibaba Group
>
> {guorui.xgr, mouna.mn, fanying.fy, piqi.pq, weijie.bwj, ericzhou.zc, xiaoqiang.zxq, jingshi.gk}@alibaba-inc.com
>
> \* 通讯作者：Guorui Zhou
>
> † 该作者为在线测试进行了艰苦的工作。源代码可访问：https://github.com/mouna99/dien.
>
> 版权 © 2019, 人工智能促进协会 (www.aaai.org)。保留所有权利。



本文介绍了深度兴趣演化网络（DIEN）——一种用于点击率（CTR）预测的新模型。它设计了兴趣抽取层（Interest Extractor Layer）从历史行为序列中捕获时序兴趣，并借助辅助损失（Auxiliary Loss）监督每一步的兴趣抽取；再通过兴趣演化层（Interest Evolving Layer）中带有注意力更新门的GRU（AUGRU）建模与目标商品相关的兴趣演化过程。核心内容：

- 设计兴趣抽取层，通过GRU建模行为依赖，并引入辅助损失监督每步隐藏状态，使其有效表示潜在兴趣
- 提出兴趣演化层，将注意力机制嵌入GRU序列结构，加强相关兴趣在兴趣演化中的影响
- 提出带有注意力更新门的GRU（AUGRU），在保留更新门各维度信息的同时用注意力分数缩放，克服兴趣漂移干扰

关键发现：

- DIEN 在公开数据集和工业数据集上均显著优于最先进的方法
- 兴趣抽取层的辅助损失与兴趣演化层的 AUGRU 分别带来显著改进
- DIEN 已在淘宝展示广告系统中部署，CTR 提升 20.7%，eCPM 提升 17.1%

---

## 摘要

点击率（Click-Through Rate, CTR）预测旨在估计用户点击商品的概率，已成为广告系统的核心任务之一。对于CTR预测模型，必须捕捉用户行为数据背后的潜在用户兴趣。此外，考虑到外部环境的变化和内部认知的改变，用户兴趣随时间动态演化。已有多种CTR预测方法用于兴趣建模，但大多数方法直接将行为表示视为兴趣，缺乏对具体行为背后潜在兴趣的专门建模。而且，很少有工作考虑兴趣的变化趋势。在本文中，我们提出了一种名为深度兴趣演化网络（Deep Interest Evolution Network, DIEN）的新模型用于CTR预测。具体而言，我们设计了兴趣抽取层（Interest Extractor Layer）来从历史行为序列中捕获时序兴趣。在这一层中，我们引入了辅助损失（Auxiliary Loss）来监督每一步的兴趣抽取。由于用户兴趣是多样化的，特别是在电子商务系统中，我们提出了兴趣演化层（Interest Evolving Layer）来捕获与目标商品相关的兴趣演化过程。在兴趣演化层中，注意力机制被新颖地嵌入到序列结构中，相关兴趣的影响在兴趣演化过程中得到加强。在公开数据集和工业数据集上的实验中，DIEN显著优于最先进的解决方案。值得注意的是，DIEN已在淘宝的展示广告系统中部署，CTR提升了20.7%。

---

## 1. 引言

按点击付费（Cost per Click, CPC）是广告系统中最常见的计费形式之一，广告主按每次广告点击付费。在CPC广告系统中，点击率（CTR）预测的性能不仅影响整个平台的最终收入，还影响用户体验和满意度。CTR预测建模已引起学术界和工业界越来越多的关注。

在大多数非搜索型电子商务场景中，用户不会主动表达他们当前的意图。设计模型来捕捉用户兴趣及其动态变化是提升CTR预测性能的关键。近年来，许多CTR模型从传统方法[3, 14]转向深度CTR模型[4, 11, 8]。大多数深度CTR模型专注于捕获不同领域特征之间的交互，较少关注用户兴趣表示。深度兴趣网络（Deep Interest Network, DIN）[23]强调用户兴趣是多样化的，它使用基于注意力的模型来捕获与目标商品相关的兴趣，并获得自适应兴趣表示。然而，包括DIN在内的大多数兴趣模型直接将行为视为兴趣，而潜在兴趣很难通过显式行为完全反映。先前的方法忽视了挖掘行为背后的真实用户兴趣。此外，用户兴趣不断演化，捕捉兴趣的动态变化对于兴趣表示非常重要。

基于这些观察，我们提出了深度兴趣演化网络（DIEN）来提升CTR预测的性能。DIEN有两个关键模块：一个用于从显式用户行为中提取潜在的时间兴趣，另一个用于建模兴趣演化过程。适当的兴趣表示是兴趣演化模型的基础。在兴趣抽取层，DIEN选择GRU[2]来建模行为之间的依赖关系。遵循兴趣直接导致连续行为的原则，我们提出了辅助损失，利用下一行为来监督当前隐藏状态的学习。我们将这些带有额外监督的隐藏状态称为兴趣状态（Interest States）。这些额外的监督信息有助于捕捉兴趣表示的更多语义信息，并推动GRU的隐藏状态有效表示兴趣。此外，用户兴趣是多样化的，这导致了兴趣漂移现象：用户在相邻访问中的意图可能非常不同，用户的一个行为可能依赖于很久以前的行为。每个兴趣都有其自身的演化轨迹。同时，用户对不同目标商品的点击行为受到不同兴趣部分的影响。在兴趣演化层，我们对与目标商品相关的兴趣演化轨迹进行建模。基于从兴趣抽取层获得的兴趣序列，我们设计了带有注意力更新门的GRU（GRU with Attentional Update Gate, AUGRU）。通过使用兴趣状态和目标商品计算相关性，AUGRU增强了相关兴趣在兴趣演化中的影响，同时削弱了因兴趣漂移导致的不相关兴趣的影响。通过将注意力机制引入更新门，AUGRU可以为不同的目标商品引导特定的兴趣演化过程。DIEN的主要贡献如下：

- 我们关注电子商务系统中的兴趣演化现象，并提出了一种新的网络结构来建模兴趣演化过程。兴趣演化模型带来了更具表达力的兴趣表示和更精确的CTR预测。

- 与直接将行为视为兴趣不同，我们专门设计了兴趣抽取层。针对GRU的隐藏状态对兴趣表示针对性不足的问题，我们提出了一种辅助损失。辅助损失利用连续行为来监督每一步隐藏状态的学习，使隐藏状态具有足够的表达力来表示潜在兴趣。

- 我们新颖地设计了兴趣演化层，其中带有注意力更新门的GRU（AUGRU）增强了相关兴趣对目标商品的影响，并克服了兴趣漂移的干扰。

在公开数据集和工业数据集上的实验中，DIEN显著优于最先进的解决方案。值得注意的是，DIEN已在淘宝展示广告系统中部署，并在各种指标下取得了显著改进。

---

## 2. 相关工作

凭借深度学习在特征表示和组合方面的强大能力，最近的CTR模型从传统的线性或非线性模型[3, 14]转向深度模型。大多数深度模型遵循嵌入和多层感知机（Embedding and Multi-layer Perceptron, MLP）的结构[23]。基于这一基本范式，越来越多的模型关注特征之间的交互：Wide & Deep[1]和DeepFM[4]都结合了低阶和高阶特征以提高表达能力；PNN[11]提出了产品层来捕获跨领域类别之间的交互模式。然而，这些方法不能清晰地反映数据背后的兴趣。DIN[23]引入了注意力机制来局部地激活与给定目标商品相关的历史行为，并成功捕获了用户兴趣的多样性特征。然而，DIN在捕获序列行为之间的依赖关系方面较弱。

在许多应用领域中，用户-商品交互可以随时间记录下来。最近的一些研究表明，这些信息可用于构建更丰富的个体用户模型并发现额外的行为模式。在推荐系统中，TDSSM[15]联合优化长期和短期用户兴趣以提高推荐质量；DREAM[19]使用递归神经网络（RNN）的结构来研究每个用户的动态表示以及商品购买历史的全局序列行为。He和McAuley[5]构建了视觉感知推荐系统，展示与用户和社区不断演变的兴趣更匹配的产品。Zhang等人[20]基于用户兴趣序列衡量用户相似性，提高了协同过滤推荐的性能。Parsana等人[10]通过使用大规模事件嵌入和递归网络的注意力输出来改进原生广告CTR预测。ATRank[21]使用基于注意力的序列框架来建模异构行为。与序列无关的方法相比，这些方法可以显著提高预测精度。然而，这些传统的基于RNN的模型存在一些问题。一方面，它们大多将序列结构的隐藏状态直接视为潜在兴趣，而这些隐藏状态缺乏对兴趣表示的特殊监督。另一方面，大多数现有的基于RNN的模型连续且平等地处理相邻行为之间的所有依赖关系。众所周知，并非用户的所有行为都严格依赖于每个相邻行为。每个用户都有多样化的兴趣，每个兴趣都有其自身的演化轨迹。对于任何目标商品，这些模型只能获得一个固定的兴趣演化轨迹，因此这些模型可能会受到兴趣漂移的干扰。

为了推动序列结构的隐藏状态有效表示潜在兴趣，应该引入对隐藏状态的额外监督。DARNN[12]使用点击级别的序列预测，对每次向用户展示广告时的点击动作进行建模。除了点击动作，还可以进一步引入排序信息。在推荐系统中，排序损失已被广泛用于排序任务[13, 6]。与这些排序损失类似，我们提出了一种用于兴趣学习的辅助损失。在每一步，辅助损失使用连续点击的商品与未点击的商品来监督兴趣表示的学习。

为了捕获与目标商品相关的兴趣演化过程，我们需要更灵活的序列学习结构。在问答（QA）领域，DMN+[18]使用基于注意力的GRU（AGRU）使注意力机制对输入事实的位置和顺序都敏感。在AGRU中，更新门的向量被简单地替换为注意力的标量分数。这种替换忽略了更新门所有维度之间的差异，而更新门包含了从前一序列传递的丰富信息。受问答领域中新颖序列结构的启发，我们提出了带有注意力门的GRU（AUGRU）来在兴趣演化过程中激活相关兴趣。与AGRU不同，AUGRU中的注意力分数作用于从更新门计算的信息。更新门和注意力分数的结合使演化过程更加具体和敏感。

---

## 3. 深度兴趣演化网络

在本节中，我们将详细介绍深度兴趣演化网络（DIEN）。首先，我们回顾基本的深度CTR模型，称为BaseModel。然后我们展示DIEN的整体结构，并介绍用于捕捉兴趣和建模兴趣演化过程的技术。

### 3.1 BaseModel回顾

BaseModel从特征表示、模型结构和损失函数三个方面介绍。

**特征表示** 在我们的在线展示系统中，我们使用四类特征：用户画像（User Profile）、用户行为（User Behavior）、广告（Ad）和上下文（Context）。值得注意的是，广告也是商品。为统一表述，本文中我们将广告称为目标商品。每类特征都有几个字段：用户画像的字段包括性别、年龄等；用户行为的字段是用户访问过的商品ID列表；广告的字段包括广告ID、店铺ID等；上下文的字段包括时间等。每个字段中的特征可以编码为独热向量，例如用户画像类别中的女性特征编码为 $[0, 1]$ 。来自用户画像、用户行为、广告和上下文的不同字段的独热向量的拼接分别构成 $\mathbf{x}_p$ 、 $\mathbf{x}_b$ 、 $\mathbf{x}_a$ 和 $\mathbf{x}_c$ 。在序列CTR模型中，值得注意的是每个字段包含一个行为列表，每个行为对应一个独热向量，可以表示为 $\mathbf{x}_b = [\mathbf{b}_1; \mathbf{b}_2; \cdots; \mathbf{b}_T] \in \mathbb{R}^{K \times T}$ ， $\mathbf{b}_t \in \{0,1\}^K$ ，其中 $\mathbf{b}_t$ 编码为独热向量并表示第 $t$ 个行为， $T$ 是用户历史行为的数量， $K$ 是用户可以点击的商品总数。

**BaseModel的结构** 大多数深度CTR模型建立在嵌入和MLP的基本结构上。基本结构由几个部分组成：

- **嵌入（Embedding）** 嵌入是将大规模稀疏特征转换为低维稠密特征的常见操作。在嵌入层中，每个特征字段对应一个嵌入矩阵，例如访问过的商品的嵌入矩阵可以表示为 $E_{goods} = [\mathbf{m}_1; \mathbf{m}_2; \cdots; \mathbf{m}_K] \in \mathbb{R}^{n_E \times K}$ ，其中 $\mathbf{m}_j \in \mathbb{R}^{n_E}$ 表示维度为 $n_E$ 的嵌入向量。特别地，对于行为特征 $\mathbf{b}_t$ ，如果 $\mathbf{b}_t[j_t] = 1$ ，则其对应的嵌入向量为 $\mathbf{m}_{j_t}$ ，一个用户的有序行为嵌入向量列表可以表示为 $\mathbf{e}_b = [\mathbf{m}_{j_1}; \mathbf{m}_{j_2}; \cdots, \mathbf{m}_{j_T}]$ 。类似地， $\mathbf{e}_a$ 表示广告类别中各字段拼接后的嵌入向量。

- **多层感知机（MLP）** 首先，来自一个类别的嵌入向量被输入到池化操作中。然后将来自不同类别的所有池化向量拼接起来。最后，拼接后的向量被输入到后续的MLP中进行最终预测。

**损失函数** 深度CTR模型中广泛使用的损失函数是负对数似然函数，它使用目标商品的标签来监督整体预测：

$$
L_{target} = -\frac{1}{N} \sum_{(x,y) \in D} \big( y \log p(x) + (1-y) \log(1-p(x)) \big) \qquad (1)
$$

其中 $\mathbf{x} = [\mathbf{x}_p, \mathbf{x}_a, \mathbf{x}_c, \mathbf{x}_b] \in D$ ， $D$ 是大小为 $N$ 的训练集。 $y \in \{0, 1\}$ 表示用户是否点击目标商品。 $p(x)$ 是网络的输出，即用户点击目标商品的预测概率。

### 3.2 深度兴趣演化网络

与付费搜索不同，在许多电子商务平台（如在线展示广告）中，用户不会明确表达他们的意图，因此捕捉用户兴趣及其动态变化对CTR预测非常重要。DIEN致力于捕捉用户兴趣并建模兴趣演化过程。如图1所示，DIEN由几个部分组成。首先，所有类别的特征通过嵌入层进行变换。接下来，DIEN分两步捕捉兴趣演化：兴趣抽取层基于行为序列抽取兴趣序列；兴趣演化层建模与目标商品相关的兴趣演化过程。然后将最终的兴趣表示与广告、用户画像、上下文的嵌入向量拼接。拼接后的向量被输入MLP进行最终预测。在本节的剩余部分，我们将详细介绍DIEN的两个核心模块。

**图1: DIEN的结构。在行为层，行为按时间排序，嵌入层将独热表示 $\mathbf{b}[t]$ 转换为嵌入向量 $\mathbf{e}[t]$ 。然后兴趣抽取层借助辅助损失抽取每个兴趣状态 $\mathbf{h}[t]$ 。在兴趣演化层，AUGRU建模与目标商品相关的兴趣演化过程。最终兴趣状态 $\mathbf{h}^{\prime}[T]$ 与剩余特征的嵌入向量拼接，输入MLP进行最终CTR预测。**

#### 3.2.1 兴趣抽取层

在电子商务系统中，用户行为是潜在兴趣的载体，兴趣会在用户采取行为后发生变化。在兴趣抽取层，我们从序列化的用户行为中抽取一系列兴趣状态。

用户在电子商务系统中的点击行为非常丰富，即使在短期内（如两周），历史行为序列的长度也很长。为了平衡效率和性能，我们采用GRU来建模行为之间的依赖关系，其中GRU的输入是按发生时间排序的行为。GRU克服了RNN的梯度消失问题，并且比LSTM[7]更快，适合电子商务系统。GRU的公式如下：

$$
\mathbf{u}_t = \sigma(W^u \mathbf{i}_t + U^u \mathbf{h}_{t-1} + \mathbf{b}^u) \qquad (2)
$$

$$
\mathbf{r}_t = \sigma(W^r \mathbf{i}_t + U^r \mathbf{h}_{t-1} + \mathbf{b}^r) \qquad (3)
$$

$$
\hat{\mathbf{h}}_t = \tanh(W^h \mathbf{i}_t + \mathbf{r}_t \circ U^h \mathbf{h}_{t-1} + \mathbf{b}^h) \qquad (4)
$$

$$
\mathbf{h}_t = (1 - \mathbf{u}_t) \circ \mathbf{h}_{t-1} + \mathbf{u}_t \circ \hat{\mathbf{h}}_t \qquad (5)
$$

其中 $\sigma$ 是sigmoid激活函数， $\circ$ 是逐元素乘积， $W^u, W^r, W^h \in \mathbb{R}^{n_H \times n_I}$ ， $U^z, U^r, U^h \in \mathbb{R}^{n_H \times n_H}$ ， $n_H$ 是隐藏层大小， $n_I$ 是输入大小。 $\mathbf{i}_t$ 是GRU的输入， $\mathbf{i}_t = \mathbf{e}_b[t]$ 表示用户执行的第 $t$ 个行为， $\mathbf{h}_t$ 是第 $t$ 个隐藏状态。

然而，仅捕获行为间依赖关系的隐藏状态 $\mathbf{h}_t$ 无法有效表示兴趣。由于目标商品的点击行为是由最终兴趣触发的， $L_{target}$ 中使用的标签只包含监督最终兴趣预测的真实值，而历史状态 $\mathbf{h}_t$ （ $t < T$ ）无法获得适当的监督。众所周知，每一步的兴趣状态直接导致连续行为。因此我们提出了辅助损失，利用行为 $\mathbf{b}_{t+1}$ 来监督兴趣状态 $\mathbf{h}_t$ 的学习。除了使用真实的下一个行为作为正样本外，我们还使用从商品集中除点击商品外采样的负样本。共有 $N$ 对行为嵌入序列： $\{\mathbf{e}_b^i, \hat{\mathbf{e}}_b^i\} \in D_B, i \in 1, 2, \cdots, N$ ，其中 $\mathbf{e}_b^i \in \mathbb{R}^{T \times n_E}$ 表示点击的行为序列， $\hat{\mathbf{e}}_b^i \in \mathbb{R}^{T \times n_E}$ 表示负样本序列。 $T$ 是历史行为数量， $n_E$ 是嵌入维度， $\mathbf{e}_b^i[t] \in G$ 表示用户 $i$ 点击的第 $t$ 个商品的嵌入向量， $G$ 是全部商品集合。 $\hat{\mathbf{e}}_b^i[t]$ 表示从商品集中除用户在 $t$ 步点击的商品外采样的商品的嵌入向量。辅助损失可以表述为：

$$
L_{aux} = -\frac{1}{N} \left( \sum_{i=1}^{N} \sum_t \Big( \log \sigma(\mathbf{h}_t^i, \mathbf{e}_b^i[t+1]) + \log(1 - \sigma(\mathbf{h}_t^i, \hat{\mathbf{e}}_b^i[t+1])) \Big) \right) \qquad (6)
$$

其中 $\sigma(x_1, x_2) = \frac{1}{1+\exp(-[x_1, x_2])}$ 是sigmoid激活函数， $\mathbf{h}_t^i$ 表示用户 $i$ 的GRU的第 $t$ 个隐藏状态。我们在CTR模型中使用的全局损失为：

$$
L = L_{target} + \alpha \cdot L_{aux} \qquad (7)
$$

其中 $\alpha$ 是平衡兴趣表示和CTR预测的超参数。

借助辅助损失，每个隐藏状态 $\mathbf{h}_t$ 在用户采取行为 $\mathbf{i}_t$ 后具有足够的表达力来表示兴趣状态。所有 $T$ 个兴趣点的拼接 $[\mathbf{h}_1, \mathbf{h}_2, \cdots, \mathbf{h}_T]$ 构成了兴趣序列，兴趣演化层可以在此基础上建模兴趣演化。

总的来说，引入辅助损失有几个优点：从兴趣学习的角度来看，辅助损失的引入有助于GRU的每个隐藏状态表达性地表示兴趣。对于GRU的优化，辅助损失减少了GRU建模长历史行为序列时的反向传播难度。最后但同样重要的是，辅助损失为嵌入层的学习提供了更多的语义信息，从而得到更好的嵌入矩阵。

#### 3.2.2 兴趣演化层

由于外部环境和内部认知的共同影响，不同类型的用户兴趣随时间演化。以对衣服的兴趣为例，随着流行趋势和用户品味的改变，用户对衣服的偏好也在演化。用户对衣服兴趣的演化过程将直接决定候选衣服的CTR预测。建模演化过程的优点如下：

- 兴趣演化模块可以为最终兴趣的表示提供更多相关的历史信息；
- 通过跟随兴趣演化趋势，可以更好地预测目标商品的CTR。

值得注意的是，兴趣在演化过程中表现出两个特征：

- 由于兴趣的多样性，兴趣可能发生漂移。兴趣漂移对行为的影响是：用户可能在一段时间内对多种书籍感兴趣，而在另一段时间需要衣服。
- 尽管兴趣可能相互影响，但每个兴趣都有其自身的演化过程，例如书籍和衣服的演化过程几乎是独立的。我们只关注与目标商品相关的演化过程。

在第一阶段，借助辅助损失，我们已经获得了具有表达力的兴趣序列表示。通过分析兴趣演化的特征，我们结合了注意力机制的局部激活能力和GRU的序列学习能力来建模兴趣演化。GRU每一步的局部激活可以增强相关兴趣的影响，并削弱兴趣漂移的干扰，这有助于建模与目标商品相关的兴趣演化过程。

与公式（2-5）类似，我们使用 $\mathbf{i}^{\prime}_t$ 、 $\mathbf{h}^{\prime}_t$ 来表示兴趣演化模块中的输入和隐藏状态，其中第二个GRU的输入是兴趣抽取层中对应的兴趣状态： $\mathbf{i}^{\prime}_t = \mathbf{h}_t$ 。最后的隐藏状态 $\mathbf{h}^{\prime}_T$ 表示最终兴趣状态。

我们在兴趣演化模块中使用的注意力函数可以表述为：

$$
a_t = \frac{\exp(\mathbf{h}_t W \mathbf{e}_a)}{\sum_{j=1}^{T} \exp(\mathbf{h}_j W \mathbf{e}_a)} \qquad (8)
$$

其中 $\mathbf{e}_a$ 是广告类别中各字段嵌入向量的拼接， $W \in \mathbb{R}^{n_H \times n_A}$ ， $n_H$ 是隐藏状态的维度， $n_A$ 是广告嵌入向量的维度。注意力分数可以反映广告 $\mathbf{e}_a$ 和输入 $\mathbf{h}_t$ 之间的关系，强相关性导致大的注意力分数。

接下来，我们将介绍几种将注意力机制和GRU结合起来建模兴趣演化过程的方法。

- **带有注意力输入的GRU（AIGRU）** 为了在兴趣演化过程中激活相关兴趣，我们提出了一种简单的方法，称为带有注意力输入的GRU（AIGRU）。AIGRU使用注意力分数影响兴趣演化层的输入。如公式（9）所示：

  $$
  \mathbf{i}^{\prime}_t = \mathbf{h}_t * a_t \qquad (9)
  $$

  其中 $\mathbf{h}_t$ 是兴趣抽取层GRU的第 $t$ 个隐藏状态， $\mathbf{i}^{\prime}_t$ 是用于兴趣演化的第二个GRU的输入， $*$ 表示标量-向量乘积。在AIGRU中，不太相关的兴趣的规模可以通过注意力分数减小。理想情况下，不太相关兴趣的输入值可以减小到零。然而，AIGRU的效果不是很好。因为即使是零输入也会改变GRU的隐藏状态，所以不太相关的兴趣也会影响兴趣演化的学习。

- **基于注意力的GRU（AGRU）** 在问答领域[18]，基于注意力的GRU（AGRU）首次被提出。通过将注意力机制的信息嵌入到GRU架构中，AGRU可以有效地提取复杂查询中的关键信息。受问答系统的启发，我们将AGRU的使用从提取查询中的关键信息新颖地迁移到在兴趣演化过程中捕获相关兴趣。具体来说，AGRU使用注意力分数替换GRU的更新门，并直接改变隐藏状态。形式上：

  $$
  \mathbf{h}^{\prime}_t = (1 - a_t) * \mathbf{h}^{\prime}_{t-1} + a_t * \hat{\mathbf{h}}^{\prime}_t \qquad (10)
  $$

  其中 $\mathbf{h}^{\prime}_{t-1}$ 、 $\hat{\mathbf{h}}^{\prime}_t$ 、 $\mathbf{h}^{\prime}_t$ 是AGRU的隐藏状态。

  在兴趣演化的场景中，AGRU利用注意力分数直接控制隐藏状态的更新。AGRU削弱了兴趣演化过程中不太相关兴趣的影响。将注意力嵌入GRU提高了注意力机制的影响力，并帮助AGRU克服了AIGRU的缺陷。

- **带有注意力更新门的GRU（AUGRU）** 尽管AGRU可以直接使用注意力分数控制隐藏状态的更新，但它使用标量（注意力分数 $a_t$ ）替换向量（更新门 $\mathbf{u}_t$ ），忽略了不同维度之间重要性的差异。我们提出了带有注意力更新门的GRU（AUGRU）来无缝地结合注意力机制和GRU：

  $$
  \tilde{\mathbf{u}}^{\prime}_t = a_t * \mathbf{u}^{\prime}_t \qquad (11)
  $$

  $$
  \mathbf{h}^{\prime}_t = (1 - \tilde{\mathbf{u}}^{\prime}_t) \circ \mathbf{h}^{\prime}_{t-1} + \tilde{\mathbf{u}}^{\prime}_t \circ \hat{\mathbf{h}}^{\prime}_t \qquad (12)
  $$

  其中 $\mathbf{u}^{\prime}_t$ 是AUGRU的原始更新门， $\tilde{\mathbf{u}}^{\prime}_t$ 是我们为AUGRU设计的注意力更新门， $\mathbf{h}^{\prime}_{t-1}$ 、 $\hat{\mathbf{h}}^{\prime}_t$ 、 $\mathbf{h}^{\prime}_t$ 是AUGRU的隐藏状态。

  在AUGRU中，我们保留了更新门的原始维度信息，这决定了每个维度的重要性。基于有区别的信息，我们使用注意力分数 $a_t$ 来缩放更新门的所有维度，这使得不太相关的兴趣对隐藏状态的影响更小。AUGRU更有效地避免了兴趣漂移的干扰，并推动相关兴趣平稳演化。

---

## 4. 实验

在本节中，我们在公开数据集和工业数据集上将DIEN与最先进的方法进行比较。此外，我们设计了实验来分别验证辅助损失和AUGRU的效果。为了观察兴趣演化的过程，我们展示了兴趣隐藏状态的可视化结果。最后，我们分享了在线服务的结果和技术。

### 4.1 数据集

我们使用公开数据集和工业数据集来验证DIEN的效果。所有数据集的统计信息如表1所示。

**表1: 数据集统计**

| 数据集 | 用户 | 商品 | 类别 | 样本数 |
|--------|------|------|------|--------|
| Books | 603,668 | 367,982 | 1,600 | 603,668 |
| Electronics | 192,403 | 63,001 | 801 | 192,403 |
| Industrial dataset | 0.8 billion | 0.82 billion | 18,006 | 7.0 billion |

**公开数据集** Amazon数据集[9]由来自Amazon的产品评论和元数据组成。我们使用Amazon数据集的两个子集：Books和Electronics，来验证DIEN的效果。在这些数据集中，我们将评论视为行为，并按时间对一个用户的评论进行排序。假设用户u有T个行为，我们的目的是使用T-1个行为来预测用户u是否会撰写第T条评论中所示的评论。

**工业数据集** 工业数据集由我们在线展示广告系统的展示和点击日志构建。对于训练集，我们将最近49天内被点击的广告作为目标商品。每个目标商品及其对应的点击行为构成一个实例。以目标商品a为例，我们将a被点击的那一天设为最后一天，该用户在前14天内采取的行为作为历史行为。类似地，测试集中的目标商品从接下来的一天中选择，行为构建方式与训练数据相同。

### 4.2 对比方法

我们将DIEN与一些主流的CTR预测方法进行比较：

- **BaseModel** BaseModel采用与DIEN相同的嵌入和MLP设置，并使用求和池化操作来整合行为嵌入。
- **Wide & Deep**[1]Wide & Deep由两部分组成：其深度模型与BaseModel相同，其宽模型是线性模型。
- **PNN**[11]PNN使用产品层来捕获跨领域类别之间的交互模式。
- **DIN**[23]DIN使用注意力机制来激活相关的用户行为。
- **Two layer GRU + Attention** 类似于[10]，我们使用两层GRU来建模序列行为，并使用注意力层来激活相关行为。

### 4.3 公开数据集上的结果

总体而言，如图1所示，DIEN的结构由GRU、AUGRU、辅助损失和其他常规组件组成。在公开数据集上，每个实验重复5次。从表2可以看出，具有手动设计特征的Wide & Deep表现不佳，而特征之间的自动交互（PNN）可以提高BaseModel的性能。同时，旨在捕获兴趣的模型可以明显提高AUC：DIN激活了与广告相关的兴趣，Two layer GRU + Attention进一步激活了兴趣序列中的相关兴趣，所有这些探索都获得了积极的反馈。DIEN不仅更有效地捕获了序列兴趣，而且建模了与目标商品相关的兴趣演化过程。对兴趣演化的建模帮助DIEN获得更好的兴趣表示，并精确捕获兴趣的动态变化，从而大幅提升了性能。

**表2: 公开数据集上的结果（AUC）**

| 模型 | Electronics (mean $\pm$ std) | Books (mean $\pm$ std) |
|------|------------------------|--------------------|
| BaseModel[23] | 0.7435 $\pm$ 0.00128 | 0.7686 $\pm$ 0.00253 |
| Wide&Deep [1] | 0.7456 $\pm$ 0.00127 | 0.7735 $\pm$ 0.00051 |
| PNN[11] | 0.7543 $\pm$ 0.00101 | 0.7799 $\pm$ 0.00181 |
| DIN[23] | 0.7603 $\pm$ 0.00028 | 0.7880 $\pm$ 0.00216 |
| Two layer GRU + Attention | 0.7605 $\pm$ 0.00059 | 0.7890 $\pm$ 0.00268 |
| DIEN | **0.7792 $\pm$ 0.00243** | **0.8453 $\pm$ 0.00476** |

### 4.4 工业数据集上的结果

我们进一步在真实展示广告数据集上进行实验。工业数据集中使用了6个FCN层，维度分别为600、400、300、200、80、2，历史行为最大长度设置为50。如表3所示，Wide & Deep和PNN获得了比BaseModel更好的性能。与Amazon数据集中仅有一类商品不同，在线广告数据集同时包含所有类型的商品。基于这一特点，基于注意力的方法（如DIN）大幅提升了性能。DIEN捕获了与目标商品相关的兴趣演化过程，并获得了最佳性能。

**表3: 工业数据集上的结果（AUC）**

| 模型 | AUC |
|------|-----|
| BaseModel[23] | 0.6350 |
| Wide&Deep [1] | 0.6362 |
| PNN[11] | 0.6353 |
| DIN[23] | 0.6428 |
| Two layer GRU + Attention | 0.6457 |
| BaseModel + GRU + AUGRU | 0.6493 |
| DIEN | **0.6541** |

### 4.5 应用研究

在本节中，我们将分别展示AUGRU和辅助损失的效果。

**表4: AUGRU和辅助损失的效果（AUC）**

| 模型 | Electronics (mean $\pm$ std) | Books (mean $\pm$ std) |
|------|------------------------|--------------------|
| BaseModel | 0.7435 $\pm$ 0.00128 | 0.7686 $\pm$ 0.00253 |
| Two layer GRU + attention | 0.7605 $\pm$ 0.00059 | 0.7890 $\pm$ 0.00268 |
| BaseModel + GRU + AIGRU | 0.7606 $\pm$ 0.00061 | 0.7892 $\pm$ 0.00222 |
| BaseModel + GRU + AGRU | 0.7628 $\pm$ 0.00015 | 0.7890 $\pm$ 0.00268 |
| BaseModel + GRU + AUGRU | 0.7640 $\pm$ 0.00073 | 0.7911 $\pm$ 0.00150 |
| DIEN | **0.7792 $\pm$ 0.00243** | **0.8453 $\pm$ 0.00476** |

#### 4.5.1 带有注意力更新门的GRU（AUGRU）的效果

表4显示了不同兴趣演化方法的结果。与BaseModel相比，Two layer GRU + Attention获得了改进，但缺乏对演化的建模限制了其能力。AIGRU采用了基本思想来建模演化过程，尽管有所进步，但注意力和演化的分离在兴趣演化过程中丢失了信息。AGRU进一步尝试融合注意力和演化，但如前所述，其在GRU中的注意力不能充分利用更新门的资源。AUGRU获得了明显的改进，反映了它理想地融合了注意力机制和序列学习，并有效捕获了相关兴趣的演化过程。

#### 4.5.2 辅助损失的效果

基于使用AUGRU获得的模型，我们进一步探索辅助损失的效果。在公开数据集中，辅助损失中使用的负样本从商品集中随机采样，排除相应评论中出现的商品。对于工业数据集，展示但未被点击的广告作为负样本。如图2所示，整体损失 $L$ 和辅助损失 $L_{aux}$ 保持相似的下降趋势，这意味着CTR预测的全局损失和兴趣表示的辅助损失都发挥了作用。

**图2: 公开数据集上的学习曲线。 $\alpha$ 设为1。**
*(a) Books (b) Electronics*

如表4所示，辅助损失在公开数据集上带来了显著的改进，反映了监督信息对于序列兴趣和嵌入表示学习的重要性。对于表3所示的工业数据集，带有辅助损失的模型进一步提高了性能。然而，我们可以看到改进不如公开数据集明显。这种差异来自几个方面。首先，对于工业数据集，它有大量的样本来学习嵌入层，这使得它从辅助损失中获益较少。其次，与Amazon数据集中所有商品来自同一类别不同，工业数据集中的行为是用户在我们平台中所有场景和所有类别中点击的商品。我们的目标是预测某一场景中广告的CTR。辅助损失的监督信息可能与目标商品异构，因此辅助损失对工业数据集的影响可能小于公开数据集，而AUGRU的效果被放大了。

### 4.6 兴趣演化可视化

AUGRU中隐藏状态的动态变化可以反映兴趣的演化过程。在本节中，我们可视化这些隐藏状态，以探索不同目标商品对兴趣演化的影响。

选择的历史行为依次来自以下类别：电脑音箱、耳机、车载GPS、SD和SDHC卡、Micro SD卡、外部硬盘、耳机、手机壳。AUGRU中的隐藏状态通过主成分分析（PCA）[17]投影到二维空间。投影后的隐藏状态按顺序连接。不同目标商品激活的隐藏状态移动路径如图3(a)所示。黄色曲线（无目标）表示公式（12）中使用的注意力分数相等，即兴趣演化不受目标商品影响。蓝色曲线显示隐藏状态被来自屏幕保护膜类别的商品激活，该类别与所有历史行为的相关性较低，因此显示出与黄色曲线相似的路径。红色曲线显示隐藏状态被来自手机壳类别的商品激活，该目标商品与最后一个行为强相关，在图3(a)中移动了一大步。相应地，最后一个行为获得了较大的注意力分数，如图3(b)所示。

**图3: 兴趣演化可视化。(a) AUGRU的隐藏状态通过PCA降维到二维。不同的曲线显示相同的历史行为被不同的目标商品激活。None表示兴趣演化不受目标商品影响。(b) 面对不同的目标商品，所有历史行为的注意力分数。**

### 4.7 在线服务与A/B测试

从2018年6月7日到2018年7月12日，在淘宝的展示广告系统中进行了在线A/B测试。如表5所示，与BaseModel相比，DIEN将CTR提升了20.7%，有效每千次展示成本（eCPM）提升了17.1%。此外，DIEN将每次点击成本（PPC）降低了3.0%。目前，DIEN已在线部署并为主要流量提供服务，贡献了显著的业务收入增长。

**表5: 在线A/B测试结果**

| 模型 | CTR增益 | PPC增益 | eCPM增益 |
|------|---------|---------|----------|
| BaseModel | 0% | 0% | 0% |
| DIN[23] | + 8.9% | - 2.0% | + 6.7% |
| DIEN | **+ 20.7%** | **- 3.0%** | **+ 17.1%** |

值得注意的是，DIEN的在线服务对商业系统是一个巨大的挑战。我们的展示广告系统中的在线系统承载着非常高的流量，在流量高峰期每秒服务超过100万用户。为了保持低延迟和高吞吐量，我们部署了几项重要技术来提高服务性能：i) 元素级并行GRU和内核融合[16]，我们尽可能多地融合独立内核。此外，GRU隐藏状态的每个元素可以并行计算。ii) 批处理（Batching）：将来自不同用户的相邻请求合并为一个批次，以利用GPU的优势。iii) 使用Rocket Launching的模型压缩[22]：我们使用[22]提出的方法训练一个轻量网络，该网络具有更小的尺寸但性能接近更深更复杂的网络。例如，在Rocket Launching的帮助下，GRU隐藏状态的维度可以从108压缩到32。借助这些技术，DIEN服务的延迟可以从38.2毫秒降低到6.6毫秒，每个工作节点的QPS（每秒查询数）能力可以提升到360。

---

## 5. 结论

在本文中，我们提出了一种新的深度网络结构，即深度兴趣演化网络（DIEN），用于建模兴趣演化过程。DIEN在在线广告系统中大大提高了CTR预测的性能。具体而言，我们设计了兴趣抽取层来特别捕获兴趣序列，该层使用辅助损失为兴趣状态提供更多监督。然后我们提出了兴趣演化层，其中DIEN使用带有注意力更新门的GRU（AUGRU）来建模与目标商品相关的兴趣演化过程。借助AUGRU，DIEN可以克服兴趣漂移的干扰。对兴趣演化的建模帮助我们有效捕获兴趣，从而进一步提高CTR预测的性能。未来，我们将尝试构建更个性化的兴趣模型用于CTR预测。

---

## 参考文献

[1] Cheng, H.-T.; Koc, L.; Harmsen, J.; Shaked, T.; Chandra, T.; Aradhye, H.; Anderson, G.; Corrado, G.; Chai, W.; Ispir, M.; et al. 2016. Wide & deep learning for recommender systems. In Proceedings of the 1st Workshop on Deep Learning for Recommender Systems, 7–10. ACM.

[2] Chung, J.; Gulcehre, C.; Cho, K.; and Bengio, Y. 2014. Empirical evaluation of gated recurrent neural networks on sequence modeling. arXiv preprint arXiv:1412.3555.

[3] Friedman, J. H. 2001. Greedy function approximation: a gradient boosting machine. Annals of statistics 1189–1232.

[4] Guo, H.; Tang, R.; Ye, Y.; Li, Z.; and He, X. 2017. Deepfm: a factorization-machine based neural network for ctr prediction. In Proceedings of the 26th International Joint Conference on Artificial Intelligence, 2782–2788.

[5] He, R., and McAuley, J. 2016. Ups and downs: Modeling the visual evolution of fashion trends with one-class collaborative filtering. In Proceedings of the 25th international conference on world wide web, 507–517.

[6] Hidasi, B., and Karatzoglou, A. 2017. Recurrent neural networks with top-k gains for session-based recommendations. arXiv preprint arXiv:1706.03847.

[7] Hochreiter, S., and Schmidhuber, J. 1997. Long short-term memory. Neural computation 9(8):1735–1780.

[8] Lian, J.; Zhou, X.; Zhang, F.; Chen, Z.; Xie, X.; and Sun, G. 2018. xdeepfm: Combining explicit and implicit feature interactions for recommender systems. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining.

[9] McAuley, J.; Targett, C.; Shi, Q.; and Van Den Hengel, A. 2015. Image-based recommendations on styles and substitutes. In Proceedings of the 38th International ACM SIGIR Conference on Research and Development in Information Retrieval, 43–52. ACM.

[10] Parsana, M.; Poola, K.; Wang, Y.; and Wang, Z. 2018. Improving native ads ctr prediction by large scale event embedding and recurrent networks. arXiv preprint arXiv:1804.09133.

[11] Qu, Y.; Cai, H.; Ren, K.; Zhang, W.; Yu, Y.; Wen, Y.; and Wang, J. 2016. Product-based neural networks for user response prediction. In Proceedings of the 16th International Conference on Data Mining, 1149–1154. IEEE.

[12] Ren, K.; Fang, Y.; Zhang, W.; Liu, S.; Li, J.; Zhang, Y.; Yu, Y.; and Wang, J. 2018. Learning multi-touch conversion attribution with dual-attention mechanisms for online advertising. arXiv preprint arXiv:1808.03737.

[13] Rendle, S.; Freudenthaler, C.; Gantner, Z.; and Schmidt-Thieme, L. 2009. Bpr: Bayesian personalized ranking from implicit feedback. In Proceedings of the twenty-fifth conference on uncertainty in artificial intelligence, 452–461. AUAI Press.

[14] Rendle, S. 2010. Factorization machines. In Proceedings of the 10th International Conference on Data Mining, 995–1000. IEEE.

[15] Song, Y.; Elkahky, A. M.; and He, X. 2016. Multi-rate deep learning for temporal recommendation. In Proceedings of the 39th International ACM SIGIR conference on Research and Development in Information Retrieval, 909–912. ACM.

[16] Wang, G.; Lin, Y.; and Yi, W. 2010. Kernel fusion: An effective method for better power efficiency on multithreaded gpu. In Proceedings of the 2010 IEEE/ACM Int'l Conference on Green Computing and Communications & Int'l Conference on Cyber, Physical and Social Computing, 344–350.

[17] Wold, S.; Esbensen, K.; and Geladi, P. 1987. Principal component analysis. Chemometrics and intelligent laboratory systems 2(1-3):37–52.

[18] Xiong, C.; Merity, S.; and Socher, R. 2016. Dynamic memory networks for visual and textual question answering. In Proceedings of the 33rd International Conference on International Conference on Machine Learning, 2397–2406.

[19] Yu, F.; Liu, Q.; Wu, S.; Wang, L.; and Tan, T. 2016. A dynamic recurrent model for next basket recommendation. In Proceedings of the 39th International ACM SIGIR conference on Research and Development in Information Retrieval, 729–732. ACM.

[20] Zhang, Y.; Dai, H.; Xu, C.; Feng, J.; Wang, T.; Bian, J.; Wang, B.; and Liu, T.-Y. 2014. Sequential click prediction for sponsored search with recurrent neural networks. In Proceedings of the 28th AAAI Conference on Artificial Intelligence, 1369–1375.

[21] Zhou, C.; Bai, J.; Song, J.; Liu, X.; Zhao, Z.; Chen, X.; and Gao, J. 2018a. Atrank: An attention-based user behavior modeling framework for recommendation. In Proceedings of the 32nd AAAI Conference on Artificial Intelligence.

[22] Zhou, G.; Fan, Y.; Cui, R.; Bian, W.; Zhu, X.; and Gai, K. 2018b. Rocket launching: A universal and efficient framework for training well-performing light net. In Proceedings of the 32nd AAAI Conference on Artificial Intelligence.

[23] Zhou, G.; Zhu, X.; Song, C.; Fan, Y.; Zhu, H.; Ma, X.; Yan, Y.; Jin, J.; Li, H.; and Gai, K. 2018c. Deep interest network for click-through rate prediction. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, 1059–1068. ACM.
