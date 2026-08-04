# Deep Session Interest Network for Click-Through Rate Prediction

> Yufei Feng, Fuyu Lv, Weichen Shen, Menghan Wang, Fei Sun, Yu Zhu, Keping Yang | Alibaba Group
>
> 冯宇飞1，吕福佑1，申伟辰1，王梦涵1,2，孙飞1，朱宇1，杨克平1
>
> 1阿里巴巴集团，中国杭州；2浙江大学，中国
>
> fyf649435349@gmail.com, {fuyu.lfy, weichen.swc, xiangyu.wmh, ofey.sf, zy143829}@alibaba-inc.com, shaoyao@taobao.com



本文介绍了深度会话兴趣网络（DSIN）——一种利用用户行为序列中多个历史会话进行CTR预测的新型模型。它使用带有偏置编码的自注意力机制提取每个会话中的用户兴趣，通过Bi-LSTM建模会话兴趣的演变与交互，并用局部激活单元根据目标item自适应地聚合不同会话兴趣的影响。核心内容：

- 强调用户行为在每个会话内高度同质、在跨会话间高度异质，并提出DSIN利用多个历史会话建模
- 设计带有偏置编码的自注意力网络，获得每个会话的准确兴趣表示
- 采用Bi-LSTM捕捉历史会话兴趣之间的序列关系，并用局部激活单元按目标item聚合会话兴趣

关键发现：

- 用户行为在每个会话内高度同质，而在跨会话间高度异质
- DSIN 在广告和产品推荐数据集上均优于其他最先进的CTR模型

---

## 摘要

点击率（CTR）预测在在线广告和推荐系统等许多工业应用中发挥着重要作用。如何从用户的行为序列中捕捉用户动态演变的兴趣，仍然是CTR预测中的一个持续研究课题。然而，大多数现有研究忽略了序列的内在结构：序列由会话组成，其中会话是按发生时间分隔的用户行为。我们观察到，用户行为在每个会话内高度同质，而在跨会话间高度异质。基于这一观察，我们提出了一种名为深度会话兴趣网络（DSIN）的新型CTR模型，该模型利用用户行为序列中的多个历史会话。我们首先使用带有偏置编码的自注意力机制来提取每个会话中用户的兴趣。然后，我们应用双向LSTM来建模用户兴趣如何在会话之间演变和交互。最后，我们采用局部激活单元自适应地学习不同会话兴趣对目标item的影响。在广告和产品推荐数据集上进行了实验，DSIN在两个数据集上都优于其他最先进的模型。

## 1 引言

推荐系统（RS）在帮助用户在亚马逊和淘宝等大规模网络应用中找到他们偏好的item方面变得越来越不可或缺。通常，工业推荐系统由两个阶段组成：候选生成和候选排序[4]。候选生成阶段采用一些简单但时间高效的推荐算法（例如基于item的协同过滤[17]），从海量item集中提供一个相对较小的item集用于排序。在候选排序阶段，应用复杂但强大的模型（例如神经网络方法）对候选item进行排序，以便选择top-k个item进行推荐。在本文中，我们主要关注候选排序阶段，并将其视为点击率（CTR）预测任务。这意味着我们假设已经提供了一个相对较小的item集用于排序，我们根据其CTR得分预测对item进行排序。

图1：来自真实工业应用的一个会话示例。图片下方的数字表示点击当前item与点击第一个item之间的时间间隔，以秒为单位。会话的划分原则是：每当存在超过30分钟的时间间隔。

一些近期有效的CTR模型[4, 24, 23, 22]通过利用用户的序列行为显示出有希望的结果，这些行为反映了用户动态演变的兴趣。然而，这些模型忽略了序列的内在结构：序列由会话组成。会话是在给定时间范围内发生的一系列交互（用户行为）。我们观察到，用户行为在每个会话内高度同质，而在跨会话间高度异质。如图1所示，从一个真实工业应用中采样了一个用户，我们将她的行为序列分成3个会话。会话的划分原则是：每当存在超过30分钟的时间间隔[6]。该用户在会话1中主要浏览裤子，在会话2中主要浏览戒指，在会话3中主要浏览外套。图1中展示的现象是普遍的。它反映了一个事实：用户在一个会话中通常有一个明确的独特意图，而当用户开始一个新的会话时，他/她的兴趣可能会发生急剧变化。

受上述观察的启发，我们提出了深度会话兴趣网络（DSIN），通过利用用户的多个历史会话，在CTR预测任务中对用户的序列行为进行建模。DSIN中有三个关键组件。首先，我们自然地将用户的序列行为划分为会话，然后使用带有偏置编码的自注意力网络对每个会话进行建模。自注意力可以捕捉会话行为的内在交互/相关性，然后提取每个会话中用户的兴趣。这些不同的会话兴趣可能彼此相关，甚至遵循序列模式[16]。因此在第二部分中，我们应用双向LSTM（Bi-LSTM）来捕捉用户不同历史会话兴趣的交互和演变。由于不同的会话兴趣对目标item有不同的影响，最后我们设计了局部激活单元，根据目标item聚合它们，以形成行为序列的最终表示。

本文的主要贡献总结如下：

- 我们强调用户行为在每个会话内高度同质，在跨会话间高度异质，并提出了一种名为DSIN的新模型，该模型可以有效地建模用户的多个会话以进行CTR预测。
- 我们设计了一个带有偏置编码的自注意力网络，以获得每个会话的准确兴趣表示。然后我们采用Bi-LSTM来捕捉历史会话之间的序列关系。最后，我们采用局部激活单元进行聚合，考虑不同会话兴趣对目标item的影响。
- 在广告和产品推荐数据集上进行了两组对比实验。实验结果表明，我们提出的DSIN在CTR预测任务中优于其他最先进的模型。

本文其余部分组织如下。第2节介绍相关工作。第3节详细描述我们的DSIN模型。第4节展示我们在广告和推荐数据集上的实验结果和分析。



## 2 相关工作

在本节中，我们主要介绍CTR预测和基于会话的推荐的现有研究。

### 2.1 点击率预测

近期的CTR模型主要关注特征之间的交互。Wide&Deep [3]结合了特征的线性表示。DeepFM [7]学习特征的二阶交叉，DCN [20]应用多层残差结构来学习特征的高阶表示。AFM [21]认为并非所有特征交互都具有相同的预测能力，并使用注意力机制自动学习交叉特征的权重。总之，特征的高阶表示和交互显著提高了特征的表达能力和模型的泛化能力。

用户的序列行为隐含了用户动态演变的兴趣，并已被广泛证明在CTR预测任务中是有效的。YouTubeNet [4]通过平均池化将用户观看列表的嵌入转换为固定长度的向量。深度兴趣网络（DIN）[24]使用注意力机制学习用户历史行为相对于目标item的表示。ATRANK [22]提出了一个基于注意力的框架来建模用户异质行为之间的影响。深度兴趣演化网络（DIEN）[23]使用辅助损失来调整当前行为到下一个行为的表达，然后使用AUGRU对不同目标item的特定兴趣演化过程进行建模。对用户序列行为进行建模丰富了用户的表示，并显著提高了预测精度。

### 2.2 基于会话的推荐

会话的概念在序列推荐中经常被提及，但在CTR预测任务中较少见。基于会话的推荐受益于用户兴趣在会话中的动态演变。通用因子分解框架（GFF）[8]使用item的求和池化来表示会话。每个item有两种表示，一种表示自身，另一种表示会话的上下文。最近，基于RNN的方法[9, 10, 14]被应用于基于会话的推荐，以捕捉会话内的顺序关系。在此基础上，[13]提出了一种新颖的注意力神经网络框架（NARM），用于建模用户的序列行为并捕捉用户在当前会话中的主要目的。层次RNN [16]被提出用于跨用户历史会话传递和演变RNN的潜在隐藏状态。除了RNN，[15, 12]仅应用基于自注意力的模型来有效捕捉会话的长短期兴趣。[18]使用卷积神经网络，[2]采用用户记忆网络来增强序列模型的表达能力。



## 3 深度会话兴趣网络

在本节中，我们将详细介绍深度会话兴趣网络（DSIN）。我们首先介绍名为BaseModel的基本深度CTR模型，然后介绍DSIN中建模用户会话兴趣提取和交互的技术设计。

### 3.1 基础模型

在本节中，我们主要介绍BaseModel中的特征表示、嵌入、MLP和损失函数。

**特征表示**
信息丰富的特征在CTR预测任务中至关重要。总的来说，我们在BaseModel中使用三组特征：用户画像、item画像和用户行为。每组由一些稀疏特征组成：用户画像包含性别、城市等；item画像包含卖家ID、品牌ID等；用户行为包含用户最近点击的itemID。注意，item的附加信息可以拼接起来表示自身。

**嵌入**
嵌入是一种将大规模稀疏特征转换为低维稠密向量的常用技术。数学上，稀疏特征可以分别表示为 $E \in \mathbb{R}^{M \times d_{model}}$，其中 $M$ 是稀疏特征的大小，$d_{model}$ 是嵌入大小。通过嵌入，用户画像可以表示为 $X_U \in \mathbb{R}^{N_u \times d_{model}}$，其中 $N_u$ 是用户画像的稀疏特征数量。item画像可以表示为 $X_I \in \mathbb{R}^{N_i \times d_{model}}$，其中 $N_i$ 是item画像的稀疏特征数量。用户行为可以表示为 $S = [b_1; \cdots; b_i; \cdots; b_N] \in \mathbb{R}^{N \times d_{model}}$，其中 $N$ 是用户历史行为的数量，$b_i$ 是第 $i$ 个行为的嵌入。

**多层感知机（MLP）**
首先，来自用户画像、item画像和用户行为的稀疏特征的嵌入被拼接、展平，然后输入到具有ReLU等激活函数的MLP中。最后使用softmax函数来预测用户点击目标item的概率。

**损失函数**
负对数似然函数广泛用于CTR模型，通常定义为：

$$
L = -\frac{1}{N} \sum_{(x,y) \in \mathcal{D}} (y \log p(x) + (1-y) \log (1-p(x))) \qquad (1)
$$

其中 $\mathcal{D}$ 是训练数据集，$x$ 由 $[X_U, X_I, S]$ 表示，是网络的输入，$y \in \{0,1\}$ 表示用户是否点击了item，$p(\cdot)$ 是网络的最终输出，表示用户点击item的预测概率。

### 3.2 模型概述

在推荐系统中，用户的行为序列由多个历史会话组成。用户在不同会话中表现出不同的兴趣。同时，用户的会话兴趣彼此之间具有序列关系。DSIN被提出来提取每个会话中的用户会话兴趣，并捕捉会话兴趣的序列关系。

如图2所示，DSIN在MLP之前由两部分组成。一部分是来自用户画像和item画像的嵌入向量。另一部分对用户行为进行建模，从下到上有四个层：（1）会话划分层将用户的行为序列划分为会话；（2）会话兴趣提取层提取用户的会话兴趣；（3）会话兴趣交互层捕捉会话兴趣之间的序列关系；（4）会话兴趣激活层对用户的会话兴趣应用局部激活单元，以根据目标item加权。最后，会话兴趣激活层的输出以及用户画像和item画像的嵌入向量被输入到MLP中进行最终预测。在以下小节中，我们将详细介绍这四个层。

图2：我们提出的DSIN模型概览。总体而言，在MLP层之前，DSIN有两个主要组件。一个是稀疏特征，另一个处理用户行为序列。从下到上，用户行为序列S首先被划分为会话Q，然后添加偏置编码并通过自注意力提取为会话兴趣I。通过Bi-LSTM，我们将会话兴趣I与上下文信息混合为隐藏状态H。会话兴趣I和隐藏状态H的向量经目标item激活后，与用户画像和item画像的嵌入向量拼接、展平，然后输入到MLP层进行最终预测。

**会话划分层**
为了提取更精确的用户会话兴趣，我们将用户的行为序列 $S$ 划分为会话 $Q$，其中第 $k$ 个会话 $Q_k = [b_1; \cdots; b_i; \cdots; b_T] \in \mathbb{R}^{T \times d_{model}}$，$T$ 是我们在会话中保留的行为数量，$b_i$ 是用户在该会话中的第 $i$ 个行为。用户会话的分割发生在时间间隔超过30分钟的相邻行为之间，遵循[6]。

**会话兴趣提取层**
同一会话中的行为彼此之间密切相关。此外，用户在会话中的随意行为会使会话兴趣偏离其原始表达。为了捕捉同一会话中行为之间的内在关系并减少那些无关行为的影响，我们在每个会话中采用多头自注意力[19]机制。我们还在自注意力机制中做出了一些改进以更好地实现我们的目标。

**偏置编码。** 为了利用序列的顺序关系，自注意力机制对输入嵌入应用位置编码。此外，会话的顺序关系以及不同表示子空间中存在的偏置也需要被捕捉。因此，我们在位置编码的基础上提出了偏置编码 $BE \in \mathbb{R}^{K \times T \times d_{model}}$，其中 $BE$ 中的每个元素定义如下：

$$
BE(k,t,c) = w^K_k + w^T_t + w^C_c \qquad (2)
$$

其中 $w^K \in \mathbb{R}^K$ 是会话的偏置向量，$k$ 是会话的索引，$w^T \in \mathbb{R}^T$ 是会话中位置的偏置向量，$t$ 是会话中行为的索引，$w^C \in \mathbb{R}^{d_{model}}$ 是行为嵌入中单元位置的偏置向量，$c$ 是行为嵌入中单元的索引。添加偏置编码后，用户的行为会话 $Q$ 更新如下：

$$
Q = Q + BE \qquad (3)
$$

**多头自注意力。** 在推荐系统中，用户的点击行为受到多种因素（例如颜色、风格和价格）的影响[22]。多头自注意力可以捕捉不同表示子空间中的关系。数学上，设 $Q_k = [Q_{k1}; \cdots; Q_{kh}; \cdots; Q_{kH}]$，其中 $Q_{kh} \in \mathbb{R}^{T \times d_h}$ 是 $Q_k$ 的第 $h$ 个头，$H$ 是头的数量，$d_h = \frac{1}{h} d_{model}$。第 $h$ 个头的输出计算如下：

$$
head_h = \mathrm{Attention}(Q_{kh} W^Q, Q_{kh} W^K, Q_{kh} W^V) = \mathrm{softmax}\left(\frac{Q_{kh} W^Q (Q_{kh} W^K)^T}{\sqrt{d_{model}}}\right) Q_{kh} W^V \qquad (4)
$$

其中 $W^Q$、$W^K$、$W^V$ 是线性矩阵。然后不同头的向量被拼接并输入到前馈网络：

$$
I^Q_k = \mathrm{FFN}(\mathrm{Concat}(head_1, \cdots, head_H) W^O) \qquad (5)
$$

其中 $\mathrm{FFN}(\cdot)$ 是前馈网络，$W^O$ 是线性矩阵。我们还依次进行了残差连接和层归一化。用户的第 $k$ 个会话兴趣 $I_k$ 计算如下：

$$
I_k = \mathrm{Avg}(I^Q_k) \qquad (6)
$$

其中 $\mathrm{Avg}(\cdot)$ 是平均池化。注意，不同会话的自注意力机制中权重是共享的。

**会话兴趣交互层**
用户的会话兴趣与上下文兴趣具有序列关系。对动态变化进行建模丰富了会话兴趣的表示。Bi-LSTM [5]在捕捉序列关系方面非常出色，自然地应用于DSIN中对会话兴趣的交互进行建模。LSTM [11]记忆单元的实现如下：

$$
i_t = \sigma(W_{xi} I_t + W_{hi} h_{t-1} + W_{ci} c_{t-1} + b_i)
$$
$$
f_t = \sigma(W_{xf} I_t + W_{hf} h_{t-1} + W_{cf} c_{t-1} + b_f)
$$
$$
c_t = f_t c_{t-1} + i_t \tanh(W_{xc} I_t + W_{hc} h_{t-1} + b_c)
$$
$$
o_t = \sigma(W_{xo} I_t + W_{ho} h_{t-1} + W_{co} c_t + b_o)
$$
$$
h_t = o_t \tanh(c_t) \qquad (7)
$$

其中 $\sigma(\cdot)$ 是逻辑函数，$i$、$f$、$o$、$c$ 分别是输入门、遗忘门、输出门和细胞向量，它们与 $I_t$ 具有相同的大小。权重矩阵的形状由下标表示。双向意味着存在前向和后向RNN，隐藏状态 $H$ 计算如下：

$$
H_t = \overrightarrow{h_t^f} \oplus \overleftarrow{h_t^b} \qquad (8)
$$

其中 $\overrightarrow{h_t^f}$ 是前向LSTM的隐藏状态，$\overleftarrow{h_t^b}$ 是后向LSTM的隐藏状态。

**会话兴趣激活层**
与目标item更相关的用户会话兴趣对用户是否会点击目标item有更大的影响。用户会话兴趣的权重需要根据目标item重新分配。注意力机制[1]在源和目标之间进行软对齐，已被证明是一种有效的权重分配机制。会话兴趣相对于目标item的自适应表示计算如下：

$$
a^I_k = \frac{\exp(I_k W^I X_I)}{\sum_k \exp(I_k W^I X_I)} \qquad (9)
$$
$$
U_I = \sum_k a^I_k I_k
$$

其中 $W^I$ 具有相应的形状。类似地，与会话信息混合的会话兴趣相对于目标item的自适应表示计算如下：

$$
a^H_k = \frac{\exp(H_k W^H X_I)}{\sum_k \exp(H_k W^H X_I)} \qquad (10)
$$
$$
U_H = \sum_k a^H_k H_k
$$

其中 $W^H$ 具有相应的形状。用户画像和item画像的嵌入向量、$U_I$ 和 $U_H$ 被拼接、展平，然后输入到MLP层。



## 4 实验

在本节中，我们首先介绍实验数据集、竞争对手和评估指标。然后我们将我们提出的DSIN与竞争对手进行比较并分析结果。我们进一步通过实验讨论DSIN中关键技术设计的有效性。

### 4.1 数据集

**广告数据集**
广告数据集是由阿里妈妈（中国的一个在线广告平台）公开的数据集。它包含8天内100万用户和80万个广告的2600万条广告展示/点击日志。2017-05-06至2017-05-12的日志用于训练，2017-05-13的日志用于测试。日志中还记录了用户最近的200个行为。

**推荐数据集**
为了验证DSIN在真实工业应用中的有效性，我们在阿里巴巴的推荐数据集上进行了实验。该数据集包含8天内1亿用户和7000万item的60亿条展示/点击日志。2018-12-13至2018-12-19的日志用于训练，2018-12-20的日志用于测试。日志中还记录了用户最近的200个行为。

### 4.2 竞争对手

- **YouTubeNet.** YouTubeNet [4]是一个精心设计的模型，使用用户的观看视频序列进行YouTube中的视频推荐。它平等对待用户的历史行为，并使用平均池化操作。我们还实验了没有用户行为的YouTubeNet，以验证历史行为的有效性。
- **Wide&Deep.** Wide&Deep [3]是一个兼具记忆和泛化能力的CTR模型。它包含两部分：记忆的宽模型和泛化的深模型。
- **DIN.** 深度兴趣网络[24]充分利用了用户历史行为与目标item之间的关系。它使用注意力机制来学习用户历史行为相对于目标item的表示。
- **DIN-RNN.** DIN-RNN具有与DIN类似的结构，不同之处在于我们使用Bi-LSTM的隐藏状态，对用户的历史行为进行建模并学习上下文关系。
- **DIEN.** DIEN [23]从用户行为中提取潜在的时间兴趣，并对兴趣演化过程进行建模。辅助损失使隐藏状态更能表达潜在兴趣，AUGRU对不同目标item的特定兴趣演化过程进行建模。

### 4.3 评估指标

AUC（ROC曲线下面积）反映了模型的排序能力。其定义如下：

$$
AUC = \frac{1}{m^+ m^-} \sum_{x^+ \in \mathcal{D}^+} \sum_{x^- \in \mathcal{D}^-} (\mathbb{I}(f(x^+) > f(x^-))) \qquad (11)
$$

其中 $\mathcal{D}^+$ 是所有正样本的集合，$\mathcal{D}^-$ 是所有负样本的集合，$f(\cdot)$ 是模型对样本 $x$ 的预测结果，$\mathbb{I}(\cdot)$ 是指示函数。

### 4.4 广告和推荐数据集上的结果

广告数据集和推荐数据集上的结果如表1所示。YouTubeNet由于使用了用户行为而优于YouTubeNet-No-User-Behavior，而Wide&Deep由于结合了宽部分的记忆能力而获得更好的结果。DIN通过根据目标item激活用户行为显著提高了AUC。特别是，DIN-RNN在两个数据集上的结果都差于DIN，这是由于用户行为序列的不连续性造成的。DIEN获得了更好的结果，但辅助损失和特殊设计的AUGRU导致偏离了行为的原始表达。DSIN在两个数据集上都取得了最好的结果。它将用户的历史行为提取为会话兴趣，并对会话兴趣的动态演化过程进行建模，这两者都丰富了用户的表示。局部激活单元有助于获得用户会话兴趣相对于目标item的自适应表示。

表1：在广告和推荐数据集上的结果（AUC）

| 模型 | 广告 | 推荐 |
|------|------|------|
| YouTubeNet-NO-UB | 0.6239 | 0.6419 |
| YouTubeNet | 0.6313 | 0.6425 |
| DIN-RNN | 0.6319 | 0.6435 |
| Wide&Deep | 0.6326 | 0.6432 |
| DIN | 0.6330 | 0.6459 |
| DIEN | 0.6343 | 0.6473 |
| DSIN-PE | 0.6357 | 0.6494 |
| DSIN-BE-NO-SIIL | 0.6365 | 0.6499 |
| DSIN-BE | 0.6375 | 0.6515 |

a YouTubeNet没有用户行为。
b DSIN使用位置编码。
c DSIN使用偏置编码，没有会话兴趣交互层和相应的激活单元。
d DSIN使用偏置编码。

### 4.5 进一步讨论

**多个会话的影响**
如表1所示，结果表明DIN-RNN的性能低于DIN，而DSIN-BE的性能优于DSIN-BE-NO-SIIL。每对之间的唯一区别在于序列建模。[24]解释说，行为上的快速跳跃和突然结束导致用户行为序列数据看起来有噪声。这将导致RNN中信息传递过程中的信息丢失，并进一步混淆用户行为序列的表示。而在DSIN中，我们将用户的行为序列划分为多个会话，原因如下：（i）用户行为在每个会话中通常是同质的；（ii）用户的会话兴趣遵循序列模式，更适合序列建模。这两点都提高了DSIN的性能。

**会话兴趣交互层的影响**
如表1所示，我们对DSIN-BE和DSIN-BE-NO-SIIL进行了对比实验，其中DSIN-BE表现更好。通过会话兴趣交互层，用户的会话兴趣与上下文信息混合，变得更具表现力，从而提高了DSIN的性能。

**偏置编码的影响**
如表1所示，我们对DSIN-BE和DSIN-PE进行了对比实验，其中DSIN-BE表现更好。与二维位置编码不同，用户会话的偏置也被捕捉到了。经验上，偏置编码成功捕捉了会话的顺序信息，提高了DSIN的性能。

**自注意力和激活单元的可视化**
如图3所示，我们可视化了局部激活单元和自注意力机制中的注意力权重。为了说明自注意力的效果，我们以第一个会话为例。该用户主要浏览与裤子相关的item，偶尔浏览与外套相关的item。我们可以观察到，与裤子相关的item的权重普遍较高。经过自注意力后，大多数与裤子相关的行为表示被保留并提取为该用户在本会话中的兴趣。此外，局部激活单元通过使与会话兴趣相关的目标item更加突出而起作用。在这种情况下，目标item是一条黑色裤子。用户与裤子相关的会话兴趣被赋予更大的权重，对最终预测有更大的影响。而会话3是与外套相关的，用户在该会话中对黑色的颜色偏好也有助于对裤子进行排序。

图3：该图可视化了DSIN中下半部分自注意力机制和上半部分激活单元的注意力权重。注意，自注意力机制中的注意力权重是每个头中注意力权重的总和。同时，线条颜色越深，权重越大。

## 5 结论

在本文中，我们为CTR预测任务提供了一个新的视角，其中用户的序列行为由多个历史会话组成。用户行为在每个会话内高度同质，在不同会话间高度异质。基于这些观察，我们提出了深度会话兴趣网络（DSIN）。我们首先使用带有偏置编码的自注意力机制来提取每个会话的用户兴趣。然后我们应用Bi-LSTM来捕捉上下文会话兴趣的序列关系。最后，我们采用局部激活单元根据目标item聚合用户的不同会话兴趣表示。实验结果证明了DSIN在广告和推荐数据集上的有效性。未来，我们将关注利用知识图谱作为先验知识来解释用户的历史行为，以实现更好的CTR预测。

## 参考文献

[1] Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural machine translation by jointly learning to align and translate. arXiv preprint arXiv:1409.0473, 2014.

[2] Xu Chen, Hongteng Xu, Yongfeng Zhang, Jiaxi Tang, Yixin Cao, Zheng Qin, and Hongyuan Zha. Sequential recommendation with user memory networks. In Proceedings of the eleventh ACM international conference on web search and data mining, pages 108–116. ACM, 2018.

[3] Heng-Tze Cheng, Levent Koc, Jeremiah Harmsen, Tal Shaked, Tushar Chandra, Hrishi Aradhye, Glen Anderson, Greg Corrado, Wei Chai, Mustafa Ispir, et al. Wide & deep learning for recommender systems. In Proceedings of the 1st Workshop on Deep Learning for Recommender Systems, pages 7–10. ACM, 2016.

[4] Covington, Paul, Adams, Jay, Sargin, and Emre. Deep neural networks for youtube recommendations. In Proceedings of the 10th ACM Conference on Recommender Systems, pages 191–198. ACM, 2016.

[5] Alex Graves and Jrgen Schmidhuber. Framewise phoneme classification with bidirectional lstm and other neural network architectures. Neural Networks, 18(5-6):602–610, 2005.

[6] Mihajlo Grbovic and Haibin Cheng. Real-time personalization using embeddings for search ranking at airbnb. In SIGKDD, pages 311–320. ACM, 2018.

[7] Huifeng Guo, Ruiming Tang, Yunming Ye, Zhenguo Li, and Xiuqiang He. Deepfm: a factorization-machine based neural network for ctr prediction. arXiv preprint arXiv:1703.04247, 2017.

[8] Balzs Hidasi and Domonkos Tikk. General factorization framework for context-aware recommendations. Data Mining and Knowledge Discovery, 30(2):342–371, 2016.

[9] Balzs Hidasi, Alexandros Karatzoglou, Linas Baltrunas, and Domonkos Tikk. Session-based recommendations with recurrent neural networks. arXiv preprint arXiv:1511.06939, 2015.

[10] Bal´azs Hidasi, Massimo Quadrana, Alexandros Karatzoglou, and Domonkos Tikk. Parallel recurrent neural network architectures for feature-rich session-based recommendations. In Proceedings of the 10th ACM Conference on Recommender Systems, pages 241–248. ACM, 2016.

[11] Sepp Hochreiter and Jrgen Schmidhuber. Long short-term memory. Neural computation, 9(8):1735–1780, 1997.

[12] Wang-Cheng Kang and Julian McAuley. Self-attentive sequential recommendation. In 2018 IEEE International Conference on Data Mining (ICDM), pages 197–206. IEEE, 2018.

[13] Jing Li, Pengjie Ren, Zhumin Chen, Zhaochun Ren, Tao Lian, and Jun Ma. Neural attentive session-based recommendation. In Proceedings of the 2017 ACM on Conference on Information and Knowledge Management, pages 1419–1428. ACM, 2017.

[14] Zhi Li, Hongke Zhao, Qi Liu, Zhenya Huang, Tao Mei, and Enhong Chen. Learning from history and present: Next-item recommendation via discriminatively exploiting user behaviors. In SIGKDD, pages 1734–1743. ACM, 2018.

[15] Qiao Liu, Yifu Zeng, Refuoe Mokhosi, and Haibin Zhang. Stamp: short-term attention/memory priority model for session-based recommendation. In SIGKDD, pages 1831–1839. ACM, 2018.

[16] Massimo Quadrana, Alexandros Karatzoglou, Bal´azs Hidasi, and Paolo Cremonesi. Personalizing session-based recommendations with hierarchical recurrent neural networks. In the Eleventh ACM Conference, 2017.

[17] Badrul Munir Sarwar, George Karypis, Joseph A Konstan, John Riedl, et al. Item-based collaborative filtering recommendation algorithms. Www, 1:285–295, 2001.

[18] Jiaxi Tang and Ke Wang. Personalized top-n sequential recommendation via convolutional sequence embedding. In Proceedings of the Eleventh ACM International Conference on Web Search and Data Mining, pages 565–573. ACM, 2018.

[19] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, ukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems, pages 5998–6008, 2017.

[20] Ruoxi Wang, Bin Fu, Gang Fu, and Mingliang Wang. Deep & cross network for ad click predictions. In Proceedings of the ADKDD'17, page 12. ACM, 2017.

[21] Jun Xiao, Hao Ye, Xiangnan He, Hanwang Zhang, Fei Wu, and Tat-Seng Chua. Attentional factorization machines: Learning the weight of feature interactions via attention networks. arXiv preprint arXiv:1708.04617, 2017.

[22] Chang Zhou, Jinze Bai, Junshuai Song, Xiaofei Liu, Zhengchao Zhao, Xiusi Chen, and Jun Gao. Atrank: An attention-based user behavior modeling framework for recommendation. In Thirty-Second AAAI Conference on Artificial Intelligence, 2018.

[23] Guorui Zhou, Na Mou, Ying Fan, Qi Pi, Weijie Bian, Chang Zhou, Xiaoqiang Zhu, and Kun Gai. Deep interest evolution network for click-through rate prediction. arXiv preprint arXiv:1809.03672, 2018.

[24] Guorui Zhou, Xiaoqiang Zhu, Chenru Song, Ying Fan, Han Zhu, Xiao Ma, Yanghui Yan, Junqi Jin, Han Li, and Kun Gai. Deep interest network for click-through rate prediction. In SIGKDD, pages 1059–1068. ACM, 2018.
