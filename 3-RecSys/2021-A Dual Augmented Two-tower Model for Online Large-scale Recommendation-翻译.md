# 面向在线大规模推荐的双重增强双塔模型

> Yantao Yu, Weipeng Wang, Zhoutian Feng, Daiyue Xue | Meituan

本文介绍了 面向在线大规模推荐的双重增强双塔模型。核心内容：

- 提出双重增强双塔模型（DAT），为每个查询和item定制增强向量作为输入特征，缓解双塔之间缺乏信息交互的问题
- 设计自适应模仿机制（AMM），结合模仿损失与停止梯度策略，根据另一个塔对正标签样本的输出表示向量更新增强向量
- 提出类别对齐损失（CAL），对齐主要类别与其他类别item表示的二阶统计量（协方差），缓解类别数据不平衡问题
- 在美团和Amazon Books两个大规模数据集上进行离线评估，并在日服务6000万用户的推荐系统上完成在线A/B测试

关键发现：

- DAT在两个数据集上相比于最佳基线的HitRate@100分别提升4.84%和6.63%
- 在线实验中，DAT较双塔模型基线在CTR和GMV上分别取得4.17%和3.46%的总体平均提升
- 借助AMM利用两个塔之间的信息交互，DAT(w/o CAL)已优于MIND和双塔模型，CAL进一步带来性能提升
- 增强向量在任意维度下均能取得更好的性能，验证了增强向量对建模信息交互的有效性

---

## 摘要

许多现代推荐系统拥有非常大的语料库，一种处理大规模检索的常见工业方案是使用双塔模型从其内容特征中学习查询和item表示。然而，该模型存在两个塔之间缺乏信息交互的问题。此外，不平衡的类别数据也阻碍了模型性能。在本文中，我们提出了一种名为双重增强双塔模型（DAT，Dual Augmented Two-tower Model）的新颖模型，该模型集成了新颖的自适应模仿机制（AMM，Adaptive-Mimic Mechanism）和类别对齐损失（CAL，Category Alignment Loss）。我们的AMM为每个查询和item定制一个增强向量，以缓解信息交互的缺乏。此外，我们的CAL可以通过对齐不均匀类别中的item表示来进一步提高性能。我们在大规模数据集上进行了离线实验，以展示DAT的卓越性能。此外，在线A/B测试证实DAT能够为工业应用带来改进的推荐质量。

**CCS概念**：信息系统 $\rightarrow$ 推荐系统；信息检索。

**关键词**：Recommender Systems, Information Retrieval, Neural Networks

## 1 引言

推荐系统在从海量item中筛选出用户感兴趣的item方面不可或缺。大规模推荐中最关键的挑战之一是实时准确地评分数百万或数十亿个item。一种常见做法是将推荐器设计为两阶段架构，其中检索模型首先从大型语料库中根据用户查询检索出一小部分相关item，然后排序模型根据点击或用户评分对检索到的item进行排序 [2]。显然，检索阶段检索到的候选质量在整个系统中起着关键作用。在这项工作中，我们专注于检索问题，旨在改进针对具有数百万查询和item的个性化推荐的检索系统性能。

一个可扩展的检索模型通常首先学习查询和item表示，然后使用查询和item表示之间的余弦相似度来获得为查询定制的推荐。然而，在工业规模的应用程序中，item语料库可能非常庞大，并且从用户反馈中收集的训练数据对于大多数查询和item来说可能非常稀疏，这可能导致对长尾用户和item的模型预测不准确。为了应对上述挑战，通常采用双塔模型 [4]，其中塔指的是基于深度神经网络（DNNs）的编码器。尽管前景广阔，但双塔模型仍然存在一些问题。由于item塔的item表示必须为在线检索服务单独预先计算，因此item塔的前向计算必须与查询塔独立，因此模型缺乏两个塔之间的信息交互，这可能不可避免地阻碍模型的性能。在现实世界的应用中，一个塔的输入与另一个塔的输入对于每次点击都存在正向交互。例如，假设item $A$ 在查询 $\{B, C, D\}$ 中被点击。显然，查询塔的输入是 $\{B, C, D\}$，这些与item塔的输入 $A$ 交互。因此，查询塔中包含的信息可以被利用来增强item塔的item表示，反之亦然。此外，item的类别多种多样（例如，食品、酒店、电影等），并且每个类别中的item数量严重不平衡。因此，单个类别中的item可能占大多数。因此，模型在item数量相对较少的类别上表现要差得多。

在本文中，为了解决上述问题，我们提出了一种新颖的用于大规模推荐的检索模型，名为双重增强双塔模型（DAT，Dual Augmented Two-tower Model）。具体来说，我们设计了一种自适应模仿机制（AMM，Adaptive-Mimic Mechanism），为每个查询和item定制一个增强向量作为其内容特征。增强向量根据另一个塔对每个正标签样本的输出表示向量进行更新。通过这种方式，作为输入特征的增强向量携带了另一个塔的有价值信息，从而隐式地建模了两个塔之间的信息交互。我们还在训练阶段引入了类别对齐损失（CAL，Category Alignment Loss），以对齐来自不同类别的item表示。综合实验表明，我们的DAT特征具有两个主要优势：i). 它为检索任务中双塔模型的信息交互提供了更深入的见解；ii). 当类别分布极端不平衡时，它能产生更好的item表示。

**图1：我们提出的双重增强双塔模型的网络架构**

## 2 模型架构

### 2.1 问题陈述

我们考虑一个推荐系统，包含一个查询集 $\{u_i\}_{i=1}^{N}$ 和一个item集 $\{v_j\}_{j=1}^{M}$，其中 $N$ 是用户数量，$M$ 是item数量。这里的 $u_i$、$v_j$ 是各种特征（例如，ID和内容特征）的拼接，由于稀疏性，这些特征可能非常高维。查询-item反馈可以用矩阵 $R \in \mathbb{R}^{N \times M}$ 表示，其中如果查询 $i$ 对item $j$ 给出正向反馈，则 $R_{ij}=1$，否则 $R_{ij}=0$。我们的目标是针对给定的查询，从整个item语料库中高效地选出可能的数千个候选item。

### 2.2 双重增强双塔模型

我们提出模型的框架如图1所示。DAT模型使用增强向量 $a_u$（$a_v$）来捕获来自另一个塔的信息，并将该向量视为一个塔的输入特征。此外，类别对齐损失将数据量最大的类别中学到的知识迁移到其他类别。

#### 2.2.1 嵌入层

类似于双塔模型，$u_i$ 和 $v_j$ 中的每个特征 $f_i \in \mathbb{R}$（例如，一个item ID）通过嵌入层映射到一个低维稠密向量 $e_i \in \mathbb{R}^K$，其中 $K$ 是嵌入维度。具体来说，我们定义嵌入矩阵 $E \in \mathbb{R}^{K \times D}$，其中 $E$ 待学习，$D$ 是唯一特征的数量，嵌入向量 $e_i$ 是嵌入矩阵 $E$ 的第 $i$ 列。

#### 2.2.2 双重增强层

对于某个查询和候选item，我们通过其ID创建两个对应的增强向量 $a_u$ 和 $a_v$，并将它们与特征嵌入向量拼接，以获得两个塔的增强输入向量 $z_u$、$z_v$。例如，如果查询 $u$ 具有特征"uid=253,city=SH,gender=male,..."，item $v$ 具有特征"iid=149,price=10,class=cate,..."，我们有：

$$
\begin{aligned}
z_u &= [e_{253} \parallel e_{sh} \parallel e_{male} \parallel \cdots \parallel a_u], \\
z_v &= [e_{149} \parallel e_{p10} \parallel e_{cate} \parallel \cdots \parallel a_v]
\end{aligned}
$$

其中 $\parallel$ 是向量拼接操作。增强输入向量 $z_u$ 和 $z_v$ 不仅包含当前查询和item的信息，还通过 $a_u$ 和 $a_v$ 包含历史正向交互的信息。

接下来，我们将 $z_u$ 和 $z_v$ 输入两个塔中，这两个塔由具有ReLU激活函数的全连接层组成，以便通过 $a_u$ 和 $a_v$（增强向量）实现两个塔之间的信息交互。接下来，全连接层的输出经过L2归一化层，以获得查询 $p_u$ 和item $p_v$ 的增强表示。形式上，这两个步骤的定义如下：

$$
\begin{aligned}
h_1 &= \mathrm{ReLU}(W_1 z + b_1), \ldots \\
h_L &= \mathrm{ReLU}(W_l h_{L-1} + b_l), \\
p &= \mathrm{L2Norm}(h_L)
\end{aligned}
\qquad (1)
$$

其中 $z$ 表示 $z_u$ 和 $z_v$，$p$ 表示 $p_u$ 和 $p_v$；$W_l$ 和 $b_l$ 分别是第 $l$ 层的权重矩阵和偏置向量；$p_u$ 和 $p_v$ 是L2归一化层的输出向量，分别表示查询嵌入和item嵌入。两个塔具有相同的结构但参数不同。

此外，为了估计增强向量 $a_u$ 和 $a_v$，我们设计了一种自适应模仿机制（AMM），该机制集成了模仿损失和停止梯度策略。模仿损失旨在使用增强向量来拟合属于相应查询或item的另一个塔中的所有正向交互。我们将模仿损失定义为对于每个标签等于1的样本，增强向量与查询/item嵌入 $p_u$、$p_v$ 之间的均方误差：

$$
\begin{aligned}
\mathrm{loss}_u &= \frac{1}{T} \sum_{(u,v,y) \in T} [y a_u + (1-y) p_v - p_v]^2, \\
\mathrm{loss}_v &= \frac{1}{T} \sum_{(u,v,y) \in T} [y a_v + (1-y) p_u - p_u]^2
\end{aligned}
\qquad (2)
$$

其中 $T$ 是训练数据集 $T$ 中查询-item对的数量，$y \in \{0, 1\}$ 是标签。我们将在下一个子节中讨论标注过程。可以看出，如果标签 $y = 1$，$a_v$ 和 $a_u$ 逼近查询嵌入 $p_u$ 和item嵌入 $p_v$；如果标签 $y = 0$，损失等于0。如图1所示，增强向量在一个塔中使用，而查询/item嵌入从另一个塔生成。也就是说，增强向量 $a_u$ 和 $a_v$ 总结了关于一个查询或item可能从另一个塔匹配什么的高层信息。由于模仿损失用于更新 $a_u$ 和 $a_v$，我们应该冻结 $p_u$ 和 $p_v$ 的值。为此，应用停止梯度策略来阻止 $\mathrm{loss}_u$ 和 $\mathrm{loss}_v$ 的梯度回流到 $p_v$ 和 $p_u$。

一旦获得两个增强向量 $a_u$ 和 $a_v$，它们就被用作两个塔的输入特征来建模两个塔之间的信息交互。最后，模型的输出是查询嵌入和item嵌入的内积：

$s(u, v) = \langle p_u, p_v \rangle$

其中 $s(u, v)$ 表示我们的检索模型提供的分数。

#### 2.2.3 类别对齐

在工业场景中，item的类别多种多样（例如，食品、酒店、电影等），并且每个类别中的item数量严重不均匀。在不平衡的类别数据下，双塔模型对不同类别的表现不同，并且在item数量相对较少的类别上表现要差得多。为了解决这个问题，我们提出了一种用于训练阶段的类别对齐损失（CAL），它将数据量大的类别中学到的知识迁移到其他类别。具体来说，对于每个批次，数据量最大的类别的item表示 $p_v$ 被取出形成主要类别集：$S_{\mathrm{major}} = \{p_v^{\mathrm{major}}\}$，而其他类别的 $p_v$ 形成它们各自的类别集：$S_2, S_3, S_4, \ldots$。我们将类别对齐损失定义为主要类别与其他类别特征之间的二阶统计量（协方差）之间的距离：

$$
\mathrm{loss}_{CA} = \sum_{i=2}^{n} \| C(S_{\mathrm{major}}) - C(S_i) \|_F^2
\qquad (3)
$$

其中 $\|\cdot\|_F^2$ 表示平方矩阵Frobenius范数，$n$ 是类别数量。$C(\cdot)$ 表示协方差矩阵。

### 2.3 模型训练

我们将检索问题视为一个二分类问题，并采用随机负采样框架。具体来说，对于每个正向查询-item对（标签=1）中的查询，我们从item语料库中随机采样 $S$ 个item，以创建与该查询的 $S$ 个负查询-item对（标签=0），并将这 $S+1$ 个对添加到训练数据集中。这些对的交叉熵损失如下：

$$
\begin{aligned}
\mathrm{loss}_p &= -\frac{1}{T} \sum_{(u,v,y) \in T} \left( y \log \sigma(\langle p_u, p_v \rangle) + (1-y) \log(1 - \sigma(\langle p_u, p_v \rangle)) \right), \\
T &= D \times (S+1)
\end{aligned}
\qquad (4)
$$

其中 $D$ 是正向反馈查询-item对的数量，$T$ 是训练对的总数。$\sigma(\cdot)$ 表示sigmoid函数。

最终损失函数公式化为：

$$
\mathrm{loss} = \mathrm{loss}_p + \lambda_1 \mathrm{loss}_u + \lambda_2 \mathrm{loss}_v + \lambda_3 \mathrm{loss}_{CA}
\qquad (5)
$$

其中 $\lambda_1$、$\lambda_2$、$\lambda_3$ 是可调参数。

**表1：数据集统计**

| 数据集 | #用户 | #item | #交互 | #类别 |
|--------|-------|-------|-------|-------|
| Amazon Books | 695,513 | 243,166 | 6,706,125 | 11 |
| Meituan | 82,347,274 | 3,561,498 | 1,182,652,197 | 9 |

## 3 实验

在本节中，我们进行了广泛的在线和离线实验，以证明DAT设计的合理性。

### 3.1 数据集

所有模型在两个离线大规模数据集上进行了评估：一个从美团 [^1] 在线系统的日常日志中采样的大型数据集，以及一个来自Amazon的数据集 [3]。美团数据集包含连续11天的数据，其中前10天的数据用于训练，第11天的数据用于测试，我们将前10天出现的所有item合并作为item语料库。Amazon Books相对较小，我们只保留至少被评论5次的item和至少评论过5个item的用户。我们保留最后一个item用于测试。两个离线数据集的详细信息列于表1。

### 3.2 实验设置

将广泛应用于工业的以下方法与提出的DAT模型进行了比较：

- WALS [1]：一种矩阵分解算法，用于将交互矩阵分解为用户和item的隐因子。
- YouTubeDNN [2]：一种基于深度神经网络的推荐方法，将向量输入多层前馈神经网络。
- FM [7]：一种模型，累积查询和item的特征向量并将其输入FM层。
- 双塔模型 [4]：检索任务中的流行模型，已被广泛引入以利用丰富的内容特征。
- MIND [6]：一种用于工业检索任务的最新最先进模型。对于美团数据集和Amazon数据集，用户兴趣数量分别设置为4和5。

我们通过分布式TensorFlow实现了这些模型，并使用Faiss [5] 从大规模item池中检索Top-N item。所有模型的嵌入维度和批次大小分别固定为32和256。所有模型均使用Adam优化器训练。为确保公平比较，所有模型的其他超参数都单独调整以获得最优结果。对于DAT，每个塔中的FC层数固定为3，维度分别为256、128和32。增强向量 $a_u$ 和 $a_v$ 的维度均设置为 $d = 32$，而 $\lambda_1$、$\lambda_2$ 设置为0.5，$\lambda_3$ 设置为1。为了评估各种模型的离线效果，我们采用了HitRate@$K$ 和MRR指标，这些指标在工业检索中被广泛使用。$K$ 设置为50和100，因为检索模块需要检索相对大量的候选item以馈送给排序模块。由于测试实例规模较大，我们采用了MRR的缩放版本，缩放因子为10。

### 3.3 离线结果

#### 3.3.1 模型比较

DAT和基线模型在两个数据集上的实验结果报告在表2中。最佳结果以粗体列出，基线模型的最佳结果以星号(*)标记。显然，DAT在所有基线模型中表现最佳，在两个数据集上相比于最佳基线的HitRate@100分别提升了4.84%和6.63%。这证明了DAT的有效性，并进一步证明了自适应模仿机制（AMM）和类别对齐损失（CAL）的重要性。WALS，即矩阵分解方法，与其他方法相比表现较差，这证实了深度学习在推荐系统检索阶段的有效性。FM相比于YouTubeDNN获得的改进凸显了特征交互的优势。此外，在深度学习的帮助下，双塔模型的表现远优于FM和YouTubeDNN，这可以解释为它能够从更有价值的内容特征中学习查询和item表示。此外，考虑到用户具有多种兴趣，MIND的表现优于双塔模型。最后，DAT(w/o CAL)相较于MIND和双塔模型的优势可以归因于AMM能够利用两个塔之间的信息交互。

**表2：两个数据集上的HitRate和MRR性能比较（w/o是without的缩写）**

| 模型 | Meituan | | | Amazon | | |
|------|---------|---|---|--------|---|---|
| | HR@50 | HR@100 | MRR | HR@50 | HR@100 | MRR |
| WALS | 0.2917 | 0.4146 | 0.3375 | 0.0242 | 0.0359 | 0.0351 |
| FM | 0.4831 | 0.6672 | 0.5012 | 0.0406 | 0.0634 | 0.0589 |
| YouTubeDNN | 0.4228 | 0.5142 | 0.4512 | 0.0378 | 0.0599 | 0.0524 |
| 双塔模型 | 0.5395 | 0.7159 | 0.5472 | 0.0464 | 0.0732 | 0.0625 |
| MIND | 0.5507* | 0.7327* | 0.5843* | 0.0490* | 0.0784* | 0.0673* |
| DAT | 0.5796 | 0.7682 | 0.6154 | 0.0519 | 0.0836 | 0.0711 |
| DAT(w/o CAL) | 0.5655 | 0.7512 | 0.6009 | 0.0503 | 0.0816 | 0.0698 |

#### 3.3.2 增强向量的维度

DAT中的增强向量在建模信息交互中起着关键作用。为了分析维度的影响，我们研究了DAT在两个数据集上关于增强向量维度的性能。如图2所示，DAT在美团上的性能随着维度的增加而提高，而DAT在Amazon上的性能先提高后下降。这是由于两个数据集之间数据量的差异造成的。此外，无论维度如何，它总能取得更好的性能，这证明了增强向量的有效性。

**图2：在两个数据集上，HR@100和MRR随增强向量维度的变化**

### 3.4 在线实验

除了离线研究，我们还进行了在线实验，将DAT部署到一个日服务6000万用户的推荐系统中处理为期一周的真实流量。为了进行公平比较，检索阶段之后采用相同的排序流程。我们使用CTR（Click-Through Rate，点击率）和GMV（Gross Merchandise Volume，成交总额）这两个广泛使用的工业指标来衡量模型服务在线流量的性能。在线实验的基线方法是双塔模型，它是服务大部分在线流量的基础检索算法。每种方法检索出一百个候选item并馈送到排序阶段。图3显示了连续7天的在线结果。我们的模型以较大幅度优于基线，在CTR和GMV方面分别取得了4.17%和3.46%的总体平均提升。

**图3：DAT与基线的在线性能**

## 4 结论

在本文中，我们提出了一种名为双重增强双塔模型（DAT）的有效检索模型，用于工业推荐系统。它旨在建模两个塔之间的信息交互，并为不平衡的类别数据产生更好的item表示。在离线和在线数据集上的大量实验表明，具有AMM和CAL的DAT模型能够有效实现卓越的性能。

## 参考文献

[1] Christopher R Aberger. 2014. Recommender: An analysis of collaborative filtering techniques. Personal and Ubiquitous Computing Journal (2014).

[2] Paul Covington, Jay Adams, and Emre Sargin. 2016. Deep neural networks for youtube recommendations. In Proceedings of the 10th ACM conference on recommender systems. 191–198.

[3] Ruining He and Julian McAuley. 2016. Ups and downs: Modeling the visual evolution of fashion trends with one-class collaborative filtering. In proceedings of the 25th international conference on world wide web. 507–517.

[4] Po-Sen Huang, Xiaodong He, Jianfeng Gao, Li Deng, Alex Acero, and Larry Heck. 2013. Learning deep structured semantic models for web search using clickthrough data. In Proceedings of the 22nd ACM international conference on Information & Knowledge Management. 2333–2338.

[5] Jeff Johnson, Matthijs Douze, and Hervé Jégou. 2017. Billion-scale similarity search with GPUs. arXiv preprint arXiv:1702.08734 (2017).

[6] Chao Li, Zhiyuan Liu, Mengmeng Wu, Yuchi Xu, Huan Zhao, Pipei Huang, Guoliang Kang, Qiwei Chen, Wei Li, and Dik Lun Lee. 2019. Multi-interest network with dynamic routing for recommendation at Tmall. In Proceedings of the 28th ACM International Conference on Information and Knowledge Management. 2615–2623.

[7] Steffen Rendle. 2010. Factorization Machines. In ICDM 2010, The 10th IEEE International Conference on Data Mining, Sydney, Australia, 14-17 December 2010.

[^1]: https://www.meituan.com/
