# DSSM: Learning Deep Structured Semantic Models for Web Search using Clickthrough Data

> Po-Sen Huang | University of Illinois at Urbana-Champaign; Xiaodong He, Jianfeng Gao, Li Deng, Alex Acero, Larry Heck | Microsoft Research
> 何晓东、高建峰、邓力



本文提出了深度结构化语义模型（DSSM，Deep Structured Semantic Models），用于基于点击数据的Web搜索语义匹配。核心内容：

- **深度语义模型**：使用深度神经网络将查询和文档映射到共同的低维语义空间，通过余弦相似度计算相关性
- **词哈希技术**：基于字母n-gram的词哈希方法，**有效处理大规模词汇表**，解决传统词向量维度灾难问题
- **判别式训练**：利用点击数据最大化 **给定查询下被点击文档的条件似然**，直接优化文档排序目标
- **实验验证**：在真实Web搜索数据集上，最佳模型在NDCG@1上显著优于其他latent语义模型2.5-4.3%

关键发现：深度架构结合 点击数据监督训练 和 词哈希技术，能够学习到比传统LSA、PLSA等更有效的语义表示。

---



## 摘要

latent语义模型（如潜在语义分析 LSA，Latent Semantic Analysis）旨在**将查询映射到其相关文档的语义层面**，而**基于关键词的匹配在该层面常常失败**。在本研究中，我们致力于开发一系列**具有深度结构的新型latent语义模型**，将查询和文档投影到一个共同的低维空间，在该空间中给定查询的文档相关性可以方便地计算为它们之间的距离。所提出的深度结构化语义模型通过使用 点击数据 最大化给定查询下被点击文档的条件似然来进行 判别式训练。为了使我们的模型适用于**大规模Web搜索应用**，我们还使用了一种称为词哈希的技术，该技术被证明能有效地将我们的语义模型扩展到处理此类任务中常见的大规模词汇表。新的模型使用真实数据集在Web文档排序任务上进行了评估。结果表明，我们最佳模型显著优于其他latent语义模型，这些模型在本文工作之前被认为是性能最先进的。

**类别和主题描述符**：H.3.3 [信息存储与检索]：信息搜索与检索；I.2.6 [人工智能]：学习

**通用术语**：算法、实验

**关键词**：深度学习、语义模型、点击数据、Web搜索



## 1. 引言

现代搜索引擎主要通过将文档中的关键词与搜索查询中的关键词进行匹配来检索Web文档。然而，词汇匹配可能不准确，因为一个概念在文档和查询中通常使用不同的词汇和语言风格来表达。

latent语义模型（如潜在语义分析 LSA，Latent Semantic Analysis）能够将查询映射到与其相关文档的语义层面，而基于关键词的词汇匹配在该层面常常失败[2,6,8,15,21]。这些latent语义模型通过 **将出现在相似上下文中的不同术语** 分组 到同一语义簇中 来解决Web文档与搜索查询之间的语言差异。因此，即使查询与文档没有任何共享术语，它们在低维语义空间中的两个向量仍可能具有较高的相似度分数。

从LSA扩展而来，**概率主题模型**（如概率LSA（PLSA，Probabilistic Latent Semantic Analysis）和潜在狄利克雷分配（LDA，Latent Dirichlet Allocation））也被提出用于语义匹配[2,15]。然而，这些模型通常以**无监督方式训练**，使用的目标函数仅与检索任务的评估指标松散耦合。因此，这些模型在Web搜索任务上的性能并不像最初预期的那样好。

最近，两条研究路线被开展以扩展上述latent语义模型，下面将简要回顾。

首先，点击数据（由查询列表及其被点击的文档组成）被用于语义建模，以桥接搜索查询和Web文档之间的语言差异[9,10]。例如，Gao等人[10]提出了使用双语主题模型（BLTMs，Bilingual Topic Models）和线性判别投影模型（DPMs，Discriminative Projection Models）在语义层面进行查询-文档匹配。这些模型在点击数据上训练，使用针对文档排序任务定制的目标函数。更具体地说，BLTM是一种生成模型，要求查询及其被点击的文档不仅共享相同的主题分布，还包含分配给每个主题的相似比例的单词。相比之下，DPM使用S2Net算法[26]学习，该算法遵循[3]中概述的 **pairwise成对学习排序** 范式。将查询和文档的术语向量投影到低维语义空间中的概念向量后，查询与其被点击文档的概念向量之间的距离 小于 查询与其未点击文档之间的距离。Gao等人[10]报告称，BLTM和DPM在文档排序任务上都显著优于无监督latent语义模型，包括LSA和PLSA。然而，BLTM的训练虽然使用了点击数据，但目的是最大化对数似然标准，这对于文档排序的评估标准来说是次优的。另一方面，DPM的训练涉及大规模矩阵乘法。这些矩阵的大小通常随词汇表大小快速增长，在Web搜索任务中可能达到数百万量级。为了使训练时间可接受，词汇表被大幅裁剪。虽然小词汇表使模型可训练，但导致了次优性能。

在第二条研究路线中，Salakhutdinov和Hinton使用深度自编码器扩展了语义建模[22]。他们证明了嵌入在查询和文档中的层次语义结构可以通过深度学习来提取。报告了优于传统LSA的性能[22]。然而，他们使用的深度学习方法仍然采用无监督学习方法，其中模型参数针对文档重构而非针对给定查询区分相关文档和不相关文档进行优化。因此，深度学习模型没有显著优于基于关键词匹配的基线检索模型。此外，语义哈希模型还面临大规模矩阵乘法的可扩展性挑战。我们将在本文中展示，学习具有大规模词汇表的语义模型的能力对于在现实Web搜索任务中获得良好结果至关重要。

在本研究中，从上述两条研究路线扩展，我们提出了一系列用于Web搜索的深度结构化语义模型（DSSM）。更具体地说，我们最佳模型使用深度神经网络（DNN，Deep Neural Network）对给定查询的一组文档进行排序，如下所示：首先，执行非线性投影将查询和文档映射到共同的语义空间。然后，给定查询的每个文档的相关性计算为该语义空间中其向量之间的余弦相似度。神经网络模型使用点击数据进行判别式训练，使得给定查询下被点击文档的条件似然最大化。与以前以无监督方式学习的latent语义模型不同，我们的模型直接针对Web文档排序进行优化，因此具有优越的性能，我们将很快展示。此外，为了处理大规模词汇表，我们提出了所谓的词哈希方法，通过该方法，查询或文档的 高维术语向量 被投影到 低维的基于字母的n-gram向量，且**信息损失很小**。在我们的实验中，我们展示了通过在语义模型中添加这一额外的表示层，**词哈希使我们能够判别式地学习具有大规模词汇表的语义模型**，这对Web搜索至关重要。我们在Web文档排序任务上使用真实数据集评估了所提出的DSSMs。结果表明，我们最佳模型以2.5-4.3%的显著优势在NDCG@1上超越了所有竞争方法。

在本文的其余部分，第2节回顾相关工作。第3节描述我们的Web搜索DSSM。第4节介绍实验，第5节总结论文。



## 2. 相关工作

我们的工作基于latent语义模型在 IR（Information Retrieval）方面的两个最新扩展。第一个是利用点击数据以监督方式学习latent语义模型[10]。第二个是引入深度学习方法进行语义建模[22]。

### 2.1 Latent语义模型及点击数据的使用

使用latent语义模型进行查询-文档匹配是 IR（Information Retrieval）界长期研究的主题。流行的模型可分为两类：线性投影模型 和 生成主题模型，我们将依次回顾。

最著名的IR线性投影模型是LSA [6]。通过对 **文档-术语矩阵** 使用奇异值分解（SVD，Singular Value Decomposition），文档（或查询）可以被映射到低维概念向量 $\hat{y}$ ，其中 $W$ 是投影矩阵。在文档搜索中，分别由 术语向量 $x_q$ 和 $x_d$ 表示的 查询和文档之间的相关性 分数被假设与其相应概念向量 $\hat{y}_q = W^T x_q$ 和 $\hat{y}_d = W^T x_d$ 的余弦相似度分数成比例：

$$
\text{score}(q, d) = \cos(\hat{y}_q, \hat{y}_d) = \frac{\hat{y}_q^T \hat{y}_d}{\|\hat{y}_q\| \|\hat{y}_d\|} \qquad (1)
$$

除了latent语义模型，在被点击的查询-文档对上训练的翻译模型提供了语义匹配的另一种方法[9]。与latent语义模型不同，**基于翻译的方法直接学习文档中的术语和查询中的术语之间的翻译关系**。最近的研究表明，给定大量点击数据进行训练，这种方法可以非常有效[9,10]。我们还将在第4节中实验性地将我们的方法与翻译模型进行比较。

### 2.2 深度学习

最近，深度学习方法已成功应用于各种语言和信息检索应用[1,4,7,19,22,23,25]。通过利用深度架构，深度学习技术能够从训练数据中发现不同抽象层次上对任务有用的隐藏结构和特征。在[22]中，Salakhutdinov和Hinton通过使用深度网络（自编码器）扩展了LSA模型，以发现嵌入在查询和文档中的层次语义结构。他们提出了 **语义哈希**（SH，Semantic Hashing）方法，**使用从深度自编码器学习的瓶颈特征进行信息检索**。这些深度模型分两个阶段学习。首先，学习一堆生成模型（即受限Boltzmann机）以逐层将文档的术语向量表示映射到低维语义概念向量。然后，微调模型参数以**最小化文档原始术语向量和重构术语向量之间的交叉熵误差**。中间层激活用作文档排序的特征（即瓶颈）。他们的评估表明，SH方法达到了优于LSA的文档检索性能。然而，SH存在两个问题，不能超越标准的基于词汇匹配的检索模型（例如，**使用TF-IDF术语权重的余弦相似度**）。第一个问题是 **模型参数针对文档术语向量的重构** 而非 **针对给定查询区分相关文档和不相关文档进行优化**。其次，**为了使计算成本可控**，文档的术语向量仅包含**最频繁的2000个单词**。在下一节中，我们将展示我们对这两个问题的解决方案。



## 3. 用于Web搜索的深度结构化语义模型

### 3.1 用于计算语义特征的DNN

我们开发的用于将原始文本特征映射到语义空间中特征的典型DNN架构如图1所示。DNN的输入（原始文本特征）**是高维术语向量**，例如 **查询或文档中未归一化的原始术语计数**，DNN的输出是 低维语义特征空间 中的概念向量。该DNN模型用于Web文档排序，如下所示：1）将 **术语向量** 映射到其相应的 **语义概念向量**；2）计算文档和查询之间的相关性分数 为其 相应语义概念向量的余弦相似度；参见式（3）到（5）。

![image-20260729195110828](/Users/dazhang/PycharmProject/Papers/3-RecSys/.picture/image-20260729195110828.png)

>  **图1：DSSM示意图。它使用DNN将 高维稀疏文本特征 映射到 语义空间中的低维稠密特征。第一个隐藏层包含30k个单元，完成词哈希。词哈希特征然后通过多层非线性投影进行投影。该DNN中最后一层的神经活动构成语义空间中的特征。**

更正式地，如果我们用 $x$ 表示 **输入术语向量**， $y$ 表示 **输出向量**， $l_i$ （ $i = 1, \ldots, N-1$ ）表示中间隐藏层， $W_i$ 表示第 $i$ 个权重矩阵， $b_i$ 表示第 $i$ 个偏置项，我们有：
$$
l_1 = \text{rect}(W_1 x + b_1)
$$

$$
l_i = \text{rect}(W_i l_{i-1} + b_i), \quad i = 2, \ldots, N-1 \qquad (2)
$$

$$
y = \text{rect}(W_N l_{N-1} + b_N) \qquad (3)
$$

其中我们在输出层和隐藏层使用 $\text{rect}(\cdot)$ 作为激活函数：

$$
\text{rect}(x) = \begin{cases} x, & \text{if } x > 0 \\ 0, & \text{otherwise} \end{cases} \qquad (4)
$$

查询 $q$ 和文档 $d$ 之间的语义相关性分数然后被度量为：

$$
\text{sim}(q, d) = \cos(y_q, y_d) = \frac{y_q^T y_d}{\|y_q\| \|y_d\|} \qquad (5)
$$

其中 $y_q$ 和 $y_d$ 分别是查询和文档的概念向量。在Web搜索中，**给定查询，文档按其语义相关性分数排序**。

传统上，**术语向量的大小（可视为IR中的原始词袋特征）与用于索引Web文档集合的词汇表大小相同**。在现实Web搜索任务中，词汇表大小通常非常大。因此，当使用术语向量作为输入时，神经网络输入层的大小对于推理和模型训练来说将是不可管理的。为了解决这个问题，我们为DNN的第一层开发了一种称为"词哈希"的方法，如图1下部所示。该层仅包含线性隐藏单元，其中不学习非常大的权重矩阵。在下一节中，我们将详细描述词哈希方法。

### 3.2 词哈希

这里描述的词哈希方法**旨在降低词袋 术语向量 的维度**。它基于字母n-gram，是专门为我们的任务开发的新方法。给定一个单词（例如good），我们首先在单词中添加起始和结束标记（例如#good#）。然后，我们将单词分解为字母n-gram（例如字母三元组：#go, goo, ood, od#）。最后，**单词使用字母n-gram向量来表示**。

这种方法的一个问题是冲突，即两个不同的单词可能具有相同的字母n-gram向量表示。表1显示了在两个词汇表上词哈希的一些统计信息。与独热向量的原始大小相比，词哈希允许我们**使用维度低得多的向量来表示查询或文档**。以40K词汇表为例，每个单词可以使用字母三元组表示为10,306维向量，实现四倍的维度缩减且冲突很少。当该技术应用于更大的词汇表时，维度缩减更为显著。如表1所示，500K词汇表中的每个单词可以使用字母三元组表示为30,621维向量，维度缩减16倍，冲突率可忽略不计，为0.0044%（22/500,000）。

**虽然英语单词的数量可以是无限的，但英语（或其他类似语言）中字母n-gram的数量通常是有限的**。此外，**词哈希能够将同一单词的形态变体映射到字母n-gram空间中彼此接近的点**。更重要的是，虽然训练集中**未出现的单词**总是对 **基于单词的表示** 造成困难，但使用 **基于字母n-gram的表示** 则不然。唯一的风险是表1中量化的**小表示冲突**。因此，基于字母n-gram的词哈希 **对词汇表外问题具有鲁棒性**，使我们能够将DNN解决方案扩展到 **需要极大词汇表的Web搜索任务**。我们将在第4节中展示该技术的好处。

在我们的实现中，**基于字母n-gram的词哈希可视为固定的（即非自适应的）线性变换**，通过该变换，输入层中的术语向量被投影到 **更高层的字母n-gram向量**，如图1所示。由于字母n-gram向量的维度低得多，DNN学习可以有效地进行。

<img src="/Users/dazhang/PycharmProject/Papers/3-RecSys/.picture/image-20260729212857919.png" alt="image-20260729212857919" style="zoom:50%;" />

**表1：词哈希token大小和冲突数量随词汇表大小和字母n-gram类型的变化。**

| 词汇表大小 | Letter-Bigram Token大小 | Letter-Bigram冲突 | Letter-Trigram Token大小 | Letter-Trigram冲突 |
|:---:|:---:|:---:|:---:|:---:|
| 40k | 1,107 | 18 | 10,306 | 2 |
| 500k | 1,607 | 1,192 | 30,621 | 22 |

### 3.3 学习DSSM

**点击日志由 查询列表 及其 被点击的文档 组成**。我们假设查询 至少部分与 其被点击的文档 相关。受语音和语言处理中判别式训练方法的启发，我们因此提出了一种监督训练方法来学习我们的模型参数，即我们神经网络中的权重矩阵 $W$ 和偏置向量 $b$ 作为DSSM的核心部分，以最大化给定查询下被点击文档的条件似然。

首先，我们通过softmax函数从它们之间的语义相关性分数计算**给定查询下文档的后验概率**：

$$
P(d|q) = \frac{\exp(\gamma \cdot \text{sim}(q, d))}{\sum_{d^{\prime} \in D} \exp(\gamma \cdot \text{sim}(q, d^{\prime}))} \qquad (6)
$$

**其中 $\gamma$ 是softmax函数中的平滑因子**，在我们的实验中根据经验在保留数据集上设置。 $D$ 表示要排序的候选文档集合。**理想情况下， $D$ 应包含所有可能的文档**。实际上，对于每个（查询，被点击文档）对，记为 $(q, d^+)$ ，其中 $q$ 是查询， $d^+$ 是被点击的文档，我们通过包含 $d^+$ 和四个随机选择的未点击文档（记为 $\{d_1^-, d_2^-, d_3^-, d_4^-\}$ ）来**近似 $D$ ** 。在我们的初步研究中，我们**没有观察到使用不同采样策略选择未点击文档时有任何显著差异**。

> [!NOTE]
>
> 可以看到DSSM对后续在推荐系统中双塔模型的影响，基本双塔模型中的基本设置都是继承自DSSM

在训练中，模型参数被估计为最大化整个训练集中给定查询下被点击文档的似然。等价地，我们需要最小化以下损失函数：

$$
L(\Theta) = -\log \prod_{(q, d^+)} P(d^+|q) = -\sum_{(q, d^+)} \log P(d^+|q) \qquad (7)
$$

其中 $\Theta$ 表示神经网络 $\{W_i, b_i\}$ 的参数集。由于 $L$ 关于 $\Theta$ 是可微的，模型可以使用**基于梯度的数值优化算法**方便地训练。详细推导因篇幅限制而省略。

### 3.4 实现细节

为了确定训练参数并避免过拟合，我们将点击数据分为两个不重叠的部分，分别称为训练集和验证集。在我们的实验中，模型在训练集上训练，训练参数在验证集上优化。对于DNN实验，我们使用图1所示的三层隐藏层架构。第一个隐藏层是词哈希层，包含约30k个节点（例如，表1中所示的字母三元组大小）。接下来的两个隐藏层各有300个隐藏节点，输出层有128个节点。词哈希基于固定的投影矩阵。相似度度量基于维度为128的输出层。按照[20]，我们用**均匀分布初始化网络权重**，范围在 $[- \sqrt{6/(n_{in} + n_{out})}, \sqrt{6/(n_{in} + n_{out})}]$ 之间，其中 $n_{in}$ 和 $n_{out}$ 分别是输入和输出单元的数量。根据经验，我们没有观察到逐层预训练带来更好的性能。在训练阶段，我们使用基于小批量的随机梯度下降（SGD，Stochastic Gradient Descent）优化模型。每个小批量包含1024个训练样本。我们观察到DNN训练通常在20个epoch（遍历整个训练数据）内收敛。



## 4. 实验

我们在Web文档排序任务上使用真实数据集评估了第3节中提出的DSSM。在本节中，我们首先描述评估模型的数据集。然后，我们将我们最佳模型的性能与其他最先进的排序模型进行比较。我们还研究了第3节中提出技术的分解影响。

### 4.1 数据集和评估方法

我们在一个大规模真实数据集上评估了检索模型，以下称为评估数据集。评估数据集包含从商业搜索引擎一年查询日志文件中采样的16,510个英文查询。**平均每个查询与15个Web文档**（URL，Uniform Resource Locator）相关联。每个查询-标题对有一个相关性标签。**标签是人工生成的，采用5级相关性尺度，0到4**，其中级别4表示文档与查询 $q$ 最相关，0表示 $q$ 与 $d$ 不相关。所有查询和文档都经过预处理，文本被空格分词并小写化，数字被保留，不执行词干提取/词形变化。

本研究中使用的所有排序模型（即DSSM、主题模型和线性投影模型）包含许多必须根据经验估计的自由超参数。在所有实验中，我们使用了2折交叉验证：一半数据上的结果集使用在另一半上优化的参数设置获得，全局检索结果从两组组合而来。

我们评估的所有排序模型的性能都通过平均归一化折损累积增益（NDCG，Normalized Discounted Cumulative Gain）[17]来衡量，我们将在本节报告截断水平1、3和10的NDCG分数。我们还使用**配对t检验**进行了显著性检验。当p值小于0.05时，差异被认为具有统计显著性。

在我们的实验中，我们 **假设查询与为其点击的文档标题平行**。我们使用类似于[11]的程序从一年查询日志文件中提取大量 **查询-标题对** 用于模型训练。一些先前的研究，例如[11,24]，**表明查询点击字段（当有效时）是Web搜索最有效的信息，其次是标题字段**。然而，许多URL的点击信息不可用，特别是新URL和尾部URL，使其点击字段无效（即该字段因稀疏性而为空或不可靠）。在本研究中，我们假设评估数据集中的每个文档都是新URL或尾部URL，因此没有点击信息（即其点击字段无效）。我们的研究目标是研究如何从具有丰富点击信息的流行URL中学习latent语义模型，并将这些模型应用于改善这些尾部或新URL的检索。为此，在我们的实验中，**仅使用Web文档的标题字段进行排序**。为了训练latent语义模型，我们使用 **随机采样的约1亿对子集**，其中文档是流行的且具有丰富的点击信息。然后我们在评估数据集中测试训练好的模型，该数据集包含没有点击信息的文档。查询-标题对 以与 评估数据相同的方式进行预处理以确保一致性。

### 4.2 结果

我们实验的主要结果总结在表2中，其中我们将最佳版本的DSSM（第12行）与三组基线模型进行了比较。第一组基线包括几个广泛使用的词汇匹配方法，如 TF-IDF（Term Frequency-Inverse Document Frequency，第1行）和 BM25（**Best Match 25**，第2行）。第二组是词语翻译模型（WTM，Word Translation Model，第3行），旨在通过学习查询词和文档词之间的词汇映射直接解决查询-文档语言差异问题[9,10]。第三组包括一组最先进的latent语义模型，这些模型要么仅以无监督方式在文档上学习（LSA、PLSA、DAE，如第4至6行），要么**以监督方式在点击数据上学习**（BLTM-PR、DPM，如第7和8行）。为了使结果具有可比性，我们按照[10]中的描述重新实现了这些模型，例如，由于模型复杂度限制，LSA和DPM模型使用40k词汇表训练，其他模型使用500K词汇表训练。详情在以下段落中阐述。

**TF-IDF（Term Frequency-Inverse Document Frequency，第1行）是基线模型，其中文档和查询都表示为具有TF-IDF术语权重的术语向量。文档通过查询和文档向量之间的余弦相似度进行排序**。我们还使用BM25（Best Match 25，第2行）排序模型作为基线之一。TF-IDF和BM25都是基于术语匹配的最先进文档排序模型。它们已被广泛用作相关研究中的基线。

WTM（第3行）是我们对[9]中描述的词语翻译模型的实现，此处列出以供比较。我们看到WTM显著优于两个基线（TF-IDF和BM25），证实了[9]中得出的结论。LSA（第4行）是我们对潜在语义分析模型的实现。我们使用PCA（Principal Component Analysis，主成分分析）而非 SVD 来计算线性投影矩阵。查询和标题被视为单独的文档，该模型未使用点击数据中的配对信息。PLSA（第5行）是我们对[15]中提出的模型的实现，仅在文档上训练（即查询-标题对的标题端）。与[15]不同，我们的PLSA版本使用[10]中的MAP（Maximum A Posteriori，最大后验）估计学习。DAE（Deep Autoencoder，深度自编码器，第6行）是我们对Salakhutdinov和Hinton在[22]中提出的基于深度自编码器的语义哈希模型的实现。由于模型训练复杂度，输入术语向量基于40k词汇表。DAE架构包含四个隐藏层，每层300个节点，中间有一个128个节点的瓶颈层。模型仅以无监督方式在文档上训练。在微调阶段，我们使用交叉熵误差作为训练标准。中心层激活用作计算查询和文档之间余弦相似度的特征。我们的结果与[22]中报告的先前结果一致。基于DNN的latent语义模型优于线性投影模型（例如LSA）。然而，LSA和DAE都仅以无监督方式在文档集合上训练，因此不能超越最先进的基于词汇匹配的排序模型。

BLTM-PR（第7行）是[10]中描述的不同版本双语主题模型中的最佳性能者。具有后验正则化的BLTM（BLTM-PR）在查询-标题对上使用EM（Expectation-Maximization，期望最大化）算法训练，约束强制配对的查询和标题具有相同的分配给每个隐藏主题的术语比例。DPM（第8行）是[10]中提出的线性判别投影模型，其中投影矩阵使用S2Net算法[26]在查询和标题的相关和不相关对上判别式学习。类似于BLTM是PLSA的扩展，DPM也可以视为LSA的扩展，其中线性投影矩阵以监督方式使用点击数据学习，针对文档排序进行优化。我们看到使用点击数据进行模型训练带来了一些显著改进。BLTM-PR和DPM都优于基线模型（TF-IDF和BM25）。

**第9至12行展示了DSSM不同设置的结果**。DNN（第9行）是不使用词哈希的DSSM。它使用与DAE（第6行）相同的结构，但在点击数据上以监督方式训练。输入术语向量基于40k词汇表，与DAE相同。L-WH线性（第10行）是使用基于字母三元组的词哈希和监督训练构建的模型。它与L-WH非线性模型（第11行）的不同之处在于其输出层不应用任何非线性激活函数（如tanh）。L-WH DNN（第12行）是我们最佳的基于DNN的语义模型，它使用三个隐藏层，包括基于字母三元组的词哈希（L-WH）层和输出层，并在查询-标题对上进行判别式训练，如第3节所述。尽管基于字母n-gram的词哈希方法可以应用于任意大的词汇表，但为了与其他竞争方法进行公平比较，该模型使用500K词汇表。

表2的结果表明，深度结构化语义模型是最佳性能者，以统计显著的优势在NDCG上超越其他方法，证明了使用DNN进行语义匹配的经验有效性。

从表2的结果还可以清楚地看出，在点击数据上进行监督学习，结合**针对排序定制的 IR 中心优化标准**，对于获得优越的文档排序性能至关重要。例如，DNN和DAE（第9行和第6行）使用相同的40k词汇表并采用相同的深度架构。前者在NDCG@1上优于后者3.2个点。

**词哈希使我们能够使用非常大的词汇表进行建模。例如，第12行的模型使用500k词汇表（带词哈希），显著优于第9行使用40k词汇表的模型，尽管前者的自由参数略少于后者，因为词哈希层仅包含约30k个节点。**

> [!NOTE]
>
> 这段话很有意思，emb层是非常重要的。

我们还评估了在建模嵌入在查询和文档中的语义信息时使用深度架构与浅层架构的影响。表2中的结果表明，DAE（第3行）优于LSA（第2行），而两者都是无监督模型。**我们在比较监督模型中的浅层与深层架构时也观察到了类似的结果。比较第11行和第12行的模型，我们观察到将非线性层从一个增加到三个使NDCG分数提高了0.4-0.5个点，这些点具有统计显著性，而如果两者都是单层浅层模型（第10行对第11行），线性和非线性模型之间没有显著差异。**

<img src="/Users/dazhang/PycharmProject/Papers/3-RecSys/.picture/image-20260729212921343.png" alt="image-20260729212921343" style="zoom:33%;" />

**表2：与先前最先进方法和DSSM各种设置的比较结果。**



## 5. 结论

我们提出并评估了一系列新的latent语义模型，特别是那些具有我们称之为DSSM的深度架构的模型。主要贡献在于我们在三个关键方面显著扩展了先前的latent语义模型（例如LSA）。首先，我们利用点击数据通过直接针对文档排序目标来优化所有版本模型的参数。其次，受最**近在语音识别中取得巨大成功的深度学习框架的启发**[5,13,14,16,18]，我们使用多个隐藏表示层将线性语义模型扩展到其非线性对应物。采用的深度架构进一步增强了建模能力，以便可以捕获和表示查询和文档中更复杂的语义结构。第三，我们使用基于字母n-gram的词哈希技术，该技术在扩展深度模型训练方面被证明至关重要，使得在现实Web搜索中可以**使用非常大的词汇表**。在我们的实验中，我们展示了与上述三个方面相关的新技术各自在文档排序任务上带来了显著的性能改进。三组新技术的结合导致了一种新的最先进语义模型，该模型以显著优势超越了所有先前开发的竞争模型。



## 参考文献

[1] Bengio, Y., 2009. "Learning deep architectures for AI." Foundumental Trends Machine Learning, vol. 2.

[2] Blei, D. M., Ng, A. Y., and Jordan, M. J. 2003. "Latent Dirichlet allocation." In JMLR, vol. 3.

[3] Burges, C., Shaked, T., Renshaw, E., Lazier, A., Deeds, M., Hamilton, and Hullender, G. 2005. "Learning to rank using gradient descent." In ICML.

[4] Collobert, R., Weston, J., Bottou, L., Karlen, M., Kavukcuoglu, K., and Kuksa, P., 2011. "Natural language processing (almost) from scratch." in JMLR, vol. 12.

[5] Dahl, G., Yu, D., Deng, L., and Acero, A., 2012. "Context-dependent pre-trained deep neural networks for large vocabulary speech recognition." in IEEE Transactions on Audio, Speech, and Language Processing.

[6] Deerwester, S., Dumais, S. T., Furnas, G. W., Landauer, T., and Harshman, R. 1990. "Indexing by latent semantic analysis." J. American Society for Information Science, 41(6): 391-407

[7] Deng, L., He, X., and Gao, J., 2013. "Deep stacking networks for information retrieval." In ICASSP

[8] Dumais, S. T., Letsche, T. A., Littman, M. L., and Landauer, T. K. 1997. "Automatic cross-linguistic information retrieval using latent semantic indexing." In AAAI-97 Spring Symposium Series: Cross-Language Text and Speech Retrieval.

[9] Gao, J., He, X., and Nie, J-Y. 2010. "Clickthrough-based translation models for web search: from word models to phrase models." In CIKM.

[10] Gao, J., Toutanova, K., Yih., W-T. 2011. "Clickthrough-based latent semantic models for web search." In SIGIR.

[11] Gao, J., Yuan, W., Li, X., Deng, K., and Nie, J-Y. 2009. "Smoothing clickthrough data for web search ranking." In SIGIR.

[12] He, X., Deng, L., and Chou, W., 2008. "Discriminative learning in sequential pattern recognition," Sept. IEEE Sig. Proc. Mag.

[13] Heck, L., Konig, Y., Sonmez, M. K., and Weintraub, M. 2000. "Robustness to telephone handset distortion in speaker recognition by discriminative feature design." In Speech Communication.

[14] Hinton, G., Deng, L., Yu, D., Dahl, G., Mohamed, A., Jaitly, N., Senior, A., Vanhoucke, V., Nguyen, P., Sainath, T., and Kingsbury, B., 2012. "Deep neural networks for acoustic modeling in speech recognition," IEEE Sig. Proc. Mag.

[15] Hofmann, T. 1999. "Probabilistic latent semantic indexing." In SIGIR.

[16] Hutchinson, B., Deng, L., and Yu, D., 2013. "Tensor deep stacking networks." In IEEE T-PAMI, vol. 35.

[17] Jarvelin, K. and Kekalainen, J. 2000. "IR evaluation methods for retrieving highly relevant documents." In SIGIR.

[18] Konig, Y., Heck, L., Weintraub, M., and Sonmez, M. K. 1998. "Nonlinear discriminant feature extraction for robust text-independent speaker recognition." in RLA2C.

[19] Mesnil, G., He, X., Deng, L., and Bengio, Y., 2013. "Investigation of recurrent-neural-network architectures and learning methods for spoken language understanding." In Interspeech.

[20] Montavon, G., Orr, G., Müller, K., 2012. Neural Networks: Tricks of the Trade (Second edition). Springer.

[21] Platt, J., Toutanova, K., and Yih, W. 2010. "Translingual document representations from discriminative projections." In EMNLP.

[22] Salakhutdinov R., and Hinton, G., 2007 "Semantic hashing." in Proc. SIGIR Workshop Information Retrieval and Applications of Graphical Models.

[23] Socher, R., Huval, B., Manning, C., Ng, A., 2012. "Semantic compositionality through recursive matrix-vector spaces." In EMNLP.

[24] Svore, K., and Burges, C. 2009. "A machine learning approach for improved BM25 retrieval." In CIKM.

[25] Tur, G., Deng, L., Hakkani-Tur, D., and He, X., 2012. "Towards deeper understanding deep convex networks for semantic utterance classification." In ICASSP.

[26] Yih, W., Toutanova, K., Platt, J., and Meek, C. 2011. "Learning discriminative projections for text similarity measures." In CoNLL.



## 附录

### I. 梯度计算和梯度下降

由于 $L$ 关于 $\Theta$ 是可微的，模型可以使用基于梯度的数值优化算法方便地训练。更新规则为：

$$
\Theta^{(t)} = \Theta^{(t-1)} - \eta^{(t)} \nabla L(\Theta^{(t-1)}) \qquad (8)
$$

其中 $\eta^{(t)}$ 是第 $t$ 次迭代的学习率， $\Theta^{(t-1)}$ 和 $\Theta^{(t)}$ 分别是第 $t-1$ 次和第 $t$ 次迭代的模型。

接下来，我们推导损失函数关于神经网络参数的梯度。假设有总共 $R$ 个（查询，被点击文档）对，我们将第 $r$ 个（查询，被点击文档）对记为 $(q_r, d_r^+)$ 。然后，如果我们记：

$$
J_r(\Theta) = -\log P(d_r^+|q_r) \qquad (9)
$$

我们有：

$$
L(\Theta) = \sum_{r=1}^{R} J_r(\Theta) \qquad (10)
$$

接下来，我们将展示 $\frac{\partial J_r}{\partial W_N}$ 的推导。

对于查询 $q$ 和文档 $d$ ，我们记 $l_i^q$ 和 $l_i^d$ 为隐藏层 $i$ 中的激活， $y_q$ 和 $y_d$ 分别为 $q$ 和 $d$ 的输出激活。它们根据式（3）计算。

我们然后如下推导 $\frac{\partial J_r}{\partial W_N}$ 。为简化起见，此后将省略下标 $r$ 。

首先，式（9）中的损失函数可以写为：

$$
J = -\log \frac{\exp(\gamma \cdot \text{sim}(y_q, y_{d^+}))}{\exp(\gamma \cdot \text{sim}(y_q, y_{d^+})) + \sum_{i=1}^{4} \exp(\gamma \cdot \text{sim}(y_q, y_{d_i^-}))} \qquad (11)
$$

其中 $\text{sim}(y_q, y_d) = \frac{y_q^T y_d}{\|y_q\| \|y_d\|}$ 。损失函数关于第 $N$ 个权重矩阵 $W_N$ 的梯度为：

$$
\frac{\partial J}{\partial W_N} = \gamma \left( \frac{\partial \text{sim}(y_q, y_{d^+})}{\partial W_N} - \sum_{k=1}^{K} P(d_k|q) \frac{\partial \text{sim}(y_q, y_d^k)}{\partial W_N} \right) \qquad (12)
$$

其中：

$$
\frac{\partial \text{sim}(y_q, y_d)}{\partial W_N} = \frac{1}{\|y_q\| \|y_d\|} \left( y_d \frac{\partial y_q^T}{\partial W_N} + y_q \frac{\partial y_d^T}{\partial W_N} \right) - \frac{\text{sim}(y_q, y_d)}{\|y_q\|^2 \|y_d\|^2} \left( y_q \|y_d\| \frac{\partial \|y_q\|^2}{\partial W_N} + y_d \|y_q\| \frac{\partial \|y_d\|^2}{\partial W_N} \right) \qquad (13)
$$

和：

$$
\frac{\partial \|y_q\|^2}{\partial W_N} = 2 l_{N-1}^q y_q^T, \quad \frac{\partial \|y_d\|^2}{\partial W_N} = 2 l_{N-1}^d y_d^T \qquad (14)
$$

为简化符号，令 $\delta_q = y_q / \|y_q\|$ ， $\delta_d = y_d / \|y_d\|$ ， $\alpha = y_q^T y_d / (\|y_q\| \|y_d\|)$ ， $\beta = \text{sim}(y_q, y_d)$ 。使用 $\text{rect}(\cdot)$ 作为我们模型中的激活函数，式（13）右边的每一项可以使用以下公式计算：

$$
\frac{\partial \text{sim}(y_q, y_d)}{\partial W_N} = \frac{1}{\|y_q\| \|y_d\|} \left( y_d (l_{N-1}^q)^T + y_q (l_{N-1}^d)^T \right) - \frac{\beta}{\|y_q\|^2 \|y_d\|^2} \left( y_q \|y_d\|^2 (l_{N-1}^q)^T + y_d \|y_q\|^2 (l_{N-1}^d)^T \right) \qquad (15)
$$

其中 $l_{N-1}^q$ 和 $l_{N-1}^d$ 对于一对 $(q, d)$ 计算为：

$$
l_{N-1}^q = \text{diag}(\text{rect}'(W_{N-1} l_{N-2}^q + b_{N-1})) \cdot W_{N-1} l_{N-2}^q
$$

$$
l_{N-1}^d = \text{diag}(\text{rect}'(W_{N-1} l_{N-2}^d + b_{N-1})) \cdot W_{N-1} l_{N-2}^d \qquad (16)
$$

其中算子 $\cdot$ 是逐元素乘法（Hadamard积）。

对于隐藏层，我们还需要为每个 $i$ 计算 $l_i^q$ 。例如，隐藏层 $l$ 中的每个 $l_i^q$ 可以通过反向传播计算为：

$$
\delta_{l}^q = \text{diag}(\text{rect}'(W_l l_{l-1}^q + b_l)) \cdot W_{l+1}^T \delta_{l+1}^q
$$

$$
\delta_{l}^d = \text{diag}(\text{rect}'(W_l l_{l-1}^d + b_l)) \cdot W_{l+1}^T \delta_{l+1}^d \qquad (17)
$$

最终我们有 $\delta_1^q = \text{diag}(\text{rect}'(W_1 x_q + b_1)) \cdot W_2^T \delta_2^q$ 和 $\delta_1^d = \text{diag}(\text{rect}'(W_1 x_d + b_1)) \cdot W_2^T \delta_2^d$ 。

相应地，损失函数关于中间权重矩阵 $W_l$ （ $l = 1, \ldots, N-1$ ）的梯度可以计算为：

$$
\frac{\partial J}{\partial W_l} = \gamma \left( \frac{\partial \text{sim}(y_q, y_{d^+})}{\partial W_l} - \sum_{k=1}^{K} P(d_k|q) \frac{\partial \text{sim}(y_q, y_d^k)}{\partial W_l} \right) \qquad (18)
$$

其中：

$$
\frac{\partial \text{sim}(y_q, y_d)}{\partial W_l} = \frac{1}{\|y_q\| \|y_d\|} \left( \delta_d^q (l_{l-1}^q)^T + \delta_d^d (l_{l-1}^d)^T \right) - \frac{\beta}{\|y_q\|^2 \|y_d\|^2} \left( \delta_q \|y_d\|^2 (l_{l-1}^q)^T + \delta_d \|y_q\|^2 (l_{l-1}^d)^T \right) \qquad (19)
$$

> [!NOTE]
>
> 这个梯度推导，厉害了！

### II. 文档排序错误分析

在测试数据中，在16,412个唯一查询中，我们使用TF-IDF和我们最佳模型（基于字母三元组的词哈希与监督DNN（L-WH DNN））比较每个查询的NDCG@1值。总共有1,985个查询L-WH DNN表现优于TF-IDF（NDCG@1差异总和为1332.3）。另一方面，TF-IDF在1,077个查询上优于L-WH DNN（NDCG@1差异总和为630.61）。对于这两种情况，我们抽样了几个具体示例。它们分别如表5和表6所示。我们在表5中观察到，**NDCG的改进主要归功于 语义层面 比 词汇层面 更好的查询-标题匹配**。

**表5：我们的深度语义模型优于TF-IDF的示例。**

<img src="/Users/dazhang/PycharmProject/Papers/3-RecSys/.picture/image-20260730202141368.png" alt="image-20260730202141368" style="zoom:50%;" />

**表6：我们的深度语义模型劣于TF-IDF的示例。**

<img src="/Users/dazhang/PycharmProject/Papers/3-RecSys/.picture/image-20260730202208879.png" alt="image-20260730202208879" style="zoom:50%;" />

为了使我们的方法更直观，我们还可视化了 查询和文档 中学习到的**隐藏词表示**。我们通过**将每个单词视为唯一文档并将其作为输入传递给训练好的DNN来实现这一点**。在每个输出节点，我们将所有**具有高激活水平的单词分组**并相应地聚类它们。表7显示了一些示例聚类，**每个聚类对应于DNN模型的一个输出节点**。有趣的是，具有相同或相关语义含义的单词确实留在同一聚类中。

**表7：训练DNN 五个不同输出节点 上的聚类词示例。聚类标准是DNN输出节点上的高激活水平。**

<img src="/Users/dazhang/PycharmProject/Papers/3-RecSys/.picture/image-20260730202221267.png" alt="image-20260730202221267" style="zoom:50%;" />

> [!NOTE]
>
> 结果很神奇。但不是很理解具体怎么做的

