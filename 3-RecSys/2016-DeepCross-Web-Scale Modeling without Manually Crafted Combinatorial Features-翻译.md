# Deep Crossing：无需手工构建组合特征的Web-Scale建模

> Ying Shan, T. Ryan Hoens, Jian Jiao, Haijing Wang, Dong Yu*, JC Mao | Bing Ads, Microsoft Research*; Microsoft Corporation

本文提出 Deep Crossing 模型，一种能自动组合特征的深度神经网络，**在Web-Scale应用中无需手工构建组合特征即可达到生产级效果**。核心发现是——**仅使用生产模型特征子集，Deep Crossing 在点击预测任务上超越了经过多年优化的生产模型**。

核心内容：

- **问题/痛点**：手工构建组合特征（Combinatorial Features）是许多成功模型的"秘方"，但在Web-Scale应用中，特征的种类和数量使得手工构建、维护和部署这些特征代价高昂
- **方案/创新点**：提出 Deep Crossing 模型，由嵌入层（Embedding）、堆叠层（Stacking）和残差单元（Residual Unit）级联组成，自动发现重要的交叉特征
- **技术细节**：输入为一组单独特征（Individual Features），可以是稠密或稀疏的；使用修改版的残差单元（不含卷积核），首次将残差网络应用于图像识别之外的领域
- **实验验证**：在主要搜索引擎的22亿样本上训练，仅使用生产模型特征子集，在CP1任务上超越生产模型；32 GPU跨8台机器将训练时间从24天降至20小时

关键发现：

- **Deep Crossing 在纯文本输入任务上持续优于 DSSM**，在CP1任务上相对AUC提升0.46%-1.02%
- **仅使用Query和Keyword文本特征时，与全特征模型的相对log loss差距约0.12**，对应AUC提升约7-8%
- **添加计数特征（Counting Features）可将相对log loss降低0.02**
- **使用22亿样本训练的Deep Crossing模型，在CP1任务上超越生产模型1.02个百分点**（相对AUC）

---

## 摘要

手工构建的组合特征一直是许多成功模型背后的"秘方"。然而，对于Web-Scale应用，特征的种类和数量使得这些手工构建的特征在创建、维护和部署方面代价高昂。本文提出了 Deep Crossing 模型，这是一种深度神经网络，能够自动组合特征以产生更优的模型。Deep Crossing 的输入是一组单独特征，可以是稠密的或稀疏的。重要的交叉特征由网络隐式发现，网络由嵌入和堆叠层以及级联的残差单元组成。

Deep Crossing 使用名为 Computational Network Tool Kit (CNTK) 的建模工具实现，由多GPU平台驱动。它能够从零开始为一个主要的付费搜索引擎构建两个Web-Scale模型，并且仅使用生产模型所用特征的一个子集就取得了优异的结果。这展示了使用 Deep Crossing 作为通用建模范式来改进现有产品的潜力，以及以特征工程和深度领域知识获取的一小部分投资来加速新模型开发的潜力。

**Keywords**：Neural Networks, Deep Learning, Convolutional Neural Network (CNN), Deep Crossing, Combinatorial Features, DSSM, Residual Net, CNTK, GPU, Multi-GPU Platform

## 1. 引言

传统的机器学习算法应该充分利用所有输入特征来预测和分类新的实例。然而，简单地使用原始特征很少能提供最优结果。因此，无论在工业界还是学术界，都存在大量关于原始特征转换工程的工作。一种主要的转换类型是基于多个特征的组合来构建函数，并将其输出用作学习器（Learner）的输入。这些组合特征有时被称为交叉特征或多路特征。

组合特征是强大的工具，特别是在领域专家手中。在我们自己与一个主要赞助搜索引擎的经验中，它们是许多模型中最强的特征。在 Kaggle 社区中，顶级数据科学家是构建此类特征的大师，甚至可以交叉三到五个维度。创建有效组合特征的直觉和能力是他们获胜公式中的重要成分。在计算机视觉社区中，类似 SIFT 的特征是当时 ImageNet 竞赛最先进性能背后的关键驱动力。SIFT 特征在图像块上提取，是组合特征的特殊形式。

组合特征的力量伴随着高昂的成本。在个人能够开始创建有意义的特征之前，需要攀登陡峭的学习曲线。随着特征数量的增长，管理、维护和部署它们变得具有挑战性。在Web-Scale应用中，由于巨大的搜索空间以及数十亿样本下训练和评估周期的缓慢周转，寻找额外的组合特征来改进现有模型是一项艰巨的任务。

深度学习承诺从单独特征中学习，无需手动干预。语音和图像识别是最早展示这一潜力的领域。通过深度卷积神经网络（CNN）从特定任务学习的卷积核已经取代了手工构建的类似 SIFT 的特征，成为图像识别的最先进方法。类似的模型已被应用于自然语言处理（NLP）应用，从零开始构建语言处理模型，无需大量特征工程。

Deep Crossing 将深度学习的成功扩展到更一般的设置，其中单独特征具有不同的性质。更具体地说，它接受文本、分类、ID 和数值等单独特征，并根据特定任务自动搜索最优组合。此外，Deep Crossing 被设计用于处理Web-Scale应用和数据大小。这不仅因为作者主要对此类应用感兴趣，而且因为在此规模上运行的通用模型没有太多选择。值得注意的是，Deep Crossing 在学习过程中确实以某种方式生成组合特征，尽管 Deep Crossing 的输出是一个没有这些特征显式表示的模型。

## 2. 相关工作

学习深度神经网络而无需手工构建特征的想法并不新鲜。在80年代早期，Fukushima 报告了一个七层的新认知机（Neocognitron）网络，该网络从图像的原始像素识别数字。通过利用部分连接结构，新认知机实现了平移不变性，这是视觉识别任务的重要属性。CNN 由 LeCun 等人在90年代末发明，具有类似的架构，特别是部分连接的卷积核。尽管 CNN 作为识别引擎具有坚实的基础，但基于类似 SIFT 特征的分类器在近十年中主导了图像识别。2012年，Krizhevsky 等人提出了 AlexNet，将基于 SIFT 的基线错误率降低了近11个百分点。最近，一个152层的残差网络在2015年赢得了 ImageNet 和 MS COCO 竞赛。

深度 CNN 的演变既鼓舞人心又令人振奋。它表明深度学习即使在最好的手工特征已经过十多年精细调优的系统中也能有所改进。换句话说，即使是最有经验的领域专家也可能错过深度 CNN 使用特定任务滤波器捕获的特征之间的深层交互。认识到这一点对我们关于 Deep Crossing 的工作具有深远影响。

深度语义相似度模型（DSSM）学习一对文本字符串之间的语义相似度，每个字符串由称为三字母克（tri-letter grams）的稀疏表示表示。学习算法通过将三字母克嵌入到两个向量中来优化基于余弦距离的目标函数。学习到的嵌入捕获了单词和句子的语义含义，并已被应用于赞助搜索、问答和机器翻译，取得了强劲的结果。

分解机（FM）以其一般形式对单独特征之间的 $d$ 路交互进行建模。在存在非常稀疏输入的情况下，FM 显示出比 SVM 更好的结果，但尚不清楚它在稠密特征上的表现如何。

对于 NLP 任务，Collobert 等人构建了一个统一的神经网络架构，避免了特定任务的特征工程。Deep Crossing 旨在覆盖更广泛的输入特征。

## 3. 赞助搜索

Deep Crossing 在一个主要搜索引擎的赞助搜索背景下进行讨论。读者可以参考 Edelman 等人关于此主题的概述。简而言之，赞助搜索负责在有机搜索结果旁边展示广告。生态系统中有三个主要参与者：用户、广告商和搜索平台。平台的目标是向用户展示最能匹配用户意图的广告，这主要通过特定查询来表达。以下是后续讨论的关键概念：

- **Query（查询）**：用户输入搜索框的文本字符串
- **Keyword（关键词）**：与产品相关的文本字符串，由广告商指定以匹配用户查询
- **Title（标题）**：赞助广告的标题，由广告商指定以吸引用户注意力
- **Landing page（落地页）**：用户点击相应广告时到达的产品网站
- **Match type（匹配类型）**：给广告商的选项，用于指定关键词应与用户查询匹配的紧密程度，通常为四种之一：精确匹配、短语匹配、广泛匹配和上下文匹配
- **Campaign（广告系列）**：共享相同设置（如预算和位置定位）的一组广告，通常用于将产品组织成类别
- **Impression（展示）**：广告展示给用户的一个实例。展示通常与其他在运行时可用的信息一起记录
- **Click（点击）**：表示展示是否被用户点击。点击通常与其他在运行时可用的信息一起记录
- **Click through rate（点击率）**：总点击次数除以总展示次数
- **Click Prediction（点击预测）**：平台的一个关键模型，预测用户在给定查询下点击给定广告的可能性

赞助搜索只是Web-Scale应用的一种。然而，鉴于问题空间的丰富性、各种类型的特征以及海量数据，我们认为我们的结果可以推广到具有类似规模的其他应用。

## 4. 特征表示

本节使用表1中列出的示例定义和比较组合特征与单独特征。表中的特征在广告展示（展示）时的运行时可用。它们也可在离线日志中用于模型训练等。

### 4.1 单独特征

每个单独特征 $X_i$ 表示为一个向量。对于文本特征（如查询），一种选择是将字符串转换为49,292维的三字母克，如 Huang 等人所述。分类输入（如 MatchType）由独热向量表示，其中精确匹配为 $[1, 0, 0, 0]$，短语匹配为 $[0, 1, 0, 0]$，依此类推。

赞助搜索系统中通常有数百万个广告系列。简单地将广告系列ID转换为独热向量将显著增加模型大小。一种解决方案是使用一对伴随特征，如表中所示，其中 CampaignID 是仅包含点击次数最多的前10,000个广告系列的独热表示。第10,000个槽位（索引从0开始）保留给所有剩余的广告系列。其他广告系列由 CampaignIDCount 覆盖，这是一个数值特征，存储每个广告系列的统计信息（如点击率）。此类特征在以下讨论中称为计数特征。除计数特征外，到目前为止引入的所有特征都是稀疏特征。

| 特征名称 | 类型 | 维度 |
|----------|------|------|
| Query | Text | 49,292 |
| Keyword | Text | 49,292 |
| Title | Text | 49,292 |
| MatchType | Category | 4 |
| CampaignID | ID | 10,001 |
| CampaignIDCount | Numerical | 5 |

**表1：单独特征示例**

### 4.2 组合特征

给定单独特征 $X_i \in \mathbb{R}^{n_i}$ 和 $X_j \in \mathbb{R}^{n_j}$，组合特征 $X_{i,j}$ 定义在 $\mathbb{R}^{n_i \times n_j}$ 中。组合特征也有稀疏和稠密表示。稀疏表示的一个例子是 CampaignId×MatchType 特征，这是一个 $10,001 \times 4 = 40,004$ 维的独热向量。稠密表示的一个例子是计算特定 CampaignId 和 MatchType 组合的广告点击次数。稠密表示的维度与其稀疏对应物相同。

Deep Crossing 避免使用组合特征。它同时处理稀疏和稠密的单独特征，并支持上述广泛范围的特征类型。这使用户可以自由选择其特定应用中的特征。虽然收集特征并将其转换为正确的表示仍然需要大量工作，但工作止于单独特征的级别。其余部分由模型处理。

## 5. 模型架构

图1是 Deep Crossing 的模型架构，其中输入是一组单独特征。该模型有四种类型的层，包括嵌入层（Embedding）、堆叠层（Stacking）、残差单元（Residual Unit）和评分层（Scoring Layer）。目标函数在我们的应用中是对数损失（log loss），但可以轻松定制为 softmax 或其他函数。对数损失定义为：

$$
\text{logloss} = -\frac{1}{N} \sum_{i=1}^{N} (y_i \log(p_i) + (1 - y_i) \log(1 - p_i))
$$

其中 $i$ 索引训练样本，$N$ 是样本数量，$y_i$ 是每个样本的标签，$p_i$ 是图1中单节点评分层的输出，在这种情况下是 Sigmoid 函数。点击预测问题中的标签 $y_i$ 是用户点击。

### 5.1 嵌入和堆叠层

嵌入应用于每个单独特征以转换输入特征。嵌入层由单层神经网络组成，一般形式为：

$$
X_j^O = \max(0, W_j X_j^I + b_j)
$$

其中 $j$ 索引单独特征，$X_j^I \in \mathbb{R}^{n_j}$ 是输入特征，$W_j$ 是 $m_j \times n_j$ 矩阵，$b \in \mathbb{R}^{n_j}$，$X_j^O$ 是嵌入特征。当 $m_j < n_j$ 时，嵌入用于降低输入特征的维度。逐元素 max 运算符在神经网络上下文中通常称为修正线性单元（ReLU）。然后输出特征被堆叠（连接）为一个向量作为下一层的输入：

$$
X^O = [X_0^O, X_1^O, \cdots, X_K^O]
$$

其中 $K$ 是输入特征的数量。注意 $\{W_j\}$ 和 $\{b_j\}$ 都是网络的参数，将与网络中的其他参数一起优化。这是 Deep Crossing 中嵌入的一个重要属性。与仅嵌入方法（如 word2vec）不同，它是整体优化过程的组成部分。

应该指出的是，嵌入层的大小对模型的整体大小有显著影响。即使对于稀疏特征，$m_j \times n_j$ 权重矩阵本质上也是稠密的。这就是为什么 Deep Crossing 使用第4节中的伴随特征来约束高基数特征的维度。对于第7节中的点击预测实验，输入特征的总大小约为200,000，查询和关键词等高维特征的 $m_j$ 统一设置为256。维度低于256的特征无需嵌入即可堆叠。图1中的特征#2就是一个例子。

### 5.2 残差层

残差层由图2中的残差单元构建。残差单元是残差网络（Residual Net）的基本构建块，该网络在 ImageNet 竞赛中创造了世界纪录。Deep Crossing 使用了一个略有修改的残差单元，不使用卷积核。据我们所知，这是残差单元首次被用于解决图像识别之外的问题。

残差单元的独特属性是在通过两层 ReLU 变换后将原始输入特征加回来。具体来说：

$$
X^O = F(X^I, \{W_0, W_1\}, \{b_0, b_1\}) + X^I
$$

其中 $W_{\{0,1\}}$ 和 $b_{\{0,1\}}$ 是两层的参数，$F$ 表示将残差单元的输入 $X^I$ 映射到输出 $X^O$ 的函数。将 $X^I$ 移到等式的左边，$F(\cdot)$ 本质上是在拟合 $X^O - X^I$ 的残差。在 He 等人的论文中，作者认为拟合残差具有数值优势。虽然残差网络能够深达152层并保持高性能的实际原因有待进一步研究，但 Deep Crossing 确实展示了一些可能受益于残差单元的属性。

在 Deep Crossing 之前，我们尝试了许多具有深层的模型架构，但没有一个能提供显著的收益来证明增加的复杂性是合理的。Deep Crossing 是我们最强的模型，轻松击败了其较浅对应物的性能。

Deep Crossing 被应用于各种任务。它也被应用于样本大小差异很大的训练数据。在所有情况下，都使用相同的模型，无需对层、节点和节点类型进行任何调整。残差单元可能隐式地执行某种正则化，从而导致这种稳定性。

### 5.3 早期交叉与晚期交叉

值得将 Deep Crossing 与 DSSM 进行比较。图3是使用对数损失作为目标函数的修改版 DSSM 的架构。修改后的 DSSM 与点击预测的应用更密切相关。它保持了绿色虚线左侧的 DSSM 基本结构，但使用对数损失将预测与真实标签进行比较。DSSM 允许两个文本输入，每个输入由其三字母克向量表示。DSSM 具有将特征交互（或交叉）延迟到前向计算后期阶段的独特属性。在到达余弦距离节点之前，输入特征通过两条独立路线上的多层变换被完全嵌入。相比之下，Deep Crossing 最多采用一层单特征嵌入，并在前向计算的更早阶段开始特征交互。

DSSM 及其卷积变体（CDSSM）已被许多现实世界应用证明是特别针对一对文本输入调优的强大学习机器。尽管如此，在第7.1节中比较 Deep Crossing 和 DSSM 在两个纯文本任务上的表现时，前者始终优于后者。除了残差单元的卓越优化能力外，在前向计算早期引入特征交互似乎发挥了重要作用。

## 6. 实现

### 6.1 使用 CNTK 建模

使用 CNTK 构建 Deep Crossing 模型非常简单。下面的脚本定义了一些稍后将使用的宏。

```
ONELAYER(Dim_XO, Dim_XI, XI){
  W = Parameter(Dim_XO, Dim_XI)
  b = Parameter(Dim_XO)
  XO = Plus(Times(W, XI), b)
}
ONELAYERSIG(Dim_XO, Dim_XI, XI){
  t = ONELAYER(Dim_XO, Dim_XI, XI)
  XO = Sigmoid(t)
}
ONELAYERRELU(Dim_XO, Dim_XI, XI){
  t = ONELAYER(Dim_XO, Dim_XI, XI)
  XO = ReLU(t)
}
RESIDUALUNIT(Dim_H, Dim_XI, XI){
  l1 = ONELAYERRELU(Dim_H, Dim_XI, XI)
  l2 = ONELAYER(Dim_XI, Dim_H, l1)
  XO = ReLU(Plus(XI, l2))
}
```

宏 ONELAYER 为网络的一层创建权重矩阵 $W$ 和偏置向量 $b$。宏 ONELAYERSIG 和 ONELAYERRELU 对 ONELAYER 的输出应用逐元素 Sigmoid 和 ReLU 函数，分别构建完整的 Sigmoid 层和 ReLU 层。并排比较 ONELAYERRELU 和等式2，$X_I$、$X_O$、$\text{Dim\_XI}$ 和 $\text{Dim\_XO}$ 分别是 $X_j^I$、$X_j^O$、$n_j$ 和 $m_j$。RESIDUALUNIT 使用大小为 Dim_H 的隐藏层，但输出与 Dim_XI 具有相同的维度。图4中的脚本详细描述了应用于表1中特征的实际 Deep Crossing 模型。从脚本中可以看出，完整模型有五个步骤。每个步骤在第5节中已描述。由于 MatchType 特征 M 只有4维，它直接进入堆叠阶段，无需嵌入。这就是为什么嵌入步骤和堆叠步骤被视为一个组合步骤。Dim_E 和 Dim_CROSS* 分别是嵌入层和残差单元内隐藏层的节点数。Dim_L=1 用于对数损失。为简单起见，此处未包含特征I/O和归一化等细节。

```
## Deep Crossing 模型
## 步骤1：读取特征，已省略
## 步骤2A：嵌入
Q = ONELAYERRELU(Dim_E, Dim_Query, Query)
K = ONELAYERRELU(Dim_E, Dim_Keyword, Keyword)
T = ONELAYERRELU(Dim_E, Dim_Title, Title)
C = ONELAYERRELU(Dim_E, Dim_CampaignID, CampaignID)
## 步骤2B：堆叠
# M = MatchType, CC = CampaignIDCount
Stack = RowStack(Q, K, T, C, M, CC)
## 步骤3：深度残差层
r1 = RESIDUALUNIT(Dim_CROSS1, Dim_Stack, Stack)
r2 = RESIDUALUNIT(Dim_CROSS2, Dim_Stack, r1)
r3 = RESIDUALUNIT(Dim_CROSS3, Dim_Stack, r2)
r4 = RESIDUALUNIT(Dim_CROSS4, Dim_Stack, r3)
r5 = RESIDUALUNIT(Dim_CROSS5, Dim_Stack, r4)
## 步骤4：最终 Sigmoid 层
Predict = ONELAYERSIG(Dim_L, Dim_Stack, r5)
## 步骤5：对数损失目标
CE = LogLoss(Label, Predict)
CriteriaNodes = (CE)
```

**图4：使用 CNTK 建模语言描述的具有五层残差单元的 Deep Crossing 模型**

### 6.2 多GPU平台

实验在为 CNTK 优化的 GPU 集群上进行，用于快速、无麻烦的深度学习模型训练和评估。该集群具有高吞吐量分布式存储、虚拟文件系统和容错能力，由专门设计的自动化集群管理和作业/容器调度软件管理。在实验期间，集群中的每台 GPU 机器包含四个 K40 GPU 卡。Infiniband 用于连接附近的 GPU 机器，以实现跨机器 GPU 之间的高速数据传输。

为了加速实验，我们利用了 CNTK 中实现的块级模型更新过滤（BMUF）分布式训练算法，该算法最初由 Chen 和 Huo 提出。BMUF 算法改进了传统的模型平均（MA）和交替方向乘子法（ADMM），同时保持了这些方法的低通信成本优势。虽然 MA 和 ADMM 可以轻松扩展，但它们在训练模型可达到的准确性方面通常不如单GPU SGD，并且需要与单GPU SGD 不同的学习率计划。然而，BMUF 算法没有这些缺点。Chen 和 Huo 报告了使用32个GPU实现28倍加速，同时在大规模语音识别实验中比单GPU SGD 实现了更好的准确性。在本文报告的实验中，我们将训练时间从单GPU的24天减少到使用32个GPU跨8台机器的20小时。

## 7. 实验

Deep Crossing 在一个主要搜索引擎的展示和点击日志上进行训练和评估。表2列出了本节实验中使用的数据集。每个实验将使用训练、验证和测试数据集的组合，由"数据集"列中的名称引用。只有同一"组"中的数据集是兼容的，这意味着不同"类型"之间没有时间重叠。例如，使用 all_cp1_tn_b（表中最后一行）训练的模型可以使用 all_cp1_vd 验证并使用 all_cp1_tt 测试，因为它们都属于 G3 组。

数据集的"任务"是 CP1 或 CP2。这两个任务代表点击预测管道中的两个不同模型。在这种情况下，CP1 是两者中更关键的模型，但模型在系统中是互补的。

图4中描述的 Deep Crossing 模型用于所有实验。模型参数也分别固定为256、512、512、256、128和64，用于 Dim_E 和 Dim_CROSS[1-5]。鼓励读者为他们的特定应用尝试这些参数，包括层数，以获得最佳结果。

注意，与基线比较的所有实验结果由于测试数据中使用的海量样本而具有统计显著性。为简单起见，我们将不再单独说明。

| 数据集 | 组 | 任务 | 类型 | 行数（百万） | 维度（千） |
|--------|-----|------|------|-------------|-----------|
| text cp1 tn s | G1 | CP1 | train | 194 | 98.5 |
| text cp1 vd | G1 | CP1 | valid | 49 | 98.5 |
| text cp1 tt | G1 | CP1 | test | 45 | 98.5 |
| text cp1 tn b | G1 | CP1 | train | 2,930 | 98.5 |
| text cp2 tn | G2 | CP2 | train | 518 | 98.5 |
| text cp2 vd | G2 | CP2 | valid | 100 | 98.5 |
| text cp2 tt | G2 | CP2 | test | 93 | 98.5 |
| all cp1 tn s | G3 | CP1 | train | 111 | 202.7 |
| all cp1 vd | G3 | CP1 | valid | 139 | 202.7 |
| all cp1 tt | G3 | CP1 | test | 156 | 202.7 |
| all cp1 tn b | G3 | CP1 | train | 2,237 | 202.7 |

**表2：本文使用的数据集，其中行数（百万）是数据集中的样本数，维度（千）是特征维度的总数**

### 7.1 文本输入对的性能

如第5.3节简要提到的，我们有兴趣在 DSSM 专门化的设置中比较 DSSM 和 Deep Crossing。为了进行公平比较，我们在 CP1 和 CP2 的数据上训练了 DSSM 和 Deep Crossing 模型，但将 Deep Crossing 模型限制为与 DSSM 相同的数据（即两者都使用包含查询文本和关键词或标题文本的一对输入，每个输入由三字母克向量表示）。

在第一个实验中，在表3中列出的两个数据集上训练了任务 CP1 的点击预测模型。在两个数据集上，Deep Crossing 在相对 AUC 方面优于 DSSM。注意，这里使用的 DSSM 模型是第5.3节详述的对数损失版本。

| 训练数据 | DSSM | Deep Crossing |
|----------|------|---------------|
| text cp1 tn s | 100 | 100.46 |
| text cp1 tn b | 100 | 101.02 |

**表3：使用一对文本输入的任务 CP1 点击预测结果，其中性能以 DSSM 为基线的相对 AUC 衡量**

在第二个实验中，两个模型都在 text_cp2_tn 数据集上训练任务 CP2，并在 text_cp2_tt 数据集上测试。表4显示了 DSSM、Deep Crossing 和当前在我们生产系统中运行的模型的性能结果。生产模型在不同的数据集上训练，但使用运行时记录的预测输出在相同数据（text_cp2_tt）上测试。可以看出，Deep Crossing 比 DSSM 表现更好，但比生产模型差。这是预期的，因为生产模型使用了更多特征——包括组合特征——并且已经过多年优化。尽管如此，仅使用单独的查询和标题特征，Deep Crossing 距离生产模型只有约一个百分点。

| 测试数据 | DSSM | Deep Crossing | 生产模型 |
|----------|------|---------------|----------|
| text cp2 tt | 98.68 | 98.99 | 100 |

**表4：使用一对文本输入的任务 CP2 点击预测结果，其中性能以生产模型为基线的相对 AUC 衡量**

虽然我们已经展示了 Deep Crossing 从简单文本输入对中学习的能力，但这不是它的主要目标。Deep Crossing 的真正力量在于处理许多单独特征，正如我们将在后续实验中看到的。

### 7.2 超越文本输入

我们现在考虑 Deep Crossing 在任务 CP1（训练集 all_cp1_tn_s）上的性能，使用约二十几个特征，包括表1中列出的那些。本小节的实验没有外部基线，我们将只比较 Deep Crossing 在不同特征组合下的性能。这里的目标不是评估相对于其他方法的性能，而是观察 Deep Crossing 的实际表现，并展示其性能如何随着特征的添加和移除而显著变化。我们将在下一小节中使用相同的丰富特征集将 Deep Crossing 的性能与生产模型进行比较。

在第一个实验中，我们多次运行 Deep Crossing，打开和关闭不同的特征集。计数特征始终关闭，并将在下一个实验中重新打开。图5中的 All_features 模型打开了所有剩余特征。正如预期的那样，它在本实验的所有模型中具有最低的对数损失。Only_Q_K 模型具有最高的对数损失。该模型仅使用查询文本和关键词文本，类似于前一小节中的模型。

注意，图5中的对数损失是相对对数损失，定义为实际对数损失除以所有时期中 All_features 模型的最低对数损失。Only_Q_K 和 All_features 之间的相对对数损失差距约为0.12。这在 AUC 方面大约是7-8%的改进，考虑到 AUC 的0.1-0.3%改进通常被认为是点击预测模型的重大改进，这是巨大的。

Without_Q_K_T 是一个移除了查询文本、关键词文本和标题文本的模型，这意味着移除大部分文本特征将使相对对数损失增加0.025。这是 Without_position 模型（移除了位置特征）增加量的一半。

在第二个实验中，我们研究了计数特征如何与其他单独特征交互。如第4节所讨论的，计数特征在降低高基数特征的维度方面发挥着重要作用。我们的完整特征集有五种类型的计数特征。在这个实验中，我们只打开了其中一种（此后称为选定的计数特征）来展示效果。

从图6a可以看出，仅使用选定计数特征的 Counting_only 模型与 All_without_counting 模型（具有除计数特征外的所有特征）相比非常弱。图6b显示了添加选定计数特征后的结果，其中新模型 All_with_counting 将相对对数损失降低了0.02。图6a和图6b的相对对数损失基点是 All_without_counting 模型在所有时期中的最低对数损失。

### 7.3 与生产模型的比较

至此，问题仍然是 Deep Crossing 是否真的能击败生产模型，这是最终的基线。为了回答这个问题，我们使用22亿个样本训练了一个 Deep Crossing 模型，使用了生产模型原始特征的一个子集。如表5所报告，新模型在任务 CP1 的离线 AUC 上轻松超越了生产模型（任务 CP2 的结果在发表时不可用）。它在任务 CP1（all_cp1_tn_b）数据集上训练，并在 all_cp1_tt 上测试。生产模型在不同的（更大的）数据集上训练，但使用相同的数据测试。如第7.1节中与任务 CP2 模型的比较，生产 AUC 基于运行时记录的预测输出。

| 测试数据 | Deep Crossing | 生产模型 |
|----------|---------------|----------|
| all cp1 tt | 101.02 | 100 |

**表5：与生产模型在任务 CP1 上的点击预测模型比较，其中性能以生产模型为基线的相对 AUC 衡量**

上述结果非常重要，因为 Deep Crossing 模型仅使用了一小部分特征，并且构建和维护所需的工作量少得多。

## 8. 结论与未来工作

Deep Crossing 实现了无需大量特征工程的Web-Scale建模。它表明，随着深度学习算法、建模语言和基于 GPU 的基础设施的最新进展，对于大规模复杂建模任务存在近乎"傻瓜式"的解决方案。虽然此类声明需要进一步测试，但它确实与深度学习先驱们所设想的一个关键优势相呼应，即将人们从繁琐的特征工程工作中解放出来。

Deep Crossing 最初是在付费搜索广告的背景下开发的，这是一种拥有海量数据的Web应用。当应用于其他领域时，我们预计大多数属性仍将保持。这是因为特征类型的广泛覆盖和通用的模型表示。

除了节省构建大型模型的工作量外，Deep Crossing 还有助于将应用领域与建模技术解耦。手工构建组合特征需要广泛的领域知识。但如果将建模复杂性从特征工程转移到建模技术，问题就向全世界的建模专家开放了，甚至包括那些不在同一领域工作的专家。作为证据，由语音识别领域的几位研究人员开发的 BMUF 分布式训练算法帮助 Deep Crossing 弥合了与赞助搜索生产模型的最后一点性能差距。虽然我们没有进行明确的经济研究，但减少特征工程工作量和从他人获得更多帮助的复合效应预计对广泛的应用具有重要意义。

Deep Crossing 在设计上是一个用于异构特征的多路融合引擎。随着我们将其应用于更多应用领域，观察这一方面如何充分展现将是有趣的。

## 9. 致谢

作者感谢 Chris Basoglu、Yongqiang Wang、Jian Sun、Xiaodong He、Jianfeng Gao 和 Qiang Huo 的支持和讨论，这些对 Deep Crossing 的开发有所裨益。

## 10. 参考文献

[1] A. Agarwal, E. Akchurin, C. Basoglu, G. Chen, S. Cyphers, J. Droppo, A. Eversole, B. Guenter, M. Hillebrand, T. R. Hoens, X. Huang, Z. Huang, V. Ivanov, A. Kamenev, P. Kranen, O. Kuchaiev, W. Manousek, A. May, B. Mitra, O. Nano, G. Navarro, A. Orlov, M. Padmilac, H. Parthasarathi, B. Peng, A. Reznichenko, F. Seide, M. L. Seltzer, M. Slaney, A. Stolcke, H. Wang, Y. Wang, K. Yao, D. Yu, Y. Zhang, and G. Zweig. An introduction to computational networks and the computational network toolkit. Technical report, Microsoft Technical Report MSR-TR-2014-112, 2014.

[2] K. Chen and Q. Huo. Scalable training of deep learning machines by incremental block training with intra-block parallel optimization and blockwise model-update filtering. In Internal Conference on Acoustics, Speech and Signal Processing, 2016.

[3] R. Collobert, J. Weston, L. Bottou, M. Karlen, K. Kavukcuoglu, and P. Kuksa. Natural language processing (almost) from scratch. The Journal of Machine Learning Research, 12:2493–2537, 2011.

[4] G. E. Dahl, D. Yu, L. Deng, and A. Acero. Context-dependent pre-trained deep neural networks for large-vocabulary speech recognition. Audio, Speech, and Language Processing, IEEE Transactions on, 20(1):30–42, 2012.

[5] B. Edelman, M. Ostrovsky, and M. Schwarz. Internet advertising and the generalized second price auction: Selling billions of dollars worth of keywords. Technical report, National Bureau of Economic Research, 2005.

[6] K. Fukushima. Neocognitron: A self-organizing neural network model for a mechanism of pattern recognition unaffected by shift in position. Biol. Cybernetics, 36:193–202, 1980.

[7] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. arXiv preprint arXiv:1512.03385, 2015.

[8] G. Hinton, L. Deng, D. Yu, G. E. Dahl, A.-r. Mohamed, N. Jaitly, A. Senior, V. Vanhoucke, P. Nguyen, T. N. Sainath, et al. Deep neural networks for acoustic modeling in speech recognition: The shared views of four research groups. Signal Processing Magazine, IEEE, 29(6):82–97, 2012.

[9] G. E. Hinton, S. Osindero, and Y.-W. Teh. A fast learning algorithm for deep belief nets. Neural computation, 18(7):1527–1554, 2006.

[10] P.-S. Huang, X. He, J. Gao, L. Deng, A. Acero, and L. Heck. Learning deep structured semantic models for web search using clickthrough data. In Proceedings of the 22nd ACM international conference on Conference on information & knowledge management, pages 2333–2338. ACM, 2013.

[11] A. Krizhevsky, I. Sutskever, and G. E. Hinton. Imagenet classification with deep convolutional neural networks. In Advances in neural information processing systems, pages 1097–1105, 2012.

[12] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11):2278–2324, 1998.

[13] D. G. Lowe. Object recognition from local scale-invariant features. In Computer vision, 1999. The proceedings of the seventh IEEE international conference on, volume 2, pages 1150–1157. IEEE, 1999.

[14] T. Mikolov, I. Sutskever, K. Chen, G. S. Corrado, and J. Dean. Distributed representations of words and phrases and their compositionality. In Advances in neural information processing systems, pages 3111–3119, 2013.

[15] S. Rendle. Factorization machines. In Data Mining (ICDM), 2010 IEEE 10th International Conference on, pages 995–1000. IEEE, 2010.

[16] J. Schmidhuber. Deep learning in neural networks: An overview. Neural Networks, 61:85–117, 2015.

[17] F. Seide, G. Li, X. Chen, and D. Yu. Feature engineering in context-dependent deep neural networks for conversational speech transcription. In Automatic Speech Recognition and Understanding (ASRU), 2011 IEEE Workshop on, pages 24–29. IEEE, 2011.

[18] F. Seide, G. Li, and D. Yu. Conversational speech transcription using context-dependent deep neural networks. In Interspeech, pages 437–440, 2011.

[19] Y. Shen, X. He, J. Gao, L. Deng, and G. Mesnil. Learning semantic representations using convolutional neural networks for web search. In Proceedings of the companion publication of the 23rd international conference on World wide web companion, pages 373–374. International World Wide Web Conferences Steering Committee, 2014.

[20] D. Yu and L. Deng. Deep learning and its applications to signal and information processing [exploratory dsp]. Signal Processing Magazine, IEEE, 28(1):145–154, 2011.

[21] D. Yu, G. Hinton, N. Morgan, J.-T. Chien, and S. Sagayama. Introduction to the special section on deep learning for speech and language processing. Audio, Speech, and Language Processing, IEEE Transactions on, 20(1):4–6, 2012.
