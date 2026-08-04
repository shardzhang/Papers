# GateNet：门控增强的深度网络用于点击率预测

> Tongwen Huang, Qingyun She, Zhiqiang Wang, Junlin Zhang | Sina Weibo & Tencent

---

## 摘要

广告和推送排序对许多互联网公司（如Facebook）至关重要。在许多实际的广告和推送排序系统中，点击率（CTR）预测扮演着核心角色。近年来，许多基于神经网络的CTR模型被提出并取得了成功，例如因子分解机支持的神经网络、DeepFM和xDeepFM。其中许多模型包含两个常用组件：嵌入层和MLP隐藏层。另一方面，门控机制也广泛应用于计算机视觉（CV）和自然语言处理（NLP）等多个研究领域。一些研究已经证明，门控机制提高了非凸深度神经网络的训练能力。受这些观察的启发，我们提出了一种名为GateNet的新模型，该模型分别将特征嵌入门或隐藏门引入DNN CTR模型的嵌入层或隐藏层。特征嵌入门提供了一个可学习的特征门控模块，用于从特征级别选择显著的潜在信息。隐藏门帮助模型更有效地隐式捕获高阶交互。在三个真实世界数据集上进行的大量实验证明了其在提升各种最先进模型（如FM、DeepFM和xDeepFM）性能方面的有效性。
本文介绍了 GateNet：门控增强的深度网络用于点击率预测。核心内容：


关键发现：


## 1 引言

广告和推送排序对许多互联网公司（如Facebook）至关重要。这些任务背后的主要技术是点击率预测，即CTR。在该领域已经提出了许多模型，如逻辑回归（LR）、多项式-2（Poly2）、基于树的模型、基于张量的模型、贝叶斯模型和基于因子分解机的模型。

随着深度学习在计算机视觉和自然语言处理等多个研究领域取得的巨大成功，近年来提出了许多基于深度学习的CTR模型。其中许多模型包含两个常用组件：嵌入层和MLP隐藏层。另一方面，门控机制也广泛应用于计算机视觉（CV）和自然语言处理（NLP）等多个研究领域。一些研究工作已经证明，门控机制提高了非凸深度神经网络的训练能力。受这些观察启发，提出了一种名为GateNet的模型，用于从特征级别选择显著的潜在信息，并更有效地隐式捕获高阶交互以用于CTR预测。

我们的主要贡献如下：

- 我们提出了特征嵌入门层来替代传统嵌入并增强模型能力。将特征嵌入门插入到许多经典模型（如FM、DeepFM、DNN和XDeepFM）的嵌入层中，我们观察到了显著的性能提升。
- MLP层是规范DNN模型中隐式捕获高阶特征交互的重要组件，我们将隐藏门引入深度模型的MLP部分，提高了经典模型的性能。
- 通过插入隐藏门来增强标准DNN模型既简单又有效，并且我们可以实现与其他最先进模型基线（如DeepFM和XDeepFM）相当的性能。

本文的其余部分组织如下。第2节回顾了与我们所提模型相关的工作，随后在第3节介绍我们所提的模型。第4节将介绍在三个真实世界数据集上的实验探索。最后，我们在第5节总结本工作。

## 2 相关工作

### 2.1 基于深度学习的CTR模型

近年来提出了许多基于深度学习的CTR模型。如何有效地建模特征交互是大多数基于神经网络的模型的关键因素。因子分解机支持的神经网络（FNN）是一种使用FM预训练嵌入层的前馈神经网络。然而，FNN只能捕获高阶特征交互。Wide & Deep模型（WDL）联合训练宽线性模型和深度神经网络，以结合推荐系统中记忆和泛化的优势。然而，WDL的宽部分输入仍然需要专业特征工程。为了减轻特征工程中的人工努力，DeepFM用FM替换了WDL的宽部分，并在FM和深度组件之间共享特征嵌入。此外，Deep & Cross Network（DCN）和极限深度因子分解机（xDeepFM）是最近的深度学习方法，它们显式地建模特征交互。

### 2.2 深度学习中的门控机制

门控机制广泛应用于许多深度学习领域，如计算机视觉（CV）、自然语言处理（NLP）和推荐系统。

门控机制用于计算机视觉，例如Highway Network，它们利用变换门和携带门来分别表示通过变换输入产生的输出和携带的输出有多少。

门控机制广泛应用于NLP，例如LSTM、GRU、语言建模、序列到序列学习，它们利用门来防止梯度消失和解决长期依赖问题。

此外，[18]在推荐系统中使用门来自动调整建模共享信息和建模任务特定信息之间的参数。另一个应用门控机制的推荐系统是层次门控网络（HGN），它们应用特征级别和实例级别的门控模块来自适应地控制哪些item潜在特征以及哪些相关item可以传递到下游层。

## 3 我们提出的模型

深度学习模型广泛应用于工业推荐系统，如WDL、YouTubeNet和DeepFM。DNN模型是许多当前DNN排序系统的子组件，其网络结构如图1左侧所示。

我们可以在大多数当前DNN排序系统中找到两个常用组件：嵌入层和MLP隐藏层。我们旨在增强模型能力，并提出了名为GateNet的模型用于CTR预测任务。首先，我们提出了特征嵌入门层，它可以将嵌入特征转换为门感知嵌入特征，并有助于从特征级别选择显著的潜在信息。其次，我们还提出了隐藏门，它可以自适应地控制哪些潜在特征以及哪些相关的特征交互可以传递到下游层。带有特征嵌入门的DNN模型和带有隐藏门的DNN模型如图1的中部和右侧所示。在以下小节中，我们将详细描述GateNet中的特征嵌入层和隐藏门层。

图1：我们提出的GateNet的架构。左图是标准DNN网络，中图是带有特征嵌入门的模型，右图是带有隐藏门的深度模型。

### 3.1 特征嵌入门

稀疏输入层和嵌入层广泛应用于基于深度学习的CTR模型，如DeepFM。稀疏输入层对原始输入特征采用稀疏表示。

嵌入层能够将稀疏特征嵌入到低维、稠密的实值向量中。嵌入层的输出是一个宽的拼接域嵌入向量：

E = [e1, e2, · · · , ei, · · · , ef]

其中f表示域的数量，ei \in Rk表示第i个域的嵌入，k是嵌入层的维度。

另一方面，最近的研究结果表明，门可以改善非凸深度神经网络训练中的可训练性。在本工作中，我们首先提出特征嵌入门，用于在DeepCTR模型中的特征级别选择显著的潜在信息。特征嵌入门的基本步骤可描述如下：

首先，对于每个域嵌入ei，我们计算表示嵌入特征级别重要性的门控值。我们将此步骤形式化为以下公式：

gi = \sigma(Wi · ei) (1)

其中\sigma是门的激活函数，ei \in Rk是原始嵌入，Wi是第i个门的学习参数，总的学习参数矩阵W = [W1, · · ·, Wi, · · ·, Wf]，i = 1, · · ·, f。

其次，我们将门控值分配给相应的特征嵌入，并生成门感知嵌入。

gei = ei \odot gi (2)

其中\odot表示Hadamard或逐元素乘积，ei \in Rk是第i个原始嵌入，i = 1, · · ·, f。

第三，我们收集所有门感知嵌入并将其视为门控特征嵌入。

GE = [ge1, ge2, · · ·, gei, · · ·, gef] (3)

通常的做法是使门输出一个标量，表示整个特征嵌入的重要性。为了学习特征嵌入中位级别的显著重要信息，我们可以使该门输出一个包含特征嵌入细粒度信息的向量。我们将这种嵌入门称为"位级"门，将普通门称为"向量级"门。向量级和位级的特征嵌入门如图2所示。

图2：特征嵌入门。左图表示向量级特征嵌入门，右图是位级特征嵌入门。

从图中可以看出，我们比较向量级特征门和位级特征门的差异如下：
向量级：gi \in R, Wi \in Rk $\times$ 1, W \in Rf $\times$ k $\times$ 1
位级：gi \in Rk, Wi \in Rk $\times$ k, W \in Rf $\times$ k $\times$ k

我们可以看到，位级门的输出是一个向量，与特征嵌入的每个位相关，而向量级门可以看作是对特征嵌入的每个位使用相同的值。向量级和位级特征嵌入门的性能比较将在第4.2节讨论。

此外，如FiBiNet等先前工作所做的那样，我们将探索特征嵌入门层的参数共享机制。特征嵌入门层中的每个门都有其自己的参数来显式学习显著的特征信息，我们也可以使所有门共享参数以减少参数数量。我们将这种门称为"域共享"，将之前的门称为"域私有"。从数学角度来看，"域共享"和"域私有"之间的最大区别在于学习的门参数Wi。在"域共享"中，Wi在所有域之间共享，而在"域私有"中，每个域的Wi不同。"域共享"和"域私有"的性能将在第4.2节进行比较。

### 3.2 隐藏门

许多DNN排序系统的深度部分通常由多个全连接层组成，这些层隐式捕获高阶特征交互。如图1所示，深度网络的输入是嵌入层的展平。令a(0) = [ge1, · · ·, gei, · · ·, gef]表示嵌入层的输出，其中gei \in Rk表示第i个特征嵌入。然后，a(0)被输入多层感知器网络，前馈过程为：

a(l) = \sigma(W(l)a(l−1) + b(l)) (4)

其中l是深度，\sigma是激活函数。W(l)、b(l)、a(l)分别是第l层的模型权重、偏置和输出。

图3：隐藏门层

与位级特征嵌入门类似，我们提出了可应用于隐藏层的隐藏门。如图3所示，我们按如下方式使用此门：

g(l) = a(l) \odot \sigmag(Wg(l)a(l)) (5)

其中\odot表示逐元素乘积，\sigmag是门激活函数，Wg(l)是隐藏门的第l层参数。同样，我们可以像经典DNN模型一样堆叠多个隐藏门层。

### 3.3 输出层

综上所述，我们给出所提模型输出的整体公式为：

ŷ = \sigma(W|L|g|L| + b|L|) (6)

其中ŷ \in (0, 1)是CTR的预测值，\sigma是sigmoid函数，b|L|是偏置，|L|是DNN的深度。学习过程旨在最小化以下目标函数（交叉熵）：

loss = −(1/N) \sum(yi log(ŷi) + (1 − yi) ∗ log(1 − ŷi)) (7)

其中yi是第i个实例的ground truth，ŷi是预测的CTR，N是样本总数。

## 4 实验

在本节中，我们进行大量实验来回答以下研究问题：

（RQ1）特征嵌入门能否增强基线模型的能力？
（RQ2）隐藏门能否增强基线模型的能力？
（RQ3）我们能否在一个模型中将两个门结合起来以取得进一步改进？
（RQ4）网络设置如何影响我们模型的性能？

在介绍一些基本实验设置后，我们将回答这些问题。

### 4.1 实验测试平台和设置

#### 4.1.1 数据集。
1) **Criteo**。Criteo¹数据集广泛用于许多CTR模型评估。它包含4500万条数据实例的点击日志。Criteo数据集中有26个匿名分类域和13个连续特征域。我们将数据集随机分为两部分：90%用于训练，其余用于测试。
2) **ICME**。ICME²数据集包含若干天的短视频点击数据。在track2中，它包含1900万条数据实例的点击日志。对于每条点击数据，我们选择5个域（user_id、user_city、item_id、author_id、item_city）来预测短视频的点赞概率。我们将其随机分为两部分：70%用于训练，其余用于测试。
3) **SafeDriver**。SafeDriver³数据集用于预测汽车保险保单持有人提出索赔的概率。SafeDriver数据集中有57个匿名字段，这些特征被分为相似的组：二元特征、分类特征、连续特征和有序特征。它包含59.5万条数据实例。我们将数据集随机分为两部分：90%用于训练，其余用于测试。

#### 4.1.2 评估指标。
在我们的实验中，我们采用AUC作为指标。AUC是评估分类问题时广泛使用的指标。此外，一些工作验证了AUC在CTR预测中是一个很好的度量。AUC对分类阈值和正例比例不敏感。AUC的上限是1，越大越好。

#### 4.1.3 基线方法。
为了验证添加的门层在各种主流模型中的效果，我们选择了一些广泛使用的CTR模型作为基线模型，包括FM、DNN、DeepFM和XDeepFM。

本工作的主要目标不是提出一个新模型，而是通过我们提出的门控机制来增强这些基线模型。请注意，AUC提升1‰通常被认为对CTR预测至关重要，因为如果公司拥有非常大的用户群，这将带来公司收入的大幅增长。

#### 4.1.4 实现细节。
我们在实验中用Tensorflow⁴实现所有模型。对于嵌入层，嵌入层的维度设置为10。对于优化方法，我们使用Adam，小批量大小为1000，学习率设置为0.0001。对于所有深度模型，层深度设置为3，所有激活函数为RELU，每层神经元数为400，dropout率设置为0.5。特征嵌入门的默认激活函数是Sigmoid，隐藏门的激活函数是Tanh。我们使用2块Tesla K40 GPU进行实验。

### 4.2 特征嵌入门的性能（RQ1）

在本小节中，我们展示了在将特征嵌入门插入典型嵌入层后所选基线模型的性能提升。实验在Criteo、ICME和SafeDriver数据集上进行，结果如表1所示。

表1：将特征嵌入门插入典型嵌入层后基线模型的整体性能提升。除非论文中特别说明，'field private'和'vec-wise'模型被用作嵌入门模型的默认设置。后缀'e'表示对该模型应用了嵌入门。

| 模型 | ICME | Criteo | SafeDriver |
|------|------|--------|------------|
| FM | 0.8696 | 0.7923 | 0.6302 |
| FMe | 0.8973 | 0.7970 | 0.6327 |
| $\Delta$ | 0.0277 | 0.0047 | 0.0025 |
| DNN | 0.8912 | 0.8067 | 0.6344 |
| DNNe | 0.9166 | 0.8096 | 0.6359 |
| $\Delta$ | 0.0254 | 0.0029 | 0.0015 |
| DeepFM | 0.9027 | 0.8087 | 0.6276 |
| DeepFMe | 0.9097 | 0.8097 | 0.6349 |
| $\Delta$ | 0.0070 | 0.0010 | 0.0073 |
| XDeepFM | 0.9052 | 0.8091 | 0.6324 |
| XDeepFMe | 0.9178 | 0.8098 | 0.6336 |
| $\Delta$ | 0.0126 | 0.0007 | 0.0012 |

将特征嵌入门插入这些基线模型中，我们发现我们提出的嵌入门机制可以一致地提升基线模型在这三个数据集上的性能，如表1所示。这些结果表明，从特征级别仔细选择显著的潜在信息有助于增强模型能力，并使基线模型获得更好的性能。在所有基线模型中，带有特征嵌入门的FM获得了显著提升，在ICME数据集上比经典FM模型高出近2%。我们推测FM是一个浅层模型，只有一组潜在向量需要学习，FM中没有其他组件显式或隐式地调整特征，因此门控机制是调整特征权重的好方法。与FM不同，还有许多深度模型如DeepFM和XDeepFM，我们的带有特征嵌入门的模型可以增强这些模型的能力并取得进一步的改进。

此外，我们设计了关于特征嵌入门的一些进一步研究。首先，我们在表2中进行了一些实验来比较门的参数共享机制（'field sharing'和'field private'）。

表2：特征嵌入门的参数共享机制：field private vs field sharing。

| 模型 | ICME Private | ICME Share | Criteo Private | Criteo Share |
|------|-------------|------------|----------------|--------------|
| FM | 0.8973 | 0.8861 | 0.7970 | 0.7957 |
| DNN | 0.9166 | 0.9076 | 0.8096 | 0.8099 |
| DeepFM | 0.9097 | 0.8985 | 0.8097 | 0.8098 |
| XDeepFM | 0.9178 | 0.9039 | 0.8098 | 0.8096 |

从表2中我们可以发现，对于ICME数据集上的许多基础模型，'field private'门的性能远优于'field sharing'门，而在Criteo数据集上不显著。虽然'field sharing'可以减少学习参数的数量，但性能也会下降。这些结果表明，不同参数共享门机制的性能取决于具体任务。总体而言，在我们的实验中选择'field private'是更好的选择。

其次，我们进行了一些实验来探索向量级和位级的特征嵌入门。表3中的结果显示了对比。

表3：嵌入门机制：向量级 vs 位级。默认门共享机制是'field private'。

| 模型 | ICME vec-wise | ICME bit-wise | Criteo vec-wise | Criteo bit-wise |
|------|---------------|---------------|-----------------|-----------------|
| FM | 0.8973 | 0.8937 | 0.7970 | 0.7985 |
| DNN | 0.9166 | 0.9018 | 0.8096 | 0.8098 |
| DeepFM | 0.9097 | 0.9112 | 0.8097 | 0.8098 |
| XDeepFM | 0.9178 | 0.9175 | 0.8098 | 0.8100 |

结果表明，在Criteo数据集上位级比向量级稍好，而在ICME数据上我们无法得出明确结论。这背后的原因需要进一步探索。

### 4.3 隐藏门的性能（RQ2）

在本小节中，将报告在表4中三个测试集上将隐藏门插入典型MLP层后所选基线模型的整体性能提升。

表4：带有隐藏门的基线模型的整体性能提升。后缀'h'表示对该模型应用了隐藏门。

| 模型 | ICME | Criteo | SafeDriver |
|------|------|--------|------------|
| DNN | 0.8912 | 0.8067 | 0.6344 |
| DNNh | 0.9105 | 0.8093 | 0.6348 |
| $\Delta$ | 0.0193 | 0.0026 | 0.0004 |
| DeepFM | 0.9027 | 0.8087 | 0.6276 |
| DeepFMh | 0.9121 | 0.8090 | 0.6324 |
| $\Delta$ | 0.0094 | 0.0003 | 0.0048 |
| XDeepFM | 0.9052 | 0.8091 | 0.6324 |
| XDeepFMh | 0.9084 | 0.8092 | 0.6344 |
| $\Delta$ | 0.0032 | 0.0001 | 0.0020 |

用隐藏门层替换传统MLP，我们提出的隐藏门机制一致地增强了这些基线模型，并在ICME、Criteo和SafeDriver数据集上取得了性能提升，如表4所示。实验结果表明，隐藏门帮助模型更有效地隐式捕获高阶交互。

虽然将隐藏门应用于MLP层很简单，但它是提升基线模型性能的有效方法。因此，我们在表5中进行了实验，比较隐藏门DNN与一些复杂的基线模型。

表5：比较通过插入隐藏门的标准DNN与其他基线模型的性能。Safe表示SafeDrive数据集，XDFM表示XDeepFM。

| 数据集 | DNN | DeepFM | XDFM | FiBiNet | DNNh |
|--------|-----|--------|------|---------|------|
| Criteo | 0.8063 | 0.8087 | 0.8091 | 0.8102 | 0.8093 |
| ICME | 0.8912 | 0.9027 | 0.9052 | 0.9030 | 0.9105 |
| Safe | 0.6344 | 0.6276 | 0.6324 | 0.6342 | 0.6348 |

从表5可以看出，通过插入隐藏门的标准DNN优于一些规范的深度学习模型，如DeepFM、XDeepFM。这是一种增强标准DNN以获得改进的简单方法，使得DNN模型在工业推荐系统中更加实用。

### 4.4 结合FE-Gate和Hidden Gate的模型性能（RQ3）

如前所述，我们发现特征嵌入门和隐藏门分别可以增强模型能力并获得良好的性能。我们能否在一个模型中将特征嵌入门和隐藏门结合起来以取得进一步的性能提升？我们进行了一些实验来回答这个研究问题，在Criteo和ICME数据集上。

表6：带有特征嵌入门和隐藏门的基线模型的整体性能。'EGate'、'HGate'、'Both'分别表示特征嵌入门、隐藏门以及特征嵌入门加隐藏门。

| 数据集 | 模型 | 基线 | EGate | HGate | Both |
|--------|------|------|-------|-------|------|
| ICME | DNN | 0.8912 | 0.9166 | 0.9105 | 0.9054 |
| ICME | DeepFM | 0.9027 | 0.9097 | 0.9121 | 0.9114 |
| ICME | XDeepFM | 0.9052 | 0.9178 | 0.9084 | 0.9054 |
| Criteo | DNN | 0.8067 | 0.8096 | 0.8093 | 0.8097 |
| Criteo | DeepFM | 0.8087 | 0.8097 | 0.8090 | 0.8097 |
| Criteo | XDeepFM | 0.8091 | 0.8098 | 0.8092 | 0.8098 |

从表6可以看出，在一个模型中组合特征嵌入门和隐藏门并不能获得进一步的性能提升。具体来说，在Criteo上没有太多性能提升，在ICME上有些性能下降。特征嵌入门可以影响隐式和显式特征交互，而隐藏门可以影响隐式特征交互，我们假设隐式特征交互已经被做了两次，导致隐式特征表示被破坏。这背后的真实原因需要进一步实验来证明这个假设。

### 4.5 超参数研究（RQ4）

我们进行了一些实验来研究所提出的门控机制中超参数的影响。我们在SafeDriver数据集上测试了我们所提出的GateNet的不同设置，并将DeepFM、DeepFMe和DeepFMh作为基线模型。

我们将超参数分为以下三个部分：
- **门激活函数**。嵌入门和隐藏门都包含门激活函数。
- **嵌入大小**。我们将嵌入大小从10更改为50，并比较基线模型与嵌入门模型的性能。
- **隐藏层**。我们将层数从2更改为6，并观察基线模型和隐藏门模型的性能。

表7：特征嵌入门和隐藏门中不同激活函数的整体性能。

| 激活函数 | DeepFMe | DeepFMh |
|-----------|---------|---------|
| Linear | 0.6356 | 0.6321 |
| Relu | 0.6343 | 0.6320 |
| Sigmoid | 0.6349 | 0.6324 |
| Tanh | 0.6320 | 0.6311 |

#### 4.5.1 门中的激活函数。
在SafeDriver数据集上使用特征嵌入门和隐藏门中不同激活函数的测试结果如表7所示。我们观察到，特征嵌入门中最好的激活函数是线性函数，而隐藏门中最好的激活函数是Tanh。

表8：特征嵌入门中不同嵌入大小的性能。

| 嵌入大小 | DeepFM | DeepFMe |
|----------|--------|---------|
| 10 | 0.6276 | 0.6349 |
| 20 | 0.6297 | 0.6322 |
| 30 | 0.6271 | 0.6319 |
| 40 | 0.6284 | 0.6329 |
| 50 | 0.6235 | 0.6307 |

#### 4.5.2 特征嵌入门中的嵌入大小。
我们将特征嵌入门中的嵌入大小从10更改为50，并将性能范围总结在表8中。从结果中，我们发现嵌入大小对GateNet影响很小。具体来说，标准DeepFM在嵌入大小为20时具有良好的性能，而DeepFMe的嵌入大小为10。因此，这些结果表明DeepFMe训练好模型所需的参数比DeepFM少。

#### 4.5.3 隐藏门中的层数。
在深度部分，我们可以改变每层的神经元数量、DNN的深度、激活函数和dropout率。为简洁起见，我们只研究DNN部分不同深度的影响。我们将隐藏门中的层数从2更改为6，并将性能总结在表9中。

表9：DNN中不同层数的性能。

| 层数 | DeepFM | DeepFMh |
|------|--------|---------|
| 2 | 0.6219 | 0.6328 |
| 3 | 0.6276 | 0.6324 |
| 4 | 0.6281 | 0.6312 |
| 5 | 0.6290 | 0.6321 |
| 6 | 0.6279 | 0.6286 |

增加层数，DeepFM的性能提高，而DeepFMh的性能下降。这些结果表明，在SafeDriver数据集上，我们的DeepFMh可以用更少的参数学习得比DeepFM更好。

## 5 结论

近年来，许多基于神经网络的CTR模型被提出，一些 recent 的研究发现门控机制可以提高非凸深度神经网络训练中的可训练性。受这些观察的启发，我们提出了一个名为GateNet的新模型，该模型分别将特征嵌入门或隐藏门引入DNN CTR模型的嵌入层或隐藏层。在三个真实世界数据集上进行的大量实验证明了它在提升各种最先进模型（如FM、DeepFM和xDeepFM）在三个真实世界数据集上的性能方面的有效性。

## 参考文献

[1] Heng-Tze Cheng, Levent Koc, Jeremiah Harmsen, Tal Shaked, Tushar Chandra, Hrishi Aradhye, Glen Anderson, Greg Corrado, Wei Chai, Mustafa Ispir, et al. 2016. Wide & deep learning for recommender systems. In Proceedings of the 1st Workshop on Deep Learning for Recommender Systems. ACM, 7–10.

[2] Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. 2014. Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation. arXiv:cs.CL/1406.1078

[3] Paul Covington, Jay Adams, and Emre Sargin. 2016. Deep neural networks for youtube recommendations. In Proceedings of the 10th ACM conference on recommender systems. ACM, 191–198.

[4] Yann N Dauphin, Angela Fan, Michael Auli, and David Grangier. 2017. Language modeling with gated convolutional networks. In Proceedings of the 34th International Conference on Machine Learning-Volume 70. JMLR. org, 933–941.

[5] Jonas Gehring, Michael Auli, David Grangier, Denis Yarats, and Yann N Dauphin. 2017. Convolutional sequence to sequence learning. In Proceedings of the 34th International Conference on Machine Learning-Volume 70. JMLR. org, 1243–1252.

[6] Felix A Gers, Jürgen Schmidhuber, and Fred Cummins. 1999. Learning to forget: Continual prediction with LSTM. (1999).

[7] Xavier Glorot and Yoshua Bengio. 2010. Understanding the difficulty of training deep feedforward neural networks. In Proceedings of the thirteenth international conference on artificial intelligence and statistics. 249–256.

[8] Thore Graepel, Joaquin Quinonero Candela, Thomas Borchert, and Ralf Herbrich. 2010. Web-scale bayesian click-through rate prediction for sponsored search advertising in microsoft's bing search engine. Omnipress.

[9] Huifeng Guo, Ruiming Tang, Yunming Ye, Zhenguo Li, and Xiuqiang He. 2017. Deepfm: a factorization-machine based neural network for ctr prediction. arXiv preprint arXiv:1703.04247 (2017).

[10] Xinran He, Junfeng Pan, Ou Jin, Tianbing Xu, Bo Liu, Tao Xu, Yanxin Shi, Antoine Atallah, Ralf Herbrich, Stuart Bowers, et al. 2014. Practical lessons from predicting clicks on ads at facebook. In Proceedings of the Eighth International Workshop on Data Mining for Online Advertising. ACM, 1–9.

[11] Tongwen Huang, Zhiqi Zhang, and Junlin Zhang. 2019. FiBiNET: combining feature importance and bilinear feature interaction for click-through rate prediction. In Proceedings of the 13th ACM Conference on RecSys 2019. 169–177.

[12] Yuchin Juan, Yong Zhuang, Wei-Sheng Chin, and Chih-Jen Lin. 2016. Field-aware factorization machines for CTR prediction. In Proceedings of the 10th ACM Conference on Recommender Systems. ACM, 43–50.

[13] Diederik P Kingma and Jimmy Ba. 2014. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980 (2014).

[14] Yehuda Koren, Robert Bell, and Chris Volinsky. 2009. Matrix factorization techniques for recommender systems. Computer 8 (2009), 30–37.

[15] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. 2012. Imagenet classification with deep convolutional neural networks. In Advances in neural information processing systems. 1097–1105.

[16] Jianxun Lian, Xiaohuan Zhou, Fuzheng Zhang, Zhongxia Chen, Xing Xie, and Guangzhong Sun. 2018. xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems. arXiv preprint arXiv:1803.05170 (2018).

[17] Chen Ma, Peng Kang, and Xue Liu. 2019. Hierarchical Gating Networks for Sequential Recommendation. arXiv preprint arXiv:1906.09217 (2019).

[18] Jiaqi Ma, Zhe Zhao, Xinyang Yi, Jilin Chen, Lichan Hong, and Ed H Chi. 2018. Modeling task relationships in multi-task learning with multi-gate mixture-of-experts. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 1930–1939.

[19] H Brendan McMahan, Gary Holt, David Sculley, Michael Young, Dietmar Ebner, Julian Grady, Lan Nie, Todd Phillips, Eugene Davydov, Daniel Golovin, et al. 2013. Ad click prediction: a view from the trenches. In Proceedings of the 19th ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 1222–1230.

[20] Tomáš Mikolov, Martin Karafiát, Lukáš Burget, Jan Černocký, and Sanjeev Khudanpur. 2010. Recurrent neural network based language model. In Eleventh Annual Conference of the International Speech Communication Association.

[21] Steffen Rendle. 2010. Factorization machines. In Data Mining (ICDM), 2010 IEEE 10th International Conference on. IEEE, 995–1000.

[22] Steffen Rendle. 2012. Factorization machines with libfm. ACM Transactions on Intelligent Systems and Technology (TIST) 3, 3 (2012), 57.

[23] Rupesh Kumar Srivastava, Klaus Greff, and Jürgen Schmidhuber. 2015. Highway networks. arXiv preprint arXiv:1505.00387 (2015).

[24] Ruoxi Wang, Bin Fu, Gang Fu, and Mingliang Wang. 2017. Deep & cross network for ad click predictions. In Proceedings of the ADKDD'17. ACM, 12.

[25] Weinan Zhang, Tianming Du, and Jun Wang. 2016. Deep learning over multi-field categorical data. In European conference on information retrieval. Springer, 45–57.
