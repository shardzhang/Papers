# AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks（中文翻译）

> 作者：Weiping Song*, Chence Shi, Zhiping Xiao, Zhijian Duan, Yewen Xu, Ming Zhang†, Jian Tang†
>
> 单位：北京大学计算机科学技术系、加州大学洛杉矶分校、Mila-Quebec AI Institute、HEC Montreal & CIFAR AI Chair
>
> 会议：CIKM 2019
>
> 原文：https://doi.org/10.1145/3357384.3357925

## 摘要

点击率（CTR）预测旨在预测用户点击广告或物品的概率，这对在线广告和推荐系统等许多在线应用至关重要。该问题极具挑战性，因为（1）输入特征（如用户ID、用户年龄、物品ID、物品类别）通常是稀疏且高维的，（2）有效的预测依赖于高阶组合特征（即交叉特征），而这些特征由领域专家手工构建非常耗时，且不可能全部枚举。因此，已有许多研究工作致力于为稀疏高维的原始特征及其有意义的组合寻找低维表示。

本文提出了一种名为AutoInt的高效方法，用于自动学习输入特征的高阶特征交互。我们提出的算法非常通用，可同时应用于数值型和类别型输入特征。具体地，我们将数值型和类别型特征映射到相同的低维空间中，然后使用带有残差连接的多头自注意力神经网络来显式建模低维空间中的特征交互。通过多层多头自注意力神经网络，可以对输入特征的不同阶次特征组合进行建模。整个模型可以端到端地高效拟合大规模原始数据。在四个真实世界数据集上的实验结果表明，我们提出的方法不仅在预测性能上超越了现有最先进的方法，而且还具有良好的可解释性。代码已开源：https://github.com/DeepGraphLearning/RecommenderSystems。

* 第一作者在访问Mila期间完成了部分工作。
† 通讯作者。

---

## 1 引言

预测用户点击广告或物品的概率（即点击率预测）是在线广告和推荐系统[8, 10, 15]等多个应用中的关键问题。预测性能直接影响业务提供商最终收入。由于其重要性，该问题在学术界和工业界都引起了越来越多的关注。

机器学习在点击率预测中扮演着关键角色，通常被形式化为以用户画像和物品属性作为输入特征的监督学习。该问题因以下几个原因极具挑战性。首先，输入特征极其稀疏且高维[8, 11, 13, 21, 32]。在现实应用中，相当大比例的用户人口统计学特征和物品属性通常是离散的和/或类别型的。为使监督学习方法适用，这些特征首先被转换为独热编码向量，这很容易导致特征维度达到数百万。以著名的CTR预测数据集Criteo¹为例，其特征维度约为3000万，稀疏度超过99.99%。面对如此稀疏且高维的输入特征，机器学习模型很容易过拟合。其次，大量文献[8, 11, 19, 32]表明，高阶特征交互²对于良好的性能至关重要。例如，向一个10岁男孩David推荐著名电子游戏《超级马里奥兄弟》是合理的。在这种情况下，三阶组合特征<性别=男, 年龄=10, 产品类别=电子游戏>对预测非常有信息量。然而，寻找这种有意义的高阶组合特征严重依赖于领域专家。此外，手工构建所有有意义的组合几乎是不可能的[8, 26]。有人可能会问，我们可以枚举所有可能的高阶特征，然后让机器学习模型选择有意义的那些。然而，枚举所有可能的高阶特征将指数级增加输入特征的维度和稀疏性，导致模型过拟合问题更加严重。因此，学术界和工业界已进行了大量研究，致力于为稀疏高维输入特征寻找低维表示，同时建模不同阶次的特征组合。

例如，分解机（FM）[26]将多项式回归模型与分解技术相结合，用于建模特征交互，并已被证明在各种任务中有效[27, 28]。然而，受限于其多项式拟合时间，它仅对建模低阶特征交互有效，而难以捕捉高阶特征交互。最近，许多基于深度神经网络的工作[8, 11, 13, 38]被提出用于建模高阶特征交互。具体地，通常使用多层非线性神经网络来捕捉高阶特征交互。然而，这类方法存在两个局限性。第一，全连接神经网络在学习乘法特征交互方面已被证明效率低下[4]。第二，由于这些模型以隐式方式学习特征交互，它们缺乏对哪些特征组合有意义的良好解释。因此，我们正在寻找一种方法，能够显式建模不同阶次的特征组合，将整个特征表示为低维空间，同时提供良好的模型可解释性。

¹ http://labs.criteo.com/2014/09/kaggle-contest-dataset-now-available-academic-use/
² 在本文中，我们将互换使用"组合特征"和"特征交互"，因为两者在文献中均有使用[11, 19, 32]。

本文基于多头自注意力机制[36]提出了这样一种方法。我们提出的方法学习了稀疏高维输入特征的有效低维表示，并适用于类别型和/或数值型输入特征。具体地，类别型和数值型特征首先被嵌入到低维空间中，这既降低了输入特征的维度，又允许不同类型的特征通过向量运算（如加法和内积）相互交互。之后，我们提出了一种新颖的交互层来促进不同特征之间的交互。在每个交互层中，每个特征允许与所有其他特征交互，并能够通过多头注意力机制[36]自动识别相关特征以形成有意义的更高阶特征。此外，多头机制将特征投影到多个子空间中，因此可以在不同子空间中捕捉不同的特征交互。这种交互层建模了特征之间的一步交互。通过堆叠多个交互层，我们能够建模不同阶次的特征交互。在实践中，残差连接[12]被添加到交互层中，从而允许结合不同阶次的特征组合。我们使用注意力机制来衡量特征之间的相关性，这提供了良好的模型可解释性。

总结而言，本文做出以下贡献：

- 我们研究了显式学习高阶特征交互的问题，并同时寻找具有良好可解释性的模型。
- 我们提出了一种基于自注意力神经网络的新方法，可以自动学习高阶特征交互并高效处理大规模高维稀疏数据。
- 我们在多个真实世界数据集上进行了大量实验。CTR预测任务上的实验结果表明，我们提出的方法不仅在预测性能上超越了现有最先进的方法，而且提供了良好的模型可解释性。

本文的组织结构如下。第2节总结相关工作。第3节正式定义问题。第4节介绍我们提出的学习特征交互的方法。第5节展示实验结果和详细分析。第6节总结本文并指出未来工作。

## 2 相关工作

我们的工作与三类研究相关：1）推荐系统和在线广告中的点击率预测；2）特征交互学习技术；3）深度学习文献中的自注意力机制和残差网络。

### 2.1 点击率预测

点击率预测对许多互联网公司至关重要，各公司已开发了多种系统[8-10, 15, 21, 29, 43]。例如，Google开发了用于推荐系统的Wide&Deep[8]学习系统，它结合了线性浅层模型和深度模型的优势。该系统在APP推荐中取得了显著性能。该问题也受到学术界的广泛关注。例如，Shan等人[31]提出了一种上下文感知的CTR预测方法，对<用户、广告、上下文>三阶张量进行分解。Oentaryo等人[24]开发了层次重要性感知分解机来建模广告的动态影响。

### 2.2 特征交互学习

学习特征交互是一个基础问题，因此在文献中得到了广泛研究。一个著名的例子是分解机（FM）[26]，它被提出主要用于捕捉一阶和二阶特征交互，并已被证明在推荐系统的许多任务中有效[27, 28]。之后，人们提出了分解机的各种变体。例如，场感知分解机（FFM）[16]建模了来自不同字段的特征之间的细粒度交互。GBFM[7]和AFM[40]考虑了不同二阶特征交互的重要性。然而，所有这些方法都聚焦于建模低阶特征交互。

最近有一些工作建模了高阶特征交互。例如，NFM[13]在二阶特征交互的输出之上堆叠深度神经网络以建模更高阶的特征。类似地，PNN[25]、FNN[41]、DeepCrossing[32]、Wide&Deep[8]和DeepFM[11]利用前馈神经网络来建模高阶特征交互。然而，所有这些方法都以隐式方式学习高阶特征交互，因此缺乏良好的模型可解释性。

相反，有三类工作以显式方式学习特征交互。第一，Deep&Cross[38]和xDeepFM[19]分别在比特级别和向量级别对特征进行外积运算。尽管它们执行了显式的特征交互，但解释哪些组合是有用的并不容易。第二，一些基于树的方法[39, 42, 44]结合了基于嵌入的模型和基于树的模型的优势，但需要将训练过程分成多个阶段。第三，HOFM[5]提出了用于高阶分解机的高效训练算法。然而，HOFM需要过多参数，实际上只能使用其低阶（通常小于5）形式。与现有工作不同，我们以端到端的方式使用注意力机制显式建模特征交互，并通过可视化来探究学习到的特征组合。

### 2.3 注意力机制与残差网络

我们提出的模型利用了深度学习文献中的最新技术：注意力机制[2]和残差网络[12]。注意力机制首先在神经机器翻译的背景下被提出[2]，并已被证明在问答[35]、文本摘要[30]和推荐系统[14, 33, 43]等多种任务中有效。Vaswani等人[36]进一步提出了多头自注意力机制来建模机器翻译中单词之间的复杂依赖关系。

残差网络[12]在ImageNet竞赛中取得了最先进的性能。由于残差连接可以简单地形式化为y = F(x) + x，它促进了梯度在内部层之间的流动，因此成为训练极深神经网络时流行的网络结构。

## 3 问题定义

我们首先正式定义点击率（CTR）预测问题如下：

**定义1（CTR预测）。** 设 x ∈ Rn 表示用户 u 的特征和物品 v 的特征的拼接，其中类别型特征用独热编码表示，n 是拼接后的特征维度。点击率预测问题旨在根据特征向量 x 预测用户 u 点击物品 v 的概率。

CTR预测的一个直接解决方案是将 x 视为输入特征，使用现成的分类器如逻辑回归。然而，由于原始特征向量 x 非常稀疏且高维，模型将很容易过拟合。因此，将原始输入特征表示为低维连续空间是值得期望的。此外，如现有文献所示，利用更高阶的组合特征对于获得良好的预测性能至关重要[6, 8, 11, 23, 26, 32]。

**图1：我们提出的AutoInt模型概述。嵌入层和交互层的细节分别如图2和图3所示。**

具体地，我们将高阶组合特征定义如下：

**定义2（p阶组合特征）。** 给定输入特征向量 x ∈ Rn，p阶组合特征定义为 g(xi1, ..., xip)，其中每个特征来自不同的字段，p 是涉及的特征字段数，g(·) 是一个非加性组合函数，例如乘法[26]和外积[19, 38]。例如，xi1 × xi2 是一个涉及 xi1 和 xi2 的二阶组合特征。

传统上，有意义的高阶组合特征由领域专家手工构建。然而，这非常耗时且难以泛化到其他领域。此外，手工构建所有有意义的更高阶特征几乎是不可能的。因此，我们旨在开发一种方法，能够自动发现有意义的更高阶组合特征，同时将所有特征映射到低维连续空间。形式化地，我们将问题定义如下：

**定义3（问题定义）。** 给定用于点击率预测的输入特征向量 x ∈ Rn，我们的目标是学习 x 的低维表示，该表示建模了高阶组合特征。

## 4 AutoInt：自动特征交互学习

在本节中，我们首先概述所提出的AutoInt方法，它可以自动学习CTR预测的特征交互。接下来，我们将全面描述如何学习一个建模高阶组合特征的低维表示，而无需手动特征工程。

### 4.1 概述

我们的方法目标是：将原始的稀疏高维特征向量映射到低维空间，同时建模高阶特征交互。如图1所示，我们的方法以稀疏特征向量 x 为输入，随后是一个嵌入层，将所有特征（即类别型和数值型特征）投影到相同的低维空间。接下来，我们将所有字段的嵌入输入到一个新颖的交互层中，该交互层实现为多头自注意力神经网络。在每个交互层中，通过注意力机制组合高阶特征，而多头机制（将特征映射到不同子空间）可以评估不同类型的组合。通过堆叠多个交互层，可以建模不同阶次的组合特征。

最后一个交互层的输出是输入特征的低维表示，它建模了高阶组合特征，并进一步通过Sigmoid函数用于估计点击率。接下来，我们将介绍我们提出的方法的细节。

**图2：输入层和嵌入层示意图，其中类别型和数值型字段均由低维稠密向量表示。**

### 4.2 输入层

我们首先将用户的画像和物品的属性表示为一个稀疏向量，即所有字段的拼接。具体地，

x = [x1; x2; ...; xM]                                  (1)

其中 M 是总的特征字段数，xi 是第 i 个字段的特征表示。如果第 i 个字段是类别型的（如图2中的 x1），则 xi 是一个独热向量。如果第 i 个字段是数值型的（如图2中的 xM），则 xi 是一个标量值。

### 4.3 嵌入层

由于类别型特征的特征表示非常稀疏且高维，一种常见的方法是将它们表示为低维空间（例如词嵌入）。具体地，我们将每个类别型特征表示为一个低维向量：

ei = Vi xi                                             (2)

其中 Vi 是字段 i 的嵌入矩阵，xi 是一个独热向量。通常类别型特征可以是多值的，即 xi 是一个多热向量。以电影观看预测为例，可能有一个名为"类型"（Genre）的特征字段，它描述电影的类型，并且可能是多值的（例如电影《泰坦尼克号》的"剧情"和"爱情"）。为了兼容多值输入，我们进一步修改了公式(2)，将多值特征字段表示为对应特征嵌入向量的平均值：

ei = (1/q) Vi xi                                       (3)

其中 q 是样本在第 i 个字段所具有的值的数量，xi 是该字段的多热向量表示。

为了允许类别型与数值型特征之间的交互，我们也将数值型特征表示在相同的低维特征空间中。具体地，我们将数值型特征表示为：

em = vm xm                                             (4)

其中 vm 是字段 m 的嵌入向量，xm 是一个标量值。

这样，嵌入层的输出将是多个嵌入向量的拼接，如图2所示。

### 4.4 交互层

一旦数值型和类别型特征位于相同的低维空间中，我们就开始在该空间中建模高阶组合特征。关键问题是确定哪些特征应该组合以形成有意义的更高阶特征。传统上，这由领域专家根据其知识创建有意义的组合来完成。在本文中，我们用一种新颖的方法——多头自注意力机制[36]——来解决这个问题。

多头自注意力网络[36]最近在建模复杂关系方面取得了显著性能。例如，它在机器翻译[36]和句子嵌入[20]中建模任意词依赖关系方面表现出优越性，并已成功应用于图嵌入中捕捉节点相似性[37]。我们在此将这一最新技术扩展到建模不同特征字段之间的相关性。

具体地，我们采用键-值注意力机制[22]来确定哪些特征组合是有意义的。以特征 m 为例，接下来我们解释如何识别涉及特征 m 的多个有意义的更高阶特征。我们首先定义在特定注意力头 h 下特征 m 和特征 k 之间的相关性如下：

α(h)m,k = exp(ψ(h)(em, ek)) / Σl=1M exp(ψ(h)(em, el))       (5)

ψ(h)(em, ek) = ⟨W(h)Query em, W(h)Key ek⟩

其中 ψ(h)(·,·) 是一个注意力函数，定义了特征 m 和 k 之间的相似度。它可以定义为一个神经网络，也可以简单如内积 ⟨·,·⟩。在本工作中，我们使用内积，因其简单且有效。W(h)Query ∈ Rd'×d 和 W(h)Key ∈ Rd'×d 是公式(5)中的变换矩阵，将原始嵌入空间 Rd 映射到新空间 Rd'。接下来，我们通过系数 α(h)m,k 的引导，组合所有相关特征来更新子空间 h 中特征 m 的表示：

ẽ(h)m = Σk=1M α(h)m,k (W(h)Value ek)                         (6)

其中 W(h)Value ∈ Rd'×d。由于 ẽ(h)m 是特征 m 及其相关特征（在头 h 下）的组合，它代表了我们方法学习到的一个新的组合特征。此外，一个特征也可能参与不同的组合特征，我们通过使用多个头来实现这一点，这些头创建不同的子空间并分别学习不同的特征交互。我们收集所有子空间中学习到的组合特征如下：

**图3：交互层的架构。组合特征依赖于注意力权重 α(h)m。**

ẽm = ẽ(1)m ⊕ ẽ(2)m ⊕ ... ⊕ ẽ(H)m                           (7)

其中 ⊕ 是拼接操作符，H 是总头数。

为了保留之前学习到的组合特征（包括原始的一阶特征），我们在网络中添加了标准的残差连接。形式化地，

eResm = ReLU(ẽm + WRes em)                                (8)

其中 WRes ∈ Rd'H×d 是在维度不匹配情况下的投影矩阵[12]，ReLU(z) = max(0, z) 是一个非线性激活函数。

通过这样的交互层，每个特征 em 的表示将被更新为新的特征表示 eResm，这是一个高阶特征的表示。我们可以堆叠多个这样的层，将前一个交互层的输出作为下一个交互层的输入。这样，我们就可以建模任意阶次的组合特征。

### 4.5 输出层

交互层的输出是一组特征向量 {eResm}Mm=1，其中包括通过残差块保留的原始个体特征和通过多头自注意力机制学习到的组合特征。对于最终的CTR预测，我们简单地将它们全部拼接，然后应用非线性投影如下：

ŷ = σ(wT(eRes1 ⊕ eRes2 ⊕ ... ⊕ eResM) + b)                 (9)

其中 w ∈ Rd'HM 是一个列投影向量，用于线性组合拼接后的特征，b 是偏置项，σ(x) = 1/(1+e-x) 将值转换为用户的点击概率。

### 4.6 训练

我们的损失函数是对数损失（Log loss），定义如下：

Logloss = -(1/N) Σj=1N (yj log(ŷj) + (1 - yj) log(1 - ŷj))   (10)

其中 yj 和 ŷj 分别是用户点击的真实值和估计的CTR值，j 索引训练样本，N 是训练样本总数。我们模型中需要学习的参数包括 {Vi, vm, W(h)Query, W(h)Key, W(h)Value, WRes, w, b}，通过梯度下降最小化总的Logloss来更新。

### 4.7 AutoInt分析

**建模任意阶组合特征。** 给定由公式(5)-(8)定义的特征交互操作，我们现在分析在我们的模型中低阶和高阶组合特征是如何建模的。

为简单起见，假设有四个特征字段（即M=4），分别记为 x1、x2、x3 和 x4。在第一个交互层内，每个特征通过注意力机制（公式(5)）与任何其他特征交互，因此一组二阶特征组合如 g(x1, x2)、g(x2, x3) 和 g(x3, x4) 以不同的相关性权重被捕捉，其中交互函数 g(·)（定义2中）的非加性属性可以通过激活函数 ReLU(·) 的非线性来保证。理想情况下，涉及 x1 的组合特征可以被编码到第一个特征字段的更新表示 eRes1 中。由于其他特征字段也可得到类似的结果，所有二阶特征交互都可以被编码在第一个交互层的输出中，其中注意力权重蒸馏出有用的特征组合。

接下来，我们证明更高阶的特征交互可以在第二个交互层中被建模。给定由第一个交互层生成的第一特征字段的表示 eRes1 和第三特征字段的表示 eRes3，涉及 x1、x2 和 x3 的三阶组合特征可以通过让 eRes1 关注 eRes3 来建模，因为 eRes1 包含交互 g(x1, x2)，而 eRes3 包含个体特征 x3（来自残差连接）。此外，组合特征的最大阶数随着交互层数量呈指数增长。例如，四阶特征交互 g(x1, x2, x3, x4) 可以通过 eRes1 和 eRes3 的组合来捕捉，它们分别包含二阶交互 g(x1, x2) 和 g(x3, x4)。因此，少数几个交互层就足以建模高阶特征交互。

基于以上分析，我们可以看到AutoInt以层次化的方式通过注意力机制学习特征交互，即从低阶到高阶，并且所有低阶特征交互通过残差连接传递。这是有前景且合理的，因为在计算机视觉和语音处理中，使用深度神经网络学习层次化表示已被证明非常有效[3, 18]。

**空间复杂度。** 嵌入层（神经网络方法中的共享组件[11, 19, 32]）包含 nd 个参数，其中 n 是输入特征稀疏表示的维度，d 是嵌入大小。一个交互层包含以下权重矩阵：{W(h)Query, W(h)Key, W(h)Value, WRes}，因此 L 层网络的参数数量为 L × (3dd' + d'Hd)，这与特征字段数 M 无关。最后，输出层有 d'HM + 1 个参数。就交互层而言，空间复杂度为 O(Ldd'H)。注意 H 和 d' 通常很小（例如在我们的实验中 H=2，d'=32），这使得交互层内存高效。

**时间复杂度。** 在每个交互层中，计算成本有两部分。首先，计算一个头的注意力权重需要 O(Mdd' + M2d') 时间。之后，在一个头下形成组合特征也需要 O(Mdd' + M2d') 时间。由于我们有 H 个头，总共需要 O(MHd'(M + d)) 时间。因为 H、d 和 d' 通常很小，所以这很高效。我们将在第5.2节中提供AutoInt的运行时间。

## 5 实验

在本节中，我们进一步评估所提出方法的有效性。我们旨在回答以下问题：

- RQ1: 我们提出的AutoInt在CTR预测问题上表现如何？对于大规模稀疏高维数据是否高效？
- RQ2: 不同模型配置的影响是什么？
- RQ3: 不同特征之间的依赖结构是什么？我们提出的模型是否可解释？
- RQ4: 整合隐式特征交互是否能进一步提升性能？

在回答这些问题之前，我们首先描述实验设置。

### 5.1 实验设置

#### 5.1.1 数据集

我们使用四个公开的真实世界数据集。数据集的统计信息总结在表1中。

**Criteo³** 是一个CTR预测的基准数据集，包含4500万用户对展示广告的点击记录。它包含26个类别型特征字段和13个数值型特征字段。

**Avazu⁴** 数据集包含用户的移动行为，包括用户是否点击了展示的移动广告。它有23个特征字段，涵盖从用户/设备特征到广告属性。

**KDD12⁵** 数据集由KDDCup 2012发布，最初旨在预测点击次数。由于我们的工作聚焦于CTR预测而非精确的点击次数，我们将此问题视为一个二分类问题（点击>0为正，无点击为0），这与FFM[16]类似。

**MovieLens-1M⁶** 数据集包含用户对电影的评分。在二值化过程中，我们将评分低于3的样本视为负样本（因为低分表示用户不喜欢该电影），评分大于3的样本视为正样本，并移除中性样本（即评分等于3）。

**数据准备。** 首先，我们移除不频繁的特征（出现在少于阈值的实例中），并将它们视为一个单独的特征"<unknown>"，其中Criteo、Avazu和KDD12数据集的阈值分别设置为{10, 5, 10}。其次，由于数值型特征可能存在较大方差并损害机器学习算法，我们通过将值 z 在 z > 2 时变换为 log2(z) 来归一化数值，该方法由Criteo竞赛的获胜者提出⁷。第三，我们随机选择80%的样本用于训练，并将剩余样本随机等分为验证集和测试集。

³ https://www.kaggle.com/c/criteo-display-ad-challenge
⁴ https://www.kaggle.com/c/avazu-ctr-prediction
⁵ https://www.kaggle.com/c/kddcup2012-track2
⁶ https://grouplens.org/datasets/movielens/
⁷ https://www.csie.ntu.edu.tw/~r01922136/kaggle-2014-criteo.pdf

#### 5.1.2 评估指标

我们使用两个流行指标来评估所有方法的性能。

**AUC（Area Under the ROC Curve）** 衡量CTR预测器将一个随机选择的正样本的得分排在一个随机选择的负样本之前的概率。AUC越高表示性能越好。

**Logloss** 由于所有模型都试图最小化公式(10)定义的Logloss，我们将其作为一个直接指标。

值得注意的是，CTR预测任务中AUC提升0.001或Logloss降低0.001级别即被视为显著，这一观点在现有工作中也有指出[8, 11, 38]。

#### 5.1.3 对比模型

我们将提出的方法与三类已有模型进行比较：（A）仅使用个体特征的线性方法；（B）基于分解机的方法，考虑二阶组合特征；（C）能够捕捉高阶特征交互的技术。我们将模型类别与模型名称对应列出。

**LR (A)。** LR仅建模原始特征的线性组合。

**FM [26] (B)。** FM使用分解技术建模二阶特征交互。

**AFM [40] (B)。** AFM是最先进的捕捉二阶特征交互的模型之一。它通过使用注意力机制来区分二阶组合特征的不同重要性，从而扩展了FM。

**DeepCrossing [32] (C)。** DeepCrossing利用带残差连接的深度全连接神经网络以隐式方式学习非线性特征交互。

**NFM [13] (C)。** NFM在二阶特征交互层之上堆叠深度神经网络。高阶特征交互通过神经网络的非线性被隐式捕捉。

**CrossNet [38] (C)。** Cross网络（Deep&Cross模型的核心）在比特级别对拼接后的特征向量进行外积运算，以显式建模特征交互。

**CIN [19] (C)。** 压缩交互网络（CIN，xDeepFM模型的核心）在向量级别对堆叠的特征矩阵进行外积运算。

**HOFM [5] (C)。** HOFM提出了基于核的高效算法来训练高阶分解机。遵循Blondel等人[5]和He与Chua[13]的设置，我们使用公开实现构建了三阶分解机。

我们将在与普通DNN联合训练的设置下（即第5.5节）与CrossNet和CIN的完整模型（即Deep&Cross和xDeepFM）进行比较。

#### 5.1.4 实现细节

所有方法均使用TensorFlow[1]实现。对于AutoInt和所有基线方法，我们根据经验将嵌入维度 d 设置为16，批量大小设置为1024。AutoInt有三个交互层，默认设置下隐藏单元数 d' 为32。每个交互层中，注意力头数为2⁸。为防止过拟合，我们使用网格搜索从{0.1-0.9}中选择MovieLens-1M数据集的dropout率[34]，并发现其他三个大数据集不需要dropout。对于基线方法，NFM在其双交互层之上使用一个大小为200的隐藏层，如其论文推荐。对于CN和CIN，我们跟随AutoInt使用三个交互层。DeepCrossing有四个前馈层，隐藏单元数为100，因为使用三个神经层时性能不佳。在确定所有网络结构后，我们同样对基线方法进行网格搜索以寻找最优超参数。最后，我们使用Adam[17]优化所有基于深度神经网络的模型。

⁸ 我们也尝试了不同的注意力头数量。使用一个头的性能不如两个头，进一步增加头数的改进不显著。

### 5.2 定量结果（RQ1）

**有效性评估。** 我们将10次不同运行的平均结果总结在表2中。我们有以下观察：（1）探索二阶特征交互的FM和AFM，在所有数据集上始终大幅优于LR，这表明个体特征在CTR预测中不足。（2）一个有趣的观察是，一些捕捉高阶特征交互的模型的性能反而较差。例如，尽管DeepCrossing和NFM使用深度神经网络作为学习高阶特征交互的核心组件，但它们并不能保证比FM和AFM有所提升。这可能归因于它们以隐式方式学习特征交互。相反，CIN显式地进行特征交互，并一致地优于低阶模型。（3）HOFM在Criteo和MovieLens-1M数据集上显著优于FM，这表明建模三阶特征交互有助于提升预测性能。（4）AutoInt在四个真实世界数据集中的三个上取得了所有基线方法中的最佳性能。在Avazu数据集上，CIN在AUC评估上略优于AutoInt，但我们获得了更低的Logloss。注意我们提出的AutoInt与DeepCrossing相比，除了特征交互层之外结构完全相同，这表明使用注意力机制学习显式组合特征是至关重要的。

**表1：评估数据集的统计信息。**

| 数据集 | #样本 | #字段 | #特征（稀疏） |
|---|---|---|---|
| Criteo | 45,840,617 | 39 | 998,960 |
| Avazu | 40,428,967 | 23 | 1,544,488 |
| KDD12 | 149,639,105 | 13 | 6,019,086 |
| MovieLens-1M | 739,012 | 7 | 3,529 |

**表2：不同算法的效果对比。** 我们强调提出的模型几乎在所有四个数据集和两个指标上均优于所有基线。进一步分析见第5.2节。

| 模型类别 | 模型 | Criteo AUC | Criteo Logloss | Avazu AUC | Avazu Logloss | KDD12 AUC | KDD12 Logloss | MovieLens-1M AUC | MovieLens-1M Logloss |
|---|---|---|---|---|---|---|---|---|---|
| 一阶 | LR | 0.7820 | 0.4695 | 0.7560 | 0.3964 | 0.7361 | 0.1684 | 0.7716 | 0.4424 |
| 二阶 | FM | 0.7836 | 0.4700 | 0.7706 | 0.3856 | 0.7759 | 0.1573 | 0.8252 | 0.3998 |
| 二阶 | AFM | 0.7938 | 0.4584 | 0.7718 | 0.3854 | 0.7659 | 0.1591 | 0.8227 | 0.4048 |
| 高阶 | DeepCrossing | 0.8009 | 0.4513 | 0.7643 | 0.3889 | 0.7715 | 0.1591 | 0.8448 | 0.3814 |
| 高阶 | NFM | 0.7957 | 0.4562 | 0.7708 | 0.3864 | 0.7515 | 0.1631 | 0.8357 | 0.3883 |
| 高阶 | CrossNet | 0.7907 | 0.4591 | 0.7667 | 0.3868 | 0.7773 | 0.1572 | 0.7968 | 0.4266 |
| 高阶 | CIN | 0.8009 | 0.4517 | 0.7758 | 0.3829 | 0.7799 | 0.1566 | 0.8286 | 0.4108 |
| 高阶 | HOFM | 0.8005 | 0.4508 | 0.7701 | 0.3854 | 0.7707 | 0.1586 | 0.8304 | 0.4013 |
| 高阶 | AutoInt (ours) | **0.8061** | **0.4455** | **0.7752** | **0.3824** | **0.7883** | **0.1546** | **0.8456** | **0.3797** |

AutoInt在Criteo、KDD12和MovieLens-1M数据集上优于最强基线的显著性水平：** p<0.01, * p<0.05（非配对t检验）。

**模型效率评估。** 我们在图4中展示了不同算法在四个数据集上的运行时间结果。不出所料，LR因其简单性是最高效的算法。FM和NFM在运行时间上表现相似，因为NFM仅在二阶交互层之上堆叠了一个单层前馈隐藏层。在所有列出的方法中，在所有基线中取得最佳预测性能的CIN最为耗时，因为其复杂的交叉层。这可能导致其在工业场景中不实用。注意AutoInt足够高效，与高效算法DeepCrossing和NFM相当。

**图4：不同算法在运行时间方面的效率对比。"DC"和"CN"分别是DeepCrossing和CrossNet的缩写。由于HOFM无法在单个GPU卡上拟合KDD12数据集，额外的通信成本使其最为耗时。进一步分析见第5.2节。**

我们还比较了不同模型的大小（即参数数量）作为效率评估的另一个标准。如表3所示，与基线中最好的模型CIN相比，AutoInt的参数数量小得多。

**表3：不同算法在Criteo数据集上模型大小的效率对比。"DC"和"CN"分别是DeepCrossing和CrossNet的缩写。统计的参数排除嵌入层。**

| 模型 | #参数 |
|---|---|
| DC | 1.6 × 10⁵ |
| CN | 3 × 10³ |
| CIN | 1.9 × 10⁶ |
| NFM | 4 × 10³ |
| AutoInt | 3.9 × 10⁴ |

总之，我们提出的AutoInt在所有对比模型中取得了最佳性能。与最具竞争力的基线模型CIN相比，AutoInt需要更少的参数，在线推理时也更高效。

### 5.3 分析（RQ2）

为进一步验证并深入了解所提出的模型，我们进行了消融研究，并比较了AutoInt的几种变体。

#### 5.3.1 残差结构的影响

标准AutoInt利用残差连接，它传递所有学习到的组合特征，从而允许建模极高阶的组合。为了验证残差单元的贡献，我们将其从标准模型中去除，而保持其他结构不变。如表4所示，我们观察到如果移除残差连接，所有数据集上的性能均下降。具体地，完整模型在KDD12和MovieLens-1M数据上大幅优于变体，这表明残差连接对于我们提出的方法中建模高阶特征交互至关重要。

**表4：消融研究——比较AutoInt有无残差连接的性能。AutoIntw/是完整模型，而AutoIntw/o是无残差连接的模型。**

| 数据集 | 模型 | AUC | Logloss |
|---|---|---|---|
| Criteo | AutoIntw/ | 0.8061 | 0.4454 |
| | AutoIntw/o | 0.8033 | 0.4478 |
| Avazu | AutoIntw/ | 0.7752 | 0.3823 |
| | AutoIntw/o | 0.7729 | 0.3836 |
| KDD12 | AutoIntw/ | 0.7888 | 0.1545 |
| | AutoIntw/o | 0.7831 | 0.1557 |
| MovieLens-1M | AutoIntw/ | 0.8460 | 0.3784 |
| | AutoIntw/o | 0.8299 | 0.3959 |

#### 5.3.2 网络深度的影响

我们的模型通过堆叠多个交互层（在第4节中介绍）来学习高阶特征组合。因此，我们感兴趣的是性能随交互层数量（即组合特征的阶数）如何变化。注意当没有交互层时（即层数为零），我们的模型将原始个体特征的加权和作为输入，即不考虑任何组合特征。

**图5：性能随交互层数量的变化。Criteo和Avazu数据集的结果类似，因此省略。**

结果总结在图5中。我们可以看到，如果使用一个交互层（即考虑了特征交互），两个数据集上的性能都大幅提升，这表明组合特征对预测非常有信息量。随着交互层数量进一步增加（即考虑了更高阶的组合特征），模型性能进一步提升。当层数达到三层时，性能趋于稳定，这表明添加极高阶的特征对预测没有更多信息量。

#### 5.3.3 不同维度的影响

接下来，我们研究关于参数 d（嵌入层的输出维度）的性能变化。在KDD12数据集上，随着维度增大（因为使用了更大模型进行预测），性能持续提升。在MovieLens-1M数据集上结果则不同。当维度达到24时，性能开始下降。原因是这个数据集很小，当使用过多参数时模型过拟合。

**图6：性能随嵌入维度的变化。Criteo和Avazu数据集的结果类似，因此省略。**

### 5.4 可解释推荐（RQ3）

一个好的推荐系统不仅能提供好的推荐，还能提供良好的可解释性。因此，在本部分中，我们展示AutoInt如何解释推荐结果。我们以MovieLens-1M数据集为例。

让我们看看我们算法推荐的一个结果，即一个用户喜欢一个物品。图7(a)展示了不同输入特征字段之间的相关性，这是通过注意力得分获得的。我们可以看到AutoInt能够识别有意义的组合特征<性别=男, 年龄=[18-24), 电影类型=动作&惊悚>（即红色虚线矩形）。这非常合理，因为年轻男性很可能喜欢动作和惊悚电影。

**图7：MovieLens-1M上案例级别和全局级别特征交互的注意力权重热力图。坐标轴代表特征字段<性别, 年龄, 职业, 邮编, 请求时间, 上映时间, 类型>。我们用矩形突出显示了一些学习到的组合特征。**

我们还对数据中不同特征字段之间的相关性感兴趣。因此，我们根据整个数据集中特征字段的平均注意力得分来衡量它们之间的相关性。不同字段之间的相关性总结在图7(b)中。我们可以看到<性别, 类型>、<年龄, 类型>、<请求时间, 上映时间>和<性别, 年龄, 类型>（即绿色实线区域）是强相关的，这些是该领域中可解释的推荐规则。

### 5.5 整合隐式交互（RQ4）

前馈神经网络能够建模隐式特征交互，并已被广泛整合到现有的CTR预测方法中[8, 11, 19]。为探究整合隐式特征交互是否能进一步提升性能，我们将AutoInt与一个两层前馈神经网络通过联合训练结合。我们将联合模型命名为AutoInt+，并与以下算法进行比较：

- **Wide&Deep [8]**。Wide&Deep整合了逻辑回归和前馈神经网络的输出。
- **DeepFM [11]**。DeepFM结合了传统的二阶分解机和前馈神经网络，并共享嵌入层。
- **Deep&Cross [38]**。Deep&Cross是CrossNet通过整合前馈神经网络的扩展。
- **xDeepFM [19]**。xDeepFM是通过整合前馈神经网络对CIN的扩展。

**表5：整合隐式特征交互的结果。我们标出各方法背后的基础模型。最后两列是与相应基础模型相比的AUC和Logloss平均变化（"+"表示提升，"−"表示下降）。**

| 模型 | Criteo AUC | Criteo Logloss | Avazu AUC | Avazu Logloss | KDD12 AUC | KDD12 Logloss | MovieLens-1M AUC | MovieLens-1M Logloss | Avg AUC变化 | Avg Logloss变化 |
|---|---|---|---|---|---|---|---|---|---|---|
| Wide&Deep (LR) | 0.8026 | 0.4494 | 0.7749 | 0.3824 | 0.7549 | 0.1619 | 0.8300 | 0.3976 | +0.0292 | -0.0213 |
| DeepFM (FM) | 0.8066 | 0.4449 | 0.7751 | 0.3829 | 0.7867 | 0.1549 | 0.8437 | 0.3846 | +0.0142 | -0.0113 |
| Deep&Cross (CN) | 0.8067 | 0.4447 | 0.7731 | 0.3836 | 0.7872 | 0.1549 | 0.8446 | 0.3809 | +0.0200 | -0.0164 |
| xDeepFM (CIN) | 0.8070 | 0.4447 | 0.7770 | 0.3823 | 0.7820 | 0.1560 | 0.8463 | 0.3808 | +0.0068 | -0.0096 |
| AutoInt+ (ours) | **0.8083** | **0.4434** | **0.7774** | **0.3811** | **0.7898** | **0.1543** | **0.8488** | **0.3753** | +0.0023 | -0.0020 |

AutoInt+在每项数据上优于最强基线的显著性水平：** p<0.01, * p<0.05（非配对t检验）。

表5展示了联合训练模型的平均结果（10次运行）。我们有以下观察：1）我们的方法在所有数据集上通过与前馈神经网络联合训练性能均有提升。这表明整合隐式特征交互确实提升了我们提出模型的预测能力。然而，从最后两列可以看出，与其他模型相比，我们性能提升的幅度相当小，这表明我们的单个模型AutoInt已经相当强大。2）在整合隐式特征交互后，AutoInt+优于所有竞争方法，并在所采用的CTR预测数据集上达到了新的最先进性能。

## 6 结论与未来工作

本文提出了一种基于自注意力机制的新型CTR预测模型，能够以显式方式自动学习高阶特征交互。我们方法的关键是新引入的交互层，该层允许每个特征与其他特征交互，并通过学习确定相关性。在四个真实世界数据集上的实验结果证明了我们提出模型的有效性和高效性。此外，我们通过可视化学习到的组合特征提供了良好的模型可解释性。当与通过前馈神经网络捕捉的隐式特征交互整合时，相比于先前的最先进方法，我们获得了更好的离线AUC和Logloss分数。

对于未来工作，我们感兴趣的是将上下文信息融入我们的方法，并提升其在在线推荐系统中的性能。此外，我们也计划将AutoInt扩展到通用的机器学习任务，如回归、分类和排序。

## 致谢

作者感谢所有匿名评审人富有洞察力的评论。我们感谢肖潇和董建波关于中国大学MOOC平台推荐机制的讨论。我们也感谢瞿萌对本文初稿的审阅。宋维平和张明受国家重点研发计划（编号：SQ2018AAA010010）、北京市科学技术委员会（编号：Z181100008918005）以及国家自然科学基金（编号：61772039和91646202）资助。宋维平还受中国留学基金委资助。唐建受加拿大自然科学与工程研究理事会以及加拿大CIFAR AI讲席计划资助。

## 参考文献

[1] Martín Abadi, Paul Barham, Jianmin Chen, Zhifeng Chen, Andy Davis, Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Geoffrey Irving, et al. 2016. TensorFlow: A System for Large-Scale Machine Learning. In OSDI, Vol. 16. 265–283.

[2] Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. 2015. Neural machine translation by jointly learning to align and translate. In International Conference on Learning Representations.

[3] Yoshua Bengio, Aaron Courville, and Pascal Vincent. 2013. Representation learning: A review and new perspectives. IEEE transactions on pattern analysis and machine intelligence 35, 8 (2013), 1798–1828.

[4] Alex Beutel, Paul Covington, Sagar Jain, Can Xu, Jia Li, Vince Gatto, and Ed H Chi. 2018. Latent Cross: Making Use of Context in Recurrent Recommender Systems. In Proceedings of the Eleventh ACM International Conference on Web Search and Data Mining. ACM, 46–54.

[5] Mathieu Blondel, Akinori Fujino, Naonori Ueda, and Masakazu Ishihata. 2016. Higher-order factorization machines. In Advances in Neural Information Processing Systems. 3351–3359.

[6] Mathieu Blondel, Masakazu Ishihata, Akinori Fujino, and Naonori Ueda. 2016. Polynomial Networks and Factorization Machines: New Insights and Efficient Training Algorithms. In International Conference on Machine Learning. 850–858.

[7] Chen Cheng, Fen Xia, Tong Zhang, Irwin King, and Michael R Lyu. 2014. Gradient boosting factorization machines. In Proceedings of the 8th ACM Conference on Recommender Systems. ACM, 265–272.

[8] Heng-Tze Cheng, Levent Koc, Jeremiah Harmsen, Tal Shaked, Tushar Chandra, Hrishi Aradhye, Glen Anderson, Greg Corrado, Wei Chai, Mustafa Ispir, et al. 2016. Wide & deep learning for recommender systems. In Proceedings of the 1st Workshop on Deep Learning for Recommender Systems. ACM, 7–10.

[9] Paul Covington, Jay Adams, and Emre Sargin. 2016. Deep neural networks for youtube recommendations. In Proceedings of the 10th ACM Conference on Recommender Systems. ACM, 191–198.

[10] Thore Graepel, Joaquin Quiñonero Candela, Thomas Borchert, and Ralf Herbrich. 2010. Web-scale Bayesian Click-through Rate Prediction for Sponsored Search Advertising in Microsoft's Bing Search Engine. In Proceedings of the 27th International Conference on Machine Learning. 13–20.

[11] Huifeng Guo, Ruiming Tang, Yunming Ye, Zhenguo Li, and Xiuqiang He. 2017. DeepFM: A Factorization-machine Based Neural Network for CTR Prediction. In Proceedings of the 26th International Joint Conference on Artificial Intelligence. AAAI Press, 1725–1731.

[12] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. 2016. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition. 770–778.

[13] Xiangnan He and Tat-Seng Chua. 2017. Neural factorization machines for sparse predictive analytics. In Proceedings of the 40th International ACM SIGIR conference on Research and Development in Information Retrieval. ACM, 355–364.

[14] Xiangnan He, Zhankui He, Jingkuan Song, Zhenguang Liu, Yu-Gang Jiang, and Tat-Seng Chua. 2018. NAIS: Neural attentive item similarity model for recommendation. IEEE Transactions on Knowledge and Data Engineering 30, 12 (2018), 2354–2366.

[15] Xinran He, Junfeng Pan, Ou Jin, Tianbing Xu, Bo Liu, Tao Xu, Yanxin Shi, Antoine Atallah, Ralf Herbrich, Stuart Bowers, et al. 2014. Practical lessons from predicting clicks on ads at facebook. In Proceedings of the Eighth International Workshop on Data Mining for Online Advertising. ACM, 1–9.

[16] Yuchin Juan, Yong Zhuang, Wei-Sheng Chin, and Chih-Jen Lin. 2016. Field-aware factorization machines for CTR prediction. In Proceedings of the 10th ACM Conference on Recommender Systems. ACM, 43–50.

[17] Diederick P Kingma and Jimmy Ba. 2015. Adam: A method for stochastic optimization. In International Conference on Learning Representations.

[18] Honglak Lee, Roger Grosse, Rajesh Ranganath, and Andrew Y Ng. 2011. Unsupervised learning of hierarchical representations with convolutional deep belief networks. Commun. ACM 54, 10 (2011), 95–103.

[19] Jianxun Lian, Xiaohuan Zhou, Fuzheng Zhang, Zhongxia Chen, Xing Xie, and Guangzhong Sun. 2018. xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. ACM, 1754–1763.

[20] Zhouhan Lin, Minwei Feng, Cicero Nogueira dos Santos, Mo Yu, Bing Xiang, Bowen Zhou, and Yoshua Bengio. 2017. A structured self-attentive sentence embedding. In International Conference on Learning Representations.

[21] H. Brendan McMahan, Gary Holt, D. Sculley, Michael Young, Dietmar Ebner, Julian Grady, Lan Nie, Todd Phillips, et al. 2013. Ad Click Prediction: A View from the Trenches. In Proceedings of the 19th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. ACM, 1222–1230.

[22] Alexander Miller, Adam Fisch, Jesse Dodge, Amir-Hossein Karimi, Antoine Bordes, and Jason Weston. 2016. Key-Value Memory Networks for Directly Reading Documents. In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing. ACL, 1400–1409.

[23] Alexander Novikov, Mikhail Trofimov, and Ivan Oseledets. 2016. Exponential machines. arXiv preprint arXiv:1605.03795 (2016).

[24] Richard J Oentaryo, Ee-Peng Lim, Jia-Wei Low, David Lo, and Michael Finegold. 2014. Predicting response in mobile advertising with hierarchical importance-aware factorization machine. In Proceedings of the 7th ACM international conference on Web search and data mining. ACM, 123–132.

[25] Yanru Qu, Han Cai, Kan Ren, Weinan Zhang, Yong Yu, Ying Wen, and Jun Wang. 2016. Product-based neural networks for user response prediction. In 2016 IEEE 16th International Conference on Data Mining (ICDM). IEEE, 1149–1154.

[26] Steffen Rendle. 2010. Factorization machines. In 2010 IEEE 10th International Conference on Data Mining (ICDM). IEEE, 995–1000.

[27] Steffen Rendle, Christoph Freudenthaler, and Lars Schmidt-Thieme. 2010. Factorizing personalized markov chains for next-basket recommendation. In Proceedings of the 19th international conference on World wide web. ACM, 811–820.

[28] Steffen Rendle, Zeno Gantner, Christoph Freudenthaler, and Lars Schmidt-Thieme. 2011. Fast context-aware recommendations with factorization machines. In Proceedings of the 34th international ACM SIGIR conference on Research and development in Information Retrieval. ACM, 635–644.

[29] Matthew Richardson, Ewa Dominowska, and Robert Ragno. 2007. Predicting clicks: estimating the click-through rate for new ads. In Proceedings of the 16th international conference on World Wide Web. ACM, 521–530.

[30] Alexander M. Rush, Sumit Chopra, and Jason Weston. 2015. A Neural Attention Model for Abstractive Sentence Summarization. In Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing. ACL, 379–389.

[31] Lili Shan, Lei Lin, Chengjie Sun, and Xiaolong Wang. 2016. Predicting ad click-through rates via feature-based fully coupled interaction tensor factorization. Electronic Commerce Research and Applications 16 (2016), 30–42.

[32] Ying Shan, T Ryan Hoens, Jian Jiao, Haijing Wang, Dong Yu, and JC Mao. 2016. Deep crossing: Web-scale modeling without manually crafted combinatorial features. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. ACM, 255–262.

[33] Weiping Song, Zhiping Xiao, Yifan Wang, Laurent Charlin, Ming Zhang, and Jian Tang. 2019. Session-based Social Recommendation via Dynamic Graph Attention Networks. In Proceedings of the Twelfth ACM International Conference on Web Search and Data Mining. ACM, 555–563.

[34] Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov. 2014. Dropout: A simple way to prevent neural networks from overfitting. The Journal of Machine Learning Research 15, 1 (2014), 1929–1958.

[35] Sainbayar Sukhbaatar, Jason Weston, Rob Fergus, et al. 2015. End-to-end memory networks. In Advances in neural information processing systems. 2440–2448.

[36] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information Processing Systems. 6000–6010.

[37] Petar Velickovic, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Lio, and Yoshua Bengio. 2018. Graph Attention Networks. In International Conference on Learning Representations.

[38] Ruoxi Wang, Bin Fu, Gang Fu, and Mingliang Wang. 2017. Deep & Cross Network for Ad Click Predictions. In Proceedings of the ADKDD'17. ACM, 12:1–12:7.

[39] Xiang Wang, Xiangnan He, Fuli Feng, Liqiang Nie, and Tat-Seng Chua. 2018. TEM: Tree-enhanced Embedding Model for Explainable Recommendation. In Proceedings of the 2018 World Wide Web Conference. WWW Steering Committee, 1543–1552.

[40] Jun Xiao, Hao Ye, Xiangnan He, Hanwang Zhang, Fei Wu, and Tat-Seng Chua. 2017. Attentional factorization machines: learning the weight of feature interactions via attention networks. In Proceedings of the 26th International Joint Conference on Artificial Intelligence. AAAI Press, 3119–3125.

[41] Weinan Zhang, Tianming Du, and Jun Wang. 2016. Deep learning over multi-field categorical data. In European conference on information retrieval. Springer, 45–57.

[42] Qian Zhao, Yue Shi, and Liangjie Hong. 2017. GB-CENT: Gradient Boosted Categorical Embedding and Numerical Trees. In Proceedings of the 26th International Conference on World Wide Web. WWW Steering Committee, 1311–1319.

[43] Guorui Zhou, Xiaoqiang Zhu, Chenru Song, Ying Fan, Han Zhu, Xiao Ma, Yanghui Yan, Junqi Jin, Han Li, and Kun Gai. 2018. Deep Interest Network for Click-Through Rate Prediction. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. ACM, 1059–1068.

[44] Jie Zhu, Ying Shan, JC Mao, Dong Yu, Holakou Rahmanian, and Yi Zhang. 2017. Deep embedding forest: Forest-based serving with deep embedding features. In Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. ACM, 1703–1711.
