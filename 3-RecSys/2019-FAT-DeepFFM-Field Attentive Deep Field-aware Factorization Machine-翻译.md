# FAT-DeepFFM：场注意力深度场感知分解机

> Junlin Zhang, Tongwen Huang, Zhiqi Zhang | Sina Weibo

本文介绍了 FAT-DeepFFM：场注意力深度场感知分解机。核心内容：


关键发现：

---


张俊林，黄桐文，张志强
新浪微博，北京，中国
{junlin6,tongwen,zhiqizhang}@staff.weibo.com


---

## 摘要

点击率（CTR）预估是个性化广告和推荐系统中的一项基础任务。近年来，基于深度学习模型和注意力机制在计算机视觉（CV）和自然语言处理（NLP）的各项任务中取得了成功。如何将注意力机制与深度CTR模型相结合是一个有前景的方向，因为它可能融合两者的优势。尽管已有一些CTR模型如注意力分解机（AFM）被提出用于建模二阶交互特征的权重，但我们认为在显式特征交互过程之前评估特征重要性对于CTR预测任务同样重要，因为当任务包含大量输入特征时，模型可以学习选择性地突出信息性特征并抑制不太有用的特征。在本文中，我们提出了一种名为场注意力深度场感知分解机（FAT-DeepFFM）的新型神经CTR模型，它将深度场感知分解机（DeepFFM）与我们提出的组合激励网络（CENet）场注意力机制相结合，CENet是压缩激励网络（SENet）的增强版本，用于突出特征重要性。我们在两个真实世界数据集上进行了大量实验，实验结果表明FAT-DeepFFM取得了最佳性能，并且相比最先进方法获得了不同程度的提升。我们还比较了两种注意力机制（显式特征交互前的注意力 vs. 显式特征交互后的注意力），并证明了前者显著优于后者。

## 1 引言

CTR预估是个性化广告和推荐系统中的一个基础任务。许多模型已被提出用于解决这一问题，例如逻辑回归（LR）[McMahan et al., 2013]、Poly2 [Juan et al., 2016]、基于树的模型[He et al., 2014]、基于张量的模型[Koren et al., 2009]、贝叶斯模型[Graepel et al., 2010]以及场感知分解机（FFM）[Juan et al., 2016]。深度学习技术已在计算机视觉[Krizhevsky et al., 2012; He et al., 2016]、语音识别[Graves et al., 2013]和自然语言理解[Mikolov et al., 2010; Cho et al., 2014]等多个研究领域展现出令人瞩目的成果。因此，在CTR预估中采用深度神经网络也已成为该领域的一个研究趋势[Zhang et al., 2016; Cheng et al., 2016; Xiao et al., 2017; Guo et al., 2017; Lian et al., 2018; Wang et al., 2017; Zhou et al., 2018; He and Chua, 2017]。一些基于深度学习的模型已被提出并取得成功，例如分解机支持的神经网络（FNN）[Zhang et al., 2016]、注意力分解机（AFM）[Xiao et al., 2017]、Wide & Deep [Cheng et al., 2016]、DeepFM [Guo et al., 2017]等。

另一方面，注意力机制可以过滤掉原始输入中无信息的特征，基于注意力的模型已被广泛使用并在各种任务中展现出令人期待的效果。如何将注意力机制与深度CTR模型相结合是一个有前景的方向，因为它可能融合两者的优势。尽管已有一些CTR模型如注意力分解机（AFM）[Xiao et al., 2017]被提出用于建模二阶交互特征的权重，但我们认为在显式特征交互之前评估特征重要性对于CTR预测任务也很重要，因为当任务包含大量输入特征时，模型可以学习选择性地突出信息性特征并抑制不太有用的特征。

在这项工作中，我们提出了一种名为场注意力深度场感知分解机（FAT-DeepFFM）的新型神经CTR模型，它将神经场感知分解机[Yang et al., 2017]与场注意力机制相结合。具体来说，我们在这项工作中所做的是将组合激励网络（CENet）——一种压缩激励网络（SENet）[Hu et al., 2017]的增强版本——类似注意力的机制引入DeepFFM模型中，以提高深度CTR网络的表示能力。我们的目标是在分解机特征交互过程之前，通过显式建模输入实例中所有不同特征之间的相互依赖关系，动态捕捉每个特征的重要性。我们的目标是使用CENet类注意力机制执行特征重新校准，通过该机制，模型可以学习选择性地突出信息性特征并有效抑制不太有用的特征。

我们的工作贡献总结如下：

1) 我们提出了一种名为FAT-DeepFFM的新颖模型，通过引入CENet场注意力，在显式特征交互过程之前动态捕捉每个特征的重要性，从而增强了DeepFFM模型。

2) 我们比较了两种不同的注意力机制（显式特征交互之前的特征注意力 vs. 显式特征交互之后的交叉特征注意力），实验结果表明前者显著优于后者。

3) 我们在两个真实世界数据集上进行了大量实验，实验结果表明FAT-DeepFFM取得了最佳性能，并且相比最先进方法获得了不同程度的提升。

本文其余部分组织如下：第2节介绍了与我们提出的模型相关的一些工作。第3节详细介绍了我们提出的场注意力深度场感知分解机（FAT-DeepFFM）模型。第4节展示并讨论了在Criteo和Avazu数据集上的实验结果。第5节总结了本文的工作。

## 2 相关工作

### 2.1 分解机与场感知分解机

分解机（FM）[Rendle, 2010]和场感知分解机（FFM）[Juan et al., 2016]是两个最成功的CTR模型。FM使用两个嵌入向量的点积来建模成对特征交互的效果。FFM扩展了分解机的思想，额外利用了场信息，并赢得了Criteo和Avazu举办的两个竞赛。当一个特征与来自不同场的其他特征交互时，FFM将为每个特征学习不同的嵌入向量。

### 2.2 基于深度学习的CTR模型

随着深度学习在计算机视觉和自然语言处理等许多研究领域的巨大成功，近年来也提出了许多基于深度学习的CTR模型。如何有效建模特征交互是大多数这类神经网络模型的关键因素。

分解机支持的神经网络（FNN）[Zhang et al., 2016]是一个使用FM预训练嵌入层的前馈神经网络。然而，FNN只能捕捉高阶特征交互。Wide & Deep Learning [Cheng et al., 2016]最初是为Google Play中的App推荐而引入的。Wide & Deep Learning联合训练宽线性模型和深度神经网络，以结合推荐系统中记忆和泛化的优势。然而，Wide & Deep模型的宽部分仍然需要专业特征工程，这意味着叉积变换也需要手动设计。为了减轻特征工程中的人工工作量，DeepFM [Guo et al., 2017]用FM替换了Wide & Deep模型的宽部分，并在FM和深度组件之间共享特征嵌入。DeepFM被认为是CTR预估领域中最先进的模型之一。

深度交叉网络（DCN）[Wang et al., 2017]以显式方式高效地捕获有界度的特征交互。类似地，极深分解机（xDeepFM）[Lian et al., 2018]通过提出一种新颖的压缩交互网络（CIN）部分，以显式方式建模低阶和高阶特征交互。

我们的方法基于神经FFM，该模型首次由Yang [Yang et al., 2017]在腾讯社交广告竞赛中提出。它可以被视为将DeepFM的FM部分替换为FFM，我们将在第3节中详细描述该模型。

### 2.3 注意力CTR模型

注意力机制的灵感来源于人类视觉注意力，它可以通过减少噪声数据的副作用来过滤掉原始输入中无信息的特征。基于注意力的模型已被广泛使用，并在语音识别和机器翻译等任务上展现出令人期待的效果。注意力机制也被引入到一些CTR模型中。例如，注意力分解机（AFM）[Xiao et al., 2017]通过神经注意力网络从数据中区分和学习不同特征交互的重要性，从而改进了FM。DIN [Zhou et al., 2018]用兴趣分布表示用户的多样化兴趣，并设计了一个类似注意力的网络结构，根据候选广告局部激活相关兴趣。

## 3 场注意力深度FFM

### 3.1 DeepFFM

我们的工作最初旨在将FFM模型引入神经CTR系统。然而，Yang等[Yang et al., 2017]在2017年腾讯社交广告竞赛中报告了一项与我们的工作类似的努力。作者报告说，在他们的CTR预测系统中使用神经FFM后取得了显著的收益。神经FFM在该竞赛中相当成功：第三名获奖方案就基于这个单一模型，而集成版本赢得了竞赛的第一名。由于很难找到关于该模型的详细技术描述，我们将首先介绍神经FFM，本文中将其称为DeepFFM模型。

众所周知，FM [Rendle, 2010]将特征i和j之间的交互建模为它们对应嵌入向量的点积，如下所示：

$$
ŷ(x) = w₀ + \Sigmaᵢ wᵢxᵢ + \Sigmaᵢ \Sigmaⱼ$_{+}$₁ ⟨vᵢ, vⱼ⟩ xᵢxⱼ    (1)
$$

FM为每个特征学习一个嵌入向量$v_i$ $\in$ ℝᵏ，其中k是一个超参数（通常是一个小整数），m是特征数量。然而，FM忽略了一个事实，即一个特征在与来自其他场的特征交互时可能表现不同。为了显式考虑这种差异，场感知分解机（FFM）为每个特征学习额外的n-1个嵌入向量（这里n表示场的数量）：

$$
ŷ(x) = w₀ + \Sigmaᵢ wᵢxᵢ + \Sigmaᵢ \Sigmaⱼ$_{+}$₁ ⟨vᵢⱼ, vⱼᵢ⟩ xᵢxⱼ    (2)
$$

其中$v_ij$ $\in$ ℝᵏ表示特征i在与场j交互时，特征i的第j个条目的嵌入向量。k是嵌入大小。

如图1所示，DeepFFM旨在通过神经网络体现FFM的思想。一个输入实例首先通过独热编码转换为高维稀疏特征，以表示原始特征输入。接下来的嵌入矩阵层与稀疏输入层全连接，将原始特征压缩为低维、稠密的实值矩阵。具体来说，对于特征i，使用一个大小为k$×$n的对应二维嵌入矩阵E$M_i$ = [$v_i1$, $v_i2$, ..., $v_ij$, ..., $v_in$]来衡量其与其他特征交互的影响，其中$v_ij$ $\in$ ℝᵏ表示场i的第j个嵌入向量，n是场的数量，k是嵌入向量的大小。因此，很明显嵌入矩阵层EM是一个大小为k$×$n$×$n的三维矩阵，因为我们有n个场，每个场对应一个二维嵌入矩阵。

接下来的特征交互层尝试在嵌入矩阵EM上捕捉任意一对来自不同场的特征之间的双向特征交互。将特征交互层记为向量A，我们有两种不同类型的特征交互方法：内积版本和哈达玛积版本。我们可以将该层中的两种方法形式化如下：

$$
A = [v₁₂ \oplus v₂₁, ..., vᵢⱼ \oplus vⱼᵢ, ..., v₍ₙ$_{-}$₁₎ₙ \oplus vₙ₍ₙ$_{-}$₁₎]    内积
A = [v₁₂ \otimes v₂₁, ..., vᵢⱼ × vⱼᵢ, ..., v₍ₙ$_{-}$₁₎ₙ \otimes vₙ₍ₙ$_{-}$₁₎]    哈达玛积
$$

其中n是场的数量，$v_ij$ $\oplus$ $v_ji$表示两个嵌入向量的内积为一个标量⟨$v_ij$, $v_ji$⟩，而$v_ij$ $×$ $v_ji$表示两个嵌入向量的哈达玛积，结果为一个向量：

$$
vᵢⱼ \oplus vⱼᵢ = [v¹ᵢⱼ·v¹ⱼᵢ, v²ᵢⱼ·v²ⱼᵢ, ..., vᵏᵢⱼ·vᵏⱼᵢ]
$$

其中k是嵌入向量$v_ji$的大小。注意需要j > i以避免重复计算。从这里可以看出，特征交互层A是一个宽的拼接向量，如果采用内积版本，该向量的大小为n(n-1)/2；如果采用哈达玛积版本，大小为kn(n-1)/2。

多个隐藏层是在特征交互层上的前馈神经网络，用于隐式学习高阶特征交互。将特征交互层的输出记为向量A，我们可以将其输入前馈神经网络的隐藏层。前向过程为：

$$
x¹ = \sigma(W¹A + b¹)    (3)
xˡ = \sigma(Wˡxˡ⁻¹ + bˡ)    (4)
$$

其中l是层深度，\sigma是激活函数，xˡ是第l个隐藏层的输出。

加上线性部分，DeepFFM的输出单元如下：

$$
ŷ(X) = \sigma(W_linear x_linear + Wˡ⁺¹xˡ + bˡ⁺¹)    (5)
$$

其中\sigma是sigmoid函数，x_linear是原始特征，xˡ是多个隐藏层的输出，W_linear、Wˡ⁺¹和bˡ⁺¹是可学习参数。

### 3.2 嵌入矩阵层上的CENet场注意力

Hu提出了"压缩激励网络"（SENet）[Hu et al., 2017]，通过显式建模各种图像分类任务中卷积特征通道之间的相互依赖关系，来提高网络的表示能力。SENet被证明在图像分类任务中非常成功，并在ILSVRC 2017分类任务中获得了第一名。

我们的工作受到SENet在计算机视觉领域成功的启发。为了提高深度CTR网络的表示能力，我们将组合激励网络（CENet）注意力机制——SENet的增强版本——引入到DeepFFM模型的嵌入矩阵层。我们的目标是在FM的特征交互过程之前，通过显式建模所有不同特征之间的相互依赖关系，动态捕捉每个特征的重要性。我们的目标是使用CENet注意力机制执行特征重新校准，通过该机制，模型可以学习选择性地突出信息性特征并抑制不太有用的特征。

从图2可以看出，CENet类场注意力机制包括两个阶段：组合阶段和激励阶段。第一阶段通过将一个嵌入向量的所有信息组合成一个简单的特征描述符，来计算每个场中每个嵌入向量的"汇总统计量"；第二阶段对这些特征描述符应用注意力变换，然后使用计算出的注意力值重新缩放原始嵌入矩阵。

**组合阶段**：令E$M_i$ = [$v_i1$, $v_i2$, ..., $v_ij$, ..., $v_in$]表示场i的二维k$×$n嵌入矩阵，其中$v_ij$ $\in$ ℝᵏ指的是场i的第j个嵌入向量，n是场的数量，k是嵌入向量的大小。在此阶段，我们将嵌入向量$v_ij$组合成单个数字，以表示该特征的汇总信息。这可以通过使用1$×$1卷积[Szegedy et al., 2016; Chollet, 2017]生成特征级统计量来实现，而不是使用SENet中常用的全局最大池化或求和操作等压缩操作。1$×$1卷积，也称为逐点卷积，负责通过计算一个输入特征嵌入的线性组合来构建新特征。在SENet中，我们通过收缩每个嵌入向量为场i生成一个统计向量z $\in$ ℝⁿ，其中z的第f个元素$z_i$ $\in$ ℝ可以通过以下方式计算：

$$
zᵢ = F_sq(vᵢ) = max_{1\leqt\leqk} vᵢᵗ    全局最大池化    (6)
$$

这里k表示每个嵌入向量的嵌入大小。CV领域最常用的压缩操作是全局最大池化，它可以捕捉相应通道中最强的特征。我们在此阶段改变了方法，使用1$×$1卷积，因为我们假设CTR任务中特征嵌入向量的每个位置都包含信息。因此，1$×$1卷积可以引入参数来学习特征嵌入中每个位置的组合权重。1$×$1卷积计算如下：

$$
zᵢⱼ = conv1d(Uᵢⱼ, vᵢⱼ) = ReLU(Uᵢⱼ vᵢⱼ)    (7)
$$

其中$U_ij$是卷积权重，卷积核大小为1$×$1，滤波器数量为1，激活函数设置为'ReLU'。

**激励阶段**：在第一阶段之后，场i的嵌入矩阵E$M_i$ = [$v_i1$, $v_i2$, ..., $v_ij$, ..., $v_in$]已被转换为描述符向量D$V_i$ = [$z_i1$, $z_i2$, ..., $z_ij$, ..., $z_in$]。我们有n个不同的场，因此我们通过拼接每个描述符向量来汇总所有描述符，如下所示：

$$
D = concat(DV₁, DV₂, ..., DVₙ)    (8)
$$

其中向量D的大小为n²。

为了从描述符向量计算注意力，使用了两个全连接（FC）层。第一层是降维层，参数为$W_1$，降维比为r（这是一个超参数），并使用ReLU作为非线性函数。第二层使用参数$W_2$增加维度，该维度等于描述符向量D的维度，也使用ReLU作为非线性激活函数。形式上，场注意力计算如下：

$$
S = F_ex(D, W) = \delta(W₂ \delta(W₁ D))    (9)
$$

其中\delta指ReLU函数，$W_1$ $\in$ ℝ^(n²/r $×$ n²)，$W_2$ $\in$ ℝ^(n² $×$ n²/r)，注意力向量S的大小为n²。

ReLU函数的激活值被用作最终的场注意力值，而不进行softmax归一化操作，因为我们希望鼓励多个特征同时重要，而不是仅有少数几个特征重要。然后，场i的原始嵌入矩阵E$M_i$中的值根据相应计算的场注意力向量$S_i$进行重新缩放，如下所示：

$$
AEMᵢ = F_scale(Sᵢ, EMᵢ) = [Sᵢ₁·vᵢ₁, ..., Sᵢⱼ·vᵢⱼ, ..., Sᵢₙ·vᵢₙ]    (10)
$$

其中F_scale($S_i$, E$M_i$)指的是嵌入向量$v_ij$与标量$S_ij$之间的逐向量乘法。较大的注意力值$S_ij$表示模型动态识别出了重要特征，该注意力值用于增强原始嵌入向量$v_ij$。相反，较小的注意力值$S_ij$会通过减少相应嵌入向量$v_ij$中的值来抑制无信息的特征甚至噪声。

经过组合阶段和激励阶段后，我们得到一个新的三维嵌入矩阵AEM，大小为k$×$n$×$n，与原始嵌入矩阵EM的大小相同。我们在本文中称这个新嵌入矩阵为注意力嵌入层。

### 3.3 结合场注意力与DeepFFM

如第3.2节所述，CENet注意力机制可以执行特征重新校准，通过该机制，模型可以学习选择性地突出信息性特征并抑制不太有用的特征。我们可以通过将CENet注意力模块插入到第3.1节描述的DeepFFM模型中来增强它。图3提供了我们提出的场注意力深度场感知分解机（FAT-DeepFFM）的整体架构。它在神经结构上与DeepFFM相似，而原始嵌入矩阵层被类似SE-Net的场注意力模块所替代。我们将这个新插入的模块称为注意力嵌入矩阵层。FAT-DeepFFM的其他组件与DeepFFM模型相同。与DeepFFM类似，根据特征交互类型，FAT-DeepFFM也有两个版本：内积版本和哈达玛积版本。

从以上描述可以看出，我们提出的注意力机制是一种在交叉特征产生之前的注意力。因此一个自然的研究问题出现了：如果在显式特征交互过程之后对交叉特征引入注意力（就像AFM所做的那样），哪一种会表现更好？为了回答这个问题，我们还进行了一些实验来比较两种注意力机制的性能差异。实验结果表明，特征交互前的注意力始终优于特征交互后的注意力。我们将在第4.3节详细讨论这些实验。

## 4 实验结果

为了全面评估我们提出的方法，我们设计了一些实验来回答以下研究问题：

- RQ1：我们提出的FAT-DeepFFM能否超越最先进的基于深度学习的CTR模型？
- RQ2：哪种注意力机制（显式特征交互之前的特征注意力 vs. 显式特征交互之后的交叉特征注意力）在真实世界CTR数据集上表现更好？
- RQ3：哪种特征交互方法（内积 vs. 哈达玛积）在基于神经网络的CTR模型中更有效？

### 4.1 实验设置

**数据集**

我们在实验中使用以下两个数据集：

1. **Criteo数据集**。作为一个非常著名的公开真实世界展示广告数据集，包含每次广告展示信息和相应的用户点击反馈，Criteo数据集被广泛用于许多CTR模型评估。Criteo数据集中有26个匿名类别字段和13个连续特征字段。我们按90%:10%的比例随机将数据分为训练集和测试集。

2. **Avazu数据集**。Avazu数据集包含按时间顺序排列的多天广告点击数据。对于每次点击数据，有24个字段，表示单次展示的元素。我们按80%:20%的比例随机将数据分为训练集和测试集。

表1列出了评估数据集的统计信息。

**表1：评估数据集统计信息**

| 数据集 | 实例数 | 字段数 | 特征数 |
|--------|--------|--------|--------|
| Criteo | 45M    | 39     | 2.3M   |
| Avazu  | 40.43M | 24     | 0.64M  |

对于这两个数据集，预测准确率的小幅提升被认为具有实际意义，因为如果公司拥有非常庞大的用户群，这将带来公司收入的大幅增长。

**评估指标**

我们在实验中使用AUC（ROC曲线下面积）和Logloss（交叉熵）作为评估指标。这两个指标在二分类任务中非常流行。AUC对分类阈值和正例比例不敏感，AUC的上限为1，值越大表示性能越好。Logloss衡量两个分布之间的距离，Logloss值越小表示性能越好。

**对比模型**

我们将以下CTR预估模型的性能作为基线进行比较：LR、FM、FFM、FNN、DeepFM、AFM、Deep & Cross Network（DCN）、xDeepFM和DeepFFM，所有这些模型都在第2节和第3节中讨论过。

**实现细节**

我们在实验中用Tensorflow实现了所有模型。对于优化方法，我们使用Adam，小批量大小为1000，学习率设置为0.0001。由于本文关注神经网络结构，我们将所有模型的场嵌入维度固定为10。对于带有DNN部分的模型，隐藏层深度设置为3，每层神经元数量：FFM相关模型为1600，其他深度模型为400，所有激活函数为ReLU，dropout率设置为0.5。对于CENet组件，激活函数为ReLU，所有相关实验中的降维比设置为1。我们使用2块Tesla K40 GPU进行实验。

### 4.2 性能比较（RQ1）

不同模型在Criteo数据集和Avazu数据集上的CTR预测总体性能如表2所示。我们有以下关键发现：

1. **FAT-DeepFFM总体上取得了最佳性能**，并且相比最先进方法获得了不同程度的提升。作为最佳模型，FAT-DeepFFM在Criteo和Avazu数据集上，Logloss方面分别优于FM 3.64%和1.80%（AUC方面分别为2.28%和1.50%），Logloss方面分别优于LR 5.64%和3.29%（AUC方面分别为3.79%和2.99%）。

2. **FAT-DeepFFM在两个数据集上始终优于DeepFFM**。这表明CENet场注意力机制对于学习原始特征的重要性非常有帮助。

**表2：不同模型在Criteo和Avazu上的总体性能（模型名称后缀"I"表示内积版本，后缀"H"表示哈达玛积版本）**

| 模型 | Criteo AUC | Criteo Logloss | Avazu AUC | Avazu Logloss |
|------|-----------|---------------|-----------|---------------|
| LR | 0.7808 | 0.4681 | 0.7633 | 0.3891 |
| FM | 0.7923 | 0.4584 | 0.7745 | 0.3832 |
| FFM | 0.8001 | 0.4525 | 0.7795 | 0.3810 |
| FNN | 0.8057 | 0.4464 | 0.7802 | 0.3800 |
| AFM | 0.7965 | 0.4541 | 0.7740 | 0.3839 |
| DeepFM | 0.8085 | 0.4445 | 0.7786 | 0.3810 |
| DCN | 0.7977 | 0.4617 | 0.7680 | 0.3940 |
| xDeepFM | 0.8091 | 0.4461 | 0.7808 | 0.3819 |
| DeepFFM-I | 0.8087 | 0.4434 | 0.7839 | 0.3783 |
| DeepFFM-H | 0.8088 | 0.4434 | 0.7835 | 0.3782 |
| FAT-DeepFFM-I | 0.8099 | 0.4421 | 0.7857 | 0.3763 |
| FAT-DeepFFM-H | 0.8104 | 0.4416 | 0.7863 | 0.3761 |

### 4.3 注意力机制比较（RQ2）

在本小节中，我们将讨论两种不同注意力机制的性能：一种是显式特征交互之前的注意力（如上述场注意力）；另一种是显式特征交互过程之后的交叉特征注意力（如AFM所做的）。

针对交叉特征注意力的具体实现方法，我们实现了两种方法：与第3.2节描述的类似的CENet注意力，以及类似AFM的基于MLP的注意力，超参数经过调优以达到最佳性能。实验结果如表3所示，前缀为"MLP"的实验指的是基于MLP的交叉特征注意力，前缀为"CE"的实验指的是在交叉特征上使用CENet注意力。

表3列出了两种注意力机制在Criteo数据集和Avazu数据集上的总体性能。我们有以下关键发现：

1. **无论使用哪种方法（CENet或类似AFM的基于MLP的模型）作为交叉特征上的特定注意力方法，特征交互之前的特征注意力始终优于显式特征交互之后的交叉特征注意力**，有时甚至大幅领先。我们推测这可能是因为特征上的注意力相比交叉特征上的注意力，能更有效地突出重要信息，同时抑制不重要的特征和噪声。

2. **在某些条件下，交叉特征上的注意力对某些真实世界CTR预测任务是有害的**。从表3所示的内积组实验结果可以看出，MLP-DeepFFM-I和CE-DeepFFM-I模型的表现都低于原始DeepFFM模型。这一结果表明，如果使用内积函数作为特征交互方法，交叉特征上的注意力对CTR预测任务是有害的。其背后的原因仍需进一步研究。

**表3：两种注意力机制在Criteo和Avazu上的总体性能（模型名称后缀"I"表示内积版本，后缀"H"表示哈达玛积版本）**

| 模型 | Criteo AUC | Criteo Logloss | Avazu AUC | Avazu Logloss |
|------|-----------|---------------|-----------|---------------|
| DeepFFM-I | 0.8087 | 0.4434 | 0.7839 | 0.3783 |
| MLP-DeepFFM-I | 0.8022 | 0.4499 | 0.7819 | 0.3796 |
| CE-DeepFFM-I | 0.808 | 0.444 | 0.7816 | 0.381 |
| FAT-DeepFFM-I | 0.8099 | 0.4422 | 0.7857 | 0.3763 |
| DeepFFM-H | 0.8088 | 0.4434 | 0.7835 | 0.3782 |
| MLP-DeepFFM-H | 0.8083 | 0.444 | 0.7847 | 0.3778 |
| CE-DeepFFM-H | 0.8092 | 0.443 | 0.7822 | 0.3786 |
| FAT-DeepFFM-H | 0.8104 | 0.4417 | 0.7861 | 0.3773 |

### 4.4 特征交互方法（RQ3）

如第3节所述，DeepFFM和FAT-DeepFFM模型都有两种特征交互方法：内积版本和哈达玛积版本。因此一个自然的研究问题出现了：哪种方法表现更好？表3还显示了两组数据集上的4组可比较实验（具有相同前缀和不同后缀的模型名称构成一组，例如DeepFFM-I和DeepFFM-H）。我们有以下关键发现：

1. 如果不对DeepFFM模型采用任何注意力，无论使用哪种特征交互方法，都没有观察到明显的性能差异（DeepFFM-I vs. DeepFFM-H）。

2. 如果对DeepFFM模型采用注意力，无论是对特征的注意力还是对交叉特征的注意力，**应优先选择哈达玛积函数而非内积函数**。从表3可以看出，这一结论在大多数情况下成立。

## 5 结论

在本文中，我们提出了一种名为场注意力深度场感知分解机（FAT-DeepFFM）的新型神经CTR模型，它将深度场感知分解机（DeepFFM）与CENet场注意力机制相结合。我们在两个真实世界数据集上进行了大量实验，实验结果表明FAT-DeepFFM取得了最佳性能，并且相比最先进方法获得了不同程度的提升。我们还展示了FAT-DeepFFM在两个数据集上始终优于DeepFFM，这表明当任务具有大量输入特征时，CENet场注意力机制对于学习原始特征的重要性非常有帮助。我们还比较了两种不同类型的注意力机制（显式特征交互之前的注意力 vs. 显式特征交互之后的注意力），实验结果表明前者显著优于后者。

## 参考文献

[Cheng et al., 2016] Heng-Tze Cheng, Levent Koc, Jeremiah Harmsen, Tal Shaked, Tushar Chandra, Hrishi Aradhye, Glen Anderson, Greg Corrado, Wei Chai, Mustafa Ispir, et al. Wide & deep learning for recommender systems. In *Proceedings of the 1st Workshop on Deep Learning for Recommender Systems*, pages 7–10. ACM, 2016.

[Cho et al., 2014] Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. Learning phrase representations using rnn encoder-decoder for statistical machine translation, 2014.

[Chollet, 2017] François Chollet. Xception: Deep learning with depthwise separable convolutions. *arXiv preprint*, pages 1610–02357, 2017.

[Graepel et al., 2010] Thore Graepel, Joaquin Quiñonero Candela, Thomas Borchert, and Ralf Herbrich. Web-scale bayesian click-through rate prediction for sponsored search advertising in microsoft's bing search engine. Omnipress, 2010.

[Graves et al., 2013] Alex Graves, Abdel-rahman Mohamed, and Geoffrey Hinton. Speech recognition with deep recurrent neural networks. In *Acoustics, speech and signal processing (icassp), 2013 ieee international conference on*, pages 6645–6649. IEEE, 2013.

[Guo et al., 2017] Huifeng Guo, Ruiming Tang, Yunming Ye, Zhenguo Li, and Xiuqiang He. Deepfm: a factorization-machine based neural network for ctr prediction. *arXiv preprint arXiv:1703.04247*, 2017.

[He and Chua, 2017] Xiangnan He and Tat-Seng Chua. Neural factorization machines for sparse predictive analytics. In *Proceedings of the 40th International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR'17*, pages 355–364, New York, NY, USA, 2017. ACM.

[He et al., 2014] Xinran He, Junfeng Pan, Ou Jin, Tianbing Xu, Bo Liu, Tao Xu, Yanxin Shi, Antoine Atallah, Ralf Herbrich, Stuart Bowers, et al. Practical lessons from predicting clicks on ads at facebook. In *Proceedings of the Eighth International Workshop on Data Mining for Online Advertising*, pages 1–9. ACM, 2014.

[He et al., 2016] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pages 770–778, 2016.

[Hu et al., 2017] Jie Hu, Li Shen, and Gang Sun. Squeeze-and-excitation networks. *arXiv preprint arXiv:1709.01507*, 7, 2017.

[Juan et al., 2016] Yuchin Juan, Yong Zhuang, Wei-Sheng Chin, and Chih-Jen Lin. Field-aware factorization machines for ctr prediction. In *Proceedings of the 10th ACM Conference on Recommender Systems*, pages 43–50. ACM, 2016.

[Koren et al., 2009] Yehuda Koren, Robert Bell, and Chris Volinsky. Matrix factorization techniques for recommender systems. *Computer*, (8):30–37, 2009.

[Krizhevsky et al., 2012] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. In *Advances in neural information processing systems*, pages 1097–1105, 2012.

[Lian et al., 2018] Jianxun Lian, Xiaohuan Zhou, Fuzheng Zhang, Zhongxia Chen, Xing Xie, and Guangzhong Sun. xdeepfm: Combining explicit and implicit feature interactions for recommender systems. *arXiv preprint arXiv:1803.05170*, 2018.

[McMahan et al., 2013] H Brendan McMahan, Gary Holt, David Sculley, Michael Young, Dietmar Ebner, Julian Grady, Lan Nie, Todd Phillips, Eugene Davydov, Daniel Golovin, et al. Ad click prediction: a view from the trenches. In *Proceedings of the 19th ACM SIGKDD international conference on Knowledge discovery and data mining*, pages 1222–1230. ACM, 2013.

[Mikolov et al., 2010] Tomáš Mikolov, Martin Karafiát, Lukáš Burget, Jan Černocký, and Sanjeev Khudanpur. Recurrent neural network based language model. In *Eleventh Annual Conference of the International Speech Communication Association*, 2010.

[Rendle, 2010] Steffen Rendle. Factorization machines. In *Data Mining (ICDM), 2010 IEEE 10th International Conference on*, pages 995–1000. IEEE, 2010.

[Szegedy et al., 2016] Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jon Shlens, and Zbigniew Wojna. Rethinking the inception architecture for computer vision. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pages 2818–2826, 2016.

[Wang et al., 2017] Ruoxi Wang, Bin Fu, Gang Fu, and Mingliang Wang. Deep & cross network for ad click predictions. In *Proceedings of the ADKDD'17*, page 12. ACM, 2017.

[Xiao et al., 2017] Jun Xiao, Hao Ye, Xiangnan He, Hanwang Zhang, Fei Wu, and Tat-Seng Chua. Attentional factorization machines: Learning the weight of feature interactions via attention networks. *arXiv preprint arXiv:1708.04617*, 2017.

[Yang et al., 2017] Yi Yang, Shaofeng Shen, and Yu liang. Neural field-aware factorization machine. https://cs.nju.edu.cn/31/60/c1654a209248/page.htm, 2017.

[Zhang et al., 2016] Weinan Zhang, Tianming Du, and Jun Wang. Deep learning over multi-field categorical data. In *European conference on information retrieval*, pages 45–57. Springer, 2016.

[Zhou et al., 2018] Guorui Zhou, Xiaoqiang Zhu, Chenru Song, Ying Fan, Han Zhu, Xiao Ma, Yanghui Yan, Junqi Jin, Han Li, and Kun Gai. Deep interest network for click-through rate prediction. In *Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*, pages 1059–1068. ACM, 2018.
