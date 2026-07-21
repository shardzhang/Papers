# MaskNet：通过实例引导掩码向CTR排序模型引入特征级乘法

> Zhiqiang Zhang, Qingyun She, Junlin Zhang | Sina Weibo

本文介绍了 MaskNet：通过实例引导掩码向CTR排序模型引入特征级乘法。核心内容：


关键发现：

---


王志强，佘青云，张俊林
新浪微博
北京，中国
roky2813@sina.com, qingyun_she@163.com, junlin6@staff.weibo.com


---

## 摘要

点击率（CTR）估计已成为许多实际应用中最基本的任务之一，对于排序模型而言，有效捕捉复杂的高阶特征至关重要。浅层前馈网络被广泛用于许多最先进的DNN模型（如FNN、DeepFM和xDeepFM）中以隐式捕捉高阶特征交互。然而，一些研究已经证明，加性特征交互，特别是前馈神经网络，在捕捉常见特征交互方面效率低下。为了解决这个问题，我们通过提出实例引导掩码，将特定的乘法操作引入DNN排序系统中，该掩码在特征嵌入层和前馈层上执行逐元素乘积，并由输入实例引导。本文还通过提出MaskBlock，将DNN模型中的前馈层转变为加性和乘性特征交互的混合体。MaskBlock结合了层归一化、实例引导掩码和前馈层，是一个基本构建块，可用于在不同配置下设计新的排序模型。由MaskBlock组成的模型在本文中称为MaskNet，我们提出了两种新的MaskNet模型，以展示MaskBlock作为构建高性能排序系统的基本构建块的有效性。在三个真实世界数据集上的实验结果表明，我们提出的MaskNet模型显著优于DeepFM和xDeepFM等最先进模型，这意味着MaskBlock是构建新型高性能排序系统的有效基础构建单元。

**ACM引用格式：**
Zhiqiang Wang, Qingyun She, Junlin Zhang. 2021. MaskNet: Introducing Feature-Wise Multiplication to CTR Ranking Models by Instance-Guided Mask. In Proceedings of DLP-KDD 2021. ACM, New York, NY, USA, 9 pages.
https://doi.org/10.1145/nnnnnnn.nnnnnnn

## 1 引言

点击率（CTR）预测是预测用户点击推荐item的概率。它在个性化广告和推荐系统中扮演着重要角色。许多模型已被提出来解决这个问题，如逻辑回归（LR）[16]、Poly2[17]、基于树的模型[7]、基于张量的模型[12]、贝叶斯模型[5]和场感知分解机（FFM）[11]。近年来，将DNN用于CTR估计也成为该领域的研究趋势，一些基于深度学习的模型已被引入，如分解机支持神经网络（FNN）[24]、注意力分解机（AFM）[3]、Wide & Deep（W&D）[22]、DeepFM[6]、xDeepFM[13]等。

特征交互对于CTR任务至关重要，排序模型有效捕捉这些复杂特征非常重要。大多数DNN排序模型（如FNN、W&D、DeepFM和xDeepFM）使用浅层MLP层以隐式方式建模高阶交互，这是当前最先进排序系统中的一个重要组成部分。

然而，Alex Beutel等人[2]已经证明，加性特征交互，特别是前馈神经网络，在捕捉常见特征交叉方面效率低下。他们提出了一种简单而有效的方法，称为"潜在交叉"（latent cross），这是一种在RNN模型中上下文嵌入和神经网络隐藏状态之间的乘法交互。最近，Rendle等人的工作[18]也表明，在协同过滤中，精心配置的点积基线在很大程度上优于MLP层。虽然MLP理论上可以逼近任何函数，但他们表明，用MLP学习点积并非易事，要在相当大小的嵌入维度上高精度地学习点积，需要较大的模型容量以及大量的训练数据。他们的工作也证明了MLP层在建模复杂特征交互方面的低效性。

受"潜在交叉"[2]和Rendle的工作[18]启发，我们关注以下问题：我们能否通过将特定的乘法操作引入DNN排序系统，使其有效地捕捉复杂的特征交互？

为了克服前馈层在捕捉复杂特征交叉方面的低效问题，本文向DNN排序系统中引入了一种特殊的乘法操作。首先，我们提出了一种实例引导掩码（instance-guided mask），在特征嵌入层和前馈层上执行逐元素乘积。实例引导掩码利用从输入实例收集的全局信息，以统一的方式动态突出显示特征嵌入和隐藏层中的信息元素。采用实例引导掩码有两个主要优点：第一，掩码与隐藏层或特征嵌入层之间的逐元素乘积以统一的方式将乘法操作引入DNN排序系统，从而更有效地捕捉复杂的特征交互。第二，这是一种由输入实例引导的细粒度逐位注意力机制，既可以减弱特征嵌入和MLP层中噪声的影响，同时突出DNN排序系统中的信息信号。

通过结合实例引导掩码、后续的前馈层和层归一化，我们提出了MaskBlock，将常用的前馈层转变为加性和乘性特征交互的混合体。实例引导掩码引入了乘法交互，后续的前馈隐藏层聚合被掩码的信息，以更好地捕捉重要的特征交互。而层归一化可以简化网络的优化。

MaskBlock可以被视为一个基本构建块，用于在某些配置下设计新的排序模型。由MaskBlock组成的模型在本文中称为MaskNet，我们提出了两种新的MaskNet模型，以展示MaskBlock作为构建高性能排序系统的基本构建块的有效性。

我们的工作贡献总结如下：

（1）在这项工作中，我们提出了一种实例引导掩码，在DNN模型的特征嵌入层和前馈层上执行逐元素乘积。实例引导掩码中包含的全局上下文信息被动态地融入到特征嵌入层和前馈层中，以突出重要的元素。

（2）我们提出了一个名为MaskBlock的基本构建块，它包含三个关键组件：实例引导掩码、后续的前馈隐藏层层和层归一化模块。通过这种方式，我们将标准DNN模型中广泛使用的前馈层转变为加性和乘性特征交互的混合体。

（3）我们还提出了一个名为MaskNet的新排序框架，利用MaskBlock作为基本构建单元来组建新的排序系统。更具体地说，本文基于MaskBlock设计了串行MaskNet模型和并行MaskNet模型。串行排序模型逐块堆叠MaskBlock，而并行排序模型在共享特征嵌入层上并行放置多个MaskBlock。

（4）我们在三个真实世界数据集上进行了大量实验，实验结果表明，我们提出的两种MaskNet模型显著优于最先进的模型。结果暗示，MaskBlock通过实例引导掩码将乘法操作引入DNN模型，确实增强了DNN模型捕捉复杂特征交互的能力。

本文的其余部分组织如下。第2节介绍了与我们提出的模型相关的一些相关工作。我们在第3节详细介绍我们提出的模型。第4节展示并讨论了在三个真实世界数据集上的实验结果。第5节总结了本文的工作。

## 2 相关工作

### 2.1 最先进的CTR模型

近年来已经提出了许多基于深度学习的CTR模型，对于这些基于神经网络的模型来说，有效建模特征交互是关键因素。

分解机支持神经网络（FNN）[24]是一种使用FM预训练嵌入层的前馈神经网络。Wide & Deep Learning[22]联合训练宽线性模型和深度神经网络，以结合推荐系统中记忆和泛化的优势。然而，Wide & Deep模型的宽部分输入仍然需要专家特征工程。为了减轻特征工程中的手动工作，DeepFM[6]用FM替换了Wide & Deep模型的宽部分，并在FM和深度组件之间共享特征嵌入。

虽然大多数DNN排序模型通过MLP层以隐式方式处理高阶特征交互，但一些工作通过子网络显式地引入高阶特征交互。Deep & Cross Network（DCN）[21]以显式方式有效地捕捉有限度数的特征交互。类似地，eXtreme Deep Factorization Machine（xDeepFM）[13]通过提出一种新颖的压缩交互网络（CIN）部分，以显式方式建模低阶和高阶特征交互。AutoInt[19]使用多头自注意力神经网络在低维空间中显式建模特征交互。FiBiNET[9]可以通过SENET机制动态学习特征重要性，并通过双线性函数学习特征交互。

### 2.2 特征级掩码或门控

特征级掩码或门控已在视觉[8, 20]、自然语言处理[4]和推荐系统[14, 15]中得到广泛探索。例如，Highway Networks[20]利用特征门控来缓解非常深网络的基于梯度的训练。Squeeze-and-Excitation Networks[8]通过将每个通道与学习到的sigmoid掩码值显式相乘来重新校准特征响应。Dauphin等人[4]提出了门控线性单元（GLU），利用它来控制语言建模任务中预测下一个单词时应传播哪些信息。门控或掩码机制也被应用于推荐系统中。Ma等人[15]提出了一种新颖的多任务学习方法——多门混合专家（MMoE），它从数据中显式学习建模任务关系。Ma等人[14]提出了层次门控网络（HGN），以捕捉用户的长期和短期兴趣。HGN中的特征门控和实例门控模块分别从特征和实例层面选择哪些item特征可以传递到下游层。

### 2.3 归一化

归一化技术已被公认为深度学习中非常有效的组件。已经提出了许多归一化方法，其中最流行的两种是BatchNorm[10]和LayerNorm[1]。批归一化（Batch Norm或BN）[10]通过在一个小批（mini-batch）内计算的均值和方差来归一化特征。另一个例子是层归一化（LayerNorm或LN）[1]，它被提出用于简化循环神经网络的优化。层归一化的统计量不是在小批中的N个样本上计算的，而是以逐层方式为每个样本独立估计的。归一化方法在加速深度网络训练方面已被证明是成功的。

## 3 我们提出的模型

在本节中，我们首先描述特征嵌入层。然后介绍我们提出的实例引导掩码、MaskBlock和MaskNet结构的细节。最后介绍作为损失函数的对数损失。

### 3.1 嵌入层

CTR任务的输入数据通常由稀疏特征和稠密特征组成，稀疏特征大多是类别类型。这些特征被编码为独热向量，这通常会导致大数据量词汇的特征空间维度过高。解决这个问题的常见方案是引入嵌入层。通常，稀疏输入可以表示为：

$$
x = [x_1, x_2, ..., x_f]      (1)
$$

其中 `f` 表示字段数，`x_i $\in$ R^n` 表示具有 `n` 个特征的类别字段的独热向量，而 `x_i $\in$ R^n` 是数值字段的只有一个值的向量。我们可以通过下式获得独热向量 `x_i` 的特征嵌入 `e_i`：

$$
e_i = W_e x_i                 (2)
$$

其中 `W_e $\in$ R^{k$×$n}` 是 `n` 个特征的嵌入矩阵，`k` 是字段嵌入的维度。数值特征 `x_j` 也可以通过下式转换到相同的低维空间：

$$
e_j = V_j x_j                 (3)
$$

其中 `V_j $\in$ R^k` 是对应于该字段的嵌入，其嵌入大小为 `k`。因此，嵌入层的结果是一个宽拼接向量：

$$
V_emb = concat(e_1, e_2, ..., e_i, ..., e_f)    (4)
$$

其中 `f` 表示字段数，`e_i $\in$ R^k` 表示一个字段的嵌入。虽然输入实例的特征长度可能不同，但它们的嵌入具有相同的长度 `f $×$ k`，其中 `k` 是字段嵌入的维度。

我们使用实例引导掩码将乘法操作引入DNN排序系统，本文后续部分中所谓的"实例"指的是当前输入实例的特征嵌入层。

### 3.2 实例引导掩码

我们利用通过实例引导掩码从输入实例收集的全局信息，动态地突出显示特征嵌入和前馈层中的信息元素。对于特征嵌入，掩码强调具有更多信息的关键元素，以有效表示该特征。对于隐藏层中的神经元，掩码通过考虑输入实例中的上下文信息帮助重要的特征交互突显出来。除了这个优势外，实例引导掩码还将乘法操作引入DNN排序系统，以更有效地捕捉复杂的特征交叉。

如图1所示，实例引导掩码中使用两个具有恒等函数的全连接（FC）层。注意，实例引导掩码的输入始终来自输入实例，即特征嵌入层。

第一个FC层称为"聚合层"（aggregation layer），与第二个FC层相比，它是一个相对较宽的层，以便更好地收集输入实例中的全局上下文信息。聚合层具有参数 `$W_{d1}$`，其中 `d` 表示第 `d` 个掩码。对于特征嵌入和不同的MLP层，我们采用不同的实例引导掩码，拥有各自的参数，以从输入实例中为每一层学习捕捉各种信息。

第二个FC层称为"投影层"（projection layer），它将维度降低到与特征嵌入层 `V_emb` 或隐藏层 `V_hidden` 相同的大小，参数为 `$W_{d2}$`。形式上：

$$
V_mask = W_{d2}(ReLU(W_{d1} V_emb + \beta_{d1})) + \beta_{d2}   (5)
$$

其中 `$W_{d1}$ $\in$ R^{t$×$m}` 和 `$W_{d2}$ $\in$ R^{z$×$t}` 是实例引导掩码的参数，`t` 和 `z` 分别表示聚合层和投影层的神经元数量，`m = f $×$ k` 指的是输入实例的嵌入大小，`f` 表示字段数，`k` 是字段嵌入的维度。`\beta_{d1} $\in$ R^{t$×$m}` 和 `\beta_{d2} $\in$ R^{z$×$t}` 是两个FC层的学习偏置。注意，聚合层通常比投影层宽，因为投影层的大小需要等于特征嵌入层或MLP层的大小。因此我们定义大小 `r = t/z` 作为缩减比（reduction ratio），这是一个用于控制两层神经元数量比例的超参数。

本文使用逐元素乘积将实例引导掩码聚合的全局上下文信息融入到特征嵌入或隐藏层中，如下所示：

$$
V_maskedEMB = V_mask \odot V_emb                     (6)
V_maskedHID = V_mask \odot V_hidden                  (7)
$$

其中 `V_emb` 表示嵌入层，`V_hidden` 表示DNN模型中的前馈层，`$\odot$` 表示两个向量之间的逐元素乘积，如下所示：

$$
V_i \odot V_j = [V_i1 · V_j1, V_i2 · V_j2, ..., V_iu · V_ju]   (7)
$$

这里 `u` 是向量 `V_i` 和 `V_j` 的大小。

实例引导掩码可以被视为一种特殊的逐位注意力或门控机制，它利用输入实例中包含的全局上下文信息来指导训练过程中的参数优化。`V_mask` 中较大的值意味着模型动态地识别出特征嵌入或隐藏层中的重要元素，用于增强向量 `V_emb` 或 `V_hidden` 中的对应元素。相反，`V_mask` 中较小的值会通过减小对应向量 `V_emb` 或 `V_hidden` 中的值来抑制非信息元素甚至噪声。

采用实例引导掩码的两个主要优势是：首先，掩码与隐藏层或特征嵌入层之间的逐元素乘积以统一的方式将乘法操作引入DNN排序系统，以更有效地捕捉复杂的特征交互。其次，这种由输入实例引导的细粒度逐位注意力，既可以减弱特征嵌入和MLP层中噪声的影响，同时突出DNN排序系统中的信息信号。

### 3.3 MaskBlock

为了克服DNN模型中前馈层捕捉复杂特征交互的低效问题，本文提出了一个名为MaskBlock的基本构建块，用于DNN排序系统，如图2和图3所示。提出的MaskBlock包含三个关键组件：层归一化模块、实例引导掩码和一个前馈隐藏层。层归一化可以简化网络的优化。实例引导掩码为标准DNN模型的前馈层引入了乘法交互，前馈隐藏层聚合被掩码的信息以更好地捕捉重要的特征交互。通过这种方式，我们将标准DNN模型中广泛使用的前馈层转变为加性和乘性特征交互的混合体。

首先，我们简要回顾一下LayerNorm的公式。

**层归一化：**
通常，归一化旨在确保信号在网络中传播时具有零均值和单位方差，以减少"协变量偏移"[10]。例如，层归一化（LayerNorm或LN）[1]被提出用于简化循环神经网络的优化。具体来说，令 `x = (x_1, x_2, ..., x_H)` 表示大小为 `H` 的输入向量到归一化层。LayerNorm对输入 `x` 进行重新中心和重新缩放：

$$
h = g \odot N(x) + b,   N(x) = (x - \mu) / \delta

\mu = (1/H) \Sigma x_i,   \delta = sqrt((1/H) \Sigma (x_i - \mu)^2)   (8)
$$

其中 `h` 是LayerNorm层的输出，`$\odot$` 是逐元素乘操作，`\mu` 和 `\delta` 是输入的均值和标准差，偏置 `b` 和增益 `g` 是具有相同维度 `H` 的参数。

作为MaskBlock中的关键组件之一，层归一化可以应用于特征嵌入层和前馈层。对于特征嵌入层，我们将每个特征的嵌入视为一层来计算LN的均值、标准差、偏置和增益，如下所示：

$$
LN_EMB(V_emb) = concat(LN(e_1), LN(e_2), ..., LN(e_i), ..., LN(e_f))   (9)
$$

对于DNN模型中的前馈层，LN的统计量是在相应隐藏层中包含的神经元之间估计的，如下所示：

$$
LN_HID(V_hidden) = ReLU(LN(W_i X))   (10)
$$

其中 `X $\in$ R^t` 指前馈层的输入，`W_i $\in$ R^{m$×$t}` 是该层的参数，`t` 和 `m` 分别表示输入层的大小和前馈层的神经元数量。注意，我们在MLP上有两个位置放置归一化操作：一个是在非线性操作之前，另一个是在非线性操作之后。我们发现非线性之前的归一化性能始终优于非线性之后的归一化操作。因此，本文中MLP部分使用的所有归一化都放在非线性操作之前，如公式（4）所示。

**特征嵌入上的MaskBlock：**

我们通过组合三个关键元素来提出MaskBlock：层归一化、实例引导掩码和后续的前馈层。MaskBlock可以堆叠形成更深的网络。根据每个MaskBlock的不同输入，我们有两种MaskBlock：特征嵌入上的MaskBlock和MaskBlock上的MaskBlock。我们将首先在本小节中介绍特征嵌入上的MaskBlock，如图2所示。

特征嵌入 `V_emb` 是特征嵌入上MaskBlock的唯一输入。在对嵌入 `V_emb` 进行层归一化操作 `LN` 之后，MaskBlock利用实例引导掩码通过逐元素乘积突出 `V_emb` 中的信息元素。形式上：

$$
V_maskedEMB = V_mask \odot LN_EMB(V_emb)   (11)
$$

其中 `$\odot$` 表示实例引导掩码与归一化向量 `LN_EMB(V_emb)` 之间的逐元素乘积，`V_maskedEMB` 表示被掩码后的特征嵌入。注意，实例引导掩码 `V_mask` 的输入也是特征嵌入 `V_emb`。

我们在MaskBlock中引入一个前馈隐藏层和后续的归一化操作，以通过归一化的非线性变换更好地聚合被掩码的信息。MaskBlock的输出可以如下计算：

$$
V_output = LN_HID(W_i V_maskedEMB)
        = ReLU(LN(W_i (V_mask \odot LN_EMB(V_emb))))   (12)
$$

其中 `W_i $\in$ R^{q$×$n}` 是第 `i` 个MaskBlock中前馈层的参数，`n` 表示 `V_maskedEMB` 的大小，`q` 表示前馈层的神经元数量。

实例引导掩码将逐元素乘积引入特征嵌入，作为一种细粒度注意力机制，而特征嵌入和隐藏层上的归一化简化了网络优化。MaskBlock中的这些关键组件帮助前馈层更有效地捕捉复杂的特征交叉。

**MaskBlock上的MaskBlock：**

在本小节中，我们将介绍如图3所示的MaskBlock上的MaskBlock。这种MaskBlock有两个不同的输入：特征嵌入 `V_emb` 和上一个MaskBlock的输出 `V^p_output`。这种MaskBlock的实例引导掩码的输入始终是特征嵌入 `V_emb`。MaskBlock利用实例引导掩码通过逐元素乘积突出上一个MaskBlock输出 `V^p_output` 中的重要特征交互。形式上：

$$
V_maskedHID = V_mask \odot V^p_output   (13)
$$

其中 `$\odot$` 表示实例引导掩码 `V_mask` 与上一个MaskBlock输出 `V^p_output` 之间的逐元素乘积，`V_maskedHID` 表示被掩码后的隐藏层。

为了更好地捕捉重要的特征交互，我们在MaskBlock中引入了一个前馈隐藏层和后续的层归一化操作。通过这种方式，我们将标准DNN模型中广泛使用的前馈层转变为加性和乘性特征交互的混合体，以避免那些加性特征交叉模型的低效性。MaskBlock的输出可以如下计算：

$$
V_output = LN_HID(W_i V_maskedHID)
        = ReLU(LN(W_i (V_mask \odot V^p_output)))   (14)
$$

其中 `W_i $\in$ R^{q$×$n}` 是第 `i` 个MaskBlock中前馈层的参数，`n` 表示 `V_maskedHID` 的大小，`q` 表示前馈层的神经元数量。

### 3.4 MaskNet

基于MaskBlock，可以根据不同的配置设计各种新的排序模型。由MaskBlock组成的排序模型在本文中称为MaskNet。我们还提出了两种利用MaskBlock作为基本构建块的MaskNet模型。

**串行MaskNet（SerialMaskNet）：**

我们可以将一个MaskBlock堆叠在另一个之后来构建排序系统，如图4左侧模型所示。第一个块是特征嵌入上的MaskBlock，所有其他块是MaskBlock上的MaskBlock，以形成更深的网络。预测层放在最终MaskBlock的输出向量上。我们将这种串行配置下的MaskNet称为SerMaskNet。每个MaskBlock中实例引导掩码的所有输入都来自特征嵌入层 `V_emb`，这使得串行MaskNet模型看起来像一个在每个时间步共享输入的RNN模型。

**并行MaskNet（ParallelMaskNet）：**

我们提出了另一种MaskNet，通过在共享特征嵌入层上并行放置多个特征嵌入上的MaskBlock，如图4右侧模型所示。每个块的输入都是相同的特征嵌入 `V_emb`。并行MaskBlock可以被视为不同的专家，每个专家关注不同的特征方面，类似于MMoE[15]的做法。每个MaskBlock关注特定类型的重要特征或特征交互。我们通过拼接每个MaskBlock的输出来收集每个专家的信息，如下所示：

$$
V_merge = concat(V^1_output, V^2_output, ..., V^i_output, ..., V^u_output)   (15)
$$

其中 `V^i_output $\in$ R^q` 是第 `i` 个MaskBlock的输出，`q` 表示MaskBlock中前馈层的神经元数量，`u` 是MaskBlock的数量。

为了进一步融合每个专家捕捉到的特征交互，在拼接信息 `V_merge` 上堆叠多个前馈层。令 `H_0 = V_merge` 表示拼接层的输出，然后将 `H_0` 输入深度神经网络，前馈过程为：

$$
H_l = ReLU(W_l H_{l-1} + \beta_l)   (16)
$$

其中 `l` 是深度，ReLU是激活函数，`W_l`、`\beta_l` 和 `H_l` 分别是第 `l` 层的模型权重、偏置和输出。预测层放在多个前馈网络的最后一层上。我们将此版本的MaskNet称为本文后续部分中的"ParaMaskNet"。

### 3.5 预测层

总结一下，我们给出我们提出模型输出的总体公式：

$$
ŷ = \delta(w_0 + \Sigma w_i x_i)   (17)
$$

其中 `ŷ $\in$ (0, 1)` 是CTR的预测值，`\delta` 是sigmoid函数，`n` 是最后一个MaskBlock输出（SerMaskNet）或前馈层（ParaMaskNet）的大小，`x_i` 是前馈层的位值，`w_i` 是每个位值的学习权重。

对于二分类，损失函数是对数损失：

$$
L = -(1/N) \Sigma [y_i log(ŷ_i) + (1 - y_i) log(1 - ŷ_i)]   (18)
$$

其中 `N` 是训练实例的总数，`y_i` 是第 `i` 个实例的真实标签，`ŷ_i` 是预测的CTR。优化过程是最小化以下目标函数：

$$
𝔏 = L + \lambda‖Θ‖   (19)
$$

其中 `\lambda` 表示正则化项，`Θ` 表示参数集，包括特征嵌入矩阵、实例引导掩码矩阵、MaskBlock中的前馈层和预测部分的参数。

## 4 实验结果

在本节中，我们在三个真实世界数据集上评估提出的方法，并进行详细的消融研究，以回答以下研究问题：

- **RQ1** 基于MaskBlock的MaskNet模型是否比现有的基于深度学习的最先进CTR模型表现更好？
- **RQ2** MaskBlock架构中各个组件的影响是什么？每个组件对于构建有效的排序系统是否都是必要的？
- **RQ3** 网络的超参数如何影响我们提出的两种MaskNet模型的性能？
- **RQ4** 实例引导掩码是否会根据输入实例突出特征嵌入和前馈层中的重要元素？

在下文中，我们将首先描述实验设置，然后回答上述研究问题。

### 4.1 实验设置

#### 4.1.1 数据集

我们在实验中使用以下三个数据集：

（1）**Criteo数据集：** 作为一个非常著名的公开真实世界展示广告数据集，每条记录包含广告展示信息和相应的用户点击反馈，Criteo数据集被广泛用于许多CTR模型评估。Criteo数据集中有26个匿名类别字段和13个连续特征字段。

（2）**Malware数据集：** Malware是来自Kaggle竞赛的数据集，发布在Microsoft Malware Prediction比赛中。该比赛的目标是预测Windows机器感染恶意软件的概率。恶意软件预测任务可以像典型的CTR估计任务一样被表述为二分类问题。

（3）**Avazu数据集：** Avazu数据集包含按时间顺序排列的几天广告点击数据。对于每条点击数据，有23个字段指示单个广告展示的元素。

我们按8:1:1随机分割实例用于训练、验证和测试，表1列出了评估数据集的统计数据。

**表1：评估数据集的统计数据**

| 数据集 | 实例数 | 字段数 | 特征数 |
|--------|--------|--------|--------|
| Criteo | 45M | 39 | 30M |
| Avazu | 40.43M | 23 | 9.5M |
| Malware | 8.92M | 82 | 0.97M |

#### 4.1.2 评估指标

AUC（ROC曲线下面积）被用作我们实验中的评估指标。AUC的上限为1，值越大表示性能越好。

与工作[23]类似，RelaImp也被用作另一个评估指标，用于衡量相对于相应基线模型的相对AUC改进。由于随机策略的AUC为0.5，我们可以去除AUC分数的常数部分，并将RelaImp公式化为：

$$
RelaImp = AUC(MeasuredModel) - 0.5 / AUC(BaseModel) - 0.5 - 1   (20)
$$

#### 4.1.3 比较模型

我们将以下CTR估计模型与我们提出的方法进行比较：FM、DNN、DeepFM、Deep & Cross Network（DCN）、xDeepFM和AutoInt模型，所有这些模型都在第2节中讨论过。FM被视为评估中的基础模型。

#### 4.1.4 实现细节

我们在实验中用Tensorflow实现所有模型。对于优化方法，我们使用Adam，小批大小为1024，学习率设置为0.0001。专注于本文中的神经网络结构，我们将所有模型的字段嵌入维度设置为固定值10。对于具有DNN部分的模型，隐藏层深度设置为3，每层的神经元数量为400，所有激活函数为ReLU。对于MaskBlock中的默认设置，实例引导掩码的缩减比设置为2。我们使用2块Tesla K40 GPU进行实验。

### 4.2 性能比较（RQ1）

不同模型在三个评估数据集上的总体性能如表2所示。从实验结果可以看出：

（1）串行模型和并行模型在所有三个数据集上都取得了更好的性能，并且相对于最先进的方法获得了显著改进。相对于基线FM，准确率提升从3.12%到11.40%；相对于基线DeepFM，提升从1.55%到5.23%；相对于xDeepFM基线，提升从1.27%到4.46%。我们还进行了显著性检验，以验证我们提出的模型以显著性水平\alpha=0.01优于基线。

尽管MaskNet模型缺乏像xDeepFM中的CIN这样的模块来显式捕捉高阶特征交互，但由于MaskBlock的存在，它仍然取得了更好的性能。实验结果表明，MaskBlock通过实例引导掩码在归一化的特征嵌入和前馈层上引入乘法操作，确实增强了DNN模型捕捉复杂特征交互的能力。

（2）就串行模型和并行模型的比较而言，实验结果在三个评估数据集上显示出可比较的性能。这明确证明了MaskBlock是构建各种高性能排序系统的有效基础构建单元。

**表2：不同模型在三个数据集上的总体性能（AUC）（特征嵌入大小=10，我们提出的两个模型都具有3个MaskBlock，使用相同的默认设置）**

| 模型 | Criteo AUC | Criteo RelaImp | Malware AUC | Malware RelaImp | Avazu AUC | Avazu RelaImp |
|------|-----------|---------------|------------|----------------|-----------|---------------|
| FM | 0.7895 | 0.00% | 0.7166 | 0.00% | 0.7785 | 0.00% |
| DNN | 0.8054 | +5.35% | 0.7246 | +3.70% | 0.7820 | +1.26% |
| DeepFM | 0.8057 | +5.46% | 0.7293 | +5.86% | 0.7833 | +1.72% |
| DCN | 0.8058 | +5.49% | 0.7300 | +6.19% | 0.7830 | +1.62% |
| xDeepFM | 0.8064 | +5.70% | 0.7310 | +6.65% | 0.7841 | +2.01% |
| AutoInt | 0.8051 | +5.39% | 0.7282 | +5.36% | 0.7824 | +1.40% |
| SerMaskNet | 0.8119 | +7.74% | 0.7413 | +11.40% | 0.7877 | +3.30% |
| ParaMaskNet | 0.8124 | +7.91% | 0.7410 | +11.27% | 0.7872 | +3.12% |

### 4.3 MaskBlock的消融研究（RQ2）

为了更好地理解每个组件在MaskBlock中的影响，我们对MaskBlock的关键组件进行了消融实验，每次只移除其中一个来观察性能变化，包括掩码模块、层归一化（LN）和前馈网络（FFN）。表3显示了我们的两个完整版MaskNet模型及其仅移除一个组件的变体的结果。

从表3的结果可以看出，移除实例引导掩码或层归一化都会降低模型性能，这意味着实例引导掩码和层归一化都是MaskBlock有效性的必要组件。至于MaskBlock中的前馈层，它对串行模型或并行模型的影响显示出差异。如果移除MaskBlock中的前馈层，串行模型的性能会显著下降，而并行模型似乎没有受到影响。我们认为这意味着MaskBlock中的前馈层对于融合实例引导掩码后的特征交互信息很重要。对于并行模型，并行MaskBlock之上的多个前馈层具有与MaskBlock中前馈层类似的功能，这可能导致移除该组件时两个模型之间的性能差异。

**表3：在Criteo数据集上移除MaskBlock中不同组件的MaskNet模型的总体性能（AUC）（特征嵌入大小=10，MaskNet模型有3个MaskBlock）**

| 模型名称 | SerMaskNet | ParaMaskNet |
|----------|-----------|------------|
| 完整模型 | 0.8119 | 0.8124 |
| -w/o Mask | 0.8090 | 0.8093 |
| -w/o LN | 0.8106 | 0.8103 |
| -w/o FFN | 0.8085 | 0.8122 |

### 4.4 超参数研究（RQ3）

在本文的以下部分，我们研究超参数对两种MaskNet模型的影响，包括：1）特征嵌入大小；2）MaskBlock数量；以及3）实例引导掩码模块中的缩减比。实验在Criteo数据集上进行，每次改变一个超参数，同时保持其他设置不变。超参数实验在其他两个数据集中显示出类似的趋势。

**特征嵌入数量。** 表4中的结果显示特征嵌入大小对模型性能的影响。可以观察到，开始时两种模型的性能都随着嵌入大小的增加而提高。然而，当SerMaskNet模型的嵌入大小大于50且ParaMaskNet模型的嵌入大小大于30时，模型性能下降。实验结果告诉我们，模型受益于更大的特征嵌入大小。

**表4：不同特征嵌入大小的MaskNet模型在Criteo数据集上的总体性能（AUC）（MaskBlock数量为3）**

| 嵌入大小 | 10 | 20 | 30 | 50 | 80 |
|---------|----|----|----|----|----|
| SerMaskNet | 0.8119 | 0.8123 | 0.8121 | 0.8125 | 0.8121 |
| ParaMaskNet | 0.8124 | 0.8128 | 0.8131 | 0.8129 | 0.8129 |

**MaskBlock数量。** 为了理解MaskBlock数量对模型性能的影响，我们进行了实验，对两种MaskNet模型从1到9个块堆叠MaskBlock。实验结果列于表5。对于SerMaskNet模型，性能一开始随着更多块而增加，直到数量大于5。而当我们不断向ParaMaskNet模型添加更多MaskBlock时，性能缓慢增加。这可能表明更多的专家提升了ParaMaskNet模型的性能，尽管它更耗时。

**表5：不同MaskBlock数量的MaskNet模型在Criteo数据集上的总体性能（AUC）（嵌入大小=10）**

| 块数量 | 1 | 3 | 5 | 7 | 9 |
|--------|----|----|----|----|----|
| SerMaskNet | 0.8110 | 0.8119 | 0.8126 | 0.8117 | 0.8115 |
| ParaMaskNet | 0.8113 | 0.8124 | 0.8127 | 0.8128 | 0.8132 |

**实例引导掩码中的缩减比。** 为了探索实例引导掩码中缩减比的影响，我们通过改变聚合层的大小来进行一些实验，将缩减比从1调整到5。实验结果如表6所示，我们可以观察到不同的缩减比对模型性能影响很小。这表明我们可以在实际应用中使用较小的缩减比以节省计算资源。

**表6：MaskBlock中掩码模块不同隐藏层大小的MaskNet模型在Criteo数据集上的总体性能（AUC）（嵌入大小=10，MaskBlock数量为3）**

| 缩减比 | 1 | 2 | 3 | 4 | 5 |
|--------|----|----|----|----|----|
| SerMaskNet | 0.8118 | 0.8119 | 0.8120 | 0.8117 | 0.8119 |
| ParaMaskNet | 0.8124 | 0.8124 | 0.8122 | 0.8122 | 0.8124 |

### 4.5 实例引导掩码研究（RQ4）

如第3.2节所讨论的，实例引导掩码可以被视为一种特殊的逐位注意力机制，根据当前输入实例突出重要信息。我们可以利用实例引导掩码来增强特征嵌入和前馈层中的信息元素，并抑制非信息元素甚至噪声。

为了验证这一点，我们设计了以下实验：训练完3个块的SerMaskNet后，我们将不同的实例输入模型，并观察相应实例引导掩码的输出。

首先，我们从Criteo数据集中随机抽取100000个不同实例，并观察不同块的实例引导掩码产生的值的分布。图5显示了结果。我们可以看到掩码值的分布遵循正态分布。超过50%的掩码值是小数值，接近零，只有一小部分掩码值是相对较大的数。这意味着特征嵌入和前馈层中的大部分信号是非信息的甚至是噪声，被小的掩码值抑制了。然而，也有一些信息通过较大的掩码值被实例引导掩码增强了。

其次，我们随机抽取两个实例，并比较实例引导掩码产生的值的差异。结果显示在图6中。我们可以看到：对于特征嵌入的掩码值，不同的输入实例导致掩码关注不同的区域。实例A的掩码输出更关注前几个特征，而实例B的掩码值聚焦于其他特征的某些位上。我们可以在前馈层的掩码值中观察到类似的趋势。这表明输入实例确实引导掩码关注特征嵌入和前馈层的不同部分。

## 5 结论

在本文中，我们通过提出实例引导掩码将乘法操作引入DNN排序系统，该掩码在特征嵌入层和前馈层上执行逐元素乘积。我们还通过结合层归一化、实例引导掩码和前馈层，提出MaskBlock，将DNN模型中的前馈层转变为加性和乘性特征交互的混合体。MaskBlock是一个基本构建块，可用于设计新的排序模型。我们还基于MaskBlock提出了两种具体的MaskNet模型。在三个真实世界数据集上的实验结果表明，我们提出的模型显著优于DeepFM和xDeepFM等最先进模型。

## 参考文献

[1] Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. 2016. Layer normalization. arXiv preprint arXiv:1607.06450 (2016).

[2] Alex Beutel, Paul Covington, Sagar Jain, Can Xu, Jia Li, Vince Gatto, and Ed H Chi. 2018. Latent cross: Making use of context in recurrent recommender systems. In Proceedings of the Eleventh ACM International Conference on Web Search and Data Mining. 46–54.

[3] Heng-Tze Cheng, Levent Koc, Jeremiah Harmsen, Tal Shaked, Tushar Chandra, Hrishi Aradhye, Glen Anderson, Greg Corrado, Wei Chai, Mustafa Ispir, et al. 2016. Wide & deep learning for recommender systems. In Proceedings of the 1st workshop on deep learning for recommender systems. ACM, 7–10.

[4] Yann N Dauphin, Angela Fan, Michael Auli, and David Grangier. 2017. Language modeling with gated convolutional networks. In Proceedings of the 34th International Conference on Machine Learning-Volume 70. JMLR.org, 933–941.

[5] Thore Graepel, Joaquin Quinonero Candela, Thomas Borchert, and Ralf Herbrich. 2010. Web-scale bayesian click-through rate prediction for sponsored search advertising in microsoft's bing search engine. Omnipress.

[6] Huifeng Guo, Ruiming Tang, Yunming Ye, Zhenguo Li, and Xiuqiang He. 2017. DeepFM: a factorization-machine based neural network for CTR prediction. arXiv preprint arXiv:1703.04247 (2017).

[7] Xinran He, Junfeng Pan, Ou Jin, Tianbing Xu, Bo Liu, Tao Xu, Yanxin Shi, Antoine Atallah, Ralf Herbrich, Stuart Bowers, et al. 2014. Practical lessons from predicting clicks on ads at facebook. In Proceedings of the Eighth International Workshop on Data Mining for Online Advertising. ACM, 1–9.

[8] Jie Hu, Li Shen, and Gang Sun. 2018. Squeeze-and-excitation networks. In Proceedings of the IEEE conference on computer vision and pattern recognition. 7132–7141.

[9] Tongwen Huang, Zhiqi Zhang, and Junlin Zhang. 2019. FiBiNET: combining feature importance and bilinear feature interaction for click-through rate prediction. In Proceedings of the 13th ACM Conference on Recommender Systems. 169–177.

[10] Sergey Ioffe and Christian Szegedy. 2015. Batch normalization: Accelerating deep network training by reducing internal covariate shift. arXiv preprint arXiv:1502.03167 (2015).

[11] Yuchin Juan, Yong Zhuang, Wei-Sheng Chin, and Chih-Jen Lin. 2016. Field-aware factorization machines for CTR prediction. In Proceedings of the 10th ACM Conference on Recommender Systems. ACM, 43–50.

[12] Yehuda Koren, Robert Bell, and Chris Volinsky. 2009. Matrix factorization techniques for recommender systems. Computer 8 (2009), 30–37.

[13] Jianxun Lian, Xiaohuan Zhou, Fuzheng Zhang, Zhongxia Chen, Xing Xie, and Guangzhong Sun. 2018. xdeepfm: Combining explicit and implicit feature interactions for recommender systems. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 1754–1763.

[14] Chen Ma, Peng Kang, and Xue Liu. 2019. Hierarchical gating networks for sequential recommendation. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 825–833.

[15] Jiaqi Ma, Zhe Zhao, Xinyang Yi, Jilin Chen, Lichan Hong, and Ed H Chi. 2018. Modeling task relationships in multi-task learning with multi-gate mixture-of-experts. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 1930–1939.

[16] H. Brendan McMahan, Gary Holt, D. Sculley, Michael Young, Dietmar Ebner, Julian Grady, Lan Nie, Todd Phillips, Eugene Davydov, Daniel Golovin, and et al. 2013. Ad Click Prediction: A View from the Trenches. In Proceedings of the 19th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '13). Association for Computing Machinery, New York, NY, USA, 1222–1230.

[17] Steffen Rendle. 2010. Factorization machines. In 2010 IEEE International Conference on Data Mining. IEEE, 995–1000.

[18] Steffen Rendle, Walid Krichene, Li Zhang, and John Anderson. 2020. Neural Collaborative Filtering vs. Matrix Factorization Revisited. arXiv preprint arXiv:2005.09683 (2020).

[19] Weiping Song, Chence Shi, Zhiping Xiao, Zhijian Duan, Yewen Xu, Ming Zhang, and Jian Tang. 2019. Autoint: Automatic feature interaction learning via self-attentive neural networks. In Proceedings of the 28th ACM International Conference on Information and Knowledge Management. 1161–1170.

[20] Rupesh Kumar Srivastava, Klaus Greff, and Jürgen Schmidhuber. 2015. Highway networks. arXiv preprint arXiv:1505.00387 (2015).

[21] Ruoxi Wang, Bin Fu, Gang Fu, and Mingliang Wang. 2017. Deep & cross network for ad click predictions. In Proceedings of the ADKDD'17. ACM, 12.

[22] Jun Xiao, Hao Ye, Xiangnan He, Hanwang Zhang, Fei Wu, and Tat-Seng Chua. 2017. Attentional factorization machines: Learning the weight of feature interactions via attention networks. arXiv preprint arXiv:1708.04617 (2017).

[23] Ruobing Xie, Cheng Ling, Yalong Wang, Rui Wang, Feng Xia, and Leyu Lin. 2020. Deep Feedback Network for Recommendation. 2491–2497.

[24] Weinan Zhang, Tianming Du, and Jun Wang. 2016. Deep learning over multi-field categorical data. In European conference on information retrieval. Springer, 45–57.
