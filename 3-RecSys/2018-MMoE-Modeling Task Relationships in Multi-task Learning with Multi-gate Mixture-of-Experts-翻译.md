# Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts (MMoE)（中文翻译）

> Jiaqi Ma, Zhe Zhao, Xinyang Yi, Jilin Chen, Lichan Hong, Ed H. Chi | Google
>
> ¹ 密歇根大学安娜堡分校信息学院
>
> ² Google Inc.
>
> KDD 2018

---
本文介绍了 Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts (MMoE。核心内容：


关键发现：




---

## 摘要

基于神经网络的多任务学习已成功应用于许多真实世界的大规模应用，例如推荐系统。例如，在电影推荐中，除了向用户提供他们倾向于购买和观看的电影外，系统还可能优化用户之后是否喜欢这些电影。通过多任务学习，我们旨在构建一个单一模型，同时学习这些多个目标和任务。然而，常用多任务模型的预测质量通常对任务之间的关系非常敏感。因此，研究任务特定目标与任务间关系之间的建模权衡非常重要。

在这项工作中，我们提出了一种新颖的多任务学习方法——多门混合专家（Multi-gate Mixture-of-Experts, MMoE），它从数据中显式学习建模任务关系。我们通过在所有任务之间共享专家子模型，同时为每个任务训练一个门控网络来优化每个任务，从而将混合专家（MoE）结构适应到多任务学习中。为了验证我们的方法在具有不同任务相关性的数据上的效果，我们首先将其应用于一个合成数据集，在该数据集中我们控制了任务相关性。我们证明，当任务相关性较低时，所提出的方法优于基线方法。我们还表明，MMoE结构根据训练数据和模型初始化的不同随机性水平，带来了额外的可训练性优势。此外，我们展示了MMoE在真实任务上的性能改进，包括一个二分类基准测试和Google的一个大规模内容推荐系统。

**CCS概念：** • 计算方法 \rightarrow 多任务学习；神经网络；• 信息系统 \rightarrow 推荐系统

**关键词：** 多任务学习；混合专家；神经网络；推荐系统

---

## 1 引言

近年来，深度神经网络模型已成功应用于许多真实世界的大规模应用，例如推荐系统[11]。这类推荐系统通常需要同时优化多个目标。例如，在向用户推荐电影观看时，我们可能希望用户不仅购买和观看电影，而且之后还喜欢这些电影，以便他们会回来观看更多电影。也就是说，我们可以同时创建模型来预测用户的购买和他们的评分。事实上，许多大规模推荐系统已经采用使用深度神经网络（DNN）模型的多任务学习[3]。

研究人员报告称，多任务学习模型可以通过利用正则化和迁移学习来改进所有任务的模型预测[8]。然而，在实践中，多任务学习模型并不总是在所有任务上都优于相应的单任务模型[23, 26]。事实上，许多基于DNN的多任务学习模型对数据分布差异和任务间关系等因素很敏感[15, 34]。任务差异带来的固有冲突实际上可能会损害至少某些任务的预测，特别是当模型参数在所有任务之间广泛共享时。

先前的工作[4, 6, 8]通过假设每个任务的特定数据生成过程，根据该假设测量任务差异，然后根据任务的差异程度提出建议，来研究多任务学习中的任务差异。然而，由于真实应用通常具有更复杂的数据模式，通常很难测量任务差异并利用这些先前工作中建议的方法。

最近的一些工作提出了新颖的建模技术来处理多任务学习中的任务差异，而无需依赖显式的任务差异测量[15, 27, 34]。然而，这些技术通常需要为每个任务添加更多模型参数以适应任务差异。由于大规模推荐系统可能包含数百万或数十亿个参数，这些额外的参数通常约束不足，可能会损害模型质量。由于服务资源有限，这些参数带来的额外计算成本在实际生产环境中通常也难以承受。

在本文中，我们提出了一种基于新颖的多门混合专家（MMoE）结构的多任务学习方法，该结构受混合专家（MoE）模型[21]和最近的MoE层[16, 31]的启发。MMoE显式建模任务关系并学习任务特定功能以利用共享表示。它允许自动分配参数以捕获共享任务信息或任务特定信息，避免为每个任务添加许多新参数。

MMoE的主干建立在最常用的共享底层多任务DNN结构[8]之上。共享底层模型结构如图1(a)所示，其中输入层之后的几个底层在所有任务之间共享，然后每个任务在底层表示之上有一个单独的"塔"网络。与所有任务共享一个底层网络不同，我们的模型（如图1(c)所示）有一组底层网络，每个网络被称为一个专家。在我们的论文中，每个专家是一个前馈网络。然后我们为每个任务引入一个门控网络。门控网络接收输入特征并输出softmax门，以不同的权重组合专家，允许不同的任务以不同的方式利用专家。组合专家的结果然后被传递到任务特定的塔网络。通过这种方式，不同任务的门控网络可以学习不同的专家组合混合模式，从而捕获任务关系。

为了理解MMoE如何针对不同级别的任务相关性学习其专家和任务门控网络，我们进行了合成实验，在其中我们可以通过Pearson相关性来测量和控制任务相关性。与[24]类似，我们使用两个合成回归任务，并使用正弦函数作为数据生成机制以引入非线性。我们的方法在这种设置下优于基线方法，特别是当任务相关性较低时。在这组实验中，我们还发现MMoE更容易训练，并在多次运行中收敛到更低的损失。这与最近的发现有关，即调制和门控机制可以改善非凸深度神经网络的训练性[10, 19]。

我们进一步在基准数据集UCI Census-income数据集上评估了MMoE的性能，采用多任务问题设置。我们与几种最先进的通过软参数共享建模任务关系的多任务模型进行了比较，并观察到我们的方法有所改进。

最后，我们在一个真实的大规模内容推荐系统上测试了MMoE，其中两个分类任务同时在向用户推荐item时进行学习。我们使用数千亿的训练样本训练MMoE模型，并将其与共享底层生产模型进行比较。我们观察到离线指标（如AUC）有显著改进。此外，我们的MMoE模型在在线实验中持续改进在线指标。

本文的贡献有三方面：首先，我们提出了一种新颖的多门混合专家模型，它显式建模任务关系。通过调制和门控网络，我们的模型自动调整建模共享信息和建模任务特定信息之间的参数化。其次，我们在合成数据上进行了控制实验。我们报告了任务相关性如何影响多任务学习中的训练动态，以及MMoE如何提高模型表达能力和可训练性。最后，我们在真实基准数据和一个拥有数亿用户和item的大规模生产推荐系统上进行了实验。我们的实验验证了我们提出的方法在真实环境中的效率和有效性。

## 2 相关工作

### 2.1 DNN中的多任务学习

多任务模型可以学习不同任务之间的共性和差异。这样做可以改善每个任务的效率和模型质量[4, 8, 30]。Caruana[8, 9]提出了一个广泛使用的多任务学习模型，该模型采用共享底层模型结构，其中底层隐藏层在各任务之间共享。这种结构大大降低了过拟合的风险，但可能遭受由任务差异引起的优化冲突，因为所有任务需要在共享底层上使用相同的参数集。

为了理解任务相关性如何影响模型质量，先前的工作使用合成数据生成并操作不同类型的任务相关性，以评估多任务模型的有效性[4-6, 8]。

一些近期方法不是在各任务之间共享隐藏层和相同的模型参数，而是对任务特定参数添加不同类型的约束[15, 27, 34]。例如，对于两个任务，Duong等人[15]在两个参数集之间添加L2约束。Cross-Stitch网络[27]为每个任务学习任务特定隐藏层嵌入的唯一组合。Yang等人[34]使用张量分解模型为每个任务生成隐藏层参数。与共享底层模型相比，这些方法具有更多任务特定参数，并且在任务差异导致共享参数更新冲突时可以实现更好的性能。然而，更多的任务特定参数需要更多的训练数据来拟合，并且在大规模模型中可能效率不高。

### 2.2 子网络集成与混合专家

在本文中，我们应用深度学习中的一些最新发现，如参数调制和集成方法，来建模多任务学习的任务关系。在DNN中，集成模型和子网络集成已被证明可以改善模型性能[9, 20]。

Eigen等人[16]和Shazeer等人[31]将混合专家模型转变为基本构建块（MoE层），并将其堆叠在DNN中。MoE层基于该层的输入在训练和服务时选择子网络（专家）。因此，该模型不仅在建模方面更强大，而且通过向门控网络引入稀疏性来降低计算成本。类似地，PathNet[17]是为处理不同任务的人工通用智能而设计的，它是一个具有多个层和每层内多个子模块的巨大神经网络。在训练一个任务时，多个路径被随机选择并由不同的工作节点并行训练。最佳路径的参数被固定，并为训练新任务选择新的路径。我们从这些工作中获得灵感，通过使用子网络（专家）集成来实现迁移学习，同时节省计算。

### 2.3 多任务学习应用

得益于分布式机器学习系统的发展[13]，许多大规模真实应用已采用基于DNN的多任务学习算法，并观察到显著的质量改进。在多语言机器翻译任务中，通过共享模型参数，训练数据有限的翻译任务可以通过与拥有大量训练数据的任务联合学习而得到改进[22]。对于构建推荐系统，多任务学习被发现有助于提供上下文感知的推荐[28, 35]。在[3]中，通过共享特征表示和较低级别的隐藏层，文本推荐任务得到了改进。在[11]中，使用共享底层模型来学习视频推荐的排序算法。与这些先前工作类似，我们在真实的大规模推荐系统上评估了我们的建模方法。我们证明了我们的方法确实具有可扩展性，并且与其他最先进的建模方法相比具有优越的性能。

## 3 预备知识

### 3.1 共享底层多任务模型

我们首先介绍图1(a)中的共享底层多任务模型，这是由Rich Caruana[8]提出并广泛应用于许多多任务学习应用的框架[18, 29]。因此，我们将其视为多任务建模中一个代表性的基线方法。

给定K个任务，该模型由一个表示为函数f的共享底层网络和K个塔网络hk组成，其中k = 1, 2, ..., K分别对应每个任务。共享底层网络位于输入层之后，塔网络构建在共享底层的输出之上。然后每个任务的单独输出yk跟随对应的任务特定塔。对于任务k，模型可以表述为：

yk = hk(f(x)) (1)

### 3.2 合成数据生成

先前的工作[15, 27]表明，多任务学习模型的性能高度依赖于数据中固有的任务相关性。然而，直接在真实应用中研究任务相关性如何影响多任务模型是困难的，因为在真实应用中我们无法轻易改变任务之间的相关性并观察效果。因此，为了建立这种关系的实证研究，我们首先使用合成数据，其中我们可以轻松测量和控制任务相关性。

受Kang等人[24]的启发，我们生成两个回归任务，并使用这两个任务标签的Pearson相关性作为任务关系的定量指标。由于我们关注DNN模型，我们不像[24]中使用线性函数，而是像[33]中那样将回归模型设置为正弦函数的组合。具体来说，我们按如下方式生成合成数据：

(1) 给定输入特征维度d，我们生成两个正交单位向量u1, u2 \in Rd，即 u1ᵀu2 = 0, ||u1||₂ = 1, ||u2||₂ = 1。

(2) 给定一个比例常数c和一个相关性分数−1 \leq p \leq 1，生成两个权重向量w1, w2使得：
    w1 = cu1, w2 = c(pu1 + $\sqrt{}$(1−p²)u2) (2)

(3) 随机采样一个输入数据点x \in Rd，其每个元素来自N(0, 1)。

(4) 为两个回归任务生成两个标签y1, y2如下：
    y1 = w1ᵀx + \Sigmaᵢ₌₁ᵐ sin(\alphaᵢw1ᵀx + \betaᵢ) + \epsilon₁ (3)
    y2 = w2ᵀx + \Sigmaᵢ₌₁ᵐ sin(\alphaᵢw2ᵀx + \betaᵢ) + \epsilon₂ (4)
    其中\alphaᵢ, \betaᵢ, i = 1, 2, ..., m是控制正弦函数形状的给定参数，\epsilon₁, \epsilon₂ ~ N(0, 0.01)

(5) 重复步骤(3)和(4)直到生成足够的数据。

由于非线性数据生成过程，直接生成具有给定标签Pearson相关性的任务并不简单。相反，我们操作公式2中权重向量的余弦相似度，即cos(w1, w2) = p，然后测量结果标签的Pearson相关性。注意在线性情况下，y1 = w1ᵀx + \epsilon₁, y2 = w2ᵀx + \epsilon₂，y1, y2的标签Pearson相关性正好是p。在非线性情况下，公式3和公式4中的y1和y2也是正相关的，如图2所示。在本文的其余部分，为简单起见，我们将权重向量的余弦相似度称为"任务相关性"。

图2：标签Pearson相关性与权重余弦相似度（任务相关性）的关系。X轴显示权重向量的余弦相似度。Y轴是标签之间的结果Pearson相关性。对于每个权重余弦相似度，我们生成10k个具有两个标签的数据点，并计算这两个标签之间的Pearson相关性。我们重复此过程并绘制平均值，误差线表示100次试验中的2个标准差。

### 3.3 任务相关性的影响

为了验证在基线多任务模型设置中低任务相关性会损害模型质量，我们在合成数据上进行如下控制实验：

(1) 给定一个任务相关性分数列表，为每个分数生成一个合成数据集；
(2) 在每个数据集上分别训练一个共享底层多任务模型，同时控制所有模型和训练超参数保持不变；
(3) 使用独立生成的数据集重复步骤(1)和(2)数百次，但控制任务相关性分数列表和超参数相同；
(4) 计算每个任务相关性分数的模型平均性能。

图3显示了不同任务相关性的损失曲线。正如预期的那样，模型性能随着任务相关性的降低而下降。这种趋势在许多不同的超参数设置下都是普遍的。这里我们仅在图3中展示一个控制实验结果的示例。在此示例中，每个塔网络是一个具有8个隐藏单元的单层神经网络，共享底层网络是一个大小为16的单层网络。该模型使用TensorFlow[1]实现，并使用默认设置的Adam优化器[25]进行训练。注意两个回归任务是对称的，因此报告一个任务的结果就足够了。这一现象验证了我们的假设：传统的多任务模型对任务关系敏感。

## 4 建模方法

### 4.1 混合专家

原始的混合专家（MoE）模型[21]可以表述为：

y = \Sigmaᵢ₌₁ⁿ g(x)ᵢ fᵢ(x) (5)

\text{where } \Sigmaᵢ₌₁ⁿ g(x)ᵢ = 1，g(x)ᵢ是g(x)输出的第i个logit，表示专家fᵢ的概率。

这里，fᵢ, i = 1, ..., n是n个专家网络，g表示一个门控网络，它组合所有专家的结果。更具体地说，门控网络g基于输入产生n个专家的分布，最终输出是所有专家输出的加权和。

图3：共享底层模型在具有不同任务相关性的合成数据上的性能。任务相关性为1意味着两个任务具有相同的权重向量但独立的噪声。X轴是训练步数。Y轴是200次独立运行的平均损失。

MoE层：虽然MoE最初是作为多个独立模型的集成方法开发的，但Eigen等人[16]和Shazeer等人[31]将其转变为基本构建块（MoE层）并堆叠在DNN中。MoE层具有与MoE模型相同的结构，但接受前一层的输出作为输入并输出到后续层。整个模型然后以端到端的方式进行训练。

Eigen等人[16]和Shazeer等人[31]提出的MoE层结构的主要目标是实现条件计算[7, 12]，其中网络的某些部分在每个样本的基础上激活。对于每个输入样本，模型能够通过门控网络基于输入条件选择仅一部分专家。

图4：MMoE、OMoE和共享底层在具有不同相关性的合成数据上的平均性能。

### 4.2 多门混合专家

我们提出了一种新的MoE模型，旨在捕获任务差异，而不需要比共享底层多任务模型显著更多的模型参数。这个新模型被称为多门混合专家（MMoE）模型，其关键思想是用公式5中的MoE层替代公式1中的共享底层网络f。更重要的是，我们为每个任务k添加一个单独的门控网络gᵏ。更确切地说，任务k的输出为：

yk = hk(fᵏ(x)) (6)

其中 fᵏ(x) = \Sigmaᵢ₌₁ⁿ gᵏ(x)ᵢ fᵢ(x) (7)

模型结构如图1(c)所示。

我们的实现由具有ReLU激活的相同多层感知器组成。门控网络仅是输入的线性变换加上一个softmax层：

gᵏ(x) = softmax(W_gᵏ x) (8)

其中W_gᵏ \in Rⁿˣᵈ是一个可训练的矩阵，n是专家数量，d是特征维度。

每个门控网络可以学习根据输入示例"选择"一部分专家来使用。这对于多任务学习情况下的灵活参数共享是可取的。作为一种特殊情况，如果只选择门控分数最高的一个专家，那么每个门控网络实际上将输入空间线性划分为n个区域，每个区域对应一个专家。MMoE能够以复杂的方式建模任务关系，通过决定不同门控产生的划分如何相互重叠。如果任务相关性较低，那么共享专家将受到惩罚，这些任务的门控网络将学习利用不同的专家。与共享底层模型相比，MMoE仅多了几个额外的门控网络，而门控网络中的模型参数数量可以忽略不计。因此，整个模型仍然尽可能多地享受多任务学习中知识迁移的好处。

为了理解为每个任务引入单独的门控网络如何帮助模型学习任务特定信息，我们与所有任务共享一个门控的模型结构进行了比较。我们称之为单门混合专家（OMoE）模型。这是将MoE层直接适应到共享底层多任务模型。模型结构如图1(b)所示。

## 5 MMoE在合成数据上的实验

在本节中，我们希望了解MMoE模型是否确实能更好地处理任务相关性较低的情况。与第3.3节类似，我们在合成数据上进行控制实验来研究这个问题。我们改变合成数据的任务相关性，并观察不同模型的行为如何变化。我们还进行了可训练性分析，并表明基于MoE的模型比共享底层模型更容易训练。

### 5.1 在不同任务相关性数据上的性能

我们针对提出的MMoE模型和两个基线模型（共享底层模型和OMoE模型）重复了第3.3节的实验。

**模型结构：** 输入维度为100。两种基于MoE的模型都有8个专家，每个专家实现为单层网络。专家网络中隐藏层的大小为16。塔网络仍然是大小为8的单层网络。我们注意到，共享专家和塔中的模型参数总数为100 $\times$ 16 $\times$ 8 + 16 $\times$ 8 $\times$ 2 = 13056。对于基线共享底层模型，我们仍然将塔网络设置为大小为8的单层网络。我们将单层共享底层网络的大小设置为13056/(100 + 8$\times$2) \approx 113。

**结果：** 所有模型都使用Adam优化器训练，学习率从[0.0001, 0.001, 0.01]中网格搜索。对于每个模型-相关性对设置，我们进行了200次运行，具有独立随机数据生成和模型初始化。平均结果如图4所示。观察结果概述如下：

(1) 对于所有模型，在相关性较高数据上的性能优于相关性较低数据上的性能。
(2) MMoE模型在不同相关性数据上的性能差距远小于OMoE模型和共享底层模型。这一趋势在比较MMoE模型和OMoE模型时尤为明显：在极端情况下，当两个任务完全相同时，MMoE模型和OMoE模型之间的性能几乎没有差异；然而，当任务之间的相关性降低时，OMoE模型的性能明显下降，而对MMoE模型几乎没有影响。因此，在低相关性情况下，拥有任务特定的门控来建模任务差异至关重要。
(3) 两种MoE模型在所有场景中的平均性能都优于共享底层模型。这表明MoE结构本身带来了额外的收益。基于这一观察，我们在下一小节中展示MoE模型比共享底层模型具有更好的可训练性。

### 5.2 可训练性

对于大型神经网络模型，我们非常关注其可训练性，即模型在超参数设置和模型初始化的某个范围内的鲁棒性。

最近，Collins等人[10]发现，一些我们认为性能优于普通RNN的门控RNN模型（如LSTM和GRU）仅仅是更容易训练，而不是具有更好的模型容量。虽然我们已经证明MMoE可以更好地处理任务相关性较低的情况，但我们还想更深入地了解它在可训练性方面的表现。

通过我们的合成数据，我们可以自然地研究模型对数据和模型初始化随机性的鲁棒性。我们在每个设置下多次重复实验。每次数据都来自相同的分布但随机种子不同，模型也以不同的方式初始化。我们在图5中绘制了重复运行中最终损失值的直方图。

从直方图中可以观察到三个有趣的现象。首先，在所有任务相关性设置中，共享底层模型的性能方差远大于基于MoE的模型。这意味着共享底层模型通常比基于MoE的模型有更多质量较差的局部最小值。其次，当任务相关性为1时，OMoE模型的性能方差与MMoE模型同样稳健，但当任务相关性降低到0.5时，OMoE的稳健性明显下降。注意MMoE和OMoE之间的唯一区别是是否有多门结构。这验证了多门结构在解决由任务差异引起的坏局部最小值方面的有用性。最后，值得注意的是，所有三种模型的最低损失是相当的。这并不奇怪，因为神经网络理论上是通用逼近器。在足够的模型容量下，应该存在一个"正确的"共享底层模型，可以同时很好地学习两个任务。然而，请注意这是200次独立实验运行的分布。我们怀疑对于更大更复杂的模型（例如，当共享底层网络是循环神经网络时），获得任务关系的"正确"模型的机会将更低。因此，显式建模任务关系仍然是可取的。

图5：MMoE、OMoE和共享底层多任务模型在具有不同相关性的合成数据上的性能直方图。

## 6 真实数据实验

在本节中，我们在真实数据集上进行实验，以验证我们方法的有效性。

### 6.1 基线方法

除了共享底层多任务模型之外，我们将我们的方法与几种尝试从数据中学习任务关系的最先进的多任务深度神经网络模型进行比较。

**L2约束[15]：** 该方法专为具有两个任务的跨语言问题设计。在该方法中，用于不同任务的参数通过L2约束进行软共享。

给定yᵏ作为任务k（k\in{1,2}）的真实标签，任务k的预测表示为ŷᵏ = f(x; \thetaᵏ)，其中\thetaᵏ是模型参数。

该方法的损失函数为：L(y¹, f(x; \theta¹)) + L(y², f(x; \theta²)) + \alpha||\theta¹−\theta²||₂²，其中y¹, y²是任务1和任务2的真实标签，\alpha是一个超参数。该方法通过\alpha的大小来建模任务相关性。

**Cross-Stitch[27]：** 该方法通过引入"Cross-Stitch"单元在两个任务之间共享知识。Cross-Stitch单元接收来自任务1和2的分离隐藏层的输入x¹和x²，并通过以下方程输出x̃¹和x̃²：

[x̃¹; x̃²] = [\alpha₁₁ \alpha₁₂; \alpha₂₁ \alpha₂₂] [x¹; x²]

其中\alphaⱼᵏ, j,k=1,2是可训练参数，表示从任务k到任务j的交叉迁移。x̃¹和x̃²分别被发送到任务1和任务2的更高级别层。

**张量分解[34]：** 在该方法中，来自多个任务的权重被建模为张量，并使用张量分解方法进行跨任务参数共享。在我们的比较中，我们实现了用于学习多任务模型的Tucker分解，据报道它提供了最可靠的结果[34]。例如，给定输入隐藏层大小m、输出隐藏层大小n和任务数量k，权重W（一个m$\times$n$\times$k张量）由以下方程得到：

W = \Sigmaᵢ₁ \Sigmaᵢ₂ \Sigmaᵢ₃ S(i₁,i₂,i₃) · U₁(:,i₁) ◦ U₂(:,i₂) ◦ U₃(:,i₃)

其中大小为r₁$\times$r₂$\times$r₃的张量S、大小为m$\times$r₁的矩阵U₁、大小为n$\times$r₂的矩阵U₂和大小为k$\times$r₃的矩阵U₃是可训练参数。它们都通过标准反向传播一起训练。r₁, r₂, r₃是超参数。

### 6.2 超参数调优

我们采用一个超参数调优器（用于最近的深度学习框架[10]），来搜索真实数据集实验中所有模型的最佳超参数。调优算法是一个高斯过程模型，类似于Spearmint，如[14, 32]中所述。

为了使比较公平，我们通过设置每层隐藏单元数的相同上限（2048）来约束所有方法的最大模型大小。对于MMoE，它是"专家数$\times$每个专家隐藏单元数"。我们的方法和所有基线方法都使用TensorFlow[1]实现。

我们调优所有方法的学习率和训练步数。我们还调优了一些方法特定的超参数：

• MMoE：专家数，每个专家的隐藏单元数。
• L2约束：隐藏层大小。L2约束的权重\alpha。
• Cross-Stitch：隐藏层大小，Cross-Stitch层大小。
• 张量分解：Tucker分解的r₁, r₂, r₃，隐藏层大小。

### 6.3 Census-income数据

#### 6.3.1 数据集描述

UCI Census-income数据集[2]从1994年人口普查数据库中提取。它包含299,285个美国成年人的人口统计信息实例。总共有40个特征。我们从该数据集中构造了两个多任务学习问题，将其中一些特征设置为预测目标，并计算10,000个随机样本上任务标签的Pearson相关系数的绝对值：

(1) 任务1：预测收入是否超过50K美元；
    任务2：预测此人的婚姻状况是否为从未结婚。
    绝对Pearson相关性：0.1768。

(2) 任务1：预测教育水平是否至少为大学；
    任务2：预测此人的婚姻状况是否为从未结婚。
    绝对Pearson相关性：0.2373。

数据集中有199,523个训练样本和99,762个测试样本。我们进一步将测试样本按1:1的比例随机拆分为验证数据集和测试数据集。

注意，我们从输入特征中移除了教育和婚姻状况，因为它们在这些设置中被视为标签。我们将MMoE与前述基线方法进行比较。由于两组任务都是二分类问题，我们使用AUC分数作为评估指标。在两组中，我们将婚姻状况任务视为辅助任务，并将第一组中的收入任务和第二组中的教育任务视为主任务。对于超参数调优，我们使用验证集上主任务的AUC作为目标。对于每种方法，我们使用超参数调优器进行数千次实验以找到最佳超参数设置。在超参数调优器为每种方法找到最佳超参数后，我们在训练数据集上训练每种方法400次，使用随机参数初始化，并报告测试数据集上的结果。

#### 6.3.2 结果

对于两组实验，我们报告400次运行的平均AUC，以及获得最佳主任务性能的运行的AUC。表1和表2显示了两组任务的结果。我们还调优并训练了单任务模型（为每个任务训练一个单独的模型），并报告其结果。

表1：UCI Census-income数据集第一组上的性能

| Group 1 | AUC/收入（最佳） | AUC/收入（平均） | AUC/婚姻状况（平均） | AUC/婚姻状况（最佳收入时） |
|---------|----------------|----------------|---------------------|------------------------|
| 单任务 | 0.9398 | 0.9337 | 0.9933 | 0.9922 |
| 共享底层 | 0.9361 | 0.9295 | 0.9915 | 0.9921 |
| L2约束 | 0.9389 | 0.9359 | 0.9922 | 0.9918 |
| Cross-Stitch | 0.9406 | 0.9361 | 0.9917 | 0.9922 |
| 张量分解 | 0.7460 | 0.6765 | 0.8175 | 0.8412 |
| OMoE | 0.9387 | 0.9319 | 0.9928 | 0.9923 |
| MMoE | 0.9410 | 0.9359 | 0.9926 | 0.9927 |

表2：UCI Census-income数据集第二组上的性能

| Group 2 | AUC/教育（最佳） | AUC/教育（平均） | AUC/婚姻状况（平均） | AUC/婚姻状况（最佳教育时） |
|---------|----------------|----------------|---------------------|------------------------|
| 单任务 | 0.8843 | 0.8792 | 0.9933 | 0.9922 |
| 共享底层 | 0.8836 | 0.8813 | 0.9927 | 0.9917 |
| L2约束 | 0.8855 | 0.8823 | 0.9923 | 0.9918 |
| Cross-Stitch | 0.8855 | 0.8819 | 0.9919 | 0.9921 |
| 张量分解 | 0.7367 | 0.7256 | 0.7453 | 0.7497 |
| OMoE | 0.8852 | 0.8813 | 0.9915 | 0.9912 |
| MMoE | 0.8860 | 0.8826 | 0.9932 | 0.9924 |

由于两组中的任务相关性（大致由Pearson相关性衡量）都不是非常强，共享底层模型几乎总是多任务模型中最差的（张量分解除外）。L2约束和Cross-Stitch都有每个任务的单独模型参数，并对如何学习这些参数添加了约束，因此比共享底层表现更好。然而，对模型参数学习施加约束严重依赖于任务关系假设，这不如MMoE使用的参数调制机制灵活。因此，在任务相关性比第一组更小的第二组中，MMoE在所有指标上都优于其他多任务模型。

张量分解方法在两组中都是最差的。这是因为它倾向于将所有任务的隐藏层权重泛化到低秩张量和矩阵中。该方法对任务相关性非常敏感，因为当任务相关性较低时，它倾向于过度泛化，并且需要更多的数据和更长的训练时间。

多任务模型没有在验证集上针对辅助婚姻状况任务进行调优，而单任务模型进行了调优。因此，单任务模型在辅助任务上获得最佳性能是合理的。

### 6.4 大规模内容推荐

在本节中，我们在Google Inc.的一个大规模内容推荐系统上进行实验，该系统的推荐来自数亿个独特item，面向数十亿用户。具体来说，给定用户当前消费某个item的行为，该推荐系统旨在向用户展示一个相关item列表供下一步消费。

我们的推荐系统采用了与现有一些内容推荐框架[11]类似的框架，该系统有一个候选生成器，后面是一个深度排序模型。我们设置中的深度排序模型被训练来优化两种类型的排序目标：(1) 优化与参与度相关的目标，如点击率和参与时间；(2) 优化与满意度相关的目标，如点赞率。我们的训练数据包括数千亿用户隐式反馈，如点击和点赞。如果单独训练，每个任务的模型需要学习数十亿个参数。因此，与分别学习多个目标相比，共享底层架构具有更小模型大小的优势。事实上，这种共享底层模型已经在生产中使用。

#### 6.4.1 实验设置

我们通过为深度排序模型创建两个二分类任务来评估多任务模型：(1) 预测与用户参与度相关的行为；(2) 预测与用户满意度相关的行为。我们将这两个任务命名为参与度子任务和满意度子任务。

我们的推荐系统对稀疏特征使用嵌入，并将所有稠密特征归一化到[0, 1]范围。对于共享底层模型，我们将共享底层网络实现为一个具有多个全连接层和ReLU激活的前馈神经网络。在每个任务的共享底层网络之上构建一个全连接层作为塔网络。对于MMoE，我们简单地将共享底层网络的顶层改为MMoE层，并保持输出隐藏单元的维度相同。因此，我们没有在模型训练和服务中增加额外的显著计算成本。我们还实现了基线方法，如L2约束和Cross-Stitch。由于它们的模型架构，它们的参数数量大约比共享底层模型多一倍。我们没有与张量分解进行比较，因为Tucker乘积的计算在没有大量效率工程的情况下无法扩展到数十亿级别。所有模型都使用小批量随机梯度下降（SGD）进行优化，批量大小为1024。

#### 6.4.2 离线评估结果

对于离线评估，我们在固定的300亿用户隐式反馈集上训练模型，并在100万保留数据集上进行评估。由于满意度子任务的标签比参与度子任务稀疏得多，离线结果具有很高的噪声水平。我们仅在表3中展示参与度子任务的AUC分数和R平方分数。

我们展示了在200万步（100亿样本，批量大小1024）、400万步和600万步训练后的结果。MMoE在两个指标上都优于其他模型。L2约束和Cross-Stitch比共享底层模型差。这可能是因为这两个模型建立在两个独立的单任务模型之上，并且有太多的模型参数而难以很好地约束。

表3：真实大规模推荐系统上的参与度性能

| 方法 | AUC@2M | AUC@4M | AUC@6M | R²@2M | R²@4M | R²@6M |
|---------|--------|--------|--------|-------|-------|-------|
| 共享底层 | 0.09287 | 0.08812 | 0.09159 | 0.6879 | 0.6888 | 0.6900 |
| L2约束 | 0.09213 | 0.08668 | 0.09030 | 0.6866 | 0.6881 | 0.6895 |
| Cross-Stitch | 0.09332 | 0.08949 | 0.09112 | 0.6880 | 0.6885 | 0.6899 |
| OMoE | 0.09230 | 0.08749 | 0.09085 | 0.6876 | 0.6891 | 0.6893 |
| MMoE | 0.09362 | 0.08978 | 0.09263 | 0.6894 | 0.6897 | 0.6908 |

为了更好地理解门控如何工作，我们在图6中展示了每个任务的softmax门控分布。我们可以看到MMoE学习了这两个任务之间的差异，并自动平衡了共享和非共享参数。由于满意度子任务的标签比参与度子任务的标签更稀疏，满意度子任务的门控更集中于单个专家。

图6：参与度和满意度子任务的Softmax门控分布。

#### 6.4.3 在线实验结果

最后，我们在内容推荐系统上进行了MMoE模型的在线实验。我们没有对L2约束和Cross-Stitch方法进行在线实验，因为这两个模型通过引入更多参数使服务时间加倍。

我们进行了两组实验。第一组实验是比较共享底层模型与单任务模型。共享底层模型在参与度子任务和满意度子任务上都进行了训练。单任务模型仅在参与度子任务上训练。注意，虽然未在满意度子任务上训练，但单任务模型在测试时作为排序模型提供服务，因此我们也能在其上计算满意度指标。第二组实验是将我们的MMoE模型与第一组实验中的共享底层模型进行比较。两组实验使用相同量的在线流量进行。

表4显示了这些在线实验的结果。首先，通过使用共享底层模型，我们看到满意度在线指标大幅提升了19.72%，而参与度在线指标略有下降-0.22%。其次，通过使用MMoE，与共享底层模型相比，我们同时改进了两个指标。在这个推荐系统中，参与度指标的原始值远大于满意度指标，因此在改进满意度指标的同时保持参与度指标不损失甚至有所提升是可取的。

表4：在线实验结果

| 在线实验 | 参与度指标 | 满意度指标 |
|---------|-----------|-----------|
| 共享底层相比单任务的改进 | -0.22%* | 19.72%** |
| MMoE相比共享底层的改进 | 0.25%** | 2.65%** |
| * 表示90%置信水平，** 表示95%置信水平 |

## 7 结论

我们提出了一种新颖的多任务学习方法——多门混合专家（MMoE），它从数据中显式学习建模任务关系。我们通过合成数据上的控制实验表明，所提出的方法可以更好地处理任务相关性较低的场景。我们还表明，与基线方法相比，MMoE更容易训练。通过在基准数据集和真实大规模推荐系统上的实验，我们证明了所提出方法相对于几种最先进的基线多任务学习模型的成功。

除了上述优点外，真实机器学习生产系统中的另一个主要设计考虑因素是计算效率。这也是共享底层多任务模型被广泛使用的最重要原因之一。模型的共享部分在服务时节省了大量计算[18, 29]。所有三种最先进的基线模型（见第6.1节）都以损失这种计算优势为代价来学习任务关系。然而，MMoE模型在很大程度上保留了计算优势，因为门控网络通常是轻量级的，并且专家网络在所有任务之间共享。此外，该模型有潜力通过使门控网络成为稀疏的top-k门[31]来实现更好的计算效率。我们希望这项工作能启发其他研究人员进一步研究使用这些方法进行多任务建模。

## 参考文献

[1] Martín Abadi, Ashish Agarwal, Paul Barham, Eugene Brevdo, Zhifeng Chen, Craig Citro, Greg S Corrado, Andy Davis, Jeffrey Dean, Matthieu Devin, et al. 2016. TensorFlow: Large-scale machine learning on heterogeneous distributed systems. arXiv preprint arXiv:1603.04467 (2016).

[2] Arthur Asuncion and David Newman. 2007. UCI machine learning repository. (2007).

[3] Trapit Bansal, David Belanger, and Andrew McCallum. 2016. Ask the gru: Multi-task learning for deep text recommendations. In Proceedings of the 10th ACM Conference on Recommender Systems. ACM, 107–114.

[4] Jonathan Baxter et al. 2000. A model of inductive bias learning. J. Artif. Intell. Res.(JAIR) 12, 149-198 (2000), 3.

[5] Shai Ben-David, Johannes Gehrke, and Reba Schuller. 2002. A theoretical framework for learning from a pool of disparate data sources. In Proceedings of the eighth ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 443–449.

[6] Shai Ben-David, Reba Schuller, et al. 2003. Exploiting task relatedness for multiple task learning. Lecture notes in computer science (2003), 567–580.

[7] Yoshua Bengio, Nicholas Léonard, and Aaron Courville. 2013. Estimating or propagating gradients through stochastic neurons for conditional computation. arXiv preprint arXiv:1308.3432 (2013).

[8] Rich Caruana. 1998. Multitask learning. In Learning to learn. Springer, 95–133.

[9] R Caruna. 1993. Multitask learning: A knowledge-based source of inductive bias. In Machine Learning: Proceedings of the Tenth International Conference. 41–48.

[10] Jasmine Collins, Jascha Sohl-Dickstein, and David Sussillo. 2016. Capacity and Trainability in Recurrent Neural Networks. arXiv preprint arXiv:1611.09913 (2016).

[11] Paul Covington, Jay Adams, and Emre Sargin. 2016. Deep neural networks for youtube recommendations. In Proceedings of the 10th ACM Conference on Recommender Systems. ACM, 191–198.

[12] Andrew Davis and Itamar Arel. 2013. Low-rank approximations for conditional feedforward computation in deep neural networks. arXiv preprint arXiv:1312.4461 (2013).

[13] Jeffrey Dean, Greg Corrado, Rajat Monga, Kai Chen, Matthieu Devin, Mark Mao, Andrew Senior, Paul Tucker, Ke Yang, Quoc V Le, et al. 2012. Large scale distributed deep networks. In Advances in neural information processing systems. 1223–1231.

[14] Thomas Desautels, Andreas Krause, and Joel W Burdick. 2014. Parallelizing exploration-exploitation tradeoffs in gaussian process bandit optimization. The Journal of Machine Learning Research 15, 1 (2014), 3873–3923.

[15] Long Duong, Trevor Cohn, Steven Bird, and Paul Cook. 2015. Low Resource Dependency Parsing: Cross-lingual Parameter Sharing in a Neural Network Parser. In ACL (2). 845–850.

[16] David Eigen, Marc'Aurelio Ranzato, and Ilya Sutskever. 2013. Learning factored representations in a deep mixture of experts. arXiv preprint arXiv:1312.4314 (2013).

[17] Chrisantha Fernando, Dylan Banarse, Charles Blundell, Yori Zwols, David Ha, Andrei A Rusu, Alexander Pritzel, and Daan Wierstra. 2017. Pathnet: Evolution channels gradient descent in super neural networks. arXiv preprint arXiv:1701.08734 (2017).

[18] Ross Girshick. 2015. Fast r-cnn. In Proceedings of the IEEE international conference on computer vision. 1440–1448.

[19] Xavier Glorot and Yoshua Bengio. 2010. Understanding the difficulty of training deep feedforward neural networks. In Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics. 249–256.

[20] Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. 2015. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531 (2015).

[21] Robert A Jacobs, Michael I Jordan, Steven J Nowlan, and Geoffrey E Hinton. 1991. Adaptive mixtures of local experts. Neural computation 3, 1 (1991), 79–87.

[22] Melvin Johnson, Mike Schuster, Quoc V Le, Maxim Krikun, Yonghui Wu, Zhifeng Chen, Nikhil Thorat, Fernanda Viégas, Martin Wattenberg, Greg Corrado, et al. 2016. Google's multilingual neural machine translation system: enabling zero-shot translation. arXiv preprint arXiv:1611.04558 (2016).

[23] Lukasz Kaiser, Aidan N Gomez, Noam Shazeer, Ashish Vaswani, Niki Parmar, Llion Jones, and Jakob Uszkoreit. 2017. One Model To Learn Them All. arXiv preprint arXiv:1706.05137 (2017).

[24] Zhuoliang Kang, Kristen Grauman, and Fei Sha. 2011. Learning with whom to share in multi-task feature learning. In Proceedings of the 28th International Conference on Machine Learning (ICML-11). 521–528.

[25] Diederik Kingma and Jimmy Ba. 2014. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980 (2014).

[26] Minh-Thang Luong, Quoc V Le, Ilya Sutskever, Oriol Vinyals, and Lukasz Kaiser. 2015. Multi-task sequence to sequence learning. arXiv preprint arXiv:1511.06114 (2015).

[27] Ishan Misra, Abhinav Shrivastava, Abhinav Gupta, and Martial Hebert. 2016. Cross-stitch networks for multi-task learning. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 3994–4003.

[28] Xia Ning and George Karypis. 2010. Multi-task learning for recommender system. In Proceedings of 2nd Asian Conference on Machine Learning. 269–284.

[29] Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun. 2015. Faster R-CNN: Towards real-time object detection with region proposal networks. In Advances in neural information processing systems. 91–99.

[30] Sebastian Ruder. 2017. An overview of multi-task learning in deep neural networks. arXiv preprint arXiv:1706.05098 (2017).

[31] Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. 2017. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. arXiv preprint arXiv:1701.06538 (2017).

[32] Jasper Snoek, Hugo Larochelle, and Ryan P Adams. 2012. Practical bayesian optimization of machine learning algorithms. In Advances in neural information processing systems. 2951–2959.

[33] Shengyang Sun, Changyou Chen, and Lawrence Carin. 2017. Learning Structured Weight Uncertainty in Bayesian Neural Networks. In Artificial Intelligence and Statistics. 1283–1292.

[34] Yongxin Yang and Timothy Hospedales. 2016. Deep multi-task representation learning: A tensor factorisation approach. arXiv preprint arXiv:1605.06391 (2016).

[35] Zhe Zhao, Zhiyuan Cheng, Lichan Hong, and Ed H Chi. 2015. Improving user topic interest profiles by behavior factorization. In Proceedings of the 24th International Conference on World Wide Web. International World Wide Web Conferences Steering Committee, 1406–1416.
