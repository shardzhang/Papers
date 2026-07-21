# 推荐接下来要观看的视频：一个多任务排序系统

> Zhe Zhao, Lichan Hong, Li Wei, Jilin Chen, Aniruddh Nath, Shawn Andrews, Aditee Kumthekar, Maheswaran Sathiamoorthy, Xinyang Yi, Ed H. Chi | Google


本文介绍了 推荐接下来要观看的视频：一个多任务排序系统。核心内容：


关键发现：

---


**Zhe Zhao, Lichan Hong, Li Wei, Jilin Chen, Aniruddh Nath, Shawn Andrews, Aditee Kumthekar, Maheswaran Sathiamoorthy, Xinyang Yi, Ed Chi**

Google, Inc.

{zhezhao,lichan,liwei,jilinc,aniruddhnath,shawnandrews,aditeek,nlogn,xinyang,edchi}@google.com


---

## 摘要

本文介绍了一个大规模多目标排序系统，用于在工业视频共享平台上推荐接下来要观看的视频。该系统面临许多实际挑战，包括存在多个相互竞争的排序目标，以及用户反馈中的隐式选择偏差。为了应对这些挑战，我们探索了多种软参数共享技术，如多门混合专家（Multi-gate Mixture-of-Experts），以高效地优化多个排序目标。此外，我们通过采用 Wide & Deep 框架来缓解选择偏差。我们证明了所提出的技术能够在一个全球最大的视频共享平台上显著提升推荐质量。

## CCS 概念

- **信息系统 \rightarrow 检索模型与排序；推荐系统；**
- **计算方法 \rightarrow 排序；多任务学习；从隐式反馈中学习。**

## 关键词

推荐与排序；多任务学习；选择偏差

## ACM 引用格式

Zhe Zhao, Lichan Hong, Li Wei, Jilin Chen, Aniruddh Nath, Shawn Andrews, Aditee Kumthekar, Maheswaran Sathiamoorthy, Xinyang Yi, Ed Chi. 2019. 推荐接下来要观看的视频：一个多任务排序系统. 载于第十三届 ACM 推荐系统会议 (RecSys '19), 2019年9月16–20日, 丹麦哥本哈根. ACM, 纽约, NY, USA, 9页.
https://doi.org/10.1145/3298689.3346997

---
本文介绍了 推荐接下来要观看的视频：一个多任务排序系统。核心内容：


关键发现：



## 1 引言

本文描述了一个用于视频推荐的大规模排序系统。具体而言，给定用户当前正在观看的视频，推荐用户可能观看和喜欢的下一个视频。典型的推荐系统采用两阶段设计：候选生成和排序 [10, 20]。本文聚焦于排序阶段。在此阶段，推荐系统接收来自候选生成阶段（如矩阵分解 [45] 或神经模型 [25]）检索到的数百个候选，并应用复杂的大容量模型对最有前景的item进行排序。

我们介绍在大型工业视频发布和共享平台上构建此类排序系统的实验和经验教训。

设计与开发一个真实世界的大规模视频推荐系统充满挑战，包括：

- **存在不同且有时相互冲突的优化目标。** 例如，除了观看之外，我们可能希望推荐用户评分高、与朋友分享的视频。
- **系统中存在隐式偏差。** 例如，用户可能仅仅因为某个视频排序靠前而点击和观看，并非因为这是用户最喜欢的视频。因此，使用当前系统生成的数据训练的模型会产生偏差，导致反馈循环效应 [33]。如何有效且高效地学习以减少此类偏差仍然是一个开放性问题。

为了应对这些挑战，我们为排序系统提出了一种高效的多任务神经网络架构，如图1所示。它通过采用多门混合专家（MMoE）[30] 扩展了 Wide & Deep [9] 模型架构以进行多任务学习。此外，它还引入了一个浅层塔（shallow tower）来建模和消除选择偏差。我们将该架构应用于视频推荐案例研究：根据用户当前正在观看的内容，推荐下一个要观看的视频。我们在工业级大规模视频发布和共享平台上展示了所提排序系统的实验，实验结果表明我们的系统取得了显著改进。

具体而言，我们首先将多个目标分为两类：1）参与度目标（engagement objectives），如用户点击和与推荐视频的参与程度；2）满意度目标（satisfaction objectives），如用户在 YouTube 上点赞视频以及对推荐进行评分。为了学习和估计多种类型的用户行为，我们使用 MMoE 自动学习在可能相互冲突的目标之间共享参数。混合专家（Mixture-of-Experts）[21] 架构将输入层模块化为多个专家，每个专家关注输入的不同方面。这改进了从多模态生成的复杂特征空间中学习到的表示。然后，通过利用多个门控网络，每个目标可以选择与其他目标共享或不共享专家。

为了对训练数据中的选择偏差（如位置偏差）进行建模和减少，我们提议向主模型添加一个浅层塔，如图1左侧所示。浅层塔接收与选择偏差相关的输入（例如当前系统决定的排序顺序），并输出一个标量作为主模型最终预测的偏置项。该模型架构将训练数据中的标签分解为两部分：主模型学习的无偏用户效用，以及浅层塔学习的估计倾向分数。我们提出的模型架构可视为 Wide & Deep 模型的扩展，其中浅层塔代表 Wide 部分。通过将浅层塔与主模型一起直接学习，我们可以在无需借助随机实验来获取倾向分数的情况下学习选择偏差 [41]。

为了评估我们提出的排序系统，我们设计和进行了离线与在线实验，以验证以下两方面的有效性：1）多任务学习；2）消除一种常见的选择偏差——位置偏差。与最先进的基线方法相比，我们展示了所提框架的显著改进。我们使用 YouTube（全球最大的视频共享平台之一）进行实验。

本文的其余部分组织如下：第2节描述了构建真实世界推荐排序系统的相关工作。第3节给出了候选生成和排序的问题描述。接下来，我们从多任务学习和消除选择偏差两个方面讨论我们的方法。第5节描述了我们如何设计离线和在线实验来评估所提出的框架。最后，第6节总结了我们的发现。

## 2 相关工作

推荐问题可以描述为：给定查询、上下文和item列表，返回一组高效用item。例如，个性化电影推荐系统可以将用户的观看历史作为查询，将上下文（如周五晚上在家中使用平板电脑）以及电影列表作为输入，返回该用户可能观看和喜欢的电影子集。在本节中，我们从三个方面讨论相关工作：工业推荐系统案例研究、多目标推荐系统以及理解训练数据中的偏差。

总之，我们的贡献如下：

- 我们引入了一个用于视频推荐的端到端排序系统。
- 我们将排序问题形式化为一个多目标学习问题，并扩展了多门混合专家架构以提高所有目标的性能。
- 我们提出应用 Wide & Deep 模型架构来建模和减轻位置偏差。
- 我们在真实世界的大规模视频推荐系统上评估了我们的方法，并展示了显著的改进。

### 2.1 工业推荐系统

要设计和开发一个由机器学习模型驱动的成功排序系统，我们需要大量的训练数据。最近的工业推荐系统严重依赖大量用户日志来构建模型。一种选择是直接询问用户对item效用的显式反馈。然而，由于成本原因，显式反馈的数量难以扩展。因此，排序系统通常利用隐式反馈，如点击和对推荐item的参与度。

大多数推荐系统 [10, 20, 42] 包含两个阶段：候选生成和排序。对于候选生成，使用多种信号源和模型。例如，[26] 使用item共现来生成候选，[11] 采用基于协同过滤的方法，[14] 和 [19] 在（共现）图上应用随机游走，[42] 学习内容表示以过滤item到候选集，[10] 描述了一种使用混合特征的混合方法。

对于排序阶段，使用学习排序（learning-to-rank）框架的机器学习算法被广泛采用。例如，[26] 探索了使用线性模型和基于树的方法的点式（point-wise）和配对式（pair-wise）学习排序框架。[16] 使用线性评分函数和配对式排序目标。[20] 应用梯度提升决策树（GBDT [24]）进行点式排序目标。[10] 使用神经网络进行点式排序目标以预测加权点击。

这些工业推荐系统的一个主要挑战是可扩展性。因此，它们通常采用基础设施改进 [11, 14, 19, 26] 和高效机器学习算法 [14, 16, 17, 42] 的组合。为了在模型质量和效率之间进行权衡，一个流行的选择是使用基于深度神经网络的点式排序模型 [10]。

在本文中，我们首先识别出工业排序系统中的一个关键问题：用户隐式反馈与推荐item的真实用户效用之间的不一致。随后，我们引入了一个基于深度神经网络的排序模型，该模型使用多任务学习技术支持多个排序目标，每个目标对应一种类型的用户反馈。

### 2.2 推荐系统的多目标学习

从训练数据中学习和预测用户行为具有挑战性。存在不同类型的用户行为，如点击 [22]、评分、评论等。然而，每种行为并不能独立反映真实的用户效用。例如，用户可能会点击一个item但最终并不喜欢；用户只能对点击和参与过的item提供评分。我们的排序系统需要能够有效地学习和估计多种类型的行为和效用，然后将这些估计组合起来计算最终的排序效用分数。

现有的行为感知和多目标推荐工作要么仅适用于候选生成阶段 [3, 28, 31, 40, 45]，要么不适合大规模在线排序 [13, 15, 38, 44]。

例如，一些推荐系统 [31, 45] 扩展了协同过滤或基于内容的系统，从多个用户信号中学习用户-item相似度。这些系统被有效地用于生成候选。但与基于深度神经网络的排序模型相比，它们在提供最终推荐方面效果较差 [10]。

另一方面，许多现有的多目标排序系统是为特定类型的特征和应用设计的，如文本 [38] 和视觉 [13]。扩展这些系统以支持来自多种模态的特征空间（例如，视频标题中的文本、缩略图的视觉特征）将具有挑战性。同时，其他考虑多种输入模态的多目标排序系统由于在高效共享多个目标的模型参数方面的局限性而无法扩展 [15, 44]。

在推荐系统研究领域之外，基于深度神经网络的多任务学习已在许多传统机器学习应用中得到广泛研究和探索，用于表示学习，例如自然语言处理 [12] 和计算机视觉 [27]。虽然许多为表示学习提出的多任务学习技术对于构建排序系统并不实用，但它们的一些构建模块启发了我们的设计。在本文中，我们描述了一个为真实世界推荐设计的基于 DNN 的排序系统，并应用了混合专家层 [21] 的扩展来支持多任务学习 [30]。

### 2.3 理解与建模训练数据中的偏差

用作训练数据的用户日志记录了用户在当前生产系统推荐下的行为和响应。用户与当前系统之间的交互在反馈中产生了选择偏差。例如，用户可能因为某个item被当前系统选中而点击了它，即使它并不是整个库中最有用的item。因此，在当前系统产生的数据上训练的新模型会偏向当前系统，导致反馈循环效应。如何有效且高效地学习以减少排序系统的此类偏差仍然是一个开放性问题。

Joachims 等人 [22] 首次分析了隐式反馈数据中用于训练学习排序模型的位置偏差和呈现偏差。通过将点击数据与相关性显式反馈进行比较，他们发现点击数据中存在位置偏差，并且会显著影响学习排序模型在估计查询与文档之间相关性时的效果。基于这一发现，许多方法被提出来消除此类选择偏差，尤其是位置偏差 [23, 34, 41]。

一种常见的做法是在模型训练中将位置作为输入特征注入，然后在服务时通过消融来消除偏差。在概率点击模型中，位置用于学习 P(相关性 | 位置)。一种消除位置偏差的方法受到 [8] 的启发，Chapelle 等人使用 P(相关性 | 位置 = 1) 评估 CTR 模型，假设在位置 1 处进行评估没有位置偏差影响。随后，为了消除位置偏差，我们可以使用位置作为输入特征训练模型，并在服务时将位置特征设置为 1（或其他固定值如缺失值）。

其他方法尝试从位置学习偏置项并将其用作归一化器或正则化器 [23, 34, 41]。通常，为了学习偏置项，需要使用一些随机数据来推断偏置项（称为"全局偏置"、"倾向性"等），而不考虑相关性 [34, 41]。在 [23] 中，使用反事实模型学习逆倾向分数（IPS），无需随机数据。它被用作训练 Rank-SVM 时的正则化项。

在真实世界的推荐系统中，特别是社交媒体平台如 Twitter [19] 和 YouTube [10]，用户行为和item流行度每天都在显著变化。因此，与基于 IPS 的方法不同，我们需要一种高效的方式来适应训练数据分布的变化，同时在训练主排序模型时对选择偏差进行建模。

最后，所有候选被汇集到一个集合中，随后由排序系统评分。

## 3 问题描述

在本节中，我们首先描述推荐下一个要观看视频的问题，然后介绍候选生成和排序的两阶段设置。本文的其余部分将聚焦于排序系统。

除了上述使用隐式反馈构建排序系统的挑战外，对于真实世界的大规模视频推荐问题，我们还需要考虑以下额外因素：

- **多模态特征空间。** 在上下文感知的个性化推荐系统中，我们需要学习候选视频的用户效用，其特征空间来源于多种模态，例如视频内容、缩略图、音频、标题和描述、用户人口统计信息。与其它机器学习应用相比，从多模态特征空间中学习推荐表示具有独特的挑战性。它涉及两个困难问题：1）弥合用于内容过滤的低层内容特征的语义鸿沟；2）从item的稀疏分布中学习以进行协同过滤。

- **可扩展性。** 可扩展性极为重要，因为我们正在为数十亿用户和视频构建推荐系统。模型必须在训练时有效且在服务时高效。尽管排序系统每次查询仅对数百个候选进行评分，但实际场景要求实时评分，因为某些查询和上下文信息仅在在线时可用。因此，排序系统不仅需要学习数十亿item和用户的表示，还必须在服务期间保持高效。

回顾一下，我们推荐系统的目标是：给定当前正在观看的视频和上下文，提供一个排序后的视频列表。为了处理多模态特征空间，对于每个视频，我们提取视频元数据和视频内容信号等特征作为其表示。对于上下文，我们使用用户人口统计信息、设备、时间和位置等特征。

为了处理可扩展性，与 [10] 中描述的方法类似，我们的推荐系统有两个阶段：候选生成和排序。在候选生成阶段，我们从海量语料库中检索出数百个候选。我们的排序系统为每个候选提供分数并生成最终的排序列表。

### 3.1 候选生成

我们的视频推荐系统使用多种候选生成算法，每种算法捕获查询视频与候选视频之间的某一方面的相似性。例如，一种算法通过匹配查询视频的主题来生成候选。另一种算法根据视频与查询视频一起被观看的频率来检索候选视频。我们构建了一个类似于 [10] 的序列模型，根据用户历史生成个性化候选。我们还使用了 [25] 中提到的技术来生成上下文感知的高召回率相关候选。

### 3.2 排序

我们的排序系统从数百个候选中生成一个排序列表。与试图过滤掉大部分item并仅保留相关item的候选生成不同，排序系统旨在提供一个排序列表，使得对用户效用最高的item显示在顶部。因此，我们在排序系统中应用使用神经网络架构的最先进的机器学习技术，以便具有足够的模型表达能力来学习特征的关联及其与效用的关系。

## 4 模型架构

在本节中，我们将详细描述我们提出的排序系统。我们首先提供系统概述，包括问题形式化、目标和特征。然后讨论我们用于学习多种类型用户行为的多目标设置。我们讨论如何应用和扩展一种称为多门混合专家（MMoE）的最先进的多任务学习模型架构来学习多个排序目标。最后，我们讨论如何将 MMoE 与浅层塔结合，以学习和减少训练数据中的选择偏差，尤其是位置偏差。

### 4.1 系统概述

我们的排序系统从两种类型的用户反馈中学习：1）参与行为，如点击和观看；2）满意度行为，如点赞和忽略。给定每个候选，排序系统使用候选、查询和上下文的特征作为输入，并学习预测多种用户行为。

对于问题形式化，我们采用学习排序框架 [6]。我们将排序问题建模为多目标分类和回归问题的组合。给定查询、候选和上下文，排序模型预测用户采取行动（如点击、观看、点赞和忽略）的概率。

这种对每个候选进行预测的方法是点式方法 [6]。相比之下，配对式或列表式方法学习对两个或多个候选的顺序进行预测。配对式或列表式方法可以潜在地提高推荐的多样性。然而，我们选择使用点式排序主要是基于服务方面的考虑。在服务时，点式排序简单且高效，可以扩展到大量候选。相比之下，配对式或列表式方法需要对配对或列表进行多次评分才能找到给定候选集的最优排序列表，从而限制了它们的可扩展性。

### 4.2 排序目标

我们使用用户行为作为训练标签。由于用户可能对推荐item有不同的行为类型，我们将排序系统设计为支持多个目标。每个目标预测与用户效用相关的一种用户行为。为了便于描述，下面我们将目标分为两类：参与度目标和满意度目标。

参与度目标捕获用户行为如点击和观看。我们将这些行为的预测形式化为两种类型的任务：针对点击等行为的二分类任务，以及针对与花费时间相关的行为的回归任务。类似地，对于满意度目标，我们将与用户满意度相关的行为预测形式化为二分类或回归任务。例如，为视频点赞的行为被形式化为二分类任务，而评分行为被形式化为回归任务。对于二分类任务，我们计算交叉熵损失。对于回归任务，我们计算平方损失。

一旦确定了多个排序目标及其问题类型，我们就训练一个多任务排序模型来完成这些预测任务。对于每个候选，我们接收多个预测的输入，并使用加权乘法形式的组合函数输出组合分数。权重通过手动调优以在用户参与度和用户满意度上都取得最佳性能。

### 4.3 使用多门混合专家建模任务关系与冲突

具有多个目标的排序系统通常使用共享底层（shared-bottom）模型架构 [7, 10]。然而，这种硬参数共享技术在任务间相关性较低时有时会损害多目标的学习 [30]。为了缓解多目标之间的冲突，我们采用并扩展了一种最近发布的模型架构——多门混合专家（MMoE）[30]。

MMoE 是一种软参数共享模型结构，旨在建模任务冲突和关系。它将混合专家（MoE）结构适配到多任务学习中，使专家在所有任务之间共享，同时为每个任务训练一个门控网络。MMoE 层旨在捕获任务差异，而无需与共享底层模型相比显著增加模型参数。关键思想是用 MoE 层替换共享的 ReLU 层，并为每个任务添加单独的门控网络。

对于我们的排序系统，我们提出在共享隐藏层之上添加专家，如图2b所示。这是因为混合专家层可以帮助从输入中学习模块化信息 [21]。当直接应用于输入层或较低隐藏层之上时，它可以更好地建模多模态特征空间。然而，直接在输入层上应用 MoE 层将显著增加模型训练和服务成本。这是因为输入层的维度通常远高于隐藏层的维度。

我们的专家网络实现与使用 ReLU 激活的多层感知器相同 [30]。给定任务 k、预测 yk 和最后一个隐藏层 hk，具有 n 个专家的 MMoE 层对任务 k 的输出 f_k(x) 可以用以下方程表示：

yk = hk(f_k(x))

其中 f_k(x) = \Sigmaᵢ₌₁ⁿ g_k⁽ⁱ⁾(x) fᵢ(x)    (1)

而 x \in Rᵈ 是较低层共享的隐藏嵌入，g_k 是任务 k 的门控网络，g_k(x) \in Rⁿ，g_k⁽ⁱ⁾(x) 是第 i 个分量，fᵢ(x) 是第 i 个专家。门控网络是带有 softmax 层的输入的线性变换：

g_k(x) = softmax(W_gk x)    (2)

其中 W_gk \in Rⁿˣᵈ 是线性变换的自由参数。与 [32] 中提到的稀疏门控网络（其中专家数量可以很大且每个训练样本仅使用 top 专家）相比，我们使用了相对较少的专家数量。这样做是为了鼓励多个门控网络共享专家并提高训练效率。

### 4.4 建模与消除位置与选择偏差

隐式反馈已被广泛用于训练学习排序模型。利用从用户日志中提取的大量隐式反馈，可以训练复杂的基于深度神经网络的模型。然而，隐式反馈是有偏差的，因为它是由现有排序系统生成的。位置偏差和许多其他类型的选择偏差，已经在不同的排序问题中得到研究和验证 [2, 23, 41]。

在我们的排序系统中，查询是当前正在观看的视频，候选是相关视频，用户倾向于点击和观看列表中显示在更靠近顶部位置的视频，而不考虑其实际用户效用——无论是对已观看视频的相关性还是用户偏好。我们的目标是从排序模型中消除这种位置偏差。在我们的训练数据中或在模型训练期间建模和减少选择偏差，可以提高模型质量并打破由选择偏差导致的反馈循环。

我们提出的模型架构类似于 Wide & Deep 模型架构。我们将模型预测分解为两个组件：来自主塔的用户效用组件和来自浅层塔的偏差组件。具体而言，我们训练一个浅层塔，使用导致选择偏差的特征（如位置特征用于位置偏差），然后将其添加到主模型的最终 logit，如图3所示。在训练中，使用所有曝光的位置，并采用 10% 的特征丢弃率以防止模型过度依赖位置特征。在服务时，位置特征被视为缺失。我们将位置特征与设备特征交叉的原因是在不同类型的设备上观察到不同的位置偏差。

## 5 实验结果

在本节中，我们描述如何在全球最大的视频共享平台之一 YouTube 上进行所提出的排序系统的实验，以推荐接下来要观看的视频。利用 YouTube 提供的用户隐式反馈，我们训练排序模型，并进行离线和在线实验。

YouTube 的规模和复杂性使其成为我们排序系统的完美试验平台。YouTube 是最大的视频共享平台，拥有 19 亿月活跃用户。该网站每天以用户与推荐结果互动的形式产生数千亿条用户日志。YouTube 的一个关键产品功能是提供在给定已观看视频的情况下推荐接下来要观看的内容，如图4所示。其用户界面提供了多种方式供用户与推荐视频互动，如点击、观看、点赞和忽略。

### 5.1 实验设置

如第3.1节所述，我们的排序系统从多种候选生成算法接收数百个候选。我们使用 TensorFlow 构建模型的训练和服务。具体来说，我们使用张量处理单元（TPU）训练模型，并使用 TFX Servo [4] 提供服务。

我们按顺序训练我们提出的模型和基线模型。这意味着我们按照时间顺序逐天遍历训练数据，并持续运行训练器以消费新到达的训练数据。通过这样做，我们的模型可以适应最新的数据。这对于许多真实世界的推荐应用至关重要，因为数据分布和用户模式会随时间动态变化。

对于离线实验，我们监控分类任务的 AUC 和回归任务的平方误差。对于在线实验，我们进行 A/B 测试，与生产系统进行比较。我们同时使用离线和在线指标来调优超参数，如学习率。我们检查多个参与度指标（如用户在 YouTube 上花费的时间）和满意度指标（如忽略率、用户调查响应等）。除了在线指标，我们还关心模型在服务时的计算成本，因为 YouTube 每秒需要响应大量的查询。

### 5.2 使用 MMoE 的多任务排序

为了评估采用 MMoE 进行多任务排序的性能，我们与基线方法进行了比较，并在 YouTube 上进行了在线实验。

#### 5.2.1 基线方法
我们的基线方法使用图2a中提到的共享底层模型架构。作为代理，我们通过每个模型架构内部的乘法次数来衡量模型复杂度，因为这是服务模型的主要计算成本。在比较 MMoE 模型和基线模型时，我们使用相同的模型复杂度。出于效率考虑，我们的 MMoE 层共享一个底层隐藏层（如图2b所示），其维度低于输入层的维度。

#### 5.2.2 在线实验结果
YouTube 上的在线实验结果如表1所示。我们报告了参与度指标（捕获用户观看推荐视频所花费的时间）和满意度指标（捕获用户调查响应评分分数）的结果。我们比较了使用 4 个或 8 个专家的共享底层模型和 MMoE 模型。从表中可以看出，在相同的模型复杂度下，MMoE 显著提高了参与度和满意度指标。

**表1：YouTube 上 MMoE 的在线实验结果**

| 模型架构 | 乘法次数 | 参与度指标 | 满意度指标 |
|---|---|---|---|
| 共享底层 | 3.7M | / | / |
| 共享底层 | 6.1M | +0.1% | + 1.89% |
| MMoE（4个专家） | 3.7M | +0.20% | + 1.22% |
| MMoE（8个专家） | 6.1M | +0.45% | + 3.07% |

#### 5.2.3 门控网络分布
为了进一步理解 MMoE 如何帮助多目标优化，我们绘制了每个任务在 softmax 门控网络中每个专家上的累积概率，如图5所示。我们看到一些参与度任务与其他参与度任务共享多个专家。而满意度任务倾向于共享一个高利用率的小专家子集（以使用这些专家的概率衡量）。如上所述，我们的 MMoE 层共享一个底层隐藏层，其门控网络从共享隐藏层接收输入。这可能使得 MMoE 层比直接从输入层构建 MMoE 层更难模块化输入信息。作为替代方案，我们让门控网络直接接收输入层（而非共享隐藏层）的输入，以便输入特征可以直接用于选择专家。然而，在线实验结果显示与图2b的 MMoE 层相比没有实质性差异。这表明图2b中 MMoE 的门控网络可以有效地将输入信息模块化为专家，以进行任务关系和冲突建模。

#### 5.2.4 门控网络稳定性
当使用多台机器训练神经网络模型时，分布式训练策略可能导致模型频繁发散。发散的一个例子是 ReLU 死亡 [1]。在 MMoE 中，softmax 门控网络被报告存在专家分布不平衡问题 [32]，即门控网络收敛到大多数专家的利用率为零。在分布式训练中，我们在模型中观察到 20% 的概率出现这种门控网络极化问题。门控网络极化会损害使用极化门控网络的任务的模型性能。为了解决这个问题，我们在门控网络上应用 dropout。通过应用 10% 的概率将专家利用率设置为 0 并重新归一化 softmax 输出，我们消除了所有门控网络的极化问题。

### 5.3 建模与减少位置偏差

使用用户隐式反馈作为训练数据的一个主要挑战是难以对隐式反馈与真实用户效用之间的差距进行建模。通过使用多种类型的隐式信号和多个排序目标，我们在服务时有更多可调参数来捕获从模型预测到item推荐中用户效用的转换。然而，我们仍然需要对隐式反馈中普遍存在的偏差（例如由用户与当前推荐系统交互引起的选择偏差）进行建模和减少。

这里我们评估如何使用我们提出的轻量级模型架构来建模和减少一种类型的选择偏差——位置偏差。我们的解决方案避免了随机实验或复杂计算的开销 [41]。

#### 5.3.1 用户隐式反馈分析
为了验证位置偏差存在于我们的训练数据中，我们对不同位置的点击率（CTR）进行了分析。图6显示了位置 1 到 9 的相对尺度 CTR 分布。正如预期，我们看到随着位置的降低，CTR 显著降低。较高位置处的较高 CTR 是推荐更相关item和位置偏差的综合效应。通过使用我们提出的采用浅层塔的方法，我们在下文中证明它可以分离用户效用学习和位置偏差学习。

#### 5.3.2 基线方法
为了评估我们提出的模型架构，我们将其与以下基线方法进行比较：

- **直接将位置特征作为输入特征：** 这种简单方法已被工业推荐系统广泛采用以消除位置偏差，主要用于线性学习排序模型。
- **对抗学习：** 受对抗学习在领域自适应 [37] 和机器学习公平性 [5] 中广泛采用的启发，我们使用类似的技术引入一个辅助任务，预测训练数据中显示的位置。随后，在反向传播阶段，我们反转传递到主模型的梯度，以确保主模型的预测不依赖于位置特征。

#### 5.3.3 在线实验结果
表2显示了我们提出的方法和基线方法的在线实验结果。我们看到，我们提出的方法通过建模和减少位置偏差显著提高了参与度指标。

**表2：YouTube 上位置偏差建模的在线实验结果**

| 方法 | 参与度指标 |
|---|---|
| 输入特征 | -0.07% |
| 对抗损失 | +0.01% |
| 浅层塔 | +0.24% |

#### 5.3.4 学习到的位置偏差
图7显示了每个位置的学习到的位置偏差。从图中我们看到，对于较低的位置，学习到的偏差较小。学习到的偏差使用有偏隐式反馈估计倾向分数。通过使用足够的训练数据进行模型训练，我们能够有效地学习减少位置偏差。

### 5.4 讨论

在本节中，我们讨论从开发和实验排序系统的过程中获得的一些见解和局限性。

#### 5.4.1 推荐与排序的神经网络模型架构
许多推荐系统研究论文 [18, 43] 扩展了最初为传统机器学习应用设计的模型架构，如用于自然语言处理的多头注意力和用于计算机视觉的 CNN。然而，我们发现这些适用于特定领域表示学习的模型架构中有许多并不直接适用于我们的需求。原因是：

- **多模态特征空间。** 我们的排序系统依赖多种来源的特征，如来自查询和item的内容特征以及上下文特征。这些特征跨度从稀疏类别空间到自然语言和图像等。从混合特征空间中学习具有挑战性。
- **可扩展性与多个排序目标。** 许多模型架构旨在捕获一种类型的信息，如特征交叉 [39] 或序列信息 [35]。它们通常改善一个排序目标但可能损害其他目标。此外，在我们的系统中应用复杂模型架构的组合难以扩展。
- **噪声大且局部稀疏的训练数据。** 我们的系统需要为item和查询训练嵌入向量。然而，我们的大多数稀疏特征遵循幂律分布，并且在用户反馈上具有高方差。例如，在系统无法捕获的略微不同的上下文中，用户可能点击也可能不点击相同的查询推荐item。这给优化尾部item的嵌入空间带来了很大困难。
- **使用小批量随机梯度下降的分布式训练。** 我们依靠具有强大表达能力的大型神经网络模型来找出特征关联。由于我们的模型消耗大量训练数据，我们必须使用分布式训练，这本身就带来了内在的挑战。

#### 5.4.2 效果与效率的权衡
对于真实世界的排序系统，效率不仅影响服务成本，还影响用户体验。一个过度复杂的模型会显著增加生成推荐item的延迟，降低用户满意度和在线指标。因此，我们通常倾向于更简单直接地模型架构。

#### 5.4.3 训练数据中的偏差
除了位置偏差，还有许多其他类型的偏差。其中一些偏差可能是未知和不可预测的，例如由于系统在提取训练数据方面的局限性造成的。如何自动学习和捕获训练数据中的已知和未知偏差是一个长期挑战，需要更多研究。

#### 5.4.4 评估挑战
由于我们的排序系统主要使用用户隐式反馈，表明每个预测任务表现如何的离线评估并不一定能转化为在线表现。事实上，我们经常观察到离线指标和在线指标之间的不一致。因此，最好选择一个总体更简单的模型，以便它能更好地泛化到在线表现。

#### 5.4.5 未来方向
除了上述 MMoE 和消除选择偏差外，我们正在以下方向改进我们的排序系统：

- **探索新的多目标排序模型架构，** 以平衡稳定性、可训练性和表达能力。我们已经观察到 MMoE 通过灵活选择要共享的专家来提高多任务排序性能。最近的一些工作在不损害预测性能的情况下进一步改善了模型稳定性 [29]。
- **理解与学习分解。** 为了建模已知和未知的偏差，我们希望探索能够自动从训练数据中识别潜在偏差并学习减少偏差的模型架构和目标。
- **模型压缩。** 受减少服务成本需求的驱动，我们正在探索不同类型的排序和推荐模型压缩技术 [36]。

## 6 结论

在本文中，我们从描述设计和开发工业推荐系统（尤其是排序系统）时的一些实际挑战入手。这些挑战包括存在多个相互竞争的排序目标，以及用户反馈中的隐式选择偏差。为了应对这些挑战，我们提出了一个大规模多目标排序系统，并将其应用于推荐接下来要观看的视频的问题。为了高效地优化多个排序目标，我们扩展了多门混合专家（MMoE）模型架构以利用软参数共享。我们提出了一种轻量且有效的方法来建模和减少选择偏差，尤其是位置偏差。此外，通过在全球最大的视频共享平台 YouTube 上进行的在线实验，我们证明了我们提出的技术在参与度和满意度指标上都取得了显著改进。

---

## 参考文献

[1] Abien Fred Agarap. 2018. Deep learning using rectified linear units (relu). arXiv preprint arXiv:1803.08375 (2018).

[2] Aman Agarwal, Ivan Zaitsev, Xuanhui Wang, Cheng Li, Marc Najork, and Thorsten Joachims. 2019. Estimating Position Bias without Intrusive Interventions. In Proceedings of the Twelfth ACM International Conference on Web Search and Data Mining. ACM, 474–482.

[3] Deepak Agarwal, Bee-Chung Chen, and Bo Long. 2011. Localized factor models for multi-context recommendation. In Proceedings of the 17th ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 609–617.

[4] Denis Baylor, Eric Breck, Heng-Tze Cheng, Noah Fiedel, Chuan Yu Foo, Zakaria Haque, Salem Haykal, Mustafa Ispir, Vihan Jain, Levent Koc, et al. 2017. Tfx: A tensorflow-based production-scale machine learning platform. In Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. ACM, 1387–1395.

[5] Alex Beutel, Jilin Chen, Zhe Zhao, and Ed H Chi. 2017. Data decisions and theoretical implications when adversarially learning fair representations. arXiv preprint arXiv:1707.00075 (2017).

[6] Christopher Burges, Tal Shaked, Erin Renshaw, Ari Lazier, Matt Deeds, Nicole Hamilton, and Gregory N Hullender. 2005. Learning to rank using gradient descent. In Proceedings of the 22nd International Conference on Machine learning (ICML-05). 89–96.

[7] Rich Caruana. 1997. Multitask learning. Machine learning 28, 1 (1997), 41–75.

[8] Olivier Chapelle and Ya Zhang. 2009. A dynamic bayesian network click model for web search ranking. In Proceedings of the 18th international conference on World wide web. ACM, 1–10.

[9] Heng-Tze Cheng, Levent Koc, Jeremiah Harmsen, Tal Shaked, Tushar Chandra, Hrishi Aradhye, Glen Anderson, Greg Corrado, Wei Chai, Mustafa Ispir, et al. 2016. Wide & deep learning for recommender systems. In Proceedings of the 1st workshop on deep learning for recommender systems. ACM, 7–10.

[10] Paul Covington, Jay Adams, and Emre Sargin. 2016. Deep neural networks for YouTube Recommendations. In Proceedings of the 10th ACM conference on recommender systems. ACM, 191–198.

[11] James Davidson, Benjamin Liebald, Junning Liu, Palash Nandy, Taylor Van Vleet, Ullas Gargi, Sujoy Gupta, Yu He, Mike Lambert, Blake Livingston, et al. 2010. The YouTube video recommendation system. In Proceedings of the fourth ACM conference on Recommender systems. ACM, 293–296.

[12] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2018. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805 (2018).

[13] Humaira Ehsan, Mohamed A Sharaf, and Panos K Chrysanthis. 2016. Muve: Efficient multi-objective view recommendation for visual data exploration. In 2016 IEEE 32nd International Conference on Data Engineering (ICDE). IEEE, 731–742.

[14] Chantat Eksombatchai, Pranav Jindal, Jerry Zitao Liu, Yuchen Liu, Rahul Sharma, Charles Sugnet, Mark Ulrich, and Jure Leskovec. 2018. Pixie: A system for recommending 3+ billion items to 200+ million users in real-time. In Proceedings of the 2018 World Wide Web Conference on World Wide Web. International World Wide Web Conferences Steering Committee, 1775–1784.

[15] Ali Mamdouh Elkahky, Yang Song, and Xiaodong He. 2015. A multi-view deep learning approach for cross domain user modeling in recommendation systems. In Proceedings of the 24th International Conference on World Wide Web. International World Wide Web Conferences Steering Committee, 278–288.

[16] Antonino Freno. 2017. Practical Lessons from Developing a Large-Scale Recommender System at Zalando. In Proceedings of the Eleventh ACM Conference on Recommender Systems. ACM, 251–259.

[17] Florent Garcin, Boi Faltings, Olivier Donatsch, Ayar Alazzawi, Christophe Bruttin, and Amr Huber. 2014. Offline and online evaluation of news recommender systems at swissinfo.ch. In Proceedings of the 8th ACM Conference on Recommender systems. ACM, 169–176.

[18] Qi Gu, Ting Bai, Wayne Xin Zhao, and Ji-Rong Wen. 2018. A Neural Labeled Network Embedding Approach to Product Adopter Prediction. In Asia Information Retrieval Symposium. Springer, 77–89.

[19] Pankaj Gupta, Ashish Goel, Jimmy Lin, Aneesh Sharma, Dong Wang, and Reza Zadeh. 2013. Wtf: The who to follow service at twitter. In Proceedings of the 22nd international conference on World Wide Web. ACM, 505–514.

[20] Xinran He, Junfeng Pan, Ou Jin, Tianbing Xu, Bo Liu, Tao Xu, Yanxin Shi, Antoine Atallah, Ralf Herbrich, Stuart Bowers, et al. 2014. Practical lessons from predicting clicks on ads at facebook. In Proceedings of the Eighth International Workshop on Data Mining for Online Advertising. ACM, 1–9.

[21] Robert A Jacobs, Michael I Jordan, Steven J Nowlan, Geoffrey E Hinton, et al. 1991. Adaptive mixtures of local experts. Neural computation 3, 1 (1991), 79–87.

[22] Thorsten Joachims, Laura Granka, Bing Pan, Helene Hembrooke, Filip Radlinski, and Geri Gay. 2007. Evaluating the accuracy of implicit feedback from clicks and query reformulations in web search. ACM Transactions on Information Systems (TOIS) 25, 2 (2007), 7.

[23] Thorsten Joachims, Adith Swaminathan, and Tobias Schnabel. 2017. Unbiased learning-to-rank with biased feedback. In Proceedings of the Tenth ACM International Conference on Web Search and Data Mining. ACM, 781–789.

[24] Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, and Tie-Yan Liu. 2017. Lightgbm: A highly efficient gradient boosting decision tree. In Advances in Neural Information Processing Systems. 3146–3154.

[25] Walid Krichene, Nicolas Mayoraz, Steffen Rendle, Li Zhang, Xinyang Yi, Lichan Hong, Ed Chi, and John Anderson. 2018. Efficient training on very large corpora via gramian estimation. arXiv preprint arXiv:1807.07187 (2018).

[26] David C Liu, Stephanie Rogers, Raymond Shiau, Dmitry Kislyuk, Kevin C Ma, Zhigang Zhong, Jenny Liu, and Yushi Jing. 2017. Related pins at pinterest: The evolution of a real-world recommender system. In Proceedings of the 26th International Conference on World Wide Web Companion. International World Wide Web Conferences Steering Committee, 583–592.

[27] Mingsheng Long and Jianmin Wang. 2015. Learning multiple tasks with deep relationship networks. arXiv preprint arXiv:1506.02117 2 (2015).

[28] Yichao Lu, Ruihai Dong, and Barry Smyth. 2018. Why I like it: multi-task learning for recommendation and explanation. In Proceedings of the 12th ACM Conference on Recommender Systems. ACM, 4–12.

[29] Jiaqi Ma, Zhe Zhao, Jilin Chen, Ang Li, Lichan Hong, and Ed Chi. 2019. SNR: Sub-Network Routing for Flexible Parameter Sharing in Multi-task Learning. AAAI (2019).

[30] Jiaqi Ma, Zhe Zhao, Xinyang Yi, Jilin Chen, Lichan Hong, and Ed H Chi. 2018. Modeling task relationships in multi-task learning with multi-gate mixture-of-experts. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 1930–1939.

[31] Xia Ning and George Karypis. 2010. Multi-task learning for recommender system. In Proceedings of 2nd Asian Conference on Machine Learning. 269–284.

[32] Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. 2017. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. arXiv preprint arXiv:1701.06538 (2017).

[33] Ayan Sinha, David F Gleich, and Karthik Ramani. 2016. Deconvolving feedback loops in recommender systems. In Advances in Neural Information Processing Systems. 3243–3251.

[34] Adith Swaminathan and Thorsten Joachims. 2015. Batch learning from logged bandit feedback through counterfactual risk minimization. Journal of Machine Learning Research 16, 1 (2015), 1731–1755.

[35] Jiaxi Tang, Francois Belletti, Sagar Jain, Minmin Chen, Alex Beutel, Can Xu, and Ed H Chi. 2019. Towards Neural Mixture Recommender for Long Range Dependent User Sequences. arXiv preprint arXiv:1902.08588 (2019).

[36] Jiaxi Tang and Ke Wang. 2018. Ranking distillation: Learning compact ranking models with high performance for recommender system. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 2289–2298.

[37] Eric Tzeng, Judy Hoffman, Kate Saenko, and Trevor Darrell. 2017. Adversarial discriminative domain adaptation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 7167–7176.

[38] Nan Wang, Hongning Wang, Yiling Jia, and Yue Yin. 2018. Explainable recommendation via multi-task learning in opinionated text data. In The 41st International ACM SIGIR Conference on Research & Development in Information Retrieval. ACM, 165–174.

[39] Ruoxi Wang, Bin Fu, Gang Fu, and Mingliang Wang. 2017. Deep & cross network for ad click predictions. In Proceedings of the ADKDD'17. ACM, 12.

[40] Shanfeng Wang, Maoguo Gong, Haoliang Li, and Junwei Yang. 2016. Multi-objective optimization for long tail recommendation. Knowledge-Based Systems 104 (2016), 145–155.

[41] Xuanhui Wang, Michael Bendersky, Donald Metzler, and Marc Najork. 2016. Learning to rank with selection bias in personal search. In Proceedings of the 39th International ACM SIGIR conference on Research and Development in Information Retrieval. ACM, 115–124.

[42] Andrew Zhai, Dmitry Kislyuk, Yushi Jing, Michael Feng, Eric Tzeng, Jeff Donahue, Yue Li Du, and Trevor Darrell. 2017. Visual discovery at pinterest. In Proceedings of the 26th International Conference on World Wide Web Companion. International World Wide Web Conferences Steering Committee, 515–524.

[43] Shuai Zhang, Lina Yao, Aixin Sun, and Yi Tay. 2019. Deep learning based recommender system: A survey and new perspectives. ACM Computing Surveys (CSUR) 52, 1 (2019), 5.

[44] Xiaojian Zhao, Guangda Li, Meng Wang, Jin Yuan, Zheng-Jun Zha, Zhoujun Li, and Tat-Seng Chua. 2011. Integrating rich information for video recommendation with multi-task rank aggregation. In Proceedings of the 19th ACM international conference on Multimedia. ACM, 1521–1524.

[45] Zhe Zhao, Zhiyuan Cheng, Lichan Hong, and Ed H Chi. 2015. Improving user topic interest profiles by behavior factorization. In Proceedings of the 24th International Conference on World Wide Web. International World Wide Web Conferences Steering Committee, 1406–1416.
