# SNR：子网络路由——多任务学习中的灵活参数共享

> Jiaqi Ma, Zhe Zhao, Jilin Chen, Ang Li, Lichan Hong, Ed H. Chi | University of Michigan & Google AI & DeepMind

第三十三届AAAI人工智能会议（AAAI-19）

贾淇·马（Jiaqi Ma）*¹，赵喆（Zhe Zhao）²，陈吉林（Jilin Chen）²，李昂（Ang Li）³，洪丽婵（Lichan Hong）²，迟埃德·H（Ed H. Chi）²

¹ 密歇根大学安娜堡分校信息学院
² Google AI
³ DeepMind

¹ jiaqima@umich.edu
²,³ {zhezhao, jilinc, anglili, lichan, edchi}@google.com

---
本文介绍了 SNR：子网络路由——多任务学习中的灵活参数共享。核心内容：


关键发现：




---

## 摘要

机器学习应用（如目标检测和内容推荐）通常需要训练单一模型同时预测多个目标。通过神经网络进行多任务学习近年来变得流行，因为它不仅能在相关任务间提升预测精度，还能通过共享模型架构和底层表示来节省计算成本。后者对于实时大规模机器学习系统至关重要。

然而，经典的多任务神经网络在任务不相关时可能会在精度上显著退化。已有研究（Misra et al. 2016; Yang and Hospedales 2016; Ma et al. 2018）表明，在多任务模型中采用更灵活的架构（无论是手动调优还是类似门控网络的软参数共享结构）有助于提升预测精度。然而，手动调优不可扩展，而先前的软参数共享模型要么灵活性不足，要么计算代价高昂。

在本工作中，我们提出了一种名为**子网络路由（Sub-Network Routing, SNR）**的新框架，以实现更灵活的参数共享，同时保持经典多任务神经网络模型的计算优势。SNR将共享的底层隐藏层模块化为多个子网络层，并通过可学习的潜变量控制子网络之间的连接，从而实现灵活的参数共享。我们在大规模数据集YouTube8M上验证了所提方法的有效性。结果表明，所提方法在保持计算效率的同时提升了多任务模型的精度。

---

## 引言

近年来，基于神经网络的多任务学习（Caruna 1993; Caruana 1998）已成功应用于多种实际场景，如实时目标检测（Girshick 2015; Ren et al. 2015）和在线推荐系统（Bansal, Belanger, and McCallum 2016; Ma et al. 2018）。给定单一输入，这些系统通常同时预测多个目标（或类别），并且在服务时通常有低延迟要求。例如，电影推荐系统可能需要同时预测用户点击电影的概率和用户喜欢观看电影的概率，以便在毫秒级别决定是否推荐该电影。众多的并发任务、低延迟要求以及用户与电影组合的巨大探索空间，使得高效的多任务学习极具吸引力。

图1(a)展示了一个经典且广泛使用的多任务学习模型（Caruana 1998），我们称之为**共享底部（Shared-Bottom, SB）模型**。该模型由若干大的底层共享层（Shared-Bottom）和若干小的任务特定高层组成。与为每个任务单独构建模型相比，SB模型学习了多个相关任务的联合表示，这不仅能够提升模型精度，还能通过共享底层网络层来节省服务时的计算成本。

尽管多任务学习有许多成功的应用案例，但经典的SB模型在任务彼此不相关时存在精度显著退化的问题（Ma et al. 2018）。与单任务模型相比，SB模型在共享的低层中引入了归纳偏置。当任务不相关时，不同任务的归纳偏置会产生冲突，损害模型精度。一个直接的解决方案是同时尝试多任务模型和单任务模型，即对相关任务使用多任务模型，对不相关任务使用单任务模型。另一种解决方案是手动调优网络架构以实现灵活的参数共享，即对高度相关的任务共享更多层，对不太相关的任务共享较少层。事实上，Misra等人（2016）表明，对于两对不同的相关任务，需要不同的多任务架构才能取得良好效果。注意，这两种方案都依赖于对任务相关性的了解。尽管先前有一些研究（Baxter 2000；Ben-David, Gehrke, and Schuller 2002；Ben-David and Schuller 2003），但在实际数据中高效度量任务相关性仍然是一个开放问题。因此，对于上述两种方案，我们都需要通过在任务精度上直接训练和优化来手动调优模型结构。这对于大规模多任务问题通常不具备可扩展性。

解决冲突问题的一种可扩展方法是设计具有灵活参数共享架构的多任务模型，例如门控结构，使其能够适应不同的表示需求和任务相关程度（Misra et al. 2016; Yang and Hospedales 2016; Ma et al. 2018）。然而，先前的方法要么灵活性有限，要么灵活但计算代价高昂（因为引入了过多的子结构）。例如，在灵活性有限方面，Ma等人（2018）将共享层拆分为一层子网络，仅允许有限的共享结构组合。另一方面，在灵活但计算昂贵方面，Misra等人（2016）通过连接单任务模型的内部层来构建多任务模型，使得模型成本与多个单任务模型相当；Yang和Hospedales（2016）使用了计算代价高昂的张量分解技术。

---

## 图1：多任务模型

(a) Shared-Bottom (SB) 模型：传统的多任务神经网络模型。
(b) SNR-Trans（带变换的子网络路由）模型：共享层被拆分为子网络，子网络之间的连接（虚线）是一个变换矩阵乘以一个标量潜变量。
(c) SNR-Aver（带平均的子网络路由）模型：共享层被拆分为子网络，子网络之间的连接（虚线）是以标量潜变量为权重的加权平均。

---

我们提出了一种名为**子网络路由（Sub-Network Routing, SNR）**的新框架，以在保持SB模型计算优势的同时实现更灵活的参数共享。SNR框架将共享的低层模块化为并行的子网络，并学习它们之间的连接。子网络模块化已知能提升多任务模型的可训练性（Ma et al. 2018）。两个子网络之间的连接由一个二元变量控制，我们称之为**编码变量**。通过多层子网络和不同的编码变量，我们可以在多任务模型中建模大量的共享架构。同时，由于相关任务可以利用相同的子网络，SB模型的计算优势得以保持。根据连接方式的不同，我们在SNR框架中设计了两种连接类型，即**SNR-Trans**和**SNR-Aver**。SNR-Trans（如图1(b)所示）使用矩阵变换将低层子网络的嵌入变换到高层子网络；SNR-Aver（如图1(c)所示）将低层子网络的嵌入进行加权平均后送入高层子网络。我们的框架也与神经架构搜索（Neural Architecture Search, NAS）领域（Zoph and Le 2016）相关，因为我们也在搜索最适合当前任务的神经架构。我们进行架构搜索的目的是为了实现多任务学习中灵活的参数共享。因此，我们可以通过上述SNR框架中的编码变量来简单地对架构空间进行编码。通过模块化，我们还能够在架构空间的灵活性和架构搜索的难度之间进行权衡。

下一个问题是如何高效地学习子网络之间的连接。我们将编码变量建模为来自参数化分布的潜随机变量。其分布参数可以通过重参数化技巧（Kingma and Welling 2013; Rezende, Mohamed, and Wierstra 2014; Louizos, Welling, and Kingma 2017）与多任务模型参数一起进行基于梯度的优化训练。在服务时，我们使用从学习到的分布派生的确定性估计器来获得服务时的编码变量。这一方法与近期加速NAS的工作（Pham et al. 2018; Liu, Simonyan, and Yang 2018）有共同见解，这些工作表明，在不同网络架构样本之间复用模型参数，并联合学习架构和模型参数，可以显著加速NAS过程。潜变量方法还带来了额外的优势，即可以施加稀疏性先验或惩罚。我们引入了用于神经网络的L0正则化技术（Louizos, Welling, and Kingma 2017），以在有限的服务模型大小下进一步提升SNR-Trans的精度。

我们在一个大型公开视频数据集YouTube8M（Abu-El-Haija et al. 2016）上评估了我们的方法。该数据集包含610万个YouTube视频ID，每个ID带有来自超过3000个实体词表的（多个）标签。每个视频ID的输入特征是预先计算好的对应视频的视觉和音频特征。3000多个标签类别被组织为24个顶级类别。我们将顶级类别作为任务来构建多任务学习数据集，因此每个任务是一个多标签分类问题。我们的实验表明，SNR-Trans和SNR-Aver均显著优于多个基线多任务模型。通过L0正则化，我们进一步将SNR-Trans的服务模型大小减小了最多11%。

本文的其余部分组织如下：第2节回顾相关工作。第3节详细介绍所提出的方法。第4节在YouTube8M数据集上评估所提出的模型。最后，第5节总结全文。

---

## 相关工作

### 多任务学习中的灵活参数共享

已有几项相关工作致力于改进任务相关性较弱情况下的多任务学习。Duong等人（2015）将多任务模型拆分为两个单任务模型，并在模型参数的差异上添加了L2约束。Cross-Stitch模型（Misra et al. 2016）同样将多任务模型拆分为单任务模型，并通过可学习参数加权拼接不同单任务模型的低层。Yang和Hospedales（2016）使用张量分解模型为每个任务生成隐藏层参数。上述所有方法均未设计为保持经典SB模型的计算优势。Duong等人（2015）的模型专门针对双任务场景设计，无法直接推广到多任务场景。

沿着这一方向最接近的工作是Ma等人（2018）的MMoE模型，该模型将共享的低层拆分为子网络（称为专家），并使用不同的门控网络使不同任务利用不同的子网络。我们的方法将这一思想推广为更灵活的形式。我们的方法还允许子网络之间的稀疏连接，这进一步提升了服务时的计算效率。

### 神经架构搜索

神经架构搜索（Neural Architecture Search, NAS）（Zoph and Le 2016; Zoph et al. 2017; Real et al. 2018; Pham et al. 2018; Liu, Simonyan, and Yang 2018）是一个新兴的方法领域，通过强化学习或进化策略自动为给定任务设计神经架构。我们的方法搜索一种既能缓解任务冲突又能保持服务时计算效率的多任务模型架构，这可以视为NAS的一个特例。我们特别关注参数共享问题，并在提出的SNR框架内拥有一个更简单的架构空间。

最早的NAS方法（Zoph and Le 2016）使用双层循环方法来搜索模型架构：在外层循环中，一个基于RNN的控制器生成模型架构，并通过强化学习以生成架构的精度为奖励进行训练；在内层循环中，生成的架构在目标任务上进行训练。可以想象，这一过程计算代价非常高昂。近期有几项工作（Pham et al. 2018; Liu, Simonyan, and Yang 2018）通过合并双层循环并同时学习架构和模型参数，提出了高效的NAS方法。我们的方法与这些高效NAS方法有相似的见解，因为我们也是同时学习架构和模型参数。然而，由于我们的架构空间相对简单，我们可以将编码变量视为潜随机变量，并使用简单的参数化分布（伯努利分布的连续松弛）作为生成架构的策略，这在训练上更加高效。

---

## 方法

在本节中，我们详细介绍所提出的方法。

### 子网络模块化与路由

我们旨在通过灵活的参数共享来改进多任务学习模型，使得相关性更高的任务共享更多模型参数，相关性较低的任务共享较少模型参数。此前几项工作（Misra et al. 2016; Ma et al. 2018）通过将整个神经网络模型拆分为某种形式的子网络来实现这一目标，允许不同任务利用不同的子网络。Ma等人（2018）表明，这种模块化有利于多任务模型的可训练性。在本工作中，我们进一步扩展了这一思想，将经典SB模型中的每个共享低层都模块化为并行的子网络（见图1(b)和图1(c)）。

基于这种模块化，我们可以通过控制不同层子网络之间的连接路由，在多任务模型中实现不同程度的参数共享。我们将这一框架称为**子网络路由（Sub-Network Routing, SNR）**。通过探索大量的连接模式，我们希望找到一个良好的架构，能够尽可能地共享子网络，从而既能缓解任务冲突，又能保持服务时的计算效率。

子网络之间的路由有多种选择。我们实现了两种自然的类型：第一种类型，我们称之为**SNR-Trans**（如图1(b)所示），是使用一个变换矩阵乘以一个标量编码变量；第二种类型，我们称之为**SNR-Aver**（如图1(c)所示），是以标量编码变量为权重的加权平均。

假设有两层连续的子网络，低层有3个子网络，高层有2个子网络。令u₁, u₂, u₃为低层子网络的输出，v₁, v₂为高层子网络的输入。那么SNR-Trans可以表示为：

$$
[v₁]   [z₁₁W₁₁  z₁₂W₁₂  z₁₃W₁₃] [u₁]
[v₂] = [z₂₁W₂₁  z₂₂W₂₂  z₂₃W₂₃] [u₂]
                               [u₃]
$$

其中Wᵢⱼ是从第j个低层子网络到第i个高层子网络的变换矩阵，z表示编码变量（一组控制连接的二元变量）。

类似地，SNR-Aver可以表示为：

$$
[v₁]   [z₁₁I₁₁  z₁₂I₁₂  z₁₃I₁₃] [u₁]
[v₂] = [z₂₁I₂₁  z₂₂I₂₂  z₂₃I₂₃] [u₂]
                               [u₃]
$$

其中Iᵢⱼ对所有i, j均为单位矩阵。

如果我们使两个模型具有相同数量的模型参数（从而具有相似的模型大小），SNR-Trans在连接处有更多的模型参数，而SNR-Aver在子网络中有更多的模型参数预算。尽管很难从模型表示角度判断两种路由方案的优劣，但我们认为，当在子网络之间应用稀疏连接时，SNR-Trans比SNR-Aver更容易减少模型参数，这有利于模型服务。

### 与手动调优和NAS的联系

手动调优网络架构等价于手动设置编码变量z，其中zᵢⱼ \in {0, 1}。例如，假设v₁, v₂表示两个任务的子网络输出，且只有一层隐藏子网络u₁, u₂, u₃。如果我们将z的所有元素设置为1，则对应模型退化为经典的共享底部模型。如果我们设置z₁₁ = z₂₂ = 1而z的其他所有元素为0，则模型退化为两个小的单任务模型。

如果我们拥有无限的计算资源，手动调优也许能够找到在预测精度和计算效率方面帕累托最优的架构。然而，当多任务模型中有许多任务和许多隐藏层时，编码变量z的搜索空间会变得指数级大：2^|z|，其中|z|表示z中的元素数量。因此，当我们缺乏任务相关性的先验知识时，手动调优可能非常低效。

受相关工作中提到的高效NAS方法的启发，我们转而自动学习在相当灵活的SNR框架内的连接路由。在我们的场景中，我们的目标是高效地探索不同的多任务架构，以实现任务间灵活的参数共享，而非通用的神经架构搜索。而且我们有一个由z编码的相对受限的架构空间。我们提出将编码变量z建模为来自参数化分布的潜随机变量，并同时学习分布参数和模型参数。

### 使用潜变量学习架构

在本节中，我们将使用潜变量学习SNR框架中连接路由的问题形式化。

令f(·; W, z)为一个以权重W和编码变量z为参数的神经网络模型，其中编码变量假设从以\pi为参数的潜策略分布p(z; \pi)中抽取。给定数据集D = {(xᵢ, yᵢ)}ⁿᵢ₌₁，其中xᵢ是样本i的特征向量，yᵢ是包含多个任务标签的对应标签向量，学习编码变量和模型参数的问题可以形式化为以下优化问题：

$$
min  E_{z∼p(z;\pi)} [1/N \Sigmaᵢ L(f(xᵢ; W, z), yᵢ)]
W,\pi
$$

其中L是损失函数。

我们简单地使用伯努利分布作为编码变量z的策略。即对于z中的所有元素zᵢ，zᵢ ∼ Bern(\piᵢ)。从图模型的角度，编码变量也可以被视为潜变量。这种潜变量方法可以同时应用于SNR-Trans和SNR-Aver。

公式1中的目标函数关于分布参数\pi不可微，但梯度可以通过梯度估计器REINFORCE（Williams 1992）进行估计。Louizos, Welling和Kingma（2017）进一步提出了一种松弛方法，使这类目标函数变得平滑，从而可以直接计算分布参数的梯度。我们在模型中采用了这种松弛方法。

该方法（Louizos, Welling, and Kingma 2017）的主要思想是：首先找到一个连续随机变量s ∼ q(s; \phi)，并将编码变量z计算为s的硬Sigmoid函数，即：

$$
z = g(s) = min(1, max(0, s))
$$

于是，公式1变为：

$$
min  E_{s∼q(s;\phi)} [1/N \Sigmaᵢ L(f(xᵢ; W, g(s)), yᵢ)]
W,\pi
$$

使用重参数化技巧（Kingma and Welling 2013; Rezende, Mohamed, and Wierstra 2014; Louizos, Welling, and Kingma 2017），公式2重新表述为：

$$
min  E_{ϵ∼r(ϵ)} [1/N \Sigmaᵢ L(f(xᵢ; W, g(h(\phi, ϵ))), yᵢ)]
W,\pi
$$

其中ϵ是一个噪声随机变量，r(ϵ)是一个无参数噪声分布，h(·, ·)是\phi和ϵ的一个确定性和可微的变换。

在实践中，使用了硬具体分布（hard concrete distribution）（Louizos, Welling, and Kingma 2017），其定义（逐元素）如下：

$$
u ∼ U(0, 1)
s = sigmoid((log(u) − log(1 − u) + log(\alpha)) / \beta)
s̄ = s(ζ − \gamma) + \gamma
z = min(1, max(s̄, 0))
$$

其中u是一个均匀随机变量，log(\alpha)是可学习的分布参数，\beta, \gamma, ζ均为超参数。关于硬具体分布的更多细节可参见Louizos, Welling和Kingma（2017）。

### 在潜变量上应用L0正则化

这种潜变量模型的另一个好处是可以在潜变量上添加先验和正则化。例如，Louizos, Welling和Kingma（2017）提供了一种通过L0正则化学习稀疏潜变量的方法。稀疏结构在我们的场景中也是理想的，因为这使得多任务模型具有计算效率。

潜变量z上的L0正则化可以表示为：

$$
E_{z∼p(z;\pi)}||z||₀ = \Sigmaᵢ p(zᵢ = 1; \piᵢ)
$$

通过将z松弛为连续随机变量s，我们有：

$$
p(zᵢ = 1; \piᵢ) = 1 − Q(sᵢ < 0; \phiᵢ)
$$

其中Q(·; \phiᵢ)是sᵢ的累积分布函数。因此L0正则化变为：

$$
E_{z∼p(z;\pi)}||z||₀ = \Sigmaᵢ (1 − Q(sᵢ < 0; \phiᵢ))
$$

带有L0正则化的完整目标函数为：

$$
E_{ϵ∼r(ϵ)} [1/N \Sigmaᵢ L(f(xᵢ; W, g(h(\phi, ϵ))), yᵢ)] + \lambda \Sigmaⱼ (1 − Q(sⱼ < 0; \phiⱼ))
$$

其中\lambda是一个超参数（更多细节参见实验设置部分）。

### 模型训练与服务的附加细节

整个模型，包括模型参数W和潜变量分布参数log(\alpha)，通过基于随机梯度的优化进行训练。在前向传播的每个小批量中，我们首先采样一组均匀随机变量u，然后计算z以获得网络架构，最后将输入数据馈送到模型中计算损失。关于W和log(\alpha)的梯度通过反向传播计算。可以抽取多个u样本以减少梯度估计的方差，但在实践中每个小批量使用一个u样本就能很好地工作。

在服务时，使用以下估计器（Louizos, Welling, and Kingma 2017）用于z：

$$
ẑ = min(1, max(0, sigmoid(log(\alpha))(ζ − \gamma) + \gamma))
$$

当sigmoid(log(\alphaᵢⱼ))(ζ − \gamma) + \gamma < 0时，ẑᵢⱼ = 0，得到的模型将具有稀疏连接。为了在SNR-Aver模型中减少服务模型参数大小，我们需要从模型中移除至少一个完整的子网络，这意味着我们需要ẑᵢⱼ = 0（对所有i成立）才能消除第j个子网络。然而对于SNR-Trans，任何ẑᵢⱼ = 0都会消除对应的Wᵢⱼ。因此SNR-Trans比SNR-Aver更容易减少模型参数。

---

## 实验

在本节中，我们在一个公开的大规模数据集YouTube8M上进行实验，以评估所提模型的有效性。

### 实验设置

#### YouTube8M数据集

我们使用YouTube8M（Abu-El-Haija et al. 2016）作为基准数据集来评估所提方法的有效性。该数据集包含610万个YouTube视频，每个视频带有来自超过3000个主题实体词表的（多个）标签。主题实体可以进一步分组为24个顶级主题类别。

为了从该数据集构建一个多任务学习问题，我们将每个顶级主题类别视为一个独立的预测任务，因此每个任务是一个多标签分类问题。为确保每个任务的数据量，我们使用了数据量最大的前16个类别。

我们使用原始数据集中提供的训练集作为我们的训练集，并将原始验证集拆分为我们自己的验证集和测试集，因为该数据集来自Kaggle竞赛，原始测试集标签不对外公开。

#### 方法

我们在实验中比较了五种多任务学习模型。由于模型之间的主要区别在于其共享的低层部分，我们将所有模型的任务特定高层部分固定为每个任务一个单层全连接隐藏层，隐藏大小为16。在适用时使用ReLU激活函数。各模型共享低层部分的实现描述如下：

- **SB**：这是经典的Shared-Bottom模型，其中若干低层网络层由所有任务共享，每个任务在共享层之上拥有自己的任务特定高层。
  - 共享部分：两个全连接隐藏层，隐藏层大小作为待调优的超参数。

- **MMoE**（Ma et al. 2018）：该模型将共享的低层拆分为子网络，并使用不同的门控网络使不同任务利用不同的子网络。
  - 共享部分：一个全连接隐藏层后接一个具有8个专家的MMoE层，每个专家为单层全连接子网络。第一个隐藏层的隐藏大小和子网络的隐藏大小均为待调优的超参数。门控网络为线性变换，不含超参数。

- **ML-MMoE**：该模型通过添加多个子网络层来扩展MMoE。从低层子网络到高层子网络的连接也由一些门控网络控制。每个高层子网络可以视为一个内部任务。所有门控网络共享相同的输入，即整个模型的输入。
  - 共享部分：两个连续的MMoE层，每层有8个专家。每个专家为单层全连接子网络。专家网络的隐藏大小为可调超参数。

- **SNR-Trans**：所提出的带变换的子网络路由模型。
  - 共享部分：两个连续的变换层。每个变换矩阵的输出大小是可调的。

- **SNR-Aver**：所提出的带平均的子网络路由模型。
  - 共享部分：两个连续的子网络层，每个子网络为单层全连接网络。每个子网络的隐藏大小是可调的。

#### 评估指标

由于YouTube8M数据集中的每个任务是一个多标签分类问题，我们使用平均精度均值（Mean Average Precision, MAP）作为每个任务预测精度的度量。具体来说，我们使用MAP@10作为指标，因为每个任务中大多数样本的正标签少于10个。为了评估多任务模型的整体精度，我们计算所有任务MAP@10的平均值，称之为Average-MAP@10。由于不同任务具有不同的样本量，我们计算两种类型的Average-MAP@10指标：第一种直接计算所有任务MAP@10分数的均值，称为**Macro Average-MAP@10**；第二种以每个任务的数据样本数为权重计算MAP@10分数的加权平均，称为**Micro Average-MAP@10**。

#### 模型训练与超参数调优

所有模型均使用Adam（Kingma and Welling 2013）进行训练，学习率为可调超参数。批量大小固定为128。在验证集上使用早停法。与模型大小相关的超参数通过网格搜索进行调优，以比较在不同模型大小下的模型精度，所有其他超参数通过随机搜索进行调优。SB中两个共享隐藏层的隐藏大小在{256, 512, 1024, 2048}中进行网格搜索，其他模型中的隐藏大小超参数以近似匹配SB模型大小的方式进行网格搜索。L0正则化参数\lambda会影响服务模型大小，因此我们在{0.001, 0.0001, 0.00001}中进行网格搜索。所有模型的学习率在[0.00001, 0.1]范围内以对数尺度进行随机搜索。L-Act和L-Param模型中使用的硬具体分布的超参数在以下范围内随机搜索：\beta ∼ [0.5, 0.9]，\gamma ∼ [−1, −0.1]，ζ ∼ [1.1, 2]。

对于每个模型大小设置，我们进行500次独立随机搜索试验（超参数与模型大小无关），并选择验证精度最高的前10个模型。然后我们报告这前10个模型测试MAP的平均值和标准误差。

### 实验结果

#### 最优调优模型的精度

我们首先在图2中展示了每种方法的最优调优模型的精度。我们报告了Macro Average-MAP@10和Micro Average-MAP@10上的测试精度。除MMoE与SB在Micro Average-MAP@10上的差异外，每对模型在两个指标上的差异在双样本t检验0.05显著性水平下均显著。两个指标上的相对趋势相同。因此，由于篇幅限制，在剩余结果分析中我们仅展示Macro Average-MAP@10的结果。

如图2所示，SNR-Trans和SNR-Aver优于所有基线模型。ML-MMoE模型的表现出人意料地差于所有其他模型。一个可能的原因是堆叠的多门控结构可能导致优化困难。此外，在两层专家之间选择门控网络的输入层也存在设计上的困难。由于来自前一层专家的内部输出众多，很难决定哪些内部输出应该被高层门控网络使用。在我们的实现中，我们使用第一层的初始输入特征作为所有门控网络的输入。然而，初始输入特征可能无法提供关于如何在更高层进行路由的足够信息。

---

## 图2：最优调优模型的精度
柱状图显示了按验证性能选择的前10个模型的平均测试性能。误差线表示平均测试性能的标准误差。

---

#### 精度 vs 模型大小

我们在超参数调优过程中注意到，虽然基线模型的精度随着模型大小的增加而趋于饱和，但SNR-Trans和SNR-Aver的精度在我们超参数范围的边界上仍在缓慢上升。

图3显示了不同模型大小的采样模型集上的测试精度。在该图中，我们将每个模型中第一个共享层的总隐藏大小固定为2048，并通过改变第二个共享层的总隐藏大小来获得不同模型大小的模型。注意，x轴是训练时的总模型参数数量，这意味着SNR模型中被L0正则化消除的模型参数也被计入。

这一结果表明，SNR方法能够比基线方法更好地训练更大的模型。这一现象与我们的假设一致，即多任务学习中的模块化可以提升模型的可训练性。

---

## 图3：Macro Average-MAP@10 vs 模型大小
y轴是Macro Average-MAP@10上的测试性能。x轴是训练时的总模型参数数量。每个模型中第一个共享层的总隐藏大小固定为2048，第二个共享层的总隐藏大小变化以得到不同的模型大小。

---

#### 稀疏模型的精度

虽然能够很好地训练大模型至关重要，但在许多大规模在线系统中，我们对服务模型有严格的低延迟要求。因此我们也关注有限服务模型大小的模型。这里我们展示，通过适当的L0正则化参数\lambda，我们可以在特定服务约束下有效减小SNR-Trans模型的服务模型大小。

我们首先观察到，L0正则化参数\lambda的值对SNR-Trans模型学习到的架构有直接影响。当\lambda设置为0.00001时，学习到的编码变量z几乎全为1，这意味着学习到的架构是密集连接的。当\lambda设置为0.0001时，学习到的架构是稀疏的，稀疏模型的服务模型大小显著小于密集模型。在相同的训练模型大小下，稀疏模型的表现通常不如密集模型。这一结果并不意外，因为\lambda控制着更高模型容量和更小服务模型大小之间的权衡，大的\lambda会缩小模型的有效容量。然而，当服务模型大小受限时，稀疏模型优于具有相似服务模型大小的密集模型。

如图4所示，"SNR-Trans-Dense"表示L0正则化参数为0.00001的SNR-Trans模型，而"SNR-Trans-Sparse"表示L0正则化参数为0.0001的SNR-Trans模型。我们将第一层的总隐藏大小固定为1024，并改变第二层的隐藏大小以获得不同的模型大小。我们将第一层隐藏大小设为其他值时的趋势类似，由于篇幅限制省略了相关图表。密集模型的精度随着服务模型大小的减少而迅速下降。而在有限的服务模型大小下，稀疏模型的精度大幅优于密集模型。换句话说，我们可以在保持相同精度的同时将服务模型大小减少最多11%。

---

## 图4：不同模型大小和L0正则化参数下SNR-Trans模型的Macro Average-MAP@10
"SNR-Trans-Dense"表示使用较小L0正则化参数的模型，得到的模型是密集的；"SNR-Trans-Sparse"表示使用较大L0正则化参数的模型，得到的模型是稀疏的。稀疏模型可以在特定服务约束下将服务模型大小减少最多11%。

---

## 图5：表现最好的SNR-Trans模型中各任务的平均子网络利用率
x轴上的任务按样本量从大到小从左到右排序。y轴是相应任务使用的子网络的平均相对比例。

---

#### 子网络利用率分析

为了更好地理解稀疏模型中不同任务如何利用子网络，我们进一步总结了前10个表现最好的稀疏模型中每个任务使用的子网络的平均相对比例，如图5所示。该图表明，子网络的利用率与任务的样本量呈正相关¹。这意味着，当我们在模型上施加更强的L0正则化时，模型将学会将更多容量分配给数据量更大的任务。

¹ YouTube8M的原始数据分布可在 https://research.google.com/youtube8m/ 找到。

---

## 结论

在本工作中，我们提出了多任务学习中的一个灵活参数共享框架——**子网络路由（Sub-Network Routing, SNR）**。SNR能够编码多种类型的多任务模型架构，并允许我们在模型结构上添加广泛的先验知识。我们提出了一种可扩展的多任务架构搜索解决方案，通过使用潜变量对架构编码变量进行建模，并同时学习潜变量和模型参数。我们通过实验证明，所提方法在大规模数据集YouTube8M上优于基线多任务模型。我们通过将L0正则化应用于潜变量，进一步减小了服务模型的大小。

---

## 参考文献

Abu-El-Haija, S.; Kothari, N.; Lee, J.; Natsev, P.; Toderici, G.; Varadarajan, B.; and Vijayanarasimhan, S. 2016. Youtube-8m: A large-scale video classification benchmark. arXiv preprint arXiv:1609.08675.

Bansal, T.; Belanger, D.; and McCallum, A. 2016. Ask the gru: Multi-task learning for deep text recommendations. In Proceedings of the 10th ACM Conference on Recommender Systems, 107–114. ACM.

Baxter, J. 2000. A model of inductive bias learning. Journal of Artificial Intelligence Research 12:149–198.

Ben-David, S., and Schuller, R. 2003. Exploiting task relatedness for multiple task learning. In Learning Theory and Kernel Machines. Springer. 567–580.

Ben-David, S.; Gehrke, J.; and Schuller, R. 2002. A theoretical framework for learning from a pool of disparate data sources. In Proceedings of the eighth ACM SIGKDD international conference on Knowledge discovery and data mining, 443–449. ACM.

Caruana, R. 1998. Multitask learning. In Learning to learn. Springer. 95–133.

Caruna, R. 1993. Multitask learning: A knowledge-based source of inductive bias. In Machine Learning: Proceedings of the Tenth International Conference, 41–48.

Duong, L.; Cohn, T.; Bird, S.; and Cook, P. 2015. Low resource dependency parsing: Cross-lingual parameter sharing in a neural network parser. In ACL (2), 845–850.

Girshick, R. 2015. Fast r-cnn. In Proceedings of the IEEE international conference on computer vision, 1440–1448.

Kingma, D. P., and Welling, M. 2013. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114.

Liu, H.; Simonyan, K.; and Yang, Y. 2018. Darts: Differentiable architecture search. arXiv preprint arXiv:1806.09055.

Louizos, C.; Welling, M.; and Kingma, D. P. 2017. Learning sparse neural networks through l0 regularization. arXiv preprint arXiv:1712.01312.

Ma, J.; Zhao, Z.; Yi, X.; Chen, J.; Hong, L.; and Chi, E. H. 2018. Modeling task relationships in multi-task learning with multi-gate mixture-of-experts. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, 1930–1939. ACM.

Misra, I.; Shrivastava, A.; Gupta, A.; and Hebert, M. 2016. Cross-stitch networks for multi-task learning. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 3994–4003.

Pham, H.; Guan, M. Y.; Zoph, B.; Le, Q. V.; and Dean, J. 2018. Efficient neural architecture search via parameter sharing. arXiv preprint arXiv:1802.03268.

Real, E.; Aggarwal, A.; Huang, Y.; and Le, Q. V. 2018. Regularized evolution for image classifier architecture search. arXiv preprint arXiv:1802.01548.

Ren, S.; He, K.; Girshick, R.; and Sun, J. 2015. Faster r-cnn: Towards real-time object detection with region proposal networks. In Advances in neural information processing systems, 91–99.

Rezende, D. J.; Mohamed, S.; and Wierstra, D. 2014. Stochastic backpropagation and approximate inference in deep generative models. arXiv preprint arXiv:1401.4082.

Williams, R. J. 1992. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine learning 8(3-4):229–256.

Yang, Y., and Hospedales, T. 2016. Deep multi-task representation learning: A tensor factorisation approach. arXiv preprint arXiv:1605.06391.

Zoph, B., and Le, Q. V. 2016. Neural architecture search with reinforcement learning. arXiv preprint arXiv:1611.01578.

Zoph, B.; Vasudevan, V.; Shlens, J.; and Le, Q. V. 2017. Learning transferable architectures for scalable image recognition. arXiv preprint arXiv:1707.07012 2(6).
