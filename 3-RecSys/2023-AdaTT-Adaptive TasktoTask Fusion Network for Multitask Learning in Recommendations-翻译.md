# AdaTT：面向推荐中多任务学习的自适应任务间融合网络

> Danwei Li∗, Zhengyu Zhang, Siyang Yuan, Mingze Gao, Weilin Zhang, Chaofei Yang, Xi Liu, Jiyan Yang | Meta AI, Meta Platforms, Inc.
>
> KDD '23: The 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, August 6-10, 2023, Long Beach, CA, USA
>
> ∗ 通讯作者：lidli@meta.com。

本文介绍了AdaTT：面向推荐中多任务学习的自适应任务间融合网络。核心内容：

- 提出自适应任务间融合网络（AdaTT），一个深度融合网络，由多层级上的任务特定融合单元和可选的共享融合单元构成，通过残差机制和门控机制自适应地学习共享知识和任务特定知识
- AdaTT在任务对和全任务层级上显式建模任务间交互，分别使用独立的融合模块建模任务特定学习和共享学习，并通过残差机制组合
- 在公开基准数据集和工业推荐数据集上的实验表明，AdaTT显著优于现有的最先进基线模型

关键发现：

- AdaTT在不同数据集和任务组中始终优于MMoE、PLE等最先进的多任务学习模型
- 消融研究验证了残差设计（结合独立建模的融合模块）在实现互补的任务特定学习和共享学习方面的有效性
- 专家权重可视化揭示了在不同融合层级、任务和任务组之间学习到的独特且有意义的共享模式

^1 代码可在 https://github.com/facebookresearch/AdaTT 获取。

---

## 摘要

多任务学习（MTL）旨在通过同时在多个任务上训练机器学习模型来提升其性能和效率。然而，MTL研究面临两个挑战：1）有效建模任务之间的关系以实现知识共享；2）联合学习任务特定知识和共享知识。在本文中，我们提出了一种称为自适应任务间融合网络（AdaTT）^1的新模型来应对这两个挑战。AdaTT是一个深度融合网络，由多层级上的任务特定融合单元和可选的共享融合单元构成。通过利用残差机制和门控机制进行任务间融合，这些单元自适应地学习共享知识和任务特定知识。为了评估AdaTT的性能，我们在一个公开基准数据集和一个工业推荐数据集上使用不同的任务组进行了实验。结果表明AdaTT显著优于现有的最先进基线模型。此外，我们的端到端实验表明，该模型相比替代方案展现出更好的性能。

**CCS概念**

• 计算方法 $\rightarrow$ 多任务学习。
• 信息系统 $\rightarrow$ 推荐系统。

**关键词**

Multi-Task Learning; Neural Networks; Recommender Systems

**ACM引用格式：**

Danwei Li, Zhengyu Zhang, Siyang Yuan, Mingze Gao, Weilin Zhang, Chaofei Yang, Xi Liu, and Jiyan Yang. 2023. AdaTT: Adaptive Task-to-Task Fusion Network for Multitask Learning in Recommendations. In Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '23), August 6-10, 2023, Long Beach, CA, USA. ACM, New York, NY, USA, 10 pages. https://doi.org/10.1145/3580305.3599769

## 1 引言

在线推荐系统旨在为用户生成个性化的高质量推荐。这些系统的有效性通常取决于它们准确学习用户偏好的能力，而这通常需要同时优化多个目标。例如，一个短视频推荐系统应同时考虑用户观看视频的可能性和用户点赞视频的可能性。多任务学习（MTL）是此类用例的典型解决方案。通过在单个框架内联合训练多个任务，MTL提供了若干优势。首先，它提高了计算效率，这对大规模在线推荐系统至关重要。此外，它通过跨任务正则化和知识共享提升了模型性能。

然而，MTL也带来了独特的挑战。主要挑战之一是建模任务之间的关系。由于每个任务与其他任务的相关程度可能不同，仅仅建模所有任务共享的通用共性是不够的。这一问题的复杂性随着任务数量的增加而增加。有效的任务关系建模是实现高效的任务自适应知识共享的关键。例如，任务"分享视频"所共享的知识可以大量侧重于"点赞视频"等相似任务，同时也可以从其他具有丰富样本的任务（如"观看视频"）中汲取不同方面的知识。另一方面，它会最小化与高度不相关任务的共享学习。先前的工作[2, 19]通常采用静态共享表示。其他工作如交叉缝合网络[24]，如图2(c)所示，学习矩阵来建模多个子网络之间的关系。然而，权重对所有样本保持固定，且子网络仅是松散的任务特定。最近的方法如MMoE（Multi-gate Mixture-of-Experts，多门控专家混合）[22]（如图2(b)所示）和PLE（Progressive Layered Extraction，渐进分层提取）[29]（如图2(e)所示）使用专门的门控网络动态组合共享子模块以实现灵活共享，但这些方法建模的任务间关系模糊且间接。

除了共享学习，任务特定学习也是多任务学习中不可或缺的部分。在两者之间取得正确的平衡对于解决任务冲突和实现跨任务正则化非常重要。一方面，MTL可能遭受负迁移，即一个任务的优化对另一个任务的性能产生负面影响，特别是当任务具有冲突目标时。在这种情况下，MTL模型应自适应地强调任务特定学习。另一方面，过度的任务特定学习和不足的共享可能导致过拟合，削弱跨任务正则化的好处。每个任务的训练数据数量和分布也会影响学习的侧重点：数据较多的任务可以更多地依赖其特定学习，而数据较少或数据高度偏斜的任务可以更多地集中于共享学习。考虑样本差异会使两者之间的权衡更加动态。因此，自动学习平衡这两类学习非常重要。许多软参数共享模型可以在无需繁琐的手动调优[2]或使用简化假设为所有样本学习静态结构[23, 28, 30]的情况下实现这一点。然而，需要进一步研究以理解如何对共享学习和任务特定学习之间的交互进行建模以提升性能。

为了共同应对这些挑战，我们提出了一种新的MTL（Multi-task Learning，多任务学习）模型——自适应任务间融合网络（AdaTT，Adaptive Task-to-Task Fusion Network）。为了改进共享学习和可解释性，我们提出引入任务特定专家、共享专家和门控模块，以在任务对和全任务层级上显式建模任务间交互。为了实现协同的任务特定学习和共享学习，我们在独立的融合模块中对它们进行区分和建模，每个模块应用不同的专家和融合策略。融合结果随后通过残差机制[12]进行组合。此外，我们采用多层级融合，每个层级专门负责不同的功能，以增强学习性能。

为了评估AdaTT的性能，我们在一个真实的短视频推荐系统上进行了实验。我们改变实验组以检验其在不同任务关系下的适应性。此外，我们使用一个公开基准数据集来进一步证明其泛化能力。在所有这些实验中，AdaTT在不同数据集和任务组中始终优于基线模型。

为了评估AdaTT在大规模场景下的性能，我们对其超参数进行了研究，特别关注融合层级和专家的数量。此外，我们设计了消融研究和可视化分析，以深入了解AdaTT的内部机制。消融研究验证了残差设计（结合独立建模的融合模块）在实现互补的任务特定学习和共享学习方面的有效性。深层和浅层融合层级上的专家权重可视化使我们能够更深入地理解在不同融合层级、任务和任务组之间学习到的独特且有意义的共享模式。

总之，本文的贡献如下：

*   我们提出了一种新的MTL模型——自适应任务间融合网络（AdaTT），该模型同时实现了自适应的任务间知识共享和鲁棒的任务特定学习。
*   通过在真实世界基准数据和大规模视频推荐系统上的全面实验，我们评估了AdaTT相比各种基线的有效性。
*   我们通过对单个融合模块进行消融研究并探究其融合单元在浅层和深层知识上的操作方式，展示了模型的可解释性。

## 2 相关工作

多任务学习在多个领域有广泛应用，包括计算机视觉[16, 19, 24, 34]、自然语言处理[5, 11]、语音识别[6]、机器人学[32]和推荐系统[10, 22, 29, 35]。许多研究专注于开发创新的MTL架构。这些模型可分为两类：硬参数共享和软参数共享。硬参数共享涉及使用预定义的模型架构，其中某些层在所有任务之间共享，而其他层则特定于单个任务。共享底部模型[2]是硬参数方法中使用最广泛的模型之一。该模型利用共享的低层进行表示学习，并在其之上设置任务特定层。多线性关系网络[20]通过在对任务特定层的参数施加张量正态先验来改进这一结构。另一个例子是UberNet[16]，它使用图像金字塔方法联合解决不同的低层、中层和高层视觉任务。它使用任务特定层和共享层处理金字塔中的每个分辨率。硬参数共享模型通常结构紧凑，但需要大量人工努力来确定共享内容，且缺乏适应性。此外，在无关或冲突任务上过度共享可能导致负迁移，进而对模型性能产生负面影响。

为了更好地解决这些挑战，许多软参数共享MTL模型被提出。交叉缝合网络[24]和水闸网络[26]使用可训练参数线性组合每个层的输出。然而，它们应用的线性组合是固定的，因此不能完全反映单个样本上的任务关系区分。其他工作提出使用注意力或门控模块（以输入为条件）来动态组合或提取每个任务的知识。例如，MTAN（Multi-Task Attention Network，多任务注意力网络）[19]采用注意力模块生成逐元素掩码，从共享表示中提取任务特定知识。MMoE[22]引入了混合专家模型，并使用门控网络为每个任务动态融合专家。最近，PLE[29]被提出以进一步增强知识共享的灵活性。PLE显式引入任务特定专家和共享专家。此外，PLE提出了渐进式分离路由与门控模块，以选择性地、动态地融合知识。在这些工作中，PLE与我们的工作最为相关。不同的是，我们的工作引入了两种互补的融合模块来分别建模任务特定学习和共享学习。此外，除了显式引入共享模块来学习所有任务的共性之外，我们还利用基于输入的直接任务对融合，以最大化知识共享的灵活性。

神经架构搜索（NAS，Neural Architecture Search）[8, 17, 18, 25, 36]方法已被应用于多任务学习（MTL）以自动学习模型结构。分支多任务网络[30]通过基于亲和度分数对任务进行聚类，并将不相似的任务分配到不同分支来生成树形结构。[9]使用Gumbel-Softmax采样进行分支操作，而非预先计算的亲和度分数，从而实现了端到端训练。软层排序技术[23]指出了MTL模型中传统的固定顺序共享方法的局限性，并提出学习任务特定的缩放参数，以实现每个任务共享层的灵活排序。AdaShare[28]学习一个任务特定的策略，为每个特定任务选择要执行的层。子网络路由（SNR，Sub-Network Routing）[21]将共享层拆分为子网络，并使用潜变量学习它们的连接。NAS方法消除了大量人工工作，提高了MTL模型中共享模式的灵活性。然而，由于对所有可能的模型配置进行穷举搜索在组合上过于复杂，这些方法通常依赖于简化假设（如分支[9, 30]、路由[21]、层排序[23]、层选择[28]等）来限制搜索空间。此外，生成的结构不会针对单个样本进行调整。

除了专注于MTL架构设计的工作之外，另一条研究路线旨在改进MTL优化。基于不确定性的加权[15]基于任务不确定性学习每个任务的权重。GradNorm[3]控制不同任务的梯度幅度以平衡其训练速度。GradDrop[4]概率性地选择一个符号并移除相反符号的梯度。梯度手术（PCGrad）[33]将冲突的任务梯度投影到彼此的法平面上。RotoGrad[14]同时操控任务梯度的幅度和方向以缓解冲突。[27]将多任务学习视为一个多目标优化问题，目标是找到帕累托最优解。[31]引入带有欠参数化小塔的自辅助损失，以平衡帕累托效率和跨任务泛化。虽然这些方法可以带来改进，但仅依赖它们而没有强大的模型架构可能会限制模型性能的上限。

## 3 模型架构

为了联合学习自适应的共享表示并增强任务特定学习，我们提出了一种新模型——自适应任务间融合网络（AdaTT）。AdaTT利用门控和残差机制在多个融合层级上自适应地融合专家。考虑一个具有两个预测任务的多任务学习场景。我们在图1中使用两个融合层级展示了AdaTT的架构。AdaTT由一个多层级融合网络和任务塔组成。融合网络由任务特定融合单元和可选的共享融合单元构建，而任务塔构建在融合网络之上，并与最终融合层级中的任务特定单元相连。我们的框架是通用的，支持专家模块、任务塔网络、门控模块的灵活选择，以及可配置的专家数量和融合层数。在以下章节中，我们将首先介绍AdaTT的一个特例，称为AdaTT-sp，它仅使用任务特定融合单元（如图1(a)所示）。然后，我们将描述通用的AdaTT设计，如图1(b)所示。

**图1：具有2个融合层级的AdaTT-sp和通用AdaTT。任务A和B的特定模块和共享模块通过颜色区分：黄色代表A，紫色代表B，蓝色代表共享模块。为了说明目的，每个任务特定单元使用2个专家。在通用AdaTT中，我们添加了一个共享融合单元，该单元具有一个专家作为示例。注意，通用AdaTT中的共享模块不是必需的，因此用虚线表示。当没有共享模块时，通用AdaTT退化为AdaTT-sp。**

### 3.1 AdaTT-sp

AdaTT-sp的详细设计如下。给定 $T$ 个任务的输入 $\mathbf{x}$，任务 $t$（$t = 1, 2, \ldots, T$）的预测公式为：

$$
y_t = h_t(f_t^L(\mathbf{x})), \qquad (1)
$$

其中 $L$ 是融合层数，$h_t$ 表示任务 $t$ 的任务塔，$f_t^L$ 表示产生任务 $t$ 在第 $L$ 个融合层级的融合单元输出的函数。这里，$f_t^L(\mathbf{x})$ 通过使用公式2和3从下到上应用融合层来计算：

$$
f_1^0(\mathbf{x}) = f_2^0(\mathbf{x}) = \ldots = f_T^0(\mathbf{x}) = \mathbf{x} \qquad (2)
$$

$$
f_t^l(\mathbf{x}) = \mathrm{FU}_t^l(f_1^{l-1}(\mathbf{x}), f_2^{l-1}(\mathbf{x}), \ldots, f_T^{l-1}(\mathbf{x})), \quad l = 1 \ldots L \qquad (3)
$$

这里，FU表示一个融合单元。

#### 3.1.1 融合单元

下面我们详细说明公式3中引入的 $\mathrm{FU}_t^l$ 的构建。对于任务 $t$，在接收来自前一融合层级的所有输出后，我们首先为该任务构建 $m_t$ 个原生专家，表示为 $\mathbf{E}_{t,i}^l$，使用函数 $e_{t,i}^l(\cdot)$ 和输入 $f_t^{l-1}(\cdot)$。即：

$$
\mathbf{E}_{t,i}^l = e_{t,i}^l(f_t^{l-1}(\mathbf{x})), \qquad (4)
$$

其中 $i = 1, 2, \ldots, m_t$，且 $\mathbf{E}_{t,i}^l \in \mathbb{R}^{1 \times d_l}$。第 $l$ 层的每个专家网络产生一个长度为 $d_l$ 的向量。为方便表示，在第 $l$ 层，我们使用 $\mathbf{E}_t^l$ 和 $\mathbf{E}^l$ 分别表示属于任务 $t$ 的专家和所有任务的所有专家的垂直拼接。具体来说，$\mathbf{E}_t^l$ 和 $\mathbf{E}^l$ 表示为：

$$
\mathbf{E}_t^l = [\mathbf{E}_{t,1}^l, \mathbf{E}_{t,2}^l, \ldots, \mathbf{E}_{t,m_t}^l], \qquad (5)
$$

$$
\mathbf{E}^l = [\mathbf{E}_1^l, \mathbf{E}_2^l, \ldots, \mathbf{E}_T^l], \qquad (6)
$$

其中 $\mathbf{E}_t^l \in \mathbb{R}^{m_t \times d_l}$ 且 $\mathbf{E}^l \in \mathbb{R}^{(m_1 + m_2 + \ldots + m_T) \times d_l}$。在上述等式中，$[\cdot]$ 表示将向量或子矩阵垂直堆叠成更大矩阵的操作。

由于一个任务可能与其他任务具有不同程度的相关性，$\mathrm{FU}_t^l$ 使用门控模块 $\mathrm{AllExpertGF}_t^l$ 直接建模任务间知识融合，该模块组合所有任务的专家 $\mathbf{E}^l$。此外，我们利用一个轻量级的线性组合 $\mathrm{NativeExpertLF}_t^l$ 来融合任务 $t$ 的原生专家 $\mathbf{E}_t^l$。概念上，门控模块建模共享学习，而原生专家的线性组合建模任务特定学习。具体地，任务 $t$ 在第 $l$ 层的特定单元的输出公式为：

$$
f_t^l(\mathbf{x}) = \mathrm{AllExpertGF}_t^l(\mathbf{E}^l, \mathbf{G}_t^l) + \mathrm{NativeExpertLF}_t^l(\mathbf{E}_t^l), \qquad (7)
$$

其中门控权重 $\mathbf{G}_t^l$ 用于组合 $\mathbf{E}^l$。相对于公式3中的表示，$\mathbf{E}^l$ 依赖于所有 $f_1^{l-1}(\mathbf{x}), f_2^{l-1}(\mathbf{x}), \ldots, f_T^{l-1}(\mathbf{x})$，而 $\mathbf{G}_t^l$ 和 $\mathbf{E}_t^l$ 仅依赖于 $f_t^{l-1}(\mathbf{x})$。具体来说，在公式7中，专家的融合方式如下：

$$
\mathrm{NativeExpertLF}_t^l(\mathbf{E}_t^l) = (\mathbf{v}_t^l)^{\top} \mathbf{E}_t^l, \qquad (8)
$$

$$
\mathrm{AllExpertGF}_t^l(\mathbf{E}^l, \mathbf{G}_t^l) = (\mathbf{G}_t^l)^{\top} \mathbf{E}^l, \qquad (9)
$$

其中 $\mathbf{E}^l$ 通过门控 $\mathbf{G}_t^l$ 相乘，$\mathbf{G}_t^l \in \mathbb{R}^{(m_1 + m_2 + \ldots + m_T) \times 1}$ 由函数 $g_t^l(\cdot)$ 生成，而 $\mathbf{E}_t^l$ 通过一个学习的向量 $\mathbf{v}_t^l \in \mathbb{R}^{m_t \times 1}$ 在 $\mathrm{NativeExpertLF}$ 中简单地组合。当 $m_1 = m_2 = \ldots = m_T = 1$ 时，即所有融合单元只有一个专家，$\mathrm{NativeExpertLF}_t^l(\mathbf{E}_t^l)$ 退化为 $\mathbf{E}_t^l$，即为原生专家分配单位权重以简化。对于 $g_t^l$ 有许多设计选项。一种常见的是使用由softmax激活的单层MLP（Multilayer Perceptron，多层感知机）：

$$
g_t^l(f_t^{l-1}(\mathbf{x})) = \mathrm{softmax}(\mathbf{W}_t^l f_t^{l-1}(\mathbf{x})), \qquad (10)
$$

这里 $\mathbf{W}_t^l \in \mathbb{R}^{(m_1 + m_2 + \ldots + m_T) \times d_{l-1}}$ 是一个学习到的矩阵。

#### 3.1.2 简化

为了实现效率，给定公式8和公式9，我们实际上可以将 $\mathbf{v}_t^l$ 用零填充以匹配 $\mathbf{G}_t^l$ 的大小，相加权重，然后执行一次乘法来组合所有专家。因此，公式7可以简化为：

$$
f_t^l(\mathbf{x}) = (\mathrm{pad}((\mathbf{v}_t^l)^{\top}) + (\mathbf{G}_t^l)^{\top}) \mathbf{E}^l, \qquad (11)
$$

如我们所见，线性融合模块的引入仅带来极小的计算增加。

ft^l(x) = (pad(v_t^{l⊺}) + G_t^{l⊺}) E^l.                     (11)

如我们所见，线性融合模块的引入仅带来极小的计算增加。

### 3.2 通用AdaTT

在其通用形式中，如图1(b)所示，AdaTT采用了可选的共享融合单元。概念上，任务特定模块对之间的融合建模细粒度的共享，而任务特定与共享模块之间的融合传递适用于所有任务的广泛知识。这实现了高效且灵活的任务间知识共享。通用AdaTT的计算与AdaTT-sp类似，除了在最终融合层级中，共享融合单元不执行任何融合操作，仅产生专家输出供任务特定融合单元处理。

总之，AdaTT显式地学习任务特定知识，并自适应地将其与共享知识融合。这种融合是任务自适应的，因为：1. 门控模块学习关于任务原生专家的残差。2. 每个任务特定单元使用一个以输入为条件的专门门控模块（从第二融合层级开始，该输入是唯一的）来融合专家。通过允许每个任务直接且灵活地从其他任务学习共享知识，AdaTT相比仅依赖共享专家作为媒介的PLE提供了更大的灵活性。此外，AdaTT可以选择仅使用任务特定专家。与PLE将所有选定的专家在一个门控模块中处理不同，AdaTT在每个融合单元内的不同线性融合模块中分别融合原生专家。这种设计增强了每一层融合后任务特定学习的鲁棒性。尽管简单，但我们的实验表明，它优于PLE——PLE对来自不同融合单元的专家应用选择，并使用不同的路由路径来区分这些专家。

**对比PLE：** PLE使用门控模块融合选定的专家。在PLE中，共享单元可以融合同一层级的所有专家，而任务特定单元仅融合其原生专家和共享专家。该模型是与AdaTT最接近的模型。

上述所有模型在图2中进行了比较展示。

**图2：我们实验中使用的MTL模型。在多层级MTL模型中，使用两个融合层级来说明其设计。模块用不同颜色表示：共享模块为蓝色，任务A的特定模块为黄色，任务B的特定模块为紫色。**

## 4 实验

在本节中，我们展示全面的实验结果，以突出我们提出的AdaTT模型的有效性，并提供对其更好的理解。

本节分为四个部分。我们首先在4.1节简要描述基线模型。其次，通过在真实世界工业数据集和公共数据集上的实验，我们评估AdaTT相对于最先进的多任务学习模型的有效性。对于工业数据集，我们使用三组不同的预测任务来检验这些多任务学习模型在不同场景下的性能。结果在4.2节和4.3节中分享。接下来，我们在4.4节和4.5节中介绍单个组件的研究。我们对NativeExpertLF模块进行消融，以验证AdaTT残差设计的重要性，该设计包含独立的模块来融合不同的专家。我们还可视化了每个任务特定单元中学习到的专家权重，以展示AdaTT如何学习任务之间的适当交互，这对于有效的知识共享至关重要。最后，在4.6节中，我们对AdaTT的超参数进行研究，以理解融合层级数和专家数与AdaTT性能之间的关系。

### 4.1 基线模型

我们采用Shared-bottom、MMoE、多层MMoE（原始单层MMoE的扩展）、PLE和交叉缝合网络作为我们的基线。在这些模型中，MMoE、PLE和交叉缝合网络均采用软参数共享技术。

*   **MMoE [22]**：该模型为每个任务学习一个专门的门控模块来融合多个共享专家。给定 $n$ 个专家模块 $e_1, e_2, \ldots, e_n$，任务 $t$ 的任务塔模块 $h_t$ 和门控模块 $g_t$，任务 $t$ 的预测计算为：

$$
y_t = h_t(f_t(\mathbf{x})), \qquad (12)
$$

其中

$$
f_t(\mathbf{x}) = g_t(\mathbf{x}) [e_1(\mathbf{x}), e_2(\mathbf{x}), \ldots, e_n(\mathbf{x})], \qquad (13)
$$

这里，$[\cdot]$ 表示将向量垂直堆叠成矩阵。
*   **多层MMoE (ML-MMoE)**：该模型通过引入多个融合层级扩展了原始的单层MMoE。在ML-MMoE中，高层级的专家使用由不同门控模块融合的低层级专家作为输入。与原始MMoE类似，所有门控模块以相同的原始输入为条件。
*   **交叉缝合 [24]**：该模型引入交叉缝合单元，通过学习到的权重线性组合不同任务的隐藏层。
*   **PLE [29]**：该模型显式引入任务特定和共享专家，并采用渐进式分离路由策略。门控模块用于融合任务特定单元和共享单元中选定的专家。在PLE中，共享单元可以融合同一层级的所有专家，而任务特定单元仅融合其原生专家和共享专家。

### 4.2 大规模短视频推荐评估

在本节中，我们展示在一个短视频推荐系统上的实验结果。该系统显示一个推荐视频列表，这些视频基于来自各种任务的分数进行排序。这些任务大致可以分为两类：参与度任务（考虑用户的显式反馈，如评论视频）和消费任务（反映用户的隐式反馈，如视频观看）。

#### 4.2.1 任务组

我们创建了三组任务，以彻底评估这些模型在不同任务关系下的性能。

*   第一组包括一个参与度任务和一个消费任务，预计它们具有相对较低的相关性。
*   第二组由两个相关性更高的消费任务组成。第一个任务与第一组中的消费任务相同。第二个任务被选择为具有与第一组中参与度任务相当的正事件率。组1和组2均仅由二分类任务组成。
*   在第三组中，我们将任务数量增加到五个，并选择高度多样化的任务。其中，三个是消费任务，两个是参与度任务。其中一个消费任务是回归任务，其余四个任务是二分类任务。在用户情感方面，我们包括一个反映用户不喜欢的任务和四个反映正面事件的任务。其中一个具有极稀疏正事件的参与度任务被用作辅助任务。

在报告所有任务组的结果时，我们首先呈现回归任务（如果存在），然后按正样本率降序呈现分类任务。

#### 4.2.2 实验设置

我们收集了约700亿个样本的数据集用于训练模型，并在约100亿个样本的测试集上测试其性能。在特征处理中，我们将所有稀疏特征转换为稠密嵌入，并与稠密特征拼接。所有任务使用相同的输入。所有模型使用相同的框架进行训练和测试，并采用相同的优化设置，如优化器、学习率和批大小。对于训练，我们使用交叉熵损失用于二分类任务，使用MSE（Mean Squared Error，均方误差）损失用于回归任务。所有任务的损失被求和并以相同的权重进行优化。对于测试，我们使用归一化熵（NE，Normalized Entropy）[13]用于二分类任务，使用MSE用于回归任务。

#### 4.2.3 模型超参数

在我们的实验中，所有模型都有3个由ReLU（Rectified Linear Unit，修正线性单元）激活的隐藏层。对于每组实验，我们进行两个比较。

首先，我们将MMoE、PLE和AdaTT与共享底部模型进行比较。为了公平比较，PLE、ML-MMoE（Multi-level MMoE，多层MMoE）和AdaTT均具有2个融合层级。我们在这两个融合层级分别使用隐藏维度为256和128的单层MLP专家。MMoE使用隐藏维度为[256, 128]的双层MLP专家。我们还对每层融合的专家总数设置了限制。这些模型中的所有门控模块使用带有softmax激活的单层MLP。值得注意的是，门控模块所需的计算量远小于专家模块。尽管两种类型的模块输入维度相同，但门控模块的总输出维度几乎小两个数量级。所有模型的任务塔有一个64单元的单隐藏层。通过这种设置，所有模型的计算量相当，因为任务塔和专家模块主导了计算。在我们的实验中，我们调整了PLE和AdaTT的任务特定专家和共享专家数量，而对于MMoE，我们调整了专家总数。

在一个单独的实验中，我们比较了AdaTT和交叉缝合模型与共享底部模型的性能。AdaTT使用与之前实验相似的超参数，但每个任务使用1个专家且没有共享专家，以便与交叉缝合模型具有可比性。交叉缝合模型有2个交叉缝合单元，并且与AdaTT具有相同的隐藏层。

#### 4.2.4 参与度和消费任务组的实验

对于这组任务，我们呈现每个模型相对于共享底部模型的NE差异，分别在使用100亿、300亿和700亿样本训练后的结果。我们还提供了测试结果。表1和表2分别展示了消费任务和参与度任务的结果。结果表明，AdaTT在两个任务中均优于所有其他模型，不仅收敛速度更快，而且质量更高。在使用100亿样本训练后，两个AdaTT模型已经显示出对两个任务的显著NE改进。对于基线模型，PLE在消费任务上需要更长的时间来收敛。另一方面，交叉缝合模型被AdaTT大幅超越，证明了自适应融合在任务关系建模中的至关重要性。值得注意的是，PLE和AdaTT在参与度任务（正事件较少）上的改进大于消费任务。然而，这一趋势在MMoE和ML-MMoE中并不明显，这突显了任务特定学习的重要性。有趣的是，尽管通过额外的融合操作具有更大的灵活性，ML-MMoE在两个任务上的表现均不如MMoE，表明其在专家融合方面的性能较差。这很可能是由于ML-MMoE设计中缺乏区分性和先验知识。共享专家高度对称，每个门控模块都使用所有共享专家，且没有显式建模的任务特定专家。此外，所有门控模块接收相同的原始输入。融合层级的增加导致更多路径，使得ML-MMoE更难以学习用于预测每个特定任务的不同权重组合。

**表1：消费+参与度任务组中消费任务的性能。**

| 模型 | 100亿样本NE差异 | 300亿样本NE差异 | 700亿样本NE差异 | 测试NE差异 |
|------|----------------|----------------|----------------|-----------|
| Shared-bottom | - | - | - | - |
| MMoE | -0.334% | -0.421% | -0.498% | -0.481% |
| ML-MMoE | -0.307% | -0.400% | -0.480% | -0.463% |
| PLE | -0.162% | -0.385% | -0.482% | -0.448% |
| AdaTT | -0.391% | -0.464% | -0.526% | -0.508% |
| Cross-stitch | -0.024% | -0.133% | -0.166% | -0.140% |
| AdaTT-sp (single task expert) | -0.259% | -0.261% | -0.277% | -0.231% |

**表2：消费+参与度任务组中参与度任务的性能。**

| 模型 | 100亿样本NE差异 | 300亿样本NE差异 | 700亿样本NE差异 | 测试NE差异 |
|------|----------------|----------------|----------------|-----------|
| Shared-bottom | - | - | - | - |
| MMoE | -0.370% | -0.436% | -0.542% | -0.532% |
| ML-MMoE | -0.260% | -0.386% | -0.496% | -0.494% |
| PLE | -0.360% | -0.627% | -0.698% | -0.691% |
| AdaTT | -0.677% | -0.795% | -0.845% | -0.863% |
| Cross-stitch | -0.046% | -0.197% | -0.232% | -0.225% |
| AdaTT-sp (single task expert) | -0.393% | -0.367% | -0.397% | -0.362% |

#### 4.2.5 两个消费任务组的实验

由于MTL模型的性能可能对任务相关性敏感，我们设计了一个实验组来评估它们在两个相关消费任务上的表现，与任务组1（任务间相关性较低）形成对比。结果如表3所示，显示该组中所有模型在两个任务上的改进相对于基线更加相似。这并不令人意外，因为当任务更密切相关时，负迁移不那么严重，两个任务都受益于更高水平的共享知识。即使具有更简单共享机制的MTL模型也能实现良好的性能，导致NE差异不那么显著。然而，AdaTT在所有MTL模型中仍然显示出最佳结果。

**表3：两个消费任务组的性能。**

| 模型 | 任务1 NE差异 | 任务2 NE差异 |
|------|------------|------------|
| Shared-bottom | - | - |
| MMoE | -0.343% | -0.372% |
| ML-MMoE | -0.415% | -0.372% |
| PLE | -0.446% | -0.368% |
| AdaTT | -0.487% | -0.443% |
| Cross-stitch | -0.170% | -0.136% |
| AdaTT-sp (single task expert) | -0.233% | -0.194% |

#### 4.2.6 五个多样化任务的实验

在这组任务中，我们利用5个高度多样化的任务来评估模型处理复杂跨任务关系的能力。我们针对4个主要任务调整模型，结果如表4所示。由于辅助任务具有高噪声水平和不一致的性能，我们未包含该具有稀疏正事件的辅助任务。结果表明，AdaTT在所有主要任务中以显著优势超越了所有比较模型，表明其在处理复杂任务关系方面的优越性。

**表4：5个任务组的模型性能。**

| 模型 | 消费任务1 MSE差异 | 消费任务2 NE差异 | 消费任务3 NE差异 | 参与度任务1 NE差异 |
|------|------------------|----------------|----------------|-----------------|
| Shared-bottom | - | - | - | - |
| MMoE | -0.770% | -0.632% | -0.708% | -1.182% |
| ML-MMoE | -0.697% | -0.608% | -0.685% | -1.013% |
| PLE | -0.697% | -0.599% | -0.698% | -1.221% |
| AdaTT | -0.873% | -0.738% | -0.815% | -1.346% |
| Cross-stitch | -0.520% | -0.454% | -0.486% | -0.818% |
| AdaTT-sp (single task expert) | -0.613% | -0.543% | -0.589% | -0.930% |

### 4.3 公共数据集评估

#### 4.3.1 数据集描述

我们使用从1994年和1995年当前人口调查中提取的 Census Income 数据集[7]。该数据集有40个特征和299,285个样本，包括199,523个训练样本和99,762个测试样本。我们将测试样本按相等比例随机拆分为验证集和测试集。任务是：1）预测收入是否超过50K；2）预测婚姻状况是否为从未结婚；3）预测教育程度是否至少为大学。

#### 4.3.2 模型超参数

本实验采用一个改编自[1]的框架来训练和测试ML-MMoE、PLE和AdaTT。模型结构类似于4.2.3节中的结构，但更改了隐藏维度和专家数量。实验分两组进行，每层融合分别有6个和9个专家。PLE和AdaTT的共享专家数ms经过调优。任务特定专家数计算为6-ms和9-ms。为确保公平性，所有其他超参数在各模型中保持一致。在调优ms后，每个模型使用不同的初始化训练100次，并报告测试集中的平均AUC（Area Under the Curve，曲线下面积）。

#### 4.3.3 结果

结果呈现于表5。AdaTT在所有任务中均优于基线模型。

**表5：UCI Census income数据集3个任务上的性能。我们将PLE、ML-MMoE和AdaTT使用2层融合进行比较。专家和任务塔网络是单层MLP，其隐藏维度已列出。AdaTT-sp配置（仅使用任务特定专家）使AdaTT取得了最优结果。**

| 模型 | 专家总数 | 专家隐藏维度 | 任务塔隐藏维度 | 任务1 AUC | 任务2 AUC | 任务3 AUC |
|------|----------|--------------|---------------|-----------|-----------|-----------|
| ML-MMoE | 6 | 128, 64 | 32 | 0.8729 | 0.9178 | 0.9731 |
| PLE | 6 | 128, 64 | 32 | 0.8683 | 0.9164 | 0.9697 |
| AdaTT | 6 | 128, 64 | 32 | 0.8766 | 0.9202 | 0.9783 |
| ML-MMoE | 9 | 96, 48 | 32 | 0.8688 | 0.9139 | 0.9730 |
| PLE | 9 | 96, 48 | 32 | 0.8645 | 0.9134 | 0.9680 |
| AdaTT | 9 | 96, 48 | 32 | 0.8744 | 0.9174 | 0.9786 |

### 4.4 NativeExpertLF模块的消融研究

在本节中，我们检验融合单元中带有NativeExpertLF模块的残差机制的效果。我们消融掉NativeExpertLF模块，仅利用AllExpertGF模块来组合每个融合单元中所有专家的输出。我们采用类似于4.2.3节的模型结构，并使用每个任务固定3个专家且无共享专家。两个模型均在700亿样本上训练，在100亿样本上测试。结果显示在表6中。

**表6：NativeExpertLF模块的消融研究。每个任务的性能退化都证明了带有分离融合的残差机制的重要性。**

| 模型 | 消费任务1 MSE差异 | 消费任务2 NE差异 | 消费任务3 NE差异 | 参与度任务1 NE差异 |
|------|------------------|----------------|----------------|-----------------|
| AdaTT | - | - | - | - |
| AdaTT（消融后） | +0.158% | +0.107% | +0.107% | +0.222% |

虽然AllExpertGF模块理论上可以学习灵活的专家组合，但我们的实验表明，分别组合原生专家并将AllExpertGF的输出作为残差添加是非常重要的。具体来说，消融NativeExpertLF项将导致所有任务的损失，分类任务的NE增加0.107%-0.222%，回归任务的MSE增加0.158%。

### 4.5 门控模块专家权重分布的可视化

在图3中，我们可视化了在添加来自NativeExpertLF和AllExpertGF模块的权重后专家权重的分布，以探究AdaTT的内部融合机制。为了评估专家利用情况，我们选择了三个任务：两个消费任务和一个参与度任务。具体来说，我们在消费任务中选择一个回归任务，在参与度和消费任务中选择具有最高正事件率的两个分类任务。我们实现了两个融合层级，每个任务一个专家，没有共享专家。专家分别是两个融合层级中隐藏维度为256和128的单层MLP。训练模型后，我们将其应用于测试数据集，计算所有测试样本的平均权重，并为每个融合层级可视化一个3 $\times$ 3的权重矩阵。有一些值得注意的观察：

**图3：双层级AdaTT-sp中每个融合层级学习到的专家权重分布可视化。任务和专家按消费回归任务、消费分类任务和参与度任务的顺序排列。注意，此图显示了来自NativeExpertLF和AllExpertGF模块的权重之和。由于每个任务只有一个原生专家，NativeExpertLF模块为其分配单位权重（映射到图中的对角线网格）。**

首先，在较低的融合层级（层级0），我们的模型能够辨别任务之间的关系。消费任务和参与度任务组之间存在明显的区分。此外，两个消费任务之间存在不对称的共享模式：分类消费任务主要使用专家2，而回归消费任务大致相等地使用专家1和专家2。

在更高的融合层级（层级1），其中监督信号更近且捕获了丰富的语义信息，我们的模型通过跨任务的共享模式展示了软参数共享的优势。虽然原生专家在任务特定学习中起重要作用，但所有专家都被灵活利用，为共享学习做出贡献。在此层级，消费分类任务旨在通过利用参与度分类任务特定的专家3以及消费回归任务特定的专家1来多样化学习。同时，具有较少正信号的参与度任务受益于来自两个消费任务的知识迁移。相比之下，消费回归任务主要依赖其原生专家1和其他消费任务特定的专家。在所有专家中，专家1——在层级0中从专家1和专家2的混合中获得了最多样化的学习——在所有任务中被赋予较高的权重。

总体而言，我们可以看到清晰的专门化，在每项任务、任务分组和融合层级上都学习到了不同的权重分布模式。

### 4.6 超参数研究

我们进行超参数研究以探究专家数量和融合层数的影响。两项研究均使用类似于4.2.6节的5个预测任务，700亿样本用于训练，100亿样本用于测试。在两项研究中，我们均采用AdaTT-sp作为模型。

#### 4.6.1 任务特定专家数量的影响

为了检验任务特定专家数量的影响，为了简单起见，我们使所有任务的任务特定专家数量保持一致，并在1到4之间变化。这些专家使用隐藏维度分别为256和128的单层MLP在两个融合层级上构建。该分析的结果可以在表7中找到。我们可以观察到，随着专家数量的增加，所有任务的性能都有所提升。然而，改进并不一致：在本研究中，当专家数量增加到2时，参与度任务相比消费任务仅表现出较小的NE改进。然而，随着专家数量进一步增加到3和4，趋势发生逆转，参与度任务在指标上展现出更显著的差异。

**表7：AdaTT在不同每任务专家数下的性能。**

| 每任务专家数 | 消费任务1 MSE差异 | 消费任务2 NE差异 | 消费任务3 NE差异 | 参与度任务1 NE差异 |
|-------------|------------------|----------------|----------------|-----------------|
| 1 | - | - | - | - |
| 2 | -0.171% | -0.150% | -0.159% | -0.075% |
| 3 | -0.248% | -0.203% | -0.239% | -0.296% |
| 4 | -0.260% | -0.230% | -0.273% | -0.398% |

#### 4.6.2 融合层级的影响

我们还通过使用每个任务一个专家来检查融合层级的不同配置。我们逐步增加融合层数，并为每个层级使用单层MLP。我们训练了四个模型，其在不同融合层级的每个MLP专家的隐藏维度分别为[256, 128]、[512, 256, 128]、[1024, 512, 256, 128]和[2048, 1024, 512, 256, 128]。对于任务塔，每个模型使用隐藏维度为64的单层MLP。结果呈现于表8。正如预期，增加更多融合层级会带来更大的性能提升。即使融合层数增加到五层，所有任务仍然观察到显著的改进。

**表8：融合层级增加时AdaTT的性能结果。我们在第一列中标注每个融合层级专家的隐藏维度。**

| 专家隐藏维度 | 消费任务1 MSE差异 | 消费任务2 NE差异 | 消费任务3 NE差异 | 参与度任务1 NE差异 |
|-------------|------------------|----------------|----------------|-----------------|
| 256, 128 | - | - | - | - |
| 512, 256, 128 | -0.225% | -0.242% | -0.284% | -0.409% |
| 1024, 512, 256, 128 | -0.503% | -0.448% | -0.522% | -0.619% |
| 2048, 1024, 512, 256, 128 | -0.664% | -0.587% | -0.655% | -0.766% |

## 5 结论

在这项工作中，我们提出了一种名为自适应任务间融合网络（AdaTT）的新MTL模型。通过利用其自适应融合机制，AdaTT有效地建模了复杂的任务关系，并促进了任务特定知识和共享知识的联合学习。通过在具有多样化任务组的真实世界工业数据集以及公共数据集上的综合评估，我们展示了AdaTT的有效性和泛化能力。我们的结果表明，AdaTT以显著优势超越了最先进的多任务学习模型。我们希望看到我们的工作能够惠及多任务学习之外的更广泛应用，其中不同的相关专业模块可以协同学习。

## 参考文献

[1] Raquel Aoki, Frederick Tung, and Gabriel L. Oliveira. 2021. Heterogeneous Multi-task Learning with Expert Diversity. In BIOKDD.

[2] Rich Caruana. 1997. Multitask learning. Machine learning 28, 1 (1997), 41–75.

[3] Zhao Chen, Vijay Badrinarayanan, Chen-Yu Lee, and Andrew Rabinovich. 2018. Gradnorm: Gradient normalization for adaptive loss balancing in deep multitask networks. In International conference on machine learning. PMLR, 794–803.

[4] Zhao Chen, Jiquan Ngiam, Yanping Huang, Thang Luong, Henrik Kretzschmar, Yuning Chai, and Dragomir Anguelov. 2020. Just pick a sign: Optimizing deep multitask models with gradient sign dropout. Advances in Neural Information Processing Systems 33 (2020), 2039–2050.

[5] Ronan Collobert and Jason Weston. 2008. A unified architecture for natural language processing: Deep neural networks with multitask learning. In Proceedings of the 25th international conference on Machine learning. 160–167.

[6] Li Deng, Geoffrey Hinton, and Brian Kingsbury. 2013. New types of deep neural network learning for speech recognition and related applications: An overview. In 2013 IEEE international conference on acoustics, speech and signal processing. IEEE, 8599–8603.

[7] Dheeru Dua and Casey Graff. 2017. UCI Machine Learning Repository. http://archive.ics.uci.edu/ml

[8] Thomas Elsken, Jan-Hendrik Metzen, and Frank Hutter. 2017. Simple and efficient architecture search for convolutional neural networks. arXiv preprint arXiv:1711.04528 (2017).

[9] Pengsheng Guo, Chen-Yu Lee, and Daniel Ulbricht. 2020. Learning to branch for multi-task learning. In International Conference on Machine Learning. PMLR, 3854–3863.

[10] Guy Hadash, Oren Sar Shalom, and Rita Osadchy. 2018. Rank and rate: multi-task learning for recommender systems. In Proceedings of the 12th ACM Conference on Recommender Systems. 451–454.

[11] Kazuma Hashimoto, Caiming Xiong, Yoshimasa Tsuruoka, and Richard Socher. 2016. A joint many-task model: Growing a neural network for multiple nlp tasks. arXiv preprint arXiv:1611.01587 (2016).

[12] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. 2016. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition. 770–778.

[13] Xinran He, Junfeng Pan, Ou Jin, Tianbing Xu, Bo Liu, Tao Xu, Yanxin Shi, Antoine Atallah, Ralf Herbrich, Stuart Bowers, et al. 2014. Practical lessons from predicting clicks on ads at facebook. In Proceedings of the eighth international workshop on data mining for online advertising. 1–9.

[14] Adrián Javaloy and Isabel Valera. 2021. RotoGrad: Gradient Homogenization in Multitask Learning. arXiv preprint arXiv:2103.02631 (2021).

[15] Alex Kendall, Yarin Gal, and Roberto Cipolla. 2018. Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. In Proceedings of the IEEE conference on computer vision and pattern recognition. 7482–7491.

[16] Iasonas Kokkinos. 2017. Ubernet: Training a universal convolutional neural network for low-, mid-, and high-level vision using diverse datasets and limited memory. In Proceedings of the IEEE conference on computer vision and pattern recognition. 6129–6138.

[17] Chenxi Liu, Barret Zoph, Maxim Neumann, Jonathon Shlens, Wei Hua, Li-Jia Li, Li Fei-Fei, Alan Yuille, Jonathan Huang, and Kevin Murphy. 2018. Progressive neural architecture search. In Proceedings of the European conference on computer vision (ECCV). 19–34.

[18] Hanxiao Liu, Karen Simonyan, and Yiming Yang. 2018. Darts: Differentiable architecture search. arXiv preprint arXiv:1806.09055 (2018).

[19] Shikun Liu, Edward Johns, and Andrew J Davison. 2019. End-to-end multi-task learning with attention. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 1871–1880.

[20] Mingsheng Long, Zhangjie Cao, Jianmin Wang, and Philip S Yu. 2017. Learning multiple tasks with multilinear relationship networks. Advances in neural information processing systems 30 (2017).

[21] Jiaqi Ma, Zhe Zhao, Jilin Chen, Ang Li, Lichan Hong, and Ed H Chi. 2019. Snr: Sub-network routing for flexible parameter sharing in multi-task learning. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 33. 216–223.

[22] Jiaqi Ma, Zhe Zhao, Xinyang Yi, Jilin Chen, Lichan Hong, and Ed H Chi. 2018. Modeling task relationships in multi-task learning with multi-gate mixture-of-experts. In Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining. 1930–1939.

[23] Elliot Meyerson and Risto Miikkulainen. 2017. Beyond shared hierarchies: Deep multitask learning through soft layer ordering. arXiv preprint arXiv:1711.00108 (2017).

[24] Ishan Misra, Abhinav Shrivastava, Abhinav Gupta, and Martial Hebert. 2016. Cross-stitch networks for multi-task learning. In Proceedings of the IEEE conference on computer vision and pattern recognition. 3994–4003.

[25] Esteban Real, Alok Aggarwal, Yanping Huang, and Quoc V Le. 2019. Regularized evolution for image classifier architecture search. In Proceedings of the aaai conference on artificial intelligence, Vol. 33. 4780–4789.

[26] Sebastian Ruder, Joachim Bingel, Isabelle Augenstein, and Anders Søgaard. 2019. Latent multi-task architecture learning. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 33. 4822–4829.

[27] Ozan Sener and Vladlen Koltun. 2018. Multi-task learning as multi-objective optimization. Advances in neural information processing systems 31 (2018).

[28] Ximeng Sun, Rameswar Panda, Rogerio Feris, and Kate Saenko. 2020. Adashare: Learning what to share for efficient deep multi-task learning. Advances in Neural Information Processing Systems 33 (2020), 8728–8740.

[29] Hongyan Tang, Junning Liu, Ming Zhao, and Xudong Gong. 2020. Progressive layered extraction (ple): A novel multi-task learning (mtl) model for personalized recommendations. In Fourteenth ACM Conference on Recommender Systems. 269–278.

[30] Simon Vandenhende, Stamatios Georgoulis, Bert De Brabandere, and Luc Van Gool. 2019. Branched multi-task networks: deciding what layers to share. arXiv preprint arXiv:1904.02920 (2019).

[31] Yuyan Wang, Zhe Zhao, Bo Dai, Christopher Fifty, Dong Lin, Lichan Hong, Li Wei, and Ed H Chi. 2022. Can Small Heads Help? Understanding and Improving Multi-Task Generalization. In Proceedings of the ACM Web Conference 2022. 3009–3019.

[32] Christopher Williams, Stefan Klanke, Sethu Vijayakumar, and Kian Chai. 2008. Multi-task gaussian process learning of robot inverse dynamics. Advances in neural information processing systems 21 (2008).

[33] Tianhe Yu, Saurabh Kumar, Abhishek Gupta, Sergey Levine, Karol Hausman, and Chelsea Finn. 2020. Gradient surgery for multi-task learning. Advances in Neural Information Processing Systems 33 (2020), 5824–5836.

[34] Zhanpeng Zhang, Ping Luo, Chen Change Loy, and Xiaoou Tang. 2014. Facial landmark detection by deep multi-task learning. In European conference on computer vision. Springer, 94–108.

[35] Zhe Zhao, Lichan Hong, Li Wei, Jilin Chen, Aniruddh Nath, Shawn Andrews, Aditee Kumthekar, Maheswaran Sathiamoorthy, Xinyang Yi, and Ed Chi. 2019. Recommending what video to watch next: a multitask ranking system. In Proceedings of the 13th ACM Conference on Recommender Systems. 43–51.

[36] Barret Zoph and Quoc V Le. 2016. Neural architecture search with reinforcement learning. arXiv preprint arXiv:1611.01578 (2016).
