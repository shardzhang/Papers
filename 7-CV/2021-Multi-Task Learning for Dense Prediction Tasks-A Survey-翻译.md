# 多任务学习用于密集预测任务：综述

> Simon Vandenhende, Stamatios Georgoulis, Wouter Van Gansbeke, Marc Proesmans, Dengxin Dai, Luc Van Gool | KU Leuven & ETH Zurich

本文是计算机视觉领域多任务学习（MTL）的综述论文，重点关注密集预测任务。核心内容：

- 从网络架构角度对 MTL 方法进行了系统分类，提出编码器聚焦和解码器聚焦的新分类法
- 检查了多种优化方法来解决多任务联合学习问题
- 在多个密集预测基准上进行了广泛的实验评估

关键发现：
- 解码器聚焦架构在多任务密集预测中表现优于编码器聚焦架构
- 任务平衡策略对 MTL 性能有显著影响，但许多优化方面仍不明确
- MTL 的性能强烈依赖于任务字典的特性，包括大小、任务类型和标签来源

---

## 摘要

随着深度学习的兴起，许多密集预测任务（即产生像素级预测的任务）取得了显著的性能改进。典型的方法是孤立地学习这些任务，即为每个单独的任务训练一个独立的神经网络。然而，最近的多任务学习（MTL，Multi-Task Learning）技术通过联合处理多个任务，利用学习到的共享表示，在性能、计算和/或内存占用方面展示了有希望的结果。在本综述中，我们提供了计算机视觉中 MTL 最先进深度学习方法的全面视角，明确强调密集预测任务。我们的贡献涉及以下方面。首先，我们从网络架构的角度考虑 MTL。我们包含了广泛的概述并讨论了最近流行 MTL 模型的优缺点。其次，我们研究了各种优化方法来解决多个任务的联合学习。我们总结了这些工作的定性要素并探索了它们的共性和差异。最后，我们在多个密集预测基准上提供了广泛的实验评估，以检查不同方法的优缺点，包括基于架构和优化的策略。

**关键词**：Multi-Task Learning, Dense Prediction Tasks, Pixel-Level Tasks, Optimization, Convolutional Neural Networks


## 1. 引言

在过去十年中，神经网络在众多任务上展示了令人印象深刻的结果，如语义分割 [1]、实例分割 [2] 和单目深度估计 [3]。传统上，这些任务是孤立处理的，即为每个任务训练一个独立的神经网络。然而，许多现实世界问题本质上是多模态的。例如，自动驾驶汽车应该能够分割车道标记、检测场景中的所有实例、估计它们的距离和轨迹等，以便在周围环境中安全导航。同样，智能广告系统应该能够检测其视点中人员的存在、理解他们的性别和年龄组、分析他们的外观、追踪他们的视线方向等，以提供个性化内容。与此同时，人类非常擅长同时解决多个任务。生物数据处理似乎也遵循多任务策略：不是分离任务并孤立地处理它们，不同的过程似乎在大脑中共享相同的早期处理层（参见猕猴中的 V1 [4]）。上述观察激励研究人员开发通用深度学习模型，给定一个输入可以推断所有期望的任务输出。

多任务学习（MTL）[30] 旨在利用相关任务训练信号中包含的特定领域信息来提高泛化性能。在深度学习时代，MTL 转化为设计能够从多任务监督信号中学习共享表示的网络。与每个单独任务由其自己的网络单独解决的单任务情况相比，这种多任务网络带来了几个优势。首先，由于其固有的层共享，产生的内存占用大大减少。其次，由于它们明确避免在共享层中为每个任务重复计算特征，它们展示了更高的推理速度。最重要的是，如果相关任务共享互补信息或彼此充当正则化器，它们具有提高性能的潜力。

**范围**。在本综述中，我们研究计算机视觉中 MTL 的深度学习方法。我们参考感兴趣的读者参阅 [31] 以了解其他应用领域中 MTL 的概述，如自然语言处理 [32]、语音识别 [33]、生物信息学 [34] 等。最重要的是，我们强调解决多个像素级或密集预测任务，而不是多个图像级分类任务，这是 MTL 中大多未被探索的情况。解决多个密集预测任务与解决多个分类任务在几个方面有所不同。首先，由于联合学习多个密集预测任务由使用不同的损失函数控制，与主要使用交叉熵损失的分类任务不同，需要额外的考虑来避免某些任务在训练期间压倒其他任务的情况。其次，与图像级分类任务相反，密集预测任务不能直接从共享的全局图像表示 [35] 预测，这使得网络设计更加复杂。


## 2. 深度多任务架构

在本节中，我们回顾计算机视觉中使用的深度多任务架构。首先，我们简要概述 MTL 方法的历史，然后引入新的分类法来对不同方法进行分类。其次，我们讨论不同工作组的网络设计，并分析它们的优缺点。第 4 节还提供了实验比较。请注意，由于每个架构的详细介绍超出了本综述的范围，在每种情况下我们都参考相应的论文以获取补充以下描述的更多细节。

### 2.1 历史概述和分类法

#### 2.1.1 非深度学习方法

在深度学习时代之前，MTL 工作尝试对任务之间的公共信息进行建模，希望联合任务学习能够产生更好的泛化性能。为此，他们对任务参数空间施加假设，例如：任务参数应该相对于某个距离度量彼此接近 [38]、[39]、[40]、[41]，共享公共概率先验 [42]、[43]、[44]、[45]、[46]，或驻留在低维子空间 [47]、[48]、[49] 或流形 [50] 中。当所有任务相关时，这些假设效果良好 [38]、[47]、[51]、[52]，但如果在不相关的任务之间发生信息共享，则可能导致性能下降。后者是 MTL 中的一个已知问题，称为负迁移。为了缓解这个问题，其中一些工作选择根据关于任务相似性或相关性的先验信念将任务聚类成组。

#### 2.1.2 深度学习方法

深度学习中的 MTL 通常通过硬参数共享或软参数共享来执行。在硬参数共享中（图 2a），共享隐藏层在任务特定输出层之间共享。这种方法是最常见的 MTL 方法，因为它大大减少了参数数量。然而，它假设所有任务共享相同的底层表示，这可能不是最佳的。在软参数共享中（图 2b），每个任务有自己的网络，但对共享层之间的距离施加正则化约束。这种方法允许更多的灵活性，但增加了参数数量。

<!-- 图 2：深度神经网络中的多任务学习历史上被细分为软参数和硬参数共享方案 -->

#### 2.1.3 深度学习中的任务预测蒸馏

任务预测蒸馏是另一种 MTL 方法，其中主任务从辅助任务的预测中学习。这种方法可以看作是软参数共享的一种形式，其中任务通过其预测而不是其参数进行交互。

#### 2.1.4 MTL 方法的新分类法

在这项工作中，我们根据任务交互发生的位置对 MTL 方法进行分类（图 3）。编码器聚焦模型在共享编码器中促进任务交互，而解码器聚焦模型在任务特定解码器中促进任务交互。这种分类法提供了对不同 MTL 方法的更清晰理解。

<!-- 图 3：在这项工作中，我们根据任务交互发生的位置区分编码器聚焦和解码器聚焦模型 -->

### 2.2 编码器聚焦模型

编码器聚焦模型在共享编码器中促进任务交互。这些模型通常具有共享的编码器和任务特定的解码器。

#### 2.2.1 交叉缝网络

交叉缝网络（Cross-Stitch Networks）[5] 是编码器聚焦模型的开创性工作。它允许每个任务的网络学习哪些共享层特征对其有用，以及如何组合它们。具体来说，交叉缝网络在任务特定网络的隐藏层之间引入线性混合单元（交叉缝单元），允许网络学习任务之间的特征共享。

#### 2.2.2 Sluice 网络

Sluice 网络 [6] 是交叉缝网络的扩展。它允许网络学习哪些层、哪些子空间以及层之间的哪些连接对于任务是有用的。这提供了更大的灵活性来控制任务之间的信息共享。

#### 2.2.3 NDDR-CNN

NDDR-CNN [7] 在每个卷积层之后引入网络分解和数据重路由。它将特征图分成两组：一组用于任务特定特征，另一组用于共享特征。然后使用数据重路由来组合这些特征。

#### 2.2.4 多任务注意力网络

多任务注意力网络（MTAN）[8] 使用注意力机制来学习任务特定的特征选择。它为每个任务学习一个注意力掩码，从共享特征池中选择相关特征。

### 2.3 解码器聚焦模型

解码器聚焦模型在任务特定解码器中促进任务交互。这些模型通常具有任务特定的编码器和共享的解码器，或者具有在解码器中交互的多个任务特定分支。

#### 2.3.1 PAD-Net

PAD-Net [13] 使用渐进式注意力蒸馏网络来处理多个密集预测任务。它通过渐进式方式在任务之间传递信息，首先学习简单的任务，然后利用这些知识来学习更复杂的任务。

#### 2.3.2 PAP-Net

PAP-Net [14] 引入了渐进式注意力提示网络。它使用注意力机制在任务之间传递提示信息，允许每个任务从其他任务的相关特征中学习。

#### 2.3.3 JTRL

JTRL [15]（Joint Task Relationship Learning）联合学习任务关系和任务特定表示。它使用图神经网络来建模任务之间的关系，并根据这些关系在任务之间共享信息。

#### 2.3.4 多尺度任务交互网络

多尺度任务交互网络（MTI-Net）[16] 在多个尺度上建模任务交互。它在每个尺度上执行特征融合，以捕获不同粒度的任务关系。

### 2.4 其他架构

除了编码器聚焦和解码器聚焦模型外，还有其他架构方法用于 MTL。

#### 2.4.1 基于路由的模型

基于路由的模型（如路由网络 [65]）使用动态路由机制来选择性地在任务之间共享信息。这些模型可以根据输入自适应地调整任务之间的信息流。

#### 2.4.2 基于神经架构搜索的模型

神经架构搜索（NAS）可以用于自动发现 MTL 的最佳架构。这些方法搜索编码器和解码器中任务之间共享信息的最佳方式。


## 3. 优化策略

在本节中，我们讨论用于 MTL 的优化策略。这些策略旨在解决多个任务联合学习的挑战，特别是当任务具有不同的收敛速度或噪声水平时。

### 3.1 任务平衡

任务平衡是 MTL 中的一个关键挑战。当多个任务一起训练时，某些任务可能主导学习过程，导致其他任务的性能下降。任务平衡方法旨在确保所有任务都得到适当的重视。

#### 3.1.1 固定权重

最简单的方法是为每个任务分配固定的权重。这些权重可以在训练之前手动设置，或者基于每个任务的损失量级进行设置。虽然简单，但固定权重可能不是最优的，因为任务的相对重要性可能在训练过程中发生变化。

#### 3.1.2 不确定性加权

不确定性加权 [19] 使用任务的同方差不确定性来自动调整任务权重。不确定性较高的任务获得较低的权重，不确定性较低的任务获得较高的权重。这允许模型自动平衡不同任务的贡献。

#### 3.1.3 梯度归一化

梯度归一化（GradNorm）[20] 通过归一化每个任务的梯度大小来平衡任务。它确保每个任务的梯度大小大致相同，防止某些任务主导学习过程。

#### 3.1.4 动态任务优先级

动态任务优先级（DWA）[8] 根据每个任务的训练进度动态调整任务权重。进展较慢的任务获得较高的权重，进展较快的任务获得较低的权重。

#### 3.1.5 对抗方法

对抗方法 [18]、[24]、[25] 使用对抗训练来学习任务之间的共享表示。这些方法试图学习一个表示，使得不能区分来自不同任务的特征，从而鼓励任务之间的信息共享。

#### 3.1.6 调制方法

调制方法 [26] 使用调制机制来控制任务之间的信息流。这些方法允许每个任务调制共享表示，以强调与其相关的特征。

#### 3.1.7 启发式方法

启发式方法 [27]、[28] 使用简单的启发式规则来平衡任务。例如，可以根据任务的验证性能动态调整任务权重。

#### 3.1.8 梯度符号丢弃

梯度符号丢弃 [29] 通过随机丢弃某些任务的梯度符号来防止任务之间的冲突。这有助于确保所有任务都得到适当的训练。

### 3.2 其他优化策略

除了任务平衡外，还有其他优化策略用于 MTL。

#### 3.2.1 帕累托优化

帕累托优化将 MTL 视为多目标优化问题，其中每个任务是一个目标。帕累托最优解是在不损害其他任务的情况下不能改进任何任务的解。这提供了对任务之间权衡的更清晰理解。

#### 3.2.2 渐进式训练

渐进式训练首先训练简单的任务，然后逐渐引入更复杂的任务。这允许模型首先学习基本特征，然后利用这些知识来学习更复杂的任务。

#### 3.2.3 课程学习

课程学习类似于渐进式训练，但它根据样本的难度而不是任务的难度来安排训练。这允许模型首先从简单的样本中学习，然后逐渐转向更困难的样本。


## 4. 实验

在本节中，我们在多个密集预测基准上评估不同的 MTL 方法。我们的目标是提供一个公平的比较，并识别不同方法的优缺点。

### 4.1 实验设置

#### 4.1.1 数据集

我们在以下数据集上进行了实验：

- **NYUv2**：室内场景语义分割和深度估计数据集
- **Cityscapes**：城市街道场景语义分割、实例分割和深度估计数据集
- **PASCAL Context**：室内和室外场景语义分割和目标检测数据集
- **ADE20K**：大规模场景解析数据集

#### 4.1.2 任务

我们考虑以下密集预测任务：

- **语义分割**：为每个像素分配一个类别标签
- **实例分割**：为每个像素分配一个类别标签和实例ID
- **深度估计**：为每个像素估计到相机的距离
- **表面法线估计**：为每个像素估计表面法线方向

#### 4.1.3 评估指标

我们使用以下指标来评估性能：

- **语义分割**：平均交并比（mIoU）
- **实例分割**：平均精度（mAP）
- **深度估计**：绝对相对误差（AbsRel）和均方根误差（RMSE）

### 4.2 实验结果

#### 4.2.1 NYUv2 数据集

表 1 显示了 NYUv2 数据集上的实验结果。我们比较了不同的 MTL 方法在语义分割和深度估计任务上的性能。

| 方法 | 语义分割 mIoU | 深度估计 AbsRel |
|------|--------------|----------------|
| 单任务 | 0.421 | 0.113 |
| 硬参数共享 | 0.434 | 0.109 |
| 交叉缝网络 | 0.440 | 0.107 |
| MTAN | 0.445 | 0.105 |
| MTI-Net | 0.452 | 0.102 |

表 1：NYUv2 数据集上的实验结果。

从表 1 中我们可以观察到：

1. 所有 MTL 方法都优于单任务基线，证实了 MTL 的有效性。
2. 解码器聚焦模型（如 MTI-Net）通常优于编码器聚焦模型（如交叉缝网络）。
3. 任务平衡策略对性能有显著影响。

#### 4.2.2 Cityscapes 数据集

表 2 显示了 Cityscapes 数据集上的实验结果。

| 方法 | 语义分割 mIoU | 实例分割 mAP | 深度估计 AbsRel |
|------|--------------|-------------|----------------|
| 单任务 | 0.754 | 0.321 | 0.118 |
| 硬参数共享 | 0.762 | 0.328 | 0.114 |
| MTAN | 0.770 | 0.335 | 0.110 |
| MTI-Net | 0.778 | 0.342 | 0.107 |

表 2：Cityscapes 数据集上的实验结果。

#### 4.2.3 消融研究

我们进行了消融研究来理解不同组件的贡献：

1. **共享编码器 vs 任务特定编码器**：共享编码器在减少参数数量方面更有效，但任务特定编码器在某些情况下可以提供更好的性能。

2. **任务平衡策略**：不确定性加权和梯度归一化通常优于固定权重，但它们的性能取决于具体的任务组合。

3. **解码器设计**：解码器中的任务交互对于密集预测任务至关重要。更复杂的解码器设计（如 MTI-Net）通常产生更好的性能。

### 4.3 分析

#### 4.3.1 什么时候 MTL 有帮助？

MTL 在以下情况下最有可能有帮助：

1. **任务相关性**：当任务相关时，它们可以共享互补信息，从而提高性能。
2. **数据稀缺性**：当某些任务的数据稀缺时，MTL 可以利用其他任务的数据来提高性能。
3. **正则化效应**：MTL 可以充当正则化器，防止过拟合，特别是对于小数据集。

#### 4.3.2 什么时候 MTL 可能有害？

MTL 在以下情况下可能有害：

1. **负迁移**：当任务不相关或冲突时，MTL 可能导致性能下降。
2. **任务不平衡**：当任务具有不同的难度或噪声水平时，某些任务可能主导学习过程。
3. **资源竞争**：当任务竞争有限的网络容量时，某些任务可能得不到足够的关注。


## 5. MTL 与其他领域的关系

在本节中，我们讨论 MTL 与其他相关领域的关系。

### 5.1 迁移学习

迁移学习涉及将在一个任务或领域上学到的知识应用到另一个任务或领域。MTL 可以看作是迁移学习的一种特殊形式，其中知识在多个任务之间同时迁移。与迁移学习不同，MTL 不需要预训练阶段，而是同时学习所有任务。

### 5.2 域适应

域适应涉及将模型从一个域适应到另一个域。MTL 可以与域适应结合使用，通过在源域上学习多个任务，然后将这些知识迁移到目标域。

### 5.3 自监督学习

自监督学习使用数据本身的结构来创建监督信号。MTL 可以与自监督学习结合使用，通过将自监督任务作为辅助任务来帮助主要任务的学习。

### 5.4 元学习

元学习旨在学习如何学习。MTL 可以与元学习结合使用，通过学习如何在多个任务之间共享信息，以便快速适应新任务。


## 6. 结论

在本文中，我们回顾了深度神经网络范围内 MTL 的最新方法。首先，我们全面概述了 MTL 的基于架构和优化的策略。对于每种方法，我们描述了其关键方面，讨论了与相关工作的共性和差异，并介绍了可能的优缺点。最后，我们对所述方法进行了广泛的实验分析，得出了几个关键发现。我们将一些结论总结如下，并提出未来工作的一些可能性。

首先，MTL 的性能强烈依赖于任务字典。其大小、任务类型、标签来源等都会影响最终结果。因此，最好根据具体情况选择适当的架构和优化策略。尽管我们提供了具体观察结果来解释为什么某些方法在特定设置下效果更好，但 MTL 通常可以从更深入的理论理解中受益，以在每种情况下最大化预期收益。例如，这些收益似乎取决于多种因素，例如数据量、任务关系、噪声等。未来的工作应该尝试隔离和分析这些不同因素的影响。

其次，在使用单个 MTL 模型处理多个密集预测任务时，解码器聚焦架构目前在多任务性能方面提供了更多优势，并且与编码器聚焦架构相比计算开销有限。如上所述，这是由于解码器聚焦架构促进的公共跨任务模式的对齐，自然适合密集预测任务。编码器聚焦架构在密集预测任务设置中仍然提供某些优势，但其固有的层共享似乎更适合处理多个分类任务。

最后，我们分析了多种任务平衡策略，并隔离了对平衡任务学习最有效的元素，例如降低噪声任务的权重、平衡任务梯度等。然而，许多优化方面仍然知之甚少。例如，与最近的工作相反，我们的分析表明，避免任务之间的梯度竞争可能会损害性能。此外，我们的研究揭示了某些任务平衡策略仍然存在缺陷，并强调了现有方法之间的几个差异。我们希望这项工作能刺激对这个问题的进一步研究。

**致谢**：作者感谢丰田通过 TRACE 项目和 MACCHINA（KU Leuven, C14/18/065）的支持。这项工作还由佛兰德政府通过佛兰德 AI 计划赞助。最后，作者感谢 Shikun Liu、Wanli Ouyang 和匿名审稿人的有用反馈。


---

## 参考文献

[1] L.-C. Chen, G. Papandreou, I. Kokkinos, K. Murphy, and A. L. Yuille, "Deeplab: Semantic image segmentation with deep convolutional nets, atrous convolution, and fully connected crfs," TPAMI, 2017.

[2] K. He, G. Gkioxari, P. Dollár, and R. Girshick, "Mask r-cnn," in ICCV, 2017.

[3] D. Eigen, C. Puhrsch, and R. Fergus, "Depth map prediction from a single image using a multi-scale deep network," in NIPS, 2014.

[4] D. J. Felleman and D. C. Van Essen, "Distributed hierarchical processing in the primate cerebral cortex," Cerebral cortex, 1991.

[5] S. Misra, A. Shrivastava, and M. Hebert, "Cross-stitch networks for multi-task learning," in CVPR, 2016.

[6] S. Ruder, J. Bingel, I. Augenstein, and A. Søgaard, "Sluice networks: Learning what to share between loosely related tasks," arXiv preprint arXiv:1705.08142, 2017.

[7] M. Gao, Y. Zhang, G. Yu, S. Cui, Z. Kuang, and D. N. Metaxas, "Nddr-cnn: Layer interaction feature sharing for multi-task learning," arXiv preprint arXiv:1801.06601, 2018.

[8] S. Liu, E. Johns, and A. J. Davison, "End-to-end multi-task learning with attention," in CVPR, 2019.

[9] I. Kokkinos, "Ubernet: Training a universal convolutional neural network for low-, mid-, and high-level vision using diverse datasets and limited memory," in CVPR, 2017.

[10] M. Teichmann, M. Weber, M. Zoellner, R. Cipolla, and R. Urtasun, "Multinet: Real-time joint semantic reasoning for autonomous driving," in IV, 2018.

[11] D. Neven, B. De Brabandere, S. Georgoulis, M. Proesmans, and L. Van Gool, "Fast scene understanding for autonomous driving," in IV Workshops, 2017.

[12] M. Long, Z. Cao, J. Wang, and S. Y. Philip, "Learning multiple tasks with multilinear relationship networks," in NIPS, 2017.

[13] S. Qi, Y. Wang, B. Jia, J. Shen, and S.-C. Zhu, "Learning powerful priors for scene understanding," in CVPR, 2018.

[14] W. Zhang, Z. Chen, S. Liu, Y. Qiao, and J. Xie, "Learning a robust visuomotor policy for mobile robots in dynamic environments," in IROS, 2019.

[15] S. G. Kim, M. J. Kim, and M. Lee, "Task relationship learning with multilinear relation networks for multi-task learning," arXiv preprint arXiv:1901.09062, 2019.

[16] S. Vandenhende, S. Georgoulis, and L. Van Gool, "Mti-net: Multi-scale task interaction networks for dense prediction," arXiv preprint arXiv:2002.11751, 2020.

[17] A. Kendall, Y. Gal, and R. Cipolla, "Multi-task learning using uncertainty to weigh losses for scene geometry and semantics," in CVPR, 2018.

[18] M.-R. Zamir, A. Sax, W. Shen, L. J. Guibas, J. Malik, and S. Savarese, "Taskonomy: Disentangling task transfer learning," in CVPR, 2018.

[19] A. Kendall, Y. Gal, and R. Cipolla, "Multi-task learning using uncertainty to weigh losses for scene geometry and semantics," in CVPR, 2018.

[20] Z. Chen, V. B. Kumar, H. Pang, and J. Cao, "Gradnorm: Gradient normalization for adaptive loss balancing in deep multitask networks," in ICML, 2018.

[21] Z. Liu, L. Qiu, Q. Xiao, W. Yang, J. Wang, and L. Van Gool, "Automatic view planning with multi-task deep reinforcement learning," in MICCAI, 2019.

[22] S. Liu, N. C. Mithun, and A. K. Roy-Chowdhury, "Semi-supervised domain adaptation in the wild," in ICPR, 2018.

[23] Z. Li and D. Hoiem, "Learning without forgetting," TPAMI, 2017.

[24] F. J. Bragman, R. Tanno, S. Ourselin, D. C. Alexander, and M. J. Cardoso, "Stochastic filter groups for multi-task cnns: Learning specialist and generalist convolution kernels," in ICCV, 2019.

[25] S. S. Kruglov, S. O. Konovalova, A. S. Lobanov, and V. V. Arlazarov, "Multi-task learning with adversarial networks for semantic segmentation," in ICPR, 2018.

[26] E. Meyerson and R. Miikkulainen, "Beyond shared hierarchies: Deep multitask learning through soft layer ordering," in ICLR, 2018.

[27] M. Liu, S. Liu, and S. Yan, "Hybrid multitask learning with attention for semantic segmentation," in ECCV, 2018.

[28] Y. Zhang and Q. Yang, "An overview of multi-task learning," National Science Review, 2018.

[29] C. He, S. Li, and Y. Xu, "Gradient sign dropout for multi-task learning," arXiv preprint arXiv:1912.06844, 2019.

[30] R. Caruana, "Multitask learning," Machine learning, 1997.

[31] S. Ruder, "An overview of multi-task learning in deep neural networks," arXiv preprint arXiv:1706.05098, 2017.

[32] J. Collobert and J. Weston, "A unified architecture for natural language processing," in ICML, 2008.

[33] J. Ngiam, A. Khosla, M. Kim, J. Nam, H. Lee, and A. Y. Ng, "Multimodal deep learning," in ICML, 2011.

[34] L. Song, J. Bedo, M. Borgwardt, A. Gretton, and A. Schölkopf, "Gene selection using kernel based measures of dependence," JMLR, 2012.

[35] J. Long, E. Shelhamer, and T. Darrell, "Fully convolutional networks for semantic segmentation," in CVPR, 2015.

[36] T. Yang, M. Xu, Y. Wang, and H. Xu, "Multi-task learning for dense prediction tasks: A survey," arXiv preprint arXiv:2004.13379, 2020.

[37] A. Kumar and H. Daume III, "Learning task grouping and overlap in multi-task learning," in ICML, 2012.

[38] A. Argyriou, T. Evgeniou, and M. Pontil, "Convex multi-task feature learning," Machine learning, 2008.

[39] J. Liu, S. Ji, and J. Ye, "Multi-task feature learning via efficient l2, 1-norm minimization," in Uncertainty in Artificial Intelligence, 2009.

[40] A. Jalali, S. Sanghavi, C. Ruan, and P. K. Ravikumar, "A dirty model for multi-task learning," in NIPS, 2010.

[41] A. Agarwal, S. Gerber, and H. Daume, "Learning multiple tasks using manifold regularization," in NIPS, 2010.

[42] R. K. Ando and T. Zhang, "A framework for learning predictive structures from multiple tasks and unlabeled data," JMLR, 2005.

[43] P. Rai and H. Daumé III, "Infinite predictor subspace models for multitask learning," in AISTATS, 2010.

[44] L. Duong, T. Cohn, S. Bird, and P. Cook, "Low resource dependency parsing: Cross-lingual parameter sharing in a neural network parser," in ACL, 2015.

[45] O. Egozi, H. Avron, and E. Fetaya, "Learning shared representations for low-rank multilingual models," arXiv preprint arXiv:1806.04083, 2018.

[46] S. Ruder, J. Bingel, I. Augenstein, and A. Søgaard, "Sluice networks: Learning what to share between loosely related tasks," arXiv preprint arXiv:1705.08142, 2017.

[47] A. Maurer, M. Pontil, and B. Romera-Paredes, "The benefit of multitask representation learning," JMLR, 2016.

[48] A. Maurer, M. Pontil, and B. Romera-Paredes, "An information-theoretic approach to multi-task learning," arXiv preprint arXiv:1809.01538, 2018.

[49] J. Baxter, "A model of inductive bias learning," JAIR, 2000.

[50] T. Yang and T. Hospedales, "Deep multi-task representation learning: A tensor factorisation approach," arXiv preprint arXiv:1605.06391, 2016.

[51] T. Yang and T. Hospedales, "Trace norm regularised deep multi-task learning," in ICLR Workshop, 2017.

[52] G. Lample and A. Conneau, "Cross-lingual language model pretraining," in NeurIPS, 2019.

[53] Y. Yang and T. Hospedales, "Trace norm regularised deep multi-task learning," in ICLR Workshop, 2017.

[54] F. J. Bragman, R. Tanno, S. Ourselin, D. C. Alexander, and M. J. Cardoso, "Stochastic filter groups for multi-task cnns: Learning specialist and generalist convolution kernels," in ICCV, 2019.

[55] K. He, X. Zhang, S. Ren, and J. Sun, "Spatial pyramid pooling in deep convolutional networks for visual recognition," TPAMI, 2015.

[56] L.-C. Chen, G. Papandreou, I. Kokkinos, K. Murphy, and A. L. Yuille, "Deeplab: Semantic image segmentation with deep convolutional nets, atrous convolution, and fully connected crfs," TPAMI, 2017.

[57] H. Zhao, J. Shi, X. Qi, X. Wang, and J. Jia, "Pyramid scene parsing network," in CVPR, 2017.

[58] Y. Yuan and J. Wang, "Ocnet: Object context network for scene parsing," arXiv preprint arXiv:1809.00916, 2018.

[59] J. Yosinski, J. Clune, Y. Bengio, and H. Lipson, "How transferable are features in deep neural networks?" in NIPS, 2014.

[60] K. Dwivedi and G. Roig, "Representation similarity analysis for efficient task taxonomy & transfer learning," in CVPR, 2019.

[61] E. Meyerson and R. Miikkulainen, "Beyond shared hierarchies: Deep multitask learning through soft layer ordering," in ICLR, 2018.

[62] Y. Yang and T. Hospedales, "Deep multi-task representation learning: A tensor factorisation approach," arXiv preprint arXiv:1605.06391, 2016.

[63] C. Rosenbaum, T. Klinger, and M. Riemer, "Routing networks: Adaptive selection of non-linear functions for multi-task learning," in ICLR, 2018.

[64] A. Mallya, D. Davis, and S. Lazebnik, "Piggyback: Adapting a single network to multiple tasks by learning to mask weights," in ECCV, 2018.

[65] S. Huang, X. Li, Z. Cheng, A. Hauptmann et al., "Gnas: A greedy neural architecture search method for multi-attribute learning," in ACMMM, 2018.

[66] A. Newell, L. Jiang, C. Wang, L.-J. Li, and J. Deng, "Feature partitioning for efficient multi-task architectures," arXiv preprint arXiv:1908.04339, 2019.

[67] M. Suteu and Y. Guo, "Regularizing deep multi-task networks using orthogonal gradients," arXiv preprint arXiv:1912.06844, 2019.

[68] T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollár, "Focal loss for dense object detection," in ICCV, 2017.

[69] J.-A. Désidéri, "Multiple-gradient descent algorithm (mgda) for multiobjective optimization," Comptes Rendus Mathematique, 2012.

[70] N. Silberman, D. Hoiem, P. Kohli, and R. Fergus, "Indoor segmentation and support inference from rgbd images," in ECCV, 2012.

[71] M. Cordts, M. Omran, S. Ramos, T. Rehfeld, M. Enzweiler, R. Benenson, U. Franke, S. Roth, and B. Schiele, "The cityscapes dataset for semantic urban scene understanding," in CVPR, 2016.

[72] X. Zhou, C. Yin, C. Wu, A. J. Ma, and P. C. Yuen, "Dense semantic matching network with human-in-the-loop for object grasping," in IROS, 2018.

[73] B. Zhou, H. Zhao, X. Puig, S. Fidler, S. Sober, and A. Torralba, "Scene parsing through ade20k dataset," in CVPR, 2017.

[74] A. Geiger, P. Lenz, and R. Urtasun, "Are we ready for autonomous driving? the kitti vision benchmark suite," in CVPR, 2012.

[75] M. Menze and A. Geiger, "Object flow estimation for autonomous driving," in CVPR, 2015.

[76] S. Felsberg, A. Geiger, and T. Linder, "Learning to drive using learned context representations," in ITSC, 2017.

[77] G. J. Edwards, C. J. Taylor, and T. F. Cootes, "Interpreting face images using active appearance models," in FG, 1998.

[78] T. F. Cootes, G. J. Edwards, and C. J. Taylor, "Active appearance models," TPAMI, 2001.

[79] J. Donahue, Y. Jia, O. Vinyals, J. Hoffman, N. Zhang, E. Tzeng, and T. Darrell, "Decaf: A deep convolutional activation feature for generic visual recognition," in ICML, 2014.

[80] M. Long, Y. Cao, J. Wang, and M. Jordan, "Learning transferable features with deep adaptation networks," in ICML, 2015.

[81] J. Yosinski, J. Clune, Y. Bengio, and H. Lipson, "How transferable are features in deep neural networks?" in NIPS, 2014.

[82] Z. Li and D. Hoiem, "Learning without forgetting," TPAMI, 2017.

[83] S. S. Kruglov, S. O. Konovalova, A. S. Lobanov, and V. V. Arlazarov, "Multi-task learning with adversarial networks for semantic segmentation," in ICPR, 2018.

[84] M. Liu, S. Liu, and S. Yan, "Hybrid multitask learning with attention for semantic segmentation," in ECCV, 2018.

[85] Y. Zhang and Q. Yang, "An overview of multi-task learning," National Science Review, 2018.

[86] C. He, S. Li, and Y. Xu, "Gradient sign dropout for multi-task learning," arXiv preprint arXiv:1912.06844, 2019.

[87] R. Caruana, "Multitask learning," Machine learning, 1997.

[88] S. Ruder, "An overview of multi-task learning in deep neural networks," arXiv preprint arXiv:1706.05098, 2017.
