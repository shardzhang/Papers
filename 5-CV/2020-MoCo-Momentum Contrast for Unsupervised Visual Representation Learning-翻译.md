# MoCo：用于无监督视觉表示学习的动量对比

> Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, Ross Girshick | Facebook AI Research (FAIR)

本文介绍了 MoCo：用于无监督视觉表示学习的动量对比。核心内容：

关键发现：

---

## 摘要

我们提出了 Momentum Contrast (MoCo) 用于无监督视觉表示学习。从对比学习 [29] 作为字典查找的角度出发，我们构建了一个带有队列和移动平均编码器的动态字典。这使得能够动态构建一个大且一致的字典，从而促进对比无监督学习。在 ImageNet 分类的常见线性协议下，MoCo 提供了具有竞争力的结果。更重要的是，通过 MoCo 学习到的表示可以很好地迁移到下游任务中。在 PASCAL VOC、COCO 和其他数据集的 7 个检测/分割任务中，MoCo 可以超越其有监督预训练对应方法，有时甚至以较大幅度超越。这表明无监督和有监督表示学习之间的差距在许多视觉任务中已基本被弥合。

## 1. 引言

无监督表示学习在自然语言处理中非常成功，例如 GPT [50, 51] 和 BERT [12] 所展示的。但在计算机视觉中，有监督预训练仍然占主导地位，无监督方法普遍落后。原因可能源于它们各自信号空间的差异。语言任务具有离散的信号空间（单词、子词单元等），可以构建用于无监督学习的标记化字典。相比之下，计算机视觉进一步涉及字典构建 [54, 9, 5]，因为原始信号处于连续的高维空间中，并且并非为人类通信而结构化（例如，不像单词那样）。

最近的一些研究 [61, 46, 36, 66, 35, 56, 2] 展示了使用与对比损失 [29] 相关的方法进行无监督视觉表示学习的有前景的结果。尽管动机各不相同，这些方法可以被视为构建动态字典。字典中的"键"（标记）从数据（例如图像或图像块）中采样，并由编码器网络表示。无监督学习训练编码器执行字典查找：编码后的"查询"应与其匹配的键相似，而与其他键不相似。学习被形式化为最小化对比损失 [29]。

从这个角度来看，我们假设构建满足以下条件的字典是可取的：(i) 大且 (ii) 在训练过程中保持一致。直观地说，更大的字典可以更好地采样底层连续的高维视觉空间，而字典中的键应由相同或相似的编码器表示，以便它们与查询的比较是一致的。然而，现有的使用对比损失的方法在这两个方面之一可能受到限制（稍后在上下文中讨论）。

我们提出了 Momentum Contrast (MoCo) 作为一种通过对比损失 [29] 构建用于无监督学习的大且一致字典的方法（图 1）。我们将字典维护为一个数据样本队列：当前 mini-batch 的编码表示入队，最早的出队。队列将字典大小与 mini-batch 大小解耦，使其能够很大。此外，由于字典键来自前面的几个 mini-batch，我们提出了一种缓慢进化的键编码器，实现为基于动量的查询编码器移动平均，以保持一致性。

MoCo 是一种为对比学习构建动态字典的机制，可与各种前置任务一起使用。在本文中，我们遵循一个简单的实例判别任务 [61, 63, 2]：如果查询和键来自同一图像的不同编码视图（例如不同的裁剪），则它们匹配。使用这个前置任务，MoCo 在 ImageNet 数据集 [11] 上的线性分类常见协议下展示出具有竞争力的结果。

无监督学习的一个主要目的是预训练可以迁移到下游任务的表示（即特征）。我们展示了在 7 个与检测或分割相关的下游任务中，MoCo 无监督预训练可以超越其 ImageNet 有监督对应方法，在某些情况下还以不小的幅度超越。在这些实验中，我们探索了在 ImageNet 或十亿级 Instagram 图像集上预训练的 MoCo，证明了 MoCo 可以在更真实、十亿级图像规模且相对未经整理的环境中良好工作。这些结果表明，MoCo 在很大程度上弥合了无监督和有监督表示学习在许多计算机视觉任务中的差距，并且可以在若干应用中作为 ImageNet 有监督预训练的替代方案。

## 2. 相关工作

无监督/自监督¹学习方法通常涉及两个方面：前置任务和损失函数。"前置"一词意味着所解决的任务并非真正的目标，而只是为了学习良好数据表示的真实目的而解决的。损失函数通常可以独立于前置任务进行研究。MoCo 关注损失函数方面。接下来我们讨论与这两个方面相关的研究。

**损失函数。** 定义损失函数的一种常见方法是衡量模型预测与固定目标之间的差异，例如通过 L1 或 L2 损失重建输入像素（如自编码器），或通过交叉熵或基于间隔的损失将输入分类到预定义的类别（例如八个位置 [13]、颜色箱 [64]）中。如下所述，其他替代方案也是可能的。

对比损失 [29] 衡量表示空间中样本对的相似性。在对比损失公式中，目标可以在训练过程中动态变化，并且可以根据网络计算的数据表示来定义，而不是将输入与固定目标匹配 [29]。对比学习是最近几项无监督学习工作的核心 [61, 46, 36, 66, 35, 56, 2]，我们稍后会在上下文中详细阐述（第 3.1 节）。

对抗损失 [24] 衡量概率分布之间的差异。这是一种广泛成功的用于无监督数据生成的技术。表示学习的对抗方法在 [15, 16] 中进行了探索。生成对抗网络与噪声对比估计 (NCE) [28] 之间存在关系（参见 [24]）。

**前置任务。** 已经提出了广泛的前置任务。示例包括在某种损坏下恢复输入，例如去噪自编码器 [58]、上下文自编码器 [48] 或跨通道自编码器（着色）[64, 65]。一些前置任务通过例如单个（"样本"）图像的变换 [17]、图像块排序 [13, 45]、视频中的跟踪 [59] 或目标分割 [47] 或聚类特征 [3, 4] 来形成伪标签。

**对比学习 vs. 前置任务。** 各种前置任务可以基于某种形式的对比损失函数。实例判别方法 [61] 与基于样本的任务 [17] 和 NCE [28] 相关。对比预测编码 (CPC) [46] 中的前置任务是一种上下文自编码 [48] 的形式，而对比多视图编码 (CMC) [56] 中则与着色 [64] 相关。

## 3. 方法

### 3.1. 对比学习作为字典查找

对比学习 [29] 及其最新发展可以被视为训练编码器执行字典查找任务，如下所述。

考虑一个编码后的查询 q 和一组编码后的样本 {$k_0$, $k_1$, $k_2$, ...}，它们是字典的键。假设字典中有一个键（记为 k$_{+}$）与 q 匹配。对比损失 [29] 是一个函数，当 q 与其正键 k$_{+}$ 相似且与所有其他键（视为 q 的负键）不相似时，其值较低。以点积衡量相似度，本文考虑一种称为 InfoNCE [46] 的对比损失函数形式：

L_q = -log( exp(q·k$_{+}$/$\tau$) / $$\Sigma$_i$$_=$$_0$ᴷ exp(q·$k_i$/$\tau$) )   (1)

其中 $\tau$ 是温度超参数（参见 [61]）。求和包括一个正样本和 K 个负样本。直观地说，这个损失是一个 (K+1) 路基于 softmax 的分类器的对数损失，该分类器试图将 q 分类为 k$_{+}$。对比损失函数也可以基于其他形式 [29, 59, 61, 36]，例如基于间隔的损失和 NCE 损失的变体。

对比损失作为训练表示查询和键的编码器网络的无监督目标函数 [29]。通常，查询表示为 q = f_q(x^q)，其中 f_q 是编码器网络，x^q 是查询样本（同样地，k = f_k(x^k)）。它们的实例化取决于具体的前置任务。输入 x^q 和 x^k 可以是图像 [29, 61, 63]、图像块 [46] 或由一组图像块组成的上下文 [46]。网络 f_q 和 f_k 可以是相同的 [29, 59, 63]、部分共享的 [46, 36, 2] 或不同的 [56]。

### 3.2. 动量对比

从上述角度来看，对比学习是一种在图像等高维连续输入上构建离散字典的方式。字典是动态的，因为键是随机采样的，并且键编码器在训练过程中会进化。我们的假设是，可以通过一个覆盖丰富负样本集的大字典来学习好的特征，同时字典键的编码器在其进化过程中尽可能保持一致。基于这一动机，我们提出了 Momentum Contrast，如下所述。

**字典作为队列。** 我们方法的核心是将字典维护为一个数据样本队列。这允许我们重用来自前面几个 mini-batch 的编码键。队列的引入将字典大小与 mini-batch 大小解耦。我们的字典大小可以远大于典型的 mini-batch 大小，并且可以作为超参数灵活独立地设置。

字典中的样本被逐步替换。当前 mini-batch 入队到字典中，队列中最旧的 mini-batch 被移除。字典始终表示所有数据的一个采样子集，而维护这个字典的额外计算是可控的。此外，移除最旧的 mini-batch 可能是有益的，因为其编码的键是最过时的，因此与最新键的一致性最差。

**动量更新。** 使用队列可以使字典变大，但也使得通过反向传播更新键编码器变得棘手（梯度需要传播到队列中的所有样本）。一个朴素的解决方案是从查询编码器 f_q 复制键编码器 f_k，忽略这个梯度。但这个解决方案在实验中产生了较差的结果（第 4.1 节）。我们假设这种失败是由于编码器快速变化降低了键表示的一致性。我们提出了动量更新来解决这个问题。

形式化地，将 f_k 的参数记为 $\theta$_k，f_q 的参数记为 $\theta$_q，我们通过以下方式更新 $\theta$_k：

$\theta$_k $\leftarrow$ m·$\theta$_k + (1-m)·$\theta$_q   (2)

其中 m $$\in$$ [0,1) 是动量系数。只有参数 $\theta$_q 通过反向传播更新。式 (2) 中的动量更新使 $\theta$_k 比 $\theta$_q 进化得更平滑。因此，尽管队列中的键由不同的编码器（在不同的 mini-batch 中）编码，但这些编码器之间的差异可以很小。在实验中，相对较大的动量（例如 m=0.999，我们的默认值）比较小的值（例如 m=0.9）效果好得多，这表明缓慢进化的键编码器是利用队列的核心。

**与先前机制的关系。** MoCo 是使用对比损失的通用机制。我们在图 2 中将其与两种现有的通用机制进行比较。它们在字典大小和一致性方面表现出不同的特性。

通过反向传播进行端到端更新是一种自然的机制（例如 [29, 46, 36, 63, 2, 35]，图 2a）。它使用当前 mini-batch 中的样本作为字典，因此键被一致地编码（由同一组编码器参数）。但字典大小与 mini-batch 大小耦合，受限于 GPU 内存大小。大规模 mini-batch 优化也是一个挑战 [25]。一些近期方法 [46, 36, 2] 基于由局部位置驱动的前置任务，其中字典大小可以通过多个位置变得更大。但这些前置任务可能需要特殊的网络设计，例如对输入进行分块 [46] 或定制感受野大小 [2]，这可能会复杂化这些网络向下游任务的迁移。

另一种机制是 [61] 提出的内存库方法（图 2b）。内存库包含数据集中所有样本的表示。每个 mini-batch 的字典是从内存库中随机采样的，无需反向传播，因此可以支持大字典大小。但是，内存库中样本的表示是在其上次被看到时更新的，因此采样的键本质上来自过去整个 epoch 中多个不同步骤的编码器，因此一致性较低。[61] 在内存库上采用了动量更新。其动量更新是针对同一样本的表示，而不是编码器。这种动量更新与我们的方法无关，因为 MoCo 不跟踪每个样本。此外，我们的方法内存效率更高，可以在十亿级数据上进行训练，这对于内存库来说可能是难以处理的。

第 4 节通过实验比较了这三种机制。

### 3.3. 前置任务

对比学习可以驱动各种前置任务。由于本文的重点不是设计新的前置任务，我们使用一个简单的任务，主要遵循 [61] 中的实例判别任务，[63, 2] 中的一些近期工作也与此相关。

遵循 [61]，我们将查询和键视为正对，如果它们来自同一图像，否则视为负样本对。遵循 [63, 2]，我们对同一图像进行两个随机"视图"（通过随机数据增强）以形成正对。查询和键分别由它们的编码器 f_q 和 f_k 编码。编码器可以是任何卷积神经网络 [39]。

算法 1 提供了 MoCo 对该前置任务的伪代码。

**技术细节。** 我们采用 ResNet [33] 作为编码器，其最后一个全连接层（在全局平均池化之后）具有固定维度的输出（128 维 [61]）。这个输出向量通过其 L2 范数进行归一化 [61]。这就是查询或键的表示。式 (1) 中的温度 $\tau$ 设为 0.07 [61]。数据增强设置遵循 [61]：从随机调整大小的图像中裁剪 224$$\times$$224 像素的图块，然后进行随机颜色抖动、随机水平翻转和随机灰度转换，这些都在 PyTorch 的 torchvision 包中可用。

**Shuffling BN。** 我们的编码器 f_q 和 f_k 都具有标准 ResNet [33] 中的批归一化 (BN) [37]。在实验中，我们发现使用 BN 会阻止模型学习良好的表示，[35] 中也类似地报告了这一点（他们避免使用 BN）。模型似乎"欺骗"了前置任务，轻易地找到了低损失的解决方案。这可能是由于样本之间的批内通信（由 BN 引起）泄露了信息。

我们通过 shuffling BN 来解决这个问题。我们使用多个 GPU 进行训练，并独立地对每个 GPU 上的样本执行 BN（如常见做法）。对于键编码器 f_k，我们在将当前 mini-batch 分配到 GPU 之前打乱其样本顺序（编码后再打乱回来）；查询编码器 f_q 的 mini-batch 样本顺序保持不变。这确保了用于计算查询及其正键的批统计量来自两个不同的子集。这有效地解决了作弊问题，并允许训练从 BN 中受益。

我们在我们的方法及其端到端消融对应方法（图 2a）中都使用了 shuffled BN。这与内存库对应方法（图 2b）无关，后者不存在这个问题，因为正键来自过去不同的 mini-batch。

## 4. 实验

我们研究在以下数据集上执行的无监督训练：
- **ImageNet-1M (IN-1M)：** 这是 ImageNet [11] 训练集，包含约 128 万张图像，1000 个类别（常称为 ImageNet-1K；我们按图像数量计数，因为无监督学习不利用类别）。这个数据集的类别分布均衡，其图像通常包含物体的标志性视图。
- **Instagram-1B (IG-1B)：** 遵循 [44]，这是一个来自 Instagram 的约 10 亿张（9.4 亿）公共图像的数据集。这些图像来自约 1500 个与 ImageNet 类别相关的标签 [44]。与 IN-1M 相比，这个数据集相对未经整理，具有长尾、不平衡的真实世界数据分布。该数据集包含标志性物体和场景级图像。

**训练。** 我们使用 SGD 作为优化器。SGD 权重衰减为 0.0001，SGD 动量为 0.9。对于 IN-1M，我们使用 8 个 GPU，mini-batch 大小为 256（算法 1 中的 N），初始学习率为 0.03。我们训练 200 个 epoch，学习率在第 120 和 160 个 epoch 时乘以 0.1 [61]，训练 ResNet-50 约需 53 小时。对于 IG-1B，我们使用 64 个 GPU，mini-batch 大小为 1024，学习率为 0.12，每 62.5k 次迭代（64M 图像）按 0.9 指数衰减。我们训练 125 万次迭代（约 1.4 个 IG-1B epoch），训练 ResNet-50 约需 6 天。

### 4.1. 线性分类协议

我们首先通过冻结特征上的线性分类来验证我们的方法，遵循常见协议。在本小节中，我们在 IN-1M 上进行无监督预训练。然后冻结特征并训练一个有监督线性分类器（一个全连接层后接 softmax）。我们在 ResNet 的全局平均池化特征上训练这个分类器，共 100 个 epoch。我们报告 ImageNet 验证集上的单裁剪 top-1 分类准确率。

对于这个分类器，我们进行网格搜索并发现最优初始学习率为 30，权重衰减为 0（[56] 中也有类似报告）。这些超参数对本小节中呈现的所有消融条目都表现一致良好。这些超参数值意味着特征分布（例如幅度）可能与 ImageNet 有监督训练的特征分布有实质差异，我们将在第 4.2 节中重新讨论这个问题。

**消融：对比损失机制。** 我们比较了图 2 中说明的三种机制。为了聚焦于对比损失机制的效果，我们按照第 3.3 节所述，在相同的前置任务中实现所有三种机制。我们还使用相同形式的 InfoNCE 作为对比损失函数（式 (1)）。因此，比较纯粹是三种机制之间的比较。

结果如图 3 所示。总体而言，所有三种机制都受益于更大的 K。[61, 56] 在内存库机制下观察到了类似的趋势，而我们在这里表明这个趋势更普遍，可以在所有机制中看到。这些结果支持了我们构建大字典的动机。

端到端机制在 K 较小时表现与 MoCo 相似。然而，由于端到端的要求，字典大小受限于 mini-batch 大小。这里高端机器（8 块 32GB Volta GPU）能承担的最大 mini-batch 是 1024。更本质地说，大 mini-batch 训练是一个开放问题 [25]：我们发现有必要在这里使用线性学习率缩放规则 [25]，否则准确率会下降（1024 mini-batch 时下降约 2%）。但使用更大的 mini-batch 进行优化更困难 [25]，即使内存足够，该趋势是否能外推到更大的 K 也是可疑的。

内存库 [61] 机制可以支持更大的字典大小。但它比 MoCo 差 2.6%。这与我们的假设一致：内存库中的键来自过去整个 epoch 中非常不同的编码器，它们不一致。请注意，内存库 58.0% 的结果反映了我们对 [61] 的改进实现。²

**消融：动量。** 下表显示了在预训练中使用不同 MoCo 动量值（式 (2) 中的 m）时的 ResNet-50 准确率（此处 K=4096）：

| 动量 m | 0 | 0.9 | 0.99 | 0.999 | 0.9999 |
|---------|---|---|------|-------|--------|
| 准确率(%) | 失败 | 55.2 | 57.8 | 59.0 | 58.9 |

当 m 在 0.99~0.9999 范围内时，表现相当好，表明缓慢进化（即相对较大的动量）的键编码器是有益的。当 m 太小（例如 0.9）时，准确率大幅下降；在无动量的极端情况（m=0）下，训练损失振荡且无法收敛。这些结果支持了我们构建一致字典的动机。

**与先前结果的比较。** 先前无监督学习方法在模型大小上可能有很大差异。为了公平和全面的比较，我们报告准确率与参数数量³的权衡。除了 ResNet-50 (R50) [33]，我们还报告了其 2 倍和 4 倍宽度（更多通道）的变体，遵循 [38]⁴。我们设置 K=65536，m=0.999。表 1 是比较结果。

使用 R50 的 MoCo 表现具有竞争力，达到 60.6% 的准确率，优于所有类似模型大小（约 24M）的竞争者。MoCo 受益于更大的模型，使用 R50$$\times$$4 达到了 68.6% 的准确率。

值得注意的是，我们使用标准的 ResNet-50 就取得了具有竞争力的结果，不需要特定的架构设计，例如分块输入 [46, 35]、精心定制的感受野 [2] 或结合两个网络 [56]。通过使用不针对前置任务定制的架构，更容易将特征迁移到各种视觉任务并进行比较，这将在下一小节中研究。

本文的重点是通用对比学习的一种机制；我们不探索可能进一步提高准确率的正交因素（例如特定的前置任务）。例如，"MoCo v2" [8] 是本文初步版本的扩展，通过对数据增强和输出投影头 [7] 进行小幅修改，使用 R50 达到了 71.1% 的准确率（从 60.6% 提升）。我们认为这个额外结果表明了 MoCo 框架的通用性和鲁棒性。

### 4.2. 特征迁移

无监督学习的一个主要目标是学习可迁移的特征。ImageNet 有监督预训练在作为下游任务微调的初始化时最具影响力（例如 [21, 20, 43, 52]）。接下来我们将 MoCo 与 ImageNet 有监督预训练进行比较，迁移到各种任务，包括 PASCAL VOC [18]、COCO [42] 等。作为先决条件，我们讨论涉及的两个重要问题 [31]：归一化和调度。

**归一化。** 如第 4.1 节所述，无监督预训练产生的特征与 ImageNet 有监督预训练的特征可能具有不同的分布。但下游任务的系统通常具有为有监督预训练选择的超参数（例如学习率）。为了缓解这个问题，我们在微调期间采用特征归一化：我们使用训练的（并在 GPU 间同步 [49]）BN 进行微调，而不是通过仿射层冻结它 [33]。我们还在新初始化的层（例如 FPN [41]）中使用 BN，这有助于校准幅度。我们在微调有监督和无监督预训练模型时都进行归一化。MoCo 使用与 ImageNet 有监督对应方法相同的超参数。

**调度。** 如果微调调度足够长，从随机初始化训练检测器可以成为强基线，并且可以在 COCO [31] 上与 ImageNet 有监督对应方法匹敌。我们的目标是研究特征的可迁移性，因此我们的实验采用受控调度，例如 COCO 上的 1$$\times$$（约 12 epoch）或 2$$\times$$ 调度 [22]，而不是 [31] 中的 6$$\times$$~9$$\times$$。在像 VOC 这样较小的数据集上，更长时间的训练可能也无法弥补差距 [31]。

尽管如此，在我们的微调中，MoCo 使用与 ImageNet 有监督对应方法相同的调度，并提供了随机初始化结果作为参考。

总之，我们的微调使用与有监督预训练对应方法相同的设置。这可能使 MoCo 处于劣势。即便如此，MoCo 仍具有竞争力。这样做也使得在多个数据集/任务上进行比较成为可能，而无需额外的超参数搜索。

#### 4.2.1 PASCAL VOC 目标检测

**设置。** 检测器是 Faster R-CNN [52]，骨干网络为 R50-dilated-C5 或 R50-C4 [32]（详见附录），BN 可调，实现基于 [60]。我们端到端地微调所有层。训练时图像尺度为 [480, 800] 像素，推理时为 800。所有条目（包括有监督预训练基线）使用相同的设置。我们评估默认的 VOC 指标 A$P_50$（即 IoU 阈值为 50%）以及更严格的 COCO 风格 AP 和 A$P_75$ 指标。评估在 VOC test2007 集上进行。

**消融：骨干网络。** 表 2 显示了在 trainval07+12（约 16.5k 图像）上微调的结果。对于 R50-dilated-C5（表 2a），在 IN-1M 上预训练的 MoCo 与有监督预训练对应方法相当，而在 IG-1B 上预训练的 MoCo 超越了它。对于 R50-C4（表 2b），使用 IN-1M 或 IG-1B 的 MoCo 优于有监督对应方法：最多提升 +0.9 A$P_50$、+3.7 AP 和 +4.9 A$P_75$。

有趣的是，迁移准确率取决于检测器结构。对于 C4 骨干网络（现有基于 ResNet 的结果 [14, 61, 26, 66] 默认使用），无监督预训练的优势更大。预训练与检测器结构之间的关系在过去一直未被揭示，应该作为一个考虑因素。

**消融：对比损失机制。** 我们指出，这些结果部分是由于我们为对比学习建立了坚实的检测基线。为了指出仅由使用 MoCo 机制带来的增益，我们微调了使用端到端或内存库机制预训练的模型（两者均由我们实现，即图 3 中最好的那些），使用与 MoCo 相同的微调设置。

这些竞争方法表现尚可（表 3）。它们在 C4 骨干网络上的 AP 和 A$P_75$ 也高于 ImageNet 有监督对应方法（参见表 2b），但其他指标较低。它们在所有指标上都比 MoCo 差。这显示了 MoCo 的优势。此外，如何在更大规模数据上训练这些竞争方法是一个开放问题，它们可能无法从 IG-1B 中受益。

**与先前结果的比较。** 跟随竞争方法，我们使用 C4 骨干网络在 trainval2007（约 5k 图像）上微调。比较结果在表 4 中。

对于 A$P_50$ 指标，没有先前方法能够赶上其各自的有监督预训练对应方法。而 MoCo 无论在 IN-1M、IN-14M（完整 ImageNet）、YFCC-100M [55] 还是 IG-1B 上预训练，都能超越有监督基线。在更严格的指标上看到了更大的增益：最高 +5.2 AP 和 +9.0 A$P_75$。这些增益大于在 trainval07+12 上看到的增益（表 2b）。

#### 4.2.2 COCO 目标检测和分割

**设置。** 模型是 Mask R-CNN [32]，骨干网络为 FPN [41] 或 C4，BN 可调，实现基于 [60]。图像尺度在训练时为 [640, 800] 像素，推理时为 800。我们端到端地微调所有层。我们在 train2017 集（约 118k 图像）上微调，并在 val2017 上评估。调度是 [22] 中的默认 1$$\times$$ 或 2$$\times$$。

**结果。** 表 5 显示了 COCO 上使用 FPN（表 5a, b）和 C4（表 5c, d）骨干网络的结果。在 1$$\times$$ 调度下，所有模型（包括 ImageNet 有监督对应方法）都严重训练不足，如与 2$$\times$$ 调度情况相比约 2 个点的差距所示。在 2$$\times$$ 调度下，MoCo 在两种骨干网络的所有指标上都优于其 ImageNet 有监督对应方法。

#### 4.2.3 更多下游任务

表 6 显示了更多下游任务（实现细节见附录）。总体而言，MoCo 与 ImageNet 有监督预训练表现相当：

- **COCO 关键点检测：** 有监督预训练相比随机初始化没有明显优势，而 MoCo 在所有指标上都表现更好。
- **COCO 密集姿态估计 [1]：** MoCo 大幅优于有监督预训练，例如在这个高度定位敏感的任务中 AP$d_p75$ 高出 3.7 个点。
- **LVIS v0.5 实例分割 [27]：** 该任务有约 1000 个长尾分布的类别。具体来说，在 LVIS 上对于 ImageNet 有监督基线，我们发现使用冻结 BN（24.4 APmk）微调比可调 BN 更好（详见附录）。因此我们在此任务中将 MoCo 与更好的有监督预训练变体进行比较。使用 IG-1B 的 MoCo 在所有指标上超越了它。
- **Cityscapes 实例分割 [10]：** 使用 IG-1B 的 MoCo 在其有监督预训练对应方法的 APmk 上持平，在 APm$k_50$ 上更高。
- **语义分割：** 在 Cityscapes [10] 上，MoCo 优于其有监督预训练对应方法最多 0.9 个点。但在 VOC 语义分割上，MoCo 至少差 0.8 个点，这是我们观察到的一个反面案例。

**总结。** 总之，MoCo 可以在 7 个检测或分割任务中超越其 ImageNet 有监督预训练对应方法⁵。此外，MoCo 在 Cityscapes 实例分割上持平，在 VOC 语义分割上落后；我们在附录中展示了另一个在 iNaturalist [57] 上的可比案例。总体而言，MoCo 已经在多个视觉任务中基本弥合了无监督和有监督表示学习之间的差距。

值得注意的是，在所有这些任务中，在 IG-1B 上预训练的 MoCo 一致优于在 IN-1M 上预训练的 MoCo。这表明 MoCo 可以在这种大规模、相对未经整理的数据集上表现良好。这代表了向真实世界无监督学习迈进的一个场景。

## 5. 讨论与结论

我们的方法在多种计算机视觉任务和数据集上展示了无监督学习的积极成果。有几个开放问题值得讨论。MoCo 从 IN-1M 到 IG-1B 的提升虽然一致可见但相对较小，表明更大规模的数据可能未被充分利用。我们希望先进的前置任务能够改善这一点。除了简单的实例判别任务 [61] 之外，还可以将 MoCo 用于类似掩码自编码的前置任务，例如在语言 [12] 和视觉 [46] 中。我们希望 MoCo 能够与涉及对比学习的其他前置任务一起发挥作用。

## 附录

### A.1. 实现：目标检测骨干网络

R50-dilated-C5 和 R50-C4 骨干网络类似于 Detectron2 [60] 中可用的那些：
(i) **R50-dilated-C5：** 骨干网络包括 ResNet 的 conv5 阶段，膨胀率为 2，步长为 1，后接一个 3$$\times$$3 卷积（带 BN），将维度降至 512。框预测头由两个隐藏全连接层组成。
(ii) **R50-C4：** 骨干网络以 conv4 阶段结束，框预测头包括 conv5 阶段（包括全局池化）后接一个 BN 层。

### A.2. 实现：COCO 关键点检测

我们使用 Mask R-CNN（关键点版本），骨干网络为 R50-FPN，实现基于 [60]，在 COCO train2017 上微调，在 val2017 上评估。使用 2$$\times$$ 调度。

### A.3. 实现：COCO 密集姿态估计

我们使用 DensePose R-CNN [1]，骨干网络为 R50-FPN，实现基于 [60]，在 COCO train2017 上微调，在 val2017 上评估。使用 "s1$$\times$$" 调度。

### A.4. 实现：LVIS 实例分割

我们使用 Mask R-CNN，骨干网络为 R50-FPN，在 LVIS [27] train v0.5 上微调，在 val v0.5 上评估。我们遵循 [27]（arXiv v3 附录 B）中的基线。

LVIS 是一个新数据集，其上的模型设计还有待探索。下表包含相关消融（均为 5 次试验的平均值）：

| 预训练 | BN | 1$$\times$$调度 APmk | 1$$\times$$调度 APm$k_50$ | 1$$\times$$调度 APm$k_75$ | 2$$\times$$调度 APmk | 2$$\times$$调度 APm$k_50$ | 2$$\times$$调度 APm$k_75$ |
|--------|----|------------|--------------|--------------|------------|--------------|--------------|
| super. IN-1M | 冻结 | 24.1 | 37.3 | 25.4 | 24.4 | 37.8 | 25.8 |
| super. IN-1M | 可调 | 23.5 | 36.6 | 24.8 | 23.2 | 36.0 | 24.4 |
| MoCo IN-1M | 可调 | 23.2 | 36.0 | 24.7 | 24.1 | 37.4 | 25.5 |
| MoCo IG-1B | 可调 | 24.3 | 37.4 | 25.9 | 24.9 | 38.2 | 26.4 |

有监督预训练基线，端到端微调但 BN 冻结，有 24.4 APmk。但在此基线中微调 BN 会导致更差的结果和过拟合（这与 COCO/VOC 不同，在那里微调 BN 会带来更好或可比的准确率）。MoCo 在 IN-1M 上有 24.1 APmk，在 IG-1B 上有 24.9 APmk，两者在相同的可调 BN 设置下都优于有监督预训练对应方法。在各自最佳设置下，MoCo 仍然可以超越有监督预训练的情况（如第 4.2 节表 6 中所报告的 24.9 vs. 24.4）。

### A.5. 实现：语义分割

我们使用基于 FCN [43] 的结构。骨干网络由 R50 中的卷积层组成，conv5 块中的 3$$\times$$3 卷积具有膨胀率 2 和步长 1。其后接两个额外的 256 通道 3$$\times$$3 卷积，带 BN 和 ReLU，然后是一个 1$$\times$$1 卷积用于逐像素分类。总步长为 16（FCN-16s [43]）。我们在两个额外的 3$$\times$$3 卷积中设置膨胀率 = 6，遵循 [6] 中的大视野范围设计。

训练使用随机缩放（比例在 [0.5, 2.0] 范围内）、裁剪和水平翻转。裁剪大小在 VOC 上为 513，在 Cityscapes [6] 上为 769。推理在原始图像大小上进行。我们使用 mini-batch 大小 16 和权重衰减 0.0001 进行训练。学习率在 VOC 上为 0.003，在 Cityscapes 上为 0.01（在训练的 70% 和 90% 处乘以 0.1）。对于 VOC，我们在 train_aug2012 集（由 [30] 增强，10582 张图像）上训练 30k 次迭代，并在 val2012 上评估。对于 Cityscapes，我们在 train_fine 集（2975 张图像）上训练 90k 次迭代，并在 val 集上评估。结果报告为 5 次试验的平均值。

### A.6. iNaturalist 细粒度分类

除了主文中的检测/分割实验外，我们在 iNaturalist 2018 数据集 [57] 上研究了细粒度分类。我们在训练集（约 437k 图像，8142 个类别）上端到端地微调预训练模型，并在验证集上评估。训练遵循 PyTorch 中典型的 ResNet 实现，共 100 个 epoch。微调的学习率为 0.025（而从零开始为 0.1），在训练的 70% 和 90% 处减少 10 倍。以下是 R50 的结果：

| 预训练 | 随机初始化 | super. IN-1M | MoCo IN-1M | MoCo IG-1B |
|--------|----------|-------------|-----------|-----------|
| 准确率(%) | 61.8 | 66.1 | 65.6 | 65.8 |

MoCo 比随机初始化好约 4%，与其 ImageNet 有监督对应方法非常接近。这再次表明 MoCo 无监督预训练具有竞争力。

### A.7. ImageNet 上的微调

冻结特征的线性分类（第 4.1 节）是评估无监督预训练方法的常见协议。但在实践中，更常见的是在下游任务中端到端地微调特征。为了完整性，下表报告了 1000 类 ImageNet 分类的端到端微调结果，并与从零开始训练进行了比较（微调使用初始学习率 0.03，而从零开始为 0.1）：

| 预训练 | 随机初始化 | MoCo IG-1B |
|--------|----------|-----------|
| 准确率(%) | 76.5 | 77.3 |

由于这里 ImageNet 是下游任务，在 IN-1M 上预训练的 MoCo 的情况并不代表真实场景（作为参考，我们报告其微调后准确率为 77.0%）。但在 IG-1B 这个独立、无标签的数据集上进行无监督预训练代表了一个典型场景：在这种情况下，MoCo 提升了 0.8%。

### A.8. COCO 更长微调

在表 5 中，我们报告了 COCO 上 1$$\times$$（约 12 epoch）和 2$$\times$$ 调度的结果。这些调度继承自原始的 Mask R-CNN 论文 [32]，考虑到该领域的后续进展，可能不是最优的。在表 A.1 中，我们补充了 6$$\times$$ 调度（约 72 epoch）[31] 的结果，并与 2$$\times$$ 调度进行了比较。

我们观察到：(i) 使用 ImageNet 有监督预训练进行微调仍有提升（41.9 APbb）；(ii) 从零开始训练基本赶上（41.4 APbb）；(iii) MoCo 对应方法进一步提升（例如到 42.8 APbb）并且差距更大（例如 6$$\times$$ 时 +0.9 APbb 对比 2$$\times$$ 时 +0.5 APbb）。表 A.1 和表 5 表明，当微调时间更长时，MoCo 预训练特征可以比 ImageNet 有监督特征具有更大的优势。

### A.9. Shuffling BN 的消融

图 A.1 提供了有或没有 shuffling BN 时 MoCo 的训练曲线：移除 shuffling BN 显示对前置任务的明显过拟合：前置任务的训练准确率（虚线）迅速增加到 >99.9%，而基于 kNN 的验证分类准确率（实线）很快下降。这在 MoCo 和端到端变体中都能观察到；内存库变体隐式地具有不同的 q 和 k 统计量，因此避免了这个问题。

这些实验表明，没有 shuffling BN 时，子批统计量可以作为"签名"来指示正键所在的子批。Shuffling BN 可以移除这个签名并避免这种作弊。

## 参考文献

[1] Rıza Alp Güler, Natalia Neverova, and Iasonas Kokkinos. DensePose: Dense human pose estimation in the wild. In CVPR, 2018.

[2] Philip Bachman, R Devon Hjelm, and William Buchwalter. Learning representations by maximizing mutual information across views. arXiv:1906.00910, 2019.

[3] Mathilde Caron, Piotr Bojanowski, Armand Joulin, and Matthijs Douze. Deep clustering for unsupervised learning of visual features. In ECCV, 2018.

[4] Mathilde Caron, Piotr Bojanowski, Julien Mairal, and Armand Joulin. Unsupervised pre-training of image features on non-curated data. In ICCV, 2019.

[5] Ken Chatfield, Victor Lempitsky, Andrea Vedaldi, and Andrew Zisserman. The devil is in the details: an evaluation of recent feature encoding methods. In BMVC, 2011.

[6] Liang-Chieh Chen, George Papandreou, Iasonas Kokkinos, Kevin Murphy, and Alan L Yuille. DeepLab: Semantic image segmentation with deep convolutional nets, atrous convolution, and fully connected CRFs. TPAMI, 2017.

[7] Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations. arXiv:2002.05709, 2020.

[8] Xinlei Chen, Haoqi Fan, Ross Girshick, and Kaiming He. Improved baselines with momentum contrastive learning. arXiv:2003.04297, 2020.

[9] Adam Coates and Andrew Ng. The importance of encoding versus training with sparse coding and vector quantization. In ICML, 2011.

[10] Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, and Bernt Schiele. The Cityscapes dataset for semantic urban scene understanding. In CVPR, 2016.

[11] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. ImageNet: A large-scale hierarchical image database. In CVPR, 2009.

[12] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. In NAACL, 2019.

[13] Carl Doersch, Abhinav Gupta, and Alexei A Efros. Unsupervised visual representation learning by context prediction. In ICCV, 2015.

[14] Carl Doersch and Andrew Zisserman. Multi-task self-supervised visual learning. In ICCV, 2017.

[15] Jeff Donahue, Philipp Krähenbühl, and Trevor Darrell. Adversarial feature learning. In ICLR, 2017.

[16] Jeff Donahue and Karen Simonyan. Large scale adversarial representation learning. arXiv:1907.02544, 2019.

[17] Alexey Dosovitskiy, Jost Tobias Springenberg, Martin Riedmiller, and Thomas Brox. Discriminative unsupervised feature learning with convolutional neural networks. In NeurIPS, 2014.

[18] Mark Everingham, Luc Van Gool, Christopher KI Williams, John Winn, and Andrew Zisserman. The Pascal Visual Object Classes (VOC) Challenge. IJCV, 2010.

[19] Spyros Gidaris, Praveer Singh, and Nikos Komodakis. Unsupervised representation learning by predicting image rotations. In ICLR, 2018.

[20] Ross Girshick. Fast R-CNN. In ICCV, 2015.

[21] Ross Girshick, Jeff Donahue, Trevor Darrell, and Jitendra Malik. Rich feature hierarchies for accurate object detection and semantic segmentation. In CVPR, 2014.

[22] Ross Girshick, Ilija Radosavovic, Georgia Gkioxari, Piotr Dollár, and Kaiming He. Detectron, 2018.

[23] Aidan N Gomez, Mengye Ren, Raquel Urtasun, and Roger B Grosse. The reversible residual network: Backpropagation without storing activations. In NeurIPS, 2017.

[24] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. In NeurIPS, 2014.

[25] Priya Goyal, Piotr Dollár, Ross Girshick, Pieter Noordhuis, Lukasz Wesolowski, Aapo Kyrola, Andrew Tulloch, Yangqing Jia, and Kaiming He. Accurate, large minibatch SGD: Training ImageNet in 1 hour. arXiv:1706.02677, 2017.

[26] Priya Goyal, Dhruv Mahajan, Abhinav Gupta, and Ishan Misra. Scaling and benchmarking self-supervised visual representation learning. In ICCV, 2019.

[27] Agrim Gupta, Piotr Dollar, and Ross Girshick. LVIS: A dataset for large vocabulary instance segmentation. In CVPR, 2019.

[28] Michael Gutmann and Aapo Hyvärinen. Noise-contrastive estimation: A new estimation principle for unnormalized statistical models. In AISTATS, 2010.

[29] Raia Hadsell, Sumit Chopra, and Yann LeCun. Dimensionality reduction by learning an invariant mapping. In CVPR, 2006.

[30] Bharath Hariharan, Pablo Arbeláez, Lubomir Bourdev, Subhransu Maji, and Jitendra Malik. Semantic contours from inverse detectors. In ICCV, 2011.

[31] Kaiming He, Ross Girshick, and Piotr Dollár. Rethinking ImageNet pre-training. In ICCV, 2019.

[32] Kaiming He, Georgia Gkioxari, Piotr Dollár, and Ross Girshick. Mask R-CNN. In ICCV, 2017.

[33] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In CVPR, 2016.

[34] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Identity mappings in deep residual networks. In ECCV, 2016.

[35] Olivier J Hénaff, Ali Razavi, Carl Doersch, SM Eslami, and Aaron van den Oord. Data-efficient image recognition with contrastive predictive coding. arXiv:1905.09272, 2019.

[36] R Devon Hjelm, Alex Fedorov, Samuel Lavoie-Marchildon, Karan Grewal, Adam Trischler, and Yoshua Bengio. Learning deep representations by mutual information estimation and maximization. In ICLR, 2019.

[37] Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In ICML, 2015.

[38] Alexander Kolesnikov, Xiaohua Zhai, and Lucas Beyer. Revisiting self-supervised visual representation learning. In CVPR, 2019.

[39] Yann LeCun, Bernhard Boser, John S Denker, Donnie Henderson, Richard E Howard, Wayne Hubbard, and Lawrence D Jackel. Backpropagation applied to handwritten zip code recognition. Neural computation, 1989.

[40] Sungbin Lim, Ildoo Kim, Taesup Kim, Chiheon Kim, and Sungwoong Kim. Fast AutoAugment. arXiv:1905.00397, 2019.

[41] Tsung-Yi Lin, Piotr Dollár, Ross Girshick, Kaiming He, Bharath Hariharan, and Serge Belongie. Feature pyramid networks for object detection. In CVPR, 2017.

[42] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft COCO: Common objects in context. In ECCV, 2014.

[43] Jonathan Long, Evan Shelhamer, and Trevor Darrell. Fully convolutional networks for semantic segmentation. In CVPR, 2015.

[44] Dhruv Mahajan, Ross Girshick, Vignesh Ramanathan, Kaiming He, Manohar Paluri, Yixuan Li, Ashwin Bharambe, and Laurens van der Maaten. Exploring the limits of weakly supervised pretraining. In ECCV, 2018.

[45] Mehdi Noroozi and Paolo Favaro. Unsupervised learning of visual representations by solving jigsaw puzzles. In ECCV, 2016.

[46] Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. arXiv:1807.03748, 2018.

[47] Deepak Pathak, Ross Girshick, Piotr Dollár, Trevor Darrell, and Bharath Hariharan. Learning features by watching objects move. In CVPR, 2017.

[48] Deepak Pathak, Philipp Krahenbuhl, Jeff Donahue, Trevor Darrell, and Alexei A Efros. Context encoders: Feature learning by inpainting. In CVPR, 2016.

[49] Chao Peng, Tete Xiao, Zeming Li, Yuning Jiang, Xiangyu Zhang, Kai Jia, Gang Yu, and Jian Sun. MegDet: A large mini-batch object detector. In CVPR, 2018.

[50] Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever. Improving language understanding by generative pre-training. 2018.

[51] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners. 2019.

[52] Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun. Faster R-CNN: Towards real-time object detection with region proposal networks. In NeurIPS, 2015.

[53] Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. In ICLR, 2015.

[54] Josef Sivic and Andrew Zisserman. Video Google: a text retrieval approach to object matching in videos. In ICCV, 2003.

[55] Bart Thomee, David A Shamma, Gerald Friedland, Benjamin Elizalde, Karl Ni, Douglas Poland, Damian Borth, and Li-Jia Li. YFCC100M: The new data in multimedia research. Communications of the ACM, 2016.

[56] Yonglong Tian, Dilip Krishnan, and Phillip Isola. Contrastive multiview coding. arXiv:1906.05849, 2019.

[57] Grant Van Horn, Oisin Mac Aodha, Yang Song, Yin Cui, Chen Sun, Alex Shepard, Hartwig Adam, Pietro Perona, and Serge Belongie. The iNaturalist species classification and detection dataset. In CVPR, 2018.

[58] Pascal Vincent, Hugo Larochelle, Yoshua Bengio, and Pierre-Antoine Manzagol. Extracting and composing robust features with denoising autoencoders. In ICML, 2008.

[59] Xiaolong Wang and Abhinav Gupta. Unsupervised learning of visual representations using videos. In ICCV, 2015.

[60] Yuxin Wu, Alexander Kirillov, Francisco Massa, Wan-Yen Lo, and Ross Girshick. Detectron2. https://github.com/facebookresearch/detectron2, 2019.

[61] Zhirong Wu, Yuanjun Xiong, Stella Yu, and Dahua Lin. Unsupervised feature learning via non-parametric instance discrimination. In CVPR, 2018.

[62] Saining Xie, Ross Girshick, Piotr Dollár, Zhuowen Tu, and Kaiming He. Aggregated residual transformations for deep neural networks. In CVPR, 2017.

[63] Mang Ye, Xu Zhang, Pong C Yuen, and Shih-Fu Chang. Unsupervised embedding learning via invariant and spreading instance feature. In CVPR, 2019.

[64] Richard Zhang, Phillip Isola, and Alexei A Efros. Colorful image colorization. In ECCV, 2016.

[65] Richard Zhang, Phillip Isola, and Alexei A Efros. Split-brain autoencoders: Unsupervised learning by cross-channel prediction. In CVPR, 2017.

[66] Chengxu Zhuang, Alex Lin Zhai, and Daniel Yamins. Local aggregation for unsupervised learning of visual embeddings. In ICCV, 2019.
