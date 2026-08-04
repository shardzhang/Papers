# 努力了解邻居的边际：局部描述子学习损失

> Anastasiya Mishchuk¹, Dmytro Mishkin², Filip Radenović², Jiří Matas²
> ¹ Szkocka Research Group, Ukraine（乌克兰斯科茨卡研究组）
> ² Visual Recognition Group, CTU in Prague（布拉格捷克理工大学视觉识别组）

我们提出了一种用于度量学习的损失函数，其灵感来源于 Lowe 为 SIFT 设计的匹配准则。我们表明，所提出的损失函数——最大化批内最近正例与最近负例之间的距离——优于复杂的正则化方法；它对浅层和深层卷积网络架构都表现良好。将这种新颖的损失函数应用于 L2Net CNN 架构，产生了一种名为 HardNet 的紧凑描述子。它具有与 SIFT 相同的维度（128），并在宽基线立体匹配、图块验证和实例检索基准测试中展示了最先进的性能。

---

## 摘要

我们提出了一种用于度量学习的损失函数，其灵感来源于 Lowe 为 SIFT 设计的匹配准则。我们表明，所提出的损失函数——最大化批内最近正例与最近负例之间的距离——优于复杂的正则化方法；它对浅层和深层卷积网络架构都表现良好。将这种新颖的损失函数应用于 L2Net CNN 架构，产生了一种名为 HardNet 的紧凑描述子。它具有与 SIFT 相同的维度（128），并在宽基线立体匹配、图块验证和实例检索基准测试中展示了最先进的性能。

## 1 引言

许多计算机视觉任务依赖于寻找局部对应关系，例如图像检索 [1, 2]、全景拼接 [3]、宽基线立体匹配 [4]、三维重建 [5, 6]。尽管有越来越多的尝试试图用端到端学习模型替代复杂的经典流水线，例如图像匹配 [7]、相机定位 [8]，但由于局部图块的经典检测器和描述子的鲁棒性、高效性以及紧密集成性，它们仍在使用。此外，将复杂流水线解决的任務重新表述为可微分的端到端过程是极具挑战性的。

作为迈向端到端学习的第一步，手工设计的描述子如 SIFT [9, 10] 或检测器 [9, 11, 12] 已被学习型描述子所取代，例如 LIFT [13]、MatchNet [14] 和 DeepCompare [15]。然而，尽管在图块验证任务上表现良好，这些描述子在实际应用中并未获得广泛认可。最近的研究证实，在图像匹配和小规模检索 [18] 以及三维重建 [19] 中，SIFT 及其变体（RootSIFT-PCA [16]、DSP-SIFT [17]）显著优于学习型描述子。[19] 中得出的结论之一是，当前的局部图块数据集在规模和多样性上不足以学习高质量且广泛适用的描述子。

在本文中，我们专注于描述子学习，并使用一种新颖的方法训练了一个卷积神经网络（CNN），称为 HardNet。我们另外表明，我们学习的描述子在真实世界任务中显著优于手工设计和学习型的描述子，例如在极端条件下进行的图像检索和双视图匹配。在训练中，我们使用标准的图块对应数据，从而表明现有数据集足以超越当前最先进水平。

## 2 相关工作

经典的 SIFT 局部特征匹配包含两个部分：寻找最近邻以及比较第一与第二近邻距离比值阈值以过滤假阳性匹配。据我们所知，在局部描述子学习中，没有工作完全模仿这种策略作为学习目标。

Simonyan 和 Zisserman [20] 提出了一种简单的滤波加池化方案，通过凸优化学习以替换 SIFT 中手工设计的滤波器和池化。Han 等人 [14] 提出了一种两阶段孪生架构——分别用于嵌入和双图块相似性。后一个网络提升了匹配性能，但阻止了使用快速近似最近邻算法（如 kd-tree [21]）。Zagoruyko 和 Komodakis [15] 独立提出了类似的基于孪生的方法，探索了不同的卷积架构。Simo-Serra 等人 [22] 利用硬负例挖掘与相对浅层的架构，利用了基于对的相似性。

以下三篇论文最接近经典的 SIFT 匹配方案。Balntas 等人 [23] 使用了三元组边际损失和三元组距离损失，并对图块三元组进行随机采样。他们展示了基于三元组的架构优于基于对的架构。然而，与 SIFT 匹配或我们的工作不同，他们随机采样负例。Choy 等人 [7] 计算距离矩阵以挖掘正例和负例，然后使用成对对比损失。

Tian 等人 [24] 在批中使用 $n$ 个匹配对生成 $n^2 - n$ 个负样本，并要求在每个行和列中与真实匹配的距离最小。没有对距离或距离比施加其他约束。相反，他们提出了对描述子维度相关性的惩罚，并通过使用中间特征图进行匹配采用深度监督 [25]。鉴于其最先进的性能，我们采用 L2Net [24] 架构作为我们描述子的基础。我们表明，使用明显更简单的学习目标，无需两个辅助损失项，就可以学习到更强大的描述子。

<img src="...">  <!-- Figure 1 placeholder: 输入图块批 \\to 描述子 \\to 距离矩阵 \\to 选取最困难三元组 -->

图 1：提出的采样过程。首先，图块由当前网络描述，然后计算距离矩阵。对来自正对（绿色）的每个 $a_i$ 和 $p_i$ 图块，分别选择最近的非匹配描述子——以红色显示。最后，在两个负例候选中选择最困难的一个。所有操作都在单次前向传播中完成。

## 3 提出的描述子

### 3.1 采样与损失

我们的学习目标模仿了 SIFT 匹配准则。该过程如图 1 所示。首先，生成匹配局部图块的一个批 $X = (A_i, P_i)_{i=1..n}$ ，其中 $A$ 表示锚点， $P$ 表示正例。图块 $A_i$ 和 $P_i$ 对应于三维表面上的同一点。我们确保在批 $X$ 中，每个三维点恰好只产生一对。

其次， $X$ 中的 $2n$ 个图块通过图 2 所示的网络进行处理。计算 L2 成对距离矩阵 $D = \text{cdist}(a, p)$ ，其中 $d(a_i, p_j) = \sqrt{2 - 2a_i p_j}$ ， $i = 1..n$ ， $j = 1..n$ ，大小为 $n \times n$ ，这里 $a_i$ 和 $p_j$ 分别表示图块 $A_i$ 和 $P_j$ 的描述子。

接下来，对于每个匹配对 $a_i$ 和 $p_i$ ，分别找到最近的非匹配描述子，即第二近邻：

 $a_i$ —— 锚点描述子， $p_i$ —— 正例描述子，

 $p_{j_{\min}}$ —— 与 $a_i$ 最近的非匹配描述子，其中 $j_{\min} = \arg\min_{j=1..n, j \neq i} d(a_i, p_j)$ ，

 $a_{k_{\min}}$ —— 与 $p_i$ 最近的非匹配描述子，其中 $k_{\min} = \arg\min_{k=1..n, k \neq i} d(a_k, p_i)$ 。

然后从每个四元组描述子 $(a_i, p_i, p_{j_{\min}}, a_{k_{\min}})$ 中形成一个三元组：如果 $d(a_i, p_{j_{\min}}) < d(a_{k_{\min}}, p_i)$ ，则取 $(a_i, p_i, p_{j_{\min}})$ ，否则取 $(p_i, a_i, a_{k_{\min}})$ 。

我们的目标是最小化匹配描述子与最近非匹配描述子之间的距离。这 $n$ 个三元组距离被输入到三元组边际损失中：

$$
L = \frac{1}{n} \sum_{i=1,n} \max\left(0, 1 + d(a_i, p_i) - \min\left(d(a_i, p_{j_{\min}}), d(a_{k_{\min}}, p_i)\right)\right) \qquad (1)
$$

其中 $\min\left(d(a_i, p_{j_{\min}}), d(a_{k_{\min}}, p_i)\right)$ 在三元组构建期间预计算。

距离矩阵计算在 GPU 上完成，与随机三元组采样相比，唯一的额外开销是距离矩阵计算以及在行和列上计算最小值。此外，与通常的三元组学习相比，我们的方案只需要双流 CNN，而不是三流，这减少了 30% 的内存消耗和计算量。

与 [24] 不同，我们没有对中间层使用深度监督，也没有对描述子维度的相关性施加约束。我们没有经历显著的过拟合。

### 3.2 模型架构

<img src="...">  <!-- Figure 2 placeholder: HardNet架构图 -->

图 2：我们网络的架构，采用自 L2Net [24]。每个卷积层后接批归一化和 ReLU，最后一层除外。在最后一个卷积层之前使用 Dropout 正则化。

HardNet 架构（图 2）与 L2Net [24] 完全相同。除最后一层外，所有卷积层都应用零填充以保持空间尺寸。没有池化层，因为我们发现它们会降低描述子的性能。因此，空间尺寸通过步长卷积来减小。每个层（最后一层除外）之后添加批归一化 [26] 层和 ReLU [27] 非线性激活函数。在最后一个卷积层之前应用丢弃率为 0.1 的 Dropout [28] 正则化。网络的输出经过 L2 归一化，产生长度为 1 的 128 维描述子。尺寸为 $32 \times 32$ 像素的灰度输入图块通过减去每个图块的均值并除以每个图块的标准差进行归一化。

优化采用随机梯度下降，学习率为 0.1，动量为 0.9，权重衰减为 0.0001。在本文的大多数实验中，学习率在 10 个 epoch 内线性衰减至零。权重初始化为正交 [29]，增益为 0.6，偏置设为 0.01。训练使用 PyTorch 库 [30] 完成。

**NIPS 后更新。** 在 PyTorch 库的两次重大更新后，我们无法重现结果。因此，我们进行了超参数搜索，发现将学习率提高到 10 并将 Dropout 率提高到 0.3 可以获得更好的结果——见图 5 和表 1。我们在不同的机器上得到了相同的结果。

### 3.3 模型训练

**UBC Phototour [3]**，也称为 Brown 数据集。它包含三个子集：Liberty、Notre Dame 和 Yosemite，每个子集约 40 万个归一化的 $64 \times 64$ 图块。关键点由 DoG 检测器检测并通过三维模型验证。

测试集包含每个序列的 10 万个匹配和非匹配对。常见的设置是在一个子集上训练描述子，并在另外两个上进行测试。度量指标是在 0.95 真阳性召回率处的假阳性率（FPR）。Michel Keller 发现 [14] 和 [23] 的评估程序报告的是 FDR（假发现率）而非 FPR（假阳性率）。为避免结果混淆，我们决定同时提供 FPR 和 FDR 率，并重新估计了分数以便直接比较。结果显示在表 1 中。无论是否使用训练增强，所提出的描述子均优于竞争对手。我们没有包含多尺度图块采样或所谓的"中心-环绕"架构的结果，原因有二。第一，架构选择超出了本文范围。第二，[24, 31] 已经表明"中心-环绕"在 Brown 数据集上持续改善不同描述子的结果，但在其他更实际的设置上损害了匹配性能，例如在 Oxford-Affine [32] 数据集上。

在本文的其余部分，我们使用在 Liberty 序列上训练的描述子，这是常见做法，以保证公平比较。TFeat [23] 和 L2Net [24] 使用相同的数据集进行训练。

表 1：Brown 数据集上的图块对应验证性能。我们报告真阳性率为 95% 时的假阳性率（FPR95）。部分论文由于源代码中的错误报告了 FDR（假发现率）而非 FPR。为保持一致性，我们提供 FPR，要么从原始文章获得，要么从给定的 FDR 重新估计（标记为 *）。最佳结果以粗体显示。

| 训练集 | Notre Dame | Yosemite | Liberty | Yosemite | Liberty | Notre Dame | 平均 | |
|---|---|---|---|---|---|---|---|---|
| 测试集 | Liberty | Notre Dame | Yosemite | | | | FDR | FPR |
| SIFT [9] | 29.84 | 22.53 | 27.29 | | | | | 26.55 |
| MatchNet* [14] | 7.04 | 11.47 | 3.82 | 5.65 | 11.6 | 8.7 | 7.74 | 8.05 |
| TFeat-M* [23] | 7.39 | 10.31 | 3.06 | 3.8 | 8.06 | 7.24 | 6.47 | 6.64 |
| PCW [33] | 7.44 | 9.84 | 3.48 | 3.54 | 6.56 | 5.02 | | 5.98 |
| L2Net [24] | 3.64 | 5.29 | 1.15 | 1.62 | 4.43 | 3.30 | | 3.24 |
| HardNet $_{\text{NIPS}}$ | 3.06 | 4.27 | 0.96 | 1.4 | 3.04 | 2.53 | 3.00 | 2.54 |
| HardNet | 1.47 | 2.67 | 0.62 | 0.88 | 2.14 | 1.65 | | 1.57 |

增强：翻转、 $90^\circ$ 随机旋转

| GLoss+ [31] | 3.69 | 4.91 | 0.77 | 1.14 | 3.09 | 2.67 | | 2.71 |
| DC2ch2st+ [15] | 4.85 | 7.2 | 1.9 | 2.11 | 5.00 | 4.10 | | 4.19 |
| L2Net+ [24] | 2.36 | 4.7 | 0.72 | 1.29 | 2.57 | 1.71 | | 2.23 |
| HardNet $_{\text{NIPS}}$ + | 2.28 | 3.25 | 0.57 | 0.96 | 2.13 | 2.22 | 1.97 | 1.9 |
| HardNet+ | 1.49 | 2.51 | 0.53 | 0.78 | 1.96 | 1.84 | | 1.51 |

### 3.4 批大小影响探究

我们研究了迷你批大小对最终描述子性能的影响。众所周知，小的迷你批有利于更快的收敛和更好的泛化 [34]，而大批量则允许更好的 GPU 利用。我们的损失函数设计应该受益于看到更多难负例图块，以学习区分它们与真正例图块。我们报告了批大小 16、64、128、512、1024、2048 的结果。我们使用 Brown 数据集的 Liberty 序列训练了第 3.2 节中描述的模型。结果显示在图 3 中。正如预期，模型性能随着迷你批大小的增加而改善，因为看到了更多样本来获得更难的负例。不过，将批大小增加到 512 以上并不会带来显著益处。

<img src="...">  <!-- Figure 3 placeholder: 批大小影响 -->

图 3：批大小对描述子性能的影响。度量指标是在真阳性率为 95% 时的假阳性率（FPR），在 Notre Dame 和 Yosemite 验证序列上取平均。

<img src="...">  <!-- Figure 4 placeholder: 图块检索性能 -->

图 4：图块检索描述子性能（mAP）与干扰项数量的关系，在 HPatches 数据集上评估。

## 4 实证评估

最近，Balntas 等人 [23] 表明，在 Brown 数据集上的图块验证任务上表现良好并不总是意味着在最近邻设置中表现良好，反之亦然。因此，我们在真实世界任务中广泛评估了学习型描述子，例如双视图匹配和图像检索。

我们选择了 RootSIFT [10]、PCW [33]、TFeat-M* [23] 和 L2Net [24] 与我们的描述子进行直接比较，因为它们在各种数据集上展示了最佳结果。

### 4.1 图块描述子评估

**HPatches [18]** 是一个用于局部图块描述子评估的最新数据集。它包含 116 个序列，每个序列 6 张图像。该数据集分为两部分：视角部分——59 个具有显著视角变化的序列和光照部分——57 个具有显著光照变化（自然和人工）的序列。关键点在参考图像中通过 DoG、Hessian 和 Harris 检测器检测，并重投影到每个序列的其余图像中，具有 3 级几何噪声：Easy（简单）、Hard（困难）和 Tough（艰难）。HPatches 基准测试定义了三个任务：图块对应验证、图像匹配和小规模图块检索。我们请读者参考 HPatches 论文 [18] 以了解每个任务的详细协议。

<img src="...">  <!-- Figure 5 placeholder: HPatches结果 -->

图 5：从左到右：在 HPatches 数据集上的验证、匹配和检索结果。所有描述子均在 Brown [3] 数据集的 Liberty 子集上训练，或是手工设计的。在其他数据集上训练的描述子的比较见图 6。标记颜色表示几何噪声级别：EASY（简单）、HARD（困难）和 TOUGH（艰难）。标记类型表示实验设置。DIFFSEQ 和 SAMESEQ 表示验证任务的负例来源。VIEWPT 和 ILLUM 表示匹配的序列类型。没有描述子在 HPatches 上训练。HardNet 和 HardNet+ 是在 NIPS 之后使用优化后的超参数（学习率）训练的。

结果显示在图 5 中。L2Net 和 HardNet 在图块验证任务上表现出相似的性能，HardNet 略占优势。在匹配任务上，即使是未经增强的 HardNet 版本也以明显优势超越了增强版的 L2Net+。这种差异在 TOUGH 和 HARD 设置中更大。对所有描述子而言，光照序列比几何序列更具挑战性。我们使用 TFeat 架构训练了网络，但使用提出的损失函数——记为 HardTFeat。它在匹配和检索上优于原始版本，而在图块验证任务上与之持平。

在图块检索中，描述子的相对性能与匹配问题相似：HardNet 击败了 L2Net+。两个描述子都显著优于之前的最高水平，显示了所选深层 CNN 架构相对于浅层 TFeat 模型的优越性。

我们还进行了另一个图块检索实验，改变了检索数据集中干扰项（非匹配图块）的数量。结果如图 4 所示。TFeat 描述子的性能在干扰项数量较少时与 L2Net 相当，但随着数据库规模的增长而迅速下降。在大约 10,000 个干扰项时，其性能下降到 SIFT 以下。这个实验解释了为什么 TFeat 在 Oxford5k [35] 和 Paris6k [36] 基准测试上表现相对较差——它们分别包含约 1200 万和 1500 万个干扰项，详见第 4.4 节。HardNet 的性能在增强版和普通版中都略有下降，其 mAP 与其他描述子的差距随着任务复杂度的增加而增大。

**NIPS 后更新。** 我们研究了训练数据集对描述子性能的影响。特别是，我们比较了常用的由 DoG 检测器提取图块的 Liberty 子集、完整的 Brown 数据集——包括由 DoG 和 Harris 检测器提取图块的 Liberty、Notredame 和 Yosemite 子集。我们还包含了 Mitra 等人的结果，他们在新的大规模 PhotoSync 数据集上训练了 HardNet。结果显示在图 6 中。

### 4.2 消融研究

为更好地理解采样策略和损失函数的重要性，我们进行了总结在表 2 中的实验。我们训练我们的 HardNet 模型（架构与 L2Net 模型完全相同），一次改变一个参数并评估其影响。

比较了以下采样策略：随机采样、提出的"批内最困难"采样和"经典"难负例挖掘，即在每个 epoch 中从完整训练集中选择最近的负例。测试了以下损失函数：距离上的 softmin、边际 $m=1$ 的三元组边际损失、边际 $m=1$ 和 $m=2$ 的对比损失。后者是单位归一化描述子的最大可能距离。HPatches 匹配任务上的平均 mAP 如表 2 所示。

提出的"批内最困难"采样在所有损失函数上明显优于所有其他采样策略，并且是 HardNet 良好性能的主要原因。随机采样和"经典"难负例挖掘导致了严重的过拟合：训练损失很高，但测试性能很低，且在不同运行之间变化数倍。在所有损失函数上都观察到了这种行为。

<img src="...">  <!-- Figure 6 placeholder: 不同数据集训练 -->

图 6：在不同数据集上训练的 HardNet 版本比较。Liberty $_{\text{NIPS}}$ 、Liberty —— Liberty [3]；FullBrown6 —— Brown 数据集 [3] 的全部六个子集；PhotoSync —— Mitra 等人 [37] 的新数据集。从左到右：在 HPatches 数据集上的验证、匹配和检索结果。标记颜色表示几何噪声级别：EASY（简单）、HARD（困难）和 TOUGH（艰难）。标记类型表示实验设置。DIFFSEQ 和 SAMESEQ 表示验证任务的负例来源。VIEWPT 和 ILLUM 表示匹配的序列类型。以上没有描述子在 HPatches 上训练。HardNet++ 的结果未呈现，因为它在 HPatches 上训练过。

表 2：HPatches 匹配任务上损失函数和采样策略的比较，报告平均 mAP。CPR 表示 [24] 中提出的描述子通道间相关性的正则化惩罚。难负例挖掘每个 epoch 执行一次。最佳结果以粗体显示。HardNet 使用批内最困难采样和三元组边际损失。

| 采样 / 损失 | Softmin | 三元组边际 $m=1$ | 对比 $m=1$ | 对比 $m=2$ |
|---|---|---|---|---|
| 随机 | 过拟合 | 过拟合 | | |
| 难负例挖掘 | 过拟合 | 过拟合 | | |
| 随机 + CPR | 0.349 | 0.286 | 0.007 | 0.083 |
| 难负例挖掘 + CPR | 0.391 | 0.346 | 0.055 | 0.279 |
| 批内最困难（我们的） | 0.474 | 0.482 | 0.444 | 0.482 |

随机采样的类似结果在 [24] 中也有报告。难负例挖掘（"训练集中最困难"）的糟糕结果令人惊讶。我们猜测这是由于数据集标签噪声 —— 挖掘出的"难负例"实际上是正例。视觉检查证实了这一点。只有加上 [24] 中提出的描述子通道相关性惩罚（CPR），我们才能通过随机采样和难负例挖掘采样得到合理的结果。

关于损失函数，Softmin 在所有采样策略中给出了最稳定的结果，但在我们的策略下略逊于对比损失和三元组边际损失。一个可能的解释是，三元组边际损失和大边界的对比损失相对于正例和负例都有恒定的非零导数，见图 7。在小边界的对比损失情况下，许多负例在优化中不被使用（导数为零），而一旦正例的距离小于负例的距离，Softmin 的导数就会变小。

### 4.3 宽基线立体匹配

为验证描述子的泛化能力及其在极端条件下工作的能力，我们在 W1BS 数据集 [4] 上进行了测试。该数据集包含 40 对图像，每对图像之间存在一种特定的极端变化：

- 外观（A）：由于季节或天气变化、遮挡等导致的外观差异；
- 几何（G）：尺度、相机和物体位置的差异；
- 光照（L）：强度、光源波长的显著差异；
- 传感器（S）：传感器数据的差异（红外、MRI）。

此外，W1BS 数据集中的局部特征由 MSER [38]、Hessian-Affine [11]（采用 [39] 的实现）和 FOCI [40] 检测器检测。它们在不同的局部结构上激活，而非 DoG。注意，描述子训练使用了 DoG 图块。与 HPatches 设置的另一个显著区别是没有几何噪声：所有图块都完美地重投影到目标图像中。测试协议与 HPatches 匹配任务相同。

结果显示在图 8 中。HardNet 和 L2Net 性能相当，前者在具有几何和外观变化的图像上表现更好，而后者在 map2photo 和可见光 vs 红外图像对上稍好。两者都优于 SIFT，但差距很小。然而，考虑到显著的领域偏移量，描述子表现非常好，而 TFeat 则远远落后于 SIFT。HardTFeat 在 W1BS 数据集上显著优于原始 TFeat 描述子，显示了所提出损失函数的优越性。

在图块匹配和验证任务上的良好性能并不自动导致实践中更好的性能，例如注册更多的图像。因此，我们还在宽基线立体匹配设置上比较了描述子，使用两个度量指标：成功匹配的图像对数量和每对匹配的平均内点数量，遵循 [4] 的匹配器比较协议。对原始协议的唯一修改是移除了使用 ORB 检测器和描述子的初始快速匹配步骤，因为我们比较的是"SIFT 替代"描述子。

<img src="...">  <!-- Figure 7 placeholder: 梯度贡献 -->

图 7：来自正例和负例对梯度幅度的贡献。水平和垂直轴分别表示从锚点（a）到负例（n）和正例（p）的距离。当 $d(a, n) > d(a, p)$ 时，Softmin 损失梯度迅速减小，与三元组边际损失不同。对于对比损失，距离 $d(a, n) > m$ 的负例对梯度的贡献为零。三元组边际损失和大边界的对比损失行为非常相似。

结果显示在表 3 中。在 Edge Foci (EF) [40]、Extreme View (EVD) [41] 和 Oxford Affine (OxAff) [11] 数据集上的结果已经饱和，所有描述子都足以匹配所有图像对。HardNet 在每个图像的内点数量上略有优势。其余数据集：SymB [42]、GDB [43]、WxBS [4] 和 LTLL [44] 有一个共同点：图像对要么来自与照片不同的领域（例如，绘图对绘图），要么是跨领域的（例如，绘图对照片）。在这些数据集上，HardNet 优于学习型描述子，并与手工设计的 RootSIFT 持平。我们希望指出，HardNet 并没有学习在不同领域内或跨领域场景下进行匹配，因此这样的结果显示了其泛化能力。

表 3：在 MODS 匹配器 [4] 中，描述子在宽基线立体匹配数据集上的比较。报告了成功匹配的图像对数量和平均内点数量。表头中的数字对应于数据集中的图像对数量。

| 描述子 | EF 33 | 内点 | EVD 15 | 内点 | OxAff 40 | 内点 | SymB 46 | 内点 | GDB 22 | 内点 | WxBS 37 | 内点 | LTLL 172 | 内点 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| RootSIFT | 33 | 32 | 15 | 34 | 40 | 169 | 45 | 43 | 21 | 52 | 11 | 93 | 123 | 27 |
| TFeat-M* | 32 | 30 | 15 | 37 | 40 | 265 | 40 | 45 | 16 | 72 | 10 | 62 | 96 | 29 |
| L2Net+ | 33 | 34 | 15 | 34 | 40 | 304 | 43 | 46 | 19 | 78 | 9 | 51 | 127 | 26 |
| HardNet+ | 33 | 35 | 15 | 41 | 40 | 316 | 44 | 47 | 21 | 75 | 11 | 54 | 127 | 31 |

### 4.4 图像检索

我们在图像检索的实际应用中评估了我们的方法，并与相关方法进行了比较。使用了标准的图像检索数据集进行评估，即 Oxford5k [35] 和 Paris6k [36] 数据集。两个数据集都包含一组图像（Oxford5k 有 5062 张，Paris6k 有 6300 张），描绘了 11 个不同的地标以及干扰项。对每个 11 个地标，有 5 个由边界框定义的不同查询区域，每个数据集共有 55 个查询区域。性能报告为平均精度均值（mAP）[35]。

在第一个实验中，对于数据集中的每张图像，提取多尺度 Hessian-affine 特征 [32]。完全相同的特征由我们的方法和所有相关方法描述，每种方法为每个特征生成一个 128 维描述子。然后，使用近似最近邻的 k-means [21] 在独立数据集上学习一个 100 万视觉词汇的词典，即在评估 Oxford5k 时，词典使用 Paris6k 的描述子学习，反之亦然。测试数据集的所有描述子被分配到相应的词汇，因此最终每张图像由视觉词出现频率的直方图表示，即词袋（BoW）[1] 表示，并使用倒排文件进行高效搜索。此外，使用空间验证（SV）[35] 和标准查询扩展（QE）[36] 对搜索结果进行重排和精化。与相关工作在图块描述上的比较见表 4。HardNet+ 和 L2Net+ 在两个数据集和所有设置上性能相当，HardNet+ 在所有结果的平均值上略好（平均 mAP 69.5 vs 69.1）。长期以来在图像检索中表现最好的描述子 RootSIFT 落后了，其所有结果的平均 mAP 为 66.0。

我们还训练了 HardNet++ 版本——使用当时所有可用的训练数据：Brown 和 HPatches 数据集的并集，而不仅仅是来自 Brown 的 Liberty 序列（用于 HardNet+）。它显示了更多训练数据的益处，并在所有设置中表现最佳。

最后，我们将我们的描述子与使用局部特征的最先进图像检索方法进行比较。为公平起见，表 5 中呈现的所有方法都使用如前所述的相同局部特征检测器，在独立数据集上学习词典，并使用空间验证（SV）和查询扩展（QE）。在我们的方法（HardNet++–HQE）中，学习了包含 65k 个视觉词的视觉词典，并附加了汉明嵌入（HE）[45] 技术，该技术使用 128 位二进制签名进一步精化描述子分配。我们遵循与 RootSIFT–HQE [46] 方法相同的过程，用我们学习的 HardNet++ 描述子替换 RootSIFT。具体来说，我们使用：（i）作为汉明距离递减函数的投票权重 [47]；（ii）突发性抑制 [47]；（iii）特征到视觉词的多重分配 [36, 48]；以及（iv）带有特征聚合的查询扩展 [46]。所有参数按照 [46] 设置。我们的方法是在 Oxford5k 和 Paris6k 上在独立数据集学习词典时报告的最佳结果（在 Oxford5k 上，[10] 在包含相关图像的相同数据集上学习时报告了 mAP 89.1），并且使用相同数量的特征（在 Oxford5k 上，[46] 在使用两倍多的局部特征时报告了 mAP 89.4，即这里使用的 1250 万 vs 2200 万）。

表 4：词袋（BoW）图像检索的性能（mAP）评估。在独立数据集上学习包含 100 万视觉词的词典，即在评估 Oxford5k 时，使用 Paris6k 的特征学习词典，反之亦然。SV：空间验证。QE：查询扩展。最佳结果以粗体突出显示。除 SIFT 和 HardNet++ 外，所有描述子均在 Brown 数据集 [3] 的 Liberty 序列上学习。HardNet++ 在 Brown 和 HPatches [18] 数据集的并集上训练。

| 描述子 | Oxford5k BoW | Oxford5k BoW+SV | Oxford5k BoW+QE | Paris6k BoW | Paris6k BoW+SV | Paris6k BoW+QE |
|---|---|---|---|---|---|---|
| TFeat-M* [23] | 46.7 | 55.6 | 72.2 | 43.8 | 51.8 | 65.3 |
| RootSIFT [10] | 55.1 | 63.0 | 78.4 | 59.3 | 63.7 | 76.4 |
| L2Net+ [24] | 59.8 | 67.7 | 80.4 | 63.0 | 66.6 | 77.2 |
| HardNet | 59.0 | 67.6 | 83.2 | 61.4 | 67.4 | 77.5 |
| HardNet+ | 59.8 | 68.8 | 83.0 | 61.0 | 67.0 | 77.5 |
| HardNet++ | 60.8 | 69.6 | 84.5 | 65.0 | 70.3 | 79.1 |

表 5：与使用局部特征的最先进图像检索方法的性能（mAP）比较。词典在独立数据集上学习，即在评估 Oxford5k 时，使用 Paris6k 的特征学习词典，反之亦然。呈现的所有结果均使用了空间验证和查询扩展。VS：词典大小。SA：单一分配。MA：多重分配。最佳结果以粗体突出显示。

| 方法 | VS | Oxford5k SA | Oxford5k MA | Paris6k SA | Paris6k MA |
|---|---|---|---|---|---|
| SIFT–BoW [39] | 1M | 78.4 | 82.2 | – | – |
| SIFT–BoW-fVocab [49] | 16M | 74.0 | 84.9 | 73.6 | 82.4 |
| RootSIFT–HQE [46] | 65k | 85.3 | 88.0 | 81.3 | 82.8 |
| HardNet++–HQE | 65k | 86.8 | 88.3 | 82.8 | 84.9 |

<img src="...">  <!-- Figure 8 placeholder: W1BS结果 -->

图 8：在 W1BS 图块数据集上的描述子评估，报告了精度-召回曲线下的平均面积。字母表示干扰因素，A：外观；G：视角/几何；L：光照；S：传感器；map2photo：卫星照片 vs 地图。

## 5 结论

我们提出了一种用于学习局部图像描述子的新颖损失函数，其依赖于迷你批内的难负例挖掘以及最大化最近正例与最近负例图块之间的距离。所提出的采样策略在 Softmin、三元组边际损失和对比损失上均优于经典的难负例挖掘和随机采样。

所得到的描述子非常紧凑——它具有与 SIFT 相同的维度（128），在标准匹配、图块验证和检索基准测试上展示了最先进的性能，并且在 GPU 上计算速度快。训练源代码和训练好的卷积网络可在 https://github.com/DagnyT/hardnet 获取。

## 致谢

作者得到了以下资助的支持：捷克科学基金会项目 GACR P103/12/G084、奥地利交通、创新和技术部、联邦科学、研究和经济部以及上奥地利州在 COMET 中心框架内的支持、CTU 学生资助 SGS17/185/OHK3/3T/13 以及 MSMT LL1303 ERC-CZ 资助。Anastasiya Mishchuk 得到了 Szkocka Research Group 资助的支持。

## 参考文献

[1] Josef Sivic and Andrew Zisserman. Video google: A text retrieval approach to object matching in videos. In *International Conference on Computer Vision (ICCV)*, pages 1470–1477, 2003.

[2] Filip Radenovic, Giorgos Tolias, and Ondrej Chum. CNN image retrieval learns from BoW: Unsupervised fine-tuning with hard examples. In *European Conference on Computer Vision (ECCV)*, pages 3–20, 2016.

[3] Matthew Brown and David G. Lowe. Automatic panoramic image stitching using invariant features. *International Journal of Computer Vision (IJCV)*, 74(1):59–73, 2007.

[4] Dmytro Mishkin, Jiri Matas, Michal Perdoch, and Karel Lenc. Wxbs: Wide baseline stereo generalizations. *arXiv 1504.06603*, 2015.

[5] Johannes L. Schonberger, Filip Radenovic, Ondrej Chum, and Jan-Michael Frahm. From single image query to detailed 3D reconstruction. In *Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 5126–5134, 2015.

[6] Johannes L. Schonberger and Jan-Michael Frahm. Structure-from-motion revisited. In *Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 4104–4113, 2016.

[7] Christopher B. Choy, JunYoung Gwak, Silvio Savarese, and Manmohan Chandraker. Universal correspondence network. In *Advances in Neural Information Processing Systems*, pages 2414–2422, 2016.

[8] Alex Kendall, Matthew Grimes, and Roberto Cipolla. Posenet: A convolutional network for real-time 6-DOF camera relocalization. In *International Conference on Computer Vision (ICCV)*, 2015.

[9] David G. Lowe. Distinctive image features from scale-invariant keypoints. *International Journal of Computer Vision (IJCV)*, 60(2):91–110, 2004.

[10] Relja Arandjelovic and Andrew Zisserman. Three things everyone should know to improve object retrieval. In *Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 2911–2918, 2012.

[11] Krystian Mikolajczyk and Cordelia Schmid. Scale & affine invariant interest point detectors. *International Journal of Computer Vision (IJCV)*, 60(1):63–86, 2004.

[12] Ethan Rublee, Vincent Rabaud, Kurt Konolige, and Gary Bradski. ORB: An efficient alternative to SIFT or SURF. In *International Conference on Computer Vision (ICCV)*, pages 2564–2571, 2011.

[13] Kwang Moo Yi, Eduard Trulls, Vincent Lepetit, and Pascal Fua. LIFT: Learned invariant feature transform. In *European Conference on Computer Vision (ECCV)*, pages 467–483, 2016.

[14] Xufeng Han, T. Leung, Y. Jia, R. Sukthankar, and A. C. Berg. Matchnet: Unifying feature and metric learning for patch-based matching. In *Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 3279–3286, 2015.

[15] Sergey Zagoruyko and Nikos Komodakis. Learning to compare image patches via convolutional neural networks. In *Conference on Computer Vision and Pattern Recognition (CVPR)*, 2015.

[16] Andrei Bursuc, Giorgos Tolias, and Herve Jegou. Kernel local descriptors with implicit rotation matching. In *ACM International Conference on Multimedia Retrieval*, 2015.

[17] Jingming Dong and Stefano Soatto. Domain-size pooling in local descriptors: DSP-SIFT. In *Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 5097–5106, 2015.

[18] Vassileios Balntas, Karel Lenc, Andrea Vedaldi, and Krystian Mikolajczyk. HPatches: A benchmark and evaluation of handcrafted and learned local descriptors. In *Conference on Computer Vision and Pattern Recognition (CVPR)*, 2017.

[19] Johannes L. Schonberger, Hans Hardmeier, Torsten Sattler, and Marc Pollefeys. Comparative evaluation of hand-crafted and learned local features. In *Conference on Computer Vision and Pattern Recognition (CVPR)*, 2017.

[20] Karen Simonyan, Andrea Vedaldi, and Andrew Zisserman. Descriptor learning using convex optimisation. In *European Conference on Computer Vision (ECCV)*, pages 243–256, 2012.

[21] Marius Muja and David G. Lowe. Fast approximate nearest neighbors with automatic algorithm configuration. In *International Conference on Computer Vision Theory and Application (VISSAPP)*, pages 331–340, 2009.

[22] Edgar Simo-Serra, Eduard Trulls, Luis Ferraz, Iasonas Kokkinos, Pascal Fua, and Francesc Moreno-Noguer. Discriminative learning of deep convolutional feature point descriptors. In *International Conference on Computer Vision (ICCV)*, pages 118–126, 2015.

[23] Vassileios Balntas, Edgar Riba, Daniel Ponsa, and Krystian Mikolajczyk. Learning local feature descriptors with triplets and shallow convolutional neural networks. In *British Machine Vision Conference (BMVC)*, 2016.

[24] Bin Fan Yurun Tian and Fuchao Wu. L2-Net: Deep learning of discriminative patch descriptor in euclidean space. In *Conference on Computer Vision and Pattern Recognition (CVPR)*, 2017.

[25] Chen-Yu Lee, Saining Xie, Patrick Gallagher, Zhengyou Zhang, and Zhuowen Tu. Deeply-supervised nets. In *Artificial Intelligence and Statistics*, pages 562–570, 2015.

[26] Sergey Ioffe and Christian Szegedy. Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. *arXiv 1502.03167*, 2015.

[27] Vinod Nair and Geoffrey E. Hinton. Rectified linear units improve restricted boltzmann machines. In *International Conference on Machine Learning (ICML)*, pages 807–814, 2010.

[28] Nitish Srivastava, Geoffrey E Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov. Dropout: a simple way to prevent neural networks from overfitting. *Journal of Machine Learning Research (JMLR)*, 15(1):1929–1958, 2014.

[29] Andrew M. Saxe, James L. McClelland, and Surya Ganguli. Exact solutions to the nonlinear dynamics of learning in deep linear neural networks. In *Proceedings of ICLR*, 2014. http://arxiv.org/abs/1312.6120.

[30] PyTorch. http://pytorch.org.

[31] Vijay Kumar B. G., Gustavo Carneiro, and Ian Reid. Learning local image descriptors with deep siamese and triplet convolutional networks by minimising global loss functions. In *Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 5385–5394, 2016.

[32] Krystian Mikolajczyk, Tinne Tuytelaars, Cordelia Schmid, Andrew Zisserman, Jiri Matas, Frederik Schaffalitzky, Timor Kadir, and Luc Van Gool. A comparison of affine region detectors. *International Journal of Computer Vision (IJCV)*, 65(1):43–72, 2005.

[33] Arun Mukundan, Giorgos Tolias, and Ondrej Chum. Multiple-kernel local-patch descriptor. In *British Machine Vision Conference*, 2017. https://arxiv.org/pdf/1707.07825.pdf.

[34] D. Randall Wilson and Tony R. Martinez. The general inefficiency of batch training for gradient descent learning. *Neural Networks*, 16(10):1429–1451, 2003.

[35] James Philbin, Ondrej Chum, Michael Isard, Josef Sivic, and Andrew Zisserman. Object retrieval with large vocabularies and fast spatial matching. In *Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 1–8, 2007.

[36] James Philbin, Ondrej Chum, Michael Isard, Josef Sivic, and Andrew Zisserman. Lost in quantization: Improving particular object retrieval in large scale image databases. In *Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 1–8, 2008.

[37] R. Mitra, N. Doiphode, U. Gautam, S. Narayan, S. Ahmed, S. Chandran, and A. Jain. A Large Dataset for Improving Patch Matching. *arXiv e-prints*, January 2018.

[38] Jiri Matas, Ondrej Chum, Martin Urban, and Tomas Pajdla. Robust wide baseline stereo from maximally stable extrema regions. In *British Machine Vision Conference (BMVC)*, pages 384–393, 2002.

[39] Michal Perdoch, Ondrej Chum, and Jiri Matas. Efficient representation of local geometry for large scale object retrieval. In *Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 9–16, 2009.

[40] C. Lawrence Zitnick and Krishnan Ramnath. Edge foci interest points. In *International Conference on Computer Vision (ICCV)*, pages 359–366, 2011.

[41] Dmytro Mishkin, Jiri Matas, and Michal Perdoch. Mods: Fast and robust method for two-view matching. *Computer Vision and Image Understanding*, 141:81–93, 2015.

[42] Daniel C. Hauagge and Noah Snavely. Image matching using local symmetry features. In *Computer Vision and Pattern Recognition (CVPR)*, pages 206–213, 2012.

[43] Gehua Yang, Charles V Stewart, Michal Sofka, and Chia-Ling Tsai. Registration of challenging image pairs: Initialization, estimation, and decision. *Pattern Analysis and Machine Intelligence (PAMI)*, 29(11):1973–1989, 2007.

[44] Basura Fernando, Tatiana Tommasi, and Tinne Tuytelaars. Location recognition over large time lags. *Computer Vision and Image Understanding*, 139:21–28, 2015.

[45] Herve Jegou, Matthijs Douze, and Cordelia Schmid. Improving bag-of-features for large scale image search. *International Journal of Computer Vision (IJCV)*, 87(3):316–336, 2010.

[46] Giorgos Tolias and Herve Jegou. Visual query expansion with or without geometry: refining local descriptors by feature aggregation. *Pattern Recognition*, 47(10):3466–3476, 2014.

[47] Herve Jegou, Matthijs Douze, and Cordelia Schmid. On the burstiness of visual elements. In *Computer Vision and Pattern Recognition (CVPR)*, pages 1169–1176, 2009.

[48] Herve Jegou, Cordelia Schmid, Hedi Harzallah, and Jakob Verbeek. Accurate image search using the contextual dissimilarity measure. *Pattern Analysis and Machine Intelligence (PAMI)*, 32(1):2–11, 2010.

[49] Andrej Mikulik, Michal Perdoch, Ondřej Chum, and Jiří Matas. Learning vocabularies over a fine quantization. *International Journal of Computer Vision (IJCV)*, 103(1):163–175, 2013.
