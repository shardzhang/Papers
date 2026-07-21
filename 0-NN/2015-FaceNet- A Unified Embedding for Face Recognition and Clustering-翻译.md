# FaceNet: A Unified Embedding for Face Recognition and Clustering

> Florian Schroff, Dmitry Kalenichenko, James Philbin — Google Inc.

**摘要：** 本文提出了一个称为 FaceNet 的系统，它直接学习从人脸图像到紧凑欧几里得空间的映射，其中距离直接对应于人脸相似度的度量。一旦这个空间被构建出来，人脸识别、验证和聚类等任务就可以使用标准技术轻松实现，以 FaceNet 的 embeddings 作为特征向量。我们的方法使用深度卷积网络进行训练，直接优化 embedding 本身，而不是像之前的深度学习方法那样优化中间瓶颈层。在训练中，我们使用通过一种新颖的在线 triplet 挖掘方法生成的大致对齐的匹配/非匹配人脸 patches 的三元组。我们方法的好处在于其更高的表示效率：仅使用每张人脸 128 字节即可实现最先进的人脸识别性能。在广泛使用的 Labeled Faces in the Wild (LFW) 数据集上，我们的系统达到了 99.63% 的新纪录准确率。在 YouTube Faces DB 上，它达到了 95.12%。在两个数据集上，我们的系统将错误率与最佳已发表结果 [15] 相比降低了 30%。我们还引入了谐波 embedding (harmonic embeddings) 和谐波 triplet loss (harmonic triplet loss) 的概念，它们描述了不同网络产生的、相互兼容并允许直接相互比较的不同版本的人脸 embeddings。

---

## 摘要

尽管人脸识别领域近期取得了显著进展 [10, 14, 15, 17]，但在大规模下高效地实现人脸验证和识别对当前方法提出了严峻挑战。在本文中，我们提出了一个称为 FaceNet 的系统，它直接学习从人脸图像到紧凑欧几里得空间的映射，其中距离直接对应于人脸相似度的度量。一旦这个空间被构建出来，人脸识别、验证和聚类等任务就可以使用标准技术轻松实现，以 FaceNet embeddings 作为特征向量。

我们的方法使用深度卷积网络进行训练，直接优化 embedding 本身，而不是像之前的深度学习方法那样优化中间瓶颈层。在训练中，我们使用通过一种新颖的在线 triplet 挖掘方法生成的大致对齐的匹配/非匹配人脸 patches 的三元组。我们方法的好处在于其更高的表示效率：仅使用每张人脸 128 字节即可实现最先进的人脸识别性能。

在广泛使用的 Labeled Faces in the Wild (LFW) 数据集上，我们的系统达到了 99.63% 的新纪录准确率。在 YouTube Faces DB 上，它达到了 95.12%。在两个数据集上，我们的系统将错误率与最佳已发表结果 [15] 相比降低了 30%。

我们还引入了谐波 embedding (harmonic embeddings) 和谐波 triplet loss (harmonic triplet loss) 的概念，它们描述了不同网络产生的、相互兼容并允许直接相互比较的不同版本的人脸 embeddings。

## 1. 引言

在本文中，我们提出了一个用于人脸验证（这是同一个人吗）、识别（这个人是谁）和聚类（在这些人脸中找出共同的人）的统一系统。我们的方法基于使用深度卷积网络为每张图像学习一个欧几里得 embedding。网络经过训练，使得 embedding 空间中的平方 $L_2$ 距离直接对应于人脸相似度：同一个人的脸具有小距离，而不同人的脸具有大距离。

一旦这个 embedding 被生成，前述任务就变得直接了：人脸验证只需对两个 embeddings 之间的距离取阈值；识别变成一个 k-NN 分类问题；聚类可以使用现成的技术如 k-means 或凝聚聚类来实现。

之前基于深度网络的人脸识别方法使用分类层 [15, 17] 在一组已知人脸身份上进行训练，然后取一个中间瓶颈层作为用于泛化识别超出训练所用身份集合的表示。这种方法的缺点是间接性和低效率：人们必须希望瓶颈表示能很好地泛化到新的人脸；并且通过使用瓶颈层，每张人脸的表示大小通常非常大（数千维）。一些近期工作 [15] 使用 PCA 降低了这种维度，但这是一个线性变换，可以很容易地在网络的一层中学习。

与这些方法相反，FaceNet 直接训练其输出成为一个紧凑的 128 维 embedding，使用基于 LMNN [19] 的 triplet 损失函数。我们的 triplets 由两个匹配的人脸缩略图和一个非匹配的人脸缩略图组成，损失旨在通过一个距离边际 (margin) 将正面对与负面对分开。这些缩略图是人脸区域的紧致裁剪，除了缩放和平移外，没有进行 2D 或 3D 对齐。

选择使用哪些 triplets 对于实现良好性能至关重要，受课程学习 (curriculum learning) [1] 的启发，我们提出了一种新颖的在线负例挖掘策略，确保随着网络训练，triplets 的难度持续增加。为了提高聚类精度，我们还探索了难正例挖掘技术，该技术鼓励同一个人 embeddings 形成球状簇。

作为我们方法能够处理的惊人可变性的例证，参见图 1。图中显示的是来自 PIE [13] 的图像对，这些图像对以前被认为对人脸验证系统非常困难。

本文其余部分的概述如下：第 2 节回顾了该领域的文献；第 3.1 节定义了 triplet loss，第 3.2 节描述了我们新颖的 triplet 选择和训练流程；第 3.3 节描述了所使用的模型架构。最后，在第 4 和第 5 节中，我们展示了我们 embeddings 的一些定量结果，并定性地探索了一些聚类结果。

## 2. 相关工作

与其他采用深度网络的最新工作 [15, 17] 类似，我们的方法是一种纯粹的数据驱动方法，直接从人脸像素中学习其表示。我们不使用工程化特征，而是使用一个大型的标记人脸数据集来获得对姿态、光照和其他变化条件的适当不变性。

在本文中，我们探索了两种最近在计算机视觉社区取得巨大成功的不同深度网络架构。两者都是深度卷积网络 [8, 11]。第一种架构基于 Zeiler & Fergus [22] 模型，该模型由多个交错排列的卷积层、非线性激活层、局部响应归一化层和最大池化层组成。我们额外添加了受 [9] 工作启发的若干个 $1 \times 1 \times d$ 卷积层。第二种架构基于 Szegedy 等人的 Inception 模型，该模型最近作为 ImageNet 2014 [16] 的获胜方法被使用。这些网络使用混合层，并行运行几个不同的卷积层和池化层，并将它们的响应拼接起来。我们发现这些模型可以减少多达 20 倍的参数数量，并有潜力在相当性能下减少所需的 FLOPS 数量。

关于人脸验证和识别有大量的研究工作。回顾这些工作超出了本文的范围，因此我们将仅简要讨论最相关的近期工作。

[15, 17, 23] 的工作都采用了复杂的多阶段系统，将深度卷积网络的输出与 PCA 降维和 SVM 分类相结合。

Zhenyao 等人 [23] 使用深度网络将人脸"变形"为规范的正面视角，然后学习 CNN 将每张人脸分类为属于一个已知身份。对于人脸验证，使用了网络输出的 PCA 以及 SVM 集成。

Taigman 等人 [17] 提出了一种多阶段方法，将人脸对齐到一个通用的 3D 形状模型。训练一个多类网络在超过四千个身份上执行人脸识别任务。作者还实验了一种称为 Siamese 网络的方法，直接优化两个人脸特征之间的 $L_1$ 距离。他们在 LFW 上的最佳性能（97.35%）来自三个使用不同对齐和颜色通道的网络的集成。这些网络的预测距离（基于 $\chi^2$ 核的非线性 SVM 预测）使用非线性 SVM 进行组合。

Sun 等人 [14, 15] 提出了一个紧凑且因此计算相对廉价的网络。他们使用了 25 个这样的网络的集成，每个网络在不同的脸部 patch 上操作。为了在 LFW 上的最终性能（99.47% [15]），作者结合了 50 个响应（常规和翻转）。使用了 PCA 和一个联合贝叶斯模型 (Joint Bayesian model) [2]，该模型实际上对应于 embedding 空间中的线性变换。他们的方法不需要显式的 2D/3D 对齐。网络通过结合分类损失和验证损失进行训练。验证损失类似于我们采用的 triplet loss [12, 19]，因为它最小化相同身份人脸之间的 $L_2$ 距离，并在不同身份人脸之间的距离上施加一个边际。主要区别在于只比较图像对，而 triplet loss 鼓励一个相对距离约束。

与本文所用损失相似的损失在 Wang 等人 [18] 的工作中已被探索，用于按语义和视觉相似性对图像进行排序。

## 3. 方法

FaceNet 使用一个深度卷积网络。我们讨论了两种不同的核心架构：Zeiler & Fergus [22] 风格的网络和最近的 Inception [16] 类型的网络。这些网络的细节在第 3.3 节中描述。

在给定模型细节并将其视为黑箱（见图 2）的情况下，我们方法最重要的部分在于整个系统的端到端学习。为此，我们采用了 triplet loss，它直接反映了我们在人脸验证、识别和聚类中希望实现的目标。也就是说，我们追求一个 embedding $f(x)$，从图像 $x$ 到特征空间 $\mathbb{R}^d$，使得同一身份的所有人脸（无论成像条件如何）之间的平方距离很小，而不同身份的一对人脸图像之间的平方距离很大。

虽然我们没有直接与其他损失进行比较，例如 [14] 中方程 (2) 使用的使用正负对的损失，但我们认为 triplet loss 更适合人脸验证。动机是 [14] 中的损失鼓励一个身份的所有人脸被投影到 embedding 空间中的一个单点上。然而，triplet loss 试图在来自一个人的每对面部与所有其他面部之间强制设置一个边际。这允许一个身份的人脸生活在一个流形上，同时仍然强制距离并因此保持与其他身份的区分性。

以下部分描述了这种 triplet loss 以及如何在大规模下高效地学习它。

### 3.1. Triplet Loss

Embedding 由 $f(x) \in \mathbb{R}^d$ 表示。它将图像 $x$ 嵌入到一个 $d$ 维欧几里得空间中。此外，我们将这个 embedding 约束在 $d$ 维超球面上，即 $\|f(x)\|_2 = 1$。这个损失在 [19] 中是在最近邻分类的背景下被提出的。在这里，我们想要确保一个特定人物的图像 $x_i^a$（anchor）比任何其他人的图像 $x_i^n$（negative）更接近同一人的所有其他图像 $x_i^p$（positive）。这一点在图 3 中可视化。

因此我们要求：

$$
\|f(x_i^a) - f(x_i^p)\|_2^2 + \alpha < \|f(x_i^a) - f(x_i^n)\|_2^2,
\qquad (1)
$$

$$
\forall (f(x_i^a), f(x_i^p), f(x_i^n)) \in \mathcal{T}.
\qquad (2)
$$

其中 $\alpha$ 是在正负对之间强制设置的边际。$\mathcal{T}$ 是训练集中所有可能的 triplets 的集合，基数为 $N$。

被最小化的损失为：

$$
L = \sum_i^N \left[ \|f(x_i^a) - f(x_i^p)\|_2^2 - \|f(x_i^a) - f(x_i^n)\|_2^2 + \alpha \right]_+.
\qquad (3)
$$

生成所有可能的 triplets 会导致许多容易满足的 triplets（即满足方程 (1) 中的约束）。这些 triplets 不会有助于训练，并会导致收敛变慢，因为它们仍然会通过网络传递。选择困难的、活跃且因此能够有助于改进模型的 triplets 至关重要。下一节将讨论我们在 triplet 选择中使用的不同方法。

### 3.2. Triplet 选择

为了确保快速收敛，选择违反方程 (1) 中 triplet 约束的 triplets 至关重要。这意味着，给定 $x_i^a$，我们想要选择一个 $x_i^p$（难正例）使得 $\operatorname{argmax}_{x_i^p} \|f(x_i^a) - f(x_i^p)\|_2^2$，类似地选择 $x_i^n$（难负例）使得 $\operatorname{argmin}_{x_i^n} \|f(x_i^a) - f(x_i^n)\|_2^2$。

在整个训练集上计算 argmin 和 argmax 是不可行的。此外，这可能导致训练效果差，因为标签错误和成像质量差的人脸会主导难正例和难负例。有两个明显的选择可以避免这个问题：

- 每 $n$ 步离线生成 triplets，使用最新的网络检查点并在数据子集上计算 argmin 和 argmax。
- 在线生成 triplets。这可以通过在一个 mini-batch 内选择难正例/负例来实现。

在这里，我们专注于在线生成，使用数千个样本数量级的大型 mini-batches，并且只在一个 mini-batch 内计算 argmin 和 argmax。

为了对 anchor-positive 距离有一个有意义的表示，需要确保每个 mini-batch 中任何单一身份的样本数量最少。在我们的实验中，我们对训练数据进行采样，使得每个 mini-batch 中每个身份大约选择 40 张人脸。此外，每个 mini-batch 中还添加了随机采样的负例人脸。

我们不挑选最难的 positive，而是使用一个 mini-batch 中的所有 anchor-positive 对，同时仍然选择难 negative。我们没有对 mini-batch 内的难 anchor-positive 对与所有 anchor-positive 对进行并列比较，但我们在实践中发现，全 anchor-positive 方法更稳定，并且在训练初期收敛稍快。

我们还探索了离线生成 triplets 与在线生成的结合，这可能允许使用更小的批大小，但实验尚无定论。

在实践中，选择最难的 negatives 可能在训练早期导致不良的局部最小值，特别是可能导致模型崩溃（即 $f(x) = 0$）。为了缓解这个问题，选择 $x_i^n$ 使得下式成立是有帮助的：

$$
\|f(x_i^a) - f(x_i^p)\|_2^2 < \|f(x_i^a) - f(x_i^n)\|_2^2.
\qquad (4)
$$

我们将这些负例称为 semi-hard，因为它们比正例离 anchor 更远，但仍然困难，因为平方距离接近 anchor-positive 距离。这些 negatives 位于边际 $\alpha$ 之内。

如前所述，正确的 triplet 选择对于快速收敛至关重要。一方面，我们倾向于使用小的 mini-batches，因为它们在随机梯度下降 (SGD) [20] 中往往能改善收敛。另一方面，实现细节使得数十到数百个样本的批次更加高效。然而，与批大小相关的主要约束是我们从 mini-batches 中选择困难相关 triplets 的方式。在大多数实验中，我们使用大约 1800 个样本的批大小。

### 3.3. 深度卷积网络

在我们所有的实验中，我们使用带有标准反向传播 [8, 11] 和 AdaGrad [5] 的随机梯度下降 (SGD) 来训练 CNN。在大多数实验中，我们从 0.05 的学习率开始，然后降低以最终确定模型。模型从随机初始化开始，类似于 [16]，并在 CPU 集群上训练 1000 到 2000 小时。在 500 小时训练后，损失下降（和精度提升）显著减慢，但额外的训练仍能显著提高性能。边际 $\alpha$ 设置为 0.2。

我们使用了两种架构类型，并在实验部分更详细地探讨了它们的权衡。它们的实际区别在于参数数量和 FLOPS 的差异。最佳模型可能因应用而异。例如，运行在数据中心中的模型可以拥有大量参数并需要大量 FLOPS，而运行在手机上的模型则需要少量参数以便能够装入内存。

我们所有的模型都使用整流线性单元 (ReLU) 作为非线性激活函数。

第一类，如表 1 所示，在 Zeiler & Fergus [22] 架构的标准卷积层之间添加了 $1 \times 1 \times d$ 卷积层，如 [9] 所建议的，形成了一个深达 22 层的模型。它总共有 1.4 亿个参数，每张图像需要大约 16 亿 FLOPS。

我们使用的第二类基于 GoogLeNet 风格的 Inception 模型 [16]。这些模型的参数少了 20 倍（大约 660 万到 750 万），FLOPS 少了多达 5 倍（在 5 亿到 16 亿之间）。其中一些模型的大小（深度和滤波器数量）被大幅缩减，以便可以在手机上运行。其中一个，NNS1，有 2600 万参数，每张图像仅需要 2.2 亿 FLOPS。另一个，NNS2，有 430 万参数和 2000 万 FLOPS。

表 2 详细描述了 NN2，我们最大的网络。NN3 架构相同但输入尺寸缩小为 160x160。NN4 的输入尺寸仅为 96x96，因此大幅降低了 CPU 需求（2.85 亿 FLOPS vs NN2 的 16 亿 FLOPS）。除了减少的输入尺寸外，它在较高层中不使用 5x5 卷积，因为那时感受野已经太小了。通常我们发现，5x5 卷积可以在整个网络中移除而仅带来微小的精度下降。图 4 比较了我们所有的模型。

## 4. 数据集与评估

我们在四个数据集上评估我们的方法，除 Labeled Faces in the Wild 和 YouTube Faces 外，我们在人脸验证任务上评估我们的方法。即，给定一对人脸图像，使用平方 $L_2$ 距离阈值 $D(x_i, x_j)$ 来确定相同或不同的分类。同一身份的所有人脸对记为 $\mathcal{P}_{\text{same}}$，而不同身份的所有对记为 $\mathcal{P}_{\text{diff}}$。

我们将所有真正接受 (true accept) 的集合定义为：

$$
TA(d) = \{(i, j) \in \mathcal{P}_{\text{same}}, \text{ with } D(x_i, x_j) \le d\}.
\qquad (5)
$$

这些是在阈值 $d$ 下被正确分类为相同的人脸对 $(i, j)$。类似地，

$$
FA(d) = \{(i, j) \in \mathcal{P}_{\text{diff}}, \text{ with } D(x_i, x_j) \le d\}
\qquad (6)
$$

是在阈值 $d$ 下被错误分类为相同的所有对的集合（错误接受）。

对于给定的面部距离 $d$，验证率 $VAL(d)$ 和错误接受率 $FAR(d)$ 定义为：

$$
VAL(d) = \frac{|TA(d)|}{|\mathcal{P}_{\text{same}}|}, \quad FAR(d) = \frac{|FA(d)|}{|\mathcal{P}_{\text{diff}}|}.
\qquad (7)
$$

### 4.1. 留出测试集

我们保留了一个大约一百万张图像的留出集，其分布与我们的训练集相同，但身份不相交。为了评估，我们将其分成五个不相交的集合，每个 20 万张图像。然后对 10 万 \\times 10 万图像对计算 FAR 和 VAL 率。报告了五个划分上的标准误差。

### 4.2. 个人照片

这是一个与我们的训练集分布相似的测试集，但经过了人工验证，具有非常干净的标签。它包含三个个人照片集，总共约 1.2 万张图像。我们对所有 1.2 万张图像的平方对计算 FAR 和 VAL 率。

### 4.3. 学术数据集

Labeled Faces in the Wild (LFW) 是人脸验证事实上的学术测试集 [7]。我们遵循无限制、标记外部数据的标准协议，并报告平均分类精度以及均值的标准误差。

YouTube Faces DB [21] 是一个在人脸识别社区中越来越受欢迎的新数据集 [17, 15]。设置与 LFW 类似，但不是验证图像对，而是使用视频对。

## 5. 实验

除非另有说明，我们使用 1 亿到 2 亿个训练人脸缩略图，包含约 800 万个不同的身份。对每张图像运行人脸检测器，并生成人脸周围的紧致边界框。这些人脸缩略图被缩放到相应网络的输入尺寸。在我们的实验中，输入尺寸从 96x96 像素到 224x224 像素不等。

### 5.1. 计算精度权衡

在深入探讨更具体实验的细节之前，我们将讨论特定模型所需的精度与 FLOPS 数量之间的权衡。图 4 显示了 x 轴上的 FLOPS 以及在我们第 4.2 节的用户标记测试数据集上 0.001 错误接受率 (FAR) 下的精度。有趣地看到模型所需的计算量与其达到的精度之间存在强相关性。图中突出了我们在实验中更详细讨论的五种模型（NN1、NN2、NN3、NNS1、NNS2）。

我们还考察了与模型参数数量相关的精度权衡。然而，在这种情况下情况并不那么清晰。例如，基于 Inception 的模型 NN2 实现了与 NN1 相当的性能，但仅有其参数数量的二十分之一。不过，FLOPS 数量是相当的。显然，如果进一步减少参数数量，性能预计会在某个点下降。其他模型架构可能允许在不损失精度的情况下进一步减少，就像本例中 Inception [16] 所做的那样。

### 5.2. CNN 模型的影响

现在我们更详细地讨论我们选择的四种模型的性能。一方面，我们拥有传统的基于 Zeiler & Fergus 的架构，带有 $1 \times 1$ 卷积 [22, 9]（见表 1）。另一方面，我们有基于 Inception [16] 的模型，大幅减小了模型规模。总体而言，在最终性能上，两种架构的最佳模型表现相当。然而，我们的一些基于 Inception 的模型，如 NN3，在显著降低 FLOPS 和模型大小的同时，仍然取得了良好的性能。

在我们的个人照片测试集上的详细评估如图 5 所示。虽然最大的模型与微小的 NNS2 相比在精度上取得了显著的提升，但后者可以在手机上以每张图像 30ms 的速度运行，并且仍然足够准确以用于人脸聚类。ROC 曲线在 FAR < $10^{-4}$ 处的急剧下降表明测试数据 groundtruth 中存在噪声标签。在极低的错误接受率下，单个错误标记的图像就可能对曲线产生显著影响。

### 5.3. 对图像质量的敏感性

表 4 显示了我们的模型在广泛的图像尺寸范围内的鲁棒性。网络对 JPEG 压缩出人意料地鲁棒，在 JPEG 质量低至 20 时也表现非常好。对于尺寸小至 120x120 像素的人脸缩略图，性能下降非常小，即使在 80x80 像素时也表现出可接受的性能。这值得注意，因为网络是在 220x220 的输入图像上训练的。使用较低分辨率的人脸进行训练可以进一步改善这一范围。

### 5.4. Embedding 维度

我们探索了各种 embedding 维度，并选择了 128 维用于除表 5 中报告的比较之外的所有实验。人们会期望较大的 embeddings 至少与较小的 embeddings 表现一样好，然而，它们可能需要更多的训练才能达到相同的精度。也就是说，表 5 中报告的性能差异在统计上是不显著的。

应该注意的是，在训练中使用的是 128 维浮点向量，但它可以被量化为 128 字节而不损失精度。因此，每张人脸被紧凑地表示为一个 128 维字节向量，这对于大规模聚类和识别是理想的。更小的 embeddings 可以在精度略有损失的情况下实现，并可用于移动设备。

### 5.5. 训练数据量

表 6 显示了大量训练数据的影响。由于时间限制，该评估是在一个较小的模型上进行的；这种影响在更大的模型上可能更大。很明显，使用数千万个样本在我们的第 4.2 节的个人照片测试集上带来了明显的精度提升。与仅有数百万张图像相比，错误的相对减少了 60%。再使用一个数量级更多的图像（数亿张）仍然带来小幅提升，但改进逐渐减弱。

### 5.6. LFW 上的性能

我们使用无限制、标记外部数据的标准协议在 LFW 上评估我们的模型。使用九个训练划分来选择 $L_2$ 距离阈值。然后对第十个测试划分进行分类（相同或不同）。选择的优化阈值对于所有测试划分均为 1.242，除了第八个划分（1.256）。

我们的模型以两种模式进行评估：
1. LFW 提供的缩略图的固定中心裁剪。
2. 对提供的 LFW 缩略图运行专有的人脸检测器（类似于 Picasa [3]）。如果它无法对齐人脸（这发生在两张图像上），则使用 LFW 对齐。

图 6 给出了所有失败案例的概览。它显示了顶部的错误接受以及底部的错误拒绝。当使用 (1) 中描述的固定中心裁剪时，我们达到了 98.87% \\pm 0.15 的分类精度，而当使用额外的人脸对齐 (2) 时，达到了创纪录的 99.63% \\pm 0.09 的均值标准误差。这比 [17] 中报告的 DeepFace 的错误减少了超过 7 倍，比 [15] 中报告的 DeepId2+ 的先前最佳结果减少了 30%。这是模型 NN1 的性能，但即使小得多的 NN3 也实现了没有统计显著差异的性能。

### 5.7. YouTube Faces DB 上的性能

我们使用我们的面部检测器在每个视频中检测到的前一百帧的所有对的平均相似度。这给出了 95.12% \\pm 0.39 的分类精度。使用前一千帧得到 95.18%。与同样评估每个视频一百帧的 [17] 的 91.4% 相比，我们将错误率降低了近一半。DeepId2+ [15] 达到了 93.2%，我们的方法将该错误减少了 30%，与我们在 LFW 上的改进相当。

### 5.8. 人脸聚类

我们紧凑的 embedding 适合用于将用户的个人照片聚类为具有相同身份的人群组。与纯粹的验证任务相比，聚类人脸所施加的分配约束带来了真正惊人的结果。图 7 显示了一个用户个人照片集中的一个示例聚类，使用凝聚聚类生成。它清楚地展示了对于遮挡、光照、姿态甚至年龄的惊人不变性。

## 6. 总结

我们提供了一种直接学习用于人脸验证的欧几里得空间 embedding 的方法。这使其与使用 CNN 瓶颈层或需要额外后处理（如多个模型的拼接和 PCA、SVM 分类）的其他方法 [15, 17] 区分开来。我们的端到端训练既简化了设置，也表明直接优化与手头任务相关的损失可以提高性能。

我们模型的另一个优势在于它只需要最小对齐（人脸区域周围的紧致裁剪）。例如，[17] 执行了复杂的 3D 对齐。我们还实验了相似变换对齐，并注意到这实际上可以略微提高性能。目前尚不清楚这种额外的复杂性是否值得。

未来的工作将集中于更好地理解错误案例、进一步改进模型，以及减少模型大小和 CPU 需求。我们还将研究改进当前极长训练时间的方法，例如使用更小的批大小以及离线和在线正负例挖掘的课程学习变体。

## 7. 附录：谐波嵌入 (Harmonic Embedding)

在本节中，我们介绍谐波 embeddings (harmonic embeddings) 的概念。我们指的是由不同模型 $v1$ 和 $v2$ 生成的一组 embeddings，它们在可以相互比较的意义上是兼容的。

这种兼容性极大地简化了升级路径。例如，在 embedding $v1$ 已经在一大组图像上计算完毕，而新的 embedding 模型 $v2$ 正在部署的场景中，这种兼容性确保了平滑过渡，无需担心版本不兼容问题。图 8 显示了在我们的 3G 数据集上的结果。可以看出，改进后的模型 NN2 显著优于 NN1，而 NN2 embeddings 与 NN1 embeddings 的比较则处于中间水平。

### 7.1. 谐波三元组损失 (Harmonic Triplet Loss)

为了学习谐波 embedding，我们将 $v1$ 的 embeddings 与被学习的 $v2$ 的 embeddings 混合。这是在 triplet loss 内部完成的，并导致额外生成的 triplets，鼓励不同 embedding 版本之间的兼容性。图 9 可视化了贡献于 triplet loss 的 triplets 的不同组合。

我们从一个独立训练的 NN2 初始化 $v2$ embedding，并使用鼓励兼容性的 triplet loss 从随机初始化重新训练最后一层（embedding 层）。首先只重新训练最后一层，然后我们继续使用谐波损失训练整个 $v2$ 网络。

图 10 展示了这种兼容性在实践中可能如何工作的一种可能解释。绝大多数 $v2$ embeddings 可能被嵌入到相应的 $v1$ embedding 附近，然而，错误放置的 $v1$ embeddings 可以被轻微扰动，使得它们在 embedding 空间中的新位置提高了验证精度。

### 7.2. 总结

这些是非常有趣的发现，并且它在如此良好的效果有些令人惊讶。未来的工作可以探索这个想法可以扩展到什么程度。据推测，$v2$ embedding 在保持兼容的同时能够超越 $v1$ 的程度是有限的。此外，训练可以在手机上运行的小型网络并使其与更大的服务器端模型兼容也将是有趣的。

## 致谢

我们要感谢 Johannes Steffens 在人脸识别方面的讨论和深刻见解，以及 Christian Szegedy 提供像 [16] 这样的新网络架构并讨论网络设计选择。同时，我们感谢 DistBelief [4] 团队的支持，特别是 Rajat Monga 在建立高效训练方案方面的帮助。

此外，没有 Chuck Rosenberg、Hartwig Adam 和 Simon Han 的支持，我们的工作是不可能完成的。

## 参考文献

[1] Y. Bengio, J. Louradour, R. Collobert, and J. Weston. Curriculum learning. In *Proc. of ICML*, New York, NY, USA, 2009.

[2] D. Chen, X. Cao, L. Wang, F. Wen, and J. Sun. Bayesian face revisited: A joint formulation. In *Proc. ECCV*, 2012.

[3] D. Chen, S. Ren, Y. Wei, X. Cao, and J. Sun. Joint cascade face detection and alignment. In *Proc. ECCV*, 2014.

[4] J. Dean, G. Corrado, R. Monga, K. Chen, M. Devin, M. Mao, M. Ranzato, A. Senior, P. Tucker, K. Yang, Q. V. Le, and A. Y. Ng. Large scale distributed deep networks. In *NIPS*, pages 1232–1240, 2012.

[5] J. Duchi, E. Hazan, and Y. Singer. Adaptive subgradient methods for online learning and stochastic optimization. *J. Mach. Learn. Res.*, 12:2121–2159, July 2011.

[6] I. J. Goodfellow, D. Warde-farley, M. Mirza, A. Courville, and Y. Bengio. Maxout networks. In *ICML*, 2013.

[7] G. B. Huang, M. Ramesh, T. Berg, and E. Learned-Miller. Labeled faces in the wild: A database for studying face recognition in unconstrained environments. Technical Report 07-49, University of Massachusetts, Amherst, October 2007.

[8] Y. LeCun, B. Boser, J. S. Denker, D. Henderson, R. E. Howard, W. Hubbard, and L. D. Jackel. Backpropagation applied to handwritten zip code recognition. *Neural Computation*, 1(4):541–551, Dec. 1989.

[9] M. Lin, Q. Chen, and S. Yan. Network in network. *CoRR*, abs/1312.4400, 2013.

[10] C. Lu and X. Tang. Surpassing human-level face verification performance on LFW with gaussianface. *CoRR*, abs/1404.3840, 2014.

[11] D. E. Rumelhart, G. E. Hinton, and R. J. Williams. Learning representations by back-propagating errors. *Nature*, 1986.

[12] M. Schultz and T. Joachims. Learning a distance metric from relative comparisons. In *NIPS*, pages 41–48. MIT Press, 2004.

[13] T. Sim, S. Baker, and M. Bsat. The CMU pose, illumination, and expression (PIE) database. In *Proc. FG*, 2002.

[14] Y. Sun, X. Wang, and X. Tang. Deep learning face representation by joint identification-verification. *CoRR*, abs/1406.4773, 2014.

[15] Y. Sun, X. Wang, and X. Tang. Deeply learned face representations are sparse, selective, and robust. *CoRR*, abs/1412.1265, 2014.

[16] C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov, D. Erhan, V. Vanhoucke, and A. Rabinovich. Going deeper with convolutions. *CoRR*, abs/1409.4842, 2014.

[17] Y. Taigman, M. Yang, M. Ranzato, and L. Wolf. Deepface: Closing the gap to human-level performance in face verification. In *IEEE Conf. on CVPR*, 2014.

[18] J. Wang, Y. Song, T. Leung, C. Rosenberg, J. Wang, J. Philbin, B. Chen, and Y. Wu. Learning fine-grained image similarity with deep ranking. *CoRR*, abs/1404.4661, 2014.

[19] K. Q. Weinberger, J. Blitzer, and L. K. Saul. Distance metric learning for large margin nearest neighbor classification. In *NIPS*. MIT Press, 2006.

[20] D. R. Wilson and T. R. Martinez. The general inefficiency of batch training for gradient descent learning. *Neural Networks*, 16(10):1429–1451, 2003.

[21] L. Wolf, T. Hassner, and I. Maoz. Face recognition in unconstrained videos with matched background similarity. In *IEEE Conf. on CVPR*, 2011.

[22] M. D. Zeiler and R. Fergus. Visualizing and understanding convolutional networks. *CoRR*, abs/1311.2901, 2013.

[23] Z. Zhu, P. Luo, X. Wang, and X. Tang. Recover canonical-view faces in the wild with deep neural networks. *CoRR*, abs/1404.3543, 2014.
