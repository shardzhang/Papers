# CNN Features off-the-shelf: an Astounding Baseline for Recognition

> Ali Sharif Razavian, Hossein Azizpour, Josephine Sullivan, Stefan Carlsson
>
> CVAP, KTH (Royal Institute of Technology), Stockholm, Sweden
>
> {razavian, azizpour, sullivan, stefanc}@csc.kth.se

本文展示了从卷积神经网络中提取的通用描述符具有非常强大的能力。我们使用公开可用的 OverFeat 网络的代码和模型（该网络在 ILSVRC13 上训练以进行目标分类），针对不同的识别任务进行了一系列实验。我们将 OverFeat 网络提取的特征作为通用图像表示，用于处理多种识别任务，包括目标图像分类、场景识别、细粒度识别、属性检测和图像检索，并在多个数据集上进行实验。我们选择这些任务和数据集，是因为它们逐步远离 OverFeat 网络原始训练时的任务和数据。令人惊讶的是，我们在所有视觉分类任务上的各种数据集中都取得了持续优于精心调优的最先进系统的结果。对于实例检索任务，除了雕塑数据集外，它在低内存占用方法中持续表现更优。这些结果是通过将线性 SVM 分类器（或检索任务中的 L2 距离）应用于从网络某层提取的大小为 4096 的特征表示实现的。该表示还通过简单的增广技术（如 jittering）进行了进一步修改。这些结果强烈表明，使用卷积网络进行深度学习得到的特征应成为大多数视觉识别任务的首选方案。

---

## 摘要

最近的结果表明，从卷积神经网络中提取的通用描述符非常强大。本文进一步证实了这一观点。我们使用公开可用的 OverFeat 网络的代码和模型（该网络在 ILSVRC13 上训练以进行目标分类），针对不同的识别任务进行了一系列实验。我们将 OverFeat 网络提取的特征作为通用图像表示，用于处理多种识别任务，包括目标图像分类、场景识别、细粒度识别、属性检测和图像检索，并在多个数据集上进行实验。我们选择这些任务和数据集，是因为它们逐步远离 OverFeat 网络原始训练时的任务和数据。令人惊讶的是，我们在所有视觉分类任务上的各种数据集中都取得了持续优于精心调优的最先进系统的结果。对于实例检索任务，除了雕塑数据集外，它在低内存占用方法中持续表现更优。这些结果是通过将线性 SVM 分类器（或检索任务中的 L2 距离）应用于从网络某层提取的大小为 4096 的特征表示实现的。该表示还通过简单的增广技术（如 jittering）进行了进一步修改。这些结果强烈表明，使用卷积网络进行深度学习得到的特征应成为大多数视觉识别任务的首选方案。

## 1. 引言

"深度学习。你觉得它对你的计算机视觉问题效果如何？" 这个问题很可能在你团队的咖啡间里被提出来。作为回应，有人引用了最近的成功案例 [29, 15, 10]，而另一些人则表示怀疑。你可能带着一丝沮丧离开咖啡间，心想："可惜我既没有时间、GPU 编程技能，也没有大量标注数据来训练自己的网络，无法快速找到答案。" 但是，当卷积神经网络 OverFeat [38] 最近公开可用时，这为一些实验提供了可能。

我们特别想知道的是——不是能否针对特定任务训练一个深度网络——而是由一个深度网络（经过精心训练的、在多样的 ImageNet 数据库上执行特定图像分类任务的网络）提取的特征，是否能够被用于广泛的视觉任务。下面我们分享我们的讨论和总体发现，因为作为一名计算机视觉研究员，你可能也有过同样的问题：

**教授：** 首先，有没有其他人研究过这个问题？

**学生：** 事实上，Donahue 等人 [10]、Zeiler 和 Fergus [48] 以及 Oquab 等人 [29] 已经提出，可以从大型 CNN 中提取通用特征，并为此提供了一些初步证据。但他们只考虑了少数视觉识别任务。如果能更全面地研究这些 CNN 特征有多强大，那会很有趣。我们应该从何入手？

**教授：** 我们可以做的最简单的事情是，从 OverFeat 网络中提取图像特征向量，并结合一个简单的线性分类器。这个特征向量可以只是网络的某个最终层以图像为输入时的响应。你认为这种方法对哪些视觉任务有效？

**学生：** 绝对是图像分类。已经有几个视觉团队在 Pascal VOC 上取得了相比之前最先进方法的巨大性能提升。但是否需要微调网络才能获得这种提升？我打算在 Pascal VOC 上试试，为了增加一点难度，我还要在 MIT 场景数据集上测试。

**回答：** OverFeat 即使不进行微调也表现非常好（详见第 3.2 节）。

**教授：** 好吧，这个结果证实了之前的发现，也许并不那么令人惊讶。我们让 OverFeat 特征去解决它们原本被训练来解决的问题。而且 ImageNet 或多或少是 Pascal VOC 的超集。不过我对室内场景数据集的结果印象深刻。那么，对于不太适合的问题呢？

**学生：** 我知道细粒度分类。这里我们需要区分一个类别下的子类别，比如不同种类的花。你认为更通用的 OverFeat 特征是否有足够的表征能力来捕捉非常相似类别之间潜在的细微差异？

**回答：** 在标准的鸟类和花卉数据库上效果很好。在最简单的形式下，它没有超越最新的最佳方法，但它是一种更简洁的解决方案，且具有很大的改进空间。实际上，采用一组简单的数据增广技术（仍然使用线性 SVM）就超越了最佳方法。令人印象深刻！（详见第 3.4 节。）

**教授：** 下一个挑战，属性检测？让我们看看 OverFeat 特征是否编码了关于人和物体的语义属性信息。

**学生：** 你认为从人物边界框中提取的全局 CNN 特征是否能应对 H3D 数据集中存在的姿态变化和遮挡？所有最好的方法在分类前和训练过程中都会进行某种部位对齐。

**回答：** 令人惊讶的是，针对 H3D 数据集中标注的人物属性，CNN 特征在平均性能上超过了 poselets 和可变形部件模型。哇，它们是怎么做到的？！它们在目标属性数据集上也表现出色。也许这些 OverFeat 特征确实编码了属性信息？（详见第 3.5 节。）

**教授：** 我们能更进一步吗？有没有哪些任务，OverFeat 特征相比于更成熟的计算机视觉系统会遇到困难？也许是实例检索。这个任务推动了 SIFT 和 VLAD 描述符以及视觉词袋方法的发展。这些高度优化的工程化向量和中层特征肯定应该轻松击败通用特征吧？

**学生：** 如果开始与那些还结合了 3D 几何约束的方法进行比较，我认为 CNN 特征没有机会。让我们专注于描述符的性能。在新学派的描述符与旧学派描述符的"主场"较量中，新学派能获胜吗？

**回答：** 非常有说服力。忽略那些施加了 3D 几何约束的系统，CNN 特征在建筑和假日数据集上非常有竞争力（第 4 节）。此外，进行标准的实例检索特征处理（即 PCA、白化、重新归一化）后，它在所有检索基准测试中（除雕塑数据集外）相比低内存占用方法表现出更优的性能。

**学生：** 所有这些结果的核心信息是什么？

**教授：** 一切都归结于特征！SIFT 和 HOG 描述符在十年前带来了巨大的性能提升，而现在深度卷积特征正在为识别领域带来类似的突破。因此，将成熟的计算机视觉流程应用于 CNN 表示，应该有可能进一步推动所报告的结果。无论如何，如果你为某个识别任务开发了新算法，它必须与通用深度特征 + 简单分类器这一强基线进行比较。

## 2. 背景与概述

在本工作中，我们使用公开可用的预训练 CNN，名为 OverFeat [38]。该网络的结构遵循 Krizhevsky 等人 [22] 的设计。卷积层包含 96 到 1024 个大小为 $3\times3$ 到 $7\times7$ 的卷积核。使用半波整流作为非线性激活函数。在不同层使用大小为 $3\times3$ 和 $5\times5$ 的最大池化卷积核，以增强对类内形变的鲁棒性。我们使用 OverFeat 网络的"large"版本。它接受大小为 $221\times221$ 的彩色图像作为输入。更多细节请参阅 [38] 和 [22]。

OverFeat 是在 ImageNet ILSVRC 2013 [1] 的图像分类任务上训练的，在 2013 年挑战赛的分类任务中获得了非常有竞争力的结果，并赢得了定位任务。ILSVRC13 包含 120 万张图像，这些图像被人工标注了 1000 个类别是否存在。这些图像大多是居中的，与 PASCAL VOC [12] 等其他目标识别数据集相比，该数据集在杂乱度和遮挡方面被认为挑战性较小。

我们报告了在不同识别任务上进行的一系列实验的结果。所选择的任务和数据集逐步远离 OverFeat 网络被训练执行的任务。我们有两个部分，分别涉及视觉分类（第 3 节）和视觉实例检索（第 4 节），在其中我们回顾了不同的任务和数据集并报告了最终结果。需要牢记的关键点是，所使用的 CNN 特征仅使用 ImageNet 数据训练，而简单的分类器则使用特定任务数据集的图像进行训练。

最后，我们必须指出，在有足够计算资源的情况下，针对特定任务/数据集优化 CNN 特征可能会进一步提升这个简单系统的性能 [29, 15, 51, 43, 41]。

## 3. 视觉分类

我们在以下小节中讨论与视觉分类相关的不同任务。

### 3.1. 方法

除非另有说明，在所有的实验中，我们使用网络的第一个全连接层（第 22 层）作为特征向量。请注意，最大池化和整流操作在 OverFeat 中均被视为单独的层，这不同于 Alex Krizhevsky 的 ConvNet 编号。对于所有实验，我们将整张图像（或裁剪的子窗口）调整为 $221\times221$。这给出了一个 4096 维的向量。我们有两种设置：

- 特征向量进一步进行 L2 归一化到单位长度，用于所有实验。我们将 4096 维的特征向量与支持向量机（SVM）结合使用，以解决不同的分类任务（CNN-SVM）。
- 我们通过添加裁剪和旋转样本以及进行逐分量的幂变换来进一步增广训练集，并报告单独的结果（CNNaug+SVM）。

对于标签不是互斥的分类场景（例如 VOC 目标分类或 UIUC 目标属性），我们使用一对多的策略；在其余实验中，我们使用一对一线性 SVM 进行投票。对于所有实验，我们使用从公式 (1) 中找到的线性 SVM，其中训练数据为 $\{(x_i, y_i)\}$。

$$
\operatorname{minimize}_{w} \frac{1}{2} \|w\|^2 + C \sum_i \max(1 - y_i w^T x_i, 0) \qquad (1)
$$

更多信息请参见第 3.6 节的实现细节。

### 3.2. 图像分类

首先，我们采用 CNN 表示来处理目标和场景的图像分类问题。系统应该为图像分配（可能多个）语义标签。请记住，与目标检测不同，目标图像分类不需要定位目标。CNN 表示已针对 ILSVRC 的目标图像分类任务进行了优化。因此，在本实验中，该表示与最终任务的对齐程度比其余实验更高。然而，我们选择了两个不同的图像分类数据集（目标和室内场景），它们的图像分布与 ILSVRC 数据集不同。

#### 3.2.1 数据集

我们使用两个具有挑战性的识别数据集，即用于目标图像分类的 Pascal VOC 2007 [12] 和用于场景识别的 MIT-67 室内场景 [36]。

**Pascal VOC。** Pascal VOC 2007 [12] 包含约 10000 张图像，涵盖 20 个类别，包括动物、人造物体和自然物体。目标不是居中的，总体而言，VOC 中目标的表观被认为比 ILSVRC 更具挑战性。Pascal VOC 图像带有边界框标注，但在我们的实验中并未使用。

**MIT-67 室内场景。** MIT 场景数据集包含 67 个室内场景类别的 15620 张图像。该数据集包括不同类型的商店（例如面包店、杂货店）、住宅房间（例如婴儿房、卧室）、公共空间（例如公交车内部、图书馆、监狱牢房）、休闲场所（例如自助餐厅、快餐店、酒吧、电影院）和工作场所（例如办公室、手术室、电视演播室）。不同室内场景中存在相似的目标，这使得 MIT 室内数据集相比于室外场景数据集尤其困难。

#### 3.2.2 PASCAL VOC 目标分类结果

表 1 展示了 OverFeat CNN 表示在目标图像分类上的结果。性能使用 VOC 2007 [12] 的平均精度（AP）标准进行衡量。由于原始表示是针对相同任务（在 ILSVRC 上）训练的，我们预计结果会相对较高。我们仅与那些使用了标准 Pascal VOC 2007 数据集之外训练数据的方法进行比较。可以看到，该方法在平均精度均值（mAP）上显著优于以往的所有方法。此外，它在 20 个类别中的 10 个上具有更优的平均精度。值得一提的是，表 1 中的基线方法使用了复杂的匹配系统。最近另一项工作 [29] 中也得出了同样的观察结果。

**不同层。** 直观上可以推断，更深层的学习权重可能对训练数据集的图像及其训练任务变得更加特异。因此，可以想象每个问题的最优表示位于网络的中间层。为了进一步研究这一点，我们使用每个网络层的输出为所有类别训练了一个线性 SVM。结果如图 2a 所示。除了最后两个全连接层外，性能持续增加。我们在各个类别的图中观察到了相同的趋势。中间层（例如第 4、8 层等）的微小下降是由于"ReLU"层对信号进行了半波整流。虽然这有助于 CNN 中训练模型的非线性，但如果直接用于分类则没有帮助。

<div>
<pre>
表 1：Pascal VOC 2007 图像分类结果，与其他也使用 VOC 外部训练数据的方法进行比较。CNN 表示未针对 Pascal VOC 数据集进行调优。然而，GHM [8] 从 VOC 学习视觉词袋和上下文信息的联合表示。AGS [11] 通过将 VOC 数据聚类为子类别来学习第二层表示。NUS [39] 从 VOC 数据集为 SIFT、HOG 和 LBP 描述符训练码本。Oquab 等人 [29] 固定所有在 ImageNet 上训练的层，然后在 VOC 数据集上添加并优化了两个全连接层，取得了更好的结果 (77.7)，表明通过进一步使表示适应目标任务/数据集来提升性能的潜力。
</pre>
</div>

|           | aero | bike | bird | boat | bottle | bus | car | cat | chair | cow | table | dog | horse | mbike | person | plant | sheep | sofa | train | tv | mAP |
|-----------|------|------|------|------|--------|-----|-----|-----|-------|-----|-------|-----|-------|-------|--------|-------|-------|------|-------|----|-----|
| GHM [8]   | 76.7 | 74.7 | 53.8 | 72.1 | 40.4   | 71.7| 83.6| 66.5| 52.5  | 57.5| 62.8  | 51.1| 81.4  | 71.5  | 86.5   | 36.4  | 55.3  | 60.6 | 80.6  | 57.8| 64.7|
| AGS [11]  | 82.2 | 83.0 | 58.4 | 76.1 | 56.4   | 77.5| 88.8| 69.1| 62.2  | 61.8| 64.2  | 51.3| 85.4  | 80.2  | 91.1   | 48.1  | 61.7  | 67.7 | 86.3  | 70.9| 71.1|
| NUS [39]  | 82.5 | 79.6 | 64.8 | 73.4 | 54.2   | 75.0| 77.5| 79.2| 46.2  | 62.7| 41.4  | 74.6| 85.0  | 76.8  | 91.1   | 53.9  | 61.0  | 67.5 | 83.6  | 70.6| 70.5|
| CNN-SVM   | 88.5 | 81.0 | 83.5 | 82.0 | 42.0   | 72.5| 85.3| 81.6| 59.9  | 58.5| 66.5  | 77.8| 81.8  | 78.8  | 90.2   | 54.8  | 71.1  | 62.6 | 87.2  | 71.8| 73.9|
| CNNaug-SVM| 90.1 | 84.4 | 86.5 | 84.1 | 48.4   | 73.4| 86.7| 85.4| 61.3  | 67.6| 69.6  | 84.0| 85.4  | 80.0  | 92.0   | 56.9  | 76.7  | 67.3 | 89.1  | 74.9| 77.2|

#### 3.2.3 MIT 67 场景分类结果

表 2 展示了不同方法在 MIT 室内数据集上的结果。性能通过不同类别的平均分类精度（混淆矩阵对角线的均值）来衡量。使用现成的 CNN 表示结合线性 SVM 训练显著优于大多数基线方法。非 CNN 基线方法受益于广泛而复杂的设计。图 2b 显示了 CNN-SVM 分类器在 67 个 MIT 类别上的混淆矩阵，其对角线非常强。少数相对明亮的非对角线点用其真实标签和估计标签进行了标注。可以看到，在这些例子中，即使是人类也可能难以区分这两个标签，尤其是场景的特写视图。

| 方法 | 平均精度 |
|------|---------|
| ROI + Gist [36] | 26.1 |
| DPM [30] | 30.4 |
| Object Bank [24] | 37.6 |
| RBow [31] | 37.9 |
| BoP [21] | 46.1 |
| miSVM [25] | 46.4 |
| D-Parts [40] | 51.4 |
| IFV [21] | 60.8 |
| MLrep [9] | 64.0 |
| CNN-SVM | 58.4 |
| CNNaug-SVM | 69.0 |
| CNN(AlexConvNet)+multiscale pooling [16] | 68.9 |

表 2：MIT-67 室内场景数据集。MLrep [9] 有一个精心调优的流程，需要数周时间选择和训练各种部件检测器。此外，改进的 Fisher 向量（IFV）表示的维数超过 20 万。[16] 最近调优了一种多尺度无序池化的 CNN 特征（现成的），适用于某些任务。通过这一简单修改，他们实现了 68.88 的显著平均分类精度。

![图 2](a) 当我们使用来自在 ILSVRC 数据集上训练的 OverFeat CNN 的更深层表示时，PASCAL VOC 2007 类别上的平均图像分类 AP 的变化。OverFeat 将卷积、最大池化、非线性激活等视为单独的层。图中反复出现的下降是激活函数层，它通过对信号进行半波整流而丢失信息。(b) MIT-67 室内数据集的混淆矩阵。标注了一些非对角线的混淆类别，这些特殊情况即使是人类也可能难以区分。

### 3.3. 目标检测

遗憾的是，我们尚未进行使用现成 CNN 特征进行目标检测的实验。但值得一提的是，Girshick 等人 [15] 报告了使用来自 Caffe 代码的现成特征在 PASCAL VOC 2007 上的显著数值。我们将他们的相关结果重复如下。使用现成特征，他们实现了 46.2 的 mAP，已经比最先进方法高出约 10%。这进一步证明了现成 CNN 特征在视觉识别任务上的强大能力。

最后，通过进一步针对 PASCAL VOC 2007 数据集微调表示（不再是现成的），他们取得了 53.1 的惊人结果。

### 3.4. 细粒度识别

细粒度识别近年来因其在商业和分类应用中的巨大潜力而变得流行。细粒度识别特别有趣，因为它涉及识别同一目标类别的子类，例如不同的鸟类、狗品种、花卉类型等。许多带有细粒度标注的新数据集的出现，如 Oxford 花卉 [27]、Caltech 鸟类 [45]、狗品种 [1]、烹饪活动 [37]、猫和狗 [32]，帮助该领域迅速发展。不同下属类别之间（相对于不同类别）差异的微妙性需要精细的表示。这一特性使得细粒度识别成为检验通用表示能否捕捉这些微妙细节的良好测试。

#### 3.4.1 数据集

我们在两个细粒度识别数据集上评估 CNN 特征：CUB 200-2011 和 102 Flowers。

选择 Caltech-UCSD Birds (CUB) 200-2011 数据集 [45] 是因为许多近期方法都在其上报告了性能。它包含 200 种鸟类的 11788 张图像。5994 张用于训练，5794 张用于评估。数据集中的许多物种表现出极其微妙的差异，有时甚至人类也难以区分。该数据集提供了多个级别的标注——鸟的边界框、15 个部位关键点、312 个二值属性和边界分割。大多数应用的方法在训练中使用边界框和部位关键点。在本工作中，我们只在训练和测试中使用边界框标注。

Oxford 102 花卉数据集 [27] 包含 102 个类别。每个类别包含 40 到 258 张图像。花卉以不同的尺度、姿态和光照条件出现。此外，该数据集为所有图像提供了分割。

#### 3.4.2 结果

表 3 报告了 CNN-SVM 与 CUB 200-2011 数据集上表现最好的基线方法的比较结果。表中的前两个条目表示仅使用边界框标注的方法。其余基线方法在训练时使用部位标注，有时在评估时也使用。

| 方法 | 部位信息 | 平均精度 |
|------|---------|---------|
| Sift+Color+SVM [45] | ✗ | 17.3 |
| Pose pooling kernel [49] | ✓ | 28.2 |
| RF [47] | ✓ | 19.2 |
| DPD [50] | ✓ | 51.0 |
| Poof [5] | ✓ | 56.8 |
| CNN-SVM | ✗ | 53.3 |
| CNNaug-SVM | ✗ | 61.8 |
| DPD+CNN(DeCaf)+LogReg [10] | ✓ | 65.0 |

表 3：CUB 200-2011 鸟类数据集上的结果。该表区分了使用部位标注进行训练（有时也用于评估）的方法和不使用的方法。[10] 使用 DPD [50] 检测器生成了姿态归一化的 CNN 表示，将结果显著提升至 64.96。

表 4 显示了 CNN-SVM 和其他基线方法在花卉数据集上的性能。除 CNN-SVM 外，所有方法都使用了花卉与背景的分割。可以看出，即使不使用分割，CNN-SVM 也优于所有基本表示及其多核组合。

| 方法 | 平均精度 |
|------|---------|
| HSV [27] | 43.0 |
| SIFT internal [27] | 55.1 |
| SIFT boundary [27] | 32.0 |
| HOG [27] | 49.6 |
| HSV+SIFTi+SIFTb+HOG(MKL) [27] | 72.8 |
| BOW(4000) [14] | 65.5 |
| SPM(4000) [14] | 67.4 |
| FLH(100) [14] | 72.7 |
| BiCos seg [7] | 79.4 |
| Dense HOG+Coding+Pooling [2] w/o seg | 76.7 |
| Seg+Dense HOG+Coding+Pooling [2] | 80.7 |
| CNN-SVM w/o seg | 74.7 |
| CNNaug-SVM w/o seg | 86.8 |

表 4：Oxford 102 花卉数据集上的结果。除非另有说明，所有方法都使用分割将花卉从背景中分离出来。

### 3.5. 属性检测

在计算机视觉的背景下，属性被定义为不同实例/类别共享的某些语义或抽象质量。

#### 3.5.1 数据集

我们使用两个数据集进行属性检测。第一个数据集是 UIUC 64 目标属性数据集 [13]。该数据集中有 3 类属性：形状（例如是 2D 盒状的）、部位（例如有头部）或材质（例如是毛茸茸的）。第二个数据集是 H3D 数据集 [6]，它为 Pascal VOC 2007 中人物图像的一个子集定义了 9 个属性。这些属性涵盖从"戴眼镜"到"是男性"等。

#### 3.5.2 结果

表 5 比较了 CNN 特征与最先进方法的性能。报告了类别内和跨类别属性检测的结果（详情参见 [13]）。

| 方法 | 类别内 | 跨类别 | mAUC |
|------|--------|--------|------|
| Farhadi et al. [13] | 83.4 | - | 73.0 |
| Latent Model [46] | 62.2 | 79.9 | - |
| Sparse Representation [44] | 89.6 | 90.2 | - |
| att. based classification [23] | - | - | 73.7 |
| CNN-SVM | 91.7 | 82.2 | 89.0 |
| CNNaug-SVM | 93.7 | 84.9 | 91.5 |

表 5：UIUC 64 目标属性数据集结果。与现有其他方法相比，CNN 特征表现非常有利。

表 6 报告了 H3D 数据集上 9 个人体属性检测的结果，包括 poselets 和 DPD [50]。Poselets 和 DPD 在训练期间都使用部位级别的标注，而对于 CNN，我们仅从人物周围的边界框提取一个特征。CNN 表示的表现与 DPD 相当，并显著优于 poselets。

| 方法 | male | lg hair | glasses | hat | tshirt | lg slvs | shorts | jeans | lg pants | mAP |
|------|------|---------|---------|-----|--------|---------|--------|-------|----------|-----|
| Freq [6] | 59.3 | 30.0 | 22.0 | 16.6 | 23.5 | 49.0 | 17.9 | 33.8 | 74.7 | 36.3 |
| SPM [6] | 68.1 | 40.0 | 25.9 | 35.3 | 30.6 | 58.0 | 31.4 | 39.5 | 84.3 | 45.9 |
| Poselets [6] | 82.4 | 72.5 | 55.6 | 60.1 | 51.2 | 74.2 | 45.5 | 54.7 | 90.3 | 65.2 |
| DPD [50] | 83.7 | 70.0 | 38.1 | 73.4 | 49.8 | 78.1 | 64.1 | 78.1 | 93.5 | 69.9 |
| CNN-SVM | 83.0 | 67.6 | 39.7 | 66.8 | 52.6 | 82.2 | 78.2 | 71.7 | 95.2 | 70.8 |
| CNNaug-SVM | 84.8 | 71.0 | 42.5 | 66.9 | 57.7 | 84.0 | 79.1 | 75.7 | 95.3 | 73.0 |

表 6：H3D 人体属性数据集结果。从人物周围的边界框中提取 CNN 表示。所有其他方法在训练时都需要部位标注。第一行显示了随机分类器的性能。Zhang 等人 [51] 的工作专门为属性检测任务调整了 CNN 架构，取得了 78.98 的 mAP 出色性能。这进一步强调了在有足够计算资源的情况下，针对不同任务调整 CNN 架构的重要性。

### 3.6. 实现细节

对于 CNN-SVM 实验，我们使用 libsvm 的预计算线性核；对于 CNNaug-SVM，我们使用 liblinear 的原求解器（样本数 $\gg$ 维数）。数据增广通过为每个样本生成 16 个表示来完成（原始图像、5 个裁剪、2 个旋转及其镜像）。裁剪的方式是使子窗口包含原始图像面积 $4/9$，从四个角和中心进行裁剪。

我们注意到所有数据集都有以下现象：在测试时，当我们对一个测试图像有多个表示时，对所有响应求和优于取最大响应。在 CNNaug-SVM 中，我们使用有符号的逐分量幂变换，将每个维度提升到 2 次方。对于带有边界框的数据集（即鸟类、H3D），我们将边界框扩大 150% 以包含一些上下文。在实验的早期阶段，我们注意到对于多类学习，一对一方法比结构化 SVM 效果更好。最后，我们注意到使用 imagemagick 库进行图像调整大小与 matlab imresize 函数相比有轻微的负面影响。不同数据集使用的交叉验证 SVM 参数 (C) 如下：VOC2007: 0.2, MIT67: 2, Birds: 2, Flowers: 2, H3D: 0.2, UIUCatt: 0.2。

> 我们系统的详细信息，包括提取的特征、脚本和更新的表格，可在我们的项目页面上找到：http://www.csc.kth.se/cvap/cvg/DL/ots/

## 4. 视觉实例检索

在本节中，我们将 CNN 表示与当前最先进的检索流程进行比较，包括 VLAD [4, 52]、BoW、IFV [33]、Hamming Embedding [17] 和 BoB [3]。与 CNN 表示不同，上述所有方法都使用在与测试数据相似或相同的数据集上训练的字典。为了公平比较，我们只报告维度数量级相关的表示的结果，并排除空间重排序和查询扩展等后处理方法。

### 4.1. 数据集

我们在该领域的五个常见数据集上报告检索结果，如下所示：

**Oxford5k 建筑 [34]** 这是从 flickr 收集的 5063 张参考照片和 55 个不同建筑的查询的集合。从建筑的角度来看，Oxford5k 中的建筑非常相似。因此，对于 CNN 等通用特征来说，这是一个具有挑战性的基准。

**Paris6k 建筑 [35]** 与 Oxford5k 类似，该集合包含来自巴黎的建筑和纪念碑的 55 张查询图像以及 6412 张参考照片。Paris6k 中的地标比 Oxford5k 中的更具多样性。

**Sculptures6k [3]** 该数据集带来了光滑和无纹理物品检索的挑战。它有 70 张查询图像，包含 6340 张参考图像，这些图像被分成训练/测试子集。该数据集上的结果突显了 CNN 特征编码形状的能力。

**Holidays 数据集 [19]** 该数据集包含 1491 张图像，其中 500 张是查询。它包含不同场景、物品和纪念碑的图像。与前三个数据集不同，它展示了一组多样化的图像。对于上述数据集，我们报告 mAP 作为衡量指标。

**UKbench [28]** 一个包含 2250 个物品的数据集，每个物品有四个不同视角的图像。UKbench 为视角变化提供了一个良好的基准。我们报告 UKBench 上前四的召回率作为性能指标。

### 4.2. 方法

与之前的任务类似，我们使用第一个全连接层的 L2 归一化输出作为表示。

**空间搜索。** 感兴趣的目标可以在测试图像和参考图像中以不同的位置和尺度出现，这使得某种形式的空间搜索成为必要。我们粗略的搜索形式如下：对于每张图像，我们在不同位置提取多个不同大小的子块。令 $h$（层数）表示我们提取的不同大小块的数量。在第 $i$ 层，$1 \le i \le h$，我们提取 $i^2$ 个相同大小的重叠子块，这些子块的并集覆盖整张图像。对于每个提取的子块，我们计算其 CNN 表示。查询子块与参考图像之间的距离定义为查询子块与相应参考子块之间的最小 L2 距离。然后，参考图像与查询图像之间的距离设置为每个查询子块到参考图像的平均距离。与视觉分类流程不同，我们从包含感兴趣区域的最小正方形中提取特征（而不是调整大小）。在后续文本中，$h_r$ 表示参考图像的层数，类似地 $h_q$ 表示查询图像的层数。

**特征增广。** 成功的实例检索方法有许多特征处理步骤。采用 [18] 提出的流程（并被他人在 [16, 42] 中沿用），我们按以下方式处理提取的 4096 维特征：L2 归一化 \\to PCA 降维 \\to 白化 \\to L2 重新归一化。最后，我们进一步使用有符号的逐分量幂变换，将特征向量的每个维度提升到 2 次方。对于所有数据集，在 PCA 步骤中，我们将特征向量的维度降至 500。所有 L2 归一化都用于实现单位长度。

### 4.3. 结果

不同检索方法在 5 个数据集上的结果见表 7。空间搜索仅用于前三个数据集（其样本具有不同的尺度和位置）。对于其他两个数据集，我们使用与第 3.1 节中解释的相同的 jittering。

需要强调的是，我们只报告了低内存占用方法的结果。

| 方法 | Dim | Oxford5k | Paris6k | Sculp6k | Holidays | UKBench |
|------|-----|----------|---------|---------|----------|---------|
| BoB [3] | N/A | N/A | N/A | 45.4 [3] | N/A | N/A |
| BoW | 200k | 36.4 [20] | 46.0 [35] | 8.1 [3] | 54.0 [4] | 70.3 [20] |
| IFV [33] | 2k | 41.8 [20] | - | - | 62.6 [20] | 83.8 [20] |
| VLAD [4] | 32k | 55.5 [4] | - | - | 64.6 [4] | - |
| CVLAD [52] | 64k | 47.8 [52] | - | - | 81.9 [52] | 89.3 [52] |
| HE+burst [17] | 64k | 64.5 [42] | - | - | 78.0 [42] | - |
| AHE+burst [17] | 64k | 66.6 [42] | - | - | 79.4 [42] | - |
| Fine vocab [26] | 64k | 74.2 [26] | 74.9 [26] | - | 74.9 [26] | - |
| ASMK*+MA [42] | 64k | 80.4 [42] | 77.0 [42] | - | 81.0 [42] | - |
| ASMK+MA [42] | 64k | 81.7 [42] | 78.2 [42] | - | 82.2 [42] | - |
| CNN | 4k | 32.2 | 49.5 | 24.1 | 64.2 | 76.0 |
| CNN-ss | 32-120k | 55.6 | 69.7 | 31.1 | 76.9 | 86.9 |
| CNNaug-ss | 4-15k | 68.0 | 79.5 | 42.3 | 84.3 | 91.1 |
| CNN+BOW [16] | 2k | - | - | - | 80.2 | - |

表 7：在 5 个数据集上的目标检索结果。除 CNN 外，所有方法的表示都是在与报告结果的数据集相似的数据集上训练的。Oxford5k、Paris6k 和 Sculpture6k 的空间搜索结果在 $h_r = 4$ 和 $h_q = 3$ 时报告。可以看出，CNN 特征在低内存占用方法中产生了一致的高结果。ASMK+MA [42] 和 fine-vocab [26] 使用了数百万量级的码本，但通过包括二值化在内的各种技巧，将内存占用降至 64k。

## 5. 结论

在本工作中，我们使用现成的 CNN 表示 OverFeat 结合简单分类器来处理不同的识别任务。预训练的 CNN 模型最初是为 ILSVRC 2013 数据集中的目标分类任务优化的。然而，它证明了自己是更复杂且精心调优的最先进方法的有力竞争者。在各种识别任务和不同数据集中都观察到了相同的趋势，这突显了学习得到的表示的有效性和通用性。实验证实并扩展了 [10] 中报告的结果。我们还指出了那些专门针对不同任务/数据集优化 CNN 表示的工作取得了更优的结果。因此，可以得出结论，从现在开始，基于 CNN 的深度学习必须被视为基本上任何视觉识别任务的首选方案。

## 致谢

我们衷心感谢 NVIDIA 公司为本研究捐赠 Tesla K40 GPU。我们还要感谢 Dr. Atsuto Maki、Dr. Pierre Sermanet、Dr. Ross Girshick 和 Dr. Relja Arandjelović 提出的宝贵意见。

## 参考文献

[1] Imagenet large scale visual recognition challenge 2013 (ilsvrc2013). http://www.image-net.org/challenges/LSVRC/2013/.

[2] A. Angelova and S. Zhu. Efficient object detection and segmentation for fine-grained recognition. In *CVPR*, 2013.

[3] R. Arandjelović and A. Zisserman. Smooth object retrieval using a bag of boundaries. In *ICCV*, 2011.

[4] R. Arandjelović and A. Zisserman. All about VLAD. In *CVPR*, 2013.

[5] T. Berg and P. N. Belhumeur. Poof: Part-based one-vs.-one features for fine-grained categorization, face verification, and attribute estimation. In *CVPR*, 2013.

[6] L. D. Bourdev, S. Maji, and J. Malik. Describing people: A poselet-based approach to attribute classification. In *ICCV*, 2011.

[7] Y. Chai, V. S. Lempitsky, and A. Zisserman. Bicos: A bi-level co-segmentation method for image classification. In *ICCV*, 2011.

[8] Q. Chen, Z. Song, Y. Hua, Z. Huang, and S. Yan. Hierarchical matching with side information for image classification. In *CVPR*, 2012.

[9] C. Doersch, A. Gupta, and A. A. Efros. Mid-level visual element discovery as discriminative mode seeking. In *NIPS*, 2013.

[10] J. Donahue, Y. Jia, O. Vinyals, J. Hoffman, N. Zhang, E. Tzeng, and T. Darrell. Decaf: A deep convolutional activation feature for generic visual recognition. In *ICML*, 2014.

[11] J. Dong, W. Xia, Q. Chen, J. Feng, Z. Huang, and S. Yan. Subcategory-aware object classification. In *CVPR*, 2013.

[12] M. Everingham, L. Van Gool, C. K. I. Williams, J. Winn, and A. Zisserman. The PASCAL Visual Object Classes Challenge 2012 (VOC2012) Results. http://www.pascal-network.org/challenges/VOC/voc2012/workshop/index.html.

[13] A. Farhadi, I. Endres, D. Hoiem, and D. A. Forsyth. Describing objects by their attributes. In *CVPR*, 2009.

[14] B. Fernando, E. Fromont, and T. Tuytelaars. Mining mid-level features for image classification. *International Journal of Computer Vision*, 2014.

[15] R. B. Girshick, J. Donahue, T. Darrell, and J. Malik. Rich feature hierarchies for accurate object detection and semantic segmentation. *arXiv:1311.2524 [cs.CV]*, 2013.

[16] Y. Gong, L. Wang, R. Guo, and S. Lazebnik. Multi-scale orderless pooling of deep convolutional activation features. *CoRR*, 2014.

[17] M. Jain, H. Jégou, and P. Gros. Asymmetric hamming embedding: taking the best of our bits for large scale image search. In *ACM Multimedia*, pages 1441–1444, 2011.

[18] H. Jégou and O. Chum. Negative evidences and co-occurences in image retrieval: The benefit of pca and whitening. In *ECCV*, pages 774–787, 2012.

[19] H. Jégou, M. Douze, and C. Schmid. Hamming embedding and weak geometric consistency for large scale image search. In *ECCV*, 2008.

[20] H. Jégou, F. Perronnin, M. Douze, J. Sánchez, P. Pérez, and C. Schmid. Aggregating local image descriptors into compact codes. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 34(9):1704–1716, 2012.

[21] M. Juneja, A. Vedaldi, C. V. Jawahar, and A. Zisserman. Blocks that shout: Distinctive parts for scene classification. In *CVPR*, 2013.

[22] A. Krizhevsky, I. Sutskever, and G. E. Hinton. Imagenet classification with deep convolutional neural networks. In *NIPS*, 2012.

[23] C. H. Lampert, H. Nickisch, and S. Harmeling. Attribute-based classification for zero-shot visual object categorization. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 36(3), 2014.

[24] L.-J. Li, H. Su, E. P. Xing, and F.-F. Li. Object bank: A high-level image representation for scene classification & semantic feature sparsification. In *NIPS*, 2010.

[25] Q. Li, J. Wu, and Z. Tu. Harvesting mid-level visual concepts from large-scale internet images. In *CVPR*, 2013.

[26] A. Mikulík, M. Perdoch, O. Chum, and J. Matas. Learning a fine vocabulary. In *ECCV*, pages 1–14, 2010.

[27] M.-E. Nilsback and A. Zisserman. Automated flower classification over a large number of classes. In *Proceedings of the Indian Conference on Computer Vision, Graphics and Image Processing*, Dec 2008.

[28] D. Nistér and H. Stewénius. Scalable recognition with a vocabulary tree. In *CVPR*, 2006.

[29] M. Oquab, L. Bottou, I. Laptev, and J. Sivic. Learning and transferring mid-level image representations using convolutional neural networks. Technical Report HAL-00911179, INRIA, 2013.

[30] M. Pandey and S. Lazebnik. Scene recognition and weakly supervised object localization with deformable part-based models. In *ICCV*, 2011.

[31] S. N. Parizi, J. G. Oberlin, and P. F. Felzenszwalb. Configurable models for scene recognition. In *CVPR*, 2012.

[32] O. M. Parkhi, A. Vedaldi, A. Zisserman, and C. V. Jawahar. Cats and dogs. In *CVPR*, 2012.

[33] F. Perronnin, Y. Liu, J. Sánchez, and H. Poirier. Large-scale image retrieval with compressed fisher vectors. In *CVPR*, 2010.

[34] J. Philbin, O. Chum, M. Isard, J. Sivic, and A. Zisserman. Object retrieval with large vocabularies and fast spatial matching. In *CVPR*, 2007.

[35] J. Philbin, O. Chum, M. Isard, J. Sivic, and A. Zisserman. Lost in quantization: Improving particular object retrieval in large scale image databases. In *CVPR*, 2008.

[36] A. Quattoni and A. Torralba. Recognizing indoor scenes. In *CVPR*, 2009.

[37] M. Rohrbach, S. Amin, M. Andriluka, and B. Schiele. A database for fine grained activity detection of cooking activities. In *CVPR*, 2012.

[38] P. Sermanet, D. Eigen, X. Zhang, M. Mathieu, R. Fergus, and Y. LeCun. Overfeat: Integrated recognition, localization and detection using convolutional networks. In *ICLR*, 2014.

[39] Z. Song, Q. Chen, Z. Huang, Y. Hua, and S. Yan. Contextualizing object detection and classification. In *CVPR*, 2011.

[40] J. Sun and J. Ponce. Learning discriminative part detectors for image classification and cosegmentation. In *ICCV*, 2013.

[41] Y. Taigman, M. Yang, M. Ranzato, and L. Wolf. Deepface: Closing the gap to human-level performance in face verification. In *CVPR*, 2014.

[42] G. Tolias, Y. S. Avrithis, and H. Jégou. To aggregate or not to aggregate: Selective match kernels for image search. In *ICCV*, pages 1401–1408, 2013.

[43] A. Toshev and C. Szegedy. Deeppose: Human pose estimation via deep neural networks. In *CVPR*, 2014.

[44] G. Tsagkatakis and A. E. Savakis. Sparse representations and distance learning for attribute based category recognition. In *ECCV Workshops (1)*, pages 29–42, 2010.

[45] C. Wah, S. Branson, P. Welinder, P. Perona, and S. Belongie. The Caltech-UCSD Birds-200-2011 Dataset. Technical Report CNS-TR-2011-001, California Institute of Technology, 2011.

[46] Y. Wang and G. Mori. A discriminative latent model of object classes and attributes. In *ECCV*, 2010.

[47] B. Yao, A. Khosla, and F.-F. Li. Combining randomization and discrimination for fine-grained image categorization. In *CVPR*, 2011.

[48] M. D. Zeiler and R. Fergus. Visualizing and understanding convolutional networks. *CoRR*, abs/1311.2901, 2013.

[49] N. Zhang, R. Farrell, and T. Darrell. Pose pooling kernels for sub-category recognition. In *CVPR*, 2012.

[50] N. Zhang, R. Farrell, F. Iandola, and T. Darrell. Deformable part descriptors for fine-grained recognition and attribute prediction. In *ICCV*, 2013.

[51] N. Zhang, M. Paluri, M. Ranzato, T. Darrell, and L. Bourdev. Panda: Pose aligned networks for deep attribute modeling. In *CVPR*, 2014.

[52] W.-L. Zhao, H. Jégou, G. Gravier, et al. Oriented pooling for dense and non-dense rotation-invariant features. In *BMVC*, 2013.
