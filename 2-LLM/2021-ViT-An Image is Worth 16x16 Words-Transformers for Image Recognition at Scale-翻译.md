# AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE

> Alexey Dosovitskiy\*,†, Lucas Beyer\*, Alexander Kolesnikov\*, Dirk Weissenborn\*, Xiaohua Zhai\*, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby\*,†
> \*同等技术贡献, †同等指导
> Google Research, Brain Team
> {adosovitskiy, neilhoulsby}@google.com

本文证明，直接将纯Transformer应用于图像块序列可以在图像分类任务上取得出色表现。当在大规模数据上预训练并迁移到多个中小规模图像识别基准（ImageNet、CIFAR-100、VTAB等）时，Vision Transformer（ViT）相比最先进的卷积网络在取得优异结果的同时，所需的训练计算资源大幅减少。

---

## 摘要

虽然Transformer架构已成为自然语言处理任务的事实标准，但其在计算机视觉中的应用仍然有限。在视觉领域，注意力要么与卷积网络结合使用，要么用于替换卷积网络的某些组件同时保留其整体结构。我们表明这种对CNN的依赖并非必要，直接将纯Transformer应用于图像块序列可以在图像分类任务上表现非常出色。当在大规模数据上预训练并迁移到多个中小规模图像识别基准（ImageNet、CIFAR-100、VTAB等）时，Vision Transformer（ViT）在取得与最先进卷积网络相媲美的优异结果的同时，所需的训练计算资源大幅减少。

## 1 引言

基于自注意力的架构，特别是Transformer（Vaswani et al., 2017），已成为自然语言处理（NLP）中的首选模型。主流方法是在大规模文本语料上预训练，然后在较小的任务特定数据集上微调（Devlin et al., 2019）。得益于Transformer的计算效率和可扩展性，训练超过1000亿参数的史无前例规模的模型已成为可能（Brown et al., 2020; Lepikhin et al., 2020）。随着模型和数据集不断增长，性能仍未出现饱和迹象。

然而在计算机视觉领域，卷积架构仍然占据主导地位（LeCun et al., 1989; Krizhevsky et al., 2012; He et al., 2016）。受NLP成功的启发，许多工作尝试将CNN类架构与自注意力相结合（Wang et al., 2018; Carion et al., 2020），有些则完全替换卷积（Ramachandran et al., 2019; Wang et al., 2020a）。后者虽然在理论上高效，但由于使用了专门的注意力模式，尚未在现代硬件加速器上有效地扩展。因此，在大规模图像识别中，经典的ResNet类架构仍是当前最优（Mahajan et al., 2018; Xie et al., 2020; Kolesnikov et al., 2020）。

受Transformer在NLP中扩展成功的启发，我们尝试将标准Transformer直接应用于图像，并做尽可能少的修改。为此，我们将图像分割成块（patch），并将这些块的线性嵌入序列作为Transformer的输入。图像块的处理方式与NLP应用中的token（词）相同。我们以监督方式在图像分类任务上训练该模型。

在中型数据集（如ImageNet）上训练且未使用强正则化时，这些模型的准确率仅比同等规模的ResNet低几个百分点。这一看似令人沮丧的结果是可以预料的：Transformer缺少CNN固有的某些归纳偏置，如平移等变性和局部性，因此在数据量不足时难以很好地泛化。

然而，当模型在更大规模的数据集（1400万至3亿张图像）上训练时，情况发生了变化。我们发现大规模训练胜过了归纳偏置。我们的Vision Transformer（ViT）在足够规模的预训练下，迁移到数据点更少的任务时取得了优异的结果。当在公开的ImageNet-21k数据集或内部的JFT-300M数据集上预训练时，ViT在多个图像识别基准上接近或超越了当前最优。具体而言，最佳模型在ImageNet上达到88.55%的准确率，在ImageNet-ReaL上达到90.72%，在CIFAR-100上达到94.55%，在包含19个任务的VTAB套件上达到77.63%。

## 2 相关工作

Transformer由Vaswani等人（2017）为机器翻译提出，此后成为众多NLP任务中当前最优的方法。基于Transformer的大型模型通常在大规模语料上预训练，然后针对具体任务微调：BERT（Devlin et al., 2019）使用去噪自监督预训练任务，而GPT系列工作使用语言建模作为其预训练任务（Radford et al., 2018; 2019; Brown et al., 2020）。

将自注意力直接应用于图像需要每个像素关注所有其他像素。由于像素数量的二次成本，这无法扩展到实际的输入尺寸。因此，过去尝试了几种近似方法将Transformer应用于图像处理。Parmar等人（2018）将自注意力仅应用于每个查询像素的局部邻域而非全局。这种局部多头点积自注意力块可以完全替换卷积（Hu et al., 2019; Ramachandran et al., 2019; Zhao et al., 2020）。在另一条工作线中，Sparse Transformer（Child et al., 2019）采用可扩展的近似方法实现全局自注意力以适用于图像。另一种扩展注意力的替代方法是将其应用于不同大小的块中（Weissenborn et al., 2019），极端情况下仅沿单个轴应用（Ho et al., 2019; Wang et al., 2020a）。许多这些专门的注意力架构在计算机视觉任务上展示了有前景的结果，但需要在硬件加速器上高效实现所需的复杂工程。

与我们最相关的是Cordonnier等人（2020）的模型，该模型从输入图像中提取$2 \times 2$大小的块并在其上应用全局自注意力。该模型与ViT非常相似，但我们的工作进一步证明了大规模预训练使原始Transformer能够与最先进的CNN竞争（甚至更好）。此外，Cordonnier等人（2020）使用$2 \times 2$像素的小patch尺寸，这使得该模型仅适用于小分辨率图像，而我们也能处理中等分辨率图像。

还有许多工作对将卷积神经网络（CNN）与自注意力形式相结合产生了浓厚兴趣，例如通过增强特征图进行图像分类（Bello et al., 2019），或使用自注意力进一步处理CNN的输出，用于目标检测（Hu et al., 2018; Carion et al., 2020）、视频处理（Wang et al., 2018; Sun et al., 2019）、图像分类（Wu et al., 2020）、无监督目标发现（Locatello et al., 2020）或统一的文本视觉任务（Chen et al., 2020c; Lu et al., 2019; Li et al., 2019）。

另一个近期相关模型是image GPT（iGPT）（Chen et al., 2020a），它在降低图像分辨率和颜色空间后将Transformer应用于图像像素。该模型以无监督方式作为生成模型进行训练，得到的表示可以微调或线性探针用于分类性能，在ImageNet上最高达到72%的准确率。

我们的工作为探索超出标准ImageNet数据集的更大规模图像识别的论文集合增添了新内容。额外数据源的使用使得在标准基准上取得最优结果成为可能（Mahajan et al., 2018; Touvron et al., 2019; Xie et al., 2020）。此外，Sun等人（2017）研究了CNN性能如何随数据集大小扩展，Kolesnikov等人（2020）和Djolonga等人（2020）从大规模数据集（如ImageNet-21k和JFT-300M）对CNN迁移学习进行了实证探索。我们也关注这两个数据集，但训练的是Transformer而非先前工作中使用的基于ResNet的模型。

## 3 方法

在模型设计上，我们尽可能遵循原始Transformer（Vaswani et al., 2017）。这种有意简化的设置的一个优势是可扩展的NLP Transformer架构及其高效实现可以几乎开箱即用。

### 3.1 Vision Transformer（ViT）

模型概述如图1所示。标准Transformer接收1D的token嵌入序列作为输入。为处理2D图像，我们将图像$x \in \mathbb{R}^{H \times W \times C}$重塑为展平的2D块序列$x_p \in \mathbb{R}^{N \times (P^2 \cdot C)}$，其中$(H, W)$是原始图像的分辨率，$C$是通道数，$(P, P)$是每个图像块的分辨率，$N = HW/P^2$是生成的块数，也作为Transformer的有效输入序列长度。Transformer在其所有层中使用恒定的潜在向量大小$D$，因此我们将块展平并用可训练的线性投影映射到$D$维（公式1）。我们将此投影的输出称为块嵌入（patch embeddings）。

类似BERT的[class] token，我们在嵌入块序列前添加一个可学习的嵌入（$z_0^0 = x_{\text{class}}$），其在Transformer编码器输出端的状态（$z_L^0$）作为图像表示$y$（公式4）。在预训练和微调期间，分类头连接在$z_L^0$上。分类头在预训练时由带一个隐藏层的MLP实现，在微调时由单个线性层实现。

位置嵌入被添加到块嵌入中以保留位置信息。我们使用标准的可学习1D位置嵌入，因为我们没有观察到使用更高级的2D感知位置嵌入带来显著的性能提升（附录D.4）。得到的嵌入向量序列作为编码器的输入。

Transformer编码器（Vaswani et al., 2017）由交替的多头自注意力（MSA，见附录A）和MLP块组成（公式2，3）。层归一化（LN）在每个块之前应用，残差连接在每个块之后应用（Wang et al., 2019; Baevski & Auli, 2019）。MLP包含两个层，使用GELU非线性激活。

$$
\begin{aligned}
z_0 &= [x_{\text{class}}; x_p^1 E; x_p^2 E; \cdots ; x_p^N E] + E_{\text{pos}}, \\
&\quad E \in \mathbb{R}^{(P^2 \cdot C) \times D}, \; E_{\text{pos}} \in \mathbb{R}^{(N+1) \times D} \qquad (1) \\
z'_\ell &= \text{MSA}(\text{LN}(z_{\ell-1})) + z_{\ell-1}, \quad \ell = 1 \ldots L \qquad (2) \\
z_\ell &= \text{MLP}(\text{LN}(z'_\ell)) + z'_\ell, \quad \ell = 1 \ldots L \qquad (3) \\
y &= \text{LN}(z_L^0) \qquad (4)
\end{aligned}
$$

**归纳偏置（Inductive bias）**。我们注意到Vision Transformer相比CNN具有更少的图像特定归纳偏置。在CNN中，局部性、二维邻域结构和平移等变性被嵌入到整个模型的每一层中。在ViT中，只有MLP层是局部和平移等变的，而自注意力层是全局的。二维邻域结构的使用非常有限：在模型开始时将图像切割成块，以及在微调时调整位置嵌入以适应不同分辨率的图像（如下所述）。除此之外，初始化时的位置嵌入不携带关于块的二维位置的任何信息，所有块之间的空间关系必须从头学习。

**混合架构（Hybrid Architecture）**。作为原始图像块的替代方案，输入序列可以由CNN的特征图形成（LeCun et al., 1989）。在这种混合模型中，块嵌入投影$E$（公式1）应用于从CNN特征图提取的块。作为一个特例，块的空间大小可以是$1 \times 1$，这意味着输入序列通过简单展平特征图的空间维度并投影到Transformer维度获得。分类输入嵌入和位置嵌入的添加方式如上所述。

### 3.2 微调与更高分辨率

通常，我们在大型数据集上预训练ViT，然后微调到（较小的）下游任务。为此，我们移除预训练预测头并附加一个零初始化的$D \times K$前馈层，其中$K$是下游类别的数量。以比预训练更高分辨率进行微调通常是有益的（Touvron et al., 2019; Kolesnikov et al., 2020）。当输入更高分辨率的图像时，我们保持patch大小不变，这会导致更大的有效序列长度。Vision Transformer可以处理任意序列长度（受限于内存限制），但预训练的位置嵌入可能不再有意义。因此，我们根据预训练位置嵌入在原始图像中的位置进行2D插值。请注意，此分辨率调整和块提取是唯一将关于图像二维结构的归纳偏置手动注入Vision Transformer的点。

## 4 实验

我们评估ResNet、Vision Transformer（ViT）以及混合模型的表示学习能力。为了理解每个模型的数据需求，我们在不同大小的数据集上预训练并在多个基准任务上评估。在考虑模型预训练的计算成本时，ViT表现非常有利，以更低的预训练成本在大多数识别基准上达到了当前最优。最后，我们使用自监督进行了一项小型实验，表明自监督ViT在未来具有潜力。

### 4.1 实验设置

**数据集**。为探索模型的可扩展性，我们使用ILSVRC-2012 ImageNet数据集（1k类，130万张图像，以下称为ImageNet）、其超集ImageNet-21k（21k类，1400万张图像，Deng et al., 2009）以及JFT（Sun et al., 2017，18k类，3.03亿张高分辨率图像）。我们按照Kolesnikov等人（2020）的方法，对下游任务的测试集去重预训练数据集。我们将在这些数据集上训练的模型迁移到多个基准任务：原始验证标签的ImageNet、清理后的ReaL标签（Beyer et al., 2020）、CIFAR-10/100（Krizhevsky, 2009）、Oxford-IIIT Pets（Parkhi et al., 2012）和Oxford Flowers-102（Nilsback & Zisserman, 2008）。对于这些数据集，预处理遵循Kolesnikov等人（2020）。

表1：Vision Transformer模型变体的细节。

| 模型 | 层数 | 隐藏大小$D$ | MLP大小 | 头数 | 参数量 |
|---|---|---|---|---|---|
| ViT-Base | 12 | 768 | 3072 | 12 | 86M |
| ViT-Large | 24 | 1024 | 4096 | 16 | 307M |
| ViT-Huge | 32 | 1280 | 5120 | 16 | 632M |

我们还在包含19个任务的VTAB分类套件上进行了评估（Zhai et al., 2019b）。VTAB评估向多样任务的低数据迁移，每个任务使用1000个训练样本。任务分为三组：Natural（自然）——类似上述任务，如Pets、CIFAR等；Specialized（专业）——医学和卫星图像；Structured（结构化）——需要几何理解的任务，如定位。

**模型变体**。我们基于BERT（Devlin et al., 2019）中使用的配置设定ViT配置，如表1总结。"Base"和"Large"模型直接采用自BERT，我们增加了更大的"Huge"模型。在后续内容中，我们使用简写表示法指示模型大小和输入patch大小：例如，ViT-L/16表示"Large"变体，输入patch大小为$16 \times 16$。注意Transformer的序列长度与patch大小的平方成反比，因此patch较小的模型计算成本更高。

对于基线CNN，我们使用ResNet（He et al., 2016），但将Batch Normalization层（Ioffe & Szegedy, 2015）替换为Group Normalization（Wu & He, 2018），并使用标准化卷积（Qiao et al., 2019）。这些修改改进了迁移性能（Kolesnikov et al., 2020），我们将修改后的模型称为"ResNet (BiT)"。对于混合模型，我们将中间特征图以大小为1个"像素"的块输入ViT。为实验不同的序列长度，我们要么（i）取标准ResNet50第4阶段的输出，要么（ii）移除第4阶段，在第3阶段放置相同数量的层（保持总层数不变），并取此扩展后的第3阶段的输出。选项（ii）产生了4倍更长的序列长度和更昂贵的ViT模型。

**训练与微调**。我们使用Adam（Kingma & Ba, 2015）训练所有模型（包括ResNet），其中$\beta_1 = 0.9$，$\beta_2 = 0.999$，batch大小为4096，并应用0.1的高权重衰减，我们发现这对所有模型的迁移都有用（附录D.1表明，与常见做法相反，在我们的设置中Adam在ResNet上略优于SGD）。我们使用线性学习率预热和衰减，详见附录B.1。对于微调，所有模型使用带动量的SGD，batch大小为512，见附录B.1.1。对于表2中的ImageNet结果，我们在更高分辨率下微调：ViT-L/16为512，ViT-H/14为518，并使用Polyak & Juditsky（1992）平均，因子为0.9999（Ramachandran et al., 2019; Wang et al., 2020b）。

**评估指标**。我们通过few-shot或微调准确率报告下游数据集的结果。微调准确率捕捉每个模型在相应数据集上微调后的性能。Few-shot准确率通过求解一个正则化最小二乘回归问题获得，该问题将（冻结的）训练图像子集的表示映射到$\{-1, 1\}^K$目标向量。这种表述使我们能够以闭合形式恢复精确解。虽然我们主要关注微调性能，但有时我们使用线性few-shot准确率进行快速的即时评估，因为微调成本过高。

### 4.2 与当前最优的比较

我们首先将最大模型——ViT-H/14和ViT-L/16——与文献中最先进的CNN进行比较。第一个比较点是大迁移（BiT）（Kolesnikov et al., 2020），它使用大型ResNet进行监督迁移学习。第二个是Noisy Student（Xie et al., 2020），这是一个使用半监督学习在ImageNet和JFT-300M（去除标签）上训练的大型EfficientNet。目前，Noisy Student是ImageNet上的当前最优，BiT-L是此处报告的其他数据集上的当前最优。所有模型均在TPUv3硬件上训练，我们报告每个模型预训练所需的TPUv3-core-days，即用于训练的TPU v3核心数（每芯片2个）乘以训练天数。

表2展示了结果。在JFT-300M上预训练的较小的ViT-L/16模型在所有任务上都优于BiT-L（在相同数据集上预训练），同时所需的训练计算资源大幅减少。更大的模型ViT-H/14进一步提升了性能，尤其是在更具挑战性的数据集——ImageNet、CIFAR-100和VTAB套件上。有趣的是，该模型预训练所需的计算量仍远小于之前的当前最优。

|  | Ours-JFT (ViT-H/14) | Ours-JFT (ViT-L/16) | Ours-I21k (ViT-L/16) | BiT-L (ResNet152x4) | Noisy Student (EfficientNet-L2) |
|---|---|---|---|---|---|
| ImageNet | $88.55 \pm 0.04$ | $87.76 \pm 0.03$ | $85.30 \pm 0.02$ | $87.54 \pm 0.02$ | $88.4/88.5^*$ |
| ImageNet ReaL | $90.72 \pm 0.05$ | $90.54 \pm 0.03$ | $88.62 \pm 0.05$ | 90.54 | 90.55 |
| CIFAR-10 | $99.50 \pm 0.06$ | $99.42 \pm 0.03$ | $99.15 \pm 0.03$ | $99.37 \pm 0.06$ | $-$ |
| CIFAR-100 | $94.55 \pm 0.04$ | $93.90 \pm 0.05$ | $93.25 \pm 0.05$ | $93.51 \pm 0.08$ | $-$ |
| Oxford-IIIT Pets | $97.56 \pm 0.03$ | $97.32 \pm 0.11$ | $94.67 \pm 0.15$ | $96.62 \pm 0.23$ | $-$ |
| Oxford Flowers-102 | $99.68 \pm 0.02$ | $99.74 \pm 0.00$ | $99.61 \pm 0.02$ | $99.63 \pm 0.03$ | $-$ |
| VTAB (19 tasks) | $77.63 \pm 0.23$ | $76.28 \pm 0.46$ | $72.72 \pm 0.21$ | $76.29 \pm 1.70$ | $-$ |
| TPUv3-core-days | 2.5k | 0.68k | 0.23k | 9.9k | 12.3k |

表2：与流行图像分类基准上的当前最优方法的比较。我们报告三次微调运行的平均准确率和标准差。在JFT-300M数据集上预训练的Vision Transformer模型在所有数据集上都优于基于ResNet的基线，同时预训练所需的计算资源大幅减少。在较小的公开ImageNet-21k数据集上预训练的ViT也表现良好。$^*$Touvron等人（2020）报告的略微改进的88.5%结果。

然而，我们注意到预训练效率不仅可能受架构选择影响，还可能受其他参数（如训练计划、优化器、权重衰减等）影响。我们在第4.4节提供了不同架构在性能与计算之间的受控研究。最后，在公开ImageNet-21k数据集上预训练的ViT-L/16模型在大多数数据集上也表现良好，同时预训练所需资源更少：使用标准8核云端TPUv3约需30天。

图2将VTAB任务分解到各自的组，并与该基准上之前的SOTA方法进行比较：BiT、VIVI（一个在ImageNet和Youtube上共同训练的ResNet，Tschannen et al., 2020）和S4L（在ImageNet上的监督加半监督学习，Zhai et al., 2019a）。ViT-H/14在Natural和Structured任务上优于BiT-R152x4和其他方法。在Specialized任务上，前两名模型的性能相似。

![图2：VTAB性能在Natural、Specialized和Structured任务组中的分解图。](../figures/vit_fig2.png)
图2：VTAB性能在Natural、Specialized和Structured任务组中的分解。

### 4.3 预训练数据需求

Vision Transformer在大型JFT-300M数据集上预训练时表现良好。在视觉归纳偏置少于ResNet的情况下，数据集大小有多关键？我们进行了两系列实验。

首先，我们在递增大小的数据集上预训练ViT模型：ImageNet、ImageNet-21k和JFT-300M。为提升在较小数据集上的性能，我们优化了三个基本正则化参数——权重衰减、dropout和标签平滑。图3显示了微调到ImageNet后的结果（其他数据集的结果见表5）[^2]。当在最小的数据集ImageNet上预训练时，尽管有（适度的）正则化，ViT-Large模型的表现仍不如ViT-Base模型。使用ImageNet-21k预训练后，它们的性能相似。只有在JFT-300M上，我们才看到更大模型的全部优势。图3还显示了不同大小的BiT模型跨越的性能区域。BiT CNN在ImageNet上优于ViT，但在更大数据集上，ViT反超。

![图3：迁移到ImageNet。当在小数据集上预训练时，大型ViT模型表现不如BiT ResNet（阴影区域），但在更大数据集上预训练时它们表现出色。类似地，随着数据集增长，更大的ViT变体超越较小的变体。](../figures/vit_fig3.png)
图3：迁移到ImageNet。当在小数据集上预训练时，大型ViT模型表现不如BiT ResNet（阴影区域），但在更大数据集上预训练时它们表现出色。类似地，随着数据集增长，更大的ViT变体超越较小的变体。

其次，我们在9M、30M和90M的随机子集以及完整的JFT-300M数据集上训练模型。我们不在较小子集上执行额外的正则化，并对所有设置使用相同的超参数。这样，我们评估的是模型的内在属性，而非正则化的效果。但我们使用了早停法，并报告训练期间达到的最佳验证准确率。为节省计算量，我们报告few-shot线性准确率而非完整的微调准确率。图4展示了结果。在较小数据集上，Vision Transformer比具有相当计算成本的ResNet更容易过拟合。例如，ViT-B/32略快于ResNet50；它在9M子集上表现差得多，但在90M+子集上表现更好。ResNet152x2和ViT-L/16也是如此。这一结果强化了直观认识：卷积归纳偏置在较小数据集上有用，但对于较大数据集，直接从数据中学习相关模式就足够了，甚至更有益。

![图4：ImageNet上线性few-shot评估与预训练大小的关系。ResNet在较小的预训练数据集上表现更好，但比ViT更早达到平台期，ViT在更大的预训练上表现更好。ViT-b是ViT-B所有隐藏维度减半的版本。](../figures/vit_fig4.png)
图4：ImageNet上线性few-shot评估与预训练大小的关系。ResNet在较小的预训练数据集上表现更好，但比ViT更早达到平台期，ViT在更大的预训练上表现更好。ViT-b是ViT-B所有隐藏维度减半的版本。

总体而言，ImageNet上的few-shot结果（图4）以及VTAB上的低数据结果（表2）对于极低数据迁移似乎很有前景。进一步分析ViT的few-shot属性是一个令人兴奋的未来研究方向。

[^2]: 注意，ImageNet预训练模型也被微调了，但还是在ImageNet上。这是因为微调期间的分辨率提升提高了性能。

### 4.4 扩展性研究

我们通过评估从JFT-300M迁移的性能，对不同模型进行受控的扩展性研究。在此设置中，数据大小不会成为模型性能的瓶颈，我们评估每个模型的性能与预训练成本的关系。模型集合包括：7个ResNet（R50x1、R50x2、R101x1、R152x1、R152x2，预训练7个epochs，加上预训练14个epochs的R152x2和R200x3）；6个Vision Transformer（ViT-B/32、B/16、L/32、L/16，预训练7个epochs，加上预训练14个epochs的L/16和H/14）；以及5个混合模型（R50+ViT-B/32、B/16、L/32、L/16，预训练7个epochs，加上预训练14个epochs的R50+ViT-L/16）。对于混合模型，模型名称末尾的数字代表ResNet主干中的总下采样比率，而非patch大小。

![图5：不同架构（Vision Transformer、ResNet和混合模型）的性能与预训练计算量的关系。Vision Transformer通常在相同计算预算下优于ResNet。混合模型在较小模型尺寸上相比纯Transformer有所改进，但差距在较大模型上消失。](../figures/vit_fig5.png)
图5：不同架构（Vision Transformer、ResNet和混合模型）的性能与预训练计算量的关系。Vision Transformer通常在相同计算预算下优于ResNet。混合模型在较小模型尺寸上相比纯Transformer有所改进，但差距在较大模型上消失。

图5展示了迁移性能与总预训练计算量的关系（计算成本的详细信息见附录D.5）。每个模型的详细结果见附录中的表6。可以观察到几个模式。首先，Vision Transformer在性能/计算权衡上优于ResNet。ViT达到相同性能（5个数据集的平均）所需的计算量约为$2-4\times$更少。其次，混合模型在较小的计算预算下略优于ViT，但差异在更大模型上消失。这一结果有些令人惊讶，因为人们可能期望卷积局部特征处理在任何规模下都能辅助ViT。第三，Vision Transformer在尝试的范围内似乎尚未饱和，这激励了未来的扩展努力。

### 4.5 探究Vision Transformer

![图6：从输出token到输入空间的注意力代表性示例。详见附录D.7。](...)

为了开始理解Vision Transformer如何处理图像数据，我们分析了其内部表示。Vision Transformer的第一层将展平的块线性投影到低维空间（公式1）。图7（左）显示了学习到的嵌入滤波器的主要主成分。这些成分类似于对每个块内精细结构进行低维表示的合理基函数。

投影之后，学习到的位置嵌入被添加到块表示中。图7（中）显示模型学会了在位置嵌入的相似度中编码图像内的距离，即更接近的块往往具有更相似的位置嵌入。此外，行列结构出现；同一行/列中的块具有相似的嵌入。最后，对于更大的网格有时会出现正弦结构（附录D）。位置嵌入学会表示2D图像拓扑这一事实解释了为什么手工设计的2D感知嵌入变体不会带来改进（附录D.4）。

自注意力允许ViT即使在最低层也能整合整个图像的信息。我们研究了网络在多大程度上利用了这一能力。具体来说，我们基于注意力权重计算信息在图像空间中整合的平均距离（图7，右）。这个"注意力距离"类似于CNN中的感受野大小。我们发现，某些头在最低层就已经关注到图像的大部分区域，表明模型确实使用了全局信息整合的能力。其他注意力头在低层具有一致较小的注意力距离。这种高度局部化的注意力在Transformer之前应用ResNet的混合模型中不太明显（图7，右），表明它可能起到类似CNN中早期卷积层的作用。此外，注意力距离随网络深度增加。从全局来看，我们发现模型关注的是对分类语义相关的图像区域（图6）。

![图7：左：ViT-L/32的RGB值初始线性嵌入的滤波器。中：ViT-L/32位置嵌入的相似度。色块显示指定行列位置嵌入与所有其他块位置嵌入之间的余弦相似度。右：按头数和网络深度的关注区域大小。每个点显示16个头之一在一层中跨图像的平均注意力距离。详见附录D.7。](...)

### 4.6 自监督

Transformer在NLP任务上展示了令人印象深刻的性能。然而，它们的成功不仅源于出色的可扩展性，还源于大规模自监督预训练（Devlin et al., 2019; Radford et al., 2018）。我们还对掩码块预测用于自监督进行了初步探索，模仿BERT中使用的掩码语言建模任务。通过自监督预训练，我们较小的ViT-B/16模型在ImageNet上达到79.9%的准确率，相比从头训练有2%的显著提升，但仍落后监督预训练4%。附录B.1.2包含更多细节。我们将对比预训练的探索（Chen et al., 2020b; He et al., 2020; Bachman et al., 2019; Hénaff et al., 2020）留待未来工作。

## 5 结论

我们探索了Transformer在图像识别中的直接应用。与之前在计算机视觉中使用自注意力的工作不同，除了初始的块提取步骤外，我们没有向架构中引入图像特定的归纳偏置。相反，我们将图像解释为一系列块，并使用NLP中使用的标准Transformer编码器处理它。这种简单但可扩展的策略在与大型数据集上的预训练结合时效果出奇地好。因此，Vision Transformer在众多图像分类数据集上匹配或超越了当前最优，同时预训练成本相对较低。

虽然这些初步结果令人鼓舞，但仍有许多挑战。一是将ViT应用于其他计算机视觉任务，如检测和分割。我们的结果与Carion等人（2020）的结果共同表明了这一方法的前景。另一个挑战是继续探索自监督预训练方法。我们的初步实验显示自监督预训练带来了改进，但自监督与大规模监督预训练之间仍有很大差距。最后，进一步扩展ViT可能会带来性能提升。

## 致谢

这项工作在柏林、苏黎世和阿姆斯特丹完成。我们感谢Google的许多同事的帮助，特别感谢Andreas Steiner在基础设施和代码开源发布方面的关键帮助；感谢Joan Puigcerver和Maxim Neumann在大型训练基础设施方面的帮助；感谢Dmitry Lepikhin、Aravindh Mahendran、Daniel Keysers、Mario Lučić、Noam Shazeer、Ashish Vaswani和Colin Raffel的有益讨论。

## 参考文献

[1] Samira Abnar and Willem Zuidema. Quantifying attention flow in transformers. In *ACL*, 2020.

[2] Philip Bachman, R Devon Hjelm, and William Buchwalter. Learning representations by maximizing mutual information across views. In *NeurIPS*, 2019.

[3] Alexei Baevski and Michael Auli. Adaptive input representations for neural language modeling. In *ICLR*, 2019.

[4] I. Bello, B. Zoph, Q. Le, A. Vaswani, and J. Shlens. Attention augmented convolutional networks. In *ICCV*, 2019.

[5] Lucas Beyer, Olivier J. Hénaff, Alexander Kolesnikov, Xiaohua Zhai, and Aäron van den Oord. Are we done with imagenet? *arXiv*, 2020.

[6] Tom B Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. *arXiv*, 2020.

[7] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-to-end object detection with transformers. In *ECCV*, 2020.

[8] Mark Chen, Alec Radford, Rewon Child, Jeff Wu, and Heewoo Jun. Generative pretraining from pixels. In *ICML*, 2020a.

[9] Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey E. Hinton. A simple framework for contrastive learning of visual representations. In *ICML*, 2020b.

[10] Yen-Chun Chen, Linjie Li, Licheng Yu, Ahmed El Kholy, Faisal Ahmed, Zhe Gan, Yu Cheng, and Jingjing Liu. UNITER: UNiversal Image-TExt Representation Learning. In *ECCV*, 2020c.

[11] Rewon Child, Scott Gray, Alec Radford, and Ilya Sutskever. Generating long sequences with sparse transformers. *arXiv*, 2019.

[12] Jean-Baptiste Cordonnier, Andreas Loukas, and Martin Jaggi. On the relationship between self-attention and convolutional layers. In *ICLR*, 2020.

[13] J. Deng, W. Dong, R. Socher, L. Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In *CVPR*, 2009.

[14] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. In *NAACL*, 2019.

[15] Josip Djolonga, Jessica Yung, Michael Tschannen, Rob Romijnders, Lucas Beyer, Alexander Kolesnikov, Joan Puigcerver, Matthias Minderer, Alexander D'Amour, Dan Moldovan, Sylvan Gelly, Neil Houlsby, Xiaohua Zhai, and Mario Lucic. On robustness and transferability of convolutional neural networks. *arXiv*, 2020.

[16] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In *CVPR*, 2016.

[17] Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised visual representation learning. In *CVPR*, 2020.

[18] Jonathan Ho, Nal Kalchbrenner, Dirk Weissenborn, and Tim Salimans. Axial attention in multidimensional transformers. *arXiv*, 2019.

[19] Han Hu, Jiayuan Gu, Zheng Zhang, Jifeng Dai, and Yichen Wei. Relation networks for object detection. In *CVPR*, 2018.

[20] Han Hu, Zheng Zhang, Zhenda Xie, and Stephen Lin. Local relation networks for image recognition. In *ICCV*, 2019.

[21] Zilong Huang, Xinggang Wang, Yunchao Wei, Lichao Huang, Humphrey Shi, Wenyu Liu, and Thomas S. Huang. Ccnet: Criss-cross attention for semantic segmentation. In *ICCV*, 2020.

[22] Olivier J. Hénaff, Aravind Srinivas, Jeffrey De Fauw, Ali Razavi, Carl Doersch, S. M. Ali Eslami, and Aaron van den Oord. Data-efficient image recognition with contrastive predictive coding. In *ICML*, 2020.

[23] Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. 2015.

[24] Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In *ICLR*, 2015.

[25] Alexander Kolesnikov, Lucas Beyer, Xiaohua Zhai, Joan Puigcerver, Jessica Yung, Sylvain Gelly, and Neil Houlsby. Big transfer (BiT): General visual representation learning. In *ECCV*, 2020.

[26] Alex Krizhevsky. Learning multiple layers of features from tiny images. Technical report, 2009.

[27] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton. Imagenet classification with deep convolutional neural networks. In *NIPS*, 2012.

[28] Y. LeCun, B. Boser, J. Denker, D. Henderson, R. Howard, W. Hubbard, and L. Jackel. Backpropagation applied to handwritten zip code recognition. *Neural Computation*, 1:541–551, 1989.

[29] Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang, Maxim Krikun, Noam Shazeer, and Zhifeng Chen. Gshard: Scaling giant models with conditional computation and automatic sharding. *arXiv*, 2020.

[30] Liunian Harold Li, Mark Yatskar, Da Yin, Cho-Jui Hsieh, and Kai-Wei Chang. VisualBERT: A Simple and Performant Baseline for Vision and Language. In *Arxiv*, 2019.

[31] Francesco Locatello, Dirk Weissenborn, Thomas Unterthiner, Aravindh Mahendran, Georg Heigold, Jakob Uszkoreit, Alexey Dosovitskiy, and Thomas Kipf. Object-centric learning with slot attention. *arXiv*, 2020.

[32] Jiasen Lu, Dhruv Batra, Devi Parikh, and Stefan Lee. ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations for Vision-and-Language Tasks. In *NeurIPS*, 2019.

[33] Dhruv Mahajan, Ross Girshick, Vignesh Ramanathan, Kaiming He, Manohar Paluri, Yixuan Li, Ashwin Bharambe, and Laurens van der Maaten. Exploring the limits of weakly supervised pretraining. In *ECCV*, 2018.

[34] M. Nilsback and A. Zisserman. Automated flower classification over a large number of classes. In *ICVGIP*, 2008.

[35] Omkar M. Parkhi, Andrea Vedaldi, Andrew Zisserman, and C. V. Jawahar. Cats and dogs. In *CVPR*, 2012.

[36] Niki Parmar, Ashish Vaswani, Jakob Uszkoreit, Lukasz Kaiser, Noam Shazeer, Alexander Ku, and Dustin Tran. Image transformer. In *ICML*, 2018.

[37] B. T. Polyak and A. B. Juditsky. Acceleration of stochastic approximation by averaging. *SIAM Journal on Control and Optimization*, 30(4):838–855, 1992.

[38] Siyuan Qiao, Huiyu Wang, Chenxi Liu, Wei Shen, and Alan Yuille. Weight standardization. *arXiv preprint arXiv:1903.10520*, 2019.

[39] Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever. Improving language understanding with unsupervised learning. Technical Report, 2018.

[40] Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners. Technical Report, 2019.

[41] Prajit Ramachandran, Niki Parmar, Ashish Vaswani, Irwan Bello, Anselm Levskaya, and Jon Shlens. Stand-alone self-attention in vision models. In *NeurIPS*, 2019.

[42] Chen Sun, Abhinav Shrivastava, Saurabh Singh, and Abhinav Gupta. Revisiting unreasonable effectiveness of data in deep learning era. In *ICCV*, 2017.

[43] Chen Sun, Austin Myers, Carl Vondrick, Kevin Murphy, and Cordelia Schmid. Videobert: A joint model for video and language representation learning. In *ICCV*, 2019.

[44] Hugo Touvron, Andrea Vedaldi, Matthijs Douze, and Herve Jegou. Fixing the train-test resolution discrepancy. In *NeurIPS*, 2019.

[45] Hugo Touvron, Andrea Vedaldi, Matthijs Douze, and Herve Jegou. Fixing the train-test resolution discrepancy: Fixefficientnet. *arXiv preprint arXiv:2003.08237*, 2020.

[46] Michael Tschannen, Josip Djolonga, Marvin Ritter, Aravindh Mahendran, Neil Houlsby, Sylvain Gelly, and Mario Lucic. Self-supervised learning of video-induced visual invariances. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, June 2020.

[47] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In *NIPS*, 2017.

[48] Huiyu Wang, Yukun Zhu, Bradley Green, Hartwig Adam, Alan Yuille, and Liang-Chieh Chen. Axial-deeplab: Stand-alone axial-attention for panoptic segmentation. In *ECCV*, 2020a.

[49] Huiyu Wang, Yukun Zhu, Bradley Green, Hartwig Adam, Alan Yuille, and Liang-Chieh Chen. Axial-deeplab: Stand-alone axial-attention for panoptic segmentation. *arXiv preprint arXiv:2003.07853*, 2020b.

[50] Qiang Wang, Bei Li, Tong Xiao, Jingbo Zhu, Changliang Li, Derek F. Wong, and Lidia S. Chao. Learning deep transformer models for machine translation. In *ACL*, 2019.

[51] Xiaolong Wang, Ross Girshick, Abhinav Gupta, and Kaiming He. Non-local neural networks. In *CVPR*, 2018.

[52] Dirk Weissenborn, Oscar Täckström, and Jakob Uszkoreit. Scaling autoregressive video models. In *ICLR*, 2019.

[53] Bichen Wu, Chenfeng Xu, Xiaoliang Dai, Alvin Wan, Peizhao Zhang, Masayoshi Tomizuka, Kurt Keutzer, and Peter Vajda. Visual transformers: Token-based image representation and processing for computer vision. *arXiv*, 2020.

[54] Yuxin Wu and Kaiming He. Group normalization. In *ECCV*, 2018.

[55] Qizhe Xie, Minh-Thang Luong, Eduard Hovy, and Quoc V. Le. Self-training with noisy student improves imagenet classification. In *CVPR*, 2020.

[56] Xiaohua Zhai, Avital Oliver, Alexander Kolesnikov, and Lucas Beyer. S4L: Self-Supervised Semi-Supervised Learning. In *ICCV*, 2019a.

[57] Xiaohua Zhai, Joan Puigcerver, Alexander Kolesnikov, Pierre Ruyssen, Carlos Riquelme, Mario Lucic, Josip Djolonga, Andre Susano Pinto, Maxim Neumann, Alexey Dosovitskiy, et al. A large-scale study of representation learning with the visual task adaptation benchmark. *arXiv preprint arXiv:1910.04867*, 2019b.

[58] Hengshuang Zhao, Jiaya Jia, and Vladlen Koltun. Exploring self-attention for image recognition. In *CVPR*, 2020.

---

## 附录

### A 多头自注意力

标准qkv自注意力（SA，Vaswani et al., 2017）是神经架构中流行的构建块。对于输入序列$z \in \mathbb{R}^{N \times D}$中的每个元素，我们计算序列中所有值$v$的加权和。注意力权重$A_{ij}$基于序列中两个元素之间的成对相似性及其各自的query $q_i$和key $k_j$表示。

$$
\begin{aligned}
[q, k, v] &= z U_{\text{qkv}}, \quad U_{\text{qkv}} \in \mathbb{R}^{D \times 3 D_h}, \qquad (5) \\
A &= \text{softmax}\left( q k^\top / \sqrt{D_h} \right), \quad A \in \mathbb{R}^{N \times N}, \qquad (6) \\
\text{SA}(z) &= A v. \qquad (7)
\end{aligned}
$$

多头自注意力（MSA）是SA的扩展，在其中我们并行运行$k$个自注意力操作（称为"头"），并将其拼接输出投影。为在改变$k$时保持计算量和参数数量不变，$D_h$（公式5）通常设置为$D/k$。

$$
\text{MSA}(z) = [\text{SA}_1(z); \text{SA}_2(z); \cdots ; \text{SA}_k(z)] U_{\text{msa}}, \quad U_{\text{msa}} \in \mathbb{R}^{k \cdot D_h \times D} \qquad (8)
$$

---

### B 实验细节

#### B.1 训练

表3总结了不同模型的训练设置。我们发现强正则化是在ImageNet上从头训练模型的关键。Dropout（使用时）在每个密集层之后应用，但qkv投影和位置嵌入添加到块嵌入之后除外。混合模型使用与其ViT对应模型完全相同的设置进行训练。最后，所有训练均在224分辨率下进行。

表3：训练超参数。所有模型使用batch size 4096和10k步学习率预热进行训练。对于ImageNet，我们发现额外应用全局范数为1的梯度裁剪有益。训练分辨率为224。

| 模型 | 数据集 | Epochs | 基础LR | LR衰减 | 权重衰减 | Dropout |
|---|---|---|---|---|---|---|
| ViT-B/{16,32} | JFT-300M | 7 | $8 \cdot 10^{-4}$ | linear | 0.1 | 0.0 |
| ViT-L/32 | JFT-300M | 7 | $6 \cdot 10^{-4}$ | linear | 0.1 | 0.0 |
| ViT-L/16 | JFT-300M | 7/14 | $4 \cdot 10^{-4}$ | linear | 0.1 | 0.0 |
| ViT-H/14 | JFT-300M | 14 | $3 \cdot 10^{-4}$ | linear | 0.1 | 0.0 |
| R50x{1,2} | JFT-300M | 7 | $10^{-3}$ | linear | 0.1 | 0.0 |
| R101x1 | JFT-300M | 7 | $8 \cdot 10^{-4}$ | linear | 0.1 | 0.0 |
| R152x{1,2} | JFT-300M | 7 | $6 \cdot 10^{-4}$ | linear | 0.1 | 0.0 |
| R50+ViT-B/{16,32} | JFT-300M | 7 | $8 \cdot 10^{-4}$ | linear | 0.1 | 0.0 |
| R50+ViT-L/32 | JFT-300M | 7 | $2 \cdot 10^{-4}$ | linear | 0.1 | 0.0 |
| R50+ViT-L/16 | JFT-300M | 7/14 | $4 \cdot 10^{-4}$ | linear | 0.1 | 0.0 |
| ViT-B/{16,32} | ImageNet-21k | 90 | $10^{-3}$ | linear | 0.03 | 0.1 |
| ViT-L/{16,32} | ImageNet-21k | 30/90 | $10^{-3}$ | linear | 0.03 | 0.1 |
| ViT-* | ImageNet | 300 | $3 \cdot 10^{-3}$ | cosine | 0.3 | 0.1 |

#### B.1.1 微调

我们使用动量为0.9的SGD微调所有ViT模型。我们对学习率进行小网格搜索，学习率范围见表4。为此，我们使用来自训练集的小子集（Pets和Flowers为10%，CIFAR为2%，ImageNet为1%）作为开发集，并在剩余数据上训练。对于最终结果，我们在整个训练集上训练并在相应测试数据上评估。对于微调ResNet和混合模型，我们使用完全相同的设置，唯一的例外是ImageNet，我们在学习率扫描中添加了另一个值0.06。此外，对于ResNet，我们还运行Kolesnikov等人（2020）的设置，并选择该运行和扫描中的最佳结果。最后，如未另外说明，所有微调实验在384分辨率下运行（在与训练不同分辨率下进行微调是常见做法，Kolesnikov et al., 2020）。

当将ViT模型迁移到另一个数据集时，我们移除整个头部（两个线性层）并用单个零初始化的线性层替换，输出目标数据集所需的类别数。我们发现这比仅仅重新初始化最后一层要稍微稳健一些。

对于VTAB，我们遵循Kolesnikov等人（2020）的协议，对所有任务使用相同的超参数设置。我们使用0.01的学习率，训练2500步（表4）。我们通过对两个学习率和两个调度进行小扫描，选择在200样本验证集上VTAB分数最高的设置。我们遵循Kolesnikov等人（2020）中使用的预处理，但未使用任务特定的输入分辨率。相反，我们发现Vision Transformer在所有任务上最受益于高分辨率（$384 \times 384$）。

表4：微调超参数。所有模型使用余弦学习率衰减、batch size 512、无权重衰减和全局范数为1的梯度裁剪进行微调。如未另外说明，微调分辨率为384。

| 数据集 | 步数 | 基础LR |
|---|---|---|
| ImageNet | 20,000 | {0.003, 0.01, 0.03, 0.06} |
| CIFAR100 | 10,000 | {0.001, 0.003, 0.01, 0.03} |
| CIFAR10 | 10,000 | {0.001, 0.003, 0.01, 0.03} |
| Oxford-IIIT Pets | 500 | {0.001, 0.003, 0.01, 0.03} |
| Oxford Flowers-102 | 500 | {0.001, 0.003, 0.01, 0.03} |
| VTAB (19 tasks) | 2,500 | 0.01 |

#### B.1.2 自监督

我们采用掩码块预测目标进行初步的自监督实验。为此，我们破坏50%的块嵌入，方式为用可学习的[mask]嵌入替换（80%）、随机其他块嵌入（10%）或保持原样（10%）。该设置与Devlin等人（2019）用于语言的设置非常相似。最后，我们使用每个被破坏块的块表示预测其3位平均颜色（即总共512种颜色）。

我们在JFT上训练了我们的自监督模型1M步（约14个epochs），batch size为4096。我们使用Adam，基础学习率为$2 \cdot 10^{-4}$，预热10k步以及余弦学习率衰减。作为预训练的预测目标，我们尝试了以下设置：1）仅预测平均3位颜色（即1个512种颜色的预测），2）并行预测$16 \times 16$块的$4 \times 4$缩小版，使用3位颜色（即16个512种颜色的预测），3）使用L2对完整块进行回归（即3个RGB通道上的256个回归）。令人惊讶的是，我们发现所有方法都工作得相当好，尽管L2略差。我们仅报告选项1的最终结果，因为它显示了最佳的few-shot性能。我们还尝试了Devlin等人（2019）使用的15%破坏率，但在我们的few-shot指标上结果也略差。

最后，我们想说明的是，我们的掩码块预测实例不需要如此大量的预训练也不需要像JFT这样的大型数据集就能在ImageNet分类上获得类似的性能提升。也就是说，我们观察到在100k预训练步后下游性能的递减收益，并且在ImageNet上预训练时也能看到类似的收益。

---

### C 补充结果

我们报告了与论文中图示相对应的详细结果。表5对应论文中的图3，展示了不同ViT模型在递增数据集（ImageNet、ImageNet-21k和JFT-300M）上预训练的迁移性能。表6对应论文中的图5，展示了ViT、ResNet和混合模型不同大小的迁移性能以及预训练的估计计算成本。

表5：Vision Transformer在不同数据集上预训练后在多个数据集上的Top1准确率（%）。这些值对应正文中的图3。模型在384分辨率下微调。注意ImageNet结果的计算未使用表2中达到结果所使用的附加技术（Polyak平均和512分辨率图像）。

| 预训练数据集 | 模型 | CIFAR-10 | CIFAR-100 | ImageNet | ImageNet ReaL | Oxford Flowers-102 | Oxford-IIIT-Pets |
|---|---|---|---|---|---|---|---|
| ImageNet | ViT-B/16 | 98.13 | 87.13 | 77.91 | 83.57 | 89.49 | 93.81 |
| | ViT-B/32 | 97.77 | 86.31 | 73.38 | 79.56 | 85.43 | 92.04 |
| | ViT-L/16 | 97.86 | 86.35 | 76.53 | 82.19 | 89.66 | 93.64 |
| | ViT-L/32 | 97.94 | 87.07 | 71.16 | 77.83 | 86.36 | 91.35 |
| ImageNet-21k | ViT-B/16 | 98.95 | 91.67 | 83.97 | 88.35 | 99.38 | 94.43 |
| | ViT-B/32 | 98.79 | 91.97 | 81.28 | 86.63 | 99.11 | 93.02 |
| | ViT-L/16 | 99.16 | 93.44 | 85.15 | 88.40 | 99.61 | 94.73 |
| | ViT-L/32 | 99.13 | 93.04 | 80.99 | 85.65 | 99.19 | 93.09 |
| | ViT-H/14 | 99.27 | 93.82 | 85.13 | 88.70 | 99.51 | 94.82 |
| JFT-300M | ViT-B/16 | 99.00 | 91.87 | 84.15 | 88.85 | 99.56 | 95.80 |
| | ViT-B/32 | 98.61 | 90.49 | 80.73 | 86.27 | 99.27 | 93.40 |
| | ViT-L/16 | 99.38 | 94.04 | 87.12 | 89.99 | 99.56 | 97.11 |
| | ViT-L/32 | 99.19 | 92.52 | 84.37 | 88.28 | 99.45 | 95.83 |
| | ViT-H/14 | 99.50 | 94.55 | 88.04 | 90.33 | 99.68 | 97.56 |

表6：模型扩展实验的详细结果。这些对应论文中的图5。我们展示了多个数据集上的迁移准确率以及预训练计算量（exaFLOPs）。

| 名称 | Epochs | ImageNet | ImageNet ReaL | CIFAR-10 | CIFAR-100 | Pets | Flowers | exaFLOPs |
|---|---|---|---|---|---|---|---|---|
| ViT-B/32 | 7 | 80.73 | 86.27 | 98.61 | 90.49 | 93.40 | 99.27 | 55 |
| ViT-B/16 | 7 | 84.15 | 88.85 | 99.00 | 91.87 | 95.80 | 99.56 | 224 |
| ViT-L/32 | 7 | 84.37 | 88.28 | 99.19 | 92.52 | 95.83 | 99.45 | 196 |
| ViT-L/16 | 7 | 86.30 | 89.43 | 99.38 | 93.46 | 96.81 | 99.66 | 783 |
| ViT-L/16 | 14 | 87.12 | 89.99 | 99.38 | 94.04 | 97.11 | 99.56 | 1567 |
| ViT-H/14 | 14 | 88.08 | 90.36 | 99.50 | 94.71 | 97.11 | 99.71 | 4262 |
| ResNet50x1 | 7 | 77.54 | 84.56 | 97.67 | 86.07 | 91.11 | 94.26 | 50 |
| ResNet50x2 | 7 | 82.12 | 87.94 | 98.29 | 89.20 | 93.43 | 97.02 | 199 |
| ResNet101x1 | 7 | 80.67 | 87.07 | 98.48 | 89.17 | 94.08 | 95.95 | 96 |
| ResNet152x1 | 7 | 81.88 | 87.96 | 98.82 | 90.22 | 94.17 | 96.94 | 141 |
| ResNet152x2 | 7 | 84.97 | 89.69 | 99.06 | 92.05 | 95.37 | 98.62 | 563 |
| ResNet152x2 | 14 | 85.56 | 89.89 | 99.24 | 91.92 | 95.75 | 98.75 | 1126 |
| ResNet200x3 | 14 | 87.22 | 90.15 | 99.34 | 93.53 | 96.32 | 99.04 | 3306 |
| R50x1+ViT-B/32 | 7 | 84.90 | 89.15 | 99.01 | 92.24 | 95.75 | 99.46 | 106 |
| R50x1+ViT-B/16 | 7 | 85.58 | 89.65 | 99.14 | 92.63 | 96.65 | 99.40 | 274 |
| R50x1+ViT-L/32 | 7 | 85.68 | 89.04 | 99.24 | 92.93 | 96.97 | 99.43 | 246 |
| R50x1+ViT-L/16 | 7 | 86.60 | 89.72 | 99.18 | 93.64 | 97.03 | 99.40 | 859 |
| R50x1+ViT-L/16 | 14 | 87.12 | 89.76 | 99.31 | 93.89 | 97.36 | 99.11 | 1668 |

---

### D 补充分析

#### D.1 ResNet的SGD vs. Adam

ResNet通常使用SGD训练，我们使用Adam作为优化器相当非常规。这里我们展示激发这一选择的实验。具体来说，我们比较用SGD和Adam在JFT上预训练的两个ResNet（50x1和152x2）的微调性能。对于SGD，我们使用Kolesnikov等人（2020）推荐的超参数。结果见表7。Adam预训练在大多数数据集上和平均性能上优于SGD预训练。这证明了在JFT上使用Adam作为预训练ResNet的优化器的选择是合理的。注意，绝对数字低于Kolesnikov等人（2020）报告的结果，因为我们仅预训练了7个epochs，而非30个。

表7：使用Adam和SGD预训练的ResNet模型的微调性能。

| 数据集 | ResNet50 (Adam) | ResNet50 (SGD) | ResNet152x2 (Adam) | ResNet152x2 (SGD) |
|---|---|---|---|---|
| ImageNet | 77.54 | 78.24 | 84.97 | 84.37 |
| CIFAR10 | 97.67 | 97.46 | 99.06 | 99.07 |
| CIFAR100 | 86.07 | 85.17 | 92.05 | 91.06 |
| Oxford-IIIT Pets | 91.11 | 91.00 | 95.37 | 94.79 |
| Oxford Flowers-102 | 94.26 | 92.06 | 98.62 | 99.32 |
| Average | 89.33 | 88.79 | 94.01 | 93.72 |

#### D.2 Transformer形状

我们对缩放Transformer架构的不同维度进行了消融实验，以找出哪些最适合扩展到非常大的模型。图8显示了不同配置在ImageNet上的5-shot性能。所有配置基于一个具有8层、$D=1024$、$D_{\text{MLP}}=2048$和patch size为32的ViT模型（所有线的交点）。我们可以看到，缩放深度带来了最大的改进，直到64层都明显可见。然而，在16层之后已经可以看到递减收益。有趣的是，缩放网络宽度似乎产生的变化最小。减小patch size从而增加有效序列长度在未引入参数的情况下显示出令人惊讶的稳健改进。这些发现表明，计算量可能比参数量更能预测性能，并且如果有所侧重，扩展应强调深度而非宽度。总体而言，我们发现按比例缩放所有维度能带来稳健的改进。

![图8：Vision Transformer不同模型维度的缩放。](...)

#### D.3 头部类型与类别token

为尽可能保持与原始Transformer模型接近，我们使用了一个额外的[class] token，作为图像表示。该token的输出随后通过一个小型多层感知器（MLP）转换为类别预测，MLP的单个隐藏层使用tanh非线性激活。

此设计继承自文本的Transformer模型，我们在整篇论文中一直使用。最初尝试仅使用图像块嵌入，进行全局平均池化（GAP），然后接线性分类器（就像ResNet的最终特征图）效果非常差。然而，我们发现这既不是由于额外的token，也不是由于GAP操作。相反，性能差异完全可以通过需要不同的学习率来解释，见图9。

![图9：Class-token与全局平均池化分类器的比较。两者工作得同样好，但需要不同的学习率。](...)

#### D.4 位置嵌入

我们对使用位置嵌入编码空间信息的不同方式进行了消融实验。我们尝试了以下情况：

- **不提供位置信息**：将输入视为块的无序集合。
- **一维位置嵌入**：将输入视为按光栅顺序排列的块序列（本文所有其他实验的默认设置）。
- **二维位置嵌入**：将输入视为二维网格中的块。在这种情况下，学习两组嵌入，每组对应一个轴——X嵌入和Y嵌入，每个大小为$D/2$。然后，根据输入中块的位置坐标，我们拼接X和Y嵌入以获得该块的最终位置嵌入。
- **相对位置嵌入**：考虑块之间的相对距离而非绝对位置来编码空间信息。为此，我们使用一维相对注意力，在其中定义所有可能块对的相对距离。因此，对于每个给定对（一个作为query，另一个作为注意力机制中的key/value），我们有一个偏移量$p_q - p_k$，每个偏移量关联一个嵌入。然后，我们运行额外的注意力，使用原始query（query的内容），但使用相对位置嵌入作为key。然后，我们使用相对注意力的logits作为偏置项，在应用softmax之前将其加到主注意力（基于内容的注意力）的logits上。

除了编码空间信息的不同方式外，我们还尝试了在模型中融入这些信息的不同方式。对于一维和二维位置嵌入，我们尝试了三种不同情况：（1）在模型的主干之后将位置嵌入添加到输入，然后再将输入送入Transformer编码器（本文所有其他实验中的默认设置）；（2）在每一层开始时学习并将位置嵌入添加到输入；（3）在每一层开始时向输入添加学习到的位置嵌入（层间共享）。

表8总结了在ViT-B/16模型上进行的此项消融研究的结果。可以看到，虽然没有位置嵌入的模型与有位置嵌入的模型之间存在较大差距，但不同编码位置信息的方式之间几乎没有差异。我们推测，由于我们的Transformer编码器在块级别的输入（而非像素级别）上运行，编码空间信息方式的差异不那么重要。更精确地说，在块级别的输入中，空间维度远小于原始像素级别的输入，例如$14 \times 14$对比$224 \times 224$，在该分辨率下学习表示空间关系对这些不同的位置编码策略来说同样容易。即便如此，网络学习到的位置嵌入相似性的具体模式取决于训练超参数（图10）。

表8：使用ViT-B/16模型在ImageNet 5-shot线性上对位置嵌入进行消融研究的结果。

| Pos. Emb. | Default/Stem | Every Layer | Every Layer-Shared |
|---|---|---|---|
| No Pos. Emb. | 0.61382 | N/A | N/A |
| 1-D Pos. Emb. | 0.64206 | 0.63964 | 0.64292 |
| 2-D Pos. Emb. | 0.64001 | 0.64046 | 0.64022 |
| Rel. Pos. Emb. | 0.64032 | N/A | N/A |

![图10：不同超参数训练模型的位置嵌入。](...)

#### D.5 实证计算成本

我们同样关注各种架构在硬件上的实际速度，由于通道宽度和缓存大小等细节，理论FLOPs并不总能很好预测实际速度。为此，我们在TPUv3加速器上对主要模型的推理速度进行了计时；推理和反向传播速度之间的差异是一个与模型无关的常数因子。

图12（左）展示了一个核心在各种输入大小下每秒能处理多少张图像。每个点指的是在广泛batch size范围内测量的峰值性能。可以看出，ViT随图像大小的理论双二次缩放仅在最大模型和最大分辨率下勉强开始出现。

另一个感兴趣的指标是每个模型能容纳到单个核心上的最大batch size，越大对扩展到大数据集越有利。图12（右）展示了相同模型集的该指标。这表明大型ViT模型在内存效率方面相比ResNet模型具有明显优势。

![图12：左：各种架构在不同输入大小下的实际时钟计时。ViT模型具有与类似ResNet相当的速度。右：各种架构在不同输入大小下适配到设备上的最大每核心batch size。ViT模型在内存效率方面明显更优。](...)

#### D.6 轴向注意力

轴向注意力（Huang et al., 2020; Ho et al., 2019）是一种简单而有效的技术，用于在组织为多维Tensor的大型输入上运行自注意力。轴向注意力的一般思想是执行多个注意力操作，每个沿输入Tensor的单个轴进行，而不是对输入的展平版本应用一维注意力。在轴向注意力中，每个注意力沿特定轴混合信息，同时保持其他轴上的信息独立。沿着这条线，Wang等人（2020b）提出了AxialResNet模型，其中ResNet50中所有kernel size为$3 \times 3$的卷积都被替换为轴向自注意力，即行和列注意力，并通过相对位置编码增强。我们将AxialResNet实现为基线模型[^3]。

此外，我们修改了ViT以处理二维形状的输入（而非一维的块序列），并融入了轴向Transformer块，其中不是由自注意力加MLP组成，而是由行自注意力加MLP后跟列自注意力加MLP组成。

图13展示了Axial ResNet、Axial-ViT-B/32和Axial-ViT-B/16在JFT数据集上预训练后，在ImageNet 5-shot线性上的性能与预训练计算量的关系，分别以FLOPs数量和推理时间（每秒样本数）表示。可以看出，Axial-ViT-B/32和Axial-ViT-B/16在性能上都优于其对应的ViT-B模型，但代价是更多的计算。这是因为在Axial-ViT模型中，每个具有全局自注意力的Transformer块被两个轴向Transformer块替换（一个行自注意力加一个列自注意力），尽管在轴向情况下自注意力操作的序列长度较小，但每个Axial-ViT块中多了一个MLP。对于AxialResNet，虽然在准确率/计算权衡方面看起来合理（图13，左），但朴素实现在TPU上极慢（图13，右）。

[^3]: 我们的实现基于https://github.com/csrhddlam/axial-deeplab 的开源PyTorch实现。在我们的实验中，我们复现了Wang等人（2020b）报告的准确率分数，但我们的实现与开源实现类似，在TPU上非常慢。因此，我们无法将其用于广泛的大规模实验。这可能需要通过精心优化的实现来实现。

![图13：基于轴向注意力的模型的性能（ImageNet 5-shot线性top-1准确率）与其速度（FLOPs数量（左）和推理时间（右））的关系。](...)

#### D.7 注意力距离

为理解ViT如何使用自注意力整合整张图像的信息，我们分析了不同层注意力权重的平均跨度距离（图11）。这个"注意力距离"类似于CNN中的感受野大小。较低层中不同头部的平均注意力距离变化很大，有些头部关注到图像的大部分区域，而其他头部则关注查询位置附近的小区域。随着深度增加，所有头部的注意力距离都增大。在网络的后面一半，大多数头部广泛关注各token。

![图11：按头部和网络深度的关注区域大小。通过平均query像素与所有其他像素之间的距离（以注意力权重加权）计算128张示例图像的注意力距离。每个点显示16个头部之一在一层中跨图像的平均注意力距离。图像宽度为224像素。](...)

#### D.8 注意力图

为计算从输出token到输入空间的注意力图（图6和图14），我们使用Attention Rollout（Abnar & Zuidema, 2020）。简言之，我们平均所有头部的ViT-L/16注意力权重，然后递归乘以所有层的权重矩阵。这解释了注意力在所有层间跨token的混合。

![图14：与图6类似的更多注意力图示例（随机选择）。](...)

#### D.9 ObjectNet结果

我们还按照Kolesnikov等人（2020）的评估设置，在ObjectNet基准上评估了我们的旗舰模型ViT-H/14，得到top-5准确率82.1%和top-1准确率61.7%。

#### D.10 VTAB详细分解

表9展示了在每个VTAB-1k任务上获得的分数。

表9：VTAB-1k任务上的性能分解。

| 模型 | Caltech101 | CIFAR-100 | DTD | Flowers102 | Pets | Sun397 | SVHN | Camelyon | EuroSAT | Resisc45 | Retinopathy | Clevr-Count | Clevr-Dist | DMLab | dSpr-Loc | dSpr-Ori | KITTI-Dist | sNORB-Azim | sNORB-Elev | Mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ViT-H/14 (JFT) | 95.3 | 85.5 | 75.2 | 99.7 | 97.2 | 65.0 | 88.9 | 83.3 | 96.7 | 91.4 | 76.6 | 91.7 | 63.8 | 53.1 | 79.4 | 63.3 | 84.5 | 33.2 | 51.2 | 77.6 |
| ViT-L/16 (JFT) | 95.4 | 81.9 | 74.3 | 99.7 | 96.7 | 63.5 | 87.4 | 83.6 | 96.5 | 89.7 | 77.1 | 86.4 | 63.1 | 49.7 | 74.5 | 60.5 | 82.2 | 36.2 | 51.1 | 76.3 |
| ViT-L/16 (I21k) | 90.8 | 84.1 | 74.1 | 99.3 | 92.7 | 61.0 | 80.9 | 82.5 | 95.6 | 85.2 | 75.3 | 70.3 | 56.1 | 41.9 | 74.7 | 64.9 | 79.9 | 30.5 | 41.7 | 72.7 |
