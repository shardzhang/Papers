# SimCLR：视觉表征对比学习的简单框架

> Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton | Google Research, Brain Team
>
> Proceedings of the 37th International Conference on Machine Learning (ICML), Vienna, Austria, PMLR 119, 2020

本文介绍了 SimCLR：一个用于视觉表征对比学习的简单框架。核心内容：

- 简化近期提出的对比自监督学习算法，无需专门架构或记忆库
- 系统研究框架的主要组成部分，包括数据增强、非线性投影头、损失函数与批次大小
- 结合各发现，在 ImageNet 自监督和半监督学习上大幅超越以往方法

关键发现：

- 数据增强的复合对定义有效的对比预测任务至关重要，对比学习受益于比监督学习更强的数据增强
- 在表征与对比损失之间引入可学习的非线性变换显著提高表征质量
- 线性分类器达到 76.5% top-1 准确率（相对提升 7%），仅用 1% 标签微调即达 85.8% top-5

---

## 摘要

本文提出了 SimCLR：一个用于视觉表征对比学习的简单框架。我们简化了近期提出的对比自监督学习算法，无需专门的架构或记忆库。为了理解对比预测任务如何能够学习到有用的表征，我们系统地研究了框架中的主要组成部分。我们表明：(1) 数据增强的复合在定义有效的预测任务中起着关键作用；(2) 在表征和对比损失之间引入可学习的非线性变换显著提高了学习到的表征的质量；(3) 与监督学习相比，对比学习受益于更大的批次大小和更多的训练步数。通过结合这些发现，我们能够在 ImageNet 上大幅超越以往的自监督和半监督学习方法。在 SimCLR 学习到的自监督表征上训练的线性分类器达到了 76.5% 的 top-1 准确率，相比之前的最先进技术提升了 7%，与有监督的 ResNet-50 性能相当。当仅使用 1% 的标签进行微调时，我们达到了 85.8% 的 top-5 准确率，以少 100 倍的数据超越了 AlexNet。^1

^1 代码可在 https://github.com/google-research/simclr 获取。

## 1. 引言

在没有人类监督的情况下学习有效的视觉表征是一个长期存在的问题。大多数主流方法分为两类：生成式或判别式。生成式方法学习生成或以其他方式建模输入空间中的像素（Hinton et al., 2006; Kingma & Welling, 2013; Goodfellow et al., 2014）。然而，像素级别的生成在计算上非常昂贵，并且可能不是表征学习所必需的。判别式方法使用与监督学习类似的 objective 函数来学习表征，但训练网络执行预文本任务，其中输入和标签都来自无标注数据集。许多此类方法依赖启发式方法来设计预文本任务（Doersch et al., 2015; Zhang et al., 2016; Noroozi & Favaro, 2016; Gidaris et al., 2018），这可能限制所学表征的通用性。基于潜在空间中对比学习的判别式方法最近显示出巨大的前景，取得了最先进的结果（Hadsell et al., 2006; Dosovitskiy et al., 2014; Oord et al., 2018; Bachman et al., 2019）。

在这项工作中，我们提出了一个用于视觉表征对比学习的简单框架，我们称之为 SimCLR。SimCLR 不仅性能优于之前的工作（图 1），而且更简单，既不需要专门架构（Bachman et al., 2019; Hénaff et al., 2019），也不需要记忆库（Wu et al., 2018; Tian et al., 2019; He et al., 2019; Misra & van der Maaten, 2019）。

为了理解什么促成了良好的对比表征学习，我们系统地研究了框架中的主要组成部分，并表明：

- 多个数据增强操作的复合对于定义能够产生有效表征的对比预测任务至关重要。此外，无监督对比学习受益于比监督学习更强的数据增强。
- 在表征和对比损失之间引入可学习的非线性变换显著提高了学习到的表征的质量。
- 使用对比交叉熵损失的表征学习受益于归一化嵌入和适当调整的温度参数。
- 与监督学习相比，对比学习受益于更大的批次大小和更长的训练。与监督学习一样，对比学习受益于更深更宽的网络。

我们结合这些发现，在 ImageNet ILSVRC-2012（Russakovsky et al., 2015）上实现了自监督和半监督学习的最新水平。在线性评估协议下，SimCLR 达到了 76.5% 的 top-1 准确率，相比之前的最先进技术（Hénaff et al., 2019）提升了 7%。当仅使用 1% 的 ImageNet 标签进行微调时，SimCLR 达到了 85.8% 的 top-5 准确率，相对提升了 10%（Hénaff et al., 2019）。当在其他自然图像分类数据集上微调时，SimCLR 在 12 个数据集中的 10 个上表现与强监督基线相当或更好（Kornblith et al., 2019）。

## 2. 方法

### 2.1. 对比学习框架

受近期对比学习算法的启发（参见第 7 节概述），SimCLR 通过在潜在空间中使用对比损失最大化同一数据示例的不同增强视图之间的一致性来学习表征。如图 2 所示，该框架包含以下四个主要组成部分。

- **随机数据增强模块**，随机变换任何给定的数据示例，产生同一示例的两个相关视图，记为 $\tilde{x}_i$ 和 $\tilde{x}_j$，我们将其视为正样本对。在这项工作中，我们顺序应用三种简单的增强：随机裁剪后调整回原始大小、随机颜色失真和随机高斯模糊。如第 3 节所示，随机裁剪和颜色失真的组合对于实现良好性能至关重要。
- **神经网络基编码器** $f(\cdot)$，从增强后的数据示例中提取表征向量。我们的框架允许选择各种网络架构，没有任何约束。我们选择简单性，采用常用的 ResNet（He et al., 2016）来获得 $h_i = f(\tilde{x}_i) = \text{ResNet}(\tilde{x}_i)$，其中 $h_i \in \mathbb{R}^d$ 是平均池化层后的输出。
- **小型神经网络投影头** $g(\cdot)$，将表征映射到应用对比损失的空间。我们使用带有一个隐藏层的 MLP（Multilayer Perceptron，多层感知机）来获得 $z_i = g(h_i) = W^{(2)}\sigma(W^{(1)}h_i)$，其中 $\sigma$ 是 ReLU（Rectified Linear Unit，线性整流单元）非线性。如第 4 节所示，我们发现在 $z_i$ 上而非 $h_i$ 上定义对比损失是有益的。
- **为对比预测任务定义的对比损失函数**。给定一个包含正样本对 $\tilde{x}_i$ 和 $\tilde{x}_j$ 的集合 $\{\tilde{x}_k\}$，对比预测任务旨在对于给定的 $\tilde{x}_i$ 从 $\{\tilde{x}_k\}_{k\neq i}$ 中识别出 $\tilde{x}_j$。

我们随机采样一个包含 $N$ 个示例的小批次，并在从该小批次派生的增强示例对上定义对比预测任务，从而得到 $2N$ 个数据点。我们不显式采样负样本。相反，对于给定的正样本对，类似于（Chen et al., 2017），我们将小批次中的其他 $2(N-1)$ 个增强示例视为负样本。令 $\text{sim}(u, v) = u^T v / (\|u\| \|v\|)$ 表示 $\ell_2$ 归一化的 $u$ 和 $v$ 之间的点积（即余弦相似度）。那么正样本对 $(i, j)$ 的损失函数定义为：

$$
\ell_{i,j} = -\log \frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_{k=1}^{2N} \mathbb{1}_{[k\neq i]} \exp(\text{sim}(z_i, z_k)/\tau)} \qquad (1)
$$

其中 $\mathbb{1}_{[k\neq i]} \in \{0,1\}$ 是一个指示函数，当 $k \neq i$ 时为 1，$\tau$ 表示温度参数。最终损失是在小批次中的所有正样本对 $(i, j)$ 和 $(j, i)$ 上计算的。该损失已在之前的工作中使用（Sohn, 2016; Wu et al., 2018; Oord et al., 2018）；为方便起见，我们称之为 NT-Xent（Normalized Temperature-scaled Cross Entropy，归一化温度标度交叉熵损失）。

**算法 1** SimCLR 的主要学习算法

$$
\begin{aligned}
& \text{输入：批次大小 } N, \text{常数 } \tau, f, g, T \text{ 的结构} \\
& \textbf{for} \text{ 采样的每个小批次 } \{x_k\}_{k=1}^N \textbf{ do} \\
& \qquad \textbf{for} \text{ 所有 } k \in \{1, ..., N\} \textbf{ do} \\
& \qquad \qquad \text{采样两个增强函数 } t \sim T, t' \sim T \\
& \qquad \qquad \tilde{x}_{2k-1} = t(x_k) \qquad \qquad \triangleright \text{ 第一次增强} \\
& \qquad \qquad h_{2k-1} = f(\tilde{x}_{2k-1}) \quad \triangleright \text{ 表征} \\
& \qquad \qquad z_{2k-1} = g(h_{2k-1}) \quad \triangleright \text{ 投影} \\
& \qquad \qquad \tilde{x}_{2k} = t'(x_k) \qquad \qquad \triangleright \text{ 第二次增强} \\
& \qquad \qquad h_{2k} = f(\tilde{x}_{2k}) \quad \triangleright \text{ 表征} \\
& \qquad \qquad z_{2k} = g(h_{2k}) \quad \triangleright \text{ 投影} \\
& \qquad \textbf{end for} \\
& \qquad \textbf{for} \text{ 所有 } i \in \{1, ..., 2N\} \text{ 和 } j \in \{1, ..., 2N\} \textbf{ do} \\
& \qquad \qquad s_{i,j} = z_i^T z_j / (\|z_i\| \|z_j\|) \quad \triangleright \text{ 成对相似度} \\
& \qquad \textbf{end for} \\
& \qquad \text{定义 } \ell(i,j) = -\log\left( \frac{\exp(s_{i,j}/\tau)}{\sum_{k=1}^{2N} \mathbb{1}_{[k\neq i]} \exp(s_{i,k}/\tau)} \right) \\
& \qquad L = \frac{1}{2N} \sum_{k=1}^N \left[ \ell(2k-1, 2k) + \ell(2k, 2k-1) \right] \\
& \qquad \text{更新网络 } f \text{ 和 } g \text{ 以最小化 } L \\
& \textbf{end for} \\
& \text{返回编码器网络 } f(\cdot)，\text{丢弃 } g(\cdot)
\end{aligned}
$$

算法 1 总结了所提出的方法。

### 2.2. 使用大批次训练

为了保持简单，我们不使用记忆库训练模型（Wu et al., 2018; He et al., 2019）。相反，我们将训练批次大小 $N$ 从 256 变化到 8192。批次大小为 8192 时，每个正样本对可以从两个增强视图中获得 16382 个负样本。当使用标准 SGD（Stochastic Gradient Descent，随机梯度下降）/动量优化器配合线性学习率缩放时，大批次训练可能不稳定（Goyal et al., 2017）。为了稳定训练，我们对所有批次大小使用 LARS（Layer-wise Adaptive Rate Scaling，逐层自适应学习率缩放）优化器（You et al., 2017）。我们使用 Cloud TPU（Tensor Processing Unit，张量处理单元）训练模型，根据批次大小使用 32 到 128 个核心。^2

^2 使用 128 个 TPU v3 核心，以批次大小 4096 训练我们的 ResNet-50 100 个 epochs 大约需要 1.5 小时。

**全局 BN（Batch Normalization，批归一化）。** 标准 ResNet 使用批归一化（Ioffe & Szegedy, 2015）。在数据并行的分布式训练中，BN 均值和方差通常在每个设备上本地聚合。在我们的对比学习中，由于正样本对在同一设备上计算，模型可以利用局部信息泄露来提高预测精度而不改进表征。我们通过在训练期间聚合所有设备上的 BN 均值和方差来解决这个问题。其他方法包括跨设备打乱数据示例（He et al., 2019），或用层归一化替换 BN（Hénaff et al., 2019）。

### 2.3. 评估协议

这里我们列出实证研究的协议，旨在理解框架中不同的设计选择。

**数据集和指标。** 我们关于无监督预训练（学习无标签的编码器网络 f）的大部分研究使用 ImageNet ILSVRC-2012 数据集（Russakovsky et al., 2015）。一些在 CIFAR-10（Krizhevsky & Hinton, 2009）上的额外预训练实验可以在附录 B.9 中找到。我们还在各种数据集上测试预训练结果的迁移学习。为了评估学习到的表征，我们遵循广泛使用的线性评估协议（Zhang et al., 2016; Oord et al., 2018; Bachman et al., 2019; Kolesnikov et al., 2019），其中在冻结的基网络之上训练线性分类器，并使用测试准确率作为表征质量的代理。除了线性评估，我们还在半监督和迁移学习上与最先进技术进行比较。

**默认设置。** 除非另有说明，对于数据增强，我们使用随机裁剪和调整大小（带随机翻转）、颜色失真和高斯模糊（详情见附录 A）。我们使用 ResNet-50 作为基编码器网络，并使用 2 层 MLP 投影头将表征投影到 128 维潜在空间。损失方面，我们使用 NT-Xent，使用 LARS 优化，学习率为 $4.8\ (= 0.3 \times \text{BatchSize}/256)$，权重衰减为 $10^{-6}$。我们使用批次大小 4096 训练 100 个 epochs。^3 此外，我们在前 10 个 epochs 使用线性热身，并使用无重启的余弦衰减调度来衰减学习率（Loshchilov & Hutter, 2016）。

^3 虽然在 100 个 epochs 内未达到最大性能，但已获得合理结果，足以进行公平且高效的消融实验。

## 3. 对比表征学习的数据增强

数据增强定义了预测任务。虽然数据增强已广泛用于监督和无监督表征学习（Krizhevsky et al., 2012; Hénaff et al., 2019; Bachman et al., 2019），但它尚未被系统地视为定义对比预测任务的方法。许多现有方法通过改变架构来定义对比预测任务。例如，Hjelm et al. (2018); Bachman et al. (2019) 通过约束网络架构中的感受野实现全局到局部视图预测，而 Oord et al. (2018); Hénaff et al. (2019) 通过固定的图像拆分程序和上下文聚合网络实现相邻视图预测。我们表明，通过对目标图像执行简单的随机裁剪（带调整大小）可以避免这种复杂性，这创建了一系列预测任务，涵盖了上述两种，如图 3 所示。这种简单的设计选择方便地将预测任务与其他组件（如神经网络架构）解耦。通过扩展增强族并以随机方式组合它们，可以定义更广泛的对比预测任务。

### 3.1. 数据增强操作的复合对于学习良好表征至关重要

为了系统地研究数据增强的影响，我们在此考虑几种常见的增强。一种类型的增强涉及数据的空间/几何变换，如裁剪和调整大小（带水平翻转）、旋转（Gidaris et al., 2018）和 cutout（DeVries & Taylor, 2017）。另一种类型的增强涉及外观变换，如颜色失真（包括颜色丢弃、亮度、对比度、饱和度、色调）（Howard, 2013; Szegedy et al., 2015）、高斯模糊和 Sobel 滤波。图 4 可视化了我们在本工作中研究的增强。

为了理解单个数据增强的效果以及增强复合的重要性，我们研究了在单独或成对应用增强时框架的性能。由于 ImageNet 图像具有不同大小，我们总是应用裁剪和调整图像大小（Krizhevsky et al., 2012; Szegedy et al., 2015），这使得在没有裁剪的情况下研究其他增强变得困难。为了消除这个混杂因素，我们在此消融研究中考虑非对称数据变换设置。具体来说，我们总是先随机裁剪图像并调整到相同分辨率，然后仅对图 2 框架的一个分支应用目标变换，而保持另一个分支为恒等映射（即 t(x_i) = x_i）。注意，这种非对称数据增强会损害性能。尽管如此，这种设置不应实质性改变单个数据增强或其复合的影响。

图 5 展示了单个和复合变换下的线性评估结果。我们观察到，没有单一的变换足以学习良好的表征，即使模型可以在对比任务中几乎完美地识别正样本对。当复合增强时，对比预测任务变得更难，但表征质量显著提高。附录 B.2 提供了关于更广泛增强组合的进一步研究。

有一种增强组合特别突出：随机裁剪和随机颜色失真。我们推测，仅使用随机裁剪作为数据增强时的一个严重问题是，图像中的大多数 patches 共享相似的颜色分布。图 6 显示，仅颜色直方图就足以区分图像。神经网络可能利用这种捷径来解决预测任务。因此，为了学习可泛化的特征，将裁剪与颜色失真相结合至关重要。

### 3.2. 对比学习需要比监督学习更强的数据增强

为了进一步证明颜色增强的重要性，我们调整颜色增强的强度，如表 1 所示。更强的颜色增强显著提高了学习到的无监督模型的线性评估结果。在这种背景下，AutoAugment（Cubuk et al., 2019）——一种使用监督学习发现的复杂增强策略——并不比简单的裁剪 +（更强）颜色失真更好。当使用相同增强集训练监督模型时，我们观察到更强的颜色增强不会改善甚至损害它们的性能。因此，我们的实验表明，无监督对比学习受益于比监督学习更强的（颜色）数据增强。尽管先前工作已报告数据增强对自监督学习有用（Doersch et al., 2015; Bachman et al., 2019; Hénaff et al., 2019; Asano et al., 2019），但我们表明，不能为监督学习带来精度提升的数据增强仍然可以显著帮助对比学习。

## 4. 编码器和头的架构

### 4.1. 无监督对比学习（更）受益于更大的模型

图 7 显示，也许不出所料，增加深度和宽度都能提升性能。虽然类似发现在监督学习中也成立（He et al., 2016），但我们发现监督模型与在无监督模型上训练的线性分类器之间的差距随着模型规模的增大而缩小，这表明无监督学习相比其监督对应方法更受益于更大的模型。

### 4.2. 非线性投影头提高了其之前层的表征质量

然后我们研究包含投影头 g(h) 的重要性。图 8 展示了使用三种不同头架构的线性评估结果：(1) 恒等映射；(2) 线性投影，如几种先前方法所用（Wu et al., 2018）；(3) 默认的带一个额外隐藏层（和 ReLU 激活）的非线性投影，类似于 Bachman et al. (2019)。我们观察到非线性投影优于线性投影（+3%），且远优于无投影（>10%）。当使用投影头时，无论输出维度如何，都观察到类似的结果。此外，即使使用非线性投影，投影头之前的层 h 仍然远优于（>10%）之后的层 z = g(h)，这表明投影头之前的隐藏层是比之后的层更好的表征。

我们推测，使用非线性投影之前的表征的重要性在于对比损失导致的信息损失。特别地，z = g(h) 被训练为对数据变换保持不变。因此，g 可能移除对下游任务有用的信息，如对象的颜色或方向。通过利用非线性变换 g(·)，更多信息可以在 h 中形成和维持。为了验证这一假设，我们进行实验，使用 $h$ 或 $g(h)$ 来学习预测预训练期间应用的变换。这里我们设置 $g(h) = W^{(2)}\sigma(W^{(1)}h)$，具有相同的输入和输出维度（即 2048）。表 3 显示 $h$ 包含更多关于所应用变换的信息，而 $g(h)$ 丢失了信息。进一步的分析可以在附录 B.4 中找到。

## 5. 损失函数和批次大小

### 5.1. 带可调温度的归一化交叉熵损失优于其他选择

我们将 NT-Xent 损失与其他常用的对比损失函数进行比较，如逻辑损失（Mikolov et al., 2013）和间隔损失（Schroff et al., 2015）。表 2 显示了目标函数以及损失函数输入的梯度。观察梯度，我们注意到 (1) $\ell_2$ 归一化（即余弦相似度）配合温度有效地加权不同样本，适当的温度可以帮助模型从困难负样本中学习；(2) 与交叉熵不同，其他目标函数不按其相对难度对负样本进行加权。因此，必须对这些损失函数应用半困难负样本挖掘（Schroff et al., 2015）：不是计算所有损失项的梯度，而是使用半困难负样本项（即那些在损失间隔内且距离最近，但比正样本更远的项）计算梯度。

为了使比较公平，我们对所有损失函数使用相同的 $\ell_2$ 归一化，调整超参数，并报告它们的最佳结果。^8 表 4 显示，虽然（半困难）负样本挖掘有所帮助，但最佳结果仍然远差于我们的默认 NT-Xent 损失。

^8 详情见附录 B.10。为简单起见，我们仅考虑来自一个增强视图的负样本。

我们接下来测试 $\ell_2$ 归一化（即余弦相似度与点积）和温度 $\tau$ 在我们的默认 NT-Xent 损失中的重要性。表 5 显示，没有归一化和适当的温度缩放，性能显著下降。没有 $\ell_2$ 归一化时，对比任务准确率更高，但所得表征在线性评估下更差。

### 5.2. 对比学习（更）受益于更大的批次大小和更长的训练

图 9 显示了当模型以不同数量的 epochs 训练时，批次大小的影响。我们发现，当训练 epoch 数较少时（例如 100 epochs），大批次相比小批次具有显著优势。随着更多训练步数/epochs，不同批次大小之间的差距减小或消失，前提是批次被随机重采样。与监督学习（Goyal et al., 2017）相反，在对比学习中，更大的批次大小提供更多的负样本，有助于收敛（即在给定准确率下需要更少的 epochs 和步数）。更长的训练也提供更多的负样本，改善结果。在附录 B.1 中，提供了更长训练步数的结果。

## 6. 与最先进技术的比较

在此小节中，类似于 Kolesnikov et al. (2019); He et al. (2019)，我们使用 3 种不同隐藏层宽度（宽度倍数为 1 $\times$ 、2 $\times$ 和 4 $\times$ ）的 ResNet-50。为了更好的收敛，我们这里的模型训练了 1000 epochs。

**线性评估。** 表 6 将我们的结果与先前方法在线性评估设置下进行了比较（Zhuang et al., 2019; He et al., 2019; Misra & van der Maaten, 2019; Hénaff et al., 2019; Kolesnikov et al., 2019; Donahue & Simonyan, 2019; Bachman et al., 2019; Tian et al., 2019）（见附录 B.6）。表 1 显示了不同方法之间更多的数值比较。我们能够使用标准网络获得比需要专门设计架构的先前方法更好的结果。我们的 ResNet-50 (4 $\times$ ) 获得的最佳结果可以匹配有监督预训练的 ResNet-50。

**半监督学习。** 我们遵循 Zhai et al. (2019) 的做法，以类别平衡的方式采样 1% 或 10% 的有标签 ILSVRC-12 训练数据集（分别约每类 12.8 和约每类 128 张图像）。^11 我们简单地在有标签数据上微调整个基网络，无需正则化（见附录 B.5）。^11

^11 采样细节和确切子集可在 https://www.tensorflow.org/datasets/catalog/imagenet2012_subset 中找到。表 7 展示了我们的结果与近期方法的比较（Zhai et al., 2019; Xie et al., 2019; Sohn et al., 2020; Wu et al., 2018; Donahue & Simonyan, 2019; Misra & van der Maaten, 2019; Hénaff et al., 2019）。来自（Zhai et al., 2019）的监督基线由于对超参数（包括增强）进行了密集搜索而表现强劲。再次，我们的方法在使用 1% 和 10% 标签的情况下都显著改善了最先进技术。有趣的是，在完整 ImageNet 上微调我们预训练的 ResNet-50 (2 $\times$ , 4 $\times$ ) 也显著优于从头训练（高达 2%，见附录 B.2）。

**迁移学习。** 我们在线性评估（固定特征提取器）和微调设置下评估了跨 12 个自然图像数据集的迁移学习性能。遵循 Kornblith et al. (2019)，我们为每个模型-数据集组合执行超参数调优，并在验证集上选择最佳超参数。表 8 显示了使用 ResNet-50 (4 $\times$ ) 模型的结果。当微调时，我们的自监督模型在 5 个数据集上显著优于监督基线，而监督基线仅在 2 个数据集上更优（即 Pets 和 Flowers）。在其余 5 个数据集上，模型在统计上无显著差异。完整的实验细节以及标准 ResNet-50 架构的结果见附录 B.8。

## 7. 相关工作

使图像的表示在小变换下相互一致的思想可以追溯到 Becker & Hinton (1992)。我们通过利用数据增强、网络架构和对比损失方面的最新进展来扩展它。类似的用于类别标签预测的一致性思想已在其他上下文中探索过，如半监督学习（Xie et al., 2019; Berthelot et al., 2019）。

**手工设计的预文本任务。** 自监督学习的近期复兴始于人工设计的预文本任务，如相对补丁预测（Doersch et al., 2015）、解决拼图（Noroozi & Favaro, 2016）、着色（Zhang et al., 2016）和旋转预测（Gidaris et al., 2018; Chen et al., 2019）。虽然使用更大的网络和更长的训练可以获得好的结果（Kolesnikov et al., 2019），但这些预文本任务依赖于某种程度的特设启发式方法，限制了所学表征的通用性。

**对比视觉表征学习。** 可以追溯到 Hadsell et al. (2006)，这些方法通过对比正样本对与负样本对来学习表征。沿着这些方向，Dosovitskiy et al. (2014) 提出将每个实例视为由特征向量（参数形式）表示的类。Wu et al. (2018) 提出使用记忆库存储实例类表征向量，该方法在几篇近期论文中被采用和扩展（Zhuang et al., 2019; Tian et al., 2019; He et al., 2019; Misra & van der Maaten, 2019）。其他工作探索使用批次内样本进行负采样，而不是记忆库（Doersch & Zisserman, 2017; Ye et al., 2019; Ji et al., 2019）。

近期文献试图将其方法的成功与潜在表征之间的互信息最大化联系起来（Oord et al., 2018; Hénaff et al., 2019; Hjelm et al., 2018; Bachman et al., 2019）。然而，尚不清楚对比方法的成功是由互信息决定的，还是由对比损失的特定形式决定的（Tschannen et al., 2019）。

我们注意到，我们框架的几乎所有单个组成部分都曾在先前工作中出现过，尽管具体的实例化可能有所不同。我们的框架相对于先前工作的优越性不是由任何单一设计选择解释的，而是由它们的组合解释的。我们在附录 C 中提供了我们的设计选择与先前工作的综合比较。

## 8. 结论

在这项工作中，我们提出了一个用于对比视觉表征学习的简单框架及其实例化。我们仔细研究了它的组成部分，并展示了不同设计选择的效果。通过结合我们的发现，我们在自监督、半监督和迁移学习方法上有了显著改进。

我们的方法与 ImageNet 上的标准监督学习仅在数据增强的选择、网络末端非线性头的使用以及损失函数上有所不同。这个简单框架的力量表明，尽管近期兴趣激增，自监督学习仍然被低估了。

## 致谢

我们要感谢 Xiaohua Zhai、Rafael Müller 和 Yani Ioannou 对草稿的反馈。我们也感谢 Google Research 多伦多及其他地方团队的一般性支持。

## 参考文献

Asano, Y. M., Rupprecht, C., and Vedaldi, A. A critical analysis of self-supervision, or what we can learn from a single image. arXiv preprint arXiv:1904.13132, 2019.

Chen, T., Sun, Y., Shi, Y., and Hong, L. On sampling strategies for neural network-based collaborative filtering. In Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pp. 767–776, 2017.

Chen, T., Zhai, X., Ritter, M., Lucic, M., and Houlsby, N. Self-supervised gans via auxiliary rotation loss. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 12154–12163, 2019.

Cimpoi, M., Maji, S., Kokkinos, I., Mohamed, S., and Vedaldi, A. Describing textures in the wild. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 3606–3613. IEEE, 2014.

Cubuk, E. D., Zoph, B., Mane, D., Vasudevan, V., and Le, Q. V. Autoaugment: Learning augmentation strategies from data. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 113–123, 2019.

DeVries, T. and Taylor, G. W. Improved regularization of convolutional neural networks with cutout. arXiv preprint arXiv:1708.04552, 2017.

Doersch, C. and Zisserman, A. Multi-task self-supervised visual learning. In Proceedings of the IEEE International Conference on Computer Vision, pp. 2051–2060, 2017.

Doersch, C., Gupta, A., and Efros, A. A. Unsupervised visual representation learning by context prediction. In Proceedings of the IEEE International Conference on Computer Vision, pp. 1422–1430, 2015.

Donahue, J. and Simonyan, K. Large scale adversarial representation learning. In Advances in Neural Information Processing Systems, pp. 10541–10551, 2019.

Donahue, J., Jia, Y., Vinyals, O., Hoffman, J., Zhang, N., Tzeng, E., and Darrell, T. Decaf: A deep convolutional activation feature for generic visual recognition. In International Conference on Machine Learning, pp. 647–655, 2014.

Bachman, P., Hjelm, R. D., and Buchwalter, W. Learning representations by maximizing mutual information across views. In Advances in Neural Information Processing Systems, pp. 15509–15519, 2019.

Dosovitskiy, A., Springenberg, J. T., Riedmiller, M., and Brox, T. Discriminative unsupervised feature learning with convolutional neural networks. In Advances in neural information processing systems, pp. 766–774, 2014.

Becker, S. and Hinton, G. E. Self-organizing neural network that discovers surfaces in random-dot stereograms. Nature, 355(6356):161–163, 1992.

Berg, T., Liu, J., Lee, S. W., Alexander, M. L., Jacobs, D. W., and Belhumeur, P. N. Birdsnap: Large-scale fine-grained visual categorization of birds. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 2019–2026. IEEE, 2014.

Berthelot, D., Carlini, N., Goodfellow, I., Papernot, N., Oliver, A., and Raffel, C. A. Mixmatch: A holistic approach to semi-supervised learning. In Advances in Neural Information Processing Systems, pp. 5050–5060, 2019.

Bossard, L., Guillaumin, M., and Van Gool, L. Food-101–mining discriminative components with random forests. In European conference on computer vision, pp. 446–461. Springer, 2014.

Everingham, M., Van Gool, L., Williams, C. K., Winn, J., and Zisserman, A. The pascal visual object classes (voc) challenge. International Journal of Computer Vision, 88(2):303–338, 2010.

Fei-Fei, L., Fergus, R., and Perona, P. Learning generative visual models from few training examples: An incremental bayesian approach tested on 101 object categories. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshop on Generative-Model Based Vision, 2004.

Gidaris, S., Singh, P., and Komodakis, N. Unsupervised representation learning by predicting image rotations. arXiv preprint arXiv:1803.07728, 2018.

Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., and Bengio, Y. Generative adversarial nets. In Advances in neural information processing systems, pp. 2672–2680, 2014.

Goyal, P., Dollár, P., Girshick, R., Noordhuis, P., Wesolowski, L., Kyrola, A., Tulloch, A., Jia, Y., and He, K. Accurate, large minibatch sgd: Training imagenet in 1 hour. arXiv preprint arXiv:1706.02677, 2017.

Hadsell, R., Chopra, S., and LeCun, Y. Dimensionality reduction by learning an invariant mapping. In 2006 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR'06), volume 2, pp. 1735–1742. IEEE, 2006.

He, K., Zhang, X., Ren, S., and Sun, J. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770–778, 2016.

He, K., Fan, H., Wu, Y., Xie, S., and Girshick, R. Momentum contrast for unsupervised visual representation learning. arXiv preprint arXiv:1911.05722, 2019.

Hénaff, O. J., Razavi, A., Doersch, C., Eslami, S., and Oord, A. v. d. Data-efficient image recognition with contrastive predictive coding. arXiv preprint arXiv:1905.09272, 2019.

Hinton, G. E., Osindero, S., and Teh, Y.-W. A fast learning algorithm for deep belief nets. Neural computation, 18(7):1527–1554, 2006.

Hjelm, R. D., Fedorov, A., Lavoie-Marchildon, S., Grewal, K., Bachman, P., Trischler, A., and Bengio, Y. Learning deep representations by mutual information estimation and maximization. arXiv preprint arXiv:1808.06670, 2018.

Howard, A. G. Some improvements on deep convolutional neural network based image classification. arXiv preprint arXiv:1312.5402, 2013.

Ioffe, S. and Szegedy, C. Batch normalization: Accelerating deep network training by reducing internal covariate shift. arXiv preprint arXiv:1502.03167, 2015.

Ji, X., Henriques, J. F., and Vedaldi, A. Invariant information clustering for unsupervised image classification and segmentation. In Proceedings of the IEEE International Conference on Computer Vision, pp. 9865–9874, 2019.

Kingma, D. P. and Welling, M. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013.

Kolesnikov, A., Zhai, X., and Beyer, L. Revisiting self-supervised visual representation learning. In Proceedings of the IEEE conference on Computer Vision and Pattern Recognition, pp. 1920–1929, 2019.

Kornblith, S., Shlens, J., and Le, Q. V. Do better ImageNet models transfer better? In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 2661–2671, 2019.

Krause, J., Deng, J., Stark, M., and Fei-Fei, L. Collecting a large-scale dataset of fine-grained cars. In Second Workshop on Fine-Grained Visual Categorization, 2013.

Krizhevsky, A. and Hinton, G. Learning multiple layers of features from tiny images. Technical report, University of Toronto, 2009.

Krizhevsky, A., Sutskever, I., and Hinton, G. E. Imagenet classification with deep convolutional neural networks. In Advances in neural information processing systems, pp. 1097–1105, 2012.

Loshchilov, I. and Hutter, F. Sgdr: Stochastic gradient descent with warm restarts. arXiv preprint arXiv:1608.03983, 2016.

Maaten, L. v. d. and Hinton, G. Visualizing data using t-sne. Journal of machine learning research, 9(Nov):2579–2605, 2008.

Maji, S., Kannala, J., Rahtu, E., Blaschko, M., and Vedaldi, A. Fine-grained visual classification of aircraft. Technical report, 2013.

Mikolov, T., Chen, K., Corrado, G., and Dean, J. Efficient estimation of word representations in vector space. arXiv preprint arXiv:1301.3781, 2013.

Misra, I. and van der Maaten, L. Self-supervised learning of pretext-invariant representations. arXiv preprint arXiv:1912.01991, 2019.

Nilsback, M.-E. and Zisserman, A. Automated flower classification over a large number of classes. In Computer Vision, Graphics & Image Processing, 2008. ICVGIP'08. Sixth Indian Conference on, pp. 722–729. IEEE, 2008.

Noroozi, M. and Favaro, P. Unsupervised learning of visual representations by solving jigsaw puzzles. In European Conference on Computer Vision, pp. 69–84. Springer, 2016.

Oord, A. v. d., Li, Y., and Vinyals, O. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748, 2018.

Parkhi, O. M., Vedaldi, A., Zisserman, A., and Jawahar, C. Cats and dogs. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 3498–3505. IEEE, 2012.

Russakovsky, O., Deng, J., Su, H., Krause, J., Satheesh, S., Ma, S., Huang, Z., Karpathy, A., Khosla, A., Bernstein, M., et al. Imagenet large scale visual recognition challenge. International journal of computer vision, 115(3):211–252, 2015.

Schroff, F., Kalenichenko, D., and Philbin, J. Facenet: A unified embedding for face recognition and clustering. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 815–823, 2015.

Simonyan, K. and Zisserman, A. Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556, 2014.

Sohn, K. Improved deep metric learning with multi-class n-pair loss objective. In Advances in neural information processing systems, pp. 1857–1865, 2016.

Sohn, K., Berthelot, D., Li, C.-L., Zhang, Z., Carlini, N., Cubuk, E. D., Kurakin, A., Zhang, H., and Raffel, C. Fixmatch: Simplifying semi-supervised learning with consistency and confidence. arXiv preprint arXiv:2001.07685, 2020.

Szegedy, C., Liu, W., Jia, Y., Sermanet, P., Reed, S., Anguelov, D., Erhan, D., Vanhoucke, V., and Rabinovich, A. Going deeper with convolutions. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 1–9, 2015.

Tian, Y., Krishnan, D., and Isola, P. Contrastive multiview coding. arXiv preprint arXiv:1906.05849, 2019.

Tschannen, M., Djolonga, J., Rubenstein, P. K., Gelly, S., and Lucic, M. On mutual information maximization for representation learning. arXiv preprint arXiv:1907.13625, 2019.

Wu, Z., Xiong, Y., Yu, S. X., and Lin, D. Unsupervised feature learning via non-parametric instance discrimination. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 3733–3742, 2018.

Xiao, J., Hays, J., Ehinger, K. A., Oliva, A., and Torralba, A. Sun database: Large-scale scene recognition from abbey to zoo. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 3485–3492. IEEE, 2010.

Xie, Q., Dai, Z., Hovy, E., Luong, M.-T., and Le, Q. V. Unsupervised data augmentation. arXiv preprint arXiv:1904.12848, 2019.

Ye, M., Zhang, X., Yuen, P. C., and Chang, S.-F. Unsupervised embedding learning via invariant and spreading instance feature. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 6210–6219, 2019.

You, Y., Gitman, I., and Ginsburg, B. Large batch training of convolutional networks. arXiv preprint arXiv:1708.03888, 2017.

Zhai, X., Oliver, A., Kolesnikov, A., and Beyer, L. S4l: Self-supervised semi-supervised learning. In The IEEE International Conference on Computer Vision (ICCV), October 2019.

Zhang, R., Isola, P., and Efros, A. A. Colorful image colorization. In European conference on computer vision, pp. 649–666. Springer, 2016.

Zhuang, C., Zhai, A. L., and Yamins, D. Local aggregation for unsupervised learning of visual embeddings. In Proceedings of the IEEE International Conference on Computer Vision, pp. 6002–6012, 2019.
---

## 附录

## A. 数据增强细节

在我们的默认预训练设置（用于训练我们最佳模型）中，我们利用随机裁剪（带调整和随机翻转）、随机颜色失真和随机高斯模糊作为数据增强。下面提供这三种增强的详细信息。

**随机裁剪并调整到 $224 \times 224$** 我们使用标准的 Inception 风格随机裁剪（Szegedy et al., 2015）。裁剪的大小在原图面积的 0.08 到 1.0 之间均匀随机选择，宽高比在原图宽高比的 3/4 到 4/3 之间随机选择。该裁剪最终被调整到原始大小。这在 TensorFlow 中已实现为 "slim.preprocessing.inception_preprocessing.distorted_bounding_box_crop"，或在 PyTorch 中为 "torchvision.transforms.RandomResizedCrop"。此外，随机裁剪（带调整）之后总是以 50% 的概率进行随机水平/左右翻转。这有帮助但并非必需。从我们的默认增强策略中移除它后，对于训练 100 epochs 的 ResNet-50 模型，top-1 线性评估从 64.5% 下降到 63.4%。

**颜色失真** 颜色失真由颜色抖动和颜色丢弃组成。我们发现更强的颜色抖动通常有帮助，因此我们设置了一个强度参数。

以下是使用 TensorFlow 的颜色失真伪代码：

$$
import tensorflow as tf
def color_distortion(image, s=1.0):
    # image 是一个值范围在 [0, 1] 的张量
    # s 是颜色失真的强度
    def color_jitter(x):
        # 也可以每次应用时打乱以下增强的顺序
        x = tf.image.random_brightness(x, max_delta=0.8*s)
        x = tf.image.random_contrast(x, lower=1-0.8*s, upper=1+0.8*s)
        x = tf.image.random_saturation(x, lower=1-0.8*s, upper=1+0.8*s)
        x = tf.image.random_hue(x, max_delta=0.2*s)
        x = tf.clip_by_value(x, 0, 1)
        return x
    def color_drop(x):
        image = tf.image.rgb_to_grayscale(image)
        image = tf.tile(image, [1, 1, 3])
        return image
    # 以概率 p 随机应用变换
    image = random_apply(color_jitter, image, p=0.8)
    image = random_apply(color_drop, image, p=0.2)
    return image
$$

以下是使用 PyTorch 的颜色失真伪代码^12：

$$
from torchvision import transforms
def get_color_distortion(s=1.0):
    # s 是颜色失真的强度
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([
        rnd_color_jitter,
        rnd_gray])
    return color_distort
$$

^12 我们的代码和结果基于 TensorFlow，这里的 PyTorch 代码仅供参考。

**高斯模糊** 此增强在我们的默认策略中。我们发现它有帮助，因为它将训练 100 epochs 的 ResNet-50 从 63.2% 提高到 64.5%。我们使用高斯核以 50% 的概率对图像进行模糊处理。我们随机采样 $\sigma \in [0.1, 2.0]$，核大小设为图像高度/宽度的 10%。

## B. 额外实验结果

### B.1. 批次大小和训练步数

图 B.1 展示了在不同批次大小和训练 epochs 下线性评估的 top-5 准确率。结论与之前显示的 top-1 准确率非常相似，只是不同批次大小和训练步数之间的差异在这里稍微小一些。

在图 9 和图 B.1 中，我们在使用不同批次大小训练时采用类似于（Goyal et al., 2017）的线性学习率缩放。虽然线性学习率缩放在 SGD/动量优化器中很流行，但我们发现在 LARS 优化器中平方根学习率缩放更可取。使用平方根学习率缩放时，我们有 $\text{LearningRate} = 0.075 \times \sqrt{\text{BatchSize}}$，而不是线性缩放情况下的 $\text{LearningRate} = 0.3 \times \text{BatchSize}/256$，但当批次大小为 4096（我们的默认批次大小）时，两种缩放方法下的学习率相同。比较结果见表 B.1，我们观察到平方根学习率缩放提高了以较小批次大小和较少 epoch 训练的模型的性能。

**表 B.1.** 不同批次大小和训练 epochs 下的线性评估（top-1）。斜线左侧是使用线性 LR 缩放的模型，右侧是使用平方根 LR 缩放的模型。如果结果优于 0.5% 以上则加粗。平方根 LR 缩放在较少 epochs 下的小批次训练中效果更好（使用 LARS 优化器）。

我们还使用更大的批次大小（高达 32K）和更长的训练（高达 3200 epochs），配合平方根学习率缩放。如图 B.2 所示，批次大小为 8192 时性能似乎饱和，而更长的训练仍能显著提高性能。

### B.2. 更广泛的数据增强组合进一步提升性能

当将默认增强策略扩展为包括以下内容时，我们在正文中（表 6 和 7）的最佳结果可以进一步提升：(1) Sobel 滤波，(2) 额外的颜色失真（均衡化、曝光化），以及 (3) 运动模糊。在线性评估协议下，使用更广泛数据增强训练的 ResNet-50 模型（1 $\times$ 、2 $\times$ 、4 $\times$ ）分别达到 70.0 (+0.7)、74.4 (+0.2)、76.8 (+0.3)。

表 B.2 显示了通过微调 SimCLR 模型获得的 ImageNet 准确率（微调过程详见附录 B.5）。有趣的是，当在完整（100%）ImageNet 训练集上微调时，我们的 ResNet (4 $\times$ ) 模型达到了 80.4% top-1 / 95.4% top-5^13，显著优于使用相同增强集（即随机裁剪和水平翻转）从头训练的结果（78.4% top-1 / 94.2% top-5）。对于 ResNet-50 (2 $\times$ )，微调我们预训练的 ResNet-50 (2 $\times$ ) 也优于从头训练（77.8% top-1 / 93.9% top-5）。对于 ResNet-50，微调没有带来改进。

^13 若不使用更广泛的增强进行 SimCLR 预训练，则为 80.1% top-1 / 95.2% top-5。

### B.3. 监督模型更长训练的效果

这里我们进行实验，看看训练步数和更强的数据增强如何影响监督训练。我们在与无监督模型相同的数据增强集（随机裁剪、颜色失真、50% 高斯模糊）下测试 ResNet-50 和 ResNet-50 (4 $\times$ )。图 B.3 显示了 top-1 准确率。我们观察到，在 ImageNet 上更长时间地训练监督模型没有显著益处。更强的数据增强略微提高了 ResNet-50 (4 $\times$ ) 的准确率，但对 ResNet-50 没有帮助。当应用更强的数据增强时，ResNet-50 通常需要更长的训练（例如 500 epochs^14）以获得最优结果，而 ResNet-50 (4 $\times$ ) 则不会从更长的训练中受益。

^14 使用 AutoAugment（Cubuk et al., 2019），最优测试精度可在 900 到 500 个 epochs 之间达到。

### B.4. 理解非线性投影头

图 B.3 显示了用于计算 $z = Wh$ 的线性投影矩阵 $W \in \mathbb{R}^{2048 \times 2048}$ 的特征值分布。该矩阵具有相对较少的大特征值，表明它近似低秩。

图 B.4 显示了我们的最佳 ResNet-50（top-1 线性评估 69.3%）对随机选择的 10 个类别的 $h$ 和 $z = g(h)$ 的 t-SNE（t-Distributed Stochastic Neighbor Embedding，t分布随机邻域嵌入）（Maaten & Hinton, 2008）可视化。与 $z$ 相比，$h$ 表示的类别分离得更好。

### B.5. 通过微调的半监督学习

**微调过程** 我们使用 Nesterov 动量优化器进行微调，批次大小为 4096，动量为 0.9，学习率为 0.8（遵循 $\text{LearningRate} = 0.05 \times \text{BatchSize}/256$），无热身。仅使用随机裁剪（带随机左右翻转并调整大小为 224x224）进行预处理。我们不使用任何正则化（包括权重衰减）。对于 1% 的有标签数据，我们微调 60 epochs；对于 10% 的有标签数据，我们微调 30 epochs。推理时，我们将给定图像调整为 256x256，并取单个 224x224 中心裁剪。

表 B.4 展示了半监督学习不同方法的 top-1 准确率比较。我们的模型显著改善了最先进技术。

### B.6. 线性评估

对于线性评估，我们遵循与微调类似的过程（附录 B.5 中描述），但使用更大的学习率 1.6（遵循 $\text{LearningRate} = 0.1 \times \text{BatchSize}/256$）和更长的训练（90 epochs）。另外，使用预训练超参数的 LARS 优化器也能产生类似的结果。此外，我们发现将线性分类器附加在基编码器之上（对线性分类器的输入使用 stop_gradient 以防止标签信息影响编码器）并在预训练期间同时训练它们可以达到类似的性能。

### B.7. 线性评估与微调之间的相关性

这里我们研究在不同训练步数和网络架构设置下线性评估与微调之间的相关性。

图 B.5 显示了当 ResNet-50（使用批次大小 4096）的训练 epochs 从 50 变化到 3200（如图 B.2）时的线性评估与微调对比。虽然它们几乎呈线性相关，但似乎在少量标签上微调更受益于更长的训练。

图 B.6 显示了不同架构选择下的线性评估与微调对比。

### B.8. 迁移学习

我们在两种设置下评估了自监督表征的迁移学习性能：线性评估（在 ImageNet 上学到的自监督表征基础上训练逻辑回归分类器对新数据集进行分类）和微调（允许所有权重在训练期间变化）。在这两种情况下，我们遵循 Kornblith et al. (2019) 描述的方法，尽管我们的预处理略有不同。

#### B.8.1. 方法

**数据集** 我们研究了在 Food-101 数据集（Bossard et al., 2014）、CIFAR-10 和 CIFAR-100（Krizhevsky & Hinton, 2009）、Birdsnap（Berg et al., 2014）、SUN397 场景数据集（Xiao et al., 2010）、Stanford Cars（Krause et al., 2013）、FGVC Aircraft（Maji et al., 2013）、PASCAL VOC 2007 分类任务（Everingham et al., 2010）、可描述纹理数据集（DTD）（Cimpoi et al., 2014）、Oxford-IIIT Pets（Parkhi et al., 2012）、Caltech-101（Fei-Fei et al., 2004）和 Oxford 102 Flowers（Nilsback & Zisserman, 2008）上的迁移学习性能。我们遵循这些数据集原始论文中的评估协议，即：对 Food-101、CIFAR-10、CIFAR-100、Birdsnap、SUN397、Stanford Cars 和 DTD 报告 top-1 准确率；对 FGVC Aircraft、Oxford-IIIT Pets、Caltech-101 和 Oxford 102 Flowers 报告平均每类准确率；对 PASCAL VOC 2007 报告 Everingham et al. (2010) 定义的 11 点 mAP（mean Average Precision，平均精度均值）指标。对于 DTD 和 SUN397，数据集创建者定义了多个训练/测试划分；我们仅报告第一个划分的结果。Caltech-101 没有定义训练/测试划分，因此我们每类随机选择 30 张图像，并在剩余图像上进行测试，以便与先前工作公平比较（Donahue et al., 2014; Simonyan & Zisserman, 2014）。

我们使用数据集创建者指定的验证集为 FGVC Aircraft、PASCAL VOC 2007、DTD 和 Oxford 102 Flowers 选择超参数。对于其他数据集，在执行超参数调优时，我们从训练集中保留一部分用于验证。在验证集上选择最优超参数后，我们使用选定的参数在所有训练和验证图像上重新训练模型。我们在测试集上报告准确率。

**通过线性分类器的迁移学习** 我们在冻结的预训练网络提取的特征上训练 $\ell_2$ 正则化的多项逻辑回归分类器。我们使用 L-BFGS（Limited-memory BFGS，有限内存 BFGS）优化 softmax 交叉熵目标函数，并且不应用数据增强。作为预处理，所有图像使用双三次重采样调整到短边 224 像素，然后取 $224 \times 224$ 的中心裁剪。我们从 $10^{-6}$ 到 $10^5$ 之间的 45 个对数间隔值中选择 $\ell_2$ 正则化参数。

**通过微调的迁移学习** 我们使用预训练网络的权重作为初始化，对整个网络进行微调。我们训练 20,000 步，批次大小为 256，使用带 Nesterov 动量的 SGD，动量参数为 0.9。我们将批归一化统计的动量参数设为 max(1 - 10/s, 0.9)，其中 s 是每个 epoch 的步数。微调期间的数据增强仅包括带调整和翻转的随机裁剪；与预训练不同，我们不进行颜色增强或模糊处理。测试时，我们将图像调整到短边 256 像素，并取 224 $\times$ 224 的中心裁剪。（进一步优化数据增强可能带来额外的精度提升，特别是在 CIFAR-10 和 CIFAR-100 数据集上。）我们选择学习率和权重衰减，搜索网格为 7 个对数间隔的学习率（0.0001 到 0.1）和 7 个对数间隔的权重衰减值（10^{-6} 到 10^{-3}），以及无权重衰减。我们将这些权重衰减值除以学习率。

**从随机初始化训练** 我们使用与微调相同的过程从随机初始化训练网络，但训练时间更长，并采用修改后的超参数网格。我们从 7 个对数间隔的学习率（0.001 到 1.0）和 8 个对数间隔的权重衰减值（$10^{-5}$ 到 $10^{-1.5}$）的网格中选择超参数。重要的是，我们的随机初始化基线训练了 40,000 步，这足以达到接近最大精度，如 Kornblith et al. (2019) 图 8 所示。

在 Birdsnap 上，各方法之间没有统计学上的显著差异；在 Food-101、Stanford Cars 和 FGVC Aircraft 数据集上，微调相比随机初始化训练仅提供很小的优势。然而，在其余 8 个数据集上，预训练具有明显优势。

**监督基线** 我们与在 ImageNet 上使用标准交叉熵损失训练的架构相同的 ResNet 模型进行比较。这些模型使用与我们自监督模型相同的数据增强（裁剪、强颜色增强和模糊）进行训练，也训练了 1000 epochs。我们发现，虽然更强的数据增强和更长的训练时间在 ImageNet 上无益于准确率，但在迁移数据子集上的线性评估中，这些模型显著优于训练 90 epochs 和普通数据增强的监督基线。监督 ResNet-50 基线在 ImageNet 上达到 76.3% top-1 准确率，而自监督对应模型为 69.3%；而 ResNet-50 (4 $\times$ ) 基线达到 78.3%，自监督模型为 76.5%。

**统计显著性检验** 我们使用置换检验来测试模型之间差异的显著性。给定两个模型的预测，我们通过随机交换每个示例的预测并计算随机化后的准确率差异，从零分布生成 100,000 个样本。然后我们计算零分布中比观察到的预测差异更极端的样本百分比。对于 top-1 准确率，此过程产生与精确 McNemar 检验相同的结果。在零假设下可交换性的假设对平均每类准确率也成立，但在计算平均精度曲线时不成立。因此，我们对 VOC 2007 上的准确率差异（而非 mAP 差异）进行显著性检验。此过程的一个局限性是它不考虑训练模型时的运行间变异性，仅考虑使用有限样本评估图像所产生的变异性。

#### B.8.2. 标准 ResNet 的结果

正文表 8 中显示的 ResNet-50 (4 $\times$ ) 结果没有显示监督或自监督模型的明显优势。然而，使用较窄的 ResNet-50 架构，监督学习相比自监督学习保持明显优势。监督 ResNet-50 模型在线性评估的所有数据集和微调的大多数（12 个中的 10 个）数据集上优于自监督模型。ResNet 模型相比 ResNet (4 $\times$ ) 模型的较弱性能可能与 ImageNet 上监督和自监督模型之间的准确率差距有关。自监督 ResNet 获得 69.3% top-1 准确率，绝对值比监督模型差 6.8%，而自监督 ResNet (4 $\times$ ) 模型获得 76.5%，仅比监督模型差 1.8%。

### B.9. CIFAR-10

虽然我们主要使用 ImageNet 作为预训练无监督模型的主要数据集，但我们的方法也适用于其他数据集。我们通过在 CIFAR-10 上进行测试来证明这一点，如下所述。

**设置** 由于我们的目标不是优化 CIFAR-10 的性能，而是进一步确认我们在 ImageNet 上的观察结果，因此我们对 CIFAR-10 实验使用相同的架构（ResNet-50）。由于 CIFAR-10 图像远小于 ImageNet 图像，我们将第一个步长为 2 的 $7 \times 7$ 卷积替换为步长为 1 的 $3 \times 3$ 卷积，并移除了第一个最大池化操作。对于数据增强，我们使用与 ImageNet 相同的 Inception 裁剪（翻转并调整到 $32 \times 32$）^15，以及颜色失真（强度=0.5），省略高斯模糊。我们使用学习率 {0.5, 1.0, 1.5}、温度 {0.1, 0.5, 1.0} 和批次大小 {256, 512, 1024, 2048, 4096} 进行预训练。其余设置（包括优化器、权重衰减等）与我们的 ImageNet 训练相同。

^15 值得注意的是，虽然 CIFAR-10 图像远小于 ImageNet 图像且图像大小在不同示例间没有差异，但带调整大小的裁剪对于对比学习仍然是一种非常有效的增强。

我们使用批次大小 1024 训练的最佳模型可以达到 94.0% 的线性评估准确率，而使用相同架构和批次大小的监督基线为 95.1%。在 CIFAR-10 上报告线性评估结果的最佳自监督模型是 AMDIM（Bachman et al., 2019），它使用比我们大 25 倍的模型达到了 91.2%。我们注意到，我们的模型可以通过结合额外的数据增强以及使用更合适的基网络来改进。

**不同批次大小和训练步数下的性能** 图 B.7 显示了不同批次大小和训练步数下的线性评估性能。结果与我们在 ImageNet 上的观察一致，尽管最大的批次大小 4096 似乎在 CIFAR-10 上导致了轻微的性能下降。

**不同批次大小下的最优温度** 图 B.8 显示了使用三种不同温度在各种批次大小下训练的模型的线性评估。我们发现，当训练收敛时（例如训练 epochs > 300），{0.1, 0.5, 1.0} 中的最优温度为 0.5，并且似乎与批次大小无关。然而，$\tau = 0.1$ 的性能随着批次大小增加而提高，这可能表明最优温度向 0.1 方向有小幅偏移。

### B.10. 其他损失函数的调优

对 NT-Xent 损失最有效的学习率可能对其他损失函数不是好的学习率。为了确保公平比较，我们也为间隔损失和逻辑损失调整超参数。具体来说，我们对两个损失函数调整学习率 {0.01, 0.1, 0.3, 0.5, 1.0}。我们进一步对间隔损失调整间隔 {0, 0.4, 0.8, 1.6}，对逻辑损失调整温度 {0.1, 0.2, 0.5, 1.0}。为简单起见，我们仅考虑来自一个增强视图的负样本（而不是两侧），这略微损害性能但确保了公平比较。

## C. 与相关方法的进一步比较

正如我们在正文中指出的，SimCLR 的大多数单个组件都曾在先前工作中出现过，改进的性能是这些设计选择组合的结果。表 C.1 提供了我们方法与先前方法在设计选择上的高层次比较。与先前工作相比，我们的设计选择通常更简单。

**表 C.1.** 每种方法的设计选择和训练设置（为在 ImageNet 上获得最佳结果）的高层次比较。注意，这里提供的描述是通用的；即使两种方法匹配，公式和实现也可能不同（例如颜色增强）。更多详情请参考原始论文。#示例被分割成多个 patches，这扩大了有效批次大小。*使用了记忆库。

下面，我们提供我们的方法与近期提出的对比表征学习方法的深入比较：

- **DIM/AMDIM**（Hjelm et al., 2018; Bachman et al., 2019）通过预测 ConvNet 的中间层来实现全局到局部/局部到邻近预测。ConvNet 是一个经过修改以对网络感受野施加显著约束的 ResNet（例如，将许多 $3 \times 3$ 卷积替换为 $1 \times 1$ 卷积）。在我们的框架中，我们通过随机裁剪（带调整大小）并使用两个增强视图的最终表征进行预测，将预测任务与编码器架构解耦，因此我们可以使用标准且更强大的 ResNet。我们的 NT-Xent 损失函数利用归一化和温度来限制相似度分数的范围，而他们使用带正则化的 tanh 函数。我们使用更简单的数据增强策略，而他们使用 FastAutoAugment 以获得最佳结果。

- **CPC v1 和 v2**（Oord et al., 2018; Hénaff et al., 2019）使用确定性策略将示例分割成 patches，并使用上下文聚合网络（PixelCNN）聚合这些 patches 来定义上下文预测任务。基编码器网络仅看到 patches，这些 patches 远小于原始图像。我们将预测任务与编码器架构解耦，因此不需要上下文聚合网络，并且我们的编码器可以查看更宽分辨率范围的图像。此外，我们使用 NT-Xent 损失函数，它利用归一化和温度，而他们使用非归一化的基于交叉熵的目标函数。我们使用更简单的数据增强。

- **InstDisc、MoCo、PIRL**（Wu et al., 2018; He et al., 2019; Misra & van der Maaten, 2019）推广了最初由 Dosovitskiy et al. (2014) 提出的 Exemplar 方法，并利用显式的记忆库。我们不使用记忆库；我们发现，使用更大的批次大小，批次内负样本采样就足够了。我们还利用非线性投影头，并使用投影头之前的表征。虽然我们使用类似类型的增强（例如随机裁剪和颜色失真），但具体参数可能不同。

- **CMC**（Tian et al., 2019）为每个视图使用独立的网络，而我们简单地为所有随机增强视图使用共享的单一网络。数据增强、投影头和损失函数也不同。我们使用更大的批次大小而不是记忆库。

- 而 **Ye et al. (2019)** 最大化同一图像的增强副本和未增强副本之间的相似性，我们在框架的两个分支上对称地应用数据增强（图 2）。我们还在基特征网络的输出上应用非线性投影，并使用投影网络之前的表征，而 Ye et al. (2019) 使用线性投影的最终隐藏向量作为表征。当使用多个加速器以大批次大小训练时，我们使用全局 BN 来避免可能大幅降低表征质量的捷径。
