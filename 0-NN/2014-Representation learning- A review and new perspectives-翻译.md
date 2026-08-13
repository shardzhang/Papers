# 表示学习：综述与新视角

> Yoshua Bengio, Aaron Courville, Pascal Vincent | University of Montreal, CIFAR

本文系统综述了表示学习的核心方法与理论基础，涵盖概率模型、自编码器、流形学习和深度网络等。核心发现是——**好的表示能够解耦数据背后的生成因子，而深度架构和分布式表示是实现这一目标的关键路径**。

核心内容：
- 机器学习系统的成功严重依赖于数据表示（特征）的选择，手工特征工程耗时且难以跨任务迁移
- 表示学习通过自动学习数据的有用变换，将原始数据映射到更容易完成后续任务的空间
- 本文从概率模型、自编码器、稀疏编码、流形学习、深度网络等多个角度梳理了表示学习的方法体系
- 分布式表示、深度架构和解耦因子变分是好的表示的三大核心原则

关键发现：
- 分布式表示能以指数级效率泛化：$k$ 个二值特征可表示 $2^k$ 个不同区域，远超局部表示的线性效率
- 深度架构通过逐层组合简单概念构建复杂抽象，ReLU等非饱和激活函数显著改善了深度网络训练
- 卷积网络在目标识别任务中取得突破：在ImageNet等大规模视觉任务上达到当时最优性能
- 解耦潜在因子是表示学习的根本目标，但如何设计无监督学习目标以实现完全解耦仍是开放问题

---

## 摘要

人们对深度学习重新产生兴趣的原因之一是，一系列令人印象深刻的实验结果表明，深度网络在实际任务中优于浅层网络。然而，在更好地理解深度学习背后的原理方面取得的进展更为缓慢。本文的目标是回顾在表示学习方面使用的几种不同方法，并更好地理解深度学习背后的理论问题和实际挑战。

**关键词**：表示学习，深度学习，自编码器，受限玻尔兹曼机，分布式表示，流形学习

## 1 引言

机器学习算法的成功很大程度上取决于数据的表示（或称为特征）。例如，当语音的底层因素（如说话者的特征、音素、周围的音素、麦克风的特性等）被很好地分离时，线性分类器就能很好地对语音进行分类。因此，许多机器学习算法的性能严重依赖于数据表示的选择。这在机器学习的实际应用中催生了一个叫做"特征工程"的实践领域。在传统的特征工程中，领域专家投入大量时间设计适合特定任务的特征。

然而，对于许多任务来说，我们不知道应该提取哪些特征。例如，在视觉对象识别中，虽然我们知道像素之间的局部相关性可能很有用，但我们不知道应该使用哪些特征来处理由于光照变化、背景变化、同一类内形状变化、视角变化等引起的各种变化。更根本的问题是，我们能否找到一种方法来自动学习数据的良好表示？

表示学习（Representation Learning）是一组允许机器接收原始数据并自动发现进行检测或分类所需的表示的方法。深度学习（Deep Learning）方法是具有多层非线性处理单元的表示学习方法，每一层学习将表示从上一层变换到更高层的、稍微更抽象的表示。随着数据表示的层级结构变得更深，我们就获得了关于数据的更抽象的描述。

本文的组织结构如下。第2节回顾了以前的综述和讨论。第3节讨论了使表示好的因素。第4节讨论了特征学习算法的主要家族。第5节讨论了关于流形学习的想法。第6节讨论了深度网络。第7节讨论了未来研究的开放问题。第8节总结了本文。

## 2 历史背景

表示学习并不是一个新的概念。在机器学习和模式识别的历史上，数据表示的选择一直被认为是至关重要的。在统计模式识别的早期实践中，设计特征提取器是整个系统设计中的关键步骤。在许多实际应用中，特征设计需要大量的领域专业知识，这限制了机器学习方法的通用性。

### 2.1 神经网络的复兴

在20世纪80年代和90年代，反向传播算法的成功使得训练多层神经网络成为可能。然而，由于训练深层网络的困难（如梯度消失问题），神经网络的研究在2000年代初期进入了低潮期。

2006年，Hinton等人[46]提出了深度信念网络（Deep Belief Networks, DBN），通过逐层无监督预训练的方法成功训练了深层网络。这项工作标志着深度学习的复兴，引发了大量后续研究。

### 2.2 深度学习的成功

自2006年以来，深度学习在多个领域取得了显著成功：

**语音识别**：深度神经网络在音素识别和大词汇量连续语音识别任务中取得了突破性成果[31, 32]。使用深度神经网络进行声学建模，在TIMIT音素识别任务上将错误率从26.1%降低到了20.7%。

**目标识别**：卷积神经网络在ImageNet大规模视觉识别挑战赛中取得了显著成果[67]。2012年，Krizhevsky等人[67]的AlexNet将ImageNet分类错误率从26%降低到15.3%。

**自然语言处理**：词嵌入和递归神经网络在语言模型、机器翻译等任务中取得了进展[11, 90]。

**多任务和迁移学习**：深度学习学到的表示在多个任务间具有良好的迁移性[96, 122]。

## 3 什么是好的表示？

要理解表示学习，首先需要理解什么样的表示是"好的"。本节讨论了判断表示质量的几个核心原则。

### 3.1 AI中表示学习的先验

好的表示应该能够捕获数据背后的底层结构。在人工智能的背景下，一个好的表示应该使后续的学习任务变得更容易。Bengio等人[12]提出了以下关于好的表示的先验假设：

**平滑性**：如果两个输入 $x_1$ 和 $x_2$ 在原始空间中相近，那么它们的表示也应该相近。这一假设是许多学习算法的基础。

**多解释因子**：观察到的数据是由多个潜在的解释因子生成的。好的表示应该能够分离（或解耦）这些因子。

**流形假设**：高维数据实际上集中在低维流形附近。好的表示应该能够发现这些流形结构。

**共享因子**：不同的任务和领域可能共享一些底层的解释因子。好的表示应该能够捕获这些共享因子，从而促进迁移学习。

### 3.2 平滑性和维度灾难

许多机器学习算法假设目标函数是平滑的，即相似的输入应该有相似的输出。然而，在高维空间中，平滑性假设面临维度灾难的挑战。

考虑一个简单的例子：假设我们将每个维度离散化为 $k$ 个值，那么 $d$ 维空间中的可能状态数为 $k^d$，它随 $d$ 指数增长。为了在所有这些状态上获得可靠的估计，我们需要指数级的样本数量。

这就是维度灾难的核心问题。它表明，仅仅依靠平滑性假设是不够的。我们需要更强大的先验知识来约束学习问题。表示学习的一个关键目标就是找到数据的低维表示，从而规避维度灾难。

![图1](/Users/dazhang/PycharmProject/Papers/0-NN/.picture/2014-Representation learning- A review and new perspectives-fig1.png)

### 3.3 分布式表示

分布式表示（Distributed Representation）是表示学习中最重要的概念之一[48, 7]。在分布式表示中，概念不是由单个神经元（局部表示）或集合中的特定元素（符号表示）来表示的，而是由多个神经元的激活模式来表示的。

分布式表示的关键优势在于其组合性（compositionality）和表达效率：

**组合性**：如果用 $k$ 个二值特征来表示数据，每个特征可以取两个值，那么可以表示 $2^k$ 个不同的概念区域。这些特征可以自由组合，形成指数级数量的新概念。

**表达效率**：局部表示（如one-hot编码）需要与可表示的概念数量成正比的维度，而分布式表示只需要对数级别的维度。例如，要表示100万个概念，局部表示需要100万个神经元，而分布式表示可能只需要 $\log_2(10^6) \approx 20$ 个二值特征。

**泛化能力**：分布式表示允许模型在训练中未见过的概念组合上进行泛化。如果模型分别学习了"红"和"车"的概念，它可以理解"红色的车"，即使训练数据中没有明确包含这个组合。

然而，分布式表示也有其局限性。它们可能更难解释，训练可能需要更多数据，而且并非所有概念都适合分布式表示。

![图2](/Users/dazhang/PycharmProject/Papers/0-NN/.picture/2014-Representation learning- A review and new perspectives-fig2.png)

### 3.4 表示的深度

深度是表示学习中的另一个关键概念。深度架构通过组合多个层次的简单变换来构建复杂的表示。

**深度的理论优势**：某些函数用深度架构表示可能比用浅层架构表示效率高得多。例如，某些多项式函数可以用深度网络以 $O(\log n)$ 的节点数表示，但用浅层网络需要 $O(2^n)$ 个节点[5, 105]。

**深度的实证优势**：在许多实际任务中，深度网络确实比浅层网络表现更好。这在语音识别[31]和计算机视觉[67]中得到了广泛验证。

**深度的生物学依据**：大脑皮层的分层处理结构也被认为支持深度表示的重要性。视觉皮层的层次结构从简单特征（如边缘）逐步构建到复杂特征（如物体）。

### 3.5 解耦因子变分

好的表示应该能够解耦（disentangle）数据背后的生成因子。这里的"因子"是指生成观察数据的潜在变量。

考虑人脸图像的例子。潜在的生成因子可能包括：光照条件、人脸身份、表情、姿态、年龄等。一个好的表示应该能够独立地捕获每个因子，使得改变一个因子（如光照）不会影响其他因子的表示。

然而，实现完全解耦的表示是一个极具挑战性的无监督学习问题。因为没有监督信号来告诉模型哪些因子是重要的，以及如何分离它们。无监督学习算法可以利用数据中的统计规律（如独立性假设）来尝试解耦因子，但成功程度取决于数据和算法的匹配程度。

解耦因子变分的重要性体现在以下方面：

- **样本效率**：如果因子被很好地解耦，那么关于一个因子的知识可以更容易地迁移到涉及同一因子的新任务。
- **鲁棒性**：解耦的表示对输入的小变化更不敏感，因为每个因子都被独立地表示。
- **可解释性**：解耦的表示更容易被人理解，每个维度可能对应一个可解释的概念。

## 4 特征学习算法

本节回顾了主要的特征学习算法家族，包括概率模型、线性逆问题、自编码器、正则化自编码器和预测稀疏分解。

### 4.1 概率模型

概率模型通过定义数据的联合概率分布来进行表示学习。它们可以分为两大类：有向图模型和无向图模型。

#### 4.1.1 有向图模型

有向图模型（也称为信念网络）使用有向无环图来表示变量之间的依赖关系。典型的有向图模型包括：

**概率主成分分析（PPCA）**[119]是主成分分析的概率版本。它假设数据由低维潜在变量通过线性变换加高斯噪声生成：

$$
\mathbf{x} = \mathbf{W}\mathbf{z} + \boldsymbol{\mu} + \boldsymbol{\epsilon}
$$

其中 $\mathbf{z}$ 是潜在变量，$\mathbf{W}$ 是权重矩阵，$\boldsymbol{\epsilon}$ 是高斯噪声。

**稀疏编码模型**[91]假设数据可以表示为少量基向量的线性组合。潜在变量 $\mathbf{z}$ 具有稀疏先验（如Laplace分布），使得大多数元素为零或接近零。

**Sigmoid信念网络**[112]使用二值潜在变量和sigmoid激活函数。它们具有更强的表达能力，但推理更加困难。

#### 4.1.2 无向图模型

无向图模型（也称为马尔可夫随机场）使用无向图来表示变量之间的关系。最重要的无向图模型之一是受限玻尔兹曼机（RBM）。

**受限玻尔兹曼机（RBM）**[45, 114]是一种具有二层结构（可见层和隐藏层）的能量基模型，层内没有连接。其能量函数定义为：

$$
E(\mathbf{v}, \mathbf{h}) = -\mathbf{b}^\top\mathbf{v} - \mathbf{c}^\top\mathbf{h} - \mathbf{v}^\top\mathbf{W}\mathbf{h}
$$

其中 $\mathbf{v}$ 是可见单元，$\mathbf{h}$ 是隐藏单元，$\mathbf{W}$ 是权重矩阵，$\mathbf{b}$ 和 $\mathbf{c}$ 是偏置向量。

RBM的训练通常使用对比散度（Contrastive Divergence, CD）算法[45]。CD-$k$ 算法通过运行 $k$ 步吉布斯采样来近似对数似然的梯度。尽管CD不是真正的梯度，但在实践中已被证明是有效的。

**深度信念网络（DBN）**[46]是由多个RBM堆叠而成的生成模型。DBN的训练采用逐层贪心预训练：首先训练底层RBM，然后将其隐藏层的激活作为上层RBM的输入，依此类推。这种逐层预训练方法在2006年标志着深度学习的复兴。

### 4.2 线性逆问题

线性逆问题试图从观察到的信号中恢复原始的稀疏表示。这类问题在信号处理和压缩感知中有广泛应用。

**基追踪（Basis Pursuit）**[20]和**匹配追踪（Matching Pursuit）**[82]是解决稀疏编码问题的经典算法。它们的目标是找到最稀疏的系数向量 $\mathbf{z}$，使得 $\mathbf{x} \approx \mathbf{W}\mathbf{z}$。

**压缩感知（Compressed Sensing）**[27, 37]表明，如果信号是稀疏的，那么可以用远少于奈奎斯特定理要求的测量数来精确重建信号。这与表示学习中的稀疏表示思想密切相关。

### 4.3 自编码器

![图7](/Users/dazhang/PycharmProject/Papers/0-NN/.picture/2014-Representation learning- A review and new perspectives-fig7.png)

自编码器（Auto-Encoder）是一种通过学习将输入编码为低维表示，然后从该表示重构输入的神经网络。它由编码器 $f$ 和解码器 $g$ 组成：

**编码器**：$\mathbf{h} = f(\mathbf{x}) = s_f(\mathbf{W}\mathbf{x} + \mathbf{b})$

**解码器**：$\hat{\mathbf{x}} = g(\mathbf{h}) = s_g(\mathbf{W}'\mathbf{h} + \mathbf{b}')$

**重构误差**：$L(\mathbf{x}, \hat{\mathbf{x}}) = \|\mathbf{x} - \hat{\mathbf{x}}\|^2$

训练目标是最小化重构误差。当隐藏层维度小于输入维度时，自编码器被迫学习数据的压缩表示。

**线性自编码器与PCA**：当编码器和解码器都是线性函数时，自编码器等价于主成分分析（PCA）[56]。此时，隐藏层学到的表示张成了数据的主子空间。

#### 4.3.1 稀疏自编码器

稀疏自编码器[98]在自编码器的基础上增加了稀疏性约束。具体来说，它要求隐藏单元的平均激活值接近一个较小的值 $\rho$（如0.05）。这通过在损失函数中添加KL散度惩罚项实现：

$$
L_{\text{sparse}} = L(\mathbf{x}, \hat{\mathbf{x}}) + \beta \sum_{j=1}^{s} \text{KL}(\rho \| \hat{\rho}_j)
$$

其中 $\hat{\rho}_j$ 是第 $j$ 个隐藏单元在训练集上的平均激活值，$\beta$ 控制稀疏性惩罚的权重。

稀疏表示在神经科学中有生物学依据：视觉皮层中的神经元在面对自然图像时表现出稀疏激活模式[92]。

### 4.4 正则化自编码器

自编码器的一个重要问题是：如果不加约束，它可能学到恒等映射（即简单地复制输入）。正则化自编码器通过在重构误差的基础上添加各种正则化项来解决这个问题。

#### 4.4.1 去噪自编码器（DAE）

![图3](/Users/dazhang/PycharmProject/Papers/0-NN/.picture/2014-Representation learning- A review and new perspectives-fig3.png)

去噪自编码器[128, 129]是正则化自编码器中最重要的变体之一。它的核心思想是：对输入添加噪声，然后训练网络从噪声版本重构原始输入。

**训练过程**：
1. 对输入 $\mathbf{x}$ 添加噪声得到 $\tilde{\mathbf{x}}$（如添加高斯噪声或随机置零部分输入）
2. 编码：$\mathbf{h} = f(\tilde{\mathbf{x}})$
3. 解码：$\hat{\mathbf{x}} = g(\mathbf{h})$
4. 最小化重构误差：$L = \|\mathbf{x} - \hat{\mathbf{x}}\|^2$

去噪自编码器的动机是：一个好的表示应该对输入的小变化具有鲁棒性。通过训练网络从噪声版本恢复原始输入，我们迫使网络学到数据流形的局部结构。

**与分数估计的关系**：Vincent[129]证明，去噪自编码器实际上在估计数据分布的分数函数 $\nabla_{\tilde{\mathbf{x}}} \log p(\tilde{\mathbf{x}})$。这一发现建立了去噪自编码器与分数匹配[68]之间的理论联系。

**去噪自编码器的堆叠**：多个去噪自编码器可以逐层堆叠，形成深度去噪网络。这种方法在特征学习中表现出色。

#### 4.4.2 
收缩自编码器（CAE）

收缩自编码器[109, 110]通过惩罚编码器函数的Jacobian矩阵的Frobenius范数来正则化：

$$
L_{\text{contractive}} = L(\mathbf{x}, \hat{\mathbf{x}}) + \lambda \left\| \frac{\partial f(\mathbf{x})}{\partial \mathbf{x}} \right\|_F^2
$$

这鼓励编码器函数对输入的小变化不敏感，从而使表示更加鲁棒。收缩自编码器与去噪自编码器有密切联系，两者都致力于学习对输入扰动鲁棒的表示。

#### 4.4.3 近似编码器与评分匹配

Alain和Bengio[3]的研究表明，去噪自编码器和收缩自编码器都与评分匹配（Score Matching）有密切联系。评分匹配是一种估计概率模型参数的方法，它避免了计算配分函数的困难。

去噪自编码器可以看作是在估计数据分布的评分函数 $\nabla_{\mathbf{x}} \log p(\mathbf{x})$。这一发现为自编码器的概率解释提供了理论基础。

### 4.5 预测稀疏分解（PSD）

预测稀疏分解[65]结合了稀疏编码和参数化编码器的优点。它训练一个参数编码器来预测稀疏编码的解，从而避免了在测试时运行优化算法的开销。

PSD的训练目标包括两部分：
1. 稀疏编码目标：$\|\mathbf{x} - \mathbf{W}\mathbf{z}\|^2 + \lambda\|\mathbf{z}\|_1$
2. 预测目标：$\|\mathbf{z} - f(\mathbf{x})\|^2$

其中 $f(\mathbf{x})$ 是参数编码器。通过联合优化这两个目标，PSD学习了一个能够快速预测稀疏编码的编码器。

### 4.6 自编码器与流形学习的关系

去噪自编码器的一个重要理论结果是，它们可以学习数据流形的局部切空间[129]。具体来说，去噪自编码器的重构方向对应于数据流形的切方向。

更正式地说，如果数据集中在低维流形上，那么![图3](/Users/dazhang/PycharmProject/Papers/0-NN/.picture/2014-Representation learning- A review and new perspectives-fig3.png)

去噪自编码器的重构函数 $r(\mathbf{x})$ 会将带噪声的点投影回流形上。在零噪声的极限情况下，
重构方向对应于流形的切方向。

这一发现建立了自编码器与流形学习之间的深刻联系，表明自编码器不仅是一种特征学习方法，也是一种流形学习方法。

## 5 流形学习

流形学习（Manifold Learning）基于这样的假设：高维数据实际上集中在低维流形附近。表示学习的一个重要目标就是发现这些流形结构。

### 5.1 局部性与稀疏性

流形学习的核心思想是利用数据的局部结构。大多数流形学习算法假设在流形的局部区域可以使用线性近似。

**局部线性嵌入（LLE）**[107]假设每个数据点可以表示为其近邻的线性组合。它首先找到每个点的近邻，然后计算重构权重，最后在低维空间中保持这些权重关系。

**拉普拉斯特征映射（Laplacian Eigenmaps）**[8]和**谱聚类**方法利用数据的图结构来发现低维表示。它们构建一个邻接图，然后计算图的拉普拉斯矩阵的特征向量。

**局部保持投影（LPP）**[50]是一种线性降维方法，它试图在低维空间中保持数据的局部邻域关系。

稀疏性在流形学习中起着重要作用。稀疏表示（每个数据点可以表示为少数几个基的线性组合）自然地揭示了数据的局部结构。稀疏编码和压缩感知与流形学习有密切联系[27]。

### 5.2 流形与切空间

更正式的流形学习方法基于微分几何的概念。

**切空间**：在流形上的每一点，存在一个切空间，它是流形在该点的局部线性近似。对于数据表示来说，切空间告诉我们哪些方向对应于数据的有意义的变化。

**局部坐标系**：流形学习的目标之一是建立数据的局部坐标系，使得每个坐标轴对应于一个有意义的变化因子。

**去噪自编码器与切空间**：如前所述，去噪自编码器的重构方向对应于流形的切方向。具体来说，如果 $r(\mathbf{x})$ 是去噪自编码器的重构函数，那么在零噪声极限下：

$$
\mathbf{x} - r(\mathbf{x}) \propto \sum_i \frac{\partial \log p(\mathbf{x})}{\partial x_i} \mathbf{e}_i
$$

这表明重构误差方向对应于数据密度的梯度方向，即垂直于流形的方向。

### 5.3 流形学习的挑战

流形学习面临几个重要挑战：

**邻域选择**：大多数流形学习算法需要定义"局部邻域"，但如何选择合适的邻域大小是一个困难的问题。太小的邻域会导致数据碎片化，太大的邻域会破坏局部线性假设。

**流形假设的局限性**：并非所有数据都集中在光滑的低维流形上。数据可能分布在多个流形上，或者具有更复杂的拓扑结构。

**外推问题**：大多数流形学习算法只能处理训练数据中的点，难以对训练数据之外的新点进行映射（out-of-sample problem）。

**与深度学习的结合**：深度学习提供了学习复杂非线性映射的能力，可以看作是流形学习的一种扩展。深度网络可以学习将数据从原始空间映射到更容易分离的表示空间。

## 6 深度网络

深度网络是表示学习的核心工具。本节讨论深度网络的训练方法、多任务学习和卷积网络。

### 6.1 逐层贪心训练

训练深度网络的一个关键挑战是优化困难。在2006年之前，直接训练深层网络往往效果不佳，因为梯度消失问题使得底层的参数难以更新。

**逐层贪心预训练**[46]是解决这一问题的重要方法。其核心思想是：

1. **第一层**：使用无监督学习算法（如RBM或自编码器）学习第一层的参数
2. **后续层**：将上一层的输出作为输入，训练下一层
3. **微调**：在所有层都预训练后，使用反向传播对整个网络进行端到端的微调

这种方法之所以有效，有几个理论解释：

**正则化效应**：预训练将网络参数初始化到一个好的区域，起到了正则化的作用[123]。

**优化效应**：预训练将参数初始化到损失函数的一个好的局部最小值附近，使得后续的微调更容易收敛。

**深度的作用**：Bengio等人[15]的研究表明，深度网络能够学习更抽象的特征，这些特征在不同任务之间具有更好的迁移性。

然而，随着训练技术的进步（如ReLU激活函数[88]、Dropout正则化[120]、批归一化[55]等），直接训练深度网络已经成为可能，逐层预训练不再是必需的。

### 6.2 多任务学习与迁移学习

多任务学习（Multi-Task Learning）[22]通过同时学习多个相关任务来改善泛化性能。在深度学习的背景下，多任务学习通常通过共享底层表示来实现。

**共享表示**：不同的任务共享底层的特征提取层，而每个任务有自己的顶层分类层。这种架构迫使网络学习对所有任务都有用的通用表示。

**迁移学习**：在一个大规模数据集上预训练的深度网络可以迁移到新的任务。例如，在ImageNet上预训练的卷积神经网络可以作为其他视觉任务的特征提取器[140]。

**领域自适应**：当训练数据和测试数据来自不同分布时，需要进行领域自适应。深度网络可以通过学习领域不变的表示来解决这一问题[34]。

多任务学习和迁移学习的成功表明，深度网络学到的表示能够捕获跨任务和跨领域的通用知识。

### 6.3 卷积网络

卷积神经网络（Convolutional Neural Network, CNN）[72, 73]是深度学习在计算机视觉中取得突破的关键架构。

#### 6.3.1 卷积操作

卷积层通过在输入上滑动一组滤波器来提取特征。对于输入 $\mathbf{X}$ 和滤波器 $\mathbf{W}$，卷积操作可以表示为：

$$
y_{i,j} = \sum_{m}\sum_{n} x_{i+m, j+n} \cdot w_{m,n} + b
$$

卷积操作的两个关键特性：

**局部连接**：每个输出只与输入的一个局部区域相连，这利用了图像的空间局部性。

**权重共享**：同一个滤波器在整个输入上共享权重，这大大减少了参数数量，并赋予了网络平移不变性。

#### 6.3.2 池化操作

池化层通过对局部区域进行下采样来减小特征图的空间尺寸。最常用的是最大池化：

$$
y_{i,j} = \max_{(m,n) \in \mathcal{R}_{i,j}} x_{m,n}
$$

池化操作提供了平移不变性，并减小了后续层的计算量。

#### 6.3.3 CNN的层次结构

典型的CNN由多个卷积层和池化层交替堆叠而成，最后接全连接层进行分类。这种层次结构使得网络能够学习从低级特征（如边缘、纹理）到高级语义特征（如物体部件、物体类别）的层次化表示。

#### 6.3.4 CNN的成功

CNN在多个视觉任务中取得了突破性成果：

**图像分类**：AlexNet[67]在ImageNet分类任务上将错误率从26%降低到15.3%。后续的VGGNet[117]、GoogLeNet[124]和ResNet[42]进一步提升了性能。

**目标检测**：R-CNN[38]及其变体将CNN应用于目标检测任务，取得了显著成果。

**语义分割**：全卷积网络（FCN）[84]将CNN应用于像素级的语义分割。

**人脸识别**：DeepFace[126]和FaceNet[113]在人脸识别任务上达到了接近人类的性能。

CNN的成功不仅体现在性能上，还体现在其学到的特征具有良好的可迁移性。在ImageNet上预训练的CNN可以作为许多其他视觉任务的通用特征提取器。

### 6.4 自编码器与深度网络的关系

自编码器可以看作是深度网络的一种特殊情况。当编码器和解码器都是深度网络时，自编码器就是一个深度网络。

**逐层训练**：深度自编码器可以通过逐层训练来初始化。首先训练一个浅层自编码器，然后将其编码器的输出作为下一层自编码器的输入，依此类推。

**与DBN的联系**：深度自编码器与深度信念网络有密切联系。Bengio等人[14]证明，在一定条件下，RBM的逐层训练等价于自编码器的逐层训练。

**生成能力**：自编码器也可以用于生成数据。通过在潜在空间中采样并运行解码器，可以生成新的数据样本。变分自编码器（VAE）[62]和生成对抗网络（GAN）[39]是两种更强大的生成模型。

### 6.5 非饱和激活函数

激活函数的选择对深度网络的训练至关重要。

**Sigmoid和Tanh**：传统的激活函数如sigmoid和tanh在输入值较大或较小时梯度接近零，导致梯度消失问题。

**ReLU（Rectified Linear Unit）**[88]：$\max(0, x)$ 是一种非饱和激活函数，它在正区间梯度恒为1，有效缓解了梯度消失问题。ReLU的生物学动机来自神经科学中对神经元激活模式的研究。

**ReLU的变体**：
- **Leaky ReLU**[81]：$\max(\alpha x, x)$，其中 $\alpha$ 是一个小正数
- **PReLU（Parametric ReLU）**[42]：将 $\alpha$ 作为可学习参数
- **ELU（Exponential Linear Unit）**[28]：在负区间使用指数函数

ReLU的使用使得直接训练深度网络成为可能，不再需要逐层预训练。

### 6.6 正则化技术

深度网络容易过拟合，因此需要有效的正则化技术。

**Dropout**[120]：在训练时随机将一部分隐藏单元的输出置零。这可以看作是训练大量子网络的集成。Dropout的典型保留概率为0.5（隐藏层）和0.8（输入层）。

**批归一化（Batch Normalization）**[55]：对每一层的输入进行归一化，使其均值为0、方差为1。这加速了训练并起到了正则化作用。

**权重衰减**：对权重施加L2正则化，防止权重过大。

**数据增强**：通过对训练数据进行变换（如旋转、翻转、缩放等）来增加训练样本数量。

## 7 挑战与未来工作

尽管表示学习取得了显著进展，仍有许多重要挑战需要解决。

### 7.1 无监督学习的评估

无监督学习面临的一个核心问题是评估。由于没有明确的标签，很难衡量无监督学习算法学到的表示的质量。

**代理任务**：一种常见的方法是使用学到的表示来完成下游任务（如分类），并用下游任务的性能来评估表示的质量。然而，这种评估方式可能不能全面反映表示的所有方面。

**生成质量**：对于生成模型，可以使用生成样本的质量来评估。然而，生成质量的评估本身就是一个困难的问题。

**信息论指标**：互信息等信息论指标可以用来衡量表示包含的信息量，但计算互信息通常是困难的。

### 7.2 学习好的推理

在概率模型中，推理（从观察推断潜在变量）通常是一个困难的问题。虽然变分推理和采样方法取得了一定进展，但如何学习高效且准确的推理仍然是一个挑战。

**摊销推理（Amortized Inference）**：使用参数化编码器来预测潜在变量的后验分布，如变分自编码器中的做法。这种方法避免了对每个数据点运行优化算法的开销。

**重要性加权自编码器（IWAE）**[20]：通过重要性采样来改进变分下界的估计。

**推理网络的训练**：如何训练推理网络使其输出准确的后验估计，仍然是一个活跃的研究方向。

### 7.3 解耦表示的学习

学习完全解耦的表示是一个重大挑战。虽然已经有一些尝试，如$\beta$-VAE[44]等，但如何在没有监督的情况下学习完全解耦的表示仍然是一个开放问题。

**独立成分分析（ICA）**[57]假设潜在因子是统计独立的，并尝试恢复这些独立因子。然而，ICA通常假设线性混合模型，对于非线性生成过程，问题变得更加困难。

**解耦表示的度量**：如何衡量表示的解耦程度？虽然已经提出了一些度量方法，但还没有公认的标准。

### 7.4 大规模无监督学习

如何在大规模数据集上进行有效的无监督学习是一个实际挑战。现有的无监督学习算法在计算效率和可扩展性方面仍有改进空间。

**分布式训练**：如何在分布式环境中高效地训练无监督模型。

**在线学习**：如何在数据流到来时实时更新表示，而不是在固定数据集上进行批处理。

**计算效率**：如何设计计算高效的无监督学习算法，使其能够处理大规模数据。

### 7.5 从序列数据中学习表示

序列数据（如文本、语音、时间序列）的表示学习面临特殊挑战。

**长距离依赖**：序列中的长距离依赖关系难以捕获。虽然LSTM[49]和GRU[26]在一定程度上解决了这一问题，但对于非常长的序列，仍然存在挑战。

**层次化表示**：如何学习序列的层次化表示，从底层的时间特征到高层的语义特征。

**多模态序列**：如何学习同时包含多种模态信息（如视频中的视觉和音频）的序列表示。

### 7.6 理论基础

表示学习的理论基础仍然不完善。

**深度的优势**：虽然已经有一些理论解释深度网络的优势[5, 105]，但对于深度网络为什么在实践中如此成功，仍然缺乏完整的理论解释。

**优化景观**：深度网络的损失函数景观非常复杂。为什么简单的优化算法（如SGD）能够找到好的解？这是一个重要的理论问题。

**泛化理论**：深度网络的参数数量通常远多于训练样本数量，但它们并没有严重过拟合。传统的泛化理论（如VC维理论）无法解释这一现象。

### 7.7 表示的可解释性

深度网络学到的表示通常被认为是"黑盒"的。如何理解和解释这些表示是一个重要问题。

**可视化**：通过可视化隐藏单元的激活模式或最大激活输入来理解网络学到的特征。

**探测实验（Probing Experiments）**[21]：训练简单的线性分类器来从表示中提取特定信息，以此来分析表示包含的信息。

**概念激活向量（CAV）**[115]：识别表示空间中与特定概念对应的方向。

## 8 结论

表示学习是机器学习中的一个核心问题，它试图自动从数据中发现有用的表示。本文综述了表示学习的多种方法，包括概率模型、自编码器、流形学习和深度网络。

好的表示应该具有以下特性：

**平滑性**：相似的输入应该有相似的表示。

**分布式**：概念应该由多个特征的组合来表示，而不是由单个特征来表示。分布式表示能够以指数级效率表示大量概念。

**深度**：通过组合多个层次的简单变换，深度表示能够构建复杂的抽象概念。某些函数用深度表示比用浅层表示效率高得多。

**解耦**：好的表示应该能够分离数据背后的生成因子，使得每个因子被独立地表示。

表示学习的未来发展可能集中在以下方向：

- 更强大的无监督学习算法，能够学习更好的表示
- 更好的理论理解，解释深度学习为什么有效
- 更有效的评估方法，衡量表示的质量
- 更好的算法，学习解耦的表示
- 更大规模的表示学习系统

表示学习不仅是深度学习的基础，也是人工智能的核心挑战之一。只有当机器能够从原始数据中自动发现有用的表示时，才能实现真正通用的人工智能。

## 参考文献

[1] J. B. Allen and C. L. Shorshort. Short-term spectral analysis, synthesis, and modification by discrete Fourier transform. *IEEE Transactions on Acoustics, Speech, and Signal Processing*, 25(3):235–238, 1977.

[2] Y. Altun, I. Tsochantaridis, and T. Hofmann. Hidden Markov support vector machines. In *ICML*, 2003.

[3] G. Alain and Y. Bengio. What regularized auto-encoders learn from the data-generating distribution. Technical Report arXiv:1211.4721, Université de Montréal, 2012.

[4] S. Amari. Backpropagation and stochastic gradient descent method. *Neurocomputing*, 5(4-5):185–196, 1993.

[5] Y. Bengio and O. Delalleau. On the expressive power of deep architectures. In *ALT*, 2011.

[6] Y. Bengio, R. Ducharme, P. Vincent, and C. Janvin. A neural probabilistic language model. *JMLR*, 3:1137–1155, 2003.

[7] Y. Bengio and Y. LeCun. Scaling learning algorithms towards AI. In L. Bottou, O. Chapelle, D. DeCoste, and J. Weston, editors, *Large-Scale Kernel Machines*. MIT Press, 2007.

[8] M. Belkin and P. Niyogi. Laplacian eigenmaps for dimensionality reduction and data representation. *Neural Computation*, 15(6):1373–1396, 2003.

[9] A. L. Berger, S. A. Della Pietra, and V. J. Della Pietra. A maximum entropy approach to natural language processing. *Computational Linguistics*, 22(1):39–71, 1996.

[10] J. Bergstra and Y. Bengio. Random search for hyper-parameter optimization. *JMLR*, 13:281–305, 2012.

[11] Y. Bengio, H. Schwenk, J.-S. Senécal, F. Morin, and J.-L. Gauvain. Neural probabilistic language models. In *Innovations in Machine Learning*, pages 137–186. Springer, 2006.

[12] Y. Bengio. Learning deep architectures for AI. *Foundations and Trends in Machine Learning*, 2(1):1–127, 2009.

[13] Y. Bengio, A. Courville, and P. Vincent. Representation learning: A review and new perspectives. *IEEE TPAMI*, 35(8):1798–1828, 2013.

[14] Y. Bengio, P. Lamblin, D. Popovici, and H. Larochelle. Greedy layer-wise training of deep networks. In *NIPS*, 2007.

[15] Y. Bengio. Deep learning of representations: Looking forward. In *SLSP*, 2013.

[16] Y. Bengio and J. Delalleau. Justifying and generalizing contrastive divergence. *Neural Computation*, 21(6):1601–1621, 2009.

[17] Y. Bengio, É. Thibodeau-Laufer, G. Alain, and J. Yosinski. Deep generative stochastic networks trainable by backprop. In *ICML*, 2014.

[18] D. P. Bertsekas and J. N. Tsitsiklis. *Neuro-Dynamic Programming*. Athena Scientific, 1996.

[19] A. Biem, S. Katagiri, E. McDermott, and B.-H. Juang. An application of discriminative feature extraction to filter-bank-based speech recognition. *IEEE Transactions on Speech and Audio Processing*, 9(2):96–110, 2001.

[20] Y. Burda, R. Grosse, and R. Salakhutdinov. Importance weighted autoencoders. *arXiv preprint arXiv:1509.00519*, 2015.

[21] A. Conneau, G. Kruszewski, G. Lample, L. Barrault, and M. Baroni. What you can cram into a single $&!#\* vector: Probing sentence embeddings for linguistic properties. In *ACL*, 2018.

[22] R. Caruana. Multitask learning. *Machine Learning*, 28(1):41–75, 1997.

[23] O. Chapelle, B. Schölkopf, and A. Zien, editors. *Semi-Supervised Learning*. MIT Press, 2006.

[24] S. Chen and J. W. Bills. Orthogonal matching pursuits for compressing sensing. *IEEE Transactions on Signal Processing*, 57(10):3855–3864, 2009.

[25] H. Chen, S. M. Lundberg, and S.-I. Lee. Explaining models by propagating Shapley values of local components. *arXiv preprint arXiv:1905.02069*, 2019.

[26] K. Cho, B. van Merriënboer, C. Gulcehre, D. Bahdanau, F. Bougares, H. Schwenk, and Y. Bengio. Learning phrase representations using RNN encoder-decoder for statistical machine translation. In *EMNLP*, 2014.

[27] E. J. Candès and M. B. Wakin. An introduction to compressive sampling. *IEEE Signal Processing Magazine*, 25(2):21–30, 2008.

[28] D.-A. Clevert, T. Unterthiner, and S. Hochreiter. Fast and accurate deep network learning by exponential linear units (ELUs). In *ICLR*, 2016.

[29] N. Dalal and B. Triggs. Histograms of oriented gradients for human detection. In *CVPR*, 2005.

[30] S. Deerwester, S. T. Dumais, G. W. Furnas, T. K. Landauer, and R. Harshman. Indexing by latent semantic analysis. *JASIS*, 41(6):391–407, 1990.

[31] G. E. Dahl, D. Yu, L. Deng, and A. Acero. Context-dependent pre-trained deep neural networks for large-vocabulary speech recognition. *IEEE Transactions on Audio, Speech, and Language Processing*, 20(1):30–42, 2012.

[32] G. E. Dahl, T. N. Sainath, and G. E. Hinton. Improving deep neural networks for LVCSR using rectified linear units and dropout. In *ICASSP*, 2013.

[33] P. Dayan, G. E. Hinton, R. M. Neal, and R. S. Zemel. The Helmholtz machine. *Neural Computation*, 7(5):889–904, 1995.

[34] H. Daumé III. Frustratingly easy domain adaptation. In *ACL*, 2007.

[35] S. C. Douglas. ICA, demixing and the cocktail party problem. In *ICASSP*, 2001.

[36] D. Erhan, Y. Bengio, A. Courville, P.-A. Manzagol, P. Vincent, and S. Bengio. Why does unsupervised pre-training help deep learning? *JMLR*, 11:625–660, 2010.

[37] Y. C. Eldar and G. Kutyniok, editors. *Compressed Sensing: Theory and Applications*. Cambridge University Press, 2012.

[38] R. Girshick, J. Donahue, T. Darrell, and J. Malik. Rich feature hierarchies for accurate object detection and semantic segmentation. In *CVPR*, 2014.

[39] I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio. Generative adversarial nets. In *NIPS*, 2014.

[40] X. Glorot and Y. Bengio. Understanding the difficulty of training deep feedforward neural networks. In *AISTATS*, 2010.

[41] X. Glorot, A. Bordes, and Y. Bengio. Deep sparse rectifier neural networks. In *AISTATS*, 2011.

[42] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In *CVPR*, 2016.

[43] G. Hinton and S. Roweis. Stochastic neighbor embedding. In *NIPS*, 2002.

[44] I. Higgins, L. Matthey, A. Pal, C. Burgess, X. Glorot, M. Botvinick, S. Mohamed, and A. Lerchner. $\beta$-VAE: Learning basic visual concepts with a constrained variational framework. In *ICLR*, 2017.

[45] G. E. Hinton. Training products of experts by minimizing contrastive divergence. *Neural Computation*, 14(8):1771–1800, 2002.

[46] G. E. Hinton, S. Osindero, and Y.-W. Teh. A fast learning algorithm for deep belief nets. *Neural Computation*, 18(7):1527–1554, 2006.

[47] G. E. Hinton and R. R. Salakhutdinov. Reducing the dimensionality of data with neural networks. *Science*, 313(5786):504–507, 2006.

[48] G. E. Hinton. Distributed representations. Technical Report CMU-CS-84-157, Carnegie Mellon University, 1984.

[49] S. Hochreiter and J. Schmidhuber. Long short-term memory. *Neural Computation*, 9(8):1735–1780, 1997.

[50] X. He and P. Niyogi. Locality preserving projections. In *NIPS*, 2003.

[51] R. Hadsell, S. Chopra, and Y. LeCun. Dimensionality reduction by learning an invariant mapping. In *CVPR*, 2006.

[52] G. E. Hinton. A practical guide to training restricted Boltzmann machines. Technical Report UTML TR 2010-003, University of Toronto, 2010.

[53] G. E. Hinton, L. Deng, D. Yu, G. E. Dahl, A. Mohamed, N. Jaitly, A. Senior, V. Vanhoucke, P. Nguyen, T. N. Sainath, and B. Kingsbury. Deep neural networks for acoustic modeling in speech recognition: The shared views of four research groups. *IEEE Signal Processing Magazine*, 29(6):82–97, 2012.

[54] K. Jarrett, K. Kavukcuoglu, M. Ranzato, and Y. LeCun. What is the best multi-stage architecture for object recognition? In *ICCV*, 2009.

[55] S. Ioffe and C. Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In *ICML*, 2015.

[56] I. T. Jolliffe. *Principal Component Analysis*. Springer-Verlag, 1986.

[57] A. Hyvärinen, J. Karhunen, and E. Oja. *Independent Component Analysis*. John Wiley & Sons, 2001.

[58] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11):2278–2324, 1998.

[59] K. Kavukcuoglu, M. Ranzato, and Y. LeCun. Fast inference in sparse coding algorithms with applications to object recognition. Technical Report CBLL-TR-2008-12-01, NYU, 2008.

[60] H. Larochelle, D. Erhan, A. Courville, J. Bergstra, and Y. Bengio. An empirical evaluation of deep architectures on problems with many factors of variation. In *ICML*, 2007.

[61] H. Larochelle and S. Bengio. Classification using discriminative restricted Boltzmann machines. In *ICML*, 2008.

[62] D. P. Kingma and M. Welling. Auto-encoding variational Bayes. In *ICLR*, 2014.

[63] D. P. Kingma and J. Ba. Adam: A method for stochastic optimization. In *ICLR*, 2015.

[64] T. Kohonen. Self-organized formation of topologically correct feature maps. *Biological Cybernetics*, 43(1):59–69, 1982.

[65] K. Kavukcuoglu, P. Sermanet, Y.-L. Boureau, K. Gregor, M. Mathieu, and Y. LeCun. Learning convolutional feature hierarchies for visual recognition. In *NIPS*, 2010.

[66] K. Kavukcuoglu, M. Ranzato, R. Fergus, and Y. LeCun. Learning invariant features through topographic filter maps. In *CVPR*, 2009.

[67] A. Krizhevsky, I. Sutskever, and G. E. Hinton. ImageNet classification with deep convolutional neural networks. In *NIPS*, 2012.

[68] A. Hyvärinen. Estimation of non-normalized statistical models using score matching. *JMLR*, 6:695–709, 2005.

[69] H. Lee, C. Ekanadham, and A. Y. Ng. Sparse deep belief net model for visual area V2. In *NIPS*, 2008.

[70] H. Lee, R. Grosse, R. Ranganath, and A. Y. Ng. Convolutional deep belief networks for scalable unsupervised learning of hierarchical representations. In *ICML*, 2009.

[71] H. Larochelle, M. Mandel, R. Pascanu, and Y. Bengio. Learning algorithms for the classification restricted Boltzmann machine. *JMLR*, 13:643–669, 2012.

[72] Y. LeCun. Une procédure d'apprentissage pour réseau à seuil assymétrique. In *Cognitiva*, 1985.

[73] Y. LeCun, B. Boser, J. S. Denker, D. Henderson, R. E. Howard, W. Hubbard, and L. D. Jackel. Backpropagation applied to handwritten zip code recognition. *Neural Computation*, 1(4):541–551, 1989.

[74] H. Lee, R. Grosse, R. Ranganath, and A. Y. Ng. Unsupervised learning of hierarchical representations with convolutional deep belief networks. *Communications of the ACM*, 54(10):95–103, 2011.

[75] Y. LeCun and F. Huang. Loss functions for discriminative training of energy-based models. In *AISTATS*, 2005.

[76] N. Le Roux and Y. Bengio. Representational power of restricted Boltzmann machines and deep belief networks. *Neural Computation*, 20(6):1631–1649, 2008.

[77] A. L. Maas, A. Y. Hannun, and A. Y. Ng. Rectifier nonlinearities improve neural network acoustic models. In *ICML Workshop on Deep Learning for Audio, Speech, and Language Processing*, 2013.

[78] J. Martens. Deep learning via Hessian-free optimization. In *ICML*, 2010.

[79] J. Martens and I. Sutskever. Learning recurrent neural networks with Hessian-free optimization. In *ICML*, 2011.

[80] V. Nair and G. E. Hinton. Rectified linear units improve restricted Boltzmann machines. In *ICML*, 2010.

[81] A. L. Maas, A. Y. Hannun, and A. Y. Ng. Rectifier nonlinearities improve neural network acoustic models. In *ICML Workshop on Deep Learning for Audio, Speech, and Language Processing*, 2013.

[82] S. G. Mallat and Z. Zhang. Matching pursuits with time-frequency dictionaries. *IEEE Transactions on Signal Processing*, 41(12):3397–3415, 1993.

[83] J. L. McClelland and J. L. Elman. The TRACE model of speech perception. *Cognitive Psychology*, 18(1):1–86, 1986.

[84] J. Long, E. Shelhamer, and T. Darrell. Fully convolutional networks for semantic segmentation. In *CVPR*, 2015.

[85] A. Mnih and G. E. Hinton. Three new graphical models for statistical language modelling. In *ICML*, 2007.

[86] A. Mnih and K. Kavukcuoglu. Learning word embeddings efficiently with noise-contrastive estimation. In *NIPS*, 2013.

[87] H. B. Nielsen. UFLDL tutorial. Technical report, Stanford University, 2010.

[88] V. Nair and G. E. Hinton. Rectified linear units improve restricted Boltzmann machines. In *ICML*, 2010.

[89] J. Ngiam, A. Khosla, M. Kim, J. Nam, H. Lee, and A. Y. Ng. Multimodal deep learning. In *ICML*, 2011.

[90] T. Mikolov, M. Karafiát, L. Burget, J. Černocký, and S. Khudanpur. Recurrent neural network based language model. In *INTERSPEECH*, 2010.

[91] B. A. Olshausen and D. J. Field. Emergence of simple-cell receptive field properties by learning a sparse code for natural images. *Nature*, 381(6583):607–609, 1996.

[92] B. A. Olshausen and D. J. Field. Sparse coding with an overcomplete basis set: A strategy employed by V1? *Vision Research*, 37(23):3311–3325, 1997.

[93] S. J. Nowlan and G. E. Hinton. Simplifying neural networks by soft weight-sharing. *Neural Computation*, 4(4):473–493, 1992.

[94] J. Pearl. *Probabilistic Reasoning in Intelligent Systems: Networks of Plausible Inference*. Morgan Kaufmann, 1988.

[95] F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, and É. Duchesnay. Scikit-learn: Machine learning in Python. *JMLR*, 12:2825–2830, 2011.

[96] R. Raina, A. Battle, H. Lee, B. Packer, and A. Y. Ng. Self-taught learning: Transfer learning from unlabeled data. In *ICML*, 2007.

[97] S. Rifai, P. Vincent, X. Muller, X. Glorot, and Y. Bengio. Contractive auto-encoders: Explicit invariance during feature extraction. In *ICML*, 2011.

[98] R. Raina, A. Madhavan, and A. Y. Ng. Large-scale deep unsupervised learning using graphics processors. In *ICML*, 2009.

[99] M. Ranzato, Y.-L. Boureau, and Y. LeCun. Sparse feature learning for deep belief networks. In *NIPS*, 2007.

[100] M. Ranzato, C. Poultney, S. Chopra, and Y. LeCun. Efficient learning of sparse representations with an energy-based model. In *NIPS*, 2006.

[101] M. Ranzato and G. E. Hinton. Modeling pixel means and covariances using factored third-order Boltzmann machines. In *CVPR*, 2010.

[102] M. Ranzato, A. Krizhevsky, and G. E. Hinton. Factored 3-way restricted Boltzmann machines for modeling natural images. In *AISTATS*, 2010.

[103] D. E. Rumelhart, G. E. Hinton, and R. J. Williams. Learning representations by back-propagating errors. *Nature*, 323(6088):533–536, 1986.

[104] S. Roweis and L. K. Saul. Nonlinear dimensionality reduction by locally linear embedding. *Science*, 290(5500):2323–2326, 2000.

[105] R. Pascanu, G. Montufar, and Y. Bengio. On the number of response regions of deep feed forward networks with piece-wise linear activations. *arXiv preprint arXiv:1312.6098*, 2013.

[106] R. Salakhutdinov and G. E. Hinton. Deep Boltzmann machines. In *AISTATS*, 2009.

[107] L. K. Saul and S. T. Roweis. Think globally, fit locally: Unsupervised learning of low dimensional manifolds. *JMLR*, 4:119–155, 2003.

[108] P. Smolensky. Information processing in dynamical systems: Foundations of harmony theory. In D. E. Rumelhart and J. L. McClelland, editors, *Parallel Distributed Processing*, volume 1, pages 194–281. MIT Press, 1986.

[109] S. Rifai, G. Mesnil, P. Vincent, X. Muller, Y. Bengio, Y. Dauphin, and X. Glorot. Higher order contractive auto-encoder. In *ECML/PKDD*, 2011.

[110] S. Rifai, P. Vincent, X. Muller, X. Glorot, and Y. Bengio. Contractive auto-encoders: Explicit invariance during feature extraction. In *ICML*, 2011.

[111] P. Simard, D. Steinkraus, and J. C. Platt. Best practices for convolutional neural networks applied to visual document analysis. In *ICDAR*, 2003.

[112] R. M. Neal. Connectionist learning of belief networks. *Artificial Intelligence*, 56(1):71–113, 1992.

[113] F. Schroff, D. Kalenichenko, and J. Philbin. FaceNet: A unified embedding for face recognition and clustering. In *CVPR*, 2015.

[114] T. Sejnowski. Higher-order Boltzmann machines. In *AIP Conference Proceedings*, volume 151, pages 398–403, 1986.

[115] B. Kim, M. Wattenberg, J. Gilmer, C. Viegas, R. Sayres, J. Wexler, D. Wilson, and F. Viegas. Interpretability beyond feature attribution: Quantitative testing with concept activation vectors (TCAV). In *ICML*, 2018.

[116] J. Schmidhuber. Learning complex, extended sequences using the principle of history compression. *Neural Computation*, 4(2):234–242, 1992.

[117] K. Simonyan and A. Zisserman. Very deep convolutional networks for large-scale image recognition. In *ICLR*, 2015.

[118] T. Serre, L. Wolf, S. Bileschi, M. Riesenhuber, and T. Poggio. Robust object recognition with cortex-like mechanisms. *IEEE TPAMI*, 29(3):411–426, 2007.

[119] M. E. Tipping and C. M. Bishop. Probabilistic principal component analysis. *JRSS-B*, 61(3):611–622, 1999.

[120] N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov. Dropout: A simple way to prevent neural networks from overfitting. *JMLR*, 15(1):1929–1958, 2014.

[121] J. Tenenbaum, V. de Silva, and J. C. Langford. A global geometric framework for nonlinear dimensionality reduction. *Science*, 290(5500):2319–2323, 2000.

[122] J. Weston, F. Ratle, and R. Collobert. Deep learning via semi-supervised embedding. In *ICML*, 2008.

[123] D. Erhan, P.-A. Manzagol, Y. Bengio, S. Bengio, and P. Vincent. The difficulty of training deep architectures and the effect of unsupervised pre-training. In *AISTATS*, 2009.

[124] C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov, D. Erhan, V. Vanhoucke, and A. Rabinovich. Going deeper with convolutions. In *CVPR*, 2015.

[125] J. Sivic and A. Zisserman. Video Google: A text retrieval approach to object matching in videos. In *ICCV*, 2003.

[126] Y. Taigman, M. Yang, M. Ranzato, and L. Wolf. DeepFace: Closing the gap to human-level performance in face verification. In *CVPR*, 2014.

[127] P. Vincent, H. Larochelle, Y. Bengio, and P.-A. Manzagol. Extracting and composing robust features with denoising autoencoders. In *ICML*, 2008.

[128] P. Vincent, H. Larochelle, I. Lajoie, Y. Bengio, and P.-A. Manzagol. Stacked denoising autoencoders: Learning useful representations in a deep network with a local denoising criterion. *JMLR*, 11:3371–3408, 2010.

[129] P. Vincent. A connection between score matching and denoising autoencoders. *Neural Computation*, 23(7):1661–1674, 2011.

[130] L. van der Maaten and G. Hinton. Visualizing data using t-SNE. *JMLR*, 9:2579–2605, 2008.

[131] M. A. Ranzato and M. Szummer. Semi-supervised learning of compact document representations with deep networks. In *ICML*, 2008.

[132] P. Vincent, H. Larochelle, I. Lajoie, Y. Bengio, and P.-A. Manzagol. Stacked denoising autoencoders: Learning useful representations in a deep network with a local denoising criterion. *JMLR*, 11:3371–3408, 2010.

[133] J. Yang, K. Yu, Y. Gong, and T. Huang. Linear spatial pyramid matching using sparse coding for image classification. In *CVPR*, 2009.

[134] K. Yu, T. Zhang, and Y. Gong. Nonlinear learning using local coordinate coding. In *NIPS*, 2009.

[135] K. Yu and T. Zhang. Improved local coordinate coding using local tangents. In *ICML*, 2010.

[136] M. Zeiler, D. Krishnan, G. Taylor, and R. Fergus. Deconvolutional networks. In *CVPR*, 2010.

[137] M. D. Zeiler and R. Fergus. Visualizing and understanding convolutional networks. In *ECCV*, 2014.

[138] W. Y. Zou, A. Y. Ng, S. Zhu, and K. Yu. Deep learning of invariant features via simulated fixations in video. In *NIPS*, 2012.

[139] H. Lee, P. Pham, Y. Largman, and A. Y. Ng. Unsupervised feature learning for audio classification using convolutional deep belief networks. In *NIPS*, 2009.

[140] A. Sharif Razavian, H. Azizpour, J. Sullivan, and S. Carlsson. CNN features off-the-shelf: An astounding baseline for recognition. In *CVPR Workshop*, 2014.

[141] K. Gregor and Y. LeCun. Learning fast approximations of sparse coding. In *ICML*, 2010.

[142] A. Hyvärinen and P. Hoyer. A two-layer sparse coding model learns simple and complex cell receptive fields and topography from natural images. *Vision Research*, 41(18):2413–2423, 2001.

[143] A. Krizhevsky. Learning multiple layers of features from tiny images. Technical report, University of Toronto, 2009.

[144] Y. Bengio, M. Mesnil, Y. Dauphin, and S. Rifai. Better mixing via deep representations. In *ICML*, 2013.

[145] G. Montavon and K.-R. Müller. Deep Boltzmann machines and the centering trick. In *Neural Networks: Tricks of the Trade*, pages 621–637. Springer, 2012.

[146] D. C. Ciresan, U. Meier, L. M. Gambardella, and J. Schmidhuber. Deep, big, simple neural nets for handwritten digit recognition. *Neural Computation*, 22(12):3207–3220, 2010.

[147] Y. LeCun, B. Boser, J. S. Denker, D. Henderson, R. E. Howard, W. Hubbard, and L. D. Jackel. Handwritten digit recognition with a back-propagation network. In *NIPS*, 1990.

[148] D. Erhan, A. Courville, and P. Vincent. Why does unsupervised pre-training help deep learning? In *AISTATS*, 2010.

[149] G. Taylor, G. Hinton, and S. Roweis. Modeling human motion using binary latent variables. In *NIPS*, 2006.

[150] R. Salakhutdinov and G. E. Hinton. An efficient learning procedure for deep Boltzmann machines. *Neural Computation*, 24(8):1967–2006, 2012.

[151] N. Tishby, F. C. Pereira, and W. Bialek. The information bottleneck method. In *Allerton Conference*, 1999.

[152] R. Shwartz-Ziv and N. Tishby. Opening the black box of deep neural networks via information. *arXiv preprint arXiv:1703.00810*, 2017.

[153] A. M. Saxe, P. W. Koh, Z. Chen, M. Bhand, B. Suresh, and A. Y. Ng. On random weights and unsupervised feature learning. In *ICML*, 2011.

[154] D. O. Hebb. *The Organization of Behavior*. Wiley, 1949.

[155] T. J. Sejnowski and C. R. Rosenberg. Parallel networks that learn to pronounce English text. *Complex Systems*, 1(1):145–190, 1987.

[156] P. Smolensky. Information processing in dynamical systems: Foundations of harmony theory. In D. E. Rumelhart and J. L. McClelland, editors, *Parallel Distributed Processing*, volume 1, pages 194–281. MIT Press, 1986.

[157] R. Tibshirani. Regression shrinkage and selection via the lasso. *JRSS-B*, 58(1):267–288, 1996.

[158] J. B. Tenenbaum, V. de Silva, and J. C. Langford. A global geometric framework for nonlinear dimensionality reduction. *Science*, 290(5500):2319–2323, 2000.

[159] G. W. Cottrell. Extracting features from faces using compression networks: Face, identity, emotion, and gender recognition using holons. In *NIPS*, 1990.

[160] Y. Bengio, N. Chapados, O. Delalleau, H. Larochelle, X. Saint-Mleux, C. Hudon, and J. Louradour. Towards biologically plausible deep learning. *arXiv preprint arXiv:1502.04164*, 2015.

[161] M. Gutmann and A. Hyvärinen. Noise-contrastive estimation: A new estimation principle for unnormalized statistical models. In *AISTATS*, 2010.

[162] M. U. Gutmann and A. Hyvärinen. Noise-contrastive estimation of unnormalized statistical models, with applications to natural image statistics. *JMLR*, 14:307–361, 2013.

[163] Y. Dauphin, X. Glorot, and Y. Bengio. Large-scale learning of embeddings with reconstruction sampling. In *ICML*, 2011.

[164] N. Le Roux and Y. Bengio. Deep belief networks are compact universal approximators. *Neural Computation*, 22(8):2192–2207, 2010.

[165] I. Sutskever and G. E. Hinton. Deep narrow sigmoid belief networks are universal approximators. *Neural Computation*, 20(11):2629–2636, 2008.

[166] G. F. Montufar and J. Morton. When does a mixture of products contain a product of mixtures? *SIAM Journal on Discrete Mathematics*, 29(1):321–347, 2015.

[167] G. F. Montufar, R. Pascanu, K. Cho, and Y. Bengio. On the number of linear regions of deep neural networks. In *NIPS*, 2014.

[168] J. M. Hernández-Lobato and R. Adams. Probabilistic backpropagation for scalable learning of Bayesian neural networks. In *ICML*, 2015.

[169] C. Blundell, J. Cornebise, K. Kavukcuoglu, and D. Wierstra. Weight uncertainty in neural networks. In *ICML*, 2015.

[170] D. P. Kingma, T. Salimans, and M. Welling. Variational dropout and the local reparameterization trick. In *NIPS*, 2015.

[171] L. Maaløe, C. K. Sønderby, S. K. Sønderby, and O. Winther. Auxiliary deep generative models. In *ICML*, 2016.

[172] A. Radford, L. Metz, and S. Chintala. Unsupervised representation learning with deep convolutional generative adversarial networks. In *ICLR*, 2016.

[173] T. Salimans, I. Goodfellow, W. Zaremba, V. Cheung, A. Radford, and X. Chen. Improved techniques for training GANs. In *NIPS*, 2016.

[174] T. Karras, T. Aila, S. Laine, and J. Lehtinen. Progressive growing of GANs for improved quality, stability, and variation. In *ICLR*, 2018.

[175] A. van den Oord, S. Dieleman, H. Zen, K. Simonyan, O. Vinyals, A. Graves, N. Kalchbrenner, A. Senior, and K. Kavukcuoglu. WaveNet: A generative model for raw audio. *arXiv preprint arXiv:1609.03499*, 2016.

[176] A. van den Oord, N. Kalchbrenner, L. Espeholt, O. Vinyals, A. Graves, et al. Conditional image generation with PixelCNN decoders. In *NIPS*, 2016.

[177] A. van den Oord, N. Kalchbrenner, and K. Kavukcuoglu. Pixel recurrent neural networks. In *ICML*, 2016.

[178] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin. Attention is all you need. In *NIPS*, 2017.

[179] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. In *NAACL-HLT*, 2019.

[180] A. Radford, K. Narasimhan, T. Salimans, and I. Sutskever. Improving language understanding by generative pre-training. Technical report, OpenAI, 2018.
