# Learning a Similarity Metric Discriminatively, with Application to Face Verification

> Sumit Chopra, Raia Hadsell, Yann LeCun | Courant Institute of Mathematical Sciences, New York University

本文提出了一种从数据中学习相似度度量的方法。该方法可用于识别或验证任务，其中类别数量非常庞大且在训练时未知，并且单个类别的训练样本数量非常少。其核心思想是学习一个将输入模式映射到目标空间的函数，使得目标空间中的 $\ell_2$ 范数近似输入空间中的"语义"距离。该方法应用于人脸验证任务。学习过程最小化一个判别性损失函数，该函数驱使同一个人的人脸对之间的相似度度量变小，而不同人的人脸对之间的相似度度量变大。从原始空间到目标空间的映射是一个卷积网络，其架构设计用于对几何变形具有鲁棒性。该系统在普渡大学/AR人脸数据库上进行了测试，该数据库在姿态、光照、表情、位置以及人工遮挡（如墨镜和遮挡围巾）方面具有非常高的变异性。

Key findings:
- 提出了一种判别性训练的相似度度量方法，使用Siamese架构和对比损失函数
- 在AT&T数据集上取得了接近零的错误率，在更具挑战性的AR/Purdue数据集上取得了19%的错误率（10% FA时）
- 证明了卷积网络结合判别性训练的相似度度量对几何变形和遮挡具有鲁棒性

---

## 摘要

我们提出了一种从数据中训练相似度度量的方法。该方法可用于识别或验证应用，其中类别数量非常庞大且在训练时未知，并且单个类别的训练样本数量非常少。其核心思想是学习一个将输入模式映射到目标空间的函数，使得目标空间中的 $\ell_2$ 范数近似输入空间中的"语义"距离。该方法应用于人脸验证任务。学习过程最小化一个判别性损失函数，该函数驱使同一个人的人脸对之间的相似度度量变小，而不同人的人脸对之间的相似度度量变大。从原始空间到目标空间的映射是一个卷积网络，其架构设计用于对几何变形具有鲁棒性。该系统在普渡大学/AR人脸数据库上进行了测试，该数据库在姿态、光照、表情、位置以及人工遮挡（如墨镜和遮挡围巾）方面具有非常高的变异性。

## 1. 引言

使用判别性方法（如神经网络或支持向量机）的传统分类方法通常要求所有类别事先已知。它们还要求所有类别都有可用的训练样本。此外，这些方法本质上局限于相当少的类别数量（大约100个）。这些方法不适用于类别数量非常大、每个类别的样本数量少、且训练时仅已知部分类别的应用场景。这类应用包括人脸识别人脸验证：类别数量可能成百上千，每个类别只有少量样本。解决这类问题的一种常见方法是基于距离的方法，包括计算待分类或验证的模式与存储的原型库之间的相似度度量。另一种常见方法是在降维空间中使用非判别性（生成式）概率方法，其中某个类别的模型可以在不使用其他类别样本的情况下进行训练。为了将判别性学习技术应用于这类应用，我们必须设计一种能够从可用数据中提取问题信息的方法，而无需关于类别的特定信息。

本文提出的解决方案是从数据中学习相似度度量。该相似度度量随后可用于比较或匹配来自先前未见过的类别的新样本（例如，训练时未见过的人的人脸）。我们提出了一种新型的判别性训练方法，用于训练相似度度量。该方法可应用于类别数量非常庞大和/或训练时并非所有类别的样本都可用的问题。

主要思想是找到一个将输入模式映射到目标空间的函数，使得目标空间中的简单距离（如欧氏距离）近似输入空间中的"语义"距离。

更精确地说，给定一族由 $\mathcal{W}$ 参数化的函数 $G_W(X)$ ，我们寻求找到一个参数 $\mathcal{W}$ 的值，使得相似度度量 $E_W(X_1, X_2) = \|G_W(X_1) - G_W(X_2)\|$ 在 $X_1$ 和 $X_2$ 属于同一类别时较小，在它们属于不同类别时较大。该系统在从训练集中采集的模式对上训练。训练过程中最小化的损失函数在 $X_1$ 和 $X_2$ 来自同一类别时减小 $E_W(X_1, X_2)$ ，在它们属于不同类别时增大 $E_W(X_1, X_2)$ 。除了关于 $\mathcal{W}$ 的可微性外，不对 $G_W(X)$ 的性质做任何假设。由于使用相同的函数 $G$ 和相同的参数 $\mathcal{W}$ 处理两个输入，该相似度度量是对称的。这被称为Siamese架构[4]。

为了用此方法构建人脸验证系统，我们首先训练模型以产生输出向量，使得同一个人的图像对在目标空间中彼此靠近，而不同人的图像对相互远离。该模型随后可作为未见过的（训练时未见过的）新的人脸图像之间的相似度度量使用。

所提方法的一个重要方面是我们在 $G_W(X)$ 的选择上具有完全的自由。特别是，我们将使用旨在提取对输入几何变形具有鲁棒性的表示的架构，例如卷积网络[8]。由此产生的相似度度量将对图像对之间的姿态微小差异具有鲁棒性。

由于目标空间的维度较低，且该空间中的自然距离对输入的不相关变形具有不变性，我们可以从非常少的样本中轻松估计每个新类别的概率模型。

### 1.1. 相关工作

将人脸图像映射到低维目标空间再进行比较的想法有着悠久的历史，从基于PCA的Eigenface方法[16]开始，其中 $G(X)$ 是一个非判别性训练的线性投影，用于最大化方差。基于LDA的Fisherface方法[3]也是线性的，但经过判别性训练以最大化类间方差与类内方差的比率。基于核PCA和核LDA的非线性扩展已被讨论[5]。关于人脸识别子空间方法的综述见[14]。所有这些方法的一个主要缺点是它们对输入图像的几何变换（平移、缩放、旋转）以及其他变异性（面部表情变化、眼镜和遮挡围巾）非常敏感。一些作者描述了局部对一组已知变换具有不变性的相似度度量。一个例子是Tangent Distance方法[19]。另一个已应用于人脸识别的例子是弹性匹配[6]。其他人则提倡基于扭曲的归一化算法，以最大程度地减少由姿态引起的外观变化[10]。所有这些模型的不变性属性都是事先手动设计的。在本文描述的方法中，不变性属性并非来自关于任务的先验知识，而是从数据中学习得到的。当与作为映射函数的卷积网络一起使用时，所提方法可以学习数据中存在的大量不变性。

我们的方法在某种程度上类似于[4]的方法，该方法使用Siamese架构进行签名验证。他们的方法与我们的方法之间的主要区别在于训练过程所最小化的损失函数的性质。我们的损失函数源自能量模型（EBM）的判别性学习框架。

我们的方法与其他降维技术（如多维缩放（MDS）[13]和局部线性嵌入（LLE）[15]）非常不同。MDS根据已知的成对不相似性计算训练集中每个输入对象的目标向量，而不构建映射。相比之下，我们的方法产生一个非线性映射，可以将任何输入向量映射到其对应的低维版本。

## 2. 总体框架

概率模型为被建模变量的每一种可能配置分配一个归一化的概率，而能量模型（EBM）则为这些配置分配一个未归一化的能量[18, 9]。此类系统中的预测通过搜索最小化能量的变量配置来执行。EBM用于需要比较各种配置的能量以做出决策（分类、验证等）的场景。可训练的相似度度量可以看作是将一个能量 $E_W(X_1, X_2)$ 关联到输入模式对上。在最简单的人脸验证设置中，我们只需将 $X_2$ 设为所声称身份的所有可用图像，并将最小能量 $E_W(X_1, X_2)$ 与预定的阈值进行比较。

EBM相对于传统概率模型（尤其是生成模型）的优势在于，无需在输入空间上估计归一化的概率分布。无需归一化使我们免于计算可能难以处理的分区函数。这也使我们在模型架构选择上拥有相当大的自由度[9]。

学习通过寻找最小化适当设计的损失函数（在训练集上评估）的 $\mathcal{W}$ 来执行。乍一看，我们可能认为只需最小化 $E_W(X_1, X_2)$ 在一组来自同一类别的输入对上的平均值就足够了。但这通常会导致灾难性的崩溃：通过简单地使 $G_W(X)$ 成为一个常数函数，能量和损失都可以变为零。因此，我们的损失函数需要一个对比项，以确保不仅来自同一类别的输入对的能量较低，而且来自不同类别的对的能量较大。这个问题不会出现在正确归一化的概率模型中，因为使特定对的概率高会自动使其他对的概率低。

### 2.1. 基于学习相似度度量的人脸验证

人脸验证[12]的任务是接受或拒绝图像中主体所声称的身份。性能使用两个指标评估：错误接受百分比和错误拒绝百分比。一个好的系统应同时最小化这两个指标。

我们的方法是构建一个可训练的系统，将原始人脸图像非线性地映射到低维空间中的点，使得如果图像属于同一个人，这些点之间的距离较小，否则距离较大。学习相似度度量通过训练一个由两个共享相同权重集的相同卷积网络组成的网络来实现——即Siamese架构[4]（见图1）。

### 2.2. EBM的能量函数

我们学习机器的架构如图1所示。 $G_W(X)$ 架构的细节在第3.2节中给出。

![图1. Siamese架构。](Figure 1. Siamese Architecture.)

令 $X_1$ 和 $X_2$ 是展示给学习机器的一对图像。令 $Y$ 是该对的二元标签， $Y=0$ 如果图像 $X_1$ 和 $X_2$ 属于同一个人（一个"genuine pair"）， $Y=1$ 否则（一个"impostor pair"）。令 $\mathcal{W}$ 是学习过程中需学习的共享参数向量， $G_W(X_1)$ 和 $G_W(X_2)$ 是通过映射 $X_1$ 和 $X_2$ 生成的低维空间中的两个点。那么我们的系统可以看作是一个标量"能量函数" $E_W(X_1, X_2)$ ，它衡量 $X_1$ 和 $X_2$ 之间的兼容性。其定义为：

$$
E_W(X_1, X_2) = \|G_W(X_1) - G_W(X_2)\| \qquad (1)
$$

给定来自训练集的一个genuine pair $(X_1, X_2)$ 和一个impostor pair $(X_1, X_2^{\prime})$ ，如果以下条件成立，则机器的行为是理想的：

**条件 1** 存在 $m>0$ ，使得 $E_W(X_1, X_2) + m < E_W(X_1, X_2^{\prime})$ 。

正数 $m$ 可以解释为一个margin（边距）。

为简化符号， $E_W(X_1, X_2)$ 在本文后续部分中记作 $E^G_W$ ， $E_W(X_1, X_2^{\prime})$ 记作 $E^I_W$ 。

### 2.3. 用于训练的对比损失函数

我们假设损失函数仅通过能量间接依赖于输入和参数。我们的损失函数形式为：

$$
\mathcal{L}(\mathcal{W}) = \sum_{i=1}^P L(\mathcal{W}, (Y, X_1, X_2)^{(i)}) = \sum_{i=1}^P [(1-Y)L_G(E_W(X_1, X_2)) + Y L_I(E_W(X_1, X_2))]
$$

其中 $(Y, X_1, X_2)^{(i)}$ 是第 $i$ 个样本，由一对图像和一个标签（genuine或impostor）组成， $L_G$ 是genuine pair的部分损失函数， $L_I$ 是impostor pair的部分损失函数， $P$ 是训练样本数量。 $L_G$ 和 $L_I$ 的设计应使得最小化 $\mathcal{L}$ 会降低genuine pair的能量并提高impostor pair的能量。实现这一点的一个简单方法是使 $L_G$ 单调递增， $L_I$ 单调递减。然而，存在一组更一般的条件，在这些条件下最小化 $\mathcal{L}$ 会使机器接近条件1。我们的论证类似于LeCun等人在[9]中给出的。我们将考虑一个训练集，包含一个能量为 $E^G_W$ 的genuine pair $(X_1, X_2)$ 和一个能量为 $E^I_W$ 的impostor pair $(X_1, X_2^{\prime})$ 。我们定义：

$$
H(E^G_W, E^I_W) = L_G(E^G_W) + L_I(E^I_W) \qquad (2)
$$

作为这两对的总损失函数。我们假设 $H$ 对其两个参数是凸的（注意：我们不假设关于 $\mathcal{W}$ 的凸性）。我们还假设存在一个 $\mathcal{W}$ 使得条件1对单个训练样本成立。对于所有 $E^G_W$ 和 $E^I_W$ 的值，损失函数 $H$ 必须满足以下条件。

**条件 2** $H(E^G_W, E^I_W)$ 的最小值应位于半平面 $E^G_W + m < E^I_W$ 内部。

该条件清楚地保证，当我们关于 $\mathcal{W}$ 最小化 $H$ 时，机器被驱动到解满足条件1的区域。

对于最小值位于无穷远处的 $H$ （见图2），以下条件是充分的：

**条件 3** $H(E^G_W, E^I_W)$ 在margin line $E^G_W + m = E^I_W$ 上的梯度负方向与方向 $[-1, 1]$ 具有正的点积。

为证明这一点，我们陈述并证明以下定理。

**定理 1** 设 $H(E^G_W, E^I_W)$ 关于 $E^G_W$ 和 $E^I_W$ 是凸的，且在无穷远处有最小值。假设存在一个样本点的 $\mathcal{W}$ 使得条件1成立。如果条件3成立，则关于 $\mathcal{W}$ 最小化 $H$ 将导致找到满足条件1的 $\mathcal{W}$ 。

**证明。** 考虑由 $E^G_W$ 和 $E^I_W$ 形成的平面的正象限（见图3）。将两个半平面 $E^G_W + m < E^I_W$ 和 $E^G_W + m \geq E^I_W$ 分别记为 $\mathcal{R}_1$ 和 $\mathcal{R}_2$ 。我们在其定义域内对所有 $\mathcal{W}$ 的值，在 $E^G_W$ 和 $E^I_W$ 上最小化 $H$ 。设 $\mathcal{F}$ 为由 $E^G_W$ 和 $E^I_W$ 形成的平面内的区域，对应于 $\mathcal{W}$ 定义域中的所有值。在最一般的情况下， $\mathcal{F}$ 可以是非凸的，并且可以位于平面中的任何位置。然而，根据我们的假设，存在至少一个 $\mathcal{W}$ 使得条件1成立，我们可以得出结论， $\mathcal{F}$ 的一部分与半平面 $\mathcal{R}_1$ 相交。为了在条件3的光照下证明该定理，我们需要证明在 $\mathcal{F}$ 和 $\mathcal{R}_1$ 的交集中至少存在一个点，使得该点处的损失 $H$ 小于 $\mathcal{F}$ 和 $\mathcal{R}_2$ 的交集中所有点处的损失。

设 $E^{*G}$ 是margin line $E^G_W + m = E^I_W$ 上使 $H$ 最小的点。即，

$$
E^{*G} = \arg\min_{E^G_W} H(E^G_W, E^G_W + m) \qquad (3)
$$

由于margin line上所有点处的 $H$ 的梯度负方向指向半平面 $\mathcal{R}_1$ 内部（条件3），由 $H$ 的凸性我们可以得出结论：

$$
H(E^{*G}, E^{*G} + m) \leq H(E^G_W, E^I_W) \qquad (4)
$$

当 $E^G_W + m = E^I_W$ 时。

现在考虑一个距离 $(E^{*G}, E^{*G} + m)$ 为 $\epsilon$ 且位于半平面 $\mathcal{R}_1$ 内的点。即点

$$
(E^{*G} - \epsilon, E^{*G} + m - \epsilon) \qquad (5)
$$

使用一阶泰勒展开，我们可以将上述写为：

$$
\begin{aligned}
H(E^{*G} - \epsilon, E^{*G} + m - \epsilon) &= H(E^{*G}, E^{*G} + m) - \epsilon \frac{\partial H}{\partial E^G_W} - \epsilon \frac{\partial H}{\partial E^I_W} + O(\epsilon^2) \\
&= H(E^{*G}, E^{*G} + m) - \epsilon \left( \frac{\partial H}{\partial E^G_W} + \frac{\partial H}{\partial E^I_W} \right) + O(\epsilon^2)
\end{aligned} \qquad (6)
$$

由条件3，公式6右侧的第二项为负。因此，对于足够小的 $\epsilon$ ，

$$
H(E^{*G} - \epsilon, E^{*G} + m - \epsilon) < H(E^{*G}, E^{*G} + m) \qquad (7)
$$

因此，在区域 $\mathcal{F}$ 和半平面 $\mathcal{R}_1$ 的交集中存在一个点，在该点损失函数小于 $\mathcal{F}$ 和 $\mathcal{R}_2$ 交集中任何点的损失。因此，结论成立。 $\blacksquare$

注意，当 $L_G$ 是单调递增函数且 $L_I$ 是单调递减函数时，条件3对任何 $H$ 显然成立。

我们对单个样本使用的确切损失函数是：

$$
\begin{aligned}
L(\mathcal{W}, Y, X_1, X_2) &= (1-Y)L_G(E_W) + Y L_I(E_W) \\
&= (1-Y)\frac{2}{Q}(E_W)^2 + (Y) 2Q e^{-\frac{2.77}{Q} E_W}
\end{aligned} \qquad (8)
$$

其中 $E_W = \|G_W(X_1) - G_W(X_2)\|$ 。在我们的架构中， $G_W$ 的分量是有界的，因此 $E_W$ 也是有界的。常数 $Q$ 被设置为 $E_W$ 的上界。

可以清楚地看到，上述损失函数在 $E^G_W$ 中单调递增，在 $E^I_W$ 中单调递减，并且关于 $E^G_W$ 和 $E^I_W$ 都是凸的。因此，根据上述论证，我们得出结论，最小化该损失函数将使机器到达一个 $\mathcal{W}$ ，使其以期望的方式运行。

![图2. 损失函数 $\mathcal{L}$ 关于 $E^G_W$ 和 $E^I_W$ 的三维图。](Figure 2. Graph of the loss function $\mathcal{L}$ against $E^G_W$ and $E^I_W$ in 3D.)

我们对自己的损失函数再做两点说明。首先，解释损失函数中的常数。我们用于最小化损失函数的优化算法基于梯度。选择这些常数是为了确保margin line上损失函数的梯度负方向始终指向区域 $\mathcal{R}$ 内部。这是为了避免我们的算法卡在 $\mathcal{R}$ 边界上的点，且梯度指向 $\mathcal{R}$ 外部的情况。在这种情况下，基于梯度的算法可能将该点识别为损失函数的局部最小值并终止。

其次，我们必须强调，使用平方范数而不是 $\ell_2$ 范数作为能量是不合适的。实际上，如果能量是两个模式输出向量之差的平方范数，那么随着能量趋近于零，能量关于参数的梯度将消失。这将在损失函数中产生危险的平坦区域。这可能导致在两张图像是impostor且对应能量接近零的情况下，机器学习失败。

### 2.4. 卷积网络

为了将原始图像映射到低维空间中的点，从而实现学习到的相似度度量，我们使用两个具有共享参数向量的相同卷积网络[8]（见图1）。卷积网络是可训练的、多层的、非线性系统，可以在像素级别操作，并以集成方式学习低级特征和高级表示。卷积网络经过端到端训练，将像素图像映射到输出。它们的主要优点是可以学习最优的平移不变局部特征检测器，并构建对输入图像几何变形具有鲁棒性的表示。我们使用的网络的具体规格在第3.2节中给出。

## 3. 实验

前一节中描述的模型和架构在3个人脸图像数据库上进行了训练，并在其中2个数据库上进行了测试。我们将详细讨论这些数据库，然后解释训练协议和架构。

### 3.1. 数据集与数据处理

第一轮训练和测试使用了来自AT&T人脸数据库[1]的相对较小的数据集，包含400张图像。该数据集包含40个受试者每人10张图像，光照、面部表情、配饰和头部位置有所变化。每张图像为112x92像素、灰度，并紧密裁剪以仅包含人脸。见图4。

![图4. 上：AT&T数据集的图像。中：AR数据集的图像。下：FERET数据集的图像。每幅图显示一个genuine pair、一个impostor pair以及一个典型受试者的图像。](Figure 4. Top: Images from AT&T dataset. Middle: Images from the AR dataset. Bottom: Images from FERET dataset. Each graphic shows a genuine pair, an impostor pair and images from a typical subject.)

无需对图像进行尺寸或光照归一化预处理，因为既定目标之一就是训练一个能够抵御此类变化的架构。然而，我们确实使用4x4子采样将图像分辨率降低到56x46。

第二轮训练和测试实验结合了两个数据集：普渡大学创建并公开的AR人脸数据库[11]，以及灰度Feret数据库[2]的一个子集。来自这两个数据集的图像对都用于训练，但仅使用AR数据集的图像进行测试。

AR数据集包含136个受试者的3,536张图像，每个受试者26张图像。每个受试者的26张图像集由2组13张图像组成，两组拍摄相隔14天。在每组13张图像中，有4张表情变化的图像、3张光照变化的图像、3张戴墨镜且光照变化的图像，以及3张带遮挡面部的围巾且光照变化的图像。由于单个受试者图像之间外观变化巨大，该数据集极具挑战性。示例见图4。由于人脸在图像中未良好居中，应用了简单的基于相关的居中算法。然后将图像裁剪并缩小到56x46像素。尽管居中足以满足裁剪目的，但许多图像中仍存在头部位置的显著变化。

Feret数据库由美国国家标准与技术研究院分发，包含来自1,209个受试者的14,051张图像。我们仅使用完整数据库的一个子集进行训练。我们的子集包含1122张图像，即187个受试者每人6张图像。唯一的预处理是裁剪和子采样到56x46像素。

**划分** 为了生成由机器在训练期间未见过的受试者图像组成的测试集，我们将数据集分为两个不相交的集合，即SET1和SET2。每个集合中的每张图像与该集合中的其他每张图像配对，以生成最大数量的genuine pair和impostor pair。

对于AT&T数据，SET1包含前35个受试者的350张图像，SET2包含最后5个受试者的50张图像。这样，从SET1生成了总共3500个genuine pair和119000个impostor pair，从SET2生成了500个genuine pair和2000个impostor pair。仅使用SET1生成的图像对进行训练。测试（验证）使用SET2的图像对以及SET1中未使用的图像对进行。

对于AR/Feret数据，SET1包含所有Feret图像以及AR数据库中96个受试者的2,496张图像。SET2包含AR数据库中剩余40个受试者的1,040张图像。取所有2张图像的组合，得到71,628个genuine pair和11,096,376个impostor pair。实际使用的训练集包含140,000个图像对，在genuine和impostor之间均匀分配。测试集从SET2中的1,081,600个对中抽取。因此，只有训练中未见过的受试者被用于测试。

### 3.2. 训练协议与架构

**Siamese架构** Siamese框架包含两个相同的网络和一个代价模块。系统的输入是一对图像和一个标签。图像通过子网络传递，产生两个输出，这些输出被传递到代价模块，该模块产生如第2.3节讨论的标量能量。损失函数将标签与能量结合起来。使用反向传播计算损失函数关于控制两个子网的参数向量的梯度。参数向量使用随机梯度方法更新，使用两个子网贡献的梯度之和。

第一组实验使用小型AT&T数据集，探索了6种不同的子网架构：一个2层全连接神经网络和五个卷积网络，它们在层数、层大小和卷积核大小上有所不同。基于这些实验，第二组实验聚焦于单个卷积网络架构。我们仅在以下各节中描述性能最佳的架构。 $C_i$ 表示卷积层， $S_i$ 表示子采样层， $F_i$ 表示全连接层，其中 $i$ 是层索引。

基本架构为 $C_1 \rightarrow S_2 \rightarrow C_3 \rightarrow S_4 \rightarrow C_5 \rightarrow F_6$ 。

 $C_1$ ：特征图数：15；尺寸：50x40；核大小：7x7。可训练参数：750；连接数：1500000。与输入全连接。

 $S_2$ ：特征图数：15；尺寸：25x20；视野：2x2。可训练参数：30；连接数：37500。

 $C_3$ ：特征图数：45；尺寸：20x15；核大小：6x6。可训练参数：7128；连接数：2139600。部分连接到 $S_2$ 。确切连接模式类似于[8]中使用的模式。其动机是打破对称性，从而推动特征图提取和学习不同的特征。

 $S_4$ ：特征图数：45；尺寸：5x5；视野：4x3。可训练参数：100；连接数：16250。

 $C_5$ ：特征图数：250；尺寸：1x1；核大小：5x5。可训练连接数：312750。全连接到 $S_4$ 。

 $F_6$ ：单元数：50。可训练参数：12550；连接数：12550。

**训练协议** 训练需要两组数据：训练集，用于实际学习系统的权重；验证集，用于在训练期间测试系统的性能。定期使用验证集进行性能评估使我们能够控制过拟合。

网络的训练使用来自SET1的图像对进行。一半的图像对是genuine，一半是impostor，通过随机配对不同受试者的图像产生。验证集由1500个图像对组成，来自SET1中未使用的对，其50% genuine、50% impostor的比例与训练集相同。

网络的性能通过计算被接受的impostor对百分比（FA）和被拒绝的genuine对百分比（FR）来衡量。该计算通过测量一对输出之间差值的范数，然后选择一个阈值来设置FA和FR百分比之间的给定权衡。

![表1. 上：两个数据集的验证集和测试集详情。下：不同错误接受百分比下的错误拒绝百分比。](Table 1. Above: Details of the validation and test sets for the two datasets. Below: False reject percentage for different false accept percentages.)

![图5. 特定示例的卷积网络内部状态。](Figure 5. Internal state of the convolutional network for a particular example.)

## 4. 测试与结果

图5显示了特定测试图像的卷积网络内部状态。第一层提取各种类型的局部梯度特征以及平滑特征。

该系统在一个人脸验证场景中进行了测试。系统被给予一张图像，并被要求确认该图像中主体所声称的身份。我们通过将测试图像与所声称主体的图像的高斯模型进行比较来执行验证。该方法如下讨论。

### 4.1. 验证

测试（验证）在大小为5000的测试集上进行。测试集包含500个genuine pair和4500个impostor pair。对于AT&T实验，测试图像来自训练中未见的5个受试者。对于AR/Feret实验，测试图像来自更具挑战性的AR数据库中40个未见过的受试者。

Siamese网络的一个子网的输出是受试者输入图像的特征向量。我们假设每个受试者图像的特征向量形成多元正态密度。通过使用每个受试者的前五张图像生成的特征向量计算均值特征向量和方差-协方差矩阵，为每个受试者构建模型。测试图像是genuine的似然度 $P(X|\Omega_g)$ 通过在所关注受试者的模型上评估测试图像的正态密度得到。测试图像是impostor的似然度 $P(X|\Omega_i)$ 被假设为常数，其值通过计算所关注受试者的所有impostor图像的平均 $P(X|\Omega_g)$ 值来估计。给定图像是genuine的概率由下式给出：

$$
P(\Omega_g|X) = \frac{P(X|\Omega_g)}{P(X|\Omega_g) + P(X|\Omega_i)}
$$

对于所有可能的阈值概率值，绘制了错误拒绝图像和错误接受图像的百分比值。最优阈值概率是将测试集划分为genuine pair和impostor pair并最小化FA和FR比率的值。

从AT&T数据库和AR/Purdue数据库测试得到的验证率惊人地不同（见表1及图6和图7），这突显了两个数据库在难度上的差异。AT&T数据集相对较小，我们的系统仅需5000个训练样本即可在测试集上取得非常高的性能。AR/Purdue数据集非常大且多样化，在表情、光照和额外遮挡方面存在巨大变化。我们的较高错误率反映了这种难度水平。

![图6. AT&T数据集：错误拒绝百分比 vs. 错误接受百分比。](Figure 6. AT&T dataset: percent false reject vs. false accept.)

![图7. AR/Purdue数据集：错误拒绝百分比 vs. 错误接受百分比。](Figure 7. AR/Purdue dataset: percent false reject vs. false accept.)

## 5. 结论与展望

我们提出了一种用于学习复杂相似度度量的通用判别性方法。该方法最适合于类别数量非常庞大和/或训练时并非所有类别的样本都可用的分类或验证场景。我们通过一个人脸验证应用说明了该方法。

我们提出了一个损失函数，并证明最小化该函数会使系统趋近于期望的行为。我们的损失函数是判别性的，因为它驱使系统做出正确的决策，但不会使其产生概率估计。该方法与概率密度模型不同，我们并不试图在输入空间中为每个类别估计密度。这为我们选择 $G_W(X)$ 提供了额外的灵活性，因为我们无需担心归一化问题。我们选择使用卷积网络架构，该架构对输入的几何变化具有鲁棒性，从而减少了对人脸图像精确配准的需求。

可训练的相似度度量具有超越本文所述内容的众多应用。除此之外，它们可用于构建不变核函数，进而构建支持向量机和其他基于核的模型[17]。

## 参考文献

[1] http://www.uk.research.att.com/facedatabase.html

[2] http://www.itl.nist.gov/iad/humanid/feret/

[3] P. Belhumeur, J. Hespanha, and D. Kriegman. Eigenfaces vs. fisherfaces: Recognition using class specific linear projection. *IEEE Trans. PAMI, Special Issue on Face Recognition*, 19(7), July 1997.

[4] J. Bromley, I. Guyon, Y. LeCun, E. Sackinger, and R. Shah. Signature verification using a siamese time delay neural network. *J. Cowan and G. Tesauro (eds) Advances in Neural Information Processing Systems*, 1993.

[5] M. Hsuan Yang, N. Ahuja, and D. Kriegman. Face recognition using kernel eigenfaces. In *Proc. of the 2000 IEEE International Conference on Image Processing (ICIP)*, 1:37–40, September 2000.

[6] M. Lades, J. C. Vorbruggen, J. Buhmann, J. Lange, C. von der Malsburg, R. P. Wurtz, and W. Konen. Distortion-invariant object recognition in the dynamic link architecture. *IEEE Trans. Computers*, 42(3):300–311, 1993.

[7] S. Lawrence, C. Lee Giles, A. Chung Tsoi, and A. D. Back. Face recognition: A convolutional neural network approach. *IEEE Transactions on Neural Networks, Special Issue on Neural Networks*, 1997.

[8] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11):2278–2324, 1998.

[9] Y. LeCun and F. Jie Huang. Loss functions for discriminative training of energy-based models. *AI-stats*, 2005.

[10] A. M. Martinez. Recognizing imprecisely localized, partially occluded and expression variant faces from a single sample per class. *IEEE Trans. on Pattern Analysis and Machine Intelligence*, 24(6):748–763, 2002.

[11] A. M. Martinez and R. Benavente. The ar face database. *CVC Technical Report*, 24, June 1998.

[12] S. Rizvi, P. J. Phillips, and H. Moon. The feret verification testing protocol for face recognition algorithms. *Technical Report NISTIR 6,281, Nat'l Inst. Standards and Technology*, 1998.

[13] Y. Rubner, L. J. Guibas, and C. Tomasi. The earth mover's distance, multi-dimenional scaling, and color-based image retrieval. *Proc. DARPA Image Understanding Workshop*, pages 661–668, May 1997.

[14] G. Shakhnarovich and B. Moghaddam. *Handbook of Face Recognition, chapter Face Recognition in Subspaces*. Springer-Verlag, 2004.

[15] S. T. Roweis and L. K. Saul. Nonlinear dimensionality reduction by locally linear embedding. *Science*, 290.

[16] M. Turk and A. Pentland. Eigenfaces for recognition. *Journal of Cognitive Neuroscience*, 3(1), 1991.

[17] P. Vincent and Y. Bengio. A neural support vector network architecture with adaptive kernels. In *Proc. of the International Joint Conference on Neural Networks*, 5, July 2000.

[18] Y. W. Teh, M. Welling, S. Osindero, and G. E. Hinton. Energy-based models for sparse overcomplete representations. *Journal of Machine Learning Research*, 4:1235–1260, 2003.

[19] P. Y. Simard, Y. LeCun, J. S. Denker, and B. Victorri. Transformation invariance in pattern recognition – tangent distance and tangent propagation. *International Journal of Imaging Systems and Technology*, 11(3), 2000.
