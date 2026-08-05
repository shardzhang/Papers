# Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift

> Sergey Ioffe, Christian Szegedy | Google Inc.

本文分享了Batch Normalization（批归一化）论文，该论文提出了一种通过在网络架构中嵌入归一化操作来加速深度网络训练的机制。核心内容如下：

- 定义了"内部协变量偏移"（Internal Covariate Shift）现象——训练过程中网络层输入分布随前层参数变化而改变，这迫使网络以更慢的学习率训练并需要精细的参数初始化
- 提出Batch Normalization（BN）层：对每个mini-batch计算均值和方差进行归一化，并引入可学习的缩放参数 $\gamma$ 和平移参数 $\beta$ 以保持网络表达能力
- BN使得可以使用更高的学习率、减少对Dropout的依赖、允许使用饱和非线性函数，并在ImageNet分类上以14倍少的训练步数达到相同精度

关键发现：

- BN显著加速训练：在ImageNet上仅需原模型7%的训练步数即可匹配性能，最终达到top-5错误率4.9%（测试集4.82%），超越人类标注者精度
- BN具有正则化效果：在某些情况下可完全去除Dropout
- BN使训练对参数尺度不敏感：梯度传播不受参数缩放影响，Jacobian矩阵的奇异值趋近于1

---

## 摘要

训练深度神经网络的复杂性在于：由于前层参数的变化，每一层输入的分布在训练过程中不断变化。这迫使训练使用更低的学习率和更精细的参数初始化，从而减慢了训练速度，并且使得训练带有饱和非线性的模型极其困难。我们将这一现象称为内部协变量偏移（Internal Covariate Shift），并通过归一化层输入来解决该问题。我们的方法的核心在于将归一化作为模型架构的一部分，并对每个训练mini-batch执行归一化。Batch Normalization允许我们使用高得多的学习率，且对初始化不再那么敏感。它还起到了正则化的作用，在某些情况下可以完全消除对Dropout的需求。将Batch Normalization应用于最先进的图像分类模型，可以在仅使用14倍更少的训练步数下达到相同的精度，并以显著优势超越原始模型。通过使用批归一化网络的集成，我们改进了ImageNet分类上已发表的最佳结果：达到4.9%的top-5验证错误率（以及4.8%的测试错误率），超过了人类标注者的精度。

---

## 1 引言

深度学习在视觉、语音和许多其他领域极大地推动了技术发展。随机梯度下降（SGD）已被证明是训练深度网络的有效方法，动量[20]和Adagrad[4]等SGD变体也被用于实现最先进的性能。SGD优化网络的参数 $\Theta$ 以最小化损失：

$$
\Theta = \arg\min_{\Theta} \frac{1}{N} \sum_{i=1}^{N} \ell(\mathbf{x}_i, \Theta)
$$

其中 $\mathbf{x}_{1\ldots N}$ 是训练数据集。使用SGD时，训练逐步进行，每一步我们考虑一个大小为 $m$ 的 mini-batch $\mathbf{x}_{1\ldots m}$ 。Mini-batch 用于通过计算 $\frac{1}{m} \frac{\partial \ell(\mathbf{x}_i, \Theta)}{\partial \Theta}$ 来近似损失函数关于参数的梯度。使用 mini-batch 而非每次一个样本在多个方面都有帮助。首先，mini-batch 上的损失梯度是对训练集上梯度的估计，其质量随着 batch 大小的增加而提高。其次，由于现代计算平台提供的并行性，一个 batch 上的计算比 $m$ 个单独样本的计算高效得多。

尽管随机梯度简单有效，但它需要仔细调整模型的超参数，特别是优化中使用的学习率以及模型参数的初始值。训练的复杂性还在于：每一层的输入受到所有前面层参数的影响——因此网络参数上的微小变化会随着网络的加深而被放大。

层输入分布的变化带来了问题，因为这些层需要不断地适应新的分布。当一个学习系统的输入分布发生变化时，我们说它经历了协变量偏移（covariate shift）[18]。这通常通过领域自适应（domain adaptation）[8]来处理。然而，协变量偏移的概念可以扩展到整个学习系统之外，应用于其组成部分，例如子网络或层。考虑一个计算 $\ell = F_2(F_1(\mathbf{u}, \Theta_1), \Theta_2)$ 的网络，其中 $F_1$ 和 $F_2$ 是任意变换，参数 $\Theta_1, \Theta_2$ 需要被学习以最小化损失 $\ell$ 。学习 $\Theta_2$ 可以看作是将输入 $\mathbf{x} = F_1(\mathbf{u}, \Theta_1)$ 送入子网络 $\ell = F_2(\mathbf{x}, \Theta_2)$ 。例如，梯度下降步骤

$$
\Theta_2 \leftarrow \Theta_2 - \frac{\alpha}{m} \sum_{i=1}^{m} \frac{\partial F_2(\mathbf{x}_i, \Theta_2)}{\partial \Theta_2}
$$

（对于 batch 大小 $m$ 和学习率 $\alpha$ ）与以 $\mathbf{x}$ 为输入的独立网络 $F_2$ 的步骤完全等价。因此，使训练更高效的输入分布特性——例如训练数据和测试数据具有相同的分布——也适用于子网络的训练。因此，让 $\mathbf{x}$ 的分布随时间保持固定是有利的。这样， $\Theta_2$ 就不必为了补偿 $\mathbf{x}$ 分布的变化而重新调整。

子网络输入分布的固定也会对子网络之外的层产生积极的影响。考虑一个带有 sigmoid 激活函数的层 $z = g(W\mathbf{u} + b)$ ，其中 $\mathbf{u}$ 是层输入，权重矩阵 $W$ 和偏置向量 $b$ 是需要学习的层参数，而 $g(x) = \frac{1}{1+\exp(-x)}$ 。随着 $|x|$ 增大， $g'(x)$ 趋近于零。这意味着对于 $\mathbf{x} = W\mathbf{u} + b$ 的所有维度——除了那些绝对值很小的维度——向下传播到 $\mathbf{u}$ 的梯度将消失，模型将训练缓慢。然而，由于 $\mathbf{x}$ 受 $W$ 、 $b$ 以及所有下面层参数的影响，训练过程中这些参数的变化很可能将 $\mathbf{x}$ 的许多维度推入非线性的饱和区域，从而减慢收敛速度。这种效应随着网络深度的增加而加剧。在实践中，饱和问题及由此产生的梯度消失通常通过使用修正线性单元（ReLU）[12]、精心的初始化[1, 17]和小学习率来应对。然而，如果我们能确保非线性输入的分布在网络训练过程中保持更稳定，那么优化器就不太可能陷入饱和区域，训练也会加速。

我们将训练过程中深度网络内部节点分布的变化称为**内部协变量偏移**（Internal Covariate Shift）。消除它为更快的训练提供了可能。我们提出了一种新机制，称为**Batch Normalization**，它朝着减少内部协变量偏移迈出了一步，从而显著加速了深度神经网络的训练。它通过一个归一化步骤固定层输入的均值和方差来实现这一点。Batch Normalization 还对流经网络的梯度流动产生有益影响，因为它减少了梯度对参数尺度或其初始值的依赖。这使我们能够使用高得多的学习率而没有发散的风险。此外，批归一化对模型有正则化作用，减少了对 Dropout[19] 的需求。最后，Batch Normalization 通过防止网络陷入饱和模式，使得使用饱和非线性函数成为可能。

在第4.2节中，我们将 Batch Normalization 应用于性能最佳的 ImageNet 分类网络，并展示我们仅需7%的训练步数即可匹配其性能，并且可以进一步以显著的优势超越其精度。通过使用集成这些经 Batch Normalization 训练的网络，我们在 ImageNet 分类上实现了 top-5 错误率，改进了已知的最佳结果。

## 2 迈向减少内部协变量偏移

我们将内部协变量偏移定义为训练过程中网络参数变化导致的网络激活值分布的变化。为了改善训练，我们寻求减少内部协变量偏移。通过固定层输入 $\mathbf{x}$ 的分布随着训练进程的推进，我们期望提高训练速度。早有研究[10, 22]表明，如果网络的输入被白化——即通过线性变换使其具有零均值和单位方差，并去除相关性——网络训练会收敛得更快。由于每一层观察其下层产生的输入，对每一层的输入实现同样的白化将是有利的。通过白化每一层的输入，我们将朝着实现输入分布的固定迈进一步，从而消除内部协变量偏移的不良影响。

我们可以考虑在每个训练步骤或每隔一定间隔对激活值进行白化，既可以直接修改网络，也可以通过改变优化算法的参数使其依赖于网络激活值[23, 15, 14, 3]。然而，如果这些修改穿插在优化步骤之间，那么梯度下降步骤可能会尝试以一种需要更新归一化的方式来更新参数，这降低了梯度步骤的效果。例如，考虑一个输入为 $\mathbf{u}$ 的层，它加上学习到的偏置 $b$ ，然后通过减去在训练数据上计算的激活值均值来归一化结果： $\hat{\mathbf{x}} = \mathbf{x} - \mathbb{E}[\mathbf{x}]$ ，其中 $\mathbf{x} = \mathbf{u} + b$ ， $\mathcal{X} = \{\mathbf{x}_{1\ldots N}\}$ 是 $\mathbf{x}$ 在训练集上的取值集合，且 $\mathbb{E}[\mathbf{x}] = \frac{1}{N} \sum_{i=1}^{N} \mathbf{x}_i$ 。如果梯度下降步骤忽略了 $\mathbb{E}[\mathbf{x}]$ 对 $b$ 的依赖，那么它将更新 $b \leftarrow b + \Delta b$ ，其中 $\Delta b \propto -\partial \ell / \partial \hat{\mathbf{x}}$ 。那么 $\mathbf{u} + (b + \Delta b) - \mathbb{E}[\mathbf{u} + (b + \Delta b)] = \mathbf{u} + b - \mathbb{E}[\mathbf{u} + b]$ 。因此，对 $b$ 的更新和随后的归一化变化相结合，导致层输出没有变化，因此损失也没有变化。随着训练的继续， $b$ 将无限增长而损失保持不变。如果归一化不仅进行中心化还进行缩放，这个问题可能变得更糟。我们在初步实验中经验性地观察到了这一点：当标准化参数在梯度下降步骤之外计算时，模型会发散。

上述方法的问题在于梯度下降优化没有考虑归一化的存在。为了解决这个问题，我们希望确保对于任何参数值，网络总是产生具有期望分布的激活值。这样做将使得损失关于模型参数的梯度能够考虑归一化及其对模型参数 $\Theta$ 的依赖。再次令 $\mathbf{x}$ 为层输入（视为向量）， $\mathcal{X}$ 为这些输入在训练数据集上的集合。归一化可以写成一个变换

$$
\hat{\mathbf{x}} = \text{Norm}(\mathbf{x}, \mathcal{X})
$$

它不仅依赖于给定的训练样本 $\mathbf{x}$ ，还依赖于所有样本 $\mathcal{X}$ ——如果 $\mathbf{x}$ 是由另一层生成的，则每个 $\mathcal{X}$ 都依赖于 $\Theta$ 。对于反向传播，我们需要计算 Jacobian $\frac{\partial \text{Norm}(\mathbf{x}, \mathcal{X})}{\partial \mathbf{x}}$ 和 $\frac{\partial \text{Norm}(\mathbf{x}, \mathcal{X})}{\partial \mathcal{X}}$ ；忽略后一项将导致上述发散。在这个框架内，白化层输入是昂贵的，因为它需要计算协方差矩阵 $\text{Cov}[\mathbf{x}] = \mathbb{E}_{\mathbf{x} \in \mathcal{X}}[\mathbf{x}\mathbf{x}^T] - \mathbb{E}[\mathbf{x}]\mathbb{E}[\mathbf{x}]^T$ 及其逆平方根，以产生白化后的激活值 $\text{Cov}[\mathbf{x}]^{-1/2}(\mathbf{x} - \mathbb{E}[\mathbf{x}])$ ，以及反向传播时这些变换的导数。这促使我们寻找一种替代方案，以可微分的方式执行输入归一化，且不需要在每次参数更新后分析整个训练集。

之前的一些方法（例如[11]）使用基于单个训练样本计算的统计量，或者在图像网络的情况下，使用给定位置的不同特征图上的统计量。然而，这丢弃了激活值的绝对尺度，从而改变了网络的表示能力。我们希望通过相对于整个训练数据的统计量来归一化训练样本中的激活值，从而保留网络中的信息。

## 3 通过Mini-Batch统计量进行归一化

由于对每一层输入进行完全白化代价高昂且并非处处可微，我们做了两个必要的简化。第一，我们不联合白化层输入和输出的特征，而是独立地归一化每个标量特征，使其均值为0、方差为1。对于一个 $d$ 维输入 $\mathbf{x} = (x^{(1)} \ldots x^{(d)})$ 的层，我们将归一化每个维度：

$$
\hat{x}^{(k)} = \frac{x^{(k)} - \mathbb{E}[x^{(k)}]}{\sqrt{\text{Var}[x^{(k)}]}}
$$

其中期望和方差在整个训练数据集上计算。如[10]所示，即使特征之间没有去相关，这种归一化也能加速收敛。

注意，仅仅归一化层的每个输入可能会改变该层所能表示的内容。例如，归一化 sigmoid 的输入会将其约束到非线性的线性区域。为了解决这个问题，我们确保插入到网络中的变换能够表示恒等变换。为此，我们为每个激活值 $x^{(k)}$ 引入一对参数 $\gamma^{(k)}$ 和 $\beta^{(k)}$ ，用于缩放和平移归一化后的值：

$$
y^{(k)} = \gamma^{(k)} \hat{x}^{(k)} + \beta^{(k)}
$$

这些参数与原始模型参数一起学习，恢复了网络的表达能力。实际上，通过设置 $\gamma^{(k)} = \sqrt{\text{Var}[x^{(k)}]}$ 和 $\beta^{(k)} = \mathbb{E}[x^{(k)}]$ ，如果这是最优选择，我们可以恢复原始激活值。

在基于整个训练集的 batch 设定中，我们将使用整个集来归一化激活值。然而，在使用随机优化时这是不现实的。因此，我们做了第二个简化：由于我们在随机梯度训练中使用 mini-batch，每个 mini-batch 会产生每个激活值的均值和方差估计。这样，用于归一化的统计量可以完全参与梯度反向传播。注意，mini-batch 的使用是通过计算每个维度的方差而非联合协方差来实现的；在联合情况下，由于 mini-batch 大小可能小于被白化的激活值数量，导致奇异协方差矩阵，因此需要正则化。

考虑一个大小为 $m$ 的 mini-batch $\mathcal{B}$ 。由于归一化独立应用于每个激活值，我们专注于一个特定的激活 $x^{(k)}$ ，为清晰起见省略 $k$ 。我们在 mini-batch 中有该激活的 $m$ 个值， $\mathcal{B} = \{x_{1\ldots m}\}$ 。令归一化后的值为 $\hat{x}_{1\ldots m}$ ，它们的线性变换为 $y_{1\ldots m}$ 。我们将变换

$$
\text{BN}_{\gamma,\beta}: x_{1\ldots m} \to y_{1\ldots m}
$$

称为**批归一化变换**（Batch Normalizing Transform）。我们在算法1中展示BN变换。在算法中， $\epsilon$ 是一个添加到 mini-batch 方差中的常数，用于数值稳定性。

---

**算法1：批归一化变换（Batch Normalizing Transform），应用于mini-batch上的激活值 $x$ 。**

**输入：** Mini-batch 上的 $x$ 值： $\mathcal{B} = \{x_{1\ldots m}\}$ ；需要学习的参数： $\gamma, \beta$

**输出：** $\{y_i = \text{BN}_{\gamma,\beta}(x_i)\}$

1: $\mu_{\mathcal{B}} \leftarrow \frac{1}{m} \sum_{i=1}^{m} x_i$ // mini-batch 均值
2: $\sigma_{\mathcal{B}}^2 \leftarrow \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu_{\mathcal{B}})^2$ // mini-batch 方差
3: $\hat{x}_i \leftarrow \frac{x_i - \mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}}$ // 归一化
4: $y_i \leftarrow \gamma \hat{x}_i + \beta \equiv \text{BN}_{\gamma,\beta}(x_i)$ // 缩放和平移

---

BN 变换可以添加到网络中，用于操作任何激活值。记法 $y = \text{BN}_{\gamma,\beta}(x)$ 表明参数 $\gamma$ 和 $\beta$ 是需要学习的，但应注意 BN 变换并不独立处理每个训练样本中的激活值。相反， $\text{BN}_{\gamma,\beta}(x)$ 既依赖于训练样本，也依赖于 mini-batch 中的其他样本。缩放和平移后的值 $y$ 被传递到其他网络层。归一化后的激活值 $\hat{x}$ 是我们的变换内部的值，但它们的存在至关重要。任何 $\hat{x}$ 值的分布都具有期望值 0 和方差 1——只要每个 mini-batch 的样本来自同一分布，并且我们忽略 $\epsilon$ 。这可以通过观察 $\sum_{i=1}^{m} \hat{x}_i = 0$ 和 $\frac{1}{m} \sum_{i=1}^{m} \hat{x}_i^2 = 1$ 并取期望看出。每个归一化后的激活值 $\hat{x}^{(k)}$ 可以视为一个子网络的输入，该子网络由线性变换 $y^{(k)} = \gamma^{(k)} \hat{x}^{(k)} + \beta^{(k)}$ 及原始网络的其他处理组成。这些子网络输入都具有固定的均值和方差，尽管这些归一化的 $\hat{x}^{(k)}$ 的联合分布在训练过程中可以变化，但我们期望归一化输入的引入能够加速子网络以及整个网络的训练。

在训练过程中，我们需要通过这个变换反向传播损失 $\ell$ 的梯度，并计算关于BN变换参数的梯度。我们使用链式法则，如下所示（简化前）：

$$
\frac{\partial \ell}{\partial \hat{x}_i} = \frac{\partial \ell}{\partial y_i} \cdot \gamma
$$

$$
\frac{\partial \ell}{\partial \sigma_{\mathcal{B}}^2} = \sum_{i=1}^{m} \frac{\partial \ell}{\partial \hat{x}_i} \cdot (x_i - \mu_{\mathcal{B}}) \cdot \frac{-1}{2} (\sigma_{\mathcal{B}}^2 + \epsilon)^{-3/2}
$$

$$
\frac{\partial \ell}{\partial \mu_{\mathcal{B}}} = \left( \sum_{i=1}^{m} \frac{\partial \ell}{\partial \hat{x}_i} \cdot \frac{-1}{\sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}} \right) + \frac{\partial \ell}{\partial \sigma_{\mathcal{B}}^2} \cdot \frac{\sum_{i=1}^{m} -2(x_i - \mu_{\mathcal{B}})}{m}
$$

$$
\frac{\partial \ell}{\partial x_i} = \frac{\partial \ell}{\partial \hat{x}_i} \cdot \frac{1}{\sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}} + \frac{\partial \ell}{\partial \sigma_{\mathcal{B}}^2} \cdot \frac{2(x_i - \mu_{\mathcal{B}})}{m} + \frac{\partial \ell}{\partial \mu_{\mathcal{B}}} \cdot \frac{1}{m}
$$

$$
\frac{\partial \ell}{\partial \gamma} = \sum_{i=1}^{m} \frac{\partial \ell}{\partial y_i} \cdot \hat{x}_i
$$

$$
\frac{\partial \ell}{\partial \beta} = \sum_{i=1}^{m} \frac{\partial \ell}{\partial y_i}
$$

因此，BN 变换是一个可微变换，它将归一化的激活值引入网络。这确保了在模型训练过程中，层可以继续在表现出较少内部协变量偏移的输入分布上学习，从而加速训练。此外，应用于这些归一化激活值的学习到的仿射变换允许 BN 变换表示恒等变换，并保留了网络容量。

### 3.1 批归一化网络的训练与推理

要对网络进行 Batch Normalization，我们指定一个激活子集，并为每个激活值根据算法1插入 BN 变换。任何之前以 $x$ 作为输入的层，现在接收 $\text{BN}(x)$ 。采用 Batch Normalization 的模型可以使用 batch 梯度下降、或 mini-batch 大小 $m > 1$ 的随机梯度下降、或其任何变体（如 Adagrad[4]）进行训练。依赖于 mini-batch 的激活值归一化可以实现高效训练，但在推理期间既非必要也不可取——我们希望输出仅确定性地依赖于输入。为此，在网络训练完成后，我们使用总体（population）统计量而非 mini-batch 统计量进行归一化：

$$
\hat{x} = \frac{x - \mathbb{E}[x]}{\sqrt{\text{Var}[x] + \epsilon}}
$$

忽略 $\epsilon$ ，这些归一化激活值与训练时具有相同的均值0和方差1。我们使用无偏方差估计 $\text{Var}[x] = \frac{m}{m-1} \cdot \mathbb{E}_{\mathcal{B}}[\sigma_{\mathcal{B}}^2]$ ，其中期望是针对大小为 $m$ 的训练 mini-batch 的， $\sigma_{\mathcal{B}}^2$ 是它们的样本方差。使用移动平均（moving averages），我们可以追踪模型在训练过程中的精度。由于推理过程中均值和方差是固定的，归一化只是一个应用于每个激活值的线性变换。它还可以进一步与 $\gamma$ 的缩放和 $\beta$ 的平移相结合，得到一个单一的线性变换来替代 $\text{BN}(x)$ 。算法2总结了训练批归一化网络的流程。

---

**算法2：训练批归一化网络**

**输入：** 具有可训练参数 $\Theta$ 的网络 $\mathcal{N}$ ；激活值子集 $\{x^{(k)}\}_{k=1}^{K}$

**输出：** 用于推理的批归一化网络 $\mathcal{N}_{\text{BN}}^{\text{inf}}$

1: $\mathcal{N}_{\text{BN}}^{\text{tr}} \leftarrow \mathcal{N}$ // 训练 BN 网络
2: **for** $k = 1 \ldots K$ **do**
3:    向 $\mathcal{N}_{\text{BN}}^{\text{tr}}$ 添加变换 $y^{(k)} = \text{BN}_{\gamma^{(k)}, \beta^{(k)}}(x^{(k)})$ （算法1）
4:    修改 $\mathcal{N}_{\text{BN}}^{\text{tr}}$ 中每个以 $x^{(k)}$ 作为输入的层，改为接收 $y^{(k)}$
5: **end for**
6: 训练 $\mathcal{N}_{\text{BN}}^{\text{tr}}$ 以优化参数 $\Theta \cup \{\gamma^{(k)}, \beta^{(k)}\}_{k=1}^{K}$
7: $\mathcal{N}_{\text{BN}}^{\text{inf}} \leftarrow \mathcal{N}_{\text{BN}}^{\text{tr}}$ // 推理 BN 网络，参数冻结
8: **for** $k = 1 \ldots K$ **do**
9:    // 为清晰起见， $x \equiv x^{(k)}$ , $\gamma \equiv \gamma^{(k)}$ , $\mu_{\mathcal{B}} \equiv \mu_{\mathcal{B}}^{(k)}$ 等
10:   处理多个大小为 $m$ 的训练 mini-batch $\mathcal{B}$ ，并对它们进行平均：
11:   $\mathbb{E}[x] \leftarrow \mathbb{E}_{\mathcal{B}}[\mu_{\mathcal{B}}]$
12:   $\text{Var}[x] \leftarrow \frac{m}{m-1} \mathbb{E}_{\mathcal{B}}[\sigma_{\mathcal{B}}^2]$
13:   在 $\mathcal{N}_{\text{BN}}^{\text{inf}}$ 中，将变换 $y = \text{BN}_{\gamma,\beta}(x)$ 替换为：
14:   $y = \frac{\gamma}{\sqrt{\text{Var}[x] + \epsilon}} \cdot x + \left( \beta - \frac{\gamma \mathbb{E}[x]}{\sqrt{\text{Var}[x] + \epsilon}} \right)$
15: **end for**

---

### 3.2 批归一化的卷积网络

Batch Normalization 可以应用于网络中任何一组激活值。这里，我们关注由仿射变换后接逐元素非线性组成的变换：

$$
z = g(W\mathbf{u} + b)
$$

其中 $W$ 和 $b$ 是模型学习到的参数， $g(\cdot)$ 是诸如 sigmoid 或 ReLU 之类的非线性。这种形式涵盖了全连接层和卷积层。我们通过在非线性之前归一化 $\mathbf{x} = W\mathbf{u} + b$ 来添加BN变换。我们也可以归一化层输入 $\mathbf{u}$ ，但由于 $\mathbf{u}$ 很可能是另一个非线性的输出，其分布的形状在训练过程中很可能发生变化，约束其一阶和二阶矩并不能消除协变量偏移。相比之下， $W\mathbf{u} + b$ 更可能具有对称的、非稀疏的分布，更接近高斯分布[7]；归一化它更可能产生具有稳定分布的激活值。

注意，由于我们对 $W\mathbf{u} + b$ 进行归一化，偏置 $b$ 可以被忽略，因为其效果会被随后的均值减法抵消（偏置的作用已被算法1中的 $\beta$ 所包含）。因此， $z = g(W\mathbf{u} + b)$ 被替换为：

$$
z = g(\text{BN}(W\mathbf{u}))
$$

其中 BN 变换独立应用于 $\mathbf{x} = W\mathbf{u}$ 的每个维度，每个维度有一对独立的学习参数 $\gamma^{(k)}, \beta^{(k)}$ 。

对于卷积层，我们额外希望归一化遵循卷积的性质——即同一特征图在不同位置的不同元素以相同方式进行归一化。为此，我们在 mini-batch 中跨所有位置联合归一化所有激活值。在算法1中，我们令 $\mathcal{B}$ 为特征图中所有值的集合，跨越 mini-batch 的元素和空间位置——因此对于大小为 $m$ 的 mini-batch 和大小为 $p \times q$ 的特征图，我们使用的有效 mini-batch 大小为 $m' = |\mathcal{B}| = m \cdot pq$ 。我们为每个特征图学习一对参数 $\gamma^{(k)}$ 和 $\beta^{(k)}$ ，而不是为每个激活值学习。算法2也进行了类似的修改，使得在推理过程中BN变换将相同的线性变换应用于给定特征图中的每个激活值。

### 3.3 Batch Normalization 支持更高的学习率

在传统深度网络中，过高的学习率可能导致梯度爆炸或消失，以及陷入不良局部最小值。Batch Normalization 有助于解决这些问题。通过在整个网络中归一化激活值，它防止了参数的小变化被放大为激活值和梯度中更大且次优的变化；例如，它防止训练陷入非线性的饱和区域。

Batch Normalization 还使训练对参数尺度更具鲁棒性。通常，大的学习率可能增加层参数的尺度，这随后在反向传播中放大梯度并导致模型爆炸。然而，使用 Batch Normalization 时，通过层的反向传播不受其参数尺度的影响。实际上，对于标量 $a$ ：

$$
\text{BN}(W\mathbf{u}) = \text{BN}((aW)\mathbf{u})
$$

并且我们可以证明：

$$
\frac{\partial \text{BN}((aW)\mathbf{u})}{\partial \mathbf{u}} = \frac{\partial \text{BN}(W\mathbf{u})}{\partial \mathbf{u}}
$$

$$
\frac{\partial \text{BN}((aW)\mathbf{u})}{\partial (aW)} = \frac{1}{a} \cdot \frac{\partial \text{BN}(W\mathbf{u})}{\partial W}
$$

尺度不影响层的 Jacobian，因此也不影响梯度传播。此外，更大的权重导致更小的梯度，Batch Normalization 将稳定参数的增长。

我们进一步推测，Batch Normalization 可能使层的 Jacobian 矩阵的奇异值接近1，众所周知这有利于训练[17]。考虑两个具有归一化输入的连续层，以及这些归一化向量之间的变换： $\hat{\mathbf{z}} = F(\hat{\mathbf{x}})$ 。如果我们假设 $\hat{\mathbf{x}}$ 和 $\hat{\mathbf{z}}$ 是高斯分布且不相关，并且 $F(\hat{\mathbf{x}}) \approx J\hat{\mathbf{x}}$ 对于给定的模型参数是一个线性变换，那么 $\hat{\mathbf{x}}$ 和 $\hat{\mathbf{z}}$ 都具有单位协方差，且 $I = \text{Cov}[\hat{\mathbf{z}}] = J \text{Cov}[\hat{\mathbf{x}}] J^T = JJ^T$ 。因此 $JJ^T = I$ ，所以 $J$ 的所有奇异值都等于1，这保持了反向传播中的梯度幅值。实际上，变换不是线性的，归一化后的值也不能保证是高斯的或独立的，但我们仍然期望 Batch Normalization 有助于使梯度传播更良性。Batch Normalization 对梯度传播的确切影响仍是进一步研究的领域。

### 3.4 Batch Normalization 对模型的正则化

当使用 Batch Normalization 训练时，一个训练样本是与 mini-batch 中的其他样本一起被观察的，训练网络不再为给定的训练样本产生确定性的值。在我们的实验中，我们发现这种效应对网络的泛化有利。虽然通常使用 Dropout[19] 来减少过拟合，但在批归一化网络中我们发现可以完全去除或降低其强度。

## 4 实验

### 4.1 激活值随时间的演变

为了验证内部协变量偏移对训练的影响以及 Batch Normalization 对抗它的能力，我们考虑了在 MNIST 数据集[9]上预测数字类别的问题。我们使用了一个非常简单的网络，输入为 $28 \times 28$ 的二值图像，包含3个全连接隐藏层，每层100个激活值。每个隐藏层使用 sigmoid 非线性计算 $y = g(W\mathbf{u} + b)$ ，权重 $W$ 初始化为小的随机高斯值。最后一个隐藏层之后是一个全连接层（每类一个激活值，共10个），以及交叉熵损失。我们训练网络50000步，每个 mini-batch 包含60个样本。我们按照第3.1节为网络的每个隐藏层添加了 Batch Normalization。我们关注的是基线网络和批归一化网络之间的比较（所描述的架构并不旨在达到 MNIST 上的最先进性能）。

图1(a)显示了两个网络在留出测试数据上的正确预测比例随训练过程的变化。批归一化网络获得了更高的测试精度。为了研究原因，我们考察了训练过程中原始网络 $\mathcal{N}$ 和批归一化网络 $\mathcal{N}_{\text{BN}}^{\text{tr}}$ （算法2）中 sigmoid 的输入。在图1(b,c)中，我们展示了每个网络最后一个隐藏层中一个典型激活值的分布如何演变。原始网络中的分布随时间的推移发生显著变化，无论是均值还是方差，这使后续层的训练变得复杂。相比之下，批归一化网络中的分布在训练过程中更加稳定，这有助于训练。

<img src="fig1.png" alt="Figure 1">

**图1：**(a) 使用和不使用 Batch Normalization 训练的 MNIST 网络的测试精度与训练步数的关系。Batch Normalization 帮助网络训练更快并获得更高的精度。(b, c) 训练过程中典型 sigmoid 输入分布的演变，显示为第15、50、85百分位数。Batch Normalization 使分布更稳定并减少了内部协变量偏移。

### 4.2 ImageNet 分类

我们将 Batch Normalization 应用于 Inception 网络[21]的一个新变体，在 ImageNet 分类任务[16]上进行训练。该网络包含大量卷积层和池化层，以及一个用于预测图像类别的 softmax 层（共1000类）。卷积层使用 ReLU 作为非线性函数。与[21]描述的网络的主要区别在于： $5 \times 5$ 卷积层被替换为两个连续的 $3 \times 3$ 卷积层，最多128个滤波器。该网络包含 $13.6 \times 10^6$ 个参数，除了顶部的 softmax 层外，没有全连接层。更多细节见附录。我们在后文中将此模型称为 Inception。该模型使用带有动量的随机梯度下降[20]进行训练，mini-batch 大小为32。训练使用大规模分布式架构（类似于[2]）。所有网络在训练过程中通过在留出集上使用每张图像单次裁剪来计算验证集 top-1 准确率（即从1000个类别中预测正确标签的概率）进行评估。

在我们的实验中，我们评估了 Inception 的几种带有 Batch Normalization 的修改版本。在所有情况下，Batch Normalization 按照第3.2节描述的卷积方式应用于每个非线性函数的输入，同时保持其余架构不变。

#### 4.2.1 加速BN网络

仅仅向网络添加 Batch Normalization 并不能充分利用我们的方法。为此，我们进一步改变了网络及其训练参数，具体如下：

- **提高学习率。** 在批归一化模型中，我们能够通过更高的学习率实现训练加速，且没有不良副作用（第3.3节）。
- **移除 Dropout。** 如第3.4节所述，Batch Normalization 实现了 Dropout 的某些相同目标。从修改后的 BN-Inception 中移除 Dropout 加速了训练，同时没有增加过拟合。
- **降低L2权重正则化。** 在 Inception 中，模型参数上的 L2 损失控制过拟合，而在修改后的 BN-Inception 中，该损失的权重降低了5倍。我们发现这提高了留出验证数据上的精度。
- **加速学习率衰减。** 在训练 Inception 时，学习率呈指数衰减。由于我们的网络训练速度比 Inception 快，我们将学习率降低的速度提高了6倍。
- **移除局部响应归一化。** 虽然 Inception 和其他网络[19]从中受益，但我们发现使用 Batch Normalization 时不再需要它。
- **更彻底地打乱训练样本。** 我们启用了训练数据的分片内洗牌（within-shard shuffling），防止同一样本总是出现在同一个 mini-batch 中。这使验证精度提高了大约1%，这与将 Batch Normalization 视为正则化器（第3.4节）的观点一致：当我们的方法带来的随机化在每次看到样本时产生不同影响时，其效果最为有益。
- **减少光度失真。** 由于批归一化网络训练更快，每个训练样本被观察的次数更少，我们通过减少失真来让训练器专注于更"真实"的图像。

<img src="fig2.png" alt="Figure 2">

**图2：** Inception 及其批归一化变体的单次裁剪验证精度与训练步数的关系。

<img src="fig3.png" alt="Figure 3">

**图3：** 对于 Inception 及其批归一化变体，达到 Inception 最大精度（72.2%）所需的训练步数，以及网络达到的最大精度。

#### 4.2.2 单网络分类

我们评估了以下网络，所有网络均在 LSVRC2012 训练数据上训练，并在验证数据上测试：

- **Inception：** 第4.2节开头描述的网络，初始学习率为0.0015。
- **BN-Baseline：** 与 Inception 相同，但在每个非线性之前添加 Batch Normalization。
- **BN-x5：** 带有 Batch Normalization 和第4.2.1节修改的 Inception。初始学习率提高了5倍，达到0.0075。对原始 Inception 进行同样的学习率提升会导致模型参数达到机器无穷大。
- **BN-x30：** 与 BN-x5 类似，但初始学习率为0.045（Inception 的30倍）。
- **BN-x5-Sigmoid：** 与 BN-x5 类似，但使用 sigmoid 非线性 $g(t) = \frac{1}{1+\exp(-x)}$ 替代 ReLU。我们也尝试用 sigmoid 训练原始 Inception，但模型精度始终停留在随机水平。

在图2中，我们展示了各网络的验证精度与训练步数的关系。Inception 在 $31 \times 10^6$ 步后达到72.2%的精度。图3显示了每个网络达到同样72.2%精度所需的训练步数、网络达到的最高验证精度以及达到该精度所需的步数。

仅通过使用 Batch Normalization（BN-Baseline），我们在不到一半的训练步数内就匹配了 Inception 的精度。通过应用第4.2.1节的修改，我们显著提高了网络训练速度。BN-x5 达到72.2%精度所需的步数比 Inception 少了14倍。有趣的是，进一步提高学习率（BN-x30）使模型最初训练稍慢，但使其达到了更高的最终精度。它在 $6 \times 10^6$ 步后达到74.8%，即仅需 Inception 达到72.2%所需步数的五分之一。

我们还验证了内部协变量偏移的减少使得使用 sigmoid 作为非线性的深度批归一化网络能够被训练，尽管众所周知这类网络很难训练。实际上，BN-x5-Sigmoid 达到了69.8%的精度。如果没有 Batch Normalization，使用 sigmoid 的 Inception 从未达到超过1/1000的精度。

#### 4.2.3 集成分类

目前 ImageNet 大规模视觉识别挑战（ILSVRC）上报告的最佳结果由传统模型的 Deep Image 集成[24]和 He 等人的集成模型[6]达到。后者报告的 top-5 错误率为4.94%（由 ILSVRC 服务器评估）。在这里我们报告 top-5 验证错误率为4.9%，测试错误率为4.82%（根据 ILSVRC 服务器）。这改进了先前的最佳结果，并超过了根据[16]估计的人类标注者精度。

我们的集成使用了6个网络。每个网络基于 BN-x30，并通过以下一些方式进行了修改：增加卷积层中的初始权重；使用 Dropout（Dropout 概率为5%或10%，而原始 Inception 为40%）；以及在模型最后隐藏层使用非卷积的、逐激活值的 Batch Normalization。每个网络在大约 $6 \times 10^6$ 训练步后达到其最大精度。集成预测基于各组成网络预测的类别概率的算术平均值。集成和多裁剪推理的细节与[21]类似。

<img src="fig4.png" alt="Figure 4">

**图4：** 批归一化 Inception 与先前最先进方法在包含50000张图像的验证集上的比较。*BN-Inception 集成在 ImageNet 测试集（100000张图像）上达到4.82%的 top-5 错误率（由测试服务器报告）。

我们在图4中展示了批归一化使我们能够在 ImageNet 分类挑战基准上以显著优势创造新的最先进水平。

## 5 结论

我们提出了一种显著加速深度网络训练的新机制。它基于这样一个前提：已知会使机器学习系统训练复杂化的协变量偏移也同样适用于子网络和层，将其从网络的内部激活中移除有助于训练。我们提出的方法的力量源于归一化激活值以及将这种归一化融入网络架构本身。这确保了归一化能够被任何用于训练网络的优化方法正确处理。为了支持深度网络训练中常用的随机优化方法，我们对每个 mini-batch 执行归一化，并通过归一化参数反向传播梯度。Batch Normalization 每个激活值仅增加两个额外参数，从而保留了网络的表示能力。我们提出了构建、训练和执行批归一化网络推理的算法。由此产生的网络可以使用饱和非线性进行训练，对更高的学习率更鲁棒，且通常不需要 Dropout 进行正则化。

仅仅将 Batch Normalization 添加到最先进的图像分类模型就能带来显著的训练加速。通过进一步提高学习率、移除 Dropout 以及应用 Batch Normalization 带来的其他修改，我们仅需一小部分训练步数就达到了先前的最先进水平——然后在单网络图像分类中超越了最先进水平。此外，通过组合多个经 Batch Normalization 训练的网络，我们以显著优势超越了 ImageNet 上已知的最佳系统。

有趣的是，我们的方法与 Gülçehre 和 Bengio[5] 的标准化层有相似之处，尽管两种方法源于截然不同的目标，并执行不同的任务。Batch Normalization 的目标是在整个训练过程中实现激活值的稳定分布，在我们的实验中我们将其应用于非线性之前，因为在那里匹配一阶和二阶矩更有可能产生稳定的分布。相反，Gülçehre 和 Bengio[5] 将标准化层应用于非线性输出，这导致更稀疏的激活值。在我们的大规模图像分类实验中，无论是否使用 Batch Normalization，我们都没有观察到非线性输入是稀疏的。Batch Normalization 的其他显著区别特征包括：学习到的缩放和平移使 BN 变换能够表示恒等变换（标准化层不需要这个，因为它后面跟着一个学习到的线性变换，该变换在概念上吸收了必要的缩放和平移）、对卷积层的处理、不依赖于 mini-batch 的确定性推理，以及对网络中每个卷积层的批归一化。

在这项工作中，我们没有探索 Batch Normalization 可能带来的全部可能性。我们未来的工作包括将我们的方法应用于循环神经网络（RNN）[13]，其中内部协变量偏移和梯度消失/爆炸可能尤为严重，这将使我们能够更彻底地检验归一化改善梯度传播的假设（第3.3节）。我们计划研究 Batch Normalization 是否有助于传统意义上的领域自适应——即网络执行的归一化是否使其能够更容易地泛化到新的数据分布，也许只需重新计算总体均值和方差（算法2）。最后，我们相信对算法进一步的理论分析将带来更多的改进和应用。

## 参考文献

[1] Bengio, Yoshua and Glorot, Xavier. Understanding the difficulty of training deep feedforward neural networks. In *Proceedings of AISTATS 2010*, volume 9, pp. 249–256, May 2010.

[2] Dean, Jeffrey, Corrado, Greg S., Monga, Rajat, Chen, Kai, Devin, Matthieu, Le, Quoc V., Mao, Mark Z., Ranzato, Marc'Aurelio, Senior, Andrew, Tucker, Paul, Yang, Ke, and Ng, Andrew Y. Large scale distributed deep networks. In *NIPS*, 2012.

[3] Desjardins, Guillaume and Kavukcuoglu, Koray. Natural neural networks. (unpublished).

[4] Duchi, John, Hazan, Elad, and Singer, Yoram. Adaptive subgradient methods for online learning and stochastic optimization. *J. Mach. Learn. Res.*, 12:2121–2159, July 2011.

[5] Gülçehre, Çağlar and Bengio, Yoshua. Knowledge matters: Importance of prior information for optimization. *CoRR*, abs/1301.4083, 2013.

[6] He, K., Zhang, X., Ren, S., and Sun, J. Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification. *ArXiv e-prints*, February 2015.

[7] Hyvärinen, A. and Oja, E. Independent component analysis: Algorithms and applications. *Neural Netw.*, 13(4-5):411–430, May 2000.

[8] Jiang, Jing. A literature survey on domain adaptation of statistical classifiers, 2008.

[9] LeCun, Y., Bottou, L., Bengio, Y., and Haffner, P. Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11):2278–2324, November 1998a.

[10] LeCun, Y., Bottou, L., Orr, G., and Muller, K. Efficient backprop. In Orr, G. and K., Muller (eds.), *Neural Networks: Tricks of the trade*. Springer, 1998b.

[11] Lyu, S and Simoncelli, E P. Nonlinear image representation using divisive normalization. In *Proc. Computer Vision and Pattern Recognition*, pp. 1–8. IEEE Computer Society, Jun 23-28 2008.

[12] Nair, Vinod and Hinton, Geoffrey E. Rectified linear units improve restricted boltzmann machines. In *ICML*, pp. 807–814. Omnipress, 2010.

[13] Pascanu, Razvan, Mikolov, Tomas, and Bengio, Yoshua. On the difficulty of training recurrent neural networks. In *Proceedings of the 30th International Conference on Machine Learning, ICML 2013, Atlanta, GA, USA, 16-21 June 2013*, pp. 1310–1318, 2013.

[14] Povey, Daniel, Zhang, Xiaohui, and Khudanpur, Sanjeev. Parallel training of deep neural networks with natural gradient and parameter averaging. *CoRR*, abs/1410.7455, 2014.

[15] Raiko, Tapani, Valpola, Harri, and LeCun, Yann. Deep learning made easier by linear transformations in perceptrons. In *International Conference on Artificial Intelligence and Statistics (AISTATS)*, pp. 924–932, 2012.

[16] Russakovsky, Olga, Deng, Jia, Su, Hao, Krause, Jonathan, Satheesh, Sanjeev, Ma, Sean, Huang, Zhiheng, Karpathy, Andrej, Khosla, Aditya, Bernstein, Michael, Berg, Alexander C., and Fei-Fei, Li. ImageNet Large Scale Visual Recognition Challenge, 2014.

[17] Saxe, Andrew M., McClelland, James L., and Ganguli, Surya. Exact solutions to the nonlinear dynamics of learning in deep linear neural networks. *CoRR*, abs/1312.6120, 2013.

[18] Shimodaira, Hidetoshi. Improving predictive inference under covariate shift by weighting the log-likelihood function. *Journal of Statistical Planning and Inference*, 90(2):227–244, October 2000.

[19] Srivastava, Nitish, Hinton, Geoffrey, Krizhevsky, Alex, Sutskever, Ilya, and Salakhutdinov, Ruslan. Dropout: A simple way to prevent neural networks from overfitting. *J. Mach. Learn. Res.*, 15(1):1929–1958, January 2014.

[20] Sutskever, Ilya, Martens, James, Dahl, George E., and Hinton, Geoffrey E. On the importance of initialization and momentum in deep learning. In *ICML (3)*, volume 28 of *JMLR Proceedings*, pp. 1139–1147. JMLR.org, 2013.

[21] Szegedy, Christian, Liu, Wei, Jia, Yangqing, Sermanet, Pierre, Reed, Scott, Anguelov, Dragomir, Erhan, Dumitru, Vanhoucke, Vincent, and Rabinovich, Andrew. Going deeper with convolutions. *CoRR*, abs/1409.4842, 2014.

[22] Wiesler, Simon and Ney, Hermann. A convergence analysis of log-linear training. In Shawe-Taylor, J., Zemel, R.S., Bartlett, P., Pereira, F.C.N., and Weinberger, K.Q. (eds.), *Advances in Neural Information Processing Systems 24*, pp. 657–665, Granada, Spain, December 2011.

[23] Wiesler, Simon, Richard, Alexander, Schlüter, Ralf, and Ney, Hermann. Mean-normalized stochastic gradient for large-scale deep learning. In *IEEE International Conference on Acoustics, Speech, and Signal Processing*, pp. 180–184, Florence, Italy, May 2014.

[24] Wu, Ren, Yan, Shengen, Shan, Yi, Dang, Qingqing, and Sun, Gang. Deep image: Scaling up image recognition, 2015.

---

## 附录：使用的 Inception 模型变体

图5记录了与 GoogLeNet 架构相比所执行的修改。关于此表的解读，请参考[21]。与 GoogLeNet 模型相比，显著的架构变化包括：

- $5 \times 5$ 卷积层被替换为两个连续的 $3 \times 3$ 卷积层。这使网络的最大深度增加了9个权重层。同时参数数量增加了25%，计算成本增加了约30%。
- $28 \times 28$ 的 Inception 模块数量从2个增加到3个。
- 在模块内部，有时使用平均池化，有时使用最大池化。这在表中对应池化层的条目中有所指示。
- 任意两个 Inception 模块之间没有全局的池化层，但在模块3c和4e的滤波器拼接之前使用了步长为2的卷积/池化层。
- 我们的模型在第一个卷积层上使用了深度乘数为8的可分离卷积（separable convolution）。这降低了计算成本，同时增加了训练时的内存消耗。

<img src="fig5.png" alt="Figure 5">

**图5：** Inception 架构
