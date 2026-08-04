# 2018-Heated-Up Softmax Embedding

> Xu Zhang†, Felix Xinnan Yu‡, Svebor Karaman†, Wei Zhang†, & Shih-Fu Chang† | †哥伦比亚大学，‡Google Research


本文分享了关于softmax函数中温度参数如何影响深度神经网络 **瓶颈层** 嵌入分布的研究。传统上，度量学习通过设计专门的损失函数（如对比损失或三元组损失）来学习嵌入，但这些方法面临 **采样困难** 的问题。本文揭示了softmax分类器中的温度参数实际上控制着**不同样本的梯度分配**，从而影响嵌入的紧致性。

核心内容：

- 理论分析了softmax温度参数 $\alpha(=1/T)$ 如何通过梯度分配影响嵌入分布——**高 $\alpha$ 使困难样本获得大梯度而简单样本梯度趋近零，低 $\alpha$ 则为所有样本分配相似梯度**
- 提出"加热"（heating-up）策略：先用中等温度 $T$ 训练使困难样本快速收敛，再用更高温度 $T$ 微调 使 **边界样本** 和中心样本获得足够更新，从而提高 **同类样本嵌入** 的紧致性
- 在Cars196、CUB200、Stanford Online Product和In-shop Clothes Retrieval四个数据集上达到最先进或可比的**度量学习性能**

关键发现：

- 温度参数 $\alpha$ 是**控制嵌入紧致性的关键因素**： $\alpha$ 过小导致训练效率低下， $\alpha$ 过大使 边界样本 更新不足、同类嵌入 不紧致，中等 $\alpha$ 可在困难样本和简单样本间取得平衡
- "加热"策略（从 $\alpha=16$ 降至 $\alpha=4$ 微调）在所有评估指标上一致优于固定温度训练的模型，且训练效率与最快的现有方法ProxyNCA相当

备注：我的理解，困难样本离正样本最远，边界样本次之，中心样本最近

---


## 摘要

度量学习旨在**学习一种与样本语义含义一致的距离度量**。该问题通常通过为每个样本学习一个嵌入来解决，使得**同类样本的嵌入在特征空间中紧致排列，而异类样本的嵌入分散分布**。本文研究了基于softmax层的交叉熵损失训练的深度神经网络分类器倒数第二层提取的特征。我们发现，使用不同softmax温度值训练分类器会产生不同紧致程度的特征。基于这些发现，我们提出了一种"加热"策略，即**使用递增的温度训练分类器**，使相应嵌入在各种度量学习基准测试中达到最先进的性能。

---


## 1 引言

度量学习旨在学习一个度量空间，在该空间中，同类样本彼此接近（紧致），异类样本彼此远离（分散）[7,22,35]。这是机器学习中的一个基础研究课题，已广泛应用于各种计算机视觉任务，包括聚类[32,33]、图像检索[13,35]、人脸识别[4,22]和行人重识别[9,14]。

解决该问题的一种方法是定义一个损失函数，以强制度量空间中的紧致性和分散性。两种最流行的损失函数是对比损失[3]和三元组损失[7]。然而，这两种损失**在采样方面都面临挑战**，因为**一个数据集中通常存在数量极为庞大的可能配对或三元组**。

为克服采样问题，研究者提出了多种 **难例挖掘策略**[22,17,5]。然而，采样"最困难"的样本（**最违反预定义规则的样本**）通常会导致**较差的局部极小值**。纳入过多的"简单"样本则**会使训练效率低下**。因此，设计一种结构化损失以 **有效且高效** 地进行难例挖掘已成为一个热门研究课题[25,24]。

使用softmax函数和交叉熵损失训练的深度神经网络分类器**倒数第二层（即瓶颈层）**提取的特征，在许多基于度量学习的应用中效果良好[20]，如图像检索[2]和人脸验证[15]。然而，**分类器训练和度量学习的目标并不相同**：前者旨在寻找最佳决策函数，而后者是学习一种嵌入，使得同类样本的嵌入紧致、异类样本的嵌入"分散"。这促使我们研究度量学习与分类器训练之间的关系。

本文证明了softmax函数中的温度参数（由Hinton等人[6]为知识迁移而定义）在决定瓶颈层嵌入分布方面起着重要作用。**基于观察到的关系，我们提出在训练开始时使用中等温度训练分类器，并在训练过程中提高温度。**与深度度量学习中最先进的方法相比，所提出的"加热"方法在大多数情况下取得了显著更优的性能，在其余情况下也至少达到了可比的性能。

本文的主要贡献如下：

*   我们研究了softmax层的梯度，并展示了温度参数如何影响瓶颈层嵌入的最终分布；
*   我们提出了一种"加热"方法，与深度度量学习中最先进的方法相比，该方法可获得有效嵌入，性能显著更优或至少可比。

---


## 2 相关工作

基于对比损失的孪生网络[3]是解决度量学习问题的最早尝试之一。**通过从同一类别中采样两个数据样本（正样本对）或从两个不同类别中采样（负样本对），对比损失试图将正样本对的两个点拉近，将负样本对的两个点推远**。三元组损失[7]进一步要求**正样本对距离与负样本对距离之间存在间隔**。对比损失和三元组损失的主要问题之一是，对于大规模数据集，可能的配对或三元组数量极其庞大。

解决采样问题的一个合理方案是**挖掘对训练信息量最大的样本**，即"难例挖掘"。大量工作致力于解决这一问题[22,17,5,34,31]。半难例挖掘[22]尝试在训练批次中找到**正样本对距离和负样本对距离都在一定间隔内的三元组**。HardNet[17]旨在挖掘单个训练批次中的最困难三元组。

设计结构化损失以考虑单个训练批次中所有可能的训练配对或三元组并执行"软"难例挖掘，是难例挖掘的另一种解决方案[25,24,27]。提升结构化损失[24]利用训练批次中的全部三元组，为难例挖掘提供平滑的损失函数。一些基于深度聚类的损失[11,25]也被提出以解决该问题。Proxy NCA[18]提出为训练数据学习语义代理，并使用NCA损失进行训练。与使用样本相比，**应用代理进行难例挖掘更加高效**。

在人脸验证领域，相当多的研究表明，训练一个分类器并将倒数第二层的输出作为嵌入使用，效果相当不错[29]。NormFace[28]和SphereFace[15]建议对嵌入和分类器权重都进行 $\ell_2$ 归一化。为获得理想的结果，通常需要将一个可学习或固定的标量乘到最终的logits上[19]。关于该标量如何影响最终嵌入，已有一些初步的讨论[28,19]。

本文表明，该标量可视为Hinton等人[6]提出的softmax函数中的温度参数。我们分析了该温度参数如何根据样本相对于分类器边界的位置为其分配不同的梯度，从而控制嵌入的分布，尤其是紧致性。受这些发现的启发，我们提出了一种用于训练嵌入的"加热"策略，即在训练分类器时**使用递增的温度**。所提出的方法使得使用softmax函数和交叉熵损失训练的嵌入能够达到与最先进的深度度量学习方法可比或更优的性能。

---


## 3 重新审视带温度的Softmax嵌入

给定一组带有标签的训练样本 $\{(x_1, y_1), \ldots, (x_n, y_n)\}$ ，其中 $x_i \in \mathbb{R}^d$ 是数据样本， $d$ 是训练数据的维度， $y_i \in \{1, \ldots, M\}$ 是样本 $x_i$ 的类别标签， $M$ 是训练样本的类别数。我们尝试学习一个嵌入函数 $f(\cdot): \mathbb{R}^d \to \mathbb{R}^k$ ，将数据样本映射到 $\mathbb{R}^k$ 中的向量，使得对于所有满足 $y_i = y_j \neq y_p$ 的 $i, j, p$ ，有 $l(f(x_i), f(x_j)) < l(f(x_i), f(x_p))$ ，其中 $l(\cdot, \cdot): \mathbb{R}^k \times \mathbb{R}^k \to \mathbb{R}$ 是一个距离函数。

我们将 $f(x) \in \mathbb{R}^k$ 称为**数据样本 $x$ 的嵌入**，并使用 $f$ 作为 $f(x)$ 的简写符号。考虑训练一个线性分类器 $W = [\mathbf{w}_1, \ldots, \mathbf{w}_M] \in \mathbb{R}^{k \times M}$ 和 $b = [b_1, \ldots, b_M]^T \in \mathbb{R}^M$ ， $z = [z_1, \ldots, z_M]^T = W^T f + b \in \mathbb{R}^M$ 称为logits。样本 $x$ 属于类别 $m \in \{1, \ldots, M\}$ 的概率可由softmax函数预测为：

$$
p(m|x) = \frac{\exp(z_m/T)}{\sum_{j=1}^M \exp(z_j/T)} = \frac{\exp(\alpha z_m)}{\sum_{j=1}^M \exp(\alpha z_j)} \qquad (1)
$$

 $T$ （通常设为1）即Hinton等人[6]提到的温度参数。为简化本文符号，我们设 $\alpha = 1/T$ 为**温度的倒数**。

设训练样本的真实分布为 $q(m|x)$ ，通常 $q(m|x)$ 是一个Dirac delta函数，当 $m = y$ 时等于1，否则为0，其中 $y$ 是 $x$ 的真实标签。关于 $x$ 的交叉熵损失及其关于 $z_m$ 的梯度定义为：

$$
\ell(x, \alpha) = -\sum_{m=1}^M \log(p(m|x, \alpha)) q(m|x)
$$

和

$$
\frac{\partial \ell}{\partial z_m} = \alpha(p(m|x, \alpha) - q(m|x)) \qquad (2)
$$

考虑到 $z_m = \mathbf{w}_m^T f + b_m$ ，我们有：

$$
\frac{\partial \ell}{\partial f} = \sum_{m=1}^M \frac{\partial \ell}{\partial z_m} \frac{\partial z_m}{\partial f} = \alpha \sum_{m=1}^M (p(m|x, \alpha) - q(m|x)) \mathbf{w}_m \qquad (3)
$$

所提出的"加热"思想基于如下观察：不同的 $\alpha$ 值会为不同样本分配不同幅度的梯度，从而改变最终嵌入的分布。为说明这一点，由于 $\alpha$ 、嵌入范数和分类器权重范数都会影响softmax函数（公式(1)和(3)），我们遵循Wang等人[28]和Ranjan等人[19]的做法，对分类器权重 $\mathbf{w}_i$ 和特征 $f$ 进行 $\ell_2$ 归一化，使其具有单位范数。归一化后的特征和权重分别记为 $\hat{f}$ 和 $\hat{\mathbf{w}}_i$ 。

本节余下部分将首先讨论在使用 $\ell_2$ 归一化嵌入和权重的情况下， $\alpha$ 如何改变梯度分配（第3.1节）和最终嵌入（第3.2节），然后讨论归一化的影响（第3.3节）。最后，我们还在第3.4节中讨论现成分类器的嵌入和梯度性质。


### 3.1 $\alpha$ 的梯度分配

在本小节中，我们将展示使用不同 $\alpha$ 值训练深度分类网络如何影响不同训练样本的梯度。由公式(1)，当 $\alpha \to +\infty$ 时， $p(m|x, \alpha)$ 满足：

$$
\lim_{\alpha \to +\infty} p(m|x, \alpha) = \begin{cases}
1/K & \text{if } z_m = \max(z_1, \ldots, z_M) \\
0 & \text{otherwise},
\end{cases} \qquad (4)
$$

其中 $K$ 是其值等于最大logits值的logits数量。另一方面，如果 $\alpha$ 趋近于0，预测概率将趋近于均匀分布。换句话说，随着 $\alpha$ 增大，预测概率在具有最大值的logits处将变得更加"尖锐"。

我们定义如图1所示的2类训练样本。图中，所有数据样本（叉号、三角形和圆形）属于同一类别。所有位于蓝色虚线标定区域内的样本都会被分类器分类为正确类别。训练样本可分为：(i) "困难"样本（叉号）：未被分类为正确类别的样本（ $\{x: \exists m \neq y, z_m \geq z_y\}$ ）；(ii) "简单"样本：被分类器正确分类的样本（ $\{x: \forall m \neq y, z_y > z_m\}$ ）。"简单"类别中有两种子类型："边界"样本（三角形）是靠近决策边界的样本；"中心"样本（圆形）是靠近该类别所属区域中心的样本。


**图1：不同 $\alpha$ 值通过为不同样本分配不同梯度来控制最终嵌入。橙色箭头表示梯度。**
(a) 小 $\alpha$
(b) 中等 $\alpha$
(c) 大 $\alpha$

损失关于归一化嵌入的梯度 $\frac{\partial \ell}{\partial \hat{f}}$ （即公式(3)中将 $f$ 和 $\mathbf{w}_i$ 替换为 $\hat{f}$ 和 $\hat{\mathbf{w}}_i$ ）包含了 $M$ 个求和项，每个类别对应一项。共有3类项：类型1，关于真实类别的项；类型2，关于具有最大值且不属于真实类别的logits的项；类型3，其他项。我们研究在 $\alpha$ 非常小和非常大时，这些项对"困难"和"简单"样本的行为。首先，容易看出当 $\alpha \to 0$ 时，任何样本的梯度幅度都将趋近于0。

对于 $\alpha \to +\infty$ ，首先考虑"困难"样本，类型1项（ $\alpha(p(y|x, \alpha) - q(y|x)) \hat{\mathbf{w}}_y$ ）的幅度将趋近于 $+\infty$ ，因为 $p(y|x, \alpha)$ 将趋近于0或 $1/K$ （ $K \geq 2$ ）[^1] 且 $q(y|x) = 1$ 。类似地，类型2项的幅度也将趋近于 $+\infty$ 。对于其他项，由于指数函数的性质——如果 $z_m \neq \max(z_1, \ldots, z_M)$ ，则 $\lim_{\alpha \to +\infty} \alpha p(m|x, \alpha) = 0$ ——所有类型3项的幅度将减小到0。因此，对于"困难样本"，当 $\alpha \to +\infty$ 时，除非在某些特殊情况下[^2]，关于归一化嵌入的梯度幅度将趋近于无穷大。图2(a)显示了从第4节网络导出的三个不同"困难"样本的梯度幅度随 $\alpha$ 变化的情况。

**

**图2： $\alpha$ 与嵌入梯度幅度之间的关系，针对"困难"和"简单"样本。**
(a) 对困难样本
(b) 对简单样本

考虑"简单"样本的类型1项，由于 $\lim_{\alpha \to +\infty} p(y|x, \alpha) = 1$ 且 $\lim_{\alpha \to +\infty} \alpha(p(y|x, \alpha) - 1) = 0$ ，该项的幅度将趋近于0。对于其他项，幅度也将趋近于0。因此，梯度幅度始终趋近于0。图2(b)显示了三个不同"简单"样本的嵌入梯度幅度随 $\alpha$ 变化的情况。

总的来说，当 $\alpha$ 较大时，困难样本的梯度幅度变得非常大，而简单样本的梯度幅度变得非常小（图1(c)）。而当 $\alpha$ 较小时，网络会为所有样本分配相似幅度的梯度（图1(a)）。选择中等 $\alpha$ 值是一种折衷（图1(b)）。不同样本的梯度分配将极大地影响最终嵌入，我们将在下一节讨论。

### 3.2 最终嵌入的分布与"加热"策略

为说明 $\alpha$ 对嵌入分布的影响，我们在图3(a)-3(d)中展示了在训练过程中使用不同 $\alpha$ 值时在MNIST数据集上获得的嵌入。不同颜色代表不同数字，每个菱形对应相应数字的分类器权重。为便于可视化，我们将权重向原点做了轻微移动。基础模型是LeNet[12]，倒数第二层的节点数设为2以进行可视化。数据集中使用50,000个样本进行训练，10,000个不同的测试样本来绘制图形。所有特征和分类器权重在训练过程中都进行了 $\ell_2$ 归一化。

**

**图3：使用 $\ell_2$ 归一化和不同 $\alpha$ 值训练的嵌入。**
(a) $\alpha = 0.25$
(b) $\alpha = 4$
(c) $\alpha = 16$
(d) $\alpha = 64$

当使用小 $\alpha$ （即高温度）训练时，网络会为所有样本分配相似的梯度（见图1(a)）。由于"困难"样本对提高分类器精度更为重要，同等程度地更新"困难"样本和"简单"样本会使训练效率低下，甚至难以收敛（图3(a)）。然而，选择非常大的 $\alpha$ （即非常低的温度）进行训练会为"困难"样本分配大梯度，而为所有"简单"样本（"边界"和"中心"）分配非常小的梯度。由于"边界"样本得不到充分的更新，它们将停留在决策边界附近。因此，同一类别的样本嵌入将不够紧致（图3(d)）。因此，一个良好的折衷是使用中等温度进行训练（见图1(b)），此时"中心"样本获得小梯度，"边界"样本获得中等梯度，而"困难"样本获得大梯度。比较使用不同 $\alpha$ 值训练的分类器，使用较小 $\alpha$ 值训练的模型其同类别的特征更加紧致（图3(d) $\to$ 图3(b)）。

我们进一步提出了一种"加热"策略，即先以较低或中等温度开始训练，使"困难"样本获得大梯度进行更新并迅速成为"简单"样本。之后，提高温度，使"边界"和"中心"样本也能获得足够的梯度进行更新。因此，与仅使用起始温度训练的模型相比，同类别样本的最终嵌入将变得更加紧致。

可以定义多种策略来在训练过程中提高温度。我们尝试了：(i) 平稳地提高温度；(ii) 使用起始温度训练至收敛，然后使用更高温度微调训练好的网络。两种方法都能得到相似的性能。然而，由于前一种方法会引入一个额外的参数来控制温度升高的速度，我们在第4节的实验中采用了后一种方法。

### 3.3 归一化的影响

我们在此讨论归一化的影响。回顾一下，我们将原始嵌入记为 $f$ ，将 $\ell_2$ 归一化嵌入记为 $\hat{f}$ 。 $\hat{f}$ 关于 $f$ 的雅可比矩阵为：

$$
J_{\hat{f}}(f) = \frac{1}{\|f\|_2} (I - \hat{f} \hat{f}^T), \qquad (5)
$$

其中 $I$ 是单位矩阵。考虑公式(3)和链式法则，我们有：

$$
\frac{\partial \ell}{\partial f} = \left(\frac{\partial \ell}{\partial \hat{f}}\right)^T J_{\hat{f}}(f) \qquad (6)
$$

考虑到分母中的 $\|f\|_2$ ，梯度幅度与嵌入的范数成反比。因此，即使归一化嵌入相同，对于具有不同范数的嵌入，关于嵌入的梯度仍然不同。范数较大的嵌入将具有较小的梯度。这看起来可能是个问题，一种可能的解决方案是移除分母中的范数项 $\|f\|_2$ 。我们在第4节的实验中尝试了这一想法，但没有带来显著的改进。原因可能在于，由于 $\partial \ell/\partial f$ 与 $f$ 始终正交[28]，沿着梯度方向更新特征不会太大改变特征的范数。我们确实观察到，当在训练中对特征应用 $\ell_2$ 归一化时，归一化之前的特征范数非常相似。相反，在不使用归一化进行训练时，特征的范数可能会有很大变化（见第3.4节）。出于数值稳定性和实现简便的考虑，我们使用公式(6)来计算梯度。因此，第3.1节和3.2节中的梯度分析对于归一化之前的非归一化特征仍然成立。

**

**图4：不使用归一化训练的分类器的嵌入和归一化嵌入。以及不使用归一化训练时特征范数与梯度幅度之间的关系。**
(a) 嵌入
(b) $\ell_2$ 归一化嵌入
(c) 对困难样本
(d) 对简单样本

我们通过实验发现，不使用学习尺度[^3]的批归一化（Batch Normalization）[8]（记为 $\widehat{\text{BN}}(\cdot)$ ）比 $\ell_2$ 归一化效果略好。我们建议将批归一化嵌入定义为：

$$
\hat{f}_{\text{BN}} = \frac{\widehat{\text{BN}}(f)}{\sqrt{k}} \qquad (7)
$$

其中 $k$ 是 $f$ 的维度数。批归一化试图使嵌入的每个维度具有零均值和单位方差。因此，批归一化之后，嵌入的范数大约为 $\sqrt{k}$ ，归一化特征 $\hat{f}_{\text{BN}}$ 的范数接近1，这与 $\ell_2$ 归一化类似。批归一化可能优于 $\ell_2$ 归一化的原因在于，在细粒度识别问题中，来自不同类别的许多嵌入可能非常相似。批归一化去除均值并重新缩放嵌入，从而创造了更多方差。对于分类器权重， $\ell_2$ 归一化总是能带来理想的结果。

### 3.4 与现成分类器的比较

将所提出方法的嵌入与现成分类器（使用未归一化的特征和权重， $\alpha = 1$ 进行训练）的嵌入进行比较是很有意义的（参见图4(a)中MNIST的结果）。正如其他工作中观察到的那样[28,19]：(i) 嵌入的幅度可能非常大；(ii) 嵌入不够"紧致"。即使对特征进行 $\ell_2$ 归一化（图4(b)），嵌入仍不如使用归一化和适当 $\alpha$ 训练的特征那样"紧致"（图3(b)）。

嵌入的范数倾向于变大，因为具有较大范数的"简单样本"会产生较小的损失[28]。为理解为何同一类别的嵌入不够紧致，检查梯度也是关键。在公式(3)中设置 $\alpha = 1$ ，当 $\|f\|_2 \to +\infty$ 时，对于"简单"样本，梯度幅度将趋近于0。而对于"困难"样本，幅度将趋近于一个常数，通常不为0。与图2(a)和2(b)类似，一些"困难"和"简单"样本的梯度幅度与特征范数之间的关系如图4(c)和4(d)所示。与使用大 $\alpha$ 训练的边界特征类似，具有大范数的边界特征无法获得足够的更新，从而导致嵌入不够紧致。

在训练过程中对特征使用 $\ell_2$ 归一化，与在测试阶段简单地对最终特征应用归一化是不同的。如公式(5)和(6)所示， $\ell_2$ 归一化会在训练过程中改变每个样本的梯度。

[^1]: $K$ 不能为1，否则 $x$ 是"简单"样本。
[^2]: 例如，两项具有完全相同的幅度但方向相反。
[^3]: 在本文中，批归一化始终指不使用学习尺度的批归一化。

---


## 4 度量学习实验

我们在以下细粒度数据集上进行了实验，使用Movshovitz-Attias等人[18]的训练/测试划分。在所有数据集中，训练集和测试集中的类别不重叠。

* **Cars (Car196)** [10]：一个细粒度汽车类别数据集，包含16,185张图片，共196种车型。前98类中的8,054张图片用于训练，其余98类中的8,131张图片用于测试。
* **Caltech-UCSD Birds-200-2011 (CUB200)** [30]：一个细粒度鸟类类别数据集，包含11,788张图片，共200种鸟类。前100种中的5,864张图片用于训练，其余100种中的5,924张图片用于测试。
* **Stanford Online Product (Product)** 数据集[24]：包含120,053张图片，共22,634个产品类别。11,318类中的59,551张图片用于训练，其余11,316类中的60,502张图片用于测试。
* **In-shop Clothes Retrieval (Fashion)** 数据集[16]：包含54,642张图片，共11,735个细粒度服装类别。数据集分为3个子集。7,982类中的52,712张图片用于训练。其余3,985类中的28,760张图片用于测试，分为一个gallery集（12,612张图片）和一个query集（14,218张图片）。

### 4.1 实现细节

我们使用TensorFlow深度学习框架[1]实现所提出的方法。为公平比较，我们严格遵循Movshovitz-Attias等人[18]的细节。使用来自TensorFlow slim的GoogLeNet V1[26]作为基础网络。网络使用ILSVRC 2012-CLS数据[21]进行预训练。所有输入图像均调整为 $256 \times 256$ 。在训练时，调整后的图像被随机裁剪为 $224 \times 224$ ，并进行随机水平翻转。在测试阶段，我们仅使用单个中心裁剪，与Movshovitz-Attias等人[18]一致。对于Car196、CUB200和Fashion数据集，使用带动量0.9的SGD优化器对网络进行微调。学习率设为0.004。对于Product数据集，使用ADAM优化器，学习率为0.01。嵌入大小设为64，批次大小设为32。我们为所有数据集选择 $\alpha = 16$ 作为"中等"温度，它对不同的嵌入大小效果良好（见第4.4节）。对于"加热"过程， $\alpha$ 从16减小（温度相应升高）到4，学习率降低到原始学习率的1/10。训练过程通常在50个训练周期内收敛，这与最快的最先进方法ProxyNCA[18]相似。我们的实现可在 https://github.com/ColumbiaDVMM/Heated_Up_Softmax_Embedding 获取。

**表1：Car196数据集的NMI和Recall(%)**

| 指标 | [22] | [24] | [25] | [11] | [5] | [18] | SM | LN | BN | HLN | HBN |
|------|------|------|------|------|-----|------|-----|-----|-----|-----|-----|
| NMI | 53.35 | 56.88 | 54.44 | 61.12 | 59.50 | 64.90 | 59.52 | 62.40 | 65.81 | 66.87 | 68.10 |
| R@1 | 51.54 | 52.98 | 58.11 | 67.54 | 64.65 | 73.22 | 60.76 | 68.59 | 71.12 | 71.93 | 74.70 |
| R@2 | 63.78 | 66.70 | 70.64 | 77.77 | 76.20 | 82.42 | 73.58 | 78.55 | 80.62 | 81.68 | 83.90 |
| R@4 | 73.52 | 76.01 | 80.27 | 85.74 | 84.23 | 86.36 | 82.50 | 86.18 | 87.82 | 88.34 | 89.77 |

**表2：CUB200数据集的NMI和Recall(%)**

| 指标 | [22] | [24] | [25] | [11] | [5] | [18] | SM | LN | BN | HLN | HBN |
|------|------|------|------|------|-----|------|-----|-----|-----|-----|-----|
| NMI | 55.38 | 56.50 | 59.23 | 56.87 | 59.90 | 59.53 | 57.19 | 59.23 | 59.20 | 60.34 | 60.75 |
| R@1 | 42.59 | 43.57 | 48.18 | 50.08 | 49.78 | 49.21 | 44.02 | 46.86 | 47.27 | 49.68 | 50.68 |
| R@2 | 55.03 | 56.55 | 61.44 | 62.24 | 62.34 | 61.90 | 55.86 | 59.79 | 59.67 | 61.85 | 62.58 |
| R@4 | 66.44 | 68.59 | 71.83 | 73.38 | 74.05 | 67.90 | 68.18 | 71.56 | 71.89 | 73.08 | 73.82 |

**表3：Stanford Product数据集的NMI和Recall(%)**

| 指标 | [22] | [24] | [25] | [11] | [18] | SM | LN | BN | HLN | HBN |
|------|------|------|------|------|------|-----|-----|-----|-----|-----|
| NMI | 89.46 | 88.65 | 89.48 | 88.70 | 90.60 | 88.66 | 90.11 | 90.45 | 90.39 | 90.61 |
| R@1 | 66.67 | 62.46 | 67.02 | 64.52 | 73.70 | 63.94 | 69.51 | 71.19 | 70.36 | 72.04 |
| R@10 | 82.39 | 80.81 | 83.65 | 82.53 | N/A | 80.07 | 84.69 | 85.89 | 85.41 | 86.25 |
| R@100 | 91.85 | 91.93 | 93.23 | 92.35 | N/A | 90.28 | 92.97 | 93.75 | 93.70 | 93.80 |

**表4：In-shop Clothes Retrieval数据集的Recall(%)**

| 指标 | [16] | [34] | SM | LN | BN | HLN | HBN |
|------|------|------|-----|-----|-----|-----|-----|
| R@1 | 53.0 | 62.1 | 78.6 | 79.6 | 80.7 | 80.5 | 81.1 |
| R@10 | 73.0 | 84.9 | 93.7 | 94.2 | 94.4 | 94.2 | 94.2 |
| R@20 | 76.0 | 91.2 | 95.4 | 96.0 | 96.1 | 96.1 | 95.9 |
| R@30 | 77.0 | 92.3 | 96.3 | 96.8 | 96.9 | 96.7 | 96.9 |

### 4.2 评估

遵循现有度量学习方法，我们评估测试集图像的聚类质量和检索性能。参照Song等人[25]的做法，所有特征在计算评估指标之前均进行 $\ell_2$ 归一化。归一化特征的表现略优于未归一化特征。

对于聚类，在测试样本的所有嵌入上运行K-Means算法。聚类数选为测试集中的类别数。每个测试样本根据其所属的聚类被分配一个聚类索引。使用聚类索引与真实标签之间的归一化互信息（NMI）[23]作为聚类指标。注意NMI对标签排列具有不变性。

对于检索，性能通过Recall@K进行评估，这也是该问题中广泛使用的指标。给定一个来自测试集的query样本，从测试集（或Fashion数据集的gallery集）的其余样本中检索出距离最小的K个样本。如果任何检索到的样本与query样本属于同一类别，则该样本的召回率设为1，否则为0。报告的Recall@K是整个测试集上的平均召回率。

我们训练一个使用softmax函数和交叉熵的分类器作为基线（SM）。对于基线分类器，在训练中，特征和权重未归一化， $\alpha$ 设为1。使用所提出的方法训练的4种不同版本的分类器用于评估：

* **LN**：使用 $\ell_2$ 归一化嵌入、 $\ell_2$ 归一化权重和 $\alpha = 16$ 的softmax。
* **BN**：使用批归一化嵌入、 $\ell_2$ 归一化权重和 $\alpha = 16$ 的softmax。
* **HLN**：使用 $\alpha = 4$ 微调LN的加热模型。
* **HBN**：使用 $\alpha = 4$ 微调BN的加热模型。

我们还将所提出的方法与多种最先进的度量学习方法进行了比较。现有文献使用了不同的基础网络和不同的评估协议。为公平比较，仅列出使用GoogLeNetV1作为基础网络且使用欧氏距离作为最终评估指标的方法：[22]（带半难例负样本挖掘的三元组学习[22]），[24]（提升结构化损失[24]），[25]（可学习结构化聚类[25]），[11]（无谱学习的深度聚类学习[11]），[5]（带智能挖掘的深度度量学习[5]）和[18]（ProxyNCA[18]）。对于Fashion数据集，我们与[16]（FashionNet[16]）和[34]（Hard-Aware Deeply Cascaded Embedding[34]）进行了比较。

### 4.3 度量学习

所有方法在所有4个数据集上的性能分别列于表1、2、3和4中。Softmax基线已经显示出与许多其他基于三元组损失的方法相当的结果。使用 $\ell_2$ 归一化或批归一化训练的嵌入提高了softmax基线的性能。由于在测试阶段，所有特征在计算指标之前都经过了 $\ell_2$ 归一化，性能提升并非来自对最终特征进行简单的 $\ell_2$ 归一化。批归一化比 $\ell_2$ 归一化效果略好。"加热"模型（HLN和HBN）在几乎所有指标上都显示出比固定温度训练嵌入更好的性能。

### 4.4 嵌入大小与 $\alpha$

我们进一步研究了不同嵌入大小和 $\alpha$ 值如何影响检索性能。嵌入大小选自 $\{64, 128, 256\}$ ， $\alpha$ 值选自 $\{4.0, 8.0, 16.0, 32.0, 64.0\}$ 。不同嵌入大小和 $\alpha$ 值下测试集上的R@1指标报告于表5中。还给出了通过未归一化的softmax函数和"加热"模型学习的特征的性能。"加热"模型在所有情况下均以显著优势优于所有其他模型。在固定 $\alpha$ 值训练的模型中， $\alpha = 16$ 训练的模型表现最优。

**表5：Car196上不同 $\alpha$ 值和嵌入大小下的R@1(%)**

| #DIM | SM | $\alpha=4$ | $\alpha=8$ | $\alpha=16$ | $\alpha=32$ | $\alpha=64$ | HBN (16 $\to$ 4) |
|------|-----|------|------|-------|------|------|-------------|
| 64 | 60.8 | 67.4 | 68.7 | 71.1 | 69.5 | 62.5 | 74.0 |
| 128 | 65.2 | 71.6 | 71.0 | 74.2 | 73.0 | 66.6 | 77.5 |
| 256 | 67.3 | 72.2 | 69.7 | 78.0 | 75.2 | 70.1 | 80.1 |

---


## 5 讨论

我们讨论了softmax函数中的温度参数如何影响深度分类模型倒数第二层嵌入的分布。使用中等温度进行训练会产生类内紧致、类间"分散"的嵌入，这对聚类和检索都有益处。我们还提出了一种"加热"方法，通过使用更高温度进行微调，进一步改善了嵌入的聚类和检索性能。我们基于分类器的方法在度量学习问题上取得了良好的性能，其训练过程比最先进的方法**更简单、更高效**。


## 致谢

本文基于美国空军研究实验室（AFRL）和国防高级研究计划局（DARPA）根据合同号FA8750-16-C-0166支持的工作。本文所表达的任何观点、发现和结论或建议均为作者个人责任，并不一定代表AFRL、DARPA或美国政府的官方立场。

---


## 参考文献

[1] Martín Abadi, Paul Barham, Jianmin Chen, Zhifeng Chen, Andy Davis, Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Geoffrey Irving, Michael Isard, et al. Tensorflow: A System for Large-Scale Machine Learning. OSDI, 2016.

[2] Artem Babenko and Victor Lempitsky. Aggregating Deep Convolutional Features for Image Retrieval. In ICCV 2015, 2015.

[3] S. Chopra, R. Hadsell, and Y. LeCun. Learning a Similarity Metric Discriminatively, with Application to Face Verification. In CVPR 2005, 2005.

[4] Matthieu Guillaumin, Jakob Verbeek, and Cordelia Schmid. Is that you? metric learning approaches for face identification. In CVPR 2009, 2009.

[5] Ben Harwood, Vijay Kumar B G, Gustavo Carneiro, Ian Reid, and Tom Drummond. Smart Mining for Deep Metric Learning. In ICCV 2017, 2017.

[6] Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the Knowledge in a Neural Network. In NIPS Workshop 2015, 2015.

**[7] Elad Hoffer and Nir Ailon. Deep Metric Learning Using Triplet Network. In International Workshop on Similarity-Based Pattern Recognition, 2015. Springer, 2015.**

[8] Sergey Ioffe and Christian Szegedy. Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. In ICML 2015, 2015.

[9] Martin Koestinger, Martin Hirzer, Paul Wohlhart, Peter M Roth, and Horst Bischof. Large scale metric learning from equivalence constraints. In CVPR 2012, 2012.

[10] Jonathan Krause, Michael Stark, Jia Deng, and Li Fei-Fei. 3D Object Representations for Fine-Grained Categorization. In 4th International IEEE Workshop on 3D Representation and Recognition, 2013, 2013.

[11] Marc T. Law, Raquel Urtasun, and Richard S. Zemel. Deep Spectral Clustering Learning. In ICML 2017, 2017.

[12] Yann LeCun et al. Lenet-5, Convolutional Neural Networks. url: http://yann.lecun.com/exdb/lenet, 2015.

[13] Jung-Eun Lee, Rong Jin, and Anil K Jain. Rank-based distance metric learning: An application to image retrieval. In CVPR 2008, 2008.

[14] Giuseppe Lisanti, Svebor Karaman, and Iacopo Masi. Multichannel-kernel canonical correlation analysis for cross-view person reidentification. ACM TOMM, 2017.

[15] Weiyang Liu, Yandong Wen, Zhiding Yu, Ming Li, Bhiksha Raj, and Le Song. SphereFace: Deep Hypersphere Embedding for Face Recognition. In CVPR 2017, 2017.

[16] Ziwei Liu, Ping Luo, Shi Qiu, Xiaogang Wang, and Xiaoou Tang. Deepfashion: Powering Robust Clothes Recognition and Retrieval with Rich Annotations. In CVPR 2016, 2016.

**[17] Anastasiya Mishchuk, Dmytro Mishkin, Filip Radenovic, and Jiri Matas. Working Hard to Know Your Neighbor's Margins: Local Descriptor Learning Loss. In NIPS 2017, 2017.**

[18] Yair Movshovitz-Attias, Alexander Toshev, Thomas K. Leung, Sergey Ioffe, and Saurabh Singh. No Fuss Distance Metric Learning Using Proxies. In ICCV 2017, 2017.

[19] Rajeev Ranjan, Carlos D. Castillo, and Rama Chellappa. L2-constrained Softmax Loss for Discriminative Face Verification. arXiv:1703.09507 [cs], 2017.

[20] Ali Sharif Razavian, Hossein Azizpour, Josephine Sullivan, and Stefan Carlsson. Cnn features off-the-Shelf: an Astounding Baseline for Recognition. In CVPRW 2014, 2014.

[21] Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg, and Li Fei-Fei. ImageNet Large Scale Visual Recognition Challenge. IJCV, 2015.

**[22] Florian Schroff, Dmitry Kalenichenko, and James Philbin. FaceNet: A Unified Embedding for Face Recognition and Clustering. In CVPR 2015, 2015.**

[23] Hinrich Schütze, Christopher D Manning, and Prabhakar Raghavan. Introduction to Information Retrieval. Cambridge University Press, 39, 2008.

[24] Hyun Oh Song, Yu Xiang, Stefanie Jegelka, and Silvio Savarese. Deep Metric Learning via Lifted Structured Feature Embedding. In CVPR 2016, 2016.

[25] Hyun Oh Song, Stefanie Jegelka, Vivek Rathod, and Kevin Murphy. Deep Metric Learning via Facility Location. In CVPR 2017, 2017.

[26] Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich, et al. Going Deeper With Convolutions. In CVPR 2015, 2015.

[27] Evgeniya Ustinova and Victor Lempitsky. Learning Deep Embeddings with Histogram Loss. In NIPS 2016, 2016.

[28] Feng Wang, Xiang Xiang, Jian Cheng, and Alan L. Yuille. NormFace: L2 Hypersphere Embedding for Face Verification. In ACM MM 2017, 2017a.

[29] Jian Wang, Feng Zhou, Shilei Wen, Xiao Liu, and Yuanqing Lin. Deep Metric Learning with Angular Loss. In ICCV 2017, 2017b.

[30] P. Welinder, S. Branson, T. Mita, C. Wah, F. Schroff, S. Belongie, and P. Perona. Caltech-UCSD Birds 200. Technical report, 2010.

[31] Chao-Yuan Wu, R. Manmatha, Alexander J. Smola, and Philipp Krähenbühl. Sampling Matters in Deep Embedding Learning. In ICCV 2017, 2017.

[32] Eric P Xing, Michael I Jordan, Stuart J Russell, and Andrew Y Ng. Distance metric learning with application to clustering with side-information. In NIPS 2003, 2003.

[33] Jieping Ye, Zheng Zhao, and Huan Liu. Adaptive distance metric learning for clustering. In CVPR 2007, 2016.

[34] Yuhui Yuan, Kuiyuan Yang, and Chao Zhang. Hard-Aware Deeply Cascaded Embedding. In ICCV 2017, 2017.

[35] Xu Zhang, Felix X. Yu, Sanjiv Kumar, and Shih-Fu Chang. Learning Spread-Out Local Feature Descriptors. In ICCV 2017, 2017.
