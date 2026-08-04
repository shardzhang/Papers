# A Generic Coordinate Descent Framework for Learning from Implicit Feedback

> Immanuel Bayer, Xiangnan He, Bhargav Kanagal, Steffen Rendle | University of Konstanz, NUS, Google


本文提出了一个通用坐标下降框架 iCD（implicit Coordinate Descent），用于从隐式反馈中高效学习推荐模型。核心内容：

- 识别 k-可分离属性，作为高效隐式CD学习的充分条件
- 为矩阵分解、因子分解机等主流模型推导高效隐式CD算法

关键发现：iCD 比传统CD在隐式数据集上快四个数量级。

---
- 基于特征的因子分解模型（iCD-FM）在冷启动、离线推荐和即时推荐场景中均优于基线方法

---


## 摘要

近年来，推荐研究的兴趣已从显式反馈转向隐式反馈数据。针对各种应用，研究者提出了多种复杂模型。尽管如此，从隐式反馈中学习仍然具有计算挑战性。到目前为止，大多数工作依赖于**易于推导但在实践中难以应用的随机梯度下降**（SGD）求解器，尤其对于包含大量item的任务。对于简单的矩阵因子分解模型，先前已提出高效的坐标下降（CD）求解器。然而，对于更复杂的模型，尚未推导出高效的CD方法。在本文中，我们提供了一个新的框架，用于为复杂推荐模型推导高效的CD算法。我们识别并引入了k-可分离模型的属性。我们证明k-可分离性是允许使用CD高效优化隐式推荐问题的**充分属性**。我们在多种最先进模型上展示了该框架的应用，包括因子分解机和Tucker分解。总之，我们的工作**提供了为复杂推荐模型推导高效隐式CD算法的理论**和构建模块。


## 1 引言

近年来，推荐系统研究的焦点已从评分预测等显式反馈问题转向隐式反馈问题。用户提供的关于其偏好的大部分信号是隐式的。隐式反馈的例子包括：用户观看视频、点击链接等。隐式反馈数据的获取成本远低于显式反馈，因为它不会给用户带来额外成本，因此可以在更大规模上获得。然而，从隐式反馈中学习推荐系统在计算上是昂贵的，因为**需要将用户的已观测行为与所有未观测行为进行对比**[5, 13]。

随机梯度下降 SGD（Stochastic Gradient Descent）和坐标下降 CD（Coordinate Descent）是两种广泛用于大规模机器学习的方法。这两种算法都被认为是利用隐式反馈学习矩阵因子分解模型的最先进方法，并已得到广泛研究。SGD和CD在各种数据集上表现出不同的优势和劣势[4, 17, 16, 8, 25, 15, 22, 26]。虽然SGD是一个优化广泛模型类别的通用框架[13]，但CD仅适用于少数简单模型[5, 10]。事实上，甚至不知道CD是否可以用于高效优化复杂的推荐模型。我们的工作填补了这一空白，识别出一种称为k-可分离性的模型属性，它是允许从隐式反馈中进行高效学习的充分条件。基于k-可分离性，我们提供了一个推导高效隐式CD求解器的通用框架。

本文组织如下：首先，我们介绍从隐式反馈中学习的问题，并展示隐式训练样本的数量使得标准算法的应用具有挑战性。接下来，我们提供使用CD进行高效隐式学习的通用框架。我们识别出模型的k-可分离性是使高效学习可行的充分条件，并引入了iCD，一种针对k-可分离模型的通用学习算法。在第5节中，我们展示了如何将iCD应用于多种不同的模型，包括矩阵因子分解 MF（Matrix Factorization）、因子分解机 FM（Factorization Machines）和张量因子分解。该节既是流行模型的解决方案，也是将该框架应用于其他复杂推荐模型的指南。

总结而言，我们的贡献如下：

- 我们识别出**推荐模型的一个基本属性**，该属性允许从隐式数据中进行高效的CD学习。
- 我们提供iCD，一个推导高效隐式CD算法的框架。
- 我们应用该框架并推导了MF、带辅助信息的MF、FM、PARAFAC和Tucker分解的算法。


## 2 相关工作

多年来，矩阵因子分解（MF）被认为是最有效的基础推荐系统模型。两种优化策略主导了基于隐式反馈数据的MF研究。第一种是贝叶斯个性化排序 BPR（Bayesian Personalized Ranking）[13]，这是一个随机梯度下降（SGD）框架，对比已消费和未消费的item对。第二种是坐标下降（CD），也称为**交替最小二乘法**，在已消费和未消费item上的逐元素损失上进行优化[5]。

在损失公式方面，BPR的成对分类损失更适合排序，而CD损失更适合数值数据。在优化任务方面，两种技术面临相同的挑战：在大量训练样本上进行学习。BPR通过采样负item来解决这个问题，但已有研究表明，当item数量很大时，BPR存在收敛问题[7, 12]。它需要更复杂的非均匀采样策略来处理这个问题[12, 6]。另一方面，对于CD-MF，Hu等人[5]推导了一个高效算法，允许在没有任何成本的情况下对大量未消费item进行优化。这种计算技巧是精确的，不涉及采样。许多作者在多种数据集上比较了CD-MF和BPR-MF，一些研究报告BPR-MF的质量更好[4, 17, 16, 8]，而另一些研究中CD-MF效果更好[8, 25, 15, 22, 26]。这些大量的结果表明CD和BPR的优势是正交的，两种方法各有优点。

到目前为止，我们的讨论集中在从隐式数据学习矩阵因子分解模型。从简单的矩阵因子分解转向更复杂的因子分解模型在许多隐式推荐问题中取得了巨大成功[2, 4, 18, 1, 9, 24]。然而，关于复杂因子分解模型的工作几乎完全依赖于使用通用BPR框架的SGD优化。我们的工作为推导此类复杂模型的CD学习器提供了理论和实践框架。与MF的CD类似，我们的通用算法能够在所有未消费item上进行优化，而无需显式遍历它们。总之，我们的论文使研究人员和从业者能够在他们的工作中应用CD，并让他们在BPR和CD的优势之间进行选择。


## 3 问题定义

设 $I$ 为item集合， $C$ 为上下文集合。设 $S$ 为已观测反馈的集合，其中元组 $(c,i,y,\alpha)\in S$ 表示在上下文 $c$ 中，分数 $y$ 以置信度 $\alpha$ 被赋予item $i$ 。有关示意图，请参见图1。我们使用上下文的通用符号，它可以包括例如用户、时间、位置、属性、历史等。第5节和第6节展示了更多上下文的示例。

**表1：数据格式**
- 左侧（显式）：评级数据， $S = \{(c_1,i_1,2),(c_1,i_3,3),(c_2,i_3,4),\ldots\}$
- 右侧（隐式）：例如观看/购买/点击次数， $|S_{\text{impl}}| = |C||I|$

### 3.1 推荐模型

推荐模型 $\hat{y}: C \times I \to \mathbb{R}$ 是一个为每个上下文-item对分配分数的函数。模型 $\hat{y}$ 由一组模型参数 $\Theta$ 参数化。 $\hat{y}$ 通常用于决定在给定上下文中展示哪些item。

学习任务是在数据 $S$ 上找到最小化损失函数的模型参数值，例如平方损失

$$
L(\Theta|S) = \sum_{(c,i,y,\alpha)\in S} \alpha (\hat{y}(c,i) - y)^2 + \sum_{\theta\in\Theta} \lambda_\theta \theta^2 \qquad (1)
$$

其中 $\lambda_\theta$ 是参数 $\theta$ 的正则化常数。

### 3.2 坐标下降算法

目标函数(1)可以通过坐标下降（CD）最小化。CD遍历模型参数，每次更新一个参数。对于选定的参数 $\theta\in\Theta$ ，CD计算 $L$ 相对于选定坐标 $\theta$ 的一阶导数 $L'$ 和二阶导数 $L''$ ：

$$
L'(\theta|S) = 2 \sum_{(c,i,y,\alpha)\in S} \alpha (\hat{y}(c,i) - y)\hat{y}'(c,i) + 2 \lambda_\theta \theta \qquad (2)
$$

$$
L''(\theta|S) = 2 \sum_{(c,i,y,\alpha)\in S} \alpha [(\hat{y}(c,i) - y)\hat{y}''(c,i) + \hat{y}'(c,i)^2] + 2 \lambda_\theta \qquad (3)
$$

并执行牛顿更新步骤：

$$
\theta \leftarrow \theta - \eta \frac{L'(\theta|S)}{L''(\theta|S)} \qquad (4)
$$

其中 $\eta\in(0,1]$ 是步长。对于多线性模型，可以选择全步长 $\eta=1$ 而不会导致发散风险[11]。第5节中的所有模型都属于这一类别。

这类CD算法已被深入研究，其运行时复杂度通常与训练样本数和嵌入维度成线性关系。对于MF，[23]证明了复杂度为 $O(|S|k)$ ，对于FM，[11]推导出复杂度为 $O(NZ(X)k)$ ，其中 $NZ(X)$ 是设计矩阵 $X$ 中非零项的数量。训练样本数量的线性运行时复杂度使得这些算法非常适合显式推荐场景，然而对于隐式问题，它们变得不可行。

### 3.3 从隐式反馈中学习

在隐式推荐问题中，未消费的item是有意义的，不能忽略。例如，在图1（右侧）中，数据描述了每个item在某个上下文中被消费的次数。未消费的item，即计数为零的item，对于学习用户偏好是有用的。为了形式化，隐式问题的训练数据 $S_{\text{impl}}$ 由已观测反馈集合 $S^+$ 和所有未消费元组 $S^0$ 组成

$$
S_{\text{impl}} = S^+ \cup S^0,\quad |S_{\text{impl}}| = |C||I| \qquad (5)
$$

对 $\forall(c,i,y,\alpha)\in S^0: y=0, \alpha=\alpha_0 \qquad (6)$

 $S^+$ 包含已观测反馈，其规模远小于 $S_{\text{impl}}$ ，通常 $|S^+| \ll |C||I|$ 。

隐式学习问题可以表述为在隐式数据 $S_{\text{impl}}$ 上最小化公式(1)中的目标函数。虽然在理论上是可能的，但在实践中，将第3.2节的学习算法应用于这个问题是不可行的，因为它们的计算运行时与训练数据的规模（对于隐式问题， $|S_{\text{impl}}| = |C||I|$ ）成线性关系。本文展示了如何推导高效的CD算法以在隐式数据上优化公式(1)。

## 4 面向隐式反馈的通用坐标下降算法

### 4.1 隐式正则化器

如第3.3节所述，在隐式数据上训练具有挑战性的原因是隐式样本 $S^0$ 数量巨大，通常 $|S^0| \in O(|C||I|)$ 。注意 $S^0$ 包括所有不在 $S^+$ 中的上下文-item对。我们现在展示可以将优化准则重新表述为对所有上下文-item对求和。这种重新表述是后续在第4.2节中分解损失的前提条件。此外，它允许在不考虑 $S^+$ 的情况下研究隐式优化。

**引理1.** 隐式学习可以重新表述为在小规模正样本集上学习与最小化任何上下文-item对的评分函数的组合。

$$
\arg\min_\Theta L(\Theta|S_{\text{impl}}) = \arg\min_\Theta L(\Theta|\tilde{S}) + \alpha_0 \left(\sum_{c\in C}\sum_{i\in I} \hat{y}(c,i)^2\right) \qquad (7)
$$

其中已观测反馈被重新缩放

$$
\tilde{S} := \left\{ \left(c,i,\frac{\alpha}{\alpha-\alpha_0}y,\alpha-\alpha_0\right) : (c,i,y,\alpha)\in S^+ \right\} \qquad (8)
$$

**证明.** 根据损失（公式1）和隐式训练集 $S_{\text{impl}}$ （公式5）的定义，

\
$$
L(\Theta|S_{\text{impl}}) = L(\Theta|S^+) + \alpha_0 \sum_{(c,i)\in S^0} \hat{y}(c,i)^2
\
$$
\
$$
= L(\Theta|S^+) - L(\Theta|\{(c,i,0,\alpha_0):(c,i,y,\alpha)\in S^+\}) + \alpha_0 R(\Theta)
\
$$
\
$$
= L(\Theta|S^+) + \alpha_0 R(\Theta) - L(\Theta|\{(c,i,0,\alpha_0):(c,i,y,\alpha)\in S^+\})
\
$$

我们可以进一步将每对样本合并为一个。我们以 $(c,i,y,\alpha)\in S$ 及其对应项 $(c,i,0,-\alpha_0)$ 为例展示。

\
$$
L(\Theta|\{(c,i,y,\alpha)\}) + L(\Theta|\{(c,i,0,-\alpha_0)\})
\
$$
\
$$
= \alpha (\hat{y}(c,i) - y)^2 - \alpha_0 \hat{y}(c,i)^2
\
$$
\
$$
= (\alpha-\alpha_0) \left( \hat{y}(c,i)^2 - 2\frac{\alpha}{\alpha-\alpha_0}y\hat{y}(c,i) + \frac{\alpha}{\alpha-\alpha_0}y^2 \right)
\
$$
\
$$
= (\alpha-\alpha_0) \left( \hat{y}(c,i) - \frac{\alpha}{\alpha-\alpha_0}y \right)^2 + \text{const}
\
$$
\
$$
= L\left( \Theta \left| \left\{ \left(c,i,\frac{\alpha}{\alpha-\alpha_0}y,\alpha-\alpha_0\right) \right\} \right. \right) + \text{const}
\
$$

附加常数不改变 $\Theta$ 的最优值，因此如公式(8)所示的样本重新缩放保留了最优值。

该引理为隐式学习提供了有趣的解释。隐式问题可以被视为显式学习任务，或带有额外隐式正则化器（也称为预测零的偏置） $R(\Theta)$ 的单类问题。与 $L_2$ 等常见正则化器相比，隐式正则化器能感知模型 $\hat{y}$ 。 $L_2$ 惩罚非零模型参数 $\Theta$ ，而隐式正则化器惩罚非零预测 $\hat{y}$ 。因此，隐式正则化器比 $L_2$ 限制更小，因为即使使用大的模型参数也能实现小的预测值。

### 4.2 面向k-可分离模型的iCD算法

如公式(7)所示，隐式学习可以表述为在小规模集合 $\tilde{S}$ 上的显式学习加上一个计算昂贵的隐式正则化器 $R$ 。在显式损失上学习模型已有充分研究[23, 11]，因此我们现在关注隐式正则化器

$$
R(\Theta) = \sum_{c\in C}\sum_{i\in I} \hat{y}(c,i)^2 \qquad (9)
$$

其通用计算复杂度为 $O(|C||I|)$ 。

在本节中，我们介绍k-可分离模型的概念。我们将为任何k-可分离模型提供一个高效的隐式CD求解器。在第5节中，我们展示了多种常见模型是k-可分离的，包括矩阵因子分解、基于特征的方法如因子分解机，以及高阶张量因子分解如PARAFAC或Tucker分解。本节推导的iCD框架不仅限于上述模型，也可以作为其他k-可分离模型的蓝图。

**定义1（k-可分离）.** 如果模型 $\hat{y}(c,i)$ 可以重写为

$$
\hat{y}(c,i) = \langle\phi(c),\psi(i)\rangle = \sum_{f=1}^k \phi_f(c)\psi_f(i) \qquad (10)
$$

其中函数

$$
\phi: C\to\mathbb{R}^k,\quad \psi: I\to\mathbb{R}^k \qquad (11)
$$

且 $\phi$ 由 $\Theta_C$ 参数化， $\psi$ 由 $\Theta_I$ 参数化，并有 $\Theta_C\cap\Theta_I=\varnothing$ ，则称模型 $\hat{y}(c,i)$ 是k-可分离的。

**引理2.** 任何k-可分离模型的隐式正则化器可以分解为：

$$
R(\Theta) = \sum_{f=1}^k\sum_{f'=1}^k \underbrace{\left(\sum_{c\in C} \phi_f(c)\phi_{f'}(c)\right)}_{=:J_C(f,f^{\prime})} \underbrace{\left(\sum_{i\in I} \psi_f(i)\psi_{f'}(i)\right)}_{=:J_I(f,f^{\prime})} \qquad (12)
$$

**证明.** 该引理通过将k-可分离模型（公式10）代入隐式正则化器（公式9）并重排求和项得到。

\
$$
R(\Theta) = \sum_{c\in C}\sum_{i\in I} \left(\sum_{f=1}^k \phi_f(c)\psi_f(i)\right) \left(\sum_{f'=1}^k \phi_{f'}(c)\psi_{f'}(i)\right)
\
$$
\
$$
= \sum_{f=1}^k\sum_{f'=1}^k \left(\sum_{c\in C} \phi_f(c)\phi_{f'}(c)\right) \left(\sum_{i\in I} \psi_f(i)\psi_{f'}(i)\right)
\
$$

该引理是高效隐式学习算法的关键。它表明上下文侧和item侧可以独立计算，从而将计算复杂度从 $O(|C||I|)$ 降至 $O((|C|+|I|)k^2)$ 。接下来，我们展示如何将其用于CD更新步骤（见公式4）所需的梯度计算。

**引理3.** 任何k-可分离模型的隐式正则化器相对于任何模型参数 $\theta\in\Theta_C$ （或类比地 $\theta\in\Theta_I$ ）的梯度可以简化为：

$$
R'(\theta) = 2\sum_{f=1}^k\sum_{f'=1}^k J_I(f,f^{\prime}) \sum_{c\in C} \phi_f(c)\phi'_{f'}(c) \qquad (13)
$$

$$
R''(\theta) = 2\sum_{f=1}^k\sum_{f'=1}^k J_I(f,f^{\prime}) \sum_{c\in C} \left[\phi_f(c)\phi''_{f'}(c) + \phi'_f(c)\phi'_{f'}(c)\right] \qquad (14)
$$

**证明.** 该引理通过对公式(12)求导得到。

该引理表明，任何上下文参数 $R'$ 和 $R''$ 的计算与 $|I|$ 无关。

由上述分析可以得到为模型 $\hat{y}$ 推导高效iCD学习算法的步骤。首先，将模型重写为 $\phi$ 和 $\psi$ 的点积。其次，构造 $\phi$ 和 $\psi$ 相对于任何模型参数 $\theta\in\Theta$ 的一阶和二阶导数。这些结果允许对任何模型参数 $\theta\in\Theta$ 高效计算 $R'(\theta)$ 和 $R''(\theta)$ 。有了这个昂贵的隐式正则化器的梯度，就可以应用牛顿步。算法1展示了利用本节思想的通用iCD算法。

---

**算法1 通用隐式CD（iCD）**

1: **procedure** iCD-Generic( $S, C, I$ )
2: &nbsp;&nbsp; $\Theta \leftarrow \mathcal{N}(0, \sigma)$
3: &nbsp;&nbsp; **repeat**
4: &nbsp;&nbsp;&nbsp;&nbsp; 必要时计算 $\Phi$ 和 $\Psi$
5: &nbsp;&nbsp;&nbsp;&nbsp; 计算 $J_I$
6: &nbsp;&nbsp;&nbsp;&nbsp; **for** $\theta\in\Theta_C$ **do**
7: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 计算 $L'(\theta|S), L''(\theta|S)$
8: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 计算 $R'(\theta), R''(\theta)$
9: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $\theta \leftarrow \theta - \eta \frac{L'(\theta|S)+\alpha_0 R'(\theta)}{L''(\theta|S)+\alpha_0 R''(\theta)}$
10: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 必要时更新 $\Phi$
11: &nbsp;&nbsp;&nbsp;&nbsp; **end for**
12: &nbsp;&nbsp;&nbsp;&nbsp; 将步骤5-11应用于item侧
13: &nbsp;&nbsp; **until** 收敛
14: &nbsp;&nbsp; **return** $\Theta$
15: **end procedure**

---

大多数模型允许进一步优化：(i) 当 $\phi$ 或 $\psi$ 的梯度是稀疏的时，公式(13, 14)中的一些求和项可以省略。(ii) 模型参数通常具有一定的结构，可以用于更系统地遍历模型参数。我们将在下一节中针对多种模型展示这两个步骤。

## 5 应用

在本节中，我们将iCD应用于两类复杂的因子分解模型，即基于特征的因子分解模型和张量因子分解模型。我们选择这两类模型是因为它们非常强大且常用。此外，它们各自在推导iCD算法方面具有一些有趣的特性。所提供的算法可以直接应用于许多常见的推荐系统任务。本节也可作为一般性推导iCD算法的指南。

### 5.1 矩阵因子分解（MF）

我们首先将框架应用于矩阵因子分解（见图2）。对于MF，评分函数为

$$
\hat{y}(c,i) := \langle w_c, h_i \rangle = \sum_{f=1}^k w_{c,f} h_{i,f} \qquad (15)
$$

其中模型参数 $\Theta = \{W, H\}$ ， $W\in\mathbb{R}^{C\times k}$ ， $H\in\mathbb{R}^{I\times k}$ 。

**图2：矩阵因子分解：** 每个上下文 $c$ 关联一个嵌入 $w_c$ ，每个item $i$ 关联一个嵌入 $h_i$ 。学习模型参数 $W\in\mathbb{R}^{C\times k}, H\in\mathbb{R}^{I\times k}$ 以通过点积 $\langle w_c, h_i\rangle$ 近似数据 $S_{\text{impl}}$ 。

MF模型是平凡的k-可分离模型，其中

$$
\phi_f(c) = w_{c,f},\quad \psi_f(i) = h_{i,f} \qquad (16)
$$

此外，梯度是稀疏的

$$
\frac{\partial\phi_f(c)}{\partial w_{c^*,f^*}} = \begin{cases} 1, & \text{if } c=c^* \land f=f^* \\ 0, & \text{else} \end{cases} \qquad (17)
$$

且所有二阶导数为零。因此，正则化器导数简化为：

$$
R'(w_{c^*,f^*}) = 2\sum_{f=1}^k J_I(f,f^*) w_{c^*,f} \qquad (18)
$$
$$
R''(w_{c^*,f^*}) = 2 J_I(f^*,f^*) \qquad (19)
$$

item侧的推导是对称的。

由于MF将每个模型参数与一个嵌入维度 $f$ 关联，我们可以一次遍历一个维度的参数。由于MF是双线性的，可以取全步长 $\eta=1$ 。算法2展示了完整过程。

**算法2 面向MF的隐式CD（iCD-MF）**

1: **procedure** iCD-MF( $S, C, I$ )
2: &nbsp;&nbsp; $W, H \leftarrow \mathcal{N}(0, \sigma)$
3: &nbsp;&nbsp; **repeat**
4: &nbsp;&nbsp;&nbsp;&nbsp; **for** $f^*\in\{1,\ldots,k\}$ **do**
5: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **for** $f\in\{1,\ldots,k\}$ **do**
6: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 计算 $J_I(f^*,f)$
7: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **end for**
8: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **for** $c^*\in C$ **do**
9: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 计算 $L'(w_{c^*,f^*}|S), L''(w_{c^*,f^*}|S)$
10: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 计算 $R'(w_{c^*,f^*}), R''(w_{c^*,f^*})$
11: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $w_{c^*,f^*} \leftarrow w_{c^*,f^*} - \frac{L'(w_{c^*,f^*}|S)+\alpha R'(w_{c^*,f^*})}{L''(w_{c^*,f^*}|S)+\alpha R''(w_{c^*,f^*})}$
12: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **end for**
13: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 将步骤5-12应用于item侧
14: &nbsp;&nbsp;&nbsp;&nbsp; **end for**
15: &nbsp;&nbsp; **until** 收敛
16: &nbsp;&nbsp; **return** $W, H$
17: **end procedure**

 $J_I(f^*,\cdot)$ 的计算复杂度为 $O(|I|k)$ 。隐式正则化器的梯度计算每个参数为 $O(k)$ ，显式部分对所有参数为 $O(|S|)$ 。整体而言，算法每次迭代的复杂度为 $O((|I|+|C|)k^2 + |S|k)$ 。

### 5.2 基于特征的因子分解模型

MF最强大的扩展之一是对上下文和item进行基于特征的建模。基于特征的因子分解模型在表达能力上严格优于MF，并在许多应用中表现出显著改进（例如[2, 11]）。例如，冷启动问题通常通过用用户和item的属性替换或补充用户和item的ID来解决[2]。另一个例子是上下文感知推荐，其中上下文由多个变量表示，例如除用户ID外的位置或时间。序列模型也可以通过基于特征的建模来表示[6]。

目前，在隐式反馈上学习通用基于特征的模型仅限于BPR。本文是第一个为此重要模型类别提供隐式CD算法的工作。

为了形式化该问题，假设每个 $c\in C$ 由特征向量 $x_c\in\mathbb{R}^p$ 表示，每个 $i\in I$ 由特征向量 $z_i\in\mathbb{R}^p$ 表示。参见图3的示意图。

**图3：带辅助信息的矩阵因子分解：** 除了数据 $S_{\text{impl}}$ ，还为每个上下文 $c\in C$ 提供特征向量 $x_c\in\mathbb{R}^p$ ，为每个item $i\in I$ 提供特征向量 $z_i\in\mathbb{R}^p$ 。每个上下文特征 $l\in\{1,\ldots,p\}$ 被分配一个 $k$ 维嵌入向量 $w_l\in\mathbb{R}^k$ ，每个item特征类似地分配 $h_l\in\mathbb{R}^k$ 。学习模型参数 $W\in\mathbb{R}^{p\times k}, H\in\mathbb{R}^{p\times k}$ 以用 $XW(ZH)^t$ 近似数据 $S_{\text{impl}}$ 。

#### 5.2.1 带辅助信息的MF（MFSI）

我们从类似于[2]的基于特征的矩阵因子分解扩展开始：

$$
\hat{y}(c,i) = x_c W (z_i H)^t = \sum_{f=1}^k \left(\sum_{l=1}^p x_{c,l} w_{l,f}\right) \left(\sum_{l=1}^p z_{i,l} h_{l,f}\right) \qquad (20)
$$

其中 $\Theta = \{W, H\}$ 。MFSI是k-可分离的，其中

$$
\phi_f(c) = \sum_{l=1}^p x_{c,l} w_{l,f},\quad \psi_f(i) = \sum_{l=1}^p z_{i,l} h_{l,f} \qquad (21)
$$

且梯度是稀疏的

$$
\frac{\partial\phi_f(c)}{\partial w_{l^*,f^*}} = \begin{cases} x_{c,l^*}, & \text{if } f=f^* \\ 0, & \text{else} \end{cases} \qquad (22)
$$

由于 $\phi$ 和 $\psi$ 的稀疏梯度，正则化器导数简化为：

$$
R'(w_{l^*,f^*}) = 2\sum_{f=1}^k J_I(f,f^*) \sum_{c\in C} x_{c,l^*} \phi_f(c) \qquad (23)
$$
$$
R''(w_{l^*,f^*}) = 2 J_I(f^*,f^*) \sum_{c\in C} x_{c,l^*}^2 \qquad (24)
$$

注意，对上下文变量的求和仅依赖于 $x_{c,l^*} \neq 0$ 的上下文，因此使用稀疏迭代器，在给定嵌入层 $f^*$ 中优化所有上下文变量的计算复杂度为 $O(k\,NZ(X))$ 。

此计算假设 $\Phi$ 和 $\Psi$ 已给定。显然，优化 $W$ 时 $\Psi$ 不变，优化 $H$ 时 $\Phi$ 不变。然而，优化 $W$ 时 $\Phi$ 会变化，但可以通过以下方式与 $W$ 的变更保持同步：

$$
\phi_{f^*}(c) \leftarrow \phi_{f^*}(c) + x_{c,l^*} (w^{\text{new}}_{l^*,f^*} - w^{\text{old}}_{l^*,f^*}) \qquad (25)
$$

item侧可以类比推导。算法3对所有变量进行一次完整遍历（一个epoch）的总运行时为 $O(k^2(NZ(X)+NZ(Z)))$ 用于隐式正则化器。

**算法3 面向带辅助信息MF的隐式CD（iCD-MFSide）**

1: **procedure** iCD-MFSide( $S, C, I$ )
2: &nbsp;&nbsp; $W, H \leftarrow \mathcal{N}(0, \sigma)$
3: &nbsp;&nbsp; **repeat**
4: &nbsp;&nbsp;&nbsp;&nbsp; 计算 $\Phi$ 和 $\Psi$
5: &nbsp;&nbsp;&nbsp;&nbsp; **for** $f^*\in\{1,\ldots,k\}$ **do**
6: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **for** $f\in\{1,\ldots,k\}$ **do**
7: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 计算 $J_I(f^*,f)$
8: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **end for**
9: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **for** $l^*\in\{1,\ldots,p\}$ **do**
10: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 计算 $L'(w_{l^*,f^*}|S), L''(w_{l^*,f^*}|S)$
11: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 计算 $R'(w_{l^*,f^*}), R''(w_{l^*,f^*})$
12: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $w_{l^*,f^*} \leftarrow w_{l^*,f^*} - \frac{L'(w_{l^*,f^*}|S)+\alpha R'(w_{l^*,f^*})}{L''(w_{l^*,f^*}|S)+\alpha R''(w_{l^*,f^*})}$
13: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 更新 $\Phi$
14: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **end for**
15: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 将步骤6-14应用于item侧
16: &nbsp;&nbsp;&nbsp;&nbsp; **end for**
17: &nbsp;&nbsp; **until** 收敛
18: &nbsp;&nbsp; **return** $W, H$
19: **end procedure**

#### 5.2.2 因子分解机

因子分解机（FM）模型[11]是一种更复杂的因子分解模型，包含偏置项和所有变量之间的交互。一般而言，对于特征向量 $x\in\mathbb{R}^p$ ，FM定义为

$$
\hat{y}(x) = b + \sum_{l=1}^p x_l \tilde{w}_l + \sum_{l=1}^p\sum_{l'>l} \langle w_l, w_{l'} \rangle x_l x_{l'} \qquad (26)
$$

其中 $b$ 是全局偏置参数， $\tilde{w}$ 是特征偏置， $W$ 是嵌入。在我们的情况下，对于上下文-item对 $(c,i)$ ，我们将输入特征向量 $x$ 设置为上下文和item特征向量的连接： $x := (x_c, z_i)$ 。

**图4：张量因子分解数据示例：** 上下文侧有两个类别变量， $C \subseteq C_1 \times C_2$ 。该数据可以解释为一个具有缺失值的3模张量。

FM模型是 $(k+2)$ -可分离的，其中

$$
\phi_f(c) = \sum_{l=1}^p x_{c,l} w_{l,f},\quad \psi_f(i) = \sum_{l=1}^p z_{i,l} h_{l,f},\quad \text{for } f=1,\ldots,k \qquad (27)
$$

$$
\phi_{k+1}(c) = b + \sum_{l=1}^p x_{c,l} \tilde{w}_l + \sum_{l=1}^p\sum_{l'>l} \langle w_l, w_{l'} \rangle x_{c,l} x_{c,l'},\quad \psi_{k+1}(i) = 1 \qquad (28)
$$

$$
\phi_{k+2}(c) = 1,\quad \psi_{k+2}(i) = \sum_{l=1}^p z_{i,l} \tilde{h}_l + \sum_{l=1}^p\sum_{l'>l} \langle h_l, h_{l'} \rangle z_{i,l} z_{i,l'} \qquad (29)
$$

其中对于上下文， $\phi$ 由线性部分的 $\tilde{w}\in\mathbb{R}^p$ 和因子部分的 $W\in\mathbb{R}^{p\times k}$ 参数化。对于item， $\psi$ 由线性部分的 $\tilde{h}\in\mathbb{R}^p$ 和因子部分的 $H\in\mathbb{R}^{p\times k}$ 参数化。

梯度是稀疏的：

$$
\frac{\partial\phi_f(c)}{\partial\tilde{w}_{l^*}} = \begin{cases} x_{c,l^*}, & \text{if } f = k+1 \\ 0, & \text{else} \end{cases} \qquad (30)
$$

$$
\frac{\partial\phi_f(c)}{\partial w_{l^*,f^*}} = \begin{cases} x_{c,l^*}, & \text{if } f = f^* \\ x_{c,l^*}(\phi_{f^*}(c) - x_{c,l^*}w_{l^*,f^*}), & \text{if } f = k+1 \\ 0, & \text{else} \end{cases} \qquad (31)
$$

与MFSI类似，由于梯度的稀疏性，一阶正则化器导数 $R'$ 的嵌套循环之一可以省略，二阶正则化器导数 $R''$ 的两个嵌套求和都可以省略。因此，FM的流程和运行时分析与MFSI相同。

### 5.3 张量因子分解

张量因子分解推广了矩阵因子分解，处理涉及两个以上类别变量的问题。例如，在书签标签的个性化推荐中[20]，上下文由两个变量组成：用户 $C_1$ 和书签 $C_2$ ，item $I$ 对应标签。对于个性化网页搜索[19]，上下文由用户 $C_1$ 和查询 $C_2$ 组成，item $I$ 对应网页。该数据可以看作是 $C_1$ 、 $C_2$ 和 $I$ 上的三模张量。图4展示了上下文 $C\subseteq C_1\times C_2$ 和item $I$ 上的观测如何对应到一个张量。张量因子分解模型试图用低秩分解来近似张量（见图5）。尽管张量因子分解模型是多线性的，我们证明它们很好地适用于我们的框架。

此外，我们要强调的是，现有的张量因子分解学习算法[19, 20, 10]要求张量数据是稠密的，即张量中的空部分（图4中的空位）用零填充。这意味着从未被观测到的上下文组合也被用于训练，即 $C = C_1 \times C_2$ 。在某些应用中，这可能没有意义，例如如果 $C_1$ 编码设备类型而 $C_2$ 编码操作系统版本。我们的iCD框架对稀疏和稠密上下文都适用。必要时我们会指出差异。

**图5：张量因子分解模型**将给定张量分解为每个模的一个矩阵，例如 $C_1$ 的 $U\in\mathbb{R}^{C_1\times k_1}$ ， $C_2$ 的 $V\in\mathbb{R}^{C_2\times k_2}$ 和 $I$ 的 $W\in\mathbb{R}^{C_3\times k_3}$ 。

#### 5.3.1 并行因子分析（PARAFAC）

我们首先讨论并行因子分析（PARAFAC）[3]模型，它是矩阵因子分解的三模扩展。

$$
\hat{y}(c_1,c_2,i) := \sum_{f=1}^k u_{c_1,f} v_{c_2,f} w_{i,f} \qquad (32)
$$

其中 $\Theta = \{U, V, W\}$ ， $U\in\mathbb{R}^{C_1\times k}$ ， $V\in\mathbb{R}^{C_2\times k}$ ， $W\in\mathbb{R}^{I\times k}$ 。PARAFAC是k-可分离的，其中

$$
\phi_f(c_1,c_2) = u_{c_1,f} v_{c_2,f},\quad \psi_f(i) = w_{i,f} \qquad (33)
$$

同样，梯度是稀疏的：

$$
\frac{\partial\phi_f(c_1,c_2)}{\partial u_{c_1^*,f^*}} = \begin{cases} v_{c_2,f}, & \text{if } c_1 = c_1^* \land f = f^* \\ 0, & \text{else} \end{cases} \qquad (34)
$$

且损失导数简化为：

$$
R'(u_{c_1^*,f^*}) = 2\sum_{f=1}^k J_I(f,f^*) u_{c^*,f} \sum_{c_2:(c_1^*,c_2)\in C} v_{c_2,f} v_{c_2,f^*} \qquad (35)
$$

$$
R''(u_{c_1^*,f^*}) = 2 J_I(f^*,f^*) \sum_{c_2:(c_1^*,c_2)\in C} v_{c_2,f^*} v_{c_2,f^*} \qquad (36)
$$

item侧与矩阵因子分解等价。

如果上下文是稠密的且包括上下文变量的所有可能组合，即 $C = C_1 \times C_2$ ，则 $J_C(f,f^{\prime})$ 的计算可以分解为：

$$
J_C(f,f^{\prime}) = \underbrace{\left(\sum_{c_1\in C_1} u_{c_1,f} u_{c_1,f'}\right)}_{=:J_{C_1}(f,f^{\prime})} \underbrace{\left(\sum_{c_2\in C_2} v_{c_2,f} v_{c_2,f'}\right)}_{=:J_{C_2}(f,f^{\prime})} \qquad (37)
$$

这意味着计算复杂度为 $O(|C_1|+|C_2|)$ 而非 $O(|C_1||C_2|)$ 。另一方面，如果 $C$ 是稀疏的且只包含已观测上下文组合的子集，即 $C\subset C_1\times C_2$ ，则无需分解这个求和。公式(35,36)的损失导数同样如此：如果建模了所有可能的上下文，则 $\{c_2:(c_1^*,c_2)\in C\} = C_2$ ，因此 $J_{C_2}(f,f^{\prime})$ 可以替代对 $C_2$ 的求和。

PARAFAC隐式正则化器的总运行时对于稀疏上下文为 $O((|C|+|I|)k^2)$ ，对于稠密上下文为 $O((|C_1|+|C_2|+|I|)k^2)$ 。对模型参数的遍历可以像MF算法一样安排。

#### 5.3.2 Tucker分解

Tucker分解（TD）[21]是PARAFAC的推广，计算因子矩阵之间的所有交互。每个交互的强度由核心张量 $B$ 给出。对于包含两个上下文变量 $c_1,c_2$ 和一个item变量 $i$ 的例子，TD定义为

$$
\hat{y}(c_1,c_2,i) = \sum_{f_1=1}^{k_1}\sum_{f_2=1}^{k_2}\sum_{f_3=1}^{k_3} b_{f_1,f_2,f_3} u_{c_1,f_1} v_{c_2,f_2} w_{i,f_3} \qquad (38)
$$

其中 $\Theta = \{B, U, V, W\}$ ， $B\in\mathbb{R}^{k_1\times k_2\times k_3}$ 是核心张量， $U\in\mathbb{R}^{|C_1|\times k_1}$ ， $V\in\mathbb{R}^{|C_2|\times k_2}$ ， $W\in\mathbb{R}^{|I|\times k_3}$ 。TD的计算成本远高于PARAFAC，仅在单个数据点上评估模型就需要 $O(k_1 k_2 k_3)$ 次操作。

尽管Tucker分解包含嵌套求和，它是 $k_3$ -可分离的，其中

$$
\phi_f(c_1,c_2) = \sum_{f_1=1}^{k_1}\sum_{f_2=1}^{k_2} b_{f_1,f_2,f} u_{c_1,f_1} v_{c_2,f_2}, \quad \psi_f(i) = w_{i,f} \qquad (39)
$$

这些函数的导数为：

$$
\frac{\partial\phi_f(c_1,c_2)}{\partial u_{c_1,f_1^*}} = \begin{cases} \sum_{f_2=1}^{k_2} b_{f_1^*,f_2,f} v_{c_2,f_2}, & \text{if } c_1 = c_1^* \\ 0, & \text{else} \end{cases} \qquad (40)
$$

$$
\frac{\partial\phi_f(c_1,c_2)}{\partial v_{c_2,f_2^*}} = \begin{cases} \sum_{f_1=1}^{k_1} b_{f_1,f_2^*,f} u_{c_1,f_1}, & \text{if } c_2 = c_2^* \\ 0, & \text{else} \end{cases} \qquad (41)
$$

$$
\frac{\partial\phi_f(c_1,c_2)}{\partial b_{f_1^*,f_2^*,f_3^*}} = \begin{cases} u_{c_1,f_1^*} v_{c_2,f_2^*}, & \text{if } f = f_3^* \\ 0, & \text{else} \end{cases} \qquad (42)
$$

$$
\frac{\partial\psi_f(i)}{\partial w_{i^*,f_3^*}} = \begin{cases} 1, & \text{if } f = f_3^* \land i = i^* \\ 0, & \text{else} \end{cases} \qquad (43)
$$

与迄今介绍的所有其他模型不同， $\phi$ 的梯度对于任何因子索引 $f\in\{1,\ldots,k_3\}$ 都是非零的。因此，损失梯度（公式13）中关于因子的嵌套循环无法进一步改进。然而，对于稀疏的 $\psi$ ，可以应用与其他模型相同的优化。

与PARAFAC类似，如果 $C$ 是稠密的，即 $C = C_1\times C_2$ ，我们可以预计算 $C_1$ 和 $C_2$ 的中间矩阵，且 $J_C(f,f^{\prime})$ 的计算简化为

$$
J_C(f,f^{\prime}) = \sum_{f_1=1}^{k_1}\sum_{f'_1=1}^{k_1}\sum_{f_2=1}^{k_2}\sum_{f'_2=1}^{k_2} b_{f_1,f_2,f} b_{f'_1,f'_2,f'} J_{C_1}(f_1,f'_1) J_{C_2}(f_2,f'_2) \qquad (44)
$$

如果 $C$ 是稀疏的，则无需此优化，我们可以直接计算 $J_C$ 。总的运行时复杂度为稠密上下文 $O(k_1^2 k_2^2 k_3^2 (|C_1|+|C_2|+|I|))$ 和稀疏上下文 $O(k_1^2 k_2^2 k_3^2 (|C|+|I|))$ 。

## 6 实验

实验的主要目标是展示iCD框架的通用性。我们展示如何将iCD应用于无法仅用MF解决的多种推荐问题。对于MF模型，之前已经提出过高效的坐标下降算法[5]，并将其性能与BPR[13]等梯度下降算法进行了比较。两种方法都被认为是最先进的，虽然CD在某些数据集上优于BPR[8, 25, 15, 22, 26]，但BPR在另一些数据集上表现更好[4, 17, 16, 8]。我们实验的目的不是在又一个数据集上比较BPR和CD，而是展示iCD框架的多功能性，并说明它如何作为未来复杂推荐模型研究的构建块。与MF类似，iCD和BPR可能在不同的应用中表现出各自的优势。

### 6.1 实验设置

我们评估的数据集包含200,000个用户与YouTube的交互。我们的子集包含 $|I|=68,000$ 个视频。该数据集还包含关于年龄、国家、性别和设备信息的辅助信息。我们将iCD应用于三种流行的推荐问题——冷启动推荐、离线推荐和即时推荐（见第6.2节）。我们比较以下算法：

- **Popularity（流行度）：** 一个返回最流行视频的静态推荐器。
- **Coview（共视图）：** 基于之前观看的视频，返回最常被选为下一个观看的视频。
- **iCD-MF：** 使用iCD优化的用户-item矩阵因子分解，类似于[5]。
- **iCD-FM：** 根据上下文使用不同特征的因子分解机（第5.2节）。我们报告不同特征选择的结果。

我们度量前100个返回视频的召回率（Recall）和归一化折损累计增益（NDCG）。注意，我们报告相对于Popularity推荐器的相对改进。所有超参数在单独的验证集上进行调优。

### 6.2 结果

#### 6.2.1 冷启动推荐

在冷启动推荐[2]场景中，假设用户第一次与推荐系统交互。为模拟此场景，我们选择用户的随机子集，保留其所有事件用于评估；我们在剩余用户上训练。

处理冷启动的常见方法是使用辅助信息表示用户[2]。这里，我们使用基于特征的FM模型（iCD-FM），以用户的年龄、性别、国家和设备信息作为上下文特征。图7显示，属性感知FM比基线实现了2倍的改进。正如预期，MF和Coview都无法比最流行推荐做得更好。

#### 6.2.2 离线推荐

在离线推荐场景中，我们保留每个用户的最后一个反馈，并使用所有之前的反馈进行训练。这是评估推荐算法性能最常用的方法。我们使用多个FM模型进行实验：(1) iCD-FM A：带用户属性的FM，(2) iCD-FM P：仅使用之前观看视频的序列FM（类似于FPMC[14]或Coview），(3) iCD-FM A+P+U：使用所有信号的FM：属性、之前观看的视频和用户ID（类似于带用户属性的FPMC[14]）。如图6a所示，具有所有特征的复杂FM模型达到了最佳质量，展示了iCD进行特征工程的灵活性。

#### 6.2.3 即时推荐

在大规模工业应用中，由于复杂的服务架构，在线训练通常不可行。通常，模型离线定期训练（例如每天或每周），并在用户交互流上应用。当模型被请求为用户生成推荐时，当前时间之前的所有反馈都被用于预测。我们通过选择一个全局截止时间来模拟此设置，截止前的所有事件用于训练，剩余事件用于评估。

在此类设置中，依赖用户ID（如MF）的模型无法捕捉最近的反馈。相反，通过之前观看视频的序列来描述用户可以实现即时个性化。这样的模型可以使用基于特征的FM模型（第5.2节）配置。我们用四种配置进行实验：(1) iCD-FM A：使用用户属性的FM，(2) iCD-FM P：基于之前观看视频的序列FM，(3) iCD-FM H：基于所有之前观看视频的FM，(4) iCD-FM A+P+H：结合所有信号的FM。如图6b所示，具有所有特征的复杂FM模型达到了最佳质量。再次，我们要强调iCD框架的通用性，它实现了灵活的特征工程。

### 6.3 计算成本

如第3.3节所述，任何传统的CD求解器（例如[11]）都可以解决隐式反馈问题。现在，我们证明这由于隐式样本的巨大数量而不可行。图8比较了在我们的包含70k个item的数据集上，使用传统CD学习FM与使用iCD的计算成本。我们使用来自图6的三种不同上下文特征。该图显示了相对于iCD-FM P的相对成本。对于所有三种上下文选择，传统CD显示出比iCD高四个数量级的计算成本。iCD的经验测量运行时为分钟级别；因此，CD四个数量级的运行时增加意味着每次迭代需要数周的训练时间。显然，使用传统CD求解器直接优化隐式损失是不可行的。

**图6：(a) 离线推荐，(b) 即时推荐。** iCD-FM模型中使用的上下文特征的不同变体：A = 性别、年龄、国家和设备，P = 之前观看的视频，H = 到目前为止观看的所有视频，U = 用户ID。

**图7：冷启动推荐**

**图8：在我们的隐式数据集上，传统CD（左，蓝色）与iCD（右，红色）的训练成本（对数尺度）。**

## 7 结论

在本文中，我们提出了一个从隐式反馈中学习推荐系统模型的通用高效框架。首先，我们展示了从隐式反馈中学习可以重新表述为优化一个廉价的显式损失和一个昂贵的隐式正则化器。然后，我们引入了k-可分离模型的概念。我们证明任何k-可分离模型的隐式正则化器都可以在不遍历所有上下文-item对的情况下高效计算。最后，我们展示了多种流行的推荐模型是k-可分离的，包括矩阵因子分解、因子分解机和张量因子分解。此外，我们基于我们的框架为这些模型提供了高效的学习算法。我们的框架不限于本文讨论的模型，而是旨在作为推导推荐系统学习算法的通用蓝图。

## 参考文献

[1] C. Cheng, H. Yang, M. R. Lyu, and I. King. Where You Like to Go Next: Successive Point-of-Interest Recommendation. In *IJCAI*, volume 13, pages 2605–2611, 2013.

[2] Z. Gantner, L. Drumond, C. Freudenthaler, S. Rendle, and L. Schmidt-Thieme. Learning attribute-to-feature mappings for cold-start recommendations. In *2010 IEEE International Conference on Data Mining*, pages 176–185. IEEE, 2010.

[3] R. A. Harshman. Foundations of the PARAFAC procedure: Models and conditions for an "explanatory" multi-modal factor analysis. *UCLA Working Papers in Phonetics*, 16(1):84, 1970.

[4] R. He and J. McAuley. **VBPR: Visual bayesian personalized ranking from implicit feedback**. In D. Schuurmans and M. P. Wellman, editors, *AAAI*, pages 144–150. AAAI Press, 2016.

[5] Y. Hu, Y. Koren, and C. Volinsky. **Collaborative filtering for implicit feedback datasets**. In *Proceedings of the 2008 Eighth IEEE International Conference on Data Mining*, ICDM '08, pages 263–272, 2008.

[6] B. Kanagal, A. Ahmed, S. Pandey, V. Josifovski, J. Yuan, and L. Garcia-Pueyo. Supercharging recommender systems using taxonomies for learning user purchase behavior. *Proc. VLDB Endow.*, 5(10):956–967, June 2012.

[7] B. McFee, T. Bertin-Mahieux, D. P. Ellis, and G. R. Lanckriet. The million song dataset challenge. In *Proceedings of the 21st International Conference on World Wide Web*, WWW '12 Companion, pages 909–916, New York, NY, USA, 2012. ACM.

[8] X. Ning and G. Karypis. Slim: Sparse linear methods for top-n recommender systems. In *2011 IEEE 11th International Conference on Data Mining*, pages 497–506. IEEE, 2011.

[9] W. Pan and L. Chen. GBPR: Group Preference Based Bayesian Personalized Ranking for One-Class Collaborative Filtering. In *IJCAI*, volume 13, pages 2691–2697, 2013.

[10] I. Pilászy, D. Zibriczky, and D. Tikk. Fast als-based matrix factorization for explicit and implicit feedback datasets. In *Proceedings of the fourth ACM conference on Recommender systems*, pages 71–78. ACM, 2010.

[11] S. Rendle. Factorization machines with libfm. *ACM Trans. Intell. Syst. Technol.*, 3(3):57:1–57:22, may 2012.

[12] S. Rendle and C. Freudenthaler. **Improving pairwise learning for item recommendation from implicit feedback**. In *Proceedings of the 7th ACM International Conference on Web Search and Data Mining*, WSDM '14, pages 273–282, New York, NY, USA, 2014. ACM.

[13] S. Rendle, C. Freudenthaler, Z. Gantner, and L. Schmidt-Thieme. **BPR: Bayesian personalized ranking from implicit feedback.** In *Proceedings of the Twenty-Fifth Conference on Uncertainty in Artificial Intelligence*, UAI '09, pages 452–461, Arlington, Virginia, United States, 2009. AUAI Press.

[14] S. Rendle, C. Freudenthaler, and L. Schmidt-Thieme. Factorizing personalized markov chains for next-basket recommendation. In *Proceedings of the 19th International Conference on World Wide Web*, WWW '10, pages 811–820. ACM, 2010.

[15] S. Sedhain, A. K. Menon, S. Sanner, and D. Braziunas. On the effectiveness of linear models for one-class collaborative filtering. In *Proceedings of the 30th Conference on Artificial Intelligence (AAAI-16)*, 2016.

[16] Y. Shi, A. Karatzoglou, L. Baltrunas, M. Larson, A. Hanjalic, and N. Oliver. TFMAP: optimizing MAP for top-n context-aware recommendation. In *Proceedings of the 35th international ACM SIGIR conference on Research and development in information retrieval*, pages 155–164. ACM, 2012.

[17] Y. Shi, A. Karatzoglou, L. Baltrunas, M. Larson, N. Oliver, and A. Hanjalic. CLiMF: learning to maximize reciprocal rank with collaborative less-is-more filtering. In *Proceedings of the sixth ACM conference on Recommender systems*, pages 139–146. ACM, 2012.

[18] E. Shmueli, A. Kagian, Y. Koren, and R. Lempel. Care to comment?: recommendations for commenting on news stories. In *Proceedings of the 21st international conference on World Wide Web*, pages 429–438. ACM, 2012.

[19] J.-T. Sun, H.-J. Zeng, H. Liu, Y. Lu, and Z. Chen. Cubesvd: A novel approach to personalized web search. In *Proceedings of the 14th International Conference on World Wide Web*, WWW '05, pages 382–390, New York, NY, USA, 2005. ACM.

[20] P. Symeonidis, A. Nanopoulos, and Y. Manolopoulos. A unified framework for providing recommendations in social tagging systems based on ternary semantic analysis. *IEEE Trans. on Knowl. and Data Eng.*, 22(2):179–192, Feb. 2010.

[21] L. R. Tucker. Some mathematical notes on three-mode factor analysis. *Psychometrika*, 31:279–311, 1966.

[22] M. Volkovs and G. W. Yu. Effective latent models for binary feedback in recommender systems. In *Proceedings of the 38th International ACM SIGIR Conference on Research and Development in Information Retrieval*, pages 313–322. ACM, 2015.

[23] H.-F. Yu, C.-J. Hsieh, S. Si, and I. Dhillon. Scalable coordinate descent approaches to parallel matrix factorization for recommender systems. In *Proceedings of the 12th International Conference on Data Mining*, ICDM '12, pages 765–774, 2012.

[24] X. Yu, X. Ren, Y. Sun, Q. Gu, B. Sturt, U. Khandelwal, B. Norick, and J. Han. Personalized entity recommendation: A heterogeneous information network approach. In *Proceedings of the 7th ACM International Conference on Web Search and Data Mining*, WSDM '14, pages 283–292. ACM, 2014.

[25] T. Zhao, J. McAuley, and I. King. Leveraging social connections to improve personalized ranking for collaborative filtering. In *Proceedings of the 23rd ACM International Conference on Conference on Information and Knowledge Management*, CIKM '14, pages 261–270, New York, NY, USA, 2014. ACM.

[26] T. Zhao, J. McAuley, and I. King. Improving latent factor models via personalized feature projection for one class recommendation. In *Proceedings of the 24th ACM International on Conference on Information and Knowledge Management*, pages 821–830. ACM, 2015.
