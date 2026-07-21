# Factorization Machines with libFM

> Steffen Rendle | University of Konstanz

## 因子分解机与 libFM

本文介绍了 Factorization Machines with libFM。核心内容：


关键发现：



因子分解方法在几个重要的预测问题中提供了高精度，例如推荐系统。然而，将因子分解方法应用于一个新的预测问题并非易事，并且需要大量的专家知识。通常，需要开发一个新模型，推导出一个学习算法，并且必须实现该方法。

因子分解机（FM）是一种通用方法，因为它们仅通过特征工程就能模仿大多数因子分解模型。通过这种方式，因子分解机将特征工程的通用性与因子分解模型在估计大领域类别变量之间交互方面的优越性结合起来。LIBFM 是因子分解机的软件实现，具有随机梯度下降（SGD）和交替最小二乘（ALS）优化，以及使用马尔可夫链蒙特卡洛（MCMC）的贝叶斯推断。本文总结了最近在建模和学习方面关于因子分解机的研究，提供了 ALS 和 MCMC 算法的扩展，并描述了软件工具 LIBFM。

类别和主题描述符：I.2.6 [人工智能]：学习——参数学习；I.5.2 [模式识别]：设计方法论——分类器设计与评估；H.3.3 [信息存储与检索]：信息搜索与检索——信息过滤

通用术语：算法、实验、测量、性能

附加关键词和短语：因子分解模型、矩阵分解、张量分解、推荐系统、协同过滤、因子分解机

ACM 引用格式：
Rendle, S. 2012. Factorization machines with libFM. ACM Trans. Intell. Syst. Technol. 3, 3, Article 57 (May 2012), 22 pages.
DOI = 10.1145/2168752.2168771 http://doi.acm.org/10.1145/2168752.2168771

## 1. 引言

最近，因子分解模型在智能信息系统和机器学习领域吸引了大量研究。它们在几个重要应用中展现了出色的预测能力，例如推荐系统。研究最充分的因子分解模型是矩阵分解 [Srebro and Jaakkola 2003]，它允许我们预测两个类别变量之间的关系。张量分解模型是针对多个类别变量之间关系的扩展；在已提出的张量分解方法中包括 Tucker 分解 [Tucker 1966]、平行因子分析 [Harshman 1970] 或成对交互张量分解 [Rendle and Schmidt-Thieme 2010]。针对特定任务，已经提出了考虑非类别变量的专门化因子分解模型，例如 SVD++ [Koren 2008]、STE [Ma et al. 2011]、FPMC

S. Rendle,

社会网络分析，大学

作者地址：
steffen.rendle@uni-konstanz.de.

允许以个人或课堂使用为目的制作本作品的部分或全部数字或硬拷贝，无需付费，前提是复制件不得为盈利或商业目的而制作或分发，并且复制件在显示的第一页或初始屏幕上显示此通知以及完整引用。必须尊重归 ACM 所有的本作品组件的版权。允许带有引用的摘要。以其他方式复制、重新发布、上传到服务器、分发给列表、或在本作品中使用任何组件用于其他作品，需要事先获得特定许可和/或支付费用。许可请求可寄往：Publications Dept., ACM, Inc., 2 Penn Plaza, Suite 701, New York, NY 10121-0701, USA, 传真 +1 (212) 869-0481, 或 permissions@acm.org.
c 2012 ACM 2157-6904/2012/05-ART57 $10.00$
DOI 10.1145/2168752.2168771 http://doi.acm.org/10.1145/2168752.2168771

康斯坦茨大学；

电子邮件：

57

ACM Transactions on Intelligent Systems and Technology, 第 3 卷, 第 3 期, Article 57, 出版日期：2012 年 5 月.

57:2

S. Rendle

[Rendle et al. 2010]（针对集合类别变量）、timeSVD++ [Koren 2009b] 和 BPTF [Xiong et al. 2010]（针对附加的数值变量）。对于基本的矩阵分解模型，已经研究了许多学习和推断方法——其中包括（随机）梯度下降、交替最小二乘（例如 [Pilászy et al. 2010]）、变分贝叶斯 [Lim and Teh 2007] 和马尔可夫链蒙特卡洛（MCMC）推断 [Salakhutdinov and Mnih 2008a]。然而，对于更复杂的因子分解模型，通常只有最简单的梯度下降学习方法可用。

尽管因子分解模型在许多应用中具有很高的预测质量，但使用它们并非易事。对于每个无法用类别变量描述的问题，必须推导出一个新的专门化模型，并且必须开发和实现一个学习算法。这非常耗时、容易出错，并且仅适用于因子分解模型的专家。

另一方面，在实践中，机器学习的典型方法是使用特征向量描述数据（一个预处理步骤，即特征工程），并应用标准工具，例如 LIBSVM [Chang and Lin 2011] 用于支持向量机、像 Weka [Hall et al. 2009] 这样的工具箱，或简单的线性回归工具。这种方法很简单，即使对底层机器学习模型和推断机制没有深入了解的用户也适用。

在本文中，介绍了因子分解机（FM）[Rendle 2010]。FM 将因子分解模型的高预测精度与特征工程的灵活性结合起来。FM 的输入数据用实值特征描述，就像其他机器学习方法（如线性回归、支持向量机等）一样。然而，FM 的内部模型使用变量之间的因子化交互，因此，它与其它因子分解模型一样，在稀疏设置中（如推荐系统）具有高预测质量。已经证明，FM 仅通过特征工程就可以模仿大多数因子分解模型 [Rendle 2010]。本文总结了关于 FM 的最新研究，包括基于随机梯度下降、交替最小二乘和使用 MCMC 的贝叶斯推断的学习算法。FM 及所有提出的算法都在公开可用的软件工具 LIBFM 中提供。使用 LIBFM，应用因子分解模型就像应用标准工具（如 SVM 或线性回归）一样简单。

本文结构如下：（1）介绍了 FM 模型及其在 LIBFM 中可用的学习算法；（2）给出了输入数据的几个示例，并展示了与专门化因子分解模型的关系；（3）简要介绍了 LIBFM 软件；（4）进行了实验。

## 2. 因子分解机模型

假设预测问题的数据由一个设计矩阵 X \in R^(n×p) 描述，其中 X 的第 i 行 x_i \in R^p 描述了一个具有 p 个实值变量的案例，并且 y_i 是第 i 个案例的预测目标（见图 1 的示例）。或者，可以将此设置描述为元组 (x, y) 的集合 S，其中（再次）x \in R^p 是一个特征向量，y 是其对应的目标。这种用数据矩阵和特征向量的表示在许多机器学习方法中很常见，例如线性回归或支持向量机（SVM）。

因子分解机（FM）[Rendle 2010] 使用因子化的交互参数对 x 中 p 个输入变量之间所有最多 d 阶的嵌套交互进行建模。阶数 d = 2 的因子分解机（FM）模型定义为

ŷ(x) := w_0 + \Sigma_{j=1}^{p} w_j x_j + \Sigma_{j=1}^{p} \Sigma_{j'=j+1}^{p} x_j $x_{j'}$ \Sigma_{f=1}^{k} $v_{j,f}$ $v_{j',f}$,   (1)

ACM Transactions on Intelligent Systems and Technology, 第 3 卷, 第 3 期, Article 57, 出版日期：2012 年 5 月.

Factorization Machines with libFM

57:3

图 1. 示例（来自 Rendle [2010]），展示了如何用实值特征向量 x 表示推荐问题。每一行表示一个特征向量 x_i 及其对应的目标 y_i。为了更容易解释，特征被分组为活跃用户（蓝色）、活跃item（红色）、同一用户评分的其他电影（橙色）、以月为单位的时间（绿色）以及最后评分的电影（棕色）的指示符。

其中 k 是因子分解的维度，模型参数 Θ = {w_0, w_1, ..., w_p, $v_{1,1}$, ... $v_{p,k}$} 为

w_0 \in R, w \in R^p, V \in R^(p×k).   (2)

FM 模型的第一部分包含每个输入变量 x_j 与目标的单变量交互——完全如同线性回归模型。第二部分包含两个嵌套求和，包含了输入变量的所有成对交互，即 x_j $x_{j'}$。与标准多项式回归的重要区别在于，交互的效果不是由独立参数 $w_{j,j'}$ 建模的，而是通过因子化参数化 $w_{j,j'}$ \approx ⟨v_j, $v_{j'}$⟩ = \Sigma_{f=1}^{k} $v_{j,f}$ $v_{j',f}$ 来建模，这对应于成对交互的效果具有低秩的假设。这使得 FM 即使在标准模型失败的高度稀疏数据中也能估计可靠的参数。FM 与标准机器学习模型的关系将在第 4.3 节中更详细地讨论。在第 4 节中，还将展示 FM 如何模仿其他著名的因子分解模型，包括矩阵分解、SVD++、FPMC、timeSVD 等。

**复杂度**。令 N_z 为矩阵 X 或向量 x 中非零元素的数量。

N_z(X) := \Sigma_i \Sigma_j \delta($x_{i,j}$ \neq 0),   (3)

其中 \delta 是指示函数

\delta(b) := { 1, 如果 b 为真；0, 如果 b 为假 }。   (4)

方程 (1) 中的 FM 模型可以在 O(k N_z(x)) 内计算，因为它等价于 [Rendle 2010]

ŷ(x) = w_0 + \Sigma_{j=1}^{p} w_j x_j + 1/2 \Sigma_{f=1}^{k} [ (\Sigma_{j=1}^{p} $v_{j,f}$ x_j)^2 - \Sigma_{j=1}^{p} $v_{j,f}$^2 x_j^2 ]。   (5)

FM 的模型参数数量 |Θ| 为 1 + p + k p，因此与预测变量的数量（= 输入特征向量的大小）线性相关，并且与因子分解大小 k 线性相关。

ACM Transactions on Intelligent Systems and Technology, 第 3 卷, 第 3 期, Article 57, 出版日期：2012 年 5 月.

57:4

S. Rendle

**多线性**。FM 的一个吸引人的特性是多线性，即对于每个模型参数 \theta \in Θ，FM 是两个函数 g_\theta 和 h_\theta 的线性组合，这两个函数独立于 \theta 的值 [Rendle et al. 2011]。

ŷ(x) = g_\theta(x) + \theta h_\theta(x)  \forall\theta \in Θ,   (6)

其中

h_\theta(x) = \partialŷ(x)/\partial\theta = { 1, 如果 \theta 是 w_0；x_l, 如果 \theta 是 w_l；x_l \Sigma_{j\neql} $v_{j,f}$ x_j, 如果 \theta 是 $v_{l,f}$ }。   (7)

省略了 g_\theta 的定义，因为在下面的内容中从未直接使用。如果需要计算其值，将使用方程 g_\theta(x) = ŷ(x) - \theta h_\theta(x)。

**表达能力**。只要 k 选择得足够大，FM 模型可以表达任何成对交互。这源于以下事实：任何对称半正定矩阵 W 都可以分解为 V V^t（例如，Cholesky 分解）。设 W 是任意成对交互矩阵，应在 FM 中表达两个不同变量之间的交互。W 是对称的，并且由于 FM 不使用对角线元素（因为方程 (1) 中 j' > j），对角线元素的任何值——尤其是任意大的值——都是可能的，这将使 W 成为半正定矩阵。

请注意，这是关于表达能力的理论说明。在实践中，k ≪ p，因为 FM 的优势在于可以使用 W 的低秩近似，因此 FM 即使在高度稀疏的数据中也能估计交互参数——参见第 4.3 节与使用完整矩阵 W 建模交互的多项式回归的比较。

**高阶 FM**。d = 2 阶的 FM 模型（方程 (1)）可以通过因子化三元及更高阶的变量交互来扩展。高阶 FM 模型 [Rendle 2010] 为

ŷ(x) := w_0 + \Sigma_{j=1}^{p} w_j x_j + \Sigma_{l=2}^{d} [ \Sigma_{j_1=1}^{p} ... \Sigma_{j_d=$j_{d-1}$+1}^{p} (\prod_{i=1}^{l} $x_{j_i}$) (\Sigma_{f=1}^{k_l} \prod_{i=1}^{l} $v_{j_i, f}$) ],   (8)

其中模型参数

w_0 \in R, w \in R^p, \foralll \in {2, ..., d}: V_l \in R^(p×k_l).   (9)

对于更高阶的交互，方程 (8) 中的嵌套求和也可以分解以实现更高效的计算。在本文的其余部分，我们将只处理二阶 FM，因为在稀疏设置中——因子分解模型尤其具有吸引力——通常高阶交互难以估计 [Rendle and Schmidt-Thieme 2010]。尽管如此，大多数公式和算法可以直接迁移到高阶 FM，因为它们与二阶 FM 共享多线性的性质。

## 3. 学习因子分解机

已经提出了三种 FM 的学习方法：随机梯度下降（SGD）[Rendle 2010]、交替最小二乘（ALS）[Rendle et al. 2011] 和马尔可夫链蒙特卡洛（MCMC）推断 [Freudenthaler et al. 2011]。所有这三种方法都在 LIBFM 中可用。

ACM Transactions on Intelligent Systems and Technology, 第 3 卷, 第 3 期, Article 57, 出版日期：2012 年 5 月.

Factorization Machines with libFM

57:5

### 3.1. 优化任务

模型参数的最优性通常用损失函数 l 来定义，其中任务是最小化观测数据 S 上的损失之和。

OPT(S) := argmin_Θ \Sigma_{(x,y)\inS} l(ŷ(x|Θ), y).   (10)

注意，我们将模型参数 Θ 添加到模型方程中，并写为 ŷ(x|Θ)，当我们想强调 ŷ 依赖于 Θ 的特定选择时。根据任务，可以选择损失函数。例如，对于回归，最小二乘损失：

$l_{LS}$(y_1, y_2) := (y_1 - y_2)^2,   (11)

或者对于二分类（y \in {-1, 1}）：

l_C(y_1, y_2) := -ln \sigma(y_1 y_2),   (12)

其中 \sigma(x) = 1/(1+e^{-x}) 是 sigmoid/逻辑函数。

FM 通常有大量的模型参数 Θ——特别是当 k 选择得足够大时。这使得它们容易过拟合。为了克服这一点，通常应用 L2 正则化，这可以由最大间隔 [Srebro et al. 2005] 或 Tikhonov 正则化来推动。

OP$T_{REG}$(S, \lambda) := argmin_Θ [ \Sigma_{(x,y)\inS} l(ŷ(x|Θ), y) + \Sigma_{\theta\inΘ} \lambda_\theta \theta^2 ],   (13)

其中 \lambda_\theta \in R^+ 是模型参数 \theta 的正则化值。对模型的不同部分使用单独的正则化参数是有意义的。在 LIBFM 中，模型参数可以被分组——例如，一组用于描述用户的参数，一组用于item，一组用于时间等（见图 1 的分组示例）——每个组使用一个独立的正则化值。此外，每个因子化层 f \in {1, ..., k} 以及单变量回归系数 w 和 w_0 可以有单独的正则化（同样带有分组）。总的来说，LIBFM 的正则化结构为

\lambda_0, \forall\pi \in {1, ..., Π}, \forallf \in {1, ..., k}: \lambda_\pi^w, \lambda_{f,\pi}^v,   (14)

其中 \pi: {1, ..., p} \rightarrow {1, ..., Π} 是模型参数的分组。这意味着，例如，$v_{l,f}$ 的正则化值将是 \lambda_{f, \pi(l)}^v。

**概率解释**。损失和正则化也可以从概率的角度来推动（例如，Salakhutdinov and Mnih [2008b]）。最小二乘损失对应于假设目标 y 服从高斯分布，均值为预测值：

y|x, Θ ∼ N(ŷ(x, Θ), 1/\alpha).   (15)

对于二分类，假设服从伯努利分布：

y|x, Θ ∼ Bernoulli(b(ŷ(x, Θ))),   (16)

其中 b: R \rightarrow [0, 1] 是一个链接函数，通常是逻辑函数 \sigma 或标准正态分布的累积分布函数（CDF）\Phi。

L2 正则化对应于模型参数上的高斯先验：

\theta|\mu_\theta, \lambda_\theta ∼ N(\mu_\theta, 1/\lambda_\theta)   (17)

ACM Transactions on Intelligent Systems and Technology, 第 3 卷, 第 3 期, Article 57, 出版日期：2012 年 5 月.

57:6

S. Rendle

图 2. 标准因子分解机中所涉变量的图形化表示。(a) 变量为目标 y、输入特征 x、模型参数 w_0、w_j、$v_{j,f}$ 以及超参数/先验 \mu、\lambda、\alpha。(b) 先验通过超先验 Θ_0 = {\alpha_0, \alpha_\lambda, \beta_0, \beta_\lambda, \gamma_0, \mu_0} 进行扩展，这允许 MCMC 算法（算法 3）自动寻找先验参数 [Freudenthaler et al. 2011]。

先验均值 \mu_\theta 应与正则化值 \lambda_\theta（见方程 (14)）以相同方式分组和组织。

概率视角的图形模型见图 2(a)。该模型的最大后验（MAP）估计器（取 \alpha = 1, \mu_\theta = 0）与方程 (13) 的优化准则相同。

**梯度**。对于损失函数的直接优化，导数为：

对于最小二乘回归：

\partial/\partial\theta $l_{LS}$(ŷ(x|Θ), y) = \partial/\partial\theta [(ŷ(x|Θ) - y)]^2 = 2(ŷ(x|Θ) - y) \partial/\partial\theta ŷ(x|Θ),   (18)

或者对于分类：

\partial/\partial\theta l_C(ŷ(x|Θ), y) = \partial/\partial\theta [-ln \sigma(ŷ(x|Θ) y)] = (\sigma(ŷ(x|Θ) y) - 1) y \partial/\partial\theta ŷ(x|Θ).   (19)

最后，由于 FM 模型的多线性，模型方程关于 \theta 的偏导数对应于 h_\theta（方程 (7)）。

\partial/\partial\theta ŷ(x|Θ) = h_\theta(x).   (20)

### 3.2. 随机梯度下降（SGD）

随机梯度下降（SGD）算法在优化因子分解模型中非常流行，因为它们简单、能很好地处理不同的损失函数，并且具有低计算和存储复杂度。算法 1 展示了如何用 SGD 优化 FM

ACM Transactions on Intelligent Systems and Technology, 第 3 卷, 第 3 期, Article 57, 出版日期：2012 年 5 月.

Factorization Machines with libFM

57:7

**算法 1: 随机梯度下降 (SGD)**
输入：训练数据 S, 正则化参数 \lambda, 学习率 \eta, 初始化 \sigma
输出：模型参数 Θ = (w_0, w, V)
w_0 \leftarrow 0; w \leftarrow (0, ..., 0); V ∼ N(0, \sigma);
重复
  for (x, y) \in S do
    w_0 \leftarrow w_0 - \eta [ \partial/\partialw_0 l(ŷ(x|Θ), y) + 2 \lambda_0 w_0 ];
    for i \in {1, ..., p} ∧ x_i \neq 0 do
      w_i \leftarrow w_i - \eta [ \partial/\partialw_i l(ŷ(x|Θ), y) + 2 \lambda_{\pi(i)}^w w_i ];
      for f \in {1, ..., k} do
        $v_{i,f}$ \leftarrow $v_{i,f}$ - \eta [ \partial/\partial$v_{i,f}$ l(ŷ(x|Θ), y) + 2 \lambda_{f,\pi(i)}^v $v_{i,f}$ ];
      end
    end
  end
直到满足停止准则;

[Rendle 2010]。该算法遍历案例 (x, y) \in S 并对模型参数执行更新。

\theta \leftarrow \theta - \eta [ \partial/\partial\theta l(ŷ(x), y) + 2 \lambda_\theta \theta ],   (21)

其中 \eta \in R^+ 是梯度下降的学习率或步长。

**复杂度**。FM 的 SGD 算法具有线性的计算复杂度和常数的存储复杂度。对于遍历所有训练案例的一次迭代，SGD 的运行时复杂度为 O(k N_z(X))，因为对于每个单个案例 (x, y) \in S，梯度步骤的复杂度为 O(k \Sigma_{i=1}^{p} \delta(x_i \neq 0)) = O(k N_z(x))。

**超参数**。执行 SGD 时，有几个关键的超参数。
— 学习率 \eta: SGD 的收敛在很大程度上取决于 \eta：如果 \eta 选择得太高，算法不会收敛，如果选择得太小，收敛速度很慢。通常，\eta 是应首先确定的第一个超参数。
— 正则化 \lambda: 如第 3.1 节所述，FM 的泛化能力以及因此预测质量在很大程度上取决于正则化 \lambda 的选择。正则化值通常在单独的保留集上搜索，例如使用网格搜索。由于存在多个正则化参数（见方程 (14)），网格具有指数大小，因此这种搜索非常耗时。为了使搜索更可行，正则化参数的数量通常会减少，例如，放弃分组并且所有因子层使用相同的正则化值。
— 初始化 \sigma: 因子化交互（V）的参数必须用非常数值初始化。在 LIBFM 中，值从一个具有标准差 \sigma 的零均值正态分布中采样。通常 \sigma 使用较小的值。

**带自适应正则化的 SGD**。在 Rendle [2012] 中，展示了在学习模型参数时，如何在 SGD 中自动调整正则化值。LIBFM 包含那里提出的自适应正则化算法，并用分组对其进行了扩展。

ACM Transactions on Intelligent Systems and Technology, 第 3 卷, 第 3 期, Article 57, 出版日期：2012 年 5 月.

57:8

S. Rendle

### 3.3. 交替最小二乘/坐标下降

SGD 的优化方法基于遍历训练数据的案例（行）并在损失减小的方向上执行小步长。坐标下降或交替最小二乘（ALS）采用了另一种方法，即最小化每个模型参数的损失。对于带有 L2 正则化的最小二乘回归，给定所有剩余参数 Θ \ {\theta}，一个模型参数 \theta 的最优值 \theta* 可以直接计算 [Rendle et al. 2011] 为

\theta* = argmin_\theta [ \Sigma_{(x,y)\inS} (ŷ(x|Θ) - y)^2 + \Sigma_{\theta\inΘ} \lambda_\theta \theta^2 ]   (22)
= argmin_\theta [ \Sigma_{(x,y)\inS} (g_\theta(x|Θ\{\theta}) + \theta h_\theta(x|Θ\{\theta}) - y)^2 + \Sigma_{\theta\inΘ} \lambda_\theta \theta^2 ]   (23)
= [ \Sigma_{i=1}^{n} (y - g_\theta(x_i|Θ\{\theta})) h_\theta(x_i|Θ\{\theta}) ] / [ \Sigma_{i=1}^{n} h_\theta(x_i)^2 + \lambda_\theta ]   (24)
= [ \Sigma_{i=1}^{n} h_\theta(x_i) e_i ] / [ \Sigma_{i=1}^{n} h_\theta(x_i)^2 + \lambda_\theta ],   (25)

其中 e_i 是第 i 个案例的"误差"项/残差：
e_i := y_i - ŷ(x_i|Θ).   (26)

这允许我们推导一个最小二乘学习算法（见算法 2），该算法迭代地为每个模型参数求解一个最小二乘问题，并用最优（局部）解更新每个模型参数：

\theta \leftarrow \theta*.   (27)

这将对所有参数迭代执行，直到收敛。

**复杂度**。ALS 学习（方程 (22)）的主要工作在于计算以下两个量：

\Sigma_{i=1}^{n} h_\theta(x_i)^2,   \Sigma_{i=1}^{n} h_\theta(x_i) e_i = \Sigma_{i=1}^{n} h_\theta(x_i) (y_i - ŷ(x_i|Θ)).   (28-29)

使用简单的实现，更新一个模型参数需要为每个对应列 j 非零（$x_{i,j}$ \neq 0）的训练案例 x_i 计算模型方程 ŷ(x_i) 和梯度 h_\theta(x_i)。例如，对于更新模型参数 $v_{j,f}$，这将使复杂度为 O(\Sigma_{i=1}^{n} \delta($x_{i,j}$ \neq 0) k N_z(x_i))。总的来说，这必须为 1 + p(k + 1) 个模型参数中的每一个计算。

在 Rendle et al. [2011] 中，展示了如何通过预计算缓存 e \in R^n（见方程 (26)）和 Q \in R^(n×k) 使得在所有模型参数 Θ 上的一次完整迭代可以在 O(N_z(X) k) 内高效完成，其中

$q_{i,f}$ := \Sigma_{l=1}^{p} $v_{l,f}$ $x_{i,l}$,   (30)

这使得我们可以快速计算 h，复杂度 O(1)：

$h_{v_{l,f}$}(x_i) = $x_{i,l}$ ($q_{i,f}$ - $v_{l,f}$ $x_{i,l}$)。   (31)

ACM Transactions on Intelligent Systems and Technology, 第 3 卷, 第 3 期, Article 57, 出版日期：2012 年 5 月.

Factorization Machines with libFM

57:9

**算法 2: 交替最小二乘 (ALS)**
输入：训练数据 S, 正则化参数 \lambda, 初始化 \sigma
输出：模型参数 Θ = (w_0, w, V)
w_0 \leftarrow 0; w \leftarrow (0, ..., 0); V ∼ N(0, \sigma);
重复
  ŷ \leftarrow 预测所有案例 S;
  e \leftarrow y - ŷ;
  w_0 \leftarrow w_0*;
  for l \in {1, ..., p} do
    w_l \leftarrow w_l*;
    更新 e;
  end
  for f \in {1, ..., k} do
    初始化 $q_{·,f}$;
    for l \in {1, ..., p} do
      $v_{l,f}$ \leftarrow $v_{l,f}$*;
      更新 e, q;
    end
  end
直到满足停止准则;

现在，对于第 l 个参数计算 \theta* 的复杂度为 O(\Sigma_{i=1}^{n} \delta($x_{i,l}$ \neq 0))。同时，更新每个缓存值 q 和 e 可以在常数额外时间内完成（参见 Rendle et al. [2011]）。

然而，加速的代价是缓存的内存消耗更高。Rendle et al. [2011] 中提出的方法由于 Q 缓存而具有 O(n k) 的额外内存复杂度。LIBFM 提供了一种更高效的实现，Q 缓存仅需 O(n) 的内存复杂度（见算法 2）。其思想是模型参数按层 f 更新（即，首先所有参数 $v_{1,1}$, $v_{2,1}$, $v_{3,1}$, ..., 然后 $v_{1,2}$, $v_{2,2}$, $v_{3,2}$, ..., 等等），并且在每一层中，只有同一层的缓存值 Q 必须存在。这意味着 LIBFM 只存储（和更新）一层的 Q 缓存（因此存储量为 O(n)），并且当切换层时，新层的 Q 值会被计算/初始化。每层 Q 值的初始化对整体计算复杂度没有负面影响。

**超参数**。ALS 相对于 SGD 的一个明显优势是 ALS 没有学习率作为超参数。然而，仍然有两个重要的超参数：正则化和初始化。找到好的正则化值尤其计算代价高昂。

**分类**。到目前为止描述的 ALS 算法仅限于最小二乘回归，无法解决分类任务。LIBFM 包含用于 ALS/坐标下降的分类能力，这基于使用 probit 链接函数。这种方法源于概率解释（第 3.1 节），将在 MCMC 部分的末尾进行描述。

### 3.4. 马尔可夫链蒙特卡洛（MCMC）推断

到目前为止使用的贝叶斯模型可以在图 2 中看到。ALS 和 SGD 都学习用于 ŷ 的点估计的最佳参数 Θ。MCMC 是一种贝叶斯推断技术，通过采样生成 ŷ 的分布。对于 FM 中使用吉布斯采样的 MCMC

ACM Transactions on Intelligent Systems and Technology, 第 3 卷, 第 3 期, Article 57, 出版日期：2012 年 5 月.

57:10

S. Rendle

推断，每个模型参数的条件后验分布为 [Freudenthaler et al. 2011]：

\theta | X, y, Θ\{\theta}, Θ_H ∼ N(\mũ_\theta, \sigmã_\theta^2),   (30)

其中

\sigmã_\theta^2 := ( \alpha \Sigma_{i=1}^{n} h_\theta(x_i)^2 + \lambda_\theta )^{-1},   (31)

\mũ_\theta := \sigmã_\theta^2 ( \alpha \Sigma_{i=1}^{n} h_\theta(x_i)^2 \theta + \alpha \Sigma_{i=1}^{n} h_\theta(x_i) e_i + \mu_\theta \lambda_\theta ),   (32)

并且 Θ_H 是超参数：

Θ_H := {(\mu_0, \lambda_0), (\mu_\pi^w, \lambda_\pi^w), (\mu_{f,\pi}^v, \lambda_{f,\pi}^v) : \forall\pi \in {1, ..., Π}, \forallf \in {1, ..., k}}.   (33)

当比较 MCMC（方程 (30)）和 ALS 解（方程 (22)）中模型参数的条件后验时，可以看到两者非常相似，即 \theta* = \mũ_\theta 且 \alpha = 1, \mu· = 0。区别在于 MCMC 从后验分布中采样，而 ALS 使用期望值。

MCMC 相对于 ALS 和 SGD 的一个主要优势是它允许将正则化参数 Θ_H 整合到模型中，从而避免了对这些超参数进行耗时的搜索。为了整合 Θ_H，贝叶斯 FM 模型通过在先验上放置分布（超先验分布）进行了扩展（图 2）。对于先验参数的每一对 (\mu_\theta, \lambda_\theta) \in Θ_H，假设 \lambda_\theta 服从 Gamma 分布，\mu_\theta 服从正态分布。即：

\mu_\pi^w ∼ N(\mu_0, \gamma_0 \lambda_\pi^w),   \lambda_\pi^w ∼ Γ(\alpha_\lambda, \beta_\lambda),   (34)
\mu_{f,\pi}^v ∼ N(\mu_0, \gamma_0 \lambda_{f,\pi}^v),   \lambda_{f,\pi}^v ∼ Γ(\alpha_\lambda, \beta_\lambda),   (35)

其中 \mu_0、\gamma_0 以及 \alpha_\lambda 和 \beta_\lambda 描述了超先验分布。最后，在 \alpha 上也放置一个 Gamma 分布：

\alpha ∼ Γ(\alpha_0, \beta_0).   (36)

总的来说，超先验导致以下新的参数 Θ_0：

Θ_0 := {\alpha_0, \beta_0, \alpha_\lambda, \beta_\lambda, \mu_0, \gamma_0}.   (37)

MCMC 允许将 Θ_H 整合到推断过程中，即 Θ_H 的值通过从它们对应的条件后验分布中采样而自动找到 [Freudenthaler et al. 2011]。

\alpha | y, X, Θ_0, Θ ∼ Γ( \alpha_0 + n/2, 1/2 \Sigma_{i=1}^{n} (y_i - ŷ(x_i|Θ))^2 + \beta_0 ),   (38)

\lambda_\pi· | Θ_0, Θ_H\{\lambda_\pi·}, Θ ∼ Γ( \alpha_\lambda + (p_\pi + 1)/2, 1/2 [ \Sigma_{j=1}^{p} \delta(\pi(j)=\pi)(\theta_j - \mu_\theta)^2 + \gamma_0(\mu_\pi· - \mu_0)^2 + \beta_\lambda ] ),   (39)

\mu_\pi· | Θ_0, Θ_H\{\lambda_\pi·}, Θ ∼ N( (p_\pi + \gamma_0)^{-1} [ \Sigma_{j=1}^{p} \delta(\pi(j)=\pi) \theta_j + \gamma_0 \mu_0 ], 1/((p_\pi + \gamma_0) \lambda_\pi·) ),   (40)

ACM Transactions on Intelligent Systems and Technology, 第 3 卷, 第 3 期, Article 57, 出版日期：2012 年 5 月.

Factorization Machines with libFM

57:11

其中

p_\pi := \Sigma_{j=1}^{p} \delta(\pi(j) = \pi).   (41)

**复杂度**。用于 MCMC 推断的吉布斯采样器在算法 3 中进行了概述，并且具有与 ALS 算法相同的复杂度。这直接源于观察到，对于这两种算法，在 MCMC 中要计算条件后验分布的和与 ALS 中要计算期望值的和是相同的。MCMC 的开销在于对 Θ_H 的推断，即计算后验（方程 (38)、(39) 和 (40)），但即使使用直接实现，这也是 O(k N_z(X))。

**超参数**。MCMC 的一个主要优势是正则化值 Θ_H 被自动确定。这是以引入超先验 Θ_0 的参数为代价的。然而，（1）超先验的数量 |Θ_H| 小于正则化参数的数量 |Θ_0|，并且（2）更重要的是，MCMC 通常对 Θ_0 的选择不敏感。也就是说，对 Θ_0 值的一个简单选择就能很好地工作。在 LIBFM 中，使用以下 Θ_0 的简单值：\alpha_0 = \beta_0 = \alpha_\lambda = \beta_\lambda = \gamma_0 = 1 且 \mu_0 = 0。

MCMC 仍然需要设置的唯一超参数是初始化 \sigma。通常在这里，甚至可以使用 0 的值（这对 ALS 和 SGD 是不可能的），因为 MCMC 的后验不确定性将识别出因子分解；然而，选择一个合适的值可以加速采样器。通常可以在前几个样本中看到初始化 \sigma 是否是一个好的选择。

**分类**。MCMC 算法 3 解决回归任务。它可以通过将正态分布的 ŷ 映射到概率 b(ŷ) \in [0, 1] 来扩展用于二分类，该概率定义了分类的伯努利分布 [Gelman et al. 2003]。这意味着，MCMC 算法将预测一个案例属于正类的概率。LIBFM 使用正态分布的 CDF 作为映射，即 b(z) = \Phi(z)，因为这样后验很容易采样。

对于分类，必须对算法 3 进行的唯一两个更改是：（1）预测时，ŷ 通过 \Phi 进行变换；（2）不是直接回归到 y，而是在每次迭代中从其具有截断正态分布的后验中采样回归目标 y'。

y'_i | x_i, y_i, Θ ∼ { N(ŷ(x_i, Θ), 1) \delta(y'_i < 0), 如果 y_i 为负类；N(ŷ(x_i, Θ), 1) \delta(y'_i \geq 0), 如果 y_i 为正类 }。   (42)

从这个分布中采样是高效的 [Robert 1995]。

如前所述，对于回归，ALS 可以看作是 MCMC 的简化，其中模型参数不是被采样，而是在每次更新中取其期望值。LIBFM 中提供了 ALS 的分类选项，它遵循相同的思路，并且对于使用 ALS 的分类，不是从截断正态中采样（如 MCMC 中所做的那样），而是计算截断正态的期望值。

### 3.5. 总结

LIBFM 中学习算法属性的概览可以在表 I 中找到。

ACM Transactions on Intelligent Systems and Technology, 第 3 卷, 第 3 期, Article 57, 出版日期：2012 年 5 月.

57:12

S. Rendle

**算法 3: 马尔可夫链蒙特卡洛推断 (MCMC)**
输入：训练数据 S, 测试数据 S_test, 初始化 \sigma
输出：测试案例的预测 ŷ_test
w_0 \leftarrow 0; w \leftarrow (0, ..., 0); V ∼ N(0, \sigma);
#samples \leftarrow 0;
重复
  ŷ \leftarrow 预测所有案例 S;
  e \leftarrow y - ŷ;
  更新超参数：
    从方程 (38) 采样 \alpha;
    for (\mu_\pi·, \lambda_\pi·) \in Θ_H do
      从方程 (39) 采样 \lambda_\pi·;
      从方程 (40) 采样 \mu_\pi·;
    end
  更新模型参数：
    从 N(\mũ_{w_0}, \sigmã_{w_0}^2) 采样 w_0;
    for l \in {1, ..., p} do
      从 N(\mũ_{w_l}, \sigmã_{w_l}^2) 采样 w_l;
      更新 e;
    end
    for f \in {1, ..., k} do
      初始化 $q_{·,f}$;
      for l \in {1, ..., p} do
        从 N(\mũ_{$v_{l,f}$}, \sigmã_{$v_{l,f}$}^2) 采样 $v_{l,f}$;
        更新 e, q;
      end
    end
  #samples \leftarrow #samples + 1;
  ŷ_test* \leftarrow 预测所有案例 S_test;
  ŷ_test \leftarrow ŷ_test + ŷ_test*;
直到满足停止准则;
ŷ_test \leftarrow (1/#samples) ŷ_test;

**表 I. LIBFM 中学习算法的属性**

| | SGD | ALS | MCMC |
|:---|:---|:---|:---|
| 算法 | | | |
| 运行时复杂度 | O(k N_z(X)) | O(k N_z(X)) | O(k N_z(X)) |
| 存储复杂度 | O(1) | O(n) | O(n) |
| 回归 | 是 | 是 | 是 |
| 分类 | 是 | 是 | 是 |
| 超参数 | 初始化, 正则化值 \lambda, 学习率 \eta | 初始化, 正则化值 \lambda | 初始化, 超先验（不敏感） |

## 4. 相关工作与因子分解机的应用

首先，展示了输入数据的示例，包括它们如何与其他专门的因子分解模型相关联。注意，FM 不限于此处呈现的选择。其次，将其他通用因子分解模型与 FM 进行比较。第三，将 FM 与多项式回归进行比较。

ACM Transactions on Intelligent Systems and Technology, 第 3 卷, 第 3 期, Article 57, 出版日期：2012 年 5 月.

Factorization Machines with libFM

57:13

### 4.1. 用因子分解机表达因子分解模型

在本节中，将通过与其他专门的最先进因子分解模型进行比较来讨论 FM 的通用性。这也展示了如何通过定义输入数据（即特征）来应用 FM。关键需要注意的是，在实践中，只需要定义特征向量 x；其余部分由 FM 隐式完成——既不需要显式重新表述模型方程，也不需要开发新的预测或学习算法。本节中对 FM 模型方程的分析只是为了展示与其他模型的理论关系。

#### 4.1.1. 矩阵分解

假设关于两个类别变量 U（例如，用户）和 I（例如，item）的数据应该在 FM 中使用。直接描述一个案例 (u, i) \in U × I 的方法是使用具有二元指示变量的特征向量 x \in R^(|U|+|I|)，即

(u, i) \rightarrow x = (0, ..., 0, 1, 0, ..., 0, 0, ..., 0, 1, 0, ..., 0),   (43)

其中 x 第一部分的第 u 个条目为 1，x 第二部分的第 i 个条目为 1，其余为 0（例如，见图 1 的前两组）。在 FM 中使用此数据时，FM 将完全等同于一个（有偏的）矩阵分解模型 [Paterek 2007; Srebro et al. 2005]：

ŷ(x) = ŷ(u, i) = w_0 + w_u + w_i + \Sigma_{f=1}^{k} $v_{u,f}$ $v_{i,f}$.   (44)

#### 4.1.2. 成对交互张量分解

如果需要描述三个类别变量，例如 U、I 和 T（例如，标签），一种直接的特征向量表示是 x \in R^(|U|+|I|+|T|)：

(u, i, t) \rightarrow x = (0, ..., 0, 1, 0, ..., 0, 0, ..., 0, 1, 0, ..., 0, 0, ..., 0, 1, 0, ..., 0).   (45)

使用此数据表示的 FM 将类似于成对交互张量分解模型（PITF）[Rendle and Schmidt-Thieme 2010]：

ŷ(x) = ŷ(u, i, t) = w_0 + w_u + w_i + w_t + \Sigma_{f=1}^{k} $v_{u,f}$ $v_{i,f}$ + \Sigma_{f=1}^{k} $v_{u,f}$ $v_{t,f}$ + \Sigma_{f=1}^{k} $v_{i,f}$ $v_{t,f}$.   (46)

这个 FM 与原始 PITF 的区别在于，此 FM 包含低阶交互，并在交互之间共享因子 V。除此之外，两种方法完全相同。

#### 4.1.3. SVD++ 和 FPMC

假设存在两个类别变量（例如，U 和 I）和一个集合类别变量（例如，P(L)）。此数据的一种简单表示是 x \in R^(|U|+|I|+|L|)：

(u, i, {l_1, ..., l_m}) \rightarrow x = (0, ..., 1, 0, ..., 0, ..., 1, 0, ..., 0, ..., 1/m, 0, ..., 1/m, 0, ...),   (47)

ACM Transactions on Intelligent Systems and Technology, 第 3 卷, 第 3 期, Article 57, 出版日期：2012 年 5 月.

57:14

S. Rendle

其中集合 {l_1, ..., l_m} 的 m 个元素中的每一个由一个非零值描述，例如，在相应列中为 1/m（例如，见图 1 的前三组）。使用此数据，FM 将等价于：

ŷ(x) = ŷ(u, i, {l_1, ..., l_m}) =

w_0 + w_u + w_i + ⟨v_u, v_i⟩ + (1/m) \Sigma_{j=1}^{m} $w_{l_j}$ + (1/m) \Sigma_{j=1}^{m} ⟨v_u, $v_{l_j}$⟩ + (1/m) \Sigma_{j=1}^{m} ⟨v_i, $v_{l_j}$⟩ + (1/m^2) \Sigma_{j=1}^{m} \Sigma_{j'>j}^{m} ⟨$v_{l_j}$, $v_{l_{j'}$}⟩。   (48)

SVD++ FPMC

如果使用隐式反馈作为 {l_1, ..., l_m} 的输入，刚刚勾画的 FM 几乎与 SVD++ 模型相同 [Koren 2008; Salakhutdinov and Mnih 2008b; Takács et al. 2009]。第一部分（标注为 SVD++）与原始 SVD++ [Koren 2008] 完全相同；第二部分（方程 (48) 的第二行）包含一些额外的交互。如果使用序列信息作为 {l_1, ..., l_m} 的输入，FM 与因子化个性化马尔可夫链（FPMC）[Rendle et al. 2010] 非常相似——特别是如果 FM 针对排序进行了优化（如 FPMC），FM 模型中存在但 FPMC 模型中不存在的几乎所有项都将消失（详见 [Rendle 2010]）。如果使用社交信息作为输入（例如，朋友），FM 与社会信任集成（STE）[Ma et al. 2011] 相似。

#### 4.1.4. BPTF 和 TimeSVD++

如果应该包含时间，最简单的方法是将时间视为一个类别变量（例如，每一天是一个级别）并应用与方程 (45) 相同的编码。使用此数据的 FM 类似于时间感知的 BPTF 模型 [Xiong et al. 2010]。区别在于 BPTF 在三个类别变量（用户、item、时间）上使用三元 PARAFAC 模型，而 FM 使用因子化的成对交互。此外，BPTF 在时间变量上有一个额外的正则化器。在 Freudenthaler et al. [2011] 中，已经表明 FM 确实比更复杂的 BPTF 模型效果更好。

另一种方法是每个用户使用一个单独的时间变量（即，使 用户-时间 交互显式化）。输入数据将是 x \in R^(|U|+|I|+|U|·|T|)，带有用户、item和用户特定日期指示符的二元指示符。使用此数据，FM 模型将等价于：

ŷ(x) = ŷ(u, i, t) = w_0 + w_u + w_i + $w_{(u,t)}$ + \Sigma_{f=1}^{k} $v_{u,f}$ $v_{i,f}$ + \Sigma_{f=1}^{k} $v_{(u,t),f}$ $v_{i,f}$ + \Sigma_{f=1}^{k} $v_{(u,t),f}$ $v_{u,f}$
= w_0 + w_u + $w_{(u,t)}$ + w_i + \Sigma_{f=1}^{k} ($v_{u,f}$ + $v_{(u,t),f}$) $v_{i,f}$ + \Sigma_{f=1}^{k} $v_{(u,t),f}$ $v_{u,f}$。   (49)
      b_u(t)                v_u,f(t)

该模型捕获了偏置 b_u(t) 和因子 v_u,f(t) 的"天特定变异性"，正如 Koren [2009b] 的 TimeSVD 模型一样。将 FM 的特征向量扩展为包含隐式反馈指示符（见第 4.1.3 节）和时间的线性指示符，将产生 TimeSVD++ 模型（图 1 的时间组是线性时间指示符的一个示例）。

ACM Transactions on Intelligent Systems and Technology, 第 3 卷, 第 3 期, Article 57, 出版日期：2012 年 5 月.

Factorization Machines with libFM

57:15

#### 4.1.5. 最近邻模型

当有其他数值测量可用时，例如，同一用户对item l_1, l_2, ... \in I 给出的其他评分 r_1, r_2, ... 等，这可以在特征向量 x \in R^(|I|+|I|) 中编码：

(i, {(r_1, l_1), ..., (r_m, l_m)}) \rightarrow x = (0, ..., 1, ..., 0, 0, ..., r_1/m, 0, ..., r_m/m, ..., 0).   (50)

使用此数据的 FM 模型将等价于：

因子化 KNN

ŷ(x) = ŷ(i, {(r_1, l_1), ..., (r_m, l_m)}) = w_0 + w_i + (1/m) \Sigma_{j=1}^{m} r_j $w_{l_j}$ + (1/m) \Sigma_{j=1}^{m} r_j ⟨v_i, $v_{l_j}$⟩ + (1/m^2) \Sigma_{j=1}^{m} \Sigma_{j'>j}^{m} r_j $r_{j'}$ ⟨$v_{l_j}$, $v_{l_{j'}$}⟩。   (51)

该模型类似于因子化最近邻模型 [Koren 2010]。

另一种可能的编码方法是对每个item使用单独的评分指示符，一个用于用户，即 x \in R^(|I|+|U|+|I|·|I|)。这意味着方程 (50) 中的评分指示符将在每个item的单独块中。使用此数据的 d = 1 阶 FM 将等价于：

ŷ(x) = ŷ(i, u, {(r_1, l_1), ..., (r_m, l_m)}) = w_0 + w_i + w_u + (1/m) \Sigma_{j=1}^{m} r_j $w_{i,l_j}$。   (52)

这种方法与 Koren [2008] 的非因子化最近邻模型相同。它可以与隐式反馈思想结合，从而产生 KNN++ 模型 [Koren 2008]。

#### 4.1.6. 属性感知模型

已有几次将关于用户和item的属性信息集成到推荐系统中的尝试。在 FM 中使用这些信息非常简单。一种直接的方法是将item（或用户）的属性（例如流派、演员等）添加到输入向量 x 中。

假设输入向量由这些item属性和一个用户指示变量组成：

(u, a_i_1, ..., a_i_m) \rightarrow x = (0, ..., 0, 1, 0, ..., 0, a_i_1, ..., a_i_m)。   (53)
                         |U|              item i 的属性

使用此数据，FM 将等价于：

ŷ(x) = ŷ(u, a_i_1, ..., a_i_m) = w_0 + w_u + \Sigma_{j=1}^{m} a_i_j w_j + \Sigma_{j=1}^{m} ⟨v_u, v_j⟩ a_i_j + \Sigma_{j=1}^{m} \Sigma_{j'>j}^{m} a_i_j a_$i_{j'}$ ⟨v_j, $v_{j'}$⟩。   (54)
                                          '属性映射'

这与 Gantner et al. [2010] 中的属性感知方法几乎相同，后者使用线性回归将item属性映射到因子（见方程 (54) 中突出显示的"属性映射"部分）——唯一的区别是 FM 包含偏置以及item属性之间的额外交互（例如，在流派和演员之间）。

如果将标准矩阵分解（方程 (43)）的输入向量扩展为包含用户的属性信息（例如，人口统计信息）和item的属性信息，则 FM

ACM Transactions on Intelligent Systems and Technology, 第 3 卷, 第 3 期, Article 57, 出版日期：2012 年 5 月.

57:16

S. Rendle

将对应于 Agarwal and Chen [2009] 中提出的属性感知模型。同样，区别在于 FM 包含用户和item属性内部的额外交互（例如，用户年龄和性别之间的交互）。

### 4.2. 其他通用因子分解模型

还有其他尝试实现更通用的因子分解模型。在 Agarwal and Chen [2009] 中，矩阵分解模型通过回归先验进行了扩展。即，正态分布因子先验的均值是一个线性回归模型。FM 可以模仿这种方法，因为对于任何使用正态分布先验的层次模型，先验的均值（因此也包括先验均值的线性回归模型）可以作为协变量添加到特征向量中。另一方面，带回归先验的 MF 比 FM 受限得多，因为 MF 本身仅限于两个类别变量的交互，因此，带回归先验的 MF 模型不适用于涉及两个以上变量交互的任务，例如标签推荐或上下文感知推荐。FM 包含任意数量变量之间的（成对）交互（也不限于类别变量）。

SVDfeature [Chen et al. 2011] 是另一种通用因子分解模型。与 Agarwal and Chen [2009] 类似，在 SVDfeature 中，矩阵分解模型被扩展为包含因子和偏置项的线性回归项。然而，与 FM 相比，它具有与 Agarwal and Chen [2009] 相同的缺陷：只有两个类别变量之间的交互可以被因子化。这意味着它无法模仿最先进的上下文感知推荐器、标签推荐器等。此外，对于 SVDfeature，只提出了 SGD 学习，而 LIBFM 提供 MCMC 推断，后者应用起来更简单，因为没有学习率，并且正则化值被自动确定。SVDfeature 相对于 LIBFM 的一个优势是，由于其更受限的模型方程，它在因子回归项中有一个改进的学习算法（遵循 Koren [2008]）来加速学习。

### 4.3. 与多项式回归的关系

在 Rendle [2010] 中，已经证明 FM 可以被视为使用因子化参数矩阵的多项式回归（或具有非齐次多项式核的 SVM）。d = 2 阶的多项式回归可以定义为

ŷ_{PR}(x) := w_0 + \Sigma_{j=1}^{p} w_j x_j + \Sigma_{j=1}^{p} \Sigma_{j'=j}^{p} $w_{j,j'}$ x_j $x_{j'}$,   (55)

其中模型参数

w_0 \in R, w \in R^p, W \in R^(p×p).   (56)

将此与 FM 模型（方程 (1)）进行比较，可以看到 FM 对成对交互使用因子化，而多项式回归对每个成对交互使用独立参数 $w_{j,j'}$。这一区别对于 FM 在稀疏设置中的成功至关重要，例如推荐系统或其他涉及大领域类别变量的预测问题。FM 可以估计对 (j, j') 的成对交互 $w_{j,j'}$，即使关于该对没有或只有很少的观测存在，

ACM Transactions on Intelligent Systems and Technology, 第 3 卷, 第 3 期, Article 57, 出版日期：2012 年 5 月.

Factorization Machines with libFM

57:17

因为做了低秩假设（$w_{j,j'}$ \approx ⟨v_j, $v_{j'}$⟩），即假设对 (j, j') 和 (j, j*) 的交互有共同之处。在多项式回归中，两个对是完全独立的（先验地）。

## 5. LIBFM 软件

LIBFM¹ 是因子分解机的一个实现。它包含第 3 节中描述的用于回归和分类任务的 SGD、ALS 和 MCMC 算法。实现了 d = 2 阶的 FM。

### 5.1. 数据格式

LIBFM 的输入数据格式与 SVMlight [Joachims 1999] 和 LIBSVM [Chang and Lin 2011] 相同。对于不适合主存的大规模数据，LIBFM 提供了一种二进制数据格式，其中只需要将部分数据保留在主存中。提供了从标准文本格式到二进制数据格式的转换器。

### 5.2. 示例

所有主要选项都可以通过易于使用的命令行界面获取。使用 MCMC 推断学习一个数据集的示例调用为：

./libFM -method mcmc -task r -dim '1;1;8' -init_stdev 0.1 -iter 100 -test ml1m-test.libfm -train ml1m-train.libfm -out ml1m-test.pred,

其中 dim 指定因子分解维度：0/1 表示是否包含 w_0，0/1 表示是否包含 w，以及 k \in N（此处 k = 8）表示 V 的维度。init_stdev 是初始化的标准差，即算法 3 中的 \sigma。iter 是抽取的样本数量。

### 5.3. 参数设置

下面给出了一些将 LIBFM 应用于预测问题的实用提示。

(1) 对于缺乏经验的用户，建议使用 MCMC 推断，因为它是最简单易用的方法。
(2) 当为新的数据集构建预测模型时，应该从较低的因子分解维度（例如，k = 8）开始，首先确定初始化的标准差（-init_stdev），因为恰当的值会加速 MCMC 采样器。
(3) 应测试几个 init_stdev 的值（例如，0.1、0.2、0.5、1.0）。通过监控训练误差，或者更好的是使用保留集进行验证，可以在前几次迭代中快速看到效果。
(4) 在确定了适当的 init_stdev 之后，可以使用大量的迭代次数和更大的因子分解维度 k 来运行 MCMC。准确性和收敛性可以在 libFM 的输出上进行监控。

对于 MCMC，不需要指定其他超参数。其他方法（ALS 和 SGD）需要调整更多的超参数（见第 3 节）。

### 5.4. 排序

LIBFM 还包含基于成对分类 [Rendle et al. 2009] 的针对排序优化 FM 模型的方法 [Liu and Yang 2008]。排序功能不能通过命令行使用，但可以嵌入到现有软件中。软件工具 Tag Recommender（也提供源代码）中提供了一个嵌入 LIBFM 的示例。

¹ 源代码可从 http://www.libfm.org/ 获取。

ACM Transactions on Intelligent Systems and Technology, 第 3 卷, 第 3 期, Article 57, 出版日期：2012 年 5 月.

57:18

S. Rendle

图 3. 使用 MF（见方程 (43)）和 KNN（见方程 (52)）输入数据的 LIBFM（SGD 和 MCMC 学习）的预测误差。(a) 与 MF 方法 PMF (SGD) [Salakhutdinov and Mnih 2008b]、BPMF (MCMC) [Salakhutdinov and Mnih 2008a] 和 MF (SGD) [Koren 2008] 进行比较。(b) 与 Koren [2008] 的相应 KNN 方法进行比较。

## 6. 评估

在第 4.1 节中，展示了 FM 能够模仿许多因子分解模型。现在将通过将 LIBFM 实现与几个经过充分研究的因子分解模型进行实证比较来证实这一点。成功将通过回归任务的均方根误差（RMSE）和排序任务的 F1 度量来衡量（参见 Gunawardana and Shani [2009] 了解推荐系统评估指标的总结）。

### 6.1. 评分预测

在推荐系统中，研究最充分的数据集是 Netflix 挑战赛²，其中包含约 480,000 名用户对 17,770 个item的大约 100,000,000 个评分。我们报告的所有结果均在 Netflix quiz 集上获得（即与 Netflix 挑战赛公共排行榜上相同的测试集）。

#### 6.1.1. 矩阵分解（MF）

Netflix 上最成功的方法基于矩阵分解（例如 [Jahrer et al. 2010; Koren 2009a; Takács et al. 2009])。对于 MF，已经提出了许多不同的变体和学习方法，例如 ALS [Pilászy et al. 2010]、MCMC [Salakhutdinov and Mnih 2008a]、变分贝叶斯 [Lim and Teh 2007; Stern et al. 2009]，但主要是 SGD 变体（例如 [Koren 2008; Salakhutdinov and Mnih 2008b])。因此，即使对于简单的 MF 模型，报告的预测质量也差异很大。我们希望通过使用 MF 指示符（见方程 (43)）设置 FM（这等价于有偏 MF）来研究 LIBFM 的学习方法有多好。在这种设置下，所有比较的方法共享相同的模型，但在学习算法和实现上有所不同。

图 3(a) 展示了 LIBFM（使用 SGD 和 MCMC³）与 PMF 的 SGD 方法 [Salakhutdinov and Mnih 2008b] 和 [Koren 2008] 的 MF (SGD) 方法，以及使用 MCMC 推断的 BPMF 方法 [Salakhutdinov

² http://www.netflixprize.com/
³ FM (SGD) 结果来自 Rendle [2012]，FM (MCMC) 结果来自 Freudenthaler et al. [2011]。

ACM Transactions on Intelligent Systems and Technology, 第 3 卷, 第 3 期, Article 57, 出版日期：2012 年 5 月.

Factorization Machines with libFM

57:19

图 4. 上下文感知推荐的 LIBFM 与 Multiverse Recommendation [Karatzoglou et al. 2010] 的比较。将 LIBFM 在标签推荐任务上与 ECML/PKDD Discovery Challenge 2009 任务 2 的最佳四种方法进行比较。

and Mnih 2008a] 的比较。对于 LIBFM 的 SGD，我们使用了 Koren [2008] 中为相关 SVD++ 模型报告的正则化值（\lambda_\theta = 0.04）。可以看出 MCMC 方法具有最低的误差，并且 LIBFM 的 MCMC 采样器略优于 BPMF 模型的采样器。

#### 6.1.2. 最近邻模型

传统上，最近邻模型在推荐系统社区中吸引了大量研究（例如 [Linden et al. 2003; Sarwar et al. 2001; Zheng and Xie 2011])。在 Netflix 挑战赛中，性能最好的邻域方法基于将item之间的相似度视为学习的模型参数，即 KNN（方程 (52)）[Koren 2008] 和因子化 KNN（方程 (51)）[Koren 2010]。我们再次希望了解 LIBFM 仅通过特征工程就能在多大程度上模仿这些模型。我们为 LIBFM 设置输入数据，使得 FM 模型对应于 Koren [2008] 中描述的 KNN 和 KNN++（即带有额外的隐式指示符）。我们使用相同的剪枝协议将邻居限制为 256 个，对于 SGD，我们使用与 Koren [2008] 中报告相同的正则化（\lambda_\theta = 0.002）。图 3(b) 显示，使用 MCMC 和 SGD 的 LIBFM 达到了与 Koren [2008] 方法相当的质量。

### 6.2. 上下文感知推荐

其次，LIBFM 已经在上下文感知推荐问题上进行了研究 [Rendle et al. 2011]。在上下文感知推荐中，除了用户和item之外，还有关于评分事件的其他信息可用，例如，用户评分时的位置、心情等。由于 FM 可以处理任意数量的特征，它们可以很容易地应用于此任务。图 4(a)⁴ 展示了使用 ALS 和 MCMC 的 LIBFM 与最先进方法 Multiverse Recommendation [Karatzoglou et al. 2010] 的比较，后者优于其他上下文感知方法，如item分割 [Baltrunas and Ricci 2009] 和 Adomavicius et al. [2005] 的多维方法。

⁴ 关于实验设置的详细信息请参见 Karatzoglou et al. [2010] 和 Rendle et al. [2011]。

ACM Transactions on Intelligent Systems and Technology, 第 3 卷, 第 3 期, Article 57, 出版日期：2012 年 5 月.

57:20

S. Rendle

### 6.3. 标签推荐

最后一个实验展示了 LIBFM 对排序的适用性。我们使用方程 (45) 中的输入数据，将 LIBFM 用于标签推荐任务（例如 Lipczak and Milios [2011]）。使用此数据，LIBFM 模仿了 PITF 模型 [Rendle and Schmidt-Thieme 2010]，该模型是 ECML/PKDD Discovery Challenge 2009 任务 2 中表现最好的方法⁵。图 4(b) 展示了 LIBFM 与 PITF 以及 Discovery Challenge 中排名第二至第四的最佳模型的预测质量比较：关系分类 [Marinho et al. 2009] 以及第三名 [Lipczak et al. 2009] 和第四名 [Zhang et al. 2009] 的模型。

## 7. 结论与未来工作

因子分解机（FM）将特征工程的灵活性与因子分解模型结合起来。本文总结了关于 FM 的最新研究，并提出了三种基于 SGD、ALS 和 MCMC 的高效推断方法。还介绍了扩展，其中包括 MCMC 和 ALS 的分类，以及变量的分组。

从复杂度和表达能力的理论角度，以及通过实证评估，讨论了 FM 的性质。已经证明 FM 可以模仿几种专门的因子分解模型——当然，FM 并不限于这些示例。实证结果表明，所描述的 FM 推断算法的预测质量可与推荐系统领域中专门模型的最佳推断方法相媲美。总的来说，这意味着 FM 的通用性并非以低预测精度或高计算复杂度为代价。所有提出的算法都在公开可用的软件工具 LIBFM 中实现。

FM 的未来工作有多个方向。首先，由于 FM 的通用性，它们有望对广泛的预测问题感兴趣，特别是涉及大领域类别变量的问题可能受益于 FM。使用 LIBFM 在这些问题上研究 FM 非常有意义。其次，FM 推断方法的复杂度可以降低，因为迄今为止提出的算法没有利用输入数据中的重复模式，而这些模式可以被利用来进一步加速。第三，软件实现 LIBFM 可以通过高阶交互（d \geq 3）进行扩展。

## 致谢

我要感谢 Christoph Freudenthaler 的许多富有成果的讨论和他宝贵的意见。

## 参考文献

ADOMAVICIUS, G., SANKARANARAYANAN, R., SEN, S., AND TUZHILIN, A. 2005. Incorporating contextual information in recommender systems using a multidimensional approach. ACM Trans. Info. Syst. 23, 1, 103–145.

AGARWAL, D. AND CHEN, B.-C. 2009. Regression-based latent factor models. In Proceedings of the 15th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD'09). ACM, New York, NY, 19–28.

BALTRUNAS, L. AND RICCI, F. 2009. Context-based splitting of item ratings in collaborative filtering. In Proceedings of the third ACM Conference on Recommender Systems (RecSys'09). ACM, New York, NY, 245–248.

⁵ 获胜模型是几个 PITF 模型和后处理步骤的集成。为了公平比较，这里报告的是没有后处理的单个 PITF 模型的 F1 分数。集成的后处理也可以以相同的方式用于 LIBFM 的预测。

ACM Transactions on Intelligent Systems and Technology, 第 3 卷, 第 3 期, Article 57, 出版日期：2012 年 5 月.

Factorization Machines with libFM

57:21

CHANG, C.-C. AND LIN, C.-J. 2011. Libsvm: A library for support vector machines. ACM Trans. Intell. Syst. Technol. 2, 27:1–27:27.

CHEN, T., ZHENG, Z., LU, Q., ZHANG, W., AND YU, Y. 2011. Feature-based matrix factorization. Tech. rep. APEX-TR-2011-07-11, Apex Data & Knowledge Management Lab, Shanghai Jiao Tong University.

FREUDENTHALER, C., SCHMIDT-THIEME, L., AND RENDLE, S. 2011. Bayesian factorization machines. In Proceedings of the NIPS Workshop on Sparse Representation and Low-rank Approximation.

GANTNER, Z., DRUMOND, L., FREUDENTHALER, C., RENDLE, S., AND LARS, S.-T. 2010. Learning attribute-to-feature mappings for cold-start recommendations. In Proceedings of the IEEE International Conference on Data Mining (ICDM'10). IEEE Computer Society, Los Alamintos, CA, 176–185.

GELMAN, A., CARLIN, J. B., STERN, H. S., AND RUBIN, D. B. 2003. Bayesian Data Analysis 2nd Ed. Chapman and Hall/CRC.

GUNAWARDANA, A. AND SHANI, G. 2009. A survey of accuracy evaluation metrics of recommendation tasks. J. Mach. Learn. Res. 10, 2935–2962.

HALL, M., FRANK, E., HOLMES, G., PFAHRINGER, B., REUTEMANN, P., AND WITTEN, I. H. 2009. The weka data mining software: An update. SIGKDD Explor. Newsl. 11, 10–18.

HARSHMAN, R. A. 1970. Foundations of the parafac procedure: Models and conditions for an 'exploratory' multimodal factor analysis. UCLA Working Papers in Phonetics, 1–84.

JAHRER, M., TÖSCHER, A., AND LEGENSTEIN, R. 2010. Combining predictions for accurate recommender systems. In Proceedings of the 16th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD'10). ACM, New York, NY, 693–702.

JOACHIMS, T. 1999. Making Large-Scale Support Vector Machine Learning Practical. MIT Press, Cambridge, MA, 169–184.

KARATZOGLOU, A., AMATRIAIN, X., BALTRUNAS, L., AND OLIVER, N. 2010. Multiverse recommendation: n-dimensional tensor factorization for context-aware collaborative filtering. In Proceedings of the 4th ACM Conference on Recommender Systems (RecSys'10). ACM, New York, NY, 79–86.

KOREN, Y. 2008. Factorization meets the neighborhood: A multifaceted collaborative filtering model. In Proceeding of the 14th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD'08). ACM, New York, NY, 426–434.

KOREN, Y. 2009a. The bellkor solution to the Netflix grand prize.

KOREN, Y. 2009b. Collaborative filtering with temporal dynamics. In Proceedings of the 15th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD'09). ACM, New York, NY, 447–456.

KOREN, Y. 2010. Factor in the neighbors: Scalable and accurate collaborative filtering. ACM Trans. Knowl. Discov. Data 4, 1:1–1:24.

LIM, Y. J. AND TEH, Y. W. 2007. Variational Bayesian approach to movie rating prediction. In Proceedings of the KDD Cup and Workshop.

LINDEN, G., SMITH, B., AND YORK, J. 2003. Amazon.com recommendations: Item-to-item collaborative filtering. Inter. Comput. IEEE 7, 1, 76–80.

LIPCZAK, M., HU, Y., KOLLET, Y., AND MILIOS, E. 2009. Tag sources for recommendation in collaborative tagging systems. In Proceedings of the ECML-PKDD Discovery Challenge Workshop.

LIPCZAK, M. AND MILIOS, E. 2011. Efficient tag recommendation for real-life data. ACM Trans. Intell. Syst. Technol. 3, 1, 2:1–2:21.

LIU, N. N. AND YANG, Q. 2008. Eigenrank: A ranking-oriented approach to collaborative filtering. In Proceedings of the 31st Annual International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR'08). ACM, New York, NY, 83–90.

MA, H., KING, I., AND LYU, M. R. 2011. Learning to recommend with explicit and implicit social relations. ACM Trans. Intell. Syst. Technol. Article 29.

MARINHO, L. B., PREISACH, C., AND SCHMIDT-THIEME, L. 2009. Relational classification for personalized tag recommendation. In Proceedings of the ECML-PKDD Discovery Challenge Workshop.

PATEREK, A. 2007. Improving regularized singular value decomposition for collaborative filtering. In Proceedings of the KDD Cup Workshop 13th ACM International Conference on Knowledge Discovery and Data Mining (SIGKDD'07). 39–42.

PILÁSZY, I., ZIBRICZKY, D., AND TIKK, D. 2010. Fast als-based matrix factorization for explicit and implicit feedback datasets. In Proceedings of the 4th ACM Conference on Recommender Systems (RecSys'10). ACM, New York, NY, 71–78.

RENDLE, S. 2010. Factorization machines. In Proceedings of the 10th IEEE International Conference on Data Mining. IEEE Computer Society.

ACM Transactions on Intelligent Systems and Technology, 第 3 卷, 第 3 期, Article 57, 出版日期：2012 年 5 月.

57:22

S. Rendle

RENDLE, S. 2012. Learning recommender systems with adaptive regularization. In Proceedings of the 5th ACM International Conference on Web Search and Data Mining (WSDM'12). ACM, New York, NY, 133–142.

RENDLE, S., FREUDENTHALER, C., GANTNER, Z., AND SCHMIDT-THIEME, L. 2009. BPR: Bayesian personalized ranking from implicit feedback. In Proceedings of the 25th Conference on Uncertainty in Artificial Intelligence (UAI09).

RENDLE, S., FREUDENTHALER, C., AND SCHMIDT-THIEME, L. 2010. Factorizing personalized Markov chains for next-basket recommendation. In Proceedings of the 19th International Conference on World Wide Web (WWW'10). ACM, New York, NY, 811–820.

RENDLE, S., GANTNER, Z., FREUDENTHALER, C., AND SCHMIDT-THIEME, L. 2011. Fast context-aware recommendations with factorization machines. In Proceedings of the 34th ACM SIGIR Conference on Research and Development in Information Retrieval.

RENDLE, S. AND SCHMIDT-THIEME, L. 2010. Pairwise interaction tensor factorization for personalized tag recommendation. In Proceedings of the third ACM International Conference on Web Search and Data Mining (WSDM'10). ACM, New York, NY, 81–90.

ROBERT, C. P. 1995. Simulation of truncated normal variables. Stat. Comput. 5, 121–125.

SALAKHUTDINOV, R. AND MNIH, A. 2008a. Bayesian probabilistic matrix factorization using Markov chain Monte Carlo. In Proceedings of the 25th International Conference on Machine Learning.

SALAKHUTDINOV, R. AND MNIH, A. 2008b. Probabilistic matrix factorization. In Advances in Neural Information Processing Systems 20.

SARWAR, B., KARYPIS, G., KONSTAN, J., AND RIEDL, J. 2001. Item-based collaborative filtering recommendation algorithms. In Proceedings of the 10th International Conference on World Wide Web. ACM Press, New York, NY, 285–295.

SREBRO, N. AND JAAKKOLA, T. 2003. Weighted low rank approximation. In Proceedings of the 20th International Conference on Machine Learning (ICML'03).

SREBRO, N., RENNIE, J. D. M., AND JAAKOLA, T. S. 2005. Maximum-margin matrix factorization. In Advances in Neural Information Processing Systems 17, MIT 1329–1336.

STERN, D. H., HERBRICH, R., AND GRAEPEL, T. 2009. Matchbox: Large-scale online Bayesian recommendations. In Proceedings of the 18th International Conference on World Wide Web (WWW'09). ACM, New York, NY, 111–120.

TAKÁCS, G., PILÁSZY, I., NÉMETH, B., AND TIKK, D. 2009. Scalable collaborative filtering approaches for large recommender systems. J. Mach. Learn. Res. 10, 623–656.

TUCKER, L. 1966. Some mathematical notes on three-mode factor analysis. Psychometrika 31, 279–311.

XIONG, L., CHEN, X., HUANG, T.-K., SCHNEIDER, J., AND CARBONELL, J. G. 2010. Temporal collaborative filtering with Bayesian probabilistic tensor factorization. In Proceedings of the SIAM International Conference on Data Mining (SIAM). 211–222.

ZHANG, N., ZHANG, Y., AND TANG, J. 2009. A tag recommendation system based on contents. In Proceedings of the ECML-PKDD Discovery Challenge Workshop.

ZHENG, Y. AND XIE, X. 2011. Learning travel recommendations from user-generated gps traces. ACM Trans. Intell. Syst. Technol. 2, 2:1–2:29.

2012 年 1 月收到；2012 年 1 月修订；2012 年 2 月接受

ACM Transactions on Intelligent Systems and Technology, 第 3 卷, 第 3 期, Article 57, 出版日期：2012 年 5 月.
