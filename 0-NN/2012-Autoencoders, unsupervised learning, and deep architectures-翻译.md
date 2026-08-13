# 自编码器、无监督学习与深度架构

> Pierre Baldi | University of California, Irvine

本文提出一个研究线性和非线性自编码器的一般数学框架，**推导出最极端形式的非线性自编码器——布尔自编码器的解析处理，并证明其学习等价于聚类问题**。

核心内容：

- 提出$n/p/n$自编码器的一般数学框架，涵盖线性与非线性自编码器
- 证明布尔自编码器的学习等价于超立方体上的聚类问题，当聚类数$k$与训练样本数$m$满足$k \sim m^\epsilon$时为NP难问题
- 建立自编码器与聚类、Hebbian学习、信息论之间的基本联系
- 从垂直组合（堆叠）和水平组合两个角度分析深度架构中的自编码器

关键发现：

- 线性自编码器的误差景观无局部最小值，所有临界点对应协方差矩阵特征向量子空间上的投影，全局最小值等价于**主成分分析（PCA）**
- 布尔自编码器的全局最优解对应将输入空间划分为$2^p$个Voronoi聚类，临界点处的$AB$是投影算子
- 无监督预训练的本质是**聚类**，Hebbian学习和自编码器是同一枚硬币的三个面
- 深度架构的真正深度可能是常数或对数级，而非多项式级，这与感官神经元回路的观察一致

---

## 摘要

自编码器在无监督学习和用于迁移学习及其他任务的深度架构中发挥着基础性作用。尽管它们具有基础性作用，但只有实数域上的线性自编码器得到了解析求解。本文提出了一个研究线性和非线性自编码器的一般数学框架。该框架允许对最极端的非线性自编码器——布尔自编码器进行解析处理。布尔自编码器中的学习等价于一个聚类问题，当聚类数较小时可在多项式时间内求解，当聚类数较大时变为NP完全问题。该框架揭示了不同类型自编码器的性质、它们的学习复杂度、在深度架构中的水平和垂直可组合性、临界点，以及它们与聚类、Hebbian学习和信息论的基本联系。

关键词：自编码器、无监督学习、压缩、聚类、主成分分析、布尔、复杂度、深度架构、Hebbian学习、信息论

## 1 引言

自编码器是简单的学习电路，旨在以尽可能少的失真将输入转换为输出。虽然概念简单，但它们在机器学习中扮演着重要角色。自编码器最早由Hinton和PDP小组在1980年代引入（Rumelhart等人，1986），用于解决"无教师反向传播"问题，即使用输入数据作为教师。与Hebbian学习规则（Hebb，1949；Oja，1982）一起，自编码器提供了无监督学习的基本范式之一，并开始解决局部生化事件诱导的突触变化如何以自组织方式协调以产生全局学习和智能行为的奥秘。

近年来，自编码器在"深度架构"方法中再次成为焦点（Hinton等人，2006；Hinton和Salakhutdinov，2006；Bengio和LeCun，2007；Erhan等人，2010），其中自编码器特别是受限玻尔兹曼机（RBM）的形式，被堆叠并以无监督方式自底向上训练，然后进行监督学习阶段来训练顶层并微调整个架构。自底向上阶段对最终任务是不可知的，因此显然可以用于迁移学习方法。这些深度架构已在许多具有挑战性的分类和回归问题上取得了最先进的结果。

尽管它们引起了广泛的兴趣，但除了一些例外（Baldi和Hornik，1988；Sutskever和Hinton，2008；Montufar和Ay，2011），迄今为止对自编码器和深度架构的理论理解仍然有限。使用"深度"这个术语可能造成了额外的混淆。从计算机科学的角度来看，深度架构应该有$n^\alpha$个多项式大小的层，其中$\alpha > 0$是一个小常数，$n$是输入向量的大小（参见Clote和Kranakis（2002）及其参考文献）。但在Hinton等人（2006）和Hinton和Salakhutdinov（2006）描述的架构中，情况并非如此，它们似乎具有常数或至多对数深度，对于计算机视觉、语音识别和其他典型问题中使用的典型$n$值，有限深度和对数深度之间的区别几乎是不可能的。因此，这项工作的主要动机是推导出对自编码器更好的理论理解，希望能更好地洞察无监督学习和深度架构的本质。

如果关于深度架构的一般理论结果存在，它们不太可能依赖于特定的硬件实现，如RBM。类似的结果应该也适用于替代的或更一般形式的计算。因此，本文提出的策略是引入一个一般框架并研究不同类型的自编码器电路，特别是布尔自编码器，它可以被视为最极端形式的非线性自编码器。预期某些自编码器和深度架构的属性可能在更简单的硬件实现中更容易在数学上识别和理解，并且对不同类型自编码器的研究可能通过识别共同属性来促进抽象和泛化。

为此，我们在第2节开始描述研究自编码器的相当一般的框架。在第3节中，我们回顾并扩展了关于线性自编码器的已知结果。从深度架构的角度，我们审视了诸如垂直组合（堆叠）以及临界点与循环稳定性（将输出反馈回输入层）之间连接等新属性。在第4节中，我们研究布尔自编码器，并证明了若干属性，包括它们与聚类的基本联系。在第5节中，我们讨论布尔自编码器学习的复杂度。在第6节中，我们研究具有大隐藏层的自编码器，并引入自编码器水平组合的概念。在第7节中，我们讨论其他类型的自编码器和推广。最后，在第8节中，我们总结结果及其对深度架构理论的可能影响。

## 2 一般自编码器框架

为推导一个相当一般的框架，$n/p/n$自编码器（图1）由元组$n, p, m, F, G, A, B, X, \Delta$定义，其中：

1. $F$和$G$是集合。
2. $n$和$p$是正整数。这里我们主要考虑$0 < p < n$的情况。
3. $A$是从$G^p$到$F^n$的函数类。
4. $B$是从$F^n$到$G^p$的函数类。
5. $X = \{x_1, \ldots, x_m\}$是$F^n$中的$m$个（训练）向量集合。当存在外部目标时，我们令$Y = \{y_1, \ldots, y_m\}$表示$F^n$中对应的目标向量集合。
6. $\Delta$是定义在$F^n$上的不相似性或失真函数（例如$L_p$范数、Hamming距离）。

对于任意$A \in \mathcal{A}$和$B \in \mathcal{B}$，自编码器将输入向量$x \in F^n$转换为输出向量$A \circ B(x) \in F^n$（图1）。对应的自编码器问题是在$A \in \mathcal{A}$和$B \in \mathcal{B}$中找到最小化整体失真函数的解：

$$
\min E(A, B) = \min_{A,B} \sum_{t=1}^{m} E(x_t) = \min_{A,B} \sum_{t=1}^{m} \Delta(A \circ B(x_t), x_t) \qquad (1)
$$

在非自关联情况下，当提供外部目标$y_t$时，最小化问题变为：

$$
\min E(A, B) = \min_{A,B} \sum_{t=1}^{m} E(x_t, y_t) = \min_{A,B} \sum_{t=1}^{m} \Delta(A \circ B(x_t), y_t) \qquad (2)
$$

注意$p < n$对应于自编码器尝试实现某种形式的压缩或特征提取的机制。$p \geq n$的情况在论文末尾讨论。

显然，从这个一般框架出发，可以推导出不同类型的自编码器，这取决于集合$F$和$G$的选择、变换类$A$和$B$、失真函数$\Delta$，以及是否存在额外约束（如正则化）。据我们所知，神经网络自编码器最早由PDP小组作为此定义的特例引入，所有向量分量在$F = G = \mathbb{R}$中，$A$和$B$对应于矩阵乘法后跟非线性sigmoid变换，使用$L_2^2$误差函数。[对于回归问题，非线性sigmoid变换通常仅在隐藏层中使用]。作为对此情况的近似，我们在下一节研究$F = G = \mathbb{R}$的线性情况。更一般地，线性自编码器对应于$F$和$G$是域且$A$和$B$是线性变换类的情况，因此$A$和$B$分别是$p \times n$和$n \times p$大小的矩阵。$F = G = \mathbb{R}$且$\Delta$为欧氏距离平方的线性实数情况已在Baldi和Hornik（1988）中讨论（另见Bourlard和Kamp（1988））。最近该理论也扩展到了复数域线性自编码器（Baldi等人，2011）。

## 3 线性自编码器

我们部分重述Baldi和Hornik（1988）对$\Delta = L_2^2$的线性实数情况推导的结果，但以旨在突出与其他类型自编码器联系的方式组织它们，并从深度架构的角度扩展它们的结果。我们用$A^t$表示任何矩阵$A$的转置。

**1) 群不变性。** 每个解在乘以可逆$p \times p$矩阵$C$的范围内定义，或者等价地在隐藏层坐标变换的范围内定义。这是显而易见的，因为$AC^{-1}CB = AB$。

**2) 问题复杂度。** 虽然代价函数是二次的且所有操作都是线性的，但整体问题不是凸的，因为隐藏层将整体变换的秩限制为最多$p$，而秩为$p$或更低的矩阵集合不是凸的。然而，$\mathbb{R}$上的线性自编码器问题可以解析求解。注意在这种情况下，人们感兴趣的是寻找恒等函数的低秩近似。

**3) 固定层解。** 如果$A$固定，或者$B$固定，问题变为凸的。当$A$固定时，假设$A$的秩为$p$且数据协方差矩阵$\Sigma_{XX} = \sum_i x_i x_i^t$可逆，则在最优点$B^* = B(A) = (A^t A)^{-1} A^t$。当$B$固定时，假设$B$的秩为$p$且$\Sigma_{XX}$可逆，则在最优点$A^* = A(B) = \Sigma_{XX} B^t (B \Sigma_{XX} B^t)^{-1}$。

**4) $E$的景观。** $E$的整体景观没有局部最小值。梯度为零的所有临界点对应于投影到与协方差矩阵$\Sigma_{XX}$的特征向量子集相关的子空间上。投影到与$p$个最大特征值相关的子空间上对应于全局最小值和主成分分析（PCA）。所有其他临界点，对应于投影到与其他特征值集合相关的子空间上，都是鞍点。更精确地说，如果$I = i_1, \ldots, i_p$（$1 \leq i_1 < \ldots < i_p \leq n$）是任何有序索引列表，令$U_I = [u_1, \ldots, u_p]$表示由$\Sigma_{XX}$的正交归一特征向量组成的矩阵，对应特征值$\lambda_{i_1}, \ldots, \lambda_{i_p}$。则两个秩为$p$的矩阵$A$和$B$定义一个临界点，当且仅当存在一个集合$I$和一个可逆$p \times p$矩阵$C$，使得$A = U_I C$，$B = C^{-1} U_I^t$，且$W = AB = P_{U_I}$，其中$P_{U_I}$是到$U_I$列张成的子空间上的正交投影。在全局最小值处，假设$C = I$，隐藏层中的活动由点积$u_1^t x, \ldots, u_p^t x$给出，对应于$x$沿$\Sigma_{XX}$的前$p$个特征向量的坐标。

**5) 聚类。** 全局最小值执行一种关于$\ker B$（$B$的核）的超平面聚类形式。对于任何给定向量$x$，所有形如$x + \ker(B)$的向量都被映射到同一个向量$y = AB(x) = AB(x + \ker B)$。

**6) 循环稳定性。** 在任何临界点处，$AB$是投影算子，因此循环输出在第一次通过时是稳定的：$(AB)^n(x) = AB(x) = U_I U_I^t(x)$，对于任意$n \geq 1$。

**7) 泛化。** 在任何临界点处，对于任意$x$，$AB(x)$等于$x$到相应子空间的投影，相应的误差可以简单地表示为$x$到投影空间的平方距离。

**8) 垂直组合。** 如果在输入层和隐藏层之间或隐藏层和输出层之间引入额外的秩大于等于$p$的矩阵，$E$的全局最小值保持不变。因此，引入这样的矩阵不会减少整体失真。然而，如果出于其他原因引入这样的矩阵，则存在组合定律，使得具有矩阵堆叠的深度自编码器的最优解可以通过组合浅层自编码器的最优解来获得。更精确地说，考虑具有层大小$n/p_1/p/p_1/n$的自编码器网络（图2），其中$n > p_1 > p$。则该网络的最优解可以通过首先计算$n/p_1/n$自编码器网络的最优解，并使用第一个网络隐藏层中的活动作为第二个网络的训练集，与$p_1/p/p_1$自编码器网络的最优解组合来获得，这与堆叠RBM的情况完全相同（Hinton等人，2006；Hinton和Salakhutdinov，2006）。这是因为投影到前$p$个特征向量张成的子空间可以通过投影到前$p_1$个特征向量张成的子空间，然后投影到前$p$个特征向量张成的子空间来组合。

**9) 外部目标。** 在适当调整后，如果提供一组目标输出向量$y_1, \ldots, y_m$而不是$x_1, \ldots, x_m$作为目标，上述结果基本保持不变（见Baldi和Hornik，1988）。

**10) 对称性和Hebbian规则。** 在全局最小值处，当$C = I$时，$A = B^t$。约束$A = B^t$可以在学习过程中通过"权重共享"施加，并且与Hebbian规则一致，该规则在突触前和突触后神经元之间对称，通过将输出单元钳制为等于输入单元（或具有折叠自编码器）应用于网络。

## 4 布尔自编码器

布尔自编码器是最极端形式的非线性自编码器。在纯布尔情况下，我们有$F = G = \{0, 1\}$，$A$和$B$是无限制的布尔函数，$\Delta$是Hamming距离。通过限制布尔函数类$A$和$B$可以获得此问题的许多变体，例如通过限制隐藏单元的连接性。$F = G = \{0, 1\} = \mathbb{F}_2$的线性情况（其中$\mathbb{F}_2$是二元域）是布尔情况的特例，将在后面讨论。由于篇幅限制，证明仅简要概述。

**1) 群不变性。** 每个解在超立方体$H^p$的$2^p$个点的置换范围内定义。这是因为布尔函数是无限制的，因此它们的查找表可以容纳任何这样的置换，或隐藏状态的重新标记。

**2) 问题复杂度。** 一般情况下，整体优化问题是NP难的。更精确地说，必须指定感兴趣的区域，该区域由$n$、$m$和$p$中哪些变量趋于无穷来表征。显然必须有$n \to \infty$。如果$p$不趋于无穷，则问题可以是多项式的，例如当质心必须属于训练集时。如果$p \to \infty$且$m$是$n$的多项式（这是机器学习中感兴趣的情况，其中$m$通常是$n$的低次多项式），则寻找最佳布尔映射（即最小化训练集上与Hamming距离相关的失真$E$的布尔映射）的问题是NP难的，或者相应的判定问题是NP完全的。更精确地说，优化问题在$p \sim \epsilon \log_2 m$且$\epsilon > 0$的区域中是NP难的。此结果的证明在下一节给出。

**3) 固定层解。** 如果映射$A$固定，则容易找到最优映射$B$。反之，如果映射$B$固定，则容易找到最优映射$A$。为看到这一点，首先假设$A$固定。则对于隐藏层的$2^p$个可能布尔向量$h_1, \ldots, h_{2^p}$中的每一个，$A(h_1), \ldots, A(h_{2^p})$在超立方体$H^n$中提供$2^p$个点（质心）。可以通过将$H^n$的每个点分配给其最近质心来构建相应的Voronoi划分（平局时任意打破），从而将$H^n$划分为$2^p$个对应聚类$C_1, \ldots, C_{2^p}$，其中$C_i = C_{Vor}(A(h_i))$。最优映射$B^*$然后容易通过设置$B^*(x) = h_i$来定义，对于$C_i = C_{Vor}(A(h_i))$中的任何$x$。反之，假设$B$固定。则对于隐藏层的$2^p$个可能布尔向量$h_1, \ldots, h_{2^p}$中的每一个，令$C_B(h_i) = \{x \in H^n : B(x) = h_i\}$。为最小化重建误差，$A^*$必须将$h_i$映射到$H^n$中最小化到$X \cap C_B(h_i)$中点的Hamming距离之和的点$y$。容易看到最小值由逐分量多数向量$A^*(h_i) = \text{Majority}[X \cap C_B(h_i)]$实现，平局时任意打破（例如通过抛硬币）。

**4) $E$的景观。** 一般情况下$E$有许多局部最小值（例如，关于应用于$A$和$B$查找表的Hamming距离）。临界点定义为同时满足上述$A^*$和$B^*$方程的点。

**5) 聚类。** 整体优化问题是一个最优聚类问题。聚类由变换$B$定义。可以通过许多算法寻求近似解，如$k$-均值、置信传播（Frey和Dueck，2007）、最小生成路径和树（Slagle等人，1975）以及层次聚类。

**6) 循环稳定性。** 在任何临界点处，循环输出在第一次通过时是稳定的，因此对于任意$x$，$(AB)^n(x) = AB(x)$（等于相应Voronoi聚类的多数向量）。

**7) 泛化。** 在任何临界点处，对于任意$x$，$AB(x)$等于相应Voronoi聚类的质心，相应的误差可以容易地表达。

**8) 垂直组合。** 如果在输入层和隐藏层之间和/或隐藏层和输出层之间引入大小等于或大于$p$的额外布尔层，全局最优值保持不变。因此，添加这样的层不会减少整体失真$E$。考虑具有层大小$n, p_1, p, p_1, n$的布尔自编码器网络（图2），其中$n > p_1 > p$。则该网络的最优解可以通过首先计算$n, p_1, n$自编码器网络的最优解，并使用第一个网络隐藏层中的活动作为第二个网络的训练集，与$p_1, p, p_1$自编码器网络的最优解组合来获得，这与堆叠RBM的情况完全相同。这是因为全局最优对应于聚类为$2^p$个聚类，这可以通过首先聚类为$2^{p_1}$个聚类，然后将这些聚类聚类为$2^p$个聚类来获得。布尔函数的堆叠执行关于输入空间的层次聚类。

**9) 外部目标。** 在适当调整后，如果提供一组目标输出向量$y_1, \ldots, y_m$而不是$x_1, \ldots, x_m$作为目标，上述结果基本保持不变。为看到这一点，考虑一个由沿Hinton等人（2006）思路的自编码器堆叠组成的深度架构。对于输出层之前最后一个隐藏层中的任何活动向量$h$，计算训练集中被堆叠架构映射到$h$的点集$C(h)$。假设不失一般性地，$C(h) = \{x_1, \ldots, x_k\}$，对应目标$\{y_1, \ldots, y_k\}$。则容易看到，顶层为$h$产生的最终输出应该是由$\text{Majority}(y_1, \ldots, y_k)$给出的目标质心。

## 5 超立方体上的聚类复杂度

在本节中，我们简要回顾聚类复杂度的一些结果，然后证明超立方体聚类判定问题一般是NP完全的。不同空间中或具有不同目标函数的各种聚类问题的复杂度已在文献中得到研究。主要有两类结果：(1) 在图$G = (V, E, \Delta)$上推导的图论结果，其中不相似性$\Delta$不一定是距离；(2) 在欧氏空间$\mathbb{R}^d$中推导的几何结果，其中$\Delta = L_2^2$、$L_2$或$L_1$。

一般情况下，聚类判定问题是NP完全的，聚类优化问题是NP难的，除了涉及常数聚类数$k$或在1维欧氏空间中聚类的一些简单情况。一般情况下，欧氏空间中的结果比图上的结果更难推导。当存在多项式时间算法时，几何问题往往利用几何性质获得更快的解决方案。然而，现有的复杂度定理都没有直接解决关于Hamming距离的超立方体聚类问题。

为处理超立方体聚类问题，必须首先理解哪些量允许趋于无穷。如果$n$不允许趋于无穷，则训练样本数$m$也受$2^n$限制，并且由于我们假设$p < n$，没有可以缩放的量。因此必须有$n \to \infty$。也必须有$m \to \infty$。机器学习中感兴趣的情况是$m$是$n$的低次多项式。显然超立方体聚类问题在NP中，它是$\mathbb{R}^n$中聚类的特例。因此唯一需要解决的重要问题是将已知的NP完全问题归约到超立方体聚类问题。

对于归约，自然从已知的NP完全图论或几何聚类问题开始。在两种情况下，都必须找到将原始问题及其原始度量嵌入到具有Hamming距离的超立方体中的方法。存在将图同胚嵌入或压缩嵌入到超立方体中的定理（Hartman，1976；Winkler，1983），然而这些嵌入不将原始不相似性函数映射到Hamming度量上。

因此，我们这里更倾向于从一些已知的几何结果开始，并使用严格的立方图嵌入。一个图是立方的，如果它是某个超立方体$H^d$（对于某个$d$）的子图（Harary，1988；Livingston和Stout，1988）。虽然判定一个图是否是立方的是NP完全的（Afrati等人，1985），但存在一个定理（Havel和Morávek，1972）为图是立方的提供了必要和充分条件。图$G(V, E)$是立方的且可嵌入$H^d$中，当且仅当可以用$d$种颜色对$G$的边着色，使得：(1) 与公共顶点关联的所有边具有不同颜色；(2) 在$G$的每条路径中，某种颜色出现奇数次；(3) 在$G$的每个环中，没有颜色出现奇数次。我们现在可以陈述并证明以下定理。

**定理。** 考虑以下超立方体聚类问题：

输入：$m$个长度为$n$的二元向量$x_1, \ldots, x_m$和一个整数$k$。

输出：$k$个长度为$n$的二元向量$c_1, \ldots, c_k$（质心）和一个从$\{x_1, \ldots, x_m\}$到$\{c_1, \ldots, c_k\}$的函数$f$，最小化失真$E = \sum_{t=1}^{m} \Delta(x_t, f(x_t))$，其中$\Delta$是Hamming距离。当$k \sim m^\epsilon$（$\epsilon > 0$）时，超立方体聚类问题是NP难的。

**证明。** 为概述归约，我们从使用聚类质心和$L_1$距离在平面$\mathbb{R}^2$中聚类$m$个点的问题开始，该问题是NP完全的（Megiddo和Supowit，1984），通过从3-SAT归约（Garey和Johnson，1979），当$k \sim m^\epsilon$（$\epsilon > 0$）时（另见Mahajan等人（2009）和Vattani（2010）中的相关结果）。不失一般性，我们可以假设这些问题中的点位于方格网的顶点上。使用Havel和Morávek（1972）中的定理，可以证明平面中的$n \times m$方格网可以嵌入到$H^{n+m}$中。事实上，图3给出了一个显式嵌入。容易验证方格网上任意两点之间的$L_1$或Manhattan距离等于$H^{n+m}$中对应点之间的Hamming距离。这一多项式归约完成了证明：如果聚类数满足$k = 2^p \sim m^\epsilon$，或等价地$p \sim \epsilon \log_2 m \sim C \log n$，则与布尔自编码器相关的超立方体聚类问题是NP难的，相应的判定问题是NP完全的。如果聚类数$k$固定且质心必须属于训练集，则只有$\binom{m}{k} \sim m^k$种可能的质心选择，诱导相应的Voronoi聚类。这产生了一个平凡但不高效的多项式时间算法。当质心不需要在训练集中时，我们推测通过调整欧氏空间中的相应定理也存在多项式时间算法。

## 6 $p \geq n$的情况

当隐藏层大于输入层且$F = G$时，存在涉及恒等函数的最优0失真解。因此，这种情况只有在向问题添加额外约束时才有意义。这些约束可以以正则化的形式出现，例如确保隐藏层表示的稀疏性，或对函数类$A$和$B$的限制，或隐藏层中的噪声（见下一节）。当这些约束强制隐藏层仅假定$k$个不同的值且$k < m$时，例如在稀疏布尔隐藏层的情况下，先前的分析成立，问题简化为$k$聚类问题。

在这种大隐藏层的背景下，除了垂直组合外，自编码器还有一个自然的水平组合，可以通过水平组合自编码器来创建大隐藏层表示（图4）。可以训练两个（或更多）具有架构$n/p_1/n$和$n/p_2/n$的自编码器，并且可以组合隐藏层以产生大小为$p_1 + p_2$的扩展隐藏表示，然后可以将其馈送到整体架构的后续层中。$p_1$和$p_2$隐藏表示之间的差异可以通过许多不同的机制引入，例如使用不同的学习算法、不同的初始化、不同的训练样本、不同的学习率或不同的失真度量。也可以设想增量添加（或移除）隐藏单元到隐藏层的算法（Reed，1993；Kwok和Yeung，1997）。例如，在$\mathbb{R}$上的线性情况下，可以训练第一个隐藏单元提取第一个主成分，然后添加第二个隐藏单元提取第二个主成分，依此类推。

## 7 其他推广

在此引入的一般框架内，可以考虑其他类型的自编码器。首先，可以考虑对$F$和$G$有不同约束，或对$A$和$B$有不同约束的混合自编码器。一个简单的例子是输入和输出层为实数$F = \mathbb{R}$，隐藏层为二元$G = \{0, 1\}$（且$\Delta = L_2^2$）。容易验证在这种情况下，只要$2^p = k < m$，自编码器旨在将实数数据聚类为$k$个聚类，并且布尔情况下获得的所有结果在适当调整后都适用。例如，与隐藏状态$h$关联的质心应该是映射到$h$的输入向量的质心。一般情况下，当$k \sim m^\epsilon$时，这种混合自编码器也是NP难的，并且从概率角度来看，它对应于$k$个高斯混合模型。

第二个自然方向是考虑在实数域以外的域上的线性自编码器，例如在复数域$\mathbb{C}$上（Baldi等人，2011），或在有限域上。对于所有这些线性自编码器，$B$的核起着重要作用，因为输入向量基本上是模该核进行聚类的。这些自编码器并非没有理论和实践意义。考虑二元域$GF(2) = \mathbb{F}_2$上的线性自编码器。容易看到这是布尔自编码器的特例，其中布尔函数限制为奇偶校验函数。这个自编码器也可以被视为实现线性码（McEliece，1977）。当隐藏层的"传输"中存在噪声且$p > n$时，可以考虑$n$个隐藏单元对应于恒等函数，其余$p - n$个单元实现从输入线性计算的额外奇偶校验位，用于纠错。因此，所有著名的线性码，如Hamming码或Reed-Solomon码，都可以在这个线性自编码器框架中看待。虽然$\mathbb{F}_2$上的线性自编码器将在其他地方讨论，但值得注意的是它可能产生NP难问题，如同无限制布尔自编码器的情况。这可以通过考虑在二元矩阵的核中找到最小（非零）权重向量，或码的半径是NP完全问题来看到（McEliece和van Tilborg，1978；Frances和Litman，1997）。图5给出了自编码器的简单分类。

## 8 讨论

详细研究线性和布尔自编码器使人能够获得自编码器的一般视角，定义不同自编码器共享的关键属性，这些属性应在任何新型自编码器中系统地检查（例如群不变性、聚类、循环稳定性）。一般视角还显示了自编码器与信息和编码理论之间的密切联系：(1) $n < p$且隐藏层有噪声的自编码器对应于经典的有噪信道传输和编码问题，有限域上的线性码是其特例；(2) $n > p$的自编码器对应于压缩，当隐藏层中的状态数小于训练样本数（$m > k$）时可以是有损的，否则是无损的。

当$n > p$时，一般出现的图景是自编码器学习通常是NP完全的，除了简单但重要的情况（例如$\mathbb{R}$上的线性、固定$k$的布尔），并且本质上所有自编码器都在执行某种形式的聚类，这表明不同形式的无监督学习的统一观点，其中Hebbian学习、自编码器和聚类是同一枚硬币的三个面。虽然自编码器和Hebbian规则提供了无监督学习的实现，但正是聚类提供了支撑它们的基本概念操作。此外，重要的是要注意，一般情况下，在不为每个聚类提供标签的情况下聚类对象对于后续的高层处理是无用的。除了聚类之外，自编码器通过隐藏层中的活动为每个聚类提供标签，因此优雅地同时解决了聚类和标记问题。

RBM及其高效的对比学习算法可能提供自编码器和自编码器学习的高效形式，但怀疑在更深层次的概念层面上RBM有什么特殊之处是合理的。因此，应该可以通过堆叠其他类型的自编码器来获得与Hinton等人（2006）和Hinton和Salakhutdinov（2006）中描述的相当的结果，更一般地通过使用垂直组合层次化堆叠一系列聚类算法，也许还可以结合水平组合。如前几节所述，在层次聚类堆叠的顶部添加用于监督回归或分类任务的顶层很容易。总体而言，这些结果表明：(1) 所谓的深度架构实际上可能具有非平凡但常数（或对数）深度，这也与感官神经元回路中的观察一致；(2) 深度架构背后的基本无监督操作，以这种或那种形式，是聚类，它在水平和垂直方向上都是可组合的——在这种观点下，聚类可能成为构建智能系统的中心和几乎充分的要素；(3) 忽略许多硬件细节时，深度架构的泛化性质可能更容易理解，用最简单形式的自编码器（例如布尔）或更基本的底层聚类操作来表述。

## 致谢

工作部分由NSF IIS-0513376、NIH LM010235和NIH-NLM T15 LM07443资助给PB。

## 参考文献

[1] F. Afrati, C.H. Papadimitriou, and G. Papageorgiou. The complexity of cubical graphs. Information and control, 66(1-2):53-60, 1985.

[2] P. Baldi and K. Hornik. Neural networks and principal component analysis: Learning from examples without local minima. Neural Networks, 2(1):53-58, 1988.

[3] P. Baldi, S. Forouzan, and Z. Lu. Complex-Valued Autoencoders. Neural Networks, 2011. Submitted.

[4] Yoshua Bengio and Yann LeCun. Scaling learning algorithms towards AI. In L. Bottou, O. Chapelle, D. DeCoste, and J. Weston, editors, Large-Scale Kernel Machines. MIT Press, 2007.

[5] H. Bourlard and Y. Kamp. Auto-association by multilayer perceptrons and singular value decomposition. Biological cybernetics, 59(4):291-294, 1988. ISSN 0340-1200.

[6] P. Clote and E. Kranakis. Boolean functions and computation models. Springer Verlag, 2002.

[7] Dumitru Erhan, Yoshua Bengio, Aaron Courville, Pierre-Antoine Manzagol, Pascal Vincent, and Samy Bengio. Why does unsupervised pre-training help deep learning? Journal of Machine Learning Research, 11:625-660, February 2010.

[8] M. Frances and A. Litman. On covering problems of codes. Theory of Computing Systems, 30(2):113-119, 1997.

[9] B.J. Frey and D. Dueck. Clustering by passing messages between data points. Science, 315(5814):972, 2007.

[10] M.R. Garey and D.S. Johnson. Computers and Intractability. Freeman San Francisco, 1979.

[11] F. Harary. Cubical graphs and cubical dimensions. Computers & Mathematics with Applications, 15(4):271-275, 1988.

[12] J. Hartman. The homeomorphic embedding of Kn in the m-cube* 1. Discrete Mathematics, 16(2):157-160, 1976.

[13] I. Havel and J. Morávek. B-valuations of graphs. Czechoslovak Mathematical Journal, 22(2):338-351, 1972.

[14] D.O. Hebb. The organization of behavior: A neurophychological study. WileyInterscience, New York, 1949.

[15] G.E. Hinton and R.R. Salakhutdinov. Reducing the dimensionality of data with neural networks. Science, 313(5786):504, 2006.

[16] G.E. Hinton, S. Osindero, and Y.W. Teh. A fast learning algorithm for deep belief nets. Neural Computation, 18(7):1527-1554, 2006.

[17] T. Kwok and D. Yeung. Constructive Algorithms for Structure Learning in Feedforward Neural Networks for Regression Problems. IEEE Transactions on Neural Networks, 8:630-645, 1997.

[18] M. Livingston and Q.F. Stout. Embeddings in hypercubes. Mathematical and Computer Modelling, 11:222-227, 1988.

[19] M. Mahajan, P. Nimbhorkar, and K. Varadarajan. The planar k-means problem is NP-hard. WALCOM: Algorithms and Computation, pages 274-285, 2009.

[20] R. McEliece and H. van Tilborg. On the inherent intractability of certain coding problems(Corresp.). IEEE Transactions on Information Theory, 24(3):384-386, 1978.

[21] R. J. McEliece. The Theory of Information and Coding. Addison-Wesley Publishing Company, Reading, MA, 1977.

[22] N. Megiddo and K.J. Supowit. On the complexity of some common geometric location problems. SIAM J. COMPUT., 13(1):182-196, 1984.

[23] G. Montufar and N. Ay. Reﬁnements of Universal Approximation Results for Deep Belief Networks and Restricted Boltzmann Machines. Neural Computation, pages 1-14, 2011. ISSN 0899-7667.

[24] E. Oja. Simplified neuron model as a principal component analyzer. Journal of mathematical biology, 15(3):267-273, 1982.

[25] R. Reed. Pruning algorithms-a survey. Neural Networks, IEEE Transactions on, 4(5):740-747, 1993.

[26] D.E. Rumelhart, G.E. Hinton, and R.J. Williams. Learning internal representations by error propagation. In Parallel Distributed Processing. Vol 1: Foundations. MIT Press, Cambridge, MA, 1986.

[27] JL Slagle, CL Chang, and SR Heller. A clustering and data reorganization algorithm. IEEE Transactions on Systems, Man and Cybernetics, 5:121-128, 1975.

[28] I. Sutskever and G.E. Hinton. Deep, narrow sigmoid belief networks are universal approximators. Neural Computation, 20(11):2629-2636, 2008.

[29] A. Vattani. A simpler proof of the hardness of k-means clustering in the plane. UCSD Technical Report, 2010.

[30] P.M. Winkler. Proof of the squashed cube conjecture. Combinatorica, 3(1):135-139, 1983.
