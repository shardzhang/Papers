# FM: Factorization Machines

> Steffen Rendle | 大阪大学 产业科学研究所 推理智能研究部门


本文介绍了因子分解机（Factorization Machines, FM），这是一种结合了支持向量机（SVM）与因子分解模型优势的新型模型类。核心内容：

- FM是一种通用预测器，可处理任意实值特征向量，通过因子化参数建模变量间所有交互
- FM在**高度稀疏数据下**仍能估计可靠参数，而SVM在此类场景下会失败
- FM模型方程可在线性时间内计算，可直接在原始空间优化，无需对偶变换
- FM仅通过特征工程即可模仿矩阵分解、SVD++、PITF、FPMC等最先进的因子分解模型

关键发现：FM的通用性不以牺牲预测精度或计算复杂度为代价，在稀疏设置下FM显著优于多项式SVM，且与专门化因子分解模型性能相当。

---


## 摘要

本文提出了因子分解机（FM），这是一种结合了支持向量机（SVM）与因子分解模型优势的新型模型类。与SVM一样，FM是一种通用预测器，可处理**任意实值特征向量**。与SVM不同的是，FM使用因子化参数 **对所有变量间的交互进行建模**。因此，即使在SVM会失败的巨大稀疏问题（如推荐系统）中，FM也能估计交互作用。我们证明FM的模型方程可以在线性时间内计算，因此FM可以直接优化。与非线性SVM不同，不需要对偶形式的变换，模型参数可以直接估计，无需解中的任何支持向量。我们展示了FM与SVM的关系，以及FM在稀疏设置下进行参数估计的优势。

另一方面，存在许多不同的因子分解模型，如矩阵因子分解、平行因子分析，或专门化模型如SVD++、PITF或FPMC。这些模型的缺点在于它们**不适用于通用预测任务**，仅能处理特定的输入数据。此外，它们的模型方程和优化算法是**针对每个任务单独推导**的。我们证明，FM仅通过指定输入数据（即特征向量）即可模仿这些模型。这使得即使不具备因子分解模型专业知识的用户也能轻松应用FM。

**关键词**——因子分解机；稀疏数据；张量因子分解；支持向量机

---


## 1. 引言

支持向量机是机器学习和数据挖掘中最流行的预测器之一。然而，在协同过滤等场景中，SVM并不重要，最好的模型要么是标准矩阵/张量因子分解模型（如PARAFAC [1]）的直接应用，要么是使用因子化参数的专门化模型 [2, 3, 4]。在本文中，我们**证明标准SVM预测器在这些任务中不成功的唯一原因在于，它们无法在非常稀疏的数据下从复杂（非线性）核空间中学习可靠的参数（"超平面"）**。另一方面，张量因子分解模型乃至专门化因子分解模型的缺点在于：（1）它们不适用于标准预测数据（例如 $\mathbb{R}^n$ 中的**实值特征向量**），（2）专门化模型通常**针对特定任务单独推导**，需要大量的建模和算法设计工作。

在本文中，我们提出了一种新的预测器——因子分解机（FM），它像SVM一样是一种**通用预测器**，但也能在极高稀疏度下估计可靠参数。因子分解机对所有嵌套变量交互（相当于SVM中的多项式核）进行建模，但使用因子化参数化而非SVM中的稠密参数化。我们证明FM的模型方程**可以在线性时间内计算**，且仅依赖于线性数量的参数。这使得可以直接优化和存储模型参数，而无需为预测存储任何训练数据（如支持向量）。相比之下，非线性SVM通常在dual形式下优化，计算预测（模型方程）依赖于部分训练数据（支持向量）。我们还证明，FM涵盖了协同过滤任务中许多最成功的方法，包括有偏MF、SVD++ [2]、PITF [3] 和 FPMC [4]。

总之，我们提出的FM的优势在于：
1. FM允许在SVM会失败的极稀疏数据下进行参数估计。
2. FM具有线性复杂度，可在原始空间优化，且不依赖于SVM中的支持向量。我们证明FM可扩展到像Netflix这样拥有1亿训练样本的大型数据集。
3. FM是一种通用预测器，可处理任意实值特征向量。相比之下，其他最先进的因子分解模型 **仅适用于非常受限的输入数据**。我们将证明，**仅通过定义输入数据的特征向量，FM即可模仿有偏MF、SVD++、PITF或FPMC等最先进模型**。


## 2. 稀疏下的预测

最常见的预测任务是估计一个函数 $y: \mathbb{R}^n \rightarrow T$ ，从实值特征向量 $x \in \mathbb{R}^n$ 到目标域 $T$ （例如 $T = \mathbb{R}$ 用于回归，或 $T = \{+, -\}$ 用于分类）。在有监督设置下，假设存在一个训练数据集 $D = \{(x^{(1)}, y^{(1)}), (x^{(2)}, y^{(2)}), \ldots\}$ ，其中包含目标函数 $y$ 的样本。我们还研究了排序任务，其中目标 $T = \mathbb{R}$ 的函数 $y$ 可用于对特征向量 $x$ 打分，并根据分值排序。打分函数可以通过成对训练数据 [5] 学习，其中特征元组 $(x^{(A)}, x^{(B)}) \in D$ 意味着 $x^{(A)}$ 应排在 $x^{(B)}$ 之前。**由于成对排序关系是反对称的，只需使用正训练实例即可。**

在本文中，我们处理 $x$ 高度稀疏的问题，即向量 $x$ 中几乎所有的元素 $x_i$ 都为零。设 $m(x)$ 为特征向量 $x$ 中非零元素的数量， $m_D$ 为所有向量 $x \in D$ 的平均非零元素数 $m(x)$ 。极大稀疏性（ $m_D \ll n$ ）出现在许多真实世界数据中，如事件交易的特征向量（例如推荐系统中的购买记录）或文本分析（例如词袋方法）。**巨大稀疏性的一个原因在于底层问题涉及大领域类别变量**。

**示例 1** 假设我们有电影评论系统的交易数据。系统记录哪个用户 $u \in U$ 在某个时间 $t \in \mathbb{R}$ 以评分 $r \in \{1, 2, 3, 4, 5\}$ 评价了一部电影（item） $i \in I$ 。设用户集合 $U$ 和 item 集合 $I$ 为：

$$
U = \{\text{Alice (A), Bob (B), Charlie (C), } \ldots\}
$$

$$
I = \{\text{Titanic (TI), Notting Hill (NH), Star Wars (SW), Star Trek (ST), } \ldots\}
$$

设观测数据 $S$ 为：

$$
S = \{\text{(A, TI, 2010-1, 5), (A, NH, 2010-2, 3), (A, SW, 2010-4, 1),}
$$

$$
\text{(B, SW, 2009-5, 4), (B, ST, 2009-8, 5),}
$$

$$
\text{(C, TI, 2009-9, 1), (C, SW, 2009-12, 5)}\}
$$

使用该数据进行预测任务的一个示例是，估计一个函数 $\hat{y}$ ，**用于预测用户在某个时间点对某部电影的评分行为**。

<img src="/Users/dazhang/PycharmProject/Papers/3-RecSys/.picture/image-20260728095850460.png" alt="image-20260728095850460" style="zoom:33%;" />

图1展示了如何从 $S$ 为该任务创建特征向量的一个示例。这里，首先有 $|U|$ 个**二元指示变量**（蓝色），表示交易的活跃用户——每个交易 $(u, i, t, r) \in S$ 中**恰好有一个活跃用户**，例如第一个交易中用户 Alice（ $x^{(1)}_A = 1$ ）。接下来 $|I|$ 个二元指示变量（红色）表示活跃 item——同样恰好有一个活跃 item（例如 $x^{(1)}_{TI} = 1$ ）。图1中的特征向量还包含该用户评分过的所有其他电影的指示变量（黄色）。对每个用户，这些**变量被归一化使其和为1**。例如，Alice评分过Titanic、Notting Hill和Star Wars。此外，示例包含一个变量（绿色），表示从2009年1月起计的月份时间。最后，向量包含用户评分活跃电影之前所评最后一部电影的信息（棕色）——例如对于 $x^{(2)}$ ，Alice在评分Notting Hill之前评分了Titanic。在第V节中，我们将展示将此类特征向量作为输入数据的因子分解机如何 与 **专门化的最先进因子分解**模型相关联。

我们将在本文中始终使用此示例数据进行说明。但请注意，FM像SVM一样是通用预测器，因此适用于任意实值特征向量，且不限于推荐系统。


## 3. 因子分解机（FM）

在本节中，我们介绍因子分解机。我们详细讨论模型方程，并简要展示如何将FM应用于若干预测任务。

### 3.1. 因子分解机模型

#### 1) 模型方程

阶数 $d = 2$ 的因子分解机模型方程定义为：

$$
\hat{y}(x) := w_0 + \sum_{i=1}^{n} w_i x_i + \sum_{i=1}^{n} \sum_{j=i+1}^{n} \langle v_i, v_j \rangle x_i x_j \qquad (1)
$$

其中需要估计的模型参数为：

$$
w_0 \in \mathbb{R}, \quad w \in \mathbb{R}^n, \quad V \in \mathbb{R}^{n \times k} \qquad (2)
$$

且 $\langle \cdot, \cdot \rangle$ 是两个大小为 $k$ 的向量的点积：

$$
\langle v_i, v_j \rangle := \sum_{f=1}^{k} v_{i,f} \cdot v_{j,f} \qquad (3)
$$

 $V$ 中的一行 $v_i$ 用 $k$ 个因子描述第 $i$ 个变量。 $k \in \mathbb{N}_0^+$ 是一个超参数，定义了因子分解的维度。

一个2路FM（阶数 $d = 2$ ）捕获变量间的**所有单变量 和 成对交互**：
- $w_0$ 是全局偏置。
- $w_i$ 建模第 $i$ 个变量的强度。
- $\hat{w}_{i,j} := \langle v_i, v_j \rangle$ 建模**第 $i$ 和第 $j$ 个变量之间的交互**。**FM不是为每个交互使用独立的模型参数 $w_{i,j} \in \mathbb{R}$ ，而是通过因子分解来建模交互**。我们将在后面看到，这是**在稀疏条件下实现高阶交互（ $d \geq 2$ ）高质量参数估计的关键**。

#### 2) 表达能力

众所周知，对于任意正定矩阵 $W$ ，只要 $k$ 足够大，存在矩阵 $V$ 使得 $W = V \cdot V^t$ 。这表明只要 $k$ 选择得足够大，FM可以表达任意交互矩阵 $W$ 。然而在稀疏设置中，通常应选择较小的 $k$ ，因为**没有足够的数据来估计复杂的交互** $W$ 。**限制 $k$ ——从而限制FM的表达能力——会导致更好的泛化**，从而在稀疏条件下得到改进的交互矩阵。

#### 3) 稀疏下的参数估计

**在稀疏设置中，通常没有足够的数据来直接且独立地估计变量间的交互**。因子分解机即使在这样也能很好地估计交互，**因为它们通过因子分解打破了交互参数的独立性**。一般而言，这意味着**一个交互的数据也有助于估计相关交互的参数**。我们将用图1中的数据示例更清晰地说明这一思想。假设我们想要估计Alice（A）和Star Trek（ST）之间的交互，以预测目标 $y$ （此处为评分）。显然，训练数据中没有 $x_A$ 和 $x_{ST}$ 同时非零的样本 $x$ ，因此直接估计将导致无交互（ $w_{A,ST} = 0$ ）。但通过因子化交互参数，我们甚至可以在这种情况下估计交互。首先，Bob和Charlie将具有相似的因子向量 $v_B$ 和 $v_C$ ，因为两者与Star Wars（ $v_{SW}$ ）在预测评分方面具有相似的交互——即 $\langle v_B, v_{SW} \rangle$ 和 $\langle v_C, v_{SW} \rangle$ 必须相似。Alice（ $v_A$ ）的因子向量将与Charlie（ $v_C$ ）不同，因为她在预测评分方面与Titanic和Star Wars的因子具有不同的交互。其次，**Star Trek的因子向量很可能与Star Wars的相似**，因为Bob对两部电影在预测 $y$ 方面具有相似的交互。总之，这意味着Alice和Star Trek的因子向量的点积（即交互）将与Alice和Star Wars的相似——这从直观上也是合理的。

#### 4) 计算

接下来，我们从计算角度展示如何使FM适用。直接计算公式（1）的复杂度为 $O(k n^2)$ ，因为**必须计算所有成对交互**。但通过重新表述，可以降至线性运行时。

**引理 3.1**：因子分解机的模型方程（公式（1））可以在线性时间 $O(k n)$ 内计算。

**证明**：由于成对交互的因子分解，没有模型参数直接依赖于两个变量（例如，索引为 $(i, j)$ 的参数）。因此，成对交互可以重新表述：

$$
\begin{aligned}
\sum_{i=1}^{n} \sum_{j=i+1}^{n} \langle v_i, v_j \rangle x_i x_j
&= \frac{1}{2} \sum_{i=1}^{n} \sum_{j=1}^{n} \langle v_i, v_j \rangle x_i x_j - \frac{1}{2} \sum_{i=1}^{n} \langle v_i, v_i \rangle x_i x_i \\
&= \frac{1}{2} \left( \sum_{i=1}^{n} \sum_{j=1}^{n} \sum_{f=1}^{k} v_{i,f} v_{j,f} x_i x_j - \sum_{i=1}^{n} \sum_{f=1}^{k} v_{i,f}^2 x_i^2 \right) \\
&= \frac{1}{2} \sum_{f=1}^{k} \left( \left( \sum_{i=1}^{n} v_{i,f} x_i \right) \left( \sum_{j=1}^{n} v_{j,f} x_j \right) - \sum_{i=1}^{n} v_{i,f}^2 x_i^2 \right) \\
&= \frac{1}{2} \sum_{f=1}^{k} \left( \left( \sum_{i=1}^{n} v_{i,f} x_i \right)^2 - \sum_{i=1}^{n} v_{i,f}^2 x_i^2 \right)
\end{aligned}
$$

该方程在 $k$ 和 $n$ 上均具有线性复杂度——即其计算复杂度为 $O(k n)$ 。

此外，在稀疏条件下， $x$ 中的大多数元素为0（即 $m(x)$ 很小），因此求和只需在非零元素上计算。因此在稀疏应用中，因子分解机的计算复杂度为 $O(k m_D)$ ——例如典型推荐系统中 $m_D = 2$ （如MF方法，见第V-A节）。

### 3.2. 因子分解机作为预测器

FM可应用于**多种预测任务**，包括：
- **回归**： $\hat{y}(x)$ 可直接用作预测器，优化准则例如 $D$ 上的最小均方误差。
- **二分类**：使用 $\hat{y}(x)$ 的符号，参数针对 hinge损失 或 logit损失优化。
- **排序**：向量 $x$ 按 $\hat{y}(x)$ 的分值排序，优化在实例向量对 $(x^{(a)}, x^{(b)}) \in D$ 上进行，使用成对分类损失（例如 [5] 中的方法）。

在所有这些情况下，通常会在优化目标中添加**L2等正则化项以防止过拟合**。

### 3.3. 学习因子分解机

如我们所示，FM具有**封闭形式的模型方程**，可在线性时间内计算。因此，FM的模型参数（ $w_0$ , $w$ , $V$ ）可以通过梯度下降方法（如随机梯度下降SGD）针对多种损失函数高效学习，包括平方损失、logit损失或hinge损失。FM模型的梯度为：

$$
\frac{\partial}{\partial \theta} \hat{y}(x) =
\begin{cases}
1, & \text{if } \theta \text{ 是 } w_0 \\
x_i, & \text{if } \theta \text{ 是 } w_i \\
x_i \sum_{j=1}^{n} v_{j,f} x_j - v_{i,f} x_i^2, & \text{if } \theta \text{ 是 } v_{i,f}
\end{cases} \qquad (4)
$$

求和 $\sum_{j=1}^{n} v_{j,f} x_j$ 与 $i$ 无关，因此可以预计算（例如在计算 $\hat{y}(x)$ 时）。一般而言，每个梯度可以在常数时间 $O(1)$ 内计算。对于一个样本 $(x, y)$ 的所有参数更新**可以在 $O(k n)$ 内完成**——或在稀疏条件下为 $O(k m(x))$ 。

我们提供了一个**通用实现 LIBFM**，使用SGD并支持 逐点损失 和 成对损失。

### 3.4. d路因子分解机

到目前为止描述的2路FM可以很容易地推广到d路FM：

$$
\hat{y}(x) := w_0 + \sum_{i=1}^{n} w_i x_i + \sum_{l=2}^{d} \sum_{i_1=1}^{n} \cdots \sum_{i_l=i_{l-1}+1}^{n} \left( \prod_{j=1}^{l} x_{i_j} \right) \left( \sum_{f=1}^{k_l} \prod_{j=1}^{l} v^{(l)}_{i_j, f} \right) \qquad (5)
$$

其中第 $l$ 阶交互的交互参数通过PARAFAC模型 [1] 因子分解，模型参数为：

$$
V^{(l)} \in \mathbb{R}^{n \times k_l}, \quad k_l \in \mathbb{N}_0^+ \qquad (6)
$$

计算公式（5）的直接复杂度为 $O(k_d n^d)$ 。但使用与引理3.1相同的论证，可以证明它可以在线性时间内计算。

### 3.5. 总结

FM使用 **因子化交互** 而非 **完全参数化交互**，对特征向量 $x$ 中**所有值之间的可能交互**进行建模。这有两个主要优势：

1. 即使在高度稀疏条件下也能估计值之间的交互。特别是，可以**泛化到未观测的交互**。
2. **参数数量以及预测和学习的时间是线性的**。这使得使用SGD进行直接优化可行，并**允许针对多种损失函数进行优化**。

在本文的剩余部分，我们将展示因子分解机与支持向量机以及矩阵、张量和专门化因子分解模型之间的关系。


## 4. FM vs. SVM

### 4.1. SVM模型

SVM [6] 的模型方程可以表示为**变换后的输入 $x$ 与模型参数 $w$ 之间的点积**： $\hat{y}(x) = \langle \phi(x), w \rangle$ ，其中 $\phi$ 是从特征空间 $\mathbb{R}^n$ 到更复杂空间 $\mathcal{F}$ 的映射。映射 $\phi$ 与核函数的关系为：

$$
K: \mathbb{R}^n \times \mathbb{R}^n \rightarrow \mathbb{R}, \quad K(x, z) = \langle \phi(x), \phi(z) \rangle
$$

下面，我们通过分析SVM的原始形式来讨论FM与SVM的关系。

#### 1) 线性核

最简单的核是线性核： $K_l(x, z) := 1 + \langle x, z \rangle$ ，对应映射 $\phi(x) := (1, x_1, \ldots, x_n)$ 。因此线性SVM的模型方程可以重写为：

$$
\hat{y}(x) = w_0 + \sum_{i=1}^{n} w_i x_i, \quad w_0 \in \mathbb{R}, \quad w \in \mathbb{R}^n \qquad (7)
$$

显然，线性SVM（公式（7））与阶数 $d = 1$ 的FM（公式（5））相同。

#### 2) 多项式核

多项式核允许SVM对**变量间的高阶交互**进行建模。其定义为 $K(x, z) := (\langle x, z \rangle + 1)^d$ 。例如对于 $d = 2$ ，对应以下映射：

$$
\phi(x) := (1, \sqrt{2} x_1, \ldots, \sqrt{2} x_n, x_1^2, \ldots, x_n^2, \sqrt{2} x_1 x_2, \ldots, \sqrt{2} x_1 x_n, \sqrt{2} x_2 x_3, \ldots, \sqrt{2} x_{n-1} x_n) \qquad (8)
$$

因此多项式SVM的模型方程可以重写为：

$$
\hat{y}(x) = w_0 + \sqrt{2} \sum_{i=1}^{n} w_i x_i + \sum_{i=1}^{n} w^{(2)}_{i,i} x_i^2 + \sqrt{2} \sum_{i=1}^{n} \sum_{j=i+1}^{n} w^{(2)}_{i,j} x_i x_j \qquad (9)
$$

其中模型参数为：

$$
w_0 \in \mathbb{R}, \quad w \in \mathbb{R}^n, \quad W^{(2)} \in \mathbb{R}^{n \times n} \text{（对称矩阵）}
$$

比较多项式SVM（公式（9））和FM（公式（1）），可以看到两者都对最高阶 $d = 2$ 的所有嵌套交互进行建模。SVM和FM之间的主要区别在于参数化：SVM的所有交互参数 $w_{i,j}$ 是**完全独立的**，例如 $w_{i,j}$ 和 $w_{i,l}$ 。相比之下，**FM的交互参数是因子化的**，因此 $\langle v_i, v_j \rangle$ 和 $\langle v_i, v_l \rangle$ 相互依赖，因为它们**重叠并共享参数**（这里是 $v_i$ ）。

### 4.2. 稀疏下的参数估计

下面，我们将展示为什么线性和多项式SVM在极其稀疏的问题上会失败。我们以用户和item指示变量的协同过滤示例（见图1示例的前两组，蓝色和红色）来展示。这里，特征向量稀疏，只有两个非零元素（活跃用户 $u$ 和活跃 item $i$ ）。

#### 1) 线性SVM

对于这类数据 $x$ ，线性SVM模型为：

$$
\hat{y}(x) = w_0 + w_u + w_i \qquad (10)
$$

因为当且仅当 $j = u$ 或 $j = i$ 时 $x_j = 1$ 。该模型对应最基本的协同过滤模型之一，仅**捕获用户和item偏置**。由于该模型非常简单，即使在稀疏条件下也能很好地估计参数。然而，经验预测质量通常较低（见图2）。

<img src="/Users/dazhang/PycharmProject/Papers/3-RecSys/.picture/image-20260728103806764.png" alt="image-20260728103806764" style="zoom:33%;" />

#### 2) 多项式SVM

使用多项式核，SVM可以捕获高阶交互（此处为用户和item之间）。在我们的稀疏案例中， $m(x) = 2$ ，SVM的模型方程等价于：

$$
\hat{y}(x) = w_0 + \sqrt{2}(w_u + w_i) + w^{(2)}_{u,u} + w^{(2)}_{i,i} + \sqrt{2} w^{(2)}_{u,i} \qquad (11)
$$

首先， $w_u$ 和 $w^{(2)}_{u,u}$ **表达相同含义——即可以舍弃其中一个**（例如 $w^{(2)}_{u,u}$ ）。现在模型方程与线性情况相同，但多了一个用户-item交互 $w^{(2)}_{u,i}$ 。在典型的协同过滤（CF）问题中，对于每个交互参数 $w^{(2)}_{u,i}$ ，**训练数据中最多有一个观测** $(u, i)$ ，而对于测试数据中的样本 $(u', i^{\prime})$ ，**训练数据中通常完全没有观测**。例如在图1中，交互 (Alice, Titanic) 只有一个观测，而交互 (Alice, Star Trek) 则没有。这意味着所有测试样本 $(u, i)$ 的交互参数 $w^{(2)}_{u,i}$ 的**最大间隔解**为0（例如 $w^{(2)}_{A,ST} = 0$ ）。因此，多项式SVM无法利用任何2路交互来预测测试样本；所以多项式**SVM仅依赖于用户和item偏置，不能提供比线性SVM更好的估计**。

对于SVM，估计高阶交互不仅是CF中的问题，在所有数据极度稀疏的场景中都是问题。**因为要可靠估计成对交互 $(i, j)$ 的参数 $w^{(2)}_{i,j}$ ，必须有"足够多"的样本 $x \in D$ 满足 $x_i \neq 0 \land x_j \neq 0$ **。只要 $x_i = 0$ 或 $x_j = 0$ ，该样本 $x$ 就不能用于估计参数 $w^{(2)}_{i,j}$ 。**总之，如果数据过于稀疏，即 $(i, j)$ 的样本太少甚至没有，SVM很可能会失败。**

### 4.3. 总结

1. **SVM的稠密参数化** 需要交互的直接观测，而这在稀疏设置中通常不可得。FM的参数即使在稀疏条件下也能很好地估计（见第III-A3节）。
2. FM可以在原始空间中直接学习。**非线性SVM通常在dual空间中学习**。
3. FM的模型方程独立于训练数据。SVM的预测依赖于部分训练数据（支持向量）。


## 5. FM vs. 其他因子分解模型

存在多种因子分解模型，从类别变量上的 **m元关系的标准模型**（例如MF、PARAFAC）到特定数据和任务的**专门化模型**（例如SVD++、PITF、FPMC）。接下来，我们将展示FM仅通过使用正确的输入数据（例如特征向量 $x$ ）即可模仿这些模型中的许多种。

### 5.1. 矩阵与张量因子分解

矩阵因子分解（MF）是研究最充分的因子分解模型之一（例如 [7, 8, 2]）。它因子分解**两个类别变量**（例如 $U$ 和 $I$ ）之间的关系。处理类别变量的标准方法是为 $U$ 和 $I$ 的每个水平定义二元指示变量（例如见图1，第一组（蓝色）和第二组（红色））：

$$
n := |U \cup I|, \quad x_j := \delta(j = u \lor j = i) \qquad (12)
$$

使用该特征向量 $x$ 的FM与矩阵因子分解模型 [2] 相同，因为 $x_j$ 仅对 $u$ 和 $i$ 非零，因此所有其他偏置和交互都消失：

$$
\hat{y}(x) = w_0 + w_u + w_i + \langle v_u, v_i \rangle \qquad (13)
$$

用同样的论证，可以看出对于**多于两个类别变量**的问题，FM包含一个嵌套的平行因子分析模型（PARAFAC）[1]。

### 5.2. SVD++

对于**评分预测任务**（即回归），Koren将矩阵因子分解模型改进为SVD++模型 [2]。FM可以通过使用以下输入数据 $x$ （如图1的**前三组**）来模仿该模型：

$$
n := |U \cup I \cup L|, \quad x_j :=
\begin{cases}
1, & \text{if } j = u \\
1, & \text{if } j = i \\
\frac{1}{\sqrt{|N_u|}}, & \text{if } j \in N_u \\
0, & \text{otherwise}
\end{cases} \qquad (14)
$$

其中 ** $N_u$ 是该用户评分过的所有电影的集合**。使用该数据的FM（ $d = 2$ ）将表现为：

$$
\begin{aligned}
\hat{y}(x) &= w_0 + w_u + w_i + \langle v_u, v_i \rangle \\
&+ \frac{1}{\sqrt{|N_u|}} \sum_{l \in N_u} \langle v_i, v_l \rangle + \frac{1}{|N_u|} \sum_{l \in N_u} \left( w_l + \langle v_u, v_l \rangle + \sum_{l' \in N_u, l' > l} \langle v_l, v_{l'} \rangle \right)
\end{aligned} \qquad (15)
$$

其中第一部分与SVD++模型完全相同。但FM还包含用户与电影 $N_u$ 之间的额外交互、**电影 $N_u$ 的基本效应** 以及  $N_u$ 中电影对之间的交互。

> [!NOTE]
>
> SVD++不是很了解，但是也不用深入了解

### 5.3. 用于标签推荐的PITF

标签预测问题 定义为 对给定用户和item组合的标签进行排序。这意味着涉及三个类别域：用户 $U$ 、item $I$ 和标签 $T$ 。在关于标签推荐的ECML/PKDD Discovery Challenge中，基于因子化成对交互（PITF）的模型取得了最佳分数 [3]。我们将展示FM如何模仿该模型。为活跃用户 $u$ 、item $i$ 和标签 $t$ 使用二元指示变量的因子分解机产生以下模型：

$$
n := |U \cup I \cup T|, \quad x_j := \delta(j = u \lor j = i \lor j = t) \qquad (16)
$$

$$
\hat{y}(x) = w_0 + w_u + w_i + w_t + \langle v_u, v_i \rangle + \langle v_u, v_t \rangle + \langle v_i, v_t \rangle \qquad (17)
$$

由于该模型用于在同一用户/item组合 $(u, i)$ 内的两个标签 $t_A, t_B$ 之间进行排序 [3]，优化和预测都基于样本 $(u, i, t_A)$ 和 $(u, i, t_B)$ 的分值差。因此，使用成对排序优化（如 [5, 3]），FM模型等价于：

$$
\hat{y}(x) := w_t + \langle v_u, v_t \rangle + \langle v_i, v_t \rangle \qquad (18)
$$

现在，原始的PITF模型 [3] 和带二元指示符的FM模型（公式（18））几乎相同。唯一的区别在于：（i）FM模型对 $t$ 有一个偏置项 $w_t$ ，（ii）标签的因子化参数 $v_t$ 在 $(u, t)$ 和 $(i, t)$ 交互之间**在FM模型中是共享的**，而在原始PITF模型中是个别的。除了这一理论分析，图3经验性地展示了两种模型在该任务上也取得了相当的预测质量。

> [!NOTE]
>
> PITF不是很了解，但是也不用深入了解

### 5.4. 因子化个性化马尔可夫链（FPMC）

FPMC模型 [4] 尝试根据用户 $u$ 的上一次购买（在时间 $t-1$ ）对在线商店中的产品进行排序。同样仅通过特征生成，因子分解机（ $d = 2$ ）具有相似的行为：

$$
n := |U \cup I \cup L|, \quad x_j :=
\begin{cases}
1, & \text{if } j = u \\
1, & \text{if } j = i \\
\frac{1}{|B_u^{t-1}|}, & \text{if } j \in B_u^{t-1} \\
0, & \text{otherwise}
\end{cases} \qquad (19)
$$

其中 $B_u^{t} \subseteq L$ 是**用户 $u$ 在时间 $t$ 购买的所有item的集合**（"购物篮"）（详见 [4]）。然后：

$$
\begin{aligned}
\hat{y}(x) &= w_0 + w_u + w_i + \langle v_u, v_i \rangle \\
&+ \frac{1}{|B_u^{t-1}|} \sum_{l \in B_u^{t-1}} \langle v_i, v_l \rangle \\
&+ \frac{1}{|B_u^{t-1}|} \sum_{l \in B_u^{t-1}} \left( w_l + \langle v_u, v_l \rangle + \frac{1}{|B_u^{t-1}|} \sum_{l' \in B_u^{t-1}, l' > l} \langle v_l, v_{l'} \rangle \right)
\end{aligned} \qquad (20)
$$

与标签推荐类似，该模型用于排序并针对排序优化（这里是对item $i$ 排序），因此预测和优化准则中只使用 $(u, i_A, t)$ 和 $(u, i_B, t)$ 之间的分值差 [4]。因此，所有不依赖于 $i$ 的加性项消失，FM模型方程等价于：

$$
\hat{y}(x) = w_i + \langle v_u, v_i \rangle + \frac{1}{|B_u^{t-1}|} \sum_{l \in B_u^{t-1}} \langle v_i, v_l \rangle \qquad (21)
$$

现在可以看出，原始FPMC模型 [4] 和FM模型几乎相同，仅有的区别在于额外的item偏置 $w_i$ ，以及FM模型中item在 $(u, i)$ 和 $(i, l)$ 交互之间的因子化参数共享。

> [!NOTE]
>
> FPMC不是很了解，但是也不用深入了解

### 5.5. 总结

1. 像PARAFAC或MF这样的**标准因子分解模型** 不像 FM因子分解机那样是通用预测模型。相反，**它们要求特征向量被划分为 $m$ 个部分，每个部分中恰好有一个元素为1，其余为0**。
2. 有许多**针对单一任务设计的专门化因子分解模型**提案。我们已经证明，因子分解机**仅通过特征提取**即可模仿许多最成功的因子分解模型（包括MF、PARAFAC、SVD++、PITF、FPMC），这使得FM在实践中易于应用。


## 6. 结论与未来工作

在本文中，我们提出了因子分解机。**FM将SVM的通用性与因子分解模型的优势结合在一起**。与SVM相比，（1）FM能够在极大稀疏度下估计参数，（2）模型方程是**线性的且仅依赖于模型参数**，因此（3）可以**在原始空间中直接优化**。FM的表达能力与多项式SVM相当。与像PARAFAC这样的张量因子分解模型不同，**FM是一种可以处理任意实值向量的通用预测器**。此外，仅通过在输入特征向量中使用正确的指示符，FM就与许多仅适用于特定任务的专门化最先进模型（包括有偏MF、SVD++、PITF和FPMC）相同或非常相似。


## 参考文献

[1] R. A. Harshman, "Foundations of the parafac procedure: models and conditions for an 'exploratory' multimodal factor analysis." UCLA Working Papers in Phonetics, pp. 1–84, 1970.

[2] Y. Koren, "Factorization meets the neighborhood: a multifaceted collaborative filtering model," in KDD '08: Proceeding of the 14th ACM SIGKDD international conference on Knowledge discovery and data mining. New York, NY, USA: ACM, 2008, pp. 426–434.

[3] S. Rendle and L. Schmidt-Thieme, "Pairwise interaction tensor factorization for personalized tag recommendation," in WSDM '10: Proceedings of the third ACM international conference on Web search and data mining. New York, NY, USA: ACM, 2010, pp. 81–90.

[4] S. Rendle, C. Freudenthaler, and L. Schmidt-Thieme, "Factorizing personalized markov chains for next-basket recommendation," in WWW '10: Proceedings of the 19th international conference on World wide web. New York, NY, USA: ACM, 2010, pp. 811–820.

[5] T. Joachims, "Optimizing search engines using clickthrough data," in KDD '02: Proceedings of the eighth ACM SIGKDD international conference on Knowledge discovery and data mining. New York, NY, USA: ACM, 2002, pp. 133–142.

[6] V. N. Vapnik, The nature of statistical learning theory. New York, NY, USA: Springer-Verlag New York, Inc., 1995.

[7] N. Srebro, J. D. M. Rennie, and T. S. Jaakola, "Maximum-margin matrix factorization," in Advances in Neural Information Processing Systems 17. MIT Press, 2005, pp. 1329–1336.

[8] R. Salakhutdinov and A. Mnih, "Bayesian probabilistic matrix factorization using Markov chain Monte Carlo," in Proceedings of the International Conference on Machine Learning, vol. 25, 2008.
