# FFM: Field-aware Factorization Machines for CTR Prediction

> Yuchin Juan, Yong Zhuang, Wei-Sheng Chin, Chih-Jen Lin | Criteo Research / Carnegie Mellon Univ. / National Taiwan Univ.



本文分享了面向点击率预测的域感知因子分解机 FFM（Field-aware Factorization Machines）的研究成果。核心内容：

- FFM 是 FM（Factorization Machines）的一种变体，通过为每个特征-域组合学习独立的 latent vector 来捕获特征交互，在 CTR（Click-Through Rate）预测竞赛中优于现有模型
- 提出了 FFM 的高效训练实现，包括并行优化算法和 early-stopping 技术以避免过拟合
- 在 Criteo 和 Avazu 等大规模 CTR 数据集上进行了全面的实验比较，证明了 FFM 相对于 LM、Poly2 和 FM 的优势
- 开源发布了 LIBFFM 软件包供公众使用

关键发现：FFM 在特定分类问题（尤其是包含类别型特征且高度稀疏的数据）上显著优于 LM、Poly2 和 FM，但训练时间较长；对于数值型数据，FFM 的优势不明显。

---



## 摘要

点击率（CTR，Click-Through Rate）预测在计算广告中扮演着重要角色。基于二阶多项式映射（Poly2，Degree-2 Polynomial Mapping）和因子分解机（FMs，Factorization Machines）的模型被广泛用于这一任务。最近，FM 的一种变体——域感知因子分解机（FFMs，Field-aware Factorization Machines），在一些全球性的 CTR 预测竞赛中超越了现有模型。基于我们赢得其中两项竞赛的经验，本文将 FFMs 确立为一种对包括 CTR 预测在内的大规模稀疏数据分类的有效方法。首先，我们提出了训练 FFMs 的高效实现。然后，我们全面分析了 FFMs 并将该方法与竞争模型进行了比较。实验表明，FFMs 对某些分类问题非常有用。最后，我们发布了 FFMs 的软件包供公众使用。

**关键词**：机器学习；点击率预测；计算广告；因子分解机



## 1. 引言

点击率（CTR）预测在广告行业中扮演着重要角色 [1, 2, 3]。逻辑回归可能是该任务中使用最广泛的模型 [3]。给定包含 $m$ 个实例 $(y_i, \mathbf{x}_i),\ i = 1, \ldots, m$ 的数据集，其中 $y_i$ 是标签， $\mathbf{x}_i$ 是 $n$ 维特征向量，模型 $\mathbf{w}$ 通过求解以下优化问题获得：

$$
\min_{\mathbf{w}} \quad \frac{\lambda}{2} \|\mathbf{w}\|_2^2 + \sum_{i=1}^{m} \log(1 + \exp(-y_i \phi_{\text{LM}}(\mathbf{w}, \mathbf{x}_i))). \qquad (1)
$$

在问题 (1) 中， $\lambda$ 是正则化参数，在损失函数中我们考虑线性模型：

$$
\phi_{\text{LM}}(\mathbf{w}, \mathbf{x}) = \mathbf{w} \cdot \mathbf{x}.
$$

学习特征交互的效应对 CTR 预测至关重要；例如，参见 [1]。这里，我们考虑表 1 中的人工数据集，以更好地理解特征交互。来自 Gucci 的广告在 Vogue 上有特别高的 CTR。然而，这一信息对线性模型来说很难学习，因为它们分别学习 Gucci 和 Vogue 两个权重。为了解决这个问题，两种模型已被用于学习特征交互的效应。第一种模型，二阶多项式映射（Poly2）[4, 5]，为每个特征交互学习一个专用权重。第二种模型，因子分解机（FMs）[6]，通过将特征交互分解为两个 latent vector 的内积来学习其效应。我们将在第 2 节讨论 Poly2 和 FMs 的详细内容。

**表 1：一个人工 CTR 数据集，其中 $+$ （ $-$ ）表示点击（未点击）展示次数。**

<img src="/Users/dazhang/PycharmProject/Papers/3-RecSys/.picture/image-20260728180711799.png" alt="image-20260728180711799" style="zoom: 50%;" />

FM 的一种变体——成对交互张量分解（PITF，Pairwise Interaction Tensor Factorization）[7]——被提出用于个性化标签推荐。在 KDD Cup 2012 中，"Team Opera Solutions" [8] 提出了 PITF 的一种泛化形式，称为"因子模型"。由于该术语过于宽泛且容易与因子分解机混淆，本文中我们将其称为"域感知因子分解机"（FFMs）。PITF 和 FFM 的区别在于，PITF 考虑了三个特殊域，包括"用户"、"item"和"标签"，而 FFM 更具通用性。由于 [8] 是关于竞赛的整体解决方案，其对 FFM 的讨论有限。我们可以从 [8] 中得出以下结果：

1. 他们使用随机梯度法（SG）求解优化问题。为避免过拟合，他们只训练一个 epoch。
2. FFM 在他们尝试的六个模型中表现最好。

在本文中，我们的目标是具体地将 FFM 确立为 CTR 预测的有效方法。我们的主要结果如下：

- 尽管 FFM 在 [8] 中已被证明有效，但这项工作可能是唯一已发表的将 FFMs 应用于 CTR 预测问题的研究。为了进一步证明 FFMs 在 CTR 预测上的有效性，我们展示了使用 FFM 作为主要模型赢得由 Criteo 和 Avazu 主办的两项全球性 CTR 竞赛的经验。
- 我们将 FFMs 与两个相关模型 Poly2 和 FMs 进行了比较。我们首先从概念上讨论为什么 FFMs 可能优于它们，然后进行实验以观察在准确率和训练时间方面的差异。
- 我们提出了训练 FFMs 的技术。它们包括一种有效的 FFMs 并行优化算法和使用 early-stopping 来避免过拟合。
- 为了使 FFMs 可供公众使用，我们发布了一个开源软件。

本文的组织结构如下。在第 3 节介绍 FFMs 及其实现之前，我们在第 2 节讨论了两个现有模型 Poly2 和 FMs。比较 FFMs 与其他模型的实验在第 4 节。最后，结论和未来方向在第 5 节。

本文实验所用的代码和 LIBFFM 软件包分别可在以下地址获取：
http://www.csie.ntu.edu.tw/~cjlin/ffm/exps
http://www.csie.ntu.edu.tw/~cjlin/libffm



## 2. Poly2 和 FM

Chang 等人 [4] 已经证明，二阶多项式映射通常能有效捕获特征交互的信息。进一步地，他们证明通过对二阶映射的显式形式应用线性模型，训练和测试时间可以比使用核方法快得多。这种方法被称为 Poly2，为每个特征对学习一个权重：

$$
\phi_{\text{Poly2}}(\mathbf{w}, \mathbf{x}) = \sum_{j_1=1}^{n} \sum_{j_2=j_1+1}^{n} w_{h(j_1, j_2)} x_{j_1} x_{j_2}, \qquad (2)
$$

其中 $h(j_1, j_2)$ 是一个将 $j_1$ 和 $j_2$ 编码为自然数的函数。计算 (2) 的复杂度为 $O(\bar{n}^2)$ ，其中 $\bar{n}$ 是每个实例的非零元素平均数量。

[6] 中提出的 FMs 为每个特征隐式地学习一个 latent vector。每个 latent vector 包含 $k$ 个 latent factor，其中 $k$ 是用户指定的参数。然后，特征交互的效应通过两个 latent vector 的内积来建模：

$$
\phi_{\text{FM}}(\mathbf{w}, \mathbf{x}) = \sum_{j_1=1}^{n} \sum_{j_2=j_1+1}^{n} (\mathbf{w}_{j_1} \cdot \mathbf{w}_{j_2}) x_{j_1} x_{j_2}. \qquad (3)
$$

变量数量为 $n \times k$ ，因此直接计算 (3) 需要 $O(\bar{n}^2 k)$ 时间。按照 [6] 的方法，通过将 (3) 重写为

$$
\phi_{\text{FM}}(\mathbf{w}, \mathbf{x}) = \frac{1}{2} \sum_{j=1}^{n} (\mathbf{s} - \mathbf{w}_j x_j) \cdot \mathbf{w}_j x_j,
$$

其中

$$
\mathbf{s} = \sum_{j'=1}^{n} \mathbf{w}_{j'} x_{j'},
$$

复杂度降低到 $O(\bar{n} k)$ 。

Rendle [6] 解释了为什么在数据稀疏时 FMs 可能优于 Poly2。这里我们使用表 1 中的数据集给出类似的说明。例如，对于配对 (ESPN, Adidas) 只有一个负训练数据。对于 Poly2，可能会为该配对学习到一个非常负的权重 $w_{\text{ESPN,Adidas}}$ 。对于 FMs，由于 (ESPN, Adidas) 的预测由 $\mathbf{w}_{\text{ESPN}} \cdot \mathbf{w}_{\text{Adidas}}$ 决定，并且 $\mathbf{w}_{\text{ESPN}}$ 和 $\mathbf{w}_{\text{Adidas}}$ 也从其他配对（例如 (ESPN, Nike)、(NBC, Adidas)）中学习，预测可能更准确。另一个例子是没有 (NBC, Gucci) 的训练数据。对于 Poly2，该配对的预测是平凡的，但对于 FMs，由于 $\mathbf{w}_{\text{NBC}}$ 和 $\mathbf{w}_{\text{Gucci}}$ 可以从其他配对学习，仍然可以进行有意义的预测。

注意，在 Poly2 中，实现 $h(j_1, j_2)$ 的朴素方法是将每对特征视为一个新特征 [4]。这种方法需要模型大小为 $O(n^2)$ ，由于 $n$ 通常非常大，对 CTR 预测来说通常不实用。Vowpal Wabbit（VW）[9]——一个广泛使用的机器学习软件包——通过对 $j_1$ 和 $j_2$ 进行哈希来解决这个问题。我们的实现与 VW 的方法类似。具体地，

$$
h(j_1, j_2) = \left(\frac{1}{2}(j_1 + j_2)(j_1 + j_2 + 1) + j_2\right) \bmod B,
$$

其中模型大小 $B$ 是用户指定的参数。

在本文中，为了简化公式，我们不包含线性项和偏置项。然而，在第 4 节中，我们在部分实验中包含了它们。



## 3. FFM

FFM 的思想源于为具有个性化标签的推荐系统提出的 PITF [7]。在 PITF 中，他们假设三个可用域：用户、item 和标签，并在单独的 latent 空间中分解 (用户, item)、(用户, 标签) 和 (item, 标签)。在 [8] 中，他们将 PITF 泛化到更多域（例如 AdID、AdvertiserID、UserID、QueryID）并有效地将其应用于 CTR 预测。由于 [7] 旨在推荐系统且仅限于三个特定域（用户、item 和标签），而 [8] 缺乏对 FFM 的详细讨论，因此在本节中我们对 CTR 预测中的 FFMs 进行更全面的研究。

对于像表 1 这样的大多数 CTR 数据集，"特征"可以被分组为"域"。在我们的示例中，三个特征 ESPN、Vogue 和 NBC 属于 Publisher 域，另外三个特征 Nike、Gucci 和 Adidas 属于 Advertiser 域。FFM 是利用这一信息的 FM 变体。为了说明 FFM 的工作原理，我们考虑以下新示例：

| 点击 | Publisher (P) | Advertiser (A) | Gender (G) |
|:---:|:---:|:---:|:---:|
| 是 | ESPN | Nike | 男 |

回忆一下，对于 FMs， $\phi_{\text{FM}}(\mathbf{w}, \mathbf{x})$ 是

$$
\mathbf{w}_{\text{ESPN}} \cdot \mathbf{w}_{\text{Nike}} + \mathbf{w}_{\text{ESPN}} \cdot \mathbf{w}_{\text{Male}} + \mathbf{w}_{\text{Nike}} \cdot \mathbf{w}_{\text{Male}}.
$$

在 FMs 中，每个特征只有一个 latent vector 来学习与任何其他特征的 latent 效应。以 ESPN 为例， $\mathbf{w}_{\text{ESPN}}$ 用于学习与 Nike（ $\mathbf{w}_{\text{ESPN}} \cdot \mathbf{w}_{\text{Nike}}$ ）和 Male（ $\mathbf{w}_{\text{ESPN}} \cdot \mathbf{w}_{\text{Male}}$ ）的 latent 效应。然而，由于 Nike 和 Male 属于不同的域，(ESPN, Nike) 和 (ESPN, Male) 的 latent 效应可能不同。

在 FFMs 中，每个特征有多个 latent vector。根据其他特征的域，选择其中一个来进行内积运算。在我们的示例中， $\phi_{\text{FFM}}(\mathbf{w}, \mathbf{x})$ 是

$$
\mathbf{w}_{\text{ESPN},A} \cdot \mathbf{w}_{\text{Nike},P} + \mathbf{w}_{\text{ESPN},G} \cdot \mathbf{w}_{\text{Male},P} + \mathbf{w}_{\text{Nike},G} \cdot \mathbf{w}_{\text{Male},A}.
$$

我们看到，为了学习 (ESPN, Nike) 的 latent 效应，使用 $\mathbf{w}_{\text{ESPN},A}$ 是因为 Nike 属于 Advertiser 域，使用 $\mathbf{w}_{\text{Nike},P}$ 是因为 ESPN 属于 Publisher 域。同样，为了学习 (ESPN, Male) 的 latent 效应，使用 $\mathbf{w}_{\text{ESPN},G}$ 是因为 Male 属于 Gender 域，使用 $\mathbf{w}_{\text{Male},P}$ 是因为 ESPN 属于 Publisher 域。数学上，

$$
\phi_{\text{FFM}}(\mathbf{w}, \mathbf{x}) = \sum_{j_1=1}^{n} \sum_{j_2=j_1+1}^{n} (\mathbf{w}_{j_1,f_2} \cdot \mathbf{w}_{j_2,f_1}) x_{j_1} x_{j_2}, \qquad (4)
$$

其中 $f_1$ 和 $f_2$ 分别是 $j_1$ 和 $j_2$ 的域。如果 $f$ 是域的数量，则 FFMs 的变量数量为 $nfk$ ，计算 (4) 的复杂度为 $O(\bar{n}^2 k)$ 。值得注意的是，在 FFMs 中，由于每个 latent vector 只需要学习与特定域的效应，通常 $k_{\text{FFM}} \ll k_{\text{FM}}$ 。

表 2 比较了不同模型的变量数量和计算复杂度。

**表 2：LM、Poly2、FM 和 FFM 的变量数量和预测复杂度比较。**

<img src="/Users/dazhang/PycharmProject/Papers/3-RecSys/.picture/image-20260728180729707.png" alt="image-20260728180729707" style="zoom:50%;" />

### 3.1 求解优化问题

优化问题与 (1) 相同，只是将 $\phi_{\text{LM}}(\mathbf{w}, \mathbf{x})$ 替换为 $\phi_{\text{FFM}}(\mathbf{w}, \mathbf{x})$ 。按照 [7, 8]，我们使用随机梯度法（SG）。最近，一些自适应学习率调度方法如 [10, 11] 被提出以加速 SG 的训练过程。我们使用 AdaGrad（Adaptive Gradient）[10]，因为 [12] 已经证明其在矩阵分解（FFMs 的一个特例）上的有效性。

在 SG 的每一步中，采样一个数据点 $(y, \mathbf{x})$ 来更新 (4) 中的 $\mathbf{w}_{j_1,f_2}$ 和 $\mathbf{w}_{j_2,f_1}$ 。注意，由于 $\mathbf{x}$ 在我们的应用中高度稀疏，我们只更新非零值对应的维度。首先，子梯度为：

$$
\mathbf{g}_{j_1,f_2} \equiv \nabla_{\mathbf{w}_{j_1,f_2}} f(\mathbf{w}) = \lambda \cdot \mathbf{w}_{j_1,f_2} + \kappa \cdot \mathbf{w}_{j_2,f_1} x_{j_1} x_{j_2}, \qquad (5)
$$

$$
\mathbf{g}_{j_2,f_1} \equiv \nabla_{\mathbf{w}_{j_2,f_1}} f(\mathbf{w}) = \lambda \cdot \mathbf{w}_{j_2,f_1} + \kappa \cdot \mathbf{w}_{j_1,f_2} x_{j_1} x_{j_2}, \qquad (6)
$$

其中

$$
\kappa = \frac{\partial \log(1 + \exp(-y \phi_{\text{FFM}}(\mathbf{w}, \mathbf{x})))}{\partial \phi_{\text{FFM}}(\mathbf{w}, \mathbf{x})} = \frac{-y}{1 + \exp(y \phi_{\text{FFM}}(\mathbf{w}, \mathbf{x}))}.
$$

<img src="/Users/dazhang/PycharmProject/Papers/3-RecSys/.picture/image-20260728180742914.png" alt="image-20260728180742914" style="zoom:50%;" />

**算法 1：使用 SG 训练 FFM**

1. 初始化 $\mathbf{G} \in \mathbb{R}^{n \times f \times k}$ 为全 1 张量
2. 运行以下循环 $t$ 个 epoch
3. **for** $i \in \{1, \ldots, m\}$ **do**
4. $\quad$ 采样一个数据点 $(y, \mathbf{x})$
5. $\quad$ 计算 $\kappa$
6. $\quad$ **for** $j_1 \in$ $\{1, \ldots, n\}$ 中的非零项 **do**
7. $\quad\quad$ **for** $j_2 \in \{j_1 + 1, \ldots, n\}$ 中的非零项 **do**
8. $\quad\quad\quad$ 通过 (5) 和 (6) 计算子梯度
9. $\quad\quad\quad$ **for** $d \in \{1, \ldots, k\}$ **do**
10. $\quad\quad\quad\quad$ 通过 (7) 和 (8) 更新梯度和
11. $\quad\quad\quad\quad$ 通过 (9) 和 (10) 更新模型

其次，对于每个坐标 $d = 1, \ldots, k$ ，梯度平方和被累积：

$$
(G_{j_1,f_2})_d \leftarrow (G_{j_1,f_2})_d + (g_{j_1,f_2})_d^2 \qquad (7)
$$

$$
(G_{j_2,f_1})_d \leftarrow (G_{j_2,f_1})_d + (g_{j_2,f_1})_d^2 \qquad (8)
$$

最后， $(\mathbf{w}_{j_1,f_2})_d$ 和 $(\mathbf{w}_{j_2,f_1})_d$ 通过以下公式更新：

$$
(\mathbf{w}_{j_1,f_2})_d \leftarrow (\mathbf{w}_{j_1,f_2})_d - \frac{\eta}{\sqrt{(G_{j_1,f_2})_d}} (g_{j_1,f_2})_d \qquad (9)
$$

$$
(\mathbf{w}_{j_2,f_1})_d \leftarrow (\mathbf{w}_{j_2,f_1})_d - \frac{\eta}{\sqrt{(G_{j_2,f_1})_d}} (g_{j_2,f_1})_d, \qquad (10)
$$

其中 $\eta$ 是用户指定的学习率。 $\mathbf{w}$ 的初始值从 $[0, 1/\sqrt{k}]$ 的均匀分布中随机采样。 $\mathbf{G}$ 的初始值设为 1，以防止 $(G_{j_1,f_2})_d^{-1/2}$ 过大。整个过程如算法 1 所示。

经验上，我们发现将每个实例归一化为单位长度可以使测试准确率略好且对参数不敏感。

### 3.2 共享内存系统上的并行化

现代计算机广泛配备多核 CPU。如果这些核心被充分利用，训练时间可以显著减少。许多 SG 的并行化方法已被提出。在本文中，我们应用 Hogwild! [13]，它允许每个线程独立运行而无需任何锁。具体地，算法 1 中第 3 行的 for 循环被并行化。

在第 4.4 节中，我们进行了大量实验来研究并行化的有效性。

### 3.3 添加域信息

考虑广泛使用的 LIBSVM 数据格式：

```
label feat1:val1 feat2:val2 · · · ,
```

其中每对 $(\text{feat}, \text{val})$ 表示特征索引和值。对于 FFMs，我们将上述格式扩展为：

```
label field1:feat1:val1 field2:feat2:val2 · · ·
```

也就是说，我们必须为每个特征分配对应的域。这种分配在某些类型的特征上很容易，但对其他特征可能不可行。我们讨论三类典型特征的这个问题。

**类别型特征**

对于线性模型，类别型特征通常被转换为多个二元特征。对于数据实例

```
Yes P:ESPN A:Nike G:Male,
```

我们生成以下 LIBSVM 格式：

```
Yes P-ESPN:1 A-Nike:1 G-Male:1
```

注意，根据类别型特征中可能值的数量，生成相同数量的二元特征，且每次只有一个值为 1。在 LIBSVM 格式中，值为零的特征不被存储。我们对所有模型应用相同的设置，因此在本文中，每个类别型特征被转换为多个二元特征。为了添加域信息，我们可以将每个类别视为一个域。然后上述实例变为：

```
Yes P:P-ESPN:1 A:A-Nike:1 G:G-Male:1.
```

**数值型特征**

考虑以下预测论文是否会被会议接受的示例。我们使用三个数值型特征："会议录取率（AR）"、"作者 h-index（Hidx）"和"作者引用次数（Cite）"：

| 接受 | AR | Hidx | Cite |
|:---:|:---:|:---:|:---:|
| 是 | 45.73 | 2 | 3 |
| 否 | 1.04 | 100 | 50,000 |

有两种可能的方式来分配域。一种朴素的方法是将每个特征视为一个虚拟域，因此生成的数据为：

```
Yes AR:AR:45.73 Hidx:Hidx:2 Cite:Cite:3
```

然而，虚拟域可能没有信息量，因为它们仅仅是特征的副本。

另一种可能的方式是将每个数值型特征离散化为类别型特征。然后，我们可以使用与类别型特征相同的设置来添加域信息。生成的数据如下：

```
Yes AR:45:1 Hidx:2:1 Cite:3:1,
```

其中 AR 特征被四舍五入为整数。主要缺点通常是不容易确定最佳的离散化设置。例如，我们可能将 45.73 转换为"45.7"、"45"、"40"，甚至是" $\text{int}(\log(45.73))$ "。此外，离散化后我们可能会丢失一些信息。

**单域特征**

在某些数据集上，所有特征都属于单个域，因此为特征分配域没有意义。通常这种情况发生在 NLP 数据集上。考虑以下预测句子是否表达好心情的示例：

| 好心情 | 句子 |
|:---:|:---|
| 是 | Hooray! Our paper is accepted! |
| 否 | Well, our paper is rejected.. |

在这个示例中，唯一的域是"句子"。如果我们为所有词分配这个域，则 FFMs 简化为 FMs。读者可能会问是否可以像数值型特征那样使用虚拟域。回顾 FFMs 的模型大小为 $O(nfk)$ 。使用虚拟域是不实用的，因为 $f = n$ 且 $n$ 通常很大。



## 4. 实验

在本节中，我们首先在第 4.1 节提供实验设置的详细信息。然后，我们研究参数的影响。我们发现与 LM 或 Poly2 不同，FFM 对 epoch 数量敏感。因此，在第 4.3 节中，我们在提出 early-stopping 技巧之前详细讨论了这个问题。并行化的加速效果在第 4.4 节中研究。在检查了 FFMs 的各种特性之后，我们在第 4.5-4.6 节将 FFMs 与包括 Poly2 和 FMs 在内的其他模型进行了比较。它们都使用相同的 SG 方法实现，因此除了准确率之外，我们还可以公平地比较它们的训练时间。此外，我们在比较中包含了最先进的软件包 LIBLINEAR [14] 和 LIBFM [15]，分别用于训练 LM/Poly2 和 FMs。

### 4.1 实验设置

**数据集**

我们主要考虑来自 Kaggle 竞赛的两个 CTR 数据集 Criteo 和 Avazu，尽管在第 4.6 节中考虑了更多数据集。对于特征工程，我们主要应用我们的获胜解决方案但移除了复杂的组件。例如，我们赢得 Avazu 的解决方案包括 20 个模型的集成，但这里我们只使用最简单的一个。其他细节请查看我们的实验代码。应用哈希技巧生成 $10^6$ 个特征。两个数据集的统计信息如下：

| 数据集 | 实例数 | 特征数 | 域数 |
|:---:|:---:|:---:|:---:|
| Criteo | 45,840,617 | $10^7$ | 39 |
| Avazu | 40,428,967 | $10^7$ | 33 |

对于两个数据集，测试集的标签未公开，因此我们将可用数据分为两个集合用于训练和验证。

数据划分遵循测试集的获取方式：对于 Criteo，最后 6,040,618 行用作验证集；对于 Avazu，我们选择最后 4,218,938 行。我们使用以下术语来表示问题的不同集合：

- Va：上述验证集。
- Tr：从原始训练数据中排除验证集后的新训练集。
- TrVa：原始训练集。
- Te：原始测试集。标签未发布，因此我们必须将预测提交到原始评估系统以获得分数。为避免过拟合测试集，竞赛组织者将此数据集分为两个子集："public set"——分数在竞赛期间可见，"private set"——分数在竞赛结束后可用。最终排名由 private set 决定。

例如，CriteoVa 表示来自 Criteo 的验证集。

**平台**

所有实验在具有两个 Intel Xeon E5-2620 2.0GHz 处理器（12 个物理核心）和 128 GB 内存的 Linux 工作站上进行。

**评估**

根据模型，我们将 (1) 中的 $\phi(\mathbf{w}, \mathbf{x})$ 替换为第 1-3 节介绍的 $\phi_{\text{LM}}(\mathbf{w}, \mathbf{x})$ 、 $\phi_{\text{Poly2}}(\mathbf{w}, \mathbf{x})$ 、 $\phi_{\text{FM}}(\mathbf{w}, \mathbf{x})$ 或 $\phi_{\text{FFM}}(\mathbf{w}, \mathbf{x})$ 。对于评估标准，我们考虑 logistic loss，定义为：

$$
\text{logloss} = \frac{1}{m} \sum_{i=1}^{m} \log(1 + \exp(-y_i \phi(\mathbf{w}, \mathbf{x}_i))),
$$

其中 $m$ 是测试实例的数量。

**实现**

我们全部用 C++ 实现了 LM、Poly2、FMs 和 FFMs。对于 FMs 和 FFMs，我们使用 SSE（Streaming SIMD Extensions）指令来提高内积运算的效率。第 3.2 节讨论的并行化通过 OpenMP（Open Multi-Processing）[16] 实现。我们的实现包含线性项和偏置项，因为它们在某些数据集上提高了性能。这些项通常应该被使用，因为我们很少看到它们有害。

注意，为了代码可扩展性，无论使用何种模型，域信息都会被存储。对于非 FFM 模型，通过使用没有域信息的更简单数据结构，实现可能会稍微更快，但我们从实验中得出的结论应该保持不变。

### 4.2 参数影响

我们进行了实验来研究 $k$ 、 $\lambda$ 和 $\eta$ 的影响。结果可在图 1 中找到。关于参数 $k$ ，图 1a 中的结果表明它对 logloss 影响不大。在图 1b 中，我们展示了 $\lambda$ 与 logloss 之间的关系。如果 $\lambda$ 太大，模型无法达到良好的性能。相反，当 $\lambda$ 较小时，模型获得更好的结果，但容易过拟合数据。我们观察到训练 logloss 持续下降。对于参数 $\eta$ ，图 1c 表明如果我们使用较小的 $\eta$ ，FFMs 将缓慢达到其最佳性能。然而，使用较大的 $\eta$ ，FFMs 能够快速降低 logloss，但随后会发生过拟合。从图 1b 和 1c 的结果来看，需要 early-stopping，这将在第 4.3 节讨论。

<img src="/Users/dazhang/PycharmProject/Papers/3-RecSys/.picture/image-20260728180807728.png" alt="image-20260728180807728" style="zoom:50%;" />

> **图 1： $\lambda$ 、 $\eta$ 和 $k$ 对 FFMs 的影响。为了加快实验速度，我们从 CriteoTr 和 CriteoVa 中随机选择 10% 的实例分别作为训练集和测试集。**（a）不同 $k$ 值下的平均运行时间（秒）和最佳 logloss。由于我们使用 SSE 指令， $k=1,2,4$ 的运行时间大致相同。（b） $\lambda$ 的影响。（c） $\eta$ 的影响。

### 4.3 Early Stopping

Early stopping 在达到训练数据上的最佳结果之前终止训练过程，可用于避免许多机器学习问题的过拟合 [17, 18, 19]。对于 FFM，我们使用的策略是：

1. 将数据集分为训练集和验证集。
2. 在每个 epoch 结束时，使用验证集计算 loss。
3. 如果 loss 上升，记录 epoch 数量。停止或转到步骤 4。
4. 如果需要，使用完整数据集以步骤 3 中获得的 epoch 数量重新训练模型。

应用 early stopping 的一个困难是 logloss 对 epoch 数量敏感。因此，验证集上的最佳 epoch 可能不是测试集上的最佳 epoch。我们尝试了其他方法来避免过拟合，如 lazy update 和基于 ALS 的优化方法。然而，结果不如使用验证集的 early stopping 成功。

### 4.4 加速

由于 SG 的并行化可能导致不同的收敛行为，我们在图 2 中使用不同数量的线程进行了实验。结果表明，我们的并行化仍然导致类似的收敛行为。利用这一特性，我们可以定义加速比为：

$$
\text{加速比} = \frac{\text{单线程一个 epoch 的运行时间}}{\text{多线程一个 epoch 的运行时间}}.
$$

图 3 中的结果表明，当线程数量较小时，加速效果良好。然而，如果使用许多线程，加速效果不会改善太多。一个解释是，如果两个或更多线程尝试访问相同的内存地址，其中一个必须等待其轮次。当使用更多线程时，这种冲突会更频繁地发生。

<img src="/Users/dazhang/PycharmProject/Papers/3-RecSys/.picture/image-20260728180828907.png" alt="image-20260728180828907" style="zoom:50%;" />

> **图 2：使用不同数量线程的收敛情况。**

<img src="/Users/dazhang/PycharmProject/Papers/3-RecSys/.picture/image-20260728180837143.png" alt="image-20260728180837143" style="zoom:50%;" />

> **图 3：使用多线程的加速效果。我们分别使用 CriteoTr 和 CriteoVa 作为训练集和测试集。**

### 4.5 与 LM、Poly2 和 FMs 在两个 CTR 竞赛数据集上的比较

为了进行公平比较，我们为 LM、Poly2、FMs 和 FFMs 实现了相同的 SG 方法。此外，我们与两个最先进的软件包进行了比较：

- LIBLINEAR：一个广泛使用的线性模型软件包。对于 $L_2$ -正则化 logistic 回归，它实现了两种优化方法：Newton 法求解原始问题，坐标下降（CD，Coordinate Descent）法求解对偶问题。我们使用两者来检查优化方法如何影响性能；参见本小节末尾的讨论。此外，LIBLINEAR 的现有 Poly2 扩展不支持哈希技巧，因此我们进行了适当的修改，并在本文中将其表示为 LIBLINEAR-Hash。
- LIBFM：作为一个广泛使用的因子分解机库，它支持三种优化方法，包括随机梯度法（SG）、交替最小二乘（ALS，Alternating Least Squares）和马尔可夫链蒙特卡洛（MCMC，Markov Chain Monte Carlo）。我们尝试了所有方法，发现 ALS 在 logloss 方面明显优于其他两种。因此，我们在实验中考虑 ALS。

对于所有模型中的参数，我们从一组网格点中选择在验证集上达到最佳性能的参数。每个优化算法都需要一个停止条件；对于 Newton 法和坐标下降（CD）法，我们使用 LIBLINEAR 的默认设置。对于其他每个模型，我们需要一个验证集来检查哪次迭代达到最佳验证分数。在获得最佳迭代次数后，我们重新训练模型直到该迭代次数。在 Criteo 和 Avazu 上的结果及所用参数列表可在表 3 中找到。显然，FFMs 在 logloss 方面优于其他模型，但它也需要比 LMs 和 FMs 更长的训练时间。另一方面，虽然 LMs 的 logloss 比其他模型差，但它明显更快。这些结果表明了 logloss 和速度之间的明确权衡。Poly2 是所有模型中最慢的。原因可能是 (2) 的计算开销。FM 在 logloss 和速度之间取得了良好的平衡。

对于 LIBFM，它在 Criteo 上的 logloss 表现与我们实现的 FMs 接近。然而，我们看到我们的实现明显更快。我们提供三个可能的原因：

- LIBFM 使用的 ALS 算法比我们使用的 SG 算法更复杂。
- 我们在 SG 中使用自适应学习率策略。
- 我们使用 SSE 指令来加速内积运算。

由于 logistic 回归是一个凸问题，理想情况下，对于 LM 或 Poly2，三种优化方法（SG、Newton 和 CD）如果收敛到全局最优，应该生成完全相同的模型。然而，实际上结果略有不同。特别是，在 Avazu 上通过 SG 的 LM 优于两个基于 LIBLINEAR 的模型。在我们的实现中，通过 SG 的 LM 只是松散地求解优化问题。因此，我们的实验表明，即使问题是凸的，优化方法的停止条件也会影响结果模型的性能。

**表 3：Criteo 和 Avazu 数据集上模型和实现的比较。此处使用的训练集是 CriteoTrVa 和 AvazuTrVa，测试集是 CriteoTe 和 AvazuTe。对于所有实验，使用单线程。public set 大约占测试数据的 20%，而 private set 包含其余部分。对于 Criteo，我们没有列出 Poly2-LIBLINEAR-Hash-Newton 的结果，因为该实验在 10 多天后仍未完成。注意，我们对不同的算法使用不同的停止条件，因此训练时间仅供参考。**

![image-20260728180854383](/Users/dazhang/PycharmProject/Papers/3-RecSys/.picture/image-20260728180854383.png)

**(a) Criteo**

**(b) Avazu**

### 4.6 在更多数据集上的比较

在上一节中，我们专注于两个竞赛数据集，但重要的是观察 FFMs 在其他数据集上的表现。为了回答这个问题，我们考虑了更多数据集进行比较，其中大多数不是 CTR 数据。注意，按照第 3.3 节的讨论，我们不考虑具有单域特征的数据集。原因是根据我们分配域的方式，FFMs 要么等效于 FMs，要么生成巨大的模型。这里我们简要介绍使用的数据集。

- KDD2010-bridge：该数据集包含数值型和类别型特征。
- KDD2012：该集合包含数值型和类别型特征。由于我们的评估是 logloss，我们将原始目标值"点击次数"转换为二元值"是否点击"。
- cod-rna：该集合仅包含数值型特征。
- ijcnn：该集合仅包含数值型特征。
- phishing：该集合仅包含类别型特征。
- adult：该数据集包含数值型和类别型特征。

对于 KDD2010-bridge、KDD2012 和 adult，我们简单地将所有数值型特征分别离散化为 29、13 和 94 个 bin。对于 cod-rna 和 ijcnn，特征全部为数值型，我们尝试了第 3.3 节中提到的两种方法来获取域信息：使用虚拟域和离散化。

对于参数选择，我们遵循第 4.5 节中的相同程序。我们将每个集合分为训练集、验证集和测试集；然后对于预测测试集，我们使用在验证集上达到最佳 logloss 的参数训练的模型。

每个数据集的统计信息和实验结果如表 4 所述。FFMs 在 KDD2010-bridge 和 KDD2012 上显著优于其他模型。这些数据集的共同特征是：

- 大多数特征是类别型的。
- 将类别型特征转换为许多二元特征后，结果集合高度稀疏。

然而，在 phishing 和 adult 上，FFM 没有明显优势。对于 phishing，原因可能是数据不稀疏，因此 FFM、FM 和 Poly2 性能接近；对于 adult，似乎特征交互没有用，因为所有模型都与线性模型表现相似。

**表 4：LM、Poly2、FM 和 FFMs 的比较。最佳 logloss 带下划线。**

<img src="/Users/dazhang/PycharmProject/Papers/3-RecSys/.picture/image-20260728180907465.png" alt="image-20260728180907465" style="zoom:50%;" />

当数据集仅包含数值型特征时，FFMs 可能没有明显优势。如果我们使用虚拟域，则 FFMs 不会优于 FMs，这一结果表明域信息没有帮助。另一方面，如果我们离散化数值型特征，虽然 FFMs 是所有模型中最好的，但性能比使用虚拟域差得多。我们总结了在不同类型数据集上应用 FFMs 的指南：

- FFMs 应该对包含类别型特征并转换为二元特征的数据集有效。
- 如果转换后的集合不够稀疏，FFMs 似乎带来的好处较少。
- 在数值型数据集上应用 FFMs 更加困难。



## 5. 结论与未来工作

在本文中，我们讨论了 FFMs 的高效实现。我们证明了对于某些类型的数据集，FFMs 在 logloss 方面优于三个著名模型 LM、Poly2 和 FM，代价是更长的训练时间。

对于未来工作，第 4.3 节中讨论的过拟合问题是我们计划研究的一个问题。此外，为了便于实现，我们使用 SG 作为优化方法。有趣的是看看其他优化方法（例如 Newton 法）在 FFMs 上的表现如何。



## 致谢

本工作部分得到了台湾 MOST 通过资助 104-2221-E-002-047-MY3 和 104-2622-E-002-012-CC2 以及台湾教育部通过资助 105R7872 的支持。



## 参考文献

[1] O. Chapelle, E. Manavoglu, and R. Rosales, "Simple and scalable response prediction for display advertising," ACM Transactions on Intelligent Systems and Technology, vol. 5, no. 4, pp. 61:1–61:34, 2015.

[2] H. B. McMahan, G. Holt, D. Sculley, M. Young, D. Ebner, J. Grady, L. Nie, T. Phillips, E. Davydov, D. Golovin, S. Chikkerur, D. Liu, M. Wattenberg, A. M. Hrafnkelsson, T. Boulos, and J. Kubica, "Ad click prediction: a view from the trenches," in Proceedings of the 19th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), 2013.

[3] M. Richardson, E. Dominowska, and R. Ragno, "Predicting clicks: estimating the click-through rate for new ADs," in Proceedings of the 16th international conference on World Wide Web, 2007.

[4] Y.-W. Chang, C.-J. Hsieh, K.-W. Chang, M. Ringgaard, and C.-J. Lin, "Training and testing low-degree polynomial data mappings via linear SVM," Journal of Machine Learning Research, vol. 11, pp. 1471–1490, 2010.

[5] T. Kudo and Y. Matsumoto, "Fast methods for kernel-based text analysis," in Proceedings of the 41st Annual Meeting of the Association of Computational Linguistics (ACL), 2003.

[6] S. Rendle, "Factorization machines," in Proceedings of IEEE International Conference on Data Mining (ICDM), pp. 995–1000, 2010.

[7] S. Rendle and L. Schmidt-Thieme, "Pairwise interaction tensor factorization for personalized tag recommendation," in Proceedings of the 3rd ACM International Conference on Web Search and Data Mining (WSDM), pp. 81–90, 2010.

[8] M. Jahrer, A. Töscher, J.-Y. Lee, J. Deng, H. Zhang, and J. Spoelstra, "Ensemble of collaborative filtering and feature engineered model for click through rate prediction," in KDD Cup 2012 Workshop, ACM, 2012.

[9] J. Langford, L. Li, and A. Strehl, "Vowpal Wabbit," 2007. https://github.com/JohnLangford/vowpal_wabbit/wiki.

[10] J. Duchi, E. Hazan, and Y. Singer, "Adaptive subgradient methods for online learning and stochastic optimization," Journal of Machine Learning Research, vol. 12, pp. 2121–2159, 2011.

[11] H. B. McMahan, "Follow-the-regularized-leader and mirror descent: Equivalence theorems and l1 regularization," in Proceedings of the 14th International Conference on Artificial Intelligence and Statistics (AISTATS), 2011.

[12] W.-S. Chin, Y. Zhuang, Y.-C. Juan, and C.-J. Lin, "A learning-rate schedule for stochastic gradient methods to matrix factorization," in Proceedings of the Pacific-Asia Conference on Knowledge Discovery and Data Mining (PAKDD), 2015.

[13] F. Niu, B. Recht, C. Ré, and S. J. Wright, "HOGWILD!: a lock-free approach to parallelizing stochastic gradient descent," in Advances in Neural Information Processing Systems 24 (J. Shawe-Taylor, R. Zemel, P. Bartlett, F. Pereira, and K. Weinberger, eds.), pp. 693–701, 2011.

[14] R.-E. Fan, K.-W. Chang, C.-J. Hsieh, X.-R. Wang, and C.-J. Lin, "LIBLINEAR: a library for large linear classification," Journal of Machine Learning Research, vol. 9, pp. 1871–1874, 2008.

[15] S. Rendle, "Factorization machines with libFM," ACM Transactions on Intelligent Systems and Technology (TIST), vol. 3, no. 3, p. 57, 2012.

[16] L. Dagum and R. Menon, "OpenMP: an industry standard API for shared-memory programming," IEEE Computational Science and Engineering, vol. 5, pp. 46–55, 1998.

[17] C. M. Bishop, Pattern Recognition and Machine Learning. Springer-Verlag New York, Inc., 2006.

[18] G. Raskutti, M. J. Wainwright, and B. Yu, "Early stopping and non-parametric regression: An optimal data-dependent stopping rule," Journal of Machine Learning Research, vol. 15, pp. 335–366, 2014.

[19] T. Zhang and B. Yu, "Boosting with early stopping: convergence and consistency," The Annals of Statistics, vol. 33, no. 4, pp. 1538–1579, 2005.
