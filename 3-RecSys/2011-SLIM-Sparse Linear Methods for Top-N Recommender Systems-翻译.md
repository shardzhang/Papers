# SLIM: Sparse Linear Methods for Top-N Recommender Systems

> Xia Ning, George Karypis | University of Minnesota


本文提出了稀疏线性方法（SLIM）用于Top-N推荐系统。核心内容：

- **稀疏线性模型**：通过学习一个稀疏的聚合系数矩阵 $\mathbf{W}$ ，将新item的推荐分数计算为其他item的加权聚合
- ** $\ell_1$ -范数正则化**：通过 $\ell_1$ -范数和 $\ell_2$ -范数联合正则化优化问题学习稀疏的 $W$ 矩阵，引入稀疏性使推荐速度显著加快
- **特征选择扩展**：提出fsSLIM方法，在学习前利用item-item相似度进行特征选择，大幅减少学习时间且不降低推荐质量
- **实验验证**：在八个真实数据集上，SLIM在推荐质量和运行时性能方面均显著优于现有的基于邻域和基于模型的Top-N推荐方法

关键发现：SLIM的稀疏性使其能够以接近itemkNN的速度生成推荐，同时推荐质量显著优于所有对比方法，对长尾分布也具有良好的鲁棒性。

---


## 摘要

本文聚焦于开发高效且有效的Top-N推荐系统算法。提出了一种新颖的稀疏线性方法（SLIM），通过聚合用户购买/评分档案来生成Top-N推荐。SLIM通过求解一个 $\ell_1$ -范数和 $\ell_2$ -范数正则化优化问题，**学习一个稀疏的聚合系数矩阵** $W$ 。 $W$ 被证明能产生高质量的推荐，且其稀疏性使SLIM能够非常快速地生成推荐。通过将SLIM方法与其他最先进的 Top-N 推荐方法进行一组全面的实验比较，结果表明SLIM在运行时性能和推荐质量方面均取得了显著改进。

**关键词**-Top-N推荐系统，稀疏线性方法， $\ell_1$ -范数正则化


## 1. 引言

电子商务的出现和快速发展通过提供海量item及详细的item信息，显著改变了人们对购买item的传统观念，从而使在线交易变得更加便捷。然而，随着符合顾客需求的item数量急剧增加，问题转变为如何有效且高效地帮助顾客识别最符合个人品味的item。**特别是，给定用户的购买/评分档案，为用户推荐一个排好序的item列表以鼓励额外购买，具有最多的应用场景**。这引出了广泛使用的Top-N推荐系统。

近年来，已开发出多种用于Top-N推荐的算法[1]。这些算法可分为两类：**基于邻域的协同过滤方法** 和 **基于模型的方法**。在基于邻域的方法中，基于item邻域的方法能够非常快速地生成推荐，但这是**以牺牲推荐质量为代价的**。另一方面，基于模型的方法，特别是那些**基于latent factor模型**的方法，在生成推荐时会产生更高的成本，但这些推荐的质量更高，并且已被证明在大型推荐任务上能达到最佳性能。

在本文中，我们提出了一种新颖的用于Top-N推荐的稀疏线性方法（SLIM），能够快速做出高质量的推荐。SLIM通过求解一个正则化优化问题，仅从用户购买/评分档案中为系统中的item学习一个**稀疏的系数矩阵**。系数矩阵中引入的稀疏性使其能够高效地生成推荐。**特征选择方法使SLIM能够大幅减少学习系数矩阵所需的时间**。此外，SLIM还可用于从评分进行Top-N推荐，这是推荐系统研究中一个较少被探索的方向。

SLIM方法同时满足了Top-N推荐系统对高质量和高效率的需求，因此更适合实时应用。我们在来自不同实际应用的各种数据集上进行了一组全面的实验。结果表明，SLIM以极高的速度产生了比最先进方法更好的推荐。此外，在使用评分进行Top-N推荐时，它也取得了良好的性能。

本文的其余部分组织如下。第2节简要回顾相关工作。第3节介绍定义和符号。第4节描述方法。第5节介绍实验所用的材料。第6节展示结果。最后，第7节是讨论和结论。


## 2. 相关工作

Top-N推荐系统用于电子商务应用中，推荐用户可能最喜欢的N个item的排序列表，并且在过去几年中得到了深入研究。Top-N推荐的方法可大致分为两类。

第一类是**基于邻域的协同过滤方法**[2]。对于某个特定用户，基于用户的k-最近邻（userkNN）协同过滤方法首先识别一组相似用户，然后根据那些相似用户购买了什么item来推荐Top-Nitem。类似地，基于item的k-最近邻（itemkNN）协同过滤方法首先为用户已购买的每个item识别一组相似item，然后基于那些相似item来推荐Top-Nitem。用户/item相似度是通过协同过滤的方式，应用某些**相似度度量**（例如，皮尔逊相关系数、余弦相似度）从用户-item购买/评分矩阵中计算得出的。基于item的方法的一个优势是，**由于item邻域是稀疏的**，它们能够高效地生成推荐。然而，它们的准确性较低，因为本质上没有学到关于item特征的知识来产生准确的Top-N推荐。

第二类是基于模型的方法，特别是**latent factor模型**，因为它们在大规模推荐任务中已取得了最先进的性能。latent factor模型的关键思想是将用户-item矩阵分解为（低秩的）用户因子和item因子，分别表示共同潜在空间中用户品味和item特征。对用户在某个item上的预测可以通过对应的用户因子和item因子的点积来计算。近年来提出了各种基于矩阵分解（MF）的方法来构建这样的latent factor模型。Cremonesi等人[3]提出了一种简单的基于纯奇异值分解（PureSVD）的矩阵分解方法，该方法通过用户-item矩阵的最主要的奇异向量来描述用户和item。Pan等人[4]和Hu等人[5]提出了一种加权正则化矩阵分解（WRMF）方法，将其表述为一个正则化最小二乘（LS）问题，其中使用权重矩阵来区分已观察到的购买/评分活动和**未观察到的活动的贡献**。Rennie[6]和Srebro[7]提出了一种最大间隔矩阵分解（MMMF）方法，该方法需要对用户-item矩阵进行低范数分解，并允许潜在空间具有无限维度。这是通过最小化由因子重构的用户-item矩阵的迹范数来实现的。Sindhwani等人[8]提出了一种加权非负矩阵分解（WNNMF）方法，其中他们对用户和item因子施加非负性约束，从而为模型赋予"基于部件"的可解释性。Hofmann[9]将概率潜在语义分析（PLSA）技术应用于协同过滤，已被证明等价于非负矩阵分解。PLSA引入了一个潜在空间，使得用户和item的共现（即，某个用户购买了某个item）可以表示为条件独立。Koren[10]提出了一种介于基于邻域的方法和MF之间的交叉方法。在他的方法中，item相似度与矩阵分解同时学习，从而利用两种方法的优势。

Top-N推荐**也被形式化为一个排序问题**。Rendle等人[11]提出了一种贝叶斯个性化排序（BPR）准则，该准则是从贝叶斯分析中得到的最大后验估计量，衡量用户购买过的item和其余item之间的排序差异。BPR可以很好地适用于itemkNN方法（BPRkNN）和MF方法（BPRMF）作为通用目标函数。


## 3. 定义与符号

在本文中，符号 $\mathbf{u}$ 和 $\mathbf{t}$ 将分别用于表示用户和item。单个用户和item将使用不同的下标表示（即 $u_i, t_j$ ）。系统中所有用户和item的集合将分别用 $U$ （ $|U| = m$ ）和 $T$ （ $|T| = n$ ）表示。完整的用户-item购买/评分集合将由一个大小为 $m \times n$ 的用户-item购买/评分矩阵 $\mathbf{A}$ 表示，其中第 $(i, j)$ 个条目（记为 $a_{ij}$ ）**为1或一个正值**（如果用户 $u_i$ 曾购买/评分过item $t_j$ ），否则该条目记为0。 $\mathbf{A}$ 的第 $i$ 行表示用户 $u_i$ 在所有item $T$ 上的购买/评分历史，该行记为 $\mathbf{a}_i^T$ 。 $\mathbf{A}$ 的第 $j$ 列表示 所有用户 $U$ 在item $t_j$ 上的购买/评分历史，该列记为 $\mathbf{a}_j$ 。

在本文中，所有向量（例如， $\mathbf{a}_i^T$ 和 $\mathbf{a}_j$ ）用粗体小写字母表示，所有矩阵（例如， $\mathbf{A}$ ）用大写字母表示。行向量通过带有转置上标 $T$ 表示，否则默认为列向量。预测/近似值用带有 $\sim$ 上标表示。如无歧义，我们将使用相应的矩阵/向量符号代替用户/item购买/评分档案。


## 4. 用于Top-N推荐的稀疏线性方法

### 4.1 用于Top-N推荐的SLIM

在本文中，我们提出了一种稀疏线性方法（SLIM）来进行Top-N推荐。在SLIM方法中，**用户 $u_i$ 在未购买/未评分的item $t_j$ 上的推荐分数被计算为用户 $u_i$ 已购买/评分item的稀疏聚合**，即：

$$
\tilde{a}_{ij} = \mathbf{a}_i^T \mathbf{w}_j, \qquad (1)
$$

其中 $a_{ij} = 0$ ， $\mathbf{w}_j$ 是一个稀疏的大小为 $n$ 的列向量，包含聚合系数。因此，SLIM所使用的模型可以表示为：

$$
\tilde{\mathbf{A}} = \mathbf{A}\mathbf{W}, \qquad (2)
$$

其中 $\mathbf{A}$ 是二元用户-item购买矩阵或用户-item评分矩阵， $\mathbf{W}$ 是一个 $n \times n$ 的稀疏聚合系数矩阵，其第 $j$ 列对应式(1)中的 $\mathbf{w}_j$ ，而 $\tilde{\mathbf{A}}$ 的每一行 $\tilde{\mathbf{a}}_i^T$ （ $\tilde{\mathbf{a}}_i^T = \mathbf{a}_i^T \mathbf{W}$ ）表示**用户 $u_i$ 在所有item上的推荐分数**。**对 $u_i$ 的Top-N推荐是通过将 $\tilde{\mathbf{a}}_i^T$ 中 $u_i$ 未购买/未评分的item按其推荐分数降序排序并推荐前N个item来完成的。**

### 4.2 SLIM中 $\mathbf{W}$ 的学习

我们将 $\mathbf{A}$ 中用户 $u_i$ 在item $t_j$ 上的购买/评分活动（即 $a_{ij}$ ）视为真实的item推荐分数。给定一个大小为 $m \times n$ 的用户-item购买/评分矩阵 $\mathbf{A}$ ，我们学习式(2)中的稀疏 $n \times n$ 矩阵 $\mathbf{W}$ ，作为以下正则化优化问题的最小化解：

$$
\min_{\mathbf{W}} \quad \frac{1}{2} \|\mathbf{A} - \mathbf{A}\mathbf{W}\|_F^2 + \frac{\beta}{2} \|\mathbf{W}\|_F^2 + \lambda \|\mathbf{W}\|_1
$$

$$
\text{subject to} \quad \mathbf{W} \geq 0
$$

$$
\text{diag}(\mathbf{W}) = 0, \qquad (3)
$$

其中 $\|\mathbf{W}\|_1 = \sum_{i=1}^{n} \sum_{j=1}^{n} |w_{ij}|$ 是逐元素的 $\ell_1$ -范数， $\|\cdot\|_F$ 是矩阵的Frobenius范数。在式(3)中， $\mathbf{A}\mathbf{W}$ 是由如式(2)中的稀疏线性模型估计的推荐分数矩阵（即 $\tilde{\mathbf{A}}$ ）。第一项 $\frac{1}{2} \|\mathbf{A} - \mathbf{A}\mathbf{W}\|_F^2$ （即残差平方和）衡量**线性模型对训练数据的拟合程度**，而 $\|\mathbf{W}\|_F^2$ 和 $\|\mathbf{W}\|_1$ 分别是 $\ell_F$ -范数和 $\ell_1$ -范数正则化项。常数 $\beta$ 和 $\lambda$ 是正则化参数。**参数越大，正则化越强**。对 $\mathbf{W}$ 施加非负性约束，使得学习到的 $\mathbf{W}$ **表示item之间的正向关系（如果存在**）。还施加了约束 $\text{diag}(\mathbf{W}) = 0$ ，**以避免平凡解**（即最优 $\mathbf{W}$ 为单位矩阵，使得item总是推荐自身以最小化 $\frac{1}{2} \|\mathbf{A} - \mathbf{A}\mathbf{W}\|_F^2$ ）。**此外，约束 $\text{diag}(\mathbf{W}) = 0$ 确保 $a_{ij}$ 不用于计算 $\tilde{a}_{ij}$ 。**

> [!NOTE]
>
> 此外，约束 $\text{diag}(\mathbf{W}) = 0$ 确保 $a_{ij}$ 不用于计算 $\tilde{a}_{ij}$ 。这个约束本质上就是padding。
>

**1) SLIM的 $\ell_1$ -范数和 $\ell_F$ -范数正则化**：为了学习一个稀疏的 $\mathbf{W}$ ，我们在式(3)中引入 $\mathbf{W}$ 的 $\ell_1$ -范数作为正则化项。众所周知， $\ell_1$ -范数正则化会在解中引入稀疏性[12]。

除了 $\ell_1$ -范数，我们还有 $\mathbf{W}$ 的 $\ell_F$ -范数作为另一个正则化项，这使优化问题成为一个**弹性网络问题**[13]。 $\ell_F$ -范**数衡量模型复杂度并防止过拟合（类似于岭回归）**。此外， $\ell_1$ -范数和 $\ell_F$ -范数正则化共同在解中隐式地分组相关item[13]。

> [!NOTE]
>
> **弹性网络（Elastic Net）** 是 **L1（Lasso）+ L2（Ridge）混合正则化**的线性回归方法：
>
> ```
> min  ½||y - Xw||²₂ + λ₁·||w||₁ + ½·λ₂·||w||²₂
> ```
>
> - **L1**（ $λ₁·||w||₁$ ）→ 产生稀疏解，自动做特征选择
> - **L2**（ $½·λ₂·||w||_2^2$ ）→ 稳定解，处理特征间相关性
>
> 单独 Lasso 在高相关特征组中会随机选一个；Elastic Net 会**一组全选或全不选**，更稳定。在这项目里：

**2) 计算 $\mathbf{W}$ **：由于 $\mathbf{W}$ 的**列是独立的**，式(3)中的优化问题可以解耦为一组如下形式的优化问题：
$$
\min_{\mathbf{w}_j} \quad \frac{1}{2} \|\mathbf{a}_j - \mathbf{A}\mathbf{w}_j\|_2^2 + \frac{\beta}{2} \|\mathbf{w}_j\|_2^2 + \lambda \|\mathbf{w}_j\|_1
$$

$$
\text{subject to} \quad \mathbf{w}_j \geq 0
$$

$$
w_{j,j} = 0, \qquad (4)
$$

这使得 $\mathbf{W}$ 的每一列可以独立求解。在式(4)中， $\mathbf{w}_j$ 是 $\mathbf{W}$ 的第 $j$ 列， $\mathbf{a}_j$ 是 $\mathbf{A}$ 的第 $j$ 列， $\|\cdot\|_2$ 是向量的 $\ell_2$ -范数， $\|\mathbf{w}_j\|_1 = \sum_{i=1}^{n} |w_{ij}|$ 是向量 $\mathbf{w}_j$ 的逐元素 $\ell_1$ -范数。**由于 $\mathbf{W}$ 的列独立性质，学习 $\mathbf{W}$ 可以轻松并行化**。式(4)的优化问题可以使用坐标下降和软阈值法求解[14]。

> [!NOTE]
>
> - $a_j$ — A 的第 j 列，即 item j 被哪些用户交互过（正样本）
> - $A·w_j$ — 用所有 item 的交互加权和，预测 item j
> - $β·||w_j||_2^2$ — L2 正则，防止过拟合
> - $λ·||w_j||₁$ — L1 正则，产生稀疏性
> - $w_{jj} = 0$ — 不能用自己预测自己
>
> 式(4)的优化问题可以使用坐标下降和软阈值法求解[14]。这里 省略了很多。

**3) 具有特征选择的SLIM**：式(4)中 $\mathbf{w}_j$ 的估计可以看作是一个**正则化回归问题的解**，其中 $\mathbf{A}$ 的第 $j$ 列是因变量， $\mathbf{A}$ 的其余 $n-1$ 列（自变量）用于估计。这种观点表明，特征选择方法可能潜在地用于在计算 $\mathbf{w}_j$ 之前减少自变量的数量。**这种特征选择方法的优势在于它们减少了 $\mathbf{A}$ 的列数，从而可以显著减少SLIM学习所需的总时间。**

受这些观察的启发，我们将SLIM方法扩展以纳入特征选择。我们将这些方法称为fsSLIM。尽管可以使用许多特征选择方法，但在本文中我们仅研究了一种方法，其灵感来自itemkNN Top-N推荐算法。具体来说，由于目标是学习一个线性模型来估计 $\mathbf{A}$ 的第 $j$ 列（即 $\mathbf{a}_j$ ），因此可以**选择与 $\mathbf{a}_j$ 最相似的 $\mathbf{A}$ 的列作为所选特征**。正如我们稍后的实验将展示的，使用余弦相似度 和 这种特征选择方法，可以得到一个 计算需求显著降低 且 质量损失最小 的方法。

### 4.3 基于SLIM的高效Top-N推荐

式(2)中的SLIM方法和 $\mathbf{W}$ 的稀疏性使得Top-N推荐速度显著加快。在式(2)中， $\mathbf{a}_i^T$ 总是非常稀疏（即用户通常只购买/评分了所有item中的一小部分），并且当 $\mathbf{W}$ 也很稀疏时，通过利用 $\mathbf{W}$ 的稀疏结构（即，沿 $\mathbf{W}$ 列对 $\mathbf{a}_i^T$ 中非零值对应行的非零值执行"收集"操作）， $\tilde{\mathbf{a}}_i^T$ 的计算可以非常快。因此，对用户 $u_i$ 进行推荐的计算复杂度为 $O(n_{a_i} \times n_w + N \log(N))$ ，其中 $n_{a_i}$ 是 $\mathbf{a}_i^T$ 中非零值的数量， $n_w$ 是 $\mathbf{W}$ 行中非零值的平均数量。 $N \log(N)$ 项用于对得分最高的N个item进行排序，这些item可以使用线性选择从 $\tilde{\mathbf{a}}_i^T$ 中潜在的 $n_{a_i} \times n_w$ 个非零条目中以线性时间选出。

### 4.4 SLIM与现有线性方法的比较

线性方法已被用于Top-N推荐。例如，[2]中的itemkNN方法具有与SLIM相似的线性模型。itemkNN的模型是一个 $k$ nn item-item余弦相似度矩阵 $\mathbf{S}$ ，即每一行 $\mathbf{s}_i^T$ 恰好有 $k$ 个非零值，表示item $t_j$ 与其 $k$ 个最相似邻居之间的余弦相似度。itemkNN和SLIM的线性模型之间的根本区别在于，**前者高度依赖于用于识别邻居的预先指定的item-item相似度度量**，而后者通过求解式(3)的优化问题来生成 $\mathbf{W}$ 。通过这种方式， $\mathbf{W}$ 可以潜在地编码item中丰富的细微关系，这些关系可能不易被传统的item-item相似度度量所捕捉。第6节的实验结果验证了这一点，表明 $\mathbf{W}$ 显著优于 $\mathbf{S}$ 。

Rendle等人[11]讨论了一种自适应k-最近邻方法，该方法使用了与[2]中itemkNN相同的模型，但自适应地学习item-item相似度矩阵。然而，[11]中的item-item相似度矩阵是完全稠密的、对称的**且具有负值**。 $\mathbf{W}$ 与Rendle等人的item-item相似度矩阵不同，除了其稀疏性导致快速推荐和低存储需求外，由于优化过程， $\mathbf{W}$ 不一定是对称的，因此允许更大的推荐灵活性。

Paterek[15]为每个item引入了一个用于评分预测的线性模型，其中用户 $u_i$ 在item $t_j$ 上的评分被计算为用户 $u_i$ 在所有其他item上的评分的聚合。他们通过为每个item求解一个 $\ell_2$ -范数正则化最小二乘问题来学习聚合系数（等价于 $\mathbf{W}$ ）。**学习到的系数是完全稠密的**。SLIM相对于Paterek方法的一个**优势在于，在学习过程中纳入了 $\ell_1$ -范数正则化，这迫使 $\mathbf{W}$ 变得稀疏**，因此最有信息的信号被捕获在 $\mathbf{W}$ 中，而噪声被丢弃。此外，与仅使用某一组购买/评分活动的Paterek方法相比，SLIM从所有购买/评分活动中学习 $\mathbf{W}$ ，从而更好地融合信息。

### 4.5 SLIM与MF方法的关系

用于Top-N推荐的MF方法具有模型：

$$
\tilde{\mathbf{A}} = \mathbf{U}\mathbf{V}^T, \qquad (5)
$$

其中 $\mathbf{U}$ 和 $\mathbf{V}^T$ 分别是用户因子和item因子。比较式(5)中的MF模型和式(2)中的SLIM方法，我们可以看到SLIM的模型可以被视为MF模型的一个特例（即 $\mathbf{A}$ 等价于 $\mathbf{U}$ ， $\mathbf{W}$ 等价于 $\mathbf{V}^T$ ）。

式(5)中的 $\mathbf{U}$ 和 $\mathbf{V}^T$ 位于潜在空间中，其维度通常作为一个参数指定。在式(2)中，"潜在"空间恰好变成了item空间，因此，在SLIM中无需在"潜在"空间中学习用户表示，从而简化了学习过程。另一方面， $\mathbf{U}$ 和 $\mathbf{V}^T$ 通常是低维的，因此在从 $\mathbf{U}$ 和 $\mathbf{V}^T$ 对 $\mathbf{A}$ 进行低秩逼近的过程中，**有用信息可能会丢失**。相反，在SLIM中，由于用户信息完全保存在 $\mathbf{A}$ 中，而对应的item信息通过优化学习得到，SLIM可能比MF方法提供更好的推荐。

此外，由于式(5)中的 $\mathbf{U}$ 和 $\mathbf{V}^T$ 通常都是稠密的， $\mathbf{a}_i^T$ 的计算需要从其对应的 $\mathbf{U}$ 和 $\mathbf{V}^T$ 中的稠密向量计算每个 $\tilde{a}_{ij}$ 。这导致MF方法进行推荐的计算复杂度很高，即每个用户为 $O(k^2 \times n)$ ，其中 $k$ 是latent factor数， $n$ 是item数。通过利用[16]、[17]、[18]中开发的稀疏矩阵分解算法，可以潜在地降低计算复杂度。然而，**由于计算成本高，这些稀疏矩阵分解算法均未被应用于解决Top-N推荐问题**。


## 5. 实验材料

### 5.1 数据集

我们在八个不同的真实数据集上评估了SLIM方法的性能，这些数据集的特征如表I所示。这些数据集可大致分为两类。

第一类（包含ccard、ctlg2、ctlg3和ecmrc[2]）源自客户购买交易。具体来说，ccard数据集对应一家大型百货商店的信用卡购买交易，其中每张卡至少有5笔交易。ctlg2和ctlg3数据集对应两家主要邮购目录零售商的目录购买交易。ecmrc数据集对应一个电子商务网站的基于网络的购买交易。这四个数据集只有二元购买信息。

第二类（包含BX、ML10M、Netflix和Yahoo）包含多值评分。如有需要，所有评分都被转换为二元指示。特别地，BX数据集是Book-Crossing数据集[^1]的一个子集，其中每个用户至少评分了20个item，每个item至少被5个用户且最多被300个用户评分。ML10M数据集对应电影评分，来自MovieLens[^2]研究item。Netflix数据集是从Netflix Prize数据集[^3]中提取的一个子集，其中每个用户评分了20-250部电影，每部电影被20-50个用户评分。最后，Yahoo数据集是从Yahoo! Music用户对歌曲的评分中提取的一个子集，作为Yahoo! Research Alliance Webscope计划[^4]的一部分提供。在Yahoo数据集上，每个用户评分了20-200首歌曲，每首音乐至少被10个用户且最多被5000个用户评分。

[^1]: http://www.informatik.uni-freiburg.de/~cziegler/BX/
[^2]: http://www.grouplens.org/node/12
[^3]: http://www.netflixprize.com/
[^4]: http://research.yahoo.com/Academic Relations

### 5.2 评估方法与度量

我们应用了**5次留一交叉验证**（LOOCV）来评估SLIM方法的性能。在每次运行中，每个数据集被分成训练集和测试集，方法是为每个用户随机选择一个非零条目放入测试集。训练集用于训练模型，然后针对每个用户，由模型生成一个大小为N的推荐item排序列表。评估通过比较每个用户的推荐列表和该用户在测试集中的item来进行。在第6节报告的大多数结果中，N等于10。但是，我们也会报告一些针对不同N值的有限结果。

推荐质量通过命中率（HR）和平均倒数命中排名（ARHR）[2]来衡量。HR定义如下：

$$
\text{HR} = \frac{\#\text{hits}}{\#\text{users}}, \qquad (6)
$$

其中 $\#\text{users}$ 是用户总数， $\#\text{hits}$ 是测试集中的item出现在大小为N的推荐列表中的用户数（即命中）。评估的第二个度量是ARHR，定义如下：

$$
\text{ARHR} = \frac{1}{\#\text{users}} \sum_{i=1}^{\#\text{hits}} \frac{1}{p_i}, \qquad (7)
$$

其中如果某个用户的item被命中， $p$ 是该item在排序推荐列表中的位置。ARHR是HR的加权版本，衡量item被推荐的强度，其中权重是命中位置在推荐列表中的倒数。

对于利用评分的实验，我们通过观察方法推荐具有特定评分值的item的能力来评估性能。为此，我们定义了每评分命中率（rHR）和累积命中率（cHR）。rHR计算为具有某个评分值的item上的命中率。cHR计算为具有不低于某个评分阈值的评分值的item上的命中率。

注意，在Top-N推荐文献中，存在其他评估度量。这些度量包括ROC曲线下面积（AUC），它衡量整个排序列表中真正例和假正例的相对位置。AUC的变体可以衡量排序列表顶部部分的位置。另一个流行的度量是**召回率**。然而，在Top-N推荐场景中，我们相信HR和ARHR是最直接、最有意义的度量，因为**用户只关心一个简短的推荐列表是否有他们感兴趣的item，而不是一个很长的推荐列表**。因此，我们在评估中使用HR和ARHR。

第6节中比较的所有算法都是用C语言实现的。所有实验都在一个具有6核Intel Xeon X7542 "Westmere"处理器、主频2.66 GHz的Linux集群上完成。


## 6. 实验结果

在本节中，我们展示SLIM方法的性能，并将其与其他流行的Top-N推荐方法进行比较。我们展示了两组实验的结果。在第一组实验中，所有Top-N推荐方法在学习时都使用二元用户-item购买信息，因此如有混淆，所有方法都附加-b以表示使用了二元数据（例如，SLIM-b）。在第二组实验中，所有Top-N推荐方法在学习时都使用用户-item评分信息，相应地，如有混淆，它们都附加-r。我们对所有算法的C实现进行了优化，以确保性能上的任何时间差异都源于算法本身，而非实现。对于所有方法，我们进行了详尽的网格搜索以确定最佳参数。在本节中，我们仅报告对应于能带来最佳结果的参数的性能。

### 6.1 二元数据上的SLIM性能

**1) 比较算法**：我们将SLIM的性能与其他三类Top-N推荐算法进行比较。第一类算法是基于item/用户的邻域协同过滤方法itemkNN、itemprob和userkNN。itemkNN和userkNN方法如第2节和第4-E节所述。itemprob方法与itemkNN类似，只是它使用修改后的item-item转移概率代替item-item余弦相似度。这些方法经过各种启发式方法的精心设计以获得更好的性能[^5]。

[^5]: http://glaros.dtc.umn.edu/gkhome/suggest/overview

第二类算法是MF方法，包括第2节讨论的PureSVD和WRMF。注意，PureSVD和WRMF在学习过程中都使用用户-item矩阵中的0值。PureSVD被证明在使用评分的Top-N推荐中优于其他MF方法[3]，包括那些将0视为缺失数据的MF方法。WRMF代表了使用二元信息进行Top-N推荐的最先进的矩阵分解方法。

第三类算法是依赖于排序/检索准则的方法，包括第2节和第4-E节讨论的BPRMF和BPRkNN。[11]中证明，在使用二元信息的Top-N推荐中，BPRMF在AUC度量方面优于其他方法。

**2) Top-N推荐性能**：表II显示了不同Top-N推荐算法的整体性能。这些结果表明，SLIM在所有数据集上（除了ML10M）产生的推荐在HR和ARHR方面一致优于其他方法（SLIM在ML10M上的HR为0.311，仅比BPRkNN的HR 0.327差）。就HR而言，在全部八个数据集上，SLIM平均比itemkNN好19.67%，比itemprob好12.91%，比userkNN好22.41%，比PureSVD好50.80%，比WRMF好13.42%，比BPRMF好14.32%，比BPRkNN好12.95%。在ARHR方面也可以观察到类似的性能增益。在三种基于MF的模型中，WRMF和BPRMF具有相似的性能，在除ML10M和Netflix之外的所有数据集上均显著优于PureSVD。BPRkNN在大数据集（即ML10M、Netflix和Yahoo）上的性能优于MF方法，但在小数据集上的性能不如MF方法。

在推荐效率方面，SLIM与itemkNN和itemprob相当（即所需时间在秒级），但比其他方法快得多（即所需时间在分钟级）。SLIM相比itemkNN效率稍差的原因是得到的最佳 $\mathbf{W}$ 矩阵比itemkNN的最佳item-item余弦相似度矩阵更稠密。PureSVD、WRMF和BPRMF具有更差的计算复杂度（即与item数和潜在空间维度的乘积成线性关系），这被它们较高的推荐运行时间所验证。BPRkNN产生了一个完全稠密的item-item相似度矩阵，这是其高推荐时间的原因。

在学习模型所需的时间方面，我们看到itemkNN/itemprob所需的时间远小于其他方法。SLIM学习其模型所需的时间相对于PureSVD、WRMF、BPRMF和BPRkNN，根据数据集的不同而有所变化。然而，尽管SLIM在某些数据集（如ML10M和Yahoo）上较慢，但这种情况可以通过基于特征选择的fsSLIM轻松补救，这将在第6-A3节中讨论。

表II中一个令人惊讶的结果是，基于MF的方法有时在HR方面甚至不如简单的itemkNN、itemprob和userkNN。例如，BPRMF在BX、ML10M、Netflix和Yahoo上的表现更差。这可能是因为在BPRMF中，作者评估了完整的AUC曲线来衡量感兴趣的item是否排在其余item之前。然而，良好的AUC值不一定导致在排序列表的Top-N上有良好表现。此外，对于PureSVD，最佳性能是在使用相当多的奇异值时实现的（例如，ccard、ctlg3、BX和Netflix）。

<img src="/Users/dazhang/PycharmProject/Papers/3-RecSys/.picture/image-20260728163815300.png" alt="image-20260728163815300" style="zoom:33%;" />

> 表1：评估中使用的数据集

对应#用户、#item和#交易的列分别显示每个数据集中的用户数、item数和交易数。对应rsize和csize的列分别显示每个数据集中每个用户的平均交易数和每个item上的平均交易数（即用户-item矩阵的行密度和列密度）。对应密度的列显示每个数据集的密度（即密度 = #交易/(#用户 $\times$ #item)）。对应评分的列显示每个数据集的评分范围，粒度为1。


<img src="/Users/dazhang/PycharmProject/Papers/3-RecSys/.picture/image-20260728163859485.png" alt="image-20260728163859485" style="zoom: 33%;" />

> **表II：Top-N推荐算法比较**

对应params的列显示相应方法的参数。对于itemkNN和userkNN方法，参数分别是邻居数。对于itemprob方法，参数是邻居数和转移参数 $\alpha$ 。对于PureSVD方法，参数是奇异值数量和SVD期间的迭代次数。对于WRMF方法，参数是潜在空间维度和对购买的权重。对于BPRMF方法，参数分别是潜在空间维度和学习率。对于BPRkNN方法，参数是学习率和正则化参数 $\lambda$ 。对于SLIM方法，参数是 $\ell_2$ -范数正则化参数 $\beta$ 和 $\ell_1$ -范数正则化参数 $\lambda$ 。对于fsSLIM方法，参数是邻居数和 $\ell_1$ -范数正则化参数 $\lambda$ 。对应HR和ARHR的列分别呈现命中率和平均倒数命中排名。对应mt和tt的列分别呈现模型学习和推荐所用的时间。mt/tt数字后的(s)、(m)和(h)分别是以秒、分钟和小时为单位的时间。粗体数字是每个数据集在HR方面的最佳性能。

**3) fsSLIM性能**：表II还展示了利用特征选择的SLIM版本（标记为fsSLIM的行）的结果。在第一组实验中，我们使用了item-item余弦相似度（与itemkNN中相同），对于 $\mathbf{A}$ 的每一列，选择其最相似的100个其他列，并使用 $m \times 100$ 矩阵来估计式(3)中的系数矩阵。结果显示在表II的第一个fsSLIM行中。在第二组实验中，我们基于item-item余弦相似度或item-item概率相似度（如itemprob中）选择 $\mathbf{A}$ 的最相似列（以性能最佳者为准）以及相应数量的列。这些实验的结果显示在第二个fsSLIM行中。

从fsSLIM结果中可以得出三个重要观察结果。首先，在几乎所有数据集上，fsSLIM的性能与SLIM相当。其次，fsSLIM学习模型所需的时间远少于SLIM。第三，使用fsSLIM来估计 $\mathbf{W}$ （其稀疏结构受itemkNN/itemprob邻居约束）比itemkNN/itemprob本身产生显著更好的推荐性能。这表明我们可以利用特征选择来减少学习时间而不降低性能。

**4) SLIM中的正则化效果**：图1显示了 $\ell_1$ -范数和 $\ell_2$ -范数正则化在**推荐时间**（直接取决于 $\mathbf{W}$ 的稀疏程度）和HR方面对BX数据集的影响（所有其他数据集都观察到类似结果）。图1表明，随着更大的 ** $\ell_1$ -范数正则化**（即式(3)中更大的 $\lambda$ ）被施加，**推荐时间降低**，表明学习到的 $\mathbf{W}$ 更稀疏。图1还显示了 $\ell_1$ -范数和 $\ell_2$ -范数正则化共同对推荐质量的影响。当两个正则化参数 $\beta$ 和 $\lambda$ 都非零时，达到了最佳推荐质量。此外，推荐质量随着正则化参数 $\beta$ 和 $\lambda$ 的变化而平滑变化。

<img src="/Users/dazhang/PycharmProject/Papers/3-RecSys/.picture/image-20260728163926640.png" alt="image-20260728163926640" style="zoom:33%;" />

> **图1：BX上 $\ell_1$ -范数和 $\ell_2$ -范数正则化效果**（a）推荐时间 （b）HR

> [!NOTE]
>
> 结论有点奇怪，似乎l2正则化没有正向影响？

**5) 用于长尾分布的SLIM**：长尾效应，即不成比例的大量购买/评分集中在少数item（热门item）上，一直是推荐系统关注的问题。热门item倾向于主导推荐，**使得难以产生新颖和多样化的推荐**。

由于在构建BX、Netflix和Yahoo数据集时，我们剔除了被许多用户购买/评分的item，这些数据集不受长尾效应的影响。表II中关于这些数据集的结果表明，**在没有显著热门item存在时，SLIM在产生非平凡的Top-N推荐方面优于其他方法**。

图2中的图表展示了ML10M数据集中item的长尾分布，**其中仅1%的item贡献了20%的评分**。我们剔除这1%最热门的item，并在学习时将剩余的评分用于所有Top-N方法。结果如表III所示。这些结果表明，所有方法的性能都明显差于表II中对应的性能（其中有"短头"即对应最热门item存在）。然而，SLIM优于其余方法。特别是，SLIM优于BPRkNN，尽管在ML10M中存在热门item时BPRkNN比SLIM做得更好（如表II所示）。这与基于BX、Netflix和Yahoo结果得出的观察一致，**即SLIM对长尾效应具有抵抗性。**

<img src="/Users/dazhang/PycharmProject/Papers/3-RecSys/.picture/image-20260728163952507.png" alt="image-20260728163952507" style="zoom: 33%;" />

> **图2：ML10M中购买/评分分布**（短头（热门）、长尾（非热门））

**表III：ML10M长尾上的性能**

<img src="/Users/dazhang/PycharmProject/Papers/3-RecSys/.picture/image-20260728164002056.png" alt="image-20260728164002056" style="zoom:33%;" />

从ML10M中剔除了前1%最热门的item。参数含义与表II中相同。

**6) 不同Top-N的推荐**：图3显示了BX、ML10M、Netflix和Yahoo数据集上不同N值（即5、10、15、20和25）下各方法的性能。表IV显示了SLIM与其余方法中最佳者在四个数据集上HR方面的性能差异。例如，表IV中BX在N=5时的0.012是SLIM的HR与BX上所有其他方法在推荐前5个item时的最佳HR之差。在BX、ML10M和Netflix数据集上，SLIM与其余方法中最佳者的性能差异在N较小时更大。图3和表IV表明，当推荐较少数量的item时，SLIM比其他方法产生更好的结果。这表明SLIM倾向于将最相关的item排在比其他方法更高的位置。

<img src="/Users/dazhang/PycharmProject/Papers/3-RecSys/.picture/image-20260728164046484.png" alt="image-20260728164046484" style="zoom:33%;" />

> **图3：不同N值下的推荐**（a）BX （b）ML10M （c）Netflix （d）Yahoo

**表IV：Top-N推荐上的性能差异**

| 数据集 | 5 | 10 | 15 | 20 | 25 |
|:---|:---:|:---:|:---:|:---:|:---:|
| BX | 0.012 | 0.013 | 0.000 | 0.000 | 0.001 |
| ML10M | 0.000 | -0.016 | -0.013 | -0.018 | -0.021 |
| Netflix | 0.009 | 0.012 | 0.008 | 0.005 | 0.003 |
| Yahoo | 0.006 | 0.015 | 0.015 | 0.016 | 0.017 |

对应N的列显示SLIM与其余方法中最佳者在相应Top-N推荐上的性能差异（以HR计）。

**7) $\mathbf{W}$ 的稀疏模式**：我们以ML10M为例说明SLIM学到的是什么。由itemkNN构建的item-item相似度矩阵 $\mathbf{S}$ 和来自SLIM的 $\mathbf{W}$ 如图4所示。注意，在图4中， $\mathbf{S}$ 矩阵是使用100个最近邻获得的。itemkNN和SLIM产生的矩阵密度分别为0.936%和0.935%，然而它们的稀疏模式不同。首先， $\mathbf{S}$ 矩阵有集中在对角线附近的item-item相似度非零值，而 $\mathbf{W}$ 的非零值分布更均匀。其次，在推荐过程中，平均有53.60个 $\mathbf{S}$ 中的非零值参与计算一个用户对一个item的推荐分数，而在 $\mathbf{W}$ 的情况下，平均有14.79个非零值做出贡献，是 $\mathbf{S}$ 中的1/3。 $\mathbf{W}$ 恢复了 $\mathbf{S}$ 中31.8%的非零条目（这些条目的值大于平均值），并且还发现了不在 $\mathbf{S}$ 中的新非零条目。新发现的项-项相似度贡献了 $\mathbf{W}$ 命中中的37.1%。这**表明， $\mathbf{W}$ 尽管也非常稀疏，但恢复了一些未被item-item余弦相似度度量捕捉到的微妙关系**，从而带来了性能提升。在SLIM中，与item $t_j$ 共同购买的item $t_k$ 也有助于item $t_j$ 与另一个item $t_i$ 之间的相似度，即使 $t_k$ 从未与 $t_i$ 共同购买过。此外，将缺失值视为0有助于泛化。在式(4)的 $\mathbf{w}_j$ 向量中包括所有缺失值为0有助于平滑item相似度，并帮助融入来自不相似/未共同购买item的影响。以上可通过坐标下降更新在理论上证明（此处省略证明）。

<img src="/Users/dazhang/PycharmProject/Papers/3-RecSys/.picture/image-20260728164100535.png" alt="image-20260728164100535" style="zoom:33%;" />

> **图4：ML10M的稀疏模式**（黑色为非零）（a）来自itemkNN的 $\mathbf{S}$ （b）来自SLIM的 $\mathbf{W}$

> [!NOTE]
>
> 结论很有趣

**8) 矩阵重构**：我们通过观察SLIM如何重构用户/item购买矩阵来将其与MF方法进行比较。我们以BPRMF作为MF方法的例子，因为它具有大多数最先进MF方法所具有的典型属性。我们关注ML10M，其**矩阵 $\mathbf{A}$ 的密度为1.3%**。来自SLIM的重构矩阵 $\tilde{\mathbf{A}}_{\text{SLIM}} = \mathbf{A}\mathbf{W}$ 的密度为25.1%，其非零值的均值为0.0593。对于 $\mathbf{A}$ 中1.3%的非零条目， $\tilde{\mathbf{A}}_{\text{SLIM}}$ 恢复了其中的99.1%，其均值为0.4489（即非零均值的7.57倍）。来自BPRMF的重构矩阵 $\tilde{\mathbf{A}}_{\text{BPRMF}} = \mathbf{U}\mathbf{V}^T$ 是完全稠密的，其中13.1%的值大于0，均值为1.8636，86.9%的值小于0，均值为-2.4718。对于 $\mathbf{A}$ 中1.3%的非零条目， $\tilde{\mathbf{A}}_{\text{BPRMF}}$ 有97.3%为正，均值为4.7623（即正均值的2.56倍）。**这表明SLIM比BPRMF更好地恢复了 $\mathbf{A}$ ，因为SLIM恢复了更多的非零条目且数值相对大得多。**

> [!NOTE]
>
> 这个结论很有趣


### 6.2 评分上的SLIM性能

**1) 比较算法**：我们将SLIM的性能与PureSVD、WRMF和BPRkNN进行比较。在SLIM中， $\mathbf{W}$ 矩阵是使用如式(2)中的用户-item评分矩阵 $\mathbf{A}$ 学习的。PureSVD也使用用户-item评分矩阵进行SVD计算。在WRMF中，评分被用作权重，遵循[5]中建议的方法。我们修改了BPRkNN，使得除了将已评分item排在未评分item之前外，它们还将高评分item排在低评分item之前。我们将在每个方法后面使用后缀-r来明确表示该方法在模型构建过程中利用了评分信息。类似地，我们将在此节中使用后缀-b来表示如第6-A节中利用二元信息的方法，以供比较。

**2) 评分上的Top-N推荐性能**：我们在具有评分信息的BX、ML10M、Netflix和Yahoo数据集上比较SLIM-r与PureSVD-r、WRMF-r和BPRkNN-r。此外，我们还在这四个数据集上评估了SLIM-b、PureSVD-b、WRMF-b和BPRkNN-b，这些模型的模型仍然是从二元用户-item购买矩阵学习的，但**推荐是基于评分进行评估的**。

![image-20260728164126762](/Users/dazhang/PycharmProject/Papers/3-RecSys/.picture/image-20260728164126762.png)

> **图5：评分上的Top-N推荐性能**（a）BX分布 （b）BX rHR （c）BX cHR （d）ML10M分布 （e）ML10M rHR （f）ML10M cHR （g）Netflix分布 （h）Netflix rHR （i）Netflix cHR （j）Yahoo分布 （k）Yahoo rHR （l）Yahoo cHR

图5展示了这些实验的结果。第一列图形显示四个数据集的评分分布。第二列图形显示四个数据集的每评分命中率（rHR）。最后，第三列图形显示四个数据集的累积命中率（cHR）。在图5中，二元Top-N模型对应于表II中各方法的最佳性能模型。使用评分的Top-N模型的结果分别是基于数据集在评分6、6、3和3上的cHR性能选择的。

图5的结果显示，所有-r方法倾向于在评分较高的item上产生更高的命中率。然而，-b方法的每评分命中率在不同评分间的变化较小。这是因为在学习过程中，-b方法中高评分item和低评分item没有区别。此外，-r方法在高评分item上的rHR优于-b方法。特别地，-r方法在所有数据集的评分高于平均水平的item上一致优于-b方法。

图5还显示，SLIM-r在所有四个数据集上，在较高评分的item上，无论是rHR还是cHR方面都一致优于其他方法。特别地，它在cHR方面优于PureSVD-r，而[3]中证明PureSVD-r是使用评分进行Top-N推荐中性能最好的方法之一。这表明在学习过程中纳入评分信息使SLIM方法能够识别更多高评分item。


## 7. 讨论与结论

### 7.1 已观察数据与缺失数据

在用户-item购买/评分矩阵 $\mathbf{A}$ 中，非零条目代表购买/评分活动。然而，值为"0"的条目可能是有歧义的。它们可能表示用户永远不会购买该item，用户可能会购买但尚未购买，或者我们不知道用户是否购买过该item或是否会购买。这是典型的"缺失数据"设定，已在推荐系统中得到充分研究[4]、[8]。

在SLIM中，我们将式(4)中 $\mathbf{a}_j$ 和 $\mathbf{A}$ 中的**所有缺失数据视为真正的负例**（即用户永远不会购买该item）。对式(4)中已观察数据和缺失数据的区分正在开发中。

### 7.2 结论

在本文中，我们提出了一种用于Top-N推荐的稀疏线性方法，能够快速生成高质量的Top-N推荐。SLIM采用了一个稀疏线性模型，其中**新item的推荐分数可以计算为其他item的聚合**。SLIM学习一个稀疏的聚合系数矩阵 $\mathbf{W}$ ，以使聚合非常快速。 $\mathbf{W}$ 通过求解一个 $\ell_1$ -范数和 $\ell_2$ -范数正则化优化问题来学习，从而在 $\mathbf{W}$ 中引入稀疏性。

我们进行了一组全面的实验，并将SLIM与其他最先进的Top-N推荐算法进行了比较。结果表明，SLIM实现了优于最先进的基于MF的方法的预测质量。此外，SLIM生成推荐非常快速。实验结果还证明了SLIM相对于其他方法的良好特性。这些特性包括：如果在学习之前应用特征选择，SLIM能够显著加速。SLIM对Top-N推荐问题中的长尾效应也具有抵抗性。此外，当使用评分进行训练时，SLIM倾向于产生也可能具有高评分的推荐。由于这些特性，SLIM非常适合于实时的Top-N推荐任务。


## 致谢

本工作部分得到了美国国家科学基金会（NSF）（IIS-0905220、OCI-1048018和IOS-0820730）以及明尼苏达大学数字技术中心的资助。研究和计算设施的访问由数字技术中心和明尼苏达超级计算研究所提供。


## 参考文献

[1] F. Ricci, L. Rokach, B. Shapira, and P. B. Kantor, Eds., Recommender Systems Handbook. Springer, 2011.

[2] M. Deshpande and G. Karypis, "Item-based top-n recommendation algorithms," ACM Transactions on Information Systems, vol. 22, pp. 143–177, January 2004.

[3] P. Cremonesi, Y. Koren, and R. Turrin, "Performance of recommender algorithms on top-n recommendation tasks," in Proceedings of the fourth ACM conference on Recommender systems, ser. RecSys '10. New York, NY, USA: ACM, 2010, pp. 39–46.

[4] R. Pan, Y. Zhou, B. Cao, N. N. Liu, R. Lukose, M. Scholz, and Q. Yang, "One-class collaborative filtering," in Proceedings of the 2008 Eighth IEEE International Conference on Data Mining. Washington, DC, USA: IEEE Computer Society, 2008, pp. 502–511.

[5] Y. Hu, Y. Koren, and C. Volinsky, "Collaborative filtering for implicit feedback datasets," in Proceedings of the 2008 Eighth IEEE International Conference on Data Mining. Washington, DC, USA: IEEE Computer Society, 2008, pp. 263–272.

[6] J. D. M. Rennie and N. Srebro, "Fast maximum margin matrix factorization for collaborative prediction," in Proceedings of the 22nd international conference on Machine learning, ser. ICML '05. New York, NY, USA: ACM, 2005, pp. 713–719.

[7] N. Srebro, J. D. M. Rennie, and T. S. Jaakkola, "Maximum-margin matrix factorization," in Advances in Neural Information Processing Systems 17. MIT Press, 2005, pp. 1329–1336.

[8] V. Sindhwani, S. S. Bucak, J. Hu, and A. Mojsilovic, "One-class matrix completion with low-density factorizations," in Proceedings of the 2010 IEEE International Conference on Data Mining, ser. ICDM '10. Washington, DC, USA: IEEE Computer Society, 2010, pp. 1055–1060.

[9] T. Hofmann, "Latent semantic models for collaborative filtering," ACM Trans. Inf. Syst., vol. 22, pp. 89–115, January 2004.

[10] Y. Koren, "Factorization meets the neighborhood: a multifaceted collaborative filtering model," in Proceeding of the 14th ACM SIGKDD international conference on Knowledge discovery and data mining, ser. KDD '08. New York, NY, USA: ACM, 2008, pp. 426–434.

[11] S. Rendle, C. Freudenthaler, Z. Gantner, and S.-T. Lars, "Bpr: Bayesian personalized ranking from implicit feedback," in Proceedings of the Twenty-Fifth Conference on Uncertainty in Artificial Intelligence, ser. UAI '09. Arlington, Virginia, United States: AUAI Press, 2009, pp. 452–461.

[12] R. Tibshirani, "Regression shrinkage and selection via the lasso," Journal of the Royal Statistical Society (Series B), vol. 58, pp. 267–288, 1996.

[13] H. Zou and T. Hastie, "Regularization and variable selection via the elastic net," Journal Of The Royal Statistical Society Series B, vol. 67, no. 2, pp. 301–320, 2005.

[14] J. H. Friedman, T. Hastie, and R. Tibshirani, "Regularization paths for generalized linear models via coordinate descent," Journal of Statistical Software, vol. 33, no. 1, pp. 1–22, 2 2010.

[15] A. Paterek, "Improving regularized singular value decomposition for collaborative filtering," Statistics, pp. 2–5, 2007.

[16] F. Bach, J. Mairal, and J. Ponce, "Convex sparse matrix factorizations," CoRR, vol. abs/0812.1869, 2008.

[17] J. Mairal, F. Bach, J. Ponce, and G. Sapiro, "Online learning for matrix factorization and sparse coding," J. Mach. Learn. Res., vol. 11, pp. 19–60, March 2010.

[18] P. O. Hoyer, "Non-negative matrix factorization with sparseness constraints," Journal of Machine Learning Research, vol. 5, pp. 1457–1469, December 2004.
