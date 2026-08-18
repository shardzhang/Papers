# GloVe: Global Vectors for Word Representation

> Jeffrey Pennington, Richard Socher, Christopher D. Manning | 斯坦福大学计算机科学系



本文提出 GloVe（Global Vectors，全局向量）模型，通过 **分析词向量中语义规律产生的模型属性**，构造了一个**全局 log-bilinear 回归模型**，融合了 **全局矩阵分解** 与 **局部上下文窗口** 两类方法的优势。核心发现是——**单词类比任务准确率达 75%，并在 词相似度 与 命名实体识别 任务上全面超越现有模型**。

核心内容：

- 分析得出：词向量线性规律（如 king − queen = man − woman）的正确起点是 **共现概率的比值** 而非 概率本身
- 推导过程从一般函数形式出发，施加向量差、点积、交换对称性等约束，逐步收敛到 **加权最小二乘目标**
- 模型仅训练词-词共现矩阵的非零元素，而非整个稀疏矩阵 或 单独上下文窗口，**高效利用语料统计信息**
- 证明 skip-gram 与 ivLBL 的在线目标 等价于 **带交叉熵加权的全局目标**，其损失函数与 GloVe 的 **加权最小二乘形式** 一致
- 分析模型复杂度：在幂律假设下非零元素数为 $O(|C|^{0.8})$，优于窗口方法的 $O(|C|)$

关键发现：

- 词类比任务准确率从 60% 级提升至 **75%（42B 语料，300 维）**，显著优于 skip-gram（69.1%）与 CBOW（65.7%）
- 在相似度任务上 42B 语料下全面领先，如 WS353 达 75.9、MC 达 83.6，超过 100B 语料训练的 CBOW
- NER 任务 F1 在 ACE 达 82.9、MUC7 达 82.2，为所有对比模型最高
- 训练速度与精度双优：相同语料/维度/训练时间下 GloVe 一致优于 word2vec
- 加权函数取 $\alpha = 3/4$ 比线性版本（$\alpha = 1$）有适度提升

---



## 摘要

近来学习词向量空间表示的方法成功利用向量算术捕获了细粒度的语义和句法规律，但这些规律的来源仍然不清楚。我们分析并明确了为使此类规律在词向量中出现**所需的模型属性**。结果是提出了一个新的 **全局 log-bilinear 回归模型**，结合了文献中两大主要模型家族的优势：**全局矩阵分解方法** 和 **局部上下文窗口方法**。我们的模型通过 仅训练 **词-词共现矩阵中的非零元素**（而不是整个稀疏矩阵或大规模语料中的单独上下文窗口）来 **高效利用统计信息**。该模型产生具有有意义 子结构的向量空间，其在最近的词类比任务上 75% 的性能即为证明。它还在相似度任务和命名实体识别上优于相关模型。



## 1 引言

语言的 语义向量空间模型 用 实值向量 表示每个词。这些向量可以作为各种应用中的特征，例如信息检索 [18]、文档分类 [26]、问答 [28]、命名实体识别 [30] 和句法分析 [27]。

大多数词向量方法依赖词向量对之间的 距离 或 角度 作为评估一组 词表示 内在质量 的主要方法。近来，Mikolov 等人（2013c）[21] 引入了 **基于词类比的评估方案**，该方案通过检查 词向量之间差异的各个维度（而非标量距离）来探测词向量空间的**更精细结构**。例如，类比"king 之于 queen 正如 man 之于 woman"应该通过向量方程 king − queen = man − woman 编码在向量空间中。这种评估方案有利于产生"意义维度"的模型，从而捕获分布式表示的多聚类思想 [3]。

学习词向量的两大主要模型家族是：1) **全局矩阵分解方法**，如潜在语义分析（LSA，Latent Semantic Analysis）[9]；2) **局部上下文窗口方法**，如 Mikolov 等人（2013c）[21] 的 skip-gram 模型。目前，两个家族都有明显的缺点。虽然像 LSA 这样的方法能高效利用统计信息，但它们在词类比任务上表现相对较差，表明其向量空间结构次优。像 skip-gram 这样的方法在类比任务上可能做得更好，但它们**没有充分利用语料库的统计信息**，因为它们**在单独的局部上下文窗口上训练，而不是在全局共现计数上训练**。

> [!NOTE]
>
> 这是全文关键

在这项工作中，我们分析了**产生意义线性方向所必需的模型属性**，并论证全局 log-bilinear 回归模型适合实现这一目标。我们提出了一个具体的加权最小二乘模型，在 全局词-词共现计数上训练，从而 高效利用统计信息。该模型产生具有有意义子结构的词向量空间，其在词类比数据集上 75% 准确率的最新性能即为证明。

我们还证明了我们的方法在几个词相似度任务上以及一个常见的命名实体识别（NER，Named Entity Recognition）基准上优于其他现有方法。

我们在 http://nlp.stanford.edu/projects/glove/ 提供模型的源代码以及训练好的词向量。



## 2 相关工作

**矩阵分解方法。** 生成低维词表示的矩阵分解方法的根源可以追溯到 LSA。这些方法利用**低秩近似**来分解 捕获 **语料库统计信息的大矩阵**。此类矩阵捕获的具体信息类型因应用而异。**在 LSA 中，矩阵是"词-文档"类型，即行对应词或术语，列对应语料库中的不同文档**。相比之下，例如语言超空间模拟（HAL，Hyperspace Analogue to Language）[16] **利用"词-词"类型的矩阵，即行和列都对应词，条目对应一个给定词在另一个给定词上下文中出现的次数。**

HAL 及相关方法的一个主要问题是，**最频繁的词 对 相似度度量贡献了不成比例的份额**：例如，两个词与 the 或 and 共现的次数会对它们的相似度产生很大影响，**尽管这对其语义相关性传递的信息相对较少**。存在许多解决 HAL 这一缺陷的技术，例如 COALS 方法 [24]，其中共现矩阵首先通过 基于熵 或 基于相关的归一化 进行变换。这类变换的一个优点是，**原始共现计数（对于规模合理的语料库可能跨越 8 或 9 个数量级）被压缩，以便在更小的区间内分布得更均匀**。各种较新的模型也采用这种方法，包括一项研究 [5] 表明 **正逐点互信息**（**PPMI**，Positive Pointwise Mutual Information）是一种好的变换。更近一些，Hellinger PCA（HPCA）[14] 形式的平方根类型变换被认为是学习词表示的有效方式。

**浅层窗口方法。** 另一种方法是学习有助于在局部上下文窗口内进行预测的词表示。例如，Bengio 等人（2003）[4] 提出了一个模型，在用于语言建模的简单神经网络架构中学习词向量表示。Collobert 和 Weston（2008）[7] **将词向量训练与下游训练目标解耦**，这为 Collobert 等人（2011）[8] 使用**词的完整上下文**（而不是语言模型那样的**仅前文上下文**）来学习词表示铺平了道路。

近来，完整的神经网络结构对学习有用词表示的重要性受到了质疑。Mikolov 等人（2013a）[19] 的 skip-gram 和连续词袋（CBOW，Continuous Bag-of-Words）模型提出了一个**基于两个词向量内积的简单单层架构**。Mnih 和 Kavukcuoglu（2013）[23] 也提出了密切相关的向量 log-bilinear 模型 vLBL 和 ivLBL，Levy 等人（2014）[15] 提出了**基于 PPMI 度量的显式词嵌入**。

在 skip-gram 和 ivLBL 模型中，目标是在给定词本身的情况下预测词的上下文，而 CBOW 和 vLBL 模型的目标是在给定上下文的情况下预测词。通过在词类比任务上的评估，这些模型展示了**将语言模式 学习为 词向量之间线性关系 的能力**。

与矩阵分解方法不同，浅层窗口方法有一个缺点：它们**不直接操作语料库的共现统计信息**。相反，这些模型**扫描整个语料库中的上下文窗口**，这**未能利用数据中大量的重复信息**。



## 3 GloVe 模型

语料库中 **词出现的统计信息** 是所有无监督学习词表示方法可用 的主要信息来源，尽管现在存在许多此类方法，但仍然存在一个问题：意义是如何从这些统计信息中产生的，以及得到的词向量可能如何表示这种意义。在本节中，我们对此问题有所阐明。我们利用这些见解构建了一个新的词表示模型，称之为 GloVe（Global Vectors，全局向量），因为 **全局语料库统计信息被模型直接捕获**。

首先我们建立一些记号。令 **词-词共现计数的矩阵** 记为 $X$，其条目 $X_{ij}$ 表示词 $j$ 在词 $i$ 上下文中出现的次数。令 $X_i = \sum_k X_{ik}$ 为任意词在词 $i$ 上下文中出现的次数。最后，令 $P_{ij} = P(j|i) = X_{ij}/X_i$ 为词 $j$ 出现在词 $i$ 上下文中的概率。

我们从一个简单的例子开始，展示意义的某些方面如何直接从共现概率中提取。考虑两个表现出某个特定关注方面的词 $i$ 和 $j$；为了具体起见，假设我们对热力学相的概念感兴趣，为此我们可以取 $i$ = ice（冰），$j$ = steam（蒸汽）。**这些词的关系可以通过研究它们与各种探针词 $k$ 的共现概率之比来考察**。对于与 ice 相关但与 steam 无关的词 $k$，例如 $k$ = solid（固体），我们期望比值 $P_{ik}/P_{jk}$ 很大。类似地，对于与 steam 相关但与 ice 无关的词 $k$，例如 $k$ = gas（气体），该比值应该很小。对于像 water（水）或 fashion（时尚）这样与 ice 和 steam 都相关或都无关的词 $k$，该比值应该接近 1。表 1 显示了一个大型语料库中这些概率及其比值，数字证实了这些预期。与原始概率相比，该比值能更好地区分相关词（solid 和 gas）与无关词（water 和 fashion），并且也能更好地区分两个相关词。

> 表 1：目标词 ice 和 steam 与来自 6 亿 token 语料库的选定上下文词的共现概率。只有在比值中，来自非判别性词（如 water 和 fashion）的噪声才会抵消，因此大值（远大于 1）与 ice 特有的属性很好地相关，小值（远小于 1）与 steam 特有的属性很好地相关。

<img src="/Users/dazhang/PycharmProject/Papers/1-NLP/.picture/image-20260817122229146.png" alt="image-20260817122229146" style="zoom:50%;" />

上述论证表明，词**向量学习的适当起点应该是共现概率的比值，而不是概率本身**。注意到比值 $P_{ik}/P_{jk}$ 依赖于三个词 $i$、$j$ 和 $k$，最一般的模型具有如下形式：
$$
F(w_i, w_j, \tilde{w}_k) = \frac{P_{ik}}{P_{jk}}, \qquad (1)
$$

其中 $w \in \mathbb{R}^d$ 是词向量，$\tilde{w} \in \mathbb{R}^d$ 是**单独的上下文词向量**，其作用将在第 4.2 节讨论。在这个等式中，右侧从语料库中提取，$F$ 可能依赖于一些尚未指定的参数。$F$ 的可能性数量巨大，但通过施加几个期望性质，我们可以选择一个唯一的选项。首先，我们希望 $F$ 在词向量空间中编码比值 $P_{ik}/P_{jk}$ 中存在的信息。由于向量空间本质上是一种线性结构，最自然的做法是使用**向量差**。基于此目标，我们可以将考虑范围限制在**只依赖于两个目标词之差的函数** $F$，将等式 (1) 修改为：

$$
F(w_i - w_j, \tilde{w}_k) = \frac{P_{ik}}{P_{jk}}. \qquad (2)
$$

接下来，我们注意到等式 (2) 中 $F$ 的参数是向量，而右侧是标量。虽然 $F$ 可以取为**由神经网络等参数化的复杂函数**，但这样做会模糊我们试图捕获的线性结构。为避免此问题，我们可以先取**参数的点积**：

$$
F\left((w_i - w_j)^{\mathrm{T}} \tilde{w}_k\right) = \frac{P_{ik}}{P_{jk}}, \qquad (3)
$$

这防止了 $F$ 以不理想的方式混合向量维度。接下来，注意到对于词-词共现矩阵，词和上下文词之间的区别是任意的，我们可以自由交换这两种角色。为了一致地做到这一点，我们不仅必须交换 $w \leftrightarrow \tilde{w}$，还要交换 $X \leftrightarrow X^{\mathrm{T}}$。我们最终的模型应该在这种重新标记下不变，但等式 (3) 不是。然而，这种对称性可以通过两步恢复。首先，我们要求 $F$ 是群 $(\mathbb{R}, +)$ 和 $(\mathbb{R}_{>0}, \times)$ 之间的同态，即：

$$
F\left((w_i - w_j)^{\mathrm{T}} \tilde{w}_k\right) = \frac{F(w_i^{\mathrm{T}} \tilde{w}_k)}{F(w_j^{\mathrm{T}} \tilde{w}_k)}, \qquad (4)
$$

根据等式 (3)，其解为：

$$
F(w_i^{\mathrm{T}} \tilde{w}_k) = P_{ik} = \frac{X_{ik}}{X_i}. \qquad (5)
$$

等式 (4) 的解是 $F = \exp$，即：

$$
w_i^{\mathrm{T}} \tilde{w}_k = \log(P_{ik}) = \log(X_{ik}) - \log(X_i). \qquad (6)
$$

接下来，我们注意到如果没有右侧的 $\log(X_i)$，等式 (6) 将表现出交换对称性。然而，这一项与 $k$ 无关，因此它可以被吸收为 $w_i$ 的偏置 $b_i$。最后，为 $\tilde{w}_k$ 添加一个额外的偏置 $\tilde{b}_k$ 恢复对称性：

$$
w_i^{\mathrm{T}} \tilde{w}_k + b_i + \tilde{b}_k = \log(X_{ik}). \qquad (7)
$$

等式 (7) 是对等式 (1) 的极大简化，但它**实际上是无定义的**，因为**当参数为零时对数发散**。解决此问题的一个方案是**在对数中引入加性偏移**，$\log(X_{ik}) \rightarrow \log(1 + X_{ik})$，这保持了 $X$ 的稀疏性，同时避免了发散。分解共现矩阵的对数的思想与 LSA 密切相关，我们将在实验中把由此产生的模型作为基线。该模型的一个主要缺点是它对所有共现一视同仁地加权，即使是那些很少发生或从未发生的共现。这种稀有共现是带噪声的，比频繁共现携带的信息少——然而，仅零条目就占 $X$ 中数据的 75–95%，具体取决于词汇量大小和语料库。

我们提出了一个新的加权最小二乘回归模型来解决这些问题。将等式 (7) 转化为最小二乘问题，并在代价函数中引入加权函数 $f(X_{ij})$，我们得到模型：

$$
J = \sum_{i,j=1}^{V} f(X_{ij}) \left( w_i^{\mathrm{T}} \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij} \right)^2, \qquad (8)
$$

其中 $V$ 是词汇量大小。加权函数应该满足以下性质：

1. $f(0) = 0$。如果 $f$ 被视为连续函数，它应该随着 $x \rightarrow 0$ 足够快地消失，使得 $\lim_{x \rightarrow 0} f(x) \log^2 x$ 有限。
2. $f(x)$ 应该是非递减的，这样**稀有共现不会被过度加权**。
3. 对于大的 $x$ 值，$f(x)$ 应该相对较小，这样**频繁共现不会被过度加权**。

当然，**有大量函数满足这些性质**，但我们发现表现良好的一类函数可以参数化为：

$$
f(x) = \begin{cases} (x/x_{max})^{\alpha} & \text{if } x < x_{max} \\ 1 & \text{otherwise} \end{cases}. \qquad (9)
$$

<img src="/Users/dazhang/PycharmProject/Papers/1-NLP/.picture/image-20260817123358873.png" alt="image-20260817123358873" style="zoom: 50%;" />

> 图1. 加权函数 $f$，$\alpha = 3/4$。

模型的性能对截止值依赖较弱，我们在所有实验中将其固定为 $x_{max} = 100$。我们发现 $\alpha = 3/4$ 比 $\alpha = 1$ 的线性版本有适度改进。虽然我们仅为选择 3/4 这个值提供了经验上的动机，但有趣的是，类似的**分数幂缩放**被发现能给出 [19] 中的最佳性能。

### 3.1 与其他模型的关系

由于所有无监督学习词向量的方法**最终都基于语料库的共现统计信息**，模型之间应该有共同之处。然而，某些模型在这方面仍然有些不清楚，特别是像 skip-gram 和 ivLBL 这样的近期窗口方法。因此，在本小节中我们展示这些模型如何与我们在等式 (8) 中定义的模型相关联。

skip-gram 或 ivLBL 方法的起点是词 $j$ 出现在词 $i$ 上下文中的概率模型 $Q_{ij}$。为了具体起见，让我们假设 $Q_{ij}$ 是 softmax：

$$
Q_{ij} = \frac{\exp(w_i^{\mathrm{T}} \tilde{w}_j)}{\sum_{k=1}^{V} \exp(w_i^{\mathrm{T}} \tilde{w}_k)}. \qquad (10)
$$

这些模型的大多数细节与我们的目的无关，除了它们**试图在上下文窗口扫描语料库时最大化对数概率**。训练以在线、随机的方式推进，但隐含的全局目标函数可以写成：

$$
J = -\sum_{i \in \text{corpus}} \sum_{j \in \text{context}(i)} \log Q_{ij}. \qquad (11)
$$

对该和中每一项计算 softmax 的归一化因子代价高昂。为了允许高效训练，skip-gram 和 ivLBL 模型对 $Q_{ij}$ 引入了近似。然而，如果我们先把那些具有相同 $i$ 和 $j$ 值的项分组，等式 (11) 中的和可以更高效地求值：

$$
J = -\sum_{i=1}^{V} \sum_{j=1}^{V} X_{ij} \log Q_{ij}, \qquad (12)
$$

其中我们**使用了同类项的数量由共现矩阵 $X$ 给出的事实**。回顾我们关于 $X_i = \sum_k X_{ik}$ 和 $P_{ij} = X_{ij}/X_i$ 的记号，我们可以将 $J$ 重写为：

$$
J = -\sum_{i=1}^{V} X_i \sum_{j=1}^{V} P_{ij} \log Q_{ij} = \sum_{i=1}^{V} X_i H(P_i, Q_i), \qquad (13)
$$

其中 $H(P_i, Q_i)$ 是分布 $P_i$ 和 $Q_i$ 的交叉熵，我们类比 $X_i$ 来定义。作为交叉熵误差的加权和，这个目标与等式 (8) 的加权最小二乘目标有某种形式上的相似性。事实上，可以直接优化等式 (13)，而不是使用 skip-gram 和 ivLBL 模型中使用的在线训练方法。人们可以将此目标解释为"全局 skip-gram"模型，进一步研究可能会很有趣。另一方面，等式 (13) 表现出许多不希望的性质，在将其作为学习词向量的模型采用之前应当解决。

> [!NOTE]
>
> 看不懂了

首先，交叉熵误差只是概率分布之间众多可能的距离度量之一，它有一个不幸的性质：长尾分布往往被建模得很差，对不太可能的事件给予过多权重。此外，为了使度量有界，它要求模型分布 $Q$ 被正确归一化。由于等式 (10) 中对整个词汇量的求和，这构成了计算瓶颈，因此最好考虑一个不需要 $Q$ 具备此性质的不同的距离度量。一个自然的选择是最小二乘目标，其中 $Q$ 和 $P$ 中的归一化因子被丢弃：

$$
\hat{J} = \sum_{i,j} X_i \left( \hat{P}_{ij} - \hat{Q}_{ij} \right)^2 \qquad (14)
$$

其中 $\hat{P}_{ij} = X_{ij}$ 和 $\hat{Q}_{ij} = \exp(w_i^{\mathrm{T}} \tilde{w}_j)$ 是未归一化的分布。在这个阶段出现了另一个问题，即 $X_{ij}$ 经常取非常大的值，这会使优化复杂化。一个有效的补救措施是最小化 $\hat{P}$ 和 $\hat{Q}$ 的对数的平方误差：

$$
\hat{J} = \sum_{i,j} X_i \left( \log \hat{P}_{ij} - \log \hat{Q}_{ij} \right)^2 = \sum_{i,j} X_i \left( w_i^{\mathrm{T}} \tilde{w}_j - \log X_{ij} \right)^2. \qquad (15)
$$

最后，我们观察到，虽然加权因子 $X_i$ 是由 skip-gram 和 ivLBL 模型固有的在线训练方法预先决定的，但它绝不能被保证是最优的。事实上，Mikolov 等人（2013a）[19] 观察到，可以通过过滤数据来降低频繁词的加权因子的有效值，从而提高性能。考虑到这一点，我们引入一个更一般的加权函数，我们可以自由地让它也依赖于上下文词。结果是：

$$
\hat{J} = \sum_{i,j} f(X_{ij}) \left( w_i^{\mathrm{T}} \tilde{w}_j - \log X_{ij} \right)^2, \qquad (16)
$$

这与我们之前推导的等式 (8) 的代价函数等价[^1]。

[^1]: 我们也可以在等式 (16) 中包含偏置项。

### 3.2 模型复杂度

从等式 (8) 和加权函数 $f(X)$ 的显式形式可以看出，模型的计算复杂度取决于矩阵 $X$ 中非零元素的数量。由于这个数量总是小于矩阵条目的总数，模型的规模不会比 $O(|V|^2)$ 更差。乍一看，这似乎是对浅层窗口方法（其规模随语料库大小 $|C|$ 变化）的实质性改进。然而，典型的词汇量有数十万个词，因此 $|V|^2$ 可能达到数千亿，这实际上比大多数语料库大得多。出于这个原因，确定是否可以对 $X$ 的非零元素数量放置一个**更紧的界**是很重要的。

为了对 $X$ 中非零元素的数量做出任何具体陈述，有必要对词共现的分布做出一些假设。特别是，我们将假设词 $i$ 与词 $j$ 的共现次数 $X_{ij}$ 可以建模为**该词对频率排名 $r_{ij}$ 的幂律函数**：

$$
X_{ij} = \frac{k}{(r_{ij})^{\alpha}}. \qquad (17)
$$

语料库中的总词数与共现矩阵 $X$ 所有元素之和成正比：

$$
|C| \sim \sum_{ij} X_{ij} = \sum_{r=1}^{|X|} \frac{k}{r^{\alpha}} = kH_{|X|,\alpha}, \qquad (18)
$$

其中我们将最后一个和用广义调和数 $H_{n,m}$ 表示。求和的上限 $|X|$ 是最大频率排名，它与矩阵 $X$ 中非零元素的数量重合。这个数量也等于等式 (17) 中使得 $X_{ij} \geq 1$ 的 $r$ 的最大值，即 $|X| = k^{1/\alpha}$。因此我们可以将等式 (18) 写为：

$$
|C| \sim |X|^{\alpha} H_{|X|,\alpha}. \qquad (19)
$$

我们感兴趣的是当两个数都很大时 $|X|$ 与 $|C|$ 的关系；因此我们可以自由地展开等式右侧的级数。为此我们使用广义调和数的展开式 [1]：

$$
H_{x,s} = \frac{x^{1-s}}{1 - s} + \zeta(s) + O(x^{-s}) \quad \text{if } s > 0, s \neq 1, \qquad (20)
$$

得到：

$$
|C| \sim \frac{|X|}{1 - \alpha} + \zeta(\alpha) |X|^{\alpha} + O(1), \qquad (21)
$$

其中 $\zeta(s)$ 是黎曼 zeta 函数。在 $X$ 很大的极限下，等式 (21) 右侧的两项中只有一项是相关的，而哪一项相关取决于 $\alpha$ 是否大于 1：

$$
|X| = \begin{cases} O(|C|) & \text{if } \alpha < 1, \\ O(|C|^{1/\alpha}) & \text{if } \alpha > 1. \end{cases} \qquad (22)
$$

对于本文研究的语料库，我们观察到 $X_{ij}$ 可以很好地由等式 (17) 建模，其中 $\alpha = 1.25$。在这种情况下我们有 $|X| = O(|C|^{0.8})$。因此我们得出结论，模型的复杂度远好于最坏情况 $O(V^2)$，实际上比规模为 $O(|C|)$ 的在线窗口方法做得更好。



## 4 实验

### 4.1 评估方法

我们在 Mikolov 等人（2013a）[19] 的词类比任务、如 [17] 中所述的各种词相似度任务，以及 NER 的 CoNLL-2003 共享基准数据集 [29] 上进行实验。

>  表 2：词类比任务的结果，以准确率百分比给出。下划线分数是同规模模型中最好的；粗体分数是总体最好的。HPCA 向量公开可用[^2]；(i)vLBL 结果来自 [23]；skip-gram（SG）和 CBOW 结果来自 [19, 20]；我们使用 word2vec 工具训练了 SG† 和 CBOW†[^3]。详见正文以及 SVD 模型的描述。

<img src="/Users/dazhang/PycharmProject/Papers/1-NLP/.picture/image-20260817123431644.png" alt="image-20260817123431644" style="zoom:33%;" />

[^2]: http://lebret.ch/words/
[^3]: http://code.google.com/p/word2vec/

**词类比。** 词类比任务包含如下问题："a 之于 b 正如 c 之于 ?"。数据集包含 19,544 个此类问题，分为语义子集和句法子集。语义问题通常是人或地方的类比，如"Athens 之于 Greece 正如 Berlin 之于 ?"。句法问题通常是关于动词时态或形容词形式的类比，例如"dance 之于 dancing 正如 fly 之于 ?"。为了正确回答问题，模型应该唯一识别缺失项，只有完全对应才被计为正确匹配。我们通过找到表示 $w_d$ 最接近 $w_b - w_a + w_c$（根据余弦相似度）的词 $d$ 来回答问题"a 之于 b 正如 c 之于 ?"。[^4]

[^4]: Levy 等人（2014）[15] 引入了一种乘法类比评估方法 3COSMUL，并在类比任务上报告了 68.24% 的准确率。这个数字是在数据集的子集上评估的，因此没有包括在表 2 中。在我们几乎所有的实验中，3COSMUL 的表现都不如余弦相似度。

**词相似度。** 虽然类比任务是我们主要的关注点，因为它测试了有趣的向量空间子结构，但我们还在表 3 中评估了我们的模型在多种词相似度任务上的表现。这些任务包括 WordSim-353 [12]、MC [22]、RG [25]、SCWS [13] 和 RW [17]。

**命名实体识别。** NER 的 CoNLL-2003 英语基准数据集是来自路透社新闻专线文章的文档集合，标注了四种实体类型：人物、地点、组织和杂项。我们在 CoNLL-03 训练数据上训练模型，并在三个数据集上测试：1) CoNLL-03 测试数据，2) ACE Phase 2（2001-02）和 ACE-2003 数据，以及 3) MUC7 Formal Run 测试集。我们采用 BIO2 标注标准，以及 [31] 中描述的所有预处理步骤。我们使用随 Stanford NER 模型标准发行版附带的一组全面离散特征 [11]。为 CoNLL-2003 训练数据集生成了总共 437,905 个离散特征。此外，为五词上下文的每个词添加了 50 维向量，用作连续特征。以这些特征作为输入，我们训练了一个条件随机场（CRF，Conditional Random Field），其设置与 [31] 的 CRFjoin 模型完全相同。

### 4.2 语料库与训练细节

我们在五个不同规模的语料库上训练模型：一个包含 10 亿 token 的 2010 年 Wikipedia 转储；一个包含 16 亿 token 的 2014 年 Wikipedia 转储；包含 43 亿 token 的 Gigaword 5；Gigaword5 + Wikipedia2014 的组合，包含 60 亿 token；以及来自 Common Crawl 的 420 亿 token 网络数据[^5]。

我们使用 Stanford 分词器对每个语料库进行分词和小写化，构建 400,000 个最频繁词的词汇表[^6]，然后构造共现计数矩阵 $X$。在构造 $X$ 时，我们必须选择上下文窗口应该多大，以及**是否区分左上下文和右上下文**。我们在下面探讨这些选择的影响。在所有情况下，我们都使用递减的加权函数，这样相距 $d$ 个词的词对总共贡献 $1/d$。这是解释以下事实的一种方式：非常远的词对预期包含关于词彼此关系的相关性较低的信息。

对于所有实验，我们设置 $x_{max} = 100$，$\alpha = 3/4$，并使用 AdaGrad [10] 训练模型，从 $X$ 中随机采样非零元素，初始学习率为 0.05。我们对小于 300 维的向量运行 50 次迭代，否则运行 100 次迭代（有关收敛速度的更多细节见第 4.6 节）。除非另有说明，我们使用目标词左侧十个词和右侧十个词的上下文。

该模型生成两组词向量 $W$ 和 $\tilde{W}$。当 $X$ 对称时，$W$ 和 $\tilde{W}$ 是等价的，只是它们的随机初始化不同；这两组向量应该表现相当。另一方面，有证据表明，对于某些类型的神经网络，训练网络的多个实例然后组合结果有助于减少过拟合和噪声，并通常改善结果 [6]。考虑到这一点，我们选择使用 $W + \tilde{W}$ 的和作为我们的词向量。这样做通常会在性能上带来小幅提升，语义类比任务提升最大。

我们与各种最先进模型的已发表结果进行比较，也与我们自己使用 word2vec 工具产生的结果以及几个使用 SVD 的基线进行比较。使用 word2vec，我们在 60 亿 token 语料库（Wikipedia 2014 + Gigaword 5）上训练 skip-gram（SG†）和连续词袋（CBOW†）模型，词汇表为前 400,000 个最频繁词，上下文窗口大小为 10。我们使用 10 个负样本，在第 4.6 节中我们表明这对该语料库是一个好的选择。对于 SVD 基线，我们生成一个截断矩阵 $X_{trunc}$，保留每个词仅与最频繁的 10,000 个词共现的频率信息。这一步在许多基于矩阵分解的方法中很典型，因为额外的列可能贡献不成比例的零条目数量，而且这些方法否则计算代价高昂。这个矩阵的奇异向量构成基线"SVD"。我们还评估了两个相关基线："SVD-S"，其中我们对 $\sqrt{X_{trunc}}$ 取 SVD；以及"SVD-L"，其中我们对 $\log(1 + X_{trunc})$ 取 SVD。两种方法都有助于压缩 $X$ 中原本很大的值范围。[^7]

[^5]: 为了展示模型的可扩展性，我们还在一个更大的第六个语料库上训练了它，该语料库包含 840 亿 token 的网络数据，但在这种情况下我们没有对词汇表进行小写化，因此结果不能直接比较。
[^6]: 对于在 Common Crawl 数据上训练的模型，我们使用约 200 万词的更大词汇表。
[^7]: 我们还研究了几种变换 $X$ 的其他加权方案；这里报告的表现最好。许多像 PPMI 这样的加权方案破坏了 $X$ 的稀疏性，因此无法在实际中用于大词汇量。对于较小的词汇量，这些信息论变换确实在词相似度度量上表现良好，但它们在词类比任务上表现很差。

### 4.3 结果

我们在表 2 中展示了词类比任务的结果。GloVe 模型的性能显著优于其他基线，通常使用更小的向量尺寸和更小的语料库。我们使用 word2vec 工具的结果比大多数先前发表的结果好一些。这归因于许多因素，包括我们选择使用负采样（通常比层次 softmax 效果更好）、负样本的数量以及语料库的选择。

我们证明了该模型可以很容易地在 420 亿 token 的大型语料库上训练，并带来相应的实质性性能提升。我们注意到，增加语料库大小并不能保证其他模型的结果得到改善，正如 SVD-L 模型在这个更大语料库上的性能下降所示。这个基本 SVD 模型不能很好地扩展到大型语料库这一事实，进一步证明了我们模型中提出的加权方案的必要性。

表 3 展示了五个不同词相似度数据集上的结果。相似度得分从词向量中获得，首先对词汇表上的每个特征进行归一化，然后计算余弦相似度。我们计算该得分与人类判断之间的 Spearman 秩相关系数。CBOW* 表示 word2vec 网站上可用的向量，它们在 100B 词的新闻数据上使用词和短语向量训练。GloVe 在使用不到一半大小的语料库时超越了它。

**表 3：词相似度任务上的 Spearman 秩相关。所有向量均为 300 维。CBOW* 向量来自 word2vec 网站，不同之处在于它们包含短语向量。**

<img src="/Users/dazhang/PycharmProject/Papers/1-NLP/.picture/image-20260817123654246.png" alt="image-20260817123654246" style="zoom:50%;" />

表 4 展示了基于 CRF 的模型在 NER 任务上的结果。当在开发集上 25 次迭代没有改进时，L-BFGS 训练终止。除此之外，所有配置都与 Wang 和 Manning（2013）[31] 使用的相同。标记为 Discrete 的模型是使用标准 Stanford NER 模型发行版附带的全面离散特征集合的基线，但没有词向量特征。除了之前讨论的 HPCA 和 SVD 模型，我们还与 Huang 等人（2012）[13]（HSMN）和 Collobert 和 Weston（2008）[7]（CW）的模型进行比较。我们使用 word2vec 工具训练了 CBOW 模型[^8]。

GloVe 模型在所有评估指标上都优于所有其他方法，除了 CoNLL 测试集，在该测试集上 HPCA 方法做得稍好。我们得出结论，GloVe 向量在下游 NLP 任务中有用，正如神经网络向量首次被证明的那样 [30]。

**表 4：使用 50d 向量的 NER 任务 F1 分数。Discrete 是没有词向量的基线。我们使用公开可用的 HPCA、HSMN 和 CW 向量。详见正文。**

<img src="/Users/dazhang/PycharmProject/Papers/1-NLP/.picture/image-20260817123632007.png" alt="image-20260817123632007" style="zoom:50%;" />

[^8]: 我们使用与上面相同的参数，只是在这种情况下我们发现 5 个负样本比 10 个略好。

### 4.4 模型分析：向量长度 与 上下文大小

在图 2 中，我们展示了改变向量长度和上下文窗口的实验结果。**向目标词左右两边延伸的上下文窗口称为 对称窗口**，仅向左延伸的称为**非对称窗口**。在 (a) 中，我们观察到大于约 200 维的向量收益递减。在 (b) 和 (c) 中，我们考察了对称和非对称上下文窗口下改变窗口大小的影响。在小且非对称的上下文窗口下，**句法子任务** 的性能更好，这与句法信息主要来自 **紧邻上下文且可能强烈依赖词序** 的直觉一致。另一方面，**语义信息**更常是非局部的，更大的窗口大小能捕获更多语义信息。

<img src="/Users/dazhang/PycharmProject/Papers/1-NLP/.picture/image-20260817123510720.png" alt="image-20260817123510720" style="zoom:50%;" />

> 图2. 类比任务准确率作为向量大小和窗口大小/类型的函数。所有模型都在 60 亿 token 语料库上训练。在 (a) 中，窗口大小为 10。在 (b) 和 (c) 中，向量大小为 100。

### 4.5 模型分析：语料库大小

在图 3 中，我们展示了在不同语料库上训练的 300 维向量在词类比任务上的表现。在**句法子任务**上，随着语料库大小的增加，性能单调提高。这是意料之中的，因为**更大的语料库通常产生更好的统计信息**。有趣的是，同样的趋势在**语义子任务**上并不成立，在较小的 Wikipedia 语料库上训练的模型比在较大的 Gigaword 语料库上训练的模型表现更好。这很可能是由于类比数据集中存在大量基于城市和国家的类比，以及 Wikipedia 对大多数此类地点都有相当全面的文章。**此外，Wikipedia 的条目会不断更新以吸收新知识，而 Gigaword 是一个固定的新闻存储库，信息过时且可能不正确。**

<img src="/Users/dazhang/PycharmProject/Papers/1-NLP/.picture/image-20260817123540233.png" alt="image-20260817123540233" style="zoom: 50%;" />

> 图3. 在不同语料库上训练的 300 维向量在类比任务上的准确率。

### 4.6 模型分析：运行时间

总运行时间分为填充 $X$ 和训练模型两部分。前者取决于许多因素，包括窗口大小、词汇量大小和语料库大小。虽然我们没有这样做，但这一步很容易在多台机器上并行化（例如，参见 [14] 中的一些基准）。使用双 2.1GHz Intel Xeon E5-2658 机器的单线程，用 10 词对称上下文窗口、400,000 词词汇表和 60 亿 token 语料库填充 $X$ 大约需要 85 分钟。给定 $X$，训练模型所需的时间取决于向量大小和迭代次数。对于 300 维向量，在上述设置下（并使用上述机器的全部 32 个核心），单次迭代需要 14 分钟。学习曲线见图 4。

### 4.7 模型分析：与 word2vec 的比较

GloVe 与 word2vec 的严格定量比较因存在许多对性能有强烈影响的参数而变得复杂。我们通过将向量长度、上下文窗口大小、语料库和词汇量大小设置为前一子节提到的配置，来控制我们在第 4.4 和 4.5 节中识别的主要变化来源。

剩下要控制的最重要变量是训练时间。对于 GloVe，相关参数是训练迭代次数。对于 word2vec，显然的选择是训练轮数。不幸的是，该代码目前仅设计为单轮训练：它指定了针对单次数据遍历特定的学习计划，使其很难修改为多次遍历。另一个选择是改变负样本的数量。增加负样本实际上增加了模型看到的训练词数量，因此在某些方面类似于额外的轮数。

我们将任何未指定的参数设置为其默认值，假设它们接近最优，尽管我们承认在更彻底的分析中应该放宽这种简化。

在图 4 中，我们绘制了类比任务上的总体性能作为训练时间的函数。底部两个 x 轴表示 GloVe 对应的训练迭代次数和 word2vec 的负样本数量。我们注意到，**如果负样本数量增加到超过约 10 个，word2vec 的性能实际上会下降**。这大概是因为负采样方法不能很好地逼近目标概率分布。[^9]

对于相同的语料库、词汇量、窗口大小和训练时间，GloVe 始终优于 word2vec。它更快地获得更好的结果，并且无论速度如何都能获得最好的结果。

<img src="/Users/dazhang/PycharmProject/Papers/1-NLP/.picture/image-20260817123600058.png" alt="image-20260817123600058" style="zoom:50%;" />

> 图4. 词类比任务的总体准确率作为训练时间的函数，训练时间由 GloVe 的迭代次数和 CBOW (a) 及 skip-gram (b) 的负样本数量决定。在所有情况下，我们在相同的 60 亿 token 语料库（Wikipedia 2014 + Gigaword 5）上训练 300 维向量，使用相同的 400,000 词词汇表，并使用大小为 10 的对称上下文窗口。

[^9]: 相比之下，噪声对比估计是一种近似方法，它会随着更多负样本而改善。在 [23] 的表 1 中，类比任务的准确率是负样本数量的**非递减函数**。



## 5 结论

近来，相当多的注意力集中在以下问题上：分布式词表示最好从 **基于计数的方法** 还是 **基于预测的方法** 中学习。目前，基于预测的模型获得了大量支持；例如，Baroni 等人（2014）[2] 认为这些模型在一系列任务上表现更好。在这项工作中，**我们认为这两类方法在根本层面上并没有显著不同，因为它们都探测 语料库底层的共现统计信息，但基于计数的方法捕获 全局统计信息 的效率可以是有利的**。我们构建了一个**利用计数数据**这一主要优势的模型，同时捕获了 word2vec 等近期 log-bilinear 基于预测方法中普遍存在的有意义的**线性子结构**。结果 GloVe 是一个用于无监督学习词表示的**新全局 log-bilinear 回归模型**，它在 词类比、词相似度 和 命名实体识别任务上优于其他模型。



## 致谢

我们感谢匿名审稿人的宝贵意见。斯坦福大学谨感谢国防威胁降低局（DTRA，Defense Threat Reduction Agency）在空军研究实验室（AFRL，Air Force Research Laboratory）合同号 FA8650-10-C-7020 下的支持，以及国防高级研究计划局（DARPA，Defense Advanced Research Projects Agency）文本深度探索与过滤（DEFT，Deep Exploration and Filtering of Text）计划在 AFRL 合同号 FA8750-13-2-0040 下的支持。本文表达的任何意见、发现、结论或建议均为作者本人观点，不一定反映 DTRA、AFRL、DEFT 或美国政府的看法。



## 参考文献

[1] Tom M. Apostol. 1976. Introduction to Analytic Number Theory. Introduction to Analytic Number Theory.

[2] Marco Baroni, Georgiana Dinu, and German Kruszewski. 2014. **Don't count, predict! A systematic comparison of context-counting vs. context-predicting semantic vectors**. In ACL.

[3] Yoshua Bengio. 2009. Learning deep architectures for AI. Foundations and Trends in Machine Learning.

[4] Yoshua Bengio, Rejean Ducharme, Pascal Vincent, and Christian Janvin. 2003. A neural probabilistic language model. JMLR, 3:1137–1155.

[5] John A. Bullinaria and Joseph P. Levy. 2007. **Extracting semantic representations from word co-occurrence statistics: A computational study**. Behavior Research Methods, 39(3):510–526.

[6] Dan C. Ciresan, Alessandro Giusti, Luca M. Gambardella, and Jurgen Schmidhuber. 2012. Deep neural networks segment neuronal membranes in electron microscopy images. In NIPS, pages 2852–2860.

[7] Ronan Collobert and Jason Weston. 2008. **A unified architecture for natural language processing: deep neural networks with multitask learning**. In Proceedings of ICML, pages 160–167.

[8] Ronan Collobert, Jason Weston, Leon Bottou, Michael Karlen, Koray Kavukcuoglu, and Pavel Kuksa. 2011. **Natural Language Processing (Almost) from Scratch**. JMLR, 12:2493–2537.

[9] Scott Deerwester, Susan T. Dumais, George W. Furnas, Thomas K. Landauer, and Richard Harshman. 1990. Indexing by latent semantic analysis. Journal of the American Society for Information Science, 41.

[10] John Duchi, Elad Hazan, and Yoram Singer. 2011. Adaptive subgradient methods for online learning and stochastic optimization. JMLR, 12.

[11] Jenny Rose Finkel, Trond Grenager, and Christopher Manning. 2005. Incorporating non-local information into information extraction systems by Gibbs sampling. In Proceedings of ACL, pages 363–370.

[12] Lev Finkelstein, Evgenly Gabrilovich, Yossi Matias, Ehud Rivlin, Zach Solan, Gadi Wolfman, and Eytan Ruppin. 2001. Placing search in context: The concept revisited. In Proceedings of the 10th international conference on World Wide Web, pages 406–414. ACM.

[13] Eric H. Huang, Richard Socher, Christopher D. Manning, and Andrew Y. Ng. 2012. **Improving Word Representations via Global Context and Multiple Word Prototypes**. In ACL.

[14] Remi Lebret and Ronan Collobert. 2014. Word embeddings through Hellinger PCA. In EACL.

[15] Omer Levy, Yoav Goldberg, and Israel Ramat-Gan. 2014. **Linguistic regularities in sparse and explicit word representations.** CoNLL-2014.

[16] Kevin Lund and Curt Burgess. 1996. Producing high-dimensional semantic spaces from lexical co-occurrence. Behavior Research Methods, Instrumentation, and Computers, 28:203–208.

[17] Minh-Thang Luong, Richard Socher, and Christopher D Manning. 2013. Better word representations with recursive neural networks for morphology. CoNLL-2013.

[18] Christopher D. Manning, Prabhakar Raghavan, and Hinrich Schutze. 2008. Introduction to Information Retrieval. Cambridge University Press.

[19] Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. 2013a. **Efficient Estimation of Word Representations in Vector Space**. In ICLR Workshop Papers.

[20] Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, and Jeffrey Dean. 2013b. **Distributed representations of words and phrases and their compositionality**. In NIPS, pages 3111–3119.

[21] Tomas Mikolov, Wen tau Yih, and Geoffrey Zweig. 2013c. **Linguistic regularities in continuous space word representations**. In HLT-NAACL.

[22] George A. Miller and Walter G. Charles. 1991. Contextual correlates of semantic similarity. Language and cognitive processes, 6(1):1–28.

[23] Andriy Mnih and Koray Kavukcuoglu. 2013. **Learning word embeddings efficiently with noise-contrastive estimation**. In NIPS.

[24] Douglas L. T. Rohde, Laura M. Gonnerman, and David C. Plaut. 2006. **An improved model of semantic similarity based on lexical co-occurence**. Communications of the ACM, 8:627–633.

[25] Herbert Rubenstein and John B. Goodenough. 1965. Contextual correlates of synonymy. Communications of the ACM, 8(10):627–633.

[26] Fabrizio Sebastiani. 2002. Machine learning in automated text categorization. ACM Computing Surveys, 34:1–47.

[27] Richard Socher, John Bauer, Christopher D. Manning, and Andrew Y. Ng. 2013. Parsing With Compositional Vector Grammars. In ACL.

[28] Stefanie Tellex, Boris Katz, Jimmy Lin, Aaron Fernandes, and Gregory Marton. 2003. Quantitative evaluation of passage retrieval algorithms for question answering. In Proceedings of the SIGIR Conference on Research and Development in Informaion Retrieval.

[29] Erik F. Tjong Kim Sang and Fien De Meulder. 2003. Introduction to the CoNLL-2003 shared task: Language-independent named entity recognition. In CoNLL-2003.

[30] Joseph Turian, Lev Ratinov, and Yoshua Bengio. 2010. **Word representations: a simple and general method for semi-supervised learning**. In Proceedings of ACL, pages 384–394.

[31] Mengqiu Wang and Christopher D. Manning. 2013. Effect of non-linear deep architecture in sequence labeling. In Proceedings of the 6th International Joint Conference on Natural Language Processing (IJCNLP).
