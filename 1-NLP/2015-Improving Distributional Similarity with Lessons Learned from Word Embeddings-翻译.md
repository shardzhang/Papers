# 借鉴词嵌入的经验教训改进分布相似度（Improving Distributional Similarity with Lessons Learned from Word Embeddings）

> Omer Levy, Yoav Goldberg, Ido Dagan | 以色列拉马特甘 巴伊兰大学（Bar-Ilan University）计算机科学系 | {omerlevy,yogo,dagan}@cs.biu.ac.il



本文通过 672 个词表示的对照实验证明，word2vec 等"神经"词嵌入相对传统计数方法（PPMI/SVD）的性能优势主要来自 **系统设计与超参数调优**，而非 **嵌入算法本身**。核心发现是——**把这些超参数（动态窗口、子采样、负采样数、上下文分布平滑等）迁移到传统方法后，二者性能大体相当，任何单一方法都不具备全局优势**。

核心内容：

- 痛点：Baroni et al. (2014) 等研究声称预测型嵌入一致且显著优于计数型分布方法，但这些比较未控制超参数
- 方法：把 word2vec/GloVe 的设计选择显式化为可迁移超参数，逐一移植到 PPMI 与 SVD（共 9 个超参数、4 种方法）
- 创新：提出 PMI 的平滑变体 $PMI_\alpha$ （cds = 0.75），缓解 PMI 对罕见词的偏置，平均每个任务提升超 3 个点
- 验证：8 个数据集（6 个相似度 + 2 个类比），4 种场景（vanilla、word2vec 默认、oracle 最优、2 折交叉验证）

关键发现：

- 超参数调优平均提升 **超 6 个点**、最高 15.7 个点，大于换算法或扩大语料库的收益
- SVD"正确"使用（eig = 1）性能暴跌——win = 2 时平均准确率从 .612 跌至 .551，是 Baroni et al. 得出"嵌入更优"结论的主因
- 公平调参后 SGNS 与 SVD 在相似度任务上难分伯仲（差距 ≤1.7 点），仅 MSR 句法类比上 SGNS/GloVe 明显占优
- 唯一可"盲开"的超参数是 cds = 0.75，一致提升且风险不显著；SGNS 是最稳健、最快、最省资源的基线

---



## 摘要

近期趋势表明，受神经网络启发的词嵌入模型在词相似度和类比检测任务上优于传统的基于计数的分布模型。我们揭示，词嵌入的大部分性能提升源于某些系统设计选择和超参数优化，而非嵌入算法本身。此外，我们表明这些修改可以迁移到传统的分布模型上，从而带来类似的提升。与此前的报告相反，我们观察到方法之间的性能差异大多是局部的或不显著的，没有任何单一方法相对其他方法具有全局优势。



## 1. 引言

理解词的含义是自然语言处理（NLP，Natural Language Processing）的核心。虽然深度的、类人式的理解仍然难以企及，但许多方法已经成功地恢复了词之间相似性的某些方面。最近，各种作者提出了基于神经网络的方法，将词"嵌入"到低维空间中（Bengio et al., 2003; Collobert and Weston, 2008）[4, 10]。这些模型将每个词表示为一个 $d$ 维实数向量，彼此接近的向量被证明在语义上相关。特别是，Mikolov 等人（2013a; 2013b）[23, 24] 的一系列论文最终提出了带负采样的 skip-gram 训练方法（SGNS，Skip-Gram with Negative Sampling）：一种高效的嵌入算法，在各种语言任务上提供了当时最先进的结果。它通过 word2vec——一个用于创建词嵌入的程序——而广为人知。

Baroni 等人 (2014) [3] 最近的一项研究进行了一组系统性实验，将 word2vec 嵌入与更传统的分布方法（例如逐点互信息（PMI，Pointwise Mutual Information）矩阵）进行比较（全面的综述见 Turney and Pantel (2010) [32] 和 Baroni and Lenci (2010) [2]）。这些结果表明，新的嵌入方法在许多面向相似度的任务上以不小的差距一致优于传统方法。然而，最先进的嵌入方法都基于相同的词的上下文袋（bag-of-contexts）表示。此外，Levy 和 Goldberg (2014c) [20] 的分析表明，word2vec 的 SGNS 隐式地分解了一个词-上下文 PMI 矩阵。也就是说，SGNS 的数学目标和可用信息来源实际上与更传统的方法所采用的非常相似。

那么，这些近期嵌入的优越性（或被感知的优越性）的来源是什么？虽然词嵌入文献的呈现焦点是数学模型和被优化的目标，但其他因素也会影响结果。特别是，嵌入算法提出了一些可以调优的自然超参数；其中许多已经在某种程度上被算法的设计者调优过。一些超参数，例如要使用的负样本数量，被明确标记为可调。其他修改，例如负采样分布的平滑，只是顺带提及，之后被视为算法的一部分。还有一些，例如动态大小的上下文窗口，甚至在一些论文中根本没有被提及，但却是标准实现的一部分。所有这些修改和系统设计选择——我们统称为超参数——都是最终算法的一部分，并且正如我们所展示的，对性能有重大影响。

在这项工作中，我们使这些超参数显式化，并展示如何将它们适配并迁移到传统的基于计数的方法中。为了评估每个超参数对算法性能的贡献，我们进行了一组全面的实验，在控制各种超参数的同时比较了四种不同的表示方法。一旦在各方法之间完成适配，超参数调优就能显著提升每个任务的性能。在许多情况下，改变单个超参数的设置所带来的性能提升，比切换到更好的算法或在更大的语料库上训练更大。

特别是，word2vec 对负采样分布的平滑可以通过引入一种新颖的 PMI 关联度量平滑变体（见 3.2 节），适配到基于 PPMI 的方法上。使用该变体平均每个任务提升超过 3 个点。我们怀疑这种平滑部分地解决了 PMI 的"阿喀琉斯之踵"：它对罕见词共现的偏置。

我们还表明，当所有方法都被允许调优一组类似的超参数时，它们的性能在很大程度上是可比的。事实上，没有任何一种算法方法相对另一种具有一致的优势，这一结果与嵌入优于基于计数的方法的主张相矛盾。

## 2. 背景

我们考虑四种词表示方法：显式 PPMI 矩阵、对该矩阵进行 SVD 分解、SGNS 和 GloVe。出于历史原因，我们将 PPMI 和 SVD 称为"基于计数"的表示，与之相对，SGNS 和 GloVe 通常被称为"神经"或"基于预测"的嵌入。所有这些方法（以及所有其他基于 skip-gram 的嵌入方法）本质上都是词袋模型，其中每个词的表示反映了与之共现的上下文词的加权词袋。此前已经证明，这种词袋嵌入模型在相似度和类比任务上的表现与更复杂的嵌入方法相当或更好（Mikolov et al., 2013a; Pennington et al., 2014）[23, 28]。

**记号**：我们假设词的集合 $w \in V_W$ 及其上下文 $c \in V_C$ ，其中 $V_W$ 和 $V_C$ 分别是词表和上下文表，并将观察到的词-上下文对的集合记为 $D$ 。我们用 $\#(w, c)$ 表示对 $(w, c)$ 在 $D$ 中出现的次数。类似地， $\#(w) = \sum_{c' \in V_C} \#(w, c')$ 和 $\#(c) = \sum_{w' \in V_W} \#(w', c)$ 分别是 $w$ 和 $c$ 在 $D$ 中出现的次数。在一些算法中，词和上下文被嵌入到 $d$ 维空间中。在这些情况下，每个词 $w \in V_W$ 关联一个向量 $\vec{w} \in \mathbb{R}^d$ ，类似地每个上下文 $c \in V_C$ 被表示为一个向量 $\vec{c} \in \mathbb{R}^d$ 。我们有时把向量 $\vec{w}$ 称为 $|V_W| \times d$ 矩阵 $W$ 中的行，把向量 $\vec{c}$ 称为 $|V_C| \times d$ 矩阵 $C$ 中的行。当提及由特定方法 $x$ 产生的嵌入时，我们可能使用 $W^x$ 和 $C^x$ （例如 $W^{SGNS}$ 或 $C^{SVD}$ ）。所有向量在用于相似度计算之前都归一化为单位长度，使余弦相似度和点积等价（进一步讨论见 3.3 节）。

**上下文**： $D$ 通常通过取一个语料库 $w_1, w_2, \ldots, w_n$ ，并把词 $w_i$ 的上下文定义为它周围 $L$ 大小窗口内的词 $w_{i-L}, \ldots, w_{i-1}, w_{i+1}, \ldots, w_{i+L}$ 来获得。虽然其他上下文定义已经被研究过（Padó and Lapata, 2007; Baroni and Lenci, 2010; Levy and Goldberg, 2014a）[26, 2, 18]，但本工作聚焦于固定窗口的词袋上下文。

### 2.1 显式表示（PPMI 矩阵）

分布方法中表示词的传统方式是构建一个高维稀疏矩阵 $M$ ，其中每一行表示词表 $V_W$ 中的一个词 $w$ ，每一列表示一个潜在的上下文 $c \in V_C$ 。每个矩阵单元 $M_{ij}$ 的值表示词 $w_i$ 与上下文 $c_j$ 之间的关联。这种关联的一种流行度量是逐点互信息（PMI）（Church and Hanks, 1990）[9]。PMI 定义为 $w$ 和 $c$ 的联合概率与其边际概率乘积之比的对数，可以通过下式估计：

$$
PMI(w, c) = \log \frac{\hat{P}(w, c)}{\hat{P}(w) \hat{P}(c)} = \log \frac{\#(w, c) \cdot |D|}{\#(w) \cdot \#(c)}
$$

$M^{PMI}$ 的行包含许多在语料库中从未观察到的词-上下文对 $(w, c)$ 的条目，对于这些对， $PMI(w, c) = \log 0 = -\infty$ 。因此，一种常见的方法是用 $M^{PMI_0}$ 替换 $M^{PMI}$ ，其中当 $\#(w, c) = 0$ 时 $PMI(w, c) = 0$ 。一种更一致的方法是使用正 PMI（PPMI，Positive Pointwise Mutual Information），其中所有负值被替换为 0：

$$
PPMI(w, c) = \max(PMI(w, c), 0)
$$

Bullinaria 和 Levy (2007) [6] 表明 $M^{PPMI}$ 在语义相似度任务上优于 $M^{PMI_0}$ 。PMI 的一个众所周知的缺点——在 PPMI 中依然存在——是它对低频事件的偏置（Turney and Pantel, 2010）[32]。一个罕见的上下文 $c$ 即使只与目标词 $w$ 共现一次，也往往会产生相对较高的 PMI 分数，因为 $\hat{P}(c)$——PMI 的分母——非常小。这造成一种情况： $w$ 最靠前的"分布特征"（上下文）往往是极其罕见的词，这些词不一定出现在与 $w$ 语义相似的词的相应表示中。尽管如此，PPMI 度量被广泛认为是这类分布相似度模型的最先进水平。

### 2.2 奇异值分解（SVD）

虽然稀疏向量表示效果很好，但使用稠密低维向量也有优势，例如更高的计算效率以及（可以说）更好的泛化能力。这样的向量可以通过对稀疏高维矩阵进行降维来获得。

这样做的一种常见方法是截断奇异值分解（SVD，Singular Value Decomposition），它找到关于 $L_2$ 损失的最优秩 $d$ 分解（Eckart and Young, 1936）[12]。它通过潜在语义分析（LSA，Latent Semantic Analysis）在 NLP 中得到普及（Deerwester et al., 1990）[11]。

SVD 将 $M$ 分解为三个矩阵的乘积 $U \cdot \Sigma \cdot V^\top$ ，其中 $U$ 和 $V$ 是标准正交的， $\Sigma$ 是按降序排列的特征值的对角矩阵。只保留 $\Sigma$ 的前 $d$ 个元素，我们得到 $M_d = U_d \cdot \Sigma_d \cdot V_d^\top$ 。 $W = U_d \cdot \Sigma_d$ 的行之间的点积等于 $M_d$ 的行之间的点积。

在词-上下文矩阵的设置中， $W$ 的稠密 $d$ 维行可以替代 $M$ 的极高维行。实际上，NLP 文献中的一种常见方法是用 SVD 分解 PPMI 矩阵 $M^{PPMI}$ ，然后取：

$$
W^{SVD} = U_d \cdot \Sigma_d \qquad C^{SVD} = V_d \qquad (1)
$$

分别作为词表示和上下文表示。

### 2.3 带负采样的 Skip-Gram（SGNS）

我们简要介绍 SGNS——(Mikolov et al., 2013a) [23] 中引入的 skip-gram 嵌入模型，使用 (Mikolov et al., 2013b) [24] 中提出的负采样过程进行训练。SGNS 的详细推导见 (Goldberg and Levy, 2014) [16]。

SGNS 试图把每个词 $w \in V_W$ 和每个上下文 $c \in V_C$ 表示为 $d$ 维向量 $\vec{w}$ 和 $\vec{c}$ ，使彼此"相似"的词具有相似的向量表示。它通过尝试最大化 $D$ 中出现的 $(w, c)$ 对的乘积 $\vec{w} \cdot \vec{c}$ 的一个函数，并对负例——不一定出现在 $D$ 中的 $(w, c_N)$ 对——最小化该函数来实现。负例通过对 $D$ 中观察到的 $(w, c)$ 对进行随机破坏来创建——因此得名"负采样"。对于每个观察到的 $(w, c)$ ，SGNS 从经验 unigram 分布 $P_D(c) = \frac{\#(c)}{|D|}$ 中抽取 $k$ 个上下文。在 word2vec 的 SGNS 实现中，这个分布被平滑，这是一个提升其性能的设计选择。我们在第 3 节中探索这一超参数及其他超参数。

**SGNS 作为隐式矩阵分解**：Levy 和 Goldberg (2014c) [20] 表明，SGNS 的语料库级目标在其取得最优值时满足：

$$
\vec{w} \cdot \vec{c} = PMI(w, c) - \log k
$$

因此，SGNS 隐式地分解了一个词-上下文矩阵，其单元值是 PMI 偏移一个全局常数（ $\log k$ ）后的结果：

$$
W \cdot C^\top = M^{PMI} - \log k
$$

SGNS 执行的分解与传统 SVD 不同（见 2.2 节）。特别是，该分解的损失函数不基于 $L_2$ ，并且由于包围 $\vec{w} \cdot \vec{c}$ 的 sigmoid 函数，它对极端值和无穷值的敏感度低得多。此外，损失是加权的，使罕见的 $(w, c)$ 对目标的影响远小于频繁的对。因此，虽然 $M^{PMI}$ 中的许多单元等于 $\log 0 = -\infty$ ，但把这些单元重建为一个小负值（例如 $-5$ 而不是 $-\infty$ ）所产生的代价可以忽略不计。¹

¹logistic（sigmoid）目标也抑制了非常高的 PMI 正值。我们怀疑这一特性，连同加权分解的特性，解决了前述 PMI 的缺点，即它对低频事件的过度加权。

与 SVD 的另一个区别——将在 3.3 节中进一步探索——是 SVD 把 $M$ 分解为三个矩阵，其中两个是标准正交的、一个是对角的，而 SGNS 把 $M$ 分解为两个无约束矩阵。

### 2.4 全局向量（GloVe）

GloVe（Pennington et al., 2014）[28] 试图把每个词 $w \in V_W$ 和每个上下文 $c \in V_C$ 表示为 $d$ 维向量 $\vec{w}$ 和 $\vec{c}$ ，使得：

$$
\vec{w} \cdot \vec{c} + b_w + b_c = \log(\#(w, c)) \qquad \forall (w, c) \in D
$$

这里， $b_w$ 和 $b_c$ （标量）是词/上下文特定的偏置，也是除 $\vec{w}$ 和 $\vec{c}$ 之外要学习的参数。

GloVe 的目标被显式定义为对数计数矩阵的分解，偏移整个词表的偏置项：

$$
M^{\log(\#(w, c))} \approx W \cdot C^\top + \vec{b}_w + \vec{b}_c
$$

其中 $\vec{b}_w$ 是一个 $|V_W|$ 维行向量， $\vec{b}_c$ 是一个 $|V_C|$ 维列向量。

如果我们固定 $b_w = \log \#(w)$ 和 $b_c = \log \#(c)$ ，这将几乎²等价于分解偏移了 $\log(|D|)$ 的 PMI 矩阵。然而，GloVe 学习这些参数，获得了相对 SVD 和 SGNS 的额外自由度。模型通过最小化加权最小二乘损失来拟合，给频繁的 $(w, c)$ 对更多权重。³

²GloVe 的目标忽略在训练语料库中不共现的 $(w, c)$ 对，把它们视为缺失值。另一方面，SGNS 通过负采样过程确实考虑了这样的对。

³加权公式是另一个可以调优的超参数，但我们保持默认加权方案。

最后，(Pennington et al., 2014) [28] 引入的一个重要创新是，假设 $V_C = V_W$ ，可以把词 $w$ 的表示取为 $\vec{w} + \vec{c}_w$ ，其中 $\vec{c}_w$ 是 $C^\top$ 中对应于 $w$ 的行。这在某些情况下可以显著改善结果，如我们在 3.3 节和 6.2 节中讨论的。

## 3. 可迁移超参数

本节介绍 word2vec 和 GloVe 中实现的各种超参数，并展示如何将它们适配并应用于基于计数的方法。我们把这些分为：预处理超参数，影响算法的输入数据；关联度量超参数，定义词-上下文交互的计算方式；以及后处理超参数，修改生成的词向量。

### 3.1 预处理超参数

所有基于矩阵的算法都依赖词-上下文对 $(w, c)$ 的集合 $D$ 作为输入。word2vec 引入了三种关于 $D$ 收集方式的新变体，它们可以轻松应用于 SGNS 之外的其他方法。

**动态上下文窗口（dyn）**：传统方法通常使用固定大小的无加权上下文窗口。例如，如果窗口大小为 5，那么与目标相距五个 token 的词与相邻词被同等对待。遵循"离目标更近的上下文更重要"的直觉，上下文词可以按照它们与焦点词的距离进行加权。GloVe 和 word2vec 都采用了这样的加权方案，虽然不太常见，但这种方法在传统的基于计数的方法中也被探索过，例如 (Sahlgren, 2006) [30]。

GloVe 的实现使用调和函数对上下文加权，例如，相距三个 token 的上下文词将被计为一次出现的 $\frac{1}{3}$ 。另一方面，word2vec 的实现等价于按与焦点词的距离除以窗口大小来加权。例如，大小为 5 的窗口将按 $\frac{5}{5}, \frac{4}{5}, \frac{3}{5}, \frac{2}{5}, \frac{1}{5}$ 对其上下文加权。

我们称这种修改为动态上下文窗口，是因为 word2vec 通过为每个 token 在 1 和 $L$ 之间均匀采样实际窗口大小来实现其加权方案（Mikolov et al., 2013a）[23]。就训练时间而言，采样方法比直接方法更快，因为 SGNS 中的 SGD 更新更少，其他方法中的非零矩阵单元也更少。对于我们的系统实验，我们对所有方法（包括 GloVe）都使用了 word2vec 风格的采样版本。

**子采样（sub）**：子采样是一种稀释非常频繁的词的方法，类似于移除停用词。(Mikolov et al., 2013a) [23] 中提出的子采样方法以概率 $p$ 随机移除比某个阈值 $t$ 更频繁的词，其中 $f$ 表示词的语料库频率：

$$
p = 1 - \sqrt{\frac{t}{f}} \qquad (2)
$$

遵循 (Mikolov et al., 2013a) [23] 中的建议，我们在实验中使用 $t = 10^{-5}$ 。⁴

⁴word2vec 的代码实现了一个略有不同的公式： $p = \frac{f-t}{f} - \sqrt{\frac{t}{f}}$ 。我们遵循原始论文中提出的公式（方程 2）。

word2vec 中子采样的另一个实现细节是，token 的移除发生在语料库被处理成词-上下文对之前。这实际上扩大了许多 token 的上下文窗口大小，因为它们现在可以接触到原本不在其 $L$ 大小窗口内的词。我们把这种子采样称为"dirty"（脏），与"clean"（干净）子采样相对，后者移除被子采样的词而不影响上下文窗口的大小。我们发现它们对性能的影响相当，只报告"dirty"变体的结果。

**删除罕见词（del）**：虽然忽略训练语料库中罕见的词很常见，但 word2vec 在创建上下文窗口之前就从语料库中移除了这些 token。与子采样一样，这种变体缩小了 token 之间的距离，在相同的窗口大小下插入了原始语料库中不存在的新词-上下文对。虽然这种变体也可能对性能有影响，但初步实验表明影响很小，因此我们在本文中不研究它的影响。

### 3.2 关联度量超参数

词与其上下文之间的 PMI（或 PPMI）在词相似度文献中被公认为一种有效的关联度量。Levy 和 Goldberg (2014c) [20] 表明 SGNS 隐式地分解了一个单元值为偏移 PMI 的词-上下文矩阵。遵循他们的分析，我们提出 PMI（以及隐式地 PPMI）关联度量的两种变体，我们从 SGNS 中采用它们。PMI 的这些增强不直接适用于 GloVe，因为 GloVe 按定义使用不同的关联度量。

**偏移 PMI（neg）**：SGNS 有一个自然的超参数 $k$ （负样本数量），它影响 SGNS 为每个 $(w, c)$ 试图优化的值： $PMI(w, c) - \log k$ 。由 $k > 1$ 引起的偏移可以通过偏移 PPMI（Levy and Goldberg, 2014c）[20] 应用于分布方法：

$$
SPPMI(w, c) = \max(PMI(w, c) - \log k, 0)
$$

重要的是要理解，在 SGNS 中， $k$ 有两个不同的功能。首先，它用于更好地估计负例的分布；更高的 $k$ 意味着更多的数据和更好的估计。其次，它作为观察到正例（ $(w, c)$ 在语料库中的实际出现）对负例的概率的先验；更高的 $k$ 意味着负例更可能。偏移 PPMI 只捕获 $k$ 的第二个方面（先验）。我们实验 $k$ 的三个值：1、5、15。

**上下文分布平滑（cds）**：在 word2vec 中，负例（上下文）按照平滑的 unigram 分布采样。为了平滑原始上下文的分布，所有上下文计数被提升到 $\alpha$ 的幂（Mikolov et al. (2013b) [24] 发现 $\alpha = 0.75$ 效果良好）。这种平滑变体在直接计算 PMI 时有一个对应物：

$$
PMI_\alpha(w, c) = \log \frac{\hat{P}(w, c)}{\hat{P}(w) \hat{P}_\alpha(c)} \qquad (3)
$$

$$
\hat{P}_\alpha(c) = \frac{\#(c)^\alpha}{\sum_c \#(c)^\alpha}
$$

与其他平滑技术（Pantel and Lin, 2002; Turney and Littman, 2003）[27, 31] 一样，上下文分布平滑缓解了 PMI 对罕见词的偏置。它通过放大采样一个罕见上下文的概率来实现这一点（因为当 $c$ 不频繁时 $\hat{P}_\alpha(c) > \hat{P}(c)$ ），这反过来降低了与罕见上下文 $c$ 共现的任何 $w$ 的 $(w, c)$ 的 PMI。在 6.2 节中我们证明，这个新颖的 PMI 变体非常有效，并且在任务、方法和配置之间一致地改进性能。我们实验 $\alpha$ 的两个值：1（未平滑）和 0.75（平滑）。

### 3.3 后处理超参数

我们介绍三个修改算法输出（词向量）的超参数。

**添加上下文向量（w+c）**：Pennington 等人 (2014) [28] 提议除了词向量之外使用上下文向量作为 GloVe 的输出。例如，词"cat"可以表示为：

$$
\vec{v}_{cat} = \vec{w}_{cat} + \vec{c}_{cat}
$$

其中 $\vec{w}$ 和 $\vec{c}$ 分别是词嵌入和上下文嵌入。

这种向量组合最初的动机是一种集成方法。在这里，我们对其对余弦相似度函数的影响提供一种不同的解释。具体来说，我们表明添加上下文向量有效地向二阶相似度函数添加了一阶相似度项。

考虑两个词的余弦相似度：

$$
\begin{aligned}
\cos(x, y) &= \frac{\vec{v}_x \cdot \vec{v}_y}{\sqrt{\vec{v}_x \cdot \vec{v}_x} \sqrt{\vec{v}_y \cdot \vec{v}_y}} \\
&= \frac{(\vec{w}_x + \vec{c}_x) \cdot (\vec{w}_y + \vec{c}_y)}{\sqrt{(\vec{w}_x + \vec{c}_x) \cdot (\vec{w}_x + \vec{c}_x)} \sqrt{(\vec{w}_y + \vec{c}_y) \cdot (\vec{w}_y + \vec{c}_y)}} \\
&= \frac{\vec{w}_x \cdot \vec{w}_y + \vec{c}_x \cdot \vec{c}_y + \vec{w}_x \cdot \vec{c}_y + \vec{c}_x \cdot \vec{w}_y}{\sqrt{\vec{w}_x^2 + 2 \vec{w}_x \cdot \vec{c}_x + \vec{c}_x^2} \sqrt{\vec{w}_y^2 + 2 \vec{w}_y \cdot \vec{c}_y + \vec{c}_y^2}} \\
&= \frac{\vec{w}_x \cdot \vec{w}_y + \vec{c}_x \cdot \vec{c}_y + \vec{w}_x \cdot \vec{c}_y + \vec{c}_x \cdot \vec{w}_y}{2 \sqrt{\vec{w}_x \cdot \vec{c}_x + 1} \sqrt{\vec{w}_y \cdot \vec{c}_y + 1}} \qquad (4)
\end{aligned}
$$

（最后一步成立是因为，如第 2 节所述，词向量和上下文向量在训练后被归一化。）

得到的表达式组合了可以分为两组的相似度项：二阶相似度（ $\vec{w}_x \cdot \vec{w}_y$ 、 $\vec{c}_x \cdot \vec{c}_y$ ）和一阶相似度（ $\vec{w}_* \cdot \vec{c}_*$ ）。二阶项衡量两个词基于它们出现在相似上下文中的倾向的可替换程度，是 Harris (1954) [17] 分布假说的体现。一阶项衡量一个词出现在另一个词上下文中的倾向。

在 SVD 和 SGNS 中， $w$ 和 $c$ 之间的一阶相似度项收敛到 $PMI(w, c)$ ，而在 GloVe 中它收敛到它们的对数计数（带有一些偏置项）。

因此，方程 4 中计算的相似度是 $x$ 和 $y$ 的一阶和二阶相似度的对称组合，由它们反射性一阶相似度的函数归一化：

$$
sim(x, y) = \frac{sim_2(x, y) + sim_1(x, y)}{\sqrt{sim_1(x, x) + 1} \sqrt{sim_1(y, y) + 1}}
$$

这种相似度度量表明，如果词倾向于出现在相似的上下文中，或者倾向于出现在彼此的上下文中（最好是两者兼有），那么它们是相似的。

加性的 w+c 表示可以轻松应用于产生不同词向量和上下文向量的其他方法（例如 SVD 和 SGNS）。另一方面，像 PPMI 这样的显式方法按定义是稀疏的，并且消除了绝大多数一阶相似度。因此，我们在本研究中不把 w+c 应用于 PPMI。

**特征值加权（eig）**：如 2.2 节所述，使用 SVD 推导的词向量和上下文向量通常表示为（方程 1）：

$$
W^{SVD} = U_d \cdot \Sigma_d \qquad C^{SVD} = V_d
$$

然而，对于词相似度任务，这并不一定是 $W^{SVD}$ 的最优构造。我们注意到，在基于 SVD 的分解中，生成的词矩阵和上下文矩阵具有非常不同的性质。特别是，上下文矩阵 $C^{SVD}$ 是标准正交的，而词矩阵 $W^{SVD}$ 不是。另一方面，SGNS 训练过程实现的分解要"对称"得多，因为 $W^{W2V}$ 和 $C^{W2V}$ 都不是标准正交的，并且训练目标没有对任何一个矩阵给予特别的偏置。通过以下分解可以实现类似的对称性：

$$
W = U_d \cdot \sqrt{\Sigma_d} \qquad C = V_d \cdot \sqrt{\Sigma_d} \qquad (5)
$$

或者，特征值矩阵可以完全被舍弃：

$$
W = U_d \qquad C = V_d \qquad (6)
$$

虽然理论上不清楚为什么对称方法对语义任务更好，但它在经验上确实效果好得多（见 6.1 节）。Caron (2001) [8] 做了类似的观察，他建议添加一个参数 $p$ 来控制特征值矩阵 $\Sigma$ ：

$$
W^{SVD_p} = U_d \cdot \Sigma_d^p
$$

后来的研究表明，用指数 $p$ 对特征值矩阵 $\Sigma_d$ 加权可以对性能产生显著影响，并且应该被调优（Bullinaria and Levy, 2012; Turney, 2012）[7, 33]。借鉴 SGNS 的对称分解概念，本研究只实验 SVD 的对称变体（ $p = 0$ 、 $p = 0.5$ ；方程 (6) 和 (5)）以及传统分解（ $p = 1$ ；方程 (1)）。

**向量归一化（nrm）**：如第 2 节所述，所有向量（即 $W$ 的行）被归一化为单位长度（ $L_2$ 归一化），使点积运算等价于余弦相似度。这种归一化本身就是一个超参数设置，其他归一化也适用。平凡的情况是完全不使用归一化。另一个设置——由 Pennington 等人 (2014) [28] 使用——归一化 $W$ 的列而不是它的行。也可以考虑结合行和列归一化的第四种设置。

注意，列归一化类似于在 SVD 中舍弃特征值。虽然超参数设置 eig = 0 对 SVD 有重要的积极影响，但对其他方法的列归一化不能这么说。在初步实验中，我们尝试了上述四种不同的归一化方案（none、row、column 和 both），发现 $W$ 行的标准 $L_2$ 归一化（即使用余弦相似度度量）一致地更优。

## 4. 实验设置

我们探索了超参数、表示和评估数据集的大空间。

### 4.1 超参数空间

表 1 列举了超参数空间。我们生成了 72 个 PPMI、432 个 SVD、144 个 SGNS 和 24 个 GloVe 表示；总共 672 个。

| 超参数 | 探索值 | 适用方法 |
| --- | --- | --- |
| win | 2, 5, 10 | 全部 |
| dyn | none, with | 全部 |
| sub | none, dirty, clean† | 全部 |
| del | none, with† | 全部 |
| neg | 1, 5, 15 | PPMI, SVD, SGNS |
| cds | 1, 0.75 | PPMI, SVD, SGNS |
| w+c | only w, w + c | SVD, SGNS, GloVe |
| eig | 0, 0.5, 1 | SVD |
| nrm | none†, row, col†, both† | 全部 |

**表 1：本工作中探索的超参数空间。†仅在初步实验中探索。**

### 4.2 词表示

**语料库**：所有模型都在英语维基百科（2013 年 8 月转储）上训练，预处理包括移除非文本元素、句子切分和分词。语料库包含 7750 万个句子、15 亿个 token。模型使用焦点词两侧各 2、5 和 10 个 token 的窗口推导（窗口大小参数记为 win）。在语料库中出现少于 100 次的词被忽略，词表和上下文表的词汇量都为 189,533 个词项。

**训练嵌入**：我们用 SVD、SGNS 和 GloVe 训练 500 维表示。SGNS 使用 word2vec 的修改版本训练，该版本接收预先提取的词-上下文对序列（Levy and Goldberg, 2014a）[18]。GloVe 使用原始实现（Pennington et al., 2014）[28] 以 50 次迭代训练，应用于预先提取的词-上下文对。

### 4.3 测试数据集

我们在覆盖相似度和类比任务的八个数据集上评估每种词表示。

**词相似度**：我们使用六个数据集评估词相似度：流行的 WordSim353（Finkelstein et al., 2002）[15]，划分为两个数据集 WordSim Similarity 和 WordSim Relatedness（Zesch et al., 2008; Agirre et al., 2009）[34, 1]；Bruni 等人 (2012) [5] 的 MEN 数据集；Radinsky 等人 (2011) [29] 的 Mechanical Turk 数据集；Luong 等人 (2013) [21] 的 Rare Words 数据集；以及 Hill 等人 (2014) [13] 的 SimLex-999 数据集。所有这些数据集都包含词对以及人工标注的相似度分数。词向量的评估方法是按余弦相似度对词对排序，并测量与人工评分之间的相关性（Spearman 的 $\rho$ ）。

**类比**：两个类比数据集给出形如"a 之于 a* 如同 b 之于 b*"的问题，其中 b* 被隐藏，必须从整个词表中猜出。MSR（Microsoft Research）的类比数据集（Mikolov et al., 2013c）[25] 包含 8000 个形态句法类比问题，例如"good 之于 best 如同 smart 之于 smartest"。Google 的类比数据集（Mikolov et al., 2013a）[23] 包含 19544 个问题，其中约一半与 MSR 中的同类（句法类比），另一半更具语义性质，例如首都（"Paris 之于 France 如同 Tokyo 之于 Japan"）。过滤掉涉及词表外词（即在英语维基百科中出现少于 100 次的词）的问题后，MSR 剩下 7118 个实例，Google 剩下 19258 个实例。类比问题使用 3CosAdd（加法和减法）回答：

$$
\arg\max_{b^* \in V_W \setminus \{a^*, b, a\}} \cos(b^*, a^* - a + b) = \arg\max_{b^* \in V_W \setminus \{a^*, b, a\}} \left( \cos(b^*, a^*) - \cos(b^*, a) + \cos(b^*, b) \right)
$$

以及 3CosMul，后者是类比恢复的最先进水平（Levy and Goldberg, 2014b）[19]：

$$
\arg\max_{b^* \in V_W \setminus \{a^*, b, a\}} \frac{\cos(b^*, a^*) \cdot \cos(b^*, b)}{\cos(b^*, a) + \varepsilon}
$$

$\varepsilon = 0.001$ 用于防止除零。我们分别把这两种方法缩写为"Add"和"Mul"。类比问题的评估指标是 argmax 结果为正确答案（ $b^*$ ）的问题所占的百分比。

## 5. 结果

我们首先比较各种超参数配置的影响，观察到不同设置对性能有重大影响（5.1 节）；有时，这种改进比切换到不同的表示方法更大。然后我们表明，在一些任务中，仔细的超参数调优也可以比增加更多数据更重要（5.2 节）。最后，我们观察到我们的结果与词嵌入文献中最近的一些主张不一致，并认为这些差异源于之前实验中未控制的超参数设置（5.3 节）。

### 5.1 超参数 vs 算法

我们首先考察一个"vanilla"（朴素）场景（表 2），其中所有超参数都被"关闭"（设置为默认值）：小上下文窗口（win = 2）、无动态上下文（dyn = none）、无子采样（sub = none）、一个负样本（neg = 1）、无平滑（cds = 1）、无上下文向量（w+c = only w）以及默认特征值权重（eig = 0.0）。⁵ 总体而言，SVD 在大多数词相似度任务上优于其他方法，通常相对第二名有相当大的优势。相比之下，类比任务呈现出混合结果；SGNS 在 MSR 类比上取得最佳结果，而 PPMI 主导 Google 数据集。

⁵虽然更常见的是设置 eig = 1，但该设置会显著降低 SVD 的性能（见 6.1 节）。

| 方法 | WordSim Similarity | WordSim Relatedness | Bruni et al. MEN | Radinsky et al. M. Turk | Luong et al. Rare Words | Hill et al. SimLex | Google Add / Mul | MSR Add / Mul |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PPMI | .709 | .540 | .688 | .648 | .393 | .338 | .491 / .650 | .246 / .439 |
| SVD | .776 | .658 | .752 | .557 | .506 | .422 | .452 / .498 | .357 / .412 |
| SGNS | .724 | .587 | .686 | .678 | .434 | .401 | .530 / .552 | .578 / .592 |
| GloVe | .666 | .467 | .659 | .599 | .403 | .398 | .442 / .465 | .529 / .576 |

**表 2："vanilla"场景（所有超参数设为默认值）中每种方法跨不同任务的性能：win = 2；dyn = none；sub = none；neg = 1；cds = 1；w+c = only w；eig = 0.0。**

第二个场景（表 3）把超参数设置为 word2vec 的默认值：小上下文窗口（win = 2）⁶、动态上下文（dyn = with）、dirty 子采样（sub = dirty）、五个负样本（neg = 5）、上下文分布平滑（cds = 0.75）、无上下文向量（w+c = only w）以及默认特征值权重（eig = 0.0）。

⁶虽然 word2vec 的默认窗口大小是 5，但我们在表 2-4 中呈现单一窗口大小（win = 2），以便把 win 的影响与其他超参数的影响隔离开。用不同的窗口大小运行相同的实验揭示出类似的趋势。更宽窗口大小的额外结果显示在表 5 中。

| 方法 | WordSim Similarity | WordSim Relatedness | Bruni et al. MEN | Radinsky et al. M. Turk | Luong et al. Rare Words | Hill et al. SimLex | Google Add / Mul | MSR Add / Mul |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PPMI | .755 | .688 | .745 | .686 | .423 | .354 | .553 / .629 | .289 / .413 |
| SVD | .784 | .672 | .777 | .625 | .514 | .402 | .547 / .587 | .402 / .457 |
| SGNS | .773 | .623 | .723 | .676 | .431 | .423 | .599 / .625 | .514 / .546 |
| GloVe | .667 | .506 | .685 | .599 | .372 | .389 | .539 / .563 | .503 / .559 |
| CBOW | .766 | .613 | .757 | .663 | .480 | .412 | .547 / .591 | .557 / .598 |

**表 3：使用 word2vec 推荐配置时每种方法跨不同任务的性能：win = 2；dyn = with；sub = dirty；neg = 5；cds = 0.75；w+c = only w；eig = 0.0。CBOW 一并呈现以供比较。**

该场景的结果与 vanilla 场景的结果相当不同，许多情况下性能更好。然而，这种变化并不均匀，因为我们观察到不同的设置提升不同的算法。事实上，"哪种方法最好？"这个问题在相同任务上但使用不同超参数值比较时，可能会有完全不同的答案。例如，看表 2 和表 3，SVD 在 vanilla 场景中是 SimLex-999 的最佳算法，而在 word2vec 场景中，它的表现不如 SGNS。

第三个场景（表 4）在小上下文窗口（win = 2）下启用全部超参数范围；我们在每个任务上、给定每种超参数配置评估每种方法，并选择最佳性能。与 vanilla（表 2）和 word2vec 场景（表 3）相比，我们看到所有方法的性能都有相当大的提升：最佳超参数组合相对 vanilla 设置最高提升 15.7 个点，平均提升超过 6 个点。看来选择正确的超参数设置往往比选择最合适的算法影响更大。

| 方法 | WordSim Similarity | WordSim Relatedness | Bruni et al. MEN | Radinsky et al. M. Turk | Luong et al. Rare Words | Hill et al. SimLex | Google Add / Mul | MSR Add / Mul |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PPMI | .755 | .697 | .745 | .686 | .462 | .393 | .553 / .679 | .306 / .535 |
| SVD | .793 | .691 | .778 | .666 | .514 | .432 | .554 / .591 | .408 / .468 |
| SGNS | .793 | .685 | .774 | .693 | .470 | .438 | .676 / .688 | .618 / .645 |
| GloVe | .725 | .604 | .729 | .632 | .403 | .398 | .569 / .596 | .533 / .580 |

**表 4：假设 win = 2，使用每种方法与任务组合的最佳配置时，每种方法跨不同任务的性能。**

**主要结果**：表 4 中的数字来自一个"oracle"（神谕）实验，其中超参数在测试数据上调优，为超参数调优的潜在性能提升提供了一个上界。这样的收益在实践中可以实现吗？

表 5 描述了一个现实的场景，其中超参数在训练集上调优，训练集与未见的测试数据分离。我们还报告了不同窗口大小（win = 2、5、10）的结果。我们使用 2 折交叉验证，其中，对于每个任务，超参数在每一半数据上调优，并在另一半上评估。表 5 中报告的数字是每个数据点两次运行的平均值。

| win | 方法 | WordSim Similarity | WordSim Relatedness | Bruni et al. MEN | Radinsky et al. M. Turk | Luong et al. Rare Words | Hill et al. SimLex | Google Add / Mul | MSR Add / Mul |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | PPMI | .732 | .699 | .744 | .654 | .457 | .382 | .552 / .677 | .306 / .535 |
| 2 | SVD | .772 | .671 | .777 | .647 | .508 | .425 | .554 / .591 | .408 / .468 |
| 2 | SGNS | .789 | .675 | .773 | .661 | .449 | .433 | .676 / .689 | .617 / .644 |
| 2 | GloVe | .720 | .605 | .728 | .606 | .389 | .388 | .649 / .666 | .540 / .591 |
| 5 | PPMI | .732 | .706 | .738 | .668 | .442 | .360 | .518 / .649 | .277 / .467 |
| 5 | SVD | .764 | .679 | .776 | .639 | .499 | .416 | .532 / .569 | .369 / .424 |
| 5 | SGNS | .772 | .690 | .772 | .663 | .454 | .403 | .692 / .714 | .605 / .645 |
| 5 | GloVe | .745 | .617 | .746 | .631 | .416 | .389 | .700 / .712 | .541 / .599 |
| 10 | PPMI | .735 | .701 | .741 | .663 | .235 | .336 | .532 / .605 | .249 / .353 |
| 10 | SVD | .766 | .681 | .770 | .628 | .312 | .419 | .526 / .562 | .356 / .406 |
| 10 | SGNS | .794 | .700 | .775 | .678 | .281 | .422 | .694 / .710 | .520 / .557 |
| 10 | GloVe | .746 | .643 | .754 | .616 | .266 | .375 | .702 / .712 | .463 / .519 |
| 10 | SGNS-LS | .766 | .681 | .781 | .689 | .451 | .414 | .739 / .758 | .690 / .729 |
| 10 | GloVe-LS | .678 | .624 | .752 | .639 | .361 | .371 | .732 / .750 | .628 / .685 |

**表 5：使用 2 折交叉验证进行超参数调优时，每种方法跨不同任务的性能。大尺度（LS）语料库上的配置也一并呈现以供比较。**

结果表明，接近神谕的改进确实是可行的。把训练配置（表 5）的性能与最优配置（表 4）的性能相比，它们的平均差异约为 1%，更大的数据集通常能找到最优配置。因此，为词相似度和类比检测任务适当地调优超参数既实用又有益。

一个有趣的观察——看表 5 时立即显现——是没有单一方法一致地比其他方法表现更好。这种行为在所有窗口大小下都可见，并在 5.3 节中进一步详细讨论。

### 5.2 超参数 vs 大数据

评估分布方法的一个重要因素是语料库和词表的大小，更大的语料库往往产生更好的表示。然而，从更大的语料库训练词向量在计算时间上代价更高，而这些时间本可以花在调优超参数上。

为了比较更大数据的效应与更灵活的超参数设置的效应，我们创建了一个包含超过 105 亿词的大型语料库（比我们原始语料库大 7 倍）。这个语料库基于一个 Mikolov 建议用于训练 word2vec 的 85 亿词语料库构建⁷，我们在此基础上添加了 UKWaC（Ferraresi et al., 2008）[14]。与原始设置一样，我们的词表包含语料库中至少出现 100 次的每个词，约有 620,000 个词。最后，我们把上下文窗口固定为宽且动态（win = 10、dyn = with），并探索了 16 种超参数设置，包括：子采样（sub）、偏移 PMI（neg = 1、5）、上下文分布平滑（cds）以及添加上下文向量（w+c）。这个空间比原始超参数空间受到更多限制。

⁷word2vec.googlecode.com/svn/trunk/demo-train-big-model-v1.sh

在计算方面，SGNS 的扩展性很好，每个设置只需要大约半天的计算。另一方面，GloVe 在此语料库上运行单个 50 次迭代实例需要几天时间。把传统的基于计数的方法应用于此设置被证明在技术上有挑战，因为它们消耗太多内存而无法高效操作。因此，我们只呈现 SGNS 和 GloVe 的结果（表 5）。

值得注意的是，在某些情况下（3/6 的词相似度任务），调优更大的超参数空间确实比扩大语料库更有益。然而，在其他情况下，更多的数据似乎确实有回报，两个类比任务都证明了这一点。

### 5.3 重新评估先前的主张

先前的文献提出了关于某些方法优于其他方法的几个主张。然而，这些研究没有控制本工作中提出的超参数。因此，我们重新审视这些主张，并根据表 5 中的结果检验它们的有效性。⁸

⁸我们注意到，本节得出的所有结论都依赖于我们实验所用的具体数据和设置。在不同任务、数据和超参数上的实验确实可能得出其他结论。

**嵌入优于基于计数的分布方法吗？** 人们普遍认为，现代基于预测的嵌入比传统的基于计数的方法表现更好。这一主张最近得到了 Baroni 等人 (2014) [3] 一系列系统评估的支持。然而，我们的结果显示出不同的趋势。表 5 显示，在词相似度任务中，当 win = 2、5 时，SGNS 的平均分数实际上低于 SVD，并且在那些情况下它从未以超过 1.7 个点的优势超越 SVD。在 Google 类比中，SGNS 和 GloVe 确实比 PPMI 表现更好，但只有 3.7 个点的差距（比较 win = 2 的 PPMI 和 win = 5 的 SGNS）。MSR 的类比数据集是 SGNS 和 GloVe 大幅超越 PPMI 和 SVD 的唯一情况。⁹ 总体而言，一种方法相对另一种似乎没有一致且显著的优势，从而驳斥了基于预测的方法优于基于计数的方法的主张。

⁹与 PPMI 不同，SVD 在两个类比任务上都表现不佳。

(Baroni et al., 2014) [3] 中矛盾的结果源于创建 word2vec 嵌入时使用了某种程度上预先调优的超参数（由 word2vec 推荐），并将它们与"vanilla"的 PPMI 和 SVD 表示进行比较。特别是，偏移 PMI（负采样）和上下文分布平滑（cds = 0.75，3.2 节中的方程 (3)）为 SGNS 开启，但没有为 PPMI 和 SVD 开启。另一个区别是 Baroni 等人设置 eig = 1，这显著恶化了 SVD 的性能（见 6.1 节）。

**GloVe 优于 SGNS 吗？** Pennington 等人 (2014) [28] 展示了一系列实验，其中 GloVe 优于 SGNS（以及其他方法）。然而，我们的结果显示出完全相反的情况。事实上，SGNS 在每个任务上都优于 GloVe（表 5）。只有当限制在 3CosAdd——一个次优配置——时，GloVe 才对 SGNS 显示出 0.8 个点的优势。这一趋势在扩展到更大的语料库和词表时依然持续。

这一矛盾可以用实验设置中的三个主要差异来解释。首先，在我们的实验中，允许超参数变化；特别是，w+c 被应用于所有方法，包括 SGNS。其次，Pennington 等人 (2014) [28] 只在 Google 类比上评估，而不是 MSR。最后，在我们的工作中，所有方法都使用相同的底层语料库进行比较。

同样重要的是要记住，按定义，GloVe 不能使用两个超参数：偏移 PMI（neg）和上下文分布平滑（cds）。相反，GloVe 学习一组偏置参数，这些参数包含了这两种修改以及对 PMI 度量的许多其他潜在更改。尽管具有更大的灵活性，GloVe 在我们的实验中并不比 SGNS 表现更好。

**PPMI 在类比任务上与 SGNS 不相上下吗？** Levy 和 Goldberg (2014b) [19] 表明 PPMI 和 SGNS 在 Google 和 MSR 的类比任务上表现相似。然而，表 5 中的结果显示出对 SGNS 的明显优势。虽然 Google 类比上的差距不是很大（PPMI 仅落后 SGNS 3.7 个点），但 SGNS 在 MSR 数据集上以很大的差距一致超越 PPMI。MSR 的类比数据集捕捉句法关系，例如名词的单复数屈折和动词的时态变化。我们推测，捕捉这些句法关系可能依赖某些类型的上下文，例如限定词和功能词，而 SGNS 可能更擅长捕捉这些——也许是由于它给不同示例分配权重的方式，或者因为它还捕捉了被 PPMI 过滤掉的负相关。

深入了解 Levy 和 Goldberg (2014b) [19] 的实验会发现，PPMI 使用了位置上下文（即每个上下文是一个词及其相对目标词位置的结合），而 SGNS 使用的是常规的词袋上下文。位置上下文可能包含恢复句法类比的相关信息，这解释了 (Levy and Goldberg, 2014b) [19] 中 PPMI 在 MSR 类比任务上相对较高的分数。

**3CosMul 比 3CosAdd 恢复更多类比吗？** Levy 和 Goldberg (2014b) [19] 表明，使用相似度乘法（3CosMul）而不是加法（3CosAdd）在所有方法和每个任务上都改进了结果。这一主张与我们的发现一致；实际上，3CosMul 在每种情况下都主导 3CosAdd。这种改进对 SVD 和 PPMI 尤其明显，它们在使用 3CosAdd 时明显落后于其他方法。

### 5.4 与 CBOW 的比较

word2vec 中出现的另一个算法是 CBOW（Continuous Bag-of-Words，连续词袋）。与其他方法不同，CBOW 不能轻易表示为词-上下文矩阵的分解；它通过把上下文向量表示为其词的向量之和，把每个上下文窗口的 token 绑定在一起。因此，它比其他方法更具表现力，并且有潜力推导出更好的词表示。

虽然 Mikolov 等人 (2013b) [24] 发现 SGNS 优于 CBOW，但 Baroni 等人 (2014) [3] 报告 CBOW 有轻微优势。我们在把所有超参数设置为 word2vec 提供的默认值时（表 3）把 CBOW 与其他方法进行比较。除了 MSR 的类比任务外，CBOW 在该场景中不是任何其他任务的最佳表现方法。在我们的初步实验中，其他场景也显示出类似的趋势。

虽然 CBOW 潜在地可以通过组合每个上下文窗口中的 token 来推导出更好的表示，但这种潜力在实践中并没有实现。尽管如此，Melamud 等人 (2014) [22] 表明，捕捉联合上下文确实可以改进词相似度任务的性能，我们相信这是一个值得追求的方向。

## 6. 超参数分析

我们分析每个超参数的个体影响，并尝试刻画某种设置有益的条件。

### 6.1 有害配置

某些超参数设置可能会削弱某种方法的性能。我们观察到 SVD 表现不佳的两种场景。

**SVD 不从偏移 PPMI 中受益。** 设置 neg > 1 一致地恶化 SVD 的性能。Levy 和 Goldberg (2014c) [20] 做了类似的观察，并假设这是零单元数量增加的结果，这可能使 SVD 偏好非常接近零矩阵的分解。SVD 的 $L_2$ 目标是无权重的，它不区分观察到的和未观察到的矩阵单元。

**"正确"使用 SVD 是糟糕的。** 使用 SVD 表示词的传统方式使用特征值矩阵（eig = 1）： $W = U_d \cdot \Sigma_d$ 。尽管在理论上有很好的动机，与其他设置（eig = 0.5 或 0）相比，该设置在实践中的结果非常差。表 6 展示了这一差距。

| win | eig | 平均性能 |
| --- | --- | --- |
| 2 | 0 | .612 |
| 2 | 0.5 | .611 |
| 2 | 1 | .551 |
| 5 | 0 | .616 |
| 5 | 0.5 | .612 |
| 5 | 1 | .534 |
| 10 | 0 | .584 |
| 10 | 0.5 | .567 |
| 10 | 1 | .484 |

**表 6：vanilla 场景中，给定不同 eig 值时 SVD 在词相似度任务上的平均性能。**

设置 eig = 1 时平均准确率的下降令人震惊。这种性能差距在不同超参数设置下也持续存在，使用 eig = 1 而不是 eig = 0.5 或 0 时性能下降超过 15 个点（绝对值）并不罕见。该设置是 Baroni 等人 (2014) [3] 研究中 SVD 结果较差的主要原因之一，也是我们在 vanilla 场景中选择使用 eig = 0.5 作为 SVD 默认设置的原因。

### 6.2 有益配置

为了确定哪些超参数设置有益，我们查看了每种方法在每个任务上的最佳配置。然后，我们统计了这些配置中每个超参数设置被选择的次数（表 7）。出现了一些趋势，例如 PPMI 和 SVD 对较短上下文窗口¹⁰（win = 2）的偏好，以及 SGNS 总是偏好大量负样本（neg > 1）。

¹⁰这也可能与 PMI 对低频事件的偏置有关（见 2.1 节）。更宽的窗口与罕见词产生更多随机共现，用具有高 PMI 分数的随机词"污染"分布向量。

| 方法 | win 2 : 5 : 10 | dyn none : with | sub none : dirty | neg 1 : 5 : 15 | cds 1.00 : 0.75 | w+c only w : w + c |
| --- | --- | --- | --- | --- | --- | --- |
| PPMI | 7 : 1 : 0 | 4 : 4 | 4 : 4 | 2 : 6 : 0 | 1 : 7 | — |
| SVD | 7 : 1 : 0 | 4 : 4 | 1 : 7 | 8 : 0 : 0 | 2 : 6 | 7 : 1 |
| SGNS | 2 : 3 : 3 | 6 : 2 | 4 : 4 | 0 : 4 : 4 | 3 : 5 | 4 : 4 |
| GloVe | 1 : 3 : 4 | 6 : 2 | 7 : 1 | — | — | 4 : 4 |

**表 7：每个超参数的影响，以最佳配置采用该超参数设置的任务数来衡量。不适用的组合用"—"标记。**

为了更近距离地观察并隔离每个超参数的影响，我们控制了所述超参数，并比较了给定该超参数每个设置时的最佳配置。表 8 显示了每个超参数的默认与非默认设置之间的差异。虽然许多超参数设置可以改进性能，但如果选择不当，它们也可能使其退化。例如，在偏移 PMI（neg）的情况下，SGNS 一致地从 neg > 1 中受益，而 SVD 的性能急剧下降。对于 PPMI，应用 neg > 1 的效用取决于任务类型：词相似度还是类比。另一个例子是动态上下文窗口（dyn），它对 MSR 的类比任务有益，但对其他任务在很大程度上是有害的。

看来唯一可以在任何情况下"盲目"应用的超参数是上下文分布平滑（cds = 0.75），它以不显著的风险产生一致的改进。注意，cds 对 PPMI 的帮助大于其他方法；我们认为这是因为降低了罕见词对分布表示的相对影响，从而解决了 PMI 的"阿喀琉斯之踵"。

（a）dyn = with 与 dyn = none 的最佳模型之间的性能差异

| 方法 | WordSim Similarity | WordSim Relatedness | Bruni et al. MEN | Radinsky et al. M. Turk | Luong et al. Rare Words | Hill et al. SimLex | Google Mul | MSR Mul |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PPMI | +0.5% | –1.0% | 0.0% | +0.1% | +0.4% | –0.1% | –0.1% | +1.2% |
| SVD | –0.8% | –0.2% | 0.0% | +0.6% | +0.4% | –0.1% | +0.6% | +2.1% |
| SGNS | –0.9% | –1.5% | –0.3% | +0.1% | –0.1% | –0.1% | –1.0% | +0.7% |
| GloVe | –0.8% | –1.2% | –0.9% | –0.8% | +0.1% | –0.9% | –3.3% | +1.8% |

（b）sub = dirty 与 sub = none 的最佳模型之间的性能差异

| 方法 | WordSim Similarity | WordSim Relatedness | Bruni et al. MEN | Radinsky et al. M. Turk | Luong et al. Rare Words | Hill et al. SimLex | Google Mul | MSR Mul |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PPMI | +0.6% | +1.9% | +1.3% | +1.0% | –3.8% | –3.9% | –5.0% | –12.2% |
| SVD | +0.7% | +0.2% | +0.6% | +0.7% | +0.8% | –0.3% | +4.0% | +2.4% |
| SGNS | +1.5% | +2.2% | +1.5% | +0.1% | –0.4% | –0.1% | –4.4% | –5.4% |
| GloVe | +0.2% | –1.3% | –1.0% | –0.2% | –3.4% | –0.9% | –3.0% | –3.6% |

（c）neg > 1 与 neg = 1 的最佳模型之间的性能差异

| 方法 | WordSim Similarity | WordSim Relatedness | Bruni et al. MEN | Radinsky et al. M. Turk | Luong et al. Rare Words | Hill et al. SimLex | Google Mul | MSR Mul |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PPMI | +0.6% | +4.9% | +1.3% | +1.0% | +2.2% | +0.8% | –6.2% | –9.2% |
| SVD | –1.7% | –2.2% | –1.9% | –4.6% | –3.4% | –3.5% | –13.9% | –14.9% |
| SGNS | +1.5% | +2.9% | +2.3% | +0.5% | +1.5% | +1.1% | +3.3% | +2.1% |
| GloVe | — | — | — | — | — | — | — | — |

（d）cds = 0.75 与 cds = 1 的最佳模型之间的性能差异

| 方法 | WordSim Similarity | WordSim Relatedness | Bruni et al. MEN | Radinsky et al. M. Turk | Luong et al. Rare Words | Hill et al. SimLex | Google Mul | MSR Mul |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PPMI | +1.3% | +2.8% | 0.0% | +2.1% | +3.5% | +2.9% | +2.7% | +9.2% |
| SVD | +0.4% | –0.2% | +0.1% | +1.1% | +0.4% | –0.3% | +1.4% | +2.2% |
| SGNS | +0.4% | +1.4% | 0.0% | +0.1% | 0.0% | +0.2% | +0.6% | 0.0% |
| GloVe | — | — | — | — | — | — | — | — |

（e）w+c = w + c 与 w+c = only w 的最佳模型之间的性能差异

| 方法 | WordSim Similarity | WordSim Relatedness | Bruni et al. MEN | Radinsky et al. M. Turk | Luong et al. Rare Words | Hill et al. SimLex | Google Mul | MSR Mul |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PPMI | — | — | — | — | — | — | — | — |
| SVD | –0.6% | –0.2% | –0.4% | –2.1% | –0.7% | +0.7% | –1.8% | –3.4% |
| SGNS | +1.4% | +2.2% | +1.2% | +1.1% | –0.3% | –2.3% | –1.0% | –7.5% |
| GloVe | +2.3% | +4.7% | +3.0% | –0.1% | –0.7% | –2.6% | +3.3% | –8.9% |

**表 8：设置每个超参数的附加价值与风险。这些数字显示了把一个超参数限制为不同值时，最佳可实现配置之间的性能差异。这一差异表明了调优给定超参数的潜在收益，以及不调优时性能下降的风险。例如，表 (d) 中的 +9.2% 条目意味着 cds = 0.75 的最佳模型比 cds = 1 的最佳模型准确率高 9.2%（绝对值）；即在 MSR 类比上，使用 cds = 0.75 而不是 cds = 1 把 PPMI 的准确率从 .443 提高到 .535。**

## 7. 实用建议

通常建议为手头的任务调优所有超参数以及算法特定的超参数。然而，这在计算上可能代价高昂。因此，我们提供一些"经验法则"，我们发现它们在我们的设置中效果良好：

- 始终使用上下文分布平滑（cds = 0.75）来修改 PMI，如 3.2 节所述。它一致地改进性能，并且适用于 PPMI、SVD 和 SGNS。
- 不要"正确"使用 SVD（eig = 1）。相反，使用对称变体之一（3.3 节）。
- SGNS 是一个稳健的基线。虽然它可能不是每个任务的最佳方法，但它在任何场景中都不会显著落后。此外，SGNS 是训练最快的方法，并且在磁盘空间和内存消耗方面（远）最便宜。
- 使用 SGNS 时，偏好大量负样本。
- 对于 SGNS 和 GloVe，值得实验 $\vec{w} + \vec{c}$ 变体，它的应用成本低（不需要重新训练），可以带来可观的收益（也可能是可观的损失）。



## 8. 结论

近期的嵌入方法引入了大量超出网络架构和优化算法的设计选择。我们揭示，这些看似微小的变化可以对词表示方法的成功产生巨大影响。通过展示如何在传统方法中适配和调优这些超参数，我们实现了表示之间的适当比较，并挑战了词嵌入文献中各种优越性主张。

这项研究还暴露了对更多受控变量实验的需求，以及把"变量"的概念从显而易见的任务、数据和方法扩展到经常被忽视的预处理步骤和超参数设置的需求。我们还强调对透明和可复现实验的需求，并赞扬 Mikolov、Pennington 等作者公开提供其代码。本着这种精神，我们也公开我们的代码。¹¹

¹¹http://bitbucket.org/omerlevy/hyperwords



## 致谢

这项工作得到了 Google Research Award Program 和德国研究基金会通过 German-Israeli Project Cooperation（资助 DA 1600/1-1）的支持。我们感谢 Marco Baroni 和 Jeffrey Pennington 的宝贵意见。



## 参考文献

[1] Eneko Agirre, Enrique Alfonseca, Keith Hall, Jana Kravalova, Marius Pasca, and Aitor Soroa. 2009. A study on similarity and relatedness using distributional and wordnet-based approaches. In Proceedings of Human Language Technologies: The 2009 Annual Conference of the North American Chapter of the Association for Computational Linguistics, pages 19–27, Boulder, Colorado, June. Association for Computational Linguistics.

[2] Marco Baroni and Alessandro Lenci. 2010. Distributional memory: A general framework for corpus-based semantics. Computational Linguistics, 36(4):673–721.

[3] Marco Baroni, Georgiana Dinu, and Germán Kruszewski. 2014. Dont count, predict! a systematic comparison of context-counting vs. context-predicting semantic vectors. In Proceedings of the 52nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 238–247, Baltimore, Maryland, June. Association for Computational Linguistics.

[4] Yoshua Bengio, Réjean Ducharme, Pascal Vincent, and Christian Jauvin. 2003. A neural probabilistic language model. Journal of Machine Learning Research, 3:1137–1155.

[5] Elia Bruni, Gemma Boleda, Marco Baroni, and Nam Khanh Tran. 2012. Distributional semantics in technicolor. In Proceedings of the 50th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 136–145, Jeju Island, Korea, July. Association for Computational Linguistics.

[6] John A Bullinaria and Joseph P Levy. 2007. Extracting semantic representations from word co-occurrence statistics: a computational study. Behavior Research Methods, 39(3):510–526.

[7] John A Bullinaria and Joseph P Levy. 2012. Extracting semantic representations from word co-occurrence statistics: Stop-lists, stemming, and SVD. Behavior Research Methods, 44(3):890–907.

[8] John Caron. 2001. Experiments with LSA scoring: optimal rank and basis. In Proceedings of the SIAM Computational Information Retrieval Workshop, pages 157–169.

[9] Kenneth Ward Church and Patrick Hanks. 1990. Word association norms, mutual information, and lexicography. Computational Linguistics, 16(1):22–29.

[10] Ronan Collobert and Jason Weston. 2008. A unified architecture for natural language processing: Deep neural networks with multitask learning. In Proceedings of the 25th International Conference on Machine Learning, pages 160–167.

[11] Scott C. Deerwester, Susan T. Dumais, Thomas K. Landauer, George W. Furnas, and Richard A. Harshman. 1990. Indexing by latent semantic analysis. JASIS, 41(6):391–407.

[12] C Eckart and G Young. 1936. The approximation of one matrix by another of lower rank. Psychometrika, 1:211–218.

[13] Roi Reichart Felix Hill and Anna Korhonen. 2014. Simlex-999: Evaluating semantic models with (genuine) similarity estimation. arXiv preprint arXiv:1408.3456.

[14] Adriano Ferraresi, Eros Zanchetta, Marco Baroni, and Silvia Bernardini. 2008. Introducing and evaluating ukwac, a very large web-derived corpus of English. In Proceedings of the 4th Web as Corpus Workshop (WAC-4), pages 47–54.

[15] Lev Finkelstein, Evgeniy Gabrilovich, Yossi Matias, Ehud Rivlin, Zach Solan, Gadi Wolfman, and Eytan Ruppin. 2002. Placing search in context: The concept revisited. ACM Transactions on Information Systems, 20(1):116–131.

[16] Yoav Goldberg and Omer Levy. 2014. word2vec explained: deriving Mikolov et al.'s negative-sampling word-embedding method. arXiv preprint arXiv:1402.3722.

[17] Zellig Harris. 1954. Distributional structure. Word, 10(23):146–162.

[18] Omer Levy and Yoav Goldberg. 2014a. Dependency-based word embeddings. In Proceedings of the 52nd Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pages 302–308, Baltimore, Maryland.

[19] Omer Levy and Yoav Goldberg. 2014b. Linguistic regularities in sparse and explicit word representations. In Proceedings of the Eighteenth Conference on Computational Natural Language Learning, pages 171–180, Baltimore, Maryland.

[20] Omer Levy and Yoav Goldberg. 2014c. Neural word embeddings as implicit matrix factorization. In Advances in Neural Information Processing Systems 27: Annual Conference on Neural Information Processing Systems 2014, December 8-13 2014, Montreal, Quebec, Canada, pages 2177–2185.

[21] Minh-Thang Luong, Richard Socher, and Christopher D. Manning. 2013. Better word representations with recursive neural networks for morphology. In Proceedings of the Seventeenth Conference on Computational Natural Language Learning, pages 104–113, Sofia, Bulgaria, August. Association for Computational Linguistics.

[22] Oren Melamud, Ido Dagan, Jacob Goldberger, Idan Szpektor, and Deniz Yuret. 2014. Probabilistic modeling of joint-context in distributional similarity. In Proceedings of the Eighteenth Conference on Computational Natural Language Learning, pages 181–190, Baltimore, Maryland, June. Association for Computational Linguistics.

[23] Tomas Mikolov, Kai Chen, Gregory S. Corrado, and Jeffrey Dean. 2013a. Efficient estimation of word representations in vector space. In Proceedings of the International Conference on Learning Representations (ICLR).

[24] Tomas Mikolov, Ilya Sutskever, Kai Chen, Gregory S. Corrado, and Jeffrey Dean. 2013b. Distributed representations of words and phrases and their compositionality. In Advances in Neural Information Processing Systems, pages 3111–3119.

[25] Tomas Mikolov, Wen-tau Yih, and Geoffrey Zweig. 2013c. Linguistic regularities in continuous space word representations. In Proceedings of the 2013 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 746–751.

[26] Sebastian Padó and Mirella Lapata. 2007. Dependency-based construction of semantic space models. Computational Linguistics, 33(2):161–199.

[27] Patrick Pantel and Dekang Lin. 2002. Discovering word senses from text. In Proceedings of the eighth ACM SIGKDD international conference on Knowledge discovery and data mining, pages 613–619. ACM.

[28] Jeffrey Pennington, Richard Socher, and Christopher Manning. 2014. Glove: Global vectors for word representation. In Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 1532–1543, Doha, Qatar, October. Association for Computational Linguistics.

[29] Kira Radinsky, Eugene Agichtein, Evgeniy Gabrilovich, and Shaul Markovitch. 2011. A word at a time: Computing word relatedness using temporal semantic analysis. In Proceedings of the 20th international conference on World wide web, pages 337–346. ACM.

[30] Magnus Sahlgren. 2006. The Word-Space Model. Ph.D. thesis, Stockholm University.

[31] Peter D. Turney and Michael L. Littman. 2003. Measuring praise and criticism: Inference of semantic orientation from association. Transactions on Information Systems, 21(4):315–346.

[32] Peter D. Turney and Patrick Pantel. 2010. From frequency to meaning: Vector space models of semantics. Journal of Artificial Intelligence Research, 37(1):141–188.

[33] Peter D. Turney. 2012. Domain and function: A dual-space model of semantic relations and compositions. Journal of Artificial Intelligence Research, 44:533–585.

[34] Torsten Zesch, Christof Müller, and Iryna Gurevych. 2008. Using wiktionary for computing semantic relatedness. In Proceedings of the 23rd National Conference on Artificial Intelligence - Volume 2, AAAI'08, pages 861–866. AAAI Press.
