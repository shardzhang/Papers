# ComiRec: 面向推荐的可控多兴趣框架

> Yukuo Cen, Jianwei Zhang, Xu Zou, Cheng Zhou, Hongxia Yang, Jie Tang | Tsinghua University; Alibaba Group

本文提出可控多兴趣框架ComiRec，**通过多兴趣提取模块从用户行为序列中捕获多个兴趣向量，再通过可控聚合模块平衡推荐准确性与多样性，实现个性化推荐**。

核心内容：

- 工业推荐系统的召回阶段通常只生成单一用户嵌入，无法表达用户多样化的真实兴趣
- 提出两种多兴趣提取模块：基于动态路由的ComiRec-DR和基于自注意力的ComiRec-SA
- 设计可控聚合模块，通过超参数$\lambda$平衡推荐准确性和多样性
- 在Amazon Books和淘宝数据集上超越所有SOTA模型，并成功部署于阿里巴巴分布式云平台

关键发现：

- ComiRec-SA和ComiRec-DR在Amazon Books上Recall@50分别达到**8.47%和8.11%**，超越MIND的7.64%
- 在淘宝数据集上Recall@50分别达到**9.46%和9.82%**，超越MIND的8.16%
- 在十亿级工业数据集上，ComiRec-SA和ComiRec-DR相比MIND的Recall@50分别提升**1.39%和8.65%**
- 通过调节$\lambda$，推荐多样性从23.2%提升至**55.1%**，同时Recall仅下降约4.3%

---

## 摘要

我们提出了一种新颖的可控多兴趣框架ComiRec，用于序列推荐。我们的多兴趣模块可以捕获用户的多个兴趣，这些兴趣可用于检索候选item。聚合模块将来自不同兴趣的item组合在一起，输出最终的推荐结果。我们在序列推荐任务上进行了实验，结果表明我们的框架优于其他最先进的模型。我们的框架也已成功部署在阿里巴巴分布式云平台上。十亿级工业数据集的结果进一步证实了我们模型在实践中的有效性和效率。

本文的主要贡献如下：
- 我们提出了一个综合框架，将可控性和多兴趣组件统一在一个推荐系统中。
- 我们通过在在线推荐场景中实现和研究，探讨了可控性在个性化系统中的作用。
- 我们的框架在两个真实世界的挑战性数据集上取得了最先进的序列推荐性能。

## 2 相关工作

本节介绍与推荐系统和推荐多样性相关的文献，以及本文中使用的胶囊网络和注意力机制。

协同过滤[47, 48]方法已被证明在真实世界的推荐系统中非常成功，它通过找到相似的用户和item并在此基础上进行推荐。矩阵分解[30]是经典推荐研究中最流行的技术，它将用户和item映射到一个联合潜在因子空间，使得用户-item交互被建模为该空间中的内积。因子分解机（FM）[44]使用分解参数对所有变量之间的交互进行建模，因此即使在像推荐系统这样高度稀疏的问题中也能估计交互。

**神经推荐系统。** 神经协同过滤（NCF）[20]使用神经网络架构来建模用户和item的潜在特征。NFM[19]无缝结合了FM在建模二阶特征交互时的线性优势和神经网络在建模高阶特征交互时的非线性优势。DeepFM[14]设计了一个端到端的学习模型，同时强调低阶和高阶特征交互用于CTR预测。xDeepFM[33]扩展了DeepFM，可以显式地学习特定的有界度特征交互。深度矩阵分解（DMF）[55]使用深度结构学习架构，基于显式评分和非偏好隐式反馈，为用户和item的表示学习一个共同的低维空间。DCN[53]保持了深度模型的优势，并引入了一种新颖的交叉网络，在学习特定的有界度特征交互方面更加高效。CMN[12]使用深度架构统一了两类协同过滤模型，利用潜在因子模型的全局结构和基于邻域的局部结构的优势以非线性方式进行建模。

**序列推荐。** 序列推荐是推荐系统中的关键问题。许多最近的推荐系统研究都聚焦于此问题。FPMC[45]将通用马尔可夫链和普通矩阵分解模型结合起来用于序列购物篮数据。HRM[52]扩展了FPMC模型，采用两层结构从最后一次交易中构建用户和item的混合表示。GRU4Rec[21]首先引入基于RNN的方法来建模整个会话以实现更准确的推荐。DREAM[57]基于循环神经网络（RNN）学习用户的动态表示以揭示用户的动态兴趣。Fossil[17]将基于相似度的方法与马尔可夫链平滑地集成，在稀疏和长尾数据集上进行个性化的序列预测。TransRec[16]将item嵌入到向量空间中，其中用户被建模为在item序列上操作的向量，用于大规模序列预测。RUM[7]使用记忆增强神经网络，结合协同过滤的洞见进行推荐。SASRec[27]使用基于自注意力的序列模型来捕获长期语义，并使用注意力机制基于相对较少的动作进行预测。DIN[60]设计了一个局部激活单元，根据特定广告从过去的行为中自适应地学习用户兴趣的表示。SDM[36]使用多头自注意力模块编码行为序列以捕获多种类型的兴趣，并使用长短期门控融合模块来整合长期偏好。

**推荐多样性。** 研究者们已经意识到，仅遵循最准确的推荐方式可能不会产生最佳的推荐结果，因为最高准确率的结果倾向于向用户推荐相似的item，产生令人厌烦的推荐结果[41]。为解决这类问题，推荐item的多样性也起着重要作用[49]。在多样性方面，存在聚合多样性[1]，它指向用户推荐"长尾item"的能力。许多研究专注于提高推荐系统的聚合多样性[1, 2, 40, 43]。其他工作关注推荐给单个用户的item的多样性，即个体多样性[1, 11, 26, 58]，它指向单个用户推荐的item之间的不相似性。

**注意力机制。** 注意力机制的起源可以追溯到几十年前的计算机视觉领域[5, 50]。然而，它在机器学习各个领域中的流行仅在近年来出现。它首先被[3]引入机器翻译，后来作为tensor2tensor[51]成为一种爆发式的方法。BERT[10]利用tensor2tensor并在自然语言处理中取得了巨大成功。注意力机制也被应用于推荐系统[6, 59]，并在真实世界的推荐任务中非常有用。

**胶囊网络。** "胶囊"的概念最初由[22]提出，自从动态路由方法[46]被提出后变得广为人知。MIND[31]将胶囊引入推荐领域，使用基于动态路由机制的胶囊网络来捕获电商用户的多个兴趣，这适用于聚类过去的行为和提取多样化的兴趣。CARP[32]首先从用户和item的评论文档中提取观点和方面，并基于其组成的观点和方面推导每个逻辑单元的表示用于评分预测。

## 3 方法

本节将形式化问题并详细介绍所提出的框架，以及展示我们的框架与现有代表性方法之间的差异。

### 3.1 问题形式化

假设我们有一组用户 $u \in \mathcal{U}$ 和一组item $i \in \mathcal{I}$。对于每个用户，我们有一个按时间排序的用户历史行为序列 $(e_1^{(u)}, e_2^{(u)}, \cdots, e_n^{(u)})$。$e_t^{(u)}$ 记录了用户交互的第 $t$ 个item。给定历史交互，序列推荐的问题是预测用户接下来可能交互的item。符号总结在表1中。

**表1：符号说明**

| 符号 | 描述 |
|------|------|
| $u$ | 一个用户 |
| $i$ | 一个item |
| $e$ | 一次交互 |
| $\mathcal{U}$ | 用户集合 |
| $\mathcal{I}$ | item集合 |
| $\mathcal{I}_u$ | 用户 $u$ 的测试item集合 |
| $d$ | 用户/item嵌入的维度 |
| $K$ | 兴趣嵌入的数量 |
| $N$ | 候选item的数量 |
| $V_u$ | 用户 $u$ 的兴趣嵌入矩阵 |
| $\delta(\cdot)$ | 指示函数 |

在实践中，由于对延迟和性能的严格要求，工业推荐系统通常由两个阶段组成：召回阶段和排序阶段。召回阶段对应于检索top-N候选item，而排序阶段用于通过更精确的分数对候选item进行排序。我们的论文主要关注提高召回阶段的有效性。在本节的后续部分，我们将介绍我们的可控多兴趣框架，并说明该框架对序列推荐问题的重要意义。

### 3.2 多兴趣框架

由于工业推荐系统的item池通常包含数百万甚至数十亿个item，召回阶段在推荐系统中起着至关重要的作用。具体来说，召回模型首先从用户历史行为中计算用户嵌入，然后为每个用户检索一组候选item。借助快速K近邻（KNN）算法从大规模item池中选择最接近的item来为每个用户生成候选集，我们主要关注用户嵌入的计算。换句话说，召回阶段的决定性因素是从用户历史行为中计算的用户嵌入的质量。

现有的召回模型通常使用RNN[21, 54]来计算用户嵌入，但大多数只为每个用户生成一个嵌入向量。这受到单一嵌入缺乏表达力的困扰，因为现实世界的客户通常心中有几种不同的item，这些item通常用于不同用途且类别差异很大。这种现实世界客户的行为突显了使用多个向量表示其多个兴趣的必要性。基于这些观察，我们为序列推荐提出了一个多兴趣框架。我们框架的输入是一个用户行为序列，其中包含表示用户按时间顺序与item交互的item ID列表。这些item ID被输入嵌入层并转换为item嵌入。多兴趣提取模块接收item嵌入并为每个用户生成多个兴趣。

为了构建多兴趣提取模块，有许多可选方法。在本文中，我们探索了两种方法：动态路由方法和自注意力方法，作为我们的多兴趣提取模块。使用动态路由方法的框架称为ComiRec-DR，使用自注意力方法的框架称为ComiRec-SA。

**动态路由。** 我们利用动态路由方法作为用户行为序列的多兴趣提取模块。用户序列的item嵌入可被视为初级胶囊，用户的多个兴趣可被视为兴趣胶囊。我们使用来自CapsNet[46]的动态路由方法。我们简要介绍用于计算胶囊的向量输入和输出的动态路由。胶囊是一组神经元，其活动向量表示特定类型实体（如物体或物体部件）的实例化参数[46]。胶囊输出向量的长度表示该胶囊所代表的实体存在于当前输入中的概率。设 $e_i$ 为初级层的胶囊 $i$。我们首先计算预测向量：

$$
\hat{e}_{j|i} = W_{ij} e_i, \qquad (1)
$$

其中 $W_{ij}$ 是变换矩阵。然后胶囊 $j$ 的总输入是所有预测向量 $\hat{e}_{j|i}$ 的加权和：

$$
s_j = \sum_i c_{ij} \hat{e}_{j|i}, \qquad (2)
$$

其中 $c_{ij}$ 是由迭代动态路由过程确定的耦合系数。胶囊 $i$ 与下一层所有胶囊之间的耦合系数之和应为1。我们使用"路由softmax"和初始logit $b_{ij}$ 来计算耦合系数：

$$
c_{ij} = \frac{\exp(b_{ij})}{\sum_k \exp(b_{ik})}, \qquad (3)
$$

其中 $b_{ij}$ 表示胶囊 $i$ 应与胶囊 $j$ 耦合的对数先验概率。一个非线性"压缩"函数[46]被提出，以确保短向量被压缩到几乎零长度，长向量被压缩到略低于1的长度。然后胶囊 $j$ 的向量计算为：

$$
v_j = \text{squash}(s_j) = \frac{\|s_j\|^2}{1 + \|s_j\|^2} \frac{s_j}{\|s_j\|}, \qquad (4)
$$

其中 $s_j$ 是胶囊 $j$ 的总输入。为了计算输出胶囊 $v_j$，我们需要基于 $v_j$ 和 $e_i$ 的内积计算概率分布。$v_j$ 的计算依赖于自身；因此提出了动态路由方法来解决这个问题。整个动态路由过程如算法1所示。

用户 $u$ 的输出兴趣胶囊随后形成矩阵 $V_u = [v_1, \ldots, v_K] \in \mathbb{R}^{d \times K}$ 用于下游任务。

**算法1：动态路由**

$$
\begin{aligned}
&\textbf{输入:} \text{ 初级胶囊 } e_i \text{，迭代次数 } r \text{，兴趣胶囊数量 } K \\
&\textbf{输出:} \text{ 兴趣胶囊 } \{v_j, j = 1, \ldots, K\} \\
&1: \textbf{for } \text{每个初级胶囊 } i \text{ 和兴趣胶囊 } j \text{：初始化 } b_{ij} = 0 \\
&2: \textbf{for } \text{iter} = 1, \cdots, r \textbf{ do} \\
&\quad 3: \textbf{for } \text{每个初级胶囊 } i \text{：} c_i = \text{softmax}(b_i) \\
&\quad 4: \textbf{for } \text{每个兴趣胶囊 } j \text{：} s_j = \sum_i c_{ij} W_{ij} e_i \\
&\quad 5: \textbf{for } \text{每个兴趣胶囊 } j \text{：} v_j = \text{squash}(s_j) \\
&\quad 6: \textbf{for } \text{每个初级胶囊 } i \text{ 和兴趣胶囊 } j \text{：} b_{ij} = b_{ij} + v_j^\top W_{ij} e_i \\
&7: \textbf{return } \{v_j, j = 1, \ldots, K\}
\end{aligned}
$$

**自注意力方法。** 自注意力方法[35]也可以应用于我们的多兴趣提取模块。给定用户行为的嵌入 $H \in \mathbb{R}^{d \times n}$，其中 $n$ 是用户序列的长度，我们使用自注意力机制来获得权重向量 $a \in \mathbb{R}^n$：

$$
a = \text{softmax}(w_2^\top \tanh(W_1 H))^\top, \qquad (5)
$$

其中 $w_2$ 和 $W_1$ 是可训练参数，大小分别为 $d_a$ 和 $d_a \times d$。上标 $\top$ 表示向量或矩阵的转置。大小为 $n$ 的向量 $a$ 表示用户行为的注意力权重。当我们根据注意力权重对用户行为的嵌入求和时，可以得到用户的一个向量表示 $v_u = Ha$。为了使自注意力方法利用用户序列的顺序，我们将可训练的位置嵌入[51]添加到输入嵌入中。位置嵌入与item嵌入具有相同的维度 $d$，两者可以直接相加。

这个向量表示专注于并反映用户 $u$ 的特定兴趣。为了表示用户的整体兴趣，我们需要从用户行为中获得多个关注不同兴趣的 $v_u$。因此我们需要执行多次注意力。我们将 $w_2$ 扩展为 $d_a \times K$ 的矩阵 $W_2$。然后注意力向量 $a$ 变成注意力矩阵 $A$：

$$
A = \text{softmax}(W_2^\top \tanh(W_1 H))^\top. \qquad (6)
$$

用户兴趣的最终矩阵 $V_u$ 可以计算为：

$$
V_u = HA. \qquad (7)
$$

**模型训练。** 通过多兴趣提取模块从用户行为中计算出兴趣嵌入后，我们使用argmax算子为目标item $i$ 选择对应的用户嵌入向量：

$$
v_u = V_u[:, \arg\max(V_u^\top e_i)], \qquad (8)
$$

其中 $e_i$ 表示目标item $i$ 的嵌入，$V_u$ 是由用户兴趣嵌入组成的矩阵。

给定训练样本 $(u, i)$，用户嵌入为 $v_u$，item嵌入为 $e_i$，我们可以计算用户 $u$ 与item $i$ 交互的概率：

$$
P_\theta(i|u) = \frac{\exp(v_u^\top e_i)}{\sum_{k \in \mathcal{I}} \exp(v_u^\top e_k)}. \qquad (9)
$$

我们模型的目标函数是最小化以下负对数似然：

$$
\text{loss} = \sum_{u \in \mathcal{U}} \sum_{i \in \mathcal{I}_u} -\log P_\theta(i|u). \qquad (10)
$$

公式(9)中的求和算子计算代价高昂；因此，我们使用采样softmax技术[9, 24]来训练我们的模型。

**在线服务。** 对于在线服务，我们使用多兴趣提取模块为每个用户计算多个兴趣。用户的每个兴趣向量可以通过最近邻库（如Faiss[25]）独立地从大规模item池中检索top-N个item。由多个兴趣检索的item被输入聚合模块以确定最终的候选item。最后，具有较高排序分数的item将被推荐给用户。

### 3.3 聚合模块

在多兴趣提取模块之后，我们基于用户的历史行为获得每个用户的多个兴趣嵌入。每个兴趣嵌入可以基于内积相似度独立检索top-N个item。但如何将来自不同兴趣的item聚合以获得最终的top-N个item？一种基本且直接的方法是基于item与用户兴趣的内积相似度进行合并和筛选，可以形式化为：

$$
f(u, i) = \max_{1 \leq k \leq K} (e_i^\top v_u^{(k)}), \qquad (11)
$$

其中 $v_u^{(k)}$ 是用户 $u$ 的第 $k$ 个兴趣嵌入。这是一种有效的聚合方法，可以最大化推荐准确性。然而，当前推荐系统不仅仅关乎准确性。人们更希望被推荐一些新的或多样化的东西。该问题可以形式化如下。给定从用户 $u$ 的 $K$ 个兴趣中检索的包含 $K \cdot N$ 个item的集合 $\mathcal{M}$，找到包含 $N$ 个item的集合 $\mathcal{S}$，使得预定义的价值函数最大化。我们的框架使用可控程序来解决这个问题。我们使用以下价值函数 $Q(u, \mathcal{S})$，通过可控因子 $\lambda \geq 0$ 来平衡推荐的准确性和多样性：

$$
Q(u, \mathcal{S}) = \sum_{i \in \mathcal{S}} f(u, i) + \lambda \sum_{i \in \mathcal{S}} \sum_{j \in \mathcal{S}} g(i, j). \qquad (12)
$$

这里 $g(i, j)$ 是多样性或不相似性函数，例如：

$$
g(i, j) = \delta(\text{CATE}(i) \neq \text{CATE}(j)), \qquad (13)
$$

其中 $\text{CATE}(i)$ 表示item $i$ 的类别，$\delta(\cdot)$ 是指示函数。对于最准确的情况，即 $\lambda = 0$，我们仅使用上述直接方法获得最终item。对于最多样的情况，即 $\lambda = \infty$，可控模块为用户找到最多样的item。我们在第4.3节中研究了可控因子。我们提出了一种贪心推理算法来近似最大化价值函数 $Q(u, \mathcal{S})$，如算法2所示。

**算法2：贪心推理**

$$
\begin{aligned}
&\textbf{输入:} \text{ 候选item集合 } \mathcal{M} \text{，输出item数量 } N \\
&\textbf{输出:} \text{ 输出item集合 } \mathcal{S} \\
&1: \mathcal{S} = \emptyset \\
&2: \textbf{for } \text{iter} = 1, \cdots, N \textbf{ do} \\
&\quad 3: j = \arg\max_{i \in \mathcal{M} \setminus \mathcal{S}} \left( f(u, i) + \lambda \sum_{k \in \mathcal{S}} g(i, k) \right) \\
&\quad 4: \mathcal{S} = \mathcal{S} \cup \{j\} \\
&5: \textbf{return } \mathcal{S}
\end{aligned}
$$

### 3.4 与现有模型的关系

我们将我们的模型与现有模型进行比较。

**MIMN。** MIMN[42]是推荐排序阶段的最新代表性工作，使用记忆网络从长序列行为数据中捕获用户兴趣。MIMN和我们的模型都针对用户的多个兴趣。对于非常长的序列行为，基于记忆的架构可能也不足以捕获用户的长期兴趣。与MIMN相比，我们的模型利用多兴趣提取模块来利用用户的多个兴趣，而不是使用带有记忆利用率正则化和记忆归纳单元的复杂记忆网络。

**MIND。** MIND[31]是推荐召回阶段的最新代表性工作，提出了一种行为到兴趣（B2I）的动态路由，用于自适应地将用户行为聚合为兴趣表示向量。与MIND相比，ComiRec-DR遵循CapsNet[46]使用的原始动态路由方法，可以捕获用户行为的序列信息。我们的框架还探索了用于多兴趣提取的自注意力方法。此外，我们的框架利用可控聚合模块来基于用户的多个兴趣平衡推荐的准确性和多样性。

## 4 实验

本节中，我们在序列推荐上进行实验，以评估我们框架的性能，并与其他最先进方法进行比较。此外，我们还报告了我们框架在十亿级工业数据集上的实验结果。

### 4.1 实验设置

我们在强泛化[34, 37, 38]下评估所有方法的性能：我们按8:1:1的比例将所有用户划分为训练集/验证集/测试集。我们使用训练用户的完整点击序列来训练模型。为了评估，我们取验证和测试用户的前80%用户行为来从训练好的模型中推断用户嵌入，并通过预测剩余20%的用户行为来计算指标。这个设置比弱泛化更困难，在弱泛化中，用户的行为序列在训练和评估过程中都被使用[34]。具体来说，我们采用序列推荐模型训练的通用设置。设用户 $u$ 的行为序列为 $(e_1^{(u)}, e_2^{(u)}, \ldots, e_k^{(u)}, \ldots, e_n^{(u)})$。每个训练样本使用 $u$ 的前 $k$ 个行为来预测第 $(k+1)$ 个行为，其中 $k = 1, 2, \ldots, (n-1)$。

**数据集。** 我们在两个具有挑战性的公开数据集上进行实验。两个数据集的统计信息如表2所示。

**表2：数据集统计信息**

| 数据集 | 用户数 | item数 | 交互数 |
|--------|--------|--------|--------|
| Amazon Books | 459,133 | 313,966 | 8,898,041 |
| Taobao | 976,779 | 1,708,530 | 85,384,110 |

- **Amazon** 包含来自Amazon的产品评论和元数据[18, 39]。在我们的实验中，我们使用Amazon数据集的Books类别。每个训练样本截断长度为20。
- **Taobao** 从淘宝推荐系统中收集用户行为[61]。在我们的实验中，我们仅使用点击行为，并按时间对同一用户的行为进行排序。每个训练样本截断长度为50。

**对比方法。** 我们将提出的模型ComiRec-SA和ComiRec-DR与最先进的模型进行比较。在我们的实验设置中，模型应对验证集和测试集中的未见用户给出预测。因此，基于分解的方法不适合此设置。

- **MostPopular** 是一种传统的推荐方法，向用户推荐最流行的item。
- **YouTube DNN** [9] 是工业推荐系统中最成功的深度学习模型之一。
- **GRU4Rec** [21] 是第一个引入循环神经网络进行推荐的工作。
- **MIND** [31] 是与我们模型相关的最先进模型。它设计了一个基于胶囊路由机制的多兴趣提取层，适用于聚类过去的行为和提取多样化的兴趣。

**实现说明。** 我们实验使用的代码使用Python 3.6中的TensorFlow 1.14实现。

**参数配置。** 嵌入维度 $d$ 设置为64。采样softmax损失的样本数设置为10。最大训练迭代次数设置为100万。多兴趣模型的兴趣嵌入数设置为4。我们使用Adam优化器[29]，学习率 $lr = 0.001$ 进行优化。

**评估指标。** 我们使用以下指标来评估所提出模型的性能。我们在实验中使用三个常用的评估标准。

- **Recall。** 我们采用逐用户平均而不是全局平均以获得更好的可解释性[7, 28]。

$$
\text{Recall@N} = \frac{1}{|\mathcal{U}|} \sum_{u \in \mathcal{U}} \frac{|\hat{\mathcal{I}}_{u,N} \cap \mathcal{I}_u|}{|\mathcal{I}_u|}, \qquad (14)
$$

其中 $\hat{\mathcal{I}}_{u,N}$ 表示为用户 $u$ 推荐的top-N个item的集合，$\mathcal{I}_u$ 是用户 $u$ 的测试item集合。

- **Hit Rate。** 命中率（HR）衡量推荐item中包含至少一个用户交互过的正确item的百分比，在之前的工作中被广泛使用[7, 28]。

$$
\text{HR@N} = \frac{1}{|\mathcal{U}|} \sum_{u \in \mathcal{U}} \delta(|\hat{\mathcal{I}}_{u,N} \cap \mathcal{I}_u| > 0), \qquad (15)
$$

其中 $\delta(\cdot)$ 是指示函数。

- **归一化折损累积增益（NDCG）。** NDCG考虑了正确推荐item的位置[23]。

$$
\text{NDCG@N} = \frac{1}{Z} \text{DCG@N} = \frac{1}{Z} \frac{1}{|\mathcal{U}|} \sum_{u \in \mathcal{U}} \sum_{k=1}^{N} \frac{\delta(\hat{i}_{u,k} \in \mathcal{I}_u)}{\log_2(k+1)}, \qquad (16)
$$

其中 $\hat{i}_{u,k}$ 表示为用户 $u$ 推荐的第 $k$ 个item，$Z$ 是归一化常数，表示理想的折损累积增益（IDCG@N），即DCG@N的最大可能值。

### 4.2 定量结果

为了与其他模型进行公平比较，我们在聚合模块中设置 $\lambda = 0$。我们详细说明了框架中检索top-N个item的过程。对于我们的框架，用户的每个兴趣独立检索top-N个候选item。因此，我们的模型为每个用户总共检索 $K \cdot N$ 个item。我们按item嵌入与相应兴趣嵌入的内积对item进行排序。排序后，从这 $K \cdot N$ 个item中选出的top-N个item被视为我们模型的最终候选item。检索候选item的方式也应用于MIND。序列推荐的模型性能如表3所示。

**表3：公开数据集上的模型性能。加粗数字为每列最佳性能。表中所有数字均为百分比数，省略了'%'。**

| 方法 | Amazon Books | | | Taobao | | |
|------|-------------|---|---|--------|---|---|
| | Recall@20 | NDCG@20 | HR@20 | Recall@20 | NDCG@20 | HR@20 |
| MostPopular | 1.368 | 2.259 | 3.020 | 0.395 | 2.065 | 5.424 |
| YouTube DNN | 4.567 | 7.670 | 10.285 | 4.205 | 14.511 | 28.785 |
| GRU4Rec | 4.057 | 6.803 | 8.945 | 5.884 | 22.095 | 35.745 |
| MIND | 4.862 | 7.933 | 10.618 | 6.281 | 20.394 | 38.119 |
| ComiRec-SA | **5.489** | **8.991** | **11.402** | **6.900** | **24.682** | **41.549** |
| ComiRec-DR | 5.311 | 9.185 | 12.005 | 6.890 | 24.007 | 41.746 |

我们的模型在所有评估标准上都大幅超越了所有最先进模型。GRU4Rec在仅输出单一嵌入的模型中获得了最佳性能。与MIND相比，ComiRec-DR由于动态路由方法的差异获得了更好的性能。ComiRec-SA展示了通过自注意力机制捕获用户兴趣的强大能力，并获得了与ComiRec-DR相当的结果。

**表4：参数敏感性的模型性能。所有数字均为百分比数，省略了'%'。**

| 方法 | Amazon Books Recall@50 | Amazon Books NDCG@50 | Taobao Recall@50 | Taobao NDCG@50 |
|------|----------------------|---------------------|-----------------|---------------|
| ComiRec-SA (K=2) | 8.835 | 14.273 | 9.935 | 32.873 |
| ComiRec-SA (K=4) | 8.467 | 13.563 | 9.462 | 31.278 |
| ComiRec-SA (K=6) | 8.901 | 14.167 | 9.378 | 31.020 |
| ComiRec-SA (K=8) | 8.547 | 13.631 | 9.493 | 31.196 |
| ComiRec-DR (K=2) | 7.081 | 12.068 | 9.293 | 30.735 |
| ComiRec-DR (K=4) | 8.106 | 13.520 | 9.818 | 31.365 |
| ComiRec-DR (K=6) | 7.904 | 13.219 | 10.836 | 34.048 |
| ComiRec-DR (K=8) | 7.760 | 12.900 | 10.841 | 33.895 |

**参数敏感性。** 我们研究了框架中兴趣数量 $K$ 的敏感性。表4展示了当超参数 $K$ 变化时框架的性能。我们的两个模型对该超参数表现出不同的特性。对于Amazon数据集，ComiRec-SA在 $K = 2, 6$ 时获得更好的性能，ComiRec-DR在 $K = 4$ 时获得最佳结果。对于淘宝数据集，ComiRec-DR在 $K$ 从2增加到8时性能更好，但ComiRec-SA在 $K = 2$ 时获得最佳结果。

### 4.3 可控性研究

为了获得每个用户的最终top-N候选item，我们提出了一种新颖的模块来聚合由每个用户的不同兴趣检索的item。除了旨在实现高预测准确性之外，一些研究建议需要多样化的推荐以避免单调性并改善客户体验[8, 13]。

推荐多样性在当前推荐系统中扮演着更重要的角色。许多研究致力于提高推荐多样性[4, 43]。我们提出的聚合模块可以控制推荐准确性和多样性之间的平衡。我们使用以下基于item类别的个体多样性定义：

$$
\text{Diversity@N} = \frac{\sum_{j=1}^{N} \sum_{k=j+1}^{N} \delta(\text{CATE}(\hat{i}_{u,j}) \neq \text{CATE}(\hat{i}_{u,k}))}{N \times (N-1)/2}, \qquad (17)
$$

其中 $\text{CATE}(i)$ 是item $i$ 的类别，$\hat{i}_{u,j}$ 表示为用户 $u$ 推荐的第 $j$ 个item，$\delta(\cdot)$ 是指示函数。

**表5：Amazon数据集可控性研究的模型性能。所有数字均为百分比数，省略了'%'。**

| | ComiRec-SA (K=4) | | ComiRec-DR (K=4) | |
|---|---|---|---|---|
| $\lambda$ | Recall@50 | Diversity@50 | Recall@50 | Diversity@50 |
| 0.00 | 8.467 | 23.237 | 8.106 | 19.036 |
| 0.05 | 8.347 | 38.808 | 7.931 | 42.915 |
| 0.10 | 8.229 | 46.731 | 7.850 | 46.258 |
| 0.15 | 8.142 | 51.135 | 7.820 | 46.912 |
| 0.20 | 8.086 | 53.671 | 7.783 | 47.581 |
| 0.25 | 8.034 | 55.100 | 7.764 | 48.375 |

表5展示了当我们控制因子 $\lambda$ 来平衡推荐质量和多样性时Amazon数据集的模型性能。从表中可以看出，当可控因子 $\lambda$ 增加时，推荐多样性显著增加，而Recall略有下降。我们的聚合模块可以通过为超参数 $\lambda$ 选择适当的值来实现准确性和多样性之间的最优权衡。

### 4.4 工业结果

我们在2020年2月8日由手机淘宝App收集的工业数据集上进一步实验。工业数据集的统计信息如表6所示。工业数据集包含2200万高质量item、1.45亿用户和它们之间的40亿行为。

**表6：工业数据集统计信息**

| 数据集 | 用户数 | item数 | 交互数 |
|--------|--------|--------|--------|
| Industrial | 145,606,322 | 22,554,170 | 4,322,505,616 |

我们的框架已部署在阿里巴巴分布式云平台上，每两个worker共享一个具有16GB内存的NVIDIA Tesla P100 GPU。我们划分用户并使用训练用户的点击序列来训练我们的模型。为了评估，我们使用我们的模型为测试集中的每个用户计算多个兴趣。用户的每个兴趣向量通过快速最近邻方法独立地从大规模item池中检索top-N个item。由不同用户兴趣检索的item被输入我们的聚合模块。在此模块之后，从 $K \cdot N$ 个item中选出的top-N个item是最终的候选item，用于计算评估指标Recall@50。

我们在工业数据集上将我们的框架与最先进的序列推荐方法MIND[31]进行了离线实验，MIND在阿里巴巴集团的推荐系统中已显示出显著改进。实验结果表明，我们的ComiRec-SA和ComiRec-DR相比MIND分别将Recall@50提升了1.39%和8.65%。

**案例研究。** 从图3中，我们可以看到我们的模型从用户的点击序列中学习了四个不同的兴趣。值得注意的是，我们的模型仅使用item ID进行训练，没有使用人工定义的item类别信息。尽管如此，我们的模型仍然能够从用户行为序列中学习item类别。我们模型学习的每个兴趣大致对应一个特定类别，并且能够从大规模工业item池中检索同一类别的相似item。

## 5 结论

在本文中，我们为序列推荐提出了一种新颖的可控多兴趣框架。我们的框架使用多兴趣提取模块生成多个用户兴趣，并使用聚合模块获得最终的top-N个item。实验结果表明，我们的模型在两个具有挑战性的数据集上可以显著超越最先进模型。我们的框架也已成功部署在阿里巴巴分布式云平台上。十亿级工业数据集的结果进一步证实了我们框架在实践中的有效性和效率。推荐系统因深度学习的快速发展而进入了一个新阶段。传统的推荐方法无法满足行业的需求。未来，我们计划利用记忆网络来捕获用户不断演变的兴趣，并引入认知理论来进行更好的用户建模。

## 致谢

本工作得到了国家杰出青年科学基金（61825602）、国家自然科学基金（61836013）以及阿里巴巴集团研究基金的支持。

## 参考文献

[1] Gediminas Adomavicius and YoungOk Kwon. 2011. Improving aggregate recommendation diversity using ranking-based techniques. TKDE 24, 5 (2011), 896–911.

[2] Sujoy Bag, Abhijeet Ghadge, and Manoj Kumar Tiwari. 2019. An integrated recommender system for improved accuracy and aggregate diversity. Computers & Industrial Engineering 130 (2019), 187–197.

[3] Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. 2014. Neural machine translation by jointly learning to align and translate. arXiv preprint arXiv:1409.0473 (2014).

[4] Keith Bradley and Barry Smyth. 2001. Improving recommendation diversity. In AICS'01. Citeseer, 85–94.

[5] Peter J Burt. 1988. Attention mechanisms for vision in a dynamic world. In ICPR'88. IEEE, 977–987.

[6] Yukuo Cen, Xu Zou, Jianwei Zhang, Hongxia Yang, Jingren Zhou, and Jie Tang. 2019. Representation learning for attributed multiplex heterogeneous network. In KDD'19. 1358–1368.

[7] Xu Chen, Hongteng Xu, Yongfeng Zhang, Jiaxi Tang, Yixin Cao, Zheng Qin, and Hongyuan Zha. 2018. Sequential recommendation with user memory networks. In WSDM'18. ACM, 108–116.

[8] Peizhe Cheng, Shuaiqiang Wang, Jun Ma, Jiankai Sun, and Hui Xiong. 2017. Learning to recommend accurate and diverse items. In WWW'17. 183–192.

[9] Paul Covington, Jay Adams, and Emre Sargin. 2016. Deep neural networks for youtube recommendations. In RecSys'16. ACM, 191–198.

[10] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2018. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805 (2018).

[11] Tommaso Di Noia, Vito Claudio Ostuni, Jessica Rosati, Paolo Tomeo, and Eugenio Di Sciascio. 2014. An analysis of users' propensity toward diversity in recommendations. In RecSys'14. 285–288.

[12] Travis Ebesu, Bin Shen, and Yi Fang. 2018. Collaborative memory network for recommendation systems. In SIGIR'18. ACM, 515–524.

[13] Anupriya Gogna and Angshul Majumdar. 2017. Balancing accuracy and diversity in recommendations using matrix completion framework. Knowledge-Based Systems 125 (2017), 83–95.

[14] Huifeng Guo, Ruiming Tang, Yunming Ye, Zhenguo Li, and Xiuqiang He. 2017. DeepFM: a factorization-machine based neural network for CTR prediction. In IJCAI'17.

[15] Will Hamilton, Zhitao Ying, and Jure Leskovec. 2017. Inductive representation learning on large graphs. In NIPS'17. 1024–1034.

[16] Ruining He, Wang-Cheng Kang, and Julian McAuley. 2017. Translation-based recommendation. In RecSys'17. ACM, 161–169.

[17] Ruining He and Julian McAuley. 2016. Fusing similarity models with markov chains for sparse sequential recommendation. In ICDM'16. IEEE, 191–200.

[18] Ruining He and Julian McAuley. 2016. Ups and downs: Modeling the visual evolution of fashion trends with one-class collaborative filtering. In WWW'16. International World Wide Web Conferences Steering Committee, 507–517.

[19] Xiangnan He and Tat-Seng Chua. 2017. Neural factorization machines for sparse predictive analytics. In SIGIR'17. ACM, 355–364.

[20] Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu, and Tat-Seng Chua. 2017. Neural collaborative filtering. In WWW'17. International World Wide Web Conferences Steering Committee, 173–182.

[21] Balázs Hidasi, Alexandros Karatzoglou, Linas Baltrunas, and Domonkos Tikk. 2016. Session-based recommendations with recurrent neural networks. In ICLR'16.

[22] Geoffrey E Hinton, Alex Krizhevsky, and Sida D Wang. 2011. Transforming auto-encoders. In ICANN'11. Springer, 44–51.

[23] Kalervo Järvelin and Jaana Kekäläinen. 2000. IR evaluation methods for retrieving highly relevant documents. In SIGIR'00. ACM, 41–48.

[24] Sébastien Jean, Kyunghyun Cho, Roland Memisevic, and Yoshua Bengio. 2015. On using very large target vocabulary for neural machine translation. ACL'15.

[25] Jeff Johnson, Matthijs Douze, and Hervé Jégou. 2017. Billion-scale similarity search with GPUs. arXiv preprint arXiv:1702.08734 (2017).

[26] M Kalaivanan and K Vengatesan. 2013. Recommendation system based on statistical analysis of ranking from user. In ICICES'13. IEEE, 479–484.

[27] Wang-Cheng Kang and Julian McAuley. 2018. Self-attentive sequential recommendation. In ICDM'18. IEEE, 197–206.

[28] George Karypis. 2001. Evaluation of item-based top-n recommendation algorithms. In CIKM'01. ACM, 247–254.

[29] Diederik P Kingma and Jimmy Ba. 2014. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980 (2014).

[30] Yehuda Koren, Robert Bell, and Chris Volinsky. 2009. Matrix factorization techniques for recommender systems. Computer 8 (2009), 30–37.

[31] Chao Li, Zhiyuan Liu, Mengmeng Wu, Yuchi Xu, Pipei Huang, Huan Zhao, Guoliang Kang, Qiwei Chen, Wei Li, and Dik Lun Lee. 2019. Multi-Interest Network with Dynamic Routing for Recommendation at Tmall. arXiv preprint arXiv:1904.08030 (2019).

[32] Chenliang Li, Cong Quan, Li Peng, Yunwei Qi, Yuming Deng, and Libing Wu. 2019. A Capsule Network for Recommendation and Explaining What You Like and Dislike. In SIGIR'19. ACM, 275–284.

[33] Jianxun Lian, Xiaohuan Zhou, Fuzheng Zhang, Zhongxia Chen, Xing Xie, and Guangzhong Sun. 2018. xDeepFM: Combining explicit and implicit feature interactions for recommender systems. In KDD'18. ACM, 1754–1763.

[34] Dawen Liang, Rahul G Krishnan, Matthew D Hoffman, and Tony Jebara. 2018. Variational autoencoders for collaborative filtering. In WWW'18. 689–698.

[35] Zhouhan Lin, Minwei Feng, Cicero Nogueira dos Santos, Mo Yu, Bing Xiang, Bowen Zhou, and Yoshua Bengio. 2017. A structured self-attentive sentence embedding. In ICLR'17.

[36] Fuyu Lv, Taiwei Jin, Changlong Yu, Fei Sun, Quan Lin, Keping Yang, and Wilfred Ng. 2019. SDM: Sequential deep matching model for online large-scale recommender system. In CIKM'19. 2635–2643.

[37] Jianxin Ma, Chang Zhou, Peng Cui, Hongxia Yang, and Wenwu Zhu. 2019. Learning disentangled representations for recommendation. In NIPS'19. 5712–5723.

[38] Benjamin Marlin. 2004. Collaborative filtering: A machine learning perspective. University of Toronto Toronto.

[39] Julian McAuley, Christopher Targett, Qinfeng Shi, and Anton Van Den Hengel. 2015. Image-based recommendations on styles and substitutes. In SIGIR'15. ACM, 43–52.

[40] Katja Niemann and Martin Wolpers. 2013. A new collaborative filtering approach for increasing the aggregate diversity of recommender systems. In KDD'13. 955–963.

[41] Umberto Panniello, Alexander Tuzhilin, and Michele Gorgoglione. 2014. Comparing context-aware recommender systems in terms of accuracy and diversity. UMUAI 24, 1-2 (2014), 35–65.

[42] Qi Pi, Weijie Bian, Guorui Zhou, Xiaoqiang Zhu, and Kun Gai. 2019. Practice on long sequential user behavior modeling for click-through rate prediction. In KDD'19. 2671–2679.

[43] Lijing Qin and Xiaoyan Zhu. 2013. Promoting diversity in recommendation by entropy regularizer. In IJCAI'13.

[44] Steffen Rendle. 2010. Factorization machines. In ICDM'10. IEEE, 995–1000.

[45] Steffen Rendle, Christoph Freudenthaler, and Lars Schmidt-Thieme. 2010. Factorizing personalized markov chains for next-basket recommendation. In WWW'10. ACM, 811–820.

[46] Sara Sabour, Nicholas Frosst, and Geoffrey E Hinton. 2017. Dynamic routing between capsules. In NIPS'17. 3856–3866.

[47] Badrul Munir Sarwar, George Karypis, Joseph A Konstan, John Riedl, et al. 2001. Item-based collaborative filtering recommendation algorithms. WWW'01 (2001), 285–295.

[48] J Ben Schafer, Dan Frankowski, Jon Herlocker, and Shilad Sen. 2007. Collaborative filtering recommender systems. In The adaptive web. Springer, 291–324.

[49] Malcolm Slaney and William White. 2006. Measuring playlist diversity for recommendation systems. In AMCMM'06 workshop. 77–82.

[50] Yaoru Sun and Robert Fisher. 2003. Object-based visual attention for computer vision. Artificial intelligence 146, 1 (2003), 77–123.

[51] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In NIPS'17. 5998–6008.

[52] Pengfei Wang, Jiafeng Guo, Yanyan Lan, Jun Xu, Shengxian Wan, and Xueqi Cheng. 2015. Learning hierarchical representation model for nextbasket recommendation. In SIGIR'15. ACM, 403–412.

[53] Ruoxi Wang, Bin Fu, Gang Fu, and Mingliang Wang. 2017. Deep & cross network for ad click predictions. In ADKDD'17. ACM, 12.

[54] Chao-Yuan Wu, Amr Ahmed, Alex Beutel, Alexander J Smola, and How Jing. 2017. Recurrent recommender networks. In WSDM'17. ACM, 495–503.

[55] Hong-Jian Xue, Xinyu Dai, Jianbing Zhang, Shujian Huang, and Jiajun Chen. 2017. Deep Matrix Factorization Models for Recommender Systems.. In IJCAI'17. 3203–3209.

[56] Rex Ying, Ruining He, Kaifeng Chen, Pong Eksombatchai, William L Hamilton, and Jure Leskovec. 2018. Graph convolutional neural networks for web-scale recommender systems. In KDD'18. ACM, 974–983.

[57] Feng Yu, Qiang Liu, Shu Wu, Liang Wang, and Tieniu Tan. 2016. A dynamic recurrent model for next basket recommendation. In SIGIR'16. ACM, 729–732.

[58] Ting Yu, Junpeng Guo, Wenhua Li, Harry Jiannan Wang, and Ling Fan. 2019. Recommendation with diversity: An adaptive trust-aware model. Decision Support Systems 123 (2019), 113073.

[59] Chang Zhou, Jinze Bai, Junshuai Song, Xiaofei Liu, Zhengchao Zhao, Xiusi Chen, and Jun Gao. 2018. Atrank: An attention-based user behavior modeling framework for recommendation. In AAAI'18.

[60] Guorui Zhou, Xiaoqiang Zhu, Chenru Song, Ying Fan, Han Zhu, Xiao Ma, Yanghui Yan, Junqi Jin, Han Li, and Kun Gai. 2018. Deep interest network for click-through rate prediction. In KDD'18. ACM, 1059–1068.

[61] Han Zhu, Xiang Li, Pengye Zhang, Guozheng Li, Jie He, Han Li, and Kun Gai. 2018. Learning tree-based deep model for recommender systems. In KDD'18. ACM, 1079–1088.

---

## 附录

### A.1 实现说明

**运行环境。** 本文的实验可分为两部分。一部分在两个公开数据集上进行，使用一台Linux服务器，配备4个Intel(R) Xeon(R) CPU E5-2680 v4 @ 2.40GHz、256G RAM和8个NVIDIA GeForce RTX 2080 Ti。该部分我们提出的模型的代码使用Python 3.6中的TensorFlow 1.14实现。另一部分在工业数据集上进行，使用阿里巴巴分布式云平台，包含数千个worker。每两个worker共享一个具有16GB内存的NVIDIA Tesla P100 GPU。该部分我们提出的模型使用Python 2.7中的TensorFlow 1.4实现。

**实现细节。** 我们在单台Linux服务器上使用的代码可分为三部分：数据迭代器、模型训练和评估。对于每次训练迭代，数据迭代器选择大小为batch_size的随机训练用户。对于每个选定的用户，我们随机选择其点击序列中的一个item作为训练标签，并使用该item之前的item作为训练序列。训练部分基于TensorFlow 1.x API按照算法3中的训练循环实现。我们的损失函数基于tf.nn.sampled_softmax_loss。评估部分依赖于Faiss，一个用于高效相似性搜索和密集向量聚类的库。我们使用Faiss的GpuIndexFlatIP类，它在GPU上实现了精确的内积搜索。所有模型参数通过随机梯度下降和Adam更新规则[29]进行更新和优化。我们提出的模型的分布式版本基于阿里巴巴分布式云平台的编码规则实现，以最大化分布式效率。

**参数配置。** 用户/item嵌入维度 $d$ 设置为64。采样softmax损失的样本数设置为10。最大训练迭代次数设置为100万，所有模型使用基于验证集Recall@50的早停。Amazon数据集和淘宝数据集的批大小分别设置为128和256。动态路由方法的迭代次数设置为3。多兴趣模型的兴趣嵌入数 $K$ 设置为4以进行公平比较。我们使用Adam优化器[29]，学习率 $lr = 0.001$ 进行优化。

**算法3：ComiRec**

$$
\begin{aligned}
&\textbf{输入:} \text{ 用户行为序列} \\
&1: \text{ 初始化所有模型参数} \\
&2: \text{ 生成训练样本 } \{(u, i)\} \text{，使用用户点击序列} \\
&3: \textbf{while } \text{未收敛} \textbf{ do} \\
&\quad 4: \textbf{for } \text{每个训练样本batch} \textbf{ do} \\
&\quad\quad 5: \text{ 使用多兴趣提取模块计算 } V_u \\
&\quad\quad 6: \text{ 基于公式(8)计算 } v_u \\
&\quad\quad 7: \text{ 使用公式(10)计算采样softmax损失} \\
&\quad\quad 8: \text{ 使用Adam优化器更新模型参数}
\end{aligned}
$$

### A.2 对比方法

以下给出所有对比方法的实现细节。

- **MostPopular** 是一种非个性化方法，向用户推荐最流行的item。该方法不需要训练，我们单独实现它。
- **YouTube DNN** 是工业推荐系统中最成功的深度学习模型之一。我们基于原始论文在代码中实现了该模型。
- **GRU4Rec** 是第一个引入循环神经网络进行推荐的工作。我们通过TensorFlow中的tf.nn.rnn_cell.GRUCell和tf.nn.dynamic_rnn在代码中实现了该模型。
- **MIND** 是最先进模型。我们基于原始论文和阿里巴巴集团的内部代码版本实现了该模型。

### A.3 数据集

我们的实验在三个数据集上进行评估，包括两个公开数据集和一个十亿级工业数据集。对于两个公开数据集，我们保留至少有5次行为的用户和item。

- **Amazon** 包含来自Amazon的产品评论和元数据[18, 39]。在我们的实验中，我们使用Amazon数据集的Books类别。对于每个用户 $u$，我们按时间对用户的评论进行排序，我们的任务是基于之前的评论预测用户是否会为该item撰写评论。每个训练样本截断长度为20。
- **Taobao** 从淘宝推荐系统中收集用户行为[61]。淘宝数据集随机选择约100万用户，这些用户在2017年11月25日至12月3日期间有点击、购买、加入购物车和加入收藏等行为。每次行为由五个字段表示，包括用户ID、item ID、item类别ID、行为类型和时间戳。在我们的实验中，我们仅使用点击行为，并按时间对同一用户的行为进行排序。每个训练样本截断长度为50。
- **工业数据集** 于2020年2月8日由手机淘宝App收集用户点击行为。工业数据集包含2200万高质量item、1.45亿用户和它们之间的40亿行为。每个训练样本截断长度为40。
