# word2vec 参数学习详解

> Xin Rong | 密歇根大学（University of Michigan）



本文详细推导了 word2vec 模型的参数更新方程。核心内容：

- 连续词袋模型（CBOW）：单上下文词与多上下文词两种设置下的网络结构与参数更新推导
- Skip-gram 模型：与 CBOW 相反的网络结构及其参数更新推导
- 层次 softmax：利用二叉树将每次训练的复杂度从 $O(V)$ 降低到 $O(\log V)$
- 负采样：仅更新采样得到的一小部分输出向量，从而大幅节省计算量
- 附录：神经元网络与反向传播的基础知识回顾

关键发现：每个词都有输入向量与输出向量两种表示；输出向量的更新是整个模型的计算瓶颈，层次 softmax 与负采样分别通过树结构和采样技巧绕过这一瓶颈。

---



## 摘要

Mikolov 等人的 word2vec 模型与应用在近两年引起了极大关注。word2vec 模型学习得到的词向量表示已被证明携带语义信息，并在各种自然语言处理（NLP）任务中发挥作用。随着越来越多的研究者希望尝试 word2vec 或类似技术，我注意到目前缺少一份全面、详细解释词嵌入模型参数学习过程的材料，这阻碍了并非神经网络专家的研究者理解此类模型的工作机制。

本笔记详细推导并解释了 word2vec 模型的参数更新方程，包括原始连续词袋模型（continuous bag-of-word，CBOW）与 skip-gram（SG）模型，以及高级优化技巧，包括层次 softmax（hierarchical softmax）与负采样（negative sampling）。在数学推导之外，还给出了梯度方程的直观解释。

在附录中，提供了神经元网络与反向传播基础知识的回顾。我还创建了一个交互式演示 wevi，以帮助直观理解模型。[^1]

## 1 连续词袋模型

### 1.1 单上下文词

我们从 Mikolov 等人 [2] 提出的连续词袋模型（CBOW）的最简版本开始。我们假设每个上下文只考虑一个词，即模型给定一个上下文词，预测一个目标词，这类似于二元语言模型（bigram model）。对于神经网络的新读者，建议在继续之前先通读附录 A，快速回顾重要的概念和术语。

图 1 展示了简化上下文定义下的网络模型。[^2] 在我们的设置中，词表大小是 $V$，隐藏层大小是 $N$。相邻层的单元之间全连接。输入是一个 one-hot 编码向量，即对于给定的输入上下文词，${x_1}, \cdots, {x_V}$ 中只有一个单元为 1，其余单元均为 0。

**图1.** 仅有一个上下文词的简单 CBOW 模型。

输入层与输出层之间的权重可以用一个 $V \times N$ 的矩阵 $W$ 表示。$W$ 的每一行是输入层相关联词的 $N$ 维向量表示 $v_w$。形式化地，$W$ 的第 $i$ 行是 $v_w^T$。给定一个上下文（一个词），假设 $x_k = 1$ 且 $x_{k'} = 0$（$k' \neq k$），我们有

$$
h = W^T x = W_{(k,\cdot)}^T := v_{w_I}^T , \qquad (1)
$$

这本质上是将 $W$ 的第 $k$ 行复制到 $h$。$v_{w_I}$ 是输入词 $w_I$ 的向量表示。这意味着隐藏层单元的连接（激活）函数是简单的线性函数（即直接将其输入的加权和传递给下一层）。

从隐藏层到输出层，存在一个不同的权重矩阵 $W' = \{w'_{ij}\}$，它是一个 $N \times V$ 的矩阵。利用这些权重，我们可以计算词表中每个词的一个得分 $u_j$，

$$
u_j = v'^T_{w_j} h , \qquad (2)
$$

其中 $v'_{w_j}$ 是矩阵 $W'$ 的第 $j$ 列。然后我们可以使用 softmax（一个对数线性分类模型）来获得词的后验分布，这是一个多项分布。

$$
p(w_j|w_I) = y_j = \frac{\exp(u_j)}{\sum_{j'=1}^{V} \exp(u_{j'})} , \qquad (3)
$$

其中 $y_j$ 是输出层第 $j$ 个单元的输出。将 (1) 和 (2) 代入 (3)，我们得到

$$
p(w_j|w_I) = \frac{\exp(v'^T_{w_j} v_{w_I})}{\sum_{j'=1}^{V} \exp(v'^T_{w_{j'}} v_{w_I})} . \qquad (4)
$$

注意，$v_w$ 和 $v'_w$ 是词 $w$ 的两种表示。$v_w$ 来自 $W$ 的行，即输入→隐藏层的权重矩阵；$v'_w$ 来自 $W'$ 的列，即隐藏→输出层的矩阵。在后续分析中，我们称 $v_w$ 为词 $w$ 的“输入向量”（input vector），称 $v'_w$ 为词 $w$ 的“输出向量”（output vector）。

#### 隐藏→输出权重的更新方程

现在让我们推导该模型的权重更新方程。尽管实际计算并不实用（下文将解释），我们进行推导是为了洞悉这个未采用任何技巧的原始模型。反向传播基础知识的回顾见附录 A。

训练目标（对于一个训练样本）是在给定输入上下文词 $w_I$ 的条件下，最大化观察到实际输出词 $w_O$（其在输出层中的索引记为 $j^*$）的条件概率：

$$
\max p(w_O|w_I) = \max y_{j^*} \qquad (5)
$$

$$
= \max \log y_{j^*} \qquad (6)
$$

$$
= u_{j^*} - \log \sum_{j'=1}^{V} \exp(u_{j'}) := -E , \qquad (7)
$$

其中 $E = -\log p(w_O|w_I)$ 是我们的损失函数（我们希望最小化 $E$），$j^*$ 是实际输出词在输出层中的索引。注意，这个损失函数可以理解为两个概率分布之间交叉熵度量的一种特殊情况。

现在让我们推导隐藏层与输出层之间权重的更新方程。取 $E$ 对第 $j$ 个单元的净输入 $u_j$ 的导数，我们得到

$$
\frac{\partial E}{\partial u_j} = y_j - t_j := e_j \qquad (8)
$$

其中 $t_j = \mathbb{1}(j = j^*)$，即仅当第 $j$ 个单元是实际输出词时 $t_j = 1$，否则 $t_j = 0$。注意，这个导数正是输出层的预测误差 $e_j$。

接下来我们对 $w'_{ij}$ 求导，以获得隐藏→输出权重上的梯度。

$$
\frac{\partial E}{\partial w'_{ij}} = \frac{\partial E}{\partial u_j} \cdot \frac{\partial u_j}{\partial w'_{ij}} = e_j \cdot h_i \qquad (9)
$$

因此，使用随机梯度下降（stochastic gradient descent，SGD），我们得到隐藏→输出权重的权重更新方程：

$$
w'^{(new)}_{ij} = w'^{(old)}_{ij} - \eta \cdot e_j \cdot h_i . \qquad (10)
$$

或者

$$
v'^{(new)}_{w_j} = v'^{(old)}_{w_j} - \eta \cdot e_j \cdot h \quad \text{for } j = 1, 2, \cdots, V . \qquad (11)
$$

其中 $\eta > 0$ 是学习率，$e_j = y_j - t_j$，$h_i$ 是隐藏层的第 $i$ 个单元；$v'_{w_j}$ 是 $w_j$ 的输出向量。注意，这个更新方程意味着我们必须遍历词表中的每一个可能的词，检查其输出概率 $y_j$，并将 $y_j$ 与其期望输出 $t_j$（0 或 1）进行比较。如果 $y_j > t_j$（“高估”），则从 $v'_{w_j}$ 中减去隐藏向量 $h$（即 $v_{w_I}$）的一部分，从而使 $v'_{w_j}$ 远离 $v_{w_I}$；如果 $y_j < t_j$（“低估”，仅在 $t_j = 1$ 时成立，即 $w_j = w_O$），则向 $v'_{w_O}$ 加上一些 $h$，从而使 $v'_{w_O}$ 更接近 $v_{w_I}$。[^3] 如果 $y_j$ 非常接近 $t_j$，那么根据更新方程，权重几乎不会发生变化。再次注意，$v_w$（输入向量）和 $v'_w$（输出向量）是词 $w$ 的两个不同的向量表示。

#### 输入→隐藏权重的更新方程

在获得 $W'$ 的更新方程之后，我们现在可以继续处理 $W$。我们对 $E$ 在隐藏层输出上求导，得到

$$
\frac{\partial E}{\partial h_i} = \sum_{j=1}^{V} \frac{\partial E}{\partial u_j} \cdot \frac{\partial u_j}{\partial h_i} = \sum_{j=1}^{V} e_j \cdot w'_{ij} := EH_i \qquad (12)
$$

其中 $h_i$ 是隐藏层第 $i$ 个单元的输出；$u_j$ 定义于 (2)，是输出层第 $j$ 个单元的净输入；$e_j = y_j - t_j$ 是输出层第 $j$ 个词的预测误差。$EH$ 是一个 $N$ 维向量，是词表中所有词的输出向量按其预测误差加权后的和。

接下来我们应该对 $W$ 求导。首先，回顾一下隐藏层对来自输入层的值执行线性计算。展开 (1) 中的向量记号，我们得到

$$
h_i = \sum_{k=1}^{V} x_k \cdot w_{ki} \qquad (13)
$$

现在我们可以对 $W$ 的每个元素求导，得到

$$
\frac{\partial E}{\partial w_{ki}} = \frac{\partial E}{\partial h_i} \cdot \frac{\partial h_i}{\partial w_{ki}} = EH_i \cdot x_k \qquad (14)
$$

这等价于 $x$ 与 $EH$ 的张量积，即

$$
\frac{\partial E}{\partial W} = x \otimes EH = x EH^T \qquad (15)
$$

由此我们得到一个 $V \times N$ 的矩阵。由于 $x$ 只有一个分量为非零，$\frac{\partial E}{\partial W}$ 只有一行非零，且该行的值是 $EH^T$，一个 $N$ 维向量。我们得到 $W$ 的更新方程

$$
v^{(new)}_{w_I} = v^{(old)}_{w_I} - \eta EH^T \qquad (16)
$$

其中 $v_{w_I}$ 是 $W$ 的一行，即唯一上下文词的“输入向量”，也是 $W$ 中导数非零的唯一一行。$W$ 的所有其他行在此次迭代后将保持不变，因为它们的导数为零。

直观地说，由于向量 $EH$ 是词表中所有词的输出向量按预测误差 $e_j = y_j - t_j$ 加权后的和，我们可以将 (16) 理解为向上下文词的输入向量加上词表中每个输出向量的一部分。如果在输出层，词 $w_j$ 作为输出词的概率被高估（$y_j > t_j$），那么上下文词 $w_I$ 的输入向量将倾向于远离 $w_j$ 的输出向量；反之，如果 $w_j$ 作为输出词的概率被低估（$y_j < t_j$），那么输入向量 $w_I$ 将倾向于靠近 $w_j$ 的输出向量；如果 $w_j$ 的概率被相当准确地预测，那么它对 $w_I$ 输入向量的移动影响很小。$w_I$ 输入向量的移动由词表中所有向量的预测误差共同决定；预测误差越大，一个词对上下文词输入向量移动的影响就越显著。

当我们遍历由训练语料库生成的上下文-目标词对来迭代更新模型参数时，对向量的影响会不断累积。我们可以想象，一个词 $w$ 的输出向量被 $w$ 的共现邻居的输入向量来回“拖拽”，就像 $w$ 的向量与其邻居向量之间存在物理的弦一样。类似地，输入向量也可以被认为是被许多输出向量拖拽。这种解释可以让我们联想到引力，或力导向图布局。每条想象中的弦的平衡长度与相关词对的共现强度以及学习率有关。经过多次迭代，输入向量与输出向量的相对位置最终会稳定下来。

### 1.2 多上下文词

图 2 展示了多上下文词设置下的 CBOW 模型。在计算隐藏层输出时，CBOW 模型并非直接复制输入上下文词的输入向量，而是取输入上下文词向量的平均值，并将输入→隐藏权重矩阵与平均向量的乘积作为输出。

$$
h = \frac{1}{C} W^T (x_1 + x_2 + \cdots + x_C) \qquad (17)
$$

$$
= \frac{1}{C} (v_{w_1} + v_{w_2} + \cdots + v_{w_C})^T \qquad (18)
$$

其中 $C$ 是上下文中词的个数，$w_1, \cdots, w_C$ 是上下文中的词，$v_w$ 是词 $w$ 的输入向量。损失函数为

$$
E = -\log p(w_O|w_{I,1}, \cdots, w_{I,C}) \qquad (19)
$$

$$
= -u_{j^*} + \log \sum_{j'=1}^{V} \exp(u_{j'}) \qquad (20)
$$

$$
= -v'^T_{w_O} \cdot h + \log \sum_{j'=1}^{V} \exp(v'^T_{w_j} \cdot h) \qquad (21)
$$

这与 (7)（单上下文词模型的目标）相同，只是 $h$ 不同，如 (18) 所定义而非 (1)。

**图2.** 连续词袋模型（CBOW）。

隐藏→输出权重的更新方程与单上下文词模型（11）相同。我们在此复制它：

$$
v'^{(new)}_{w_j} = v'^{(old)}_{w_j} - \eta \cdot e_j \cdot h \quad \text{for } j = 1, 2, \cdots, V . \qquad (22)
$$

注意，对于每个训练样本，我们需要对隐藏→输出权重矩阵的每个元素应用该更新。

输入→隐藏权重的更新方程与 (16) 类似，只是现在我们需要对上下文中的每个词 $w_{I,c}$ 应用下列方程：

$$
v^{(new)}_{w_{I,c}} = v^{(old)}_{w_{I,c}} - \frac{1}{C} \cdot \eta \cdot EH^T \quad \text{for } c = 1, 2, \cdots, C . \qquad (23)
$$

其中 $v_{w_{I,c}}$ 是输入上下文中第 $c$ 个词的输入向量；$\eta$ 是正的学习率；$EH = \frac{\partial E}{\partial h_i}$ 由 (12) 给出。该更新方程的直观理解与 (16) 相同。

## 2 Skip-gram 模型

skip-gram 模型由 Mikolov 等人 [2, 3] 提出。图 3 展示了 skip-gram 模型。它与 CBOW 模型相反：目标词现在位于输入层，而上下文词位于输出层。

我们仍用 $v_{w_I}$ 表示输入层唯一词的输入向量，因此我们与 (1) 中有相同的隐藏层输出 $h$ 的定义，即 $h$ 只是复制（并转置）输入→隐藏权重矩阵 $W$ 中与输入词 $w_I$ 相关联的一行。我们在下方复制 $h$ 的定义：

$$
h = W^T_{(k,\cdot)} := v^T_{w_I} , \qquad (24)
$$

在输出层，我们输出的不是一种多项分布，而是 $C$ 种多项分布。每次输出都使用同一个隐藏→输出矩阵计算：

$$
p(w_{c,j} = w_{O,c}|w_I) = y_{c,j} = \frac{\exp(u_{c,j})}{\sum_{j'=1}^{V} \exp(u_{j'})} \qquad (25)
$$

其中 $w_{c,j}$ 是输出层第 $c$ 个面板上的第 $j$ 个词；$w_{O,c}$ 是输出上下文词中实际的第 $c$ 个词；$w_I$ 是唯一的输入词；$y_{c,j}$ 是输出层第 $c$ 个面板上第 $j$ 个单元的输出；$u_{c,j}$ 是输出层第 $c$ 个面板上第 $j$ 个单元的净输入。由于输出层面板共享相同的权重，因此

$$
u_{c,j} = u_j = v'^T_{w_j} \cdot h, \quad \text{for } c = 1, 2, \cdots, C \qquad (26)
$$

其中 $v'_{w_j}$ 是词表中第 $j$ 个词 $w_j$ 的输出向量，$v'_{w_j}$ 取自隐藏→输出权重矩阵 $W'$ 的一列。

**图3.** Skip-gram 模型。

参数更新方程的推导与单上下文词模型差别不大。损失函数变为

$$
E = -\log p(w_{O,1}, w_{O,2}, \cdots, w_{O,C}|w_I) \qquad (27)
$$

$$
= -\log \prod_{c=1}^{C} \frac{\exp(u_{c,j^*_c})}{\sum_{j'=1}^{V} \exp(u_{j'})} \qquad (28)
$$

$$
= -\sum_{c=1}^{C} u_{j^*_c} + C \cdot \log \sum_{j'=1}^{V} \exp(u_{j'}) \qquad (29)
$$

其中 $j^*_c$ 是词表中实际第 $c$ 个输出上下文词的索引。

我们取 $E$ 对输出层每个面板上每个单元的净输入 $u_{c,j}$ 的导数，得到

$$
\frac{\partial E}{\partial u_{c,j}} = y_{c,j} - t_{c,j} := e_{c,j} \qquad (30)
$$

这与 (8) 中一样，是单元上的预测误差。为简化记号，我们定义一个 $V$ 维向量 $EI = \{EI_1, \cdots, EI_V\}$，作为所有上下文词预测误差之和：

$$
EI_j = \sum_{c=1}^{C} e_{c,j} \qquad (31)
$$

接下来，我们对隐藏→输出矩阵 $W'$ 求导，得到

$$
\frac{\partial E}{\partial w'_{ij}} = \sum_{c=1}^{C} \frac{\partial E}{\partial u_{c,j}} \cdot \frac{\partial u_{c,j}}{\partial w'_{ij}} = EI_j \cdot h_i \qquad (32)
$$

由此我们得到隐藏→输出矩阵 $W'$ 的更新方程，

$$
w'^{(new)}_{ij} = w'^{(old)}_{ij} - \eta \cdot EI_j \cdot h_i \qquad (33)
$$

或者

$$
v'^{(new)}_{w_j} = v'^{(old)}_{w_j} - \eta \cdot EI_j \cdot h \quad \text{for } j = 1, 2, \cdots, V . \qquad (34)
$$

该更新方程的直观理解与 (11) 相同，只是预测误差在输出层的所有上下文词上求和。注意，对于每个训练样本，我们需要对隐藏→输出矩阵的每个元素应用该更新方程。

输入→隐藏矩阵更新方程的推导与 (12) 至 (16) 完全相同，只是需要考虑将预测误差 $e_j$ 替换为 $EI_j$。我们直接给出更新方程：

$$
v^{(new)}_{w_I} = v^{(old)}_{w_I} - \eta \cdot EH^T \qquad (35)
$$

其中 $EH$ 是一个 $N$ 维向量，其每个分量定义为

$$
EH_i = \sum_{j=1}^{V} EI_j \cdot w'_{ij} . \qquad (36)
$$

(35) 的直观理解与 (16) 相同。

## 3 优化计算效率

到目前为止，我们讨论的模型（“二元”模型、CBOW 和 skip-gram）都处于其原始形式，没有应用任何效率优化技巧。

对于所有这些模型，词表中的每个词都存在两种向量表示：输入向量 $v_w$ 和输出向量 $v'_w$。学习输入向量很便宜；但学习输出向量非常昂贵。从更新方程 (22) 和 (33) 我们可以发现，为了更新 $v'_w$，对于每个训练样本，我们必须遍历词表中的每一个词 $w_j$，计算它们的净输入 $u_j$、概率预测 $y_j$（对于 skip-gram 是 $y_{c,j}$）、它们的预测误差 $e_j$（对于 skip-gram 是 $EI_j$），最后用它们的预测误差来更新它们的输出向量 $v'_j$。

对每个训练样本对所有词进行这样的计算非常昂贵，使得其难以扩展到大规模词表或大规模训练语料库。为了解决这个问题，一个直觉是限制每个训练样本必须更新的输出向量的数量。一个实现这一点的优雅方法是层次 softmax；另一种方法是通过采样，将在下一节讨论。

这两种技巧都只优化输出向量更新的计算。在我们的推导中，我们关心三个值：(1) $E$，新的目标函数；(2) $\frac{\partial E}{\partial v'_w}$，输出向量的新更新方程；(3) $\frac{\partial E}{\partial h}$，用于更新输入向量的预测误差加权和。

### 3.1 层次 Softmax

层次 softmax 是一种高效计算 softmax 的方法 [5, 4]（Morin and Bengio, 2005; Mnih and Hinton, 2009）。该模型使用一棵二叉树来表示词表中的所有词。$V$ 个词必须是树的叶子单元。可以证明存在 $V - 1$ 个内部单元。对于每个叶子单元，从根到该单元存在一条唯一路径；这条路径被用来估计该叶子单元所代表词的概率。示例树见图 4。

**图4.** 层次 softmax 模型的一棵示例二叉树。白色单元是词表中的词，深色单元是内部单元。从根到 $w_2$ 的一条示例路径被高亮显示。在所示示例中，路径长度 $L(w_2) = 4$。$n(w, j)$ 表示从根到词 $w$ 的路径上的第 $j$ 个单元。

在层次 softmax 模型中，词没有输出向量表示。相反，$V - 1$ 个内部单元中的每一个都有一个输出向量 $v'_{n(w,j)}$。一个词作为输出词的概率定义为

$$
p(w = w_O) = \prod_{j=1}^{L(w)-1} \sigma(\mathbb{1}[n(w, j+1) = ch(n(w, j))] \cdot v'^T_{n(w,j)} h) \qquad (37)
$$

其中 $ch(n)$ 是单元 $n$ 的左子节点；$v'_{n(w,j)}$ 是内部单元 $n(w, j)$ 的向量表示（“输出向量”）；$h$ 是隐藏层的输出值（在 skip-gram 模型中 $h = v_{w_I}$；在 CBOW 中 $h = \frac{1}{C} \sum_{c=1}^{C} v_{w_c}$）；$\mathbb{1}[x]$ 是一个特殊函数，定义为

$$
\mathbb{1}[x] =
\begin{cases}
1 & \text{if } x \text{ is true;} \\
-1 & \text{otherwise.}
\end{cases} \qquad (38)
$$

让我们通过一个例子直观理解这个方程。看图 4，假设我们要计算 $w_2$ 作为输出词的概率。我们将这个概率定义为从根开始、到所考察的叶子单元结束的随机游走的概率。在每个内部单元（包括根单元）处，我们需要分配向左走和向右走的概率。[^4] 我们定义在内部单元 $n$ 处向左走的概率为

$$
p(n, \text{left}) = \sigma(v'^T_n \cdot h) \qquad (39)
$$

它同时由内部单元的向量表示和隐藏层的输出值（它又由输入词的向量表示决定）决定。显然，在单元 $n$ 处向右走的概率为

$$
p(n, \text{right}) = 1 - \sigma(v'^T_n \cdot h) = \sigma(-v'^T_n \cdot h) \qquad (40)
$$

沿着图 4 中从根到 $w_2$ 的路径，我们可以计算出 $w_2$ 作为输出词的概率为

$$
p(w_2 = w_O) = p(n(w_2, 1), \text{left}) \cdot p(n(w_2, 2), \text{left}) \cdot p(n(w_2, 3), \text{right}) \qquad (41)
$$

$$
= \sigma(v'^T_{n(w_2,1)} h) \cdot \sigma(v'^T_{n(w_2,2)} h) \cdot \sigma(-v'^T_{n(w_2,3)} h) \qquad (42)
$$

这正是 (37) 所给出的结果。不难验证

$$
\sum_{i=1}^{V} p(w_i = w_O) = 1 \qquad (43)
$$

使得层次 softmax 在所有词之间构成一个良定义的多项分布。

现在让我们推导内部单元向量表示的参数更新方程。为简单起见，我们首先考察单上下文词模型。将其扩展到 CBOW 和 skip-gram 模型很容易。

为简化记号，我们在不引入歧义的前提下定义以下缩写：

$$
\mathbb{1}[\cdot] := \mathbb{1}[n(w, j+1) = ch(n(w, j))] \qquad (44)
$$

$$
v'_j := v'_{n(w,j)} \qquad (45)
$$

对于一个训练样本，误差函数定义为

$$
E = -\log p(w = w_O|w_I) = -\sum_{j=1}^{L(w)-1} \log \sigma(\mathbb{1}[\cdot] v'^T_j h) \qquad (46)
$$

我们取 $E$ 对 $v'^T_j h$ 的导数，得到

$$
\frac{\partial E}{\partial v'^T_j h} = \sigma(\mathbb{1}[\cdot] v'^T_j h) - 1 \cdot \mathbb{1}[\cdot] \qquad (47)
$$

$$
=
\begin{cases}
\sigma(v'^T_j h) - 1 & (\mathbb{1}[\cdot] = 1) \\
\sigma(v'^T_j h) & (\mathbb{1}[\cdot] = -1)
\end{cases} \qquad (48)
$$

$$
= \sigma(v'^T_j h) - t_j \qquad (49)
$$

其中 $t_j = 1$ 若 $\mathbb{1}[\cdot] = 1$，否则 $t_j = 0$。

接下来我们取 $E$ 对内部单元 $n(w, j)$ 向量表示的导数，得到

$$
\frac{\partial E}{\partial v'_j} = \frac{\partial E}{\partial v'^T_j h} \cdot \frac{\partial v'^T_j h}{\partial v'_j} = (\sigma(v'^T_j h) - t_j) \cdot h \qquad (50)
$$

由此得到如下更新方程：

$$
v'^{(new)}_j = v'^{(old)}_j - \eta (\sigma(v'^T_j h) - t_j) \cdot h \qquad (51)
$$

该更新应应用于 $j = 1, 2, \cdots, L(w) - 1$。我们可以将 $\sigma(v'^T_j h) - t_j$ 理解为内部单元 $n(w, j)$ 的预测误差。每个内部单元的“任务”是预测在随机游走中应当跟随左子节点还是右子节点。$t_j = 1$ 意味着真实情况是跟随左子节点；$t_j = 0$ 意味着应当跟随右子节点。$\sigma(v'^T_j h)$ 是预测结果。对于一个训练样本，如果内部单元的预测非常接近真实情况，那么它的向量表示 $v'_j$ 将移动得很少；否则 $v'_j$ 将通过向适当方向移动（远离或靠近 $h$）[^5] 来减小该样本的预测误差。这个更新方程可用于 CBOW 和 skip-gram 模型。当用于 skip-gram 模型时，我们需要对输出上下文中的 $C$ 个词中的每一个重复此更新过程。

为了将误差反向传播以学习输入→隐藏权重，我们取 $E$ 对隐藏层输出的导数，得到

$$
\frac{\partial E}{\partial h} = \sum_{j=1}^{L(w)-1} \frac{\partial E}{\partial v'^T_j h} \cdot \frac{\partial v'^T_j h}{\partial h} \qquad (52)
$$

$$
= \sum_{j=1}^{L(w)-1} (\sigma(v'^T_j h) - t_j) \cdot v'_j \qquad (53)
$$

$$
:= EH \qquad (54)
$$

它可以被直接代入 (23) 以获得 CBOW 输入向量的更新方程。对于 skip-gram 模型，我们需要为 skip-gram 上下文中的每个词计算一个 $EH$ 值，并将各 $EH$ 值之和代入 (35) 以获得输入向量的更新方程。

从更新方程可以看出，每个训练样本每个上下文词的计算复杂度从 $O(V)$ 降低到 $O(\log(V))$，这是速度上的巨大提升。我们仍然有大致相同数量的参数（内部单元有 $V - 1$ 个向量，相比原来词的 $V$ 个输出向量）。

### 3.2 负采样

负采样的想法比层次 softmax 更直接：为了应对每轮迭代需要更新太多输出向量的困难，我们只更新它们的一个采样。

显然，输出词（即真实情况，或正样本）应当保留在我们的样本中并被更新，而且我们需要采样几个词作为负样本（因此称为“负采样”）。采样过程需要一个概率分布，它可以被任意选择。我们称这个分布为噪声分布，记为 $P_n(w)$。人们可以凭经验确定一个好的分布。[^6]

在 word2vec 中，作者没有使用能够产生良定义后验多项分布形式的负采样，而是论证了以下简化的训练目标能够产生高质量的词嵌入：[^7]

$$
E = -\log \sigma(v'^T_{w_O} h) - \sum_{w_j \in W_{neg}} \log \sigma(-v'^T_{w_j} h) \qquad (55)
$$

其中 $w_O$ 是输出词（即正样本），$v'_{w_O}$ 是它的输出向量；$h$ 是隐藏层的输出值：在 CBOW 模型中 $h = \frac{1}{C} \sum_{c=1}^{C} v_{w_c}$，在 skip-gram 模型中 $h = v_{w_I}$；$W_{neg} = \{w_j | j = 1, \cdots, K\}$ 是基于 $P_n(w)$ 采样得到的词的集合，即负样本。

为了获得负采样下词向量的更新方程，我们首先取 $E$ 对输出单元 $w_j$ 净输入的导数：

$$
\frac{\partial E}{\partial v'^T_{w_j} h} =
\begin{cases}
\sigma(v'^T_{w_j} h) - 1 & \text{if } w_j = w_O \\
\sigma(v'^T_{w_j} h) & \text{if } w_j \in W_{neg}
\end{cases} \qquad (56)
$$

$$
= \sigma(v'^T_{w_j} h) - t_j \qquad (57)
$$

其中 $t_j$ 是词 $w_j$ 的“标签”。当 $w_j$ 是正样本时 $t_j = 1$；否则 $t_j = 0$。

接下来我们取 $E$ 对词 $w_j$ 输出向量的导数，

$$
\frac{\partial E}{\partial v'_{w_j}} = \frac{\partial E}{\partial v'^T_{w_j} h} \cdot \frac{\partial v'^T_{w_j} h}{\partial v'_{w_j}} = (\sigma(v'^T_{w_j} h) - t_j) \cdot h \qquad (58)
$$

由此得到其输出向量的如下更新方程：

$$
v'^{(new)}_{w_j} = v'^{(old)}_{w_j} - \eta (\sigma(v'^T_{w_j} h) - t_j) \cdot h \qquad (59)
$$

它只需要应用于 $w_j \in \{w_O\} \cup W_{neg}$ 而不是词表中的每一个词。这表明为什么我们可以在每轮迭代中节省大量的计算开销。

上述更新方程的直观理解应当与 (11) 相同。该方程可用于 CBOW 和 skip-gram 模型。对于 skip-gram 模型，我们一次针对一个上下文词应用该方程。

为了将误差反向传播到隐藏层从而更新词的输入向量，我们需要取 $E$ 对隐藏层输出的导数，得到

$$
\frac{\partial E}{\partial h} = \sum_{w_j \in \{w_O\} \cup W_{neg}} \frac{\partial E}{\partial v'^T_{w_j} h} \cdot \frac{\partial v'^T_{w_j} h}{\partial h} \qquad (60)
$$

$$
= \sum_{w_j \in \{w_O\} \cup W_{neg}} (\sigma(v'^T_{w_j} h) - t_j) \cdot v'_{w_j} := EH \qquad (61)
$$

通过将 $EH$ 代入 (23)，我们得到 CBOW 模型输入向量的更新方程。对于 skip-gram 模型，我们需要为 skip-gram 上下文中的每个词计算一个 $EH$ 值，并将各 $EH$ 值之和代入 (35) 以获得输入向量的更新方程。

## 致谢

作者感谢 Eytan Adar、Qiaozhu Mei、Jian Tang、Dragomir Radev、Daniel Pressel、Thomas Dean、Sudeep Gandhe、Peter Lau、Luheng He、Tomas Mikolov、Hao Jiang 和 Oded Shmueli 就本文主题的讨论以及/或对本文写作的改进。

## 参考文献

[1] Goldberg, Y. and Levy, O. (2014). word2vec explained: deriving mikolov et al.’s negative-sampling word-embedding method. arXiv:1402.3722 [cs, stat]. arXiv: 1402.3722.

[2] Mikolov, T., Chen, K., Corrado, G., and Dean, J. (2013a). Efficient estimation of word representations in vector space. arXiv preprint arXiv:1301.3781.

[3] Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., and Dean, J. (2013b). Distributed representations of words and phrases and their compositionality. In Advances in Neural Information Processing Systems, pages 3111–3119.

[4] Mnih, A. and Hinton, G. E. (2009). A scalable hierarchical distributed language model. In Koller, D., Schuurmans, D., Bengio, Y., and Bottou, L., editors, Advances in Neural Information Processing Systems 21, pages 1081–1088. Curran Associates, Inc.

[5] Morin, F. and Bengio, Y. (2005). Hierarchical probabilistic neural network language model. In AISTATS, volume 5, pages 246–252. Citeseer.

## 附录 A 反向传播基础

### A.1 单单元的学习算法

图 5 展示了一个人工神经元（单元）。$\{x_1, \cdots, x_K\}$ 是输入值；$\{w_1, \cdots, w_K\}$ 是权重；$y$ 是标量输出；$f$ 是连接函数（也称为激活/决策/转移函数）。

**图5.** 一个人工神经元。

该单元按以下方式工作：

$$
y = f(u) , \qquad (62)
$$

其中 $u$ 是一个标量，是神经元的净输入（或“新输入”）。$u$ 定义为

$$
u = \sum_{i=0}^{K} w_i x_i . \qquad (63)
$$

使用向量记号，我们可以写成

$$
u = w^T x \qquad (64)
$$

注意，这里我们忽略了 $u$ 中的偏置项。要包含偏置项，只需添加一个恒为 1 的输入维度（例如 $x_0$）。

显然，不同的连接函数会导致神经元的不同行为。这里我们讨论两个连接函数的示例选择。

$f(u)$ 的第一个示例选择是单位阶跃函数（又名 Heaviside 阶跃函数）：

$$
f(u) =
\begin{cases}
1 & \text{if } u > 0 \\
0 & \text{otherwise}
\end{cases} \qquad (65)
$$

具有这种连接函数的神经元称为感知机（perceptron）。感知机的学习算法是感知机算法。其更新方程定义为：

$$
w^{(new)} = w^{(old)} - \eta \cdot (y - t) \cdot x \qquad (66)
$$

其中 $t$ 是标签（gold standard），$\eta$ 是学习率（$\eta > 0$）。注意，感知机是线性分类器，这意味着其描述能力可能非常有限。如果我们想要拟合更复杂的函数，就需要使用非线性模型。

$f(u)$ 的第二个示例选择是逻辑函数（最常用的一种 sigmoid 函数），定义为

$$
\sigma(u) = \frac{1}{1 + e^{-u}} \qquad (67)
$$

逻辑函数有两个主要的优良性质：(1) 输出 $y$ 总是在 0 和 1 之间；(2) 与单位阶跃函数不同，$\sigma(u)$ 是光滑可微的，使得更新方程的推导非常容易。

注意，$\sigma(u)$ 还具有以下两个非常方便且将在我们后续推导中使用的性质：

$$
\sigma(-u) = 1 - \sigma(u) \qquad (68)
$$

$$
\frac{d\sigma(u)}{du} = \sigma(u)\sigma(-u) \qquad (69)
$$

我们使用随机梯度下降作为该模型的学习算法。为了推导更新方程，我们需要定义误差函数，即训练目标。以下目标函数似乎很方便：

$$
E = \frac{1}{2} (t - y)^2 \qquad (70)
$$

我们取 $E$ 对 $w_i$ 的导数，

$$
\frac{\partial E}{\partial w_i} = \frac{\partial E}{\partial y} \cdot \frac{\partial y}{\partial u} \cdot \frac{\partial u}{\partial w_i} \qquad (71)
$$

$$
= (y - t) \cdot y(1 - y) \cdot x_i \qquad (72)
$$

其中 $\frac{\partial y}{\partial u} = y(1 - y)$ 因为 $y = f(u) = \sigma(u)$，并利用了 (68) 和 (69)。一旦我们有了导数，就可以应用随机梯度下降：

$$
w^{(new)} = w^{(old)} - \eta \cdot (y - t) \cdot y(1 - y) \cdot x . \qquad (73)
$$

### A.2 多层网络中的反向传播

图 6 展示了一个多层神经网络，具有一个输入层 $\{x_k\} = \{x_1, \cdots, x_K\}$、一个隐藏层 $\{h_i\} = \{h_1, \cdots, h_N\}$ 和一个输出层 $\{y_j\} = \{y_1, \cdots, y_M\}$。为清晰起见，我们分别用 $k$、$i$、$j$ 作为输入层、隐藏层和输出层单元的下标。我们用 $u_i$ 和 $u'_j$ 分别表示隐藏层单元和输出层单元的净输入。我们想要推导学习输入层与隐藏层之间权重 $w_{ki}$ 以及隐藏层与输出层之间权重 $w'_{ij}$ 的更新方程。我们假设所有计算单元（即隐藏层和输出层中的单元）都使用逻辑函数 $\sigma(u)$ 作为连接函数。因此，对于隐藏层中的一个单元 $h_i$，其输出定义为

**图6.** 带一个隐藏层的多层神经网络。

$$
h_i = \sigma(u_i) = \sigma \left( \sum_{k=1}^{K} w_{ki} x_k \right) . \qquad (74)
$$

类似地，对于输出层中的一个单元 $y_j$，其输出定义为

$$
y_j = \sigma(u'_j) = \sigma \left( \sum_{i=1}^{N} w'_{ij} h_i \right) . \qquad (75)
$$

我们使用平方和误差函数，由下式给出

$$
E(x, t, W, W') = \frac{1}{2} \sum_{j=1}^{M} (y_j - t_j)^2 , \qquad (76)
$$

其中 $W = \{w_{ki}\}$，一个 $K \times N$ 权重矩阵（输入-隐藏），$W' = \{w'_{ij}\}$，一个 $N \times M$ 权重矩阵（隐藏-输出）。$t = \{t_1, \cdots, t_M\}$，一个 $M$ 维向量，是输出的 gold-standard 标签。

为了获得 $w_{ki}$ 和 $w'_{ij}$ 的更新方程，我们只需分别对权重取误差函数 $E$ 的导数。为了使推导直接明了，我们从最右侧的层（即输出层）开始计算导数，然后向左移动。对于每一层，我们将计算分为三个步骤，分别计算误差对输出、净输入和权重的导数。此过程如下所示。

我们从输出层开始。第一步是计算误差对输出的导数：

$$
\frac{\partial E}{\partial y_j} = y_j - t_j . \qquad (77)
$$

第二步是计算误差对输出层净输入的导数。注意，当对某个量求导时，我们需要保持其他一切固定。还要注意，这个值非常重要，因为它将在后续计算中被多次重用。为简单起见，我们将其记为 $EI'_j$。

$$
\frac{\partial E}{\partial u'_j} = \frac{\partial E}{\partial y_j} \cdot \frac{\partial y_j}{\partial u'_j} = (y_j - t_j) \cdot y_j(1 - y_j) := EI'_j \qquad (78)
$$

第三步是计算误差对隐藏层与输出层之间权重的导数。

$$
\frac{\partial E}{\partial w'_{ij}} = \frac{\partial E}{\partial u'_j} \cdot \frac{\partial u'_j}{\partial w'_{ij}} = EI'_j \cdot h_i \qquad (79)
$$

到目前为止，我们已经得到了隐藏层与输出层之间权重的更新方程。

$$
w'^{(new)}_{ij} = w'^{(old)}_{ij} - \eta \cdot \frac{\partial E}{\partial w'_{ij}} \qquad (80)
$$

$$
= w'^{(old)}_{ij} - \eta \cdot EI'_j \cdot h_i . \qquad (81)
$$

其中 $\eta > 0$ 是学习率。

我们可以重复相同的三个步骤来获得前一层权重的更新方程，这本质上就是反向传播的思想。

我们重复第一步，计算误差对隐藏层输出的导数。注意，隐藏层的输出与输出层中的所有单元相关。

$$
\frac{\partial E}{\partial h_i} = \sum_{j=1}^{M} \frac{\partial E}{\partial u'_j} \cdot \frac{\partial u'_j}{\partial h_i} = \sum_{j=1}^{M} EI'_j \cdot w'_{ij} . \qquad (82)
$$

然后我们重复上面的第二步，计算误差对隐藏层净输入的导数。这个值再次非常重要，我们将其记为 $EI_i$。

$$
\frac{\partial E}{\partial u_i} = \frac{\partial E}{\partial h_i} \cdot \frac{\partial h_i}{\partial u_i} = \left( \sum_{j=1}^{M} EI'_j \cdot w'_{ij} \right) \cdot h_i (1 - h_i) := EI_i \qquad (83)
$$

接下来我们重复上面的第三步，计算误差对输入层与隐藏层之间权重的导数。

$$
\frac{\partial E}{\partial w_{ki}} = \frac{\partial E}{\partial u_i} \cdot \frac{\partial u_i}{\partial w_{ki}} = EI_i \cdot x_k , \qquad (84)
$$

最后，我们可以获得输入层与隐藏层之间权重的更新方程。

$$
w^{(new)}_{ki} = w^{(old)}_{ki} - \eta \cdot EI_i \cdot x_k . \qquad (85)
$$

从上面的例子中，我们可以看到在计算某一层导数时的中间结果（$EI'_j$）可以被前一层重用。想象一下，如果在输入层之前还有另一层，那么 $EI_i$ 也可以被重用，以高效地继续计算导数链。比较方程 (78) 和 (83)，我们可能会发现，在 (83) 中，因子 $\sum_{j=1}^{M} EI'_j w'_{ij}$ 就像隐藏层单元 $h_i$ 的“误差”。我们可以将这个项解释为从下一层“反向传播”回来的误差，如果网络有更多隐藏层，这种传播可以继续向更深处进行。

## 附录 B wevi：词嵌入可视化检查器

一个交互式可视化界面 wevi（word embedding visual inspector，词嵌入可视化检查器）可在线上使用，用于演示本文所描述模型的工作机制。wevi 的截图见图 7。

该演示允许用户直观地检查输入向量和输出向量在消费每个训练样本时的移动情况。训练过程也可以批量模式运行（例如连续消费 500 个训练样本），这可以揭示权重矩阵和相应词向量中模式的出现。主成分分析（PCA）被用来在二维散点图中可视化“高”维向量。该演示支持 CBOW 和 skip-gram 两种模型。

训练模型之后，用户可以手动激活一个或多个输入层单元，并检查哪些隐藏层单元和输出层单元变得活跃。用户还可以自定义训练数据、隐藏层大小和学习率。提供了几个预设的训练数据集，它们可以产生一些看起来有趣的不同结果，例如使用一个玩具词表重现著名的词类比：king - queen = man - woman。

希望通过与此演示交互，人们能够快速洞悉模型的工作机制。该系统可在 http://bit.ly/wevi-online 获得。源代码可在 http://github.com/ronxin/wevi 获得。

**图7.** wevi 截图（http://bit.ly/wevi-online）。

---

## 脚注

[^1]: 在线交互式演示位于：http://bit.ly/wevi-online。

[^2]: 在图 1、2、3 及本文其余部分中，$W'$ 不是 $W$ 的转置，而是一个不同的矩阵。

[^3]: 当我在此说“更近”或“更远”时，我指的是以内积而非欧氏距离作为距离度量。

[^4]: 虽然二叉树的一个内部单元不一定总是有两个子节点，但二叉 Huffman 树的内部单元总是如此。尽管理论上人们可以为层次 softmax 使用许多不同类型的树，但 word2vec 使用二叉 Huffman 树以加快训练。

[^5]: 同样，这里的距离度量是内积。

[^6]: 如 Mikolov 等人 [3] 所述，word2vec 使用一元分布（unigram distribution）的 3/4 次方以获得最佳结果质量。

[^7]: Goldberg and Levy [1] 对为何使用此目标函数提供了理论分析。
