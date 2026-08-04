# Self-Attentive Sequential Recommendation

> Wang-Cheng Kang, Julian McAuley | UC San Diego
>
> {wckang,jmcauley}@ucsd.edu

本文提出了 SASRec，一种基于自注意力机制的序列推荐模型，能够像 RNN 一样捕获长期语义，同时像 MC 一样仅基于少量动作进行预测。核心内容：

- 使用自注意力机制对用户历史行为序列建模，自适应地为每个时间步的历史 item 分配权重
- 通过位置嵌入（Positional Embedding）、层归一化（Layer Normalization）、残差连接（Residual Connection）和 Dropout 等技术构建深层自注意力网络
- 在稀疏和稠密数据集上均超越 MC/CNN/RNN 等现有方法，且训练速度比 CNN/RNN 方法快一个数量级
- 注意力可视化显示模型能自适应处理不同密度的数据集，并能学习到有意义的 item 间模式

关键发现：

- SASRec 在稀疏和稠密数据集上均优于所有基线方法，平均 Hit Rate 提升 6.9%，NDCG 提升 9.6%
- 在稀疏数据集上，模型倾向于关注最近几个 item；在稠密数据集上，模型会关注更长范围的 item
- 训练速度是 Caser 的 11 倍以上，是 GRU4Rec+ 的 18 倍以上
- 共享 item 嵌入显著优于非共享方案，残差连接和 Dropout 对性能有重要影响

---

## 摘要

序列动态（Sequential dynamics）是许多现代推荐系统的关键特征，这类系统试图基于用户最近执行的动作来捕获其活动的"上下文"。为了捕获这些模式，主要有两类方法：马尔可夫链（MC）和循环神经网络（RNN）。马尔可夫链假设用户的下一个动作仅基于最后（或最后几个）动作即可预测，而 RNN 原则上能够发现更长程的语义。通常来说，基于 MC 的方法在极度稀疏的数据集上表现最佳——此时模型的简洁性至关重要，而 RNN 在数据更稠密的数据集上表现更好——此时可以承担更高的模型复杂度。我们工作的目标是在这两个目标之间取得平衡，通过提出一个基于自注意力的序列模型（SASRec），使其能够捕获长期语义（像 RNN 一样），同时利用注意力机制仅基于相对较少的动作进行预测（像 MC 一样）。在每个时间步，SASRec 从用户的历史行为中识别出哪些 item 是"相关的"，并用它们来预测下一个 item。广泛的实证研究表明，我们的方法在稀疏和稠密数据集上均优于各种最先进的序列模型（包括基于 MC/CNN/RNN 的方法）。此外，该模型比同类 CNN/RNN 模型快一个数量级。注意力权重的可视化还展示了我们的模型如何自适应地处理不同密度的数据集，并揭示了行为序列中有意义的模式。

## 1. 引言

序列推荐系统的目标是将用户行为的个性化模型（基于历史活动）与基于用户最近动作的某种"上下文"概念相结合。从序列动态中捕获有用的模式具有挑战性，主要原因在于输入空间的维度会随着作为上下文的过去动作数量呈指数增长。因此，序列推荐研究主要关注如何简洁地捕获这些高阶动态。

马尔可夫链（MC）是一个经典例子，它假设下一个动作仅取决于前一个动作（或前几个），并已被成功应用于刻画短程的 item 转移以进行推荐 [1]。另一类工作使用循环神经网络（RNN）通过隐藏状态汇总所有之前的动作，并用其预测下一个动作 [2]。

这两种方法虽然在特定情况下表现强劲，但在某些数据类型上都有局限性。基于 MC 的方法通过做出强简化假设，在高稀疏性设置下表现良好，但可能无法捕获更复杂场景中的精细动态。相反，RNN 虽然表达力强，但在需要大量数据（尤其是稠密数据）之后才能超越更简单的基线模型。

最近，一种名为 Transformer 的新型序列模型在机器翻译任务上取得了最先进的性能和效率 [3]。与使用卷积或循环模块的现有序列模型不同，Transformer 完全基于一种称为"自注意力"（self-attention）的注意力机制，该机制非常高效，能够发现句子中单词之间的句法和语义模式。

受此方法的启发，我们尝试将自注意力机制应用于序列推荐问题。我们希望这一思想能够解决上述两个问题：一方面能够像 RNN 一样从过去所有动作中提取上下文，另一方面能够像 MC 一样仅基于少量动作进行预测。具体来说，我们构建了一个基于自注意力的序列推荐模型（SASRec），该模型在每个时间步自适应地为历史 item 分配权重（图1）。

所提出的模型在多个基准数据集上显著优于最先进的基于 MC/CNN/RNN 的序列推荐方法。特别地，我们考察了性能随数据集稀疏度的变化，发现模型性能与上述模式高度吻合。由于自注意力机制，SASRec 在稠密数据集上倾向于考虑长程依赖，而在稀疏数据集上则更关注最近的活动。这对于自适应地处理不同密度的数据集至关重要。

此外，SASRec 的核心组件（即自注意力块）适合并行加速，使得模型比 CNN/RNN 替代方案快一个数量级。我们还分析了 SASRec 的复杂度和可扩展性，进行了全面的消融研究以展示关键组件的影响，并通过可视化注意力权重定性揭示了模型的行为。

## 2. 相关工作

有几类工作与我们的研究密切相关。我们首先讨论通用推荐，然后是时间推荐，接着讨论序列推荐（特别是 MC 和 RNN）。最后介绍注意力机制，尤其是自注意力模块——这是我们模型的核心。

### 2.1 通用推荐

推荐系统主要关注基于历史反馈（如点击、购买、点赞）对用户与 item 之间的兼容性进行建模。用户反馈可以是显式的（如评分）或隐式的（如点击、购买、评论）[4, 5]。由于难以解释"未观测到"（如未购买）的数据，对隐式反馈建模具有挑战性。为解决此问题，逐点（point-wise）[4] 和成对（pairwise）[5] 方法被提出来应对这些挑战。

矩阵分解（MF）方法试图发现用户偏好和 item 属性的 latent 维度，并通过用户嵌入与 item 嵌入之间的内积来估计交互 [6, 7]。此外，另一类工作基于 item 相似度模型（ISM），不显式地用 latent 因子对每个用户建模（如 FISM [8]）。它们学习一个 item 到 item 的相似度矩阵，并通过衡量目标 item 与用户历史交互过的 item 之间的相似度来估计用户对某个 item 的偏好。

近年来，由于深度学习在相关问题上的成功，各种深度学习技术被引入推荐系统 [9]。一类工作尝试使用神经网络提取 item 特征（如图像 [10, 11]、文本 [12, 13] 等）用于内容感知推荐。另一类工作尝试替代传统的 MF。例如，NeuMF [14] 通过多层感知机（MLP）估计用户偏好，AutoRec [15] 使用自编码器预测评分。

### 2.2 时间推荐

追溯到 Netflix Prize 时期，时间推荐通过显式建模用户活动的时间戳，在各种任务上表现出了强劲的性能。TimeSVD++ [16] 通过将时间分为多个段，并在每个段中分别对用户和 item 建模，取得了优异的结果。这类模型对于理解具有显著（短期或长期）时间"漂移"的数据集至关重要（例如"过去 10 年电影偏好如何变化？"或"用户在下午 4 点会访问什么类型的商家？"等）[16, 17, 18]。序列推荐（或下一个 item 推荐）与这种设置略有不同，因为它只考虑动作的顺序，并建模与时间无关的序列模式。本质上，序列模型试图基于用户的近期活动建模其"上下文"，而非考虑时间模式本身。

### 2.3 序列推荐

许多序列推荐系统尝试建模 item-item 转移矩阵，以捕获连续 item 之间的序列模式。例如，FPMC 融合了一个 MF 项和一个 item-item 转移项，分别捕获长期偏好和短期转移 [1]。本质上，捕获的转移是一阶马尔可夫链（MC），而高阶 MC 假设下一个动作与之前的多个动作相关。由于最后一个访问的 item 通常是影响用户下一个动作的关键因素（即提供"上下文"），基于一阶 MC 的方法表现出强劲的性能，尤其是在稀疏数据集上 [19]。也有方法采用高阶 MC 来考虑更多之前的 item [20, 21]。特别地，卷积序列嵌入（Caser）——一种基于 CNN 的方法——将 L 个之前 item 的嵌入矩阵视为一张"图像"，并应用卷积操作来提取转移特征 [22]。

除了基于 MC 的方法，另一类工作采用 RNN 对用户序列建模 [2, 23, 24, 25]。例如，GRU4Rec 使用门控循环单元（GRU）对点击序列建模以进行基于会话的推荐 [2]，其改进版本进一步提升了对 Top-N 推荐的性能 [26]。在每个时间步，RNN 将上一步的状态和当前动作作为输入。这些依赖关系使得 RNN 效率较低，不过已有技术如"会话并行"（session-parallelism）被提出来提高效率 [2]。

### 2.4 注意力机制

注意力机制已被证明在各种任务中有效，如图像描述生成 [27] 和机器翻译 [28] 等。这些机制的核心思想是：序列输出（例如）依赖于模型应依次关注的输入中的某些"相关"部分。另一个好处是，基于注意力的方法通常更可解释。最近，注意力机制已被引入推荐系统 [29, 30, 31]。例如，注意力因子分解机（AFM）[30] 学习内容感知推荐中每个特征交互的重要性。

然而，上述使用的注意力技术本质上是原始模型的附加组件（如 attention+RNN、attention+FM 等）。最近，一种纯基于注意力的序列到序列方法 Transformer [3] 在机器翻译任务上取得了最先进的性能和效率，而此前该领域由基于 RNN/CNN 的方法主导 [32, 33]。Transformer 模型严重依赖于所提出的"自注意力"模块来捕获句子中的复杂结构，并检索相关单词（在源语言中）以生成下一个单词（在目标语言中）。受 Transformer 启发，我们尝试构建一个基于自注意力方法的新型序列推荐模型，尽管序列推荐问题与机器翻译有很大不同，需要专门设计的模型。

## 3. 方法

在序列推荐设置中，给定用户的行为序列 $S^u = (S^u_1, S^u_2, \ldots, S^u_{|S^u|})$ ，我们需要预测下一个 item。在训练过程中，在时间步 $t$ ，模型根据之前的 $t$ 个 item 预测下一个 item。如图1所示，将模型的输入视为 $(S^u_1, S^u_2, \ldots, S^u_{|S^u|-1})$ ，期望输出视为同一序列的"移位"版本 $(S^u_2, S^u_3, \ldots, S^u_{|S^u|})$ 会更方便。在本节中，我们描述如何通过嵌入层、多个自注意力块和预测层构建一个序列推荐模型。我们还将分析其复杂度，并进一步讨论 SASRec 与相关模型的区别。

**表1：符号说明。**（ $U, I$ — 用户和 item 集合； $S^u$ — 用户 $u$ 的历史交互序列 $(S^u_1, S^u_2, \ldots, S^u_{|S^u|})$ ； $d \in \mathbb{N}$ — latent 向量维度； $n \in \mathbb{N}$ — 最大序列长度； $b \in \mathbb{N}$ — 自注意力块数量； $M \in \mathbb{R}^{|I| \times d}$ — item 嵌入矩阵； $P \in \mathbb{R}^{n \times d}$ — 位置嵌入矩阵； $\hat{E} \in \mathbb{R}^{n \times d}$ — 输入嵌入矩阵； $S^{(b)} \in \mathbb{R}^{n \times d}$ — 第 $b$ 个自注意力层之后的 item 嵌入； $F^{(b)} \in \mathbb{R}^{n \times d}$ — 第 $b$ 个前馈网络之后的 item 嵌入）

### 3.1 嵌入层

我们将训练序列 $(S^u_1, S^u_2, \ldots, S^u_{|S^u|-1})$ 转换为一个固定长度的序列 $s = (s_1, s_2, \ldots, s_n)$ ，其中 $n$ 表示模型能够处理的最大长度。如果序列长度大于 $n$ ，我们取最近 $n$ 个动作。如果序列长度小于 $n$ ，我们重复在左侧添加"填充"（padding）item，直到长度为 $n$ 。我们创建一个 item 嵌入矩阵 $M \in \mathbb{R}^{|I| \times d}$ ，其中 $d$ 是 latent 维度，并获取输入嵌入矩阵 $E \in \mathbb{R}^{n \times d}$ ，其中 $E_i = M_{s_i}$ 。零向量 $\mathbf{0}$ 用作填充 item 的嵌入。

**位置嵌入（Positional Embedding）：** 正如我们将在下一节中看到的，由于自注意力模型不包含任何循环或卷积模块，它无法感知之前 item 的位置信息。因此，我们向输入嵌入中注入一个可学习的位置嵌入 $P \in \mathbb{R}^{n \times d}$ ：

$$
\hat{E} = [M_{s_1} + P_1; M_{s_2} + P_2; \ldots; M_{s_n} + P_n]^\top \qquad (1)
$$

我们也尝试了 [3] 中使用的固定位置嵌入，但发现在我们的场景中这会导致性能下降。我们在实验中定性和定量地分析了位置嵌入的影响。

### 3.2 自注意力块

缩放点积注意力（scaled dot-product attention）[3] 定义为：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right) V \qquad (2)
$$

其中 $Q$ 表示查询（queries）， $K$ 表示键（keys）， $V$ 表示值（values）（每行代表一个 item）。直观上，注意力层计算所有值的加权和，其中查询 $i$ 与值 $j$ 之间的权重与查询 $i$ 和键 $j$ 之间的交互有关。缩放因子 $\sqrt{d}$ 用于避免内积值过大，特别是在维度较高时。

**自注意力层：** 在机器翻译等 NLP 任务中，注意力机制通常与 $K = V$ 一起使用（例如，使用 RNN 编码器-解码器进行翻译：编码器的隐藏状态作为键和值，解码器的隐藏状态作为查询）[28]。最近，一种自注意力方法被提出，它使用相同的对象作为查询、键和值 [3]。在我们的场景中，自注意力操作以嵌入 $\hat{E}$ 作为输入，通过线性投影将其转换为三个矩阵，然后输入到注意力层：

$$
S = \text{SA}(\hat{E}) = \text{Attention}(\hat{E}W^Q, \hat{E}W^K, \hat{E}W^V) \qquad (3)
$$

其中投影矩阵 $W^Q, W^K, W^V \in \mathbb{R}^{d \times d}$ 。这些投影使模型更加灵活。例如，模型可以学习非对称的交互（即 $\langle \text{query } i, \text{key } j \rangle$ 和 $\langle \text{query } j, \text{key } i \rangle$ 可以有不同交互）。

**因果性（Causality）：** 由于序列的固有属性，模型在预测第 $t+1$ 个 item 时应该只考虑前 $t$ 个 item。然而，自注意力层的第 $t$ 个输出 $S_t$ 包含了后续 item 的嵌入，这使得模型不适定。因此，我们通过禁止 $Q_i$ 与 $K_j$ （ $j > i$ ）之间的所有连接来修改注意力机制。

**逐点前馈网络（Point-Wise Feed-Forward Network）：** 尽管自注意力能够以自适应权重聚合所有之前 item 的嵌入，但它本质上仍然是一个线性模型。为了赋予模型非线性能力并考虑不同 latent 维度之间的交互，我们对所有 $S_i$ 应用一个逐点两层前馈网络（共享参数）：

$$
F_i = \text{FFN}(S_i) = \text{ReLU}(S_i W^{(1)} + b^{(1)}) W^{(2)} + b^{(2)} \qquad (4)
$$

其中 $W^{(1)}, W^{(2)} \in \mathbb{R}^{d \times d}$ 为矩阵， $b^{(1)}, b^{(2)} \in \mathbb{R}^d$ 为 $d$ 维向量。注意， $S_i$ 与 $S_j$ （ $i \neq j$ ）之间没有交互，这意味着我们仍然防止了从后向前的信息泄漏。

### 3.3 堆叠自注意力块

在第一个自注意力块之后， $F$ 实质上聚合了所有之前 item 的嵌入（即 $\hat{E}_j, j \leq i$ ）。然而，基于 $F$ 再通过另一个自注意力块来学习更复杂的 item 转移可能是有用的。具体来说，我们堆叠自注意力块（即一个自注意力层和一个前馈网络），第 $b$ 个（ $b > 1$ ）块定义为：

$$
S^{(b)} = \text{SA}(F^{(b-1)}) \qquad (5)
$$
$$
F^{(b)}_i = \text{FFN}(S^{(b)}_i), \quad \forall i \in \{1, 2, \ldots, n\}
$$

第一个块定义为 $S^{(1)} = S$ 和 $F^{(1)} = F$ 。

然而，当网络变深时，几个问题会加剧：（1）模型容量增加导致过拟合；（2）训练过程变得不稳定（由于梯度消失等）；（3）参数更多的模型通常需要更长的训练时间。受 [3] 启发，我们执行以下操作以缓解这些问题：

$$
g(x) = x + \text{Dropout}(g(\text{LayerNorm}(x)))
$$

其中 $g(x)$ 表示自注意力层或前馈网络。也就是说，对于每个块中的层 $g$ ，我们在将输入 $x$ 送入 $g$ 之前对其进行层归一化，对 $g$ 的输出应用 Dropout，然后将输入 $x$ 加到最后输出中。

**残差连接（Residual Connections）：** 在某些情况下，多层神经网络已展现出分层学习有意义特征的能力 [34]。然而，直到残差网络被提出之前，简单地增加更多层并不能很容易地转化为更好的性能 [35]。残差网络的核心思想是通过残差连接将低层特征传播到更高层。因此，如果低层特征有用，模型可以轻松地将其传播到最终层。类似地，我们假设残差连接在我们的场景中也是有用的。例如，现有序列推荐方法已表明最后访问的 item 对预测下一个 item 起着关键作用 [1, 19, 21]。然而，经过多个自注意力块之后，最后访问 item 的嵌入与所有之前的 item 纠缠在一起；添加残差连接将最后访问 item 的嵌入传播到最终层，将使模型更容易利用低层信息。

**层归一化（Layer Normalization）：** 层归一化用于对跨特征的输入进行归一化（即零均值和单位方差），这有助于稳定和加速神经网络训练 [36]。与批归一化（batch normalization）[37] 不同，层归一化中使用的统计数据与同一批次中的其他样本无关。具体来说，假设输入是一个包含样本所有特征的向量 $x$ ，其操作定义为：

$$
\text{LayerNorm}(x) = \alpha \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

其中 $\odot$ 是逐元素乘积（即 Hadamard 积）， $\mu$ 和 $\sigma$ 是 $x$ 的均值和标准差， $\alpha$ 和 $\beta$ 是学习的缩放因子和偏置项。

**Dropout：** 为了缓解深度神经网络中的过拟合问题，"Dropout"正则化技术已被证明在各种神经网络架构中有效 [38]。Dropout 的思想很简单：在训练期间以概率 $p$ 随机"关闭"神经元，在测试时使用所有神经元。进一步的分析指出，Dropout 可以被视为一种集成学习形式，它考虑了数量庞大的模型（数量与神经元和输入特征的数量呈指数关系）共享参数 [39]。我们还在嵌入 $\hat{E}$ 上应用了 Dropout 层。

### 3.4 预测层

在 $b$ 个自注意力块自适应且分层地提取已消费 item 的信息之后，我们基于 $F^{(b)}_t$ 预测下一个 item（给定前 $t$ 个 item）。具体来说，我们采用一个 MF 层来预测 item $i$ 的相关性：

$$
r_{i,t} = F^{(b)}_t N^\top_i
$$

其中 $r_{i,t}$ 是给定前 $t$ 个 item（即 $s_1, s_2, \ldots, s_t$ ）时，item $i$ 成为下一个 item 的相关性分数， $N \in \mathbb{R}^{|I| \times d}$ 是一个 item 嵌入矩阵。因此，交互分数 $r_{i,t}$ 越高表示相关性越高，我们可以通过对分数进行排序来生成推荐。

**共享 item 嵌入（Shared Item Embedding）：** 为了减小模型规模并缓解过拟合，我们考虑另一种方案，仅使用一个 item 嵌入矩阵 $M$ ：

$$
r_{i,t} = F^{(b)}_t M^\top_i \qquad (6)
$$

注意 $F^{(b)}_t$ 可以表示为依赖于 item 嵌入 $M$ 的函数 $F^{(b)}_t = f(M_{s_1}, M_{s_2}, \ldots, M_{s_t})$ 。使用同构 item 嵌入的一个潜在问题是它们的内积无法表示非对称的 item 转移（例如 item $i$ 经常在 $j$ 之后被购买，但反之则不然），因此像 FPMC 这样的现有方法倾向于使用异构 item 嵌入。然而，我们的模型不存在这个问题，因为它学习了一个非线性变换。例如，前馈网络可以轻松地使用相同的 item 嵌入实现非对称性： $\text{FFN}(M_i) M^\top_j \neq \text{FFN}(M_j) M^\top_i$ 。经验上，使用共享 item 嵌入显著提升了我们模型的性能。

**显式用户建模（Explicit User Modeling）：** 为提供个性化推荐，现有方法通常采用两种方式之一：（1）学习一个显式的用户嵌入来表示用户偏好（如 MF [40], FPMC [1] 和 Caser [22]）；（2）考虑用户之前的行为，从已访问 item 的嵌入中推导出一个隐式的用户嵌入（如 FISM [8], Fossil [21], GRU4Rec [2]）。我们的方法属于后一类，因为我们通过考虑用户的所有行为来生成嵌入 $F^{(b)}_n$ 。然而，我们也可以在最后一层插入显式的用户嵌入，例如通过加法： $r_{u,i,t} = (U_u + F^{(b)}_t) M^\top_i$ ，其中 $U$ 是用户嵌入矩阵。但我们在经验上发现，添加显式用户嵌入并不会提升性能（可能是因为模型已经考虑了用户的所有行为）。

### 3.5 网络训练

回顾一下，我们将每个用户序列（排除最后一个动作） $(S^u_1, S^u_2, \ldots, S^u_{|S^u|-1})$ 通过截断或填充转换为一个固定长度序列 $s = \{s_1, s_2, \ldots, s_n\}$ 。我们将 $o_t$ 定义为时间步 $t$ 的期望输出：

$$
o_t = \begin{cases}
\langle \text{pad} \rangle & \text{if } s_t \text{ is a padding item} \\
s_{t+1} & \text{if } 1 \leq t < n \\
S^u_{|S^u|} & \text{if } t = n
\end{cases}
$$

其中 $\langle \text{pad} \rangle$ 表示填充 item。我们的模型将序列 $s$ 作为输入，相应的序列 $o$ 作为期望输出，并采用二元交叉熵损失作为目标函数：

$$
-\sum_{S^u \in \mathcal{S}} \sum_{t \in [1,2,\ldots,n]} \left[ \log(\sigma(r_{o_t,t})) + \sum_{j \notin S^u} \log(1 - \sigma(r_{j,t})) \right]
$$

注意我们忽略了 $o_t = \langle \text{pad} \rangle$ 的项。

网络使用 Adam 优化器 [41] 进行优化，它是带有自适应矩估计的随机梯度下降（SGD）的变体。在每个 epoch 中，我们为每个序列中的每个时间步随机生成一个负 item $j$ 。实现细节稍后描述。

### 3.6 复杂度分析

**空间复杂度：** 我们模型中的学习参数来自嵌入层以及自注意力层、前馈网络和层归一化中的参数。总参数量为 $O(|I|d + nd + d^2)$ ，与其他方法相比是适度的（例如 FPMC 为 $O(|U|d + |I|d)$ ），因为它不随用户数量增长，且 $d$ 在推荐问题中通常很小。

**时间复杂度：** 我们模型的计算复杂度主要来自自注意力层和前馈网络，为 $O(n^2 d + n d^2)$ 。主导项通常是来自自注意力层的 $O(n^2 d)$ 。然而，我们模型的一个便利特性是每个自注意力层中的计算是完全可并行的，适合 GPU 加速。相比之下，基于 RNN 的方法（如 GRU4Rec [2]）对时间步有依赖关系（即时间步 $t$ 的计算必须等待时间步 $t-1$ 的结果），导致顺序操作的 $O(n)$ 时间。我们在经验上发现，我们的方法在使用 GPU 时比基于 RNN 和 CNN 的方法快十倍以上（结果与 [3] 中机器翻译任务的结果类似），最大长度 $n$ 可以轻松扩展到几百，这通常足以满足现有基准数据集的需求。

在测试时，对于每个用户，在计算出嵌入 $F^{(b)}_n$ 之后，过程与标准 MF 方法相同（评估对某个 item 偏好为 $O(d)$ ）。

**处理长序列：** 尽管我们的实验实证验证了我们方法的效率，但最终它无法扩展到非常长的序列。未来有几个有前景的方向可以探索：（1）使用受限自注意力（restricted self-attention）[42]，仅关注最近的动作而非所有动作，远距离的动作可以在更高层中考虑；（2）如 [22] 中那样将长序列分割为短段。

### 3.7 讨论

我们发现 SASRec 可以被视为一些经典协同过滤（CF）模型的泛化。我们还将从概念上讨论我们的方法和现有方法如何处理序列建模。

**归约为现有模型：**
- **分解马尔可夫链（Factorized Markov Chains, FMC）：** FMC 分解一阶 item 转移矩阵，并根据最后访问的 item $i$ 预测下一个 item $j$ ： $P(j|i) \propto M^\top_i N_j$ 。如果我们设自注意力块为零，使用非共享的 item 嵌入，并移除位置嵌入，则 SASRec 归约为 FMC。此外，SASRec 也与分解个性化马尔可夫链（FPMC）[1] 密切相关，后者融合了 MF 和 FMC 以分别捕获用户偏好和短期动态： $P(j|u,i) \propto [U_u, M_i] N^\top_j$ 。按照上述对 FMC 的归约操作，并添加一个显式的用户嵌入（通过拼接），SASRec 等价于 FPMC。
- **分解 item 相似度模型（Factorized Item Similarity Models）[8]：** FISM 通过考虑目标 item $i$ 与用户之前消费过的 item 之间的相似度来估计偏好分数： $P(j|u) \propto \frac{1}{|S^u|} \sum_{i \in S^u} M_i N^\top_j$ 。如果我们使用一个自注意力层（排除前馈网络），在 item 上设置均匀的注意力权重（即 $1/|S^u|$ ），使用非共享的 item 嵌入，并移除位置嵌入，则 SASRec 归约为 FISM。因此，我们的模型可以被视为一个用于下一个 item 推荐的自适应、分层、序列化 item 相似度模型。

**基于 MC 的推荐：** 马尔可夫链（MC）可以有效地捕获局部序列模式，假设下一个 item 仅依赖于之前的 $L$ 个 item。现有的基于 MC 的序列推荐方法要么依赖于一阶 MC（如 FPMC [1], HRM [43], TransRec [19]），要么依赖于高阶 MC（如 Fossil [21], Vista [20], Caser [22]）。第一类方法在稀疏数据集上往往表现最佳。相比之下，基于高阶 MC 的方法有两个局限性：（1）MC 阶数 $L$ 需要在训练前指定，而非自适应选择；（2）性能和效率随阶数 $L$ 的扩展性不佳，因此 $L$ 通常很小（例如小于 5）。我们的方法解决了第一个问题，因为它可以自适应地关注相关的前序 item（例如在稀疏数据集上仅关注最后一个 item，在稠密数据集上关注更多 item）。此外，我们的模型本质上以 $n$ 个前序 item 为条件，并且经验上可以扩展到几百个前序 item，在训练时间适度增加的情况下显示出性能提升。

**基于 RNN 的推荐：** 另一类工作尝试使用 RNN 对用户行为序列建模 [2, 17, 26]。RNN 通常适合对序列建模，尽管最近的研究表明 CNN 和自注意力在某些序列设置中可以更强 [3, 44]。我们基于自注意力的模型可以从 item 相似度模型推导出来，这是序列建模用于推荐的一个合理替代方案。对于 RNN，除了它们在并行计算方面的低效率（第 3.6 节）外，它们的最大路径长度（从输入节点到相关输出节点）为 $O(n)$ 。相比之下，我们的模型具有 $O(1)$ 的最大路径长度，这有利于学习长程依赖关系 [45]。

## 4. 实验

在本节中，我们介绍实验设置和实验结果。我们的实验旨在回答以下研究问题：
- **RQ1：** SASRec 是否超越了包括基于 CNN/RNN 方法在内的最先进模型？
- **RQ2：** SASRec 架构中各种组件的影响是什么？
- **RQ3：** SASRec 的训练效率和可扩展性（关于 $n$ ）如何？
- **RQ4：** 注意力权重能否学习到与位置或 item 属性相关的有意义的模式？

### 4.1 数据集

我们在来自三个真实世界应用的四个数据集上评估我们的方法。这些数据集在领域、平台和稀疏度上差异显著：

- **Amazon：** [46] 中引入的一系列数据集，包含从 Amazon.com 爬取的大规模产品评论语料。Amazon 上的顶级产品类别被视为独立的数据集。我们考虑两个类别："Beauty"和"Games"。该数据集以其高稀疏度和变异性著称。
- **Steam：** 我们引入了一个从 Steam（一个大型在线视频游戏分发平台）爬取的新数据集。该数据集包含 2,567,538 个用户、15,474 个游戏和 7,793,069 条英文评论，时间跨度为 2010 年 10 月到 2018 年 1 月。该数据集还包含可能在未来工作中有用的丰富信息，如用户游戏时长、定价信息、媒体评分、类别、开发者等。
- **MovieLens：** 一个广泛使用的评估协同过滤算法的基准数据集。我们使用包含 100 万条用户评分的版本（MovieLens-1M）。

我们遵循与 [1, 19, 21] 相同的预处理流程。对于所有数据集，我们将评论或评分的存在视为隐式反馈（即用户与该 item 进行过交互），并使用时间戳来确定行为序列的顺序。我们丢弃相关行为少于 5 个的用户和 item。对于数据划分，我们将每个用户 $u$ 的历史序列 $S^u$ 分为三部分：（1）最近的行为 $S^u_{|S^u|}$ 用于测试；（2）第二近的行为 $S^u_{|S^u|-1}$ 用于验证；（3）其余所有行为用于训练。注意在测试时，输入序列包含训练行为和验证行为。

**表2：数据集统计（预处理后）**
| 数据集 | 用户数 | item 数 | 平均行为/用户 | 平均行为/item | 行为数 |
|---|---|---|---|---|---|
| Amazon Beauty | 52,024 | 57,289 | 7.6 | 6.9 | 0.4M |
| Amazon Games | 31,013 | 23,715 | 9.3 | 12.1 | 0.3M |
| Steam | 334,730 | 13,047 | 11.0 | 282.5 | 3.7M |
| MovieLens-1M | 6,040 | 3,416 | 163.5 | 289.1 | 1.0M |

### 4.2 对比方法

为了展示我们方法的有效性，我们纳入三组推荐基线方法。

第一组包括仅考虑用户反馈而不考虑行为序列顺序的通用推荐方法：
- **PopRec：** 一个简单的基线方法，根据 item 的流行度（即相关行为的数量）对 item 进行排序。
- **Bayesian Personalized Ranking (BPR) [5]：** BPR 是一种从隐式反馈中学习个性化排名的经典方法。使用有偏矩阵分解作为底层推荐器。

第二组包含基于一阶马尔可夫链的序列推荐方法，它们考虑最后访问的 item：
- **Factorized Markov Chains (FMC)：** 一种一阶马尔可夫链方法。FMC 使用两个 item 嵌入分解 item 转移矩阵，并仅基于最后访问的 item 生成推荐。
- **Factorized Personalized Markov Chains (FPMC) [1]：** FPMC 使用矩阵分解和分解一阶马尔可夫链的组合作为推荐器，捕获用户的长期偏好以及 item 到 item 的转移。
- **Translation-based Recommendation (TransRec) [19]：** 一种最先进的一阶序列推荐方法，将每个用户建模为一个平移向量，以捕获从当前 item 到下一个 item 的转移。

最后一组包含基于深度学习的序列推荐系统，它们考虑多个（或全部）之前访问过的 item：
- **GRU4Rec [2]：** 一种开创性方法，使用 RNN 对用户行为序列建模以进行基于会话的推荐。我们将每个用户的反馈序列视为一个会话。
- **GRU4Rec+ [26]：** GRU4Rec 的改进版本，采用了不同的损失函数和采样策略，在 Top-N 推荐上显示出显著的性能提升。
- **Convolutional Sequence Embeddings (Caser) [22]：** 一种最近提出的基于 CNN 的方法，通过对最近的 $L$ 个 item 的嵌入矩阵应用卷积操作来捕获高阶马尔可夫链，并取得了最先进的序列推荐性能。

由于其他序列推荐方法（如 PRME [47], HRM [43], Fossil [21]）在上述基线方法中已在类似数据集上被超越，我们省略了与它们的对比。我们也不包括时间推荐方法（如 TimeSVD++ [16] 和 RRN [17]），它们的设置与我们这里的不同。

为公平比较，我们使用 TensorFlow 和 Adam [41] 优化器实现了 BPR、FMC、FPMC 和 TransRec。对于 GRU4Rec、GRU4Rec+ 和 Caser，我们使用对应作者提供的代码。对于除 PopRec 外的所有方法，我们考虑 latent 维度 $d \in \{10, 20, 30, 40, 50\}$ 。对于 BPR、FMC、FPMC 和 TransRec， $\ell_2$ 正则化参数选自 $\{0.0001, 0.001, 0.01, 0.1, 1\}$ 。所有其他超参数和初始化策略使用作者建议的设置。我们使用验证集调整超参数，如果验证性能在 20 个 epoch 内没有提升，则终止训练。

### 4.3 实现细节

对于默认版本的 SASRec 架构，我们使用两个自注意力块（ $b = 2$ ），并使用学习到的位置嵌入。嵌入层和预测层中的 item 嵌入是共享的。我们使用 TensorFlow 实现 SASRec。优化器为 Adam 优化器 [41]，学习率设置为 $0.001$ ，批次大小为 $128$ 。神经元的 Dropout 率设置为：MovieLens-1M 为 $0.2$ ，其他三个数据集由于稀疏性为 $0.5$ 。最大序列长度 $n$ 设置为：MovieLens-1M 为 $200$ ，其他三个数据集为 $50$ ，即大致与用户的平均行为数成比例。变体版本和不同超参数的性能将在下文检验。所有代码和数据将在论文发表时发布。

### 4.4 评估指标

我们采用两种常见的 Top-N 指标，即 Hit Rate@10 和 NDCG@10，来评估推荐性能 [14, 19]。Hit@10 统计真实的下一个 item 出现在前 10 个 item 中的比例，而 NDCG@10 是一种位置感知的指标，对更高的位置赋予更大的权重。注意由于每个用户只有一个测试 item，Hit@10 等价于 Recall@10，且与 Precision@10 成正比。

为避免对所有 user-item 对进行大量计算，我们遵循 [14, 48] 中的策略。对于每个用户 $u$ ，我们随机采样 100 个负 item，并将这些 item 与真实 item 一起排序。基于这 101 个 item 的排序结果，可以计算 Hit@10 和 NDCG@10。

### 4.5 推荐性能

**表3：推荐性能。** 每行中表现最佳的方法加粗显示，表现第二的方法加下划线。对非神经网络方法和神经网络方法的改进分别显示在最后两列中。

| 数据集 | 指标 | PopRec | BPR | FMC | FPMC | TransRec | GRU4Rec | GRU4Rec+ | Caser | SASRec | vs. (a)-(e) | vs. (f)-(h) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Beauty | Hit@10 | 0.4003 | 0.3775 | 0.3771 | 0.4310 | 0.4607 | 0.2125 | 0.3949 | 0.4264 | **0.4854** | 5.4% | 13.8% |
| Beauty | NDCG@10 | 0.2277 | 0.2183 | 0.2477 | 0.2891 | 0.3020 | 0.1203 | 0.2556 | 0.2547 | **0.3219** | 6.6% | 25.9% |
| Games | Hit@10 | 0.4724 | 0.4853 | 0.6358 | 0.6802 | 0.6838 | 0.2938 | 0.6599 | 0.5282 | **0.7410** | 8.5% | 12.3% |
| Games | NDCG@10 | 0.2779 | 0.2875 | 0.4456 | 0.4680 | 0.4557 | 0.1837 | 0.4759 | 0.3214 | **0.5360** | 14.5% | 12.6% |
| Steam | Hit@10 | 0.7172 | 0.7061 | 0.7731 | 0.7710 | 0.7624 | 0.4190 | 0.8018 | 0.7874 | **0.8729** | 13.2% | 8.9% |
| Steam | NDCG@10 | 0.4535 | 0.4436 | 0.5193 | 0.5011 | 0.4852 | 0.2691 | 0.5595 | 0.5381 | **0.6306** | 21.4% | 12.7% |
| ML-1M | Hit@10 | 0.4329 | 0.5781 | 0.6986 | 0.7599 | 0.6413 | 0.5581 | 0.7501 | 0.7886 | **0.8245** | 8.5% | 4.6% |
| ML-1M | NDCG@10 | 0.2377 | 0.3287 | 0.4676 | 0.5176 | 0.3969 | 0.3381 | 0.5513 | 0.5538 | **0.5905** | 14.1% | 6.6% |

表3 展示了所有方法在四个数据集上的推荐性能（RQ1）。通过观察所有数据集上的第二佳方法，可以看出一股模式：非神经网络方法（即 (a)-(e)）在稀疏数据集上表现更好，而神经网络方法（即 (f)-(h)）在稠密数据集上表现更好。这大概是因为神经网络方法有更多参数来捕获高阶转移（即它们表达力强但容易过拟合），而精心设计但更简单的模型在高稀疏度设置中更有效。

我们的方法 SASRec 在稀疏和稠密数据集上均优于所有基线方法，相对于最强基线平均 Hit Rate 提升 6.9%，NDCG 提升 9.6%。一个可能的原因是，我们的模型可以在不同数据集上自适应地关注不同范围内的 item（例如在稀疏数据集上仅关注前一个 item，在稠密数据集上关注更多 item）。我们将在第 4.8 节进一步分析这种行为。

在图2中，我们还分析了关键超参数——latent 维度 $d$ 的影响，展示了 $d$ 从 10 到 50 变化时所有方法的 NDCG@10。我们看到，我们的模型通常受益于更大的 latent 维度。对于所有数据集，当 $d \geq 40$ 时，我们的模型取得了令人满意的性能。

### 4.6 消融研究

由于我们的架构中有许多组件，我们通过消融研究分析它们的影响（RQ2）。

**表4：消融分析（NDCG@10）在四个数据集上。** 优于默认版本的性能加粗显示。" $\downarrow$ "表示性能严重下降（超过 10%）。

| 架构 | Beauty | Games | Steam | ML-1M |
|---|---|---|---|---|
| (0) 默认 | 0.3142 | 0.5360 | 0.6306 | 0.5905 |
| (1) 移除 PE | 0.3183 | 0.5301 | 0.6036 | 0.5772 |
| (2) 非共享 IE | 0.2437 $\downarrow$ | 0.4266 $\downarrow$ | 0.4472 $\downarrow$ | 0.4557 $\downarrow$ |
| (3) 移除 RC | 0.2591 $\downarrow$ | 0.4303 $\downarrow$ | 0.5693 | 0.5535 |
| (4) 移除 Dropout | 0.2436 $\downarrow$ | 0.4375 $\downarrow$ | 0.5959 | 0.5801 |
| (5) 0 块 (b=0) | 0.2620 $\downarrow$ | 0.4745 $\downarrow$ | 0.5588 $\downarrow$ | 0.4830 $\downarrow$ |
| (6) 1 块 (b=1) | 0.3066 | 0.5408 | 0.6202 | 0.5653 |
| (7) 3 块 (b=3) | 0.3078 | 0.5312 | 0.6275 | 0.5931 |
| (8) 多头 | 0.3080 | 0.5311 | 0.6272 | 0.5885 |

我们介绍各变体并分别分析其影响：
1. **移除 PE（位置嵌入）：** 没有位置嵌入 $P$ 时，每个 item 上的注意力权重仅取决于 item 嵌入。模型基于用户过去的行为进行推荐，但行为的顺序无关紧要。该变体在最稀疏的数据集（Beauty）上表现优于默认模型，但在其他更稠密的数据集上表现更差。
2. **非共享 IE（item 嵌入）：** 使用两个 item 嵌入一直会损害性能，可能是由于过拟合。
3. **移除 RC（残差连接）：** 没有残差连接时，性能显著下降。可能是因为低层信息（如最后一个 item 的嵌入和第一个块的输出）无法轻易传播到最终层。
4. **移除 Dropout：** Dropout 可以有效地正则化模型以获得更好的测试性能，尤其是在稀疏数据集上。结果还表明，在稠密数据集上，过拟合问题不那么严重。
5. **– (7) 块的数量：** 零个块时结果较差（模型将仅依赖于最后一个 item）。一个块表现尚可，但两个块在稠密数据集上提升了性能。三个块取得了与默认模型相似的性能。
6. **多头注意：** 使用两个头时性能一致地略差于单头注意力。这可能是因为我们问题中的 $d$ 较小（Transformer 中 $d = 512$ ），不适合分解为更小的子空间。

### 4.7 训练效率与可扩展性

我们评估训练效率的两个方面（RQ3）：训练速度（一个 epoch 所需的训练时间）和收敛时间（达到满意性能所需的时间）。我们还考察模型在最大长度 $n$ 方面的可扩展性。所有实验均在单个 GTX-1080 Ti GPU 上进行。

**训练效率：** 图3展示了基于深度学习的方法在 GPU 加速下的效率。由于性能较差，省略了 GRU4Rec。为公平比较，Caser 和 GRU4Rec+ 有两种训练选项：使用完整训练数据，或仅使用最近 200 个动作（如 SASRec 中）。在计算速度方面，SASRec 每个 epoch 仅需 1.7 秒用于模型更新，比 Caser（19.1s/epoch）快 11 倍以上，比 GRU4Rec+（30.7s/epoch）快 18 倍。我们还看到，SASRec 在 ML-1M 上大约在 350 秒内收敛到最优性能，而其他模型需要更长的时间。我们还发现，使用完整数据可以使 Caser 和 GRU4Rec+ 获得更好的性能。

**可扩展性：** 与标准 MF 方法一样，SASRec 在用户总数、item 总数和行为数方面呈线性扩展。一个潜在的可扩展性问题是最大长度 $n$ ，但计算可以借助 GPU 有效并行化。这里我们测量了不同 $n$ 下 SASRec 的训练时间和性能，通过实验研究其可扩展性，并分析它是否能处理大多数情况下的序列推荐。

**表5：可扩展性：不同最大长度 $n$ 下在 ML-1M 上的性能和训练时间。**

| $n$ | 10 | 50 | 100 | 200 | 300 | 400 | 500 | 600 |
|---|---|---|---|---|---|---|---|---|
| 时间 (s) | 75 | 101 | 157 | 341 | 613 | 965 | 1406 | 1895 |
| NDCG@10 | 0.480 | 0.557 | 0.571 | 0.587 | 0.593 | 0.594 | 0.596 | 0.595 |

 $n$ 越大性能越好，直到 $n$ 约为 500 时性能饱和（可能是因为 99.8% 的行为已被覆盖）。然而，即使在 $n = 600$ 时，模型也可以在 2,000 秒内完成训练，这仍然比 Caser 和 GRU4Rec+ 更快。因此，我们的模型可以轻松扩展到用户序列长达数百个动作的场景，这适用于典型的评论和购买数据集。

### 4.8 注意力权重可视化

回顾一下，在时间步 $t$ ，我们模型中的自注意力机制根据位置嵌入和 item 嵌入，自适应地为前 $t$ 个 item 分配权重。为回答 RQ4，我们检查所有训练序列，通过展示在位置和 item 上的平均注意力权重，试图揭示有意义的模式。

**位置上的注意力：** 图4显示了四个热力图，展示了最后 15 个时间步上在最后 15 个位置的平均注意力权重。注意当我们计算平均权重时，分母是有效权重的数量，以避免短序列中填充 item 的影响。

我们考虑热力图之间的几个比较：
- **(a) vs. (c)：** 这一比较表明，在稀疏数据集 Beauty 上，模型倾向于关注更近期的 item；在稠密数据集 ML-1M 上，则关注较早期的 item。这是我们的模型能够自适应处理稀疏和稠密数据集的关键因素，而现有方法往往只擅长处理其中一种。
- **(b) vs. (c)：** 这一比较显示了使用位置嵌入（PE）的效果。没有位置嵌入时，注意力权重在之前的 item 上基本均匀分布，而默认模型（c）对位置更敏感，倾向于关注近期 item。
- **(c) vs. (d)：** 由于我们的模型是分层的，这显示了注意力在不同块之间的变化。显然，高层注意力倾向于更关注近期位置。大概是因为第一个自注意力块已经考虑了所有之前的 item，第二个块不需要关注更远的位置。

总体而言，可视化结果表明，我们的自注意力机制的行为是自适应、位置感知和分层的。

**item 之间的注意力：** 展示几个精心挑选的 item 之间的注意力权重可能不具有统计意义。为进行更广泛的比较，我们使用 MovieLens-1M（每部电影有多个类别），随机选择两个不相交的集合，每个集合包含来自 4 个类别（科幻片、爱情片、动画片和恐怖片）的 200 部电影。第一个集合用作查询，第二个集合用作键。图5展示了两个集合之间平均注意力权重的热力图。我们可以看到热力图近似为一个块对角矩阵，这意味着注意力机制能够识别相似的 item（例如共享同一类别的 item），并倾向于在它们之间分配更大的权重（而无需事先知道类别信息）。

## 5. 结论

在这项工作中，我们提出了一种新颖的基于自注意力的序列模型 SASRec，用于下一个 item 推荐。SASRec 对整个用户序列进行建模（无需任何循环或卷积操作），并自适应地考虑已消费的 item 进行预测。在稀疏和稠密数据集上广泛的经验结果表明，我们的模型优于最先进的基线方法，并且比基于 CNN/RNN 的方法快一个数量级。未来，我们计划通过融入丰富的上下文信息（如停留时间、行为类型、位置、设备等）来扩展模型，并研究处理非常长序列（如点击）的方法。

## 参考文献

[1] S. Rendle, C. Freudenthaler, and L. Schmidt-Thieme, "Factorizing personalized Markov chains for next-basket recommendation," in WWW, 2010.
[2] B. Hidasi, A. Karatzoglou, L. Baltrunas, and D. Tikk, "Session-based recommendations with recurrent neural networks," in ICLR, 2016.
[3] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin, "Attention is all you need," in NIPS, 2017.
[4] Y. Hu, Y. Koren, and C. Volinsky, "Collaborative filtering for implicit feedback datasets," in ICDM, 2008.
[5] S. Rendle, C. Freudenthaler, Z. Gantner, and L. Schmidt-Thieme, "BPR: Bayesian personalized ranking from implicit feedback," in UAI, 2009.
[6] F. Ricci, L. Rokach, B. Shapira, and P. Kantor, Recommender systems handbook. Springer US, 2011.
[7] Y. Koren and R. Bell, "Advances in collaborative filtering," in Recommender Systems Handbook. Springer, 2011.
[8] S. Kabbur, X. Ning, and G. Karypis, "FISM: Factored item similarity models for top-n recommender systems," in SIGKDD, 2013.
[9] S. Zhang, L. Yao, and A. Sun, "Deep learning based recommender system: A survey and new perspectives," arXiv, vol. abs/1707.07435, 2017.
[10] S. Wang, Y. Wang, J. Tang, K. Shu, S. Ranganath, and H. Liu, "What your images reveal: Exploiting visual contents for point-of-interest recommendation," in WWW, 2017.
[11] W. Kang, C. Fang, Z. Wang, and J. McAuley, "Visually-aware fashion recommendation and design with generative image models," in ICDM, 2017.
[12] H. Wang, N. Wang, and D. Yeung, "Collaborative deep learning for recommender systems," in SIGKDD, 2015.
[13] D. H. Kim, C. Park, J. Oh, S. Lee, and H. Yu, "Convolutional matrix factorization for document context-aware recommendation," in RecSys, 2016.
[14] X. He, L. Liao, H. Zhang, L. Nie, X. Hu, and T. Chua, "Neural collaborative filtering," in WWW, 2017.
[15] S. Sedhain, A. K. Menon, S. Sanner, and L. Xie, "AutoRec: Autoencoders meet collaborative filtering," in WWW, 2015.
[16] Y. Koren, "Collaborative filtering with temporal dynamics," Communications of the ACM, 2010.
[17] C. Wu, A. Ahmed, A. Beutel, A. J. Smola, and H. Jing, "Recurrent recommender networks," in WSDM, 2017.
[18] L. Xiong, X. Chen, T.-K. Huang, J. Schneider, and J. G. Carbonell, "Temporal collaborative filtering with Bayesian probabilistic tensor factorization," in SDM, 2010.
[19] R. He, W. Kang, and J. McAuley, "Translation-based recommendation," in RecSys, 2017.
[20] R. He, C. Fang, Z. Wang, and J. McAuley, "Vista: A visually, socially, and temporally-aware model for artistic recommendation," in RecSys, 2016.
[21] R. He and J. McAuley, "Fusing similarity models with Markov chains for sparse sequential recommendation," in ICDM, 2016.
[22] J. Tang and K. Wang, "Personalized top-n sequential recommendation via convolutional sequence embedding," in WSDM, 2018.
[23] H. Jing and A. J. Smola, "Neural survival recommender," in WSDM, 2017.
[24] Q. Liu, S. Wu, D. Wang, Z. Li, and L. Wang, "Context-aware sequential recommendation," in ICDM, 2016.
[25] A. Beutel, P. Covington, S. Jain, C. Xu, J. Li, V. Gatto, and E. H. Chi, "Latent cross: Making use of context in recurrent recommender systems," in WSDM, 2018.
[26] B. Hidasi and A. Karatzoglou, "Recurrent neural networks with top-k gains for session-based recommendations," CoRR, vol. abs/1706.03847, 2017.
[27] K. Xu, J. Ba, R. Kiros, K. Cho, A. C. Courville, R. Salakhutdinov, R. S. Zemel, and Y. Bengio, "Show, attend and tell: Neural image caption generation with visual attention," in ICML, 2015.
[28] D. Bahdanau, K. Cho, and Y. Bengio, "Neural machine translation by jointly learning to align and translate," in ICLR, 2015.
[29] J. Chen, H. Zhang, X. He, L. Nie, W. Liu, and T. Chua, "Attentive collaborative filtering: Multimedia recommendation with item- and component-level attention," in SIGIR, 2017.
[30] J. Xiao, H. Ye, X. He, H. Zhang, F. Wu, and T. Chua, "Attentional factorization machines: Learning the weight of feature interactions via attention networks," in IJCAI, 2017.
[31] S. Wang, L. Hu, L. Cao, X. Huang, D. Lian, and W. Liu, "Attention-based transactional context embedding for next-item recommendation," in AAAI, 2018.
[32] Y. Wu, M. Schuster, Z. Chen, Q. V. Le, M. Norouzi, W. Macherey, M. Krikun, Y. Cao, Q. Gao, K. Macherey et al., "Google's neural machine translation system: Bridging the gap between human and machine translation," arXiv preprint arXiv:1609.08144, 2016.
[33] J. Zhou, Y. Cao, X. Wang, P. Li, and W. Xu, "Deep recurrent models with fast-forward connections for neural machine translation," TACL, 2016.
[34] M. D. Zeiler and R. Fergus, "Visualizing and understanding convolutional networks," in ECCV, 2014.
[35] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in CVPR, 2016.
[36] L. J. Ba, R. Kiros, and G. E. Hinton, "Layer normalization," CoRR, vol. abs/1607.06450, 2016.
[37] S. Ioffe and C. Szegedy, "Batch normalization: Accelerating deep network training by reducing internal covariate shift," in ICML, 2015.
[38] N. Srivastava, G. E. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov, "Dropout: A simple way to prevent neural networks from overfitting," JMLR, 2014.
[39] D. Warde-Farley, I. J. Goodfellow, A. C. Courville, and Y. Bengio, "An empirical analysis of dropout in piecewise linear networks," CoRR, vol. abs/1312.6197, 2013.
[40] Y. Koren, R. Bell, and C. Volinsky, "Matrix factorization techniques for recommender systems," Computer, 2009.
[41] D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," in ICLR, 2015.
[42] D. Povey, H. Hadian, P. Ghahremani, K. Li, and S. Khudanpur, "A time-restricted self-attention layer for ASR," in ICASSP, 2018.
[43] P. Wang, J. Guo, Y. Lan, J. Xu, S. Wan, and X. Cheng, "Learning hierarchical representation model for next basket recommendation," in SIGIR, 2015.
[44] S. Bai, J. Z. Kolter, and V. Koltun, "An empirical evaluation of generic convolutional and recurrent networks for sequence modeling," CoRR, vol. abs/1803.01271, 2018.
[45] S. Hochreiter, Y. Bengio, P. Frasconi, J. Schmidhuber et al., "Gradient flow in recurrent nets: the difficulty of learning long-term dependencies," 2001.
[46] J. J. McAuley, C. Targett, Q. Shi, and A. van den Hengel, "Image-based recommendations on styles and substitutes," in SIGIR, 2015.
[47] S. Feng, X. Li, Y. Zeng, G. Cong, Y. M. Chee, and Q. Yuan, "Personalized ranking metric embedding for next new POI recommendation," in IJCAI, 2015.
[48] Y. Koren, "Factorization meets the neighborhood: A multifaceted collaborative filtering model," in SIGKDD, 2008.
