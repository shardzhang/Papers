# Attention Is All You Need

> Ashish Vaswani\* | Google Brain | avaswani@google.com
> Noam Shazeer\* | Google Brain | noam@google.com
> Niki Parmar\* | Google Research | nikip@google.com
> Jakob Uszkoreit\* | Google Research | usz@google.com
> Llion Jones\* | Google Research | llion@google.com
> Aidan N. Gomez\*† | University of Toronto | aidan@cs.toronto.edu
> Łukasz Kaiser\* | Google Brain | lukaszkaiser@google.com
> Illia Polosukhin\*‡ | illia.polosukhin@gmail.com

本文提出 Transformer，一种完全基于注意力机制、摒弃循环和卷积的全新网络架构。在两套机器翻译任务上的实验表明，该模型在质量上更优、可并行性更强且训练时间大幅缩短。

- 提出 Transformer —— 首个完全依靠自注意力计算输入/输出表示的序列转导模型
- WMT 2014 英 $\to$ 德翻译任务取得 28.4 BLEU，超越此前最优结果（含集成模型）2 BLEU 以上
- WMT 2014 英 $\to$ 法翻译任务以 8 块 GPU 训练 3.5 天即达到 41.8 BLEU 的单模型 SOTA
- 成功将 Transformer 应用于英语成分句法分析，验证了其良好的泛化能力

---

## 摘要

主流的序列转导模型基于复杂的循环或卷积神经网络，包含编码器和解码器。性能最佳的模型还通过注意力机制连接编码器和解码器。我们提出了一种新颖的简单网络架构 —— Transformer，它完全基于注意力机制，彻底摒弃了循环和卷积。在两项机器翻译任务上的实验表明，这些模型在质量上更优，同时具有更强的可并行性，且训练时间大幅减少。我们的模型在 WMT 2014 英译德任务上达到了 28.4 BLEU，比现有最佳结果（包括集成模型）提高了 2 BLEU 以上。在 WMT 2014 英译法任务上，我们的模型在 8 块 GPU 上训练 3.5 天后，建立了 41.8 BLEU 的新的单模型最佳 BLEU 分数，训练成本仅为文献中最佳模型的一小部分。我们还将 Transformer 成功应用于英语成分句法分析（无论训练数据规模大小），表明它能够很好地泛化到其他任务。

## 1 引言

循环神经网络，特别是长短期记忆 [1] 和门控循环 [2] 神经网络，已经成为序列建模和转导问题（如语言建模和机器翻译 [3, 4, 5]）中公认的最先进方法。此后，大量工作不断突破循环语言模型和编码器-解码器架构的边界 [6, 7, 8]。

循环模型通常沿着输入和输出序列的符号位置分解计算。通过将位置对齐到计算时间步，它们生成隐藏状态 $h_t$ 的序列，作为前一个隐藏状态 $h_{t-1}$ 和位置 $t$ 的输入的函数。这种固有的顺序特性阻止了训练样本内部的并行化，这在序列长度较长时变得至关重要，因为内存限制限制了跨样本的批处理。最近的工作通过分解技巧 [9] 和条件计算 [10] 在计算效率上取得了显著改进，同时后一种方法还提升了模型性能。然而，顺序计算的根本约束仍然存在。

注意力机制已成为各种任务中引人入胜的序列建模和转导模型不可或缺的组成部分，它允许建模依赖关系而无需考虑其在输入或输出序列中的距离 [4, 11]。然而，除了少数情况 [12] 之外，这类注意力机制通常与循环网络结合使用。

在这项工作中，我们提出了 Transformer，一种摒弃循环、完全依赖注意力机制来捕捉输入和输出之间全局依赖关系的模型架构。Transformer 允许更高程度的并行化，并且在 8 块 P100 GPU 上仅需训练短短 12 小时就能达到翻译质量的新高度。

## 2 背景

减少顺序计算的目标也是 Extended Neural GPU [13]、ByteNet [14] 和 ConvS2S [15] 的基础，它们都使用卷积神经网络作为基本构建块，并行计算所有输入和输出位置的隐藏表示。在这些模型中，关联两个任意输入或输出位置信号所需的操作数量随着位置之间的距离而增长（ConvS2S 为线性，ByteNet 为对数）。这使得学习 distant 位置之间的依赖关系更加困难 [16]。在 Transformer 中，这被减少为常数数量的操作，但代价是由于对注意力加权位置进行平均而导致有效分辨率降低，我们通过第 3.2 节描述的多头注意力来抵消这种影响。

自注意力（self-attention），有时称为内部注意力（intra-attention），是一种将单个序列的不同位置关联起来以计算序列表示的注意力机制。自注意力已成功应用于多种任务，包括阅读理解、抽象式摘要、文本蕴含以及学习独立于任务的句子表示 [17, 12, 18, 19]。

端到端记忆网络基于循环注意力机制而非序列对齐的循环，并已被证明在简单语言问答和语言建模任务上表现良好 [20]。

据我们所知，Transformer 是第一个完全依靠自注意力来计算其输入和输出表示、而不使用序列对齐的 RNN 或卷积的转导模型。在接下来的章节中，我们将描述 Transformer，阐述自注意力的动机，并讨论其相对于 [21, 14] 和 [15] 等模型的优势。

## 3 模型架构

大多数有竞争力的神经序列转导模型都采用编码器-解码器结构 [5, 3, 4]。在这里，编码器将输入符号表示序列 $(x_1, ..., x_n)$ 映射为连续表示序列 $z = (z_1, ..., z_n)$ 。给定 $z$ ，解码器然后逐个元素地生成输出符号序列 $(y_1, ..., y_m)$ 。在每个步骤中，模型是自回归的 [22]，在生成下一个符号时会将先前生成的符号作为额外输入。

Transformer 遵循这种总体架构，对编码器和解码器都使用堆叠的自注意力和逐位置的全连接层，分别如图 1 的左半部分和右半部分所示。

![Figure 1: The Transformer - model architecture.](figure1.png)

**图 1：Transformer —— 模型架构。**

### 3.1 编码器和解码器堆栈

**编码器：** 编码器由 $N = 6$ 个相同层的堆栈组成。每一层有两个子层。第一个是多头自注意力机制，第二个是简单的逐位置全连接前馈网络。我们在每个子层周围采用残差连接 [23]，之后进行层归一化 [24]。也就是说，每个子层的输出是 $\text{LayerNorm}(x + \text{Sublayer}(x))$ ，其中 $\text{Sublayer}(x)$ 是子层本身实现的函数。为便于这些残差连接，模型中的所有子层以及嵌入层都产生维度 $d_{\text{model}} = 512$ 的输出。

**解码器：** 解码器也由 $N = 6$ 个相同层的堆栈组成。除了每个编码器层中的两个子层之外，解码器还插入了第三个子层，该子层对编码器堆栈的输出执行多头注意力。与编码器类似，我们在每个子层周围采用残差连接，之后进行层归一化。我们还修改了解码器堆栈中的自注意力子层，以防止位置关注后续位置。这种掩码机制，加上输出嵌入偏移一个位置的事实，确保了位置 $i$ 的预测只能依赖于位置小于 $i$ 的已知输出。

### 3.2 注意力

注意力函数可以描述为将查询和一组键-值对映射到输出，其中查询、键、值和输出都是向量。输出计算为值的加权和，其中分配给每个值的权重由查询与对应键的兼容性函数计算得出。

![Figure 2: (left) Scaled Dot-Product Attention. (right) Multi-Head Attention consists of several attention layers running in parallel.](figure2.png)

**图 2：（左）缩放点积注意力。（右）多头注意力由多个并行运行的注意力层组成。**

#### 3.2.1 缩放点积注意力

我们将我们的特定注意力称为"缩放点积注意力"（图 2）。输入包括维度为 $d_k$ 的查询和键，以及维度为 $d_v$ 的值。我们计算查询与所有键的点积，每个除以 $\sqrt{d_k}$ ，然后应用 softmax 函数以获得值上的权重。

在实际应用中，我们同时计算一组查询的注意力函数，将这些查询打包成一个矩阵 $Q$ 。键和值也分别打包成矩阵 $K$ 和 $V$ 。我们计算输出矩阵为：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V \qquad (1)
$$

两种最常用的注意力函数是加性注意力 [4] 和点积（乘法）注意力。点积注意力与我们的算法相同，只是没有缩放因子 $1/\sqrt{d_k}$ 。加性注意力使用带有单隐藏层的前馈网络来计算兼容性函数。虽然两者在理论复杂度上相似，但点积注意力在实践中更快、更节省空间，因为它可以使用高度优化的矩阵乘法代码实现。

虽然对于较小的 $d_k$ 值，两种机制表现相似，但对于较大的 $d_k$ 值，加性注意力优于未缩放的點積注意力 [25]。我们推测，对于较大的 $d_k$ ，点积的幅度会变得很大，将 softmax 函数推入梯度极小的区域 ¹。为了抵消这一影响，我们用 $1/\sqrt{d_k}$ 缩放点积。

> ¹ 为了说明为什么点积会变大，假设 $q$ 和 $k$ 的分量是均值为 0、方差为 1 的独立随机变量。那么它们的点积 $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$ 的均值为 0，方差为 $d_k$ 。

#### 3.2.2 多头注意力

我们发现，与其使用 $d_{\text{model}}$ 维的键、值和查询执行单个注意力函数，不如将查询、键和值用不同的学习到的线性投影分别线性投影 $h$ 次到 $d_k$ 、 $d_k$ 和 $d_v$ 维。在这些查询、键和值的每个投影版本上，我们并行执行注意力函数，产生 $d_v$ 维的输出值。这些值被拼接起来并再次投影，得到最终值，如图 2 所示。

多头注意力使得模型能够共同关注来自不同位置的不同表示子空间的信息。使用单个注意力头时，平均操作会抑制这一点。

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h) W^O
$$

其中 $\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$

投影是参数矩阵 $W_i^Q $ \in $ \mathbb{R}^{d_{\text{model}} $ \times $ d_k}$ 、 $W_i^K $ \in $ \mathbb{R}^{d_{\text{model}} $ \times $ d_k}$ 、 $W_i^V $ \in $ \mathbb{R}^{d_{\text{model}} $ \times $ d_v}$ 和 $W^O $ \in $ \mathbb{R}^{hd_v $ \times $ d_{\text{model}}}$ 。

在这项工作中，我们使用 $h = 8$ 个并行注意力层（或称头）。对每个头我们使用 $d_k = d_v = d_{\text{model}} / h = 64$ 。由于每个头的维度降低，总的计算成本与全维度的单头注意力相似。

#### 3.2.3 注意力在我们模型中的应用

Transformer 以三种不同方式使用多头注意力：

- 在"编码器-解码器注意力"层中，查询来自前一个解码器层，记忆键和值来自编码器的输出。这允许解码器中的每个位置关注输入序列中的所有位置。这模仿了序列到序列模型（如 [6, 4, 15]）中典型的编码器-解码器注意力机制。
- 编码器包含自注意力层。在自注意力层中，所有的键、值和查询都来自同一个地方，在这里是编码器中前一层的输出。编码器中的每个位置都可以关注编码器前一层中的所有位置。
- 类似地，解码器中的自注意力层允许解码器中的每个位置关注解码器中到该位置为止的所有位置。我们需要防止解码器中的向左信息流以保持自回归特性。我们通过在缩放点积注意力内部屏蔽（设置为 $-$ \in $fty$ ）softmax 输入中所有对应非法连接的值来实现这一点。见图 2。

### 3.3 逐位置前馈网络

除了注意力子层之外，编码器和解码器中的每一层都包含一个全连接前馈网络，该网络分别且相同地应用于每个位置。它由两个线性变换组成，中间有一个 ReLU 激活。

$$
\text{FFN}(x) = \max(0, xW_1 + b_1) W_2 + b_2 \qquad (2)
$$

虽然线性变换在不同位置上是相同的，但它们在层与层之间使用不同的参数。另一种描述方式是将其视为两个卷积核大小为 1 的卷积。

输入和输出的维度为 $d_{\text{model}} = 512$ ，内层的维度为 $d_{\text{ff}} = 2048$ 。

### 3.4 嵌入和 Softmax

与其他序列转导模型类似，我们使用学习到的嵌入将输入 token 和输出 token 转换为维度 $d_{\text{model}}$ 的向量。我们还使用通常的学习到的线性变换和 softmax 函数将解码器输出转换为预测的下一个 token 的概率。在我们的模型中，我们在两个嵌入层和 pre-softmax 线性变换之间共享相同的权重矩阵，类似于 [26]。在嵌入层中，我们将这些权重乘以 $\sqrt{d_{\text{model}}}$ 。

### 3.5 位置编码

由于我们的模型不包含循环也不包含卷积，为了使模型能够利用序列的顺序信息，我们必须注入一些关于序列中 token 的相对或绝对位置的信息。为此，我们在编码器和解码器堆栈底部的输入嵌入中添加"位置编码"。位置编码与嵌入具有相同的维度 $d_{\text{model}}$ ，因此两者可以相加。位置编码有很多选择，可以是学习得到的，也可以是固定不变的 [15]。

在这项工作中，我们使用不同频率的正弦和余弦函数：

$$
PE_{(pos, 2i)} = \sin(pos / 10000^{2i / d_{\text{model}}})
$$

$$
PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i / d_{\text{model}}})
$$

其中 $pos$ 是位置， $i$ 是维度。也就是说，位置编码的每个维度对应一个正弦波。波长构成从 $2\pi$ 到 $10000 \cdot 2\pi$ 的几何级数。我们选择这个函数是因为我们假设它能让模型容易地学会通过相对位置进行注意力，因为对于任何固定的偏移量 $k$ ， $PE_{pos+k}$ 都可以表示为 $PE_{pos}$ 的线性函数。

我们还尝试使用学习到的位置嵌入 [15] 代替，发现两个版本产生的结果几乎相同（见表 3 行 (E)）。我们选择正弦版本是因为它可能允许模型外推到训练期间遇到的序列长度更长的序列。

## 4 为什么选择自注意力

在本节中，我们将自注意力层与通常用于将一个可变长度符号表示序列 $(x_1, ..., x_n)$ 映射为另一等长序列 $(z_1, ..., z_n)$ （其中 $x_i, z_i $ \in $ \mathbb{R}^d$ ）的循环层和卷积层进行比较，例如典型序列转导编码器或解码器中的隐藏层。为说明我们使用自注意力的动机，我们考虑三个需求。

一是每层的总计算复杂度。二是可并行化的计算量，以所需的最少顺序操作数衡量。

三是网络中长程依赖之间的路径长度。学习长程依赖是许多序列转导任务中的关键挑战。影响学习此类依赖能力的一个关键因素是前向和反向信号在网络中必须遍历的路径长度。输入和输出序列中任意位置组合之间的路径越短，学习长程依赖就越容易 [16]。因此，我们还比较了由不同层类型组成的网络中任意两个输入和输出位置之间的最大路径长度。

如表 1 所示，自注意力层以常数数量的顺序执行操作连接所有位置，而循环层需要 $O(n)$ 个顺序操作。就计算复杂度而言，当序列长度 $n$ 小于表示维度 $d$ 时，自注意力层比循环层更快，这在机器翻译的最先进模型所使用的句子表示（如 word-piece [6] 和 byte-pair [27] 表示）中通常是这种情况。为了改进涉及非常长序列的任务的计算性能，可以将自注意力限制为仅考虑以相应输出位置为中心、大小为 $r$ 的邻域内的输入序列。这将增加最大路径长度为 $O(n/r)$ 。我们计划在未来工作中进一步研究这种方法。

**表 1：不同层类型的最大路径长度、每层复杂度和最少顺序操作数。 $n$ 是序列长度， $d$ 是表示维度， $k$ 是卷积的卷积核大小， $r$ 是受限自注意力中邻域的大小。**

| 层类型 | 每层复杂度 | 顺序操作数 | 最大路径长度 |
| :--- | :--- | :--- | :--- |
| 自注意力 | $O(n^2 \cdot d)$ | $O(1)$ | $O(1)$ |
| 循环 | $O(n \cdot d^2)$ | $O(n)$ | $O(n)$ |
| 卷积 | $O(k \cdot n \cdot d^2)$ | $O(1)$ | $O(\log_k(n))$ |
| 自注意力（受限） | $O(r \cdot n \cdot d)$ | $O(1)$ | $O(n/r)$ |

卷积核宽度 $k < n$ 的单卷积层不能连接所有输入和输出位置对。要做到这一点，在连续卷积核的情况下需要 $O(n/k)$ 个卷积层的堆栈，或在膨胀卷积的情况下需要 $O(\log_k(n))$ 个卷积层 [14]，从而增加了网络中任意两个位置之间最长路径的长度。卷积层通常比循环层更昂贵，开销为因子 $k$ 。然而，可分离卷积 [28] 显著降低了复杂度，降至 $O(k \cdot n \cdot d + n \cdot d^2)$ 。但即使 $k = n$ ，可分离卷积的复杂度也等于自注意力层和逐点前馈层的组合，这正是我们在模型中采用的方法。

作为附带好处，自注意力可能产生更具可解释性的模型。我们检查了模型中的注意力分布，并在附录中展示和讨论了示例。不仅单个注意力头清楚地学会了执行不同的任务，许多头还表现出与句子的句法和语义结构相关的行为。

## 5 训练

本节描述我们模型的训练方案。

### 5.1 训练数据和批处理

我们在标准的 WMT 2014 英-德数据集上训练，该数据集包含约 450 万个句子对。句子使用 byte-pair 编码 [25] 编码，具有约 37000 个 token 的共享源-目标词汇表。对于英-法，我们使用了更大的 WMT 2014 英-法数据集，包含 3600 万个句子，并将 token 分割为 32000 个 word-piece 词汇 [6]。句子对按近似序列长度分组为批次。每个训练批次包含一组句子对，包含约 25000 个源 token 和 25000 个目标 token。

### 5.2 硬件和时间安排

我们在一台配备 8 块 NVIDIA P100 GPU 的机器上训练模型。对于使用本文所述超参数的 base 模型，每个训练步骤大约需要 0.4 秒。我们总共训练 base 模型 100,000 步或 12 小时。对于 big 模型（见表 3 底部行），每步时间为 1.0 秒。big 模型训练了 300,000 步（3.5 天）。

### 5.3 优化器

我们使用 Adam 优化器 [29]，参数为 $\beta_1 = 0.9$ ， $\beta_2 = 0.98$ 和 $\epsilon = 10^{-9}$ 。我们在训练过程中变化学习率，公式如下：

$$
\text{lrate} = d_{\text{model}}^{-0.5} \cdot \min(\text{step\_num}^{-0.5}, \text{step\_num} \cdot \text{warmup\_steps}^{-1.5}) \qquad (3)
$$

这相当于在前 $\text{warmup\_steps}$ 步中线性增加学习率，然后按步数的平方根反比递减。我们使用 $\text{warmup\_steps} = 4000$ 。

### 5.4 正则化

我们在训练中采用三种类型的正则化：

**残差 Dropout** 我们对每个子层的输出应用 dropout [30]，然后再将其添加到子层输入并进行归一化。此外，我们对编码器和解码器堆栈中的嵌入和位置编码之和应用 dropout。对于 base 模型，我们使用 $P_{\text{drop}} = 0.1$ 的比率。

**标签平滑** 在训练过程中，我们使用了值为 $\epsilon_{\text{ls}} = 0.1$ 的标签平滑 [31]。这会损害困惑度，因为模型学会了更加不确定，但提高了准确性和 BLEU 分数。

## 6 结果

### 6.1 机器翻译

**表 2：Transformer 在英译德和英译法 newstest2014 测试集上以更低的训练成本取得比此前最先进模型更好的 BLEU 分数。**

| 模型 | BLEU (EN-DE) | BLEU (EN-FR) | 训练成本 (FLOPs) EN-DE | 训练成本 (FLOPs) EN-FR |
| :--- | :--- | :--- | :--- | :--- |
| ByteNet [14] | 23.75 | | | |
| Deep-Att + PosUnk [32] | | 39.2 | | $1.0 \cdot 10^{20}$ |
| GNMT + RL [6] | 24.6 | 39.92 | $2.3 \cdot 10^{19}$ | $1.4 \cdot 10^{20}$ |
| ConvS2S [15] | 25.16 | 40.46 | $9.6 \cdot 10^{18}$ | $1.5 \cdot 10^{20}$ |
| MoE [10] | 26.03 | 40.56 | $2.0 \cdot 10^{19}$ | $1.2 \cdot 10^{20}$ |
| Deep-Att + PosUnk Ensemble [32] | | 40.4 | | $8.0 \cdot 10^{20}$ |
| GNMT + RL Ensemble [6] | 26.30 | 41.16 | $1.8 \cdot 10^{20}$ | $1.1 \cdot 10^{21}$ |
| ConvS2S Ensemble [15] | 26.36 | 41.29 | $7.7 \cdot 10^{19}$ | $1.2 \cdot 10^{21}$ |
| Transformer (base model) | 27.3 | 38.1 | | $3.3 \cdot 10^{18}$ |
| Transformer (big) | 28.4 | 41.8 | $2.3 \cdot 10^{19}$ | |

在 WMT 2014 英译德翻译任务中，big transformer 模型（表 2 中的 Transformer (big)）超过此前最佳报告模型（包括集成模型）2.0 BLEU 以上，建立了 28.4 BLEU 的新最先进水平。该模型配置列于表 3 底部行。训练在 8 块 P100 GPU 上耗时 3.5 天。即使是我们的 base 模型也以任何竞争模型一小部分的训练成本超越了所有以前发布的模型和集成模型。

在 WMT 2014 英译法翻译任务中，我们的 big 模型达到了 41.0 BLEU 分数，超越了所有以前发布的单模型，训练成本不到此前最先进模型的 1/4。用于英译法的 Transformer (big) 模型使用了 dropout 率 $P_{\text{drop}} = 0.1$ ，而非 0.3。

对于 base 模型，我们使用了通过对最后 5 个检查点（每 10 分钟写入一次）取平均获得的单个模型。对于 big 模型，我们平均了最后 20 个检查点。我们使用了集束大小为 4、长度惩罚 $\alpha = 0.6$ 的集束搜索 [6]。这些超参数是在开发集上实验后选择的。我们在推理时将最大输出长度设置为输入长度 + 50，但在可能时提前终止 [6]。

表 2 总结了我们的结果，并将我们的翻译质量和训练成本与文献中的其他模型架构进行了比较。我们通过将训练时间、使用的 GPU 数量以及每个 GPU 的持续单精度浮点容量估计值相乘来估计训练模型使用的浮点操作数 ²。

> ² 我们使用的 TFLOPS 值为：K80: 2.8, K40: 3.7, M40: 6.0, P100: 9.5。

### 6.2 模型变体

为了评估 Transformer 不同组件的重要性，我们以不同方式改变 base 模型，测量开发集 newstest2013 上英译德性能的变化。我们使用了前一节中描述的集束搜索，但没有使用检查点平均。我们在表 3 中展示了这些结果。

**表 3：Transformer 架构的变体。未列出的值与 base 模型相同。所有指标均在英译德翻译开发集 newstest2013 上。列出的困惑度是基于 byte-pair 编码的每 word-piece 困惑度，不应与每词困惑度进行比较。**

| | $N$ | $d_{\text{model}}$ | $d_{\text{ff}}$ | $h$ | $d_k$ | $d_v$ | $P_{\text{drop}}$ | $\epsilon_{\text{ls}}$ | 训练步数 | PPL (dev) | BLEU (dev) | 参数量 $\times 10^6$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| base | 6 | 512 | 2048 | 8 | 64 | 64 | 0.1 | 0.1 | 100K | 4.92 | 25.8 | 65 |
| (A) | | | | 1 | 512 | 512 | | | | 5.29 | 24.9 | |
| | | | | 4 | 128 | 128 | | | | 5.00 | 25.5 | |
| | | | | 16 | 32 | 32 | | | | 4.91 | 25.8 | |
| | | | | 32 | 16 | 16 | | | | 5.01 | 25.4 | |
| (B) | | | | | 16 | | | | | 5.16 | 25.1 | 58 |
| | | | | | 32 | | | | | 5.01 | 25.4 | 60 |
| (C) | 2 | | | | | | | | | 6.11 | 23.7 | 36 |
| | 4 | | | | | | | | | 5.19 | 25.3 | 50 |
| | 8 | | | | | | | | | 4.88 | 25.5 | 80 |
| | | 256 | | | 32 | 32 | | | | 5.75 | 24.5 | 28 |
| | | 1024 | | | 128 | 128 | | | | 4.66 | 26.0 | 168 |
| | | | 1024 | | | | | | | 5.12 | 25.4 | 53 |
| | | | 4096 | | | | | | | 4.75 | 26.2 | 90 |
| (D) | | | | | | | 0.0 | | | 5.77 | 24.6 | |
| | | | | | | | 0.2 | | | 4.95 | 25.5 | |
| | | | | | | | | 0.0 | | 4.67 | 25.3 | |
| | | | | | | | | 0.2 | | 5.47 | 25.7 | |
| (E) | | | | | | | | | | 4.92 | 25.7 | |
| | 使用位置嵌入代替正弦波 | | | | | | | | | 4.92 | 25.7 | |
| big | 6 | 1024 | 4096 | 16 | | | 0.3 | | 300K | 4.33 | 26.4 | 213 |

在表 3 行 (A) 中，我们改变了注意力头的数量以及注意力键和值的维度，保持计算量不变，如第 3.2.2 节所述。单头注意力比最佳设置差 0.9 BLEU，但头数过多时质量也会下降。

在表 3 行 (B) 中，我们观察到减小注意力键大小 $d_k$ 会损害模型质量。这表明确定兼容性并不容易，并且比点积更复杂的兼容性函数可能是有益的。我们进一步在行 (C) 和 (D) 中观察到，正如预期的那样，更大的模型更好，并且 dropout 对避免过拟合非常有帮助。在行 (E) 中，我们将正弦位置编码替换为学习到的位置嵌入 [15]，并观察到与 base 模型几乎相同的结果。

### 6.3 英语成分句法分析

**表 4：Transformer 很好地泛化到英语成分句法分析（结果在 WSJ 第 23 节上）。**

| 解析器 | 训练方式 | WSJ 23 F1 |
| :--- | :--- | :--- |
| Vinyals & Kaiser et al. (2014) [33] | 仅 WSJ, 判别式 | 88.3 |
| Petrov et al. (2006) [34] | 仅 WSJ, 判别式 | 90.4 |
| Zhu et al. (2013) [35] | 仅 WSJ, 判别式 | 90.4 |
| Dyer et al. (2016) [36] | 仅 WSJ, 判别式 | 91.7 |
| Transformer (4 layers) | 仅 WSJ, 判别式 | 91.3 |
| Zhu et al. (2013) [35] | 半监督 | 91.3 |
| Huang & Harper (2009) [37] | 半监督 | 91.3 |
| McClosky et al. (2006) [38] | 半监督 | 92.1 |
| Vinyals & Kaiser et al. (2014) [33] | 半监督 | 92.1 |
| Transformer (4 layers) | 半监督 | 92.7 |
| Luong et al. (2015) [39] | 多任务 | 93.0 |
| Dyer et al. (2016) [36] | 生成式 | 93.3 |

为了评估 Transformer 是否能泛化到其他任务，我们在英语成分句法分析上进行了实验。该任务具有特定的挑战：输出受到强结构约束，并且比输入长得多。此外，RNN 序列到序列模型在小型数据场景下未能达到最先进结果 [33]。

我们在华尔街日报（WSJ）部分的 Penn Treebank [40]（约 40K 训练句子）上训练了一个 4 层 Transformer， $d_{\text{model}} = 1024$ 。我们还在半监督设置下进行了训练，使用了来自约 1700 万个句子的更大规模的高置信度和 BerkleyParser 语料库 [33]。对于仅 WSJ 设置，我们使用了 16K token 的词汇表；对于半监督设置，使用了 32K token 的词汇表。

我们只进行了少量实验来选择第 22 节开发集上的 dropout（包括注意力和残差，见第 5.4 节）、学习率和集束大小，所有其他参数与英译德 base 翻译模型保持一致。在推理时，我们将最大输出长度增加到输入长度 + 300。对于仅 WSJ 和半监督设置，我们都使用了集束大小 21 和 $\alpha = 0.3$ 。

我们在表 4 中的结果显示，尽管缺乏针对特定任务的调优，我们的模型表现得出奇地好，除了循环神经网络语法 [36] 之外，优于所有以前报告的模型。

与 RNN 序列到序列模型 [33] 相比，Transformer 即使在仅使用 40K 句子的 WSJ 训练集上训练时也优于 BerkeleyParser [34]。

## 7 结论

在这项工作中，我们提出了 Transformer，这是第一个完全基于注意力的序列转导模型，用多头自注意力取代了编码器-解码器架构中最常用的循环层。

对于翻译任务，Transformer 的训练速度显著快于基于循环或卷积层的架构。在 WMT 2014 英译德和 WMT 2014 英译法翻译任务上，我们都达到了新的最先进水平。在前一项任务中，我们的最佳模型甚至超过了所有以前报告的集成模型。

我们对基于注意力的模型的未来感到兴奋，并计划将它们应用于其他任务。我们计划将 Transformer 扩展到涉及文本之外输入和输出模态的问题，并研究局部的、受限的注意力机制以有效处理大型输入和输出，如图像、音频和视频。使生成更少顺序性是我们的另一个研究目标。

用于训练和评估我们模型的代码可在 https://github.com/tensorflow/tensor2tensor 获取。

## 致谢

我们感谢 Nal Kalchbrenner 和 Stephan Gouws 富有成果的评论、更正和启发。

## 参考文献

[1] Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. *Neural computation*, 9(8):1735–1780, 1997.

[2] Junyoung Chung, Çaglar Gülçehre, Kyunghyun Cho, and Yoshua Bengio. Empirical evaluation of gated recurrent neural networks on sequence modeling. *CoRR*, abs/1412.3555, 2014.

[3] Ilya Sutskever, Oriol Vinyals, and Quoc VV Le. Sequence to sequence learning with neural networks. In *Advances in Neural Information Processing Systems*, pages 3104–3112, 2014.

[4] Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural machine translation by jointly learning to align and translate. *CoRR*, abs/1409.0473, 2014.

[5] Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. Learning phrase representations using rnn encoder-decoder for statistical machine translation. *CoRR*, abs/1406.1078, 2014.

[6] Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Wolfgang Macherey, Maxim Krikun, Yuan Cao, Qin Gao, Klaus Macherey, et al. Google's neural machine translation system: Bridging the gap between human and machine translation. *arXiv preprint arXiv:1609.08144*, 2016.

[7] Minh-Thang Luong, Hieu Pham, and Christopher D Manning. Effective approaches to attention-based neural machine translation. *arXiv preprint arXiv:1508.04025*, 2015.

[8] Rafal Jozefowicz, Oriol Vinyals, Mike Schuster, Noam Shazeer, and Yonghui Wu. Exploring the limits of language modeling. *arXiv preprint arXiv:1602.02410*, 2016.

[9] Oleksii Kuchaiev and Boris Ginsburg. Factorization tricks for LSTM networks. *arXiv preprint arXiv:1703.10722*, 2017.

[10] Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. *arXiv preprint arXiv:1701.06538*, 2017.

[11] Yoon Kim, Carl Denton, Luong Hoang, and Alexander M. Rush. Structured attention networks. In *International Conference on Learning Representations*, 2017.

[12] Ankur Parikh, Oscar Täckström, Dipanjan Das, and Jakob Uszkoreit. A decomposable attention model. In *Empirical Methods in Natural Language Processing*, 2016.

[13] Łukasz Kaiser and Samy Bengio. Can active memory replace attention? In *Advances in Neural Information Processing Systems, (NIPS)*, 2016.

[14] Nal Kalchbrenner, Lasse Espeholt, Karen Simonyan, Aaron van den Oord, Alex Graves, and Koray Kavukcuoglu. Neural machine translation in linear time. *arXiv preprint arXiv:1610.10099v2*, 2017.

[15] Jonas Gehring, Michael Auli, David Grangier, Denis Yarats, and Yann N. Dauphin. Convolutional sequence to sequence learning. *arXiv preprint arXiv:1705.03122v2*, 2017.

[16] Sepp Hochreiter, Yoshua Bengio, Paolo Frasconi, and Jürgen Schmidhuber. Gradient flow in recurrent nets: the difficulty of learning long-term dependencies, 2001.

[17] Jianpeng Cheng, Li Dong, and Mirella Lapata. Long short-term memory-networks for machine reading. *arXiv preprint arXiv:1601.06733*, 2016.

[18] Romain Paulus, Caiming Xiong, and Richard Socher. A deep reinforced model for abstractive summarization. *arXiv preprint arXiv:1705.04304*, 2017.

[19] Zhouhan Lin, Minwei Feng, Cicero Nogueira dos Santos, Mo Yu, Bing Xiang, Bowen Zhou, and Yoshua Bengio. A structured self-attentive sentence embedding. *arXiv preprint arXiv:1703.03130*, 2017.

[20] Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston, and Rob Fergus. End-to-end memory networks. In C. Cortes, N. D. Lawrence, D. D. Lee, M. Sugiyama, and R. Garnett, editors, *Advances in Neural Information Processing Systems 28*, pages 2440–2448. Curran Associates, Inc., 2015.

[21] Łukasz Kaiser and Ilya Sutskever. Neural GPUs learn algorithms. In *International Conference on Learning Representations (ICLR)*, 2016.

[22] Alex Graves. Generating sequences with recurrent neural networks. *arXiv preprint arXiv:1308.0850*, 2013.

[23] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, pages 770–778, 2016.

[24] Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. *arXiv preprint arXiv:1607.06450*, 2016.

[25] Denny Britz, Anna Goldie, Minh-Thang Luong, and Quoc V. Le. Massive exploration of neural machine translation architectures. *CoRR*, abs/1703.03906, 2017.

[26] Ofir Press and Lior Wolf. Using the output embedding to improve language models. *arXiv preprint arXiv:1608.05859*, 2016.

[27] Rico Sennrich, Barry Haddow, and Alexandra Birch. Neural machine translation of rare words with subword units. *arXiv preprint arXiv:1508.07909*, 2015.

[28] Francois Chollet. Xception: Deep learning with depthwise separable convolutions. *arXiv preprint arXiv:1610.02357*, 2016.

[29] Diederik Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In *ICLR*, 2015.

[30] Nitish Srivastava, Geoffrey E Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov. Dropout: a simple way to prevent neural networks from overfitting. *Journal of Machine Learning Research*, 15(1):1929–1958, 2014.

[31] Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, and Zbigniew Wojna. Rethinking the inception architecture for computer vision. *CoRR*, abs/1512.00567, 2015.

[32] Jie Zhou, Ying Cao, Xuguang Wang, Peng Li, and Wei Xu. Deep recurrent models with fast-forward connections for neural machine translation. *CoRR*, abs/1606.04199, 2016.

[33] Vinyals & Kaiser, Koo, Petrov, Sutskever, and Hinton. Grammar as a foreign language. In *Advances in Neural Information Processing Systems*, 2015.

[34] Slav Petrov, Leon Barrett, Romain Thibaux, and Dan Klein. Learning accurate, compact, and interpretable tree annotation. In *Proceedings of the 21st International Conference on Computational Linguistics and 44th Annual Meeting of the ACL*, pages 433–440. ACL, July 2006.

[35] Muhua Zhu, Yue Zhang, Wenliang Chen, Min Zhang, and Jingbo Zhu. Fast and accurate shift-reduce constituent parsing. In *Proceedings of the 51st Annual Meeting of the ACL (Volume 1: Long Papers)*, pages 434–443. ACL, August 2013.

[36] Chris Dyer, Adhiguna Kuncoro, Miguel Ballesteros, and Noah A. Smith. Recurrent neural network grammars. In *Proc. of NAACL*, 2016.

[37] Zhongqiang Huang and Mary Harper. Self-training PCFG grammars with latent annotations across languages. In *Proceedings of the 2009 Conference on Empirical Methods in Natural Language Processing*, pages 832–841. ACL, August 2009.

[38] David McClosky, Eugene Charniak, and Mark Johnson. Effective self-training for parsing. In *Proceedings of the Human Language Technology Conference of the NAACL, Main Conference*, pages 152–159. ACL, June 2006.

[39] Minh-Thang Luong, Quoc V. Le, Ilya Sutskever, Oriol Vinyals, and Lukasz Kaiser. Multi-task sequence to sequence learning. *arXiv preprint arXiv:1511.06114*, 2015.

[40] Mitchell P Marcus, Mary Ann Marcinkiewicz, and Beatrice Santorini. Building a large annotated corpus of english: The penn treebank. *Computational linguistics*, 19(2):313–330, 1993.

## 附录：注意力可视化

![Figure 3: An example of the attention mechanism following long-distance dependencies in the encoder self-attention in layer 5 of 6. Many of the attention heads attend to a distant dependency of the verb 'making', completing the phrase 'making...more difficult'. Attentions here shown only for the word 'making'. Different colors represent different heads. Best viewed in color.](figure3.png)

**图 3：编码器第 5 层（共 6 层）自注意力中跟踪长距离依赖的注意力示例。许多注意力头关注动词"making"的远程依赖，完成"making...more difficult"短语。此处仅显示单词"making"的注意力。不同颜色代表不同的注意力头。请彩色查看。**

![Figure 4: Two attention heads, also in layer 5 of 6, apparently involved in anaphora resolution. Top: Full attentions for head 5. Bottom: Isolated attentions from just the word 'its' for attention heads 5 and 6. Note that the attentions are very sharp for this word.](figure4.png)

**图 4：同样在第 5 层（共 6 层）的两个注意力头，明显参与了指代消解。上：头 5 的全部注意力。下：仅针对单词"its"的注意力头 5 和 6 的孤立注意力。注意该词的注意力非常尖锐。**

![Figure 5: Many of the attention heads exhibit behaviour that seems related to the structure of the sentence. We give two such examples above, from two different heads from the encoder self-attention at layer 5 of 6. The heads clearly learned to perform different tasks.](figure5.png)

**图 5：许多注意力头表现出似乎与句子结构相关的行为。我们在上面给出了两个示例，来自第 5 层（共 6 层）编码器自注意力的两个不同头。这些头清楚地学会了执行不同的任务。**
