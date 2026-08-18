# BERT4Rec: 基于 Transformer 的双向编码器表示的序列推荐

> Fei Sun, Jun Liu, Jian Wu, Changhua Pei, Xiao Lin, Wenwu Ou, and Peng Jiang | 阿里巴巴集团（Alibaba Group），北京，中国



本文提出 BERT4Rec，**把 NLP 领域大杀四方的 BERT 深度双向自注意力架构搬进序列推荐，用 Cloze 任务破解双向建模的信息泄漏难题**。核心发现是——**相比最强基线，BERT4Rec 在四个数据集上平均 HR@10 提升 7.24%、NDCG@10 提升 11.03%、MRR 提升 11.46%**。

核心内容：

- 传统序列推荐用从左到右的单向模型编码用户行为，限制隐藏表示表达力，且假设序列严格有序，现实中并不总是成立
- 借鉴 BERT 的成功，用 L 层双向 Transformer 层堆叠建模用户行为序列，让每个 item 同时融合左右两侧上下文，直接捕获任意距离依赖
- 为规避双向训练的信息泄漏，引入 Cloze 任务（掩码语言模型）：随机掩码部分 item 并预测其 ID，测试时在序列末尾追加 [mask] 预测下一个 item
- 在四个真实数据集（Beauty、Steam、ML-1m、ML-20m）上与 8 种基线全面对比，并做注意力可视化、超参数分析与消融研究

关键发现：

- **BERT4Rec 全面碾压所有基线：平均 HR@10 提升 7.24%、NDCG@10 提升 11.03%、MRR 提升 11.46%**，其中 ML-20m 上 HR@1 相对 SASRec 提升高达 35.22%
- 隔离实验证实双向表示本身就有用：仅用 1 mask 的 BERT4Rec 在 Beauty 上 HR@10 达 0.2940，显著超过 SASRec 的 0.2653
- 最优掩码比例依赖序列长度：短序列数据集取大 ρ（Beauty 0.6、Steam 0.4），长序列数据集取小 ρ（ML-1m/ML-20m 取 0.2）
- 消融显示位置嵌入对长序列数据集至关重要（ML-1m NDCG@10 从 0.4759 暴跌至 0.2155），堆叠更深层与更多头在长序列上更受益

---



## 摘要

从用户的历史行为中建模其动态偏好，对推荐系统而言既充满挑战又至关重要。以往的方法采用序列神经网络，将用户的历史交互从左到右编码为隐藏表示，用于做出推荐。尽管这些方法行之有效，但我们认为这种从左到右的单向模型并非最优，其原因包括：a) 单向架构限制了用户行为序列中隐藏表示的表达能力；b) 它们通常假设序列具有严格顺序，而这在现实中并不总是成立。为了解决上述局限，我们提出了一种名为 BERT4Rec 的序列推荐模型，该模型采用深度双向自注意力对用户行为序列进行建模。为避免信息泄漏并高效地训练双向模型，我们将 Cloze 目标引入序列推荐，通过联合利用掩码 item 的左右上下文来预测序列中被随机掩码的 item。通过这种方式，我们学习得到一个双向表示模型来进行推荐，它允许用户历史行为中的每个 item 同时融合来自左右两侧的信息。在四个基准数据集上的大量实验表明，我们的模型一致地优于各种最先进的序列模型。



## CCS 概念

- 信息系统 → 推荐系统。



## 关键词

Sequential Recommendation; Bidirectional Sequential Model; Cloze

---



## 1 引言

准确刻画用户兴趣是推荐系统有效运作的核心。在许多真实世界的应用中，用户的当前兴趣本质上是动态且不断演化的，并受其历史行为的影响。例如，一个人在购买了任天堂 Switch 之后不久可能会购买配件（如 Joy-Con 手柄），尽管在一般情况下她/他并不会购买主机配件。

为了对用户行为中的这种序列动态进行建模，研究者们提出了多种基于用户历史交互进行序列推荐的方法 [15, 22, 40]。它们旨在根据用户过去的交互来预测用户接下来可能与之交互的 item。近年来，大量工作采用序列神经网络（例如循环神经网络（RNN，Recurrent Neural Network））进行序列推荐，并取得了令人瞩目的成果 [7, 14, 15, 56, 58]。以往工作的基本范式是：使用从左到右的序列模型将用户的历史交互编码为一个向量（即用户偏好的表示），并基于该隐藏表示做出推荐。

尽管这类从左到右的单向模型普遍且有效，但我们认为它们不足以学习用户行为序列的最优表示。其主要局限，如图 1c 和 1d 所示，在于这类单向模型限制了历史序列中 item 的隐藏表示能力——每个 item 只能编码来自其之前 item 的信息。另一个局限是，以往的单向模型最初是为具有自然顺序的序列数据（例如文本和时间序列数据）而引入的。它们通常假设数据具有严格有序的序列，而这对真实世界应用中的用户行为而言并不总是成立。事实上，由于各种不可观测的外部因素 [5]，用户历史交互中 item 的选择可能并不遵循严格顺序假设 [18, 54]。在这种情况下，在用户行为序列建模中融合来自两个方向的上下文就显得至关重要。

为了解决上述局限，我们寻求使用双向模型来学习用户历史行为序列的表示。具体来说，受 BERT [6] 在文本理解领域成功的启发，我们提出将深度双向自注意力模型应用于序列推荐，如图 1b 所示。就表示能力而言，深度双向模型在文本序列建模任务上的优异结果表明，在序列表示学习时融合来自两侧的上下文是有益的 [6]。就严格顺序假设而言，我们的模型在建模用户行为序列时比单向模型更合适，因为双向模型中的所有 item 都可以利用来自左侧和右侧的上下文。

然而，为序列推荐训练双向模型并非简单直观。传统的序列推荐模型通常采用从左到右的方式训练，即为输入序列中的每个位置预测下一个 item。如图 1 所示，在深度双向模型中联合利用左右上下文会导致信息泄漏，即允许每个 item 间接"看到目标 item"。这会使预测未来变得微不足道，网络也将无法学到任何有用的东西。

为解决这一问题，我们引入 Cloze 任务 [6, 50] 来替代单向模型中的目标（即顺序预测下一个 item）。具体来说，我们在输入序列中随机掩码一些 item（即用特殊 token [mask] 替换它们），然后基于其周围上下文预测这些被掩码 item 的 id。通过这种方式，我们避免了信息泄漏，并通过让输入序列中每个 item 的表示融合左右两侧上下文，学习得到一个双向表示模型。除了训练双向模型之外，Cloze 目标的另一个优势是它能够在多个 epoch 中产生更多样本来训练更强大的模型。然而，Cloze 任务的一个缺点是它与最终任务（即序列推荐）不一致。为解决此问题，在测试时，我们在输入序列末尾追加特殊 token "[mask]" 来表示需要预测的 item，然后基于其最终隐藏向量做出推荐。在四个数据集上的大量实验表明，我们的模型一致地优于各种最先进的基线方法。

本文的贡献如下：

- 我们提出通过 Cloze 任务，利用双向自注意力网络对用户行为序列进行建模。据我们所知，这是首项将深度双向序列模型和 Cloze 目标引入推荐系统领域的研究。
- 我们将我们的模型与最先进的方法进行比较，并通过在四个基准数据集上的定量分析，证明了双向架构和 Cloze 目标的有效性。
- 我们进行了全面的消融研究，以分析所提出模型中关键组件的贡献。



## 2 相关工作

在本节中，我们将简要回顾与我们的工作密切相关的几类研究，包括通用推荐、序列推荐和注意力机制。

### 2.1 通用推荐

早期推荐系统的工作通常使用协同过滤（CF，Collaborative Filtering）基于用户的交互历史来建模其偏好 [26, 43]。在各种 CF 方法中，矩阵分解（MF，Matrix Factorization）是最流行的一种，它将用户和 item 投影到共享的向量空间中，并通过它们向量之间的内积来估计用户对某个 item 的偏好 [26, 27, 41]。另一类工作是基于 item 的邻域方法 [20, 25, 31, 43]。它们利用预先计算的 item 到 item 相似度矩阵，通过度量目标 item 与用户交互历史中 item 的相似度来估计用户对某个 item 的偏好。近年来，深度学习正在极大地革新推荐系统。早期的开创性工作是 Salakhutdinov 等人 [42] 在 Netflix Prize 1 中提出的用于协同过滤的两层受限玻尔兹曼机（RBM，Restricted Boltzmann Machines）1 。一类基于深度学习的方法致力于将通过辅助信息（如文本 [23, 53]、图像 [21, 55] 和声学特征 [51]）学习得到的分布式 item 表示整合到 CF 模型中，以提高推荐性能。另一类工作则致力于取代传统的矩阵分解。例如，神经协同过滤（NCF，Neural Collaborative Filtering）[12] 通过多层感知机（MLP，Multi-Layer Perceptron）而非内积来估计用户偏好，而 AutoRec [44] 和 CDAE [57] 则使用自编码器框架预测用户的评分。

> 1 https://www.netflixprize.com

### 2.2 序列推荐

遗憾的是，上述方法都不是为序列推荐而设计的，因为它们都忽略了用户行为中的顺序。早期的序列推荐工作通常使用马尔可夫链（MC，Markov Chain）从用户的历史交互中捕获序列模式。例如，Shani 等人 [45] 将推荐生成形式化为一个序列优化问题，并采用马尔可夫决策过程（MDP，Markov Decision Process）来解决。随后，Rendle 等人 [40] 通过分解个性化马尔可夫链（FPMC，Factorizing Personalized Markov Chains）将 MC 与 MF 相结合，同时建模序列行为和一般兴趣。除了一阶 MC，高阶 MC 也被用来考虑更多之前的 item [10, 11]。近年来，RNN 及其变体——门控循环单元（GRU，Gated Recurrent Unit）[4] 和长短期记忆网络（LSTM，Long Short-Term Memory）[17]——在建模用户行为序列方面变得越来越流行 [7, 14, 15, 28, 37, 56, 58]。这些方法的基本思想是使用各种循环架构和损失函数，将用户之前的记录编码为一个向量（即用于预测的用户偏好表示），包括带排序损失的基于会话的 GRU（GRU4Rec）[15]、动态循环购物篮模型（DREAM，Dynamic REcurrent bAsket Model）[58]、基于用户的 GRU [7]、基于注意力的 GRU（NARM）[28]，以及采用新损失函数（即 BPR-max 和 TOP1-max）和改进采样策略的 GRU4Rec 改进版 [14]。除了循环神经网络，各种深度学习模型也被引入序列推荐 [3, 22, 33, 49]。例如，Tang 和 Wang [49] 提出了卷积序列模型（Caser，Convolutional Sequence Model），使用水平和垂直卷积滤波器学习序列模式。Chen 等人 [3] 和 Huang 等人 [19] 采用记忆网络（Memory Network）改进序列推荐。STAMP 使用带注意力的 MLP 网络同时捕获用户的整体兴趣和当前兴趣 [33]。

### 2.3 注意力机制

注意力机制在建模序列数据方面展现出了巨大潜力，例如机器翻译 [2, 52] 和文本分类 [?]。近来，一些工作尝试利用注意力机制来提高推荐的性能和可解释性 [28, 33]。例如，Li 等人 [28] 将注意力机制引入 GRU，在基于会话的推荐中同时捕获用户的序列行为与主要意图。上述工作基本上将注意力机制视为原始模型的附加组件。相比之下，Transformer [52] 和 BERT [6] 完全基于多头自注意力构建，并在文本序列建模上取得了最先进的结果。近年来，由于纯注意力神经网络在建模序列数据上的有效性和高效性，将其应用于序列数据建模的热情日益高涨 [30, 32, 38, 46?]。对于序列推荐，Kang 和 McAuley [22] 引入了名为 SASRec 的两层 Transformer 解码器（即 Transformer 语言模型）来捕获用户的序列行为，并在多个公开数据集上取得了最先进的结果。SASRec 与我们的工作密切相关。然而，它仍然是一个使用因果注意力掩码（causal attention mask）的单向模型，而我们则借助 Cloze 任务使用双向模型来编码用户的行为序列。



## 3 BERT4Rec

在深入细节之前，我们首先介绍本文研究的问题、基本概念和符号表示。

### 3.1 问题描述

在序列推荐中，设 $U=\{u_1, u_2, \ldots, u_{|U|}\}$ 表示一组用户， $V=\{v_1, v_2, \ldots, v_{|V|}\}$ 表示一组 item，列表 $S_u=[v^{(u)}_1, \ldots, v^{(u)}_t, \ldots, v^{(u)}_{n_u}]$ 表示用户 $u \in U$ 按时间顺序排列的交互序列，其中 $v^{(u)}_t \in V$ 是用户 $u$ 在时间步 t 交互的 item 2 ， $n_u$ 是用户 $u$ 的交互序列长度。给定交互历史 $S_u$ ，序列推荐旨在预测用户在时间步 $n_u+1$ 将要交互的 item。它可以形式化为对用户在时间步 $n_u+1$ 在所有可能 item 上的概率建模：

$$
p\left(v^{(u)}_{n_u+1} = v \mid S_u\right)
$$

> 2 这里，我们遵循 [22, 40]，使用相对时间索引而非绝对时间索引对交互记录进行编号。

### 3.2 模型架构

在这里，我们介绍一种名为 BERT4Rec 的新序列推荐模型，它将 Transformer 的双向编码器表示（Bidirectional Encoder Representations from Transformers）应用于一个新任务——序列推荐。它建立在流行的自注意力层——"Transformer 层"之上。如图 1b 所示，BERT4Rec 由 L 个双向 Transformer 层堆叠而成。在每一层，它通过 Transformer 层并行地在上一层所有位置之间交换信息，从而迭代地修正每个位置的表示。与图 1d 中基于 RNN 的方法那样逐步学习向前传递相关信息不同，自注意力机制赋予 BERT4Rec 直接捕获任意距离依赖的能力。这一机制带来了全局感受野，而像 Caser 这样基于 CNN 的方法通常只有有限的感受野。此外，与基于 RNN 的方法相比，自注意力易于并行化。对比图 1b、1c 和 1d，最显著的区别是 SASRec 和基于 RNN 的方法都是从左到右的单向架构，而我们的 BERT4Rec 使用双向自注意力来建模用户的行为序列。通过这种方式，我们提出的模型可以获得更强大的用户行为序列表示，从而提高推荐性能。

<!-- FIGURE1 -->
**图 1：** 序列推荐模型架构的差异。BERT4Rec 通过 Cloze 任务学习双向模型，而 SASRec 和基于 RNN 的方法都是从左到右的单向模型，顺序地预测下一个 item。

### 3.3 Transformer 层

如图 1b 所示，给定长度为 t 的输入序列，我们通过应用来自 [52] 的 Transformer 层，在每一层 l 同时迭代计算每个位置 i 的隐藏表示 $h^l_i$ 。由于在实践中我们对所有位置同时计算注意力函数，因此我们将 $h^l_i \in \mathbb{R}^d$ 堆叠为矩阵 $H^l \in \mathbb{R}^{t \times d}$ 。如图 1a 所示，Transformer 层 Trm 包含两个子层：多头自注意力子层和逐位置前馈网络。

**多头自注意力。** 注意力机制已成为多种任务中序列建模不可或缺的组成部分，它能够在忽略表示对在序列中距离的情况下捕获它们之间的依赖关系。以往的研究表明，联合关注不同位置、来自不同表示子空间的信息是有益的 [6, 29, 52]。因此，我们在这里采用多头自注意力，而不是执行单一的注意力函数。具体来说，多头注意力首先使用不同的、可学习的线性投影将 $H^l$ 线性投影到 h 个子空间，然后并行应用 h 个注意力函数，产生输出表示，这些输出被拼接后再次投影：

$$
\text{MH}(H^l) = [\text{head}_1; \text{head}_2; \ldots; \text{head}_h]W^O \qquad (1)
$$

$$
\text{head}_i = \text{Attention}\left(H^l W^Q_i, H^l W^K_i, H^l W^V_i\right)
$$

其中每个头的投影矩阵 $W^Q_i \in \mathbb{R}^{d \times d/h}$ 、 $W^K_i \in \mathbb{R}^{d \times d/h}$ 、 $W^V_i \in \mathbb{R}^{d \times d/h}$ 和 $W^O_i \in \mathbb{R}^{d \times d}$ 均为可学习参数。为简洁起见，我们省略了层下标 l。事实上，这些投影参数在各层之间并不共享。这里的 Attention 函数是缩放点积注意力（Scaled Dot-Product Attention）：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d/h}}\right) V \qquad (2)
$$

其中查询 Q、键 K 和值 V 按照公式 (1) 的方式，由同一个矩阵 $H^l$ 通过不同的已学习投影矩阵投影得到。引入温度 $\sqrt{d/h}$ 是为了产生更柔和的注意力分布，以避免极小的梯度 [16, 52]。

**逐位置前馈网络。** 如上所述，自注意力子层主要基于线性投影。为了赋予模型非线性以及不同维度之间的交互能力，我们在每个位置对自注意力子层的输出分别且相同地应用逐位置前馈网络（PFFN，Position-wise Feed-Forward Network）。它由两个仿射变换组成，中间使用高斯误差线性单元（GELU，Gaussian Error Linear Unit）激活函数：

$$
\text{PFFN}(H^l) = \left[\text{FFN}(h^l_1)^\top; \ldots; \text{FFN}(h^l_t)^\top\right]^\top
$$

$$
\text{FFN}(x) = \text{GELU}\left(xW^{(1)} + b^{(1)}\right) W^{(2)} + b^{(2)}
$$

$$
\text{GELU}(x) = x\Phi(x) \qquad (3)
$$

其中 $\Phi(x)$ 是标准高斯分布的累积分布函数， $W^{(1)} \in \mathbb{R}^{d \times 4d}$ 、 $W^{(2)} \in \mathbb{R}^{4d \times d}$ 、 $b^{(1)} \in \mathbb{R}^{4d}$ 和 $b^{(2)} \in \mathbb{R}^d$ 是可学习参数，并且在所有位置共享。为方便起见，我们省略了层下标 l。事实上，这些参数逐层不同。在本工作中，遵循 OpenAI GPT [38] 和 BERT [6]，我们使用更平滑的 GELU [13] 激活函数，而非标准的 ReLu 激活函数。

**堆叠 Transformer 层。** 如上所述，我们可以利用自注意力机制轻松捕获整个用户行为序列中的 item 间交互。然而，通过堆叠自注意力层来学习更复杂的 item 转移模式通常是有益的。但网络越深，训练就越困难。因此，如图 1a 所示，我们在两个子层周围各使用一个残差连接（RC，Residual Connection）[9]，随后进行层归一化（LN，Layer Normalization）[1]。此外，我们还在每个子层的输出归一化之前对其应用 Dropout [47]。也就是说，每个子层的输出为 $\text{LN}(x + \text{Dropout}(\text{sublayer}(x)))$ ，其中 sublayer(·) 是子层自身实现的函数，LN 是 [1] 中定义的层归一化函数。我们使用 LN 对同一层的所有隐藏单元进行输入归一化，以稳定并加速网络训练。总之，BERT4Rec 按如下方式更新每一层的隐藏表示：

$$
H^l = \text{Trm}(H^{l-1}), \quad \forall l \in [1, \ldots, L] \qquad (4)
$$

$$
\text{Trm}(H^{l-1}) = \text{LN}\left(A^{l-1} + \text{Dropout}\left(\text{PFFN}(A^{l-1})\right)\right) \qquad (5)
$$

$$
A^{l-1} = \text{LN}\left(H^{l-1} + \text{Dropout}\left(\text{MH}(H^{l-1})\right)\right) \qquad (6)
$$

### 3.4 嵌入层

如上所述，由于没有任何循环或卷积模块，Transformer 层 Trm 无法感知输入序列的顺序。为了利用输入的序列信息，我们在 Transformer 层堆栈的底部将位置嵌入（PE，Positional Embedding）注入输入 item 嵌入中。对于给定的 item $v_i$ ，其输入表示 $h^0_i$ 通过将对应的 item 嵌入与位置嵌入相加构建：

$$
h^0_i = v_i + p_i
$$

其中 $v_i \in E$ 是 item $v_i$ 的 d 维嵌入， $p_i \in P$ 是位置索引 i 的 d 维位置嵌入。在本工作中，为了获得更好的性能，我们使用可学习的位置嵌入，而不是 [52] 中的固定正弦嵌入。位置嵌入矩阵 $P \in \mathbb{R}^{N \times d}$ 允许我们的模型识别正在处理的输入位置。然而，它也限制了模型能够处理的最大句子长度 N。因此，当 $t > N$ 时，我们需要将输入序列 $[v_1, \ldots, v_t]$ 截断为最后 N 个 item $[v^u_{t-N+1}, \ldots, v_t]$ 。

### 3.5 输出层

经过 L 层在上一层所有位置之间分层交换信息后，我们得到输入序列所有 item 的最终输出 $H^L$ 。假设我们在时间步 t 掩码了 item $v_t$ ，那么我们如图 1b 所示，基于 $h^L_t$ 预测被掩码的 item $v_t$ 。具体来说，我们应用一个中间带 GELU 激活的两层前馈网络，以产生目标 item 上的输出分布：

$$
P(v) = \text{softmax}\left(\text{GELU}(h^L_t W^P + b^P) E^\top + b^O\right) \qquad (7)
$$

其中 $W^P$ 是可学习的投影矩阵， $b^P$ 和 $b^O$ 是偏置项， $E \in \mathbb{R}^{|V| \times d}$ 是 item 集合 V 的嵌入矩阵。我们在输入层和输出层共享 item 嵌入矩阵，以缓解过拟合并减小模型规模。

### 3.6 模型学习

**训练。** 传统的单向序列推荐模型通常通过为输入序列中的每个位置预测下一个 item 来训练模型，如图 1c 和 1d 所示。具体来说，输入序列 $[v_1, \ldots, v_t]$ 的目标是其移位版本 $[v_2, \ldots, v_{t+1}]$ 。然而，如图 1b 所示，在双向模型中联合利用左右上下文会导致每个 item 的最终输出表示包含目标 item 的信息。这使预测未来变得微不足道，网络也将无法学到任何有用的东西。针对此问题的一个简单解决方案是：从长度为 t 的原始行为序列中创建 $t-1$ 个样本（带下一个 item 的子序列，如 $([v_1], v_2)$ 和 $([v_1, v_2], v_3)$ ），然后用双向模型分别编码每个历史子序列来预测目标 item。然而，这种方法非常耗费时间和资源，因为我们需要为序列中的每个位置创建一个新样本并分别预测。

为了高效地训练我们提出的模型，我们将一个新的目标——Cloze 任务 [50]（在 [6] 中也称为"掩码语言模型"）应用于序列推荐。它是一种测试，由删除了部分单词的一段语言组成，要求参与者填写缺失的单词。在我们的场景中，对于每个训练步骤，我们随机掩码输入序列中比例为 $\rho$ 的 item（即用特殊 token "[mask]" 替换），然后仅根据被掩码 item 的左右上下文预测其原始 id。例如：

输入： $[v_1, v_2, v_3, v_4, v_5]$ → 随机掩码 → $[v_1, [mask]_1, v_3, [mask]_2, v_5]$

标签： $[mask]_1 = v_2$ ， $[mask]_2 = v_4$

与传统的序列推荐一样，与 "[mask]" 对应的最终隐藏向量被送入 item 集合上的输出 softmax。最终，我们将每个掩码输入 $S'_u$ 的损失定义为掩码目标上的负对数似然：

$$
\mathcal{L} = \frac{1}{|S^m_u|} \sum_{v_m \in S^m_u} -\log P(v_m = v^*_m \mid S'_u) \qquad (8)
$$

其中 $S'_u$ 是用户行为历史 $S_u$ 的掩码版本， $S^m_u$ 是其随机掩码的 item， $v^*_m$ 是被掩码 item $v_m$ 的真实 item，概率 $P(\cdot)$ 在公式 (7) 中定义。

Cloze 任务的一个额外优势是它可以产生更多样本来训练模型。假设一个长度为 n 的序列，图 1c 和 1d 中的传统序列预测只产生 n 个唯一的训练样本，而 BERT4Rec 在多个 epoch 中可以获得 $\binom{n}{k}$ 个样本（如果随机掩码 k 个 item）。这使我们能够训练一个更强大的双向表示模型。

**测试。** 如上所述，我们在训练与最终的序列推荐任务之间造成了不匹配，因为 Cloze 目标旨在预测当前被掩码的 item，而序列推荐旨在预测未来。为解决这个问题，我们在用户行为序列末尾追加特殊 token "[mask]"，然后基于该 token 的最终隐藏表示预测下一个 item。为了更好地匹配序列推荐任务（即预测最后一个 item），在训练期间我们还产生只掩码输入序列中最后一个 item 的样本。这类似于对序列推荐的微调，可以进一步提高推荐性能。

### 3.7 讨论

在这里，我们讨论我们的模型与以往相关工作之间的关系。

**SASRec。** 显然，SASRec 是我们的 BERT4Rec 的一个从左到右的单向版本，它使用单头注意力和因果注意力掩码。不同的架构导致了不同的训练方法。SASRec 为序列中每个位置预测下一个 item，而 BERT4Rec 使用 Cloze 目标预测序列中被掩码的 item。

**CBOW 与 SG。** 另一个非常相似的工作是连续词袋（CBOW，Continuous Bag-of-Words）和跳字模型（SG，Skip-Gram）[35]。CBOW 使用其上下文（左右两侧）中所有词向量的平均值来预测目标词。如果我们让 BERT4Rec 使用一层自注意力，对 item 使用均匀注意力权重，不共享 item 嵌入，移除位置嵌入，并且只掩码中心 item，那么它可以被看作是 BERT4Rec 的一个简化情形。与 CBOW 类似，SG 也可以看作是 BERT4Rec 经过类似化简操作（掩码除一个之外的所有 item）后的简化情形。从这个角度看，Cloze 可以看作是 CBOW 和 SG 目标的通用形式。此外，CBOW 使用简单的聚合器来建模词序列，因为它的目标是学习好的词表示，而不是句子表示。相反，我们致力于学习一个强大的行为序列表示模型（本工作中的深度自注意力网络）来进行推荐。

**BERT。** 尽管我们的 BERT4Rec 受到 NLP 中 BERT 的启发，但它与 BERT 仍有几个不同之处：a) 最关键的区别在于 BERT4Rec 是一个用于序列推荐的端到端模型，而 BERT 是一个用于句子表示的预训练模型。BERT 利用大规模、与任务无关的语料库为各种文本序列任务预训练句子表示模型，因为这些任务共享关于语言的相同背景知识。然而，这一假设在推荐任务中并不成立。因此，我们对不同的序列推荐数据集端到端地训练 BERT4Rec。b) 与 BERT 不同，我们移除了下一句损失和分段嵌入（segment embeddings），因为在序列推荐任务中，BERT4Rec 将用户的历史行为建模为单个序列。

## 4 实验

### 4.1 数据集

我们在四个真实世界、具有代表性的数据集上评估所提出的模型，这些数据集在领域和稀疏度方面差异显著。

- **Amazon Beauty** 3 ：这是 McAuley 等人 [34] 从 Amazon.com 爬取的一系列产品评论数据集。他们根据 Amazon 的一级产品类别将数据划分为单独的数据集。本工作中，我们采用"Beauty"类别。
- **Steam** 4 ：这是 Kang 和 McAuley [22] 从 Steam（一个大型在线视频游戏分发平台）收集的数据集。
- **MovieLens** [8]：这是评估推荐算法的流行基准数据集。本工作中，我们采用两个成熟的版本：MovieLens 1m（ML-1m）5 和 MovieLens 20m（ML-20m）6 。

对于数据预处理，我们遵循 [22, 40, 49] 中的常见做法。对所有数据集，我们将所有数值评分或评论的存在性转换为值为 1 的隐式反馈（即用户与该 item 发生了交互）。之后，我们按用户对交互记录进行分组，并按照时间戳对这些交互记录排序，为每个用户构建交互序列。为确保数据集质量，遵循常见做法 [12, 22, 40, 49]，我们保留至少具有五次反馈的用户。处理后的数据集统计信息汇总于表 1。

> 3 http://jmcauley.ucsd.edu/data/amazon/
>
> 4 https://cseweb.ucsd.edu/~jmcauley/datasets.html#steam_data
>
> 5 https://grouplens.org/datasets/movielens/1m/
>
> 6 https://grouplens.org/datasets/movielens/20m/

**表 1：数据集统计信息。**

| 数据集 | 用户数 | item 数 | 行为数 | 平均长度 | 密度 |
|---|---|---|---|---|---|
| Beauty | 40,226 | 54,542 | 0.35m | 8.8 | 0.02% |
| Steam | 281,428 | 13,044 | 3.5m | 12.4 | 0.10% |
| ML-1m | 6040 | 3416 | 1.0m | 163.5 | 4.79% |
| ML-20m | 138,493 | 26,744 | 20m | 144.4 | 0.54% |

### 4.2 任务设置与评估指标

为了评估序列推荐模型，我们采用了留一法（leave-one-out）评估（即下一个 item 推荐）任务，该方法在 [12, 22, 49] 中被广泛使用。对于每个用户，我们留出行为序列的最后一个 item 作为测试数据，将最后一个 item 之前的那个 item 作为验证集，并使用其余 item 进行训练。为便于公平评估，我们遵循 [12, 22, 49] 中的常见策略，将测试集中的每个真实 item 与该用户未交互过的 100 个随机采样的负样本 item 配对。为了使采样可靠且具有代表性 [19]，这 100 个负样本 item 是根据其流行度进行采样的。因此，任务变为对每个用户将这 100 个负样本与真实 item 进行排序。

**评估指标。** 为了评估所有模型的排序列表，我们采用了多种评估指标，包括命中率（HR，Hit Ratio）、归一化折损累计增益（NDCG，Normalized Discounted Cumulative Gain）和平均倒数排名（MRR，Mean Reciprocal Rank）。考虑到每个用户只有一个真实 item，HR@k 等价于 Recall@k 并与 Precision@k 成正比；MRR 等价于平均精度均值（MAP，Mean Average Precision）。本工作中，我们报告 k = 1, 5, 10 时的 HR 和 NDCG。对于所有这些指标，数值越高，性能越好。

### 4.3 基线与实现细节

为了验证我们方法的有效性，我们将其与以下具有代表性的基线方法进行比较：

- **POP**：这是最简单的基线，根据 item 的流行度（以交互次数衡量）对 item 进行排序。
- **BPR-MF** [39]：它使用成对排序损失，在隐式反馈下优化矩阵分解。
- **NCF** [12]：它使用 MLP 建模用户–item 交互，而非矩阵分解中的内积（inner product）。
- **FPMC** [40]：它通过将 MF 与一阶马尔可夫链相结合，捕获用户的整体品味以及其序列行为。
- **GRU4Rec** [15]：它使用基于排序损失的 GRU 对用户序列进行建模，用于基于会话的推荐。
- **GRU4Rec+** [14]：它是 GRU4Rec 的改进版本，采用了一类新的损失函数和采样策略。
- **Caser** [49]：它以水平和垂直两种方式使用 CNN，为序列推荐建模高阶马尔可夫链。
- **SASRec** [22]：它使用从左到右的 Transformer 语言模型捕获用户的序列行为，并在序列推荐上取得了最先进的性能。

对于 NCF 7 、GRU4Rec 8 、GRU4Rec+ 8 、Caser 9 和 SASRec 10 ，我们使用相应作者提供的代码。对于 BPR-MF 和 FPMC，我们使用 TensorFlow 实现它们。对于所有模型的公共超参数，我们考虑隐藏维度 d ∈ {16, 32, 64, 128, 256}， $\ell_2$ 正则化器 ∈ {1, 0.1, 0.01, 0.001, 0.0001}，以及 Dropout 率 ∈ {0, 0.1, 0.2, $\cdots$, 0.9}。所有其他超参数（例如 Caser 中的马尔可夫阶数）和初始化策略要么遵循方法作者的建议，要么在验证集上进行调优。我们报告每个基线在其最优超参数设置下的结果。

> 7 https://github.com/hexiangnan/neural_collaborative_filtering
>
> 8 https://github.com/hidasib/GRU4Rec
>
> 9 https://github.com/graytowne/caser_pytorch
>
> 10 https://github.com/kang205/SASRec

我们使用 TensorFlow 实现 BERT4Rec 11 。所有参数使用范围为 [−0.02, 0.02] 的截断正态分布进行初始化。我们使用 Adam [24] 训练模型，学习率为 1e-4， $\beta_1 = 0.9$ ， $\beta_2 = 0.999$ ， $\ell_2$ 权重衰减为 0.01，并采用学习率线性衰减。当梯度的 $\ell_2$ 范数超过阈值 5 时进行梯度裁剪。为公平比较，我们设置层数 L = 2、头数 h = 2，并使用与 [22] 相同的最大序列长度：ML-1m 和 ML-20m 为 N = 200，Beauty 和 Steam 数据集为 N = 50。对于头的设置，我们根据经验将每个头的维度设为 32（若 d < 32 则使用单头）。我们使用验证集调优掩码比例 ρ，结果为 Beauty 取 ρ = 0.6，Steam 取 ρ = 0.4，ML-1m 和 ML-20m 取 ρ = 0.2。所有模型均从头训练，不使用任何预训练，在单块 NVIDIA GeForce GTX 1080 Ti GPU 上训练，批大小为 256。

> 11 https://github.com/FeiSun/BERT4Rec

### 4.4 整体性能对比

表 2 总结了所有模型在四个基准数据集上的最优结果。最后一列是 BERT4Rec 相对于最优基线的改进幅度。我们省略了 NDCG@1 的结果，因为它在我们的实验中与 HR@1 相等。可以观察到：

非个性化的 POP 方法在所有数据集上给出了最差的性能 12 ，因为它没有利用历史记录来建模用户的个性化偏好。在所有基线方法中，序列方法（如 FPMC 和 GRU4Rec+）在所有数据集上一致地优于非序列方法（如 BPR-MF 和 NCF）。与 BPR-MF 相比，FPMC 的主要改进在于它以序列方式建模用户的历史记录。这一观察验证了考虑序列信息有助于提升推荐系统的性能。

在序列推荐基线中，Caser 在所有数据集上都优于 FPMC，尤其是在稠密数据集 ML-1m 上，这表明高阶马尔可夫链对序列推荐是有益的。然而，高阶马尔可夫链通常使用非常小的阶数 L，因为它们不能很好地随阶数 L 扩展。这导致 Caser 的性能不如 GRU4Rec+ 和 SASRec，尤其是在稀疏数据集上。此外，SASRec 的性能明显优于 GRU4Rec 和 GRU4Rec+，这表明自注意力机制是进行序列推荐的更强大工具。

根据这些结果，显而易见的是 BERT4Rec 在四个数据集上就所有评估指标而言均表现最佳。它相对最强基线平均取得了 HR@10 提升 7.24%、NDCG@10 提升 11.03% 和 MRR 提升 11.46% 的改进。

**表 2：不同方法在下一个 item 预测上的性能对比。每行加粗分数为最优，带下划线分数为次优。相比基线的改进在 p < 0.01 水平下具有统计显著性。**

| 数据集 | 指标 | POP | BPR-MF | NCF | FPMC | GRU4Rec | GRU4Rec+ | Caser | SASRec | BERT4Rec | 改进 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Beauty | HR@1 | 0.0077 | 0.0415 | 0.0407 | 0.0435 | 0.0402 | 0.0551 | 0.0475 | 0.0906 | 0.0953 | 5.19% |
| Beauty | HR@5 | 0.0392 | 0.1209 | 0.1305 | 0.1387 | 0.1315 | 0.1781 | 0.1625 | 0.1934 | 0.2207 | 14.12% |
| Beauty | HR@10 | 0.0762 | 0.1992 | 0.2142 | 0.2401 | 0.2343 | 0.2654 | 0.2590 | 0.2653 | 0.3025 | 14.02% |
| Beauty | NDCG@5 | 0.0230 | 0.0814 | 0.0855 | 0.0902 | 0.0812 | 0.1172 | 0.1050 | 0.1436 | 0.1599 | 11.35% |
| Beauty | NDCG@10 | 0.0349 | 0.1064 | 0.1124 | 0.1211 | 0.1074 | 0.1453 | 0.1360 | 0.1633 | 0.1862 | 14.02% |
| Beauty | MRR | 0.0437 | 0.1006 | 0.1043 | 0.1056 | 0.1023 | 0.1299 | 0.1205 | 0.1536 | 0.1701 | 10.74% |
| Steam | HR@1 | 0.0159 | 0.0314 | 0.0246 | 0.0358 | 0.0574 | 0.0812 | 0.0495 | 0.0885 | 0.0957 | 8.14% |
| Steam | HR@5 | 0.0805 | 0.1177 | 0.1203 | 0.1517 | 0.2171 | 0.2391 | 0.1766 | 0.2559 | 0.2710 | 5.90% |
| Steam | HR@10 | 0.1389 | 0.1993 | 0.2169 | 0.2551 | 0.3313 | 0.3594 | 0.2870 | 0.3783 | 0.4013 | 6.08% |
| Steam | NDCG@5 | 0.0477 | 0.0744 | 0.0717 | 0.0945 | 0.1370 | 0.1613 | 0.1131 | 0.1727 | 0.1842 | 6.66% |
| Steam | NDCG@10 | 0.0665 | 0.1005 | 0.1026 | 0.1283 | 0.1802 | 0.2053 | 0.1484 | 0.2147 | 0.2261 | 5.31% |
| Steam | MRR | 0.0669 | 0.0942 | 0.0932 | 0.1139 | 0.1420 | 0.1757 | 0.1305 | 0.1874 | 0.1949 | 4.00% |
| ML-1m | HR@1 | 0.0141 | 0.0914 | 0.0397 | 0.1386 | 0.1583 | 0.2092 | 0.2194 | 0.2351 | 0.2863 | 21.78% |
| ML-1m | HR@5 | 0.0715 | 0.2866 | 0.1932 | 0.4297 | 0.4673 | 0.5103 | 0.5353 | 0.5434 | 0.5876 | 8.13% |
| ML-1m | HR@10 | 0.1358 | 0.4301 | 0.3477 | 0.5946 | 0.6207 | 0.6351 | 0.6692 | 0.6629 | 0.6970 | 4.15% |
| ML-1m | NDCG@5 | 0.0416 | 0.1903 | 0.1146 | 0.2885 | 0.3196 | 0.3705 | 0.3832 | 0.3980 | 0.4454 | 11.91% |
| ML-1m | NDCG@10 | 0.0621 | 0.2365 | 0.1640 | 0.3439 | 0.3627 | 0.4064 | 0.4268 | 0.4368 | 0.4818 | 10.32% |
| ML-1m | MRR | 0.0627 | 0.2009 | 0.1358 | 0.2891 | 0.3041 | 0.3462 | 0.3648 | 0.3790 | 0.4254 | 12.24% |
| ML-20m | HR@1 | 0.0221 | 0.0553 | 0.0231 | 0.1079 | 0.1459 | 0.2021 | 0.1232 | 0.2544 | 0.3440 | 35.22% |
| ML-20m | HR@5 | 0.0805 | 0.2128 | 0.1358 | 0.3601 | 0.4657 | 0.5118 | 0.3804 | 0.5727 | 0.6323 | 10.41% |
| ML-20m | HR@10 | 0.1378 | 0.3538 | 0.2922 | 0.5201 | 0.5844 | 0.6524 | 0.5427 | 0.7136 | 0.7473 | 4.72% |
| ML-20m | NDCG@5 | 0.0511 | 0.1332 | 0.0771 | 0.2239 | 0.3090 | 0.3630 | 0.2538 | 0.4208 | 0.4967 | 18.04% |
| ML-20m | NDCG@10 | 0.0695 | 0.1786 | 0.1271 | 0.2895 | 0.3637 | 0.4087 | 0.3062 | 0.4665 | 0.5340 | 14.47% |
| ML-20m | MRR | 0.0709 | 0.1503 | 0.1072 | 0.2273 | 0.2967 | 0.3476 | 0.2529 | 0.4026 | 0.4785 | 18.85% |

**表 3：d = 256 时对双向性与 Cloze 的分析。**

| 模型 | Beauty HR@10 | Beauty NDCG@10 | Beauty MRR | ML-1m HR@10 | ML-1m NDCG@10 | ML-1m MRR |
|---|---|---|---|---|---|---|
| SASRec | 0.2653 | 0.1633 | 0.1536 | 0.6629 | 0.4368 | 0.3790 |
| BERT4Rec (1 mask) | 0.2940 | 0.1769 | 0.1618 | 0.6869 | 0.4696 | 0.4127 |
| BERT4Rec | 0.3025 | 0.1862 | 0.1701 | 0.6970 | 0.4818 | 0.4254 |

**问题 1：性能提升究竟来自双向自注意力模型，还是来自 Cloze 目标？** 为了回答这个问题，我们尝试将这两个因素的影响分离开来：将 Cloze 任务约束为每次只掩码一个 item。这样，我们的 BERT4Rec（1 mask）与 SASRec 之间的主要区别在于，BERT4Rec 联合利用左右上下文来预测目标 item。由于篇幅所限，我们在表 3 中报告了 d = 256 时 Beauty 和 ML-1m 上的结果。结果表明，带 1 mask 的 BERT4Rec 在所有指标上都显著优于 SASRec。这证明了双向表示对序列推荐的重要性。此外，最后两行表明 Cloze 目标也能改善性能。关于 Cloze 任务中掩码比例 ρ 的详细分析见 § 4.6。

**问题 2：为什么以及如何使双向模型优于单向模型？** 为了回答这个问题，我们尝试通过在 Beauty 上测试期间可视化最后 10 个 item 的平均注意力权重（图 2）来揭示有意义的模式。由于篇幅所限，我们只报告不同层和头中四个具有代表性的注意力热力图。

<!-- FIGURE2 -->
**图 2：** Beauty 上平均注意力权重的热力图，最后一个位置 "9" 表示 "[mask]"（建议以彩色查看）。

我们从这些结果中做出几点观察。a) 不同头之间的注意力不同。例如，在第 1 层，头 1 倾向于关注左侧的 item，而头 2 则倾向于关注右侧的 item。b) 不同层之间的注意力也不同。显然，第 2 层的注意力倾向于聚焦于更近期的 item。这是因为第 2 层直接连接到输出层，而近期 item 在预测未来时扮演着更重要的角色。另一个有趣的模式是，图 2a 和 2b 中的头也倾向于关注 [mask] 13 。这可能是自注意力将序列级状态传播到 item 级的一种方式。c) 最后也是最重要的是，与单向模型只能关注左侧 item 不同，BERT4Rec 中的 item 倾向于同时关注两侧的 item。这表明双向性对于用户行为序列建模是必要且有帮助的。

> 13 这一现象在使用 BERT 的文本序列建模中也存在。

在接下来的研究中，我们考察超参数的影响，包括隐藏维度 d、掩码比例 ρ 和最大序列长度 N。我们通过将其余超参数固定在其最优设置，一次分析一个超参数。由于篇幅所限，后续实验我们只报告 NDCG@10 和 HR@10。

### 4.5 隐藏维度 d 的影响

我们现在研究隐藏维度 d 如何影响推荐性能。图 3 展示了在保持其他最优超参数不变、隐藏维度 d 从 16 变化到 256 时，各神经序列方法的 NDCG@10 和 HR@10。我们从图中做出一些观察。

这些子图中最明显的观察是，随着维度增加，每个模型的性能趋于收敛。更大的隐藏维度不一定带来更好的模型性能，尤其是在 Beauty 和 Steam 这样的稀疏数据集上。这可能是由过拟合导致的。具体而言，Caser 在四个数据集上表现不稳定，这可能限制其实用性。基于自注意力的方法（即 SASRec 和 BERT4Rec）在所有数据集上都取得了优异的性能。最后，即使在相对较小的隐藏维度下，我们的模型在所有数据集上也一致地优于所有其他基线。考虑到我们的模型在 d ≥ 64 时就已取得令人满意的性能，在后续分析中我们只报告 d = 64 时的结果。

<!-- FIGURE3 -->
**图 3：** 隐藏维度 d 对神经序列模型 HR@10 和 NDCG@10 的影响。

### 4.6 掩码比例 ρ 的影响

如 § 3.6 所述，掩码比例 ρ 是模型训练中的一个关键因素，它直接影响损失函数（公式 (8)）。显然，掩码比例 ρ 不能太小，否则不足以学习一个强模型；同时，它也不能太大，否则训练会变得困难，因为在这种情况下有太多 item 需要仅基于少量上下文进行猜测。为了考察这一点，我们研究掩码比例 ρ 如何影响不同数据集上的推荐性能。

图 4 展示了掩码比例 ρ 从 0.1 变化到 0.9 时的结果。综合所有数据集上 ρ > 0.6 的结果，出现了一个普遍的模式：随着 ρ 增大，性能下降。从前两列的结果可以很容易看出，在所有数据集上，ρ = 0.2 的表现都优于 ρ = 0.1。这些结果验证了我们在上文的论断。

此外，我们观察到最优 ρ 高度依赖于数据集的序列长度。对于序列长度较短的数据集（如 Beauty 和 Steam），最佳性能分别在 ρ=0.6（Beauty）和 ρ=0.4（Steam）时取得；而序列长度较长的数据集（如 ML-1m 和 ML-20m）则倾向于较小的 ρ=0.2。这是合理的，因为与短序列数据集相比，长序列数据集中的较大 ρ 意味着需要预测的 item 数量要多得多。以 ML-1m 和 Beauty 为例，ρ=0.6 意味着 ML-1m 每个序列平均需要预测 $98=\lfloor 163.5 \times 0.6 \rfloor$ 个 item，而 Beauty 只需要 $5=\lfloor 8.8 \times 0.6 \rfloor$ 个 item。前者对模型训练来说太难了。

<!-- FIGURE4 -->
**图 4：** 在 d = 64 下不同掩码比例 ρ 的性能。加粗符号表示每条线中的最优分数。

### 4.7 最大序列长度 N 的影响

我们还考察最大序列长度 N 对模型推荐性能和效率的影响。

表 4 展示了在 Beauty 和 ML-1m 上，不同最大长度 N 下的推荐性能和训练速度。我们观察到，合适的最大长度 N 也高度依赖于数据集的平均序列长度。Beauty 偏好较小的 N = 20，而 ML-1m 在 N = 200 时取得最佳性能。这表明在短序列数据集上，用户行为受更近期 item 的影响；而在长序列数据集上，则受不那么近期的 item 影响。模型并不会一致地从更大的 N 中受益，因为更大的 N 往往会同时引入额外信息和更多噪声。然而，随着长度 N 变大，我们的模型表现非常稳定。这表明我们的模型能够从有噪声的历史记录中关注到信息量大的 item。

BERT4Rec 的一个可扩展性问题是其每层计算复杂度为 $O(n^2d)$ ，与长度 n 呈二次关系。幸运的是，表 4 的结果表明自注意力层可以借助 GPU 被有效地并行化。

**表 4：不同最大长度 N 下的性能。**

| Beauty | N=10 | N=20 | N=30 | N=40 | N=50 |
|---|---|---|---|---|---|
| 样本数/秒 | 5504 | 3256 | 2284 | 1776 | 1441 |
| HR@10 | 0.3006 | 0.3061 | 0.3057 | 0.3054 | 0.3047 |
| NDCG@10 | 0.1826 | 0.1875 | 0.1837 | 0.1833 | 0.1832 |

| ML-1m | N=10 | N=50 | N=100 | N=200 | N=400 |
|---|---|---|---|---|---|
| 样本数/秒 | 14255 | 8890 | 5711 | 2918 | 1213 |
| HR@10 | 0.6788 | 0.6854 | 0.6947 | 0.6955 | 0.6898 |
| NDCG@10 | 0.4631 | 0.4743 | 0.4758 | 0.4759 | 0.4715 |

### 4.8 消融研究

最后，我们对 BERT4Rec 的若干关键组件进行消融实验，以更好地理解它们的影响，包括位置嵌入（PE）、逐位置前馈网络（PFFN）、层归一化（LN）、残差连接（RC）、Dropout、自注意力的层数 L，以及多头注意力中的头数 h。表 5 展示了默认版本（L = 2, h = 2）及其十一个变体在所有四个数据集上、维度 d = 64、其他超参数（如 ρ）保持最优设置时的结果。

我们分别介绍这些变体并分析它们的影响：

(1) **PE。** 结果表明，移除位置嵌入会导致 BERT4Rec 在长序列数据集（即 ML-1m 和 ML-20m）上的性能大幅下降。没有位置嵌入时，每个 item $v_i$ 的隐藏表示 $H^L_i$ 只依赖于 item 嵌入。在这种情况下，我们使用 "[mask]" 的相同隐藏表示来预测不同的目标 item。这使得模型不适定。这一问题在长序列数据集上更为严重，因为这类数据集有更多需要预测的掩码 item。

(2) **PFFN。** 结果表明，长序列数据集（如 ML-20m）从 PFFN 中获益更多。这是合理的，因为 PFFN 的目的之一是整合来自多个头的信息，而正如消融研究 (5) 中关于头数 h 的分析所讨论的，长序列数据集更偏好多头。

(3) **LN、RC 和 Dropout。** 引入这些组件主要是为了缓解过拟合。显然，它们在 Beauty 这样的小数据集上更有效。为了验证它们在大型数据集上的有效性，我们在 ML-20m 上进行了一个层数 L=4 的实验。结果表明，移除 RC（w/o RC）后 NDCG@10 下降了约 10%。

(4) **层数 L。** 结果表明，堆叠 Transformer 层可以提升性能，尤其是在大型数据集（如 ML-20m）上。这验证了通过深度自注意力架构学习更复杂的 item 转移模式是有帮助的。Beauty 在 L = 4 时的性能下降很大程度上是由于过拟合。

(5) **头数 h。** 我们观察到，长序列数据集（如 ML-20m）从更大的 h 中获益，而短序列数据集（如 Beauty）则偏好更小的 h。这一现象与 [48] 中的经验结果一致，即较大的 h 对于使用多头自注意力捕获长距离依赖至关重要。

**表 5：四个数据集上的消融分析（NDCG@10）。加粗分数表示性能优于默认版本，而 ↓ 表示性能下降超过 10%。**

| 架构 | Beauty | Steam | ML-1m | ML-20m |
|---|---|---|---|---|
| L = 2, h = 2 | 0.1832 | 0.2241 | 0.4759 | 0.4513 |
| w/o PE | 0.1741 | 0.2060 | 0.2155↓ | 0.2867↓ |
| w/o PFFN | 0.1803 | 0.2137 | 0.4544 | 0.4296 |
| w/o LN | 0.1642↓ | 0.2058 | 0.4334 | 0.4186 |
| w/o RC | 0.1619↓ | 0.2193 | 0.4643 | 0.4483 |
| w/o Dropout | 0.1658 | 0.2185 | 0.4553 | 0.4471 |
| 1 layer (L = 1) | 0.1782 | 0.2122 | 0.4412 | 0.4238 |
| 3 layers (L = 3) | **0.1859** | **0.2262** | **0.4864** | **0.4661** |
| 4 layers (L = 4) | **0.1834** | **0.2279** | **0.4898** | **0.4732** |
| 1 head (h = 1) | **0.1853** | 0.2187 | 0.4568 | 0.4402 |
| 4 heads (h = 4) | 0.1830 | **0.2245** | **0.4770** | **0.4520** |
| 8 heads (h = 8) | 0.1823 | **0.2248** | 0.4743 | **0.4550** |



## 5 结论与未来工作

深度双向自注意力架构已在语言理解领域取得了巨大成功。在本文中，我们为序列推荐引入了名为 BERT4Rec 的深度双向序列模型。对于模型训练，我们引入 Cloze 任务，利用左右上下文预测被掩码的 item。在四个真实世界数据集上的大量实验结果表明，我们的模型优于最先进的基线方法。

仍有一些方向有待探索。一个有价值的方向是将丰富的 item 特征（例如产品的类别和价格、电影的演员阵容）融入 BERT4Rec，而不仅仅是对 item id 建模。未来工作的另一个有趣方向是在模型中引入用户组件，以便在用户拥有多个会话时进行显式的用户建模。



## 参考文献

[1] Lei Jimmy Ba, Ryan Kiros, and Geoffrey E. Hinton. 2016. **Layer Normalization**. CoRR abs/1607.06450 (2016).

[2] Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. 2015. Neural Machine Translation by Jointly Learning to Align and Translate. In Proceedings of ICLR.

[3] Xu Chen, Hongteng Xu, Yongfeng Zhang, Jiaxi Tang, Yixin Cao, Zheng Qin, and Hongyuan Zha. 2018. **Sequential Recommendation with User Memory Networks**. In Proceedings of WSDM. ACM, New York, NY, USA, 108–116.

[4] Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. 2014. Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation. In Proceedings of EMNLP. Association for Computational Linguistics, 1724–1734.

[5] Paul Covington, Jay Adams, and Emre Sargin. 2016. Deep Neural Networks for YouTube Recommendations. In Proceedings of RecSys. ACM, New York, NY, USA, 191–198.

[6] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2018. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. CoRR abs/1810.04805 (2018).

[7] Tim Donkers, Benedikt Loepp, and Jürgen Ziegler. 2017. **Sequential User-based Recurrent Neural Network Recommendations.** In Proceedings of RecSys. ACM, New York, NY, USA, 152–160.

[8] F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Trans. Interact. Intell. Syst. 5, 4, Article 19 (Dec. 2015), 19 pages.

[9] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. 2016. **Deep Residual Learning for Image Recognition**. In Proceedings of CVPR. 770–778.

[10] Ruining He, Wang-Cheng Kang, and Julian McAuley. 2017. Translation-based Recommendation. In Proceedings of RecSys. ACM, New York, NY, USA, 161–169.

[11] Ruining He and Julian McAuley. 2016. Fusing Similarity Models with Markov Chains for Sparse Sequential Recommendation. In Proceedings of ICDM. 191–200.

[12] Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu, and Tat-Seng Chua. 2017. Neural Collaborative Filtering. In Proceedings of WWW. 173–182.

[13] Dan Hendrycks and Kevin Gimpel. 2016. Bridging Nonlinearities and Stochastic Regularizers with Gaussian Error Linear Units. CoRR abs/1606.08415 (2016).

[14] Balázs Hidasi and Alexandros Karatzoglou. 2018. Recurrent Neural Networks with Top-k Gains for Session-based Recommendations. In Proceedings of CIKM. ACM, New York, NY, USA, 843–852.

[15] Balázs Hidasi, Alexandros Karatzoglou, Linas Baltrunas, and Domonkos Tikk. 2016. **Session-based Recommendations with Recurrent Neural Networks**. In Proceedings of ICLR.

[16] Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. 2015. **Distilling the knowledge in a neural network**. In Deep Learning and Representation Learning Workshop.

[17] Sepp Hochreiter and Jürgen Schmidhuber. 1997. **Long Short-Term Memory**. Neural Computation 9, 8 (Nov. 1997), 1735–1780.

[18] Liang Hu, Longbing Cao, Shoujin Wang, Guandong Xu, Jian Cao, and Zhiping Gu. 2017. Diversifying Personalized Recommendation with User-session Context. In Proceedings of IJCAI. 1858–1864.

[19] Jin Huang, Wayne Xin Zhao, Hongjian Dou, Ji-Rong Wen, and Edward Y. Chang. 2018. **Improving Sequential Recommendation with Knowledge-Enhanced Memory Networks**. In Proceedings of SIGIR. ACM, New York, NY, USA, 505–514.

[20] Santosh Kabbur, Xia Ning, and George Karypis. 2013. FISM: Factored Item Similarity Models for top-N Recommender Systems. In Proceedings of KDD. ACM, New York, NY, USA, 659–667.

[21] Wang-Cheng Kang, Chen Fang, Zhaowen Wang, and Julian McAuley. 2017. Visually-Aware Fashion Recommendation and Design with Generative Image Models. In Proceedings of ICDM. IEEE Computer Society, 207–216.

[22] Wang-Cheng Kang and Julian McAuley. [n. d.]. Self-Attentive Sequential Recommendation. In Proceedings of ICDM. 197–206.

[23] Donghyun Kim, Chanyoung Park, Jinoh Oh, Sungyoung Lee, and Hwanjo Yu. 2016. Convolutional Matrix Factorization for Document Context-Aware Recommendation. In Proceedings of RecSys. ACM, New York, NY, USA, 233–240.

[24] Diederik P. Kingma and Jimmy Ba. 2015. Adam: A Method for Stochastic Optimization. In Proceedings of ICLR.

[25] Yehuda Koren. 2008. **Factorization Meets the Neighborhood: A Multifaceted Collaborative Filtering Mode**l. In Proceedings of KDD. ACM, 426–434.

[26] Yehuda Koren and Robert Bell. 2011. Advances in Collaborative Filtering. In Recommender Systems Handbook. Springer US, Boston, MA, 145–186.

[27] Yehuda Koren, Robert Bell, and Chris Volinsky. 2009. Matrix Factorization Techniques for Recommender Systems. Computer 42, 8 (Aug. 2009), 30–37.

[28] Jing Li, Pengjie Ren, Zhumin Chen, Zhaochun Ren, Tao Lian, and Jun Ma. 2017. Neural Attentive Session-based Recommendation. In Proceedings of CIKM. ACM, New York, NY, USA, 1419–1428.

[29] Jian Li, Zhaopeng Tu, Baosong Yang, Michael R. Lyu, and Tong Zhang. 2018. Multi-Head Attention with Disagreement Regularization. In Proceedings of EMNLP. 2897–2903.

[30] Zhouhan Lin, Minwei Feng, Cícero Nogueira dos Santos, Mo Yu, Bing Xiang, Bowen Zhou, and Yoshua Bengio. 2017. A Structured Self-attentive Sentence Embedding. In Proceedings of ICLR.

[31] Greg Linden, Brent Smith, and Jeremy York. 2003. Amazon.Com Recommendations: Item-to-Item Collaborative Filtering. IEEE Internet Computing 7, 1 (Jan. 2003), 76–80.

[32] Peter J. Liu, Mohammad Saleh, Etienne Pot, Ben Goodrich, Ryan Sepassi, Lukasz Kaiser, and Noam Shazeer. 2018. Generating Wikipedia by Summarizing Long Sequences. In Proceedings of ICLR.

[33] Qiao Liu, Yifu Zeng, Refuoe Mokhosi, and Haibin Zhang. 2018. STAMP: Short-Term Attention/Memory Priority Model for Session-based Recommendation. In Proceedings of KDD. ACM, New York, NY, USA, 1831–1839.

[34] Julian McAuley, Christopher Targett, Qinfeng Shi, and Anton van den Hengel. 2015. Image-Based Recommendations on Styles and Substitutes. In Proceedings of SIGIR. ACM, New York, NY, USA, 43–52.

[35] Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. 2013. Efficient Estimation of Word Representations in Vector Space. CoRR abs/1301.3781 (2013).

[36] Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, and Jeffrey Dean. 2013. Distributed Representations of Words and Phrases and Their Compositionality. In Proceedings of NIPS. Curran Associates Inc., USA, 3111–3119.

[37] Massimo Quadrana, Alexandros Karatzoglou, Balázs Hidasi, and Paolo Cremonesi. 2017. Personalizing Session-based Recommendations with Hierarchical Recurrent Neural Networks. In Proceedings of RecSys. ACM, New York, NY, USA, 130–137.

[38] Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever. 2018. Improving language understanding by generative pre-training. In OpenAI Technical report.

[39] Steffen Rendle, Christoph Freudenthaler, Zeno Gantner, and Lars Schmidt-Thieme. 2009. BPR: Bayesian Personalized Ranking from Implicit Feedback. In Proceedings of UAI. AUAI Press, Arlington, Virginia, United States, 452–461.

[40] Steffen Rendle, Christoph Freudenthaler, and Lars Schmidt-Thieme. 2010. Factorizing Personalized Markov Chains for Next-basket Recommendation. In Proceedings of WWW. ACM, New York, NY, USA, 811–820.

[41] Ruslan Salakhutdinov and Andriy Mnih. 2007. Probabilistic Matrix Factorization. In Proceedings of NIPS. Curran Associates Inc., USA, 1257–1264.

[42] Ruslan Salakhutdinov, Andriy Mnih, and Geoffrey Hinton. 2007. Restricted Boltzmann Machines for Collaborative Filtering. In Proceedings of ICML. 791–798.

[43] Badrul Sarwar, George Karypis, Joseph Konstan, and John Riedl. 2001. Item-based Collaborative Filtering Recommendation Algorithms. In Proceedings of WWW. ACM, New York, NY, USA, 285–295.

[44] Suvash Sedhain, Aditya Krishna Menon, Scott Sanner, and Lexing Xie. 2015. AutoRec: Autoencoders Meet Collaborative Filtering. In Proceedings of WWW. ACM, New York, NY, USA, 111–112.

[45] Guy Shani, David Heckerman, and Ronen I. Brafman. 2005. An MDP-Based Recommender System. J. Mach. Learn. Res. 6 (Dec. 2005), 1265–1295.

[46] Peter Shaw, Jakob Uszkoreit, and Ashish Vaswani. 2018. **Self-Attention with Relative Position Representations**. In Proceedings of NAACL. Association for Computational Linguistics, 464–468.

[47] Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov. 2014. **Dropout: A Simple Way to Prevent Neural Networks from Overfitting**. J. Mach. Learn. Res. 15, 1 (Jan. 2014), 1929–1958.

[48] Gongbo Tang, Mathias Müller, Annette Rios, and Rico Sennrich. 2018. Why Self-Attention? A Targeted Evaluation of Neural Machine Translation Architectures. In Proceedings of EMNLP. 4263–4272.

[49] Jiaxi Tang and Ke Wang. 2018. **Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding**. In Proceedings of WSDM. 565–573.

[50] Wilson L. Taylor. 1953. "Cloze Procedure": A New Tool for Measuring Readability. Journalism Bulletin 30, 4 (1953), 415–433.

[51] Aaron van den Oord, Sander Dieleman, and Benjamin Schrauwen. 2013. Deep content-based music recommendation. In Proceedings of NIPS. 2643–2651.

[52] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is All you Need. In NIPS. Curran Associates, Inc., 5998–6008.

[53] Hao Wang, Naiyan Wang, and Dit-Yan Yeung. 2015. Collaborative Deep Learning for Recommender Systems. In Proceedings of KDD. ACM, New York, NY, USA, 1235–1244.

[54] Shoujin Wang, Liang Hu, Longbing Cao, Xiaoshui Huang, Defu Lian, and Wei Liu. 2018. **Attention-Based Transactional Context Embedding for Next-Item Recommendation**. 2532–2539 pages.

[55] Suhang Wang, Yilin Wang, Jiliang Tang, Kai Shu, Suhas Ranganath, and Huan Liu. 2017. What Your Images Reveal: Exploiting Visual Contents for Point-of-Interest Recommendation. In Proceedings of WWW. 391–400.

[56] Chao-Yuan Wu, Amr Ahmed, Alex Beutel, Alexander J. Smola, and How Jing. 2017. Recurrent Recommender Networks. In Proceedings of WSDM. ACM, New York, NY, USA, 495–503.

[57] Yao Wu, Christopher DuBois, Alice X. Zheng, and Martin Ester. 2016. Collaborative Denoising Auto-Encoders for Top-N Recommender Systems. In Proceedings of WSDM. ACM, New York, NY, USA, 153–162.

[58] Feng Yu, Qiang Liu, Shu Wu, Liang Wang, and Tieniu Tan. 2016. A Dynamic Recurrent Model for Next Basket Recommendation. In Proceedings of SIGIR. ACM, New York, NY, USA, 729–732.

---



## ACM 引用格式

Fei Sun, Jun Liu, Jian Wu, Changhua Pei, Xiao Lin, Wenwu Ou, and Peng Jiang. 2019. BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer. In The 28th ACM International Conference on Information and Knowledge Management (CIKM '19), November 3–7, 2019, Beijing, China. ACM, New York, NY, USA, 11 pages. https://doi.org/10.1145/3357384.3357895
