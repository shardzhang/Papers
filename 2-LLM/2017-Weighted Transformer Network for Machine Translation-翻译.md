# WEIGHTED TRANSFORMER NETWORK FOR MACHINE TRANSLATION

> Karim Ahmed, Nitish Shirish Keskar & Richard Socher | Salesforce Research

本文提出加权 Transformer（Weighted Transformer），一种具有改进注意力机制的 Transformer 变体。该模型在 BLEU 分数上超越基线网络，同时收敛速度提升 15–40%。具体地，我们将多头注意力替换为多个自注意力分支，模型在训练过程中学习如何组合这些分支。在 WMT 2014 英德翻译任务上，我们的模型将当前最优性能提升了 0.5 个 BLEU 点；在英法翻译任务上提升了 0.4 个 BLEU 点。

---

## 摘要

神经机器翻译的最新最优结果通常使用带有某种形式的卷积或循环的注意力序列到序列模型。Vaswani 等人（2017）提出了一种全新的架构，完全避免了循环和卷积，仅使用自注意力和前馈层。虽然该架构在多个机器翻译任务上取得了最优结果，但它需要大量参数和训练迭代才能收敛。我们提出加权 Transformer（Weighted Transformer），一种带有改进注意力层的 Transformer，不仅在 BLEU 分数上超越基线网络，而且收敛速度提升 15–40%。具体地，我们将多头注意力替换为多个自注意力分支，模型在训练过程中学习如何组合这些分支。在 WMT 2014 英德翻译任务上，我们的模型将当前最优性能提升了 0.5 个 BLEU 点；在英法翻译任务上提升了 0.4 个 BLEU 点。

## 1 引言

循环神经网络（RNN），如长短期记忆网络（LSTM）（Hochreiter & Schmidhuber, 1997），是许多需要序列数据建模任务的重要构建模块。RNN 已成功应用于多种此类任务，包括语言建模（Melis et al., 2017; Merity et al., 2017）、语音识别（Xiong et al., 2017; Graves et al., 2013）和机器翻译（Wu et al., 2016; Bahdanau et al., 2014）。RNN 通过基于当前输入 `token` 和之前状态计算隐藏状态向量 $h_t$，在每个时间步生成输出预测。这种顺序计算是其映射任意输入-输出序列对能力的基础。然而，由于其自回归性质——需要先计算之前的隐藏状态才能进入当前时间步——它们无法受益于并行化。

使用跨步卷积的循环网络变体摒弃了传统的基于时间步的计算（Kaiser & Bengio, 2016; Lei & Zhang, 2017; Bradbury et al., 2016; Gehring et al., 2016; 2017; Kalchbrenner et al., 2016）。然而，在这些模型中，学习远距离位置之间依赖关系所需的操作可能难以学习（Hochreiter et al., 2001; Hochreiter, 1998）。注意力机制通常与循环模型结合使用，由于能够促进此类依赖关系的学习，已成为复杂序列任务中不可或缺的一部分（Luong et al., 2015; Bahdanau et al., 2014; Parikh et al., 2016; Paulus et al., 2017; Kim et al., 2017）。

在 Vaswani 等人（2017）的工作中，作者介绍了 Transformer 网络，这是一种全新的架构，避免了循环方程，仅使用注意力将输入序列映射到隐藏状态。具体地，作者将位置编码与多头注意力机制结合使用。这增强了并行计算能力并减少了收敛时间。作者报告的神经机器翻译结果表明，Transformer 网络在 WMT 2014 英德和英法任务上达到了当前最优性能，同时比先前方法快数个数量级。

Transformer 网络仍然需要大量参数才能实现当前最优性能。在 newstest2013 英德翻译任务中，base 模型需要 65M 参数，large 模型需要 213M 参数。我们提出 Transformer 网络的一种变体，称为加权 Transformer（Weighted Transformer），它使用自注意力分支来代替多头注意力。这些分支替代了原始 Transformer 网络注意力机制中的多个头，模型在训练过程中学习如何组合这些分支。这种分支架构使网络能够在显著降低计算成本的情况下达到可比的性能。实际上，通过这种修改，我们在 WMT 2014 英德和英法任务上分别将当前最优性能提升了 0.5 和 0.4 个 BLEU 分数。最后，我们提供了证据表明所提架构具有正则化效果。

## 2 相关工作

大多数神经机器翻译（NMT）架构使用依赖于深度循环神经网络（如 LSTM）的编码器和解码器（Luong et al., 2015; Sutskever et al., 2014; Bahdanau et al., 2014; Wu et al., 2016; Barone et al., 2017; Cho et al., 2014）。已有多种架构被提出来减少基于循环计算的负载（Gehring et al., 2016; 2017; Kaiser & Bengio, 2016; Kalchbrenner et al., 2016）。自注意力依赖于输入序列元素之间的点积来计算加权和（Lin et al., 2017; Bahdanau et al., 2014; Parikh et al., 2016; Kim et al., 2017），也是现代 NMT 架构的关键组成部分。Transformer 网络（Vaswani et al., 2017）完全避免了循环，仅使用自注意力。

我们提出一种改进的 Transformer 网络，其中多头注意力层被替换为分支自注意力层。各分支的贡献在训练过程中作为训练过程的一部分被学习。多分支网络的思想已在多个领域得到探索（Ahmed & Torresani, 2017; Gastaldi, 2017; Shazeer et al., 2017; Xie et al., 2016）。据我们所知，这是首次在 Transformer 网络中使用分支结构。在 Shazeer 等人（2017）的工作中，作者使用一个具有数十亿权重的大型网络，结合稀疏专家模型来实现具有竞争力的性能。Ahmed & Torresani（2017）在计算机视觉背景下分析了通过门控进行的学习分支，而 Gastaldi（2017）则在图像分类背景下分析了一个具有随机采样权重的两分支模型。

### 2.1 Transformer 网络

原始 Transformer 网络采用编码器-解码器架构，每一层由一个新颖的注意力机制（作者称之为多头注意力）后接一个前馈网络组成。下面我们描述这两个组件。

从源 `token` 出发，生成维度为 $d_{\text{model}}$ 的学习嵌入，然后通过加性位置编码进行修改。位置编码是必要的，因为网络本身不包含循环或卷积，因此没有任何利用序列顺序的手段。作者使用的加性编码定义如下：

$$ \text{PE}_{(pos, 2i)} = \sin\left(pos / 10000^{2i / d_{\text{model}}}\right) $$
$$ \text{PE}_{(pos, 2i+1)} = \cos\left(pos / 10000^{2i / d_{\text{model}}}\right) $$

其中 $pos$ 是句子中单词的位置，$i$ 是向量的维度。作者也尝试了学习嵌入（Gehring et al., 2016; 2017），但发现这样做没有收益。编码后的词嵌入随后用作编码器的输入，编码器由 $N$ 层组成，每层包含两个子层：（a）多头注意力机制，和（b）前馈网络。

多头注意力机制建立在缩放点积注意力之上，后者对查询 $Q$、键 $K$ 和值 $V$ 进行操作：

$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^{\mathsf{T}}}{\sqrt{d_k}}\right) V \qquad (1) $$

其中 $d_k$ 是键的维度。

在第一层中，输入被拼接起来，使得 $(Q, K, V)$ 均等于词向量矩阵。这与点积注意力相同，只是增加了缩放因子 $\sqrt{d_k}$，从而提高了数值稳定性。

多头注意力机制获取 $(Q, K, V)$ 的 $h$ 种不同表示，对每种表示计算缩放点积注意力，将结果拼接起来，并使用前馈层对拼接结果进行投影。这可以用与方程（1）相同的符号表示：

$$ \text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V) \qquad (2) $$
$$ \text{MultiHead}(Q, K, V) = \text{Concat}_i(\text{head}_i) W^O \qquad (3) $$

其中 $W_i$ 和 $W^O$ 是学习得到的参数投影矩阵。

注意 $W_i^Q $\in$ \mathbb{R}^{d_{\text{model}} $\times$ d_k}$，$W_i^K $\in$ \mathbb{R}^{d_{\text{model}} $\times$ d_k}$，$W_i^V $\in$ \mathbb{R}^{d_{\text{model}} $\times$ d_v}$，$W^O $\in$ \mathbb{R}^{h d_v $\times$ d_{\text{model}}}$，其中 $h$ 表示多头注意力中的头数。Vaswani 等人（2017）按比例减小 $d_k = d_v = d_{\text{model}} / h$，使得多头注意力的计算量与简单自注意力相同。

Transformer 网络每层的第二个组件是前馈网络。作者提出使用带有 ReLU 激活的双层网络。给定可训练权重 $W_1, W_2, b_1, b_2$，该子层定义为：

$$ \text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2 \qquad (4) $$

内部层的维度为 $d_{\text{ff}}$，在实验中设置为 2048。为简洁起见，我们将架构的其他细节请读者参考 Vaswani 等人（2017）。

为了正则化和便于训练，网络在每个子层后使用层归一化（Ba et al., 2016），并在每个完整层周围使用残差连接（He et al., 2016）。类似地，解码器的每一层包含上述两个子层，以及一个额外的多头注意力子层，该子层接收来自对应编码层输出的 $(V, K)$ 作为输入。在解码器多头注意力子层中，缩放点积注意力被掩码处理，以防止关注未来位置，换句话说，防止非法的从左到右信息流动。

关于 Transformer 网络的一个自然问题是，为什么自注意力应该优于循环或卷积模型。Vaswani 等人（2017）给出了三个理由：（a）每层的计算复杂度，（b）并发性，以及（c）长距离依赖之间的路径长度。假设序列长度为 $n$，向量维度为 $d$，自注意力层的复杂度为 $O(n^2 d)$，而循环层的复杂度为 $O(n d^2)$。由于通常 $d > n$，自注意力层的复杂度低于循环层。此外，自注意力层的顺序计算次数为 $O(1)$，而循环层为 $O(n)$。这有助于提高并行计算架构的利用率。最后，自注意力层依赖之间的最大路径长度为 $O(1)$，而循环层为 $O(n)$。这一差异是阻碍循环模型学习长距离依赖能力的关键因素。

## 3 提出的网络架构

现在描述所提出的架构——加权 Transformer（Weighted Transformer），它训练效率更高，且能更好地利用表示能力。

在方程（3）和（4）中，我们描述了 Vaswani 等人（2017）提出的注意力层，包含多头注意力子层和 FFN 子层。对于加权 Transformer，我们提出一种分支注意力，它修改了 Transformer 网络中的整个注意力层（包括多头注意力和前馈网络）。所提出的注意力层可以描述为：

$$ \text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V), \qquad (5) $$
$$ \text{head}_i = \text{head}_i W_i^O $\times$ $\kappa$_i, \qquad (6) $$
$$ \text{BranchedAttention}(Q, K, V) = \sum_{i=1}^{M} $\alpha$_i \text{FFN}(\text{head}_i). \qquad (7) $$

其中 $M$ 表示分支总数，$\kappa_i, \alpha_i \in \mathbb{R}^+$ 是学习得到的参数，$W_i^O \in \mathbb{R}^{d_v \times d_{\text{model}}}$。上面的 FFN 函数与方程（4）相同。此外，我们要求 $\sum \kappa_i = 1$ 且 $\sum \alpha_i = 1$，使得方程（7）是各分支注意力值的加权和。

![图 1：我们提出的网络架构。](Figure 1: Our proposed network architecture.)

在上面的方程中，$\kappa$ 可以解释为学习得到的拼接权重，$\alpha$ 可以解释为学习得到的加法权重。实际上，$\kappa$ 在 $\alpha$ 以加权方式求和之前，对各分支的贡献进行缩放。我们通过投影确保每个训练步骤中所有边界都得到满足。

虽然 $$\alpha$$ 和 $$\kappa$$ 有可能合并为一个变量进行训练，但我们发现将它们分开训练效果更好。这还提高了模型的可解释性，因为 $($\alpha$, $\kappa$)$ 可以被视为各分支上的概率质量。

可以证明，如果对所有 $i$ 有 $\kappa_i = 1$ 且 $\alpha_i = 1$，我们就得到了多头注意力的方程（3）。然而，由于 $\sum_i $\kappa$_i = 1$ 和 $\sum_i $\alpha$_i = 1$ 的约束，这些值在加权 Transformer 中是不允许的。对我们提出的架构的一种解释是，它将多头注意力替换为多分支注意力。不同头的贡献不再是拼接，而是被视为多分支网络学习组合的分支。这种机制增加了 $O(M)$ 个可训练权重。与总权重数相比，这微不足道。实际上，在我们的实验中，所提机制为一个已经包含 213M 权重的模型增加了 192 个权重。没有这些额外的可训练权重时，所提机制与 Transformer 中的多头注意力机制完全相同。所提出的注意力机制在编码器和解码器层中都使用，并在解码器层中像 Transformer 网络一样被掩码处理。同样，编码器-解码器层中的位置编码、层归一化和残差连接也被保留。为清晰起见，我们在图 1 中省略了这些细节。除了使用 $($\alpha$, $\kappa$)$ 学习权重，还可以通过 softmax 层使用专家混合归一化（Shazeer et al., 2017）。然而，我们发现它的表现不如我们提出的方法。

与 Transformer 对所有头赋予相同权重不同，所提机制允许为不同头分配不同的重要性。这进而优先处理它们的梯度，简化了优化过程。此外，正如计算机视觉中多分支网络所知的（Gastaldi, 2017），这种机制往往使得各分支学习去相关的输入-输出映射。这减少了共适应并提高了泛化能力。这一观察也是专家混合模型的基础（Shazeer et al., 2017）。

## 4 实验

### 4.1 训练细节

权重 $\kappa$ 和 $\alpha$ 随机初始化，与 Transformer 的其他权重相同。

除了层归一化和残差连接之外，我们还使用了 $\epsilon_{\text{ls}} = 0.1$ 的标签平滑、注意力 dropout 和概率 $P_{\text{drop}} = 0.1$ 的残差 dropout。注意力 dropout 会随机丢弃（Srivastava et al., 2014）方程（1）中 softmax 的元素。

与 Vaswani 等人（2017）相同，我们使用 Adam 优化器（Kingma & Ba, 2014），设 $($\beta$_1, $\beta$_2) = (0.9, 0.98)$ 和 $\epsilon = 10^{-9}$。我们还对 Adam 使用学习率预热策略，其中学习率 $\text{lr}$ 的形式为：

$$ \text{lr} = d_{\text{model}}^{-0.5} \cdot \min(\text{iterations}^{-0.5}, \text{iterations} \cdot 4000^{-1.5}) $$

适用于除 $($\alpha$, $\kappa$)$ 之外的所有参数，以及

$$ \text{lr} = (d_{\text{model}} / N)^{-0.5} \cdot \min(\text{iterations}^{-0.5}, \text{iterations} \cdot 400^{-1.5}) $$

适用于 $($\alpha$, $\kappa$)$。

这对应于原始 Transformer 网络使用的预热策略，只是我们对 $($\alpha$, $\kappa$)$ 使用了更大的峰值学习率以补偿它们的边界限制。此外，我们发现在最后 10K 次迭代中冻结权重 $($\kappa$, $\alpha$)$ 有助于收敛。在这段时间内，我们继续训练网络的其余部分。我们假设这种冻结过程有助于在加权方案下稳定网络其余部分的权重。

我们注意到，加权 Transformer 收敛到最终分数所需的迭代次数大幅减少。我们发现加权 Transformer 的收敛速度提升了 15–40%（以达到最优性能的总迭代次数衡量）。我们对基线模型训练：小变体 100K 步，大变体 300K 步。我们分别对加权 Transformer 的相应变体训练 60K 和 250K 次迭代。我们发现，运行更长时间目标没有显著改善。此外，我们没有使用 Vaswani 等人（2017）中使用的任何平均策略，而是直接返回最终模型用于测试。

为了减少填充带来的计算负担，句子被分批组合使得长度大致相同。所有句子使用字节对编码（Sennrich et al., 2015）进行编码，并共享同一个词汇表。词嵌入的权重与最终 softmax 层中的对应条目绑定（Inan et al., 2016; Press & Wolf, 2016）。我们在 NVIDIA K80 GPU 上训练所有网络，每批包含大约 25,000 个源和目标 `token`。

### 4.2 基准数据集上的结果

我们在 WMT 2014 英德和英法任务上对所提架构进行了基准测试。WMT 2014 英德数据集包含 450 万句对。英法数据集包含 3600 万句对。

我们的实验结果总结在表 1 中。加权 Transformer 在较小网络的英德任务上实现了比当前最优高 1.1 个 BLEU 分数的提升，在较大网络上提升 0.5 个 BLEU。在较大的英法任务上，我们的较小模型提升了 0.8 个 BLEU，较大模型提升了 0.4 个 BLEU。另外需要注意的是，加权 Transformer 较小模型的性能接近较大基线模型的性能，尤其是在英德任务上。这表明加权 Transformer 能更好地利用可用模型容量，因为它仅需基线 Transformer 30% 的参数即可匹配其性能。我们的相对改进并不依赖于使用 BLEU 分数进行比较；使用 Wu 等人（2016）提出的 GLEU 分数的实验也产生了类似的改进。

最后，我们对加权 Transformer 的正则化效果进行评论。鉴于改进的结果，一个自然的问题是，这些结果是否源于模型正则化的改善。为了探究这一点，我们报告了加权 Transformer 和基线 Transformer 的测试损失与训练损失的对比如图 2 所示。具有正则化效果的模型在相同训练损失下通常具有更低的测试损失。我们在实验中观察到了这一效果，表明所提架构可能具有更好的正则化特性。考虑到其他基于分支策略（如 Shake-Shake（Gastaldi, 2017）和专家混合（Shazeer et al., 2017））的类似结果，这并不意外。

![图 2：newstest2013 英德任务的测试损失与训练损失。加权 Transformer 在相同训练损失下比基线 Transformer 具有更低的测试损失，表明其具有正则化效果。](Figure 2: Testing v/s Training Loss for the newstest2013 English-to-German task. The Weighted Transformer has lower testing loss compared to the baseline Transformer for the same training loss, suggesting a regularizing effect.)

| 模型 | EN-DE BLEU | EN-FR BLEU |
|---|---|---|
| Transformer (small) (Vaswani et al., 2017) | 27.3 | 38.1 |
| Weighted Transformer (small) | 28.4 | 38.9 |
| Transformer (large) (Vaswani et al., 2017) | 28.4 | 41.0 |
| Weighted Transformer (large) | 28.9 | 41.4 |
| ByteNet (Kalchbrenner et al., 2016) | 23.7 | — |
| Deep-Att+PosUnk (Zhou et al., 2016) | — | 39.2 |
| GNMT+RL (Wu et al., 2016) | 24.6 | 39.9 |
| ConvS2S (Gehring et al., 2017) | 25.2 | 40.5 |
| MoE (Shazeer et al., 2017) | 26.0 | 40.6 |

表 1：WMT 2014 英德（EN-DE）和英法（EN-FR）翻译任务的实验结果。我们提出的模型优于包括 Transformer（Vaswani et al., 2017）在内的当前最优模型。small 模型对应表 2 中的配置（A），large 对应配置（B）。

### 4.3 敏感性分析

在表 2 中，我们报告了在 newstest2013 英德任务上的敏感性结果。具体地，我们改变编码器/解码器的层数，并比较加权 Transformer 和基线 Transformer 的性能。结果清晰地展示了分支注意力的优势：对于每个实验，加权 Transformer 都优于基线 Transformer，在某些情况下达到了高达 1.3 个 BLEU 点的提升。与基线 Transformer 的情况一样，增加层数并不一定能提高性能；当层数 $N$ 从 2 增加到 4 和从 4 增加到 6 时，性能有适度提升，但当 $N$ 增加到 8 时性能下降。在配置（A）中将头数从 8 增加到 16 得到了更好的 BLEU 分数。然而，与 $N$ 的情况类似，$h=16$ 和 $h=32$ 的初步实验会降低模型性能。

| 模型 | 设置 |  |  |  |  |  | BLEU | 参数量 $\times 10^6$ |
|---|---|---|---|---|---|---|---|---|
| | $N$ | $d_{\text{model}}$ | $d_{\text{ff}}$ | $h$ | $M$ | $P_{\text{drop}}$ | train steps | |
| Transformer (C) | 2 | 512 | 2048 | 8 | NA | 0.1 | 100K | 23.7 | 36 |
| Weighted Transformer (C) | 2 | 512 | 2048 | 8 | 8 | 0.1 | 60K | 24.8 | 36 |
| Transformer | 4 | 512 | 2048 | 8 | NA | 0.1 | 100K | 25.3 | 50 |
| Weighted Transformer | 4 | 512 | 2048 | 8 | 8 | 0.1 | 60K | 26.2 | 50 |
| Transformer (A) | 6 | 512 | 2048 | 8 | NA | 0.1 | 100K | 25.8 | 65 |
| Weighted Transformer (A) | 6 | 512 | 2048 | 8 | 8 | 0.1 | 60K | 26.5 | 65 |
| Transformer | 8 | 512 | 2048 | 8 | NA | 0.1 | 100K | 25.5 | 80 |
| Weighted Transformer | 8 | 512 | 2048 | 8 | 8 | 0.3 | 60K | 25.6 | 80 |
| Transformer (B) | 6 | 1024 | 4096 | 16 | NA | 0.3 | 300K | 26.4 | 213 |
| Weighted Transformer (B) | 6 | 1024 | 4096 | 16 | 16 | 0.3 | 250K | 27.2 | 213 |

表 2：Transformer（Vaswani et al., 2017）架构不同变体与我们所提加权 Transformer 的实验比较。报告的 BLEU 分数在英德翻译开发集 newstest2013 上评估。

![图 3：配置（C）的第二个编码器层 $($\alpha$, $\kappa$)$ 权重的收敛情况（英德 newstest2013 任务）。我们使用均值滤波器对曲线进行平滑处理。这表明网络确实对某些分支赋予了比其他分支更高的优先级，且该架构并非只利用一部分分支而忽略其他分支。](Figure 3: Convergence of the ($\alpha$, $\kappa$) weights for the second encoder layer of Configuration (C) for the English-to-German newstest2013 task. We smoothen the curves using a mean filter. This shows that the network does prioritize some branches more than others and that the architecture does not exploit a subset of the branches while ignoring others.)

在图 3 中，我们展示了配置（C）的第二个编码器层权重 $($\alpha$, $\kappa$)$ 在英德 newstest2013 任务上的行为。该图表明，就相对权重而言，网络确实对某些分支赋予了比其他分支更高的优先级；有时差距可达 2 倍。此外，各分支的相对排序随时间变化，表明网络并非纯粹利用性的。一个纯粹利用性的网络——即学会利用一部分分支而牺牲其余分支——是不可取的，因为它会有效减少可用参数数量并限制表示能力。在其他层（包括解码器层）中也观察到了类似的结果；为简洁起见我们省略了。

### 4.4 随机化基线

所提修改也可以解释为 Gastaldi（2017）提出的 Shake-Shake 正则化的一种形式。在这种正则化策略中，在前向和后向传播过程中，对多分支网络中各个分支的权重进行随机采样。在测试时，则对它们平均加权。而在我们的策略中，权重是通过学习得到的，而非随机采样。因此，测试时无需对模型进行任何修改。

| 权重 $($\alpha$, $\kappa$)$ | BLEU |
|---|---|
| Learned | 24.8 |
| Random | 21.1 |
| Uniform | 23.4 |

表 3：配置（C）在 newstest2013 英德任务上使用随机和均匀归一化权重的架构性能。这表明加权 Transformer 学习得到的 $($\alpha$, $\kappa$)$ 权重对其性能至关重要。

为了更好地理解网络是否受益于学习得到的权重，或者测试时随机或均匀权重是否就足够，我们设计了以下实验：加权 Transformer 的权重（包括 $($\alpha$, $\kappa$)$）照常训练，但在测试时，我们将其替换为（a）随机采样的权重，和（b）$1/M$（其中 $M$ 是输入分支数量）。

在表 3 中，我们报告了加权 Transformer 配置（C）在英德 newstest2013 数据集上的实验结果（配置详情见表 2）。显然，随机或均匀权重无法在测试时替代学习得到的权重。初步实验表明，在训练时随机采样权重的类似 Shake-Shake 策略也会导致性能下降。

### 4.5 门控机制

为了分析硬（离散）选择是否通过门控优于我们的归一化策略，我们尝试使用门控来代替所提出的拼接-加法策略。具体地，我们将方程（7）中的求和替换为一种门控结构，该结构对概率最高的前 $k$ 个分支的贡献求和。这类似于 Shazeer 等人（2017）的稀疏门控专家混合模型。尽管对 $k$ 和 $M$ 进行了大量的超参数调优，我们发现该策略的性能比我们提出的机制差得多。我们假设这是因为分支数量很少（通常少于 16 个）。因此，稀疏门控模型由于容量减少而损失了表示能力。我们计划在未来工作中研究具有大量分支和稀疏门控的设置。

## 5 结论

我们提出了加权 Transformer（Weighted Transformer），它训练速度更快，且性能优于原始 Transformer 网络。所提架构将 Transformer 网络中的多头注意力替换为多个自注意力分支，这些分支的贡献作为训练过程的一部分被学习。我们报告了在 WMT 2014 英德和英法任务上的数值结果，表明加权 Transformer 分别将当前最优 BLEU 分数提升了 0.5 和 0.4 个点。此外，我们提出的架构训练速度比基线 Transformer 快 15–40%。最后，我们提供了证据表明该提议的正则化效果，并强调 BLEU 分数的相对提升在各种超参数设置下（包括小模型和大模型）均能观察到。

## 参考文献

[1] Karim Ahmed and Lorenzo Torresani. BranchConnect: Large-Scale Visual Recognition with Learned Branch Connections. *arXiv preprint arXiv:1704.06010*, 2017.

[2] Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. *arXiv preprint arXiv:1607.06450*, 2016.

[3] Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural machine translation by jointly learning to align and translate. *arXiv preprint arXiv:1409.0473*, 2014.

[4] Antonio Valerio Miceli Barone, Jindřich Helcl, Rico Sennrich, Barry Haddow, and Alexandra Birch. Deep architectures for neural machine translation. *arXiv preprint arXiv:1707.07631*, 2017.

[5] James Bradbury, Stephen Merity, Caiming Xiong, and Richard Socher. Quasi-recurrent neural networks. *arXiv preprint arXiv:1611.01576*, 2016.

[6] Kyunghyun Cho, Bart Van Merriënboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. Learning phrase representations using RNN encoder-decoder for statistical machine translation. *arXiv preprint arXiv:1406.1078*, 2014.

[7] Xavier Gastaldi. Shake-Shake regularization. *arXiv preprint arXiv:1705.07485*, 2017.

[8] Jonas Gehring, Michael Auli, David Grangier, and Yann N Dauphin. A convolutional encoder model for neural machine translation. *arXiv preprint arXiv:1611.02344*, 2016.

[9] Jonas Gehring, Michael Auli, David Grangier, Denis Yarats, and Yann N Dauphin. Convolutional Sequence to Sequence Learning. *arXiv preprint arXiv:1705.03122*, 2017.

[10] Alex Graves, Abdel-rahman Mohamed, and Geoffrey Hinton. Speech recognition with deep recurrent neural networks. In *Acoustics, speech and signal processing (icassp), 2013 ieee international conference on*, pp. 6645–6649. IEEE, 2013.

[11] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pp. 770–778, 2016.

[12] Sepp Hochreiter. The vanishing gradient problem during learning recurrent neural nets and problem solutions. *International Journal of Uncertainty, Fuzziness and Knowledge-Based Systems*, 6(02):107–116, 1998.

[13] Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. *Neural computation*, 9(8):1735–1780, 1997.

[14] Sepp Hochreiter, Yoshua Bengio, Paolo Frasconi, Jürgen Schmidhuber, et al. Gradient flow in recurrent nets: the difficulty of learning long-term dependencies, 2001.

[15] Hakan Inan, Khashayar Khosravi, and Richard Socher. Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling. *arXiv preprint arXiv:1611.01462*, 2016.

[16] Łukasz Kaiser and Samy Bengio. Can active memory replace attention? In *Advances in Neural Information Processing Systems*, pp. 3781–3789, 2016.

[17] Nal Kalchbrenner, Lasse Espeholt, Karen Simonyan, Aaron van den Oord, Alex Graves, and Koray Kavukcuoglu. Neural machine translation in linear time. *arXiv preprint arXiv:1610.10099*, 2016.

[18] Yoon Kim, Carl Denton, Luong Hoang, and Alexander M Rush. Structured attention networks. *arXiv preprint arXiv:1702.00887*, 2017.

[19] Diederik Kingma and Jimmy Ba. Adam: A method for stochastic optimization. *arXiv preprint arXiv:1412.6980*, 2014.

[20] Tao Lei and Yu Zhang. Training RNNs as fast as CNNs. *arXiv preprint arXiv:1709.02755*, 2017.

[21] Zhouhan Lin, Minwei Feng, Cicero Nogueira dos Santos, Mo Yu, Bing Xiang, Bowen Zhou, and Yoshua Bengio. A structured self-attentive sentence embedding. *arXiv preprint arXiv:1703.03130*, 2017.

[22] Minh-Thang Luong, Hieu Pham, and Christopher D Manning. Effective approaches to attention-based neural machine translation. *arXiv preprint arXiv:1508.04025*, 2015.

[23] Gábor Melis, Chris Dyer, and Phil Blunsom. On the state of the art of evaluation in neural language models. *arXiv preprint arXiv:1707.05589*, 2017.

[24] Stephen Merity, Nitish Shirish Keskar, and Richard Socher. Regularizing and optimizing LSTM language models. *arXiv preprint arXiv:1708.02182*, 2017.

[25] Ankur P Parikh, Oscar Täckström, Dipanjan Das, and Jakob Uszkoreit. A decomposable attention model for natural language inference. *arXiv preprint arXiv:1606.01933*, 2016.

[26] Romain Paulus, Caiming Xiong, and Richard Socher. A deep reinforced model for abstractive summarization. *arXiv preprint arXiv:1705.04304*, 2017.

[27] Ofir Press and Lior Wolf. Using the output embedding to improve language models. *arXiv preprint arXiv:1608.05859*, 2016.

[28] Rico Sennrich, Barry Haddow, and Alexandra Birch. Neural machine translation of rare words with subword units. *arXiv preprint arXiv:1508.07909*, 2015.

[29] Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. *arXiv preprint arXiv:1701.06538*, 2017.

[30] Nitish Srivastava, Geoffrey E Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov. Dropout: a simple way to prevent neural networks from overfitting. *Journal of machine learning research*, 15(1):1929–1958, 2014.

[31] Ilya Sutskever, Oriol Vinyals, and Quoc V Le. Sequence to sequence learning with neural networks. In *Advances in neural information processing systems*, pp. 3104–3112, 2014.

[32] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. *arXiv preprint arXiv:1706.03762*, 2017.

[33] Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Wolfgang Macherey, Maxim Krikun, Yuan Cao, Qin Gao, Klaus Macherey, et al. Google's neural machine translation system: Bridging the gap between human and machine translation. *arXiv preprint arXiv:1609.08144*, 2016.

[34] Saining Xie, Ross Girshick, Piotr Dollár, Zhuowen Tu, and Kaiming He. Aggregated residual transformations for deep neural networks. *arXiv preprint arXiv:1611.05431*, 2016.

[35] Wayne Xiong, Jasha Droppo, Xuedong Huang, Frank Seide, Mike Seltzer, Andreas Stolcke, Dong Yu, and Geoffrey Zweig. The Microsoft 2016 conversational speech recognition system. In *Acoustics, Speech and Signal Processing (ICASSP), 2017 IEEE International Conference on*, pp. 5255–5259. IEEE, 2017.

[36] Jie Zhou, Ying Cao, Xuguang Wang, Peng Li, and Wei Xu. Deep recurrent models with fast-forward connections for neural machine translation. *arXiv preprint arXiv:1606.04199*, 2016.
