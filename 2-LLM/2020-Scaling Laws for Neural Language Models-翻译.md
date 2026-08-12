# Scaling Laws for Neural Language Models

> **Jared Kaplan\***, **Sam McCandlish\***, **Tom Henighan**, **Tom B. Brown**, **Benjamin Chess**, **Rewon Child**, **Scott Gray**, **Alec Radford**, **Jeffrey Wu**, **Dario Amodei** | Johns Hopkins University, OpenAI

本文系统研究了语言模型性能的缩放规律：损失与模型大小、数据集大小、计算量各自呈幂律关系，跨越七个数量级。核心发现是——**模型越大越强，且越大越省数据**。

核心内容：

- 语言模型性能（交叉熵损失）与参数量 $N$、数据量 $D$、计算量 $C$ 各自呈幂律关系，且对架构细节（深度/宽度/注意力头数）几乎不敏感
- 过拟合程度可预测：每增加 8× 模型大小，只需约 5× 数据即可避免过拟合（$D \propto N^{0.74}$）
- 训练曲线遵循简单幂律，可从早期表现外推最终收敛损失
- 最优计算分配策略：**大部分预算应砸向更大的模型**，而非更多训练步数或更多数据

关键发现：

- 固定计算预算下，最优策略是训练大模型并在收敛前大幅提前停止（$N \propto C^{0.73}$，步数几乎不增长）
- 大模型比小模型**显著更省样本**：达到相同损失，大模型需要的优化步数和数据量都更少
- 最优批量大小 ≈ 200万 token（在收敛时），可通过梯度噪声尺度实时确定
- 幂律趋势在远超当前规模时出现矛盾（$C^* \sim 10^4$ PF-days, $N^* \sim 10^{12}$），暗示 Transformer 语言模型的性能天花板

---

## 摘要

我们研究了交叉熵损失与语言模型的模型规模、数据集规模和训练计算量之间的缩放关系。损失与模型参数数量、数据集大小和训练计算量各自呈现幂律关系，某些趋势跨越超过七个数量级。其他架构细节（如深度、宽度或注意力头数量）在足够大的范围内对性能影响很小。这些关系使我们能够确定给定计算预算的最优分配。我们发现，大部分计算预算应分配给训练更大的模型，而相对较少的计算用于在大量数据上进行长时间训练。

## 1 引言

近年来，语言模型取得了显著进展 [DCLT18, RWC+19, RSR+19, LOG+19]。尽管许多进展源于新的架构和训练技术，但一个重要的共同主题是模型规模的持续增长——从 [RNSS18] 中的约1亿参数到 [RSR+19, Tur20] 中的数十亿参数。这种趋势表明，更好的性能来自于更大的模型，但缺乏系统性的研究来量化这种依赖关系。

在本文中，我们通过系统地研究神经语言模型的性能如何随三个关键因素变化来填补这一空白：模型参数数量（ $N$ ）、数据集规模（ $D$ ）和训练计算量（ $C$ ）。我们发现，交叉熵损失与这三个因素之间存在简单的幂律关系，这些关系在跨越多个数量级时成立。这些结果使我们能够：

1. 预测给定规模下模型的性能
2. 确定给定计算预算的最优分配策略
3. 理解不同架构选择的影响

我们的主要发现是，性能主要由规模决定，而架构细节（如深度与宽度的比例、注意力头的数量等）在足够大的范围内几乎不影响性能。此外，我们发现最优的计算分配策略是训练更大的模型并提前停止，而非在大量数据上训练较小的模型直到收敛。

### 1.1 总结

我们的主要结果可以总结如下：

- **幂律关系**：语言模型的交叉熵损失 $L$ 与参数数量 $N$ 、数据集规模 $D$ 和训练计算量 $C$ 各自呈现幂律关系：
  $$
  L(N) \approx \left(\frac{N_c}{N}\right)^{\alpha_N}, \quad L(D) \approx \left(\frac{D_c}{D}\right)^{\alpha_D}, \quad L(C) \approx \left(\frac{C_c}{C}\right)^{\alpha_C}
  $$

- **架构无关性**：在固定计算预算下，性能对模型架构的具体选择（如层数、宽度、注意力头数量）相对不敏感，只要模型足够大。

- **最优分配**：在固定计算预算 $C$ 下，最优策略是训练参数数量为 $N_{opt} \propto C^{a}$ 的模型，其中 $a \approx 0.73$ 。这意味着大部分计算应分配给更大的模型，而非更多的训练步数。

- **过拟合的可预测性**：当模型在数据不足的情况下训练时，过拟合的程度可以被预测，并且存在一个简单的方程来描述损失如何同时依赖于 $N$ 和 $D$ 。

### 1.2 缩放定律总结

我们发现的缩放定律可以概括为几个关键点：

1. **性能与规模的幂律关系**：交叉熵损失与模型规模（参数数量）、数据集规模和计算量之间存在精确的幂律关系。这些关系跨越超过七个数量级的参数变化（从768个参数到超过10亿参数）。

2. **平滑且可预测**：这些缩放关系是平滑的、单调的，且非常精确。这意味着我们可以通过在小规模上进行实验来预测大规模模型的性能。

3. **架构细节次要**：在控制总参数数量的情况下，模型架构的具体细节（如深度、宽度、注意力头数量、前馈层大小等）对性能的影响很小。唯一的例外是，模型不能太"深"或太"浅"——存在一个最优的深度范围。

4. **最优计算分配**：给定固定的计算预算，最优策略是将大部分预算分配给训练更大的模型（更多参数），而非在更多数据上训练更长时间。具体来说，模型大小应随计算预算的快速增长而增长。

### 1.3 符号说明

在本文中，我们使用以下符号：

- $N$ ：模型中非嵌入层的可训练参数数量
- $D$ ：数据集规模，以训练 token 的数量衡量
- $C$ ：训练过程中使用的总计算量，以浮点运算次数（FLOP）衡量
- $L$ ：交叉熵损失（以 nats 为单位，除非另有说明）
- $S$ ：训练步数
- $B$ ：批大小（以 token 数衡量）
- $\alpha_N$ 、 $\alpha_D$ 、 $\alpha_C$ ：分别表示损失相对于 $N$ 、 $D$ 、 $C$ 的缩放指数

我们关注的是非嵌入层参数数量 $N$ ，因为嵌入层的参数数量与词表大小相关，而词表大小是一个独立的设计选择，与模型的计算能力关系不大。

## 2 背景与方法

### 2.1 参数与计算的缩放

对于标准的Transformer语言模型，非嵌入层参数数量 $N$ 可以通过以下公式计算：

$$
N = 2 d_{model} \cdot n_{layer} \cdot (2 d_{attn} + d_{ff})
$$

其中 $d_{model}$ 是模型的隐藏维度， $n_{layer}$ 是层数， $d_{attn}$ 是注意力层的维度， $d_{ff}$ 是前馈层的维度。对于使用交替注意力和前馈层的架构（如我们研究的大多数模型），这简化为：

$$
N \approx 2 n_{layer} d_{model}^2
$$

因为我们通常设置 $d_{ff} = 4 d_{model}$ 和 $d_{attn} = d_{model}$ 。

训练每个token的计算量（以FLOP为单位）大约是每个token的前向传播计算量的6倍：

$$
C_{forward} \approx 2N
$$

$$
C_{total} \approx 6 N S = 6 N D
$$

其中 $S$ 是训练步数， $D$ 是以token数量衡量的总训练数据量（假设每个token只训练一次）。这个因子6来自于：前向传播需要约 $2N$ 次乘加运算，反向传播需要约 $4N$ 次（前向传播的2倍）。

### 2.2 训练过程

我们训练了一系列Transformer语言模型，参数数量从768到超过10亿不等。所有模型使用标准的Transformer架构 [VSP+17]，使用自回归目标（预测下一个token）。

训练细节如下：

- **优化器**：Adam优化器 [KB14]，参数为 $\beta_1 = 0.9$ ， $\beta_2 = 0.95$ ， $\epsilon = 10^{-8}$
- **学习率调度**：学习率从峰值线性衰减到峰值的十分之一，使用2000步的线性预热
- **批大小**：批大小根据模型规模进行了调整，通常在 $0.5M$ 到 $1M$ token之间
- **上下文长度**：所有模型使用 $n_{ctx} = 1024$ 或 $n_{ctx} = 2048$ 的上下文窗口
- **权重初始化**：正态分布初始化，标准差为 $0.02/\sqrt{N}$

我们没有使用dropout，因为我们的主要关注点是研究缩放行为，而dropout会引入额外的超参数。

### 2.3 数据集

我们使用了以下数据集进行训练和评估：

- **WebText2**：OpenAI内部的网页文本数据集，约等于 [RWC+19] 中使用的WebText数据集的扩展版本。包含约 $8M$ 个网页，经过去重和过滤后约有 $40B$ token。
- **Common Crawl**：经过过滤和去重的Common Crawl数据，用于研究更大规模数据集的影响。
- **Books**：Books1和Books2数据集，分别是不同规模的图书语料库。

对于大多数实验，我们使用WebText2数据集。当研究数据集规模的影响时，我们通过对WebText2进行子采样来创建不同规模的数据集。

评估使用了以下数据集：
- **WebText2**：保留的验证集
- **Books**：保留的验证集
- **Wikipedia**：英文维基百科的验证集

我们发现，在不同数据集上的损失遵循相似的缩放规律，尽管绝对值可能有所不同。

## 3 经验结果与基本幂律

在本节中，我们展示语言模型性能的缩放行为。我们发现，交叉熵损失与模型参数数量、数据集规模和训练计算量之间存在精确的幂律关系。

### 3.1 架构独立性

我们首先研究模型架构的具体选择如何影响性能。在固定参数数量的情况下，我们比较了不同架构配置的模型：

- **深度与宽度**：我们训练了具有不同深度（层数）和宽度（隐藏维度）的模型，但保持总参数数量不变。我们发现，只要模型不是特别深或特别浅（即层数在合理范围内），性能差异很小。

- **注意力头数量**：我们比较了使用不同注意力头数量（从1到32）的模型，发现这对性能的影响可以忽略不计，只要总参数数量保持不变。

- **前馈层大小**：我们测试了不同的前馈层与注意力层的维度比例，发现 $d_{ff} = 4 d_{model}$ 是一个合理的选择，但其他比例（如 $d_{ff} = 2 d_{model}$ 或 $d_{ff} = 8 d_{model}$ ）在总参数数量固定时性能相似。

这些结果表明，性能主要由模型的总参数数量决定，而不是架构的具体细节。这一发现非常重要，因为它意味着我们可以专注于研究规模的影响，而不必担心架构选择会混淆结果。

### 3.2 性能与参数数量的关系

我们发现，交叉熵损失 $L$ 与模型参数数量 $N$ 之间存在精确的幂律关系：

$$
L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N}
$$

其中 $N_c \approx 8.8 \times 10^{13}$ （以非嵌入参数计）， $\alpha_N \approx 0.076$ 。

这个关系在超过七个数量级的范围内成立——从768个参数的模型到超过 $10^9$ 个参数的模型。图3.1展示了这种关系，可以看到数据点紧密地沿着幂律拟合线分布。

![图1](/Users/dazhang/PycharmProject/Papers/2-LLM/.picture/2020-Scaling Laws for Neural Language Models-fig1.png)
> **图 1：模型规模的缩放定律。** 交叉熵测试损失与非嵌入层参数数量 $N$ 的关系。对于每个数据集，损失遵循 $L(N) \approx (N_c/N)^{\alpha_N}$ 的幂律（虚线）。拟合线的斜率 $\alpha_N \approx 0.076$ 。数据点跨越超过七个数量级。

这种幂律关系的存在意味着：
1. 模型性能的改进是平滑且可预测的
2. 增加模型规模将持续带来收益，尽管收益递减（因为 $\alpha_N < 1$ ）
3. 我们可以通过在小规模上的实验来预测大规模模型的性能

### 3.3 性能与数据集规模和计算量的关系

类似地，我们发现损失与数据集规模 $D$ 和训练计算量 $C$ 之间也存在幂律关系：

**数据集规模**：

$$
L(D) = \left(\frac{D_c}{D}\right)^{\alpha_D}
$$

其中 $D_c \approx 5.4 \times 10^{13}$ token， $\alpha_D \approx 0.095$ 。

**训练计算量**：

$$
L(C) = \left(\frac{C_c}{C}\right)^{\alpha_C}
$$

其中 $C_c \approx 3.1 \times 10^8$ PF-days， $\alpha_C \approx 0.050$ 。

这些关系如图3.2和图3.3所示。与参数数量的关系类似，这些幂律关系在多个数量级上成立，并且非常精确。

![图2](/Users/dazhang/PycharmProject/Papers/2-LLM/.picture/2020-Scaling Laws for Neural Language Models-fig2.png)
> **图 2：数据集规模的缩放定律。** 交叉熵测试损失与数据集规模 $D$ （以token计）的关系。损失遵循 $L(D) \approx (D_c/D)^{\alpha_D}$ 的幂律。拟合线的斜率 $\alpha_D \approx 0.095$ 。

![图3](/Users/dazhang/PycharmProject/Papers/2-LLM/.picture/2020-Scaling Laws for Neural Language Models-fig3.png)
> **图 3：计算量的缩放定律。** 交叉熵测试损失与训练计算量 $C$ （以FLOP计）的关系。损失遵循 $L(C) \approx (C_c/C)^{\alpha_C}$ 的幂律。拟合线的斜率 $\alpha_C \approx 0.050$ 。

这些结果表明，语言模型的性能可以通过三个独立的幂律关系来描述，每个关系描述了损失如何随一个关键因素变化。

## 4 探索无限数据极限与过拟合

在本节中，我们研究当模型在有限数据上训练时会发生什么。我们发现，过拟合的程度可以被预测，并且存在一个简单的方程来描述损失如何同时依赖于模型规模和数据集规模。

### 4.1 $L(N,D)$ 方程

当数据量有限时，模型可能会过拟合。我们发现，损失可以被建模为两个独立项的和：

$$
L(N,D) = \left[\left(\frac{N_c}{N}\right)^{\alpha_N / \alpha_D} + \frac{D_c}{D}\right]^{\alpha_D}
$$

这个方程的关键特性是：

1. 当 $D \to \infty$ 时，它简化为 $L(N) = (N_c/N)^{\alpha_N}$ ，即数据无限时的幂律关系
2. 当 $N$ 固定时，损失随 $D$ 的增加而减少，但遵循不同的幂律指数
3. 过拟合的程度取决于 $N/D$ 的比例——当模型相对于数据量过大时，过拟合会更严重

这个方程使我们能够预测在给定数据集规模下训练给定规模模型时的期望损失。

### 4.2 结果

我们在不同规模的模型和数据集上验证了 $L(N,D)$ 方程。图4.1展示了这个方程与实验数据的吻合程度。

![图4](/Users/dazhang/PycharmProject/Papers/2-LLM/.picture/2020-Scaling Laws for Neural Language Models-fig4.png)
> **图 4：同时依赖于模型规模和数据集规模的损失。** 当数据有限时，测试损失会偏离无限数据极限的幂律关系（虚线）。实线显示了 $L(N,D)$ 方程的预测，与实验数据高度吻合。

我们的主要发现包括：

1. **过拟合是可预测的**：给定 $N$ 和 $D$ 的值，我们可以准确预测过拟合的程度。

2. **最优数据集规模**：对于给定的模型规模 $N$ ，存在一个最优的数据集规模 $D_{opt}$ ，使得进一步增加数据带来的收益递减。这个最优规模大约满足 $D_{opt} \propto N^{1/\alpha_D}$ 。

3. **临界比例**：当 $N/D$ 超过某个临界值时，模型开始严重过拟合。这个临界比例约为 $N/D \approx 5.4 \times 10^{13} / 8.8 \times 10^{13} \approx 0.61$ token per parameter。

4. **实际意义**：对于实际应用，这意味着我们应该确保数据集规模与模型规模成比例。如果数据有限，训练较小的模型可能更好。

## 5 模型规模与训练时间的缩放定律

在本节中，我们研究在固定计算预算下，如何最优地分配计算资源。我们发现，最优策略是训练更大的模型并提前停止，而非在大量数据上训练较小的模型直到收敛。

### 5.1 对 $B_{crit}$ 的调整

在研究训练时间的影响之前，我们需要考虑批大小的影响。我们发现，存在一个临界批大小 $B_{crit}$ ，使得：

- 当 $B < B_{crit}$ 时，增加批大小可以提高训练效率（减少所需的训练步数）
- 当 $B > B_{crit}$ 时，增加批大小会降低效率（收益递减）

临界批大小与损失有关：

$$
B_{crit} \approx \frac{B^*}{L^{1/\alpha_B}}
$$

其中 $B^*$ 和 $\alpha_B$ 是常数。这意味着当模型性能较好（损失较低）时，可以有效地使用更大的批大小。

在我们的实验中，我们通常使用接近 $B_{crit}$ 的批大小，以确保训练效率。

### 5.2 $L(N, S_{min})$ 的结果

对于固定的模型规模 $N$ ，损失与训练步数 $S$ 之间的关系可以表示为：

$$
L(N, S) = \left[\left(\frac{N_c}{N}\right)^{\alpha_N / \alpha_S} + \frac{S_c}{S}\right]^{\alpha_S}
$$

其中 $S_c$ 是一个常数， $\alpha_S$ 是训练步数的缩放指数。

这个方程的形式与 $L(N,D)$ 方程类似，反映了数据集规模和训练步数之间的对偶性——在固定批大小下， $D = B \cdot S$ 。

图5.1展示了不同规模模型的损失随训练步数的变化。

![图5](/Users/dazhang/PycharmProject/Papers/2-LLM/.picture/2020-Scaling Laws for Neural Language Models-fig5.png)
> **图 5：损失与训练步数的关系。** 对于不同规模的模型，损失随训练步数的增加而减少，但遵循不同的曲线。更大的模型在更少的步数内就能达到相同损失。

关键观察：

1. **更大的模型收敛更快**：在固定计算预算下，更大的模型可以用更少的训练步数达到相同的损失。

2. **提前停止是最优的**：对于给定的计算预算，存在一个最优的训练步数，在此之后继续训练会浪费计算资源。这个最优步数取决于模型规模。

3. **计算分配**：最优策略是训练更大的模型并提前停止，而非训练较小的模型直到收敛。

### 5.3 早期停止的下界

我们发现，对于给定的计算预算 $C$ ，存在一个简单的下界来描述最优的早期停止点：

$$
S_{min} \approx \frac{S^*}{N^{a_S}}
$$

其中 $S^*$ 和 $a_S$ 是常数。这意味着更大的模型应该使用更少的训练步数。

这个结果的实际意义是：
1. 当增加模型规模时，我们应该相应减少训练步数
2. 计算预算的大部分应该分配给更大的模型，而非更多的训练步数
3. 这种策略比"训练较小模型直到收敛"更高效

## 6 计算预算的最优分配

在本节中，我们研究在固定计算预算 $C$ 下，如何最优地分配资源以最小化损失。

### 6.1 最优性能

给定固定的计算预算 $C$ ，我们需要决定：
1. 模型规模 $N$
2. 训练步数 $S$（或等价地，数据集规模 $D = B \cdot S$）

使得 $C = 6 N S$ 固定，且损失 $L(N, S)$ 最小。

通过求解这个优化问题，我们发现最优的模型规模为：

$$
N_{opt} \propto C^{a}
$$

其中 $a \approx 0.73$ 。相应地，最优的训练步数为：

$$
S_{opt} \propto C^{1-a} \approx C^{0.27}
$$

这意味着：
- 当计算预算增加10倍时，模型规模应该增加约 $10^{0.73} \approx 5.4$ 倍
- 训练步数应该增加约 $10^{0.27} \approx 1.9$ 倍
- 大部分计算预算应该分配给更大的模型

### 6.2 预测

基于这些缩放定律，我们可以预测在给定计算预算下的最优性能：

$$
L_{opt}(C) \propto C^{-\alpha_C}
$$

其中 $\alpha_C \approx 0.050$ 。

图6.1展示了这个预测与实验数据的吻合程度。

![图6](/Users/dazhang/PycharmProject/Papers/2-LLM/.picture/2020-Scaling Laws for Neural Language Models-fig6.png)
> **图 6：计算预算的最优分配。** 在固定计算预算下，最优策略是训练更大的模型并提前停止。实线显示了理论预测，虚线显示了实际实验结果。

这些预测使我们能够：
1. 估算达到目标性能所需的计算预算
2. 给定计算预算，确定最优的模型规模和训练步数
3. 规划长期的计算资源需求

### 6.3 与先前工作的矛盾

我们的结果与某些先前的工作存在表面上的矛盾。例如，一些研究 [LOG+19, RSR+19] 发现，增加模型规模带来的收益有限，或者在某些情况下，更大的模型需要更多的训练数据才能避免过拟合。

然而，这些矛盾可以通过以下方式解决：

1. **计算约束 vs. 模型约束**：先前的工作通常在固定训练步数下比较不同规模的模型，而非在固定计算预算下。这导致了不同的结论。

2. **数据规模**：当数据规模不足时，更大的模型确实可能过拟合。我们的 $L(N,D)$ 方程准确地描述了这种情况。

3. **训练策略**：使用最优的训练策略（更大的模型+提前停止）可以避免这些问题，并获得更好的性能。

我们的结果表明，在正确的训练策略下，增加模型规模将持续带来收益，尽管收益递减（因为 $\alpha_C < 1$ ）。

## 7 相关工作

我们的工作建立在先前对神经网络缩放行为研究的基础上。

**语言模型缩放**：近年来，语言模型的规模持续增长。GPT-2 [RWC+19] 使用了15亿参数，T5 [RSR+19] 使用了110亿参数，Megatron-LM [SPP+19] 使用了83亿参数。这些工作表明，更大的模型通常性能更好，但缺乏系统性的缩放定律研究。

**神经网络缩放定律**：先前的工作已经研究了神经网络在不同任务上的缩放行为。例如，[HNA+17] 研究了图像分类任务上的缩放定律，[MG19] 研究了机器翻译任务上的缩放定律。这些工作发现了类似的幂律关系，但主要关注特定任务。

**计算最优训练**：一些工作研究了如何最优地分配计算资源。例如，[RRBS19] 研究了在固定计算预算下如何选择模型规模和训练步数。我们的工作扩展了这些研究，并提供了更系统性的分析。

**过拟合与泛化**：过拟合是机器学习中的一个核心问题。我们的工作提供了预测过拟合程度的方程，并展示了如何通过调整模型规模和数据集规模来控制过拟合。

## 8 讨论

我们的研究提供了语言模型缩放行为的系统性分析。主要发现包括：

1. **幂律关系**：语言模型的性能与模型规模、数据集规模和训练计算量之间存在精确的幂律关系。这些关系在超过七个数量级上成立。

2. **架构无关性**：在固定参数数量的情况下，模型架构的具体细节（如深度、宽度、注意力头数量）对性能的影响很小。

3. **最优计算分配**：在固定计算预算下，最优策略是训练更大的模型并提前停止。模型规模应随计算预算的快速增长而增长（约 $N \propto C^{0.73}$ ）。

4. **过拟合可预测性**：过拟合的程度可以通过 $L(N,D)$ 方程准确预测。

这些发现具有重要的实际意义：

- **资源规划**：我们可以使用缩放定律来估算达到目标性能所需的计算资源
- **训练策略**：最优的训练策略是使用更大的模型并提前停止
- **架构选择**：架构细节相对不重要，可以专注于规模

然而，我们的研究也有一些局限性：

1. **数据依赖性**：我们的结果主要基于特定的数据集（WebText2），可能不完全适用于其他领域或语言
2. **任务依赖性**：我们主要研究了语言建模任务（下一个token预测），其他任务的缩放行为可能不同
3. **效率考虑**：我们没有考虑推理效率，而在实际部署中这可能是一个重要约束

未来的工作方向包括：

- 研究不同任务上的缩放定律
- 探索更高效的训练策略
- 理解缩放定律的根本原因
- 研究推理效率与模型规模的关系

## 附录

### 附录 A：缩放定律的推导

在本附录中，我们提供缩放定律的理论推导。我们假设语言模型的性能可以通过一个简单的函数形式来描述：

$$
L(N, D) = f(N) + g(D) + h(N, D)
$$

其中 $f(N)$ 表示模型规模的影响， $g(D)$ 表示数据规模的影响， $h(N, D)$ 表示交互项。

通过实验，我们发现这些函数具有幂律形式：

$$
f(N) = \left(\frac{N_c}{N}\right)^{\alpha_N}, \quad g(D) = \left(\frac{D_c}{D}\right)^{\alpha_D}
$$

交互项 $h(N, D)$ 相对较小，可以忽略。

### 附录 B：实验细节

**模型架构**：所有模型使用标准的Transformer架构 [VSP+17]。具体配置如表B.1所示。

| 参数数量 | 层数 | 隐藏维度 | 注意力头数 | 前馈维度 |
|---|---|---|---|---|
| 768 | 1 | 64 | 1 | 256 |
| 3,072 | 2 | 128 | 2 | 512 |
| 12,288 | 4 | 256 | 4 | 1,024 |
| 49,152 | 6 | 512 | 8 | 2,048 |
| 196,608 | 8 | 1,024 | 16 | 4,096 |
| 786,432 | 12 | 1,536 | 16 | 6,144 |
| 3,145,728 | 16 | 2,048 | 32 | 8,192 |
| 12,582,912 | 24 | 3,072 | 32 | 12,288 |
| 50,331,648 | 32 | 4,096 | 64 | 16,384 |
| 201,326,592 | 40 | 6,144 | 64 | 24,576 |
| 805,306,368 | 48 | 8,192 | 96 | 32,768 |
| 1,073,741,824 | 64 | 10,240 | 128 | 40,960 |

> **表 B.1：模型架构配置。** 所有模型使用标准Transformer架构， $d_{ff} = 4 d_{model}$ 。

**训练细节**：
- 优化器：Adam [KB14]， $\beta_1 = 0.9$ ， $\beta_2 = 0.95$ ， $\epsilon = 10^{-8}$
- 学习率：线性预热2000步，然后线性衰减到峰值的十分之一
- 峰值学习率：根据模型规模调整，范围从 $10^{-4}$ 到 $6 \times 10^{-4}$
- 批大小：根据模型规模调整，范围从 $0.5M$ 到 $1M$ token
- 权重初始化：正态分布，标准差为 $0.02/\sqrt{N}$

**计算资源**：所有实验在 V100 GPU 上进行。训练一个10亿参数的模型约需要256个GPU训练约1周。

### 附录 C：补充图表

图C.1至图C.5展示了更多关于缩放行为的实验结果。

![图C.1](/Users/dazhang/PycharmProject/Papers/2-LLM/.picture/2020-Scaling Laws for Neural Language Models-fig8.png)
> **图 C.1：不同数据集上的缩放行为。** 缩放定律在不同数据集上成立，尽管绝对损失可能不同。

![图C.2](/Users/dazhang/PycharmProject/Papers/2-LLM/.picture/2020-Scaling Laws for Neural Language Models-fig16.png)
> **图 C.2：训练损失与测试损失的比较。** 训练损失和测试损失遵循相似的缩放定律，但训练损失略低。

![图C.3](/Users/dazhang/PycharmProject/Papers/2-LLM/.picture/2020-Scaling Laws for Neural Language Models-fig11.png)
> **图 C.3：不同训练步数的缩放行为。** 损失随训练步数的增加而减少，但遵循幂律关系。

![图C.4](/Users/dazhang/PycharmProject/Papers/2-LLM/.picture/2020-Scaling Laws for Neural Language Models-fig14.png)
> **图 C.4：计算预算的最优分配。** 实验结果与理论预测高度吻合。

![图C.5](/Users/dazhang/PycharmProject/Papers/2-LLM/.picture/2020-Scaling Laws for Neural Language Models-fig9.png)
> **图 C.5：过拟合程度的预测。** $L(N,D)$ 方程准确预测了过拟合的程度。

### 附录 D：理论分析

在本附录中，我们提供缩放定律的理论解释。我们假设语言模型的性能受限于其能够学习的模式数量。

**模型容量**：模型的容量（即它可以表示的函数数量）大致与其参数数量 $N$ 成正比。因此，损失应该随 $N$ 的增加而减少。

**数据覆盖**：数据集的覆盖范围（即它包含的模式数量）大致与其规模 $D$ 成正比。因此，损失应该随 $D$ 的增加而减少。

**计算预算**：训练计算量 $C$ 限制了模型可以从数据中学习的模式数量。因此，损失应该随 $C$ 的增加而减少。

这些观察导致了幂律关系：

$$
L \propto N^{-\alpha_N} D^{-\alpha_D} C^{-\alpha_C}
$$

然而，这些因素不是完全独立的，存在交互效应。我们的 $L(N,D)$ 方程和 $L(N,S)$ 方程正是这些交互效应的简化描述。

## 参考文献

[1] R. Child, S. Gray, A. Radford, and I. Sutskever. Generating long sequences with sparse transformers. *arXiv preprint arXiv:1904.10509*, 2019.

[2] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. *arXiv preprint arXiv:1810.04805*, 2018.

[3] D. Hendrycks and K. Gimpel. Gaussian error linear units (GELUs). *arXiv preprint arXiv:1606.08415*, 2016.

[4] J. Hoffmann, S. Borgeaud, A. Mensch, E. Buchatskaya, T. Cai, E. Rutherford, D. d. L. Casas, L. A. Hendricks, J. Welbl, A. Clark, et al. Training compute-optimal large language models. *arXiv preprint arXiv:2203.15556*, 2022.

[5] Y. Huang, Y. Cheng, A. Bapna, O. Firat, D. Chen, M. Chen, H. Lee, J. Ngiam, Q. V. Le, Y. Wu, et al. GPipe: Efficient training of giant neural networks using pipeline parallelism. *Advances in Neural Information Processing Systems*, 32, 2019.

[6] D. P. Kingma and J. Ba. Adam: A method for stochastic optimization. *arXiv preprint arXiv:1412.6980*, 2014.

[7] N. Kitaev, Ł. Kaiser, and A. Levskaya. Reformer: The efficient transformer. *arXiv preprint arXiv:2001.04451*, 2020.

[8] T. B. Brown, B. Mann, N. Ryder, M. Subbiah, J. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell, et al. Language models are few-shot learners. *Advances in Neural Information Processing Systems*, 33:1877–1901, 2020.

[9] Y. Liu, M. Ott, N. Goyal, J. Du, M. Joshi, D. Chen, O. Levy, M. Lewis, L. Zettlemoyer, and V. Stoyanov. RoBERTa: A robustly optimized BERT pretraining approach. *arXiv preprint arXiv:1907.11692*, 2019.

[10] S. McCandlish, J. Kaplan, D. Amodei, and the D. Amodei. An empirical model of large-batch training. *arXiv preprint arXiv:1812.06162*, 2018.

[11] P. Micikevicius, S. Narang, J. Alben, G. Diamos, E. Elsen, D. Garcia, B. Ginsburg, M. Houston, O. Kuchaiev, G. Venkatesh, et al. Mixed precision training. *arXiv preprint arXiv:1710.03740*, 2017.

[12] A. Radford, K. Narasimhan, T. Salimans, and I. Sutskever. Improving language understanding by generative pre-training. *OpenAI Blog*, 2018.

[13] A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, and I. Sutskever. Language models are unsupervised multitask learners. *OpenAI Blog*, 2019.

[14] C. Raffel, N. Shazeer, A. Roberts, K. Lee, S. Narang, M. Matena, Y. Zhou, W. Li, and P. J. Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. *arXiv preprint arXiv:1910.10683*, 2019.

[15] S. Rajbhandari, J. Rasley, O. Ruwase, and Y. He. ZeRO: Memory optimizations toward training trillion parameter models. *arXiv preprint arXiv:1910.02054*, 2019.

[16] M. Shoeybi, M. Patwary, R. Puri, P. LeGresley, J. Casper, and B. Catanzaro. Megatron-LM: Training multi-billion parameter language models using model parallelism. *arXiv preprint arXiv:1909.08053*, 2019.

[17] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin. Attention is all you need. *Advances in Neural Information Processing Systems*, 30:5998–6008, 2017.

[18] A. Wang, A. Singh, J. Michael, F. Hill, O. Levy, and S. R. Bowman. GLUE: A multi-task benchmark and analysis platform for natural language understanding. *arXiv preprint arXiv:1804.07461*, 2018.

[19] T. Wolf, L. Debut, V. Sanh, J. Chaumond, C. Delangue, A. Moi, P. Cistac, T. Rault, R. Louf, M. Funtowicz, et al. HuggingFace's transformers: State-of-the-art natural language processing. *arXiv preprint arXiv:1910.03771*, 2019.

[20] Z. Yang, Z. Dai, Y. Yang, J. Carbonell, R. Salakhutdinov, and Q. V. Le. XLNet: Generalized autoregressive pretraining for language understanding. *Advances in Neural Information Processing Systems*, 32:5753–5763, 2019.
