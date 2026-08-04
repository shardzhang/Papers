# Accurate Methods for the Statistics of Surprise and Coincidence——惊喜与巧合统计的精确方法

> Ted Dunning | 新墨西哥州立大学

本文探讨了文本统计分析中常用方法的适用性问题，指出基于正态性假设的统计检验（如 $\chi^2$ 检验和 $z$ -score检验）在分析稀有事件时存在严重缺陷，会极大高估其显著性。文章提出并详细推导了基于似然比的检验方法，该方法在二项分布和多项分布假设下具有更好的渐近行为，适用于小样本和稀有事件的分析。通过对金融文本的二元组（bigram）分析实验表明，似然比方法能够直观、准确地识别显著共现的词对，而传统的 $\chi^2$ 检验则在稀有事件上产生严重失真。

关键发现：

- 稀有事件在自然语言文本中占比极高（约20–30%的词汇出现频率低于五万分之一），基于正态近似的传统统计方法在此场景下完全失效
- 当 $np(1-p) < 5$ 时，正态分布对二项分布的近似误差急剧增大；当 $np = 0.01$ 时，正态近似高估显著性可达 $4 \times 10^{20}$ 倍
- 似然比检验统计量 $-2 \log \lambda$ 渐近服从 $\chi^2$ 分布，且收敛速度在二项和多项分布情况下非常快，即使小样本也能得到准确结果
- 在 31,777 词的金融文本二元组分析实验中，似然比方法排名的前 50 个二元组全部符合直觉认知；而 $\chi^2$ 检验的排名几乎全部被仅出现一次的稀有二元组主导

---

## 摘要

在文本统计分析领域已有大量研究工作。然而，文献中部分工作使用了不恰当的统计方法，且未对结果的统计显著性予以充分关注。特别地，渐近正态性假设经常被不当使用，导致结果存在缺陷。

正态分布假设限制了分析稀有事件的能力。然而，稀有事件恰恰构成了真实文本的很大一部分。

幸运的是，基于似然比检验的、更具适用性的方法能够在相对较小的样本上取得良好效果。这些检验可以高效实现，并已被用于复合词项的检测和领域特定词项的确定。在某些情况下，这些方法的性能远超先前使用的方法。而在传统列联表方法表现良好的情形下，本文描述的似然比检验结果与之几乎一致。

本文描述了一种基于似然比的度量基础，可应用于文本分析。

---

## 1 引言

近年来，文本统计分析重新成为研究趋势。这一趋势促使大量研究者在信息检索和自然语言处理领域做出了优秀的工作。不幸的是，其中许多工作对其结果所引发的统计问题采取了草率的态度。

这些研究者所采取的方法大致可分为三类：

1. **收集海量文本**，以使简单直接的基于统计的度量能够良好工作。
2. **对相对小规模的文本进行简单化的统计分析**，然后要么"凭经验修正"误差，要么完全忽略该问题。
3. **完全不进行任何统计分析**。

第一种方法以 IBM 小组在统计机器翻译方面的研究为代表（Brown et al. [2]）。他们从内部备忘录、技术手册和言情小说等多种来源收集了近十亿词的英文文本，并对加拿大议会记录（Hansards）中大部分电子化的内容进行了对齐。这项工作规模浩大，也获得了丰硕的有趣成果。其大部分工作的统计显著性无可指摘，但所需的文本量在许多场景下完全不现实。

第二种方法以 Gale 和 Church 的大部分工作为代表（Gale and Church [5],[6]；Church et al. [3]）。他们工作的许多结果是完全可用的，其度量方法在论文给出的示例中表现良好。但总体而言，这些方法存在问题。例如，直接基于计数的互信息估计在涉及小计数值时会被高估，而 $z$ -score 会显著高估稀有事件的显著性。

第三种方法几乎代表了所有信息检索文献。即便是最近非常创新的工作，如使用 latent semantic indexing（Dumais et al. [4]）和 Pathfinder 网络（Schvaneveldt [9]）的研究，也未涉及其内部处理的统计可靠性。不过，他们在分析方法的整体有效性时使用了良好的统计方法。

即使是像文本检索中的逆文档频率加权（Salton and McGill [9]）这样广为接受的技术，通常也只是建立在非常粗略的依据之上。

本文的目标是提出一种基于统计考量、实用且可在多种场景下使用的度量方法。该方法在大样本和小样本文本上都能表现合理，并允许直接比较稀有现象和常见现象的显著性。这种比较之所以可能，是因为本文描述的度量比传统度量具有更好的渐近行为。

在后续内容中，部分章节主要由背景材料或数学细节构成，熟悉统计学的读者或时间紧迫的读者可以跳过。不应跳过的章节标有 **，包含大量背景材料的章节标有 *，详细的推导则不加标记。这种"精华部分"约定应能使本文对只想浏览的实现者或读者更加有用。

## 2 正态性假设 *

对采样随机变量的简单函数呈正态分布或近似正态分布的假设，是许多常见统计检验的基础。这尤其包括 Pearson $\chi^2$ 检验和 $z$ -score 检验。这一假设在许多情况下是完全有效的。由于涉及的方法得以简化，即使在边缘情况下该假设也完全合理。

然而，在比较稀有事件的发生率时，这些检验所依赖的假设会失效，因为文本主要由这类稀有事件构成。例如，对中等规模语料库的简单词频统计显示，频率低于五万分之一（即每五万词中出现少于一次）的词约占典型英文新闻电讯文本的 20–30%。这个"稀有的"四分之一英文词汇包含了大量实义词和几乎全部的技术术语。以下是从五十万词的 Reuters 报道样本中随机选取的约 0.2% 的词，这些词至少出现了一次，但次数少于五次：

abandonment, aerobics, alternating, altitude, amateur, appearance, assertion, barrack, biased, bookies, broadcaster, cadres, charging, clause, collating, compile, confirming, contemptuously, corridors, crushed, deadly, demented, detailing, landscape, seldom, directorship, lobbyists, sheet, dispatched, malfeasances, simplified, dogfight, meat, snort, duds, miners, specify, eluded, monsoon, staffing, enigmatic, napalm, substitute, euphemism, northeast, surreptitious, experiences, oppressive, tall, fares, overburdened, terraced, finals, parakeets, tipping, foiling, penetrate, transform, gangsters, poi, turbid, guide, praised, understatement, headache, prised, unprofitable, hobbled, protector, vagaries, identities, query, villas, inappropriate, redoubtable, watchful, inflamed, remark, winter, instilling, resignations, intruded, ruin, unction, scant

这份列表中唯一不算生僻的词是 *poi*（一种用芋头制作的夏威夷传统菜肴）。如果我们只采样五万词而非上述列表所用的五十万词，则该列表中任何词的期望出现次数都将少于 0.1——远低于常规检验应使用的阈值。

如果这些普通词汇都属于"稀有"，那么任何基于文本的统计工作都必须处理稀有事件的现实。有趣的是，虽然连续文本中的大多数词是常见词，但总词汇表中的大多数词是稀有词。

不幸的是，计算语言学中最常用的统计分析方法的基础假设是，被分析的事件相对常见。对于前述 Reuters 语料库中五万词的样本，上表中的任何一个词都不够常见，无法期望这些分析能良好工作。

## 3 卡方检验的传统 *

在文本分析中，基于统计的度量通常依赖于某些具有已知分布的检验统计量。这种分布最常见的是正态分布或 $\chi^2$ 分布。这些度量非常有用，可以在多种不同场景下准确评估显著性。然而，它们基于的若干假设对大多数文本分析并不成立。

这些度量背后的假设为何及如何不成立的细节主要对统计学家有价值，但结果本身对统计使用者（在本例中，是对词项计数感兴趣的人）很有意义。更具适用性的技术在文本分析中至关重要。下一节将描述这样一种技术；该技术的实现在后续章节中描述。

（图1 正态分布和二项分布）

## 4 文本分析的二项分布 **

当待分析数据来自对重复、相同且独立的实验中的正向结果进行计数时，二项分布常在统计分析中出现。抛硬币是这类实验的典型例子。

词项计数任务可以转化为一系列此类二元试验的重复序列，将文本中的每个词与目标词进行比较。这些比较可视为类似于抛硬币的二元实验序列。在文本中，每次比较显然并非完全独立，但依赖性随距离迅速减弱。另一个在实践中相对有效的假设是，出现特定词的概率是恒定的。当然，这并非完全正确，因为主题变化可能导致频率变化。实际上，正是这一假设的适度失效使得浅层信息检索技术成为可能。

在这些独立性和平稳性假设有效的范围内，我们可以转向关于伯努利试验的抽象讨论（而非文本中的词），并利用一系列标准结论。伯努利试验是对抛硬币的统计理想化，其中每次试验的成功概率固定不变。

特别地，如果下一个词匹配某个原型的实际概率为 $p$ ，则在接下来 $n$ 个词中产生的匹配次数是一个具有二项分布的随机变量 $K$ ：

$$
P(K = k) = \binom{n}{k} p^k (1-p)^{n-k}
$$

其均值为 $np$ ，方差为 $np(1-p)$ 。如果 $np(1-p) > 5$ ，则该变量的分布近似正态；随着 $np(1-p)$ 超过该阈值，分布越来越接近正态分布。这可以在上方的图 1 中看到，其中二项分布（虚线）与近似正态分布（实线）一起绘制， $np$ 分别设为 5、10 和 20， $n$ 固定为 100。 $n$ 更大而 $np$ 保持不变时，曲线与所示曲线无明显差异。在这些情况下， $np \approx np(1-p)$ 。

二项分布与正态分布之间的这种一致正是基于正态性假设的检验统计量在基于计数的实验分析中如此有用的原因。对于二项分布，通常认为当 $np(1-p) > 5$ 时正态性假设足够成立。

当 $np(1-p)$ 小于 5 时情况不同，当 $np(1-p)$ 小于 1 时则显著不同。首先，用连续分布（如正态分布）近似离散分布（如二项分布）的意义要小得多。其次，使用正态近似计算的概率越来越不准确。

表 1 显示了使用二项分布和正态分布分别在 $np = 0.001$ 、 $np = 0.01$ 、 $np = 0.1$ 和 $np = 1$ （其中 $n = 100$ ）时，在 100 词文本中找到一次或多次匹配的概率。大多数词足够稀有，即使对于 $n$ 高达几千的文本样本， $np$ 也处于这一范围的底部。而短短语的数量如此之多，以至于即使 $n$ 高达几百万，几乎所有短语的 $np \ll 1$ 。

表 1 显示，对于稀有事件，正态分布甚至无法近似二项分布。事实上，当 $np = 0.1$ 且 $n = 100$ 时，使用正态分布将一次或多次出现的显著性高估了约 40 倍；而当 $np = 0.01$ 时，使用正态分布将显著性高估了约 $4 \times 10^{20}$ 倍。当 $n$ 增加到超过 100 时，表中的数字没有显著变化。

如果这种高估是恒定的，那么使用正态分布的估计仍可校正并继续使用，但误差并非恒定的事实意味着，依赖正态近似的方法不应用于分析正向结果概率非常小的伯努利试验。然而，在许多真实的文本分析中，比较 $np = 0.001$ 和 $np > 1$ 的情况是一个常见问题。

**表 1：正态近似引入的误差**

| $np$ | $P(k > 1)$ 使用二项分布 | 使用正态分布的估计 |
|---|---|---|
| 0.001 | 0.000099 | $0.34 \times 10^{-217}$ |
| 0.01 | 0.0099 | $0.29 \times 10^{-22}$ |
| 0.1 | 0.095 | 0.0022 |
| 1 | 0.63 | 0.5 |

## 5 似然比检验 *

另一类检验方法并不如此严重地依赖正态性假设，而是使用广义似然比的渐近分布。对于文本分析和类似问题，使用似然比可以大幅改善统计结果。这种改善的实际效果是，基于似然比的统计文本分析可以在比传统基于正态分布假设的检验小得多的文本量上有效进行，并允许比较稀有现象和常见现象的显著性。

### 5.1 参数空间与似然函数

似然比检验基于这样一种思想：统计假设可以被视为指定了所用统计模型未知参数所构成空间的子空间。这些检验假定模型已知，但模型参数未知。这类检验称为参数检验。也有完全不假设底层模型的其他检验可用，它们称为无分布检验。本文仅描述一种特定的参数检验。有关参数检验和无分布检验的更多信息，可参考 Bradley [1] 以及 Mood, Graybill 和 Boes [8]。

对于由参数 $p_1, p_2, \ldots$ 描述的给定模型，观察到由 $k_1, \ldots, k_n$ 描述的特定实验结果的概率，称为该模型的似然函数，记为：

$$
H(p_1, p_2, \ldots; k_1, \ldots, k_m)
$$

其中 $H$ 的所有参数中，分号左侧为模型参数，右侧为观测值。在连续情况下，概率由概率密度替代。对于二项分布和多项分布，我们只处理离散情况。

对于重复伯努利试验， $m = 2$ ，因为我们同时观察试验次数和正向结果次数，并且只有一个 $p$ 。似然函数的显式形式为：

$$
H(p; n, k) = \binom{n}{k} p^k (1-p)^{n-k}
$$

参数空间是 $p$ 的所有可能值的集合，而假设 $p = p_0$ 是其中的一个单点。为简洁起见，模型参数可以汇集为单个参数，观测值亦然。因此似然函数写为：

$$
H(\boldsymbol{\omega}; \mathbf{k})
$$

其中 $\boldsymbol{\omega}$ 被视为参数空间 $\Omega$ 中的一个点， $\mathbf{k}$ 为观测空间 $\mathcal{K}$ 中的一个点。特定的假设或观测分别通过对 $\Omega$ 或 $\mathcal{K}$ 加下标来表示。

更多关于似然比检验的信息可参考理论统计学教材（Mood et al. [8]）。

### 5.2 似然比

假设的似然比是似然函数在假设所表示的子空间上的最大值与在整个参数空间上的最大值之比，即：

$$
\lambda = \frac{\max_{\boldsymbol{\omega} \in \Omega_0} H(\boldsymbol{\omega}; \mathbf{k})}{\max_{\boldsymbol{\omega} \in \Omega} H(\boldsymbol{\omega}; \mathbf{k})}
$$

其中 $\Omega$ 是整个参数空间， $\Omega_0$ 是被检验的特定假设。

似然比的一个特别重要的性质是，统计量 $-2 \log \lambda$ 渐近服从 $\chi^2$ 分布，自由度等于 $\Omega$ 与 $\Omega_0$ 的维数之差。重要的是，在二项分布和多项分布情况下，该渐近收敛非常迅速。

### 5.3 二项分布和多项分布的似然比

使用似然比可以相当容易地比较两个二项或多项过程。对于两个二项分布的情况：

$$
H(p_1, p_2; k_1, n_1, k_2, n_2) = \binom{n_1}{k_1} p_1^{k_1} (1-p_1)^{n_1-k_1} \binom{n_2}{k_2} p_2^{k_2} (1-p_2)^{n_2-k_2}
$$

两个分布具有相同底层参数的假设由集合 $\{(p_1, p_2) \mid p_1 = p_2\}$ 表示。

该检验的似然比为：

$$
\lambda = \frac{\max_p H(p, p; k_1, n_1, k_2, n_2)}{\max_{p_1, p_2} H(p_1, p_2; k_1, n_1, k_2, n_2)}
$$

这些最大值在分母中由 $p_1 = \frac{k_1}{n_1}$ 和 $p_2 = \frac{k_2}{n_2}$ 实现，在分子中由 $p = \frac{k_1 + k_2}{n_1 + n_2}$ 实现。这使比率简化为：

$$
\lambda = \frac{\max_p L(p, k_1, n_1) L(p, k_2, n_2)}{\max_{p_1, p_2} L(p_1, k_1, n_1) L(p_2, k_2, n_2)}
$$

其中

$$
L(p, k, n) = p^k (1-p)^{n-k}
$$

对似然比取对数得：

$$
-2 \log \lambda = 2 [\log L(p_1, k_1, n_1) + \log L(p_2, k_2, n_2) - \log L(p, k_1, n_1) - \log L(p, k_2, n_2)]
$$

对于多项分布情况，使用双重下标和缩写符号更为便利。记：

$$
\mathbf{P}_i = p_{1i}, p_{2i}, \ldots, p_{ji}, \ldots
$$
$$
\mathbf{K}_i = k_{1i}, k_{2i}, \ldots, k_{ji}, \ldots
$$
$$
\mathbf{Q} = q_1, q_2, \ldots, q_j, \ldots
$$

于是可以写出：

$$
H(\mathbf{P}_1, \mathbf{P}_2; \mathbf{K}_1, n_1, \mathbf{K}_2, n_2) = \prod_{i=1,2} \frac{n_i!}{\prod_j k_{ji}!} \prod_j p_{ji}^{k_{ji}}
$$

似然比为：

$$
\lambda = \frac{\max_{\mathbf{Q}} H(\mathbf{Q}, \mathbf{Q}; \mathbf{K}_1, n_1, \mathbf{K}_2, n_2)}{\max_{\mathbf{P}_1, \mathbf{P}_2} H(\mathbf{P}_1, \mathbf{P}_2; \mathbf{K}_1, n_1, \mathbf{K}_2, n_2)}
$$

这可以类似二项分布情况进行分解，使用函数：

$$
L(\mathbf{P}, \mathbf{K}) = \prod_j p_j^{k_j}
$$

得到：

$$
\lambda = \frac{\max_{\mathbf{Q}} L(\mathbf{Q}, \mathbf{K}_1) L(\mathbf{Q}, \mathbf{K}_2)}{\max_{\mathbf{P}_1, \mathbf{P}_2} L(\mathbf{P}_1, \mathbf{K}_1) L(\mathbf{P}_2, \mathbf{K}_2)}
$$

该表达式隐含地包含 $n$ ，因为 $\sum_j k_j = n$ 。

最大化并取对数得：

$$
-2 \log \lambda = 2 [\log L(\mathbf{P}_1, \mathbf{K}_1) + \log L(\mathbf{P}_2, \mathbf{K}_2) - \log L(\mathbf{Q}, \mathbf{K}_1) - \log L(\mathbf{Q}, \mathbf{K}_2)]
$$

其中

$$
p_{ji} = \frac{k_{ji}}{\sum_j k_{ji}}
$$
$$
q_j = \frac{\sum_i k_{ji}}{\sum_{i,j} k_{ji}}
$$

如果零假设成立，则对数似然比渐近服从 $\chi^2$ 分布，自由度为 $k/2 - 1$ 。当 $j = 2$ （二项分布）时， $-2 \log \lambda$ 服从自由度为 1 的 $\chi^2$ 分布。

如果我们最初用均值为 $np$ 、方差为 $np(1-p)$ 的正态分布来近似二项分布，那么我们会得到另一个形式，当 $np(1-p)$ 大致大于 5 时，该形式是 $-2 \log \lambda$ 的良好近似：

$$
-2 \log \lambda \approx \sum_{i,j} \frac{(k_{ji} - n_i q_j)^2}{n_i q_j (1 - q_j)}
$$

其中 $q_j = \frac{\sum_i k_{ji}}{\sum_{i,j} k_{ji}}$ （与上述多项分布情况相同），且 $n_i = \sum_j k_{ji}$ 。

有趣的是，该表达式恰好是 Pearson $\chi^2$ 检验的统计量，尽管所示形式并非惯常写法。图 2 显示了该表达式与之前导出的精确二项对数似然比在 $p = 0.1$ 、 $n_1 = n_2 = 1000$ 且不同 $k_1$ 和 $k_2$ 取值下的良好一致性。

另一方面，图 3 显示了当 $p = 0.01$ 、 $n_1 = 100$ 、 $n_2 = 10000$ 时 Pearson 统计量与对数似然比之间的分歧。注意纵坐标尺度的巨大变化。当 $k_1$ 大于基于 $k_2$ 观测值的期望值时，出现显著差异。 $n_1 < n_2$ 且 $\frac{k_1}{n_1} > \frac{k_2}{n_2}$ 的情况正是许多文本分析中最受关注的情形。

对数似然比收敛于渐近分布的过程在图 4 中得到戏剧性的展示。图中较平滑的线是使用符号代数包计算的，代表理想的、自由度为 1 的累积 $\chi^2$ 分布。较粗糙的曲线是通过数值实验计算的，设 $p = 0.01$ 、 $n_1 = 100$ 、 $n_2 = 10000$ ，对应于图 3 中的情况。两者之间的紧密吻合表明，即使在正态 $\chi^2$ 度量与理想值严重偏离的范围内，似然比度量在六个数量级的显著性水平上仍能产生准确结果。

（图 2：对数似然比与 Pearson $\chi^2$ 的比较）
（图 3：对数似然比与 Pearson $\chi^2$ 的比较）
（图 4：理想与模拟的对数似然比）

## 6 实际结果

### 6.1 小文本的二元组分析

为检验似然方法的效果，我们对从瑞士联合银行（Union Bank of Switzerland）获取的 30,000 词文本样本进行了分析，旨在找出那些彼此相邻出现且频率显著高于基于词频预期的词对。该文本为 31,777 词的金融文本，主要描述 1986 和 1987 年的市场状况。

这种二元组分析的结果应能突出英语中的常见搭配以及所分析文本的金融领域特有搭配。正如我们将看到的，基于似然比检验的排序确实做到了这一点。对通用文本的大规模语料库与特定领域文本进行类似比较，可用于生成仅包含该领域文本特有词项和二元组的列表。

该比较通过创建一个列联表进行，表中包含文本中每个出现的二元组的以下计数：

| $k(AB)$ | $k(\neg A B)$ |
|---|---|
| $k(A \neg B)$ | $k(\neg A \neg B)$ |

其中 $\neg A B$ 表示第一个词不是 $A$ 而第二个词是 $B$ 的二元组。

如果词 $A$ 和 $B$ 独立出现，则我们期望 $p(AB) = p(A) p(B)$ ，其中 $p(AB)$ 是 $A$ 和 $B$ 按顺序出现的概率， $p(A)$ 是 $A$ 出现在第一个位置的概率， $p(B)$ 是 $B$ 出现在第二个位置的概率。我们可以通过将 $A$ 和 $B$ 独立的零假设表述为 $p(A \mid B) = p(A \mid \neg B) = p(A)$ ，将其转化为之前二项分析的框架。这意味着检验 $A$ 和 $B$ 的独立性可以通过检验在存在 $B$ 的条件下 $A$ 的分布（表的第一行）与不存在 $B$ 的条件下 $A$ 的分布（表的第二行）是否相同来实现。当然，实际上我们并非真正进行统计检验来验证 $A$ 和 $B$ 是否独立——我们知道它们在文本中通常不独立。相反，我们只是希望将检验统计量作为一种度量，帮助突出文本中高度关联的特定 $A$ 和 $B$ 。

这些计数使用前述二项检验进行分析，前 50 个最显著的二元组列于表 2。该表包含最显著的 200 个二元组，并按第一列（包含 $-2 \log \lambda$ 的值）逆序排列。其他列包含上述列联表的四个计数以及二元组本身。

检查该表可以发现，它与我们对表中二元组自然程度的直觉判断有很好的相关性。这与表 3 形成鲜明对比，表 3 包含相同的数据，但第一列使用 Pearson $\chi^2$ 检验统计量计算。对仅出现几次的词项显著性的高估是惊人的。事实上，整个表的前半部分完全被那些稀有到在当前文本样本中仅出现一次的二元组主导。请注意，原文中二元组 "sees posibilities" 的拼写错误保留不变。

在分析的 2693 个二元组中，有 2682 个超出了正态 $\chi^2$ 检验的适用范围。适合使用 $\chi^2$ 检验分析的 11 个二元组列于表 4。值得注意的是，所有这些二元组都包含英语中最常见的词 `the`。

**表 2：按对数似然检验排序的二元组**

| $-2 \log \lambda$ | $k(AB)$ | $k(A \neg B)$ | $k(\neg A B)$ | $k(\neg A \neg B)$ | $A$ | $B$ |
|---|---|---|---|---|---|---|
| 270.72 | 110 | 2442 | 111 | 29114 | the | swiss |
| 263.90 | 29 | 13 | 123 | 31612 | can | be |
| 256.84 | 31 | 23 | 139 | 31584 | previous | year |
| 167.23 | 10 | 0 | 3 | 31764 | mineral | water |
| 157.21 | 76 | 104 | 2476 | 29121 | at | the |
| 157.03 | 16 | 16 | 51 | 31694 | real | terms |
| 146.80 | 9 | 0 | 5 | 31763 | natural | gas |
| 115.02 | 16 | 0 | 865 | 30896 | owing | to |
| 104.53 | 10 | 9 | 41 | 31717 | health | insurance |
| 100.96 | 8 | 2 | 27 | 31740 | stiff | competition |
| 98.72 | 12 | 111 | 14 | 31640 | is | likely |
| 95.29 | 8 | 5 | 24 | 31740 | qualified | personnel |
| 94.50 | 10 | 93 | 6 | 31668 | an | estimated |
| 91.40 | 12 | 111 | 21 | 31633 | is | expected |
| 81.55 | 10 | 45 | 35 | 31687 | 1 | 2 |
| 76.30 | 5 | 13 | 0 | 31759 | balance | sheet |
| 73.35 | 16 | 2536 | 1 | 29224 | the | united |
| 68.96 | 6 | 2 | 45 | 31724 | accident | insurance |
| 68.61 | 24 | 43 | 1316 | 30394 | terms | of |
| 61.61 | 3 | 0 | 0 | 31774 | natel | c |
| 60.77 | 6 | 92 | 2 | 31677 | will | probably |
| 57.44 | 4 | 11 | 1 | 31761 | great | deal |
| 57.44 | 4 | 11 | 1 | 31761 | government | bonds |
| 57.14 | 13 | 7 | 1327 | 30430 | part | of |
| 53.98 | 4 | 1 | 18 | 31754 | waste | paper |
| 53.65 | 4 | 13 | 2 | 31758 | machine | exhibition |
| 52.33 | 7 | 61 | 27 | 31682 | rose | slightly |
| 52.30 | 5 | 9 | 25 | 31738 | passenger | service |
| 49.79 | 4 | 61 | 0 | 31712 | not | yet |
| 48.94 | 9 | 12 | 429 | 31327 | affected | by |
| 48.85 | 13 | 1327 | 12 | 30425 | of | september |
| 48.80 | 9 | 4 | 872 | 30892 | continue | to |
| 47.84 | 4 | 41 | 1 | 31731 | 2 | nd |
| 47.20 | 8 | 27 | 157 | 31585 | competition | from |
| 46.38 | 10 | 472 | 20 | 31275 | a | positive |
| 45.53 | 4 | 18 | 6 | 31749 | per | 100 |
| 44.36 | 7 | 0 | 1333 | 30437 | course | of |
| 43.93 | 5 | 18 | 33 | 31721 | generally | good |
| 43.61 | 19 | 50 | 1321 | 30387 | level | of |
| 43.35 | 20 | 2532 | 25 | 29200 | the | stock |
| 43.07 | 6 | 875 | 0 | 30896 | to | register |
| 43.06 | 3 | 1 | 10 | 31763 | french | speaking |
| 41.69 | 3 | 29 | 0 | 31745 | 3 | rd |
| 41.67 | 3 | 1 | 13 | 31760 | knitting | machines |
| 40.68 | 4 | 5 | 40 | 31728 | 25 | 000 |
| 39.23 | 9 | 5 | 1331 | 30432 | because | of |
| 39.20 | 5 | 40 | 25 | 31707 | stock | markets |
| 38.87 | 2 | 0 | 1 | 31774 | scanner | cash |
| 38.79 | 3 | 0 | 48 | 31726 | pent | up |
| 38.51 | 3 | 23 | 1 | 31750 | firms | surveyed |
| 38.46 | 4 | 2 | 98 | 31673 | restaurant | business |
| 38.28 | 3 | 12 | 3 | 31759 | fell | back |
| 38.14 | 6 | 4 | 432 | 31335 | climbed | by |
| 37.20 | 6 | 41 | 70 | 31660 | total | production |
| 37.15 | 2 | 0 | 2 | 31773 | hay | crop |
| 36.98 | 3 | 10 | 5 | 31759 | current | transactions |

**表 3：按 $\chi^2$ 检验排序的二元组（前 60 条）**

| $\chi^2$ | $k(AB)$ | $k(A \neg B)$ | $k(\neg A B)$ | $k(\neg A \neg B)$ | $A$ | $B$ |
|---|---|---|---|---|---|---|
| 31777.00 | 3 | 0 | 0 | 31774 | natel | offs |
| 31777.00 | 1 | 0 | 0 | 31776 | write | pulp |
| 31777.00 | 1 | 0 | 0 | 31776 | wood | frames |
| 31777.00 | 1 | 0 | 0 | 31776 | window | leathers |
| 31777.00 | 1 | 0 | 0 | 31776 | upholstery | expert |
| 31777.00 | 1 | 0 | 0 | 31776 | surveys | posibilities |
| 31777.00 | 1 | 0 | 0 | 31776 | sees | drawn |
| 31777.00 | 1 | 0 | 0 | 31776 | practically | farms |
| 31777.00 | 1 | 0 | 0 | 31776 | poultry | fees |
| 31777.00 | 1 | 0 | 0 | 31776 | physicians' | varnishes |
| 31777.00 | 1 | 0 | 0 | 31776 | paints | hovered |
| 31777.00 | 1 | 0 | 0 | 31776 | maturity | bacteria |
| 31777.00 | 1 | 0 | 0 | 31776 | listeriosis | presse |
| 31777.00 | 1 | 0 | 0 | 31776 | la | 280 |
| 31777.00 | 1 | 0 | 0 | 31776 | instance | casing |
| 31777.00 | 1 | 0 | 0 | 31776 | cans | crans |
| 31777.00 | 1 | 0 | 0 | 31776 | bluche | intercontinental |
| 31777.00 | 1 | 0 | 0 | 31776 | a313 | water |
| 24441.54 | 10 | 0 | 3 | 31764 | mineral | cash |
| 21184.00 | 2 | 0 | 1 | 31774 | scanner | gas |
| 20424.86 | 9 | 0 | 5 | 31763 | natural | responsibilities |
| 15888.00 | 1 | 1 | 0 | 31775 | suva's | questionable |
| 15888.00 | 1 | 1 | 0 | 31775 | suva's | clients |
| 15888.00 | 1 | 1 | 0 | 31775 | responsible | ink |
| 15888.00 | 1 | 1 | 0 | 31775 | red | forces |
| 15888.00 | 1 | 1 | 0 | 31775 | joined | density |
| 15888.00 | 1 | 1 | 0 | 31775 | highest | modest |
| 15888.00 | 1 | 1 | 0 | 31775 | generating | conversations |
| 15888.00 | 1 | 1 | 0 | 31775 | enables | cherry |
| 15888.00 | 1 | 1 | 0 | 31775 | dessert | lagging |
| 15888.00 | 1 | 1 | 0 | 31775 | consolidated | converter |
| 15888.00 | 1 | 1 | 0 | 31775 | catalytic | grains |
| 15888.00 | 1 | 1 | 0 | 31775 | bread | booking |
| 15888.00 | 1 | 1 | 0 | 31775 | bottlenecks | association's |
| 15888.00 | 1 | 1 | 0 | 31775 | bankers' | abrupt |
| 15888.00 | 1 | 1 | 0 | 31775 | appenzell | 513 |
| 15888.00 | 1 | 1 | 0 | 31775 | 56 | O82 |
| 15888.00 | 1 | 1 | 0 | 31775 | 56 | 520 |
| 15888.00 | 1 | 1 | 0 | 31775 | 46 | classified |
| 15888.00 | 1 | 1 | 0 | 31775 | 43 | 502 |
| 15888.00 | 1 | 1 | 0 | 31775 | 43 | drive |
| 15888.00 | 1 | 0 | 1 | 31775 | wheel | joined |
| 15888.00 | 1 | 0 | 1 | 31775 | shops | collections |
| 15888.00 | 1 | 0 | 1 | 31775 | selected | railcars |
| 15888.00 | 1 | 0 | 1 | 31775 | propelled | arising |
| 15888.00 | 1 | 0 | 1 | 31775 | overcapacities | job |
| 15888.00 | 1 | 0 | 1 | 31775 | listed | fuels |
| 15888.00 | 1 | 0 | 1 | 31775 | liquid | cellulose |
| 15888.00 | 1 | 0 | 1 | 31775 | incl. | oils |
| 15888.00 | 1 | 0 | 1 | 31775 | fats | deteriorate |
| 15888.00 | 1 | 0 | 1 | 31775 | drastically | constructions |
| 15888.00 | 1 | 0 | 1 | 31775 | completing | apples |
| 15888.00 | 1 | 0 | 1 | 31775 | cider | tags |
| 15888.00 | 1 | 0 | 1 | 31775 | bicycle | collections |
| 15888.00 | 1 | 0 | 1 | 31775 | auctioning | crop |
| 15887.50 | 2 | 0 | 2 | 31773 | hay | ... |

**表 4：适合 $\chi^2$ 分析的二元组**

| $\chi^2$ | $k(AB)$ | $k(A \neg B)$ | $k(\neg A B)$ | $k(\neg A \neg B)$ | $A$ | $B$ |
|---|---|---|---|---|---|---|
| 525.02 | 110 | 2442 | 111 | 29114 | the | swiss |
| 286.52 | 76 | 104 | 2476 | 29121 | at | the |
| 51.12 | 26 | 2526 | 66 | 29159 | the | volume |
| 6.03 | 4 | 148 | 2548 | 29077 | be | the |
| 4.48 | 1 | 73 | 2551 | 29152 | months | the |
| 4.31 | 1 | 71 | 2551 | 29154 | increased | the |
| 0.69 | 4 | 70 | 2548 | 29155 | 1986 | the |
| 0.42 | 7 | 62 | 2545 | 29163 | level | the |
| 0.28 | 4 | 60 | 2548 | 29165 | again | the |
| 0.12 | 5 | 2547 | 67 | 29158 | the | increased |
| 0.03 | 18 | 198 | 2534 | 29027 | as | the |

## 7 结论

基于正态分布假设的统计方法在大多数统计文本分析场景中都是无效的，除非使用极其庞大的语料库，或者分析仅限于最常见的词（即那些最不可能令人感兴趣的词）。这一事实在该领域的大部分工作中通常被忽略。使用这类无效方法可能会严重高估相对稀有事件的显著性。

基于二项分布或多项分布的参数统计分析将统计方法的应用范围扩展到了远小于正态分布模型所需文本量的情境，并在该方法的早期应用中展现出良好的前景。

需要进一步开发软件工具，以便使用这些方法对文本进行直接分析。部分工具已经开发完成，将由词汇研究联合会（Consortium for Lexical Research）分发。有关该软件的更多信息，请通过电子邮件联系作者或联合会：ted@nmsu.edu 或 lexical@nmsu.edu。

此外，还有各种无分布方法，它们甚至可能避免文本可用多项分布建模的假设。基于 Fischer 精确检验的度量可能比本文描述的似然比度量更令人满意。此外，使用泊松分布而非多项分布作为计数分布的极限分布也可能带来一些好处。所有这些可能性都应进行测试。

## 8 公式总结 **

对于二项分布情况，对数似然统计量由下式给出：

$$
-2 \log \lambda = 2 [\log L(p_1, k_1, n_1) + \log L(p_2, k_2, n_2) - \log L(p, k_1, n_1) - \log L(p, k_2, n_2)]
$$

其中

$$
\log L(p, n, k) = k \log p + (n - k) \log(1 - p)
$$

且

$$
p_1 = \frac{k_1}{n_1},\quad p_2 = \frac{k_2}{n_2},\quad p = \frac{k_1 + k_2}{n_1 + n_2}
$$

对于多项分布情况，该统计量变为：

$$
-2 \log \lambda = 2 [\log L(\mathbf{P}_1, \mathbf{K}_1) + \log L(\mathbf{P}_2, \mathbf{K}_2) - \log L(\mathbf{Q}, \mathbf{K}_1) - \log L(\mathbf{Q}, \mathbf{K}_2)]
$$

其中

$$
p_{ji} = \frac{k_{ji}}{\sum_j k_{ji}},\quad q_j = \frac{\sum_i k_{ji}}{\sum_{i,j} k_{ji}}
$$

$$
\log L(\mathbf{P}, \mathbf{K}) = \sum_j k_j \log p_j
$$

## 参考文献

[1] Bradley, James V. (1968). *Distribution-Free Statistical Tests*. Prentice Hall.

[2] Brown, Peter F.; Cocke, John; Della Pietra, Stephen A.; Della Pietra, Vincent J.; Jelinek, Frederick; Lafferty, John D.; Mercer, Robert L.; and Roossin, Paul S. (1989). "A statistical approach to machine translation." Technical Report RC 14773 (#66226), IBM Research Division.

[3] Church, Ken W.; Gale, William A.; Hanks, Patrick; and Hindle, Donald (1989). "Parsing, word associations and typical predicate-argument relations." In *Proceedings, International Workshop on Parsing Technologies*, CMU.

[4] Dumais, S.; Furnas, G.; Landauer, T.; Deerwester, S.; and Harshman, R. (1988). "Using latent semantic analysis to improve access to textual information." In *Proceedings, CHI '88*. 281–285.

[5] Gale, William A., and Church, Ken W. (1993). "A program for aligning sentences in bilingual corpora." *Computational Linguistics*, 19(1), 00–00.

[6] Gale, William A., and Church, Ken W. (in press). "Identifying word correspondences in parallel texts."

[7] McDonald, James E.; Plate, Tony; and Schvaneveldt, Roger (1990). "Using Pathfinder to extract semantic information from text." In *Pathfinder Associative Networks: Studies in Knowledge Organization*, edited by Roger Schvaneveldt, 149–164. Ablex.

[8] Mood, A. M.; Graybill, F. A.; and Boes, D. C. (1974). *Introduction to the Theory of Statistics*. McGraw Hill.

[9] Salton, Gerald, and McGill, M. J. (1983). *Introduction to Modern Information Retrieval*. McGraw Hill.

[10] Schvaneveldt, Roger, ed. (1990). *Pathfinder Associative Networks: Studies in Knowledge Organization*. Ablex.
