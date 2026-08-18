# 使用噪声对比估计高效学习词嵌入（Learning word embeddings efficiently with noise-contrastive estimation）

> Andriy Mnih, Koray Kavukcuoglu | DeepMind Technologies | NIPS 2013

本文提出用噪声对比估计（NCE，Noise-Contrastive Estimation）训练 log-bilinear 模型来学习词嵌入，替代复杂且性能敏感的树结构模型。核心发现是——**在单核上只用四分之一的数据和不到一个数量级的计算时间，就达到此前需在 125 核集群上训练才能取得的最佳结果**。

核心内容：

- 提出 vLBL（vector Log-Bilinear，向量 log-bilinear）与 ivLBL（inverse vLBL，逆 vLBL）两类轻量模型，参数更新复杂度与嵌入维度呈线性
- NCE 将密度估计转化为概率二分类：训练 logistic 回归判别数据分布样本与噪声分布样本，使训练时间与词汇量大小无关
- ivLBL 基于分布式假设做逆语言建模（由词预测上下文），等价于朴素贝叶斯分类器或非层次 Skip-gram
- 用 AdaGrad 自适应学习率，通过共享嵌入维度的学习率把内存需求降为每向量一个标量
- 在 15 亿词 Wikipedia 与 4,700 万词 Gutenberg 语料上评估 Google/Microsoft 的类比任务

关键发现：

- ivLBL+NCE25 的 300 维模型训练约 1 天，比同数据训练近两倍的 300D Skip-gram 高 **3-9 个百分点**（GOOGLE 总分 59.7 vs 53.8）
- 训练 4 天的模型仅比 1000D Skip-gram（6B 词、125 核）低 2-4 个百分点；拼接条件与目标表示后差距缩至 2 个百分点
- NCE 训练时间与噪声样本数线性相关且与词汇量无关；5-10 个噪声样本是时间与性能的最佳折中
- Gutenberg 上位置无关权重（I）普遍优于位置相关权重（D），ivLBL 差异尤为明显
- MSR 句子补全挑战：600D ivLBL 达 55.5%，超过文献最佳单模型 LBL 的 54.7%

---

## 摘要

神经语言模型学习的连续值词嵌入最近被证明能非常好地捕获词的语义和句法信息，在多个词相似度任务上创下性能纪录。最佳结果是通过从非常大量的数据中学习高维嵌入获得的，这使得训练方法的可扩展性成为关键因素。

我们提出一种简单且可扩展的学习词嵌入的新方法，基于用噪声对比估计训练 log-bilinear 模型。我们的方法比当前最先进的方法更简单、更快，并产生更好的结果。我们取得了与最佳报告结果相当的成绩——那些结果是在集群上获得的——而我们的方法使用四分之一的数据和少一个数量级的计算时间。我们还研究了多种模型类型，发现更简单模型学习的嵌入至少与更复杂模型学习的嵌入一样好。

## 1 引言

自然语言处理和信息检索系统通常可以从纳入准确的词相似度信息中受益。从大规模无结构文本集合中学习词表示是捕获此类信息的有效方式。这项任务的经典方法是使用词空间模型，用每个词与其他词的共现计数向量表示该词 [16]。这类表示由于词计数向量维度极高而遭受数据稀疏问题。为了解决这个问题，潜在语义分析（LSA，Latent Semantic Analysis）对此类向量进行降维，产生更低维的实值词嵌入。

然而，更好的实值表示由神经语言模型学习，这些模型被训练来在给定前面词的情况下预测句子中的下一个词。此类表示已被用于在经典 NLP 任务上取得出色性能 [4, 18, 17]。不幸的是，由于使用隐藏层和计算归一化概率的成本，很少有神经语言模型能良好扩展到大型数据集和词汇表。

最近，[10] 提出了一种使用轻量级树结构神经语言模型学习词嵌入的可扩展方法。虽然树结构模型可以快速训练，但它们比传统（扁平）模型复杂得多，而且它们的性能对词上树的选择敏感 [13]。受 [10] 出色结果的启发，我们研究了一种基于噪声对比估计（NCE）[6] 的更简单方法，它无需处理树结构模型的复杂性即可快速训练。我们通过使用 log-bilinear 模型 [14] 的非常简单的变体，将使用 NCE 消除训练期间归一化成本所获得的加速进一步复合，使参数更新复杂度与词嵌入维度呈线性。

我们在两个基于类比的词相似度任务 [11, 10] 上评估了我们的方法，并表明尽管训练时间明显更短，我们的模型仍优于 [10] 中在同一 15 亿词 Wikipedia 数据集上训练的 Skip-gram 模型。此外，我们可以在单核上仅训练四天、使用四倍少的训练数据，就获得与在 125-CPU 核集群上训练的巨大 Skip-gram 和 CBOW 模型相当的性能。最后，我们探索了几种模型架构，发现最简单的架构学习的嵌入至少与更复杂架构学习的嵌入一样好。

## 2 神经概率语言模型

神经概率语言模型（NPLM，Neural Probabilistic Language Model）在给定词序列 $h$（称为上下文）的情况下指定目标词 $w$ 的分布。在统计语言建模中，$w$ 通常是句子中的下一个词，而上下文 $h$ 是 $w$ 前面的词序列。虽然一些模型如循环神经语言模型 [9] 可以处理任意长的上下文，但在本文中，我们将注意力限制在固定长度的上下文上。由于我们感兴趣的是学习词表示而非给句子分配概率，我们不需要将模型限制为预测下一个词，例如可以从围绕它的词预测 $w$，正如 [4] 中所做的那样。

给定上下文 $h$，NPLM 使用评分函数 $s_{\theta}(w, h)$ 定义待预测词的分布，该函数量化上下文与候选目标词之间的兼容性。这里 $\theta$ 是模型参数，包括词嵌入。得分通过取指数和归一化转换为概率：

$$
P_{\theta}^{h}(w) = \frac{\exp(s_{\theta}(w, h))}{\sum_{w'} \exp(s_{\theta}(w', h))}. \qquad (1)
$$

不幸的是，评估 $P_{\theta}^{h}(w)$ 和计算相应的似然梯度都需要在整个词汇表上归一化，这意味着此类模型的最大似然训练时间与词汇量大小呈线性关系，因此对除最小词汇表外的所有词汇表都昂贵得令人望而却步。

扩展 NPLM 以支持大词汇表有两种主要方法。第一种涉及使用以词为叶子的树结构词汇表，使训练时间与词汇量大小呈对数关系 [15]。不幸的是，这种方法比最大似然训练复杂得多，而且找到表现良好的树并不简单 [13]。另一种方法是保留模型但使用不同的训练策略。使用重要性采样近似似然梯度是第一个提出的此类方法 [2, 3]，虽然它能产生显著的加速，但遭受稳定性问题。最近，一种训练未归一化概率模型的方法，称为噪声对比估计（NCE）[6]，已被证明是训练 NPLM 的一种稳定且高效的方式 [14]。由于它也比基于树的预测方法简单得多，我们在本文中使用 NCE 训练模型。我们将在第 3.1 节详细描述 NCE。

## 3 可扩展的 log-bilinear 模型

我们感兴趣的是高度可扩展的模型，可以在单核上几天内训练数十亿词的数据集和数十万词的词汇表，这排除了大多数传统神经语言模型，如 [1] 和 [4] 中的那些。我们将使用 log-bilinear 语言模型（LBL，Log-Bilinear language model）[12] 作为起点，它不像传统 NPLM 那样有隐藏层，而是在词特征向量空间中进行线性预测。特别是，我们将使用更可扩展的 LBL 版本 [14]，它对上下文权重使用向量而非矩阵，以避免矩阵-向量乘法的高成本。该模型与我们描述的所有其他模型一样，有两组词表示：一组用于目标词（即被预测的词），一组用于上下文词。我们用 $q_w$ 和 $r_w$ 分别表示词 $w$ 的目标和上下文表示。给定上下文词序列 $h = w_1, ..., w_n$，模型通过取上下文词特征向量的线性组合来计算目标词的预测表示：

$$
\hat{q}(h) = \sum_{i=1}^{n} c_i \odot r_{w_i}, \qquad (2)
$$

其中 $c_i$ 是位置 $i$ 处上下文词的权重向量，$\odot$ 表示逐元素乘法。上下文可以由被预测词之前、之后或周围的词组成。然后评分函数计算预测特征向量与词 $w$ 的特征向量之间的相似度：

$$
s_{\theta}(w, h) = \hat{q}(h)^{\top} q_w + b_w, \qquad (3)
$$

其中 $b_w$ 是捕获词 $w$ 与上下文无关频率的偏置。我们将此模型称为 vLBL（向量 LBL）。

vLBL 可以通过消除位置相关权重并简单地平均上下文词特征向量来计算预测特征向量而变得更简单：$\hat{q}(h) = \frac{1}{n} \sum_{i=1}^{n} r_{w_i}$。结果有点像局部主题模型，它忽略上下文词的顺序，可能迫使其捕获更多语义信息，也许以句法为代价。简单平均上下文词特征向量的想法在 [8] 中引入，用于以大型上下文（如整个文档）为条件。由此产生的模型可以被视为 [10] 中 CBOW 模型的非层次版本。

由于我们的主要关注点是学习词表示而非创建有用的语言模型，我们可以自由地脱离从上下文预测目标词的范式，例如反其道而行之。这种方法由分布式假设驱动，该假设指出意义相似的词往往出现在相同的上下文中 [7]，从而建议寻找捕获其上下文分布的词表示。从词预测上下文的逆语言建模方法是做到这一点的自然方式。一些经典词空间模型如 HAL 和 COALS [16] 遵循这种方法，用词袋表示上下文分布，但它们不从这些信息中学习嵌入。

不幸的是，预测 $n$ 词上下文需要建模 $n$ 个词的联合分布，这比建模单个词的分布困难得多。我们通过假设不同上下文位置的词在给定当前词 $w$ 时条件独立来使任务可处理：

$$
P_{\theta}^{w}(h) = \prod_{i=1}^{n} P_{i,\theta}^{w}(w_i). \qquad (4)
$$

虽然这个假设可以通过在上下文分布中引入一些马尔可夫结构而轻松放宽而不放弃可处理性，但我们把调查这个方向留作未来工作。上下文词分布 $P_{i,\theta}^{w}(w_i)$ 只是以当前词为条件的 vLBL 模型，由评分函数定义：

$$
s_{i,\theta}(w_i, w) = (c_i \odot r_w)^{\top} q_{w_i} + b_{w_i}. \qquad (5)
$$

由此产生的模型可以看作是以词嵌入参数化的朴素贝叶斯分类器。由于该模型执行逆语言建模，我们将它称为 ivLBL。

与我们的传统语言模型一样，我们也考虑没有位置相关权重的更简单版本，由评分函数定义：

$$
s_{i,\theta}(w_i, w) = r_w^{\top} q_{w_i} + b_{w_i}. \qquad (6)
$$

由此产生的模型是 Skip-gram 模型 [10] 的非层次对应物。注意与上面论文中的树模型（只为词学习条件嵌入）不同，在我们的模型中每个词都有条件嵌入和目标嵌入，它们可能捕获互补的信息。树模型用与树节点相关的参数向量（而非单个词）替换目标嵌入。

### 3.1 噪声对比估计

我们使用噪声对比估计训练模型，这是一种拟合未归一化模型的方法 [6]，在 [14] 中改编用于神经语言建模。NCE 基于将密度估计简化为概率二分类。基本思想是训练一个 logistic 回归分类器，基于样本在模型和噪声分布下的概率之比，区分来自数据分布的样本和来自某个"噪声"分布的样本。NCE 的主要优势是它允许我们拟合未显式归一化的模型，使训练时间实际上与词汇量大小无关。因此，我们将能够从等式 1 中丢弃归一化因子，在训练期间简单地使用 $\exp(s_{\theta}(w, h))$ 代替 $P_{\theta}^{h}(w)$。使用这种方法训练的 NPLM 的困惑度已被证明与用最大似然学习训练的模型相当，但计算成本只是一小部分。

假设我们想学习某个特定上下文 $h$ 的词分布，记为 $P^{h}(w)$。为此，我们创建一个辅助二分类问题，将训练数据视为正例，将从噪声分布 $P_n(w)$ 抽取的样本视为负例。我们可以自由选择任何易于采样和计算概率且不给任何词分配零概率的噪声分布。我们将使用训练数据的（全局）一元分布作为噪声分布，这一选择已知对训练语言模型效果很好。如果我们假设噪声样本的频率是数据样本的 $k$ 倍，那么给定样本来自数据的概率是 $P^{h}(D = 1|w) = \frac{P_d^{h}(w)}{P_d^{h}(w) + kP_n(w)}$。我们通过使用我们的模型分布代替 $P_d^{h}$ 来获得对这个概率的估计：

$$
P^{h}(D = 1|w, \theta) = \frac{P_{\theta}^{h}(w)}{P_{\theta}^{h}(w) + kP_n(w)} = \sigma \left( \Delta s_{\theta}(w, h) \right), \qquad (7)
$$

其中 $\sigma(x)$ 是 logistic 函数，$\Delta s_{\theta}(w, h) = s_{\theta}(w, h) - \log(kP_n(w))$ 是词 $w$ 在模型和（缩放后的）噪声分布下得分之差。$P_n(w)$ 前面的缩放因子 $k$ 说明噪声样本的频率是数据样本的 $k$ 倍。注意在上面的等式中，我们使用 $s_{\theta}(w, h)$ 代替 $\log P_{\theta}^{h}(w)$，忽略归一化项，因为我们在处理未归一化的模型。我们可以这样做，因为 NCE 目标鼓励模型近似归一化，并且如果模型类包含数据分布，则恢复完全归一化的模型 [6]。

我们通过最大化正确标签 $D$ 的对数后验概率（在数据和噪声样本上平均）来拟合模型：

$$
J_{h}(\theta) = \mathbb{E}_{P_d^{h}} \left[ \log P^{h}(D = 1|w, \theta) \right] + k\mathbb{E}_{P_n} \left[ \log P^{h}(D = 0|w, \theta) \right] = \mathbb{E}_{P_d^{h}} \left[ \log \sigma \left( \Delta s_{\theta}(w, h) \right) \right] + k\mathbb{E}_{P_n} \left[ \log \left( 1 - \sigma \left( \Delta s_{\theta}(w, h) \right) \right) \right], \qquad (8)
$$

在实践中，对噪声分布的期望通过采样来近似。因此，我们通过生成 $k$ 个噪声样本 $\{x_i\}$ 并计算下式来估计词/上下文对 $w, h$ 对等式 8 梯度的贡献：

$$
\frac{\partial}{\partial \theta} J_{h,w}(\theta) = \left( 1 - \sigma \left( \Delta s_{\theta}(w, h) \right) \right) \frac{\partial}{\partial \theta} \log P_{\theta}^{h}(w) - \sum_{i=1}^{k} \left[ \sigma \left( \Delta s_{\theta}(x_i, h) \right) \frac{\partial}{\partial \theta} \log P_{\theta}^{h}(x_i) \right]. \qquad (9)
$$

注意等式 9 中的梯度涉及对 $k$ 个噪声样本的求和，而不是对整个词汇表的求和，使 NCE 训练时间与噪声样本数线性相关，与词汇量大小无关。随着我们增加噪声样本数 $k$，这个估计接近归一化模型的似然梯度，使我们能够在计算成本和估计精度之间权衡 [6]。

NCE 与一种用于非概率神经语言模型的训练方法有一些相似之处，后者涉及优化基于间隔的排序目标 [4]。由于该方法是非概率的，它超出了本文的范围，尽管看看它是否可以用来学习有竞争力的词嵌入会很有趣。

## 4 评估词嵌入

在语言建模之外使用神经语言模型学习的词嵌入是一个相对较新的发展。一个早期例子是 [4] 的多层神经网络，它被训练执行几个 NLP 任务，完全用学习到的词嵌入表示词。[18] 提供了用不同方法学习的几个词嵌入的首次比较，并表明将它们纳入已建立的 NLP 流水线可以提升它们的性能。

最近焦点已转向更直接地评估此类表示，而不是测量它们对更大系统性能的影响。微软研究院（MSR，Microsoft Research）发布了两个挑战集：一组每个句子有一个待填空词 [20]，以及一组类比问题 [11]，分别设计用于评估词表示的语义和句法内容。另一个由语义和句法类比问题组成的数据集由 Google 发布 [10]。

在本文中，我们将专注于两个基于类比的挑战集，它们由"a 之于 b 正如 c 之于 ?"形式的问题组成，记为 a : b → c : ?。任务是识别保留的第四个词，只有完全匹配的词才算正确。神经语言模型学习的词嵌入在使用以下基于向量相似度的协议回答这些问题时已被证明在这些数据集上表现非常好。假设 $\vec{w}$ 是词 $w$ 的归一化为单位范数的表示向量。那么，遵循 [11]，我们通过找到表示最接近 $\vec{b} - \vec{a} + \vec{c}$（根据余弦相似度）的词 $d^{*}$ 来回答 a : b → c : ?：

$$
d^{*} = \arg \max_{x} \frac{(\vec{b} - \vec{a} + \vec{c})^{\top} \vec{x}}{\| \vec{b} - \vec{a} + \vec{c} \|}. \qquad (10)
$$

我们发现，为公开可用的词嵌入重现 [10] 和 [11] 中报告的结果，需要在用等式 10 寻找 $d^{*}$ 时从词汇表中排除 $b$ 和 $c$，尽管论文中没有明确说明。要理解为什么这是必要的，我们可以将等式 10 重写为：

$$
d^{*} = \arg \max_{x} \vec{b}^{\top} \vec{x} - \vec{a}^{\top} \vec{x} + \vec{c}^{\top} \vec{x} \qquad (11)
$$

并注意到将 $x$ 设为 $b$ 或 $c$ 分别最大化第一项或第三项（因为向量是归一化的），从而产生高相似度得分。这个等式建议了对 $d^{*}$ 的如下解释：它就是表示与 $\vec{b}$ 和 $\vec{c}$ 最相似且与 $\vec{a}$ 不相似的词，这使得将 $b$ 和 $c$ 本身排除在考虑之外非常自然。

## 5 实验评估

### 5.1 数据集

我们在第 4 节描述的 Google 和微软研究院最近发布的两个基于类比的词相似度任务上评估了词嵌入。我们无法在原始论文中用于学习嵌入的数据上训练，因为那些数据不容易获得。[10] 使用了由 60 亿词组成的专有 Google News 语料库，而 [11] 中使用的 3.2 亿词训练集是几个语言数据联盟（LDC，Linguistic Data Consortium）语料库的汇编，其中一些仅对其订阅者开放。

相反，我们决定使用两个免费可用的数据集：2013 年 4 月的英语 Wikipedia 转储和构成 MSR 句子补全挑战 [19] 规范训练数据的约 500 个 Project Gutenberg 文本集合。我们通过剥离 XML 格式、将所有词映射为小写、将所有数字替换为 7 来预处理 Wikipedia，留下 15 亿词。保留所有出现至少 10 次的词，产生约 87.2 万个词的词汇表。使用如此大的词汇表是为了展示我们方法的可扩展性，并确保模型已经看到几乎所有它们将被测试的词。在预处理 4,700 万词的 Gutenberg 数据集时，我们保留了所有出现 5 次或更多次的词，产生 8 万个词的词汇表。注意许多用于测试表示的词在该数据集中缺失，这大大限制了使用它时能达到的准确率。为了使我们的结果与其他论文直接可比，我们报告使用等式 10 计算的准确率得分，将问题中的第二个和第三个词排除在考虑之外，如第 4 节所述。

### 5.2 训练细节

所有模型都在单核上训练，使用大小为 100 的小批量，初始学习率为 $3 \times 10^{-2}$。未使用正则化。最初我们使用了 [14] 中描述的基于验证集的学习率调整方案，该方案在验证集困惑度一段时间内未改善时将学习率减半，但我们发现尽管困惑度得分低，它却导致糟糕的表示，这很可能是由于训练不足。[10] 中描述的线性学习率计划产生了更好的结果。不幸的是，使用它需要提前知道将对数据执行多少次遍历，这并不总是可能或方便的。也许更严重的是，这种方法可能导致稀有词的表示训练不足，因为所有表示共享相同的学习率。

AdaGrad [5] 提供了一种自动处理此问题的方法。虽然 AdaGrad 已经在分布式设置中用于训练神经语言模型 [10]，但我们发现即使在单个 CPU 核上它也帮助学习更好的词表示。我们通过为词嵌入的所有维度使用相同的学习率，降低了 AdaGrad 潜在令人望而却步的内存需求（它需要为每个参数存储梯度平方值的运行和）。因此，我们只为每个嵌入向量存储一个额外的数字，这在训练具有数亿参数的模型时很有帮助。

**表 1：词相似度任务上的准确率百分比。模型有 100D 词嵌入，并在 15 亿词 Wikipedia 数据集上训练来预测当前词两侧各 5 个词。Skip-gram(\*) 是 [10] 中模型的我们的实现。ivLBL 是没有位置相关权重的逆语言模型。NCEk 表示使用 $k$ 个噪声样本的 NCE 训练。**

| 模型 | GOOGLE 语义 | GOOGLE 句法 | GOOGLE 总体 | MSR | 时间（小时） |
|---|---|---|---|---|---|
| SKIP-GRAM(\*) | 28.0 | 36.4 | 32.6 | 31.7 | 12.3 |
| IVLBL+NCE1 | 28.4 | 42.1 | 35.9 | 34.9 | 3.1 |
| IVLBL+NCE2 | 30.8 | 44.1 | 38.0 | 36.2 | 4.0 |
| IVLBL+NCE3 | 34.2 | 43.6 | 39.4 | 36.3 | 5.1 |
| IVLBL+NCE5 | 37.2 | 44.7 | 41.3 | 36.7 | 7.3 |
| IVLBL+NCE10 | 38.9 | 45.0 | 42.2 | 36.0 | 12.2 |
| IVLBL+NCE25 | 40.0 | 46.1 | 43.3 | 36.7 | 26.8 |

### 5.3 结果

受 [10] 中树模型出色性能的启发，我们从比较那篇论文中表现最好的模型 Skip-gram 与其非层次对应物（第 3 节提出的没有位置相关权重的 ivLBL，用 NCE 训练）开始。由于没有公开可用的 Skip-gram 实现，我们自己编写。我们的实现忠实于论文中的描述，只有一个例外。为加速训练，我们不预测当前词周围的所有上下文词，而只预测一个上下文词，使用论文中的非均匀加权方案随机采样。注意我们的模型也使用相同的上下文词采样方法训练。为使比较公平，我们在这些实验中不为模型使用 AdaGrad，而是使用 [10] 中那样的线性学习率计划。

表 1 显示了在 Wikipedia 数据集上训练的两种模型在词相似度任务上的结果。我们用不同数量的噪声样本运行了几次 NCE 训练，以研究该参数对表示质量和训练时间的影响。模型训练了三个轮次，根据我们的经验，这在训练时间和表示质量之间提供了合理的折中。[^1] 所有 NCE 训练的模型都优于 Skip-gram。准确率随着使用的噪声样本数量稳步提高，训练时间也是如此。运行时间和性能之间的最佳折中似乎是用 5 或 10 个噪声样本实现的。

然后我们尝试用 AdaGrad 训练模型，发现它显著提高了使用 10 或 25 个噪声样本训练时获得的嵌入质量，使 NCE25 模型的语义得分提高了 10 多个百分点。受到鼓舞，我们用这种方法训练了两个具有位置无关权重和不同嵌入维度的 ivLBL 模型，训练了几天。由于 [10] 中的一些最佳结果是用 CBOW 模型获得的，我们也训练了其第 3 节中的非层次对应物（具有位置无关权重的 vLBL），使用 100/300/600 维嵌入和 5 个噪声样本的 NCE，训练时间较短。注意由于那篇论文中使用的 Google News 数据集不可用，我们在 Wikipedia 上训练。ivLBL 和 vLBL 模型的得分分别使用条件词和目标词表示获得，而标有 d × 2 的得分是通过在归一化后拼接两种词表示获得的。

**表 2：大型模型在词相似度任务上的准确率百分比。Skip-gram† 和 CBOW† 结果来自 [10]。ivLBL 模型预测当前词前后各 5 个词。vLBL 模型从前面 5 个和后面 5 个词预测当前词。**

| 模型 | 嵌入维度 | 训练集大小 | GOOGLE 语义 | GOOGLE 句法 | GOOGLE 总体 | MSR | 时间（天） |
|---|---|---|---|---|---|---|---|
| SKIP-GRAM† | 300 | 1.6B | 52.2 | 55.1 | 53.8 | - | 2.0 |
| SKIP-GRAM† | 300 | 785M | 56.7 | 52.2 | 55.5 | - | 2.5 |
| SKIP-GRAM† | 1000 | 6B | 66.1 | 65.1 | 65.6 | - | 2.5×125 |
| IVLBL+NCE25 | 300 | 1.5B | 61.2 | 58.4 | 59.7 | 48.8 | 1.2 |
| IVLBL+NCE25 | 300 | 1.5B | 63.6 | 61.8 | 62.6 | 52.4 | 4.1 |
| IVLBL+NCE25 | 300×2 | 1.5B | 65.2 | 63.0 | 64.0 | 54.2 | 4.1 |
| IVLBL+NCE25 | 100 | 1.5B | 52.6 | 48.5 | 50.3 | 39.2 | 1.2 |
| IVLBL+NCE25 | 100 | 1.5B | 55.9 | 50.1 | 53.2 | 42.3 | 2.9 |
| IVLBL+NCE25 | 100×2 | 1.5B | 59.3 | 54.2 | 56.5 | 44.6 | 2.9 |
| CBOW† | 300 | 1.6B | 16.1 | 52.6 | 36.1 | - | 0.6 |
| CBOW† | 1000 | 6B | 57.3 | 68.9 | 63.7 | - | 2×140 |
| VLBL+NCE5 | 300 | 1.5B | 40.3 | 55.4 | 48.5 | 48.7 | 0.3 |
| VLBL+NCE5 | 100 | 1.5B | 45.0 | 56.8 | 51.5 | 52.3 | 2.0 |
| VLBL+NCE5 | 300 | 1.5B | 54.2 | 64.8 | 60.0 | 58.1 | 2.0 |
| VLBL+NCE5 | 600 | 1.5B | 57.3 | 66.0 | 62.1 | 59.1 | 2.0 |
| VLBL+NCE5 | 600×2 | 1.5B | 60.5 | 67.1 | 64.1 | 60.8 | 3.0 |

表 2 报告的结果表明，在使用相当的时间和数据量训练时，我们的模型大幅优于其层次对应物。例如，300D ivLBL 模型仅训练一天多一点，就比在相同数据上训练近两倍时间的 300D Skip-gram 获得高 3-9 个百分点的准确率得分。同一模型训练四天取得的准确率得分仅比用四倍数据、75 倍 CPU 周期训练的 1000D Skip-gram 低 2-4 个百分点。通过将条件词和目标词表示拼接在一起计算词相似度得分，我们可以将准确率差距缩小到 2 个百分点，且没有额外的计算成本。vLBL 模型与 CBOW 模型相比取得的准确率遵循类似模式。我们的模型再次更快地获得更好的准确率得分，并且我们可以用少得多的数据和远少的计算量将结果差距缩小到 3 个百分点以内。

**表 3：在 4,700 万词 Gutenberg 数据集上用 NCE5 和 AdaGrad 训练 20 轮的各种模型的结果。(D) 和 (I) 分别表示有和没有位置相关权重的模型。对于每个任务，左（右）列给出使用条件（目标）词嵌入获得的准确率。nL（nR）表示当前词左（右）侧 $n$ 个词。**

| 模型 | 上下文大小 | GOOGLE 语义 | GOOGLE 句法 | GOOGLE 总体 | MSR | 时间（小时） |
|---|---|---|---|---|---|---|
| VLBL(D) | 5L+5R | 2.4 / 2.6 | 24.7 / 23.8 | 14.6 / 14.2 | 23.4 / 23.1 | 2.6 |
| VLBL(D) | 10L | 1.9 / 2.8 | 22.1 / 14.8 | 12.9 / 9.3 | 20.9 / 9.0 | 2.6 |
| VLBL(D) | 10R | 2.7 / 2.4 | 13.1 / 24.1 | 8.4 / 14.2 | 8.8 / 23.0 | 2.6 |
| VLBL(I) | 5L+5R | 3.0 / 2.9 | 27.5 / 29.6 | 16.4 / 17.5 | 22.9 / 24.2 | 2.3 |
| VLBL(I) | 10L | 2.5 / 2.8 | 23.5 / 16.1 | 14.0 / 10.1 | 19.8 / 10.1 | 2.3 |
| VLBL(I) | 10R | 2.3 / 2.6 | 16.2 / 24.6 | 9.9 / 14.6 | 10.0 / 20.3 | 2.1 |
| IVLBL(D) | 5L+5R | 2.8 / 2.3 | 15.1 / 13.0 | 9.5 / 8.1 | 14.5 / 14.0 | 1.2 |
| IVLBL(I) | 5L+5R | 2.8 / 2.6 | 26.8 / 26.8 | 15.9 / 15.8 | 21.4 / 21.0 | 1.2 |

为确定我们是否因使用位置无关权重而削弱了模型，我们在 Gutenberg 语料库上评估了第 3 节描述的所有模型架构。模型用 NCE5 和 AdaGrad 训练 20 轮。我们在表 3 中报告每个模型用条件表示和目标表示（分别为左列和右列）获得的准确率。也许令人惊讶的是，结果表明用位置无关权重（标为 (I)）学习的表示往往优于用位置相关权重学习的表示。传统语言模型（vLBL）的差异很小，但对逆语言模型（ivLBL）来说相当明显。表现最好的表示由上下文围绕词且权重位置无关的传统语言模型学习。

**句子补全：** 我们还将我们的方法应用于 MSR 句子补全挑战 [19]，任务是补全 1,040 个测试句子中的每一个，从五个候选词列表中选择缺失的词。使用按 [14] 预处理的 4,700 万词 Gutenberg 数据集作为训练集，我们训练了几个用 NCE5 预测当前词前后各 5 个词的 ivLBL 模型。为补全句子，我们为每个候选词计算缺失词周围 10 个词的概率（使用等式 4），并选择产生最高值的那个。表 4 中给出的准确率得分（连同几个基线的得分）显示 ivLBL 模型表现非常好。即使嵌入维度最低为 100 的模型也达到 51.0% 的正确率，而 [10] 中报告的使用 640D 嵌入的 Skip-gram 模型为 48.0% 正确。600D 嵌入模型达到的 55.5% 正确率也优于文献中该数据集的最佳单模型得分（[14] 中的 54.7%）。

**表 4：MSR 句子补全挑战数据集上的准确率。**

| 模型 | 上下文大小 | 潜在维度 | 正确百分比 |
|---|---|---|---|
| LSA [19] | SENTENCE | 300 | 49 |
| SKIP-GRAM [10] | 10L+10R | 640 | 48.0 |
| LBL [14] | 10L | 300 | 54.7 |
| IVLBL | 5L+5R | 100 | 51.0 |
| IVLBL | 5L+5R | 300 | 55.2 |
| IVLBL | 5L+5R | 600 | 55.5 |

[^1]: 我们通过将 Skip-gram 模型训练 10 轮来检查这一点，结果准确率没有大幅提高。

## 6 讨论

我们提出了一种学习词嵌入的新高度可扩展方法，涉及用噪声对比估计训练轻量级 log-bilinear 语言模型。它比 [10] 的树结构语言建模方法更简单，并更快地产生性能更好的嵌入。使用我们方法的简单单核实现学习的嵌入取得的准确率得分与最佳报告结果相当，而最佳结果是在大型集群上用四倍数据和几乎两个数量级的 CPU 周期获得的。我们在这篇论文中报告的得分也易于比较，因为我们只在公开可用的数据上训练模型。

还有几个有前景的方向有待探索。[8] 最近提出了一种为每个词学习多种表示的方法，通过在训练模型之前对词出现的上下文进行聚类，并为每个聚类分配不同的表示。由于 ivLBL 从词预测上下文，它自然允许每个当前词使用多个上下文表示，从而基于混合建模产生更规范的方法。在上下文词和目标词之间共享表示也值得研究，因为它可能带来更好估计的稀有词表示。

## 致谢

我们感谢 Volodymyr Mnih 的有益评论。

## 参考文献

[1] Yoshua Bengio, Rejean Ducharme, Pascal Vincent, and Christian Jauvin. A neural probabilistic language model. Journal of Machine Learning Research, 3:1137–1155, 2003.

[2] Yoshua Bengio and Jean-Sébastien Senécal. Quick training of probabilistic neural nets by importance sampling. In AISTATS'03, 2003.

[3] Yoshua Bengio and Jean-Sébastien Senécal. Adaptive importance sampling to accelerate training of a neural probabilistic language model. IEEE Transactions on Neural Networks, 19(4):713–722, 2008.

[4] R. Collobert and J. Weston. A unified architecture for natural language processing: Deep neural networks with multitask learning. In Proceedings of the 25th International Conference on Machine Learning, 2008.

[5] John Duchi, Elad Hazan, and Yoram Singer. Adaptive subgradient methods for online learning and stochastic optimization. Journal of Machine Learning Research, 12:2121–2159, 2010.

[6] M.U. Gutmann and A. Hyvärinen. Noise-contrastive estimation of unnormalized statistical models, with applications to natural image statistics. Journal of Machine Learning Research, 13:307–361, 2012.

[7] Zellig S Harris. Distributional structure. Word, 1954.

[8] Eric H Huang, Richard Socher, Christopher D Manning, and Andrew Y Ng. Improving word representations via global context and multiple word prototypes. In Proceedings of the 50th Annual Meeting of the Association for Computational Linguistics, pages 873–882, 2012.

[9] T. Mikolov, M. Karafiát, L. Burget, J. Cernocky, and S. Khudanpur. Recurrent neural network based language model. In Eleventh Annual Conference of the International Speech Communication Association, 2010.

[10] Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. Efficient estimation of word representations in vector space. International Conference on Learning Representations 2013, 2013.

[11] Tomas Mikolov, Wen-tau Yih, and Geoffrey Zweig. Linguistic regularities in continuous space word representations. Proceedings of NAACL-HLT, 2013.

[12] A. Mnih and G. Hinton. Three new graphical models for statistical language modelling. Proceedings of the 24th International Conference on Machine Learning, pages 641–648, 2007.

[13] Andriy Mnih and Geoffrey Hinton. A scalable hierarchical distributed language model. In Advances in Neural Information Processing Systems, volume 21, 2009.

[14] Andriy Mnih and Yee Whye Teh. A fast and simple algorithm for training neural probabilistic language models. In Proceedings of the 29th International Conference on Machine Learning, pages 1751–1758, 2012.

[15] Frederic Morin and Yoshua Bengio. Hierarchical probabilistic neural network language model. In AISTATS'05, pages 246–252, 2005.

[16] Magnus Sahlgren. The Word-Space Model: Using distributional analysis to represent syntagmatic and paradigmatic relations between words in high-dimensional vector spaces. PhD thesis, Stockholm, 2006.

[17] R. Socher, C.C. Lin, A.Y. Ng, and C.D. Manning. Parsing natural scenes and natural language with recursive neural networks. In International Conference on Machine Learning (ICML), 2011.

[18] J. Turian, L. Ratinov, and Y. Bengio. Word representations: A simple and general method for semi-supervised learning. In Proceedings of the 48th Annual Meeting of the Association for Computational Linguistics, pages 384–394, 2010.

[19] G. Zweig and C.J.C. Burges. The Microsoft Research Sentence Completion Challenge. Technical Report MSR-TR-2011-129, Microsoft Research, 2011.

[20] Geoffrey Zweig and Chris J.C. Burges. A challenge set for advancing language modeling. In Proceedings of the NAACL-HLT 2012 Workshop: Will We Ever Really Replace the N-gram Model? On the Future of Language Modeling for HLT, pages 29–36, 2012.
