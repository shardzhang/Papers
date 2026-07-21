# 关于神经机器翻译中使用极大目标词表的研究（On Using Very Large Target Vocabulary for Neural Machine Translation）

> Sébastien Jean, KyungHyun Cho, Roland Memisevic, Yoshua Bengio | Université de Montréal

本文分享了2015年关于神经机器翻译中处理极大目标词表的经典论文。核心内容：
- 提出基于重要性采样的近似训练方法，在不增加训练复杂度的情况下使用极大目标词表
- 解码时通过候选列表（candidate list）选择目标词子集，避免全词表计算开销
- 在WMT'14英法、英德任务上达到当时最佳性能，BLEU分数与顶尖系统相当

关键发现：
- 重要性采样使得训练复杂度与目标词表大小无关，可处理50万词规模
- 结合候选列表 + UNK替换 + 模型集成可进一步提升翻译质量
- 单模型性能超越基线RNNsearch，集成8模型后达到当时State-of-the-art水平

---

## 摘要

神经机器翻译（Neural Machine Translation）是一种最近提出的、完全基于神经网络的机器翻译方法，与基于短语的统计机器翻译等现有方法相比，已展现出有潜力的结果。尽管最近取得了成功，神经机器翻译在处理较大词表方面仍存在局限性，因为训练复杂度和解码复杂度都随着目标词数量的增加而成比例增长。在本文中，我们提出了一种基于重要性采样的方法，该方法允许我们在不增加训练复杂度的情况下使用极大的目标词表。我们展示了即使模型拥有极大的目标词表，也可以通过仅选择整个目标词表的一小部分子集来高效地进行解码。实验发现，由所提方法训练的模型在性能上与使用小词表的基线模型以及基于LSTM的神经机器翻译模型相当，甚至在某些情况下更优。此外，当我们使用少量具有极大目标词表的模型进行集成时，我们在WMT'14的英\rightarrow德和英\rightarrow法翻译任务上均取得了与当时最先进技术可比的性能（以BLEU衡量）。

## 1 引言

神经机器翻译（NMT）是近年来引入的一种解决机器翻译问题的方法（Kalchbrenner and Blunsom, 2013; Bahdanau et al., 2014; Sutskever et al., 2014）。在神经机器翻译中，我们构建一个单一的神经网络，该网络读取源句子并生成其翻译。整个神经网络被联合训练，以最大化在给定源句子的条件下正确翻译的条件概率，使用双语语料库。NMT模型已被证明其性能与最广泛使用的传统翻译系统相当（Sutskever et al., 2014; Bahdanau et al., 2014）。

神经机器翻译相较于现有的统计机器翻译系统（特别是基于短语的系统（Koehn et al., 2003））具有若干优势。首先，NMT只需要最少的领域知识。例如，（Sutskever et al., 2014）、（Bahdanau et al., 2014）或（Kalchbrenner and Blunsom, 2013）中提出的所有模型都不假设源句子和目标句子具有任何语言属性，除它们是词序列之外。其次，整个系统被联合调优以最大化翻译性能，而不像现有的基于短语的系统那样由许多单独调优的特征函数组成。最后，NMT模型的内存占用通常远小于依赖维护大型短语对表的现有系统。

尽管有这些优势和令人鼓舞的结果，NMT与现有的基于短语的方法相比仍存在一个主要限制：目标词的数量必须受到限制。这主要是因为训练和使用NMT模型的复杂度随着目标词数量的增加而增加。

通常的做法是构建一个由k个最频繁词组成的目标词表（所谓的shortlist），其中k通常在30,000（Bahdanau et al., 2014）到80,000（Sutskever et al., 2014）之间。任何未包含在此词表中的词都被映射到一个代表未知词的特殊标记[UNK]。当目标句子中只有少量未知词时，这种方法效果良好，但已有观察表明，翻译性能会随着未知词数量的增加而迅速下降（Cho et al., 2014a; Bahdanau et al., 2014）。

在本文中，我们提出了一种基于（有偏）重要性采样的近似训练算法，该算法允许我们训练具有更大目标词表的NMT模型。所提出的算法有效地将训练期间的计算复杂度维持在仅使用全词表的一小部分子集的水平。一旦具有极大目标词表的模型训练完成，我们可以选择使用所有目标词或仅使用其中一部分。

我们将所提出的算法与基线基于shortlist的方法在英\rightarrow法和英\rightarrow德翻译任务上进行了比较，使用了（Bahdanau et al., 2014）中引入的NMT模型。实验结果表明，使用更大的词表有可能获得更好的翻译性能，并且我们的方法在训练和解码速度上都没有过多牺牲。此外，我们展示了使用该算法训练的模型在WMT'14英\rightarrow法翻译任务上取得了当时单个NMT模型的最佳翻译性能。

## 2 神经机器翻译与有限词表问题

在本节中，我们简要描述（Bahdanau et al., 2014）中最近提出的一种神经机器翻译方法。基于此描述，我们解释神经机器翻译中有限词表的问题。

### 2.1 神经机器翻译

神经机器翻译是一种最近提出的机器翻译方法，它使用单个联合训练的神经网络来最大化翻译性能（Forcada and ˜Neco, 1997; Kalchbrenner and Blunsom, 2013; Cho et al., 2014b; Sutskever et al., 2014; Bahdanau et al., 2014）。

神经机器翻译通常实现为编码器-解码器网络。编码器读取源句子 $x = (x_1, \ldots, x_T)$ 并将其编码为隐藏状态序列 $h = (h_1, \cdots, h_T)$：

$$h_t = f (x_t, h_{t-1}). \qquad (1}$$

然后，解码器（另一个循环神经网络）根据编码后的隐藏状态序列 $h$ 生成相应的翻译 $y = (y_1, \cdots, y_{T'})$：

$$p(y_t | y_{<t}, x) \propto \exp \{q (y_{t-1}, z_t, c_t)\}, \qquad (2}$$

其中

$$z_t = g (y_{t-1}, z_{t-1}, c_t), \qquad (3}$$
$$c_t = r (z_{t-1}, h_1, \ldots, h_T), \qquad (4}$$

且 $y_{<t} = (y_1, \ldots, y_{t-1})$。

整个模型被联合训练，以最大化在给定源句子的条件下正确翻译的条件对数概率，其中 $\theta$ 为模型参数：

$$\theta^* = \arg\max_{\theta} \sum_{n=1}^{N} \sum_{t=1}^{T_n} \log p(y^n_t | y^n_{<t}, x^n),$$

其中 $(x^n, y^n)$ 是第 $n$ 个训练句子对，$T_n$ 是第 $n$ 个目标句子 $(y^n)$ 的长度。

#### 2.1.1 详细描述

在本文中，我们使用一种采用注意力机制的神经机器翻译的具体实现，如（Bahdanau et al., 2014）中最近提出的那样。

在（Bahdanau et al., 2014）中，公式 (1) 中的编码器由双向循环神经网络实现，使得：

$$h_t = [\overleftarrow{h}_t; \overrightarrow{h}_t],$$

其中

$$\overleftarrow{h}_t = f(x_t, \overleftarrow{h}_{t+1}), \quad \overrightarrow{h}_t = f(x_t, \overrightarrow{h}_{t-1}).$$

他们使用门控循环单元（GRU）作为 $f$（参见，例如 (Cho et al., 2014b)）。

解码器在每个时间步将上下文向量 $c_t$ 计算为隐藏状态 $(h_1, \ldots, h_T)$ 的凸组合，系数 $\alpha_1, \ldots, \alpha_T$ 由下式计算：

$$\alpha_t = \frac{\exp \{a (h_t, z_{t-1})\}}{\sum_k \exp \{a (h_k, z_{t-1})\}}, \qquad (5}$$

其中 $a$ 是一个具有单隐藏层的前馈神经网络。

公式 (3) 中解码器的新隐藏状态 $z_t$ 基于先前隐藏状态 $z_{t-1}$、先前生成的符号 $y_{t-1}$ 和计算出的上下文向量 $c_t$ 来计算。解码器也像编码器一样使用门控循环单元。

然后，公式 (2) 中下一个目标词的概率由下式计算：

$$p(y_t | y_{<t}, x) = \frac{1}{Z} \exp \{w_t^\top \phi (y_{t-1}, z_t, c_t) + b_t\} \qquad (6}$$

其中 $\phi$ 是一个仿射变换后接非线性激活函数，$w_t$ 和 $b_t$ 分别是目标词向量和目标词偏置。$Z$ 是归一化常数，由下式计算：

$$Z = \sum_{k: y_k \in V} \exp \{w_k^\top \phi (y_{t-1}, z_t, c_t) + b_k\}, \qquad (7}$$

其中 $V$ 是所有目标词的集合。

关于实现的详细描述，我们请读者参考 (Bahdanau et al., 2014) 的附录。

### 2.2 有限词表问题与常规解决方案

训练该神经机器翻译模型的主要困难之一在于计算目标词概率（公式 (6)）所涉及的计算复杂度。更具体地说，为了计算归一化常数（公式 (6) 中的分母），我们需要计算特征 $\phi(y_{t-1}, z_t, c_t)$ 与词向量 $w_t$ 之间的点积，次数等于目标词表中的词数。这必须对每个句子平均 20-30 个词进行，即使目标词数量适中，也容易变得过于昂贵。此外，内存需求随着目标词数量线性增长。与现有的非参数化方法（如基于短语的翻译系统）相比，这一直是神经机器翻译的主要障碍。

因此，最近提出的神经机器翻译模型使用包含 30,000 到 80,000 个最频繁词的 shortlist（Bahdanau et al., 2014; Sutskever et al., 2014）。这使得训练更可行，但带来了若干问题。首先，如果源句子的翻译需要许多不在 shortlist 中的词，模型性能会严重下降（Cho et al., 2014a）。这也影响系统的性能评估，通常以 BLEU 衡量。其次，第一个问题在具有丰富词汇的语言（如德语或其他高度屈折语）中变得更加严重。

针对大目标词表问题，有两种模型特定的方法。第一种方法是随机近似目标词概率。这种方法最近在 (Mnih and Kavukcuoglu, 2013; Mikolov et al., 2013) 中基于噪声对比估计（Gutmann and Hyvarinen, 2010）被提出。第二种方法是将目标词聚合成多个类别或层次化类别，并将目标概率 $p(y_t|y_{<t}, x)$ 分解为类别概率 $p(c_t|y_{<t}, x)$ 和类内词概率 $p(y_t|c_t, y_{<t}, x)$ 的乘积。这将所需的点积数量减少为类别数与类内词数之和。这些方法主要旨在降低训练期间的计算复杂度，但通常在测试时解码翻译时不会带来加速1。

除了这些模型特定的方法之外，还存在翻译特定的方法。翻译特定的方法利用了稀有目标词的属性。例如，Luong 等人在 (Luong et al., 2014) 中为神经机器翻译提出了这样一种方法。他们使用词对齐模型将源句和目标句子中的稀有词（不在 shortlist 中的词）替换为相应的 $\langle\text{OOV}_n\rangle$ 标记。一旦源句子被翻译，翻译中的每个 $\langle\text{OOV}_n\rangle$ 将根据由相应 $\langle\text{OOV}_n\rangle$ 标记的源词进行替换。

需要指出的是，模型特定的方法和翻译特定的方法通常是互补的，可以一起使用以进一步提高翻译性能并降低计算复杂度。

> 1 这是因为束搜索（beam search）无论输出概率如何参数化，都需要在每个时间步计算每个目标词的条件概率。

## 3 极大目标词表的近似学习方法

### 3.1 描述

在本文中，我们提出了一种模型特定的方法，允许我们训练具有极大目标词表的神经机器翻译模型。使用所提出的方法，训练的计算复杂度相对于目标词表大小变为常数。此外，所提出的方法允许我们高效地使用内存有限的快速计算设备（如GPU）来训练具有更大目标词表的神经机器翻译模型。

如前所述，训练神经机器翻译模型的计算低效源于公式 (6) 中的归一化常数。为了避免计算归一化常数的复杂度不断增长，我们在此提出在每次更新时仅使用目标词表的一小部分子集 $V'$。所提出的方法基于 (Bengio and Sénécal, 2008) 的早期工作。

让我们考虑公式 (6) 中输出的对数概率。梯度由正部和负部组成：

$$\nabla \log p(y_t | y_{<t}, x) = \nabla E(y_t) - \sum_{k: y_k \in V} p(y_k | y_{<t}, x) \nabla E(y_k), \qquad (8}$$

其中我们将能量 $E$ 定义为

$$E(y_j) = w_j^\top \phi (y_{j-1}, z_j, c_j) + b_j.$$

梯度的第二项（即负项）本质上是能量的期望梯度：

$$\mathbb{E}_{P} [\nabla E(y)], \qquad (9}$$

其中 $P$ 表示 $p(y | y_{<t}, x)$。

所提出方法的主要思想是通过使用少量样本的重要性采样来近似这个期望（即梯度的负项）。给定预定义的提议分布 $Q$ 和从 $Q$ 中采样的集合 $V'$，我们将公式 (9) 中的期望近似为：

$$\mathbb{E}_{P} [\nabla E(y)] \approx \sum_{k: y_k \in V'} \frac{\omega_k}{\sum_{k': y_{k'} \in V'} \omega_{k'}} \nabla E(y_k), \qquad (10}$$

其中

$$\omega_k = \exp \{E(y_k) - \log Q(y_k)\}. \qquad (11}$$

这种方法允许我们在训练期间仅使用目标词表的一小部分子集来计算归一化常数，从而每次参数更新的计算复杂度大大降低。直观地说，在每次参数更新时，我们只更新与正确词 $w_t$ 以及采样词 $V'$ 相关联的向量。

尽管所提出的方法自然地解决了计算复杂度问题，但天真地使用这种方法并不能保证每个句子对（包含多个目标词）更新的参数数量是有限的或可控的。当训练在例如内存有限的GPU上进行时，这就会成为问题。

因此，在实践中，我们划分训练语料库，并在训练之前为每个划分预定义一个目标词表子集 $V'$。在训练开始之前，我们依次检查训练语料库中的每个目标句子，并累积唯一目标词，直到唯一目标词的数量达到预定义的阈值 $\tau$。累积的词表将用于语料库的这个划分。我们重复此过程直到训练集的末尾。我们将用于第 $i$ 个划分的目标词子集记为 $V'_i$。

这可以理解为对训练语料库的每个划分都有一个单独的提议分布 $Q_i$。该分布 $Q_i$ 将相等的概率质量分配给子集 $V'_i$ 中包含的所有目标词，并将零概率质量分配给所有其他词，即：

$$Q_i(y_k) = \begin{cases}
\frac{1}{|V'_i|} & \text{if } y_t \in V'_i \\
0 & \text{otherwise}.
\end{cases}$$

这种提议分布的选择抵消了重要性权重 (10)-(11) 中来自 $-\log Q(y_k)$ 的修正项，这使得所提出的方法等价于将公式 (6) 中的精确输出概率近似为：

$$p(y_t | y_{<t}, x) \approx \frac{\exp \{w_t^\top \phi (y_{t-1}, z_t, c_t) + b_t\}}{\sum_{k: y_k \in V'} \exp \{w_k^\top \phi (y_{t-1}, z_t, c_t) + b_k\}}.$$

应当注意的是，这种 $Q$ 的选择使得估计量有偏。

所提出的过程相比通常的重要性采样带来了加速，因为它利用了现代计算机在矩阵-矩阵乘法相对于矩阵-向量乘法方面的优势。

#### 3.1.1 关于后果的非正式讨论

公式 (6) 中输出概率的参数化可以理解为安排与目标词相关联的向量，使得最可能（或正确）目标词的向量与当前隐藏状态之间的点积最大化。之后的指数运算和归一化只是将点积转换为适当概率的过程。

因此，随着学习的进行，所有可能目标词的向量倾向于彼此对齐，而不与其它词对齐。这是通过将正确词的向量沿 $\phi(y_{t-1}, z_t, c_t)$ 方向移动，同时将所有其他向量推开来精确实现的，这是在最大化公式 (6) 中精确输出概率的对数梯度时发生的情况。我们的近似方法则只移动正确词和仅一部分采样目标词（即包含在 $V'$ 中的词）的词向量。

### 3.2 解码

一旦使用所提出的近似方法训练好模型，我们可以在给定新源句子解码翻译时使用完整的目标词表。虽然这样做是有利的，因为它允许训练后的模型在生成翻译时利用整个词表，但这样做可能在计算上过于昂贵，例如在实时应用中。

由于训练将目标词向量放置在空间中，使得它们只有在可能成为正确词时才与解码器的隐藏状态良好对齐，因此我们可以在解码时仅使用一部分候选目标词。这与我们在训练时的做法类似，区别在于测试时我们无法获得一组正确的目标词。

选择候选目标词子集的最朴素方法是仅取前 $K$ 个最频繁的目标词，其中 $K$ 可以根据计算需求进行调整。然而，这实际上抵消了使用极大目标词表训练模型的全部目的。相反，我们可以使用现有的词对齐模型来对齐训练语料库中的源词和目标词，并构建一个词典。利用该词典，对于每个源句子，我们构建一个目标词集合，该集合由 $K$ 个最频繁的词（根据估计的 unigram 概率）和最多 $K'$ 个每个源词的可能目标词（根据词典）组成。$K$ 和 $K'$ 可以选择以满足计算需求或最大化开发集上的翻译性能。我们将以这些方式之一构建的子集称为候选列表（candidate list）。

### 3.3 未知词的源词处理

在实验中，我们使用名为 RNNsearch 的神经机器翻译模型（Bahdanau et al., 2014）（见第 2.1.1 节）来评估所提出的方法。在该模型中，作为解码过程的一部分，我们通过公式 (5) 中的对齐模型获得目标词与源位置之间的对齐。

我们可以利用这一特性来推断每个目标词最对齐的源词（由公式 (5) 中最大的 $\alpha_t$ 指示）。这在模型生成了 [UNK] 标记时尤其有用。一旦给定源句子生成了翻译，每个 [UNK] 都可以使用基于对齐源词的翻译特定技术进行替换。例如，在实验中，我们尝试将每个 [UNK] 标记替换为对齐的源词，或替换为由另一个词对齐模型确定的最可能的翻译。也可以使用诸如音译等其他技术来进一步提高性能（Koehn, 2010）。

## 4 实验

我们在英\rightarrow法和英\rightarrow德翻译任务上评估了所提出的方法。我们仅使用 WMT'14 提供的双语平行语料库来训练神经机器翻译模型。对于每个语言对，我们使用的数据集为：

- **英\rightarrow法**2: Europarl v7, Common Crawl, UN, News Commentary, Gigaword
- **英\rightarrow德**: Europarl v7, Common Crawl, News Commentary

为了确保公平比较，英\rightarrow法语料库包含约 1200 万个句子，与 (Kalchbrenner and Blunsom, 2013; Bahdanau et al., 2014; Sutskever et al., 2014) 中使用的语料库相同。对于英\rightarrow德语料库，我们按照类似于 (Peitz et al., 2014; Li et al., 2014) 的方式进行预处理，以移除许多翻译质量较差的句子。

我们在 WMT'14 测试集（news-test 2014）上评估模型3，同时使用 news-test-2012 和 news-test-2013 的拼接用于模型选择（开发集）。表 1 展示了目标端在不同词表大小下的数据覆盖率（%）。

表 1: 不同词表大小下目标端语料库的数据覆盖率（%）。"All" 指训练集中的所有标记。

| 词表大小 | 英-法 训练 | 英-法 测试 | 英-德 训练 | 英-德 测试 |
|:-------:|:---------:|:---------:|:---------:|:---------:|
| 15k | 93.5 | 90.8 | 88.5 | 83.8 |
| 30k | 96.0 | 94.6 | 91.8 | 87.9 |
| 50k | 97.3 | 96.3 | 93.7 | 90.4 |
| 500k | 99.5 | 99.3 | 98.4 | 96.1 |
| All | 100.0 | 99.6 | 100.0 | 97.3 |

除非另有说明，所有报告的 BLEU 分数（Papineni et al., 2002）均使用 multi-bleu.perl 脚本4对保留大小写的分词翻译进行计算。

> 2 预处理后的数据可以从 http://www-lium.univ-lemans.fr/~schwenk/nnmt-shared-task/README 查找并下载。
> 3 为了与之前的提交进行比较，我们使用过滤后的测试集。
> 4 https://github.com/moses-smt/mosesdecoder/blob/master/scripts/generic/multi-bleu.perl

### 4.1 设置

作为英\rightarrow法翻译的基线，我们使用 (Bahdanau et al., 2014) 提出的 RNNsearch 模型，具有 30,000 个源词和目标词5。对于英\rightarrow德翻译，我们训练另一个 RNNsearch 模型，具有 50,000 个源词和目标词。

对于每个语言对，我们使用所提出的方法训练另一组具有更大词表（500,000 个源词和目标词）的 RNNsearch 模型。我们将这些模型称为 RNNsearch-LV。我们改变训练期间使用的 shortlist 大小（第 3.1 节中的 $\tau$）。我们尝试了 15,000 和 30,000（英\rightarrow法），以及 15,000 和 50,000（英\rightarrow德）。我们随后报告开发集上最佳性能的结果，模型通常每十二小时评估一次。

对于两个语言对，我们还通过在每个 epoch 开始时重新打乱数据集来训练新模型，$\tau = 15,000$ 和 $\tau = 50,000$。虽然这会导致不可忽略的开销，但这种变化允许每个 epoch 中词与不同的其他词集合进行对比。

为了稳定除词嵌入之外的其他参数，在训练阶段结束时，我们冻结词嵌入，并在观察到开发集上达到最佳性能后再仅调优其他参数大约两天。这有助于提高开发集上的 BLEU 分数。

我们使用束搜索来生成给定源句子的翻译。在束搜索期间，我们保留一组 12 个假设，并按候选句子的长度对概率进行归一化，如 (Cho et al., 2014a) 中所述6。候选列表的选择旨在最大化开发集上的性能，其中 $K \in \{15k, 30k, 50k\}$，$K' \in \{10, 20\}$。如第 3.2 节所述，我们使用双语词典来加速解码并替换翻译中的未知词。双语词典使用 fast_align 构建（Dyer et al., 2013）。我们仅在词以小写字母开头时使用词典，否则直接复制源词。这导致了开发集上更好的性能。

> 5 (Bahdanau et al., 2014) 的作者允许我们访问他们训练好的模型。我们选择了验证集上最好的一个并继续训练。
> 6 这些实验细节与 (Bahdanau et al., 2014) 不同。

### 4.2 翻译性能

在表 2 中，我们展示了使用极大目标词表训练的模型所获得的结果，以及之前在 (Sutskever et al., 2014)、(Luong et al., 2014)、(Buck et al., 2014) 和 (Durrani et al., 2014) 中报告的结果。在不使用翻译特定策略的情况下，我们可以清晰地看到 RNNsearch-LV 优于基线 RNNsearch。

在英\rightarrow法任务中，即使没有任何翻译特定技术（第 3.2-3.3 节），RNNsearch-LV 也接近了先前最佳单模型神经机器翻译（NMT）的性能水平。然而，结合这些技术后，RNNsearch-LV 超越了它。RNNsearch-LV 的性能也优于标准的基于短语的翻译系统（Cho et al., 2014b）。此外，通过组合 8 个模型，我们能够实现与当时最先进技术可比的翻译性能（以 BLEU 衡量）。

对于英\rightarrow德任务，RNNsearch-LV 在未知词替换前优于基线，但进行替换后，两个系统的性能相近。我们通过重新打乱数据集可以达到更高的大词表单模型性能，但这一步也可能帮助基线。在这种情况下，通过构建 8 个模型的集成，我们能够超越该任务上先前报告的最佳翻译结果。

当 $\tau = 15,000$ 时，RNNsearch-LV 的性能略有下降，在不重新打乱的情况下，英\rightarrow法和英\rightarrow德的最佳 BLEU 分数分别为 33.76 和 18.59。

表 2: 不同模型在 (a) 英\rightarrow法和 (b) 英\rightarrow德翻译任务上获得的 BLEU 翻译性能。RNNsearch 是 (Bahdanau et al., 2014) 提出的模型，RNNsearch-LV 是使用本文提出的方法训练的 RNNsearch，Google 是 (Sutskever et al., 2014) 提出的基于 LSTM 的模型。除非另有说明，我们报告使用 $\tau = 30,000$（英\rightarrow法）和 $\tau = 50,000$（英\rightarrow德）的 RNNsearch-LV 单模型分数。对于我们自己运行的实验，我们在括号中也显示了开发集上的分数。($\star$) (Sutskever et al., 2014), ($\circ$) (Luong et al., 2014), ($\bullet$) (Durrani et al., 2014), ($*$) 标准 Moses 设置 (Cho et al., 2014b), ($\diamondsuit$) (Buck et al., 2014)。

**(a) 英\rightarrow法**

| 模型 | RNNsearch | RNNsearch-LV | Google | 基于短语的 SMT |
|:---|:---------:|:-----------:|:-----:|:------------:|
| 基础 NMT | 29.97 (26.58) | 32.68 (28.76) | 30.6$\star$ | 37.03$\bullet$ |
| +候选列表 | – | 33.36 (29.32) | – | |
| +UNK替换 | 33.08 (29.08) | 34.11 (29.98) | 33.1$\circ$ | |
| +重新打乱 ($\tau$=50k) | – | 34.60 (30.53) | – | |
| +集成 | – | 37.19 (31.98) | 37.5$\circ$ | |

**(b) 英\rightarrow德**

| 模型 | RNNsearch | RNNsearch-LV | 基于短语的 SMT |
|:---|:---------:|:-----------:|:------------:|
| 基础 NMT | 16.46 (17.13) | 16.95 (17.85) | 20.67$\diamondsuit$ |
| +候选列表 | – | 17.46 (18.00) | |
| +UNK替换 | 18.97 (19.16) | 18.89 (19.03) | |
| +重新打乱 | – | 19.40 (19.37) | |
| +集成 | – | 21.59 (21.06) | |

### 4.3 关于集成的说明

对于每个语言对，我们开始训练四个模型，从每个模型中收集对应于开发集上最佳和第二佳性能的两个检查点。我们从每个检查点继续训练，同时保持词嵌入固定，直到达到最佳开发性能，并将该点的模型作为集成中的一个单模型。这个过程最终得到总共八个模型，但由于大部分训练是共享的，集成的构成可能并非最优。这一观点得到以下事实的支持：在部分一起训练的模型之间观察到了更高的跨模型 BLEU 分数（Freitag et al., 2014）。

### 4.4 分析

#### 4.4.1 解码速度

在表 3 中，我们展示了不同模型解码的时间信息。显然，使用完整目标词表的 RNNsearch-LV 解码最慢。如果我们在解码每个翻译时使用候选列表，解码速度显著提高，并接近基线 RNNsearch。

表 3: 每词平均解码时间。这里的解码不包括参数加载和未知词替换。基线使用 30,000 个词。候选列表使用 $K = 30,000$ 和 $K' = 10$ 构建。($\star$) i7-4820K（单线程），($\circ$) GTX TITAN Black

| 模型 | CPU$\star$ | GPU$\circ$ |
|:---|:--------:|:--------:|
| RNNsearch | 0.02 s | 0.09 s |
| RNNsearch-LV | 0.25 s | 0.80 s |
| RNNsearch-LV + 候选列表 | 0.05 s | 0.12 s |

使用候选列表的一个潜在问题是，对于每个源句子，我们必须重新构建目标词表并随后替换部分参数，这可能很容易变得耗时。我们可以通过例如为多个源句子构建一个共同的候选列表来解决这个问题。通过这样做，我们能够匹配基线 RNNsearch 模型的解码速度。

#### 4.4.2 解码目标词表

对于英\rightarrow法（$\tau = 30,000$），我们通过使用 30,000 个常见词的固定集合与（最多）$K'$ 个根据词典得到的每个源词的可能候选词的并集，来评估翻译测试句子时目标词表的影响。结果如图 1 所示。当 $K' = 0$（未显示）时，系统的性能与不替换未知词时的基线相当（30.12），但进行替换时的改进不大（31.14）。由于大词表模型在训练期间不太预测 [UNK]，因此在解码时生成它的可能性也较小，这限制了后处理步骤在这一情况下的有效性。当 $K' = 1$ 时，限制了允许的非常见词的多样性，BLEU 不如适度更大的 $K'$，这表明我们的模型可以在一定程度上正确选择罕见的备选词。如果我们使用 $K = 50,000$（如我们基于验证性能所做的测试那样），与 $K' = 1$ 相比的改进约为 0.2 BLEU。

在验证 $K$ 的选择时，我们发现它与训练期间选择的 $\tau$ 相关。例如，在英\rightarrow法验证集上，当 $\tau = 15,000$（且 $K' = 10$）时，$K = 15,000$ 的 BLEU 分数为 29.44，但 $K = 30,000$ 和 $50,000$ 时分别下降到 29.19 和 28.84。对于 $\tau = 30,000$，分数从 $K = 15,000$ 到 $K = 50,000$ 适度增加。在英\rightarrow德和测试集上也观察到了类似的效果。由于我们实现的重要性采样没有对梯度进行通常的修正，测试词表与训练时使用的词表相似似乎是有益的。

图 1: 每个源词允许的词典条目数 $K'$ 对应的单模型测试 BLEU 分数（英\rightarrow法）。

![Figure 1](单模型测试BLEU分数随K变化.png)

*图 1: 每个源词允许的词典条目数 $K'$ 对应的单模型测试 BLEU 分数（英\rightarrow法）。带 UNK 替换与不带 UNK 替换两种情况。*

## 5 结论

在本文中，我们提出了一种扩展神经机器翻译目标词表大小的方法。所提出的方法允许我们在计算复杂度没有实质性增加的情况下训练具有更大目标词表的模型。它基于 (Bengio and Sénécal, 2008) 的早期工作，该工作使用重要性采样来降低神经语言模型中输出词概率归一化常数的计算复杂度。

在英\rightarrow法和英\rightarrow德翻译任务上，我们观察到使用所提出方法训练的神经机器翻译模型与仅使用有限目标词集合的模型表现相当，甚至更好，即使在替换未知词后也是如此。由于 RNNsearch-LV 模型在解码时仅使用目标词表的一个选定子集时性能有所提升，这使得所提出的学习算法更加实用。

以 BLEU 衡量，我们的模型在英\rightarrow法和英\rightarrow德任务上均展示出与当时最先进的翻译系统可比的翻译性能。在英\rightarrow法任务上，使用所提出方法训练的模型比 (Luong et al., 2014) 的最佳单模型神经机器翻译（NMT）高出约 1 个 BLEU 点。尽管集成构成多样性相对较低，多个模型集成的性能距离最佳系统（Luong et al., 2014）仅约 0.3 个 BLEU 点。在英\rightarrow德任务上，我们模型的最佳性能 21.59 BLEU 高于 (Buck et al., 2014) 中报告的此前最先进水平（20.67）。

## 致谢

作者感谢 Theano 的开发者（Bergstra et al., 2010; Bastien et al., 2012）。我们感谢以下机构在研究资助和计算支持方面的帮助：NSERC、Calcul Québec、Compute Canada、加拿大研究讲席计划（Canada Research Chairs）和 CIFAR。

## 参考文献

[1] Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. 2014. Neural machine translation by jointly learning to align and translate. Technical report, arXiv preprint arXiv:1409.0473.

[2] Frédéric Bastien, Pascal Lamblin, Razvan Pascanu, James Bergstra, Ian J. Goodfellow, Arnaud Bergeron, Nicolas Bouchard, and Yoshua Bengio. 2012. Theano: new features and speed improvements. Deep Learning and Unsupervised Feature Learning NIPS 2012 Workshop.

[3] Yoshua Bengio and Jean-Sébastien Sénécal. 2008. Adaptive importance sampling to accelerate training of a neural probabilistic language model. IEEE Trans. Neural Networks, 19(4):713–722.

[4] James Bergstra, Olivier Breuleux, Frédéric Bastien, Pascal Lamblin, Razvan Pascanu, Guillaume Desjardins, Joseph Turian, David Warde-Farley, and Yoshua Bengio. 2010. Theano: a CPU and GPU math expression compiler. In Proceedings of the Python for Scientific Computing Conference (SciPy), June. Oral Presentation.

[5] Christian Buck, Kenneth Heafield, and Bas van Ooyen. 2014. N-gram counts and language models from the common crawl. In Proceedings of the Language Resources and Evaluation Conference, Reykjavík, Iceland, May.

[6] Kyunghyun Cho, Bart van Merriënboer, Dzmitry Bahdanau, and Yoshua Bengio. 2014a. On the properties of neural machine translation: Encoder–Decoder approaches. In Eighth Workshop on Syntax, Semantics and Structure in Statistical Translation, October.

[7] Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. 2014b. Learning phrase representations using RNN encoder-decoder for statistical machine translation. In Proceedings of the Empirical Methods in Natural Language Processing (EMNLP 2014), October.

[8] Nadir Durrani, Barry Haddow, Philipp Koehn, and Kenneth Heafield. 2014. Edinburgh's phrase-based machine translation systems for WMT-14. In Proceedings of the Ninth Workshop on Statistical Machine Translation, pages 97–104. Association for Computational Linguistics Baltimore, MD, USA.

[9] Chris Dyer, Victor Chahuneau, and Noah A. Smith. 2013. A simple, fast, and effective reparameterization of IBM Model 2. In Proceedings of the 2013 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 644–648, Atlanta, Georgia, June. Association for Computational Linguistics.

[10] Mikel L. Forcada and Ramón P. ˜Neco. 1997. Recursive hetero-associative memories for translation. In José Mira, Roberto Moreno-Díaz, and Joan Cabestany, editors, Biological and Artificial Computation: From Neuroscience to Technology, volume 1240 of Lecture Notes in Computer Science, pages 453–462. Springer Berlin Heidelberg.

[11] Markus Freitag, Stephan Peitz, Joern Wuebker, Hermann Ney, Matthias Huck, Rico Sennrich, Nadir Durrani, Maria Nadejde, Philip Williams, Philipp Koehn, et al. 2014. Eu-bridge MT: Combined machine translation. ACL 2014, page 105.

[12] M. Gutmann and A. Hyvarinen. 2010. Noise-contrastive estimation: A new estimation principle for unnormalized statistical models. In Proceedings of The Thirteenth International Conference on Artificial Intelligence and Statistics (AISTATS'10).

[13] Nal Kalchbrenner and Phil Blunsom. 2013. Recurrent continuous translation models. In Proceedings of the ACL Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 1700–1709. Association for Computational Linguistics.

[14] Philipp Koehn, Franz Josef Och, and Daniel Marcu. 2003. Statistical phrase-based translation. In Proceedings of the 2003 Conference of the North American Chapter of the Association for Computational Linguistics on Human Language Technology - Volume 1, NAACL '03, pages 48–54.

[15] Philipp Koehn. 2010. Statistical Machine Translation. Cambridge University Press, New York, NY, USA, 1st edition.

[16] Liangyou Li, Xiaofeng Wu, Santiago Cortes Vaillo, Jun Xie, Andy Way, and Qun Liu. 2014. The DCU-ICTCAS MT system at WMT 2014 on German-English translation task. In Proceedings of the Ninth Workshop on Statistical Machine Translation, pages 136–141, Baltimore, Maryland, USA, June. Association for Computational Linguistics.

[17] Thang Luong, Ilya Sutskever, Quoc V Le, Oriol Vinyals, and Wojciech Zaremba. 2014. Addressing the rare word problem in neural machine translation. arXiv preprint arXiv:1410.8206.

[18] Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. 2013. Efficient estimation of word representations in vector space. In International Conference on Learning Representations: Workshops Track.

[19] Andriy Mnih and Koray Kavukcuoglu. 2013. Learning word embeddings efficiently with noise-contrastive estimation. In C.J.C. Burges, L. Bottou, M. Welling, Z. Ghahramani, and K.Q. Weinberger, editors, Advances in Neural Information Processing Systems 26, pages 2265–2273. Curran Associates, Inc.

[20] Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. 2002. BLEU: A method for automatic evaluation of machine translation. In Proceedings of the 40th Annual Meeting on Association for Computational Linguistics, ACL '02, pages 311–318, Stroudsburg, PA, USA. Association for Computational Linguistics.

[21] Stephan Peitz, Joern Wuebker, Markus Freitag, and Hermann Ney. 2014. The RWTH Aachen German-English machine translation system for WMT 2014. In Proceedings of the Ninth Workshop on Statistical Machine Translation, pages 157–162, Baltimore, Maryland, USA, June. Association for Computational Linguistics.

[22] Ilya Sutskever, Oriol Vinyals, and Quoc V. Le. 2014. Sequence to sequence learning with neural networks. In NIPS'2014.
