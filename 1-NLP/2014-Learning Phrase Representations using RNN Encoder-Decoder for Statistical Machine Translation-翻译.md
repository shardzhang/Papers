# 使用RNN编码器-解码器学习短语表示用于统计机器翻译

> Kyunghyun Cho, Bart van Merriënboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, Yoshua Bengio | Université de Montréal; Jacobs University; Université du Maine

本文提出RNN编码器-解码器（RNN Encoder-Decoder）模型，**由两个循环神经网络组成，将变长源序列编码为固定长度向量表示，再解码为目标序列，并通过新颖的门控隐藏单元自适应地记忆和遗忘信息**。

核心内容：

- 提出RNN编码器-解码器架构：编码器将变长序列映射到固定长度向量，解码器将向量映射回变长序列
- 提出新的门控隐藏单元（GRU），包含重置门和更新门，比LSTM更简单但同样有效
- 将RNN编码器-解码器用于为SMT系统中的短语对打分，作为对数线性模型的额外特征
- 模型学习的短语表示同时保留语义和句法结构

关键发现：

- 使用RNN编码器-解码器打分后，BLEU分数从基线**33.30**提升至**33.87**
- 结合CSLM（连续空间语言模型）后BLEU进一步提升至**34.64**
- 定性分析表明RNN编码器-解码器能更好地捕获短语表中的语言规律
- 学习的词和短语表示在语义和句法上都有意义，语义相似的词在嵌入空间中聚类

---

## 摘要

在本文中，我们提出了一种新颖的神经网络模型，称为RNN编码器-解码器，它由两个循环神经网络（RNN）组成。一个RNN将符号序列编码为固定长度的向量表示，另一个将该表示解码为另一个符号序列。所提出模型的编码器和解码器联合训练以最大化给定源序列的目标序列的条件概率。通过使用RNN编码器-解码器计算的短语对条件概率作为现有对数线性模型的额外特征，统计机器翻译系统的性能在经验上得到了提升。从定性角度看，我们表明所提出模型学习了语言短语在语义和句法上有意义的表示。

## 1 引言

深度神经网络在各种应用中显示出巨大成功，如目标识别[1]和语音识别[2]。此外，许多最近的工作表明神经网络可以成功用于自然语言处理（NLP）中的许多任务。这些包括但不限于语言建模[3]、复述检测[4]和词嵌入提取[5]。在统计机器翻译（SMT）领域，深度神经网络已开始显示出有希望的结果。[6]总结了前馈神经网络在基于短语的SMT系统框架中的成功使用。

沿着使用神经网络进行SMT的研究路线，本文专注于一种新颖的神经网络架构，可用作传统基于短语SMT系统的一部分。所提出的神经网络架构，我们称之为RNN编码器-解码器，由两个作为编码器和解码器对的循环神经网络（RNN）组成。编码器将变长源序列映射到固定长度向量，解码器将向量表示映射回变长目标序列。两个网络联合训练以最大化给定源序列的目标序列的条件概率。此外，我们提出使用一个相当复杂的隐藏单元，以提高记忆容量和训练便利性。

所提出的带有新颖隐藏单元的RNN编码器-解码器在英法翻译任务上进行了经验评估。我们训练模型学习英语短语到对应法语短语的翻译概率。然后将该模型用作标准基于短语SMT系统的一部分，为短语表中的每个短语对打分。经验评估表明，使用RNN编码器-解码器为短语对打分的方法提高了翻译性能。

我们通过比较RNN编码器-解码器的短语分数与现有翻译模型给出的分数来定性分析训练好的RNN编码器-解码器。定性分析表明RNN编码器-解码器在捕获短语表中的语言规律方面更好，间接解释了整体翻译性能的定量改进。对模型的进一步分析揭示，RNN编码器-解码器学习了短语的连续空间表示，保留了短语的语义和句法结构。

## 2 RNN编码器-解码器

### 2.1 预备知识：循环神经网络

循环神经网络（RNN）是一种由隐藏状态 $h$ 和可选输出 $y$ 组成的神经网络，在变长序列 $x = (x_1, \ldots, x_T)$ 上操作。在每个时间步 $t$，RNN的隐藏状态 $h^{\langle t \rangle}$ 通过以下方式更新：

$$
h^{\langle t \rangle} = f(h^{\langle t-1 \rangle}, x_t) \qquad (1)
$$

其中 $f$ 是非线性激活函数。$f$ 可以像逐元素logistic sigmoid函数一样简单，也可以像长短期记忆（LSTM）单元[8]一样复杂。

RNN可以通过训练来预测序列中的下一个符号，从而学习序列上的概率分布。在这种情况下，每个时间步 $t$ 的输出是条件分布 $p(x_t | x_{t-1}, \ldots, x_1)$。例如，可以使用softmax激活函数输出多项分布（1-of-$K$编码）：

$$
p(x_{t,j} = 1 | x_{t-1}, \ldots, x_1) = \frac{\exp(w_j h^{\langle t \rangle})}{\sum_{j'=1}^{K} \exp(w_{j'} h^{\langle t \rangle})} \qquad (2)
$$

对所有可能的符号 $j = 1, \ldots, K$，其中 $w_j$ 是权重矩阵 $W$ 的行。通过组合这些概率，我们可以使用以下公式计算序列 $x$ 的概率：

$$
p(x) = \prod_{t=1}^{T} p(x_t | x_{t-1}, \ldots, x_1) \qquad (3)
$$

从这个学习的分布中，通过在每个时间步迭代采样符号来采样新序列是很直接的。

### 2.2 RNN编码器-解码器

在本文中，我们提出了一种新颖的神经网络架构，它学习将变长序列编码为固定长度向量表示，并将给定的固定长度向量表示解码回变长序列。从概率角度来看，这种新模型是学习以另一个变长序列为条件的变长序列上的条件分布的通用方法，例如 $p(y_1, \ldots, y_{T'} | x_1, \ldots, x_T)$，其中应注意输入和输出序列长度 $T$ 和 $T'$ 可能不同。

编码器是一个RNN，它顺序读取输入序列 $x$ 的每个符号。当它读取每个符号时，RNN的隐藏状态根据公式(1)变化。在读取序列末尾（由序列结束符号标记）后，RNN的隐藏状态是整个输入序列的摘要 $c$。

所提出模型的解码器是另一个RNN，它被训练为通过给定隐藏状态 $h^{\langle t \rangle}$ 预测下一个符号 $y_t$ 来生成输出序列。然而，与第2.1节描述的RNN不同，$y_t$ 和 $h^{\langle t \rangle}$ 都以 $y_{t-1}$ 和输入序列的摘要 $c$ 为条件。因此，解码器在时间 $t$ 的隐藏状态通过以下方式计算：

$$
h^{\langle t \rangle} = f(h^{\langle t-1 \rangle}, y_{t-1}, c)
$$

类似地，下一个符号的条件分布为：

$$
P(y_t | y_{t-1}, y_{t-2}, \ldots, y_1, c) = g(h^{\langle t \rangle}, y_{t-1}, c)
$$

对于给定的激活函数 $f$ 和 $g$（后者必须产生有效的概率，例如使用softmax）。

所提出RNN编码器-解码器的两个组件联合训练以最大化条件对数似然：

$$
\max_\theta \frac{1}{N} \sum_{n=1}^{N} \log p_\theta(y_n | x_n) \qquad (4)
$$

其中 $\theta$ 是模型参数集，每个 $(x_n, y_n)$ 是来自训练集的（输入序列，输出序列）对。在我们的情况下，由于解码器的输出从输入开始是可微的，我们可以使用基于梯度的算法来估计模型参数。

一旦RNN编码器-解码器被训练，该模型可以两种方式使用。一种方式是使用模型给定输入序列生成目标序列。另一方面，该模型可用于为给定的输入和输出序列对打分，其中分数简单地是来自公式(3)和(4)的概率 $p_\theta(y | x)$。

### 2.3 自适应记忆和遗忘的隐藏单元

除了新颖的模型架构外，我们还提出了一种新型隐藏单元（公式(1)中的 $f$），它受LSTM单元启发但计算和实现更简单。图2显示了所提出隐藏单元的图形描述。

让我们描述第 $j$ 个隐藏单元的激活是如何计算的。首先，重置门 $r_j$ 通过以下方式计算：

$$
r_j = \sigma\left([W_r x]_j + [U_r h^{\langle t-1 \rangle}]_j\right) \qquad (5)
$$

其中 $\sigma$ 是logistic sigmoid函数，$[.]_j$ 表示向量的第 $j$ 个元素。$x$ 和 $h^{t-1}$ 分别是输入和前一隐藏状态。$W_r$ 和 $U_r$ 是学习的权重矩阵。

类似地，更新门 $z_j$ 通过以下方式计算：

$$
z_j = \sigma\left([W_z x]_j + [U_z h^{\langle t-1 \rangle}]_j\right) \qquad (6)
$$

所提出单元 $h_j$ 的实际激活通过以下方式计算：

$$
h^{\langle t \rangle}_j = z_j h^{\langle t-1 \rangle}_j + (1 - z_j) \tilde{h}^{\langle t \rangle}_j \qquad (7)
$$

其中：

$$
\tilde{h}^{\langle t \rangle}_j = \phi\left([W x]_j + [U(r \odot h^{\langle t-1 \rangle})]_j\right) \qquad (8)
$$

在这种表述中，当重置门接近0时，隐藏状态被迫忽略前一隐藏状态并仅用当前输入重置。这有效地允许隐藏状态丢弃任何发现与未来无关的信息，从而允许更紧凑的表示。

另一方面，更新门控制从前一隐藏状态携带到当前隐藏状态的信息量。这类似于LSTM网络中的记忆单元，帮助RNN记住长期信息。此外，这可以被认为是泄漏积分单元[9]的自适应变体。

由于每个隐藏单元有独立的重置门和更新门，每个隐藏单元将学习捕获不同时间尺度上的依赖关系。那些学习捕获短期依赖的单元倾向于有频繁激活的重置门，而那些捕获较长依赖的单元将有主要激活的更新门。

在我们的初步实验中，我们发现使用带有门控单元的这种新单元至关重要。我们无法使用没有门控的常用tanh单元获得有意义的结果。

## 3 统计机器翻译

在常用的统计机器翻译系统（SMT）中，系统（具体来说是解码器）的目标是找到给定源句子 $e$ 的翻译 $f$，最大化：

$$
p(f | e) \propto p(e | f) p(f)
$$

其中右边第一项称为翻译模型，后者称为语言模型[10]。然而在实践中，大多数SMT系统将 $\log p(f | e)$ 建模为具有额外特征和相应权重的对数线性模型：

$$
\log p(f | e) = \sum_{n=1}^{N} w_n f_n(f, e) + \log Z(e) \qquad (9)
$$

其中 $f_n$ 和 $w_n$ 分别是第 $n$ 个特征和权重。$Z(e)$ 是不依赖于权重的归一化常数。权重通常被优化以在开发集上最大化BLEU分数。

### 3.1 使用RNN编码器-解码器为短语对打分

这里我们提议在短语对表上训练RNN编码器-解码器（见第2.2节），并在调优SMT解码器时使用其分数作为公式(9)中对数线性模型的额外特征。

当我们训练RNN编码器-解码器时，我们忽略原始语料库中每个短语对的（归一化）频率。采取此措施是为了(1)减少根据归一化频率从大短语表中随机选择短语对的计算开销，以及(2)确保RNN编码器-解码器不会简单地学习根据短语对的出现次数来排名。

这种选择的一个潜在原因是短语表中现有的翻译概率已经反映了短语对在原始语料库中的频率。通过RNN编码器-解码器的固定容量，我们试图确保模型的大部分容量专注于学习语言规律，即区分合理和不合理的翻译，或学习合理翻译的"流形"（概率集中区域）。

一旦RNN编码器-解码器被训练，我们为现有短语表中的每个短语对添加新分数。这允许新分数以最小的额外计算开销进入现有的调优算法。

### 3.2 相关方法：机器翻译中的神经网络

在展示经验结果之前，我们讨论了最近一些提议在SMT背景下使用神经网络的工作。

[6]提出了类似的短语对打分方法。他使用前馈神经网络，具有固定大小输入（他的情况下为7个词，对较短短语进行零填充）和固定大小输出（目标语言中的7个词）。当专门用于为SMT系统打分短语时，最大短语长度通常选择较小。然而，随着短语长度增加或我们将神经网络应用于其他变长序列数据，神经网络能够处理变长输入和输出非常重要。所提出的RNN编码器-解码器非常适合这些应用。

与[6]类似，[11]提出使用前馈神经网络建模翻译模型，但是一次预测目标短语中的一个词。他们报告了令人印象深刻的改进，但他们的方法仍然需要预先固定输入短语（或上下文词）的最大长度。

虽然[12]训练的不完全是神经网络，但该文作者提出学习词/短语的双语嵌入。他们使用学习的嵌入来计算短语对之间的距离，该距离用作SMT系统中短语对的额外分数。

所提出RNN编码器-解码器与[12]和[13]方法的一个重要区别是源和目标短语中词的顺序被考虑在内。RNN编码器-解码器自然区分具有相同词但顺序不同的序列，而上述方法实际上忽略了顺序信息。

与所提出RNN编码器-解码器最接近的方法是[14]中提出的循环连续翻译模型（模型2）。在他们的论文中，他们提出了一个类似的由编码器和解码器组成的模型。与我们模型的区别是他们使用卷积n-gram模型（CGM）作为编码器，使用逆CGM和循环神经网络的混合作为解码器。然而，他们仅在重新排序传统SMT系统提出的n-best列表和计算黄金标准翻译的困惑度方面评估了他们的模型。

## 4 实验

我们在WMT'14研讨会的英法翻译任务上评估我们的方法。

### 4.1 数据和基线系统

在WMT'14翻译任务框架下，有大量资源可用于构建英法SMT系统。双语语料库包括Europarl（6100万词）、新闻评论（550万）、联合国（4.21亿）以及两个分别为9000万和7.8亿词的爬取语料库。后两个语料库相当嘈杂。为了训练法语语言模型，除了双语文本的目标端外，还有约7.12亿词的爬取报纸材料。

人们普遍认为，在所有这些数据的连接上训练统计模型不一定导致最佳性能，并且产生难以处理的极大模型。相反，应该专注于给定任务的最相关数据子集。我们通过应用[15]中提出的数据选择方法及其对双语文本的扩展[16]来实现这一点。通过这些手段，我们从超过20亿词中选择了4.18亿词的子集用于语言建模，从8.5亿词中选择了3.48亿词的子集用于训练RNN编码器-解码器。

我们使用newstest2012和2013测试集进行数据选择和MERT权重调优，使用newstest2014作为我们的测试集。每个集合有超过7万词和单个参考翻译。

为了训练神经网络，包括所提出的RNN编码器-解码器，我们将源和目标词汇表限制为英语和法语最常见的15,000个词。这覆盖了约93%的数据集。所有词表外词被映射到特殊标记（[UNK]）。

基线基于短语的SMT系统使用默认设置的Moses构建。该系统在开发集和测试集上分别达到30.64和33.3的BLEU分数（见表1）。

### 4.1.1 RNN编码器-解码器

实验中使用的RNN编码器-解码器在编码器和解码器处有1000个带有提议门控的隐藏单元。每个输入符号 $x^{\langle t \rangle}$ 和隐藏单元之间的输入矩阵用两个低秩矩阵近似，输出矩阵同样近似。我们使用秩-100矩阵，等价于为每个词学习100维的嵌入。公式(8)中用于 $\tilde{h}$ 的激活函数是双曲正切函数。从解码器中的隐藏状态到输出的计算实现为具有单个中间层的深度神经网络[17]，该层有500个maxout单元，每个池化2个输入[18]。

RNN编码器-解码器中的所有权重参数通过从各向同性零均值（白）高斯分布采样初始化，标准差固定为0.01，循环权重参数除外。对于循环权重矩阵，我们首先从白高斯分布采样并使用其左奇异向量矩阵[19]。

我们使用Adadelta和随机梯度下降训练RNN编码器-解码器，超参数 $\epsilon = 10^{-6}$，$\rho = 0.95$[20]。每次更新时，我们使用从短语表（由3.48亿词创建）中随机选择的64个短语对。模型训练了大约三天。

### 4.1.2 神经语言模型

为了评估使用所提出RNN编码器-解码器为短语对打分的有效性，我们还尝试了更传统的方法，使用神经网络学习目标语言模型（CSLM）[21]。特别是，使用CSLM的SMT系统与使用所提出的RNN编码器-解码器短语打分方法的SMT系统之间的比较将阐明SMT系统不同部分中多个神经网络的贡献是累加的还是冗余的。

我们在目标语料库的7-gram上训练CSLM模型。每个输入词被投影到嵌入空间 $\mathbb{R}^{512}$，并拼接形成3072维向量。拼接向量通过两个修正层（大小为1536和1024）[22]。输出层是简单的softmax层（见公式(2)）。所有权重参数在-0.01和0.01之间均匀初始化，模型训练直到验证困惑度在10个epoch内不再改善。训练后，语言模型达到45.80的困惑度。

### 4.2 定量分析

我们尝试了以下组合：

**表1：使用不同方法组合在开发集和测试集上计算的BLEU分数。WP表示词惩罚，我们惩罚对神经网络未知的词数量。**

| 模型 | 开发集 | 测试集 |
|------|--------|--------|
| 基线 | 30.64 | 33.30 |
| RNN | 31.20 | 33.87 |
| CSLM + RNN | 31.48 | 34.64 |
| CSLM + RNN + WP | 31.50 | 34.54 |

结果如表1所示。正如预期的那样，添加由神经网络计算的特征始终提高了基线性能。

当同时使用CSLM和RNN编码器-解码器的短语分数时，达到了最佳性能。这表明CSLM和RNN编码器-解码器的贡献不太相关，可以通过独立改进每种方法来获得更好的结果。

### 4.3 定性分析

为了理解性能改进的来源，我们将RNN编码器-解码器计算的短语对分数与翻译模型中相应的 $p(f|e)$ 进行比较。由于现有翻译模型仅依赖于语料库中短语对的统计，我们期望其分数对频繁短语估计得更好，但对稀有短语估计得较差。此外，正如我们在第3.1节中提到的，我们进一步期望在没有任何频率信息的情况下训练的RNN编码器-解码器更基于语言规律而非基于其在语料库中的出现统计来为短语对打分。

我们关注那些源短语较长（每个源短语超过3个词）且频繁的短语对。对于每个这样的源短语，我们查看由翻译概率 $p(f|e)$ 或RNN编码器-解码器打分较高的目标短语。类似地，我们对那些源短语较长但在语料库中稀有的短语对执行相同的过程。

表2列出了每个源短语中翻译模型或RNN编码器-解码器偏好的前3个目标短语。源短语是从超过4或5个词的较长短语中随机选择的。

在大多数情况下，RNN编码器-解码器选择的目标短语更接近实际或字面翻译。我们可以观察到RNN编码器-解码器通常更喜欢较短的短语。

有趣的是，许多短语对被翻译模型和RNN编码器-解码器相似地打分，但也有同样多的其他短语对被打分截然不同（见图3）。这可能源于在唯一短语对集上训练RNN编码器-解码器的方法，阻止了RNN编码器-解码器简单地学习语料库中短语对的频率，如前面所解释的。

### 4.4 词和短语表示

由于所提出的RNN编码器-解码器不是专门为机器翻译任务设计的，这里我们简要查看训练模型的性质。

已经知道一段时间了，使用神经网络的连续空间语言模型能够学习语义上有意义的嵌入[3, 5]。由于所提出的RNN编码器-解码器也将词序列投影到连续空间向量并映射回来，我们期望看到所提出模型的类似性质。

图4中的左图显示了使用RNN编码器-解码器学习的词嵌入矩阵的词的2D嵌入。投影由最近提出的Barnes-Hut-SNE[23]完成。我们可以清楚地看到语义相似的词彼此聚类（见图4中的放大图）。

所提出RNN编码器-解码器自然生成短语的连续空间表示。在这种情况下，表示（图1中的 $c$）是1000维向量。与词表示类似，我们在图5中使用Barnes-Hut-SNE可视化了由四个或更多词组成的短语的表示。

从可视化中可以清楚地看出，RNN编码器-解码器捕获了短语的语义和句法结构。例如，在左下图中，大多数短语是关于时间持续的，而那些句法相似的短语聚类在一起。右下图显示了语义相似（国家或地区）的短语簇。另一方面，右上图显示了句法相似的短语。

## 5 结论

在本文中，我们提出了一种新的神经网络架构，称为RNN编码器-解码器，它能够学习从任意长度的序列到另一个可能来自不同集合的任意长度序列的映射。所提出的RNN编码器-解码器能够为序列对打分（以条件概率的形式）或给定源序列生成目标序列。伴随着新架构，我们提出了一种新颖的隐藏单元，包括重置门和更新门，自适应地控制每个隐藏单元在读取/生成序列时记住或遗忘多少。

我们用统计机器翻译任务评估了所提出模型，其中我们使用RNN编码器-解码器为短语表中的每个短语对打分。从定性角度看，我们能够表明新模型能够很好地捕获短语对中的语言规律，并且RNN编码器-解码器能够提出形式良好的目标短语。

RNN编码器-解码器的分数被发现以BLEU分数的形式提高了整体翻译性能。此外，我们发现RNN编码器-解码器的贡献与在SMT系统中使用神经网络的现有方法相当正交，因此我们可以通过同时使用RNN编码器-解码器和神经网络语言模型来进一步提高性能。

我们对训练模型的定性分析表明它确实在多个层面捕获了语言规律，即在词层面和短语层面。这表明可能有更多的自然语言相关应用可以从所提出的RNN编码器-解码器中受益。

所提出的架构有很大的进一步改进和分析的潜力。这里未研究的一种方法是通过让RNN编码器-解码器提出目标短语来替换整个或部分短语表。此外，注意到所提出模型不限于与书面语言一起使用，将所提出的架构应用于其他应用（如语音转录）将是重要的未来研究。

## 致谢

KC、BM、CG、DB和YB感谢NSERC、Calcul Québec、Compute Canada、Canada Research Chairs和CIFAR。FB和HS部分由欧盟委员会在MateCat项目下资助，由DARPA在BOLT项目下资助。

## 参考文献

[1] Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton. 2012. ImageNet classification with deep convolutional neural networks. In Advances in Neural Information Processing Systems 25 (NIPS'2012).

[2] George E. Dahl, Dong Yu, Li Deng, and Alex Acero. 2012. Context-dependent pretrained deep neural networks for large vocabulary speech recognition. IEEE Transactions on Audio, Speech, and Language Processing, 20(1):33–42.

[3] Yoshua Bengio, Réjean Ducharme, Pascal Vincent, and Christian Janvin. 2003. A neural probabilistic language model. J. Mach. Learn. Res., 3:1137–1155, March.

[4] Richard Socher, Eric H. Huang, Jeffrey Pennington, Andrew Y. Ng, and Christopher D. Manning. 2011. Dynamic pooling and unfolding recursive autoencoders for paraphrase detection. In Advances in Neural Information Processing Systems 24.

[5] Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, and Jeff Dean. 2013. Distributed representations of words and phrases and their compositionality. In Advances in Neural Information Processing Systems 26, pages 3111–3119.

[6] Holger Schwenk. 2012. Continuous space translation models for phrase-based statistical machine translation. In Martin Kay and Christian Boitet, editors, Proceedings of the 24th International Conference on Computational Linguistics (COLIN), pages 1071–1080.

[7] Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. 2014. Learning phrase representations using rnn encoder-decoder for statistical machine translation. In Proceedings of the Empirical Methods in Natural Language Processing (EMNLP 2014), October.

[8] S. Hochreiter and J. Schmidhuber. 1997. Long short-term memory. Neural Computation, 9(8):1735–1780.

[9] Y. Bengio, N. Boulanger-Lewandowski, and R. Pascanu. 2013. Advances in optimizing recurrent networks. In Proceedings of the 38th International Conference on Acoustics, Speech, and Signal Processing (ICASSP 2013), May.

[10] P. Koehn. 2005. Europarl: A parallel corpus for statistical machine translation. In Machine Translation Summit X, pages 79–86, Phuket, Thailand.

[11] Jacob Devlin, Rabih Zbib, Zhongqiang Huang, Thomas Lamar, Richard Schwartz, and John Makhoul. 2014. Fast and robust neural network joint models for statistical machine translation. In Proceedings of the ACL 2014 Conference, ACL '14, pages 1370–1380.

[12] Will Y. Zou, Richard Socher, Daniel M. Cer, and Christopher D. Manning. 2013. Bilingual word embeddings for phrase-based machine translation. In Proceedings of the ACL Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 1393–1398.

[13] Sarath Chandar, Stanislas Lauly, Hugo Larochelle, Mitesh Khapra, Balaraman Ravindran, Vikas Raykar, and Amrita Saha. 2014. An autoencoder approach to learning bilingual word representations. arXiv:1402.1454 [cs.CL], February.

[14] Nal Kalchbrenner and Phil Blunsom. 2013. Two recurrent continuous translation models. In Proceedings of the ACL Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 1700–1709.

[15] Robert C. Moore and William Lewis. 2010. Intelligent selection of language model training data. In Proceedings of the ACL 2010 Conference Short Papers, ACLShort '10, pages 220–224, Stroudsburg, PA, USA.

[16] Amittai Axelrod, Xiaodong He, and Jianfeng Gao. 2011. Domain adaptation via pseudo in-domain data selection. In Proceedings of the ACL Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 355–362.

[17] R. Pascanu, C. Gulcehre, K. Cho, and Y. Bengio. 2014. How to construct deep recurrent neural networks. In Proceedings of the Second International Conference on Learning Representations (ICLR 2014), April.

[18] Ian J. Goodfellow, David Warde-Farley, Mehdi Mirza, Aaron Courville, and Yoshua Bengio. 2013. Maxout networks. In ICML'2013.

[19] Andrew M. Saxe, James L. McClelland, and Surya Ganguli. 2014. Exact solutions to the nonlinear dynamics of learning in deep linear neural networks. In Proceedings of the Second International Conference on Learning Representations (ICLR 2014), April.

[20] Matthew D. Zeiler. 2012. ADADELTA: an adaptive learning rate method. Technical report, arXiv 1212.5701.

[21] Holger Schwenk. 2007. Continuous space language models. Comput. Speech Lang., 21(3):492–518, July.

[22] X. Glorot, A. Bordes, and Y. Bengio. 2011. Deep sparse rectifier neural networks. In AISTATS'2011.

[23] Laurens van der Maaten. 2013. Barnes-hut-sne. In Proceedings of the First International Conference on Learning Representations (ICLR 2013), May.

[24] Philipp Koehn, Franz Josef Och, and Daniel Marcu. 2003. Statistical phrase-based translation. In Proceedings of the 2003 Conference of the North American Chapter of the Association for Computational Linguistics on Human Language Technology - Volume 1, NAACL '03, pages 48–54.

[25] Daniel Marcu and William Wong. 2002. A phrase-based, joint probability model for statistical machine translation. In Proceedings of the ACL-02 Conference on Empirical Methods in Natural Language Processing - Volume 10, EMNLP '02, pages 133–139.

[26] Holger Schwenk, Marta R. Costa-Jussà, and José A. R. Fonollosa. 2006. Continuous space language models for the iwslt 2006 task. In IWSLT, pages 166–173.

[27] Le Hai Son, Alexandre Allauzen, and François Yvon. 2012. Continuous space translation models with neural networks. In Proceedings of the 2012 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL HLT '12, pages 39–48, Stroudsburg, PA, USA.

[28] Jianfeng Gao, Xiaodong He, Wen tau Yih, and Li Deng. 2013. Learning semantic representations for the phrase translation model. Technical report, Microsoft Research.

[29] Ashish Vaswani, Yinggong Zhao, Victoria Fossum, and David Chiang. 2013. Decoding with large-scale neural language models improves translation. Proceedings of the Conference on Empirical Methods in Natural Language Processing, pages 1387–1392.

[30] Michael Auli, Michel Galley, Chris Quirk, and Geoffrey Zweig. 2013. Joint language and translation modeling with recurrent neural networks. In Proceedings of the ACL Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 1044–1054.

[31] Alex Graves. 2012. Supervised Sequence Labelling with Recurrent Neural Networks. Studies in Computational Intelligence. Springer.

[32] Frédéric Bastien, Pascal Lamblin, Razvan Pascanu, James Bergstra, Ian J. Goodfellow, Arnaud Bergeron, Nicolas Bouchard, and Yoshua Bengio. 2012. Theano: new features and speed improvements. Deep Learning and Unsupervised Feature Learning NIPS 2012 Workshop.

[33] James Bergstra, Olivier Breuleux, Frédéric Bastien, Pascal Lamblin, Razvan Pascanu, Guillaume Desjardins, Joseph Turian, David Warde-Farley, and Yoshua Bengio. 2010. Theano: a CPU and GPU math expression compiler. In Proceedings of the Python for Scientific Computing Conference (SciPy), June.
