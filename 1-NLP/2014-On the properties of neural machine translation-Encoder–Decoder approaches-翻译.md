# 神经机器翻译的性质：编码器-解码器方法

> Kyunghyun Cho, Bart van Merriënboer, Dzmitry Bahdanau, Yoshua Bengio | Université de Montréal; Jacobs University



本文分析了神经机器翻译（NMT）模型的性质，**通过RNN编码器-解码器和新提出的门控递归卷积神经网络（grConv），揭示NMT在短句上表现良好但随 句子长度 和 未知词数量 增加而性能下降的关键特性**。

核心内容：

- 神经机器翻译完全基于神经网络，仅需约500MB内存，远小于传统SMT系统的数十GB
- 提出门控递归卷积神经网络（grConv），通过门控机制自适应地学习源语言的句法结构
- 在英法翻译任务上系统分析句子长度、未知词数量对翻译质量的影响
- 验证NMT与传统SMT系统结合可进一步提升翻译性能

关键发现：

- NMT在短句（10-20词）上BLEU达到**27.03**（RNNenc），接近Moses的**35.40**
- 随句子长度增加，NMT性能显著下降，而传统SMT在长句上表现更优
- 未知词数量增加导致NMT性能快速退化，词汇表大小是关键瓶颈
- grConv无需监督即可学习输入句子的语法结构，如自动识别"of the United States"等短语

---



## 摘要

神经机器翻译是一种相对较新的统计机器翻译方法，完全基于神经网络。神经机器翻译模型通常由一个编码器和一个解码器组成。编码器从变长输入句子中提取固定长度的表示，解码器从该表示生成正确的翻译。在本文中，我们专注于使用两个模型分析神经机器翻译的性质：RNN编码器-解码器和新提出的门控递归卷积神经网络。我们表明，神经机器翻译在没有未知词的短句上表现相对较好，但随着句子长度和未知词数量的增加，其性能迅速下降。此外，我们发现所提出的门控递归卷积神经网络能够自动学习句子的语法结构。

## 1 引言

最近提出了一种完全基于神经网络的统计机器翻译新方法[1, 2]。这种新方法，我们称之为神经机器翻译，受到深度表示学习最新趋势的启发。[1, 2, 3]中使用的所有神经网络模型都由一个编码器和一个解码器组成。编码器从变长输入句子中提取固定长度的向量表示，解码器从该表示生成正确的、变长的目标翻译。

神经机器翻译的出现在实践和理论上都具有重要意义。神经机器翻译模型所需的内存仅为传统统计机器翻译（SMT）模型的一小部分。我们为本文训练的模型总共仅需500MB内存。这与现有SMT系统形成鲜明对比，后者通常需要数十GB内存。这使得神经机器翻译在实践中具有吸引力。此外，与传统翻译系统不同，神经翻译模型的每个组件都是联合训练以最大化翻译性能的。

由于这种方法相对较新，关于分析这些模型的性质和行为的工作还不多。例如：这种方法在哪些性质的句子上表现更好？源/目标词汇表的选择如何影响性能？在哪些情况下神经机器翻译会失败？

理解这种新的神经机器翻译方法的性质和行为对于确定未来研究方向至关重要。此外，理解神经机器翻译的弱点和优势可能会导致更好的SMT和神经机器翻译系统集成方式。

在本文中，我们分析了两个神经机器翻译模型。一个是最近在[3]中提出的RNN编码器-解码器。另一个模型用一种新颖的神经网络（我们称之为门控递归卷积神经网络，grConv）替代了RNN编码器-解码器模型中的编码器。我们在法语到英语的翻译任务上评估这两个模型。

我们的分析表明，随着源句子长度的增加，神经机器翻译模型的性能迅速下降。此外，我们发现词汇表大小对翻译性能有很大影响。然而，从定性角度看，我们发现两个模型在大多数情况下都能生成正确的翻译。此外，新提出的grConv模型能够在没有监督的情况下学习源语言的某种句法结构。

## 2 处理变长序列的神经网络

在本节中，我们描述两种能够处理变长序列的神经网络。这些是循环神经网络和所提出的门控递归卷积神经网络。

### 2.1 带门控隐藏神经元的循环神经网络

循环神经网络（RNN，图1(a)）通过随时间维护隐藏状态 $h$ 来处理变长序列 $x = (x_1, x_2, \cdots, x_T)$。在每个时间步 $t$，隐藏状态 $h^{(t)}$ 通过以下方式更新：

$$
h^{(t)} = f(h^{(t-1)}, x_t)
$$

其中 $f$ 是激活函数。通常 $f$ 很简单，对输入向量执行线性变换、求和，然后应用逐元素logistic sigmoid函数。

RNN可以通过学习下一个输入的分布 $p(x_{t+1} | x_t, \cdots, x_1)$ 来有效地学习变长序列上的分布。例如，在1-of-$K$向量的情况下，该分布可以由输出为以下形式的RNN学习：

$$
p(x_{t,j} = 1 | x_{t-1}, \ldots, x_1) = \frac{\exp(w_j h^{\langle t \rangle})}{\sum_{j'=1}^{K} \exp(w_{j'} h^{\langle t \rangle})}
$$

对所有可能的符号 $j = 1, \ldots, K$，其中 $w_j$ 是权重矩阵 $W$ 的行。这产生联合分布：

$$
p(x) = \prod_{t=1}^{T} p(x_t | x_{t-1}, \ldots, x_1)
$$

最近，在[3]中为RNN提出了一种新的激活函数。新的激活函数用两个称为重置门 $r$ 和更新门 $z$ 的门控单元增强了通常的logistic sigmoid激活函数。每个门依赖于前一隐藏状态 $h^{(t-1)}$，当前输入 $x_t$ 控制信息流。这让人想起长短期记忆（LSTM）单元[5]。关于此单元的详细信息，我们请读者参阅[3]和图1(b)。在本文的其余部分，我们始终使用这种新的激活函数。

### 2.2 门控递归卷积神经网络

除了RNN，处理变长序列的另一种自然方法是使用递归卷积神经网络，其中每一层的参数在整个网络中共享（见图2(a)）。在本节中，我们介绍一个二元卷积神经网络，其权重被递归地应用于输入序列，直到输出单个固定长度向量。除了通常的卷积架构外，我们提议使用前面提到的门控机制，这允许递归网络动态学习源句子的结构。

令 $x = (x_1, x_2, \cdots, x_T)$ 为输入序列，其中 $x_t \in \mathbb{R}^d$。所提出的门控递归卷积神经网络（grConv）由四个权重矩阵 $W_l, W_r, G_l$ 和 $G_r$ 组成。在每个递归层 $t \in [1, T-1]$，第 $j$ 个隐藏单元 $h_j^{(t)}$ 的激活通过以下方式计算：

$$
h_j^{(t)} = \omega_c \tilde{h}_j^{(t)} + \omega_l h_{j-1}^{(t-1)} + \omega_r h_j^{(t-1)} \qquad (1)
$$

其中 $\omega_c, \omega_l$ 和 $\omega_r$ 是和为1的门控值。隐藏单元初始化为：

$$
h_j^{(0)} = U x_j
$$

其中 $U$ 将输入投影到隐藏空间。新激活 $\tilde{h}_j^{(t)}$ 按常规计算：

$$
\tilde{h}_j^{(t)} = \phi(W_l h_j^{(t-1)} + W_r h_j^{(t)})
$$

其中 $\phi$ 是逐元素非线性函数。门控系数 $\omega$ 通过以下方式计算：

$$
\begin{bmatrix} \omega_c \\ \omega_l \\ \omega_r \end{bmatrix} = \frac{1}{Z} \exp\left(G_l h_{j-1}^{(t-1)} + G_r h_j^{(t-1)}\right)
$$

其中 $G_l, G_r \in \mathbb{R}^{3 \times d}$，$Z$ 是归一化常数。

根据这种激活，可以将递归层 $t$ 中单个节点的激活视为三种选择之一：从左右子节点计算的新激活、来自左子节点的激活，或来自右子节点的激活。这种选择允许递归卷积的整体结构相对于输入样本自适应地改变。见图2(b)的说明。

在这方面，我们甚至可以将所提出的grConv视为进行一种无监督解析。如果我们考虑门控单元做出硬决策的情况，即 $\omega$ 遵循1-of-$K$编码，很容易看出网络适应输入并形成树状结构（见图2(c-d)）。然而，我们将该模型所学结构的进一步研究留待未来。

## 3 纯神经机器翻译

### 3.1 编码器-解码器方法

翻译任务可以从机器学习的角度理解为学习目标句子（翻译）$f$ 给定源句子 $e$ 的条件分布 $p(f|e)$。一旦模型学习了条件分布，就可以使用模型直接采样目标句子给定源句子，无论是通过实际采样还是使用（近似）搜索算法找到分布的最大值。

最近的一些论文提出使用神经网络直接从双语平行语料库学习条件分布[1, 2, 3]。例如，[1]的作者提出了一种涉及卷积n-gram模型的方法来提取源句子的固定长度向量，然后用增强了RNN的逆卷积n-gram模型解码。在[2]中，使用带有LSTM单元的RNN来编码源句子，并从最后一个隐藏状态开始解码目标句子。类似地，[3]的作者提出使用RNN来编码和解码源和目标短语对。

所有这些最新工作的核心是一个编码器-解码器架构（见图3）。编码器处理变长输入（源句子）并构建固定长度的向量表示（在图3中表示为 $z$）。基于编码的表示，解码器生成变长序列（目标句子）。

在[2]之前，这种编码器-解码器方法主要用作现有统计机器翻译（SMT）系统的一部分。在[1]中，该方法用于对SMT系统生成的n-best列表重新排序，[3]的作者使用该方法为现有短语表提供额外分数。

在本文中，我们专注于分析直接翻译性能，如[2]中那样，使用两种模型配置。在两个模型中，我们都使用带门控隐藏单元的RNN[3]，因为这是唯一不需要非平凡方式来确定目标长度的选项之一。第一个模型将使用与[3]相同的带门控隐藏单元的RNN作为编码器，第二个将使用所提出的门控递归卷积神经网络（grConv）。我们旨在理解编码器-解码器方法对以BLEU衡量的翻译性能的归纳偏置。

## 4 实验设置

### 4.1 数据集

我们在英法翻译任务上评估编码器-解码器模型。我们使用双语平行语料库，这是通过[6]的方法从Europarl（6100万词）、新闻评论（550万）、联合国（4.21亿）以及两个分别为9000万和7.8亿词的爬取语料库组合中选择的3.4亿句对。我们没有使用单独的单语数据。

神经机器翻译模型的性能在news-test2012、news-test2013和news-test2014集合（各3000行）上测量。当与SMT系统比较时，我们使用news-test2012和news-test2013作为我们的开发集来调优SMT系统，使用news-test2014作为我们的测试集。

在准备好的平行语料库的所有句对中，出于计算效率的原因，我们仅使用英语和法语句子都最多30个词的句对来训练神经网络。此外，我们对英语和法语都仅使用最常见的30,000个词。所有其他稀有词被视为未知词并映射到特殊标记（[UNK]）。

### 4.2 模型

我们训练两个模型：RNN编码器-解码器（RNNenc）[3]和新提出的门控递归卷积神经网络（grConv）。注意两个模型都使用带门控隐藏单元的RNN作为解码器（见第2.1节）。

我们使用小批量随机梯度下降和AdaDelta[7]来训练两个模型。我们将方阵（转移矩阵）初始化为正交矩阵，其谱半径在RNNenc情况下设为1，在grConv情况下设为0.4。tanh和修正线性单元（$\max(0, x)$）分别用作RNNenc和grConv的逐元素非线性函数。

grConv有2000个隐藏神经元，而RNNenc有1000个隐藏神经元。两种情况下的词嵌入都是620维的。两个模型都训练了大约110小时，分别相当于296,144次更新和846,322次更新。

### 4.3 使用束搜索进行翻译

我们使用基本形式的束搜索来找到最大化特定模型（在这种情况下是RNNenc或grConv）给出的条件概率的翻译。在解码器的每个时间步，我们保留具有最高对数概率的 $s$ 个翻译候选，其中 $s = 10$ 是束宽。在束搜索期间，我们排除任何包含未知词的假设。对于在最高评分候选中选择的每个序列结束符号，束宽减少1，直到束宽达到零。

在RNN下（近似）找到最大对数概率序列的束搜索在[8]和[9]中被提出并成功使用。最近，[2]的作者发现这种方法在基于LSTM单元的纯神经机器翻译中是有效的。

当我们使用束搜索找到 $k$ 个最佳翻译时，我们不使用通常的对数概率，而是使用相对于翻译长度归一化的对数概率。这防止RNN解码器偏爱较短的翻译，这种行为在例如[10]中 earlier 被观察到。

## 5 结果与分析

### 5.1 定量分析

在本文中，我们对神经机器翻译模型的性质感兴趣。具体来说，翻译质量相对于源和/或目标句子的长度以及每个源/目标句子中模型未知词的数量。

首先，我们观察BLEU分数（反映翻译性能）如何随句子长度变化（见图4(a)-(b)）。显然，两个模型在短句上表现相对较好，但随着句子长度的增加显著下降。

我们在图4(c)中观察到与未知词数量类似的趋势。正如预期的那样，随着未知词数量的增加，性能迅速下降。这表明未来增加神经机器翻译系统使用的词汇表大小将是一个重要挑战。虽然我们只展示了RNNenc的结果，但我们在grConv上也观察到了类似的行为。

在表1(a)中，我们展示了使用两个模型获得的翻译性能以及基线短语SMT系统。显然，短语SMT系统仍然显示出优于所提出的纯神经机器翻译系统的性能，但我们可以看到在某些条件下（源句子和参考句子中都没有未知词），差异显著减小。此外，如果我们仅考虑短句（每句10-20个词），差异进一步减小（见表1(b)）。

此外，可以将神经机器翻译模型与现有的短语系统结合使用，最近在[3, 2]中发现这可以提高整体翻译性能（见表1(a)）。

该分析表明，当前的神经翻译方法在处理长句子方面存在弱点。最明显的解释假设是固定长度向量表示没有足够的容量来编码具有复杂结构和含义的长句子。为了编码变长序列，神经网络可能"牺牲"输入句子中的一些重要主题以记住其他主题。

这与传统的基于短语的机器翻译系统[11]形成鲜明对比。从图5可以看出，在相同数据集上训练的传统系统（带有用于语言模型的额外单语数据）倾向于在较长句子上获得更高的BLEU分数。

事实上，如果我们将源句子和参考翻译的长度限制在10到20个词之间，并且仅使用没有未知词的句子，测试集上的BLEU分数对于RNNenc和Moses分别为27.81和33.08。

注意，即使我们使用长达50个词的句子来训练这些模型，我们也观察到了类似的趋势。

### 5.2 定性分析

虽然BLEU分数被用作评估机器翻译系统性能的事实标准指标，但它不是完美的指标（见例如[12, 13]）。因此，这里我们展示从两个模型RNNenc和grConv生成的一些实际翻译。

在表2(a)-(b)中，我们展示了从开发集和测试集中随机选择的一些句子的翻译。我们选择了没有未知词的句子。(a)列出了长句子（超过30个词），(b)列出了短句子（少于10个词）。我们可以看到，尽管BLEU分数有差异，所有三个模型（RNNenc、grConv和Moses）在翻译方面都做得不错，特别是短句子。然而，当源句子很长时，我们注意到神经机器翻译模型的性能下降。

此外，我们在这里展示所提出的门控递归卷积网络学习表示什么类型的结构。使用示例句子"Obama is the President of the United States"，我们展示了grConv编码器学习的解析结构和生成的翻译，如图6所示。该图表明grConv首先将"of the United States"与"is the President of"合并，最后将其与"Obama is"和"."组合，这与我们的直觉很好地相关。

尽管grConv与RNN编码器-解码器相比表现出较低的性能，但我们发现grConv自动学习语法结构的这一特性很有趣，相信需要进一步研究。

## 6 结论与讨论

在本文中，我们研究了最近引入的基于纯神经网络的机器翻译系统的性质。我们专注于评估最近在[1, 2, 3]中提出的编码器-解码器方法在句子到句子翻译任务上的表现。在许多可能的编码器-解码器模型中，我们特别选择了两个在编码器选择上不同的模型：(1)带门控隐藏单元的RNN和(2)新提出的门控递归卷积神经网络。

在英法句对上训练这两个模型后，我们使用BLEU分数分析了它们的性能，关注句子长度和句子中未知/稀有词的存在。我们的分析揭示，神经机器翻译的性能受到句子长度的显著影响。然而，从定性角度看，我们发现两个模型都能够很好地生成正确的翻译。

这些分析表明了纯基于神经网络的机器翻译的若干未来研究方向。首先，找到一种在计算和内存方面扩展神经网络训练的方法很重要，以便可以使用更大的源语言和目标语言词汇表。特别是，当涉及到形态丰富的语言时，我们可能需要提出一种全新的方法来处理词。

其次，需要更多研究来防止神经机器翻译系统在长句子上表现不佳。最后，我们需要探索不同的神经架构，特别是解码器。尽管用作编码器的RNN和grConv在架构上有根本差异，但两个模型都受到句子长度的诅咒。这表明这可能是由于解码器缺乏表示能力。需要进一步调查和研究。

除了通用神经机器翻译系统的性质外，我们观察到所提出的门控递归卷积神经网络（grConv）的一个有趣性质。发现grConv在没有语言句法结构监督的情况下模仿输入句子的语法结构。我们认为这一性质使其适合机器翻译以外的自然语言处理应用。



## 致谢

作者感谢以下机构对研究资金和计算支持的支持：NSERC、Calcul Québec、Compute Canada、Canada Research Chairs和**CIFAR**。



## 参考文献

[1] Nal Kalchbrenner and Phil Blunsom. 2013. Two recurrent continuous translation models. In Proceedings of the ACL Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 1700–1709. Association for Computational Linguistics.

[2] Ilya Sutskever, Oriol Vinyals, and Quoc Le. 2014. Sequence to sequence learning with neural networks. In Advances in Neural Information Processing Systems (NIPS 2014), December.

[3] Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. 2014. Learning phrase representations using rnn encoder-decoder for statistical machine translation. In Proceedings of the Empirical Methods in Natural Language Processing (EMNLP 2014), October.

[4] Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. 2014. Neural machine translation by jointly learning to align and translate. arXiv preprint arXiv:1409.0473.

[5] S. Hochreiter and J. Schmidhuber. 1997. Long short-term memory. Neural Computation, 9(8):1735–1780.

[6] Amittai Axelrod, Xiaodong He, and Jianfeng Gao. 2011. Domain adaptation via pseudo in-domain data selection. In Proceedings of the ACL Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 355–362. Association for Computational Linguistics.

[7] Matthew D. Zeiler. 2012. ADADELTA: an adaptive learning rate method. Technical report, arXiv 1212.5701.

[8] Alex Graves. 2012. Sequence transduction with recurrent neural networks. In Proceedings of the 29th International Conference on Machine Learning (ICML 2012).

[9] Nicolas Boulanger-Lewandowski, Yoshua Bengio, and Pascal Vincent. 2013. Audio chord recognition with recurrent neural networks. In ISMIR.

[10] A. Graves. 2013. Generating sequences with recurrent neural networks. arXiv:1308.0850 [cs.NE], August.

[11] Philipp Koehn, Franz Josef Och, and Daniel Marcu. 2003. Statistical phrase-based translation. In Proceedings of the 2003 Conference of the North American Chapter of the Association for Computational Linguistics on Human Language Technology - Volume 1, NAACL '03, pages 48–54, Stroudsburg, PA, USA. Association for Computational Linguistics.

[12] Xingyi Song, Trevor Cohn, and Lucia Specia. 2013. BLEU deconstructed: Designing a better MT evaluation metric. In Proceedings of the 14th International Conference on Intelligent Text Processing and Computational Linguistics (CICLING), March.

[13] Chang Liu, Daniel Dahlmeier, and Hwee Tou Ng. 2011. Better evaluation metrics lead to better machine translation. In Proceedings of the Conference on Empirical Methods in Natural Language Processing, pages 375–384. Association for Computational Linguistics.
