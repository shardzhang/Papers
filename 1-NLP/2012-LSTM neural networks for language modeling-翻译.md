# LSTM神经网络用于语言建模

> Martin Sundermeyer, Ralf Schlüter, Hermann Ney | RWTH Aachen University



本文将LSTM（Long Short-Term Memory，长短期记忆）神经网络架构引入语言建模领域，**通过解决标准循环神经网络的梯度消失问题，实现更准确的词序列概率建模，在困惑度和词错误率上均取得显著改进**。

核心内容：

- 前馈神经网络LM仅利用固定上下文长度，标准RNN虽理论上可利用全部前驱词，但受梯度消失问题困扰难以训练
- LSTM通过门控机制（输入门、遗忘门、输出门）重新设计网络单元，使梯度缩放因子固定为1，显式避免梯度消失
- 在英语Treebank和大规模法语语料上验证LSTM LM的有效性，并与标准RNN LM进行系统比较
- 探索不同网络拓扑（投影层、聚类输出层）和输入序列长度对性能的影响

关键发现：

- LSTM LM在英语Treebank上困惑度比标准RNN LM降低约**8%**
- 在法语Quaero语音识别任务上，LSTM LM与Kneser-Ney模型插值后词错误率降低**0.3%~0.5%**
- LSTM隐藏层150个节点对应7.6M参数，而sigmoid网络350个节点仅7.1M参数，但LSTM性能更优
- 聚类输出层可显著加速训练和测试，理论加速比为$\sqrt{V}$（$V$为词汇表大小）

---



## 摘要

神经网络在语言建模任务中日益流行。前馈网络仅利用固定上下文长度来预测序列中的下一个词，而标准循环神经网络在概念上可以考虑所有前驱词。另一方面，众所周知循环网络难以训练，因此不太可能展现循环模型的全部潜力。

这些问题由长短期记忆（LSTM）神经网络架构来解决。在本工作中，我们在一个英语和一个大规模法语语言建模任务上分析了这种类型的网络。实验显示困惑度相比标准循环神经网络LM有约8%的相对改进。此外，我们在最先进的语音识别系统之上获得了词错误率的显著改进。

**索引术语：** 语言建模，循环神经网络，LSTM神经网络



## 1 引言

在自动语音识别中，识别系统的语言模型（LM）是整合给定自然语言的语法和语义约束的核心组件。虽然目前主要使用回退模型[1]进行识别通道，但前馈神经网络LM（首先在[2]中提出）已成为重排序阶段现有技术的重要补充[3]。

两种方法都依赖于n-gram近似，其中词序列 $w_1^N$ 的概率 $p(w_1^N)$ 被分解为：

$$
p(w_1^N) = \prod_{m=1}^{N} p(w_m | h_m)
$$

使得仅使用前 $n-1$ 个词 $h_m := w_{m-n+1}^{m-1}$ 来估计位置 $m$ 处词的概率。然而，神经网络LM克服了回退模型的一个主要缺点[5]：每当n-gram $(h, w)$ 在训练中未被观察到时，回退模型缺乏对该n-gram概率的显式估计。因此它回退到 $(n-1)$-gram $(\bar{h}, w)$ 的估计，其中 $h$ 最左边的词被移除以构造 $\bar{h}$，$\gamma(h)$ 是归一化常数：

$$
p(w|h) = \gamma(h) p(w|\bar{h})
$$

与回退模型相比，神经网络LM始终基于完整历史估计概率，无论n-gram是否在训练中被观察到。

另一方面，当使用前馈神经网络LM时，n-gram假设仍然导致建模中的不准确性。根据概率论的链式法则，需要考虑所有前驱词 $w_1^{m-1}$ 来预测句子的第 $m$ 个词：

$$
p(w_1^N) = \prod_{m=1}^{N} p(w_m | w_1^{m-1})
$$

这可以通过将前馈架构替换为适合序列建模的循环神经网络架构来补救[6, 7]。

不幸的是，循环神经网络难以使用随时间反向传播[8]进行训练。主要困难在于众所周知的梯度消失问题[9]，这意味着通过网络反向传播的误差函数的梯度会指数级衰减或增长。

改进循环神经网络训练的一种方法是利用高阶信息的更好优化算法[10]。然而，这通常以显著增加计算成本为代价，使得这些方法对于训练数据量极大的语言建模不太有吸引力。

另一种称为长短期记忆（LSTM）的解决方案在[11]中被提出：网络架构被修改以显式避免梯度消失问题，而训练算法保持不变。

在本工作中，我们将LSTM引入语言建模领域。我们在一个英语和一个大规模法语语料库上分析其在困惑度和词错误率方面的有效性。此外，我们研究了减少训练时间的技术并比较了不同的神经网络LM架构。



## 2 LSTM神经网络

在[11]中，梯度消失问题被详细分析。每当神经网络误差函数的梯度通过神经网络的一个单元反向传播时，它会被某个因子缩放。对于几乎所有实际相关的情况，该因子要么大于1，要么小于1。结果，在循环神经网络中，梯度随时间指数级爆炸或衰减。（从语言建模的角度来看，时间步对应于句子中的词位置。）因此，梯度要么主导下一个权重适应步骤，要么有效地丢失。

为了避免这种缩放效应，作者重新设计了神经网络的单元，使其对应的缩放因子固定为1。从这个设计目标获得的新单元类型在学习能力上相当有限。因此，该单元通过几个所谓的门控单元进行了丰富。最终单元如图1所示，其中我们包含了[12]和[13]中提出的原始LSTM单元的两个修改。

标准神经网络单元 $i$ 仅包含输入激活 $a_i$ 和输出激活 $b_i$，当使用tanh激活函数时，它们的关系为：

$$
b_i = \tanh(a_i)
$$

LSTM单元添加了几个中间步骤：对 $a_i$ 应用激活函数后，结果乘以因子 $b_\iota$。然后，由于循环自连接，上一时间步的内激活值乘以 $b_\phi$ 后被加入。最后，结果被 $b_\omega$ 缩放并输入另一个激活函数，产生 $b_i$。因子 $b_\iota, b_\phi, b_\omega \in (0, 1)$ 由额外的单元（称为输入门、输出门和遗忘门）控制。门控单元对前一隐藏层的激活和当前层前一时间步的激活以及LSTM单元的内激活求和。结果值通过logistic sigmoid函数压缩，然后分别设置为 $b_\iota$、$b_\phi$ 或 $b_\omega$。

整个LSTM单元包括门控单元可以被解释为计算机存储器的可微版本[14]。因此，LSTM单元有时也被称为LSTM记忆单元。无论是否遵循所提出的门控单元解释，LSTM架构以小的额外计算成本解决了梯度消失问题。此外，它具有将标准循环神经网络单元作为特殊情况包含在内的理想属性。



## 3 神经网络语言模型

尽管迄今为止成功应用的神经网络语言模型存在若干差异，但它们都共享一些基本原理：

- 输入词通过1-of-$K$编码，其中 $K$ 是词汇表中的词数。
- 在输出层，使用softmax激活函数产生正确归一化的概率值。
- 作为训练准则，使用交叉熵误差，等价于最大似然。

我们也遵循这种方法。通常建议对神经网络的输入数据进行归一化[15]，这意味着应用线性变换使数据具有零均值和单位方差。当使用1-of-$K$编码时，显然不是这种情况。

放弃输入特征的稀疏性（通常被利用来加速矩阵计算[16]），数据可以很容易地被归一化，因为存在依赖于训练数据中观察到的unigram计数的1-of-$K$编码输入特征的均值和方差的闭式解。相反，我们观察到归一化显著减慢了收敛速度。似乎当每个维度的输入数据位于相同的 $[0, 1]$ 范围内就足够了。

由于输入特征高度相关（例如，对于输入变量 $x$ 的第 $i$ 维，我们有 $x_i = 1 - \sum_{j \neq i} x_j$），对特征应用白化变换似乎更有前景。由于高维性，这在实践中似乎不可行。

关于网络拓扑，[6]中使用了单个循环隐藏层，而在[3]中应用了具有两个隐藏层的架构，第一层具有将输入词投影到连续空间的解释。以类似的精神，我们坚持图2所示的拓扑，其中我们将LSTM单元插入第二循环层，将其与不同投影层的标准神经网络单元组合。

对于大词汇量语言建模，训练强烈受限于softmax输出层输入激活 $a_i$ 的计算，与输入层不同，它不是稀疏的：

$$
a_i = \sum_{j=1}^{J} \omega_{ij} b_j
$$

其中 $J$ 表示最后一个隐藏层中的节点数，$\omega_{ij}$ 是最后一个隐藏层和输出层之间的权重，$i = 1, \ldots, V$，其中 $V$ 是词汇表大小。

为了减少计算量，[17]中（遵循[18]的思想）提出将词分成一组不相交的词类。然后概率 $p(w_m | w_1^{m-1})$ 可以分解如下：

$$
p(w_m | w_1^{m-1}) = p(w_m | c(w_m), w_1^{m-1}) \cdot p(c(w_m) | w_1^{m-1})
$$

其中 $w_m \in c(w_m)$，$c(w_m)$ 是词 $w_m$ 的类别。如何定义合理的类别集合在[19]中描述。使用这个恒等式，计算复杂度可以显著降低。



## 4 实验结果

对于实验结果，我们专注于两个语料库：英语Treebank-3语料库和来自Quaero项目的法语语料库。详细信息见表1。

**表1：语料库大小（以运行词数计）；Treebank语料库的词汇表大小为10K，Quaero法语为170K；dev1用作神经网络训练的验证数据，dev2用于优化LM缩放**

| LM | 训练 | dev1 | dev2 | 测试 |
|---|------|------|------|------|
| Treebank | 930K | 74K | – | 82K |
| Quaero French | 27M | 46K | 36K | 35K |

Treebank-3语料库的结果总结在图3中。首先，我们训练了一个具有图2所示架构的循环神经网络LM，只是省略了投影层。对于循环隐藏层，我们一次选择具有sigmoid激活函数的标准单元，一次选择LSTM单元，见图3(a)。我们发现LSTM模型的困惑度始终比标准循环神经网络低约8%。我们通过sigmoid循环网络获得的困惑度与rnnlm工具包[20]获得的非常接近。

这两个模型的训练时间处于相似的数量级。然而，对于给定的隐藏层节点数，相应的模型大小实际上差异很大：具有150个隐藏节点的LSTM版本对应7.6M参数，而sigmoid网络仅有3.0M权重参数。另一方面，当增加sigmoid网络的模型大小直到达到可比较的参数数量时，无法获得显著改进（350个隐藏节点对应7.1M参数）。此外，当使用投影层且词汇表大小巨大时，LSTM变体的模型大小开销可以忽略不计。

在第二组实验中，我们试图找出额外的投影层是否能带来进一步改进，见图3(b)。不幸的是，与原始LSTM版本相比，无论是线性层（激活函数为恒等函数）还是sigmoid层都没有导致更低的困惑度。我们对结果的解释是，这样的投影层创建了模糊的输入特征，使LSTM单元的学习任务复杂化。

对于迄今为止展示的结果，在训练和测试期间向网络呈现单个输入句子。这意味着最大上下文长度限制在约21个词，这是Treebank语料库中的平均句子长度。然而，与标准循环神经网络不同，LSTM网络可能能够利用更长的上下文大小。因此，我们通过拼接固定数量的连续句子来增加输入序列的大小。对性能的影响可以在图3(c)中看到。

我们观察到，在具有LSTM单元的单隐藏层情况下，当从一个句子切换到两个拼接句子时，可以获得小幅改进。对于具有sigmoid投影层的LSTM也是如此。有趣的是，与单个句子相比，线性投影层对较长的输入序列有帮助。

可能神经网络在输出层必须区分的不同词的数量太大，无法学习复杂的长程依赖。因此我们观察到一个普遍趋势：当上下文长度超过某个阈值时，困惑度显著增加，无论任何预处理。

然而，似乎线性层引入的输入特征模糊对长输入序列有益，当LSTM单元与此类型的投影层组合时，我们获得了最佳困惑度。

最后，我们研究了LSTM网络与聚类输出层之间的交互。对于此实验，我们使用了具有200个隐藏节点且没有投影层的LSTM网络。

如图3(d)所示，聚类对困惑度的影响仅为中等，而训练（以及测试）可以获得大的加速。理论上，当 $C = \sqrt{V}$ 时加速最大，其中 $C$ 表示类别数，$V$ 是词汇表大小。事实证明，这种行为在实践中并不完全匹配，因为类别大小不同。

除了相对较小的英语语料库的结果外，我们还将LSTM网络应用于大词汇量法语语音识别任务。在Quaero研究项目中，每年举行评估，对广播对话播客数据上的语音识别系统进行评估。

我们采用了在2011年评估中表现有竞争力的最佳法语识别系统。该系统包括最先进的声学模型，包括交叉适应、MLP特征和判别训练。回退LM在超过40亿词上训练。

从语音识别器创建的格中，我们提取了大小为 $n = 1000$ 的n-best列表。我们使用300个隐藏节点和2700万运行词的领域内训练数据训练了一个LSTM LM。尽管Kneser-Ney（KN）回退模型在超过一百倍的数据上训练，通过插值，我们在2011年评估的开发数据上获得了0.5%的词错误率改进，在测试数据上获得了0.3%的改进。

**表2：Quaero法语的词错误率结果**

| LM | dev2 | 测试 |
|---|------|------|
| KN 4-gram | 19.7% | 17.6% |
| KN 4-gram + LSTM | 19.2% | 17.3% |



## 5 结论

在本文中，我们将LSTM神经网络架构应用于两个语言建模任务。这种网络类型特别适合语言建模，因为理论上它允许对词序列的概率进行精确建模。与以前的方法相比，它不遭受标准循环神经网络训练的概念性问题。

我们探索了几种不同的神经网络拓扑，并分析了广泛使用的额外隐藏投影层的重要性。我们展示了LSTM网络可以与现有的聚类技术结合，以在小的性能损失下获得训练和测试时间的大幅加速。

实验表明，标准循环神经网络架构的性能可以通过LSTM在困惑度方面提高约8%。最后，当在最先进的法语识别系统之上将LSTM LM与巨大的Kneser-Ney平滑回退模型插值时，获得了相对较大的改进。

对于未来工作，分析标准网络和LSTM网络之间的差异以及对语音识别器识别质量的影响似乎很有趣。



## 6 致谢

本工作部分作为Quaero计划的一部分实现，由法国国家创新机构OSEO资助。



## 参考文献

[1] Kneser, R., and Ney, H., "Improved Backing-Off For M-Gram Language Modeling", Proc. of ICASSP 1995, pp. 181–184.

[2] Bengio, Y., Ducharme, R., "A neural probabilistic language model", Proc. of Advances in Neural Information Processing Systems (2001), vol. 13., pp. 932–938.

[3] Schwenk, H., "Continuous space language models", Computer Speech and Language 21 (2007), pp. 492–518.

[5] Oparin, I., Sundermeyer, M., Ney, H., Gauvain, J.-L., "Performance Analysis of Neural Networks in Combination with n-Gram Language Models", Proc. of ICASSP 2012, accepted for publication.

[6] Mikolov, T., Karafiát, M., Burget, L., Černocký, J. H., and Khudanpur, S., "Recurrent neural network based language model" Proc. of Interspeech 2010, pp. 1045–1048.

[7] Elman, J., "Finding Structure in Time", Cognitive Science 14 (1990), pp. 179–211.

[8] Rumelhart, D. E., Hinton, G. E., Williams, R. J., "Learning representations by back-propagating errors", Nature 323 (1986), pp. 533–536.

[9] Bengio, Y., Simard, P., Frasconi, P., "Learning long-term dependencies with gradient descent is difficult" IEEE Transactions on Neural Networks 5 (1994), pp. 157–166.

[10] Martens, J., Sutskever, I., "Learning Recurrent Neural Networks with Hessian-Free Optimization", Proc. of the 28th Int. Conf. on Machine Learning 2011.

[11] Hochreiter, S., Schmidhuber, J., "Long Short-Term Memory", Neural Computation 9 (8), 1997, pp. 1735–1780.

[12] Gers, F. A., "Learning to Forget: Continual Prediction with LSTM", Proc. of the 9th Int. Conf. on Artificial Neural Networks, 1999, pp. 850–855.

[13] Gers, F. A., Schraudolph, N. N., Schmidhuber, J., "Learning Precise Timing with LSTM Recurrent Networks", Journal of Machine Learning Research 3, 2002, pp. 115–143.

[14] Graves, A., Schmidhuber, J., "Framewise Phoneme Classification with Bidirectional LSTM and Other Neural Network Architectures", Neural Networks, Vol. 18, Issue 5–6, 2005, pp. 602–610.

[15] Bishop, C., "Neural Networks for Pattern Recognition", Clarendon Press, Oxford, 1995.

[16] Le, H. S., Allauzen, A., Wisniewski, G., Yvon, F., "Training continuous space language models: some practical issues", Proc. of the 2010 Conf. on Emp. Methods in NLP, pp. 778–788.

[17] Morin, F., Bengio, Y., "Hierarchical Probabilistic Neural Network Language Model", Proc. of the 10th Int. Workshop on Artificial Intelligence and Statistics.

[18] Goodman, J., "Classes for fast maximum entropy training", Proc. of the ICASSP, 2001.

[19] Mikolov, T., Kombrink, S., Burget, L., Černocký, J., Khudanpur, S., "Extensions of Recurrent Neural Network Language Model", Proc. of the ICASSP 2011, pp. 5528–5531.

[20] Mikolov, T., Kombrink, S., Deoras, A., Burget, L., Černocký, J., "RNNLM – Recurrent Neural Network Language Modeling Toolkit", Proc. of the 2011 ASRU Workshop, pp. 196–201.
