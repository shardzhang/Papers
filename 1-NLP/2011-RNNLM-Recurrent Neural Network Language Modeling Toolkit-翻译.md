# RNNLM - Recurrent Neural Network Language Modeling Toolkit

> Tomáš Mikolov #1, Stefan Kombrink #2, Anoop Deoras *3, Lukáš Burget #4, Jan "Honza" Černocký #5
>
> 布尔诺理工大学（Brno University of Technology）Speech@FIT，捷克布尔诺：imikolov@fit.vutbr.cz, kombrink@fit.vutbr.cz, burget@fit.vutbr.cz, cernocky@fit.vutbr.cz
>
> 约翰霍普金斯大学（Johns Hopkins University）语言与语音处理中心（CLSP），美国马里兰州巴尔的摩：adeoras@jhu.edu



本文介绍了一个免费开源的**循环神经网络语言模型**训练工具包，核心思路是——**用 Elman 型简单 RNN 加 BPTT 训练的语言模型可显著优于 n-gram 模型，且训练数据越多优势越大**；工具包提供类别层、输入-输出直连 和 哈希最大熵模型（RNNME）三种降复杂度方案，可即插即用地 **重打分** 改进现有 ASR/MT 系统。

核心内容：

- 架构：输入层 = 上一个词 $w(t)$ 的 1-of-N 编码 + 隐藏层上一时刻状态 $s(t-1)$ ，隐藏层 $s(t)$ 用 sigmoid 激活，输出层 $y(t)$ 表示下一个词的概率分布，可选类别层 $c(t)$ 降复杂度
- 训练：截断 BPTT（如 -bptt 6 或分块 -bptt-block 10），随机梯度下降；稳定性技巧：双精度、梯度裁剪、正则化、循环权重一次性大更新
- 功能：训练 RNN LM / 哈希 ME LM / RNNME 联合模型；测试输出困惑度与 log10 概率；支持 **n-best 列表重打分**、与 SRILM 概率线性插值、随机采样生成句子
- 超参数经验：最高精度用 -class 1 + 长 BPTT + 大隐藏层 + 多模型插值；中型任务用约 $\sqrt{|V|}$ 个类（300-500）与 300-1000 隐藏单元；超大数据集用 RNNME 联合模型

关键发现：

- 与 SRILM 结果可直接对比；RNN 相比 n-gram 的改进随训练数据增加而扩大
- RNNME 联合训练与 RNN、ME 简单插值表现"非常不同"——RNN 可专注发现与 ME 互补的信息
- 用 n-gram 近似 RNN 的变分方法只需约 20%–40% 的完整重打分改进，但无需改动解码系统
- 训练好的权重可量化到几个 bit 而无明显性能损失

---



## 摘要

我们介绍一个免费可用的开源工具包，用于训练基于循环神经网络的语言模型。它可以很容易地用于改进现有的**语音识别和机器翻译**系统。同时，它也可以作为未来高级语言建模技术研究的基线。在本文中，我们讨论了最优参数选择以及不同的功能模式。该工具包、示例脚本和基本配置可以在 http://rnnlm.sourceforge.net/ 免费获取。



## I. 引言、动机与目标

统计语言建模吸引了大量关注，因为自然语言模型是许多实际系统的重要组成部分。此外，可以估计，随着研究的进一步进展，语言模型将更接近人类对语言的理解 [1] [2]，并且全新的应用将变得实际可行。很快，语言建模的任何重大进展都可以被用于现有的语音识别和统计机器翻译系统。

然而，**整个研究领域几十年来一直在努力克服基于 n-gram 频率的非常简单但也有效的模型** [3] [4]。人们开发了许多技术来打败 n-gram，但这些改进是以计算复杂度为代价的。而且，这些改进通常是**在非常基本的系统上**报告的，在应用到用海量数据训练的 n-gram 模型的最先进配置后，许多技术带来的改进消失了。这导致了语音识别研究者的怀疑态度。

在我们之前的工作中，我们比较了许多众所周知的高级语言建模技术，发现基于神经网络的语言模型（NNLM，Neural Network based Language Model）在几个标准配置上表现最好 [5]。这类模型由 Bengio [6] 在大约十年前引入。它们的主要弱点是巨大的计算复杂度和不平凡的实现。神经语言模型的成功训练需要**超参数的良好选择，如学习率和隐藏层的大小**。

为了帮助克服这些基本障碍，我们决定发布我们的循环神经网络语言模型（RNNLM，Recurrent Neural Network based Language Model）训练工具包。我们已经在 **[7] 中展示了循环架构在几个配置上优于前馈架构**。实现简单且易于理解。

最重要的是，循环神经网络从研究的角度来看非常有趣，因为它们允许对任意长度的序列和模式进行有效处理——这些模型可以学习在隐藏层中存储过去的信息。循环神经网络可以有记忆，因此是克服 n-gram 模型最痛苦且常被批评的缺点——统计上只依赖前几个词——的重要一步。

在本文中，我们介绍一个开源且免费可用的工具包，用于训练基于循环神经网络和基于哈希的最大熵模型的统计语言模型。该工具包包括降低计算复杂度的技术（输出层的类别和输入层与输出层之间的直连）。它被设计为提供与流行的 n-gram 模型训练工具包 SRILM [8] 可比较的结果。RNNLM 工具包的主要目标如下：

- 促进高级语言建模技术的研究
- 易于使用
- 简单可移植的代码，不依赖任何外部库
- 计算效率

在本文中，我们描述如何将 RNNLM 轻松应用于几乎任何语音识别或机器翻译系统。



## II. 循环神经网络

工具包中使用的循环神经网络架构如图 1 所示（通常被称为 Elman 网络或简单 RNN）。输入层使用前一个词 $w(t)$ 的 1-of-N 表示与隐藏层前一个状态 $s(t-1)$ 的拼接。隐藏层 $s(t)$ 中的神经元使用 sigmoid 激活函数。输出层 $y(t)$ 与 $w(t)$ 具有相同的维度，在网络训练完成后，它表示给定前一个词和上一时间步隐藏层状态时下一个词的概率分布 [9]。类别层 $c(t)$ 可以选择性使用，以降低模型的计算复杂度，代价是准确率的小幅损失 [7]。训练通过标准的随机梯度下降算法执行，表示循环权重的矩阵 $W$ 使用时间反向传播算法（BPTT，Backpropagation Through Time）训练 [10]。

<img src=".picture/2011-RNNLM-Recurrent Neural Network Language Modeling Toolkit-fig1.png" alt="图1" style="zoom:50%;" />

**图 1：** 带类别的循环神经网络语言模型。

在工具包中，我们使用截断 BPTT——网络在时间上展开指定的时间步数。为了更快训练，可以在处理几个时间步后再展开网络的循环部分，这会导致训练期间的计算复杂度显著降低。

循环神经网络似乎是建模序列数据的一个非常好的选择。然而，在人们证明基于梯度下降的传统训练算法存在梯度消失和梯度爆炸问题后，RNN 受到了很多怀疑 [11]。这就是为什么 RNN 有时被认为难以仅靠基于梯度下降的方法成功训练的原因。

事实上，像时间反向传播 [10] 这样的算法的问题部分可能在于实际实现，因为很容易犯错，而且算法难以调试。关于 BPTT 实现的好描述可以在 [12] 中找到。此外，在某些情况下训练可能会发散。训练的稳定性可以通过以下方式提高：

- 权重使用双精度而不是单精度浮点数
- 限制最大梯度以防止梯度爆炸
- 使用正则化
- 循环权重在一次大更新中更新 [12]

一旦网络训练完成，权重的精确值就不再重要——我们最近证明了权重的值可以被量化到几个 bit 而没有显著的性能损失 [13]。



## III. 基本功能

该工具包支持几个功能，主要用于基本的语言建模操作：训练 RNN LM、训练基于哈希的最大熵模型（ME LM）以及 RNNME LM（联合训练的 RNN 和 ME 模型 [14]）。对于评估，可以在一些测试数据上计算困惑度（perplexity），或者对 n-best 列表进行重打分，以评估模型对词错误率或 BLEU 分数的影响。此外，我们支持从模型生成随机词序列的选项，这可用于用 n-gram 模型近似 RNN 模型，代价是内存复杂度 [15]。

### A. 训练阶段

输入数据预期是简单的 ASCII 文本格式，词之间有空格，每个句子末尾有行结束符。在指定训练数据集之后，会自动构建一个词表，并作为 RNN 模型文件的一部分保存。注意，如果想使用受限词表（例如用于开放词表实验），文本数据应该在工具包之外修改，首先将所有词表外的词改写为 <unk> 或类似的特珠 token。

在词表学习完成后，训练阶段开始（可选地，如果使用 -debug 2 选项，可以显示进度）。默认预期通过 -valid 选项提供一些验证数据，以控制训练轮数（epoch）和学习率。然而，也可以在没有验证数据的情况下训练模型；可以使用 -one-iter 选项来实现这一目的。模型在每个完成的 epoch 后保存（或者在处理指定数量的词后也保存）；训练过程如果被打断可以继续。

### B. 测试阶段

模型训练完成后，可以在一些测试数据上评估，并显示困惑度和 log10 概率作为结果。RNNLM 工具包被设计为提供可以与流行的 SRILM 工具包给出的结果相比较的结果。我们还支持线性插值各种模型给出的词概率的选项。对于 RNNLM 和 SRILM，都可以使用 -debug 2 选项在测试阶段获得详细输出，使用 -lm-prob 开关可以插值两个模型给出的概率。我们在 RNNLM 网页的示例脚本中提供了进一步的细节。

对于 n-best 列表重打分，我们通常对整句的概率感兴趣，这些概率在重排序期间用作分数。RNNLM 预期的输入是要打分的句子列表，每个假设的第一个 token 是唯一标识符。输出是所有句子的分数列表。此模式通过使用 -nbest 开关指定。n-best 列表输入文件的示例：

```
1 WE KNOW
1 WE DO KNOW
1 WE DONT KNOW
2 I AM
2 I SAY
```



## IV. 超参数的典型选择

由于基于神经网络的语言模型的计算复杂度巨大，在合理时间内成功训练模型可能需要一些经验，因为某些参数组合太昂贵而无法探索。存在几种可能的场景，取决于一个人是想优化最终模型的准确率、训练速度、重打分速度还是模型的大小。我们将简要提及一些有用的参数配置。

### A. 最佳准确率的选择

为了达到可能的最佳准确率，建议通过 -class 1 关闭类别，并使用 -min-improvement 1 开关，在观察到验证数据上任何改进时持续训练。接下来，BPTT 算法应至少运行 6 步（-bptt 6）。隐藏层的大小应尽可能大。训练几个具有不同随机权重初始化（通过使用 -rand-seed 开关）的模型，并将所有模型给出的结果概率插值在一起是有用的 [5]。

### B. 中型任务的参数

对于小型数据集，上述参数选择也会非常耗时。对于 2000-5000 万训练词，最好牺牲一点准确率来换取更低的计算复杂度。最有用的选项是使用类别（-class），大约 $\sqrt{|V|}$ 个类别，其中 $|V|$ 是未截断词表的大小（通常，类别数量应在 300-500 左右）。应注意，工具包的用户只需指定类别数量，这些类别会根据词的 unigram 频率自动找到。BPTT 算法应以分块模式运行，例如使用 -bptt-block 10。隐藏层的大小应设置为大约 300-1000 个单元，使用 -hidden 开关。数据越多，需要的隐藏层越大。同样，词表越小，隐藏层应越大，以确保模型有足够的容量。隐藏层的大小严重影响性能；并行训练几个不同隐藏层大小的模型可能是有用的，这样可以估计使用更大的隐藏层能获得多少性能。

### C. 超大数据集的参数

对于 1-10 亿词的数据集，仍然可以在合理时间内训练具有小隐藏层的 RNN 模型。然而，这种选择严重降低了最终性能，因为在大量数据上训练的网络具有小隐藏层时，存储信息的容量不足。在我们之前的工作中，将 RNN 模型与最大熵模型联合训练被证明是非常有益的（后者可以看作原始 RNN 模型中输入层和输出层之间的权重矩阵）。我们将这种架构表示为 RNNME [14]，应该指出，它的表现与 RNN 和 ME 模型的简单插值非常不同——主要区别在于两个模型是联合训练的，因此 RNN 模型可以专注于发现与 ME 模型互补的信息。

基于哈希的 ME 实现可以通过使用 -direct 开关指定为哈希保留的参数数量来启用（此选项只增加内存复杂度，不增加计算复杂度），ME 模型的 n-gram 特征阶数由 -direct-order 指定。计算复杂度随 ME 模型的阶数线性增加，对于阶数为 N 的模型，它大约与具有 N 个隐藏神经元的 RNN 模型相同。通常，使用最高 4-gram 特征的 ME 就足够了。由于实现的基于哈希的性质，如果哈希的大小不足，更高的阶数实际上可能降低性能。RNNME 架构的缺点在于其高内存复杂度。



## V. 在 ASR/MT 系统中的应用

该工具包可以很容易地用于重打分来自任何可以产生格（lattice）的系统的 n-best 列表。n-best 列表可以例如使用 SRILM 的 lattice-tool 从格中提取。RNNLM 在 ASR 系统中的典型用法包括以下步骤：

- 训练 RNN 语言模型（一个或多个）
- 解码话语（utterance），产生格
- 从格中提取 n-best 列表
- 计算基线 n-gram 模型和 RNN 模型给出的句子级分数
- 执行各种 LM 给出的对数分数的加权线性插值（权重应在开发数据上调优）
- 使用新的 LM 分数对 n-best 列表重排序

应确保输入格足够宽以获得任何改进——这可以通过测量 oracle 词错误率来验证。通常，即使是 20-best 列表重打分也能以可忽略的计算复杂度提供大部分可实现的改进。另一方面，可以通过构建完整的 n-best 列表来执行全格重打分，因为每个格包含有限数量的唯一路径。然而，这种方法的计算很复杂，在 [16] 中提出了一种使用 RNNLM 进行格重打分的更有效方法，并附带一个免费工具¹。

一个自包含的示例在下载部分提供（http://rnnlm.sourceforge.net 下），演示了在中等规模的华尔街日报 ASR 任务上使用 Kaldi 语音识别工具包进行 RNN 重打分。

或者，可以用 n-gram 模型近似 RNN 语言模型。这可以通过以下步骤完成：

- 训练 RNN 语言模型
- 从 RNN 模型生成大量随机句子
- 基于随机句子构建 n-gram 模型
- 将近似的 n-gram 模型与基线 n-gram 模型插值
- 使用新的 n-gram 模型解码话语

这种方法的好处是系统中不需要任何 RNNLM 重打分代码。其代价是额外的内存复杂度（需要生成大量随机句子），并且使用近似，在通常情况下只能实现完整 RNNLM 重打分所能实现改进的大约 20%-40%。我们在 [15] [17] 中更详细地描述了这种技术。

¹http://www.clsp.jhu.edu/~adeoras/HomePage/Code Release.html



## VI. 结论与未来工作

所介绍的 RNN 语言模型训练工具包可以用于改进现有的语音识别和机器翻译系统。我们将工具包设计为易于使用和安装——它用简单的 C/C++ 代码编写，不依赖任何外部库（如 BLAS）。发布该工具包的主要动机是促进高级语言建模技术的研究——尽管在过去三十年中投入了巨大的研究努力，n-gram 仍被认为是最先进的技术，我们希望在未来改变这一点。

我们之前已经表明，对于语音识别，RNN 模型显著优于 n-gram，而且随着训练数据的增加，改进也在扩大。因此，从实际的角度来看，主要问题是在非常大的语料库上快速训练这些模型。尽管设计简单，RNNLM 工具包可以在几天内在数亿词的语料库上训练出非常好的 RNN 语言模型。

未来的工作可能专注于增量式改进，即训练算法的并行化 [18]、在 GPU 上训练 RNN [19]、优化重打分 [16]、降低 RNNME 架构的内存复杂度 [20]、RNNLM 的压缩 [13]。然而，我们也希望该工具包能推动语言模型的研究，并引起人们对一些非常有趣的研究问题和课题的关注——语言是否可以从原始文本数据中无监督地学习、处理序列数据的模型对记忆的需求、统计语言建模中语言学知识的有用性存疑、可以发现长程规律的高级 RNN 架构的训练等等。"更多数据更好"的策略在统计语言建模（以及一般的自动语音识别和机器翻译）中已经主导了相当长的时间；然而，遵循这一策略，我们似乎并没有更接近人类水平的性能。



## 致谢

这项工作部分得到了捷克共和国技术局（Technology Agency of the Czech Republic）资助 No. TA01011328、捷克教育部项目 No. MSM0021630528、捷克共和国资助局项目 No. 102/08/0707 以及捷克贸易与商务部项目 No. FR-TI1/034 的支持。Anoop Deoras 部分由约翰霍普金斯大学 HLT-COE 资助。



## 参考文献

[1] J. T. Goodman, "A Bit of Progress in Language Modeling Extended Version," Microsoft Research, Tech. Rep. MSR-TR-2001-72, 2001.

[2] M. Hutter, "The Human knowledge compression prize," 2006.

[3] F. Jelinek, "Up From Trigrams! The struggle for improved language models," in Proceedings of Eurospeech, 1991.

[4] R. Rosenfeld, "Two decades of statistical language modeling: where do we go from here?" Proceedings of the IEEE, vol. 88, pp. 1270–1278, 2000.

[5] T. Mikolov, A. Deoras, S. Kombrink, L. Burget, and J. Černocký, "Empirical evaluation and combination of advanced language modeling techniques," in Proceedings of Interspeech, 2011.

[6] Y. Bengio, R. Ducharme, P. Vincent et al., "A neural probabilistic language model," Journal of Machine Learning Research, vol. 3, pp. 1137–1155, 2003.

[7] T. Mikolov, S. Kombrink, L. Burget, J. Černocký, and S. Khudanpur, "**Extensions of recurrent neural network language model**," in Proceedings of ICASSP, 2011.

[8] A. Stolcke, "SRILM – an extensible language modeling toolkit," in Proceedings of ICSLP, 2002.

[9] T. Mikolov, M. Karafiát, L. Burget, J. Černocký, and S. Khudanpur, "**Recurrent neural network based language model**," in Proceedings of Interspeech, 2010.

[10] D. E. Rumelhart, G. E. Hinton, and R. J. Williams, "Learning internal representations by error propagation," Mit Press Computational Models Of Cognition And Perception Series, pp. 318–362, 1986.

[11] Y. Bengio, P. Simard, and P. Frasconi, "**Learning long-term dependencies with gradient descent is difficult**." IEEE transactions on neural networks, vol. 5, pp. 157–166, 1994.

[12] M. Bodén, "A guide to recurrent neural networks and backpropagation," in In the Dallas project, SICS Technical Report T2002:03, SICS, 2002.

[13] T. Mikolov, I. Sutskever, A. Deoras, H. S. Le, S. Kombrink, and J. Černocký, "Compression of Language Models Using Subword Neural Networks," in Submitted to ICASSP, 2012.

[14] T. Mikolov, A. Deoras, D. Povey, L. Burget, and J. Černocký, "**Strategies for Training Large Scale Neural Network Language Models**," in Accepted to ASRU, 2011.

[15] A. Deoras, T. Mikolov, S. Kombrink, M. Karafiát, and S. Khudanpur, "Variational Approximation of Long-Span Language Models for LVCSR," in Proceedings of ICASSP, 2011.

[16] A. Deoras, T. Mikolov, and K. Church, "Fast Rescoring Strategy to Capture Long Distance Dependencies," in Proceedings of EMNLP, 2011.

[17] S. Kombrink, T. Mikolov, M. Karafiát, and L. Burget, "Recurrent Neural Network based Language Modeling in Meeting Recognition," in Proceedings of Interspeech, 2011.

[18] H. Schwenk, "Continuous space language models," Comput. Speech Lang., vol. 21, pp. 492–518, July 2007.

[19] I. Sutskever, J. Martens, and G. Hinton, "**Generating Text with Recurrent Neural Networks**," in Proceedings of ICML, 2011.

[20] P. Xu, S. Khudanpur, and A. Gunawardana, "Randomized Maximum Entropy Language Models," in Accepted to ASRU, 2011.
