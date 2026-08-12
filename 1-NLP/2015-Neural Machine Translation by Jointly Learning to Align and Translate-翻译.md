# 通过联合学习对齐和翻译的神经机器翻译

> Dzmitry Bahdanau | Jacobs University Bremen, Germany
> KyungHyun Cho, Yoshua Bengio∗ | Université de Montréal



本文提出了一种新的神经机器翻译架构，通过引入注意力机制来解决传统编码器-解码器模型将整个源句子**压缩为固定长度向量**的瓶颈问题。核心内容：

- 提出了一种**新的对齐模型**，允许解码器在生成每个目标词时自动（软）搜索源句子中与预测该目标词相关的部分，无需显式地将这些部分形成硬分段
- 使用双向 RNN 作为编码器来注释源序列，**使每个词的注释同时包含前文和后文的信息**
- 在英法翻译任务上实现了与 **传统基于短语的翻译系统**（Moses）相当的翻译性能

关键发现：

- **固定长度向量是基础编码器-解码器架构性能瓶颈的根本原因——随着输入句子长度增加，基础编码器-解码器性能迅速下降**
- 提出的 RNNsearch 模型对句子长度具有更强的鲁棒性，RNNsearch-50 即使对长度超过 50 个词的句子也没有性能恶化
- 通过 **可视化注释权重** 发现，模型学到的（软）对齐 与 **人类直觉** 高度一致，能够正确处理英法之间的**非单调对齐**
- 软对齐 相比于 硬对齐具有天然优势，能够自然地处理 **源语言和目标语言短语长度不同** 的情况

---



## 摘要

神经机器翻译是最近提出的一种机器翻译方法。与传统的 **统计机器翻译**不同，神经机器翻译旨在构建一个单一的神经网络，该网络可以联合调优以最大化翻译性能。最近提出的神经机器翻译模型通常属于**编码器-解码器**家族，**将源句子编码为一个固定长度的向量，解码器从中生成翻译**。在本文中，我们推测使用固定长度向量是改进这种基础编码器-解码器架构性能的瓶颈，并提议通过**允许模型在预测目标词时自动（软）搜索源句子中与预测该目标词相关的部分**来扩展该架构，而无需显式地将这些部分形成硬分段。通过这种新方法，我们在英法翻译任务上取得了与现有最先进的基于短语的系统相当的翻译性能。此外，定性分析表明，模型找到的（软）对齐与我们的直觉很好地吻合。



## 1 引言

神经机器翻译是一种新兴的机器翻译方法，最近由 Kalchbrenner 和 Blunsom[18]、Sutskever 等人[28] 以及 Cho 等人[8] 提出。与传统的基于短语的翻译系统（例如，Koehn 等人 [20]）由许多独立调优的小子组件组成不同，神经机器翻译试图构建和训练一个单一的、大型的神经网络，该网络读取一个句子并输出正确的翻译。

大多数提出的神经机器翻译模型属于编码器-解码器家族[28,7]，每种语言都有一个编码器和一个解码器，或者涉及一个针对每个句子进行编码的语言特定编码器，其输出随后进行比较[15]。编码器神经网络读取源句子并将其编码为一个固定长度的向量。解码器随后从编码向量输出翻译。**整个编码器-解码器系统（由该语言对的编码器和解码器组成）被联合训练以最大化给定源句子的正确翻译概率。**

这种编码器-解码器方法的一个潜在问题是，神经网络需要能够将源句子的所有必要信息压缩到一个固定长度的向量中。这可能使神经网络**难以处理长句子**，尤其是那些比训练语料中的句子更长的句子。[8]的研究表明，**基础编码器-解码器的性能确实随着输入句子长度的增加而迅速下降。**

为了解决这个问题，我们引入了编码器-解码器模型的一个扩展，该扩展学习联合对齐和翻译。每次提出的模型生成一个翻译词时，它（软）搜索源句子中最重要的信息集中的位置集合。然后，模型基于与这些源位置相关的上下文向量以及所有先前生成的目标词来预测一个目标词。



## 2 背景：神经机器翻译

从概率的角度来看，翻译等价于找到一个目标句子 $y$ ，使得给定源句子 $x$ 的条件概率 $\text{argmax}_y p(y \mid x)$ 最大。在神经机器翻译中，我们拟合一个参数化模型，使用并行训练语料 **最大化句子对的条件概率**。一旦翻译模型学习了条件分布，给定一个源句子，可以通过搜索最大化条件概率的句子来生成相应的翻译。

最近，许多论文提出使用神经网络直接学习这个条件分布（例如，[18,7,28,8,10]）。**这种神经机器翻译方法通常由两个组件组成，第一个组件对源句子 $x$ 进行编码，第二个组件解码为目标句子 $y$** 。例如，[7]和[28]使用了两个循环神经网络（RNN）来将可变长度的源句子编码为固定长度向量，并将该向量解码为可变长度的目标句子。

尽管是一种相当新的方法，神经机器翻译已经显示出有希望的结果。[28]报告称，基于具有长短期记忆（LSTM）单元的 RNN 的神经机器翻译在英法翻译任务上取得了接近传统基于短语的机器翻译系统最先进性能的水平[†1]。将神经组件添加到现有的翻译系统中，例如，对短语表中的短语对进行评分[7]或对候选翻译进行重新排序[28]，已经能够超越以前的最先进性能水平。

[^1]: 我们所说的最先进性能是指传统的基于短语的系统在不使用任何基于神经网络的组件的情况下的性能。



### 2.1 RNN 编码器-解码器

在这里，我们简要描述由 [7]和 [28]提出的基础框架，称为 RNN 编码器-解码器，我们在其上构建了一种**学习同时对齐和翻译的新架构**。

在编码器-解码器框架中，编码器读取输入句子——一个**向量序列** $x = (x_1, \cdots, x_{T_x})$ ——到一个向量 $c$ 中[†2]。最常用的方法是使用一个 RNN，使得：

$$
h_t = f(x_t, h_{t-1}) \qquad (1)
$$

和

$$
c = q(\{h_1, \cdots, h_{T_x}\}),
$$

其中 $h_t \in \mathbb{R}^n$ 是时间 $t$ 的隐藏状态， $c$ 是从隐藏状态序列生成的向量。 $f$ 和 $q$ 是某些非线性函数。[28]使用了 **LSTM** 作为 $f$ ，并且例如 $q(\{h_1, \cdots, h_T\}) = h_T$ 。

[^2]: 尽管大多数以前的工作（例如，[7,28,18]）使用固定长度向量来编码可变长度的输入句子，但这并非必需，而且正如我们稍后将展示的，拥有一个可变长度向量甚至可能是有益的。

解码器通常被训练为在给定上下文向量 $c$ 和所有先前预测的词 $\{y_1, \cdots, y_{t'-1}\}$ 的条件下预测下一个词 $y_{t'}$ 。换句话说，解码器通过将联合概率分解为 **有序的条件概率** 来定义翻译 $y$ 上的概率：

$$
p(y) = \prod_{t=1}^{T} p(y_t \mid \{y_1, \cdots, y_{t-1}\}, c), \qquad (2)
$$

其中 $y = (y_1, \cdots, y_{T_y})$ 。**使用 RNN**，每个条件概率被建模为：

$$
p(y_t \mid \{y_1, \cdots, y_{t-1}\}, c) = g(y_{t-1}, s_t, c), \qquad (3)
$$

其中 $g$ 是一个非线性的、可能多层的函数，输出 $y_t$ 的概率， $s_t$ 是 RNN 的隐藏状态。应当注意，也可以使用其他架构，例如 RNN 和反卷积神经网络的混合[18]。



## 3 学习对齐和翻译

在本节中，我们提出了一种用于神经机器翻译的新架构。新架构由一个**双向 RNN 作为编码器**（第 3.2 节）和一个在解码翻译时模拟搜索源句子的解码器（第 3.1 节）组成。

### 3.1 解码器：一般描述

在新模型架构中，我们将公式（2）中的每个条件概率定义为：

$$
p(y_i \mid y_1, \ldots, y_{i-1}, x) = g(y_{i-1}, s_i, c_i), \qquad (4)
$$

其中 $s_i$ 是时间 $i$ 的 RNN 隐藏状态，由下式计算：

$$
s_i = f(s_{i-1}, y_{i-1}, c_i).
$$

应当注意，与现有的编码器-解码器方法（见公式（2））不同，这里的概率是基于每个目标词 $y_i$ 对应的**不同上下文向量 $c_i$ 进行条件化**的。

上下文向量 $c_i$ 依赖于编码器将输入句子映射到的注释序列 $(h_1, \cdots, h_{T_x})$ 。**每个注释 $h_i$ 包含关于整个输入序列的信息，并重点关注输入序列第 $i$ 个词周围的部分**。我们将在下一节详细解释注释是如何计算的。

上下文向量 $c_i$ 随后被计算为这些注释 $h_j$ 的加权和：

$$
c_i = \sum_{j=1}^{T_x} \alpha_{ij} h_j. \qquad (5)
$$

每个注释 $h_j$ 的权重 $\alpha_{ij}$ 由下式计算：

$$
\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^{T_x} \exp(e_{ik})}, \qquad (6)
$$

其中

$$
e_{ij} = a(s_{i-1}, h_j)
$$

是一个对齐模型，用于评估输入位置 $j$ 附近的词与输出位置 $i$ 的匹配程度。该分数基于 RNN 隐藏状态 $s_{i-1}$ （就在发出 $y_i$ 之前，公式（4））和输入句子的第 $j$ 个注释 $h_j$ 。

**我们将对齐模型 $a$ 参数化为一个前馈神经网络，该网络与提出的系统的所有其他组件联合训练**。注意，与传统机器翻译不同，对齐不被视为潜在变量。相反，对齐模型直接计算软对齐，这允许成本函数的梯度通过其反向传播。**该梯度可用于联合训练 对齐模型 以及 整个翻译模型。**

我们可以将通过计算**所有注释的加权和**的方法理解为**计算一个期望注释**，其中期望是在可能的对齐上计算的。**令 $\alpha_{ij}$ 为目标词 $y_i$ 对齐到（或翻译自）源词 $x_j$ 的概率**。**那么，第 $i$ 个上下文向量 $c_i$ 就是所有注释以概率 $\alpha_{ij}$ 加权的期望注释。**

概率 $\alpha_{ij}$ 或其关联的能量 $e_{ij}$ 反映了注释 $h_j$ 相对于先前隐藏状态 $s_{i-1}$ 在决定下一个状态 $s_i$ 和生成 $y_i$ 时的重要性。直观地，这在解码器中实现了**一种注意力机制。解码器决定要关注源句子的哪些部分**。通过让解码器具有注意力机制，我们**减轻了编码器必须将源句子的所有信息编码为固定长度向量的负担**。通过这种新方法，信息可以分布在整个注释序列中，解码器可以相应地选择性检索。

> [!NOTE]
>
> 这是全文关键



### 3.2 编码器：用于注释序列的双向 RNN

通常的 RNN（如公式（1）所述）按顺序读取输入序列 $x$ ，从第一个符号 $x_1$ 到最后一个符号 $x_{T_x}$ 。然而，在提出的方案中，我们希望每个词的注释不仅总结前面的词，还要总结后面的词。因此，我们提议使用双向 RNN（BiRNN[25]），该网络最近在语音识别中得到了成功应用（例如，Graves 等人 [14]）。

BiRNN 由正向和反向 RNN 组成。正向 RNN $\overrightarrow{f}$ 按顺序读取输入序列（从 $x_1$ 到 $x_{T_x}$ ）并计算正向隐藏状态序列 $(\overrightarrow{h}_1, \cdots, \overrightarrow{h}_{T_x})$ 。反向 RNN $\overleftarrow{f}$ 按逆序读取序列（从 $x_{T_x}$ 到 $x_1$ ），得到反向隐藏状态序列 $(\overleftarrow{h}_1, \cdots, \overleftarrow{h}_{T_x})$ 。

我们通过拼接正向隐藏状态 $\overrightarrow{h}_j$ 和反向隐藏状态 $\overleftarrow{h}_j$ 来获得每个词 $x_j$ 的注释，即 $h_j = [\overrightarrow{h}_j^\top; \overleftarrow{h}_j^\top]^\top$ 。通过这种方式，注释 $h_j$ 包含了前面词和后面词的摘要。由于 RNN 倾向于更好地表示最近的输入，**注释 $h_j$ 将聚焦于 $x_j$ 周围的词**。这个注释序列后来被解码器和对齐模型用来计算上下文向量（公式（5）-（6））。

所提出模型的图示见图 1。

<img src="/Users/dazhang/PycharmProject/Papers/1-NLP/.picture/image-20260811161819194.png" alt="image-20260811161819194" style="zoom:33%;" />



## 4 实验设置

我们在英法翻译任务上评估提出的方法。我们使用 ACL WMT '14 提供的**双语并行语料库**[†3]。作为对比，我们还报告了由 [7]最近提出的 RNN 编码器-解码器的性能。我们对两个模型使用相同的训练程序和相同的数据集[†4]。

[^3]: http://www.statmt.org/wmt14/translation-task.html
[^4]: 实现可从 https://github.com/lisa-groundhog/GroundHog 获取。

### 4.1 数据集

WMT '14 包含以下英法并行语料库：Europarl（6100 万词）、news commentary（550 万词）、UN（4.21 亿词）以及两个分别包含 9000 万词和 2.725 亿词的爬取语料库，总计 8.5 亿词。按照 [7]中描述的程序，我们使用 [1]的数据选择方法将合并语料库的大小减少到 3.48 亿词[†5]。除了上述提到的并行语料库外，我们不使用任何单语数据，尽管使用更大的单语语料库来预训练编码器是可能的。我们合并 news-test-2012 和 news-test-2013 作为开发（验证）集，并在 WMT '14 的测试集（news-test-2014）上评估模型，该测试集包含 3003 个未出现在训练数据中的句子。

[^5]: 在线获取：http://www-lium.univ-lemans.fr/~schwenk/cslm_joint_paper/。

在常规的分词处理后[†6]，我们使用**每种语言中最频繁的 30,000 个词的短列表**来训练我们的模型。不在短列表中的任何词都被映射到一个**特殊标记（[UNK]）**。我们没有对数据应用任何其他特殊预处理，如小写化或词干提取。

[^6]: [†6] 我们使用了来自开源机器翻译包 Moses 的分词脚本。



### 4.2 模型

我们训练两种类型的模型。第一种是 RNN 编码器-解码器（RNNencdec，Cho 等人，2014a），另一种是我们提出的模型，称为 RNNsearch。我们训练每个模型两次：首先使用长度不超过 30 个词的句子（RNNencdec-30，RNNsearch-30），然后使用长度不超过 50 个词的句子（RNNencdec-50，RNNsearch-50）。

RNNencdec 的编码器和解码器各有 1000 个隐藏单元[†7]。RNNsearch 的编码器由正向和反向循环神经网络（RNN）组成，每个有 1000 个隐藏单元。其解码器有 1000 个隐藏单元。在两种情况下，我们使用一个具有单个 maxout[11]隐藏层的多层网络来计算每个目标词的条件概率[23]。

> [†7] 在本文中，我们说"隐藏单元"时总是指门控隐藏单元（见附录 A.1.1）。

我们使用小批量随机梯度下降（SGD）算法结合 Adadelta[29]来训练每个模型。每个 SGD 更新方向使用包含 80 个句子的小批量计算。我们训练每个模型大约 5 天。

一旦模型训练完成，我们使用**束搜索**来找到近似最大化条件概率的翻译（例如，[12,6]）。[28]使用这种方法从他们的神经机器翻译模型中生成翻译。

关于实验中使用的模型架构和训练程序的更多细节，请参见附录 A 和 B。



## 5 结果

### 5.1 定量结果

在表 1 中，我们列出了以 BLEU 分数衡量的翻译性能。从表中可以清楚地看出，在所有情况下，提出的 RNNsearch 都优于传统的 RNNencdec。更重要的是，当仅考虑由已知词组成的句子时，RNNsearch 的性能与传统的基于短语的翻译系统（Moses）一样高。这是一个显著的成就，考虑到 Moses 除了我们用于训练 RNNsearch 和 RNNencdec 的并行语料库外，还使用了单独的单语语料库（4.18 亿词）。

<img src="/Users/dazhang/PycharmProject/Papers/1-NLP/.picture/image-20260811161858561.png" alt="image-20260811161858561" style="zoom:50%;" />

图 2 展示了测试集上生成翻译的 BLEU 分数与句子长度的关系。RNNsearch-50、RNNsearch-30、RNNenc-50、RNNenc-30 的曲线分别用不同标记表示。结果是在包含模型未知词的完整测试集上计算的。

**表 1：训练模型在测试集上的 BLEU 分数。** 第二列和第三列分别显示在所有句子上的分数和在自身及参考翻译中都没有未知词的句子上的分数。注意 RNNsearch-50(★) 被训练得更久，直到开发集性能停止提升。（◊）当仅评估没有未知词的句子时（最后一列），我们禁止模型生成 [UNK] 标记。

<img src="/Users/dazhang/PycharmProject/Papers/1-NLP/.picture/image-20260811162040616.png" alt="image-20260811162040616" style="zoom:50%;" />

### 5.2 定性分析

#### 5.2.1 对齐

提出的方法提供了一种直观的方式来检查生成翻译中的词与源句子中的词之间的（软）对齐。这通过可视化公式（6）中的注释权重 $\alpha_{ij}$ 来实现，如图 3 所示。每个图中矩阵的每一行指示与注释相关联的权重。从中我们可以看到，在生成目标词时，源句子中的哪些位置被认为更重要。

<img src="/Users/dazhang/PycharmProject/Papers/1-NLP/.picture/image-20260811162011242.png" alt="image-20260811162011242" style="zoom:33%;" />

从图 3 中的对齐可以看出，**英语和法语之间的词对齐在很大程度上是单调的**。每个矩阵的对角线上我们看到很强的权重。然而，我们也观察到一些**非平凡的、非单调的对齐**。**形容词和名词在法语和英语中的顺序通常是不同的**，我们在图 3(a) 中看到了一个例子。从该图中，我们看到模型正确地将短语 [European Economic Area] 翻译为 [zone économique européen]。RNNsearch 能够正确地将 [zone] 与 [Area] 对齐，跳过了两个词（[European] 和 [Economic]），然后一次回看一个词来完成整个短语 [zone économique européenne]。

软对齐相较于硬对齐的优势，例如，从图 3(d) 中可以明显看出。考虑源短语 [the man] 被翻译为 [l'homme]。任何硬对齐都会将 [the] 映射到 [l']，[man] 映射到 [homme]。这对翻译没有帮助，因为必须考虑 [the] 后面的词才能确定是否应翻译为 [le]、[la]、[les] 或 [l']。我们的软对齐通过让模型同时关注 [the] 和 [man] 自然地解决了这个问题，在这个例子中，我们看到模型能够正确地将 [the] 翻译为 [l']。我们在图 3 的所有展示案例中都观察到类似的行为。**软对齐的另一个好处是它自然地处理源语言和目标语言短语长度不同的情况，而无需一种反直觉的方式将某些词映射到无**（[NULL]）（例如，见 [19] 的第 4 章和第 5 章）。

#### 5.2.2 长句子

从图 2 中可以清晰地看出，提出的模型（RNNsearch）比传统模型（RNNencdec）在**翻译长句子方面**好得多。这很**可能是因为 RNNsearch 不需要将长句子完美地编码为固定长度向量，而只需要准确地编码输入句子中围绕某个特定词的部分。**

<img src="/Users/dazhang/PycharmProject/Papers/1-NLP/.picture/image-20260811161858561.png" alt="image-20260811161858561" style="zoom:50%;" />

例如，考虑测试集中的这个源句子：

> An admitting privilege is the right of a doctor to admit a patient to a hospital or a medical centre to carry out a diagnosis or a procedure, based on his status as a healthcare worker at a hospital.

RNNencdec-50 将这个句子翻译为：

> Un privilège d'admission est le droit d'un médecin de reconnaître un patient à l'hôpital ou un centre médical d'un diagnostic ou de prendre un diagnostic en fonction de son état de santé.

RNNencdec-50 正确翻译了源句子直到 [a medical center]。但是从那里开始（下划线部分），它偏离了源句子的原始含义。例如，它将源句子中的 [based on his status as a healthcare worker at a hospital] 替换为 [en fonction de son état de santé]（"based on his state of health"）。

另一方面，RNNsearch-50 生成了以下正确的翻译，保留了输入句子的完整含义，没有遗漏任何细节：

> Un privilège d'admission est le droit d'un médecin d'admettre un patient à un hôpital ou un centre médical pour effectuer un diagnostic ou une procédure, selon son statut de travailleur de soins de santé à l'hôpital.

让我们考虑测试集中的另一个句子：

> This kind of experience is part of Disney's efforts to "extend the lifetime of its series and build new relationships with audiences via digital platforms that are becoming ever more important," he added.

RNNencdec-50 的翻译是：

> Ce type d'expérience fait partie des initiatives du Disney pour "prolonger la durée de vie de ses nouvelles et de développer des liens avec les lecteurs numériques qui deviennent plus complexes.

与前面的例子一样，RNNencdec 在生成大约 30 个词后开始偏离源句子的实际含义（见下划线短语）。在那之后，翻译质量恶化，出现了诸如缺少闭合引号之类的基本错误。

再次，RNNsearch-50 能够正确翻译这个长句子：

> Ce genre d'expérience fait partie des efforts de Disney pour "prolonger la durée de vie de ses séries et créer de nouvelles relations avec des publics via des plateformes numériques de plus en plus importantes", a-t-il ajouté.

结合已经呈现的定量结果，这些定性观察证实了我们的假设，即 RNNsearch 架构能够比标准 RNNencdec 模型更可靠地翻译长句子。

在附录 C 中，我们提供了由 RNNencdec-50、RNNsearch-50 和 Google 翻译生成的一些长源句子的翻译样本，以及参考翻译。



## 6 相关工作

### 6.1 学习对齐

类似的对齐输出符号与输入符号的方法最近由 [13]在手写合成背景下提出。手写合成是要求模型为给定的字符序列生成手写体的任务。在他的工作中，他使用高斯核的混合来计算注释的权重，其中每个核的位置、宽度和混合系数由对齐模型预测。更具体地说，他的对齐被限制为预测位置使得位置单调递增。

与我们方法的主要区别在于，在[13]中，注释权重的模式只能向一个方向移动。在机器翻译的背景下，这是一个严重的限制，因为（长距离）重排序通常是生成语法正确翻译所必需的（例如，英译德）。

另一方面，我们的方法需要为翻译中的每个词计算源句子中每个词的注释权重。这个缺点在翻译任务中并不严重，因为大多数输入和输出句子只有 15-40 个词。然而，这可能会限制提出的方案在其他任务中的适用性。

### 6.2 用于机器翻译的神经网络

自从 [4]引入神经概率语言模型（该模型使用神经网络在给定固定数量的前序词的条件下建模词的条件概率）以来，神经网络已被广泛用于机器翻译。然而，神经网络的作用在很大程度上仅限于为现有的统计机器翻译系统提供单一特征，或对现有系统提供的候选翻译列表进行重新排序。

例如，[26]提出使用前馈神经网络计算一对源短语和目标短语的分数，并**将该分数作为基于短语的统计机器翻译系统的附加特征**。最近，[18]以及 [9]报告了神经网络作为现有翻译系统**子组件**的成功使用。传统上，作为目标端语言模型训练的神经网络已被用于对候选翻译列表进行重新评分或重新排序（例如，[27]）。

尽管上述方法被证明能够提高现有最先进的机器翻译系统的翻译性能，但我们更感兴趣的是一个更雄心勃勃的目标：设计一个**完全基于神经网络的翻译系统**。因此，我们在本文中考虑的神经机器翻译方法与这些早期工作有着根本的不同。**我们的模型不将神经网络用作现有系统的一部分，而是独立工作，直接从源句子生成翻译。**



## 7 结论

传统的神经机器翻译方法，称为编码器-解码器方法，将整个输入句子编码为一个固定长度的向量，然后从中解码出翻译。我们基于 [8]和 [24]最近的实证研究，推测使用固定长度上下文向量对于翻译长句子是有问题的。

在本文中，我们提出了一种解决这个问题的新架构。我们通过允许模型在生成每个目标词时（软）搜索一组输入词（或由编码器计算的它们的注释）来扩展基础编码器-解码器。这使得模型不再需要将整个源句子编码为固定长度向量，也使得**模型可以只关注与生成下一个目标词相关的信息**。这对神经机器翻译系统在较长句子上获得良好结果的能力产生了重大的积极影响。与传统机器翻译系统不同，翻译系统的所有部分，包括对齐机制，都**朝着更好的生成正确翻译的对数概率进行联合训练**。

我们将提出的模型称为 RNNsearch，并在英法翻译任务上进行了测试。实验表明，提出的 RNNsearch 显著优于传统的编码器-解码器模型（RNNencdec），无论句子长度如何，并且对源句子长度更加鲁棒。通过我们检查 RNNsearch 生成的（软）对齐的定性分析，我们得出结论，模型在生成正确翻译时能够**正确地将每个目标词与源句子中的相关词（或其注释）对齐**。

也许更重要的是，提出的方法取得了与现有 基于短语的统计机器翻译 相当的翻译性能。这是一个引人注目的结果，考虑到提出的架构（或整个神经机器翻译家族）直到今年才被提出。我们相信这里提出的架构是迈向更好的机器翻译和更好地理解自然语言的一个有希望的步骤。

未来面临的挑战之一是更好地处理未知词或稀有词。这将需要使模型得以更广泛地使用，并在所有上下文中匹配当前最先进的机器翻译系统的性能。



## 致谢

作者感谢 Theano 的开发者[5,2]。我们感谢以下机构在研究经费和计算支持方面的支持：NSERC、Calcul Québec、Compute Canada、加拿大研究主席和 CIFAR。Bahdanau 感谢 Planet Intelligent Systems GmbH 的支持。我们还感谢 Felix Hill、Bart van Merriënboer、Jean Pouget-Abadie、Coline Devin 和 Tae-Ho Kim。



## 参考文献

[1] Axelrod, A., He, X., and Gao, J. (2011). Domain adaptation via pseudo in-domain data selection. In *Proceedings of the ACL Conference on Empirical Methods in Natural Language Processing (EMNLP)*, pages 355–362. Association for Computational Linguistics.

[2] Bastien, F., Lamblin, P., Pascanu, R., Bergstra, J., Goodfellow, I. J., Bergeron, A., Bouchard, N., and Bengio, Y. (2012). Theano: new features and speed improvements. Deep Learning and Unsupervised Feature Learning NIPS 2012 Workshop.

[3] Bengio, Y., Simard, P., and Frasconi, P. (1994). Learning long-term dependencies with gradient descent is difficult. *IEEE Transactions on Neural Networks*, 5(2), 157–166.

[4] Bengio, Y., Ducharme, R., Vincent, P., and Janvin, C. (2003). **A neural probabilistic language model.** *J. Mach. Learn. Res.*, 3, 1137–1155.

[5] Bergstra, J., Breuleux, O., Bastien, F., Lamblin, P., Pascanu, R., Desjardins, G., Turian, J., Warde-Farley, D., and Bengio, Y. (2010). Theano: a CPU and GPU math expression compiler. In *Proceedings of the Python for Scientific Computing Conference (SciPy)*. Oral Presentation.

[6] Boulanger-Lewandowski, N., Bengio, Y., and Vincent, P. (2013). Audio chord recognition with recurrent neural networks. In *ISMIR*.

[7] Cho, K., van Merrienboer, B., Gulcehre, C., Bougares, F., Schwenk, H., and Bengio, Y. (2014a). **Learning phrase representations using RNN encoder-decoder for statistical machine translation**. In *Proceedings of the Empiricial Methods in Natural Language Processing (EMNLP 2014)*. to appear.

[8] Cho, K., van Merriënboer, B., Bahdanau, D., and Bengio, Y. (2014b). **On the properties of neural machine translation: Encoder–Decoder approaches**. In *Eighth Workshop on Syntax, Semantics and Structure in Statistical Translation*. to appear.

[9] Devlin, J., Zbib, R., Huang, Z., Lamar, T., Schwartz, R., and Makhoul, J. (2014). Fast and robust neural network joint models for statistical machine translation. In *Association for Computational Linguistics*.

[10] Forcada, M. L. and Ñeco, R. P. (1997). Recursive hetero-associative memories for translation. In J. Mira, R. Moreno-Díaz, and J. Cabestany, editors, *Biological and Artificial Computation: From Neuroscience to Technology*, volume 1240 of *Lecture Notes in Computer Science*, pages 453–462. Springer Berlin Heidelberg.

[11] Goodfellow, I., Warde-Farley, D., Mirza, M., Courville, A., and Bengio, Y. (2013). Maxout networks. In *Proceedings of The 30th International Conference on Machine Learning*, pages 1319–1327.

[12] Graves, A. (2012). Sequence transduction with recurrent neural networks. In *Proceedings of the 29th International Conference on Machine Learning (ICML 2012)*.

[13] Graves, A. (2013). Generating sequences with recurrent neural networks. *arXiv:1308.0850 [cs.NE]*.

[14] Graves, A., Jaitly, N., and Mohamed, A.-R. (2013). Hybrid speech recognition with deep bidirectional LSTM. In *Automatic Speech Recognition and Understanding (ASRU), 2013 IEEE Workshop on*, pages 273–278.

[15] Hermann, K. and Blunsom, P. (2014). Multilingual distributed representations without word alignment. In *Proceedings of the Second International Conference on Learning Representations (ICLR 2014)*.

[16] Hochreiter, S. (1991). Untersuchungen zu dynamischen neuronalen Netzen. Diploma thesis, Institut für Informatik, Lehrstuhl Prof. Brauer, Technische Universität München.

[17] Hochreiter, S. and Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735–1780.

[18] Kalchbrenner, N. and Blunsom, P. (2013). Recurrent continuous translation models. In *Proceedings of the ACL Conference on Empirical Methods in Natural Language Processing (EMNLP)*, pages 1700–1709. Association for Computational Linguistics.

[19] Koehn, P. (2010). *Statistical Machine Translation*. Cambridge University Press, New York, NY, USA.

[20] Koehn, P., Och, F. J., and Marcu, D. (2003). Statistical phrase-based translation. In *Proceedings of the 2003 Conference of the North American Chapter of the Association for Computational Linguistics on Human Language Technology - Volume 1*, NAACL '03, pages 48–54, Stroudsburg, PA, USA. Association for Computational Linguistics.

[21] Pascanu, R., Mikolov, T., and Bengio, Y. (2013a). On the difficulty of training recurrent neural networks. In *ICML'2013*.

[22] Pascanu, R., Mikolov, T., and Bengio, Y. (2013b). On the difficulty of training recurrent neural networks. In *Proceedings of the 30th International Conference on Machine Learning (ICML 2013)*.

[23] Pascanu, R., Gulcehre, C., Cho, K., and Bengio, Y. (2014). How to construct deep recurrent neural networks. In *Proceedings of the Second International Conference on Learning Representations (ICLR 2014)*.

[24] Pouget-Abadie, J., Bahdanau, D., van Merriënboer, B., Cho, K., and Bengio, Y. (2014). Overcoming the curse of sentence length for neural machine translation using automatic segmentation. In *Eighth Workshop on Syntax, Semantics and Structure in Statistical Translation*. to appear.

[25] Schuster, M. and Paliwal, K. K. (1997). Bidirectional recurrent neural networks. *Signal Processing, IEEE Transactions on*, 45(11), 2673–2681.

[26] Schwenk, H. (2012). Continuous space translation models for phrase-based statistical machine translation. In M. Kay and C. Boitet, editors, *Proceedings of the 24th International Conference on Computational Linguistics (COLING)*, pages 1071–1080. Indian Institute of Technology Bombay.

[27] Schwenk, H., Dchelotte, D., and Gauvain, J.-L. (2006). Continuous space language models for statistical machine translation. In *Proceedings of the COLING/ACL on Main conference poster sessions*, pages 723–730. Association for Computational Linguistics.

[28] Sutskever, I., Vinyals, O., and Le, Q. (2014). **Sequence to sequence learning with neural networks**. In *Advances in Neural Information Processing Systems (NIPS 2014)*.

[29] Zeiler, M. D. (2012). ADADELTA: An adaptive learning rate method. *arXiv:1212.5701 [cs.LG]*.

---



## 附录 A 模型架构

### A.1 架构选择

第 3 节中提出的方案是一个通用框架，在这个框架中可以自由定义循环神经网络（RNN）的激活函数 $f$ 和对齐模型 $a$ 等。这里，我们描述在本文实验中做出的选择。

#### A.1.1 循环神经网络

对于 RNN 的激活函数 $f$ ，我们使用 Cho 等人（2014a）最近提出的门控隐藏单元。门控隐藏单元是传统简单单元（如逐元素 tanh）的替代方案。这种门控单元与 Hochreiter 和 Schmidhuber（1997）早期提出的长短期记忆（LSTM）单元类似，共享其更好地建模和学习长期依赖的能力。这是通过在展开的 RNN 中拥有导数乘积接近 1 的计算路径来实现的。这些路径允许梯度轻松地反向流动，而不会受到太多梯度消失效应的影响（Hochreiter，1991；Bengio 等人，1994；Pascanu 等人，2013a）。因此，也可以使用本文描述的门控隐藏单元替代 LSTM 单元，正如 Sutskever 等人（2014）在类似背景下所做的那样。

采用 $n$ 个门控隐藏单元[†8]的 RNN 的新状态 $s_i$ 由下式计算：

$$
s_i = f(s_{i-1}, y_{i-1}, c_i) = (1 - z_i) \circ s_{i-1} + z_i \circ \tilde{s}_i,
$$

其中 $\circ$ 是逐元素乘法， $z_i$ 是更新门的输出（见下文）。提出的更新状态 $\tilde{s}_i$ 由下式计算：

$$
\tilde{s}_i = \tanh(W e(y_{i-1}) + U [r_i \circ s_{i-1}] + C c_i),
$$

其中 $e(y_{i-1}) \in \mathbb{R}^m$ 是词 $y_{i-1}$ 的 $m$ 维嵌入， $r_i$ 是重置门的输出（见下文）。当 $y_i$ 表示为 1-of-K 向量时， $e(y_i)$ 仅仅是嵌入矩阵 $E \in \mathbb{R}^{m \times K}$ 的一列。只要可能，我们省略偏置项以使公式更简洁。

更新门 $z_i$ 允许每个隐藏单元维持其先前激活，重置门 $r_i$ 控制从先前状态中应该重置多少以及什么信息。我们通过以下方式计算它们：

$$
\begin{aligned}
z_i &= \sigma(W_z e(y_{i-1}) + U_z s_{i-1} + C_z c_i), \\
r_i &= \sigma(W_r e(y_{i-1}) + U_r s_{i-1} + C_r c_i),
\end{aligned}
$$

其中 $\sigma(\cdot)$ 是逻辑 sigmoid 函数。

在解码器的每一步，我们计算输出概率（公式（4））作为一个多层函数（Pascanu 等人，2014）。我们使用一个单层 maxout 单元（Goodfellow 等人，2013）的隐藏层，并使用 softmax 函数归一化输出概率（每个词一个）（见公式（6））。

[^8]: 这里我们展示了解码器的公式。同样的公式可以通过忽略上下文向量 $c_i$ 和相关项用于编码器。



#### A.1.2 对齐模型

对齐模型的设计需要考虑模型需要对每个句子对（长度分别为 $T_x$ 和 $T_y$ ）计算 $T_x \times T_y$ 次。为了减少计算，我们使用单层多层感知机，使得：

$$
a(s_{i-1}, h_j) = v_a^\top \tanh(W_a s_{i-1} + U_a h_j),
$$

其中 $W_a \in \mathbb{R}^{n \times n}$ ， $U_a \in \mathbb{R}^{n \times 2n}$ ， $v_a \in \mathbb{R}^n$ 是权重矩阵。由于 $U_a h_j$ 不依赖于 $i$ ，我们可以提前预计算以最小化计算成本。

### A.2 模型详细描述

#### A.2.1 编码器

在本节中，我们详细描述实验（见第 4-5 节）中使用的提出模型（RNNsearch）的架构。从现在开始，为了增加可读性，我们省略所有偏置项。

模型接收一个 1-of-K 编码的词向量形式的源句子作为输入：

$$
x = (x_1, \ldots, x_{T_x}), \quad x_i \in \mathbb{R}^{K_x}
$$

并输出一个 1-of-K 编码的词向量形式的翻译句子：

$$
y = (y_1, \ldots, y_{T_y}), \quad y_i \in \mathbb{R}^{K_y},
$$

其中 $K_x$ 和 $K_y$ 分别是源语言和目标语言的词汇表大小。 $T_x$ 和 $T_y$ 分别表示源句子和目标句子的长度。

首先，计算双向循环神经网络（BiRNN）的正向状态：

$$
\overrightarrow{h}_i =
\begin{cases}
(1 - \overrightarrow{z}_i) \circ \overrightarrow{h}_{i-1} + \overrightarrow{z}_i \circ \overrightarrow{\tilde{h}}_i, & \text{if } i > 0 \\
0, & \text{if } i = 0
\end{cases}
$$

其中

$$
\begin{aligned}
\overrightarrow{\tilde{h}}_i &= \tanh(\overrightarrow{W} E x_i + \overrightarrow{U} [\overrightarrow{r}_i \circ \overrightarrow{h}_{i-1}]) \\
\overrightarrow{z}_i &= \sigma(\overrightarrow{W}_z E x_i + \overrightarrow{U}_z \overrightarrow{h}_{i-1}) \\
\overrightarrow{r}_i &= \sigma(\overrightarrow{W}_r E x_i + \overrightarrow{U}_r \overrightarrow{h}_{i-1}).
\end{aligned}
$$

 $E \in \mathbb{R}^{m \times K_x}$ 是词嵌入矩阵。 $\overrightarrow{W}, \overrightarrow{W}_z, \overrightarrow{W}_r \in \mathbb{R}^{n \times m}$ ， $\overrightarrow{U}, \overrightarrow{U}_z, \overrightarrow{U}_r \in \mathbb{R}^{n \times n}$ 是权重矩阵。 $m$ 和 $n$ 分别是词嵌入维度和隐藏单元数量。 $\sigma(\cdot)$ 照例是逻辑 sigmoid 函数。

反向状态 $(\overleftarrow{h}_1, \cdots, \overleftarrow{h}_{T_x})$ 的计算类似。我们与权重矩阵不同，在正向和反向 RNN 之间共享词嵌入矩阵 $E$ 。

我们拼接正向和反向状态以获得注释 $(h_1, h_2, \cdots, h_{T_x})$ ，其中：

$$
h_i = \begin{bmatrix}
\overrightarrow{h}_i \\
\overleftarrow{h}_i
\end{bmatrix} \qquad (7)
$$

#### A.2.2 解码器

给定编码器的注释，解码器的隐藏状态 $s_i$ 由下式计算：

$$
s_i = (1 - z_i) \circ s_{i-1} + z_i \circ \tilde{s}_i,
$$

其中

$$
\begin{aligned}
\tilde{s}_i &= \tanh(W E y_{i-1} + U [r_i \circ s_{i-1}] + C c_i) \\
z_i &= \sigma(W_z E y_{i-1} + U_z s_{i-1} + C_z c_i) \\
r_i &= \sigma(W_r E y_{i-1} + U_r s_{i-1} + C_r c_i)
\end{aligned}
$$

 $E$ 是目标语言的词嵌入矩阵。 $W, W_z, W_r \in \mathbb{R}^{n \times m}$ ， $U, U_z, U_r \in \mathbb{R}^{n \times n}$ ，以及 $C, C_z, C_r \in \mathbb{R}^{n \times 2n}$ 是权重。同样， $m$ 和 $n$ 分别是词嵌入维度和隐藏单元数量。初始隐藏状态 $s_0$ 由 $s_0 = \tanh(W_s \overleftarrow{h}_1)$ 计算，其中 $W_s \in \mathbb{R}^{n \times n}$ 。

上下文向量 $c_i$ 在每一步由对齐模型重新计算：

$$
c_i = \sum_{j=1}^{T_x} \alpha_{ij} h_j, \qquad (8)
$$

其中

$$
\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^{T_x} \exp(e_{ik})}
$$

$$
e_{ij} = v_a^\top \tanh(W_a s_{i-1} + U_a h_j),
$$

 $h_j$ 是源句子中的第 $j$ 个注释（见公式（7））。 $v_a \in \mathbb{R}^{n'}$ ， $W_a \in \mathbb{R}^{n' \times n}$ 和 $U_a \in \mathbb{R}^{n' \times 2n}$ 是权重矩阵。注意，如果我们将 $c_i$ 固定为 $\overrightarrow{h}_{T_x}$ ，则该模型变为 RNN 编码器-解码器（Cho 等人，2014a）。

使用解码器状态 $s_{i-1}$ 、上下文 $c_i$ 和最后生成的词 $y_{i-1}$ ，我们定义目标词 $y_i$ 的概率为：

$$
p(y_i \mid s_i, y_{i-1}, c_i) \propto \exp \left( y_i^\top W_o t_i \right),
$$

其中

$$
t_i = \left[ \max \{ \tilde{t}_{i,2j-1}, \tilde{t}_{i,2j} \} \right]_{j=1, \ldots, l}^\top
$$

且 $\tilde{t}_{i,k}$ 是向量 $\tilde{t}_i$ 的第 $k$ 个元素，由下式计算：

$$
\tilde{t}_i = U_o s_{i-1} + V_o E y_{i-1} + C_o c_i.
$$

 $W_o \in \mathbb{R}^{K_y \times l}$ ， $U_o \in \mathbb{R}^{2l \times n}$ ， $V_o \in \mathbb{R}^{2l \times m}$ 和 $C_o \in \mathbb{R}^{2l \times 2n}$ 是权重矩阵。这可以理解为具有一个单层 maxout 隐藏层（Goodfellow 等人，2013）的深度输出（Pascanu 等人，2014）。

#### A.2.3 模型大小

对于本文中使用的所有模型，隐藏层大小 $n$ 为 1000，词嵌入维度 $m$ 为 620，深度输出中 maxout 隐藏层的大小 $l$ 为 500。对齐模型中隐藏单元的数量 $n'$ 为 1000。



## 附录 B 训练过程

### B.1 参数初始化

我们将循环权重矩阵 $\overrightarrow{U}, \overrightarrow{U}_z, \overrightarrow{U}_r, \overleftarrow{U}, \overleftarrow{U}_z, \overleftarrow{U}_r, U, U_z, U_r$ 初始化为随机正交矩阵。对于 $W_a$ 和 $U_a$ ，我们通过从均值为 0、方差为 $0.001^2$ 的高斯分布中采样每个元素来初始化它们。 $V_a$ 的所有元素和所有偏置向量都初始化为零。任何其他权重矩阵均通过从均值为 0、方差为 $0.01^2$ 的高斯分布中采样来初始化。

### B.2 训练

我们使用随机梯度下降（SGD）算法。使用 Adadelta（Zeiler，2012）来自动调整每个参数的学习率（ $\epsilon = 10^{-6}$ ， $\rho = 0.95$ ）。我们每次显式地将成本函数梯度的 $L_2$ 范数归一化为最多为预定义的阈值 1（当范数大于阈值时）（Pascanu 等人，2013b）。每个 SGD 更新方向使用包含 80 个句子的小批量计算。

在我们的实现中，每次更新的时间与小批量中最长句子的长度成正比。因此，为了最小化计算浪费，每 20 次更新之前，我们检索 1600 个句子对，根据长度排序，并将它们分成 20 个小批量。训练数据在训练前打乱一次，并按此方式顺序遍历。

在表 2 中，我们展示了与训练实验中所有模型相关的统计数据。

**表 2：学习统计和相关数据。** 每次更新对应于使用单次小批量更新一次参数。一个 epoch 是一次遍历训练集。NLL 是训练集或开发集中句子的平均条件对数概率。注意句子的长度不同。

<img src="/Users/dazhang/PycharmProject/Papers/1-NLP/.picture/image-20260811162116670.png" alt="image-20260811162116670" style="zoom:33%;" />



## 附录 C 长句子的翻译

**表 3：由 RNNenc-50 和 RNNsearch-50 从测试集中选取的长源句子（30 词及以上）生成的翻译。** 对于每个源句子，我们还展示了黄金标准翻译。Google 翻译的翻译是在 2014 年 8 月 27 日进行的。

**源句子 1：**
> An admitting privilege is the right of a doctor to admit a patient to a hospital or a medical centre to carry out a diagnosis or a procedure, based on his status as a healthcare worker at a hospital.

**参考翻译：**
> Le privilège d'admission est le droit d'un médecin, en vertu de son statut de membre soignant d'un hôpital, d'admettre un patient dans un hôpital ou un centre médical afin d'y délivrer un diagnostic ou un traitement.

**RNNenc-50：**
> Un privilège d'admission est le droit d'un médecin de reconnaître un patient à l'hôpital ou un centre médical d'un diagnostic ou de prendre un diagnostic en fonction de son état de santé.

**RNNsearch-50：**
> Un privilège d'admission est le droit d'un médecin d'admettre un patient à un hôpital ou un centre médical pour effectuer un diagnostic ou une procédure, selon son statut de travailleur de soins de santé à l'hôpital.

**Google 翻译：**
> Un privilège admettre est le droit d'un médecin d'admettre un patient dans un hôpital ou un centre médical pour effectuer un diagnostic ou une procédure, fondé sur sa situation en tant que travailleur de soins de santé dans un hôpital.

---

**源句子 2：**
> This kind of experience is part of Disney's efforts to "extend the lifetime of its series and build new relationships with audiences via digital platforms that are becoming ever more important," he added.

**参考翻译：**
> Ce type d'expérience entre dans le cadre des efforts de Disney pour "étendre la durée de vie de ses séries et construire de nouvelles relations avec son public grâce à des plateformes numériques qui sont de plus en plus importantes", a-t-il ajouté.

**RNNenc-50：**
> Ce type d'expérience fait partie des initiatives du Disney pour "prolonger la durée de vie de ses nouvelles et de développer des liens avec les lecteurs numériques qui deviennent plus complexes".

**RNNsearch-50：**
> Ce genre d'expérience fait partie des efforts de Disney pour "prolonger la durée de vie de ses séries et créer de nouvelles relations avec des publics via des plateformes numériques de plus en plus importantes", a-t-il ajouté.

**Google 翻译：**
> Ce genre d'expérience fait partie des efforts de Disney à "étendre la durée de vie de sa série et construire de nouvelles relations avec le public par le biais des plates-formes numériques qui deviennent de plus en plus important", at-il ajouté.

---

**源句子 3：**
> In a press conference on Thursday, Mr Blair stated that there was nothing in this video that might constitute a "reasonable motive" that could lead to criminal charges being brought against the mayor.

**参考翻译：**
> En conférence de presse, jeudi, M. Blair a affirmé qu'il n'y avait rien dans cette vidéo qui puisse constituer des "motifs raisonnables" pouvant mener au dépôt d'une accusation criminelle contre le maire.

**RNNenc-50：**
> Lors de la conférence de presse de jeudi, M. Blair a dit qu'il n'y avait rien dans cette vidéo qui pourrait constituer une "motivation raisonnable" pouvant entraîner des accusations criminelles portées contre le maire.

**RNNsearch-50：**
> Lors d'une conférence de presse jeudi, M. Blair a déclaré qu'il n'y avait rien dans cette vidéo qui pourrait constituer un "motif raisonnable" qui pourrait conduire à des accusations criminelles contre le maire.

**Google 翻译：**
> Lors d'une conférence de presse jeudi, M. Blair a déclaré qu'il n'y avait rien dans cette vidéo qui pourrait constituer un "motif raisonnable" qui pourrait mener à des accusations criminelles portes contre le maire.
