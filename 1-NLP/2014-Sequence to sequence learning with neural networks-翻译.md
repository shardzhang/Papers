# Sequence to Sequence Learning with Neural Networks

> **Ilya Sutskever**, Google | Oriol Vinyals, Google | Quoc V. Le, Google



本文提出了一种通用的 **端到端序列学习方法**，该方法 **对序列结构的假设极少**。核心内容：

- 使用**多层长短期记忆网络**（LSTM）将输入序列映射到固定维度的向量，再用另一个深度LSTM从该向量解码目标序列
- 在WMT'14英语到法语翻译任务上，LSTM的BLEU分数达到34.8，超过了**基于短语的SMT系统**
- 发现 **反转源句子**（但不反转目标句子）的词序能显著提升LSTM性能

关键发现：
- 深度LSTM能够学习**对词序敏感**、对主动语态和被动语态相对不变的合理短语和句子表示
- 该方法首次证明 **纯神经翻译系统** 在大规模机器翻译任务上能显著超越 **基于短语的SMT** 基线系统
- **反转词序** 引入了许多 **短期依赖关系**，使优化问题更容易解决

---



## 摘要

深度神经网络（Deep Neural Networks, DNNs）是强大的模型，在困难的学习任务上取得了优异的性能。尽管DNNs在有大量标注训练集的情况下表现良好，但它们无法用于 **将序列映射到序列**。在本文中，我们提出了一种通用的端到端序列学习方法，该方法对序列结构的假设极少。我们的**方法使用多层长短期记忆网络（Long Short-Term Memory, LSTM）将输入序列映射到 固定维度的向量，然后使用另一个深度LSTM从该向量解码目标序列。**我们的主要结果是，在WMT'14数据集的英语到法语翻译任务上，LSTM生成的翻译在整个测试集上达到了34.8的BLEU分数，其中LSTM的BLEU分数 因 **词汇表外单词** 而受到惩罚。此外，LSTM在处理长句时没有遇到困难。相比之下，基于短语的SMT系统在同一数据集上达到了33.3的BLEU分数。当我们使用LSTM对上述SMT系统产生的1000个假设进行重排序时，其BLEU分数提高到36.5，接近该任务之前的最佳结果。LSTM还学习到了合理的短语和句子表示，这些表示对词序敏感，且对主动语态和被动语态相对不变。最后，我们发现反转所有源句子（但不反转目标句子）中的词序能显著提升LSTM的性能，因为这样做在源句和目标句之间引入了许多短期依赖关系，使优化问题更容易解决。



## 1 引言

深度神经网络（Deep Neural Networks, DNNs）是极其强大的机器学习模型，在诸如语音识别[13, 7]和视觉目标识别[19, 6, 21, 20]等困难问题上取得了优异的性能。**DNNs之所以强大，是因为它们能够在适度的步骤内执行任意 并行计算。**DNNs能力的一个惊人例子是它们能够仅使用两个二次大小的隐藏层来对 $N$ 个 $N$ 位数进行排序[27]。因此，虽然神经网络与传统统计模型相关，但它们学习的是一种复杂的计算。此外，只要有监督训练集有足够的信息来指定网络参数，就可以使用有监督反向传播来训练大型DNNs。因此，如果存在一个大型DNN的参数设置能够取得良好的结果（例如，因为人类可以非常快速地解决该任务），有监督反向传播将找到这些参数并解决问题。

尽管DNNs具有灵活性和强大的能力，但它们**只能应用于 输入 和 目标 可以用固定维度向量合理编码的问题**。这是一个重要的局限性，因为许多重要的问题最适合 **用长度事先未知的序列** 来表达。例如，**语音识别 和 机器翻译都是序列问题**。同样，问答也可以看作是将代表问题的单词序列映射到代表答案的单词序列。因此，很明显，一种能够学习将序列映射到序列的 **领域无关方法** 将非常有用。

**序列对DNNs提出了挑战，因为它们要求输入和输出的维度是已知且固定的**。在本文中，我们展示了长短期记忆（Long Short-Term Memory, LSTM）架构[16]的直接应用可以解决通用的序列到序列问题。其思想是使用一个LSTM逐时间步读取输入序列，以获得大的固定维度向量表示，然后使用另一个LSTM从该向量中提取输出序列（图1）。第二个LSTM本质上是一个循环神经网络语言模型[28, 23, 30]，只不过它是**以输入序列作为条件**。**LSTM能够成功学习具有长期时间依赖关系的数据，这使其成为此应用的自然选择，因为输入和对应输出之间存在 相当大的时间滞后（图1）**。

已有许多相关尝试使用神经网络来解决通用的序列到序列学习问题。我们的方法与Kalchbrenner和Blunsom [18]密切相关，他们是第一个将整个输入句子映射到向量的人，也与Cho等人[5]相关，尽管后者仅用于对**基于短语的系统**产生的假设进行重评分。Graves [10]引入了一种新颖的可微注意力机制，允许神经网络关注其输入的不同部分，这一思想的优雅变体被Bahdanau等人[2]成功应用于机器翻译。联结主义序列分类（Connectionist Sequence Classification）是另一种流行的使用神经网络将序列映射到序列的技术，但它假设输入和输出之间存在单调对齐[11]。

<img src="/Users/dazhang/PycharmProject/Papers/1-NLP/.picture/image-20260812181920118.png" alt="image-20260812181920118" style="zoom: 33%;" />

>  图1：我们的模型读取输入句子“ABC”并产生“WXYZ”作为输出句子。模型在输出句子**结束标记**后停止预测。注意LSTM以相反顺序读取输入句子，因为这样做在数据中引入了**许多短期依赖关系**，使优化问题更容易解决。

这项工作的主要结果如下。在WMT'14英语到法语翻译任务上，我们通过使用简单的从左到右束搜索解码器从5个深度LSTM的集合（每个LSTM有3.84亿参数和8000维状态）中直接提取翻译，获得了34.81的BLEU分数。这是迄今为止使用大型神经网络直接翻译所取得的最佳结果。相比之下，该数据集上SMT基线的BLEU分数为33.30 [29]。34.81的BLEU分数是由一个词汇量为80k的LSTM实现的，因此每当参考翻译包含这80k词汇未覆盖的单词时，分数就会受到惩罚。这一结果表明，**一个相对 未优化的小词汇量 神经网络架构（有很大的改进空间）超越了 基于短语的SMT系统。**

最后，我们使用LSTM对相同任务[29]上公开可用的SMT基线1000-best列表进行重评分。通过这样做，我们获得了36.5的BLEU分数，比基线提高了3.2个BLEU点，接近该任务之前发布的最佳结果（37.0 [9]）。令人惊讶的是，尽管其他研究人员最近使用相关架构的经验[26]表明可能会出现问题，但LSTM在处理非常长的句子时并未受到影响。**我们能够在长句上表现良好，是因为我们在训练和测试集中反转了源句子但不反转目标句子中的词序**。通过这样做，我们引入了许多短期依赖关系，使优化问题变得简单得多（见第2节和3.3节）。因此，SGD能够学习到在处理长句时没有问题的LSTMs。**反转源句子中词序 的简单技巧 是这项工作的关键技术贡献之一。**

**LSTM的一个有用的特性是它能够 学习将可变长度的输入句子 映射 到固定维度的向量表示**。鉴于翻译往往是源句子的释义，翻译目标鼓励LSTM找到捕捉句子意义的句子表示，因为意义相近的句子彼此接近，而不同句子的意义则相距较远。定性评估支持这一说法，表明我们的模型对词序敏感，且对主动语态和被动语态相对不变。



## 2 模型

循环神经网络（Recurrent Neural Network, RNN）[31, 28]是 **前馈神经网络在序列上的自然推广**。给定一个输入序列 $(x_1, \ldots, x_T)$，**标准RNN**通过迭代以下方程计算一个输出序列 $(y_1, \ldots, y_T)$：

$$
h_t = \text{sigm}(W_{hx} x_t + W_{hh} h_{t-1})
$$

$$
y_t = W_{yh} h_t
$$

**RNN可以轻松地将序列映射到序列，只要 输入和输出之间的对齐 是预先已知的**。然而，如何将RNN应用于输入和输出序列长度不同 且具有 **复杂非单调关系** 的问题尚不清楚。

通用序列学习最简单的策略是使用一个RNN将输入序列映射到固定大小的向量，然后使用另一个RNN将该向量映射到目标序列（Cho等人[5]也采用了这种方法）。**虽然这在原则上是可行的，因为RNN提供了所有相关信息，但由于由此产生的长期依赖关系（图1）[14, 4, 16, 15]，训练RNN将非常困难。**然而，长短期记忆网络（Long Short-Term Memory, LSTM）[16]已知能够学习具有长期时间依赖关系的问题，因此LSTM可能在此设置下取得成功。

LSTM的目标是估计条件概率 $p(y_1, \ldots, y_{T'} | x_1, \ldots, x_T)$，其中 $(x_1, \ldots, x_T)$ 是输入序列，$y_1, \ldots, y_{T'}$ 是其对应的输出序列，其长度 $T'$ 可能与 $T$ 不同。LSTM通过首先获得输入序列 $(x_1, \ldots, x_T)$ 的固定维度表示 $v$（**由LSTM的 最后一个隐藏状态 给出**），然后使用标准**LSTM-LM**公式计算 $y_1, \ldots, y_{T'}$ 的概率（其初始隐藏状态设置为 $x_1, \ldots, x_T$ 的表示 $v$）来计算此条件概率：

$$
p(y_1, \ldots, y_{T'} | x_1, \ldots, x_T) = \prod_{t=1}^{T'} p(y_t | v, y_1, \ldots, y_{t-1}) \qquad (1)
$$

在此方程中，每个 $p(y_t | v, y_1, \ldots, y_{t-1})$ 分布由词汇表中所有单词上的softmax表示。我们使用Graves [10]中的LSTM公式。注意我们要求每个句子以特殊的句子结束符号“<EOS>”结尾，这**使得模型能够定义所有可能长度序列上的分布**。整体方案如图1所示，其中显示的LSTM计算“A”、“B”、“C”、“<EOS>”的表示，然后使用该表示计算“W”、“X”、“Y”、“Z”、“<EOS>”的概率。

我们的实际模型与上述描述在三个方面有重要不同。首先，我们使用了**两个不同的LSTM**：一个用于输入序列，另一个用于输出序列，因为这样做在计算成本可忽略不计的情况下增加了模型参数数量，并且可以自然地同时训练多个语言对的LSTM [18]。其次，我们发现深度LSTM显著优于浅层LSTM，因此我们选择了**具有四层的LSTM**。第三，我们发现**反转输入句子的词序极其有价值**。因此，例如，LSTM被要求将句子 $c, b, a$ 映射到 $\alpha, \beta, \gamma$，而不是将句子 $a, b, c$ 映射到 $\alpha, \beta, \gamma$，其中 $\alpha, \beta, \gamma$ 是 $a, b, c$ 的翻译。这样，$a$ 与 $\alpha$ 接近，$b$ 与 $\beta$ 相当接近，依此类推，**这一事实使得SGD容易在输入和输出之间“建立通信”**。我们发现这种简单的数据转换大大提高了LSTM的性能。



## 3 实验

我们将我们的方法应用于WMT'14英语到法语机器翻译任务，采用两种方式。我们用它来直接翻译输入句子，不使用参考SMT系统，并且用它来**对SMT基线的n-best列表进行重评分**。我们报告了这些翻译方法的准确性，展示了示例翻译，并可视化了所得的句子表示。

### 3.1 数据集详情

我们使用了WMT'14英语到法语数据集。我们在一个包含1200万句子（包含3.48亿法语单词和3.04亿英语单词）的子集上训练我们的模型，这是来自[29]的一个干净的“精选”子集。我们选择这个翻译任务和这个特定的训练集子集，是因为**公开提供了分词的训练集和测试集以及来自基线SMT的1000-best列表**[29]。

由于典型的神经语言模型依赖于 **每个单词的向量表示**，我们为两种语言使用了**固定的词汇表**。我们为源语言使用了160,000个最频繁的单词，为目标语言使用了80,000个最频繁的单词。每个词汇表外的单词都被替换为一个特殊的“UNK”token。

### 3.2 解码和重评分

我们实验的核心是在 许多**句子对** 上训练一个大型深度LSTM。我们通过最大化给定源句子 $S$ 的正确翻译 $T$ 的 **对数概率** 来训练它，因此训练目标是：

$$
\frac{1}{|S|} \sum_{(T,S) \in S} \log p(T|S)
$$

其中 $S$ 是训练集。一旦训练完成，我们通过找到LSTM中**最可能的翻译来生成翻译**：

$$
\hat{T} = \arg\max_T p(T|S) \qquad (2)
$$

我们使用简单的 **从左到右束搜索解码器** 来搜索最可能的翻译，该解码器维护少量 $B$ 个部分假设，其中部分假设是某个翻译的前缀。在每个时间步，我们用词汇表中的每个可能单词扩展束中的每个部分假设。这大大增加了假设的数量，因此我们只保留根据模型对数概率最可能的 $B$ 个假设。一旦“<EOS>”符号被附加到假设中，它就从束中移除并添加到完整假设的集合中。虽然这个解码器是近似的，但它实现简单。有趣的是，即使束大小为1，我们的系统也表现良好，束大小为2提供了束搜索的大部分好处（表1）。

> [!NOTE]
>
> TODO

我们还使用LSTM对基线系统[29]产生的1000-best列表进行重评分。为了对n-best列表进行重评分，我们使用我们的LSTM计算每个假设的对数概率，并将其分数与LSTM分数进行均匀平均。

### 3.3 反转源句子

虽然LSTM能够解决具有长期依赖关系的问题，但我们发现当源句子被反转时（目标句子不反转），LSTM学习得更好。通过这样做，LSTM的测试困惑度从5.8下降到4.7，其解码翻译的测试BLEU分数从25.9提高到30.6。

虽然我们没有对此现象的完整解释，但我们相信这是由数据集中引入的许多短期依赖关系引起的。通常，当我们连接源句子和目标句子时，源句子中的每个单词都远离其在目标句子中的对应单词。因此，该问题具有很大的“最小时间滞后”[17]。通过反转源句子中的词序，源语言和目标语言中对应单词之间的平均距离保持不变。然而，源语言中的前几个单词现在非常接近目标语言中的前几个单词，因此问题的最小时间滞后大大减少。因此，反向传播更容易在源句子和目标句子之间“建立通信”，这反过来导致整体性能显著提高。

最初，我们相信反转输入句子只会导致目标句子前半部分的预测更自信，后半部分的预测更不自信。然而，在反转源句子上训练的LSTM在长句上的表现远好于在原始源句子上训练的LSTM（见3.7节），这表明反转输入句子使LSTM具有更好的记忆利用率。

### 3.4 训练详情

我们发现LSTM模型相当容易训练。我们使用了具有4层的深度LSTM，每层1000个单元和1000维词嵌入，输入词汇量为160,000，输出词汇量为80,000。因此，深度LSTM使用8000个实数来表示一个句子。我们发现深度LSTM显著优于浅层LSTM，每增加一层困惑度降低近10%，这可能是由于它们更大的隐藏状态。我们在每个输出上使用80,000个单词上的朴素softmax。所得的LSTM有3.84亿参数，其中6400万是纯循环连接（3200万用于“编码器”LSTM，3200万用于“解码器”LSTM）。完整的训练详情如下：

- 我们用-0.08到0.08之间的均匀分布初始化所有LSTM参数
- 我们使用不带动量的随机梯度下降，固定学习率为0.7。5个epoch后，我们开始每半个epoch将学习率减半。我们总共训练了7.5个epoch
- 我们使用大小为128的序列批次计算梯度，并除以批次大小（即128）
- 虽然LSTM往往不会受到梯度消失问题的影响，但它们可能会遇到梯度爆炸。因此，我们通过对梯度范数进行缩放来强制执行硬约束[10, 25]，当其范数超过阈值时。对于每个训练批次，我们计算 $s = \|g\|_2$，其中 $g$ 是除以128的梯度。如果 $s > 5$，我们设置 $g = \frac{5g}{s}$
- 不同的句子具有不同的长度。大多数句子很短（例如，长度20-30），但有些句子很长（例如，长度>100），因此由128个随机选择的训练句子组成的小批次将包含许多短句和很少的长句，结果，小批次中的大部分计算被浪费了。为了解决这个问题，我们确保小批次中的所有句子长度大致相同，从而获得2倍的加速

### 3.5 并行化

具有上一节配置的深度LSTM的C++实现在单个GPU上处理速度约为每秒1,700个单词。这对于我们的目的来说太慢了，因此我们使用8-GPU机器对模型进行了并行化。LSTM的每一层在不同的GPU上执行，并在计算完成后立即将其激活值传递给下一个GPU/层。我们的模型有4层LSTM，每一层都驻留在单独的GPU上。其余4个GPU用于并行化softmax，因此每个GPU负责乘以一个 $1000 \times 20000$ 矩阵。所得的实现达到了每秒6,300个单词（英语和法语）的速度，小批次大小为128。使用此实现训练大约需要十天。

### 3.6 实验结果

我们使用区分大小写的BLEU分数[24]来评估翻译质量。我们使用multi-bleu.pl1在分词的预测和真实值上计算BLEU分数。这种评估BELU分数的方式与[5]和[2]一致，并重现了[29]的33.3分。

然而，如果我们以这种方式评估最佳WMT'14系统[9]（其预测可以从statmt.org\matrix下载），我们得到37.0，大于statmt.org\matrix报告的35.8。

结果如表1和表2所示。我们最好的结果来自具有不同随机初始化和小批次随机顺序的LSTM集成。虽然LSTM集成的解码翻译没有超过最佳WMT'14系统，但这是纯神经翻译系统首次在大规模机器翻译任务上显著超越基于短语的SMT基线系统，尽管其无法处理词汇表外的单词。如果LSTM用于对基线系统的1000-best列表进行重评分，则其与最佳WMT'14结果的BLEU差距在0.5以内。

表1：LSTM在WMT'14英语到法语测试集（ntst14）上的性能。注意，5个LSTM的集成（束大小为2）比单个LSTM（束大小为12）更便宜。

| 方法 | 测试BLEU分数（ntst14） |
|------|----------------------|
| Bahdanau等人[2] | 28.45 |
| 基线系统[29] | 33.30 |
| 单个前向LSTM，束大小12 | 26.17 |
| 单个反转LSTM，束大小12 | 30.59 |
| 5个反转LSTM的集成，束大小1 | 33.00 |
| 2个反转LSTM的集成，束大小12 | 33.27 |
| 5个反转LSTM的集成，束大小2 | 34.50 |
| 5个反转LSTM的集成，束大小12 | 34.81 |

表2：在WMT'14英语到法语测试集（ntst14）上使用神经网络和SMT系统的方法。

| 方法 | 测试BLEU分数（ntst14） |
|------|----------------------|
| 基线系统[29] | 33.30 |
| Cho等人[5] | 34.54 |
| 最佳WMT'14结果[9] | 37.0 |
| 使用单个前向LSTM重评分基线1000-best | 35.61 |
| 使用单个反转LSTM重评分基线1000-best | 35.85 |
| 使用5个反转LSTM的集成重评分基线1000-best | 36.5 |
| 基线1000-best列表的Oracle重评分 | ~45 |

### 3.7 在长句上的性能

我们惊讶地发现LSTM在长句上表现良好，这在图3中定量显示。表3展示了几个长句及其翻译的示例。

### 3.8 模型分析

图2：该图显示了处理图中短语后获得的LSTM隐藏状态的2D PCA投影。这些短语按意义聚类，在这些示例中，意义主要是词序的函数，这用词袋模型很难捕捉。注意两个聚类具有相似的内部结构。

我们的模型的一个吸引人的特性是它能够将单词序列转换为固定维度的向量。图2可视化了一些学习到的表示。该图清楚地表明，这些表示对词序敏感，而对主动语态替换为被动语态相当不敏感。二维投影是使用PCA获得的。

表3：LSTM生成的一些长句翻译示例与真实翻译对照。读者可以使用Google翻译验证这些翻译是否合理。

类型 | 句子
---|---
我们的模型 | Ulrich UNK , membre du conseil d' administration du constructeur automobile Audi , affirme qu' il s' agit d' une pratique courante depuis des années pour que les téléphones portables puissent être collectés avant les réunions du conseil d' administration afin qu' ils ne soient pas utilisés comme appareils d' écoute à distance .
真实值 | Ulrich Hackenberg , membre du conseil d' administration du constructeur automobile Audi , déclare que la collecte des téléphones portables avant les réunions du conseil , afin qu' ils ne puissent pas être utilisés comme appareils d' écoute à distance , est une pratique courante depuis des années .
我们的模型 | " Les téléphones cellulaires , qui sont vraiment une question , non seulement parce qu' ils pourraient potentiellement causer des interférences avec les appareils de navigation , mais nous savons , selon la FCC , qu' ils pourraient interférer avec les tours de téléphone cellulaire lorsqu' ils sont dans l' air " , dit UNK .
真实值 | " Les téléphones portables sont véritablement un problème , non seulement parce qu' ils pourraient éventuellement créer des interférences avec les instruments de navigation , mais parce que nous savons , d' après la FCC , qu' ils pourraient perturber les antennes-relais de téléphonie mobile s' ils sont utilisés à bord " , a déclaré Rosenker .
我们的模型 | Avec la crémation , il y a un " sentiment de violence contre le corps d' un être cher " , qui sera " réduit à une pile de cendres " en très peu de temps au lieu d' un processus de décomposition " qui accompagnera les étapes du deuil " .
真实值 | Il y a , avec la crémation , " une violence faite au corps aimé " , qui va être " réduit à un tas de cendres " en très peu de temps , et non après un processus de décomposition , qui " accompagnerait les phases du deuil " .



## 4 相关工作

关于神经网络在机器翻译中的应用有大量的工作。到目前为止，将RNN语言模型（RNN Language Model, RNNLM）[23]或前馈神经网络语言模型（Feedforward Neural Network Language Model, NNLM）[3]应用于机器翻译任务的最简单且最有效的方法是对强大的机器翻译基线系统的n-best列表进行重评分[22]，这可以可靠地提高翻译质量。

最近，研究人员开始研究将源语言信息纳入NNLM的方法。这方面的工作包括Auli等人[1]，他们将NNLM与输入句子的主题模型相结合，提高了重评分性能。Devlin等人[8]采用了类似的方法，但他们将NNLM集成到机器翻译系统的解码器中，并利用解码器的对齐信息为NNLM提供输入句子中最相关的单词。他们的方法非常成功，并且比基线取得了显著改进。

我们的工作与Kalchbrenner和Blunsom [18]密切相关，他们是第一个将输入句子映射到向量然后再映射回句子的人，尽管他们使用卷积神经网络将句子映射到向量，这会丢失单词的顺序。与这项工作类似，Cho等人[5]使用类似LSTM的RNN架构将句子映射到向量然后再映射回句子，尽管他们的主要重点是将神经网络集成到SMT系统中。Bahdanau等人[2]也尝试使用神经网络进行直接翻译，该神经网络使用注意力机制来克服Cho等人[5]在长句上遇到的性能不佳问题，并取得了令人鼓舞的结果。同样，Pouget-Abadie等人[26]试图通过以产生平滑翻译的方式翻译源句子的部分内容来解决Cho等人[5]的记忆问题，这类似于基于短语的方法。我们怀疑他们可以通过在反转源句子上训练网络来获得类似的改进。

端到端训练也是Hermann等人[12]的重点，他们的模型使用前馈网络表示输入和输出，并将它们映射到空间中的相似点。然而，他们的方法无法直接生成翻译：要获得翻译，他们需要在预计算的句子数据库中查找最接近的向量，或者对句子进行重评分。



## 5 结论

在这项工作中，我们证明了一个具有有限词汇量且对问题结构几乎没有假设的大型深度LSTM能够在大规模机器翻译任务上超越标准的基于SMT的系统，后者具有无限的词汇量。我们简单的基于LSTM的方法在机器翻译上的成功表明，只要有足够的训练数据，它应该在许多其他序列学习问题上表现良好。

我们对反转源句子中词序所获得的改进程度感到惊讶。我们得出结论，找到具有最多短期依赖关系的问题编码非常重要，因为它们使学习问题变得简单得多。特别是，虽然我们无法在非反转翻译问题上训练标准RNN（如图1所示），但我们相信当源句子被反转时，标准RNN应该很容易训练（尽管我们没有通过实验验证这一点）。

我们还对LSTM正确翻译非常长的句子的能力感到惊讶。我们最初确信LSTM会因为有限的记忆而在长句上失败，并且其他研究人员报告说类似的模型[5, 2, 26]在长句上性能不佳。然而，在反转数据集上训练的LSTMs在翻译长句时几乎没有困难。

最重要的是，我们证明了一种简单、直接且相对未优化的方法可以超越SMT系统，因此进一步的工作可能会带来更高的翻译准确性。这些结果表明，我们的方法可能在其他具有挑战性的序列到序列问题上表现良好。



## 6 致谢

我们感谢Samy Bengio、**Jeff Dean**、Matthieu Devin、G**eoffrey Hinton**、Nal Kalchbrenner、Thang Luong、Wolfgang Macherey、Rajat Monga、Vincent Vanhoucke、Peng Xu、Wojciech Zaremba以及Google Brain团队提供的有益评论和讨论。



## 参考文献

[1] M. Auli, M. Galley, C. Quirk, and G. Zweig. Joint language and translation modeling with recurrent neural networks. In EMNLP, 2013.

[2] D. Bahdanau, K. Cho, and Y. Bengio. Neural machine translation by jointly learning to align and translate. arXiv preprint arXiv:1409.0473, 2014.

[3] Y. Bengio, R. Ducharme, P. Vincent, and C. Jauvin. A neural probabilistic language model. In Journal of Machine Learning Research, pages 1137–1155, 2003.

[4] Y. Bengio, P. Simard, and P. Frasconi. Learning long-term dependencies with gradient descent is difficult. IEEE Transactions on Neural Networks, 5(2):157–166, 1994.

[5] K. Cho, B. Merrienboer, C. Gulcehre, F. Bougares, H. Schwenk, and Y. Bengio. Learning phrase representations using RNN encoder-decoder for statistical machine translation. In Arxiv preprint arXiv:1406.1078, 2014.

[6] D. Ciresan, U. Meier, and J. Schmidhuber. Multi-column deep neural networks for image classification. In CVPR, 2012.

[7] G. E. Dahl, D. Yu, L. Deng, and A. Acero. Context-dependent pre-trained deep neural networks for large vocabulary speech recognition. IEEE Transactions on Audio, Speech, and Language Processing - Special Issue on Deep Learning for Speech and Language Processing, 2012.

[8] J. Devlin, R. Zbib, Z. Huang, T. Lamar, R. Schwartz, and J. Makhoul. Fast and robust neural network joint models for statistical machine translation. In ACL, 2014.

[9] Nadir Durrani, Barry Haddow, Philipp Koehn, and Kenneth Heafield. Edinburgh's phrase-based machine translation systems for wmt-14. In WMT, 2014.

[10] A. Graves. **Generating sequences with recurrent neural networks**. In Arxiv preprint arXiv:1308.0850, 2013.

[11] A. Graves, S. Fernández, F. Gomez, and J. Schmidhuber. Connectionist temporal classification: labelling unsegmented sequence data with recurrent neural networks. In ICML, 2006.

[12] K. M. Hermann and P. Blunsom. Multilingual distributed representations without word alignment. In ICLR, 2014.

[13] G. Hinton, L. Deng, D. Yu, G. Dahl, A. Mohamed, N. Jaitly, A. Senior, V. Vanhoucke, P. Nguyen, T. Sainath, and B. Kingsbury. **Deep neural networks for acoustic modeling in speech recognition**. IEEE Signal Processing Magazine, 2012.

[14] S. Hochreiter. Untersuchungen zu dynamischen neuronalen netzen. Master's thesis, Institut fur Informatik, Technische Universitat, Munchen, 1991.

[15] S. Hochreiter, Y. Bengio, P. Frasconi, and J. Schmidhuber. Gradient flow in recurrent nets: the difficulty of learning long-term dependencies, 2001.

[16] S. Hochreiter and J. Schmidhuber. Long short-term memory. Neural Computation, 1997.

[17] S. Hochreiter and J. Schmidhuber. LSTM can solve hard long time lag problems. 1997.

[18] N. Kalchbrenner and P. Blunsom. Recurrent continuous translation models. In EMNLP, 2013.

[19] A. Krizhevsky, I. Sutskever, and G. E. Hinton. ImageNet classification with deep convolutional neural networks. In NIPS, 2012.

[20] Q.V. Le, M.A. Ranzato, R. Monga, M. Devin, K. Chen, G.S. Corrado, J. Dean, and A.Y. Ng. Building high-level features using large scale unsupervised learning. In ICML, 2012.

[21] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 1998.

[22] T. Mikolov. Statistical Language Models based on Neural Networks. PhD thesis, Brno University of Technology, 2012.

[23] T. Mikolov, M. Karafiát, L. Burget, J. Cernockỳ, and S. Khudanpur. Recurrent neural network based language model. In INTERSPEECH, pages 1045–1048, 2010.

[24] K. Papineni, S. Roukos, T. Ward, and W. J. Zhu. BLEU: a method for automatic evaluation of machine translation. In ACL, 2002.

[25] R. Pascanu, T. Mikolov, and Y. Bengio. On the difficulty of training recurrent neural networks. arXiv preprint arXiv:1211.5063, 2012.

[26] J. Pouget-Abadie, D. Bahdanau, B. van Merrienboer, K. Cho, and Y. Bengio. Overcoming the curse of sentence length for neural machine translation using automatic segmentation. arXiv preprint arXiv:1409.1257, 2014.

[27] A. Razborov. On small depth threshold circuits. In Proc. 3rd Scandinavian Workshop on Algorithm Theory, 1992.

[28] D. Rumelhart, G. E. Hinton, and R. J. Williams. Learning representations by back-propagating errors. Nature, 323(6088):533–536, 1986.

[29] H. Schwenk. University le Mans. http://www-lium.univ-lemans.fr/˜schwenk/cslm_joint_paper/, 2014. [Online; accessed 03-September-2014].

[30] M. Sundermeyer, R. Schluter, and H. Ney. LSTM neural networks for language modeling. In INTERSPEECH, 2010.

[31] P. Werbos. Backpropagation through time: what it does and how to do it. Proceedings of IEEE, 1990.