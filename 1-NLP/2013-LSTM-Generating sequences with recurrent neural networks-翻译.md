# LSTM: 用循环神经网络生成序列

> Alex Graves | University of Toronto

本文展示了如何使用长短期记忆（LSTM）循环神经网络，通过逐点预测的方式生成具有长程结构的复杂序列。核心发现是——**LSTM 能够仅凭下一个数据点预测，生成包含长程依赖的高质量离散文本和连续手写序列，并可通过条件生成实现逼真的手写合成**。

核心内容：
- 标准循环神经网络（RNN）难以存储长期信息，导致生成序列时不稳定，尤其在实值数据上容易偏离训练流形
- 提出基于堆叠 LSTM 层的深度预测网络，结合跳跃连接和混合密度输出层，实现离散和实值序列生成
- 引入软窗口卷积机制，使网络能根据文本序列条件生成手写，动态学习文本与笔迹的对齐关系
- 在 Penn Treebank、Wikipedia 和 IAM 手写数据库上验证，文本预测达 1.24 BPC，手写合成肉眼难辨真伪

关键发现：
- **文本预测：Penn Treebank 上达到 1.24 BPC（困惑度 122），动态评估下与 PAQ-8 压缩算法竞争力相当**
- Wikipedia 实验中动态评估 BPC 为 1.33，静态评估为 1.67，动态评估优势显著
- 手写预测：三层网络 + 自适应权重噪声使对数损失降低至 -1057.7 nats
- 手写合成：合成网络将均方误差降低 44%，偏置采样可在多样性与可读性间灵活权衡

---

## 摘要

本文展示了如何使用长短期记忆（LSTM）循环神经网络，通过逐个预测数据点的方式生成具有长程结构的复杂序列。该方法在文本数据（离散数据）和在线手写数据（实值数据）上进行了验证。随后，通过允许网络根据文本序列调整其预测，将该方法扩展到手写合成任务。最终系统能够生成多种风格的高度逼真的手写体。

## 1 引言

循环神经网络（RNN）是一类功能强大的动态模型，已被用于生成音乐 [6, 4]、文本 [30] 和动作捕捉数据 [29] 等多种领域的序列。RNN 可以通过逐步处理真实数据序列并预测下一个数据点来进行序列生成训练。假设预测是概率性的，则可以从训练好的网络中通过迭代采样网络的输出分布来生成新序列，然后将采样结果作为下一步的输入。换句话说，让网络将其自身的"发明"当作真实数据来处理，就像人在做梦一样。尽管网络本身是确定性的，但通过采样注入的随机性在序列上诱导出一个分布。这个分布是条件分布，因为网络的内部状态及其预测分布取决于之前的输入。

RNN 具有"模糊性"，即它们不使用训练数据的精确模板进行预测，而是像其他神经网络一样，利用其内部表示在训练样本之间进行高维插值。这使它们区别于 n-gram 模型和诸如部分匹配预测（Prediction by Partial Matching）[5] 等压缩算法，后者的预测分布是通过计算近期历史与训练集之间的精确匹配来确定的。其结果——从本文的样本中可以立即看出——是 RNN（不同于基于模板的算法）以复杂的方式综合和重构训练数据，很少生成两次相同的内容。此外，模糊预测不会遭受维度灾难，因此在建模实值或多变量数据方面比精确匹配好得多。

原则上，一个足够大的 RNN 应该足以生成任意复杂度的序列。然而在实践中，标准 RNN 无法长时间存储关于过去输入的信息 [15]。这种"健忘症"不仅削弱了它们建模长程结构的能力，还使它们在生成序列时容易出现不稳定问题。这个问题（所有条件生成模型共有的）在于，如果网络的预测仅基于最近的几个输入，而这些输入本身又是由网络预测的，那么它几乎没有机会从过去的错误中恢复。更长的记忆具有稳定作用，因为即使网络无法理解其近期历史，它也可以回顾更远的过去来制定预测。不稳定问题在实值数据上尤为严重，因为预测很容易偏离训练数据所在的流形。一种针对条件模型的补救方法是在将预测反馈到模型之前向其注入噪声 [31]，从而提高模型对意外输入的鲁棒性。但我们认为，更好的记忆是一个更深刻、更有效的解决方案。

长短期记忆（LSTM）[16] 是一种 RNN 架构，旨在比标准 RNN 更好地存储和访问信息。LSTM 近期在语音和手写识别 [10, 12] 等多种序列处理任务中取得了最先进的结果。本文的主要目标是证明 LSTM 可以利用其记忆生成包含长程结构的复杂、逼真序列。

第 2 节定义了一个由堆叠 LSTM 层组成的"深度"RNN，并解释了如何训练它进行下一步预测以及序列生成。第 3 节将预测网络应用于 Penn Treebank 和 Hutter Prize Wikipedia 数据集的文本。该网络的性能与最先进的语言模型相当，且逐字符预测与逐词预测的效果几乎一样好。本节的亮点是生成的 Wikipedia 文本样本，展示了网络建模长程依赖的能力。第 4 节展示了如何通过混合密度输出层将预测网络应用于实值数据，并在 IAM 在线手写数据库上提供了实验结果。还展示了生成的手写样本，证明了网络能够直接从笔迹中学习字母和短词，并建模手写风格的全局特征。第 5 节介绍了预测网络的一个扩展，允许其根据一个与预测对齐未知的短注释序列来调整输出。这使其适用于手写合成，即用户输入文本，算法生成其手写版本。合成网络在 IAM 数据库上训练，然后用于生成手写样本，其中一些肉眼无法与真实数据区分。还描述了一种使样本偏向更高概率（及更高可读性）的方法，以及一种基于真实数据"预引导"样本从而模仿特定书写者风格的技术。最后，第 6 节给出了结论和未来工作方向。

## 2 预测网络

图 1 展示了本文使用的基本循环神经网络预测架构。输入向量序列 $\mathbf{x}=(x_{1},\ldots,x_{T})$ 通过加权连接传递到 $N$ 个循环连接的隐藏层堆栈，首先计算隐藏向量序列 $\mathbf{h}^{n}=(h^{n}_{1},\ldots,h^{n}_{T})$，然后计算输出向量序列 $\mathbf{y}=(y_{1},\ldots,y_{T})$。每个输出向量 $y_{t}$ 用于参数化关于下一个可能输入 $x_{t+1}$ 的预测分布 $\Pr(x_{t+1}|y_{t})$。每个输入序列的第一个元素 $x_{1}$ 始终是所有条目为零的空向量；因此网络在没有任何先验信息的情况下对第一个真实输入 $x_{2}$ 发出预测。该网络在空间和时间上都是"深度"的，即通过计算图垂直或水平传递的每条信息都将被多个连续的权重矩阵和非线性函数作用。

![图1](/Users/dazhang/PycharmProject/Papers/1-NLP/.picture/2013-LSTM-Generating sequences with recurrent neural networks-fig1.png)

图 1：深度循环神经网络预测架构。圆圈代表网络层，实线代表加权连接，虚线代表预测。

注意从输入到所有隐藏层以及从所有隐藏层到输出的"跳跃连接"。这些连接通过减少网络底部和顶部之间的处理步骤数量，使得深度网络更容易训练，从而缓解"梯度消失"问题 [1]。在 $N=1$ 的特殊情况下，该架构简化为普通的单层下一步预测 RNN。

隐藏层激活通过从 $t=1$ 到 $T$ 以及从 $n=2$ 到 $N$ 迭代以下方程来计算：

$$
h^{1}_{t} = \mathcal{H}\left(W_{ih^{1}}x_{t}+W_{h^{1}h^{1}}h^{1}_{t-1}+b_{h}^{1}\right) \qquad (1)
$$

$$
h^{n}_{t} = \mathcal{H}\left(W_{ih^{n}}x_{t}+W_{h^{n-1}h^{n}}h^{n-1}_{t}+W_{h^{n}h^{n}}h^{n}_{t-1}+b_{h}^{n}\right) \qquad (2)
$$

其中 $W$ 项表示权重矩阵（例如 $W_{ih^{n}}$ 是连接输入到第 $n$ 个隐藏层的权重矩阵，$W_{h^{1}h^{1}}$ 是第一个隐藏层的循环连接等），$b$ 项表示偏置向量（例如 $b_{y}$ 是输出偏置向量），$\mathcal{H}$ 是隐藏层函数。

给定隐藏序列，输出序列计算如下：

$$
\hat{y}_{t} = b_{y}+\sum_{n=1}^{N}{W_{h^{n}y}h^{n}_{t}} \qquad (3)
$$

$$
y_{t} = \mathcal{Y}(\hat{y}_{t}) \qquad (4)
$$

其中 $\mathcal{Y}$ 是输出层函数。因此，完整的网络定义了一个由权重矩阵参数化的函数，从输入历史 $\mathbf{x}_{1:t}$ 映射到输出向量 $y_{t}$。

输出向量 $y_{t}$ 用于参数化下一个输入的预测分布 $\Pr(x_{t+1}|y_{t})$。$\Pr(x_{t+1}|y_{t})$ 的形式必须仔细选择以匹配输入数据。特别是，为高维实值数据找到一个好的预测分布（通常称为密度建模）可能非常具有挑战性。

网络赋予输入序列 $\mathbf{x}$ 的概率为：

$$
\Pr(\mathbf{x})=\prod_{t=1}^{T}{\Pr(x_{t+1}|y_{t})} \qquad (5)
$$

用于训练网络的序列损失 $\mathcal{L}(\mathbf{x})$ 是 $\Pr(\mathbf{x})$ 的负对数：

$$
\mathcal{L}(\mathbf{x})=-\sum_{t=1}^{T}{\log\Pr(x_{t+1}|y_{t})} \qquad (6)
$$

损失对网络权重的偏导数可以通过应用于图 1 所示计算图的基于时间的反向传播 [33] 高效计算，然后网络可以用梯度下降进行训练。

### 2.1 长短期记忆

在大多数 RNN 中，隐藏层函数 $\mathcal{H}$ 是 sigmoid 函数的逐元素应用。然而我们发现，使用专门设计的记忆单元来存储信息的长短期记忆（LSTM）架构 [16] 更善于发现和利用数据中的长程依赖。图 2 展示了一个 LSTM 记忆单元。对于本文使用的 LSTM 版本 [7]，$\mathcal{H}$ 由以下复合函数实现：

$$
i_{t} = \sigma\left(W_{xi}x_{t}+W_{hi}h_{t-1}+W_{ci}c_{t-1}+b_{i}\right) \qquad (7)
$$

$$
f_{t} = \sigma\left(W_{xf}x_{t}+W_{hf}h_{t-1}+W_{cf}c_{t-1}+b_{f}\right) \qquad (8)
$$

$$
c_{t} = f_{t}c_{t-1}+i_{t}\tanh\left(W_{xc}x_{t}+W_{hc}h_{t-1}+b_{c}\right) \qquad (9)
$$

$$
o_{t} = \sigma\left(W_{xo}x_{t}+W_{ho}h_{t-1}+W_{co}c_{t}+b_{o}\right) \qquad (10)
$$

$$
h_{t} = o_{t}\tanh(c_{t}) \qquad (11)
$$

其中 $\sigma$ 是逻辑 sigmoid 函数，$i$、$f$、$o$ 和 $c$ 分别是输入门、遗忘门、输出门、单元和单元输入激活向量，它们都与隐藏向量 $h$ 大小相同。权重矩阵下标具有直观含义，例如 $W_{hi}$ 是隐藏-输入门矩阵，$W_{xo}$ 是输入-输出门矩阵等。从单元到门向量的权重矩阵（如 $W_{ci}$）是对角的，因此每个门向量中的第 $m$ 个元素只接收来自单元向量第 $m$ 个元素的输入。偏置项（加到 $i$、$f$、$c$ 和 $o$ 上的）为清晰起见已省略。

![图2](/Users/dazhang/PycharmProject/Papers/1-NLP/.picture/2013-LSTM-Generating sequences with recurrent neural networks-fig2.png)

图 2：长短期记忆单元

原始 LSTM 算法使用定制设计的近似梯度计算，允许在每个时间步更新权重 [16]。然而，完整梯度可以通过基于时间的反向传播 [11] 来计算，这也是本文使用的方法。使用完整梯度训练 LSTM 的一个困难是，导数有时会变得过大，导致数值问题。为了防止这种情况，本文所有实验都将损失对 LSTM 层网络输入的导数（在应用 sigmoid 和 tanh 函数之前）裁剪在预定义范围内。

## 3 文本预测

文本数据是离散的，通常使用"one-hot"输入向量呈现给神经网络。也就是说，如果总共有 $K$ 个文本类别，在时间 $t$ 输入类别 $k$，则 $x_{t}$ 是一个长度为 $K$ 的向量，其所有条目为零，除了第 $k$ 个为一。因此 $\Pr(x_{t+1}|y_{t})$ 是一个多项分布，可以通过输出层的 softmax 函数自然参数化：

$$
\Pr(x_{t+1}=k|y_{t})=y^{k}_{t}=\frac{\exp\left(\hat{y}^{k}_{t}\right)}{\sum_{k^{\prime}=1}^{K}{\exp\left(\hat{y}^{k^{\prime}}_{t}\right)}} \qquad (12)
$$

代入方程 (6)，我们得到：

$$
\mathcal{L}(\mathbf{x}) = -\sum_{t=1}^{T}{\log y^{x_{t+1}}_{t}} \qquad (13)
$$

$$
\implies\frac{\partial\mathcal{L}(\mathbf{x})}{\partial\hat{y}^{k}_{t}} = y^{k}_{t}-\delta_{k,x_{t+1}} \qquad (14)
$$

剩下的唯一问题是使用哪组类别。在大多数情况下，文本预测（通常称为语言建模）是在词级别进行的。因此 $K$ 是字典中的词数。这对于实际任务来说可能有问题，因为词的数量（包括变位动词、专有名词等）通常超过 100,000。除了需要大量参数来建模外，拥有如此多的类别需要大量训练数据来充分覆盖词的可能上下文。对于 softmax 模型，另一个困难是在训练期间计算所有指数的高计算成本（尽管已经设计了几种方法来使训练大型 softmax 层更高效，包括基于树的模型 [25, 23]、低秩近似 [27] 和随机导数 [26]）。此外，词级模型不适用于包含非词字符串的文本数据，如多位数或网址。

最近有人考虑了使用神经网络进行字符级语言建模 [30, 24]，发现其性能略差于等效的词级模型。然而，从序列生成的角度来看，逐字符预测更有趣，因为它允许网络创造新词和字符串。总的来说，本文的实验旨在以数据中找到的最细粒度进行预测，以最大化网络的生成灵活性。

### 3.1 Penn Treebank 实验

第一组文本预测实验集中在华尔街日报语料库的 Penn Treebank 部分 [22]。这是一项初步研究，主要目的是评估网络的预测能力，而不是生成有趣的序列。

尽管是一个相对较小的文本语料库（总共略多于一百万词），Penn Treebank 数据被广泛用作语言建模基准。训练集包含 930,000 词，验证集包含 74,000 词，测试集包含 82,000 词。词汇量限制为 10,000 词，所有其他词映射到一个特殊的"未知词" token。句末 token 被包含在输入序列中，并计入序列损失。句首标记被忽略，因为其角色已经由开始序列的空向量完成（参见第 2 节）。

实验比较了词级和字符级 LSTM 预测器在 Penn 语料库上的性能。在两种情况下，网络架构都是包含 1000 个 LSTM 单元的单隐藏层。对于字符级网络，输入和输出层大小为 49，总共约 4.3M 个权重，而词级网络有 10,000 个输入和输出，约 54M 个权重。因此比较有些不公平，因为词级网络有更多参数。然而，由于数据集较小，两个网络都很容易过拟合训练数据，尚不清楚字符级网络是否会从更多权重中受益。所有网络都用随机梯度下降训练，学习率为 0.0001，动量为 0.99。LSTM 导数被裁剪在 $[-1,1]$ 范围内（参见第 2.1 节）。

神经网络通常在固定权重下在测试数据上评估。然而对于预测问题，输入即目标，允许网络在评估时适应其权重是合理的（只要它只看一次测试数据）。Mikolov 称之为动态评估。动态评估允许与压缩算法进行更公平的比较，因为后者没有训练集和测试集的划分，所有数据只预测一次。

由于两个网络都过拟合训练数据，我们还实验了两种正则化：权重噪声 [18]（标准差为 0.075，在每个训练序列开始时应用于网络权重）和自适应权重噪声 [8]（噪声的方差与权重一起使用最小描述长度（或等效地，变分推断）损失函数学习）。使用权重噪声时，网络用未正则化网络的最终权重初始化。类似地，使用自适应权重噪声时，权重用权重噪声训练的网络的权重初始化。我们发现，通过迭代增加正则化进行重训练比从随机权重用正则化训练要快得多。自适应权重噪声对于词级网络来说过于缓慢，因此只用固定方差权重噪声进行正则化。自适应权重的一个优点是不需要早停（网络可以安全地在训练数据上最小总"描述长度"点停止）。然而，为了保持公平，所有实验使用相同的训练、验证和测试集。

结果用两个等效指标呈现：每字符比特数（BPC），即整个测试集上 $-\log_{2}\Pr(x_{t+1}|y_{t})$ 的平均值；以及困惑度，即每词平均比特数的 2 次幂（测试集上的平均词长约为 5.6 个字符，因此困惑度 $\approx 2^{5.6BPC}$）。困惑度是语言建模的常用性能指标。

表 1：Penn Treebank 测试集结果。"BPC"是每字符比特数。"Error"是下一步分类错误率，针对字符或词。

| 输入 | 正则化 | 动态 | BPC | 困惑度 | 错误率 (%) | 轮数 |
|------|--------|------|-----|--------|-----------|------|
| 字符 | 无 | 否 | 1.32 | 167 | 28.5 | 9 |
| 字符 | 无 | 是 | 1.29 | 148 | 28.0 | 9 |
| 字符 | 权重噪声 | 否 | 1.27 | 140 | 27.4 | 25 |
| 字符 | 权重噪声 | 是 | 1.24 | 124 | 26.9 | 25 |
| 字符 | 自适应权重噪声 | 否 | 1.26 | 133 | 27.4 | 26 |
| 字符 | 自适应权重噪声 | 是 | 1.24 | 122 | 26.9 | 26 |
| 词 | 无 | 否 | 1.27 | 138 | 77.8 | 11 |
| 词 | 无 | 是 | 1.25 | 126 | 76.9 | 11 |
| 词 | 权重噪声 | 否 | 1.25 | 126 | 76.9 | 14 |
| 词 | 权重噪声 | 是 | 1.23 | 117 | 76.2 | 14 |

表 1 显示词级 RNN 优于字符级网络，但使用正则化时差距似乎缩小了。总体而言，结果与 Tomas Mikolov 论文 [23] 中收集的结果相比表现良好。例如，他记录了使用 Keyser-Ney 平滑的 5-gram 困惑度为 141，词级前馈神经网络为 141.8，最先进的压缩算法 PAQ8 为 131.1，动态评估的词级 RNN 为 123.2。然而通过将多个 RNN、一个 5-gram 和一个缓存模型组合成集成，他能够达到 89.4 的困惑度。有趣的是，动态评估的好处在这里比在 Mikolov 的论文中更为显著（他记录词级 RNN 的困惑度从 124.7 提高到 123.2）。这表明 LSTM 比普通 RNN 更善于快速适应新数据。

### 3.2 Wikipedia 实验

2006 年，Marcus Hutter、Jim Bowery 和 Matt Mahoney 组织了以下挑战，通常称为 Hutter 奖 [17]：将完整英文 Wikipedia 数据的前 1 亿字节（截至 2006 年 3 月 3 日某个时间点）压缩成尽可能小的文件。该文件不仅必须包含压缩数据，还必须包含实现压缩算法的代码。因此其大小可以被视为使用两部分编码方案的数据最小描述长度 [13] 的度量。

Wikipedia 数据从序列生成的角度来看很有趣，因为它不仅包含大量字典词，还包含许多传统语言建模文本语料库中不会包含的字符序列。例如外来词（包括非拉丁字母，如阿拉伯语和中文）、用于定义元数据的缩进 XML 标签、网址以及用于指示页面格式（如标题、要点等）的标记。Hutter 奖数据集的摘录如图 3 和图 4 所示。

数据中的前 96M 字节被均匀分成 100 字节的序列用于训练网络，其余 4M 用于验证。数据总共包含 205 个单字节 unicode 符号。字符总数要高得多，因为许多字符（尤其是非拉丁语言的字符）被定义为多符号序列。遵循建模数据中最细粒度有意义单元的原则，网络一次预测一个字节，因此输入和输出层大小为 205。

Wikipedia 包含长程规律性，例如文章主题，可以跨越数千个词。为了使网络能够捕获这些规律，其内部状态（即隐藏层的输出激活 $h_{t}$ 和层内 LSTM 单元的激活 $c_{t}$）每 100 个序列才重置一次。此外，序列顺序在训练期间没有像通常那样被打乱。因此网络在进行预测时能够访问过去多达 10K 字符的信息。误差项只反向传播到每个 100 字节序列的开头，这意味着梯度计算是近似的。这种截断反向传播的形式之前已在 RNN 语言建模 [23] 中被考虑过，发现可以加速训练（通过减少序列长度从而增加随机权重更新的频率）而不影响网络学习长程依赖的能力。

该数据使用了比 Penn 数据大得多的网络（反映了训练集更大的规模和复杂性），包含 7 个隐藏层，每层 700 个 LSTM 单元，约 21.3M 个权重。网络用随机梯度下降训练，学习率为 0.0001，动量为 0.9。训练 4 个轮次后收敛。LSTM 导数被裁剪在 $[-1,1]$ 范围内。

与 Penn 数据一样，我们在验证数据上测试了网络在有和没有动态评估（在预测数据时更新权重）的情况下的性能。如表 2 所示，动态评估的性能好得多。这可能是因为 Wikipedia 数据的长程一致性；例如，某些词在某些文章中比其他文章中出现频率高得多，能够在评估期间适应这一点是有利的。验证集上的动态结果明显好于训练集，这似乎令人惊讶。然而这很容易用两个因素解释：首先，网络欠拟合训练数据；其次，数据的某些部分比其他部分难得多（例如，纯文本比 XML 标签更难预测）。

为了将结果放在背景中，Hutter 奖的当前获胜者（PAQ-8 压缩算法的一个变体 [20]）在相同数据上达到 1.28 BPC（包括实现算法所需的代码），主流压缩器如 zip 通常超过 2，应用于纯文本版本数据（即所有 XML、标记标签等被移除）的字符级 RNN 在保留数据上达到 1.54，当 RNN 与最大熵模型 [24] 结合时提高到 1.47。

表 2：Wikipedia 结果（每字符比特数）

| 训练集 | 验证集（静态） | 验证集（动态） |
|--------|--------------|--------------|
| 1.42 | 1.67 | 1.33 |

预测网络生成的四页样本如图 5 至图 8 所示。该样本表明网络从数据中学到了大量不同尺度的结构。最明显的是，它学会了大量字典词以及一个子词模型，使其能够创造看起来合理的词和名称：例如 "Lochroom River"、"Mughal Ralvaldens"、"submandration"、"swalloped"。它还学会了基本的标点符号，逗号、句号和段落分隔在文本块中以大致正确的节奏出现。

能够正确开启和关闭引号和括号是语言模型记忆的明确指标，因为闭合符号无法从中间文本预测，因此无法用短程上下文建模 [30]。样本表明网络不仅能够平衡括号和引号，还能平衡格式标记（如用于表示标题的等号），甚至嵌套的 XML 标签和缩进。

网络生成非拉丁字符，如西里尔字母、中文和阿拉伯文，似乎已经学会了英语以外语言的基本模型（例如，它为文章的西班牙语"版本"生成 "es:Geotnia slago"，为荷兰语版本生成 "nl:Rodenbaueri"）。它还生成看起来令人信服的网址（似乎都不是真实的）。

网络生成不同的大区域，如 XML 标头、要点列表和文章正文。与图 3 和图 4 的比较表明，这些区域相当准确地反映了真实数据的构成（尽管生成的版本往往更短且更混杂）。这很重要，因为每个区域可能跨越数百甚至数千个时间步。网络能够在如此大的间隔上保持连贯（甚至将区域放在大致正确的顺序中，例如将标题放在文章开头，将要点"另请参阅"列表放在结尾），这证明了其长程记忆能力。

与所有语言模型生成的文本一样，样本在短语级别之外没有意义。真实感或许可以通过更大的网络和/或更多数据来提高。然而，期望一台从未接触过语言所指的感官世界的机器产生有意义的语言似乎是徒劳的。

最后，网络在训练期间对近期序列的适应（使其能够从动态评估中受益）在摘录中可以清楚地观察到。训练集结束前（此时权重被存储）的最后一篇完整文章是关于洲际弹道导弹的。这篇文章对网络语言模型的影响可以从大量导弹相关术语中看出。其他近期主题包括"个人无政府主义"、意大利作家 Italo Calvino 和国际标准化组织（ISO），所有这些都在网络的词汇中有所体现。

图 3：真实 Wikipedia 数据

图 4：真实 Wikipedia 数据（续）

图 5：生成的 Wikipedia 数据

图 6：生成的 Wikipedia 数据（续）

图 7：生成的 Wikipedia 数据（续）

图 8：生成的 Wikipedia 数据（续）

## 4 手写预测

为了测试预测网络是否也能用于生成令人信服的实值序列，我们将其应用于在线手写数据（这里的"在线"是指书写被记录为笔尖位置序列，与离线手写只有页面图像可用相对）。在线手写由于其低维度（每个数据点两个实数）和易于可视化，是序列生成的一个有吸引力的选择。

本文使用的所有数据均来自 IAM 在线手写数据库（IAM-OnDB）[21]。IAM-OnDB 由使用"智能白板"从 221 位不同书写者收集的手写行组成。书写者被要求书写 Lancaster-Oslo-Bergen 文本语料库 [19] 的表格，他们的笔位置使用白板角落的红外设备跟踪。训练数据的样本如图 9 所示。原始输入数据由 $x$ 和 $y$ 笔坐标以及笔从白板上抬起的序列点组成。$x,y$ 数据中的记录错误通过插值填充缺失读数和移除超过特定阈值长度的步骤来纠正。除此之外，没有使用预处理，网络被训练逐点预测 $x,y$ 坐标和笔画结束标记。这与大多数依赖复杂预处理和特征提取技术的手写识别和合成方法形成对比。我们避免了这些技术，因为它们往往会减少数据中的变化（例如通过归一化字符大小、倾斜度、偏斜度等），而我们希望网络来建模这些变化。逐点预测笔迹给了网络最大的灵活性来创造新颖的手写，但也需要大量记忆，平均每个字母占用超过 25 个时间步，平均一行占用约 700 个时间步。预测延迟笔画（如在词的其余部分写完后添加的 'i' 的点或 't' 的横划）尤其具有挑战性。

![图9](/Users/dazhang/PycharmProject/Papers/1-NLP/.picture/2013-LSTM-Generating sequences with recurrent neural networks-fig9.png)

图 9：IAM 在线手写数据库的训练样本。注意各种书写风格、行角度和字符大小的变化，以及书写和记录错误，如第一行中涂掉的字母和最后一行中重复的词。

IAM-OnDB 分为训练集、两个验证集和测试集，分别包含 5364、1438、1518 和 3859 行手写，取自 775、192、216 和 544 个表格。在我们的实验中，每一行被视为一个独立序列（意味着忽略了连续行之间可能的依赖关系）。为了最大化训练数据量，我们使用训练集、测试集和较大的验证集进行训练，较小的验证集用于早停。缺乏独立测试集意味着记录的结果可能在验证集上有些过拟合；然而验证结果是次要的，因为没有基准结果存在，主要目标是生成看起来令人信服的手写。

将预测网络应用于在线手写数据的主要挑战是确定适合实值输入的预测分布。以下部分描述了如何做到这一点。

### 4.1 混合密度输出

混合密度网络 [2, 3] 的思想是使用神经网络的输出来参数化混合分布。一部分输出用于定义混合权重，其余输出用于参数化各个混合分量。混合权重输出用 softmax 函数归一化以确保它们形成有效的离散分布，其他输出通过适当的函数传递以保持其值在有意义的范围内（例如，指数函数通常应用于用作尺度参数的输出，这些参数必须为正）。混合密度网络通过最大化目标在诱导分布下的对数概率密度来训练。注意密度是归一化的（直到一个固定常数），因此易于微分和获取无偏样本，这与受限玻尔兹曼机 [14] 和其他无向模型形成对比。

混合密度输出也可与循环神经网络一起使用 [28]。在这种情况下，输出分布不仅以当前输入为条件，还以之前的输入历史为条件。直觉上，分量的数量是网络在给定目前输入的情况下对下一个输出的选择数量。

对于本文的手写实验，基本 RNN 架构和更新方程与第 2 节保持不变。每个输入向量 $x_{t}$ 由一个实值对 $x_{1},x_{2}$（定义相对于前一个输入的笔偏移）和一个二值 $x_{3}$（如果向量结束一个笔画则值为 1，即在下一个向量记录之前笔从白板上抬起，否则值为 0）组成。使用二元高斯混合来预测 $x_{1}$ 和 $x_{2}$，而 $x_{3}$ 使用伯努利分布。因此每个输出向量 $y_{t}$ 由笔画结束概率 $e$ 以及 $M$ 个混合分量的均值 $\mu^{j}$、标准差 $\sigma^{j}$、相关系数 $\rho^{j}$ 和混合权重 $\pi^{j}$ 组成。即：

$$
x_{t} \in \mathbb{R}\times\mathbb{R}\times\{0,1\} \qquad (15)
$$

$$
y_{t} = \left(e_{t},\{\pi_{t}^{j},\mu_{t}^{j},\sigma_{t}^{j},\rho_{t}^{j}\}_{j=1}^{M}\right) \qquad (16)
$$

注意均值和标准差是二维向量，而分量权重、相关系数和笔画结束概率是标量。向量 $y_{t}$ 从网络输出 $\hat{y}_{t}$ 获得，其中：

$$
\hat{y}_{t} = \left(\hat{e}_{t},\{\hat{w}_{t}^{j},\hat{\mu}_{t}^{j},\hat{\sigma}_{t}^{j},\hat{\rho}_{t}^{j}\}_{j=1}^{M}\right) = b_{y}+\sum_{n=1}^{N}W_{h^{n}y}h^{n}_{t} \qquad (17)
$$

如下所示：

$$
e_{t} = \frac{1}{1+\exp\left(\hat{e}_{t}\right)} \implies e_{t}\in(0,1) \qquad (18)
$$

$$
\pi_{t}^{j} = \frac{\exp\left(\hat{\pi}_{t}^{j}\right)}{\sum_{{j^{\prime}}=1}^{M}{\exp\left(\hat{\pi}_{t}^{j^{\prime}}\right)}} \implies\pi_{t}^{j}\in(0,1),\ \ \sum_{j}{\pi_{t}^{j}}=1 \qquad (19)
$$

$$
\mu_{t}^{j} = \hat{\mu}_{t}^{j} \implies\mu_{t}^{j}\in\mathbb{R} \qquad (20)
$$

$$
\sigma_{t}^{j} = \exp\left(\hat{\sigma}_{t}^{j}\right) \implies\sigma_{t}^{j}>0 \qquad (21)
$$

$$
\rho_{t}^{j} = \tanh(\hat{\rho}_{t}^{j}) \implies\rho_{t}^{j}\in(-1,1) \qquad (22)
$$

给定输出向量 $y_{t}$，下一个输入 $x_{t+1}$ 的概率密度 $\Pr(x_{t+1}|y_{t})$ 定义如下：

$$
\Pr(x_{t+1}|y_{t}) = \sum_{j=1}^{M}{\pi_{t}^{j}\ \mathcal{N}(x_{t+1}|\mu_{t}^{j},\sigma_{t}^{j},\rho_{t}^{j})}
\begin{cases}
e_{t} & \text{if }(x_{t+1})_{3}=1 \\
1-e_{t} & \text{otherwise}
\end{cases} \qquad (23)
$$

其中：

$$
\mathcal{N}(x|\mu,\sigma,\rho) = \frac{1}{2\pi\sigma_{1}\sigma_{2}\sqrt{1-\rho^{2}}}\exp\left[\frac{-Z}{2(1-\rho^{2})}\right] \qquad (24)
$$

其中：

$$
Z = \frac{(x_{1}-\mu_{1})^{2}}{\sigma_{1}^{2}}+\frac{(x_{2}-\mu_{2})^{2}}{\sigma_{2}^{2}}-\frac{2\rho(x_{1}-\mu_{1})(x_{2}-\mu_{2})}{\sigma_{1}\sigma_{2}} \qquad (25)
$$

这可以代入方程 (6) 来确定序列损失（直到一个仅依赖于数据量化的常数，不影响网络训练）：

$$
\mathcal{L}(\mathbf{x}) = \sum_{t=1}^{T}{-\log\left(\sum_{j}{\pi^{j}_{t}\mathcal{N}(x_{t+1}|\mu_{t}^{j},\sigma_{t}^{j},\rho_{t}^{j})}\right)} -
\begin{cases}
\log e_{t} & \text{if }(x_{t+1})_{3}=1 \\
\log(1-e_{t}) & \text{otherwise}
\end{cases} \qquad (26)
$$

损失对笔画结束输出的导数很简单：

$$
\frac{\partial\mathcal{L}(\mathbf{x})}{\partial\hat{e}_{t}} = (x_{t+1})_{3}-e_{t} \qquad (27)
$$

对混合密度输出的导数可以通过首先定义分量责任 $\gamma^{j}_{t}$ 来求得：

$$
\hat{\gamma}^{j}_{t} = \pi^{j}_{t}\mathcal{N}(x_{t+1}|\mu_{t}^{j},\sigma_{t}^{j},\rho_{t}^{j}) \qquad (28)
$$

$$
\gamma^{j}_{t} = \frac{\hat{\gamma}^{j}_{t}}{\sum_{j^{\prime}=1}^{M}{\hat{\gamma}^{j^{\prime}}_{t}}} \qquad (29)
$$

然后观察到：

$$
\frac{\partial\mathcal{L}(\mathbf{x})}{\partial\hat{\pi}^{j}_{t}} = \pi^{j}_{t}-\gamma^{j}_{t} \qquad (30)
$$

$$
\frac{\partial\mathcal{L}(\mathbf{x})}{\partial(\hat{\mu}^{j}_{t},\hat{\sigma}^{j}_{t},\hat{\rho}^{j}_{t})} = -\gamma^{j}_{t}\frac{\partial\log\mathcal{N}(x_{t+1}|\mu_{t}^{j},\sigma_{t}^{j},\rho_{t}^{j})}{\partial(\hat{\mu}^{j}_{t},\hat{\sigma}^{j}_{t},\hat{\rho}^{j}_{t})} \qquad (31)
$$

其中：

$$
\frac{\partial\log\mathcal{N}(x|\mu,\sigma,\rho)}{\partial\hat{\mu}_{1}} = \frac{C}{\sigma_{1}}\left(\frac{x_{1}-\mu_{1}}{\sigma_{1}}-\frac{\rho(x_{2}-\mu_{2})}{\sigma_{2}}\right) \qquad (32)
$$

$$
\frac{\partial\log\mathcal{N}(x|\mu,\sigma,\rho)}{\partial\hat{\mu}_{2}} = \frac{C}{\sigma_{2}}\left(\frac{x_{2}-\mu_{2}}{\sigma_{2}}-\frac{\rho(x_{1}-\mu_{1})}{\sigma_{1}}\right) \qquad (33)
$$

$$
\frac{\partial\log\mathcal{N}(x|\mu,\sigma,\rho)}{\partial\hat{\sigma}_{1}} = \frac{C(x_{1}-\mu_{1})}{\sigma_{1}}\left(\frac{x_{1}-\mu_{1}}{\sigma_{1}}-\frac{\rho(x_{2}-\mu_{2})}{\sigma_{2}}\right)-1 \qquad (34)
$$

$$
\frac{\partial\log\mathcal{N}(x|\mu,\sigma,\rho)}{\partial\hat{\sigma}_{2}} = \frac{C(x_{2}-\mu_{2})}{\sigma_{2}}\left(\frac{x_{2}-\mu_{2}}{\sigma_{2}}-\frac{\rho(x_{1}-\mu_{1})}{\sigma_{1}}\right)-1 \qquad (35)
$$

$$
\frac{\partial\log\mathcal{N}(x|\mu,\sigma,\rho)}{\partial\hat{\rho}} = \frac{(x_{1}-\mu_{1})(x_{2}-\mu_{2})}{\sigma_{1}\sigma_{2}}+\rho\left(1-CZ\right) \qquad (36)
$$

其中 $Z$ 如方程 (25) 定义，且：

$$
C = \frac{1}{1-\rho^{2}} \qquad (37)
$$

图 10 展示了混合密度输出层应用于在线手写预测的操作。

![图10](/Users/dazhang/PycharmProject/Papers/1-NLP/.picture/2013-LSTM-Generating sequences with recurrent neural networks-fig10.png)

图 10：手写预测的混合密度输出。上方热图显示了书写 "under" 一词时预测笔位置的概率分布序列。连续预测的密度叠加在一起，在分布重叠处给出高值。从密度图中可以看到两种类型的预测：拼写出字母的小斑点是笔画书写时的预测，三个大斑点是笔画结束时对下一个笔画第一个点的预测。笔画结束预测的方差要大得多，因为笔离开白板时位置没有被记录，因此一个笔画的结束和下一个笔画的开始之间可能存在很大距离。下方热图显示了同一序列中的混合分量权重。笔画结束在这里也可见，最活跃的分量在三个位置关闭，其他分量开启：显然笔画结束预测使用了与笔画内预测不同的混合分量集。

### 4.2 实验

数据序列中的每个点由三个数字组成：相对于前一个点的 $x$ 和 $y$ 偏移，以及二值笔画结束特征。因此网络输入层大小为 3。坐标偏移在训练集上归一化为均值 0、标准差 1。使用 20 个混合分量来建模偏移，每个时间步共有 120 个混合参数（20 个权重、40 个均值、40 个标准差和 20 个相关系数）。另一个参数用于建模笔画结束概率，输出层大小为 121。比较了两种隐藏层网络架构：一种是三个隐藏层，每层 400 个 LSTM 单元，另一种是单个隐藏层 900 个 LSTM 单元。两个网络都有约 3.4M 个权重。三层网络用自适应权重噪声 [8] 重新训练，所有标准差初始化为 0.075。使用固定方差权重噪声训练被证明无效，可能是因为它阻止了混合密度层使用精确指定的权重。

网络用 rmsprop 训练，这是一种随机梯度下降形式，梯度除以其近期幅度的运行平均值 [32]。定义 $\epsilon_{i}=\frac{\partial\mathcal{L}(\mathbf{x})}{\partial w_{i}}$，其中 $w_{i}$ 是第 $i$ 个网络权重。权重更新方程为：

$$
n_{i} = \aleph n_{i}+(1-\aleph)\epsilon_{i}^{2} \qquad (38)
$$

$$
g_{i} = \aleph g_{i}+(1-\aleph)\epsilon_{i} \qquad (39)
$$

$$
\Delta_{i} = \beth\Delta_{i}-\gimel\frac{\epsilon_{i}}{\sqrt{n_{i}-g_{i}^{2}+\daleth}} \qquad (40)
$$

$$
w_{i} = w_{i}+\Delta_{i} \qquad (41)
$$

参数如下：

$$
\aleph = 0.95 \qquad (42)
$$

$$
\beth = 0.9 \qquad (43)
$$

$$
\gimel = 0.0001 \qquad (44)
$$

$$
\daleth = 0.0001 \qquad (45)
$$

输出导数 $\frac{\partial\mathcal{L}(\mathbf{x})}{\partial\hat{y}_{t}}$ 被裁剪在 $[-100,100]$ 范围内，LSTM 导数被裁剪在 $[-10,10]$ 范围内。裁剪输出梯度对数值稳定性至关重要；即便如此，网络有时在训练后期过拟合训练数据后会出现数值问题。

表 3 显示三层网络的平均每序列损失比单层网络低 15.3 nats。然而单层网络的均方误差略低。使用自适应权重噪声相对于未正则化的三层网络将损失又降低了 16.7 nats，但没有显著改变均方误差。自适应权重噪声网络似乎生成了最好的样本。

表 3：手写预测结果。所有结果在验证集上记录。"Log-Loss"是 $\mathcal{L}(\mathbf{x})$ 的均值（单位为 nats）。"SSE"是每个数据点的平均均方误差。

| 网络 | 正则化 | Log-Loss | SSE |
|------|--------|----------|-----|
| 1 层 | 无 | -1025.7 | 0.40 |
| 3 层 | 无 | -1041.0 | 0.41 |
| 3 层 | 自适应权重噪声 | -1057.7 | 0.41 |

### 4.3 样本

图 11 展示了预测网络生成的手写样本。网络显然学会了建模笔画、字母甚至短词（尤其是常见词如 'of' 和 'the'）。它似乎还学会了基本的字符级语言模型，因为它创造的词（'eald'、'bryoes'、'lenrest'）在英语中看起来有些合理。考虑到平均每个字符占用超过 25 个时间步，这再次证明了网络生成连贯长程结构的能力。

![图11](/Users/dazhang/PycharmProject/Papers/1-NLP/.picture/2013-LSTM-Generating sequences with recurrent neural networks-fig11.png)

图 11：预测网络生成的在线手写样本。所有样本长度为 700 个时间步。

## 5 手写合成

手写合成是为给定文本生成手写。显然，我们到目前为止描述的预测网络无法做到这一点，因为没有办法约束网络写哪些字母。本节描述了一种增强，允许预测网络根据某个高级注释序列（在手写合成的情况下是字符串）生成数据序列。生成的序列足够令人信服，通常无法与真实手写区分。此外，这种真实性是在不牺牲上一节中展示的写作风格多样性的情况下实现的。

根据文本调整预测的主要挑战是两个序列长度差异很大（笔迹平均比文本长二十五倍），并且它们之间的对齐在数据生成之前是未知的。这是因为用于书写每个字符的坐标数量根据风格、大小、笔速等差异很大。一种能够基于两个长度不同且对齐未知的序列进行顺序预测的神经网络模型是 RNN 转换器 [9]。然而，使用 RNN 转换器进行手写合成的初步实验并不令人鼓舞。一个可能的解释是转换器使用两个独立的 RNN 来处理两个序列，然后组合它们的输出来做决策，而通常更可取的是让所有信息对单个网络可用。这项工作提出了一个替代模型，其中"软窗口"与文本字符串卷积并作为额外输入馈送到预测网络。窗口的参数由网络在做出预测的同时输出，因此它动态地确定文本和笔位置之间的对齐。简单来说，它学会了决定接下来写哪个字符。

![图12](/Users/dazhang/PycharmProject/Papers/1-NLP/.picture/2013-LSTM-Generating sequences with recurrent neural networks-fig12.png)

图 12：合成网络架构。圆圈代表层，实线代表连接，虚线代表预测。拓扑结构与图 1 中的预测网络类似，只是来自字符序列 $\mathbf{c}$ 的额外输入通过窗口层呈现给隐藏层（与第一个隐藏层的连接有延迟以避免图中的循环）。

### 5.1 合成网络

图 12 展示了用于手写合成的网络架构。与预测网络一样，隐藏层堆叠在彼此之上，每层向上层馈送，从输入到所有隐藏层以及从所有隐藏层到输出都有跳跃连接。区别是通过窗口层增加了来自字符序列的输入。

给定长度为 $U$ 的字符序列 $\mathbf{c}$ 和长度为 $T$ 的数据序列 $\mathbf{x}$，在时间步 $t$（$1 \leq t \leq T$）对 $\mathbf{c}$ 的软窗口 $w_{t}$ 由以下与 $K$ 个高斯函数混合的离散卷积定义：

$$
\phi(t,u) = \sum_{k=1}^{K}{\alpha^{k}_{t}\exp\left(-\beta_{t}^{k}\left(\kappa_{t}^{k}-u\right)^{2}\right)} \qquad (46)
$$

$$
w_{t} = \sum_{u=1}^{U}{\phi(t,u)c_{u}} \qquad (47)
$$

其中 $\phi(t,u)$ 是 $c_{u}$ 在时间步 $t$ 的窗口权重。直觉上，$\kappa_{t}$ 参数控制窗口的位置，$\beta_{t}$ 参数控制窗口的宽度，$\alpha_{t}$ 参数控制窗口在混合中的重要性。软窗口向量的大小与字符向量 $c_{u}$ 的大小相同（假设使用 one-hot 编码，这将是字母表中字符的数量）。注意窗口混合没有归一化，因此不构成概率分布；然而窗口权重 $\phi(t,u)$ 可以粗略地解释为网络在时间 $t$ 正在书写字符 $c_{u}$ 的信念。图 13 显示了训练序列中窗口权重所隐含的对齐。

![图13](/Users/dazhang/PycharmProject/Papers/1-NLP/.picture/2013-LSTM-Generating sequences with recurrent neural networks-fig13.png)

图 13：手写合成序列中的窗口权重。图上的每个点显示 $\phi(t,u)$ 的值，其中 $t$ 沿水平轴索引笔迹，$u$ 沿垂直轴索引文本字符。亮线是网络在字符和书写之间选择的对齐。注意该线在字符边界处展开；这意味着网络在转换时接收有关下一个和前一个字母的信息，这有助于指导其预测。

大小为 $3K$ 的窗口参数向量 $p$ 由网络第一个隐藏层的输出如下确定：

$$
(\hat{\alpha}_{t},\hat{\beta}_{t},\hat{\kappa}_{t}) = W_{h^{1}p}h^{1}_{t}+b_{p} \qquad (48)
$$

$$
{\alpha}_{t} = \exp\left(\hat{\alpha}_{t}\right) \qquad (49)
$$

$$
{\beta}_{t} = \exp\left(\hat{\beta}_{t}\right) \qquad (50)
$$

$$
{\kappa}_{t} = \kappa_{t-1}+\exp\left(\hat{\kappa}_{t}\right) \qquad (51)
$$

注意位置参数 ${\kappa}_{t}$ 被定义为相对于前一个位置 $c_{t-1}$ 的偏移，且偏移大小被约束为大于零。直觉上，这意味着网络学会在每一步滑动窗口多远，而不是绝对位置。使用偏移对于使网络将文本与笔迹对齐至关重要。

$w_{t}$ 向量在时间 $t$ 传递到第二和第三个隐藏层，在时间 $t+1$ 传递到第一个隐藏层（以避免在处理图中创建循环）。隐藏层的更新方程为：

$$
h^{1}_{t} = \mathcal{H}\left(W_{ih^{1}}x_{t}+W_{h^{1}h^{1}}h^{1}_{t-1}+W_{wh^{1}}w_{t-1}+b_{h}^{1}\right) \qquad (52)
$$

$$
h^{n}_{t} = \mathcal{H}\left(W_{ih^{n}}x_{t}+W_{h^{n-1}h^{n}}h^{n-1}_{t}+W_{h^{n}h^{n}}h^{n}_{t-1}+W_{wh^{n}}w_{t}+b_{h}^{n}\right) \qquad (53)
$$

输出层的方程与方程 (17) 到 (22) 保持不变。序列损失为：

$$
\mathcal{L}(\mathbf{x}) = -\log\Pr(\mathbf{x}|\mathbf{c}) \qquad (54)
$$

其中：

$$
\Pr(\mathbf{x}|\mathbf{c}) = \prod_{t=1}^{T}{\Pr\left(x_{t+1}|y_{t}\right)} \qquad (55)
$$

注意 $y_{t}$ 现在是 $\mathbf{c}$ 和 $\mathbf{x}_{1:t}$ 的函数。

损失对输出 $\hat{e}_{t},\hat{\pi}_{t},\hat{\mu}_{t},\hat{\sigma}_{t},\hat{\rho}_{t}$ 的导数与方程 (27)、(30) 和 (31) 保持不变。给定通过将输出导数通过图 12 中的计算图反向传播获得的关于大小为 $W$ 的窗口向量 $w_{t}$ 的损失导数 $\frac{\partial\mathcal{L}(\mathbf{x})}{\partial w_{t}}$，对窗口参数的导数如下：

$$
\epsilon(k,t,u) = \alpha^{k}_{t}\exp\left(-\beta_{t}^{k}\left(\kappa_{t}^{k}-u\right)^{2}\right)\sum_{j=1}^{W}{\frac{\partial\mathcal{L}(\mathbf{x})}{\partial w^{j}_{t}}c^{j}_{u}} \qquad (56)
$$

$$
\frac{\partial\mathcal{L}(\mathbf{x})}{\partial\hat{\alpha}^{k}_{t}} = \sum_{u=1}^{U}{\epsilon(k,t,u)} \qquad (57)
$$

$$
\frac{\partial\mathcal{L}(\mathbf{x})}{\partial\hat{\beta}^{k}_{t}} = -{\beta}^{k}_{t}\sum_{u=1}^{U}{\epsilon(k,t,u)(\kappa^{k}_{t}-u)^{2}} \qquad (58)
$$

$$
\frac{\partial\mathcal{L}(\mathbf{x})}{\partial{\kappa}^{k}_{t}} = \frac{\partial\mathcal{L}(\mathbf{x})}{\partial{\kappa}^{k}_{t+1}}+2\beta^{k}_{t}\sum_{u=1}^{U}{\epsilon(k,t,u)(u-\kappa^{k}_{t})} \qquad (59)
$$

$$
\frac{\partial\mathcal{L}(\mathbf{x})}{\partial\hat{\kappa}^{k}_{t}} = \exp\left(\hat{\kappa}^{k}_{t}\right)\frac{\partial\mathcal{L}(\mathbf{x})}{\partial{\kappa}^{k}_{t}} \qquad (60)
$$

图 14 展示了混合密度输出层应用于手写合成的操作。

![图14](/Users/dazhang/PycharmProject/Papers/1-NLP/.picture/2013-LSTM-Generating sequences with recurrent neural networks-fig14.png)

图 14：手写合成的混合密度输出。上方热图显示笔位置的预测分布，下方热图显示混合分量权重。与图 10 的比较表明，合成网络做出更精确的预测（密度斑点更小），尤其是在笔画结束处，合成网络具有知道下一个字母的优势。

### 5.2 实验

合成网络应用于与上一节手写预测网络相同的输入数据。IAM-OnDB 的字符级转录现在用于定义字符序列 $\mathbf{c}$。完整转录包含 80 个不同字符（大写字母、小写字母、数字和标点符号）。然而我们只使用了 57 个字符的子集，所有数字和大多数标点符号被替换为通用的"非字母"标签。网络架构尽可能与最佳预测网络相似：三个隐藏层，每层 400 个 LSTM 单元，输出层 20 个二元高斯混合分量，大小为 3 的输入层。字符序列用 one-hot 向量编码，因此窗口向量大小为 57。窗口参数使用 10 个高斯函数的混合，需要大小为 30 的参数向量。权重总数增加到约 3.7M。

网络用 rmsprop 训练，使用与上一节相同的参数。网络用自适应权重噪声重新训练，初始标准差 0.075，输出和 LSTM 梯度分别裁剪在 $[-100,100]$ 和 $[-10,10]$ 范围内。

表 4 显示自适应权重噪声在对数损失上给出了相当大的改善（约 31.3 nats），但均方误差没有显著变化。正则化网络似乎生成了稍微更逼真的序列，尽管差异肉眼难以辨别。两个网络的性能都明显优于最佳预测网络。特别是均方误差降低了 44%。这可能很大程度上归因于笔画结束处预测的改善，那里的误差最大。

表 4：手写合成结果。所有结果在验证集上记录。"Log-Loss"是 $\mathcal{L}(\mathbf{x})$ 的均值（单位为 nats）。"SSE"是每个数据点的平均均方误差。

| 正则化 | Log-Loss | SSE |
|--------|----------|-----|
| 无 | -1096.9 | 0.23 |
| 自适应权重噪声 | -1128.2 | 0.23 |

### 5.3 无偏采样

给定 $\mathbf{c}$，可以通过从 $\Pr\left(x_{t+1}|y_{t}\right)$ 中迭代抽取 $x_{t+1}$ 来从 $\Pr(\mathbf{x}|\mathbf{c})$ 中获取无偏样本，就像预测网络一样。唯一的区别是我们还必须决定合成网络何时完成书写文本并应停止做出任何未来决策。为此，我们使用以下启发式方法：一旦 $\phi(t,U+1)>\phi(t,u)\ \forall\ 1\leq u\leq U$，当前输入 $x_{t}$ 被定义为序列结束，采样结束。无偏合成样本的示例如图 15 所示。这些以及所有后续图都是使用自适应权重噪声重新训练的合成网络生成的。注意风格特征（如字符大小、倾斜度、连笔程度等）在样本之间差异很大，但在样本内部基本保持一致。这表明网络在序列早期识别出这些特征，然后记住它们直到最后。通过为给定文本查看足够多的样本，似乎可以找到几乎任何风格特征的组合，这表明网络将它们独立于彼此和文本进行建模。

作者在演示期间进行的"盲品测试"表明，至少一些无偏样本肉眼无法与真实手写区分。尽管如此，网络确实会犯人类书写者不会犯的错误，通常涉及缺失、混淆或乱码的字母；这表明网络有时在确定字符和笔迹之间的对齐时存在困难。当不太常见的词或短语包含在字符序列中时，错误数量会显著增加。推测这是因为网络从训练集中学习了一个隐式的字符级语言模型，当发生罕见或未知的转换时会感到困惑。

![图15](/Users/dazhang/PycharmProject/Papers/1-NLP/.picture/2013-LSTM-Generating sequences with recurrent neural networks-fig15.png)

图 15：真实和生成的手写。每个块中的第一行是真实的，其余是合成网络的无偏样本。两个文本来自验证集，在训练期间未被看到。

### 5.4 偏置采样

无偏样本的一个问题是它们往往难以阅读（部分是因为真实手写难以阅读，部分是因为网络是不完美的模型）。直觉上，我们期望网络赋予好的手写更高的概率，因为它往往比坏的手写更平滑、更可预测。如果这是真的，如果我们希望样本更易读，我们应该旨在输出 $\Pr(\mathbf{x}|\mathbf{c})$ 中概率更高的元素。对高概率样本的原则性搜索可能导致困难的推断问题，因为每个输出的概率取决于所有先前的输出。然而一种简单的启发式方法，其中采样器在每一步独立地偏向更可能的预测，通常能给出好结果。将概率偏置 $b$ 定义为大于或等于零的实数。在从 $\Pr(x_{t+1}|y_{t})$ 抽取样本之前，高斯混合中的每个标准差 $\sigma^{j}_{t}$ 从方程 (21) 重新计算为：

$$
\sigma^{j}_{t} = \exp\left(\hat{\sigma}^{j}_{t}-b\right) \qquad (61)
$$

每个混合权重从方程 (19) 重新计算为：

$$
\pi^{j}_{t} = \frac{\exp\left(\hat{\pi}^{j}_{t}(1+b)\right)}{\sum_{j^{\prime}=1}^{M}{\exp\left(\hat{\pi}^{j^{\prime}}_{t}(1+b)\right)}} \qquad (62)
$$

这人为地减少了混合中分量选择和分量本身分布的方差。当 $b=0$ 时恢复无偏采样，当 $b\rightarrow\infty$ 时采样中的方差消失，网络总是输出混合中最可能分量的模式（不一定是混合的模式，但至少是合理的近似）。图 16 显示了逐渐增加偏置的效果，图 17 显示了对与图 15 相同文本使用低偏置生成的样本。

![图16](/Users/dazhang/PycharmProject/Papers/1-NLP/.picture/2013-LSTM-Generating sequences with recurrent neural networks-fig16.png)

图 16：偏向更高概率的样本。概率偏置 $b$ 显示在左侧。随着偏置增加，多样性减少，样本趋向于一种"平均手写"，极其规整且易于阅读（实际上比训练集中的大多数真实手写更容易阅读）。注意即使方差消失，同一个字母在序列中的不同点也不会以相同方式书写（例如 "exactly the same" 中的 'e'，"until they all look" 中的 'l'），因为预测仍然受到先前输出的影响。如果你仔细看，最后三行并不完全相同。

![图17](/Users/dazhang/PycharmProject/Papers/1-NLP/.picture/2013-LSTM-Generating sequences with recurrent neural networks-fig17.png)

图 17：轻微偏置。每个块中的第一行是真实的。其余是概率偏置为 0.15 的合成网络样本，这似乎在多样性和可读性之间给出了良好的平衡。

### 5.5 预引导采样

约束采样的另一个原因是以特定书写者的风格生成手写（而不是随机选择的风格）。最简单的方法是仅在该书写者的数据上重新训练。但即使不重新训练，也可以通过用真实序列"预引导"网络，然后在真实序列仍在网络记忆中时生成扩展来模仿特定风格。对于真实的 $\mathbf{x}$、$\mathbf{c}$ 和合成字符串 $\mathbf{s}$，可以通过将字符序列设置为 $\mathbf{c}^{\prime}=\mathbf{c}+\mathbf{s}$ 并在前 $T$ 个时间步将数据输入固定为 $\mathbf{x}$，然后像往常一样采样直到序列结束来实现。预引导样本的示例如图 18 和图 19 所示。预引导有效的事实证明了网络能够记住在序列早期识别的风格特征。这种技术似乎对训练数据中的序列比网络从未见过的序列效果更好。

![图18](/Users/dazhang/PycharmProject/Papers/1-NLP/.picture/2013-LSTM-Generating sequences with recurrent neural networks-fig18.png)

图 18：用真实序列预引导的样本。预引导序列（取自训练集）显示在每个块的顶部。采样文本中没有一行存在于训练集中。样本为可读性而选择。

![图19](/Users/dazhang/PycharmProject/Papers/1-NLP/.picture/2013-LSTM-Generating sequences with recurrent neural networks-fig19.png)

图 19：用真实序列预引导的样本（续）。

预引导采样和降低方差采样也可以结合使用。如图 20 和图 21 所示，这往往会以预引导风格的"清理"版本生成样本，保留了整体风格特征（如倾斜度和连笔程度），但笔画看起来更平滑、更规整。一个可能的应用是人工增强糟糕的手写。

![图20](/Users/dazhang/PycharmProject/Papers/1-NLP/.picture/2013-LSTM-Generating sequences with recurrent neural networks-fig20.png)

图 20：用真实序列预引导并偏向更高概率的样本。预引导序列在块的顶部。概率偏置为 1。采样文本中没有一行存在于训练集中。

![图21](/Users/dazhang/PycharmProject/Papers/1-NLP/.picture/2013-LSTM-Generating sequences with recurrent neural networks-fig21.png)

图 21：用真实序列预引导并偏向更高概率的样本（续）。

## 6 结论与未来工作

本文证明了长短期记忆循环神经网络使用下一步预测生成具有复杂长程结构的离散和实值序列的能力。它还引入了一种新颖的卷积机制，允许循环网络根据辅助注释序列调整其预测，并使用这种方法合成了多样且逼真的在线手写样本。此外，它展示了如何将这些样本偏向更高的可读性，以及如何基于特定书写者的风格建模。

未来工作的几个方向自然显现。一个是将网络应用于语音合成，由于数据点维度更高，这可能比手写合成更具挑战性。另一个是更好地理解数据的内部表示，并利用这一点直接操纵样本分布。开发一种从序列数据中自动提取高级注释的机制也将是有趣的。在手写的情况下，这可以允许比纯文本更细致的注释，例如风格特征、同一字母的不同形式、笔画顺序信息等。

## 致谢

感谢多伦多大学的 Yichuan Tang、Ilya Sutskever、Navdeep Jaitly、Geoffrey Hinton 和其他同事提供的众多有用评论和建议。这项工作得到了加拿大高级研究所全球奖学金的支持。

## 参考文献

[1] Y. Bengio, P. Simard, and P. Frasconi. Learning long-term dependencies with gradient descent is difficult. IEEE Transactions on Neural Networks, 5(2):157–166, March 1994.

[2] C. Bishop. Mixture density networks. Technical report, 1994.

[3] C. Bishop. Neural Networks for Pattern Recognition. Oxford University Press, Inc., 1995.

[4] N. Boulanger-Lewandowski, Y. Bengio, and P. Vincent. Modeling temporal dependencies in high-dimensional sequences: Application to polyphonic music generation and transcription. In Proceedings of the Twenty-nine International Conference on Machine Learning (ICML'12), 2012.

[5] J. G. Cleary, Ian, and I. H. Witten. Data compression using adaptive coding and partial string matching. IEEE Transactions on Communications, 32:396–402, 1984.

[6] D. Eck and J. Schmidhuber. A first look at music composition using lstm recurrent neural networks. Technical report, IDSIA USI-SUPSI Instituto Dalle Molle.

[7] F. Gers, N. Schraudolph, and J. Schmidhuber. Learning precise timing with LSTM recurrent networks. Journal of Machine Learning Research, 3:115–143, 2002.

[8] A. Graves. Practical variational inference for neural networks. In Advances in Neural Information Processing Systems, volume 24, pages 2348–2356. 2011.

[9] A. Graves. Sequence transduction with recurrent neural networks. In ICML Representation Learning Worksop, 2012.

[10] A. Graves, A. Mohamed, and G. Hinton. Speech recognition with deep recurrent neural networks. In Proc. ICASSP, 2013.

[11] A. Graves and J. Schmidhuber. Framewise phoneme classification with bidirectional LSTM and other neural network architectures. Neural Networks, 18:602–610, 2005.

[12] A. Graves and J. Schmidhuber. Offline handwriting recognition with multidimensional recurrent neural networks. In Advances in Neural Information Processing Systems, volume 21, 2008.

[13] P. D. Grünwald. The Minimum Description Length Principle (Adaptive Computation and Machine Learning). The MIT Press, 2007.

[14] G. Hinton. A Practical Guide to Training Restricted Boltzmann Machines. Technical report, 2010.

[15] S. Hochreiter, Y. Bengio, P. Frasconi, and J. Schmidhuber. Gradient Flow in Recurrent Nets: the Difficulty of Learning Long-term Dependencies. In S. C. Kremer and J. F. Kolen, editors, A Field Guide to Dynamical Recurrent Neural Networks. 2001.

[16] S. Hochreiter and J. Schmidhuber. Long Short-Term Memory. Neural Computation, 9(8):1735–1780, 1997.

[17] M. Hutter. The Human Knowledge Compression Contest, 2012.

[18] K.-C. Jim, C. Giles, and B. Horne. An analysis of noise in recurrent neural networks: convergence and generalization. Neural Networks, IEEE Transactions on, 7(6):1424 –1438, 1996.

[19] S. Johansson, R. Atwell, R. Garside, and G. Leech. The tagged LOB corpus user's manual; Norwegian Computing Centre for the Humanities, 1986.

[20] B. Knoll and N. de Freitas. A machine learning perspective on predictive coding with paq. CoRR, abs/1108.3298, 2011.

[21] M. Liwicki and H. Bunke. IAM-OnDB - an on-line English sentence database acquired from handwritten text on a whiteboard. In Proc. 8th Int. Conf. on Document Analysis and Recognition, volume 2, pages 956–961, 2005.

[22] M. P. Marcus, B. Santorini, and M. A. Marcinkiewicz. Building a large annotated corpus of english: The penn treebank. COMPUTATIONAL LINGUISTICS, 19(2):313–330, 1993.

[23] T. Mikolov. Statistical Language Models based on Neural Networks. PhD thesis, Brno University of Technology, 2012.

[24] T. Mikolov, I. Sutskever, A. Deoras, H. Le, S. Kombrink, and J. Cernocky. Subword language modeling with neural networks. Technical report, Unpublished Manuscript, 2012.

[25] A. Mnih and G. Hinton. A Scalable Hierarchical Distributed Language Model. In Advances in Neural Information Processing Systems, volume 21, 2008.

[26] A. Mnih and Y. W. Teh. A fast and simple algorithm for training neural probabilistic language models. In Proceedings of the 29th International Conference on Machine Learning, pages 1751–1758, 2012.

[27] T. N. Sainath, A. Mohamed, B. Kingsbury, and B. Ramabhadran. Low-rank matrix factorization for deep neural network training with high-dimensional output targets. In Proc. ICASSP, 2013.

[28] M. Schuster. Better generative models for sequential data problems: Bidirectional recurrent mixture density networks. pages 589–595. The MIT Press, 1999.

[29] I. Sutskever, G. E. Hinton, and G. W. Taylor. The recurrent temporal restricted boltzmann machine. pages 1601–1608, 2008.

[30] I. Sutskever, J. Martens, and G. Hinton. Generating text with recurrent neural networks. In ICML, 2011.

[31] G. W. Taylor and G. E. Hinton. Factored conditional restricted boltzmann machines for modeling motion style. In Proc. 26th Annual International Conference on Machine Learning, pages 1025–1032, 2009.

[32] T. Tieleman and G. Hinton. Lecture 6.5 - rmsprop: Divide the gradient by a running average of its recent magnitude, 2012.

[33] R. Williams and D. Zipser. Gradient-based learning algorithms for recurrent networks and their computational complexity. In Back-propagation: Theory, Architectures and Applications, pages 433–486. 1995.
