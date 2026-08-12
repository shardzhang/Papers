# 神经概率语言模型

> Yoshua Bengio, Réjean Ducharme, Pascal Vincent, Christian Jauvin | Université de Montréal



本文提出了一种通过分布式表示来应对统计语言建模中维度灾难的方法。核心内容：

- 将词汇表中的每个词与一个分布式词特征向量（实数向量 $\in \mathbb{R}^m$ ）关联
- 将词序列的联合概率函数表示为这些特征向量的平滑函数
- 使用多层神经网络同时学习词特征向量和概率函数的参数
- 在 Brown 语料库和 AP News 语料库上进行了大规模实验验证

关键发现：

- 神经网络语言模型在困惑度上显著优于最先进的 n-gram 模型（Brown 语料库降低约 24%，AP News 降低约 8%）
- 隐藏层和更多上下文对于模型性能至关重要
- 模型能够利用分布式表示自动泛化到语义和语法上相似的词序列
- 并行化策略（参数并行）使得在大规模数据上训练成为可行

---



## 摘要

统计语言模型的一个基本问题是需要在词序列上指定一个高维联合分布。由于自然语言中词序列的"长尾"特性，一个词序列在训练集中被观测到的可能性往往随着上下文长度的增加而以指数级下降——这是维度灾难的典型表现。传统的 n-gram 模型通过离散截断（只考虑最近的少数字）来解决这个问题，但这样做的代价是无法捕捉长距离依赖。本文提出了一种通过学习**分布式表示**来对抗维度灾难的概率语言模型。该模型为每个词学习一个实数向量（词特征），并使用这些特征向量来计算词序列的概率函数。学习的特征向量编码了词之间的不同语义和语法特征，因此一个训练句可以为模型提供关于与该句在语义上相似的其他句子的信息。在 Brown 语料库和 Associated Press News 语料库上的实验结果表明，与最佳 n-gram 模型相比，该模型在测试集困惑度上获得了 10% 到 20% 的显著改进，同时隐式地学习了词特征向量，捕捉了词之间的语法和语义相似性。


## 1. 引言

对自然语言进行概率建模在诸多应用中都很重要，例如语音识别、机器翻译和信息检索。然而，由于词序列的长尾特性，语言建模面临一个根本性挑战。任何特定的连续词序列都极有可能以前从未出现在训练语料中。因此，一个语言模型必须能够从其训练期间见过的其他句子中进行"泛化"。

对于 n-gram 模型 ¹，当给定更早词语的情况下预测下一个词时，它只依赖于序列中最近的 $n-1$ 个词。通过只考虑局部上下文来限制历史长度，模型限制了可复用的信息量。除了 n-gram 模型中众所周知的稀疏性问题外，这些模型根本不可能利用长度为 3 或 4 以上的上下文，因为（以词汇表大小 $|V| = 100,000$ 为例）长度为 10 的序列有 $10^{50}$ 个可能序列。即使有大量训练数据，几乎所有这些序列的计数值都将为 0。这应该会迫使模型将概率质量集中在一个非常受限的空间上。事实证明，最先进的 n-gram 模型使用巧妙的平滑技术能够很好地推广到不可见的事件（例如，请参见 Chen and Goodman [10] 的综述），但由于它们仅在局部泛化，这是通过**截断上下文**和**泛化出现模式**实现的（基于词的聚类或相似性），仍然存在显著的改进空间。显然，在紧接要预测的词之前的序列中包含的信息远不止前几个词的 identity。该方法中至少有两个特征有待改进，也是本文的重点。首先，它没有考虑远于 1 或 2 个词的上下文 ²，其次它没有考虑词之间的"相似性"。例如，在训练语料中看到句子"The cat is walking in the bedroom"应该有助于我们泛化到句子"A dog was running in a room"，使其几乎同样可能，仅仅是因为"dog"和"cat"（以及"the"和"a"、"room"和"bedroom"等）具有相似的语义和语法角色。

有许多方法被提出来解决这两个问题，我们将在第 1.2 节简要解释本文提出的方法与其中一些早期方法之间的关系。我们首先讨论提出方法的基本思想。第 2 节将给出更正式的形式化描述，使用一个基于共享参数多层神经网络的实现。本文的另一个贡献涉及训练如此巨大的神经网络（具有数百万参数）用于非常大的数据集（数百万或数千万样本）所面临的挑战。最后，本文的一个重要贡献是表明训练这种大规模模型虽然昂贵但可行，能够扩展到较大的上下文，并产生良好的比较结果（第 4 节）。

本文的许多运算采用矩阵表示法，小写 $v$ 表示列向量， $v^{\prime}$ 为其转置， $A_j$ 表示矩阵 $A$ 的第 $j$ 行， $x \cdot y = x^{\prime}y$ 。

### 1.1 用分布式表示对抗维度灾难

简而言之，提出的方法的核心思想可以总结如下：

1. 将词汇表中的每个词与一个分布式词特征向量（一个 $\mathbb{R}^m$ 中的实值向量）关联起来
2. 将这些特征向量作为输入，将词序列的联合概率函数表示为这些特征向量的函数
3. 同时学习词特征向量和该概率函数的参数

特征向量表示词的不同方面：每个词与向量空间中的一个点相关联。特征的数量（例如实验中的 $m = 30, 60$ 或 $100$ ）远小于词汇表的大小（例如 $17,000$ ）。概率函数被表示为给定前词条件下下一个词的条件概率的乘积（例如，在实验中使用多层神经网络根据前词预测下一个词）。该函数的参数可以通过迭代调整以最大化训练数据的对数似然或正则化准则（例如，添加权重衰减惩罚 ³）。每个词关联的特征向量是学习得到的，但也可以使用语义特征的先验知识进行初始化。

**为什么有效？**

在前面的例子中，如果我们知道 dog 和 cat 扮演相似的角色（语义和句法上），并且类似的 (the, a)、(bedroom, room)、(is, was)、(running, walking)，我们可以自然地泛化（即转移概率质量）从
"The cat is walking in the bedroom"
到
"A dog was running in a room"
以及同样到
"The cat is running in a room"
"A dog is walking in a bedroom"
"The dog was walking in the room"
...
以及许多其他组合。在提出的模型中，之所以能够如此泛化，是因为"相似"的词被期望具有相似的特征向量，并且因为概率函数是这些特征值的平滑函数，特征上的微小变化将导致概率上的微小变化。因此，训练数据中即使只出现上述句子中的一个，不仅会增加该句子的概率，还会增加其在句子空间中的大量"邻居"（由特征向量序列表示）的概率。

### 1.2 与先前工作的关系

使用神经网络对高维离散分布进行建模的想法已被发现对于学习 $Z_1, \ldots, Z_n$ 的联合概率很有用，其中每个 $Z_i$ 可能具有不同性质[3, 5]。在该模型中，联合概率被分解为条件概率的乘积：

$$
\hat{P}(Z_1 = z_1, \ldots, Z_n = z_n) = \prod_i \hat{P}(Z_i = z_i \;|\; g_i(Z_{i-1} = z_{i-1}, Z_{i-2} = z_{i-2}, \ldots, Z_1 = z_1))
$$

其中 $g(\cdot)$ 是由一个具有特殊从左到右架构的神经网络表示的函数，第 $i$ 个输出块 $g_i()$ 计算参数以表示 $Z_i$ 在给定之前 $Z$ 值（按某种任意顺序）条件下的条件分布。在四个 UCI 数据集上的实验表明该方法效果非常好[3, 5]。这里我们必须处理可变长度的数据（如句子），因此上述方法必须加以适应。另一个重要的区别是，这里所有的 $Z_i$ （第 $i$ 个位置的词）都指同一种类型的对象（一个词）。因此本文提出的模型引入了跨时间的参数共享——相同的 $g_i$ 在不同时间步和不同位置的输入词之间共享。这是同一思想在大规模应用上的成功，结合了为符号数据学习分布式表示的（旧）思想，这一点在连接主义早期就得到过倡导 [16], [13]。最近，Hinton 的方法得到改进并成功展示了在多个符号关系上的学习 [26]。使用神经网络进行语言建模的想法也不新鲜（例如 [23]）。相比之下，本文将该思想推向了大规模，并集中精力于学习词序列分布的统计模型，而不是学习词在句子中的作用。本文提出的方法也与之前基于字符的文本压缩建议有关，即使用神经网络预测下一个字符的概率 [29]。使用神经网络进行语言建模的想法也被 [33] 独立提出，尽管其实验使用的是没有隐藏单元且只有一个输入词的网络，这限制了模型基本上只能捕捉一元和二元统计量。

发现词之间的某些相似性以从训练序列泛化到新序列的想法也不是新的。例如，它被基于学习词聚类的各种方法所利用（[9], [27], [25], [1]）：每个词被确定性地或概率性地与一个离散类别相关联，同一类中的词在某些方面是相似的。在本文提出的模型中，我们不使用离散随机或确定变量来刻画相似性（这对应于词集的软或硬划分），而是为每个词使用一个连续的实数向量，即一个学习到的分布式特征向量，来表示词之间的相似性。本文的实验比较包括基于类的 n-gram 方法的结果（[9], [24], [25]）。

使用词向量空间表示的想法在信息检索领域得到了很好的利用（例如参见 [30] 的工作），其中词的特征向量是基于它们在同一文档中共同出现的概率学习的（潜在语义索引 LSI，参见 [11]）。一个重要区别在于，本文寻找的是有助于紧凑表示自然语言文本中词序列概率分布的词表示。实验表明，联合学习表示（词特征）和模型是非常有用的。我们尝试过（未成功）使用每个词 $w$ 与 $w$ 在文本中出现的上下文周围的词之间共现频率的前几个主成分作为固定的词特征。这与使用 LSI 进行文档信息检索的做法类似。使用词的连续表示的想法之前已被 [2] 在基于 n-gram 的统计语言模型背景下成功利用，他使用 LSI 动态识别话语主题。

在神经网络背景下使用符号向量空间表示的想法之前也被提出为参数共享层的形式，例如用于二级结构预测的 [28]，以及用于文本到语音映射的 [19]。


## 2. 神经模型

训练集是一个词序列 $w_1 \cdots w_T$ ，其中 $w_t \in V$ ，词汇表 $V$ 是一个大而有限的集合。目标是学习一个好的模型 $f(w_t, \ldots, w_{t-n+1}) = \hat{P}(w_t \;|\; w^{t-1}_{t-n+1})$ ，使其在样本外具有高似然。下面我们报告 $1/\hat{P}(w_t \;|\; w^{t-1}_{t-n+1})$ 的几何平均值，也称为困惑度，它也是平均负对数似然的指数。模型的唯一约束是，对于任何 $w^{t-1}_{t-n+1}$ 的选择， $\sum_{i=1}^{|V|} f(i, w_{t-1}, \ldots, w_{t-n+1}) = 1$ ，且 $f > 0$ 。通过这些条件概率的乘积，我们得到了词序列联合概率的模型。

我们将函数 $f(w_t, \ldots, w_{t-n+1}) = \hat{P}(w_t \;|\; w^{t-1}_{t-n+1})$ 分解为两部分：

1. 从 $V$ 中的任何元素 $i$ 到实数向量 $C(i) \in \mathbb{R}^m$ 的映射 $C$ 。它表示词汇表中每个词关联的分布式特征向量。在实践中， $C$ 由一个 $|V| \times m$ 的自由参数矩阵表示。

2. 基于 $C$ 表示的词的概率函数：函数 $g$ 将上下文中的词的特征向量序列 $(C(w_{t-n+1}), \ldots, C(w_{t-1}))$ 映射为下一个词 $w_t$ 在 $V$ 上的条件概率分布。 $g$ 的输出是一个向量，其第 $i$ 个元素估计概率 $\hat{P}(w_t = i \;|\; w^{t-1}_{t-n+1})$ ，如图 1 所示。

$$
f(i, w_{t-1}, \ldots, w_{t-n+1}) = g(i, C(w_{t-1}), \ldots, C(w_{t-n+1}))
$$

函数 $f$ 是这两个映射（ $C$ 和 $g$ ）的复合，其中 $C$ 在上下文中的所有词之间共享。这两个部分各自关联着一些参数。映射 $C$ 的参数就是特征向量本身，由一个 $|V| \times m$ 的矩阵 $C$ 表示，其第 $i$ 行是词 $i$ 的特征向量 $C(i)$ 。函数 $g$ 可以由前馈或循环神经网络或其他参数化函数实现，参数为 $\omega$ 。整体参数集为 $\theta = (C, \omega)$ 。

训练通过寻找最大化训练语料的惩罚对数似然的 $\theta$ 来实现：

$$
L = \frac{1}{T} \sum_t \log f(w_t, w_{t-1}, \ldots, w_{t-n+1}; \theta) + R(\theta)
$$

其中 $R(\theta)$ 是正则化项。例如，在我们的实验中， $R$ 是一个权重衰减惩罚，仅应用于神经网络的权重和 $C$ 矩阵，不应用于偏置 ⁴。

在上述模型中，自由参数的数量仅随 $V$ （词汇表中的词数）线性增长，也仅随阶数 $n$ 线性增长：如果引入更多的共享结构（例如使用时延神经网络或循环神经网络，或两者组合），缩放因子可以降低到次线性。

在下面的大多数实验中，神经网络在词特征映射之上有一个隐藏层，并且可选地从词特征到输出有直接连接。因此实际上有两个隐藏层：共享的词特征层 $C$ （没有非线性——增加非线性不会带来任何好处）和普通的双曲正切隐藏层。更精确地说，神经网络计算以下函数，使用 softmax 输出层，保证正概率之和为 1：

$$
\hat{P}(w_t \;|\; w_{t-1}, \ldots, w_{t-n+1}) = \frac{e^{y_{w_t}}}{\sum_i e^{y_i}}
$$

其中 $y_i$ 是每个输出词 $i$ 的未归一化对数概率，按如下方式计算，参数为 $b, W, U, d$ 和 $H$ ：

$$
y = b + Wx + U \tanh(d + Hx)
$$

其中双曲正切 $\tanh$ 逐元素应用， $W$ 可选为零（无直接连接）， $x$ 是词特征层激活向量，由矩阵 $C$ 中的输入词特征拼接而成：

$$
x = (C(w_{t-1}), C(w_{t-2}), \ldots, C(w_{t-n+1}))
$$

令 $h$ 为隐藏单元数， $m$ 为每个词关联的特征数。当不需要从词特征到输出的直接连接时，矩阵 $W$ 设置为 0。模型的自由参数是输出偏置 $b$ （ $|V|$ 个元素）、隐藏层偏置 $d$ （ $h$ 个元素）、隐藏到输出权重 $U$ （一个 $|V| \times h$ 矩阵）、词特征到输出权重 $W$ （一个 $|V| \times (n-1)m$ 矩阵）、隐藏层权重 $H$ （一个 $h \times (n-1)m$ 矩阵）以及词特征 $C$ （一个 $|V| \times m$ 矩阵）：

$$
\theta = (b, d, W, U, H, C)
$$

自由参数的数量为 $|V|(1 + nm + h) + h(1 + (n-1)m)$ 。主导因素是 $|V|(nm + h)$ 。注意理论上，如果对权重 $W$ 和 $H$ 有权重衰减而对 $C$ 没有，那么 $W$ 和 $H$ 可能收敛到 0 而 $C$ 会增长。在实践中，当我们使用随机梯度上升训练时没有观察到这种行为。

神经网络的随机梯度上升包括在呈现训练语料中第 $t$ 个词后执行以下迭代更新：

$$
\theta \leftarrow \theta + \varepsilon \frac{\partial \log \hat{P}(w_t \;|\; w_{t-1}, \ldots, w_{t-n+1})}{\partial \theta}
$$

其中 $\varepsilon$ 是"学习率"。注意，大部分参数在每个样本之后不需要更新或访问：所有未出现在输入窗口中的词 $j$ 的词特征 $C(j)$ 不需要更新。

**模型混合。** 在我们的实验（见第 4 节）中，我们发现将神经网络的概率预测与插值三元模型结合可以改进性能，无论是使用简单的固定权重 0.5、学习的权重（在验证集上最大似然估计）还是根据上下文频率条件的一组权重（使用与插值三元模型组合三元、二元和一元相同的程序，这是一个混合模型）。


## 3. 并行实现

尽管参数数量具有良好的扩展性，即与输入窗口大小线性相关且与词汇表大小线性相关，但计算输出概率所需的计算量远大于 n-gram 模型。主要原因是，对于 n-gram 模型，获得特定的 $P(w_t \;|\; w_{t-1}, \ldots, w_{t-n+1})$ 不需要计算词汇表中所有词的概率，因为相对频率的线性组合提供了易于归一化的性质。神经实现的主要计算瓶颈在于输出层激活值的计算。

在并行计算机上运行模型（训练和测试）是减少计算时间的一种方式。我们探索了两种平台上的并行化：共享内存处理器机器和具有快速网络的 Linux 集群。

### 3.1 数据并行处理

在共享内存处理器的情况下，由于处理器之间通过共享内存的通信开销很低，并行化很容易实现。在这种情况下，我们选择了数据并行实现，其中每个处理器处理不同的数据子集。每个处理器计算其样本的梯度，并对模型的参数执行随机梯度更新，这些参数简单地存储在共享内存区域中。我们的最初实现非常慢，因为依赖同步命令来确保每个处理器不会同时在上述参数子集之一上写入。每个处理器的大部分周期都花费在等待另一个处理器释放对参数的写入权限锁。

相反，我们选择了一个异步实现，其中每个处理器可以随时在共享内存区域中写入。有时，一个处理器对参数向量的部分更新会丢失，被另一个处理器的更新覆盖，这在参数更新中引入了一些噪声。然而，这种噪声似乎非常小，并没有明显减慢训练速度。

不幸的是，大型共享内存并行计算机非常昂贵，而且它们的处理器速度往往落后于可以在集群中连接的主流 CPU。因此，我们在快速网络集群上获得了更快的训练速度。

### 3.2 参数并行处理

如果并行计算机是一个 CPU 网络，我们通常无法承受在处理器之间频繁交换所有参数，因为这代表了几十兆字节（对于我们最大的网络接近 100 MB），通过本地网络传输需要太多时间。相反，我们选择在参数上进行并行化，特别是输出单元的参数，因为这是在我们架构中占绝大多数计算量的地方。每个 CPU 负责计算一个输出子集的非归一化概率，并对相应的输出单元参数（进入该单元的权重）进行更新。这种策略使我们能够以可忽略的通信开销执行并行化随机梯度上升。CPU 基本上需要通信两种信息：(1) 输出 softmax 的归一化因子，以及 (2) 隐藏层（下面记为 $a$ ）和词特征层（记为 $x$ ）上的梯度。所有 CPU 重复执行输出单元激活计算之前的计算，即词特征的选择和隐藏层激活 $a$ 的计算，以及相应的反向传播和更新步骤。然而，这些计算对于我们网络的总体计算来说只是微不足道的一小部分。

例如，考虑在 AP（Associated Press）新闻数据实验中使用以下架构：词汇表大小 $|V| = 17,964$ ，隐藏单元数 $h = 60$ ，模型阶数 $n = 6$ ，词特征数 $m = 100$ 。处理单个训练样本的总数值运算次数约为 $|V|(1+nm+h) + h(1+nm) + nm$ （其中各项分别对应于输出单元、隐藏单元和词特征单元的计算）。在这个例子中，计算输出单元加权和所需的计算量占总计算量的比例因此约为：

$$
\frac{|V|(1+(n-1)m+h)}{|V|(1+(n-1)m+h) + h(1+(n-1)m) + (n-1)m} = 99.7\%
$$

这个计算是近似的，因为不同操作的实际 CPU 时间不同，但它表明并行化输出单元计算通常是有利的。所有 CPU 都将重复一小部分计算的事实不会伤害本文所寻求的并行化水平（即几十个处理器）的总计算时间。如果隐藏单元数量很大，并行化它们的计算也将变得有益，但在实验中我们没有研究这种方法。

该策略的实现是在一个通过 Myrinet 网络（一种低延迟千兆局域网）连接的 1.2 GHz Athlon 处理器集群（32 x 2 CPUs）上完成的，使用 MPI（Message Passing Interface）库（[12]）进行并行化例程。下面概述了对于单个样本 $(w_{t-n+1}, \ldots, w_t)$ 的并行化算法，由集群中 $M$ 个处理器中的 CPU $i$ 并行执行。CPU $i$ （ $i$ 从 0 到 $M-1$ ）负责从编号 $\text{start}_i = i \times \lceil |V|/M \rceil$ 开始的一个输出单元块，该块的长度为 $\min(\lceil |V|/M \rceil, |V| - \text{start}_i)$ 。

**处理器 $i$ 的计算，样本 $t$ **

**1. 前向阶段**

(a) 执行词特征层的前向计算：
$$
x^{(k)} \leftarrow C(w_{t-k}), \quad x = (x^{(1)}, x^{(2)}, \ldots, x^{(n-1)})
$$

(b) 执行隐藏层的前向计算：
$$
o \leftarrow d + Hx, \quad a \leftarrow \tanh(o)
$$

(c) 执行第 $i$ 块输出单元的前向计算：
 $s_i \leftarrow 0$
循环第 $i$ 块中的 $j$ ：
i. $y_j \leftarrow b_j + a \cdot U_j$
ii. 如果（有直接连接） $y_j \leftarrow y_j + x \cdot W_j$
iii. $p_j \leftarrow e^{y_j}$
iv. $s_i \leftarrow s_i + p_j$

(d) 计算并在处理器间共享 $S = \sum_i s_i$ 。这可以通过 MPI Allreduce 操作轻松实现，该操作可以高效地计算并共享这个和。

(e) 归一化概率：
循环第 $i$ 块中的 $j$ ： $p_j \leftarrow p_j / S$

(f) 更新对数似然。如果 $w_t$ 落在 CPU $i$ 的块中且 $i > 0$ ，则 CPU $i$ 将 $p_{w_t}$ 发送给 CPU 0。CPU 0 计算 $L = \log p_{w_t}$ 并跟踪总对数似然。

**2. 反向/更新阶段，学习率为 $\varepsilon$ **

(a) 执行第 $i$ 块输出单元的反向梯度计算：
清空梯度向量 $\frac{\partial L}{\partial a}$ 和 $\frac{\partial L}{\partial x}$
循环第 $i$ 块中的 $j$ ：
i. $\frac{\partial L}{\partial y_j} \leftarrow 1_{j==w_t} - p_j$
ii. $b_j \leftarrow b_j + \varepsilon \frac{\partial L}{\partial y_j}$
如果（有直接连接） $\frac{\partial L}{\partial x} \leftarrow \frac{\partial L}{\partial x} + \frac{\partial L}{\partial y_j} W_j$
 $\frac{\partial L}{\partial a} \leftarrow \frac{\partial L}{\partial a} + \frac{\partial L}{\partial y_j} U_j$
如果（有直接连接） $W_j \leftarrow W_j + \varepsilon \frac{\partial L}{\partial y_j} x$
 $U_j \leftarrow U_j + \varepsilon \frac{\partial L}{\partial y_j} a$

(b) 跨处理器求和并共享 $\frac{\partial L}{\partial x}$ 和 $\frac{\partial L}{\partial a}$ 。这可以通过 MPI Allreduce 操作轻松实现。

(c) 反向传播穿过隐藏层并更新隐藏层权重：
循环 $k$ 从 1 到 $h$ ：
 $\frac{\partial L}{\partial o_k} \leftarrow (1 - a_k^2) \frac{\partial L}{\partial a_k}$
 $\frac{\partial L}{\partial x} \leftarrow \frac{\partial L}{\partial x} + H^{\prime} \frac{\partial L}{\partial o}$
 $d \leftarrow d + \varepsilon \frac{\partial L}{\partial o}$
 $H \leftarrow H + \varepsilon \frac{\partial L}{\partial o} x^{\prime}$

(d) 更新输入词的词特征向量：
循环 $k$ 从 1 到 $n-1$ ：
 $C(w_{t-k}) \leftarrow C(w_{t-k}) + \varepsilon \frac{\partial L}{\partial x^{(k)}}$

其中 $\frac{\partial L}{\partial x^{(k)}}$ 是向量 $\frac{\partial L}{\partial x}$ 的第 $k$ 块（长度为 $m$ ）。

上述实现中未显示权重衰减正则化，但可以很容易地加入（从每个参数中减去权重衰减因子乘学习率乘参数值，每次更新时进行）。注意更新直接执行而不是通过参数梯度向量，以提高速度，因为在我们实验中计算速度的一个限制因素是对内存的访问。

前向阶段指数计算可能存在数值问题，使得所有 $p_j$ 可能数值为 0，或其中一个可能太大而无法计算指数（上述步骤 1(c)ii）。为避免此问题，通常的解决方案是在 softmax 中取指数之前减去 $y_j$ 的最大值。因此我们增加了一个额外的 Allreduce 操作，在计算 $p_j$ 中的指数之前，在 $M$ 个处理器之间共享 $y_j$ 的最大值。令 $q_i$ 为块 $i$ 中 $y_j$ 的最大值。然后全局最大值 $Q = \max_i q_i$ 被集体计算并在 $M$ 个处理器之间共享。然后指数按如下方式计算： $p_j \leftarrow e^{y_j - Q}$ （替代步骤 1(c)ii），以保证至少有一个 $p_j$ 在数值上非零，且指数参数的最大值为 1。

通过比较并行版本的时钟时间与单处理器的时钟时间，我们发现通信开销仅占总时间的 1/15（对于一个训练周期）：因此使用该算法在快速网络上通过并行化获得了几乎完美的加速。在具有慢速网络的集群上，可能仍然可以通过每 $K$ 个样本（一个小批量）执行一次通信而不是每个样本都通信来获得高效的并行化。这需要在每个处理器中存储 $K$ 个版本的神经网络激活值和梯度。在 $K$ 个样本的前向阶段之后，概率和必须在处理器之间共享。然后启动 $K$ 个反向阶段，以获得 $K$ 个偏梯度向量 $\frac{\partial L}{\partial a}$ 和 $\frac{\partial L}{\partial x}$ 。在这些梯度向量在处理器之间交换之后，每个处理器可以完成反向阶段并更新参数。这种方法主要由于节省了网络通信延迟（传输的数据量相同）而节省了时间。如果 $K$ 太大，它可能会在收敛时间上有所损失，原因与批量梯度下降通常比随机梯度下降慢得多相同（[22]）。


## 4. 实验结果

在 Brown 语料库上进行了对比实验，该语料库是一个包含 1,181,041 个词的流，来自大量不同类型的英语文本和书籍。前 800,000 个词用于训练，接下来的 200,000 个词用于验证（模型选择、权重衰减、早停），剩余的 181,041 个词用于测试。不同词的数量为 47,578（包括标点符号，区分大小写，并包括用于分隔文本和段落的语法标记）。频率 $\leq 3$ 的罕见词合并为单个符号，将词汇表大小减少到 $|V| = 16,383$ 。

还在 1995 年和 1996 年 Associated Press (AP) News 的文本上进行了实验。训练集是一个约 1400 万（13,994,528）个词的流，验证集是一个约 100 万（963,138）个词的流，测试集也是一个约 100 万（963,071）个词的流。原始数据有 148,721 个不同的词（包括标点符号），通过仅保留最频繁的词（并保留标点符号）、大写转小写、数字形式映射到特殊符号、罕见词映射到特殊符号以及专有名词映射到另一个特殊符号，将词汇表减少到 $|V| = 17,964$ 。

对于训练神经网络，初始学习率设置为 $\varepsilon_0 = 10^{-3}$ （经过在小数据集上的少量试验），并根据以下调度逐渐下降： $\varepsilon_t = \frac{\varepsilon_0}{1 + rt}$ ，其中 $t$ 表示执行的参数更新次数， $r$ 是一个启发式选择的下降因子 $r = 10^{-8}$ 。

### 4.1 N-Gram 模型

第一个与神经网络比较的基准是插值或平滑三元模型 [18]。令 $q_t = l(\text{freq}(w_{t-1}, w_{t-2}))$ 表示输入上下文 $(w_{t-1}, w_{t-2})$ 的离散化出现频率 ⁵。那么条件概率估计具有条件混合的形式：

$$
\hat{P}(w_t \;|\; w_{t-1}, w_{t-2}) = \alpha_0(q_t) p_0 + \alpha_1(q_t) p_1(w_t) + \alpha_2(q_t) p_2(w_t \;|\; w_{t-1}) + \alpha_3(q_t) p_3(w_t \;|\; w_{t-1}, w_{t-2})
$$

其中条件权重 $\alpha_i(q_t) \geq 0$ ， $\sum_i \alpha_i(q_t) = 1$ 。基础预测器如下： $p_0 = 1/|V|$ ， $p_1(i)$ 是一元模型（词 $i$ 在训练集中的相对频率）， $p_2(i|j)$ 是二元模型（前词为 $j$ 时词 $i$ 的相对频率）， $p_3(i|j,k)$ 是三元模型（前两个词为 $j$ 和 $k$ 时词 $i$ 的相对频率）。其动机是，当 $(w_{t-1}, w_{t-2})$ 的频率很大时， $p_3$ 最可靠；而当频率较低时，低阶统计量 $p_2$ 、 $p_1$ 甚至 $p_0$ 更可靠。对于每个 $q_t$ 的离散值（即上下文频率区间），有一组不同的混合权重 $\alpha$ 。它们可以很容易地使用 EM 算法在约 5 次迭代中，在一组不用于估计一元、二元和三元相对频率的数据（验证集）上估计。插值 n-gram 用于与 MLP 形成混合模型，因为它们似乎以非常不同的方式产生"错误"。

还与其他最先进的 n-gram 模型进行了比较：使用 Modified Kneser-Ney 算法的 back-off n-gram 模型 [21]和 Chen and Goodman [10]，以及基于类的 n-gram 模型（[9], [24], [25]）。验证集用于选择 n-gram 的阶数和基于类模型的词类数量。我们使用了 SRI Language Modeling 工具包中这些算法的实现，由 [32] 描述，见 www.speech.sri.com/projects/srilm/。用于计算下面报告的 back-off 模型困惑度，注意我们没有像神经网络困惑度那样给句尾 token 特殊待遇。所有 token（词和标点符号）在平均对数似然时都同等对待（因此在获得困惑度时也是如此）。

### 4.2 结果

以下是不同模型 $\hat{P}$ 的测试集困惑度测量（ $1/\hat{P}(w_t \;|\; w^{t-1}_1)$ 的几何平均值）。随机梯度上升过程的明显收敛在 Brown 语料库上大约在 10 到 20 个周期后达到。在 AP News 语料库上，我们没有看到过拟合的迹象（在验证集上），可能是因为我们只运行了 5 个周期（使用 40 个 CPU 超过 3 周）。使用了验证集上的早停，但仅在 Brown 实验中是必要的。在 Brown 实验中使用 $10^{-4}$ 的权重衰减惩罚，在 AP News 实验中使用 $10^{-5}$ 的权重衰减（通过少量试验根据验证集困惑度选择）。

表 1 总结了在 Brown 语料库上获得的结果。表中所有 back-off 模型都是 Modified Kneser-Ney n-gram，其表现显著优于标准 back-off 模型。当表中为 back-off 模型指定 $m$ 时，使用的是基于类的 n-gram（ $m$ 是词类的数量）。词特征使用的是随机初始化（类似于神经网络权重的初始化），但我们怀疑基于知识的初始化可能会获得更好的结果。

主要结果是，与最佳 n-gram 相比，使用神经网络可以获得显著更好的结果，在 Brown 上测试集困惑度差异约为 24%，在 AP News 上约为 8%（当选择在验证集上效果最好的 MLP 与 n-gram 进行比较时）。表格还表明，神经网络能够利用更多的上下文（在 Brown 上，将上下文从 2 个词增加到 4 个词给神经网络带来了改进，但 n-gram 没有）。它还显示隐藏单元是有用的（MLP3 vs MLP1 和 MLP4 vs MLP2），并且将神经网络的输出概率与插值三元模型混合总是有助于降低困惑度。简单平均有帮助的事实表明神经网络和三元模型在不同的地方产生错误（即给观测词分配低概率）。结果不能说明从输入到输出的直接连接是否有用，但表明至少在较小的语料库上，没有直接输入到输出的连接可以获得更好的泛化，代价是训练时间更长：没有直接连接时网络收敛所需时间加倍（20 个周期而不是 10 个），尽管困惑度略低。一个合理的解释是，直接的输入到输出连接提供了更多的容量，以及从词特征到对数概率映射的"线性"部分的更快学习。另一方面，没有这些连接时，隐藏单元形成了一个紧密的瓶颈，可能促使更好的泛化。

表 2 给出了在更大的语料库（AP News）上的类似结果，尽管困惑度差异较小（8%）。只进行了 5 个周期（大约三周时间，使用 40 个 CPU）。在这种情况下，基于类的模型似乎没有帮助 n-gram 模型，但高阶 Modified Kneser-Ney back-off 模型在 n-gram 模型中给出了最好的结果。


## 5. 扩展与未来工作

在本节中，我们描述了对上述模型的扩展以及未来工作的方向。

### 5.1 能量最小化网络

上述神经网络的一个变体可以解释为遵循 Hinton 关于专家乘积（products of experts）的最新工作 [17]的能量最小化模型。在前几节描述的神经网络中，分布式词特征仅用于"输入"词，而不用于"输出"词（下一个词）。此外，非常大量的参数（大多数）在输出层展开：输出词之间的语义或句法相似性没有被利用。在此处描述的变体中，输出词也由其特征向量表示。网络输入一个词子序列（映射到其特征向量），并输出一个能量函数 $E$ ，当这些词形成一个可能的子序列时 $E$ 较低，当不太可能时 $E$ 较高。例如，网络输出一个"能量"函数

$$
E(w_{t-n+1}, \ldots, w_t) = v \cdot \tanh(d + Hx) + \sum_{i=0}^{n-1} b_{w_{t-i}}
$$

其中 $b$ 是偏置向量（对应于无条件概率）， $d$ 是隐藏单元偏置向量， $v$ 是输出权重向量， $H$ 是隐藏层权重矩阵，与之前的模型不同的是，输入和输出词都对 $x$ 有贡献：

$$
x = (C(w_t), C(w_{t-1}), C(w_{t-2}), \ldots, C(w_{t-n+1}))
$$

能量函数 $E(w_{t-n+1}, \ldots, w_t)$ 可以解释为 $(w_{t-n+1}, \ldots, w_t)$ 联合出现的非归一化对数概率。为了获得条件概率 $\hat{P}(w_t \;|\; w^{t-1}_{t-n+1})$ ，需要对 $w_t$ 的可能值进行归一化（虽然计算量大），如下所示：

$$
\hat{P}(w_t \;|\; w_{t-1}, \ldots, w_{t-n+1}) = \frac{e^{-E(w_{t-n+1}, \ldots, w_t)}}{\sum_i e^{-E(w_{t-n+1}, \ldots, w_{t-1}, i)}}
$$

注意总计算量与前面介绍的架构相当，如果 $v$ 参数由目标词 ( $w_t$ ) 的 identity 索引，参数数量也可以匹配。注意经过上述 softmax 归一化后只留下 $b_{w_t}$ （任何关于 $w_{t-i}$ 对于 $i > 0$ 的线性函数都会被 softmax 归一化消去）。与之前一样，模型的参数可以通过对 $\log \hat{P}(w_t \;|\; w_{t-1}, \ldots, w_{t-n+1})$ 进行随机梯度上升来调整，使用类似的计算。

在专家乘积框架中，隐藏单元可以被视为专家：一个子序列 $(w_{t-n+1}, \ldots, w_t)$ 的联合概率正比于与每个隐藏单元 $j$ 相关的项的和的指数，即 $v_j \tanh(d_j + H_j x)$ 。注意因为我们选择将整个序列的概率分解为每个元素的条件概率，梯度的计算是可行的。例如，与产品 HMM [8]不同，该模型中专家乘积覆盖整个序列，并且可以使用诸如对比散度算法 [8]之类的近似梯度算法来训练。还要注意，这个架构和专家乘积公式可以看作是非常成功的最大熵模型（[7]）的扩展，但其中基函数（或"特征"，即这里的隐藏单元激活值）是通过惩罚最大似然同时学习的，与特征线性组合的参数一起，而不是在外层循环中使用贪婪特征子集选择方法学习。

我们已经实现并实验了上述架构，并开发了一种基于重要性采样的神经网络训练加速技术，实现了 100 倍的加速（[6]）。

**词汇表外词。** 该架构相比前一个架构的一个优势是它能够轻松处理词汇表外词（甚至为它们分配概率！）。主要想法是首先猜测这样一个词的初始特征向量，取可能出现在同一上下文中的其他词的特征向量的加权凸组合，权重正比于它们的条件概率。假设网络在上下文 $w^{t-1}_{t-n+1}$ 中为词 $i \in V$ 分配了概率 $\hat{P}(i \;|\; w^{t-1}_{t-n+1})$ ，并且在此上下文中我们观察到一个新词 $j \notin V$ 。我们按如下方式初始化 $j$ 的特征向量 $C(j)$ ： $C(j) \leftarrow \sum_{i \in V} C(i) \hat{P}(i \;|\; w^{t-1}_{t-n+1})$ 。然后我们可以将 $j$ 纳入 $V$ 并重新计算这个稍大的集合的概率（这只需要对所有词重新归一化，除了词 $i$ 需要一次通过网络的计算）。然后这个特征向量 $C(i)$ 可以在输入上下文部分中使用，当我们试图预测词 $i$ 之后的词的概率时。

### 5.2 其他未来工作

后续这项研究还有许多挑战需要解决。短期内，需要设计和评估加速训练和识别的方法。长期内，除了本文利用的两种主要方式外，应该引入更多的泛化方式。以下是我们打算探索的一些想法：

1. 将网络分解为子网络，例如使用词的聚类。训练许多较小的网络应该更容易且更快。

2. 使用树结构表示条件概率，其中在每个节点应用一个神经网络，每个节点表示给定上下文时一个词类的概率，叶子表示给定上下文时词的概率。这种类型的表示有潜力将计算时间减少 $|V|/\log |V|$ 倍（参见 [4]）。

3. 仅从输出词的一个子集传播梯度。可以是条件概率最高的词（基于更快的模型，如三元模型，参见 [31] 对该思想的应用），也可以是三元模型表现不佳的那些词的子集。如果语言模型与语音识别器耦合，那么只需要计算声学上模糊的词的分数（非归一化概率）。另请参见 [6] 的新加速训练方法，该方法使用重要性采样来选择词。

4. 引入先验知识。可以引入多种形式的先验知识，例如：语义信息（例如来自 WordNet，参见 [14]）、低层语法信息（例如使用词性标注）和高层语法信息（例如将模型与随机语法耦合，如 [4] 所建议）。更长上下文的影响可以通过在神经网络中引入更多结构和参数共享来捕获，例如使用时延或循环神经网络。在这种多层网络中，当网络输入窗口移动时，已经为连续词的小组执行的计算不需要重做。类似地，可以使用循环网络来捕获关于文本主题的潜在的更长期信息。

5. 解释（并可能使用）神经网络学习的词特征表示。一个简单的第一步可以从 $m = 2$ 个特征开始，这更容易可视化。我们相信，更有意义的表示将需要更大的训练语料库，特别是对于更大的 $m$ 值。

6. 多义词可能无法被本文提出的模型很好地服务，该模型为每个词在连续语义空间中分配一个单一点。我们正在研究该模型的扩展，其中每个词与该空间中的多个点关联，每个点对应词的不同含义。


## 6. 结论

在两个语料库上的实验——一个超过 100 万样本，另一个更大的超过 1500 万词——表明，提出的方法相比最先进的平滑三元模型方法产生了更好的困惑度，困惑度差异在 10% 到 20% 之间。

我们相信，这些改进的主要原因是提出的方法允许利用学习到的分布式表示，以其人之道还治其人之身地对抗维度灾难：每个训练句子都会告知模型关于大量其他句子的组合信息。

可能还有更多工作可以改进模型，在架构层面、计算效率层面和利用先验知识层面。未来研究的一个重要优先事项应该是改进加速技术 ⁶ 以及增加容量而不大幅增加训练时间的方法（以处理数亿或更多词的语料库）。利用时间结构并将输入窗口大小扩展到可能包含整个段落（而不过多增加参数数量或计算时间）的一个简单想法是使用时延网络，可能还有循环神经网络。在应用背景下评估本文提出的模型类型也将是有用的，但请参见 [31] 在语音识别词错误率改进方面已经完成的工作。

更一般地说，本文提出的工作为统计语言模型的改进打开了大门，通过用基于分布式表示的更紧凑和更平滑的表示来替代"条件概率表"，这种表示可以容纳更多的条件变量。尽管在统计语言模型（如随机语法）中已经花费了大量精力来限制或总结条件变量以避免过拟合，但本文描述的这类模型将困难转移到了别处：需要更多的计算，但计算和内存需求与条件变量的数量呈线性关系，而不是指数关系。

⁵ 我们使用 $l(x) = \lceil -\log((1+x)/T) \rceil$ ，其中 $\text{freq}(w_{t-1}, w_{t-2})$ 是输入上下文的出现频率， $T$ 是训练语料的大小。

⁶ 参见 [6] 的工作，实现了 100 倍加速。



## 致谢

作者感谢 Léon Bottou, Yann Le Cun 和 Geoffrey Hinton 的有益讨论。本研究由 NSERC 资助机构以及 MITACS 和 IRIS 网络资助。



## 参考文献

[1] D. Baker and A. McCallum. Distributional clustering of words for text classification. In SIGIR'98, 1998.

[2] J.R. Bellegarda. A latent semantic analysis framework for large-span language modeling. In Proceedings of Eurospeech 97, pages 1451–1454, Rhodes, Greece, 1997.

[3] S. Bengio and Y. Bengio. Taking on the curse of dimensionality in joint distributions using neural networks. IEEE Transactions on Neural Networks, special issue on Data Mining and Knowledge Discovery, 11(3):550–557, 2000a.

[4] Y. Bengio. New distributed probabilistic language models. Technical Report 1215, Dept. IRO, Université de Montréal, 2002.

[5] Y. Bengio and S. Bengio. Modeling high-dimensional discrete data with multi-layer neural networks. In S. A. Solla, T. K. Leen, and K-R. Müller, editors, Advances in Neural Information Processing Systems, volume 12, pages 400–406. MIT Press, 2000b.

[6] Y. Bengio and J-S. Senécal. Quick training of probabilistic neural nets by importance sampling. In AISTATS, 2003.

[7] A. Berger, S. Della Pietra, and V. Della Pietra. A maximum entropy approach to natural language processing. Computational Linguistics, 22:39–71, 1996.

[8] A. Brown and G.E. Hinton. Products of hidden markov models. Technical Report GCNU TR 2000-004, Gatsby Unit, University College London, 2000.

[9] P.F. Brown, V.J. Della Pietra, P.V. DeSouza, J.C. Lai, and R.L. Mercer. Class-based n-gram models of natural language. Computational Linguistics, 18:467–479, 1992.

[10] S.F. Chen and J.T. Goodman. An empirical study of smoothing techniques for language modeling. Computer, Speech and Language, 13(4):359–393, 1999.

[11] S. Deerwester, S.T. Dumais, G.W. Furnas, T.K. Landauer, and R. Harshman. Indexing by latent semantic analysis. Journal of the American Society for Information Science, 41(6):391–407, 1990.

[12] J. Dongarra, D. Walker, and The Message Passing Interface Forum. MPI: A message passing interface standard. Technical Report http://www-unix.mcs.anl.gov/mpi, University of Tennessee, 1995.

[13] J.L. Elman. Finding structure in time. Cognitive Science, 14:179–211, 1990.

[14] C. Fellbaum. WordNet: An Electronic Lexical Database. MIT Press, 1998.

[15] J. Goodman. A bit of progress in language modeling. Technical Report MSR-TR-2001-72, Microsoft Research, 2001.

[16] G.E. Hinton. Learning distributed representations of concepts. In Proceedings of the Eighth Annual Conference of the Cognitive Science Society, pages 1–12, Amherst 1986. Lawrence Erlbaum, Hillsdale, 1986.

[17] G.E. Hinton. Training products of experts by minimizing contrastive divergence. Technical Report GCNU TR 2000-004, Gatsby Unit, University College London, 2000.

[18] F. Jelinek and R. L. Mercer. Interpolated estimation of Markov source parameters from sparse data. In E. S. Gelsema and L. N. Kanal, editors, Pattern Recognition in Practice. North-Holland, Amsterdam, 1980.

[19] K.J. Jensen and S. Riis. Self-organizing letter code-book for text-to-phoneme neural network model. In Proceedings ICSLP, 2000.

[20] S.M. Katz. Estimation of probabilities from sparse data for the language model component of a speech recognizer. IEEE Transactions on Acoustics, Speech, and Signal Processing, ASSP-35(3):400–401, March 1987.

[21] R. Kneser and H. Ney. Improved backing-off for m-gram language modeling. In International Conference on Acoustics, Speech and Signal Processing, pages 181–184, 1995.

[22] Y. LeCun, L. Bottou, G.B. Orr, and K.-R. Müller. Efficient backprop. In G.B. Orr and K.-R. Müller, editors, Neural Networks: Tricks of the Trade, pages 9–50. Springer, 1998.

[23] R. Miikkulainen and M.G. Dyer. Natural language processing with modular neural networks and distributed lexicon. Cognitive Science, 15:343–399, 1991.

[24] H. Ney and R. Kneser. Improved clustering techniques for class-based statistical language modelling. In European Conference on Speech Communication and Technology (Eurospeech), pages 973–976, Berlin, 1993.

[25] T.R. Niesler, E.W.D. Whittaker, and P.C. Woodland. Comparison of part-of-speech and automatically derived category-based language models for speech recognition. In International Conference on Acoustics, Speech and Signal Processing, pages 177–180, 1998.

[26] A. Paccanaro and G.E. Hinton. Extracting distributed representations of concepts and relations from positive and negative propositions. In Proceedings of the International Joint Conference on Neural Network, IJCNN'2000, Como, Italy. IEEE, New York, 2000.

[27] F. Pereira, N. Tishby, and L. Lee. Distributional clustering of english words. In 30th Annual Meeting of the Association for Computational Linguistics, pages 183–190, Columbus, Ohio, 1993.

[28] S. Riis and A. Krogh. Improving protein secondary structure prediction using structured neural networks and multiple sequence profiles. Journal of Computational Biology, pages 163–183, 1996.

[29] J. Schmidhuber. Sequential neural text compression. IEEE Transactions on Neural Networks, 7(1):142–146, 1996.

[30] H. Schutze. Word space. In S. J. Hanson, J. D. Cowan, and C. L. Giles, editors, Advances in Neural Information Processing Systems 5, pages 895–902, San Mateo CA. Morgan Kaufmann, 1993.

[31] H. Schwenk and J-L. Gauvain. Connectionist language modeling for large vocabulary continuous speech recognition. In International Conference on Acoustics, Speech and Signal Processing, pages 765–768, Orlando, Florida, 2002.

[32] A. Stolcke. SRILM - an extensible language modeling toolkit. In Proceedings of the International Conference on Statistical Language Processing, Denver, Colorado, 2002.

[33] W. Xu and A. Rudnicky. Can artificial neural network learn language models. In International Conference on Statistical Language Processing, pages M1–13, Beijing, China, 2000.
