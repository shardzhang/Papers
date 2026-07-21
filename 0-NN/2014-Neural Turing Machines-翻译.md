# 神经图灵机

> Alex Graves, Greg Wayne, Ivo Danihelka | Google DeepMind, 伦敦, 英国

本文介绍了神经图灵机（Neural Turing Machine, NTM），一种通过注意力过程与外部记忆资源交互的神经网络架构。该组合系统类似于图灵机或冯·诺依曼架构，但端到端可微，因此可以通过梯度下降进行高效训练。NTM由神经网络控制器和可寻址外部记忆库组成，通过"模糊"读写操作与记忆交互，其模糊程度由注意力"聚焦"机制决定。架构设计了基于内容和基于位置的两种互补寻址机制，支持迭代访问和随机访问。

核心内容：

- 提出了一种将神经网络与可寻址外部记忆相结合的端到端可微架构，使网络能够通过注意力机制进行读写操作
- 设计了基于内容寻址和基于位置寻址的互补机制，分别适用于按内容检索和按位置访问的不同场景
- 在复制、重复复制、联想召回、动态N-Gram和优先级排序五项算法任务上进行了验证

关键发现：

- NTM能够学习紧凑的内部程序，并在远超训练数据范围的序列上展现出强大的泛化能力（如从长度20泛化到长度100的复制任务）
- 与标准LSTM相比，NTM在需要长期记忆存储的任务上学习速度更快、收敛成本更低，且具备质的差异
- NTM可以学习创建和遍历数组、实现嵌套循环、利用间接寻址进行关联检索以及模拟N-Gram模型等基本算法操作
- 基于前馈控制器的NTM在某些任务上比基于LSTM控制器的NTM泛化性能更优，表明外部记忆比内部状态更适合维护数据结构

---

## 摘要

我们通过将神经网络与外部记忆资源耦合来扩展其能力，网络可以通过注意力过程与这些记忆资源进行交互。该组合系统类似于图灵机或冯·诺依曼架构，但它是端到端可微的，因此可以通过梯度下降进行高效训练。初步结果表明，神经图灵机能够从输入输出示例中推断出简单的算法，例如复制、排序和联想召回。

## 1 引言

计算机程序利用了三种基本机制：基本操作（例如算术运算）、逻辑流程控制（分支）和外部记忆——在计算过程中可以对其进行写入和读取（Von Neumann, 1945）[41]。尽管现代机器学习在建模复杂数据方面取得了广泛的成功，但在很大程度上忽视了逻辑流程控制和外部记忆的使用。

循环神经网络（RNN）因其能够学习和执行长时间跨度上的复杂数据变换而在其他机器学习方法中脱颖而出。此外，众所周知RNN是图灵完备的（Siegelmann and Sontag, 1995）[35]，因此理论上只要连接得当，就有能力模拟任意过程。然而，原则上可行的并不总是实践中简单的。因此，我们丰富了标准循环网络的能力，以简化算法任务的求解。这种丰富主要通过一个大型的可寻址记忆来实现，因此，类比于图灵通过无限记忆磁带对有限状态机的丰富，我们将我们的设备命名为"神经图灵机"（NTM）。与图灵机不同，NTM是一种可微计算机，可以通过梯度下降进行训练，提供了一种学习程序的实际机制。

在人类认知中，与算法操作最相似的过程被称为"工作记忆"。尽管工作记忆的机制在神经生理学层面仍有些模糊，但其文字定义被理解为短期存储信息并基于规则进行操作的能力（Baddeley et al., 2009）[1]。用计算术语来说，这些规则就是简单的程序，而存储的信息构成了这些程序的参数。因此，NTM类似于工作记忆系统，因为它旨在解决需要将近似规则应用于"快速创建变量"的任务。快速创建变量（Hadley, 2009）[16]是那些快速绑定到记忆槽位的数据，就像在传统计算机中数字3和数字4被放入寄存器中相加得到7一样（Minsky, 1967）[28]。NTM与工作记忆模型还有另一个密切的相似之处：NTM架构使用注意力过程来选择性地读写记忆。与大多数工作记忆模型不同，我们的架构可以学习使用其工作记忆，而不是在符号数据上部署固定的过程集。

本报告的组织结构如下：首先简要回顾心理学、语言学和神经科学中与工作记忆相关的研究，以及人工智能和神经网络中的相关工作。然后描述我们的基本贡献——一种我们认为非常适合需要归纳和执行简单程序的任务的记忆架构和注意力控制器。为了测试这一架构，我们构建了一系列问题，并提供精确的问题描述和实验结果。最后总结该架构的优势。

## 2 基础研究

### 2.1 心理学与神经科学

工作记忆的概念在心理学中得到了最为深入的发展，用于解释涉及短期信息操作的任务表现。总体图景是"中央执行系统"集中注意力并对记忆缓冲区的数据执行操作（Baddeley et al., 2009）[1]。心理学家广泛研究了工作记忆的容量限制，通常通过可以轻易回忆的信息"块"的数量来量化（Miller, 1956）[26][1]。这些容量限制有助于理解人类工作记忆系统中的结构约束，但在我们自己的工作中，我们乐于超越这些限制。

在神经科学中，工作记忆过程被归因于由前额叶皮层和基底神经节组成的系统的功能（Goldman-Rakic, 1995）[12]。典型的实验涉及在猴子执行任务时记录前额叶皮层中单个神经元或神经元群的放电活动。任务包括观察瞬态线索，等待一个"延迟期"，然后以依赖于该线索的方式做出响应。某些任务会在延迟期引起单个神经元的持续放电或更复杂的神经动力学。最近的一项研究基于群体编码的"维度"度量，量化了复杂上下文依赖任务中前额叶皮层的延迟期活动，并表明该活动预测了记忆表现（Rigotti et al., 2013）[32]。

工作记忆的建模研究范围广泛，从考虑生物物理回路如何实现持续神经元放电（Wang, 1999）[42]到试图解决明确任务（Hazy et al., 2006）[17]（Dayan, 2008）[6]（Eliasmith, 2013）[7]。其中，Hazy等人的模型与我们的工作最为相关，因为它本身类似于我们自己所修改的长短期记忆（LSTM）架构。与我们的架构一样，Hazy等人的模型具有将信息门控送入记忆槽位的机制，用于解决由嵌套规则构成的记忆任务。与我们的工作相比，该模型没有包含复杂的记忆寻址概念，这限制了系统只能存储和回忆相对简单的原子数据。寻址是我们工作的基础，但通常被排除在神经科学的计算模型之外，不过值得提及的是Gallistel和King（Gallistel and King, 2009）[11]以及Marcus（Marcus, 2003）[25]认为寻址必然涉及大脑的运作。

### 2.2 认知科学与语言学

从历史来看，认知科学和语言学与人工智能几乎同时兴起，都深受计算机问世的影响（Chomsky, 1956）[4]（Miller, 2003）[27]。它们的目标是基于信息或符号处理的隐喻来解释人类的心智行为。在20世纪80年代早期，这两个领域都将递归或程序性（基于规则的）符号处理视为认知的最高标志。并行分布式处理（PDP）或联结主义革命推翻了符号处理隐喻，转而采用所谓的"亚符号"思维过程描述（Rumelhart et al., 1986）[33]。

Fodor和Pylyshyn（Fodor and Pylyshyn, 1988）[9]提出了关于神经网络在认知建模中局限性的两个尖锐论点。他们首先反对联结主义理论无法进行变量绑定，即将特定数据分配到数据结构中的特定槽位。在语言中，变量绑定无处不在；例如，当一个人生成或解释"Mary spoke to John"这样的句子时，他将"Mary"分配为主语角色，"John"分配为宾语角色，"spoke to"分配为及物动词角色。Fodor和Pylyshyn还认为，具有固定长度输入域的神经网络无法再现人类在处理变长结构任务中的能力。针对这一批评，神经网络研究者包括Hinton（Hinton, 1986）[18]、Smolensky（Smolensky, 1990）[36]、Touretzky（Touretzky, 1990）[40]、Pollack（Pollack, 1990）[31]、Plate（Plate, 2003）[30]和Kanerva（Kanerva, 2009）[24]研究了能够在联结主义框架内支持变量绑定和变长结构的具体机制。我们的架构借鉴并增强了这些工作。

变长结构的递归处理仍然被认为是人类认知的标志。在过去十年中，语言学领域的一场论战使多位领域领军人物相互对立。争论的焦点在于，递归处理是否是"人类独有的"进化创新，它使得语言成为可能并且专属于语言——这一观点由Fitch、Hauser和Chomsky（Fitch et al., 2005）[8]支持；还是多种新的适应性变化共同导致了人类语言的进化，而递归处理在语言之前就已存在（Jackendoff and Pinker, 2005）[23]。无论递归处理的进化起源如何，所有人都一致认为它对人类认知的灵活性至关重要。

### 2.3 循环神经网络

循环神经网络构成了一类具有动态状态的广泛机器；也就是说，它们的状态演化既依赖于系统的输入，也依赖于当前状态。与同样包含动态状态的隐马尔可夫模型相比，RNN具有分布式状态，因此具有显著更大和更丰富的记忆与计算能力。动态状态至关重要，因为它提供了上下文依赖计算的可能性；在某一时刻进入的信号可以改变网络在更晚时刻的行为。

循环网络的一个关键创新是长短期记忆（LSTM）（Hochreiter and Schmidhuber, 1997）[20]。这一非常通用的架构是为特定目的而开发的——解决"梯度消失和爆炸"问题（Hochreiter et al., 2001a）[19]，我们可以将其重新标记为"敏感性消失和爆炸"问题。LSTM通过在网络中嵌入完美积分器（Seung, 1998）[34]用于记忆存储来改善这一问题。完美积分器最简单的例子是方程 $x(t+1) = x(t) + i(t)$，其中 $i(t)$ 是系统的输入。隐式单位矩阵 $I x(t)$ 意味着信号不会动态消失或爆炸。如果给这个积分器附加一个机制，允许外围网络选择积分器何时监听输入——即一个依赖于上下文的可编程门控，我们就得到形式为 $x(t+1) = x(t) + g(\text{context}) i(t)$ 的方程。现在我们可以选择性地将信息存储无限长的时间。

循环网络无需修改即可自然地处理变长结构。在序列问题中，网络输入在不同时间到达，使得变长或复合结构可以在多个步骤上被处理。由于它们原生地处理变长结构，近期已被用于各种认知问题，包括语音识别（Graves et al., 2013; Graves and Jaitly, 2014）[15,14]、文本生成（Sutskever et al., 2011）[38]、手写生成（Graves, 2013）[13]和机器翻译（Sutskever et al., 2014）[39]。考虑到这一特性，我们认为构建显式解析树来贪婪地合并复合结构并不紧迫，甚至未必有价值（Pollack, 1990）[31]（Socher et al., 2012）[37]（Frasconi et al., 1998）[10]。

其他重要的前期工作包括可微注意力模型（Graves, 2013）[13]（Bahdanau et al., 2014）[2]和使用循环神经网络构建的程序搜索（Hochreiter et al., 2001b）[21]（Das et al., 1992）[5]。

## 3 神经图灵机

神经图灵机（NTM）架构包含两个基本组件：一个神经网络控制器和一个记忆库。图1展示了NTM架构的高层示意图。与大多数神经网络一样，控制器通过输入和输出向量与外部世界交互。与标准网络不同，它还使用选择性读写操作与记忆矩阵交互。类比于图灵机，我们将参数化这些操作的网络输出称为"头"。

<img src="figure1.png" alt="图1：神经图灵机架构">

**图1：神经图灵机架构。** 在每个更新周期中，控制器网络从外部环境接收输入并发出输出作为响应。它还通过一组并行读写头对记忆矩阵进行读写。虚线表示NTM电路与外部世界的分界。

关键在于，架构的每个组件都是可微的，使得通过梯度下降进行训练变得简单直接。我们通过定义"模糊"读写操作来实现这一点，这些操作与记忆中的所有元素都有不同程度的交互（而不是像普通图灵机或数字计算机那样寻址单个元素）。模糊程度由注意力"聚焦"机制决定，该机制约束每个读写操作只与记忆的一小部分交互，而忽略其余部分。由于与记忆的交互高度稀疏，NTM倾向于无干扰地存储数据。注意力聚焦的记忆位置由头部发出的专门输出决定。这些输出定义了记忆矩阵行（称为记忆"位置"）上的归一化权重。每个权重（每个读头或写头对应一个）定义了该头在每个位置读写或写入的程度。因此，一个头可以锐利地关注记忆中的单个位置，或微弱地关注多个位置。

### 3.1 读取

令 $\mathbf{M}_t$ 为 $t$ 时刻 $N \times M$ 记忆矩阵的内容，其中 $N$ 是记忆位置的数量，$M$ 是每个位置的向量大小。令 $\mathbf{w}_t$ 为 $t$ 时刻读头发出的 $N$ 个位置上的权重向量。由于所有权重都是归一化的，$\mathbf{w}_t$ 的 $N$ 个元素 $w_t(i)$ 满足以下约束：

$$\sum_i w_t(i) = 1, \quad 0 \leq w_t(i) \leq 1, \forall i. \qquad (1)$$

读头返回的长度为 $M$ 的读取向量 $\mathbf{r}_t$ 定义为记忆行向量 $\mathbf{M}_t(i)$ 的凸组合：

$$\mathbf{r}_t \leftarrow \sum_i w_t(i) \mathbf{M}_t(i), \qquad (2)$$

这显然对记忆和权重都是可微的。

### 3.2 写入

受LSTM中输入门和遗忘门的启发，我们将每次写入分解为两个部分：先擦除后添加。

给定 $t$ 时刻写头发出的权重 $\mathbf{w}_t$，以及一个其 $M$ 个元素均在 $(0, 1)$ 范围内的擦除向量 $\mathbf{e}_t$，上一时间步的记忆向量 $\mathbf{M}_{t-1}(i)$ 按如下方式修改：

$$\tilde{\mathbf{M}}_t(i) \leftarrow \mathbf{M}_{t-1}(i) \left[ \mathbf{1} - w_t(i) \mathbf{e}_t \right], \qquad (3)$$

其中 $\mathbf{1}$ 是全1行向量，与记忆位置的乘法是逐元素进行的。因此，仅当位置的权重和擦除元素均为1时，记忆位置的元素才被重置为零；如果权重或擦除任一为零，则记忆保持不变。当存在多个写头时，由于乘法是可交换的，擦除可以按任意顺序执行。

每个写头还会产生一个长度为 $M$ 的添加向量 $\mathbf{a}_t$，在执行擦除步骤后将其添加到记忆中：

$$\mathbf{M}_t(i) \leftarrow \tilde{\mathbf{M}}_t(i) + w_t(i) \mathbf{a}_t. \qquad (4)$$

同样，多个写头执行添加的顺序无关紧要。所有写头的擦除和添加操作共同产生 $t$ 时刻记忆的最终内容。由于擦除和添加都是可微的，复合写入操作也是可微的。注意，擦除向量和添加向量都有 $M$ 个独立分量，允许对每个记忆位置中的哪些元素被修改进行细粒度控制。

### 3.3 寻址机制

虽然我们已经展示了读写方程，但尚未描述权重是如何产生的。这些权重是通过结合两种具有互补功能的寻址机制产生的。第一种机制——"基于内容寻址"——根据位置当前值与控制器发出的值之间的相似度来聚焦注意力。这与Hopfield网络的内容寻址相关（Hopfield, 1982）[22]。基于内容寻址的优势在于检索简单，只需控制器生成存储数据一部分的近似值，然后与记忆进行比对以得到精确的存储值。

然而，并非所有问题都适合基于内容寻址。在某些任务中，变量的内容是任意的，但变量仍需要可识别的名称或地址。算术问题属于这一类：变量 $x$ 和变量 $y$ 可以取任意两个值，但过程 $f(x, y) = x \times y$ 仍应被定义。该任务的控制器可以获取变量 $x$ 和 $y$ 的值，将它们存储在不同的地址，然后检索它们并执行乘法算法。在这种情况下，变量是通过位置而非内容来寻址的。我们将这种形式的寻址称为"基于位置寻址"。基于内容寻址严格来说比基于位置寻址更通用，因为记忆位置的内容可以在其内部包含位置信息。然而，在我们的实验中，提供基于位置寻址作为基本操作对某些形式的泛化至关重要，因此我们同时使用这两种机制。

<img src="figure2.png" alt="图2：寻址机制流程示意图">

**图2：寻址机制的流程示意图。** 关键向量 $\mathbf{k}_t$ 和关键强度 $\beta_t$ 用于对记忆矩阵 $\mathbf{M}_t$ 进行基于内容寻址。得到的基于内容的权重根据插值门 $g_t$ 的值与上一时间步的权重进行插值。移位权重 $\mathbf{s}_t$ 决定权重是否旋转以及旋转多少。最后，根据 $\gamma_t$ 对权重进行锐化并用于记忆访问。

#### 3.3.1 基于内容聚焦

对于内容寻址，每个头（无论是用于读取还是写入）首先产生一个长度为 $M$ 的关键向量 $\mathbf{k}_t$，通过相似度度量 $\mathcal{K}(\cdot, \cdot)$ 与每个向量 $\mathbf{M}_t(i)$ 进行比较。基于内容的系统根据相似度和正的关键强度 $\beta_t$（可以放大或缩小聚焦的精度）产生归一化权重 $\mathbf{w}_t^c$：

$$w_t^c(i) \leftarrow \frac{\exp\left(\beta_t \mathcal{K}(\mathbf{k}_t, \mathbf{M}_t(i))\right)}{\sum_j \exp\left(\beta_t \mathcal{K}(\mathbf{k}_t, \mathbf{M}_t(j))\right)}. \qquad (5)$$

在当前实现中，相似度度量是余弦相似度：

$$\mathcal{K}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \cdot \|\mathbf{v}\|}. \qquad (6)$$

#### 3.3.2 基于位置聚焦

基于位置寻址机制旨在促进记忆位置的简单迭代和随机访问跳转。它通过实现权重的循环移位来实现这一点。例如，如果当前权重完全聚焦在单个位置上，则移位1会将焦点移到下一个位置。负移位则向相反方向移动权重。

在旋转之前，每个头发出一个标量插值门 $g_t$，其值在 $(0, 1)$ 范围内。$g$ 的值用于混合头在上一时间步产生的权重 $\mathbf{w}_{t-1}$ 和内容系统在当前时间步产生的权重 $\mathbf{w}_t^c$，得到门控权重 $\mathbf{w}_t^g$：

$$\mathbf{w}_t^g \leftarrow g_t \mathbf{w}_t^c + (1 - g_t) \mathbf{w}_{t-1}. \qquad (7)$$

如果门为零，则完全忽略内容权重，使用上一时间步的权重。反之，如果门为一，则忽略上一迭代的权重，系统应用基于内容寻址。

插值之后，每个头发出一个移位权重 $\mathbf{s}_t$，定义允许的整数移位上的归一化分布。例如，如果允许 -1 到 1 之间的移位，$\mathbf{s}_t$ 有三个元素，分别对应执行 -1、0 和 1 移位的程度。定义移位权重的最简单方法是使用连接到控制器的适当大小的softmax层。我们还实验了另一种技术，其中控制器发出一个标量，解释为宽度为1的均匀分布移位下限。例如，如果移位标量为6.7，则 $s_t(6) = 0.3$，$s_t(7) = 0.7$，其余 $s_t$ 为零。

如果将 $N$ 个记忆位置索引为 $0$ 到 $N-1$，则 $\mathbf{s}_t$ 对 $\mathbf{w}_t^g$ 施加的旋转可以表示为以下循环卷积：

$$\tilde{w}_t(i) \leftarrow \sum_{j=0}^{N-1} w_t^g(j) s_t(i-j) \qquad (8)$$

其中所有索引运算均以 $N$ 为模数进行。如果移位权重不够尖锐，方程(8)中的卷积运算可能导致权重随时间的泄漏或分散。例如，如果 -1、0 和 1 的移位权重分别为 0.1、0.8 和 0.1，则旋转会将聚焦于单个点的权重变换为略微模糊分布在三个点上的权重。为了解决这一问题，每个头再发出一个标量 $\gamma_t \geq 1$，其作用是按如下方式锐化最终权重：

$$w_t(i) \leftarrow \frac{\tilde{w}_t(i)^{\gamma_t}}{\sum_j \tilde{w}_t(j)^{\gamma_t}} \qquad (9)$$

权重插值与基于内容和基于位置寻址的组合寻址系统可以在三种互补模式下运行。第一，权重可以由内容系统选择而不被位置系统修改。第二，内容寻址系统产生的权重可以被选择然后移位。这使得焦点可以跳转到与内容访问地址相邻但不在其上的位置；用计算术语来说，这允许头找到连续的数据块，然后访问该块中的特定元素。第三，上一时间步的权重可以在没有任何基于内容寻址输入的情况下进行旋转。这允许权重通过在每一步前进相同距离来迭代遍历一系列地址。

### 3.4 控制器网络

上述NTM架构有几个自由参数，包括记忆大小、读写头的数量以及允许的位置移位范围。但可能最重要的架构选择是用作控制器的神经网络类型。具体来说，需要决定使用循环网络还是前馈网络。像LSTM这样的循环控制器有自己的内部记忆，可以补充矩阵中的更大记忆。如果将控制器比作数字计算机中的中央处理单元（尽管具有自适应而非预定义的指令）并将记忆矩阵比作RAM，则循环控制器的隐藏激活类似于处理器中的寄存器。它们允许控制器在多个操作时间步上混合信息。另一方面，前馈控制器可以通过在每一步对记忆中的相同位置进行读写来模拟循环网络。此外，前馈控制器通常赋予网络操作更大的透明性，因为从记忆矩阵读写的方式通常比RNN的内部状态更容易解释。然而，前馈控制器的一个局限性是并发读写头的数量对NTM可以执行的计算类型施加了瓶颈。使用单个读头，它每步只能对单个记忆向量执行一元变换；使用两个读头，它可以执行二元向量变换，依此类推。循环控制器可以在内部存储之前时间步的读向量，因此不具有这一限制。

## 4 实验

本节展示在一组简单算法任务上的初步实验，例如复制和排序数据序列。目标不仅是验证NTM能够解决问题，而且还验证它能通过学习紧凑的内部程序来做到这一点。此类解决方案的标志是它们能很好地泛化到远超训练数据范围的情况。例如，我们很好奇，一个经过训练可以复制长度不超过20的序列的网络，能否在没有进一步训练的情况下复制长度为100的序列。

在所有实验中，我们比较了三种架构：带前馈控制器的NTM、带LSTM控制器的NTM以及标准LSTM网络。由于所有任务都是情节性的，我们在每个输入序列开始时重置网络的动态状态。对于LSTM网络，这意味着将先前的隐藏状态设置为学习到的偏置向量。对于NTM，控制器的先前状态、先前读向量的值以及记忆的内容都重置为偏置值。所有任务都是具有二值目标的监督学习问题；所有网络都具有逻辑sigmoid输出层，并使用交叉熵目标函数进行训练。序列预测误差以每序列比特数报告。更多实验参数细节见第4.6节。

### 4.1 复制

复制任务测试NTM是否能够存储和回忆任意信息的长时间序列。网络接收随机二值向量的输入序列，后跟一个定界符标志。长期以来，长时期信息的存储和访问对RNN和其他动态架构来说一直是个问题。我们特别感兴趣的是，看看NTM是否能够桥接比LSTM更长的时间延迟。

网络被训练来复制八位随机向量的序列，序列长度在1到20之间随机。目标序列就是输入序列的副本（不含定界符标志）。注意，当网络接收目标时没有提供任何输入，以确保网络在没有任何中间辅助的情况下回忆整个序列。

<img src="figure3.png" alt="图3：复制学习曲线">

**图3：复制学习曲线。**

从图3可以看出，NTM（无论是前馈还是LSTM控制器）的学习速度都比单独的LSTM快得多，并且收敛到更低的代价。NTM与LSTM学习曲线之间的差异足够显著，表明两种模型在解决问题的方式上存在质的而非量的差异。

我们还研究了网络泛化到比训练时更长的序列的能力（它们能泛化到新向量这一点从训练误差可以明显看出）。图4和5展示了LSTM和NTM在这种情况下的行为截然不同。NTM随着长度增加继续复制[2]，而LSTM在超过长度20后迅速退化。

上述分析表明，与LSTM不同，NTM已经学习了某种形式的复制算法。为了确定这种算法是什么，我们检查了控制器与记忆之间的交互（图6）。我们认为网络执行的操作序列可以总结为以下伪代码：

```
初始化：将头移动到起始位置
while 未看到输入定界符 do
    接收输入向量
    将输入写入到头位置
    将头位置递增 1
end while
将头返回到起始位置
while true do
    从头位置读取输出向量
    发出输出
    将头位置递增 1
end while
```

这本质上就是人类程序员在低级编程语言中执行相同任务的方式。就数据结构而言，我们可以说NTM学会了如何创建和遍历数组。请注意，该算法结合了基于内容寻址（跳到序列起始位置）和基于位置寻址（沿序列移动）。还要注意，如果没有使用来自先前读写权重的相对移位的能力（方程7），迭代就不能泛化到长序列，并且如果没有聚焦锐化机制（方程9），权重可能会随时间失去精度。

<img src="figure4.png" alt="图4：NTM在复制任务上的泛化">

**图4：NTM在复制任务上的泛化。** 顶行的四对图分别展示了测试序列长度为10、20、30和50时的网络输出和相应的复制目标。底行的图对应长度为120的序列。该网络仅训练过长度不超过20的序列。前四个序列以高置信度再现，几乎没有错误。最长的序列有更多局部错误和一个全局错误：在底部红色箭头指示的位置，一个向量被重复，导致所有后续向量后退一步。尽管主观上接近正确复制，但这导致了高损失。

<img src="figure5.png" alt="图5：LSTM在复制任务上的泛化">

**图5：LSTM在复制任务上的泛化。** 这些图展示了与图4相同序列长度下的输入和输出。与NTM一样，LSTM学会了几乎完美地再现长度不超过20的序列。然而，它明显无法泛化到更长的序列。还要注意，准确前缀的长度随着序列长度的增加而减少，表明网络难以长时间保持信息。

<img src="figure6.png" alt="图6：NTM在复制任务中的记忆使用情况">

**图6：NTM在复制任务中的记忆使用情况。** 左列的图展示了复制任务单个测试序列期间网络的输入（顶部）、添加到记忆中的向量（中间）和相应的写入权重（底部）。右列的图展示了网络的输出（顶部）、从记忆中读取的向量（中间）和读取权重（底部）。仅显示了记忆位置的子集。注意所有权重对记忆中单个位置的锐利聚焦（黑色为零权重，白色为一权重）。还要注意焦点随时间的平移，反映了网络使用第3.3.2节所述的迭代移位进行基于位置寻址。最后，观察读取位置与写入位置完全匹配，读取向量与添加向量匹配。这表明网络在输入阶段依次将每个输入向量写入特定的记忆位置，然后在输出阶段从相同的位置序列读取。

### 4.2 重复复制

重复复制任务扩展了复制任务，要求网络将复制的序列输出指定次数，然后发出序列结束标记。主要动机是看看NTM能否学习简单的嵌套函数。理想情况下，我们希望它能够执行包含已学任何子例程的"for循环"。

网络接收随机长度的随机二值向量序列，后跟一个指示所需复制次数的标量值，该值出现在单独的输入通道上。为了在正确时间发出结束标记，网络必须既能解释额外输入，又能记录已执行的复制次数。与复制任务一样，在初始序列和重复次数之后，不再向网络提供输入。网络被训练来复制大小为八位的随机二值向量序列，其中序列长度和重复次数均在1到10之间随机选择。表示重复次数的输入被归一化为均值为零、方差为一。

<img src="figure7.png" alt="图7：重复复制学习曲线">

**图7：重复复制学习曲线。**

图7显示NTM学习任务的速度比LSTM快得多，但两者都能完美解决该任务[3]。两种架构之间的差异只有在要求泛化到训练数据之外时才变得明显。在这种情况下，我们感兴趣的是沿两个维度的泛化：序列长度和重复次数。图8展示了对于LSTM和NTM，先加倍其中一个、再加倍另一个的效果。LSTM两项测试都失败，而NTM在更长的序列上成功，并且能够执行超过十次重复；然而它无法记录已完成多少次重复，并且不能正确预测结束标记。这可能是由于以数值方式表示重复次数所致，这种方式不容易泛化到固定范围之外。

<img src="figure8.png" alt="图8：NTM和LSTM在重复复制任务上的泛化">

**图8：NTM和LSTM在重复复制任务上的泛化。** NTM几乎完美地泛化到比训练时更长的序列。当重复次数增加时，它能够相当准确地继续复制输入序列；但它无法预测序列何时结束，在第11次重复之后每次重复结束时都会发出结束标记。LSTM在长度和次数增加时都表现困难，两种情况都迅速偏离输入序列。

图9表明NTM学习了前一节中复制算法的简单扩展，即在必要时重复顺序读取。

<img src="figure9.png" alt="图9：NTM在重复复制任务中的记忆使用情况">

**图9：NTM在重复复制任务中的记忆使用情况。** 与复制任务一样，网络首先使用迭代移位将输入向量写入记忆。然后它读取整个序列以根据需要多次复制输入（本例中为六次）。读取权重底部的白点似乎对应一个中间位置，用于将头重定向到序列的起始位置（相当于NTM的goto语句）。

### 4.3 联想召回

前面的任务表明NTM可以将算法应用于相对简单的线性数据结构。数据组织的下一个复杂度来自"间接引用"——即一个数据项指向另一个数据项。我们通过构建一个item列表来测试NTM学习这一类更有趣的实例的能力，使得用其中一个item进行查询时，网络需要返回后续的item。更具体地说，我们将一个item定义为一个由左右定界符符号界定的二值向量序列。在多个item被传播到网络后，我们通过显示一个随机item进行查询，并要求网络产生下一个item。在我们的实验中，每个item由三个六位二值向量组成（每item共18位）。训练期间，我们使用最少2个item到最多6个item的情节。

<img src="figure10.png" alt="图10：NTM和LSTM的联想召回学习曲线">

**图10：NTM和LSTM的联想召回学习曲线。**

图10显示NTM学习此任务的速度显著快于LSTM，在大约30,000个情节内接近零代价终止，而LSTM在100万个情节后仍未达到零代价。此外，带前馈控制器的NTM比带LSTM控制器的NTM学习更快。这两个结果表明，NTM的外部记忆是维护数据结构比LSTM内部状态更有效的方式。NTM的泛化能力也远优于LSTM，如图11所示。带前馈控制器的NTM在最多12个item的序列上（训练中使用的最大长度的两倍）几乎完美，对于15个item的序列平均代价仍低于每序列1比特。

<img src="figure11.png" alt="图11：更长item序列上联想召回的泛化性能">

**图11：更长item序列上联想召回的泛化性能。** 无论是带前馈还是LSTM控制器的NTM，都能泛化到比单独的LSTM长得多的item序列。特别是，带前馈控制器的NTM在训练集序列长度两倍的item序列上几乎完美。

在图12中，我们展示了由一个LSTM控制器（带有一个头）控制的NTM记忆在单个测试情节中的操作。在"输入"中，我们看到输入将item定界符表示为第7行中的单个比特。在item序列传播完毕后，第8行中的定界符准备网络接收一个查询item。本例中，查询item对应序列中的第二个item（包含在绿色框中）。在"输出"中，我们看到网络清晰地输出序列中的item 3（来自红色框）。在"读取权重"中，在最后三个时间步上，我们看到控制器从连续的位置读取，每个位置存储了item 3的时间切片。这很有趣，因为网络似乎直接跳到了存储item 3的正确位置。但我们可以通过观察"写入权重"来解释这一行为。在这里，我们看到即使在输入呈现item之间的定界符符号时，记忆也在被写入。可以在"添加"中确认，当定界符呈现时数据确实被写入了记忆（例如黑色框内的数据）；此外，每次呈现定界符时，添加到记忆中的向量都不同。对记忆的进一步分析揭示，网络通过基于内容查找产生一个权重，然后将该权重移位一，来访问查询后读取的位置。此外，用于内容查找的关键向量对应于黑色框中添加的向量。这意味着以下记忆访问算法：当呈现每个item定界符时，控制器写入该item前面三个时间切片的压缩表示。在查询到达后，控制器重新计算查询item的相同压缩表示，使用基于内容查找找到写入第一个表示的位置，然后移位一以产生序列中的后续item（从而将基于内容查找与基于位置偏移相结合）。

<img src="figure12.png" alt="图12：NTM在联想召回任务中的记忆使用情况">

**图12：NTM在联想召回任务中的记忆使用情况。** 在"输入"中，一个由三个连续二值随机向量组成的item序列被传播给控制器。item之间的区别由定界符符号指定（"输入"中的第7行）。在几个item呈现之后，呈现一个指定查询的定界符（"输入"中的第8行）。呈现单个查询item（绿色框），网络目标对应于序列中的后续item（红色框）。在"输出"中，我们看到网络正确生成了目标item。读写权重中的红色框突出显示了目标item被写入然后读取的三个位置。网络找到的解决方案是对每个item形成一个压缩表示（"添加"中的黑色框），该表示可以存储在单个位置中。更多分析见正文。

### 4.4 动态N-Gram

动态N-Gram任务的目标是测试NTM能否快速适应新的预测分布。我们特别感兴趣的是，它是否能将其记忆用作可重写的表格来记录转移统计，从而模拟传统的N-Gram模型。

我们考虑所有可能的6-Gram二值序列分布。每个6-Gram分布可以表示为一个有 $2^5 = 32$ 个数字的表格，指定在所有可能长度为5的二值历史条件下下一个比特为1的概率。对于每个训练样本，我们首先通过从 Beta(1/2, 1/2) 分布中独立抽取所有32个概率来生成随机6-Gram概率。然后，我们使用当前查找表绘制200个连续比特生成一个特定的训练序列[4]。网络逐个比特地观察序列，然后被要求预测下一个比特。该问题的最优估计器可以通过贝叶斯分析得到（Murphy, 2012）[29]：

$$P(B = 1 | N_1, N_0, c) = \frac{N_1 + \frac{1}{2}}{N_1 + N_0 + 1} \qquad (10)$$

其中 $c$ 是五个比特的先前上下文，$B$ 是下一比特的值，$N_0$ 和 $N_1$ 分别是序列中到目前为止在上下文 $c$ 之后观察到的0和1的数量。因此我们可以将NTM与最优预测器以及LSTM进行比较。为了评估性能，我们使用了一个包含1000个长度为200的序列的验证集，这些序列从与训练数据相同的分布中采样。如图13所示，NTM相比LSTM取得了虽小但显著的性能优势，但从未完全达到最优代价。

<img src="figure13.png" alt="图13：动态N-Gram学习曲线">

**图13：动态N-Gram学习曲线。**

两种架构的预测在观察新输入时的演变如图14所示，同时还有最优预测。对NTM记忆使用的仔细分析（图15）表明，控制器使用记忆来记录在不同上下文中观察到的1和0的数量，使其能够实现类似于最优估计器的算法。

<img src="figure14.png" alt="图14：动态N-Gram推理">

**图14：动态N-Gram推理。** 顶行展示来自N-Gram任务的测试序列，下面几行展示最优估计器、NTM和LSTM发出的相应预测分布。在大多数地方，NTM预测与最优预测几乎无法区分。然而在两个箭头指示的点，它犯了明显的错误，其中一个错误在图15中解释。LSTM在某些地方紧密跟随最优预测，但随着序列的进行似乎偏离得更远；我们推测这是由于LSTM"遗忘"了序列开始时的观察。

<img src="figure15.png" alt="图15：NTM在动态N-Gram任务中的记忆使用情况">

**图15：NTM在动态N-Gram任务中的记忆使用情况。** 红色和绿色箭头指示测试序列中反复观察到相同上下文的位置（绿色箭头为"00010"，红色箭头为"01111"）。在每个这样的位置，读头访问相同的位置，然后在下一个时间步，写头访问该位置。我们假设网络使用写操作来记录到目前为止序列中每个上下文后紧跟1和0的比例。添加向量支持这一点，在输入为1或0的位置它们明显是反相关的，表明一个分布式的"计数器"。注意，随着相同上下文被反复看到，写入权重变得更弱；这可能是因为记忆记录的是1与0的比例而非绝对计数。预测序列中的红色框对应图14中第一个红色箭头处的错误；控制器似乎访问了错误的记忆位置，因为先前的上下文是"01101"而非"01111"。

### 4.5 优先级排序

该任务测试NTM能否对数据进行排序——这是一项重要的基础算法。一串随机二值向量序列被输入网络，同时每个向量带有一个标量优先级评分。优先级从范围 [-1, 1] 中均匀抽取。目标序列包含按优先级排序的二值向量，如图16所示。

每个输入序列包含20个带有对应优先级的二值向量，每个目标序列是输入中优先级最高的16个向量[5]。对NTM记忆使用的检查使我们假设它使用优先级来确定每次写入的相对位置。为了检验这个假设，我们拟合了一个从优先级到观察到的写入位置的线性函数。图17显示该线性函数返回的位置与观察到的写入位置密切匹配。它还显示网络按递增顺序从记忆位置读取，从而遍历排序后的序列。

<img src="figure16.png" alt="图16：优先级排序任务的示例输入和目标序列">

**图16：优先级排序任务的示例输入和目标序列。** 输入序列包含随机二值向量和随机标量优先级。目标序列是按优先级排序的输入向量子集。

<img src="figure17.png" alt="图17：NTM在优先级排序任务中的记忆使用情况">

**图17：NTM在优先级排序任务中的记忆使用情况。** 左：通过拟合优先级与观察到的写入位置之间的线性函数得到的写入位置。中：观察到的写入位置。右：读取位置。

图18的学习曲线表明，带有前馈和LSTM控制器的NTM在此任务上都显著优于LSTM。注意，在此任务上使用前馈控制器需要八个并行读写头才能达到最佳性能；这可能反映了仅使用一元向量操作对向量进行排序的难度（见第3.4节）。

<img src="figure18.png" alt="图18：优先级排序学习曲线">

**图18：优先级排序学习曲线。**

### 4.6 实验细节

在所有实验中，使用如（Graves, 2013）[13]所述形式的RMSProp算法进行训练，动量为 $0.9$。表1至表3给出了实验中使用的网络配置和学习率的详细信息。所有LSTM网络都有三个堆叠隐藏层。注意，LSTM的参数数量随着隐藏单元数量呈二次增长（由于隐藏层中的循环连接）。这与NTM形成对比，后者的参数数量不随记忆位置数量增加。在训练反向传播过程中，所有梯度分量按元素裁剪到范围 $(-10, 10)$。

**表1：带前馈控制器的NTM实验设置**

| 任务 | 头数量 | 控制器大小 | 记忆大小 | 学习率 | 参数数量 |
|------|--------|-----------|---------|-------|---------|
| 复制 | 1 | 100 | $128 \times 20$ | $10^{-4}$ | 17,162 |
| 重复复制 | 1 | 100 | $128 \times 20$ | $10^{-4}$ | 16,712 |
| 联想召回 | 4 | 256 | $128 \times 20$ | $10^{-4}$ | 146,845 |
| N-Gram | 1 | 100 | $128 \times 20$ | $3 \times 10^{-5}$ | 14,656 |
| 优先级排序 | 8 | 512 | $128 \times 20$ | $3 \times 10^{-5}$ | 508,305 |

**表2：带LSTM控制器的NTM实验设置**

| 任务 | 头数量 | 控制器大小 | 记忆大小 | 学习率 | 参数数量 |
|------|--------|-----------|---------|-------|---------|
| 复制 | 1 | 100 | $128 \times 20$ | $10^{-4}$ | 67,561 |
| 重复复制 | 1 | 100 | $128 \times 20$ | $10^{-4}$ | 66,111 |
| 联想召回 | 1 | 100 | $128 \times 20$ | $10^{-4}$ | 70,330 |
| N-Gram | 1 | 100 | $128 \times 20$ | $3 \times 10^{-5}$ | 61,749 |
| 优先级排序 | 5 | $2 \times 100$ | $128 \times 20$ | $3 \times 10^{-5}$ | 269,038 |

**表3：LSTM网络实验设置**

| 任务 | 网络大小 | 学习率 | 参数数量 |
|------|---------|-------|---------|
| 复制 | $3 \times 256$ | $3 \times 10^{-5}$ | 1,352,969 |
| 重复复制 | $3 \times 512$ | $3 \times 10^{-5}$ | 5,312,007 |
| 联想召回 | $3 \times 256$ | $10^{-4}$ | 1,344,518 |
| N-Gram | $3 \times 128$ | $10^{-4}$ | 331,905 |
| 优先级排序 | $3 \times 128$ | $3 \times 10^{-5}$ | 384,424 |

## 5 结论

我们引入了神经图灵机，这是一种同时借鉴生物工作记忆模型和数字计算机设计的神经网络架构。与传统神经网络一样，该架构端到端可微，可以通过梯度下降进行训练。我们的实验表明，它能够从示例数据中学习简单算法，并利用这些算法很好地泛化到训练范围之外。

## 致谢

许多人提供了富有洞见的见解，但我们特别要感谢Daan Wierstra、Peter Dayan、Ilya Sutskever、Charles Blundell、Joel Veness、Koray Kavukcuoglu、Dharshan Kumaran、Georg Ostrovski、Chris Summerfield、Jeff Dean、Geoffrey Hinton和Demis Hassabis。

## 参考文献

[1] Baddeley, A., Eysenck, M., and Anderson, M. (2009). Memory. Psychology Press.

[2] Bahdanau, D., Cho, K., and Bengio, Y. (2014). Neural machine translation by jointly learning to align and translate. arXiv preprint arXiv:1409.0473.

[3] Barrouillet, P., Bernardin, S., and Camos, V. (2004). Time constraints and resource sharing in adults' working memory spans. Journal of Experimental Psychology: General, 133(1):83.

[4] Chomsky, N. (1956). Three models for the description of language. Information Theory, IEEE Transactions on, 2(3):113–124.

[5] Das, S., Giles, C. L., and Sun, G.-Z. (1992). Learning context-free grammars: Capabilities and limitations of a recurrent neural network with an external stack memory. In Proceedings of The Fourteenth Annual Conference of Cognitive Science Society. Indiana University.

[6] Dayan, P. (2008). Simple substrates for complex cognition. Frontiers in neuroscience, 2(2):255.

[7] Eliasmith, C. (2013). How to build a brain: A neural architecture for biological cognition. Oxford University Press.

[8] Fitch, W., Hauser, M. D., and Chomsky, N. (2005). The evolution of the language faculty: clarifications and implications. Cognition, 97(2):179–210.

[9] Fodor, J. A. and Pylyshyn, Z. W. (1988). Connectionism and cognitive architecture: A critical analysis. Cognition, 28(1):3–71.

[10] Frasconi, P., Gori, M., and Sperduti, A. (1998). A general framework for adaptive processing of data structures. Neural Networks, IEEE Transactions on, 9(5):768–786.

[11] Gallistel, C. R. and King, A. P. (2009). Memory and the computational brain: Why cognitive science will transform neuroscience, volume 3. John Wiley & Sons.

[12] Goldman-Rakic, P. S. (1995). Cellular basis of working memory. Neuron, 14(3):477–485.

[13] Graves, A. (2013). Generating sequences with recurrent neural networks. arXiv preprint arXiv:1308.0850.

[14] Graves, A. and Jaitly, N. (2014). Towards end-to-end speech recognition with recurrent neural networks. In Proceedings of the 31st International Conference on Machine Learning (ICML-14), pages 1764–1772.

[15] Graves, A., Mohamed, A., and Hinton, G. (2013). Speech recognition with deep recurrent neural networks. In Acoustics, Speech and Signal Processing (ICASSP), 2013 IEEE International Conference on, pages 6645–6649. IEEE.

[16] Hadley, R. F. (2009). The problem of rapid variable creation. Neural computation, 21(2):510–532.

[17] Hazy, T. E., Frank, M. J., and O'Reilly, R. C. (2006). Banishing the homunculus: making working memory work. Neuroscience, 139(1):105–118.

[18] Hinton, G. E. (1986). Learning distributed representations of concepts. In Proceedings of the eighth annual conference of the cognitive science society, volume 1, page 12. Amherst, MA.

[19] Hochreiter, S., Bengio, Y., Frasconi, P., and Schmidhuber, J. (2001a). Gradient flow in recurrent nets: the difficulty of learning long-term dependencies.

[20] Hochreiter, S. and Schmidhuber, J. (1997). Long short-term memory. Neural computation, 9(8):1735–1780.

[21] Hochreiter, S., Younger, A. S., and Conwell, P. R. (2001b). Learning to learn using gradient descent. In Artificial Neural Networks—ICANN 2001, pages 87–94. Springer.

[22] Hopfield, J. J. (1982). Neural networks and physical systems with emergent collective computational abilities. Proceedings of the national academy of sciences, 79(8):2554–2558.

[23] Jackendoff, R. and Pinker, S. (2005). The nature of the language faculty and its implications for evolution of language (reply to fitch, hauser, and chomsky). Cognition, 97(2):211–225.

[24] Kanerva, P. (2009). Hyperdimensional computing: An introduction to computing in distributed representation with high-dimensional random vectors. Cognitive Computation, 1(2):139–159.

[25] Marcus, G. F. (2003). The algebraic mind: Integrating connectionism and cognitive science. MIT press.

[26] Miller, G. A. (1956). The magical number seven, plus or minus two: some limits on our capacity for processing information. Psychological review, 63(2):81.

[27] Miller, G. A. (2003). The cognitive revolution: a historical perspective. Trends in cognitive sciences, 7(3):141–144.

[28] Minsky, M. L. (1967). Computation: finite and infinite machines. Prentice-Hall, Inc.

[29] Murphy, K. P. (2012). Machine learning: a probabilistic perspective. MIT press.

[30] Plate, T. A. (2003). Holographic Reduced Representation: Distributed representation for cognitive structures. CSLI.

[31] Pollack, J. B. (1990). Recursive distributed representations. Artificial Intelligence, 46(1):77–105.

[32] Rigotti, M., Barak, O., Warden, M. R., Wang, X.-J., Daw, N. D., Miller, E. K., and Fusi, S. (2013). The importance of mixed selectivity in complex cognitive tasks. Nature, 497(7451):585–590.

[33] Rumelhart, D. E., McClelland, J. L., Group, P. R., et al. (1986). Parallel distributed processing, volume 1. MIT press.

[34] Seung, H. S. (1998). Continuous attractors and oculomotor control. Neural Networks, 11(7):1253–1258.

[35] Siegelmann, H. T. and Sontag, E. D. (1995). On the computational power of neural nets. Journal of computer and system sciences, 50(1):132–150.

[36] Smolensky, P. (1990). Tensor product variable binding and the representation of symbolic structures in connectionist systems. Artificial intelligence, 46(1):159–216.

[37] Socher, R., Huval, B., Manning, C. D., and Ng, A. Y. (2012). Semantic compositionality through recursive matrix-vector spaces. In Proceedings of the 2012 Joint Conference on Empirical Methods in Natural Language Processing and Computational Natural Language Learning, pages 1201–1211. Association for Computational Linguistics.

[38] Sutskever, I., Martens, J., and Hinton, G. E. (2011). Generating text with recurrent neural networks. In Proceedings of the 28th International Conference on Machine Learning (ICML-11), pages 1017–1024.

[39] Sutskever, I., Vinyals, O., and Le, Q. V. (2014). Sequence to sequence learning with neural networks. arXiv preprint arXiv:1409.3215.

[40] Touretzky, D. S. (1990). Boltzcons: Dynamic symbol structures in a connectionist network. Artificial Intelligence, 46(1):5–46.

[41] Von Neumann, J. (1945). First draft of a report on the edvac.

[42] Wang, X.-J. (1999). Synaptic basis of cortical persistent activity: the importance of nmda receptors to working memory. The Journal of Neuroscience, 19(21):9587–9603.

---

[1] 关于如何最好地描述容量限制仍然存在激烈的争论（Barrouillet et al., 2004）[3]。

[2] 限制因素是记忆的大小（128个位置），之后循环移位会折回，之前的写入会被覆盖。

[3] LSTM在此任务上比复制任务表现更好，这让我们感到惊讶。可能的原因是序列更短（最多10而非20），且LSTM网络更大，因此具有更多记忆容量。

[4] 前5个比特（由于上下文不足无法从表格中采样）从一个 $p = 0.5$ 的伯努利分布中独立同分布抽取。

[5] 我们将排序限制为16个，因为我们想知道NTM是否会使用深度为4的二叉堆排序来解决该任务。
