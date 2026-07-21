# 2013-PS-面向分布式机器学习的参数服务器

> Mu Li¹, Li Zhou¹, Zichao Yang¹, Aaron Li¹, Fei Xia¹, David G. Andersen¹, Alexander Smola¹`²` | ¹卡内基梅隆大学, ²Google 战略技术

本文提出了一种参数服务器框架用于解决分布式机器学习问题。核心内容：

- 数据和计算分布到**客户端节点**，服务器节点维护全局共享参数（稀疏向量和矩阵）
- 管理客户端与服务器之间的**异步数据通信**
- 支持灵活的**一致性模型**、**弹性可扩展性**和**容错性**
- 针对具有挑战性的非凸和非光滑问题给出算法和**理论分析**
- 在真实数据上展示了具有数十亿参数规模的实验可扩展性

关键发现：

- **异步通信通过并行使用 CPU、磁盘和网络提高了系统效率，但会带来数据不一致并可能减慢算法收敛速度**
- **提供最佳努力、最大延迟时间和用户自定义过滤器三种灵活的一致性模型，允许算法设计者在收敛速度与系统效率之间取得平衡**
- **使用分布式哈希表和 Paxos（Zookeeper）实现动态节点扩展和容错恢复（< 1 秒）**
- **在 256 AMD CPU 核心集群上实现了近线性的加速比：ℓ1-LR 加速 9 倍（31B 非零特征），RICA 加速 13.5 倍（1B 参数）**
- **理论上保证了非凸非光滑问题在弱一致性模型下收敛到驻点**

---

## 摘要

我们提出了一个参数服务器框架来求解分布式机器学习问题。数据和计算负载均分布到客户端节点上，而服务器节点维护全局共享参数，这些参数表示为稀疏向量和矩阵。该框架管理客户端与服务器之间的异步数据通信。框架支持灵活的一致性模型、弹性可扩展性和容错性。我们针对具有挑战性的非凸和非光滑问题给出了算法与理论分析。为了展示所提框架的可扩展性，我们展示了在具有数十亿参数的真实数据上的实验结果。

## 1. 引言

分布式优化和推理在解决大规模机器学习问题方面正变得越来越流行。由于数据在观测数量和参数数量上的增长，使用机器集群克服了单台机器无法足够快速地解决这些问题的困难。然而，实现高效的分布式算法并不容易。密集的计算工作负载和数据通信量都要求精心的系统设计。

值得指出的是，我们的系统所针对的场景超越了典型的集群计算场景——即研究者拥有专享的、数量适中的同构、高可靠机器的场景。也就是说，我们针对的是云计算场景——机器可能不可靠，任务可能被抢占，数据可能丢失，网络延迟和临时负载会导致更加多样化的性能表现。例如，众所周知，由于个别服务器的偶发减速、重启、迁移等，同步操作可能会显著降级。换句话说，我们针对的是适用于 Google、百度、Amazon、Microsoft 等的真实云计算场景，而不是低利用率、专享的高性能超级计算机集群。这需要一种更鲁棒的计算方法。

存在几种通用的分布式机器学习系统。基于 Hadoop [1] 的 Mahout [5] 和基于 Spark [29] 的 MLI [27] 采用了迭代 MapReduce [14] 框架。虽然由于保留了状态和优化的执行策略，Spark 明显优于 Hadoop MapReduce，但这两种方法都使用了同步迭代通信模式。这使得它们容易受到迭代机器学习算法中非均匀性能分布的影响，即那些可能在任意时刻变慢的机器。为了克服这一限制，分布式 GraphLab [21] 使用图抽象异步调度通信。然而，它缺乏基于 map/reduce 的框架的弹性可扩展性，并依赖粗粒度的快照进行恢复。此外，全局变量同步不是一个一等公民的原语。当然，除了这些通用框架外，还有许多系统针对特定应用而开发，例如 [3, 13, 24, 22, 28, 10, 15]。

我们发现许多推理问题在其参数化方面具有相当受限的结构，利用这种设计可以带来显著的收益。例如，广义线性模型通常使用单个巨大的参数向量，或主题模型使用一组稀疏向量。一般来说，许多相关的大规模图模型主要由少量板块组成，从而允许用少量在观测和机器之间共享的组件重复结构。通过批量执行这些操作以及为特定数据类型专门化同步原语，这提供了相当高的效率。

在本文中，我们专注于参数服务器的分布式优化方法。在该模型中，计算节点被分为客户端和服务器。每个客户端"拥有"一部分数据和计算负载，而服务器共同维护全局共享参数。这种架构思想并不新鲜：它已被应用于多种机器学习应用，包括潜在变量模型 [26, 2, 17]、图上的分布式推理 [3] 和深度学习 [13]。我们的目标是构建一个通用系统，其特性仅被先前工作部分支持：¹

**易用性。** 全局共享参数表示为（可能是稀疏的）向量和矩阵，这对于机器学习应用来说比广泛使用的 (key,value) 存储或表更方便的数据结构。提供高性能且便捷的多线程线性代数运算，如参数与本地训练数据之间的向量-矩阵乘法，以方便应用开发。

**高效性。** 节点之间的通信是异步的。重要的是，同步不阻塞计算。该框架允许算法设计者在算法收敛速度和系统效率之间取得平衡，而最佳折衷取决于数据、算法和硬件。

**弹性可扩展性。** 新节点可以在不重启正在运行的框架的情况下添加。例如，对于流式 sketch 或当部署参数服务器作为必须长时间保持可用的在线服务时，这一特性是可取的。我们使用分布式哈希表 [9] 允许新服务器节点在任何时候动态插入到集合中。

**容错性和持久性。** 反过来，节点故障是不可避免的，特别是在大规模使用商用服务器时。例如，3 年的 MTBF（平均故障间隔时间）在 1000 个节点上相当于每天一次故障。在工业部署中，调度器抢占会显著增加这一比率。

我们使用一种优化的数据复制架构，该架构高效地将数据存储在多个服务器节点上，以实现从节点故障中快速（不到 1 秒）恢复。此外，由于客户端节点相互独立，当一个客户端失败时，新的客户端可以像 MapReduce 能够重新调度新的 mapper 一样自动启动。

![图 1：客户端和服务器之间的通信模式。客户端处理数据，而服务器同步参数并执行全局更新。注意，客户端和服务器之间大部分代码是共享的，主要区别在于它们更新参数的方式。]()

**图 1：客户端和服务器之间的通信模式。** 客户端处理数据，而服务器同步参数并执行全局更新。注意，客户端和服务器之间大部分代码是共享的，主要区别在于它们更新参数的方式。

## 2. 架构

### 2.1 概述

参数服务器架构如上所示，有两类节点：服务器节点维护全局共享参数的一个分区（机器局部参数默认不同步）。它们彼此通信以复制和/或迁移参数，以实现可靠性和扩展。客户端节点执行大部分计算；服务器节点主要执行簿记和全局聚合步骤。每个客户端通常本地存储一部分训练数据，计算局部统计量如梯度。客户端仅与服务器节点通信，更新和检索共享参数。客户端可以被添加或移除；这需要将适当部分的训练数据集传输到新机器，并查询相应的参数集合。

参数服务器可以同时支持多个独立的参数向量（即通道）用于不同的算法。例如，当服务器可能正在存储一个正在被某些节点积极查询的操作模型的参数，同时也在使用另一组客户端节点训练一个供未来使用的新模型时，这很有用。这种方法大大简化了模型更新和部署，因为客户端只需切换通道即可。

### 2.2 应用示例

通过讨论一些表面上不同但都适合同一框架的用例，我们的模型最容易理解。需要理解的是，具体问题可能比这些示例更加多样和复杂。

**通过分布式次梯度迭代的风险最小化。** 目标是求解如下形式的优化问题

$$
F(w) = \sum_{i=1}^{n} \ell(x_i, y_i, w) + \Omega(w). \qquad (1)
$$

这里 $\ell(x_i, y_i, w)$ 是一个损失函数，例如回归或分类误差，它依赖于数据 $x_i$、标签 $y_i$ 和参数 $w$，且仅通过 $x_i$ 中的非零项发生依赖。优化算法通常迭代计算 $F(w)$ 的一阶梯度，这适合客户端-服务器架构：全局共享参数 $w$ 由服务器维护。客户端并行存储训练数据并计算其梯度：每个客户端取一组对 $(x_i, y_i)$ 以及计算更新梯度所需的 $w$ 的条目。在推理过程中，这些局部梯度由服务器聚合。新的更新后的 $w$ 值被发送回客户端。

**通过参数同步的风险最小化。** 目标与上述场景相同。然而，在这种情况下，局部参数更新在客户端侧执行，与服务器的通信仅用于参数同步，例如使用 ADMM [8] 的分布式变体。

**分布式 Gibbs 采样器。** 潜在变量模型通常在生成式设置中从观测数据 $X$ 推断辅助的未观测变量 $Z$。例如，在潜在狄利克雷分配（LDA）中，目标是通过主题的混合来解释观测到的文档 [7]。折叠 Gibbs 采样 [16] 是一种广泛使用的推理算法，它迭代地统计 (document,word)、(document,topic) 和 (word,topic) 上的计数，然后基于条件概率将主题重新分配给词。在参数服务器设置中，文档被划分到客户端，以便前两个统计量可以在本地计算。全局共享的词到主题分配则由服务器节点维护 [26, 2]，而客户端向服务器发送状态变化更新。

**深度学习。** 深度学习本质上迭代若干非线性函数类。虽然函数类本身相当紧凑地描述，但大量数据上的推理仍然需要对观测集进行并行化。[13] 描述了两种互补的（同步）变体：他们将变量集分解到多台机器上，并分别计算目标函数的不同部分。其次，他们将观测分解到不同的机器上。

**Sketches。** 通常数据 sketch [12, 6] 被设计为在给定的时间间隔内表现良好（例如通过计数自 sketch 初始化以来观测到的 item 数量），而不是存储完整的频率分布。此外，它们是为单机存储而设计的。使用一致性哈希使我们能够将事件流分布到多台机器上，从而提高吞吐量和精度。

## 3. 接口

### 3.1 键值向量

与现有方法的一个主要区别是我们假设键的索引集是有序的并且可能是密集的。这使我们能够使用向量语义，并批量发送大量数据，而不是处理单个 (key,value) 对。此外，它简化了内存管理和网络流量，并使我们免于为密集向量存储单独的索引集。注意，这种方法严格来说是典型 (key,value) 服务器所提供的超集。为具体起见，假设参数服务器仅持有这样一个向量。参数服务器将共享参数作为（稀疏）向量展示给客户端和服务器。应用程序可以将这些数据视为向量/矩阵或一组 key-value 对，以更方便的方式为准。可以使用它们的键访问或修改单个数据条目，例如风险最小化问题中的 feature_id，或 LDA 中的 word_id 和 topic_id 的组合。客户端或服务器还可以对整个向量执行线性代数运算，例如加法 $w + u$、计算 2-范数 $\|w\|_2$，以及更一般的运算 $\alpha A x + \beta y$，如 Level 1 BLAS 子程序所编码的那样。此外，如果局部训练数据也是向量或矩阵，参数可以容易地与之交互。

除了方便之外，这种接口设计还带来了高效的代码。利用向量和矩阵的结构，线性代数运算内部的执行针对空间和时间局部性进行了优化，这在 BLAS/Lapack [4] 等库中已有充分探索。也更容易对这些运算符的内部实现进行高效的多线程处理，并利用对稀疏和密集向量的 SIMD/向量支持。

### 3.2 Push 和 Pull

节点之间的数据通信通过两个操作捕获：push 和 pull。前者将共享参数的本地修改数据条目发送给其他节点，而后者检索远程修改。应用程序可以指定是使用新的局部值 $w_k$ 还是新的局部修改 $w_k - w^{(\text{synced})}_k$ 进行通信。换句话说，对 push 或 pull 请求的响应是因问题而异的。

参数服务器通过仅发送所需数据来最小化网络流量。例如，每个服务器节点通常只维护共享参数的一个段。当客户端 push 时，框架找到所有本地更新的数据条目，然后将每个条目发送给维护该条目键的服务器节点。在接收端，客户端通常只需要共享参数的一个子集。收到客户端的 pull 请求后，服务器仅返回客户端所需的特定键的条目——该列表要么包含在 pull 请求中，要么预先与服务器协商以进一步减少流量。²

push 和 pull 操作都是非阻塞的。调用者（通常是计算线程）将其请求插入队列，然后恢复计算。由框架管理的独立 I/O 线程执行实际的网络通信。这种异步通信产生了一种数据一致性模型，我们在第 4.1 节中解释，并在第 5 节中进行理论分析。

### 3.3 服务器上的用户自定义函数

除了聚合来自客户端的数据外，服务器节点还可以执行用户自定义函数。这可能是有益的，因为服务器节点通常拥有关于共享参数的更完整或更新的信息。例如，考虑用于求解风险最小化的近端梯度方法，这将在第 5 节中进一步讨论。在每次迭代中，该算法首先聚合损失函数的梯度，然后通过与梯度和正则化子相关的近端算子求解新的 $w$。例如，对于 $\ell_1$ 正则化子，这就是软收缩算子。在服务器节点上而不是客户端节点上使用服务器端函数来求解近端算子，减少了必须在节点之间移动的数据量。同样，在 sketch 的上下文中，客户端几乎不执行任何操作，而大部分工作发生在服务器上。

## 4. 迈向规模和可靠性

### 4.1 一致性模型

异步通信通过并行使用 CPU、磁盘和网络带宽提高了系统效率。然而，它带来了节点间的数据不一致，并可能减慢优化算法的收敛速度。系统效率和算法收敛速度之间的最佳权衡通常取决于多种因素，包括算法对数据不一致的敏感性、训练数据中的特征相关性以及硬件组件的容量差异。参数服务器不局限于某一特定策略，而是提供灵活的数据一致性模型供应用选择：

**最佳努力（Best Effort）。** 在这种情况下，无论资源可用性如何，参数服务器都不会停顿。例如，[26] 描述了这样一个系统。然而，只有当底层算法对延迟具有鲁棒性时，这才值得推荐。

**最大延迟时间（Maximal Delayed Time）。** 当为 push 操作设置了最大延迟时间 $\tau$ 时，新的 push 调用将被阻塞，直到 $\tau$ 时间前的所有先前 push 调用都已完成。换句话说，如果使用迭代次数作为（逻辑）时间并设置 $\tau = 2$，那么在迭代 4 调用 push 时，如果迭代 3 之前的任何 push 操作尚未完成（即与该操作关联的网络包尚未成功发送），则该调用将被阻塞。因此，如果 $\tau = 0$，我们就得到批量同步并行模型，其中每次 push 调用都将被阻塞，直到数据已发送。

还要注意，对于无限延迟 $\tau = \infty$，我们得到 best-effect 模型 [26]。同样适用于 pull 操作。

**用户自定义过滤器（User-defined Filters）。** 参数服务器支持用户自定义过滤器用于选择性同步。一个例子是显著修改过滤器（significantly modified filter），它只推送那些变化超过显著量的条目，例如

$$
|w_k - w^{(\text{synced})}_k| > \Delta.
$$

一个直观的选择是在开始时使用大的 $\Delta$，然后在接近解时不断减小 $\Delta$。

### 4.2 弹性可扩展性和容错性

我们使用共享参数的 key-value 对视图。基本思想来自分布式哈希表 [9, 25]，其中 key-value 对和服务器节点都被插入哈希环中。每个节点管理从其插入点开始到环中其他节点在逆时针方向的下一个点之间的键段，这被称为逆时针邻居。

在右侧所示的示例中，服务器节点管理相同颜色的段。与执行键发现和路由的 [18] 不同，我们使用 DHT 进行分配，并将键段到节点的映射存储在 Paxos [19] 中，如 Zookeeper 中所实现的那样。注意，一个物理节点以虚拟节点的形式被插入 $\log p$ 次，以促进负载均衡。

每个键段随后被复制到其 $k$ 个逆时针邻居服务器节点以实现容错。如果 $k = 1$，则示例中带有标记的段将在服务器 3 处被复制。一个新节点首先（通过哈希函数）随机插入到环中，然后从其顺时针邻居处获取键段。另一方面，如果一个节点被移除或发生故障，其段将由最近的逆时针邻居提供服务，如果 $k > 0$，这些邻居已经拥有一个副本。要恢复故障节点，我们只需将一个节点插回故障节点之前的位置，然后从其逆时针邻居请求段数据。

## 5. 理论分析

### 5.1 非凸和非光滑优化

我们为以下非凸优化问题提供收敛性分析

$$
\operatorname{minimize} \; F(w) := f(w) + h(w) \quad \text{对于 } w \in \mathcal{X}. \qquad (2)
$$

这里 $f: \mathbb{R}^p \to \mathbb{R}$ 是连续可微但不一定是凸的，而 $h: \mathbb{R}^p \to \mathbb{R} \cup \{\infty\}$ 是下半连续、凸但可能非光滑的。

我们考虑近端梯度方法 [11, 23]。给定一个闭真凸函数 $h(w): \mathbb{R}^p \to \mathbb{R} \cup \{+\infty\}$，定义近端算子为

$$
\operatorname{Prox}_\gamma(x) = \underset{y \in \mathcal{X}}{\operatorname{argmin}} \; h(y) + \frac{1}{2\gamma} \|x - y\|^2.
$$

为了最小化组合目标函数 $f(w) + h(w)$，近端梯度方法分两步更新 $w$：对 $f$ 执行最速梯度下降的前向步骤，以及使用 $h$ 执行投影的后向步骤。对于迭代 $t$ 处给定的学习率 $\gamma_t > 0$，这可以写为

$$
w(t + 1) = \operatorname{Prox}_{\gamma_t} \big[ w(t) - \gamma_t \nabla f(w(t)) \big] \quad \text{对于 } t \in \mathbb{N}. \qquad (3)
$$

### 5.2 异步与收敛保证

为简化讨论，我们将服务器节点视为一个单一单元。这一简化不影响分析的正确性。然而，我们假设近端算子 $\operatorname{Prox}_\gamma(x)$ 可以在参数的分布式表示所诱导的分区上独立执行。例如，对于 $\ell_1$ 惩罚，这显然成立，因为近端算子是逐坐标作用的。

首先，假设数据被划分为 $m$ 个客户端，因此我们可以重写 $f(w) = \sum_{i=1}^{m} f_i(w)$。接下来，在每次迭代中，每个客户端 $i$ 同时计算局部梯度 $\nabla f_i$，然后将其 push 到服务器。注意，每个客户端按照自己的节奏运行，在每次迭代开始时没有同步。同时，服务器收集来自客户端的更新，并在求解近端算子后将 $w$ 的新值推回。

我们分析"最大延迟时间"模型与"显著修改"过滤器的组合。假设前者允许最大延迟 $\tau$ 次迭代。换句话说，如果客户端 $i$ 在迭代 $t$ 时具有值 $w(t_i)$，那么我们有 $t - \tau \le t_i \le t$。对于后者，假设仅当其绝对局部修改大于 $\Delta_t$ 时才发送值。因此，在迭代 $t$ 时，服务器从客户端 $i$ 接收到的梯度为

$$
G_i(t) = \nabla f_i(w(t_i) + \sigma_w(t_i)) + \sigma_{\nabla i}(t).
$$

这里 $w(t_i)$ 是客户端 $i$ 处的延迟参数副本，$\sigma_w$ 和 $\sigma_{\nabla}$ 是由于小变化值被过滤而产生的误差，满足 $\|\sigma_w(t_i)\|_\infty \le \Delta_{t_i}$ 和 $\|\sigma_{\Delta}(t)\|_\infty \le \Delta_t$。服务器用于计算 $w(t+1)$ 的不精确梯度为 $s(t) = \sum_{i=1}^{m} G_i(t)$。

以下定理 2 表明，如果选择合适的学习率和 $\Delta_t$，则在上述弱一致性模型下，该算法保证收敛到驻点。该定理需要以下假设。

**假设 1（Lipschitz 连续性）** 存在正常数 $L_i$，使得对于任意 $x, y \in \mathcal{X}$ 和所有 $i = 1, \ldots, m$，有 $\|\nabla f_i(x) - \nabla f_i(y)\| \le L_i \|x - y\|$。

**定理 2** 假设假设 1 成立，并记 $L = \sum_{i=1}^{m} L_i$。设 $\tau$ 为最大延迟，对于显著修改过滤器设 $\Delta_t = O(\frac{1}{t})$。对于任意 $\epsilon > 0$，如果学习率 $\gamma_t$ 满足 $\gamma_t \le ((1 + \tau)L + \epsilon)^{-1}$ 对所有 $t > 0$ 成立，则异步实现将收敛到驻点。

**证明。** 记 $\delta(t) = w(t + 1) - w(t)$，我们首先给出从迭代 $t$ 到 $t+1$ 时 $F$ 变化的上界。注意到 $w(t+1) = \operatorname{Prox}_{\gamma_t}(w(t) - \gamma_t s(t))$，对近端算子定义两边求导可得 $\frac{1}{\gamma_t}(w(t) - w(t + 1)) - s(t) \in \partial h(w(t + 1))$。由 $h$ 的凸性，

$$
\begin{aligned}
h(w(t + 1)) - h(w(t)) &\le \left\langle \frac{1}{\gamma_t}(w(t) - w(t + 1)) - s(t), \; w(t + 1) - w(t) \right\rangle \\
&= -\frac{1}{\gamma_t}\|\delta(t)\|^2 - \langle s(t), \delta(t) \rangle. \qquad (4)
\end{aligned}
$$

另一方面，应用假设 1，

$$
\begin{aligned}
f(w(t + 1)) - f(w(t)) &= \sum_{i=1}^{m} f_i(w(t + 1)) - f_i(w(t)) \qquad (5) \\
&\le \sum_{i=1}^{m} \big\langle \delta(t), \nabla f_i(w(t)) \big\rangle + L_i \|\delta(t)\|^2 \qquad (6) \\
&= \big\langle \delta(t), \nabla f(w(t)) \big\rangle + L \|\delta(t)\|^2. \qquad (7)
\end{aligned}
$$

结合 (4) 和 (7)，我们有

$$
\begin{aligned}
F(w(t + 1)) - F(w(t))
&\le \left(L - \frac{1}{\gamma_t}\right) \|\delta(t)\|^2 + \big\langle \delta(t), \nabla f(w(t)) - s(t) \big\rangle \\
&\le \left(L - \frac{1}{\gamma_t}\right) \|\delta(t)\|^2 + \|\delta(t)\| \|\nabla f(w(t)) - s(t)\|. \qquad (8)
\end{aligned}
$$

接下来我们给出梯度 $\nabla f(w(t))$ 与不精确梯度 $s(t)$ 之差的上界。根据 $s(t)$ 的计算方式，

$$
\begin{aligned}
\|\nabla f(w(t)) - s(t)\|
&= \left\| \sum_{i=1}^{m} \nabla f_i(w(t)) - \nabla f_i(w(t_i) + \sigma_w(t_i)) - \sigma_{\nabla i}(t) \right\| \\
&\le \sum_{i=1}^{m} \sum_{j=1}^{t - t_i} \big\| \nabla f_i(w(t - j + 1)) - \nabla f_i(w(t - j)) \big\| \\
&\qquad + \big\| \nabla f_i(w(t_i)) - \nabla f_i(w(t_i) + \sigma_w(t_i)) \big\| + \big\| \sigma_{\nabla i}(t) \big\| \\
&\qquad \text{(三角不等式)} \\
&\le \sum_{i=1}^{m} \sum_{j=1}^{t - t_i} L_i \|\delta(t - j)\| + L_i \|\sigma_w(t_i)\| + L_i \|\sigma_{\nabla i}(t)\| \\
&\qquad \text{(假设 1)} \\
&\le \sum_{i=1}^{m} \sum_{j=1}^{\tau} L_i \|\delta(t - j)\| + 2L_i \sqrt{p} \Delta_{t - \tau} \\
&\qquad (\text{延迟} \le \tau \text{ 且 } \|w\| \le \sqrt{p} \|w\|_\infty) \\
&= \sum_{j=1}^{\tau} L \|\delta(t - j)\| + 2L \sqrt{p} \Delta_{t - \tau}. \qquad (9)
\end{aligned}
$$

将 (9) 代入 (8)，

$$
\begin{aligned}
F(w(t + 1)) - F(w(t))
&\le \left(L - \frac{1}{\gamma_t}\right) \|\delta(t)\|^2 + \sum_{j=1}^{\tau} L \|\delta(t)\| \|\delta(t - j)\| + \|\delta(t)\| 2L \sqrt{p} \Delta_{t - \tau} \\
&\le \left(L + \frac{\tau L}{2} + \frac{\epsilon}{2} - \frac{1}{\gamma_t}\right) \|\delta(t)\|^2 + \sum_{t=1}^{\tau} \frac{L}{2} \|\delta(t - j)\|^2 + \frac{2L^2 p}{\epsilon} \Delta_{t - \tau}^2. \qquad (10)
\end{aligned}
$$

对 $t$ 求和，我们得到以下链式不等式

$$
F(w(T + 1)) - F(w(0))
\le \sum_{t=0}^{T} \left(\frac{(1 + \tau)L + \epsilon}{2} - \frac{1}{\gamma_t}\right) \|\delta(t)\|^2 + \sum_{t=1}^{T} \frac{2L^2 p}{\epsilon} \Delta_{t - \tau}^2. \qquad (11)
$$

记 $c(t) = \frac{1}{\gamma_t} - (1 + \tau)L - \frac{\epsilon}{2}$，由于对所有 $t$ 有 $\gamma_t \le \frac{1}{(1 + \tau)L + \epsilon}$，则所有 $c(t) \ge \frac{\epsilon}{2} > 0$。因此

$$
\frac{\epsilon}{2} \sum_{t=0}^{T} \|\delta(t)\|^2
\le \sum_{t=0}^{T} c(t) \|\delta(t)\|^2
\le F(x(0)) - F(x(T + 1)) + \sum_{t=1}^{T} \frac{2L^2 p}{\epsilon} \Delta_{t - \tau}^2. \qquad (12)
$$

对于任意 $T$ 成立。由于 $\Delta_t = O(\frac{1}{t})$，并且利用 $1 + \frac{1}{2^2} + \frac{1}{3^2} + \ldots = \frac{\pi^2}{6}$ 的事实。则当 $T \to \infty$ 时 (12) 的右端是常数，这意味着 $\lim_{t \to \infty} \delta(t) \to 0$。所以 $\lim_{t \to \infty} \operatorname{Prox}_{\gamma_t}(w(t)) - w(t) \to 0$，因此我们找到了一个局部极小点。

直观上，当接近驻点时，$w(t - \tau)$ 与 $w(t)$ 之间的差异会很小。此外，由于 $\Delta_t$ 递减到零，$\sigma_w$ 和 $\sigma_{\nabla i}$ 中的值也递减到零。因此，不精确梯度 $s(t)$ 将是真实梯度 $\nabla f(w(t))$ 的良好近似，因此可以应用近端梯度方法的收敛性结果。

## 6. 实验结果

在本节中，我们求解两个具有挑战性的问题——具有稀疏训练数据的 $\ell_1$ 正则化逻辑回归（$\ell_1$-LR）：

$$
\min_{w \in \mathbb{R}^p} \sum_{i=1}^{n} \log(1 + \exp(-y_i \langle x_i, w \rangle)) + \lambda \|w\|_1,
$$

以及具有密集训练数据的重建 ICA（RICA）：

$$
\min_{W \in \mathbb{R}^{\ell \times p}} \sum_{i=1}^{n} \frac{1}{2} \|W W^\top x_i - x_i\|_2^2 + \lambda \|W x_i\|_1.
$$

前者是凸的，但后者是高度非凸的。我们使用异步近端梯度方法。近端算子在 $\ell_1$-LR 上具有闭式解，即软收缩算子。对于 RICA，我们通过 ADMM [8] 来求解。更多细节见 [20]。

我们分别在具有 256 个 AMD CPU 核心和 16 个 Tesla K20 显卡的集群上运行 $\ell_1$-LR 和 RICA。评估了一系列稀疏和密集数据集。CTRb 和 imagenet 是最大的两个。前者是来自一家主要搜索公司的稀疏广告点击率数据集。它包含 0.34B 个观测，2.2B 个唯一特征，以及 31B 个非零条目。后者来自 http://www.image-net.org，包含 10 万张调整为 100x100 像素的图像。

由于篇幅限制，我们仅报告这两个数据集上在最大延迟 $\tau = 4$ 时的可扩展性结果。对于 RICA，我们使用 1B 个参数 $W \in \mathbb{R}^{10^6 \times 10^4}$。我们在图 2 中测量了可扩展性，即达到相同收敛精度所需的训练时间加速比。对于 $\ell_1$-LR，当客户端数量增加 16 倍时，观察到 9 倍的加速，而对于 RICA，我们看到了 13.5 倍的加速。

![图 2：两个问题及其相关分解的线性可扩展性。左：具有 31B 属性的二分类问题上的优化。右：ImageNet 上 RICA 的问题分解。两个实例在超过两个数量级上基本呈现完美的加速比。]()

**图 2：两个问题及其相关分解的线性可扩展性。** 左：具有 31B 属性的二分类问题上的优化。右：ImageNet 上 RICA 的问题分解。两个实例在超过两个数量级上基本呈现完美的加速比。

## 7. 结论

在本文中，我们描述了一个用于求解分布式机器学习问题的参数服务器框架。该框架易于使用：全局共享参数可以作为局部稀疏向量或矩阵使用，与局部训练数据执行线性代数运算。它高效：所有通信都是异步的，支持灵活的一致性模型以平衡系统效率和快速算法收敛速度之间的折衷。此外，它提供了弹性可扩展性和容错性，旨在稳定的长期部署。我们在弱数据一致性要求下给出了收敛性分析。最后，我们展示了两个具有数十亿变量真实数据集上具有挑战性任务的实验，以证明线性可扩展性。

## 参考文献

[1] Apache hadoop, 2009. http://hadoop.apache.org/core/.

[2] Amr Ahmed, Mohamed Aly, Joseph Gonzalez, Shravan Narayanamurthy, and A. J. Smola. Scalable inference in latent variable models. In *Proceedings of The 5th ACM International Conference on Web Search and Data Mining (WSDM)*, 2012.

[3] Amr Ahmed, Nino Shervashidze, Shravan Narayanamurthy, Vanja Josifovski, and Alexander J. Smola. Distributed large-scale natural graph factorization. In *World Wide Web Conference*, Rio de Janeiro, 2013.

[4] E. Anderson, Z. Bai, C. Bischof, J. Demmel, J. Dongarra, J. Du Croz, A. Greenbaum, S. Hammarling, A. McKenney, S. Ostrouchov, and D. Sorensen. *LAPACK Users' Guide*. SIAM, Philadelphia, second edition, 1995.

[5] Apache Foundation. Mahout project, 2012. http://mahout.apache.org.

[6] R. Berinde, G. Cormode, P. Indyk, and M.J. Strauss. Space-optimal heavy hitters with strong error bounds. In J. Paredaens and J. Su, editors, *Proceedings of the Twenty-Eigth ACM SIGMOD-SIGACT-SIGART Symposium on Principles of Database Systems, PODS*, pages 157–166. ACM, 2009.

[7] D. Blei, A. Ng, and M. Jordan. Latent Dirichlet allocation. *Journal of Machine Learning Research*, 3:993–1022, January 2003.

[8] S. Boyd, N. Parikh, E. Chu, B. Peleato, and J. Eckstein. Distributed optimization and statistical learning via the alternating direction method of multipliers. *Foundations and Trends in Machine Learning*, 3(1):1–123, 2010.

[9] J. Byers, J. Considine, and M. Mitzenmacher. Simple load balancing for distributed hash tables. In *Peer-to-peer systems II*, pages 80–87. Springer, 2003.

[10] W.Y. Chen, Y. Song, H. Bai, C.J. Lin, and E.Y. Chang. Parallel spectral clustering in distributed systems. *Pattern Analysis and Machine Intelligence, IEEE Transactions on*, 33(3):568–586, 2011.

[11] P. L. Combettes and J. C. Pesquet. Proximal splitting methods in signal processing. In *Fixed-Point Algorithms for Inverse Problems in Science and Engineering*, pages 185–212. Springer, 2011.

[12] G. Cormode and S. Muthukrishnan. Summarizing and mining skewed data streams. In *SDM*, 2005.

[13] J. Dean, G. Corrado, R. Monga, K. Chen, M. Devin, Q. Le, M. Mao, M. Ranzato, A. Senior, P. Tucker, K. Yang, and A. Ng. Large scale distributed deep networks. In *Neural Information Processing Systems*, 2012.

[14] J. Dean and S. Ghemawat. MapReduce: simplified data processing on large clusters. *CACM*, 51(1):107–113, 2008.

[15] John Duchi, Alekh Agarwal, and Martin Wainwright. Distributed dual averaging in networks. In *Advances in Neural Information Processing Systems 23*, 2010.

[16] T.L. Griffiths and M. Steyvers. Finding scientific topics. *Proceedings of the National Academy of Sciences*, 101:5228–5235, 2004.

[17] Q. Ho, J. Cipar, H. Cui, S. Lee, J. Kim, P. Gibbons, G. Gibson, G. Ganger, and E. Xing. More effective distributed ml via a stale synchronous parallel parameter server. In *NIPS*, 2013.

[18] D. Karger, E. Lehman, T. Leighton, M. Levine, D. Lewin, and R. Panigrahy. Consistent hashing and random trees: Distributed caching protocols for relieving hot spots on the world wide web. In *Symposium on the Theory of Computing STOC*, pages 654–663, New York, May 1997. Association for Computing Machinery.

[19] L. Lamport. Paxos made simple. *ACM Sigact News*, 32(4):18–25, 2001.

[20] M. Li, D. G. Andersen, and A. J. Smola. Distributed delayed proximal gradient methods. In *NIPS Workshop on Optimization for Machine Learning*, 2013.

[21] Yucheng Low, Joseph Gonzalez, Aapo Kyrola, Danny Bickson, Carlos Guestrin, and Joseph M. Hellerstein. Distributed graphlab: A framework for machine learning and data mining in the cloud. In *PVLDB*, 2012.

[22] N. Parikh and S. Boyd. Graph projection block splitting for distributed optimization, 2012. submitted.

[23] N. Parikh and S. Boyd. Proximal algorithms. *To appear in Foundations and Trends in Optimization*, 2013.

[24] P. Richtarik and M. Takac. Distributed coordinate descent method for learning with big data. Technical report, 2013.

[25] Antony Rowstron and Peter Druschel. Pastry: Scalable, decentralized object location and routing for large-scale peer-to-peer systems. In *IFIP/ACM International Conference on Distributed Systems Platforms (Middleware)*, pages 329–350, Heidelberg, Germany, November 2001.

[26] A. J. Smola and S. Narayanamurthy. An architecture for parallel topic models. In *Very Large Databases (VLDB)*, 2010.

[27] E. Sparks, A. Talwalkar, V. Smith, J. Kottalam, X. Pan, J. Gonzalez, M. J. Franklin, M. I. Jordan, and T. Kraska. Mli: An api for distributed machine learning. 2013.

[28] Christina Teflioudi, Faraz Makari, and Rainer Gemulla. Distributed matrix completion. In *Data Mining (ICDM), 2012 IEEE 12th International Conference on*, pages 655–664. IEEE, 2012.

[29] Matei Zaharia, Mosharaf Chowdhury, Tathagata Das, Ankur Dave, Justin Ma, Murphy Mccauley, Michael J. Franklin, Scott Shenker, and Ion Stoica. Fast and interactive analytics over hadoop data with spark. *USENIX ;login:*, 37(4):45–51, August 2012.

¹ C++ 代码可在 http://parameterserver.org/ 获取
² 通过省略键列表而只发送值，我们可以使网络吞吐量翻倍。例如，通过仅传输一段键范围的校验和而不是实际键，这很容易实现。
