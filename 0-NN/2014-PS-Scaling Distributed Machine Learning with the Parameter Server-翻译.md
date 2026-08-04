# Scaling Distributed Machine Learning with the Parameter Server

> Mu Li, Carnegie Mellon University and Baidu; David G. Andersen and Jun Woo Park, Carnegie Mellon University; Alexander J. Smola, Carnegie Mellon University and Google, Inc.; Amr Ahmed, Vanja Josifovski, James Long, Eugene J. Shekita, and Bor-Yiing Su, Google, Inc.

本文发表于 Proceedings of the 11th USENIX Symposium on Operating Systems Design and Implementation (OSDI '14)，2014年10月6–8日，Broomfield, CO。

---

## 摘要

我们提出一个用于分布式机器学习问题的参数服务器框架。数据和计算负载均分布在 worker 节点上，而 server 节点维护全局共享的参数，这些参数表示为稠密或稀疏的向量和矩阵。该框架管理节点之间的异步数据通信，并支持灵活的一致性模型、弹性可扩展性和持续的容错能力。为了展示所提出框架的可扩展性，我们展示了在 PB 级真实数据上的实验结果，数据包含数十亿的样本和参数，涉及从稀疏逻辑回归到潜在狄利克雷分配（Latent Dirichlet Allocation）和分布式草图（Distributed Sketching）等问题。

---

## 1 引言

分布式优化和推理正成为解决大规模机器学习问题的前提条件。在大规模场景下，由于数据和由此产生的模型复杂性的增长（通常表现为参数数量的增加），没有单台机器能够足够快速地解决这些问题。然而，实现高效的分布式算法并不容易。密集的计算工作负载和数据通信量都要求精心的系统设计。

实际规模的训练数据量范围可以从 1TB 到 1PB。这使得可以创建具有 $10^9$ 到 $10^{12}$ 个参数的强大而复杂的模型 [9]。这些模型通常被所有 worker 节点全局共享，worker 节点在执行计算以改进模型时必须频繁访问共享参数。共享带来了三个挑战：

- 访问参数需要巨大的网络带宽。
- 许多机器学习算法是顺序的。当同步成本和机器延迟很高时，由此产生的屏障（barrier）会损害性能。
- 在大规模场景下，容错能力至关重要。学习任务通常在云环境中执行，其中机器可能不可靠且任务可能被抢占。

为了说明最后一点，我们收集了一个大型互联网公司中一个集群三个月的所有任务日志。我们在表1中展示了服务于生产环境的批量机器学习任务的统计数据。在这里，任务失败主要是由于被抢占或机器丢失，而缺乏必要的容错机制。与许多研究环境中任务在无争用的集群上独占运行不同，在实际部署中容错是必需品。

### 1.1 贡献

自从参数服务器框架 [43] 被引入以来，它已经在学术界和工业界得到广泛应用。本文描述了一个第三代开源参数服务器实现，其重点在于分布式推理的系统层面。它为开发者带来两个优势：首先，通过分离出机器学习系统中常用的组件，使得特定于应用的代码保持简洁。同时，作为一个面向系统级优化的共享平台，它提供了一个健壮、通用且高性能的实现，能够处理从稀疏逻辑回归到主题模型和分布式草图等多种算法。我们的设计决策由实际系统中的工作负载所指导。我们的参数服务器提供了五个关键特性：

**高效通信：** 异步通信模型不会阻塞计算（除非请求）。它针对机器学习任务进行了优化，以减少网络流量和开销。

**灵活的一致性模型：** 宽松的一致性进一步隐藏了同步成本和延迟。我们允许算法设计者平衡算法收敛速度和系统效率。最佳权衡取决于数据、算法和硬件。

**弹性可扩展性：** 可以在不重启运行中框架的情况下添加新节点。

**容错与持久性：** 能够在1秒内从非灾难性机器故障中恢复和修复，且不中断计算。向量时钟确保在网络分区和故障后的行为明确定义。

**易于使用：** 全局共享参数表示为（可能是稀疏的）向量和矩阵，以促进机器学习应用的开发。线性代数数据类型附带高性能多线程库。

所提出系统的新颖之处在于选择正确的系统技术、使其适应机器学习算法、并修改机器学习算法使其更加系统友好的协同效应。特别地，由于相关的机器学习算法对扰动具有相当强的容忍度，我们可以放宽一些原本困难的系统约束。其结果是第一个能够扩展到工业规模大小的通用机器学习系统。

### 1.2 工程挑战

在解决分布式数据分析问题时，读取和更新不同 worker 节点之间共享的参数是普遍存在的问题。参数服务器框架为聚合和同步 worker 之间的模型参数和统计数据提供了一种高效机制。每个参数服务器节点只维护参数的一部分，而每个 worker 节点在运行时通常只需要这些参数的一个子集。构建高性能参数服务器系统面临两个关键挑战：

**通信。** 虽然参数可以在传统数据存储中作为键值对（key-value pairs）更新，但直接使用这种抽象是低效的：值通常很小（浮点数或整数），而将每个更新作为键值操作发送的开销很高。我们改进这种情况的洞察来自一个观察：许多学习算法将参数表示为结构化的数学对象，例如向量、矩阵或张量。在每个逻辑时间（或一次迭代），通常只更新对象的一部分。也就是说，worker 通常发送一个向量段或矩阵的整行。这提供了机会来自动批处理更新的通信和在参数服务器上的处理，并允许高效实现一致性跟踪。

**容错能力**，如前所述，在大规模场景下至关重要，并且为了高效运行，它不能要求长时间运行的计算完全重启。服务器之间的实时参数复制支持热故障切换。故障切换和自我修复反过来支持动态扩展，将机器移除或添加分别视为故障或修复。

图1提供了多个系统上执行的最大规模有监督和无监督机器学习实验的规模概览。在可能的情况下，我们与每个系统的作者确认了扩展限制（数据截至2014年4月）。显然，我们能够比任何其他已发表的系统在数量级更多的处理器上处理数量级更多的数据。此外，表2提供了几个机器学习系统的主要特性概览。我们的参数服务器在一致性方面提供了最大的灵活度。它是唯一提供持续容错的系统。其原生数据类型使其对数据分析特别友好。

### 1.3 相关工作

相关系统已在 Amazon、Baidu、Facebook、Google [13]、Microsoft 和 Yahoo [1] 实现。也存在开源代码，如 YahooLDA [1] 和 Petuum [24]。此外，Graphlab [34] 在尽力而为模型上支持参数同步。

第一代参数服务器，由 [43] 引入，缺乏灵活性和性能——它重新利用 memcached 分布式（key,value）存储作为同步机制。YahooLDA 通过实现一个具有用户可定义更新原语（set、get、update）和更原理化的负载分布算法的专用服务器来改进该设计 [1]。这种第二代特定于应用的参数服务器也可以在 Distbelief [13] 和 [33] 的同步机制中找到。Petuum [24] 迈出了走向通用平台的第一步。它通过一个有界延迟模型改进了 YahooLDA，同时对 worker 线程模型施加了进一步约束。我们描述了一个克服这些限制的第三代系统。

最后，将参数服务器与更通用的分布式机器学习系统进行比较是有用的。其中一些系统强制使用同步、迭代通信。它们在数十个节点上扩展良好，但在大规模场景下，这种同步性会带来挑战，因为节点运行缓慢的概率增加。Mahout [4]（基于 Hadoop [18]）和 MLI [44]（基于 Spark [50]）都采用了迭代 MapReduce [14] 框架。Spark 和 MLI 的一个关键洞察是在迭代之间保留状态，这也是参数服务器的核心目标。

相反，分布式 GraphLab [34] 使用图抽象异步调度通信。目前，GraphLab 缺乏基于 map/reduce 的框架的弹性可扩展性，并且它依赖粗粒度的快照进行恢复，这两者都阻碍了可扩展性。由于缺乏作为高效一等原语的全局变量同步，其对某些算法的适用性受到限制。从某种意义上说，参数服务器框架的一个核心目标是捕捉 GraphLab 异步性的优点，同时避免其结构性限制。

Piccolo [39] 使用与参数服务器相关的策略在机器之间共享和聚合状态。在其中，worker 在本地预聚合状态并将更新发送到维护聚合状态的服务器。因此它基本上实现了我们系统功能的一个子集，缺乏机器学习专业化的优化：消息压缩、复制和通过依赖图表达的变量一致性模型。

## 2 机器学习

机器学习系统广泛应用于网络搜索、垃圾邮件检测、推荐系统、计算广告和文档分析。这些系统从示例（称为训练数据）中自动学习模型，通常由三个组件组成：特征提取、目标函数和学习。

特征提取处理原始训练数据（如文档、图像和用户查询日志）以获得特征向量，其中每个特征捕获训练数据的一个属性。预处理可以由现有框架（如 MapReduce）高效执行，因此不在本文讨论范围之内。

### 2.1 目标

许多机器学习算法的目标可以通过一个"目标函数"来表达。该函数捕捉学习模型的属性，例如在将电子邮件分类为正常邮件和垃圾邮件时的低错误率、在估计文档主题时数据被解释得有多好、或是在草图数据时计数的简洁摘要。

学习算法通常最小化这个目标函数以获得模型。一般来说，没有闭式解；相反，学习从一个初始模型开始。它通过处理训练数据（可能多次）迭代地改进该模型，以逼近解。当找到（近似）最优解或模型被认为已收敛时停止。

训练数据可能非常庞大。例如，一个大型互联网公司使用一年的广告展示日志 [27] 来训练广告点击预测器，将有数万亿个训练样本。每个训练样本通常表示为一个可能非常高维的"特征向量"[9]。因此，训练数据可能由数万亿个万亿长度的特征向量组成。迭代处理如此大规模的数据需要巨大的计算和带宽资源。此外，每天可能有数十亿的新广告展示到达。将这些数据加入系统通常可以提高预测准确性和覆盖率。但它也要求学习算法每日运行 [35]，甚至可能是实时运行。高效执行这些算法是本文的主要焦点。

为了引出我们系统中的设计决策，接下来我们简要概述两种广泛使用的机器学习技术，我们将用它们来展示参数服务器的有效性。更详细的概述可参考 [36, 28, 42, 22, 6]。

### 2.2 风险最小化

最直观的机器学习问题是风险最小化。"风险"大致是预测误差的度量。例如，如果我们预测明天的股票价格，风险可能是预测值与股票实际值之间的偏差。

训练数据由 $n$ 个样本组成。 $x_i$ 是第 $i$ 个这样的样本，通常是一个长度为 $d$ 的向量。如前所述， $n$ 和 $d$ 都可能达到数十亿到数万亿的数量级。在许多情况下，每个训练样本 $x_i$ 关联一个标签 $y_i$ 。例如，在广告点击预测中， $y_i$ 可能是1表示"点击"或-1表示"未点击"。

风险最小化学习一个模型，该模型可以预测未来样本 $x$ 的值 $y$ 。模型由参数 $w$ 组成。在最简单的示例中，模型参数可能是广告展示中每个特征的"点击倾向"。为了预测一个新的展示是否会被点击，系统可能简单地根据展示中存在的特征对其"点击倾向"求和，即 $x^\top w := \sum_{j=1}^d x_j w_j$ ，然后根据符号做决定。

在任何学习算法中，训练数据量和模型大小之间都有一个重要的关系。更详细的模型通常会提高准确性，但只能到一定程度：如果训练数据太少，过于详细的模型会过拟合，仅仅成为唯一地记忆训练集中每个 item 的系统。另一方面，太小的模型将无法捕捉到对做出正确决策重要的有趣且相关的数据属性。

正则化风险最小化 [48, 19] 是一种找到在模型复杂度和训练误差之间取得平衡的模型的方法。它通过最小化两项之和来实现：表示训练数据上预测误差的损失 $\ell(x, y, w)$ 和惩罚模型复杂度的正则化项 $\Omega[w]$ 。一个好的模型是低误差和低复杂度的模型。因此我们力求最小化

$$
F(w) = \sum_{i=1}^n \ell(x_i, y_i, w) + \Omega(w). \qquad (1)
$$

所使用的具体损失函数和正则化函数对于机器学习算法的预测性能很重要，但对于本文的目的来说相对不重要：我们提出的算法可以与所有最流行的损失函数和正则化器一起使用。

在第5.1节中，我们使用一个高性能分布式学习算法来评估参数服务器。为简单起见，我们描述一个更简单的模型 [46]，称为分布式次梯度下降（distributed subgradient descent）[footnote]^1。

如图2和算法1所示，训练数据被分区到所有 worker 之间，它们共同学习参数向量 $w$ 。该算法迭代运行。在每次迭代中，每个 worker 独立地使用自己的训练数据来确定应对 $w$ 进行何种更改，以使其更接近最优值。由于每个 worker 的更新仅反映其自己的训练数据，系统需要一种机制来允许这些更新混合。它通过将更新表示为次梯度（subgradient）——参数向量 $w$ 应该被移动的方向——并在将所有次梯度应用于 $w$ 之前将其聚合来实现。这些梯度通常按比例缩小，在算法设计中相当关注应使用的正确学习率 $\eta$ ，以确保算法快速收敛。

算法1中最昂贵的步骤是计算用于更新 $w$ 的次梯度。该任务被分配给所有 worker，每个 worker 执行 WORKERITERATE。作为其中的一部分，worker 计算 $w^\top x_{ik}$ ，这对于非常高维的 $w$ 可能是不可行的。幸运的是，当且仅当某个 worker 的一些训练数据引用了 $w$ 的某个坐标时，该 worker 才需要知道该坐标。例如，在广告点击预测中，关键特征之一是广告中的词语。如果只有非常少的广告包含短语"OSDI 2014"，那么大多数 worker 不会对 $w$ 中相应条目产生任何更新，因此不需要该条目。虽然 $w$ 的总大小可能超过单台机器的容量，但特定 worker 所需条目的工作集可以轻松缓存在本地。为了说明这一点，我们随机将数据分配给 worker，然后统计了第5.1节使用的数据集上每个 worker 的平均工作集大小。图3显示，对于100个 worker，每个 worker 仅需要总参数的7.8%。对于10,000个 worker，这减少到0.15%。

---

[footnote]^1 不熟悉的读者可以将其理解为梯度下降；次梯度方面只是对不需要连续可微的损失函数和正则化器（例如在 $w=0$ 处的 $|w|$ ）的一种推广。

---

### 2.3 生成模型

在第二大类机器学习算法中，要应用于训练样本的标签是未知的。这种情况下需要无监督算法（对于有标签的训练数据可以使用监督或半监督算法）。它们试图捕捉数据的底层结构。例如，该领域的一个常见问题是主题建模：给定一组文档，推断每个文档中包含的主题。

例如，当在 SOSP'13 会议论文集上运行时，算法可能生成诸如"分布式系统"、"机器学习"和"性能"之类的主题。算法从文档本身的内容推断这些主题，而不是从外部主题列表。在实际场景中，例如推荐系统 [2] 的内容个性化，这些问题的规模巨大：数亿用户和数十亿文档，这使得在大型集群上并行化这些算法变得至关重要。

由于其规模和数据量，这些算法只有在第一代参数服务器 [43] 引入之后才具有商业可行性。主题模型中的一个关键挑战是，描述文档应如何被生成的当前估计的参数必须被共享。

一种流行的主题建模方法是潜在狄利克雷分配（Latent Dirichlet Allocation, LDA）[7]。虽然统计模型相当不同，但用于学习它的结果算法与算法1非常相似 [footnote]^2。然而，关键区别在于更新步骤不是梯度计算，而是对文档能被当前模型解释得有多好的估计。该计算需要访问每个文档的辅助元数据，每次访问文档时都会更新该元数据。由于文档数量众多，元数据通常在文档被处理时从磁盘读取并写回磁盘。

这个辅助数据是分配给文档每个单词的主题集合，而正在学习的参数 $w$ 由单词出现的相对频率组成。和之前一样，每个 worker 只需要存储其处理的文档中出现的单词的参数。因此，将文档分布到各个 worker 与前一节有相同的效果：我们可以处理比单个 worker 所能容纳的大得多的模型。

---

[footnote]^2 我们在评估中使用的具体算法是随机变分采样器 [25] 的并行化变体，其更新策略与 YahooLDA [1] 中使用的类似。

---

## 3 架构

参数服务器的一个实例可以同时运行多个算法。参数服务器节点被分组为一个服务器组（server group）和几个 worker 组（worker groups），如图4所示。服务器组中的服务器节点维护全局共享参数的一个分区。服务器节点之间相互通信以复制和/或迁移参数，以实现可靠性和扩展。服务器管理器（server manager）节点维护服务器元数据的一致性视图，例如节点存活性和参数分区的分配。

每个 worker 组运行一个应用程序。worker 通常在本地存储一部分训练数据，以计算局部统计量如梯度。worker 仅与服务器节点通信（不在彼此之间通信），更新和检索共享参数。每个 worker 组有一个调度器（scheduler）节点。它为 worker 分配任务并监控其进度。如果添加或移除 worker，它会重新调度未完成的任务。

参数服务器支持独立的参数命名空间。这允许一个 worker 组将其共享参数集与其他组隔离。多个 worker 组也可以共享同一个命名空间：我们可以使用多个 worker 组来解决相同的深度学习应用 [13] 以增加并行化。另一个例子是由某些节点主动查询的模型，例如使用该模型的在线服务；同时，模型由一组不同的 worker 节点在新的训练数据到达时进行更新。

参数服务器旨在简化分布式机器学习应用的开发，如第2节讨论的那些。共享参数表示为（key,value）向量，以方便线性代数操作（第3.1节）。它们分布在一组服务器节点上（第4.3节）。任何节点都可以将其本地参数推送（push）出去以及从远程节点拉取（pull）参数（第3.2节）。默认情况下，工作负载或任务由 worker 节点执行；然而，它们也可以通过用户定义函数（第3.3节）分配给服务器节点。任务是异步且并行运行的（第3.4节）。参数服务器通过任务依赖图（第3.5节）和谓词（第3.6节）为算法设计者提供了选择一致性模型的灵活性。

### 3.1 （Key, Value）向量

节点间共享的模型可以表示为一组（key, value）对。例如，在损失最小化问题中，该对是特征 ID 及其权重。对于 LDA，该对是单词 ID 和主题 ID 的组合以及一个计数。模型的每个条目可以通过其键在本地或远程读取和写入。这种（key,value）抽象已被现有方法广泛采用 [37, 29, 12]。

我们的参数服务器通过认识这些键值 item 的底层含义改进了这种基本方法：机器学习算法通常将模型视为线性代数对象。例如， $w$ 在目标函数（1）和算法1的风险最小化优化中都作为向量使用。通过将这些对象视为稀疏线性代数对象，参数服务器可以提供与（key,value）抽象相同的功能，但允许重要的优化操作，如向量加法 $w + u$ 、乘法 $Xw$ 、求2-范数 $\|w\|_2$ ，以及其他更复杂的操作 [16]。

为了支持这些优化，我们假设键是有序的。这让我们可以将参数视为（key,value）对，同时赋予它们向量和矩阵语义，其中不存在的键关联为零。这有助于机器学习中的线性代数。它减少了实现优化算法的编程工作量。除了便利性之外，这种接口设计通过利用 CPU 高效的多线程自调优线性代数库（如 BLAS [16]、LAPACK [3] 和 ATLAS [49]）来产生高效的代码。

### 3.2 范围 Push 和 Pull

节点之间使用 push 和 pull 操作发送数据。在算法1中，每个 worker 将其整个局部梯度推送到服务器，然后拉取更新后的权重。算法3中描述的更高级算法使用相同的模式，只是每次只通信一个范围的键。

参数服务器通过支持基于范围的 push 和 pull 来优化这些更新，以提高编程便捷性以及计算和网络带宽效率。如果 $R$ 是一个键范围，那么 `w.push(R, dest)` 将 $w$ 在键范围 $R$ 内的所有现有条目发送到目标，目标可以是一个特定的节点，或者一个节点组如服务器组。类似地，`w.pull(R, dest)` 从目标读取 $w$ 在键范围 $R$ 内的所有现有条目。如果我们将 $R$ 设置为整个键范围，则整个向量 $w$ 将被通信。如果我们将 $R$ 设置为包含单个键，则只会发送单个条目。

该接口可以扩展为通信与 $w$ 共享相同键的任何本地数据结构。例如，在算法1中，worker 将其临时局部梯度 $g$ 推送到参数服务器进行聚合。一种选择是使 $g$ 成为全局共享的。然而，请注意 $g$ 共享 worker 工作集 $w$ 的键。因此程序员可以使用 `w.push(R, g, dest)` 来处理局部梯度，以节省内存并同时享受后续章节中讨论的优化。

### 3.3 服务器上的用户定义函数

除了聚合来自 worker 的数据，服务器节点还可以执行用户定义函数。这是有益的，因为服务器节点通常拥有关于共享参数的更完整或更及时的信息。在算法1中，服务器节点评估正则化器 $\Omega$ 的次梯度以更新 $w$ 。同时在算法3中，服务器求解一个更复杂的近端算子（proximal operator）来更新模型。在草图（sketching）的上下文中（第5.3节），几乎所有操作都发生在服务器端。

### 3.4 异步任务和依赖

任务通过远程过程调用（RPC）发出。它可以是由 worker 向服务器发出的 push 或 pull。它也可以是由调度器向任何节点发出的用户定义函数。任务可以包含任意数量的子任务。例如，算法1中的任务 WorkerIterate 包含一个 push 和一个 pull。

任务是异步执行的：调用者在发出任务后可以立即执行进一步的计算。调用者仅当接收到被调用者的回复时才将任务标记为已完成。回复可以是用户定义函数的返回值、由 pull 请求的（key,value）对、或一个空的确认。被调用者仅当任务的调用已返回且该调用发出的所有子任务都已完成时才将任务标记为已完成。

默认情况下，被调用者并行执行任务以获得最佳性能。希望串行化任务执行的调用者可以在任务之间放置一个"完成后执行"（execute-after-finished）的依赖。图5描绘了 WorkerIterate 的三个示例迭代。迭代10和11是独立的，但迭代12依赖于11。因此被调用者在迭代10中计算完局部梯度后立即开始迭代11。然而，迭代12被推迟到迭代11的 pull 完成后才执行。

任务依赖有助于实现算法逻辑。例如，算法1的 ServerIterate 中的聚合逻辑仅在所有 worker 的梯度都被聚合后才更新权重 $w$ 。这可以通过让更新任务依赖于所有 worker 的 push 任务来实现。依赖的第二个重要用途是支持接下来描述的灵活一致性模型。

### 3.5 灵活的一致性

独立的任务通过并行化 CPU、磁盘和网络带宽的使用来提高系统效率。然而，这可能导致节点之间的数据不一致。在上面的图示中，worker $r$ 在 $w^{(11)}$ 被拉回之前就开始了迭代11，因此它在这次迭代中使用了旧的 $w_r^{(10)}$ ，从而获得与迭代10相同的梯度，即 $g_r^{(11)} = g_r^{(10)}$ 。这种不一致可能会减慢算法1的收敛进度。然而，一些算法可能对这类不一致不那么敏感。例如，在算法3中，每次只更新 $w$ 的一个段。因此，在不等待迭代10的情况下开始迭代11只会导致 $w$ 的一部分不一致。

系统效率和算法收敛速度之间的最佳权衡通常取决于多种因素，包括算法对数据不一致的敏感性、训练数据中的特征相关性以及硬件组件的能力差异。参数服务器不是强迫用户采用某种可能不适合问题的特定依赖，而是让算法设计者可以灵活地定义一致性模型。这是与其他机器学习系统的一个本质区别。

我们展示了可以通过任务依赖实现的三种不同模型。它们关联的有向无环图（DAG）如图6所示。

**顺序（Sequential）：** 在顺序一致性中，所有任务一个接一个地执行。下一个任务只能在当前任务完成后才能开始。它产生与单线程实现相同的结果，也称为批量同步处理（Bulk Synchronous Processing）。

**最终（Eventual）：** 最终一致性是相反的情况：所有任务可以同时开始。例如，[43] 描述了这样的系统。然而，只有当下层算法对延迟足够健壮时，这才值得推荐。

**有界延迟（Bounded Delay）：** 当设置了最大延迟时间 $\tau$ 时，新任务将被阻塞，直到 $\tau$ 时间之前的所有前期任务都已完成。算法3使用了这种模型。该模型提供了比前两种更灵活的控制： $\tau = 0$ 是顺序一致性模型，而无限延迟 $\tau = \infty$ 则变为最终一致性模型。

注意依赖图可以是动态的。例如，调度器可以根据运行时进度增加或减少最大延迟，以平衡系统效率和下层优化算法的收敛速度。在这种情况下，调用者遍历 DAG。如果图是静态的，调用者可以将所有任务及其 DAG 一起发送给被调用者，以减少同步开销。

### 3.6 用户定义过滤器

作为基于调度器的流量控制的补充，参数服务器支持用户定义过滤器来选择性地同步单个（key,value）对，允许在任务内实现细粒度的数据一致性控制。其洞察在于，优化算法本身通常拥有关于哪些参数对同步最有用的信息。一个例子是"显著修改"过滤器，它只推送自上次同步以来变化超过阈值的条目。在第5.1节中，我们讨论了另一个名为 KKT 的过滤器，它利用优化问题的最优性条件：worker 只推送可能影响服务器上权重的梯度。

---

**算法2** 为范围 $R$ 和节点 $i$ 设置向量时钟为 $t$

1: **for** $S \in \{S_i : S_i \cap R \neq \emptyset, i = 1, \dots, n\}$ **do**
2:     **if** $S \subseteq R$ **then** $vc_i(S) \leftarrow t$ **else**
3:         $a \leftarrow \max(S_b, R_b)$ and $b \leftarrow \min(S_e, R_e)$
4:         split range $S$ into $[S_b, a), [a, b), [b, S_e)$
5:         $vc_i([a, b)) \leftarrow t$
6:     **end if**
7: **end for**

---

## 4 实现

服务器使用一致性哈希（consistent hashing）[45] 存储参数（key-value 对）（第4.3节）。为了容错，使用链式复制（chain replication）[47] 复制条目（第4.4节）。与先前的（key,value）系统不同，参数服务器针对基于范围的通信进行了优化，对数据（第4.2节）和基于范围的向量时钟（第4.1节）都进行了压缩。

### 4.1 向量时钟

鉴于潜在复杂的任务依赖图和快速恢复的需要，每个（key,value）对关联一个向量时钟 [30, 15]，该时钟记录每个单独节点在此（key,value）对上的时间。向量时钟很方便，例如用于跟踪聚合状态或拒绝重复发送的数据。然而，朴素实现的向量时钟需要 $O(nm)$ 空间来处理 $n$ 个节点和 $m$ 个参数。对于数千个节点和数十亿个参数，这在内存和带宽方面是不可行的。

幸运的是，由于参数服务器的基于范围的通信模式，许多参数共享相同的时间戳：如果一个节点推送一个范围内的参数，那么与该节点关联的参数的时间戳很可能相同。因此，它们可以被压缩为单个范围向量时钟。更具体地说，假设 $vc_i(k)$ 是节点 $i$ 的键 $k$ 的时间。给定一个键范围 $R$ ，范围向量时钟 $vc_i(R) = t$ 意味着对于任何键 $k \in R$ ， $vc_i(k) = t$ 。

初始时，每个节点 $i$ 只有一个范围向量时钟。它覆盖整个参数键空间，初始时间戳为0。每个范围集合可能会分割范围并创建最多3个新的向量时钟（见算法2）。设 $k$ 为算法通信的唯一范围总数，那么最多有 $O(mk)$ 个向量时钟，其中 $m$ 是节点数。 $k$ 通常远小于参数总数。这大大减少了范围向量时钟所需的空间。[footnote]^3

[footnote]^3 范围也可以合并以减少片段数量。然而在实践中， $m$ 和 $k$ 都足够小，可以轻松处理。我们将合并留作未来工作。

### 4.2 消息

节点可以向单个节点或节点组发送消息。一条消息由键范围 $R$ 内的（key,value）对列表和关联的范围向量时钟组成：

$$
[vc(R), (k_1, v_1), \dots, (k_p, v_p)] \quad k_j \in R, j \in \{1, \dots, p\}
$$

这是参数服务器的基本通信格式，不仅用于共享参数，也用于任务。对于后者，（key,value）对可能采用（任务 ID，参数或返回结果）的形式。

消息可以携带范围 $R$ 内所有可用键的一个子集。缺失的键被分配相同的时间戳而不改变其值。消息可以根据键范围进行拆分。当 worker 向整个服务器组发送消息时，或当接收节点的键分配发生变化时，就会发生这种情况。通过这样做，我们划分（key,value）列表并类似于算法2地拆分范围向量时钟。

由于机器学习问题通常需要高带宽，消息压缩是可取的。训练数据在迭代之间通常保持不变。worker 可能会再次发送相同的键列表。因此接收节点缓存键列表是值得的。之后，发送方只需发送列表的哈希值而不是列表本身。反过来，值可能包含许多零条目。例如，在稀疏逻辑回归中，很大一部分参数保持不变，如第5.1节所评估的。同样，用户定义过滤器也可能将很大一部分值置零（见图12）。因此我们只需要发送非零的（key,value）对。我们使用快速的 Snappy 压缩库 [21] 来压缩消息，有效地移除零。注意键缓存和值压缩可以联合使用。

### 4.3 一致性哈希

参数服务器对键进行分区的方式与传统分布式哈希表 [8, 41] 非常相似：键和服务器节点 ID 都被插入到哈希环中（图7）。每个服务器节点管理从其插入点到逆时针方向下一个其他节点插入点之间的键范围。该节点被称为该键范围的主节点（master）。一个物理服务器通常在环中通过多个"虚拟"服务器来表示，以改善负载均衡和恢复。

我们通过使用直接映射的 DHT 设计简化管理。服务器管理器处理环的管理。所有其他节点在本地缓存键分区。这样它们可以直接确定哪个服务器负责一个键范围，并在任何变化时得到通知。

### 4.4 复制和一致性

每个服务器节点存储相对于其自身所拥有的范围逆时针方向的 $k$ 个相邻键范围的副本。我们将持有副本的节点称为相应键范围的从节点（slave）。上面的图示展示了一个 $k = 2$ 的示例，其中服务器1复制了服务器2和服务器3拥有的键范围。

Worker 节点在 push 和 pull 时都与键范围的主节点通信。主节点上的任何修改都连同其时间戳一起复制到从节点。数据的修改被同步推送到从节点。图8展示了一个情况，其中 worker 1 将 $x$ 推送到服务器1，服务器1调用用户定义函数 $f$ 来修改共享数据。仅当数据修改 $f(x)$ 被复制到从节点后，push 任务才算完成。

朴素的复制可能会将网络流量增加 $k$ 倍。这对于许多依赖高网络带宽的机器学习应用来说是不希望的。参数服务器框架为许多算法允许一项重要的优化：聚合后复制。服务器节点通常聚合来自 worker 节点的数据，例如对局部梯度求和。因此服务器可以将复制推迟到聚合完成之后。在图中的右侧，两个 worker 分别推送 $x$ 和 $y$ 到服务器。服务器首先通过 $x+y$ 聚合推送，然后应用修改 $f(x+y)$ ，最后执行复制。对于 $n$ 个 worker，复制仅使用 $k/n$ 的带宽。通常 $k$ 是小的常数，而 $n$ 是数百到数千。虽然聚合增加了任务回复的延迟，但这可以通过宽松的一致性条件来隐藏。

### 4.5 服务器管理

为了实现容错和动态扩展，我们必须支持节点的添加和移除。为方便起见，下文我们指的是虚拟服务器。当服务器加入时，发生以下步骤：

1. 服务器管理器为新节点分配一个键范围作为主节点。这可能导致另一个键范围被分割或从已终止节点移除。
2. 该节点获取要维护为主节点的数据范围以及另外 $k$ 个要作为从节点保存的范围。
3. 服务器管理器广播节点变更消息。消息的接收者可以根据它们不再持有的键范围缩小自己的数据，并将未完成的任务重新提交给新节点。

从某个节点 $S$ 获取范围 $R$ 中的数据分两个阶段进行，类似于 Ouroboros 协议 [38]。首先， $S$ 预拷贝该范围内的所有（key,value）对及其关联的向量时钟。这可能导致范围向量时钟类似于算法2地分裂。如果新节点在此阶段失败， $S$ 保持不变。在第二阶段， $S$ 不再接受影响键范围 $R$ 的消息，而是丢弃这些消息而不执行或回复。同时， $S$ 将预拷贝阶段中 $R$ 内发生的所有变更发送给新节点。

收到节点变更消息时，节点 $N$ 首先检查它是否也维护键范围 $R$ 。如果是且此键范围不再由 $N$ 维护，则删除 $R$ 中所有关联的（key,value）对和向量时钟。接下来， $N$ 扫描所有尚未收到回复的传出消息。如果某个键范围与 $R$ 相交，则消息将被拆分并重新发送。

由于延迟、故障和丢失的确认， $N$ 可能两次发送消息。由于使用了向量时钟，原始接收者和新节点都能够拒绝该消息，因此不会影响正确性。

服务器节点的离开（自愿或由于故障）类似于加入。服务器管理器安排新节点接管离开节点的键范围。服务器管理器通过心跳信号检测节点故障。与集群资源管理器（如 Yarn [17] 或 Mesos [23]）的集成留作未来工作。

### 4.6 Worker 管理

添加新的 worker 节点 $W$ 与添加新的服务器节点类似但更简单：

1. 任务调度器为 $W$ 分配一个数据范围。
2. 该节点从网络文件系统或现有 worker 加载训练数据范围。训练数据通常是只读的，因此没有两阶段获取。接下来， $W$ 从服务器拉取共享参数。
3. 任务调度器广播此变更，可能使其他 worker 释放一些训练数据。

当 worker 离开时，任务调度器可能启动一个替代者。我们为算法设计者提供控制恢复的选项，原因有两个：如果训练数据量巨大，恢复 worker 节点可能比恢复服务器节点更昂贵。第二，在优化过程中丢失少量训练数据通常对模型影响很小。因此算法设计者可能更倾向于在不替换失败 worker 的情况下继续。甚至可能需要终止最慢的 worker。

## 5 评估

我们基于第2节的应用场景——稀疏逻辑回归和潜在狄利克雷分配（Latent Dirichlet Allocation）——来评估我们的参数服务器。我们还展示了草图的结果，以说明我们框架的通用性。实验在两个（不同的）大型互联网公司和一所大学研究集群上运行，以证明我们方法的多功能性。

### 5.1 稀疏逻辑回归

**问题与数据：** 稀疏逻辑回归是大规模风险最小化最流行的算法之一 [9]。它将逻辑丢失（logistic loss）[footnote]^4 与第2.2节的 $\ell_1$ 正则化器 [footnote]^5 相结合。后者偏向于具有大量零值条目的紧凑解。然而，该正则化器的非光滑性使得学习更加困难。

我们收集了一个包含1700亿个样本和650亿个唯一特征的广告点击预测数据集。该数据集未压缩为636 TB（压缩后为141 TB）。我们在1000台机器上运行参数服务器，每台机器有16个物理核心、192GB DRAM，并通过10 Gb 以太网连接。800台机器作为 worker，200台作为参数服务器。该集群在运行期间同时被其他（无关）任务使用。

**算法：** 我们使用了最先进的分布式回归算法（算法3，[31, 32]）。它与前面描述的更简单变体有四个不同之处：第一，每次迭代只更新一个参数块。第二，worker 在该块上同时计算梯度和二阶导数的对角部分。第三，参数服务器本身必须执行复杂的计算：服务器通过基于聚合的局部梯度求解近端算子来更新模型。第四，我们在迭代上使用有界延迟模型，并使用"KKT"过滤器来抑制传输生成的梯度更新中足够小的部分，这些部分的效果可能可以忽略不计。[footnote]^6

据我们所知，没有开源系统可以将稀疏逻辑回归扩展到本文描述的规模。[footnote]^7 我们将参数服务器与两个专用系统（称为系统 A 和 B）进行比较，它们由一家大型互联网公司开发。值得注意的是，系统 A 和 B 都由超过10K行代码组成。参数服务器仅需300行代码即可实现与系统 B 相同的功能。[footnote]^8 参数服务器成功地将大部分系统复杂性从算法实现转移到一个可重用的通用组件中。

**结果：** 我们首先通过运行这三个系统达到相同的目标值来比较它们。更好的系统在更短的时间内实现更低的目标。图9显示了结果：系统 B 优于系统 A，因为它使用了更好的算法。而参数服务器在使用相同算法的情况下优于系统 B。这是因为减少了网络流量和宽松的一致性模型的有效性。

图10显示，宽松的一致性模型显著提高了 worker 节点的利用率。Worker 可以开始处理下一个块而无需等待前一个块完成，从而隐藏了原本由屏障同步带来的延迟。系统 A 中的 worker 在等待每个块中的屏障时有32%的空闲，系统 B 中有53%的空闲。参数服务器将此开销降低到2%以下。这并非完全没有代价：参数服务器比系统 B 使用稍多的 CPU，原因有二。第一（不那么根本），系统 B 通过仔细的数据预处理优化其梯度计算。第二，使用参数服务器的异步更新需要更多迭代才能达到相同的目标值。由于通信成本显著降低，参数服务器将总时间减半。

接下来，我们评估各系统组件对网络流量的减少。图11显示了服务器和 worker 的结果。可以看出，允许发送方和接收方缓存键可以节省近50%的流量。这是因为键（int64）和值（double）大小相同，且键集在优化过程中不变。此外，数据压缩在压缩服务器端的值（>20倍）和应用 KKT 过滤器的 worker 端的值（>6倍）方面是有效的。原因有两个。首先， $\ell_1$ 正则化器鼓励稀疏模型（ $w$ ），因此从服务器拉取的大部分值为0。其次，KKT 过滤器迫使发送到服务器的大部分梯度为0。这在图12中更清楚地可见，该图显示超过93%的唯一特征被 KKT 过滤器过滤掉。

最后，我们分析有界延迟一致性模型。在不同最大允许延迟（ $\tau$ ）下 worker 达到相同收敛标准的时间分解如图13所示。正如预期，等待时间随着允许延迟的增加而减少。使用顺序一致性模型（ $\tau = 0$ ）时，worker 有50%的空闲时间，而当 $\tau$ 设为16时，空闲率降低到1.7%。然而，计算时间几乎随 $\tau$ 线性增加。因为数据不一致减慢了收敛速度，需要更多迭代才能达到相同的收敛标准。结果是， $\tau = 8$ 是算法收敛和系统性能之间的最佳权衡。

---

[footnote]^4 $\ell(x_i, y_i, w) = \log(1 + \exp(-y_i \langle x_i, w \rangle))$

[footnote]^5 $\Omega(w) = \sum_{i=1}^n |w_i|$

[footnote]^6 用户定义的 Karush-Kuhn-Tucker (KKT) 过滤器 [26]。如果 $w_k = 0$ 且 $|\hat{g}_k| \leq \Delta$ ，则特征 $k$ 被过滤。这里 $\hat{g}_k$ 是基于 worker 本地信息的全局梯度估计， $\Delta > 0$ 是用户定义的参数。

[footnote]^7 Graphlab 仅提供多线程的单机实现，而 Petuum、Mlbase 和 REEF 不支持稀疏逻辑回归。我们已于2014年4月与作者确认了这一点。

[footnote]^8 系统 B 由本文的一位作者开发。

---

### 5.2 潜在狄利克雷分配（Latent Dirichlet Allocation）

**问题与数据：** 为了展示我们方法的多功能性，我们将相同的参数服务器架构应用于根据用户在搜索结果中点击的 URL 中出现的域来建模用户兴趣的问题。我们收集了包含50亿个唯一用户标识符的搜索日志数据，并评估了结果集中500万个最频繁点击的域的模型。我们分别使用800个 worker 和200个服务器以及5000个 worker 和1000个服务器运行算法。机器有10个物理核心、128GB DRAM，以及至少10 Gb/s的网络连接。我们再次与同时运行的生产作业共享集群。

**算法：** 我们使用随机变分方法（Stochastic Variational Methods）[25]、折叠吉布斯采样（Collapsed Gibbs sampling）[20] 和分布式梯度下降的组合来执行 LDA。在这里，梯度在从 worker 到达时被异步聚合，类似于 [1] 的方法。

我们将模型中的参数分为局部参数和全局参数。局部参数（即辅助元数据）与特定用户相关，当访问该用户时从磁盘流式读取。全局参数在用户之间共享，表示为（key,value）对，存储在参数服务器中。用户数据在 worker 之间分片。每个 worker 运行一组计算线程来对其分配的用户执行推理。我们异步同步以向服务器发送和接收局部更新，并接收全局参数的新值。

据我们所知，没有其他系统（例如 YahooLDA、Graphlab 或 Petuum）能够在 LDA 中处理如此规模的数据和模型复杂度，使用多达100亿个（500万个 token 和2000个主题）共享参数。先前报道的最大实验 [2] 在任何时刻活跃用户不超过1亿，token 数少于10万，主题数少于1000（数据的2%，参数的1%）。

**结果：** 为了评估推理算法的质量，我们监控训练对数似然（衡量拟合优度）收敛的速度。如图14所示，我们观察到当机器数量从1000增加到6000时，收敛速度大约提升了4倍。在图14（最左图）中观察到的落后者（stragglers）也说明了拥有能够应对 worker 间性能差异的架构的重要性。

---

**表4：** 使用 LDA 在50亿数据集上学习到的示例主题。每个主题代表一个用户兴趣。

| 主题名称 | 顶部 URL |
|---|---|
| 编程 | stackoverflow.com, w3schools.com, cplusplus.com, github.com, tutorialspoint.com, jquery.com, codeproject.com, oracle.com, qt-project.org, bytes.com, android.com, mysql.com |
| 音乐 | ultimate-guitar.com, guitaretab.com, 911tabs.com, e-chords.com, songsterr.com, chordify.net, musicnotes.com, ukulele-tabs.com |
| 婴儿相关 | babycenter.com, whattoexpect.com, babycentre.co.uk, circleofmoms.com, thebump.com, parents.com, momtastic.com, parenting.com, americanpregnancy.org, kidshealth.org |
| 力量训练 | bodybuilding.com, muscleandfitness.com, mensfitness.com, menshealth.com, t-nation.com, livestrong.com, muscleandstrength.com, myfitnesspal.com, elitefitness.com, crossfit.com, steroid.com, gnc.com, askmen.com |

---

### 5.3 草图（Sketches）

**问题与数据：** 我们将草图作为评估的一部分来测试通用性，因为它们的运作方式与机器学习算法非常不同。它们通常观察来自流式数据源的大量事件写入 [11, 5]。

我们评估了将页面浏览量的流式日志插入到一个近似结构中所需的时间，该结构可以高效地跟踪大量网页的页面浏览计数。我们使用维基百科（以及其它 Wiki 项目）页面浏览统计作为基准。每个条目是一个网页的唯一键，附带一小时内服务的请求数。从2007年12月到2014年1月，共有3000亿条记录，超过1亿个唯一键。我们在一个研究集群 [40] 的15台机器上运行参数服务器，配置了90个虚拟服务器节点（每台机器64核，通过40 Gb 以太网连接）。

**算法：** 草图算法高效地存储海量数据的摘要，以便能够快速回答近似查询。这些算法在流式应用中尤其重要，其中数据和查询实时到达。一些最高吞吐量的应用涉及诸如 Cloudflare 的 DDoS 防御服务之类的例子，该服务必须分析其整个内容交付服务架构中的页面请求，以识别可能的 DDoS 目标和攻击者。

此类应用中记录的数据量远超单台机器的容量。虽然传统方法可能是将工作负载分片到键值集群（如 Redis）上，但这些系统通常不允许用户定义的聚合语义来实现近似聚合。算法4给出了 CountMin sketch [11] 的简要概述。根据设计，查询结果是观察到的键 $x$ 数量的上界。将键划分到范围中自动允许我们并行化草图。与前两个应用不同，worker 简单地将更新分发到适当的服务器。

---

**算法4** CountMin Sketch

初始化： $M[i, j] = 0$ 对于 $i \in \{1, \dots, n\}$ 且 $j \in \{1, \dots, k\}$

Insert( $x$ ):
1: **for** $i = 1$ **to** $k$ **do**
2:     $M[i, hash(i, x)] \leftarrow M[i, hash(i, x)] + 1$

Query( $x$ ):
1: **return** $\min \{M[i, hash(i, x)] \text{ for } 1 \leq i \leq k\}$

---

**结果：** 该系统实现了非常高的插入速率，如表5所示。其性能良好的原因有两个：首先，批量通信降低了通信成本。其次，消息压缩将平均（key,value）大小减少到约50位。重要的是，当我们在插入过程中终止一个服务器节点时，参数服务器能够在1秒内恢复故障节点，这使得我们的系统非常适合实时场景。

---

**表5：** 分布式 CountMin 的结果

| 指标 | 数值 |
|---|---|
| 峰值每秒插入数 | 13亿 |
| 平均每秒插入数 | 11亿 |
| 每台机器峰值净带宽 | 4.37 GBit/s |
| 恢复故障节点的时间 | 0.8秒 |

---

## 6 总结与讨论

我们描述了一个用于解决分布式机器学习问题的参数服务器框架。该框架易于使用：全局共享参数可以作为局部稀疏向量或矩阵，与局部训练数据一起执行线性代数操作。它高效：所有通信都是异步的。支持灵活的一致性模型以平衡系统效率和算法快速收敛速率之间的权衡。此外，它提供弹性可扩展性和容错能力，旨在实现稳定的长期部署。最后，我们展示了在具有数十亿变量的真实数据集上针对几个具有挑战性任务的实验，以证明其效率。我们相信这个第三代参数服务器是可扩展机器学习的重要构建模块。

代码可在 parameterserver.org 获取。

---

## 致谢

本工作部分得到了 Google、Amazon、Baidu、PRObE 和 Microsoft 的捐赠和/或机器时间支持；NSF 奖励 1409802；以及 Intel 云计算科学与技术中心的支持。我们感谢审稿人和同事对本论文早期版本的宝贵意见。

---

## 参考文献

[1] A. Ahmed, M. Aly, J. Gonzalez, S. Narayanamurthy, and A. J. Smola. Scalable inference in latent variable models. In *Proceedings of The 5th ACM International Conference on Web Search and Data Mining (WSDM)*, 2012.

[2] A. Ahmed, Y. Low, M. Aly, V. Josifovski, and A. J. Smola. Scalable inference of dynamic user interests for behavioural targeting. In *Knowledge Discovery and Data Mining*, 2011.

[3] E. Anderson, Z. Bai, C. Bischof, J. Demmel, J. Dongarra, J. Du Croz, A. Greenbaum, S. Hammarling, A. McKenney, S. Ostrouchov, and D. Sorensen. *LAPACK Users' Guide*. SIAM, Philadelphia, second edition, 1995.

[4] Apache Foundation. Mahout project, 2012. http://mahout.apache.org.

[5] R. Berinde, G. Cormode, P. Indyk, and M.J. Strauss. Space-optimal heavy hitters with strong error bounds. In J. Paredaens and J. Su, editors, *Proceedings of the Twenty-Eigth ACM SIGMOD-SIGACT-SIGART Symposium on Principles of Database Systems, PODS*, pages 157–166. ACM, 2009.

[6] C. Bishop. *Pattern Recognition and Machine Learning*. Springer, 2006.

[7] D. Blei, A. Ng, and M. Jordan. Latent Dirichlet allocation. *Journal of Machine Learning Research*, 3:993–1022, January 2003.

[8] J. Byers, J. Considine, and M. Mitzenmacher. Simple load balancing for distributed hash tables. In *Peer-to-peer systems II*, pages 80–87. Springer, 2003.

[9] K. Canini. Sibyl: A system for large scale supervised machine learning. Technical Talk, 2012.

[10] B.-G. Chun, T. Condie, C. Curino, C. Douglas, S. Matus evych, B. Myers, S. Narayanamurthy, R. Ramakrishnan, S. Rao, J. Rosen, R. Sears, and M. Weimer. Reef: Retainable evaluator execution framework. *Proceedings of the VLDB Endowment*, 6(12):1370–1373, 2013.

[11] G. Cormode and S. Muthukrishnan. Summarizing and mining skewed data streams. In *SDM*, 2005.

[12] W. Dai, J. Wei, X. Zheng, J. K. Kim, S. Lee, J. Yin, Q. Ho, and E. P. Xing. Petuum: A framework for iterative-convergent distributed ml. *arXiv preprint arXiv:1312.7651*, 2013.

[13] J. Dean, G. Corrado, R. Monga, K. Chen, M. Devin, Q. Le, M. Mao, M. Ranzato, A. Senior, P. Tucker, K. Yang, and A. Ng. Large scale distributed deep networks. In *Neural Information Processing Systems*, 2012.

[14] J. Dean and S. Ghemawat. MapReduce: simplified data processing on large clusters. *CACM*, 51(1):107–113, 2008.

[15] G. DeCandia, D. Hastorun, M. Jampani, G. Kakulapati, A. Lakshman, A. Pilchin, S. Sivasubramanian, P. Vosshall, and W. Vogels. Dynamo: Amazon's highly available key-value store. In T. C. Bressoud and M. F. Kaashoek, editors, *Symposium on Operating Systems Principles*, pages 205–220. ACM, 2007.

[16] J. J. Dongarra, J. Du Croz, S. Hammarling, and R. J. Hanson. An extended set of fortran basic linear algebra subprograms. *ACM Transactions on Mathematical Software*, 14:18–32, 1988.

[17] The Apache Software Foundation. Apache hadoop nextgen mapreduce (yarn). http://hadoop.apache.org/.

[18] The Apache Software Foundation. Apache hadoop, 2009. http://hadoop.apache.org/core/.

[19] F. Girosi, M. Jones, and T. Poggio. Priors, stabilizers and basis functions: From regularization to radial, tensor and additive splines. A.I. Memo 1430, Artificial Intelligence Laboratory, Massachusetts Institute of Technology, 1993.

[20] T.L. Griffiths and M. Steyvers. Finding scientific topics. *Proceedings of the National Academy of Sciences*, 101:5228–5235, 2004.

[21] S. H. Gunderson. Snappy: A fast compressor/decompressor. https://code.google.com/p/snappy/.

[22] T. Hastie, R. Tibshirani, and J. Friedman. *The Elements of Statistical Learning*. Springer, New York, 2 edition, 2009.

[23] B. Hindman, A. Konwinski, M. Zaharia, A. Ghodsi, A. D. Joseph, R. Katz, S. Shenker, and I. Stoica. Mesos: A platform for fine-grained resource sharing in the data center. In *Proceedings of the 8th USENIX conference on Networked systems design and implementation*, pages 22–22, 2011.

[24] Q. Ho, J. Cipar, H. Cui, S. Lee, J. Kim, P. Gibbons, G. Gibson, G. Ganger, and E. Xing. More effective distributed ml via a stale synchronous parallel parameter server. In *NIPS*, 2013.

[25] M. Hoffman, D. M. Blei, C. Wang, and J. Paisley. Stochastic variational inference. In *International Conference on Machine Learning*, 2012.

[26] W. Karush. Minima of functions of several variables with inequalities as side constraints. Master's thesis, Dept. of Mathematics, Univ. of Chicago, 1939.

[27] L. Kim. How many ads does Google serve in a day?, 2012. http://goo.gl/oIidXO.

[28] D. Koller and N. Friedman. *Probabilistic Graphical Models: Principles and Techniques*. MIT Press, 2009.

[29] T. Kraska, A. Talwalkar, J. C. Duchi, R. Griffith, M. J. Franklin, and M. I. Jordan. Mlbase: A distributed machine-learning system. In *CIDR*, 2013.

[30] L. Lamport. Paxos made simple. *ACM Sigact News*, 32(4):18–25, 2001.

[31] M. Li, D. G. Andersen, and A. J. Smola. Distributed delayed proximal gradient methods. In *NIPS Workshop on Optimization for Machine Learning*, 2013.

[32] M. Li, D. G. Andersen, and A. J. Smola. Communication Efficient Distributed Machine Learning with the Parameter Server. In *Neural Information Processing Systems*, 2014.

[33] M. Li, L. Zhou, Z. Yang, A. Li, F. Xia, D.G. Andersen, and A. J. Smola. Parameter server for distributed machine learning. In *Big Learning NIPS Workshop*, 2013.

[34] Y. Low, J. Gonzalez, A. Kyrola, D. Bickson, C. Guestrin, and J. M. Hellerstein. Distributed Graphlab: A framework for machine learning and data mining in the cloud. In *PVLDB*, 2012.

[35] H. B. McMahan, G. Holt, D. Sculley, M. Young, D. Ebner, J. Grady, L. Nie, T. Phillips, E. Davydov, and D. Golovin. Ad click prediction: a view from the trenches. In *KDD*, 2013.

[36] K. P. Murphy. *Machine learning: a probabilistic perspective*. MIT Press, 2012.

[37] D. G. Murray, F. McSherry, R. Isaacs, M. Isard, P. Barham, and M. Abadi. Naiad: a timely dataflow system. In *Proceedings of the Twenty-Fourth ACM Symposium on Operating Systems Principles*, pages 439–455. ACM, 2013.

[38] A. Phanishayee, D. G. Andersen, H. Pucha, A. Povzner, and W. Belluomini. Flex-KV: Enabling high-performance and flexible KV systems. In *Proceedings of the 2012 workshop on Management of big data systems*, pages 19–24. ACM, 2012.

[39] R. Power and J. Li. Piccolo: Building fast, distributed programs with partitioned tables. In R. H. Arpaci-Dusseau and B. Chen, editors, *Operating Systems Design and Implementation, OSDI*, pages 293–306. USENIX Association, 2010.

[40] PRObE Project. Parallel Reconfigurable Observational Environment. https://www.nmc-probe.org/wiki/Machines:Susitna.

[41] A. Rowstron and P. Druschel. Pastry: Scalable, decentralized object location and routing for large-scale peer-to-peer systems. In *IFIP/ACM International Conference on Distributed Systems Platforms (Middleware)*, pages 329–350, Heidelberg, Germany, November 2001.

[42] B. Scholkopf and A. J. Smola. *Learning with Kernels*. MIT Press, Cambridge, MA, 2002.

[43] A. J. Smola and S. Narayanamurthy. An architecture for parallel topic models. In *Very Large Databases (VLDB)*, 2010.

[44] E. Sparks, A. Talwalkar, V. Smith, J. Kottalam, X. Pan, J. Gonzalez, M. J. Franklin, M. I. Jordan, and T. Kraska. Mli: An api for distributed machine learning. 2013.

[45] I. Stoica, R. Morris, D. Karger, M. F. Kaashoek, and H. Balakrishnan. Chord: A scalable peer-to-peer lookup service for internet applications. *ACM SIGCOMM Computer Communication Review*, 31(4):149–160, 2001.

[46] C.H. Teo, Q. Le, A. J. Smola, and S. V. N. Vishwanathan. A scalable modular convex solver for regularized risk minimization. In *Proc. ACM Conf. Knowledge Discovery and Data Mining (KDD)*. ACM, 2007.

[47] R. van Renesse and F. B. Schneider. Chain replication for supporting high throughput and availability. In *OSDI*, volume 4, pages 91–104, 2004.

[48] V. Vapnik. *The Nature of Statistical Learning Theory*. Springer, New York, 1995.

[49] R.C. Whaley, A. Petitet, and J.J. Dongarra. Automated empirical optimization of software and the ATLAS project. *Parallel Computing*, 27(1–2):3–35, 2001.

[50] M. Zaharia, M. Chowdhury, T. Das, A. Dave, J. M. Ma, M. McCauley, M. J. Franklin, S. Shenker, and I. Stoica. Fast and interactive analytics over Hadoop data with Spark. *USENIX ;login:*, 37(4):45–51, August 2012.

---

**图1：** 各系统公开的最大机器学习实验规模比较。问题按颜色编码如下：蓝色圆形——稀疏逻辑回归；红色方形——潜变量图模型；灰色五边形——深度网络。

**图2：** 执行分布式次梯度下降所需的步骤，如 [46] 中所述。每个 worker 只缓存 $w$ 的工作集而非所有参数。

**图3：** 随着更多 worker 的使用，每个 worker 的参数集缩小，每台机器所需的内存减少。

**图4：** 参数服务器与多个 worker 组通信的架构。

**图5：** 迭代12依赖于11，而10和11是独立的，因此允许异步处理。

**图6：** 不同一致性模型的有向无环图。DAG 的大小随延迟增加而增加。

**图7：** 服务器节点布局。

**图8：** 副本生成。左：单个 worker。右：多个 worker 同时更新值。

**图9：** 稀疏逻辑回归的收敛情况。目标是快速最小化目标值。

**图10：** 稀疏逻辑回归期间每个 worker 在计算和等待上的时间。

**图11：** 不同组件节省的传出网络流量。左：每台服务器。右：每个 worker。

**图12：** 优化过程中被 KKT 过滤器过滤的唯一特征（键）。

**图13：** 不同最大延迟下 worker 达到相同收敛标准所花费的时间。

**图14：** 左：1000台机器和50亿用户下 worker 对数似然随时间分布。部分低值是由于落后者初始同步缓慢。中：相同分布，按迭代次数分层。右：1000台和6000台机器在5亿用户上的收敛情况（时间以千秒计）。

**表1：** 数据中心中三个月期间的机器学习作业统计。

**表2：** 分布式数据分析系统的属性。

**表3：** 评估的系统。

**表4：** 使用 LDA 在50亿数据集上学习到的示例主题。每个主题代表一个用户兴趣。

**表5：** 分布式 CountMin 的结果。
