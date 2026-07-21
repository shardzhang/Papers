# 2012-大规模分布式深度网络

> Jeffrey Dean, Greg S. Corrado, Rajat Monga, Kai Chen, Matthieu Devin, Quoc V. Le, Mark Z. Mao, Marc'Aurelio Ranzato, Andrew Senior, Paul Tucker, Ke Yang, Andrew Y. Ng | Google Inc.



本文介绍了 DistBelief 软件框架 及其在 大规模分布式深度网络训练 中的两项核心算法。核心内容：

- DistBelief 框架，支持模型内（多线程）和跨机器（消息传递）的模型并行，自动管理并行、同步和通信
- Downpour SGD，一种**异步随机梯度下降算法**，支持大量模型副本，结合 Adagrad 自适应学习率效果显著
- Sandblaster L-BFGS，一种分布式批处理优化框架，支持 L-BFGS 等批量优化方法
- 使用数千个 CPU 核心训练具有数十亿参数的深度网络，在 ImageNet（1600 万图像、21k 类别）上取得当时最先进的结果

关键发现：

- **异步 SGD 在非凸问题上效果出乎意料地好，特别是配合 Adagrad 自适应学习率时**
- **给定足够资源，L-BFGS 可以与 SGD 竞争甚至更快**
- 可使用数十个机器（512 CPU 核心）训练单个大型神经网络，结合多副本可扩展到数万核心
- 将语音识别模型的训练加速到 GPU 的 10 倍以上
- 成功训练了 30 倍于此前文献报道规模的深度网络

---



## 摘要

最近无监督特征学习 和 深度学习方面的工作表明，训练大型模型可以显著提升性能。本文考虑使用数万个 CPU 核心训练具有**数十亿参数**的深度网络的问题。我们开发了一个名为 DistBelief 的软件框架，可以利用拥有 **数千台机器的计算集群** 来训练大型模型。在该框架中，我们开发了两种**大规模分布式训练算法**：(i) Downpour SGD，一种 异步随机梯度下降过程，**支持大量模型副本**，以及 (ii) Sandblaster，一个支持多种分布式 批处理优化过程的框架，包括 L-BFGS 的分布式实现。Downpour SGD 和 Sandblaster L-BFGS 都提升了深度网络训练的规模和速度。我们已成功使用该系统训练了比此前文献报道大 30 倍的深度网络，并在 ImageNet 数据集上取得了当时最先进的性能。我们还展示了这些技术显著加速了中等规模深度网络在商业语音识别服务中的训练。尽管我们关注并报告了这些方法在训练大型神经网络中的应用，但**底层算法适用于任何基于梯度的机器学习算法**。



## 1. 引言

深度学习 和 无监督特征学习在许多实际应用中展现出巨大的潜力。在语音识别 [1, 2]、视觉目标识别 [3, 4] 以及**文本处理 [5, 6]** 等多个领域都报告了最先进的性能。

研究还发现，**增加深度学习的规模——无论是训练样本数量、模型参数数量还是两者**——都可以显著提升最终的分类精度 [3, 4, 7]。这些结果引发了扩大这些模型训练和推理算法规模 [8] 以及改进适用优化过程 [7, 9] 的热潮。近年来 GPU 的使用 [1, 2, 3, 8] 是一项重大进步，使得中等规模深度网络的训练变得切实可行。**GPU 方法的一个已知限制是当模型无法放入 GPU 内存（通常小于 6 GB）时，训练加速很小**。**为了有效使用 GPU，研究人员通常会减少数据或参数的大小，以使 CPU 到 GPU 的传输不会成为显著瓶颈**。虽然数据和参数缩减对小问题（如语音识别的声学建模）表现良好，但对于具有大量样本和维度的问题（如高分辨率图像）则不太有吸引力。

本文描述了一种替代方法：使用大规模机器集群来分布深度网络的训练和推理。我们开发了一个名为 DistBelief 的软件框架，支持机器内部（通过多线程）和跨机器（通过消息传递）的**模型并行**，并行、同步和通信的细节由框架管理。除了支持模型并行，DistBelief 框架还支持**数据并行**，**即使用模型的多个副本来优化单一目标**。在该框架内，我们设计并实现了两种新的大规模分布式训练方法：(i) Downpour SGD，一种利用自适应学习率并支持大量模型副本的异步随机梯度下降过程，以及 (ii) Sandblaster L-BFGS，L-BFGS 的分布式实现，同时使用数据和模型并行。¹ Downpour SGD 和 Sandblaster L-BFGS 相比更传统的 SGD 和 L-BFGS 实现都有显著的加速效果。

我们的实验揭示了几项关于大规模非凸优化的惊人结果。首先，**异步 SGD（很少应用于非凸问题）在训练深度网络时效果非常好**，特别是与 Adagrad [10] 自适应学习率结合时。其次，我们表明，给定足够的资源，L-BFGS 可以与许多 SGD 变体竞争甚至更快。

在深度学习的特定应用方面，我们报告了两个主要发现：我们的分布式优化方法可以极大地加速中等规模模型的训练，也可以训练出原本无法想象的大规模模型。为了说明第一点，我们表明可以使用机器集群将中等规模的语音模型训练到相同的分类精度，所用时间不到 GPU 所需时间的 1/10。为了说明第二点，我们训练了一个超过 10 亿参数的大型神经网络，并使用该网络在 ImageNet 数据集（计算机视觉领域最大的数据集之一）上大幅改进了当时最先进的性能。



## 2. 相关工作

近年来，商业和学术机器学习数据集以前所未有的速度增长。作为回应，许多研究者探索通过并行化和分布式来扩展机器学习算法 [11, 12, 13, 14, 15, 16, 17]。这些研究大多集中于**线性凸模型**，其中**分布式梯度计算**是自然的第一步。**在该领域中，一些小组放宽了同步要求，探索了凸问题的延迟梯度更新** [12, 17]。与此同时，其他研究稀疏梯度问题（即对于任何给定训练样本，梯度向量中只有极小部分坐标非零的问题）的小组探索了在共享内存架构（即单台机器）上**无锁异步随机梯度下降** [5, 18]。我们感兴趣的方法兼具两者的优点，**允许使用机器集群异步计算梯度，但不要求问题是凸的或稀疏的**。

在深度学习背景下，大多数工作集中于在单台机器上训练相对较小的模型（例如 Theano [19]）。扩展深度学习的建议包括使用 GPU 集群训练一组小模型并随后平均它们的预测 [20]，或修改标准深度网络使其本质上更易于并行化 [21]。我们的重点是将深度学习技术向训练非常大的模型（数十亿参数）方向扩展，**但不限制模型的形式**。在特殊情况下，当一层占主导计算量时，一些研究者考虑了**在该层分布计算并在其余层复制计算** [5]。但在一般情况下，**当模型的许多层都是计算密集型时，需要类似于 [22] 的完全模型并行**。然而，要成功，我们认为模型并行必须与利用数据并行的巧妙分布式优化技术相结合。

我们考虑了许多现有的大规模计算工具来应对我们的问题，其中 MapReduce [23] 和 GraphLab [24] 是突出的例子。我们得出结论，MapReduce 设计用于并行数据处理，不适合深度网络训练中固有的迭代计算；而 GraphLab 设计用于通用（非结构化）图计算，无法利用深度网络中典型结构化图所提供的计算效率。

¹ 我们在 Sandblaster 框架中实现了 L-BFGS，但该通用方法也适用于各种其他批处理优化方法。



## 3. 模型并行

为了促进非常大的深度网络的训练，我们开发了一个软件框架 DistBelief，支持神经网络和分层图模型中的分布式计算。用户定义模型每层每个节点上发生的计算以及计算的上行和下行阶段应传递的消息。² 对于大型模型，用户可以将模型划分到多个机器上（图 1），使得不同节点的计算职责分配给不同的机器。框架自动使用所有可用核心并行化每台机器上的计算，并在训练和推理期间管理机器之间的通信、同步和数据传输。

![image-20260720115107990](/Users/dazhang/PycharmProject/Papers/3-RecSys/.picture/image-20260720115107990.png)

**图 1：DistBelief 中模型并行的一个示例。** 图中展示了一个五层、具有局部连接性的深度神经网络，它被划分到四台机器（蓝色矩形）上。只有那些连线跨越分区边界的节点（粗线）才需要在机器之间传输其状态。即使某个节点有多条连线跨越分区边界，其状态也只会被发送到该边界另一侧的机器一次。在每个分区内部，各个节点的计算将会在所有可用的 CPU 核心上并行执行。

将深度网络分布到多台机器的性能收益取决于 **模型的连接结构和计算需求**。**具有大量参数或高计算需求的模型通常受益于更多 CPU 和内存的访问，直到通信成本成为主导。**我们在 DistBelief 框架中成功运行了最多 144 个分区的大型模型，并获得了显著加速，而中等规模模型在最多 8 或 16 个分区时也能显示出不错的加速。（参见第 5 节中"模型并行基准"标题下的实验结果。）显然，具有 **局部连接结构**的模型比 **全连接结构**更易于大规模分布，**因为它们的通信需求较低**。**导致加速效果不理想的典型原因是不同机器间处理时间的差异，导致许多机器等待最慢的单台机器完成某个计算阶段**。尽管如此，对于我们最大的模型，我们可以有效使用 32 台机器，每台机器平均利用 16 个核心，总共使用 512 个 CPU 核心训练单个大型神经网络。**当与下一节描述的利用整个神经网络多个副本的分布式优化算法结合时，可以使用数万个 CPU 核心来训练单一模型，从而显著减少整体训练时间。**

² 对于神经网络，"上行"和"下行"也可以称为"前馈"和"反向传播"，而对于隐马尔可夫模型，它们可能更熟悉的名称是"前向"和"后向"。



## 4. 分布式优化算法

在 DistBelief 框架内并行化计算使我们能够实例化和运行比之前报道的大得多的神经网络。但为了在合理时间内训练如此大的模型，我们不仅需要**在单个模型实例内并行化计算**，还需要**跨多个模型实例分布训练**。本节我们描述第二级并行性，**即使用一组 DistBelief 模型实例（副本）同时解决单个优化问题**。

我们比较两种大规模分布式优化过程：Downpour SGD（一种在线方法）和 Sandblaster L-BFGS（一种批处理方法）。两种方法都利用集中式分片参数服务器的概念，模型副本通过该服务器共享参数。两种方法都利用了 DistBelief 在每个副本内部允许的分布式计算。但最重要的是，两种方法都设计为能够**容忍不同模型副本处理速度的差异，甚至模型副本被整体下线或随机重启的故障**。

从某种意义上说，这两种优化算法实现了一种**智能的数据并行版本**。两种方法都允许我们在许多模型副本中**同时处理不同的训练样本，并定期组合它们的结果以优化目标函数**。

### 4.1 Downpour SGD

随机梯度下降（SGD）可能是训练深度神经网络最常用的优化过程 [25, 26, 3]。不幸的是，**传统形式的 SGD 本质上是顺序的，这使得它难以应用于非常大的数据集，因为完全串行地遍历数据所需的时间是不可接受的。**

为了将 SGD 应用于大数据集，我们引入了 Downpour SGD，一种异步随机梯度下降的变体，**使用单个 DistBelief 模型的多个副本**。基本方法如下：我们将训练数据划分为多个子集，在每个子集上运行模型的一个副本。模型通过一个集中的参数服务器通信更新，该服务器维护模型所有参数的当前状态，并跨多台机器分片存储（例如，如果我们有 10 个参数服务器分片，每个分片负责存储和更新 1/10 的模型参数）（图 2）。这种方法在两个不同的方面是异步的：**模型副本彼此独立运行，参数服务器分片也彼此独立运行。**

![image-20260720115150947](/Users/dazhang/PycharmProject/Papers/3-RecSys/.picture/image-20260720115150947.png)

**图 2：左图：Downpour SGD（分布式随机梯度下降）。** 模型副本会异步地从参数服务器获取参数 w*，并将梯度 $\Delta w^*$ 推送回参数服务器。**右图：Sandblaster L-BFGS（一种分布式 L-BFGS 优化算法）。** 一个单一的“协调器”会向各个副本和参数服务器发送小型消息，以协调整个批处理优化过程。

在最简单的实现中，在处理每个小批量之前，模型副本向参数服务器请求其模型参数的更新副本。**由于 DistBelief 模型本身跨多台机器分区，每台机器只需要与持有其分区相关模型参数的那部分参数服务器分片通信**。在接收到其参数的更新副本后，DistBelief 模型副本处理一小批数据以计算梯度，并将梯度发送给参数服务器，然后参数服务器将该梯度应用于模型参数的当前值。

可以通过限制每个模型副本仅每 $n_{\text{fetch}}$ 步请求更新参数，且仅每 $n_{\text{push}}$ 步发送更新梯度值来降低 Downpour SGD 的通信开销（其中 $n_{\text{fetch}}$ 可能与 $n_{\text{push}}$ 不同）。实际上，获取参数、推送梯度以及处理训练数据的过程可以在三个仅 **弱同步的线程**中执行（参见附录算法 1.1）。在下面报告的实验中，为简单并与传统 SGD 易于比较，我们设置了 $n_{\text{fetch}} = n_{\text{push}} = 1$。

<img src="/Users/dazhang/PycharmProject/Papers/3-RecSys/.picture/image-20260720133026844.png" alt="image-20260720133026844" style="zoom:33%;" />

**算法 1.1 Downpour SGD 客户端**

> [!NOTE]
>
> - 获取参数、推送梯度、计算梯度在三个线程中运行，彼此仅**弱同步**（不需要等待对方完成）。
> - 本地参数在每一步都被更新（使用最新计算的梯度），但参数服务器只收到 `npush` 步才聚合一次的累积梯度——这就是**异步并行**的精髓，避免频繁通信瓶颈。

**Downpour SGD 比标准（同步）SGD 对机器故障更加鲁棒**。对于同步 SGD，如果一台机器发生故障，整个训练过程被延迟；而对于异步 SGD，如果一个模型副本中的一台机器发生故障，其他模型副本继续处理它们的训练数据并通过参数服务器更新模型参数。另一方面，Downpour SGD 中的多种异步处理形式在优化过程中**引入了大量额外的随机性**。最明显的是，**模型副本几乎肯定是在基于略微过时的参数集计算梯度——因为在此期间另一个模型副本很可能已经更新了参数服务器上的参数**。但除此之外还有几个其他的随机性来源：由于参数服务器**分片独立运作**，无法保证在任何给定时刻参数服务器每个分片上的参数经历了相同次数的更新，或更新以相同的顺序应用。此外，由于允许模型副本在单独的线程中获取参数和推送梯度，参数的时间戳可能存在额外的微妙不一致性。**这些操作对非凸问题的安全性几乎没有理论依据，但在实践中我们发现放松一致性要求非常有效。**

我们发现一种能极大增强 Downpour SGD 鲁棒性的技术是使用 Adagrad [10] 自适应学习率过程。Adagrad 不是在参数服务器上使用单一的固定学习率（图 2 中的 $\eta$），而是为每个参数使用单独的自适应学习率。令 $\eta_{i,K}$ 为第 $i$ 个参数在第 $K$ 次迭代时的学习率，$\Delta w_{i,K}$ 为其梯度，则我们有：

$$
\eta_{i,K} = \frac{\gamma}{\sqrt{\sum_{j=1}^K \Delta w_{i,j}^2}}.
$$

**由于这些学习率仅由每个参数的梯度平方和计算得出**，Adagrad 很容易在每个参数服务器分片内本地实现。$\gamma$（所有学习率的常数缩放因子）的值通常大于不使用 Adagrad 时的最佳固定学习率（可能大一个数量级）。Adagrad 的使用扩展了可以同时有效工作的最大模型副本数量，结合先仅使用单个模型副本进行"热启动"再释放其他副本的做法，几乎消除了使用 Downpour SGD 训练深度网络时的稳定性问题（见第 5 节中的结果）。

### 4.2 Sandblaster L-BFGS

批处理方法已被证明在小规模深度网络训练中效果良好 [7]。为了将这些方法应用于大型模型和大规模数据集，我们引入了 Sandblaster 批处理优化框架，并讨论了使用该框架实现 L-BFGS。

Sandblaster 的一个关键思想是**分布式参数存储和操作**。优化算法（如 L-BFGS）的核心驻留在协调器进程（图 2）中，该进程不直接访问模型参数。相反，协调器发出由一小组操作（如点积、缩放、逐系数加法、乘法）组成的命令，这些操作可由每个参数服务器分片独立执行，结果存储在同一分片的本地。附加信息（如 L-BFGS 的历史缓存）也存储在计算它的参数服务器分片上。这使得运行大型模型（数十亿参数）成为可能，而无需将所有参数和梯度发送到单个中心服务器的开销。（参见附录中的伪代码。）

<img src="/Users/dazhang/PycharmProject/Papers/3-RecSys/.picture/image-20260720131825653.png" alt="image-20260720131825653" style="zoom:33%;" />

> Sandblaster 是一个用于分布式批处理优化过程的框架。Sandblaster 中的一个核心概念是将操作分解为在 DistBelief 参数服务器上的本地计算。举例来说，假设我们有 10 亿个参数和 10 个参数服务器分片，那么每个分片拥有 $1/10$ 的参数。我们可以将 L-BFGS 分解为一系列标量-向量乘积（$\alpha \times \mathbf{x}$）和向量-向量内积（$\mathbf{x}^T \mathbf{y}$）的序列，其中每个向量都是 10 亿维的。如果让第一个分片始终负责 L-BFGS 内部使用的每个向量的前 $1/10$，第二个分片始终负责每个向量的第二个 $1/10$，依此类推直到最后一个分片始终负责每个向量的最后 $1/10$，那么就可以证明，这些标量-向量运算和向量-向量运算都可以通过极少的通信以分布式方式完成。这样一来，任何中间向量值结果都会自动以同样的分布式方式存储，而任何中间标量值结果则会广播到所有分片。


在典型的并行化 L-BFGS 实现中，数据分布到多台机器，每台机器负责计算特定数据子集上的梯度。**梯度被发送回中心服务器（或通过树状结构聚合 [16]）。许多此类方法等待最慢的机器完成，因此不能很好地扩展到大型共享集群**。为了解决这个问题，我们采用以下负载均衡方案：协调器为 $N$ 个模型副本中的每一个分配一小部分工作（远小于批处理总大小的 $1/N$），并在副本空闲时分配新的部分。使用这种方法，更快的模型副本比慢速副本做更多的工作。为了进一步管理批处理末尾的慢速模型副本，协调器调度待处理部分的多个副本，并使用最先完成的模型副本的结果。该方案类似于 MapReduce 框架中"备份任务"的使用 [23]。数据预取，以及通过将连续的数据部分分配给同一工作节点来支持数据亲和性，使得数据访问不成为问题。与 Downpour SGD（需要与参数服务器进行相对高频率、高带宽的参数同步）相比，Sandblaster 工作节点仅在每个批处理开始时获取参数（此时它们已被协调器更新），并且仅每几个已完成的部分发送一次梯度（以防止副本故障和重启）。



## 5. 实验

我们通过将优化算法应用于训练两个不同深度学习问题的模型来评估它们：静态图像中的目标识别 和 语音识别中的声学处理。

语音识别任务是将短音频片段中的中心区域（或帧）分类为数千个声学状态之一。我们使用了一个五层深度网络：**四个隐藏层采用 sigmoid 激活**，每层 2560 个节点，以及一个具有 8192 个节点的 softmax 输出层。输入表示为 11 个连续重叠的 25 ms 语音帧，每帧由 40 个对数能量值表示。网络是**层间全连接的**，总共有约 4200 万个模型参数。我们使用 11 亿个弱标签样本的数据集进行训练，并在保留测试集上评估。有关类似的深度网络配置和训练过程，请参见 [27]。

对于视觉目标识别，我们在 ImageNet 数据集（1600 万张图像，每张缩放到 100x100 像素）上训练了具有局部连接感受野的大型神经网络 [28]。该网络包含三个阶段，每个阶段由滤波、池化和局部对比度归一化组成，滤波层中的每个节点连接到下一层中的 10x10 图像块。我们的基础设施允许许多节点连接到相同的输入图像块，我们进行了实验，将相同连接的节点数从 8 变化到 36。输出层由 21000 个一对多逻辑分类器节点组成，每个对应一个 ImageNet 目标类别。有关类似的深度网络配置和训练过程，请参见 [29]。

**模型并行基准：** 为了探索 DistBelief 模型并行（第 3 节）的扩展行为，我们测量了单个模型实例中处理单次小批量的平均时间，作为所用分区（机器）数量的函数。在图 3 中，我们通过报告**平均训练加速比**（仅使用单台机器所花时间与使用 $N$ 台机器所花时间的比值）来量化**跨 $N$ 台机器并行化的影响**。这些模型中推理步骤的加速比类似，在此未展示。

![image-20260720115234983](/Users/dazhang/PycharmProject/Papers/3-RecSys/.picture/image-20260720115234983.png)

中等规模的语音模型在 8 台机器上运行最快，比使用单台机器快 2.2 倍。（模型配置为每台机器使用不超过 20 个核心）将模型分区到超过 8 台机器实际上减慢了训练，因为**网络开销在全连接网络结构中开始占主导地位**，且更多分区意味着每台机器执行的工作更少。

相比之下，更大的、局部连接的图像模型可以受益于使用每模型副本更多数量的机器。最大的模型具有 17 亿个参数，受益最大，使用 81 台机器获得了超过 12 倍的加速比。**对于这些大模型，使用更多机器继续增加速度，但收益递减。**

**优化方法比较：** 为了评估所提出的分布式优化过程，我们以上述语音模型以多种配置运行。我们考虑两种基线优化过程：使用传统（单副本）SGD 训练 DistBelief 模型（在 8 个分区上），以及使用 CUDA [27] 在 GPU 上训练相同模型。与这些基线方法比较的三种分布式优化方法是：使用固定学习率的 Downpour SGD、使用 Adagrad 学习率的 Downpour SGD 以及 Sandblaster L-BFGS。

图 4 显示了每种方法在训练集和测试集上的分类性能作为训练时间的函数。我们的目标是**在最短的训练时间内获得最大的测试集精度**，不考虑资源需求。传统的单副本 SGD（黑色曲线）训练最慢。使用 20 个模型副本的 Downpour SGD（蓝色曲线）显示出显著的改进。使用 20 个副本并结合 Adagrad 的 Downpour SGD（橙色曲线）适度更快。使用 2000 个模型副本的 Sandblaster L-BFGS（绿色曲线）又快了相当多。然而，**最快的是使用 Adagrad 和 200 个模型副本的 Downpour SGD（红色曲线）**。在拥有足够 CPU 资源的情况下，Sandblaster L-BFGS 和带 Adagrad 的 Downpour SGD 都可以比高性能 GPU 快得多地训练模型。

<img src="/Users/dazhang/PycharmProject/Papers/3-RecSys/.picture/image-20260720115301750.png" alt="image-20260720115301750" style="zoom:33%;" />

虽然我们未将上述实验限制在固定资源预算内，但考虑各种方法如何权衡资源消耗与性能是很有趣的。我们通过任意选择一个固定的测试集精度（16%），并测量每种方法达到该精度所花费的时间（作为机器和 CPU 核心数量的函数）来分析这一点，如图 5 所示。每条轨迹上的四个点之一对应图 4 中显示的训练配置，其他三个点是替代配置。

<img src="/Users/dazhang/PycharmProject/Papers/3-RecSys/.picture/image-20260720115334457.png" alt="image-20260720115334457" style="zoom: 33%;" />

在此图中，**靠近原点的点更优，因为它们耗时更少且使用资源更少**。在这方面，使用 Adagrad 的 Downpour SGD 似乎是最佳权衡：对于任何固定的机器或核心预算，使用 Adagrad 的 Downpour SGD 达到精度目标所需时间少于使用固定学习率的 Downpour SGD 或 Sandblaster L-BFGS。对于达到精度目标所分配的任何训练时间，使用 Adagrad 的 Downpour SGD 使用的资源少于 Sandblaster L-BFGS，并且在许多情况下**使用固定学习率的 Downpour SGD 甚至无法在截止时间内达到目标**。Sandblaster L-BFGS 系统**在其随核心数量扩展方面显示出潜力**，表明如果使用极大的资源预算（例如 30000 核心），它可能最终产生最快的训练时间。

**在 ImageNet 上的应用：** 前面的实验证明我们的技术可以加速具有数千万参数的神经网络训练。然而，我们基于集群的分布式优化方法的**更显著优势是其能够扩展到比单台机器（更不用说单个 GPU）所能容纳的大得多的模型**。作为探索非常大神经网络能力的第一步，我们使用 Downpour SGD 在上面描述的 ImageNet 目标分类任务上训练了具有 17 亿参数的图像模型。如 [29] 所述，该网络达到了超过 15% 的交叉验证分类精度，相比我们已知的 21k 类别 ImageNet 分类任务上的最佳性能提高了 60% 以上。



## 6. 结论

本文介绍了 DistBelief，一个用于深度网络并行分布式训练的框架。在该框架内，我们发现了几种有效的分布式优化策略。我们发现 Downpour SGD（一种高度异步的 SGD 变体）在训练非凸深度学习模型时效果出奇地好。Sandblaster L-BFGS（L-BFGS 的一种分布式实现）可以与 SGD 竞争，其更高效的网络带宽使用使其能够扩展到更大数量的并发核心来训练单个模型。尽管如此，Downpour SGD 与 Adagrad 自适应学习率过程的结合在使用 2000 个 CPU 核心或更少的计算预算时显然是占主导地位的方法。

Adagrad 最初并非设计用于异步 SGD，两种方法通常也不应用于非凸问题。因此，它们在高度非线性的深度网络上如此协同工作令人惊讶。我们推测 Adagrad 在面对大量异步更新的情况下自动稳定了易变参数，并自然地根据深度网络中不同层的需求调整学习率。

我们的实验表明，我们新的大规模训练方法可以使用机器集群训练即使是中等规模的深度网络，且速度显著快于 GPU，并且没有 GPU 对模型最大尺寸的限制。为了展示能够训练更大模型的价值，我们训练了一个超过 10 亿参数的模型，在 ImageNet 目标识别挑战中取得了超越当时最先进的性能。



## 致谢

作者感谢 Samy Bengio, Tom Dean, John Duchi, Yuval Netzer, Patrick Nguyen, Yoram Singer, Sebastian Thrun 和 Vincent Vanhoucke 给予的不可或缺的建议、支持和意见。



## 参考文献

[1] G. Dahl, D. Yu, L. Deng, and A. Acero. Context-dependent pre-trained deep neural networks for large vocabulary speech recognition. IEEE Transactions on Audio, Speech, and Language Processing, 2012.

[2] G. Hinton, L. Deng, D. Yu, G. Dahl, A. Mohamed, N. Jaitly, A. Senior, V. Vanhoucke, P. Nguyen, T. Sainath, and B. Kingsbury. Deep neural networks for acoustic modeling in speech recognition. IEEE Signal Processing Magazine, 2012.

[3] D. C. Ciresan, U. Meier, L. M. Gambardella, and J. Schmidhuber. Deep big simple neural nets excel on handwritten digit recognition. CoRR, 2010.

[4] A. Coates, H. Lee, and A. Y. Ng. An analysis of single-layer networks in unsupervised feature learning. In AISTATS 14, 2011.

[5] Y. Bengio, R. Ducharme, P. Vincent, and C. Jauvin. A neural probabilistic language model. Journal of Machine Learning Research, 3:1137–1155, 2003.

[6] R. Collobert and J. Weston. A unified architecture for natural language processing: Deep neural networks with multitask learning. In ICML, 2008.

[7] Q.V. Le, J. Ngiam, A. Coates, A. Lahiri, B. Prochnow, and A.Y. Ng. On optimization methods for deep learning. In ICML, 2011.

[8] R. Raina, A. Madhavan, and A. Y. Ng. Large-scale deep unsupervised learning using graphics processors. In ICML, 2009.

[9] J. Martens. Deep learning via hessian-free optimization. In ICML, 2010.

[10] J. C. Duchi, E. Hazan, and Y. Singer. Adaptive subgradient methods for online learning and stochastic optimization. Journal of Machine Learning Research, 12:2121–2159, 2011.

[11] Q. Shi, J. Petterson, G. Dror, J. Langford, A. Smola, A. Strehl, and V. Vishwanathan. Hash kernels. In AISTATS, 2009.

[12] J. Langford, A. Smola, and M. Zinkevich. Slow learners are fast. In NIPS, 2009.

[13] G. Mann, R. McDonald, M. Mohri, N. Silberman, and D. Walker. Efficient large-scale distributed training of conditional maximum entropy models. In NIPS, 2009.

[14] R. McDonald, K. Hall, and G. Mann. Distributed training strategies for the structured perceptron. In NAACL, 2010.

[15] M. Zinkevich, M. Weimer, A. Smola, and L. Li. Parallelized stochastic gradient descent. In NIPS, 2010.

[16] A. Agarwal, O. Chapelle, M. Dudik, and J. Langford. A reliable effective terascale linear learning system. In AISTATS, 2011.

[17] A. Agarwal and J. Duchi. Distributed delayed stochastic optimization. In NIPS, 2011.

[18] F. Niu, B. Retcht, C. Re, and S. J. Wright. Hogwild! A lock-free approach to parallelizing stochastic gradient descent. In NIPS, 2011.

[19] J. Bergstra, O. Breuleux, F. Bastien, P. Lamblin, R. Pascanu, G. Desjardins, J. Turian, D. Warde-Farley, and Y. Bengio. Theano: a CPU and GPU math expression compiler. In SciPy, 2010.

[20] D. Ciresan, U. Meier, and J. Schmidhuber. Multi-column deep neural networks for image classification. Technical report, IDSIA, 2012.

[21] L. Deng, D. Yu, and J. Platt. Scalable stacking and learning for building deep architectures. In ICASSP, 2012.

[22] A. Krizhevsky. Learning multiple layers of features from tiny images. Technical report, U. Toronto, 2009.

[23] J. Dean and S. Ghemawat. Map-Reduce: simplified data processing on large clusters. CACM, 2008.

[24] Y. Low, J. Gonzalez, A. Kyrola, D. Bickson, C. Guestrin, and J. Hellerstein. Distributed GraphLab: A framework for machine learning in the cloud. In VLDB, 2012.

[25] L. Bottou. Stochastic gradient learning in neural networks. In Proceedings of Neuro-N^imes 91, 1991.

[26] Y. LeCun, L. Bottou, G. Orr, and K. Muller. Efficient backprop. In Neural Networks: Tricks of the trade. Springer, 1998.

[27] V. Vanhoucke, A. Senior, and M. Z. Mao. Improving the speed of neural networks on cpus. In Deep Learning and Unsupervised Feature Learning Workshop, NIPS 2011, 2011.

[28] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei. ImageNet: A Large-Scale Hierarchical Image Database. In CVPR, 2009.

[29] Q.V. Le, M.A. Ranzato, R. Monga, M. Devin, K. Chen, G.S. Corrado, J. Dean, and A.Y. Ng. Building high-level features using large scale unsupervised learning. In ICML, 2012.
