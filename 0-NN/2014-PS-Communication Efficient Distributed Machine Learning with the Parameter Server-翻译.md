# Communication Efficient Distributed Machine Learning with the Parameter Server

> Mu Li∗†, David G. Andersen∗, Alexander Smola∗‡, and Kai Yu† | ∗Carnegie Mellon University, †Baidu, ‡Google

This paper describes a third-generation parameter server framework for distributed machine learning, offering two relaxations to balance system performance and algorithm efficiency. A new algorithm is proposed that leverages this framework to solve non-convex non-smooth problems with convergence guarantees. Experiments on two large-scale problems — ℓ1-regularized logistic regression on CPUs and reconstruction ICA on GPUs — using 636TB of real data with hundreds of billions of samples and dimensions demonstrate that the parameter server framework is an effective and straightforward way to scale machine learning to unprecedented sizes.

Key findings: (1) Asynchronous task dependencies and user-defined filters provide flexible consistency models that trade off convergence rate for system performance. (2) The delayed block proximal gradient method provably converges to a stationary point under relaxed consistency. (3) The framework achieves 2\\times speedup over specialized systems on a 636TB dataset, requiring only 300 lines of code. (4) User-defined filters (KKT, key caching, compressing) yield 40\\times and 12\\times compression for servers and workers respectively.

---

## 摘要

本文描述了一种用于分布式机器学习的第三代参数服务器（parameter server）框架。该框架提供了两种松弛机制来平衡系统性能和算法效率。我们提出了一种新算法，利用该框架解决非凸非光滑问题，并具有收敛保证。我们对两个大规模机器学习问题进行了深入分析，从CPU上的ℓ1-正则化逻辑回归到GPU上的重建ICA（reconstruction ICA），使用了636TB的真实数据，包含数千亿的样本和维度。我们通过这些示例证明，参数服务器框架是一种有效且直接的方式，可以将机器学习扩展到比以往更大的问题和系统。

## 1 引言

在现实的工业机器学习应用中，数据集范围从1TB到1PB。例如，一个拥有1亿用户、每个用户1KB数据的社交网络就有100TB。在线广告和用户生成内容分析中的问题具有类似数量级的复杂性[12]。如此海量的数据使得学习拥有 $10^9$ 到 $10^{12}$ 个参数的强大而复杂的模型成为可能[9]，在这样的规模下，单台机器通常无法及时完成任务。

分布式优化正成为解决大规模机器学习问题的关键工具[1, 3, 10, 21, 19]。工作负载被划分到工作机器（worker machines）上，这些机器在同时执行局部计算以优化模型时访问全局共享的模型。然而，为机器学习应用高效实现分布式优化算法并不容易。一个主要挑战是机器间的数据通信：

*   工作机器必须频繁读写全局共享参数。这种大规模数据访问需要巨大的网络带宽。然而，带宽是数据中心中最稀缺的资源之一[6]，通常比内存带宽小10-100倍，并且在所有运行的应用和机器之间共享。这导致了巨大的通信开销，并成为分布式优化算法的瓶颈。
*   许多优化算法是顺序的，需要工作机器之间频繁同步。在每次同步中，所有机器都需要等待最慢的机器。然而，由于不完美的工作负载划分、网络拥塞或其他运行任务的干扰，慢机器是不可避免的，这便成为另一个瓶颈。

在这项工作中，我们基于先前设计开源第三代参数服务器框架的工作[4]，以理解其可应用的机器学习算法范围及其收益。图1展示了在一些最先进系统上执行的最大规模机器学习实验的概况。我们在可能的情况下与这些系统的作者进行了确认。

$$
\begin{array}{c}
\text{Figure 1: Comparison of the public largest machine learning experiments each system performed.} \\
\text{The results are current as of April 2014.} \\
\text{Compared to these systems, our parameter server is several orders of magnitude more scalable} \\
\text{in terms of both parameters and nodes.}
\end{array}
$$

参数服务器异步通信数据以减少通信成本。由此产生的数据不一致性是系统性能和算法收敛速度之间的权衡。该系统提供两种松弛机制来解决数据（不）一致性问题：第一，我们不主张特定的一致性模型（consistency model）[29, 7, 15]，而是支持灵活的一致性模型。第二，系统允许用户自定义过滤器（user-specific filters）进行细粒度的一致性管理。此外，系统还提供数据复制、即时故障切换和弹性可扩展等其他功能。

**激励性应用（Motivating Application）。** 考虑以下一般的正则化优化问题：

$$
\operatorname*{minimize}_{w} F(w) \quad \text{where} \quad F(w) := f(w) + h(w) \quad \text{and} \quad w \in \mathbb{R}^p, \qquad (1)
$$

我们假设损失函数 $f: \mathbb{R}^p \to \mathbb{R}$ 是连续可微的但不一定是凸的，正则化项 $h: \mathbb{R}^p \to \mathbb{R}$ 是凸的、左连续的、分块可分的（block separable），但可能是非光滑的。

所提出的算法基于近端梯度法（proximal gradient method）[23] 解决该问题。然而，它在四个方面与后者不同，以高效处理超高维和稀疏数据：

*   每次只更新坐标的一个子集（块）：(块) Gauss-Seidel 更新被证明在稀疏数据上是高效的[36, 27]。
*   由于异步数据通信，工作机器维护的模型仅与其他机器部分一致。
*   近端算子（proximal operator）使用坐标特定的学习率（coordinate-specific learning rates），以适应数据中固有的稀疏性模式。
*   只通信那些会改变相关模型权重的坐标，以减少网络流量。

我们通过将该算法应用于两个具有挑战性的问题来展示其效率：(1) 在超过1000亿样本和特征（features）的稀疏文本数据集上的非光滑 ℓ1-正则化逻辑回归；(2) 一个非凸非光滑的 ICA 重建问题[18]，从密集图像数据中提取数十亿稀疏特征。我们证明，所提出的算法和系统的组合有效减少了通信成本和编程工作量。特别是，仅需300行代码即可实现工业规模问题中几乎零通信开销的 ℓ1-正则化逻辑回归。

**论文概览（Outline）：** 我们首先在第2节提供背景。接下来，我们在第3节讨论两种松弛机制，在第4节讨论所提出的算法。在第5节（以及附录B和C），我们展示应用和实验结果。最后在第6节进行总结讨论。

## 2 背景

**相关工作（Related Work）。** 参数服务器框架（parameter server framework）[29] 已在学术界和工业界广泛普及。相关的系统已在亚马逊、百度、Facebook、谷歌[10]、微软和雅虎[2]实现。也有开源代码，如 YahooLDA [2] 和 Petuum [15]。正如[29, 2]中介绍的，第一代参数服务器缺乏灵活性和性能。第二代参数服务器是应用特定的，以 Distbelief [10] 和 [20] 中的同步机制为例。Petuum 通过施加有界延迟（bounded delay）而非最终一致性（eventual consistency）修改了 YahooLDA，并旨在成为一个通用平台[15]，但它对工作机器的线程模型施加了更多约束。与先前工作相比，我们的第三代系统显著提升了系统性能，同时还提供了灵活性和容错性。

除参数服务器之外，还有许多用于机器学习应用的通用分布式系统。许多系统强制采用同步和迭代通信。例如，基于 Hadoop [13] 的 Mahout [5] 和基于 Spark [37] 的 MLI [30] 都采用了迭代 MapReduce 框架（iterative MapReduce framework）[11]。另一方面，Graphlab [21] 以尽力而为的方式支持全局参数同步。这些系统在专用研究集群上能够扩展到数百个节点。然而，在更大规模下，同步需求会造成性能瓶颈。与这些系统相比，参数服务器的主要优势在于其一致性模型的灵活性。

人们对异步算法的兴趣也在增长。Shotgun [7] 作为 Graphlab 的一部分，执行并行坐标下降（parallel coordinate descent）以解决 ℓ1 优化问题。其他方法将观测数据划分到多台机器上，并以数据并行方式更新模型[34, 17, 38, 3, 1, 19]。Hogwild [26] 提出了免锁变体。将数据和参数划分为非重叠组件的混合变体在[33]中引入，但代价是必须在多台机器上移动或复制数据。最后，NIPS 框架[31]讨论了一般的非凸近似近端方法。

所提出的算法与现有方法主要在两个方面的不同。首先，我们专注于解决大规模问题。鉴于数据规模和有限的网络带宽，Shotgun 和 Hogwild 的共享内存方法以及在训练期间移动整个数据都是不可取的。其次，我们旨在解决一般的非凸非光滑复合目标函数。与[31]不同，我们在更弱的假设下推导收敛定理，并且我们进行了规模大许多数量级的实验。

**参数服务器架构（The Parameter Server Architecture）。** 参数服务器的一个实例[4]包含一个服务器组（server group）和若干个工作组（worker groups），每个组包含多台机器。服务器组中的每台机器维护全局参数的一部分，所有服务器相互通信以复制和/或迁移参数，从而实现可靠性和扩展。

工作机器仅存储一部分训练数据，并计算局部梯度或其他统计量。工作机器仅与服务器通信以检索和更新共享参数。在每个工作组中，可能有一个调度机器（scheduler machine），它分配工作负载给工作机器并监控其进度。当工作机器被添加或从组中移除时，调度器可以重新调度未完成的工作负载。每个工作组运行一个应用程序，从而支持多租户。例如，广告投放系统和推理算法可以在不同的工作组中同时运行。

共享模型参数被表示为排序的 (key, value) 对。或者，我们可以将其视为一个稀疏向量或矩阵，通过内置的多线程线性代数函数与训练数据进行交互。数据交换可以通过两种操作实现：push 和 pull。工作机器可以将一个范围内的所有 (key, value) 对 push 到服务器，或从服务器 pull 相应的值。

**分布式次梯度下降（Distributed Subgradient Descent）。** 对于(1)中介绍的激励性示例，我们可以使用参数服务器实现标准的分布式次梯度下降算法[34]。如图2和算法1所示，训练数据被划分并分布到所有工作机器上。模型 $w$ 被迭代学习。在每次迭代中，每个工作机器使用自己的训练数据计算局部梯度，服务器聚合这些梯度以更新全局共享参数 $w$ 。然后工作机器从服务器检索更新后的权重。

工作机器需要模型 $w$ 来计算梯度。然而，对于非常高维的训练数据，模型可能无法放入一台工作机器中。幸运的是，这些数据通常是稀疏的，工作机器通常只需要模型的一个子集。为了说明这一点，我们将第5节中使用的数据集中的样本随机分配给工作机器，然后统计工作机器计算梯度所需的模型参数。我们发现，当使用100个工作机器时，平均每个工作机器只需要模型的7.8%。当使用10,000个工作机器时，这一比例降至0.15%。因此，尽管 $w$ 的总规模很大，但特定工作机器所需的 $w$ 的工作集（working set）可以轻松地被缓存。

---

**Algorithm 1 Distributed Subgradient Descent Solving (1) in the Parameter Server**

Worker $r = 1, \ldots, m$ :
1: Load a part of training data $\{y_{i_k}, x_{i_k}\}_{k=1}^{n_r}$
2: Pull the working set $w^{(0)}_r$ from servers
3: **for** $t = 1$ to $T$ **do**
4: $\quad$ Gradient $g^{(t)}_r \leftarrow \sum_{k=1}^{n_r} \partial \ell(x_{i_k}, y_{i_k}, w^{(t)}_r)$
5: $\quad$ Push $g^{(t)}_r$ to servers
6: $\quad$ Pull $w^{(t+1)}_r$ from servers
7: **end for**

Servers:
1: **for** $t = 1$ to $T$ **do**
2: $\quad$ Aggregate $g^{(t)} \leftarrow \sum_{r=1}^m g^{(t)}_r$
3: $\quad$ $w^{(t+1)} \leftarrow w^{(t)} - \eta \big( g^{(t)} + \partial h(w^{(t)}) \big)$
4: **end for**

---

Figure 2: One iteration of Algorithm 1. Each worker only caches the working set of $w$ .

## 3 数据一致性的两种松弛机制

我们现在介绍对所提出系统至关重要的两种松弛机制。我们鼓励对系统细节（如服务器键布局、弹性可扩展性和持续容错）感兴趣的读者参阅我们先前的工作[4]。

### 3.1 异步任务依赖

我们将参数服务器中的工作负载分解为由调用方（caller）向远程被调用方（callee）发起的任务（tasks）。在任务的构成方面有很大的灵活性：例如，一个任务可以是工作机器向服务器发起的 push 或 pull，也可以是调度器向任何节点发起的用户自定义函数，例如分布式次梯度算法中的一次迭代。任务也可以包含子任务（subtasks）。例如，在算法1中，工作机器每次迭代执行一次 push 和一次 pull。

任务是异步执行的：调用方在发出任务后可以立即执行进一步的计算。调用方只有在收到被调用方的回复后才将任务标记为完成。回复可以是用户自定义函数的返回值、pull 请求的 (key, value) 对，或一个空的确认。被调用方只有在该任务的调用返回且该调用发出的所有子任务都完成时，才将该任务标记为完成。

```
iter 10:    gradient  push & pull
iter 11:       gradient  push & pull
iter 12:          gradient    pu
```

默认情况下，被调用方并行执行任务以获得最佳性能。希望使任务执行顺序化的调用方可以在任务之间插入一个执行完成后执行（execute-after-finished）依赖关系。右侧的图表说明了三个任务的执行。任务10和11是独立的，但12依赖于11。因此，被调用方在任务10中计算完梯度后立即开始任务11。然而，任务12被推迟到任务11的 pull 完成之后。

任务依赖有助于实现算法逻辑。例如，算法1中服务器的聚合逻辑可以通过使更新任务依赖于所有工作机器的 push 任务来实现。通过这种方式，权重 $w$ 仅在所有工作机器梯度被聚合后才更新。

### 3.2 通过任务依赖图实现灵活的一致性模型

上述引入的依赖图可用于放松一致性要求。独立任务通过并行化 CPU、磁盘和网络带宽的使用来提高系统效率。然而，这可能导致节点间的数据不一致。在上面的图中，工作机器 $r$ 在拉回更新后的模型 $w^{(11)}_r$ 之前就开始了迭代11，因此它使用了过时的模型 $w^{(10)}_r$ 并计算了与迭代10中相同的梯度，即 $g^{(11)}_r = g^{(10)}_r$ 。这种不一致可能会潜在地降低算法1的收敛速度。然而，某些算法可能对这种不一致不那么敏感。例如，如果在算法2的每次迭代中只更新 $w$ 的一个块（block），那么不等待迭代10就启动迭代11仅会导致 $w$ 的一部分不一致。

算法效率和系统性能之间的权衡在实践中取决于多种因素，如特征相关性、硬件能力、数据中心负载等。与其他强制算法设计者采用可能不适合实际情况的特定一致性模型的系统不同，参数服务器可以通过创建任务依赖图（task dependency graphs）为不同的一致性模型提供完全的灵活性。任务依赖图是由任务及其依赖关系定义的有向无环图。考虑以下三个示例：

```
(a) Sequential:   0 \\to 1 \\to 2
(b) Eventual:     0, 1, 2  (all concurrent)
(c) 1 Bounded delay: 0, 1 can start; 2 must wait for 0; 3 must wait for 1; etc.
```

**顺序一致性（Sequential Consistency）** 要求所有任务一个一个地执行。下一个任务只有在前一个任务完成后才能开始。它产生与单线程实现相同的结果。批量同步处理（Bulk Synchronous Processing）使用这种方法。

**最终一致性（Eventual Consistency）** 则允许所有任务同时启动。[29]为 LDA 描述了这样一个系统。只有当底层算法对延迟非常鲁棒时，才推荐使用这种方法。

**有界延迟（Bounded Delay）** 限制了参数的陈旧度。当设置最大延迟时间 $\tau$ 时，新任务将被阻塞，直到 $\tau$ 时间之前的所有先前任务都已完成（ $\tau=0$ 产生顺序一致性，而 $\tau=\infty$ 则恢复为最终一致性）。算法2使用了这种模型。

请注意，依赖图允许更高级的一致性模型。例如，调度器可以根据运行时的进展增加或减少最大延迟，以动态平衡效率与收敛之间的权衡。

### 3.3 通过用户自定义过滤器实现灵活的一致性模型

任务依赖图管理任务之间的数据一致性。用户自定义过滤器允许对一致性进行更细粒度的控制（例如，在任务内部）。过滤器可以转换和选择性同步任务中通信的 (key, value) 对。多个过滤器可以一起使用以实现更好的数据压缩。一些示例过滤器包括：

*   **显著修改过滤器（Significantly modified filter）：** 仅推送自上次同步以来变化超过阈值的条目。
*   **随机跳过过滤器（Random skip filter）：** 在发送前对条目进行子采样。它们在计算中被跳过。
*   **KKT 过滤器（KKT filter）：** 利用求解近端算子时的最优性条件：工作机器仅推送可能影响服务器上权重的梯度。我们将在第5节中更详细地讨论它。
*   **键缓存过滤器（Key caching filter）：** 每次通信一个范围内的 (key, value) 对，因为 push 和 pull 是基于范围的。当再次选择相同的范围时，很可能只有值被修改而键保持不变。如果发送方和接收方都缓存了这些键，那么发送方只需发送带有键签名的值。因此，我们有效地将网络带宽翻倍。
*   **压缩过滤器（Compressing filter）：** 通信的值通常是可压缩的数字，如零、小整数和精度过剩的浮点数。该过滤器通过使用无损或有损数据压缩算法来减少数据大小[1]。

## 4 延迟块近端梯度法

在本节中，我们提出一种高效算法，利用参数服务器解决先前定义的非凸非光滑优化问题(1)。

---

**Algorithm 2 Delayed Block Proximal Gradient Method Solving (1)**

Scheduler:
1: Partition parameters into $k$ blocks $b_1, \ldots, b_k$
2: **for** $t = 1$ to $T$ : Pick a block $b_{i_t}$ and issue the task to workers

Worker $r$ at iteration $t$ :
1: Wait until all iterations before $t - \tau$ are finished
2: Compute first-order gradient $g^{(t)}_r$ and coordinate-specific learning rates $u^{(t)}_r$ on block $b_{i_t}$
3: Push $g^{(t)}_r$ and $u^{(t)}_r$ to servers with user-defined filters, e.g., the random skip or the KKT filter
4: Pull $w^{(t+1)}_r$ from servers with user-defined filters, e.g., the significantly modified filter

Servers at iteration $t$ :
1: Aggregate $g^{(t)}$ and $u^{(t)}$
2: Solve the generalized proximal operator (2) $w^{(t+1)} \leftarrow \operatorname{Prox}^{U}_{\gamma_t}(w^{(t)})$ with $U = \operatorname{diag}(u^{(t)})$ .

---

**近端梯度法（Proximal Gradient Methods）。** 对于一个闭真凸函数 $h(x): \mathbb{R}^p \to \mathbb{R} \cup \{\infty\}$ ，定义广义近端算子

$$
\operatorname{Prox}^{U}_{\gamma}(x) := \arg\min_{y \in \mathbb{R}^p} h(y) + \frac{1}{2\gamma} \|x - y\|^2_U \quad \text{where} \quad \|x\|^2_U := x^\top U x. \qquad (2)
$$

马氏范数（Mahalanobis norm） $\|x\|_U$ 是相对于一个半正定矩阵 $U \succeq 0$ 定义的。许多近端算法选择 $U = 1$ 。为最小化复合目标函数 $f(w) + h(w)$ ，近端梯度算法分两步更新 $w$ ：对 $f$ 进行最速梯度下降的前向步骤（forward step）和使用 $h$ 进行投影的后向步骤（backward step）。给定迭代 $t$ 的学习率 $\gamma_t > 0$ ，这两步可以写成：

$$
w^{(t+1)} = \operatorname{Prox}^{U}_{\gamma_t} \left[ w^{(t)} - \gamma_t \nabla f(w^{(t)}) \right] \quad \text{for } t = 1, 2, \ldots \qquad (3)
$$

**算法（Algorithm）。** 我们通过块方案（block scheme）放松了近端梯度法的一致性模型，以降低对数据不一致的敏感性。所提出的算法如算法2所示。它在四个重要方面与标准方法以及算法1不同，以利用参数服务器提供的机会并处理高维稀疏数据。

1.  每次迭代仅更新一个参数块。
2.  工作机器在此块上同时计算梯度和坐标特定的学习率，例如二阶导数的对角部分。
3.  迭代是异步的。我们在迭代间使用有界延迟模型（bounded-delay model）。
4.  我们使用用户自定义过滤器来抑制传输那些对模型影响可能可忽略的部分数据。

**收敛性分析（Convergence Analysis）。** 为了证明收敛性，我们需要做出一些假设。和之前一样，我们将损失 $f$ 分解为与工作机器 $i$ 存储的训练数据相关的块 $f_i$ ，即 $f = \sum_i f_i$ 。接下来，我们假设在第 $t$ 次迭代选择了块 $b_t$ 。一个关键假设是，对于给定的参数变化， $f$ 的梯度变化率是有界的。更具体地说，我们需要限制影响当前块的变化量以及与其他块的"串扰"量。

**假设1（块 Lipschitz 连续性）** 存在正常数 $L_{\text{var},i}$ 和 $L_{\text{cov},i}$ ，使得对于任意迭代 $t$ 和所有 $x, y \in \mathbb{R}^p$ ，其中对于任何 $i \notin b_t$ 有 $x_i = y_i$ ，满足：

$$
\|\nabla_{b_t} f_i(x) - \nabla_{b_t} f_i(y)\| \le L_{\text{var},i} \|x - y\| \quad \text{for } 1 \le i \le m \qquad (4a)
$$
$$
\|\nabla_{b_s} f_i(x) - \nabla_{b_s} f_i(y)\| \le L_{\text{cov},i} \|x - y\| \quad \text{for } 1 \le i \le m,\; t < s \le t + \tau \qquad (4b)
$$

其中 $\nabla_b f(x)$ 是 $\nabla f(x)$ 的块 $b$ 。进一步定义 $L_{\text{var}} := \sum_{i=1}^m L_{\text{var},i}$ 和 $L_{\text{cov}} := \sum_{i=1}^m L_{\text{cov},i}$ 。

下面的定理2表明，在松弛的一致性模型下，只要选择了合适的学习率，该算法会收敛到一个驻点（stationary point）。注意，由于总体目标是非凸的，通常不可能保证最优性。

**定理2** 假设更新操作的延迟以 $\tau$ 为界，还假设我们在推送梯度时应用随机跳过过滤器，在拉取权重时应用显著修改过滤器，阈值为 $O(t^{-1})$ 。此外，假设损失的梯度如假设1所述是 Lipschitz 连续的。记 $M_t$ 为时间 $t$ 时最小坐标特定学习率。对于任意 $\epsilon > 0$ ，如果学习率 $\gamma_t$ 满足：

$$
\gamma_t \le \frac{M_t}{L_{\text{var}} + \tau L_{\text{cov}} + \epsilon} \quad \text{对于所有 } t > 0, \qquad (5)
$$

则算法2期望收敛到一个驻点。

证明见附录A。直观地说，当接近驻点时， $w^{(t-\tau)}$ 和 $w^{(t)}$ 之间的差异会很小。因此，梯度的变化也会消失。因此，通过延迟和不精确模型得到的不精确梯度很可能是真实梯度的良好近似，从而可以应用近端梯度法的收敛结果。

注意，当延迟增加时，我们应减小学习率以保证收敛。然而，当选择仔细的块划分和顺序时，可以使用更大的值。例如，如果块中的特征相关性较小，则 $L_{\text{var}}$ 减小。如果块与先前块的相关性较小，则 $L_{\text{cov}}$ 减小，如[26, 7]中所利用的。

## 5 实验

我们现在展示如何使用上述通用框架解决具有挑战性的机器学习问题。由于篇幅限制，我们仅展示下面一个0.6PB数据集的实验结果。较小数据集的详细信息见附录B。此外，我们在附录C中讨论非光滑重建 ICA。

**设置（Setup）。** 我们选择 ℓ1-正则化逻辑回归进行评估，因为它是工业界用于大规模风险最小化最流行的算法之一[9]。我们收集了一个广告点击预测数据集，包含1700亿个样本和650亿个唯一特征。未压缩的数据集大小为636TB。我们在1000台机器上运行参数服务器，每台机器配备16个CPU核心、192GB DRAM，并通过10 Gb以太网连接。800台机器作为工作机器，200台作为服务器。集群在操作期间同时被其他作业使用。

**算法（Algorithm）。** 我们采用算法2，使用 Hessian 矩阵对角元素的上界作为坐标特定的学习率。特征根据特征组信息被随机分成580个块。我们通过观察收敛速度选择了固定学习率。

我们设计了一个 Karush-Kuhn-Tucker (KKT) 过滤器来跳过非活动（inactive）坐标。它类似于 SVM 优化的主动集选择策略（active-set selection strategies）[16]和主动集选择器[22]。假设坐标 $k$ 的 $w_k = 0$ ， $g_k$ 为当前梯度。根据近端算子的最优性条件（也称为软收缩算子 soft-shrinkage operator），如果 $|g_k| \le \lambda$ ， $w_k$ 将保持为0。因此，工作机器无需发送 $g_k$ （以及 $u_k$ ）。我们使用旧值 $\hat{g}_k$ 来近似 $g_k$ 以进一步避免计算 $g_k$ 。因此，如果 $|\hat{g}_k| \le \lambda - \delta$ ，坐标 $k$ 将在 KKT 过滤器中被跳过，其中 $\delta \in [0, \lambda]$ 控制过滤的激进程度。

**实现（Implementation）。** 据我们所知，没有开源系统能够将稀疏逻辑回归扩展到本文所描述的规模。Graphlab 仅提供多线程的单机实现。我们在附录B中将其与我们的进行了比较。Mlbase、Petuum 和 REEF 不支持稀疏逻辑回归（已于2014年4月与作者确认）。我们将参数服务器与一家大型互联网公司开发的两个专用第二代参数服务器（称为系统A和系统B）进行比较。

系统A和B都采用顺序一致性模型，但前者使用 L-BFGS 的变体，而后者运行与我们类似的算法。值得注意的是，这两个系统都由超过10,000行代码组成。参数服务器实现与系统B（后者由本文的一位作者开发）相同的功能仅需300行代码。参数服务器成功地将大部分系统复杂性从算法实现转移到了可重用组件中。

$$

\begin{array}{c}
\text{Figure 3: Convergence of sparse logistic regression on a 636TB dataset.} \\
\text{Figure 4: Average time per worker spent on computation and waiting during optimization.} \\
\text{Figure 5: Time to reach the same convergence criteria under various allowed delays.} \\
\text{Figure 6: The reduction of sent data size when stacking various filters together.}
\end{array}
$$

**实验结果（Experimental Results）。** 我们通过运行这些系统以达到相同的收敛标准来比较它们。图3显示，系统B由于其更好的算法而优于系统A。参数服务器在使用基本相同算法的情况下，进一步将系统B加速了2倍。它之所以能做到这一点，是因为一致性松弛显著减少了等待时间（图4）。图5显示，增加允许的延迟显著减少了等待时间，尽管略微降低了收敛速度。最佳权衡是8-延迟，与顺序一致性模型相比实现了1.6倍的加速。如图6所示，键缓存节省了50%的网络流量。由于模型稀疏性，压缩显著减少了服务器的流量，但由于梯度通常非零，它对工作机器的效果较差。但这些梯度可以被 KKT 过滤器高效过滤。总的来说，这些过滤器为服务器和工作机器分别提供了40倍和12倍的压缩率。

## 6 结论

本文研究了将第三代参数服务器框架应用于现代分布式机器学习算法的问题。我们证明，可以设计出非常适合该框架的算法；本文中，我们提出了一种异步块近端梯度法，用于解决一般非凸非光滑问题，并具有可证明的收敛性。该算法与参数服务器框架中可用的松弛机制完美匹配：通过任务依赖实现可控异步性，以及通过用户可定义过滤器减少数据通信量。我们对真实数据集上的几个具有挑战性的任务进行了实验（数据集大小达0.6PB，包含数千亿个样本和特征），以证明其效率。我们相信，这种第三代参数服务器是可扩展机器学习的一个重要且有用的构建模块。最后，源代码可在 http://parameterserver.org 获取。

## 参考文献

[1] A. Agarwal and J. C. Duchi. Distributed delayed stochastic optimization. In *IEEE CDC*, 2012.
[2] A. Ahmed, M. Aly, J. Gonzalez, S. Narayanamurthy, and A. J. Smola. Scalable inference in latent variable models. In *WSDM*, 2012.
[3] A. Ahmed, N. Shervashidze, S. Narayanamurthy, V. Josifovski, and A. J. Smola. Distributed large-scale natural graph factorization. In *WWW*, 2013.
[4] M. Li, D. G. Andersen, J. Park, A. J. Smola, A. Amhed, V. Josifovski, J. Long, E. Shekita, and B. Y. Su. Scaling Distributed Machine Learning with the Parameter Server. In *OSDI*, 2014.
[5] Apache Foundation. Mahout project, 2012. http://mahout.apache.org.
[6] L. A. Barroso and H. Hölzle. The datacenter as a computer: An introduction to the design of warehouse-scale machines. *Synthesis lectures on computer architecture*, 4(1):1–108, 2009.
[7] J.K. Bradley, A. Kyrola, D. Bickson, and C. Guestrin. Parallel coordinate descent for L1-regularized loss minimization. In *ICML*, 2011.
[8] J. Byers, J. Considine, and M. Mitzenmacher. Simple load balancing for distributed hash tables. In *Peer-to-peer systems II*, pages 80–87. Springer, 2003.
[9] K. Canini. Sibyl: A system for large scale supervised machine learning. Technical Talk, 2012.
[10] J. Dean, G. Corrado, R. Monga, K. Chen, M. Devin, Q. Le, M. Mao, M. Ranzato, A. Senior, P. Tucker, K. Yang, and A. Ng. Large scale distributed deep networks. In *NIPS*, 2012.
[11] J. Dean and S. Ghemawat. MapReduce: simplified data processing on large clusters. *CACM*, 2008.
[12] Domo. Data Never Sleeps 2.0, 2014. http://www.domo.com/learn.
[13] The Apache Software Foundation. Apache hadoop, 2009. http://hadoop.apache.org/core/.
[14] S. H. Gunderson. Snappy. https://code.google.com/p/snappy/.
[15] Q. Ho, J. Cipar, H. Cui, S. Lee, J. Kim, P. Gibbons, G. Gibson, G. Ganger, and E. Xing. More effective distributed ml via a stale synchronous parallel parameter server. In *NIPS*, 2013.
[16] T. Joachims. Making large-scale SVM learning practical. *Advances in Kernel Methods*, 1999.
[17] J. Langford, A. J. Smola, and M. Zinkevich. Slow learners are fast. In *NIPS*, 2009.
[18] Q.V. Le, A. Karpenko, J. Ngiam, and A.Y. Ng. ICA with reconstruction cost for efficient overcomplete feature learning. *NIPS*, 2011.
[19] M. Li, D. G. Andersen, and A. J. Smola. Distributed delayed proximal gradient methods. In *NIPS Workshop on Optimization for Machine Learning*, 2013.
[20] M. Li, L. Zhou, Z. Yang, A. Li, F. Xia, D.G. Andersen, and A. J. Smola. Parameter server for distributed machine learning. In *Big Learning NIPS Workshop*, 2013.
[21] Y. Low, J. Gonzalez, A. Kyrola, D. Bickson, C. Guestrin, and J. M. Hellerstein. Distributed graphlab: A framework for machine learning and data mining in the cloud. In *PVLDB*, 2012.
[22] S. Matsushima, S.V.N. Vishwanathan, and A.J. Smola. Linear support vector machines via dual cached loops. In *KDD*, 2012.
[23] N. Parikh and S. Boyd. Proximal algorithms. In *Foundations and Trends in Optimization*, 2013.
[24] K. B. Petersen and M. S. Pedersen. The matrix cookbook, 2008. Version 20081110.
[25] A. Phanishayee, D. G. Andersen, H. Pucha, A. Povzner, and W. Belluomini. Flex-kv: Enabling high-performance and flexible KV systems. In *Management of big data systems*, 2012.
[26] B. Recht, C. Re, S.J. Wright, and F. Niu. Hogwild: A lock-free approach to parallelizing stochastic gradient descent. *NIPS*, 2011.
[27] P. Richtárik and M. Takáč. Iteration complexity of randomized block-coordinate descent methods for minimizing a composite function. *Mathematical Programming*, 2012.
[28] A. Rowstron and P. Druschel. Pastry: Scalable, decentralized object location and routing for large-scale peer-to-peer systems. In *Distributed Systems Platforms*, 2001.
[29] A. J. Smola and S. Narayanamurthy. An architecture for parallel topic models. In *VLDB*, 2010.
[30] E. Sparks, A. Talwalkar, V. Smith, J. Kottalam, X. Pan, J. Gonzalez, M. J. Franklin, M. I. Jordan, and T. Kraska. MLI: An API for distributed machine learning. 2013.
[31] S. Sra. Scalable nonconvex inexact proximal splitting. In *NIPS*, 2012.
[32] I. Stoica, R. Morris, D. Karger, M. F. Kaashoek, and H. Balakrishnan. Chord: A scalable peer-to-peer lookup service for internet applications. *SIGCOMM Computer Communication Review*, 2001.
[33] C. Teflioudi, F. Makari, and R. Gemulla. Distributed matrix completion. In *ICDM*, 2012.
[34] C. H. Teo, S. V. N. Vishwanthan, A. J. Smola, and Q. V. Le. Bundle methods for regularized risk minimization. *JMLR*, January 2010.
[35] R. van Renesse and F. B. Schneider. Chain replication for supporting high throughput and availability. In *OSDI*, 2004.
[36] G. X. Yuan, K. W. Chang, C. J. Hsieh, and C. J. Lin. A comparison of optimization methods and software for large-scale l1-regularized linear classification. *JMLR*, 2010.
[37] M. Zaharia, M. Chowdhury, T. Das, A. Dave, J. Ma, M. Mccauley, M. J. Franklin, S. Shenker, and I. Stoica. Fast and interactive analytics over hadoop data with spark. *USENIX ;login:*, August 2012.
[38] M. Zinkevich, A. J. Smola, M. Weimer, and L. Li. Parallelized stochastic gradient descent. In *NIPS*, 2010.

## 附录A 延迟块近端梯度的收敛性证明

为证明定理2，我们需要几个技术引理。记 $b \subseteq \{1, \ldots, p\}$ 为一个坐标子集，并令 $x_b \in \mathbb{R}^p$ 为将 $x$ 中不在块 $b$ 中的条目设置为0后得到的向量。我们首先证明在假设1下，目标函数在子空间移动下表现良好。

**引理3** 假设块 $b$ 在时间 $t$ 被选中，则在假设1下，对于任意 $f_i$ 和任意时间 $t$ ，对于任意 $x, y \in \mathbb{R}^p$ ，有：

$$
f_i(x + y_b) \le f_i(x) + \langle \nabla f_i(x), y_b \rangle + \frac{L_{\text{var},i}}{2} \|y_b\|^2, \qquad (6)
$$

**证明。** 根据中值定理可得：

$$
f_i(x + y_b) = f_i(x) + \langle \nabla f_i(x + \xi y_b), y_b \rangle \quad \text{对某个 } \xi \in [0,1]. \qquad (7)
$$

使用假设1的 Lipschitz 性质，可得 $x + \xi y_b$ 处的梯度可以通过 $|\nabla f_i(x + \xi y_b) - \nabla f_i(x)| \le L_{\text{var},i} \xi \|y_b\|$ 来界定。将此与 $\xi \le 1$ 结合即证毕。

接下来我们证明，对于块可分正则化项，解也满足适当的分解性质：

**引理4** 假设 $h$ 是块可分的且 $0 \in \partial h(0)$ ，并且 $U$ 是对角的。对于任意 $x$ 和 $\gamma > 0$ ，记 $z = \operatorname{Prox}^{U}_{\gamma}(x)$ 和 $z_b = \operatorname{Prox}^{U}_{\gamma}(x_b)$ 分别为作用于全向量和仅作用于子集的近端算子的解。则对于任意块 $b$ ，下式成立：

$$
U (x_b - z_b) \in \gamma \partial h(z_b) \qquad (8)
$$

**证明。** 由于 $0 \in \partial h(0)$ ，可得 $\operatorname{Prox}_{\gamma}(0) = 0$ 。进一步由于 $h$ 是块可分的，邻近函数 $h(y) + \frac{1}{2\gamma} \|x - y\|^2_U$ 也是块可分的。通过将 $x$ 中除块 $b$ 外的所有条目设置为0，即得 $z_b = \operatorname{Prox}_{\gamma}(x_b)$ 。最后，(8) 通过对近端算子定义两边求导得到。

记 $\tilde{g}^{(t)}$ 和 $\tilde{u}^{(t)}$ 分别为服务器节点上的聚合梯度和缩放系数。假设每个工作机器以概率 $1 - q$ 随机跳过一个坐标，其中 $0 < q < 1$ 。令 $g^{(t)} := q^{-1} \tilde{g}^{(t)}$ 和 $u^{(t)} := q^{-1} \tilde{u}^{(t)}$ 分别为无偏不精确梯度和缩放系数估计（注意，也可以使用更复杂的子采样技术，如水库采样 reservoir sampling）。

下一步是利用更新 $\Delta^{(t)} = w^{(t+1)} - w^{(t)}$ 以及 $g^{(t)}$ 和 $\nabla f(w^{(t)})$ 之间的差异，来界定相邻迭代 $t$ 和 $t+1$ 之间目标函数的变化。

**引理5** 设 $g^{(t)}$ 为时间 $t$ 由服务器聚合的无偏不精确梯度。在定理2的假设下，有：

$$
\mathbb{E}\left[ F(w^{(t+1)}) - F(w^{(t)}) \right] \le \left( L_{\text{var}} - \frac{M_t}{\gamma_t} \right) \|\Delta^{(t)}\|^2 + \|\Delta^{(t)}\| \left\| \nabla_{b_t} f(w^{(t)}) - \mathbb{E}[g^{(t)}] \right\| \qquad (9)
$$

其中期望是关于随机跳过过滤器的。

**证明。** 为记号简洁，我们省略块指示符 $b_t$ 、缩放矩阵 $U^{(t)}$ 、学习率 $\gamma_t$ 和常数 $M_t$ 的下标 $t$ （回忆 $M_t = \min_i U^{(t)}_i$ 是由近端算子中的马氏度量引起的最小系数特定学习率）。

首先注意到 $g^{(t)}_b = g^{(t)}$ ，因为梯度是在块 $b$ 上计算的。因此更新 $\Delta^{(t)}$ 也局限于块 $b$ 。由引理4，我们有：

$$
\Delta^{(t)}_b = \operatorname{Prox}^{U}_{\gamma} \left[ w^{(t)}_b - \gamma U^{-1} g^{(t)} \right] - w^{(t)}_b = \Delta^{(t)}
$$

因此 $w^{(t+1)}_b = \operatorname{Prox}^{U}_{\gamma} (w^{(t)}_b - \gamma U^{-1} g^{(t)})$ 。再次使用引理4，我们有：

$$
\frac{U}{\gamma} \left( w^{(t)}_b - \gamma g^{(t)} - w^{(t+1)}_b \right) \in \partial h(w^{(t+1)}_b)
$$

由于 $h$ 是块可分的，我们可以分解更新得到：

$$
\begin{aligned}
h(w^{(t+1)}) - h(w^{(t)}) &= h(w^{(t+1)}_b) - h(w^{(t)}_b) \\
&\le \left\langle \frac{U}{\gamma} \left( w^{(t)}_b - \gamma U^{-1} g^{(t)} - w^{(t+1)}_b \right), w^{(t+1)}_b - w^{(t)}_b \right\rangle \\
&= -\frac{1}{\gamma} \|\Delta^{(t)}\|^2_U - \langle g^{(t)}, \Delta^{(t)} \rangle \\
&\le -\frac{M}{\gamma} \|\Delta^{(t)}\|^2 - \langle g^{(t)}, \Delta^{(t)} \rangle \qquad (10)
\end{aligned}
$$

另一方面，只有 $w^{(t+1)}$ 中块 $b$ 的条目相对于 $w^{(t)}$ 发生了改变，这满足了假设1的要求，因此，由引理3：

$$
\begin{aligned}
f(w^{(t+1)}) - f(w^{(t)}) &\le \left\langle w^{(t+1)} - w^{(t)}, \sum_{i=1}^m \nabla_b f_i(w^{(t)}) \right\rangle + \sum_{i=1}^m L_{\text{var},i} \|\Delta^{(t)}\|^2 \\
&= \langle \Delta^{(t)}, \nabla_b f(w^{(t)}) \rangle + L_{\text{var}} \|\Delta^{(t)}\|^2 \qquad (11)
\end{aligned}
$$

结合(10)和(11)，我们有：

$$
\begin{aligned}
\mathbb{E}\left[ F(w^{(t+1)}) - F(w^{(t)}) \right] &\le \left( L_{\text{var}} - \frac{M}{\gamma} \right) \|\Delta^{(t)}\|^2 + \mathbb{E}\left[ \langle \Delta^{(t)}, \nabla_b f(w^{(t)}) - g^{(t)} \rangle \right] \\
&\le \left( L_{\text{var}} - \frac{M}{\gamma} \right) \|\Delta^{(t)}\|^2 + \|\Delta^{(t)}\| \left\| \nabla_b f(w^{(t)}) - \mathbb{E}[g^{(t)}] \right\|
\end{aligned}
$$

换句话说，目标函数之间的变化量上界由参数变化量 $\Delta^{(t)}$ 和块梯度差异共同决定。

**定理2的证明。** 我们现在拥有证明收敛到驻点的所有要素。简而言之，我们必须界定 $\|\Delta^{(t)}\|$ ，其余一切随之而来。给定时间 $t$ ，记所选块 $b = b_t$ 。我们首先界定(9)中 $\|\nabla_b f(w^{(t)}) - \mathbb{E}[g^{(t)}]\|$ 项。由假设1，对于 $1 \le k \le \tau$ 有：

$$
\left\| \nabla_b f_i(w^{(t-k+1)}) - \nabla_b f_i(w^{(t-k)}) \right\| \le L_{\text{cov},i} \left\| w^{(t-k+1)} - w^{(t-k)} \right\| = L_{\text{cov},i} \left\| \Delta^{(t-k)} \right\|.
$$

由于有界延迟，工作机器 $i$ 的模型在时间 $t$ 仅在过去 $t - \tau \le t_i \le t$ 范围内过时。显著修改过滤器在模型上增加了一个额外的噪声项 $\sigma(t_i)$ 。根据我们使用的过滤器的设计， $\|\sigma(t_i)\|_{\infty} \le \delta_{t_i} = O\left( \frac{1}{t_i} \right)$ 。

此外，通过随机跳过过滤器，在时间 $t$ 聚合的无偏不精确梯度的期望由下式给出：

$$
\mathbb{E}[g^{(t)}] = \sum_{i=1}^m \nabla_b f_i(w^{(t_i)} + \sigma(t_i)).
$$

于是有：

$$
\begin{aligned}
&\left\| \nabla_b f(w^{(t)}) - \mathbb{E}[g^{(t)}] \right\| \\
&= \left\| \sum_{i=1}^m \sum_{k=1}^{t - t_i} \left( \nabla_b f_i(w^{(t-k+1)}) - \nabla_b f_i(w^{(t-k)}) \right) + \nabla_b f_i(w^{(t_i)}) - \nabla_b f_i(w^{(t_i)} + \sigma(t_i)) \right\| \\
&\le \sum_{i=1}^m \sum_{k=1}^{t - t_i} \left\| \nabla_b f_i(w^{(t-k+1)}) - \nabla_b f_i(w^{(t-k)}) \right\| + \left\| \nabla_b f_i(w^{(t_i)}) - \nabla_b f_i(w^{(t_i)} + \sigma(t_i)) \right\| \\
&\le \sum_{i=1}^m \sum_{k=1}^{t - t_i} L_{\text{cov},i} \left\| \Delta^{(t-k)} \right\| + L_{\text{cov},i} \left\| \sigma(t_i) \right\| \\
&\le \sum_{i=1}^m \sum_{k=1}^{\tau} L_{\text{cov},i} \left\| \Delta^{(t-k)} \right\| + L_{\text{cov},i} \sqrt{p} \delta_{t-\tau} \\
&= \sum_{k=1}^{\tau} L_{\text{cov}} \left\| \Delta^{(t-k)} \right\| + L_{\text{cov}} \sqrt{p} \delta_{t-\tau} \qquad (12)
\end{aligned}
$$

其中我们使用了 $\sigma(t_i) = \sigma(t_i)_{b_{t_i}}$ ，使得假设1适用，且 $\|x\| \le \sqrt{p} \|x\|_{\infty}$ 。

将(12)代入引理5的(9)中，我们有：

$$
\begin{aligned}
\mathbb{E}\left[ F(w^{(t+1)}) - F(w^{(t)}) \right] &\le \left( L_{\text{var}} - \frac{M_t}{\gamma_t} \right) \|\Delta^{(t)}\|^2 + \sum_{k=1}^{\tau} L_{\text{cov}} \|\Delta^{(t)}\| \left( \|\Delta^{(t-k)}\| + \sqrt{p} \delta_{t-\tau} \right) \\
&\le \left( L_{\text{var}} + \frac{L_{\text{cov}} \tau}{2} - \frac{M_t}{\gamma_t} \right) \|\Delta^{(t)}\|^2 + \sum_{k=1}^{\tau} \frac{L_{\text{cov}}}{2} \|\Delta^{(t-k)}\|^2 + L_{\text{cov}} p \delta^2_{t-\tau}
\end{aligned}
$$

对 $t$ 求和得到：

$$
\mathbb{E}\left[ F(w^{(T+1)}) - F(w^{(1)}) \right] \le \sum_{t=1}^T \left( L_{\text{var}} + L_{\text{cov}} \tau - \frac{M_t}{\gamma_t} \right) \|\Delta^{(t)}\|^2 + L_{\text{cov}} p \delta^2_{t-\tau} \qquad (13)
$$

记 $c_t = \frac{M_t}{\gamma_t} - L_{\text{var}} - L_{\text{cov}} \tau$ ，由于对所有 $t$ 有 $\gamma_t \le \frac{M_t}{L_{\text{var}} + L_{\text{cov}} \tau + \epsilon}$ ，则所有 $c_t \ge \epsilon > 0$ 。因此：

$$
\epsilon \sum_{t=1}^T \|\Delta^{(t)}\|^2 \le \sum_{t=0}^T c(t) \|\Delta^{(t)}\|^2 \le \mathbb{E}\left[ F(w^{(1)}) - F(w^{(T+1)}) \right] + L_{\text{cov}} p \delta^2_{t-\tau} \qquad (14)
$$

对于任意 $T$ 成立。由于 $\delta_t = O\left(\frac{1}{t}\right)$ ，且根据 $1 + \frac{1}{2^2} + \frac{1}{3^2} + \ldots = \frac{\pi^2}{6}$ 的事实，当 $T \to \infty$ 时(14)的右侧为常数，这意味着 $\lim_{t \to \infty} \Delta^{(t)} \to 0$ 。因此 $\lim_{t \to \infty} \operatorname{Prox}^{U_t}_{\gamma_t}(w^{(t)}) - w^{(t)} \to 0$ ，于是我们找到了一个局部极小点。

## 附录B 稀疏逻辑回归

除了第5节报告的最大规模实验外，我们还在以下一系列稀疏训练数据上展示了更多实验结果。URL 和 KDDa 是公开的稀疏文本数据集[2]，而点击率数据集 CTRa 和 CTRb 是从第5节使用的数据集中子采样得到的。

| 数据集 | URL | KDDa | CTRa | CTRb |
|---|---|---|---|---|
| # of examples | 2M | 8M | 4M | 0.34B |
| # of coordinates | 3M | 20M | 60M | 2.2B |
| # of nnz entries | 277M | 305M | 400M | 31B |

我们专注于目标值（objective value）和运行时间的收敛情况。更精确地说，我们报告每次数据传递（data pass）中通过 $F(w^{(t)}) / F(w^*) - 1$ 计算的相对目标值，该过程由若干迭代组成。最优值 $w^*$ 的估计是通过执行为收敛所需迭代次数的4倍得到的。

### B.1 与其他算法的比较

由于我们无法找到能够扩展到本研究数据集规模的其他分布式多机稀疏逻辑回归算法，我们将求解器与多核环境下可用的其他求解器进行了比较。这意味着我们将自己限制在相对较小的数据集上，仅包含数百万个观测值，如上表所述。

更具体地说，我们将求解器与在一台具有32个线程/工作机器的机器上运行的 Shotgun [7] 进行了比较。同时报告了 CDN（单线程 Shotgun）的结果以供参考。图7显示了目标值随时间的变化。可以看出，所有三种算法在50次数据传递后获得了相似的目标值，然而，参数服务器在运行时间上比 Shotgun 和 CDN 都快4倍。

 $^2$ www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets
 $^3$ www.select.cs.cmu.edu/projects/shotgun/
 $^4$ www.csie.ntu.edu.tw/~cjlin/liblinear

主要原因在于数据划分策略。Shotgun 的每个线程一次处理一个坐标，这通常具有不规则的非零条目模式，因此难以进行负载均衡。在最高维的数据集 CTRa 上，Shotgun 甚至比单线程版本还要慢。另一方面，参数服务器在训练数据的较大块上使用多线程线性代数算子。这种粗粒度的并行化带来了更好的加速效果。

$$
\begin{array}{c}
\text{Figure 7: Both shotgun and our algorithm use 32 threads on a single machine. Each point indicates} \\
\text{one pass through the data. In total 50 passes are shown.}
\end{array}
$$

### B.2 可扩展性

我们通过将工作机器数量从16增加到256来研究可扩展性。报告了与16个工作机器相比的运行时间加速比。当工作机器数量增加16倍时，观察到9倍的加速。

## 附录C 重建 ICA

### C.1 问题

重建 ICA（Reconstruction ICA）旨在寻找原始数据集的稀疏表示。它通过允许过完备解（overcomplete solution）来放松独立成分分析（Independent Component Analysis）[18]。记 $\{x_i\}_{i=1}^n \in \mathbb{R}^p$ 为观测值。RICA 的目标函数具有非凸的损失函数 $f(W)$ 和凸但非光滑的惩罚项 $h(W)$ ：

$$
\operatorname*{minimize}_{W \in \mathbb{R}^{\ell \times p}} \sum_{i=1}^n \frac{1}{2} \left\| W W^\top x_i - x_i \right\|_2^2 + \lambda \|W x_i\|_1, \qquad (15)
$$

记 $X = (x_1, \ldots, x_n)^\top \in \mathbb{R}^{n \times p}$ 为数据矩阵。光滑损失 $f$ 的梯度为：

$$
\nabla f(W) = W \left( (W^\top W - I) X^\top X + X^\top X (W^\top W - I) \right) \qquad (16)
$$

这可以通过使用一些迹恒等式重写目标函数 $f(W)$ 得到：

$$
\begin{aligned}
f(W) &= \frac{1}{2} \left\| X W^\top W - X \right\|_F^2 \\
&= \frac{1}{2} \operatorname{tr} \left( W^\top W X^\top X W^\top W - 2 X^\top X W^\top W + X^\top X \right) \quad (\|A\|_F^2 = \operatorname{tr}(A^\top A)) \\
&= \frac{1}{2} \operatorname{tr} \left( W^\top W X^\top X W^\top W \right) - \operatorname{tr} \left( X^\top X W^\top W \right) + \frac{1}{2} \operatorname{tr} \left( X^\top X \right) \\
&= \frac{1}{2} \operatorname{tr} \left( X W^\top W W^\top W X^\top \right) - \operatorname{tr} \left( W X^\top X W^\top \right) + \frac{1}{2} \operatorname{tr} \left( X^\top X \right) \quad (\operatorname{tr}(AB) = \operatorname{tr}(BA))
\end{aligned}
$$

直接应用[24]中的(112)和(100)，我们得到(16)。

不幸的是，在上述情况下调用近端算子并不简单，因为 $\|W X\|_1$ 具有不可分离的分量（但它是块可分的）。这意味着我们需要同时求解以下每个独立的优化问题（例如，使用 ADMM）：

$$
\operatorname*{minimize}_{u_i} \frac{1}{2\gamma} \|u_i - z_i\|^2_{H_i} + \lambda \|X u_i\|_1, \quad \text{对于 } i = 1 \ldots n \qquad (17)
$$

其中 $w_i \in \mathbb{R}^p$ 表示 $W$ 的第 $i$ 行，我们设 $z_i = w_i - \gamma H_i^{-1} \nabla_i f(W)$ 。这里 $\gamma$ 是学习率， $H_i \in \mathbb{R}^{d \times d}$ 是调整空间度量的缩放矩阵。遵循[10]，我们通过下式选择缩放矩阵：

$$
H_i(t+1)^2 = H_i(t)^2 + \operatorname{diag}\left( w(t)_i - w_i(t-1) \right)^2 \quad \text{对于 } t \ge 0
$$
且
$$
H_i(0) = 1,
$$

这可以在本地计算。为方便起见，我们从(17)的近端步骤中省略了下标 $i$ 。引入辅助变量 $y := X u$ 后，增广拉格朗日函数（augmented Lagrangian）为：

$$
\mathcal{L}(u, y, \mu) = \frac{1}{2\gamma} \|u - z\|^2_H + \lambda \|y\|_1 + \langle \mu, X u - y \rangle + \frac{1}{2\theta} \|X u - y\|^2. \qquad (18)
$$

相应地，我们得到更新规则：

$$
\begin{aligned}
u &\leftarrow \left( \gamma^{-1} H + \theta^{-1} X^\top X \right)^{-1} \left( \gamma^{-1} H z + X^\top (\theta^{-1} y - \mu) \right) \qquad (19a) \\
y &\leftarrow S_{\lambda} \left( \theta^{-1} X u + \mu \right) \qquad (19b) \\
\mu &\leftarrow \mu + \theta^{-1} (X u - y), \qquad (19c)
\end{aligned}
$$

其中 $S_{\lambda}(\cdot)$ 是软阈值函数（soft-thresholding function）。注意，如果工作机器拥有所有观测值和 $W^\top W$ （通常远小于 $W$ ），它可以独立更新其参数。因此，我们对 RICA 采用按参数划分的方式。服务器维护 $W^\top W$ ，而每个工作机器拥有 $X$ 以及 $W$ 的一部分行。换句话说，工作机器计算并保留部分参数空间。

### C.2 实验

由于大多数计算是密集矩阵运算，我们使用 CUBLAS 在 GPU 上实现了所提出的算法。后者默认使用 GPU 内的所有计算单元。

我们在每张 GPU 卡上运行一个工作机器。我们在一个集群上进行实验，每台机器配备一块 Nvidia Tesla K20。使用的数据集是 ImageNet，包含100,000张随机选择并调整为 $100 \times 100$ 像素的图像[5]。

$$
\begin{array}{c}
\text{Figure 8: We tested scalability of the algorithm when increasing the number of clients (for a fixed} \\
\text{number of servers) from 16 to 256. We achieve almost a perfect speedup. Much of the delay is} \\
\text{likely due to the increased network load for the servers.} \\
\text{Figure 9: Reconstruction ICA on dataset ImageNet. Left: Varying delays on 16 GPU machines.} \\
\text{Right: Decomposition of running times. Bottom: Scalability when increasing the number of workers} \\
\text{from 1 to 16.}
\end{array}
$$

实验结果如图9所示。与 ℓ1-正则化逻辑回归类似，异步性带来的明显改进同样可观察到。与前者不同，增加延迟对运行时间和收敛性的影响都很小。这是因为 RICA 的实际更新延迟通常为1。当工作机器数量增加16倍时，我们在图9中看到 RICA 有13.5倍的加速。RICA 的加速比优于 ℓ1-正则化逻辑回归的主要原因是 RICA 主要由密集矩阵运算组成。它们比稀疏矩阵更容易平衡，因此提供了更好的可扩展性。

 $^5$ www.image-net.org
