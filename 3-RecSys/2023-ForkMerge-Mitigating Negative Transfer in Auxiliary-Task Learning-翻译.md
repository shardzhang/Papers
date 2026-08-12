# ForkMerge：缓解辅助任务学习中的负迁移

> Junguang Jiang, Baixu Chen, Junwei Pan, Ximei Wang, Dapeng Liu, Jie Jiang, Mingsheng Long | 清华大学 & 腾讯

\*两位作者对本文贡献相等。



本文介绍了 ForkMerge：缓解辅助任务学习中的负迁移。核心内容：

- 从优化与泛化两个视角系统分析负迁移的成因：实验发现梯度冲突并不必然导致负迁移（例如权重衰减与目标任务存在梯度冲突，但仍有益于目标性能），而当多任务训练数据与目标测试数据之间的分布偏移扩大时，负迁移更可能发生
- 提出ForkMerge方法：将模型周期性地分叉为多个分支，以不同的任务权重在多样的数据分布上独立优化各分支参数，然后通过最小化目标验证误差自动搜索最优任务权重，并动态合并所有分支以过滤掉有害的参数更新
- 将"组合任务分布"的问题转化为"组合模型假设"的问题：通过最大化目标验证性能自动搜索分支参数的线性组合系数 $\Lambda^*$，支持在联合训练的同时自动选择辅助任务，并引入分支剪枝与贪心搜索策略降低计算复杂度

关键发现：

- ForkMerge在四个基准上均取得最佳性能：NYUv2上 $\Delta_m$ 达4.03%，DomainNet上达+2.00%，AliExpress大规模推荐数据集上以 $\Delta_m = 1.30\%$ 领先所有对比方法
- 在CIFAR-10/SVHN半监督学习（SSL，Semi-Supervised Learning）中，S4L+ForkMerge‡将测试误差降至13.1%和5.49%，$\Delta_m$ 高达46.3%
- 负迁移在任务间普遍存在（DomainNet上30个任务对中有23个发生负迁移），而ForkMerge在全部30个组合中成功避免了负迁移；且ForkMerge在不同骨干网络与多任务架构（ViT-Base、MTAN、MMoE）上均保持优势

---

## 摘要

辅助任务学习（ATL，Auxiliary-Task Learning）旨在通过利用从相关任务获取的知识来提高目标任务的性能。偶尔，同时学习多个任务会导致比仅学习目标任务更低的准确率，这被称为负迁移。这个问题通常归因于任务间的梯度冲突，并且在先前的工作中常常通过协调任务梯度来处理。然而，这些基于优化的方法在很大程度上忽略了辅助-目标泛化能力。为了更好地理解负迁移的根本原因，我们从优化和泛化两个角度对其进行了实验研究。基于我们的发现，我们引入了ForkMerge，一种新颖的方法，它周期性地将模型分叉成多个分支，通过最小化目标验证误差来自动搜索变化的任务权重，并动态合并所有分支以过滤掉有害的任务参数更新。在一系列辅助任务学习基准上，ForkMerge优于现有方法，并有效缓解了负迁移。

## 1 引言

深度神经网络在各种机器学习应用中取得了显著成功，例如计算机视觉[23, 22]、自然语言处理[62, 11, 57]和推荐系统[46]。然而，训练深度神经网络的一个主要挑战是标记数据的稀缺性。近年来，辅助任务学习（ATL，Auxiliary-Task Learning）已成为解决这一挑战的一种有前景的技术[67, 39, 43]。ATL通过利用一些相关辅助任务提供的有用信号来提高目标任务的泛化能力。例如，较大规模的任务（如用户点击预测）可以作为辅助任务来提高较小规模的目标任务（如推荐中的用户转化预测）的性能[47, 36]。无标记数据上的自监督任务可以作为辅助任务来提高计算机视觉和自然语言处理中目标任务的性能，而无需额外的标记数据[34, 69, 11, 3]。

然而，在实践中，同时学习多个任务有时会导致性能下降，相比于仅学习目标任务，这种现象被称为负迁移（NT，Negative Transfer）[84, 75]。即使在大语言模型中，负迁移问题可能仍然存在。例如，RLHF（Reinforcement Learning from Human Feedback，基于人类反馈的强化学习）[7]（ChatGPT [57] 的一个关键组件）在后训练GPT-4 [58] 时，在近一半的多选题任务上产生了负面效果。已经有大量方法被提出来缓解ATL中的负迁移[71, 79, 15, 39]。值得注意的是，先前研究将负迁移归因于优化困难，特别是不同任务之间的梯度冲突，并提出了通过减少任务梯度间的干扰来缓解负迁移的方法[79, 15]。其他工作专注于选择最相关的辅助任务，并通过避免具有严重任务冲突的任务组来减少负迁移[71, 17]。然而，尽管为解决负迁移付出了巨大努力，其根本原因仍未完全被理解。

在这方面，我们从优化和泛化的角度实验分析了ATL中负迁移的潜在原因。从优化的角度来看，我们的实验表明梯度冲突并不必然导致负迁移。例如，权重衰减（一种特殊的辅助任务）可能在梯度上与目标任务冲突，但仍然对目标性能有益。从泛化的角度来看，我们观察到当多任务训练数据与目标测试数据之间的分布偏移扩大时，负迁移更可能发生。

基于上述发现，我们提出了一种名为ForkMerge的新方法。由于我们无法预先知道哪种任务分布组合能带来更好的泛化，并且为每种可能的分布训练模型代价过高，我们将组合任务分布的问题转化为组合模型假设的问题。具体来说，我们将模型分叉成多个分支，通过改变任务权重在不同数据分布上优化不同分支的参数。然后每隔一定时间，我们合并并同步每个分支的参数以逼近最优模型假设。通过这种方式，我们将过滤掉有害的参数更新以缓解负迁移，并保留期望的参数更新以促进正迁移。

本文的贡献总结如下：(1) 我们系统地识别了问题并分析了ATL中负迁移的原因。(2) 我们提出了ForkMerge，一种缓解负迁移并提升ATL性能的新方法。(3) 我们进行了广泛的实验，验证了ForkMerge在一系列ATL基准上优于先前的方法。

## 2 相关工作

### 2.1 辅助任务学习

辅助任务学习（ATL，Auxiliary-Task Learning）通过利用来自相关辅助任务的知识来提高模型在目标任务上的性能。ATL中的两个主要挑战是选择合适的辅助任务以及将它们与目标任务联合优化。为了找到ATL的合适辅助任务，最近的研究通过将正相关任务分组在一起并将不相关任务分配到不同组以避免任务干扰来探索任务关系[81, 71, 17, 70]。一旦确定了辅助任务，大多数ATL方法通过线性组合目标和辅助损失来创建统一损失。然而，由于搜索空间随任务数量呈指数增长，选择任务权重具有挑战性，并且固定每个任务损失的权重可能导致负迁移[32]。最近的研究提出了各种自动选择任务权重的方法，例如使用单步或多步梯度相似性[15, 39, 9]，最小化基于表示的任务距离[2]或梯度差距[67]，使用参数化级联辅助网络[54]，或从讨价还价博弈的角度[66]。然而，这些方法主要处理引入辅助任务后的优化困难，可能忽略了泛化问题。

最近，AANG [10] 构建了一个新颖的辅助任务搜索空间，并采用优先考虑目标任务泛化的元学习技术来学习单步任务权重。这一并行的发现突出了目标任务泛化的重要性，我们进一步引入了多步任务权重以减少估计不确定性。另一个并行方法ColD Fusion [12] 探索了协作多任务学习，并提出融合每个贡献者的参数以构建共享模型。在本文中，我们进一步考虑了任务的多样性和任务关系的复杂性，并推导出一种从任务组合权重出发来组合模型参数的方法。

### 2.2 多任务学习

与ATL不同，多任务学习（MTL，Multi-Task Learning）旨在通过从共享表示中学习多个目标来提高所有任务的性能。为了促进信息共享并最小化任务冲突，已经设计了许多多任务架构，包括硬参数共享[30, 22, 24]和软参数共享[51, 64, 16, 46, 44, 48, 72]。另一条工作线旨在优化策略以减少任务冲突。诸如损失平衡和梯度平衡等方法提出通过各种标准来寻找合适的任务权重，例如任务不确定性[28]、任务损失幅度[44]、梯度范数[5]和梯度方向[79, 6, 40, 41, 25, 55]。

尽管MTL方法可以直接用于联合训练辅助任务和目标任务，但ATL中的非对称任务关系通常不在MTL的考虑范围内。

### 2.3 负迁移

负迁移（NT，Negative Transfer）是机器学习中广泛存在的现象，其中从源数据或模型迁移知识可能对目标学习器产生负面影响[63, 60, 27]。为了缓解负迁移，领域自适应方法设计了重要性采样或实例加权策略来优先处理相关源数据[75, 83]。微调方法通过抑制表示中不可迁移的频谱分量来过滤有害的预训练知识[4]。MTL方法使用梯度手术或任务权重来减少跨任务的梯度冲突[79, 76, 25, 42]。与先前工作不同，我们提出在训练过程中动态过滤有害的参数更新以缓解负迁移。此外，我们对ATL中负迁移的原因提供了深入的实验分析，这在该领域较为罕见，但对未来研究将有所帮助。

## 3 负迁移分析

**问题与符号。** 在本节中，我们假设目标任务 $T_{tgt}$ 和辅助任务 $T_{aux}$ 都已给定。目标是通过与辅助任务的联合训练找到在目标任务上实现更高性能的模型参数 $\theta$，

$$
\min_{\theta} \ \mathbb{E}_{T_{tgt}} L_{tgt}(\theta) + \lambda \mathbb{E}_{T_{aux}} L_{aux}(\theta) \qquad (1)
$$

其中 $L$ 是训练损失，$\lambda$ 是辅助任务与目标任务之间的相对权重超参数。我们的最终目标是 $\max_{\theta} [P(\theta)]$，其中 $P$ 是目标任务 $T_{tgt}$ 的相对性能度量，例如分类中的准确率。接下来我们定义迁移增益来衡量 $T_{aux}$ 对 $T_{tgt}$ 的影响。

**定义3.1（迁移增益，TG，Transfer Gain）。** 将由某个ATL算法 $A$ 获得的模型记为 $\theta_A(T_{tgt}, T_{aux}, \lambda)$，将目标任务上的单任务学习获得的模型记为 $\theta(T_{tgt})$。设 $P$ 为目标任务 $T_{tgt}$ 上的性能度量。则算法 $A$ 可以通过下式评估：

$$
TG(\lambda, A) = P(\theta_A(T_{tgt}, T_{aux}, \lambda)) - P(\theta(T_{tgt})) \qquad (2)
$$

超越先前关于负迁移（NT，Negative Transfer）的工作[75, 84]，我们进一步将ATL中的负迁移分为两类。

**定义3.2（弱负迁移，WNT，Weak Negative Transfer）。** 对于某个ATL算法 $A$，其权重超参数为 $\lambda$，如果 $TG(\lambda, A) < 0$，则发生弱负迁移。

**定义3.3（强负迁移，SNT，Strong Negative Transfer）。** 对于某个ATL算法 $A$，如果 $\max_{\lambda > 0} TG(\lambda, A) < 0$，则发生强负迁移。

**图1：弱负迁移（WNT）与强负迁移（SNT）。**

图1说明了弱负迁移和强负迁移之间的区别。最根本的区别在于，我们或许能够通过选择合适的权重超参数 $\lambda$ 来避免弱负迁移，但无法通过这种方式避免强负迁移。

接下来，我们将从两个不同的角度分析ATL中的负迁移：优化和泛化。我们在使用ImageNet预训练的ResNet-18 [23] 的多域图像识别数据集DomainNet [61] 上进行分析。具体来说，我们分别使用DomainNet中的Painting和Quickdraw任务作为目标任务来展示弱负迁移和强负迁移，并混合DomainNet中的所有其他任务作为辅助任务。我们将在附录C.3中详细阐述DomainNet数据集，并在附录B中提供详细的实验设计。

### 3.1 梯度冲突的影响

人们普遍认为不同任务之间的梯度冲突会导致优化困难[79, 40]，进而导致负迁移。梯度冲突的程度通常通过梯度余弦相似性（GCS，Gradient Cosine Similarity）来衡量[79, 76, 15]。

**定义3.4（梯度余弦相似性，GCS，Gradient Cosine Similarity）。** 设 $\phi_{ij}$ 为两个任务梯度 $g_i$ 和 $g_j$ 之间的夹角，则我们将梯度余弦相似性定义为 $\cos \phi_{ij}$，并且当 $\cos \phi_{ij} < 0$ 时梯度为冲突的。

在图2中，我们绘制了梯度余弦相似性与迁移增益之间的相关曲线。有些反直觉的是，我们观察到负迁移和梯度冲突并不强相关，并且当任务梯度高度一致时负迁移可能更严重。

**发现1.** 负迁移不一定由梯度冲突引起，梯度冲突也不一定导致负迁移。

这似乎与先前的工作[79, 15]相矛盾，原因在于先前工作主要考虑训练过程中的优化收敛，而在我们的实验中，我们进一步考虑了评估过程中的泛化（迁移增益在验证集上估计）。尽管辅助任务的冲突梯度会增加目标任务的训练损失并减慢其收敛速度[37]，但它也可能起到类似于正则化的作用[32]，减少目标任务的过拟合，从而降低其泛化误差。为证实我们的假设，我们用L2正则化替换辅助任务重复了上述实验，并观察到了类似的现象，如图2(c)-(d)所示，这表明ATL中的梯度冲突不一定是有害的，因为它可以作为一种适当的正则化。

**图2：梯度冲突的影响。** 不同 $\lambda$ 下迁移增益（TG）与梯度余弦相似性（GCS）之间的相关曲线。为了公平比较，每个数据点从训练过程中间的相同模型参数开始，并通过单步多任务梯度下降进行更新。 $P$ 和 $Q$ 分别是Painting和Quickdraw任务的缩写。

图2还表明ATL中的权重超参数 $\lambda$ 对负迁移有很大影响。合适的 $\lambda$ 不仅减少了负迁移，还促进了正迁移。

### 3.2 分布偏移的影响

接下来，我们将从泛化的角度分析负迁移。我们注意到调整 $\lambda$ 会改变模型正在拟合的数据分布。例如，当 $\lambda = 0$ 时，模型仅拟合目标任务的数据分布；当 $\lambda = 1$ 时，模型将拟合目标任务和辅助任务的插值分布。形式上，给定目标分布 $T_{tgt}$ 和辅助分布 $T_{aux}$，目标任务和辅助任务的插值分布为 $T_{inter}$，

$$
T_{inter} \sim (1 - Z)T_{tgt} + ZT_{aux}, \quad Z \sim \text{Bernoulli}\left(\frac{\lambda}{1 + \lambda}\right) \qquad (3)
$$

其中 $\lambda$ 是任务权重超参数。图3(a)使用t-SNE [74] 定量可视化了不同 $\lambda$ 下的分布偏移。

为了定量衡量ATL中的分布偏移，我们引入以下定义。遵循[53]的符号，我们考虑多类分类，其假设空间 $F$ 为评分函数 $f: X \times \mathcal{Y} \rightarrow \mathbb{R}$，其中 $f(x, y)$ 表示将 $x$ 预测为 $y$ 的置信度。

**定义3.5（置信度分数差异，CSD，Confidence Score Discrepancy）。** 给定评分函数假设 $F$，将分布 $D$ 上的最优假设记为 $f_D^*$，则由 $F$ 诱导的分布 $D$ 与 $D'$ 之间的置信度分数差异定义为

$$
d_F(D, D') \triangleq 1 - \mathbb{E}_{x \sim D'} \max_{y \in \mathcal{Y}} f_D^*(x, y) \qquad (4)
$$

训练数据和测试数据之间的置信度分数差异表示模型对测试数据的不确信程度，预计当数据偏移扩大时该值会增加[59, 50]。

**图3：分布偏移的影响。** (a) 不同 $\lambda$ 下训练分布和测试分布的可视化。 (b) 对于弱负迁移任务，随着 $\lambda$ 增加，置信度分数差异（CSD）先下降后上升，迁移增益（TG）先正后负。对于强负迁移任务，CSD单调增加，TG保持为负。

图3(b)表明了置信度分数差异与迁移增益之间的相关性。对于弱负迁移任务，当 $\lambda$ 最初增加时，引入的辅助任务将使训练分布向测试分布偏移，从而减少训练和测试数据之间的置信度分数差异，并提高目标任务的泛化能力。然而，当 $\lambda$ 继续增加时，分布偏移逐渐增大，最终导致负迁移。对于强负迁移任务，引入的辅助任务分布与目标任务分布之间存在较大差距。因此，增加 $\lambda$ 总是扩大置信度分数差异，并总是导致负迁移。总结来说，

**发现2.** 如果引入的辅助任务扩大了目标任务训练数据与测试数据之间的分布偏移，则负迁移可能发生。

## 4 方法

在第4.1节中，基于上述分析，我们将介绍当辅助任务确定时如何缓解负迁移。然后在第4.2节中，我们将进一步讨论如何使用所提出的方法同时选择合适的辅助任务并与目标任务联合优化它们。

### 4.1 ForkMerge

在本节中，我们假设辅助任务 $T_{aux}$ 已给定。当在训练步骤 $t$ 使用公式(1)更新参数 $\theta_t$ 时，我们有

$$
\theta_{t+1}(\lambda) = \theta_t - \eta(g_{tgt}(\theta_t) + \lambda g_{aux}(\theta_t)) \qquad (5)
$$

其中 $\eta$ 是学习率，$g_{tgt}$ 和 $g_{aux}$ 分别是根据 $L_{tgt}$ 和 $L_{aux}$ 计算的梯度。第3.1节揭示了只要 $\lambda$ 经过仔细调整，$g_{tgt}$ 和 $g_{aux}$ 之间的梯度冲突不一定导致负迁移，第3.2节表明负迁移与泛化相关。因此，我们提出根据目标验证性能 $\hat{P}$ 动态调整 $\lambda$ 以缓解负迁移：

$$
\max_{\lambda} \hat{P}(\theta_{t+1}) = \hat{P}(\theta_t - \eta(g_{tgt}(\theta_t) + \lambda g_{aux}(\theta_t))) \qquad (6)
$$

**图4：ForkMerge训练流程。** 模型参数将被分叉成两个分支，一个仅用目标任务损失优化，另一个联合训练，并每隔 $\Delta t$ 步定期合并。

**算法1：ForkMerge训练流程。**

$$
\begin{aligned}
&\textbf{输入：初始模型参数 } \theta_0 \text{，总迭代次数 } T \text{，间隔 } \Delta t \\
&\textbf{输出：最终模型参数 } \theta_T^* \text{，任务相关性 } \lambda^* \\
&\text{将模型分叉为 2 份 } \{\theta_b\}_{b=0}^{1} \\
&\textbf{for } b = 0 \textbf{ to } 1 \textbf{ do} \\
&\quad \theta_0^b \leftarrow \theta_0 \quad \triangleright \text{初始化} \\
&\textbf{end for} \\
&\textbf{while } t < T \textbf{ do} \\
&\quad \textbf{for } b = 0 \textbf{ to } 1 \textbf{ do} \quad \triangleright \text{独立更新} \\
&\quad\quad \textbf{for } t' = t \textbf{ to } t + \Delta t - 1 \textbf{ do} \\
&\quad\quad\quad \theta_{t'+1}^b = \theta_{t'}^b - \eta(g_{tgt}(\theta_{t'}^b) + b \cdot g_{aux}(\theta_{t'}^b)) \\
&\quad\quad \textbf{end for} \\
&\quad \textbf{end for} \\
&\quad \lambda^* \leftarrow \arg\max_{\lambda} \hat{P}\big((1 - \lambda)\theta_{t+\Delta t}^0 + \lambda\theta_{t+\Delta t}^1\big) \quad \triangleright \text{在验证集上搜索 } \lambda \\
&\quad \theta_{t+\Delta t}^* \leftarrow (1 - \lambda^*)\theta_{t+\Delta t}^0 + \lambda^*\theta_{t+\Delta t}^1 \quad \triangleright \text{合并参数} \\
&\quad \textbf{for } b = 0 \textbf{ to } 1 \textbf{ do} \\
&\quad\quad \theta_{t+\Delta t}^b \leftarrow \theta_{t+\Delta t}^* \quad \triangleright \text{同步参数} \\
&\quad \textbf{end for} \\
&\quad t \leftarrow t + \Delta t \\
&\textbf{end while}
\end{aligned}
$$

公式(6)是一个双层优化问题。一种常见的方法是先用验证集上一批数据的损失来近似 $\hat{P}$，然后使用一阶近似来求解 $\lambda$ [18, 43]。然而，这些在单步梯度下降内的近似给 $\lambda$ 的估计引入了大量噪声，也增加了过拟合验证集的风险。为了解决这些问题，我们首先将公式(6)等价重写为

$$
\max_{\lambda} \hat{P}\big((1 - \lambda)\theta_{t+1}(0) + \lambda\theta_{t+1}(1)\big) \qquad (7)
$$

其中 $\theta_{t+1}(0) = \theta_t - \eta g_{tgt}(\theta_t)$，$\theta_{t+1}(1) = \theta_t - \eta(g_{tgt}(\theta_t) + g_{aux}(\theta_t))$。证明见附录A.1。注意我们假设最优 $\lambda^*$ 满足 $0 \leq \lambda^* \leq 1$，这可以通过在必要时增加 $L_{aux}$ 的规模来保证。然而，公式(7)中性能 $\hat{P}$ 的准确估计仍然计算代价高昂且容易过拟合，因此我们将单步梯度扩展到 $\Delta t$ 步，

$$
\lambda^* = \arg\max_{\lambda} \hat{P}\big((1 - \lambda)\theta_{t+\Delta t}(0) + \lambda\theta_{t+\Delta t}(1)\big) \qquad (8)
$$

**算法。** 如图4和算法1所示，训练步骤 $t$ 的初始模型参数 $\theta_t$ 将首先被分叉成两个分支。第一个分支仅使用目标任务损失 $L_{tgt}$ 优化 $\Delta t$ 次迭代以获得 $\theta_{t+\Delta t}(0)$，而另一个分支将联合训练 $\Delta t$ 次迭代以获得 $\theta_{t+\Delta t}(1)$。然后我们将搜索最优 $\lambda^*$，它线性组合上述两组参数以最大化验证性能 $\hat{P}$。当联合训练分支发生弱负迁移时，我们可以选择0到1之间的合适 $\lambda^*$。当发生强负迁移时，我们可以简单地将 $\lambda^*$ 设为0。最后，新合并的参数 $\theta_{t+\Delta t}^* = (1 - \lambda^*)\theta_{t+\Delta t}(0) + \lambda^*\theta_{t+\Delta t}(1)$ 将进入新一轮，再次分叉成两个分支，重复优化过程 $\lceil T / \Delta t \rceil$ 次。

**讨论。** 与实践中广泛使用的网格搜索 $\lambda$ 相比，ForkMerge可以在训练期间通过变化的 $\lambda^*$ 动态地将知识从辅助任务迁移到目标任务。在计算成本方面，ForkMerge具有较低的复杂度，因为它只需要训练2个分支，而网格搜索的成本与要搜索的超参数数量成正比。

### 4.2 同时进行任务选择的ForkMerge

当有多个辅助任务可用时，我们可以简单地将所有辅助任务混合在一起形成一个单一的辅助任务。这种简单策略在大多数场景下实际上效果很好（见第5.2节），且计算成本低廉。然而，当需要进一步提高性能时，我们也可以动态地为每个辅助任务选择最优权重。形式上，使用多个辅助任务 $\{T_k\}_{k=1}^{K}$ 优化目标任务 $T_0$ 的模型时，目标为

$$
\min_{\theta} \ \mathbb{E}_{T_0} L_0(\theta) + \sum_{k=1}^{K} \lambda_k \mathbb{E}_{T_k} L_k(\theta) \qquad (9)
$$

其中 $\sum_{k=1}^{K} \lambda_k \leq 1$ 且 $\forall k, \lambda_k \geq 0$。使用梯度下降在训练步骤 $t$ 更新 $\theta_t$，我们有

$$
\theta_{t+1}(\lambda) = \theta_t - \eta \sum_{k=0}^{K} \lambda_k g_k(\theta_t) \qquad (10)
$$

其中 $\lambda_0 = 1$。给定 $K$ 个任务权重向量 $\{\omega_k\}_{k=0}^{K}$，满足 $\omega_k^i = \mathbf{1}[i = k \text{ or } i = 0]$，即 $\omega_k$ 的第 $k$ 维和第 $0$ 维为1，其余为0，以及一个满足以下条件的向量 $\Lambda$：

$$
\Lambda_k = \begin{cases} 1 - \sum_{i \neq 0} \lambda_i, & k = 0 \\ \lambda_k, & k \neq 0 \end{cases} \qquad (11)
$$

则优化公式(10)中的 $\lambda^*$ 等价于

$$
\Lambda^* = \arg\max_{\Lambda} \hat{P}\Big(\sum_{k=0}^{K} \Lambda_k \theta_{t+1}(\omega_k)\Big) \qquad (12)
$$

在公式(12)中，初始模型参数被分叉成 $K + 1$ 个分支，其中一个分支仅用目标任务优化，其他分支与一个辅助任务和目标任务联合优化。然后我们找到最优 $\Lambda^*$，它线性组合 $K + 1$ 组参数以最大化验证性能（见公式(12)的证明和附录A.2中的详细算法）。公式(12)的训练计算复杂度为 $O(K)$，远低于网格搜索的指数复杂度，但仍然相当大。受任务分组方法[71]中使用的早期停止近似的启发，我们可以剪枝掉 $\Lambda_k = 0$（强负迁移）的分叉分支，在早期合并步骤后只保留 $\Lambda$ 中具有最大 $K' < K$ 值的分支。这样，那些包含无关辅助任务的无用分支可以提前停止。此外，我们在算法3中引入了一种贪心搜索策略，以进一步降低网格搜索 $\Lambda$ 所有可能值时的计算复杂度。

最后，我们引入ForkMerge的通用形式。假设 $B$ 个候选分支具有任务权重向量 $\nu_b$（$b = 1, \ldots, B$），目标是优化 $\Lambda^*$：

$$
\Lambda^* = \arg\max_{\Lambda} \hat{P}\Big(\sum_{b=1}^{B} \Lambda_b \theta_{t+\Delta t}(\nu_b)\Big) \qquad (13)
$$

从泛化的角度来看，由不同 $\nu$ 构建的混合分布导致了与目标分布不同的数据偏移，然而我们无法预测哪个 $\nu$ 会带来更好的泛化。因此，我们将混合分布问题转化为混合假设问题[49]，并且在不同分布上训练的模型通过 $\Lambda^*$ 动态组合以逼近最优参数。这里，公式(12)是通过代入 $B = K + 1$ 和 $\nu_b^i = \mathbf{1}[i = b - 1 \text{ or } i = 0]$ 的一个特例。相比之下，公式(13)允许我们通过构建更高效的分支将先验知识引入ForkMerge，也为将ForkMerge与先前的任务分组方法[81, 71, 17]结合提供了可能性。公式(13)的详细算法见算法2。

## 5 实验

我们在各种设置下评估ForkMerge的有效性，包括多任务学习、多域学习和半监督学习。首先，在第5.1节中，我们展示了负迁移的普遍性并解释了ForkMerge如何缓解这一问题。在第5.2节中，我们考察了ForkMerge在联合训练辅助任务和目标任务时是否能缓解负迁移，并与其他方法进行了比较。在第5.3节中，我们进一步将ForkMerge用于同时进行任务选择。实验细节见附录C。我们将在附录D中提供额外的分析和比较实验。我们的方法和比较方法的代码库将在 https://github.com/thuml/ForkMerge 提供。

### 5.1 动机实验

负迁移在不同任务间普遍存在。在图5(a)中，我们可视化了DomainNet上30个任务对之间的迁移增益，其中辅助任务和目标任务权重相等，我们观察到在这种情况下负迁移很常见（30个组合中有23个导致负迁移）。此外，如定义3.2和3.3所述，是否发生负迁移与特定的ATL算法有关，在图5(b)中，我们观察到当我们使用ForkMerge算法时，所有30个组合中的负迁移都可以成功避免。这一观察进一步表明了任务分组方法[71, 17]的局限性，因为它们在任务之间使用等权重，可能会丢弃一些有用的辅助任务。

**图5：DomainNet上的负迁移。** 每个矩阵的行表示辅助任务，列表示目标任务。蓝色和红色单元格分别对应负迁移增益和正迁移增益。颜色越深表示影响越强。

假设混合是分布混合的近似。图6使用三元热图可视化了用不同任务权重优化25K迭代的一组三个模型（包括一个单任务模型和两个多任务模型）的线性组合。类似于弱负迁移任务Painting的分布混合（见图3），混合Painting和Painting+Real模型时的迁移增益先增加后减少。同样，类似于强负迁移任务Quickdraw的分布混合，混合Quickdraw和Quickdraw+Real模型时的迁移增益单调递减。此外，图6还表明了深度模型的一个良好特性：过参数化深度神经网络的损失曲面在收敛后表现得相当规范和光滑，这一点先前的工作[20, 35]也有提及，并为ForkMerge中的合并步骤提供了直观解释。

**图6：模型假设混合的三元热图。** 每个三角形顶点代表一个优化后的模型，例如P+R是与Painting和Real任务联合优化的模型。三角形内的每个点对应模型假设的混合，其热值衡量迁移增益（TG）。

### 5.2 使用ForkMerge进行联合优化

首先，我们仅将ForkMerge用于目标任务和辅助任务的联合训练。当数据集包含多个任务时，我们将所有任务混合在一起形成一个单一的辅助任务用于ForkMerge。然而对于比较的方法，我们仍然对不同任务进行区分以获得更好的性能。

具体来说，我们将ForkMerge与以下方法进行比较：(1) 单任务学习（STL，Single Task Learning）；(2) EW，为所有任务分配相等的权重；(3) GCS [15]，一种使用目标任务和辅助任务之间梯度相似性的ATL方法；(4) OL_AUX [39]，一种基于梯度内积调整损失权重的ATL方法；(5) ARML [67]，一种基于梯度差异调整损失权重的ATL方法；(6) Auto-$\lambda$ [43]，一种通过有限差分近似[18]估计损失权重的ATL方法；(7) Post-train，一种先在所有任务上预训练模型然后为每个任务单独微调的ATL方法；(8) UW [28]，基于任务不确定性调整权重；(9) DWA [44]，基于损失变化调整权重；(10) MGDA [65]，计算具有最小范数的梯度凸组合以平衡任务；(11) GradNorm [5]，将不同任务的梯度范数重新缩放到相同范围；(12) PCGrad [79]，消除冲突的梯度分量；(13) IMTL [41]，使用在任务梯度上具有相等投影的更新方向；(14) CAGrad [40]，优化跨任务的平均损失和最小衰减率；(15) NashMTL [55]，使用纳什议价解组合梯度。由于不同任务具有不同的评估指标，我们将使用 $\Delta_m$（定义见附录C.1）报告每种方法的平均每任务性能改进。

**辅助任务场景理解。** 我们在广泛使用的多任务场景理解数据集NYUv2 [68] 上进行评估，该数据集包含3个任务：13类语义分割、深度估计和表面法线预测。遵循[55]，我们使用636、159和654张图像分别用于训练、验证和测试。我们的实现基于LibMTL [38] 和MTAN [44]。结果如表1所示。负迁移在这个数据集上并不严重，分割和深度都受益于ATL，只有法线任务变差。在这种情况下，我们的方法仍然在所有任务上取得了显著改进。我们还发现Post-train在我们大多数的ATL实验中是一个强基线。它的缺点是在预训练阶段未能考虑任务关系，并且在微调过程中遭受灾难性遗忘。

**表1：NYUv2数据集上的性能。** 分割任务的指标为mIoU和Pix Acc（越高越好），深度任务的指标为Abs Err和Rel Err（越低越好），法线任务的指标为Mean角度距离（越低越好），$\Delta_m$ 为平均每任务性能改进（越高越好）。

| 方法 | 分割：mIoU↑ | 分割：Pix Acc↑ | 深度：Abs Err↓ | 深度：Rel Err↓ | 法线：Mean↓ | $\Delta_m$↑ |
| --- | --- | --- | --- | --- | --- | --- |
| STL | 51.42 | 74.14 | 41.74 | 17.37 | 22.82 | - |
| EW | 52.13 | 74.51 | 39.03 | 16.43 | 24.14 | 0.30% |
| UW | 52.51 | 74.72 | 39.15 | 16.56 | 23.99 | 0.63% |
| DWA | 52.10 | 74.45 | 39.26 | 16.57 | 24.12 | 0.07% |
| MGDA | 50.79 | 73.81 | 39.19 | 16.25 | 23.14 | 1.44% |
| GradNorm | 52.25 | 74.54 | 39.31 | 16.37 | 23.86 | 0.56% |
| PCGrad | 51.77 | 74.72 | 38.91 | 16.36 | 24.31 | 0.22% |
| IMTL | 52.24 | 74.73 | 39.46 | 15.92 | 23.25 | 2.10% |
| CAGrad | 52.04 | 74.25 | 39.06 | 16.30 | 23.39 | 1.41% |
| NashMTL | 51.73 | 74.10 | 39.55 | 16.50 | 23.21 | 1.11% |
| GCS | 52.67 | 74.59 | 39.72 | 16.64 | 24.10 | 0.09% |
| OL_AUX | 52.07 | 74.28 | 39.32 | 16.30 | 23.98 | 0.17% |
| ARML | 52.73 | 74.85 | 39.61 | 16.65 | 23.89 | 0.37% |
| Auto-$\lambda$ | 52.40 | 74.62 | 39.25 | 16.25 | 23.38 | 1.17% |
| Post-train | 52.08 | 74.86 | 39.58 | 16.77 | 22.98 | 1.49% |
| ForkMerge | 53.67 | 75.64 | 38.91 | 16.47 | 22.18 | 4.03% |

**辅助域图像识别。** 此外，我们在广泛使用的多域图像识别数据集DomainNet [61] 上进行评估，该数据集包含6个不同的视觉域，约60万张图像分布在345个类别中，任务差异体现在边缘分布上。我们的实现基于TLlib [26]。由于原始DomainNet没有提供单独的验证集，我们从测试集中随机拆分50%的数据作为验证集。结果如表2所示。DomainNet既包含正迁移任务（Clipart）、弱负迁移任务（Infograph、Painting、Real、Sketch），也包含强负迁移任务（Quickdraw）。当发生负迁移时，先前的ATL方法导致严重的性能下降，而我们的方法可以在其他情况下自动避免强负迁移并提高相对于STL的性能。

**表2：DomainNet数据集上的性能。** C、I、P、Q、R、S分别代表Clipart、Infograph、Painting、Quickdraw、Real、Sketch六个域，Avg为六个域的平均准确率，$\Delta_m$ 为平均每任务性能改进（越高越好）。

| 方法 | C | I | P | Q | R | S | Avg | $\Delta_m$↑ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| STL | 77.6 | 41.4 | 71.8 | 73.0 | 84.6 | 70.2 | 69.8 | - |
| EW | 78.0 | 38.1 | 67.2 | 50.8 | 77.1 | 67.0 | 63.0 | -9.62% |
| UW | 79.1 | 35.8 | 68.2 | 50.5 | 77.9 | 67.0 | 63.1 | -9.98% |
| DWA | 78.3 | 38.2 | 67.8 | 51.4 | 77.3 | 67.2 | 63.4 | -9.15% |
| MGDA | 78.1 | 37.2 | 69.2 | 51.0 | 80.0 | 67.3 | 63.8 | -8.80% |
| GradNorm | 78.4 | 38.9 | 69.4 | 52.9 | 79.0 | 67.7 | 64.4 | -7.68% |
| PCGrad | 78.3 | 38.0 | 68.2 | 50.4 | 77.4 | 67.3 | 63.3 | -9.32% |
| IMTL | 79.4 | 38.6 | 68.6 | 53.7 | 79.3 | 67.6 | 64.5 | -7.55% |
| CAGrad | 79.1 | 38.6 | 69.4 | 53.6 | 79.8 | 67.6 | 64.7 | -7.35% |
| NashMTL | 71.8 | 32.9 | 62.2 | 39.5 | 73.5 | 61.4 | 56.9 | -18.8% |
| GCS | 74.6 | 36.0 | 67.6 | 56.6 | 76.4 | 62.3 | 62.3 | -11.0% |
| OL_AUX | 68.2 | 33.5 | 65.3 | 54.1 | 76.3 | 60.9 | 59.7 | -14.8% |
| ARML | 75.6 | 36.8 | 67.8 | 52.4 | 77.6 | 64.2 | 62.4 | -10.7% |
| Auto-$\lambda$ | 78.3 | 37.8 | 70.2 | 56.3 | 79.7 | 67.1 | 64.9 | -7.18% |
| Post-train | 78.7 | 42.3 | 72.7 | 73.0 | 84.7 | 71.2 | 70.4 | +1.07% |
| ForkMerge | 79.9 | 42.7 | 73.5 | 73.0 | 85.2 | 72.0 | 71.1 | +2.00% |

### 5.3 同时使用ForkMerge进行任务选择

如第4.2节所述，当有多个候选辅助任务时，我们可以使用ForkMerge同时选择辅助任务并与目标任务联合训练，表示为ForkMerge‡。

**辅助任务场景理解。** 在NYUv2中，对于任何目标任务，我们有2个辅助任务，因此我们可以在公式(12)中构建3个具有不同任务权重的分支。通过这种方式，我们能够在合并步骤中通过为不同分支学习不同的 $\Lambda$ 来自适应地选择辅助任务。如表3所示，这种策略产生了更好的整体性能。

**表3：分支数量对NYUv2的影响。** $B$ 为分支数量，其余指标同表1。

| 方法 | $B$ | 分割：mIoU↑ | 分割：Pix Acc↑ | 深度：Abs Err↓ | 深度：Rel Err↓ | 法线：Mean↓ | $\Delta_m$↑ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| STL | 1 | 51.42 | 74.14 | 41.74 | 17.37 | 22.82 | - |
| EW | - | 52.13 | 74.51 | 39.03 | 16.43 | 24.14 | 0.30% |
| ForkMerge | 2 | 53.67 | 75.64 | 38.91 | 16.47 | 22.18 | 4.03% |
| ForkMerge‡ | 3 | 54.30 | 75.78 | 38.42 | 16.11 | 22.41 | 4.59% |

**辅助域图像识别。** 对于DomainNet中的任何目标任务，我们可以在公式(12)中构建多达6个具有不同任务权重的分支，这在计算上是昂贵的。如第4.2节所述，我们将在第一次合并步骤后剪枝分支以减少计算成本。表4揭示了剪枝策略的影响。随着分支数量的增加，辅助任务带来的增益会增加，而每个分支带来的增益会减少。因此，剪枝是在性能和效率之间实现更好平衡的有效策略。在实际中，当面对多个辅助任务时，用户可以灵活地调整分支数量以匹配其可用的计算资源。

**表4：分支数量对DomainNet的影响。** $B$ 为分支数量，C、I、P、Q、R、S分别代表六个域，$\Delta_m$ 为相对STL的平均每任务性能改进，$\Delta_m^{B-1}$ 为每增加一个分支带来的增量改进。

| 方法 | $B$ | C | I | P | Q | R | S | $\Delta_m$↑ | $\Delta_m^{B-1}$↑ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| STL | 1 | 77.6 | 41.4 | 71.8 | 73.0 | 84.6 | 70.2 | - | - |
| ForkMerge | 2 | 79.9 | 42.7 | 73.5 | 73.0 | 85.2 | 72.0 | 2.0% | 2.0% |
| ForkMerge‡ | 3 | 81.1 | 44.0 | 73.7 | 73.0 | 85.2 | 72.7 | 3.0% | 1.5% |
| ForkMerge‡ | 4 | 81.1 | 44.2 | 74.4 | 73.1 | 85.3 | 73.0 | 3.3% | 1.1% |
| ForkMerge‡ | 6 | 81.3 | 44.4 | 74.7 | 73.2 | 85.3 | 73.4 | 3.6% | 0.7% |

**CTR和CTCVR预测。** 我们在AliExpress数据集[36]上进行评估，这是一个来自工业界的推荐数据集，包含2个任务：CTR（点击率，Click-Through Rate）和CTCVR（点击转化率，Click-Through Conversion Rate），4个场景，超过1亿条记录。我们的实现基于MTReclib [85]。对于AliExpress中的任何目标任务，我们最多可以构建8个具有不同任务权重的分支，并在第一次合并步骤后剪枝到3个分支。结果如表5所示。注意，在这样的超大规模数据集上用辅助任务进行改进是相当困难的。尽管如此，ForkMerge仍然以 $\Delta_m = 1.30\%$ 实现了最佳性能。

**表5：AliExpress数据集上的性能。** ES、FR、NL、US分别代表西班牙、法国、荷兰、美国四个场景，Avg为CTR和CTCVR八个指标的平均值（AUC，Area Under the ROC Curve，ROC曲线下面积，越高越好），$\Delta_m$ 为平均每任务性能改进。

| 方法 | CTR：ES | CTR：FR | CTR：NL | CTR：US | CTCVR：ES | CTCVR：FR | CTCVR：NL | CTCVR：US | Avg | $\Delta_m$↑ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| STL | 0.7299 | 0.7316 | 0.7237 | 0.7077 | 0.8778 | 0.8682 | 0.8652 | 0.8659 | 0.7963 | - |
| EW | 0.7299 | 0.7300 | 0.7248 | 0.7008 | 0.8855 | 0.8516 | 0.8606 | 0.8618 | 0.7931 | -0.39% |
| UW | 0.7276 | 0.7235 | 0.7250 | 0.7048 | 0.8814 | 0.8709 | 0.8599 | 0.8793 | 0.7966 | +0.00% |
| DWA | 0.7317 | 0.7284 | 0.7297 | 0.7061 | 0.8663 | 0.8695 | 0.8696 | 0.8484 | 0.7937 | -0.28% |
| MGDA | 0.6985 | 0.6926 | 0.7000 | 0.6676 | 0.8215 | 0.8145 | 0.7978 | 0.7917 | 0.7480 | -5.94% |
| GradNorm | 0.7239 | 0.7178 | 0.7101 | 0.7035 | 0.8851 | 0.8671 | 0.8465 | 0.8685 | 0.7903 | -0.79% |
| PCGrad | 0.7209 | 0.7193 | 0.7199 | 0.6892 | 0.8563 | 0.8621 | 0.8479 | 0.8413 | 0.7821 | -1.76% |
| IMTL | 0.7203 | 0.7193 | 0.7268 | 0.6852 | 0.8472 | 0.8502 | 0.8481 | 0.8282 | 0.7782 | -2.20% |
| CAGrad | 0.7280 | 0.7271 | 0.7223 | 0.6996 | 0.8712 | 0.8650 | 0.8417 | 0.8648 | 0.7900 | -0.77% |
| NashMTL | 0.7229 | 0.7245 | 0.7272 | 0.6972 | 0.8562 | 0.8606 | 0.8667 | 0.8497 | 0.7881 | -1.00% |
| GCS | 0.7229 | 0.7245 | 0.7272 | 0.6972 | 0.8562 | 0.8606 | 0.8667 | 0.8497 | 0.7881 | -0.49% |
| OL_AUX | 0.7311 | 0.7211 | 0.7239 | 0.7050 | 0.8779 | 0.8651 | 0.8610 | 0.8727 | 0.7947 | +0.54% |
| ARML | 0.7278 | 0.7247 | 0.7236 | 0.7030 | 0.8780 | 0.8671 | 0.8678 | 0.8670 | 0.7949 | +0.55% |
| Auto-$\lambda$ | 0.7282 | 0.7282 | 0.7263 | 0.7114 | 0.8852 | 0.8646 | 0.8640 | 0.8750 | 0.7979 | +0.19% |
| Post-train | 0.7291 | 0.7227 | 0.7244 | 0.7086 | 0.8889 | 0.8808 | 0.8654 | 0.8613 | 0.7977 | +0.14% |
| ForkMerge‡ | 0.7402 | 0.7427 | 0.7416 | 0.7069 | 0.8928 | 0.8786 | 0.8753 | 0.8752 | 0.8067 | +1.30% |

**半监督学习（SSL，Semi-Supervised Learning）。** 我们还在两个SSL数据集CIFAR-10 [31] 和SVHN [56] 上进行了评估。遵循[67]，我们使用自监督半监督学习（S4L，Self-supervised Semi-supervised Learning）[82] 作为基线算法，并使用2个自监督任务Rotation [19] 和Exempler-MT [14] 作为辅助任务。表6展示了使用不同ATL方法的S4L的测试误差以及其他SSL方法，并表明ForkMerge持续优于比较的ATL方法。注意，本文的目标不是提出一种新颖或最先进的SSL方法。相反，我们发现一些SSL方法使用了ATL，并且辅助任务权重有很大影响（见表6中的网格搜索）。因此，我们使用ForkMerge来改进SSL背景下的辅助任务训练。

**表6：CIFAR-10和SVHN数据集上的性能（测试误差，越低越好）。** 括号内为文献引用，$\Delta_m$ 为相对STL的平均每任务性能改进。

| 方法 | CIFAR-10（4000个标签） | SVHN（1000个标签） | $\Delta_m$↑ |
| --- | --- | --- | --- |
| STL | 20.3 | 12.80 | - |
| Π-Model [33] | 16.4 | 7.19 | 31.5% |
| Mean Teacher [73] | 15.9 | 5.65 | 38.8% |
| VAT [52] | 13.9 | 5.63 | 43.8% |
| Pseudo-Label [34] | 17.8 | 7.62 | 26.4% |
| S4L + EW | 15.7 | 7.83 | 30.7% |
| S4L + GradNorm | 14.1 | 7.68 | 35.3% |
| S4L + GCS | 15.0 | 7.02 | 35.6% |
| S4L + OL_AUX | 16.1 | 7.82 | 29.8% |
| S4L + ARML | 13.7 | 5.89 | 43.2% |
| S4L + Auto-$\lambda$ | 14.2 | 6.14 | 41.0% |
| S4L + Post-train | 15.8 | 7.85 | 30.4% |
| S4L + 网格搜索 | 13.8 | 6.07 | 42.3% |
| S4L + ForkMerge‡ | 13.1 | 5.49 | 46.3% |

## 6 结论

已经提出了许多方法来缓解辅助任务学习中的负迁移，但仍然缺乏对负迁移原因的深入实验分析。在本文中，我们系统地深入研究了负迁移问题，并提出了ForkMerge，一种实现具有正迁移增益的辅助任务学习的方法。实验上，ForkMerge在四个不同的辅助任务学习基准上实现了最先进的准确率，同时计算效率高。我们认为将先前的任务分组方法与我们的辅助任务学习方法相结合是未来研究的一个有前景的方向。

## 致谢

我们要感谢许多同事，特别是Yuchen Zhang、Jialong Wu、Haoyu Ma、Yuhong Yang和Jincheng Zhong，感谢他们宝贵的讨论。本工作得到了国家重点研发计划（2020AAA0109201）、国家自然科学基金（62022050和62021002）以及北京新星计划（Z201100006820041）的支持。

## 参考文献

[1] Liang-Chieh Chen, Yukun Zhu, George Papandreou, Florian Schroff, and Hartwig Adam. Encoder-decoder with atrous separable convolution for semantic image segmentation. In ECCV, 2018.

[2] Shuxiao Chen, Koby Crammer, Hangfeng He, Dan Roth, and Weijie J Su. Weighted training for cross-task learning. In ICLR, 2022.

[3] Ting Chen, Simon Kornblith, Kevin Swersky, Mohammad Norouzi, and Geoffrey Hinton. Big self-supervised models are strong semi-supervised learners. In NeurIPS, 2020.

[4] Xinyang Chen, Sinan Wang, Bo Fu, Mingsheng Long, and Jianmin Wang. Catastrophic forgetting meets negative transfer: Batch spectral shrinkage for safe transfer learning. In NeurIPS, 2019.

[5] Zhao Chen, Vijay Badrinarayanan, Chen-Yu Lee, and Andrew Rabinovich. Gradnorm: Gradient normalization for adaptive loss balancing in deep multitask networks. In ICML, 2018.

[6] Zhao Chen, Jiquan Ngiam, Yanping Huang, Thang Luong, Henrik Kretzschmar, Yuning Chai, and Dragomir Anguelov. Just pick a sign: Optimizing deep multitask models with gradient sign dropout. In NeurIPS, 2020.

[7] Paul F Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep reinforcement learning from human preferences. In NeurIPS, 2017.

[8] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In CVPR, 2009.

[9] Lucio M Dery, Yann Dauphin, and David Grangier. Auxiliary task update decomposition: The good, the bad and the neutral. In ICLR, 2021.

[10] Lucio M Dery, Paul Michel, Mikhail Khodak, Graham Neubig, and Ameet Talwalkar. Aang: Automating auxiliary learning. In ICLR, 2023.

[11] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In NAACL, 2019.

[12] Shachar Don-Yehiya, Elad Venezian, Colin Raffel, Noam Slonim, Yoav Katz, and Leshem Choshen. Cold fusion: Collaborative descent for distributed multitask finetuning. arXiv preprint arXiv:2212.01378, 2022.

[13] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. In ICLR, 2020.

[14] Alexey Dosovitskiy, Jost Tobias Springenberg, Martin Riedmiller, and Thomas Brox. Discriminative unsupervised feature learning with convolutional neural networks. In NeurIPS, 2014.

[15] Yunshu Du, Wojciech M Czarnecki, Siddhant M Jayakumar, Mehrdad Farajtabar, Razvan Pascanu, and Balaji Lakshminarayanan. Adapting auxiliary losses using gradient similarity. arXiv preprint arXiv:1812.02224, 2018.

[16] Chrisantha Fernando, Dylan Banarse, Charles Blundell, Yori Zwols, David Ha, Andrei A. Rusu, Alexander Pritzel, and Daan Wierstra. Pathnet: Evolution channels gradient descent in super neural networks. CoRR, abs/1701.08734, 2017.

[17] Christopher Fifty, Ehsan Amid, Zhe Zhao, Tianhe Yu, Rohan Anil, and Chelsea Finn. Efficiently identifying task groupings for multi-task learning. In NeurIPS, 2021.

[18] Chelsea Finn, Pieter Abbeel, and Sergey Levine. Model-agnostic meta-learning for fast adaptation of deep networks. In ICML, 2017.

[19] Spyros Gidaris, Praveer Singh, and Nikos Komodakis. Unsupervised representation learning by predicting image rotations. In ICLR, 2018.

[20] Ian Goodfellow, Oriol Vinyals, and Andrew Saxe. Qualitatively characterizing neural network optimization problems. In ICLR, 2015.

[21] Priya Goyal, Piotr Dollár, Ross B. Girshick, Pieter Noordhuis, Lukasz Wesolowski, Aapo Kyrola, Andrew Tulloch, Yangqing Jia, and Kaiming He. Accurate, large minibatch SGD: training imagenet in 1 hour. arXiv preprint arXiv:1706.02677, 2017.

[22] Kaiming He, Georgia Gkioxari, Piotr Dollár, and Ross Girshick. Mask r-cnn. In ICCV, 2017.

[23] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In CVPR, 2016.

[24] Falk Heuer, Sven Mantowsky, Saqib Bukhari, and Georg Schneider. Multitask-centernet (mcn): Efficient and diverse multitask learning using an anchor free approach. In ICCV, 2021.

[25] Adrián Javaloy and Isabel Valera. Rotograd: Gradient homogenization in multitask learning. In ICLR, 2022.

[26] Junguang Jiang, Baixu Chen, Bo Fu, and Mingsheng Long. Transfer-learning-library. https://github.com/thuml/Transfer-Learning-Library, 2020.

[27] Junguang Jiang, Yang Shu, Jianmin Wang, and Mingsheng Long. Transferability in deep learning: A survey. arXiv preprint arXiv:2201.05867, 2022.

[28] Alex Kendall, Yarin Gal, and Roberto Cipolla. Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. In CVPR, 2018.

[29] Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In ICLR, 2015.

[30] Iasonas Kokkinos. Ubernet: Training a 'universal' convolutional neural network for low-, mid-, and high-level vision using diverse datasets and limited memory. In CVPR, 2017.

[31] Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. Technical report, University of Toronto, 2009.

[32] Vitaly Kurin, Alessandro De Palma, Ilya Kostrikov, Shimon Whiteson, and M Pawan Kumar. In defense of the unitary scalarization for deep multi-task learning. In NeurIPS, 2022.

[33] Samuli Laine and Timo Aila. Temporal ensembling for semi-supervised learning. In ICLR, 2017.

[34] Dong-Hyun Lee. Pseudo-label: The simple and efficient semi-supervised learning method for deep neural networks. In ICML, 2013.

[35] Hao Li, Zheng Xu, Gavin Taylor, Christoph Studer, and Tom Goldstein. Visualizing the loss landscape of neural nets. In NeurIPS, 2018.

[36] Pengcheng Li, Runze Li, Qing Da, An-Xiang Zeng, and Lijun Zhang. Improving multi-scenario learning to rank in e-commerce by exploiting task relationships in the label space. In CIKM, 2020.

[37] Baijiong Lin, YE Feiyang, Yu Zhang, and Ivor Tsang. Reasonable effectiveness of random weighting: A litmus test for multi-task learning. In TMLR, 2022.

[38] Baijiong Lin and Yu Zhang. LibMTL: A python library for multi-task learning. arXiv preprint arXiv:2203.14338, 2022.

[39] Xingyu Lin, Harjatin Baweja, George Kantor, and David Held. Adaptive auxiliary task weighting for reinforcement learning. In NeurIPS, 2019.

[40] Bo Liu, Xingchao Liu, Xiaojie Jin, Peter Stone, and Qiang Liu. Conflict-averse gradient descent for multi-task learning. In NeurIPS, 2021.

[41] L Liu, Y Li, Z Kuang, J Xue, Y Chen, W Yang, Q Liao, and Wayne Zhang. Towards impartial multi-task learning. In ICLR, 2021.

[42] Shengchao Liu, Yingyu Liang, and Anthony Gitter. Loss-balanced task weighting to reduce negative transfer in multi-task learning. In AAAI, 2019.

[43] Shikun Liu, Stephen James, Andrew J Davison, and Edward Johns. Auto-lambda: Disentangling dynamic task relationships. In TMLR, 2022.

[44] Shikun Liu, Edward Johns, and Andrew J Davison. End-to-end multi-task learning with attention. In CVPR, 2019.

[45] Ilya Loshchilov and Frank Hutter. SGDR: stochastic gradient descent with warm restarts. In ICLR, 2017.

[46] Jiaqi Ma, Zhe Zhao, Xinyang Yi, Jilin Chen, Lichan Hong, and Ed H Chi. Modeling task relationships in multi-task learning with multi-gate mixture-of-experts. In SIGKDD, 2018.

[47] Xiao Ma, Liqin Zhao, Guan Huang, Zhi Wang, Zelin Hu, Xiaoqiang Zhu, and Kai Gai. Entire space multi-task model: An effective approach for estimating post-click conversion rate. In SIGIR, 2018.

[48] Kevis-Kokitsi Maninis, Ilija Radosavovic, and Iasonas Kokkinos. Attentive single-tasking of multiple tasks. In CVPR, 2019.

[49] Yishay Mansour, Mehryar Mohri, and Afshin Rostamizadeh. Domain adaptation with multiple sources. In NIPS, 2008.

[50] Matthias Minderer, Josip Djolonga, Rob Romijnders, Frances Hubis, Xiaohua Zhai, Neil Houlsby, Dustin Tran, and Mario Lucic. Revisiting the calibration of modern neural networks. In NeurIPS, 2021.

[51] Ishan Misra, Abhinav Shrivastava, Abhinav Gupta, and Martial Hebert. Cross-stitch networks for multi-task learning. In CVPR, 2016.

[52] Takeru Miyato, Shin-ichi Maeda, Masanori Koyama, and Shin Ishii. Virtual adversarial training: a regularization method for supervised and semi-supervised learning. In TPAMI, 2018.

[53] Mehryar Mohri and Andres Muñoz Medina. New analysis and algorithm for learning with drifting distributions. In International Conference on Algorithmic Learning Theory, 2012.

[54] Aviv Navon, Idan Achituve, Haggai Maron, Gal Chechik, and Ethan Fetaya. Auxiliary learning by implicit differentiation. In ICLR, 2021.

[55] Aviv Navon, Aviv Shamsian, Idan Achituve, Haggai Maron, Kenji Kawaguchi, Gal Chechik, and Ethan Fetaya. Multi-task learning as a bargaining game. In ICML, 2022.

[56] Yuval Netzer, Tao Wang, Adam Coates, Alessandro Bissacco, Bo Wu, and Andrew Y Ng. Reading digits in natural images with unsupervised feature learning. In NeurIPS, 2011.

[57] OpenAI. Introducing chatgpt, 2022.

[58] OpenAI. Gpt-4 technical report, 2023.

[59] Yaniv Ovadia, Emily Fertig, Jie Ren, Zachary Nado, David Sculley, Sebastian Nowozin, Joshua Dillon, Balaji Lakshminarayanan, and Jasper Snoek. Can you trust your model's uncertainty? evaluating predictive uncertainty under dataset shift. In NeurIPS, 2019.

[60] Sinno Jialin Pan and Qiang Yang. A survey on transfer learning. In TKDE, 2010.

[61] Xingchao Peng, Qinxun Bai, Xide Xia, Zijun Huang, Kate Saenko, and Bo Wang. Moment matching for multi-source domain adaptation. ICCV, 2019.

[62] Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever. Improving language understanding by generative pre-training. Technical report, OpenAI, 2018.

[63] Michael T. Rosenstein. To transfer or not to transfer. In NeurIPS, 2005.

[64] Andrei A. Rusu, Neil C. Rabinowitz, Guillaume Desjardins, Hubert Soyer, James Kirkpatrick, Koray Kavukcuoglu, Razvan Pascanu, and Raia Hadsell. Progressive neural networks. CoRR, abs/1606.04671, 2016.

[65] Ozan Sener and Vladlen Koltun. Multi-task learning as multi-objective optimization. In NeurIPS, 2018.

[66] Aviv Shamsian, Aviv Navon, Neta Glazer, Kenji Kawaguchi, Gal Chechik, and Ethan Fetaya. Auxiliary learning as an asymmetric bargaining game. arXiv preprint arXiv:2301.13501, 2023.

[67] Baifeng Shi, Judy Hoffman, Kate Saenko, Trevor Darrell, and Huijuan Xu. Auxiliary task reweighting for minimum-data learning. In NeurIPS, 2020.

[68] Nathan Silberman, Derek Hoiem, Pushmeet Kohli, and Rob Fergus. Indoor segmentation and support inference from rgbd images. In ECCV, 2012.

[69] Kihyuk Sohn, David Berthelot, Chun-Liang Li, Zizhao Zhang, Nicholas Carlini, Ekin D Cubuk, Alex Kurakin, Han Zhang, and Colin Raffel. Fixmatch: Simplifying semi-supervised learning with consistency and confidence. In NeurIPS, 2020.

[70] Xiaozhuang Song, Shun Zheng, Wei Cao, James Yu, and Jiang Bian. Efficient and effective multi-task grouping via meta learning on task combinations. In NeurIPS, 2022.

[71] Trevor Standley, Amir Zamir, Dawn Chen, Leonidas J. Guibas, Jitendra Malik, and Silvio Savarese. Which tasks should be learned together in multi-task learning? In ICML, 2020.

[72] Hongyan Tang, Junning Liu, Ming Zhao, and Xudong Gong. Progressive layered extraction (ple): A novel multi-task learning (mtl) model for personalized recommendations. In RecSys, 2020.

[73] Antti Tarvainen and Harri Valpola. Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results. In NeurIPS, 2017.

[74] Laurens Van der Maaten and Geoffrey Hinton. Visualizing data using t-sne. In JMLR, 2008.

[75] Zirui Wang, Zihang Dai, Barnabás Póczos, and Jaime Carbonell. Characterizing and avoiding negative transfer. In CVPR, 2019.

[76] Zirui Wang, Yulia Tsvetkov, Orhan Firat, and Yuan Cao. Gradient vaccine: Investigating and improving multi-task optimization in massively multilingual models. In ICLR, 2021.

[77] Derrick Xin, Behrooz Ghorbani, Justin Gilmer, Ankush Garg, and Orhan Firat. Do current multi-task optimization methods in deep learning even help? In NeurIPS, 2022.

[78] Yang You, Jing Li, Jonathan Hseu, Xiaodan Song, James Demmel, and Cho-Jui Hsieh. Reducing BERT pre-training time from 3 days to 76 minutes. In ICLR, 2020.

[79] Tianhe Yu, Saurabh Kumar, Abhishek Gupta, Sergey Levine, Karol Hausman, and Chelsea Finn. Gradient surgery for multi-task learning. In NeurIPS, 2020.

[80] Sergey Zagoruyko and Nikos Komodakis. Wide residual networks. In BMVC, 2016.

[81] Amir Roshan Zamir, Alexander Sax, William B. Shen, Leonidas J. Guibas, Jitendra Malik, and Silvio Savarese. Taskonomy: Disentangling task transfer learning. In CVPR, 2018.

[82] Xiaohua Zhai, Avital Oliver, Alexander Kolesnikov, and Lucas Beyer. S4l: Self-supervised semi-supervised learning. In ICCV, 2019.

[83] Jing Zhang, Zewei Ding, Wanqing Li, and Philip Ogunbona. Importance weighted adversarial nets for partial domain adaptation. In CVPR, 2018.

[84] Wen Zhang, Lingfei Deng, Lei Zhang, and Dongrui Wu. A survey on negative transfer. IEEE/CAA Journal of Automatica Sinica, 2022.

[85] Yongchun Zhu, Yudan Liu, Ruobing Xie, Fuzhen Zhuang, Xiaobo Hao, Kaikai Ge, Xu Zhang, Leyu Lin, and Juan Cao. Learning to expand audience via meta hybrid experts and critics for recommendation and advertising. In KDD, 2021.

## A 算法细节

### A.1 ForkMerge

**公式(7)的证明。**

$$
\begin{aligned}
\lambda^* &= \arg\max_{\lambda} \hat{P}(\theta_{t+1}) \\
&= \arg\max_{\lambda} \hat{P}\big(\theta_t - \eta(g_{tgt}(\theta_t) + \lambda g_{aux}(\theta_t))\big) \\
&= \arg\max_{\lambda} \hat{P}\big((\theta_t - \eta g_{tgt}(\theta_t)) + \lambda(-\eta g_{aux}(\theta_t))\big) \\
&= \arg\max_{\lambda} \hat{P}\big((1 - \lambda)(\theta_t - \eta g_{tgt}(\theta_t)) + \lambda(\theta_t - \eta(g_{tgt}(\theta_t) + g_{aux}(\theta_t)))\big) \\
&= \arg\max_{\lambda} \hat{P}\big((1 - \lambda)\theta_{t+1}(0) + \lambda\theta_{t+1}(1)\big)
\end{aligned}
$$

**关于搜索步骤的说明。** 我们提供以下两种搜索策略，在我们的实验中我们使用第一种策略。

- **网格搜索：** 通过手动指定的超参数空间子集彻底搜索任务权重超参数 $\lambda$，例如 $\{0, 0.2, 0.4, 0.6, 0.8, 1.0\}$。
- **二分搜索：** 将 $\lambda$ 的搜索区间重复对半分割并保留更好的超参数。

随机搜索、贝叶斯优化、基于梯度的优化和其他超参数优化方法也可以在此使用，留待后续工作探索。

在实践中，搜索步骤中估计 $\hat{P}$ 的成本通常可以忽略不计。然而当验证集中的数据量相对较大时，我们可以对验证集进行采样以减少估计 $\hat{P}$ 的成本。

**关于从单步梯度扩展到 $\Delta t$ 步的说明。**

1. 它可以有效降低每步估计 $\hat{P}$ 的平均成本，并避免过拟合验证集。
2. 它允许来自辅助任务的更长周期奖励，并导致更安全的任务迁移。例如，当某些辅助任务的累积梯度对最终目标性能有害时，合并步骤可以通过将其相关权重 $\lambda$ 设为0来抵消这些辅助任务的影响，以缓解强负迁移。
3. 它增加了产生不良模型参数的风险。然而，这种风险仍然很低，因为如第5.1节所示，深度模型在收敛后通常具有光滑的损失曲面。

**图7：合并步长 $\Delta t$ 对NYUv2的影响。**

图7说明了适当的 $\Delta t$ 可以有效提升ForkMerge算法的性能，表明了从先前工作中的单步梯度扩展到 $\Delta t$ 步的必要性。当 $\Delta t$ 较小时，$\lambda$ 的估计是短视的，可能无法在发生负迁移时移除有害的参数更新，这也表明了使用单步梯度下降估计 $\lambda$ 的方法[15, 43]的局限性。当 $\Delta t$ 较大时，从线性组合中获得不良模型参数的风险也会增加。因此，在我们的实验中，我们使用验证集为每个数据集选择合适的 $\Delta t$，并将其用于该数据集中的所有任务。

### A.2 使用ForkMerge同时选择任务

**详细算法。** 算法2提供了任何任务权重向量 $\{\nu_b\}_{b=1}^{B}$ 的通用优化过程。对于公式(12)，我们有 $B = K + 1$ 且 $\nu_b^i = \mathbf{1}[i = b - 1 \text{ or } i = 0]$。对于公式(13)，我们对 $B$ 或 $\nu_b$ 没有约束。

**算法2：多分支ForkMerge训练流程。**

$$
\begin{aligned}
&\textbf{输入：初始模型参数 } \theta_0 \text{，任务权重向量 } \{\nu_b\}_{b=1}^{B} \text{，总迭代次数 } T \text{，间隔 } \Delta t \\
&\textbf{输出：最终模型参数 } \theta_T^* \text{，任务相关性 } \Lambda^* \\
&\text{将模型分叉为 } B \text{ 份 } \{\theta_b\}_{b=1}^{B} \\
&\textbf{for } b = 1 \textbf{ to } B \textbf{ do} \\
&\quad \theta_0^b \leftarrow \theta_0 \\
&\textbf{end for} \\
&\textbf{while } t < T \textbf{ do} \\
&\quad \textbf{for } b = 1 \textbf{ to } B \textbf{ do} \\
&\quad\quad \textbf{for } t' = t \textbf{ to } t + \Delta t - 1 \textbf{ do} \\
&\quad\quad\quad \theta_{t'+1}^b = \theta_{t'}^b - \eta \sum_k \nu_b^k g_k(\theta_{t'}^b) \\
&\quad\quad \textbf{end for} \\
&\quad \textbf{end for} \\
&\quad \Lambda^* \leftarrow \arg\max_{\Lambda} \hat{P}\Big(\sum_b \Lambda_b \theta_{t+\Delta t}^b\Big) \\
&\quad \theta_{t+\Delta t}^* \leftarrow \sum_b \Lambda_b^* \theta_{t+\Delta t}^b \\
&\quad \textbf{for } b = 1 \textbf{ to } B \textbf{ do} \\
&\quad\quad \theta_{t+\Delta t}^b \leftarrow \theta_{t+\Delta t}^* \\
&\quad \textbf{end for} \\
&\quad t \leftarrow t + \Delta t \\
&\textbf{end while}
\end{aligned}
$$

**公式(12)的证明。**

在公式(10)中选择 $\lambda^*$ 的目标是最大化模型 $\theta_{t+1}$ 的验证性能，

$$
\begin{aligned}
\lambda^* &= \arg\max_{\lambda} \hat{P}(\theta_{t+1}) \\
&= \arg\max_{\lambda} \hat{P}\Big(\theta_t - \eta \sum_k \lambda_k g_k(\theta_t)\Big) \\
&= \arg\max_{\lambda} \hat{P}\Big(\theta_t - \eta \lambda_0 g_0(\theta_t) - \eta \sum_{k \neq 0} \lambda_k g_k(\theta_t)\Big) \\
&= \arg\max_{\lambda} \hat{P}\Big(\theta_t - \eta g_0(\theta_t) - \eta \sum_{k \neq 0} \lambda_k g_k(\theta_t)\Big) \quad \text{// } \lambda_0 = 1 \\
&= \arg\max_{\lambda} \hat{P}\Big((1 - \sum_{k \neq 0} \lambda_k)(\theta_t - \eta g_0(\theta_t)) + \sum_{k \neq 0} \lambda_k(\theta_t - \eta g_0(\theta_t) - \eta g_k(\theta_t))\Big) \quad \text{// 梯度下降，} \sum_{k \neq 0} \lambda_k \leq 1
\end{aligned}
$$

根据 $\Lambda$ 和 $\{\omega_k\}_{k=0}^{K}$ 的定义

$$
\Lambda_k = \begin{cases} 1 - \sum_{i \neq 0} \lambda_i, & k = 0 \\ \lambda_k, & k \neq 0 \end{cases} \qquad \omega_k^i = \begin{cases} 1, & i = 0 \text{ or } i = k \\ 0, & \text{otherwise} \end{cases}
$$

我们可以证明，在公式(10)中优化 $\lambda$ 等价于如下优化 $\Lambda$：

$$
\Lambda^* = \arg\max_{\Lambda} \hat{P}\Big(\sum_k \Lambda_k \theta_{t+1}(\omega_k)\Big)
$$

**关于搜索步骤的说明。** 网格搜索 $\Lambda$ 的所有可能值在计算上是昂贵的，特别是当 $\lVert \Lambda \rVert$ 很大时。因此，我们在算法3中引入了一种贪心搜索策略，将计算复杂度从指数复杂度降低到 $O(\lVert \Lambda \rVert)$。

**算法3：$\Lambda^*$ 的贪心搜索。**

$$
\begin{aligned}
&\textbf{输入：按 } \hat{P}(\theta_b) \text{ 降序排列的模型参数列表 } \theta_1, \dots, \theta_B \\
&\textbf{输出：最优线性组合系数 } \Lambda^* \\
&\text{未归一化的组合系数 } \tilde{\Lambda} \leftarrow e_1 \\
&\textbf{for } b = 2 \textbf{ to } B \textbf{ do} \\
&\quad \text{设置上界 } U \leftarrow 1 \\
&\quad \text{在范围 } [0, U] \text{ 内网格搜索最优 } \tilde{\Lambda}_m \text{，以最大化 } \hat{P}\Big(\frac{1}{\lVert \tilde{\Lambda} \rVert} \sum_{m=1}^{b} \tilde{\Lambda}_m \theta_m\Big) \\
&\textbf{end for} \\
&\Lambda^* \leftarrow \frac{1}{\lVert \tilde{\Lambda} \rVert} \tilde{\Lambda}
\end{aligned}
$$

## B 分析细节

在本节中，我们提供第3节中分析实验的实现细节。

我们在多域图像识别数据集DomainNet [61] 上进行分析。在我们的分析中，我们使用DomainNet中的Painting和Quickdraw任务作为弱负迁移和强负迁移的例子，并使用DomainNet中的其他任务（Real、Sketch、Infograph、Clipart）作为辅助任务。这些任务的细节总结在表8中。我们在所有实验中使用在ImageNet [8] 上预训练的ResNet-18 [23]。

### B.1 梯度冲突的影响

首先，我们在目标任务上优化模型 $T = 25K$ 次迭代以获得 $\theta_T$。我们采用动量为0.9、批量大小为48的小批量SGD（Stochastic Gradient Descent，随机梯度下降），初始学习率设为0.01，并采用余弦退火策略[45]。

**图8：梯度余弦相似性（GCS）的分布。** $P$ 和 $Q$ 分别是Painting和Quickdraw任务的缩写。

我们重复采样一小批数据并估计目标任务和辅助任务的梯度 $g_{tgt}$ 和 $g_{aux}$。图8绘制了 $g_{tgt}$ 和 $g_{aux}$ 之间的梯度余弦相似性（GCS）分布。我们发现，在大多数情况下不同任务的梯度几乎是正交的（$\cos \phi_{ij} \approx 0$），高度一致的梯度或严重冲突的梯度都相对罕见。

然后，我们使用从不同数据估计的单步多任务梯度下降优化相同的 $\theta_T$，以获得不同的 $\theta_{T+1}$，

$$
\theta_{T+1}(\lambda) = \theta_T - \eta(g_{tgt}(\theta_T) + \lambda g_{aux}(\theta_T)) \qquad (14)
$$

其中 $\eta = 0.01$，$\lambda$ 取值 $\{0, 1/16, 1/8, 1/4, 1/2, 1\}$。我们在目标任务的验证集上评估 $\theta_{T+1}(\lambda)$ 和 $\theta_{T+1}(0)$，以计算来自单步多任务梯度下降的迁移增益（TG）：

$$
TG(\lambda) = \hat{P}(\theta_{T+1}(\lambda)) - \hat{P}(\theta_{T+1}(0)) \qquad (15)
$$

注意，为简洁起见，我们在公式(14)和(15)中省略了算法 $A$ 的符号。然后，在图2中，我们标记每个数据点的GCS和TG，并用3阶多项式拟合它们以获得相应的相关曲线。

### B.2 分布偏移的影响

**定性可视化。** 我们使用t-SNE [74] 在图3(a)中可视化了由第B.1节训练的模型 $\theta_T$ 对训练和测试数据的表示。为了更好的可视化，我们仅保留DomainNet中频率最高的前10个类别。为了可视化 $\lambda$ 对插值训练分布的影响，我们让辅助任务点的频率与 $\lambda$ 成比例。换句话说，当辅助任务的权重超参数增加时，辅助任务对插值分布的影响也会增加。

图3提供了当 $\lambda$ 取值 $\{0, 1/16, 1/4, 1\}$ 时训练和测试分布的t-SNE可视化。我们观察到，对于弱负迁移任务，当 $\lambda$ 最初增加时，训练分布区域可以更好地覆盖测试分布区域。但随着 $\lambda$ 继续增加，测试集和训练集之间的分布偏移将逐渐增大。然而对于强负迁移任务，插值训练分布与测试分布之间的偏移随着 $\lambda$ 的增加单调增加。

**定量度量。** 首先，我们使用不同的权重超参数 $\lambda$ 在目标任务和辅助任务上联合优化模型 $T = 25K$ 次迭代以获得 $\theta_T(\lambda)$。我们采用与第B.1节相同的超参数。然后我们在目标任务的测试集上评估 $\theta_T(\lambda)$ 并计算测试集上的平均置信度。我们可以通过定义3.5计算置信度分数差异（CSD），并通过以下公式计算迁移增益（TG）：

$$
TG(\lambda) = \hat{P}(\theta_T(\lambda)) - \hat{P}(\theta_T(0)) \qquad (16)
$$

同样，为简洁起见我们省略了算法 $A$ 的符号。最后，我们在图3(b)中绘制了不同 $\lambda$ 下CSD和TG之间的曲线。

## C 实验细节

### C.1 $\Delta_m$ 的定义

遵循[55, 37]，我们报告 $\Delta_m$ 作为性能度量，这是方法 $m$ 相对于STL基线 $b$ 的平均每任务性能改进。形式上，$\Delta_m = \frac{1}{K} \sum_{k=1}^{K} (-1)^{z_k} \frac{M_{m,k} - M_{b,k}}{M_{b,k}}$，其中 $M_{b,k}$ 和 $M_{m,k}$ 分别是基线方法 $b$ 和比较方法 $m$ 获得的第 $k$ 个任务的性能。如果第 $k$ 个任务的较高值表示更好的性能，则 $z_k$ 设为0，否则为1。

### C.2 NYU上的辅助任务场景理解

**实验细节。** 我们使用DeepLabV3+架构[1]，其中在ImageNet数据集[8]上使用扩张卷积预训练的ResNet-50网络[23]作为任务间的共享编码器，并使用Atrous Spatial Pyramid Pooling模块作为每个任务的任务特定头部。遵循[44, 79]，每种方法使用Adam优化器[29]和批量大小8训练200个epoch。初始学习率为 $10^{-4}$，在100个epoch后减半为 $5 \times 10^{-5}$。在ForkMerge中，参数每10个epoch合并一次。表7展示了表1的完整评估结果。

**表7：NYUv2数据集上的性能（表1的完整评估结果）。** 分割任务的指标为mIoU和Pix Acc（越高越好），深度任务的指标为Abs Err和Rel Err（越低越好），法线任务的角度距离指标为Mean和Median（越低越好）以及Within $t^{\circ}$ 11.25、22.5、30（越高越好），$\Delta_m$ 为平均每任务性能改进（越高越好）。

| 方法 | 分割：mIoU↑ | 分割：Pix Acc↑ | 深度：Abs Err↓ | 深度：Rel Err↓ | 法线：Mean↓ | 法线：Median↓ | 法线：11.25↑ | 法线：22.5↑ | 法线：30↑ | $\Delta_m$↑ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| STL | 51.42 | 74.14 | 41.74 | 17.37 | 22.82 | 16.23 | 36.58 | 62.75 | 73.52 | - |
| EW | 52.13 | 74.51 | 39.03 | 16.43 | 24.14 | 17.62 | 33.98 | 59.63 | 70.93 | 0.30% |
| UW | 52.51 | 74.72 | 39.15 | 16.56 | 23.99 | 17.36 | 34.46 | 60.13 | 71.32 | 0.63% |
| DWA | 52.10 | 74.45 | 39.26 | 16.57 | 24.12 | 17.62 | 33.88 | 59.72 | 71.08 | 0.07% |
| RLW | 52.88 | 74.99 | 39.75 | 16.67 | 23.83 | 17.23 | 34.76 | 60.42 | 71.50 | 0.66% |
| MGDA | 50.79 | 73.81 | 39.19 | 16.25 | 23.14 | 16.46 | 36.15 | 62.17 | 72.97 | 1.44% |
| GradNorm | 52.25 | 74.54 | 39.31 | 16.37 | 23.86 | 17.46 | 34.13 | 60.09 | 71.45 | 0.56% |
| PCGrad | 51.77 | 74.72 | 38.91 | 16.36 | 24.31 | 17.66 | 33.93 | 59.43 | 70.62 | 0.22% |
| IMTL | 52.24 | 74.73 | 39.46 | 15.92 | 23.25 | 16.64 | 35.86 | 61.81 | 72.73 | 2.10% |
| GradVac | 52.84 | 74.77 | 39.48 | 16.28 | 24.00 | 17.49 | 34.21 | 59.94 | 71.26 | 0.75% |
| CAGrad | 52.04 | 74.25 | 39.06 | 16.30 | 23.39 | 16.89 | 35.35 | 61.28 | 72.42 | 1.41% |
| NashMTL | 51.73 | 74.10 | 39.55 | 16.50 | 23.21 | 16.74 | 35.39 | 61.80 | 72.92 | 1.11% |
| GCS | 52.67 | 74.59 | 39.72 | 16.64 | 24.10 | 17.56 | 34.04 | 59.80 | 71.04 | 0.09% |
| OL_AUX | 52.07 | 74.28 | 39.32 | 16.30 | 23.98 | 17.87 | 33.89 | 59.53 | 71.08 | 0.17% |
| ARML | 52.73 | 74.85 | 39.61 | 16.65 | 23.89 | 17.50 | 34.24 | 59.87 | 71.39 | 0.37% |
| Auto-$\lambda$ | 52.40 | 74.62 | 39.25 | 16.25 | 23.38 | 17.20 | 34.05 | 61.18 | 72.05 | 1.17% |
| Post-train | 52.08 | 74.86 | 39.58 | 16.77 | 22.98 | 16.48 | 36.04 | 62.27 | 73.20 | 1.49% |
| ForkMerge | 53.67 | 75.64 | 38.91 | 16.47 | 22.18 | 15.60 | 37.93 | 64.29 | 74.81 | 4.03% |

### C.3 DomainNet上的辅助域图像识别

**数据集细节。** 由于原始DomainNet [61] 没有提供单独的验证集，我们从测试集中随机拆分50%的数据作为验证集，其余50%的数据作为测试集。对于每个任务，训练集、验证集和测试集的比例约为70%/15%/15%。表8总结了该数据集的统计数据。DomainNet采用Custom（仅研究、非商业）许可证。

**表8：DomainNet数据集概览。** 每个域的训练集、验证集和测试集样本数量及其描述。

| 任务 | 训练集 | 验证集 | 测试集 | 描述 |
| --- | --- | --- | --- | --- |
| Clipart | 33.5K | 7.3K | 7.3K | 剪贴画图像集合 |
| Real | 120.9K | 26.0K | 26.0K | 照片和真实世界图像 |
| Sketch | 48.2K | 10.5K | 10.5K | 特定对象的素描 |
| Infograph | 36.0K | 7.8K | 7.8K | 信息图图像 |
| Painting | 50.4K | 10.9K | 10.9K | 对象的绘画描绘 |
| Quickdraw | 120.7K | 25.9K | 25.9K | 游戏"Quick Draw"的涂鸦 |

**实验细节。** 我们采用动量为0.9、批量大小为48的小批量SGD。我们在 $\{0.003, 0.01, 0.03\}$ 中搜索初始学习率，并采用余弦退火策略[45]在训练过程中调整学习率。我们采用在ImageNet上预训练的ResNet-101作为骨干网络。每种方法训练50K次迭代。在ForkMerge中，参数每12.5K次迭代合并一次。

### C.4 AliExpress上的CTR和CTCVR预测

**数据集细节。** AliExpress [36] 来自淘宝AliExpress搜索系统的真实流量日志，总计包含超过1亿条记录。我们将时间序列中前90%的数据作为训练集，其余5%和5%分别作为验证集和测试集。AliExpress包含2个任务：点击率（CTR，Click-Through Rate）和点击转化率（CTCVR，Click-Through Conversion Rate），以及4个场景：西班牙（ES）、法国（FR）、荷兰（NL）和美国（US）。表9总结了该数据集的统计数据。AliExpress采用知识共享署名-非商业性使用-相同方式共享4.0国际（CC BY-NC-SA 4.0）许可证。

**表9：AliExpress数据集概览。** 其中CTR = 点击次数 / 展示次数，CTCVR = 购买次数 / 展示次数。

| 统计量 | ES | FR | NL | US |
| --- | --- | --- | --- | --- |
| 商品数 | 8.7M | 7.4M | 6M | 8M |
| 页面浏览数 | 2M | 1.7M | 1.2M | 1.8M |
| 展示数 | 31.6M | 27.4M | 17.7M | 27.4M |
| 点击数 | 841K | 535K | 382K | 450K |
| 购买数 | 19.1K | 14.4K | 13.8K | 10.9K |
| CTR | 2.66% | 2.01% | 2.16% | 1.64% |
| CTCVR | 0.60‰ | 0.54‰ | 0.78‰ | 0.40‰ |

**实验细节。** 大多数方法的架构基于ESMM [47]，由一个所有任务共享的单一嵌入层和每个任务的多个独立DNN（Deep Neural Network，深度神经网络）塔组成。每个特征域的嵌入维度为128。每种方法使用Adam优化器训练50个epoch，批量大小为2048，学习率为 $10^{-3}$，权重衰减为 $10^{-6}$。

### C.5 CIFAR10和SVHN上的半监督学习

**数据集细节。** 遵循[67]，我们首先将CIFAR10 [31] 和SVHN [56] 的原始训练集分为训练集和验证集。然后，我们从训练集中随机采样标记图像。表10总结了CIFAR-10和SVHN的统计数据。

**表10：CIFAR-10和SVHN数据集概览。**

| 数据集 | 标记样本数 | 未标记样本数 | 验证集 | 测试集 |
| --- | --- | --- | --- | --- |
| CIFAR-10 | 4000 | 41000 | 5000 | 10000 |
| SVHN | 1000 | 64931 | 7326 | 26032 |

**实验细节。** (1) 辅助任务。遵循[82, 67]，我们考虑两个自监督辅助任务Rotation [19] 和Exempler-MT [14]。在Rotation中，我们将每个图像旋转 $[0^{\circ}, 90^{\circ}, 180^{\circ}, 270^{\circ}]$ 并让网络预测角度。在Exemplar-MT中，模型被训练提取对广泛图像变换不变的特性。(2) 超参数。我们采用Adam [29] 优化器，初始学习率为0.005。每种方法训练200K次迭代，在160K次迭代时将学习率衰减0.2倍。我们使用Wide ResNet-28-2 [80] 作为骨干网络。在ForkMerge中，参数每10K次迭代合并一次。

### C.6 ForkMerge的数据划分策略

如第4.2节所述，在ForkMerge中，我们可以用不同的辅助任务集构建分支。下面我们概述实验中使用的具体数据划分策略，这与先前的ATL文献一致：

- 对于NYUv2数据集，多个任务共享相同的输入，但它们的输出不同。在这种设置下，每个分支都有相同的输入数据，包括整个数据集。不同分支之间的区别完全在于任务权重向量 $\{\nu_b\}_{b=1}^{B}$。
- 对于DomainNet、AliExpress、CIFAR-10和SVHN数据集，不同任务既有不同的输入也有不同的输出。在这些情况下，对于每个分支，如果特定任务的任务权重设为0，则该特定任务的数据将不会用于训练对应的分支。

## D 额外实验

### D.1 不同分叉分支重要性的分析

不同分叉分支的重要性是动态的。如图9所示，每个分叉分支的相对比例是动态的，并且因任务而异，这表明了动态合并机制的重要性。

**图9：训练期间NYUv2上不同分叉分支的重要性。**

### D.2 计算成本分析

算法2的计算复杂度为 $O(K)$，剪枝版本的计算复杂度为 $O(B)$。通常，大多数先前的多任务学习方法只优化一个模型，但它们的计算成本不一定是 $O(1)$。梯度平衡方法，包括MGDA [65]、GradNorm [5]、PCGrad [79]、IMTL [41]、GradVac [76]、CAGrad [40]、NashMTL [55]、GCS [15]、OL_AUX [39]和ARML [67]，需要计算每个任务的梯度，因此导致 $O(K)$ 复杂度。此外，计算梯度的内积或范数会带来与网络参数数量成比例的计算成本。一种常见的实践改进是计算共享表示的梯度[65]。然而加速效果依赖于架构，且这种技术可能会降低性能[55]。

在图10中，我们还比较了NYUv2上这些方法的实际训练时间。我们可以观察到ForkMerge所需的时间不比大多数其他方法多。考虑到它带来的显著性能提升，这些额外的计算成本也是值得的。此外，我们的分叉和合并机制使得异步优化变得极为容易，这在先前方法中并不直接可行，因此当有多个GPU可用时，我们方法的训练时间可以降低到 $O(1)$。

**图10：不同MTL方法在NYUv2上的训练速度（10次重复）。**

### D.3 收敛性和方差分析

图11绘制了NYUv2上STL、EW和ForkMerge在整个训练过程中的验证性能。每条曲线是通过使用5个不同种子优化相同方法获得的。与单任务学习或最小化所有任务的平均损失相比，ForkMerge不仅提高了最终泛化能力，还加快了收敛速度并减少了训练过程中的波动。

**图11：比较NYUv2上不同方法的学习曲线。** 每条曲线绘制了一种方法使用5个不同随机种子的验证性能的均值和标准差。

### D.4 与网格搜索 $\lambda$ 的比较

在第3节中，我们观察到调整任务权重超参数 $\lambda$ 可以有效减少负迁移并促进正迁移。[77]也提出遍历任务权重应足以充分探索帕累托前沿（至少在凸设置下），并观察到先前的MTL算法在最终性能方面与网格搜索相比没有改进。

在图12中，我们比较了NYUv2上所有方法与网格搜索的性能。在网格搜索中，每个任务的权重超参数取值 $\{0.3, 1.0, 3.0\}$，NYUv2中有3个任务，因此共有27种组合。我们发现先前方法只是在标量化帕累托前沿上产生性能权衡点，这在先前工作中也有观察到[77]。相比之下，我们提出的ForkMerge产生的点远离帕累托前沿，并比简单优化损失的加权平均实现了显著改进。这一增益的一个可能原因是网格搜索中的任务权重在训练期间是固定的，并且由于计算资源的限制取有限值，而ForkMerge中的任务权重在时间上是动态的，在值上几乎是连续的，因此能更好地避免负迁移并促进正迁移。

**图12：与NYUv2上网格搜索的比较。** 我们使用mIoU进行语义分割，1/绝对误差进行深度估计，1/平均角距离进行表面法线预测。我们绘制每对任务性能曲线的2D投影。右上角更好。

### D.5 与更大批量大小训练的比较

在某种意义上，ForkMerge中的多个分支增加了等效批量大小。已有研究揭示批量大小可能对深度模型的性能有很大影响[21, 78]。为了消融批量大小的影响，我们增加了等权重方法的批量大小。如表11所示，ForkMerge本身带来的改进显著大于单纯增加批量大小。

**表11：更大批量大小训练下不同方法的比较。** 指标同表7，$\Delta_m$ 为平均每任务性能改进（越高越好）。

| 方法 | 批量大小 | 分割：mIoU↑ | 分割：Pix Acc↑ | 深度：Abs Err↓ | 深度：Rel Err↓ | 法线：Mean↓ | 法线：Median↓ | 法线：11.25↑ | 法线：22.5↑ | 法线：30↑ | $\Delta_m$↑ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EW | 8 | 52.13 | 74.51 | 39.03 | 16.43 | 24.14 | 17.62 | 33.98 | 59.63 | 70.93 | 0.30% |
| EW | 32 | 51.40 | 73.99 | 38.86 | 16.20 | 23.99 | 17.34 | 34.58 | 60.08 | 70.85 | 0.55% |
| ForkMerge‡ | 8 | 54.30 | 75.78 | 38.42 | 16.11 | 22.41 | 15.72 | 37.81 | 63.89 | 74.35 | 4.59% |

### D.6 更多网络架构上的ForkMerge

**使用Vision Transformers的ForkMerge。** 我们将骨干网络ResNet-101替换为在ImageNet-21K上预训练的先进ViT-Base [13]，并在DomainNet数据集（第5.2节）上重复实验。如表12所示，当使用容量更大的Vision Transformer模型时，在有限数据下过拟合的风险变得更加明显。

**表12：将ResNet-101架构替换为ViT-Base架构后在DomainNet上的性能。** C、I、P、Q、R、S分别代表六个域，Avg为平均准确率，$\Delta_m$ 为平均每任务性能改进（越高越好）。

| 方法 | C | I | P | Q | R | S | Avg | $\Delta_m$↑ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| STL | 75.7 | 37.8 | 69.0 | 72.1 | 84.4 | 69.0 | 68.0 | - |
| EW | 81.9 | 43.7 | 74.0 | 71.3 | 84.1 | 73.0 | 71.3 | +5.90% |
| Auto-$\lambda$ | 81.3 | 44.1 | 73.8 | 72.1 | 84.4 | 73.5 | 71.5 | +6.62% |
| Post-train | 76.2 | 38.8 | 69.5 | 71.7 | 83.2 | 69.7 | 68.2 | +0.51% |
| ForkMerge | 83.0 | 45.6 | 76.3 | 73.2 | 87.1 | 74.7 | 73.3 | +8.97% |

这使得单任务学习（STL）效果较差，从而导致等权重（EW）方法优于STL，使得Post-train方法不及EW和Auto-$\lambda$。在这种情况下，ForkMerge仍然表现出优越的性能，验证了其在不同网络架构上的有效性。

**使用多任务架构的ForkMerge。** ForkMerge与不同的多任务架构是互补的。在表13和14中，我们提供了以MTAN [44] 和MMoE [46] 为架构的不同优化策略的比较，这两种架构分别广泛用于多任务计算机视觉任务和多任务推荐任务。在这些专门设计的多任务架构上，ForkMerge仍然显著优于其他方法。

**表13：将DeepLabV3+架构替换为MTAN架构后在NYUv2数据集上的性能。** 指标同表7，$\Delta_m$ 为平均每任务性能改进（越高越好）。

| 方法 | 分割：mIoU↑ | 分割：Pix Acc↑ | 深度：Abs Err↓ | 深度：Rel Err↓ | 法线：Mean↓ | 法线：Median↓ | 法线：11.25↑ | 法线：22.5↑ | 法线：30↑ | $\Delta_m$↑ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| STL | 52.10 | 74.42 | 40.45 | 16.34 | 22.35 | 15.23 | 38.96 | 64.56 | 74.51 | 3.05% |
| EW | 53.27 | 75.36 | 39.37 | 16.38 | 23.61 | 17.00 | 35.00 | 61.01 | 72.07 | 1.62% |
| GCS | 53.05 | 74.79 | 39.50 | 16.49 | 24.05 | 17.49 | 34.14 | 59.88 | 71.13 | 0.57% |
| OL_AUX | 52.47 | 74.70 | 39.27 | 16.39 | 23.66 | 17.43 | 34.49 | 59.96 | 71.76 | 0.82% |
| ARML | 52.33 | 74.59 | 39.46 | 16.61 | 23.57 | 17.41 | 34.56 | 60.12 | 72.04 | 0.55% |
| Auto-$\lambda$ | 52.90 | 75.03 | 39.67 | 16.45 | 22.71 | 15.60 | 38.35 | 64.09 | 73.92 | 3.18% |
| ForkMerge‡ | 55.25 | 76.16 | 38.45 | 16.08 | 21.94 | 15.22 | 38.96 | 65.04 | 75.33 | 5.76% |

**表14：将ESMM架构替换为MMoE架构后在AliExpress数据集上的性能。** 指标为CTR和CTCVR（CVCTR）各场景的AUC（越高越好），Avg为平均值，$\Delta_m$ 为平均每任务性能改进（越高越好）。

| 方法 | CTR：ES | CTR：FR | CTR：NL | CTR：US | CVCTR：ES | CVCTR：FR | CVCTR：NL | CVCTR：US | Avg | $\Delta_m$↑ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EW | 0.7287 | 0.7244 | 0.7225 | 0.7068 | 0.8874 | 0.8669 | 0.8688 | 0.8742 | 0.7974 | 0.11% |
| GCS | 0.7300 | 0.7190 | 0.7270 | 0.7102 | 0.8857 | 0.8773 | 0.8680 | 0.8740 | 0.7989 | 0.29% |
| OL_AUX | 0.7265 | 0.7283 | 0.7264 | 0.7146 | 0.8849 | 0.8750 | 0.8710 | 0.8770 | 0.8005 | 0.50% |
| ARML | 0.7289 | 0.7278 | 0.7248 | 0.7081 | 0.8869 | 0.8801 | 0.8714 | 0.8610 | 0.7986 | 0.26% |
| Auto-$\lambda$ | 0.7269 | 0.7273 | 0.7256 | 0.7111 | 0.8827 | 0.8811 | 0.8721 | 0.8726 | 0.7999 | 0.42% |
| ForkMerge‡ | 0.7368 | 0.7349 | 0.7359 | 0.7116 | 0.8942 | 0.8791 | 0.8717 | 0.8840 | 0.8060 | 1.20% |
