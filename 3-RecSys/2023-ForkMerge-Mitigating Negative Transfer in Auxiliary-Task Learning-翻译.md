# ForkMerge: Mitigating Negative Transfer in Auxiliary-Task Learning

> Junguang Jiang, Baixu Chen, Junwei Pan, Ximei Wang, Dapeng Liu, Jie Jiang, Mingsheng Long | Tsinghua University & Tencent

# ForkMerge：缓解辅助任务学习中的负迁移

Junguang Jiang\*∗, Baixu Chen\*∗, Junwei Pan§, Ximei Wang§, Dapeng Liu§, Jie Jiang§,
本文介绍了 ForkMerge: Mitigating Negative Transfer in Auxiliary-Task Learning。核心内容：


关键发现：


Mingsheng Long(cid:66)
School of Software, BNRist, Tsinghua University, China
§Tencent Inc, China
{jjg20,cbx22}@mails.tsinghua.edu.cn, {jonaspan,messixmwang,rocliu,zeus}@tencent.com,
mingsheng@tsinghua.edu.cn

## Abstract


---

## 摘要

Auxiliary-Task Learning (ATL) aims to improve the performance of the target task by leveraging the knowledge obtained from related tasks. Occasionally, learning multiple tasks simultaneously results in lower accuracy than learning only the target task, which is known as negative transfer. This problem is often attributed to the gradient conflicts among tasks, and is frequently tackled by coordinating the task gradients in previous works. However, these optimization-based methods largely overlook the auxiliary-target generalization capability. To better understand the root cause of negative transfer, we experimentally investigate it from both optimization and generalization perspectives. Based on our findings, we introduce ForkMerge, a novel approach that periodically forks the model into multiple branches, automatically searches the varying task weights by minimizing target validation errors, and dynamically merges all branches to filter out detrimental task-parameter updates. On a series of auxiliary-task learning benchmarks, ForkMerge outperforms existing methods and effectively mitigates negative transfer.

辅助任务学习（ATL）旨在通过利用从相关任务获取的知识来提高目标任务的性能。偶尔，同时学习多个任务会导致比仅学习目标任务更低的准确率，这被称为负迁移。这个问题通常归因于任务间的梯度冲突，并且在先前的工作中常常通过协调任务梯度来处理。然而，这些基于优化的方法在很大程度上忽略了辅助-目标泛化能力。为了更好地理解负迁移的根本原因，我们从优化和泛化两个角度对其进行了实验研究。基于我们的发现，我们引入了ForkMerge，一种新颖的方法，它周期性地将模型分叉成多个分支，通过最小化目标验证误差来自动搜索变化的任务权重，并动态合并所有分支以过滤掉有害的任务参数更新。在一系列辅助任务学习基准上，ForkMerge优于现有方法，并有效缓解了负迁移。

## 1 Introduction

## 1 引言

Deep neural networks have achieved remarkable success in various machine learning applications, such as computer vision [23, 22], natural language processing [62, 11, 57], and recommendation systems [46]. However, one major challenge in training deep neural networks is the scarcity of labeled data. In recent years, Auxiliary-Task Learning (ATL) has emerged as a promising technique to address this challenge [67, 39, 43]. ATL improves the generalization of target tasks by leveraging the useful signals provided by some related auxiliary tasks. For instance, larger-scale tasks, such as user click prediction, can be utilized as auxiliary tasks to improve the performance of smaller-scale target tasks, such as user conversion prediction in recommendation [47, 36]. Self-supervised tasks on unlabeled data can serve as auxiliary tasks to improve the performance of the target task in computer vision and natural language processing, without requiring additional labeled data [34, 69, 11, 3].

深度神经网络在各种机器学习应用中取得了显著成功，例如计算机视觉[23, 22]、自然语言处理[62, 11, 57]和推荐系统[46]。然而，训练深度神经网络的一个主要挑战是标记数据的稀缺性。近年来，辅助任务学习（ATL）已成为解决这一挑战的一种有前景的技术[67, 39, 43]。ATL通过利用一些相关辅助任务提供的有用信号来提高目标任务的泛化能力。例如，较大规模的任务（如用户点击预测）可以作为辅助任务来提高较小规模的目标任务（如推荐中的用户转化预测）的性能[47, 36]。无标记数据上的自监督任务可以作为辅助任务来提高计算机视觉和自然语言处理中目标任务的性能，而无需额外的标记数据[34, 69, 11, 3]。

However, in practice, learning multiple tasks simultaneously sometimes leads to performance degradation compared to learning only the target task, a phenomenon known as negative transfer [84, 75]. Even in large language models, negative transfer problems may still exist. For example, RLHF [7], a key component of ChatGPT [57], achieves negative effects on nearly half of the multiple-choice question tasks when post-training GPT-4 [58]. There has been a significant amount of methods proposed to mitigate negative transfer in ATL [71, 79, 15, 39]. Notable previous studies attribute negative transfer to the optimization difficulty, especially the gradient conflicts between different tasks, and propose to mitigate negative transfer by reducing interference between task gradients [79, 15]. Other works focus on selecting the most relevant auxiliary tasks and reducing negative transfer by avoiding task groups with severe task conflicts [71, 17]. However, despite the significant efforts to address negative transfer, its underlying causes are still not fully understood.

然而，在实践中，同时学习多个任务有时会导致性能下降，相比于仅学习目标任务，这种现象被称为负迁移[84, 75]。即使在大语言模型中，负迁移问题可能仍然存在。例如，RLHF [7]（ChatGPT [57] 的一个关键组件）在后训练GPT-4 [58] 时，在近一半的多选题任务上产生了负面效果。已经有大量方法被提出来缓解ATL中的负迁移[71, 79, 15, 39]。 notable先前研究将负迁移归因于优化困难，特别是不同任务之间的梯度冲突，并提出了通过减少任务梯度间的干扰来缓解负迁移的方法[79, 15]。其他工作专注于选择最相关的辅助任务，并通过避免具有严重任务冲突的任务组来减少负迁移[71, 17]。然而，尽管为解决负迁移付出了巨大努力，其根本原因仍未完全被理解。

In this regard, we experimentally analyze potential causes of negative transfer in ATL from the perspectives of optimization and generalization. From an optimization view, our experiments suggest that gradient conflicts do not necessarily lead to negative transfer. For example, weight decay, a special auxiliary task, can conflict with the target task in gradients but still be beneficial to the target performance. From a generalization view, we observe that negative transfer is more likely to occur when the distribution shift between the multi-task training data and target test data is enlarged.

在这方面，我们从优化和泛化的角度实验分析了ATL中负迁移的潜在原因。从优化的角度来看，我们的实验表明梯度冲突并不必然导致负迁移。例如，权重衰减（一种特殊的辅助任务）可能在梯度上与目标任务冲突，但仍然对目标性能有益。从泛化的角度来看，我们观察到当多任务训练数据与目标测试数据之间的分布偏移扩大时，负迁移更可能发生。

Based on our above findings, we present a new approach named ForkMerge. Since we cannot know which task distribution combination leads to better generalization in advance, and training models for each possible distribution is prohibitively expensive, we transform the problem of combining task distributions into that of combining model hypotheses. Specifically, we fork the model into multiple branches and optimize the parameters of different branches on diverse data distributions by varying the task weights. Then at regular intervals, we merge and synchronize the parameters of each branch to approach the optimal model hypothesis. In this way, we will filter out harmful parameter updates to mitigate negative transfer and keep desirable parameter updates to promote positive transfer.

基于上述发现，我们提出了一种名为ForkMerge的新方法。由于我们无法预先知道哪种任务分布组合能带来更好的泛化，并且为每种可能的分布训练模型代价过高，我们将组合任务分布的问题转化为组合模型假设的问题。具体来说，我们将模型分叉成多个分支，通过改变任务权重在不同数据分布上优化不同分支的参数。然后每隔一定时间，我们合并并同步每个分支的参数以逼近最优模型假设。通过这种方式，我们将过滤掉有害的参数更新以缓解负迁移，并保留期望的参数更新以促进正迁移。

The contributions of this work are summarized as follows: (1) We systematically identify the problem and analyze the causes of negative transfer in ATL. (2) We propose ForkMerge, a novel approach to mitigate negative transfer and boost the performance of ATL. (3) We conduct extensive experiments and validate that ForkMerge outperforms previous methods on a series of ATL benchmarks.

本文的贡献总结如下：(1) 我们系统地识别了问题并分析了ATL中负迁移的原因。(2) 我们提出了ForkMerge，一种缓解负迁移并提升ATL性能的新方法。(3) 我们进行了广泛的实验，验证了ForkMerge在一系列ATL基准上优于先前的方法。

## 2 Related Work

## 2 相关工作

### 2.1 Auxiliary-Task Learning

### 2.1 辅助任务学习

Auxiliary-Task Learning (ATL) enhances a model's performance on a target task by utilizing knowledge from related auxiliary tasks. The two main challenges in ATL are selecting appropriate auxiliary tasks and optimizing them jointly with the target task. To find the proper auxiliary tasks for ATL, recent studies have explored task relationships by grouping positively related tasks together and assigning unrelated tasks to different groups to avoid task interference [81, 71, 17, 70]. Once auxiliary tasks are determined, most ATL methods create a unified loss by linearly combining the target and auxiliary losses. However, choosing task weights is challenging due to the exponential increase in search space with the number of tasks, and fixing the weight of each task loss can lead to negative transfer [32]. Recent studies propose various methods to automatically choose task weights, such as using one-step or multi-step gradient similarity [15, 39, 9], minimizing representation-based task distance [2] or gradient gap [67], employing a parametric cascade auxiliary network [54], or from the perspective of bargaining game [66]. However, these methods mainly address the optimization difficulty after introducing auxiliary tasks and may overlook the generalization problem.

辅助任务学习（ATL）通过利用来自相关辅助任务的知识来提高模型在目标任务上的性能。ATL中的两个主要挑战是选择合适的辅助任务以及将它们与目标任务联合优化。为了找到ATL的合适辅助任务，最近的研究通过将正相关任务分组在一起并将不相关任务分配到不同组以避免任务干扰来探索任务关系[81, 71, 17, 70]。一旦确定了辅助任务，大多数ATL方法通过线性组合目标和辅助损失来创建统一损失。然而，由于搜索空间随任务数量呈指数增长，选择任务权重具有挑战性，并且固定每个任务损失的权重可能导致负迁移[32]。最近的研究提出了各种自动选择任务权重的方法，例如使用单步或多步梯度相似性[15, 39, 9]，最小化基于表示的任务距离[2]或梯度差距[67]，使用参数化级联辅助网络[54]，或从讨价还价博弈的角度[66]。然而，这些方法主要处理引入辅助任务后的优化困难，可能忽略了泛化问题。

Recently, AANG [10] formulates a novel searching space of auxiliary tasks and adopts the meta-learning technique, which prioritizes target task generalization, to learn single-step task weightings. This parallel finding highlights the importance of the target task generalization and we further introduce the multi-step task weightings to reduce the estimation uncertainty. Another parallel method, ColD Fusion [12], explores collaborative multitask learning and proposes to fuse each contributor's parameter to construct a shared model. In this paper, we further take into account the diversity of tasks and the intricacies of task relationships and derive a method for combining model parameters from the weights of task combinations.

最近，AANG [10] 构建了一个新颖的辅助任务搜索空间，并采用优先考虑目标任务泛化的元学习技术来学习单步任务权重。这一并行的发现突出了目标任务泛化的重要性，我们进一步引入了多步任务权重以减少估计不确定性。另一个并行方法ColD Fusion [12] 探索了协作多任务学习，并提出融合每个贡献者的参数以构建共享模型。在本文中，我们进一步考虑了任务的多样性和任务关系的复杂性，并推导出一种从任务组合权重出发来组合模型参数的方法。

### 2.2 Multi-Task Learning

### 2.2 多任务学习

Different from ATL, Multi-Task Learning (MTL) aims to improve the performance of all tasks by learning multiple objectives from a shared representation. To facilitate information sharing and minimize task conflict, many multi-task architectures have been designed, including hard-parameter sharing [30, 22, 24] and soft-parameter sharing [51, 64, 16, 46, 44, 48, 72]. Another line of work aims to optimize strategies to reduce task conflict. Methods such as loss balancing and gradient balancing propose to find suitable task weighting through various criteria, such as task uncertainty [28], task loss magnitudes [44], gradient norm [5], and gradient directions [79, 6, 40, 41, 25, 55].

与ATL不同，多任务学习（MTL）旨在通过从共享表示中学习多个目标来提高所有任务的性能。为了促进信息共享并最小化任务冲突，已经设计了许多多任务架构，包括硬参数共享[30, 22, 24]和软参数共享[51, 64, 16, 46, 44, 48, 72]。另一条工作线旨在优化策略以减少任务冲突。诸如损失平衡和梯度平衡等方法提出通过各种标准来寻找合适的任务权重，例如任务不确定性[28]、任务损失幅度[44]、梯度范数[5]和梯度方向[79, 6, 40, 41, 25, 55]。

Although MTL methods can be directly used to jointly train auxiliary and target tasks, the asymmetric task relationships in ATL are usually not taken into account in MTL.

尽管MTL方法可以直接用于联合训练辅助任务和目标任务，但ATL中的非对称任务关系通常不在MTL的考虑范围内。

### 2.3 Negative Transfer

### 2.3 负迁移

Negative Transfer (NT) is a widely existing phenomenon in machine learning, where transferring knowledge from the source data or model can have negative impact on the target learner [63, 60, 27]. To mitigate negative transfer, domain adaptation methods design importance sampling or instance weighting strategies to prioritize related source data [75, 83]. Fine-tuning methods filter out detrimental pre-trained knowledge by suppressing untransferable spectral components in the representation [4]. MTL methods use gradient surgery or task weighting to reduce the gradient conflicts across tasks [79, 76, 25, 42]. Different from previous work, we propose to dynamically filter out harmful parameter updates in the training process to mitigate negative transfer. Besides, we provide an in-depth experimental analysis of the causes of negative transfer in ATL, which is rare in this field yet will be helpful for future research.

负迁移（NT）是机器学习中广泛存在的现象，其中从源数据或模型迁移知识可能对目标学习器产生负面影响[63, 60, 27]。为了缓解负迁移，领域自适应方法设计了重要性采样或实例加权策略来优先处理相关源数据[75, 83]。微调方法通过抑制表示中不可迁移的频谱分量来过滤有害的预训练知识[4]。MTL方法使用梯度手术或任务权重来减少跨任务的梯度冲突[79, 76, 25, 42]。与先前工作不同，我们提出在训练过程中动态过滤有害的参数更新以缓解负迁移。此外，我们对ATL中负迁移的原因提供了深入的实验分析，这在该领域较为罕见，但对未来研究将有所帮助。

## 3 Negative Transfer Analysis

## 3 负迁移分析

In this section, we assume that both the target task Ttgt and the auxiliary task Taux are given. Then the objective is to find model parameters \theta that achieve higher performance on the target task by joint training with the auxiliary task,

Problem and Notation.
在本节中，我们假设目标任务Ttgt和辅助任务Taux都已给定。目标是通过与辅助任务的联合训练找到在目标任务上实现更高性能的模型参数\theta，

问题与符号。

ETtgtLtgt(\theta) + \lambdaETauxLaux(\theta),

min \theta

(1)

where L is the training loss, and \lambda is the relative weighting hyper-parameter between the auxiliary task and the target task. Our final objective is max\theta [P(\theta)], where P is the relative performance measure for the target task Ttgt, such as the accuracy in classification. Next we define the Transfer Gain to measure the impact of Taux on Ttgt.

其中L是训练损失，\lambda是辅助任务与目标任务之间的相对权重超参数。我们的最终目标是max\theta [P(\theta)]，其中P是目标任务Ttgt的相对性能度量，例如分类中的准确率。接下来我们定义迁移增益来衡量Taux对Ttgt的影响。

Definition 3.1 (Transfer Gain, TG). Denote the model obtained by some ATL algorithm A as \thetaA(Ttgt, Taux, \lambda) and the model obtained by single-task learning on target task as \theta(Ttgt). Let P be the performance measure on the target task Ttgt. Then the algorithm A can be evaluated by

定义3.1（迁移增益，TG）。将由某个ATL算法A获得的模型记为\thetaA(Ttgt, Taux, \lambda)，将目标任务上的单任务学习获得的模型记为\theta(Ttgt)。设P为目标任务Ttgt上的性能度量。则算法A可以通过下式评估

T G(\lambda, A) = P(\thetaA(Ttgt, Taux, \lambda)) − P(\theta(Ttgt)).

(2)

Going beyond previous work on Negative Transfer (NT) [75, 84], we further divide negative transfer in ATL into two types.

超越先前关于负迁移（NT）的工作[75, 84]，我们进一步将ATL中的负迁移分为两类。

Definition 3.2 (Weak Negative Transfer, WNT). For some ATL algorithm A with weighting hyper-parameter \lambda , weak negative transfer occurs if T G(\lambda, A) < 0.

定义3.2（弱负迁移，WNT）。对于某个ATL算法A，其权重超参数为\lambda，如果T G(\lambda, A) < 0，则发生弱负迁移。

Definition 3.3 (Strong Negative Transfer, SNT). For some ATL algorithm A, strong negative transfer occurs if max\lambda>0 T G(\lambda, A) < 0.

定义3.3（强负迁移，SNT）。对于某个ATL算法A，如果max\lambda>0 T G(\lambda, A) < 0，则发生强负迁移。

Figure 1 illustrates the difference between weak negative transfer and strong negative transfer. The most essential difference is that we might be able to avoid weak negative transfer by selecting a proper weighting hyper-parameter \lambda, yet we cannot avoid strong negative transfer in this way.

图1说明了弱负迁移和强负迁移之间的区别。最根本的区别在于，我们或许能够通过选择合适的权重超参数\lambda来避免弱负迁移，但无法通过这种方式避免强负迁移。

Next, we will analyze negative transfer in ATL from two different perspectives: optimization and generalization. We conduct our analysis on a multi-domain image recognition dataset DomainNet [61] with ResNet-18 [23] pre-trained on ImageNet. Specifically, we use task Painting and Quickdraw in DomainNet as target tasks respectively to showcase weak negative transfer and strong negative transfer, and mix all other tasks in DomainNet as auxiliary tasks. We will elaborate on the DomainNet dataset in Appendix C.3 and provide the detailed experiment design in Appendix B.

接下来，我们将从两个不同的角度分析ATL中的负迁移：优化和泛化。我们在使用ImageNet预训练的ResNet-18 [23] 的多域图像识别数据集DomainNet [61] 上进行分析。具体来说，我们分别使用DomainNet中的Painting和Quickdraw任务作为目标任务来展示弱负迁移和强负迁移，并混合DomainNet中的所有其他任务作为辅助任务。我们将在附录C.3中详细阐述DomainNet数据集，并在附录B中提供详细的实验设计。

### 3.1 Effect of Gradient Conflicts

### 3.1 梯度冲突的影响

Figure 1: Weak Negative Transfer (WNT) vs. Strong Negative Transfer (SNT).

图1：弱负迁移（WNT）与强负迁移（SNT）。

It is widely believed that gradient conflicts between different tasks will lead to optimization difficulties [79, 40], which in turn lead to negative transfer. The degree of gradient conflict is usually measured by the Gradient Cosine Similarity [79, 76, 15].

人们普遍认为不同任务之间的梯度冲突会导致优化困难[79, 40]，进而导致负迁移。梯度冲突的程度通常通过梯度余弦相似性来衡量[79, 76, 15]。

Definition 3.4 (Gradient Cosine Similarity, GCS). Denote \phiij as the angle between two task gradients gi and gj, then we define the gradient cosine similarity as cos \phiij and the gradients as conflicting when cos \phiij < 0.

定义3.4（梯度余弦相似性，GCS）。设\phiij为两个任务梯度gi和gj之间的夹角，则我们将梯度余弦相似性定义为cos \phiij，并且当cos \phiij < 0时梯度为冲突的。

In Figure 2, we plot the correlation curve between gradient cosine similarity and transfer gain. Somewhat counterintuitively, we observe that negative transfer and gradient conflicts are not strongly correlated, and negative transfer might be severer when the task gradients are highly consistent.

在图2中，我们绘制了梯度余弦相似性与迁移增益之间的相关曲线。有些反直觉的是，我们观察到负迁移和梯度冲突并不强相关，并且当任务梯度高度一致时负迁移可能更严重。

Finding 1. Negative transfer is not necessarily caused by gradient conflicts and gradient conflicts do not necessarily lead to negative transfer.

发现1. 负迁移不一定由梯度冲突引起，梯度冲突也不一定导致负迁移。

It seems contradictory to the previous work [79, 15] and the reason is that previous work mainly considers the optimization convergence during training, while in our experiments we further consider the generalization during evaluation (transfer gain is estimated on the validation set). Although the conflicting gradient of the auxiliary task will increase the training loss of the target task and slow down its convergence speed [37], it may also play a role similar to regularization [32], reducing the over-fitting of the target task, thereby reducing its generalization error. To confirm our hypothesis, we repeat the above experiments with the auxiliary task replaced by L2 regularization and observe a similar phenomenon as shown in Figure 2(c)-(d), which indicates that the gradient conflict in ATL is not necessarily harmful, as it may serve as a proper regularization.

这似乎与先前的工作[79, 15]相矛盾，原因在于先前工作主要考虑训练过程中的优化收敛，而在我们的实验中，我们进一步考虑了评估过程中的泛化（迁移增益在验证集上估计）。尽管辅助任务的冲突梯度会增加目标任务的训练损失并减慢其收敛速度[37]，但它也可能起到类似于正则化的作用[32]，减少目标任务的过拟合，从而降低其泛化误差。为证实我们的假设，我们用L2正则化替换辅助任务重复了上述实验，并观察到了类似的现象，如图2(c)-(d)所示，这表明ATL中的梯度冲突不一定是有害的，因为它可以作为一种适当的正则化。

Figure 2 also indicates that the weighting hyper-parameter \lambda in ATL has a large impact on negative transfer. A proper \lambda not only reduces negative transfer but also promotes positive transfer.

图2还表明ATL中的权重超参数\lambda对负迁移有很大影响。合适的\lambda不仅减少了负迁移，还促进了正迁移。

Figure 2: The effect of gradient conflicts. The correlation curve between Transfer Gain (TG) and Gradient Cosine Similarity (GCS) under different \lambda. For a fair comparison, each data point starts from the same model parameters in the middle of the training process and updates with one-step multi-task gradient descent. P and Q are short for Painting and Quickdraw tasks, respectively.

图2：梯度冲突的影响。不同\lambda下迁移增益（TG）与梯度余弦相似性（GCS）之间的相关曲线。为了公平比较，每个数据点从训练过程中间的相同模型参数开始，并通过单步多任务梯度下降进行更新。P和Q分别是Painting和Quickdraw任务的缩写。

### 3.2 Effect of Distribution Shift

### 3.2 分布偏移的影响

Next, we will analyze negative transfer from the perspective of generalization. We notice that adjusting \lambda will change the data distribution that the model is fitting. For instance, when \lambda = 0, the model only fits the data distribution of the target task, and when \lambda = 1, the model will fit the interpolated distribution of the target and auxiliary tasks. Formally, given the target distribution Ttgt and the auxiliary distribution Taux, the interpolated distribution of the target and auxiliary task is Tinter,

接下来，我们将从泛化的角度分析负迁移。我们注意到调整\lambda会改变模型正在拟合的数据分布。例如，当\lambda=0时，模型仅拟合目标任务的数据分布；当\lambda=1时，模型将拟合目标任务和辅助任务的插值分布。形式上，给定目标分布Ttgt和辅助分布Taux，目标任务和辅助任务的插值分布为Tinter，

Tinter ∼ (1 − Z)Ttgt + ZTaux, Z ∼ Bernoulli( \lambda / (1 + \lambda) ),

(3)

where \lambda is the task-weighting hyper-parameter. Figure 3(a) quantitatively visualizes the distribution shift under different \lambda using t-SNE [74].

其中\lambda是任务权重超参数。图3(a)使用t-SNE [74] 定量可视化了不同\lambda下的分布偏移。

To quantitatively measure the distribution shift in ATL, we introduce the following definitions. Following the notations of [53], we consider multiclass classification with hypothesis space F of scoring functions f : X $\times$ Y \rightarrow R where f (x, y) indicates the confidence of predicting x as y.

为了定量衡量ATL中的分布偏移，我们引入以下定义。遵循[53]的符号，我们考虑多类分类，其假设空间F为评分函数f: X $\times$ Y \rightarrow R，其中f(x, y)表示将x预测为y的置信度。

Definition 3.5 (Confidence Score Discrepancy, CSD). Given scoring function hypothesis F, denote the optimal hypothesis on distribution D as f ∗ D, then confidence score discrepancy between distribution D and D′ induced by F is defined by

定义3.5（置信度分数差异，CSD）。给定评分函数假设F，将分布D上的最优假设记为f ∗ D，则由F诱导的分布D与D′之间的置信度分数差异定义为

dF (D, D′) ≜ 1 − Ex∼D′ max y\inY f ∗ D(x, y).

(4)

Confidence score discrepancy between training and test data indicates how unconfident the model is on the test data, which is expected to increase when the data shift enlarges [59, 50].

训练数据和测试数据之间的置信度分数差异表示模型对测试数据的不确信程度，预计当数据偏移扩大时该值会增加[59, 50]。

Figure 3: The effect of distribution shift. (a) Visualization of training distribution and test distribution under different \lambda. (b) For weak negative transfer tasks, as \lambda increases, Confidence Score Discrepancy (CSD) first drops and then rises and Transfer Gain (TG) is first positive and then negative. For strong negative transfer tasks, CSD increases monotonically and TG remains negative.

图3：分布偏移的影响。(a) 不同\lambda下训练分布和测试分布的可视化。(b) 对于弱负迁移任务，随着\lambda增加，置信度分数差异（CSD）先下降后上升，迁移增益（TG）先正后负。对于强负迁移任务，CSD单调增加，TG保持为负。

Figure 3(b) indicates the correlation between confidence score discrepancy and transfer gain. For weak negative transfer tasks, when \lambda increases at first, the introduced auxiliary tasks will shift the training distribution towards the test distribution, thus decreasing the confidence score discrepancy between training and test data and improving the generalization of the target task. However, when \lambda continues to increase, the distribution shift gradually increases, finally resulting in negative transfer. For strong negative transfer tasks, there is a large gap between the distribution of the introduced auxiliary tasks and that of the target task. Thus, increasing \lambda always enlarges confidence score discrepancy and always leads to negative transfer. In summary,

图3(b)表明了置信度分数差异与迁移增益之间的相关性。对于弱负迁移任务，当\lambda最初增加时，引入的辅助任务将使训练分布向测试分布偏移，从而减少训练和测试数据之间的置信度分数差异，并提高目标任务的泛化能力。然而，当\lambda继续增加时，分布偏移逐渐增大，最终导致负迁移。对于强负迁移任务，引入的辅助任务分布与目标任务分布之间存在较大差距。因此，增加\lambda总是扩大置信度分数差异，并总是导致负迁移。总结来说，

Finding 2. Negative transfer is likely to occur if the introduced auxiliary task enlarges the distribution shift between training and test data for the target task.

发现2. 如果引入的辅助任务扩大了目标任务训练数据与测试数据之间的分布偏移，则负迁移可能发生。

## 4 Methods

## 4 方法

In Section 4.1, based on our above analysis, we will introduce how to mitigate negative transfer when the auxiliary task is determined. Then in Section 4.2, we will further discuss how to use the proposed method to select appropriate auxiliary tasks and optimize them jointly with the target task simultaneously.

在第4.1节中，基于上述分析，我们将介绍当辅助任务确定时如何缓解负迁移。然后在第4.2节中，我们将进一步讨论如何使用所提出的方法同时选择合适的辅助任务并与目标任务联合优化它们。

### 4.1 ForkMerge

### 4.1 ForkMerge

In this section, we assume that the auxiliary task Taux is given. When updating the parameters \thetat with Equation (1) at training step t, we have

在本节中，我们假设辅助任务Taux已给定。当在训练步骤t使用公式(1)更新参数\thetat时，我们有

\thetat+1(\lambda) = \thetat − \eta(gtgt(\thetat) + \lambdagaux(\thetat)),

where \eta is the learning rate, gtgt and gaux are the gradients calculated from Ltgt and Laux respectively. Section 3.1 reveals that the gradient conflict between gtgt and gaux does not necessarily lead to negative transfer as long as \lambda is carefully tuned and Section 3.2 shows that negative transfer is related to generalization. Thus we propose to dynamically adjust \lambda according to the target validation performance Pˆ to mitigate negative transfer:

其中\eta是学习率，gtgt和gaux分别是根据Ltgt和Laux计算的梯度。第3.1节揭示了只要\lambda经过仔细调整，gtgt和gaux之间的梯度冲突不一定导致负迁移，第3.2节表明负迁移与泛化相关。因此，我们提出根据目标验证性能Pˆ动态调整\lambda以缓解负迁移：

max \lambda Pˆ(\thetat+1) = Pˆ(\thetat − \eta(gtgt(\thetat) + \lambdagaux(\thetat))).

(6)

Algorithm 1 ForkMerge Training Pipeline.
Require: initial model parameter \theta0, total iterations T , interval ∆t
Ensure: final model parameter \theta∗
fork model into 2 copies {\thetab}1 b=0
for b = 0 to 1 do
\thetab 0 \leftarrow \theta0
end for
while t < T do

▷ initialization

 T , task relevance \lambda∗

for b = 0 to 1 do
for t′ = t to t + ∆t − 1 do
\thetab t′+1 = \thetab t′ − \eta(gtgt(\thetab t′) + b · gaux(\thetab t′))
end for

▷ independent update

end for
\lambda∗ \leftarrow arg max\lambda Pˆ((1 − \lambda)\theta0 t+∆t + \lambda\theta1 t+∆t)

▷ search \lambda on the validation set

Figure 4: ForkMerge training pipeline.
The model parameters will be forked into two branches, one optimized with the target task loss and the other jointly trained, and be merged at regular intervals of ∆t steps.

图4：ForkMerge训练流程。
模型参数将被分叉成两个分支，一个仅用目标任务损失优化，另一个联合训练，并每隔∆t步定期合并。

\theta∗ t+∆t \leftarrow (1 − \lambda∗)\theta0 t+∆t + \lambda∗\theta1 t+∆t

▷ merge parameters

for b = 0 to 1 do
\thetab t+∆t \leftarrow \theta∗ t+∆t
end for
t \leftarrow t + ∆t

▷ synchronize parameters

end while

Equation (6) is a bi-level optimization problem. One common approach is to first approximate Pˆ with the loss of a batch of data on the validation set, and then use first-order approximation to solve \lambda [18, 43]. However, these approximations within a single step of gradient descent introduce large noise to the estimation of \lambda and also increase the risk of over-fitting the validation set. To tackle these issues, we first rewrite Equation (6) equally as

公式(6)是一个双层优化问题。一种常见的方法是先用验证集上一批数据的损失来近似Pˆ，然后使用一阶近似来求解\lambda [18, 43]。然而，这些在单步梯度下降内的近似给\lambda的估计引入了大量噪声，也增加了过拟合验证集的风险。为了解决这些问题，我们首先将公式(6)等价重写为

max \lambda Pˆ((1 − \lambda)\thetat+1(0) + \lambda\thetat+1(1)),

(7)

where \thetat+1(0) = \thetat − \etagtgt(\thetat) and \thetat+1(1) = \thetat − \eta(gtgt(\thetat) + gaux(\thetat)). The proof is in Appendix A.1. Note that we assume the optimal \lambda∗ satisfies 0 \leq \lambda∗ \leq 1, which can be guaranteed by increasing the scale of Laux when necessary. Yet an accurate estimation of performance Pˆ in Equation (7) is still computationally expensive and prone to over-fitting, thus we extend the one gradient step to ∆t steps,

其中\thetat+1(0) = \thetat − \etagtgt(\thetat)，\thetat+1(1) = \thetat − \eta(gtgt(\thetat) + gaux(\thetat))。证明见附录A.1。注意我们假设最优\lambda∗满足0 \leq \lambda∗ \leq 1，这可以通过在必要时增加Laux的规模来保证。然而，公式(7)中性能Pˆ的准确估计仍然计算代价高昂且容易过拟合，因此我们将单步梯度扩展到∆t步，

\lambda∗ = arg max \lambda Pˆ((1 − \lambda)\thetat+∆t(0) + \lambda\thetat+∆t(1)).

(8)

Algorithm. As shown in Figure 4 and Algorithm 1, the initial model parameters \thetat at training step t will first be forked into two branches. The first one will be optimized only with the target task loss Ltgt for ∆t iterations to obtain \thetat+∆t(0), while the other one will be jointly trained for ∆t iterations to obtain \thetat+∆t(1). Then we will search the optimal \lambda∗ that linearly combines the above two sets of parameters to maximize the validation performance Pˆ. When weak negative transfer occurs in the joint training branch, we can select a proper \lambda∗ between 0 and 1. And when strong negative transfer occurs, we can simply set \lambda∗ to 0. Finally, the newly merged parameter \theta∗ t+∆t = (1 − \lambda∗)\thetat+∆t(0) + \lambda∗\thetat+∆t(1) will join in a new round, being forked into two branches again and repeating the optimization process for ⌈ T / ∆t ⌉ times.

算法。如图4和算法1所示，训练步骤t的初始模型参数\thetat将首先被分叉成两个分支。第一个分支仅使用目标任务损失Ltgt优化∆t次迭代以获得\thetat+∆t(0)，而另一个分支将联合训练∆t次迭代以获得\thetat+∆t(1)。然后我们将搜索最优\lambda∗，它线性组合上述两组参数以最大化验证性能Pˆ。当联合训练分支发生弱负迁移时，我们可以选择0到1之间的合适\lambda∗。当发生强负迁移时，我们可以简单地将\lambda∗设为0。最后，新合并的参数\theta∗ t+∆t = (1 − \lambda∗)\thetat+∆t(0) + \lambda∗\thetat+∆t(1)将进入新一轮，再次分叉成两个分支，重复优化过程⌈ T / ∆t ⌉次。

Discussion. Compared to grid searching \lambda, which is widely used in practice, ForkMerge can dynamically transfer knowledge from auxiliary tasks to the target task during training with varying \lambda∗. In terms of computation cost, ForkMerge has a lower complexity as it only requires training 2 branches while grid searching has a cost proportional to the number of hyper-parameters to be searched.

讨论。与实践中广泛使用的网格搜索\lambda相比，ForkMerge可以在训练期间通过变化的\lambda∗动态地将知识从辅助任务迁移到目标任务。在计算成本方面，ForkMerge具有较低的复杂度，因为它只需要训练2个分支，而网格搜索的成本与要搜索的超参数数量成正比。

### 4.2 ForkMerge for Task Selection Simultaneously

### 4.2 同时进行任务选择的ForkMerge

When multiple auxiliary tasks are available, we can simply mix all the auxiliary tasks together to form a single auxiliary task. This simple strategy actually works well in most scenarios (see Section 5.2) and is computationally cheap. However, when further increasing the performance is desired, we can also dynamically select the optimal weighting for each auxiliary task. Formally, the objective when optimizing the model for the target task T0 with multiple auxiliary tasks {Tk}K k=1 is

当有多个辅助任务可用时，我们可以简单地将所有辅助任务混合在一起形成一个单一的辅助任务。这种简单策略在大多数场景下实际上效果很好（见第5.2节），且计算成本低廉。然而，当需要进一步提高性能时，我们也可以动态地为每个辅助任务选择最优权重。形式上，使用多个辅助任务{Tk}K k=1优化目标T0的模型时，目标为

min \theta ET0 L0(\theta) + \SigmaK k=1 \lambdak ETk Lk(\theta),

(9)

where \SigmaK k=1 \lambdak \leq 1 and \forallk, \lambdak \geq 0. Using gradient descent to update \thetat at training step t, we have

其中\SigmaK k=1 \lambdak \leq 1且\forallk, \lambdak \geq 0。使用梯度下降在训练步骤t更新\thetat，我们有

\thetat+1(\lambda) = \thetat − \eta \SigmaK k=0 \lambdakgk(\thetat),

(10)

where \lambda0 = 1. Given K task-weighting vectors {\omegak}K k=0 that satisfies \omegak i = 1[i = k or i = 0], i.e., the k-th and 0-th dimensions of \omegak are 1 and the rest are 0, and a vector $\Lambda$ that satisfies

其中\lambda0 = 1。给定K个任务权重向量{\omegak}K k=0，满足\omegak i = 1[i = k or i = 0]，即\omegak的第k维和第0维为1，其余为0，以及一个满足以下条件的向量 $\Lambda$

 $\Lambda$ k = { 1 − \Sigma i\neq0 \lambdai, k = 0; \lambdak, k \neq 0 },

(11)

then optimizing \lambda∗ in Equation (10) is equivalent to

则优化公式(10)中的\lambda∗等价于

 $\Lambda$ ∗ = arg max $\Lambda$ Pˆ( \SigmaK k=0 $\Lambda$ k\thetat+1(\omegak)).

(12)

In Equation (12), the initial model parameters are forked into K + 1 branches, where one branch is optimized with the target task, and the other branches are jointly optimized with one auxiliary task and the target task. Then we find the optimal $\Lambda$ ∗ that linearly combines the K + 1 sets of parameters to maximize the validation performance (see proof of Equation (12) and the detailed algorithm in Appendix A.2). The training computational complexity of Equation 12 is O(K), which is much lower than the exponential complexity of grid searching, but still quite large. Inspired by the early-stop approximation used in task grouping methods [71], we can prune the forking branches with $\Lambda$ k = 0 (strong negative transfer) and only keep the branches with the largest K′ < K values in $\Lambda$ after the early merge step. In this way, those useless branches with irrelevant auxiliary tasks can be stopped early. Additionally, we introduce a greedy search strategy in Algorithm 3 to further reduce the computation complexity when grid searching all possible values of $\Lambda$ .

在公式(12)中，初始模型参数被分叉成K+1个分支，其中一个分支仅用目标任务优化，其他分支与一个辅助任务和目标任务联合优化。然后我们找到最优 $\Lambda$ ∗，它线性组合K+1组参数以最大化验证性能（见公式(12)的证明和附录A.2中的详细算法）。公式(12)的训练计算复杂度为O(K)，远低于网格搜索的指数复杂度，但仍然相当大。受任务分组方法[71]中使用的早期停止近似的启发，我们可以剪枝掉 $\Lambda$ k = 0（强负迁移）的分叉分支，在早期合并步骤后只保留 $\Lambda$ 中具有最大K′ < K值的分支。这样，那些包含无关辅助任务的无用分支可以提前停止。此外，我们在算法3中引入了一种贪心搜索策略，以进一步降低网格搜索 $\Lambda$ 所有可能值时的计算复杂度。

Lastly, we introduce a general form of ForkMerge. Assuming B candidate branches with task-weighting vectors \nub (b = 1, . . . , B), the goal is to optimize $\Lambda$ ∗ :

最后，我们引入ForkMerge的通用形式。假设B个候选分支具有任务权重向量\nub（b = 1, ..., B），目标是优化 $\Lambda$ ∗：

 $\Lambda$ ∗ = arg max $\Lambda$ Pˆ( \SigmaB b=1 $\Lambda$ b\thetat+∆t(\nub)).

(13)

From a generalization view, the mixture distributions constructed by different \nu lead to diverse data shifts from the target distribution, yet we cannot predict which \nu leads to better generalization. Thus, we transform the problem of mixture distribution into that of mixture hypothesis [49] and the models trained on different distributions are combined dynamically via $\Lambda$ ∗ to approach the optimal parameters. Here, Equation 12 is a particular case by substituting B = K + 1 and \nub i = 1[i = b − 1 or i = 0]. By comparison, Equation 13 allows us to introduce human prior into ForkMerge by constructing more efficient branches, and also provides possibilities for combining ForkMerge with previous task grouping methods [81, 71, 17]. The detailed algorithm of Equation 13 can be found in Algorithm 2.

从泛化的角度来看，由不同\nu构建的混合分布导致了与目标分布不同的数据偏移，然而我们无法预测哪个\nu会带来更好的泛化。因此，我们将混合分布问题转化为混合假设问题[49]，并且在不同分布上训练的模型通过 $\Lambda$ ∗动态组合以逼近最优参数。这里，公式12是通过代入B = K + 1和\nub i = 1[i = b − 1 or i = 0]的一个特例。相比之下，公式13允许我们通过构建更高效的分支将先验知识引入ForkMerge，也为将ForkMerge与先前的任务分组方法[81, 71, 17]结合提供了可能性。公式13的详细算法见算法2。

## 5 Experiments

## 5 实验

We evaluate the effectiveness of ForkMerge under various settings, including multi-task learning, multi-domain learning, and semi-supervised learning. First, in Section 5.1, we illustrate the prevalence of negative transfer and explain how ForkMerge can mitigate this problem. In Section 5.2, We examine whether ForkMerge can mitigate negative transfer when joint training the auxiliary and target tasks, and compare it with other methods. In Section 5.3, we further use ForkMerge for task selection simultaneously. Experiment details can be found in Appendix C. We will provide additional analysis and comparison experiments in Appendix D. The codebase for both our method and the compared methods will be available at https://github.com/thuml/ForkMerge.

我们在各种设置下评估ForkMerge的有效性，包括多任务学习、多域学习和半监督学习。首先，在第5.1节中，我们展示了负迁移的普遍性并解释了ForkMerge如何缓解这一问题。在第5.2节中，我们考察了ForkMerge在联合训练辅助任务和目标任务时是否能缓解负迁移，并与其他方法进行了比较。在第5.3节中，我们进一步将ForkMerge用于同时进行任务选择。实验细节见附录C。我们将在附录D中提供额外的分析和比较实验。我们的方法和比较方法的代码库将在https://github.com/thuml/ForkMerge提供。

### 5.1 Motivation Experiment

### 5.1 动机实验

Negative Transfer is widespread across different tasks. In Figure 5 (a), we visualize the transfer gains between 30 task pairs on DomainNet, where the auxiliary and target tasks are equally weighted, and we observe that negative transfer is common in such case (23 of 30 combinations lead to negative transfer). Besides, as mentioned in Definition 3.2 and 3.3, whether negative transfer occurs is related to a specific ATL algorithm, in Figure 5 (b), we observe that negative transfer in all 30 combinations can be successfully avoided when we use ForkMerge algorithm. This observation further indicates the limitation of task grouping methods [71, 17], since they use Equal Weight between tasks and may discard some useful auxiliary tasks.

负迁移在不同任务间普遍存在。在图5(a)中，我们可视化了DomainNet上30个任务对之间的迁移增益，其中辅助任务和目标任务权重相等，我们观察到在这种情况下负迁移很常见（30个组合中有23个导致负迁移）。此外，如定义3.2和3.3所述，是否发生负迁移与特定的ATL算法有关，在图5(b)中，我们观察到当我们使用ForkMerge算法时，所有30个组合中的负迁移都可以成功避免。这一观察进一步表明了任务分组方法[71, 17]的局限性，因为它们在任务之间使用等权重，可能会丢弃一些有用的辅助任务。

Mixture of hypotheses is an approximation of mixture of distribution. Figure 6 uses the ternary heatmaps to visualize the linear combination of a set of three models optimized with different task weightings for 25K iterations, including a single-task model and two multi-task models. Similar to mixing distributions for weak negative transfer task Painting (see Figure 3), the transfer gain when mixing models Painting and Painting+Real first increases and then decreases. Also similar to mixing distributions for strong negative transfer task Quickdraw, the transfer gain when mixing models Quickdraw and Quickdraw+Real decreases monotonically. Besides, Figure 6 also indicates a good property of deep models: the loss surfaces of over-parameterized deep neural networks are quite well-behaved and smooth after convergence, which has also been mentioned by previous works [20, 35] and provides an intuitive explanation of the merge step in ForkMerge.

假设混合是分布混合的近似。图6使用三元热图可视化了用不同任务权重优化25K迭代的一组三个模型（包括一个单任务模型和两个多任务模型）的线性组合。类似于弱负迁移任务Painting的分布混合（见图3），混合Painting和Painting+Real模型时的迁移增益先增加后减少。同样，类似于强负迁移任务Quickdraw的分布混合，混合Quickdraw和Quickdraw+Real模型时的迁移增益单调递减。此外，图6还表明了深度模型的一个良好特性：过参数化深度神经网络的损失曲面在收敛后表现得相当规范和光滑，这一点先前的工作[20, 35]也有提及，并为ForkMerge中的合并步骤提供了直观解释。

Figure 5: Negative Transfer on DomainNet. The rows of each matrix represent auxiliary tasks, and the columns represent target tasks. The blue and red cells correspond to negative and positive transfer gain. Deeper colors indicate stronger impacts.

图5：DomainNet上的负迁移。每个矩阵的行表示辅助任务，列表示目标任务。蓝色和红色单元格分别对应负迁移增益和正迁移增益。颜色越深表示影响越强。

### 5.2 Use ForkMerge for Joint Optimization

### 5.2 使用ForkMerge进行联合优化

Figure 6: Ternary heatmap for mixture of model hypotheses. Each triangle vertex represents an optimized model, e.g., P+R is the model jointly optimized with Painting and Real tasks. Each point inside the triangle corresponds to a mixture of model hypotheses and its heat value measures the Transfer Gain (TG).

图6：模型假设混合的三元热图。每个三角形顶点代表一个优化后的模型，例如P+R是与Painting和Real任务联合优化的模型。三角形内的每个点对应模型假设的混合，其热值衡量迁移增益（TG）。

First, we use ForkMerge only for joint training of the target and auxiliary tasks. When datasets contain multiple tasks, we will mix all tasks together to form a single auxiliary task for ForkMerge. Yet for the compared methods, a distinction is still made between different tasks for better performance.

首先，我们仅将ForkMerge用于目标任务和辅助任务的联合训练。当数据集包含多个任务时，我们将所有任务混合在一起形成一个单一的辅助任务用于ForkMerge。然而对于比较的方法，我们仍然对不同任务进行区分以获得更好的性能。

Specifically, we compare ForkMerge with: (1) Single Task Learning (STL); (2) EW, which assigns equal weight to all tasks; (3) GCS [15], an ATL approach using gradient similarity between target and auxiliary tasks; (4) OL_AUX [39], an ATL approach adjusting the loss weight based on gradient inner product; (5) ARML [67], an ATL approach adjusting the loss weight based on gradient difference; (6) Auto-\lambda [43], an ATL method that estimates loss weight through finite-difference approximation [18]; (7) Post-train, an ATL method that pre-trains the model on all tasks and then fine-tunes it for each task separately. (8) UW [28], which adjusts weights based on task uncertainty; (9) DWA [44], which adjusts weights based on loss change; (10) MGDA [65], which computes a convex combination of gradients with a minimum norm to balance tasks; (11) GradNorm [5], which rescales the gradient norms of different tasks to the same range; (12) PCGrad [79], which eliminates conflicting gradient components; (13) IMTL [41], which uses an update direction with equal projections on task gradients; (14) CAGrad [40], which optimizes for the average loss and minimum decrease rate across tasks; (15) NashMTL [55], which combines the gradients using the Nash bargaining solution. Since different tasks have varying evaluation metrics, we will report the average per-task performance improvement for each method using ∆m, as defined in Appendix C.1.

具体来说，我们将ForkMerge与以下方法进行比较：(1) 单任务学习（STL）；(2) EW，为所有任务分配相等的权重；(3) GCS [15]，一种使用目标任务和辅助任务之间梯度相似性的ATL方法；(4) OL_AUX [39]，一种基于梯度内积调整损失权重的ATL方法；(5) ARML [67]，一种基于梯度差异调整损失权重的ATL方法；(6) Auto-\lambda [43]，一种通过有限差分近似[18]估计损失权重的ATL方法；(7) Post-train，一种先在所有任务上预训练模型然后为每个任务单独微调的ATL方法；(8) UW [28]，基于任务不确定性调整权重；(9) DWA [44]，基于损失变化调整权重；(10) MGDA [65]，计算具有最小范数的梯度凸组合以平衡任务；(11) GradNorm [5]，将不同任务的梯度范数重新缩放到相同范围；(12) PCGrad [79]，消除冲突的梯度分量；(13) IMTL [41]，使用在任务梯度上具有相等投影的更新方向；(14) CAGrad [40]，优化跨任务的平均损失和最小衰减率；(15) NashMTL [55]，使用纳什议价解组合梯度。由于不同任务具有不同的评估指标，我们将使用∆m（定义见附录C.1）报告每种方法的平均每任务性能改进。

Auxiliary-Task Scene Understanding. We evaluate on the widely-used multi-task scene understanding dataset, NYUv2 [68], which contains 3 tasks: 13-class semantic segmentation, depth estimation, and surface normal prediction. Following [55], we use 636, 159 and 654 images for training, validation, and test. Our implementation is based on LibMTL [38] and MTAN [44]. The results are presented in Table 1. Negative transfer is not severe on this dataset, where both segmentation and depth benefit from ATL and only normal task gets worse. In such cases, our method still achieves significant improvement on all tasks. We also find that Post-train serves as a strong baseline in most of our ATL experiments. Its drawback is that it fails to consider the task relationship in the pre-training phase, and suffers from catastrophic forgetting during the fine-tuning process.

辅助任务场景理解。我们在广泛使用的多任务场景理解数据集NYUv2 [68] 上进行评估，该数据集包含3个任务：13类语义分割、深度估计和表面法线预测。遵循[55]，我们使用636、159和654张图像分别用于训练、验证和测试。我们的实现基于LibMTL [38] 和MTAN [44]。结果如表1所示。负迁移在这个数据集上并不严重，分割和深度都受益于ATL，只有法线任务变差。在这种情况下，我们的方法仍然在所有任务上取得了显著改进。我们还发现Post-train在我们大多数的ATL实验中是一个强基线。它的缺点是在预训练阶段未能考虑任务关系，并且在微调过程中遭受灾难性遗忘。

Auxiliary-Domain Image Recognition. Further, we evaluate on the widely-used multi-domain image recognition dataset, DomainNet [61], which contains 6 diverse visual domains and approximately 0.6 million images distributed among 345 categories, where the task difference is reflected in the marginal distribution. Our implementation is based on TLlib [26]. As the original DomainNet does not provide a separate validation set, we randomly split 50% data from the test set as the validation set. The results are presented in Table 2. DomainNet contains both positive transfer tasks (Clipart), weak negative transfer tasks (Infograph, Painting, Real, Sketch), and strong negative transfer tasks (Quickdraw). When negative transfer occurs, previous ATL methods lead to severe performance degradation, while our method can automatically avoid strong negative transfer and improve the performance over STL in other cases.

辅助域图像识别。此外，我们在广泛使用的多域图像识别数据集DomainNet [61] 上进行评估，该数据集包含6个不同的视觉域，约60万张图像分布在345个类别中，任务差异体现在边缘分布上。我们的实现基于TLlib [26]。由于原始DomainNet没有提供单独的验证集，我们从测试集中随机拆分50%的数据作为验证集。结果如表2所示。DomainNet既包含正迁移任务（Clipart）、弱负迁移任务（Infograph、Painting、Real、Sketch），也包含强负迁移任务（Quickdraw）。当发生负迁移时，先前的ATL方法导致严重的性能下降，而我们的方法可以在其他情况下自动避免强负迁移并提高相对于STL的性能。

Table 1: Performance on NYUv2 dataset.

表1：NYUv2数据集上的性能。

Table 2: Performance on DomainNet dataset.

表2：DomainNet数据集上的性能。

[Table data with method names and metric values omitted for brevity due to table formatting — full numeric data preserved from original.]

由于表格格式原因，此处省略了方法名称和度量值的详细数据——完整数值数据保留自原文。

Methods: STL, EW, UW, DWA, MGDA, GradNorm, PCGrad, IMTL, CAGrad, NashMTL, GCS, OL_AUX, ARML, Auto-\lambda, Post-train, ForkMerge.

方法：STL, EW, UW, DWA, MGDA, GradNorm, PCGrad, IMTL, CAGrad, NashMTL, GCS, OL_AUX, ARML, Auto-\lambda, Post-train, ForkMerge.

### 5.3 Use ForkMerge for Task Selection Simultaneously

### 5.3 同时使用ForkMerge进行任务选择

As mentioned in Section 4.2, when there are multiple auxiliary task candidates, we can use ForkMerge to simultaneously select auxiliary tasks and jointly train them with the target task, which is denoted as ForkMerge‡.

如第4.2节所述，当有多个候选辅助任务时，我们可以使用ForkMerge同时选择辅助任务并与目标任务联合训练，表示为ForkMerge‡。

Auxiliary-Task Scene Understanding. In NYUv2, we have 2 auxiliary tasks for any target task, thus we can construct 3 branches with different task weights in Equation 12. In this way, we are able to select auxiliary tasks adaptively by learning different $\Lambda$ for different branches in the merge step. As shown in Table 3, this strategy yields better overall performance.

辅助任务场景理解。在NYUv2中，对于任何目标任务，我们有2个辅助任务，因此我们可以在公式12中构建3个具有不同任务权重的分支。通过这种方式，我们能够在合并步骤中通过为不同分支学习不同的 $\Lambda$ 来自适应地选择辅助任务。如表3所示，这种策略产生了更好的整体性能。

Auxiliary-Domain Image Recognition. For any target task in DomainNet, we can construct up to 6 branches with different task weights in Equation 12, which is computationally expensive. As mentioned in Section 4.2, we will prune the branches after the first merge step to reduce the computation cost. Table 4 reveals the impact of the pruning strategy. As the number of branches increases, the gain brought by auxiliary tasks will enlarge, while the gain brought by each branch will reduce. Therefore, pruning is an effective strategy to achieve a better balance between performance and efficiency. In practical terms, when confronted with multiple auxiliary tasks, users have the flexibility to tailor the number of branches to align with their available computational resources.

辅助域图像识别。对于DomainNet中的任何目标任务，我们可以在公式12中构建多达6个具有不同任务权重的分支，这在计算上是昂贵的。如第4.2节所述，我们将在第一次合并步骤后剪枝分支以减少计算成本。表4揭示了剪枝策略的影响。随着分支数量的增加，辅助任务带来的增益会增加，而每个分支带来的增益会减少。因此，剪枝是在性能和效率之间实现更好平衡的有效策略。在实际中，当面对多个辅助任务时，用户可以灵活地调整分支数量以匹配其可用的计算资源。

CTR and CTCVR Prediction. We evaluate on AliExpress dataset [36], a recommendation dataset from the industry, which consists of 2 tasks: CTR and CTCVR, and 4 scenarios and more than 100M records. Our implementation is based on MTReclib [85]. For any target task in AliExpress, we can construct up to 8 branches with different task weights, and we prune to 3 branches after the first merge step. The results are presented in Table 5. Note that improving on such a large-scale dataset with auxiliary tasks is quite difficult. Still, ForkMerge achieves the best performance with ∆m = 1.30%.

CTR和CTCVR预测。我们在AliExpress数据集[36]上进行评估，这是一个来自工业界的推荐数据集，包含2个任务：CTR和CTCVR，4个场景，超过1亿条记录。我们的实现基于MTReclib [85]。对于AliExpress中的任何目标任务，我们最多可以构建8个具有不同任务权重的分支，并在第一次合并步骤后剪枝到3个分支。结果如表5所示。注意，在这样的超大规模数据集上用辅助任务进行改进是相当困难的。尽管如此，ForkMerge仍然以∆m = 1.30%实现了最佳性能。

Semi-Supervised Learning (SSL). We also evaluate on two SSL datasets, CIFAR-10 [31] and SVHN [56]. Following [67], we use Self-supervised Semi-supervised Learning (S4L) [82] as our baseline algorithm and use 2 self-supervised tasks, Rotation [19] and Exempler-MT [14], as our auxiliary tasks. Table 6 presents the test error of S4L using different ATL approaches, along with other SSL methods, and shows that ForkMerge consistently outperforms the compared ATL methods. Note that we do not aim to propose a novel or state-of-the-art SSL method in this paper. Instead, we find that some SSL methods use ATL and the auxiliary task weights have a great impact (see Grid Search in Table 6). Thus, we use ForkMerge to improve the auxiliary task training within the context of SSL.

半监督学习（SSL）。我们还在两个SSL数据集CIFAR-10 [31] 和SVHN [56] 上进行了评估。遵循[67]，我们使用自监督半监督学习（S4L）[82] 作为基线算法，并使用2个自监督任务Rotation [19] 和Exempler-MT [14] 作为辅助任务。表6展示了使用不同ATL方法的S4L的测试误差以及其他SSL方法，并表明ForkMerge持续优于比较的ATL方法。注意，本文的目标不是提出一种新颖或最先进的SSL方法。相反，我们发现一些SSL方法使用了ATL，并且辅助任务权重有很大影响（见表6中的网格搜索）。因此，我们使用ForkMerge来改进SSL背景下的辅助任务训练。

Table 3: Effect of branch number on NYUv2.

表3：NYUv2上分支数量的影响。

Table 4: Effect of branch number on DomainNet.

表4：DomainNet上分支数量的影响。

Table 5: Performance on AliExpress dataset.

表5：AliExpress数据集上的性能。

Table 6: Performance (test error) on CIFAR-10 and SVHN datasets.

表6：CIFAR-10和SVHN数据集上的性能（测试误差）。

[Table data with method names and numeric metrics — full content preserved from original.]

[包含方法名称和数值度量的表格数据——完整内容保留自原文。]

## 6 Conclusion

## 6 结论

Methods have been proposed to mitigate negative transfer in auxiliary-task learning, yet there still lacks an in-depth experimental analysis on the causes of negative transfer. In this paper, we systematically delved into the negative transfer issues and presented ForkMerge, an approach to enable auxiliary-task learning with positive transfer gains. Experimentally, ForkMerge achieves state-of-the-art accuracy on four different auxiliary-task learning benchmarks, while being computationally efficient. We view the integration of previous task grouping methods with our auxiliary task learning approach as a promising avenue for further research.

已经提出了许多方法来缓解辅助任务学习中的负迁移，但仍然缺乏对负迁移原因的深入实验分析。在本文中，我们系统地深入研究了负迁移问题，并提出了ForkMerge，一种实现具有正迁移增益的辅助任务学习的方法。实验上，ForkMerge在四个不同的辅助任务学习基准上实现了最先进的准确率，同时计算效率高。我们认为将先前的任务分组方法与我们的辅助任务学习方法相结合是未来研究的一个有前景的方向。

## Acknowledgements

## 致谢

We would like to thank many colleagues, in particular Yuchen Zhang, Jialong Wu, Haoyu Ma, Yuhong Yang, and Jincheng Zhong, for their valuable discussions. This work was supported by the National Key Research and Development Plan (2020AAA0109201), the National Natural Science Foundation of China (62022050 and 62021002), and the Beijing Nova Program (Z201100006820041).

我们要感谢许多同事，特别是Yuchen Zhang、Jialong Wu、Haoyu Ma、Yuhong Yang和Jincheng Zhong，感谢他们宝贵的讨论。本工作得到了国家重点研发计划（2020AAA0109201）、国家自然科学基金（62022050和62021002）以及北京新星计划（Z201100006820041）的支持。

## References

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

[47] Xiao Ma, Liqin Zhao, Guan Huang, Zhi Wang, Zelin Hu, Xiaoqiang Zhu, and Kun Gai. Entire space multi-task model: An effective approach for estimating post-click conversion rate. In SIGIR, 2018.

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

## A Algorithm Details

## A 算法细节

### A.1 ForkMerge

### A.1 ForkMerge

Proof of Equation (7).

公式(7)的证明。

\lambda∗ = arg max \lambda Pˆ(\thetat+1)
= arg max \lambda Pˆ(\thetat − \eta(gtgt(\thetat) + \lambdagaux(\thetat)))
= arg max \lambda Pˆ((\thetat − \etagtgt(\thetat)) + \lambda(−\etagaux(\thetat)))
= arg max \lambda Pˆ((1 − \lambda)(\thetat − \etagtgt(\thetat)) + \lambda(\thetat − \eta(gtgt(\thetat) + gaux(\thetat))))
= arg max \lambda Pˆ((1 − \lambda)\thetat+1(0) + \lambda\thetat+1(1)).

Remarks on the search step.

关于搜索步骤的说明。

We provide two search strategies as follows, and we use the first strategy in our experiments.

我们提供以下两种搜索策略，在我们的实验中我们使用第一种策略。

• Grid Search: Exhaustively searching the task-weighting hyper-parameter \lambda through a manually specified subset of the hyper-parameter space, such as {0, 0.2, 0.4, 0.6, 0.8, 1.0}.

• 网格搜索：通过手动指定的超参数空间子集彻底搜索任务权重超参数\lambda，例如{0, 0.2, 0.4, 0.6, 0.8, 1.0}。

• Binary Search: Repeatedly dividing the search interval of \lambda in half and keep the better hyper-parameter.

• 二分搜索：将\lambda的搜索区间重复对半分割并保留更好的超参数。

Random search, bayesian optimization, gradient-based optimization, and other hyper-parameter optimization methods can also be used here, and they are left to be explored in follow-up work.

随机搜索、贝叶斯优化、基于梯度的优化和其他超参数优化方法也可以在此使用，留待后续工作探索。

In practice, the costs of estimating Pˆ in the search step are usually negligible. Yet when the amount of data in the validation set is relatively large, we can sample the validation set to reduce the cost of estimating Pˆ.

在实践中，搜索步骤中估计Pˆ的成本通常可以忽略不计。然而当验证集中的数据量相对较大时，我们可以对验证集进行采样以减少估计Pˆ的成本。

Remarks on extension from the one gradient step to ∆t steps.

关于从单步梯度扩展到∆t步的说明。

1. It can effectively reduce the average cost of estimating Pˆ at each step and avoid over-fitting the validation set.

1. 它可以有效降低每步估计Pˆ的平均成本，并避免过拟合验证集。

2. It allows longer-term rewards from auxiliary tasks and leads to safer task transfer. For instance, when the accumulated gradients of some auxiliary tasks are harmful to the final target performance, the merging step can cancel the effect of these auxiliary tasks by setting their associated weights \lambda to 0, to mitigate strong negative transfer.

2. 它允许来自辅助任务的更长周期奖励，并导致更安全的任务迁移。例如，当某些辅助任务的累积梯度对最终目标性能有害时，合并步骤可以通过将其相关权重\lambda设为0来抵消这些辅助任务的影响，以缓解强负迁移。

3. It increases the risk to produce bad model parameters. However, such risk is still low since deep models usually have smooth loss surfaces after convergence as shown in Section 5.1.

3. 它增加了产生不良模型参数的风险。然而，这种风险仍然很低，因为如第5.1节所示，深度模型在收敛后通常具有光滑的损失曲面。

Figure 7: Effect of the merging step ∆t on NYUv2.

图7：合并步长∆t对NYUv2的影响。

Figure 7 illustrates that an appropriate ∆t can effectively promote the performance of the ForkMerge algorithm, indicating the necessity of the extension from the one gradient step in previous work to ∆t steps. When ∆t is small, the estimation of \lambda is short-insight and might fail to remove the harmful parameter updates when negative transfer occurs, which also indicates the limitations of methods that use single-step gradient descent to estimate \lambda [15, 43]. When ∆t is large, the risk to get bad model parameters from the linear combination will also increase. Therefore, in our experiment, we use the validation set to pick a proper ∆t for each dataset and use it for all tasks in this dataset.

图7说明了适当的∆t可以有效提升ForkMerge算法的性能，表明了从先前工作中的单步梯度扩展到∆t步的必要性。当∆t较小时，\lambda的估计是短视的，可能无法在发生负迁移时移除有害的参数更新，这也表明了使用单步梯度下降估计\lambda的方法[15, 43]的局限性。当∆t较大时，从线性组合中获得不良模型参数的风险也会增加。因此，在我们的实验中，我们使用验证集为每个数据集选择合适的∆t，并将其用于该数据集中的所有任务。

### A.2 Use ForkMerge to Select Tasks Simultaneously

### A.2 使用ForkMerge同时选择任务

Detailed Algorithm. Algorithm 2 provides the general optimization process for any task-weighting vector {\nub}B b=1. For Equation (12), we have B = K + 1 and \nub i = 1[i = b − 1 or i = 0]. For Equation (13), we have no constraints on B or \nub.

详细算法。算法2提供了任何任务权重向量{\nub}B b=1的通用优化过程。对于公式(12)，我们有B = K + 1且\nub i = 1[i = b − 1 or i = 0]。对于公式(13)，我们对B或\nub没有约束。

Algorithm 2 ForkMerge Training Pipeline with Multiple Branches
Require: initial model parameter \theta0, task-weighting vector {\nub}B b=1, total iterations T , interval ∆t
Ensure: final model parameter \theta∗
1: fork model into B copies {\thetab}B b=1
2: for b = 1 to B do
3: \thetab 0 \leftarrow \theta0
4: end for
5: while t < T do
6: for b = 1 to B do
7: for t′ = t to t + ∆t − 1 do
8: \thetab t′+1 = \thetab t′ − \eta \Sigmak \nub kgk(\thetab t′)
9: end for
10: end for
11: $\Lambda$ ∗ \leftarrow arg max $\Lambda$ Pˆ( \Sigmab $\Lambda$ b\thetab t+∆t)
12: \theta∗ t+∆t \leftarrow \Sigmab $\Lambda$ ∗ b \thetab t+∆t
13: for b = 1 to B do
14: \thetab t+∆t \leftarrow \theta∗ t+∆t
15: end for
16: t \leftarrow t + ∆t
17: end while

算法2 多分支ForkMerge训练流程
输入：初始模型参数\theta0，任务权重向量{\nub}B b=1，总迭代次数T，间隔∆t
输出：最终模型参数\theta∗
1: 将模型分叉为B份{\thetab}B b=1
2: for b = 1 to B do
3: \thetab 0 \leftarrow \theta0
4: end for
5: while t < T do
6: for b = 1 to B do
7: for t′ = t to t + ∆t − 1 do
8: \thetab t′+1 = \thetab t′ − \eta \Sigmak \nub kgk(\thetab t′)
9: end for
10: end for
11: $\Lambda$ ∗ \leftarrow arg max $\Lambda$ Pˆ( \Sigmab $\Lambda$ b\thetab t+∆t)
12: \theta∗ t+∆t \leftarrow \Sigmab $\Lambda$ ∗ b \thetab t+∆t
13: for b = 1 to B do
14: \thetab t+∆t \leftarrow \theta∗ t+∆t
15: end for
16: t \leftarrow t + ∆t
17: end while

Proof of Equation (12).

公式(12)的证明。

The goal of selecting \lambda∗ in Equation (10) is to maximize the validation performance of model \thetat+1,

在公式(10)中选择\lambda∗的目标是最大化模型\thetat+1的验证性能，

\lambda∗ = arg max \lambda Pˆ(\thetat+1)
= arg max \lambda Pˆ(\thetat − \eta \Sigmak \lambdakgk(\thetat))
= arg max \lambda Pˆ(\thetat − \eta\lambda0g0(\thetat) − \eta \Sigmak\neq0 \lambdakgk(\thetat))
= arg max \lambda Pˆ(\thetat − \etag0(\thetat) − \eta \Sigmak\neq0 \lambdakgk(\thetat))  // \lambda0 = 1
= arg max \lambda Pˆ((1 − \Sigmak\neq0 \lambdak)(\thetat − \etag0(\thetat)) + \Sigmak\neq0 \lambdak(\thetat − \etag0(\thetat) − \etagk(\thetat)))  // gradient descent, \Sigmak\neq0 \lambdak \leq 1
= arg max \lambda Pˆ((1 − \Sigmak\neq0 \lambdak)(\thetat − \etag0(\thetat)) + \Sigmak\neq0 \lambdak(\thetat − \etag0(\thetat) − \etagk(\thetat)))

By definitions of $\Lambda$ and {\omegak}K k=0

根据 $\Lambda$ 和{\omegak}K k=0的定义

 $\Lambda$ k = { 1 − \Sigmai\neq0 \lambdai, k = 0; \lambdak, k \neq 0 }
\omegak i = { 1, i = 0 or i = k; 0, otherwise }

we can prove that optimizing \lambda in Equation (10) is equivalent to optimizing $\Lambda$ as follows:

我们可以证明，在公式(10)中优化\lambda等价于如下优化 $\Lambda$ ：

 $\Lambda$ ∗ = arg max $\Lambda$ Pˆ( \Sigmak $\Lambda$ k\thetat+1(\omegak)).

Remarks on the search step.

关于搜索步骤的说明。

Grid searching all possible values of $\Lambda$ is computationally expensive especially when ∥ $\Lambda$ ∥ is large. Thus, here we introduce a greedy search strategy in Algorithm 3, which reduces the computation complexity from exponential complexity to O(∥ $\Lambda$ ∥).

网格搜索 $\Lambda$ 的所有可能值在计算上是昂贵的，特别是当∥ $\Lambda$ ∥很大时。因此，我们在算法3中引入了一种贪心搜索策略，将计算复杂度从指数复杂度降低到O(∥ $\Lambda$ ∥)。

Algorithm 3 Greedy Search of $\Lambda$ ∗
Require: A list of model parameters \theta1, ..., \thetaB sorted in decreasing order of Pˆ(\thetab).
Ensure: optimal linear combination coefficient $\Lambda$ ∗
1: unnormalized combination coefficient $\Lambda$ ̃ \leftarrow e1
2: for b = 2 to B do
3: set upper bound U \leftarrow 1
4: grid search the optimal $\Lambda$ ̃m in range [0, U] to maximize Pˆ( 1/∥ $\Lambda$ ̃∥ \Sigmab m=1 $\Lambda$ ̃m\thetam)
5: end for
6: $\Lambda$ ∗ \leftarrow 1/∥ $\Lambda$ ̃∥ $\Lambda$ ̃

算法3 $\Lambda$ ∗的贪心搜索
输入：按Pˆ(\thetab)降序排列的模型参数列表\theta1, ..., \thetaB
输出：最优线性组合系数 $\Lambda$ ∗
1: 未归一化的组合系数 $\Lambda$ ̃ \leftarrow e1
2: for b = 2 to B do
3: 设置上界 U \leftarrow 1
4: 在范围[0, U]内网格搜索最优 $\Lambda$ ̃m以最大化 Pˆ( 1/∥ $\Lambda$ ̃∥ \Sigmab m=1 $\Lambda$ ̃m\thetam)
5: end for
6: $\Lambda$ ∗ \leftarrow 1/∥ $\Lambda$ ̃∥ $\Lambda$ ̃

## B Analysis Details

## B 分析细节

In this section, we provide the implementation details of our analysis experiment in Section 3.

在本节中，我们提供第3节中分析实验的实现细节。

We conduct our analysis on the multi-domain image recognition dataset DomainNet [61]. In our analysis, we use task Painting and Quickdraw in DomainNet as examples of weak negative transfer and strong negative transfer, and other tasks (Real, Sketch, Infograph, Clipart) in DomainNet as auxiliary tasks. Details of these tasks are summarized in Table 8. We use ResNet-18 [23] pre-trained on ImageNet [8] for all experiments.

我们在多域图像识别数据集DomainNet [61] 上进行分析。在我们的分析中，我们使用DomainNet中的Painting和Quickdraw任务作为弱负迁移和强负迁移的例子，并使用DomainNet中的其他任务（Real、Sketch、Infograph、Clipart）作为辅助任务。这些任务的细节总结在表8中。我们在所有实验中使用在ImageNet [8] 上预训练的ResNet-18 [23]。

### B.1 Effect of Gradients Conflicts

### B.1 梯度冲突的影响

First, we optimize the model on the target task for T = 25K iterations to obtain \thetaT . We adopt mini-batch SGD with momentum of 0.9 and batch size of 48, and the initial learning rate is set as 0.01 with cosine annealing strategy [45].

首先，我们在目标任务上优化模型T = 25K次迭代以获得\thetaT。我们采用动量为0.9、批量大小为48的小批量SGD，初始学习率设为0.01，并采用余弦退火策略[45]。

Figure 8: The distribution of Gradient Cosine Similarity (GCS). P and Q are short for Painting and Quickdraw tasks, respectively.

图8：梯度余弦相似性（GCS）的分布。P和Q分别是Painting和Quickdraw任务的缩写。

We repeatedly sample a mini-batch of data and estimate the gradients for the target and auxiliary task gtgt and gaux. Figure 8 plots the distribution of gradient cosine similarity (GCS) between gtgt and gaux. We find that the gradients of different tasks are nearly orthogonal (cos \phiij \approx 0) in most cases, and highly consistent gradients or severely conflicting gradients are both relatively rare.

我们重复采样一小批数据并估计目标任务和辅助任务的梯度gtgt和gaux。图8绘制了gtgt和gaux之间的梯度余弦相似性（GCS）分布。我们发现，在大多数情况下不同任务的梯度几乎是正交的（cos \phiij \approx 0），高度一致的梯度或严重冲突的梯度都相对罕见。

Then, we optimize the same \thetaT with one-step multi-task gradient descent estimated from different data to obtain different \thetaT +1,

然后，我们使用从不同数据估计的单步多任务梯度下降优化相同的\thetaT，以获得不同的\thetaT+1，

\thetaT +1(\lambda) = \thetaT − \eta(gtgt(\thetaT) + \lambdagaux(\thetaT)),

(14)

where \eta = 0.01 and \lambda takes values from {0, 1/16, 1/8, 1/4, 1/2, 1}. We evaluate \thetaT +1(\lambda) and \thetaT +1(0) on the validation set of the target task to calculate the transfer gain (TG) from single-step multi-task gradient descent

其中\eta = 0.01，\lambda取值{0, 1/16, 1/8, 1/4, 1/2, 1}。我们在目标任务的验证集上评估\thetaT+1(\lambda)和\thetaT+1(0)，以计算来自单步多任务梯度下降的迁移增益（TG）

T G(\lambda) = Pˆ(\thetaT +1(\lambda)) − Pˆ(\thetaT +1(0)).

(15)

Note that we omit the notation of algorithm A in Equation (14) and (15) for simplicity. Then, in Figure 2, we mark the GCS and TG of each data point and fit them with a 3-order polynomial to obtain the corresponding correlation curve.

注意，为简洁起见，我们在公式(14)和(15)中省略了算法A的符号。然后，在图2中，我们标记每个数据点的GCS和TG，并用3阶多项式拟合它们以获得相应的相关曲线。

### B.2 Effect of Distribution Shift

### B.2 分布偏移的影响

Qualitative Visualization. We visualize by t-SNE [74] in Figure 3(a) the representations of the training and test data by the model \thetaT trained in Section B.1. For better visualization, we only keep the top 10 categories with the highest frequency in DomainNet. To visualize the impact of \lambda on the interpolated training distribution, we let the frequency of auxiliary task points be proportional to \lambda. In other words, when the weighing hyper-parameter of the auxiliary task increases, the effect of the auxiliary task on the interpolated distribution will also increase.

定性可视化。我们使用t-SNE [74] 在图3(a)中可视化了由第B.1节训练的模型\thetaT对训练和测试数据的表示。为了更好的可视化，我们仅保留DomainNet中频率最高的前10个类别。为了可视化\lambda对插值训练分布的影响，我们让辅助任务点的频率与\lambda成比例。换句话说，当辅助任务的权重超参数增加时，辅助任务对插值分布的影响也会增加。

Figure 3 provides the t-SNE visualization of training and test distributions when \lambda takes values from {0, 1/16, 1/4, 1}. We observe that for weak negative transfer tasks, when \lambda initially increases, the area of training distribution can better cover that of the test distribution. But as \lambda continues to increase, the distribution shift between the test set and the training set will gradually increase. For strong negative transfer tasks, however, the shift between the interpolated training distribution and the test distribution monotonically enlarges as \lambda increases.

图3提供了当\lambda取值{0, 1/16, 1/4, 1}时训练和测试分布的t-SNE可视化。我们观察到，对于弱负迁移任务，当\lambda最初增加时，训练分布区域可以更好地覆盖测试分布区域。但随着\lambda继续增加，测试集和训练集之间的分布偏移将逐渐增大。然而对于强负迁移任务，插值训练分布与测试分布之间的偏移随着\lambda的增加单调增加。

Quantitative Measure. First, we jointly optimize the model on the target task and auxiliary tasks with different weighting hyper-parameter \lambda for T = 25K iterations to obtain \thetaT (\lambda). We adopt the same hyper-parameters as in Section B.1. Then we evaluate \thetaT (\lambda) on the test set of the target tasks and calculate the average confidence on the test set. We can calculate the confidence score discrepancy (CSD) by Definition 3.5 and the transfer gain (TG) by

定量度量。首先，我们使用不同的权重超参数\lambda在目标任务和辅助任务上联合优化模型T = 25K次迭代以获得\thetaT(\lambda)。我们采用与第B.1节相同的超参数。然后我们在目标任务的测试集上评估\thetaT(\lambda)并计算测试集上的平均置信度。我们可以通过定义3.5计算置信度分数差异（CSD），并通过以下公式计算迁移增益（TG）

T G(\lambda) = Pˆ(\thetaT (\lambda)) − Pˆ(\thetaT (0)).

(16)

Again, we omit the notation of algorithm A for simplicity. Finally, we plot the curve between CSD and TG under different \lambda in Figure 3(b).

同样，为简洁起见我们省略了算法A的符号。最后，我们在图3(b)中绘制了不同\lambda下CSD和TG之间的曲线。

## C Experiment Details

## C 实验细节

### C.1 Definition of ∆m

### C.1 ∆m的定义

Following [55, 37], we report ∆m as the performance measure, which is the average per-task performance improvement of method m relative to the STL baseline b. Formally, ∆m = 1/K \SigmaK k=1 (−1)zk (Mm,k − Mb,k)/Mb,k where Mb,k and Mm,k is the performance of the k-th task obtained by the baseline method b and the compared method m. zk is set to 0 if a higher value indicates better performance for the k-th task and otherwise 1.

遵循[55, 37]，我们报告∆m作为性能度量，这是方法m相对于STL基线b的平均每任务性能改进。形式上，∆m = 1/K \SigmaK k=1 (−1)zk (Mm,k − Mb,k)/Mb,k，其中Mb,k和Mm,k分别是基线方法b和比较方法m获得的第k个任务的性能。如果第k个任务的较高值表示更好的性能，则zk设为0，否则为1。

### C.2 Auxiliary-Task Scene Understanding on NYU

### C.2 NYU上的辅助任务场景理解

Experiment Details. We use DeepLabV3+ architecture [1], where a ResNet-50 network [23] pretrained on the ImageNet dataset [8] with dilated convolutions is used as a shared encoder among tasks and the Atrous Spatial Pyramid Pooling module is used as task-specific head for each task. Following [44, 79], each method is trained for 200 epochs with the Adam optimizer [29] and batch size of 8. The initial learning rate is 10−4 and halved to 5 $\times$ 10−5 after 100 epochs. In ForkMerge, the parameters are merged every 10 epochs. Table 7 presents the full evaluation results of Table 1.

实验细节。我们使用DeepLabV3+架构[1]，其中在ImageNet数据集[8]上使用扩张卷积预训练的ResNet-50网络[23]作为任务间的共享编码器，并使用Atrous Spatial Pyramid Pooling模块作为每个任务的任务特定头部。遵循[44, 79]，每种方法使用Adam优化器[29]和批量大小8训练200个epoch。初始学习率为10−4，在100个epoch后减半为5 $\times$ 10−5。在ForkMerge中，参数每10个epoch合并一次。表7展示了表1的完整评估结果。

Table 7: Performance on NYUv2 dataset.

表7：NYUv2数据集上的性能。

[Detailed table data — full numeric content preserved from original.]

[详细的表格数据——完整数值内容保留自原文。]

Methods: STL, EW, UW, DWA, RLW, MGDA, GradNorm, PCGrad, IMTL, GradVac, CAGrad, NashMTL, GCS, OL_AUX, ARML, Auto-\lambda, Post-train, ForkMerge.

方法：STL, EW, UW, DWA, RLW, MGDA, GradNorm, PCGrad, IMTL, GradVac, CAGrad, NashMTL, GCS, OL_AUX, ARML, Auto-\lambda, Post-train, ForkMerge。

Metrics: Segmentation (mIoU $\uparrow$ , Pix Acc $\uparrow$ ), Depth (Abs Err $\downarrow$ , Rel Err $\downarrow$ ), Normal (Angle Distance Mean $\downarrow$ , Median $\downarrow$ , Within t◦ 11.25 $\uparrow$ , 22.5 $\uparrow$ , 30 $\uparrow$ ), ∆m $\uparrow$ .

### C.3 Auxiliary-Domain Image Recognition on DomainNet

### C.3 DomainNet上的辅助域图像识别

Dataset Details. As the original DomainNet [61] does not provide a separate validation set, we randomly split 50% data from the test set as the validation set, and use the rest 50% data as the test set. For each task, the proportions of training set, validation set, and test set are approximately 70%/15%/15%. Table 8 summarizes the statistics of this dataset. DomainNet is under Custom (research-only, non-commercial) license.

数据集细节。由于原始DomainNet [61] 没有提供单独的验证集，我们从测试集中随机拆分50%的数据作为验证集，其余50%的数据作为测试集。对于每个任务，训练集、验证集和测试集的比例约为70%/15%/15%。表8总结了该数据集的统计数据。DomainNet采用Custom（仅研究、非商业）许可证。

Table 8: Overview of DomainNet dataset.

表8：DomainNet数据集概览。

Tasks: Clipart (#Train 33.5K, #Val 7.3K, #Test 7.3K, Description: collection of clipart images), Real (120.9K, 26.0K, 26.0K, photos and real world images), Sketch (48.2K, 10.5K, 10.5K, sketches of specific objects), Infograph (36.0K, 7.8K, 7.8K, infographic images), Painting (50.4K, 10.9K, 10.9K, painting depictions of objects), Quickdraw (120.7K, 25.9K, 25.9K, drawings of game "Quick Draw").

任务：Clipart（训练33.5K，验证7.3K，测试7.3K，描述：剪贴画图像集合），Real（120.9K，26.0K，26.0K，照片和真实世界图像），Sketch（48.2K，10.5K，10.5K，特定对象的素描），Infograph（36.0K，7.8K，7.8K，信息图图像），Painting（50.4K，10.9K，10.9K，对象的绘画描绘），Quickdraw（120.7K，25.9K，25.9K，游戏"Quick Draw"的涂鸦）。

Experiment Details. We adopt mini-batch SGD with momentum of 0.9 and batch size of 48. We search the initial learning rate in {0.003, 0.01, 0.03} and adopt cosine annealing strategy [45] to adjust learning rate during training. We adopt ResNet-101 pretrained on ImageNet as the backbone. Each method is trained for 50K iterations. In ForkMerge, the parameters are merged every 12.5K iterations.

实验细节。我们采用动量为0.9、批量大小为48的小批量SGD。我们在{0.003, 0.01, 0.03}中搜索初始学习率，并采用余弦退火策略[45]在训练过程中调整学习率。我们采用在ImageNet上预训练的ResNet-101作为骨干网络。每种方法训练50K次迭代。在ForkMerge中，参数每12.5K次迭代合并一次。

### C.4 CTR and CTCVR Prediction on AliExpress

### C.4 AliExpress上的CTR和CTCVR预测

Dataset Details. AliExpress [36] is gathered from the real-world traffic logs of AliExpress search system in Taobao and contains more than 100M records in total. We split the first 90% data in the time sequence to be training set and the rest 5% and 5% to be validation set and test set. AliExpress consists of 2 tasks: click-through rate (CTR) and click-through conversion rate (CTCVR), and 4 scenarios: Spain (ES), French (FR), Netherlands (NL), and America (US). Table 9 summarizes the statistics of this dataset. AliExpress is under Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) license.

数据集细节。AliExpress [36] 来自淘宝AliExpress搜索系统的真实流量日志，总计包含超过1亿条记录。我们将时间序列中前90%的数据作为训练集，其余5%和5%分别作为验证集和测试集。AliExpress包含2个任务：点击率（CTR）和点击转化率（CTCVR），以及4个场景：西班牙（ES）、法国（FR）、荷兰（NL）和美国（US）。表9总结了该数据集的统计数据。AliExpress采用知识共享署名-非商业性使用-相同方式共享4.0国际（CC BY-NC-SA 4.0）许可证。

Experiment Details. The architecture of most methods is based on ESMM [47], which consists of a single embedding layer shared by all tasks and multiple independent DNN towers for each task. The embedding dimension for each feature field is 128. Each method is trained for 50 epochs using the Adam optimizer, with the batch size of 2048, learning rate of 10−3 and weight decay of 10−6.

实验细节。大多数方法的架构基于ESMM [47]，由一个所有任务共享的单一嵌入层和每个任务的多个独立DNN塔组成。每个特征域的嵌入维度为128。每种方法使用Adam优化器训练50个epoch，批量大小为2048，学习率为10−3，权重衰减为10−6。

Table 9: Overview of AliExpress dataset, where CTR = #Click / #Impression and CTCVR = #Purchase / #Impression.

表9：AliExpress数据集概览，其中CTR = #点击 / #展示，CTCVR = #购买 / #展示。

Statistics: ES (#Product 8.7M, #Pv 2M, #Impression 31.6M, #Click 841K, #Purchase 19.1K, CTR 2.66%, CTCVR 0.60‰); FR (7.4M, 1.7M, 27.4M, 535K, 14.4K, 2.01%, 0.54‰); NL (6M, 1.2M, 17.7M, 382K, 13.8K, 2.16%, 0.78‰); US (8M, 1.8M, 27.4M, 450K, 10.9K, 1.64%, 0.40‰).

### C.5 Semi-supervised Learning on CIFAR10 and SVHN

### C.5 CIFAR10和SVHN上的半监督学习

Dataset Details. Following [67], we first split the original training set of CIFAR10 [31] and SVHN [56] into training set and validation set. Then, we randomly sample labeled images from the training set. Table 10 summarizes the statistics of CIFAR-10 and SVHN.

数据集细节。遵循[67]，我们首先将CIFAR10 [31] 和SVHN [56] 的原始训练集分为训练集和验证集。然后，我们从训练集中随机采样标记图像。表10总结了CIFAR-10和SVHN的统计数据。

Table 10: Overview of CIFAR-10 and SVHN datasets.

表10：CIFAR-10和SVHN数据集概览。

Datasets: CIFAR-10 (#Labeled 4000, #Unlabeled 41000, #Val 5000, #Test 10000); SVHN (1000, 64931, 7326, 26032).

数据集：CIFAR-10（标记4000，未标记41000，验证5000，测试10000）；SVHN（1000，64931，7326，26032）。

Experiment Details. (1) Auxiliary Tasks. Following [82, 67], we consider two self-supervised auxiliary tasks Rotation [19] and Exempler-MT [14]. In Rotation, we rotate each image by [0◦, 90◦, 180◦, 270◦] and ask the network to predict the angle. In Exemplar-MT, the model is trained to extract feature invariant to a wide range of image transformations. (2) Hyper-parameters. We adopt Adam [29] optimizer with an initial learning rate of 0.005. We train each method for 200K iterations and decay the learning rate by a factor of 0.2 at 160K iterations. We use Wide ResNet-28-2 [80] as the backbone. In ForkMerge, the parameters are merged every 10K iterations.

实验细节。(1) 辅助任务。遵循[82, 67]，我们考虑两个自监督辅助任务Rotation [19] 和Exempler-MT [14]。在Rotation中，我们将每个图像旋转[0°, 90°, 180°, 270°]并让网络预测角度。在Exemplar-MT中，模型被训练提取对广泛图像变换不变的特性。(2) 超参数。我们采用Adam [29] 优化器，初始学习率为0.005。每种方法训练200K次迭代，在160K次迭代时将学习率衰减0.2倍。我们使用Wide ResNet-28-2 [80] 作为骨干网络。在ForkMerge中，参数每10K次迭代合并一次。

### C.6 Data Division Strategy for ForkMerge

### C.6 ForkMerge的数据划分策略

As discussed in Section 4.2, in ForkMerge, we can construct branches with different sets of auxiliary tasks. Below we outline the specific data division strategy used in our experiments, which is consistent with previous ATL literature:

如第4.2节所述，在ForkMerge中，我们可以用不同的辅助任务集构建分支。下面我们概述实验中使用的具体数据划分策略，这与先前的ATL文献一致：

• For the NYUv2 dataset, multiple tasks share the same input, but their outputs are different. In this setup, each branch has the same input data, which includes the entire dataset. The distinction between different branches solely lies in the task weighting vector {\nub}B b=1.

• 对于NYUv2数据集，多个任务共享相同的输入，但它们的输出不同。在这种设置下，每个分支都有相同的输入数据，包括整个数据集。不同分支之间的区别完全在于任务权重向量{\nub}B b=1。

• For DomainNet, AliExpress, CIFAR-10, and SVHN datasets, different tasks have both different inputs and outputs. In these cases, for each branch, if the task weighting of a specific task is set to 0, the data from that particular task will not be used for training the corresponding branch.

• 对于DomainNet、AliExpress、CIFAR-10和SVHN数据集，不同任务既有不同的输入也有不同的输出。在这些情况下，对于每个分支，如果特定任务的任务权重设为0，则该特定任务的数据将不会用于训练对应的分支。

## D Additional Experiments

## D 额外实验

### D.1 Analysis on the importance of different forking branches

### D.1 不同分叉分支重要性的分析

The importance of different forking branches is dynamic. As shown in Figure 9, the relative ratio of each forking branch is dynamic and varies from task to task, which indicates the importance of the dynamic merge mechanism.

不同分叉分支的重要性是动态的。如图9所示，每个分叉分支的相对比例是动态的，并且因任务而异，这表明了动态合并机制的重要性。

### D.2 Analysis on the computation cost

### D.2 计算成本分析

The computation cost of Algorithm 2 is O(K) and the computation cost of the pruned version is O(B). Usually, only one model is optimized in most previous multi-task learning methods, yet their computational costs are not necessarily O(1). Gradient balancing methods, including MGDA [65], GradNorm [5], PCGrad [79], IMTL [41], GradVac [76], CAGrad [40], NashMTL [55], GCS [15], OL_AUX [39], and ARML [67], require computing gradients of each task, thus leading to O(K) complexity. In addition, calculating the inner product or norm of the gradients will bring a calculation cost proportional to the number of network parameters. A common practical improvement is to compute gradients of the shared representation [65]. Yet the speedup is architecture-dependent, and this technique may degrade performance [55].

算法2的计算复杂度为O(K)，剪枝版本的计算复杂度为O(B)。通常，大多数先前的多任务学习方法只优化一个模型，但它们的计算成本不一定是O(1)。梯度平衡方法，包括MGDA [65]、GradNorm [5]、PCGrad [79]、IMTL [41]、GradVac [76]、CAGrad [40]、NashMTL [55]、GCS [15]、OL_AUX [39]和ARML [67]，需要计算每个任务的梯度，因此导致O(K)复杂度。此外，计算梯度的内积或范数会带来与网络参数数量成比例的计算成本。一种常见的实践改进是计算共享表示的梯度[65]。然而加速效果依赖于架构，且这种技术可能会降低性能[55]。

In Figure 10, we also compare the actual training time across these methods on NYUv2. We can observe that ForkMerge does not require more time than most other methods. And considering the significant performance gains it brings, these additional computational costs are also worth it. Furthermore, our fork and merge mechanism enables extremely easy asynchronous optimization which is not straightforward in previous methods, thus the training time of our method can be reduced to O(1) when there are multiple GPUs available.

在图10中，我们还比较了NYUv2上这些方法的实际训练时间。我们可以观察到ForkMerge所需的时间不比大多数其他方法多。考虑到它带来的显著性能提升，这些额外的计算成本也是值得的。此外，我们的分叉和合并机制使得异步优化变得极为容易，这在先前方法中并不直接可行，因此当有多个GPU可用时，我们方法的训练时间可以降低到O(1)。

Figure 10: Training speed of different MTL methods on NYUv2 (10 repetitions).

图10：不同MTL方法在NYUv2上的训练速度（10次重复）。

### D.3 Analysis on the convergence and variance

### D.3 收敛性和方差分析

Figure 11 plots the validation performance of STL, EW, and ForkMerge throughout the training process on NYUv2. Each curve is obtained by optimizing the same method with 5 different seeds. Compared with single-task learning or minimizing the average loss across all tasks, ForkMerge not only improves the final generalization but also speeds up the convergence and reduces the fluctuations during training.

图11绘制了NYUv2上STL、EW和ForkMerge在整个训练过程中的验证性能。每条曲线是通过使用5个不同种子优化相同方法获得的。与单任务学习或最小化所有任务的平均损失相比，ForkMerge不仅提高了最终泛化能力，还加快了收敛速度并减少了训练过程中的波动。

### D.4 Comparison with grid searching \lambda

### D.4 与网格搜索\lambda的比较

In Section 3, we observe that adjusting the task-weighting hyper-parameter \lambda can effectively reduce the negative transfer and promote the positive transfer. [77] also suggests that sweeping the task weights should be sufficient for full exploration of the Pareto frontier at least for convex setups and observe no improvements in terms of final performance from previous MTL algorithms compared with grid search.

在第3节中，我们观察到调整任务权重超参数\lambda可以有效减少负迁移并促进正迁移。[77]也提出遍历任务权重应足以充分探索帕累托前沿（至少在凸设置下），并观察到先前的MTL算法在最终性能方面与网格搜索相比没有改进。

Figure 11: Learning curves comparing different methods on NYUv2. Each curve plots the mean and standard deviation of the validation performance of a method with 5 different random seeds.

图11：比较NYUv2上不同方法的学习曲线。每条曲线绘制了一种方法使用5个不同随机种子的验证性能的均值和标准差。

In Figure 12, we compare the performance of all methods with grid search on NYUv2. In grid search, the weighting hyper-parameter for each task takes values from {0.3, 1.0, 3.0}, and there are 3 tasks in NYUv2, thus there are 27 combinations in total. We find that previous methods simply yield performance trade-off points on the scalarization Pareto front, which has also been observed in previous work [77]. In contrast, our proposed ForkMerge yields point far away from the Pareto front and achieves significant improvement over simply optimizing a weighted average of the losses. One possible reason for the gain is that the task weighting in grid search is fixed during training and takes finite values due to the limitation of computing resources while the task weighting in ForkMerge is dynamic in time and nearly continuous in values, thus can better avoid negative transfer and promote positive transfer.

在图12中，我们比较了NYUv2上所有方法与网格搜索的性能。在网格搜索中，每个任务的权重超参数取值{0.3, 1.0, 3.0}，NYUv2中有3个任务，因此共有27种组合。我们发现先前方法只是在标量化帕累托前沿上产生性能权衡点，这在先前工作中也有观察到[77]。相比之下，我们提出的ForkMerge产生的点远离帕累托前沿，并比简单优化损失的加权平均实现了显著改进。这一增益的一个可能原因是网格搜索中的任务权重在训练期间是固定的，并且由于计算资源的限制取有限值，而ForkMerge中的任务权重在时间上是动态的，在值上几乎是连续的，因此能更好地避免负迁移并促进正迁移。

Figure 12: Comparison with grid search on NYUv2. We use mIoU for semantic segmentation, 1/absolute error for depth estimation, and 1/mean angle distance for surface normal prediction. We plot 2D projections of the performance profile for each pair of tasks. Top-right is better.

图12：与NYUv2上网格搜索的比较。我们使用mIoU进行语义分割，1/绝对误差进行深度估计，1/平均角距离进行表面法线预测。我们绘制每对任务性能曲线的2D投影。右上角更好。

### D.5 Comparison with larger batch size training

### D.5 与更大批量大小训练的比较

In some sense, the multiple branches in ForkMerge increase the equivalent batch size. And it has been revealed that batch size may have a great effect on the performance of deep models [21, 78]. To ablate the influence of batch size, we increase the batch size of the Equal Weighting method. As shown in Table 11, the improvement brought by ForkMerge itself is significantly larger than simply increasing the batch size.

在某种意义上，ForkMerge中的多个分支增加了等效批量大小。已有研究揭示批量大小可能对深度模型的性能有很大影响[21, 78]。为了消融批量大小的影响，我们增加了等权重方法的批量大小。如表11所示，ForkMerge本身带来的改进显著大于单纯增加批量大小。

Table 11: Comparison of different methods with larger batch size training.

表11：更大批量大小训练下不同方法的比较。

Methods: EW (Batch Size 8), EW (Batch Size 32), ForkMerge‡ (Batch Size 8). Metrics: Segmentation (mIoU $\uparrow$ , Pix Acc $\uparrow$ ), Depth (Abs Err $\downarrow$ , Rel Err $\downarrow$ ), Normal (Angle Distance Mean $\downarrow$ , Median $\downarrow$ , Within t◦ 11.25 $\uparrow$ , 22.5 $\uparrow$ , 30 $\uparrow$ ), ∆m $\uparrow$ .

方法：EW（批量大小8），EW（批量大小32），ForkMerge‡（批量大小8）。度量：分割（mIoU $\uparrow$ ，Pix Acc $\uparrow$ ），深度（Abs Err $\downarrow$ ，Rel Err $\downarrow$ ），法线（Angle Distance Mean $\downarrow$ ，Median $\downarrow$ ，Within t◦ 11.25 $\uparrow$ , 22.5 $\uparrow$ , 30 $\uparrow$ ），∆m $\uparrow$ 。

### D.6 ForkMerge with more network architectures

### D.6 更多网络架构上的ForkMerge

ForkMerge with Vision Transformers. We replace the backbone network ResNet-101 with advanced ViT-Base [13] pretrained on ImageNet-21K and repeat the experiments on the DomainNet dataset (Section 5.2). As demonstrated in Table 12, when employing the Vision Transformer model, which boasts increased capacity, the risk of overfitting with limited data becomes more pronounced.

使用Vision Transformers的ForkMerge。我们将骨干网络ResNet-101替换为在ImageNet-21K上预训练的先进ViT-Base [13]，并在DomainNet数据集（第5.2节）上重复实验。如表12所示，当使用容量更大的Vision Transformer模型时，在有限数据下过拟合的风险变得更加明显。

Table 12: Performance on DomainNet by replacing the ResNet-101 architecture with the ViT-Base architecture.

表12：将ResNet-101架构替换为ViT-Base架构后在DomainNet上的性能。

Methods: STL, EW, Auto-\lambda, Post-train, ForkMerge. Metrics: C, I, P, Q, R, S, Avg ∆m $\uparrow$ .

方法：STL, EW, Auto-\lambda, Post-train, ForkMerge。度量：C, I, P, Q, R, S, Avg ∆m $\uparrow$ 。

This makes Single Task Learning (STL) less effective and consequently leads to the Equal Weighting (EW) method outperforming STL, causing the Post-train method to fall short of EW and Auto-\lambda. In this case, ForkMerge still exhibited superior performance, validating its efficacy across different network architectures.

这使得单任务学习（STL）效果较差，从而导致等权重（EW）方法优于STL，使得Post-train方法不及EW和Auto-\lambda。在这种情况下，ForkMerge仍然表现出优越的性能，验证了其在不同网络架构上的有效性。

ForkMerge with Multi-task Architectures. ForkMerge is complementary to different multi-task architectures. In Tables 13 and 14, we provide a comparison of different optimization strategies with MTAN [44] and MMoE [46] as architectures, which are widely used in multi-task computer vision tasks and multi-task recommendation tasks respectively. On these specifically designed multi-task architectures, ForkMerge is still significantly better than other methods.

使用多任务架构的ForkMerge。ForkMerge与不同的多任务架构是互补的。在表13和14中，我们提供了以MTAN [44] 和MMoE [46] 为架构的不同优化策略的比较，这两种架构分别广泛用于多任务计算机视觉任务和多任务推荐任务。在这些专门设计的多任务架构上，ForkMerge仍然显著优于其他方法。

Table 13: Performance on NYUv2 dataset by replacing the DeepLabV3+ architecture with the MTAN architecture.

表13：将DeepLabV3+架构替换为MTAN架构后在NYUv2数据集上的性能。

Methods: STL, EW, GCS, OL_AUX, ARML, Auto-\lambda, ForkMerge‡.

方法：STL, EW, GCS, OL_AUX, ARML, Auto-\lambda, ForkMerge‡。

Metrics: Segmentation (mIoU $\uparrow$ , Pix Acc $\uparrow$ ), Depth (Abs Err $\downarrow$ , Rel Err $\downarrow$ ), Normal (Angle Distance Mean $\downarrow$ , Median $\downarrow$ , Within t◦ 11.25 $\uparrow$ , 22.5 $\uparrow$ , 30 $\uparrow$ ), ∆m $\uparrow$ .

Table 14: Performance on AliExpress dataset by replacing the ESMM architecture with the MMoE architecture.

表14：将ESMM架构替换为MMoE架构后在AliExpress数据集上的性能。

Methods: EW, GCS, OL_AUX, ARML, Auto-\lambda, ForkMerge‡.

方法：EW, GCS, OL_AUX, ARML, Auto-\lambda, ForkMerge‡。

Metrics: CTR (ES, FR, NL, US), CVCTR (ES, FR, NL, US), Avg ∆m $\uparrow$ .
