# Improving Pairwise Learning for Item Recommendation from Implicit Feedback

> Steffen Rendle, Christoph Freudenthaler, University of Konstanz, Germany | steffen.rendle@uni-konstanz.de, chr.freudenthaler@gmail.com



本文提出了面向隐式反馈item推荐的 **自适应Pairwise成对采样算法**，通过 **非均匀采样信息性pair** 来加速BPR的收敛。核心内容：

- **问题识别**：**item流行度的长尾分布**导致 均匀采样BPR 收敛缓慢
- **自适应采样器**：上下文相关的非均匀采样器，对信息量大的pair过采样
- **高效实现**：常数摊还运行时开销，适用多种推荐模型

关键发现：自适应采样器大幅提升收敛速度，且不损害预测质量或迭代运行时。

---



## 摘要

Pairwise算法是从隐式反馈中学习推荐系统的流行方法。对于每个用户，或者更一般地每个上下文，它们试图**区分一小部分已选item和大量剩余（无关）item**。学习通常基于随机梯度下降（SGD）和 **均匀采样的pair**。在这项工作中，我们表明，如果item流行度具有长尾分布，此类SGD学习算法的收敛速度会显著减慢。我们提出了一种**非均匀item采样器**来解决这个问题。所提出的采样器是上下文相关的，并对信息性pair进行过采样以加速收敛。我们开发了一种具有常数摊还运行时开销的高效实现。此外，我们展示了所提出的学习算法如何应用于**一大类推荐模型**。我们在两个真实的推荐系统问题上对新学习算法的性质进行了实证研究。实验表明，所提出的自适应采样器在收敛性上大幅改进了现有最优的学习算法，同时不会对预测质量或迭代运行时产生负面影响。



## 1 引言

推荐系统已成为现代网站（如Amazon、Netflix或Flickr）的重要功能。点击率、收入和其他成功指标可以通过应用有效的推荐系统来提高。**困难的任务是识别相关的item，即使它们通常不受欢迎**。推荐系统利用可用的上下文（如用户信息、时间、位置等）来筛选相关的item。因此，来自流行度分布 **尾部的item**也能被成功推荐。

在实践中，通常只有隐式反馈可用于学习推荐系统。隐式反馈的例子包括点击、观看的电影、播放的歌曲、购买或分配的标签。**隐式反馈的一个特点是它是单类的，即只有正观测可用**。例如，网站记录用户在特定日期购买了某个item或为特定资源选择了某个标签。**推荐系统中处理这些数据的一种常见方法是假设 所有未被选择的内容 在该情境下对用户兴趣较少。这个想法导致了成对排序损失，试图区分一小部分已选item和大量剩余item**。由于pair数量非常庞大，学习算法通常基于（均匀）采样pair并应用随机梯度下降（SGD）。这个优化框架也被称为贝叶斯个性化排序（BPR）[14]。许多最近发表的推荐系统使用BPR进行学习，包括用于标签推荐的张量分解模型[15]、关系抽取[17]、带分类法的序列购物推荐[8]、用于广告的聚焦矩阵分解[7]、层次latent因子模型[1]或协同分解机[5]。

**本文表明，均匀采样pair会导致收敛缓慢，特别是当item池很大且整体item流行度呈长尾分布时**。这两个性质是大多数真实数据集所共有的。我们认为**大多数SGD更新没有效果，因为梯度消失**。**原因是均匀采样的负item很可能被正确排在某个观测到的正item之下，因此该pair的梯度接近0**。我们通过实验表明，简单的 全局流行度过采样不足以解决这个问题。我们提出了一种非均匀采样分布，它**同时适应上下文和学习当前状态**。我们开发了一种具有 常数摊还运行时复杂度 的高效采样算法，**用于从提议分布proposed distribution中进行采样**。在两个真实推荐应用上的实验表明，所提出的自适应采样器在**收敛性**上大幅改进了现有最优的学习算法，同时不会对预测质量或迭代运行时产生负面影响。最后，我们展示了如何将该算法推广到一大类分解模型。

> [!NOTE]
>
> 1、原因是均匀采样的负item很可能被正确排在某个观测到的正item之下，因此该pair的梯度接近0。
>
> 这是因为BPR建模的是正样本排在负样本之前的概率，通过sigmoid对正负样本的预测分差计算概率。当分差很大时概率近接为1，此时损失接近为0，梯度消失。因此对于成对采样，只有正负样本分差较小（概率为0.5，损失为-ln0.5）甚至分差为负时（概率小于0.5，损失更大）时，才能有效更新模型参数。
>
> 2、隐式反馈的一个特点是它是单类的，即只有正观测可用。
>
> 这句话是否有失偏颇？比如曝光、点击同为隐式反馈，但是不是单类。而是能够区分正负类的？
>
> 这个结论仅在没有记录曝光数据的前提下有效。在现代推荐系统背景下，确实有问题



## 2 问题陈述

首先描述从隐式反馈数据中推荐item的问题。然后简要回顾BPR [14]的成对学习。本节的新贡献是**表明BPR算法的收敛由于负item的均匀采样而减慢**。

### 2.1 从隐式反馈中排序

设 $S \subset C \times I$ 是一组观测到的行为集合，其中 $C$ 是上下文集合，$I$ 是item集合。例如，$C$ 可以是用户集合，$I$ 是电影集合，$S$ 表示用户观看过哪些电影，即**用户-电影pair**。更复杂的例子也可以处理，例如 $C$ 可能包含额外的变量如位置、心情、时间、序列或属性；$I$ 也可能通过额外变量来描述，例如属性或分类法（见4.2节）。

item推荐的任务是为每个上下文找到一个排序 $\hat{r}$。我们通过一个双射函数 $\hat{r} : I \times C \to \{1, \ldots, |I|\}$ 来形式化这个问题，其中 $\hat{r}(i|c)$ 是item $i$ 在给定上下文 $c$ 下的排名。排序函数通常由一个 **评分函数** $\hat{y}(i|c)$ 建模，该函数本身由一组模型参数 $\Theta$ 参数化。例如，如果 $i$ 和 $c$ 是分类变量，矩阵分解（MF）是评分模型的常见选择。**使用任何评分模型进行排序都可以通过计算所有item的得分（给定上下文 $c$）并按得分排序item（对每个上下文）来完成**。排名 $\hat{r}$ 和评分函数 $\hat{y}$ 之间的正式联系可定义为：

$$
\hat{r}(i|c) := |\{j : \hat{y}(j|c) \geq \hat{y}(i|c)\}|. \qquad (1}
$$

例如，对于排名第一的item，只有一个item（它本身）得分大于或等于它。对于排名第二的item，只有两个item的得分大于或等于它，以此类推。

排序本身由模型参数 $\Theta$ 的值通过评分函数 $\hat{y}$ 唯一确定（并列情况除外）。

> [!NOTE]
>
> 等价形式：$\hat{r}(i|c) = 1 + |\{j \neq i : \hat{y}(j|c) \geq \hat{y}(i|c)\}|$
>
> 即：item $i$ 的排名 = 所有得分不低于它的 item 数量。排名值越小越好（排名第 1 最佳）。

### 2.2 从隐式反馈中成对学习

模型参数 $\Theta$ 的值是从隐式反馈数据 $S$ 中学习的。一种流行的学习模型参数 $\Theta$ 的方法基于成对学习。其思想是对于每个上下文 $c \in C$，区分已选item $I^+(c) := \{i : (i, c) \in S\}$  和 剩余item $I \setminus I^+(c)$。在上下文 $c$ 下，item $i$ 优于item $j$（记作 $i \succ_c j$），**当且仅当 $i$ 被选择而 $j$ 未被选择**：

$$
i \succ_c j \Leftrightarrow i \in I^+(c) \land j \in I \setminus I^+(c). \qquad (2}
$$

所有成对偏好的集合 $D_S \subseteq C \times I \times I$ 可定义为：

$$
(c, i, j) \in D_S :\Leftrightarrow i \in I^+(c) \land j \in I \setminus I^+(c). \qquad (3}
$$

从成对偏好 到 模型/评分函数 $\hat{y}$ 的联系由下式建立：

$$
p(i \succ_c j) := \sigma(\hat{y}(i|c) - \hat{y}(j|c)) \qquad (4}
$$

其中 $\sigma(x) = 1/(1 + \exp(-x))$。目标是**最大化正确排序偏好的似然**：

$$
\arg\max_{\Theta} \prod_{(c,i,j) \in D_S} p(i \succ_c j), \qquad (5}
$$

这等价于最小化 **负对数似然**（NLL）：

$$
NLL := - \sum_{(c,i,j) \in D_S} \ln \sigma(\hat{y}(c, i) - \hat{y}(c, j)). \qquad (6}
$$

**SGD学习。** 任意模型参数 $\theta \in \Theta$ 的梯度为：

$$
\frac{\partial NLL}{\partial \theta} = \sum_{(c,i,j) \in D_S} (1 - \sigma(\hat{y}(c, i) - \hat{y}(c, j))) \frac{\partial (\hat{y}(c, i) - \hat{y}(c, j))}{\partial \theta}. \qquad (7}
$$

由于pair数量 $|D_S|$ 非常庞大，学习算法通常基于随机梯度下降（SGD）。均匀采样一个pair $(c, i, j) \in D_S$ 并执行一个随机梯度下降步骤：

$$
\theta \leftarrow \theta - \eta \underbrace{(1 - \sigma(\hat{y}(c, i) - \hat{y}(c, j)))}_{=: \Delta_{c,i,j}} \frac{\partial}{\partial \theta} (\hat{y}(c, i) - \hat{y}(c, j)). \qquad (8}
$$

其中 $\eta$ 是学习率，必须选择足够小以确保步骤沿正确方向进行——即梯度仅在 $\theta$ 附近的小区域内（近似）正确。注意隐式反馈 $S \subseteq C \times I$  和 训练pair $D_S \subseteq C \times I \times I$ 之间的区别。

<img src="/Users/dazhang/PycharmProject/Papers/3-RecSys/.picture/image-20260719113143586.png" alt="image-20260719113143586" style="zoom:33%;" />

均匀采样偏好 $(c, i, j) \in D_S$ 可以在**不显式存储 $D_S$ 的情况下**完成：首先采样 $(c, i) \in S$，然后采样负item $j \in I \setminus I^+(c)$。完整算法见图2。在[14]中，以贝叶斯个性化排序（BPR）的名义提出了用于item推荐的整个框架，包括已选item和所有剩余item之间的成对损失以及使用均匀采样的SGD算法。

> [!CAUTION]
>
> (7) 和 (8)公式存在问题
>
> $\frac{\partial \text{NLL}}{\partial\theta} = \color{red}{-}\sum_{(c,i,j)}(1-\sigma(\Delta))\frac{\partial\Delta}{\partial\theta}$
>
> $\theta \leftarrow \theta \color{red}{+} \alpha \cdot (1-\sigma(\hat{x}_{uij})) \cdot \frac{\partial \hat{x}_{uij}}{\partial\theta}$



### 2.3 长尾item分布中的问题

尽管BPR已成功应用于众多推荐应用和各种模型，但下面将表明，**当item池 $I$ 很大且item流行度呈非均匀分布时，收敛速度会显著减慢**。

**梯度幅度。** 使用BPR学习模型参数是通过循环执行式(8)完成的。可以看出，**每个梯度步骤都有一个乘性标量**：
$$
\Delta_{c,i,j} := (1 - \sigma(\hat{y}(c, i) - \hat{y}(c, j))) = (1 - p(i \succ_c j)). \qquad (9}
$$

该量取决于评分模型（使用当前模型参数 $\Theta$）如何在上下文 $c$ 下区分正item $i$ 和负item $j$。量 $\Delta_{c,i,j}$ 显然是一个概率，如果 $i$ 被正确分配了比 $j$ 更大的得分，则接近0。如果 $j$ 被错误分配了比 $i$ 更大的得分，则接近1。这意味着 $\Delta_{c,i,j}$ 可以被理解为**成对偏好 $(c, i, j)$ 对改善 $\Theta$ 有多大影响**。如果 $\Delta_{c,i,j}$ 接近0，则无法从该案例 $(c, i, j)$ 中学到任何东西，因为其梯度消失，即 $\theta$ 不会被更新步骤（式8）改变。因此，在后续内容中 $\Delta_{c,i,j}$ 被称为**采样案例 $(c, i, j)$ 的梯度幅度**。注意 $\Delta_{c,i,j}$ 依赖于模型参数 $\Theta$，因此 $\Delta_{c,i,j}$ 在学习过程中会变化。

<img src="/Users/dazhang/PycharmProject/Papers/3-RecSys/.picture/image-20260717172324517.png" alt="image-20260717172324517" style="zoom:33%;" />

**长尾item分布。** 在推荐系统中，item流行度通常是非均匀分布的，某些item总体上比其他的更受欢迎。图1显示了一个电影数据集和一个社交标签数据集的两种item流行度分布——参见5.1节了解数据集的详细信息。**两图均显示大多数item总体上很少被选择**。

<img src="/Users/dazhang/PycharmProject/Papers/3-RecSys/.picture/image-20260717172446284.png" alt="image-20260717172446284" style="zoom:33%;" />

> **图 3：梯度幅度（式 9）衡量训练样本对当前学习过程的影响程度（区间 [0,1]）。三条曲线分别描绘了当使用均匀采样（BPR）时梯度幅度小于 0.01、0.1 和 0.5 的样本比例。在仅仅几个训练 epoch 之后，几乎所有幅度都小于 0.1，且大多数小于 0.01。**幅度接近零的训练样本会减慢学习速度，因为对应的 SGD 更新步不会改变模型参数，但仍需消耗计算时间。平均梯度的变化情况见图 6，实验设置详情见 5.1 节。

如前所述，SGD算法无法从幅度 $\Delta_{c,i,j}$ 接近零的样本 $(c, i, j)$ 中学习。当 $i$ 被正确排在 $j$ 之上时，$\Delta_{c,i,j}$ 应该很小——即差值 $\hat{y}(c, i) - \hat{y}(c, j)$ 越大，$\Delta_{c,i,j}$ 越小。因为 $(c, i)$ 是一个正观测（且总体正观测呈如图1所示的长尾分布），而 $j$ 是均匀采样的，模型得分 $\hat{y}(c, i)$ 很可能也大于 $\hat{y}(c, j)$，因此梯度幅度很小。图3显示了样本 $(c, i, j)$ 的梯度幅度分别小于0.01、0.1和0.5的概率。**可以看出，在仅仅几个训练epoch之后（一个epoch包含 $10 \cdot |S|$ 次BPR更新），几乎所有样本的幅度都非常小，因此大多数样本在SGD算法中是无用的——即更新（式8）不会改变参数值**。需要注意，这并不意味着损失不合适，而是好的样本尚未被算法看到。

例如，在给定上下文下，一个正item可能被估计排在第10位而不是第1位，这意味着有9个信息量非常大的样本，而剩余的大部分 $|I| - 10$ 个item主要是无信息的。如果 $|I|$ 很大，均匀采样很可能需要大量迭代才能找到这9个样本中的一些，而在此期间算法花费了大量时间对无用的样本进行更新。

> [!NOTE]
>
> 非常有价值的一段解释

在本工作的后续部分，提出了非均匀采样器来替换BPR算法中的负item采样器（见图2第5行）以解决这个问题。



## 3 改进的item采样

在本节中，提出了用于**负item的非均匀采样器以加速收敛**。负item的采样分布将记为 $p(j|c)$。首先，介绍了**基于全局item流行度的采样**作为基线。然后，**提出了一种自适应、上下文感知的采样分布，该分布遵循排序模型 $\hat{y}$ 的信念**。

### 3.1 静态与全局采样

在2.3节中，成对学习的均匀采样假设（$p(j|c) \propto 1$）被确定为SGD学习收敛缓慢的主要原因。这导致了**大多数pair是无信息的**，且可以轻松地以正确顺序排序。

增加 **困难pair** 比例的一个简单方法是对 **流行item进行过采样**。经验流行度/选择频率（见图1）可以直接用来**定义采样分布**：

$$
p(j|c) \propto |\{(c', j') \in S : j = j'\}|. \qquad (10}
$$


> [!NOTE]
>
> $p(j|c) \propto 1$：正比于（不是等于，需要归一化）
>
> $S$：所有正观测集合，每条为 $(c', j')$
>
> 概率 ∝ 流行度。一个 item 在正观测中出现的次数越多，它被选为负样本的概率就越高。

或者，可以使用**参数化采样**。通常，经验分布近似遵循某种解析规律，例如 几何分布 或 Zipf分布（见图1）。例如，对于几何分布：
$$
p(j|c) = \gamma (1 - \gamma)^{r(j)}, \quad \gamma \in (0, 1) \qquad (11}
$$

或者等价地使用另一种表示/参数化：

$$
p(j|c) \propto \exp(-r(j)/\lambda), \quad \lambda \in \mathbb{R}^+, \qquad (12}
$$

其中 $r(j)$ 是**根据全局流行度排序的item $j$ 的排名**。式(12)分布中的期望排名由参数 $\lambda$ 确定。

> [!NOTE]
>
> 疑问：参数化采样？几何分布？
>

选择 经验分布（式10）还是 其参数化对应物大致会产生相同的结果。在实践中，经验分布可能更容易实现，我们在评估中使用它。然而，后面提出的更复杂模型使用了参数化方法，因此在此讨论参数化对应物有助于提高可读性。

**性质。** 所提出的采样分布具有两个重要性质：

1. **全局性：** item分布与上下文无关，即对不同上下文，分布是相同的。
2. **静态性：** 分布在模型参数学习期间不改变。

**算法。** 由于该分布是全局且静态的，从参数化item分布（式12）中采样很简单：首先，从几何分布中以 $O(1)$ 时间采样一个排名 $r$。其次，以 $O(1)$ 时间返回全局流行度列表中排名第 $r$ 位的item。

总之，从式(12)采样可以在不增加原始BPR算法计算复杂度的情况下完成。

**讨论。** 开发采样器的动机是：为给定上下文 $c$ 选择一个item $j$，使得pair $(i, j)$ 在当前学习状态下（即对于当前 $\Theta$ 值）是信息性的。所提出的基于流行度的采样器在两个方面未能反映这一点：(1) 它是静态的，因此没有考虑item $j$ 的估计排名 $\hat{r}(j|c)$ 在学习过程中会变化。例如，一个item可能在开始时排名很高，但在若干学习步骤后排名变低。(2) 采样器是全局的，没有反映item的信息性取决于上下文。例如，一个item可能对一群人有兴趣但对另一群人则不然。这两个方面也可以在梯度幅度 $\Delta_{c,i,j}$ 中看到，它依赖于 $c$ 并在学习过程中变化。



### 3.2 自适应与上下文相关的采样（TODO）

下面开发了一个细粒度的采样器，它同时适应上下文和模型的当前信念（即通过 $\Theta$ 适应 $\hat{y}$）。类似于全局item流行度（式10），可以定义一个**静态的上下文特定流行度分布**：

$$
p(j|c) \propto |\{(c', j') \in S : c = c', j = j'\}| = \delta((c, j) \in S). \qquad (13}
$$

然而，该分布有两个缺点：(1) 它定义在非常少的样本上，即对于给定的上下文 $c$，只存在一个小的已选item子集 $I^+(c)$，其中通常没有item被多次选择。这将使item分布成为一个阶跃函数，其中负item无法区分。(2) 它没有考虑当前信念（即 $\Theta$）。

相反，我们提出**使用评分函数 $\hat{y}$ 来定义采样分布**。直观地说，当需要为给定观测 $(c, i) \in S$ 采样一个负item $j$ 时，$j$ 越接近顶部（排名 $\hat{r}(j|c)$ 越小），$j$ 的信息量越大。这也可以在梯度幅度 $\Delta_{c,i,j}$ 中看出：如果给定 $(c, i)$，我们应该选择使得 $\hat{y}(j|c)$ 大的 $j$，因为这会增大 $\Delta_{c,i,j}$。与其使用大得分这一概念，不如形式化一个小的预测排名 $\hat{r}(j|c)$，因为得分的大小是相对于其他item而言的，而排名是一个绝对值。这使我们能够公式化一个自适应且上下文感知的采样分布：

$$
p(j|c) \propto \exp(-\hat{r}(j|c)/\lambda), \quad \lambda \in \mathbb{R}^+. \qquad (14}
$$

**性质。** item分布（式14）依赖于 $\hat{r}(j|c)$。提醒一下，$\hat{r}(j|c)$ 是item $j$ 在所有item $I$ 中使用评分模型 $\hat{y}(j|c)$ 对item进行排序得到的排名。因此，所提出的采样器是：

1. **上下文相关的：** 采样概率依赖于上下文，因为 $\hat{r}(j|c)$ 是item $j$ 在给定上下文 $c$ 下的估计排名。模型 $\hat{y}$ 区分不同上下文的能力越强，排序以及采样器就越具有上下文感知能力。
2. **自适应的：** 采样器在模型参数 $\Theta$ 被学习时会发生变化，因为 $\Theta$ 的变化依次导致评分模型 $\hat{y}$、排序 $\hat{r}$ 和采样器的变化。

**朴素算法。** 给定正观测 $(c, i) \in S$ 的负item $j$ 的采样器可以实现为：首先，以 $O(1)$ 时间从几何分布中采样一个排名 $r$。其次，返回当前排名在第 $r$ 位的item $j$，即找到满足 $\hat{r}(j|c) = r$ 的 $j$ 或 $j = \hat{r}^{-1}(r|c)$。第二步的朴素实现需要计算所有 $j \in I$ 的 $\hat{y}(j|c)$，然后按得分对item $j$ 排序，最后返回位置 $r$ 处的item。该算法的复杂度为 $O(|I| \cdot T_{pred} + |I| \log(|I|))$，其中 $T_{pred}$ 是预测一个得分的时间。注意，该采样器应替换算法2中的第5行，会使BPR的运行时间增加 $O(|I| \cdot T_{pred} + |I| \log(|I|))$ 倍。这在实际中显然不可行。



## 4 高效的采样算法

下面展示如何对一大类分解模型在**摊还常数时间**内高效实现对式(14)的近似采样。首先展示矩阵分解的基本思想，然后推广到分解机[13]。

### 4.1 矩阵分解（MF）

假设上下文 $C$ 和item $I$ 由分类变量表示，即 $C = \{c_1, c_2, \ldots\}$ 和 $I = \{i_1, i_2, \ldots\}$。例如，在个性化设置中，每个上下文 $c$ 可以对应一个用户。设评分模型 $\hat{y}$ 为矩阵分解（MF）：

$$
\hat{y}(l|c) := \sum_{f=1}^{k} v_{c,f} v_{l,f}, \quad V \in \mathbb{R}^{(C \cup I) \times k}. \qquad (15}
$$

其中 $k \in \mathbb{N}$ 是latent维度，因子 $V$ 是模型参数 $\Theta$。使用MF（式15）对一个item进行评分的复杂度为 $O(k) =: T_{pred}$。

下面，我们推导出一种快速的自适应和上下文相关的采样算法，该算法在摊还常数时间内近似于式(14)的采样器。其思想是将式(14)形式化为一个关于归一化因子的排序分布的混合。混合概率从MF评分函数式(15)的归一化版本导出。注意，最终的采样器适用于任何MF模型，无需显式执行变换，但该变换对于推导算法是必要的。

**排名不变归一化。** 首先，定义 $\hat{y}$ 的一个变换 $\hat{y}^*$：

$$
\hat{y}^*(l|c) := \sum_{f=1}^{k} p(f|c) \operatorname{sgn}(v_{c,f}) v_{l,f}^* \qquad (16}
$$

其中 $p(f|c)$ 是概率函数：$$p(f|c) \propto |v_{c,f}| \sigma_f \qquad (17}$$

而 $v_{l,f}^*$ 是标准化的item因子：

$$
v_{l,f}^* = \frac{v_{l,f} - \mu_f}{\sigma_f} \qquad (18}
$$

其中所有item因子的经验均值和方差为：

$$
\mu_f = \mathbb{E}(v_{\cdot,f}), \quad \sigma_f^2 = \operatorname{Var}(v_{\cdot,f}). \qquad (19}
$$

**引理4.1（排名不变性）。** 由 $\hat{y}^*$ 评分生成的排名 $\hat{r}^*$ 与由 $\hat{y}$ 生成的排名 $\hat{r}$ 相同。

**证明。** 首先，评分函数可以重写为：

$$
\hat{y}(l|c) = \sum_{f=1}^{k} v_{c,f} v_{l,f} = \sum_{f=1}^{k} |v_{c,f}| \operatorname{sgn}(v_{c,f}) (\sigma_f v_{l,f}^* + \mu_f)
$$

$$
= \sum_{f=1}^{k} |v_{c,f}| \operatorname{sgn}(v_{c,f}) \sigma_f v_{l,f}^* + \sum_{f=1}^{k} |v_{c,f}| \operatorname{sgn}(v_{c,f}) \mu_f
$$

$$
= \hat{y}^*(l|c) + \underbrace{\sum_{f=1}^{k} |v_{c,f}| \operatorname{sgn}(v_{c,f}) \mu_f}_{=: b(c)}.
$$

附加项 $b(c)$ 与item $l$ 无关。因此 $\hat{y}(l|c)$ 是 $\hat{y}^*(l|c)$ 的一个线性变换。一般来说，线性变换是排名不变的，因为：

$$\hat{y}(i|c) \geq \hat{y}(j|c) \Leftrightarrow a(c) \hat{y}(i|c) \geq a(c) \hat{y}(j|c)$$

$$\Leftrightarrow a(c) \hat{y}(i|c) + b(c) \geq a(c) \hat{y}(j|c) + b(c)$$

$$\Leftrightarrow \hat{y}^*(i|c) \geq \hat{y}^*(j|c).$$

由此引理得证。

这意味着，如果我们对 $\hat{y}$ 生成的排名感兴趣，我们也可以使用 $\hat{y}^*$。尽管 $\hat{y} \neq \hat{y}^*$，生成的排名 $\hat{r} = \hat{r}^*$ 是相等的。

**排名混合。** 表示 $\hat{y}^*$ 的优点在于 $p(f|c)$ 可以被解读为标准化item因子上的混合概率。即 $p(f|c)$ 越大，维度 $f$ 对特定上下文 $c$ 越重要。这允许将采样分布定义为混合：

$$p(j|c) := \sum_{f=1}^{k} p(f|c) p(j|c, f). \qquad (20}$$

由于 $v_{l,f}^*$ 的标准化，根据式(14)类比地定义 $p(j|c, f)$ 是合理的：

$$p(j|c, f) \propto \exp(-\hat{r}^*(j|c, f)/\lambda), \qquad (21}$$

其中排名 $\hat{r}^*(j|c, f)$ 由上下文和因子相关的评分函数 $\hat{y}^*(j|c, f)$ 生成。根据式(16)，该评分函数可定义为：

$$\hat{y}^*(l|c, f) := \operatorname{sgn}(v_{c,f}) v_{l,f}^*. \qquad (22}$$

我们可以摆脱 $v_{l,f}^*$ 的标准化，使用一个排名不变但更简单的函数：

$$\hat{y}(l|c, f) := \operatorname{sgn}(v_{c,f}) v_{l,f}. \qquad (23}$$

注意 $\hat{y}(l|c, f)$ 依赖于原始参数 $V$ 而不是它们的归一化。评分函数 $\hat{y}(l|c, f)$ 与其排名 $\hat{r}(l|c, f)$ 有非常简单的关系：排名 $r$ 上的item具有第 $r$ 大的因子 $v_{l,f}$——如果 $\operatorname{sgn}(v_{c,f})$ 为正，否则是第 $r$ 大的负因子。

**从排名混合中采样。** 将采样分布形式化为混合模型（式20）产生了负item的简单采样算法：

1. 从几何分布中采样一个排名 $r$。
2. 从 $p(f|c)$（式17）中采样一个因子维度 $f$。
3. 根据 $v_{\cdot,f}$ 对item排序，这等价于逆排名函数 $\hat{r}^{-1} : \mathbb{N} \times \{1, \ldots, k\} \to I$。
4. 返回排序列表中位置 $r$ 上的item $j$，即如果 $\operatorname{sgn}(v_{c,f}) = 1$ 则返回 $\hat{r}^{-1}(r|f)$，否则返回 $\hat{r}^{-1}(|I| - r + 1|f)$。

步骤1和4可以在 $O(1)$ 时间内完成，步骤2（包括计算 $p(f|c)$）在 $O(k) = T_{pred}$ 时间内完成。唯一计算密集的步骤是3，其中因子排序需要 $O(|I| \log |I|)$。

为了进一步降低复杂度，我们建议预先计算每个因子维度 $f$ 的排序 $\hat{r}^{-1}(\cdot|f)$，并每隔几个随机更新步骤重新计算一次。在单个梯度步骤之后，排序 $\hat{r}^{-1}(\cdot|f)$ 只发生微小变化，需要许多更新步骤才能显著改变预计算的排序。我们建议每 $|I| \log |I|$ 次迭代重新计算 $k$ 个排序，这在评估中也取得了良好的结果。所描述的预计算策略的摊还运行时间为 $O(k)$，因为每 $|I| \log |I|$ 次迭代需要 $O(k |I| \log |I|)$ 的工作。此外，预计算需要 $k|I|$ 的额外内存来存储所有 $\hat{r}^{-1}(\cdot|f)$。

总之，采样算法抽取一个item的摊还运行时间为 $O(k)$，这与MF模型单个梯度步骤的成本（$= T_{pred}$）相同。由于每个梯度步骤对应一个样本，原始SGD算法的计算复杂度没有增加。图4描绘了改进后的学习算法的伪代码。

<img src="/Users/dazhang/PycharmProject/Papers/3-RecSys/.picture/image-20260717172536309.png" alt="image-20260717172536309" style="zoom:33%;" />

| 算法: LearnAdaptiveOversampling($\eta$, $S$) |
|---|
| 1: 随机初始化 $\Theta$, $q = 0$ |
| 2: **repeat** |
| 3: **for** $f \in \{1, \ldots, k\}$ **do** |
| 4: 计算 $\sigma_f^2$ 和 $\mu_f$ $\triangleright O(|I|)$ |
| 5: 计算 $\hat{r}^{-1}(\cdot|f)$ $\triangleright O(|I| \log |I|)$ |
| 6: **end for** |
| 7: **repeat** |
| 8: $q \leftarrow q + 1$ |
| 9: 均匀抽取 $(c, i) \in S$ |
| 10: 从 $p(r) \propto \exp(-r/\lambda)$ 抽取 $r$ $\triangleright O(1)$ |
| 11: 从 $p(f|c) \propto |v_{c,f}| \sigma_f$ 抽取 $f$ $\triangleright O(k)$ |
| 12: **if** $\operatorname{sgn}(v_{c,f}) = 1$ **then** |
| 13: $j \leftarrow \hat{r}^{-1}(r|f)$ |
| 14: **else** |
| 15: $j \leftarrow \hat{r}^{-1}(|I| - r + 1|f)$ |
| 16: **end if** $\triangleright O(1)$ |
| 17: **for** $\theta \in \Theta$ **do** |
| 18: 用式(8)更新 $\theta$ |
| 19: **end for** |
| 20: **if** $q \% |I| \log |I| = 0$ **then** $\triangleright$ 每 $|I| \log |I|$ 次抽取 |
| 21: **break** |
| 22: **end if** |
| 23: **until** convergence |
| 24: **until** convergence |
| 25: **return** $\Theta$ |

**图4:** 用于矩阵分解的带有自适应和上下文相关负item过采样的BPR。



### 4.2 复杂分解模型

本节展示如何修改图4中的高效算法来学习通用分解机（FM）模型。

设 $x_i \in \mathbb{R}^{p_I}$ 是一个描述item $i$ 的任意特征向量，包含 $p_I$ 个实值变量，$x_c \in \mathbb{R}^{p_C}$ 是一个描述上下文 $c$ 的特征向量，包含 $p_C$ 个变量。这种灵活表示允许描述许多不同类型的数据，包括属性、时间或序列上下文以及它们的组合[13]。特征向量 $x \in \mathbb{R}^p$（这里 $x = (x_c, x_i)$，$p = p_C + p_I$）上的二阶分解机（FM）定义为：

$$\hat{y}(x) = w_0 + \sum_{l=1}^{p} w_l x_l + \sum_{l=1}^{p} \sum_{l' > l} x_l x_{l'} \langle v_l, v_{l'} \rangle \qquad (24}$$

其中 $w_0, w_1, \ldots, w_p, v_{1,1}, \ldots, v_{p,k}$ 是模型参数 $\Theta$。计算评分函数（式24）的复杂度为 $T_{pred} = O(k \cdot NZ(x))$，其中 $NZ(x)$ 是 $x$ 中非零值的数量。SGD步骤也具有此复杂度[13]。下面展示如何在摊还 $T_{pred}$ 时间内采样一个负item。

**排名不变变换。** 为了推导高效采样算法，首先为每个上下文和item定义一个 $k+1$ 维的因子向量 $v'$：

$$v'_{c,f} := \sum_{l=1}^{p_C} v_{l,f} x_{c,l}, \quad f = 1, \ldots, k, \qquad v'_{c,0} := 1, \qquad (25}$$

$$v'_{i,f} := \sum_{l=1}^{p_I} v_{l+p_C, f} x_{i,l}, \quad f = 1, \ldots, k, \qquad v'_{i,0} := \sum_{l=1}^{p_I} w_{l+p_C} x_{i,l} + \sum_{l=1}^{p_I} \sum_{l' > l} x_{i,l+p_C} x_{i,l'+p_C} \langle v_{l+p_C}, v_{l'+p_C} \rangle.$$

变换后的因子 $v'$ 可用于定义矩阵分解模型：

$$\hat{y}(i|c) := \sum_{f=0}^{k} v'_{c,f} v'_{i,f}. \qquad (26}$$

该模型对于式(24)的FM是排名不变的，因此MF节中的所有推导都可以在此应用（现在使用 $v'$ 代替 $v$）。排名不变性的证明直接来自将 $v'$ 的定义代入式(26)，其结果等于式(24)除了一个常数（与排名无关的）项之外。

**高效算法。** 为了从FM评分模型中采样负item，需要对矩阵分解算法（见图4）稍作调整：因子维度 $f \in \{0, \ldots, k\}$ 根据 $p(f|c) \propto |v'_{c,f}| \sigma_f$ 采样，其中 $V'$ 是根据式(25)变换后的向量。其次，item的排序 $\hat{r}^{-1}$ 也不是在原始因子 $V$ 上生成，而是在变换后的 $V'$ 上生成。

注意，变换后的表示 $V'$ 仅用于采样负item。SGD参数学习仍应在原始参数 $w, V$ 上进行。



## 5 评估

在两个实际推荐任务上研究了所提出的**过采样算法**的性质。考察了 **预测质量 和 梯度幅度的收敛行为**。

### 5.1 实验设置

**数据集与模型。** 使用了来自BBC的视频数据集（播放事件，用户-电影pair）和来自ECML PKDD 2009 Discovery Challenge的社交标签数据集（标注事件，用户-文章-标签三元组）。对于BBC，上下文是用户；对于ECML'09，上下文是用户-文章对。对于BBC数据集，从随机选择的100,000个用户中选取所有活动。从结果集合中，每个用户随机选择一个活动（仅针对至少有10个活动的用户）到测试集 $S_{test}$，其余活动形成训练集 $S$。超参数（学习率、正则化和采样率）在第二个随机选择的100,000个用户子集上调优，该子集通过上述相同协议生成。对于ECML'09数据集，我们使用了挑战赛的官方划分。

对于二元的BBC数据集，使用矩阵分解模型。对于三元的ECML'09数据集，应用了获奖的成对交互张量分解（PITF）模型[15]。总共研究了简单的MF模型和复杂模型（PITF）。使用成对学习（见2.2节）来学习模型参数。

<img src="/Users/dazhang/PycharmProject/Papers/3-RecSys/.picture/image-20260717172356710.png" alt="image-20260717172356710" style="zoom: 33%;" />

**比较的采样方法。** 负item采样器被变化为：

1. **均匀采样：** 等同于常见的BPR算法[14]（图2）。
2. **静态过采样：** 3.1节描述的算法，从全局流行度分布中采样负item。
3. **自适应过采样：** 3.2节提出的采样分布，使用图4的算法，根据该上下文的估计排名采样负item。

**度量与协议。** 推荐质量在测试集 $S_{test}$ 上评估。对于测试数据集中的每个上下文，生成一个排名，并测量测试集中的item出现在哪些排名位置。报告所有测试上下文的平均度量，所选度量包括：截断为1000的平均精度（MAP）和半衰期效用（HLU）[2]。采样pair的影响通过平均梯度幅度来衡量。对每个采样pair测量梯度幅度（式9），并报告一个训练epoch上的平均幅度。一个训练epoch定义为 $10 \cdot |S|$ 次SGD更新步骤。

**实验可重复性。** 所有报告的结果使用分解维度 $k = 64$。$k = 16, 32, 128$ 的结果表现类似，但出于空间原因省略。BBC的超参数为：学习率 $\eta = 0.05$，随机高斯初始化 $\mathcal{N}(0, 0.01)$，过采样正则化0.01，均匀采样正则化0.001，几何分布 $\lambda = 500$，训练epoch数为256。ECML'09的超参数不同之处在于：过采样正则化为0.005，均匀采样正则化为0.00005，epoch数为2000。学习算法的源代码可从我们的网站获取。

### 5.2 预测质量

<img src="/Users/dazhang/PycharmProject/Papers/3-RecSys/.picture/image-20260717172630829.png" alt="image-20260717172630829" style="zoom:50%;" />

图5显示了预测质量作为训练epoch的函数。为比较，还展示了一个非个性化的最常见基线模型，该模型根据观测数据 $S$ 中的全局流行度对item进行排序。所有三种个性化模型（无论采样算法如何）都优于该基线。

**自适应过采样 vs. 均匀采样（BPR）。** 在两个数据集和两种度量上，提出的自适应过采样比BPR算法具有更快的收敛速度和更好的预测质量。更陡峭的学习曲线证实了过采样改进了学习。在两个数据集上，自适应过采样在很少几次迭代后就达到了BPR需要数百次迭代才能达到的相同精度。例如，对于ECML'09，自适应过采样大约50次迭代后的MAP与BPR 1000次迭代后的质量相当。同样对于BBC，自适应过采样在10次迭代后的质量已经优于BPR 100次迭代的质量。从长期来看（特别是ECML'09的2000次迭代），BPR缓慢地追赶自适应过采样，这强调了均匀采样（BPR）生成了许多无用的训练pair。

**静态过采样。** 第二个观察结果是，使用全局item分布的简单静态过采样无法产生有竞争力的质量。在最开始的几次迭代中，静态过采样似乎效果不错，优于标准BPR。然而，收敛过早停止，在所有数据集和度量上，最终质量远差于BPR或自适应过采样。这表明过采样分布应在学习过程中适应模型参数/评分函数。自适应过采样表明在整个训练过程中采样有意义的负item是可能的。

**运行时。** 在BBC数据集上一个训练epoch的经验运行时间从BPR的12秒增加到自适应过采样的16秒。这证实了自适应过采样没有增加计算复杂度，经验开销只是微不足道的。

### 5.3 梯度幅度

![image-20260717172716648](/Users/dazhang/PycharmProject/Papers/3-RecSys/.picture/image-20260717172716648.png)

图6显示了平均梯度幅度。第一个观察结果是，两种过采样方法都成功地提高了梯度幅度，即采样的pair导致了实际改变模型参数的SGD更新（式8）。例如，在BBC上，自适应过采样256次迭代后的平均梯度为0.034，而BPR为0.004——提高了8.5倍。对于ECML'09，自适应过采样2000次迭代后的数字为0.016，BPR为0.0005——提高了32倍。

比较静态过采样和自适应过采样，静态过采样在所有epoch中甚至具有比自适应过采样更大的平均梯度幅度。然而，图5中的定性结果显示静态过采样的预测质量劣于自适应过采样，甚至劣于均匀采样（长期来看）。这表明大的梯度幅度本身是不够的，采样的pair还应该具有信息性。定性结果（图5）表明自适应过采样满足这两个要求。



## 6 相关工作

分解模型在现代推荐系统中扮演着核心角色。流行的矩阵分解模型（例如[16]）已经被扩展到许多推荐场景，例如使用隐式信息[10, 18]、时间[11]、邻域信息[10]或属性[20]。**对于上下文感知的设置，张量分解方法已被应用（例如[9, 13]）。尽管这些工作解决了评分预测任务（即回归），我们的工作处理的是从隐式（单类）反馈中的item推荐（即排序）。item推荐任务要困难得多，因为优化目标没有被直接观测到。**Hu等人[6]和Pan等人[12]研究了从隐式反馈中的item推荐，并提出通过将所有未观测值归入0并应用回归，将单类问题转化为两类问题。这类似于标准的（代数）奇异值分解（SVD），但包含了置信权重和L2正则化。**Weimer等人[21]优化了针对排序度量NDCG的矩阵分解模型**。该方法主要针对存在评分（或其他用户反馈排序信息）的数据设计。最近，CLiMF [19]被提出用于在隐式反馈数据集上为倒数排名度量优化矩阵分解模型。贝叶斯个性化排序（BPR）[14]是一个从隐式反馈中学习推荐系统的通用优化框架（不限于矩阵分解）。BPR已被应用于矩阵分解和最近邻[14]、用于标签推荐的张量分解[15]、用于具有分类法感知的序列购物篮推荐的分解马尔可夫链[8]、社交更新流[4]和关系抽取[17]。**所有关于BPR的这些工作都使用均匀采样假设，因此应该会遭受收敛缓慢的问题**。我们提出的自适应采样器有潜力改进所有基于BPR学习的现有推荐系统。

**Gantner等人[3]将BPR扩展到非均匀采样，其中负item的采样概率是问题描述中先验给定的权重**。在我们的工作中，采样概率不是固定的，而是**从当前模型生成的**。在WARP算法[22]中，负item（= 'comment'）被重复抽取，直到抽取到的item的得分足够大。该算法增加了运行时间，因为在执行SGD更新（成本为 $O(T_{pred})$）之前，需要抽取多达 $N \leq |I|$ 个样本并每次计算其得分（成本为 $O(N T_{pred})$）。此外，在拒绝采样器中，WARP中的负item是均匀采样的，这可能需要大量的抽取次数 $N$ 才能找到一个不在尾部的item。



## 7 结论

在本文中，我们展示了如何 **提高BPR风格学习算法的收敛速度**。其**动机是负item的均匀采样导致大多数更新是无信息性的**。我们提出了一种 **自适应且上下文相关的负item采样分布**，它对排名靠前的item进行过采样。我们为MF和一大类分解机模型（包括PITF、属性感知MF或序列MF）开发了一种高效的近似采样算法。所提出的算法具有摊还常数运行时间。实验表明，相对于常见BPR的计算开销约为33%。在两个真实世界数据集上，所提出的采样器大幅提高了收敛速度——在我们的实验中大约**提高了10到20倍**。我们期望我们的改进算法对基于BPR算法的现有推荐系统也具有价值，例如[17, 8, 7, 1, 5]。



## 8 致谢

我们感谢BBC的Chris Newell为我们提供 **匿名的视频推荐数据集**。本工作得到了德国研究基金会（DFG）研究培训小组GK-1042 "Explorative Analysis and Visualization of Large Information Spaces"（康斯坦茨大学）的支持。



## 参考文献

[1] A. Ahmed, B. Kanagal, S. Pandey, V. Josifovski, L. G. Pueyo, and J. Yuan. Latent factor models with additive and hierarchically-smoothed user preferences. In Proceedings of the sixth ACM international conference on Web search and data mining, WSDM '13, pages 385–394, New York, NY, USA, 2013. ACM.

[2] J. S. Breese, D. Heckerman, and C. Kadie. Empirical analysis of predictive algorithms for collaborative filtering. In Proceedings of the Fourteenth Conference on Uncertainty in Artificial Intelligence (UAI-98), pages 43–52, San Francisco, 1998. Morgan Kaufmann.

**[3] Z. Gantner, L. Drumond, C. Freudenthaler, and L. Schmidt-Thieme. Personalized ranking for non-uniformly sampled items. Journal of Machine Learning Research Workshop and Conference Proceedings, 2012.**

[4] L. Hong, R. Bekkerman, J. Adler, and B. D. Davison. Learning to rank social update streams. In Proceedings of the 35th international ACM SIGIR conference on Research and development in information retrieval, SIGIR '12, pages 651–660, New York, NY, USA, 2012. ACM.

[5] L. Hong, A. S. Doumith, and B. D. Davison. Co-factorization machines: modeling user interests and predicting individual decisions in twitter. In Proceedings of the sixth ACM international conference on Web search and data mining, WSDM '13, pages 557–566, New York, NY, USA, 2013. ACM.

[6] Y. Hu, Y. Koren, and C. Volinsky. Collaborative filtering for implicit feedback datasets. In IEEE International Conference on Data Mining (ICDM 2008), pages 263–272, 2008.

[7] B. Kanagal, A. Ahmed, S. Pandey, V. Josifovski, L. Garcia-Pueyo, and J. Yuan. Focused matrix factorization for audience selection in display advertising. In Data Engineering (ICDE), 2013 IEEE 29th International Conference on, pages 386–397, 2013.

[8] B. Kanagal, A. Ahmed, S. Pandey, V. Josifovski, J. Yuan, and L. G. Pueyo. Supercharging recommender systems using taxonomies for learning user purchase behavior. PVLDB, 5(10):956–967, 2012.

[9] A. Karatzoglou, X. Amatriain, L. Baltrunas, and N. Oliver. Multiverse recommendation: n-dimensional tensor factorization for context-aware collaborative filtering. In RecSys '10: Proceedings of the fourth ACM conference on Recommender systems, pages 79–86, New York, NY, USA, 2010. ACM.

[10] Y. Koren. Factorization meets the neighborhood: a multifaceted collaborative filtering model. In KDD '08: Proceeding of the 14th ACM SIGKDD international conference on Knowledge discovery and data mining, pages 426–434, New York, NY, USA, 2008. ACM.

[11] Y. Koren. Collaborative filtering with temporal dynamics. In KDD '09: Proceedings of the 15th ACM SIGKDD international conference on Knowledge discovery and data mining, pages 447–456, New York, NY, USA, 2009. ACM.

[12] R. Pan, Y. Zhou, B. Cao, N. N. Liu, R. M. Lukose, M. Scholz, and Q. Yang. One-class collaborative filtering. In IEEE International Conference on Data Mining (ICDM 2008), pages 502–511, 2008.

[13] S. Rendle. Factorization machines with libFM. ACM Trans. Intell. Syst. Technol., 3(3):57:1–57:22, May 2012.

[14] S. Rendle, C. Freudenthaler, Z. Gantner, and L. Schmidt-Thieme. BPR: Bayesian personalized ranking from implicit feedback. In Proceedings of the 25th Conference on Uncertainty in Artificial Intelligence (UAI 2009), 2009.

[15] S. Rendle and L. Schmidt-Thieme. Pairwise interaction tensor factorization for personalized tag recommendation. In WSDM '10: Proceedings of the third ACM international conference on Web search and data mining, pages 81–90, New York, NY, USA, 2010. ACM.

[16] J. D. M. Rennie and N. Srebro. Fast maximum margin matrix factorization for collaborative prediction. In ICML '05: Proceedings of the 22nd international conference on Machine learning, pages 713–719. ACM, 2005.

[17] S. Riedel, L. Yao, B. M. Marlin, and A. McCallum. Relation extraction with matrix factorization and universal schemas. In Joint Human Language Technology Conference/Annual Meeting of the North American Chapter of the Association for Computational Linguistics (HLT-NAACL '13), June 2013.

[18] R. Salakhutdinov and A. Mnih. Probabilistic matrix factorization. In Advances in Neural Information Processing Systems, volume 20, 2008.

[19] Y. Shi, A. Karatzoglou, L. Baltrunas, M. Larson, N. Oliver, and A. Hanjalic. Climf: learning to maximize reciprocal rank with collaborative less-is-more filtering. In Proceedings of the sixth ACM conference on Recommender systems, RecSys '12, pages 139–146, New York, NY, USA, 2012. ACM.

[20] D. H. Stern, R. Herbrich, and T. Graepel. Matchbox: large scale online bayesian recommendations. In Proceedings of the 18th international conference on World wide web, WWW '09, pages 111–120, New York, NY, USA, 2009. ACM.

[21] M. Weimer, A. Karatzoglou, Q. V. Le, and A. J. Smola. CoFi rank - maximum margin matrix factorization for collaborative ranking. In J. Platt, D. Koller, Y. Singer, and S. Roweis, editors, Advances in Neural Information Processing Systems 20, pages 1593–1600, Cambridge, MA, 2008. MIT Press.

[22] J. Weston, S. Bengio, and N. Usunier. Wsabie: scaling up to large vocabulary image annotation. In Proceedings of the Twenty-Second international joint conference on Artificial Intelligence - Volume Volume Three, IJCAI'11, pages 2764–2770. AAAI Press, 2011.
