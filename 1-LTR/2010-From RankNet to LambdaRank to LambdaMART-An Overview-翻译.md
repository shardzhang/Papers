# From RankNet to LambdaRank to LambdaMART: An Overview

> Christopher J.C. Burges | Microsoft Research Technical Report MSR-TR-2010-82

LambdaMART是LambdaRank的boosted tree版本，后者基于RankNet。RankNet、LambdaRank和LambdaMART已被证明是解决实际排序问题非常成功的算法：例如，一组LambdaMART排序器赢得了2010年Yahoo! Learning To Rank挑战赛Track 1。这些算法的细节分散在多篇论文和报告中，因此本文给出了一个自包含、详细且完整的描述。

- 对RankNet、LambdaRank和LambdaMART算法进行了完整而统一的阐述
- 证明了可以通过直接指定梯度（lambda值）来优化非平滑的IR度量，而无需显式定义代价函数
- 展示了LambdaMART如何高效地将MART（梯度提升树）与LambdaRank的思想结合
- 提供了分类和排序场景中牛顿步长的详细推导

---

## 摘要

LambdaMART是LambdaRank的boosted tree版本，后者基于RankNet。RankNet、LambdaRank和LambdaMART已被证明是解决实际排序问题非常成功的算法：例如，一组LambdaMART排序器赢得了最近的Yahoo! Learning To Rank挑战赛（Track 1）[5]。虽然本文将专注于排序，但很容易将MART（尤其是LambdaMART）修改为适用于广泛的监督学习问题（包括最大化信息检索函数，如NDCG，这些函数关于模型得分不是平滑的）。

本文试图给出这些算法的自包含解释。所需的数学背景仅仅是基本的向量微积分；假设读者对学习排序问题有一定了解。我们希望本概述足够自包含，例如，希望训练一个boosted tree模型来优化某个信息检索指标的读者，能够理解如何使用这些方法来实现这一目标。全文以Web搜索排序作为具体示例。灰色部分为背景材料，对理解主线并非必需。这些思想在数年间陆续提出并出现在多篇论文中；本报告的目的是将所有思想汇总到一个易于查阅的地方（并在需要时补充背景和更多细节）。为保持简洁，本章不呈现实验结果，也不与其他学习排序方法进行比较（此类方法有很多）；这些主题已在其他地方广泛讨论。

## 1 引言

LambdaMART是LambdaRank的boosted tree版本，后者基于RankNet。RankNet、LambdaRank和LambdaMART已被证明是解决实际排序问题非常成功的算法：例如，一组LambdaMART排序器赢得了最近的Yahoo! Learning To Rank挑战赛（Track 1）[5]。虽然本文将专注于排序，但很容易将MART（尤其是LambdaMART）修改为适用于广泛的监督学习问题（包括最大化信息检索函数，如NDCG，这些函数关于模型得分不是平滑的）。

本文试图给出这些算法的自包含解释。所需的数学背景仅仅是基本的向量微积分；假设读者对学习排序问题有一定了解。我们希望本概述足够自包含，例如，希望训练一个boosted tree模型来优化某个信息检索指标的读者，能够理解如何使用这些方法来实现这一目标。全文以Web搜索排序作为具体示例。灰色部分为背景材料，对理解主线并非必需。这些思想在数年间陆续提出并出现在多篇论文中；本报告的目的是将所有思想汇总到一个易于查阅的地方（并在需要时补充背景和更多细节）。为保持简洁，本章不呈现实验结果，也不与其他学习排序方法进行比较（此类方法有很多）；这些主题已在其他地方广泛讨论。

## 2 RankNet

对于RankNet [2]，底层模型可以是任何输出为模型参数可微函数的模型（通常我们使用神经网络，但我们也使用boosted trees实现了RankNet，将在下文描述）。RankNet的训练过程如下。训练数据按查询划分。在训练过程中的某个时刻，RankNet将一个输入特征向量$x \in \mathbb{R}^n$映射为一个数值$f(x)$。对于给定查询，选择每一对标签不同的URL $U_i$和$U_j$，并将每一对（含特征向量$x_i$和$x_j$）输入模型，模型计算得分$s_i = f(x_i)$和$s_j = f(x_j)$。令$U_i \triangleright U_j$表示$U_i$应排在$U_j$之前的事件（例如，因为$U_i$被标记为"优秀"而$U_j$被标记为"差"；注意同一URL在不同查询下的标签可能不同）。模型的两个输出通过sigmoid函数映射为一个学习到的概率，表示$U_i$应排在$U_j$之前的概率，即：

$$P_{ij} \equiv P(U_i \triangleright U_j) \equiv \frac{1}{1 + e^{-\sigma(s_i - s_j)}}$$

其中参数$\sigma$的选择决定了sigmoid的形状。使用sigmoid是神经网络训练中已知的手段，已被证明能产生良好的概率估计[1]。然后我们应用交叉熵代价函数，该函数惩罚模型输出概率与期望概率之间的偏差：令$\bar{P}_{ij}$为训练URL $U_i$应排在训练URL $U_j$之前的已知概率，则代价为：

$$C = -\bar{P}_{ij} \log P_{ij} - (1 - \bar{P}_{ij}) \log(1 - P_{ij})$$

对于给定查询，定义$S_{ij} \in \{0, \pm 1\}$，如果文档$i$被标记为比文档$j$更相关则为1，如果文档$i$被标记为比文档$j$更不相关则为-1，如果标签相同则为0。在本文中，我们假设期望排序是确定已知的，因此$\bar{P}_{ij} = \frac{1}{2}(1 + S_{ij})$。（注意该模型可以处理更一般的测量概率情况，例如$\bar{P}_{ij}$可以通过将配对展示给多位评审员来估计）。结合以上两个方程得到：

$$C = \frac{1}{2}(1 - S_{ij})\sigma(s_i - s_j) + \log(1 + e^{-\sigma(s_i - s_j)})$$

该代价具有令人满意的对称性（交换$i$和$j$并改变$S_{ij}$的符号应保持代价不变）：对于$S_{ij} = 1$，

$$C = \log(1 + e^{-\sigma(s_i - s_j)})$$

而对于$S_{ij} = -1$，

$$C = \log(1 + e^{-\sigma(s_j - s_i)})$$

注意当$s_i = s_j$时代价为$\log 2$，因此模型包含了一个间隔（即标签不同但模型赋予相同得分的文档在排序中仍会被相互推开）。此外，渐近地，代价变为线性（如果得分给出了错误的排序）或零（如果得分给出了正确的排序）。由此得到：

$$\frac{\partial C}{\partial s_i} = \sigma \left( \frac{1}{2}(1 - S_{ij}) - \frac{1}{1 + e^{\sigma(s_i - s_j)}} \right) = -\frac{\partial C}{\partial s_j} \qquad (1)$$

该梯度用于更新权重$w_k \in \mathbb{R}$（即模型参数），通过随机梯度下降来降低代价[^1]：

$$w_k \to w_k - \eta \frac{\partial C}{\partial w_k} = w_k - \eta \left( \frac{\partial C}{\partial s_i} \frac{\partial s_i}{\partial w_k} + \frac{\partial C}{\partial s_j} \frac{\partial s_j}{\partial w_k} \right) \qquad (2)$$

其中$\eta$是一个正的学习率（使用验证集选择的参数；在我们的实验中通常为$1 \times 10^{-3}$到$1 \times 10^{-5}$）。显式地：

$$\delta C = \sum_k \frac{\partial C}{\partial w_k} \delta w_k = \sum_k \frac{\partial C}{\partial w_k} \left( -\eta \frac{\partial C}{\partial w_k} \right) = -\eta \sum_k \left( \frac{\partial C}{\partial w_k} \right)^2 < 0$$

通过梯度下降进行学习的思想是本文的一个关键思想（即使期望的代价没有良定义的梯度，甚至当模型（如boosted trees集成）没有可微参数时也是如此）：要更新模型，我们必须指定代价关于模型参数$w_k$的梯度，而要做到这一点，我们需要代价关于模型得分$s_i$的梯度。Boosted trees（如MART [8]）的梯度下降公式通过直接建模$\partial C / \partial s_i$绕过了计算$\partial C / \partial w_k$的需求。

[^1]: 我们使用如下约定：如果两个量以乘积形式出现且共享一个索引，则对该索引求和。

### 2.1 分解RankNet：加速RankNet训练

上述推导引出了导致LambdaRank [4]的关键观察的一个分解：对于一对给定的URL $U_i$, $U_j$（再次假定对重复索引求和）：

$$\frac{\partial C}{\partial w_k} = \frac{\partial C}{\partial s_i} \frac{\partial s_i}{\partial w_k} + \frac{\partial C}{\partial s_j} \frac{\partial s_j}{\partial w_k} = \sigma \left( \frac{1}{2}(1 - S_{ij}) - \frac{1}{1 + e^{\sigma(s_i - s_j)}} \right) \left( \frac{\partial s_i}{\partial w_k} - \frac{\partial s_j}{\partial w_k} \right) = \lambda_{ij} \left( \frac{\partial s_i}{\partial w_k} - \frac{\partial s_j}{\partial w_k} \right)$$

其中我们定义了：

$$\lambda_{ij} \equiv \frac{\partial C(s_i - s_j)}{\partial s_i} = \sigma \left( \frac{1}{2}(1 - S_{ij}) - \frac{1}{1 + e^{\sigma(s_i - s_j)}} \right) \qquad (3)$$

令$I$表示我们希望$U_i$与$U_j$排序不同的索引对$\{i, j\}$的集合（对于给定查询）。$I$必须每对只包含一次，因此方便采用如下约定：$I$包含满足$U_i \triangleright U_j$的索引对$\{i, j\}$，因此$S_{ij} = 1$（这大大简化了符号，我们将从此时起采用此约定）。注意由于RankNet从概率学习并输出概率，它不要求URL必须有标签；它只需要集合$I$，该集合也可以通过收集成对偏好来确定（这更加通用，因为它可以是不一致的：例如一个困惑的评审员可能判定对于给定查询，$U_1 \triangleright U_2$，$U_2 \triangleright U_3$且$U_3 \triangleright U_1$）。现在将所有对权重$w_k$更新的贡献求和：

$$\delta w_k = -\eta \sum_{\{i,j\} \in I} \left( \lambda_{ij} \frac{\partial s_i}{\partial w_k} - \lambda_{ij} \frac{\partial s_j}{\partial w_k} \right) \equiv -\eta \sum_i \lambda_i \frac{\partial s_i}{\partial w_k}$$

其中我们引入了$\lambda_i$（每个URL对应一个$\lambda_i$：注意带单下标的$\lambda$是带双下标的$\lambda$的和）。要计算$\lambda_i$（对于URL $U_i$），我们找到所有使得$\{i, j\} \in I$的$j$和所有使得$\{k, i\} \in I$的$k$。对于前者，我们在$\lambda_i$上增加$\lambda_{ij}$；对于后者，我们在$\lambda_i$上减去$\lambda_{ki}$。例如，如果只有一对$U_1 \triangleright U_2$，则$I = \{\{1, 2\}\}$，且$\lambda_1 = \lambda_{12} = -\lambda_2$。一般情况下：

$$\lambda_i = \sum_{j: \{i,j\} \in I} \lambda_{ij} - \sum_{j: \{j,i\} \in I} \lambda_{ij} \qquad (4)$$

正如我们将在下文看到的，你可以将$\lambda$视为小箭头（或力），每个箭头附着在（排序后的）URL上，其方向表示我们希望URL移动的方向（以增加相关性），其长度表示移动的幅度，其中给定URL的$\lambda$由该URL参与的所有配对计算得出。当我们首次实现RankNet时，我们使用了真正的随机梯度下降：每检查一对URL（标签不同）就更新一次权重。以上推导表明，我们可以改为累积每个URL的$\lambda$，将其从所有URL对（其中一对由两个标签不同的URL组成）中的贡献求和，然后进行更新。这是小批量学习，其中首先针对给定查询计算所有权重更新，然后应用之，但加速来自于问题分解的方式，而非仅仅使用小批量。这导致了RankNet训练速度的显著提升（因为权重更新代价高昂，例如对于神经网络模型，它需要反向传播）。实际上训练时间从接近查询内URL数量的二次方降至接近线性。这也为LambdaRank奠定了基础，但在讨论LambdaRank之前，让我们先回顾一下我们希望学习的信息检索指标。

## 3 信息检索指标

信息检索研究人员使用排序质量指标，如平均倒数排名（MRR）、平均平均精度（MAP）、期望倒数排名（ERR）和归一化折损累计增益（NDCG）。NDCG [9]和ERR [6]的优势在于它们处理多级相关性（而MRR和MAP是为二元相关性级别设计的），并且指标包含对展示给用户的结果的位置依赖性（给予排名靠前的结果更高权重），这对Web搜索尤为适合。然而，所有这些指标都有一个不幸的性质：作为模型得分的函数，它们处处要么不连续要么平坦，因此梯度下降似乎存在问题，因为梯度处处为零或未定义。例如，NDCG定义如下。对于给定的一组搜索结果（对于给定查询），DCG（折损累计增益）为：

$$DCG@T \equiv \sum_{i=1}^T \frac{2^{l_i} - 1}{\log(1 + i)} \qquad (5)$$

其中$T$是截断级别（例如，如果我们只关心返回结果的第一页，可能取$T = 10$），$l_i$是第$i$个列出URL的标签。我们通常使用五级相关性：$l_i \in \{0, 1, 2, 3, 4\}$。NDCG是归一化版本：

$$NDCG@T \equiv \frac{DCG@T}{\max DCG@T}$$

其中分母是该查询能达到的最大$DCG@T$，因此$NDCG@T \in [0, 1]$。

ERR是最近引入的[6]，其灵感来自级联模型，其中假设用户从上到下浏览返回的URL列表直到找到喜欢的为止。ERR定义为：

$$\sum_{r=1}^n \frac{1}{r} R_r \prod_{i=1}^{r-1} (1 - R_i)$$

其中，如果$l_m$是最大标签值，则：

$$R_i = \frac{2^{l_i} - 1}{2^{l_m}}$$

$R_i$模拟用户在排名位置$i$发现文档相关性的概率。直观上可能认为计算$\Delta ERR$（交换两个文档的排名而保持其他所有排名不变所导致的ERR变化）的代价对于给定查询会是文档数量的三次方，因为排名介于这两个文档之间的文档也会贡献于$\Delta ERR$，并且这种贡献必须对每对文档进行计算。然而计算具有二次代价（二次代价源于需要为每对标签不同的文档计算$\Delta ERR$），因为它可以按如下方式排序。令$\Delta$表示$\Delta ERR$。令$T_i \equiv 1 - R_i$，创建数组$A$，其第$i$个分量是仅计算到级别$i$的ERR。（计算$A$具有线性复杂度）。然后：

1. 计算$\pi_i \equiv \prod_{n=1}^i T_n$对于$i > 0$，并设$\pi_0 = 1$。
2. 设$\Delta = \pi_{i-1}(R_i - R_j) / r_i$。
3. 增加$\Delta$：

$$\Delta = (T_i - T_j) \left( \frac{R_{i+1}}{r_{i+1}} + \frac{T_{i+1}R_{i+2}}{r_{i+2}} + \cdots + \frac{T_{i+1} \cdots T_{j-2}R_{j-1}}{r_{j-1}} \right) \pi_{i-1} \equiv (T_i - T_j) \left( \frac{A_{j-1} - A_i}{T_i} \right)$$

4. 进一步增加$\Delta$：

$$\Delta = \frac{\pi_{i-1}}{r_j} T_{i+1}T_{i+2} \cdots T_{j-1} (T_i R_j - T_j R_i) = \frac{\pi_{j-1}}{r_j} \left( R_j - \frac{T_j R_i}{T_i} \right)$$

注意与NDCG不同，交换两个URL $U_i$和$U_j$（保持所有其他URL的排名不变）所引起的ERR变化取决于排名在$U_i$和$U_j$之间的URL的标签和排名。因此，好奇的读者可能会想ERR是否具有一致性：如果$U_i \triangleright U_j$，且交换两个排名位置，ERR是否必然降低？不难看出这是必然的，通过证明当$R_i > R_j$时上述$\Delta$非负。整理各项，我们有：

$$\Delta = \frac{\pi_i (R_i - R_j)}{T_i} \left( \frac{1}{r_i} - \frac{R_{i+1}}{r_{i+1}} - \frac{T_{i+1}R_{i+2}}{r_{i+2}} - \cdots - \frac{T_{i+1} \cdots T_{j-2}R_{j-1}}{r_{j-1}} - \frac{T_{i+1} \cdots T_{j-1}}{r_j} \right)$$

由于$r_{i+1} > r_i$，如果满足以下条件，这必然非负：

$$R_{i+1} + T_{i+1}R_{i+2} + \cdots + T_{i+1} \cdots T_{j-2}R_{j-1} + T_{i+1} \cdots T_{j-1} \leq 1$$

但由于$R_i + T_i = 1$，左边恰好坍缩为1。

![图1：针对给定查询使用二元相关性指标排序的一组URL。浅灰色条代表与查询不相关的URL，深蓝色条代表与查询相关的URL。左图：成对错误总数为十三。右图：通过将顶部URL向下移动三个排名级别，并将底部相关URL向上移动五个级别，成对错误总数减少到十一。然而对于像NDCG和ERR这样强调顶部结果的IR指标，这不是我们想要的。左图上的（黑色）箭头表示RankNet梯度（随成对错误数量增加而增加），而我们真正想要的是右图上的（红色）箭头。](fig:lambda_vs_ranknet)

## 4 LambdaRank

虽然通过简单地将指标用作验证集上的停止准则可以使RankNet在上述指标上工作得相当好，但我们可以做得更好。RankNet优化的是（平滑凸近似下的）成对错误数量，如果这正是期望的代价固然很好，但它与其他一些信息检索指标的匹配不佳。图1是一个示意问题所在的示意图。直接写出期望的梯度（图中以箭头表示）而非从代价推导，是LambdaRank [4]的核心思想之一：它使我们能够绕过大多数IR指标中排序引入的困难。注意这并不意味着这些梯度不是某个代价的梯度。在本节中，为具体起见，我们假设正在设计一个学习NDCG的模型。

### 4.1 从RankNet到LambdaRank

因此LambdaRank的关键观察是：要训练一个模型，我们不需要代价本身：我们只需要梯度（代价关于模型得分的梯度）。上述提到的箭头（$\lambda$）正是这些梯度。给定URL $U_1$的$\lambda$从同一查询中所有标签不同的其他URL获得贡献。$\lambda$也可以解释为力（当力是保守力时，它们是势函数的梯度）：如果$U_2$比$U_1$更相关，则$U_1$将获得一个大小为$|\lambda|$的向下推力（而$U_2$获得一个大小相等、方向相反的向上推力）；如果$U_2$比$U_1$更不相关，则$U_1$将获得一个大小为$|\lambda|$的向上推力（而$U_2$获得一个大小相等、方向相反的向下推力）。

实验表明，只需将式(3)乘以交换$U_1$和$U_2$的排名位置（保持所有其他URL的排名位置不变）所导致的NDCG变化大小（$|\Delta NDCG|$），就能得到非常好的结果[4]。因此在LambdaRank中，我们设想存在一个效用$C$使得[^2]：

$$\lambda_{ij} = \frac{\partial C(s_i - s_j)}{\partial s_i} = \frac{-\sigma}{1 + e^{\sigma(s_i - s_j)}} |\Delta NDCG| \qquad (6)$$

由于这里我们希望最大化$C$，方程(2)（对于第$k$个权重的贡献）被替换为：

$$w_k \to w_k + \eta \frac{\partial C}{\partial w_k}$$

因此：

$$\delta C = \frac{\partial C}{\partial w_k} \delta w_k = \eta \left( \frac{\partial C}{\partial w_k} \right)^2 > 0 \qquad (7)$$

因此，尽管信息检索指标作为模型得分的函数处处平坦或不连续，LambdaRank的思想通过在URL按其得分排序后计算梯度来绕过这一问题。我们的实验经验表明，有趣的是，这样的模型实际上直接优化了NDCG [12, 7]。事实上我们进一步证明，如果想要优化其他信息检索指标，如MRR或MAP，可以简单地修改LambdaRank来完成：唯一的变化是将上述$|\Delta NDCG|$替换为所选IR指标的相应变化[7]。

对于给定配对，计算一个$\lambda$，$U_1$和$U_2$的$\lambda$都增加该$\lambda$值，符号选择使得$s_2 - s_1$变得更负，从而$U_1$倾向于在排序列表中上移而$U_2$倾向于下移。再次，给定多于一对URL，如果每个URL $U_i$有一个得分$s_i$，那么对于任何特定配对$\{U_i, U_j\}$（回顾我们假设$U_i$比$U_j$更相关），我们分离计算：

$$\delta s_i = \frac{\partial s_i}{\partial w_k} \delta w_k = \frac{\partial s_i}{\partial w_k} \left( -\eta \lambda \frac{\partial s_i}{\partial w_k} \right) < 0$$

$$\delta s_j = \frac{\partial s_j}{\partial w_k} \delta w_k = \frac{\partial s_j}{\partial w_k} \left( \eta \lambda \frac{\partial s_j}{\partial w_k} \right) > 0$$

因此，每一对生成一个大小相等方向相反的$\lambda$，对于给定URL，所有lambda值从所有该URL参与且配对另一方标签不同的贡献中累加。对于给定查询，对每个URL进行此累加；当计算完成后，基于计算出的lambda值使用小步长（随机梯度）调整权重。

[^2]: 这里我们从代价切换为效用，因为NDCG是一种优良性指标（我们希望最大化它）。同时回顾$S_{ij} = 1$。

### 4.2 LambdaRank：NDCG（或其他IR指标）的经验性优化

这里我们简要描述我们如何经验性地证明LambdaRank直接优化NDCG [12, 7]。假设我们已经训练了一个模型，其（学习得到的）参数值为$w_k^*$。我们可以通过固定除了一个权重（称为$w_i$）之外的所有权重，计算NDCG（在大量训练查询上平均）如何变化，并形成比率来经验性地估计梯度的平滑版本：

$$\frac{\delta M}{\delta w_i} = \frac{M - M^*}{w_i - w_i^*}$$

其中对于$n$个查询：

$$M \equiv \frac{1}{n} \sum_{i=1}^n NDCG(i) \qquad (8)$$

且其中第$i$个查询的NDCG等于$NDCG(i)$。现在假设我们绘制$M$关于每个$w_i$的函数。如果观察到$M$在$w_i = w_i^*$处对每个$i$都取最大值，那么我们知道函数在学习到的权重值$w = w^*$处具有消失的梯度。（当然，如果以足够高的放大倍数观察图形，我们会发现曲线是小阶梯函数；我们考虑的是曲线平滑尺度上的梯度）。这是必要条件但不充分条件，不足以证明NDCG在$w = w^*$处达到最大值：它可能是一个鞍点。我们可以尝试通过证明Hessian矩阵是负定的来证明该点是最大值，但这一点在多个方面计算上具有挑战性。然而我们可以通过应用单侧蒙特卡洛检验来获得任意紧的界：在权重空间中随机选择足够多的方向，沿每个方向稍微移动权重，并检验$M$在远离$w^*$时是否总是减小。具体地，我们通过从球面高斯分布采样来均匀随机选择方向。令$p$为导致$M$增加的方向比例。则：

$$P(\text{we haven\'t found an ascent direction after n trials}) = (1 - p)^n$$

将$1 - P$称为置信度。如果我们要求99%的置信度（即选择$\delta = 0.01$并要求$P \leq \delta$），那么要使$p \leq p_0$（其中选择$p_0 = 0.01$），$n$需要多大？我们有：

$$(1 - p_0)^n \leq \delta \implies n \geq \frac{\log \delta}{\log(1 - p_0)} \qquad (9)$$

得到$n \geq 459$（即选择459个随机方向并总是发现$M$沿着这些方向减小；注意更大的$p_0$值需要更少的测试）。一般来说，只要至少进行$n = \frac{\log \delta}{\log(1 - p_0)}$次测试，我们就有至少$1 - \delta$的置信度认为$p \leq p_0$。

### 4.3 Lambda何时实际上是梯度？

假设你写下一些任意的$\lambda$值。是否必然存在一个代价，使得这些$\lambda$值是梯度？存在这样一个代价的一般条件是什么？很容易写出不存任何函数的表达式，使得这些表达式是其偏导数：例如，不存在$F(x, y)$使得：

$$\frac{\partial F(x, y)}{\partial x} = 1, \quad \frac{\partial F(x, y)}{\partial y} = x$$

事实证明，函数$f_1(x_1, \ldots, x_n), f_2(x_1, \ldots, x_n), \ldots, f_n(x_1, \ldots, x_n)$是某个函数$F$的偏导数（即$\partial F / \partial x_i = f_i$）的充分必要条件是：

$$\frac{\partial f_i}{\partial x_j} = \frac{\partial f_j}{\partial x_i} \qquad (10)$$

在$F$的定义域$D$是开的且星形的条件下（换句话说，如果$x \in D$，则连接$x$与原点的所有点也在$D$中）（例如参见[10]）。这个结果被称为Poincaré引理。在我们的情况下，对于给定配对，对于模型参数的任何特定值（即，根据当前模型得分对URL的特定排序），很容易写出这样的代价：它是乘以$\Delta NDCG$的RankNet代价。然而一般表达式以复杂的方式依赖于模型参数，因为模型生成用于排序URL的得分，且$\lambda$是在排序后计算的。

## 5 MART

LambdaMART结合了MART [8]和LambdaRank。要理解LambdaMART，我们首先回顾MART。由于MART是一种boosted tree模型，其输出是一组回归树输出的线性组合，我们首先简要回顾回归树。假设给定数据集$\{x_i, y_i\}$，$i = 1, \ldots, m$，其中$x_i \in \mathbb{R}^d$且标签$y_i \in \mathbb{R}$。对于给定向量$x_i$，我们通过$x_{ij}$（$j = 1, \ldots, d$）索引其特征值。首先考虑一个回归桩，包含一个根节点和两个叶节点，有向弧将根节点连接到每个叶节点。我们假设所有数据都在根节点上，对于给定特征，遍历所有样本并找到阈值$t$，使得如果所有满足$x_{ij} \leq t$的样本落入左子节点，其余落入右子节点，则下述和式最小化：

$$S_j \equiv \sum_{i \in L} (y_i - \mu_L)^2 + \sum_{i \in R} (y_i - \mu_R)^2 \qquad (11)$$

其中$L$（$R$）是落入左（右）的样本索引集合，$\mu_L$（$\mu_R$）是落入左（右）的样本标签值的均值。（和式中对$j$的依赖性出现在$L$、$R$以及$\mu_L$、$\mu_R$中）。然后对所有特征$j$的选择和该特征的所有阈值选择计算$S_j$，选择使得总体$S_j$最小的分裂（特定特征$j$和阈值$t$的选择）。然后将该分裂附加到根节点。对于桩的两个叶节点，计算值$\gamma_l$（$l = 1, 2$），这仅仅是落入该处的样本的$y$的均值。在一般的回归树中，此过程持续$L-1$次以形成具有$L$个叶节点的树[^3]。

[^3]: 仅在本段中，我们重载了$L$的符号含义。

MART是一类可视为使用回归树在函数空间中执行梯度下降的boosting算法。最终模型同样将输入特征向量$x \in \mathbb{R}^d$映射到得分$F(x) \in \mathbb{R}$。MART是一类算法而非单一算法，因为它可以被训练来最小化通用的代价（例如解决分类、回归或排序问题）。然而注意，无论MART解决什么问题，其底层模型都是最小二乘回归树。MART的输出$F(x)$可以写为：

$$F_N(x) = \sum_{i=1}^N \alpha_i f_i(x)$$

其中每个$f_i(x) \in \mathbb{R}$是由单个回归树建模的函数，$\alpha_i \in \mathbb{R}$是与第$i$棵回归树关联的权重。$f_i$和$\alpha_i$都在训练过程中学习。给定树$f_i$通过将$x$沿树向下传递将给定的$x$映射为实数值，树中给定节点处的路径（左或右）由特定特征$x_j$（$j = 1, \ldots, d$）的值决定，树的输出取为与每个叶子关联的固定值$\gamma_{kn}$（$k = 1, \ldots, L$，$n = 1, \ldots, N$），其中$L$是叶子数量，$N$是树的数量。给定训练集和验证集，用户选择的训练算法参数为$N$、固定学习率$\eta$（乘以每棵树每个$\gamma_{kn}$）和$L$。（也可以为不同树选择不同的$L$）。$\gamma_{kn}$也在训练过程中学习。

假设已经训练了$n$棵树，下一棵树应如何训练？MART使用梯度下降来减少损失：具体地，下一棵回归树对代价关于当前模型得分在每个训练点处的$m$个导数进行建模：$\frac{\partial C}{\partial F_n(x_i)}$，$i = 1, \ldots, m$。因此：

$$\delta C \approx \frac{\partial C(F_n)}{\partial F_n} \delta F \qquad (12)$$

因此如果我们取$\delta F = -\eta \frac{\partial C}{\partial F_n}$，则$\delta C < 0$。因此每棵树对代价关于模型得分的梯度进行建模，新树以步长$\eta \gamma_{kn}$添加到集成中。步长$\gamma_{kn}$可以在某些情况下精确计算，或在其他情况下使用牛顿近似。$\eta$作为全局学习率的作用类似于其他随机梯度下降算法中的学习率：采用比最优步长（即最大程度减少代价的步长）更小的步长作为一种正则化形式，可以显著提高测试精度。

显然，由于MART对导数建模，而LambdaRank通过在训练过程中任意时刻指定导数来工作，这两种算法非常匹配：LambdaMART是两者的结合[11]。

为了理解MART，接下来让我们详细研究它如何工作，针对可能最简单的监督学习任务：二分类。

## 6 用于二分类的MART

我们遵循[8]，但在此处给出一些不同的侧重点；特别地，我们允许通用的sigmoid参数$\sigma$（虽然$\sigma$的选择不影响模型，但了解其原因是有意义的）。

我们选择标签$y_i \in \{\pm 1\}$（这具有$y_i^2 = 1$的优点，将在后面使用）。样本$x \in \mathbb{R}^n$的模型得分记为$F(x)$。为保持符号简洁，记建模的条件概率为$P_+ \equiv P(y = 1|x)$和$P_- \equiv P(y = -1|x)$，并定义指示函数$I_+(x_i) = 1$如果$y_i = 1$否则为0，以及$I_-(x_i) = 1$如果$y_i = -1$否则为0。我们使用交叉熵损失函数（负二项对数似然）：

$$L(y, F) = -I_+ \log P_+ - I_- \log P_-$$

（注意与RankNet损失的相似性）。逻辑回归对对数几率建模。因此如果$F_N(x)$是模型输出，我们选择（这里$1/2$因子是为了与[8]匹配）：

$$F_N(x) = \frac{1}{2} \log\left( \frac{P_+}{P_-} \right) \qquad (13)$$

或等价地：

$$P_+ = \frac{1}{1 + e^{-2\sigma F_N(x)}}, \quad P_- = 1 - P_+ = \frac{1}{1 + e^{2\sigma F_N(x)}}$$

由此得到：

$$L(y, F_N) = \log(1 + e^{-2y\sigma F_N}) \qquad (14)$$

代价关于模型得分的梯度（在[8]中称为伪响应）为：

$$\tilde{y}_i = -\left[ \frac{\partial L(y_i, F(x_i))}{\partial F(x_i)} \right]_{F(x) = F_{m-1}(x)} = \frac{2y_i\sigma}{1 + e^{2y_i\sigma F_{m-1}(x)}}$$

这些与LambdaRank中的Lambda梯度完全类似（事实上是LambdaMART中的梯度）。它们是回归树建模的值。记落入第$m$棵树第$j$个叶节点的样本集合为$R_{jm}$，我们想为每个叶子找到一个近似最优的步长，即最小化损失的值：

$$\gamma_{jm} = \arg\min_\gamma \sum_{x_i \in R_{jm}} \log\left( 1 + e^{-2y_i\sigma(F_{m-1}(x_i) + \gamma)} \right) \equiv \arg\min_\gamma g(\gamma)$$

使用牛顿近似来寻找$\gamma$：对于函数$g(\gamma)$，牛顿-拉弗森步走向$g$的极值点为：

$$\gamma_{n+1} = \gamma_n - \frac{g'(\gamma_n)}{g''(\gamma_n)}$$

这里我们从$\gamma = 0$开始，只走一步。为简化表达式，我们压缩符号：定义$S_i(\gamma) \equiv 1 + e^{-2v_i} \equiv 1 + e^{-2y_i\sigma(F_{m-1}(x) + \gamma)}$。我们希望计算：

$$\arg\min_\gamma g(\gamma) \equiv \arg\min_\gamma \sum_{x_i \in R_{jm}} \log S_i(\gamma)$$

现在：

$$g' = \sum_{x_i \in R_{jm}} \frac{1}{S_i} (-2y_i\sigma e^{-2v_i})$$

$$g'' = \sum_{x_i \in R_{jm}} \left[ -\frac{1}{S_i^2} (-2y_i\sigma e^{-2v_i})^2 - \frac{2y_i\sigma}{S_i} (-2y_i\sigma) e^{-2v_i} \right] = \sum_{x_i \in R_{jm}} \frac{4}{S_i^2} y_i^2 \sigma^2 e^{-2v_i}$$

但：

$$\tilde{y}_i \equiv \frac{2y_i\sigma}{1 + e^{2y_iF}} \implies g' = \sum_{x_i \in R_{jm}} \frac{-2y_i\sigma}{e^{2v_i}S_i} = \sum_{x_i \in R_{jm}} -\tilde{y}_i$$

同样地：

$$g'' = \sum_{x_i \in R_{jm}} \frac{4y_i^2 \sigma^2}{(1 + e^{2v_i})^2} e^{2v_i}$$

由于$y_i^2 = 1$，我们有：

$$|\tilde{y}_i| = \frac{2\sigma}{1 + e^{2v_i}}$$

因此：

$$|\tilde{y}_i|(2\sigma - |\tilde{y}_i|) = \frac{4\sigma^2 e^{2v_i}}{(1 + e^{2v_i})^2}$$

得到简洁的表达式：

$$\gamma_{jm} = -\frac{g'}{g''} = \frac{\sum_{x_i \in R_{jm}} \tilde{y}_i}{\sum_{x_i \in R_{jm}} |\tilde{y}_i|(2\sigma - |\tilde{y}_i|)}$$

（这对应于[8]中的算法5）。注意此步长同时结合了梯度的估计（分子）和通常的梯度下降步长（此处为$1/g''$）。为包含学习率$r$，我们只需将每个叶节点值$\gamma_{jm}$乘以$r$。

有趣的是，对于该算法，$\sigma$的值没有影响。牛顿步长$\gamma_{jm}$与$1/\sigma$成正比。由于$F_m(x) = F_{m-1}(x) + \gamma$，且$F$在损失中整体乘以$\sigma$，因此在损失函数中$\sigma$被抵消。

### 6.1 扩展到不平衡数据集

为完整起见，我们包含了一节讨论如何将算法扩展到数据非常不平衡的情况（例如正例远多于负例）。

如果$n_+$（$n_-$）是正（负）例的总数，对于不平衡数据，一个有用的错误计数方式是：

$$\text{error rate} = \frac{\text{number false positives}}{2n_-} + \frac{\text{number false negatives}}{2n_+}$$

分母中的因子2仅为缩放使最大错误率为1。我们仍然使用逻辑回归（对对数几率建模），但使用平衡损失：

$$L_B(y, F) = -\frac{I_+}{n_+} \log P_+ - \frac{I_-}{n_-} \log P_- = \left( \frac{I_+}{n_+} + \frac{I_-}{n_-} \right) \log(1 + e^{-2y\sigma F})$$

方程与之前类似。梯度$\tilde{y}_i$被替换为：

$$\tilde{z}_i = -\left[ \frac{\partial L_B(y_i, F(x_i))}{\partial F(x_i)} \right]_{F(x) = F_{m-1}(x)} = \tilde{y}_i \left( \frac{I_+}{n_+} + \frac{I_-}{n_-} \right)$$

牛顿步长变为：

$$\gamma_{jm}^B = -\frac{g'}{g''} = \frac{\sum_{x_i \in R_{jm}} \tilde{y}_i \left( \frac{I_+}{n_+} + \frac{I_-}{n_-} \right)}{\sum_{x_i \in R_{jm}} |\tilde{y}_i|(2\sigma - |\tilde{y}_i|) \left( \frac{I_+}{n_+} + \frac{I_-}{n_-} \right)} \qquad (15)$$

（注意这与将$\tilde{y}_i$替换为$\tilde{z}_i$之前的牛顿步长不同。）

## 7 LambdaMART

要实现LambdaMART，我们只需使用MART，指定适当的梯度和牛顿步长。梯度$\tilde{y}_i$很简单：就是$\lambda_i$。与MART中一贯的做法一样，最小二乘法用于计算分裂。在LambdaMART中，每棵树对整个数据集（而非单个查询）的$\lambda_i$进行建模。为计算牛顿步长，我们汇总以上的部分结果：对于任意一对URL $U_i$和$U_j$，满足$U_i \triangleright U_j$，则在URL按得分排序后，$\lambda_{ij}$定义为（见式(4)）：

$$\lambda_{ij} = \frac{-\sigma |\Delta Z_{ij}|}{1 + e^{\sigma(s_i - s_j)}}$$

其中我们将交换$U_i$和$U_j$的排名位置所产生的效用差异记为$Z_{ij}$（例如$Z$可以是NDCG）。同时：

$$\lambda_i = \sum_{j: \{i,j\} \in I} \lambda_{ij} - \sum_{j: \{j,i\} \in I} \lambda_{ij}$$

为简化符号，将上述求和运算记为：

$$\sum_{\{i,j\} \rightleftharpoons I} \lambda_{ij} \equiv \sum_{j: \{i,j\} \in I} \lambda_{ij} - \sum_{j: \{j,i\} \in I} \lambda_{ij}$$

因此，对于模型的任何给定状态（即任何特定的得分集合），对于特定URL $U_i$，我们可以写出一个效用函数，使得$\lambda_i$是该效用的导数[^4]：

$$C = \sum_{\{i,j\} \rightleftharpoons I} |\Delta Z_{ij}| \log\left( 1 + e^{-\sigma(s_i - s_j)} \right)$$

因此：

$$\frac{\partial C}{\partial s_i} = \sum_{\{i,j\} \rightleftharpoons I} \frac{-\sigma |\Delta Z_{ij}|}{1 + e^{\sigma(s_i - s_j)}} \equiv \sum_{\{i,j\} \rightleftharpoons I} -\sigma |\Delta Z_{ij}| \rho_{ij}$$

其中我们定义了：

$$\rho_{ij} \equiv \frac{1}{1 + e^{\sigma(s_i - s_j)}} = \frac{-\lambda_{ij}}{\sigma |Z_{ij}|}$$

那么：

$$\frac{\partial^2 C}{\partial s_i^2} = \sum_{\{i,j\} \rightleftharpoons I} \sigma^2 |\Delta Z_{ij}| \rho_{ij} (1 - \rho_{ij})$$

（使用了$\frac{e^x}{(1 + e^x)^2} = \left(1 - \frac{1}{1 + e^x}\right) \frac{1}{1 + e^x}$），第$m$棵树第$k$个叶节点的牛顿步长为：

$$\gamma_{km} = \frac{\sum_{x_i \in R_{km}} \frac{\partial C}{\partial s_i}}{\sum_{x_i \in R_{km}} \frac{\partial^2 C}{\partial s_i^2}} = \frac{-\sum_{x_i \in R_{km}} \sum_{\{i,j\} \rightleftharpoons I} |\Delta Z_{ij}| \rho_{ij}}{\sum_{x_i \in R_{km}} \sum_{\{i,j\} \rightleftharpoons I} |\Delta Z_{ij}| \sigma \rho_{ij} (1 - \rho_{ij})}$$

（注意符号的变化，因为这里我们是在最大化）。在实现中，方便的做法是为每个样本$x_i$计算$\rho_{ij}$；然后计算任何特定叶节点的$\gamma_{km}$只需要进行求和与除法。注意，就像逻辑回归一样，对于给定的学习率$\eta$，$\sigma$的选择对训练没有影响，因为$\gamma_{km}$与$1/\sigma$成比例，模型得分总是增加$\eta \gamma_{km}$，且得分总是乘以$\sigma$出现。

下面我们示意性地总结LambdaMART算法。如[11]所指出的，通过从初始基模型给出的得分开始，可以轻松进行模型自适应。

**算法：LambdaMART**

设定树的数量$N$，训练样本数量$m$，每棵树的叶子数量$L$，学习率$\eta$

```
for i = 0 to m do
    F_0(x_i) = BaseModel(x_i)
    // 如果BaseModel为空，设F_0(x_i) = 0
end for
for k = 1 to N do
    for i = 0 to m do
        y_i = λ_i
        w_i = ∂y_i / ∂F_{k-1}(x_i)
    end for
    {R_lk}_{l=1}^L       // 在 {x_i, y_i}_{i=1}^m 上创建L叶树
    γ_lk = (∑_{x_i∈R_lk} y_i) / (∑_{x_i∈R_lk} w_i)    // 基于牛顿步长分配叶子值
    F_k(x_i) = F_{k-1}(x_i) + η ∑_l γ_lk I(x_i ∈ R_lk)  // 以学习率η进行步进
end for
```

最后，比较LambdaRank和LambdaMART如何更新参数是有用的。LambdaRank在每次查询后更新所有权重。而LambdaMART中的决策（节点分裂）是使用落入该节点的所有数据计算的，因此LambdaMART每次只更新少数参数（即当前叶节点的分裂值），但使用所有数据（因为每个$x_i$都落入某个叶子）。这特别意味着LambdaMART能够选择可能降低某些查询效用但总体效用增加的分裂和叶子值。

[^4]: 这里我们切换回效用，但使用$C$以匹配之前的符号。

### 7.1 最后说明：如何最优地组合排序器

最后给出以下说明：[11]包含一个相对简单的方法来线性组合两个排序器，使得在所有可能的线性组合中，得到的排序器获得尽可能高的NDCG（实际上这对于任何IR指标都可行）。关键技巧在于注意到如果$s_i^1$是第一个排序器的输出（对于给定查询），$s_i^2$是第二个的输出，则组合排序器的输出可以写为$\alpha s_i^1 + (1 - \alpha) s_i^2$，且当$\alpha$在其值域内变化时（比如从0到1），NDCG只会改变有限次。我们可以简单（且相当高效地）跟踪每次变化，并保留给出最佳组合NDCG的$\alpha$。然而也应注意，通过简单地将先前模型的得分作为输入来训练一个线性（或非线性）LambdaRank模型，也可以达到类似的目标（但无法保证最优性）。这通常给出同样好的结果，并且更容易扩展到多个模型。

## 致谢

这里描述的工作受到了许多人的影响。除了参考文献中列出的合著者之外，我还要特别感谢Qiang Wu、Krysta Svore、Ofer Dekel、Pinar Donmez、Yisong Yue和Galen Andrew。

## 参考文献

[1] E.B. Baum and F. Wilczek. Supervised Learning of Probability Distributions by Neural Networks. *Neural Information Processing Systems*, American Institute of Physics, 1988.

[2] C.J.C. Burges, T. Shaked, E. Renshaw, A. Lazier, M. Deeds, N. Hamilton and G. Hullender. Learning to Rank using Gradient Descent. *Proceedings of the Twenty Second International Conference on Machine Learning*, 2005.

[3] C.J.C. Burges. Ranking as Learning Structured Outputs. *Neural Information Processing Systems workshop on Learning to Rank*, Eds. S. Agerwal, C. Cortes and R. Herbrich, 2005.

[4] C.J.C. Burges, R. Ragno and Q.V. Le. Learning to Rank with Non-Smooth Cost Functions. *Advances in Neural Information Processing Systems*, 2006.

[5] O. Chapelle, Y. Chang and T-Y Liu. The Yahoo! Learning to Rank Challenge. http://learningtorankchallenge.yahoo.com, 2010.

[6] O. Chapelle, D. Metzler, Y. Zhang and P. Grinspan. Expected Reciprocal Rank for Graded Relevance Measures. *International Conference on Information and Knowledge Management (CIKM)*, 2009.

[7] P. Donmez, K. Svore and C.J.C. Burges. On the Local Optimality of LambdaRank. *Special Interest Group on Information Retrieval (SIGIR)*, 2009.

[8] J.H. Friedman. Greedy function approximation: A gradient boosting machine. Technical Report, IMS Reitz Lecture, Stanford, 1999; see also *Annals of Statistics*, 2001.

[9] K. Jarvelin and J. Kekalainen. IR evaluation methods for retrieving highly relevant documents. *Special Interest Group on Information Retrieval (SIGIR)*, 2000.

[10] M. Spivak. *Calculus on Manifolds*. Addison-Wesley, 1965.

[11] Q. Wu, C.J.C. Burges, K. Svore and J. Gao. Adapting Boosting for Information Retrieval Measures. *Journal of Information Retrieval*, 2007.

[12] Y. Yue and C.J.C. Burges. On Using Simultaneous Perturbation Stochastic Approximation for Learning to Rank, and the Empirical Optimality of LambdaRank. Microsoft Research Technical Report MSR-TR-2007-115, 2007.
