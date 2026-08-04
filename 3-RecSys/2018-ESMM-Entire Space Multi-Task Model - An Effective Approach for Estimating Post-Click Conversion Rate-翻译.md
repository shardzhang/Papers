# Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate

> Xiao Ma, Liqin Zhao, Guan Huang, Zhi Wang, Zelin Hu, Xiaoqiang Zhu, Kun Gai | Alibaba Group
>
> {maxiao.mx, liqin.zlq, bingyi.wz, xiaoqiang.zxq, jingshi.gk}@alibaba-inc.com



本文介绍了全空间多任务模型（ESMM）——一种通过充分利用用户行为的序列模式（曝光$\rightarrow$点击$\rightarrow$转化）来估计点击后转化率（CVR）的方法。它同时在全空间上建模CVR并采用特征表示迁移学习策略，从而消除样本选择偏差和数据稀疏两个问题。核心内容：

- 提出全空间多任务模型（ESMM），利用用户行为的序列模式建模CVR
- 引入CTR和CTCVR两个辅助任务，在全空间上直接建模CVR，消除样本选择偏差（SSB）
- CVR网络与CTR网络共享嵌入参数，通过特征表示迁移学习缓解数据稀疏（DS）问题

关键发现：

- ESMM 在从淘宝推荐系统流量日志收集的数据集上显著优于竞争方法
- 消除了样本选择偏差和数据稀疏问题，为CVR建模提供了优雅的解决方案
- 发布了第一个包含点击和转化标签序列依赖样本的公开数据集，用于CVR建模

---



## 摘要

准确估计点击后转化率（CVR）对于工业应用（如推荐和广告）中的排序系统至关重要。传统的CVR建模应用流行的深度学习方法并取得了最先进的性能。然而，它在实践中遇到了几个任务特定的问题，使得CVR建模具有挑战性。例如，传统的CVR模型使用点击曝光样本进行训练，但在所有曝光样本的全空间上进行推断。这导致了样本选择偏差问题。此外，还存在极端的数据稀疏问题，使得模型拟合相当困难。在本文中，我们通过充分利用用户行为的序列模式（即曝光$\rightarrow$点击$\rightarrow$转化）从一个全新的角度对CVR进行建模。提出的全空间多任务模型（ESMM）可以通过以下方式同时消除这两个问题：i）直接在全空间上建模CVR，ii）采用特征表示迁移学习策略。在从淘宝推荐系统流量日志收集的数据集上的实验表明，ESMM显著优于竞争方法。我们还发布了一个该数据集的采样版本以促进未来研究。据我们所知，这是第一个包含具有点击和转化标签序列依赖的样本的公开数据集，用于CVR建模。

**关键词** post-click conversion rate; multi-task learning; sample selection bias; data sparsity; entire-space modeling



## 1. 引言

转化率（CVR）预测是工业应用中排序系统（如在线广告和推荐等）的基本任务。例如，预测的CVR在OCPC（优化每点击成本）广告中用于调整每次点击的出价，以实现平台和广告主的双赢[4]。它也是推荐系统中平衡用户点击偏好和购买偏好的重要因素。

在本文中，我们专注于点击后CVR估计的任务。为简化讨论，我们以电子商务网站推荐系统中的CVR建模为例。给定推荐的商品，用户可能会点击感兴趣的商品，并进一步购买其中一些。换句话说，用户行为遵循曝光$\rightarrow$点击$\rightarrow$转化的序列模式。这样，CVR建模指的是估计点击后转化率的任务，即 $\mathrm{pCVR} = p(\text{转化}|\text{点击}, \text{曝光})$。

通常，传统的CVR建模方法采用与点击率（CTR）预测任务中开发的类似技术，例如最近流行的深度网络[2, 3]。然而，存在几个任务特定的问题，使得CVR建模具有挑战性。其中，我们报告了在实际实践中遇到的两个关键问题：i）样本选择偏差（SSB）问题[12]。如图1所示，传统的CVR模型在由点击曝光组成的数据集上训练，但在所有曝光样本的全空间上进行推断。SSB问题会损害训练模型的泛化性能。ii）数据稀疏（DS）问题。在实践中，为训练CVR模型收集的数据通常远少于CTR任务。训练数据的稀疏性使得CVR模型拟合相当困难。

已有几项研究试图解决这些挑战。在[5]中，构建了不同特征上的分层估计器，并与逻辑回归模型结合以解决DS问题。然而，它依赖于先验知识来构建分层结构，这难以应用于拥有数千万用户和商品的推荐系统中。过采样方法[11]复制稀有类样本，有助于缓解数据稀疏性，但对采样率敏感。将所有缺失视为负样本（AMAN）采用随机采样策略选择未点击的曝光作为负样本[6]。它通过引入未观测样本可以在一定程度上消除SSB问题，但会导致持续低估的预测。无偏方法[10]通过拒绝采样从观测中拟合真实潜在分布来解决CTR建模中的SSB问题。然而，它在通过拒绝概率的除法对样本加权时可能会遇到数值不稳定性问题。总之，在CVR建模场景中，SSB和DS问题均未被很好地解决，并且上述方法均未利用序列动作的信息。

在本文中，通过充分利用用户行为的序列模式，我们提出了一种名为全空间多任务模型（ESMM）的新方法，该方法能够同时消除SSB和DS问题。在ESMM中，引入了两个辅助任务：预测曝光后点击率（CTR）和曝光后点击&转化率（CTCVR）。ESMM不是直接使用点击曝光样本训练CVR模型，而是将 $\mathrm{pCVR}$ 视为一个中间变量，其乘以 $\mathrm{pCTR}$ 等于 $\mathrm{pCTCVR}$。$\mathrm{pCTCVR}$ 和 $\mathrm{pCTR}$ 都是在全空间上使用所有曝光样本估计的，因此推导出的 $\mathrm{pCVR}$ 也适用于全空间。这表明SSB问题被消除了。此外，CVR网络的特征表示参数与CTR网络共享。后者使用丰富得多的样本进行训练。这种参数迁移学习[7]有助于显著缓解DS问题。

在这项工作中，我们从淘宝推荐系统收集流量日志。完整数据集包含89亿个样本，具有点击和转化的序列标签。进行了仔细的实验。ESMM始终优于竞争模型，证明了所提出方法的有效性。我们还发布了我们的数据集¹用于该领域的未来研究。

图1：传统CVR建模中样本选择偏差问题的示意图。训练空间由点击曝光样本组成。它只是推断空间的一部分，推断空间由所有曝光组成。



## 2. 提出的方法

### 2.1 符号说明

我们假设观测数据集为 $S = \{(x_i, y_i \rightarrow z_i)\}_{i=1}^{N}$，样本 $(x, y \rightarrow z)$ 从分布 $D$ 中抽取，其域为 $X \times Y \times Z$，其中 $X$ 是特征空间，$Y$ 和 $Z$ 是标签空间，$N$ 是总曝光数。$x$ 表示观测曝光的特征向量，通常是具有多字段[8]的高维稀疏向量，例如用户字段、商品字段等。$y$ 和 $z$ 是二元标签，$y = 1$ 或 $z = 1$ 分别表示是否发生点击或转化事件。$y \rightarrow z$ 揭示了点击和转化标签的序列依赖性，即当转化事件发生时总是存在先前的点击。

点击后CVR建模是估计概率 $\mathrm{pCVR} = p(z = 1|y = 1, x)$。两个相关的概率是：曝光后点击率（CTR），$pCTR = p(y = 1|x)$；以及曝光后点击&转化率（CTCVR），$pCTCVR = p(y = 1, z = 1|x)$。给定曝光 $x$，这些概率满足公式(1)：

$$
p(y = 1, z = 1|x) = p(y = 1|x) \times p(z = 1|y = 1, x) \qquad (1)
$$

### 2.2 CVR建模与挑战

最近，基于深度学习的方法已被提出用于CVR建模，取得了最先进的性能。它们大多遵循类似的Embedding&MLP网络架构，如[3]中介绍的。图2的左侧部分说明了这种架构，为简单起见，我们将其称为BASE模型。

简而言之，传统的CVR建模方法直接估计点击后转化率 $p(z = 1|y = 1, x)$。它们使用点击曝光样本训练模型，即 $S_c = \{(x_j, z_j)|y_j = 1\}_{j=1}^{M}$。$M$ 是所有曝光中的点击次数。显然，$S_c$ 是 $S$ 的子集。注意在 $S_c$ 中，（点击的）没有转化的曝光被视为负样本，有转化（也被点击）的曝光被视为正样本。在实践中，CVR建模遇到几个任务特定的问题，使其具有挑战性。

样本选择偏差（SSB）[12]。实际上，传统的CVR建模通过引入辅助特征空间 $X_c$ 来近似 $p(z = 1|y = 1, x) \approx q(z = 1|x_c)$。$X_c$ 表示与 $S_c$ 相关的有限空间²。$\forall x_c \in X_c$，存在一对 $(x = x_c, y_x = 1)$，其中 $x \in X$，$y_x$ 是 $x$ 的点击标签。这样，$q(z = 1|x_c)$ 在 $X_c$ 空间上使用 $S_c$ 的点击样本进行训练。在推断阶段，在全空间 $X$ 上对 $p(z = 1|y = 1, x)$ 的预测被计算为 $q(z = 1|x)$，假设对于任何满足 $x \in X$ 的 $(x, y_x = 1)$ 对，$x$ 都属于 $X_c$。由于 $X_c$ 只是全空间 $X$ 的一小部分，这个假设很可能会被违反。它受到很少发生的点击事件的随机性的严重影响，其概率在空间 $X$ 的不同区域变化。此外，实践中没有足够的观测，空间 $X_c$ 可能与 $X$ 有相当大差异。这将导致训练样本的分布偏离真实潜在分布，并损害CVR建模的泛化性能。

数据稀疏（DS）。传统方法使用 $S_c$ 的点击样本训练CVR模型。点击事件的罕见发生导致CVR建模的训练数据极其稀疏。直观上，它通常比相关的CTR任务少1-3个数量级，后者是在包含所有曝光的数据集 $S$ 上训练的。表1显示了我们的实验数据集的统计数据，其中CVR任务的样本数仅为CTR任务的4%。

空间 $X_c$ 在条件 $\forall x \in X, p(y = 1|x) > 0$ 且观测曝光数量足够大的情况下等于 $X$。否则，空间 $X_c$ 是 $X$ 的一部分。

值得提及的是，CVR建模还存在其他挑战，例如延迟反馈[1]。本文不关注这一点。一个原因是我们的系统中转化延迟的程度是可以接受的。另一个原因是我们的方法可以与先前的工作[1]结合来处理它。

### 2.3 全空间多任务模型

提出的ESMM如图2所示，它充分利用了用户行为的序列模式。借鉴多任务学习的思想[9]，ESMM引入了CTR和CTCVR两个辅助任务，并同时消除了CVR建模的上述问题。

总的来说，ESMM针对给定的曝光同时输出 $\mathrm{pCTR}$、$\mathrm{pCVR}$ 以及 $\mathrm{pCTCVR}$。它主要由两个子网络组成：CVR网络（图2左侧部分）和CTR网络（右侧部分）。CVR和CTR网络都采用与BASE模型相同的结构。CTCVR将CVR和CTR网络的输出相乘作为输出。ESMM中有一些亮点，这些亮点对CVR建模有显著影响，并使ESMM区别于传统方法。

图2：用于CVR建模的ESMM架构概览。在ESMM中，引入了CTR和CTCVR两个辅助任务：i）帮助在整个输入空间上建模CVR，ii）提供特征表示迁移学习。ESMM主要由两个子网络组成：图左侧部分展示的CVR网络和右侧部分的CTR网络。CTR和CVR网络的嵌入参数是共享的。CTCVR将CTR和CVR网络的输出相乘作为输出。

全空间建模。公式(1)给出了提示，可以转化为公式(2)。

$$
p(z = 1|y = 1, x) = \frac{p(y = 1, z = 1|x)}{p(y = 1|x)} \qquad (2)
$$

这里 $p(y = 1, z = 1|x)$ 和 $p(y = 1|x)$ 是在包含所有曝光的数据集 $S$ 上建模的。公式(2)告诉我们，通过估计 $\mathrm{pCTCVR}$ 和 $\mathrm{pCTR}$，可以在整个输入空间 $X$ 上推导出 $\mathrm{pCVR}$，这直接解决了样本选择偏差问题。这似乎很简单，通过分别训练独立的CTR和CTCVR模型来估计 $\mathrm{pCTR}$ 和 $\mathrm{pCTCVR}$，并通过公式(2)获得 $\mathrm{pCVR}$，我们将其简称为DIVISION。然而，$\mathrm{pCTR}$ 在实践中是一个小数字，除以其会产生数值不稳定性。ESMM通过乘法形式避免了这一点。在ESMM中，$\mathrm{pCVR}$ 只是一个中间变量，受公式(1)的等式约束。$\mathrm{pCTR}$ 和 $\mathrm{pCTCVR}$ 是ESMM在全空间上实际估计的主要因子。乘法形式使得三个相关联且共同训练的估计器能够利用数据的序列模式，并在训练期间相互传递信息。此外，它确保估计的 $\mathrm{pCVR}$ 值在 $[0,1]$ 范围内，而在DIVISION方法中可能超过1。

ESMM的损失函数定义为公式(3)。它包含来自CTR和CTCVR任务的两个损失项，这些项在所有曝光样本上计算，而不使用CVR任务的损失。

$$
L(\theta_{cvr}, \theta_{ctr}) = \sum_{i=1}^{N} l(y_i, f(x_i; \theta_{ctr})) + \sum_{i=1}^{N} l(y_i \& z_i, f(x_i; \theta_{ctr}) \times f(x_i; \theta_{cvr})) \qquad (3)
$$

其中 $\theta_{ctr}$ 和 $\theta_{cvr}$ 是CTR和CVR网络的参数，$l(\cdot)$ 是交叉熵损失函数。数学上，公式(3)将 $y \rightarrow z$ 分解为两部分³：$y$ 和 $y \& z$，这实际上利用了点击和转化标签的序列依赖性。

³对应于CTR和CTCVR任务的标签，其构建训练数据集如下：i）样本由所有曝光组成，ii）对于CTR任务，点击曝光标记为y = 1，否则y = 0，iii）对于CTCVR任务，同时发生点击和转化事件的曝光标记为y&z = 1，否则y&z = 0。

特征表示迁移。如2.2节所述，嵌入层将大规模稀疏输入映射为低维表示向量。它贡献了深度网络的大部分参数，其学习需要大量训练样本。在ESMM中，CVR网络的嵌入字典与CTR网络的嵌入字典共享。这遵循了特征表示迁移学习范式。CTR任务的所有曝光训练样本相对比CVR任务丰富得多。这种参数共享机制使得ESMM中的CVR网络能够从未点击的曝光中学习，并为缓解数据稀疏问题提供了巨大帮助。

请注意，ESMM中的子网络可以替换为最近开发的一些模型[2, 3]，这可能会获得更好的性能。由于篇幅有限，我们省略了这一部分，并专注于解决CVR建模实践中遇到的挑战。



## 3. 实验

### 3.1 实验设置

数据集。在我们的调研中，在CVR建模领域没有找到具有点击和转化序列标签的公开数据集。为评估所提出的方法，我们从淘宝推荐系统收集流量日志，并发布了整个数据集的1%随机采样版本，其大小仍达到38GB（未压缩）。在本文的其余部分，我们将发布的数据集称为公共数据集，整个数据集称为产品数据集。表1总结了这两个数据集的统计数据。详细描述可以在公共数据集网站¹中找到。

¹https://tianchi.aliyun.com/datalab/dataSet.html?dataId=408

表1：实验数据集的统计数据。

数据集 | #用户 | #商品 | #曝光 | #点击 | #转化
公共数据集 | 0.4M | 4.3M | 84M | 3.4M | 18k
产品数据集 | 48M | 23.5M | 8950M | 324M | 1774k

竞争对手。我们使用几种竞争方法在CVR建模上进行实验。（1）BASE是2.2节介绍的基线模型。（2）AMAN [6]采用负采样策略，并在{10%, 20%, 50%, 100%}中搜索采样率，报告最佳结果。（3）OVERSAMPLING [11]复制正样本以降低稀疏数据训练的难度，采样率在{2, 3, 5, 10}中搜索。（4）UNBIAS遵循[10]通过拒绝采样从观测中拟合真实潜在分布。$\mathrm{pCTR}$ 作为拒绝概率。（5）DIVISION使用独立训练的CTR和CTCVR网络估计 $\mathrm{pCTR}$ 和 $\mathrm{pCTCVR}$，并通过公式(2)计算 $\mathrm{pCVR}$。（6）ESMM-NS是ESMM的轻量版，不共享嵌入参数。

前四种方法是基于最先进的深度网络直接建模CVR的不同变体。DIVISION、ESMM-NS和ESMM共享相同的思路，即在全空间上建模CVR，涉及CVR、CTR和CTCVR三个网络。ESMM-NS和ESMM协同训练三个网络，并从CVR网络取输出进行模型比较。为公平起见，所有竞争对手（包括ESMM）共享与BASE模型相同的网络结构和超参数，即：i）使用ReLU激活函数，ii）设置嵌入向量维度为18，iii）设置MLP网络各层维度为 $360 \times 200 \times 80 \times 2$，iv）使用adam求解器，参数 $\beta_1 = 0.9$，$\beta_2 = 0.999$，$\epsilon = 10^{-8}$。

评估指标。比较在两个不同任务上进行：（1）传统的CVR预测任务，在点击曝光数据集上估计 $\mathrm{pCVR}$；（2）CTCVR预测任务，在所有曝光数据集上估计 $\mathrm{pCTCVR}$。任务（2）旨在比较不同CVR建模方法在整个输入空间上的表现，这反映了模型对应SSB问题的性能。在CTCVR任务中，所有模型通过 $\mathrm{pCTR} \times \mathrm{pCVR}$ 计算 $\mathrm{pCTCVR}$，其中：i）$\mathrm{pCVR}$ 由各模型分别估计，ii）$\mathrm{pCTR}$ 使用相同的独立训练的CTR网络（与BASE模型相同的结构和超参数）估计。这两个任务都将时间序列中前1/2的数据划分为训练集，其余为测试集。采用ROC曲线下面积（AUC）作为性能指标。所有实验重复10次，报告平均结果。

### 3.2 公共数据集上的结果

表2：不同模型在公共数据集上的比较。

模型 | CVR任务上的AUC（均值 $\pm$ 标准差） | CTCVR任务上的AUC（均值 $\pm$ 标准差）
BASE | 66.00 $\pm$ 0.37 | 62.07 $\pm$ 0.45
AMAN | 65.21 $\pm$ 0.59 | 63.53 $\pm$ 0.57
OVERSAMPLING | 67.18 $\pm$ 0.32 | 63.05 $\pm$ 0.48
UNBIAS | 66.65 $\pm$ 0.28 | 63.56 $\pm$ 0.70
DIVISION | 67.56 $\pm$ 0.48 | 63.62 $\pm$ 0.09
ESMM-NS | 68.25 $\pm$ 0.44 | 64.44 $\pm$ 0.62
ESMM | 68.56 $\pm$ 0.37 | 65.32 $\pm$ 0.49

表2显示了不同模型在公共数据集上的结果。（1）在BASE模型的三个变体中，只有AMAN在CVR任务上表现稍差，这可能是由于随机采样的敏感性。OVERSAMPLING和UNBIAS在CVR和CTCVR任务上均比BASE模型有所改进。（2）DIVISION和ESMM-NS都在全空间上估计 $\mathrm{pCVR}$，并取得了比BASE模型显著的提升。由于避免了数值不稳定性，ESMM-NS表现优于DIVISION。（3）ESMM进一步改进了ESMM-NS。通过利用用户行为的序列模式并通过迁移机制从未点击数据中学习，ESMM为CVR建模提供了一个优雅的解决方案，同时消除了SSB和DS问题，并击败了所有竞争对手。与BASE模型相比，ESMM在CVR任务上获得了2.56%的绝对AUC提升，这表明即使对于有偏样本也具有很好的泛化性能。在包含全样本的CTCVR任务上，它带来了3.25%的AUC提升。这些结果验证了我们建模方法的有效性。

### 3.3 产品数据集上的结果

我们进一步在产品数据集上评估ESMM，该数据集包含89亿个样本，比公共数据集大两个数量级。为验证训练集规模的影响，我们在这个大规模数据集上针对不同采样率进行了仔细的比较，如图3所示。首先，所有方法都随着训练样本数量的增长而表现出改进。这表明了数据稀疏性的影响。在所有情况下，除了AMAN在1%采样CVR任务上，BASE模型都被击败了。其次，ESMM-NS和ESMM在不同的采样率下始终优于所有竞争对手。特别是，ESMM在CVR和CTCVR任务上对所有竞争对手都保持了较大的AUC提升幅度。BASE模型是在我们实际系统中承载主要流量的最新版本。使用整个数据集训练，ESMM在CVR任务上取得了比BASE模型2.18%的绝对AUC提升，在CTCVR任务上取得了2.32%的提升。这对于工业应用来说是一个显著的改进，因为在工业应用中0.1%的AUC提升就已经很显著了。

图3：产品数据集上不同模型在不同采样率下的比较。



## 4. 结论与未来工作

在本文中，我们提出了一种用于CVR建模任务的新方法ESMM。ESMM充分利用了用户行为的序列模式。借助CTR和CTCVR两个辅助任务，ESMM优雅地解决了CVR建模在实际实践中遇到的样本选择偏差和数据稀疏的挑战。在真实数据集上的实验证明了所提出的ESMM的优越性能。该方法可以轻松推广到具有序列依赖性的场景中的用户动作预测。未来，我们打算在具有多阶段动作（如请求$\rightarrow$曝光$\rightarrow$点击$\rightarrow$转化）的应用中设计全局优化模型。

## 参考文献

[1] Olivier Chapelle. 2014. Modeling delayed feedback in display advertising. In Proceedings of the 20th ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 1097–1105.

[2] Heng-Tze Cheng and Levent Koc. 2016. Wide & deep learning for recommender systems. In Proceedings of the 1st Workshop on Deep Learning for Recommender Systems. ACM, 7–10.

[3] Zhou G., Song C., et al. 2017. Deep Interest Network for Click-Through Rate Prediction. arXiv preprint arXiv:1706.06978 (2017).

[4] Zhu H., Jin J., et al. 2017. Optimized cost per click in taobao display advertising. In Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. ACM, 2191–2200.

[5] Lee K., Orten B., et al. 2012. Estimating conversion rate in display advertising from past performance data. In Proceedings of the 18th ACM SIGKDD international conference on Knowledge discovery and data mining. ACM.

[6] Rong Pan, Yunhong Zhou, Bin Cao, Nathan N Liu, Rajan Lukose, Martin Scholz, and Qiang Yang. 2008. One-class collaborative filtering. In Data Mining, 2008. ICDM'08. Eighth IEEE International Conference on. IEEE, 502–511.

[7] Sinno Jialin Pan and Q. Yang. 2010. A Survey on Transfer Learning. In IEEE Transactions on Knowledge and Data Engineering. 1345–1359.

[8] Steffen Rendle. 2010. Factorization machines. In Data Mining (ICDM), 2010 IEEE 10th International Conference on. IEEE, 995–1000.

[9] Sebastian Ruder. 2017. An overview of multi-task learning in deep neural networks. arXiv preprint arXiv:1706.05098 (2017).

[10] Zhang W., Zhou T., et al. 2016. Bid-aware gradient descent for unbiased learning with censored data in display advertising. In Proceedings of the 22nd International Conference on Knowledge Discovery and Data Mining. ACM.

[11] Gary M Weiss. 2004. Mining with rarity: a unifying framework. ACM Sigkdd Explorations Newsletter 6, 1 (2004), 7–19.

[12] Bianca Zadrozny. 2004. Learning and evaluating classifiers under sample selection bias. In Proceedings of the 21th international conference on Machine learning. ACM.
