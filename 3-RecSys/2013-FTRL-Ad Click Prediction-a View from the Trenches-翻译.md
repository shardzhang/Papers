# 2013-广告点击预测：一线视角

> H. Brendan McMahan, Gary Holt, D. Sculley, Michael Young, Dietmar Ebner, Julian Grady, Lan Nie, Todd Phillips, Eugene Davydov, Daniel Golovin, Sharat Chikkerur, Dan Liu, Martin Wattenberg, Arnar Mar Hrafnkelsson, Tom Boulos, Jeremy Kubica | Google, Inc.


本文分享了来自已部署的**广告点击率（CTR）预测系统**的近期实验案例研究和主题。核心内容：

- 基于 FTRL-Proximal 在线学习算法的传统监督学习改进，以及每坐标学习率的使用
- 实际系统中的内存节省技巧、性能评估与可视化、**预测概率置信度估计**、**校准方法** 和 **自动化特征管理**
- 文献中有 promising 结果但在本系统中未见益处的多个方向的详细分析

关键发现：

- FTRL-Proximal 在相同或更好精度下显著优于 RDA 和 FOBOS 的稀疏性
- 每坐标学习率相 比 全局学习率 将 AucLoss 降低了 11.2%
- 布隆过滤器方法在 **RAM 节省** 与 **预测质量** 间提供了最佳权衡（节省 66% RAM，仅 0.008% 损害）
- q2.13 定点编码 替代 64 位浮点节省 75% 系数存储 RAM，且无精度损失
- 概率特征包含、dropout、特征 bagging 等方法在此系统中未产生显著益处

---


## 摘要

预测广告点击率（CTR）是一个大规模学习问题，对于价值数十亿美元的在线广告行业至关重要。我们提出了一系列案例研究和主题，这些内容来源于已部署的CTR预测系统中的近期实验。这些内容包括**基于FTRL-Proximal在线学习算法**（该算法具有出色的 **稀疏性** 和 **收敛性**）的传统监督学习方面的改进，以及 **每坐标学习率** 的使用。

我们还探讨了在实际系统中出现的一些挑战，这些挑战乍看之下可能属于传统机器学习研究领域之外。这些挑战包括内存节省的有用技巧、评估和可视化性能的方法、为预测概率提供置信度估计的实用方法、校准方法以及特征自动化管理的方法。最后，我们还详细介绍了几个方向，尽管在文献中其他地方有 promising 的结果，但对我们的系统并未带来益处。本文的目标是强调在这一工业环境中**理论进展与实际工程之间的密切关系**，并展示在复杂动态系统中应用传统机器学习方法时出现的挑战深度。


## 分类和主题描述符

I.5.4 [计算方法]：模式识别——应用


## 关键词

在线广告, 数据挖掘, 大规模学习


## 1. 引言

在线广告是一个价值数十亿美元的行业，已成为机器学习最成功的案例之一。赞助搜索广告、上下文广告、展示广告以及实时竞价拍卖都严重依赖学习模型准确、快速且可靠地预测广告点击率的能力[28, 15, 33, 1, 16]。这一问题的设定也推动该领域解决了即使在十年前也几乎难以想象的规模问题。一个典型的工业模型每天可能对数十亿事件提供预测，使用相应庞大的特征空间，然后从由此产生的大量数据中学习。

在本文中，我们提出了一系列案例研究，这些案例来源于Google用于预测 **赞助搜索广告** 点击率的已部署系统的近期实验。由于这一问题的设定已得到充分研究，我们选择聚焦于一系列较少受到关注但在实际运行系统中同等重要的主题。因此，我们以传统上用于设计有效学习算法同样严谨的态度，探讨了内存节省、性能分析、预测置信度、校准和特征管理等问题。本文的目标是让读者**了解实际工业环境中出现的挑战深度**，并分享可应用于其他大规模问题领域的技巧和见解。


## 2. 系统简要概述

当用户进行搜索 $q$ 时，一组初始候选广告会根据广告主选择的关键词 与查询 $q$ 进行匹配。然后，拍卖机制决定这些广告是否展示给用户、以什么顺序展示，以及如果广告被点击，广告主需要支付的价格。除了广告主出价外，拍卖的一个重要输入是每个广告 $a$ 的 $P(click | q, a)$ 估计值，即 **广告展示后被点击的概率**。

我们系统中使用的特征来自多种来源，包括**查询、广告创意的文本以及各种广告相关的元数据**。数据通常**极其稀疏**，**每个样本中非零特征值通常只占极小比例**。

诸如 **regularized logistic regression（正则化逻辑回归）** 等方法天然适合这一问题设定。每天需要进行数十亿次预测，并随着新观察到点击和非点击数据快速更新模型。当然，这种数据速率意味着训练数据集是庞大的。数据由基于Photon系统的流服务提供——详细讨论请参见[2]。

由于近年来大规模学习已得到深入研究（例如[3]），我们不会在本文中花费大量篇幅详细描述我们的系统架构。然而，我们将指出，训练方法与Google Brain团队描述的Downpour SGD方法有相似之处[8]，不同之处在于我们**训练的是单层模型而非多层深度网络**。这使我们能够处理比已有文献报道的显著更大的数据集和模型，拥有**数十亿的系数**。由于训练好的模型会复制到多个数据中心进行服务（见图1），我们**更关注服务时的稀疏化而非训练期间的稀疏化**。

<img src="/Users/dazhang/PycharmProject/Papers/3-RecSys/.picture/image-20260719153024194.png" alt="image-20260719153024194" style="zoom:33%;" />

图1：高层系统概览。稀疏化在第3节中介绍，**概率特征包含**在第4节中介绍，渐进验证在第5节中介绍，校准方法在第7节中介绍。


## 3. 在线学习与稀疏性

对于大规模学习，广义线性模型（如逻辑回归）的在线算法具有许多优势。尽管特征向量 $\mathbf{x}$ 可能有数十亿维度，但通常每个实例只有数百个非零值。这使得通过从磁盘或网络流式传输样本即可在大规模数据集上进行高效训练成为可能 [3]，**因为每个训练样本只需要被处理一次。**

为精确描述算法，我们需要建立一些符号。我们用粗体表示向量如 $\mathbf{g}_t \in \mathbb{R}^d$ ，其中 $t$ 索引当前训练样本；向量 $\mathbf{g}_t$ 的第 $i$ 个条目记为 $g_{t,i}$ 。我们还使用压缩求和符号 $\mathbf{g}_{1:t} = \sum_{s=1}^t \mathbf{g}_s$ 。

如果我们希望使用**逻辑回归**对问题进行建模，可以使用以下**在线框架**。在第 $t$ 轮，我们被要求对由特征向量 $\mathbf{x}_t \in \mathbb{R}^d$ 描述的实例进行预测，给定模型参数 $\mathbf{w}_t$ ，我们预测 $p_t = \sigma(\mathbf{w}_t \cdot \mathbf{x}_t)$ ，其中 $\sigma(a) = 1/(1 + \exp(-a))$ 是 sigmoid 函数。然后，我们观察到标签 $y_t \in \{0, 1\}$ ，并承受由此产生的 LogLoss（逻辑损失）：

$$
\ell_t(\mathbf{w}_t) = -y_t \log p_t - (1 - y_t) \log(1 - p_t), \qquad (1)
$$

即给定 $p$ 的 $y$ 的**负对数似然**。很容易证明 $\nabla \ell_t(\mathbf{w}_t) = (\sigma(\mathbf{w}_t \cdot \mathbf{x}_t) - y_t)\mathbf{x}_t = (p_t - y_t)\mathbf{x}_t$ ，而这个梯度就是我们在优化中需要的一切。

**在线梯度下降（Online Gradient Descend, OGD）已被证明对这类问题非常有效，以最小的计算资源产生出色的预测精度。**然而，在实践中另一个关键考虑因素是**最终模型的大小**；**由于模型可以稀疏存储， $\mathbf{w}$ 中非零系数的数量是内存使用的决定因素**。¹

> ¹ OGD 本质上与随机梯度下降相同；名称"在线"强调我们不是在解决批处理问题，而是对可能不是 IID 的样本序列进行预测。

不幸的是，**OGD在生成稀疏模型方面并不特别有效**。事实上，**简单地将L1惩罚的子梯度**加到损失的梯度（ $\nabla_{\mathbf{w}} \ell_t(\mathbf{w})$ ）上，**基本上永远不会产生精确为零的系数**。更复杂的方法，如**FOBOS和截断梯度**，确实成功地引入了稀疏性[11, 20]。正则化对偶平均（Regularized Dual Averaging，RDA）算法在精度与稀疏性的权衡方面比FOBOS取得了更好的效果[32]。然而，我们观察到梯度下降类方法在我们的数据集上可以 比 正则化对偶平均RDA 产生更好的精度[24]。那么，问题是我们能否**同时获得RDA提供的稀疏性 和 在线梯度下降OGD改进的精度**？答案是肯定的，使用"Follow The (Proximally) Regularized Leader"算法，即FTRL-Proximal。在没有正则化的情况下，该算法 与 标准在线梯度下降相同，但由于它使用了模型系数 $\mathbf{w}$ 的替代惰性表示，L1正则化可以更有效地实现。


<img src="/Users/dazhang/PycharmProject/Papers/3-RecSys/.picture/image-20260719153054628.png" alt="image-20260719153054628" style="zoom:33%;" />

>  算法1：带 L1 和 L2 正则化用于逻辑回归的每坐标 FTRL-Proximal


FTRL-Proximal 算法之前已在便于理论分析的方式下被提出 [24]。在这里，我们专注于**描述一个实际的实现**。给定梯度序列 $\mathbf{g}_t \in \mathbb{R}^d$ ，在线梯度下降OGD 执行更新

$$
\mathbf{w}_{t+1} = \mathbf{w}_t - \eta_t \mathbf{g}_t,
$$

其中 $\eta_t$ 是一个**非递增的学习率调度**，例如 $\eta_t = 1/\sqrt{t}$ 。而 FTRL-Proximal 算法使用更新

$$
\mathbf{w}_{t+1} = \arg\min_{\mathbf{w}} \left( \mathbf{g}_{1:t} \cdot \mathbf{w} + \frac{1}{2} \sum_{s=1}^t \sigma_s \|\mathbf{w} - \mathbf{w}_s\|^2 + \lambda_1 \|\mathbf{w}\|_1 \right),
$$

其中我们**根据学习率调度定义 $\sigma_s$ ，使得 $\sigma_{1:t} = 1/\eta_t$ **。表面上，这些更新看起来非常不同，但事实上当取 $\lambda_1 = 0$ 时，它们会产生完全相同的系数向量序列。然而，使用 $\lambda_1 > 0$ 的 **FTRL-Proximal 更新**在引入稀疏性方面表现出色（见下面的实验结果）。

粗略一看，人们可能认为 FTRL-Proximal 更新比梯度下降更难实现，或者**需要存储过去所有的系数**。然而实际上，每个系数只需要存储一个数字，因为我们可以将更新重写为 $\mathbf{w} \in \mathbb{R}^d$ 上的 argmin：

$$
\left( \mathbf{g}_{1:t} - \sum_{s=1}^t \sigma_s \mathbf{w}_s \right) \cdot \mathbf{w} + \frac{1}{\eta_t} \|\mathbf{w}\|_2^2 + \lambda_1 \|\mathbf{w}\|_1 + \text{(const)}.
$$

因此，如果我们存储了 $\mathbf{z}_{t-1} = \mathbf{g}_{1:t-1} - \sum_{s=1}^{t-1} \sigma_s \mathbf{w}_s$ ，则在第 $t$ 轮开始时，我们通过令 $\mathbf{z}_t = \mathbf{z}_{t-1} + \mathbf{g}_t + (1/\eta_t - 1/\eta_{t-1}) \mathbf{w}_t$ 来更新，并在每坐标基础上通过**闭式**求解 $\mathbf{w}_{t+1}$ ：

$$
w_{t+1,i} =
\begin{cases}
0 & \text{if } |z_{t,i}| \leq \lambda_1 \\
-\eta_t (z_{t,i} - \operatorname{sgn}(z_{t,i}) \lambda_1) & \text{otherwise.}
\end{cases}
$$

因此，FTRL-Proximal 在内存中存储 $\mathbf{z} \in \mathbb{R}^d$ ，而 OGD 存储 $\mathbf{w} \in \mathbb{R}^d$ 。算法1 采用了这种方法，但还添加了每坐标学习率调度（接下来讨论），并支持强度为 $\lambda_2$ 的 L2 正则化。或者，我们可以存储 $-\eta_t \mathbf{z}_t$ 而不是直接存储 $\mathbf{z}_t$ ；那么，当 $\lambda_1 = 0$ 时，我们存储的就是正常的梯度下降系数。注意当 $\eta_t$ 是常数 $\eta$ 且 $\lambda_1 = 0$ 时，**很容易看出与在线梯度下降的等价性**，因为 $\mathbf{w}_{t+1} = -\eta \mathbf{z}_t = -\eta \sum_{s=1}^t \mathbf{g}_s$ ，这正是梯度下降所达到的点。

> [!NOTE]
>
> 当 $\eta_t$ 是常数 $\eta$ 且 $\lambda_1 = 0$ 时，**很容易看出与在线梯度下降的等价性**。证明详见笔记

**实验结果。** 在早期对较小原型版本数据的实验中，McMahan [24]表明，使用L1正则化的FTRL-Proximal在产生的 **规模与精度** 权衡方面显著优于在线梯度下降RDA 和 FOBOS；这些先前的结果总结在表1的第2和第3行。

<img src="/Users/dazhang/PycharmProject/Papers/3-RecSys/.picture/image-20260719153130922.png" alt="image-20260719153130922" style="zoom: 33%;" />

> 表1：FTRL结果，显示了竞争方法的 **非零系数值的相对数量** 和 AucLoss（1-AUC）（两者都是越小越好）。总体而言，FTRL在相同或更好精度下提供了更好的稀疏性（0.6%的损害对我们的应用来说是显著的）。RDA和FOBOS在数百万样本的原型数据集上与FTRL进行比较，而OGD-Count在**全规模数据集**上与FTRL进行比较。


> [!NOTE]
>
> OGD-Count是什么？
>
> FTRL-Proximal 的关键思想：Follow The (Proximally) Regularized Leader
>
> 它用一种**惰性表示**来存储模型，使得L1正则化能更有效地产生精确的零系数。公式上：
>
> - OGD: $\mathbf{w}_{t+1} = \mathbf{w}_t - \eta_t \mathbf{g}_t$
> - FTRL: $\mathbf{w}_{t+1} = \arg\min_{\mathbf{w}} (\mathbf{g}_{1:t} \cdot \mathbf{w} + \frac{1}{2}\sum \sigma_s \|\mathbf{w} - \mathbf{w}_s\|^2 + \lambda_1 \|\mathbf{w}\|_1)$
>
> 关键洞察：当 $\lambda_1 = 0$ 时，FTRL 和 OGD 产生完全相同的序列。但当 $\lambda_1 > 0$ 时，FTRL 通过闭式解可以产生精确的零系数：
>
>
$$
> w_{t+1,i} =
> \begin{cases}
> 0 & \text{if } |z_{t,i}| \leq \lambda_1 \\
> -\eta_t (z_{t,i} - \operatorname{sgn}(z_{t,i})\lambda_1) & \text{otherwise}
> \end{cases}
>
$$
>
> 存储上只比OGD多存一个 $z$ 向量，代价极小

在许多情况下，简单的启发式方法与更原则化的方法效果几乎一样好，但这并非其中之一。我们的**草稿算法OGD-Count简单地维护它看到某个特征的计数**；在计数超过阈值 $k$ 之前，系数固定为零，但在计数超过 $k$ 之后，在线梯度下降（没有任何L1正则化）照常进行。为了在这个更简单的启发式方法上测试FTRL-Proximal，我们在一个非常大的数据集上运行。我们调整k以产生与FTRL-Proximal相等的精度；使用更大的k会导致更差的AucLoss。结果如表1第4行所示。

总体而言，这些结果表明FTRL-Proximal在相同或更好的预测精度下显著提高了稀疏性。


> [!NOTE]
>
> 这句话在说：**FTRL-Proximal 相比简单的启发式方法有本质优势，不是那种"差不多就行"的情况。**
>
> OGD-Count 算法
>
> - 对每个特征维护一个计数器 $n_i$ （该特征出现的次数）
>- 阈值 $k$ 之前：系数固定为 0（不参与学习）
> - 超过 $k$ 次之后：用普通 OGD 正常更新（没有 L1 正则化）
>
> 实验设计
>
> 1. 在超大数据集上对比 FTRL-Proximal 和 OGD-Count
>2. **调整 $k$ 使两者精度相等**（ $k$ 越大，越晚激活特征，稀疏性越高但精度越差）
> 3. 结果发现：要想达到和 FTRL 相同的精度，OGD-Count 的 $k$ 必须设得很小 $\to$ 导致模型几乎没有稀疏性（大量非零系数）
>
> 核心结论
>
> - OGD-Count 无法兼顾精度和稀疏性——要么精度差，要么模型大
>- FTRL-Proximal 能**同时**达到高精度+高稀疏性
> - 作者说这话是在反驳一种常见的偏见："简单的启发式方法通常和复杂方法差不多"——但在这里不是，FTRL 确实更好


### 3.1 每坐标学习率

**在线梯度下降的标准理论建议使用全局学习率调度** $\eta_t = 1/\sqrt{t}$ ，这**对所有坐标都是通用的** [34]。一个简单的思想实验表明这可能不是理想的：假设我们正在使用逻辑回归估计 10 枚硬币的 $Pr(\text{heads} | \text{coin}_i$ )。每轮 $t$ ，抛掷一枚硬币 $i$ ，我们看到一个特征向量 $\mathbf{x} \in \mathbb{R}^{10}$ ，其中 $x_i = 1$ 且对于 $j \neq i$ 有 $x_j = 0$ 。**因此，我们本质上是在解决 10 个独立的逻辑回归问题，打包成一个单一问题。**

我们可以运行 10 个独立的在线梯度下降副本，其中问题 $i$ 的算法实例会使用类似 $\eta_{t,i} = 1/\sqrt{n_{t,i}}$ 的学习率，其中 $n_{t,i}$ 是到目前为止硬币 $i$ 被抛掷的次数。如果硬币 $i$ 比硬币 $j$ 被抛掷得频繁得多，那么硬币 $i$ 的学习率会下降得更快，反映了我们有更多数据这一事实；而硬币 $j$ 的学习率将保持较高，因为我们**当前估计的置信度较低，因此需要更快地对新数据做出反应**。

另一方面，如果我们将此看作一个单一学习问题，标准的学习率调度 $\eta_t = 1/\sqrt{t}$  被应用于所有坐标：**也就是说，即使硬币 $i$ 没有被抛掷，我们也在降低它的学习率**。这显然不是最优行为。事实上，Streeter和McMahan [29]已经展示了一类问题，**其中标准算法的性能渐近地比运行独立副本差得多**。²因此，至少对于某些问题，每坐标学习率可以带来显著优势。

回顾 $g_{s,i}$ 是梯度 $\mathbf{g}_s = \nabla \ell_s(\mathbf{w}_s)$ 的第 $i$ 个坐标。那么，仔细分析表明**每坐标学习率**

$$
\eta_{t,i} = \frac{\alpha}{\beta + \sqrt{\sum_{s=1}^t g_{s,i}^2}} \qquad (2)
$$

在某种意义上是接近最优的。³在实践中，我们使用的学习率的 $\alpha$ 和 $\beta$ 是通过渐进验证（见第 5.1 节）选择的，以产生良好的性能。我们还尝试了对计数器 $n_{t,i}$ 使用 $0.5$ 以外的幂。 $\alpha$ 的最优值可能因特征和数据集而有较大差异，而 $\beta = 1$ 通常就足够好；这简单地确保了早期的学习率不会太高。

如前所述，该**算法要求我们跟踪每个特征的梯度之和以及梯度的平方和**。第4.5节提出了一种**替代的内存节省方案**，其中**梯度的平方和在多个模型之间分摊**。

关于每坐标学习率的一个相对简单的分析出现在[29]中，以及在小型Google数据集上的实验结果；这项工作直接建立在Zinkevich [34]的方法之上。FTRL-Proximal的更理论化处理出现在[26]中。Duchi等人[10]分析了RDA和镜像下降版本，并提供了大量实验结果。

**实验结果。** 我们通过测试两个相同的模型来评估每坐标学习率的影响，一个使用**单一的全局学习率**，另一个使用**每坐标学习率**。基础参数 $\alpha$ 为每个模型单独调整。我们在一个代表性数据集上运行，并使用 AucLoss 作为评估指标（见第 5 节）。结果表明，使用每坐标学习率相比全局学习率基线将 AucLoss 降低了 11.2%。**为了理解这一结果的意义，在我们的环境中，AucLoss 降低 1% 就被认为是大的。**

> [!NOTE]
>
> 3.1 每坐标学习率
>
> 核心洞察：**不同特征出现的频率不同，应该有不同的学习率。**
>
> - 全局学习率 $\eta_t = 1/\sqrt{t}$ 对所有坐标相同，不区分特征频率
> - 每坐标学习率： $\eta_{t,i} = \frac{\alpha}{\beta + \sqrt{\sum_{s=1}^t g_{s,i}^2}}$
>
> 直觉：**出现频繁的特征，学习率快速下降（因为已经有足够数据）；罕见特征保持高学习率（需要对新数据快速反应）。**
>
> 效果：相比全局学习率，**AucLoss降低了11.2%**（在广告场景中，1%就是显著的改进）。


## 4. 大规模下的内存节省

如上所述，我们**使用L1正则化来节省预测时的内存**。在本节中，我们描述**在训练期间节省内存的额外技巧**。

### 4.1 概率特征包含

在许多具有**高维数据**的领域中，**绝大多数特征都极其罕见**。事实上，在我们的一些模型中，**一半的唯一特征在包含数十亿样本的整个训练集中只出现一次**。⁴

> 注脚：⁴ 由于本文专门处理极其稀疏的数据，我们说一个特征在样本中以非零值出现时，该特征"出现"。

**跟踪那些永远不可能真正有用的罕见特征的统计信息是昂贵的**。不幸的是，我们无法事先知道哪些特征会是罕见的。在在线环境中预处理数据以移除罕见特征是有问题的：额外的数据读取和写入非常昂贵，并且如果某些特征被丢弃（比如，因为它们出现次数少于k次），就无法再尝试使用这些特征的模型来估计预处理在精度方面的成本。

一类方法通过实现 L1正则化 来在训练中实现稀疏性，这种方法不需要跟踪任何系数为零的特征的统计信息（例如[20]）。这使得信息量较少的特征可以在训练过程中被移除。然而，我们发现这种风格的稀疏化相比那些 **在训练中跟踪更多特征 而 仅为服务进行稀疏化的方法**（如FTRL-Proximal），会**导致不可接受的精度损失**。另一个常见的解决方案是**带冲突的哈希**，也没有带来有用的益处（见第9.1节）。

我们探索的另一类方法是 **概率特征包含**，其中**新特征在首次出现时以概率方式被包含到模型中**。**这实现了预处理数据的效果，但可以在在线环境中执行**。

我们测试了两种方法。

*   **泊松包含。** 当我们遇到一个尚未在模型中的特征时，我们仅以概率 $p$ 将其添加到模型中。一旦特征被添加，在后续的观察中，我们照常更新其系数值和OGD使用的相关统计信息。一个特征在被添加到模型之前需要被看到的次数服从期望值为 $1/p$ 的几何分布。

*   **布隆过滤器包含。** 我们使用一组滚动的计数布隆过滤器[4, 12]来检测特征在训练中被遇到的首次 $n$ 次。一旦特征出现次数超过 $n$ 次（根据过滤器），我们将其添加到模型中，并在后续观察中如上所述用于训练。注意，这种方法也是概率性的，因为计数布隆过滤器可能产生假阳性（但不会产生假阴性）。也就是说，我们有时会包含一个实际出现次数少于 $n$ 次的特征。

**实验结果。** 这些方法的效果见表2，表明两种方法都效果良好，但布隆过滤器方法在RAM节省与预测质量损失之间提供了更好的权衡集。

| 方法 | RAM节省 | AucLoss损害 |
|------|---------|-----------|
| Bloom (n=2) | 66% | 0.008% |
| Bloom (n=1) | 55% | 0.003% |
| Poisson (p=0.03) | 60% | 0.020% |
| Poisson (p=0.1) | 40% | 0.006% |

表2：概率特征包含的效果。两种方法都有效，但布隆过滤器方法在RAM节省与预测精度之间提供了更好的权衡。


### 4.2 使用更少位编码值

OGD的朴素实现使用32位或64位浮点编码来存储 **系数值**。浮点编码通常因其 **大动态范围** 和 **细粒度精度** 而具有吸引力；然而，对于我们的正则化逻辑回归模型的系数而言，这被证明是过度的。几乎所有的系数值都落在(-2, +2)范围内。分析表明，细粒度精度也是不必要的[14]，这促使我们探索使用**定点q2.13编码**而不是浮点。

在q2.13编码中，我们在二进制小数点左侧保留两位，二进制小数点右侧保留十三位，以及一位符号位，每个值总共使用16位。

这种降低的精度可能会在OGD环境中产生累积舍入误差问题，OGD需要累积大量微小的步骤。（事实上，我们甚至在使用32位浮点而非64位时看到了严重的舍入问题。）然而，一个简单的随机舍入策略以增加一个小额外遗憾项为代价纠正了这一点[14]。关键在于，通过显式舍入，我们可以确保离散化误差的均值为零。

特别地，如果我们正在存储系数 $\mathbf{w}$ ，我们设置

$$
w_{i,\text{rounded}} = 2^{-13} \lfloor 2^{13} w_i + R \rfloor \qquad (3)
$$

其中 $R$ 是均匀分布在 $0$ 和 $1$ 之间的随机偏差。 $g_{i,\text{rounded}}$ 随后以 q2.13 定点格式存储；范围 $[-4, 4)$ 之外的值被截断。对于 FTRL-Proximal，我们可以以这种方式存储 $\eta_t \mathbf{z}_t$ ，其量级与 $\mathbf{w}_t$ 相似。

**实验结果。** 在实践中，我们观察到使用q2.13编码代替64位浮点值的模型相比，没有可测量的精度损失，并且我们节省了75%的系数存储RAM。

> [!NOTE]
>
> 4.2 完全未理解。纯数学，很高级的玩法


### 4.3 训练多个相似模型

在测试超参数设置或特征的更改时，评估一种或另一种形式的许多轻微变体通常是有用的。这种常见用例允许高效的训练策略。这方面一项有趣的工作是[19]，它使用一个固定模型作为先验，并允许通过残差误差评估变体。这种方法非常廉价，但不能轻松评估特征移除或替代学习设置。

我们的主要方法基于这样的观察：每个坐标依赖于某些可以在模型变体之间高效共享的数据，而其他数据（如系数值本身）是每个模型变体特有的，不能共享。如果我们将模型系数存储在哈希表中，我们可以为所有变体使用单个表，分摊存储键（可以是字符串或多字节哈希）的成本。在下一节（4.5）中，我们展示了每个模型的学习率计数器 $n_i$ 可以被所有变体共享的统计信息替代，这也减少了存储空间。

任何不具有某个特定特征的变体将把该特征的系数存储为0，浪费少量空间。（我们通过将这些特征的学习率设置为0来强制执行这一点。）由于我们只一起训练高度相似的模型，不表示每个模型的键和计数所节省的内存远大于非共同特征造成的损失。

当多个模型一起训练时，所有每坐标元数据（如每坐标学习率所需的计数）的分摊成本降低，额外模型的增量成本仅取决于需要存储的额外系数值。这不仅节省了内存，还节省了网络带宽（值以相同方式通过网络传输，并且我们只读取训练数据一次）、CPU（只需一次哈希表查找而不是多次，并且特征只从训练数据生成一次而不是每个模型一次）以及磁盘空间。这种捆绑架构显著提高了我们的训练容量。

> [!NOTE]
>
> 4.3 完全未理解


### 4.4 单值结构

有时我们希望一起评估非常大的模型变体集合，这些变体仅通过添加或移除小群组特征而不同。在这里，我们可以使用一种更加压缩的数据结构，它既有损又特设，但在实践中给出了非常实用的结果。这种单值结构为每个坐标仅存储一个系数值，由包含该特征的所有模型变体共享，而不是为每个模型变体存储单独的系数值。一个位域用于跟踪哪些模型变体包含给定坐标。注意，这在思想上类似于[19]的方法，但也允许评估特征的移除以及添加。RAM成本随额外模型变体的增长速度比第4.3节的方法慢得多。

学习过程如下。对于OGD中的给定更新，每个模型变体使用其包含的坐标子集计算预测和损失，利用为每个系数存储的单值。对于每个特征i，每个使用i的模型计算给定系数的新期望值。结果值被平均并存储为单个值，然后在下一步由所有变体共享。

我们通过比较使用单值结构训练的大型模型变体组与使用第4.3节设置精确训练的相同变体来评估这种启发式方法。结果显示各变体之间的相对性能几乎相同，但单值结构节省了一个数量级的RAM。

> [!NOTE]
>
> 4.4 完全未理解


### 4.5 使用计数计算学习率

如第3.1节所述，我们需要为每个特征存储梯度的和以及梯度的平方和。梯度计算必须正确，但对于学习率计算可以做出粗略近似。

假设所有包含某个特征的事件具有相同的概率。（通常，这是一个很差的近似，但对此目的是有效的。）进一步假设模型已准确学习了该概率。如果有 $N$ 个负事件和 $P$ 个正事件，则概率为 $p = P/(N + P)$ 。如果我们使用逻辑回归，正事件的梯度为 $p-1$ ，负事件的梯度为 $p$ ，学习率等式 (2) 所需的梯度平方和为

$$
\sum g_{t,i}^2 = \sum_{\text{positive events}} (1 - p_t)^2 + \sum_{\text{negative events}} p_t^2
\approx P \left(1 - \frac{P}{N+P}\right)^2 + N \left(\frac{P}{N+P}\right)^2
= \frac{PN}{N+P}
$$

这种粗略的近似允许我们只跟踪计数 $N$ 和 $P$ ，并省去存储 $\sum g_{t,i}^2$ 。经验上，使用这种近似计算的学习率对我们来说与使用完整和计算的学习率效果一样好。使用第 4.3 节的框架，总存储成本更低，因为**所有变体模型具有相同的计数**，因此 $N$ 和 $P$ 的存储成本被分摊。计数可以以可变长度位编码存储，并且绝大多数特征不需要很多位。


### 4.6 训练数据子采样

典型的CTR远低于50%，这意味着正样本（点击）相对罕见。因此，简单的统计计算表明**点击在学习CTR估计中相对更有价值**。我们可以利用这一点**在最小影响精度的情况下显著减少训练数据大小**。我们通过在我们的样本中包括以下内容来**创建子采样训练数据**：

*   至少有一个广告被点击的任何查询。
*   所有广告都未被点击的查询中的一个比率 $r \in (0, 1]$ 。

**在查询级别进行采样是可取的**，因为计算许多特征需要对查询短语进行共同处理。当然，在这种子采样数据上朴素地训练会导致显著有偏的预测。这个问题可以通过为每个样本分配重要性权重 $\omega_t$ 轻松解决，其中
$$
\omega_t =
\begin{cases}
1 & \text{event } t \text{ in a query with a click} \\
1/r & \text{event } t \text{ in a query without a click}
\end{cases}
$$

由于我们控制采样分布，我们不需要像一般样本选择那样估计权重 $\omega$ [7]。重要性权重简单地按比例缩放每个事件上的损失（等式 (1)），从而也按比例缩放梯度。要看到这具有预期的效果，考虑未采样数据中随机选择的事件 $t$ 对子采样目标函数的期望贡献。令 $s_t$ 为事件 $t$ 被采样的概率（ $1$ 或 $r$ ），因此根据定义 $s_t = 1/\omega_t$ 。于是，我们有

$$
\mathbb{E}[\ell_t(w_t)] = s_t \omega_t \ell_t(w_t) + (1 - s_t) \cdot 0 = s_t \cdot \frac{1}{s_t} \ell_t(w_t) = \ell_t(w_t).
$$

期望的线性性意味着子采样训练数据上的期望加权目标函数等于原始数据集上的目标函数。实验已验证，即使相当激进的未点击查询子采样对精度的影响也非常温和，且预测性能不会特别受 $r$ 的具体值影响。

> [!NOTE]
>
> 可以借鉴


## 5. 评估模型性能

评估我们模型的质量最廉价的方式是通过使用记录的历史数据。（**在实时流量的一部分上评估模型是重要的但更昂贵的评估手段**；参见例如[30]。）

由于不同的指标对模型变化的响应方式不同，我们发现通常在一系列可能的性能指标上评估模型变化是有用的。我们计算诸如 AucLoss（即 $1-\text{AUC}$ ，其中 AUC 是标准 ROC 曲线下面积指标 [13]）、LogLoss（见等式 (1)）和 SquaredError 等指标。为了一致性，我们还设计指标使得值越小总是越好。

### 5.1 渐进验证

我们通常使用渐进验证（有时称为**在线损失**）[5]而不是交叉验证或在保留数据集上评估。因为计算用于学习的梯度无论如何都需要计算预测，我们可以廉价地将这些预测流式输出用于后续分析，按小时聚合。我们还在数据的各种子切片上计算这些指标，例如按国家、查询主题和布局的细分。

在线损失是我们服务查询精度的一个良好代理，因为它仅**衡量我们在训练之前对最新数据的性能——正好类似于模型服务查询时发生的情况**。在线损失还具有比保留验证集好得多的统计特性，因为我们可以使用100%的数据进行训练和测试。这一点很重要，因为即使小的改进也能在规模上产生有意义的影响，并且需要大量数据才能以高置信度观察到。

绝对指标值通常具有误导性。即使预测是完美的，LogLoss和其他指标也会根据问题的难度（即贝叶斯风险）而变化。如果点击率接近50%，可达到的最佳LogLoss远高于点击率接近2%的情况。这一点很重要，因为点击率因国家和查询而异，因此在单一天的进程中平均值也会变化。

因此，我们总是查看相对变化，通常表示为相对于基线模型的指标百分比变化。根据我们的经验，相对变化随时间要稳定得多。我们还注意只比较从完全相同数据计算的指标；例如，在一个时间范围内对模型计算的损失指标与在另一个时间范围内对另一个模型计算的相同损失指标是不可比较的。


### 5.2 通过可视化深入理解

大规模学习中一个潜在的陷阱是聚合性能指标可能隐藏特定于数据某些子群体的影响。例如，一个指标上的小幅聚合精度提升实际上可能是由不同国家或特定查询主题的正向和负向变化混合造成的。这使得不仅提供数据的聚合指标，而且提供数据的各种切片的指标至关重要，例如按国家或按主题。

由于有数百种有意义的数据切片方式，能够有效地查看数据的可视摘要变得至关重要。为此，我们开发了一个高维交互式可视化工具GridViz，以允许全面理解模型性能。

![image-20260719153342397](/Users/dazhang/PycharmProject/Papers/3-RecSys/.picture/image-20260719153342397.png)

GridViz 的一个视图截图如图 2 所示，显示了一组按查询主题的切片，针对两个模型与一个对照模型进行比较。指标值由彩色单元格表示，行对应模型名称，列对应数据的每个唯一切片。列宽度表示切片的重要性，可以设置为反映诸如展示次数或点击次数等数量。单元格的颜色反映了与所选基线相比的指标值，这使得可以快速扫描异常值和感兴趣的区域，以及直观理解整体性能。当列足够宽时，显示所选指标的数值。可以选择多个指标；这些指标在每行中一起显示。当用户鼠标悬停在单元格上时，会弹出该单元格的详细报告。

由于有数百种可能的切片，我们设计了一个交互式界面，允许用户通过下拉菜单或通过切片名称的正则表达式选择不同的切片组。列可以排序，颜色刻度的动态范围可以修改以适应手头的数据。总体而言，这个工具使我们能够显著加深对模型在各种数据子集上性能的理解，并识别出高影响力的改进领域。


## 6. 置信度估计

对于许多应用，重要的是不仅估计广告的CTR，还要**量化预测的期望精度**。特别是，这样的估计可用于**测量和控制探索/利用权衡**：为了做**出准确的预测，系统必须有时展示数据较少的广告，但这应与展示已知良好的广告的收益进行平衡**[21, 22]。

置信区间捕捉了不确定性的概念，但出于实际和统计原因，它们不适合我们的应用。标准方法会评估一个完全收敛的批处理模型（没有正则化）的预测置信度；我们的模型是在线的，不假设IID数据（因此收敛甚至没有明确定义），并且高度正则化。标准的统计方法（例如 [18]，第 2.5 节）还需要求逆 $n \times n$ 矩阵；当 $n$ 达到数十亿时，这是不可行的。

此外，任何置信度估计必须能够在预测时以极低的成本计算——比如说大约与进行预测本身相同的时间。

我们提出了一种称为不确定性分数的启发式方法，它在计算上可行且经验上很好地量化了预测精度。基本的观察是，学习算法本身在用于学习率控制的每特征计数器 $n_{t,i}$ 中维护了不确定性的概念。 $n_i$ 较大的特征获得较小的学习率，正是因为我们相信当前的系数值更可能是准确的。逻辑损失关于对数几率分数的梯度是 $(p_t - y_t)$ ，因此其绝对值以 $1$ 为界。因此，如果我们假设特征向量被归一化使得 $|x_{t,i}| \leq 1$ ，我们可以限制由于观察单个训练样本 $(\mathbf{x}, y)$ 而导致的对数几率预测的变化。为简单起见，考虑 $\lambda_1 = \lambda_2 = 0$ ，此时 FTRL-Proximal 等价于在线梯度下降。令 $n_{t,i} = \beta + \sum_{s=1}^t g_{s,i}^2$ ，按照等式 (2)，我们有

$$
|\mathbf{x} \cdot \mathbf{w}_t - \mathbf{x} \cdot \mathbf{w}_{t+1}| = \sum_{i : |x_i| > 0} \eta_{t,i} |g_{t,i}|
\leq \alpha \sum_{i : |x_i| > 0} \frac{x_{t,i}}{\sqrt{n_{t,i}}}
= \alpha \boldsymbol{\eta} \cdot \mathbf{x} \equiv u(\mathbf{x})
$$

其中 $\boldsymbol{\eta}$ 是学习率向量。我们将不确定性分数定义为上界 $u(\mathbf{x}) \equiv \alpha \boldsymbol{\eta} \cdot \mathbf{x}$ ；它可以通过一次稀疏点积计算，就像预测 $p = \sigma(\mathbf{w} \cdot \mathbf{x})$ 一样。


**实验结果。** 我们如下验证了这种方法。首先，我们在真实数据上训练了一个"真实"模型，但使用了与通常略有不同的特征。然后，我们丢弃了真实的点击标签，并根据真实模型预测作为真实 CTR 来采样新的标签。这是必要的，因为评估置信度过程的有效性需要知道真实标签。然后我们在重新标记的数据上运行 FTRL-Proximal，记录预测 $p_t$ ，这使得我们能够比较对数几率空间中的预测精度， $e_t = |\sigma^{-1}(p_t) - \sigma^{-1}(p^*_t)|$ ，其中 $p^*_t$ 是真实 CTR（由真实模型给出）。图 3 绘制了作为不确定性分数 $u_t = u(\mathbf{x}_t)$ 函数的误差 $e_t$ ；两者之间存在高度相关性。

<img src="/Users/dazhang/PycharmProject/Papers/3-RecSys/.picture/image-20260719153408849.png" alt="image-20260719153408849" style="zoom:33%;" />

额外的实验表明，不确定性分数（在上述评估机制下）与通过对数据随机子样本训练的32个模型的bootstrap获得的昂贵得多估计相比，性能相当。


## 7. 校准预测

准确且良好校准的预测不仅对运行拍卖至关重要，它们还允许松散耦合的整体系统设计，将拍卖中的优化问题与机器学习机制分离开来。

系统偏差（在某些数据切片上**平均预测CTR与观察到的CTR之间的差异**）可能由多种因素引起，例如不准确的建模假设、学习算法的缺陷或训练和/或服务时不可用的隐藏特征。为解决这个问题，我们可以使用校准层来匹配预测CTR与观察到的点击率。

如果在平均而言当我们预测 $p$ 时实际的观察 CTR 接近 $p$ ，则我们的预测在数据切片 $d$ 上被校准。我们可以通过应用校正函数 $\tau_d(p)$ 来改进校准，其中 $p$ 是预测 CTR， $d$ 是训练数据的一个划分元素。我们将成功定义为在数据的各种可能划分上提供良好校准的预测。

建模 $\tau$ 的一个简单方法是拟合函数 $\tau(p) = \gamma p^\kappa$ 到数据。我们可以使用泊松回归在聚合数据上学习 $\gamma$ 和 $\kappa$ 。一个稍微更通用的方法能够处理偏置曲线中更复杂的形状，是使用分段线性或分段常数校正函数。唯一的限制是映射函数 $\tau$ 应该是等张的（单调递增）。我们可以使用等张回归找到这样的映射，它在该约束下计算输入数据的加权最小二乘拟合（参见例如 [27, 23]）。与上述合理的基线方法相比，这种分段线性方法显著降低了预测在范围高低两端的偏差。

值得注意的是，没有强的额外假设，系统中固有的反馈循环使得不可能提供校准影响的理论保证[25]。

> [!NOTE]
>
> 没看懂


## 8. 自动化特征管理

可扩展机器学习的一个重要方面是**管理安装的规模**，包括构成机器学习系统的所有配置、开发者、代码和计算资源。一个由多个团队组成的安装环境建模数十个特定领域的问题需要一些开销。一个特别有趣的案例是用于机器学习的特征空间的管理。

我们可以将特征空间描述为一组上下文和语义信号，其中每个信号（例如"广告中的词语"、"来源国"等）可以被转换为用于学习的一组实值特征。在一个大型安装环境中，许多开发者可能异步地从事信号开发。一个信号可能有多个版本，对应于配置更改、改进和替代实现。一个工程团队可能使用他们不直接开发的信号。信号可能在多个不同的学习平台上被使用，并应用于不同的学习问题（例如预测搜索与展示广告CTR）。为了处理用例的组合增长，我们部署了一个元数据索引，用于管理数百个活跃模型对数千个输入信号的使用。

索引信号被手动和自动注释，涉及各种问题；示例包括废弃、特定平台的可用性以及特定领域的适用性。新模型和活跃模型使用的信号通过自动警报系统进行审查。不同的学习平台共享一个通用接口，用于向中心索引报告信号使用情况。当一个信号被弃用时（例如当新版本可用时），我们可以快速识别该信号的所有使用者并跟踪替换工作。当信号的改进版本可用时，可以提醒使用者尝试新版本。

新信号可以通过自动测试进行审查并加入白名单。白名单可用于确保生产系统的正确性，以及用于使用自动特征选择的学习系统。不再被使用的旧信号被自动标记以进行代码清理和任何相关数据的删除。

有效的自动信号消费管理确保更多的学习在第一次就正确完成。这减少了浪费和重复的工程努力，节省大量的工程工时。在运行学习算法之前验证配置的正确性，消除了许多可能导致不可用模型的情况，避免了显著的潜在资源浪费。

> [!NOTE]
>
> 没读懂


## 9. 不成功的实验

在最后一节中，我们简要报告几个（可能令人惊讶的）没有产生显著益处的方向。

### 9.1 激进特征哈希

近年来，围绕**使用特征哈希来降低大规模学习的 RAM 成本**出现了大量活动。值得注意的是，[31] 报告了使用 哈希技巧 将能够学习个性化垃圾邮件过滤模型的特征空间投影到只有 $2^{24}$ 个特征的空间中的出色结果，使得**模型足够小可以轻松放入单台机器的 RAM 中**。类似地，Chapelle 报告了使用哈希技巧得到 $2^{24}$ 个结果特征来建模展示广告数据 [6]。

我们测试了这种方法，但发现我们**无法在不产生可观察损失的情况下将特征投影到低于数十亿个特征**。这并没有为我们提供显著的节省，并且我们更倾向于维护**可解释的**（非哈希）特征向量。

### 9.2 Dropout

最近的工作对训练中随机化"dropout"的新技术产生了兴趣，特别是在深度信念网络社区中 [17]。主要思想是**以概率 $p$ 独立地从输入样本向量中随机移除特征**，并通过**在测试时将结果权重向量乘以因子 $(1-p)$ ** 来补偿这一点。这被视为一种正则化形式，模拟了在可能的特征子集上的 bagging。

我们实验了一系列从0.1到0.5的dropout率，每个都伴随有学习率设置的网格搜索，包括在数据上改变遍历次数。在所有情况下，我们发现dropout训练在预测精度指标 或 泛化能力上没有带来益处，并且大多数情况下产生了损害。

我们认为这些负面结果与视觉社区的有希望结果之间的差异根源在于**特征分布的差异**。**在视觉任务中，输入特征通常是密集的，而在我们的任务中，输入特征是稀疏的且标签是有噪声的。在密集环境中，dropout有助于 分离强相关特征 的效果，从而产生更稳健的分类器。但在我们稀疏、噪声的环境中，添加dropout似乎只是减少了可用于学习的数据量。**

### 9.3 特征Bagging

我们研究的另一种类似dropout的训练变体是特征bagging，其中**k个模型在k个重叠的特征子集上独立训练。模型的输出被平均以得到最终预测**。这种方法在数据挖掘社区中被广泛使用，最著名的是与决策树集成一起使用[9]，提供了**管理偏差-方差权衡的潜在有用方式**。我们也对此感兴趣，认为它是一种可能有助于进一步并行化训练的方法。然而，我们发现特征bagging实际上略微降低了预测质量，根据bagging方案的不同，AucLoss在0.1%到0.6%之间。

### 9.4 特征向量归一化

在我们的模型中，**每个事件的非零特征数量可能显著不同**，导致不同的样本 $\mathbf{x}$ 具有不同的量级 $\|\mathbf{x}\|$ 。我们担心这种变异性可能会 **减慢收敛速度或影响预测精度**。我们探索了几种归一化的方式，使用 $\mathbf{x}/\|\mathbf{x}\|$ 进行训练，采用多种范数，目标是**减少样本向量之间量级的方差**。尽管早期结果显示了一些小的精度提升，我们无法将其转化为总体正向指标。事实上，我们的实验看起来有些损害，可能是由于与每坐标学习率和正则化的交互作用。


## 10. 致谢

我们衷心感谢以下人员的贡献：Vinay Chaudhary, Jean-Francois Crespo, Jonathan Feinberg, Mike Hochberg, Philip Henderson, Sridhar Ramaswamy, Ricky Shan, Sajid Siddiqi, 以及 Matthew Streeter。

---

### 脚注

¹ OGD 本质上与随机梯度下降相同；名称"在线"强调我们不是在解决批处理问题，而是对可能不是 IID 的样本序列进行预测。

² 形式上，标准梯度下降的遗憾（regret）为 $\Omega(T^{2/3})$ （参见例如 [34]），而独立副本的遗憾为 $O(T^{1/2})$ 。

³ 对于固定的梯度序列，如果取 $\alpha$ 为 $w_i$ 最大允许幅度的两倍，且 $\beta = 0$ ，则我们的遗憾上界在任意非递增每坐标学习率调度 [29] 的最佳可能遗憾上界（不是遗憾）的 $\sqrt{2}$ 倍以内。

---


## 11. 参考文献

[1] D. Agarwal, B.-C. Chen, and P. Elango. Spatio-temporal models for estimating click-through rate. In Proceedings of the 18th international conference on World wide web, pages 21–30. ACM, 2009.

[2] R. Ananthanarayanan, V. Basker, S. Das, A. Gupta, H. Jiang, T. Qiu, A. Reznichenko, D. Ryabkov, M. Singh, and S. Venkataraman. Photon: Fault-tolerant and scalable joining of continuous data streams. In SIGMOD Conference, 2013. To appear.

**[3] R. Bekkerman, M. Bilenko, and J. Langford. Scaling up machine learning: Parallel and distributed approaches. 2011.**

[4] B. H. Bloom. Space/time trade-offs in hash coding with allowable errors. Commun. ACM, 13(7), July 1970.

[5] A. Blum, A. Kalai, and J. Langford. Beating the hold-out: Bounds for k-fold and progressive cross-validation. In COLT, 1999.

**[6] O. Chapelle. Click modeling for display advertising. In AdML: 2012 ICML Workshop on Online Advertising, 2012.**

[7] C. Cortes, M. Mohri, M. Riley, and A. Rostamizadeh. Sample selection bias correction theory. In ALT, 2008.

**[8] J. Dean, G. S. Corrado, R. Monga, K. Chen, M. Devin, Q. V. Le, M. Z. Mao, M. Ranzato, A. Senior, P. Tucker, K. Yang, and A. Y. Ng. Large scale distributed deep networks. In NIPS, 2012.**

[9] T. G. Dietterich. An experimental comparison of three methods for constructing ensembles of decision trees: Bagging, boosting, and randomization. Machine learning, 40(2):139–157, 2000.

[10] J. Duchi, E. Hazan, and Y. Singer. Adaptive subgradient methods for online learning and stochastic optimization. In COLT, 2010.

[11] J. Duchi and Y. Singer. Efficient learning using forward-backward splitting. In Advances in Neural Information Processing Systems 22, pages 495–503. 2009.

[12] L. Fan, P. Cao, J. Almeida, and A. Broder. Summary cache: a scalable wide-area web cache sharing protocol. IEEE/ACM Transactions on Networking, 8(3), jun 2000.

[13] T. Fawcett. An introduction to roc analysis. Pattern recognition letters, 27(8):861–874, 2006.

[14] D. Golovin, D. Sculley, H. B. McMahan, and M. Young. Large-scale learning with a small-scale footprint. In ICML, 2013. To appear.

[15] T. Graepel, J. Q. Candela, T. Borchert, and R. Herbrich. Web-scale Bayesian click-through rate prediction for sponsored search advertising in microsofts bing search engine. In Proc. 27th Internat. Conf. on Machine Learning, 2010.

[16] D. Hillard, S. Schroedl, E. Manavoglu, H. Raghavan, and C. Leggetter. Improving ad relevance in sponsored search. In Proceedings of the third ACM international conference on Web search and data mining, WSDM '10, pages 361–370, 2010.

[17] G. E. Hinton, N. Srivastava, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov. Improving neural networks by preventing co-adaptation of feature detectors. CoRR, abs/1207.0580, 2012.

[18] D. W. Hosmer and S. Lemeshow. Applied logistic regression. Wiley-Interscience Publication, 2000.

[19] H. A. Koepke and M. Bilenko. Fast prediction of new feature utility. In ICML, 2012.

[20] J. Langford, L. Li, and T. Zhang. Sparse online learning via truncated gradient. JMLR, 10, 2009.

[21] S.-M. Li, M. Mahdian, and R. P. McAfee. Value of learning in sponsored search auctions. In WINE, 2010.

[22] W. Li, X. Wang, R. Zhang, Y. Cui, J. Mao, and R. Jin. Exploitation and exploration in a performance based contextual advertising system. In KDD, 2010.

[23] R. Luss, S. Rosset, and M. Shahar. Efficient regularized isotonic regression with application to gene–gene interaction search. Ann. Appl. Stat., 6(1), 2012.

[24] H. B. McMahan. Follow-the-regularized-leader and mirror descent: Equivalence theorems and L1 regularization. In AISTATS, 2011.

[25] H. B. McMahan and O. Muralidharan. On calibrated predictions for auction selection mechanisms. CoRR, abs/1211.3955, 2012.

[26] H. B. McMahan and M. Streeter. Adaptive bound optimization for online convex optimization. In COLT, 2010.

[27] A. Niculescu-Mizil and R. Caruana. Predicting good probabilities with supervised learning. In ICML, ICML '05, 2005.

[28] M. Richardson, E. Dominowska, and R. Ragno. Predicting clicks: estimating the click-through rate for new ads. In Proceedings of the 16th international conference on World Wide Web, pages 521–530. ACM, 2007.

[29] M. J. Streeter and H. B. McMahan. Less regret via online conditioning. CoRR, abs/1002.4862, 2010.

**[30] D. Tang, A. Agarwal, D. O'Brien, and M. Meyer. Overlapping experiment infrastructure: more, better, faster experimentation. In KDD, pages 17–26, 2010.**

**[31] K. Weinberger, A. Dasgupta, J. Langford, A. Smola, and J. Attenberg. Feature hashing for large scale multitask learning. In ICML, pages 1113–1120. ACM, 2009.**

[32] L. Xiao. Dual averaging method for regularized stochastic learning and online optimization. In NIPS, 2009.

[33] Z. A. Zhu, W. Chen, T. Minka, C. Zhu, and Z. Chen. A novel click model and its applications to online advertising. In Proceedings of the third ACM international conference on Web search and data mining, pages 321–330. ACM, 2010.

[34] M. Zinkevich. Online convex programming and generalized infinitesimal gradient ascent. In ICML, 2003.

