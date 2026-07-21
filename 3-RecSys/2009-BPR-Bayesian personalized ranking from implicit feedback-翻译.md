# BPR: Bayesian Personalized Ranking from Implicit Feedback

> Steffen Rendle, Christoph Freudenthaler, Zeno Gantner, Lars Schmidt-Thieme | University of Hildesheim
机器学习实验室，希尔德斯海姆大学，德国  
{srendle, freudenthaler, gantner, schmidt-thieme}@ismll.de



本文介绍了BPR（贝叶斯个性化排序），将 隐式反馈 推荐问题形式化为 **item对排序**，提出BPR-Opt准则和LearnBPR算法。核心内容：

- **BPR-Opt**：基于贝叶斯后验概率最大化推导的排序优化目标，**与AUC最大化等价**
- **LearnBPR**：**基于自助采样的随机梯度下降**，解决隐式反馈数据 稀疏和偏斜问题
- **应用**：将BPR应用于矩阵分解（BPR-MF）和自适应k近邻（BPR-kNN）

关键发现：直接优化排序比优化评分预测显著提升推荐质量；**模型预测质量不仅取决于模型本身，更取决于优化准则的选择**。

> [!CAUTION]
>
> 「**基于自助采样的随机梯度下降**」BPR 原文在此处用语不严谨，实际为梯度上升。因为全文的优化目标都是最大化后验概率，是一个最大化问题，因此后续所有「梯度下降」都应理解为「梯度上升」

---



## 摘要

item推荐是一项预测item集（例如网站、电影、产品）上个性化排序的任务。在本文中，我们研究了最常见的带有隐式反馈（例如点击、购买）的场景。有许多方法可用于从隐式反馈中进行item推荐，如矩阵分解（MF）或自适应k近邻（kNN）。尽管这些方法是为个性化排序这一item预测任务而设计的，**但其中没有一种方法直接针对排序进行优化**。

在本文中，我们提出了一种**通用的优化准则BPR-Opt**用于个性化排序，它是**从问题的贝叶斯分析中推导出的最大后验估计**。我们还提供了一种通用的学习算法，用于根据BPR-Opt优化模型。该学习方法**基于带自助采样的随机梯度下降**。我们展示了如何将我们的方法应用于两种最先进的推荐模型：矩阵分解和自适应kNN。我们的实验表明，对于个性化排序任务，我们的优化方法优于MF和kNN的**标准学习技术**。结果显示了**为正确的准则优化模型的重要性**。



## 1 引言

推荐内容在许多信息系统中都是一项重要任务。例如，像Amazon这样的在线购物网站为每位顾客提供个性化产品推荐，这些产品可能是用户感兴趣的。其他例子包括像YouTube这样的视频门户网站，它们向顾客推荐电影。个性化对于内容提供商（可以增加销售额或浏览量）和顾客（可以更容易找到感兴趣的内容）都很有吸引力。在本文中，我们聚焦于item推荐。item推荐的任务是为一组item创建用户特定的排序。**关于item的用户偏好从用户过去与系统的交互中学习——例如用户的购买历史、浏览历史等。**

推荐系统是一个活跃的研究课题。大多数近期工作研究的都是用户提供显式反馈的场景，例如评分形式。然而，在现实场景中，大多数反馈不是显式的，而是隐式的。**隐式反馈被自动追踪，如监控点击、观看时间、购买等。因此它更容易收集，因为用户不需要显式表达他的喜好。**事实上，隐式反馈几乎已经在任何信息系统中可用——例如，Web服务器在日志文件中记录每次页面访问。

在本文中，我们提出了一种用于**学习个性化排序模型的通用方法**。本文的贡献如下：

1. 我们提出了**通用优化准则BPR-Opt**，从最优个性化排序的最大后验估计推导而来。我们展示了BPR-Opt与ROC曲线下面积最大化的类比关系。

2. 为了最大化BPR-Opt，我们提出了**通用学习算法LearnBPR**，它基于带训练三元组自助采样的随机梯度下降。我们展示了我们的算法在优化BPR-Opt方面优于**标准梯度下降技术**。

3. 我们展示了如何将LearnBPR应用于两类最先进的推荐模型。

4. 我们的实验经验性地表明，对于**个性化排序任务，使用BPR学习模型优于其他学习方法**。



## 2 相关工作

最流行的推荐系统模型是k近邻（kNN）协同过滤[2]。传统上，**kNN的相似度矩阵通过启发式方法计算**——例如皮尔逊相关系数——但在近期工作中[8]，**相似度矩阵被视为模型参数并专门针对任务进行学习**。近年来，矩阵分解（MF）在推荐系统中变得非常流行，既用于隐式反馈也用于显式反馈。在早期工作中[13]，奇异值分解（SVD）被提出用于学习特征矩阵。通过SVD学习的MF模型已被证明非常容易过拟合。因此，**正则化学习方法被提出**。对于item预测，Hu等人[5]和Pan等人[10]提出了带案例权重的正则化最小二乘优化（WR-MF）。案例权重可用于减少负例的影响。Hofmann[4]提出了一种用于item推荐的概率潜在语义模型。Schmidt-Thieme[14]将问题转换为多类问题，并用一组二元分类器解决。尽管上述讨论的所有关于item预测的工作都是在个性化排序数据集上评估的，但这些方法中没有一个直接针对排序优化其模型参数。相反，**它们优化的是预测一个item是否被用户选择**。

在我们的工作中，我们推导了一个基于item对（即用户对两个item的特定顺序）的个性化排序优化准则。**我们将展示如何将像MF或自适应kNN这样的最先进模型相对于这个准则进行优化，以提供比通常学习方法更好的排序质量。**关于我们的方法与Hu等人[5]和Pan等人[10]的WR-MF方法以及最大间隔矩阵分解[15]之间关系的详细讨论可以在第5节找到。在第4.1.1节中，我们还将讨论我们的优化准则与[3]中AUC优化之间的关系。

在本文中，我们专注于模型参数的离线学习。将学习方法扩展到在线学习场景——例如新用户被添加，其历史记录从0增加到1、2……个反馈事件——已经针对评分预测这一相关任务在MF中进行了研究[11]。同样的fold-in策略可以用于BPR。

还有一些关于使用非协同模型学习排序的相关工作。一个方向是对排列上的分布进行建模[7, 6]。Burges等人[1]使用梯度下降优化用于排序的神经网络模型。所有这些方法只学习一个排序——即它们是非个性化的。与之相反，我们的模型是**学习个性化排序的协同模型**，即为每个用户学习一个单独的排序。在我们的评估中，我们经验性地表明，在典型的推荐场景中，我们的个性化BPR模型甚至优于非个性化排序的理论上界。



## 3 个性化排序

个性化排序的任务是为用户提供item的排序列表，这也被称为**item推荐**。一个例子是在线商店想要推荐一个用户可能想要购买的个性化排序的item列表。在本文中，我们研究需要从用户的隐式行为（例如过去的购买行为）中推断排序的场景。**隐式反馈系统的有趣之处在于只有正观测可用**。**未观测到的用户-item对**（例如用户尚未购买某个item）是**真实负反馈**（用户对购买该item不感兴趣）和 **缺失值**（用户未来可能会购买该item）的混合。

### 3.1 形式化

<img src="/Users/dazhang/PycharmProject/Papers/3-RecSys/.picture/image-20260717155759043.png" alt="image-20260717155759043" style="zoom:33%;" />

设 $U$ 为所有用户的集合，$I$ 为所有item的集合。在我们的场景中，隐式反馈 $S \subseteq U \times I$ 是可用的（见图1左侧）。这种反馈的例子包括在线商店中的购买、视频门户中的观看或网站上的点击。推荐系统的任务现在是为用户提供所有item的个性化全序 $>_u \subset I^2$，其中 $>_u$ 必须满足全序的性质：

- $\forall i, j \in I : i \neq j \Rightarrow i >_u j \vee j >_u i$ （完备性）
- $\forall i, j \in I : i >_u j \wedge j >_u i \Rightarrow i = j$ （反对称性）
- $\forall i, j, k \in I : i >_u j \wedge j >_u k \Rightarrow i >_u k$ （传递性）

为方便起见，我们还定义：
- $I^{+}_u := \{i \in I : (u, i) \in S\}$
- $U^{+}_i := \{u \in U : (u, i) \in S\}$

### 3.2 问题设置分析

正如我们之前指出的，**在隐式反馈系统中，只有正类被观测到**，其余数据是**实际负值和缺失值的混合**。处理缺失值问题最常见的方法是忽略所有缺失值，但这样一来典型的机器学习模型将无法学习任何东西，因为它们无法再区分两个层次。

item推荐器的通常方法是为item预测一个个性化分数 $\hat{x}_{ui}$，该**分数反映用户对该item的偏好**。然后通过根据该分数对item进行排序来生成排序。用于item推荐器的机器学习方法[5, 10]通常从 $S$ 创建训练数据，给 **pairs $(u, i) \in S$ 正类标签**，给 $(U \times I) \setminus S$ 中的所有其他组合负类标签（见图1），然后模型被拟合到这个数据。这意味着模型被优化为对 $S$ 中的元素预测值1，对其余元素预测值0。**这种方法的问题在于，模型未来需要排序的所有元素 $((U \times I) \setminus S)$ 在训练期间都被呈现给学习算法作为负反馈。这意味着一个具有足够表达能力（可以精确拟合训练数据）的模型根本无法排序，因为它只预测0。这类机器学习方法能够预测排序的唯一原因是防止过拟合的策略，比如正则化。**

<img src="/Users/dazhang/PycharmProject/Papers/3-RecSys/.picture/image-20260717155822050.png" alt="image-20260717155822050" style="zoom:33%;" />

我们使用了一种不同的方法，**将item对 作为训练数据，并优化以正确排序item对，而不是对单个item评分，因为这比仅用负值替换缺失值更好地代表了问题。**我们从 $S$ 出发，尝试为每个用户重构 $>_u$ 的部分信息。如果一个item $i$ 被用户 $u$ 浏览过——即 $(u, i) \in S$——那么我们假设该用户偏好这个item胜过所有其他未观测的item。例如，在图2中，用户 $u_1$ 浏览了item $i_2$ 但没有浏览item $i_1$，因此我们假设该用户偏好 $i_2$ 胜过 $i_1$：$i_2 >_u i_1$。**对于用户已经看过的两个item，我们无法推断任何偏好。同样的情况也适用于用户尚未看过的两个item**（例如用户 $u_1$ 的item $i_1$ 和 $i_4$）。为了形式化这一点，我们通过以下方式创建训练数据 $D_S : U \times I \times I$：

$$
D_S := \{(u, i, j) \mid i \in I^{+}_u \land j \in I \setminus I^{+}_u\}
$$

$(u, i, j) \in D_S$ 的语义是：假设用户 $u$ 偏好 $i$ 胜过 $j$。由于 $>_u$ 是反对称的，负例被隐式地考虑。

我们的方法有两个优点：

1. 我们的训练数据同时包含正负对和缺失值。**两个未观测item之间的缺失值正是未来需要排序的item对。这意味着从成对的角度来看，训练数据 $D_S$ 和测试数据是不相交的。**

2. 训练数据是为排序的实际目标创建的，即使用 $>_u$ 的观测子集 $D_S$ 作为训练数据。



## 4 贝叶斯个性化排序（BPR）

在本节中，我们推导一种解决个性化排序任务的通用方法。它包括个性化排序的**通用优化准则BPR-Opt**，该准则通过使用 $p(i >_u j|\Theta)$ 的似然函数和模型参数的先验概率 $p(\Theta)$ 对问题进行贝叶斯分析推导而来。我们展示了与排序统计量AUC（ROC曲线下面积）的类比。**为了根据BPR-Opt学习模型，我们提出了LearnBPR算法**。最后，我们展示了如何将BPR-Opt和LearnBPR应用于两种最先进的推荐算法：矩阵分解和自适应kNN。使用BPR优化后，这些模型能够比使用通常的训练方法生成更好的排序。

### 4.1 BPR优化准则

对所有item $i \in I$ 找到正确个性化排序的贝叶斯形式是最大化以下后验概率，其中 $\Theta$ 表示任意模型类（如矩阵分解）的参数向量。

$$
p(\Theta | >_u) \propto p(>_u | \Theta) p(\Theta)
$$

这里，$>_u$ 是用户 $u$ 期望但潜在的偏好结构。所有用户被假定为彼此独立地行动。我们还假设特定用户的每个item对 $(i, j)$ 的排序独立于其他每个item对的排序。因此，上述用户特定的似然函数 $p(>_u | \Theta)$ 可以首先被重写为单个密度的乘积，其次对所有用户 $u \in U$ 进行组合。

$$
p(>_u | \Theta) = \prod_{u \in U} p(>_u | \Theta) = \prod_{(u,i,j) \in U \times I \times I} p(i >_u j|\Theta)^{\delta((u,i,j) \in D_S)} \cdot (1 - p(i >_u j|\Theta))^{\delta((u,j,i) \notin D_S)}
$$

其中 $\delta$ 是指示函数：$\delta(b) := \begin{cases} 1 & \text{if } b \text{ is true} \\ 0 & \text{otherwise} \end{cases}$

由于一个合理的成对排序方案的**完备性和反对称性**，上述公式可以简化为：

$$
p(>_u | \Theta) = \prod_{u \in U} p(>_u | \Theta) = \prod_{(u,i,j) \in D_S} p(i >_u j|\Theta)
$$

到目前为止，通常不能保证得到一个个性化的全序。为了建立这一点，需要满足已经提到的合理性质（完备性、反对称性和传递性）。为此，我们定义用户确实偏好item $i$ 胜过item $j$ 的个体概率为：

$$
p(i >_u j|\Theta) := \sigma(\hat{x}_{uij}(\Theta))
$$

其中 $\sigma$ 是logistic sigmoid函数：$\sigma(x) := \frac{1}{1 + e^{-x}}$

这里 $\hat{x}_{uij}(\Theta)$ 是模型参数向量 $\Theta$ 的**任意实值函数**，它捕捉用户 $u$、item $i$ 和 item $j$ 之间的特定关系。换句话说，我们的通用框架将对 $u$、$i$ 和 $j$ 之间关系建模的任务**委托给底层的模型类**，如矩阵分解或自适应kNN，这些模型类负责估计 $\hat{x}_{uij}(\Theta)$。因此，在统计上对个性化全序 $>_u$ 进行建模变得可行。为方便起见，在以下内容中我们将从 $\hat{x}_{uij}$ 中省略参数 $\Theta$。

到目前为止，我们只讨论了似然函数。为了完成个性化排序任务的贝叶斯建模方法，我们引入了一个**通用先验密度** $p(\Theta)$，它是具有零均值和方差-协方差矩阵 $\Sigma_\Theta$ 的正态分布。

$$
p(\Theta) \sim \mathcal{N}(0, \Sigma_\Theta)
$$

在以下内容中，为了减少未知超参数的数量，我们设置 $\Sigma_\Theta = \lambda_\Theta I$。现在我们可以公式**化最大后验估计**，以推导出我们用于个性化排序的通用优化准则BPR-Opt。

$$
\begin{align}
\text{BPR-Opt} &:= \ln p(\Theta | >_u) \\
&= \ln p(>_u | \Theta) p(\Theta) \\
&= \ln \prod_{(u,i,j) \in D_S} \sigma(\hat{x}_{uij}) p(\Theta) \\
&= \sum_{(u,i,j) \in D_S} \ln \sigma(\hat{x}_{uij}) + \ln p(\Theta) \\
&= \sum_{(u,i,j) \in D_S} \ln \sigma(\hat{x}_{uij}) - \lambda_\Theta \|\Theta\|^2
\end{align}
$$

其中 $\lambda_\Theta$ 是模型特定的正则化参数。

> [!NOTE]
>
>  **BPR-Opt 是最大化问题**，目标是**最大化**后验概率。因此后续优化算法是梯度上升算法



#### 4.1.1 与AUC优化的类比

通过这种贝叶斯个性化排序（BPR）方案的形式化，现在很容易理解BPR与AUC之间的类比。每个用户的AUC通常定义为：

$$
\text{AUC}(u) := \frac{1}{|I^{+}_u| |I \setminus I^{+}_u|} \sum_{i \in I^{+}_u} \sum_{j \in I \setminus I^{+}_u} \delta(\hat{x}_{uij} > 0)
$$

因此平均AUC为：

$$
\text{AUC} := \frac{1}{|U|} \sum_{u \in U} \text{AUC}(u)
$$

用我们的 $D_S$ 表示法，这可以写为：

$$
\text{AUC}(u) = \sum_{(u,i,j) \in D_S} z_u \delta(\hat{x}_{uij} > 0) \qquad (1)
$$

其中 $z_u$ 是归一化常数：

$$
z_u = \frac{1}{|U| |I^{+}_u| |I \setminus I^{+}_u|}
$$

(1)和BPR-Opt之间的类比是明显的。除了归一化常数 $z_u$ 之外，它们仅在损失函数上有所不同。AUC使用**不可微的损失** $\delta(x > 0)$，它等同于Heaviside函数：

$$
\delta(x > 0) = H(x) := \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{otherwise} \end{cases}
$$

取而代之，我们使用可微的损失 $\ln \sigma(x)$。**在优化AUC时，用可微函数替代不可微的Heaviside函数是常见的做法**[3]。替代函数的选择通常是启发式的，使用类似 $\sigma$ 形状的函数（见图3）。在本文中，我们推导了替代函数 $\ln \sigma(x)$，这是由MLE（最大似然估计）驱动的。

<img src="/Users/dazhang/PycharmProject/Papers/3-RecSys/.picture/image-20260717155847901.png" alt="image-20260717155847901" style="zoom:33%;" />

### 4.2 BPR学习算法

在上一节中，我们推导了一个用于个性化排序的优化准则。由于该准则是可微的，基于梯度下降的算法是最大化的明显选择。但正如我们将看到的，**标准梯度下降不是我们问题的正确选择**。为了解决这个问题，我们提出了LearnBPR，一种基于训练三元组自助采样的随机梯度下降算法（见图4）。

<img src="/Users/dazhang/PycharmProject/Papers/3-RecSys/.picture/image-20260717155906334.png" alt="image-20260717155906334" style="zoom:33%;" />

首先，BPR-Opt相对于模型参数的梯度为：

$$
\frac{\partial \text{BPR-Opt}}{\partial \Theta} = \sum_{(u,i,j) \in D_S} \frac{\partial}{\partial \Theta} \ln \sigma(\hat{x}_{uij}) - \lambda_\Theta \frac{\partial}{\partial \Theta} \|\Theta\|^2 \propto \sum_{(u,i,j) \in D_S} \frac{-e^{-\hat{x}_{uij}}}{1 + e^{-\hat{x}_{uij}}} \cdot \frac{\partial}{\partial \Theta} \hat{x}_{uij} - \lambda_\Theta \Theta
$$

> [!NOTE]
>
> $\frac{e^{-\hat{x}_{uij}}}{1+e^{-\hat{x}_{uij}}} = \sigma(-\hat{x}_{uij}) = 1 - \sigma(\hat{x}_{uij})$ 是 BPR 损失的梯度系数
>
> $\frac{\partial}{\partial \Theta} \ln{\sigma(\hat{x}_{uij})} = \frac{1}{\sigma(\hat{x}_{uij})}\cdot \sigma(\hat{x}_{uij}) \cdot (1 - \sigma(\hat{x}_{uij})) \cdot \frac{\partial}{\partial \Theta} \hat{x}_{uij}$


两种最常见的梯度下降算法是 批量梯度下降 和 随机梯度下降。在第一种情况下，每一步计算所有训练数据的完整梯度，然后用学习率 $\alpha$ 更新模型参数：
$$
\Theta \leftarrow \Theta + \alpha \frac{\partial \text{BPR-Opt}}{\partial \Theta}
$$

总的来说，这种方法会导致向"正确"方向下降，但**收敛速度慢**。由于我们在 $D_S$ 中有 $O(|S| |I|)$ 个训练三元组，**在每次更新步骤中计算完整梯度是不可行的**。

此外，对于使用批量梯度下降优化BPR-Opt，**训练对中的偏斜也导致较差的收敛性**。想象一个经常为正的item $i$。那么损失中会有许多 $\hat{x}_{uij}$ 形式的项，因为对许多用户 $u$ 来说，item $i$ 与所有负item $j$（主导类）进行比较。因此，依赖于 $i$ 的模型参数的梯度将极大地主导梯度。这意味着必须选择非常小的学习率。其次，由于梯度差异很大，正则化也很困难。

> [!NOTE]
>
>没有理解

另一种流行的方法是**随机梯度下降。**在这种情况下，对**每个**三元组 $(u, i, j) \in D_S$ 执行一次更新。

$$
\Theta \leftarrow \Theta + \alpha \left( \frac{e^{-\hat{x}_{uij}}}{1 + e^{-\hat{x}_{uij}}} \cdot \frac{\partial}{\partial \Theta} \hat{x}_{uij} + \lambda_\Theta \Theta \right)
$$

总的来说，这对于我们的偏斜问题是一个好的方法，但**训练对的遍历顺序至关重要**。按item或按用户遍历数据的典型方法**会导致较差的收敛性**，因为**在同一个用户-item对上有太多连续的更新**——即对于一个用户-item对 $(u, i)$，有许多 $j$ 满足 $(u, i, j) \in D_S$。

为了解决这个问题，我们建议使用一种随机梯度下降算法，**随机（均匀分布）选择三元组**。使用这种方法，**在连续的更新步骤中选择相同用户-item组合的可能性很小**。我们建议使用**带放回的自助采样方法**，因为可以在任何步骤停止。**放弃完整遍历数据的想法在我们的情况下特别有用，因为样本数量非常大，并且对于收敛来说，完整遍历的一小部分通常就足够了**。在我们的评估中，我们根据观测到的正反馈S的数量线性地选择单步的数量。

> [!NOTE]
>
> 详见-笔记

图5显示了典型的按用户随机梯度下降与我们带自助采样的LearnBPR方法的比较¹。模型是16维的BPR-MF。正如您所见，LearnBPR比按用户梯度下降收敛得快得多。

<img src="/Users/dazhang/PycharmProject/Papers/3-RecSys/.picture/image-20260717155944245.png" alt="image-20260717155944245" style="zoom:33%;" />

### 4.3 使用BPR学习模型

在下文中，我们描述了两类最先进的item推荐模型，以及如何使用我们提出的BPR方法学习它们。我们选择了矩阵分解[5, 12]和学习型k近邻[8]这两个不同的模型类。这两类模型都试图对用户在item上的隐藏偏好进行建模。它们预测每个用户-item对 $(u, l)$ 的实数 $\hat{x}_{ul}$。

由于在我们的优化中有三元组 $(u, i, j) \in D_S$，我们首先分解估计量 $\hat{x}_{uij}$ 并将其定义为：

$$
\hat{x}_{uij} := \hat{x}_{ui} - \hat{x}_{uj}
$$

现在我们可以**应用任何预测 $\hat{x}_{ul}$ 的标准协同过滤模型**。

重要的是要注意，**尽管我们使用了与其他工作相同的模型，但我们针对不同的准则优化它们**。这将导致更好的排序，因为我们的准则对排序任务是最优的。我们的准则不是尝试将单个预测器 $\hat{x}_{ul}$ 回归到单个数值，而是尝试对两个预测 $\hat{x}_{ui} - \hat{x}_{uj}$ 的差异进行分类。

#### 4.3.1 矩阵分解

预测 $\hat{x}_{ui}$ 的问题可以看作是估计矩阵 $X : U \times I$ 的任务。通过矩阵分解，目标矩阵 $X$ 被两个低秩矩阵 $W : |U| \times k$ 和 $H : |I| \times k$ 的矩阵乘积近似：

$$
\hat{X} := W H^t
$$

其中 $k$ 是近似的维度/秩。$W$ 中的每一行 $w_u$ 可以被看作描述用户 $u$ 的特征向量，类似地，$H$ 中的每一行 $h_i$ 描述 item $i$。因此预测公式也可以写为：

$$
\hat{x}_{ui} = \langle w_u, h_i \rangle = \sum_{f=1}^{k} w_{uf} \cdot h_{if}
$$

除了点积 $\langle \cdot, \cdot \rangle$ 之外，**通常任何核函数都可以使用**，如[11]中所示。矩阵分解的模型参数是 $\Theta = (W, H)$。模型参数也可以被视为潜在变量，对用户未观测到的品味和item未观测到的属性进行建模。

一般来说，$\hat{X}$ 对 $X$ 在最小二乘意义上的最佳近似是通过奇异值分解（SVD）实现的。对于机器学习任务，已知SVD会过拟合，因此许多其他矩阵分解方法被提出，包括正则化最小二乘优化、非负分解、最大间隔分解等。

对于排序任务，即估计用户是否偏好一个item胜过另一个item，更好的方法是针对BPR-Opt准则进行优化。这可以通过使用我们提出的算法LearnBPR来实现。如前所述，对于使用LearnBPR进行优化，只需要知道 $\hat{x}_{uij}$ 相对于每个模型参数 $\theta$ 的梯度。对于矩阵分解模型，导数为：

$$
\frac{\partial}{\partial \theta} \hat{x}_{uij} =
\begin{cases}
h_{if} - h_{jf}, & \text{if } \theta = w_{uf},\\
w_{uf}, & \text{if } \theta = h_{if},\\
-w_{uf}, & \text{if } \theta = h_{jf},\\
0, & \text{otherwise}
\end{cases}
$$

此外，我们使用三个正则化常数：一个 $\lambda_W$ 用于用户特征 $W$；对于item特征 $H$，我们有两个正则化常数，$\lambda_{H+}$ 用于对 $h_{if}$ 的正更新，$\lambda_{H-}$ 用于对 $h_{jf}$ 的负更新。

> [!NOTE]
>
> 详见笔记



#### 4.3.2 自适应k近邻

**近邻方法在协同过滤中非常流行**。它们依赖于item之间（基于item）或用户之间（基于用户）的**相似度度量**。在下文中我们描述基于item的方法，因为它们通常提供更好的结果，但基于用户的方法类似地工作。

其思想是，**对用户 $u$ 和 item $i$ 的预测取决于 $i$ 与用户过去看到的所有其他item——即 $I^{+}_u$——的相似度**。通常只考虑 $I^{+}_u$ 中最相似的 $k$ 个item——k近邻。如果item之间的相似度选择得当，也可以与 $I^{+}_u$ 中的所有item进行比较。对于item预测，基于item的k近邻模型是：
$$
\hat{x}_{ui} = \sum_{l \in I^{+}_u, l \neq i} c_{il}
$$

其中 $C : I \times I$ 是对称的item-关联/item-相似度矩阵。因此**kNN的模型参数是 $\Theta = C$。**

选择 $C$ 的常用方法是应用**启发式相似度度量**，例如余弦向量相似度：

$$
c^{\text{cosine}}_{i,j} := \frac{|U^{+}_i \cap U^{+}_j|}{\sqrt{|U^{+}_i| \cdot |U^{+}_j|}}
$$

**更好的策略是通过学习使相似度度量 $C$ 适应问题**。这可以通过直接使用 $C$ 作为模型参数来实现，或者如果item数量太大，可以学习 $C$ 的分解 $HH^t$，其中 $H : I \times k$。**在下文中以及在我们的评估中，我们使用直接学习 $C$ 而不进行分解的第一种方法**。

**再次强调，为了优化kNN模型的排序，我们应用BPR优化准则并使用LearnBPR算法**。为了应用该算法，$\hat{x}_{uij}$ 相对于模型参数 $C$ 的梯度为：
$$
\frac{\partial}{\partial \theta} \hat{x}_{uij} =
\begin{cases}
+1, & \text{if } \theta \in \{c_{il}, c_{li}\} \land l \in I^{+}_u \land l \neq i,\\
-1, & \text{if } \theta \in \{c_{jl}, c_{lj}\} \land l \in I^{+}_u \land l \neq j,\\
0, & \text{else }
\end{cases}
$$

我们有两个正则化常数，$\lambda_+$ 用于对 $c_{il}$ 的更新，$\lambda_-$ 用于对 $c_{jl}$ 的更新。

> [!NOTE]
>
> 详见笔记



## 5 与其他方法的关系

我们讨论我们提出的排序方法与其他两种item预测模型的关系。

### 5.1 加权正则化矩阵分解（WR-MF）

Pan等人[10] 和 Hu等人[5]都提出了一种**从隐式反馈进行item预测的矩阵分解方法**。因此模型类与我们第4.3.1节描述的相同，即 $\hat{X} := W H^t$，矩阵 $W : |U| \times k$ 和 $H : |I| \times k$。优化准则和学习方法与我们的方法有本质区别。他们的方法是对SVD的改编，最小化平方损失。他们的扩展包括**防止过拟合的正则化 和 误差函数中的权重 以增加正反馈的影响**。他们的**总优化准则**是：

$$
\sum_{u \in U} \sum_{i \in I} c_{ui}(\langle w_u, h_i \rangle - 1)^2 + \lambda \|W\|_F^2 + \lambda \|H\|_F^2
$$

其中 $c_{ui}$ 不是模型参数，而是每个元组 $(u, i)$ 的**先验给定权重**。Hu等人有额外的数据来估计正反馈的 $c_{ui}$，并对其余部分设置 $c_{ui} = 1$。Pan等人建议对正反馈设置 $c_{ui} = 1$，并对其余部分选择较低的常数。

> [!NOTE]
>
> 对于正样本：$t_{ui} = 1$ \rightarrow 误差项为 $(\hat{x}_{ui} - 1)^2$ 。-1 就是告诉模型："对于用户交互过的 item，请把预测分数往 1 靠拢"。
>
> 对于负样本：$t_{ui} = 0$ \rightarrow 误差项为 $(\hat{x}_{ui} - 0)^2 = \hat{x}_{ui}^2$ 
>
> WR-MF公式中只体现了正样本，负样本是从为交互item中随机采样的

首先，很明显这个优化是在**实例级别**（一个item）而不是像BPR那样的**对级别**（两个item）。除此之外，他们的优化是最小二乘法，已知它对应于**正态分布随机变量的MLE**。然而，**item预测任务实际上不是回归（定量）问题，而是分类（定性）问题**，因此logistic优化更合适。

> [!NOTE]
>
> Item预测任务不是回归（定量）问题，而是分类（定性）问题。
>
> 定量问题学习难度更大，因此效果更差。

WR-MF的一个强点是它可以 $O(\text{iter} (|S| k^2 + k^3 (|I| + |U|)))$ 的时间复杂度学习，前提是 $c_{ui}$ 对非正对是常数。我们的评估表明，尽管要学习的三元组多得多，LearnBPR通常能在 $m \cdot |S|$ 个**单步更新子样本后收敛**。

### 5.2 最大间隔矩阵分解（MMMF）

Weimer等人[15]使用最大间隔矩阵分解（MMMF）方法进行**序数排序**。他们的MMMF是为**具有评分形式显式反馈的场景设计的**。尽管他们的排序MMMF并非为隐式反馈数据集设计，但可以通过将所**有未观测item赋予"评分"0，观测item赋予1**（见图1）来将其应用于我们的场景。通过这些修改，他们需要**最小化的优化准则**将与应用于矩阵分解的BPR非常相似：

$$
\sum_{(u,i,j) \in D_s} \max(0, 1 - \langle w_u, h_i - h_j \rangle) + \lambda_w \|W\|_F^2 + \lambda_h \|H\|_F^2
$$

一个区别是误差函数不同——我们的合页损失hinge loss是平滑的，并由MLE驱动。此外，我们的BPR-Opt准则是通用的，可以应用于多种模型，而他们的方法是特定于MF的。

> [!NOTE]
>
> 这里 1 是合页损失（hinge loss）的边界参数，即期望间隔（margin）
>
> 当 $\hat{x}_{ui} - \hat{x}_{uj} \geq 1$ 时：正样本分数已经比负样本高出至少 1，模型对此样本对满意，损失为 0
>
> 当 $\hat{x}_{ui} - \hat{x}_{uj} < 1$ 时：间隔不足，损失为 $1 - (\hat{x}_{ui} - \hat{x}_{uj})$，梯度会推动正样本分数更高 或 负样本分数更低

除此之外，他们的MMMF学习方法与我们通用的LearnBPR方法不同。他们的学习方法被设计用于处理稀疏显式数据，即他们假设有许多缺失值，因此假设比隐式设置中少得多的对。**但当他们的学习方法应用于隐式反馈数据集时，数据必须像上面描述的那样稠密化**，训练对 $D_S$ 的数量为 $O(|S| |I|)$。我们的方法LearnBPR可以通过从 $D_S$ 自助采样来处理这种情况（见第4.2节）。



## 6 评估

在我们的评估中，我们将使用BPR的学习与其他学习方法进行比较。我们选择了矩阵分解（MF）和k近邻（kNN）这两个流行的模型类。已知MF模型[12]在**协同评分预测**的相关任务上优于许多其他模型，包括贝叶斯模型URP[9]和PLSA[4]。在我们的评估中，矩阵分解模型通过三种不同的方法学习，即SVD-MF、WR-MF[5, 10]和我们的BPR-MF。对于kNN，我们比较了余弦向量相似度（Cosine-kNN）和使用我们的BPR方法优化的模型（BPR-kNN）。此外，我们报告了基准方法most-popular的结果，该方法独立于用户地加权每个item，例如：$\hat{x}^{\text{most-pop}}_{ui} := |U^{+}_i|$。此外，我们给出了任何非个性化排序方法在AUC上的理论上界（npmax）。

### 6.1 数据集

我们使用了两个来自不同应用的数据集。

Rossmann数据集来自一个在线商店。它包含10,000个用户在4000个item上的购买历史。总共记录了426,612次购买。**任务是预测用户想要下一次购买的个性化item列表**。

第二个数据集是Netflix的DVD租赁数据集。该数据集包含用户的评分行为，用户对某些电影提供1到5星的显式评分。由于我们希望解决隐式反馈任务，我们**从数据集中移除了评分分数**。现在任务是**预测用户是否可能对一部电影进行评分**。我们再次对从最可能被评分的电影开始的个性化排序列表感兴趣。对于Netflix，我们创建了一个包含10,000个用户、5000个item、565,738次评分行为的子样本。我们抽取子样本的方式使得**每个用户至少有10个item**（$\forall u \in U : |I^{+}_u| \geq 10$）且**每个item至少有10个用户**（$\forall i \in I : |U^{+}_i| \geq 10$）。

> [!NOTE]
>
> Netflix的DVD租赁数据集和MovieLens的电影评分数据集很相似。并且子样本的处理逻辑也是类似的。值得借鉴！



### 6.2 评估方法

我们使用 **留一法** 评估方案，其中对每个用户，我们从其历史中随机移除一个行为（一个用户-item对），即我们对每个用户$u$ 从$I^{+}_u$中移除一个Item。这导致一个不相交的训练集$S_{train}$ 和 测试集$S_{test}$。然后在 $S_{train}$ 上学习模型，并通过平均AUC统计量在测试集 $S_{test}$ 上评估它们预测的个性化排序：

$$
\text{AUC} = \frac{1}{|U|} \sum_u \frac{1}{|E(u)|} \sum_{(i,j) \in E(u)} \delta(\hat{x}_{ui} > \hat{x}_{uj}) \qquad (2)
$$

其中每个用户 $u$ 的评估对为：

$$
E(u) := \{(i, j) \mid (u, i) \in S_{\text{test}} \land (u, j) \notin (S_{\text{test}} \cup S_{\text{train}})\}
$$

较高的AUC值表示较好的质量。随机猜测方法的平凡AUC为0.5，可达到的最佳质量为1。

我们重复所有实验10次，每次轮次抽取新的训练/测试划分。所有方法的超参数在第一轮中通过网格搜索进行优化，之后在其余9次重复中保持不变。

> [!NOTE] 
>
> 值得借鉴！
>
> 训练集：从每个用户的历史行为中随机移除一个交互后剩余的数据
>
> 测试集：被移除的那个用户-item 对，每个用户恰好一个
>
> $i$ 是正样本（测试集中的 item，是用户真正交互过的 item），$j$ 是从 $S_{train}$ 和 $S_{test}$ 之外的 item 中采样的负样本（用户从未交互过的item）。设计目的：评估模型能否将正样本（用户喜欢的）排到负样本（用户未交互的）前面



### 6.3 结果与讨论

<img src="/Users/dazhang/PycharmProject/Papers/3-RecSys/.picture/image-20260717233710418.png" alt="image-20260717233710418" style="zoom: 33%;" />

图6显示了所有模型在两个数据集上的AUC质量。首先，可以看到两个BPR优化的方法在预测质量上优于所有其他方法。比较相同模型之间的差异，可以看到优化方法的重要性。例如，所有MF方法（SVD-MF、WR-MF和BPR-MF）**共享完全相同的模型**，但它们的预测质量差异很大。尽管SVD-MF已知能在训练数据上产生**关于逐元素最小二乘的最佳拟合**，但它是机器学习任务的糟糕预测方法，因为它会导致过拟合。**这可以从SVD-MF的质量随着维度数量的增加而下降看出**。WR-MF是一种更成功的排序任务学习方法。**由于正则化，它的性能不会下降，而是随着维度数量的增加稳步上升**。但在两个数据集上，BPR-MF在排序任务上明显优于WR-MF。例如，在Netflix上，使用BPR-MF优化的8维MF模型达到了与使用WR-MF优化的128维MF模型相当的质量。

总之，我们的结果显示了**为正确的准则优化模型参数的重要性**。实证结果表明，我们的通过LearnBPR学习的BPR-Opt准则优于其他用于隐式反馈个性化排序的最先进方法。这些结果通过对问题的分析（第3.2节）和从MLE对BPR-Opt的理论推导得到了验证。

### 6.4 非个性化排序

最后，我们将个性化排序方法的AUC质量与**最佳可能的非个性化排序方法**进行比较。与我们的个性化排序方法相反，非个性化排序方法**为所有用户创建相同的排序** $>$。我们通过在测试集 $S_{test}$ 上优化排序 $>$ 来计算任何非个性化排序方法的理论上界 $np_{max}^2$。图6显示，即使是像Cosine-kNN这样简单的个性化方法也大大优于上界 $np_{max}$——因此也**优于所有非个性化方法**。



## 7 结论

在本文中，我们提出了一个通用的优化准则和学习算法用于个性化排序。优化准则BPR-Opt是从问题的贝叶斯分析推导出的最大后验估计。为了根据BPR-Opt学习模型，我们提出了**通用的学习算法LearnBPR**，它**基于带自助采样的随机梯度下降**。我们展示了如何将这个通用方法应用于两类最先进的推荐模型：矩阵分解和自适应kNN。在我们的评估中，我们经验性地表明，对于个性化排序任务，通过BPR学习的模型优于针对其他准则优化的相同模型。我们的结果表明，预测质量不仅取决于模型，还在**很大程度上取决于优化准则**。我们的理论和实证结果都表明，BPR优化方法对于个性化排序这一重要任务是正确的选择。



## 致谢

作者们感谢他们的工作得到了欧盟FP7项目MyMedia（www.mymediaproject.org）的部分联合资助，资助协议ID为215006。如有疑问，请联系info@mymediaproject.org。



## 参考文献

[1] C. Burges, T. Shaked, E. Renshaw, A. Lazier, M. Deeds, N. Hamilton, and G. Hullender. Learning to rank using gradient descent. In *ICML '05: Proceedings of the 22nd international conference on Machine learning*, pages 89–96, New York, NY, USA, 2005. ACM Press.

[2] M. Deshpande and G. Karypis. Item-based top-n recommendation algorithms. *ACM Transactions on Information Systems*. Springer-Verlag, 22/1, 2004.

[3] A. Herschtal and B. Raskutti. Optimising area under the roc curve using gradient descent. In *ICML '04: Proceedings of the twenty-first international conference on Machine learning*, page 49, New York, NY, USA, 2004. ACM.

[4] T. Hofmann. Latent semantic models for collaborative filtering. *ACM Trans. Inf. Syst.*, 22(1):89–115, 2004.

[5] Y. Hu, Y. Koren, and C. Volinsky. Collaborative filtering for implicit feedback datasets. In *IEEE International Conference on Data Mining (ICDM 2008)*, pages 263–272, 2008.

[6] J. Huang, C. Guestrin, and L. Guibas. Efficient inference for distributions on permutations. In J. Platt, D. Koller, Y. Singer, and S. Roweis, editors, *Advances in Neural Information Processing Systems 20*, pages 697–704, Cambridge, MA, 2008. MIT Press.

[7] R. Kondor, A. Howard, and T. Jebara. Multi-object tracking with representations of the symmetric group. In *Proceedings of the Eleventh International Conference on Artificial Intelligence and Statistics*, San Juan, Puerto Rico, March 2007.

[8] Y. Koren. Factorization meets the neighborhood: a multifaceted collaborative filtering model. In *KDD '08: Proceeding of the 14th ACM SIGKDD international conference on Knowledge discovery and data mining*, pages 426–434, New York, NY, USA, 2008. ACM.

[9] B. Marlin. Modeling user rating profiles for collaborative filtering. In S. Thrun, L. Saul, and B. Schölkopf, editors, *Advances in Neural Information Processing Systems 16*, Cambridge, MA, 2004. MIT Press.

[10] R. Pan, Y. Zhou, B. Cao, N. N. Liu, R. M. Lukose, M. Scholz, and Q. Yang. One-class collaborative filtering. In *IEEE International Conference on Data Mining (ICDM 2008)*, pages 502–511, 2008.

[11] S. Rendle and L. Schmidt-Thieme. Online-updating regularized kernel matrix factorization models for large-scale recommender systems. In *RecSys '08: Proceedings of the 2008 ACM conference on Recommender systems*. ACM, 2008.

[12] J. D. M. Rennie and N. Srebro. Fast maximum margin matrix factorization for collaborative prediction. In *ICML '05: Proceedings of the 22nd international conference on Machine learning*, pages 713–719, New York, NY, USA, 2005. ACM.

[13] B. Sarwar, G. Karypis, J. Konstan, and J. Riedl. Incremental singular value decomposition algorithms for highly scalable recommender systems. In *Proceedings of the 5th International Conference in Computers and Information Technology*, 2002.

[14] L. Schmidt-Thieme. Compound classification models for recommender systems. In *IEEE International Conference on Data Mining (ICDM 2005)*, pages 378–385, 2005.

[15] M. Weimer, A. Karatzoglou, and A. Smola. Improving maximum margin matrix factorization. *Machine Learning*, 72(3):263–276, 2008.

---

**脚注：**

¹关于数据集和评估方法的详细信息可以在第6节找到。

^2我们计算了一个真实但非紧的AUC分数上界。请注意，**在测试集上按most-popular排序并不是AUC的上界**。但在我们的实验中，两个AUC得分非常相似，例如在Netflix上，测试集上的most-popular为0.8794，而我们的上界为0.8801。

> [!NOTE]
>
> 疑问：非个性化排序的AUC上界是如何计算出的？

