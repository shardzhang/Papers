# NCF：Neural Collaborative Filtering

> Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu, Tat-Seng Chua | National University of Singapore, Shandong University, Columbia University, Texas A&M University, WWW 2017

---



本文分享了NCF（基于神经网络的协同过滤）框架，该框架用 **神经网络架构** 替代传统的 **矩阵分解内积操作**，通过多层感知机学习用户与item之间的交互函数。

**核心内容：**

- 提出通用NCF框架，用神经网络参数化交互函数，支持逐点和成对学习
- 证明矩阵分解（MF）是NCF的特例，并提出GMF（广义矩阵分解）
- 提出MLP（多层感知机）模型学习用户-item交互的非线性
- 融合GMF和MLP得到NeuMF（神经矩阵分解），兼具线性和非线性建模能力
- **将隐式反馈推荐建模为二分类问题，使用对数损失（log loss）优化**

**关键发现：**

- NeuMF显著优于eALS和BPR等现有方法，相对提升约4.5%-4.9%
- **对数损失+负采样策略优于BPR的成对排序损失**
- 更深层的网络结构有助于提升推荐性能
- 预训练策略对NeuMF的收敛和性能有积极作用

---





## 摘要

近年来，深度神经网络在语音识别、计算机视觉和自然语言处理领域取得了巨大成功。然而，深度神经网络在推荐系统上的探索相对较少。在本文中，我们致力于开发基于神经网络的技术，以解决推荐中的关键问题——基于隐式反馈的协同过滤。

尽管近期一些工作已将深度学习用于推荐，但它们主要将深度学习用于建模辅助信息，例如item的文本描述和音乐的声学特征。在建模协同过滤的关键因素——用户特征和item特征之间的交互时，这些工作仍然采用矩阵分解，对用户和item的latent特征应用内积。

通过用能够从数据中学习任意函数的神经架构替换内积，我们提出了一个名为NCF（Neural network-based Collaborative Filtering的缩写）的通用框架。NCF具有通用性，可以在其框架下表达和推广矩阵分解。为了给NCF建模注入非线性能力，我们提出利用多层感知机来学习用户-item交互函数。在两个**真实世界数据集**上的大量实验表明，我们提出的NCF框架显著优于现有方法。实验证据表明，使用更深层的神经网络可以获得更好的推荐性能。

## 关键词

协同过滤，神经网络，深度学习，矩阵分解，隐式反馈



## 1 引言

在信息爆炸的时代，推荐系统在缓解信息过载方面发挥着关键作用，已被电子商务、在线新闻和社交媒体等众多在线服务广泛采用。个性化推荐系统的关键在于**根据用户过去的行为（如评分和点击）建模用户对item的偏好，这被称为协同过滤**[31, 46]。在众多协同过滤技术中，矩阵分解（MF）[14, 21]是最流行的方法，它将用户和item投影到共享的latent空间中，用latent特征向量表示用户或item，然后**将用户对item的交互建模为它们latent向量的内积**。

因Netflix Prize而普及的MF已成为基于latent因子模型的推荐的事实标准方法。大量研究工作致力于增强MF，例如将其与基于邻居的模型[21]结合（如SVD++）、与item内容的主题模型[38]结合，以及将其扩展为因子分解机[26]以**实现特征的通用建模**。这些扩展虽然在一定程度上提升了MF的性能，但它们**仍然基于内积作为核心交互函数，没有从根本上改变交互函数的表达方式**。尽管MF在协同过滤中很有效，但众所周知，其性能可能受到交互函数（即内积）的简单选择的限制。例如，对于显式反馈的评分预测任务，众所周知，通过在交互函数中引入 **用户和item偏置项** 可以提升MF模型的性能。虽然这看起来只是对内积运算符的一个微小调整[14]，但它表明了设计更好、更专用的交互函数来建模用户和item之间latent特征交互的积极作用。**内积只是线性地组合latent特征的乘积，可能不足以捕捉用户交互数据的复杂结构。**

本文探索使用深度神经网络从数据中学习交互函数，而非像许多先前工作[18, 21]那样**手工设计交互函数**。神经网络已被证明能够逼近任何连续函数[17]，并且最近深度神经网络（DNN）在计算机视觉、语音识别到文本处理等多个领域展现了有效性[5, 10, 15, 47]。然而，与MF方法的大量文献相比，将DNN用于推荐的工作相对较少。尽管近期的一些进展[37, 38, 45]已将DNN应用于推荐任务并显示出有希望的结果，但它们大多使用DNN来建模辅助信息，如item的文本描述、音乐的音频特征和图像的视觉内容。在建模关键的协同过滤效应方面，这些工作仍然采用MF，使用内积组合用户和item的latent特征。

本文通过形式化一种用于协同过滤的神经网络建模方法来解决上述研究问题。我们专注于**隐式反馈**，它通过观看视频、购买商品和点击item等行为间接反映用户的偏好。与显式反馈（即评分和评论）相比，隐式反馈可以自动跟踪，因此内容提供商更容易收集。然而，利用隐式反馈更具挑战性，因为用户满意度无法直接观察，并且天然缺乏负反馈。在本文中，我们探讨的核心主题是如何利用DNN来建模带有噪声的隐式反馈信号。

本文的主要贡献如下：

1. 我们提出了一种用于建模用户和itemlatent特征的神经网络架构，并设计了一个基于神经网络的通用协同过滤框架NCF。

2. 我们表明MF可以解释为NCF的一种特化，并利用多层感知机为NCF建模注入高度的非线性。

3. 我们在两个真实世界数据集上进行了大量实验，以证明我们的NCF方法的有效性以及深度学习在协同过滤中的前景。



## 2 预备知识

我们首先形式化问题，并讨论现有的用于隐式反馈协同过滤的解决方案。然后我们简要回顾广泛使用的MF模型，强调其由于使用内积而带来的局限性。

### 2.1 从隐式数据中学习

设 M 和 N 分别表示用户数和item数。我们从用户的隐式反馈定义用户-item交互矩阵 $Y \in \mathbb{R}^{M \times N}$ 为：

$$
y_{ui} = 1，\text{if interaction (user } u，\text{item } i \text{) is observed;} 0，\text{otherwise.}\qquad (1)
$$

这里，$y_{ui}$ 值为 1 表示用户 $u$ 和item $i$ 之间存在交互；但是，这并不意味着 $u$ 真正喜欢 $i$。同样，值为 0 也不一定意味着 $u$ 不喜欢 $i$，可能是用户不知道该item。这给从隐式数据中学习带来了挑战，因为它只**提供了关于用户偏好的噪声信号**。虽然**被观察到的条目至少反映了用户对item的兴趣，但未被观察到的条目可能只是缺失数据，并且天然缺乏负反馈。**

隐式反馈下的推荐问题被形式化为**估计 $Y$ 中未观察条目的分数**，这些分数用于对item进行排序。**与显式反馈中的评分预测任务不同，隐式反馈推荐的目标是生成一个排序的item列表，而不仅仅是预测用户会给item打多少分**。因此，推荐系统的性能通常通过排序指标（如HR和NDCG）来评估。

基于模型的方法假设数据可以由底层模型生成（或描述）。形式上，这些方法可以抽象为学习 $\hat{y}_{ui} = f(u, i|\Theta)$，其中 $\hat{y}_{ui}$ 表示交互 $y_{ui}$ 的预测分数，$\Theta$ 表示模型参数，$f$ 表示**将模型参数映射到预测分数的函数**（我们称之为交互函数）。

为了估计参数 $\Theta$，现有方法通常遵循优化目标函数的机器学习范式。文献中最常用的两类目标函数是 pointwise 损失[14, 19]和 pairwise 损失[27, 33]。**逐点损失为每个训练实例单独计算预测误差，而成对损失则关注正样本和负样本之间的相对排序关系**。作为丰富显式反馈工作[21, 46]的自然扩展，逐点学习方法通常遵循回归框架，通过最小化 $\hat{y}_{ui}$ 与其目标值 $y_{ui}$ 之间的平方损失。**为了处理负数据缺失的问题，它们要么将所有未观察条目视为负反馈，要么从未观察条目中采样负实例[14]**。对于成对学习[27, 44]，其**核心思想是观察到的条目应该比未观察到的条目排名更高**。因此，成对学习方法不是最小化 $\hat{y}_{ui}$ 和 $y_{ui}$ 之间的损失，而是**最大化观察条目 $\hat{y}_{ui}$ 和未观察条目 $\hat{y}_{uj}$ 之间的间隔**。BPR（贝叶斯个性化排序）是成对学习方法中最具代表性的工作，它通过优化成对排序损失来学习模型参数。

更进一步，我们的NCF框架使用神经网络参数化交互函数 $f$ 来估计 $\hat{y}_{ui}$，因此它天然支持逐点和成对学习这两种方式。在本文中，我们主要关注逐点学习方式，采用对数损失作为目标函数，并在实验部分详细评估其效果。

### 2.2 矩阵分解

MF为每个用户和item关联一个实值latent特征向量。设 $\mathbf{p}_u$ 和 $\mathbf{q}_i$ 分别表示用户 $u$ 和item $i$ 的latent向量；MF将交互 $y_{ui}$ 估计为 $\mathbf{p}_u$ 和 $\mathbf{q}_i$ 的内积：

$$
\hat{y}_{ui} = f(u, i|\mathbf{p}_u, \mathbf{q}_i) = \mathbf{p}_u^T \mathbf{q}_i = \sum_{k=1}^{K} p_{uk} q_{ik} \quad (2)
$$

其中 K 表示latent空间的维度。我们可以看到，MF对用户和itemlatent因子的双向交互进行建模，假设latent空间的每个维度相互独立并以相同权重线性组合。因此，MF可被视为latent因子的线性模型。

<img src="/Users/dazhang/PycharmProject/Papers/3-RecSys/.picture/image-20260715204303341.png" alt="image-20260715204303341" style="zoom:50%;" />

> **图1：一个说明MF局限性的示例。** 从数据矩阵(a)中，$u_4$ 与 $u_1$ 最相似，其次是 $u_3$，最后是 $u_2$。然而在latent空间(b)中，将 p_4 放置在离 p_1 最近的位置会使 p_4 比 p_3 更接近 p_2，导致较大的排序损失。



图1说明了内积函数如何限制MF的表达能力。为了理解这个例子，需要事先明确两个设置。首先，由于MF将用户和item映射到相同的latent空间，两个用户之间的相似度也可以用内积（或等价地，它们latent向量夹角的余弦）来衡量。其次，不失一般性，我们使用Jaccard系数作为MF需要恢复的两个用户之间的真实相似度。

设 $R_u$ 为用户 $u$ 交互过的item集合，则用户 $i$ 和 $j$ 之间的Jaccard相似度定义为 $s_{ij} = |R_i \cap R_j| / |R_i \cup R_j|$。

我们先关注图1a中的前三行（用户）。从交互矩阵可以看出，用户 $u_1$ 交互了item $\{i_1, i_2, i_3\}$，用户 $u_2$ 交互了item $\{i_1, i_4\}$，用户 $u_3$ 交互了item $\{i_2, i_4, i_5\}$。计算Jaccard相似度可得 $s_{23}(0.66) > s_{12}(0.5) > s_{13}(0.4)$。因此，$p_1$、$p_2$ 和 $p_3$ 在latent空间中的几何关系可以如图1b所示。现在考虑一个新用户 $u_4$，其输入由图1a中的虚线给出。我们有 $s_{41}(0.6) > s_{43}(0.4) > s_{42}(0.2)$，这意味着 $u_4$ 与 $u_1$ 最相似，其次是 $u_3$，最后是 $u_2$。然而，如果MF模型将 $p_4$ 放置在离 $p_1$ 最近的位置（两种选项如图1b虚线所示），将导致 $p_4$ 比 $p_3$ 更接近 $p_2$，这不幸地会造成很大的排序损失。

上述例子说明了MF由于使用简单且固定的内积来估计低维latent空间中的复杂用户-item交互而产生的局限性。我们注意到，解决该问题的一种方法是使用大量latent因子 $K$。然而，这可能**对模型的泛化产生不利影响（例如过拟合数据），尤其是在稀疏设置下**[26]。在本文中，我们通过使用DNN从数据中学习交互函数来解决这一局限性。



## 3 神经协同过滤

我们首先介绍通用的NCF框架，详细说明如何使用 **强调隐式数据二值属性的概率模型** 来学习NCF。然后我们展示MF可以在NCF框架下进行表达和推广。为了探索用于协同过滤的DNN，我们接着提出NCF的一个实例化模型，使用多层感知机（MLP）来学习用户-item交互函数。最后，我们提出一个新的神经矩阵分解模型，它在NCF框架下集成了MF和MLP，**统一了MF的线性优势和MLP的非线性优势**，用于建模用户-item latent结构。

### 3.1 通用框架

为了实现完全神经化的协同过滤处理，我们采用多层表示来建模用户-item交互 $y_{ui}$，如图2所示，其中一层的输出作为下一层的输入。最底部的输入层由 **两个特征向量** $v_u^U$ 和 $v_i^I$ 组成，分别描述用户 $u$ 和 item $i$。这两个特征向量可以被定制以支持广泛的用户和item建模，例如考虑上下文感知[28, 1]、基于内容[3]和基于邻居[26]的方法。由于本文专注于纯协同过滤设置，我们仅使用用户和item的身份作为输入特征，通过独**热编码将其转换为二值化稀疏向量**。注意，通过这种通用的输入特征表示，我们的方法可以通过使用**内容特征**来表示用户和item，从而轻松调整以解决冷启动问题。

输入层之上是嵌入层。嵌入层是一个全连接层，将稀疏表示投影到稠密向量。所获得的用户（item）嵌入可以看作latent因子模型上下文中的用户（item）latent向量。用户嵌入和item嵌入随后被输入到多层神经架构中，我们称之为**神经协同过滤层**，用于将latent向量映射到预测分数。神经CF层的每一层可以被定制以发现用户-item交互的某些latent结构。最后一个隐藏层 $X$ 的维度决定了模型的能力。最终输出层是预测分数 $\hat{y}_{ui}$，训练通过**最小化 $\hat{y}_{ui}$ 与其目标值 $y_{ui}$ 之间的逐点损失来进行**。我们注意到，训练模型的另一种方式是进行成对学习，例如使用**贝叶斯个性化排序[27]和基于间隔的损失**[33]。由于本文的重点是神经网络建模部分，我们将NCF扩展到成对学习的工作留待未来进行。

现在我们将NCF的预测模型形式化为：

$$
\hat{y}_{ui} = f(\mathbf{P}^T v_u^U, \mathbf{Q}^T v_i^I | \mathbf{P}, \mathbf{Q}, \Theta_f) \qquad (3)
$$

其中 $\mathbf{P} \in \mathbb{R}^{M \times K}$ 和 $\mathbf{Q} \in \mathbb{R}^{N \times K}$ 分别表示用户和item的latent因子矩阵；$\Theta_f$ 表示交互函数 $f$ 的模型参数。由于函数 $f$ 被定义为多层神经网络，它可以形式化为：

$$
f(\mathbf{P}^T v_u^U, \mathbf{Q}^T v_i^I) = \phi_{out}(\phi_X(...\phi_2(\phi_1(\mathbf{P}^T v_u^U, \mathbf{Q}^T v_i^I))...)) \qquad (4)
$$

其中 $\phi_{out}$ 和 $\phi_x$ 分别表示输出层和第 $x$ 个神经协同过滤（CF）层的映射函数，总共有 $X$ 个神经CF层。

<img src="/Users/dazhang/PycharmProject/Papers/3-RecSys/.picture/image-20260715204334445.png" alt="image-20260715204334445" style="zoom:50%;" />

> **图2：神经协同过滤框架**

#### 3.1.1 学习NCF

为了学习模型参数，现有的逐点方法[14, 39]主要通过平方损失的回归：

$$
L_{sqr} = \sum_{(u,i) \in Y \cup Y^-} w_{ui}(y_{ui} - \hat{y}_{ui})^2 \qquad (5)
$$

**其中 $Y$ 表示 $Y$ 中观察到的交互集合，$Y^-$ 表示负实例集合**，可以是所有（或从中采样的）未观察交互；$w_{ui}$ 是超参数，表示训练实例 $(u,i)$ 的权重。虽然平方损失可以通过假设观测值由高斯分布生成[29]来解释，但我们指出平方损失可能不适用于隐式数据。这是因为对于隐式数据，目标值 $y_{ui}$ 是二值化的 1 或 0，表示 $u$ 是否与 $i$ 有过交互。接下来，我们提出一种概率方法用于学习逐点NCF，该方法特别关注隐式数据的二值属性。

考虑到隐式反馈的单类性质，我们可以将 $y_{ui}$ 的值视为标签——1 表示item $i$ 与 $u$ 相关，0 表示不相关。预测分数 $\hat{y}_{ui}$ 则表示 $i$ 与 $u$ 相关的可能性大小。为了赋予NCF这种概率解释，我们需要将输出 $\hat{y}_{ui}$ 约束在 $[0, 1]$ 范围内，这可以通过在输出层 $\phi_{out}$ 中使用概率函数（例如Logistic或Probit函数）作为激活函数来轻松实现。通过上述设置，我们定义似然函数为：

$$
p(Y, Y^- | \mathbf{P}, \mathbf{Q}, \Theta_f) = \prod_{(u,i) \in Y} \hat{y}_{ui} \cdot \prod_{(u,j) \in Y^-} (1 - \hat{y}_{uj}) \qquad (6)
$$

对似然取负对数，我们得到：

$$
L = - \sum_{(u,i) \in Y} \log \hat{y}_{ui} - \sum_{(u,j) \in Y^-} \log(1 - \hat{y}_{uj}) = - \sum_{(u,i) \in Y \cup Y^-} y_{ui} \log \hat{y}_{ui} + (1 - y_{ui}) \log(1 - \hat{y}_{ui}) \qquad (7)
$$

> [!CAUTION]
>
> 注意：这里与CTR建模中的二元交叉熵不同，这里每次计算loss是按照<一个用户>对应一条训练样本设计的，每条样本包含了多个正样本和多个负样本。

这是NCF方法要最小化的目标函数，其优化可以通过随机梯度下降（SGD）来完成。细心的读者可能已经意识到，这个目标函数与二值交叉熵损失完全相同，也称为对数损失（log loss）。通过对NCF采用概率处理方式，我们将隐式反馈推荐问题转化为一个二分类问题，其中模型需要预测用户是否会与item发生交互。

由于分类感知的对数损失在推荐文献中很少被研究，本文对其进行了深入探索，并在第4.3节中通过大量实验展示了其有效性。对于负实例 $Y^-$，我们在每次迭代中从非观察交互中均匀采样，并根据观察交互的数量控制采样比率。这种负采样策略**确保了模型在每个训练批次中都能接触到足够的负样本，从而学习到有区分度的表示**。虽然**非均匀采样策略**（例如，基于item流行度的采样[14, 12]）可能进一步提升性能，但我们将这一探索留作未来工作。

### 3.2 广义矩阵分解（GMF）

我们现在展示MF如何被解释为NCF框架的一个特例。由于MF是最流行的推荐模型并且在文献中得到了广泛研究，能够恢复MF使得NCF可以模仿大量因子分解模型[26]。

**由于输入层用户（item）ID的独热编码，获得的嵌入向量可以视为用户（item）的latent向量**。设用户latent向量 $\mathbf{p}_u = \mathbf{P}^T v_u^U$，item latent向量 $\mathbf{q}_i = \mathbf{Q}^T v_i^I$。我们将第一个神经CF层的映射函数定义为：

$$
\phi_1(\mathbf{p}_u, \mathbf{q}_i) = \mathbf{p}_u \odot \mathbf{q}_i \qquad (8)
$$


其中 $\odot$ 表示向量的**逐元素乘积**。然后我们将该向量投影到输出层：


$$
\hat{y}_{ui} = a_{out}(\mathbf{h}^T (\mathbf{p}_u \odot \mathbf{q}_i)) \quad (9)
$$


其中 $a_{out}$ 和 $\mathbf{h}$ 分别表示输出层的激活函数和边权重。直观地说，**如果我们对 $a_{out}$ 使用恒等函数并强制 h 为全1向量，就可以精确地恢复MF模型**。

在NCF框架下，MF可以轻松地进行推广和扩展。例如，如果允许 h 从数据中学习而不加均匀约束，将会得到一个允许latent维度具有不同重要性的MF变体。如果对 $a_{out}$ 使用非线性函数，则会将MF推广到非线性设置，这可能比线性MF模型更具表达力。

在本文中，我们在NCF下实现了一个广义版本的MF，使用 sigmoid 函数 $\sigma(x) = 1/(1+e^{-x})$ 作为 $a_{out}$，并使用 **对数损失** 从数据中学习 $\mathbf{h}$（见第3.1.1节）。我们将其称为GMF（Generalized Matrix Factorization，广义矩阵分解）。

### 3.3 多层感知机（MLP）

由于NCF采用两条路径来建模用户和item，直观的做法是通过拼接来组合两条路径的特征。这种设计已在多模态深度学习工作中被广泛采用[47, 34]。然而，**简单的向量拼接没有考虑用户和item latent特征之间的任何交互，不足以建模协同过滤效应**。为了解决这个问题，我们**建议在拼接向量上添加隐藏层，使用标准的MLP来学习用户和item latent特征之间的交互**。通过这种方式，我们可以赋予模型很大的灵活性和非线性来学习 $\mathbf{p}_u$ 和 $\mathbf{q}_i$ 之间的交互，而不是像GMF那样仅对它们使用固定的逐元素乘积。更准确地说，NCF框架下的MLP模型定义如下：


$$
z_1 = \phi_1(\mathbf{p}_u, \mathbf{q}_i) = [\mathbf{p}_u; \mathbf{q}_i], \qquad (10)
$$

$$
\phi_2(z_1) = a_2(W_2^T z_1 + b_2),
$$

$$
\cdots
$$

$$
\phi_L(z_{L-1}) = a_L(W_L^T z_{L-1} + b_L),
$$

$$
\hat{y}_{ui} = \sigma(\mathbf{h}^T \phi_L(z_{L-1})),
$$


其中 $W_x$、$b_x$ 和 $a_x$ 分别表示第 $x$ 层感知机的权重矩阵、偏置向量和激活函数。对于MLP层的激活函数，可以自由选择sigmoid、双曲正切（tanh）和整流线性单元（ReLU）等。我们对每个函数进行分析：

1. sigmoid函数将每个神经元限制在(0,1)范围内，这可能限制模型性能；并且sigmoid已知存在饱和问题，当输出接近0或1时神经元停止学习。

2. 尽管tanh是一个更好的选择且已被广泛采用[6, 44]，但它只能在一定程度上缓解sigmoid的问题，因为它可以被视为sigmoid的重新缩放版本（$tanh(x/2) = 2\sigma(x)-1$）。

3. 因此，我们选择ReLU，它在生物学上更合理且已被证明是非饱和的[9]；此外，**ReLU鼓励稀疏激活**，非常适合稀疏数据并使**模型不太可能过拟合**。

**我们的实验结果表明ReLU略优于tanh，而tanh又显著优于sigmoid**。具体来说，在相同的网络结构下，使用ReLU作为激活函数的MLP模型在HR@10和NDCG@10两个指标上均取得了最佳结果。这一发现与ReLU在计算机视觉和自然语言处理领域中的成功经验是一致的。

对于网络结构的设计，常见的解决方案是采用**塔式模式**，即底层最宽，每层神经元的数量逐层递减（如图2所示）。其理论前提是，通过在上层使用少量隐藏单元，这些**上层可以学习到数据更抽象的特征**[10]。我们经验性地实现了这种塔式结构，**每层大小依次减半**。例如，当使用3个隐藏层时，典型的神经元配置为 $32 \rightarrow 16 \rightarrow 8$，即第一层32个神经元，第二层16个神经元，第三层8个神经元。

### 3.4 GMF和MLP的融合

到目前为止，我们已经开发了NCF的两个实例——GMF（应用 **线性核** 来建模latent特征交互）和MLP（使用 **非线性核** 从数据中学习交互函数）。那么问题自然就产生了：**如何在NCF框架下融合GMF和MLP，使它们能够相互增强以更好地建模复杂的用户-item交互**？

一个直接的解决方案是**让GMF和MLP共享相同的嵌入层，然后组合它们交互函数的输出**。这种方式与著名的神经张量网络（NTN）[33]具有相似的精神。具体来说，将GMF与单层MLP组合的模型可以形式化为：


$$
\hat{y}_{ui} = \sigma(\mathbf{h}^T (\mathbf{p}_u \odot \mathbf{q}_i + \mathbf{W} [\mathbf{p}_u; \mathbf{q}_i] + \mathbf{b})) \quad (11)
$$


然而，共享GMF和MLP的嵌入可能会限制融合模型的性能。例如，这隐含着**GMF和MLP必须使用相同大小的嵌入**；对于两个模型的最优嵌入大小差异很大的数据集，这种解决方案可能无法获得最优的集成效果。此外，**共享嵌入也会限制模型的自由度**，使得GMF和MLP无法独立地学习最适合各自任务的用户和item表示。

为了给融合模型提供更大的灵活性，我们**允许GMF和MLP学习各自独立的嵌入**，并通过拼接它们的最后一个隐藏层来组合两个模型。图3展示了我们的方案，其公式如下：


$\phi_{GMF} = \mathbf{p}_u^G \odot \mathbf{q}_i^G,$

$\phi_{MLP} = a_L(W_L^T (a_{L-1}(...a_2(W_2^T [\mathbf{p}_u^M; \mathbf{q}_i^M] + b_2)...)) + b_L),$

$$
\hat{y}_{ui} = \sigma(\mathbf{h}^T [\phi_{GMF}; \phi_{MLP}]) \quad (12)
$$

其中 $\mathbf{p}_u^G$ 和 $\mathbf{p}_u^M$ 分别表示GMF和MLP部分的用户嵌入；$\mathbf{q}_i^G$ 和 $\mathbf{q}_i^M$ 类似地表示item嵌入。如前所述，我们使用ReLU作为MLP层的激活函数。该模型结合了MF的线性和DNN的非线性来建模用户-item latent结构。我们将此模型称为"NeuMF"（Neural Matrix Factorization，神经矩阵分解）。模型相对于每个参数的导数可以通过标准反向传播计算，由于篇幅限制在此省略。

<img src="/Users/dazhang/PycharmProject/Papers/3-RecSys/.picture/image-20260715204358553.png" alt="image-20260715204358553" style="zoom:50%;" />

> **图3：神经矩阵分解模型**

#### 3.4.1 预训练

**由于NeuMF目标函数是非凸的，基于梯度的优化方法只能找到局部最优解**。已有文献表明，初始化对深度学习模型的收敛和性能起着重要作用[7]。由于**NeuMF是GMF和MLP的集成模型**，我们建议使用GMF和MLP的预训练模型来初始化NeuMF。

我们首先使用随机初始化训练GMF和MLP直到收敛。然后使用它们的模型参数作为NeuMF对应部分参数的初始化值。唯一的调整是在输出层，我们拼接两个模型的权重：

$$
\mathbf{h} \leftarrow [\alpha h_{GMF}; (1-\alpha) h_{MLP}] \qquad (13)
$$

其中 $h_{GMF}$ 和 $h_{MLP}$ 分别表示预训练GMF和MLP模型的 $\mathbf{h}$ 向量；$\alpha$ 是超参数，决定两个预训练模型之间的权衡。

对于从头训练GMF和MLP，**我们采用自适应矩估计（Adam**）[20]优化器。Adam优化器通过为**高频更新参数分配较小的学习率、为低频更新参数分配较大的学习率**，自适应地调整每个参数的学习率。与普通的随机梯度下降（SGD）相比，Adam方法使两个模型收敛更快，同时也减轻了手动调整学习率的负担。

将预训练参数输入NeuMF后，我们使用普通SGD（而非Adam）对其进行进一步优化。这是因为Adam优化器需要在训练过程中保存动量信息（即梯度的一阶矩和二阶矩估计），才能正确地更新参数。由于我们仅使用预训练模型参数初始化NeuMF，**而没有同时保存Adam的动量信息**，如果继续使用Adam进行优化，动量信息将需要从零重新开始积累，这可能导致不稳定的更新。因此，我们**选择使用普通SGD对NeuMF进行微调**，以获得更稳定的收敛效果。

> [!NOTE]
>
> 为什么不保存Adam优化器信息，从而可以在微调阶段继续使用Adam呢？



## 4 实验

在本节中，我们进行实验，旨在回答以下research questions（研究问题）：

- **RQ1**：我们提出的NCF方法是否优于最先进的隐式协同过滤方法？
- **RQ2**：我们提出的优化框架（带负采样的对数损失）对推荐任务效果如何？
- **RQ3**：更深的隐藏层是否有助于从用户-item交互数据中学习？

接下来，我们首先介绍实验设置，然后逐一回答上述三个研究问题。

### 4.1 实验设置

**数据集。** 我们在两个公开数据集上进行了实验：MovieLens和Pinterest。两个数据集的特征总结在表1中。

1. **MovieLens。** 这个电影评分数据集已被广泛用于评估协同过滤算法。我们使用了包含100万评分的数据版本（**MovieLens 1M**），每个用户至少有20个评分。该数据集包含**6,040个用户和3,706部电影**，共有1,000,209条评分记录，数据稀疏度为95.53%。虽然这是一个**显式反馈数据**，但我们特意选择它来**研究从显式反馈的隐式信号[21]中学习的性能**。为此，**我们将其转换为隐式数据，每个条目标记为0或1，表示用户是否对该item进行了评分**。

2. **Pinterest。** 这个隐式反馈数据由[8]构建，用于评估基于内容的图像推荐。原始数据非常大但高度稀疏。例如，超过20%的用户只有一个pin，这使得评估协同过滤算法变得困难。因此，我们以与MovieLens数据相同的方式过滤数据集，仅保留至少有20次交互（pin）的用户。这产生了包含55,187个用户、9,916个item和1,500,809次交互的数据子集，数据稀疏度高达99.73%。每次交互表示用户是否将图像固定到了她自己的面板。相比于MovieLens，Pinterest的数据更加稀疏，这为评估协同过滤算法在真实稀疏场景下的性能提供了更具挑战性的测试环境。

> **表1：评估数据集的统计信息**
>
> | 数据集 | 交互数 | item数 | 用户数 | 稀疏度 |
> |--------|--------|--------|--------|--------|
> | MovieLens | 1,000,209 | 3,706 | 6,040 | 95.53% |
> | Pinterest | 1,500,809 | 9,916 | 55,187 | 99.73% |



**评估协议。** 为了评估item推荐的性能，我们采用了 **留一法** 评估，这种方法在文献中被广泛使用[1, 14, 27]。**对于每个用户，我们将其最后一次交互作为测试集，并使用剩余数据进行训练**。由于在评估期间**对所有item进行排序对于每个用户来说过于耗时**，我们遵循常见策略[6, 21]，**随机采样100个用户未交互的item，将测试item在这100个item中进行排序**。这种策略在保证评估效率的同时，也能较好地反映模型对item的排序能力。

> [!NOTE]
>
> 这是什么评估方法？留一法



排序列表的性能由两个指标来衡量：命中率（HR）和归一化折损累计增益（NDCG）[11]。除非特别说明，我们对两个指标都将排序列表截断在10位以内（即评估top-10推荐列表的性能）。HR@10直观地衡量**测试item是否出现在推荐列表的前10个位置中**，它是一个基于**是否命中的二值指标**。NDCG@10则更进一步，它通过为**顶部排名的命中分配更高的分数来考虑命中的具体位置，对排序位置更加敏感**。例如，测试item排在第1位比排在第10位获得更高的NDCG分数。我们计算每个测试用户的这两个指标，并报告所有用户的平均分数。

**基线方法。** 我们将提出的NCF方法（GMF、MLP和NeuMF）与以下方法进行了比较：

- **ItemPop**：item根据其**交互数量判断的流行度进行排序**。这是一种非个性化的方法，用于作为推荐性能的基准[27]。
- **ItemKNN [31]**：这是标准的基于item的协同过滤方法。我们遵循[19]的设置将其调整为适用于隐式数据。
- **BPR [27]**：该方法使用成**对排序损失优化公式(2)的MF模型**，专门用于从隐式反馈中学习。它是item推荐的一个高度竞争性的基线方法。我们使用固定的学习率，变化学习率并报告最佳性能。
- **eALS [14]**：这是一种最先进的用于item推荐的MF方法。它优化公式(5)的平方损失，**将所有未观察交互视为负实例**并**根据item流行度对其进行非均匀加权**。由于eALS显示出优于均匀加权方法WMF[19]的性能，我们不再报告WMF的性能。

由于我们提出的方法旨在建模用户和item之间的关系，我们主要与用户-item模型进行比较。我们省略了与item-item模型（如SLIM[25]和CDAE[44]）的比较，因为性能差异可能由个性化用户模型引起（因为它们是item-item模型）。

**参数设置。** 我们基于Keras深度学习框架实现了我们提出的方法。为了确定NCF方法的超参数，我们**为每个用户随机采样一个交互作为验证数据，并在验证数据上调整超参数**。所有**NCF模型通过优化公式(7)的对数损失来学习，其中每个正实例采样四个负实例**。对于从头训练的NCF模型，我们使用高斯分布（均值为0，标准差为0.01）随机初始化模型参数，使用小批量Adam[20]优化模型。我们测试的批量大小为[128, 256, 512, 1024]，学习率为[0.0001, 0.0005, 0.001, 0.005]。由于NCF的**最后一个隐藏层**决定了模型的能力，我们将其称为预测因子，并评估了因子数为[8, 16, 32, 64]的性能。值得注意的是，较大的因子可能导致过拟合并降低性能。除非特别说明，我们为MLP使用三个隐藏层；例如，如果预测因子大小为8，则神经CF层的架构为 $32 \rightarrow 16 \rightarrow 8$，嵌入大小为16。对于带预训练的NeuMF，$\alpha$ 设为0.5，使**预训练的GMF和MLP对NeuMF的初始化贡献相等**。

### 4.2 性能比较（RQ1）

图4显示了HR@10和NDCG@10随预测因子数量的变化情况。对于MF方法BPR和eALS，预测因子的数量等于latent因子的数量。对于ItemKNN，我们**测试了不同的邻居大小**并报告最佳性能。由于ItemPop性能较弱，图4中省略了它以更好地突出个性化方法的性能差异。

首先，我们可以看到NeuMF在两个数据集上都取得了最佳性能，显著优于最先进的方法eALS和BPR，且差距较大（平均而言，相对于eALS和BPR的相对提升分别为4.5%和4.9%）。对于Pinterest，即使使用较小的预测因子8，NeuMF也显著优于使用较大因子64的eALS和BPR。这表明通过融合线性MF和非线性MLP模型，NeuMF具有高度的表达能力。其次，另外两种NCF方法——GMF和MLP——也展示了相当强的性能。在它们之间，MLP略逊于GMF。注意，MLP可以通过添加更多隐藏层来进一步提升（见第4.4节），这里我们仅展示三层的性能。对于较小的预测因子，GMF在两个数据集上都优于eALS；尽管GMF在较大因子时会过拟合，但其获得的最佳性能优于（或持平于）eALS。最后，GMF相较于BPR表现出持续改进，这证实了分类感知的对数损失对推荐任务的有效性，因为GMF和BPR学习相同的MF模型但使用不同的目标函数。

![image-20260715204459722](/Users/dazhang/PycharmProject/Papers/3-RecSys/.picture/image-20260715204459722.png)

> **图4：两个数据集上HR@10和NDCG@10随预测因子数量的变化。** (a) MovieLens HR@10；(b) MovieLens NDCG@10；(c) Pinterest HR@10；(d) Pinterest NDCG@10。

从图4中我们可以观察到几个重要趋势。**在MovieLens数据集上，所有方法的性能随着预测因子数量的增加而提升，但提升幅度逐渐减小**。NeuMF在因子数为64时达到最佳性能，HR@10约为0.73，NDCG@10约为0.447。在Pinterest数据集上，由于数据更加稀疏（99.73%），性能整体低于MovieLens，但NeuMF仍然保持显著优势。值得注意的是，在Pinterest上，即使仅使用8个预测因子，NeuMF的HR@10也已达到约0.878，远高于eALS和BPR使用64个因子时的性能，这充分说明**NeuMF融合线性与非线性的架构设计在稀疏场景下具有更强的泛化能力。**

GMF在两个数据集上均一致地优于BPR，这一现象值得深入分析。**GMF和BPR本质上学习的是相同的MF模型结构，唯一的区别在于目标函数——GMF使用带负采样的对数损失（即二分类损失），而BPR使用成对排序损失。实验结果清晰地表明，对于隐式反馈推荐任务，将问题建模为二分类任务并使用对数损失进行优化，比使用成对排序损失更为有效**。这为NCF框架选择对数损失提供了强有力的经验支持。

MLP性能略低于GMF，但这并不意味着MLP模型没有价值。实际上，MLP通过多层非线性变换能够学习到更加复杂的用户-item交互模式。本文中MLP仅使用了3个隐藏层，正如后续第4.4节所示，增加MLP的层数可以进一步提升其性能。此外，MLP作为NCF框架中的一个重要组件，在与GMF融合形成NeuMF后发挥了关键作用——NeuMF之所以能够取得最佳性能，正是因为同时利用了MF的线性和MLP的非线性建模能力。

**图5显示了Top-K推荐列表的性能**，其中排序位置K从1到10。为了使图表更清晰，我们仅展示NeuMF而非所有三种NCF方法的性能。可以看出，NeuMF在各个位置上均展现出优于其他方法的持续性改进，我们进一步进行了**单样本配对t检验**，验证了所有改进在p<0.01时均具有统计显著性。对于基线方法，eALS在MovieLens上优于BPR约5.1%的相对提升，而在Pinterest上在NDCG方面劣于BPR。这与[14]的发现一致，即BPR由于其成对排序感知的学习器，在排序性能方面可以表现强劲。**基于邻居的ItemKNN性能不如基于模型的方法。ItemPop表现最差，表明需要建模用户的个性化偏好，而不仅仅是向用户推荐流行item。**

![image-20260715204520082](/Users/dazhang/PycharmProject/Papers/3-RecSys/.picture/image-20260715204520082.png)

> **图5：两个数据集上Top-Kitem推荐评估（K从1到10）。** (a) MovieLens HR@K；(b) MovieLens NDCG@K；(c) Pinterest HR@K；(d) Pinterest NDCG@K。

#### 4.2.1 预训练的作用

为了展示预训练对NeuMF的作用，我们比较了两个版本的NeuMF——有预训练和无预训练的性能。在两种设置中，所有其他超参数保持相同，以确保公平比较。**对于无预训练的NeuMF，我们使用Adam随机初始化进行学习**。如表2所示，有预训练的NeuMF在大多数情况下取得了更好的性能；只有在MovieLens上使用较小的预测因子8时，预训练方法性能略差。有预训练的NeuMF在MovieLens和Pinterest上的相对提升分别为2.2%和1.1%。这一结果证明了**我们用于初始化NeuMF的预训练方法的有用性**。

> **表2：NeuMF有预训练和无预训练的性能**
>
> | 因子 | 有预训练 HR@10 | 有预训练 NDCG@10 | 无预训练 HR@10 | 无预训练 NDCG@10 |
> |------|---------------|-----------------|---------------|-----------------|
> | MovieLens 8 | 0.684 | 0.403 | 0.688 | 0.410 |
> | MovieLens 16 | 0.707 | 0.426 | 0.696 | 0.420 |
> | MovieLens 32 | 0.726 | 0.445 | 0.701 | 0.425 |
> | MovieLens 64 | 0.730 | 0.447 | 0.705 | 0.426 |
> | Pinterest 8 | 0.878 | 0.555 | 0.869 | 0.546 |
> | Pinterest 16 | 0.880 | 0.558 | 0.871 | 0.547 |
> | Pinterest 32 | 0.879 | 0.555 | 0.870 | 0.549 |
> | Pinterest 64 | 0.877 | 0.552 | 0.872 | 0.551 |

### 4.3 带负采样的对数损失（RQ2）

为了处理隐式反馈的单类性质，我们将推荐视为二分类任务。通过将NCF视为概率模型，我们使用对数损失对其进行优化。图6显示了MovieLens上NCF方法每轮迭代的训练损失（**所有实例的平均值**）和推荐性能。Pinterest上的结果呈现相同趋势，因此由于篇幅限制而省略。

首先，我们可以看到随着迭代次数的增加，NCF模型的训练损失逐渐减少，推荐性能得到提升。最有效的更新发生在前10次迭代中，更多的迭代可能会过拟合模型（例如，尽管NeuMF的训练损失在10次迭代后持续下降，但其推荐性能实际上下降了）。其次，在三种NCF方法中，NeuMF取得了最低的训练损失，其次是MLP，然后是GMF。推荐性能也呈现出相同的趋势：NeuMF > MLP > GMF。上述发现为使用对数损失从隐式数据中学习的合理性和有效性提供了经验证据。

**逐点对数损失相对于成对目标函数[27, 33]的一个重要优势是负实例的灵活采样比率**。成对目标函数（如BPR）只能将一个采样的负实例与一个正实例配对，而我们在逐点损失中可以灵活控制每个正实例对应的负样本数量，即采样比率（sampling ratio）。这种灵活性允许我们**更充分地利用未观察交互中的信息**。

为了说明负采样对NCF方法的影响，我们在图7中展示了NCF方法在不同负采样比率下的性能。实验中我们固定预测因子数为16，变化每个正实例对应的负样本数量从1到10。从图中可以清楚地看到几个重要现象。

**首先，每个正实例仅一个负样本（采样比率为1）不足以达到最优性能，增加采样数量可以显著提升推荐效果**。例如，在MovieLens数据集上，当采样比率从1增加到4时，NeuMF的HR@10从约0.68提升至约0.71，相对提升约4.4%。这表明更多的负样本可以为模型提供更丰富的训练信号。

其次，比较GMF和BPR，我们可以看到采样比率为1时GMF的性能与BPR相当，而GMF在更大采样比率下显著优于BPR。由于GMF和BPR使用相同的MF模型结构，唯一的区别在于目标函数——GMF使用逐点对数损失，BPR使用成对排序损失——这一对比清晰地展示了**逐点对数损失相对于成对BPR损失的优势**。

第三，在两个数据集上，最优采样比率约为3到6。在Pinterest上，我们发现当采样比率大于7时，NCF方法的性能开始下降。这表明设置**过大的采样比率可能对性能产生不利影响，因为过多的负样本可能引入噪声或导致正负样本不平衡问题**。因此，在实践中需要谨慎选择负采样比率。

![image-20260715204537778](/Users/dazhang/PycharmProject/Papers/3-RecSys/.picture/image-20260715204537778.png)

> **图6：NCF方法在MovieLens上训练损失和推荐性能随迭代次数的变化（因子=8）。** (a) 训练损失；(b) HR@10；(c) NDCG@10。

![image-20260715204553501](/Users/dazhang/PycharmProject/Papers/3-RecSys/.picture/image-20260715204553501.png)

> **图7：NCF方法性能随每个正实例的负样本数量的变化（因子=16）。** 同时显示了BPR的性能，它只采样一个负实例与正实例配对学习。(a) MovieLens HR@10；(b) MovieLens NDCG@10；(c) Pinterest HR@10；(d) Pinterest NDCG@10。

### 4.4 深度学习是否有帮助？（RQ3）

由于关于用神经网络学习用户-item交互函数的工作很少，因此探讨使用深层网络结构是否对推荐任务有益是一个值得深入研究的开放性问题。在本文中，我们通过控制变量实验来回答这一问题。为此，我们进一步研究了具有不同数量隐藏层的MLP。结果总结在表3和表4中。MLP-3表示具有三个隐藏层（除了嵌入层之外）的MLP方法，其他以此类推。我们可以看到，即使对于具有相同能力的模型，**堆叠更多层也有利于性能**。这一结果非常令人鼓舞，表明使用深度模型进行协同推荐的有效性。我们将**改进归因于堆叠更多非线性层所带来的高度非线性。为了验证这一点，我们进一步尝试堆叠线性层，使用恒等函数作为激活函数，其性能远差于使用ReLU单元。**

对于没有隐藏层的MLP-0（即嵌入层直接投影到预测），性能非常弱，甚至不如非个性化的ItemPop。这验证了我们在第3.3节中的论点，即简单地拼接用户和item的latent向量不足以建模它们的特征交互，因此需要用隐藏层进行变换。

> **表3：不同层数MLP的HR@10**
>
> | 因子 | MLP-0 | MLP-1 | MLP-2 | MLP-3 | MLP-4 |
> |------|-------|-------|-------|-------|-------|
> | MovieLens 8 | 0.452 | 0.628 | 0.655 | 0.671 | 0.678 |
> | MovieLens 16 | 0.454 | 0.663 | 0.674 | 0.684 | 0.690 |
> | MovieLens 32 | 0.453 | 0.682 | 0.687 | 0.692 | 0.699 |
> | MovieLens 64 | 0.453 | 0.687 | 0.696 | 0.702 | 0.707 |
> | Pinterest 8 | 0.275 | 0.848 | 0.855 | 0.859 | 0.862 |
> | Pinterest 16 | 0.274 | 0.855 | 0.861 | 0.865 | 0.867 |
> | Pinterest 32 | 0.273 | 0.861 | 0.863 | 0.868 | 0.867 |
> | Pinterest 64 | 0.274 | 0.864 | 0.867 | 0.869 | 0.873 |
>
> **表4：不同层数MLP的NDCG@10**
>
> | 因子 | MLP-0 | MLP-1 | MLP-2 | MLP-3 | MLP-4 |
> |------|-------|-------|-------|-------|-------|
> | MovieLens 8 | 0.253 | 0.359 | 0.383 | 0.399 | 0.406 |
> | MovieLens 16 | 0.252 | 0.391 | 0.402 | 0.410 | 0.415 |
> | MovieLens 32 | 0.252 | 0.406 | 0.410 | 0.425 | 0.423 |
> | MovieLens 64 | 0.251 | 0.409 | 0.417 | 0.426 | 0.432 |
> | Pinterest 8 | 0.141 | 0.526 | 0.534 | 0.536 | 0.539 |
> | Pinterest 16 | 0.141 | 0.532 | 0.536 | 0.538 | 0.544 |
> | Pinterest 32 | 0.142 | 0.537 | 0.538 | 0.542 | 0.546 |
> | Pinterest 64 | 0.141 | 0.538 | 0.542 | 0.545 | 0.550 |



## 5 相关工作

虽然早期的推荐文献主要集中在显式反馈[30, 31]上，但最近的关注点逐渐转向隐式数据[1, 14, 23]。**隐式反馈具有易于收集、数据量大、覆盖面广等优点，因此在实际工业推荐系统中得到了广泛应用**。隐式反馈下的协同过滤任务通常被形式化为item推荐问题，其目标是为用户推荐一个简短的item列表。**与已通过显式反馈工作广泛解决的评分预测相比，解决item推荐问题更实用但也更具挑战性[**1, 11]。一个关键的见解是对缺失数据进行建模，而显式反馈工作通常忽略这一点[21, 48]。为了使latent因子模型适用于隐式反馈的item推荐，早期工作[19, 27]采用了均匀加权策略，提出了两种方法——**要么将所有缺失数据视为负实例[19]，要么从缺失数据中采样负实例[27]**。最近，He等人[14]和Liang等人[23]提出了专门对缺失数据进行加权的模型，Rendle等人[1]为基于特征的因子分解模型开发了一种隐式坐标下降解决方案（iCD），在item推荐方面达到了最先进的性能。下面我们讨论使用神经网络的推荐工作。

Salakhutdinov等人[30]的早期先驱工作提出了两层受限玻尔兹曼机（RBM）来建模用户对item的显式评分。该工作后来被扩展以建模评分的序数性质[36]。最近，自编码器已成为构建推荐系统的流行选择[32, 22, 35]。基于用户的AutoRec[32]的核心思想是学习能够根据用户的历史评分作为输入来重构用户评分的隐藏结构。在用户个性化方面，这种方法与item-item模型[31, 25]（将用户表示为其评分过的item）有相似的精神。为了避免自编码器学习恒等函数而无法泛化到未见数据，去噪自编码器（DAE）已被应用于从有意损坏的输入中学习[22, 35]。更近期，Zheng等人[48]提出了一种用于CF的神经自回归方法。虽然先前的努力支持了神经网络解决CF的有效性，但大多数工作专注于显式评分且仅对观察到的数据进行建模。因此，这些方法很容易无法从仅包含正反馈的隐式数据中学习用户偏好。

尽管一些近期工作[6, 37, 38, 43, 45]已经探索了基于隐式反馈的推荐深度学习模型，但它们主要使用DNN来建模辅助信息，例如item的文本描述[38]、音乐的声学特征[37, 43]、用户的跨域行为[6]以及知识库中的丰富信息[45]。由DNN学习到的特征随后与MF集成用于CF。与我们工作最相关的是[44]，该工作由Wu等人提出，是一种用于隐式反馈CF的协同去噪自编码器（CDAE）。与基于DAE的CF[35]相比，CDAE额外将一个用户节点插入到自编码器的输入中，用于重构用户的评分。如作者所示，当对CDAE的隐藏层使用恒等函数激活时，CDAE等价于SVD++模型[21]。这意味着尽管CDAE是一种用于CF的神经建模方法，它仍然应用线性核（即内积）来建模用户-item交互。这可以部分解释为什么对CDAE使用深层网络并不能提升性能（参见[44]第6节）。与CDAE不同，我们的NCF采用双路径架构，使用多层前馈神经网络建模用户-item交互。这使得NCF能够从数据中学习任意函数，比固定的内积函数更强大、更具表达力。

沿着类似的思路，学习两个实体（如用户和item）之间的关系在知识图文献中得到了深入研究[2, 33]。许多关系机器学习方法已被设计出来[24]。与我们的提议最相似的是神经张量网络（NTN）[33]，它使用神经网络学习两个实体的交互并展示了强大的性能。这里我们关注的是CF的不同问题设置。虽然结合MF和MLP的NeuMF的思想部分受NTN启发，但我们的NeuMF比NTN更灵活和通用，因为它允许MF和MLP学习不同的嵌入集合。最近，Google公开了他们用于应用推荐的Wide & Deep学习方法[4]。该方法的深度组件类似地在特征嵌入上使用MLP，已被报告具有很强的泛化能力。虽然他们的工作侧重于整合用户和item的各种特征，我们的目标则是探索DNN用于纯协同过滤系统。我们表明，**DNN是建模用户-item交互**的一个有前景的选择，据我们所知，这一点之前尚未被研究过。



## 6 结论与未来工作

在本文中，我们探索了用于协同过滤的神经网络架构。我们设计了一个**通用框架NCF，并提出了三种实例化模型**——GMF（广义矩阵分解）、MLP（多层感知机）和NeuMF（神经矩阵分解）——它们以不同的方式建模用户-item交互。我们的框架简单且通用；它不仅限于本文提出的模型，而且旨在作为开发基于深度学习的推荐方法的指导方针。本文补充了主流的浅层协同过滤模型，为基于深度学习的推荐开辟了新的研究可能性。

本文的主要贡献可以总结为以下三点：第一，我们首次将神经网络系统地应用于协同过滤中的用户-item交互建模，提出了一个通用的NCF框架。第二，我们在此框架下实现了三种具有代表性的模型——GMF（广义矩阵分解）、MLP（多层感知机）和NeuMF（神经矩阵分解），其中NeuMF通过融合线性和非线性组件取得了最优性能。第三，我们在两个真实数据集上进行了全面的实验验证，结果表明NCF方法显著优于现有的隐式反馈推荐方法。

在未来的工作中，我们将研究NCF模型的成对学习器，并将NCF扩展到建模辅助信息，例如用户评论[11]、知识库[45]和时间信号[1]。虽然现有的个性化模型主要关注个体用户，但开发面向用户群体的模型也很有意义，这有助于社会群体[15, 42]的决策制定。此外，我们特别对构建多媒体item的推荐系统感兴趣，这是一个有趣的任务，但在推荐社区中受到的关注相对较少[3]。多媒体item（如图像和视频）包含更丰富的视觉语义[16, 41]，这些视觉语义可以反映用户的兴趣。为了构建多媒体推荐系统，我们需要开发从多视图和多模态数据中学习的有效方法[13, 40]。另一个新兴的研究方向是探索循环神经网络和哈希方法[46]在提供高效在线推荐[14]方面的潜力。

我们还计划探索以下方向：**使用注意力机制来增强NCF模型的表达能力，使模型能够自动学习哪些latent特征对当前预测更为重要**；引入**序列建模方法，捕捉用户行为随时间演化的动态模式**；以及研究如何将NCF模型部署到大规模工业推荐系统中，解决在线推理的延迟和效率问题。



## 致谢

作者感谢匿名审稿人的宝贵意见，这些意见对作者在推荐系统方面的思考以及本文的修订都很有帮助。



## 参考文献

[1] I. Bayer, X. He, B. Kanagal, and S. Rendle. A generic coordinate descent framework for learning from implicit feedback. In WWW, 2017.

[2] A. Bordes, N. Usunier, A. Garcia-Duran, J. Weston, and O. Yakhnenko. Translating embeddings for modeling multi-relational data. In NIPS, pages 2787–2795, 2013.

[3] T. Chen, X. He, and M.-Y. Kan. Context-aware image tweet modelling and recommendation. In MM, pages 1018–1027, 2016.

[4] H.-T. Cheng, L. Koc, J. Harmsen, T. Shaked, T. Chandra, H. Aradhye, G. Anderson, G. Corrado, W. Chai, M. Ispir, et al. Wide & deep learning for recommender systems. arXiv preprint arXiv:1606.07792, 2016.

[5] R. Collobert and J. Weston. A unified architecture for natural language processing: Deep neural networks with multitask learning. In ICML, pages 160–167, 2008.

[6] A. M. Elkahky, Y. Song, and X. He. A multi-view deep learning approach for cross domain user modeling in recommendation systems. In WWW, pages 278–288, 2015.

[7] D. Erhan, Y. Bengio, A. Courville, P.-A. Manzagol, P. Vincent, and S. Bengio. Why does unsupervised pre-training help deep learning? Journal of Machine Learning Research, 11:625–660, 2010.

[8] X. Geng, H. Zhang, J. Bian, and T.-S. Chua. Learning image and user features for recommendation in social networks. In ICCV, pages 4274–4282, 2015.

[9] X. Glorot, A. Bordes, and Y. Bengio. Deep sparse rectifier neural networks. In AISTATS, pages 315–323, 2011.

[10] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In CVPR, 2016.

[11] X. He, T. Chen, M.-Y. Kan, and X. Chen. TriRank: Review-aware explainable recommendation by modeling aspects. In CIKM, pages 1661–1670, 2015.

[12] X. He, M. Gao, M.-Y. Kan, Y. Liu, and K. Sugiyama. Predicting the popularity of web 2.0 items based on user comments. In SIGIR, pages 233–242, 2014.

[13] X. He, M.-Y. Kan, P. Xie, and X. Chen. Comment-based multi-view clustering of web 2.0 items. In WWW, pages 771–782, 2014.

[14] X. He, H. Zhang, M.-Y. Kan, and T.-S. Chua. Fast matrix factorization for online recommendation with implicit feedback. In SIGIR, pages 549–558, 2016.

[15] R. Hong, Z. Hu, L. Liu, M. Wang, S. Yan, and Q. Tian. Understanding blooming human groups in social networks. IEEE Transactions on Multimedia, 17(11):1980–1988, 2015.

[16] R. Hong, Y. Yang, M. Wang, and X. S. Hua. Learning visual semantic relationships for efficient visual retrieval. IEEE Transactions on Big Data, 1(4):152–161, 2015.

[17] K. Hornik, M. Stinchcombe, and H. White. Multilayer feedforward networks are universal approximators. Neural Networks, 2(5):359–366, 1989.

[18] L. Hu, A. Sun, and Y. Liu. Your neighbors affect your ratings: On geographical neighborhood influence to rating prediction. In SIGIR, pages 345–354, 2014.

[19] Y. Hu, Y. Koren, and C. Volinsky. Collaborative filtering for implicit feedback datasets. In ICDM, pages 263–272, 2008.

[20] D. Kingma and J. Ba. Adam: A method for stochastic optimization. In ICLR, pages 1–15, 2014.

[21] Y. Koren. Factorization meets the neighborhood: A multifaceted collaborative filtering model. In KDD, pages 426–434, 2008.

[22] S. Li, J. Kawale, and Y. Fu. Deep collaborative filtering via marginalized denoising auto-encoder. In CIKM, pages 811–820, 2015.

[23] D. Liang, L. Charlin, J. McInerney, and D. M. Blei. Modeling user exposure in recommendation. In WWW, pages 951–961, 2016.

[24] M. Nickel, K. Murphy, V. Tresp, and E. Gabrilovich. A review of relational machine learning for knowledge graphs. Proceedings of the IEEE, 104:11–33, 2016.

[25] X. Ning and G. Karypis. Slim: Sparse linear methods for top-n recommender systems. In ICDM, pages 497–506, 2011.

**[26] S. Rendle. Factorization machines. In ICDM, pages 995–1000, 2010.**

**[27] S. Rendle, C. Freudenthaler, Z. Gantner, and L. Schmidt-Thieme. Bpr: Bayesian personalized ranking from implicit feedback. In UAI, pages 452–461, 2009.**

[28] S. Rendle, Z. Gantner, C. Freudenthaler, and L. Schmidt-Thieme. Fast context-aware recommendations with factorization machines. In SIGIR, pages 635–644, 2011.

[29] R. Salakhutdinov and A. Mnih. Probabilistic matrix factorization. In NIPS, pages 1–8, 2008.

[30] R. Salakhutdinov, A. Mnih, and G. Hinton. Restricted boltzmann machines for collaborative filtering. In ICDM, pages 791–798, 2007.

[31] B. Sarwar, G. Karypis, J. Konstan, and J. Riedl. Item-based collaborative filtering recommendation algorithms. In WWW, pages 285–295, 2001.

[32] S. Sedhain, A. K. Menon, S. Sanner, and L. Xie. Autorec: Autoencoders meet collaborative filtering. In WWW, pages 111–112, 2015.

[33] R. Socher, D. Chen, C. D. Manning, and A. Ng. Reasoning with neural tensor networks for knowledge base completion. In NIPS, pages 926–934, 2013.

[34] N. Srivastava and R. R. Salakhutdinov. Multimodal learning with deep boltzmann machines. In NIPS, pages 2222–2230, 2012.

[35] F. Strub and J. Mary. Collaborative filtering with stacked denoising autoencoders and sparse inputs. In NIPS Workshop on Machine Learning for eCommerce, 2015.

[36] T. T. Truyen, D. Q. Phung, and S. Venkatesh. Ordinal boltzmann machines for collaborative filtering. In UAI, pages 548–556, 2009.

[37] A. Van den Oord, S. Dieleman, and B. Schrauwen. Deep content-based music recommendation. In NIPS, pages 2643–2651, 2013.

[38] H. Wang, N. Wang, and D.-Y. Yeung. Collaborative deep learning for recommender systems. In KDD, pages 1235–1244, 2015.

[39] M. Wang, W. Fu, S. Hao, D. Tao, and X. Wu. Scalable semi-supervised learning by efficient anchor graph regularization. IEEE Transactions on Knowledge and Data Engineering, 28(7):1864–1877, 2016.

[40] M. Wang, H. Li, D. Tao, K. Lu, and X. Wu. Multimodal graph-based reranking for web image search. IEEE Transactions on Image Processing, 21(11):4649–4661, 2012.

[41] M. Wang, X. Liu, and X. Wu. Visual classification by l1 hypergraph modeling. IEEE Transactions on Knowledge and Data Engineering, 27(9):2564–2574, 2015.

[42] X. Wang, L. Nie, X. Song, D. Zhang, and T.-S. Chua. Unifying virtual and physical worlds: Learning towards local and global consistency. ACM Transactions on Information Systems, 2017.

[43] X. Wang and Y. Wang. Improving content-based and hybrid music recommendation using deep learning. In MM, pages 627–636, 2014.

[44] Y. Wu, C. DuBois, A. X. Zheng, and M. Ester. Collaborative denoising auto-encoders for top-n recommender systems. In WSDM, pages 153–162, 2016.

[45] F. Zhang, N. J. Yuan, D. Lian, X. Xie, and W.-Y. Ma. Collaborative knowledge base embedding for recommender systems. In KDD, pages 353–362, 2016.

[46] H. Zhang, F. Shen, W. Liu, X. He, H. Luan, and T.-S. Chua. Discrete collaborative filtering. In SIGIR, pages 325–334, 2016.

[47] H. Zhang, Y. Yang, H. Luan, S. Yang, and T.-S. Chua. Start from scratch: Towards automatically identifying, modeling, and naming visual attributes. In MM, pages 187–196, 2014.

[48] Y. Zheng, B. Tang, W. Ding, and H. Zhou. A neural autoregressive approach to collaborative filtering. In ICML, pages 764–773, 2016.

---

> **译者注：** 本文是NCF（Neural Collaborative Filtering）论文的完整中文翻译。NCF是推荐系统领域里程碑式的工作，首次系统性地将深度学习应用于协同过滤的用户-item交互建模，对后续的深度学习推荐模型（如DeepFM、xDeepFM、AutoInt等）产生了深远影响。

> **引用信息：** He X, Liao L, Zhang H, et al. Neural collaborative filtering[C]//Proceedings of the 26th International Conference on World Wide Web. 2017: 173-182.
>
> **许可协议：** 本文遵循Creative Commons CC BY 4.0许可协议。
