# 耦合组 Lasso 用于展示广告中的Web-Scale CTR预测

> Ling Yan, Shanghai Jiao Tong University, China; Wu-Jun Li, Nanjing University, China; Gui-Rong Xue, Alibaba Group, China; Dingyi Han, Alibaba Group, China

本文提出耦合组 Lasso（CGL）模型用于展示广告中的CTR预测，**通过无缝整合用户特征和广告特征的联合信息来建模，同时自动消除无用特征**。核心发现是——**CGL 在三个真实世界Web-Scale数据集上显著优于LR基线，相对改进达3%-5%，同时实现3%-15%的特征稀疏度**。

核心内容：

- **问题/痛点**：逻辑回归（LR）无法捕捉用户特征和广告特征之间的非线性联合信息；手动构建联合特征会导致参数数量二次增长，难以学习
- **方案/创新点**：提出CGL模型，通过 $(x_u^T W)(x_a^T V)^T$ 项自动建模用户-广告联合信息；使用L2,1范数正则化实现组级特征选择
- **技术细节**：参数矩阵 $W$ 和 $V$ 分别对应用户特征和广告特征；交替使用L-BFGS优化；通过特征哈希和分布式实现确保可扩展性
- **实验验证**：在阿里巴巴淘宝的三个真实数据集上验证，每个数据集包含数十亿展示实例；使用80个节点的MPI集群进行分布式训练

关键发现：

- **CGL在所有三个数据集上显著优于LR**，相对AUC改进为3%-5%（Dataset-1: 3.90%, Dataset-2: 3.42%, Dataset-3: 4.28%）
- **Lasso（L1正则化LR）与LR无显著差异**，相对改进仅为-0.019%至+0.086%
- **CGL可自动消除无用特征**，GSparsity达3%-15%时仍保持良好预测精度
- **分布式训练接近线性加速**，80节点训练时间约3,000-4,300秒

---

## 摘要

在展示广告中，点击率（CTR）预测是估计广告在特定上下文中展示给用户时被点击概率的问题。由于其易于实现和 promising 的性能，逻辑回归（LR）模型已被广泛用于CTR预测，特别是在工业系统中。然而，LR不容易从用户特征和广告特征中捕捉非线性信息，如联合信息。在本文中，我们提出了一个新颖的模型，称为耦合组 Lasso（CGL），用于展示广告中的CTR预测。CGL可以无缝整合用户特征和广告特征的联合信息进行建模。此外，CGL可以自动消除用户和广告的无用特征，这有助于快速在线预测。CGL的可扩展性通过特征哈希和分布式实现得到保证。在真实世界数据集上的实验结果表明，我们的CGL模型可以在Web-Scale CTR预测任务上实现最先进的性能。

## 1. 引言

最近，在线广告已成为品牌推广和产品营销最受欢迎和有效的方法。它是网络上的数十亿美元业务，占主要互联网公司（如Google、Yahoo和阿里巴巴）收入的大部分。展示广告是在线广告的重要组成部分，广告商向发布商付费在其网页上放置图形广告。发布商在其网页上分配一些位置并出售给不同的广告商。用户访问网页并可以查看发布的广告。还有一些其他角色，如广告代理和发布商网络，构成复杂的广告系统。但这不是本文的重点。因此我们将只关注用户-广告商-发布商三方业务的场景，其中三方有独立的目标，最终可以归结为统一的任务。广告商更关注期望的用户行为，如点击广告、订阅邮件列表或购买产品。不同的广告商针对不同类型的用户。例如，篮球公司会对最近购买许多体育设备的用户感兴趣，酒店会更愿意向经常旅行的人展示其广告。广告商有不同的付费选项，如每次点击成本（CPC）、每千次展示成本（CPM）和每次转化成本（CPA）。对于发布商部分，他们的目标是最大化来自广告商的收入并吸引更多用户到其网页。因此他们最好精确地向特定用户展示合适的广告，避免影响用户的网页体验。从用户部分，他们希望从网页中找到有用信息并找到他们真正感兴趣的广告。

为了满足三方的愿望，精确的广告系统定向非常重要，其中用户对特定广告的点击率（CTR）预测起着关键作用。CTR预测是估计向特定用户展示广告将导致点击的概率问题。这个具有挑战性的问题是展示广告的核心，必须处理几个困难问题，如非常大的数据集、频繁更新的用户和广告，以及用户特征（特征）和广告特征之间固有的模糊连接。

最近，许多模型被提出用于展示广告中的CTR预测。一些模型在用户和广告特征的简单连接上训练标准分类器，如逻辑回归（LR）或广义线性模型。其他一些模型使用先验知识，如对数线性模型或LR模型中固有的层次信息进行统计平滑。提出了矩阵分解方法，但它没有利用用户特征。提出了概率模型将用户和物品元数据与协同过滤信息一起使用，其中用户和物品特征向量被映射到低维空间，使用内积来衡量相似性。然而，它没有从用户和物品元特征中自动进行特征选择的效果。此外，模型的推理太复杂，无法用于大规模场景。提出了基于LR的高度可扩展框架，并使用来自真实应用的TB级数据进行评估。由于其易于实现和最先进的性能，LR模型已成为CTR预测中最受欢迎的模型，特别是在工业系统中。然而，LR是线性模型，其中特征独立地对最终预测做出贡献。因此，LR无法捕捉用户特征和广告特征之间的非线性信息，如联合（笛卡尔积）信息。在真实应用中，联合信息对CTR预测非常重要。例如，购买力高的人可能比购买力低的人对奢侈品更感兴趣，大学生可能比高中生更有可能购买机器学习书籍。通过特征联合利用用户-广告两部分混合特征可以预期更好的性能。

在本文中，我们提出了一个新颖的模型，称为耦合组 Lasso（CGL），用于展示广告中的CTR预测。主要贡献概述如下：

- CGL可以无缝整合用户特征和广告特征的联合信息进行建模，这使其比LR更好地捕捉用户和广告之间的潜在连接。
- CGL可以自动消除用户和广告的无用特征，这有助于快速在线预测。
- CGL通过利用特征哈希和分布式实现具有可扩展性。

## 2. 背景

在本节中，我们介绍模型的背景，包括CTR预测任务的描述、LR模型和组 Lasso。

### 2.1 符号和任务

我们使用粗体小写字母（如 $\mathbf{v}$）表示列向量，$v_i$ 表示 $\mathbf{v}$ 的第 $i$ 个元素。粗体大写字母（如 $\mathbf{M}$）用于表示矩阵，$\mathbf{M}$ 的第 $i$ 行和第 $j$ 列分别由 $\mathbf{M}_{i*}$ 和 $\mathbf{M}_{*j}$ 表示。$M_{ij}$ 是 $\mathbf{M}$ 第 $i$ 行第 $j$ 列的元素。$\mathbf{M}^T$ 是 $\mathbf{M}$ 的转置，$\mathbf{v}^T$ 是 $\mathbf{v}$ 的转置。

虽然一些展示广告系统只能访问用户或广告的某些ID信息，但在本文中我们专注于可以收集用户和广告特征的场景。实际上，发布商通常可以收集用户在网页上的行为，如点击广告、购买产品或输入一些查询关键词。他们可以分析这些历史行为然后构建用户特征（特征）。另一方面，当广告商向发布商提交一些广告时，他们通常选择一些描述词、展示广告的人群组或其他有用的特征。

我们将向特定用户在特定页面视图中展示广告称为广告展示。每个展示是用户在特定上下文（如白天、工作日和发布位置）中遇到广告的情况。因此，每个展示包含三个方面的信息：用户、广告和上下文。我们使用长度为 $l$ 的 $\mathbf{x}_u$ 表示用户 $u$ 的特征向量，长度为 $s$ 的 $\mathbf{x}_a$ 表示广告 $a$ 的特征向量。上下文信息与一些广告商ID或广告ID信息一起组成长度为 $d$ 的特征向量 $\mathbf{x}_o$。$\mathbf{x}$ 用于表示展示的特征向量，$\mathbf{x}^T = (\mathbf{x}_u^T, \mathbf{x}_a^T, \mathbf{x}_o^T)$。因此，如果我们使用 $z$ 表示向量 $\mathbf{x}$ 的长度，我们有 $z = l + s + d$。展示的结果是点击或非点击，这构成数据集中的一个实例。

给定训练集 $\{(\mathbf{x}^{(i)}, y^{(i)}) | i = 1, ..., N\}$，其中 $\mathbf{x}^T = (\mathbf{x}_u^T, \mathbf{x}_a^T, \mathbf{x}_o^T)$，$y \in \{0, 1\}$，$y = 1$ 表示点击，$y = 0$ 表示展示中的非点击，CTR预测问题是学习函数 $h(\mathbf{x}) = h(\mathbf{x}_u, \mathbf{x}_a, \mathbf{x}_o)$，可用于预测用户 $u$ 在特定上下文 $o$ 中点击广告 $a$ 的概率。

### 2.2 逻辑回归

LR的似然定义为 $h_1(\mathbf{x}) = \Pr(y = 1 | \mathbf{x}, \mathbf{w}) = \frac{1}{1 + \exp(-\mathbf{w}^T \mathbf{x})}$，其中 $\mathbf{w}$ 是要学习的参数（权重向量）。请注意，通过向特征向量添加常数值1的额外特征，LR的偏置项已被集成到 $\mathbf{w}$ 中。给定训练集 $\{(\mathbf{x}^{(i)}, y^{(i)}) | i = 1, ..., N\}$，通过最小化以下正则化损失函数找到权重向量 $\mathbf{w}$：

$$
\min_{\mathbf{w}} \lambda \Omega_1(\mathbf{w}) + \sum_{i=1}^{N} \xi_1(\mathbf{w}; \mathbf{x}^{(i)}, y^{(i)})
$$

其中 $\xi_1(\mathbf{w}; \mathbf{x}^{(i)}, y^{(i)}) = -\log([h_1(\mathbf{x}^{(i)})]^{y^{(i)}} [1 - h_1(\mathbf{x}^{(i)})]^{1 - y^{(i)}})$，$\Omega_1(\mathbf{w})$ 是正则化项。

在真实应用中，我们可以使用以下L2范数进行正则化：$\Omega_1(\mathbf{w}) = \frac{1}{2} ||\mathbf{w}||_2^2 = \frac{\mathbf{w}^T \mathbf{w}}{2}$。得到的模型是标准LR模型。我们也可以使用以下L1范数进行正则化：$\Omega_1(\mathbf{w}) = ||\mathbf{w}||_1 = \sum_{i=1}^{z} |w_i|$，其中 $z$ 是向量 $\mathbf{w}$ 的长度。得到的模型将是Lasso，可用于特征选择或消除。

等式(1)中的优化函数易于实现且具有 promising 的性能，这使得LR在工业中非常受欢迎。请注意，在以下内容中，LR指具有L2范数正则化的LR模型，具有L1范数的LR将如许多文献中那样称为Lasso。

### 2.3 组 Lasso

组 Lasso 是一种在（预定义）变量组上进行变量选择的技术。对于参数向量 $\boldsymbol{\beta} \in \mathbb{R}^z$，组 Lasso 中的正则化项定义如下：

$$
\sum_{g=1}^{G} ||\boldsymbol{\beta}_{I_g}||_2
$$

其中 $I_g$ 是属于第 $g$ 个预定义变量组的索引集，$g = 1, 2, \cdots, G$。组 Lasso 可以与线性回归或逻辑回归一起作为惩罚使用。它因其在组级别进行变量选择的属性而具有吸引力，其中学习后某些组中的所有变量将为零。

## 3. 耦合组 Lasso

虽然LR已被广泛用于CTR预测，但它无法捕捉用户特征和广告特征之间的联合信息。一种可能的解决方案是从原始输入特征手动构建联合作为LR的输入。然而，如前所述，手动特征联合将导致二次数量的新特征，这使得学习参数变得异常困难。因此，LR的建模能力太弱，无法捕捉数据中的复杂关系。

在本节中，我们介绍我们的耦合组 Lasso（CGL）模型，它可以轻松建模用户和广告之间的联合信息，以实现比LR更好的性能。

### 3.1 模型

CGL的似然表述如下：

$$
h(\mathbf{x}) = \Pr(y = 1 | \mathbf{x}, \mathbf{W}, \mathbf{V}, \mathbf{b}) = \sigma \left( (\mathbf{x}_u^T \mathbf{W})(\mathbf{x}_a^T \mathbf{V})^T + \mathbf{b}^T \mathbf{x}_o \right)
$$

其中 $\mathbf{W}$ 是大小为 $l \times k$ 的矩阵，$\mathbf{V}$ 是大小为 $s \times k$ 的矩阵，$\mathbf{b}$ 是长度为 $d$ 的向量，$\sigma(x)$ 是 sigmoid 函数，$\sigma(x) = \frac{1}{1 + \exp(-x)}$。这里，$\mathbf{W}$、$\mathbf{V}$ 和 $\mathbf{b}$ 是要学习的参数，$k$ 是超参数。

此外，我们在负对数似然上放置正则化以得到以下CGL的优化问题：

$$
\min_{\mathbf{W}, \mathbf{V}, \mathbf{b}} \sum_{i=1}^{N} \xi \left( \mathbf{W}, \mathbf{V}, \mathbf{b}; \mathbf{x}^{(i)}, y^{(i)} \right) + \lambda \Omega(\mathbf{W}, \mathbf{V})
$$

其中

$$
\xi(\mathbf{W}, \mathbf{V}, \mathbf{b}; \mathbf{x}^{(i)}, y^{(i)}) = -\log \left( [h(\mathbf{x}^{(i)})]^{y^{(i)}} [1 - h(\mathbf{x}^{(i)})]^{1 - y^{(i)}} \right)
$$

$$
\Omega(\mathbf{W}, \mathbf{V}) = ||\mathbf{W}||_{2,1} + ||\mathbf{V}||_{2,1}
$$

这里，$||\mathbf{W}||_{2,1} = \sum_{i=1}^{l} \sqrt{\sum_{j=1}^{k} W_{ij}^2} = \sum_{i=1}^{l} ||\mathbf{W}_{i*}||_2$ 是矩阵 $\mathbf{W}$ 的L2,1范数。类似地，$||\mathbf{V}||_{2,1}$ 是矩阵 $\mathbf{V}$ 的L2,1范数。从等式(2)中，很容易发现L2,1范数实际上是组 Lasso 正则化，每行是一个组。请注意，我们不对 $\mathbf{b}$ 放置正则化，因为从实验中我们发现这种正则化不影响性能。

我们可以发现等式(4)中有两个组 Lasso，一个用于用户特征，另一个用于广告特征。此外，两个组 Lasso 耦合在一起以确定展示的CTR。因此，我们的模型称为耦合组 Lasso（CGL）。

因为 $(\mathbf{x}_u^T \mathbf{W})(\mathbf{x}_a^T \mathbf{V})^T = \mathbf{x}_u^T \mathbf{W} \mathbf{V}^T \mathbf{x}_a = \mathbf{x}_u^T (\mathbf{W} \mathbf{V}^T) \mathbf{x}_a$，非常有趣的是，等式(3)中的项 $(\mathbf{x}_u^T \mathbf{W})(\mathbf{x}_a^T \mathbf{V})^T$ 可以有效建模用户特征和广告特征之间的联合信息。$\mathbf{W}$ 和 $\mathbf{V}$ 中的参数数量仅为 $(l + s)k$，其中 $k$ 通常是一个小数字（在我们的实验中小于50）。相反，如果我们选择为LR手动构建联合特征，手动联合特征的数量为 $l \times s$，其中 $l$ 和 $s$ 在CTR预测中通常都是数万。因此，CGL的参数数量远少于具有手动联合特征的LR，这使得CGL在建模联合特征方面比LR更具可扩展性。此外，更少的参数数量将导致模型过拟合的概率更低。

CGL的另一个 nice 属性来自组 Lasso 的正则化项。很容易发现学习后 $\mathbf{W}$ 和 $\mathbf{V}$ 的某些行将全为零。因为每行对应用户或广告的一个特征，我们可以消除对应于全零参数值的特征。因此，当我们使用学习到的模型进行在线预测时，没有必要为用户和广告收集那些被消除的特征。这不仅可以节省内存，还可以加速在线预测过程。

### 3.2 学习

我们学习算法的目标是找到最优值 $\mathbf{b}^* \in \mathbb{R}^d$、$\mathbf{W}^* \in \mathbb{R}^{l \times k}$、$\mathbf{V}^* \in \mathbb{R}^{s \times k}$，以最小化等式(4)中的目标函数。$(\mathbf{x}_u^T \mathbf{W})(\mathbf{x}_a^T \mathbf{V})^T$ 的耦合部分使目标函数非凸。我们采用交替学习方法来学习参数。每次我们优化一个参数，其他参数固定。将重复几次迭代，直到满足某些终止条件。

更具体地说，我们首先固定广告参数 $\mathbf{V}$，并使用有限内存BFGS（L-BFGS）优化关于 $\mathbf{W}$ 和 $\mathbf{b}$ 的目标函数，直到收敛。然后我们固定用户参数 $\mathbf{W}$，并优化关于 $\mathbf{V}$ 和 $\mathbf{b}$ 直到收敛。显然，目标函数在其任何一个参数矩阵 $\mathbf{W}$ 或 $\mathbf{V}$ 中都是凸的。

L-BFGS算法属于拟牛顿方法家族。L-BFGS只存储几个梯度向量来近似Hessian矩阵。因此，它更适合具有大量变量的优化问题。为了使用L-BFGS算法，我们只需要计算参数的梯度。

为便于展示，我们使用 $\xi(\mathbf{x}, y)$ 表示等式(5)中的 $\xi(\mathbf{W}, \mathbf{V}, \mathbf{b}; \mathbf{x}, y)$，省略参数。对于每个实例 $(\mathbf{x}, y)$，该实例贡献的梯度可以推导如下：

$$
\frac{\partial \xi(\mathbf{x}, y)}{\partial b_i} = (h(\mathbf{x}) - y) x_{o_i}
$$

$$
\frac{\partial \xi(\mathbf{x}, y)}{\partial W_{ij}} = x_{u_i} (h(\mathbf{x}) - y) \mathbf{x}_a^T \mathbf{V}_{*j}
$$

$$
\frac{\partial \xi(\mathbf{x}, y)}{\partial V_{ij}} = x_{a_i} (h(\mathbf{x}) - y) \mathbf{x}_u^T \mathbf{W}_{*j}
$$

其中 $x_{u_i}$、$x_{a_i}$ 和 $x_{o_i}$ 分别表示向量 $\mathbf{x}_u$、$\mathbf{x}_a$ 和 $\mathbf{x}_o$ 中的第 $i$ 个元素。

等式(6)中的正则化部分可以展开如下：

$$
\Omega(\mathbf{W}, \mathbf{V}) = ||\mathbf{W}||_{2,1} + ||\mathbf{V}||_{2,1} = \sum_{i=1}^{l} \sqrt{\sum_{j=1}^{k} W_{ij}^2} + \sum_{i=1}^{s} \sqrt{\sum_{j=1}^{k} V_{ij}^2}
$$

$$
\approx \sum_{i=1}^{l} \sqrt{\sum_{j=1}^{k} W_{ij}^2 + \epsilon} + \sum_{i=1}^{s} \sqrt{\sum_{j=1}^{k} V_{ij}^2 + \epsilon}
$$

其中 $\epsilon$ 是一个非常小的正数，使正则化项可微。实际上，它在我们的应用中效果很好。

$\Omega(\mathbf{W}, \mathbf{V})$ 的梯度可以推导如下：

$$
\frac{\partial \Omega(\mathbf{W}, \mathbf{V})}{\partial W_{ij}} = \frac{W_{ij}}{\sqrt{\sum_{j=1}^{k} W_{ij}^2 + \epsilon}}
$$

$$
\frac{\partial \Omega(\mathbf{W}, \mathbf{V})}{\partial V_{ij}} = \frac{V_{ij}}{\sqrt{\sum_{j=1}^{k} V_{ij}^2 + \epsilon}}
$$

我们可以将参数组 $(\mathbf{W}, \mathbf{b})$ 连接成参数向量，然后计算梯度向量 $g(\mathbf{W}, \mathbf{b})$。类似地，我们也可以计算参数组 $(\mathbf{V}, \mathbf{b})$ 的梯度向量 $g(\mathbf{V}, \mathbf{b})$。假设 $t$ 是每个参数组中的第 $\tau$ 个参数，梯度向量 $g(\cdot)$ 中的第 $\tau$ 个元素具有以下形式：

$$
g_\tau(\cdot) = \sum_{i=1}^{N} \frac{\partial \xi(\mathbf{x}^{(i)}, y^{(i)})}{\partial t} + \lambda \frac{\partial \Omega(\mathbf{W}, \mathbf{V})}{\partial t}
$$

其中 $\frac{\partial \xi(\mathbf{x}^{(i)}, y^{(i)})}{\partial t}$ 使用等式(7)、(8)或(9)计算，$\frac{\partial \Omega(\mathbf{W}, \mathbf{V})}{\partial t}$ 使用等式(10)或(11)计算，取决于 $t$ 的值。然后我们可以为L-BFGS构建近似Hessian矩阵 $\tilde{\mathbf{H}}$。

CGL的整个学习过程总结在算法1中。在每次迭代中，交替学习算法确保目标函数值总是减少。此外，目标函数以0为下界。实际上，当整个损失函数趋于平坦且减少变得平坦时，我们可以将其视为收敛。因为目标函数在 $\mathbf{W}$ 和 $\mathbf{V}$ 中不是联合凸的，所以解是局部最优。在我们的实现中，算法的收敛条件是目标函数值的相对减少小于阈值。

**算法1：CGL的交替学习**

$$
\begin{aligned}
&\textbf{输入：} \text{数据集 } \{(\mathbf{x}^{(i)}, y^{(i)}) | i = 1, ..., N\} \text{，超参数 } k \in \mathbb{N}^+ \text{ 和 } \lambda \in \mathbb{R}^+ \\
&\textbf{输出：} \mathbf{W}^*, \mathbf{V}^*, \mathbf{b}^* \\
&\text{初始化 } \mathbf{b} = 0 \\
&\text{初始化 } \mathbf{W} = \text{random}(\mathbb{R}^{l \times k}), \mathbf{V} = \text{random}(\mathbb{R}^{s \times k}) \\
&\textbf{repeat} \\
&\quad \text{固定 } \mathbf{V} \\
&\quad \textbf{repeat} \\
&\quad\quad \text{使用等式(12)计算梯度 } g(\mathbf{W}, \mathbf{b}) \\
&\quad\quad \text{计算关于 } (\mathbf{W}, \mathbf{b}) \text{ 的近似Hessian } \tilde{\mathbf{H}}_{\mathbf{W},\mathbf{b}} \\
&\quad\quad \mathbf{d}(\mathbf{W}, \mathbf{b}) = -\tilde{\mathbf{H}}_{\mathbf{W},\mathbf{b}} * g(\mathbf{W}, \mathbf{b}) \\
&\quad\quad \text{在方向 } \mathbf{d}(\mathbf{W}, \mathbf{b}) \text{ 上执行线搜索并更新 } \mathbf{W}, \mathbf{b} \\
&\quad \textbf{until } \mathbf{W}, \mathbf{b} \text{ 收敛} \\
&\quad \text{固定 } \mathbf{W} \\
&\quad \textbf{repeat} \\
&\quad\quad \text{使用等式(12)计算梯度 } g(\mathbf{V}, \mathbf{b}) \\
&\quad\quad \text{计算关于 } (\mathbf{V}, \mathbf{b}) \text{ 的近似Hessian } \tilde{\mathbf{H}}_{\mathbf{V},\mathbf{b}} \\
&\quad\quad \mathbf{d}(\mathbf{V}, \mathbf{b}) = -\tilde{\mathbf{H}}_{\mathbf{V},\mathbf{b}} * g(\mathbf{V}, \mathbf{b}) \\
&\quad\quad \text{在方向 } \mathbf{d}(\mathbf{V}, \mathbf{b}) \text{ 上执行线搜索并更新 } \mathbf{V}, \mathbf{b} \\
&\quad \textbf{until } \mathbf{V}, \mathbf{b} \text{ 收敛} \\
&\textbf{until } \text{收敛}
\end{aligned}
$$

### 3.3 复杂度分析

令 $q = (l + s)k + d$ 表示 $\mathbf{W}$、$\mathbf{V}$ 和 $\mathbf{b}$ 中的总参数数量。为了训练模型，我们需要 $O(qN)$ 时间计算梯度 $g(\cdot)$，$O(q^2)$ 时间计算近似Hessian矩阵，$O(q^2)$ 时间进行矩阵乘法和参数更新。因此，对于 $\mu$ 次迭代，总时间复杂度为 $O(qN + q^2)\mu$。

## 4. Web-Scale实现

Web-Scale应用总是包含大量用户和广告，有数十亿展示实例。因此，我们需要可扩展的学习框架。在本节中，我们首先介绍用于内存节省和类不平衡处理的哈希和子采样技术。此外，我们提出了基于消息传递接口（MPI）的分布式学习框架，可以在具有数百个计算节点的集群上运行。通过结合这些技术，我们的学习算法对于Web-Scale应用具有高度可扩展性。

### 4.1 哈希和子采样

我们使用哈希技术进行有效的特征映射和实例生成。真实CTR预测系统中使用的原始特征主要是分类的，其数量通常非常大。为了使特征映射（编码）结果均匀并实现实例生成，我们将用户、广告和其他（上下文）特征哈希到位向量的三个单独子空间中。哈希框架的结构如图1所示，其中展示的原始表示被哈希到位向量的实例表示。每个展示的原始表示由几个三元组组成，每个三元组是（域、特征名、特征值）。域可以是用户、广告或其他，指的是特征的类型。例如，图1中的展示包含 $k$ 个三元组，$(d_1, f_1, v_1)$，$(d_2, f_2, v_2)$，...，$(d_k, f_k, v_k)$。每个展示的实例表示由位向量的三个子空间组成，在图1中表示为用户特征、广告特征和其他特征。给定一个三元组，我们可以使用哈希函数快速获得其在特征空间中编码的索引（也称为位置或键）。例如，如果一个展示有一个用户（域）特征"buypower"（特征名），值为"5"（特征值），哈希函数将这个三元组（用户，buypower，5）映射到用户特征子空间中的一个位置，并将相应位置设置为1。<位置（键），三元组>对存储在特征映射中，以便稍后搜索和检查。哈希技术实现了有效的特征工程和快速预测以及直接的实现。根据研究，哈希不仅可以因碰撞而压缩特征空间，还可以带来正则化效果。

数据集通常高度不平衡，只有很小比例的正实例。为了在对准确性影响最小的情况下降低学习复杂度，我们以 $\gamma = 10\%$ 的概率对负实例进行采样，并保留所有正实例。采样后，我们在学习过程中给每个负实例赋权重 $\frac{1}{\gamma}$，以使目标计算无偏。

### 4.2 分布式学习

在算法1的每次迭代中，我们需要使用等式(12)计算所有参数的梯度，并计算等式(4)中目标函数的当前值。这两个等式都包含对所有实例的求和，可以很容易地在一台机器（节点）上使用多线程技术并行化，或者使用MPI在集群中的几台机器上分布式。

我们基于具有数百个节点（机器）的MPI集群实现了CGL的分布式学习框架，并充分利用了参数梯度计算的并行属性。令 $P$ 为节点数。我们首先将整个数据集均匀分配到每个节点，数据分配不会随着算法的进行而改变，这可以最小化数据移动和通信成本。

MPI中的AllReduce和BroadCast接口用于在主节点和从节点之间进行通信。此外，使用同步例程来确保我们系统的正确性。我们将节点 $p$ 上参数的本地梯度表示为 $\mathbf{g}_p^0$。请注意，$\mathbf{g}_p^0$ 只包含实例部分的梯度，正则化部分被丢弃。梯度向量 $\mathbf{g}_p^0$ 中的第 $\tau$ 个元素具有以下形式：

$$
g_{p\tau}^0 = \sum_{i=1}^{p_n} \frac{\partial \xi(\mathbf{x}^{(i)}, y^{(i)})}{\partial t}
$$

其中 $p_n$ 是分配给节点 $p$ 的实例数，$t$ 是等式(12)中所述的第 $\tau$ 个参数。

分布式学习框架的概要总结在算法2中。

**算法2：CGL的分布式学习框架**

$$
\begin{aligned}
&\textbf{输入：} \text{数据集 } \{(\mathbf{x}^{(i)}, y^{(i)}) | i = 1, ..., N\} \text{，超参数 } k \in \mathbb{N}^+ \text{ 和 } \lambda \in \mathbb{R}^+ \\
&\text{初始化节点数：} P \\
&\text{初始化参数：} \mathbf{W}, \mathbf{V}, \mathbf{b} \\
&\text{将数据集均匀分割到 } P \text{ 个节点} \\
&\textbf{repeat} \\
&\quad \textbf{for all nodes } \{p = 1, 2, \cdots, P\} \textbf{ do in parallel} \\
&\quad\quad \text{使用等式(13)在节点 } p \text{ 上本地计算梯度 } \mathbf{g}_p^0 \\
&\quad \textbf{end for} \\
&\quad \text{使用 AllReduce 计算梯度 } \mathbf{g}^0 = \sum_{p=1}^{P} \mathbf{g}_p^0 \\
&\quad \text{在主节点中将正则化项的梯度添加到 } \mathbf{g}^0 \\
&\quad \text{在主节点中执行L-BFGS步骤} \\
&\quad \text{使用 BroadCast 将更新的参数广播到每个从节点} \\
&\textbf{until } \text{收敛}
\end{aligned}
$$

## 5. 实验

我们有一个具有数百个节点的MPI集群，每个节点是具有2.2GHz Intel(R) Xeon(R) E5-2430处理器和96GB RAM的24核服务器。然而，我们只使用集群中的80个节点进行实验。一个原因是80个节点足以处理真实的Web-Scale CTR应用。另一个原因是整个集群由许多运行不同作业的组共享。我们不想中断其他作业。然而，我们的方法对于使用更多节点处理更大规模问题具有高度可扩展性，这将通过我们以下的实验验证（参见图4）。

### 5.1 数据集

我们在从阿里巴巴淘宝收集的三个真实世界数据集上进行实验。这三个数据集的训练集包含不同时间段不同时间窗口大小的展示广告日志信息，每个训练集的后续（下一天）日志信息用于测试以评估我们模型的性能。我们在不同月份的工作日或假期组成数据集，使数据集彼此不同，并使我们模型的结果更有说服力。每个数据集有数十亿展示实例，每个展示的特征向量维度为数万。这三个数据集分别命名为Dataset-1、Dataset-2和Dataset-3。Train 1和Test 1是Dataset-1的训练集和测试集。其他两个数据集可以得到类似的名称。这三个数据集的特征简要总结在表1中。请注意，表1中数据集的CTR计算为点击数/展示数，这实际上是数据集中正实例的比例。我们可以发现所有数据集都高度不平衡，只有很小比例的正实例。很容易发现我们实验的数据集大小是Web-Scale的。

我们从每个训练集中采样20%用于验证，以指定我们CGL模型和其他基线的超参数。除非另有说明，CGL中的 $k$ 在我们的实验中固定为50。

| 数据集 | 实例数（十亿） | CTR（%） | 广告数 | 用户数（百万） | 存储（TB） |
|--------|---------------|----------|--------|---------------|-----------|
| Train 1 | 1.011 | 1.62 | 21,318 | 874.7 | 1.895 |
| Test 1 | 0.295 | 1.70 | 11,558 | 331.0 | 0.646 |
| Train 2 | 1.184 | 1.61 | 21,620 | 958.6 | 2.203 |
| Test 2 | 0.145 | 1.64 | 6,848 | 190.3 | 0.269 |
| Train 3 | 1.491 | 1.75 | 33,538 | 1119.3 | 2.865 |
| Test 3 | 0.126 | 1.70 | 9,437 | 183.7 | 0.233 |

**表1：三个数据集的特征，分别包含来自不同时间段的4天、10天和7天的训练数据。每个训练集的后续一天的日志信息用于测试集。**

### 5.2 评估指标和基线

#### 5.2.1 指标

我们可以将CTR预测视为二分类问题。因为数据集高度不平衡，只有很小比例的正实例，预测准确性不是评估的好指标。此外，精确率和召回率都不是好指标。在本文中，我们采用接收者操作特征曲线下面积（AUC）作为衡量预测准确性的指标，这在现有CTR预测文献中已被广泛使用。对于随机猜测者，AUC值将为0.5，意味着完全缺乏区分能力。为了与基线模型进行良好比较，我们首先从AUC值中移除此常数部分（0.5），然后计算我们模型的相对改进（RelaImpr），其具有以下数学形式：

$$
\text{RelaImpr} = \frac{\text{AUC}(\text{model}) - 0.5}{\text{AUC}(\text{baseline}) - 0.5} \times 100\%
$$

这个RelaImpr指标实际上已在工业中被广泛采用用于比较模型的区分能力。

我们的CGL模型具有为用户和广告选择特征或消除特征的效果。我们引入组稀疏度（GSparsity）来衡量我们模型在特征消除中的能力：$\text{GSparsity} = \frac{\nu}{l + s} \times 100\%$，其中 $\nu$ 是参数矩阵 $\mathbf{W}$ 和 $\mathbf{V}$ 中全零行的总数，$l$ 和 $s$ 分别是 $\mathbf{W}$ 和 $\mathbf{V}$ 中的行数。

#### 5.2.2 基线

因为LR模型（具有L2范数）已被广泛用于CTR预测并实现了最先进的性能，特别是在工业系统中，我们采用LR作为比较的基线。请注意，LR指等式(1)中具有L2范数正则化的模型，等式(1)中具有L1范数正则化的模型在本文中称为Lasso。

### 5.3 Lasso的准确性

表2是Lasso相对于基线（LR）的相对改进（RelaImpr）。我们可以看到LR和Lasso在预测准确性方面不存在显著差异。

| 数据集 | RelaImpr |
|--------|----------|
| Dataset-1 | -0.019% |
| Dataset-2 | -0.096% |
| Dataset-3 | +0.086% |

**表2：Lasso相对于基线（LR）的相对改进。**

### 5.4 CGL的准确性

请注意，在算法1中，$\mathbf{W}$ 和 $\mathbf{V}$ 是随机初始化的，这可能影响性能。我们使用不同初始化进行六轮独立实验。我们CGL模型（$k = 50$）相对于基线LR的相对改进的均值和方差报告在图2中。很容易发现我们的CGL模型在所有三个数据集上都能显著优于LR。此外，我们可以发现随机初始化对性能的影响可以忽略。因此，在以下实验中，我们不会报告值的方差。

**图2：CGL相对于基线（LR）的相对改进。**

### 5.5 对超参数的敏感性

在本小节中，我们研究CGL模型中两个关键超参数 $k$ 和 $\lambda$ 的影响。

对不同的 $k$ 进行实验，结果如图3(a)所示。我们可以发现，随着 $k$ 的增加，性能总体上变得更好。但更大的 $k$ 意味着更多参数，这会使学习在内存和速度方面都更加困难。我们发现 $k = 50$ 是我们实验的合适值。因此，我们在本文的所有实验中选择 $k = 50$。

我们改变超参数 $\lambda$ 的值，并在图3(b)中绘制对性能的影响。我们可以发现，当 $\lambda$ 在1左右时，所有数据集都可以实现非常好的性能，并且我们的CGL在相对较大的范围（从0.1到10）内对 $\lambda$ 不敏感。

实际上，$\lambda$ 控制着预测准确性与消除特征数量（GSparsity）之间的权衡。我们选择Dataset-2进行演示。Dataset-2上相对改进和GSparsity的关系如表3所示。我们可以发现我们的CGL确实具有消除某些特征的能力。对于此数据集，3%-15%的GSparsity将是特征消除和预测准确性之间的良好权衡。

| GSparsity | RelaImpr |
|-----------|----------|
| 2% | 3.90% |
| 3% | 3.42% |
| 5% | 3.02% |
| 15% | 2.50% |
| 20% | 1.97% |

**表3：Dataset-2上相对于基线（LR）的性能改进与GSparsity之间的权衡。**

我们深入查看用户和广告的参数矩阵，并在表4中显示最重要的特征和最无用的特征。它们都是分类特征，其中广告部分最重要的特征是流行或热门产品类别，如衣服、裙子和连衣裙。这似乎是合理的。广告部分的无用特征包括电影、活动、外卖、食品预订服务。这也是合理的，因为很少有用户从淘宝购买这些产品如食品预订服务。用户部分最重要的特征是他们表现出极大兴趣的类别，如日用品和衣服。用户的无用特征是一些冷门类别和一些稀有物品，如舞台服装和地板。

| 部分 | 重要特征 | 无用特征 |
|------|----------|----------|
| 广告 | 女装、裙子、连衣裙、童装、鞋子、手机、手表、内衣、皮草服装、家具 | 电影、活动、外卖、食品预订服务、舞台服装、地板、铅笔、户外袜 |

**表4：特征选择结果。**

### 5.6 可扩展性

为了研究我们分布式学习框架的可扩展性，我们通过将节点数从20变化到80来计算相对于20个节点运行时间的加速因子。实验重复几次，加速因子的均值和方差报告在图4中。我们可以发现加速接近线性，接近理想的加速因子。因此，我们的CGL对于Web-Scale应用非常可扩展。

**图4：分布式学习框架的加速。**

CGL在80个节点上的整个训练时间（秒）如表5所示，从中我们可以发现CGL足够快以处理Web-Scale应用。

| 数据集 | 时间（秒） |
|--------|-----------|
| Dataset-1 | 3,184 ± 431 |
| Dataset-2 | 3,296 ± 387 |
| Dataset-3 | 4,281 ± 541 |

**表5：训练时间（秒）。**

## 6. 结论与未来工作

在本文中，提出了一个新颖的模型CGL来捕捉特征之间的联合信息，它可以在展示广告的Web-Scale CTR预测中优于最先进的模型。实际上，我们的CGL足够通用，可以建模其他类似的具有由两个交互角色决定的输出的应用。我们未来的其中一个工作是追求我们模型的新应用。

## 7. 致谢

这项工作得到NSFC（No. 61100125）、中国863计划（No. 2012AA011003）和中国大学创新研究团队计划（IRT1158, PCSIRT）的支持。Wu-Jun Li是通讯作者。

## 参考文献

[1] Agarwal, Deepak, Agrawal, Rahul, Khanna, Rajiv, and Kota, Nagaraj. Estimating rates of rare events with multiple hierarchies through scalable log-linear models. In KDD, pp. 213–222, 2010.

[2] Andrew, Galen and Gao, Jianfeng. Scalable training of l1-regularized log-linear models. In ICML, pp. 33–40, 2007.

[3] Bradley, Andrew P. The use of the area under the roc curve in the evaluation of machine learning algorithms. Pattern Recognition, 30(7):1145–1159, 1997.

[4] Broyden, C. G. The convergence of a class of double-rank minimization algorithms 1. general considerations. IMA Journal of Applied Mathematics, 6(1):76–90, 1970.

[5] Byrd, Richard H., Nocedal, Jorge, and Schnabel, Robert B. Representations of quasi-newton matrices and their use in limited memory methods. Mathematical Programming, 63:129–156, 1994.

[6] Chapelle, Oliver, Manavoglu, Eren, and Rosales, Romer. Simple and scalable response prediction for display advertising. ACM Transactions on Intelligent Systems and Technology, 2013.

[7] Golub, Gene H, Hansen, Per Christian, and O'Leary, Dianne P. Tikhonov regularization and total least squares. SIAM Journal on Matrix Analysis and Applications, 21(1):185–194, 1999.

[8] Graepel, Thore, Candela, Joaquin Quiñonero, Borchert, Thomas, and Herbrich, Ralf. Web-scale bayesian click-through rate prediction for sponsored search advertising in microsoft's bing search engine. In ICML, pp. 13–20, 2010.

[9] Kuang-chih, Lee, Orten, Burkay, Dasdan, Ali, and Li, Wentong. Estimating conversion rate in display advertising from past performance data. In KDD, pp. 768–776, 2012.

[10] Mahdian, Mohammad and Tomak, Kerem. Pay-per-action model for online advertising. In WINE, pp. 549–557, 2007.

[11] Malouf, Robert. A comparison of algorithms for maximum entropy parameter estimation. In CoNLL, pp. 49–55, 2002.

[12] McMahan, H. Brendan, Holt, Gary, Sculley, David, Young, Michael, Ebner, Dietmar, Grady, Julian, Nie, Lan, Phillips, Todd, Davydov, Eugene, Golovin, Daniel, Chikkerur, Sharat, Liu, Dan, Wattenberg, Martin, Hrafnkelsson, Arnar Mar, Boulos, Tom, and Kubica, Jeremy. Ad click prediction: a view from the trenches. In KDD, pp. 1222–1230, 2013.

[13] Meier, Lukas, Van De Geer, Sara, and Bühlmann, Peter. The group lasso for logistic regression. Journal of the Royal Statistical Society, Series B, 70(1):53–71, 2008.

[14] Menon, Aditya Krishna, Chitrapura, Krishna Prasad, Garg, Sachin, Agarwal, Deepak, and Kota, Nagaraj. Response prediction using collaborative filtering with hierarchies and side-information. In KDD, pp. 141–149, 2011.

[15] Muthukrishnan, S. Ad exchanges: Research issues. In WINE, pp. 1–12, 2009.

[16] Neter, John, Wasserman, William, and Kutner, Michael H. Applied linear statistical models, volume 4. Irwin Chicago, 1996.

[17] Nocedal, Jorge. Updating quasi-newton matrices with limited storage. Mathematics of Computation, 35(151):773–782, 1980.

[18] Richardson, Matthew, Dominowska, Ewa, and Ragno, Robert. Predicting clicks: estimating the click-through rate for new ads. In WWW, pp. 521–530, 2007.

[19] Stern, David H., Herbrich, Ralf, and Graepel, Thore. Matchbox: large scale online bayesian recommendations. In WWW, pp. 111–120, 2009.

[20] Tibshirani, Robert. Regression shrinkage and selection via the lasso. Journal of the Royal Statistical Society. Series B, pp. 267–288, 1996.

[21] Weinberger, Kilian Q., Dasgupta, Anirban, Langford, John, Smola, Alexander J., and Attenberg, Josh. Feature hashing for large scale multitask learning. In ICML, pp. 140, 2009.

[22] Yuan, Ming and Lin, Yi. Model selection and estimation in regression with grouped variables. Journal of the Royal Statistical Society, Series B, 68:49–67, 2006.
