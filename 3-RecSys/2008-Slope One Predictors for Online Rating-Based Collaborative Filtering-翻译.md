# Slope One 预测器：基于评分的在线协同过滤

> Daniel Lemire, Anna Maclachlan | Université du Québec à Montréal, University of Prince Edward Island

本文分享了 Slope One 系列协同过滤（CF，Collaborative Filtering）预测器，该系列算法基于 $f(x) = x + b$ 形式的线性回归模型，通过预先计算 item 间的平均评分差异来进行评分预测。核心内容：

- 提出了三种 Slope One 方案：基本 Slope One、加权 Slope One 和双极 Slope One，这些方案简单、易实现、支持在线查询和动态更新
- 通过将用户喜欢的 item 与不喜欢的 item 分开处理，双极 Slope One 方案进一步提升了预测精度
- 在 EachMovie 和 MovieLens 数据集上的实验表明，Slope One 方案在精度上与复杂得多的基于记忆的协同过滤方案（如 Pearson 方案）相当

关键发现：

- Slope One 方案在 EachMovie 和 MovieLens 数据集上取得了与 Pearson 记忆基方案几乎相同的精度（MovieLens 上 MAE 1.90 vs 1.88），同时更简单、更易维护、支持动态更新
- 将评分划分为"喜欢"和"不喜欢"两个子集可以提高 1.5%-2% 的预测精度
- 基本 Slope One 即可超越 BIAS FROM MEAN 等基线方案，加权方案和双极方案进一步提升了性能

---

## 摘要

基于评分的协同过滤（Rating-based Collaborative Filtering）是根据其他用户的评分来预测某个用户将如何评价给定item的过程。我们提出了三种相关的Slope One方案，其预测器形式为 $f(x) = x + b$ ，这些方案预先计算了同时对两个item进行评分的用户对这两个item评分之间的平均差异。Slope One算法易于实现、查询效率高、精度合理，并且支持在线查询和动态更新，这使其成为现实世界系统的理想候选。我们建议将基本的SLOPE ONE方案作为协同过滤的新参考方案。通过将用户喜欢的item与用户不喜欢的item分开处理，我们获得了与在标准基准数据集EachMovie和MovieLens上速度较慢的记忆基方案相竞争的结果，同时更好地满足了协同过滤应用的各项需求。

**关键词：** Collaborative Filtering, Recommender System, e-Commerce, Data Mining, Knowledge Discovery

## 1 引言

在线基于评分的协同过滤（CF，Collaborative Filtering）查询由单个用户的一组（item，评分）对组成。对该查询的响应是一组预测的（item，评分）对，用于该用户尚未评分的item。我们的目标是提供稳健的CF方案，这些方案应满足：

1. **易于实现和维护：** 所有聚合数据应能被普通工程师轻松解释，算法应易于实现和测试；
2. **可动态更新：** 新增一个评分应能立即改变所有预测；
3. **查询时高效：** 查询应快速，必要时可以牺牲存储空间；
4. **对新用户要求低：** 评分很少的用户也应获得有效的推荐；
5. **精度合理：** 方案应与最精确的方案具有竞争力，但精度的微小提升并不总是值得在简单性或可扩展性上做出重大牺牲。

本文的目标不是比较各种CF算法的精度，而是证明Slope One方案能够同时满足上述所有五个目标。尽管我们的方案简单、可更新、计算高效且可扩展，但它们在精度上与那些放弃了部分其他优势的方案相当。

我们的Slope One算法基于一个直观原则——用户对item间的"流行度差异"。我们以成对方式确定一个item比另一个item更受欢迎的程度。衡量这种差异的一种方法就是简单地计算两个item的平均评分之差。反过来，这种差异可用于根据用户对另一个item的评分来预测该用户对其中一个item的评分。考虑两个用户A和B、两个item $I$ 和 $J$ 以及图1。用户A给item $I$ 的评分为1，用户B给item $I$ 的评分为2，而用户A给item $J$ 的评分为1.5。我们观察到item $J$ 比item $I$ 高 $1.5 - 1 = 0.5$ 分，因此我们可以预测用户B将给item $J$ 打 $2 + 0.5 = 2.5$ 分。我们称用户B为被预测用户，item $J$ 为被预测item。训练集中对每个未知评分存在许多这样的差异，我们取这些差异的平均值。本文提出的Slope One方案系列源于我们选择相关差异以获得单一预测的三种方式。

本文的主要贡献是提出了Slope One CF预测器，并证明它们与记忆基方案相比具有几乎相同的精度，同时更适合CF任务，从而证明了 $f(x) = x + b$ 形式的预测器可以与记忆基方案竞争。这是一个重要的结果。

## 2 相关工作

### 2.1 记忆基方案

记忆基协同过滤使用用户对之间的相似度度量来构建预测，通常通过加权平均实现[2, 12, 13, 18]。所选相似度度量决定了预测的精度，已有多种替代方案被研究[8]。记忆基CF的一些潜在缺点包括可扩展性和对数据稀疏性的敏感性。通常，依赖用户间相似度的方案无法预先计算以用于快速在线查询。另一个关键问题是记忆基方案必须计算用户之间的相似度度量，这通常要求有一定数量的用户（例如至少100个用户）输入了某些最低数量的评分（例如至少20个评分），包括当前用户。我们将我们的方案与一个著名的记忆基方案——Pearson方案——进行对比。

### 2.2 模型基方案

存在许多基于模型的CF方法。一些基于线性代数（SVD、PCA或特征向量）[3, 6, 7, 10, 15, 16]；另一些则更直接地借鉴了人工智能技术，如贝叶斯方法、潜在类别和神经网络[1, 2, 9]；还有一些基于聚类[4, 5]。与记忆基方案相比，模型基CF算法通常在查询时更快，尽管它们可能具有昂贵的训练或更新阶段。当查询速度至关重要时，模型基方案可能优于记忆基方案。

我们可以用以下代数形式将我们的预测器与文献中描述的某些预测器进行比较。我们的预测器形式为 $f(x) = x + b$ ，因此得名"slope one"（斜率为一），其中 $b$ 是常数， $x$ 是表示评分值的变量。对于任意一对item，我们试图找到最佳函数 $f$ ，使其能够根据一个item的评分预测另一个item的评分。这个函数对于每对item可能不同。一个CF方案将对预测器生成的多个预测进行加权。在[14]中，作者考虑了item对之间的相关性，然后推导出用户评分的加权平均值作为预测器。在他们算法的简单版本中，预测器形式为 $f(x) = x$ 。在基于回归的版本中，预测器形式为 $f(x) = ax + b$ 。在[17]中，作者也采用了 $f(x) = ax + b$ 形式的预测器。这两篇论文工作的一个自然扩展是考虑 $f(x) = ax^2 + bx + c$ 形式的预测器。然而，在本文中，我们使用 $f(x) = x + b$ 形式的朴素预测器。我们也使用朴素加权。[14]中观察到，即使他们基于回归的 $f(x) = ax + b$ 算法也没有带来相对于记忆基算法的显著改进。因此，我们证明 $f(x) = x + b$ 形式的预测器可以与记忆基方案竞争，这本身就是一个重要结果。

## 3 CF算法

我们提出了三种新的CF方案，并将我们的方案与四种参考方案进行对比：PER USER AVERAGE（用户平均）、BIAS FROM MEAN（均值偏差）、ADJUSTED COSINE ITEM-BASED（调整余弦item基，一种模型基方案）以及PEARSON方案（记忆基方案的代表）。

### 3.1 符号说明

在描述方案时，我们使用以下符号。来自某个用户的评分（称为一个评价）表示为一个不完整的数组 $u$ ，其中 $u_i$ 是该用户对 item $i$ 的评分。在 $u$ 中被评分item的子集表示为 $S(u)$ 。训练集中所有评价的集合为 $\chi$ 。集合 $S$ 中的元素数量为 $\mathrm{card}(S)$ 。评价 $u$ 中评分的平均值记为 $\bar{u}$ 。集合 $S_i(\chi)$ 是所有包含 item $i$ （ $i \in S(u)$ ）的评价 $u \in \chi$ 的集合。给定两个评价 $u$ 和 $v$ ，我们定义标量积 $\langle u, v \rangle$ 为 $\sum_{i \in S(u) \cap S(v)} u_i v_i$ 。预测结果 $P(u)$ 表示一个向量，其中每个分量对应一个item的预测值：预测结果隐式地依赖于训练集 $\chi$ 。

### 3.2 基线方案

最基本的预测算法之一是 PER USER AVERAGE（用户平均）方案，其公式为 $P(u) = \bar{u}$ 。即，我们预测用户将根据其平均评分来评价所有item。

另一个简单方案称为 BIAS FROM MEAN（均值偏差，有时也称为 NON PERSONALIZED[8]），其公式为：

$$
P(u)_i = \bar{u} + \frac{1}{\mathrm{card}(S_i(\chi))} \sum_{v \in S_i(\chi)} (v_i - \bar{v})
$$

即，该预测基于用户平均评分加上训练集中所有用户对特定item与其自身平均值的平均偏差。

我们还将与效果最好的item基方法[14]进行比较，该方法使用以下调整余弦相似度度量，给定两个item $i$ 和 $j$ ：

$$
\mathrm{sim}_{ij} = \frac{\sum_{u \in S_{ij}(\chi)} (u_i - \bar{u})(u_j - \bar{u})}{\sqrt{\sum_{u \in S_{ij}(\chi)} (u_i - \bar{u})^2 \cdot \sum_{u \in S_{ij}(\chi)} (u_j - \bar{u})^2}}
$$

预测结果通过以下加权和获得：

$$
P(u)_i = \frac{\sum_{j \in S(u)} |\mathrm{sim}_{ij}|(\alpha_{ij} u_j + \beta_{ij})}{\sum_{j \in S(u)} |\mathrm{sim}_{ij}|}
$$

其中回归系数 $\alpha_{ij}$ 、 $\beta_{ij}$ 通过最小化 $\sum_{u \in S_{ij}(\chi)} (\alpha_{ij} u_j + \beta_{ij} - u_i)^2$ 来选取，其中 $i$ 和 $j$ 固定。

### 3.3 PEARSON参考方案

由于我们希望证明我们的方案在预测能力上与记忆基方案相当，我们选择实现一个此类方案作为该类别的代表，并承认存在许多该类型的已有文献记载的方案。在最流行和精确的记忆基方案中，有 PEARSON 方案[13]。它采用对 $\chi$ 中所有用户的加权和形式：

$$
P(u)_i = \bar{u} + \frac{\sum_{v \in S_i(\chi)} \gamma(u, v)(v_i - \bar{v})}{\sum_{v \in S_i(\chi)} |\gamma(u, v)|}
$$

其中 $\gamma$ 是根据皮尔逊相关性计算的相似度度量：

$$
\mathrm{Corr}(u, w) = \frac{\langle u - \bar{u}, w - \bar{w} \rangle}{\sqrt{\sum_{i \in S(u) \cap S(w)} (u_i - \bar{u})^2 \cdot \sum_{i \in S(u) \cap S(w)} (w_i - \bar{w})^2}}
$$

按照[2, 8]的做法，我们设定：

$$
\gamma(u, w) = \mathrm{Corr}(u, w) \cdot |\mathrm{Corr}(u, w)|^{\rho-1}
$$

其中 $\rho = 2.5$ ， $\rho$ 是案例放大（Case Amplification）的幂次。案例放大减少了数据中的噪声：如果相关性高，例如 $\mathrm{Corr} = 0.9$ ，那么经过案例放大后仍然较高（ $0.9^{2.5} \approx 0.8$ ）；而如果相关性低，例如 $\mathrm{Corr} = 0.1$ ，那么它就变得可忽略不计（ $0.1^{2.5} \approx 0.003$ ）。[2]表明，皮尔逊相关性结合案例放大是一种精度合理的记忆基CF方案，尽管存在更精确的方案。

### 3.4 SLOPE ONE方案

Slope One方案同时考虑了来自其他对同一item评分的用户的信息（类似于ADJUSTED COSINE ITEM-BASED）和来自同一用户评分的其他item的信息（类似于PER USER AVERAGE）。然而，该方案也依赖于既不属于用户数组也不属于item数组的数据点（例如图1中用户A对itemI的评分），但这些数据点对于评分预测而言仍然是重要信息。该方法的大部分优势来自于那些未被单独纳入的数据。具体而言，只有那些与待预测用户共同评价过某个item的用户评分，以及只有那些待预测用户也已评分的item评分，才会进入Slope One方案下的评分预测。

形式化地，给定两个评价数组 $v_i$ 和 $w_i$ （其中 $i = 1, \ldots, n$ ），我们寻找最佳形式的预测器 $f(x) = x + b$ 来从 $v$ 预测 $w$ ，通过最小化 $\sum_i (v_i + b - w_i)^2$ 。对 $b$ 求导并令导数为零，得到 $b = \frac{\sum_i (w_i - v_i)}{n}$ 。换句话说，常数 $b$ 必须被选为两个数组之间的平均差异。这一结果启发了以下方案。

给定训练集 $\chi$ 以及任意两个item $j$ 和 $i$ ，它们在某些用户评价 $u$ 中分别具有评分 $u_j$ 和 $u_i$ （记为 $u \in S_{ji}(\chi)$ ），我们考虑 item $i$ 相对于 item $j$ 的平均偏差为：

$$
\mathrm{dev}_{ji} = \frac{\sum_{u \in S_{ji}(\chi)} (u_j - u_i)}{\mathrm{card}(S_{ji}(\chi))}
$$

注意，任何不同时包含 $u_j$ 和 $u_i$ 的用户评价 $u$ 都不会被纳入求和。由 $\mathrm{dev}_{ji}$ 定义的对称矩阵可以一次性计算完毕，并在新数据输入时快速更新。

鉴于 $\mathrm{dev}_{ji} + u_i$ 是根据 $u_i$ 对 $u_j$ 的一个预测，一个合理的预测器可能是所有此类预测的平均值：

$$
P(u)_j = \frac{1}{\mathrm{card}(R_j)} \sum_{i \in R_j} (\mathrm{dev}_{ji} + u_i)
$$

其中 $R_j = \{i \mid i \in S(u), i \neq j, \mathrm{card}(S_{ji}(\chi)) > 0\}$ 是所有相关item的集合。有一种近似方法可以简化此预测的计算。对于一个足够密集的数据集，其中几乎所有item对都有评分（即对于几乎所有 $i, j$ ，都有 $\mathrm{card}(S_{ji}(\chi)) > 0$ ），大多数情况下，对于 $j \notin S(u)$ 有 $R_j = S(u)$ ，而当 $j \in S(u)$ 时有 $R_j = S(u) - \{j\}$ 。由于对于大多数 $j$ ， $\bar{u} = \frac{\sum_{i \in S(u)} u_i}{\mathrm{card}(S(u))} \approx \frac{\sum_{i \in R_j} u_i}{\mathrm{card}(R_j)}$ ，我们可以将 SLOPE ONE 方案的预测公式简化为：

$$
P^{S1}(u)_j = \bar{u} + \frac{1}{\mathrm{card}(R_j)} \sum_{i \in R_j} \mathrm{dev}_{ji}
$$

有趣的是，我们实现的 SLOPE ONE 并不依赖于用户如何评价单个item，而仅依赖于用户的平均评分以及（至关重要的）用户评价了哪些item。

### 3.5 加权SLOPE ONE方案

SLOPE ONE的一个缺点是没有考虑观测到的评分数量。直观地说，要预测用户A对itemL的评分（已知用户A对itemJ和K的评分），如果有2000个用户对item对(J, L)进行了评分，而只有20个用户对item对(K, L)进行了评分，那么用户A对itemJ的评分很可能比用户A对itemK的评分更适合作为itemL的预测器。因此，我们将加权 SLOPE ONE 预测定义为以下加权平均：

$$
P^{wS1}(u)_j = \frac{\sum_{i \in S(u) - \{j\}} (\mathrm{dev}_{ji} + u_i) \cdot c_{ji}}{\sum_{i \in S(u) - \{j\}} c_{ji}}
$$

其中 $c_{ji} = \mathrm{card}(S_{ji}(\chi))$ 。

### 3.6 双极SLOPE ONE方案

加权方案倾向于频繁出现的评分模式而非不频繁的评分模式，现在我们考虑另一种特别相关的评分模式。我们通过将预测分成两部分来实现这一点。使用加权SLOPE ONE算法，我们根据用户喜欢的item推导出一个预测，再根据用户不喜欢的item推导出另一个预测。

给定一个评分范围（比如0到10），使用范围中值5作为阈值，将高于5分的item视为喜欢、低于5分的视为不喜欢似乎是合理的。如果用户的评分分布均匀，这种方法效果很好。然而，EachMovie数据集中超过70%的评分高于范围中值。由于我们希望支持所有类型的用户（包括均衡型、乐观型、悲观型和双峰型用户），我们使用用户的平均评分作为用户喜欢和不喜欢item之间的阈值。例如，乐观型用户喜欢他们评价的每一件item，我们假设他们不喜欢低于其平均评分的item。这个阈值确保我们的算法对每个用户都有合理数量的喜欢和不喜欢item。

再次参见图1，与往常一样，我们根据同时对itemI和J评分的用户（如用户A）对itemI的偏差，来预测用户B对itemJ的评分。双极SLOPE ONE方案进一步限制了用于预测的评分集合。首先，在item方面，只考虑两个喜欢item之间的偏差或两个不喜欢item之间的偏差。其次，在用户方面，只有那些同时对itemI和J进行了评分并且对itemI有着相同喜欢或不喜欢的用户对的偏差，才被用于预测itemJ的评分。

将每个用户拆分为用户喜欢和用户不喜欢，实际上使用户数量增加了一倍。然而，请注意，上述双极限制必然会减少预测计算中的评分总数。尽管在数据稀疏性问题下这种减少似乎会提高精度（这似乎违反直觉），但未能过滤掉不相关的评分可能带来更大的问题。关键在于，双极SLOPE ONE方案不会根据"用户A喜欢itemK而用户B不喜欢同一itemK"这一事实进行预测。

形式化地，我们将每个评价 $u$ 拆分为两个已评分item集合： $S^{\mathrm{like}}(u) = \{i \in S(u) \mid u_i > \bar{u}\}$ 和 $S^{\mathrm{dislike}}(u) = \{i \in S(u) \mid u_i < \bar{u}\}$ 。对于每对item $i, j$ ，将所有评价 $\chi$ 的集合拆分为 $S^{\mathrm{like}}_{ij} = \{u \in \chi \mid i, j \in S^{\mathrm{like}}(u)\}$ 和 $S^{\mathrm{dislike}}_{ij} = \{u \in \chi \mid i, j \in S^{\mathrm{dislike}}(u)\}$ 。使用这两个集合，我们计算喜欢item的偏差矩阵以及不喜欢item的偏差矩阵：

$$
\mathrm{dev}^{\mathrm{like}}_{ji} = \frac{\sum_{u \in S^{\mathrm{like}}_{ji}(\chi)} (u_j - u_i)}{\mathrm{card}(S^{\mathrm{like}}_{ji}(\chi))}
$$

$$
\mathrm{dev}^{\mathrm{dislike}}_{ji} = \frac{\sum_{u \in S^{\mathrm{dislike}}_{ji}(\chi)} (u_j - u_i)}{\mathrm{card}(S^{\mathrm{dislike}}_{ji}(\chi))}
$$

根据 $i$ 属于 $S^{\mathrm{like}}(u)$ 还是 $S^{\mathrm{dislike}}(u)$ ，基于item $i$ 的评分对item $j$ 的预测分别为 $p^{\mathrm{like}}_{ji} = \mathrm{dev}^{\mathrm{like}}_{ji} + u_i$ 或 $p^{\mathrm{dislike}}_{ji} = \mathrm{dev}^{\mathrm{dislike}}_{ji} + u_i$ 。双极 SLOPE ONE 方案由以下公式给出：

$$
P^{bpS1}(u)_j = \frac{\sum_{i \in S^{\mathrm{like}}(u) - \{j\}} p^{\mathrm{like}}_{ji} \cdot c^{\mathrm{like}}_{ji} + \sum_{i \in S^{\mathrm{dislike}}(u) - \{j\}} p^{\mathrm{dislike}}_{ji} \cdot c^{\mathrm{dislike}}_{ji}}{\sum_{i \in S^{\mathrm{like}}(u) - \{j\}} c^{\mathrm{like}}_{ji} + \sum_{i \in S^{\mathrm{dislike}}(u) - \{j\}} c^{\mathrm{dislike}}_{ji}}
$$

其中权重 $c^{\mathrm{like}}_{ji} = \mathrm{card}(S^{\mathrm{like}}_{ji}(\chi))$ 和 $c^{\mathrm{dislike}}_{ji} = \mathrm{card}(S^{\mathrm{dislike}}_{ji}(\chi))$ 与加权 SLOPE ONE 方案中的权重类似。

## 4 实验结果

给定CF算法的有效性可以被精确度量。为此，我们采用了留一法平均绝对误差（MAE，Mean Absolute Error）[2]。在计算MAE时，我们依次从测试集中的所有评价中隐藏单个评分，预测该隐藏评分，然后计算预测的平均误差。给定预测器P和来自用户的评价u，P在一组评价 $\chi^{\prime}$ 上的误差率由以下公式给出：

$$
\mathrm{MAE} = \frac{1}{\mathrm{card}(\chi^{\prime})} \sum_{u \in \chi^{\prime}} \left( \frac{1}{\mathrm{card}(S(u))} \sum_{i \in S(u)} |P(u^{(i)}) - u_i| \right)
$$

其中 $u^{(i)}$ 是隐藏了用户对第 $i$ 个item评分 $u_i$ 后的用户评价 $u$ 。

我们在 Compaq Research 提供的 EachMovie 数据集和明尼苏达大学 Grouplens Research Group 提供的 MovieLens 数据集上测试了我们的方案。这些数据来自电影评分网站，EachMovie 的评分范围为 0.0 到 1.0，步长为 0.2；MovieLens 的评分范围为 1 到 5，步长为 1。按照[8, 11]的做法，我们使用足够多的评价来构成总量为 50,000 个评分的训练集 $\chi$ ，并使用额外的一组评价构成总量至少 100,000 个评分的测试集 $\chi^{\prime}$ 。当预测值超出给定数据集的允许评分范围时，进行相应修正：例如，对于 EachMovie 的 0 到 1 评分范围，预测值 1.2 被解释为 1。由于 MovieLens 的评分范围是 EachMovie 的 4 倍，MovieLens 上的 MAE 除以 4 以使结果直接可比。

各种方案使用相同误差度量并在相同数据集上的结果总结在表1中。各种子结果在后续的图表中突出显示。

考虑各种基线方案的测试结果。不出所料，我们发现BIAS FROM MEAN在第3.2节所述的三个参考基线方案中表现最佳。然而有趣的是，第3.4节所述的基本SLOPE ONE方案的精度高于BIAS FROM MEAN。

第3.5节和第3.6节所述的基本SLOPE ONE的增强方案确实提高了在EachMovie上的精度。SLOPE ONE与加权SLOPE ONE之间存在微小差异（约1%）。将不喜欢和喜欢的评分分开可以将结果提高1.5%-2%。

最后，比较记忆基的PEARSON方案与三种Slope One方案。Slope One方案实现了与PEARSON方案相当的精度。这一结果足以支持我们的主张：Slope One方案尽管简单且具有其他理想特性，但仍然具有合理的精度。

| 方案 | EachMovie | MovieLens |
|------|-----------|-----------|
| 双极SLOPE ONE | 0.194 | 0.188 |
| 加权SLOPE ONE | 0.198 | 0.188 |
| SLOPE ONE | 0.200 | 0.188 |
| BIAS FROM MEAN | 0.203 | 0.191 |
| 调整余弦item基 | 0.209 | 0.198 |
| 用户平均 | 0.231 | 0.208 |
| PEARSON | 0.194 | 0.190 |

**表1：所有方案对比：EachMovie和MovieLens数据集上的留一法平均绝对误差率，数值越低越好。**

## 5 结论

本文表明，一种基于平均评分差异的易于实现的CF模型可以与更昂贵的记忆基方案相竞争。与当前使用的方案相比，我们的方法能够同时满足五个相互对抗的目标。Slope One方案易于实现、可动态更新、查询时高效、对新用户要求低，同时具有与其他常见方案相当的精度（例如，在MovieLens上MAE为1.90 vs 1.88）。考虑到所比较的记忆基方案的相对复杂性，这非常引人注目。我们方法的另一个创新之处在于，将评分分为不喜欢和喜欢子集可以成为提高精度的有效技术。希望本文提出的通用Slope One预测器能够作为CF社区的参考方案。

注意，截至2004年11月，加权SLOPE ONE是Bell/MSN网站inDiscover.net使用的协同过滤算法。

## 参考文献

[1] D. Billsus and M. Pazzani. Learning collaborative information filterings. In AAAI Workshop on Recommender Systems, 1998.

[2] J. S. Breese, D. Heckerman, and C. Kadie. Empirical analysis of predictive algorithms for collaborative filtering. In Fourteenth Conference on Uncertainty in AI. Morgan Kaufmann, July 1998.

[3] J. Canny. Collaborative filtering with privacy via factor analysis. In SIGIR 2002, 2002.

[4] S. H. S. Chee. Rectree: A linear collaborative filtering algorithm. Master's thesis, Simon Fraser University, November 2000.

[5] S. H. S.g Chee, J. H., and K. Wang. Rectree: An efficient collaborative filtering method. Lecture Notes in Computer Science, 2114, 2001.

[6] Petros Drineas, Iordanis Kerenidis, and Prabhakar Raghavan. Competitive recommendation systems. In Proc. of the thirty-fourth annual ACM symposium on Theory of computing, pages 82–90. ACM Press, 2002.

[7] K. Goldberg, T. Roeder, D. Gupta, and C. Perkins. Eigentaste: A constant time collaborative filtering algorithm. Information Retrieval, 4(2):133–151, 2001.

[8] J. Herlocker, J. Konstan, A. Borchers, and J. Riedl. An algorithmic framework for performing collaborative filtering. In Proc. of Research and Development in Information Retrieval, 1999.

[9] T. Hofmann and J. Puzicha. Latent class models for collaborative filtering. In International Joint Conference in Artificial Intelligence, 1999.

[10] K. Honda, N. Sugiura, H. Ichihashi, and S. Araki. Collaborative filtering using principal component analysis and fuzzy clustering. In Web Intelligence, number 2198 in Lecture Notes in Artificial Intelligence, pages 394–402. Springer, 2001.

[11] Daniel Lemire. Scale and translation invariant collaborative filtering systems. Information Retrieval, 8(1):129–150, January 2005.

[12] D. M. Pennock and E. Horvitz. Collaborative filtering by personality diagnosis: A hybrid memory- and model-based approach. In IJCAI-99, 1999.

[13] P. Resnick, N. Iacovou, M. Suchak, P. Bergstrom, and J. Riedl. Grouplens: An open architecture for collaborative filtering of netnews. In Proc. ACM Computer Supported Cooperative Work, pages 175–186, 1994.

[14] B. M. Sarwar, G. Karypis, J. A. Konstan, and J. Riedl. Item-based collaborative filtering recommender algorithms. In WWW10, 2001.

[15] B. M. Sarwar, G. Karypis, J. A. Konstan, and J. T. Riedl. Application of dimensionality reduction in recommender system - a case study. In WEBKDD '00, pages 82–90, 2000.

[16] B.M. Sarwar, G. Karypis, J. Konstan, and J. Riedl. Incremental svd-based algorithms for highly scaleable recommender systems. In ICCIT'02, 2002.

[17] S. Vucetic and Z. Obradovic. A regression-based approach for scaling-up personalized recommender systems in e-commerce. In WEBKDD '00, 2000.

[18] S.M. Weiss and N. Indurkhya. Lightweight collaborative filtering method for binary encoded data. In PKDD '01, 2001.
