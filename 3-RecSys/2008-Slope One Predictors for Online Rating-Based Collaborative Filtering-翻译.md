# Slope One Predictors for Online Rating-Based Collaborative Filtering（中文翻译）
本文分享了Slope One系列协同过滤预测器，该系列算法基于`f(x) = x + b`形式的线性回归模型，通过预先计算item间的平均评分差异来进行评分预测。核心内容：
- 提出了三种Slope One方案：基本Slope One、加权Slope One和双极Slope One，这些方案简单、易实现、支持在线查询和动态更新
- 通过将用户喜欢的item与不喜欢的item分开处理，双极Slope One方案进一步提升了预测精度
- 在EachMovie和MovieLens数据集上的实验表明，Slope One方案在精度上与复杂得多的基于记忆的协同过滤方案（如Pearson方案）相当

关键发现：
- Slope One方案在EachMovie和MovieLens数据集上取得了与Pearson记忆基方案几乎相同的精度（MovieLens上MAE 1.90 vs 1.88），同时更简单、更易维护、支持动态更新
- 将评分划分为"喜欢"和"不喜欢"两个子集可以提高1.5%-2%的预测精度
- 基本Slope One即可超越BIAS FROM MEAN等基线方案，加权方案和双极方案进一步提升了性能


> Daniel Lemire, Anna Maclachlan | Université du Québec à Montréal, University of Prince Edward Island

---

## 摘要

基于评分的协同过滤（Rating-based Collaborative Filtering）是根据其他用户的评分来预测某个用户将如何评价给定item的过程。我们提出了三种相关的Slope One方案，其预测器形式为 `f(x) = x + b`，这些方案预先计算了同时对两个item进行评分的用户对这两个item评分之间的平均差异。Slope One算法易于实现、查询效率高、精度合理，并且支持在线查询和动态更新，这使其成为现实世界系统的理想候选。我们建议将基本的SLOPE ONE方案作为协同过滤的新参考方案。通过将用户喜欢的item与用户不喜欢的item分开处理，我们获得了与在标准基准数据集EachMovie和MovieLens上速度较慢的记忆基方案相竞争的结果，同时更好地满足了协同过滤应用的各项需求。

**关键词：** 协同过滤，推荐系统，电子商务，数据挖掘，知识发现

## 1 引言

在线基于评分的协同过滤（CF）查询由单个用户的一组（item，评分）对组成。对该查询的响应是一组预测的（item，评分）对，用于该用户尚未评分的item。我们的目标是提供稳健的CF方案，这些方案应满足：

1. **易于实现和维护：** 所有聚合数据应能被普通工程师轻松解释，算法应易于实现和测试；
2. **可动态更新：** 新增一个评分应能立即改变所有预测；
3. **查询时高效：** 查询应快速，必要时可以牺牲存储空间；
4. **对新用户要求低：** 评分很少的用户也应获得有效的推荐；
5. **精度合理：** 方案应与最精确的方案具有竞争力，但精度的微小提升并不总是值得在简单性或可扩展性上做出重大牺牲。

本文的目标不是比较各种CF算法的精度，而是证明Slope One方案能够同时满足上述所有五个目标。尽管我们的方案简单、可更新、计算高效且可扩展，但它们在精度上与那些放弃了部分其他优势的方案相当。

我们的Slope One算法基于一个直观原则——用户对item间的"流行度差异"。我们以成对方式确定一个item比另一个item更受欢迎的程度。衡量这种差异的一种方法就是简单地计算两个item的平均评分之差。反过来，这种差异可用于根据用户对另一个item的评分来预测该用户对其中一个item的评分。考虑两个用户A和B、两个itemI和J以及图1。用户A给itemI的评分为1，用户B给itemI的评分为2，而用户A给itemJ的评分为1.5。我们观察到itemJ比itemI高1.5 - 1 = 0.5分，因此我们可以预测用户B将给itemJ打2 + 0.5 = 2.5分。我们称用户B为被预测用户，itemJ为被预测item。训练集中对每个未知评分存在许多这样的差异，我们取这些差异的平均值。本文提出的Slope One方案系列源于我们选择相关差异以获得单一预测的三种方式。

本文的主要贡献是提出了Slope One CF预测器，并证明它们与记忆基方案相比具有几乎相同的精度，同时更适合CF任务，从而证明了 `f(x) = x + b` 形式的预测器可以与记忆基方案竞争。这是一个重要的结果。

## 2 相关工作

### 2.1 记忆基方案

记忆基协同过滤使用用户对之间的相似度度量来构建预测，通常通过加权平均实现[2, 12, 13, 18]。所选相似度度量决定了预测的精度，已有多种替代方案被研究[8]。记忆基CF的一些潜在缺点包括可扩展性和对数据稀疏性的敏感性。通常，依赖用户间相似度的方案无法预先计算以用于快速在线查询。另一个关键问题是记忆基方案必须计算用户之间的相似度度量，这通常要求有一定数量的用户（例如至少100个用户）输入了某些最低数量的评分（例如至少20个评分），包括当前用户。我们将我们的方案与一个著名的记忆基方案——Pearson方案——进行对比。

### 2.2 模型基方案

存在许多基于模型的CF方法。一些基于线性代数（SVD、PCA或特征向量）[3, 6, 7, 10, 15, 16]；另一些则更直接地借鉴了人工智能技术，如贝叶斯方法、潜在类别和神经网络[1, 2, 9]；还有一些基于聚类[4, 5]。与记忆基方案相比，模型基CF算法通常在查询时更快，尽管它们可能具有昂贵的训练或更新阶段。当查询速度至关重要时，模型基方案可能优于记忆基方案。

我们可以用以下代数形式将我们的预测器与文献中描述的某些预测器进行比较。我们的预测器形式为 `f(x) = x + b`，因此得名"slope one"（斜率为一），其中b是常数，x是表示评分值的变量。对于任意一对item，我们试图找到最佳函数f，使其能够根据一个item的评分预测另一个item的评分。这个函数对于每对item可能不同。一个CF方案将对预测器生成的多个预测进行加权。在[14]中，作者考虑了item对之间的相关性，然后推导出用户评分的加权平均值作为预测器。在他们算法的简单版本中，预测器形式为 `f(x) = x`。在基于回归的版本中，预测器形式为 `f(x) = ax + b`。在[17]中，作者也采用了 `f(x) = ax + b` 形式的预测器。这两篇论文工作的一个自然扩展是考虑 `f(x) = ax² + bx + c` 形式的预测器。然而，在本文中，我们使用 `f(x) = x + b` 形式的朴素预测器。我们也使用朴素加权。[14]中观察到，即使他们基于回归的 `f(x) = ax + b` 算法也没有带来相对于记忆基算法的显著改进。因此，我们证明 `f(x) = x + b` 形式的预测器可以与记忆基方案竞争，这本身就是一个重要结果。

## 3 CF算法

我们提出了三种新的CF方案，并将我们的方案与四种参考方案进行对比：PER USER AVERAGE（用户平均）、BIAS FROM MEAN（均值偏差）、ADJUSTED COSINE ITEM-BASED（调整余弦item基，一种模型基方案）以及PEARSON方案（记忆基方案的代表）。

### 3.1 符号说明

在描述方案时，我们使用以下符号。来自某个用户的评分（称为一个评价）表示为一个不完整的数组u，其中uᵢ是该用户对itemi的评分。在u中被评分item的子集表示为S(u)。训练集中所有评价的集合为\chi。集合S中的元素数量为card(S)。评价u中评分的平均值记为ū。集合Sᵢ(\chi)是所有包含itemi（i \in S(u)）的评价u \in \chi的集合。给定两个评价u和v，我们定义标量积⟨u, v⟩为\sumᵢ\inS(u)\capS(v) uᵢvᵢ。预测结果P(u)表示一个向量，其中每个分量对应一个item的预测值：预测结果隐式地依赖于训练集\chi。

### 3.2 基线方案

最基本的预测算法之一是PER USER AVERAGE（用户平均）方案，其公式为 `P(u) = ū`。即，我们预测用户将根据其平均评分来评价所有item。

另一个简单方案称为BIAS FROM MEAN（均值偏差，有时也称为NON PERSONALIZED[8]），其公式为：

`P(u)ᵢ = ū + (1 / card(Sᵢ(\chi))) * \Sigmaᵥ\inSᵢ(\chi) (vᵢ - v̄)`

即，该预测基于用户平均评分加上训练集中所有用户对特定item与其自身平均值的平均偏差。

我们还将与效果最好的item基方法[14]进行比较，该方法使用以下调整余弦相似度度量，给定两个itemi和j：

`simᵢⱼ = (\Sigmaᵤ\inSᵢⱼ(\chi) (uᵢ - ū)(uⱼ - ū)) / sqrt(\Sigmaᵤ\inSᵢⱼ(\chi) (uᵢ - ū)² * \Sigmaᵤ\inSᵢⱼ(\chi) (uⱼ - ū)²)`

预测结果通过以下加权和获得：

`P(u)ᵢ = (\Sigmaⱼ\inS(u) |simᵢⱼ|(\alphaᵢⱼuⱼ + \betaᵢⱼ)) / (\Sigmaⱼ\inS(u) |simᵢⱼ|)`

其中回归系数\alphaᵢⱼ、\betaᵢⱼ通过最小化\Sigmaᵤ\inSᵢⱼ(u)(\alphaᵢⱼuⱼ + \betaᵢⱼ - uᵢ)² 来选取，其中i和j固定。

### 3.3 PEARSON参考方案

由于我们希望证明我们的方案在预测能力上与记忆基方案相当，我们选择实现一个此类方案作为该类别的代表，并承认存在许多该类型的已有文献记载的方案。在最流行和精确的记忆基方案中，有PEARSON方案[13]。它采用对\chi中所有用户的加权和形式：

`P(u)ᵢ = ū + (\Sigmaᵥ\inSᵢ(\chi) \gamma(u, v)(vᵢ - v̄)) / (\Sigmaᵥ\inSᵢ(\chi) |\gamma(u, v)|)`

其中\gamma是根据皮尔逊相关性计算的相似度度量：

`Corr(u, w) = ⟨u - ū, w - w̄⟩ / sqrt(\Sigmaᵢ\inS(u)\capS(w) (uᵢ - ū)² * \Sigmaᵢ\inS(u)\capS(w) (wᵢ - w̄)²)`

按照[2, 8]的做法，我们设定：

`\gamma(u, w) = Corr(u, w) * |Corr(u, w)|^(\rho-1)`

其中\rho = 2.5，\rho是案例放大（Case Amplification）的幂次。案例放大减少了数据中的噪声：如果相关性高，例如Corr = 0.9，那么经过案例放大后仍然较高（0.9^2.5 \approx 0.8）；而如果相关性低，例如Corr = 0.1，那么它就变得可忽略不计（0.1^2.5 \approx 0.003）。[2]表明，皮尔逊相关性结合案例放大是一种精度合理的记忆基CF方案，尽管存在更精确的方案。

### 3.4 SLOPE ONE方案

Slope One方案同时考虑了来自其他对同一item评分的用户的信息（类似于ADJUSTED COSINE ITEM-BASED）和来自同一用户评分的其他item的信息（类似于PER USER AVERAGE）。然而，该方案也依赖于既不属于用户数组也不属于item数组的数据点（例如图1中用户A对itemI的评分），但这些数据点对于评分预测而言仍然是重要信息。该方法的大部分优势来自于那些未被单独纳入的数据。具体而言，只有那些与待预测用户共同评价过某个item的用户评分，以及只有那些待预测用户也已评分的item评分，才会进入Slope One方案下的评分预测。

形式化地，给定两个评价数组vᵢ和wᵢ（其中i = 1, …, n），我们寻找最佳形式的预测器 `f(x) = x + b` 来从v预测w，通过最小化 \Sigmaᵢ(vᵢ + b - wᵢ)²。对b求导并令导数为零，得到 `b = (\Sigmaᵢ(wᵢ - vᵢ)) / n`。换句话说，常数b必须被选为两个数组之间的平均差异。这一结果启发了以下方案。

给定训练集\chi以及任意两个itemj和i，它们在某些用户评价u中分别具有评分uⱼ和uᵢ（记为 u \in Sⱼᵢ(\chi)），我们考虑itemi相对于itemj的平均偏差为：

`devⱼᵢ = \Sigmaᵤ\inSⱼᵢ(\chi) (uⱼ - uᵢ) / card(Sⱼᵢ(\chi))`

注意，任何不同时包含uⱼ和uᵢ的用户评价u都不会被纳入求和。由devⱼᵢ定义的对称矩阵可以一次性计算完毕，并在新数据输入时快速更新。

鉴于 `devⱼᵢ + uᵢ` 是根据uᵢ对uⱼ的一个预测，一个合理的预测器可能是所有此类预测的平均值：

`P(u)ⱼ = (1 / card(Rⱼ)) * \Sigmaᵢ\inRⱼ (devⱼᵢ + uᵢ)`

其中 `Rⱼ = {i | i \in S(u), i \neq j, card(Sⱼᵢ(\chi)) > 0}` 是所有相关item的集合。有一种近似方法可以简化此预测的计算。对于一个足够密集的数据集，其中几乎所有item对都有评分（即对于几乎所有i, j，都有card(Sⱼᵢ(\chi)) > 0），大多数情况下，对于j ∉ S(u)有 `Rⱼ = S(u)`，而当j \in S(u)时有 `Rⱼ = S(u) - {j}`。由于对于大多数j，`ū = (\Sigmaᵢ\inS(u) uᵢ) / card(S(u)) \approx (\Sigmaᵢ\inRⱼ uᵢ) / card(Rⱼ)`，我们可以将SLOPE ONE方案的预测公式简化为：

`P^S1(u)ⱼ = ū + (1 / card(Rⱼ)) * \Sigmaᵢ\inRⱼ devⱼᵢ`

有趣的是，我们实现的SLOPE ONE并不依赖于用户如何评价单个item，而仅依赖于用户的平均评分以及（至关重要的）用户评价了哪些item。

### 3.5 加权SLOPE ONE方案

SLOPE ONE的一个缺点是没有考虑观测到的评分数量。直观地说，要预测用户A对itemL的评分（已知用户A对itemJ和K的评分），如果有2000个用户对item对(J, L)进行了评分，而只有20个用户对item对(K, L)进行了评分，那么用户A对itemJ的评分很可能比用户A对itemK的评分更适合作为itemL的预测器。因此，我们将加权SLOPE ONE预测定义为以下加权平均：

`P^wS1(u)ⱼ = (\Sigmaᵢ\inS(u)-{j} (devⱼᵢ + uᵢ) * cⱼᵢ) / (\Sigmaᵢ\inS(u)-{j} cⱼᵢ)`

其中 `cⱼᵢ = card(Sⱼᵢ(\chi))`。

### 3.6 双极SLOPE ONE方案

加权方案倾向于频繁出现的评分模式而非不频繁的评分模式，现在我们考虑另一种特别相关的评分模式。我们通过将预测分成两部分来实现这一点。使用加权SLOPE ONE算法，我们根据用户喜欢的item推导出一个预测，再根据用户不喜欢的item推导出另一个预测。

给定一个评分范围（比如0到10），使用范围中值5作为阈值，将高于5分的item视为喜欢、低于5分的视为不喜欢似乎是合理的。如果用户的评分分布均匀，这种方法效果很好。然而，EachMovie数据集中超过70%的评分高于范围中值。由于我们希望支持所有类型的用户（包括均衡型、乐观型、悲观型和双峰型用户），我们使用用户的平均评分作为用户喜欢和不喜欢item之间的阈值。例如，乐观型用户喜欢他们评价的每一件item，我们假设他们不喜欢低于其平均评分的item。这个阈值确保我们的算法对每个用户都有合理数量的喜欢和不喜欢item。

再次参见图1，与往常一样，我们根据同时对itemI和J评分的用户（如用户A）对itemI的偏差，来预测用户B对itemJ的评分。双极SLOPE ONE方案进一步限制了用于预测的评分集合。首先，在item方面，只考虑两个喜欢item之间的偏差或两个不喜欢item之间的偏差。其次，在用户方面，只有那些同时对itemI和J进行了评分并且对itemI有着相同喜欢或不喜欢的用户对的偏差，才被用于预测itemJ的评分。

将每个用户拆分为用户喜欢和用户不喜欢，实际上使用户数量增加了一倍。然而，请注意，上述双极限制必然会减少预测计算中的评分总数。尽管在数据稀疏性问题下这种减少似乎会提高精度（这似乎违反直觉），但未能过滤掉不相关的评分可能带来更大的问题。关键在于，双极SLOPE ONE方案不会根据"用户A喜欢itemK而用户B不喜欢同一itemK"这一事实进行预测。

形式化地，我们将每个评价u拆分为两个已评分item集合：`S^like(u) = {i \in S(u) | uᵢ > ū}` 和 `S^dislike(u) = {i \in S(u) | uᵢ < ū}`。对于每对itemi, j，将所有评价\chi的集合拆分为 `S^likeᵢⱼ = {u \in \chi | i, j \in S^like(u)}` 和 `S^dislikeᵢⱼ = {u \in \chi | i, j \in S^dislike(u)}`。使用这两个集合，我们计算喜欢item的偏差矩阵以及不喜欢item的偏差矩阵：

`dev^likeⱼᵢ = \Sigmaᵤ\inS^likeⱼᵢ(\chi) (uⱼ - uᵢ) / card(S^likeⱼᵢ(\chi))`

`dev^dislikeⱼᵢ = \Sigmaᵤ\inS^dislikeⱼᵢ(\chi) (uⱼ - uᵢ) / card(S^dislikeⱼᵢ(\chi))`

根据i属于 `S^like(u)` 还是 `S^dislike(u)`，基于itemi的评分对itemj的预测分别为 `p^likeⱼᵢ = dev^likeⱼᵢ + uᵢ` 或 `p^dislikeⱼᵢ = dev^dislikeⱼᵢ + uᵢ`。双极SLOPE ONE方案由以下公式给出：

`P^bpS1(u)ⱼ = (\Sigmaᵢ\inS^like(u)-{j} p^likeⱼᵢ * c^likeⱼᵢ + \Sigmaᵢ\inS^dislike(u)-{j} p^dislikeⱼᵢ * c^dislikeⱼᵢ) / (\Sigmaᵢ\inS^like(u)-{j} c^likeⱼᵢ + \Sigmaᵢ\inS^dislike(u)-{j} c^dislikeⱼᵢ)`

其中权重 `c^likeⱼᵢ = card(S^likeⱼᵢ(\chi))` 和 `c^dislikeⱼᵢ = card(S^dislikeⱼᵢ(\chi))` 与加权SLOPE ONE方案中的权重类似。

## 4 实验结果

给定CF算法的有效性可以被精确度量。为此，我们采用了留一法平均绝对误差（All But One Mean Average Error，MAE）[2]。在计算MAE时，我们依次从测试集中的所有评价中隐藏单个评分，预测该隐藏评分，然后计算预测的平均误差。给定预测器P和来自用户的评价u，P在一组评价\chi'上的误差率由以下公式给出：

`MAE = (1 / card(\chi')) * \Sigmaᵤ\in\chi' ((1 / card(S(u))) * \Sigmaᵢ\inS(u) |P(u^(i)) - uᵢ|)`

其中 `u^(i)` 是隐藏了用户对第i个item评分uᵢ后的用户评价u。

我们在Compaq Research提供的EachMovie数据集和明尼苏达大学Grouplens Research Group提供的MovieLens数据集上测试了我们的方案。这些数据来自电影评分网站，EachMovie的评分范围为0.0到1.0，步长为0.2；MovieLens的评分范围为1到5，步长为1。按照[8, 11]的做法，我们使用足够多的评价来构成总量为50,000个评分的训练集(\chi)，并使用额外的一组评价构成总量至少100,000个评分的测试集(\chi')。当预测值超出给定数据集的允许评分范围时，进行相应修正：例如，对于EachMovie的0到1评分范围，预测值1.2被解释为1。由于MovieLens的评分范围是EachMovie的4倍，MovieLens上的MAE除以4以使结果直接可比。

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

表1：所有方案对比：EachMovie和MovieLens数据集上的留一法平均绝对误差率，数值越低越好。

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
