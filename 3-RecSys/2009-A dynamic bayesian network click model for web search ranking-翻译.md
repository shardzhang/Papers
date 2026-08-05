# 基于动态贝叶斯网络的网页搜索排序点击模型

> Olivier Chapelle, Ya Zhang | Yahoo! Labs



本文介绍了 DBN（Dynamic Bayesian Network，动态贝叶斯网络）点击模型：一种用于网页搜索排序的新型点击模型。核心内容：

- 问题动机：点击日志是搜索排序的重要隐式反馈来源，但存在位置偏差——即使相关，排在低位的 URL 也不易被点击。现有位置模型和级联模型无法同时区分感知相关性和实际相关性
- 模型创新：提出动态贝叶斯网络模型，引入满意度变量 $s_u$ 区分感知相关性（用户点击的吸引力 $a_u$）和实际相关性（用户对落地页的满意度），并用持续性参数 $\gamma$ 建模用户继续浏览的概率
- 简化版本：当 $\gamma = 1$ 时，模型退化为简单计数形式，无需 EM 算法，便于实际部署
- 实验验证：在 58M 会话数据上，DBN 模型在位置 1 的 CTR 预测和排序质量上均优于位置模型、级联模型和逻辑回归模型；预测相关性仅比大量人工标注训练的排序函数差 6.3%，两者结合可提升 2% DCG

关键发现：

- 区分感知相关性和实际相关性至关重要：用户基于搜索结果页展示的标题和摘要决定是否点击（感知相关性），但满意度取决于落地页内容（实际相关性），两者存在显著差异
- DBN 模型能捕获 URL 间的交互效应：当位置 1 的结果非常优秀时，位置 2 的低 CTR 并非因为不相关，而是用户已满意离开；DBN 能正确估计其相关性，而位置模型会产生偏差
- 点击数据可有效替代人工标注：仅使用点击数据训练的排序函数比人工标注训练的版本仅差 4%，两者结合可获得 2% 的显著 DCG 提升

---



## 摘要

与任何机器学习应用一样，网页搜索排序需要标注数据。这些标注通常以编辑人员做出的相关性评估形式出现。点击日志也可以提供重要的隐式反馈来源，并可以作为编辑标注的廉价替代品。然而，主要困难来自所谓的位置偏差——排在较低位置的 URL 即使相关也不太可能被点击。在本文中，我们提出了一种动态贝叶斯网络，旨在**从点击日志中为我们提供无偏的相关性估计**。实验表明，所提出的点击模型在预测点击率和相关性方面均优于其他现有点击模型。



## 关键词

Click-through Rate, Click Modeling, Web Search, Ranking, Dynamic Bayesian Network



## 1 引言

网页排序传统上基于手工设计的排序函数，如 BM25 [18]。随着排序使用数千个特征，手工调整排序函数变得不可行。几种机器学习算法已被应用于自动优化排序函数 [4, 5]。机器学习排序需要大量训练样本，相关性标签指示每个查询-文档对的相关性程度。编辑标注的成本通常相当昂贵。此外，训练样本的相关性标签可能随时间变化。例如，如果查询是时效性的或周期性的（如"www"或"总统选举"），搜索引擎需要向用户提供最新的文档/网站。然而，保持所有相关性标签的更新将是不可行的。点击日志嵌入了用户对搜索引擎满意度的重要信息，并可以提供高度有价值的相关性信息来源。与编辑标注相比，点击获取成本低得多，并且总是反映当前的相关性。

搜索引擎已通过多种方式使用点击：调整搜索参数、评估不同的排序函数 [7, 13, 14, 15]，或作为直接影响排序的信号 [1, 13]。然而，点击已知存在偏差，受展示顺序、文档的外观（如标题和摘要）以及个别网站声誉的影响。许多研究 [8, 10] 试图解释点击的位置偏差。Carterette 和 Jones [7] 提出建模点击与相关性之间的关系，以便在缺乏编辑相关性判断时使用点击来无偏地评估搜索引擎。其他研究 [10, 21, 16] 试图建模用户搜索过程中的点击行为，以便根据过去点击的观察准确预测未来的点击。

两种不同类型的点击模型是位置模型 [8, 10, 17] 和级联模型 [8]。位置模型假设点击取决于相关性和检查两个因素。每个排名位置有一定的被检查概率，该概率随排名衰减且仅取决于排名。对 URL 的点击表明该 URL 被检查且被用户认为相关。然而，该模型将搜索结果页中的各个 URL 独立处理，未能捕获 URL 之间在检查概率上的交互。以两个对查询同样相关的 URL 为例：用户可能只点击上面那个，感到满意，然后离开搜索结果页。在这种情况下，位置偏差不能完全解释第二个 URL 缺乏点击的现象。

级联模型假设用户从上到下顺序查看结果，一旦点击了相关文档就停止浏览。这里，检查概率由两个因素间接决定：URL 的排名和所有之前 URL 的相关性。级联模型做了一个强假设，即每次搜索只有一次点击，因此无法解释放弃搜索或多次点击的搜索。尽管级联模型有相当大的限制，但该论文的作者表明它可以比上述位置模型更准确地预测点击率（CTR）。

上述模型均未区分感知相关性和实际相关性。因为用户在点击 URL 之前无法检查文档的内容，所以点击的决定是基于感知相关性做出的。虽然感知相关性与实际相关性之间存在强相关性，但也有许多情况下它们不同。

在本文中，我们提出了一种动态贝叶斯网络（DBN）模型来建模用户的浏览行为。与位置模型一样，我们假设当且仅当用户检查了 URL 并认为其相关时才会发生点击。与级联模型类似，我们的模型假设用户线性遍历结果，并根据文档的感知相关性决定是否点击。如果用户对点击的 URL 不满意（基于实际相关性），则选择检查下一个 URL。我们的模型与级联模型在两个方面不同：1. 因为点击不一定意味着用户对点击的文档满意，我们试图区分感知相关性和实际相关性；2. 我们不限制用户在搜索过程中可以进行的点击次数。

我们将所提出的模型与之前的模型进行比较，表明基于动态贝叶斯网络的模型优于其他模型。然后以两种方式使用每个 URL 的预测相关性：要么作为排序函数中的特征，要么作为学习排序函数的补充数据。我们表明，使用这些预测相关性学习的函数与使用大量编辑数据训练的函数相差不大。我们进一步表明，结合两种类型的数据可以导致更准确的排序函数。



## 2 建模展示偏差

如上所述，展示偏差指的是用户更有可能点击排名顶部文档的事实。

### 2.1 位置模型

处理此展示偏差问题的一类流行方法是基于位置的模型 [8, 10, 17]。这些方法的核心假设是，用户在以下两个条件满足时点击链接：用户检查了 URL 并发现其相关；此外，检查概率仅取决于位置。更精确地说，给定位置 $p$ 处的 URL $u$，点击概率通过隐藏变量 $E$ （表示 $u$ 是否被检查）建模：

$$
P(C = 1|u, p) = \sum_{e \in \{0,1\}} P(C = 1|u, p, E = e) P(E = e|u, p) = \underbrace{P(C = 1|u, E = 1)}_{:= \alpha_u} \underbrace{P(E = 1|p)}_{:= \beta_p}
$$

最后一个等式使用了以下假设：如果用户没有检查 URL，则不会点击；如果 URL 被检查，点击概率仅取决于其相关性；检查概率仅取决于位置。因此，点击概率是两个概率 $\alpha_u$ 和 $\beta_p$ 的乘积：第一个建模 URL 对查询的相关性，而第二个捕获位置效应。请记住，我们的目标是根据点击日志推断 URL 的相关性。这正是 $\alpha_u$ 所表示的：URL 对用户的感知相关性，与位置无关。如果我们做出额外假设 $\beta_1 = 1$——即用户总是检查第一个结果——那么 $\alpha_u$ 可以解释为位置 1 的等效 CTR，即该 URL 如果放在第一个位置时的 CTR。注意查询 $q$ 在这里是隐式的；更形式化地，我们应该写 $\alpha_{uq}$ 以强调对查询的依赖，但在本文的其余部分我们假设查询是固定的。

**COEC 模型。** 一种廉价且直接的方法是将 $\beta_p$ 估计为位置 $p$ 的聚合 CTR（在所有查询和会话上）。假设有 $N$ 个会话中出现了 $u$，第 $i$ 个会话中 $c_i \in \{0, 1\}$ 表示是否有点击，$p_i$ 是 URL $u$ 出现的位置。然后 $\alpha_u$ 计算为 [19]：

$$
\alpha_u = \frac{\sum_{i=1}^{N} c_i}{\sum_{i=1}^{N} \beta_{p_i}} \qquad (1)
$$

如 [19] 中所述，我们将此方法称为点击除以期望点击（COEC），因为分母可以被视为给定 URL 出现位置的"期望"点击数。

COEC 模型的问题在于 $\beta$ 的估计是有偏的。如果搜索引擎以随机顺序给出结果，它将是有效的。但由于更相关的文档往往出现在排名较高的位置，给定位置的观察 CTR 不仅捕获了位置偏差，还捕获了该位置的典型相关性。

**检查模型。** 另一种方法是通过最大似然来找到 $\alpha_u$ 和 $\beta_p$。注意 URL 需要在不同位置展示过才能使此方法（和下面的其他方法）有意义。否则解是不适定的。这很合理，因为为了捕获位置效应，需要观察同一 URL 在不同位置的 CTR。由于搜索引擎的持续变化，这通常是成立的。给定向量 $\beta_p$，$\alpha_u$ 的最大似然解为：

$$
\alpha_u = \arg\max_{\alpha} \sum_{i=1}^{N} c_i \log(\alpha \beta_{p_i}) + (1 - c_i) \log(1 - \alpha \beta_{p_i}) \qquad (2)
$$

向量 $\beta_p$ 通过 $\alpha_u$ 和 $\beta_p$ 之间的交替（或联合）最大化似然来估计。上述方法的一个缺点是它可能导致 $\alpha_u > 1$。这不理想，因为 $\alpha_u$ 应该表示一个概率。可以直接使用期望最大化（EM）算法，其中隐藏变量是检查变量 $E$ [10]。这确保 $\alpha_u \le 1$。我们在检查模型的实现中使用了 EM 算法。

**逻辑回归模型。** 另一种替代方案是使用与逻辑回归相关的略微不同的模型 [8]：

$$
P(C = 1|u, p) := \frac{1}{1 + \exp(-\tilde{\alpha}_u - \tilde{\beta}_p)} \qquad (3)
$$

点击概率不再是概率的乘积，但它仍然是 URL 和位置的函数。主要优点是它确保结果概率始终在 0 和 1 之间；此外优化更容易，因为它是无约束且联合凸的问题。

### 2.2 级联模型

级联模型 [8] 与上述位置模型的不同之处在于，它考虑了同一搜索结果页中 URL 之间的依赖关系，并同时建模一个会话中的所有点击和跳过。它假设用户从上到下查看搜索结果，并决定是否点击每个 URL。一旦发出点击，被点击结果下方的文档无论位置如何都不会被检查。在级联模型中，每个文档 $d$ 要么以概率 $r_d$ （即文档相关的概率）被点击，要么以概率 $(1 - r_d)$ 被跳过。级联模型假设点击的用户永不回头，跳过的用户总是继续。对第 $i$ 个文档的点击表明：1. 用户必定决定跳过上面的排名；2. 用户认为第 $i$ 个文档相关。因此第 $i$ 个文档的点击概率可以表示为：

$$
P(C_i = 1) = r_i \prod_{j=1}^{i-1} (1 - r_j) \qquad (4)
$$



## 3 动态贝叶斯网络

我们现在引入另一种模型，它将结果集作为一个整体来考虑，并在从点击日志估计给定 URL 的相关性时考虑其他 URL 的影响。考虑其他 URL 相关性的原因如下：以位置 3 的一个相关文档为例；如果位置 1 和 2 的文档都非常相关，该文档可能只有很少的点击；另一方面，如果顶部两个文档不相关，它将获得大量点击。仅依赖位置的点击模型将无法区分这两种情况。我们扩展级联模型的思想，提出了一种动态贝叶斯网络（DBN）[11] 来同时建模所有文档的相关性。

### 3.1 模型

我们提出的动态贝叶斯网络如图 1 所示。序列覆盖搜索结果列表中的文档。为简单起见，我们只保留第一页结果的前 10 个文档，这意味着序列从 1 到 10。框内的变量在会话级别定义，框外的变量在查询级别定义。如前所述，我们假设查询是固定的。

对于给定的位置 $i$，除了表示该位置是否有点击的观察变量 $C_i$ 外，还定义了以下二值隐藏变量来分别建模检查、感知相关性和实际相关性：

- $E_i$：用户是否检查了该 URL？
- $A_i$：用户是否被该 URL 吸引？
- $S_i$：用户是否对落地页满意？

以下方程描述了该模型：

$$
A_i = 1, E_i = 1 \Leftrightarrow C_i = 1 \qquad (5a)
$$

$$
P(A_i = 1) = a_u \qquad (5b)
$$

$$
P(S_i = 1 | C_i = 1) = s_u \qquad (5c)
$$

$$
C_i = 0 \Rightarrow S_i = 0 \qquad (5d)
$$

$$
S_i = 1 \Rightarrow E_{i+1} = 0 \qquad (5e)
$$

$$
P(E_{i+1} = 1 | E_i = 1, S_i = 0) = \gamma \qquad (5f)
$$

$$
E_i = 0 \Rightarrow E_{i+1} = 0 \qquad (5g)
$$

与检查模型一样，我们假设当且仅当用户查看了 URL 并被其吸引时才会发生点击 (5a)。被吸引的概率仅取决于 URL (5b)。与级联模型类似，用户从上到下线性扫描 URL，直到决定停止。用户点击并访问 URL 后，有一定概率会对该 URL 满意 (5c)。另一方面，如果不点击，则不会满意 (5d)。一旦用户对访问的 URL 满意，就停止搜索 (5e)。如果用户对当前结果不满意，有概率 $1 - \gamma$ 放弃搜索 (5f)，有概率 $\gamma$ 检查下一个 URL。换句话说，$\gamma$ 衡量用户的持续性。如果用户没有检查位置 $i$，则不会检查后续位置 (5g)。此外，$a_u$ 和 $s_u$ 有 Beta 先验。这个先验的选择是自然的，因为 Beta 分布是二项分布的共轭分布。显然某些假设并不现实，我们在第 8 节讨论如何扩展它们。然而，如实验部分所示，该模型已经可以准确解释观察到的点击。

与检查模型不同，我们的模型有两个与文档相关性相关的变量 $a_u$ 和 $s_u$。第一个建模感知相关性，因为它衡量基于 URL 的点击概率。第二个是用户在点击链接后满意的概率；因此它可以被理解为实际相关性与感知相关性之间的"比率"。事实上，如果我们定义 URL 的相关性为用户在看到 URL 后满意的概率，我们有：

$$
r_u := P(S_i = 1 | E_i = 1) = P(S_i = 1 | C_i = 1) P(C_i = 1 | E_i = 1) = a_u s_u \qquad (6)
$$

据我们所知，这是第一个试图建模实际相关性而非仅建模感知相关性的点击模型。

### 3.2 与其他模型的联系

检查模型可以看作是我们模型的特例，其中 $E_i$ 是独立的且分布仅取决于位置。在这种情况下 $S_i$ 是无意义的，因为无法推断它们。

[8] 的级联模型是我们模型在 $\gamma = 1$ 和 $s_u = 1$ 时的特例。即用户持续检查直到找到看起来相关的文档，然后点击并停止。

在 [7] 中，文档的相关性从点击日志中预测，其他文档的 CTR 在预测过程中被使用。他们的动机不同（评估搜索引擎质量），但他们解决的是同一问题：其他文档对 CTR 的影响。然而，如果文档集在会话之间发生变化，其他位置的 CTR 定义是有问题的。

Joachims [13] 引入了所谓的跳过-上方对：当用户在位置 $i$ 没有点击但在位置 $j > i$ 点击了，这表明位置 $j$ 的文档优于位置 $i$ 的文档。我们用我们的模型恢复了这类对，因为在那种情况下 $1 = A_j > A_i = 0$。然而，用跳过-上方对学习排序函数的问题是，人们倾向于学习与生产中相反的函数（在某种意义上只有"负"实例）。

对于位置模型，已经观察到对不同类型查询（如导航型与信息型）使用不同的向量 $\beta$ 更好。原因是对于导航型查询，CTR 随位置衰减得更快。但我们认为这种衰减不是查询类型的函数，而是顶部 URL 质量的函数。对于导航型查询 [3]，顶部结果通常非常优秀，较低位置的点击非常少。我们的点击模型直接捕获了这一效应，不需要为不同类型查询使用不同的浏览模型。

### 3.3 推断

图 1 的动态贝叶斯网络仅比标准隐马尔可夫模型（HMM）稍微复杂一些，因为位置 $i + 1$ 的隐藏状态与位置 $i$ 的观察之间存在条件依赖。使用期望最大化（EM）算法 [9] 来找到变量 $a_u$ 和 $s_u$ 的最大似然估计，使用前向-后向算法来计算隐藏变量的后验概率。E 步和 M 步的详细推导以及潜在变量的置信度计算请参阅论文附录。尽管我们也可以用 EM 估计 $\gamma$ [2]，但我们简单地将 $\gamma$ 作为模型的可配置参数。



## 4 实验

进行了三种类型的实验来验证我们的模型。首先我们评估点击模型在位置 1 预测 CTR 方面的表现。然后我们使用预测相关性作为排序信号。在第三种类型的实验中，我们使用预测相关性作为目标来训练排序函数。

点击日志来自一个商业搜索引擎。会话可以用各种方式定义，但在我们所有的实验中，定义如下。会话总是有唯一的用户和唯一的查询。当用户发出查询时开始，用户端 60 分钟空闲时间后结束。对于每个会话，我们获取查询、结果集中的 URL 列表和点击的 URL 列表。对查询和 URL 应用简单的归一化。如前所述，我们只限于第一页结果的前 10 个 URL。点击顺序与排名顺序不一致的会话（例如用户先点击第 4 个链接再点击第 2 个链接）在我们的模型下概率为零。平均大约 3% 的会话包含一个或多个乱序点击。我们可以交换这些会话的点击顺序，但我们简单地丢弃了它们。我们还丢弃了所有少于 10 个会话的查询。

### 4.1 预测点击率

我们首先评估我们提出的点击模型的准确性。对于此任务，我们从英国市场的点击日志中获取 58M 会话和 682k 个唯一查询。评估点击模型的自然方式如 [10] 所述：对于给定查询，取一些会话用于训练，评估其他会话的点击似然。但我们的目标不同：我们不太关注点击预测，更关注潜在变量 $a_u$ 和 $s_u$ 的准确性：它们确实是后续用于排序的变量。$s_u$ 的质量难以评估，但 $a_u$ 容易评估。确实 $a_u$ （像检查模型的 $\alpha_u$）是对位置 1 CTR 的预测。使用以下留一出实验方案：

1. 检索与给定查询相关的所有会话；
2. 考虑在位置 1 和其他位置都出现过的 URL；
3. 将该 URL 出现在位置 1 的所有会话作为测试集留出；
4. 在剩余会话上训练模型并预测 $a_u$；
5. 在留出的会话上计算位置 1 的测试 CTR；
6. 计算这两个量之间的误差；
7. 在所有这样的 URL 和查询上平均误差，按测试会话数加权。

在计算位置 1 的实际 CTR 与预测 CTR 之间的误差时，使用两种类型的误差指标：均方误差（MSE）和 KL 散度。注意 KL 散度与测试会话上点击的似然相差一个常数值。

DBN 模型需要输入参数 $\gamma$。我们首先执行上述留一出程序来经验确定 $\gamma$ 的最优值。结果如图 3 所示。当 $\gamma = 0.9$ 时 DBN 模型达到最小 MSE，这表明用户在寻找相关文档方面是持续的。在其余实验中，我们将 $\gamma$ 设为 0.9。

我们将 DBN 模型与其他第 2 节描述的点击模型进行比较。结果如图 2 所示。COEC、检查和逻辑回归模型分别由公式 (1)、(2) 和 (3) 描述。对于这些方法，我们使用经验贝叶斯方法 [6] 配合 Beta 先验来平滑观察到的不同位置的点击概率。这产生更稳定和准确的结果。更精确地说，给定位置 $p$ 的 CTR 假设从参数为 $a_p$ 和 $b_p$ 的 Beta 分布中抽取。这些参数通过在整个数据集上的最大似然找到。然后我们用平滑版本替换原始 CTR。COEC 模型有时预测的 CTR 大于 1：在这种情况下，计算 MSE 误差时将预测截断为 1。然而，即使截断到 1，当测试 CTR 小于 1 时 KL 散度是无穷的。这就是 COEC 不出现在图 2 右侧的原因。对于公式 (4) 描述的级联模型，它只能处理恰好有一次点击的会话。测试此方法时，我们因此丢弃了所有零次或多次点击的会话。虽然级联模型因假设用户非常持续地检查直到找到相关文档而受到很大限制，但我们看到它比 COEC 和检查模型表现更好，与逻辑回归模型相当，这表明考虑同一搜索结果页中 URL 之间的交互是有用的。这些结果与 [8] 中报告的一致。

在图 2 中，我们将误差按用于训练的最小会话数的函数进行了分解。例如 1000 意味着 MSE 是在所有 (查询, URL) 对上计算的，其中该 URL 不出现在位置 1 的会话数至少为 1000。当会话数很大时，$a_u$ 和 $s_u$ 上的先验分布选择不起重要作用。人们通常期望准确性随会话数提高。对于级联和我们的模型大致如此，但令人惊讶的是，其他模型的准确性在会话数非常大时恶化了。我们通过查看大误差的 (查询, URL) 对来调查此问题。一个例子是英国市场的查询"myspace"和 URL www.myspace.com。位置 1 的 URL 应该是 uk.myspace.com，但由于排名的变化（例如测试导致的），URL www.myspace.com 多次出现在位置 1。对于这些会话，CTR 非常高，为 0.97，如预期。在其他会话中，它出现在位置 2，CTR 为 0.11 的低值。这个低 CTR 是预期的，因为第一位置的 URL 是 uk.myspace.com，大多数用户甚至不会看 www.myspace.com。然而，逻辑回归模型预测 CTR 为 0.21。逻辑回归模型的预测可以理解，因为平均而言 URL 在位置 1 和位置 2 的 CTR 大约差 2 倍。查询 myspace 差 9 倍的原因是位置 1 的 URL 已经非常优秀，用户几乎不看位置 2。检查和逻辑回归模型没有考虑这一信息但 DBN 考虑了：它预测 www.myspace.com 的 CTR 为 0.95。位置 1 的完美 URL 对于导航型查询更常发生。对于这类查询，我们有很多会话；这解释了为什么当考虑大量会话时位置模型的性能恶化。

### 4.2 预测相关性作为排序特征

CTR 预测的准确性可能不会直接转化为相关性。在第二组实验中，我们直接使用预测相关性来对 URL 排序。在这种情况下，点击预测的相对顺序比其绝对值更重要。我们将 DBN 模型与级联模型和逻辑回归模型进行比较。我们还在比较中包含一个基线排序函数 $\phi$，它使用许多其他排序信号（如 BM25 分数）。该函数是网络搜索引擎使用的典型排序函数。

数据与上面不同：我们只考虑有编辑判断且在几个月期间至少有 10 个会话的查询。这产生了 3153 个查询和 44.5M 个会话。

使用以下实验方案：

1. 检索与给定查询相关的所有会话；
2. 考虑所有有编辑相关性判断的 URL；
3. 在会话上训练模型并预测 URL 的 $a_u$ 和 $s_u$；
4. 根据 DBN 模型给出的预测相关性 (6) 对 URL 排序；
5. 计算排名 5 处的归一化折扣累积增益（NDCG）[12]；
6. 在所有查询上平均 NDCG@5。

结果如图 5 所示，其中我们将 NDCG@5 按 URL 的最小会话数的函数进行了分解。随着每个 URL 会话数的增加，我们期望点击预测更准确和更有信心，从而改善 URL 的排序。这正是我们在图 5 中观察到的——对于所有点击模型，NDCG@5 随会话数提高。然而，当我们限制实验到 URL 的更高会话数时，每个查询留下的 URL 更少。在极端情况下，一个查询可能只包含一个 URL，NDCG@5 将总是 1。确实，每个查询的平均 URL 数总体为 10.5，但如果限制到超过 10,000 个会话的 URL，这个数字降到 8。因此当每个 URL 的会话数非常高时，NDCG@5 的区分度较低，基线函数的性能不是常数。这就是为什么我们也绘制相对于基线函数的 NDCG@5（图 5 右侧）以消除每个查询 URL 数变化的影响。

总体而言，DBN 模型比逻辑回归模型和级联模型能更好地对 URL 排序。如预期，随着 URL 会话数的增加（即预测更有信心），NDCG@5 和相对 NDCG@5 都增加。作为 DBN 模型的特例（$\gamma = 1$ 和 $s_u = 1$），级联模型的行为与 DBN 模型非常相似但 NDCG@5 更低，这证实了引入满意度 $s_u$ 和持续性 $\gamma$ 的必要性。级联模型确实因只能考虑恰好有一次点击的会话而受到影响。另一方面，逻辑回归模型的行为与级联模型和 DBN 模型非常不同。当会话数较小时，DBN 模型和逻辑回归模型之间的相对 NDCG@5 差异很大。当会话数变大时，DBN 和逻辑回归模型之间的差异变小，主要是因为每个查询的 URL 数更少。

给定上述观察，我们然后固定最小会话数为 10，每个查询的最小 URL 数为 10。结果 392 个查询通过标准，平均每个查询包含 13.4 个 URL。然后我们在此数据集上计算 NDCG@5。如表 1 所示，DBN 模型的 NDCG@5 分别比逻辑回归模型和级联模型好 5.8% 和 2.4%。所有差异根据 Wilcoxon 符号秩检验都是统计显著的（$p \le 0.001$）。为了量化满意度变量 $s_u$ 的影响，我们也仅按 $a_u$ 而非 (6) 排序。差异仅为 0.5%，统计不显著。我们将在第 7 节讨论模型的扩展以更好地建模满意度。

### 4.3 使用预测相关性学习排序函数

网页搜索的机器学习排序首先在 [4] 中引入；我们这里遵循应用于成对偏好的梯度提升决策树框架 [20]。我们有两组成对偏好：

1. $P_E$ 来自对 4180 个查询和 126k 个 URL 的编辑判断，产生约 1M 个偏好对；
2. $P_C$ 来自我们的点击模型：我们通过基于置信度阈值的过滤只保留 1.1M 个 URL（对应 420k 个唯一查询）；将相关性分数转换为偏好产生约 2M 个对。

对于每个 (查询, URL) 对，我们提取特征向量 $\mathbf{x}$。偏好集中的对 $(\mathbf{x}_i, \mathbf{x}_j)$ 表示 $\mathbf{x}_i$ 优于 $\mathbf{x}_j$，理想情况下应转化为 $f(\mathbf{x}_i) > f(\mathbf{x}_j)$，其中 $f$ 是排序函数。

提升算法优化以下目标函数（提升过程的细节见 [20]）：

$$
\frac{1 - \delta}{|P_E|} \sum_{(\mathbf{x}_i, \mathbf{x}_j) \in P_E} \max(0, 1 - (f(\mathbf{x}_i) - f(\mathbf{x}_j)))^2 + \frac{\delta}{|P_C|} \sum_{(\mathbf{x}_i, \mathbf{x}_j) \in P_C} \max(0, 1 - (f(\mathbf{x}_i) - f(\mathbf{x}_j)))^2 \qquad (7)
$$

因此目标函数是基于编辑和基于点击的偏好的组合。测试集是编辑判断的留出集，在其上计算排名 5 处的折扣累积增益（DCG）。相对性能作为 $\delta$ 的函数绘制在图 6 中。

我们可以从这个图中得出两个有趣的结论：

1. 仅从点击学习，DCG 仅比使用编辑判断学习的标准模型差 4%；这很了不起，因为在这个实验中编辑判断集相对较大。这表明从点击学习对于编辑判断很少或没有的市场非常有价值。
2. 结合两种类型的数据在 DCG 上带来 2% 的增益，这在网页搜索排序社区被认为是显著的。因此即使在有大量编辑判断的市场中，我们仍然可以利用点击来达到更高的 DCG。

最后注意评估是使用编辑指标进行的。但由于下面讨论的点击和编辑判断之间的差异，我们期望在点击数据上训练的模型如果用基于点击的指标评估会表现更好。



## 5 简化模型

如图 3 所示，位置 1 CTR 的最佳预测是在 $\gamma = 0.9$ 时获得的。但 $\gamma = 1$ 仅产生略微更差的预测。这个特定设置很有趣，因为在这种情况下推断更简单。确实用户持续检查直到满意，这意味着最后一次点击提供了满意的结果，其下方的结果未被检查。因此不需要前向-后向算法和 EM，因为检查变量没有歧义：$E_1 = \cdots = E_\ell = 1$ 且 $E_{\ell+1} = \cdots = E_{10} = 0$，其中 $\ell$ 是最后一次点击的位置。潜在变量 $a_u$ 和 $s_u$ 使用简单计数估计，如算法 1 所述。

**算法 1** $\gamma = 1$ 时的简化模型估计。

$$
\begin{aligned}
&\text{Initialize all URLs } u \text{ of } a_u^N, a_u^D, s_u^N, s_u^D \text{ to 0.} \\
&\textbf{for } \text{all sessions } \textbf{do} \\
&\quad \textbf{for } \text{all } u \text{ above or at the last clicked URL } u \textbf{ do} \\
&\quad\quad a_u^D \leftarrow a_u^D + 1 \\
&\quad \textbf{end for} \\
&\quad \textbf{for } \text{all } u \text{ that got clicked } u \textbf{ do} \\
&\quad\quad a_u^N \leftarrow a_u^N + 1 \\
&\quad\quad s_u^D \leftarrow s_u^D + 1 \\
&\quad \textbf{end for} \\
&\quad s_u^N \leftarrow s_u^N + 1, \text{ where } u \text{ is the last clicked URL.} \\
&\textbf{end for} \\
&\text{// } \alpha^a, \beta^a, \alpha^s, \beta^s \text{ are } a_u \text{ and } s_u \text{ Beta prior parameters.} \\
&\textbf{for } \text{all URLs } u \textbf{ do} \\
&\quad a_u = (a_u^N + \alpha^a) / (a_u^D + \alpha^a + \beta^a) \\
&\quad s_u = (s_u^N + \alpha^s) / (s_u^D + \alpha^s + \beta^s) \\
&\textbf{end for}
\end{aligned}
$$

## 6 点击与编辑判断

在最后两组实验中，点击被用作编辑判断的替代品。评估我们模型估计的相关性与编辑给出的实际相关性之间的相关性很重要。一个自然的衡量标准是矛盾对的数量（与 Kendall's tau 检验相关）。我们将所有编辑判断转换为成对偏好，从我们模型提取的相关性分数也做了同样的事情。在两组对的交集中，20% 的对在偏好上存在分歧。我们调查了这些差异的原因，发现排除编辑判断中的错误后，这些原因可以概括为两个主要类别：

1. 流行度不一定与相关性一致；
2. 点击主要衡量感知相关性，而编辑判断落地页的相关性。

第一个类别的一个例子是查询"adobe"：主页 www.adobe.com 似乎是最相关的 URL，但大多数用户点击 acrobat reader 链接。另一个例子查询是"bank of america"。大多数用户更愿意点击网上银行页面 http://www.bankofamerica.com/onlinebanking/，而编辑倾向于认为主页 http://www.bankofamerica.com 是此查询的目标页面。鉴于相关性和点击之间的这种固有差异，我们可能永远无法完全消除差距。另一方面，利用预测相关性来细化相关性定义和编辑相关性判断的指导方针可能是有用的。

第二种类型的不一致可以进一步分为两个子类别：搜索结果摘要的相关性与落地页的相关性差异很大的情况；以及用户基于页面的可信度而非页面的相关性点击的情况。第一个子类别最常与 URL 的标题和摘要的呈现有关。第二个子类别的一个例子查询是"travel insurance"。虽然有许多小型保险公司专注于销售旅行保险（在相关性判断方面更相关），但用户仍然倾向于更频繁地点击品牌保险公司的网站，而旅行保险只是其业务的一小部分。

总之，这项研究表明定义相关性是一个复杂的问题，点击和编辑判断是两种相关但不同的满足用户需求的方式。



## 7 扩展

到目前为止，上述所有实验只考虑了网页搜索结果。事实上，大量信息已被混合到搜索结果页中：最引人注目的是赞助搜索结果。此外，如今的搜索引擎倾向于包含许多链接来帮助用户快速导航到搜索目的地，包括相关搜索、查询拼写错误建议和快捷方式。

我们发现在许多情况下，搜索结果缺乏点击是由于用户选择点击了上述一个或多个部分的 URL。因此我们希望对整个搜索结果页的点击进行建模。我们稍微修改 DBN 模型以考虑整页点击。之前我们只考虑一个会话中检索到的前 10 个结果。这里我们定义两个虚拟 URL：前导 URL 定义为搜索结果页顶部的 URL（如赞助搜索、拼写建议）；尾随 URL 定义为搜索结果页底部的 URL（如分页）。

点击前导 URL 可能表明用户从未检查搜索结果部分的 URL；而点击尾随 URL 表明用户很可能对之前的 URL 不满意。同样，我们使用改进的 DBN 模型的预测相关性来对 URL 排序并计算 NDCG@5。结果汇总在表 2 中。与表 1 一样，我们只保留至少有 10 个会话的 URL 和至少有 10 个 URL 的查询。改进的 DBN 模型比原始 DBN 模型优 2.2%，差异统计显著（$p \le 0.001$）。

此外，对于这个改进的 DBN 模型，满意度变量 $s_u$ 似乎估计得更好——如果仅按 $a_u$ 排序，准确性下降 1.2%（$p \le 0.001$）（相比原始模型的 0.5%，见表 1）。这可能是因为我们现在建模了搜索页面底部的点击，如"下一步"：当用户点击下一步按钮时，很可能他对最后访问的 URL 不满意。我们原始的 DBN 忽略了这一事实，错误地将此类 URL 归因于高满意度。



## 8 结论与未来工作

从点击日志中提取相关性信息对于网页搜索排序是一项具有挑战性但有价值的任务。在本文中，我们提出了一种基于动态贝叶斯网络的新型点击模型。该工作的主要贡献是引入满意度的概念来分别建模落地页的相关性和搜索结果页的感知相关性。我们在本文中证明了 DBN 模型优于其他点击模型。

有几个扩展可以提高我们模型的准确性。除了我们已做的初步实验（考虑整个搜索结果页的 URL 进行点击建模），另一个扩展是整合用户在页面上花费的时间，预计这在预测用户满意度方面非常有帮助。我们也可以允许用户在不点击的情况下满意（例如他可能仅通过阅读摘要就满足了请求）。此外，满意度变量可以是连续的而非二值的：对于信息型查询，用户通常在每个页面上找到部分信息，并在整体信息需求被满足时停止。这可以通过引入 $S_i$ 变量之间的依赖来实现。最后，一个更具挑战性的扩展是考虑非线性检查模型：这将需要建模前向和后向跳跃。

大多数现有的点击建模方法受到用于收集点击的搜索引擎的偏差影响，它们主要作为"正反馈"：如果文档从未展示给用户，则文档不会被点击。扩展工作的另一个方向是利用查询平滑来推断额外文档的相关性。



## 致谢

作者感谢 Ralf Gutsche 在点击数据处理方面的宝贵协助。作者还感谢 Georges Dupret、Narayanan Sadagopan 和 Belle Tseng 的富有洞察力的讨论。



## 参考文献

[1] E. Agichtein, E. Brill, S. Dumais, and R. Ragno. Learning user interaction models for predicting web search result preferences. In Proceedings of the 29th annual international ACM SIGIR conference on Research and development in information retrieval (SIGIR), pages 3–10, 2006.

[2] M. Beal and Z. Ghahraman. Variational bayesian learning of directed graphical models with hidden variables. Bayesian Analysis, 1(4):793–832, 2006.

[3] A. Broder. A taxonomy of web search. SIGIR Forum, 36(2):3–10, 2002.

[4] C. Burges, T. Shaked, E. Renshaw, A. Lazier, M. Deeds, N. Hamilton, and G. Hullender. Learning to rank using gradient descent. In Proceedings of the 22nd international conference on Machine learning, pages 89–96, 2005.

[5] Y. Cao, J. Xu, T.-Y. Liu, H. Li, Y. Huang, and H.-W. Hon. Adapting ranking svm to document retrieval. In Proceedings of the 29th annual international ACM SIGIR conference on Research and development in information retrieval, 2006.

[6] B. Carlin and T. Louis. Bayes and Empirical Bayes Methods for Data Analysis. Chapman & Hall/CRC, 2000.

[7] B. Carterette and R. Jones. Evaluating search engines by modeling the relationship between relevance and clicks. In J. Platt, D. Koller, Y. Singer, and S. Roweis, editors, Advances in Neural Information Processing Systems 20, pages 217–224. MIT Press, 2008.

[8] N. Craswell, O. Zoeter, M. Taylor, and B. Ramsey. An experimental comparison of click position-bias models. In WSDM '08: Proceedings of the international conference on Web search and web data mining, pages 87–94. ACM, 2008.

[9] N. M. Dempster, A. Laird, and D. B. Rubin. Maximum likelihood from incomplete data via the EM algorithm. Journal of the Royal Statistical Society B, 39:185–197, 1977.

[10] G. Dupret and B. Piwowarski. User browsing model to predict search engine click data from past observations. In SIGIR 08: Proceedings of the 31st Annual International Conference on Research and Development in Information Retrieval, 2008.

[11] Z. Ghahramani. Learning dynamic bayesian network. In C. L. Giles and M. Gori, editors, Adaptive processing of temporal information, Lecture notes in artificial intelligence. Springer-Verlag, 1998.

[12] K. Jarvelin and J. Kekalainen. Cumulated gain-based evaluation of IR techniques. ACM Transactions on Information Systems, 20(4):422–446, 2002.

[13] T. Joachims. Optimizing search engines using clickthrough data. In ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD), pages 133–142, 2002.

[14] T. Joachims. Evaluating retrieval performance using clickthrough data. In Text mining, pages 79–96, 2003.

[15] T. Joachims, L. A. Granka, B. Pan, H. Hembrooke, and G. Gay. Accurately interpreting clickthrough data as implicit feedback. In Proceedings of the 28th annual international ACM SIGIR conference on Research and development in information retrieval, pages 154–161, 2005.

[16] M. Richardson, E. Dominowska, and R. Ragno. Predicting clicks: estimating the click-through rate for new ads. In Proceedings of the 16th international conference on World Wide Web (WWW), pages 521–530, 2007.

[17] M. Richardson, E. Dominowska, and R. Ragno. Predicting clicks: estimating the click-through rate for new ads. In WWW '07: Proceedings of the 16th international conference on World Wide Web, pages 521–530. ACM, 2007.

[18] S. E. Robertson and S. Walker. Some simple effective approximations to the 2-poisson model for probabilistic weighted retrieval. In Proceedings of the 17th annual international ACM SIGIR conference on Research and development in information retrieval, 1994.

[19] V. Zhang and R. Jones. Comparing click logs and editorial labels for training query rewriting. In Query Log Analysis: Social And Technological Challenges. A workshop at the 16th International World Wide Web Conference, 2007.

[20] Z. Zheng, H. Zha, T. Zhang, O. Chapelle, K. Chen, and G. Sun. A general boosting method and its application to learning ranking functions for web search. In Advances in Neural Information Processing Systems 20, pages 1697–1704. MIT Press, 2008.

[21] D. Zhou, L. Bolelli, J. Li, C. L. Giles, and H. Zha. Learning user clicks in web search. In International Joint Conference on Artificial Intelligence (IJCAI07), 2007.
