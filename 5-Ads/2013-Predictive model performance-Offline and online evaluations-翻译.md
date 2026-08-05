# 预测模型性能：离线与在线评估

> Jeonghee Yi, Ye Chen, Jie Li, Swaraj Sett, Tak W. Yan | Microsoft Corporation, Mountain View, CA

本文研究了用于估计预测模型有效性的评估指标的准确性，分析了离线评估指标与在线实际性能之间存在显著差异的原因。核心内容：

- 分析了 AUC（AUC，Area Under the ROC Curve，ROC 曲线下面积）和 RIG（RIG，Relative Information Gain，相对信息增益）等离线评估指标的特性和局限性，发现它们对预测分数低端区域的误差不敏感，而这恰恰对在线性能影响最大
- 提出了模拟指标（Simulated Metric）这一新的模型评估范式，通过拍卖模拟离线复现在线用户行为，从而更准确地估计点击预测模型的在线性能
- 基于 Bing 搜索引擎的真实广告数据验证了模拟指标的有效性，其在线预测精度远优于传统离线指标

关键发现：

- AUC 和 RIG 对评估数据的类别分布高度敏感，不同分布下的分数不可直接比较；拟合较差的模型可能因负样本集中在低分区域而获得更高的 AUC
- 在低预测点击率（pClick）范围的过度估计对在线性能的负面影响远大于高范围的过度估计，但 AUC 和 RIG 无法捕捉这种差异
- 模拟指标与在线 A/B 测试结果高度一致（CY 提升 39%-40%，Mainline CY 提升 44%-46%），而 AUC 和 RIG 的离线提升与在线结果差异巨大

---

## 摘要

我们研究了用于估计预测模型有效性的评估指标的准确性。离线评估指标是模型在真实数据上预期性能的指标。然而，在实践中，我们经常遇到模型离线和在线性能之间的显著差异。

我们通过在 Bing 搜索引擎的在线广告数据上进行实验，从分析和实证两个角度研究了离线和在线测试中评估指标的特性和行为。我们的发现之一是，某些离线指标如 AUC（ROC 曲线下面积）和 RIG（相对信息增益）总结了模型在整个操作点范围上的性能，有时可能相当误导，并导致离线和在线指标之间的显著差异。例如，对于搜索广告的点击预测模型，预测点击分数在非常低的范围内的误差对在线性能的负面影响远大于其他区域的误差。然而，我们研究的大多数离线指标（包括 AUC 和 RIG）对这种模型行为不敏感。

我们设计了一种新的模型评估范式，模拟预测模型的在线行为。对于由新预测模型选择的一组广告，根据搜索日志中的历史用户行为估计在线用户行为。点击预测模型在搜索广告上的实验结果非常有前景。

**关键词：** evaluation metric, offline evaluation, online evaluation, AUC, RIG, log-likelihood, prediction error, simulated metric, online advertising, sponsored search, click prediction

---

## 1 引言

在机器学习领域，评估指标通常用于判断和比较预测模型在基准数据集上的性能。显然，对其准确性的良好定量评估对于构建成功的预测系统至关重要。尽管已有大量评估指标可用[5, 13]，且特定预测问题可能存在事实上的标准指标，但它们并非没有局限性和缺点。先前的研究表明，某些指标可能高估了偏斜样本的模型性能[9, 10, 14]，且在某些情况下（如交叉验证）存在导致不同结果的指标变体[14]。

对于典型的机器学习问题，训练和评估（或测试）样本从需要建模的总体中随机选择，预测模型在训练样本上构建。然后将学习到的模型应用于评估数据，并使用选定的评估指标衡量模型的质量。这称为离线评估。

此外，高度复杂的现代应用（如 Google 和 Bing 等搜索引擎，以及 Amazon 和 eBay 等在线购物引擎）通常在受控的 A/B 测试平台上对表现最佳的离线模型进行在线评估。在线 A/B 测试平台可以设置两个隔离的测试环境，除了一个使用基线（或控制）模型、另一个使用待测试的新模型外，这两个环境完全相同。它们在相同的时间段内向每个环境发送预定义数量的实时流量。评估在线用户行为的差异（如点击数和每用户搜索次数）以及其他性能指标（如每次搜索收入），以在做出新模型的最终发布决策之前确定差异是否具有统计显著性。这里的假设是，如果新模型提供了更好的结果，在线性能指标会更好。

模型评估中的一个现实问题是，有时离线评估中模型性能的改进在在线评估中未能实现多少，有时甚至出现逆转。与静态离线评估不同，即使在受控环境下的在线测试也是高度动态的，当然，离线建模期间未考虑的许多因素在结果中起作用。尽管如此，这些观察引发了一个问题：是否存在导致这种差异的离线评估指标的根本偏差或局限性。

另一个问题是比较使用不同类型数据（特别是稀有事件数据）构建的预测模型的性能。稀有事件的发生频率远低于对应事件，因此导致类别之间的样本分布偏斜。这在现实世界问题中是相当普遍的现象。稀有事件的示例包括点击网络搜索结果链接、点击展示广告以及点击产品广告后进行购买。先前的研究表明，某些指标可能高估了偏斜样本的模型性能[9]。这些观察引出了以下问题。在这种偏差下，我们如何解释和比较应用于不同类型数据的模型性能？例如，当我们为文本广告和展示广告构建预测模型时，我们能否使用离线指标作为比较指标来预测其真实性能？假设我们知道一个模型的真实性能，并获得另一个模型的等效离线指标。我们能否估计另一个模型的真实性能？如果不能，我们应该使用什么类型的指标？

我们提出了一种新的模型评估范式：模拟指标。我们实现了拍卖模拟以离线模拟在线行为，并使用模拟指标来估计点击预测模型的在线模型性能。由于模拟指标旨在模拟在线行为，我们预计它们受性能差异问题的影响较小。此外，由于模拟指标直接估计在线指标（如用户点击率（CTR，Click-Through Rate）），即使它们是针对不同类型数据构建的模型，也可以直接比较。

本文的贡献有四个方面：

* 我们分析了离线评估指标的特性和局限性，并分享了我们关于其在离线和在线数据上行为的发现。
* 我们分享了在 Bing 搜索引擎上为生产在线广告系统训练、评估和部署点击预测模型的经验，并为大规模预测模型评估提供了最佳实践指南。
* 据我们所知，这是公开文献中第一篇提出并应用模拟指标作为模型评估范式的论文。
* 据我们所知，这同样是公开文献中第一篇分析导致预测模型在线和离线性能差异的离线评估指标行为问题的论文。

本文的其余部分组织如下。在下一节中，我们简要回顾在线广告和二分类误差测量。在第 3 节中，我们调查了公开文献中的预测模型评估指标。然后在第 4 节中回顾了一些在调查文献中经常使用的指标。在第 5 节中，我们描述了 AUC 和 RIG 度量在大规模赞助搜索点击预测模型上的问题和局限性。在第 6 节中，我们讨论了部署在 Bing 搜索引擎实时生产流量上的模型的离线和在线性能差异。最后，我们总结了我们的发现，并基于我们的分析和来自在线广告数据的真实世界经验提出了最佳实践指南。

## 2 预备知识

我们研究的目标应用是在线广告。本研究中讨论的一些问题领域可能是特定于领域的。在本节中，我们简要回顾在线广告的主要领域，感兴趣的读者可以参考下面提供的优秀教程。

### 2.1 在线广告

赞助（或付费）搜索[11, 21, 28, 34]（如 Google AdWords 和 Bing 的 Paid Search）是在搜索引擎结果页面（SERP）上在算法搜索结果旁边展示广告的搜索广告。赞助搜索接触到主动在线寻找产品和服务信息的人，因此与其他类型的广告相比具有相对较高的点击率（CTR）。

广告主通过广义二价（GSP）拍卖[11]对关键词进行竞价。具有最高排名分数（ $r$ ）的竞价者赢得拍卖：

$$
r = b \cdot p^{\alpha} \qquad (1)
$$

其中 $b$ 是出价金额， $p$ 是估计的位置无偏 CTR， $\alpha$ 是一个参数，称为点击投资力度。如果 $\alpha > 1$ ，拍卖倾向于估计 CTR 较高的广告，否则倾向于出价较高的广告。排名分数是按每次点击出价加权的估计 CTR。

广告按估计排名分数的降序分配，拍卖获胜者仅在人们点击其广告时为广告展示支付每次点击价格（a.k.a. cost per click，或 CPC）。在 GSP 拍卖中，CPC 取决于下一个更高竞价者的出价金额 $c_i$ ：

$$
c_i = \frac{b_{i+1} \cdot p_{i+1}^{\alpha}}{p_i^{\alpha}}
$$

用户点击高度依赖于广告的位置[7, 15]。通常，展示在算法搜索结果上方部分（称为主栏）的广告比展示在算法结果右侧（称为侧栏）的广告获得更高的 CTR。在相同部分内，广告位置越高，同一广告获得的点击越多。

展示广告[32]是出现在网站、内容页面或应用程序（如即时通讯、电子邮件等）上的图形广告。上下文广告[7]（如 Google AdSense 或 Bing 的 Contextual Search）是上下文优化的广告，放置在发布者的网站上，通常具有发布者网站的定制外观。

准确估计用户点击概率对于广告交换的效率至关重要[25]。估计点击概率的问题已针对算法搜索[24, 30, 31, 36]和广告[6, 8, 16, 27]进行了广泛研究。

### 2.2 二分类误差测量

考虑一个特征向量 $x$ 和观测到的二值响应 $y \in \{0, 1\}$ 。 $x$ 被视为随机向量 $X$ 的实现， $y$ 被视为伯努利随机变量 $Y$ 。类别 1 概率 $\eta = P[Y = 1]$ 是 $x$ 的函数： $\eta(x) = P[Y = 1 | X = x]$ 。二分类器将 $\eta(x) > c$ 的样本预测为类别 1，其中 $c$ 是一个参数；否则预测为类别 0。

预测的有效性使用各种标准估计，包括主要标准（如预测误差）和代理标准（如对数损失和平方误差损失）[4]。主要标准用于直接估计类别，代理标准用于估计类别预测概率。预测（或误分类）误差在估计模型性能方面本质上是不稳定的。相反，对数损失和平方误差损失通常用于概率估计和提升，定义如下：

* **对数损失：**

$$
L(y|p) = -\log(p^y (1-p)^{1-y}) = -y\log(p) - (1-y)\log(1-p)
$$

* **平方误差损失（或二次损失）：**

$$
L(y|p) = (y-p)^2 = y(1-p)^2 + (1-y)p^2
$$

其中 $p$ 是 $\eta(x)$ 的估计概率。平方误差损失的等式仅对二分类器成立，即 $y \in \{0, 1\}$ 。

对数损失是伯努利模型的负对数似然。其期望值 $-\eta\log(p) - (1-\eta)\log(1-p)$ 称为 Kullback-Leibler 损失[19]或交叉熵。

### 2.3 实验数据集

在整篇论文中，我们展示了 Microsoft Bing 搜索引擎上点击预测模型性能的激励示例和分析。我们从 2012 年 6 月至 8 月期间的 Bing 赞助搜索日志中采样数据。我们使用了两组数据：一组从 Bing 的付费搜索数据中采样，另一组从 Microsoft 发布者网络上合作伙伴网站的上下文广告中采样。

## 3 指标调查

我们研究了 2011 年和 2012 年国际万维网会议（WWW）、ACM 国际网络搜索和数据挖掘会议（WSDM）以及 ACM SIGKDD 国际知识发现和数据挖掘会议（SIGKDD）论文集中算法搜索和在线广告领域的论文。我们手动分类了论文的主题领域和它们使用的评估指标。表 1 总结了结果。

**表 1：2011 和 2012 年 WWW、WSDM 和 SIGKDD 会议算法搜索和在线广告领域论文使用的评估指标总结**

| | 离线指标 | | | | | | 在线指标 | 总计 |
|---|---|---|---|---|---|---|---|---|
| | 概率 | 对数似然 | PE | NDCG | IR | 其他 | | |
| 推荐 | 1 | 1 | 2 | 3 | 3 | 1 | | 11 |
| 搜索 | 1 | 2 | | 10 | 16 | 4 | 1 | 34 |
| 在线广告 | 6 | 1 | | 2 | 1 | 6 | 2 | 18 |
| CTR 估计 | | | 3 | 1 | 2 | | 1 | 7 |
| 总计 | 8 | 7 | 5 | 15 | 20 | 11 | 4 | 70 |

我们发现了四个主要主题类别：推荐、搜索、在线广告和 CTR 估计。搜索和在线广告进一步细分为子类别。高级别的计数是其子类别计数的总和。

指标的类别分为离线和在线指标。在线指标包括模型性能统计数据（如广告展示量、广告覆盖率）和用户反应指标（如 CTR 和用户会话长度）。离线指标分为以下六种类型[18, 1, 26, 35]：

* **基于概率：** AUC、MLE（最大似然估计）等
* **基于对数似然：** RIG、交叉熵等
* **PE（预测误差）：** MSE（均方误差）、MAE（平均绝对误差）、RMSE（均方根误差）等
* **基于 DCG：** DCG（折扣累积增益）、NDCG（归一化 DCG）、RDCG（相对 DCG）等
* **IR（信息检索）：** 精确率/召回率、F 值、AP（平均精度）、MAP（平均精度均值）、RBP（基于排名的精度）、MRR（平均倒数排名）等
* **其他：** 不属于其他类别的所有内容

NDCG 是搜索排序算法的事实标准首选指标。尽管基于概率的指标在广告领域相对受欢迎，但该领域仍然不存在像 NDCG 对搜索排序问题那样主导的单一指标。尽管先前的研究表明 AUC 更可靠[3, 29, 22]，但我们只找到 2 篇测量 AUC 的论文。我们将 AUC 应用于广告领域的点击预测（pClick）问题，发现它是最可靠的指标之一，但并非没有问题。我们将在下一节详细讨论各个指标。

## 4 评估指标

我们专注于主要点击预测问题所用指标的审查。点击预测模型估计给定查询的广告的位置无偏 CTR。我们将其视为二分类问题。

我们将 NDCG 从审查中排除，因为它旨在偏好将更多相关结果排在更靠前位置的排序算法。如第 2.1 节所述，在搜索广告中，排名不是由 pClick（即估计点击）分数决定，而是由排名分数决定。因此，使用 NDCG 通过排名顺序衡量 pClick 的性能是不合适的。

我们还将精确率-召回率（PR）分析从审查中排除，因为 PR 曲线和 ROC（受试者工作特征）曲线之间存在联系，因此 PR 曲线和 AUC 之间也存在联系[9]。Davis 和 Goadrich 表明，当且仅当曲线在 PR 空间中占优时，曲线才在 ROC 空间中占优[9]。

### 4.1 AUC

考虑一个产生事件概率 $p$ 的二分类器。 $p$ 和 $1-p$ （事件不发生的概率）代表每个案例属于两个事件之一的程度。阈值对于预测类别成员资格是必要的。AUC（ROC 曲线下面积）[12, 33]提供了在分类器上应用的所有可能阈值范围内的判别度量。

ROC 曲线是二分类器的灵敏度（或 TPR）作为其阈值变化的函数的图形描述，相对于误报率（或 FPR）。AUC 计算如下：

* 按模型预测分数降序排列记录
* 对每个预测值计算 TPR 和 FPR
* 绘制 ROC 曲线
* 使用梯形近似计算 AUC

经验上，AUC 是任何评分模型预测能力的良好可靠指标。对于赞助搜索，AUC（尤其是仅在主栏广告上测量的 AUC）是模型预测能力最可靠的指标之一。好模型（AUC > 0.8）通常在 AUC 提高 1 个点（0.01）时具有统计显著的改进。

使用 AUC 的好处包括：

* AUC 提供了一个单一数字的判别分数，总结了模型在所有可能阈值范围上的整体性能。这避免了阈值选择中的主观性。
* 它适用于任何具有评分函数的预测模型。
* AUC 分数有界于 $[0, 1]$ ，随机预测为 0.5，完美预测为 1。
* AUC 可用于预测模型的离线和在线监控。

经验上，它在估计赞助搜索 pClick 模型有效性方面也表现良好。它与 AUC 一起是最可靠的指标之一。

### 4.2 RIG

RIG（相对信息增益）是对数损失的线性变换[15, 36]：

$$
\mathrm{RIG} = 1 - \frac{\mathrm{log\ loss}}{\mathrm{Entropy}(\gamma)} = 1 - \frac{-c \cdot \log(p) - (1-c)\log(1-p)}{-\gamma \cdot \log(\gamma) - (1-\gamma)\log(1-\gamma)} \qquad (2)
$$

其中 $c$ 和 $p$ 分别代表观测到的点击和 pClick。 $\gamma$ 代表评估数据的 CTR。对数损失代表预期的点击概率。最小化对数损失意味着 pClick 应收敛到预期点击率，RIG 分数增加。

### 4.3 MSE

MSE（均方误差）测量平方损失的平均值：

$$
\mathrm{MSE}(P) = \frac{\sum_{i=1}^{n} (c_i \cdot (1-p_i) + (1-c_i) \cdot p_i)^2}{n}
$$

其中 $p_i$ 和 $c_i$ 分别是样本 $i$ 的 pClick 和观测到的点击。

NMSE（归一化 MSE）是按 CTR $\gamma$ 归一化的 MSE：

$$
\mathrm{NMSE}(P) = \frac{\mathrm{MSE}(P)}{\gamma \cdot (1-\gamma)}
$$

### 4.4 MAE

平均绝对误差（MAE）由下式给出：

$$
\mathrm{MAE}(P) = \frac{1}{n} \sum_{i=1}^{n} e_i
$$

其中 $e_i = |p_i - c_i|$ 是绝对误差。MAE 无论到关键操作点的距离如何，都同等加权预测和观测之间的距离。MAE 通常用于时间序列分析中的预测误差测量。

### 4.5 预测误差

预测误差（PE）测量按 CTR 归一化的平均 pClick：

$$
\mathrm{PE}(P) = \frac{\mathrm{avg}(p)}{\gamma} - 1
$$

当平均 pClick 分数恰好估计了 CTR 时，PE 变为零。另一方面，即使估计的 pClick 分数相当不准确，只要平均值与底层 CTR 相似，PE 仍可能非常接近零。这使得预测误差相当不稳定，不能可靠地用于估计分类精度。

### 4.6 模拟指标

尽管在受控 A/B 测试环境上的在线实验通过用户参与提供了被比较模型的真实性能指标，但 A/B 测试环境预设了固定的参数值集，因此测试环境上的模型性能指标仅适用于给定的操作点集。在众多操作点集上进行在线实验是不切实际的，因为在线实验不仅非常耗时，而且如果新模型表现不佳，在用户体验和收入方面都可能非常昂贵。

Kumar 等人为联邦搜索开发了在线性能模拟方法[20]。与其使用昂贵且耗时的在线评估，不如使用历史在线用户参与数据模拟模型在整个可行操作点范围上的性能。

拍卖模拟首先离线重新运行给定查询的广告拍卖，基于新模型预测分数和/或各种操作点集选择一组广告。我们使用赞助搜索点击日志数据实现了拍卖模拟[15]并产生了各种模拟指标。在模拟期间，使用日志中可用的给定（查询、广告）对的历史用户点击估计用户点击：

* 如果在日志中找到了（查询、广告）对的用户点击数据，且在与模拟广告位置相同的广告展示位置（称为广告位置），则直接使用历史 CTR 作为预期 CTR。
* 如果在日志中找到了（查询、广告）对，但模拟广告位置与日志中的位置不同，则预期 CTR 通过位置偏差的历史 CTR（或点击曲线）进行校准。通常，同一（查询、广告）对的主栏广告获得的 CTR 远高于侧栏广告，且在同一广告块内较高位置的广告获得更高的 CTR。
* 如果预测的（查询、广告）对未出现在历史日志中，则使用该广告位置上广告的平均 CTR（称为参考 CTR）。

点击曲线和参考 CTR 来自搜索广告日志中的历史用户响应。经验上，拍卖模拟为给定操作点集产生了由新模型选择的高度准确的广告集。模拟指标通常被证明是在线模型性能最强大的离线估计器之一。

## 5 真实世界问题中的指标经验

在本节中，我们详细分析了搜索广告点击预测背景下各种指标的行为、局限性和缺点。注意，我们并非建议由于这些局限性和缺点而完全放弃这些指标。我们建议在指标可能产生误导性估计的情况下谨慎应用和解释这些指标。

### 5.1 AUC

虽然 AUC 是评估预测模型性能的相当可靠的方法，但在样本数据的某些条件下仍然存在缺点。AUC 是模型性能充分测试指标的假设需要重新审视[23]。

首先，它忽略了预测概率值。这使其对保持排名的预测概率转换不敏感。一方面，这可能是一个优势，因为它能够比较在不同测量尺度上产生数值结果的测试。另一方面，两个测试也可能产生截然不同的预测输出，但具有相似的 AUC 分数。拟合较差的模型（高估或低估所有预测）可能具有良好的判别能力[17]，而如果存在的概率仅适度高于不存在的概率，拟合良好的模型可能判别能力较差。

表 2 展示了一个拟合较差但具有更高 AUC 分数的示例——大量负样本集中在 pClick 分数范围的低端，从而降低了 CTR。这具有在相对较高的 pClick 分数范围内降低 FPR 的效果，从而提高了 AUC 分数。

**表 2：AUC 异常 1：在大量负样本集中在 pClick 分数范围低端的情况下，拟合较差的模型甚至具有更高的 AUC。（第一个表显示了拟合更好的模型。）**

| 平均 pClick | 点击数 | 未点击数 | 实际 CTR | TPR | FPR | 梯形 | 实际CTR |
|---|---|---|---|---|---|---|---|
| 0.030000 | 300 | 9,700 | 0.030000 | 0.2500 | 0.0086 | 0.0011 | 1.0 |
| 0.020000 | 200 | 9,800 | 0.020000 | 0.4167 | 0.0173 | 0.0029 | 1.0 |
| 0.010000 | 100 | 9,900 | 0.010000 | 0.5000 | 0.0260 | 0.0040 | 1.0 |
| 0.005000 | 500 | 99,500 | 0.005000 | 0.9167 | 0.1142 | 0.0624 | 1.0 |
| 0.000100 | 100 | 999,900 | 0.000100 | 1.0000 | 1.0000 | 0.8499 | 1.0 |
| 总计 | 1,200 | 1,128,800 | | | AUC | 0.9193 | |

| 平均 pClick | 点击数 | 未点击数 | 实际 CTR | TPR | FPR | 梯形 | 实际CTR |
|---|---|---|---|---|---|---|---|
| 0.030000 | 300 | 9,700 | 0.030000 | 0.2500 | 0.0010 | 0.0001 | 1.0 |
| 0.020000 | 200 | 9,800 | 0.020000 | 0.4167 | 0.0019 | 0.0003 | 1.0 |
| 0.010000 | 100 | 9,900 | 0.010000 | 0.5000 | 0.0029 | 0.0004 | 1.0 |
| 0.005000 | 500 | 99,500 | 0.005000 | 0.9167 | 0.0127 | 0.0070 | 1.0 |
| 0.000100 | 100 | 9,999,000 | 0.000010 | 1.0000 | 1.0000 | 0.9461 | 10.0 |
| 总计 | 1,200 | 10,127,900 | | | AUC | 0.9540 | |

其次，它总结了整个 ROC 空间范围上的测试性能，包括人们很少操作的区域。例如，对于赞助搜索，将广告放置在主栏会显著影响 CTR，而一旦广告展示在主栏上或根本不展示，预测 CTR 与实际 CTR 的拟合程度就不那么重要了。换句话说，ROC 空间的极右和极左区域通常不太有用。Baker 和 Pinsky 提出了部分 ROC 曲线作为整个 ROC 曲线的替代方案[2]。

已经观察到，更高的 AUC 并不一定意味着更好的排名。如表 3 所示，FPR 两端样本分布的变化对 AUC 分数影响相当大。然而，对 CTR 方面模型性能的影响可能相同，尤其是在实际操作点处。

**表 3：AUC 异常 2：FPR 两端样本分布的变化对 AUC 分数影响相当大，尽管在实际操作点处的实际模型性能相当相似。**

| 平均 pClick | 点击数 | 未点击数 | 实际 CTR | TPR | FPR | 梯形 | 实际CTR |
|---|---|---|---|---|---|---|---|
| 0.030000 | 3,000 | 97,000 | 0.030000 | 0.4545 | 0.0093 | 0.0021 | 1.0 |
| 0.020000 | 2,000 | 98,000 | 0.020000 | 0.7576 | 0.0188 | 0.0057 | 1.0 |
| 0.010000 | 1,000 | 99,000 | 0.010000 | 0.9091 | 0.0283 | 0.0079 | 1.0 |
| 0.005000 | 500 | 99,500 | 0.005000 | 0.9848 | 0.0379 | 0.0091 | 1.0 |
| 0.000010 | 100 | 9,999,900 | 0.000010 | 1.0000 | 1.0000 | 0.9548 | 1.0 |
| 总计 | 6,600 | 10,392,500 | 0.000635 | | AUC | 0.9797 | |

| 平均 pClick | 点击数 | 未点击数 | 实际 CTR | TPR | FPR | 梯形 | 实际CTR |
|---|---|---|---|---|---|---|---|
| 0.030000 | 3,000 | 97,000 | 0.030000 | 0.4545 | 0.0093 | 0.0021 | 1.0 |
| 0.020000 | 2,000 | 98,000 | 0.020000 | 0.7576 | 0.0188 | 0.0057 | 1.0 |
| 0.010000 | 1,000 | 99,000 | 0.010000 | 0.9091 | 0.0283 | 0.0079 | 1.0 |
| 0.005000 | 100 | 9,999,900 | 0.000010 | 0.9242 | 0.9904 | 0.8820 | 500.0 |
| 0.000010 | 500 | 99,500 | 0.005000 | 1.0000 | 1.0000 | 0.0092 | 0.0 |
| 总计 | 6,600 | 10,392,500 | 0.000635 | | AUC | 0.9069 | |

第三，它同等加权遗漏和误报错误。例如，在赞助搜索的背景下，未将最优广告放置在主栏（遗漏错误）的惩罚远超过放置次优广告（误报错误）的惩罚。当误分类成本不相等时，总结所有阈值值是有缺陷的。

最后，AUC 高度依赖于底层数据分布。为具有不同负样本率的两个数据集计算的 AUC 度量会相当不同。见表 4。具有较低内在 CTR 的拟合较差模型与拟合良好模型具有相同的 AUC。这也意味着，对于用较高速率负样本训练的模型，更高的 AUC 分数并不一定意味着模型具有更好的预测性能。

**表 4：AUC 异常 3：拟合较差的模型与拟合良好的模型具有相同的 AUC。（第一个表显示了拟合更好的模型。）**

| 平均 pClick | 点击数 | 未点击数 | 实际 CTR | TPR | FPR | 梯形 | 实际CTR |
|---|---|---|---|---|---|---|---|
| 0.030000 | 300 | 9,700 | 0.030000 | 0.2500 | 0.0086 | 0.0011 | 1.0 |
| 0.020000 | 200 | 9,800 | 0.020000 | 0.4167 | 0.0173 | 0.0029 | 1.0 |
| 0.010000 | 100 | 9,900 | 0.010000 | 0.5000 | 0.0260 | 0.0040 | 1.0 |
| 0.005000 | 500 | 99,500 | 0.005000 | 0.9167 | 0.1142 | 0.0624 | 1.0 |
| 0.000100 | 100 | 999,900 | 0.000100 | 1.0000 | 1.0000 | 0.8499 | 1.0 |
| 总计 | 1,200 | 1,128,800 | | | AUC | 0.9193 | |

| 平均 pClick | 点击数 | 未点击数 | 实际 CTR | TPR | FPR | 梯形 | 实际CTR |
|---|---|---|---|---|---|---|---|
| 0.030000 | 300 | 97,000 | 0.003083 | 0.2500 | 0.0086 | 0.0011 | 9.7 |
| 0.020000 | 200 | 98,000 | 0.002037 | 0.4167 | 0.0173 | 0.0029 | 9.8 |
| 0.010000 | 100 | 99,000 | 0.001009 | 0.5000 | 0.0260 | 0.0040 | 9.9 |
| 0.005000 | 500 | 995,000 | 0.000502 | 0.9167 | 0.1142 | 0.0624 | 10.0 |
| 0.000100 | 100 | 9,999,000 | 0.000010 | 1.0000 | 1.0000 | 0.8499 | 10.0 |
| 总计 | 1,200 | 11,288,000 | | | AUC | 0.9193 | |

> **图 1：赞助搜索和上下文广告的 ROC 曲线**

图 1 绘制了赞助搜索和上下文广告的 pClick 模型的 ROC 曲线。如图所示，上下文广告模型的 AUC 分数比赞助搜索的 AUC 高约 3%，即使前者不太准确：赞助搜索的实际 $\frac{\mathrm{avg\ pClick}}{\mathrm{CTR}} = 1.02$ ，而上下文广告为 0.86。

### 5.2 RIG

RIG 的一个问题是，与 AUC 一样，它也对评估数据的底层分布高度敏感。由于评估数据的 RIG 分数范围根据数据分布变化相当大，仅凭 RIG 分数可能无法判断预测模型有多好。

> **图 2：不同 CTR 样本数据上的 RIG 和 PE 分数：RIG 分数随 CTR 增加而下降。**

图 2 说明了 RIG（实线）和 PE（虚线）在典型感兴趣 CTR 范围上如何变化。我们观察到，即使使用相同的预测模型，RIG 分数也会随着数据集 CTR 的增加而下降。图 2 中绘制的预测误差大致表明预测分数与真实 CTR 的接近程度。如预期，低 pClick 分数范围的点击预测误差更高。

这种行为与我们之前在不同内在 CTR 水平的各种点击预测数据集上的观察一致。这些观察在实践中表明以下几点：

* 不应直接使用 RIG 分数的面值来比较两个预测模型的性能，如果分数来自具有不同分布的多个数据集。
* RIG 分数可用于比较在同一数据上训练和测试的多个模型的相对性能。
* 单独的 RIG 分数不足以估计预测模型的性能，因为该分数不仅取决于模型性能的质量，还受数据分布的严重偏差影响。

## 6 离线和在线性能差异

实践中离线评估指标的一个更显著的问题是离线和在线测试之间的性能差异。有时在离线评估中取得显著收益的预测模型在部署到在线测试环境时表现不佳，有时甚至表现更差。

表 5 总结了使用 Bing 搜索引擎赞助搜索数据构建的点击预测模型的离线和在线指标，并在 Bing 上的在线 A/B 测试环境中使用实时用户流量进行了测试。点击量（CY）是在线用户点击的指标，衡量每次搜索页面浏览的广告点击数。Mainline CY 是每次搜索页面浏览的主栏广告点击数。新模型在在线环境中经历了用户点击相对于基线模型的显著下降，尽管两者在离线指标上都表现出显著收益。

**表 5：新模型（model-2）与基线模型的离线和在线指标**

| | 离线指标 | | 在线指标 | |
|---|---|---|---|---|
| | AUC | RIG | CY | Mainline CY |
| model-2 指标 | 8.6% | 19.5% | -9.96% | -8.07% |

> **图 3：典型感兴趣 pClick 范围上对数损失的相对贡献**

图 3 比较了两个点击预测模型（model-1 为基线，model-2 为测试）在典型感兴趣 pClick 分数范围内每个分位数的对数损失[4]。Model-2 在较低 pClick 分数范围的分位数上大幅高估了 pClick 分数，而在较高 pClick 分数的分位数上高估程度小得多。图 4 绘制了具有相似模式的相同数据的预测误差。

> **图 4：pClick 预测误差（PE）**

在实践中，高范围 pClick 分数上的点击概率过度估计对在线性能的影响小于低 pClick 分数范围上的过度估计，因为高 pClick 分数范围的广告最可能被任何一个模型选择。一旦展示给用户，用户点击主要由广告位置和广告的相关性决定，而不是分配的 pClick 分数。

另一方面，低 pClick 范围上的 pClick 分数过度估计可能对在线性能产生显著负面影响，因为它给低质量广告更高的被选择机会 compared to 基模型。由于高估 pClick 分数而选择的较低质量广告将导致较低的用户点击率，从而损害在线指标。

大多数离线指标（包括 RIG 和 AUC）无法捕捉这些行为，因为这些指标在整个 pClick 分数范围内累积影响。

### 6.1 模拟指标

我们通过第 4.6 节描述的拍卖模拟计算了模拟指标。模拟点击指标与离线和在线指标的实验结果总结在表 6 中。我们首先训练了一个新模型，并通过基于历史日志数据的拍卖模拟优化了提供最佳预期用户点击指标的参数设置。表中报告了模型最佳性能操作点的点击指标作为模拟指标。然后我们使用最佳设置设置了 A/B 测试环境，并运行了在线 A/B 测试实验以获得在线指标。您可以看到在线指标与模拟指标高度一致，而 AUC 和 RIG 指标的改进差异巨大。

**表 6：新模型（model-2）与基线模型的模拟指标**

| | 离线指标 | | 模拟指标 | | 在线指标 | |
|---|---|---|---|---|---|---|
| | AUC | RIG | CY | Mainline CY | CY | Mainline CY |
| model-2 指标 | 0.2% | 78% | 40% | 46% | 39% | 44% |

## 7 总结与讨论

我们回顾并研究了各种预测模型离线指标的行为，特别是在搜索广告点击预测的背景下。总结如下：

* 模拟指标是预测点击预测模型在线性能最可靠的指标之一。在线行为的模拟对于各种任务（包括性能估计和拍卖优化）非常有用。
* 对于点击预测模型，AUC 比其他离线指标更好地估计了模型有效性。特别是，仅在主栏广告上测量的 AUC 对搜索广告最可靠。然而，仅 AUC 不足以可靠地估计模型性能。
* RIG 和 AUC 都对评估数据的类别分布高度敏感。
* 如果评估数据的类别分布不同，通过 AUC 或 RIG 分数进行模型性能的交叉比较可能具有误导性。
* 建议在各种分位数上测量模型性能，并仔细分析分位数范围上的模型行为变化将如何影响在线环境。可以一起审查各种指标以发现结果中的任何不匹配，这可能表明指标中存在某些问题。

---

## 参考文献

[1] A. Ashkan and C. L.A. Alarke. On the informativeness of cascade and intent-aware effectiveness measures. In Proc. of the WWW Conference, pages 407–416, 2011.

[2] S. G. Baker and P. F. Pinsky. A proposed design and analysis for comparing digital and analog mammography: social receiver operating characteristic methods for cancer screening. Journal of the American Statistical Association, 96(454):421–428, 2001.

[3] J. R. Beck and E.K. Shultz. The use of relative operating characteristic (roc) curves in test performance evaluation. Archive of Pathological Lab Medicine, 110(10):13–20, Oct. 1986.

[4] A. Buja, W. Stuetzle, and Y. Shen. Loss functions for binary class probability estimation: Structure and applications. Technical report, 2003.

[5] R. Caruana and A. Niculescu-Mizil. Data mining in metric space: an empirical analysis of supervised learning performance criteria. In ACM SIGKDD Conference, pages 69–78, 2004.

[6] D. Chakrabarti, D. Agarwal, and V. Josifovski. Contextual advertising by combining relevance with click feedback. In Proc. of the WWW Conference, 2008.

[7] Y. Chen, P. Berkhin, J. Li, S. Wan, and T. W. Yan. Fast and cost-efficient bid estimation for contextual ads. In Proc. of the WWW Conference, 2012.

[8] Y. Chen, D. Pavlov, M. Kapralov, and J. F. Canny. Factor modeling for advertisement targeting. In Proc. of NIPS, 2009.

[9] J. Davis and M. Goadrich. The relationship between precision-recall and roc curves. In Proc. of the 23rd ICML Conference, pages 233–240, 2006.

[10] C. Drummond and R. Holte. Explicitly representing expected cost: an alternative to roc representation. In ACM SIGKDD Conference, pages 198–207, 2000.

[11] B. Edelman, M. Ostrovsky, and M. Schwarz. Internet advertising and the generalized second-price auction: Selling billions of dollars worth of keywords. American Economic Review, 97(1):242–259, 2007.

[12] T. Fawcett. Roc graphs: Notes and practical considerations for data mining researchers, 2003.

[13] C. Ferry, J. Hernandez-Orallo, and R. Modriou. An empirical comparison of performance measures for classification. Pattern Recognition Letters, 30:27–38, 2009.

[14] G. Forman and M. Scholz. Apples-to-apples in cross-validation studies: pitfalls in classifier performance measurement. ACM SIGKDD Explorations Newsletter, 12:49–57, June 2010.

[15] T. Graepel, J.Q. Candela, T. Borchert, and R. Herbrich. Web-scale bayesian click-through rate prediction for sponsored search advertising in microsoft's bing search engine. In Proc. of the 27th ICML Conference, 2010.

[16] N. Gupta, U. Khurana, T. Lee, and S. Nawathe. Optimizing display advertisements based on historic user trails. In Proc. of SIGIR Workshop on Internet Advertising.

[17] D.W. Hosmer and S. Lemeshow. Applied logistic regression. Wiley-Interscience Publication, 2002.

[18] K. Jarvelin and J. Kekalainen. Ir evaluation methods for retrieving highly relevant documents. In Proc. of the ACM SIGIR Conference, pages 41–48, 2000.

[19] S. Kullback and R.A. Leibler. On information and sufficiency. The Annals of Mathematical Statistics, 22(1):79–86, 1951.

[20] A. Kumar, K. Pattabiraman, D. Brand, and T. Kanungo. Model characterization curves for federated search using click-logs: Predicting user engagement metrics for the span of feasible operating points. In Proc. of the WWW Conference, pages 67–76, 2011.

[21] S. Lahaie, D. M. Pennock, A. Saberi, and R. V. Vohra. Sponsored search auctions. Algorithmic Game Theory, 2007.

[22] P. Langley. Crafting papers on machine learning. In Proc. of the 17th ICML Conference, pages 1207–1212, 2000.

[23] J. M. Lobo, A. Jimenez-Valverde, and R. Real. Auc: a misleading measure of the performance of predictive distribution models. Global Ecology and Biogeography, 17(2):145–151, 2008.

[24] V. Murdock, M. Ciaramita, and V. Plachouras. Online learning from click data for sponsored search. In Proc. of the WWW Conference, 2008.

[25] R. McAfee. The design of advertising exchanges. Review of Industrial Organization, pages 1–17, 2007.

[26] A. Moffat and J. Zobel. Rank-biased precision for measurement of retrieval effectiveness. ACM Trans. on Information Systems, 27(1):1–27, 2008.

[27] V. Murdock, M. Ciaramita, and V. Plachouras. A noisy-channel approach to contextual advertising. In Proc. of the Int'l Workshop on Data Mining for Online Advertising and Internet Economy, 2007.

[28] P. Papadimitriou and H. Garcia-Molina. Sponsored search auctions with conflict constraints. In Proc. of the ACM WSDM Conference, pages 4–14, 2012.

[29] F. Provost, T. Fawcett, and R. Kohavi. The case against accuracy estimation for comparing induction algorithms. In Proc. of the 15th ICML Conference, pages 445–453, 1998.

[30] M. Regelson and D. C. Fain. Predicting click-through rate using keyword clusters. In Proc. of the ACM Electronic Commerce Conference, 2007.

[31] M. Richardson, E. Dominowska, and R. Ragno. Predicting clicks: estimating the click-through rate for new ads. In Proc. of the WWW Conference, 2007.

[32] R. Rosales, H. Cheng, and E. Manavoglu. Post-click conversion modeling and analysis for non-guaranteed delivery display advertising. In Proc. of the ACM WSDM Conference, 2012.

[33] John A. Swets. Measuring the accuracy of diagnostic systems. Science, 240(4857):1285–93, 2008.

[34] H. R. Varian. Position auctions. Int'l Journal of Industrial Organization, 25(6):1163–1178, 2007.

[35] E. M. Voorhees. The trec-8 question answering track report. In Proc. of the TREC Conference, 1999.

[36] C. Xiong, T. Wang, W. Ding, Y. Shen, and T. Liu. Relational click prediction for sponsored search. In Proc. of the ACM WSDM Conference, pages 493–502, 2012.
