# 互联网广告与广义第二价格拍卖：价值数十亿美元的关键词销售

> Benjamin Edelman, Michael Ostrovsky, Michael Schwarz | Harvard University; Stanford University; Yahoo! Research

本文研究了搜索引擎用于销售在线广告的广义第二价格拍卖（GSP）机制。核心内容：

- 问题背景：GSP 是搜索引擎销售在线广告的主导交易机制，2005 年 Google 超过 98% 的收入来自 GSP 拍卖，Yahoo! 超过一半的收入也来自 GSP 拍卖
- 机制分析：GSP 看似类似于 VCG 机制，但性质截然不同——GSP 一般不存在优势策略均衡，真实报价不是 GSP 的均衡
- 核心结论：证明了对应于 GSP 的广义英式拍卖存在唯一均衡，该均衡是事后均衡，所有参与者的收益与 VCG 的优势策略均衡相同
- 实际意义：GSP 拍卖的年收入约 100 亿美元，该机制是从低效市场制度逐步演化而来的 [20]，具有显著的理论性质和巨大的实际影响力

关键发现：

- 局部无嫉妒均衡的存在性：GSP 的局部无嫉妒均衡集包含一个均衡，其中参与者的收益与 VCG 拍卖的优势策略均衡相同，尽管两者的出价和支付规则截然不同
- 均衡的稳健性：广义英式拍卖的均衡是事后均衡，不依赖于竞标者对彼此类型的信念，这使其成为实践中极少数具有此性质的非优势策略可解机制之一
- 机制演化的历史教训：互联网广告市场从 1994 年的按展示付费、1997 年的广义第一价格拍卖，逐步演化到 2002 年的 GSP 机制，其演化速度远快于医疗匹配市场

---

## 摘要

我们研究了"广义第二价格"（GSP）拍卖，这是一种搜索引擎用于销售在线广告的新机制。虽然 GSP 看起来类似于 Vickrey-Clarke-Groves（VCG）机制，但其性质截然不同。与 VCG 机制不同，GSP 一般不存在优势策略均衡，真实报价不是 GSP 的均衡。为了分析 GSP 的性质，我们描述了对应于 GSP 的广义英式拍卖，并证明它存在唯一均衡。这是一个事后均衡，所有参与者的收益与 VCG 的优势策略均衡相同。（JEL D44, L81, M37）

## 关键词

Generalized Second-Price Auction, Internet Advertising, Sponsored Search, Mechanism Design, VCG Mechanism

## 1 引言

本文研究了一种新的拍卖机制，我们称之为"广义第二价格"拍卖，或 GSP。GSP 是为在线广告市场的独特环境量身定制的，此前机制设计文献中既没有研究过这种环境，也没有研究过这种机制。虽然研究新机制的性质本身通常就很有趣，但我们的研究兴趣也受到 GSP 商业成功的推动。它是一个庞大且快速增长行业中的主导交易机制。例如，Google 2005 年的总收入为 61.4 亿美元，超过 98% 的收入来自 GSP 拍卖。Yahoo! 2005 年的总收入为 52.6 亿美元，Yahoo! 收入的很大一部分来自通过 GSP 拍卖的销售。据信 Yahoo! 超过一半的收入来自通过 GSP 拍卖的销售。截至 2006 年 5 月，这些公司的合并市值超过 1500 亿美元。

让我们简要描述这些拍卖的运作方式。当互联网用户将搜索词（"查询"）输入搜索引擎时，他会得到一个包含结果的页面，其中既包含与查询最相关的链接，也包含赞助链接，即付费广告。广告与实际搜索结果明显可区分，不同的搜索会产生不同的赞助链接：广告商根据搜索关键词定向投放广告。例如，如果一家旅行社购买了"Hawaii"这个词，那么每次用户搜索该词时，旅行社的链接都会出现在搜索结果页面上。当用户点击赞助链接时，他会被发送到广告商的网页。广告商然后向搜索引擎付费，因为搜索引擎将用户发送到其网页，因此得名——"按点击付费"定价。

搜索引擎可以向用户展示的广告数量是有限的，搜索结果页面上的不同位置对广告商具有不同的吸引力：显示在页面顶部的广告比显示在底部的广告更有可能被点击。因此，搜索引擎需要一个系统来将位置分配给广告商，而拍卖是自然的选择。目前，搜索引擎最广泛使用的机制基于 GSP。

在最简单的 GSP 拍卖中，对于特定关键词，广告商提交出价，说明他们愿意为一次点击支付的最高金额。当用户输入关键词时，他会收到搜索结果以及赞助链接，后者按出价降序显示。特别是，出价最高的广告显示在顶部，出价次高的广告显示在第二个位置，依此类推。如果用户随后点击了位置 $i$ 的广告，该广告商被搜索引擎收取的金额等于次高出价，即位置 $(i+1)$ 的广告商的出价。如果搜索引擎在每个结果页面上只提供一个广告，该机制将等同于标准的第二价格拍卖，与 Vickrey-Clarke-Groves（VCG）机制 [29; 6; 10] 一致。在有多个位置可用时，GSP 推广了第二价格拍卖（因此得名）。这里，每个广告商支付下一个最高广告商的出价。但正如我们将证明的，多单位 GSP 拍卖不再等同于 VCG 拍卖，并且缺乏 VCG 的一些理想性质。特别是，与 VCG 机制不同，GSP 一般不存在优势策略均衡，真实报价不是 GSP 的均衡。

在第 I 节中，我们描述了互联网广告市场的演化以及该市场中环境的独特特征。在第 II 节中，我们引入了赞助搜索拍卖的模型，并在第 III 节中开始对模型进行分析。由于广告商可以频繁更改出价，赞助搜索拍卖可以建模为连续或无限重复博弈。然而，根据民间定理，此类博弈将具有极大的均衡集，因此我们转而关注一次性同时移动完全信息阶段博弈，并引入广告商行为的限制，这些限制由市场的动态结构所建议。我们将满足这些限制的均衡称为"局部无嫉妒"均衡。

然后我们继续表明，局部无嫉妒均衡集包含一个均衡，其中参与者的收益与 VCG 拍卖的优势策略均衡相同，尽管两个机制中的参与者出价和支付规则都非常不同。此外，该均衡是搜索引擎最差的局部无嫉妒均衡，也是广告商最好的局部无嫉妒均衡。因此，在 GSP 的任何局部无嫉妒均衡中，卖家的总期望收入至少与 VCG 拍卖的优势策略均衡中一样高。

在第 IV 节中，我们展示主要结果。我们引入了具有独立私有价值的广义英式拍卖，它对应于广义第二价格拍卖，旨在捕获竞价行为向静态均衡的收敛，与一般均衡理论中的试探过程 [15; 13]和劳动力市场匹配理论中的延迟接受工资调整过程 [8]具有相同的精神。广义英式拍卖有几个显著特征。虽然它不是优势策略可解的，但它在连续策略中具有唯一的完美贝叶斯均衡。在该均衡中，所有参与者获得 VCG 收益。此外，该均衡是事后均衡 [12; 24]，即即使特定参与者在博弈前了解了其他参与者的价值，他也不会想要改变策略。这反过来意味着该均衡是稳健的 [3; 17]，即它不依赖于价值的底层分布：我们识别的策略配置对于任何竞标者私有价值分布集都是事后贝叶斯纳什均衡。

有几篇最近的理论和实证论文与赞助搜索拍卖相关。Aggarwal 和 Hartline [1]、Mehta 等人 [22] 以及 Mahdian、Nazerzadeh 和 Saberi [18] 提出了计算速度快、在预算约束和随机冲击存在时的近最优定价和分配机制。Meek、Chickering 和 Wilson [21] 描述了具有随机分配规则的激励兼容拍卖，推广了 Vickrey 拍卖，并认为此类拍卖尽管低效，但在销售互联网广告方面可能有用。Zhang [31]、Asdemir [2] 以及 Edelman 和 Ostrovsky [9] 提供了广义第一价格和广义第二价格拍卖中出价和排名波动的实证证据。他们认为依赖历史的策略可能导致此类波动。然而，Varian [28] 实证分析了来自 Google 的 GSP 拍卖数据，并报告局部无嫉妒纳什均衡"相当准确地描述了 Google 广告拍卖中观察到的价格的基本性质"。

### 1.1 互联网广告市场的显著特征

互联网广告市场有几个特征使其独一无二。首先，出价可以随时更改。广告商对特定关键词的出价将适用于每次搜索引擎用户输入该关键词时，直到广告商更改或撤回出价。例如，某一时刻某个关键词出价第二高的广告商将在该时刻搜索该关键词的用户显示为第二个赞助链接。下次用户搜索该关键词时，广告的顺序可能不同，因为出价可能已经更改。

其次，搜索引擎有效地销售易腐广告服务流，而不是可存储的对象：如果在某段时间内没有特定搜索词的广告，"容量"就被浪费了。

最后，与其他通常清楚如何衡量所售商品的中心化市场不同，从所有相关方的角度来看，没有自然的互联网广告"单位"。从广告商的角度来看，相关单位是吸引进行购买的客户的成本。这最直接地对应于广告商仅在客户实际完成交易时付费的定价模式。从搜索引擎的角度来看，相关单位是每次用户搜索特定关键词时搜索引擎收取的收入。这对应于每次向潜在客户展示广告商链接时向广告商收费的定价模式。"按点击付费"是两种模式之间的折中：广告商每次用户点击链接时付费。所有三种付费模式都在互联网上被广泛使用。我们研究的互联网广告特定领域——赞助搜索拍卖——已经收敛到按点击付费定价。

我们忽略的一个重要可能性是广告商在每次点击价值以外的维度上不同，即在放置在相同位置时被点击的概率不同 [5]（这些概率在行业中被称为"点击率"或 CTR）。不同的搜索引擎对待这种可能性的方式不同。Yahoo! 忽略差异，纯粹按出价降序排列广告商，并收取下一个最高广告商的出价。Google 将每个广告商的出价乘以其"质量分数"（基于 CTR 和其他因素），计算其"排名数字"，按排名数字排列广告，然后向每个广告商收取足以超过下一个广告商排名数字的最小金额。在我们的分析中，我们假设所有广告商在每次点击价值以外的维度上是相同的，这消除了 Google 和 Yahoo! 机制之间的差异。

### 1.2 市场制度的演化

赞助搜索拍卖的历史是一个有趣的案例研究，展示了市场是否、如何以及多快地解决其结构性缺陷。许多重要机制最近基本上是从零开始设计的，完全取代了完全不同的历史分配机制：无线电频谱拍卖 [23; 4]、电力拍卖 [30] 等。相比之下，让人想起医疗住院医师匹配规则 [7] 的逐步演化 [25]，赞助搜索广告拍卖随着时间的推移逐步演化 [26]。在医疗住院医师匹配和搜索广告中，有缺陷的机制逐渐被越来越优秀的设计所取代。值得注意的是，互联网广告市场的演化速度远快于医疗匹配市场。这可能是由于前者中存在而后者中不存在的机制设计者面临的竞争压力、低得多的进入和实验成本、对市场机制理解的进步以及改进的技术。

**早期互联网广告。** 从 1994 年开始，互联网广告主要按展示方式销售。广告商支付固定费用以展示其广告固定次数（通常为 1000 次展示或"印象"）。合同逐案谈判，广告购买的最低合同很大（通常每月几千美元），进入缓慢。

**广义第一价格拍卖。** 1997 年，Overture（当时名为 GoTo；现在是 Yahoo! 的一部分）引入了一种全新的销售在线广告的模式：广告商对关键词进行出价，出价最高的广告商出现在搜索结果的顶部。最初，这是一个纯粹的第一价格拍卖：广告商每次用户点击其广告时支付其出价。因此得名"广义第一价格"拍卖。

**广义第二价格拍卖。** 广义第一价格拍卖存在严重的低效性。2002 年，Overture 引入了一种新机制，即本文研究的广义第二价格拍卖。Google 和 Yahoo! 都使用 GSP 的变体。

## 2 GSP 的规则

让我们现在正式描述赞助搜索拍卖的规则。对于给定的关键词，有 $N$ 个对象（屏幕上的位置，可以显示与该关键词相关的广告）和 $K$ 个竞标者（广告商）。广告商 $g(i)$ 的广告放置在位置 $i$ 时获得的每次期间预期点击数为 $\alpha_i$。广告商 $k$ 的每次点击价值为 $s_k$。广告商是风险中性的 [19]，广告商 $k$ 在位置 $i$ 的收益等于 $\alpha_i s_k$ 减去其向搜索引擎的支付。注意这些假设意味着特定位置被点击的次数不依赖于该位置和其他位置的广告，并且广告商的每次点击价值不依赖于其广告显示的位置。不失一般性，位置按降序标记：对于任何 $i$ 和 $j$，如果 $i < j$，则 $\alpha_i > \alpha_j$。

我们将 GSP 拍卖建模如下。假设在某个时间 $t$，搜索引擎用户输入给定关键词，对于每个 $k$，广告商 $k$ 在 $t$ 之前提交的该关键词的最后一个出价为 $b_k$；如果广告商 $k$ 没有提交出价，我们设 $b_k = 0$。令 $b_{(j)}$ 和 $g(j)$ 分别表示第 $j$ 个最高广告商的出价和身份。如果几个广告商提交相同的出价，他们被随机排序。该机制然后将顶部位置分配给出价最高的广告商 $g(1)$，第二个位置分配给 $g(2)$，依此类推。如果广告商 $g(i)$ 的广告被放置在位置 $i$，则该广告商被搜索引擎收取的金额为 $\alpha_i b_{(i+1)}$。因此，广告商 $g(i)$ 的收益为：

$$
u_{g(i)} = \alpha_i (s_{g(i)} - b_{(i+1)})
$$

我们将此机制称为 GSP。注意，如果只有一个位置（$N = 1$），GSP 简化为标准的第二价格拍卖。

为了进行比较，让我们描述 VCG 拍卖。在 VCG 中，位置的分配规则与 GSP 相同：位置 $i$ 分配给出价第 $i$ 高的广告商 $g(i)$。然而，支付规则不同。每个广告商的支付等于他施加给其他人的负外部性，假设出价等于价值。因此，最后一个被分配位置的广告商的支付与 GSP 下相同：如果 $N \ge K$ 则为零；否则为 $\alpha_N b_{(N+1)}$。对于所有其他 $i < \min\{N, K\}$，VCG 诱导的支付 $p^V$ 与 GSP 的支付 $p$ 不同。具体来说，$p^V_{(i)} = (\alpha_i - \alpha_{i+1}) b_{(i+1)} + p^V_{(i+1)}$。

在接下来的两节中，我们将考虑两种完成模型的替代方式：作为完全信息的同时移动博弈（类似于密封出价第二价格拍卖），以及作为不完全信息的扩展形式博弈（类似于递增英式拍卖）。在继续这些模型之前，让我们对 GSP 和 VCG 做一些观察。

**注记 1：** 如果所有广告商在两种机制下出价相同，那么每个广告商在 GSP 下的支付至少与在 VCG 下一样大。

这很容易通过对广告商支付的归纳来证明，从最后一个被分配位置的广告商开始。对于 $i = \min\{K, N\}$，$p_{(i)} = p^V_{(i)} = \alpha_i b_{(i+1)}$。对于任何 $i < \min\{K, N\}$，$p^V_{(i)} - p^V_{(i+1)} = (\alpha_i - \alpha_{i+1}) b_{(i+1)} \le \alpha_i b_{(i+1)} - \alpha_{i+1} b_{(i+2)} = p_{(i)} - p_{(i+1)}$。

**注记 2：** 真实报价是 VCG 下的优势策略。

这是 VCG 机制的一个众所周知的性质。

**注记 3：** 真实报价不是 GSP 下的优势策略。

例如，考虑对第 I 节中示例的轻微修改。仍然有三个广告商，每次点击价值为 \$10、\$4 和 \$2，两个位置。然而，这些位置的点击率现在几乎相同：第一个位置每小时获得 200 次点击，第二个位置获得 199 次。如果所有参与者真实报价，那么广告商 1 的收益为 $(10 - 4) \times 200 = 1{,}200$。相反，如果他压低出价，仅出价每次点击 \$3，他将获得第二个位置，他的收益将为 $(10 - 2) \times 199 = 1{,}592 > 1{,}200$。

## 3 GSP 与局部无嫉妒均衡

在 Yahoo! 和 Google 上出价的广告商可以非常频繁地更改出价。因此，我们将这些赞助搜索拍卖视为连续时间或无限重复博弈，其中广告商最初拥有其类型的私有信息，逐渐了解其他人的价值，并可以反复调整出价。原则上，此类重复博弈中的均衡集可能非常大，玩家可能因偏离而互相惩罚。然而，支持此类均衡所需的策略通常相当复杂，需要对环境的精确了解和仔细的实施。在理论上，广告商可以通过自动机器人实施此类策略，但在实践中他们可能无法做到：出价软件必须首先获得搜索引擎的授权，搜索引擎不太可能允许会允许广告商串通并大幅减少收入的策略。

因此，我们关注简单策略，并研究竞价过程的不动点：如果出价向量稳定下来，它能在什么出价上稳定？我们施加几个假设和限制。首先，我们假设所有价值是共同知识：随着时间的推移，广告商可能了解彼此价值的所有相关信息。其次，由于出价可以随时更改，稳定的出价必须是彼此的最佳响应——否则，出价不是最佳响应的广告商将有动机更改它。因此，我们假设出价形成完全信息同时移动博弈中的均衡。第三，除了对其他玩家出价的简单最佳响应之外，广告商可以使用的增加其收益的简单策略是什么？

一个明显的策略是尝试挤出占据紧上方位置的玩家。假设广告商 $k$ 出价 $b_k$ 并被分配到位置 $i$，广告商 $k'$ 出价 $b_{k'} > b_k$ 并被分配到位置 $(i-1)$。注意如果 $k$ 略微提高出价，他自己的收益不会改变，但他上方玩家的收益减少。当然，玩家 $k'$ 可以报复，她能做的最多是略微低于广告商 $k$ 的出价，有效地与他交换位置。如果广告商 $k$ 在报复后变得更好，他确实会想要挤出玩家 $k'$，出价向量将改变。因此，如果出价向量收敛到不动点，位置 $i$ 的广告商不应想要与位置 $(i-1)$ 的广告商"交换"位置。我们将此类出价向量称为"局部无嫉妒"。

**定义 1：** GSP 诱导的同时移动博弈的均衡是局部无嫉妒的，如果玩家不能通过与分配给另一个玩家的位置重新匹配来改善其收益，条件是其他玩家保持其出价不变。

等价地，均衡是局部无嫉妒的，如果对于所有 $i$，$\alpha_i s_{g(i)} - p(i) \ge \alpha_{i+1} s_{g(i)} - p(i+1)$。

我们现在证明两个引理，它们将帮助我们描述 GSP 的局部无嫉妒均衡集。

**引理 1：** 在任何局部无嫉妒均衡中，结果是有效匹配：对于任何 $i$，分配到位置 $i$ 的广告商的每次点击价值至少与分配到位置 $i+1$ 的广告商一样高。

**引理 2：** 对于任何有效的匹配，存在一个具有相应结果的局部无嫉妒均衡。此外，对于给定的有效匹配，存在一个产生该匹配的局部无嫉妒均衡，其中所有广告商的支付与 VCG 支付相同。

**定理 1：** 在 GSP 的任何局部无嫉妒均衡中，卖家的总期望收入至少与 VCG 拍卖的优势策略均衡中一样高。

在我们的环境中，假设所有广告商相同（除了每次点击价值），对于 Yahoo! 和 Google 的机制，分析基本相同。在 Yahoo! 的系统下，广告商仍按出价降序排列。在 Google 的系统下，广告商按排名数字排列，但均衡分析仍然类似。

## 4 广义英式拍卖

我们通过引入广义英式拍卖来完成模型，这是 GSP 的扩展形式对应物，旨在捕获竞价行为向静态均衡的收敛，与一般均衡理论中的试探过程和劳动力市场匹配理论中的延迟接受工资调整过程具有相同的精神。

我们将广义英式拍卖视为如下。拍卖师从零开始逐渐提高时钟价格 $p$。当价格达到广告商不愿支付的水平时，广告商退出。最后两个广告商在其中一人退出时停止。在均衡中，广告商的退出策略如下：当价格达到 $p_k(i, h, s_0) = s_0 - (\alpha_i / \alpha_{i-1})(s_0 - b_{i+1})$ 时，价值为 $s_0$ 的广告商退出，其中 $b_{i+1}$ 是下一个退出的广告商的出价。

广义英式拍卖有几个显著特征。虽然它不是优势策略可解的，但它在连续策略中具有唯一的完美贝叶斯均衡。在该均衡中，所有参与者获得 VCG 收益。

**定理 2：** 广义英式拍卖具有唯一的完美贝叶斯均衡（在连续策略中）。在该均衡中，每个广告商的收益等于其 VCG 收益。该均衡是事后均衡。

定理 2 的结果类似于英式拍卖和第二价格密封出价拍卖在私有价值下等价的经典结果 [29]。然而，直觉非常不同：Vickrey 的结果简单地来自优势策略均衡的存在，而在我们的情况下，此类策略不存在，出价依赖于其他玩家的出价。此外，我们的结果与收入等价定理非常不同：广义英式拍卖中的收益与 VCG 支付在所有价值实现中一致，不仅在期望中，并且该结果不依赖于对称竞标者或共同先验的假设。

定理 2 中描述的均衡是事后均衡。只要广告商 $k$ 以外的所有广告商遵循定理 2 中描述的均衡策略，对于其他广告商价值的任何实现，广告商 $k$ 遵循其均衡策略都是最佳响应。因此，该机制实现的结果仅依赖于广告商价值的实现，不依赖于广告商对彼此类型的信念。

显然，任何优势策略可解的博弈都具有事后均衡。然而，广义英式拍卖不是优势策略可解的。这些性质的组合相当引人注目：均衡是唯一且有效的，每个广告商的策略不依赖于其他广告商价值的分布，然而广告商没有优势策略。具有如此显著理论性质和如此巨大实际流行度的机制，是低效市场制度演化的结果，这些制度逐渐被越来越优秀的设计所取代，这一点特别有趣。

## 5 结论

我们研究了一种新的拍卖机制，我们称之为广义第二价格拍卖。GSP 是为互联网广告市场的独特特征量身定制的。据我们所知，该机制于 2002 年首次使用。截至 2006 年 5 月，GSP 拍卖的年收入约为 100 亿美元。

GSP 看起来类似于 VCG 机制，因为就像在标准的第二价格拍卖中一样，竞标者的支付不直接依赖于其出价。虽然 GSP 看起来类似于 VCG，但其性质截然不同，均衡行为远非直接。特别是，与 VCG 机制不同，GSP 一般不存在优势策略均衡，真实报价不是 GSP 的均衡。我们证明了对应于广义第二价格拍卖的广义英式拍卖存在唯一均衡。

该均衡有一些显著性质。出价函数具有显式解析公式，结合均衡唯一性，使我们的结果成为实证分析的有用起点。此外，这些函数不依赖于竞标者对彼此类型的信念：拍卖的结果仅依赖于竞标者价值的实现 [14]。这是实践中遇到的极少数不是优势策略可解但仍然具有此性质的机制之一 [16]。一个具有如此显著理论性质和如此巨大实际流行度的机制，是低效市场制度演化的结果，这些制度逐渐被越来越优秀的设计所取代，这一点特别有趣。

## 致谢

我们感谢 Drew Fudenberg、Louis Kaplow、Robin Lee、David McAdams、Paul Milgrom、Muriel Niederle、Ariel Pakes、David Pennock 和 Al Roth 的有益讨论。

## 附录：证明

**引理 1 的证明：** 根据定义，在任何局部无嫉妒均衡结果中，没有广告商可以有利地与分配给其正上方广告商的位置重新匹配。也没有广告商 (a) 可以有利地与分配给其下方广告商 (b) 的位置重新匹配——如果存在此类有利的重新匹配，广告商 (a) 会发现在博弈 $\Gamma$ 中略微低于广告商 (b) 的出价并获得 (b) 的位置和支付是有利的。但这将与我们处于均衡中的假设矛盾。

因此，我们只需要证明没有广告商可以有利地与分配给其上方超过一个位置的广告商的位置重新匹配。首先，注意在任何局部无嫉妒均衡中，结果匹配必须是分类匹配，即对于任何 $i$，分配到位置 $i$ 的广告商的每次点击价值高于分配到位置 $i+1$ 的广告商，因此每次点击价值最高的广告商必须被分配到顶部位置，次高价值的广告商到第二高的位置，依此类推。

现在，让我们证明没有广告商可以有利地与分配给其上方超过一个位置的广告商的位置重新匹配。假设分配到位置 $i$ 的广告商正在考虑与位置 $m < i-1$ 重新匹配。由于均衡是局部无嫉妒的，我们有：

$$
\alpha_i s_{g(i)} - p(i) \ge \alpha_{i-1} s_{g(i)} - p(i-1)
$$

$$
\alpha_{i-1} s_{g(i-1)} - p(i-1) \ge \alpha_{i-2} s_{g(i-1)} - p(i-2)
$$

$$
\vdots
$$

$$
\alpha_{m+1} s_{g(m+1)} - p(m+1) \ge \alpha_m s_{g(m+1)} - p(m)
$$

由于对于任何 $j$，$\alpha_j > \alpha_{j+1}$，且对于任何 $i < j$，$s_{g(i)} > s_{g(j)}$，上述不等式在将 $s_{g(i)}$ 替换为 $s_{g(j)}$ 后仍然有效。这样做，然后将所有不等式相加并消去冗余元素，我们得到 $\alpha_i s_{g(i)} - p(i) \ge \alpha_m s_{g(i)} - p(m)$。但这意味着分配到位置 $i$ 的广告商不能有利地与位置 $m$ 重新匹配，证明完成。

**引理 2 的证明：** 取一个稳定分配。根据 Shapley 和 Shubik [27] 的结果，该分配必须是有效的 [11]，该分配必须是有效的，因此是分类匹配，不失一般性我们可以假设广告商按其出价降序标记（即当 $j < k$ 时 $s_j > s_k$），广告商 $i$ 与位置 $i$ 匹配，关联支付为 $p_i$。

让我们构造一个具有相应结果的局部无嫉妒均衡。设 $b_1 = s_1$，对于 $i > 1$ 设 $b_i = p_{i-1}/\alpha_{i-1}$。让我们证明这组策略是局部无嫉妒均衡。首先，注意对于任何 $i$，$b_i > b_{i+1}$。因此位置分配和支付与原始分配相同。

## 参考文献

[1] Aggarwal, Gagan, and Jason D. Hartline. 2005. "Knapsack Auctions." Paper presented at the First Workshop on Sponsored Search Auctions, Vancouver, BC.

[2] Asdemir, Kursad. 2006. "Bidding Patterns in Search Engine Auctions." Paper presented at the Second Workshop on Sponsored Search Auctions, Ann Arbor, MI.

[3] Bergemann, Dirk, and Stephen Morris. 2005. "Robust Mechanism Design." Econometrica, 73(6): 1771–1813.

[4] Binmore, Ken, and Paul Klemperer. 2002. "The Biggest Auction Ever: The Sale of the British 3G Telecom Licenses." Economic Journal, 112(478): C74–96.

[5] Brooks, Nico. 2004. The Atlas Rank Report—Part II: How Search Engine Rank Impacts Conversion. Seattle: Atlas Institute.

[6] Clarke, Edward H. 1971. "Multipart Pricing of Public Goods." Public Choice, 11(0): 17–33.

[7] Crawford, Vincent P., and Elsie M. Knoer. 1981. "Job Matching with Heterogeneous Firms and Workers." Econometrica, 49(2): 437–50.

[8] Demange, Gabrielle, David Gale, and Marilda Sotomayor. 1986. "Multi-Item Auctions." Journal of Political Economy, 94(4): 863–72.

[9] Edelman, Benjamin, and Michael Ostrovsky. Forthcoming. "Strategic Bidder Behavior in Sponsored Search Auctions." Decision Support Systems.

[10] Groves, Theodore. 1973. "Incentives in Teams." Econometrica, 41(4): 617–31.

[11] Hatfield, John William, and Paul R. Milgrom. 2005. "Matching with Contracts." American Economic Review, 95(4): 913–35.

[12] Jehiel, Philippe, and Benny Moldovanu. 2001. "Efficient Design with Interdependent Valuations." Econometrica, 69(5): 1237–59.

[13] Jehiel, Philippe, Moritz Meyer-ter-Vehn, Benny Moldovanu, and William R. Zame. 2006. "The Limits of Ex Post Implementation." Econometrica, 74(3): 585–610.

[14] Kagel, John H., Ronald M. Harstad, and Dan Levin. 1987. "Information Impact and Allocation Rules in Auctions with Affiliated Private Values: A Laboratory Study." Econometrica, 55(6): 1275–1304.

[15] Kelso, Alexander S., Jr., and Vincent P. Crawford. 1982. "Job Matching, Coalition Formation, and Gross Substitutes." Econometrica, 50(6): 1483–1504.

[16] Kittsteiner, Thomas, and Benny Moldovanu. 2005. Management Science, 51(2): 236–48.

[17] Leonard, Herman B. 1983. "Elicitation of Honest Preferences for the Assignment of Individuals to Positions." Journal of Political Economy, 91(3): 461–79.

[18] Mahdian, Mohammad, Hamid Nazerzadeh, and Amin Saberi. 2006. "AdWords Allocation Problem with Unreliable Estimates." Unpublished.

[19] Mas-Colell, Andreu, Michael D. Whinston, and Jerry R. Green. 1995. Microeconomic Theory. New York: Oxford University Press.

[20] McAdams, David, and Michael Schwarz. Forthcoming. "Who Pays When Auction Rules are Bent?" International Journal of Industrial Organization.

[21] Meek, Christopher, David M. Chickering, and David B. Wilson. 2005. "Stochastic and Contingent-Payment Auctions." Paper presented at the First Workshop on Sponsored Search Auctions, Vancouver, BC.

[22] Mehta, Aranyak, Amin Saberi, Umesh Vazirani, and Vijay Vazirani. 2005. "AdWords and Generalized On-line Matching." Paper presented at the First Workshop on Sponsored Search Auctions, Vancouver, BC.

[23] Milgrom, Paul. 2000. "Putting Auction Theory to Work: The Simultaneous Ascending Auction." Journal of Political Economy, 108(2): 245–72.

[24] Moldovanu, Benny, and Aner Sela. 2001. "The Optimal Allocation of Prizes in Contests." American Economic Review, 91(3): 542–58.

[25] Roth, Alvin E. 1984. "The Evolution of the Labor Market for Medical Interns and Residents: A Case Study in Game Theory." Journal of Political Economy, 92(6): 991–1016.

[26] Roth, Alvin E., and Axel Ockenfels. 2002. "Last-Minute Bidding and the Rules for Ending Second-Price Auctions: Evidence from eBay and Amazon Auctions on the Internet." American Economic Review, 92(4): 1093–1103.

[27] Shapley, Lloyd S., and Martin Shubik. 1971. "The Assignment Game I: The Core." International Journal of Game Theory, 1(1): 111–30.

[28] Varian, Hal R. Forthcoming. "Position Auctions." International Journal of Industrial Organization.

[29] Vickrey, William. 1961. "Counterspeculation, Auctions, and Competitive Sealed Tenders." Journal of Finance, 16(1): 8–37.

[30] Wilson, Robert. 2002. "Architecture of Power Markets." Econometrica, 70(4): 1299–1340.

[31] Zhang, Xiaoquan. 2005. "Finding Edgeworth Cycles in Online Advertising Auctions." Unpublished.
