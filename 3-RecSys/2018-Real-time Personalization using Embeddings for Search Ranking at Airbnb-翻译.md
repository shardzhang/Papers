# Airbnb 搜索排序中基于嵌入的实时个性化

> Mihajlo Grbovic, Haibin Cheng | Airbnb, Inc., San Francisco, California, USA

本文介绍了在 Airbnb 短租市场中，利用 listing 嵌入和用户类型嵌入实现实时个性化搜索排序和相似 listing 推荐的方法。核心内容：

- 提出基于点击会话训练的 listing 嵌入（Listing Embeddings），通过 Skip-gram 模型学习 listing 的低维向量表示，用于短期实时个性化和相似 listing 推荐
- 提出用户类型（User Type）和 listing 类型（Listing Type）嵌入，通过预订会话训练，捕获用户长期兴趣偏好，实现跨市场的个性化排序
- 将预订 listing 作为全局上下文、将同市场负采样和房东拒绝作为显式负例引入嵌入训练过程，显著提升嵌入质量

关键发现：

- 基于嵌入的相似 listing 推荐方案在 A/B 测试中使相似 listing 轮播的点击率（CTR，Click-Through Rate）提升 21%，在搜索排序模型中加入嵌入特征后 NDCU（Normalized Discounted Cumulative Utility）提升 2.27%，预订 DCU 提升 2.58%
- 将预订 listing 作为全局上下文并加入同市场负采样（d32 book + neg）训练的嵌入在离线评估中优于仅使用标准 Skip-gram（d32）和仅使用全局上下文（d32 book）的嵌入
- 五项嵌入特征进入 GBDT（Gradient Boosting Decision Tree，梯度提升决策树）模型 104 个特征中的前 20 名，其中 EmbClickSim 排名第 5、EmbSkipSim 排名第 8

---

## 摘要

搜索排序和推荐是主要互联网公司关注的基本问题，包括网络搜索引擎、内容发布网站和市场平台。然而，尽管存在一些共同特征，该领域并不存在通用的解决方案。鉴于需要排序、个性化和推荐的内容差异巨大，每个市场都面临独特的挑战。相应地，在 Airbnb 这一短租市场中，搜索和推荐问题相当独特——它是一个双边市场，需要同时优化房东和租客的偏好，且用户很少重复消费同一 item，一个房源在特定日期只能接受一位租客。在本文中，我们介绍了为搜索排序和相似 listing 推荐中的实时个性化目的而开发和部署的 listing 嵌入和用户嵌入技术，这两个渠道驱动了 99% 的预订转化。嵌入模型专门针对 Airbnb 市场定制，能够捕获租客的短期和长期兴趣，提供有效的房源推荐。我们对嵌入模型进行了严格的离线测试，随后成功进行在线测试，最终将其完全部署到生产环境中。

**关键词：** Search Ranking, User Modeling, Personalization, Embedding

---

## 1 引言

在过去十年中，通常基于经典信息检索的搜索架构在其各个组件中越来越多地引入了机器学习[2]，尤其是在搜索排序方面，其目标往往取决于所搜索内容的类型。这一趋势的主要原因是可收集和分析的搜索数据量的增长。大量收集的数据为利用机器学习实现个性化搜索结果提供了可能——基于用户先前的搜索为特定用户个性化搜索结果，并推荐与最近消费内容相似的内容。

任何搜索算法的目标因平台而异。有些平台旨在提高网站参与度（例如点击量和在被搜索的新闻文章上花费的时间），有些平台旨在最大化转化率（例如购买被搜索的商品或服务），而在双边市场的情况下，我们通常需要为市场的双方优化搜索结果，即卖家和买家。双边市场已成为许多现实应用中可行的商业模式。特别是，我们已经从社交网络范式转变为具有两种不同类型参与者（代表供给和需求）的网络。示例行业包括住宿（Airbnb）、网约车（Uber、Lyft）、在线商店（Etsy）等。可以说，这些类型市场的内容发现和搜索排序需要同时满足生态系统的供给和需求双方，才能增长和繁荣。

在 Airbnb 的案例中，显然需要同时为房东和租客优化搜索结果，这意味着给定一个包含位置和旅行日期的输入查询，我们需要将位置、价格、风格、评论等对租客有吸引力的房源排在前面，同时在旅行时长和提前天数方面也要符合房东的偏好。此外，我们需要检测那些可能因为差评、宠物、入住时长、团体大小或其他因素而拒绝租客的房源，并将这些房源排在较低位置。为此，我们采用排序学习（Learning to Rank）。具体来说，我们将问题形式化为成对回归，其中预订具有正效用，拒绝具有负效用，我们使用 Lambda Rank[4]模型的修改版本进行优化，该模型联合优化市场双方的排序。

由于租客通常在预订前进行多次搜索，即在搜索会话中点击多个房源并联系多个房东，我们可以利用这些会话内信号（如点击、房东联系等）进行实时个性化，目的是向租客展示更多与他们自搜索会话开始以来可能喜欢的房源相似的 listing。同时，我们可以利用负信号（例如跳过排名靠前的房源）向租客展示更少与他们可能不喜欢的房源相似的 listing。为了能够计算租客交互过的房源与需要排序的候选房源之间的相似度，我们提出使用 listing 嵌入——从搜索会话中学习的低维向量表示。我们利用这些相似度为搜索排序模型创建个性化特征，并驱动相似 listing 推荐——这两个平台在 Airbnb 驱动了 99% 的预订。

除了利用即时用户行为（如点击）作为短期用户兴趣代理信号的实时个性化外，我们还介绍了另一种在预订数据上训练的嵌入类型，以捕获用户的长期兴趣。由于旅行业务的性质——用户平均每年旅行 1-2 次——预订信号是稀疏的，存在大量只有一次预订的长尾用户。为解决这一问题，我们提出在用户类型（User Type）级别而非特定用户 ID 级别训练嵌入，其中类型通过利用已知用户属性的多对一基于规则的映射确定。同时，我们在与用户类型嵌入相同的向量空间中学习 listing 类型嵌入。这使我们能够计算正在搜索的用户类型嵌入与需要排序的候选 listing 类型嵌入之间的相似度。

与先前发表的网络嵌入个性化工作相比，本文的新贡献包括：

* **实时个性化** — 此前大多数使用嵌入进行个性化和 item 推荐的工作[8, 11]通过离线构建用户-item 和 item-item 推荐表来部署到生产中，然后在推荐时从表中读取。我们实现了一种解决方案，其中用户最近交互的 item 的嵌入以在线方式组合，以计算与需要排序的 item 的相似度。
* **适应聚集搜索的训练** — 与网络搜索不同，旅行平台上的搜索通常是聚集的，用户经常只在特定市场内搜索（例如巴黎），很少跨不同市场搜索。我们调整了嵌入训练算法，在进行负采样时考虑了这一点，从而更好地捕获市场内的 listing 相似度。
* **利用转化作为全局上下文** — 我们认识到以转化（在我们的案例中是预订）结束的点击会话的重要性。在学习 listing 嵌入时，我们将预订的 listing 作为全局上下文，无论其是否在上下文窗口内，都会被预测。
* **用户类型嵌入** — 此前关于训练用户嵌入以捕获长期兴趣的工作[6, 27]为每个用户训练单独的嵌入。当目标信号稀疏时，没有足够的数据为每个用户训练良好的嵌入表示。更不用说为每个用户存储嵌入以进行在线计算需要大量内存。因此，我们提出在用户类型级别训练嵌入，具有相同类型的用户组将拥有相同的嵌入。
* **拒绝作为显式负例** — 为了减少导致拒绝的推荐，我们在训练中将房东拒绝信号编码到用户和 listing 类型嵌入中，将其作为显式负例。

## 2 相关工作

在许多自然语言处理（NLP，Natural Language Processing）应用中，将词表示为高维稀疏向量的经典语言建模方法已被神经语言模型所取代，这些模型通过使用神经网络学习词嵌入，即词的低维表示[25, 27]。这些网络通过直接考虑词序和共现来训练，基于一个假设：在句子中频繁一起出现的词也共享更多的统计依赖性。随着高度可扩展的连续词袋（CBOW，Continuous Bag-of-Words）和跳字（SG，Skip-Gram）词表示学习模型的发展[17]，嵌入模型在大规模文本数据训练后，在许多传统语言任务上取得了最先进的性能。

最近，嵌入的概念已经从词表示扩展到 NLP 领域之外的其他应用。来自网络搜索、电子商务和市场领域的研究人员很快意识到，就像可以将句子中的词序列作为上下文来训练词嵌入一样，也可以通过将用户行为序列作为上下文来训练用户行为的嵌入，例如被点击或购买的 item[11, 18]、被点击的查询和广告[8, 9]。此后，我们看到嵌入被应用于网络上的各种推荐，包括音乐推荐[26]、求职搜索[13]、应用推荐[21]、电影推荐[3, 7]等。此外，研究表明用户交互过的 item 可以用来直接在同一特征空间中学习用户嵌入，从而可以直接进行用户-item 推荐[6, 10, 11, 24, 27]。另一种方法特别适用于冷启动推荐，即仍然使用文本嵌入（例如在 https://code.google.com/p/word2vec 上公开可用的嵌入），并利用 item 和/或用户元数据（如标题和描述）来计算其嵌入[5, 14, 19, 28]。最后，类似的嵌入方法扩展已被提出用于社交网络分析，其中图上的随机游走可用于学习图结构中节点的嵌入[12, 20]。

嵌入方法在学术界和工业界都产生了重大影响。最近的行业会议出版物和演讲表明，它们已成功部署在主要网络公司的各种个性化、推荐和排序引擎中，如 Yahoo[8, 11, 29]、Etsy[1]、Criteo[18]、LinkedIn[15, 23]、Tinder[16]、Tumblr[10]、Instacart[22]、Facebook[28]。

## 3 方法

在下文中，我们介绍为 Airbnb 搜索中的 listing 推荐和 listing 排序任务提出的两种不同方法，即用于短期实时个性化的 listing 嵌入，以及用于长期个性化的用户类型和 listing 类型嵌入。

### 3.1 Listing 嵌入

假设我们有一组 $S$ 个来自 $N$ 个用户的点击会话，其中每个会话 $s = (l_1, \ldots, l_M) \in S$ 定义为用户点击的 $M$ 个 listing ID 的不间断序列。当两次连续用户点击之间存在超过 30 分钟的时间间隔时，开始一个新的会话。给定此数据集，目标是学习每个唯一 listing $l_i$ 的 $d$ 维实值表示 $\mathbf{v}_{l_i} \in \mathbb{R}^d$ ，使得相似的 listing 在嵌入空间中彼此接近。

更形式化地，模型的目标是使用 Skip-gram 模型[17]通过最大化整个搜索会话集 $S$ 上的目标函数 $L$ 来学习 listing 表示，定义如下：

$$
L = \sum_{s \in S} \sum_{l_i \in s} \sum_{-m \leq j \leq m, j \neq 0} \log P(l_{i+j} | l_i) \qquad (1)
$$

在点击 listing $l_i$ 的上下文邻域中观察到 listing $l_{i+j}$ 的概率 $P(l_{i+j} | l_i)$ 使用 softmax 定义：

$$
P(l_{i+j} | l_i) = \frac{\exp(\mathbf{v}_{l_i}^{\top} \mathbf{v}_{l_{i+j}}^{\prime})}{\sum_{l=1}^{|V|} \exp(\mathbf{v}_{l_i}^{\top} \mathbf{v}_l^{\prime})} \qquad (2)
$$

其中 $\mathbf{v}_l$ 和 $\mathbf{v}_l^{\prime}$ 是 listing $l$ 的输入和输出向量表示，超参数 $m$ 定义为点击 listing 的前向和后向上下文（邻域）的相关长度， $V$ 是词汇表，定义为数据集中唯一 listing ID 的集合。从 (1) 和 (2) 中我们看到，所提出的方法对 listing 点击序列的时间上下文进行建模，其中具有相似上下文（即在搜索会话中具有相似邻近 listing）的 listing 将具有相似的表示。

计算 (1) 中目标函数梯度 $\nabla L$ 所需的时间与词汇表大小 $|V|$ 成正比，对于大型词汇表（例如数百万个 listing ID），这是不可行的任务。作为替代，我们使用了[17]中提出的负采样方法，显著降低了计算复杂度。负采样可以形式化如下。我们生成一组正对 $D_p$ ，由点击 listing $l$ 及其上下文 $c$ 组成（即在同一用户在点击 listing $l$ 前后窗口长度 $m$ 内对其他 listing 的点击），以及一组负对 $D_n$ ，由点击 listing 和从整个词汇表 $V$ 中随机采样的 $n$ 个 listing 组成。优化目标变为：

$$
\arg\max_{\theta} \sum_{(l,c) \in D_p} \log \frac{1}{1 + e^{-\mathbf{v}_c^{\prime} \mathbf{v}_l}} + \sum_{(l,c) \in D_n} \log \frac{1}{1 + e^{\mathbf{v}_c^{\prime} \mathbf{v}_l}} \qquad (3)
$$

其中待学习的参数 $\theta$ 是 $\mathbf{v}_l$ 和 $\mathbf{v}_c^{\prime}$ ， $l, c \in V$ 。

**预订 listing 作为全局上下文。** 我们可以将点击会话集 $S$ 分解为：1）预订会话，即以用户预订某个房源结束的点击会话；2）探索性会话，即不以预订结束的点击会话，用户只是在浏览。从捕获上下文相似性的角度来看，两者都是有用的，然而预订会话可以用来调整优化，使得在每一步中，我们不仅预测邻近的点击 listing，还预测最终预订的 listing。这种调整可以通过将预订 listing 作为全局上下文来实现，使得无论其是否在上下文窗口内，都会被预测。因此，对于预订会话，嵌入更新规则变为：

$$
\arg\max_{\theta} \sum_{(l,c) \in D_p} \log \frac{1}{1 + e^{-\mathbf{v}_c^{\prime} \mathbf{v}_l}} + \sum_{(l,c) \in D_n} \log \frac{1}{1 + e^{\mathbf{v}_c^{\prime} \mathbf{v}_l}} + \log \frac{1}{1 + e^{-\mathbf{v}_{l_b}^{\prime} \mathbf{v}_l}} \qquad (4)
$$

其中 $\mathbf{v}_{l_b}$ 是预订 listing $l_b$ 的嵌入。对于探索性会话，更新仍通过优化目标 (3) 进行。图 1 展示了如何使用大小为 $2n + 1$ 的滑动窗口从第一个点击 listing 到预订 listing 学习 listing 嵌入的图形表示。在每一步中，中心 listing 的嵌入 $\mathbf{v}_l$ 被更新，使其预测上下文 listing $D_p$ 的嵌入 $\mathbf{v}_c$ 和预订 listing 的嵌入 $\mathbf{v}_{l_b}$ 。随着窗口滑动，一些 listing 进入和离开上下文集，而预订 listing 始终作为全局上下文保留在其中（虚线）。

> **图 1：Listing 嵌入的 Skip-gram 模型**

**适应聚集搜索的训练。** 在线旅行预订网站的用户通常只在单一市场内搜索，即他们想要入住的位置。因此， $D_p$ 包含来自同一市场的 listing 的概率很高。另一方面，由于负例的随机采样， $D_n$ 很可能包含的 listing 大多与 $D_p$ 中的 listing 不在同一市场。在每一步中，对于给定的中心 listing $l$ ，正上下文主要由与 $l$ 同市场的 listing 组成，而负上下文主要由与 $l$ 不同市场的 listing 组成。我们发现这种不平衡导致学习到的市场内相似度次优。为解决此问题，我们提出添加一组随机负例 $D_{mn}$ ，从中心 listing $l$ 的市场中采样：

$$
\arg\max_{\theta} \sum_{(l,c) \in D_p} \log \frac{1}{1 + e^{-\mathbf{v}_c^{\prime} \mathbf{v}_l}} + \sum_{(l,c) \in D_n} \log \frac{1}{1 + e^{\mathbf{v}_c^{\prime} \mathbf{v}_l}} + \log \frac{1}{1 + e^{-\mathbf{v}_{l_b}^{\prime} \mathbf{v}_l}} + \sum_{(l,m_n) \in D_{mn}} \log \frac{1}{1 + e^{\mathbf{v}_{m_n}^{\prime} \mathbf{v}_l}} \qquad (5)
$$

其中待学习的参数 $\theta$ 是 $\mathbf{v}_l$ 和 $\mathbf{v}_c^{\prime}$ ， $l, c \in V$ 。

> **图 2：加利福尼亚 listing 嵌入聚类**

> **图 3：使用嵌入的相似 listing**

> **图 4：嵌入评估工具**

**冷启动 listing 嵌入。** 每天都有新房源由房东创建并在 Airbnb 上发布。此时这些 listing 没有嵌入，因为它们不在点击会话 $S$ 训练数据中。为新房源创建嵌入，我们提出利用其他 listing 的现有嵌入。在房源创建时，房东需要提供有关房源的信息，如位置、价格、listing 类型等。我们使用提供的元数据找到 3 个地理位置最近的（10 英里半径内）已有嵌入的 listing，这些 listing 与新房源类型相同（例如独立房间）且属于相同价格区间（例如每晚 $20 - $25）。然后，我们使用找到的 3 个 listing 的嵌入计算均值向量作为新房源的嵌入。使用此技术，我们能够覆盖 98% 以上的新房源。

**检查 listing 嵌入。** 为了评估嵌入捕获了 listing 的哪些特征，我们检查了使用 (5) 在 8 亿点击会话上训练的 $d = 32$ 维嵌入。首先，通过对学习到的嵌入执行 k-means 聚类，我们评估地理相似性是否被编码。图 2 显示了加利福尼亚的 100 个聚类结果，确认了来自相似位置的 listing 被聚类在一起。我们发现这些聚类对于重新评估旅行市场的定义非常有用。接下来，我们评估来自洛杉矶不同 listing 类型（表 1）的 listing 之间的平均余弦相似度，以及不同价格区间（表 2）的 listing 之间的平均余弦相似度。从这些表中可以观察到，相同类型和价格区间的 listing 之间的余弦相似度远高于不同类型和价格区间的 listing 之间的相似度。因此，我们可以得出结论，这两个 listing 特征也被很好地编码在学习到的嵌入中。

虽然某些 listing 特征（如价格）不需要学习，因为它们可以从 listing 元数据中提取，但其他类型的 listing 特征（如建筑风格、风格和感觉）更难以 listing 特征的形式提取。为了评估这些特征是否被嵌入捕获，我们可以检查独特建筑风格 listing 在 listing 嵌入空间中的 k 最近邻。图 3 展示了一个这样的案例，对于左侧的独特建筑风格 listing，最相似的 listing 具有相同的风格和建筑。为了能够在 listing 嵌入空间中进行快速便捷的探索，我们开发了如图 4 所示的内部相似度探索工具。

**表 1：不同 listing 类型之间的余弦相似度**

| 房间类型 | 独立房间 | 私人房间 | 合住房间 |
|----------|----------|----------|----------|
| 独立房间 | 0.895 | 0.875 | 0.848 |
| 私人房间 | | 0.901 | 0.865 |
| 合住房间 | | | 0.896 |

**表 2：不同价格区间之间的余弦相似度**

| 价格区间 | < $30 | $30 - $60 | $60 - $90 | $90 - $120 | $120+ |
|----------|------|---------|---------|----------|-------|
| < $30 | 0.916 | 0.887 | 0.882 | 0.871 | 0.854 |
| $30 - $60 | | 0.906 | 0.889 | 0.876 | 0.865 |
| $60 - $90 | | | 0.902 | 0.883 | 0.880 |
| $90 - $120 | | | | 0.898 | 0.890 |
| $120+ | | | | | 0.909 |

### 3.2 用户类型和 Listing 类型嵌入

第 3.1 节描述的使用点击会话训练的 listing 嵌入非常擅长在同一市场中找到相似的 listing。因此，它们适用于短期、会话内个性化，目的是向用户展示与他们在当前搜索会话中点击的 listing 相似的房源。

然而，除了基于会话内信号的个性化外，基于用户更长期历史信号来个性化搜索也是有用的。例如，给定一个当前在洛杉矶搜索房源、过去在纽约和伦敦预订过的用户，推荐与之前预订过的房源相似的 listing 会很有用。

虽然在使用点击训练的 listing 嵌入中捕获了一些跨市场相似性，但学习此类跨市场相似性的更原则性方法是从特定用户随时间预订的 listing 构建的会话中学习。具体来说，假设我们有一组来自 $N$ 个用户的预订会话 $S_b$ ，其中每个预订会话 $s_b = (l_{b_1}, \ldots, l_{b_M})$ 定义为用户 $j$ 按时间顺序预订的 listing 序列。尝试使用此类数据为每个 listing_id 学习嵌入 $\mathbf{v}_{l_{id}}$ 在许多方面具有挑战性：

* 首先，预订会话数据 $S_b$ 比点击会话数据 $S$ 小得多，因为预订是不太频繁的事件。
* 其次，许多用户在过去只预订过一个 listing，我们无法从长度为 1 的会话中学习。
* 第三，为了从上下文信息中为任何实体学习有意义的嵌入，数据中至少需要 5-10 次该实体的出现，而平台上有许多 listing_id 被预订少于 5-10 次。
* 最后，用户两次连续预订之间可能经过很长的时间间隔，在此期间用户偏好（如价格点）可能会改变，例如由于职业变化。

为解决这些在实践中非常常见的市场问题，我们提出在 listing_type 级别而非 listing_id 级别学习嵌入。给定某个 listing_id 的可用元数据（如位置、价格、listing 类型、容量、床位数等），我们使用表 3 中定义的基于规则的映射来确定其 listing_type。例如，一个来自美国的独立房间 listing，容量为 2 人、1 张床、1 间卧室和 1 间浴室，平均每晚价格 $60.8，每位客人平均每晚价格 $29.3，5 条评论，全部 5 星，100% 的新客人接受率，将映射为 listing_type = US_lt1_pn3_pg3_r3_5s4_c2_b1_bd2_bt2_nu3。桶的确定以数据驱动的方式进行，以最大化每个 listing_type 桶的覆盖范围。从 listing_id 到 listing_type 的映射是多对一映射，意味着许多 listing 将映射到相同的 listing_type。

**表 3：listing 元数据到 listing 类型桶的映射**

| 桶 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|-----|---|---|---|---|---|---|---|---|
| 国家 | US | CA | GB | FR | MX | AU | ES | ... |
| Listing 类型 | Ent | Priv | Share | | | | | |
| 每晚价格 | <40 | 40-55 | 56-69 | 70-83 | 84-100 | 101-129 | 130-189 | 190+ |
| 每位客人价格 | <21 | 21-27 | 28-34 | 35-42 | 43-52 | 53-75 | 76+ | |
| 评论数 | 0 | 1 | 2-5 | 6-10 | 11-35 | 35+ | | |
| Listing 5星率 | 0-40 | 41-60 | 61-90 | 90+ | | | | |
| 容量 | 1 | 2 | 3 | 4 | 5 | 6+ | | |
| 床数 | 1 | 2 | 3 | 4+ | | | | |
| 卧室数 | 0 | 1 | 2 | 3 | 4+ | | | |
| 浴室数 | 0 | 1 | 2 | 3+ | | | | |
| 新客人接受率 | <60 | 61-90 | >91 | | | | | |

为了考虑用户随时间变化的偏好，我们提出在与 listing_type 嵌入相同的向量空间中学习 user_type 嵌入。user_type 使用与我们应用于 listing 的类似程序确定，即利用用户及其先前预订的元数据，定义在表 4 中。例如，对于一个来自旧金山、使用 MacBook 笔记本、英语语言设置、有完整资料和用户照片、来自房东的平均 5 星评分为 83.4%、过去预订了 3 次、预订 listing 的平均统计为每晚价格 $52.52、每位客人每晚价格 $31.85、容量 2.33、评论数 8.24、listing 5 星率 76.1% 的用户，生成的 user_type 为 SF_lg1_dt1_fp1_pp1_nb1_ppn2_ppg3_c2_nr3_l5s3_g5s3。在生成用于训练嵌入的预订会话时，我们计算 user_type 直到最新预订。对于首次预订的用户，user_type 基于表 4 的前 5 行计算，因为在预订时我们没有关于先前预订的信息。这很方便，因为基于前 5 行学习的 user_type 嵌入可以用于已退出用户和没有过去预订的新用户的冷启动个性化。

**表 4：用户元数据到用户类型桶的映射**

| 桶 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|-----|---|---|---|---|---|---|---|---|
| 市场 | SF | NYC | LA | HK | PHL | AUS | LV | ... |
| 语言 | en | es | fr | jp | ru | ko | de | ... |
| 设备类型 | Mac | Msft | Andr | Ipad | Tablet | Iphone | | |
| 完整资料 | Yes | No | | | | | | |
| 资料照片 | Yes | No | | | | | | |
| 预订数 | 0 | 1 | 2-7 | 8+ | | | | |
| 每晚价格 | <40 | 40-55 | 56-69 | 70-83 | 84-100 | 101-129 | 130-189 | 190+ |
| 每位客人价格 | <21 | 21-27 | 28-34 | 35-42 | 43-52 | 53-75 | 76+ | |
| 容量 | <2 | 2-2.6 | 2.7-3 | 3.1-4 | 4.1-6 | 6.1+ | | |
| 评论数 | <1 | 1-3.5 | 3.6-10 | >10 | | | | |
| Listing 5星率 | 0-40 | 41-60 | 61-90 | 90+ | | | | |
| 客人5星率 | 0-40 | 41-60 | 61-90 | 90+ | | | | |

**训练过程。** 为了在相同向量空间中学习 user_type 和 listing_type 嵌入，我们将 user_type 合并到预订会话中。具体来说，我们构建一组 $S_b$ ，由来自 $N$ 个用户的 $N_b$ 个预订会话组成，其中每个会话 $s_b = (u_{\mathrm{type}_1} l_{\mathrm{type}_1}, \ldots, u_{\mathrm{type}_M} l_{\mathrm{type}_M}) \in S_b$ 定义为预订事件的序列，即按时间排序的（user_type, listing_type）元组。注意，每个会话由同一 user_id 的预订组成，但单个 user_id 的 user_type 可能随时间变化，类似于同一 listing 的 listing_type 随着更多预订而变化。

需要优化的目标可以类似于 (3) 定义，其中代替 listing $l$ ，需要更新的中心 item 是 user_type（ $u_t$ ）或 listing_type（ $l_t$ ），取决于哪个在滑动窗口中被捕获。例如，为了更新中心 item 为 user_type（ $u_t$ ）的情况，我们使用：

$$
\arg\max_{\theta} \sum_{(u_t, c) \in D_{\mathrm{book}}} \log \frac{1}{1 + e^{-\mathbf{v}_c^{\prime} \mathbf{v}_{u_t}}} + \sum_{(u_t, c) \in D_{\mathrm{neg}}} \log \frac{1}{1 + e^{\mathbf{v}_c^{\prime} \mathbf{v}_{u_t}}} \qquad (6)
$$

其中 $D_{\mathrm{book}}$ 包含来自用户近期历史的 user_type 和 listing_type，特别是相对于中心 item 时间戳的近期和近期预订，而 $D_{\mathrm{neg}}$ 包含用作负例的随机 user_type 或 listing_type 实例。类似地，如果中心 item 是 listing_type（ $l_t$ ），我们优化以下目标：

$$
\arg\max_{\theta} \sum_{(l_t, c) \in D_{\mathrm{book}}} \log \frac{1}{1 + e^{-\mathbf{v}_c^{\prime} \mathbf{v}_{l_t}}} + \sum_{(l_t, c) \in D_{\mathrm{neg}}} \log \frac{1}{1 + e^{\mathbf{v}_c^{\prime} \mathbf{v}_{l_t}}} \qquad (7)
$$

> **图 5：Listing 类型和用户类型 Skip-gram 模型**

图 5a（左侧）展示了此模型的图形表示，其中中心 item 代表 user_type（ $u_t$ ），更新按 (6) 进行。由于预订会话按定义主要包含来自不同市场的 listing，因此不需要像我们在第 3.1 节中那样从同一市场采样额外的负例来应对点击会话中的聚集搜索。

**拒绝作为显式负例。** 与仅反映租客偏好的点击不同，预订也反映了房东的偏好，因为存在来自房东的显式反馈——接受或拒绝租客的预订请求。房东拒绝的一些原因包括差评、不完整或空白的租客资料、没有头像等。这些特征是表 4 中 user_type 定义的一部分。房东拒绝可以在训练期间用于在嵌入空间中编码房东偏好信号，作为租客偏好信号的补充。纳入拒绝信号的整个目的是使某些 listing_type 对没有预订、资料不完整且低于平均客人评分的 user_type 不那么敏感，我们希望这些 listing_type 和 user_type 的嵌入在向量空间中更接近，使得基于嵌入相似度的推荐除了最大化预订机会外，还能减少未来的拒绝。

我们以下列方式形式化使用拒绝作为显式负例。除了 $D_{\mathrm{book}}$ 和 $D_{\mathrm{neg}}$ 集合外，我们生成一组 $D_{\mathrm{rej}}$ ，由参与拒绝事件的 user_type 或 listing_type 对（ $u_t, l_t$ ）组成。如图 5b（右侧）所示，我们特别关注房东拒绝（标有减号）后同一用户成功预订另一个 listing（标有加号）的情况。新的优化目标可以形式化为：

$$
\arg\max_{\theta} \sum_{(u_t, c) \in D_{\mathrm{book}}} \log \frac{1}{1 + e^{-\mathbf{v}_c^{\prime} \mathbf{v}_{u_t}}} + \sum_{(u_t, c) \in D_{\mathrm{neg}}} \log \frac{1}{1 + e^{\mathbf{v}_c^{\prime} \mathbf{v}_{u_t}}} + \sum_{(u_t, l_t) \in D_{\mathrm{rej}}} \log \frac{1}{1 + e^{\mathbf{v}_{l_t}^{\prime} \mathbf{v}_{u_t}}} \qquad (8)
$$

用于更新中心 item 为 user_type（ $u_t$ ）的情况，以及：

$$
\arg\max_{\theta} \sum_{(l_t, c) \in D_{\mathrm{book}}} \log \frac{1}{1 + e^{-\mathbf{v}_c^{\prime} \mathbf{v}_{l_t}}} + \sum_{(l_t, c) \in D_{\mathrm{neg}}} \log \frac{1}{1 + e^{\mathbf{v}_c^{\prime} \mathbf{v}_{l_t}}} + \sum_{(l_t, u_t) \in D_{\mathrm{rej}}} \log \frac{1}{1 + e^{\mathbf{v}_{u_t}^{\prime} \mathbf{v}_{l_t}}} \qquad (9)
$$

用于更新中心 item 为 listing_type（ $l_t$ ）的情况。

给定所有 user_type 和 listing_type 的学习嵌入，我们可以基于用户的当前 user_type 嵌入与候选 listing 的 listing_type 嵌入之间的余弦相似度，向用户推荐最相关的 listing。例如，在表 5 中，我们展示了 user_type = SF_lg1_dt1_fp1_pp1_nb3_ppn5_ppg5_c4_nr3_l5s3_g5s3（通常预订高质量、宽敞、有大量好评的 listing）与美国几个不同 listing_type 之间的余弦相似度。可以观察到，最符合这些用户偏好的 listing 类型（即独立房间、大量好评、大面积和高于平均价格）具有高余弦相似度，而不符合用户偏好的 listing 类型（即空间较小、价格较低和评论数较少）具有低余弦相似度。

**表 5：基于类型嵌入的推荐**

| User Type | SF_lg1_dt1_fp1_pp1_nb3_ppn5_ppg5_c4_nr3_l5s3_g5s3 |
|-----------|-----------------------------------------------------|
| Listing Type | 相似度 |
| US_lt1_pn4_pg5_r5_5s4_c2_b1_bd3_bt3_nu3（大面积，好评） | 0.629 |
| US_lt1_pn3_pg3_r5_5s2_c2_b1_bd2_bt2_nu3（较便宜，差评） | 0.350 |
| US_lt2_pn3_pg3_r5_5s4_c1_b1_bd2_bt2_nu3（私人房间，好评） | 0.241 |
| US_lt2_pn2_pg2_r5_5s2_c1_b1_bd2_bt2_nu3（较便宜，差评） | 0.169 |
| US_lt3_pn1_pg1_r5_5s3_c1_b1_bd2_bt2_nu3（合住房间，差评） | 0.121 |

## 4 实验

在本节中，我们首先介绍训练 listing 嵌入的细节及其离线评估。然后展示使用 listing 嵌入进行相似 listing 推荐的在线实验结果。最后，介绍搜索排序模型的背景，并描述如何使用 listing 嵌入和 listing 类型及用户类型嵌入实现搜索中的实时个性化特征。两种嵌入应用都已成功部署到生产中。

### 4.1 训练 Listing 嵌入

为了训练 listing 嵌入，我们从搜索中创建了 8 亿个点击会话，方法是获取已登录用户的所有搜索，按用户 ID 分组，并按时间对 listing ID 的点击进行排序。随后，基于 30 分钟不活动规则将一个大型有序 listing ID 列表拆分为多个列表。接下来，我们删除了意外和短暂的点击，即用户在 listing 页面停留少于 30 秒的点击，并仅保留包含 2 次或更多点击的会话。最后，通过删除用户 ID 列对会话进行了匿名化。如前所述，点击会话包括探索性会话和预订会话（以预订结束的点击序列）。根据离线评估结果，我们在训练数据中对预订会话进行了 5 倍过采样，这产生了性能最佳的 listing 嵌入。

**设置每日训练。** 我们为 450 万个 Airbnb listing 学习 listing 嵌入，训练数据的实际参数使用下面介绍的离线评估技术进行调优。我们的训练数据以滑动窗口方式在多个月内每日更新，通过处理最新一天的搜索会话并将其添加到数据集中，同时丢弃数据集中最旧一天的搜索会话。我们为每个 listing_id 训练嵌入，在训练前随机初始化向量（每次都使用相同的随机种子）。我们发现，如果每天从头开始重新训练 listing 嵌入，而不是在现有向量上增量继续训练，可以获得更好的离线性能。逐日的向量差异不会在我们的模型中造成不一致，因为在我们的应用中我们使用余弦相似度作为主要信号，而不是实际向量本身。即使向量随时间变化，余弦相似度度量的含义和范围也不会改变。

Listing 嵌入的维度设置为 $d = 32$ ，因为我们发现这是离线性能和在搜索机器 RAM 中存储向量以进行实时相似度计算所需内存之间的良好折衷。上下文窗口大小设置为 $m = 5$ ，我们在训练数据上执行了 10 次迭代。为了实现聚集搜索的算法修改，我们修改了原始 word2vec C 代码[†]。训练使用 MapReduce，其中 300 个映射器读取数据，单个归约器以多线程方式训练模型。端到端的每日数据生成和训练流水线使用 Airflow[‡] 实现，这是 Airbnb 的开源调度平台。

> [†] https://code.google.com/p/word2vec
>
> [‡] http://airbnb.io/projects/airflow

### 4.2 Listing 嵌入的离线评估

为了能够快速对优化函数、训练数据构建、超参数等的不同想法做出决策，我们需要一种快速比较不同嵌入的方法。评估训练嵌入的一种方式是测试它们在基于用户最近点击推荐用户会预订的 listing 方面有多好。更具体地说，假设我们给定最近点击的 listing 和需要排序的候选 listing，其中包含用户最终预订的 listing。通过计算点击 listing 和候选 listing 的嵌入之间的余弦相似度，我们可以对候选进行排序并观察预订 listing 的排名位置。

> **图 6：Listing 嵌入的离线评估**

为了评估目的，我们使用大量此类搜索、点击和预订事件，其中排名已由搜索排序模型分配。在图 6 中，我们展示了离线评估结果，其中我们比较了几个版本的 $d = 32$ 嵌入在基于预订前的点击对预订 listing 排名方面的表现。预订 listing 的排名对导致预订的每次点击取平均值，最远回溯到预订前的 17 次点击，直到预订前的最后一次点击。较低的值意味着较高的排名。我们比较的嵌入版本为：1）d32：使用 (3) 训练，2）d32 book：使用预订作为全局上下文 (4) 训练，3）d32 book + neg：使用预订作为全局上下文和来自同一市场的显式负例 (5) 训练。可以观察到，搜索排序模型随着更多点击而变得更好，因为它使用了记忆化特征。还可以观察到，基于嵌入相似度重新排序 listing 将是有用的，特别是在搜索漏斗的早期阶段。最后，我们可以得出结论，d32 book + neg 优于其他两个嵌入版本。相同类型的图用于对超参数、数据构建等做出决策。

### 4.3 使用嵌入的相似 Listing

每个 Airbnb 房源页面[§] 包含相似 listing 轮播，推荐与之相似且在相同日期可用的 listing。在我们测试时，相似 listing 轮播的现有算法是调用主要搜索排序模型对与给定 listing 相同位置进行搜索，然后按可用性、价格区间和 listing 类型过滤给定 listing。

> [§] https://www.airbnb.com/rooms/433392

我们进行了 A/B 测试，将现有相似 listing 算法与基于嵌入的方案进行比较，其中相似 listing 通过在 listing 嵌入空间中找到 k 最近邻产生。给定学习到的 listing 嵌入，给定 listing $l$ 的相似 listing 通过计算其向量 $\mathbf{v}_l$ 与来自同一市场且在相同日期可用的所有 listing 的向量 $\mathbf{v}_j$ 之间的余弦相似度来找到（如果设置了入住和退房日期）。检索具有最高相似度的 $K$ 个 listing 作为相似 listing。计算在线执行，使用我们的分片架构并行进行，其中嵌入的部分存储在每台搜索机器上。

A/B 测试表明，基于嵌入的方案使相似 listing 轮播的 CTR 提升了 21%（当 listing 页面设置了日期时提升 23%，无日期页面提升 20%），在相似 listing 轮播中找到最终预订 listing 的租客增加了 4.9%。鉴于这些结果，我们将基于嵌入的相似 listing 部署到生产中。

### 4.4 使用嵌入的搜索排序实时个性化

**背景。** 为了形式化描述搜索排序模型，假设我们给定关于每次搜索 $D_s = (\mathbf{x}_i, y_i), i = 1 \ldots K$ 的训练数据，其中 $K$ 是搜索返回的 listing 数量， $\mathbf{x}_i$ 是第 $i$ 个 listing 结果的特征向量， $y_i \in \{0, 0.01, 0.25, 1, -0.4\}$ 是分配给第 $i$ 个 listing 结果的标签。为了将标签分配给搜索结果中的特定 listing，我们在搜索发生后等待 1 周以观察最终结果：如果 listing 被预订则 $y_i = 1$ ，如果 listing 的房东被租客联系但未发生预订则 $y_i = 0.25$ ，如果 listing 的房东拒绝了租客则 $y_i = -0.4$ ，如果 listing 被点击则 $y_i = 0.01$ ，如果 listing 仅被查看但未被点击则 $y_i = 0$ 。在那 1 周的等待后， $D_s$ 也被缩短为仅保留用户点击的最后一个结果 $K_c \leq K$ 的搜索结果。最后，为了形成数据 $D = \bigcup_{s=1}^{N} D_s$ ，我们仅保留包含至少一个预订标签的 $D_s$ 集。每次训练新的排序模型时，我们使用最近 30 天的数据。

第 $i$ 个 listing 结果的特征向量 $\mathbf{x}_i$ 由 listing 特征、用户特征、查询特征和交叉特征组成。Listing 特征是与 listing 本身关联的特征，如每晚价格、listing 类型、房间数、拒绝率等。查询特征是与发出的查询关联的特征，如客人人数、入住时长、提前天数等。用户特征是与正在进行搜索的用户关联的特征，如平均预订价格、客人评分等。交叉特征是从两个或更多这些特征来源（listing、用户、查询）派生的特征。此类特征的示例包括查询 listing 距离：查询位置与 listing 位置之间的距离，容量匹配：查询客人人数与 listing 容量之间的差异，价格差异：listing 价格与用户历史预订平均价格之间的差异，拒绝概率：房东拒绝这些查询参数的概率，点击百分比：跟踪用户点击中该特定 listing 的百分比的实时记忆化特征等。模型使用大约 100 个特征。为简洁起见，我们不会列出所有特征。

接下来，我们将问题形式化为以搜索标签为效用的成对回归，并使用数据 $D$ 训练梯度提升决策树（GBDT）模型，使用修改为支持 Lambda Rank 的包[¶]。在离线评估不同模型时，我们使用 NDCG（Normalized Discounted Cumulative Gain，归一化折扣累积增益）——一种标准排序指标——在搜索会话的留出集上，即 80% 的 $D$ 用于训练，20% 用于测试。

> [¶] https://github.com/yarny/gbdt

最后，一旦模型训练完成，就用于在线搜索中 listing 的评分。计算搜索查询 $q$ 返回的每个 listing 的特征向量 $\mathbf{x}_i$ 所需的信号都以在线方式计算，评分使用我们的分片架构并行进行。给定所有分数，listing 按预测效用的降序展示给用户。

**表 6：搜索排序的嵌入特征**

| 特征名 | 描述 |
|--------|------|
| EmbClickSim | 与 $H_c$ 中点击 listing 的相似度 |
| EmbSkipSim | 与 $H_s$ 中跳过 listing 的相似度 |
| EmbLongClickSim | 与 $H_{lc}$ 中长点击 listing 的相似度 |
| EmbWishlistSim | 与 $H_w$ 中收藏 listing 的相似度 |
| EmbInqSim | 与 $H_i$ 中联系 listing 的相似度 |
| EmbBookSim | 与 $H_b$ 中预订 listing 的相似度 |
| EmbLastLongClickSim | 与最后一次长点击 listing 的相似度 |
| UserTypeListingTypeSim | 用户类型和 listing 类型相似度 |

**Listing 嵌入特征。** 在搜索排序模型中添加嵌入特征的第一步是将 450 万个嵌入加载到搜索后端，以便实时访问以进行特征计算和模型评分。

接下来，我们引入了几个用户短期历史集合，保存用户最近 2 周的操作，这些集合在新用户操作发生时实时更新。逻辑使用 Kafka[‖] 实现。具体来说，对于每个 user_id，我们收集并维护（定期更新）以下 listing ID 集合：

> [‖] https://kafka.apache.org

1. $H_c$ ：点击的 listing_id——用户在最近 2 周内点击的 listing。
2. $H_{lc}$ ：长点击的 listing_id——用户点击并在 listing 页面停留超过 60 秒的 listing。
3. $H_s$ ：跳过的 listing_id——用户跳过以 favor 较低位置 listing 点击的 listing。
4. $H_w$ ：收藏的 listing_id——用户在最近 2 周内添加到收藏的 listing。
5. $H_i$ ：联系的 listing_id——用户在最近 2 周内联系但未预订的 listing。
6. $H_b$ ：预订的 listing_id——用户在最近 2 周内预订的 listing。

我们进一步将每个短期历史集合 $H^*$ 拆分为包含来自同一市场的 listing 的子集。例如，如果用户点击了来自纽约和洛杉矶的 listing，他们的 $H_c$ 集合将进一步拆分为 $H_c(\mathrm{NY})$ 和 $H_c(\mathrm{LA})$ 。

最后，我们定义利用已定义集合和 listing 嵌入为每个候选 listing 产生分数的嵌入特征。特征总结在表 6 中。

下面我们描述 EmbClickSim 特征如何使用 $H_c$ 计算。表 6 顶部行的其余特征使用其对应的用户短期历史集合 $H^*$ 以相同方式计算。

为了计算候选 listing $l_i$ 的 EmbClickSim，我们需要计算其 listing 嵌入 $\mathbf{v}_{l_i}$ 与 $H_c$ 中 listing 嵌入之间的余弦相似度。我们通过首先计算 $H_c$ 市场级别的质心嵌入来实现。为了说明，假设 $H_c$ 包含 5 个来自纽约的 listing 和 3 个来自洛杉矶的 listing。这需要计算两个市场级别的质心嵌入，一个用于纽约，一个用于洛杉矶，通过平均来自每个市场的 listing ID 的嵌入。最后，EmbClickSim 计算为候选 listing 嵌入 $\mathbf{v}_{l_i}$ 与 $H_c$ 市场级别质心嵌入之间的两个相似度的最大值。

更一般地，EmbClickSim 可以表示为：

$$
\mathrm{EmbClickSim}(l_i, H_c) = \max_{m \in M} \sum_{l_h \in m, l_h \in H_c} \cos(\mathbf{v}_{l_i}, \mathbf{v}_{l_h}) \qquad (10)
$$

其中 $M$ 是用户有点击的市场集合。

除了与所有用户点击的相似度外，我们还添加了一个衡量与最新长点击相似度的特征 EmbLastLongClickSim。对于候选 listing $l_i$ ，它通过找到其嵌入 $\mathbf{v}_{l_i}$ 与 $H_{lc}$ 中最新长点击 listing $l_{\mathrm{last}}$ 的嵌入之间的余弦相似度来计算：

$$
\mathrm{EmbLastLongClickSim}(l_i, H_{lc}) = \cos(\mathbf{v}_{l_i}, \mathbf{v}_{l_{\mathrm{last}}}) \qquad (11)
$$

> **图 7：EmbClickSim、EmbSkipSim 和 UserTypeListTypeSim 的偏依赖图**

**用户类型和 Listing 类型嵌入特征。** 我们遵循类似的过程引入基于用户类型和 listing 类型嵌入的特征。我们使用 5000 万用户预订会话为 50 万个用户类型和 50 万个 listing 类型训练了嵌入。嵌入维度为 $d = 32$ ，使用滑动窗口 $m = 5$ 在预订会话上训练。用户类型和 listing 类型嵌入被加载到搜索机器内存中，以便我们可以在线计算类型相似度。

为了计算候选 listing $l_i$ 的 UserTypeListingTypeSim 特征，我们简单地查找其当前 listing 类型 $l_t$ 以及正在进行搜索的用户的当前 user_type $u_t$ ，并计算它们嵌入之间的余弦相似度：

$$
\mathrm{UserTypeListingTypeSim}(u_t, l_t) = \cos(\mathbf{v}_{u_t}, \mathbf{v}_{l_t}) \qquad (12)
$$

表 6 中的所有特征被记录了 30 天，以便它们可以被添加到搜索排序训练集 $D$ 中。特征的覆盖范围（即 $D$ 中具有特定特征填充的比例）报告在表 7 中。如预期，基于用户点击和跳过的特征具有最高的覆盖范围。

**表 7：嵌入特征覆盖范围和重要性**

| 特征名 | 覆盖范围 | 特征重要性 |
|--------|----------|------------|
| EmbClickSim | 76.16% | 5/104 |
| EmbSkipSim | 78.64% | 8/104 |
| EmbLongClickSim | 51.05% | 20/104 |
| EmbWishlistSim | 36.50% | 47/104 |
| EmbInqSim | 20.61% | 12/104 |
| EmbBookSim | 8.06% | 46/104 |
| EmbLastLongClickSim | 48.28% | 11/104 |
| UserTypeListingTypeSim | 86.11% | 22/104 |

最后，我们训练了一个添加了嵌入特征的新 GBDT 搜索排序模型。嵌入特征的特征重要性（在 104 个特征中排名）显示在表 7 中。排名靠前的特征是与用户点击的 listing 的相似度（EmbClickSim：总体排名第 5）和与用户跳过的 listing 的相似度（EmbSkipSim：总体排名第 8）。五个嵌入特征排在前 20 个特征中。如预期，基于所有过去用户预订的长期特征 UserTypeListingTypeSim 比仅考虑最近 2 周预订的短期特征 EmbBookSim 排名更好。这也表明，基于过去预订的推荐使用历史预订会话训练的嵌入比使用点击会话训练的嵌入更好。

为了评估模型是否按我们预期使用了特征，我们绘制了 3 个嵌入特征的偏依赖图：EmbClickSim、EmbSkipSim 和 UserTypeListTypeSim。这些图显示了如果我们固定除一个特征（我们正在检查的特征）之外的所有特征值，listing 的排序分数会发生什么变化。在左侧子图中可以看到，EmbClickSim 的大值（表示 listing 与用户最近点击的 listing 相似）导致更高的模型分数。中间子图显示，EmbSkipSim 的大值（表示 listing 与用户跳过的 listing 相似）导致更低的模型分数。最后，右侧子图显示，UserTypeListingTypeSim 的大值（表示用户类型与 listing 类型相似）如预期导致更高的模型分数。

**在线实验结果总结。** 我们进行了离线和在线实验（A/B 测试）。首先，我们比较了在相同数据上训练的有和没有嵌入特征的两个搜索排序模型。在表 8 中，我们以每个效用（展示、点击、拒绝和预订）的 DCU（Discounted Cumulative Utility，折扣累积效用）和整体 NDCU 总结了结果。可以观察到，添加嵌入特征导致 NDCU 提升 2.27%，其中预订 DCU 增加了 2.58%，意味着预订的 listing 在留出集中排名更高，而拒绝没有受到任何影响（DCU -0.4 持平），意味着被拒绝的 listing 没有比没有嵌入特征的模型排名更高。

**表 8：离线实验结果**

| 指标 | 百分比提升 |
|------|-----------|
| DCU -0.4（拒绝） | +0.31% |
| DCU 0.01（点击） | +1.48% |
| DCU 0.25（联系） | +1.95% |
| DCU 1（预订） | +2.58% |
| NDCU | +2.27% |

表 8 的观察结果，加上嵌入特征在 GBDT 特征重要性中排名靠前（表 7）以及特征行为与我们直觉预期一致（图 7）的发现，足以决定进行在线实验。在在线实验中，我们看到了统计显著的预订增长，嵌入特征被部署到生产中。几个月后，我们进行了一次回溯测试，尝试移除嵌入特征，结果出现了负预订增长，这是实时嵌入特征有效的另一个指标。

## 5 结论

我们提出了一种在 Airbnb 搜索排序中实现实时个性化的新方法。该方法基于用户点击和预订会话中的上下文共现学习房源和用户的低维表示。为了更好地利用可用的搜索上下文，我们将全局上下文和显式负信号等概念纳入训练过程。我们在相似 listing 推荐和搜索排序中评估了所提出的方法。在实时搜索流量上成功测试后，两种嵌入应用都被部署到生产中。

---

## 致谢

我们要感谢整个 Airbnb 搜索排序团队对项目的贡献，特别是 Qing Zhang 和 Lynn Yang。我们还要感谢 Phillippe Siclait 和 Matt Jones 创建了嵌入评估工具。本文的摘要发表在 Airbnb 的 Medium 博客[††] 上。

> [††] https://medium.com/airbnb-engineering/listing-embeddings-for-similar-listing-recommendations-and-real-time-personalization-in-search-601172f7603e

---

## 参考文献

[1] Kamelia Aryafar, Devin Guillory, and Liangjie Hong. 2016. An Ensemble-based Approach to Click-Through Rate Prediction for Promoted Listings at Etsy. In arXiv preprint arXiv:1711.01377.

[2] Ricardo Baeza-Yates, Berthier Ribeiro-Neto, et al. 1999. Modern information retrieval. Vol. 463. ACM press New York.

[3] Oren Barkan and Noam Koenigstein. 2016. Item2vec: neural item embedding for collaborative filtering. In Machine Learning for Signal Processing (MLSP), 2016 IEEE 26th International Workshop on. IEEE, 1–6.

[4] Christopher J Burges, Robert Ragno, and Quoc V Le. 2011. Learning to rank with nonsmooth cost functions. In Advances in NIPS 2007.

[5] Ting Chen, Liangjie Hong, Yue Shi, and Yizhou Sun. 2017. Joint Text Embedding for Personalized Content-based Recommendation. In arXiv preprint arXiv:1706.01084.

[6] Nemanja Djuric, Vladan Radosavljevic, Mihajlo Grbovic, and Narayan Bhamidipati. 2014. Hidden conditional random fields with distributed user embeddings for ad targeting. In IEEE International Conference on Data Mining.

[7] Nemanja Djuric, Hao Wu, Vladan Radosavljevic, Mihajlo Grbovic, and Narayan Bhamidipati. 2015. Hierarchical neural language models for joint representation of streaming documents and their content. In Proceedings of the 24th International Conference on World Wide Web. International World Wide Web Conferences Steering Committee, 248–255.

[8] Mihajlo Grbovic, Nemanja Djuric, Vladan Radosavljevic, Fabrizio Silvestri, Ricardo Baeza-Yates, Andrew Feng, Erik Ordentlich, Lee Yang, and Gavin Owens. 2016. Scalable semantic matching of queries to ads in sponsored search advertising. In SIGIR 2016. ACM, 375–384.

[9] Mihajlo Grbovic, Nemanja Djuric, Vladan Radosavljevic, Fabrizio Silvestri, and Narayan Bhamidipati. 2015. Context-and content-aware embeddings for query rewriting in sponsored search. In SIGIR 2015. ACM, 383–392.

[10] Mihajlo Grbovic, Vladan Radosavljevic, Nemanja Djuric, Narayan Bhamidipati, and Ananth Nagarajan. 2015. Gender and interest targeting for sponsored post advertising at tumblr. In Proceedings of the 21th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. ACM, 1819–1828.

[11] Mihajlo Grbovic, Vladan Radosavljevic, Nemanja Djuric, Narayan Bhamidipati, Jaikit Savla, Varun Bhagwan, and Doug Sharp. 2015. E-commerce in your inbox: Product recommendations at scale. In Proceedings of the 21th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.

[12] Aditya Grover and Jure Leskovec. 2016. node2vec: Scalable feature learning for networks. In Proceedings of the 22nd ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 855–864.

[13] Krishnaram Kenthapadi, Benjamin Le, and Ganesh Venkataraman. 2017. Personalized Job Recommendation System at LinkedIn: Practical Challenges and Lessons Learned. In Proceedings of the Eleventh ACM Conference on Recommender Systems. ACM, 346–347.

[14] Maciej Kula. 2015. Metadata embeddings for user and item cold-start recommendations. arXiv preprint arXiv:1507.08439 (2015).

[15] Benjamin Le. 2017. Deep Learning for Personalized Search and Recommender Systems. In Slideshare: https://www.slideshare.net/BenjaminLe4/deep-learning-for-personalized-search-and-recommender-systems.

[16] Steve Liu. 2017. Personalized Recommendations at Tinder: The TinVec Approach. In Slideshare: https://www.slideshare.net/SessionsEvents/dr-steve-liu-chief-scientist-tinder-at-mlconf-sf-2017.

[17] Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg S Corrado, and Jeff Dean. 2013. Distributed representations of words and phrases and their compositionality. In Advances in neural information processing systems. 3111–3119.

[18] Thomas Nedelec, Elena Smirnova, and Flavian Vasile. 2017. Specializing Joint Representations for the task of Product Recommendation. arXiv preprint arXiv:1706.07625 (2017).

[19] Shumpei Okura, Yukihiro Tagami, Shingo Ono, and Akira Tajima. 2017. Embedding-based news recommendation for millions of users. In Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. ACM, 1933–1942.

[20] Bryan Perozzi, Rami Al-Rfou, and Steven Skiena. 2014. Deepwalk: Online learning of social representations. In Proceedings of the 20th ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 701–710.

[21] Vladan Radosavljevic, Mihajlo Grbovic, Nemanja Djuric, Narayan Bhamidipati, Daneo Zhang, Jack Wang, Jiankai Dang, Haiying Huang, Ananth Nagarajan, and Peiji Chen. 2016. Smartphone app categorization for interest targeting in advertising marketplace. In Proceedings of the 25th International Conference Companion on World Wide Web. International World Wide Web Conferences Steering Committee, 93–94.

[22] Sharath Rao. 2017. Learned Embeddings for Search at Instacart. In Slideshare: https://www.slideshare.net/SharathRao6/learned-embeddings-for-search-and-discovery-at-instacart.

[23] Thomas Schmitt, François Gonard, Philippe Caillou, and Michèle Sebag. 2017. Language Modelling for Collaborative Filtering: Application to Job Applicant Matching. In IEEE International Conference on Tools with Artificial Intelligence.

[24] Yukihiro Tagami, Hayato Kobayashi, Shingo Ono, and Akira Tajima. 2015. Modeling User Activities on the Web using Paragraph Vector. In Proceedings of the 24th International Conference on World Wide Web. ACM, 125–126.

[25] Joseph Turian, Lev Ratinov, and Yoshua Bengio. 2010. Word representations: a simple and general method for semi-supervised learning. In Proceedings of the 48th annual meeting of the association for computational linguistics. Association for Computational Linguistics, 384–394.

[26] Dongjing Wang, Shuiguang Deng, Xin Zhang, and Guandong Xu. 2016. Learning music embedding with metadata for context aware recommendation. In Proceedings of the 2016 ACM on International Conference on Multimedia Retrieval.

[27] Jason Weston, Ron J Weiss, and Hector Yee. 2013. Nonlinear latent factorization by embedding multiple user interests. In Proceedings of the 7th ACM conference on Recommender systems. ACM, 65–68.

[28] Ledell Wu, Adam Fisch, Sumit Chopra, Keith Adams, Antoine Bordes, and Jason Weston. 2017. StarSpace: Embed All The Things! arXiv preprint arXiv:1709.03856.

[29] Dawei Yin, Yuening Hu, Jiliang Tang, Tim Daly, Mianwei Zhou, Hua Ouyang, Jianhui Chen, Changsung Kang, Hongbo Deng, Chikashi Nobata, et al. 2016. Ranking relevance in yahoo search. In Proceedings of the 22nd ACM SIGKDD.
