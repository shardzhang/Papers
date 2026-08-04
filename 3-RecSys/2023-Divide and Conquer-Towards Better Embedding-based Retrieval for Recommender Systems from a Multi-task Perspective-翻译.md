# 分而治之：面向更好的基于嵌入的检索在多任务视角下的推荐系统

> Yuan Zhang, Biao Li | Kuaishou Technology

本文介绍了 分而治之：面向更好的基于嵌入的检索在多任务视角下的推荐系统。核心内容：


关键发现：


**Biao Li**
快手科技
biaoli6@139.com

**Xue Dong**
山东大学
dongxue.sdu@gmail.com

**Peng Jiang**
快手科技
jp2006@139.com

**Weijie Ding**
快手科技
dingweijie@kuaishou.com

**Kun Gai**
独立学者
gai.kun@qq.com


---

## 摘要

基于嵌入的检索方法因其简洁性和有效性被广泛应用于现代推荐系统。然而，在部署和迭代基于嵌入的检索生产系统的过程中，我们仍然发现现有方法存在一些根本性问题。首先，当处理大规模候选项集时，基于嵌入的检索模型通常难以平衡区分高相关项（正样本）与不相关项（简单负样本）以及区分高相关项与部分相关但不具竞争力的项（困难负样本）之间的性能。此外，由于最近邻向量搜索的"贪婪"特性，我们对检索结果的多样性和公平性几乎无法控制。这些问题在大规模工业场景中严重影响了基于嵌入的检索方法的性能。

本文提出了一种简单且经过生产验证的解决方案来克服这些问题。所提出的解决方案采用了分而治之的思路：将整个候选集划分为多个聚类，并在每个聚类上并行运行基于嵌入的检索以检索相关候选；然后通过可控的合并策略将每个聚类的顶部候选组合起来。这种方法使得基于嵌入的检索模型只需专注于区分正样本与困难负样本。它还从多任务学习的角度实现了进一步的改进：每个聚类内的检索问题可以被视为独立的任务；受近期提示学习和前缀微调成功应用的启发，我们提出了一种高效的任务适应技术，以极小的开销进一步提升每个聚类内的检索性能。

离线评估和在线A/B实验都证明了所提出解决方案的有效性。该方案自A/B测试取得正向结果以来，已在快手（中国最受欢迎的短视频平台之一，拥有数亿活跃用户）部署超过四个月。

### CCS概念

- **信息系统** \rightarrow **推荐系统**。

### 关键词

推荐系统、基于嵌入的检索、多任务学习

## 1 引言

在有限时间内从极其庞大的候选项集中找到精确相关的候选项，是众多工业推荐系统面临的主要挑战之一。最广泛使用的解决方案是所谓的两阶段方法[2]，其中候选生成阶段首先负责缩小候选项集，以便在后续排序阶段中利用更精确但耗时更高的模型。传统上，候选生成通常通过基于规则的方法实现，如基于标签的推荐和基于item的协同过滤[15, 24]。随着深度学习的成功应用，基于嵌入的检索方法[2, 9, 16, 23, 25]如今变得非常普遍。在基于嵌入的检索中，用户和item特征通过两个并行的深度神经网络编码为嵌入向量，用户向量与item向量之间的距离（如点积）被学习用于区分相关项（即正样本）与不相关项（即负样本）。在部署期间，item嵌入可以预先计算并由近似最近邻搜索系统（如FAISS[10]）建立索引，从而在服务时以亚线性时间高效检索 top-k 相关项。

然而，尽管基于嵌入的检索在简洁性和有效性之间取得了良好的平衡，它仍然面临一些根本性问题：

- **区分简单负样本与困难负样本之间的权衡。** 与排序阶段的候选大多具有一定相关性不同，候选生成阶段面对的是整个候选集，需要将具有竞争力的候选（正样本）与不相关项（简单负样本）以及部分相关但不具竞争力的项（困难负样本）区分开来。挑战在于，嵌入模型在这两个任务上的性能往往是相互冲突的（见图1(a)）。通常，训练基于嵌入的检索模型的最佳实践是使用混合比例经过精心调优的简单负样本和困难负样本。尽管如此，我们认为这个表面上的"甜点"并非最优，而是一种折中，它限制了基于嵌入的检索模型的潜力。

- **检索结果的多样性和公平性不可控[20]。** 在大多数平台上，用户通常对多种item感兴趣，且兴趣强度各不相同。例如，一个大部分时间观看喜剧的用户有时也会观看惊悚片。理想情况下，推荐系统应呈现多样化的结果，覆盖用户的所有信息需求。然而，基于嵌入的检索中使用的近似最近邻搜索算法只贪婪地检索得分最高的项，忽略了用户意图潜在的多样性（见图1(b)）。因此，结果的多样性完全取决于嵌入模型，而长尾用户兴趣（例如例子中的惊悚片）在这种情况下很容易被低估。

受这些问题的启发，我们提出了一种简单且实用的方法，几乎可以零成本地显著提升基于嵌入的检索模型的性能。所提出的方法类似于分而治之算法。具体来说，我们将整个item集划分为若干个聚类，并假设同一聚类内的item大多是相互关联的。我们不再使用单一的基于嵌入的检索模型从整个候选集中搜索相关候选，而是在每个聚类（或相关聚类的子集）上并行运行基于嵌入的检索，然后通过某些策略合并结果[1]。这种方式排除了来自其他不相关聚类的简单负样本，使得基于嵌入的检索模型只需专注于每个聚类内的困难负样本。因此，它实现了双赢局面：嵌入模型可以更精确，同时通过设计适当的合并策略，多样性和公平性变得可控。

值得注意的是，所提出的分而治之方法自然地引出了多任务学习场景[3]。从每个聚类中检索相关候选可以被视为一个独立的任务。这种多任务视角为从多任务学习角度优化模型精度开辟了新的方向。然而，在实际应用中，我们发现最先进的多任务学习方法（如MMoE[18]、PLE[21]）显著增加了计算成本（尤其是在训练阶段），但并未达到预期的性能提升。在本文中，我们从近期基于提示的轻量级微调研究[12, 14]中汲取灵感，通过引入极少的额外参数和可忽略的计算开销，在输入层促进任务适应。由此产生的方法能够在统一的模型架构下对每个任务进行更高效的微调，并实现显著提升的精度。借助这一多任务学习范式，我们将基于嵌入的检索的潜力进一步推向深入。

我们将所提出的方法部署在快手（中国最受欢迎的短视频平台之一）的大规模推荐系统中。在线A/B测试表明，我们的方法显著改善了包括App使用时长在内的关键用户参与度指标。为确保更广泛研究社区的可复现性，我们还在公开数据集上进行了广泛的实验。结果表明，我们的方法在召回率上取得了高达40%的提升。详细的消融研究进一步验证了本文两个主要贡献——所提出的分而治之方法（第2.2节）和基于提示的任务适应技术（第2.3节）——的有效性。

通过这项初步研究，我们希望展示在现实应用场景中改进基于嵌入的检索的一种新的可能性和视角。一些细节设计可以作为开放性问题留待未来研究。例如，可能存在更好且更有原则的方式来划分候选空间，以及根据不同的需求合并每个聚类的结果。此外，研究人员还可以为此场景探索更先进的多任务学习方法，以在效果和效率之间取得更好的平衡。

## 2 方法

### 2.1 背景：基于嵌入的检索

假设我们有一个用户集 U 和一个item集 I。我们将用户侧特征表示为每个用户 u \in U 的 **x**_u，将item侧特征表示为每个item i \in I 的 **x**_i。基于嵌入的检索方法使用用户编码器 f 和item编码器 g 将用户和item特征转换为嵌入 **e**_u = f(**x**_u) 和 **e**_i = g(**x**_i)。用户嵌入 **e**_u 与item嵌入 **e**_i 之间的距离（最常见的是内积）用于表示item i 对用户 u 的相关性，记为 r_ui。

在部署期间，item嵌入 {**e**_i}_i\inI 在离线或近线系统中预先计算，并由近似最近邻搜索系统（如FAISS[10]）建立索引。因此，只有用户编码器 **e**_u = f(**x**_u) 需要实时计算，近似最近邻系统可以高效地以亚线性时间检索最近邻item。基于嵌入的检索最吸引人的地方在于，f 和 g 可以是几乎任何神经网络模型，且部署成本极低。

在本文中，我们以SASRec[11]为例介绍所提出的解决方案。SASRec使用Transformer[22]作为用户编码器：

**e**_u = f(**x**_u) = Transformer([**s**^u_1, **s**^u_2, ..., **s**^u_n])   (1)

其中 **s**^u_1, **s**^u_2, ..., **s**^u_n 是用户 u 最近点击的 n 个item的嵌入，并使用隐式itemID嵌入单独作为item表示。

在训练过程中，item相关分数被训练用于区分用户点击的item（即 I_u）和其余item（即 I \ I_u）。例如，SASRec使用二元交叉熵损失来训练相关性模型：

L = −\sum_{u\inU} \sum_{i^+\inI_u} [log(\sigma(r_ui^+)) + $E_{i^−∼I\I_u}$ [log(1 − \sigma(r_ui^−))]]   (2)

### 2.2 提出的分而治之方法

如上文引言所述，负样本的结构 I \ I_u 可能非常多样化，其中一些明显不相关，而另一些可能相关但与某些其他项相比不具备足够的竞争力。随着候选集 I 规模的增大，这个问题变得更加突出。

我们的解决方案源于"分而治之"的经典思想。我们首先将整个候选集划分为 K 个语义相关的聚类 I = {C_1, C_2, ..., C_K}，然后利用基于嵌入的检索模型在每个聚类内而不是整个候选集中检索相关项（见图2）。需要注意的是，聚类标准可以根据不同的应用场景选择。我们的离线实验表明，使用Word2Vec[19]训练的item嵌入进行K-means[17]聚类是一个不错的默认选择。在我们的生产系统中，出于冷启动和可解释性的考虑，我们使用由内容特征（标题、脚本和图像等）预测的内部视频类别（如体育、美食、儿童等）。

检索空间的划分使得基于嵌入的检索模型只需处理每个聚类内的候选。因此，我们只需使用与正样本属于同一聚类（如前所述，这些主要是困难负样本）的负样本来进行训练。换句话说，这种方式帮助基于嵌入的检索模型变得更加专注，从而可能更加"高效"。训练损失相应地改写为：

L = −\sum_{k=1}^{K} \sum_{u\inU} \sum_{i^+\inI_u\capC_k} [log(\sigma(r_ui^+)) + $E_{i^−∼C_k\I_u}$ [log(1 − \sigma(r_ui^−))]]   (3)

一旦从每个聚类中获得了得分最高的item，我们可以为每个聚类分配适当的配额，以满足最终检索结果的自定义要求。例如，假设我们想要生成一个大小为 M 的最终候选集。我们可以使用与用户编码器相同的特征训练一个用户意图模型，预测相关项落入聚类 C_k 的概率 p_uk。然后，我们从每个聚类中合并 top-M_k 个项作为最终结果，其中 M_k = M · [(p_uk)^\alpha / \sum_{k'=1}^{K} (p_uk^{\prime})^\alpha]，\alpha 是一个可调节的超参数[2]。由于基于嵌入的检索可以在不同的item聚类上并行运行，且每次运行的搜索空间大约缩小了 K 倍，这种分而治之的过程不会显著增加响应延迟。

[1] 例如，如果追求聚类之间的绝对公平，我们可以从每个聚类中提取相同大小的item作为最终结果。
[2] 设置 \alpha = 0 导致绝对公平推荐，而 \alpha \rightarrow +\infty 对应仅推荐最相关的聚类。

### 2.3 提示式多任务学习

观察式(3)中的训练损失，我们实际上是在训练 K 个独立的子任务，其中第 k 个子任务对应于从 C_k 中检索相关项。因此，可能有机会利用这些子任务之间的正向迁移，而不是简单地将所有任务样本混合在一起。这种多任务学习的视角促使我们尝试一些最先进的多任务学习方法，例如MMoE[18]。然而，如上所述，无论是其效果还是效率都不如预期（见第3.1.3节）。

受提示式微调方法[7, 12, 14]的启发，我们尝试将任务标识作为提示输入到用户编码器中，以参数高效的方式促进任务适应。即，对于来自每个聚类 C_k 的样本：

**e**_u = f(**x**_u) = Transformer([**t**_k; **s**^u_1, **s**^u_2, ..., **s**^u_n])   (4)

其中 **t**_k 是第 k 个任务的可训练嵌入。然而，尽管这种方法减少了训练时间，性能仍未得到显著提升。

我们假设原因在于推荐模型中使用的Transformer层数远小于大多数自然语言处理场景[3]。因此，仅通过几层自注意力块无法充分捕捉提示与原始输入之间的交互。于是，我们提出通过哈达玛积（Hadamard product）来施加显式的特征交互：

**e**_u = f(**x**_u) = Transformer([**s**^u_1 \odot **t**_k, **s**^u_2 \odot **t**_k, ..., **s**^u_n \odot **t**_k])   (5)

这个技巧与HyperPrompt[7]中的处理方法类似，即将每个输入标记嵌入与提示拼接。

[3] SASRec论文报告称两层Transformer给出了最佳结果。

## 3 实验

### 3.1 离线评估

#### 3.1.1 数据集

我们在两个公开数据集上进行了大量实验。两个数据集的统计信息如表1所示。

**表1：数据集统计（预处理后）**

| 数据集 | #用户 | #item | #交互 | 密度 | #聚类 |
|--------|-------|-------|-------|------|-------|
| ML-1M | 6,040 | 3,706 | 1,000,209 | 0.045 | 10 |
| KuaiRand | 25,828 | 108,025 | 20,141,835 | 0.007 | 15 |

- **ML-1M[5]** 包含由6,040名MovieLens用户对三千多部电影进行的一百万条匿名评分。用户观看过的所有电影都被视为相关。

- **KuaiRand[4]** 是一个从快手推荐系统日志中收集的公开数据集。在我们的实验中，我们将点击的item视为与用户相关的item，并仅保留来自主要推荐场景的交互（"tab"字段等于1）。我们还过滤掉了频率分别低于70和10的item和用户。

#### 3.1.2 评估协议与基线

遵循文献[6, 11]，我们将用户最后和倒数第二次交互的item作为测试和验证的标准答案。为了模拟真实应用场景，我们让每种方法从所有未见item中生成一个包含 M 个相关item的候选集，并评估其召回率。我们将 M 设置为item总数的约5%至10%，即ML-1M数据集的 M \in [20, 50]，KuaiRand数据集的 M \in [500, 1000]。

我们将所提出的分而治之方法与以下基线模型进行比较：

- **矩阵分解[8]** 是一种经典的推荐方法。我们使用与式(2)相同的训练目标来训练矩阵分解模型。

- **SASRec[11]** 采用自注意力块来建模用户交互序列。除非另有说明，我们使用与其原始论文相同的超参数。

- **SASRec+** 是SASRec的扩展，使用混合了简单负样本和困难负样本（与正样本属于同一聚类的负样本）进行训练。我们在验证集上调优混合比例，并报告最佳结果。

- **MIND[13]** 使用胶囊网络中的路由机制将用户行为分组为多个聚类，并获得多个用户嵌入用于检索。

- **ComiRec-SA[1]** 使用多头注意力机制为每个用户生成多个嵌入，以捕捉其多样化的兴趣。

- **MMoE[18]** 是一种广泛使用的多任务学习方法。我们在实验中将自注意力块作为独立的专家网络，并使用四个专家。

#### 3.1.3 实验结果

**表2：在ML-1M和KuaiRand上的性能对比。最后一行报告了我们提出的方法相对于其基础模型SASRec的相对提升。R@M是Recall@M的缩写。**

| 方法 | ML-1M R@20 | ML-1M R@50 | KuaiRand R@500 | KuaiRand R@1000 |
|------|-----------|-----------|---------------|----------------|
| MF | 0.120 | 0.231 | 0.086 | 0.132 |
| SASRec | 0.183 | 0.337 | 0.178 | 0.254 |
| SASRec+ | 0.220 | 0.367 | 0.200 | 0.284 |
| MIND | 0.186 | 0.343 | 0.138 | 0.201 |
| ComiRec | 0.192 | 0.341 | 0.188 | 0.269 |
| Ours w/ Naive MTL | 0.234 | 0.375 | 0.254 | 0.359 |
| Ours w/ MMoE | 0.221 | 0.366 | 0.245 | 0.344 |
| Ours | 0.224 | 0.371 | 0.239 | 0.341 |
| **提升** | **+27.9%** | **+11.3%** | **+42.7%** | **+41.3%** |

首先，所提出的方法在两个数据集上始终优于所有基线方法，在ML-1M和KuaiRand数据集上相比其基础模型SASRec分别取得了高达27.9%和42.7%的提升。如引言所述，虽然SASRec+通过使用精心调优的简单负样本和困难负样本混合策略确实优于SASRec，但SASRec+与我们的方法之间仍然存在很大差距。

我们还进行了消融研究，以评估所提出的分而治之方法和提示式多任务学习方法。结果表明，去掉提示模块后（即使用朴素多任务学习，对所有任务共享同一个SASRec模型），在ML-1M和KuaiRand上的整体性能分别下降了高达5.6%和4.2%。然而，它仍然显著优于SASRec+的性能，这表明我们的分而治之策略确实起到了预期的作用。为了进一步证明这一点，我们通过将候选集限制在与标准答案item相同聚类内的item来评估聚类内检索性能。如表3所示，通过在训练中只关注聚类内的负样本，朴素多任务学习已经能够显著提高每个聚类内的检索性能，进而带来整体性能的提升。当添加提示以促进任务适应时，聚类内检索性能表现更好。

**表3：不同多任务学习方法在标准答案item所在聚类内的检索性能。**

| 方法 | ML-1M R@5 | ML-1M R@20 | ML-1M R@50 | ML-1M R@100 | KuaiRand R@500 | KuaiRand R@1000 |
|------|----------|-----------|-----------|------------|---------------|----------------|
| SASRec | 0.191 | 0.424 | 0.639 | 0.172 | 0.399 | 0.543 |
| Naive MTL | 0.217 | 0.463 | 0.674 | 0.228 | 0.502 | 0.658 |
| MMoE | 0.223 | 0.482 | 0.682 | 0.218 | 0.496 | 0.655 |
| Ours | 0.235 | 0.483 | 0.682 | 0.237 | 0.522 | 0.682 |

同时，MMoE在ML-1M上取得了可比较的结果，但从表4可以看出，它显著增加了训练成本，训练吞吐量下降了约70%。相比之下，所提出的方法几乎没有带来额外的训练成本。

**表4：不同方法的训练吞吐量。**

| 方法 | SASRec | MMoE | Ours (提示式) |
|------|--------|------|-------------|
| 吞吐量（样本/秒） | 25K | 7.2K | 25K |

### 3.2 在线A/B实验

所提出的方法还在快手的在线A/B实验中进行了测试。A/B测试持续了三天（2022年7月3日至5日），实验组涉及超过4000万用户，得出了统计上非常显著的实验结果。在实验组中，所提出的检索方法作为候选生成阶段的候选来源之一。与对照组相比，我们的方法在大多数关键用户参与度指标上带来了显著的改进，如表5所示。我们进一步比较了来自我们来源与ComiRec来源最终展示给用户的item的参与率。如表6所示，我们的方法表现也更好。

**表5：快手在线A/B实验结果。所有性能提升在 p < 0.05 水平下统计显著。**

| App使用时长 | 点赞 | 关注 | 分享 | 下载 |
|-----------|------|------|------|------|
| +0.096% | +0.75% | +1.01% | +1.04% | +2.40% |

**表6：ComiRec与我们的方法推荐的item的在线参与率对比。**

| 指标(%) | 点击率 | 点赞率 | 关注率 | 分享率 |
|---------|--------|--------|--------|--------|
| ComiRec | 56.1 | 2.85 | 0.283 | 0.209 |
| Ours | 60.5 | 2.92 | 0.431 | 0.239 |
| 提升 | +7.8% | +2.5% | +52.3% | +14.4% |

## 4 结论

本文介绍了一种简单且经过生产验证的解决方案，用于克服基于嵌入的检索方法现有的局限性，并有效提升其性能。我们的贡献有两个方面：分而治之方法（第2.2节）和提示式多任务学习技术（第2.3节）。在离线实验和在线A/B实验中的大量结果证明了所提出解决方案的有效性。

## 致谢

我们感谢匿名审稿人的宝贵意见和建议。

## 参考文献

[1] Yukuo Cen, Jianwei Zhang, Xu Zou, Chang Zhou, Hongxia Yang, and Jie Tang. 2020. Controllable Multi-Interest Framework for Recommendation. In Conference on Knowledge Discovery and Data Mining. ACM, 2942–2951.

[2] Paul Covington, Jay Adams, and Emre Sargin. 2016. Deep neural networks for youtube recommendations. In Proceedings of the 10th ACM conference on recommender systems. 191–198.

[3] Michael Crawshaw. 2020. Multi-task learning with deep neural networks: A survey. arXiv preprint arXiv:2009.09796 (2020).

[4] Chongming Gao, Shijun Li, Yuan Zhang, Jiawei Chen, Biao Li, Wenqiang Lei, Peng Jiang, and Xiangnan He. 2022. KuaiRand: An Unbiased Sequential Recommendation Dataset with Randomly Exposed Videos. In Proceedings of the 31st ACM International Conference on Information and Knowledge Management (Atlanta, GA, USA) (CIKM '22). 5 pages. https://doi.org/10.1145/3511808.3557624

[5] F. Maxwell Harper and Joseph A. Konstan. 2016. The MovieLens Datasets: History and Context. ACM Trans. Interact. Intell. Syst. 5, 4 (2016), 19:1–19:19.

[6] Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu, and Tat-Seng Chua. 2017. Neural collaborative filtering. In Proceedings of the 26th international conference on world wide web. 173–182.

[7] Yun He, Steven Zheng, Yi Tay, Jai Gupta, Yu Du, Vamsi Aribandi, Zhe Zhao, YaGuang Li, Zhao Chen, Donald Metzler, et al. 2022. Hyperprompt: Prompt-based task-conditioning of transformers. In International Conference on Machine Learning. PMLR, 8678–8690.

[8] Yifan Hu, Yehuda Koren, and Chris Volinsky. 2008. Collaborative filtering for implicit feedback datasets. In 2008 Eighth IEEE international conference on data mining. Ieee, 263–272.

[9] Jui-Ting Huang, Ashish Sharma, Shuying Sun, Li Xia, David Zhang, Philip Pronin, Janani Padmanabhan, Giuseppe Ottaviano, and Linjun Yang. 2020. Embedding-based retrieval in facebook search. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 2553–2561.

[10] Jeff Johnson, Matthijs Douze, and Hervé Jégou. 2019. Billion-scale similarity search with GPUs. IEEE Transactions on Big Data 7, 3 (2019), 535–547.

[11] Wang-Cheng Kang and Julian J. McAuley. 2018. Self-Attentive Sequential Recommendation. In International Conference on Data Mining. IEEE Computer Society, 197–206.

[12] Brian Lester, Rami Al-Rfou, and Noah Constant. 2021. The Power of Scale for Parameter-Efficient Prompt Tuning. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics, Online and Punta Cana, Dominican Republic, 3045–3059. https://doi.org/10.18653/v1/2021.emnlp-main.243

[13] Chao Li, Zhiyuan Liu, Mengmeng Wu, Yuchi Xu, Huan Zhao, Pipei Huang, Guoliang Kang, Qiwei Chen, Wei Li, and Dik Lun Lee. 2019. Multi-Interest Network with Dynamic Routing for Recommendation at Tmall. In International Conference on Information and Knowledge Management. ACM, 2615–2623.

[14] Xiang Lisa Li and Percy Liang. 2021. Prefix-Tuning: Optimizing Continuous Prompts for Generation. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers). Association for Computational Linguistics, Online, 4582–4597. https://doi.org/10.18653/v1/2021.acl-long.353

[15] Greg Linden, Brent Smith, and Jeremy York. 2003. Amazon. com recommendations: Item-to-item collaborative filtering. IEEE Internet computing 7, 1 (2003), 76–80.

[16] Yiqun Liu, Kaushik Rangadurai, Yunzhong He, Siddarth Malreddy, Xunlong Gui, Xiaoyi Liu, and Fedor Borisyuk. 2021. Que2Search: fast and accurate query and document understanding for search at Facebook. In Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining. 3376–3384.

[17] Stuart Lloyd. 1982. Least squares quantization in PCM. IEEE transactions on information theory 28, 2 (1982), 129–137.

[18] Jiaqi Ma, Zhe Zhao, Xinyang Yi, Jilin Chen, Lichan Hong, and Ed H Chi. 2018. Modeling task relationships in multi-task learning with multi-gate mixture-of-experts. In Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining. 1930–1939.

[19] Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. 2013. Efficient estimation of word representations in vector space. arXiv preprint arXiv:1301.3781 (2013).

[20] Marco Morik, Ashudeep Singh, Jessica Hong, and Thorsten Joachims. 2020. Controlling fairness and bias in dynamic learning-to-rank. In Proceedings of the 43rd international ACM SIGIR conference on research and development in information retrieval. 429–438.

[21] Hongyan Tang, Junning Liu, Ming Zhao, and Xudong Gong. 2020. Progressive Layered Extraction (PLE): A Novel Multi-Task Learning (MTL) Model for Personalized Recommendations. In Conference on Recommender Systems. ACM, 269–278.

[22] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. Advances in neural information processing systems 30 (2017).

[23] Ji Yang, Xinyang Yi, Derek Zhiyuan Cheng, Lichan Hong, Yang Li, Simon Xiaoming Wang, Taibai Xu, and Ed H Chi. 2020. Mixed negative sampling for learning two-tower neural networks in recommendations. In Companion Proceedings of the Web Conference 2020. 441–447.

[24] Xiaoyong Yang, Yadong Zhu, Yi Zhang, Xiaobo Wang, and Quan Yuan. 2020. Large Scale Product Graph Construction for Recommendation in E-commerce. arXiv preprint arXiv:2010.05525 (2020).

[25] Xinyang Yi, Ji Yang, Lichan Hong, Derek Zhiyuan Cheng, Lukasz Heldt, Aditee Kumthekar, Zhe Zhao, Li Wei, and Ed Chi. 2019. Sampling-bias-corrected neural modeling for large corpus item recommendations. In Proceedings of the 13th ACM Conference on Recommender Systems. 269–277.
