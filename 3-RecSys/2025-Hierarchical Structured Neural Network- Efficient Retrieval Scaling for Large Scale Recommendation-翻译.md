# 层次化结构化神经网络：面向大规模推荐的高效检索扩展


本文介绍了 层次化结构化神经网络：面向大规模推荐的高效检索扩展。核心内容：


关键发现：

---


> **Kaushik Rangadurai** krangadu@meta.com *Meta Platforms Inc., Sunnyvale, CA, USA*
**Siyang Yuan** syyuan@meta.com *Meta Platforms Inc., Sunnyvale, CA, USA*
**Minhui Huang** mhhuang@meta.com *Meta Platforms Inc., Sunnyvale, CA, USA*
**Yiqun Liu** yiqliu@meta.com *Meta Platforms Inc., Sunnyvale, CA, USA*
**Golnaz Ghasemiesfeh** golnazghasemi@meta.com *Meta Platforms Inc., Sunnyvale, CA, USA*
**Yunchen Pu** pyc40@meta.com *Meta Platforms Inc., Sunnyvale, CA, USA*
**Haiyu Lu** hylu@meta.com *Meta Platforms Inc., Sunnyvale, CA, USA*
**Xingfeng He** xingfenghe@meta.com *Meta Platforms Inc., Sunnyvale, CA, USA*
**Fangzhou Xu** fxu@meta.com *Meta Platforms Inc., Sunnyvale, CA, USA*
**Andrew Cui** andycui97@meta.com *Meta Platforms Inc., Sunnyvale, CA, USA*
**Vidhoon Viswanathan** vidhoon@meta.com *Meta Platforms Inc., Sunnyvale, CA, USA*
**Lin Yang** ylin1@meta.com *Meta Platforms Inc., Sunnyvale, CA, USA*
**Liang Wang** liangwang@meta.com *Meta Platforms Inc., Sunnyvale, CA, USA*
**Jiyan Yang** chocjy@meta.com *Meta Platforms Inc., Sunnyvale, CA, USA*
**Chonglin Sun** clsun@meta.com *Meta Platforms Inc., Sunnyvale, CA, USA*

## 摘要

检索（Retrieval）是推荐系统的初始阶段，其任务是从数千万候选池中筛选出几千个候选对象。基于嵌入的检索（Embedding Based Retrieval, EBR）是解决该问题的典型方法，它能够应对海量item语料库对深度神经网络带来的计算压力。EBR 利用双塔（Two Tower）或孪生网络（Siamese Networks）学习用户和item的表示，并采用近似最近邻（ANN）搜索高效检索相关item。尽管 EBR 在工业界广受欢迎，但它也存在局限性。双塔架构依赖于单一的点积交互，由于在学习用户与item之间表达性交互方面的能力有限，因此难以捕捉复杂的数据分布。此外，ANN 索引构建以及用户和item的表示学习通常是分离的，这会导致不一致性，且这种不一致性会因表示漂移（例如持续的在线训练）和item漂移（例如item过期和新item加入）而加剧。在本文中，我们提出层次化结构化神经网络（Hierarchical Structured Neural Network, HSNN），这是一种高效的深度神经网络模型，能够在检索任务中学习超越常用点积的复杂用户-item交互，且实现相对于语料库大小的次线性计算成本。我们设计了模块化神经网络（Modular Neural Network, MoNN），在保持高效性的同时确保交互学习的高表达力。多个 MoNN 的混合模型在层次化item索引上运行，实现大量计算共享，从而能够扩展到大规模语料库。MoNN 与层次化索引联合学习，以持续适应分布漂移（包括用户兴趣漂移和item分布漂移）。与现有方法相比，HSNN 在离线评估中取得了显著改进。HSNN 已成功部署于 Meta 的广告推荐系统，并取得了显著的在线指标增益，证明了所提出方法在生产环境中的有效性。

### CCS 概念

- **计算方法** $\to$ 机器学习。

### 关键词

深度检索，聚类，推荐系统

### ACM 引用格式

Kaushik Rangadurai, Siyang Yuan, Minhui Huang, Yiqun Liu, Golnaz Ghasemiesfeh, Yunchen Pu, Haiyu Lu, Xingfeng He, Fangzhou Xu, Andrew Cui, Vidhoon Viswanathan, Lin Yang, Liang Wang, Jiyan Yang, and Chonglin Sun. 2024. Hierarchical Structured Neural Network: Efficient Retrieval Scaling for Large Scale Recommendation. *In Proceedings of ... ACM, New York, NY, USA, 10 pages.*

---

## 1 引言

机器学习在推荐系统中扮演着识别潜在用户兴趣的关键角色。为了应对每次请求的大量候选对象，工业实践 [6, 10] 通常采用计算成本递增的级联推荐系统。第一阶段称为检索阶段（retrieval stage），将数百万候选对象缩小到几千个。

检索阶段具有严格的基础设施约束，因此像孪生网络 [2] 这样的模型架构成为常见选择。孪生网络（也称为双塔模型）采用晚期融合（late fusion）技术，每个塔输出一个固定大小的表示。用户塔仅使用用户特征（如用户所在国家或用户点击历史），在查询时计算。item（广告）塔仅使用item特征（如item的内容或主题），异步计算，通常由item更新触发。用户和item嵌入之间的单一逐元素点积交互产生一个 logit，用于预测交互可能性。

尽管双塔模型架构广受欢迎，但其晚期融合技术存在局限性。这限制了用户与item之间复杂交互的学习，迫使特征要么属于用户（如用户观看过的视频），要么属于item（如item的内容），无法利用 <用户, item> 交叉交互特征（如用户对该item主题的历史交互）或排序阶段常见的基于模型的交互（如 NeuMF [16]、DCN [38] 和 DHEN [44]）。

训练完双塔模型后，用户塔和item塔独立部署。item嵌入被异步计算，用于使用 Manas ([8, 9]) 和 FAISS ([15, 18]) 等系统创建向量索引。基于嵌入的检索（EBR）[20] 结合近似最近邻（ANN）搜索 [21] 算法被广泛用于检索最相关的候选对象。尽管这些方法在工业界很流行 [8, 9, 15, 18]，但从索引构建到评分时刻，item嵌入存在不一致性。导致item嵌入分布动态变化的主要因素有两个：(a) 模型通过在线训练持续学习和更新权重，导致嵌入分布不断演变；(b) item分布本身也随时间变化，包括过期item被移除和新item被添加到索引中。

NeuMF [16]、DCN [38] 和 DHEN [44] 是先进的架构，但其计算复杂度使其无法直接用于检索。以往的研究，如基于树的深度模型（TDM）[46] 和深度检索（DR）[11]，提出了在检索中实现超越双塔模型的先进神经网络的方法。然而，这些方法受限于在树层次结构中使用相同的模型架构，这限制了模型复杂度的进一步扩展。此外，TDM 和 DR 提出的架构在捕捉用户与item交互方面缺乏表达力，无法利用 <用户, item> 交叉交互特征。此外，它们需要使用期望最大化（EM）等方法迭代训练来学习层次化索引，这既困难又低效。

我们提出层次化结构化神经网络（HSNN），它可以作为任何基于嵌入的检索（EBR）系统的即插即用替代方案。HSNN 将深度神经网络引入检索，能够在整个语料库中搜索，同时在准确性和效率上均有提升。本文的主要贡献包括：

- 我们提出 HSNN，能够处理数千万到数亿的item，推理成本为次线性。具体包括：a) 建立模块化神经网络（MoNN），高效学习用户与item之间的复杂深度神经网络交互；b) 在 MoNN 中引入高度优化的特征转换算法，以消费 <用户, item> 交互特征；c) 多个不同复杂度的 MoNN 在一个层次化索引上协调运行，落在同一索引节点下的item可共享大部分计算。
- 我们提出一种基于梯度下降的算法，用于联合学习先进的神经网络和层次化索引，从而消除因在线训练和item漂移导致的神经网络学习与索引学习之间的不一致性。
- 我们证明了所提出方法的有效性，并在离线评估和在线 A/B 实验中展示了显著的性能提升。

## 2 模块化神经网络（MoNN）

本节介绍一种称为模块化神经网络（Modular Neural Network, MoNN）的新模型架构，并展示 MoNN 是一种更强大的模型架构，能够在不同的基础设施约束下灵活运行，带来更好的性能。

模块化神经网络（MoNN）增强了对复杂用户-item交互的学习，超越了单一的点积，同时保持高效率。如图 1 所示，它通过模块化设计实现这一目标，包含三个独立的模块：用户表示模块（用户塔）、item表示模块（item塔）以及用户与item之间的交互模块（交互塔）。MoNN 提供高度的灵活性，能够控制每个塔的复杂度，以有效平衡表达力和计算成本。

**用户塔（User Tower）。** 用户塔处理用户特征以生成固定大小的用户嵌入。这些特征可以是稠密特征（例如用户的点击次数）或稀疏特征（例如用户观看过的视频）。稀疏特征输入到嵌入表中，所有特征嵌入被拼接后输入到塔网络。用户塔输出大小为 `num_embed_user * dim_user` 的嵌入，在查询时计算。由于用户塔只需计算一次，并在大量item间共享，因此它可以具有非常高的复杂度。

**item塔（Item Tower）。** item塔镜像用户塔，处理item的稠密特征（例如item的历史点击率）和稀疏特征（例如item的内容）。稀疏特征输入到嵌入表中，所有特征输出被拼接后输入到塔网络，生成大小为 `num_embed_item * dim_item` 的嵌入。

**塔网络（Tower Network）。** 用户塔、item塔和交互塔的神经网络采用与 EDCN [5] 类似的堆叠架构。它包括一个用于捕捉隐式特征交互的 MLP 层和一个用于显式成对特征交互的 Attention FM [40] 层。

**训练设置（Training Setup）。** MoNN 模型在大规模训练数据集上训练，使用点击和转化作为标签，曝光（非点击或转化）作为负样本，以及额外的无标签数据进行半监督学习以消除偏差。模型使用 O(1000) 量级的特征作为输入。模型针对多个任务进行优化，例如点击任务和转化任务。然后使用多任务交叉熵损失训练模型：

$$
$L_{sup}$ = -\frac{1}{S} \su$m_{i=1}$^{S} \su$m_{t=1}$^{T} w_t ($y_{ti}$ \log(\hat{y}_{ti}) + (1 - $y_{ti}$)(\log(1 - \hat{y}_{ti}))) \qquad (1)
$$

$$
$L_{unsup}$ = -\frac{1}{S} \su$m_{i=1}$^{S} \su$m_{t=1}$^{T} distil(\hat{y}_{ti}, $y_{ti}$^{model}) \qquad (2)
$$

$$
L = $L_{sup}$ + $L_{unsup}$ \qquad (3)
$$

其中 $w_t$ 是任务 $t$ 的权重， $t = 1,2,...T$ ，表示其在最终损失中的重要性。 $y_{ti} \in \{0,1\}$ 是样本 $i$ 在任务 $t$ 上的标签。 $\hat{y}_{ti}$ 是模型对样本 $i$ 在任务 $t$ 上的预测值， $y_{ti}^{model}$ 是由 MoNN 模型或更强的模型生成的软标签。 $S$ 是样本数量。

**交互塔（Interaction Tower）。** 交互塔以 <用户, item> 交互特征（稠密和稀疏）为输入。其架构与用户塔和item塔类似，为每对 <用户, item> 产生一个大小为 `num_embed_interaction * dim_interaction` 的嵌入。为了最小化 <用户, item> 交互特征的计算成本，我们引入了基于倒排索引的交互特征（Inverted Index Based Interaction Features, I2IF），其中使用倒排索引对item信息进行索引，用户信息作为查询来执行高效的交叉计算。在图 2 所示的示例中，item类别特征（每个item的）保存在倒排索引中，用户特征（用户交互过的item类别）作为查询传递，以产生稠密的交互特征。I2IF 可以类似地用于生成稀疏交互特征。

**总架构（OverArch）。** 位于三个底层塔（用户塔、item塔和交互塔）之上的总架构组件以它们的输出为输入。它采用 DHEN 风格 [44] 的模型架构来生成 logit。值得注意的是，与传统的双塔模型不同，`dim_item`、`dim_user` 和 `dim_interaction` 可以不同，且 `num_embed_user`、`num_embed_interaction` 和 `num_embed_item` 可以大于 1。

**在线训练（Online Training）。** 检索模型通过在线学习进行训练，数据以连续流的方式输入，无需物化，并且会定期（例如每隔几分钟）创建新的模型快照用于在线服务。这种在线训练和频繁的快照发布使得 MoNN 能够持续提供最新的预测，并及时追赶生态系统中item漂移的变化。

---

## 3 层次化结构化神经网络（HSNN）

虽然 MoNN 展示了捕捉复杂用户和item交互的良好能力，但其潜力受制于item的庞大规模。为克服这一限制，我们提出 HSNN，它能够以相对于语料库大小的次线性成本扩展 MoNN。以下部分详细阐述 HSNN 如何实现高效扩展。

**实体表示的粒度至关重要（Granularity of Entity Representation Matters）。** 在检索阶段部署最先进的神经网络的一个基本挑战是item的巨大量级。推荐系统中典型的深度神经网络处理的item数量少几千到几万倍，导致：

- 先进神经网络的计算成本过高。
- 使用大量特征（尤其是交叉 <用户, item> 特征）时 I/O 访问和计算成本过高。

然而，如果我们将item的基数降低 $K$ 倍（例如，构建一个item索引，每个索引节点包含 $K$ 个item），理论上可以在不增加服务成本的情况下，在检索中使用 $K$ 倍复杂度的模型。

**启发示例（Motivating Example）。** 考虑一个包含 $C$ 个索引节点的item索引，每个节点包含 $K$ 个item。假设同一索引节点内的item主题相似，我们可以将候选选择的复杂度从 $O(K * C)$ 降低到 $O(K + T)$ ，其中 $T$ 是要返回的item数量。这使得可以采用更先进的神经网络（模型 1）来学习用户与item之间的复杂交互（在索引粒度级别）。相比之下，双塔模型（模型 2）仅限于通过单一的点积学习基本交互，缺乏任何 <用户, item> 交互特征。模型 1 和模型 2 的比较如图 3 所示。

虽然两种模型各有利弊，但模型 2 在item级别捕捉细粒度信号，而模型 1 利用先进的神经网络和更大的信号量在item索引级别运作。理想情况下，检索模型范式应结合两种方法的优势。

**HSNN 简介（HSNN Introduction）。** 为发挥图 3 中两种模型的优势，我们提出 HSNN，在层次化item索引上学习多个不同粒度（即不同item表示粒度）的混合模型。通过精心设计item索引的粒度（即多少个item共享一个索引节点），HSNN 能够在检索中部署先进的 ML 架构，并利用跨实体的计算共享。索引的粒度越粗，实现的计算共享越多。借助层次化item索引，HSNN 可采用不同复杂度的混合 ML 模型来联合优化个性化能力。

HSNN 包含两个主要组件：(1) 层次化索引；(2) 索引层次结构中每一层的神经网络设计。第 3.1 节详细阐述模型架构设计，第 3.2 节深入介绍层次化索引的学习。

### 3.1 模型架构

HSNN 是在item层次结构上运行的模块化神经网络（MoNN）的混合体。虽然 HSNN 可以应用于 $N$ 层层次结构，但图 4 展示了一个 3 层 HSNN 模型的架构。在该架构中，HSNN 在具有三层的层次化索引上运作：L1 索引、L2 索引和 L3 索引。L1 索引在最粗的索引粒度上运作，能够利用具有最高复杂度（通过计算共享）和广泛交互特征（<用户, L1 索引节点>）的 MoNN 模型架构。另一方面，L $N$ 索引（本例中 N=3）在item粒度上运作，并利用具有最低复杂度和最少组 <用户, item> 交互特征的 MoNN 模型架构。三个 MoNN 模块通过集成层（ensemble layer）组合。这种方法允许最终预测利用多个不同模型复杂度的 MoNN，消费不同粒度的特征，从而实现更准确的预测。

**特征（Features）。** MoNN Small 在单个item级别处理特征，利用用户特征、item特征和 <用户, item> 交互特征。相比之下，MoNN Medium 和 MoNN Large 在更粗的粒度（分别为 L2 索引和 L1 索引）上运作，消费用户特征、索引节点特征和 <用户, 索引节点> 交互特征，其中索引节点由一个代表性item代替进行特征计算。关于代表性item选择的更多信息见算法 1。

**损失函数（Loss Function）。** 有 $N$ 个监督损失，层次结构中每一层一个，其中包含 1 个 <用户, item> 监督损失和 (N-1) 个 <用户, 索引节点> 监督损失。每个损失还被校准，以确保模型不会低估或高估预测。如 MoNN 部分所述，使用多任务交叉熵损失来优化模型：

$$
$L_{sup}$ = -\frac{1}{S} \su$m_{i=1}$^{S} \su$m_{j=1}$^{N} \su$m_{t=1}$^{T} w_t ($y_{ti}$ \log(\hat{y}_{ti}^{L_j}) + (1 - $y_{ti}$)(\log(1 - \hat{y}_{ti}^{L_j}))) \qquad (4)
$$

其中 $w_t$ 是任务 $t$ 的权重， $t = 1,2,...T$ ，表示其在最终损失中的重要性。 $y_{ti} \in \{0,1\}$ 是样本 $i$ 在任务 $t$ 上的标签。 $\hat{y}_{ti}^{L_j}$ 是模型对样本 $i$ 在任务 $t$ 上第 $L_j$ 层的预测值。 $S$ 是样本数量， $N$ 是 HSNN 的层数。

### 3.2 学习层次化索引

层次化索引可以通过三种方法构建：

(a) **利用现有数据结构（Exploiting existing data structures）：** 利用现有数据结构中 Item -> Item Creator -> Item Creator Group 的自然层次结构。

(b) **聚类item嵌入（Clustering item embeddings）：** 应用传统聚类方法（如 k-means）基于item嵌入获得item-索引分配。

(c) **联合优化（Joint optimization）：** 联合优化索引分配和 MoNN item模块。

虽然方法 (a) 和 (b) 通过独立学习层次化索引和检索模型已被证明是有效的，但我们假设方法 (c) 通过协同训练各个组件能够带来额外的收益。因此，我们提出联合优化的索引和 MoNN（Jointly Optimized indexing and MoNN, JOIM），它可以产生更好的结果，并泛化到学习用户与item之间的复杂交互。在以下部分中，我们介绍一种基于梯度下降的算法，该算法与 MoNN 模型训练一起学习索引分配和表示。这将涵盖用于学习单层索引的学习索引（Learning To Index, LTI）、通过残差学习学习层次化索引，以及层次化索引与 MoNN 的联合学习。

#### 3.2.1 学习索引（LTI）

我们提出一种新的学习索引（Learning-to-Index, LTI）算法，具有三个关键优势：

- **无缝集成（Seamless integration）：** 我们的基于梯度下降的方法可以轻松集成到现有的训练基础设施中。
- **轻量且通用（Lightweight and versatile）：** 该算法设计为计算高效，能够在不对训练 QPS 产生显著影响的情况下与各种模型架构集成。
- **梯度友好（Gradient-friendly）：** 通过克服索引分配中的 argmax 算子，我们的算法允许梯度通过，促进更有效的优化。

**算法。** LTI 算法将来自 MoNN 模型的item嵌入作为输入，通过最小化它们之间的 L2 距离来学习粗粒度的索引节点嵌入。为解决 argmax 算子（用于识别离item最近的索引节点）的问题，它采用了一种基于注意力的方法：以item嵌入作为查询，以可学习的索引节点嵌入作为键和值，并基于 L2 距离的注意力机制来计算该item的索引嵌入。LTI 算法在算法 1 中详细描述。该方法基于归一化的 L2 距离计算item到每个索引节点的软分配（第 6 行）。它使用该分配通过对索引节点嵌入进行加权求和来计算item在索引中的表示（第 7 行）。软分配学习的一个关键优势是允许item以不同程度的成员身份属于多个索引节点，从而捕捉数据中的不确定性。

**算法 1：学习索引（LTI）**

```
输入: MoNN模型, 每层节点数(K)
输出: mapping_item_index m 和 representative_item_index r

1: 初始化索引嵌入 index {c_k}_{k=1}^K
2: for batch中的每个j do
3:   计算MoNN用户嵌入 u_j 和item嵌入 v_j
4:   distance_item_index d(j,k) = ||v_j - c_k||_2
5:   mapping_item_index m_j = argmin_k d(j,k)
6:   affinity_item_index a_k = exp(-alpha * d(j,k)) / sum_{k'} exp(-alpha * d(j,k^{\prime}))
7:   embedding_item_index c_bar = sum_k a_k * c_k
8:   添加监督 logloss_index(y, <u_j, c_bar>)
9: end for
10: 初始化 representative_item_index
11: for corpusV中的每个itemj do
12:   for k in K do
13:     representative_item_index_k r_k = j 如果 d(j,k) 比现有值更近
14:   end for
15: end for
16: 发布 mapping_item_index m 和 representative_item_index r
```

#### 3.2.2 残差学习（Residual Learning）

受残差量化变分自编码器（RQ-VAE）[24][43] 的启发，一个索引层中的item嵌入与对应索引节点嵌入之间的残差被作为输入传递到下一索引层。

具体地，给定item嵌入 $v$ ， $N$ 级残差学习算法初始化索引节点 $\{c_{nk}\}_{k=1,...,K; n=1,...,N}$ ，并在每个层级 $n$ 递归地量化残差向量 $r_n$ ：

$$
r_1 = v \qquad (5)
$$
$$
r_n = $r_{n-1}$ - \bar{c}_{n-1} \qquad (6)
$$

其中 $\bar{c}_{n-1}$ 是 embedding_item_index（算法 1 中的第 7 行）。在每个层级 $n$ ，量化后的item嵌入计算为 $q_n = \sum_{t=1}^{n-1} \bar{c}_t$ 。重构损失计算为：

$$
reconstruction\_loss = ||q_N - v||^2 \qquad (7)
$$

残差的幅度在进一步向下层移动时逐渐减小。因此，较粗的索引层表达更一般的概念，而细粒度的索引层捕捉更详细的含义。

#### 3.2.3 索引与 MoNN 的联合优化（JOIM）

由于 LTI 是一种用于学习层次化索引的梯度下降方法，它可以自然地与 MoNN 联合训练。以下是使其有效工作的一些关键考虑：

- **梯度流（Gradient Flow）：** 在分离的索引学习中，item索引分配仅由item嵌入决定。然而，联合优化允许索引分配受到 <用户, item> 监督信号的影响。来自该监督信号的梯度通过item塔流向 LTI 模块，影响item节点嵌入和索引分配。
- **交互塔（Interaction Tower）：** 图 4 说明了 MoNN 模块如何在 L1 和 L2 索引粒度上处理交互特征以推导交互嵌入。然而，在训练期间访问索引级别的交互特征具有挑战性。为解决这个问题，我们为每个索引层引入一个专用的交互塔，仅使用用户和item特征作为输入。此外，在索引级和item级交互塔之间应用辅助均方误差损失以帮助收敛。

#### 3.2.4 训练优化

我们引入了若干额外技术来改善训练稳定性，包括 softmax 温度调度器、平衡索引分布和预热策略。

**Softmax 温度调度器（Softmax Temperature Scheduler）。** HSNN 在服务期间使用代表性item的特征（算法 1 第 13 行）。然而，在训练期间 LTI 算法使用软 <item, 索引节点> 分配。为缓解这种不一致，采用调度器，通过逐渐增加温度（alpha），从初始阶段的软分配过渡到后期的硬分配。较小的 alpha 值产生平衡的item-索引分配分布，而较大的 alpha 值导致偏斜的分布。调度器基于以下函数：

$$
alpha = max\_alpha * \left(\frac{current\_iter}{max\_iters}\right)^{exp} \qquad (8)
$$

**平衡索引分布（Balanced Index Distribution）。** 索引学习常常受到聚类坍缩（cluster collapse）的影响，即模型仅使用有限子集的索引节点。平衡的索引分布对于启用具有高复杂度的神经网络模型至关重要。我们采用 FLOPs 正则化器来解决这个问题。其动机来自 Paria 等人 [31] 的工作，如果所有item被分配到同一个索引节点，或者 <item, 索引节点> 分配分布不平衡，则对模型进行惩罚。引入稀疏性损失以最小化分配均值的平方和。由于这对小批次敏感，将最近 $K$ 个批次的数据聚合在一起，对聚合后的软分配矩阵（ $K$ * batch_size, num_index_nodes）应用 FLOPs 正则化器。

**预热（Warmup）。** 对索引损失权重采用线性预热策略，以逐渐增加学习率。该方法稳定了模型参数，并缓解了训练初期item分配在索引节点之间振荡的问题。

---

## 4 离线消融实验

对 HSNN 进行消融研究以验证关键组件的价值。我们使用两个离线指标——归一化熵（Normalized Entropy, NE）和 Recall@K 来衡量性能。

**归一化熵（NE）。** 归一化熵 [17] 等于每次曝光的平均对数损失除以如果模型预测训练数据集的平均经验 CTR/CVR 时的平均对数损失。值越低，模型预测越好。任何超过 0.05% 的 NE 增益被认为是显著的。

**Recall@K。** Recall@K 衡量所有相关item中，出现在前 $K$ 个中的相关item的比例，其中 $K$ 是为用户生成的推荐数量。0.5% 的召回率增益被认为是显著的。

### 4.1 高级模型架构

本节探讨 MoNN 是否能提高准确性。我们进行了一项离线研究，使用 MoNN 对所有广告进行评分。我们将双塔模型与四种 MoNN 变体（详见第 2 节）进行比较，如表 1 所示。模型的复杂度以浮点运算数（FLOPs）衡量，表示单次推理所需的计算量。所有报告的指标（推理 FLOPs、评估 NE、基础设施成本和召回率）均相对于双塔 XS 模型。MoNN Small、Medium、Large 和 XL 模型在 FLOPs 上展现出逐步递增的数量级。

MoNN Small 相对于双塔模型架构实现了 0.29% 的 NE 增益，突出了 MoNN 中交互塔、总架构和 <用户, item> 交互特征所提供的复杂交互学习的好处。虽然大规模 MoNN 变体展示了显著的 NE 增益，但其指数级的基础设施计算成本使其在检索阶段不实用。

### 4.2 层次化结构化神经网络（HSNN）

表 1 展示了 HSNN 相对于双塔架构的巨大收益（高达 1.46% NE 增益和 10% 召回率提升），而基础设施成本比原始 MoNN 低几个数量级，在 Meta 广告中产生了高投资回报率。表中显示的相对基础设施成本衡量了在生产中以相同吞吐量和 QPS 服务模型所需的硬件容量。HSNN 通过有效利用层次化索引和更复杂的 MoNN 架构，明显脱颖而出。在所有 HSNN 实验中，MoNN 模型使用 LTI 算法与层次化索引联合优化。

**表 1：HSNN 使得像 MoNN XL 这样的更复杂模型架构能够进入检索阶段。每次规模扩展都带来数量级的增益。I1、I2、V 的量级分别为 O(1,000)、O(100,000)、O(10,000,000)。任何超过 0.05% 的 NE 增益和超过 0.5% 的召回率增益被认为是显著的。**

| 模型架构 | FLOPs | Eval NE ( $\downarrow$ ) | 理论成本 | 基础设施成本 ( $\downarrow$ ) | 召回率 ( $\uparrow$ ) |
|---|---|---|---|---|---|
| Two Tower (XS) | 0.25x | baseline | $M_{XS} * V$ | 1x | 0 |
| MoNN Small | 1x | -0.29% | $M_S * V$ | 2.5x | +2.4% |
| MoNN Medium | 50x | -0.70% | $M_M * V$ | 17.3x | +4.2% |
| MoNN Large | 30,000x | -1.70% | $M_L * V$ | 24.6x | +9.4% |
| MoNN XL | 200,000x | -2.20% | $M_{XL} * V$ | 64x | +12% |
| EBR (Two Tower) | 0.25x | +0.03% | $M_{XS} * I1$ | 0.49x | -0.1% |
| 2层 HSNN (L1: MoNN Small, L2: TTSN) | 1.25x | -0.23% | $M_S * I1 + M_{XS} * V$ | 1.7x | +2.2% |
| 2层 HSNN (L1: MoNN Medium, L2: MoNN Small) | 51x | -0.47% | $M_M * I1 + M_S * V$ | 3.3x | +3.6% |
| 2层 HSNN (L1: MoNN Large, L2: MoNN Small) | 30,001x | -0.97% | $M_L * I1 + M_S * V$ | 3.9x | +6% |
| 3层 HSNN (L1: MoNN XL, L2: MoNN Large, L3: MoNN Small) | 230,001x | -1.46% | $M_{XL} * I1 + M_L * I2 + M_S * V$ | 4.5x | +10% |

对于理论分析，我们基于以下参数展示 HSNN 模型的服务成本：I1（L1 索引中的节点数）、I2（L2 索引中的节点数）、V（语料库中的item数）， $M_{XS}$ 表示服务双塔模型的成本， $M_S$ 表示服务 MoNN Small 的成本， $M_M$ 表示服务 MoNN Medium 的成本， $M_L$ 表示服务 MoNN Large 的成本， $M_{XL}$ 表示服务 MoNN XL 的成本。

### 4.3 联合优化

**表 2：对 HSNN（L1: MoNN Small, L2: Two Tower）的消融研究表明联合优化带来 0.15% 的 NE 增益。**

| 模型架构 | Eval NE ( $\downarrow$ ) |
|---|---|
| HSNN w/ SIL | baseline |
| HSNN + JOIM w/ EM | -0.11% |
| HSNN + JOIM w/ LTI | -0.15% |

如表 2 所示，联合优化（JOIM）在相同模型架构下，相对于使用分离索引学习（SIL）的 HSNN，带来了约 0.15% 的 NE 增益。考虑到大规模item语料库，我们使用 k-means 聚类因其良好的可扩展性。LTI 算法还与 EM 变体进行了比较，其中 MoNN 模型架构和层次化索引以交替方式联合学习——即用当前的层次化索引训练 MoNN 模型直至收敛，然后使用收敛的 MoNN 模型更新层次化索引结构。LTI 展现出比 EM 变体更好的性能。需要注意的是，JOIM 的基础设施成本与 HSNN 相同。在所有实验中，模型复杂度和节点数量保持不变以进行比较。

**表 3：各种 LTI 组件的消融研究。**

| 消融研究 | Eval NE ( $\downarrow$ ) |
|---|---|
| HSNN (MoNN Small) | baseline |
| - Softmax 温度调度器 | +0.1% |
| - 预热策略 | +0.03% |
| - 平衡索引分布 | +0.05% |

如表 3 所示，我们对 LTI 算法的各个组件进行了消融研究。NE 增益证明了训练优化策略的有效性。

---

## 5 在线实验

图 5 展示了 HSNN 在 Meta 广告中的生产部署。在查询时，用户塔被执行以生成用户嵌入，而item塔异步运行以索引item嵌入。<用户, item> 交互特征使用 I2IF 框架实时生成，输入到交互塔以实时产生交互嵌入。所有嵌入随后被传递到总架构和集成层，在查询时计算 logit。

HSNN 框架已在 Meta 广告检索系统中广泛部署超过 2 年。在部署 HSNN 之前，我们已将 MoNN Small 架构发布到生产环境中，因其基础设施成本相对较低。如表 4 所示，在线 A/B 测试表明 HSNN 带来了 2.57% 的在线广告指标增益（0.1% 的增益被认为是统计显著的）。

**表 4：HSNN 在 Meta 广告生产中的在线性能。任何超过 0.1% 的在线 Topline 指标增益被认为是显著的。**

| 模型架构 | 在线 Topline 指标 ( $\uparrow$ ) |
|---|---|
| Two Tower | -0.21% |
| MoNN Small | baseline |
| HSNN + SIL (L1: MoNN Medium, L2: MoNN Small) | +1.06% |
| HSNN + JOIM (L1: MoNN Medium, L2: MoNN Small) | +1.22% |
| HSNN + JOIM (L1: MoNN Large, L2: MoNN Small) | +2.57% |

---

## 6 相关工作

**基于嵌入的检索（Embedding Based Retrieval）。** 基于嵌入的检索（EBR）已成功应用于搜索和推荐系统的检索中 [25, 32]。[20] 将这一概念扩展，结合文本、用户、上下文和社交图信息到统一的嵌入中，使得能够检索既与查询相关又为用户个性化的文档。在系统方面，许多机构已经开发或部署了近似最近邻（ANN）算法，这些算法能够在次线性时间内识别给定查询嵌入的前 k 个候选对象 [8, 9, 18]。高效的最大内积搜索（MIPS）或 ANN 算法包括基于树的方法 [19, 30]、局部敏感哈希（LSH）[33, 34]、乘积量化（PQ）[13, 22]、层次化可导航小世界图（HNSW）[27] 等。另一个研究方向关注在离散潜空间中编码向量。向量量化变分自编码器（VQ-VAE）[37] 提出了一种简单而强大的生成模型来学习离散表示。HashRec [23] 和 Merged-Averaged Classifiers via Hashing [29] 采用多索引哈希函数在大规模推荐系统中编码item。层次化量化方法如残差量化变分自编码器（RQ-VAE）[43] 和 Tree-VAE [28] 也被用于学习向量的树结构。然而，这些系统通常假设一个稳定的item词汇表，未考虑嵌入分布漂移。

**ANN 的聚类方法（Clustering Methods for ANN）。** 聚类算法可以广泛分为层次化方法和划分方法。凝聚聚类 [41] 是一种层次化方法，从大量小簇开始，逐步合并它们。在划分技术中，K-means [26] 是最著名的，旨在最小化数据点与其最近聚类中心之间的平方距离之和。这一类别的其他方法包括期望最大化（EM）[7]、谱聚类 [14] 和基于非负矩阵分解（NMF）[3] 的聚类。

**高级模型架构（Advanced Model Architectures）。** DCN、DHEN 等模型架构促进模型中的早期特征交互。由于服务成本较高，这些架构通常用于排序阶段，评估有限数量的候选对象。最近，生成式检索作为一种新的文档检索范式出现 [1, 4, 35, 36, 39]。这种方法与我们的工作并行，将item视为文档，用户视为查询。生成式检索中的学习编码类似于本文提出的层次化索引中的item节点。

**联合优化（Joint Optimization）。** 与我们的工作最相似的是联合优化系统，其中item（广告）层次结构和大规模检索模型被联合优化。Gao 等人 [12] 提出深度检索（Deep Retrieval, DR），通过端到端神经网络设计将所有item编码到离散潜空间中。深度检索在可以使用的模型架构和特征方面存在限制。Zhu 等人 [46] 提出了一种新颖的基于树的方法，即使使用更具表达力的模型（如深度神经网络），也能提供相对于语料库大小的对数复杂度。在这一研究领域中，基于树的方法 [42, 45, 47] 是研究热点。这些方法将每个item映射到树结构中的一个叶节点，并联合学习树结构和模型参数的目标函数。

---

## 7 结论与下一步工作

在本文中，我们介绍了一个已部署的检索模型——层次化结构化神经网络（HSNN）。HSNN 是一个强大的框架，能够使先进的神经网络以高准确性和高效率学习用户与item特征之间的复杂交互。在 Meta，HSNN 已成功部署到最大的广告推荐系统之一。

虽然层次化索引在item方面已被广泛探索，但如何为用户采用类似策略以进一步提高性能仍然是一个重要课题。此外，随着生成式检索的兴起，将生成式组件融入 HSNN 以进一步增强其性能是一个有前景的方向。

---

## 8 致谢

作者感谢 Le Fang、Tushar Tiwari、Wei Lu、Jason Liu、Trevor Waite、Shu Yan、Alexander Petrov、Dheevatsa Mudigere、Benny Chen、GP Musumeci、Yiping Han、Bo Long、Wenlin Chen、Santanu Kolay 以及其他为本文做出贡献、支持和合作的人。

---

## 附录

### A.1 MoNN 模型配置

除表 1 中显示的四种不同复杂度的 MoNN 模型的 FLOPs 和理论成本外，表 5 还提供了关于模型配置的额外信息，如用户塔、item塔和交互塔的特征数量和超参数。

**表 5：MoNN 模型复杂度。用户 (u)、item (i) 和交互 (ux) 为缩写。**

| 模型 | 特征数量 | 塔参数 |
|---|---|---|
| Two Tower Tiny | user: O(100), item: O(100), interaction: 0 | num_embed_u=1, dim_u=40; num_embed_i=1, dim_i=40 |
| MoNN Small | user: O(100), item: O(100), interaction: O(10) | num_embed_u=8, dim_u=40; num_embed_i=1, dim_i=40; num_embed_ui=1, dim_ui=40 |
| MoNN Medium | user: O(100), item: O(100), interaction: O(10) | num_embed_u=10, dim_u=48; num_embed_i=4, dim_i=48; num_embed_ui=1, dim_ui=48 |
| MoNN Large | user: O(1000), item: O(100), interaction: O(100) | num_embed_u=30, dim_u=48; num_embed_i=8, dim_i=48; num_embed_ui=1, dim_ui=48 |
| MoNN XL | user: O(1000), item: O(1000), interaction: O(100) | num_embed_u=80, dim_u=96; num_embed_i=4, dim_i=96; num_embed_ui=1, dim_ui=96 |

### A.2 HSNN 服务算法

HSNN 服务包括两部分：MoNN 模型的服务和针对层次化索引的推理。

MoNN 模型在服务时被分为 4 部分——用户塔、item塔、交互塔和总架构模块。

- **用户塔**接收用户特征并返回用户嵌入。这在查询时发生。
- **item塔**接收item特征并返回item嵌入和item到索引的映射。item索引是异步构建和维护的，新item被立即索引，旧item从索引中删除。
- **交互塔和总架构模型**在广告检索系统中于查询时计算交互嵌入和 logit。

HSNN 服务算法在算法 2 中提供。HSNN 服务是逐层进行的。对于每一层，选择item的代表性索引节点，并利用其特征作为对应 MoNN 模型的输入。MoNN 模型的输出被传递到下一层 MoNN 模型的线性集成层，以获取 logit。每一层仅选择 top-k 个节点。

**算法 2：HSNN 服务**

```
输入: MoNN模型, item-索引映射 m
输出: Top k 个item

1: for 每一层 i in N do
2:   识别每个item的索引节点
3:   使用索引节点中代表性item的特征，评估 MoNN 模型的 f(user, index-node)
4:   将 MoNN 模型输出与之前 MoNN 模型的输出一起传递给集成层，得到 logit
5:   从该层选择 top-k 个节点作为下一层的候选
6: end for
7: 发布 top-k 个item到排序阶段
```

---

## 参考文献

[1] Michele Bevilacqua, Giuseppe Ottaviano, Patrick Lewis, Wentau Yih, Sebastian Riedel, and Fabio Petroni. 2022. Autoregressive Search Engines: Generating Substrings as Document Identifiers. arXiv:cs.CL/2204.10628

[2] Jane Bromley, Isabelle Guyon, Yann LeCun, Eduard Säckinger, and Roopak Shah. 1993. Signature Verification Using a "Siamese" Time Delay Neural Network. In *Proceedings of the 6th International Conference on Neural Information Processing Systems* (Denver, Colorado) *(NIPS'93)*. Morgan Kaufmann Publishers Inc., San Francisco, CA, USA, 737–744.

[3] Deng Cai, Xiaofei He, Xuanhui Wang, Hujun Bao, and Jiawei Han. 2009. Locality Preserving Nonnegative Matrix Factorization. In *Proceedings of the 21st International Joint Conference on Artificial Intelligence* (Pasadena, California, USA) *(IJCAI'09)*. Morgan Kaufmann Publishers Inc., San Francisco, CA, USA, 1010–1015.

[4] Nicola De Cao, Gautier Izacard, Sebastian Riedel, and Fabio Petroni. 2021. Autoregressive Entity Retrieval. arXiv:cs.CL/2010.00904

[5] Bo Chen, Yichao Wang, Zhirong Liu, Ruiming Tang, Wei Guo, Hongkun Zheng, Weiwei Yao, Muyu Zhang, and Xiuqiang He. 2021. Enhancing Explicit and Implicit Feature Interactions via Information Sharing for Parallel Deep CTR Models. In *Proceedings of the 30th ACM International Conference on Information & Knowledge Management* (Virtual Event, Queensland, Australia) *(CIKM'21)*. Association for Computing Machinery, New York, NY, USA, 3757–3766.

[6] Paul Covington, Jay Adams, and Emre Sargin. 2016. Deep Neural Networks for YouTube Recommendations. In *Proceedings of the 10th ACM Conference on Recommender Systems* (Boston, Massachusetts, USA) *(RecSys'16)*. Association for Computing Machinery, New York, NY, USA, 191–198.

[7] A. P. Dempster, N. M. Laird, and D. B. Rubin. 2018. Maximum Likelihood from Incomplete Data Via the EM Algorithm. *Journal of the Royal Statistical Society: Series B (Methodological)* 39, 1 (12 2018), 1–22.

[8] Ishita Doshi, Dhritiman Das, Ashish Bhutani, Rajeev Kumar, Rushi Bhatt, and Niranjan Balasubramanian. 2020. LANNS: A Web-Scale Approximate Nearest Neighbor Lookup System. arXiv:cs.IR/2010.09426

[9] Pinterest Engineering. 2021. Manas HNSW Realtime: Powering Realtime Embedding-Based Retrieval.

[10] Luke Gallagher, Ruey-Cheng Chen, Roi Blanco, and J. Shane Culpepper. 2019. Joint Optimization of Cascade Ranking Models. In *Proceedings of the Twelfth ACM International Conference on Web Search and Data Mining* (Melbourne VIC, Australia) *(WSDM'19)*. Association for Computing Machinery, New York, NY, USA, 15–23.

[11] Weihao Gao, Xiangjun Fan, Chong Wang, Jiankai Sun, Kai Jia, Wenzhi Xiao, Ruofan Ding, Xingyan Bin, Hui Yang, and Xiaobing Liu. 2021. Deep Retrieval: Learning a Retrievable Structure for Large-Scale Recommendations. arXiv:cs.IR/2007.07203

[12] Weihao Gao, Xiangjun Fan, Chong Wang, Jiankai Sun, Kai Jia, Wenzhi Xiao, Ruofan Ding, Xingyan Bin, Hui Yang, and Xiaobing Liu. 2021. Deep Retrieval: Learning a Retrievable Structure for Large-Scale Recommendations. arXiv:cs.IR/2007.07203

[13] Tiezheng Ge, Kaiming He, Qifa Ke, and Jian Sun. 2014. Optimized Product Quantization. *IEEE Transactions on Pattern Analysis and Machine Intelligence* 36, 4 (2014), 744–755.

[14] Cuimei Guo, Sheng Zheng, Yaocheng Xie, and Wei Hao. 2012. A survey on spectral clustering. In *World Automation Congress 2012*. 53–56.

[15] Ruiqi Guo, Philip Sun, Erik Lindgren, Quan Geng, David Simcha, Felix Chern, and Sanjiv Kumar. 2020. Accelerating Large-Scale Inference with Anisotropic Vector Quantization. arXiv:cs.LG/1908.10396

[16] Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu, and Tat-Seng Chua. 2017. Neural Collaborative Filtering. arXiv:cs.IR/1708.05031

[17] Xinran He, Junfeng Pan, Ou Jin, Tianbing Xu, Bo Liu, Tao Xu, Yanxin Shi, Antoine Atallah, Ralf Herbrich, Stuart Bowers, and Joaquin Quiñonero Candela. 2014. Practical Lessons from Predicting Clicks on Ads at Facebook. In *Proceedings of the Eighth International Workshop on Data Mining for Online Advertising* (New York, NY, USA) *(ADKDD'14)*. Association for Computing Machinery, New York, NY, USA, 1–9.

[18] Jeff Johnson, Hervé Jegou, Matthijs Douze. 2017. Faiss: A library for efficient similarity search.

[19] Michael E. Houle and Michael Nett. 2015. Rank-Based Similarity Search: Reducing the Dimensional Dependence. *IEEE Transactions on Pattern Analysis and Machine Intelligence* 37, 1 (2015), 136–150.

[20] Jui-Ting Huang, Ashish Sharma, Shuying Sun, Li Xia, David Zhang, Philip Pronin, Janani Padmanabhan, Giuseppe Ottaviano, and Linjun Yang. 2020. Embedding-based Retrieval in Facebook Search.

[21] Jeff Johnson, Matthijs Douze, and Hervé Jégou. 2017. Billion-scale similarity search with GPUs. arXiv:cs.CV/1702.08734

[22] Herve Jégou, Matthijs Douze, and Cordelia Schmid. 2011. Product Quantization for Nearest Neighbor Search. *IEEE Transactions on Pattern Analysis and Machine Intelligence* 33, 1 (2011), 117–128.

[23] Wang-Cheng Kang and Julian McAuley. 2019. Candidate Generation with Binary Codes for Large-Scale Top-N Recommendation. arXiv:cs.IR/1909.05475

[24] Doyup Lee, Chiheon Kim, Saehoon Kim, Minsu Cho, and Wook-Shin Han. 2022. Autoregressive Image Generation using Residual Quantization. arXiv:cs.CV/2203.01941

[25] Yiqun Liu, Kaushik Rangadurai, Yunzhong He, Siddarth Malreddy, Xunlong Gui, Xiaoyi Liu, and Fedor Borisyuk. 2021. Que2Search: Fast and Accurate Query and Document Understanding for Search at Facebook.

[26] James MacQueen et al. 1967. Some methods for classification and analysis of multivariate observations. 281–297.

[27] Yu. A. Malkov and D. A. Yashunin. 2018. Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs. arXiv:cs.DS/1603.09320

[28] Laura Manduchi, Moritz Vandenhirtz, Alain Ryser, and Julia Vogt. 2023. Tree Variational Autoencoders. arXiv:cs.LG/2306.08984

[29] Tharun Medini, Qixuan Huang, Yiqiu Wang, Vijai Mohan, and Anshumali Shrivastava. 2019. Extreme Classification in Log Memory using Count-Min Sketch: A Case Study of Amazon Search with 50M Products. arXiv:cs.LG/1910.13830

[30] Marius Muja and David G. Lowe. 2014. Scalable Nearest Neighbor Algorithms for High Dimensional Data. *IEEE Transactions on Pattern Analysis and Machine Intelligence* 36, 11 (2014), 2227–2240.

[31] Biswajit Paria, Chih-Kuan Yeh, Ian E. H. Yen, Ning Xu, Pradeep Ravikumar, and Barnabás Póczos. 2020. Minimizing FLOPs to Learn Efficient Sparse Representations. arXiv:cs.LG/2004.05665

[32] Kaushik Rangadurai, Yiqun Liu, Siddarth Malreddy, Xiaoyi Liu, Piyush Maheshwari, Vishwanath Sangale, and Fedor Borisyuk. 2022. NxtPost: User to Post Recommendations in Facebook Groups. arXiv:cs.LG/2202.03645

[33] Anshumali Shrivastava and Ping Li. 2014. Asymmetric LSH (ALSH) for Sublinear Time Maximum Inner Product Search (MIPS). arXiv:stat.ML/1405.5869

[34] Ryan Spring and Anshumali Shrivastava. 2017. A New Unbiased and Efficient Class of LSH-Based Samplers and Estimators for Partition Function Computation in Log-Linear Models. arXiv:stat.ML/1703.05160

[35] Weiwei Sun, Lingyong Yan, Zheng Chen, Shuaiqiang Wang, Haichao Zhu, Pengjie Ren, Zhumin Chen, Dawei Yin, Maarten de Rijke, and Zhaochun Ren. 2023. Learning to Tokenize for Generative Retrieval. arXiv:cs.IR/2304.04171

[36] Yi Tay, Vinh Q. Tran, Mostafa Dehghani, Jianmo Ni, Dara Bahri, Harsh Mehta, Zhen Qin, Kai Hui, Zhe Zhao, Jai Gupta, Tal Schuster, William W. Cohen, and Donald Metzler. 2022. Transformer Memory as a Differentiable Search Index. arXiv:cs.CL/2202.06991

[37] Aaron van den Oord, Oriol Vinyals, and Koray Kavukcuoglu. 2018. Neural Discrete Representation Learning. arXiv:cs.LG/1711.00937

[38] Ruoxi Wang, Bin Fu, Gang Fu, and Mingliang Wang. 2017. Deep & Cross Network for Ad Click Predictions. arXiv:cs.LG/1708.05123

[39] Yujing Wang, Yingyan Hou, Haonan Wang, Ziming Miao, Shibin Wu, Hao Sun, Qi Chen, Yuqing Xia, Chengmin Chi, Guoshuai Zhao, Zheng Liu, Xing Xie, Hao Allen Sun, Weiwei Deng, Qi Zhang, and Mao Yang. 2023. A Neural Corpus Indexer for Document Retrieval. arXiv:cs.IR/2206.02743

[40] Jun Xiao, Hao Ye, Xiangnan He, Hanwang Zhang, Fei Wu, and Tat-Seng Chua. 2017. Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks. arXiv:cs.LG/1708.04617

[41] Jianwei Yang, Devi Parikh, and Dhruv Batra. 2016. Joint Unsupervised Learning of Deep Representations and Image Clusters. arXiv:cs.CV/1604.03628

[42] Ronghui You, Zihan Zhang, Ziye Wang, Suyang Dai, Hiroshi Mamitsuka, and Shanfeng Zhu. 2019. AttentionXML: Label Tree-based Attention-Aware Deep Model for High-Performance Extreme Multi-Label Text Classification. arXiv:cs.CL/1811.01727

[43] Neil Zeghidour, Alejandro Luebs, Ahmed Omran, Jan Skoglund, and Marco Tagliasacchi. 2021. SoundStream: An End-to-End Neural Audio Codec. arXiv:cs.SD/2107.03312

[44] Buyun Zhang, Liang Luo, Xi Liu, Jay Li, Zeliang Chen, Weilin Zhang, Xiaohan Wei, Yuchen Hao, Michael Tsang, Wenjun Wang, Yang Liu, Huayu Li, Yasmine Badr, Jongsoo Park, Jiyan Yang, Dheevatsa Mudigere, and Ellie Wen. 2022. DHEN: A Deep and Hierarchical Ensemble Network for Large-Scale Click-Through Rate Prediction. arXiv:cs.IR/2203.11014

[45] Han Zhu, Daqing Chang, Ziru Xu, Pengye Zhang, Xiang Li, Jie He, Han Li, Jian Xu, and Kun Gai. 2019. Joint Optimization of Tree-based Index and Deep Model for Recommender Systems. arXiv:cs.IR/1902.07565

[46] Han Zhu, Xiang Li, Pengye Zhang, Guozheng Li, Jie He, Han Li, and Kun Gai. 2018. Learning Tree-based Deep Model for Recommender Systems.

[47] Jingwei Zhuo, Ziru Xu, Wei Dai, Han Zhu, Han Li, Jian Xu, and Kun Gai. 2020. Learning Optimal Tree Models Under Beam Search. arXiv:stat.ML/2006.15408
