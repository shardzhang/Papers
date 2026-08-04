# SIM：基于搜索的用户兴趣建模——利用长期序列行为数据进行点击率预测

> Pi Qi, Xiaoqiang Zhu, Guorui Zhou, Yujing Zhang, Zhe Wang, Lejian Ren, Ying Fan, and Kun Gai* | 阿里巴巴集团，北京，中国

本文介绍了阿里巴巴提出的基于搜索的兴趣模型（SIM，Search-based Interest Model），通过两级级联搜索单元对超长用户行为序列进行建模。核心内容：

- **通用搜索单元（GSU，General Search Unit）**：以候选item为查询，从原始长序列中检索相关的子行为序列
- **精确搜索单元（ESU，Exact Search Unit）**：建模候选item与子行为序列之间的精确关系
- **级联范式**：兼顾可扩展性与准确性，支持最长54000的序列长度

关键发现：SIM部署于阿里巴巴展示广告系统，带来7.1%的点击率（CTR）提升和4.4%的RPM提升，将序列建模长度推至54000，是此前最先进水平的54倍。

---

## 摘要

丰富的用户行为数据已被证明对点击率预测任务具有重要价值，尤其是在推荐系统和在线广告等工业应用中。工业界和学术界都高度关注这一课题，并提出了不同的方法来对长序列用户行为数据进行建模。其中，阿里巴巴提出的基于记忆网络的模型MIMN[8]（Multi-channel user Interest Memory Network，多通道用户兴趣记忆网络），通过学习算法和服务系统的协同设计，实现了最先进的性能。MIMN是首个能够对长度扩展到1000的序列用户行为数据进行建模的工业解决方案。然而，当用户行为序列长度进一步增加（例如增加10倍或更多）时，MIMN无法精确捕捉给定特定候选item的用户兴趣。这一挑战广泛存在于先前提出的方法中。

在本文中，我们通过设计一种新的建模范式来解决这个问题，我们将其命名为基于搜索的兴趣模型（Search-based Interest Model, SIM）。SIM通过两个级联的搜索单元提取用户兴趣：（i）通用搜索单元（General Search Unit, GSU）从原始且任意长的序列行为数据中进行通用搜索，以候选item作为查询信息，得到与候选item相关的子用户行为序列（Sub user Behavior Sequence, SBS）；（ii）精确搜索单元（Exact Search Unit, ESU）建模候选item与SBS之间的精确关系。这种级联搜索范式使SIM在可扩展性和准确性方面具有更好的对长期序列行为数据进行建模的能力。除了学习算法，我们还将介绍如何在大规模工业系统中实现SIM的实践经验。自2019年以来，SIM已部署在阿里巴巴的展示广告系统中，带来了7.1%的点击率提升和4.4%的RPM（Revenue Per Mille，千次展示收入）提升，这对业务意义重大。SIM目前服务于我们实际系统中的主要流量，其对序列用户行为数据建模的最大长度达到54000，将最先进水平推高了54倍。

## CCS 概念

- **信息系统 $\rightarrow$ 学习排序；信息检索；检索模型与排序；**

## 关键词

Click-Through Rate Prediction; User Interest Modeling; Long Sequential User Behavior Data

## 1 引言

点击率（CTR，Click-Through Rate）预测建模在推荐系统和在线广告等工业应用中扮演着关键角色。由于用户历史行为数据的快速增长，用户兴趣建模——专注于学习用户兴趣的意图表示——已被广泛引入点击率预测模型[2, 8, 19, 20]。然而，受限于实际在线系统中的计算和存储负担[19, 20]，大多数提出的方法只能对长度扩展到数百的序列用户行为数据进行建模。丰富的用户行为数据被证明具有重要价值[8]。例如，在全球领先的电子商务网站淘宝中，23%的用户在过去5个月内点击了超过1000件商品[8, 10]。如何设计一个可行的解决方案来建模长序列用户行为数据一直是一个开放且热门的话题，吸引了来自工业界和学术界的研究人员。

一个研究分支借鉴自然语言处理领域的思路，提出使用记忆网络来建模长序列用户行为数据，并取得了一些突破。阿里巴巴提出的MIMN[8]是典型工作之一，它通过学习算法和服务系统的协同设计实现了最先进性能。MIMN是首个能够对长度扩展到1000的序列用户行为数据进行建模的工业解决方案。具体而言，MIMN将一个用户的不同兴趣增量地嵌入到一个固定大小的记忆矩阵中，该矩阵由每个新行为更新。通过这种方式，用户建模的计算与点击率预测解耦。因此，对于在线服务，延迟不再成为问题，存储成本取决于记忆矩阵的大小，这远小于原始行为序列。长期兴趣建模[10]中也有类似的想法。然而，基于记忆网络的方法建模任意长的序列数据仍然具有挑战性。在实践中，我们发现当用户行为序列长度进一步增加（例如达到10000或更多）时，MIMN无法精确捕捉给定特定候选item的用户兴趣。这是因为将所有用户历史行为编码到一个固定大小的记忆矩阵中会导致记忆单元中包含大量噪声。

另一方面，正如深度兴趣网络（DIN，Deep Interest Network）[20]的先前工作所指出的，用户的兴趣是多样化的，并且面对不同的候选item时会发生变化。DIN的关键思想是从用户行为中搜索有效信息，以建模用户面对不同候选item时的特定兴趣。通过这种方式，我们可以解决将所有用户兴趣编码到固定大小参数中的挑战。DIN确实为使用用户行为数据的点击率建模带来了巨大改进。但DIN的搜索公式在处理我们上述的长序列用户行为数据时，面临不可接受的计算和存储开销。那么，我们能否应用类似的搜索技巧，设计一种更有效的方式从长序列用户行为数据中提取知识呢？

在本文中，我们通过设计一种新的建模范式来应对这一挑战，我们将其命名为基于搜索的兴趣模型（SIM）。SIM采用了DIN[20]的思想，仅捕捉与特定候选item相关的用户兴趣。在SIM中，用户兴趣通过两个级联的搜索单元提取：（i）通用搜索单元（GSU）从原始且任意长的序列行为数据中进行通用搜索，以候选item作为查询信息，得到与候选item相关的子用户行为序列（SBS）。为了满足严格的延迟和计算资源限制，GSU中使用了通用但有效的方法。根据我们的经验，SBS的长度可以缩减到数百，并且原始长序列行为数据中的大部分噪声信息可以被过滤掉。（ii）精确搜索单元（ESU）建模候选item与SBS之间的精确关系。在这里，我们可以轻松应用DIN[20]或深度兴趣演化网络（DIEN，Deep Interest Evolution Network）[19]提出的类似方法。

本文的主要贡献总结如下：

*   我们提出了一种新的范式SIM，用于对长序列用户行为数据进行建模。级联的两阶段搜索机制的设计使SIM在可扩展性和准确性方面具有更好的对长期序列行为数据进行建模的能力。
*   我们介绍了在大规模工业系统中实现SIM的实践经验。自2019年以来，SIM已部署在阿里巴巴的展示广告系统中，带来了7.1%的点击率提升和4.4%的RPM提升。目前，SIM正服务于主要流量。
*   我们将长序列用户行为数据建模的最大长度推高至54000，比该任务已发表的最先进工业解决方案MIMN大54倍。

## 2 相关工作

### 2.1 用户兴趣模型

基于深度学习的方法在点击率预测任务中取得了巨大成功[1, 11, 18]。
早期，大多数开创性工作[1, 4, 7, 9, 15]使用深度神经网络捕捉来自不同字段的特征之间的交互，从而使工程师能够摆脱繁琐的特征工程工作。最近，一系列我们称为用户兴趣模型的工作，专注于从历史行为中学习潜在用户兴趣的表示，使用不同的神经网络架构，如卷积神经网络（CNN，Convolutional Neural Network）[14, 17]、循环神经网络（RNN，Recurrent Neural Network）[5, 19]、Transformer[3, 13]和Capsule[6]等。DIN[20]强调用户兴趣是多样化的，并在DIN中引入注意力机制来捕捉用户对不同目标item的多样化兴趣。DIEN[19]指出历史行为之间的时间关系对建模用户漂移的兴趣很重要。DIEN中设计了基于门控循环单元（GRU，Gated Recurrent Unit）并带有辅助损失的兴趣提取层。MIND[6]认为使用单个向量来表示一个用户不足以捕捉用户兴趣的变化性质。MIND引入了胶囊网络和动态路由方法来学习用户兴趣的表示作为多个向量。此外，受自注意力架构在序列到序列学习任务中取得成功的启发，[3]引入了Transformer来建模用户的跨会话和会话内兴趣。

### 2.2 长期用户兴趣

[8]表明，在用户兴趣模型中考虑长期历史行为序列可以显著提高点击率模型的性能。
虽然更长的用户行为序列为用户兴趣建模带来了更多有用的信息，但它极大地加重了在线服务系统的延迟和存储负担，同时包含了用于逐点点击率预测的大量噪声。一系列工作专注于解决长期用户兴趣建模中的挑战，这些工作通常基于极长甚至终身长度的历史行为序列来学习用户兴趣表示。[10]提出了一种分层周期记忆网络，用于对每个用户的序列模式进行个性化记忆的终身序列建模。[16]选择了一种基于注意力的框架来结合用户的长期和短期偏好。他们采用注意力非对称奇异值分解（SVD，Singular Value Decomposition）范式来建模长期兴趣。[8]中提出了一种名为MIMN的基于记忆的架构，将用户的长期兴趣嵌入到固定大小的记忆网络中，以解决用户行为数据的大存储问题。并设计了用户兴趣中心（UIC，User Interest Center）模块来增量记录新的用户行为，以应对延迟限制。但MIMN在记忆网络中放弃了来自目标item的信息，而这已被证明对用户兴趣建模很重要。

## 3 基于搜索的兴趣模型

通过建模用户行为数据进行点击率预测已被证明是有效的。通常，基于注意力的点击率模型，如DIN[20]和DIEN[19]，设计复杂的模型结构并引入注意力机制，通过从用户行为序列中搜索有效知识，以不同候选item作为输入，来捕捉用户的多样化兴趣。但在实际系统中，这些模型只能处理短期行为序列数据，其长度通常小于150。另一方面，长期用户行为数据是有价值的，建模用户的长期兴趣可能为用户带来更多样化的推荐结果。我们似乎陷入了两难境地：在实际系统中，我们无法用有效但复杂的方法处理有价值的终身用户行为数据。

为了解决这一挑战，我们提出了一种新的建模范式，命名为基于搜索的兴趣模型（SIM）。SIM采用两阶段搜索策略，能够以高效的方式处理长用户行为序列。在本节中，我们将首先介绍SIM的总体工作流程，然后详细介绍这两个提出的搜索单元。

### 3.1 总体工作流程

SIM的总体工作流程如图1所示。SIM采用级联的两阶段搜索策略，包含两个相应的单元：通用搜索单元（GSU）和精确搜索单元（ESU）。

在第一阶段，我们利用通用搜索单元（GSU）以亚线性时间复杂度从原始长期行为序列中寻找 Top- $K$ 相关的子行为序列。这里 $K$ 通常远短于行为序列的原始长度。如果可以在时间和计算资源的限制下搜索到相关行为，则可以执行高效的搜索方法。在第3.2节中，我们提供了GSU的两种直接实现：软搜索和硬搜索。GSU采用通用但有效的策略来截断原始序列行为的长度，以满足严格的时间和计算资源限制。

同时，长期用户行为序列中存在的大量噪声可能会破坏用户兴趣建模，这些噪声可以通过第一阶段的搜索策略进行过滤。

在第二阶段，引入精确搜索单元（ESU），以过滤后的子序列用户行为作为输入，进一步捕捉精确的用户兴趣。这里可以应用具有复杂架构的复杂模型，如DIN[20]和DIEN[19]，因为长期行为的长度已减少到数百。

需要注意的是，虽然我们分别介绍这两个阶段，但实际上它们是联合训练的。

**图1：基于搜索的兴趣模型（SIM）。** SIM采用两阶段搜索策略，由两个单元组成：

（i）通用搜索单元从超过一万个用户行为中寻找最相关的 $K$ 个行为。
（ii）精确搜索单元利用多头注意力捕捉多样化的用户兴趣。

然后遵循传统的Embedding&MLP范式（MLP，Multi-Layer Perceptron，多层感知机），以精确的长期用户兴趣输出和其他特征作为输入。在本文中，我们为GSU引入了硬搜索和软搜索。硬搜索意味着选择与候选item属于同一类别的行为数据。软搜索意味着基于嵌入向量对每个用户行为数据进行索引，并使用最大内积搜索来寻找 Top- $K$ 行为。对于软搜索，GSU和ESU共享相同的嵌入参数，这些参数在学习过程中同时训练， Top- $K$ 行为序列基于最新的参数生成。

### 3.2 通用搜索单元

给定一个候选item（待由点击率模型评分的目标item），只有一部分用户行为是有价值的。这部分用户行为与最终的用户决策密切相关。挑选出这些相关的用户行为有助于用户兴趣建模。然而，使用完整的用户行为序列直接建模用户兴趣将带来巨大的资源消耗和响应延迟，这在实际应用中通常是不可接受的。为此，我们提出了一个通用搜索单元来减少用户兴趣建模中用户行为的输入数量。这里我们介绍两种通用搜索单元：硬搜索和软搜索。

给定用户行为列表 $B = [b_1; b_2; \cdots; b_T]$ ，其中 $b_i$ 是第 $i$ 个用户行为， $T$ 是用户行为的长度。通用搜索单元计算每个行为 $b_i$ 相对于候选item的相关性得分 $r_i$ ，然后选择得分 $r_i$ 最高的 Top- $K$ 相关行为作为子行为序列 $B^*$ 。硬搜索和软搜索之间的区别在于相关性得分 $r_i$ 的公式：

$$
r_i =
\begin{cases}
\text{Sign}(C_i = C_a) & \text{hard-search} \\
(W_b e_i) \odot (W_a e_a)^T & \text{soft-search}\end{cases}
\qquad (1)
$$

#### 3.2.1 硬搜索

硬搜索模型是非参数的。
只有与候选item属于同一类别的行为才会被选择并聚合为子行为序列，发送给精确搜索单元。这里 $C_a$ 和 $C_i$ 分别表示目标item和第 $i$ 个行为 $b_i$ 所属的类别。硬搜索很直观，但在第4节中我们将展示它非常适合在线服务。

#### 3.2.2 软搜索

在软搜索模型中， $b_t$ 首先被编码为独热向量，然后嵌入到低维向量 $E = [e_1; e_2; \cdots; e_T]$ 中，如图1所示。 $W_b$ 和 $W_a$ 是权重参数。 $e_a$ 和 $e_i$ 分别表示目标item和第 $i$ 个行为 $b_i$ 的嵌入向量。为了进一步加速对超过一万长度的用户行为进行 Top- $K$ 搜索，基于嵌入向量 $E$ 使用亚线性时间最大内积搜索方法 ALSH[12] 来搜索与目标item相关的 Top- $K$ 行为。借助训练良好的嵌入和最大内积搜索（MIPS，Maximum Inner Product Search）方法，超过一万的用户行为可以缩减到数百个。

需要注意的是，长期和短期数据的分布是不同的。因此，直接在软搜索模型中使用从短期用户兴趣建模中学到的参数可能会误导长期用户兴趣建模。在本文中，软搜索模型的参数是在基于长期行为数据的辅助点击率预测任务下训练的，如图1左侧的软搜索训练所示。行为表示 $U_r$ 通过 $r_i$ 和 $e_i$ 相乘得到：

$$
U_r = \sum_{i=1}^T r_i e_i \qquad (2)
$$

行为表示 $U_r$ 和目标向量 $e_a$ 随后拼接作为后续MLP（多层感知机）的输入。注意，如果用户行为增长到一定程度，将所有用户行为直接输入模型是不可能的。在这种情况下，可以从长序列用户行为中随机采样子序列集合，这仍然遵循与原始序列相同的分布。

### 3.3 精确搜索单元

在第一个搜索阶段，从长期用户行为中选择了与目标item相关的 Top- $K$ 子用户行为序列 $B^{*}$ 。为了进一步从相关行为中建模用户兴趣，我们引入了精确搜索单元，这是一个基于注意力的模型，以 $B^{*}$ 作为输入。

考虑到这些被选择的用户行为跨越了很长时间，因此用户行为的贡献是不同的，我们为每个行为引入了序列时间属性。具体而言，目标item与被选中的 $K$ 个用户行为之间的时间间隔 $D = [\Delta_1; \Delta_2; ...; \Delta_K]$ 用于提供时间距离信息。 $B^{*}$ 和 $D$ 被编码为嵌入 $E^{*} = [e^{*}_1; e^{*}_2; ...; e^{*}_K]$ 和 $E_t = [e_{t1}; e_{t2}; ...; e_{tK}]$ 。 $e^{*}_j$ 和 $e_{t_j}$ 拼接作为用户行为的最终表示，记为 $z_j = \text{concat}(e^{*}_j, e_{t_j})$ 。我们利用多头注意力来捕捉多样化的用户兴趣：

$$
atti\_score = \text{Softmax}(W_{bi} z_b \odot W_{ai} e_a) \qquad (3)
$$

$$
head_i = atti\_score\ z_b \qquad (4)
$$

其中 $atti\_score$ 是第 $i$ 个注意力得分， $head_i$ 是多头注意力中的第 $i$ 个头。 $W_{bi}$ 和 $W_{ai}$ 是第 $i$ 个权重参数。最终的用户长期多样化兴趣表示为 $U_{lt} = \text{concat}(head_1; ...; head_q)$ 。然后将其输入MLP进行点击率预测。

最后，通用搜索单元和精确搜索单元在交叉熵损失函数下同时训练。

$$
Loss = \alpha Loss_{GSU} + \beta Loss_{ESU} \qquad (5)
$$

其中 $\alpha$ 和 $\beta$ 是控制损失权重的超参数。在我们的实验中，如果GSU使用软搜索模型， $\alpha$ 和 $\beta$ 均设为1。使用硬搜索模型的GSU是非参数的， $\alpha$ 设为0。

## 4 在线服务实现

在本节中，我们将介绍在阿里巴巴展示广告系统中实现SIM的实践经验。

**图2：我们工业展示广告系统中用于点击率任务的实时预测（RTP，Real-Time Prediction）系统。**

它由两个关键组件组成：计算节点和预测服务器。长序列用户行为数据会给RTP在线系统带来巨大的存储和延迟压力。

### 4.1 在线服务中终身用户行为数据的挑战

工业推荐或广告系统需要在一秒内处理海量流量请求，这要求点击率模型实时响应。通常，服务延迟应小于30毫秒。图2简要说明了我们在线展示广告系统中用于点击率任务的实时预测（RTP）系统。

考虑到终身用户行为，使长期用户兴趣模型在实时工业系统中服务变得更加困难。存储和延迟约束可能是长期用户兴趣模型的瓶颈[8]。流量会随着用户行为序列长度的增长而线性增加。此外，我们的系统在流量高峰时每秒服务超过100万用户。因此，将长期模型部署到在线系统是一个巨大的挑战。

### 4.2 用于在线服务系统的基于搜索的兴趣模型

在第3.2节中，我们提出了两种通用搜索单元：软搜索模型和硬搜索模型。对于软搜索和硬搜索模型，我们在工业数据上进行了广泛的离线实验，数据收集自阿里巴巴的在线展示广告系统。我们观察到，软搜索模型生成的 Top- $K$ 行为与硬搜索模型的结果极为相似。换句话说，软搜索的大部分 Top- $K$ 行为通常属于目标item的类别。这是我们场景中数据的一个特征。在电商网站中，大多数情况下属于同一类别的item是相似的。考虑到这一点，尽管软搜索模型在离线实验中略优于硬搜索模型（详情参见表4），但在平衡了性能提升和资源消耗后，我们选择硬搜索模型在我们的广告系统中部署SIM。

对于硬搜索模型，包含所有长期序列行为数据的索引是一个关键组件。我们观察到，行为可以通过其所属的类别自然地实现分组。因此，我们为每个用户构建了一个两级结构化的索引，我们称之为用户行为树（User Behavior Tree, UBT），如图3所示。简而言之，UBT遵循Key-Key-Value数据结构：第一个键是用户ID，第二个键是类别ID，最后一个值是属于每个类别的具体行为项。UBT实现为分布式系统，大小达到22 TB，并且足够灵活以提供高吞吐量查询。然后，我们将目标item的类别作为我们的硬搜索查询。经过通用搜索单元后，用户行为的长度可以从超过一万减少到数百。因此，在线系统中终身行为的存储压力得以释放。图3显示了新的基于搜索的兴趣模型的点击率预测系统。

需要注意的是，通用搜索单元的用户行为树索引可以离线预建。这样，在线系统中通用搜索单元的响应时间可以非常短，与GSU的计算相比可以忽略不计。此外，其他用户特征可以并行计算。

**图3：采用所提SIM模型的点击率预测系统。**

新系统加入了一个硬搜索模块，用于从长序列行为数据中寻找与目标item相关的有效行为。用户行为树的索引以离线方式提前构建，节省了在线服务的大部分延迟成本。

## 5 实验

在本节中，我们将详细介绍我们的实验，包括数据集、实验设置、模型比较以及一些相应的分析。我们将所提出的搜索模型与两个公共数据集和一个工业数据集上的若干最先进工作进行比较，如表1所示。由于SIM已部署在我们的在线广告系统中，我们还进行了细致的在线A/B测试，与几个著名的工业模型进行了比较。

### 5.1 数据集

模型比较在两个公共数据集以及一个从阿里巴巴在线展示广告系统收集的工业数据集上进行。表1显示了所有数据集的统计信息。

**表1：本文使用的数据集统计信息。**

| 数据集 | 用户数 | item数a | 类别数 | 实例数 |
|--------|--------|----------|--------|--------|
| Amazon (图书) | 75053 | 358367 | 1583 | 150016 |
| Taobao | 7956431 | 34196612 | 5597 | 7956431 |
| 工业数据集 | 2.9亿 | 6亿 | 100,000 | 122亿 |

a 对于工业数据集，item指广告。

**Amazon数据集¹** 由来自Amazon的产品评论和元数据组成。我们使用Amazon数据集的Books子集，包含75053个用户、358367个item和1583个类别。对于该数据集，我们将评论视为一种交互行为，并按时间对一个用户的所有评论进行排序。Amazon图书数据集的最大行为序列长度为100。我们将最近的10个用户行为分割为短期用户序列特征，将最近的90个用户行为分割为长期用户序列特征。这种预处理方法已在相关工作中广泛使用。

¹http://jmcauley.ucsd.edu/data/amazon/

**Taobao数据集** 是来自淘宝推荐系统的用户行为集合。该数据集包含多种用户行为类型，包括点击、购买等。它包含了约八百万用户的用户行为序列。我们提取每个用户的点击行为并按时间排序，以构建行为序列。Taobao数据集的最大行为序列长度为500。我们将最近的100个用户行为分割为短期用户序列特征，将最近的400个用户行为分割为长期用户序列特征。该数据集即将发布。

**工业数据集** 收集自阿里巴巴的在线展示广告系统。样本从曝光日志中构建，以"点击"或"未点击"作为标签。训练集由过去49天的样本组成，测试集由接下来一天的样本组成，这是工业建模的经典设置。在该数据集中，每天样本的用户行为特征包含过去180天的历史行为序列作为长期行为特征，以及过去14天的历史行为序列作为短期行为特征。超过30%的样本包含长度超过10000的序列行为数据。此外，行为序列的最大长度达到54000，是MIMN[8]的54倍。

### 5.2 竞争对手与实验设置

我们将SIM与以下主流的点击率预测模型进行比较。

*   **DIN [20]** 是用户行为建模的早期工作，提出对与候选item相关的用户行为进行软搜索。与其他长期用户兴趣模型相比，DIN仅以短期用户行为作为输入。
*   **Avg-Pooling Long DIN** 为了比较长期用户兴趣上的模型性能，我们对长期行为应用平均池化操作，并将长期嵌入与其他特征嵌入拼接。
*   **MIMN [8]** 巧妙设计了模型架构来捕捉长期用户兴趣，实现了最先进的性能。
*   **SIM (硬搜索)** 是所提出的SIM模型，在第一阶段使用硬搜索，ESU中不包含时间嵌入。
*   **SIM (软搜索)** 是所提出的SIM模型，在第一阶段使用软搜索，ESU中不包含时间嵌入。
*   **SIM (硬搜索/软搜索) 带时间信息** 是SIM在第一阶段使用硬搜索或软搜索，并包含时间嵌入。

#### 5.2.1 实验设置

我们采用与相关工作[8]相同的实验设置，以便公平比较实验结果。
对于所有模型，我们使用Adam优化器。我们应用指数衰减，学习率初始值为0.001。全连接网络（FCN，Fully Connected Network）层设为 $200 \times 80 \times 2$ 。嵌入维度设为4。我们采用广泛使用的AUC作为模型性能度量指标。

### 5.3 公共数据集结果

表2展示了所有比较模型的结果。与DIN相比，引入长期用户行为特征的其他模型表现更好。这表明长期用户行为对点击率预测任务是有帮助的。SIM相比MIMN取得了显著改进，因为MIMN将所有未经过滤的用户历史行为编码到固定长度的记忆中，这使其难以捕捉多样化的长期兴趣。SIM采用两阶段搜索策略，从海量历史序列行为中搜索相关行为，并建模随不同目标item变化的多样化长期兴趣。实验结果表明，SIM优于所有其他长期兴趣模型，这有力地证明了所提出的两阶段搜索策略对于长期用户兴趣建模是有用的。此外，引入时间嵌入可以带来进一步的改进。

**表2：公共数据集上的模型性能（AUC）**

| 模型 | Taobao (均值 $\pm$ 标准差) | Amazon (均值 $\pm$ 标准差) |
|------|------------------------|------------------------|
| DIN | 0.9214 $\pm$ 0.00017 | 0.7276 $\pm$ 0.00051 |
| Avg-Pooling Long DIN | 0.9281 $\pm$ 0.00025 | 0.7280 $\pm$ 0.00012 |
| MIMN | 0.9278 $\pm$ 0.00035 | 0.7396 $\pm$ 0.00037 |
| SIM (软搜索)a | 0.9416 $\pm$ 0.00049 | 0.7510 $\pm$ 0.00052 |
| SIM (软搜索) 带时间信息 | 0.9501 $\pm$ 0.00017 | -b |

a SIM（软搜索）是使用软搜索但不包含时间间隔嵌入的SIM。
b 我们未在Amazon数据集上进行该实验，因为其中不包含时间戳特征。

**表3：两阶段搜索架构对长期用户兴趣建模的有效性评估**

| 操作 | Taobao (均值 $\pm$ 标准差) | Amazon (均值 $\pm$ 标准差) |
|------|------------------------|------------------------|
| 无搜索的平均池化 | 0.9281 $\pm$ 0.00025 | 0.7280 $\pm$ 0.00012 |
| 仅第一阶段（硬搜索） | 0.9330 $\pm$ 0.00031 | 0.7365 $\pm$ 0.00022 |
| 仅第一阶段（软搜索） | 0.9357 $\pm$ 0.00025 | 0.7342 $\pm$ 0.00012 |
| SIM（硬搜索） | 0.9332 $\pm$ 0.00008 | 0.7413 $\pm$ 0.00016 |
| SIM（软搜索） | 0.9416 $\pm$ 0.00049 | 0.7510 $\pm$ 0.00052 |
| SIM（软搜索）带时间信息 | 0.9501 $\pm$ 0.00017 | - |

### 5.4 消融研究

#### 5.4.1 两阶段搜索的有效性

如上所述，所提出的搜索兴趣模型使用两阶段搜索策略。
第一阶段采用通用搜索策略来过滤出与目标item相关的历史行为。第二阶段在第一阶段的行为上执行基于注意力的精确搜索，以准确捕捉用户在目标item上的多样化长期兴趣。在本节中，我们将通过对长期历史行为应用不同操作的实验来评估所提出的两阶段搜索架构的有效性。如表3所示，无搜索的平均池化是简单地对长期行为嵌入进行平均池化而不进行任何过滤，与Avg-pooling Long DIN相同。仅第一阶段（硬搜索）是在第一阶段对长期历史行为应用硬搜索，并通过平均池化将过滤后的嵌入整合为固定大小的向量作为MLP的输入。仅第一阶段（软搜索）与仅第一阶段（硬搜索）几乎相同，只是第一阶段应用了参数化的软搜索而非硬搜索。在第三个实验中，我们基于预训练的嵌入向量离线计算目标item与长期用户行为之间的内积相似度得分。软搜索根据相似度得分选择Top-50相关行为来执行。最后三个实验是采用两阶段搜索架构的所提搜索模型。

如表3所示，所有带有过滤策略的方法相比简单地对嵌入进行平均池化都极大地提升了模型性能。这表明原始长期行为序列中确实存在大量噪声，可能会破坏长期用户兴趣学习。与仅有一个阶段搜索的模型相比，所提出的带有两阶段搜索策略的搜索模型通过在第二阶段引入基于注意力的搜索，取得了进一步的进展。这表明在目标item上精确建模用户多样化的长期兴趣对点击率预测任务是有帮助的。并且在第一阶段搜索之后，过滤后的行为序列通常比原始序列短得多。注意力操作不会给在线服务RTP系统带来太多负担。

引入时间嵌入取得了进一步的改进，这表明不同时期用户行为的贡献是不同的。

**图4：来自DIEN和SIM的点击样本分布。**

点击被分为两部分：长期（ $>14$ 天）和短期（ $\leq 14$ 天），根据我们提出的指标"距离上次同类行为的天数"（dcategory）进行聚合。聚合尺度在短期（2天）和长期（20天）中不同。方框展示了SIM相对于不同dcategory的点击比例提升。

### 5.5 工业数据集结果

我们进一步在从阿里巴巴在线展示广告系统收集的数据集上进行实验。表4显示了结果。与第一阶段的硬搜索相比，软搜索表现更好。同时，我们注意到两种搜索策略在第一阶段的表现差距很小。在第一阶段应用软搜索会消耗更多的计算和存储资源。因为在线服务中需要使用最近邻搜索方法，而硬搜索仅需要从两级索引表中搜索，该索引表在离线构建。因此，硬搜索更高效且对系统更友好。此外，对于两种不同的搜索策略，我们对工业数据集中具有长期历史行为的超过100万个样本和10万个用户进行了统计。结果显示，硬搜索策略保留的用户行为可以覆盖软搜索策略结果的75%。最终，在效率和性能之间的权衡中，我们在第一阶段选择了更简单的硬搜索策略。SIM在AUC上比MIMN提升了0.008，这对我们的业务意义重大。

**表4：工业数据集上的模型性能（AUC）**

| 模型 | AUC |
|------|-----|
| DIEN | 0.6452 |
| MIMN | 0.6541 |
| SIM（硬搜索） | 0.6604 |
| SIM（软搜索） | 0.6625 |
| SIM（硬搜索）带时间信息a | 0.6624 |

a 该模型已部署在我们的在线服务系统中，目前服务于主要流量。

### 5.6 在线A/B测试

自2019年以来，我们已将所提出的解决方案部署到阿里巴巴的展示广告系统中。
从2020年1月7日到2020年2月7日，我们进行了严格的在线A/B测试实验来验证所提出的SIM模型。与MIMN（我们上一个产品模型）相比，SIM在阿里巴巴展示广告场景中取得了巨大的收益，如表5所示。目前，SIM已部署上线，每天服务于主要场景流量，为业务收入增长做出了重要贡献。

**表5：SIM相比MIMN在线结果的提升率，2020年1月7日至2月7日，淘宝App首页"猜你喜欢"栏目**

| 指标 | 提升率 |
|------|--------|
| CTR | 7.1% |
| RPM | 4.4% |

#### 5.6.1 重新审视搜索模型

我们在用户长期兴趣建模方面付出了巨大努力，所提出的SIM在离线与在线评估中都取得了良好的性能。
但SIM表现更好是否真的是由于精确的长期兴趣建模？SIM是否更倾向于推荐与人们长期兴趣相关的item？为了回答这两个问题，我们制定了另一个指标。一个点击样本的"距离上次同类行为的天数"（dcategory）定义为用户对与该点击样本同类item的过去行为与该点击事件发生之间的天数。例如，用户 $u_1$ 点击了一个类别为 $c_1$ 的item $i_1$ ， $u_1$ 在5天前点击了与 $i_1$ 同类的item $i_2$ ，这是 $u_1$ 在 $c_1$ 上的过去行为。如果该点击事件记为 $s_1$ ，那么样本 $s_1$ 的"距离上次同类行为的天数"即为5（ $d_{category}^{s_1} = 5$ ）。此外，如果用户 $u_1$ 从未有过类别 $c_1$ 上的行为，我们将 $d_{category}^{s_1}$ 设为 $-1$ 。对于特定模型，dcategory可以用来评估模型对长期或短期兴趣的选择偏好。

在线A/B测试后，我们基于所提出的指标dcategory分析了来自SIM和DIEN（短期点击率预测模型的最新版本）的点击样本。点击在dcategory上的分布如图4所示。可以发现在短期部分（ $d_{category} < 14$ ）两种模型几乎没有差异，因为SIM和DIEN都有最近14天的短期用户行为特征。而在长期部分，SIM占据了更大的比例。此外，我们在工业数据集上统计了dcategory的平均值和用户在目标item上有历史类别行为的概率（ $p(d_{category} > -1)$ ），如表6所示。工业数据集上的统计结果证明，SIM的改进确实源于更好的长期兴趣建模，并且与DIEN相比，SIM更倾向于推荐与人们长期行为相关的item。

**表6：工业数据集推荐中dcategory的统计信息**

| 模型 | 平均 $d_{category}$ | $p(d_{category} > -1)$ |
|------|---------------|-------------------|
| DIEN | 11.2 | 0.91 |
| SIM | 13.3 | 0.94 |

**部署实践经验。** 这里我们介绍在在线服务系统中实现SIM的实践经验。阿里巴巴的高流量是众所周知的，在流量高峰时每秒服务超过100万用户。此外，对于每个用户，RTP系统需要计算数百个候选item的预测点击率。我们为所有用户行为数据构建了离线两级索引，并每日更新。第一阶段是用户ID。第二阶段，一个用户的终身行为数据按该用户交互过的类别建立索引。虽然候选item有数百个，但这些item的类别数量通常少于20个。同时，GSU对每个类别生成的子行为序列长度被截断为200（原始长度通常小于150）。这样，每个请求的用户流量是有限且可接受的。此外，我们通过深度核融合优化了ESU中多头注意力的计算。

图5展示了DIEN、MIMN和SIM在不同吞吐量下的实时点击率预测系统延迟性能。值得注意的是，MIMN能处理的最大用户行为长度为1000，所展示的性能基于截断后的行为数据。而SIM中的用户行为长度未被截断，可以扩展到54000，将最大长度推高了54倍。SIM在服务超过一万个行为时，相比MIMN使用截断用户行为，延迟仅增加了5毫秒。

**图5：不同吞吐量下实时点击率预测系统的系统性能。**

MIMN和DIEN中用户行为长度被截断为1000，而SIM中用户行为长度可扩展到一万。DIEN的最大吞吐量为200，因此图中只有一个点。

## 6 结论

在本文中，我们专注于在真实工业中利用超过一万的序列用户行为数据。提出了基于搜索的兴趣模型来捕捉用户相对于目标item的多样化长期兴趣。在第一阶段，我们提出了通用搜索单元，将上万个行为减少到数百个。在第二阶段，精确搜索单元利用数百个相关行为来建模精确的用户兴趣。我们在阿里巴巴的展示广告系统中实现了SIM。SIM带来了显著的业务改进，并正在服务主要流量。

SIM引入了比先前方法更多的用户行为数据，实验结果表明SIM更加关注长期兴趣。但搜索单元在所有用户之间仍然共享相同的公式和参数。未来，我们将尝试构建用户特定的模型，根据个人意识来组织每个用户的终身行为数据。这样，每个用户将拥有自己的个性化模型，持续建模用户不断演化的兴趣。

## 参考文献

[1] Heng-Tze Cheng, Levent Koc, Jeremiah Harmsen, Tal Shaked, Tushar Chandra, Hrishi Aradhye, Glen Anderson, Greg Corrado, Wei Chai, Mustafa Ispir, et al. 2016. Wide & deep learning for recommender systems. In *Proceedings of the 1st Workshop on Deep Learning for Recommender Systems*. ACM, 7–10.

[2] Paul Covington, Jay Adams, and Emre Sargin. 2016. Deep neural networks for youtube recommendations. In *Proceedings of the 10th ACM Conference on Recommender Systems*. ACM, 191–198.

[3] Yufei Feng, Fuyu Lv, Weichen Shen, Menghan Wang, Fei Sun, Yu Zhu, and Keping Yang. 2019. Deep session interest network for click-through rate prediction. *arXiv preprint arXiv:1905.06482* (2019).

[4] Huifeng Guo, Ruiming Tang, Yunming Ye, Zhenguo Li, and Xiuqiang He. 2017. Deepfm: a factorization-machine based neural network for ctr prediction. In *Proceedings of the 26th International Joint Conference on Artificial Intelligence*. Melbourne, Australia., 2782–2788.

[5] Balázs Hidasi, Alexandros Karatzoglou, Linas Baltrunas, and Domonkos Tikk. 2015. Session-based recommendations with recurrent neural networks. *arXiv preprint arXiv:1511.06939* (2015).

[6] Chao Li, Zhiyuan Liu, Mengmeng Wu, Yuchi Xu, Huan Zhao, Pipei Huang, Guoliang Kang, Qiwei Chen, Wei Li, and Dik Lun Lee. 2019. Multi-interest network with dynamic routing for recommendation at Tmall. In *Proceedings of the 28th ACM International Conference on Information and Knowledge Management*. 2615–2623.

[7] Jianxun Lian, Xiaohuan Zhou, Fuzheng Zhang, Zhongxia Chen, Xing Xie, and Guangzhong Sun. 2018. xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems. In *Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*. London, United Kingdom.

[8] Qi Pi, Weijie Bian, Guorui Zhou, Xiaoqiang Zhu, and Kun Gai. 2019. Practice on Long Sequential User Behavior Modeling for Click-through Rate Prediction. In *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*. ACM, 1059–1068.

[9] Yanru Qu, Han Cai, Kan Ren, Weinan Zhang, Yong Yu, Ying Wen, and Jun Wang. 2016. Product-Based Neural Networks for User Response Prediction. (2016), 1149–1154.

[10] Kan Ren, Jiarui Qin, Yuchen Fang, Weinan Zhang, Lei Zheng, Weijie Bian, Guorui Zhou, Jian Xu, Yong Yu, Xiaoqiang Zhu, et al. 2019. Lifelong Sequential Modeling with Personalized Memorization for User Response Prediction. In *Proceedings of the 42nd International ACM SIGIR Conference on Research and Development in Information Retrieval*. 565–574.

[11] Ying Shan, T Ryan Hoens, Jian Jiao, Haijing Wang, Dong Yu, and JC Mao. [n.d.]. Deep Crossing: Web-scale modeling without manually crafted combinatorial features.

[12] Anshumali Shrivastava and Ping Li. 2014. Asymmetric LSH (ALSH) for sublinear time maximum inner product search (MIPS). In *Advances in Neural Information Processing Systems*. 2321–2329.

[13] Fei Sun, Jun Liu, Jian Wu, Changhua Pei, Xiao Lin, Wenwu Ou, and Peng Jiang. 2019. BERT4Rec: Sequential recommendation with bidirectional encoder representations from transformer. In *Proceedings of the 28th ACM International Conference on Information and Knowledge Management*. 1441–1450.

[14] Jiaxi Tang and Ke Wang. 2018. Personalized top-n sequential recommendation via convolutional sequence embedding. In *Proceedings of the Eleventh ACM International Conference on Web Search and Data Mining*. 565–573.

[15] Ruoxi Wang, Bin Fu, Gang Fu, and Mingliang Wang. 2017. Deep & Cross Network for Ad Click Predictions. (2017), 12.

[16] Zeping Yu, Jianxun Lian, Ahmad Mahmoody, Gongshen Liu, and Xing Xie. 2019. Adaptive user modeling with long and short-term preferences for personalized recommendation. In *Proceedings of the 28th International Joint Conference on Artificial Intelligence*. AAAI Press, 4213–4219.

[17] Fajie Yuan, Alexandros Karatzoglou, Ioannis Arapakis, Joemon M Jose, and Xiangnan He. 2019. A simple convolutional generative network for next item recommendation. In *Proceedings of the Twelfth ACM International Conference on Web Search and Data Mining*. 582–590.

[18] Shuangfei Zhai, Keng-hao Chang, Ruofei Zhang, and Zhongfei Mark Zhang. 2016. Deepintent: Learning attentions for online advertising with recurrent neural networks. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*. ACM, 1295–1304.

[19] Guorui Zhou, Na Mou, Ying Fan, Qi Pi, Weijie Bian, Chang Zhou, Xiaoqiang Zhu, and Kun Gai. 2019. Deep interest evolution network for click-through rate prediction. In *Proceedings of the 33nd AAAI Conference on Artificial Intelligence*. Honolulu, USA.

[20] Guorui Zhou, Xiaoqiang Zhu, Chenru Song, Ying Fan, Han Zhu, Xiao Ma, Yanghui Yan, Junqi Jin, Han Li, and Kun Gai. 2018. Deep interest network for click-through rate prediction. In *Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*. ACM, 1059–1068.
