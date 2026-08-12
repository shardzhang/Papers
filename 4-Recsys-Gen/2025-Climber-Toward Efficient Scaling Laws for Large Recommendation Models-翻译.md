# Climber: Toward Efficient Scaling Laws for Large Recommendation Models


本文介绍了 Climber: Toward Efficient Scaling Laws for Large Recommendation Models。核心内容：


关键发现：

---


## Climber：面向大型推荐模型的高效扩展定律

Songpei Xu, Shijia Wang\*, Da Guo
> xusongpei@corp.netease.com, wangshijia1@corp.netease.com, guoda@corp.netease.com
NetEase Cloud Music, Hangzhou, China

Xianwen Guo, Qiang Xiao†, Bin Huang
guoxianwen@corp.netease.com, hzxiaoqiang@corp.netease.com, huangbin02@corp.netease.com
NetEase Cloud Music, Hangzhou, China

Guanlin Wu, Chuanjiang Luo
wuguanlin03@corp.netease.com, luochuanjiang03@corp.netease.com
NetEase Cloud Music, Hangzhou, China

---


## 摘要

基于Transformer的生成模型已在各个领域取得成功，并展现出不同的**扩展定律**表现形式。然而，我们的大量实验揭示了将Transformer应用于推荐系统时存在的持续挑战：(1) 由于与推荐系统特定特征（如多源数据异质性）的结构性不兼容，Transformer在增加计算资源时的扩展并不理想；(2) 关键的在线推理延迟约束（几十毫秒）随着更长的用户行为序列和不断增长的计算需求而加剧。我们提出Climber，一个高效的推荐框架，包含两个协同组件：用于高效扩展的模型架构 和 协同设计的加速技术。我们提出的模型采用两项核心创新：(1) 多尺度序列提取，实现时间复杂度的常数因子降低，从而实现随序列长度更高效的扩展；(2) 动态温度调节，使注意力分布适应多场景和多行为模式。配合加速技术，Climber通过采用"单用户多item"批量处理和高内存效率的键值缓存，实现了5.15倍的吞吐量提升且无性能下降。

在多个数据集上的全面离线实验验证了Climber展现出更理想的扩展曲线。据我们所知，这是第一个公开记录的可控模型扩展驱动持续在线指标增长（总体提升12.19%）且无需过高资源成本的框架。Climber已成功部署在网易云音乐上——中国最大的音乐流媒体平台之一，每天服务数千万用户。

**CCS概念**

- 信息系统 $\to$ 推荐系统。

**关键词**

推荐系统；Transformer；扩展定律；生成式推荐

---


## 1 引言

扩展定律最初在语言模型[13, 18]中探索，建立了模型性能与关键因素（如模型大小和训练数据量）之间的可预测关系。例如，Kaplan等人[18]证明了基于Transformer的语言模型[32]在困惑度上随着模型参数和token数量的增加而遵循幂律改进。类似的趋势在视觉模型[7, 39]和多模态模型[2, 30]中也观察到，其中模型的扩展维度和数据的多样性直接影响下游任务的性能。

生成式推荐已成为在推荐系统中实现扩展定律的最有前景的新技术范式。我们认为其实际实现是分阶段展开的，当前阶段主要致力于将Transformer架构适配到推荐系统——核心目标是建立强大的扩展定律。最近的研究[1, 10]验证了扩展定律在推荐系统中的有效性，为模型设计和资源分配提供了宝贵的见解。HSTU模型[38]采用层次化自注意力机制来建模长期用户行为序列，实现了比传统Transformer更好的性能。类似地，MARM模型[25]引入记忆增强来降低计算复杂度，实现了以最小推理成本进行多层序列建模。然而，这些方法未能充分解决Transformer架构与推荐系统特定特征之间的固有不兼容性。尽管通过扩展资源来扩展模型仍然是可行的，但这种策略在实际工业部署中被证明是低效的。此外，关键扩展因素——序列长度、模型深度和异质性用户行为——之间的相互作用在传统推荐系统中仍未得到充分探索[33, 34, 37]，导致了次优的资源分配和递减的扩展收益。

受DeepSeek系列[3, 21, 22]（其显著提升了大型语言模型（LLM）开发的效率并降低了计算资源成本）的启发，我们旨在解决以下问题：我们如何以大幅降低的成本高效扩展推荐模型？

为了获得见解，我们对两种主流模型——深度学习推荐模型（DLRM）[27]和Transformer模型——进行了工业规模的分析。在图1(a)中，我们展示了DLRM\*和Transformer的扩展曲线，并包含了一条理想扩展曲线（oracle曲线），其特点是更高的起点和更大的斜率。在图1(b)中，我们展示了从各种序列长度和层数组合的模拟中得出的AUC曲线，并引入了"性能区间"的概念，它代表了模型在等效FLOPs下的AUC变化范围。然而，我们的发现揭示了在推荐系统中应用Transformer时仍然存在一些问题：

- **Transformer在FLOPs约束下的性能下降**：如图1(a)所示，交叉点对应的FLOPs为10^8.2。以此FLOPs值为界，DLRM和Transformer的性能比较显示出不同的趋势。当FLOPs超过10^8.2量级时，Transformer模型优于传统架构如DLRM。然而，当FLOPs小于10^8.2时，Transformer模型的表现不如DLRM。这凸显了对更高效模型的追求，即图1(a)中所示的理想曲线。理想曲线代表了一条具有更大截距和斜率的更高效扩展曲线，这使得模型即使在FLOPs有限的情况下也能实现更好的性能。

- **Transformer与推荐系统特定特征之间的不兼容性**：与NLP的连续句法序列不同，推荐系统处理的是跨越多个场景的碎片化用户行为[14, 15, 20, 24, 41]，导致注意力分布混乱，因为Transformer难以在稀疏的多源模式中优先处理相关行为。此外，多场景推荐面临分布差异的问题，用户表现出不同的行为，但现有方法将场景视为辅助特征而非显式的分布控制器。这种不兼容性使得Transformer在与DLRM等专门化架构相比时效率低下，尤其是在计算约束条件下。

- **等效FLOPs下因素组合对模型性能的影响**：在推荐系统中，序列长度和层数等因素显著影响FLOPs，而这些因素的不同组合会导致不同的模型性能。例如，如图1(b)所示，在10^9 FLOPs下，不同组合的性能区间接近1%。当前研究缺乏对因素组合如何影响推荐模型性能的全面分析，阻碍了高效的模型扩展。

这些挑战证明了Transformer在推荐系统中的低效性，当扩展特征和模型容量以处理更长的序列时，这种低效性会加剧，导致二次方的计算需求和更严格的延迟约束[6, 29]。

基于上述见解，我们引入了Climber，一个重新思考推荐系统扩展范式的新型框架。其核心是，Climber整合了两项互补的创新：一个面向推荐的基于Transformer的模型架构和协同设计的加速技术。我们提出的模型通过引入多尺度序列提取重新定义了推荐系统处理用户行为的方式，该技术将用户行为序列分解为更小、更细粒度的子序列。这种方法不仅降低了计算复杂度，还能跨不同场景更精确地对用户兴趣进行建模。此外，该模型还融入了动态温度调节机制，自适应调整注意力分数以适应不同行为和场景的变化重要性。在工程方面，我们引入了统一的加速技术，将传统的"单用户单item"样本组织形式转变为与实际在线请求一致的格式——"单用户多item"。基于这些技术，训练和推理过程中使用编码器级KV缓存的前向传播实现了显著的效率提升。最后，我们研究了Climber的可扩展性以及等效FLOPs下因素组合对AUC的影响，为理性资源分配和快速模型扩展的关键因素提供了新颖的见解。

我们的贡献主要分类如下：

- 我们提出了推荐系统中扩展定律的工业规模研究，明确量化了等效FLOPs下因素组合的影响。该分析揭示了平衡扩展——交替进行序列和深度扩展——能同时带来离线指标和在线指标的增长。
- 我们提出了一种新颖的Transformer变体Climber，通过多尺度提取和自适应温度调节解决了推荐系统中的扩展困境。据我们所知，所提出的方法开创了可持续扩展——实现了+12.19%的在线指标增长，这是我们生产系统中年内最高的改进幅度。
- 统一的加速技术通过"单用户多item"批量处理和块级并行KV缓存提升了训练和推理效率。部署在网易云音乐上，这些技术实现了5.15倍的训练加速，并使我们的模型在线推理比DLRM快高达14.38倍，使得在不增加计算资源的情况下实现100倍的模型扩展成为可能。

---

## 2 相关工作

Wukong[40]探索了检索模型中的参数扩展，但依赖于对特征工程的强假设。HSTU[38]将推荐重构为序列转导任务，通过层次化注意力和随机序列采样实现了具有线性计算扩展的万亿参数模型。然而，HSTU对生成式建模的关注在桥接传统基于特征的DLRM方面存在空白。同时，MARM[25]提出缓存中间注意力结果，将推理复杂度从O(n²d)降至O(nd)，验证了缓存大小作为一个新的扩展维度。虽然有效，但MARM的缓存策略假设用户模式是静态的，忽视了实时行为的变化。

缓解计算开销的技术已被广泛采用。在NLP中，KV缓存[9, 23]避免了自回归推理中冗余的注意力计算。MARM将这一思想适配到推荐系统中，通过存储历史注意力输出，实现了以最小FLOPs开销进行多层目标注意力。类似地，HSTU引入了随机长度（Stochastic Length）来算法化地对长序列进行稀疏化处理，在无质量下降的情况下将训练成本降低了80%。对于广告检索，Wang等人[35]设计了R/R\*，一种eCPM感知的离线指标，以低实验成本估计在线收入扩展定律。这些工作共同强调了针对推荐系统特定约束（如高基数特征和毫秒级延迟要求）定制效率策略的重要性。

---


## 3 方法

为了实现推荐系统的高效扩展，我们提出了一个面向推荐的Transformer变体。该变体支持沿三个关键维度进行扩展：多尺度序列处理、多场景适应和多兴趣建模。此外，我们展示了对所提出模型的全面部署细节。

### 3.1 模型架构

#### 3.1.1 总体架构

为了解决推荐系统中的计算复杂度和扩展挑战，我们从推荐的角度提出了该模型。该模型将推荐特性融入Transformer架构中，同时具备资源感知的可扩展性。它从三个维度实现扩展：多尺度序列、多场景和多兴趣。

我们的模型包含三个模块：多尺度序列提取（MSE）、自适应Transformer层（ATL）和逐位门控融合（BGF）。具体来说，MSE从用户生命周期序列生成多尺度序列。这些多尺度序列代表不同类型的子序列。每个子序列由一组堆叠的ATL组成的对应块进行处理，以进行兴趣提取。此外，我们将重要子序列的时间跨度扩展到覆盖用户的整个生命周期。ATL采用自适应温度系数来调整多场景中的注意力分布。最后，BGF通过逐位门控机制聚合来自自适应Transformer块的表示，实现多尺度序列之间的多兴趣融合。图2展示了详细的工作流程。

#### 3.1.2 多尺度序列扩展

我们提出了多尺度序列提取（MSE）方法用于多尺度序列扩展。该方法基于不同的策略对用户序列进行重组，可以用以下公式表示：

**S** = { $x_1$ , $x_2$ , ..., $x_{ns}$ }                                          (1)

** $S_k$ ** = MSE(**S**, $a_k$ ) = { $x_{ak}
$$
_1$, $x_{ak}
$$
_2$ , ..., $x_{ak}$ _{ $n_k$ }}        (2)

其中 **S** 表示用户生命周期序列， $x_i$ 表示来自整个item集合 X 的第 i 个itemID。** $S_k$ ** 表示基于提取策略 $a_k$ 从用户生命周期序列 **S** 中提取的第 k 个子序列， $x_{ak}
$$
_j$ 表示子序列 **$S_k$** 中的第 j 个item。$n_s$ 和 $n_k$ 分别表示 **S** 和 **$S_k$** 的长度。我们假设共有 N_b 种提取策略，且由于提取策略在实际应用中基于用户的真实提取注意力（即仅关注正行为）和推荐时间戳，因此
$$
\sum
$$
_{k=1}^{N_b} $n_k$ = $n_s$ 且 $n_k$ $≪$ $n_s$。因此，单个Transformer下的计算复杂度可以从 O($n_s$²d) 降低到 O($n_k$²d)。

我们通过为每个子序列 **$S_k$** 使用对应的Transformer块进一步改进了训练过程，从而实现了 O($n_k$²d) 的时间复杂度。我们设计每种提取策略提取等长的子序列，即 $n_k$ = $n_s$ / N_b。在全串行操作的情况下，这导致时间复杂度为 O($n_s$²d / N_b)。因此，即使当 N_b = 2 时，我们仍然可以实现显著的训练加速。在拥有充足计算资源的全并行条件下，复杂度降低到 O(max($n_k$)²d)，且复杂度仅取决于最长子序列的长度。这保证了在纳入比 $n_k$ 短的序列时训练效率的保持，从而实现了渐进式的多尺度序列扩展。

我们的提取策略包括业务驱动序列（例如，点击/喜欢/分享）、模型过滤序列等。总之，MSE降低了计算复杂度并将用户生命周期序列转化为多尺度序列。这些改进提升了推荐系统的效率和可扩展性。

#### 3.1.3 多场景扩展

Softmax函数在Transformer架构中扮演着关键角色，它将注意力分数归一化为一个概率分布。标准做法中，注意力分数通过 **QK^T** 计算，然后除以 $\sqrt{}
$$
d_k$ 再乘以 **V**。除以 $\sqrt{}
$$
d_k$ 确保了注意力矩阵的分布与 **Q** 和 **K** 的分布对齐[12]。

然而，我们基于不同的提取策略从用户生命周期序列生成多尺度序列，并对每个子序列应用相应的Transformer块。每个子序列在不同场景下的分布也表现出显著差异。单个缩放因子 $\sqrt{}
$$
d_k$ 不足以满足多场景中与多尺度序列对应的所有Transformer块的多样化需求[11]。

为了进一步细化每个Transformer块中的注意力分布，我们为每个块的每一层引入了自适应温度系数。我们将此变化称为自适应Transformer层（ATL），其数学表达如下：

**Q, K, V** = $f_{QKV}$ (**X**(** $S_k$ **))

**R**(** $S_k$ **) = **QK^T** + $f_{b}$ ^{(p,t)}( $a_k$ , r)                       (3)

**A**(** $S_k$ **) = Softmax(**R**(** $S_k$ **) / $f_{tc}$ ( $a_k$ , r))

**Y**(** $S_k$ **) = $f_{FFN}$ (**A**(** $S_k$ **) **V**)

与传统Transformer层相比，我们从推荐的角度引入了自适应温度系数并调整了相对注意力偏置。这里，**X**(** $S_k$ **)
$$
\in
$$
 ℝ^{s
$$
\times
$$
d}, **R**(** $S_k$ **)
$$
\in
$$
 ℝ^{h
$$
\times
$$
s
$$
\times
$$
s}, **A**(** $S_k$ **)
$$
\in
$$
 ℝ^{h
$$
\times
$$
s
$$
\times
$$
s} 和 **Y**(** $S_k$ **)
$$
\in
$$
 ℝ^{s
$$
\times
$$
d} 分别表示层输入、原始注意力矩阵、归一化注意力矩阵和层输出。s, h, d 分别表示序列长度、头数和特征维度。 $f_{QKV}$ (**X**(** $S_k$ **)) 用于从输入 **X**(** $S_k$ **) 推导出查询、键和值矩阵。f_b^{(p,t)}( $a_k$ , r) 表示结合了位置 (p) 和时间 (t) 信息的相对注意力偏置[28, 38]。 $f_{tc}$ ( $a_k$ , r) 表示推导温度系数的函数。需要注意的是， $f_{tc}$ 同时受提取策略 $a_k$ 和推荐场景 r 的影响。 $f_{FFN}$ 通过前馈神经网络（FFN）处理注意力加权的值矩阵，以生成层的最终输出。

这种方法受推荐系统的多场景和多行为特性的启发。与HSTU的固定温度系数[38]不同，我们的自适应温度系数允许更灵活的注意力加权，解决了固定缩放因子 $\sqrt{}
$$
d_k$ 在捕捉多样化行为和场景内在特性方面的局限性。

#### 3.1.4 多兴趣扩展

然而，当用户生命周期序列被分离为多尺度序列时，它们之间缺乏交互[19, 36]。因此，我们提出了逐位门控融合模块，该模块整合了 N_b 个子序列之间的信息。

具体来说，每个子序列 **$S_k$** 通过每个自适应Transformer块产生输出向量，这些 N_b 个输出向量被拼接为 **E**(**S**)
$$
\in
$$
 ℝ^{N_b
$$
\times
$$
d}。拼接后的向量随后通过一个新的ATL处理，然后经过sigmoid激活函数实现逐位级门控。最后，向量通过一个后续网络产生最终输出分数。逐位门控融合模块可以表示为：

**E**(**S**) = {**E**(**$S_1$**), **E**(**$S_2$**), ..., **E**(**$S_{N_b}$**)}

**G**(**S**) = ATL(**E**(**S**))                                         (4)

**Y**(**S**) = **G**(**S**)
$$
\odot
$$
 $\sigma$($f_{gate}$(**G**(**S**)))

其中 $\sigma$ 是sigmoid激活函数，$f_{gate}$ 代表挤压-激励模块[16]，它确保输入和输出维度的一致性，以动态调整 **G**(**S**)
$$
\in
$$
 ℝ^{N_b
$$
\times
$$
d} 中每个元素的贡献。$f_{gate}$(**G**(**S**))
$$
\in
$$
 ℝ^{N_b
$$
\times
$$
d} 和 **Y**(**S**)
$$
\in
$$
 ℝ^{N_b
$$
\times
$$
d} 分别表示逐位注意力矩阵和融合模块的输出。ATL（公式3中的自适应Transformer层）用于计算不同块之间的相似度，以实现子序列中不同兴趣的交互。

相比之下，融合函数中的ATL不包含相对注意力偏置，温度系数仅由推荐场景决定。在 N_b 个块上的ATL注意力操作可以被视为域级交互[16, 17]，促进了特征级别的信息交换。我们的方法通过添加逐位交互增强了多兴趣融合。这使得模型能够捕获序列之间的精确关系。因此，模型理解多尺度序列的能力得到了显著提升。需要注意的是，由于提取策略的数量 N_b 远小于特征向量的维度 d，融合阶段的逐位门控融合模块的计算复杂度 O(N_b²d) 相对较低。

### 3.2 部署

我们的加速部署涉及两个阶段：离线训练和在线服务。在离线训练阶段，用户交互日志以"单用户单item"模式记录。这些原始日志被压缩并归档为"单用户多item"模式。这里，"单用户单item"表示记录用户与item之间一次原子交互（例如，点击/购买）的日志条目；"单用户多item"将用户的历史交互聚合成包含多个item的单条记录，从而实现对用户-item特征计算的批处理。"单用户多item"模式在每个候选item与整个历史之间使用全可见掩码，同时在候选item之间采用对角掩码实现item间隔离。这种压缩机制显著减少了样本量，同时为模型训练提供了5.15倍的加速。

在在线服务阶段，受M-FALCON[38]的启发，系统首先生成用户特征的多层KV缓存向量，然后从特征服务器获取候选item特征，最后计算item特征与缓存的KV表示之间的基于注意力的交互。值得注意的是，该模型在离线训练和在线服务阶段均采用"单用户多item"数据模式，从而利用KV缓存加速用户特征计算。此外，我们实现了算子融合，将顺序操作（如嵌入查找、注意力层）合并为统一的计算内核，以减少频繁的全局内存访问，并集成了FlashAttention的矩阵分块操作以优化内存利用[8]。这些设计通过缓存利用提高了计算效率，同时保持了预测准确性，从而以更高的用户满意度提供个性化推荐服务。

---

## 4 实验

在本节中，我们详细介绍了在真实工业数据上进行的离线实验和在线实验，以评估我们提出的方法，旨在回答以下四个研究问题：

- **RQ1**：与最先进（SOTA）模型相比，Climber在离线评估中表现如何？
- **RQ2**：与DLRM和Transformer相比，Climber如何展现出更优的可扩展性？
- **RQ3**：我们如何分配资源来扩展模型，考虑不同因素组合在等效FLOPs下对AUC的影响？
- **RQ4**：Climber在工业系统中的表现如何？

### 4.1 实验设置

#### 4.1.1 数据集

为了验证我们的方法在推荐系统中的有效性，我们使用真实用户行为序列作为主要特征构建了数据集。该数据集用于基于历史交互预测候选item上的不同用户行为。重要的用户行为包括完整播放、喜欢、分享和评论。此外，我们在三个推荐数据集上评估了我们的模型：Spotify[4]、30Music[31]和Amazon-Book[26]。表1展示了四个已处理数据集的用户数、item数和交互数。为保护数据隐私，我们仅展示推荐场景的统计数据，并对工业数据集进行了特殊处理。因此，表中显示的数据量低于实际数量。然而，很明显我们的工业数据集规模仍然显著超过其他数据集。这种庞大的数据量为进行扩展实验提供了坚实的基础。

**表1：数据集统计**

|           | Spotify  | 30Music  | Amazon-Book | Industrial |
|-----------|----------|----------|-------------|------------|
| #User     | 0.16M    | 0.02M    | 0.54M       | >40M       |
| #Item     | 3.7M     | 4.5M     | 0.37M       | >6M        |
| #Interaction | 1.2M  | 16M      | 1.09M       | >1B        |

#### 4.1.2 对比方法

本节详细说明了工业数据集中每个模型的实验设置。

- **DLRM**：该模型利用终身用户行为序列和复杂的特征交互，已部署在我们的在线系统中。实验中，序列长度固定为2000。
- **DIN**：DIN通过目标注意力机制建模用户历史行为与目标item之间的交互，捕获用户兴趣。实验中，序列长度设置为1000。
- **TWIN**：该模型对齐GSU和ESU以增强长期用户行为建模的一致性。实验中，GSU和ESU序列长度分别设置为2000和1000。
- **Transformer**：一种用于序列建模的基于Transformer的模型。实验中，单阶段Transformer编码器处理固定长度为2000的行为序列。
- **HSTU**：特征级序列通过item-动作对的时间顺序进行重新组织，并经过HSTU模型处理，以预测目标item特定的用户动作。实验中，序列长度固定为2000。
- **Climber系列**：先前的方法将序列长度限制在固定时间窗口内的2000，且无行为过滤。通过多尺度序列提取，新方法通过业务逻辑驱动的策略将行为序列扩展到整个用户生命周期，保留的长度缩减为200。模型保持2层，与基线配置一致。在工业环境中，Climber-large变体扩展到12层和800序列长度。

### 4.2 总体性能（RQ1）

#### 4.2.1 性能比较

如表2所示，Climber在四个推荐数据集上取得了最佳性能。值得注意的是，Spotify和30Music的应用场景与我们的工业数据集属于同一领域，即音乐推荐系统。相比之下，Amazon-Book与我们的场景差异显著。然而，我们的模型在此数据集上仍然取得了良好的结果，表明其具有适应多样化应用的潜在能力。

接下来，我们重点关注Climber相对于其他方法的AUC改进比较。

1) 作为我们的主要在线模型，DLRM优于DIN和TWIN，因为DLRM包含了超出终身序列和注意力机制的更广泛的特征交互结构。
2) Transformer相比TWIN实现了+0.134%的AUC提升。这是因为Transformer在单阶段中计算所有历史item与目标item之间的相似度。
3) HSTU对Transformer实施了几项增强，在我们的数据集上相比Transformer实现了+0.036%的AUC改进。然而，这些增强也导致了计算复杂度的增加。
4) Climber通过序列提取和多场景及多行为下的自适应温度系数调整注意力分布，降低了计算复杂度。相比DLRM实现了+0.170%的提升。此外，Climber-large通过扩展模型实现了+2.21%的AUC提升，达到了过去一年中最大的离线指标增长。

**表2：公开/工业数据集上的方法评估**

|                       | Spotify          | Amazon-Book       | 30Music           | Industrial        |
|-----------------------|------------------|-------------------|-------------------|-------------------|
|                       | AUC    | LogLoss | AUC    | LogLoss | AUC    | LogLoss | AUC    | LogLoss |
| DLRM (Baseline)       | 0.7606 | 0.5761  | 0.7842 | 0.5541  | 0.8927 | 0.2012  | 0.8216 | 0.7067  |
| DIN                   | 0.7557 | 0.5803  | 0.7796 | 0.5580  | 0.8861 | 0.2044  | 0.8158 | 0.7109  |
| TWIN                  | 0.7589 | 0.5772  | 0.7831 | 0.5563  | 0.8903 | 0.2019  | 0.8203 | 0.7078  |
| Transformer           | 0.7621 | 0.5735  | 0.7836 | 0.5557  | 0.8930 | 0.2007  | 0.8214 | 0.7074  |
| HSTU                  | 0.7626 | 0.5722  | 0.7869 | 0.5520  | 0.8938 | 0.1982  | 0.8217 | 0.7053  |
| Climber(-ATL,-BGF)    | 0.7635 | 0.5710  | 0.7873 | 0.5518  | 0.8944 | 0.1978  | 0.8221 | 0.7045  |
| Climber(-BGF)         | 0.7655 | 0.5694  | 0.7881 | 0.5510  | 0.8950 | 0.1972  | 0.8225 | 0.7034  |
| Climber               | 0.7663 | 0.5687  | 0.7887 | 0.5501  | 0.8986 | 0.1957  | 0.8230 | 0.7029  |
| Climber-large         | 0.7702 | 0.5666  | 0.7914 | 0.5472  | 0.9035 | 0.1916  | 0.8398 | 0.6911  |
| %Improve              | +1.26% | -1.64%  | +0.91% | -1.24%  | +1.20% | -4.77%  | +2.21% | -2.20%  |

#### 4.2.2 消融研究

为了评估Climber模型中每个组件的贡献，我们在多个数据集上进行了一组全面的实验。为便于说明，我们重点关注在工业数据集上进行的消融研究，并选择Transformer与Climber系列进行比较。

通过加入MSE，Climber(-ATL,-BGF)将用户生命周期序列转换为多尺度子序列块。这一增强相比Transformer模型带来了+0.085%的正向AUC增长。Climber(-BGF)通过引入自适应温度系数进一步改进了模型。该组件动态调整注意力分布，带来了+0.134%的AUC提升。最后，BGF的加入带来了+0.195%的AUC提升。该模块整合了不同子序列表示的用户兴趣，强调了推荐系统中兴趣融合的重要性。

总之，基于多个数据集的离线评估，我们的模型展示了强大的性能和适应性。

### 4.3 可扩展性（RQ2）

在讨论模型可扩展性之前，我们正式定义FLOPs为 **C ∝ s \* l**，其中 s 表示序列长度，l 表示模型中的层数。在大规模场景下，即使对于更长的序列，注意力机制的二次计算复杂度仅占模型总FLOPs的一小部分[5, 13, 18]。因此，在我们的计算分析中，我们只关注 **C ∝ s \* l** 关系中序列长度 s 的线性部分，FLOPs可以通过TensorFlow的特定工具计算和验证。

DLRM、Transformer和Climber的扩展曲线如图3(a)所示。虽然Transformer在FLOPs超过10^9时可以实现比DLRM更好的性能，但其在10^7到10^8之间的效率明显低于DLRM。与Transformer相比，Climber由于具有更高的起点和更大的斜率，展现出更理想的扩展曲线。当FLOPs低于10^7.5时，Climber的性能仍然弱于DLRM，但交叉点向左移动，使得Climber模型比Transformer更高效地实现性能转变。

在该实验中，影响FLOPs的两个主要因素是层数和序列长度。对于Climber，我们分别在图3(b)和3(c)中展示了模型性能与层数以及序列长度之间的关系。当序列长度固定时，模型性能随层数的增加以类似幂律的方式提升；当层数固定时，模型性能也随序列长度的增加而类似提升。因此，我们提出的Climber模型在FLOPs、序列长度和层数方面均表现出扩展曲线，并且与Transformer相比具有更高效的扩展曲线。

### 4.4 高效分配（RQ3）

从图3(b,c)可以明显看出，增加序列长度和层数都可以提高模型的AUC。然而，这两个因素的优先级缺乏讨论。表3展示了在不同层数和序列长度下等效FLOPs时的模型AUC。

可以看出，在等效FLOPs下，层数和序列长度的组合可以导致离线测试AUC的显著变化。根据 **C ∝ s \* l**，等效FLOPs下层数和序列长度的乘积保持不变。当FLOPs为4.11
$$
\times
$$
10^8时，模型在(400s
$$
\times
$$
4l)配置下达到了0.8301的最佳AUC；当FLOPs为1.01
$$
\times
$$
10^9时，模型在(400s
$$
\times
$$
8l)配置下达到了0.8335的最佳AUC。我们观察到，扩展单一因素可能会限制模型的发展。因此，在扩展模型时，最好同时考虑层数和序列长度。例如，对于一个(400s
$$
\times
$$
4l)模型，如果我们需要将FLOPs增加4倍，我们可以在(1600s
$$
\times
$$
4l)、(800s
$$
\times
$$
8l)和(400s
$$
\times
$$
16l)之间选择。从表3中可以发现，最佳选择是(800s
$$
\times
$$
8l)，它同时扩展了两个因素，将模型的AUC从0.8301提升到0.8382。这一结论也指导了如何在线上分配资源。

在我们的实际推荐系统中，通常每次迭代只选择一个因素。因此，在线扩展时，我们交替增加序列长度和层数。

**表3：等效FLOPs下的性能比较**

| FLOPs      | 序列长度 | 层数 | AUC    |
|------------|----------|------|--------|
| 4.11
$$
\times
$$
10^8  | 1600     | 1    | 0.8212 |
|            | 800      | 2    | 0.8280 |
|            | 400      | 4    | 0.8301 |
|            | 200      | 8    | 0.8297 |
| 1.01
$$
\times
$$
10^9  | 1600     | 2    | 0.8286 |
|            | 800      | 4    | 0.8323 |
|            | 400      | 8    | 0.8335 |
|            | 200      | 16   | 0.8321 |
| 2.55
$$
\times
$$
10^9  | 1600     | 4    | 0.8365 |
|            | 800      | 8    | 0.8382 |
|            | 400      | 16   | 0.8367 |

**表4：与DLRM相比的在线指标提升**

| 方法            | FLOPs          | 序列长度 | 层数 | 在线指标 |
|-----------------|----------------|----------|------|----------|
| DLRM            | 3.45
$$
\times
$$
10^7 (6
$$
\times
$$
) | -        | -    | +0%      |
| Climber         | 5.82
$$
\times
$$
10^6 (1
$$
\times
$$
) | 100      | 1    | -4.95%   |
|                 | 1.84
$$
\times
$$
10^7 (3
$$
\times
$$
) | 400      | 1    | -1.31%   |
|                 | 3.46
$$
\times
$$
10^7 (6
$$
\times
$$
) | 800      | 1    | -1.22%   |
|                 | 4.11
$$
\times
$$
10^8 (71
$$
\times
$$
)| 400      | 4    | +3.65%   |
|                 | 1.01
$$
\times
$$
10^9 (174
$$
\times
$$
)| 800     | 4    | +4.29%   |
|                 | 2.79
$$
\times
$$
10^9 (479
$$
\times
$$
)| 1600    | 4    | +7.78%   |
|                 | 2.31
$$
\times
$$
10^9 (397
$$
\times
$$
)| 800     | 8    | +10.68%  |
|                 | 3.61
$$
\times
$$
10^9 (620
$$
\times
$$
)| 800     | 12   | +12.19%  |

### 4.5 在线A/B测试（RQ4）

表4总结了我们提出的Climber框架的在线A/B测试结果。与序列长度和层数同等重要的结论一致，我们的模型通过仅调整序列长度和层数，展示了指标和FLOPs的在线扩展曲线。

首先，FLOPs为5.82
$$
\times
$$
10^6的Climber显示出负的指标提升。当Climber(6
$$
\times
$$
)模型的FLOPs值与DLRM(6
$$
\times
$$
)相匹配时，仅有轻微负指标，表明在较少FLOPs下Climber模型的效率低于DLRM。此外，当Climber(479
$$
\times
$$
)模型的FLOPs值为2.79
$$
\times
$$
10^9时，在线指标提升达到+7.78%。最后，当Climber(620
$$
\times
$$
)模型的FLOPs值为3.61
$$
\times
$ $
10^9时，实现了+12.19%的在线指标提升。

在在线推理阶段，Climber在不同序列长度和层数下实现了显著更低的延迟——具体而言，每请求比DLRM快2.92倍到14.38倍。这一加速是通过我们在与DLRM相同的推理预算下所采用的加速技术实现的（第3.2节），因此我们可以部署复杂度高达100倍的模型。

据我们所知，Climber是第一个在保持资源平衡的同时展示离线与在线扩展曲线的推荐模型。此外，它实现了+12.19%的指标提升，代表了过去一年中最大的改进。

---

## 5 结论

我们提出了Climber——一个高效的扩展框架，包含特定的Transformer变体和协同设计的加速技术。我们的模型通过处理多尺度序列、多场景和多兴趣方面，有效降低了计算复杂度并打破了推荐系统的扩展困境。这种集成使得模型在离线评估中相比DLRM和Transformer模型均展现出更优的可扩展性。此外，我们引入了加速技术，该技术采用"单用户多item"样本格式和编码器级KV缓存。这些技术使得在不增加过高计算资源的情况下，部署复杂度高达100倍的模型成为可能。Climber在线上展示了扩展曲线，并实现了12.19%的在线指标提升。总之，这项工作在资源受限的条件下将Transformer架构适配到推荐系统，为生成式推荐范式的下一阶段奠定了基础。未来，我们将探索推荐系统中更多的生成式技术，旨在持续释放扩展潜力。

---


## 参考文献

[1] Newsha Ardalani, Carole-Jean Wu, Zeliang Chen, Bhargav Bhushanam, and Adnan Aziz. 2022. Understanding scaling laws for recommendation models. *arXiv preprint arXiv:2208.08489* (2022).

[2] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. 2023. Qwen-vl: A frontier large vision-language model with versatile abilities. *arXiv preprint arXiv:2308.12966* (2023).

[3] Xiao Bi, Deli Chen, Guanting Chen, Shanhuang Chen, Damai Dai, Chengqi Deng, Honghui Ding, Kai Dong, Qiushi Du, Zhe Fu, et al. 2024. Deepseek llm: Scaling open-source language models with longtermism. *arXiv preprint arXiv:2401.02954* (2024).

[4] Brian Brost, Rishabh Mehrotra, and Tristan Jehan. 2019. The music streaming sessions dataset. In *The World Wide Web Conference*. 2594–2600.

[5] Adam Casson. 2023. Transformer FLOPs. (2023). https://adamcasson.com/posts/transformer-flops

[6] Jianxin Chang, Chenbin Zhang, Zhiyi Fu, Xiaoxue Zang, Lin Guan, Jing Lu, Yiqun Hui, Dewei Leng, Yanan Niu, Yang Song, et al. 2023. TWIN: TWo-stage interest network for lifelong user behavior modeling in CTR prediction at kuaishou. In *Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining*. 3785–3794.

[7] Mehdi Cherti, Romain Beaumont, Ross Wightman, Mitchell Wortsman, Gabriel Ilharco, Cade Gordon, Christoph Schuhmann, Ludwig Schmidt, and Jenia Jitsev. 2023. Reproducible scaling laws for contrastive language-image learning. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*. 2818–2829.

[8] Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. 2022. FlashAttention: Fast and memory-efficient exact attention with io-awareness. *Advances in Neural Information Processing Systems* 35 (2022), 16344–16359.

[9] Harry Dong, Xinyu Yang, Zhenyu Zhang, Zhangyang Wang, Yuejie Chi, and Beidi Chen. 2024. Get More with LESS: Synthesizing Recurrence with KV Cache Compression for Efficient LLM Inference. *arXiv preprint arXiv:2402.09398* (2024).

[10] Wei Guo, Hao Wang, Luankang Zhang, Jin Yao Chin, Zhongzhou Liu, Kai Cheng, Qiushi Pan, Yi Quan Lee, Wanqi Xue, Tingjia Shen, et al. 2024. Scaling New Frontiers: Insights into Large Recommendation Models. *arXiv preprint arXiv:2412.00714* (2024).

[11] Yu-Lin He, Xiao-Liang Zhang, Wei Ao, and Joshua Zhexue Huang. 2018. Determining the optimal temperature parameter for Softmax function in reinforcement learning. *Applied Soft Computing* 70 (2018), 80–85.

[12] Geoffrey Hinton. 2015. Distilling the Knowledge in a Neural Network. *arXiv preprint arXiv:1503.02531* (2015).

[13] Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. 2022. Training compute-optimal large language models. *arXiv preprint arXiv:2203.15556* (2022).

[14] Yupeng Hou, Zhankui He, Julian McAuley, and Wayne Xin Zhao. 2023. Learning vector-quantized item representation for transferable sequential recommenders. In *Proceedings of the ACM Web Conference 2023*. 1162–1171.

[15] Yupeng Hou, Shanlei Mu, Wayne Xin Zhao, Yaliang Li, Bolin Ding, and Ji-Rong Wen. 2022. Towards universal sequence representation learning for recommender systems. In *Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining*. 585–593.

[16] Jie Hu, Li Shen, and Gang Sun. 2018. Squeeze-and-excitation networks. In *Proceedings of the IEEE conference on computer vision and pattern recognition*. 7132–7141.

[17] Tongwen Huang, Zhiqi Zhang, and Junlin Zhang. 2019. FiBiNET: combining feature importance and bilinear feature interaction for click-through rate prediction. In *Proceedings of the 13th ACM conference on recommender systems*. 169–177.

[18] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. 2020. Scaling laws for neural language models. *arXiv preprint arXiv:2001.08361* (2020).

[19] Chao Li, Zhiyuan Liu, Mengmeng Wu, Yuchi Xu, Huan Zhao, Pipei Huang, Guoliang Kang, Qiwei Chen, Wei Li, and Dik Lun Lee. 2019. Multi-interest network with dynamic routing for recommendation at Tmall. In *Proceedings of the 28th ACM international conference on information and knowledge management*. 2615–2623.

[20] Jiacheng Li, Ming Wang, Jin Li, Jinmiao Fu, Xin Shen, Jingbo Shang, and Julian McAuley. 2023. Text is all you need: Learning language representations for sequential recommendation. In *Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining*. 1258–1267.

[21] Aixin Liu, Bei Feng, Bin Wang, Bingxuan Wang, Bo Liu, Chenggang Zhao, Chengqi Dengr, Chong Ruan, Damai Dai, Daya Guo, et al. 2024. Deepseek-v2: A strong, economical, and efficient mixture-of-experts language model. *arXiv preprint arXiv:2405.04434* (2024).

[22] Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. 2024. Deepseek-v3 technical report. *arXiv preprint arXiv:2412.19437* (2024).

[23] Zichang Liu, Aditya Desai, Fangshuo Liao, Weitao Wang, Victor Xie, Zhaozhuo Xu, Anastasios Kyrillidis, and Anshumali Shrivastava. 2024. Scissorhands: Exploiting the persistence of importance hypothesis for llm kv cache compression at test time. *Advances in Neural Information Processing Systems* 36 (2024).

[24] Jinwei Luo, Mingkai He, Xiaolin Lin, Weike Pan, and Zhong Ming. 2022. Dual-task learning for multi-behavior sequential recommendation. In *Proceedings of the 31st ACM international conference on information & knowledge management*. 1379–1388.

[25] Xiao Lv, Jiangxia Cao, Shijie Guan, Xiaoyou Zhou, Zhiguang Qi, Yaqiang Zang, Ming Li, Ben Wang, Kun Gai, and Guorui Zhou. 2024. MARM: Unlocking the Future of Recommendation Systems through Memory Augmentation and Scalable Complexity. *arXiv preprint arXiv:2411.09425* (2024).

[26] Julian McAuley, Christopher Targett, Qinfeng Shi, and Anton Van Den Hengel. 2015. Image-based recommendations on styles and substitutes. In *Proceedings of the 38th international ACM SIGIR conference on research and development in information retrieval*. 43–52.

[27] Dheevatsa Mudigere, Yuchen Hao, Jianyu Huang, Zhihao Jia, Andrew Tulloch, Srinivas Sridharan, Xing Liu, Mustafa Ozdal, Jade Nie, Jongsoo Park, et al. 2022. Software-hardware co-design for fast and scalable training of deep learning recommendation models. In *Proceedings of the 49th Annual International Symposium on Computer Architecture*. 993–1011.

[28] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. *Journal of machine learning research* 21, 140 (2020), 1–67.

[29] Zihua Si, Lin Guan, Zhong Xiang Sun, Xiaoxue Zang, Jing Lu, Yiqun Hui, Xingchao Cao, Zeyu Yang, Yichen Zheng, Dewei Leng, et al. 2024. Twinv2: Scaling ultra-long user behavior sequence modeling for enhanced ctr prediction at kuaishou. In *Proceedings of the 33rd ACM International Conference on Information and Knowledge Management*. 4890–4897.

[30] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. 2023. Llama: Open and efficient foundation language models. *arXiv preprint arXiv:2302.13971* (2023).

[31] Roberto Turrin, Massimo Quadrana, Andrea Condorelli, Roberto Pagano, Paolo Cremonesi, et al. 2015. 30Music Listening and Playlists Dataset. *RecSys Posters* 75 (2015).

[32] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. *Advances in neural information processing systems* 30 (2017).

[33] Shijia Wang, Tianpei Ouyang, Yunfan Zhou, Qiang Xiao, Yintao Ren, Yifei Pan, Fangjian Li, and Chuanjiang Luo. 2025. Enhanced Emotion-aware Music Recommendation via Large Language Models. In *Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2*. 4986–4994.

[34] Shijia Wang, Yi Zheng, Qiang Xiao, Yilong Zhao, Qimeng Yang, and Chuanjiang Luo. 2024. Sparsity-Aware Personalized Pattern Extractor Network for Music Multi-task Learning. In *International Conference on Database Systems for Advanced Applications*. Springer, 352–363.

[35] Yunli Wang, Zixuan Yang, Zhen Zhang, Zhiqiang Wang, Jian Yang, Shiyang Wen, Peng Jiang, and Kun Gai. 2024. Scaling Laws for Online Advertisement Retrieval. *arXiv preprint arXiv:2411.13322* (2024).

[36] Zhibo Xiao, Luwei Yang, Wen Jiang, Yi Wei, Yi Hu, and Hao Wang. 2020. Deep multi-interest network for click-through rate prediction. In *Proceedings of the 29th ACM International Conference on Information & Knowledge Management*. 2265–2268.

[37] Qimeng Yang, Shijia Wang, Da Guo, Dongjin Yu, Qiang Xiao, Dongjing Wang, and Chuanjiang Luo. 2024. Cascading Multimodal Feature Enhanced Contrast Learning for Music Recommendation. In *2024 IEEE International Conference on Data Mining (ICDM)*. IEEE, 905–910.

[38] Jiaqi Zhai, Lucy Liao, Xing Liu, Yueming Wang, Rui Li, Xuan Cao, Leon Gao, Zhaojie Gong, Fangda Gu, Michael He, et al. 2024. Actions speak louder than words: Trillion-parameter sequential transducers for generative recommendations. *arXiv preprint arXiv:2402.17152* (2024).

[39] Xiaohua Zhai, Alexander Kolesnikov, Neil Houlsby, and Lucas Beyer. 2022. Scaling vision transformers. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*. 12104–12113.

[40] Buyun Zhang, Liang Luo, Yuxin Chen, Jade Nie, Xi Liu, Daifeng Guo, Yanli Zhao, Shen Li, Yuchen Hao, Yantao Yao, et al. 2024. Wukong: Towards a Scaling Law for Large-Scale Recommendation. *arXiv preprint arXiv:2403.02545* (2024).

[41] Gaowei Zhang, Yupeng Hou, Hongyu Lu, Yu Chen, Wayne Xin Zhao, and Ji-Rong Wen. 2024. Scaling law of large sequential recommendation models. In *Proceedings of the 18th ACM Conference on Recommender Systems*. 444–453.
