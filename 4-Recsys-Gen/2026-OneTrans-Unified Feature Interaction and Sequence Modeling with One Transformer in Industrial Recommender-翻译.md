# OneTrans: 统一特征交互与序列建模——工业级推荐中的单Transformer架构

> haolei.pei@bytedance.com

## OneTrans: Unified Feature Interaction and Sequence Modeling with One Transformer in Industrial Recommender
Haolei Pei∗
本文介绍了 OneTrans: 统一特征交互与序列建模——工业级推荐中的单Transformer架构。核心内容：
关键发现：
Zhaoqi Zhang∗
字节跳动
南洋理工大学
新加坡
字节跳动
新加坡
zhaoqi.zhang@bytedance.com
Jun Guo∗
字节跳动
新加坡
jun.guo@bytedance.com
Tianyu Wang
字节跳动
新加坡
tianyu.wang01@bytedance.com
Yufei Feng
字节跳动
中国杭州
fengyihui@bytedance.com
Hui Sun
字节跳动
中国杭州
sunhui.sunh@bytedance.com
Shaowei Liu†
字节跳动
新加坡
liushaowei.nphard@bytedance.com
Aixin Sun†
南洋理工大学
新加坡
axsun@ntu.edu.sg
## 摘要
在推荐系统中，扩展特征交互模块（如Wukong、RankMixer）或用户行为序列模块（如LONGER）已取得了显著成功。然而，这些工作通常沿着各自独立的轨道推进，这不仅阻碍了双向信息交换，还阻止了统一的优化和扩展。在本文中，我们提出了OneTrans，一个统一的Transformer主干网络，能够同时执行用户行为序列建模和特征交互。OneTrans采用统一的分词器，将序列属性和非序列属性转换为单一的token序列。堆叠的OneTrans块在相似的序列token之间共享参数，同时为非序列token分配token特定的参数。通过因果注意力和跨请求KV缓存，OneTrans实现了中间表示的预计算和缓存，显著降低了训练和推理过程中的计算成本。在工业级规模数据集上的实验结果表明，OneTrans能够随参数增加高效扩展，始终优于强基线，并在在线A/B测试中实现了每位用户GMV 5.68%的提升。
**CCS概念**：• 信息系统 $\to$ 信息检索；推荐系统；
**关键词**：推荐系统，排序模型，规模定律
*这些作者贡献相同。
†通讯作者。
本作品采用知识共享署名4.0国际许可协议。
Conference acronym 'XX, Woodstock, NY
© 2025 版权归作者/权利人所有。
ACM ISBN 978-1-4503-XXXX-X/18/06
**ACM引用格式**：
Zhaoqi Zhang, Haolei Pei, Jun Guo, Tianyu Wang, Yufei Feng, Hui Sun, Shaowei Liu, and Aixin Sun. 2025. OneTrans: 统一特征交互与序列建模——工业级推荐中的单Transformer架构. 见《请从权利确认邮件中填入正确的会议名称》(Conference acronym 'XX). ACM, 纽约, 美国, 9页.
## 1 引言
推荐系统在各种信息服务中发挥着基础性作用，例如电子商务[9, 35]、流媒体[2, 20, 28]和社交网络[31, 32]。工业级推荐系统通常采用级联排序架构[6, 16, 22]。首先，召回阶段从十亿级语料库中选取数百个候选[13, 36]。然后，排序阶段对每个候选进行评分并返回Top-Kitem[11, 27, 28, 32, 37]。深度学习推荐模型（DLRM）[19]被广泛应用于工业级推荐器的排序阶段。
本文将聚焦于排序阶段，遵循DLRM风格的排序范式。对于排序而言，主流方法围绕两个独立的模块进行迭代：(a) 序列建模，通过局部注意力或Transformer编码器将用户多行为序列编码为候选感知的表示[1, 14, 25, 35]；(b) 特征交互，通过因子分解、显式交叉网络或特征组上的注意力来学习非序列特征（如用户画像、item画像和上下文）之间的高阶交叉[11, 12, 27, 37]。如图1(a)所示，这些方法通常将用户行为编码为压缩的序列表示，然后将其与非序列特征拼接，并应用特征交互模块来学习更高阶的交互；本文将这种设计称为"编码-后-交互"流水线。
大语言模型（LLM）的成功表明，扩展模型规模（如参数量、训练数据）能够带来可预测的性能提升[15]，这启发了推荐系统中的类似研究[1, 32, 37]。在特征交互方面，Wukong[32]堆叠了带线性压缩的因子分解机模块来捕捉高阶特征交互，并建立了规模定律；RankMixer[37]则通过硬件友好的token混合与token特定的前馈网络实现了良好的规模扩展。在序列建模方面，LONGER[1]将因果Transformer应用于长用户历史，并表明扩展深度和宽度能够带来单调的性能提升。尽管在实践中有效，但将序列建模和特征交互作为独立模块分离带来了两个主要局限性。首先，"编码-后-交互"流水线限制了双向信息流，限制了静态/上下文特征如何塑造序列表示[30]。其次，模块分离导致执行碎片化并增加延迟，而单一的Transformer风格主干网络可以复用LLM优化（如KV缓存、内存高效注意力和混合精度），从而实现更有效的扩展[13]。
在本文中，我们提出OneTrans，一种创新的架构范式，采用统一的Transformer主干网络，联合执行用户行为序列建模和特征交互。如图1(b)所示，OneTrans在统一的主干网络内实现了双向信息交换。它采用统一的分词器，将序列特征（多样化的行为序列）和非序列特征（静态用户/item特征及上下文特征）转换为单一的token序列，然后由金字塔式堆叠的OneTrans块（一种为工业级推荐系统量身定制的Transformer变体）处理。为了适应推荐系统中多样化的token来源（与LLM中纯文本token不同），每个OneTrans块采用了类似于HiFormer[11]的混合参数化策略。具体来说，所有序列token（来自序列特征）共享一组Q/K/V和FFN权重，而每个非序列token（来自非序列特征）则获得token特定的参数以保持其独特的语义。
与传统的"编码-后-交互"框架不同，OneTrans通过统一的因果Transformer主干网络消除了序列特征和非序列特征之间的架构障碍。这一方案使推荐系统的规模扩展与LLM实践保持一致：整个模型可以通过调整主干网络的深度和宽度进行扩展，同时无缝继承成熟的LLM优化技术，如FlashAttention[7]和混合精度训练[17]。
图1：架构对比。(a) 传统的"编码-后-交互"流水线：对序列特征进行编码，合并非序列特征，然后送入后置的特征交互块。(b) OneTrans：在单个OneTrans（Transformer风格）堆栈中对序列和非序列特征进行联合建模。
特别是，跨候选和跨请求的KV缓存[1]将会话的时间复杂度从O(C)降低到O(1)，使得大规模部署OneTrans成为可能。
总之，我们的主要贡献有四个方面：(1) **统一框架**。我们提出了OneTrans，一个用于排序的单一Transformer主干网络，配备统一的分词器，将序列和非序列特征编码为一个token序列，以及一个统一的Transformer块，联合执行序列建模和特征交互。(2) **推荐系统定制化**。为了弥合LLM和推荐系统任务之间的差距，OneTrans引入了一种混合参数化方案，为多样化的非序列token分配token特定的参数，同时为所有序列token共享参数。(3) **高效训练和服务**。我们通过金字塔策略（逐步剪枝序列token）和跨请求KV缓存（跨候选复用用户侧计算）来提升效率。此外，我们采用LLM优化技术如FlashAttention、混合精度训练和半精度推理，以进一步减少内存和计算。(4) **扩展与部署**。OneTrans在模型规模增加时表现出近乎对数线性的性能提升，为实际生产数据中的规模定律提供了证据。在线部署时，它在保持生产级延迟的同时，实现了业务关键指标上的统计显著提升。
## 2 相关工作
早期的推荐系统如DIN[35]及其会话感知变体DSIN[9]使用局部注意力来学习候选条件下的用户历史摘要，但将行为压缩为每个候选的固定长度向量，限制了长距离依赖建模[34]。自注意力方法如SASRec[14]、BERT4Rec[25]和BST[4]通过让每个位置关注整个历史来消除这一瓶颈，并通过双向掩码提高了样本效率。最近，随着推荐系统中规模定律[15]的不断探索，LONGER[1]通过高效注意力和服务友好型设计来面向超长行为历史，将序列建模推向工业级规模。然而，在主流流水线中，这些序列编码器通常与特征交互堆栈分离，导致后期融合而非与静态上下文特征的联合优化[30]。
在特征交互方面，早期推荐系统依赖于人工设计的交叉特征或自动化的乘法交互层。经典模型如Wide&Deep[5]、FM/DeepFM[3, 12]和DCN/DCNv2[26, 27]提供了高效的低阶或有界阶交互。然而，正如最近的规模扩展研究所观察到的[32]，一旦模型堆叠了足够多的交叉层，增加更多层就不再有帮助：模型质量趋于平稳而不是持续改善。为了克服预设交叉形式的僵化性，基于注意力的方法自动学习高阶交互。AutoInt[24]学习任意阶的关系，HiFormer[11]引入组特定的投影以更好地捕捉异构、非对称的交互。随着特征交互模块的不断扩展，诸如Wukong[32]这样的大规模系统通过堆叠带有线性压缩的FM风格交互块展示了可预测的收益，而RankMixer[37]通过并行token混合和稀疏MoE在严格的延迟预算下实现了有利的扩展。然而，这些交互模块通常遵循"编码-后-交互"范式，将交互推向一个独立的阶段，并阻碍了与用户序列建模的统一优化[30]。
迄今为止，推荐系统的进展主要沿着两条独立的轨道推进：序列建模和特征交互。InterFormer[30]试图通过基于摘要的双向交叉架构来弥合这一差距，实现在两个组件之间的相互信号交换。然而，它仍然将它们作为独立模块维护，交叉架构引入了架构复杂性和执行碎片化。如果没有统一的主干网络进行联合建模和优化，将系统作为一个整体进行扩展仍然具有挑战性。
最近关于生成式推荐系统（GR）的工作将推荐视为序列转导，并提出了高效的长上下文主干网络，如HSTU[31]。这一方向与依赖丰富非序列特征的DLRM是互补的。
## 3 方法
在详细介绍我们的方法之前，我们简要描述任务设置。在级联的工业级推荐系统中，每次召回阶段为某个用户u返回一个候选集（通常包含数百个候选item）。然后排序模型对每个候选itemi预测一个分数：
ŷ_u,i = f(i ⊕ NS, S; $\Theta$ )   (1)
其中，NS是从用户、候选item和上下文中派生的一组非序列特征；S是来自用户的一组历史行为序列； $\Theta$ 是可训练参数。常见的任务预测包括点击率（CTR）和点击后转化率（CVR）。
CTR_u,i = P(click=1 | NS, S; $\Theta$ ),   CVR_u,i = P(conv=1 | click=1, NS, S; $\Theta$ )   (2)
### 3.1 OneTrans框架概述
如图2(a)所示，OneTrans采用统一的分词器，将序列特征S映射为Stoken（S-tokens），将非序列特征NS映射为NStoken（NS-tokens）。然后，金字塔式堆叠的Transformer在单个计算图中联合处理统一的token序列。我们记初始token序列为：
X^(0) = [S-tokens ; NS-tokens] $\in$ R^(L_S + L_NS) $\times$ d   (3)
该token序列通过拼接L_S个Stoken和L_NS个NStoken构建而成，所有token的维度均为d。注意，Stoken中包含可学习的[SEP]token，用于分隔不同类型的用户行为序列。如图2(b)所示，每个OneTrans块通过以下方式逐步精炼token状态：
Z^(n) = MixedMHA(Norm(X^(n-1))) + X^(n-1)   (4)
X^(n) = MixedFFN(Norm(Z^(n))) + Z^(n)   (5)
这里，MixedMHA（混合多头注意力）和MixedFFN（混合前馈网络）采用了混合参数化策略（见图2(c)），在序列token之间共享权重，同时在注意力和前馈层中为非序列token分配独立的参数。
统一的因果掩码强制执行自回归约束，限制每个位置只能关注前面的token。具体来说，NStoken被允许关注整个Stoken历史，从而实现全面的跨token交互。通过堆叠这样的块，并对Stoken应用金字塔式的尾部截断，模型逐步将紧凑的高阶信息蒸馏到NStoken中。最终的token状态然后传递给任务特定的预测头。
通过将非序列和序列特征统一为一个token序列，并用因果Transformer对其进行建模，OneTrans脱离了传统的"编码-后-交互"流水线。这种统一设计自然实现了：(i) 每个行为序列内的序列内交互，(ii) 跨多个序列的跨序列交互，(iii) item、用户和上下文特征之间的多源特征交互，以及(iv) 序列-特征交互，所有这些都在单个Transformer堆栈中完成。
统一的方案使我们能够无缝继承成熟的LLM工程优化，包括KV缓存和内存高效注意力，从而显著降低推理延迟。我们认为这种统一的公式非常适合在单一、可扩展的架构中应对多序列和跨域推荐挑战。接下来，我们将详细描述设计。
### 3.2 特征与分词
为了构建初始token序列X^(0)，OneTrans首先应用特征预处理流水线，将所有原始特征输入映射为嵌入向量。这些嵌入然后被划分为：(i) 多行为序列子集和(ii) 表示用户、item或上下文特征的非序列子集。对每个子集应用独立的分词器。
#### 3.2.1 非序列分词
非序列特征NS包括数值输入（如价格、CTR）和类别输入（如用户ID、item类别）。所有特征要么被分桶编码，要么被独热编码，然后进行嵌入。由于工业级系统通常涉及数百个重要性各异的特征，控制非序列token数量（记为L_NS）有两种选择：
**分组分词器（与RankMixer[37]对齐）**。特征被手动划分为语义组{g1, ..., g_L_NS}。每组被拼接并通过组特定的MLP：
NS-tokens = [MLP1(concat(g1)), ..., MLP_L_NS(concat(g_L_NS))]   (6)
**自动分割分词器**。或者，所有特征被拼接并由单个MLP投影一次，然后进行分割：
NS-tokens = split(MLP(concat(NS)), L_NS)   (7)
自动分割分词器相比分组方法减少了内核启动开销，因为它使用了单一的稠密投影。我们将通过实验评估这两种选择。
最终，非序列分词产生L_NS个非序列token，每个维度为d。
#### 3.2.2 序列分词
OneTrans接受多行为序列作为输入：
S = {S1, ..., Sn}   (8)
每个序列Si由L_i个事件嵌入e组成，通过拼接itemID及其对应的辅助信息（如item类别和价格）构建而成：
Si = [e_i1, ..., e_iL_i]   (9)
多行为序列的原始维度可能不同。因此，对于每个序列Si，我们使用一个共享的投影MLP_i将其所有事件e_ij转换为统一的维度d：
~S_i = [MLP_i(e_i1), ..., MLP_i(e_iL_i)] $\in$ R^(L_i $\times$ d)   (10)
对齐后的序列~S_i通过以下两种规则之一合并为单个token序列：(1) **时间戳感知**：按时间交错所有事件，附带序列类型指示符；(2) **时间戳无关**：按事件影响度拼接序列，例如 购买 $\to$ 加购 $\to$ 点击，在序列之间插入可学习的[SEP]token。在后一种方案中，用户意图更强的行为被放置在序列更靠前的位置。消融实验结果表明，当时间戳可用时，时间戳感知规则优于影响度排序方案。
形式化地，我们有：
S-Tokens = Merge(~S1, ..., ~Sn) $\in$ R^(L_S $\times$ d),   L_S = $\Sigma$ ^n_i=1 L_i + L_SEP   (11)
### 3.3 OneTrans块
如图2(b)所示，每个OneTrans块是一个应用于归一化token序列的预层归一化因果Transformer：L_S个序列Stoken，后接L_NS个非序列NStoken。受异构特征组发现[11]的启发，我们对Transformer进行了轻量级修改以允许混合参数方案，见图2(c)。具体来说，同质的Stoken共享一组参数。而异构的NStoken（来源/语义各不相同）则获得token特定的参数。
与LLM输入不同，推荐系统中的token序列将序列Stoken与值范围和统计量差异很大的多样化NStoken结合在一起。后层归一化设置可能会因这些差异而导致注意力坍塌和训练不稳定。为防止这种情况，我们对所有token应用RMSNorm[33]作为预层归一化，以对齐各token类型的尺度并稳定优化。
#### 3.3.1 混合（共享/token特定）因果注意力
OneTrans采用标准的多头注意力（MHA）以及因果注意力掩码；唯一的改变是Q/K/V的参数化方式。设x_i $\in$ R^d为第i个token。为了计算Q/K/V，我们对Stoken（i $\leq$ L_S）使用共享投影，对NStoken（i > L_S）使用L_NS个token特定的投影：
(q_i, k_i, v_i) = (W^Q_i x_i, W^K_i x_i, W^V_i x_i)   (12)
其中W^ $\Psi$ _i（ $\Psi$ $\in$ {Q, K, V}）遵循混合参数化方案：
W^ $\Psi$ _i = { W^ $\Psi$ _S (i $\leq$ L_S, 共享于Stoken), W^ $\Psi$ _NS,i (i > L_S, token特定于NStoken) }   (13)
注意力使用标准因果掩码，NStoken放置在Stoken之后。这导致了：(1) **S侧**。每个Stoken只关注前面的S位置。对于时间戳感知序列，每个事件以其历史为条件；对于时间戳无关序列（按意图排序，如 购买 $\to$ 加购 $\to$ 点击/曝光），因果掩码让高意图信号通知并过滤后面的低意图行为。(2) **NS侧**。每个NStoken关注整个S历史（本质上是序列证据的目标注意力聚合），以及前面的NStoken，增加了token级交互的多样性。(3) **金字塔支持**。在S侧和NS侧，因果掩码逐步将信息集中到后面的位置，自然支持逐层剪枝token的金字塔调度，稍后将详述。
#### 3.3.2 混合（共享/token特定）FFN
类似地，前馈网络遵循相同的参数化策略：为NStoken使用token特定的FFN，为Stoken使用共享的FFN：
MixedFFN(x_i) = W2_i $\phi$ (W1_i x_i)   (14)
这里W1_i和W2_i遵循公式(13)的混合参数化，即i $\leq$ L_S时共享，i > L_S时token特定。
总之，相对于标准的因果Transformer，OneTrans只改变了参数化方式：NStoken使用token特定的QKV和FFN；Stoken共享单组参数。单个因果掩码将序列联系在一起，允许NStoken聚合整个行为历史，同时保持高效的Transformer风格计算。
### 3.4 金字塔堆叠
如第3.3节所述，因果掩码将信息集中到后面的位置。利用这种近期结构，我们采用金字塔调度：在每个OneTrans块层中，只有最近Stoken的子集发出查询，而键/值仍在整个序列上计算；查询集随着深度增加而缩小。
设X = {x_i}^L_i=1为输入token列表，Q = {L-L'+1, ..., L}为尾部索引集，其中L' $\leq$ L。遵循公式(13)，我们将查询修改为i $\in$ Q：
q_i = W^Q_i x_i, i $\in$ Q   (15)
而键和值照常在整个序列{1, ..., L}上计算。注意力之后，只保留i $\in$ Q的输出，将token长度减少到L'，从而在各层之间形成金字塔层次结构。
该设计带来两个好处：(i) **渐进式蒸馏**：长行为历史被汇聚到少量尾部查询中，将容量集中在最信息量的事件上，并将信息整合到NStoken中；(ii) **计算效率**：注意力成本变为O(L L' d)，FFN随L'线性扩展。缩小查询集直接减少了FLOPs和激活内存。
### 3.5 训练与部署优化
#### 3.5.1 跨请求KV缓存
在工业级推荐系统中，来自同一请求的样本在训练和服务期间连续处理：它们的Stoken在不同候选之间保持不变，而NStoken因候选item而异。利用这一结构，我们将广泛采用的KV缓存[1]集成到OneTrans中，形成一个统一的两阶段范式。
**阶段I（S侧，每个请求一次）**。使用因果掩码处理所有Stoken，并缓存它们的键/值对和注意力输出。此阶段每个请求执行一次。
**阶段II（NS侧，每个候选）**。对于每个候选，计算其NStoken，并对缓存的S侧键/值执行交叉注意力，随后是token特定的FFN层。特别地，候选特定的序列（如SIM[21]）通过池化预聚合成NStoken，因为它们无法复用共享的S侧缓存。
KV缓存将S侧计算摊分到多个候选上，使得每个候选的工作量保持轻量，并消除了冗余计算，从而大幅提升吞吐量。
由于用户行为序列是仅追加的，我们将KV缓存扩展到跨请求复用：每个新请求复用了之前的缓存，仅计算新增行为的增量键/值。这将每个请求的序列计算从O(L)降低到O( $\Delta$ L)，其中 $\Delta$ L是自上次请求以来的新行为数量。
#### 3.5.2 统一的LLM优化
我们采用FlashAttention-2[8]通过分块和内核融合来减少普通注意力的I/O和二次激活占用，从而在训练和推理中获得更低的内存使用和更高的吞吐量。为进一步缓解内存压力，我们使用混合精度训练（BF16/FP16）[18]以及激活重计算[10]，后者丢弃部分前向激活并在反向传播期间重新计算。这种组合用适度的额外计算换取了大量的内存节省，使得在不改变架构的情况下能够使用更大的批次和更深的模型。
## 4 实验
通过离线评估和在线测试，我们旨在回答以下研究问题（RQ）：**RQ1：统一堆栈 vs. 编码-后-交互**。在可比计算量下，单一Transformer堆栈是否带来一致的性能提升？**RQ2：哪些设计选择至关重要？** 我们对输入层（如分词器、序列融合）和OneTrans块（如参数共享、注意力类型、金字塔堆叠）进行消融实验，以评估不同设计选择对性能和效率的重要性。**RQ3：系统效率**。金字塔堆叠、跨请求KV缓存、FlashAttention-2以及混合精度加重计算是否在相同的OneTrans图下减少FLOPs/内存和延迟？**RQ4：规模定律**。随着长度（token序列长度）、宽度（d_model）、深度（层数）的扩展，损失/性能是否表现出预期的对数线性趋势？**RQ5：在线A/B测试**。在线部署OneTrans是否在生产延迟约束下，在关键业务指标（如人均订单量、人均GMV）上产生统计显著的提升？
### 4.1 实验设置
#### 4.1.1 数据集
对于离线评估，我们在大规模工业级排序场景下使用生产日志评估OneTrans，严格遵守隐私合规要求（所有个人身份信息均已匿名化和哈希处理）。数据按时间顺序划分，所有特征在曝光时刻快照，以防止时间泄漏并确保线上-线下一致性。标签（如点击和订单）在与生产设置对齐的固定窗口内聚合。表1总结了数据集统计信息。
**表1：OneTrans实验的数据集概览**
| 指标 | 值 |
|------|------|
| 曝光量（样本数） | 291亿 |
| 用户数（唯一） | 2790万 |
| item数（唯一） | 1020万 |
| 每日曝光量（均值 $\pm$ 标准差） | 1.182亿 $\pm$ 1430万 |
| 每日活跃用户（均值 $\pm$ 标准差） | 230万 $\pm$ 30万 |
#### 4.1.2 任务与指标
我们按照公式(2)评估两个二分类排序任务：CTR和CVR。性能通过AUC和UAUC（曝光加权用户级AUC）衡量。
**下一批次评估**。数据按时间顺序处理。对于每个小批次，我们(i)在评估模式下记录预测结果，然后(ii)在同一批次上训练。AUC和UAUC每天根据当天的预测计算，最后按天进行宏平均。
**效率指标**。我们报告参数量（不包括稀疏嵌入的模型参数）和TFLOPs（批次大小为2048时的训练计算量）。
#### 4.1.3 基线方法
我们使用相同的特征和匹配的计算预算构建行业标准的模型组合作为基线。在"编码-后-交互"范式下，我们从广泛使用的生产基线DCNv2+DIN[27, 35]出发，逐步增强特征交互模块：DCNv2 $\to$ Wukong[32] $\to$ HiFormer[11] $\to$ RankMixer[37]。在固定RankMixer的基础上，我们改变序列建模模块：StackDIN $\to$ Transformer[4] $\to$ LONGER[1]。
#### 4.1.4 超参数设置
我们报告两种配置：**OneTransS**使用6个堆叠的OneTrans块，宽度d=256，H=4个注意力头，目标参数约1亿。**OneTransL**扩展到8层，宽度d=384。
输入通过统一的分词器处理（多行为序列的时间戳感知融合；非序列特征的自动分割分词器）以及启发式金字塔调度，该调度在每层将序列查询token的数量线性缩减：OneTransS从1190缩减到12；OneTransL从1500缩减到16。具体地，我们在各层间线性减少序列查询token数量，将每层的token数四舍五入到最近的32的倍数，并将顶层设置为与非序列token数量匹配。
**优化与基础设施**。我们采用无双衰减的双优化器策略：稀疏嵌入使用Adagrad优化（ $\beta$ 1=0.1, $\beta$ 2=1.0），稠密参数使用RMSProp优化（lr=0.005, alpha=0.99999, momentum=0）。我们应用大规模Transformer训练中常用的稳定化技术，包括预层归一化[29]和全局梯度范数裁剪[23]。训练时每GPU批次大小设置为2048，稠密层梯度裁剪阈值为90，稀疏层为120。在线推理时，我们采用每GPU更小的批次大小100以平衡吞吐量和延迟。训练使用16块H100 GPU进行数据并行全归约。
### 4.2 RQ1：性能评估
我们以DCNv2+DIN为基准进行对比，这是我们场景中预扩展的生产基线（表2）。在"编码-后-交互"范式下，独立扩展任一组件都是有益的：升级特征交互模块（DCNv2 $\to$ Wukong $\to$ HiFormer $\to$ RankMixer）或序列建模模块（StackDIN $\to$ Transformer $\to$ LONGER）在CTR AUC/UAUC和CVR AUC上都能带来一致的提升。在我们的系统中，这些指标上超过+0.1%的提升被认为是有意义的，而超过+0.3%的提升通常对应在线A/B测试中的统计显著效果。然而，CVR UAUC由于每位用户样本量较小且波动性更高，需要谨慎对待。
转向统一设计，OneTransS相对于基线提升了+1.13%/+1.77%（CTR AUC/UAUC）和+0.90%/+1.66%（CVR AUC/UAUC）。在可比的参数规模下，它还在相近的训练FLOPs下（2.64T vs 2.51T）优于RankMixer+Transformer，展示了统一建模的优势。进一步扩展，OneTransL实现了最大的整体提升：+1.53%/+2.79%（CTR AUC/UAUC）和+1.14%/+3.23%（CVR AUC/UAUC），显示出随模型容量增长的可预测质量提升。总之，在单一Transformer中统一序列建模和特征交互比独立扩展任一组件能带来更可靠且计算效率更高的改进。
**表2：离线效果（CTR/CVR）和效率；更高的AUC/UAUC更好。*表示在我们的生产中按时间顺序部署的模型：DCNv2+DIN $\to$ RankMixer+DIN $\to$ RankMixer+Transformer $\to$ OneTransS $\to$ OneTransL**
| 类型 | 模型 | CTR AUC $\uparrow$ | CTR UAUC $\uparrow$ | CVR(order) AUC $\uparrow$ | CVR(order) UAUC $\uparrow$ | 参数量(M) | TFLOPs |
|------|------|-----------|-----------|------------------|-------------------|-----------|--------|
| (1) 基础模型 | DCNv2 + DIN (base)* | 0.79623 | 0.71927 | 0.90361 | 0.71955 | 10 | 0.06 |
| (2) 特征交互 | Wukong + DIN | +0.08% | +0.11% | +0.14% | +0.11% | 28 | 0.54 |
| | HiFormer + DIN | +0.11% | +0.18% | +0.23% | -0.20% | 108 | 1.35 |
| | RankMixer + DIN* | +0.27% | +0.36% | +0.43% | +0.19% | 107 | 1.31 |
| (3) 序列建模 | RankMixer + StackDIN | +0.40% | +0.37% | +0.63% | -1.28% | 108 | 1.43 |
| | RankMixer + LONGER | +0.49% | +0.59% | +0.47% | +0.44% | 109 | 1.87 |
| | RankMixer + Transformer* | +0.57% | +0.90% | +0.52% | +0.75% | 109 | 2.51 |
| (4) 统一框架 | OneTransS* | +1.13% | +1.77% | +0.90% | +1.66% | 91 | 2.64 |
| | OneTransL (default)* | +1.53% | +2.79% | +1.14% | +3.23% | 330 | 8.62 |
### 4.3 RQ2：通过消融实验研究设计选择
我们对提出的OneTrans模型进行消融实验，以量化关键设计选择的贡献。完整结果总结在表3中。我们评估以下变体：
**输入变体**：(i) 将自动分割分词器替换为分组分词器（行1）；(ii) 使用时间戳无关的融合策略替代时间戳感知的序列融合（行2）；(iii) 在时间戳感知的序列融合中移除[SEP]token（行3）；
**OneTrans块变体**：(i) 在所有token之间共享单组Q/K/V和FFN参数，而不是为NStoken分配独立参数（行4）；(ii) 用全注意力替换因果注意力（行5）；(iii) 通过在所有层保持完整token序列来禁用金字塔堆叠（行6）。
**表3：输入设计和OneTrans块设计选择的影响，以OneTransS模型为参考**
| 类型 | 变体 | CTR AUC $\uparrow$ | CTR UAUC $\uparrow$ | CVR(order) AUC $\uparrow$ | CVR(order) UAUC $\uparrow$ | 参数量(M) | TFLOPs |
|------|------|-----------|-----------|------------------|-------------------|-----------|--------|
| 输入 | 分组分词器 | -0.10% | -0.15% | -0.30% | -0.29% | 78 | 2.35 |
| | 时间戳无关融合 | -0.09% | +0.00% | -0.22% | +0.01% | 91 | 2.64 |
| | 时间戳无关融合无Septoken | -0.13% | -0.05% | -0.32% | +0.06% | 91 | 2.62 |
| OneTrans块 | 共享参数 | -0.12% | -0.14% | -0.10% | -0.29% | 24 | 2.64 |
| | 全注意力 | -0.20% | -0.03% | -0.21% | +0.06% | 91 | 2.64 |
| | 无金字塔堆叠 | -0.29% | -0.04% | -0.33% | -0.42% | 92 | 8.08 |
总之，消融实验表明：(1) **自动分割分词器**相对于手动将非序列特征分组为token提供了明显优势，表明让模型自动构建非序列token比依赖人工定义的特征分组更有效；(2) **时间戳感知融合**在时间戳存在时优于基于意图的排序，表明时间顺序应优先于事件影响度；(3) 在时间戳无关融合下，可学习的[SEP]token帮助模型分隔序列；(4) **NStoken的token特定参数**优于共享投影，能够实现更好的特征区分；(5) 因果注意力和全注意力表现相似，但全注意力禁用了标准优化如KV缓存；(6) 在所有层保持完整长度token没有益处：OneTrans有效地将信息总结到少量尾部中，因此金字塔设计可以安全地剪枝查询以节省计算。此外，在固定TFLOPs预算下，金字塔设计相比全长设计支持近1.75倍的更长序列，更好地利用长度扩展带来的收益。
### 4.4 RQ3：系统效率
为了量化第3.5节中的优化，我们在未优化的OneTransS基线上对其进行了消融，并在表4中报告训练/推理指标。
**表4：各变体相对于未优化OneTransS的影响。内存为峰值GPU使用量。**
| 变体 | 训练运行时间(ms) | 训练内存(GB) | 推理延迟(p99;ms) | 推理内存(GB) |
|------|-----------------|-------------|-----------------|-------------|
| 未优化的OneTransS | 407 | 53.13 | 54.00 | 1.70 |
| + 金字塔堆叠 | -28.7% | -42.6% | -8.4% | -6.9% |
| + 跨请求KV缓存 | -30.2% | -58.4% | -29.6% | -52.9% |
| + FlashAttention | -50.1% | -58.9% | -12.3% | -11.6% |
| + 混合精度与重计算 | -32.9% | -49.0% | -69.1% | -30.0% |
如表所示，(i) **金字塔堆叠**通过剪枝序列查询token减少了训练成本（运行时间/内存）和服务开销（p99延迟/内存）；(ii) **跨请求KV缓存**消除了冗余的序列计算，在训练和服务中持续改善了运行时间/延迟和内存；(iii) **FlashAttention**带来了显著的训练增益和适度的服务改进；(iv) **混合精度与重计算**提供了最大的服务增益（p99延迟和推理内存），同时也提高了训练效率。
这些结果展示了LLM优化技术对大规模推荐系统的有效性。基于这些结果，我们扩展到OneTransL，并证明其保持了与更小的DCNv2+DIN基线相当的在线效率（表5），突显了统一的Transformer主干网络能够直接采用LLM优化技术。
**表5：OneTransL与DCNv2+DIN基线的关键效率对比**
| 指标 | DCNv2+DIN | OneTransL |
|------|-----------|-----------|
| TFLOPs | 0.06 | 8.62 |
| 参数量(M) | 10 | 330 |
| MFU | 13.4 | 30.8 |
| 推理延迟(p99, ms) | 13.6 | 13.2 |
| 训练内存(GB) | 20 | 32 |
| 推理内存(GB) | 1.8 | 0.8 |
### 4.5 RQ4：规模定律验证
我们沿三个维度探索OneTrans的规模定律：(1) **长度** - 输入token序列长度，(2) **深度** - 堆叠块的数量，(3) **宽度** - 隐藏状态维度。
如图3(a)所示，增加长度通过引入更多行为证据带来了最大的收益。在深度和宽度之间，我们观察到明显的权衡：增加深度通常比单纯增加宽度带来更大的性能提升，因为更深的堆栈能够提取更高阶的交互和更丰富的抽象表示。然而，更深的模型也会增加串行计算，而增加宽度更有利于并行化。因此，在深度和宽度之间的选择应平衡性能收益和在目标硬件预算下的系统效率。
图3：(a) 权衡：FLOPs vs. $\Delta$ UAUC (b) 规模定律： $\Delta$ UAUC vs. FLOPs (对数)
我们进一步通过同时加宽和加深OneTrans来分析规模定律行为，并作为对比，在RankMixer侧扩展RankMixer+Transformer基线直至1B参数；然后在对数坐标上绘制 $\Delta$ UAUC与训练FLOPs的关系。如图3(b)所示，OneTrans和RankMixer都表现出清晰的对数线性趋势，但OneTrans的斜率更陡，这可能是因为RankMixer中心的扩展缺乏统一主干网络，且其基于MoE的扩展主要加宽了FFN隐藏维度。这些结果共同表明，OneTrans在参数和计算方面更高效，为工业级部署提供了有利的性能-计算权衡。
虽然我们可以在严格的在线p99延迟约束下部署OneTransL，但在该范围之外的显著扩展仍受在线效率的限制，我们将进一步的系统-模型协同优化留给未来的工作。
### 4.6 RQ5：在线A/B测试
我们在两个大规模工业级场景中评估OneTrans的业务影响：(i) **Feeds**（首页信息流），和(ii) **Mall**（包含Feeds和其他子场景的总体设置）。流量通过哈希在用户/账户级别进行分割，并采用用户级随机化。对照组和实验组模型均使用过去1.5年的生产数据进行训练和部署，以确保公平比较。
我们之前的生产基线RankMixer+Transformer作为对照（约1亿神经网络参数），不使用序列KV缓存。实验组部署了OneTransL，采用了第3.5节中描述的服务优化。
我们报告用户级点击量/用户（click/u）、订单量/用户（order/u）和GMV/用户（gmv/u）相对于RankMixer+Transformer对照组的相对变化（ $\Delta$ %），使用双尾95%置信区间（用户级分层自助法），以及端到端延迟（以p99每次曝光时间从请求到达至响应发出的相对变化衡量， $\Delta$ %，越低越好）。如表6所示，OneTransL带来了一致的收益。在Feeds场景中，它实现了+7.737%的点击量/用户、+4.351%的订单量/用户、+5.685%的GMV/用户和-3.91%的延迟。在Mall场景中，它实现了+5.143%的点击量/用户、+2.577%的订单量/用户、+3.670%的GMV/用户和-3.26%的延迟。这些结果表明，相对于强大的非统一基线，统一建模框架改善了业务指标，同时减少了服务时间。
**表6：在线A/B结果：OneTransL（实验组）vs. RankMixer+Transformer（对照组）。Click/u、Order/u、GMV/u为相对变化率（%）。延迟为端到端每次曝光相对变化 $\Delta$ %（越低越好）。*表示p<0.05，**表示p<0.01**
| 场景 | gmv/u | order/u | click/u | 延迟(p99) $\downarrow$ |
|------|-------|---------|---------|-----------|
| Feeds | +5.685%** | +4.351%* | +7.737%** | -3.91% |
| Mall | +3.670%* | +2.577%** | +5.143%** | -3.26% |
我们进一步观察到用户活跃天数增加了+0.748%，冷启动产品的订单量/用户显著提升了+13.59%，突显了所提出模型强大的泛化能力。
## 5 结论
我们提出了OneTrans，一个用于个性化排序的统一Transformer主干网络，以取代传统的"编码-后-交互"范式。统一的分词器将序列和非序列属性转换为一个token序列，统一的Transformer块通过同质（序列）token的共享参数和异质（非序列）token的token特定参数，联合执行序列建模和特征交互。为了使统一堆栈在规模上保持高效，我们采用金字塔调度逐步剪枝序列token，以及跨请求KV缓存复用用户侧计算；该设计还受益于LLM风格的系统优化（如FlashAttention、混合精度）。通过大规模评估，OneTrans在宽度/深度增加时表现出近乎对数线性的性能提升，并在保持生产级延迟的同时带来了统计上显著的业务指标提升。我们相信，这种统一设计为扩展推荐系统提供了一条实用路径，同时复用了推动近期LLM进步的系统优化。
## 参考文献
[1] Zheng Chai, Qin Ren, Xijun Xiao, Huizhi Yang, Bo Han, Sijun Zhang, Di Chen, Hui Lu, Wenlin Zhao, Lele Yu, et al. 2025. LONGER: Scaling Up Long Sequence Modeling in Industrial Recommenders. arXiv preprint arXiv:2505.04421 (2025).
[2] Jianxin Chang, Chenbin Zhang, Yiqun Hui, Dewei Leng, Yanan Niu, Yang Song, and Kun Gai. 2023. Pepnet: Parameter and embedding personalized network for infusing with personalized prior information. In Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 3795–3804.
[3] Yin-Wen Chang, Cho-Jui Hsieh, Kai-Wei Chang, Chih-Jen Lin, et al. 2010. Training and testing low-degree polynomial data mappings via linear svm. Journal of Machine Learning Research 11, 4 (2010).
[4] Qiwei Chen, Huan Zhao, Wei Li, Pipei Huang, and Wenwu Ou. 2019. Behavior Sequence Transformer for E-commerce Recommendation in Alibaba. arXiv:1905.06874 [cs.IR].
[5] Heng-Tze Cheng, Levent Koc, Jeremiah Harmsen, Tal Shaked, Tushar Chandra, Hrishi Aradhye, Glen Anderson, Greg Corrado, Wei Chai, Mustafa Ispir, Rohan Anil, Zakaria Haque, Lichan Hong, Vihan Jain, Xiaobing Liu, and Hemal Shah. 2016. Wide & Deep Learning for Recommender Systems. arXiv:1606.07792 [cs.LG].
[6] Paul Covington, Jay Adams, and Emre Sargin. 2016. Deep neural networks for youtube recommendations. In Proceedings of the 10th ACM conference on recommender systems. 191–198.
[7] Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. 2022. Flashattention: Fast and memory-efficient exact attention with io-awareness. Advances in neural information processing systems 35 (2022), 16344–16359.
[8] Tri Dao, Aleksander Thomas, Anima Anandkumar, Matei Zaharia, and Christopher Re. 2023. FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning. arXiv preprint arXiv:2307.08691 (2023).
[9] Yufei Feng, Fuyu Lv, Weichen Shen, Menghan Wang, Fei Sun, Yu Zhu, and Keping Yang. 2019. Deep Session Interest Network for Click-Through Rate Prediction. arXiv:1905.06482 [cs.IR].
[10] Audrunas Gruslys, Remi Munos, Ivo Daniel, Oriol Vinyals, and Koray Kavukcuoglu. 2016. Memory-Efficient Backpropagation through Time. In Advances in Neural Information Processing Systems (NeurIPS).
[11] Huan Gui, Ruoxi Wang, Ke Yin, Long Jin, Maciej Kula, Taibai Xu, Lichan Hong, and Ed H. Chi. 2023. Hiformer: Heterogeneous Feature Interactions Learning with Transformers for Recommender Systems. arXiv:2311.05884 [cs.IR].
[12] Huifeng Guo, Ruiming Tang, Yunming Ye, Zhenguo Li, Xiuqiang He, and Zhenhua Dong. 2018. DeepFM: An End-to-End Wide & Deep Learning Framework for CTR Prediction. arXiv:1804.04950 [cs.IR].
[13] Junjie Huang, Jizheng Chen, Jianghao Lin, Jiarui Qin, Ziming Feng, Weinan Zhang, and Yong Yu. 2024. A comprehensive survey on retrieval methods in recommender systems. arXiv preprint arXiv:2407.21022 (2024).
[14] Wang-Cheng Kang and Julian McAuley. 2018. Self-Attentive Sequential Recommendation. arXiv:1808.09781 [cs.IR].
[15] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. 2020. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361 (2020).
[16] Shichen Liu, Fei Xiao, Wenwu Ou, and Luo Si. 2017. Cascade ranking for operational e-commerce search. In Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. 1557–1565.
[17] Paulius Micikevicius, Sharan Narang, Jonah Alben, Gregory Diamos, Erich Elsen, David Garcia, Boris Ginsburg, Michael Houston, Oleksii Kuchaiev, Ganesh Venkatesh, et al. 2017. Mixed precision training. arXiv preprint arXiv:1710.03740 (2017).
[18] Paulius Micikevicius, Sharan Narang, Jonah Alben, Greg Diamos, Erich Elsen, David Garcia, Boris Ginsburg, Michael Houston, Oleksii Kuchaiev, Ganesh Venkatesh, and Hao Wu. 2018. Mixed Precision Training. In International Conference on Learning Representations (ICLR).
[19] Maxim Naumov, Dheevatsa Mudigere, Hao-Jun Michael Shi, Jianyu Huang, Narayanan Sundaraman, Jongsoo Park, Xiaodong Wang, Udit Gupta, Carole-Jean Wu, Alisson G Azzolini, et al. 2019. Deep learning recommendation model for personalization and recommendation systems. arXiv preprint arXiv:1906.00091 (2019).
[20] Nikil Pancha, Andrew Zhai, Jure Leskovec, and Charles Rosenberg. 2022. Pinnerformer: Sequence modeling for user representation at pinterest. In Proceedings of the 28th ACM SIGKDD conference on knowledge discovery and data mining. 3702–3712.
[21] Qi Pi, Guorui Zhou, Yujing Zhang, Zhe Wang, Lejian Ren, Ying Fan, Xiaoqiang Zhu, and Kun Gai. 2020. Search-based user interest modeling with lifelong sequential behavior data for click-through rate prediction. In Proceedings of the 29th ACM International Conference on Information & Knowledge Management. 2685–2692.
[22] Jiarui Qin, Jiachen Zhu, Bo Chen, Zhirong Liu, Weiwen Liu, Ruiming Tang, Rui Zhang, Yong Yu, and Weinan Zhang. 2022. Rankflow: Joint optimization of multi-stage cascade ranking systems as flows. In Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval. 814–824.
[23] Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, and Bryan Catanzaro. 2019. Megatron-lm: Training multi-billion parameter language models using model parallelism. arXiv preprint arXiv:1909.08053 (2019).
[24] Weiping Song, Chence Shi, Zhiping Xiao, Zhijian Duan, Yewen Xu, Ming Zhang, and Jian Tang. 2019. AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks. In Proceedings of the 28th ACM International Conference on Information and Knowledge Management (CIKM '19). ACM, 1161–1170.
[25] Fei Sun, Jun Liu, Jian Wu, Changhua Pei, Xiao Lin, Wenwu Ou, and Peng Jiang. 2019. BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer. arXiv:1904.06690 [cs.IR].
[26] Ruoxi Wang, Bin Fu, Gang Fu, and Mingliang Wang. 2017. Deep & Cross Network for Ad Click Predictions. arXiv:1708.05123 [cs.LG].
[27] Ruoxi Wang, Rakesh Shivanna, Derek Cheng, Sagar Jain, Dong Lin, Lichan Hong, and Ed Chi. 2021. DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems. In Proceedings of the Web Conference 2021 (WWW '21). ACM, 1785–1797.
[28] Xue Xia, Pong Eksombatchai, Nikil Pancha, Dhruvil Deven Badani, Po-Wei Wang, Neng Gu, Saurabh Vishwas Joshi, Nazanin Farahpour, Zhiyuan Zhang, and Andrew Zhai. 2023. Transact: Transformer-based realtime user action model for recommendation at pinterest. In Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 5249–5259.
[29] Ruibin Xiong, Yunchang Yang, Di He, Kai Zheng, Shuxin Zheng, Chen Xing, Huishuai Zhang, Yanyan Lan, Liwei Wang, and Tieyan Liu. 2020. On layer normalization in the transformer architecture. In International conference on machine learning. PMLR, 10524–10533.
[30] Zhichen Zeng, Xiaolong Liu, Mengyue Hang, Xiaoyi Liu, Qinghai Zhou, Chaofei Yang, Yiqun Liu, Yichen Ruan, Laming Chen, Yuxin Chen, et al. 2024. Interformer: Towards effective heterogeneous interaction learning for click-through rate prediction. arXiv preprint arXiv:2411.09852 (2024).
[31] Jiaqi Zhai, Lucy Liao, Xing Liu, Yueming Wang, Rui Li, Xuan Cao, Leon Gao, Zhaojie Gong, Fangda Gu, Michael He, et al. 2024. Actions speak louder than words: Trillion-parameter sequential transducers for generative recommendations. arXiv preprint arXiv:2402.17152 (2024).
[32] Buyun Zhang, Liang Luo, Yuxin Chen, Jade Nie, Xi Liu, Daifeng Guo, Yanli Zhao, Shen Li, Yuchen Hao, Yantao Yao, et al. 2024. Wukong: Towards a scaling law for large-scale recommendation. arXiv preprint arXiv:2403.02545 (2024).
[33] Biao Zhang and Rico Sennrich. 2019. Root mean square layer normalization. Advances in neural processing systems 32 (2019).
[34] Guorui Zhou, Na Mou, Ying Fan, Qi Pi, Weijie Bian, Chang Zhou, Xiaoqiang Zhu, and Kun Gai. 2018. Deep Interest Evolution Network for Click-Through Rate Prediction. arXiv:1809.03672 [stat.ML].
[35] Guorui Zhou, Chengru Song, Xiaoqiang Zhu, Ying Fan, Han Zhu, Xiao Ma, Yanghui Yan, Junqi Jin, Han Li, and Kun Gai. 2018. Deep Interest Network for Click-Through Rate Prediction. arXiv:1706.06978 [stat.ML].
[36] Han Zhu, Xiang Li, Pengye Zhang, Guozheng Li, Jie He, Han Li, and Kun Gai. 2018. Learning tree-based deep model for recommender systems. In Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining. 1079–1088.
[37] Jie Zhu, Zhifang Fan, Xiaoxie Zhu, Yuchen Jiang, Hangyu Wang, Xintian Han, Haoran Ding, Xinmin Wang, Wenlin Zhao, Zhen Gong, Huizhi Yang, Zheng Chai, Zhe Chen, Yuchao Zheng, Qiwei Chen, Feng Zhang, Xun Zhou, Peng Xu, Xiao Yang, Di Wu, and Zuotao Liu. 2025. RankMixer: Scaling Up Ranking Models in Industrial Recommenders. arXiv:2507.15551 [cs.IR].