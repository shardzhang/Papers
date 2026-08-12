# MTGR Industrial Scale Generative Recommendation Framework in Meituan

> jingchunzhen, hanyueming02, zhoumenglei, yulei37, liuchuan11, linwei31}@meituan.com

## MTGR：美团工业级生成式推荐框架
美团
本文介绍了 MTGR Industrial Scale Generative Recommendation Framework in Meituan。核心内容：
关键发现：
---
韩瑞东，尹斌，陈尚宇，蒋赫，蒋菲，李想，马驰，黄敏聪，
李晓光，景春振，韩跃明，周梦磊，于磊，刘川，林伟∗
{hanruidong, yinbin05, chenshangyu03, jianghe06, jiangfei05, lixiang245, machi04, huangmincong, lixiaoguang12,
美团
中国北京
## 摘要
缩放定律已在自然语言处理和计算机视觉等诸多领域得到广泛验证。在推荐系统中，近期工作采用生成式推荐来实现可扩展性，但其生成式方法需要放弃传统推荐模型中精心构造的交叉特征。我们发现，这种方式会显著降低模型性能，且缩放根本无法弥补这一损失。本文中，我们提出MTGR（美团生成式推荐）来解决此问题。MTGR基于HSTU[23]架构进行建模，能够保留原始深度学习推荐模型（DLRM）的特征，包括交叉特征。此外，MTGR通过用户级压缩实现训练和推理加速，以确保高效缩放。我们还提出组级层归一化（GLN）以增强不同语义空间内的编码性能，以及动态掩码策略以避免信息泄露。我们进一步优化训练框架，使其支持计算复杂度为DLRM 10至100倍的模型，而成本无明显增加。MTGR在单样本前向推理中实现了相较于DLRM模型65倍FLOPs的计算量，在近两年离线与在线实验中均取得了最大幅度的提升。这一突破已在全球最大的外卖平台——美团成功部署，并持续承载主要流量。
## CCS概念
• 信息系统 $\to$ 推荐系统。
## 关键词
缩放定律；生成式推荐
ACM引用格式：
Ruidong Han, Bin Yin, Shangyu Chen, He Jiang, Fei Jiang, Xiang Li, Chi
Ma, Mincong Huang, Xiaoguang Li, Chunzhen Jing, Yueming Han, Menglei
Zhou, Lei Yu, Chuan Liu, Wei Lin. 2025. MTGR: Industrial-Scale Generative
Recommendation Framework in Meituan. In Proceedings of the 34th ACM
International Conference on Information and Knowledge Management (CIKM
'25), November 10–14, 2025, Seoul, Republic of Korea. ACM, New York, NY,
USA, 8 pages. https://doi.org/10.1145/3746252.3761565
∗通讯作者。
允许为个人或课堂使用复制或分发本作品的全部或部分副本，前提是复制件不以营利或商业优势为目的，且复制件在首页附带本声明及完整引用。对本作品中他人拥有的版权的组件必须予以尊重。允许带署名的摘要引用。如需以其他方式复制、重新发布、上传至服务器或分发给列表，需事先获得特定许可和/或支付相应费用。请向 permissions@acm.org 申请许可。
CIKM '25，韩国首尔
© 2025 版权所有 所有者/作者。出版权已授权给ACM。
ACM ISBN 979-8-4007-2040-6/2025/11
https://doi.org/10.1145/3746252.3761565
## 1 引言
缩放定律已被证明适用于大多数深度学习任务，包括语言模型[9]、计算机视觉[14, 24]和信息检索[4]。我们的工作聚焦于如何在工业推荐系统中有效缩放排序模型。在工业推荐系统对高QPS（每秒查询数）和低延迟的要求下，模型的缩放通常受到训练成本和推理时间的双重限制。目前，关于缩放排序模型的研究可分为两类：深度学习推荐模型（DLRM）和生成式推荐模型（GRM）。DLRM对单个用户-item对进行建模以学习排序的兴趣概率，并通过开发更复杂的映射进行扩展。GRM像自然语言一样按token组织数据，并通过Transformer架构执行下一token预测。
在工业推荐系统中，DLRM已使用了近十年，其输入通常包含大量精心设计的人工特征，例如交叉特征¹，以提升模型性能。然而，DLRM在缩放方面存在两个显著缺陷：1）随着用户行为的指数级增长，传统DLRM无法高效处理完整的用户行为，通常诉诸于序列检索或设计低复杂度模块进行学习，这限制了模型的学习能力[1, 15]；2）基于DLRM的缩放，其训练和推理成本随候选数量近似线性增长，导致开销高得难以承受。
对于GRM，近期研究指出了其优异的可扩展性[3, 23]。我们识别出两个关键因素：1）GRM直接对用户行为的完整链条进行建模，将同一用户下的多个曝光样本压缩为一个。这显著减少了计算冗余，同时相比DLRM能够端到端编码更长的序列；2）GRM采用了具有高效注意力计算的Transformer架构[2, 23]，使得模型的训练和推理能够满足工业推荐系统的要求。然而，GRM严重依赖下一token预测来建模完整的用户行为序列，这要求移除候选与用户之间的交叉特征。
¹交叉特征衡量多个原始特征之间的交互，例如用户对目标候选的历史点击率
CIKM '25，2025年11月10–14日，韩国首尔
韩瑞东，尹斌，陈尚宇等
我们发现，排除交叉特征会严重损害模型性能，且这种退化无法通过缩放来弥补。
我们如何构建一个既利用交叉特征确保有效性，又具备GRM可扩展性的排序模型？为解决此问题，我们提出了美团生成式推荐（MTGR）。相比传统DLRM和GRM，MTGR汲取两者优点、摒弃各自缺点。MTGR保持输入特征与传统DLRM一致，包括交叉特征。特别地，MTGR通过将用户特征和候选特征转换为不同的token来重新组织特征，形成token序列以实现高效的模型缩放。然后，MTGR将交叉特征纳入候选token中，并使用判别性损失进行学习。
MTGR采用与[23]中类似的HSTU（层次化序列转换单元）架构进行建模。在HSTU中，我们提出组级层归一化（GLN）来分别归一化不同类型的token，从而同时更好地建模多种异构信息。此外，我们还提出了一种动态掩码策略，采用全注意力、自回归和仅对自身可见的方式，以确保性能并避免信息泄露。
与业界常用的TensorFlow不同，MTGR的训练框架基于TorchRec[8]构建并针对计算效率进行了优化。具体而言，为处理稀疏嵌入条目的实时插入/删除，我们采用动态哈希表替代静态表。为提高效率，我们进行动态序列平衡以解决GPU间的计算负载不均衡，并采用嵌入ID去重以及自动表合并来加速嵌入查找。我们还整合了混合精度训练和算子融合等实现优化。相比TorchRec，我们的优化框架将训练吞吐量提升了1.6倍至2.4倍，同时在超过100个GPU上运行时实现了良好的可扩展性。
我们在小规模数据集上验证了MTGR的可扩展性。然后，我们设计了三个不同规模的模型，使用超过六个月的数据进行训练，以验证离线与在线性能的缩放定律。大规模版本相比经过多年优化的DLRM基线，前向计算的每样本FLOPs提升了65倍，在我们的场景中转化量提升了1.22%，点击率（CTR）提升了1.31%。同时，训练成本保持不变，推理成本降低了12%。MTGR-large已部署于美团外卖推荐系统，服务于数亿用户。
综上所述，我们的贡献总结如下：
• MTGR结合了DLRM和GRM的优势，保留了DLRM的所有特征（包括交叉特征），同时展现出与GRM相当的优异可扩展性。
• 我们提出组级层归一化和动态掩码策略以实现更好的性能。
• 我们基于TorchRec对MTGR训练框架进行了系统优化，以提升训练性能。
• 通过离线和在线实验，证明了MTGR性能与计算复杂度之间的幂律关系，以及其相较于DLRM的优越性。
## 2 相关工作
2.1 深度学习推荐模型
经典的DLRM结构通常包含多种输入，例如上下文（如时间、地点）、用户画像（如性别、年龄）、用户行为序列以及带有许多交叉特征的目标item。排序模型中两个特别重要的模块是行为序列处理和特征交互学习。行为序列模块[15, 17, 27]通常采用目标注意力机制来捕获用户历史行为与待估计item之间的相似性。特征交互模块[12, 18–20]旨在捕获不同特征（包括用户和item）之间的交互，以生成最终预测。
2.2 扩展推荐模型
基于DLRM中不同的缩放模块，存在两种不同的方法。一种是缩放交叉模块，即扩展融合用户和item信息的特征交互模块。[25]引入可堆叠的Wukong层进行扩展。[6]采用多嵌入策略解决嵌入坍塌问题，从而增强模型的可扩展性。另一种方法是缩放用户模块，仅扩展用户部分，使得该方法对推理更加友好。[7, 26]通过仅缩放用户表示并将其缓存或广播到待评估的不同item，来降低在线推理成本。[16, 22]设计了用户表示的预训练方法，在下游任务中展示了可扩展性。
与DLRM对应的是GRM。[23]通过HSTU验证了缩放定律，参数规模扩展至万亿级别。[3]使用语义编码替代传统ID表示，将DPO优化与基于Transformer的框架相结合，用统一的生成式模型替代级联学习框架。
## 3 预备知识
3.1 数据组织
传统上，对于一个用户及其对应的K个候选，用户与第i个候选对的第i个样本可表示为D𝑖 = [U, S, R, C𝑖, I𝑖]。具体而言，U = [U1, ..., U𝑁U]表示用户的画像特征（U𝑖），如年龄、性别等。每个特征U𝑖是一个标量，𝑁U表示所使用的特征数量。S = [S1, ..., S𝑁S]包含用户历史上交互过的item序列。S𝑖 = [s1, ..., s𝑁s]中的每个元素表示一个item，由选定的特征（s𝑖）组成，如item的ID、标签、item的平均CTR等。R记录用户最近几小时或一天内与当前请求最接近的交互，表示用户的实时行为和偏好。它与S共享相同的特征。C = [C1, ..., C𝑁C]包含用户与候选之间的交叉特征。I = [I1, ..., I𝑁I]包含候选使用的特征，如item的ID、标签和品牌。I依赖于候选，且对不同用户共享。
图1展示了传统排序模型下的简化数据组织和工作流程。这些特征首先被嵌入，主要的嵌入以不同方式处理。最后，处理后的特征被拼接并通过MLP进行特征交互，为每个候选生成最终的logit。
3.3 推荐系统中的缩放困境
模型缩放一直是排序中性能提升的常用方法。通常，模型缩放旨在扩展用户模块和交叉模块中的参数。用户模块处理包括序列特征在内的用户特征，并生成依赖用户的表示。扩展用户模块可以带来更好的用户表示。此外，由于用户对所有候选是共享且只推断一次的，用户模块中较大的推理成本不会导致系统过重的负担。然而，仅扩展用户模块并不能直接促进用户和item之间的特征交互。
相反，另一种方法趋势旨在扩展交叉模块，即特征拼接后的特征交互MLP。这些方法通过更多地关注用户与候选之间的交互来增强排序能力。然而，由于交叉模块对每个候选都进行推断，计算量随候选数量线性增加。交叉模块的扩展会带来不可接受的系统延迟。
传统推荐系统中的缩放困境需要一种新的缩放方法，既能实现用户与候选之间的高效特征交互，同时使推理成本随候选数量呈次线性增长。MTGR通过数据重排和相应的架构优化，创新了推荐系统中的缩放方法。
## 4 MTGR的数据重排与架构
4.1 用于训练与推理效率的用户样本聚合
与第3.1节中的特征分类相比，对于候选中的第i个样本，MTGR将特征组织为D𝑖 = [U, S, R, [C𝑖, I𝑖]]。特别地，交叉特征C被安排为候选item特征的一部分。在MTGR中，候选在训练时按特定窗口由用户聚合，推理时按请求聚合。由于聚合由同一用户完成，聚合后的样本可以使用相同的用户表示（U, S, R）。特别地，R按交互时间的时间顺序排列用户在该特定窗口内的所有实时交互item。
图2(a)展示了聚合过程：与图1中仅预测一个候选相比，图2(a)在一个样本中聚合了三个item，复用相同的用户表示。形式上，同一用户的特征表示如下：
D = [U, S, R, [C, I]1, ..., [C, I]K] (3)
通过将候选聚合到一个样本中，MTGR仅执行单次计算并为所有候选生成分数，从而大幅减少资源消耗。特别地，用户聚合过程大幅减少了训练样本数量。
图1：传统排序模型的数据组织与工作流程。展示了使用简化特征的示例：U包含'年龄'和'性别'；S和R分别由2个item组成，每个item拥有'ID'、'标签'和'品牌'；C包含'ctr'和'pv'，表示用户对候选的历史CTR和曝光次数，使用'ID'、'标签'和'品牌'。
3.2 推荐系统中的排序模型
使用输入样本D，传统推荐系统独立处理样本。具体而言，它首先对D中的特征进行嵌入，将样本转换为稠密表示。形式上，U、C、I中的特征被嵌入并拼接为EmbU
$$
\in
$$
 R^K
$$
\times
$$
dU、EmbC
$$
\in
$$
 R^K
$$
\times
$$
dC和EmbI
$$
\in
$$
 R^K
$$
\times
$$
dI。对于S和R中的特征²，每个item(S𝑖)的特征被类似地嵌入并拼接为EmbS𝑖
$$
\in
$$
 R^ds，S中的item沿另一维度拼接，得到EmbS
$$
\in
$$
 R^N_S
$$
\times
$$
ds²。
²在以下描述中，由于S和R以类似方式处理，为清晰起见仅描述S的处理。
³N_∗表示∗的序列长度。
为提取历史交互item与候选之间的用户兴趣，通常使用目标注意力，以目标为查询、序列特征为键/值。形式上：
FeatS = Attention(EmbI, EmbS, EmbS)
$$
\in
$$
 R^K
$$
\times
$$
dS (1)
公式(1)根据I聚合了S。最后，来自D的嵌入和处理后的特征被拼接并表示为：
FeatD = [EmbU, FeatS, EmbC, Embs]
$$
\in
$$
 R^K
$$
\times
$$
(dU+dS+dC+dI) (2)
FeatD进一步馈入多层感知机（MLP），为每个样本输出logit。该logit用于训练时的学习和推理时的排序。
CIKM '25，2025年11月10–14日，韩国首尔
韩瑞东，尹斌，陈尚宇等
所有token的统一维度。对于序列特征S和R，每个itemS被视为一个token。S中的特征首先被嵌入并拼接，然后采用MLP模块进行维度统一。形式上，FeatS𝑖 = MLP(Concat(Embs))
$$
\in
$$
 R^dmodel。序列中S的特征沿另一维度拼接，得到FeatS
$$
\in
$$
 R^N_S
$$
\times
$$
dmodel。
类似地，候选中的每个itemI被视为一个token。候选中的特征被嵌入并拼接，通过另一个MLP转换为统一维度。候选被拼接为一系列token：FeatI = Concat(MLP(Concat(EmbC𝑖, EmbI𝑖)))
$$
\in
$$
 R^N_I
$$
\times
$$
dmodel。最后，来自U、S、R、[C, s]的构造token被拼接，形成一个长token序列：
FeatD = Concat([FeatU, FeatS, FeatR, FeatI]) (4)
$$
\in
$$
 R^(N_U+N_S+N_R+N_I)
$$
\times
$$
dmodel
4.2 统一的HSTU编码器
来自同一用户的样本被聚合成一个token序列，这天然适合使用自注意力进行处理。受HSTU[23]启发，MTGR使用堆叠的自注意力层和仅编码器架构进行建模。
类似于LLM，输入token序列逐层处理。如图2所示，在自注意力块中，输入序列X首先通过组级层归一化进行归一化。相同域（例如U）的特征形成一个组。然后，组级层归一化确保来自不同域的token在自注意力前具有相似的分布，并对齐不同域的不同语义空间：X̃ = GroupLN(X)。归一化后的输入随后投影为4种不同的表示：K, Q, V, U = ML $P_{K/Q/V/U}$ (X̃)。Q和K用于带silu非线性激活的多头注意力计算。前导注意力除以输入特征的总长度作为平均因子。接着，对注意力分数施加自定义掩码（M），并应用投影后的V进行值更新：
Ṽ = (silu(K^T Q) / (N_U + N_S + N_R + N_I)) M V (5)
投影后的U与更新后的Ṽ进行点积。然后，应用另一个组级层归一化。最后，我们添加残差模块并在其上放置另一个MLP：
X = MLP(GroupLN(Ṽ
$$
\odot
$$
 U)) + X (6)
动态掩码[23]利用因果掩码进行序列建模。然而，这种实现在MTGR中并未带来显著改进。此外，由于R记录用户最近的交互，其时间可能与样本聚合窗口重合。在MTGR中使用简单的因果掩码可能导致信息泄露。例如，晚上的交互不应暴露给下午的候选，但这些信息可能被聚合到一个样本中。这一困境需要灵活且高效的掩码方案。
在MTGR中，U、S被视为静态（下文将U、S称为"静态序列"），因为其信息来自聚合窗口之前，因此不会引起因果关系错误。而R是动态的，因为它逐步包含用户的实时交互。
图2：MTGR的数据组织与架构。(a)展示了MTGR的数据组织和总体工作流程：3个候选特征与一个用户的对应部分聚合。特征被嵌入并通过MLP转换为token，形成一系列输入序列，用于带掩码的自注意力。候选token的表示通过另一个MLP模块用于logit生成。(b)L层自注意力模块的详细描述：输入序列首先经过组级层归一化和4个投影（Q/K/V/U）。使用自定义掩码进行自注意力计算。值更新后，对更新后的V和投影后的U进行点积操作。最后，更新后的值再次进行组级层归一化并添加残差连接。(c)自定义掩码以避免信息泄露：'rt'和'target'按从最近到过去排序：用户特征（U, S）对所有token可见；'rt'对后面的其他token可见；'target'仅对自身可见。
所有候选对所有用户的规模。推理时，请求中的候选按上述方式分组，MTGR仅对全部候选排序执行一次推理，而非按候选数量分别推理。这种聚合规避了推理成本对候选数量的依赖，为模型缩放留下了可能性和潜力。
公式(3)是标量特征和序列特征的组合。为统一输入格式，MTGR将特征和序列转换为token。具体地，对于U中的标量特征，每个特征自然转换为维度为FeatU
$$
\in
$$
 R^{N_U
$$
\times
$$
dmodel}的独立token。dmodel是所有token的统一维度。对于序列特征S和R，每个itemS被视为一个token。S中的特征首先被嵌入并拼接，然后采用MLP模块进行维度统一。形式上，FeatS𝑖 = MLP(Concat(Embs))
$$
\in
$$
 R^dmodel。序列中S的特征沿另一维度拼接，得到FeatS
$$
\in
$$
 R^{N_S
$$
\times
$$
dmodel}。
类似地，候选中的每个itemI被视为一个token。候选中的特征被嵌入并拼接，通过另一个MLP转换为统一维度。候选被拼接为一系列token：FeatI = Concat(MLP(Concat(EmbC𝑖, EmbI𝑖)))
$$
\in
$$
 R^{N_I
$$
\times
$$
dmodel}。最后，来自U、S、R、[C, s]的构造token被拼接，形成一个长token序列：
FeatD = Concat([FeatU, FeatS, FeatR, FeatI]) (4)
$$
\in
$$
 R^{(N_U+N_S+N_R+N_I)
$$
\times
$$
dmodel}
CIKM '25，2025年11月10–14日，韩国首尔
韩瑞东，尹斌，陈尚宇等
（实时的"动态序列"）。MTGR对静态序列应用全注意力，对R应用带动态掩码的自回归注意力，对候选间应用对角掩码。具体而言，MTGR的掩码设置了3条规则：
• 静态序列对所有token可见。
• 动态序列的可见性遵循因果关系，每个token仅对出现其后的token（包括候选token）可见。
• 候选token（C, I）仅对自身可见。
图2(c)展示了动态掩码的示例：'age'、'ctr'代表U中的特征token；'seq1'、'seq2'代表S；'rt1'、'rt2'代表R；'target1'–'target3'代表候选。行中的白色方块表示该token能够使用来自其他token的信息，而列表示该token对其他token是否可见。U和S使用全注意力，形成了从'age'到'seq2'的白色方块。对于'rt1'到'rt2'，我们假设'rt1'出现在'rt2'之后，因此从'rt1'到'rt2'构建了一个白色方块在上三角的小方块，意味着'rt1'能够使用'rt2'的信息，而'rt1'对'rt2'不可见。此外，'target2'和'target3'假定出现在'rt1'之前，因此'rt1'对它们不可见。'rt2'出现在所有'target1'和'target2'之前，但在'target3'之后，因此'rt2'对'target3'不可见，'target3'出现在所有'rt'之前，因此无法使用来自'rt'的信息。
## 5 训练系统
为促进MTGR模型结构的设计和开发，并方便地从快速发展的LLM社区引入更多特性，我们决定不再沿用之前基于TensorFlow的训练框架，而是选择在PyTorch生态系统中重建训练框架。具体而言，我们扩展并优化了TorchRec的功能，针对MTGR模型的特点进行了增强，最终实现了MTGR模型的高效训练。相比TorchRec，我们的优化框架将训练吞吐量提升了1.6倍至2.4倍，同时在超过100个GPU上运行时实现了良好的可扩展性。相比DLRM基线，我们实现了每样本前向计算65倍的FLOPs，而训练成本几乎保持不变。以下是我们的一些关键工作。
动态哈希表。TorchRec使用固定大小的表来处理稀疏嵌入，这不适用于大规模工业流式训练场景。首先，一旦静态表达到预设容量，就无法实时为新用户和新item分配额外的嵌入。其次，静态嵌入表通常需要预留比所需更多的空间以避免ID溢出，导致内存资源的低效使用。为解决这些问题，我们开发了一种利用哈希技术的高性能嵌入表，能够在训练过程中动态分配空间以容纳稀疏ID。我们的设计采用了解耦的哈希表架构[21]，将键存储和值存储分离为独立实体。键存储提供了一种轻量级映射系统，将键链接到指向嵌入向量的指针，而值存储包含嵌入向量以及额外的元数据（如计数器和时间戳），用于驱逐策略。这种两部分系统实现了两个主要目标：（1）通过仅复制键存储而非庞大的嵌入来实现容量的动态扩展；（2）通过以紧凑格式排列键来提高键扫描的效率。
嵌入查找。嵌入查找过程采用All-to-all通信进行跨设备嵌入交换。为最小化设备间的重复ID传输，我们实现了一个两步过程，确保ID在通信前后都保持唯一。
负载均衡。在推荐系统中，用户行为序列通常呈现长尾分布，只有少数用户拥有长序列，而大多数用户序列较短。这导致使用固定批大小（BS）训练时出现显著的计算负载不均衡。一种常见解决方案是使用序列打包技术[10]，将多个短序列合并为一个长序列。然而，这种方法需要仔细调整掩码以防止不同序列在注意力计算中相互干扰，实现成本较高。我们直接的解决方案是引入动态BS。根据输入数据的实际序列长度调整每个GPU的本地BS，确保相似的计算负载。此外，我们调整了梯度聚合策略，根据每个GPU的BS对其梯度进行加权，保持与固定BS一致的计算逻辑。
其他优化。为进一步提升训练效率，我们实现了利用三个不同流（拷贝、调度和计算）的流水线技术。拷贝流负责将输入数据从CPU传输到GPU，调度流使用ID执行表查找，计算流处理前向计算和反向更新。例如，当计算流执行批T的前向和后向传播时，拷贝流同时加载批T+1，从而最小化I/O延迟。一旦批T的反向更新完成，调度流立即开始批T+1的表查找和通信。此外，我们采用bf16混合精度训练，并基于cutlass设计了专门的注意力内核以加速训练进程。
## 6 实验
6.1 实验设置
数据集。公开数据集广泛使用独立的ID和属性特征，很少引入交叉特征。然而，交叉特征在实际应用中显示出重要性。交叉特征是我们场景中的重要特征类别。它们通常经过精心的人工构建，包括用户-item、用户与更高级别类别、item与时空信息等交互。为弥补公开数据集中交叉特征的缺失，我们基于美团真实工业级推荐系统的日志构建了训练数据集。与公开数据集不同，我们的真实数据集包含更丰富的交叉特征集和更长的用户行为序列。使用我们的工业级数据进行实验，能更好地突出这些交叉特征对实际推荐系统的重要影响。此外，我们的数据集体量较大，使复杂模型在训练中能够实现更充分的收敛。对于离线实验，我们收集了10天内的数据。数据集的统计信息如表1所示。对于在线实验，为与已训练超过2年的DLRM基线进行比较，我们构建了更长周期的数据集，使用了超过6个月的数据。
表1：数据集统计
数据集
  #用户
  #item
  #曝光
  #点击
  #购买
训练集
测试集
2.1亿
3,021,198
4,302,391
3,141,997
237.4亿
76,855,608
10.8亿
4,545,386
1.8亿
769,534
基线模型。对于DLRM，我们比较了序列建模中的两种方法：基于序列检索的SIM和原始长序列的端到端建模（E2E）。在缩放方面，我们比较了DNN、MoE[13]、Wukong[25]、MultiEmbed[6]和UserTower。
MoE使用4个专家，每个专家包含与基础DNN相同复杂度的网络。Wukong和MultiEmbed配置为与MoE相同的计算复杂度。UserTower使用一组可学习的查询，在用户行为上插入一个qFormer[11]层和另一个MoE（16个专家）模块。UserTower的计算复杂度是MoE方法的三倍，但在推理时可以对同一用户的多个预测item共享此计算，从而降低推理成本。它已在我们场景中取得了良好效果。
MTGR采用E2E处理所有序列信息。此外，如表2所示，我们设置了三个不同规模以验证MTGR的可扩展性。
表2：不同设置和计算复杂度的模型对比
模型
设置
学习率 GFLOPs/样本
UserTower-SIM
n_layer = 3, d_model = 512, n_heads = 2
MTGR-small
MTGR-medium n_layer = 5, d_model = 768, n_heads = 3
n_layer = 15, d_model = 768, n_heads = 3
MTGR-large
8
$$
\times
$$
10^−4
3
$$
\times
$$
10^−4
3
$$
\times
$$
10^−4
1
$$
\times
$$
10^−4
0.86
5.47
18.59
55.76
评估指标。离线方面，我们关注两个任务的学习：CTR和CTCVR（点击转化率），并使用AUC[5]和GAUC（分组AUC）进行评估。GAUC是对用户下AUC的平均值。相比AUC，GAUC更关注模型对同一用户的排序能力。在线评估方面，我们关注两个指标：PV_CTR（每次页面浏览的CTR）和UV_CTCVR（每次用户浏览的CTCVR），其中UV_CTCVR是评估业务增长的最关键指标。
参数设置。我们的模型使用Adam优化器进行训练。对于DLRM，每个GPU处理批大小为2400，使用8块NVIDIA A100 GPU进行训练。对于MTGR，批大小设置为96，使用16块NVIDIA A100 GPU进行训练。如表2所示，学习率随模型复杂度增加而降低。此外，随着计算复杂度的增长，我们通过配置不同的嵌入维度按比例增大稀疏参数的规模。假设一个token由k个特征组成，每个特征的嵌入维度通常设置为接近d_model/k的整数。值得注意的是，为防止稀疏参数过度扩展导致过大开销，我们主要增加基数值较小的稀疏特征的维度，同时保持极稀疏特征的维度不变。最后，S的最大长度设置为1000，R的最大长度设置为100。
6.2 整体性能对比
我们使用10天数据集评估了MTGR及其他基线方法的性能。表3展示了不同模型的性能。不同模型在各项离线指标上的差异相当一致。根据以往经验，离线指标提升0.001即被视为显著。在DLRM的各个版本中，Wukong-SIM和MultiEmbed-SIM取得了比MoE-SIM更好的结果。UserTower-SIM表现最佳，而UserTower-E2E相比UserTower-SIM略有下降。我们推测，在DLRM范式下，模型复杂度不足以建模所有序列信息，导致了欠拟合。我们提出的MTGR，即使是最小版本，也超过了最强的DLRM模型。此外，三个不同规模的模型呈现出可扩展性，其性能随着模型复杂度的增加而平稳提升。
表3：整体性能。Impr.%代表最佳MTGR模型相比最强DLRM基线（带下划线）的相对提升。
模型
CTR AUC
CTR GAUC
CTCVR AUC
CTCVR GAUC
DNN-SIM
0.7432
0.6679
0.8737
0.6504
MoE-SIM
0.7484
0.6698
0.8750
0.6519
MultiEmbed-SIM
0.7501
0.6715
0.8766
0.6525
Wukong-SIM
0.7568
0.6759
0.8800
0.6530
UserTower-SIM
0.7593
0.6792
0.8815
0.6550
UserTower-E2E
0.7576
0.6787
0.8818
0.6548
MTGR-small
0.7631
0.6826
0.8840
0.6603
MTGR-medium
0.7645
0.6843
0.8849
0.6625
MTGR-large
0.7661
0.6865
0.8862
0.6646
Impr.%
0.8956
1.0748
0.4990
1.4656
表4：MTGR的消融研究
模型
CTR AUC
CTR GAUC
CTCVR AUC
CTCVR GAUC
MTGR-small
0.7631
0.6826
0.8840
0.6603
w/o cross features
0.7495
0.6689
0.8736
0.6514
w/o GLN
0.7606
0.6809
0.8826
0.6585
w/o dynamic mask
0.7620
0.6810
0.8828
0.6587
6.3 消融研究
我们在小模型上对MTGR的两个组件进行了消融研究：动态掩码和组级层归一化（GLN）。消融结果如表4所示。移除MTGR中的任何一项都会导致性能显著下降，下降程度相当于MTGR-small到MTGR-medium的提升幅度。这表明了动态掩码和GLN对MTGR的重要性。此外，我们对交叉特征对MTGR的重要性进行了额外实验。移除交叉特征后，性能指标出现显著下降，甚至抹平了MTGR-large相对于DLRM的提升，突显了交叉特征在实际推荐系统中的关键作用。
随着训练token数量的增加，相比于DLRM的收益持续放大。最终，在CTCVR GAUC指标上，我们的大版本甚至超过了过去一年所有优化的累计提升。
该模型已在我们场景中全面部署，训练成本与DLRM持平，推理成本降低12%。对于DLRM，其推理成本大约与候选数量呈线性关系。然而，MTGR对一次请求中的所有候选使用用户聚合，使得推理成本随候选数量呈次线性扩展。这帮助我们降低了在线推理的开销。
表5：MTGR不同版本的离线与在线效果对比。
离线指标差值
在线指标差值
CTR GAUC CTCVR GAUC PV_CTR UV_CTCVR
MTGR-small
+0.0036
+0.0154
+1.04%
+0.04%
MTGR-medium
+0.0071
+0.0182
+2.29%
+0.62%
MTGR-large
+0.0153
+0.0288
+1.90%
+1.02%
## 7 结论
本文提出了MTGR，一种基于HSTU探索推荐系统中缩放定律的新的排序框架。MTGR结合了DLRM和GRM的优势，能够使用交叉特征确保模型性能，同时具备与GRM相同的可扩展性。MTGR已在我们场景中部署并取得了显著收益。未来，我们将探索如何将MTGR扩展到多场景建模，类似于大型语言模型，以建立一个拥有广泛知识的推荐基础模型。
图3：随着HSTU块数量、d_model和训练序列长度的增加，MTGR性能平稳提升。
图3展示了我们MTGR的可扩展性。我们基于MTGR-small对三个不同的超参数进行了测试：HSTU块数量、d_model和输入序列长度。可以看出，MTGR在不同超参数下均表现出良好的可扩展性。此外，图3(d)展示了性能与计算复杂度之间的幂律关系。纵轴表示相对于我们最佳DLRM模型UserTower-SIM的CTCVR GAUC指标增益，横轴反映相对于UserTower-SIM的计算复杂度对数倍数。
6.5 在线实验
为进一步验证MTGR的有效性，我们在美团外卖平台部署了MTGR，使用2%流量进行AB测试。实验流量覆盖每日数百万曝光，证明了实验的置信度。对比基线是经过2年持续学习的最先进在线DLRM模型（UserTower-SIM）。我们使用最近6个月的数据训练MTGR模型，然后部署上线进行比较。
尽管训练数据量显著低于DLRM模型，离线和在线指标仍大幅超过DLRM基线。如表5所示，离线和在线指标均展现出可扩展性。我们还发现，随着训练token数量的增加，相比于DLRM的收益持续放大。最终，在CTCVR GAUC指标上，我们的大版本甚至超过了过去一年所有优化的累计提升。
该模型已在我们场景中全面部署，训练成本与DLRM持平，推理成本降低12%。对于DLRM，其推理成本大约与候选数量呈线性关系。然而，MTGR对一次请求中的所有候选使用用户聚合，使得推理成本随候选数量呈次线性扩展。这帮助我们降低了在线推理的开销。
CIKM '25，2025年11月10–14日，韩国首尔
韩瑞东，尹斌，陈尚宇等
## 参考文献
[1] Qiwei Chen, Changhua Pei, Shanshan Lv, Chao Li, Junfeng Ge, and Wenwu Ou. 2021. End-to-end user behavior retrieval in click-through rateprediction model. arXiv preprint arXiv:2108.04468 (2021).
[2] Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. 2022. Flashattention: Fast and memory-efficient exact attention with io-awareness. Advances in neural information processing systems 35 (2022), 16344–16359.
[3] Jiaxin Deng, Shiyao Wang, Kuo Cai, Lejian Ren, Qigen Hu, Weifeng Ding, Qiang Luo, and Guorui Zhou. 2025. OneRec: Unifying Retrieve and Rank with Generative Recommender and Iterative Preference Alignment. arXiv preprint arXiv:2502.18965 (2025).
[4] Yan Fang, Jingtao Zhan, Qingyao Ai, Jiaxin Mao, Weihang Su, Jia Chen, and Yiqun Liu. 2024. Scaling laws for dense retrieval. In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval. 1339–1349.
[5] Cesar Ferri, José Hernández-Orallo, and Peter A Flach. 2011. A coherent interpretation of AUC as a measure of aggregated classification performance. In Proceedings of the 28th International Conference on Machine Learning (ICML-11). 657–664.
[6] Xingzhuo Guo, Junwei Pan, Ximei Wang, Baixu Chen, Jie Jiang, and Mingsheng Long. 2023. On the embedding collapse when scaling up recommendation models. arXiv preprint arXiv:2310.04400 (2023).
[7] Ruidong Han, Qianzhong Li, He Jiang, Rui Li, Yurou Zhao, Xiang Li, and Wei Lin. 2024. Enhancing CTR Prediction through Sequential Recommendation Pre-training: Introducing the SRP4CTR Framework. In Proceedings of the 33rd ACM International Conference on Information and Knowledge Management. 3777–3781.
[8] Dmytro Ivchenko, Dennis Van Der Staay, Colin Taylor, Xing Liu, Will Feng, Rahul Kindi, Anirudh Sudarshan, and Shahin Sefati. 2022. Torchrec: a pytorch domain library for recommendation systems. In Proceedings of the 16th ACM Conference on Recommender Systems. 482–483.
[9] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. 2020. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361 (2020).
[10] Mario Michael Krell, Matej Kosec, Sergio P Perez, and Andrew Fitzgibbon. 2021. Efficient sequence packing without cross-contamination: Accelerating large language models without impacting performance. arXiv preprint arXiv:2107.02027 (2021).
[11] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. 2023. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In International conference on machine learning. PMLR, 19730–19742.
[12] Jianxun Lian, Xiaohuan Zhou, Fuzheng Zhang, Zhongxia Chen, Xing Xie, and Guangzhong Sun. 2018. xdeepfm: Combining explicit and implicit feature interactions for recommender systems. In Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining. 1754–1763.
[13] Jiaqi Ma, Zhe Zhao, Xinyang Yi, Jilin Chen, Lichan Hong, and Ed H Chi. 2018. Modeling task relationships in multi-task learning with multi-gate mixture-of-experts. In Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining. 1930–1939.
[14] William Peebles and Saining Xie. 2023. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision. 4195–4205.
[15] Qi Pi, Guorui Zhou, Yujing Zhang, Zhe Wang, Lejian Ren, Ying Fan, Xiaoqiang Zhu, and Kun Gai. 2020. Search-based user interest modeling with lifelong sequential behavior data for click-through rate prediction. In Proceedings of the 29th ACM International Conference on Information & Knowledge Management. 2685–2692.
[16] Kyuyong Shin, Hanock Kwak, Su Young Kim, Max Nihlén Ramström, Jisu Jeong, Jung-Woo Ha, and Kyung-Min Kim. 2023. Scaling law for recommendation models: Towards general-purpose user representations. In Proceedings of the AAAI conference on artificial intelligence, Vol. 37. 4596–4604.
[17] Zihua Si, Lin Guan, ZhongXiang Sun, Xiaoxue Zang, Jing Lu, Yiqun Hui, Xingchao Cao, Zeyu Yang, Yichen Zheng, Dewei Leng, et al. 2024. Twin v2: Scaling ultra-long user behavior sequence modeling for enhanced ctr prediction at kuaishou. In Proceedings of the 33rd ACM International Conference on Information and Knowledge Management. 4890–4897.
[18] Hongyan Tang, Junning Liu, Ming Zhao, and Xudong Gong. 2020. Progressive layered extraction (ple): A novel multi-task learning (mtl) model for personalized recommendations. In Proceedings of the 14th ACM conference on recommender systems. 269–278.
[19] Ruoxi Wang, Rakesh Shivanna, Derek Cheng, Sagar Jain, Dong Lin, Lichan Hong, and Ed Chi. 2021. Dcn v2: Improved deep & cross network and practical lessons for web-scale learning to rank systems. In Proceedings of the web conference 2021. 1785–1797.
[20] Xu Wang, Jiangxia Cao, Zhiyi Fu, Kun Gai, and Guorui Zhou. 2024. HoME: Hierarchy of Multi-Gate Experts for Multi-Task Learning at Kuaishou. arXiv preprint arXiv:2408.05430 (2024).
[21] Yuxiang Wang, Xiao Yan, Chi Ma, Mincong Huang, Xiaoguang Li, Lei Yu, Chuan Liu, Ruidong Han, He Jiang, Bin Yin, et al. 2025. MTGRBoost: Boosting Large-scale Generative Recommendation Models in Meituan. arXiv preprint arXiv:2505.12663 (2025).
[22] Bencheng Yan, Shilei Liu, Zhiyuan Zeng, Zihao Wang, Yizhen Zhang, Yujin Yuan, Langming Liu, Jiaqi Liu, Di Wang, Wenbo Su, et al. 2025. Unlocking Scaling Law in Industrial Recommendation Systems with a Three-step Paradigm based Large User Model. arXiv preprint arXiv:2502.08309 (2025).
[23] Jiaqi Zhai, Lucy Liao, Xing Liu, Yueming Wang, Rui Li, Xuan Cao, Leon Gao, Zhaojie Gong, Fangda Gu, Michael He, et al. 2024. Actions speak louder than words: Trillion-parameter sequential transducers for generative recommendations. arXiv preprint arXiv:2402.17152 (2024).
[24] Xiaohua Zhai, Alexander Kolesnikov, Neil Houlsby, and Lucas Beyer. 2022. Scaling vision transformers. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 12104–12113.
[25] Buyun Zhang, Liang Luo, Yuxin Chen, Jade Nie, Xi Liu, Daifeng Guo, Yanli Zhao, Shen Li, Yuchen Hao, Yantao Yao, et al. 2024. Wukong: Towards a scaling law for large-scale recommendation. arXiv preprint arXiv:2403.02545 (2024).
[26] Wei Zhang, Dai Li, Chen Liang, Fang Zhou, Zhongke Zhang, Xuewei Wang, Ru Li, Yi Zhou, Yaning Huang, Dong Liang, et al. 2024. Scaling User Modeling: Large-scale Online User Representations for Ads Personalization in Meta. In Companion Proceedings of the ACM Web Conference 2024. 47–55.
[27] Guorui Zhou, Xiaoqiang Zhu, Chenru Song, Ying Fan, Han Zhu, Xiao Ma, Yanghui Yan, Junqi Jin, Han Li, and Kun Gai. 2018. Deep interest network for click-through rate prediction. In Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining. 1059–1068.