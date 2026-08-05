# YouTubeDNN: Deep Neural Networks for YouTube Recommendations

> Paul Covington, Jay Adams, Emre Sargin | Google



本文介绍了 YouTube 推荐系统的深度神经网络架构。核心内容：

- 提出两阶段推荐架构：候选生成（超大规模多分类 + 采样 softmax）和排序（加权逻辑回归优化观看时间）
- 引入Example Age特征解决 视频流行度的非平稳性问题
- 深度神经网络**通过嵌入层融合异构信号**（观看历史、搜索历史、人口统计特征）
- 排序阶段使用加权逻辑回归直接建模预期观看时间而非点击率

关键发现：

- 深度神经网络显著优于YouTube之前使用的矩阵分解方法，深层隐藏单元和异构信号均带来明显收益
- 引入"Example Age"特征有效消除了对过去的固有偏差，在A/B测试中显著增加了新上传视频的观看时间
- 排序阶段使用加权逻辑回归（正样本按观看时间加权）比直接预测点击率效果更好
- 向分类器隐藏具有区分能力的信号（如最后搜索查询）对防止替代问题过拟合至关重要
- 预测用户的下一个观看（而非随机保留的观看）能更好地捕获不对称的共同观看模式

---



## 摘要

YouTube 代表了现存规模最大、最复杂的工业推荐系统之一。本文从高层次描述了该系统，并重点介绍了深度学习带来的显著性能提升。本文按照经典的信息检索两阶段二分法进行组织：首先，我们详细介绍了一个深度候选生成模型，然后描述了一个独立的深度排序模型。我们还提供了从设计、迭代和维护一个对用户具有巨大影响力的大规模推荐系统中获得的实践经验与见解。



## 关键词

recommender system; deep learning; scalability

---



## 1. 引言

YouTube 是世界上最大的视频内容创建、分享和发现平台。YouTube 推荐系统负责帮助超过十亿用户从不断增长的视频库中发现个性化内容。本文聚焦于深度学习最近对 YouTube 视频推荐系统产生的巨大影响。图 1 展示了 YouTube 移动应用首页上的推荐内容。

<img src="/Users/dazhang/PycharmProject/Papers/3-RecSys/.picture/image-20260801100614557.png" alt="image-20260801100614557" style="zoom:33%;" />

从三个主要角度来看，推荐 YouTube 视频极具挑战性：

- **规模**：许多在小型问题上表现良好的现有推荐算法无法在我们的规模上运行。高度专业化的分布式学习算法和高效的服务系统对于处理 YouTube 庞大的用户群和语料库至关重要。

- **新鲜度**：YouTube 拥有非常动态的语料库，每秒上传数小时的视频。推荐系统应足够灵敏，能够对新上传的内容以及用户的最新行为进行建模。从探索/利用的角度看，理解新内容与成熟内容之间的平衡至关重要。

- **噪声**：由于稀疏性和各种不可观测的外部因素，YouTube 上的历史用户行为本质上难以预测。我们很少能获得用户满意度的真实标签，而是对噪声隐式反馈信号进行建模。此外，与内容相关的元数据缺乏良好的本体，结构混乱。我们的算法需要对训练数据的这些特定特征具有鲁棒性。

与 Google 其他产品领域协同，YouTube 经历了一场根本性的范式转变，转向使用深度学习作为几乎所有学习问题的通用解决方案。我们的系统构建于 Google Brain [4] 之上，后者最近已作为 TensorFlow [1] 开源。TensorFlow 提供了一个灵活的框架，用于使用大规模分布式训练试验各种深度神经网络架构。我们的模型学习约十亿个参数，并在数千亿个样本上进行训练。

与大量关于矩阵分解方法的研究 [19] 相比，将深度神经网络用于推荐系统的工作相对较少。神经网络被用于推荐新闻 [17]、文献引用 [8] 和评论评分 [20]。协同过滤被表述为深度神经网络 [22] 和自编码器 [18]。Elkahky 等人使用深度学习进行跨域用户建模 [5]。在基于内容的场景中，van den Oord 等人使用深度神经网络进行音乐推荐 [21]。

本文组织如下：第 2 节简要介绍系统概览。第 3 节详细描述了候选生成模型，包括其训练方式以及如何用于服务推荐。实验结果将展示模型如何从深层隐藏单元和额外的异构信号中受益。第 4 节详细介绍了排序模型，包括如何修改经典逻辑回归以训练预测预期观看时间（而非点击概率）的模型。实验结果将表明，在这种情况下隐藏层深度同样有益。最后，第 5 节总结了我们的结论和经验教训。

---



## 2. 系统概览

<img src="/Users/dazhang/PycharmProject/Papers/3-RecSys/.picture/image-20260801101126671.png" alt="image-20260801101126671" style="zoom:50%;" />

我们推荐系统的整体结构如图 2 所示。该系统由两个神经网络组成：一个用于候选生成，另一个用于排序。候选生成网络以用户 YouTube 活动历史中的事件作为输入，从大型语料库中检索出一个小子集（数百个）视频。这些候选视频旨在具有高精度的普遍相关性。候选生成网络仅通过协同过滤提供广泛的个性化。用户之间的相似性通过粗粒度特征来表示，例如视频观看的 ID、搜索查询的 token 和人口统计特征。

呈现一个包含几个"最佳"推荐的列表需要细粒度的表示，以便在高召回率下区分候选的相对重要性。排序网络通过使用丰富的特征（描述视频和用户）根据期望的目标函数为每个视频分配分数来完成此任务。得分最高的视频按分数排序后呈现给用户。

这种两阶段推荐方法使我们能够从非常大的语料库（数百万）中进行推荐，同时确保出现在设备上的少量视频对用户来说是个性化和有吸引力的。此外，这种设计可以混合由其他来源生成的候选，例如先前工作 [3] 中描述的那些。

在开发过程中，我们广泛使用离线指标（精确率、召回率、排序损失等）来指导系统的迭代改进。然而，对于算法或模型有效性的最终判断，我们依赖于通过在线实验进行的 A/B 测试。在在线实验中，我们可以衡量点击率、观看时间以及许多其他衡量用户参与度指标的细微变化。这一点非常重要，因为在线 A/B 测试结果并不总是与离线实验相关。

---



## 3. 候选生成

在候选生成阶段，庞大的 YouTube 语料库被缩减为数百个可能与用户相关的视频。本文描述的推荐系统的前身是一种基于排序损失 [23] 训练的矩阵分解方法。我们早期神经网络模型仅嵌入用户之前观看视频的浅层网络，模仿了这种分解行为。从这个角度来看，我们的方法可以被视为分解技术的非线性泛化。

### 3.1 推荐作为分类问题

我们将推荐视为极多类分类问题，其中预测任务是根据用户 $U$ 和上下文 $C$，在时间 $t$ 从拥有数百万视频 $i$（类别）的语料库 $V$ 中准确分类特定的视频观看 $w_t$：

$$
P(w_t = i|U,C) = \frac{e^{\mathbf{v}_i \mathbf{u}}}{\sum_{j \in V} e^{\mathbf{v}_j \mathbf{u}}} \qquad (1)
$$

其中 $\mathbf{u} \in \mathbb{R}^N$ 表示用户-上下文对的高维"嵌入"，$\mathbf{v}_j \in \mathbb{R}^N$ 表示每个候选视频的嵌入。在此设置中，嵌入只是将稀疏实体（单个视频、用户等）映射到 $\mathbb{R}^N$ 中的稠密向量。深度神经网络的任务是学习用户嵌入 $\mathbf{u}$，将其作为用户历史和上下文的函数，以便在使用 softmax 分类器区分不同视频时发挥作用。

尽管 YouTube 上存在显式反馈机制（赞/踩、产品内调查等），但我们使用观看的隐式反馈 [16] 来训练模型，其中用户完成一个视频即为正样本。这一选择基于以下考量：可用的隐式用户历史记录数量多出数个数量级，这使我们能够在显式反馈极为稀疏的长尾区域生成推荐。

#### 高效极多类分类

为了高效训练具有数百万类别的模型，我们依赖于从背景分布中采样负类别的技术（"候选采样"），然后通过重要性加权 [10] 对此采样进行校正。对于每个样本，交叉熵损失针对真实标签和采样的负类别进行最小化。实际上，我们会采样数千个负样本，相比传统 softmax 实现了超过 100 倍的加速。另一种流行的替代方法是层次化 softmax [15]，但我们未能达到相当的精度。在层次化 softmax 中，遍历树中的每个节点都需要在通常不相关的类别集合之间进行判别，这使得分类问题更加困难并降低了性能。

在服务时，我们需要计算最可能的 $N$ 个类别（视频），以便选择前 $N$ 个呈现给用户。在严格的服务延迟（几十毫秒）下对数百万个 item 进行评分，需要一种与类别数量呈次线性关系的近似评分方案。YouTube 早期的系统依赖于哈希 [24]，本文描述的分类器也使用了类似的方法。由于服务时不需要 softmax 输出层的校准似然，评分问题简化为点积空间中的最近邻搜索，可以使用通用库 [12] 来实现。我们发现 A/B 结果对最近邻搜索算法的选择并不特别敏感。

### 3.2 模型架构

受连续词袋语言模型 [14] 的启发，我们为固定词汇表中的每个视频学习高维嵌入，并将这些嵌入输入到前馈神经网络中。用户的观看历史由可变长度的稀疏视频 ID 序列表示，通过嵌入映射为稠密向量表示。网络需要固定大小的稠密输入，在多种策略（求和、逐分量取最大值等）中，对嵌入取平均表现最佳。重要的是，嵌入通过正常的梯度下降反向传播更新，与所有其他模型参数联合学习。特征被拼接成一个宽的第一层，后接多个全连接修正线性单元 ReLU（Rectified Linear Unit）[6] 层。图 3 展示了通用网络架构以及下面描述的额外非视频观看特征。

### 3.3 异构信号

使用深度神经网络作为矩阵分解泛化的一个关键优势是，可以轻松地将任意连续和类别特征添加到模型中。

**搜索历史**的处理方式与观看历史类似——每个查询被分词为 unigram 和 bigram，每个 token 被嵌入。取平均后，用户的 token 化、嵌入后的查询代表了总结性的稠密搜索历史。

**人口统计特征**对于为新用户提供先验知识以使推荐行为合理非常重要。用户的地理区域和设备被嵌入并拼接。简单的二值和连续特征，如用户的性别、登录状态和年龄，作为归一化到 $[0,1]$ 的实数值直接输入网络。

**"Example Age"特征**

每秒都有数小时的视频上传到 YouTube。推荐这些最近上传的（"新鲜"）内容对于 YouTube 产品来说极为重要。我们一致观察到用户更喜欢新鲜内容，尽管不能以牺牲相关性为代价。除了推荐用户想观看的新视频这一阶效应之外，还存在一个关键的次要现象——引导和传播病毒式内容 [11]。

机器学习系统通常表现出对过去的固有偏见，因为它们被训练来根据历史样本预测未来行为。视频流行度的分布是高度非平稳的，但我们的推荐器在语料库上产生的多项分布将反映训练窗口内平均的观看可能性。为了纠正这一点，我们在训练时将训练样本的年龄作为一个特征输入。在服务时，此特征被设置为零（或略负），以反映模型在训练窗口的最末端进行预测。

图 4 以任意选择的视频 [26] 为例展示了该方法的有效性。
<img src="/Users/dazhang/PycharmProject/Papers/3-RecSys/.picture/image-20260801101155721.png" alt="image-20260801101155721" style="zoom:33%;" />

### 3.4 标签与上下文选择

需要强调的是，推荐通常涉及解决一个替代问题并将结果迁移到特定上下文。一个经典的例子是，假设准确预测评分可以带来有效的电影推荐 [2]。我们发现，这种替代学习问题的选择对 A/B 测试中的性能有着超乎寻常的重要影响，但很难通过离线实验来衡量。

训练样本是从所有的 YouTube 观看行为（包括嵌入在其他网站上的观看）中生成的，而不仅仅是我们生成的推荐上的观看。否则，新内容将很难浮现，推荐器将过度偏向于利用。如果用户通过推荐之外的其他方式发现视频，我们希望能够通过协同过滤将这一发现快速传播给其他人。另一个改善在线指标的关键见解是，为每个用户生成固定数量的训练样本，从而在损失函数中有效地平等加权所有用户。这防止了一小部分高度活跃的用户主导损失。

有些反直觉的是，必须非常小心地向分类器隐藏某些信息，以防止模型利用网站的结构并对替代问题过拟合。考虑一个例子：用户刚刚发出了一个关于"taylor swift"的搜索查询。由于我们的问题是预测下一个观看的视频，获得此信息的分类器将预测最可能被观看的视频是"taylor swift"对应搜索结果页面上出现的视频。不足为奇的是，将用户最后的搜索结果页面直接复制为首页推荐效果非常差。通过丢弃序列信息并将搜索查询表示为无序的 token 袋，分类器不再直接感知标签的来源。

视频的自然消费模式通常会导致非常不对称的共同观看概率。连续剧通常是按顺序观看的，用户通常从某一类型中最广泛流行的内容开始发现艺人，然后逐渐聚焦于更小众的作品。因此，我们发现预测用户的**下一个**观看（而非随机保留的观看）效果更好（图 5）。许多协同过滤系统隐式地通过保留一个随机 item 并从用户历史中的其他 item 来预测它，从而选择标签和上下文（5a）。这会泄露未来信息并忽略不对称的消费模式。相比之下，我们通过选择一个随机观看并仅输入该保留标签观看之前用户采取的行为来回退用户的"历史"（5b）。

<img src="/Users/dazhang/PycharmProject/Papers/3-RecSys/.picture/image-20260801101250749.png" alt="image-20260801101250749" style="zoom:33%;" />

### 3.5 特征与深度实验

<img src="/Users/dazhang/PycharmProject/Papers/3-RecSys/.picture/image-20260801101309952.png" alt="image-20260801101309952" style="zoom:33%;" />

添加特征和深度显著提高了留出数据的精确率，如图 6 所示。在这些实验中，100 万个视频和 100 万个搜索 token 的词汇表各嵌入 $256$ 维浮点数，最大词袋大小为最近 $50$ 次观看和最近 $50$ 次搜索。Softmax 层在相同的 100 万个视频类别上输出多项分布，维度为 $256$（可以认为是独立的输出视频嵌入）。这些模型在所有 YouTube 用户上训练直至收敛，对应于数据上的数个 epoch。网络结构遵循常见的"塔"模式，即网络底部最宽，每个连续的隐藏层将单元数减半（类似于图 3）。深度为零的网络实际上是线性分解方案，其表现与前身系统非常相似。不断增加宽度和深度，直到增量收益减少且收敛变得困难：

<img src="/Users/dazhang/PycharmProject/Papers/3-RecSys/.picture/image-20260801101226070.png" alt="image-20260801101226070" style="zoom: 33%;" />

- **深度 0**：线性层简单地将拼接层转换为匹配 softmax 的 $256$ 维度
- **深度 1**：$256$ ReLU
- **深度 2**：$512 \rightarrow 256$ ReLU
- **深度 3**：$1024 \rightarrow 512 \rightarrow 256$ ReLU
- **深度 4**：$2048 \rightarrow 1024 \rightarrow 512 \rightarrow 256$ ReLU

---



## 4. 排序

排序的主要作用是使用展现数据来针对特定用户界面专门化和校准候选预测。例如，用户可能通常会以高概率观看某个视频，但由于缩略图的选择而不太可能点击特定的首页展现项。在排序阶段，我们可以访问更多描述视频以及用户与视频关系的特征，因为只需对几百个视频进行评分，而不是候选生成阶段的数百万个。排序对于集成不同候选来源也至关重要，因为这些来源的分数不能直接比较。

我们使用与候选生成类似架构的深度神经网络，通过逻辑回归为每个视频展现分配独立分数（图 7）。视频列表按此分数排序后返回给用户。我们最终的排序目标基于在线 A/B 测试结果不断调整，但通常是每次展现预期观看时间的简单函数。按点击率排序常常推广那些用户不会看完的欺骗性视频（"标题党"），而观看时间能更好地捕捉用户参与度 [13, 25]。

### 4.1 特征表示

我们的特征按照传统的分类法分为类别特征和连续/序数特征。我们使用的类别特征的基数差异很大——有些是二值的（例如用户是否已登录），而其他可能拥有数百万个可能值（例如用户的最后一次搜索查询）。特征进一步根据它们贡献的是单值还是多值集合来划分。单值类别特征的一个例子是被评分的展现的视频 ID，而相应的多值特征可能是用户最近看过的 $N$ 个视频 ID 的词袋。我们还根据特征是描述 item（"展现"）的属性还是描述用户/上下文（"查询"）的属性来对特征进行分类。查询特征每个请求计算一次，而展现特征则为每个被评分的 item 计算。

#### 特征工程

我们的排序模型通常使用数百个特征，大致在类别特征和连续特征之间平均分配。尽管深度学习有望减轻手工特征工程的负担，但我们的原始数据特性使其不易直接输入到前馈神经网络中。我们仍然投入了大量的工程资源来将用户和视频数据转换为有用的特征。主要挑战在于表示用户行为的时间序列以及这些行为与被评分的视频展现之间的关系。

我们观察到，最重要的信号是那些描述用户之前与 item 本身及其他相似 item 交互的信号，这与他人对广告排序的经验一致 [7]。例如，考虑用户与被评分视频的上传频道的过去历史——用户从该频道观看过多少视频？用户上次观看此主题的视频是什么时候？这些描述用户在过去对相关 item 行为的连续特征特别强大，因为它们能够在不同的 item 之间很好地泛化。我们还发现，以特征的形式将信息从候选生成传播到排序至关重要，例如：哪个来源推荐了这个视频候选？它们分配了什么分数？

描述过去视频展现频率的特征对于在推荐中引入变化也是至关重要的（连续的请求不会返回完全相同的列表）。如果用户最近被推荐了一个视频但没有观看，模型会在下次页面加载时自然地降低该展现的排序。提供精确到秒的展现和观看历史本身就是一项工程壮举，超出了本文的范围，但对于产生响应式推荐至关重要。

#### 嵌入类别特征

与候选生成类似，我们使用嵌入将稀疏的类别特征映射为适合神经网络的稠密表示。每个唯一的 ID 空间（"词汇表"）有一个独立的学习嵌入，其维度大致与唯一值数量的对数成比例。这些词汇表是简单的查找表，通过在训练前对数据进行一次遍历来构建。非常大基数的 ID 空间（例如视频 ID 或搜索查询词）通过在点击展现中按频率排序后仅保留前 $N$ 个来截断。词汇表外的值简单地映射到零嵌入。与候选生成一样，多值类别特征的嵌入在输入网络之前取平均。

重要的是，同一 ID 空间中的类别特征共享底层嵌入。例如，存在一个全局的视频 ID 嵌入，被多个不同的特征使用（展现的视频 ID、用户最后观看的视频 ID、"种子"推荐生成的视频 ID 等）。尽管嵌入是共享的，但每个特征单独输入网络，以便上层能为每个特征学习专门的表示。共享嵌入对于提高泛化能力、加速训练和减少内存需求非常重要。绝大多数模型参数都在这些高基数的嵌入空间中——例如，在 32 维空间中嵌入 100 万个 ID 的参数数量是 2048 单元全连接层的 7 倍。

#### 连续特征归一化

神经网络对其输入的缩放和分布非常敏感 [9]，而决策树集成等替代方法则不受单个特征缩放的影响。我们发现连续特征的正确归一化对于收敛至关重要。具有分布 $f$ 的连续特征 $x$ 通过使用累积分布将值缩放到在 $[0,1)$ 中均匀分布，从而变换为 $\tilde{x}$：

$$
\tilde{x} = \int_{-\infty}^{x} df \qquad (2)
$$

该积分通过在训练开始前对数据进行单次遍历计算的特征值分位数上进行线性插值来近似。

除了原始归一化特征 $\tilde{x}$，我们还输入幂次 $\tilde{x}^2$ 和 $\sqrt{\tilde{x}}$，使网络能够更容易地形成特征的超线性和次线性函数，从而赋予网络更强的表达能力。提供连续特征的幂次被发现可以提高离线精度。

### 4.2 建模预期观看时间

我们的目标是预测预期观看时间，其中训练样本要么是正样本（视频展现被点击），要么是负样本（展现未被点击）。正样本标注了用户观看视频的时间量。为了预测预期观看时间，我们使用了为此目的而开发的加权逻辑回归技术。

<img src="/Users/dazhang/PycharmProject/Papers/3-RecSys/.picture/image-20260801101327367.png" alt="image-20260801101327367" style="zoom: 33%;" />

该模型在交叉熵损失下使用逻辑回归进行训练（图 7）。然而，正（被点击）展现按观察到的视频观看时间进行加权。负（未被点击）展现都赋予单位权重。这样，逻辑回归学习到的几率是：

$$
\frac{\sum_i T_i}{N - k} \qquad (3)
$$

其中 $N$ 是训练样本数，$k$ 是正样本数，$T_i$ 是第 $i$ 个展现的观看时间。假设正样本的比例很小（在我们的情况下确实如此），则学习到的几率近似为 $E[T](1+P)$，其中 $P$ 是点击概率，$E[T]$ 是该展现的预期观看时间。由于 $P$ 很小，该乘积接近 $E[T]$。在推理时，我们使用指数函数 $e^x$ 作为最终的激活函数来产生这些近似预期观看时间的几率。

### 4.3 隐藏层实验

表 1 展示了我们在次日留出数据上使用不同隐藏层配置获得的结果。每个配置显示的数值（"加权逐用户损失"）是通过考虑单个页面上呈现给用户的正（被点击）和负（未被点击）展现获得的。我们首先用模型对这两个展现进行评分。如果负展现获得的分数高于正展现，则认为该正展现的观看时间已被错误预测。加权逐用户损失随后是错误预测的观看时间总量占留出展现对上总观看时间的比例。

这些结果表明，增加隐藏层的宽度以及增加其深度都可以改善结果。然而，权衡在于推理所需的服务器 CPU 时间。$1024$ 宽 ReLU + $512$ 宽 ReLU + $256$ 宽 ReLU 的配置在为我们带来最佳结果的同时，使我们能够保持在服务 CPU 预算范围内。

对于 $1024\rightarrow512\rightarrow256$ 模型，我们尝试仅输入归一化的连续特征而不输入其幂次，这导致损失增加了 0.2%。在相同隐藏层配置下，我们还训练了一个正样本和负样本权重相等的模型。不出所料，这导致加权观看时间损失增加了惊人的 4.1%。

**表 1：更宽和更深的隐藏 ReLU 层对计算次日留出数据上的观看时间加权逐对损失的影响**

<img src="/Users/dazhang/PycharmProject/Papers/3-RecSys/.picture/image-20260801101355582.png" alt="image-20260801101355582" style="zoom:33%;" />

---



## 5. 结论

我们描述了用于推荐 YouTube 视频的深度神经网络架构，该架构分为两个不同的问题：候选生成和排序。

我们的深度协同过滤模型能够**有效吸收多种信号**，并通过深度层对其交互进行建模，优于 YouTube 之前使用的矩阵分解方法 [23]。为推荐选择替代问题与其说是科学，不如说是艺术，我们发现通过对未来观看进行分类能够捕获不对称的共同观看行为并防止未来信息的泄露，从而在在线指标上表现良好。向分类器隐藏具有区分能力的信号对于获得良好结果也至关重要——否则模型会对替代问题过拟合，难以很好地迁移到首页场景。

我们证明了将训练样本的年龄作为输入特征可以消除对过去的固有偏见，并使模型能够表示视频随时间变化的流行度行为。这提高了离线留出精确率，并在 A/B 测试中显著增加了新上传视频的观看时间。

排序是一个更经典的机器学习问题，然而我们的深度学习方法在观看时间预测上优于以前的线性和基于树的方法。推荐系统特别受益于描述用户与 item 过去行为的专业特征。深度神经网络需要特殊的类别特征和连续特征表示，我们分别通过嵌入和分位数归一化进行变换。深度层被证明可以有效建模数百个特征之间的非线性交互。

逻辑回归被修改为对正样本按观看时间加权、对负样本赋予单位权重，这使我们能够学到接近预期观看时间的几率。与直接预测点击率相比，该方法在观看时间加权的排序评估指标上表现更好。

---



## 6. 致谢

作者要感谢 Jim McFadden 和 Pranav Khaitan 的宝贵指导和支持。Sujeet Bansal、Shripad Thite 和 Radek Vingralek 实现了训练和服务基础设施的关键组件。Chris Berg 和 Trevor Walker 贡献了富有洞察力的讨论和详细的反馈。

---



## 参考文献

[1] M. Abadi, A. Agarwal, P. Barham, E. Brevdo, Z. Chen, C. Citro, G. S. Corrado, A. Davis, J. Dean, M. Devin, S. Ghemawat, I. Goodfellow, A. Harp, G. Irving, M. Isard, Y. Jia, R. Jozefowicz, L. Kaiser, M. Kudlur, J. Levenberg, D. Mané, R. Monga, S. Moore, D. Murray, C. Olah, M. Schuster, J. Shlens, B. Steiner, I. Sutskever, K. Talwar, P. Tucker, V. Vanhoucke, V. Vasudevan, F. Viégas, O. Vinyals, P. Warden, M. Wattenberg, M. Wicke, Y. Yu, and X. Zheng. TensorFlow: Large-scale machine learning on heterogeneous systems, 2015. Software available from tensorflow.org.

[2] X. Amatriain. Building industrial-scale real-world recommender systems. In *Proceedings of the Sixth ACM Conference on Recommender Systems*, RecSys '12, pages 7–8, New York, NY, USA, 2012. ACM.

[3] J. Davidson, B. Liebald, J. Liu, P. Nandy, T. VanVleet, U. Gargi, S. Gupta, Y. He, M. Lambert, B. Livingston, and D. Sampath. **The youtube video recommendation system**. In *Proceedings of the Fourth ACM Conference on Recommender Systems*, RecSys '10, pages 293–296, New York, NY, USA, 2010. ACM.

[4] J. Dean, G. S. Corrado, R. Monga, K. Chen, M. Devin, Q. V. Le, M. Z. Mao, M. Ranzato, A. Senior, P. Tucker, K. Yang, and A. Y. Ng. Large scale distributed deep networks. In *NIPS*, 2012.

[5] A. M. Elkahky, Y. Song, and X. He. **A multi-view deep learning approach for cross domain user modeling in recommendation systems**. In *Proceedings of the 24th International Conference on World Wide Web*, WWW '15, pages 278–288, New York, NY, USA, 2015. ACM.

[6] X. Glorot, A. Bordes, and Y. Bengio. **Deep sparse rectifier neural networks**. In G. J. Gordon and D. B. Dunson, editors, *Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics (AISTATS-11)*, volume 15, pages 315–323. Journal of Machine Learning Research - Workshop and Conference Proceedings, 2011.

[7] X. He, J. Pan, O. Jin, T. Xu, B. Liu, T. Xu, Y. Shi, A. Atallah, R. Herbrich, S. Bowers, and J. Q. n. Candela. **Practical lessons from predicting clicks on ads at facebook**. In *Proceedings of the Eighth International Workshop on Data Mining for Online Advertising*, ADKDD'14, pages 5:1–5:9, New York, NY, USA, 2014. ACM.

[8] W. Huang, Z. Wu, L. Chen, P. Mitra, and C. L. Giles. A neural probabilistic model for context based citation recommendation. In *AAAI*, pages 2404–2410, 2015.

[9] S. Ioffe and C. Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. *CoRR*, abs/1502.03167, 2015.

[10] S. Jean, K. Cho, R. Memisevic, and Y. Bengio. **On using very large target vocabulary for neural machine translation**. *CoRR*, abs/1412.2007, 2014.

[11] L. Jiang, Y. Miao, Y. Yang, Z. Lan, and A. G. Hauptmann. Viral video style: A closer look at viral videos on youtube. In *Proceedings of International Conference on Multimedia Retrieval*, ICMR '14, pages 193:193–193:200, New York, NY, USA, 2014. ACM.

[12] T. Liu, A. W. Moore, A. Gray, and K. Yang. An investigation of practical approximate nearest neighbor algorithms. pages 825–832. MIT Press, 2004.

[13] E. Meyerson. Youtube now: Why we focus on watch time. http://youtubecreator.blogspot.com/2012/08/youtube-now-why-we-focus-on-watch-time.html. Accessed: 2016-04-20.

[14] T. Mikolov, I. Sutskever, K. Chen, G. Corrado, and J. Dean. **Distributed representations of words and phrases and their compositionality**. *CoRR*, abs/1310.4546, 2013.

[15] F. Morin and Y. Bengio. Hierarchical probabilistic neural network language model. In *AISTATS'05*, pages 246–252, 2005.

[16] D. Oard and J. Kim. **Implicit feedback for recommender systems**. In *Proceedings of the AAAI Workshop on Recommender Systems*, pages 81–83, 1998.

[17] K. J. Oh, W. J. Lee, C. G. Lim, and H. J. Choi. Personalized news recommendation using classified keywords to capture user preference. In *16th International Conference on Advanced Communication Technology*, pages 1283–1287, Feb 2014.

[18] S. Sedhain, A. K. Menon, S. Sanner, and L. Xie. Autorec: Autoencoders meet collaborative filtering. In *Proceedings of the 24th International Conference on World Wide Web*, WWW '15 Companion, pages 111–112, New York, NY, USA, 2015. ACM.

[19] X. Su and T. M. Khoshgoftaar. A survey of collaborative filtering techniques. *Advances in artificial intelligence*, 2009:4, 2009.

[20] D. Tang, B. Qin, T. Liu, and Y. Yang. User modeling with neural network for review rating prediction. In *Proc. IJCAI*, pages 1340–1346, 2015.

[21] A. van den Oord, S. Dieleman, and B. Schrauwen. Deep content-based music recommendation. In C. J. C. Burges, L. Bottou, M. Welling, Z. Ghahramani, and K. Q. Weinberger, editors, *Advances in Neural Information Processing Systems 26*, pages 2643–2651. Curran Associates, Inc., 2013.

[22] H. Wang, N. Wang, and D.-Y. Yeung. Collaborative deep learning for recommender systems. In *Proceedings of the 21th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, KDD '15, pages 1235–1244, New York, NY, USA, 2015. ACM.

[23] J. Weston, S. Bengio, and N. Usunier. Wsabie: Scaling up to large vocabulary image annotation. In *Proceedings of the International Joint Conference on Artificial Intelligence*, IJCAI, 2011.

[24] J. Weston, A. Makadia, and H. Yee. Label partitioning for sublinear ranking. In S. Dasgupta and D. Mcallester, editors, *Proceedings of the 30th International Conference on Machine Learning (ICML-13)*, volume 28, pages 181–189. JMLR Workshop and Conference Proceedings, May 2013.

[25] X. Yi, L. Hong, E. Zhong, N. N. Liu, and S. Rajan. Beyond clicks: Dwell time for personalization. In *Proceedings of the 8th ACM Conference on Recommender Systems*, RecSys '14, pages 113–120, New York, NY, USA, 2014. ACM.

[26] Zayn. Pillowtalk. https://www.youtube.com/watch?v=C_3d6GntKbk.
