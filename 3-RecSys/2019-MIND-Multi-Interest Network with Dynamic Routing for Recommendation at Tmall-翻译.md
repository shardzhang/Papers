# Multi-Interest Network with Dynamic Routing for Recommendation at Tmall

> Chao Li, Zhiyuan Liu, Mengmeng Wu, Yuchi Xu, Pipei Huang, Huan Zhao, Guoliang Kang, Qiwei Chen, Wei Li, Dik Lun Lee | Alibaba Group, Hong Kong University of Science and Technology, University of Technology Sydney

本文提出了带动态路由的多兴趣网络（MIND，Multi-Interest Network with Dynamic routing）模型，用于在推荐系统匹配阶段对用户多样化兴趣进行建模。核心内容：

- **多兴趣提取层**：基于胶囊机制中的动态路由，自适应地将用户的历史行为聚合到多个用户表示向量中，通过软聚类捕捉用户的多样化兴趣
- **标签感知注意力**：训练时使目标item选择相关的兴趣胶囊，帮助学习具有多个向量的用户表示
- **实际部署验证**：在多个公开基准和一个来自天猫的大规模工业数据集上优于现有最先进方法，MIND已部署在手机天猫App首页处理主要线上流量

关键发现：用多个向量表示用户能有效捕捉用户兴趣的多样性，在匹配阶段显著提升推荐准确率（在线CTR），且动态兴趣数量机制在不损失CTR的情况下降低服务成本。

---

## 摘要

工业推荐系统通常由匹配阶段和排序阶段组成，以处理十亿级别的用户和item。匹配阶段检索与用户兴趣相关的候选item，而排序阶段根据用户兴趣对候选item进行排序。因此，对于任一阶段而言，最关键的能力是建模和表示用户兴趣。现有大多数基于深度学习的方法将一个用户表示为一个单一的向量，这不足以捕捉用户兴趣的变化特性。在本文中，我们从不同的角度处理这个问题，即将一个用户用多个向量表示，编码用户兴趣的不同方面。我们提出了带动态路由的多兴趣网络（MIND，Multi-Interest Network with Dynamic routing），用于在匹配阶段处理用户的多样化兴趣。具体而言，我们设计了一个基于胶囊路由机制的多兴趣提取层，该层适用于对历史行为进行聚类并提取多样化的兴趣。此外，我们开发了一种名为标签感知注意力的技术，以帮助学习具有多个向量的用户表示。通过在多个公开基准和一个来自天猫的大规模工业数据集上的广泛实验，我们证明了MIND在推荐方面能够达到优于现有最先进方法的性能。目前，MIND已部署在手机天猫App首页，处理主要的线上流量。

## 1 引言

**图1：左图：虚线矩形标记的区域是为天猫十亿级用户进行个性化展示的区域；右图：用户A与来自几个不同类别的产品交互，包括服装、体育和食品，而用户B与书籍、玩具和手机产品交互。**

天猫是中国最大的企业对消费者（B2C，Business-to-Customer）电子商务平台，通过提供十亿级别的在线产品服务于十亿级别的用户。在2018年11月11日，著名的天猫全球购物节，商品交易总额（GMV）约为2130亿元，比2017年同日增长26.9%。随着用户和产品数量的持续增长，帮助每个用户找到他/她可能感兴趣的产品变得越来越重要。近年来，天猫在开发个性化推荐系统方面投入了大量精力，这显著促进了用户体验的优化和商业价值的提升。例如，手机天猫App首页（如图1（左）所示），约占天猫总流量的一半，已经部署了推荐系统来展示个性化产品以满足客户的个性化需求。

由于用户和item规模达十亿级别，为天猫设计的推荐过程包括两个阶段：匹配阶段和排序阶段。匹配阶段负责检索与用户兴趣相关的数千个候选item，之后排序阶段预测用户与这些候选item交互的精确概率。对于这两个阶段，建模用户兴趣并找到捕捉用户兴趣的用户表示至关重要，以支持高效检索满足用户兴趣的item。然而，在天猫建模用户兴趣并非易事，因为用户存在多样化的兴趣。平均而言，十亿级别的用户访问天猫，每个用户每天与数百种产品交互。交互的产品往往属于不同的类别，表明用户兴趣的多样性。例如，如图1（右）所示，不同用户的兴趣各不相同，同一用户也可能对多种类型的item感兴趣。因此，捕捉用户多样化兴趣的能力对天猫的推荐系统变得至关重要。

现有的推荐算法以不同方式建模和表示用户兴趣。基于协同过滤的方法通过历史交互item或隐藏因子来表示用户兴趣，但这存在稀疏性问题或计算量大的问题。基于深度学习的方法通常用低维嵌入向量表示用户兴趣。例如，为YouTube视频推荐提出的深度神经网络（YouTube DNN）将每个用户表示为一个从用户过去行为转换而来的固定长度向量，这对于建模多样化兴趣可能成为一个瓶颈，因为为了表达天猫上大量的兴趣画像，其维度必须很大。深度兴趣网络（DIN，Deep Interest Network）通过注意力机制使用户表示随不同item而变化，以捕捉用户兴趣的多样性。然而，注意力机制的采用也使其在拥有十亿item的大规模应用中计算代价高昂，因为它需要对每个item重新计算用户表示，这使得DIN仅适用于排序阶段。

在本文中，我们聚焦于在匹配阶段建模用户多样化兴趣的问题。为了克服现有方法的局限性，我们提出了带动态路由的多兴趣网络（MIND），用于学习反映用户在工业推荐系统匹配阶段多样化兴趣的用户表示。为了推断用户表示向量，我们设计了一个名为多兴趣提取层的新型层，该层利用动态路由自适应地将用户的历史行为聚合到用户表示中。动态路由的过程可以看作软聚类，将用户的历史行为分组到多个簇中。每个历史行为簇进一步用于推断对应一个特定兴趣的用户表示向量。通过这种方式，对于特定用户，MIND输出多个表示向量，共同表示用户的多样化兴趣。用户表示向量仅需计算一次，即可在匹配阶段用于从十亿级item中检索相关item。总结来说，本文的主要贡献如下：

- 为了从用户行为中捕捉用户的多样化兴趣，我们设计了多兴趣提取层，利用动态路由自适应地将用户的历史行为聚合到用户表示向量中。
- 通过使用多兴趣提取层产生的用户表示向量和新提出的标签感知注意力层，我们构建了一个用于个性化推荐任务的深度神经网络。与现有方法相比，MIND在多个公开数据集和一个来自天猫的工业数据集上展现了优越的性能。
- 为了部署MIND以服务于天猫十亿级别的用户，我们构建了一个系统来实现数据收集、模型训练和在线服务的完整流程。该部署系统显著提升了手机天猫App首页的点击率（CTR，Click-Through Rate）。

本文的其余部分组织如下：第2节回顾相关工作；第3节详细阐述MIND的技术细节；第4节详细介绍MIND与现有方法在多个公开基准和在线服务上的对比实验；第5节介绍MIND在大规模工业应用中的部署；最后一节给出本文的结论和未来工作。

## 2 相关工作

**推荐中的深度学习。** 受深度学习在计算机视觉和自然语言处理领域成功的启发，人们投入了大量精力开发基于深度学习的推荐算法。除了提出的工业应用外，各种类型的深度模型也获得了显著关注。神经协同过滤（NCF，Neural Collaborative Filtering）、DeepFM（Deep Factorization Machine）和深度矩阵分解模型（DMF，Deep Matrix Factorization Models）构建了由多个多层感知机（MLP，Multi-Layer Perceptron）组成的神经网络来建模用户与item之间的交互。有工作通过提供一个统一且灵活的网络来捕捉更多特征，为Top- $N$ 序列推荐提出了一个新的解决方案。

**用户表示。** 将用户表示为向量在推荐系统中很常见。传统方法将用户偏好组装为由感兴趣item、关键词和主题组成的向量。随着分布式表示学习的出现，通过神经网络获得的用户嵌入被广泛使用。有工作采用门控循环单元（RNN-GRU，Gated Recurrent Unit）从时间排序的评论文档中学习用户嵌入。有的工作从词嵌入向量中学习用户嵌入向量，并将其应用于推荐学术微博。还有工作提出了一个新颖的基于卷积神经网络的模型，该模型结合从话语中提取的特征来显式学习和利用用户嵌入。

**胶囊网络。** "胶囊"的概念——一小群神经元组合在一起输出一个整体向量——由Hinton于2011年首次提出。动态路由用于学习胶囊之间连接的权重，而不是反向传播，后来通过利用期望最大化算法改进以克服若干缺陷并获得了更好的准确性。这两个与传统神经网络的主要区别使胶囊网络能够编码部分与整体之间的关系，这在计算机视觉和自然语言处理领域都很先进。SegCaps证明胶囊能够比传统卷积神经网络（CNN，Convolutional Neural Network）更好地成功建模物体的空间关系。有工作研究了用于文本分类的胶囊网络，并提出了3种提升性能的策略。

## 3 方法

### 3.1 问题形式化

工业推荐系统匹配阶段的目标是从十亿级别的item池 $\mathcal{I}$ 中为每个用户 $u \in \mathcal{U}$ 检索一个item子集，使得该子集仅包含数千个item且每个item与该用户的兴趣相关。为了实现这一目标，收集推荐系统产生的历史数据用于构建匹配模型。具体而言，每个实例可以用一个元组 $(\mathcal{I}_u, \mathcal{P}_u, \mathcal{F}_i)$ 表示，其中 $\mathcal{I}_u$ 表示用户 $u$ 交互的item集合（也称为用户行为）， $\mathcal{P}_u$ 表示用户 $u$ 的基本画像（如用户性别和年龄）， $\mathcal{F}_i$ 表示目标item的特征（如itemID和品类ID）。

MIND的核心任务是学习一个将原始特征映射为用户表示的函数，可以形式化为：

$$
\mathbf{V}_u = f_{user}(\mathcal{I}_u, \mathcal{P}_u) \qquad (1)
$$

其中 $\mathbf{V}_u = (\mathbf{v}_u^1, \ldots, \mathbf{v}_u^K) \in \mathbb{R}^{d \times K}$ 表示用户 $u$ 的表示向量， $d$ 为维度， $K$ 为表示向量的数量。当 $K=1$ 时，使用一个表示向量，就像YouTube DNN一样。此外，目标item $i$ 的表示向量通过一个嵌入函数获得：

$$
\mathbf{e}_i = f_{item}(\mathcal{F}_i) \qquad (2)
$$

其中 $\mathbf{e}_i \in \mathbb{R}^{d \times 1}$ 表示item $i$ 的表示向量， $f_{item}$ 的细节将在"嵌入与池化层"部分说明。

当学习到用户表示向量和item表示向量后，根据评分函数检索Top $N$ 候选item：

$$
f_{score}(\mathbf{V}_u, \mathbf{e}_i) = \max_{1 \leq k \leq K} \mathbf{e}_i^{T} \mathbf{v}_u^{k} \qquad (3)
$$

其中 $N$ 是匹配阶段预定义的待检索item数量。

### 3.2 嵌入与池化层

**图2：MIND总体架构。** MIND以用户行为及用户画像特征作为输入，输出用于推荐匹配阶段item检索的用户表示向量。输入层的ID特征通过嵌入层转换为嵌入，每个item的嵌入进一步经池化层平均。用户行为嵌入被输入多兴趣提取层，产生兴趣胶囊。通过将兴趣胶囊与用户画像嵌入拼接，并经多个ReLU层变换，得到用户表示向量。训练时，额外引入一个标签感知注意力层来指导训练过程。服务时，多个用户表示向量用于通过近似最近邻查找方式检索item。

如图2所示，MIND的输入由三组组成：用户画像 $\mathcal{P}_u$ 、用户行为 $\mathcal{I}_u$ 和标签item $\mathcal{F}_i$ 。每组包含几个类别ID特征，这些ID特征具有极高的维度。例如，itemID的数量约为数十亿，因此我们采用广泛使用的嵌入技术将这些ID特征嵌入到低维稠密向量（即嵌入）中，这显著减少了参数数量并简化了学习过程。对于来自 $\mathcal{P}_u$ 的ID特征（性别、年龄等），对应的嵌入被拼接起来形成用户画像嵌入 $\mathbf{p}_u$ 。对于来自 $\mathcal{F}_i$ 的itemID以及其他已被证明对冷启动item有用的类别ID（品牌ID、店铺ID等），对应的嵌入进一步通过平均池化层形成标签item嵌入 $\mathbf{e}_i$ 。最后，对于来自用户行为 $\mathcal{I}_u$ 的item，收集对应的item嵌入形成用户行为嵌入 $\mathbf{E}_u = \{\mathbf{e}_j, j \in \mathcal{I}_u\}$ 。

### 3.3 多兴趣提取层

我们认为用一个表示向量来表示用户兴趣可能成为捕捉用户多样化兴趣的瓶颈，因为我们必须将所有与用户多样化兴趣相关的信息压缩到一个表示向量中。因此，关于用户多样化兴趣的所有信息混合在一起，导致匹配阶段的item检索不准确。相反，我们采用多个表示向量来分别表达用户的不同兴趣。通过这种方式，在匹配阶段用户多样化的兴趣被分开考虑，从而使得兴趣的每个方面都能实现更准确的item检索。

为了学习多个表示向量，我们利用聚类过程将用户的历史行为分组到多个簇中。来自一个簇的item预期是密切相关的，并共同表示用户兴趣的一个特定方面。在此，我们设计了多兴趣提取层用于对历史行为进行聚类并为生成的簇推断表示向量。由于多兴趣提取层的设计受到最近在胶囊网络中提出的用于表示学习的动态路由的启发，我们首先回顾一些必要的基础知识以使本文自包含。

#### 3.3.1 动态路由回顾

我们简要介绍用于胶囊表示学习的动态路由，胶囊是一种由向量表示的新型神经单元。假设我们有两层胶囊，我们将第一层和第二层的胶囊分别称为低级胶囊和高级胶囊。动态路由的目标是以迭代的方式，在给定低级胶囊值的情况下计算高级胶囊的值。在每次迭代中，给定低级胶囊 $i \in \{1,...,m\}$ 及其对应向量 $\mathbf{c}_i^{l} \in \mathbb{R}^{N_l \times 1}$ ，和高级胶囊 $j \in \{1,...,n\}$ 及其对应向量 $\mathbf{c}_j^{h} \in \mathbb{R}^{N_h \times 1}$ ，低级胶囊 $i$ 与高级胶囊 $j$ 之间的路由对数 $b_{ij}$ 通过下式计算：

$$
b_{ij} = (\mathbf{c}_j^{h})^{T} \mathbf{S}_{ij} \mathbf{c}_i^{l} \qquad (4)
$$

其中 $\mathbf{S}_{ij} \in \mathbb{R}^{N_h \times N_l}$ 表示待学习的双线性映射矩阵。

计算出路由对数后，高级胶囊 $j$ 的候选向量计算为所有低级胶囊的加权和：

$$
\mathbf{z}_j^{h} = \sum_{i=1}^{m} w_{ij} \mathbf{S}_{ij} \mathbf{c}_i^{l} \qquad (5)
$$

其中 $w_{ij}$ 表示连接低级胶囊 $i$ 和高级胶囊 $j$ 的权重，通过对路由对数执行softmax计算得到：

$$
w_{ij} = \frac{\exp(b_{ij})}{\sum_{k=1}^{m} \exp(b_{ik})} \qquad (6)
$$

最后，应用非线性"squash"函数获得高级胶囊的向量：

$$
\mathbf{c}_j^{h} = \text{squash}(\mathbf{z}_j^{h}) = \frac{\|\mathbf{z}_j^{h}\|^{2}}{1 + \|\mathbf{z}_j^{h}\|^{2}} \frac{\mathbf{z}_j^{h}}{\|\mathbf{z}_j^{h}\|} \qquad (7)
$$

$b_{ij}$ 的值初始化为零，路由过程通常重复三次以达到收敛。路由完成后，高级胶囊的值被固定，可以作为下一层的输入。

#### 3.3.2 B2I动态路由

简而言之，胶囊是一种新型神经元，由向量表示，而不是普通神经网络中使用的标量。基于向量的胶囊期望能够表示实体的不同属性，其中胶囊的方向表示一个属性，胶囊的长度用于表示该属性存在的概率。相应地，多兴趣提取层的目标是学习用于表达用户兴趣属性以及相应兴趣是否存在的表示。胶囊与兴趣表示之间的语义联系促使我们将行为/兴趣表示视为行为/兴趣胶囊，并采用动态路由从行为胶囊中学习兴趣胶囊。然而，最初为图像数据提出的路由算法不能直接用于处理用户行为数据。因此，我们提出了行为到兴趣（B2I，Behavior-to-Interest）动态路由，用于自适应地将用户行为聚合到兴趣表示向量中，它与原始路由算法在三个方面有所不同。

**共享双线性映射矩阵。** 我们使用固定的双线性映射矩阵 $\mathbf{S}$ ，而不是原始动态路由中为每对低级胶囊和高级胶囊分别使用单独的双线性映射矩阵，这基于两点考虑。一方面，用户行为是变长的，对于天猫用户而言从几十到几百不等，因此使用固定的双线性映射矩阵更具泛化性。另一方面，我们希望兴趣胶囊位于同一个向量空间中，但不同的双线性映射矩阵会将兴趣胶囊映射到不同的向量空间。因此，路由对数通过下式计算：

$$
b_{ij} = \mathbf{u}_j^{T} \mathbf{S} \mathbf{e}_i, \qquad i \in \mathcal{I}_u, \quad j \in \{1,...,K\} \qquad (8)
$$

其中 $\mathbf{e}_i \in \mathbb{R}^{d}$ 表示行为item $i$ 的嵌入， $\mathbf{u}_j \in \mathbb{R}^{d}$ 表示兴趣胶囊 $j$ 的向量。双线性映射矩阵 $\mathbf{S} \in \mathbb{R}^{d \times d}$ 在每对行为胶囊和兴趣胶囊之间共享。

**随机初始化的路由对数。** 由于使用了共享的双线性映射矩阵 $\mathbf{S}$ ，将路由对数初始化为零会导致初始兴趣胶囊相同。然后，后续迭代将陷入不同兴趣胶囊始终相同的困境。为缓解这一现象，我们从高斯分布 $\mathcal{N}(0, \sigma^2)$ 中采样随机矩阵用于初始化路由对数，使初始兴趣胶囊彼此不同，类似于成熟的K-Means聚类算法。

**动态兴趣数量。** 由于不同用户拥有的兴趣胶囊数量可能不同，我们引入一个启发式规则来自适应地调整不同用户的 $K$ 值。具体而言，用户 $u$ 的 $K$ 值计算为：

$$
K_u^{\prime} = \max(1, \min(K, \log_2(|\mathcal{I}_u|))) \qquad (9)
$$

这种调整兴趣胶囊数量的策略可以为兴趣较少的用户节省一些资源，包括计算资源和内存资源。

整个动态路由过程在算法1中列出。

**算法1：B2I动态路由。**

$$
\begin{aligned}
&\textbf{Input: } \text{行为嵌入 } \{\mathbf{e}_i, i \in \mathcal{I}_u\}，\text{迭代次数 } r，\text{兴趣胶囊数量 } K \\
&\textbf{Output: } \text{兴趣胶囊 } \{\mathbf{u}_j, j = 1, ..., K_u^{\prime}\} \\
&\text{通过公式(9)计算自适应兴趣胶囊数量 } K_u^{\prime} \\
&\textbf{for } \text{所有行为胶囊 } i \text{ 和兴趣胶囊 } j \textbf{ do} \\
&\quad \text{初始化 } b_{ij} \sim \mathcal{N}(0, \sigma^2) \\
&\textbf{end for} \\
&\textbf{for } k \leftarrow 1, r \textbf{ do} \\
&\quad \textbf{for } \text{所有行为胶囊 } i \textbf{ do} \\
&\quad\quad w_{ij} \leftarrow softmax(b_{ij}) \\
&\quad \textbf{end for} \\
&\quad \textbf{for } \text{所有兴趣胶囊 } j \textbf{ do} \\
&\quad\quad \mathbf{z}_j = \sum_{i \in \mathcal{I}_u} w_{ij} \mathbf{S} \mathbf{e}_i \\
&\quad\quad \mathbf{u}_j \leftarrow squash(\mathbf{z}_j) \\
&\quad \textbf{end for} \\
&\quad \textbf{for } \text{所有行为胶囊 } i \text{ 和兴趣胶囊 } j \textbf{ do} \\
&\quad\quad b_{ij} \leftarrow b_{ij} + \mathbf{u}_j^{T} \mathbf{S} \mathbf{e}_i \\
&\quad \textbf{end for} \\
&\textbf{end for} \\
&\textbf{return } \{\mathbf{u}_j, j = 1, ..., K_u^{\prime}\}
\end{aligned}
$$

### 3.4 标签感知注意力层

通过多兴趣提取层，从用户的行为嵌入中生成了多个兴趣胶囊。不同的兴趣胶囊表示用户兴趣的不同方面，相关的兴趣胶囊用于评估用户对特定item的偏好。因此，在训练期间，我们设计了一个基于缩放点积注意力的标签感知注意力层，使目标item能够选择使用哪个兴趣胶囊。具体而言，对于一个目标item，我们计算每个兴趣胶囊与目标item嵌入之间的兼容性，并计算兴趣胶囊的加权和作为用户针对该目标item的表示向量，其中每个兴趣胶囊的权重由相应的兼容性决定。在标签感知注意力中，标签是查询，兴趣胶囊既是键也是值，如图2所示。用户 $u$ 关于item $i$ 的输出向量计算为：

$$
\mathbf{v}_u = \text{Attention}(\mathbf{e}_i, \mathbf{V}_u, \mathbf{V}_u) = \mathbf{V}_u \text{softmax}(\text{pow}(\mathbf{V}_u^{T} \mathbf{e}_i, p))
$$

其中 pow 表示逐元素指数运算， $p$ 是一个可调参数，用于调整注意力分布。当 $p$ 接近0时，每个兴趣胶囊获得均匀的注意力。当 $p$ 大于1时，随着 $p$ 增加，具有更大点积的值将获得越来越多的权重。考虑极限情况，当 $p$ 趋于无穷时，注意力机制变成一种硬注意力，选择具有最大注意力的值而忽略其他值。在我们的实验中，我们发现使用硬注意力可以导致更快的收敛。

### 3.5 训练与服务

有了用户向量 $\mathbf{v}_u$ 和标签item嵌入 $\mathbf{e}_i$ ，我们计算用户 $u$ 与标签item $i$ 交互的概率为：

$$
\text{Pr}(i|u) = \text{Pr}(\mathbf{e}_i|\mathbf{v}_u) = \frac{\exp(\mathbf{v}_u^{T} \mathbf{e}_i)}{\sum_{j \in \mathcal{I}} \exp(\mathbf{v}_u^{T} \mathbf{e}_j)} \qquad (10)
$$

那么，训练MIND的整体目标函数为：

$$
L = \sum_{(u,i) \in \mathcal{D}} \log \text{Pr}(i|u) \qquad (11)
$$

其中 $\mathcal{D}$ 是包含用户-item交互的训练数据集合。由于item数量达到数十亿级别，分母中的求和运算在计算上是不可行的。因此，我们使用采样softmax技术来使目标函数可处理，并选择自适应矩估计（Adam，Adaptive Moment Estimation）优化器来训练MIND。

训练完成后，除了标签感知注意力层以外的MIND网络可以作为用户表示映射函数 $f_{user}$ 使用。在服务时，用户的行为序列和用户画像被输入 $f_{user}$ 函数，为每个用户生成多个表示向量。然后，这些表示向量通过近似最近邻方法用于检索Top $N$ item。与用户表示向量具有最高相似度的这些item被检索出来，构成推荐系统匹配阶段的最终候选item集合。请注意，当用户产生新的行为时，会改变其行为序列以及相应的用户表示向量，因此MIND实现了匹配阶段的实时个性化。

### 3.6 与现有方法的联系

在此，我们说明MIND与两种现有方法之间的关系，阐述它们的相似性和差异性。

**YouTube DNN。** MIND和YouTube DNN都利用深度神经网络对行为数据进行建模以生成用户表示，这些表示用于工业推荐系统匹配阶段的大规模item检索。然而，YouTube DNN使用一个向量表示一个用户，而MIND使用多个向量。当算法1中的 $K$ 值等于1时，MIND退化为YouTube DNN，因此MIND可以视为YouTube DNN的泛化。

**DIN。** 在捕捉用户多样化兴趣方面，MIND和DIN具有相似的目标。然而，这两种方法在实现目标的方式以及适用性上有所不同。为了处理多样化兴趣，DIN在item级别应用注意力机制，而MIND采用动态路由生成兴趣胶囊，并在兴趣级别考虑多样性。此外，DIN专注于排序阶段，因为它处理数千个item，而MIND将推断用户表示和衡量用户-item兼容性的过程解耦，使其适用于匹配阶段的十亿级item。

## 4 实验

### 4.1 离线评估

在本节中，我们在离线设置下，在多个数据集上展示MIND与现有方法在推荐准确性方面的比较。

#### 4.1.1 数据集与实验设置

我们选择两个数据集来评估推荐性能。一个是Amazon Books，是最广泛使用的电子商务推荐公共数据集之一。另一个称为TmallData，来自手机天猫App，包含随机抽样的200万天猫用户在10天内的历史行为。对于Amazon Books，我们只保留被评论至少10次的item和评论至少10个item的用户。对于TmallData，我们过滤掉被少于600个独立用户点击的item。两个数据集的统计信息如表1所示。

**表1：用于离线评估的两个数据集的统计信息。**

| 数据集 | 用户 | 商品 | 类别 | 样本 |
|--------|------|------|------|------|
| Amazon Books | 351,356 | 393,801 | 1 | 6,271,511 |
| TmallData | 2,014,865 | 934,751 | 6,377 | 50,929,802 |

我们选择下一项item预测问题，即预测用户的下一次交互，来评估方法的性能，因为这是推荐系统匹配阶段的核心任务。将每个数据集的用户-item交互数据按19:1的比例随机划分为训练集和测试集后，对于每个用户，随机选择一个该用户交互过的item作为目标item，而在目标item之前交互的item被收集作为用户行为。采用命中率作为衡量推荐性能的主要指标，定义为：

$$
\text{HitRate@}N = \frac{\sum_{(u,i) \in \mathcal{D}_{test}} \mathbb{I}\,(\text{target item occurs in top } N)}{|\mathcal{D}_{test}|} \qquad (12)
$$

其中 $\mathcal{D}_{test}$ 表示由用户和目标item对 $(u, i)$ 组成的测试集， $\mathbb{I}$ 表示指示函数。

#### 4.1.2 对比方法

- **WALS（Weighted Alternating Least Squares，加权交替最小二乘法）：** 是一种经典的矩阵分解算法，用于将用户-item交互矩阵分解为用户和item的隐藏因子。推荐基于用户和目标item隐藏因子之间的兼容性进行。

- **YouTube DNN:** 如上所述，YouTube DNN是最成功的用于工业推荐系统的深度学习方法之一。

- **MaxMF:** 该方法引入了一种高度可扩展的方法来学习非线性潜在因子分解，以建模多个用户兴趣。

#### 4.1.3 实验结果

**表2：不同方法在两个数据集上的HitRate，其中最佳性能以粗体显示。HP表示超参数，包括兴趣数量 $K$ 和嵌入维度 $d$ 。仅展示具有最佳性能超参数的结果以展示相应方法的有效性。括号中的百分比表示相对于YouTube DNN的相对提升。**

| 数据集 | HP | 指标 | WALS | YouTube DNN | MaxMF- $K$ -interest | MIND-1-interest | MIND- $K$ -interest |
|--------|----|------|------|-------------|--------------------|-----------------|--------------------|
| Amazon Books | $K=3, d=36$ | HR@10 | 0.0144 (-37.66%) | 0.0231 | 0.0285 (+23.38%) | 0.0273 (+18.18%) | 0.0309 (+33.77%) |
| | | HR@50 | 0.0553 (-25.87%) | 0.0746 | 0.0862 (+15.55%) | 0.0978 (+31.10%) | 0.1101 (+47.59%) |
| | | HR@100 | 0.0907 (-20.65%) | 0.1143 | 0.1304 (+14.09%) | 0.1459 (+27.65%) | 0.1631 (+42.69%) |
| TmallData | $K=5, d=64$ | HR@10 | 0.0372 (-36.84%) | 0.0589 | 0.0628 (+6.62%) | 0.0720 (+22.24%) | 0.0972 (+65.03%) |
| | | HR@50 | 0.0831 (-33.84%) | 0.1256 | 0.1820 (+44.90%) | 0.1512 (+20.38%) | 0.2080 (+65.60%) |
| | | HR@100 | 0.1126 (-31.67%) | 0.1648 | 0.2567 (+55.76%) | 0.1930 (+17.11%) | 0.2699 (+63.77%) |

表2总结了MIND以及基线方法在两个数据集上以HitRate@ $N$ （ $N=10,50,100$ ）衡量的性能。显然，MIND在两个数据集上均取得了与所有基线方法相当或更优的性能。矩阵分解（MF，Matrix Factorization）方法WALS被其他方法击败，揭示了深度学习在改进推荐系统匹配阶段方面的能力。然而，没有配备深度学习的情况下，MaxMF的表现远优于WALS，这可以解释为MaxMF将标准MF推广到非线性模型并采用了多个用户表示向量。可以观察到，采用多个用户表示向量的方法（MaxMF- $K$ -interest、MIND- $K$ -interest）通常优于其他方法（WALS、YouTube DNN、MIND-1-interest）。因此，使用多个用户表示向量被证明是建模用户多样化兴趣以及提升推荐准确性的有效方式。此外，我们可以观察到，多个用户表示向量带来的改进对于TmallData更为显著，因为天猫用户倾向于展现更多样化的兴趣。这种多样性的增加也可以从每个数据集的最佳 $K$ 值反映出来，TmallData的最佳 $K$ 值大于Amazon Books的最佳 $K$ 值。MIND-1-interest相对于YouTube DNN的改进表明，动态路由作为一种池化策略优于平均池化。考虑到MaxMF和MIND- $K$ -interest的结果，验证了通过动态路由从用户行为中提取多个兴趣优于MaxMF中使用的非线性建模策略。这可以归因于两点：(1)多兴趣提取层利用聚类过程生成兴趣表示，实现了更精确的用户表示。(2)标签感知注意力层使目标item能够关注多个用户表示向量，实现了用户兴趣与目标item之间更准确的匹配。

通过在一组根据每个数据集的规模和数据分布预定义的参数上进行实验，进行嵌入向量维度 $d$ 和用户兴趣数量 $K$ 的超参数调优，并且每种方法都使用最佳超参数进行测试，以实现公平比较。

**图3：超参数的影响。** 上半部分表明MIND在不同 $\sigma$ 下可以获得相当的结果；下半部分表明MIND在更大的 $p$ 下表现更好。

### 4.2 超参数分析

在本节中，我们在Amazon Books上进行了两个实验，研究多兴趣提取层和标签感知注意力层中超参数的影响。

**路由对数的初始化。** 多兴趣提取层中采用的路由对数随机初始化类似于K-means质心的初始化，其中初始聚类中心的分布对最终聚类结果有很强的影响。由于路由对数根据高斯分布 $\mathcal{N}(0, \sigma^2)$ 初始化，我们关注不同的 $\sigma$ 值可能导致不同的收敛，从而影响性能。为了研究 $\sigma$ 的影响，我们使用3个不同的 $\sigma$ 值（0.1、1和5）初始化路由对数 $b_{ij}$ 。结果如图3上半部分所示，3个值的曲线几乎重叠。这一观察结果表明MIND对 $\sigma$ 值具有鲁棒性，因此在实际应用中选择 $\sigma=1$ 是合理的。

**标签感知注意力中的幂指数。** 如前所述，标签感知注意力中的幂指数 $p$ 控制每个兴趣对组合后的标签感知兴趣表示的比例。我们比较了 $p$ 从0到 $\infty$ 变化时MIND的性能，结果如图3下半部分所示。显然， $p=0$ 的性能远差于其他取值。原因是当 $p=0$ 时，每个兴趣具有相同的注意力，因此组合后的兴趣表示等于所有兴趣的平均值，与标签无关。当 $p \geq 1$ 时，注意力分数与兴趣表示向量和目标item嵌入之间的相似度成比例，这使得组合后的兴趣表示成为兴趣的加权和。结果还表明，随着 $p$ 增加性能变得更好，因为与目标item具有更高相似度的兴趣表示向量获得更大的注意力，当 $p=\infty$ 时演变为硬注意力机制。通过这种机制，最接近目标item的兴趣表示主导组合后的兴趣表示，使MIND收敛更快且性能最佳。

### 4.3 在线实验

我们通过将MIND部署在天猫首页处理真实流量一周来进行在线实验。为了公平比较，所有部署在匹配阶段的方法之后都跟随相同的排序流程。CTR（点击率），一种广泛使用的工业指标，用于衡量方法在服务在线流量时的性能。

**图4：一周内的在线CTR。具有5~7个兴趣的MIND在所有对比方法中表现最佳。MIND显著优于两个基线方法，即基于item的CF和YouTube DNN。**

在线实验有两个基线方法。一个是基于item的协同过滤（CF，Collaborative Filtering），它是服务于大部分在线流量的基础匹配算法。另一个是YouTube DNN，它是众所周知的基于深度学习的匹配模型。我们在A/B测试框架中部署了所有对比方法，每种方法检索一千个候选item，然后送入排序阶段进行最终推荐。

实验结果总结在图4中。显然，MIND优于基于item的CF和YouTube DNN，这表明MIND生成了更好的用户表示。此外，我们做了如下观察：(1)经过长期实践优化，基于item的CF优于YouTube DNN，而单兴趣的MIND也超过了YouTube DNN。(2)一个非常明显的趋势是，随着提取的兴趣数量从1增加到5，MIND的性能越来越好。(3)当提取的兴趣数量达到5时MIND性能达到峰值，之后CTR保持恒定，7个兴趣的提升可忽略不计。(4)具有动态兴趣数量的MIND与具有7个兴趣的MIND性能相当。从上述观察中，我们得出几个结论。第一，对于天猫而言，用户兴趣的最优数量是5~7个，这揭示了用户兴趣的平均多样性。第二，动态兴趣数量机制并未带来CTR提升，但在实验过程中我们认识到该方案可以降低服务成本，这有利于大规模服务如天猫，在实践中更易采用。总之，在线实验验证了MIND在建模具有多样化兴趣的用户方面实现了更好的解决方案，并能显著提升整个推荐系统。

### 4.4 案例研究

#### 4.4.1 耦合系数

**图5：两个用户的耦合系数热力图。每一类行为在对应的兴趣上具有最大的耦合系数。用户C（上图）和用户D（下图）具有不同粒度的兴趣。**

行为胶囊与兴趣胶囊之间的耦合系数量化了行为对兴趣的隶属程度。在本节中，我们将这些耦合系数可视化以展示兴趣提取过程是可解释的。

图5展示了从天猫日活用户中随机选择的两个用户对应的耦合系数，其中每行对应一个兴趣胶囊，每列对应一个行为。它显示用户C（上图）与4类商品（耳机、零食、手袋和服装）进行了交互，每个类别的商品在一个兴趣胶囊上具有最大的耦合系数并形成了相应的兴趣。而用户D（下图）只对服装感兴趣，因此从行为中解析出了3个更细粒度的兴趣（毛衣、大衣和羽绒服）。关于这一结果，我们确认每一类用户行为被聚类在一起并形成相应的兴趣表示向量。

#### 4.4.2 item分布

**图6：与左侧示例用户行为相对应的、由每个兴趣召回的item分布。每个兴趣由一个坐标轴展示，其坐标为item与兴趣之间的相似度。点的大小与具有特定相似度的item数量成正比。**

在服务时，通过最近邻搜索检索与用户兴趣相似的item。我们根据每个兴趣对应的相似度，可视化由每个兴趣召回的item的分布。图6展示了图5（上）中同一用户（用户C）的item分布。这些分布分别通过两种方法获得，其中上面4个坐标轴展示了基于MIND的4个兴趣召回的item，而最下面的坐标轴展示了基于YouTube DNN召回的item。item根据它们与兴趣的相似度散落在坐标轴上，相似度通过最小-最大归一化缩放到0~1并四舍五入到最近的0.5。一个点由落在特定范围内的item汇集而成，因此每个点的大小代表具有相应相似度的item数量。我们还展示了一些从所有候选中随机选取的item。正如预期的那样，MIND召回的item与相应的兴趣高度相关，而YouTube DNN召回的item沿着item类别变化很大，并且与用户行为的相似度较低。

## 5 系统部署

**图7：天猫推荐系统架构。**

在本节中，我们描述MIND在天猫的实现和部署。一个由几个基础平台组成的典型工作流如图7所示，详细说明如下：

当用户启动手机天猫App时，推荐请求被发送到天猫个性化平台——一个集成了大量插件模块的服务器集群，作为天猫的在线推荐服务。用户的近期行为由天猫个性化平台检索并发送到用户兴趣提取器，这是实现MIND将用户行为转换为多个用户兴趣的主要模块。随后，召回引擎搜索其嵌入向量与用户兴趣最接近的item。由不同兴趣触发的item被合并为候选item，并根据它们与用户兴趣的相似度进行排序。由于基于MIND的服务的高效性，通过用户兴趣提取器和召回引擎从十亿级item池中选择数千个候选item的整个过程可以在不到15毫秒内完成。在item范围和系统响应时间之间进行权衡后，这些候选item中的前1000个由排序服务进行评分，该服务使用大量特征预测CTR。最后，天猫个性化平台完成作为推荐结果显示给用户的item列表。用户兴趣提取器和排序服务都在模型训练平台上使用100个GPU进行训练，训练可以在8小时内完成。得益于模型训练平台的优越性能，用于预测的深度网络每天更新，这保证了新发布的产品能够被计算和曝光。

## 6 结论与未来工作

在本文中，我们提出了一种新的神经网络结构——带动态路由的多兴趣网络（MIND），用于在电子商务推荐（涉及十亿级用户和item）的匹配阶段表示用户多样化的兴趣。具体而言，我们设计了一个带有变体动态路由的多兴趣提取层来提取用户的多样化兴趣，然后这些兴趣通过一种新颖的标签感知注意力机制进行训练。通过离线实验证明MIND在公开基准上达到了优越的性能。还报告了在线CTR以证明MIND在天猫线上生产环境中的有效性和可行性。对于未来工作，我们将追求两个方向。第一是融入更多关于用户行为序列的信息，如行为时间等。第二个方向是优化动态路由的初始化方案，参考K-means++初始化方案，以实现更好的用户表示。

## 致谢

我们感谢我们团队的同事——王继哲、Andreas Pfadler、徐佳明、陈文、王立峰、郭鑫和郭成——对这项工作的有益讨论和支持。我们感谢我们的合作团队——搜索工程团队。我们也感谢匿名审稿人提出的宝贵意见和建议，这些帮助提高了本文的质量。

## 参考文献

[1] Christopher R Aberger. 2016. Recommender: An analysis of collaborative filtering techniques. Technical Report.

[2] Silvio Amir, Byron C. Wallace, Hao Lyu, Paula Carvalho, and Mario J. Silva. 2016. Modelling Context with User Embeddings for Sarcasm Detection in Social Media. In Proceedings of The 20th SIGNLL Conference on Computational Natural Language Learning. Association for Computational Linguistics, 167–177.

[3] Zeynep Batmaz, Ali Yurekli, Alper Bilge, and Cihan Kaleli. 2018. A review on deep learning for recommender systems: challenges and remedies. Artificial Intelligence Review (2018), 1–37.

[4] Robert M Bell and Yehuda Koren. 2007. Improved neighborhood-based collaborative filtering. In KDD cup and workshop at the 13th ACM SIGKDD international conference on knowledge discovery and data mining. Citeseer, 7–14.

[5] Iván Cantador, Alejandro Bellogín, and David Vallet. 2010. Content-based recommendation in social tagging systems. In Proceedings of the fourth ACM conference on Recommender systems. ACM, 237–240.

[6] Tao Chen, Ruifeng Xu, Yulan He, Yunqing Xia, and Xuan Wang. 2016. Learning user and product distributed representations using a sequence model for sentiment analysis. IEEE Computational Intelligence Magazine 11, 3 (2016), 34–44.

[7] Paul Covington, Jay Adams, and Emre Sargin. 2016. Deep neural networks for youtube recommendations. In Proceedings of the 10th ACM Conference on Recommender Systems. ACM, 191–198.

[8] Ali Mamdouh Elkahky, Yang Song, and Xiaodong He. 2015. A multi-view deep learning approach for cross domain user modeling in recommendation systems. In Proceedings of the 24th International Conference on World Wide Web. International World Wide Web Conferences Steering Committee, 278–288.

[9] Huifeng Guo, Ruiming Tang, Yunming Ye, Zhenguo Li, and Xiuqiang He. 2017. DeepFM: A Factorization-machine Based Neural Network for CTR Prediction. In Proceedings of the 26th International Joint Conference on Artificial Intelligence (IJCAI'17). AAAI Press, 1725–1731.

[10] Ruining He and Julian McAuley. 2016. Ups and downs: Modeling the visual evolution of fashion trends with one-class collaborative filtering. In proceedings of the 25th international conference on world wide web. International World Wide Web Conferences Steering Committee, 507–517.

[11] Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu, and Tat-Seng Chua. 2017. Neural collaborative filtering. In Proceedings of the 26th International Conference on World Wide Web. International World Wide Web Conferences Steering Committee, 173–182.

[12] Jon Herlocker, Joseph A Konstan, and John Riedl. 2002. An empirical analysis of design choices in neighborhood-based collaborative filtering algorithms. Information retrieval 5, 4 (2002), 287–310.

[13] Geoffrey E Hinton, Alex Krizhevsky, and Sida D Wang. 2011. Transforming auto-encoders. In International Conference on Artificial Neural Networks. Springer, 44–51.

[14] Geoffrey E Hinton, Sara Sabour, and Nicholas Frosst. 2018. Matrix capsules with EM routing. In International Conference on Learning Representations.

[15] Jeff Johnson, Matthijs Douze, and Hervé Jégou. 2017. Billion-scale similarity search with gpus. arXiv preprint arXiv:1702.08734 (2017).

[16] Diederik P Kingma and Jimmy Ba. 2014. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980 (2014).

[17] Yehuda Koren, Robert Bell, and Chris Volinsky. 2009. Matrix factorization techniques for recommender systems. Computer 8 (2009), 30–37.

[18] Rodney LaLonde and Ulas Bagci. 2018. Capsules for Object Segmentation. arXiv preprint arXiv:1804.04241 (2018).

[19] Yann LeCun, Yoshua Bengio, and Geoffrey Hinton. 2015. Deep learning. nature 521, 7553 (2015), 436.

[20] Julian McAuley, Christopher Targett, Qinfeng Shi, and Anton Van Den Hengel. 2015. Image-based recommendations on styles and substitutes. In Proceedings of the 38th International ACM SIGIR Conference on Research and Development in Information Retrieval. ACM, 43–52.

[21] Sara Sabour, Nicholas Frosst, and Geoffrey E Hinton. 2017. Dynamic routing between capsules. In Advances in Neural Information Processing Systems. 3856–3866.

[22] Badrul Sarwar, George Karypis, Joseph Konstan, and John Riedl. 2001. Item-based collaborative filtering recommendation algorithms. In Proceedings of the 10th international conference on World Wide Web. ACM, 285–295.

[23] Jiaxi Tang and Ke Wang. 2018. Personalized top-n sequential recommendation via convolutional sequence embedding. In Proceedings of the Eleventh ACM International Conference on Web Search and Data Mining. ACM, 565–573.

[24] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information Processing Systems. 5998–6008.

[25] Jizhe Wang, Pipei Huang, Huan Zhao, Zhibo Zhang, Binqiang Zhao, and Dik Lun Lee. 2018. Billion-scale Commodity Embedding for E-commerce Recommendation in Alibaba. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD '18). 839–848.

[26] Jason Weston, Ron J Weiss, and Hector Yee. 2013. Nonlinear latent factorization by embedding multiple user interests. In Proceedings of the 7th ACM conference on Recommender systems. ACM, 65–68.

[27] Hong-Jian Xue, Xinyu Dai, Jianbing Zhang, Shujian Huang, and Jiajun Chen. 2017. Deep Matrix Factorization Models for Recommender Systems. In IJCAI. 3203–3209.

[28] Min Yang, Wei Zhao, Jianbo Ye, Zeyang Lei, Zhou Zhao, and Soufei Zhang. 2018. Investigating Capsule Networks with Dynamic Routing for Text Classification. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics, 3110–3119.

[29] Hongzhi Yin, Bin Cui, Ling Chen, Zhiting Hu, and Xiaofang Zhou. 2015. Dynamic user modeling in social media systems. ACM Transactions on Information Systems (TOIS) 33, 3 (2015), 10.

[30] Yang Yu, Xiaojun Wan, and Xinjie Zhou. 2016. User embedding for scholarly microblog recommendation. In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), Vol. 2. 449–453.

[31] Guorui Zhou, Xiaoqiang Zhu, Chenru Song, Ying Fan, Han Zhu, Xiao Ma, Yanghui Yan, Junqi Jin, Han Li, and Kun Gai. 2018. Deep interest network for click-through rate prediction. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 1059–1068.
