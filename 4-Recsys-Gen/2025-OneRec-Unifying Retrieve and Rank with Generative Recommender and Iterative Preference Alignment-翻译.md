# OneRec: Unifying Retrieve and Rank with Generative Recommender and Preference Alignment

> 邓佳欣*、王诗瑶*、蔡阔*、任乐健*、胡启根*、丁伟峰*、罗强*、周国瑞*†
> 快手科技，北京，中国
> *共同第一作者；†通讯作者

本文介绍了快手提出的 OneRec 模型，用统一的生成式模型取代传统的检索-排序级联框架。核心内容：

- **编码器-解码器结构**：编码用户历史行为序列，逐步解码用户感兴趣的视频
- **会话级生成**：替代逐点生成，实现上下文更连贯的推荐结果
- **迭代偏好对齐**：结合 DPO 与奖励模型，对齐用户兴趣偏好

关键发现：OneRec 部署于快手主要场景，观看时长提升 1.6%，是首个在实际场景中显著超越传统级联推荐系统的端到端生成模型。

---

## 摘要

近年来，基于生成式检索的推荐系统（GR）已成为一种有前景的范式，它通过自回归方式直接生成候选视频。然而，大多数现代推荐系统采用检索-排序（retrieve-and-rank）策略，其中生成模型仅在检索阶段作为选择器。在本文中，我们提出了 **OneRec**，它用统一的生成模型取代了级联学习框架。据我们所知，这是第一个在实际场景中显著超越当前复杂且精心设计的推荐系统的端到端生成模型。具体来说，OneRec 包括：1）**编码器-解码器结构**，它对用户的历史行为序列进行编码，并逐步解码出用户可能感兴趣的视频。我们采用稀疏混合专家（MoE）来扩展模型容量，而无需按比例增加计算 FLOPs。2）**会话级（session-wise）生成方法**。与传统的下一项预测不同，我们提出会话级生成，它比依赖于手工规则来适当组合生成结果的逐点生成更加优雅且上下文连贯。3）**迭代偏好对齐模块**结合直接偏好优化（DPO）来提升生成结果的质量。与 NLP 中的 DPO 不同，推荐系统通常只有一次机会为每个用户的浏览请求展示结果，因此无法同时获得正负样本。为了解决这一限制，我们设计了一个奖励模型来模拟用户生成，并根据推荐系统在线学习的属性定制采样策略。大量实验表明，有限数量的 DPO 样本可以对齐用户兴趣偏好，并显著提高生成结果的质量。我们将 OneRec 部署在快手（一个拥有数亿日活跃用户的短视频推荐平台）的主要场景中，观看时长提升了 1.6%，这是一个显著的改进。

**CCS 概念：** • 信息系统 $\to$ 计算广告；多媒体信息系统

**关键词：** 生成式推荐，自回归生成，语义分词，直接偏好优化

---

## 1 引言

为了平衡效率和效果，大多数现代推荐系统采用级联排序策略 [6, 26, 34, 43]。如图 1(b) 所示，一个典型的级联排序系统采用三阶段流水线：召回 [6, 19, 54]、粗排 [28, 46] 和精排 [2, 3, 15, 16, 33, 52, 53]。每个阶段负责从接收到的item中选择 top-k item并将结果传递给下一阶段，共同平衡系统响应时间和排序精度。

**图 1：(a) 我们提出的端到端生成统一架构。(b) 典型的级联排序系统，从下到上包括三个阶段：检索、粗排和精排。**

尽管在实践中高效，但现有方法通常独立处理每个排序器，每个孤立阶段的效果作为后续排序阶段的上限，从而限制了整体排序系统的性能。尽管有多种努力 [11, 13, 18, 20, 34, 44] 通过使排序器之间能够交互来改善整体推荐性能，但它们仍然保持传统的级联排序范式。最近，基于生成式检索的推荐系统（GR）[36, 45, 51] 已成为一种有前景的范式，它通过自回归序列生成方式直接生成候选item的标识符。通过用量化的语义ID [24] 索引item来编码item语义，推荐器可以利用item内丰富的语义信息。GR 的生成性质使其适合通过束搜索解码直接选择候选item，并产生更多样化的推荐结果。然而，当前的生成模型仅作为检索阶段的选择器，其推荐精度尚未能与精心设计的多级级联排序器相匹配。

为了解决上述挑战，我们提出了一个统一的端到端生成式框架，用于单阶段推荐，命名为 **OneRec**。首先，我们提出一个编码器-解码器架构。从训练大型语言模型中观察到的缩放定律获得灵感，我们发现扩展推荐模型的容量也能持续改善性能。因此，我们基于 MoE [7, 9, 55] 结构扩展模型参数，这显著提高了模型刻画用户兴趣的能力。其次，与传统的逐点预测下一个item不同，我们提出了一种**会话级列表生成**方法，该方法考虑了每个会话内item的相对内容和顺序。逐点生成方法需要手工策略来确保生成结果的连贯性和多样性。相比之下，会话级学习过程使模型能够通过提供偏好数据来自主学习最优的会话结构。最后但同样重要的是，我们探索了通过使用直接偏好优化（DPO）[35] 进行偏好学习，以进一步提高生成结果的质量。为了构建偏好对，我们从硬负采样 [37] 中获得灵感，从束搜索结果中创建自生成硬拒绝样本，而不是随机采样。我们提出了一种迭代偏好对齐（IPA）策略，基于预训练的奖励模型（RM）提供的分数对采样响应进行排序，识别最佳的 chosen 样本和最差的 rejected 样本。我们在大规模工业数据集上的实验显示了所提出方法的优越性。我们还进行了一系列消融实验，以详细证明每个模块的有效性。本文的主要贡献总结如下：

- 为克服级联排序的限制，我们引入了 **OneRec**，一个单阶段生成式推荐框架。据我们所知，这是第一个在工业环境中使用单一生成模型显著超越传统多阶段排序流水线的工作。
- 我们强调了**模型容量**和**目标item的上下文信息**通过会话级生成方式的必要性，这使得预测更加准确，并增强了生成item的多样性。
- 我们提出了一种基于个性化奖励模型的**新颖的自生成硬负样本选择策略**。通过直接偏好优化，我们增强了 OneRec 在更广泛的用户偏好范围上的泛化能力。大量的离线实验和在线 A/B 测试证明了其有效性和效率。

---

## 2 相关工作

### 2.1 生成式推荐

近年来，随着生成模型的显著进展，生成式推荐受到了越来越多的关注。与传统的基于嵌入的检索方法（主要依赖双塔模型计算每个候选item的排序分数，并利用高效的 MIPS 或 ANN [14, 17, 21, 31, 38] 搜索系统检索 top-k 相关item）不同，生成式检索（GR）[41] 方法将检索数据库中相关文档的问题形式化为序列生成任务，按顺序生成相关文档的token。文档token可以是文档标题、文档 ID 或预训练的语义 ID [42]。GENRE [8] 首先采用 Transformer 架构进行实体检索，以自回归方式基于条件上下文生成实体名称。DSI [42] 首次提出了为文档分配结构化语义 ID 的概念，并训练编码器-解码器模型进行生成式文档检索。遵循这一范式，TIGER [36] 将生成式item检索模型的形式化引入推荐系统。

除了生成框架，如何索引item也引起了越来越多的关注。最近的研究集中在语义索引技术 [12, 36, 42] 上，旨在基于内容信息索引item。具体来说，TIGER [36] 和 LC-Rec [51] 将残差量化（RQ-VAE）应用于从item标题和描述中导出的文本嵌入进行分词。RecForest [12] 利用层次化 k-means 聚类对item文本嵌入进行聚类，以获得聚类索引作为token。此外，最近的研究如 EAGER [45] 探索将语义和协同信息都集成到分词过程中。

### 2.2 语言模型的偏好对齐

在大型语言模型（LLM）的后训练 [10] 阶段，基于人类反馈的强化学习（RLHF）[32, 39] 是一种通过使用代表人类反馈的奖励模型引导的强化学习技术来使 LLM 与人类价值观对齐的主流方法。然而，RLHF 存在不稳定和低效的问题。直接偏好优化（DPO）[35] 被提出，它通过闭式推导最优策略，并能够直接使用偏好数据进行优化。除此之外，还提出了几种变体来进一步改进原始 DPO。例如，IPO [1] 通过通用目标绕过了 DPO 中的两个近似。cDPO [35] 通过引入超参数 $\varepsilon$ 来缓解噪声标签的影响。rDPO [5] 设计了原始二元交叉熵损失的无偏估计。其他变体包括 CPO [47]、simDPO [5] 也在各个方面增强或扩展了 DPO。然而，与通过人类明确标注偏好数据的传统 NLP 场景不同，推荐系统中的偏好学习面临独特的挑战，因为用户-item交互数据稀疏。这一挑战导致 DPO 在推荐中的改编仍 largely unexplored。与 S-DPO（专注于在基于 LM 的推荐器中整合多个负样本的用户偏好数据）不同，我们训练了一个奖励模型，并根据奖励模型的分数为不同用户选择个性化的偏好数据。

---

## 3 方法

在本节中，我们提出 OneRec，一个通过单阶段检索方式生成目标item的端到端框架。

### 3.1 预备知识

在本节中，我们从特征工程的角度介绍单阶段生成式推荐流水线的构建。对于用户侧特征，OneRec 将正向历史行为序列 $H_u = \{v^h_1, v^h_2, ..., v^h_n\}$ 作为输入，其中 $v$ 表示用户有效观看或交互过（点赞、关注、分享）的视频， $n$ 是行为序列的长度。OneRec 的输出是一个视频列表，由一个会话 $S = \{v_1, v_2, ..., v_m\}$ 组成，其中 $m$ 是会话内的视频数量（"会话"的详细定义见第 3.2 节）。

对于每个视频 v_i，我们使用与真实用户-item行为分布对齐的多模态嵌入 e_i
$$
\in
$$
 R^d 来描述它们 [27]。基于预训练的多模态表示，现有的生成式推荐框架 [25, 36] 使用 RQ-VAE [49] 将嵌入编码为语义token。然而，由于不平衡的编码分布（称为沙漏现象 [23]），这种方法不是最优的。我们应用一种多级均衡量化机制，使用残差 K-means 量化算法 [27] 来转换 e_i。在第一级（l=1），初始残差定义为 r^1_i = e_i。在每一级 l，我们有一个码本 C_l = {c^l_1, ..., c^l_K}，其中 K 是码本大小。通过 argmin 找到最近质心嵌入的索引，下一级的残差定义为 r^{l+1}_i = r^l_i - c^ $l_{s^l_i}$ 。因此，对应的码本token通过层次化索引生成。为了构建均衡码本，我们应用算法 1 中详述的均衡 K-means 对item集进行划分。给定总视频集 V，该算法将集合划分为 K 个聚类，每个聚类恰好包含 w = |V|/K 个视频。在迭代计算中，每个质心顺序分配其 w 个最近的未分配视频（基于欧氏距离），然后使用分配视频的均值向量进行质心重新校准。当聚类分配达到收敛时终止。

### 3.2 会话级列表生成

与仅从用户的历史交互中预测下一个item的逐点生成方法不同，OneRec 旨在基于用户的历史交互序列生成一个高价值会话的列表，这使推荐模型能够捕获推荐列表中视频之间的依赖关系。具体来说，一个会话是指在一次用户请求中返回的一批短视频，通常由 5 到 10 个视频组成。

会话定义为关于用户兴趣、连贯性和多样性的自然单元。我们设计了几个标准来识别高质量的会话，包括：
- 用户在会话中实际观看的短视频数量大于等于 5；
- 用户观看该会话的总时长超过一定阈值；
- 用户表现出交互行为，如点赞、收藏或分享视频。

这种方法确保我们的会话级模型从真实的用户参与模式中学习，并捕获会话列表中更准确的上下文信息。因此，我们的会话级模型 M 的目标可以形式化为：

S := M(H_u)   (1)

其中 H_u 由语义 ID 表示。我们的模型采用基于 Transformer 的框架，由两个主要组件组成：用于建模用户历史交互的编码器和用于会话列表生成的解码器。编码器是一个 Transformer 编码器，它将 H_u 作为输入通过自注意力层处理。解码器是一个 Transformer 解码器，它接收编码器的输出 H = Encoder(H_u)，并以自回归方式生成目标。为了以合理的经济成本训练更大的模型，对于解码器中的前馈神经网络（FNN），我们采用了 MoE 架构 [7, 9, 55] 来替代密集 FNN 层，类似于基于 Transformer 的语言模型。

在训练中，我们在目标会话的语义 ID 开头添加起始token s[BOS] 来构建解码器输入。我们使用交叉熵损失进行下一个token预测。在会话级列表生成任务上进行一定量的训练后，我们获得种子模型 M_t。

### 3.3 基于奖励模型的迭代偏好对齐

第 3.2 节定义的高质量会话提供了有价值的训练数据，使模型能够学习什么是好的推荐结果。然而，为了进一步提高模型生成内容的质量，我们探索通过直接偏好优化（DPO）来进行偏好学习。在传统的自然语言处理（NLP）场景中，偏好数据由人类明确标注。然而，推荐系统中的偏好学习面临独特的挑战，因为用户-item交互数据稀疏，这需要一个奖励模型（RM）。因此，我们在第 3.3.1 节中引入了会话级奖励模型。此外，我们通过提出一种迭代直接偏好优化方法来改进传统 DPO，使模型能够自我改进，如第 3.3.2 节所述。

#### 3.3.1 奖励模型训练

我们使用 $R(u, S)$ 表示奖励模型，它为不同用户选择偏好数据。在这里，输出 $r$ 表示用户 $u$ （通常由用户行为表示）对会话 $S = \{v_1, v_2, ..., v_m\}$ 的偏好对应的奖励。为了使 RM 具有对会话进行排序的能力，我们首先提取会话 $S$ 中每个item $v_i$ 的目标感知表示 $e_i = v_i \odot u$ ，其中 $\odot$ 表示目标感知操作（如对用户行为的目标注意力）。因此，我们得到会话 $S$ 的目标感知表示 $h = \{e_1, e_2, ..., e_m\}$ 。然后，会话内的item通过自注意力层相互交互以融合必要的信息：

 $h_f = \text{SelfAttention}(hW^Q_s, hW^K_s, hW^V_s)$   (5)

接下来，我们使用不同的 Tower 对多目标奖励进行预测：

 $\hat{r}_{swt} = \text{Tower}_{swt}(\text{Sum}(h_f))$ , $\hat{r}_{vtr} = \text{Tower}_{vtr}(\text{Sum}(h_f))$ , ...   (6)

其中 $\text{Tower}(\cdot) = \text{Sigmoid}(\text{MLP}(\cdot))$ 。

得到所有估计奖励 $\hat{r}_{swt}, ...$ 和真实标签 $y_{swt}, ...$ 后，对于每个会话，我们直接最小化二元交叉熵损失来训练 RM。

#### 3.3.2 迭代偏好对齐

基于预训练的 RM R(u, S) 和当前 OneRec M_t，我们通过束搜索为每个用户生成 N 个不同的响应：

 $S^n_u \sim M_t(H_u)$ , $\forall u \in U$ , $n \in [N]$   (8)

然后我们基于 RM $R(u, S)$ 计算每个响应的奖励 $r^n_u$ ：

r^n_u = R(u, S^n_u)   (9)

选择具有最高奖励值的获胜响应 (S^w_u, H_u) 和具有最低奖励值的失败响应 (S^l_u, H_u)。给定偏好对，我们现在可以训练一个新模型 $M_{t+1}$ ，它从模型 M_t 初始化，并使用结合了 DPO 损失 [35] 的损失函数进行更新，以从偏好对中学习。

如算法 2 和图 2(b) 所示，整个过程涉及训练一系列模型 $M_t, ..., M_T$ 。为了减轻束搜索推理期间的计算负担，我们仅随机采样 $r_{DPO} = 1\%$ 的数据进行偏好对齐。对于每个后续模型 $M_{t+1}$ ，它从前一个模型 $M_t$ 初始化，并使用 $M_t$ 生成的偏好数据 $D_{pairs}^t$ 进行训练。

### 3.4 模型扩展

鉴于 OneRec-0.1B 作为基础配置，我们通过增加解码器中的 MoE 层数将模型扩展到 OneRec-1B。MoE 架构使模型容量能够显著增长，而计算 FLOPs 仅略有增加。在推理过程中，即使模型有 1B 参数，每个token仅激活约 130M 参数（由于 top-2 路由选择）。

---

## 4 系统部署

OneRec 已成功应用于真实工业场景。平衡稳定性和性能，我们部署 OneRec-1B 进行在线服务。如图 3 所示，我们的部署架构由三个核心组件组成：1）训练系统，2）在线服务系统，3）DPO 采样服务器。系统将收集到的交互日志作为训练数据处理，最初采用下一个token预测目标 L_NTP 训练种子模型。收敛后，我们添加 DPO 损失 L_DPO 进行偏好对齐，利用 XLA 和 bfloat16 混合精度训练来优化计算效率和内存利用率。训练好的参数同步到在线推理模块和 DPO 采样服务器，用于实时服务和基于偏好的数据选择。为了提升推理性能，我们实现了两个关键优化：键值缓存解码机制结合 float16 量化以减少 GPU 内存开销，以及束大小 128 的束搜索配置以平衡生成质量和延迟。此外，得益于 MoE 架构，推理期间仅激活 13% 的参数。

**图 2：OneRec 的整体框架，包括两个阶段：(a) 会话训练阶段，使用会话级数据训练 OneRec；(b) IPA 阶段，使用自生成硬负样本进行迭代直接偏好优化。**

**图 3：OneRec 的在线部署框架。**

---

## 5 实验

在本节中，我们首先在离线设置中将 OneRec 与逐点方法和几种 DPO 变体进行比较。然后，我们对提出的模块进行消融实验以验证 OneRec 的有效性。最后，我们将 OneRec 部署到线上并进行 A/B 测试，以进一步验证其在快手的性能。

### 5.1 实验设置

**实现细节：** 我们的模型使用 Adam 优化器训练，初始学习率 2
$$
\times
$$
10⁻⁴。我们使用 NVIDIA A800 GPU 进行 OneRec 优化。DPO 采样率 r_DPO 在整个训练中设为 1%，我们为每个用户通过束搜索生成 N=128 个不同响应；语义标识符聚类过程对每个码本层使用 K=8192 个聚类，码本层数设为 L=3；MoE 架构包含 N_MoE=24 个专家，每次前向传播通过 top-k 选择激活 K_MoE=2 个专家；对于会话建模，我们考虑 m=5 个目标会话item，并采用 n=256 个历史行为作为上下文。

**基线方法：** 我们采用以下代表性推荐模型、DPO 及其变体作为额外基线进行比较：
- **SASRec [22]**：使用单向 Transformer 架构捕获用户-item交互中的序列依赖关系，用于下一项预测。
- **BERT4Rec [40]**：利用双向 Transformer 和掩码语言建模，通过序列重建学习上下文item表示。
- **FDSA [50]**：实现双自注意力路径，在异构推荐场景中联合建模item级转换和特征级变换模式。
- **TIGER [36]**：利用层次化语义标识符和生成式检索技术，通过自回归序列生成进行序列推荐。
- **DPO [35]**：通过隐式奖励建模，从人类反馈数据中以闭式奖励函数形式化偏好优化。
- 以及其他 DPO 变体如 IPO、cDPO、rDPO、CPO、simPO、S-DPO。

**评估指标：** 我们使用几个关键指标评估模型性能。每个指标在评估模型输出的不同方面时服务于不同的目的，我们在每次迭代中在随机采样的测试用例集上进行评估。为了估计每个特定用户-会话对的各类交互概率，我们使用预训练的奖励模型评估推荐会话的价值。我们计算不同目标指标的平均奖励，包括会话观看时间（swt）、观看概率（vtr）、关注概率（wtr）和点赞概率（ltr）。其中，swt 和 vtr 是观看时长指标，而 wtr 和 ltr 是交互指标。

### 5.2 离线性能

表 1 展示了 OneRec 与各种基线的全面比较。对于观看时长指标，我们主要关心会话观看时间（swt）和交互指标中的点赞概率（ltr）。我们的结果揭示了三个关键观察：

**第一，提出的会话级生成方法显著优于传统的基于点积的方法和逐点生成方法（如 TIGER）。** OneRec-1B 相比 TIGER-1B 在最大 swt 上提高了 1.78%，在最大 ltr 上提高了 3.36%。这证明了会话级建模在保持推荐内容上下文连贯性方面的优势，而逐点方法难以平衡生成输出的连贯性和多样性。

**第二，小比例的 DPO 训练带来了显著的收益。** 仅使用 1% 的 DPO 训练比例（r_DPO），OneRec-1B+IPA 相比基础 OneRec-1B 在最大 swt 上提升了 4.04%，在最大 ltr 上提升了 5.43%。这表明有限的 DPO 训练可以有效地使模型与期望的生成模式对齐。

**第三，提出的 IPA 策略优于各种现有的 DPO 变体。** 如表 1 所示，IPA 相比其他 DPO 实现取得了优越的性能。值得注意的是，一些 DPO 基线的表现甚至不如未对齐的 OneRec-1B 模型，这表明从自生成输出中迭代挖掘偏好数据比其他方法更有效。

### 5.3 消融研究

#### 5.3.1 DPO 采样率消融

为了研究 DPO 训练中采样率 r_DPO 的影响，我们在受控条件下将 DPO 采样率从 1% 变化到 5%。消融结果表明，增加采样率在多个评估目标上带来了边际性能提升。值得注意的是，尽管计算开销增加，但以 1% 基线为基准的性能增益仍然不显著。值得注意的是，DPO 采样率和 GPU 资源利用率之间存在线性关系：5% 的采样率需要比 1% 基线多 5 倍的 GPU 资源。这种缩放特性在计算效率和模型性能之间建立了明确的权衡。因此，在平衡计算效率和性能的最佳权衡后，我们应用 1% 的 DPO 采样率进行训练，这在仅需 20% 计算资源的情况下实现了约 95% 的最大观测性能。

#### 5.3.2 模型缩放消融

我们评估了 OneRec 在模型规模增加时的表现。将 OneRec 从 0.05B 扩展到 1B 实现了持续的精度提升，展示了一致的缩放特性。具体来说，与 OneRec-0.05B 相比，OneRec-0.1B 实现了显著的最大精度提升 14.45%，而扩展到 0.2B、0.5B 和 1B 时可以额外获得 5.09%、5.70% 和 5.69% 的精度增益。

### 5.4 OneRec 的预测动态

如图 5 所示，我们展示了不同层上 8192 个编码的预测概率分布，其中红星表示具有最高奖励值的item的语义 ID。与 OneRec 基线相比，OneRec+IPA 在预测分布中表现出显著的置信度转移，表明我们提出的偏好对齐策略有效地鼓励基础模型产生偏好的生成模式。此外，我们观察到第一层的概率分布表现出更大的分散度（熵=6.00），而后续层（第二层平均熵=3.71，第三层熵=0.048）呈现出逐渐集中的分布。这种层次化的不确定性降低可以归因于自回归解码机制：初始层的预测继承了先前解码步骤的更高不确定性，而后续层受益于约束决策空间的累积上下文。

### 5.5 在线 A/B 测试

为了评估 OneRec 的在线性能，我们在快手视频推荐场景的主页上进行严格的在线 A/B 测试，并将 OneRec 与当前多阶段推荐系统在 1% 的主流量上进行比较。我们使用总观看时长来衡量用户观看视频的总时间，平均观看时长计算用户在被推荐系统展示请求会话时每个视频的平均观看时间。在线评估显示，OneRec 在总观看时长上实现了 1.68% 的提升，在平均观看时长上实现了 6.56% 的提升，这表明 OneRec 取得了更好的推荐结果，并为平台带来了可观的收入增长。

**表 1：OneRec 的离线性能。绿色表示 OneRec，棕色表示逐点方法，蓝色表示列表方法，黄色表示偏好对齐方法。最优结果加粗，次优结果加下划线。**

（详细数值见原表 1）

**表 2：OneRec 与当前多阶段系统在在线 A/B 测试中的绝对提升。**

| 模型 | 总观看时长 | 平均观看时长 |
|------|-----------|-------------|
| OneRec-0.1B | +0.57% | +4.26% |
| OneRec-1B | +1.21% | +5.01% |
| OneRec-1B+IPA | +1.68% | +6.56% |

**图 4：DPO 采样率 r_DPO 的消融研究。结果表明 1% 比例的 DPO 训练带来显著增益，但进一步增加采样率带来的改进有限。**

**图 5：OneRec 的可扩展性。结果表明 OneRec 在参数扩展时持续受益于性能提升。**

**图 6：语义 ID 各层 softmax 输出概率分布的可视化。红星代表具有最高奖励值的item的语义 ID。**

---

## 6 结论

在本文中，我们专注于介绍一个工业级解决方案，用于单阶段生成式推荐。我们的解决方案建立了三个关键贡献：首先，我们通过应用 MoE 架构以高计算效率有效扩展了模型参数，为大规模工业推荐提供了可扩展的蓝图。其次，我们发现了以会话级生成方式建模目标item上下文信息的必要性，证明了上下文序列建模本质上比孤立的逐点方式更好地捕获用户偏好动态。此外，我们提出了迭代偏好对齐（IPA）策略，以改善 OneRec 在多样化用户偏好模式上的泛化能力。大量的离线实验和在线 A/B 测试验证了 OneRec 的有效性和效率。此外，我们对在线结果的分析揭示，除了用户观看时间外，我们的模型在交互指标（如点赞）方面存在局限性。在未来的研究中，我们旨在增强端到端生成式推荐在多目标建模中的能力，以提供更好的用户体验。

---

## 参考文献

[1] Mohammad Gheshlaghi Azar, Zhaohan Daniel Guo, Bilal Piot, Remi Munos, Mark Rowland, Michal Valko, and Daniele Calandriello. 2024. A general theoretical paradigm to understand learning from human preferences. In *International Conference on Artificial Intelligence and Statistics*. PMLR, 4447–4455.

[2] Christopher JC Burges. 2010. From ranknet to lambdarank to lambdamart: An overview. *Learning* 11, 23-581 (2010), 81.

[3] Jianxin Chang, Chenbin Zhang, Zhiyi Fu, Xiaoxue Zang, Lin Guan, Jing Lu, Yiqun Hui, Dewei Leng, Yanan Niu, Yang Song, et al. 2023. TWIN: TWo-stage interest network for lifelong user behavior modeling in CTR prediction at kuaishou. In *Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining*. 3785–3794.

[4] Yuxin Chen, Junfei Tan, An Zhang, Zhengyi Yang, Leheng Sheng, Enzhi Zhang, Xiang Wang, and Tat-Seng Chua. 2024. On Softmax Direct Preference Optimization for Recommendation. In *The Thirty-eighth Annual Conference on Neural Information Processing Systems*.

[5] Sayak Ray Chowdhury, Anush Kini, and Nagarajan Natarajan. 2024. Provably Robust DPO: Aligning Language Models with Noisy Feedback. In *ICML 2024*.

[6] Paul Covington, Jay Adams, and Emre Sargin. 2016. Deep neural networks for youtube recommendations. In *Proceedings of the 10th ACM conference on recommender systems*. 191–198.

[7] Damai Dai, Chengqi Deng, Chenggang Zhao, RX Xu, Huazuo Gao, Deli Chen, Jiashi Li, Wangding Zeng, Xingkai Yu, Y Wu, et al. 2024. Deepseekmoe: Towards ultimate expert specialization in mixture-of-experts language models. *arXiv preprint arXiv:2401.06066* (2024).

[8] Nicola De Cao, Gautier Izacard, Sebastian Riedel, and Fabio Petroni. 2020. Autoregressive entity retrieval. *arXiv preprint arXiv:2010.00904* (2020).

[9] Nan Du, Yanping Huang, Andrew M Dai, Simon Tong, Dmitry Lepikhin, Yuanzhong Xu, Maxim Krikun, Yanqi Zhou, Adams Wei Yu, Orhan Firat, et al. 2022. Glam: Efficient scaling of language models with mixture-of-experts. In *International Conference on Machine Learning*. PMLR, 5547–5569.

[10] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. 2024. The llama 3 herd of models. *arXiv preprint arXiv:2407.21783* (2024).

[11] Hongliang Fei, Jingyuan Zhang, Xingxuan Zhou, Junhao Zhao, Xinyang Qi, and Ping Li. 2021. GemNN: gating-enhanced multi-task neural networks with feature interaction learning for CTR prediction. In *Proceedings of the 44th international ACM SIGIR conference on research and development in information retrieval*. 2166–2171.

[12] Chao Feng, Wuchao Li, Defu Lian, Zheng Liu, and Enhong Chen. 2022. Recommender forest for efficient retrieval. *Advances in Neural Information Processing Systems* 35 (2022), 38912–38924.

[13] Luke Gallagher, Ruey-Cheng Chen, Roi Blanco, and J Shane Culpepper. 2019. Joint optimization of cascade ranking models. In *Proceedings of the 28th ACM International Conference on Information and Knowledge Management*. 2225–2228.

[14] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. 2024. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. *arXiv preprint arXiv:2501.12948* (2024).

[15] Huigi Gao, et al. 2023. Learning an end-to-end structure for retrieval in large-scale recommendations. In *Proceedings of the 30th ACM International Conference on Information and Knowledge Management*.

[16] Guo, H. et al. 2017. DeepFM: A Factorization-Machine based Neural Network for CTR Prediction. In *IJCAI*.

[17] Xinran He, Junfeng Pan, Ou Jin, Tianbing Xu, Bo Liu, Tao Xu, Yanxin Shi, Antoine Atallah, Ralf Herbrich, Stuart Bowers, et al. 2014. Practical lessons from predicting clicks on ads at facebook. In *Proceedings of the eighth international workshop on data mining for online advertising*. 1–9.

[18] Said, et al. 2023. Recommendations as language processing (RLP): A unified pretrain, personalized prompt & predict paradigm (P5). In *Proceedings of the 17th ACM Conference on Recommender Systems*.

[19] Xiangnan He, et al. 2017. Neural collaborative filtering. In *Proceedings of the 26th international conference on world wide web*. 173–182.

[20] Wenjie Hu, et al. 2020. Listen to what you want: A new neural voice generation approach for content creation. In *Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*.

[21] Po-Sen Huang, et al. 2013. Learning deep structured semantic models for web search using clickthrough data. In *Proceedings of the 22nd ACM international conference on Conference on information & knowledge management*. 2333–2338.

[22] Wang-Cheng Kang and Julian McAuley. 2018. Self-attentive sequential recommendation. In *2018 IEEE International Conference on Data Mining (ICDM)*. IEEE, 197–206.

[23] Wang-Cheng Kang, et al. 2023. Towards the next generation of pre-ranking system. In *Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining*.

[24] Diederik P Kingma and Jimmy Ba. 2014. Adam: A method for stochastic optimization. *arXiv preprint arXiv:1412.6980* (2014).

[25] Rajiv Khanna, et al. 2023. Semantic ids for industrial recommendation. *arXiv preprint arXiv:2302.01614* (2023).

[26] Thomas N Kipf and Max Welling. 2016. Semi-supervised classification with graph convolutional networks. *arXiv preprint arXiv:1609.02907* (2016).

[27] Yehuda Koren, Robert Bell, and Chris Volinsky. 2009. Matrix factorization techniques for recommender systems. *Computer* 42, 8 (2009), 30–37.

[28] Walid Krichene and Steffen Rendle. 2020. On sampled softmax for recommendations. In *Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*.

[29] Jiacheng Li, et al. 2023. Text is all you need: Learning language representations for sequential recommendation. In *Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining*.

[30] Jie Lei, et al. 2023. Scaling law for recommendation models: Towards general-purpose user representations. In *Proceedings of the AAAI Conference on Artificial Intelligence*.

[31] Jianxun Lian, et al. 2018. xDeepFM: Combining explicit and implicit feature interactions for recommender systems. In *Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*.

[32] Shayegan Omidshafiei, et al. 2023. Reinforcement learning from human feedback for recommendation. In *Proceedings of the 17th ACM Conference on Recommender Systems*.

[33] Aaron van den Oord, Oriol Vinyals, and Koray Kavukcuoglu. 2017. Neural discrete representation learning. *arXiv preprint arXiv:1711.00937* (2017).

[34] Junwei Pan, et al. 2020. COLD: Towards the next generation of pre-ranking system. *arXiv preprint arXiv:2007.16122* (2020).

[35] Richard Yuanzhe Pang, et al. 2024. Direct preference optimization: Your language model is secretly a reward model. In *Advances in Neural Information Processing Systems*.

[36] Zhen Qin, et al. 2023. Generative recommendations with semantic ids. In *Proceedings of the 17th ACM Conference on Recommender Systems*.

[37] Steffen Rendle. 2010. Factorization machines. In *2010 IEEE International conference on data mining*. IEEE, 995–1000.

[38] Steffen Rendle, et al. 2020. Neural collaborative filtering vs. matrix factorization revisited. In *Fourteenth ACM Conference on Recommender Systems*.

[39] Nisan Stiennon, et al. 2020. Learning to summarize from human feedback. In *Advances in Neural Information Processing Systems*.

[40] Fei Sun, et al. 2019. BERT4Rec: Sequential recommendation with bidirectional encoder representations from transformer. In *Proceedings of the 28th ACM International Conference on Information and Knowledge Management*.

[41] Jiaxi Tang and Ke Wang. 2018. Personalized top-n sequential recommendation via convolutional sequence embedding. In *Proceedings of the Eleventh ACM International Conference on Web Search and Data Mining*. 565–573.

[42] Hugo Touvron, et al. 2023. Llama 2: Open foundation and fine-tuned chat models. *arXiv preprint arXiv:2307.09288* (2023).

[43] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. *Advances in neural information processing systems* 30 (2017).

[44] Petar Velickovic, et al. 2017. Graph attention networks. *arXiv preprint arXiv:1710.10903* (2017).

[45] Ruoxi Wang, et al. 2017. Deep & cross network for ad click predictions. In *Proceedings of the ADKDD'17*. 1–7.

[46] Ruoxi Wang, et al. 2021. DCN V2: Improved deep & cross network and practical lessons for web-scale learning to rank systems. In *Proceedings of the Web Conference 2021*.

[47] Haoran Xu, et al. 2024. A preliminary study of direct preference optimization for recommendation. In *Proceedings of the 18th ACM Conference on Recommender Systems*.

[48] Ji Yang, et al. 2020. Mixed negative sampling for learning two-tower neural networks in recommendations. In *Companion Proceedings of the Web Conference 2020*.

[49] Jiahui Yu, et al. 2022. Vector-quantized image modeling with improved VQGAN. In *International Conference on Learning Representations*.

[50] Jiaqi Zhai, et al. 2023. Revisiting neural retrieval on accelerators. In *Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining*.

[51] Jiaqi Zhai, et al. 2024. Actions speak louder than words: Trillion-parameter sequential transducers for generative recommendations. *arXiv preprint arXiv:2402.17152* (2024).

[52] Guorui Zhou, et al. 2018. Deep interest network for click-through rate prediction. In *Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining*. 1059–1068.

[53] Guorui Zhou, et al. 2019. Practice on long sequential user behavior modeling for click-through rate prediction. In *Proceedings of the 25th ACM SIGKDD international conference on knowledge discovery & data mining*. 1059–1068.

[54] Han Zhu, Xiang Li, Pengye Zhang, Guozheng Li, Jie He, Han Li, and Kun Gai. 2018. Learning tree-based deep model for recommender systems. In *Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining*. 1079–1088.

[55] Barret Zoph, Irwan Bello, Sameer Kumar, Nan Du, Yanping Huang, Jeff Dean, Noam Shazeer, and William Fedus. 2022. Designing effective sparse expert models. *arXiv preprint arXiv:2202.08906* (2022).
