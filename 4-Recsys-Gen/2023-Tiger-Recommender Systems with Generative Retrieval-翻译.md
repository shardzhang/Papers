# Recommender Systems with Generative Retrieval（中文翻译）


本文介绍了 Recommender Systems with Generative Retrieval。核心内容：


关键发现：

---


> Shashank Rajput*、Nikhil Mehta*、Anima Singh、Raghunandan Keshavan、Trung Vu、Lukasz Heldt、Lichan Hong、Yi Tay、Vinh Q. Tran、Jonah Samost、Maciej Kula、Ed H. Chi、Maheswaran Sathiamoorthy
> 威斯康星大学麦迪逊分校、Google DeepMind、Google
> *共同第一作者。工作由 SR 在 Google 期间完成。
> NeurIPS 2023

---


## 摘要

现代推荐系统通过将查询和item候选嵌入到同一统一空间中，然后进行近似最近邻搜索，以根据查询嵌入选择 top 候选，从而执行大规模检索。在本文中，我们提出了一种新颖的**生成式检索方法**，其中**检索模型自回归地解码目标候选的标识符**。为此，我们创建了语义上**有意义的码字元组**作为每个item的**语义 ID**。给定用户会话中item的语义 ID，我们训练一个基于 Transformer 的序列到序列Seq2Seq模型来预测用户将交互的下一个item的语义 ID。我们表明，使用所提出范式训练的推荐系统在各种数据集上显著优于当前的 SOTA 模型。此外，我们证明将语义 ID 纳入Seq2Seq模型增强了其泛化能力，这一点通过观察到的 **对没有先前交互历史的item的改进检索性能** 得到了验证。

---


## 1 引言

推荐系统帮助用户发现感兴趣的内容，并在各种推荐领域中无处不在，包括视频 [4, 43, 9]、APP [3]、产品 [6, 8] 和音乐 [18, 19]。现代推荐系统采用**检索-排序策略**，其中在检索阶段选择一组可行的候选，然后由排序模型进行排序。由于排序模型仅作用于它接收到的候选，因此期望检索阶段产生高度相关的候选。

构建检索模型有标准且成熟的方法。矩阵分解 [19] 学习在同一空间中的查询和候选嵌入。为了更好地捕获数据中的非线性，双编码器架构 [39]（即一个塔用于查询，另一个用于候选）采用 **内积** 将查询和候选嵌入到同一空间中，在近年变得流行。在推理时使用这些模型时，使用 候选塔 创建一个存储所有item嵌入的索引。对于给定的查询，使用 查询塔 获得其嵌入，并使用近似最近邻（ANN）算法进行检索。近年来，双编码器架构也已扩展到序列推荐 [11, 24, 41, 17, 32, 6, 44]，这些架构明确考虑了**用户-item交互的顺序**。

我们提出了一种构建生成式检索模型用于序列推荐的新范式。与传统查询-候选匹配方法不同，我们的方法使用端到端生成模型直接预测候选 ID。我们提出利用 Transformer [36] 的记忆（参数）作为推荐系统中检索的端到端索引，类似于 Tay 等人 [34] 使用 Transformer 记忆进行文档检索。我们将我们的方法称为 **TIGER**（Transformer Index for GEnerative Recommenders）。TIGER 的高级概述如图 1 所示。

**图 1：TIGER 框架概述。使用 TIGER，序列推荐通过将每个item表示为一个由离散语义token组成的元组，来表达为一个生成式检索任务。**

TIGER 以称为**语义 ID** 的item新颖语义表示为独特特征——语义 ID 是一个从每个item的内容信息中导出的token序列。具体来说，给定一个item的文本特征，我们使用预训练的文本编码器（例如 Sentence-T5 [27]）生成密集的内容嵌入。然后对item的嵌入应用量化方案，形成一组有序的token/码字，我们称之为该item的语义 ID。最终，这些语义 ID 被用于在序列推荐任务上训练 Transformer 模型。

将item表示为语义token的序列有许多优势。在有语义意义的数据上训练 Transformer 记忆允许相似item之间的知识共享。这使我们能够摒弃以前在推荐模型中用作item特征的原子式和随机item ID [33, 42, 11, 8]。使用item的语义token表示，模型不太容易受到推荐系统中固有的反馈循环 [1, 26, 39] 的影响，使模型能够泛化到语料库中新添加的item。此外，使用token序列进行item表示有助于缓解与item语料库规模相关的挑战；可以使用token表示的item数量是序列中每个token基数的乘积。通常，item语料库规模可能达到数十亿量级，为每个item学习唯一的嵌入会占用大量内存。虽然可以采用基于随机哈希的技术 [16] 来减少item表示空间，但在这项工作中，我们表明使用语义有意义的token进行item表示是一个有吸引力的替代方案。本文的主要贡献总结如下：

1. 我们提出 **TIGER**，一个新颖的基于生成式检索的推荐框架，为每个item分配语义 ID，并训练一个检索模型来预测给定用户可能与之互动的item的语义 ID。
2. 我们表明 TIGER 在多个数据集上根据 Recall 和 NDCG 指标优于现有的 SOTA 推荐系统。
3. 我们发现这种生成式检索的新范式为序列推荐系统带来了两种额外能力：1）推荐新item和低频item的能力，从而改善冷启动推荐；2）使用可调参数生成多样化推荐的能力。

**论文概览**。在第 2 节，我们简要回顾了推荐系统、生成式检索和本文使用的语义 ID 生成技术的相关工作。在第 3 节，我们解释了我们提出的框架，并概述了我们用于语义 ID 生成的各种技术。我们在第 4 节展示实验结果，并在第 5 节总结论文。

---

## 2 相关工作

**序列推荐器**。在推荐系统中使用深度序列模型已经发展出了丰富的文献。GRU4REC [11] 是第一个将基于 GRU 的 RNN 用于序列推荐的。Li 等人 [24] 提出了神经注意力会话基础推荐（NARM），其中注意力机制与 GRU 层一起用于跟踪用户的长期意图。Zhang 等人提出的 AttRec [41] 使用自注意力机制来建模用户在当前会话中的意图，并通过度量学习建模用户-item亲和度来实现个性化。同时，Kang 等人也提出了 SASRec [17]，它使用了类似于仅解码器 Transformer 模型的自注意力。受语言任务中掩码语言建模成功的启发，BERT4Rec [32] 和 Transformers4Rec [6] 利用具有掩码策略的 Transformer 模型进行序列推荐任务。S3-Rec [44] 通过预训练四个自监督任务来改进数据表示。上述描述的模型为每个item学习一个高维嵌入，并在最大内积搜索（MIPS）空间中执行 ANN 来预测下一个item。相比之下，我们提出的技术 TIGER 使用生成式检索直接预测下一个item的语义 ID。

P5 [8] 微调预训练的大型语言模型用于多任务推荐系统。P5 模型依赖于 LLM 分词器（SentencePiece 分词器 [29]）从随机分配的item ID 生成token。而我们使用基于item内容信息学习的item语义 ID 表示。在我们的实验（表 2）中，我们证明基于item语义 ID 表示的推荐系统比使用随机编码的方法产生更好的结果。

**语义 ID**。Hou 等人提出了 VQ-Rec [12] 来生成使用内容信息的"编码"（类似于语义 ID）用于item表示。然而，他们的重点在于构建可迁移的推荐系统，并且没有以生成式方式使用编码进行检索。虽然他们也使用乘积量化 [15] 生成编码，但我们使用 RQ-VAE 生成语义 ID，这导致了item的层次化表示（第 4.2 节）。在与我们同期的工作中，Singh 等人 [31] 表明层次化语义 ID 可用于替代排序模型中的item ID，在大规模推荐系统中改善模型泛化。

**生成式检索**。虽然过去已经提出了学习搜索索引的技术 [20]，但生成式检索是一种最近发展起来的方法，用于文档检索，其中任务是从数据库中返回一组相关文档。一些例子包括 GENRE [5]、DSI [34]、NCI [37] 和 CGR [22]。相关工作更详细的覆盖见附录 A。据我们所知，我们是第一个提出使用item语义 ID 表示进行推荐系统的生成式检索的。

---

## 3 提出的框架

我们提出的框架包含两个阶段：

1. **使用内容特征生成语义 ID**。这包括将item内容特征编码为嵌入向量，并将嵌入量化为语义码字的元组。生成的码字元组称为item的语义 ID。
2. **在语义 ID 上训练生成式推荐系统**。在序列推荐任务上，使用语义 ID 序列训练一个 Transformer 模型。

### 3.1 语义 ID 生成

在本节中，我们描述推荐语料库中item的语义 ID 生成过程。我们假设每个item具有关联的内容特征，这些特征捕获有用的语义信息（例如标题、描述或图像）。此外，我们假设我们可以访问一个预训练的内容编码器来生成语义嵌入 x
$$
\in
$$
 ℝᵈ。例如，通用预训练文本编码器如 Sentence-T5 [27] 和 BERT [7] 可用于将item的文本特征转换为语义嵌入。然后量化这些语义嵌入，为每个item生成语义 ID。图 2a 给出了该过程的高级概述。

**图 2：TIGER 中使用的建模方法概述。(a) 使用内容嵌入的量化生成item语义 ID。(b) 基于 Transformer 的编码器-解码器设置，用于构建用于生成式检索的序列到序列模型。**

我们将语义 ID 定义为一个长度为 m 的码字元组。元组中的每个码字来自不同的码本。因此语义 ID 可以唯一表示的item数量等于码本大小的乘积。虽然生成语义 ID 的不同技术会导致 ID 具有不同的语义属性，但我们希望它们至少具有以下属性：**相似的item**（具有相似内容特征或其语义嵌入接近的item）**应该具有重叠的语义 ID**。例如，语义 ID 为 (10, 21, 35) 的item应该比 ID 为 (10, 23, 32) 的item更接近于语义 ID 为 (10, 21, 40) 的item。接下来，我们讨论用于语义 ID 生成的量化方案。

**用于语义 ID 的 RQ-VAE**。残差量化变分自编码器（RQ-VAE）[40] 是一个多级向量量化器，它通过对残差应用量化来生成码字元组（也称为语义 ID）。自编码器通过更新量化码本和 DNN 编码器-解码器参数进行联合训练。图 3 说明了通过残差量化生成语义 ID 的过程。

**图 3：RQ-VAE：在图中，由 DNN 编码器输出的向量 $r_0$ （由蓝色条表示）被输入量化器，它迭代地工作。首先，在第一级码本中找到最接近 $r_0$ 的向量。设这个最接近的向量为 e_ $c_0$ （由红色条表示）。然后计算残差误差 $r_1$ := $r_0$ − e_ $c_0$ 。这被输入量化器的第二级，并重复该过程：在第二级中找到最接近 $r_1$ 的向量，例如 e_ $c_1$ （由绿色条表示），然后计算第二级残差误差 $r_2$ := $r_1$ − e_ $c_1$ 。然后在 $r_2$ 上第三次重复该过程。语义码被计算为 e_ $c_0$ 、e_ $c_1$ 和 e_ $c_2$ 在其各自码本中的索引。在图中所示的示例中，这导致编码 (7, 1, 4)。**

RQ-VAE 首先通过编码器 E 对输入 x 进行编码，以学习潜在表示 z := E(x)。在第零级（d=0），初始残差简单地定义为 $r_0$ := z。在每一级 d，我们有一个码本 C_d := {e_k}_k=1ᴷ，其中 K 是码本大小。然后通过将 $r_0$ 映射到该级码本中最近的嵌入来量化 $r_0$ 。在 d=0 时最近嵌入 e_ $c_0$ 的索引 $c_0$ = argmin_i || $r_0$ − e_i|| 表示第零级码字。对于下一级 d=1，残差定义为 $r_1$ := $r_0$ − e_ $c_0$ 。然后，类似于第零级，通过在第一级的码本中找到最接近 $r_1$ 的嵌入来计算第一级的编码。这个递归过程重复 m 次，得到表示语义 ID 的 m 个码字的元组。这种递归方法从粗到细的粒度逼近输入。注意，我们选择为 m 级中的每一级使用一个大小为 K 的独立码本，而不是使用一个单一的 mK 大小的码本。这样做是因为残差的范数趋向于随着级别的增加而减小，从而允许不同级别具有不同的粒度。

一旦我们有了语义 ID ( $c_0$ , ..., $c_{m-1}$ )，z 的量化表示计算为 ẑ := $\Sigma$ _{d=0}^{m-1} e_c_d。然后 ẑ 被传递给解码器，解码器尝试使用 ẑ 重建输入 x。RQ-VAE 损失定义为 L(x) := L_recon + L_rqvae，其中 L_recon := ||x − x̂||²，L_rqvae := $\Sigma$ _{d=0}^{m-1} ||sg[r_d] − e_c_d||² + $\beta$ ||r_d − sg[e_c_d]||²。这里 x̂ 是解码器的输出，sg 是停止梯度操作 [35]。这个损失联合训练编码器、解码器和码本。

如 [40] 中所提出的，为防止 RQ-VAE 出现码本崩溃（即大多数输入仅映射到少数码本向量），我们使用基于 k-means 聚类的码本初始化。具体来说，我们在第一个训练批次上应用 k-means 算法，并使用质心作为初始化。

**其他量化替代方案**。生成语义 ID 的一个简单替代方案是使用局部敏感哈希（LSH）。我们在第 4.2 节进行了消融研究，发现 RQ-VAE 确实比 LSH 效果更好。另一个选择是层次化地使用 k-means 聚类 [34]，但它会丢失不同聚类之间的语义含义 [37]。我们还尝试了 VQ-VAE，虽然它在生成候选进行检索时表现与 RQ-VAE 相似，但它失去了 ID 的层次化特性，而层次化特性赋予了第 4.3 节中讨论的许多新能力。

**处理碰撞**。根据语义嵌入的分布、码本大小的选择和码字的长度，可能会发生语义碰撞（即多个item可能映射到相同的语义 ID）。为去除碰撞，我们在有序语义编码的末尾附加一个额外的token，使它们唯一。例如，如果两个item共享语义 ID (12, 24, 52)，我们附加额外的token来区分它们，将两个item表示为 (12, 24, 52, 0) 和 (12, 24, 52, 1)。为检测碰撞，我们维护一个将语义 ID 映射到对应item的查找表。注意，碰撞检测和修复仅在 RQ-VAE 模型训练完成后执行一次。此外，由于语义 ID 是整数元组，查找表在存储方面相比于高维嵌入是高效的。

### 3.2 使用语义 ID 的生成式检索

我们按时间顺序对每个用户交互过的item进行排序，构建item序列。然后，给定形式为 ( $item_1$ , ..., item_n) 的序列，推荐系统的任务是预测下一个item ite $m_{n+1}$ 。我们提出一种生成式方法，直接预测下一个item的语义 ID。形式化地，令 ( $c_{i,0}$ , ..., $c_{i,m-1}$ ) 为item i 的 m 长度语义 ID。然后，我们将item序列转换为序列 ( $c_{1,0}$ , ..., $c_{1,m-1}$ , $c_{2,0}$ , ..., $c_{2,m-1}$ , ..., $c_{n,0}$ , ..., $c_{n,m-1}$ )。然后训练序列到序列模型来预测 ite $m_{n+1}$ 的语义 ID，即 ( $c_{n+1,0}$ , ..., $c_{n+1,m-1}$ )。鉴于我们框架的生成式特性，解码器生成的一个语义 ID 有可能不匹配推荐语料库中的任何item。然而，正如我们在附录中展示的（图 6），这种事件发生的概率很低。我们进一步在附录 E 中讨论了如何处理这类事件。

---

## 4 实验

**数据集**。我们在来自 Amazon Product Reviews 数据集 [10] 的三个公开真实世界基准上进行评估，该数据集包含从 1996 年 5 月到 2014 年 7 月的用户评论和item元数据。具体来说，我们使用 Amazon Product Reviews 数据集的三个类别进行序列推荐任务："Beauty"、"Sports and Outdoors"和"Toys and Games"。我们在附录 C 中讨论数据集统计信息和预处理。

**评估指标**。我们使用 top-k Recall（Recall@K）和归一化折现累积增益（NDCG@K），其中 K = 5, 10 来评估推荐性能。

**RQ-VAE 实现细节**。如第 3.1 节所述，RQ-VAE 用于量化item的语义嵌入。我们使用预训练的 Sentence-T5 [27] 模型来获得数据集中每个item的语义嵌入。具体来说，我们使用item的内容特征如标题、价格、品牌和类别来构建一个句子，然后将其传递给预训练的 Sentence-T5 模型以获得item的 768 维语义嵌入。

RQ-VAE 模型由三个组件组成：一个将输入语义嵌入编码为潜在表示的 DNN 编码器、一个输出量化表示的残差量化器，以及一个将量化表示解码回语义输入嵌入空间的 DNN 解码器。编码器有三个中间层，大小分别为 512、256 和 128，使用 ReLU 激活，最终潜在表示维度为 32。为量化这个表示，进行三个级别的残差量化。每个级别维护一个基数为 256 的码本，其中码本中的每个向量维度为 32。在计算总损失时，我们使用 $\beta$ = 0.25。RQ-VAE 模型训练 20k 个 epoch，以确保高码本使用率（
$$
\geq
$$
 80%）。我们使用 Adagrad 优化器，学习率为 0.4，批大小为 1024。训练完成后，我们使用学习到的编码器和量化组件为每个item生成一个 3 元组的语义 ID。为避免多个item映射到相同的语义 ID，我们为共享相同前三个码字的item添加一个唯一的第 4 个码，即关联到元组 (7,1,4) 的两个item分别被分配 (7,1,4,0) 和 (7,1,4,1)（如果没有碰撞，我们仍然分配 0 作为第四个码字）。这为推荐语料库中的每个item产生了一个长度为 4 的唯一语义 ID。

**序列到序列模型实现细节**。我们使用开源 T5X 框架 [28] 来实现我们基于 Transformer 的编码器-解码器架构。为使模型能够处理序列推荐任务的输入，序列到序列模型的词汇表包含每个语义码字的token。具体来说，词汇表包含 1024（256
$$
\times
$$
4）个token来表示语料库中的item。除了item的语义码字外，我们还向词汇表中添加用户特定的token。为保持词汇表大小有限，我们只添加 2000 个用户 ID token。我们使用哈希技巧 [38] 将原始用户 ID 映射到 2000 个用户 ID token之一。我们将输入序列构建为用户 ID token后跟给定用户的item交互历史对应的语义 ID token序列。我们发现将用户 ID 添加到输入中，使模型能够个性化检索到的item。

我们为基于 Transformer 的编码器和解码器模型各使用 4 层，每层有 6 个维度为 64 的自注意力头。我们对所有层使用 ReLU 激活函数。MLP 和输入维度分别设为 1024 和 128。我们使用 0.1 的 dropout。总体而言，该模型约有 1300 万个参数。我们对"Beauty"和"Sports and Outdoors"数据集训练 200k 步。由于"Toys and Games"数据集较小，它只训练 100k 步。我们使用 256 的批大小。学习率在前 10k 步为 0.01，然后遵循逆平方根衰减调度。

### 4.1 序列推荐性能

在本节中，我们将我们提出的生成式检索框架与以下序列推荐方法进行比较（这些方法在附录 B 中简要描述）：GRU4Rec [11]、Caser [33]、HGN [25]、SASRec [17]、BERT4Rec [32]、FDSA [42]、S3-Rec [44] 和 P5 [8]。值得注意的是，所有基线（除 P5 外）都使用双编码器学习高维向量空间，其中用户过去的item交互和候选item被编码为高维表示，并使用最大内积搜索（MIPS）来检索用户可能交互的下一个候选item。相比之下，我们新颖的生成式检索框架使用序列到序列模型直接逐token预测item的语义 ID。

**推荐性能**。我们对提出的 TIGER 在序列推荐任务上进行了广泛分析，并与上述基线进行比较。除 P5 外，所有基线的结果均来自 Zhou 等人 [44] 公开可访问的结果。对于 P5，我们使用了作者提供的源代码。然而，为了公平比较，我们更新了数据预处理方法，使其与其他基线和我们的方法一致。我们在附录 D 中提供了与我们所做更改相关的更多细节。

结果如表 1 所示。我们观察到 TIGER 一致地优于现有基线。我们在考虑的所有三个基准上都看到了显著的改进。特别地，TIGER 在 Beauty 基准上表现相当出色，相比次优基线，NDCG@5 最高提升 29%（相比 SASRec），Recall@5 提升 17.3%（相比 S3-Rec）。类似地，在 Toys and Games 数据集上，TIGER 在 NDCG@5 和 NDCG@10 上分别提升了 21% 和 15%。

**表 1：序列推荐性能比较。最后一行表示 TIGER 相对最佳基线的 % 提升。粗体（下划线）用于表示最佳（次优）指标。**（详细结果表见原文表 1）

### 4.2 item表示

在本节中，我们分析 RQ-VAE 语义 ID 的几个重要特性。具体来说，我们首先进行定性分析以观察语义 ID 的层次化特性。接下来，我们通过对比基于哈希的量化方法的性能，评估我们使用 RQ-VAE 进行量化的设计选择的重要性。最后，我们进行消融研究，通过比较使用随机 ID 的序列到序列模型与 TIGER，来研究使用语义 ID 的重要性。

**定性分析**。我们在图 4 中分析了为 Amazon Beauty 数据集学习的 RQ-VAE 语义 ID。为便于展示，我们将 RQ-VAE 级数设为 3，码本大小分别为 4、16 和 256，即对于item的给定语义 ID ( $c_1$ , $c_2$ , $c_3$ )，0
$$
$$\leq
$$
$$ $c_1$
$$
\leq
$$
 3，0
$$
\leq
$$
 $c_2$
$$
\leq
$$
 15，0
$$
\leq
$$
 $c_3$
$$
\leq
$$
 255。在图 4a 中，我们使用 $c_1$ 标注每个item的类别，以在数据集的整体类别分布中可视化 $c_1$ 特定的类别。如图 4a 所示， $c_1$ 捕获了item的高级类别。例如， $c_1$ = 3 包含了大部分与"Hair"相关的产品。类似地， $c_1$ = 1 的大多数item是面部、唇部和眼部的"Makeup"和"Skin"产品。我们还通过固定 $c_1$ 并可视化在 Fig. 4b 中所有可能的 $c_2$ 值的类别分布，来可视化 RQ-VAE 语义 ID 的层次化特性。我们再次发现，第二个码字 $c_2$ 进一步将 $c_1$ 捕获的高级语义细分为细粒度的类别。RQ-VAE 学习到的语义 ID 的层次化特性开辟了一系列广泛的新能力，这些将在第 4.3 节中讨论。与基于随机原子 ID 学习item嵌入的现有推荐系统不同，TIGER 使用语义 ID，其中语义相似的item具有重叠的码字，这使模型能够有效共享数据集中语义相似item的知识。

**图 4：Amazon Beauty 数据集上 RQ-VAE 语义 ID ( $c_1$ , $c_2$ , $c_3$ , $c_4$ ) 的定性研究。我们展示了真实类别分布在不同语义token上的分布。此外，RQ-VAE 语义 ID 形成了item的层次结构，其中第一个语义token ( $c_1$ ) 对应于粗粒度类别，而第二/第三语义token ( $c_2$ / $c_3$ ) 对应于细粒度类别。**

**哈希 vs. RQ-VAE 语义 ID**。我们通过将 RQ-VAE 与局部敏感哈希（LSH）[14, 13, 2] 进行比较，研究 RQ-VAE 在我们框架中的重要性。LSH 是一种流行的哈希技术，可以轻松适应我们的设置。为生成 LSH 语义 ID，我们使用 h 个随机超平面 $w_1$ , ..., w_h 对嵌入向量 x 进行随机投影，并计算以下二进制向量：(1_{ $w_1$ ᵀx>0}, ..., 1_{w_hᵀx>0})。该向量被转换为整数码 $c_0$ = $\Sigma$ _{i=1}^{h} 2^{i-1} 1_{w_iᵀx>0}。使用一组独立的随机超平面重复此过程 m 次，得到 m 个码字 ( $c_0$ , $c_1$ , ..., $c_{m-1}$ )，我们称之为 LSH 语义 ID。

在表 2 中，我们比较了 LSH 语义 ID 与我们提出的 RQ-VAE 语义 ID 的性能。在此实验中，对于 LSH 语义 ID，我们使用了 h=8 个随机超平面并设置 m=4，以确保与 RQ-VAE 具有可比的基数。超平面的参数从标准正态分布中随机采样，这确保了超平面是球对称的。我们的结果表明 RQ-VAE 一致地优于 LSH。这说明了在给定相同基于内容的语义嵌入的情况下，通过非线性深度神经网络（DNN）架构学习语义 ID 比使用随机投影产生更好的量化效果。

**随机 ID vs. 语义 ID**。我们还比较了语义 ID 在我们的生成式检索推荐系统中的重要性。特别地，我们比较了随机生成的 ID 与语义 ID。为生成 Random ID 基线，我们为每个item分配 m 个随机码字。item的长度为 m 的 Random ID 只是 ( $c_1$ , ..., c_m)，其中 c_i 从 {1,2,...,K} 中均匀随机采样。我们为 Random ID 基线设置 m = 4 和 K = 255，以使基数与 RQ-VAE 语义 ID 相似。Random ID 与 RQ-VAE 和 LSH 语义 ID 的比较如表 2 所示。我们看到语义 ID 一致地优于 Random ID 基线，突出了利用基于内容的语义信息的重要性。

**表 2：用于生成式检索的不同 ID 生成技术的消融研究。我们表明 RQ-VAE 语义 ID（SID）的性能显著优于哈希 SID 和随机 ID。**（详细结果表见原文表 2）

### 4.3 新能力

我们描述了直接来源于我们提出的生成式检索框架的两种新能力，即冷启动推荐和推荐多样性。我们将这些能力称为"新"，因为现有的序列推荐模型（参见第 4.1 节的基线）不能直接用于满足这些真实世界用例。这些能力源于基于 RQ-VAE 的语义 ID 与我们框架的生成式检索方法之间的协同作用。

**冷启动推荐**。在本节中，我们研究提出的框架的冷启动推荐能力。由于真实世界推荐语料库的快速变化特性，新item不断被引入。由于新添加的item在训练语料库中缺乏用户展示，使用随机原子 ID 进行item表示的现有推荐模型无法检索到新item作为潜在候选。相比之下，TIGER 框架可以轻松执行冷启动推荐，因为它在预测下一个item时利用了item语义。

为进行此分析，我们使用 Amazon Reviews 中的 Beauty 数据集。为模拟新添加的item，我们从训练数据划分中移除 5% 的测试item。我们将这些移除的item称为"未见item"。将item从训练划分中移除确保了关于未见item没有数据泄露。和之前一样，我们使用长度为 4 的语义 ID 来表示item，其中前 3 个token使用 RQ-VAE 生成，第 4 个token用于确保所有已见item存在唯一 ID。我们在训练划分上训练 RQ-VAE 量化器和序列到序列模型。训练完成后，我们使用 RQ-VAE 模型为数据集中所有item（包括item语料库中的任何未见item）生成语义 ID。

给定模型预测的语义 ID ( $c_1$ , $c_2$ , $c_3$ , $c_4$ )，我们检索具有相同对应 ID 的已见item。注意，根据定义，模型预测的每个语义 ID 最多可以匹配训练数据集中的一个item。此外，具有相同前三个语义token ( $c_1$ , $c_2$ , $c_3$ ) 的未见item也被包括在检索候选列表中。最后，在检索 top-K 候选集时，我们引入一个超参数 $\epsilon$ ，它指定了我们的框架选择的未见item的最大比例。

我们在图 5 中将 TIGER 的性能与 k 近邻（KNN）方法在冷启动推荐设置中进行比较。对于 KNN，我们使用语义表示空间来执行最近邻搜索。我们将基于 KNN 的基线称为 Semantic_KNN。图 5a 显示，我们设置 $\epsilon$ =0.1 的框架在所有 Recall@K 指标上一致优于 Semantic_KNN。在图 5b 中，我们提供了对于各种 $\epsilon$ 值，我们的方法与 Semantic_KNN 的比较。在所有 $\epsilon$
$$
\geq
$$
 0.1 的设置中，我们的方法优于基线。

**图 5：冷启动检索设置中的性能。(a) Recall@K vs. K ( $\epsilon$ =0.1)。(b) Recall@10 vs. $\epsilon$ 。**

**推荐多样性**。虽然 Recall 和 NDCG 是用于评估推荐系统的主要指标，但预测的多样性是另一个关键目标。多样性差的推荐系统可能对用户的长期参与度有害。在这里，我们讨论如何使用我们的生成式检索框架来预测多样化的item。我们表明，在解码过程中使用基于温度的采样可以有效地控制模型预测的多样性。虽然基于温度的采样可以应用于任何现有的推荐模型，但 TIGER 凭借 RQ-VAE 语义 ID 的特性，允许在层次结构的各个级别进行采样。例如，采样语义 ID 的第一个token允许从粗粒度类别中检索item，而采样第二/第三个token允许在类别内采样item。

**表 3：Beauty 数据集中模型预测的类别分布的熵。更高的熵对应模型预测的更多样化的item。**（表格见原文表 3）

**表 4：基于温度解码的推荐多样性。**（表格见原文表 4）

我们使用 Entropy@K 指标定量测量预测的多样性，其中熵是根据模型预测的 top-K item的真实类别分布计算的。我们在表 3 中报告了各种温度值的 Entropy@K。我们观察到，解码阶段的温度采样可以有效地用于增加item真实类别的多样性。我们还在表 4 中进行了定性分析。

### 4.4 消融研究

我们在表 5 中测量了序列到序列模型中变化层数的效果。我们看到随着网络变大，指标略有改善。我们还测量了提供用户信息的效果，其结果在附录的表 8 中提供。

**表 5：不同层数的 Recall 和 NDCG 指标。**

| 层数 | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 |
|------|----------|--------|-----------|---------|
| 3    | 0.04499  | 0.03062 | 0.06699   | 0.03768 |
| 4    | 0.0454   | 0.0321  | 0.0648     | 0.0384  |
| 5    | 0.04633  | 0.03206 | 0.06596    | 0.03834 |

### 4.5 无效 ID

由于模型自回归地解码目标语义 ID 的码字，模型可能预测无效的 ID（即不映射到推荐数据集中任何item的 ID）。在我们的实验中，我们使用了长度为 4 的语义 ID，每个码字的基数为 256（即每个级别码本大小 = 256）。这种组合所覆盖的可能 ID 数量为 256⁴，约为 4 万亿。另一方面，我们考虑的数据集中的item数量为 10K-20K（见表 6）。尽管有效 ID 仅占全部完整 ID 空间的一小部分，但我们观察到模型几乎总是预测有效的 ID。我们在图 6 中将 TIGER 产生的无效 ID 的比例可视化为检索item数 K 的函数。对于 top-10 预测，无效 ID 的比例在三个数据集中约为 0.1%-1.6%。为抵消无效 ID 的影响并始终获得 top-10 有效 ID，我们可以增加束大小并过滤掉无效 ID。

重要的是要注意，尽管产生了无效 ID，TIGER 在与用于序列推荐的其他流行方法相比时仍达到了最先进的性能。处理无效token的一个扩展方案是在模型生成了无效token时进行前缀匹配。语义 ID 的前缀匹配将允许检索与模型生成的token具有相似语义含义的item。鉴于我们 RQ-VAE token的层次化特性，前缀匹配可以被视为模型预测item类别而不是item索引。注意，这样的扩展可以进一步改善 Recall/NDCG 指标。我们将这样的扩展留作未来工作。

**图 6：TIGER 产生的无效 ID 的比例（关于检索item数 K）。**

---

## 5 结论

本文提出了一种称为 TIGER 的新范式，用于使用生成模型检索推荐系统中的候选。支持这种方法的是一种新颖的item语义 ID 表示，它在内容嵌入上使用层次化量化器（RQ-VAE）来生成构成语义 ID 的token。我们的框架产生了一个模型，可以在无需创建索引的情况下进行训练和服务——Transformer 记忆充当了item的语义索引。我们注意到，我们的嵌入表的基数不随item空间的基数线性增长，相比于需要在训练期间创建大型嵌入表或为每个item生成索引的系统，这对我们有利。通过在三个数据集上的实验，我们展示了我们的模型可以实现 SOTA 检索性能，同时泛化到新item和未见item。

---

## 附录

### A 相关工作（续）

**生成式检索**。文档检索传统上涉及训练一个双塔模型，该模型将查询和文档映射到相同的高维向量空间，然后对查询在所有文档上执行 ANN 或 MIPS 以返回最接近的文档。这种技术存在一些缺点，如大型嵌入表 [22, 23]。生成式检索是一种最近提出的技术，旨在通过逐token生成文档的标题、名称或文档 ID 字符串来修复传统方法的一些问题。Cao 等人 [5] 提出了用于实体检索的 GENRE，它使用基于 Transformer 的架构逐token返回给定查询中引用的实体的名称。Tay 等人 [34] 提出了用于文档检索的 DSI，这是第一个为每个文档分配结构化语义 DocID 的系统。然后给定一个查询，模型自回归地逐token返回文档的 DocID。DSI 的工作标志着 IR 向生成式检索方法的范式转变，并且是端到端 Transformer 在检索应用中的第一个成功应用。随后，Lee 等人 [23] 表明生成式文档检索在多跳设置中也很有用，其中复杂查询不能由单个文档直接回答，因此他们的模型以思维链方式生成中间查询，最终为复杂查询生成输出。Wang 等人 [37] 通过提出一种专门考虑语义 DocID 中前缀的新解码器架构，补充了 Tay 等人 [34] 的基于层次化 k-means 聚类的语义 DocID。在 CGR [22] 中，作者提出了一种利用双编码器技术和生成式检索技术的方法，允许其编码器-解码器模型的解码器学习单独的上下文嵌入，这些嵌入内在地存储关于文档的信息。据我们所知，我们是第一个使用通过自编码器（RQ-VAE [40, 21]）创建的生成式语义 ID 用于检索模型的。

**向量量化**。我们将向量量化称为将高维向量转换为低维码字元组的过程。最直接的技术之一使用层次化聚类，例如 [34] 中使用的方法，其中在特定迭代中创建的聚类在下一个迭代中进一步划分为子聚类。另一种流行的替代方案是向量量化变分自编码器（VQ-VAE），它在 [35] 中被引入作为一种将自然图像编码为编码序列的方法。该技术的工作原理是首先将输入向量（或图像）通过一个降低维度的编码器。降低维度的向量被分区，每个分区被单独量化，从而产生一个编码序列：每个分区一个编码。然后解码器使用这些编码来重建原始向量（或图像）。

RQ-VAE [40, 21] 对 VQ-VAE 编码器的输出应用残差量化，以实现更低的重建误差。我们在第 3.1 节中更详细地讨论了这种技术。局部敏感哈希（LSH）[14, 13] 是一种用于聚类和近似最近邻搜索的流行技术。我们在本文中用于聚类的特定版本是 SimHash [2]，它使用随机超平面创建二进制向量作为item的哈希。由于它具有较低的计算复杂度和可扩展性 [13]，我们将其用作向量量化的基线技术。

### B 基线

以下是对 TIGER 比较的每个基线的简要描述：
- **GRU4Rec [11]**：首个基于 RNN 的方法，为序列推荐任务使用定制的 GRU。
- **Caser [33]**：使用 CNN 架构，通过应用水平和垂直卷积操作捕获高阶马尔可夫链进行序列推荐。
- **HGN [25]**：层次化门控网络，通过一种新的门控架构捕获用户的长期和短期兴趣。
- **SASRec [17]**：自注意力序列推荐，使用因果掩码 Transformer 建模用户的序列交互。
- **BERT4Rec [32]**：BERT4Rec 通过使用双向自注意力 Transformer 解决单向架构的局限性。
- **FDSA [42]**：特征级深度自注意力网络，将item特征作为输入序列的一部分纳入 Transformer。
- **S3-Rec [44]**：用于序列推荐的自监督学习，提出在自监督任务上预训练双向 Transformer 来改进序列推荐。
- **P5 [8]**：P5 是一种近期的方法，使用预训练的大型语言模型（LLM）在单个模型中统一不同的推荐任务。

### C 数据集统计

**表 6：三个真实世界基准的数据集统计。**

| 数据集 | #用户 | #item | 序列长度均值 | 序列长度中位数 |
|--------|-------|-------|-------------|--------------|
| Beauty | 22,363 | 12,101 | 8.87 | 6 |
| Sports and Outdoors | 35,598 | 18,357 | 8.32 | 6 |
| Toys and Games | 19,412 | 11,924 | 8.63 | 6 |

我们使用 Amazon Product Reviews 数据集 [10] 的三个公开基准，包含从 1996 年 5 月到 2014 年 7 月的用户评论和item元数据。我们使用 Amazon Product Reviews 数据集的三个类别进行序列推荐任务："Beauty"、"Sports and Outdoors"和"Toys and Games"。表 6 总结了数据集的统计信息。我们使用用户的评论历史按时间戳排序创建item序列，并过滤掉评论少于 5 条的用户。遵循标准评估协议 [17, 8]，我们使用留一法进行评估。对于每个item序列，最后一个item用于测试，倒数第二个item用于验证，其余用于训练。在训练期间，我们将用户历史中的item数量限制为 20。

### D P5 数据预处理的修改

**表 7：使用标准预处理的 P5 [8] 结果。**（详细结果见原文表 7）

P5 源代码预处理 Amazon 数据集时，首先为每个用户创建会话，其中包含按时间顺序排列的用户评论过的item列表。在创建这些会话后，...（详情见原文）

---

## 参考文献

[1] Himan Abdollahpouri, Masoud Mansoury, Robin Burke, and Bamshad Mobasher. The unfairness of popularity bias in recommendation. arXiv preprint arXiv:1907.13286, 2019.

[2] Moses S Charikar. Similarity estimation techniques from rounding algorithms. In Proceedings of the thirty-fourth annual ACM symposium on Theory of computing, pages 380–388, 2002.

[3] Heng-Tze Cheng, Levent Koc, Jeremiah Harmsen, Tal Shaked, Tushar Chandra, Hrishi Aradhye, Glen Anderson, Greg Corrado, Wei Chai, Mustafa Ispir, et al. Wide & deep learning for recommender systems. In Proceedings of the 1st workshop on deep learning for recommender systems, pages 7–10, 2016.

[4] Paul Covington, Jay Adams, and Emre Sargin. Deep neural networks for youtube recommendations. In Proceedings of the 10th ACM conference on recommender systems, pages 191–198, 2016.

[5] Nicola De Cao, Gautier Izacard, Sebastian Riedel, and Fabio Petroni. Autoregressive entity retrieval. arXiv preprint arXiv:2010.00904, 2020.

[6] Gabriel de Souza Pereira Moreira, Sara Rabhi, Jeong Min Lee, Ronay Ak, and Even Oldridge. Transformers4rec: Bridging the gap between nlp and sequential/session-based recommendation. In Fifteenth ACM Conference on Recommender Systems, pages 143–153, 2021.

[7] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018.

[8] Shijie Geng, Shuchang Liu, Zuohui Fu, Yingqiang Ge, and Yongfeng Zhang. Recommendation as language processing (rlp): A unified pretrain, personalized prompt & predict paradigm (p5). arXiv preprint arXiv:2203.13366, 2022.

[9] Carlos A Gomez-Uribe and Neil Hunt. The netflix recommender system: Algorithms, business value, and innovation. ACM Transactions on Management Information Systems (TMIS), 6(4):1–19, 2015.

[10] Ruining He and Julian McAuley. Ups and downs: Modeling the visual evolution of fashion trends with one-class collaborative filtering. In proceedings of the 25th international conference on world wide web, pages 507–517, 2016.

[11] Balázs Hidasi, Alexandros Karatzoglou, Linas Baltrunas, and Domonkos Tikk. Session-based recommendations with recurrent neural networks. arXiv preprint arXiv:1511.06939, 2015.

[12] Yupeng Hou, Zhankui He, Julian McAuley, and Wayne Xin Zhao. Learning vector-quantized item representation for transferable sequential recommenders. arXiv preprint arXiv:2210.12316, 2022.

[13] Piotr Indyk and Rajeev Motwani. Approximate nearest neighbors: towards removing the curse of dimensionality. In Proceedings of the thirtieth annual ACM symposium on Theory of computing, pages 604–613, 1998.

[14] Piotr Indyk, Rajeev Motwani, Prabhakar Raghavan, and Santosh Vempala. Locality-preserving hashing in multidimensional spaces. In Proceedings of the twenty-ninth annual ACM symposium on Theory of computing, pages 618–625, 1997.

[15] Herve Jegou, Matthijs Douze, and Cordelia Schmid. Product quantization for nearest neighbor search. IEEE transactions on pattern analysis and machine intelligence, 33(1):117–128, 2010.

[16] Wang-Cheng Kang, Derek Zhiyuan Cheng, Tiansheng Yao, Xinyang Yi, Ting Chen, Lichan Hong, and Ed H Chi. Learning to embed categorical features without embedding tables for recommendation. arXiv preprint arXiv:2010.10784, 2020.

[17] Wang-Cheng Kang and Julian McAuley. Self-attentive sequential recommendation. In 2018 IEEE international conference on data mining (ICDM), pages 197–206. IEEE, 2018.

[18] Dongmoon Kim, Kun-su Kim, Kyo-Hyun Park, Jee-Hyong Lee, and Keon Myung Lee. A music recommendation system with a dynamic k-means clustering algorithm. In Sixth international conference on machine learning and applications (ICMLA 2007), pages 399–403. IEEE, 2007.

[19] Yehuda Koren, Robert Bell, and Chris Volinsky. Matrix factorization techniques for recommender systems. Computer, 42(8):30–37, 2009.

[20] Tim Kraska, Alex Beutel, Ed H Chi, Jeffrey Dean, and Neoklis Polyzotis. The case for learned index structures. In Proceedings of the 2018 international conference on management of data, pages 489–504, 2018.

[21] Doyup Lee, Chiheon Kim, Saehoon Kim, Minsu Cho, and Wook-Shin Han. Autoregressive image generation using residual quantization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11523–11532, 2022.

[22] Hyunji Lee, Jaeyoung Kim, Hoyeon Chang, Hanseok Oh, Sohee Yang, Vlad Karpukhin, Yi Lu, and Minjoon Seo. Contextualized generative retrieval. arXiv preprint arXiv:2210.02068, 2022.

[23] Hyunji Lee, Sohee Yang, Hanseok Oh, and Minjoon Seo. Generative retrieval for long sequences. arXiv preprint arXiv:2204.13596, 2022.

[24] Jing Li, Pengjie Ren, Zhumin Chen, Zhaochun Ren, Tao Lian, and Jun Ma. Neural attentive session-based recommendation. In Proceedings of the 2017 ACM on Conference on Information and Knowledge Management, pages 1419–1428, 2017.

[25] Chen Ma, Peng Kang, and Xue Liu. Hierarchical gating networks for sequential recommendation. In Proceedings of the 25th ACM SIGKDD international conference on knowledge discovery & data mining, pages 825–833, 2019.

[26] Masoud Mansoury, Himan Abdollahpouri, Mykola Pechenizkiy, Bamshad Mobasher, and Robin Burke. Feedback loop and bias amplification in recommender systems, 2020.

[27] Jianmo Ni, Gustavo Hernandez Abrego, Noah Constant, Ji Ma, Keith Hall, Daniel Cer, and Yinfei Yang. Sentence-t5: Scalable sentence encoders from pre-trained text-to-text models. In Findings of the Association for Computational Linguistics: ACL 2022, pages 1864–1874, Dublin, Ireland, May 2022. Association for Computational Linguistics.

[28] Adam Roberts, Hyung Won Chung, Anselm Levskaya, Gaurav Mishra, James Bradbury, Daniel Andor, Sharan Narang, Brian Lester, Colin Gaffney, Afroz Mohiuddin, Curtis Hawthorne, Aitor Lewkowycz, Alex Salcianu, Marc van Zee, Jacob Austin, Sebastian Goodman, Livio Baldini Soares, Haitang Hu, Sasha Tsvyashchenko, Aakanksha Chowdhery, Jasmijn Bastings, Jannis Bulian, Xavier Garcia, Jianmo Ni, Andrew Chen, Kathleen Kenealy, Jonathan H. Clark, Stephan Lee, Dan Garrette, James Lee-Thorp, Colin Raffel, Noam Shazeer, Marvin Ritter, Maarten Bosma, Alexandre Passos, Jeremy Maitin-Shepard, Noah Fiedel, Mark Omernick, Brennan Saeta, Ryan Sepassi, Alexander Spiridonov, Joshua Newlan, and Andrea Gesmundo. Scaling up models and data with t5x and seqio. arXiv preprint arXiv:2203.17189, 2022.

[29] Rico Sennrich, Barry Haddow, and Alexandra Birch. Neural machine translation of rare words with subword units. arXiv preprint arXiv:1508.07909, 2015.

[30] Rico Sennrich, Barry Haddow, and Alexandra Birch. Neural machine translation of rare words with subword units. In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1715–1725, Berlin, Germany, August 2016. Association for Computational Linguistics.

[31] Anima Singh, Trung Vu, Raghunandan Keshavan, Nikhil Mehta, Xinyang Yi, Lichan Hong, Lukasz Heldt, Li Wei, Ed Chi, and Maheswaran Sathiamoorthy. Better generalization with semantic ids: A case study in ranking for recommendations. arXiv preprint arXiv:2306.08121, 2023.

[32] Fei Sun, Jun Liu, Jian Wu, Changhua Pei, Xiao Lin, Wenwu Ou, and Peng Jiang. Bert4rec: Sequential recommendation with bidirectional encoder representations from transformer. In Proceedings of the 28th ACM international conference on information and knowledge management, pages 1441–1450, 2019.

[33] Jiaxi Tang and Ke Wang. Personalized top-n sequential recommendation via convolutional sequence embedding. In Proceedings of the eleventh ACM international conference on web search and data mining, pages 565–573, 2018.

[34] Yi Tay, Vinh Q Tran, Mostafa Dehghani, Jianmo Ni, Dara Bahri, Harsh Mehta, Zhen Qin, Kai Hui, Zhe Zhao, Jai Gupta, et al. Transformer memory as a differentiable search index. arXiv preprint arXiv:2202.06991, 2022.

[35] Aaron Van Den Oord, Oriol Vinyals, et al. Neural discrete representation learning. Advances in neural information processing systems, 30, 2017.

[36] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

[37] Yujing Wang, Yingyan Hou, Haonan Wang, Ziming Miao, Shibin Wu, Hao Sun, Qi Chen, Yuqing Xia, Chengmin Chi, Guoshuai Zhao, et al. A neural corpus indexer for document retrieval. arXiv preprint arXiv:2206.02743, 2022.

[38] Kilian Weinberger, Anirban Dasgupta, John Langford, Alex Smola, and Josh Attenberg. Feature hashing for large scale multitask learning. In Proceedings of the 26th annual international conference on machine learning, pages 1113–1120, 2009.

[39] Xinyang Yi, Ji Yang, Lichan Hong, Derek Zhiyuan Cheng, Lukasz Heldt, Aditee Kumthekar, Zhe Zhao, Li Wei, and Ed Chi. Sampling-bias-corrected neural modeling for large corpus item recommendations. In Proceedings of the 13th ACM Conference on Recommender Systems, pages 269–277, 2019.

[40] Neil Zeghidour, Alejandro Luebs, Ahmed Omran, Jan Skoglund, and Marco Tagliasacchi. Soundstream: An end-to-end neural audio codec. CoRR, abs/2107.03312, 2021.

[41] Shuai Zhang, Yi Tay, Lina Yao, and Aixin Sun. Next item recommendation with self-attention. arXiv preprint arXiv:1808.06414, 2018.

[42] Tingting Zhang, Pengpeng Zhao, Yanchi Liu, Victor S Sheng, Jiajie Xu, Deqing Wang, Guanfeng Liu, and Xiaofang Zhou. Feature-level deeper self-attention network for sequential recommendation. In IJCAI, pages 4320–4326, 2019.

[43] Zhe Zhao, Lichan Hong, Li Wei, Jilin Chen, Aniruddh Nath, Shawn Andrews, Aditee Kumthekar, Maheswaran Sathiamoorthy, Xinyang Yi, and Ed Chi. Recommending what video to watch next: a multitask ranking system. In Proceedings of the 13th ACM Conference on Recommender Systems, pages 43–51, 2019.

[44] Kun Zhou, Hui Wang, Wayne Xin Zhao, Yutao Zhu, Sirui Wang, Fuzheng Zhang, Zhongyuan Wang, and Ji-Rong Wen. S3-rec: Self-supervised learning for sequential recommendation with mutual information maximization. In Proceedings of the 29th ACM International Conference on Information & Knowledge Management, pages 1893–1902, 2020.
