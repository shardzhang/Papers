# 如何为推荐基础模型索引itemID

> Wenyue Hua, Shuyuan Xu, Yingqiang Ge, Yongfeng Zhang | 罗格斯大学

本文介绍了针对推荐基础模型中itemID创建与索引问题的系统性研究。核心内容：

- 推荐基础模型利用大语言模型（LLM，Large Language Model）将推荐任务转化为自然语言任务，实现生成式推荐，直接生成待推荐item而非逐一计算候选item的排序分数
- 为避免生成过长文本和幻觉推荐，需为每个item创建LLM兼容的唯一ID（即token序列），使其由少量特征性token表示且相互可区分
- 本文首先讨论三种简单item索引方法的局限性：随机索引（RID，Random Indexing）、标题索引（TID，Title Indexing）和独立索引（IID，Independent Indexing）
- 进而提出四种简单有效的解决方案：顺序索引（SID，Sequential Indexing）、协同索引（CID，Collaborative Indexing）、语义（基于内容的）索引（SemID，Semantic Indexing）和混合索引（HID，Hybrid Indexing）

关键发现：

- item索引方法对基于LLM的推荐性能具有显著影响，RID和TID在大多数情况下无法超越基线，IID虽有改善但仍有限
- CID基于谱聚类树构建索引，使共现频率更高的item共享更多token；SemID利用类别元数据构建层次化索引；两者均显著优于简单方法
- 混合索引CID+IID和SemID+IID表现最优，结合了协同/语义信息与独立区分能力
- 顺序索引中时间敏感排序效果最佳；协同索引的最优平均ID长度在3到4之间；语义索引要求类别信息遵循树结构才能获得良好性能

---

## 摘要

推荐基础模型利用大语言模型（LLM）将推荐任务转化为自然语言任务，实现生成式推荐——直接生成待推荐item而非像传统推荐模型那样逐一计算每个候选item的排序分数，从而将推荐流程从多阶段过滤简化为单阶段过滤。为避免在决定推荐哪个item时生成过长文本和幻觉推荐，创建兼容LLM的item ID以唯一标识每个item对推荐基础模型至关重要。在本研究中，我们以P5为骨干LLM示例，系统性地研究了推荐基础模型的item ID创建与索引问题。为强调item索引的重要性，我们首先讨论了几种简单item索引方法的问题，如随机索引、标题索引和独立索引。随后提出四种简单有效的解决方案，包括顺序索引、协同索引、语义（基于内容的）索引和混合索引。我们的研究突出了item索引方法对基于LLM推荐性能的显著影响，在真实数据集上的结果验证了所提方案的有效性。本研究还展示了语言建模的最新进展与传统信息检索（IR，Information Retrieval）原理中的索引技术如何相互促进，以实现更好的学习和推理。源代码和数据可在 https://github.com/Wenyueh/LLM-RecSys-ID 获取。

## CCS 概念

- 信息系统 $\rightarrow$ 推荐系统
- 计算方法学 $\rightarrow$ 机器学习；自然语言处理

## 关键词

Large Language Model; Recommendation; Item ID and Indexing

## ACM 引用格式

Wenyue Hua, Shuyuan Xu, Yingqiang Ge, and Yongfeng Zhang. 2023. How to Index Item IDs for Recommendation Foundation Models. In Annual International ACM SIGIR Conference on Research and Development in Information Retrieval in the Asia Pacific Region (SIGIR-AP '23), November 26–28, 2023, Beijing, China. ACM, New York, NY, USA, 10 pages. https://doi.org/10.1145/3624918.3625339

允许以个人或课堂使用为目的制作或分发本作品的数字或硬拷贝，前提是未以盈利或商业优势为目的进行分发，且副本承载本声明和首页的完整引用。本作品中由作者以外其他人拥有的版权必须予以尊重。允许以注明方式引用进行摘要。如需以其他方式复制、重新发布、发布到服务器或重新分发到列表，则需要事先获得特定许可和/或费用。如需许可，请联系 permissions@acm.org。

SIGIR-AP '23，2023年11月26日至28日，中国北京

© 2023 版权归所有者/作者所有。出版权由ACM许可。

ACM ISBN 979-8-4007-0408-6/23/11...\$15.00

https://doi.org/10.1145/3624918.3625339

## 1 引言

基础模型（Foundation Models）如大语言模型（LLM，Large Language Model）[3, 4, 27]已经显著影响了自然语言处理（NLP，Natural Language Processing）和计算机视觉（CV，Computer Vision）[19]等研究领域，并已被应用于各种推荐系统（RS，Recommender System）任务。最近的研究如P5 [9]和M6Rec [6]利用了预训练LLM用于推荐的优势[17]：它们将丰富的用户行为和知识信息融入预训练中，并受益于基础模型强大的学习能力。预训练的LLM还具有改进的推理能力[11]，能够基于上下文推断用户兴趣。因此，这些模型旨在利用在大量自然语言语料库上预训练的LLM，通过将推荐任务转化为语言生成任务来实现生成式推荐。

由于item描述可能包含大量词汇（例如，产品标题/描述可能包含数十/数百个词，新闻文章可能包含数千个词），我们很难期望LLM在决定推荐哪个item时生成完整且精确的item描述，因为生成的文本可能甚至不对应item数据库中真实存在的item，导致基于LLM推荐中的幻觉问题[8, 18]。因此，为每个item分配唯一ID，使其由少量特征性token表示且相互可区分，至关重要。例如，Yelp中的一个商业位置可能被分配ID"location_4332"，并进一步表示为token序列 $\langle location \rangle \langle \_ \rangle \langle 43 \rangle \langle 32 \rangle$ [9]。注意，item ID不一定必须是数字token，只要它是item的唯一标识符，就可以被视为该item的ID。例如，电影"The Lord of the Rings"的标题可以作为该电影的ID，它由一系列词token而非数字token组成。ID甚至可以是一系列不传达明确含义的词序列，例如"ring epic journey fellowship adventure"。

然而，为item分配兼容LLM的ID并非易事。首先，可能存在大量甚至无限的item，而每个item应被分配唯一ID，使其对于基础模型而言相互可区分。其次，item ID应与自然语言兼容，使其能被整合到LLM预训练、微调和提示的自然语言指令中。第三，简单的item索引方法如随机索引可能不仅无益，甚至可能损害推荐基础模型，因为它们可能错误地将相关ID分配给不相关的item，误导LLM的训练和提示。因此，需要对面向LLM的item索引进行全面研究，以实现推荐任务与LLM的无缝适配，释放LLM用于推荐的潜力。

此外，确保生成的文本与真实item对齐以避免幻觉问题的一个自然想法是采用约束解码方法[7]。然而，对自由形式的长文本使用约束生成并不实际。这是因为约束解码本质上规定了一种单一的内容表达方式，否定了长文本叙事的灵活性。通过强制模型遵循特定的描述模式，模型除了推荐特定知识外还需要记忆刚性的文本模式。这种额外的复杂性可能稀释模型的主要目的，并阻碍其执行核心推荐任务的有效性。

基于上述原因，本文专注于基于LLM的推荐器中的item索引问题：如何为每个item分配唯一ID（即token序列）。我们基于P5 [9]——一个代表性的推荐系统LLM模型——来研究该问题。P5采用基于基础模型的预训练，并根据个性化提示将推荐任务转化为自然语言句子。我们首先在三种简单索引方法上进行实验并展示其局限性，其中一些曾被先前模型采用：独立索引（IID）、标题索引（TID）和随机索引（RID）。基于分析，我们进一步探索四种新颖的索引技术：顺序索引（SID）、协同索引（CID）、语义索引（SemID）和混合索引（HID）。为确保推荐阶段生成的ID与真实item对齐以避免幻觉，我们开发了一种约束解码方法[7]，其通过从有效ID集合构建前缀树（即trie树），并在解码阶段将不存在的ID的生成概率设为零来实现。我们在三个广泛使用的数据集（Amazon Beauty、Amazon Sports、Yelp）上展示了各种ID方法的性能，并为不同方法用于基于LLM的推荐模型提供了见解。

## 2 相关工作

许多传统推荐模型使用基于匹配的范式[1, 2, 13, 15, 29, 30, 32]。它们将用户（或用户行为历史）和item投影到共享嵌入空间中，然后通过计算嵌入向量的排序分数来估计用户对item的偏好，例如矩阵分解中用户和item向量的内积[15]。通常，这涉及为每个候选item计算排序分数，使得匹配和排序过程在item池很大时非常耗时[34]。因此，工业推荐系统通常必须使用多阶段（通常是两阶段）过滤流程[5]，其中简单高效的过滤方法（如基于规则的过滤方法）用于早期阶段，而先进的过滤方法用于候选item较少的后期阶段。结果，最先进的模型仅应用于item的一个小子集。

最近，有多项尝试对基础模型进行预训练以实现生成式推荐，这省去了昂贵的逐个候选item匹配过程，转而直接生成待推荐item。例如，P5 [9]将多样化的推荐任务统一为序列到序列生成框架内的自然语言生成任务。推荐数据如用户-item交互、用户描述、item元数据和用户评论通过多个个性化提示模板转换为自然语言序列的通用格式。每个用户或item由唯一的token序列表示作为用户或item ID。M6Rec [6]将各种推荐任务（如内容供应、投放和展示）转化为自然语言理解或生成任务。输入提示包含用户属性、过往行为和卖家提供的详细item描述。用户和item表示为基于其属性和描述的预计算嵌入。LMRecSys [33]将基于item的推荐任务转化为文本填空任务。该模型在MovieLens-1M数据集[10]上进行测试，该数据集包含预训练LLM可能已在网络文本中见过的电影。Item由其标题表示，作为索引。这种索引方法对模型性能产生了负面影响，正如原始论文所报告的：LLM不仅在推断多token跨度的概率分布方面效果不佳，而且标题中包含的语言偏差可能误导模型，因为标题可能几乎不包含关于电影内容的信息。

这三种模型使用不同的方法索引item：P5使用数字token，M6Rec使用基于元数据的嵌入，LMRecSys使用item标题。本文使用P5作为示例骨干，研究基于LLM的生成式推荐框架下的不同item索引方法，比较不同索引方法的有效性，揭示item索引与基础模型预训练之间的关系，并为推荐基础模型的预训练提供关于哪些item索引方法最适合的见解。

## 3 预备知识与前期研究

### 3.1 P5范式简介

本文基于P5 [9]研究索引问题。P5是一个代表性的推荐基础模型，通过集成各种任务和个性化指令提示来预训练推荐基础模型，增强现有推荐系统的泛化能力。这些任务包括序列推荐、评分预测、解释生成、评论摘要和直接推荐。P5使用为不同用户和item生成的输入-目标文本对进行训练，这些文本对来源于一组包含个性化字段的提示模板：序列推荐的示例输入提示可以是用户-item交互的描述，如"According to the places user_1 has visited: location_1123, location_4332, location_8463, location_12312, can you recommend another place for the user?"，输出文本是下一个生成的item，如"Output: location_1934"。在本研究中，我们聚焦于序列推荐任务，因为它明确依赖于输入提示中呈现的item交互，使其对不同索引方法高度敏感。

### 3.2 尖括号表示法

在本文中，我们需要引入词汇外（OOV，Out-of-Vocabulary）token来构建某些索引方法中的item索引，这些token不属于语言模型的常规词汇表。在我们的案例中，它们是不存在于默认T5词汇表[23]中的token。为区分新创建的OOV token和现有token，我们使用尖括号 $\langle\rangle$ 来表示新创建的OOV token，而使用不带 $\langle\rangle$ 的文本来表示现有token。所有OOV token在模型中都是随机初始化的，因此 $\langle\rangle$ 内的文本不会影响OOV token的嵌入。尖括号 $\langle\rangle$ 内的文本可以是单词或数字，但无论哪种情况，尖括号内的文本仅用于区分不同的OOV token，与现有token无关。例如，$\langle restaurant \rangle \langle Greek \rangle \langle 2 \rangle$ 是Yelp中某个item的索引，由三个OOV token组成，其中 $\langle restaurant \rangle$ 是不同于普通英文单词"restaurant"的token，$\langle 2 \rangle$ 是不同于数字"2"的token。当我们需要使用现有的普通词token时，将不使用尖括号，如"restaurant"和"2"。

### 3.3 数据格式与预处理

实验在Amazon Sports & Outdoors、Amazon Beauty和Yelp数据集上进行。Amazon数据集[22]¹来自Amazon.com用于产品推荐，Yelp数据集²提供了用于商业推荐的用户评分和评论集合。我们使用2019年1月1日至2019年12月31日的交易记录，与原始P5论文[9]一致。这些数据集的详细统计信息见表1。

**表1：数据集的基本统计信息。**

| 指标 | Sports | Beauty | Yelp |
|------|--------|--------|------|
| 用户数 | 35,598 | 22,363 | 30,431 |
| item数 | 18,357 | 12,101 | 20,033 |
| 交互数 | 296,337 | 198,502 | 316,354 |
| 稀疏度(%) | 0.0453 | 0.0734 | 0.0519 |

这些数据集按单个用户组织用户-item交互。我们按照常用的留一法（leave-one-out）设置将数据集划分为训练集、验证集和测试集：对于每个用户的交互序列，我们将倒数第二个item放入验证集，将最后一个item放入测试集，将序列中的所有其他item放入训练集。例如，假设用户 $i$ 的交互序列是 $\{item_{i,1}, item_{i,2}, item_{i,3}, \cdots, item_{i,k-1}, item_{i,k}\}$。则基于序列 $\{item_{i,1}, item_{i,2}, item_{i,3}, \cdots, item_{i,k-2}\}$ 对 $item_{i,k-1}$ 的预测用于验证，基于序列 $\{item_{i,1}, item_{i,2}, item_{i,3}, \cdots, item_{i,k-1}\}$ 对 $item_{i,k}$ 的预测用于测试。

### 3.4 item索引的动机分析

我们从三种简单索引方法出发，探索索引方法的动机：

- **随机索引（RID）**：为每个item分配一个随机数字作为item ID。该数字进一步基于SentencePiece分词器[24]被分词为子token序列，与P5 [9]的做法相同。例如，一个Yelp item被随机分配数字"4332"，"4332"被表示为token序列"43""32"。
- **标题索引（TID）**：使用item标题来表示item，同样由SentencePiece [24]分词。例如，Yelp item"Las Vegas Cigar Outlet"被表示为token序列"Las""Vegas""Ci""gar""Outlet"。
- **独立索引（IID）**：为每个item创建一个需要学习的独立OOV额外token。例如，一个Yelp item被表示为 $\langle IID5 \rangle$，这是一个专门为该item分配的独立额外token。在本文的其余部分，为IID创建的token将始终以字母"IID"开头。

RID生成随机数字索引，导致分词后不相关item之间可能存在重叠。例如，两个item"4332"和"4389"将分别被分词为"43""32"和"43""89"，这意味着它们总是共享相同的子token"43"，即使这两个item可能完全不相关。这种非预期的重叠可能在item之间建立任意关系，为模型训练引入不必要的偏差。由于重叠源于索引结构，无论模型如何从数据中学习，都无法消除这些重叠。因此，RID被认为是一种不理想的方法。

TID使任务更具挑战性，因为模型需要记忆和生成冗长的item标题。此外，标题中的某些词汇或表达可能与item的真实内容无关，而且非常不同的item可能在其标题中共享重叠的token，因此从标题中导出的语义可能引入强烈偏差[33]。例如，电影"The Lord of the Rings"和"The Lord of War"在标题中共享许多token（"the"、"lord"、"of"），但它们是两部非常不同的电影：前者是史诗奇幻片，后者是犯罪剧情片。一般来说，两个不相关的item可能有非常相似的标题，如Apple（水果）和Apple（公司），而两个紧密相关的item可能有非常不同的标题，如数据挖掘中经典的"beer and diaper"例子[17]。因此，使用标题作为ID可能将误导性语义编码到生成过程中，类似于随机索引的问题。

IID为item使用单token索引，不假设关于item的任何先验信息，与RID和TID相比，使item表示更容易被语言模型学习。尽管优于RID和TID，但由于在创建item ID时假设所有item相互独立，其性能仍然有限。如果需要创建大量新token，也可能导致过长的训练时间。

上述分析表明三种方法都不是最优的。为验证这一点，我们提供实验结果来展示其次优性能。我们针对两个强大且广泛使用的基线评估了三种索引方法：SASRec [14]和S3-Rec [35]。结果如表2所示，每个指标的最佳结果以粗体显示，次佳结果以波浪下划线显示。基于表2，RID和TID相对于基线表现较差，而IID以引入更多可学习token为代价提供了有限的收益。因此，这些索引方法被认为是次优的，我们将在下一节进一步探索非平凡的索引方法。

**表2：简单索引方法用于P5的性能以及基线的性能。粗体数字代表最佳结果，波浪下划线数字代表次佳结果。RID和TID在Sports和Beauty上的结果在配对Student's $t$ 检验下显著较差（$p < 0.05$）。**

| 方法 | HR@5 | NCDG@5 | HR@10 | NCDG@10 | HR@5 | NCDG@5 | HR@10 | NCDG@10 | HR@5 | NCDG@5 | HR@10 | NCDG@10 |
|------|------|--------|-------|---------|------|--------|-------|---------|------|--------|-------|---------|
| | Sports | | | | Beauty | | | | Yelp | | | |
| SASRec | 0.0233 | 0.0154 | 0.0350 | 0.0192 | 0.0387 | 0.0249 | 0.0605 | 0.0318 | 0.0170 | 0.0110 | 0.0284 | 0.0147 |
| S3-Rec | 0.0251 | 0.0161 | 0.0385 | 0.0204 | 0.0387 | 0.0244 | 0.0647 | 0.0327 | 0.0201 | 0.0123 | 0.0341 | 0.0168 |
| RID | 0.0208 | 0.0122 | 0.0288 | 0.0153 | 0.0213 | 0.0178 | 0.0479 | 0.0277 | 0.0225 | 0.0159 | 0.0329 | 0.0193 |
| TID | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0182 | 0.0132 | 0.0432 | 0.0254 | 0.0058 | 0.0040 | 0.0086 | 0.0049 |
| IID | 0.0268 | 0.0151 | 0.0386 | 0.0195 | 0.0394 | 0.0268 | 0.0615 | 0.0341 | 0.0232 | 0.0146 | 0.0393 | 0.0197 |

## 4 非平凡索引方法

基于上述分析，一个最优的item索引方法应满足两个标准以实现有效的学习过程：(1) 保持合适的长度以降低文本生成难度；(2) 将先验信息整合到item索引结构中，确保相似item共享尽可能多的token同时可区分，不相似的item共享最少的token。

为实现这些目标，我们引入并探索了四种复杂度递增的索引方法：顺序索引（SID）、协同索引（CID）、语义（基于内容的）索引（SemID）和混合索引（HID）。SID和CID利用协同信息，使共现item共享token。SemID利用自然语言中的元数据，使语义相似的item共享token。HID组合多种索引方法，旨在利用每种方法的优势以生成最优索引。在接下来的子节中，我们将提供四种索引方法的详细信息。

### 4.1 顺序索引

顺序索引是一种利用协同信息进行item索引的直观方法。用户连续交互的item被分配连续的数字索引，反映其共现关系。以表3为例，item从第一个用户开始依次分配ID，直到最后一个用户。如果一个item已在先前用户的交互序列中被索引过（如用户2序列中的item 1001及表中所有方框标记的item），则使用该item已分配的ID；否则，将创建并分配一个新的递增ID。注意，item索引过程仅依赖于训练序列，验证和测试item不参与索引过程。索引过程完成后，验证和测试item被分配在索引过程中已建立的相应ID。在基于SentencePiece分词器[24]进行分词后，ID"1001"将被分词为"100""1"，而"1002"将被分词为"100""2"，导致这两个连续item共享token"100"。因此，这种简单的顺序索引方法能够在某些场合捕捉协同信息。

一个小细节是我们从1001开始item索引枚举。我们从1001而非1开始有两个原因：(1) SentencePiece分词器不会将某些小于1000的数字分词为多个子token（例如数字12），因此分配了这些小数字的item将完全相互独立；(2) 分词后，较小的数字可能成为较大分词数字的完整子集，例如ID"12"可以是ID"12""34"的子集，这可能在item之间强制建立错误的相关性。

尽管如此，顺序索引也有局限性：(1) 被索引为相邻但未被同一用户交互的item可能错误地共享token；例如，用户的最后一个item被索引为1014（分词为"10""14"），而下一个用户的第一个item被索引为1015（分词为"10""15"），则token"10"将被共享，尽管两个item之间缺乏共现；(2) 它无法捕捉基于共现频率的相似性；例如，假设item 1001和1002共现一次，而item 1002和1003共现十次，两个pair仍然只共享一个token，无法传达频率信息；(3) 训练数据中的用户顺序影响结果；例如，如果我们在表3中交换用户1和用户2的行，则索引结果将不同。尽管顺序索引有其缺点，它仍能产生相对不错的结果，接近甚至超越基线。

**表3：顺序索引方法的说明。方框中的数字代表先前已索引的item。**

| | 训练序列 | | | | | | | | 验证 | 测试 |
|------|------|------|------|------|------|------|------|------|------|------|
| 用户1 | 1001 | 1002 | 1003 | 1004 | 1005 | 1006 | 1007 | 1008 | 1009 | 1018 | 1019 |
| 用户2 | 1010 | 1011 | 1001 | 1012 | 1008 | 1009 | 1013 | 1014 | 1022 | 1023 |
| 用户3 | 1015 | 1016 | 1017 | 1007 | 1018 | 1019 | 1020 | 1021 | 1009 | 1015 | 1016 |
| 用户4 | 1022 | 1023 | 1005 | 1002 | 1006 | 1024 | | | 1002 | 1008 |
| 用户5 | 1025 | 1026 | 1027 | 1028 | 1029 | 1030 | 1024 | 1020 | 1021 | 1031 | 1033 | 1034 |

### 4.2 协同索引

顺序索引是将协同信息整合到item索引中的初步方法。为有效捕捉协同过滤的本质，我们探索了协同索引（CID）方法，该方法采用基于谱矩阵分解（SMF，Spectral Matrix Factorization）[21, 28]的谱聚类来生成item索引。该方法基于一个前提：共现频率更高的item更相似，应在索引构建中共享更多重叠的token。核心概念是基于训练数据集为所有item构建共现图，并使用谱聚类将item分组到聚类中，确保同一聚类内的item在构建索引时共享token。

#### 4.2.1 基于谱矩阵分解的谱聚类

具体而言，我们基于训练集创建一个图，如图1(a)所示：每个item作为一个节点，两个item之间的边代表它们的共现关系（即两个item共同出现在用户的交互序列中），边权重表示共现频率（即两个item共同出现的用户交互序列数量）。图的邻接矩阵（图1(b)）表示item在共现频率方面的相似性，图的拉普拉斯矩阵（图1(c)）可以分解以实现谱聚类[21, 28]。谱聚类过程将item分组到聚类中，使得共享更多共现相似性的item被分到同一聚类中；每个聚类可以通过递归地在该大聚类内应用谱聚类过程进一步细分为更细粒度的聚类，从而形成层次化的聚类层级，如图1(a)所示。

**图1：基于谱矩阵分解的item共现图谱聚类示意图。(a) item共现图上的递归谱聚类；(b) 邻接矩阵；(c) 拉普拉斯矩阵。**

更具体地说，谱聚类利用拉普拉斯矩阵的特征向量将节点分组到聚类中[21, 28]。它确保同一聚类内的item具有更高的相似性，而不同聚类的item表现出较低的相似性。我们使用Python scikit-learn包³中的标准谱聚类实现。我们不展开谱聚类算法的过多细节，因为它被认为是数据分析的教科书级算法[16]。然而，我们确实想讨论用于控制递归聚类过程的两个重要参数：(1) $N$：我们在聚类的每一层将item划分为 $N$ 个聚类；(2) $k$：最终聚类中允许的最大item数，作为递归聚类过程的停止标准，即当一个聚类包含最多 $k$ 个item时，我们将不再进一步缩减其大小。最后，聚类结果可以被制定为层次化的树结构，如图2所示。在该图中，每个非叶节点（图中的大黄色节点）代表在对应层创建的聚类，每个叶节点（小蓝色节点）代表对应最终聚类中的一个item。在下一子节中，我们将介绍如何基于层次化树结构创建item ID。

³https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html

#### 4.2.2 基于谱聚类树的item索引

如上所述，递归聚类过程为聚类和item生成了一个树结构，如图2所示（以 $N=4$ 和 $k=20$ 为例），这意味着谱聚类的每次迭代将item划分为4个聚类，该过程递归应用于每个聚类，直到聚类大小小于或等于20。每个非叶节点（大黄色节点）代表一个聚类，而所有item作为叶节点（小蓝色节点）存在于最终聚类下。注意，由于最终聚类中允许的最大item数为 $k$，这意味着我们最多只需要 $k$ 个独立额外token来区分同一最终聚类内的item（即同一黄色节点下的小蓝色节点最多为 $k$ 个）。因此，我们将 $k$ 个独立额外token引入词汇表，记为 $\langle 0 \rangle, \langle 1 \rangle, \langle 2 \rangle, \cdots, \langle k-1 \rangle$。

我们首先为非叶节点分配token。非叶节点使用 $k$ 个独立token从 $\langle 0 \rangle$ 到 $\langle k-1 \rangle$ 逐层遍历整棵树进行枚举，如图2所示。一旦所有 $k$ 个token被使用，我们简单地从 $\langle 0 \rangle$ 重新开始。如前所述，每个父聚类节点有 $N$ 个子聚类节点。然而，如果 $N > k$，则我们没有足够的token来区分同一父节点下的不同子节点。因此，我们要求 $N \leq k$ 用于协同索引。结合逐层token分配过程，这可以保证同一父节点下的不同子节点被分配不同的token。

**图2：基于谱聚类树的协同索引（$N=4$，$k=20$）。**

然后我们为叶节点（小蓝色节点）分配token，其中每个叶节点是一个item。这相当直接：对于每个最终聚类，我们从 $\langle 0 \rangle$ 开始为其每个子item节点分配独立的额外token。由于聚类过程确保每个最终聚类包含最多 $k$ 个item，因此 $k$ 个独立额外token足以区分同一最终聚类下的不同item。

最后，item的ID是其非叶祖先节点token和自身叶节点token的拼接。例如，图2中加粗路径下的item被索引为 $\langle 1 \rangle \langle 9 \rangle \langle 5 \rangle \langle 4 \rangle$。这种索引过程保证同一最终聚类内的任意两个item将共享直到其在最终聚类内自身token之前的所有token，这意味着两个item共现越频繁，它们将共享越多的token，很好地利用了用户行为序列中隐藏的协同信息。

### 4.3 语义（基于内容的）索引

语义（基于内容的）索引（SemID）利用item元数据为item构建ID。如图3所示，item的类别形成层次化结构[36]，每个非叶节点（大黄色节点）代表一个类别，每个叶节点（小蓝色节点）代表一个item。每个非叶节点被分配一个独立的额外token，每个叶节点在其父节点下接收一个唯一的额外token。为创建item索引，非叶节点和叶节点的token沿从根到叶的路径拼接。以图3中加粗路径为例，该item的类别从粗粒度到细粒度为 $\langle Makeup \rangle$、$\langle Lips \rangle$、$\langle Lip\_Liners \rangle$，其叶节点token为 $\langle 5 \rangle$（用于区分该item与Lip Liners类别下的其他item），则该item被索引为 $\langle Makeup \rangle \langle Lips \rangle \langle Lip\_Liners \rangle \langle 5 \rangle$。

**图3：语义索引示例。**

### 4.4 混合索引

混合索引（HID）不是单一的特定索引方法，而是一类方法。它将上述多种索引拼接为一个索引，如SID+IID、CID+IID、SemID+IID、SemID+CID等。该方法旨在利用不同索引技术的优势以生成更好的索引。在本文中我们实现了四种组合，详情如下：

对于SID+IID：我们在每个item的顺序ID末尾追加一个独立额外token。假设一个item的SID分词后为"10""18"，其IID索引为 $\langle IID982 \rangle$，则HID索引将为"10""18"$\langle IID982 \rangle$。因此它包含来自SID的一些item共现信息，同时通过IID确保item区分性。

对于CID和SemID，在将它们与IID拼接之前，我们首先移除最后一个token（叶节点token），因为最后一个token仅用于区分同一父非叶节点下的item。对于CID+IID：假设一个item的CID为 $\langle 1 \rangle \langle 9 \rangle \langle 5 \rangle \langle 4 \rangle$，其IID为 $\langle IID28 \rangle$，则该item的HID为 $\langle 1 \rangle \langle 9 \rangle \langle 5 \rangle \langle IID28 \rangle$。对于SemID+IID：假设一个item的SemID为 $\langle Makeup \rangle \langle Lips \rangle \langle Lip\_Liners \rangle \langle 5 \rangle$，其IID为 $\langle IID1023 \rangle$，则HID为 $\langle Makeup \rangle \langle Lips \rangle \langle Lip\_Liners \rangle \langle IID1023 \rangle$。最终索引同时包含来自CID（或SID中的元数据内容信息）的协同信息和区分该item与所有其他item的特殊IID token，在保留CID（或SID）优势的同时确保item区分性。

对于SemID+CID，我们以任意顺序拼接SemID和CID，希望同时结合元数据内容信息和协同信息。由于SemID和CID都包含用于区分一个父节点下item的叶节点token，我们只需保留其中一个，例如保留CID叶节点token。假设SemID为 $\langle Makeup \rangle \langle Lips \rangle \langle Lip\_Liners \rangle \langle 5 \rangle$，CID为 $\langle 1 \rangle \langle 9 \rangle \langle 5 \rangle \langle 4 \rangle$。如果SemID在前，最终HID索引为 $\langle Makeup \rangle \langle Lips \rangle \langle Lip\_Liners \rangle \langle 1 \rangle \langle 9 \rangle \langle 5 \rangle \langle 4 \rangle$；否则，HID索引为 $\langle 1 \rangle \langle 9 \rangle \langle 5 \rangle \langle 4 \rangle \langle Makeup \rangle \langle Lips \rangle \langle Lip\_Liners \rangle$。

在以下实验中，我们将评估和比较各种不同的HID。

## 5 实验

### 5.1 数据集与基线

数据集及其预处理方法已在第3.3节中介绍。在本节中，我们介绍基线。我们将各种item索引方法应用于P5框架[9]进行序列推荐，并与几种代表性的序列推荐方法作为基线进行比较：

- **Caser [26]**：该方法将序列推荐视为马尔可夫链，利用卷积神经网络建模用户兴趣。
- **HGN [20]**：该方法利用层次化门控网络从长期和短期视角学习用户行为。
- **GRU4Rec [12]**：最初为基于会话的推荐提出，该方法利用GRU建模用户点击历史序列。
- **BERT4Rec [25]**：该方法模拟BERT风格的掩码语言建模，学习双向表示用于序列推荐。
- **FDSA [31]**：聚焦于特征转移模式，该方法使用自注意力模块建模特征序列。
- **SASRec [14]**：在序列推荐模型中采用自注意力机制，该方法调和了马尔可夫链和基于RNN方法的特性。
- **S3-Rec [35]**：利用item元信息上的自监督目标，该方法帮助序列推荐模型更好地发现不同item及其属性之间的相关性。为进行比较，我们使用S3-Rec及其基线的实现。

### 5.2 实现细节

遵循P5框架[9]，我们的实现使用T5 [23]作为骨干：编码器和解码器各有6层，模型维度为512，8头注意力。对于分词，我们使用默认的SentencePiece分词器[24]，词汇表大小为32,128，用于解析子词单元。所有独立额外token不再进一步分词。我们使用与P5 [9]相同的序列推荐提示将序列信息转化为文本。我们使用AdamW优化器在两张NVIDIA RTX A5000 GPU上预训练P5共20个epoch，批大小为64，峰值学习率为 $10^{-3}$。我们在前5%的训练步数上应用预热来调整学习率。

RID、TID和SID不涉及创建OOV token，因为它们的item索引由默认T5分词器中的token组成；而IID、CID、SemID和HID涉及创建额外的OOV token，扩展了原始词汇表。除TID外，这些索引方法中使用的所有token都是随机初始化的，而非使用T5的预训练嵌入进行初始化。这是因为我们在实验中观察到预训练T5关于数字的先验语义对item语义的学习和推荐性能产生了不利影响。对于TID token，我们使用T5的预训练token嵌入进行初始化，因为TID仅涉及普通词token。

### 5.3 整体结果

整体实验结果如表4所示，包含所有基线。每个指标的最佳结果以粗体显示，次佳结果以波浪下划线显示。对于每种索引方法，如果结果超越了最佳基线结果，则以直线下划线强调。总体而言，RID、TID和IID在大多数情况下无法超越基线结果，而大多数高级索引方法（SID、CID、SemID及HID）超越了基线结果。更详细的分解分析如下。

在表4中，第一个块包含所有基线结果。第二个块包含基本索引方法，其中RID和TID始终表现差于基线，而IID总体表现更好。第三个块包含三种高级索引方法。我们可以看到SID在Amazon数据集上表现差于CID和SemID，但在Yelp上表现更好；而CID在不同数据集上均表现优于SemID，表明使用协同信息构建索引比使用元数据更有益，因为CID能够通过从群体智慧中进行协同学习，从用户行为中更好地捕捉item关系，这可能比仅使用item元数据更有效。表中的第四个块包含HID结果及几种不同实现：SID+IID、CID+IID、SemID+IID和SemID+CID。CID+IID和SemID+IID的表现远优于所有其他索引方法，而SID+IID和SemID+CID表现较差。在接下来的子节中，我们将基于更全面的实验进一步详细分析第三和第四块中的结果。

**表4：所有基线结果和所有索引方法在P5下的性能。粗体数字代表最佳结果，波浪下划线数字代表次佳结果，直线下划线表示优于最佳基线结果。优于基线的结果经配对Student's $t$ 检验在 $p < 0.05$ 下显著。**

| 方法 | HR@5 | NCDG@5 | HR@10 | NCDG@10 | HR@5 | NCDG@5 | HR@10 | NCDG@10 | HR@5 | NCDG@5 | HR@10 | NCDG@10 |
|------|------|--------|-------|---------|------|--------|-------|---------|------|--------|-------|---------|
| | Sports | | | | Beauty | | | | Yelp | | | |
| Caser | 0.0116 | 0.0072 | 0.0194 | 0.0097 | 0.0205 | 0.0131 | 0.0347 | 0.0176 | 0.0150 | 0.0099 | 0.0263 | 0.0134 |
| HGN | 0.0189 | 0.0120 | 0.0313 | 0.0159 | 0.0325 | 0.0206 | 0.0512 | 0.0266 | 0.0186 | 0.0115 | 0.0326 | 0.0159 |
| GRU4Rec | 0.0129 | 0.0086 | 0.0204 | 0.0110 | 0.0164 | 0.0099 | 0.0283 | 0.0137 | 0.0176 | 0.0110 | 0.0285 | 0.0145 |
| BERT4Rec | 0.0115 | 0.0075 | 0.0191 | 0.0099 | 0.0203 | 0.0124 | 0.0347 | 0.0170 | 0.0051 | 0.0033 | 0.0090 | 0.0090 |
| FDSA | 0.0182 | 0.0122 | 0.0288 | 0.0156 | 0.0267 | 0.0163 | 0.0407 | 0.0208 | 0.0158 | 0.0098 | 0.0276 | 0.0136 |
| SASRec | 0.0233 | 0.0154 | 0.0350 | 0.0192 | 0.0387 | 0.0249 | 0.0605 | 0.0318 | 0.0170 | 0.0110 | 0.0284 | 0.0147 |
| S3-Rec | 0.0251 | 0.0161 | 0.0385 | 0.0204 | 0.0387 | 0.0244 | 0.0647 | 0.0327 | 0.0201 | 0.0123 | 0.0341 | 0.0168 |
| RID | 0.0208 | 0.0122 | 0.0288 | 0.0153 | 0.0213 | 0.0178 | 0.0479 | 0.0277 | 0.0225 | 0.0159 | 0.0329 | 0.0193 |
| TID | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0182 | 0.0132 | 0.0432 | 0.0254 | 0.0058 | 0.0040 | 0.0086 | 0.0049 |
| IID | 0.0268 | 0.0151 | 0.0386 | 0.0195 | 0.0394 | 0.0268 | 0.0615 | 0.0341 | 0.0232 | 0.0146 | 0.0393 | 0.0197 |
| SID | 0.0264 | 0.0186 | 0.0358 | 0.0216 | 0.0430 | 0.0288 | 0.0602 | 0.0368 | 0.0346 | 0.0242 | 0.0486 | 0.0287 |
| CID | 0.0313 | 0.0224 | 0.0431 | 0.0262 | 0.0489 | 0.0318 | 0.0680 | 0.0357 | 0.0261 | 0.0171 | 0.0428 | 0.0225 |
| SemID | 0.0274 | 0.0193 | 0.0406 | 0.0235 | 0.0433 | 0.0299 | 0.0652 | 0.0370 | 0.0202 | 0.0131 | 0.0324 | 0.0170 |
| SID+IID | 0.0235 | 0.0161 | 0.0339 | 0.0195 | 0.0420 | 0.0297 | 0.0603 | 0.0355 | 0.0329 | 0.0236 | 0.0465 | 0.0280 |
| CID+IID | 0.0321 | 0.0227 | 0.0456 | 0.0270 | 0.0512 | 0.0356 | 0.0732 | 0.0427 | 0.0287 | 0.0195 | 0.0468 | 0.0254 |
| SemID+IID | 0.0291 | 0.0196 | 0.0436 | 0.0242 | 0.0501 | 0.0344 | 0.0724 | 0.0411 | 0.0229 | 0.0150 | 0.0382 | 0.0199 |
| SemID+CID | 0.0043 | 0.0031 | 0.0070 | 0.0039 | 0.0355 | 0.0248 | 0.0545 | 0.0310 | 0.0021 | 0.0016 | 0.0056 | 0.0029 |

### 5.4 顺序索引的不同设置

表4显示，尽管SID本质上很简单，但它能生成接近或超越基线的良好结果。在第4.1节中，我们探讨了SID的构建及其局限性，具体而言，索引结果可能受用户顺序的影响。在本节中，我们展示使用四种不同用户顺序的SID结果，证实了这一说法，并建议最有效的排序方式：

**(1) 时间敏感排序（TSO，Time-Sensitive Ordering）**：用户按其与系统的首次交互在原始数据集中按时间顺序排列。后续交互被记录，新用户在首次与系统交互时创建新记录。通过按时间戳排序和处理交互，我们确保具有较早首次交互的用户先被记录。

**(2) 随机排序（RO，Random Ordering）**：用户随机排序。

**(3) 短到长排序（S2LO，Short-to-Long Ordering）**：用户按交互数量组织，从最少到最多交互按升序排列。

**(4) 长到短排序（L2SO，Long-to-Short Ordering）**：用户按交互数量从最多到最少交互按降序排列。

**表5：顺序索引的不同设置用于P5，在三个数据集上与两个基线的比较。粗体数字代表最佳结果，波浪下划线数字代表次佳结果。TSO在Amazon Beauty和Yelp上的结果经检验相对于其他设置显著。**

| 方法 | HR@5 | NCDG@5 | HR@10 | NCDG@10 | HR@5 | NCDG@5 | HR@10 | NCDG@10 | HR@5 | NCDG@5 | HR@10 | NCDG@10 |
|------|------|--------|-------|---------|------|--------|-------|---------|------|--------|-------|---------|
| | Sports | | | | Beauty | | | | Yelp | | | |
| SASRec | 0.0233 | 0.0154 | 0.0350 | 0.0192 | 0.0387 | 0.0249 | 0.0605 | 0.0318 | 0.0170 | 0.0110 | 0.0284 | 0.0147 |
| S3-Rec | 0.0251 | 0.0161 | 0.0385 | 0.0204 | 0.0387 | 0.0244 | 0.0647 | 0.0327 | 0.0201 | 0.0123 | 0.0341 | 0.0168 |
| SID-TSO | 0.0264 | 0.0186 | 0.0358 | 0.0216 | 0.0430 | 0.0288 | 0.0602 | 0.0368 | 0.0346 | 0.0242 | 0.0486 | 0.0287 |
| SID-RO | 0.0214 | 0.0150 | 0.0291 | 0.0175 | 0.0392 | 0.0257 | 0.0512 | 0.0335 | 0.0324 | 0.0219 | 0.0461 | 0.0263 |
| SID-S2LO | 0.0304 | 0.0230 | 0.0395 | 0.0259 | 0.0395 | 0.0259 | 0.0520 | 0.0337 | 0.0335 | 0.0237 | 0.0442 | 0.0277 |
| SID-L2SO | 0.0244 | 0.0176 | 0.0356 | 0.0209 | 0.0409 | 0.0286 | 0.0586 | 0.0343 | 0.0316 | 0.0215 | 0.0472 | 0.0265 |

表5展示了四种设置的性能。我们的观察表明，总体而言，相对性能排序为：时间敏感 $>$ {长到短，短到长} $>$ 随机。这些观察表明时间在顺序索引中起着重要作用：在相似时间交互的item，即使是不同用户交互的，也可能比在极不同时时间交互的item更相似。因此，在相似时间出现的item更可能被某些用户共同交互。因此，在对用户排序时使用时间相关信息可能提高性能。

基于这些观察，我们建议未来对简单SID方法的实现考虑使用时间敏感的用户排序策略来增强性能。注意，原始Amazon和Yelp数据集已经使用时间敏感排序来排列用户。因此，为使用SID生成索引，我们只需从第一个用户到最后一个用户递增地索引item。

### 5.5 协同索引的不同设置

CID涉及两个超参数：$N$ 和 $k$，其中 $N$ 是聚类每一层的聚类数，$k$ 是最终聚类中允许的最大item数。改变这些超参数会导致不同数量的独立额外token和不同的推荐性能。

**图4：CID在Beauty上关于 $N$（每层聚类数）和 $k$（最终聚类中允许的最大item数）的消融实验。**

在图4中，我们展示了Beauty数据集上各种 $N$ 和 $k$ 组合的hit@10结果。当 $k=50$ 时，性能低于4.5%，显著低于基线和某些基本索引方法。然而，当 $k$ 大于100时，性能显著提升。

**表6：不同参数和数据集下CID的hit@10结果。粗体数字为最佳结果，波浪下划线数字为次佳结果。所有数据集中得分最高的设置经配对Student's $t$ 检验在 $p < 0.05$ 下相对于其他设置显著。**

| 数据集 | Sports | | Beauty | | Yelp | |
|--------|--------|--------|--------|--------|--------|--------|
| | $N=10$ | $N=20$ | $N=10$ | $N=20$ | $N=10$ | $N=20$ |
| SASRec | 0.0350 | | 0.0605 | | 0.0284 | |
| S3-Rec | 0.0385 | | 0.0647 | | 0.0341 | |
| $k=200$ | 0.0302 | 0.0423 | 0.0566 | 0.0635 | 0.0416 | 0.0428 |
| $k=500$ | 0.0400 | 0.0431 | 0.0680 | 0.0668 | 0.0388 | 0.0403 |
| $k=1000$ | 0.0435 | 0.0416 | 0.0658 | 0.0638 | 0.0385 | 0.0388 |

基于我们的观察，可以得出以下结论：(1) 极小的 $k$ 值无论选择哪个 $N$ 都会导致次优性能。当 $k=50$ 时，性能低于基线。这可归因于少量新token的有限表达能力，无法充分捕捉item的多样性。(2) 不同的 $k$ 和 $N$ 组合产生不同的ID长度（即ID中的token数）。我们计算了每个 $k$ 和 $N$ 超参数设置的平均ID长度，结果如图5（Beauty）和表7（所有数据集）所示。结合图4和5，以及表6和7，我们发现最优推荐结果通常在平均ID长度在3到4之间时观察到。例如，图5中的方框点显示了Beauty数据集上所有平均ID长度在3到4之间的情况，我们可以看到这些点也对应于图4中每条线上的最优性能。类似地，表6中的最佳或次佳结果在大多数情况下也对应于表7中3到4的ID长度。

**图5：Beauty上的CID平均长度。**

**表7：不同参数下的平均ID长度。表中的粗体数字对应表6中的最佳结果（即表6中的粗体数字）。**

| 数据集 | Sports | | Beauty | | Yelp | |
|--------|--------|--------|--------|--------|--------|--------|
| | $N=10$ | $N=20$ | $N=10$ | $N=20$ | $N=10$ | $N=20$ |
| $k=200$ | 4.25 | 3.35 | 4.31 | 3.23 | 3.88 | 3.25 |
| $k=500$ | 3.66 | 3.66 | 3.80 | 2.94 | 3.57 | 2.91 |
| $k=1000$ | 3.31 | 2.78 | 3.54 | 3.54 | 3.21 | 2.76 |

基于这些观察，我们建议未来的CID实现使用生成平均ID长度在3到4之间的超参数。然而，值得注意的是，不同数据集可能需要略微不同的长度以获得最优性能。

### 5.6 语义索引何时有效

SemID使用元数据构建item索引。在我们的实验中，我们观察到如果类别遵循层次化树结构，则性能趋于提升。数据集中的类别信息通常不是树结构，因为在某些情况下，一个类别名称可能出现在不同的父类别下，这使得类别成为图而非树。表8是Amazon Beauty中的两个例子，其中类别"Eyes"同时出现在"Skin Care"和"Makeup Remover"下，类别"Creams"同时出现在"Skin Care"和"Moisturizers"下。

**表8：Amazon Beauty数据集中非树结构类别的示例。**

| 类别路径 |
|---------|
| Beauty > Skin Care > Eyes > Combinations |
| Beauty > Skin Care > Eyes > Creams |
| Beauty > Makeup > Makeup Remover > Eyes |
| Beauty > Makeup > Body > Moisturizers > Creams |

为测试类别中的树结构是否关键，我们在实验中比较了两种不同设置：

**(1) 非树结构设置**：我们直接使用类别名称创建相应的独立OOV额外token。例如，一个属于"Beauty"、"Skin Care"、"Eyes"的item和另一个属于"Beauty"、"Makeup"、"Makeup Remover"、"Eyes"的item将共享token $\langle Eyes \rangle$。

**(2) 树结构设置**：通过在相同类别名称出现在不同位置时创建不同的OOV token，我们在类别上强制执行树结构。例如，"Beauty"、"Skin Care"下的类别"Eyes"将对应token $\langle Eyes1 \rangle$，而"Beauty"、"Makeup"、"Makeup Remover"下的对应 $\langle Eyes2 \rangle$。

**表9：不同设置下的SemID结果。粗体数字为最佳结果，波浪下划线数字为次佳结果。Amazon Beauty和Yelp上的树结构设置结果经检验相对于非树结构设置显著。**

| 方法 | HR@5 | NCDG@5 | HR@10 | NCDG@10 | HR@5 | NCDG@5 | HR@10 | NCDG@10 | HR@5 | NCDG@5 | HR@10 | NCDG@10 |
|------|------|--------|-------|---------|------|--------|-------|---------|------|--------|-------|---------|
| | Sports | | | | Beauty | | | | Yelp | | | |
| SASRec | 0.0233 | 0.0154 | 0.0350 | 0.0192 | 0.0387 | 0.0249 | 0.0605 | 0.0318 | 0.0170 | 0.0110 | 0.0284 | 0.0147 |
| S3-Rec | 0.0251 | 0.0161 | 0.0385 | 0.0204 | 0.0387 | 0.0244 | 0.0647 | 0.0327 | 0.0201 | 0.0123 | 0.0341 | 0.0168 |
| SemID-non-tree | 0.0281 | 0.0192 | 0.0410 | 0.0233 | 0.0423 | 0.0288 | 0.0632 | 0.0354 | 0.0028 | 0.0019 | 0.0050 | 0.0025 |
| SemID-tree | 0.0274 | 0.0193 | 0.0406 | 0.0235 | 0.0433 | 0.0299 | 0.0652 | 0.0370 | 0.0202 | 0.0131 | 0.0324 | 0.0170 |

表9说明了层次化信息对SemID有效性的重要性。类别越遵循层次化结构，模型的性能越好。这可能是因为层次化组织的类别列表有助于减少生成过程中的搜索空间。因此，这一发现突出了在推荐基础模型中实现SemID时适当组织和结构化类别信息的重要性。

### 5.7 哪些HID有效及其原因

基于表4中的结果，CID+IID和SemID+IID表现出远优于其各自CID和SemID的性能。但SID+IID并未改善SID，而SemID+CID不仅没有改善反而大幅降低了性能。CID+IID和SemID+IID通过为每个item分配独立额外token并将其拼接在聚类ID或类别ID序列之后来构建。这些组合保持了原始索引长度同时保留了层次化结构。性能的提升可归因于额外token提供的索引表达能力的增强，以及混合索引中协同信息或元数据信息的保留。这些因素的结合促成了CID+IID和SemID+IID方法观察到的性能提升。

SID+IID通过在原始顺序索引后追加独立额外token创建，将ID长度增加了1。SID+IID未能提升性能，可能是因为额外token干扰了原始顺序索引中以数字风格编码的时间敏感信息。SemID+CID通过拼接类别ID和聚类ID（或反之）创建，表现出次优性能，如表4所示。这对两种拼接顺序都成立：类别ID后跟CID索引以及聚类ID后跟SemID索引。这次优性能的原因是它生成了过长的索引，并破坏了SemID和CID中编码的层次化结构。

基于我们的发现，我们建议在推荐基础模型中采用CID+IID和SemID+IID作为混合索引，因为它们在此类场景中表现出了卓越的性能。

## 6 结论

本文以P5为示例骨干模型研究了各种索引方法。我们研究了三种简单索引方法：随机索引（RID）、标题索引（TID）和独立索引（IID），并强调了它们的局限性。这突出了为推荐基础模型选择合适索引方法的重要性，因为它极大地影响模型性能。随后我们研究了四种简单有效的索引方法：顺序索引（SID）、协同索引（CID）、语义索引（SemID）和混合索引（HID）。在Amazon Sports、Amazon Beauty和Yelp数据集上的实验结果展示了它们的强大性能。这四种有效索引方法满足了本文提出的两个标准：(1) 保持合适的ID长度；(2) 将有用的先验信息整合到item ID构建中。我们希望本研究能为推荐基础模型及更广泛领域的未来索引方法研究提供启发。

致谢：本工作部分由NSF IIS-2046457和IIS-2007907资助。

## 参考文献

[1] Gediminas Adomavicius and Alexander Tuzhilin. 2005. Toward the next generation of recommender systems: A survey of the state-of-the-art and possible extensions. ICDE 17, 6 (2005), 734–749.

[2] Hanxiong Chen, Shaoyun Shi, Yunqi Li, and Yongfeng Zhang. 2021. Neural collaborative reasoning. In Proceedings of the Web Conference 2021. 1516–1527.

[3] Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. 2022. Palm: Scaling language modeling with pathways. arXiv preprint arXiv:2204.02311 (2022).

[4] Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Eric Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, et al. 2022. Scaling instruction-finetuned language models. arXiv preprint arXiv:2210.11416 (2022).

[5] Paul Covington, Jay Adams, and Emre Sargin. 2016. Deep neural networks for youtube recommendations. In Proceedings of the 10th ACM conference on recommender systems. 191–198.

[6] Zeyu Cui, Jianxin Ma, Chang Zhou, Jingren Zhou, and Hongxia Yang. 2022. M6-Rec: Generative Pretrained Language Models are Open-Ended Recommender Systems. arXiv preprint arXiv:2205.08084 (2022).

[7] Nicola De Cao, Gautier Izacard, Sebastian Riedel, and Fabio Petroni. 2020. Autoregressive entity retrieval. arXiv preprint arXiv:2010.00904 (2020).

[8] Owain Evans, Owen Cotton-Barratt, Lukas Finnveden, Adam Bales, Avital Balwit, Peter Wills, Luca Righetti, and William Saunders. 2021. Truthful AI: Developing and governing AI that does not lie. arXiv preprint arXiv:2110.06674 (2021).

[9] Shijie Geng, Shuchang Liu, Zuohui Fu, Yingqiang Ge, and Yongfeng Zhang. 2022. Recommendation as language processing (rlp): A unified pretrain, personalized prompt & predict paradigm (p5). In RecSys. 299–315.

[10] F Maxwell Harper and Joseph A Konstan. 2015. The movielens datasets: History and context. ACM TIIS 5, 4 (2015), 1–19.

[11] Jie Huang and Kevin Chen-Chuan Chang. 2022. Towards Reasoning in Large Language Models: A Survey. arXiv preprint arXiv:2212.10403 (2022).

[12] Dietmar Jannach and Malte Ludewig. 2017. When recurrent neural networks meet the neighborhood for session-based recommendation. In RecSys. 306–310.

[13] Dietmar Jannach, Markus Zanker, Alexander Felfernig, and Gerhard Friedrich. 2010. Recommender systems: an introduction. Cambridge University Press.

[14] Wang-Cheng Kang and Julian McAuley. 2018. Self-attentive sequential recommendation. In ICDM. IEEE, 197–206.

[15] Yehuda Koren, Robert Bell, and Chris Volinsky. 2009. Matrix factorization techniques for recommender systems. Computer 42, 8 (2009), 30–37.

[16] Jure Leskovec, Anand Rajaraman, and Jeffrey David Ullman. 2020. Mining of massive data sets. Cambridge university press.

[17] Lei Li, Yongfeng Zhang, Dugang Liu, and Li Chen. 2023. Large Language Models for Generative Recommendation: A Survey and Visionary Discussions. arXiv preprint arXiv:2309.01157 (2023).

[18] Stephanie Lin, Jacob Hilton, and Owain Evans. 2021. Truthfulqa: Measuring how models mimic human falsehoods. arXiv preprint arXiv:2109.07958 (2021).

[19] Xudong Lin, Fabio Petroni, Gedas Bertasius, Marcus Rohrbach, Shih-Fu Chang, and Lorenzo Torresani. 2022. Learning to recognize procedural activities with distant supervision. In CVPR. 13853–13863.

[20] Chen Ma, Peng Kang, and Xue Liu. 2019. Hierarchical gating networks for sequential recommendation. In Proceedings of the 25th ACM SIGKDD international conference on knowledge discovery & data mining. 825–833.

[21] Andrew Ng, Michael Jordan, and Yair Weiss. 2001. On spectral clustering: Analysis and an algorithm. Advances in neural information processing systems 14 (2001).

[22] Jianmo Ni, Jiacheng Li, and Julian McAuley. 2019. Justifying recommendations using distantly-labeled reviews and fine-grained aspects. In EMNLP-IJCNLP.

[23] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. JMLR 21, 1 (2020).

[24] Rico Sennrich, Barry Haddow, and Alexandra Birch. 2016. Neural Machine Translation of Rare Words with Subword Units. In ACL. 1715–1725.

[25] Fei Sun, Jun Liu, Jian Wu, Changhua Pei, Xiao Lin, Wenwu Ou, and Peng Jiang. 2019. BERT4Rec: Sequential recommendation with bidirectional encoder representations from transformer. In Proceedings of the 28th ACM international conference on information and knowledge management. 1441–1450.

[26] Jiaxi Tang and Ke Wang. 2018. Personalized top-n sequential recommendation via convolutional sequence embedding. In Proceedings of the eleventh ACM international conference on web search and data mining. 565–573.

[27] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. 2023. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971 (2023).

[28] Ulrike Von Luxburg. 2007. A tutorial on spectral clustering. Statistics and computing 17 (2007), 395–416.

[29] Hong-Jian Xue, Xinyu Dai, Jianbing Zhang, Shujian Huang, and Jiajun Chen. 2017. Deep matrix factorization models for recommender systems. In IJCAI, Vol. 17. Melbourne, Australia, 3203–3209.

[30] Shuai Zhang, Lina Yao, Aixin Sun, and Yi Tay. 2019. Deep learning based recommender system: A survey and new perspectives. ACM Computing Surveys (CSUR) 52, 1 (2019), 1–38.

[31] Tingting Zhang, Pengpeng Zhao, Yanchi Liu, Victor S Sheng, Jiajie Xu, Deqing Wang, Guanfeng Liu, Xiaofang Zhou, et al. 2019. Feature-level Deeper Self-Attention Network for Sequential Recommendation. In IJCAI. 4320–4326.

[32] Yongfeng Zhang, Qingyao Ai, Xu Chen, and W Bruce Croft. 2017. Joint representation learning for top-n recommendation with heterogeneous information sources. In Proceedings of the 2017 ACM on Conference on Information and Knowledge Management. 1449–1458.

[33] Yuhui Zhang, Hao Ding, Zeren Shui, Yifei Ma, James Zou, Anoop Deoras, and Hao Wang. 2021. Language models as recommender systems: Evaluations and limitations. (2021).

[34] Wayne Xin Zhao, Junhua Chen, Pengfei Wang, Qi Gu, and Ji-Rong Wen. 2020. Revisiting alternative experimental settings for evaluating top-n item recommendation algorithms. In Proceedings of the 29th ACM International Conference on Information & Knowledge Management. 2329–2332.

[35] Kun Zhou, Hui Wang, Wayne Xin Zhao, Yutao Zhu, Sirui Wang, Fuzheng Zhang, Zhongyuan Wang, and Ji-Rong Wen. 2020. S3-rec: Self-supervised learning for sequential recommendation with mutual information maximization. In CIKM.

[36] Han Zhu, Xiang Li, Pengye Zhang, Guozheng Li, Jie He, Han Li, and Kun Gai. 2018. Learning tree-based deep model for recommender systems. In KDD.

¹ https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/

² https://www.yelp.com/dataset
