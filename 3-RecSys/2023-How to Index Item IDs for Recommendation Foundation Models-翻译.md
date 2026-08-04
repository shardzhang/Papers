# 如何为推荐基础模型索引itemID

> Wenyue Hua, Shuyuan Xu, Yingqiang Ge, Yongfeng Zhang | 罗格斯大学

本文介绍了针对推荐基础模型中itemID创建与索引问题的系统性研究，以P5作为骨干LLM的示例。核心内容：

- 讨论了几种平凡item索引方法（随机索引RID、标题索引TID、独立索引IID）的问题及其局限性
- 提出了四种简单有效的非平凡索引方法：顺序索引（SID）、协同索引（CID）、语义索引（SemID）和混合索引（HID）
- 在Amazon Sports、Amazon Beauty和Yelp三个真实世界数据集上验证了所提方法的有效性

关键发现：

- item索引方法对基于LLM的推荐性能有显著影响，平凡方法（RID、TID）甚至可能损害性能
- 四种非平凡索引方法均满足两个标准：保持合适的ID长度，并将先验信息整合到itemID构建中
- CID+IID和SemID+IID作为混合索引展现出最佳性能，显著超越基线

---

## 摘要

推荐基础模型利用大语言模型（LLM）进行推荐，通过将推荐任务转换为自然语言任务。它实现了生成式推荐，即直接生成要推荐的item，而不是像传统推荐模型那样为每个候选item计算排名分数，从而将推荐流程从多阶段过滤简化为单阶段过滤。为了避免在决定推荐哪些item时生成过长的文本和幻觉推荐，创建LLM兼容的itemID以唯一标识每个item对于推荐基础模型至关重要。在本研究中，我们系统性地研究了推荐基础模型的itemID创建和索引问题，以P5作为骨干LLM的示例。为了强调item索引的重要性，我们首先讨论了几种平凡item索引方法的问题，如随机索引、标题索引和独立索引。然后我们提出了四种简单而有效的解决方案，包括顺序索引、协同索引、语义（基于内容的）索引和混合索引。我们的研究突显了item索引方法对基于LLM的推荐性能的显著影响，我们在真实世界数据集上的结果验证了我们提出的解决方案的有效性。该研究还展示了语言建模和传统IR原则（如索引）方面的最新进展如何相互促进，以实现更好的学习和推理。源代码和数据可在 https://github.com/Wenyueh/LLM-RecSys-ID 获取。

## CCS概念
• 信息系统 \rightarrow 推荐系统；• 计算方法论 \rightarrow 机器学习；自然语言处理。

## 关键词
大语言模型；推荐；itemID与索引

ACM引用格式：
Wenyue Hua, Shuyuan Xu, Yingqiang Ge, and Yongfeng Zhang. 2023. 如何为推荐基础模型索引itemID. 载于Annual International ACM SIGIR Conference on Research and Development in Information Retrieval in the Asia Pacific Region (SIGIR-AP '23), 2023年11月26–28日, 中国北京. ACM, 纽约, 美国, 10页. https://doi.org/10.1145/3624918.3625339

允许为个人或课堂使用免费制作本作品的全部或部分数字或硬拷贝，前提是复制品不以营利或商业利益为目的分发，并且复制品在第一页上带有此声明和完整引用。本作品中由作者以外的他人拥有的组件的版权必须得到尊重。允许注明来源的摘要。如需以其他方式复制、重新发布、在服务器上发布或分发给列表，需要事先获得特定许可和/或支付费用。请向permissions@acm.org请求许可。

SIGIR-AP '23, 2023年11月26–28日, 中国北京
© 2023 版权归作者/所有者所有。出版权已授予ACM。
ACM ISBN 979-8-4007-0408-6/23/11. . . $15.00$
https://doi.org/10.1145/3624918.3625339

1 引言

基础模型如大语言模型（LLM）[3, 4, 27]显著影响了自然语言处理（NLP）和计算机视觉（CV）[19]等研究领域，并已被应用于各种推荐系统（RS）任务。最近的研究如P5 [9]和M6Rec [6]利用了预训练LLM在推荐[17]方面的优势：它们将丰富的用户行为和知识信息纳入预训练，并受益于基础模型在推荐方面的强大学习能力。预训练的LLM还具有改进的推理能力[11]，能够根据上下文推断用户兴趣。因此，这些模型旨在通过将推荐任务转换为语言生成任务，利用在大量自然语言语料库上预训练的LLM进行RS，从而实现生成式推荐。

由于item描述可能包含大量词汇（例如，一个产品标题/描述可能包含数十/数百个词，一篇新闻文章可能包含数千个词），我们很难期望LLM在决定推荐哪些item时生成完整且精确的item描述，因为生成的文本可能甚至不对应item数据库中真实存在的item，导致基于LLM的推荐中出现幻觉问题[8, 18]。因此，为每个item分配一个唯一ID非常重要，这样每个item由少量特征性token表示，同时彼此可区分。例如，Yelp中的一个商家位置可能被分配ID"location_4332"，并进一步表示为诸如\langle location\rangle \langle _\rangle \langle 43\rangle \langle 32\rangle [9]的token序列。注意，itemID不一定必须是数字token，只要它是item的唯一标识符，就可以被视为该item的ID。例如，电影"指环王"的标题可以被视为该电影的ID，它由一系列单词token而非数字token组成。ID甚至可以是一串不表达明确含义的单词，例如"ring epic journey fellowship adventure"。

然而，为item分配LLM兼容的ID并非易事。首先，可能存在大量甚至无限的item，而每个item都应分配一个唯一ID，以便基础模型能够区分不同item。其次，itemID应与自然语言兼容，以便ID可以集成到自然语言指令中，用于LLM的预训练、微调和提示。第三，平凡的item索引方法如随机索引可能无助于甚至损害推荐基础模型，因为它们可能错误地将相关ID分配给不相关的item，误导LLM的训练和提示。因此，需要对面向LLM的item索引进行全面研究，以实现推荐任务与LLM的无缝适配，发挥LLM在推荐方面的潜力。

此外，确保生成的文本与真实item对齐以避免幻觉问题的一个自然思路是采用约束解码方法[7]。然而，对自由形式的长文本使用约束生成是不切实际的。这是因为约束解码本质上规定了表达内容的单一模式，否定了长文本叙事的灵活性。通过强制模型遵循特定的描述模式，模型除了要记忆推荐特定知识外，还需要记忆僵化的文本模式。这种额外的复杂性可能会淡化模型的主要目的，并阻碍其执行核心推荐任务的效能。

受上述原因驱动，本文聚焦于基于LLM的推荐器的item索引问题：如何为每个item分配一个唯一ID（即token序列）。我们基于P5 [9]（一个代表性的RS LLM模型）研究该问题。P5在基础模型上进行预训练，并基于个性化提示将推荐任务转换为自然语言句子。我们首先实验了三种平凡索引方法并展示了它们的局限性，其中一些方法在以前的模型中使用过：独立索引（IID）、标题索引（TID）和随机索引（RID）。基于分析，我们进一步探索了四种新颖的索引技术：顺序索引（SID）、协同索引（CID）、语义（基于内容的）索引（SemID）和混合索引（HID）。为了确保在推荐阶段生成的ID与真实item对齐以避免幻觉，我们开发了一种约束解码方法[7]，该方法通过从有效ID集合构建前缀树（trie）并在解码阶段将不存在的ID的生成概率设为零来实现。我们展示了各种ID方法在三个广泛使用的数据集（Amazon Beauty、Amazon Sports、Yelp）上的性能，并提供了关于不同方法对基于LLM的推荐模型性能的见解。

2 相关工作

许多传统推荐模型使用基于匹配的范式[1, 2, 13, 15, 29, 30, 32]。它们将用户（或用户行为历史）和item投影到一个共享的嵌入空间中，然后通过使用它们的嵌入向量计算排名分数来估计用户对item的偏好，例如矩阵分解中用户与item向量之间的内积[15]。通常，这涉及为每个候选item计算排名分数，当item池很大时，匹配和排序过程非常耗时[34]。因此，工业RS通常不得不使用多阶段（通常是两阶段）过滤流水线[5]，其中早期阶段使用简单高效的过滤方法（如基于规则的过滤方法），而后期阶段在候选item较少时使用高级过滤方法。因此，最先进的模型仅应用于一小部分item。

最近，有多次尝试预训练基础模型用于生成式推荐，这省去了昂贵的一对一候选item匹配过程，而是直接生成要推荐的item。例如，P5 [9]将多种推荐任务统一为序列到序列生成框架内的自然语言生成任务。推荐数据如用户-item交互、用户描述、item元数据和用户评论通过使用多个个性化提示模板被转换为通用格式——自然语言序列。每个用户或item由一个唯一的token序列表示作为用户或itemID。M6Rec [6]将各种推荐任务（如内容供应、分发和呈现）转换为自然语言理解或生成任务。输入提示包含用户属性、过去行为以及卖家提供的详细item描述。用户和item由其属性和描述的预计算嵌入表示。LMRecSys [33]将基于item的推荐任务转换为基于文本的完形填空任务。该模型在MovieLens-1M数据集[10]上测试，该数据集包含预训练LLM可能在网络文本中见过的电影。item由其标题表示，这些标题充当索引。正如原始论文所报告的，这种索引方法对模型性能产生了负面影响：LLM不仅无法有效推断多token跨度的概率分布，而且标题中包含的语言偏见可能会误导模型，因为标题可能包含很少关于电影内容的信息。

这三个模型使用不同的方法来索引item：P5使用数字token，M6Rec使用基于元数据的嵌入，LMRecSys使用item标题。本文以P5为示例骨干，在基于LLM的生成式推荐框架下研究不同的item索引方法，比较了不同索引方法的有效性，揭示了item索引与基础模型预训练之间的关系，并就哪种item索引方法最适合预训练推荐基础模型提供了见解。

3 预备知识与前期研究

3.1 P5范式简介

本文基于P5 [9]研究索引问题。P5是一个代表性的推荐基础模型，它通过整合各种任务和个性化指令提示来预训练用于推荐的基座模型，从而增强现有推荐系统的泛化能力。这些任务包括顺序推荐、评分预测、解释生成、评论摘要和直接推荐。P5使用从一组提示模板生成的输入-目标文本对进行训练，这些模板具有针对不同用户和item的个性化字段：顺序推荐的示例输入提示可以是用户-item交互的描述，如"根据user_1访问过的地方：location_1123, location_4332, location_8463, location_12312，你能为用户推荐另一个地方吗？"，输出文本是下一个生成的item，如"Output: location_1934"。在本研究中，我们聚焦于顺序推荐任务，因为它显式依赖于输入提示中呈现的item交互，使其对不同索引方法高度敏感。

3.2 尖括号表示法

在本文中，我们需要引入词汇外（OOV）token来在某些索引方法中构建item索引，这些token不属于语言模型正常词汇的一部分。在我们的案例中，它们是不存在于默认T5词汇表[23]中的token。为了区分新创建的OOV token和已有token，我们使用尖括号"\langle \rangle "表示新创建的OOV token，并使用不带"\langle \rangle "的文本表示默认分词器中的已有token。所有OOV token在模型中随机初始化，因此"\langle \rangle "内的文本不会影响OOV token的嵌入。尖括号"\langle \rangle "内的文本可以是单词或数字，但无论哪种情况，尖括号内的文本仅用于区分不同的OOV token，与已有token无关。例如，\langle restaurant\rangle  \langle Greek\rangle  \langle 2\rangle 是Yelp中一个item的索引，由三个OOV token组成，其中\langle restaurant\rangle 与普通英文单词"restaurant"是不同的token，\langle 2\rangle 与数字"2"是不同的token。当我们需要使用已有的普通单词token时，我们将不带尖括号使用它们，如"restaurant"和"2"。

3.3 数据格式与预处理

实验在Amazon Sports & Outdoors、Amazon Beauty和Yelp数据集上进行。Amazon数据集[22]¹源自Amazon.com用于产品推荐，而Yelp数据集²提供了用于商业推荐的用户评分和评论集合。我们使用2019年1月1日至2019年12月31日的交易记录，与原始P5论文[9]相同。这些数据集的详细统计数据见表1。

表1：数据集的基本统计

 Sports Beauty Yelp
用户数 35,598 22,363 30,431
item数 18,357 12,101 20,033
交互数 296,337 198,502 316,354
稀疏度(%) 0.0453 0.0734 0.0519

这些数据集按单个用户组织用户-item交互。我们使用常用的留一法设置将数据集分为训练集、验证集和测试集：对于每个用户的交互序列，我们将倒数第二个item放入验证集，将最后一个item放入测试集，并将序列中的所有其他item放入训练集。例如，假设用户 $i$ 的交互序列是 $\{item_{i,1}, item_{i,2}, item_{i,3}, \cdots, item_{i,k-1}, item_{i,k}\}$ 。那么基于序列 $\{item_{i,1}, item_{i,2}, item_{i,3}, \cdots, item_{i,k-2}\}$ 对 $item_{i,k-1}$ 的预测用于验证，基于序列 $\{item_{i,1}, item_{i,2}, item_{i,3}, \cdots, item_{i,k-1}\}$ 对 $item_{i,k}$ 的预测用于测试。

3.4 item索引的动机分析

我们从三种平凡索引方法开始探讨索引方法的动机：

• 随机索引（RID）：为每个item分配一个随机数字作为itemID。该数字进一步基于SentencePiece分词器[24]被分词为子token序列，如P5 [9]中所做。例如，一个Yelpitem被随机分配数字"4332"，"4332"被表示为token序列"43""32"。

• 标题索引（TID）：使用item标题来表示item，同样由SentencePiece[24]分词。例如，Yelpitem"Las Vegas Cigar Outlet"被表示为token序列"Las""Vegas""Ci""gar""Outlet"。

• 独立索引（IID）：为每个item创建一个需要学习的独立OOV额外token。例如，一个Yelpitem被表示为\langle IID5\rangle ，这是一个专门为此item分配的独立额外token。在本文的其余部分，为IID创建的token始终以字母"IID"开头。

RID生成随机的数字索引，导致分词后无关item之间可能出现重叠。例如，两个item"4332"和"4389"将分别被分词为"43""32"和"43""89"，这意味着即使这两个item可能完全不相关，它们也总是共享相同的子token"43"。这种意外重叠可能在item之间建立任意关系，给模型训练引入不必要的偏差。由于重叠源于索引结构，无论模型如何从数据中学习，都无法消除。因此，RID被认为是一种不利的方法。

TID使任务更具挑战性，因为模型需要记忆和生成冗长的item标题。此外，标题中的某些词语或表达可能与item的真实内容无关，而且非常不同的item可能在其标题中共享重叠的token，因此从标题中推导出的语义可能引入强烈的语言偏见[33]。例如，电影"指环王"和"战争之王"在其标题中共享许多token（"the"、"lord"、"of"），但它们是两部非常不同的电影：前者是史诗奇幻，而后者是犯罪剧情片。一般来说，两个无关的item可能有非常相似的标题，例如作为水果的Apple和作为公司的Apple，而两个密切相关的item可能有非常不同的标题，如数据挖掘中经典的"啤酒与尿布"例子[17]。因此，使用标题作为ID可能会将误导性的语义编码到生成过程中，类似于随机索引的问题。

IID为item使用单token索引，不假设关于item的任何先验信息，使得语言模型更容易学习item表示。虽然优于RID和TID，但由于在创建itemID时将所有item视为相互独立，其性能仍然有限。如果需要创建大量新token，还可能导致过长的训练时间。

上述分析表明这三种方法都不是最优的。为了验证这一点，我们提供实验结果来展示它们的次优性能。我们针对两个强且广泛使用的基线SASRec [14]和S3-Rec [35]评估了这三种索引方法。结果如表2所示，其中每个指标的最佳结果以粗体突出显示，次优结果以波浪线下划线标出。基于表2，RID和TID的表现不如基线，而IID以引入更多可学习token为代价提供了微小的收益，因为每个item被视为一个独立的新token。因此，这些索引方法被认为是次优的，我们将在下一节进一步探索非平凡索引方法。

表2：P5的平凡索引方法以及基线的性能。粗体数字代表最佳结果，波浪线数字代表次优结果。RID和TID的结果在Sports和Beauty上显著更差，在配对Student t检验下 $p$ 值 < 0.05。

4 非平凡索引方法

基于上述分析，最优的item索引方法应满足两个标准以实现有效的学习过程：

（1）保持合适的长度以减轻文本生成的难度。

（2）将先验信息整合到item索引结构中，确保相似item共享最大数量的token同时保持可区分，而不相似item共享最少数量的token。

为实现这些目标，我们引入并探索四种复杂度递增的索引方法：顺序索引（SID）、协同索引（CID）、语义（基于内容的）索引（SemID）和混合索引（HID）。SID和CID利用协同信息，使共现item能够共享token。SemID使用自然语言中的元数据，使语义相似的item能够共享token。HID结合多种索引方法，试图利用每种方法的优势以生成最优索引。在以下小节中，我们将提供四种索引方法的详细信息。

4.1 顺序索引

顺序索引是一种利用协同信息进行item索引的直接方法。由同一用户连续交互的item被分配连续的数字索引，反映它们的共现性。以表3为例，item从第一个用户开始一直到最后一个用户被连续分配ID。如果一个item在之前用户的交互序列中已经被索引，例如User 2序列中的item1001（以及表中所有其他方框中的item），则使用该item已分配的ID，否则将创建并分配一个递增的新ID。注意，item索引过程仅依赖于训练序列，而验证和测试item不参与索引过程。索引过程完成后，验证和测试item被分配索引过程中已建立的相应ID。基于SentencePiece分词器[24]进行分词后，像"1001"这样的ID将被分词为"100""1"，而"1002"将被分词为"100""2"，导致这两个连续item共享token"100"。这为我们提供了至少在一位用户序列中共现的item之间的编码相似性。因此，这种简单的顺序索引方法能够在某些情况下捕获协同信息。

一个小提示是我们从1001开始item索引枚举。我们从1001而不是1开始有两个原因：1）SentencePiece分词器不会将一些小于1000的数字分词为多个子token，例如数字12，因此分配这些小数字的item将彼此完全独立；2）分词后，较小的数字可能成为较大分词数字的完整子集，例如ID"12"可以是ID"12""34"的子集，这可能导致item之间的虚假相关性。

尽管如此，顺序索引也有局限性：1）并非由同一用户一起交互的相邻索引item可能错误地共享token；例如，User 2的最后一个item被索引为1014（分词为"10""14"），User 3的第一个item被索引为1015（分词为"10""15"），那么token"10"将在两个item之间共享，尽管它们之间没有共现；2）它无法捕获基于共现频率的相似性；例如，假设item1001和1002共现一次，而item1002和1003共现十次，这两对仍然只共享一个token，无法传递频率信息；3）训练数据中的用户顺序影响结果；例如，如果我们交换表3中User 1和User 2的行，那么索引结果将不同。尽管顺序索引有其缺点，它仍然可以产生相对较好的结果，接近甚至超越基线。

表3：顺序索引方法图示。方框中的数字表示先前已索引的item。

4.2 协同索引

顺序索引是将协同信息整合到item索引中的初步方法。为了有效捕获协同过滤的本质，我们探索了协同索引（CID）方法，该方法基于谱矩阵分解（SMF）[21, 28]使用谱聚类来生成item索引。该方法基于这样的前提：共现更频繁的item更相似，在索引构建中应共享更多重叠的token。核心概念包括基于训练数据集为所有item构建共现图，并使用谱聚类将item分组为聚类，确保同一聚类内的item在构建索引时共享token。

4.2.1 基于谱矩阵分解的谱聚类

具体来说，我们基于训练集创建一个图，如图1(a)所示：每个item作为一个节点，两个item之间的边表示它们的共现（即两个item出现在用户的交互序列中），边权重表示共现的频率（即两个item共同出现的用户交互序列数量）。与该图对应的邻接矩阵（图1(b)）表示item在共现频率方面的相似性，与该图对应的拉普拉斯矩阵（图1(c)）可以被分解以实现谱聚类[21, 28]。谱聚类过程将item分组为聚类，使得共享更多共现相似性的item被分到同一聚类；每个聚类可以通过在聚类内递归应用谱聚类过程进一步分组为更细粒度的聚类，从而产生层次级别的聚类，如图1(a)所示。

图1：基于谱矩阵分解的item共现图谱聚类示意图

更具体地说，谱聚类利用拉普拉斯矩阵的特征向量将节点分组为聚类[21, 28]。它确保同一聚类内的item具有更高的相似度，而不同聚类中的item表现出较低的相似度。我们使用Python scikit-learn包中的标准谱聚类实现³。我们不扩展太多谱聚类算法的细节，因为它被认为是数据分析的教科书级算法[16]。然而，我们确实想讨论用于控制递归聚类过程的两个重要参数：1） $N$ ：我们在每个聚类级别将item划分为 $N$ 个聚类；2） $k$ ：最终聚类中允许的最大item数量，作为递归聚类过程的停止准则，即当一个聚类包含最多 $k$ 个item时，我们不再进一步缩小其规模。最后，聚类结果可以表述为层次树结构，如图2所示。在此图中，每个非叶节点（图中的大黄色节点）表示相应级别创建的聚类，每个叶节点（小蓝色节点）表示相应最终聚类中的item。在下一小节中，我们将介绍如何基于层次树结构创建itemID。

4.2.2 基于谱聚类树的item索引

如上所述，递归聚类过程为聚类和item生成树结构，如图2所示，以 $N$ =4和 $k$ =20为例，这意味着每次谱聚类迭代将item划分为4个聚类，该过程在每个聚类上递归应用，直到聚类大小小于或等于20。每个非叶节点（大黄色节点）表示一个聚类，而所有item作为叶节点（小蓝色节点）出现在最终聚类下。注意，由于最终聚类中允许的最大item数量为 $k$ ，这意味着我们最多只需要 $k$ 个独立额外token来区分同一最终聚类内的item（即同一黄色节点下的小蓝色节点最多为 $k$ 个）。因此，我们引入 $k$ 个独立额外token到词汇表中，记为\langle 0\rangle , \langle 1\rangle , \langle 2\rangle , $\cdots$ , \langle $k$ −1\rangle 。我们首先为非叶节点分配token。非叶节点在整个树中逐层枚举，使用从\langle 0\rangle 到\langle $k$ −1\rangle 的 $k$ 个独立token，如图2所示。一旦所有 $k$ 个token都用完，我们只需从\langle 0\rangle 重新开始。如前所述，每个父聚类节点有 $N$ 个子聚类节点。然而，如果 $N$ > $k$ ，那么我们将没有足够的token来区分同一父节点下的不同子节点。因此，我们要求 $N$ \leq $k$ 用于协同索引。结合逐层token分配过程，这可以保证同一父节点下的不同子节点被分配不同的token。

图2：基于谱聚类树的协同索引（ $N$ =4, $k$ =20）

然后我们为叶节点（小蓝色节点）分配token，其中每个叶节点是一个item。这相当直接：对于每个最终聚类，我们为其每个子item节点分配一个独立额外token，从\langle 0\rangle 开始依次进行。由于聚类过程确保每个最终聚类最多包含 $k$ 个item，因此 $k$ 个独立额外token足以区分同一最终聚类下的不同item。

最后，一个item的ID是其非叶祖先节点token及其自身叶节点token的拼接。例如，图2中加粗路径下的item被索引为\langle 1\rangle \langle 9\rangle \langle 5\rangle \langle 4\rangle 。这种索引过程保证同一最终聚类内的任意两个item将在最终聚类内各自的token之前共享token，这意味着两个item共现越频繁，它们共享的token越多，从而很好地利用了用户行为序列中隐藏的协同信息。

4.3 语义（基于内容的）索引

语义（基于内容的）索引（SemID）利用item元数据为item构建ID。如图3所示，item的类别形成层次结构[36]，每个非叶节点（大黄色节点）表示一个类别，每个叶节点（小蓝色节点）表示一个item。每个非叶节点被分配一个独立额外token，每个叶节点在其父节点下接收一个唯一额外token。为了创建item索引，非叶节点和叶节点的token沿着从根到叶的路径拼接。以图3中加粗路径为例，该item的类别从粗粒度到细粒度为\langle Makeup\rangle 、\langle Lips\rangle 、\langle Lip_Liners\rangle ，其叶节点token为\langle 5\rangle ，这将该item与Lip Liners类别下的其他item区分开来，则该item将被索引为\langle Makeup\rangle \langle Lips\rangle \langle Lip_Liners\rangle \langle 5\rangle 。

图3：语义索引示例

4.4 混合索引

混合索引（HID）不是一种单一的特定索引方法，而是一类方法。它将上述介绍的多个索引拼接为一个索引，如SID+IID、CID+IID、SemID+IID、SemID+CID等。这种方法旨在利用不同索引技术的优势来产生更好的索引。在本文中，我们实现了四种组合，以下是详细信息：

对于SID+IID：我们为每个item在顺序ID的末尾附加一个独立额外token。假设一个item分词后的SID是"10""18"，其IID索引是\langle IID982\rangle ，则HID索引将是"10""18"\langle IID982\rangle 。因此它包含来自SID的一些item共现信息，同时通过IID确保item区分。

对于CID和SemID，在我们将它们与IID拼接之前，首先移除它们的最后一个token（叶节点token），因为最后一个token仅用于区分同一父非叶节点下的不同item。对于CID+IID：假设一个item的CID是\langle 1\rangle \langle 9\rangle \langle 5\rangle \langle 4\rangle ，其IID是\langle IID28\rangle ，则该item的HID将是\langle 1\rangle \langle 9\rangle \langle 5\rangle \langle IID28\rangle 。对于SemID+IID：假设一个item的SemID是\langle Makeup\rangle \langle Lips\rangle \langle Lip_Liners\rangle \langle 5\rangle ，其IID是\langle IID1023\rangle ，则HID是\langle Makeup\rangle \langle Lips\rangle \langle Lip_Liners\rangle \langle IID1023\rangle 。最终索引既包含来自CID的协同信息（或SID中的元数据内容信息），又包含一个特殊的IIDtoken将该item与所有其他item区分开来，在保留CID（或SID）优势的同时确保item区分。

对于SemID+CID：我们按任一顺序拼接SemID和CID，希望结合元数据内容信息和协同信息。由于SemID和CID都包含叶节点token来区分一个父节点下的item，我们只需要保留其中之一，例如，我们保留CID叶节点token。假设SemID是\langle Makeup\rangle \langle Lips\rangle \langle Lip_Liners\rangle \langle 5\rangle ，CID是\langle 1\rangle \langle 9\rangle \langle 5\rangle \langle 4\rangle 。如果我们把SemID放在前面，最终HID索引是\langle Makeup\rangle \langle Lips\rangle \langle Lip_Liners\rangle \langle 1\rangle \langle 9\rangle \langle 5\rangle \langle 4\rangle ；否则，HID索引是\langle 1\rangle \langle 9\rangle \langle 5\rangle \langle 4\rangle \langle Makeup\rangle \langle Lips\rangle \langle Lip_Liners\rangle 。

在接下来的实验中，我们将评估和比较各种不同的HID。

5 实验

5.1 数据集与基线

数据集及其预处理方法已在第3.3节介绍。在本节中，我们介绍基线。我们将各种item索引方法应用到P5框架[9]中进行顺序推荐，并与几个代表性的顺序推荐方法作为基线进行比较：

Caser [26]：该方法将顺序推荐视为马尔可夫链，并利用卷积神经网络对用户兴趣进行建模。HGN [20]：该方法利用层次门控网络从长期和短期角度学习用户行为。GRU4Rec [12]：最初为基于会话的推荐提出，该方法采用GRU对用户点击历史序列进行建模。BERT4Rec [25]：该方法模仿BERT风格的掩码语言建模，学习用于顺序推荐的双向表示。FDSA [31]：聚焦于特征转换模式，该方法使用自注意力模块对特征序列进行建模。SASRec [14]：在顺序推荐模型中采用自注意力机制，该方法协调了马尔可夫链和基于RNN的方法的特性。S3-Rec [35]：利用item元信息上的自监督目标，该方法帮助顺序推荐模型更好地发现不同item及其属性之间的相关性。为了比较，我们使用了S3-Rec及其基线的实现。

5.2 实现细节

遵循P5框架[9]，我们的实现使用T5作为骨干[23]：编码器和解码器都有6层，模型维度为512，使用8头注意力。对于分词，我们使用默认的SentencePiece分词器[24]，词汇表大小为32,128，用于解析子词单元。所有独立的额外token不再进一步分词。我们使用与P5 [9]相同的顺序推荐提示，将顺序信息转换为文本。我们使用AdamW优化器在两个NVIDIA RTX A5000 GPU上预训练P5 20个epoch，批大小为64，峰值学习率为1e-3。我们在所有训练步骤的前5%应用预热以调整学习率。

RID、TID和SID不涉及创建OOV token，因为它们的item索引由默认T5分词器中的token组成，而IID、CID、SemID和HID涉及创建额外的OOV token，扩展了原始词汇表。这些索引方法中使用的所有token（TID除外）都是随机初始化的，而不是使用T5的预训练嵌入进行初始化。这是因为我们观察到，在实验过程中，T5关于数字的预训练先验语义对item语义的学习和推荐性能产生不利影响。我们使用T5的预训练token嵌入来初始化TIDtoken，因为TID仅涉及普通单词token。

表4：P5下所有基线结果和所有索引方法的性能。粗体数字代表最佳结果，波浪线下划线代表次优结果，直线下划线表示它们优于最佳基线结果。此处优于基线的结果已通过配对Student t检验测试为显著， $p$ 值 < 0.05。

5.3 总体结果

总体实验结果如表4所示，包含所有基线。每个指标的最佳结果以粗体突出显示，次优结果以波浪线下划线标出。对于每种索引方法，如果结果超过最佳基线结果，则通过直线下划线强调。总体而言，RID、TID和IID在大多数情况下无法击败基线结果，而大多数高级索引方法（SID、CID、SemID和各种HID）超过了基线结果。更详细的分析如下。

在表4中，第一个块包含所有基线结果。第二个块包含基本索引方法，其中RID和TID的表现始终比基线差，而IID总体表现更好。第三个块包含三种高级索引方法。我们可以看到SID在Amazon数据集上表现不如CID和SemID，但在Yelp上表现更好，而CID在不同数据集上的表现优于SemID，这表明使用协同信息构建索引比使用元数据更有益，因为CID可以通过从群体智慧中进行协同学习，更好地从用户行为中捕获item关系，这可能比仅使用item的元数据更有效。表中的第四个块包含多个不同实现的HID结果：SID+IID、CID+IID、SemID+IID和SemID+CID。CID+IID和SemID+IID的表现远优于所有其他索引方法，而SID+IID和SemID+CID表现较差。在以下小节中，我们将基于更全面的实验进一步详细分析第三和第四块中的结果。

5.4 顺序索引的不同设置

表4显示，尽管本质上简单，SID可以产生接近或超过基线的良好结果。在第4.1节中，我们探讨了SID的构建及其局限性，具体来说，索引结果可能受用户顺序的影响，例如，如果我们交换表3中User 1和User 2的行，那么索引结果将不同。在本节中，我们展示了使用四种不同用户顺序的SID结果，这证实了这一说法，并建议了最有效的使用顺序：

（1）时间敏感顺序（TSO）：用户在原始数据集中根据其与系统的初始交互按时间顺序排序。后续交互被记录，并且对于以前未记录的用户，在其首次与系统交互时创建新记录。通过基于时间戳对交互进行排序和处理，我们确保初始交互较早的用户先被记录。

（2）随机顺序（RO）：用户随机排序。

（3）短到长顺序（S2LO）：用户按其交互数量排序，从最少到最多交互升序排列。

（4）长到短顺序（L2SO）：用户按从最多到最少交互的降序排列。

表5展示了四种设置的性能。我们的观察表明，总体而言，相对性能如下：时间敏感 > {长到短、短到长} > 随机。这些观察结果表明，时间在顺序索引中扮演着重要角色：在相似时间被交互的item，即使由不同用户交互，也可能比在迥异时间被交互的item更相似。因此，在相似时间出现的item更有可能被某些用户共同交互。因此，在对用户排序时使用时间相关信息可能会提高性能。

表5：P5在不同设置下的顺序索引与三个数据集上的两个基线的比较。粗体数字代表最佳结果，波浪线数字代表次优结果。Amazon Beauty和Yelp上的TSO结果被测试为相对于其他设置显著。

考虑到这些观察结果，我们建议未来的简单SID方法实现考虑使用时间敏感的用户排序策略以增强性能。注意，原始的Amazon和Yelp数据集已经使用时间敏感排序来安排用户。因此，使用SID生成索引时，我们只需从第一个用户到最后一个用户递增地为item索引。

5.5 协同索引的不同设置

CID涉及两个超参数： $N$ 和 $k$ ，其中 $N$ 是每个聚类级别的聚类数量， $k$ 是最终聚类中允许的最大item数量。改变这些超参数会导致不同数量的独立额外token和推荐性能。

在图4中，我们展示了Beauty数据集上各种 $N$ 和 $k$ 值组合的hit@10结果。当 $k$ =50时，性能低于4.5%，显著低于基线和一些基本索引方法。然而，当 $k$ 大于100时，性能显著提高。

表6显示了多个配置在 $k \in \{200, 500, 1000\}$ , $N \in \{10, 20\}$ 以及所有三个数据集上的hit@10结果。在这些不同的设置中，几乎所有CID结果都优于基线，表明CID相对于其超参数相对容易调优。

图4：CID Beauty上 $N$ （每层聚类数）和 $k$ （最终聚类中允许的最大item数）的消融实验。

表6：CID在不同参数和数据集下的hit@10结果。粗体数字是最佳结果，波浪线数字是次优结果。所有数据集中得分最高的设置相对于其他设置在配对Student t检验下显著， $p$ 值 < 0.05。

基于我们的观察，我们可以得出以下结论：（1）无论选择什么样的 $N$ ，极小的 $k$ 值都会导致次优性能。当 $k$ =50时，性能低于基线。这可以归因于少量新token的有限表达能力，无法充分捕获item的多样性。（2）不同的 $k$ 和 $N$ 组合产生不同的ID长度（即ID中的token数量）。我们计算了每个 $k$ 和 $N$ 超参数设置的平均ID长度，结果显示在图5（Beauty）和表7（所有数据集）中。结合图4和5，以及表6和7，我们发现最优推荐结果通常在平均ID长度在3到4之间时观察到。例如，图5中的方块点显示了Beauty数据集上平均ID长度在3到4之间的所有情况，我们可以看到这些点也对应了图4中每条线上的最优性能。类似地，表6中的最佳或次佳结果在大多数情况下也对应表7中3∼4的ID长度。

图5：Beauty上的CID平均长度。

表7：不同参数下的平均ID长度。此表中的粗体数字对应表6中的最佳结果（即表6中的粗体数字）。

基于这些观察，我们建议未来的CID实现使用能产生平均ID长度在3到4之间的超参数。然而，值得注意的是，不同的数据集可能需要稍有不同的长度才能达到最佳性能。

5.6 语义索引何时有效

SemID使用元数据构建item索引。在我们的实验中，我们观察到如果类别遵循层次树结构，性能往往会提高。数据集中的类别信息通常不是树结构，因为在某些情况下，一个类别名称可以出现在不同的父类别下，这使得类别变成图而非树。表8是Amazon Beauty中的两个示例，其中类别"Eyes"同时出现在"Skin Care"和"Makeup Remover"下，类别"Creams"同时出现在"Skin Care"和"Moisturizers"下。

表8：Amazon Beauty数据集中非树结构类别的示例。

为了测试类别中的树结构是否关键，我们在实验中比较了两种不同的设置：

（1）非树结构设置：我们直接使用类别名称创建相应的独立OOV额外token。例如，一个在"Beauty"、"Skin Care"、"Eyes"下的item和另一个在"Beauty"、"Makeup"、"Makeup Remover"、"Eyes"下的item将共享token\langle Eyes\rangle 。

（2）树结构设置：我们通过在相同类别名称出现在不同位置时创建不同的OOV token来强制类别上的树结构。例如，在"Beauty"、"Skin Care"下的类别"Eyes"将对应token\langle Eyes1\rangle ，而在"Beauty"、"Makeup"、"Makeup Remover"下的对应\langle Eyes2\rangle 。

表9说明了层次信息对SemID有效性的重要性。类别越严格遵循层次结构，模型的性能越好。这可能是因为层次组织的类别列表有助于减少生成过程中的搜索空间。因此，这一发现凸显了在推荐基础模型中实施SemID时正确组织和构建类别信息的重要性。

表9：SemID在不同设置下的结果。粗体数字是最佳结果，波浪线数字是次优结果。Amazon Beauty和Yelp上的树结构设置结果被测试为相对于非树结构设置显著。

5.7 哪种类型的HID有效以及为什么

基于表4中展示的结果，CID+IID和SemID+IID相比它们各自的CID和SemID对应方法表现出更好的性能。但SID+IID相对于SID没有改进，而SemID+CID不仅没有改进反而大幅降低了性能。CID+IID和SemID+IID都是通过为每个item分配一个独立额外token并将其拼接在聚类ID或类别ID序列之后构建的。这些组合保持了原始索引长度，同时保留了层次结构。性能改进可以归因于额外token提供的索引表达能力的增强，以及混合索引中保留了协同信息或元数据信息。这些因素的组合促成了在CID+IID和SemID+IID方法中观察到的性能提升。

SID+IID通过在原始顺序索引后附加一个独立额外token创建，将ID长度增加1。SID+IID没有提高性能，可能是因为额外token干扰了原始顺序索引中以数字风格编码的时间敏感信息。SemID+CID通过拼接类别ID与聚类ID或反之创建，表现出次优性能，如表4所示。这对两种拼接顺序都成立：类别ID后接CID索引，以及聚类ID后接SemID索引。这种次优性能背后的原因是它产生了过长的索引并破坏了SemID和CID中编码的层次结构。考虑到我们的发现，我们建议使用CID+IID和SemID+IID作为推荐基础模型的混合索引，因为它们在此类场景中已展现出优越的性能。

6 结论

本文以P5作为示例骨干模型，考察了各种索引方法。我们考察了三种平凡索引方法：随机索引（RID）、标题索引（TID）和独立索引（IID），并强调了它们的局限性。这突显了为基础推荐模型选择适当索引方法的重要性，因为它极大地影响模型性能。然后我们考察了四种简单而有效的索引方法：顺序索引（SID）、协同索引（CID）、语义索引（SemID）和混合索引（HID）。在Amazon Sports、Amazon Beauty和Yelp数据集上的实验结果证明了它们强大的性能。这四种有效的索引方法满足本文引入的两个标准：（1）保持合适的ID长度，（2）将有用的先验信息整合到itemID构建中。我们希望本研究能为未来关于推荐基础模型及更广范围的索引方法研究提供启示。

致谢：本工作得到NSF IIS-2046457和IIS-2007907的部分支持。

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

[29] Hong-Jian Xue, Xinyu Dai, Jianbing Zhang, Shujian Huang, and Jiajun Chen. 2017. Deep matrix factorization models for recommender systems.. In IJCAI, Vol. 17. Melbourne, Australia, 3203–3209.

[30] Shuai Zhang, Lina Yao, Aixin Sun, and Yi Tay. 2019. Deep learning based recommender system: A survey and new perspectives. ACM Computing Surveys (CSUR) 52, 1 (2019), 1–38.

[31] Tingting Zhang, Pengpeng Zhao, Yanchi Liu, Victor S Sheng, Jiajie Xu, Deqing Wang, Guanfeng Liu, Xiaofang Zhou, et al. 2019. Feature-level Deeper Self-Attention Network for Sequential Recommendation.. In IJCAI. 4320–4326.

[32] Yongfeng Zhang, Qingyao Ai, Xu Chen, and W Bruce Croft. 2017. Joint representation learning for top-n recommendation with heterogeneous information sources. In Proceedings of the 2017 ACM on Conference on Information and Knowledge Management. 1449–1458.

[33] Yuhui Zhang, Hao Ding, Zeren Shui, Yifei Ma, James Zou, Anoop Deoras, and Hao Wang. 2021. Language models as recommender systems: Evaluations and limitations. (2021).

[34] Wayne Xin Zhao, Junhua Chen, Pengfei Wang, Qi Gu, and Ji-Rong Wen. 2020. Revisiting alternative experimental settings for evaluating top-n item recommendation algorithms. In Proceedings of the 29th ACM International Conference on Information & Knowledge Management. 2329–2332.

[35] Kun Zhou, Hui Wang, Wayne Xin Zhao, Yutao Zhu, Sirui Wang, Fuzheng Zhang, Zhongyuan Wang, and Ji-Rong Wen. 2020. S3-rec: Self-supervised learning for sequential recommendation with mutual information maximization. In CIKM.

[36] Han Zhu, Xiang Li, Pengye Zhang, Guozheng Li, Jie He, Han Li, and Kun Gai. 2018. Learning tree-based deep model for recommender systems. In KDD.

¹https://cseweb.ucsd.edu/ jmcauley/datasets/amazon_v2/
²https://www.yelp.com/dataset
³https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html
