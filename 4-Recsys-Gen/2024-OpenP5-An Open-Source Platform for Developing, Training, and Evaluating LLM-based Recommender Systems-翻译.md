# OpenP5 An Open Source Platform for Developing, Training, and Evaluating LLM based Recommender Systems

> shuyuan.xu@rutgers.edu

本文介绍了 OpenP5 An Open Source Platform for Developing, Training, and Evaluating LLM based Recommender Systems。核心内容：
关键发现：
---
OpenP5: 用于开发、训练和评估基于LLM的推荐系统的开源平台
Shuyuan Xu
罗格斯大学
美国新泽西州新不伦瑞克
Wenyue Hua
罗格斯大学
美国新泽西州新不伦瑞克
wenyue.hua@rutgers.edu
Yongfeng Zhang
罗格斯大学
美国新泽西州新不伦瑞克
yongfeng.zhang@rutgers.edu
4
2
0
2
r
p
A
1
1
]
R
I
.
s
c
[
2
v
4
3
1
1
1
.
6
0
3
2
:
v
i
X
r
a
## 摘要
近年来，将大型语言模型（LLM）集成到推荐系统中已引起从业者和研究者的兴趣。尽管有这样的兴趣，该领域仍处于新兴阶段，缺乏开源研发平台可能会阻碍基于LLM的推荐探索。本文介绍了OpenP5，一个旨在促进基于LLM的生成式推荐系统开发、训练和评估的开源平台，供研究使用。该平台使用编码器-解码器LLM（例如T5）和仅解码器LLM（例如Llama-2）在10个广泛认可的公共数据集上实现，涵盖两个基本的推荐任务：序列推荐和直接推荐。认识到itemID在基于LLM的推荐中的关键作用，我们还在OpenP5平台中集成了三种item索引方法：随机索引、序列索引和协作索引。该平台基于Transformers库构建，便于用户轻松定制基于LLM的推荐。OpenP5拥有一系列功能，包括可扩展的数据处理、以任务为中心的优化、全面的数据集和检查点、高效加速以及标准化评估，使其成为实现和评估基于LLM的推荐系统的宝贵工具。OpenP5库的开源代码和预训练检查点可在https://github.com/agiresearch/OpenP5公开获取。
## 关键词
大型语言模型；推荐系统；生成式推荐；开源
ACM参考格式：
Shuyuan Xu, Wenyue Hua, and Yongfeng Zhang. 2024. OpenP5: 用于开发、训练和评估基于LLM的推荐系统的开源平台. 见SIGIR '24: 第46届国际ACM SIGIR信息检索研究与发展会议, 2024年7月14–18日, 美国华盛顿特区. ACM, 纽约, 美国, 9页. https://doi.org/10.1145/nnnnnnn.nnnnnnn
允许为个人或课堂使用制作或分发本作品的全部或部分数字或硬拷贝，无需付费，前提是复制品不以盈利或商业优势为目的分发，且在第一页上注明此通知和完整引用。必须尊重本作品中他人拥有的版权。允许带引用的摘要。如需以其他方式复制、重新发布、上传到服务器或分发给列表，需事先获得特定许可和/或支付费用。请向permissions@acm.org请求许可。
SIGIR '24, 2024年7月14–18日, 美国华盛顿特区
© 2024 美国计算机协会。
ACM ISBN 978-x-xxxx-xxxx-x/YY/MM. . . $15.00$
https://doi.org/10.1145/nnnnnnn.nnnnnnn
1 引言
近来对基础模型（包括大型语言模型（LLM））在学术界和工业界的兴趣激增，很大程度上归因于它们在各个研究领域（包括自然语言处理（NLP）[1, 4, 35]和计算机视觉（CV）[50, 54]）的重大贡献。在推荐系统领域，从业者和研究者正逐步将这些模型整合到推荐任务中。某些近期研究，如P5 [12]和M6 [5]，已有效利用大型语言模型的优势，通过将推荐任务转换为自然语言格式来促进生成式推荐。然而，尽管对推荐系统中使用基础模型的关注日益增强，该领域仍相对初期，缺乏标准化开发平台可能会阻碍这一新兴领域的快速发展。
本文致力于通过引入OpenP5来解决推荐基础模型领域缺乏标准化开发平台的问题。OpenP5是一个基于P5模型[12]原则构建的、用于开发、训练和评估基于LLM的生成式推荐模型的开源平台。它整合了P5模型[12]的四个维度：主干模型、下游任务、推荐数据集和item索引方法。
在使用大型语言模型的推荐系统中，生成能力源自基础LLM。当代LLM架构主要分为三种类型：仅编码器、编码器-解码器和仅解码器。新兴的LLM大多采用编码器-解码器或仅解码器架构。为此，OpenP5平台为编码器-解码器和仅解码器架构各自集成了一个典型的LLM代表。具体来说，T5 [35]模型作为编码器-解码器架构的代表，而Llama-2 [42]模型则代表仅解码器架构。
在推荐基础模型中，语言作为一种有效的媒介，可以将各种推荐下游任务整合到单一模型中。因此，OpenP5考虑了推荐系统中最常见的两个任务：序列推荐和直接推荐。前者要求模型根据用户ID和用户历史生成推荐item，而后者要求模型仅根据用户ID生成推荐。
为了方便研究者和从业者，OpenP5平台包含了多种常用的公共推荐数据集。我们对近年来流行的数据集进行了全面调研，并将排名前10的数据集纳入库中。我们还设计了一个Super P5（SP5）模型，以初步探索能够使用单一模型跨各种数据集推荐item的推荐基础模型的潜力。
在OpenP5中，我们还包含了多种用语言表示item的方法。强调了在推荐基础模型中为每个item分配唯一ID的关键性，确保每个item由最少数量的token表示，能够与其他item区分，并避免生成式推荐中的幻觉问题[17]。此外，item索引方法会极大影响推荐基础模型的性能。现有研究在将推荐任务转化为语言生成任务时，采用了多种item表示方法。例如，P5 [12]使用数字token，M6 [5]利用丰富的元数据生成基于元数据的嵌入来表示item，LMRecSys [53]使用item标题作为表示。然而，考虑到许多公共数据集可能不包含丰富的元数据或文本信息，OpenP5平台仅包含三种仅基于用户-item交互的item索引方法：随机索引、序列索引和协作索引[17]。
总之，OpenP5提供了一个基于P5原则[12]开发、训练和评估基于LLM的推荐系统的平台，涵盖两个下游任务、十个数据集、三种ID创建方法，并支持编码器-解码器和仅解码器LLM架构。它还提供了基于两种主干模型在十个流行公共数据集上的检查点，以及一个在三个item索引方法上预训练了所有数据集的SP5实现。此外，该平台还支持用户基于我们提供的API开发自定义方法，例如新的ID创建方法、主干、数据集、任务或评估方法，以促进未来对基于LLM的生成式推荐的研究。
本文的其余部分组织如下。在第2节中，我们提供必要的背景和相关工作。在第3节到第7节中，我们介绍如何在OpenP5中处理数据，包括原始数据预处理、item索引方法、个性化提示收集和数据准备，解释OpenP5的预训练和微调细节，提供评估方法，并展示实验结果，这可以帮助平台用户轻松地将OpenP5平台适配到其他数据或任务。最后，我们在第8节总结工作并讨论未来方向。
2 相关工作
最近，已有若干尝试将大型语言模型的能力引入推荐系统。根据[16, 27]，我们从三个维度介绍现有工作：LLM的角色、如何适配LLM以及推荐任务。
凭借强大的能力，LLM可以参与推荐系统的多个组件。LLM可用于特征工程，即接受原始数据作为输入，生成丰富的文本特征作为数据增强[32, 47]。LLM还可作为特征表示提取器，将特征表示为嵌入。使用LLM获取特征表示可以提供具有丰富语义信息的item或用户表示[23, 46, 48]，并且可能有助于以自然语言作为连接的跨域或冷启动推荐[8, 10, 44]。一些工作[5, 12, 24, 29]直接使用LLM作为推荐系统，能够通过LLM完成推荐相关任务。除了作为推荐系统的一部分，LLM还可以用作控制器，可能导致更具交互性和可解释性的推荐[9, 11]。
关于如何适配LLM，推荐系统中的LLM可以进行调优或不调优，这取决于模型是否在训练阶段调优LLM。这包括全微调和其他参数高效微调方法，如LoRA [15]。随着大型基础模型的出现，研究者们倾向于分析LLM在推荐场景中的零样本或少样本性能。许多现有工作[6, 25, 28, 38, 40, 45]研究了基于LLM的无微调零样本推荐，通过构建提示来指导LLM完成各种推荐任务，如评分预测、成对比较、重排序等。尽管LLM可能提供良好的语言理解性能，但无微调的推荐性能仍需未来改进，这表明了来自推荐系统的用户-item交互信息等领域知识的重要性[6, 28]。
随着推荐系统领域的发展，推荐系统的任务不仅限于评分预测或item推荐。传统推荐系统通常针对特定任务设计，因为不同任务的数据格式和模型架构存在差异。借助LLM，语言可以充当桥梁，将各种下游任务整合到单一模型中。现有基于LLM的推荐系统工作可以从下游任务的角度分为多任务推荐和特定任务推荐。一些工作[18, 22, 34]引入LLM以提高特定任务的性能，而另一些工作[5, 12, 51]则使用LLM通过统一语言格式处理多任务。基于上述方面，我们将OpenP5平台定义为一种工具，供有兴趣将微调LLM开发为推荐系统以执行多个推荐任务的研究者和从业者使用。
0 20 40 60 80 100 120 140 160 Diginetica Book-Crossing Amazon Sports Amazon Phones Foursquare MIND Amazon Music Amazon Games MovieLens-10M Amazon Toys Gowalla MovieLens-20M Amazon Books Amazon Electronics Taobao Amazon Movies Amazon CDs Amazon Clothing MovieLens-100K Amazon Beauty LastFM Yelp MovieLens-1M
用户
交互历史
B004756YJA B004ZT0SSG B0020YLEYK 7806397051 B002WLWX82
B0009P4PZC B009HULFLW B00BZ1QN2C B00G2TQNZ4 B00812ZWOS 7806397051 B0000YUX4O
A1YJEY40YUW4SE
A60XNB876KYML
A3G6XNM240RMWA 7806397051 B003H8180I B00538TSMU B002S8TOYU B001MP471K B00011JI88 B00C1F13CQ B003ATNYJC B003ZS6ONQ
A1PQFP6SAJ6D80
A38FVHZTNQ271F
B0030HKJ8I B00027D8IC B002PMLGOU B00BN1MPPS 7806397051 B004Z40048
7806397051 B009PZVOF6 B008LQX8J0 B007EHWDTS B009DDGHFC B002BGDLDO B003VWZCMK B00DQ2ILQY B00DAYGJVW
表1: 此表展示了Amazon Beauty数据中的几个实例。数据以文本文件格式存储，每行包含单个用户的信息。在每一行中，第一个元素代表用户的原始ID，后续元素是按用户交互历史的时间顺序列出的item原始ID。
3 数据处理
在本节中，我们将讨论平台的数据处理模块，该模块允许用户集成新的数据集并创建自定义扩展。
3.1 原始数据预处理
OpenP5平台提供了10个流行的预处理公共数据集。为了确定适合推荐的流行公共数据集，我们对它们在近期出版物中的出现频率进行了分析。更具体地说，我们考察了过去三年在相关会议（包括SIGIR、RecSys、WSDM、KDD、WWW和CIKM）上录用的论文。使用ACM数字图书馆1，我们以关键词"推荐"、"推荐器"、"推荐"和"协作"筛选相关出版物。由于公共数据集数量众多，我们在图1中仅展示了出现次数超过20次的数据集频率。我们在OpenP5库中包含了排名前10的流行公共数据集，包括Movielens-1M、Yelp、LastFM、Amazon Beauty、Movielens-100K、Amazon Clothing、Amazon CDs、Amazon Movies、Taobao和Amazon Electronics。这种方法确保所选数据集不仅流行，而且与推荐系统当前的研究趋势一致。我们在GitHub仓库2中提供了所有数据集的统计概览。
预处理后的数据保存为txt文件，表1展示了来自Amazon Beauty数据集的一个示例。我们的平台主要需要用户-item交互数据，因为大多数公共数据集可能不提供额外信息。更准确地说，我们将不同用户的信息分隔到单独的行中。在每一行内，元素由空格分隔，其中第一个元素表示用户原始ID，后续元素（item原始ID）按时间顺序描述用户的交互历史。平台用户可以通过将原始数据转换为指定的数据格式，轻松地在新数据集上训练模型。平台将自动将数据划分为训练集、验证集和测试集。
3.2 item索引
为了将推荐任务转化为语言生成任务，用户和item标识符需要与自然语言兼容。这种兼容性确保这些标识符可以无缝地融入用于大型语言模型预训练、微调和提示阶段的自然语言指令中。我们的平台提供了三种item索引方法的实现：随机索引、序列索引和协作索引。应用索引方法后，结果保存为一个txt文件，每行由两个值组成，其中第一个值表示原始ID，第二个值表示重新索引后的ID。索引后的用户-item交互数据将与预处理数据（即表1）格式相同。我们将介绍平台中提供的索引方法的更多细节。
随机索引。随机索引是一种直接的item索引方法。该方法为每个item分配一个唯一的随机数作为itemID。在模型内部，使用SentencePiece分词器[37]进一步将该随机数ID分词为token序列。例如，一个随机分配唯一ID为"2048"的item，在推荐基础模型中将被分词为token"20"和"48"。
虽然随机索引在传统推荐系统中经常使用，但它可能并不最适合基础模型[17]：其潜在缺陷在于，随机分配的ID被进一步分词后，可能会无意中导致不相关的item共享相同的token。例如，item"2048"和"2049"尽管完全不相关，甚至没有被同一用户交互过，却共享了token"20"。因此，模型可能会错误地在这些item之间建立语义关系。由于这种关系源于索引结构，无论模型如何从数据中学习都无法消除，从而影响推荐的准确性[17]。因此，RID被认为是一种不利的方法。然而，我们仍在此平台中包含这种简单的索引方法，以便研究者将其用作比较和探索的基线。
序列索引。为了缓解与随机索引相关的问题，一种可行的策略是将协作信息整合到itemID中。这种方法的一个基本实现可以在序列索引方法中看到，如[17]所示。该方法从第一个用户开始，依次遍历到最后一个用户，为用户连续的交互分配连续的编号ID。它遍历所有交互，为任何尚未分配的item分配一个新的、递增的ID。重要的是，我们仅对训练数据应用序列索引方法，以避免评估阶段的潜在数据泄露。
对于经过序列索引的item，如果在分词后两个item在同一位置共享相同的token，这表明这两个item可能被同一个用户交互过。因此，嵌入在itemID中的序列信息可能潜在地增强基础模型推荐的效果。
协作索引。为了将更多的协作信息整合到item索引中，我们在OpenP5库中集成了协作索引方法。协作索引方法的基本直觉基于这样的理念：item共现的频率应影响它们在相同位置共享相同token的程度[17]。这一概念表示为图，其中节点表示item，边权重表示共现频率。为了生成协作索引，我们采用谱聚类方法[33, 43]。由于协作索引方法需要引入词汇表外（OOV）token来构建item索引，我们用尖括号"⟨⟩"表示这些OOVtoken（例如，"⟨𝐶𝐼 1⟩"）。该方法的详细说明见算法1。
算法1 协作索引方法
输入：训练数据用户序列𝐷，要创建的簇数𝑁，最大允许簇中的item数𝑘
1: 实例化一个队列，将所有item作为一个集合入队
2: while 队列不为空 do
3:   出队第一个item集合𝑆
4:   if 𝑆的大小 < 𝑘 then
5:     为𝑆内的所有item分配唯一token
6:   else
7:     基于𝐷计算𝑆中item的共现矩阵𝑀
8:     对𝑀应用谱聚类，分为𝑁个簇
9:     为每个簇生成唯一token，并基于聚类结果为𝑆内的所有item分配相应token
10:    将所有结果簇入队
11:  end if
12: end while
3.3 个性化提示收集
推荐基础模型具备将推荐的各种下游任务整合到单一生成模型中的能力[5, 12]。考虑到某些公共数据集可能不包含某些信息，如评论、元数据、显式反馈等，OpenP5库仅关注推荐系统中最常用的两个下游任务：序列推荐任务和直接推荐任务。这两个任务都包含针对个别用户的个性化提示。更具体地说，我们为这两个任务设计了多种提示模板，这些模板填充了用户ID和itemID等个性化信息。此外，为了避免在SP5中推荐来自不同数据集的item（例如，向Amazon用户推荐Yelp餐厅），我们设计的提示中包含了数据集名称。
序列推荐任务需要基于用户历史生成推荐item，因此个性化提示包含数据集名称、用户ID、用户历史和目标itemID。直接推荐任务要求模型仅根据用户ID生成推荐item，因此该任务的提示不包括用户历史。
以下示例说明了这两个下游任务的提示模板。
序列推荐
输入模板: Considering {dataset} use $r_{user_id}$ has interacted with {dataset} items {history} . What is the next recommendation for the user ?
目标模板: {dataset} {target}
直接推荐
输入模板: What should we recommend for {dataset} use $r_{user_id}$ ?
目标模板: {dataset} {target}
OpenP5平台为每个下游任务提供11个不同的提示模板。从每个任务中，选择一个提示模板作为未见提示，用于评估模型的零样本泛化能力。值得注意的是，OpenP5平台在设计上考虑了灵活性，使用户能够根据自己的特定需求或目标修改提示模板。更具体地说，提示模板保存在txt文件中，表2展示了一个代表性示例。每行描述一个独特的提示模板，包含由分号分隔的四类信息：第一项指定提示所属的任务；第二项表示该提示模板在训练期间是否暴露给模型；第三项概述输入的指令；第四项表示作为输出的推荐item。提示模板中的个性化信息用花括号括起来，在数据处理过程中会被替换为具体数据。
4 多任务学习
在之前的讨论中，我们强调了OpenP5平台支持两个主要的推荐任务：序列推荐和直接推荐。值得一提的是，该平台在架构上也具备将额外任务整合到其训练范式中的能力。这种可扩展性是平台设计的基石，允许更广泛的学习和适应性。本节致力于阐述平台提供的多任务学习框架。
当任务按顺序而非同时学习时，模型容易受到"遗忘问题"[20, 21, 26, 49]的影响，这主要会提升其在最新任务上的性能，而损害之前任务的性能。为了解决这个问题，同时任务学习是必要的。一个常见但直观的解决方案是混合来自不同任务的训练数据。然而，这种方法并非没有缺陷。就我们的Super P5（SP5）模型而言，当数据集大小不均时，不加区分地混合来自不同数据集的训练数据仍可能导致"遗忘问题"。此外，不同任务的文本长度需求也不同；序列推荐需要交互历史，导致与直接推荐相比输入序列更长。这种差异在批处理来自多个任务的数据时可能导致过多的填充。为了避免这些问题，我们的平台确保每个批次是任务同质的——即数据来自同一任务。这种策略有效地缓解了遗忘问题，并在不同的任务需求之间保持了效率。
表2: 此表展示了提示模板的几个实例。提示模板存储在一个文本文件中，每行代表一个唯一的提示模板。每个提示包含四类由分号分隔的信息：推荐任务、训练期间是可见还是不可见指示、输入模板和输出模板。
图像通过视觉编码器转换为图像token，这些token由分词器视为OOV。
5 预训练和微调
在给出了多个推荐任务的个性化提示之后，我们接下来介绍OpenP5平台的预训练和微调。
为了提高预训练和微调的效率，OpenP5平台引入了两种技术。一种是在多GPU环境中启用分布式学习。分布式学习使模型能够在更短的时间内完成学习，从而提高效率。除了分布式学习，OpenP5平台还集成了高效训练方法，如LoRA [15]，它冻结预训练模型权重并注入可训练的低秩分解矩阵，以减少可训练参数并提高效率。
6 如何自定义OpenP5
该平台促进用户开发自定义的基于LLM的推荐模型。本节通过示例说明如何从不同角度适配OpenP5。
• 集成新数据集：使用我们的平台，只要数据集格式正确，将新数据集集成到推荐模型中就很直接。这种易集成性支持基于LLM的推荐模型无缝地使用新数据源进行训练。
• 用户/itemID：我们的平台支持三种不同的item索引方法，并具有引入其他索引策略的灵活性。例如，可以通过生成使用这些新ID的预处理数据来实现将item标题作为唯一标识符。
• 采用新主干模型：由于平台的架构基于Transformers库3，用户可以方便地从Transformers库中替换为其他替代主干模型。
• 自定义个性化提示：平台上的个性化提示模板可以轻松替换，从而允许将新任务纳入训练过程。此外，平台管理词汇表外（OOV）token的能力增强了其对需要此类功能的基于LLM的推荐模型的实用性。例如，Geng等人[13]引入了一个多模态基础模型，将item图像集成到提示中，利用视觉编码器将图像转换为token。
总之，OpenP5在容纳新数据集、索引方法、主干模型和任务方面表现出显著的灵活性。这种适应性突显了其作为基于LLM的生成式推荐系统领域开创性研究基础工具的潜力。
7 实验
7.1 数据集和基线
我们已在第3节中介绍了数据集集合。由于篇幅限制，我们在三个数据集上呈现实验结果：Movielens-1M、Amazon Beauty、LastFM。其余结果可通过我们的GitHub仓库4访问。为了展示OpenP5平台的优越性能，我们收集了一系列针对不同下游任务的代表性方法。
序列推荐。由于我们的OpenP5仅使用用户交互信息进行预测，为了公平比较，我们采用了几个同样仅使用用户交互信息的著名序列推荐基线。我们介绍序列推荐的基线模型如下。
• Caser [41]将序列推荐视为马尔可夫链，并使用卷积神经网络（CNN）对用户进行建模。
• HGN [30]使用层次门控网络从长期和短期角度学习用户行为。
• GRU4Rec [14]利用门控循环单元（GRU）[3]对用户交互序列进行建模。
• Bert4Rec [39]利用BERT风格的掩码语言建模[7]学习序列推荐的双向表示。
• FDSA [52]使用自注意力模块对特征序列进行建模。
• SASRec [19]在序列推荐模型中部署自注意力机制。
直接推荐。我们使用三种现有方法作为直接推荐任务的基线。
• BPR-MF [36]利用矩阵分解配合成对贝叶斯个性化排序（BPR）损失。
• BPR-MLP [2]利用MLP对用户和item进行建模。
• SimpleX [31]在协同过滤中利用余弦对比损失（CCL）进行推荐，这是一个非常强的基线，击败了许多基于图的推荐模型。
表3: 序列推荐任务的性能结果。R、S、C代表三种item索引。
7.2 实现细节
遵循P5框架[12]，我们的实现基于T5模型[35]和LLaMA-2 [42]模型。T5主干在全部参数上训练，而LLaMA-2使用LoRA [15]训练。值得注意的是，我们随机初始化预训练检查点中与数字相关token的嵌入。这是基于以下事实：虽然这些嵌入在预训练阶段封装了token之间的语义相似性，但当item用数字标识符索引时，这种语义模式可能不会延续到推荐任务中。对于训练期间的提示，我们为每个任务选择10个提示，保留一个用于评估零样本泛化能力。为了缓解潜在的遗忘问题，我们采用了一种训练方案，其中来自不同任务的批次交替进行。对于SP5，解决数据不平衡和潜在的遗忘问题很重要。因此，我们交替使用来自不同数据集和任务的批次。对于较小的数据集，我们重复迭代直到最大数据集的训练完成，以确保平衡的训练过程。
7.3 结果分析
序列推荐任务和直接推荐任务的性能指标分别如表3和表4所示。具体来说，我们使用top-𝑘命中率（HR@𝑘）和归一化折损累计增益（NDCG@𝑘）来评估性能，提供了HR@5,10和NDCG@5,10的结果。每个指标的最佳结果以粗体突出显示，次优结果以下划线标出。
从表3和表4中显示的推荐性能，我们可以观察到，与基线相比，生成式推荐在大多数情况下能够达到最佳性能，但在很大程度上依赖于预训练主干模型和item索引方法。比较基于T5的OpenP5和基于Llama的OpenP5，在大多数情况下，使用T5主干的表现优于Llama主干。这可能是由于Llama主干的参数数量庞大，导致在稀疏推荐数据上欠拟合。对于基于T5的OpenP5，比较三种item索引方法可以预期随机索引方法的性能较低，序列索引方法略逊于协作索引方法。相反，基于Llama的OpenP5模型没有显示出对任何索引方法的明显偏好，这可能是由于稀疏数据对其更大参数空间的影响。这为开发既有效又参数高效的基于LLM的生成式推荐模型指明了未来方向。
表4: 直接推荐任务的性能结果。R、S、C代表三种item索引。
8 结论与未来工作
在本文中，我们提供了OpenP5平台，作为促进基于大型语言模型的推荐系统开发、训练和评估的资源。我们从四个角度考虑了实现：主干模型、下游任务、推荐数据集和item索引方法。该平台作为开发和评估推荐基础模型的持续努力，帮助社区在这一方向上通过未来创新取得进一步进展。未来，我们将考虑将更多的item索引方法、更多的基础模型训练和推理范式、更多的数据模态以及更多的主干LLM整合到平台中。
附录
在本附录中，我们提供了两个下游任务的个性化提示完整列表。
A 序列推荐
提示 可见: A1
输入模板: Considering {dataset} use $r_{user_id}$ has interacted with {dataset} items {history} . What is the next recommendation for the user ?
目标模板: {dataset} {target}
提示 可见: A2
输入模板: Here is the purchase history of {dataset} use $r_{user_id}$ : {dataset} item {history} . I wonder what is the next recommended item for the user .
目标模板: {dataset} {target}
提示 可见: A3
输入模板: {dataset} use $r_{user_id}$ has purchased {dataset} items {history} , predict next possible item to be bought by the user ?
目标模板: {dataset} {target}
提示 可见: A4
输入模板: I find the purchase list of {dataset} use $r_{user_id}$ : {dataset} items {history} , I wonder what other items does the user need . Can you help me decide ?
目标模板: {dataset} {target}
提示 可见: A5
输入模板: According to what items {dataset} use $r_{user_id}$ has purchased : {dataset} items {history} , Can you recommend another item to the user ?
目标模板: {dataset} {target}
提示 可见: A6
输入模板: What would {dataset} use $r_{user_id}$ be likely to purchase next after buying {dataset} items {history} ?
目标模板: {dataset} {target}
提示 可见: A7
输入模板: By analyzing the {dataset} use $r_{user_id}$ 's purchase of {dataset} items {history} , what is the next item expected to be bought ?
目标模板: {dataset} {target}
提示 可见: A8
输入模板: Can you recommend the next item for {dataset} use $r_{user_id}$ , given the user 's purchase of {dataset} items {history} ?
目标模板: {dataset} {target}
提示 可见: A9
输入模板: After buying {dataset} items {history} , what is the next item that could be recommended for {dataset} use $r_{user_id}$ ?
目标模板: {dataset} {target}
提示 可见: A10
输入模板: The {dataset} use $r_{user_id}$ has bought items : {dataset} items {history} , What else do you think is necessary for the user ?
目标模板: {dataset} {target}
提示 未见: A11
输入模板: What is the top recommended item for {dataset} use $r_{user_id}$ who interacted with {dataset} item {history} ?
目标模板: {dataset} {target}
B 直接推荐
提示 可见: B1
输入模板: What should we recommend for {dataset} use $r_{user_id}$ ?
目标模板: {dataset} {target}
提示 可见: B2
输入模板: {dataset} use $r_{user_id}$ is looking for some items . Do you have any recommendations ?
目标模板: {dataset} {target}
提示 可见: B3
输入模板: Do you have any suggested items for dataset use $r_{user_id}$ ?
目标模板: {dataset} {target}
提示 可见: B4
输入模板: Which recommendation should we provide to {dataset} use $r_{user_id}$ ?
目标模板: {dataset} {target}
提示 可见: B5
输入模板: How can we assist {dataset} use $r_{user_id}$ with a recommendation ?
目标模板: {dataset} {target}
提示 可见: B6
输入模板: What would be a suitable recommendation for {dataset} use $r_{user_id}$ ?
目标模板: {dataset} {target}
提示 可见: B7
输入模板: What would be a helpful recommendation for {dataset} use $r_{user_id}$ ?
目标模板: {dataset} {target}
提示 可见: B8
输入模板: Can you recommend an item for {dataset} use $r_{user_id}$ ?
目标模板: {dataset} {target}
提示 可见: B9
输入模板: Based on {dataset} use $r_{user_id}$ 's interests and requirements , what item would you suggest to try ?
目标模板: {dataset} {target}
提示 可见: B10
输入模板: For {dataset} use $r_{user_id}$ , what item stands out as a top recommendation that they should consider ?
目标模板: {dataset} {target}
提示 未见: B11
输入模板: What is the top recommendation for {dataset} use $r_{user_id}$ ?
目标模板: {dataset} {target}
## 参考文献
[1] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020. Language models are few-shot learners. Advances in neural information processing systems 33 (2020), 1877–1901.
[2] Heng-Tze Cheng, Levent Koc, Jeremiah Harmsen, Tal Shaked, Tushar Chandra, Hrishi Aradhye, Glen Anderson, Greg Corrado, Wei Chai, Mustafa Ispir, et al. 2016. Wide & deep learning for recommender systems. In Proceedings of the 1st workshop on deep learning for recommender systems. 7–10.
[3] Kyunghyun Cho, Bart Van Merriënboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. 2014. Learning phrase representations using RNN encoder-decoder for statistical machine translation. arXiv preprint arXiv:1406.1078 (2014).
[4] Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Eric Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, et al. 2022. Scaling instruction-finetuned language models. arXiv preprint arXiv:2210.11416 (2022).
[5] Zeyu Cui, Jianxin Ma, Chang Zhou, Jingren Zhou, and Hongxia Yang. 2022. M6-Rec: Generative Pretrained Language Models are Open-Ended Recommender Systems. arXiv preprint arXiv:2205.08084 (2022).
[6] Sunhao Dai, Ninglu Shao, Haiyuan Zhao, Weijie Yu, Zihua Si, Chen Xu, Zhongxiang Sun, Xiao Zhang, and Jun Xu. 2023. Uncovering ChatGPT's Capabilities in Recommender Systems. In Proceedings of the 17th ACM Conference on Recommender Systems.
[7] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers). 4171–4186.
[8] Hao Ding, Yifei Ma, Anoop Deoras, Yuyang Wang, and Hao Wang. 2021. Zero-shot recommender systems. arXiv preprint arXiv:2105.08318 (2021).
[9] Luke Friedman, Sameer Ahuja, David Allen, Terry Tan, Hakim Sidahmed, Changbo Long, Jun Xie, Gabriel Schubiner, Ajay Patel, Harsh Lara, et al. 2023. Leveraging Large Language Models in Conversational Recommender Systems. arXiv preprint arXiv:2305.07961 (2023).
[10] Junchen Fu, Fajie Yuan, Yu Song, Zheng Yuan, Mingyue Cheng, Shenghui Cheng, Jiaqi Zhang, Jie Wang, and Yunzhu Pan. 2023. Exploring Adapter-based Transfer Learning for Recommender Systems: Empirical Studies and Practical Insights. arXiv preprint arXiv:2305.15036 (2023).
[11] Yunfan Gao, Tao Sheng, Youlin Xiang, Yun Xiong, Haofen Wang, and Jiawei Zhang. 2023. Chat-rec: Towards interactive and explainable llms-augmented recommender system. arXiv preprint arXiv:2303.14524 (2023).
[12] Shijie Geng, Shuchang Liu, Zuohui Fu, Yingqiang Ge, and Yongfeng Zhang. 2022. Recommendation as language processing (rlp): A unified pretrain, personalized prompt & predict paradigm (p5). In Proceedings of the 16th ACM Conference on Recommender Systems. 299–315.
[13] Shijie Geng, Juntao Tan, Shuchang Liu, Zuohui Fu, and Yongfeng Zhang. 2023. VIP5: Towards Multimodal Foundation Models for Recommendation. In Findings of the Association for Computational Linguistics: EMNLP 2023.
[14] Balázs Hidasi, Alexandros Karatzoglou, Linas Baltrunas, and Domonkos Tikk. 2016. Session-based recommendations with recurrent neural networks. In ICLR.
[15] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2021. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685 (2021).
[16] Wenyue Hua, Lei Li, Shuyuan Xu, Li Chen, and Yongfeng Zhang. 2023. Tutorial on Large Language Models for Recommendation. In Proceedings of the 17th ACM Conference on Recommender Systems. 1281–1283.
[42] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. 2023. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971 (2023).
[43] Ulrike Von Luxburg. 2007. A tutorial on spectral clustering. Statistics and computing 17 (2007), 395–416.
[44] Jie Wang, Fajie Yuan, Mingyue Cheng, Joemon M Jose, Chenyun Yu, Beibei Kong, Xiangnan He, Zhijin Wang, Bo Hu, and Zang Li. 2022. TransRec: Learning Transferable Recommendation from Mixture-of-Modality Feedback. arXiv preprint arXiv:2206.06190 (2022).
[45] Lei Wang and Ee-Peng Lim. 2023. Zero-Shot Next-Item Recommendation using Large Pretrained Language Models. arXiv preprint arXiv:2304.03153 (2023).
[46] Chuhan Wu, Fangzhao Wu, Tao Qi, Chao Zhang, Yongfeng Huang, and Tong Xu. 2022. Mm-rec: Visiolinguistic model empowered multimodal news recommendation. In Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval. 2560–2564.
[47] Yunjia Xi, Weiwen Liu, Jianghao Lin, Jieming Zhu, Bo Chen, Ruiming Tang, Weinan Zhang, Rui Zhang, and Yong Yu. 2023. Towards Open-World Recommendation with Knowledge Augmentation from Large Language Models. arXiv preprint arXiv:2306.10933 (2023).
[48] Yang Yu, Fangzhao Wu, Chuhan Wu, Jingwei Yi, and Qi Liu. 2021. Tiny-newsrec: Effective and efficient plm-based news recommendation. arXiv preprint arXiv:2112.00944 (2021).
[49] Fajie Yuan, Guoxiao Zhang, Alexandros Karatzoglou, Joemon Jose, Beibei Kong, and Yudong Li. 2021. One person, one model, one world: Learning continual user representation without forgetting. In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval. 696–705.
[50] Lu Yuan, Dongdong Chen, Yi-Ling Chen, Noel Codella, Xiyang Dai, Jianfeng Gao, Houdong Hu, Xuedong Huang, Boxin Li, Chunyuan Li, et al. 2021. Florence: A new foundation model for computer vision. arXiv preprint arXiv:2111.11432 (2021).
[51] Junjie Zhang, Ruobing Xie, Yupeng Hou, Wayne Xin Zhao, Leyu Lin, and Ji-Rong Wen. 2023. Recommendation as instruction following: A large language model empowered recommendation approach. arXiv preprint arXiv:2305.07001 (2023).
[52] Tingting Zhang, Pengpeng Zhao, Yanchi Liu, Victor S Sheng, Jiajie Xu, Deqing Wang, Guanfeng Liu, Xiaofang Zhou, et al. 2019. Feature-level Deeper Self-Attention Network for Sequential Recommendation. In IJCAI. 4320–4326.
[53] Yuhui Zhang, Hao Ding, Zeren Shui, Yifei Ma, James Zou, Anoop Deoras, and Hao Wang. 2021. Language models as recommender systems: Evaluations and limitations. In NeurIPS 2021 Workshop on I (Still) Can't Believe It's Not Better.
[54] Kaiyang Zhou, Jingkang Yang, Chen Change Loy, and Ziwei Liu. 2022. Learning to prompt for vision-language models. International Journal of Computer Vision 130, 9 (2022), 2337–2348.
[17] Wenyue Hua, Shuyuan Xu, Yingqiang Ge, and Yongfeng Zhang. 2023. How to Index Item IDs for Recommendation Foundation Models. SIGIR-AP (2023).
[18] Jianchao Ji, Zelong Li, Shuyuan Xu, Wenyue Hua, Yingqiang Ge, Juntao Tan, and Yongfeng Zhang. 2023. Genrec: Large language model for generative recommendation. arXiv e-prints (2023), arXiv–2307.
[19] Wang-Cheng Kang and Julian McAuley. 2018. Self-attentive sequential recommendation. In 2018 IEEE international conference on data mining (ICDM). IEEE, 197–206.
[20] Ronald Kemker, Marc McClure, Angelina Abitino, Tyler Hayes, and Christopher Kanan. 2018. Measuring catastrophic forgetting in neural networks. In Proceedings of the AAAI conference on artificial intelligence, Vol. 32.
[21] James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, et al. 2017. Overcoming catastrophic forgetting in neural networks. Proceedings of the national academy of sciences 114, 13 (2017), 3521–3526.
[22] Lei Li, Yongfeng Zhang, and Li Chen. 2023. Personalized prompt learning for explainable recommendation. ACM Transactions on Information Systems 41, 4 (2023), 1–26.
[23] Ruyu Li, Wenhao Deng, Yu Cheng, Zheng Yuan, Jiaqi Zhang, and Fajie Yuan. 2023. Exploring the Upper Limits of Text-Based Collaborative Filtering Using Large Language Models: Discoveries and Insights. arXiv preprint arXiv:2305.11700 (2023).
[24] Xinyi Li, Yongfeng Zhang, and Edward C Malthouse. 2023. PBNR: Prompt-based News Recommender System. arXiv preprint arXiv:2304.07862 (2023).
[25] Xinyi Li, Yongfeng Zhang, and Edward C Malthouse. 2023. A Preliminary Study of ChatGPT on News Recommendation: Personalization, Provider Fairness, Fake News. arXiv preprint arXiv:2306.10702 (2023).
[26] Zhizhong Li and Derek Hoiem. 2017. Learning without forgetting. IEEE transactions on pattern analysis and machine intelligence 40, 12 (2017), 2935–2947.
[27] Jianghao Lin, Xinyi Dai, Yunjia Xi, Weiwen Liu, Bo Chen, Xiangyang Li, Chenxu Zhu, Huifeng Guo, Yong Yu, Ruiming Tang, et al. 2023. How Can Recommender Systems Benefit from Large Language Models: A Survey. arXiv preprint arXiv:2306.05817 (2023).
[28] Junling Liu, Chao Liu, Renjie Lv, Kang Zhou, and Yan Zhang. 2023. Is chatgpt a good recommender? a preliminary study. arXiv preprint arXiv:2304.10149 (2023).
[29] Peng Liu, Lemei Zhang, and Jon Atle Gulla. 2023. Pre-train, prompt and recommendation: A comprehensive survey of language modelling paradigm adaptations in recommender systems. arXiv preprint arXiv:2302.03735 (2023).
[30] Chen Ma, Peng Kang, and Xue Liu. 2019. Hierarchical gating networks for sequential recommendation. In Proceedings of the 25th ACM SIGKDD international conference on knowledge discovery & data mining. 825–833.
[31] Kelong Mao, Jieming Zhu, Jinpeng Wang, Quanyu Dai, Zhenhua Dong, Xi Xiao, and Xiuqiang He. 2021. SimpleX: A simple and strong baseline for collaborative filtering. In Proceedings of the 30th ACM International Conference on Information & Knowledge Management. 1243–1252.
[32] Sheshera Mysore, Andrew McCallum, and Hamed Zamani. 2023. Large Language Model Augmented Narrative Driven Recommendations. arXiv preprint arXiv:2306.02250 (2023).
[33] Andrew Ng, Michael Jordan, and Yair Weiss. 2001. On spectral clustering: Analysis and an algorithm. Advances in neural information processing systems 14 (2001).
[34] Aleksandr V Petrov and Craig Macdonald. 2023. Generative Sequential Recommendation with GPTRec. arXiv preprint arXiv:2306.11114 (2023).
[35] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. The Journal of Machine Learning Research 21, 1 (2020), 5485–5551.
[36] Steffen Rendle, Christoph Freudenthaler, Zeno Gantner, and Lars Schmidt-Thieme. 2009. BPR: Bayesian personalized ranking from implicit feedback. In Proceedings of the Twenty-Fifth Conference on Uncertainty in Artificial Intelligence. 452–461.
[37] Rico Sennrich, Barry Haddow, and Alexandra Birch. 2016. Neural Machine Translation of Rare Words with Subword Units. In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 1715–1725.
[38] Damien Sileo, Wout Vossen, and Robbe Raymaekers. 2022. Zero-shot recommendation as language modeling. In European Conference on Information Retrieval. Springer, 223–230.
[39] Fei Sun, Jun Liu, Jian Wu, Changhua Pei, Xiao Lin, Wenwu Ou, and Peng Jiang. 2019. BERT4Rec: Sequential recommendation with bidirectional encoder representations from transformer. In Proceedings of the 28th ACM international conference on information and knowledge management. 1441–1450.
[40] Weiwei Sun, Lingyong Yan, Xinyu Ma, Pengjie Ren, Dawei Yin, and Zhaochun Ren. 2023. Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agent. arXiv preprint arXiv:2304.09542 (2023).
[41] Jiaxi Tang and Ke Wang. 2018. Personalized top-n sequential recommendation via convolutional sequence embedding. In Proceedings of the eleventh ACM international conference on web search and data mining. 565–573.