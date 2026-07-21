# Methodologies for Improving Modern Industrial Recommender Systems

> Shusen Wang

3
2
0
2

本文介绍了 。核心内容：


关键发现：


l
u
J

1
2

]

R

I
.
s
c
[

1
v
4
0
2
1
0
.
8
0
3
2
:
v
i
X
r
a

提升现代工业级推荐系统的方法论

王树森
wssatzju@gmail.com


---

## 摘要

推荐系统是一项成熟的技术，在社交媒体、电子商务、娱乐等领域取得了成功的应用。推荐系统确实是许多流行APP（如YouTube、抖音、小红书、哔哩哔哩等）成功的关键。本文探讨了改进现代工业级推荐系统的方法论。本文是为那些勤奋工作以改善其关键性能指标（如留存率和时长）的经验丰富的推荐系统工程师而写的。本文分享的经验已在一些真实的工业级推荐系统中得到验证，并很可能推广到其他推荐系统。本文大部分内容是没有公开发表参考文献的行业经验。

关键词：推荐系统，召回，排序，深度学习

1. 评估指标

本文详细阐述了改进工业级推荐系统的方法论。业界如何衡量一个实验是让推荐系统变得更好还是更差？

• 最重要的评估指标是流量和留存率。流量通过日活跃用户数和月活跃用户数来衡量。用户留存率可以通过多种指标来评估；其中，LT7和LT30最近已成为最广泛接受的留存指标。在电子商务中，商品交易总额可能与用户数量和用户留存率同等重要。

• 时长、展示次数和点击次数是推荐系统的重要评估指标，尽管它们不如流量和留存率关键。时长是指APP总使用时间（通常以分钟计）除以活跃用户数。时长与留存率强相关。展示次数（或点击次数）在很大程度上决定了每位用户的广告收入。

• 用户生成内容平台重视其创作者，因为平台依赖他们创建新内容来吸引用户。每日或每周发布内容的数量是一个至关重要的评估指标。在YouTube和抖音等平台上，普通用户也可以成为创作者。参与率，定义为日活跃创作者除以日活跃用户数，衡量普通用户发布内容的意愿。

• 其他指标，如点击率、互动率和不喜欢率，也会被监控。在决定是否全量发布一个实验时，除非这些指标受到很大程度的负面影响，否则不会考虑它们。

一个实验在增加一项指标的同时减少另一项指标是很常见的。如果LT增长且日活跃用户数至少持平，那么任何其他指标的下降都将被容忍。然而，如果日活跃用户数和LT7都持平，则将根据具体情况决定是否全量发布该实验。

尽管LT很流行，但许多读者可能对它并不熟悉。在此，我对LT7做一些解释；LT30的定义类似。假设一个用户在第T天登录APP。在从T到T+6的7天中，该用户在不同的4天登录了APP。在第T天，该用户的LT7等于4。对所有活跃用户取平均值，我们就得到该天的平均LT7。自然地，LT7的取值范围是从最小值1到最大值7。

如果用户的整体满意度大幅提高，那么日活跃用户数和LT都会增加。然而，如果用户的整体满意度保持不变，而不活跃用户的满意度恶化，日活跃用户数将下降，但LT会增加。这是因为在计算LT时，分母的下降比分子的下降更显著。因此，仅凭LT的增长并不一定表示用户满意度提高；日活跃用户数必须增长或至少保持持平。

2. 召回

召回，也称为候选生成，是推荐系统流程中的第一步。一个当代推荐系统通常由20多个召回模型组成。双塔模型和item到item模型是最主要的召回模型家族，共同占总配额的超过一半。虽然其他召回模型的配额要小得多，但加入这些模型可以改善留存率等指标。

同一个召回模型可以应用于多个item池。一个召回通道被定义为一个应用于某个item池的召回模型。一个现代推荐系统可能包含50多个召回通道。每个召回通道都有一个特定的配额；例如，应用于过去7天内发布的新item的ItemCF模型有5%的配额。值得注意的是，配额在不同用户群体之间可能有所不同。

主要有四种增强召回过程的方式：改进双塔模型、增强item到item模型、添加新的召回模型，以及为特定目的引入新的item池。在本节中，我们将讨论前三种方法，最后一种方法将在后续章节中再次讨论。

2.1 双塔模型

首先且最重要的，推荐系统工程师应该仔细检查正样本和负样本，特别关注它们的来源和配额。训练样本的正确混合对于双塔模型至关重要。

• 简单正样本由有点击的用户-item对组成，而简单负样本是随机组合的用户-item对。（一个用户对随机item表现出兴趣的概率非常低。这就是为什么随机对是简单负样本。）

• 困难样本介于简单正样本和简单负样本之间。被召回但未通过粗排的item可以作为困难负样本。使用适当的困难负样本对于提升双塔模型的性能至关重要。简单负样本、困难负样本和正样本之间的平衡应该仔细调整。

• 一个常见错误是将有曝光但没有点击的item用作负样本。相反，这些可以用作正样本。一个被排序选中的item很可能符合用户的兴趣，即使它没有被点击。

有比普通双塔模型更好的神经网络架构。

• 标准的用户塔和item塔是简单的多层感知机。先进的架构如DCN-V2比多层感知机效果更好。

• 用户历史上交互过的item（我们称之为last-n）可以作为用户塔的输入。读者可以参考Google的论文。

• 标准的用户塔输出一个向量。它与item向量的点积估计用户对该item感兴趣的程度。我们称这种模型为单向量双塔模型。多向量双塔模型在实践中更好。其用户塔输出多个向量，每个向量负责估计一个目标，如点击、点赞、关注、分享、评论等。

双塔模型可以通过使用改进的损失函数和训练方法来增强。

• 使用批内负样本训练双塔模型自Google的论文以来已经变得普遍。这种方法旨在纠正采样偏差，因为热门item更可能同时成为正样本和负样本。读者可以参考该论文了解更多细节。

• 点击次数较少的item在训练期间作为正样本或负样本出现的可能性较小，导致ID嵌入学习效果不佳。Google的自监督学习方法可以解决这个问题，并改进对不太热门item的嵌入学习。

2.2 item到item到item是基于item-item相似度的一大类召回模型。大多数item到item模型以"用户\rightarrowitem\rightarrowitem"或U2I2I的方式运作。这里，"用户\rightarrowitem"表示一个包含用户交互过item的索引，意味着用户明确表达了他们对这些item的兴趣。"item\rightarrowitem"表示一个将一个item与许多相似item连接起来的索引。有各种各样的item到item模型，它们的区别在于item-item相似度的度量方式。

• item-item相似度可以基于用户行为。如果许多用户与两个item都有交互，则认为这两个item相似。

  - ItemCF是一种经典且广泛使用的方法。Swing是ItemCF的一种变体。两种方法基于相同的概念，仅在item-item相似度的计算上有所不同。
  - 在线ItemCF和在线Swing是在线学习实现，与批量实现相对。
  - 所有四种召回模型都在真实的工业推荐系统中使用。它们彼此之间差异足够大，以至于这四个通道的集成效果优于去掉任何一个召回通道。

• item-item相似度也可以基于从双塔模型或图神经网络获得的item嵌入。

2.3 不太流行的召回模型

一个工业级推荐系统通常包含几个不太重要的召回模型，每个模型的配额小于5%。引入一个新的召回模型（在不增加召回item总数的情况下）可以改善留存率等指标。然而，添加一个新的召回模型不可避免地需要更多的计算资源，例如额外的1000个CPU核心。以下召回模型已被证明是有帮助的。

• U2U2I（用户\rightarrow用户\rightarrowitem）、U2A2I（用户\rightarrow创作者\rightarrowitem）和U2A2A2I（用户\rightarrow创作者\rightarrow创作者\rightarrowitem）的功能类似于U2I2I。

• PDN，由阿里巴巴开发，联合建模了item到item和双塔模型。

• Deep Retrieval，由字节跳动开发，部署在其许多生产线中。它类似于早期的工作TDM，由阿里巴巴开发并部署在其内部。请注意，Deep Retrieval和TDM尚未被任何其他公司复现和验证为有用。

• 各种其他创新的召回方法，如SINE和M2GRL，在实践中也很有用。

• 缓存召回不是一个标准的召回通道。它缓存那些排序得分高但由于重排规则而没有被展示的item。与其他召回通道相比，缓存的item有更高的可能性通过粗排和排序，并且有更大的机会被展示。

• item冷启动严重依赖于基于内容的召回方法，如分类、实体、文本嵌入、图像嵌入、聚类等。这是因为新发布的item与用户的交互为零或很少，使得item到item和双塔方法不适用。

一个真实的工业级推荐系统通常包含数十个召回通道。这些通道共同生成固定数量的item，例如5000个item，然后将其发送给粗排。仔细调整这些召回通道的配额可以改善关键指标。此外，不同的用户群体可能有不同的最优配额。

3. 深入探讨深度学习

让我们更详细地讨论深度学习。深度神经网络应用于推荐系统流程的各个环节。让这些神经网络做出更准确的预测可以从根本上改进推荐系统。

3.1 升级排序模型

在我们之前的讨论中，我们已经涵盖了双塔模型；因此，在本节中，我们将更多地关注排序。我将介绍排序模型的几个改进。一旦这些改进得以实施，仅仅通过改进排序模型来进一步提高留存率等指标将变得困难。

业界各排序模型之间的差异微乎其微；它们基本上类似于Wide&Deep模型。输入包括稀疏特征（例如用户ID、itemID、分类）和稠密特征（例如用户统计和item统计）。模型包含多个头，每个头负责预测一个目标，如点击、点赞、分享、关注、评论等。由于训练数据量巨大且稠密层相对较小（只有几百万个参数），排序模型会出现欠拟合。增加排序模型的深度和宽度通常会导致更高的精度。Wide&Deep模型的深度组件通常有1到6个稠密层，具体取决于（1）推理成本和预测精度之间的权衡，以及（2）训练和推理基础设施（CPU vs GPU）。

大量改进的神经网络架构已被发表。其中一些架构被证明是有效的，而另一些则因未知原因无效。

• 自动特征交叉方法，特别是双线性交叉，可以在不显著增加计算成本的情况下提高排序模型的精度。双线性交叉不仅对推荐系统排序有效，对搜索排序和广告排序也有效。LHUC，也称为参数个性化网络，是另一种特征交叉方法，可以提高排序模型精度，但需要显著更多的计算。参数和嵌入个性化网络是PPNet的升级版本。

• 排序模型是多任务的。它们的每个头估计一个目标，如点击、点赞、分享、关注、评论等。一个效用函数将这些得分作为输入，并输出一个主要决定item排序的单一得分。推荐系统工程师不断寻找与用户兴趣或item质量相关的新目标。当他们发现新目标时，他们会在排序模型中添加一个新的头，并将预测得分作为新的输入加入效用函数。

• 多门混合专家和渐进式分层提取是多任务学习中众所周知的改进方法。然而，一个常见的误解是用MMoE或PLE替换Wide&deep会立即带来更好的预测精度。根据我的经验和许多推荐系统工程师的经验，在保持计算成本不变的情况下，MMoE和PLE并不优于Wide&Deep。虽然MMoE和PLE在某些情况下可能有帮助，但它们并不普遍优于Wide&Deep。

• 位置偏置在推荐系统排序和搜索排序中普遍存在。许多论文研究了位置偏置并开发了去偏技术。我在推荐系统和搜索场景中都观察到了严重的位置偏置。尽管我的许多同事尝试过去偏，但没有人成功。

通过改进神经网络架构来改善留存率等指标是本文讨论的最具挑战性的方法之一。最终，神经网络架构进一步改进的潜力会变得非常有限。

建模用户的last-n序列是一种更容易的改进排序模型的方法。用户最近交互的n个item是其兴趣的强指示器。

• 我们可以简单地平均n个itemID嵌入，并将结果向量作为排序模型的输入向量。当n较小时，这种朴素平均方法效果良好。

• 深度兴趣网络，本质上是一个单头注意力层，将目标item（待排序的候选item）作为查询，用户的last-nitem作为键和值。

• 基于搜索的用户兴趣建模是业界长序列建模的主流方法。它首先使用目标item的分类从用户的last-nitem中剔除不相关的item。然后对剩余的n′个item应用DIN。

• 一个公认的事实是，更大的n会导致更准确的预测，但代价是增加的计算和通信成本。增加n的挑战主要在于基础设施方面，而非机器学习模型方面。

粗排模型是排序模型的轻量级版本。与排序相比，粗排处理的候选item多10倍，而每个item的计算量少10倍。最简单的粗排模型可能是多向量双塔模型，它同时估计点击、点赞、分享和所有其他目标。COLD，一个三塔模型，比双塔模型准确得多；然而，它对基础设施的要求也高得多。

保持粗排和排序之间的一致性对于整体性能至关重要。如果粗排和排序显著不同，本应排序高的item可能会被粗排淘汰。

• 在训练期间，我们不让粗排模型拟合真实标签y，而是让拟合(y+p)/2，其中p是排序模型的预测。例如，如果y=1表示用户实际点击了item，预测的CTR为p=0.6，我们使用(y+p)/2=0.8而不是y=1作为训练标签。

• 粗排模型的列表式训练在业界也很普遍。给定一个包含k个候选item的列表，我们根据排序模型的预测对item进行排序。然后，我们让粗排模型拟合这k个item的顺序。这种方法类似于搜索引擎的"学习排序"。

• 尽管上述方法改善了留存率等指标，但它们通过将排序模型的错误传播到粗排，降低了系统的可靠性。当排序模型出错时（也许由于基础设施故障），粗排训练数据可能被污染。

3.2 在线学习

排序、粗排和双塔召回模型至少需要每天更新一次。例如，在午夜，过去24小时内发生的曝光、点击和互动被用来创建训练样本。然后，模型基于前一天的检查点增量训练一个epoch。

在线学习以更高的频率更新模型。在线学习有两个主要好处。首先，用户的最新兴趣不仅被他们的last-nitem捕获，还被他们的用户ID嵌入捕获，从而产生更符合用户最新兴趣的推荐。其次，新发布item的ID嵌入使用默认向量初始化，这可能导致对点击和互动的不准确预测。在点击和互动发生后不久更新新item的ID嵌入将使后续推荐更准确。

尽管有这些好处，在线学习也有几个缺点。

• 一个缺点是在线学习容易受到基础设施故障的影响。这种脆弱性是因为在线学习依赖于流式数据管道，并且需要频繁训练和部署模型。

• 一个更严重的缺点是在线学习会显著拖慢模型开发。例如，如果推荐系统包含一个留置模型、一个基线模型和四组实验模型，则需要六套昂贵的CPU/GPU集群来在线训练这六个模型。请注意，即使有六套CPU/GPU集群，也只能同时实验四个不同的模型。

在一个不成熟的推荐系统中实现在线学习可能是有害的。它会严重阻碍模型开发并引入大量稳定性问题。总之，只有当你的模型已经达到高水平的性能并且难以进一步改进时，才考虑部署在线学习。

4. 增加多样性

增强展示item的多样性将带来用户满意度的提升。我相信经验丰富的推荐系统工程师知道如何在最终排序中鼓励多样性。事实上，有方法可以在整个流程中提升多样性，包括召回、粗排和排序阶段。

4.1 召回中的多样性

上述的双塔和U2I2I召回模型是确定性的。然而，在实践中，引入一些随机性可以增强item多样性，从而改善留存率等指标。

首先，让我们讨论双塔召回模型。在召回过程中，用户塔以用户特征为输入，并生成用户的向量表示。

• 用户向量用于从向量数据库中检索item。令人惊讶的是，向向量注入随机高斯噪声实际上可以改善推荐系统的关键性能指标！更具体地说，如果用户的兴趣范围狭窄，例如，用户的last-nitem只覆盖少量分类，那么随机噪声应该更强。

• 用户塔以用户最近交互的n个item为输入。我们不使用所有item，而是保留最近的r个item，并从剩余的n-r个item中随机采样一个子集。这种方法为用户的向量表示引入了额外的随机性。

对于U2I2I召回模型，我们应该专注于多样化前一个"I"，即用户最近交互的n个item。这n个item可能集中在少数几个分类中，例如，n个item中的90%只属于3个分类。这可能导致召回的item缺乏多样性。

为了解决这个问题，我们可以对每个分类设置大小限制，并相应地进行下采样。例如，我们可以将大小限制为10个item。在最近的n个item中，假设30个item属于"电影"分类，6个属于"体育"分类。在这种情况下，我们会从"电影"分类中随机采样10个item，丢弃剩余的20个，同时保留所有6个"体育"item。

4.2 排序中的多样性

排序和粗排主要基于预测的点击和互动概率，次要基于多样性得分和规则。在本次讨论中，我们将探讨排序和粗排中两种多样性方法。

• 最大边际相关性和行列式点过程是最流行的增加多样性的方法。设i是一个item，d_i是由MMR或DPP计算的多样性得分，s_i是由效用函数输出的得分（通过结合点击和互动概率）。我们使用s_i或d_i+s_i来排序候选item。排序和粗排存在一些微妙的差异：

  - 在排序中，我们总是使用d_i + s_i来排序候选item。在粗排中，我们首先使用s_i来排序候选item，并选择最符合用户兴趣的顶部item。然后我们使用d_i + s_i来排序剩余item并选择顶部item。

  - 滑动窗口是MMR和DPP的一个可选设置。它在排序中使用，但在粗排中不使用。

• MMR和DPP也被称为"软打散"。在排序中，我们需要"硬打散"，以确保相邻展示item不会过于相似。例如，来自同一分类的item必须至少相隔五个item。除了分类，我们还使用item聚类来打散item。具体来说，我们可以使用文本、图像和视频的嵌入来将所有item划分为1000个簇，离线执行。在排序期间，同一簇中的item必须至少相隔五个item。

除了上述多样性方法，现代推荐系统可能会预留2%的展示位用于兴趣探索。例如，我们基于item内容质量、创作者质量和在同一用户群体上计算的互动统计，为30至40岁的男性用户维护一个高质量item池。在为这类用户做推荐时，2%的展示item从这个item池中选择。

由于弱个性化，兴趣探索可能在短期内略微降低留存率。然而，从长期来看，留存率将逐渐恢复并最终超过对照组。

5. 关注特殊用户群体

新用户和不活跃用户需要特别关注，原因有二。首先，这些用户流失的可能性更高。对他们来说，推荐系统的主要目标是留存，而时长、阅读、购买和广告点击则完全不考虑。其次，这些用户的行为与长期用户显著不同。一方面，在所有用户数据上训练的排序模型在应用于新用户和不活跃用户时往往表现不佳。另一方面，可以采用特定策略来吸引和留住这些用户。

5.1 专门的item池

新用户和不活跃用户通常点击和互动次数有限，这可能导致个性化推荐不够准确。为了提高他们的满意度，重点应放在推荐高质量item上，这些item可能匹配也可能不匹配他们的特定兴趣。

可以根据人口统计数据将用户分组。例如，30至40岁的男性可以组成一个用户组，并创建一个可能匹配他们兴趣的item池。选择高质量item的最简单方法是使用互动统计，如点赞率、关注率、分享率和评论率。更先进的技术是因果推断，它识别有助于用户留存的内容或统计特征。

尽管可能有几个item池，但同一个双塔模型可以应用于所有池。不需要为每个item池单独训练一个双塔模型。在推理过程中，item塔计算item嵌入，并为每个item池构建一个近似最近邻索引。由于每个item池比主池小数个数量级，为每个item池构建近似最近邻索引相对便宜。

5.2 专门的排序策略

有几种保护性策略可以提升新用户和不活跃用户的满意度。广告和其他低质量item的数量应严格控制，甚至减少到零。从这些用户身上盈利完全没有必要；相反，重点应放在防止他们流失上。

新发布的item通常缺乏用户交互，可能不会被推荐给正确的受众。在排序过程中提升新item会增加它们被看到的机会。经过几次尝试后，推荐系统可以识别哪些用户对此类新item感兴趣。然而，由于初始推荐可能不准确，最好避免在新用户和不活跃用户身上测试它们。对于这些用户，新item在排序过程中不应被提升，确保它们与旧item受到同等对待。

很大一部分新用户和不活跃用户甚至可能不愿意点击任何item，更不用说点赞、分享、评论或关注了。鼓励这些用户点击至关重要，因为不感兴趣可能导致流失。因此，在排序效用函数中，对于新用户和不活跃用户，点击的权重应设置得比长期用户更高。

5.3 专门的排序模型

特殊用户的行为与大多数用户明显不同。例如，他们的点击率可能低于典型用户，而他们的点赞率往往更高。当排序模型在所有用户的数据上训练时，它可能被活跃用户的行为主导，导致对特殊用户的点击率高估和点赞率低估。可以使用几种方法来校准排序模型的预测。

• 可以使用两个排序模型：一个针对所有用户的大模型和一个针对特殊用户的小模型。两个模型都是具有大规模稀疏嵌入的深度神经网络，但它们在大小上显著不同。

  - 大模型在所有用户的数据上训练。对活跃用户的预测仅使用这个模型。
  - 针对特殊用户的小模型被训练来拟合残差。对于一个特殊用户，设y为真实标签，p为大模型的预测。小模型被训练来拟合差值y-p。当为特殊用户排序时，两个模型的预测被结合起来。

• 或者，可以使用梯度提升决策树进行校准。大模型预测点击率和互动率。然后小模型以预测率和用户特征作为输入，并输出校准后的预测率。

• 可以使用一个具有多个专家的单一大规模深度神经网络（类似于MMoE）。专家的权重取决于用户的特征。

一个常见的陷阱是为每个用户组分配单独的排序模型。在短期内，这种方法可以产生更准确的预测，从而改善留存率等关键指标。然而，维护多个排序模型会增加更新推荐系统的复杂性。在添加一个特征或更新神经网络架构后，必须对每个模型进行A/B测试。如果推荐系统工程师持续更新主模型而忽视特殊用户群体的模型，这些模型最终会变得过时，导致其性能相比主模型下降。

6. 用户互动

三种关键的互动行为——关注、分享和评论——是推荐系统中的金矿。在先进的工业级推荐系统中，深度神经网络模型在不产生显著资源成本的情况下很难得到改进。近年来，最先进的工业级推荐系统主要通过利用这三种互动行为来改进，而不是更新深度神经网络模型。

6.1 关注

在社交媒体推荐系统中，如果用户发现内容有趣，可以选择关注创作者。对于一个新用户，他们关注的创作者数量与他们的留存率呈正相关。当f较低时，f和r之间的关系尤为强。当f达到足够大的值时，增加f对r的影响很小。有几种方法可以通过利用关注行为来增强新用户的留存，我们将讨论其中两种：

• 向效用函数添加一个额外项w(f)*p。这里，w(f)表示随着f增加而减小的权重，p表示用户关注创作者的概率。如果新用户没有关注很多创作者，推荐系统会尝试鼓励更多的关注行为。

• 维护一个具有高关注率（#关注/#点击和#关注/#曝光）的item池。对于新用户，从这个池中召回一些item。用户在观看item后，可能会关注创作者。随着他们关注更多创作者，用户流失的可能性降低。

建模关注关系不仅对新用户有益，对所有用户普遍有益。通常，用户更有可能点击他们关注的创作者发布的item。

• 关注关系可以在召回中使用，例如用户\rightarrow创作者\rightarrowitem和用户\rightarrow创作者\rightarrow创作者\rightarrowitem。

• 有时，用户U极有可能点击创作者A发布的item，但由于某些未知原因，U没有点击A个人主页上的关注按钮。我们称之为隐式关注。隐式关注的创作者数量可以是显式关注的好几倍。隐式关注关系在召回中非常有用。

用户生成内容平台，如抖音，鼓励创作者发布内容。每日发布内容数量是用户生成内容推荐系统的关键评估指标之一。可以利用关注关系来激励创作者发布更多内容。特别是，当创作者粉丝较少时，粉丝数量的增加可以强烈鼓励他们发布内容。推荐系统应帮助这些创作者吸引更多粉丝。例如，如果此类创作者发布了一个item，预测的关注率应被赋予更高的权重。

6.2 分享

YouTube用户可能会在Twitter上分享视频，这为YouTube带来新的流量甚至新的用户注册。中国的一些社交媒体推荐系统正在积极努力促进分享以增加他们的日活跃用户数。

通过增加效用函数中预测分享率（#分享/#点击）的权重，分享次数自然会增加。是否提高预测分享率的权重取决于用户是否是关键意见领袖。

• 如果用户是KOL，例如在Twitter上有10万粉丝，他在Twitter上分享YouTube视频可以为YouTube带来大量流量。如果推荐系统知道该用户是KOL，它应该通过提高分享权重来利用用户的价值。

• 一个用户可能在Twitter上有10万粉丝，但在YouTube上零粉丝。YouTube如何识别KOL？答案很简单：如果用户过去的分享产生了显著流量，他很可能是KOL。估计的KOL得分应记录在用户画像中。

此外，可以建立一个可能在外部社交媒体平台上被分享的item池。当向KOL推荐内容时，从这个池中额外召回item以鼓励分享。

6.3 评论

在社交媒体平台上，用户经常在他们喜欢或不喜欢的item上留下评论。近期的行业研究（不会发表）表明评论在推荐系统中具有重要价值。在此，我讨论一些初步发现。

评论可以显著鼓励创作者发布更多内容。例如，假设你制作了一个视频并上传到YouTube。如果你的视频迅速收到几条正面评论，你会有动力制作更多视频。事实上，关注和评论是鼓励创作者发布内容的最重要的互动行为。在排序中，如果一个item已经被互动过（例如点赞和分享）但还没有被评论，则效用函数中预测评论率的权重应被提高。

经常留下评论的用户流失的可能性较小。具体来说，一些用户有充足的闲暇时间，愿意留下评论。推荐系统应该给这些用户更多评论和参与讨论的机会。为了实现这一点，可以维护一个具有高评论数量和评论率的item池。这些用户可以获得基于该item池的定制化召回通道，提供更多互动的机会。

6.4 分享（续）

1. 免责声明：我以YouTube为例。我不知道YouTube是否有任何专门鼓励用户在其他平台上分享视频的机制。

7. 结论

本文总结了关于如何改进工业级推荐系统的理解。在开发推荐系统的早期阶段，应集中精力改进模型。没有对点击和互动的可靠预测，本文介绍的许多方法将无法有效运作。一旦双塔和item到item召回模型、三塔粗排模型以及Wide&Deep多任务排序模型建立起来，推荐系统就可以被认为是现代的。

经过多次更新后，通过特征工程和改进的神经网络架构来改进粗排和排序模型变得越来越困难。这时，是时候考虑在线学习和长序列建模（例如SIM）了；这些方法将显著提高模型预测的准确性。是否实施它们取决于你的基础设施强度以及你如何平衡成本和收益。

最终，特别是在上线在线学习之后，进一步提升排序和粗排模型变得极其困难。在这种情况下，你可能需要依赖于引入更先进的召回模型和item池、增强多样性、对特定用户群体实施特殊处理以及充分利用用户互动。

## 参考文献

Jaime Carbonell and Jade Goldstein. The use of MMR, diversity-based reranking for reordering documents and producing summaries. In Proceedings of the 21st annual international ACM SIGIR conference on Research and development in information retrieval, pages 335–336, 1998.

Jianxin Chang, Chenbin Zhang, Yiqun Hui, Dewei Leng, Yanan Niu, and Yang Song. PEPNet: Parameter and embedding personalized network for infusing with personalized prior information. arXiv preprint arXiv:2302.01115, 2023.

Laming Chen, Guoxin Zhang, and Eric Zhou. Fast greedy map inference for determinantal point process to improve recommendation diversity. Advances in Neural Information Processing Systems, 31, 2018.

Heng-Tze Cheng, Levent Koc, Jeremiah Harmsen, Tal Shaked, Tushar Chandra, Hrishi Aradhye, Glen Anderson, Greg Corrado, Wei Chai, Mustafa Ispir, et al. Wide & deep learning for recommender systems. In Proceedings of the 1st workshop on deep learning for recommender systems, pages 7–10, 2016.

Paul Covington, Jay Adams, and Emre Sargin. Deep neural networks for youtube recommendations. In Proceedings of the 10th ACM conference on recommender systems, pages 191–198, 2016.

Weihao Gao, Xiangjun Fan, Chong Wang, Jiankai Sun, Kai Jia, Wenzi Xiao, Ruofan Ding, Xingyan Bin, Hui Yang, and Xiaobing Liu. Learning an end-to-end structure for retrieval in large-scale recommendations. In Proceedings of the 30th ACM International Conference on Information & Knowledge Management, pages 524–533, 2021.

Tongwen Huang, Zhiqi Zhang, and Junlin Zhang. FiBiNET: combining feature importance and bilinear feature interaction for click-through rate prediction. In Proceedings of the 13th ACM Conference on Recommender Systems, pages 169–177, 2019.

Houyi Li, Zhihong Chen, Chenliang Li, Rong Xiao, Hongbo Deng, Peng Zhang, Yongchao Liu, and Haihong Tang. Path-based deep network for candidate item matching in recommenders. In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 1493–1502, 2021.

Jiaqi Ma, Zhe Zhao, Xinyang Yi, Jilin Chen, Lichan Hong, and Ed H Chi. Modeling task relationships in multi-task learning with multi-gate mixture-of-experts. In Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining, pages 1930–1939, 2018.

Qi Pi, Guorui Zhou, Yujing Zhang, Zhe Wang, Lejian Ren, Ying Fan, Xiaoqiang Zhu, and Kun Gai. Search-based user interest modeling with lifelong sequential behavior data for click-through rate prediction. In Proceedings of the 29th ACM International Conference on Information & Knowledge Management, pages 2685–2692, 2020.

Pawel Swietojanski, Jinyu Li, and Steve Renals. Learning hidden unit contributions for unsupervised acoustic model adaptation. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 24(8):1450–1463, 2016.

Qiaoyu Tan, Jianwei Zhang, Jiangchao Yao, Ninghao Liu, Jingren Zhou, Hongxia Yang, and Xia Hu. Sparse-interest network for sequential recommendation. In Proceedings of the 14th ACM International Conference on Web Search and Data Mining, pages 598–606, 2021.

Hongyan Tang, Junning Liu, Ming Zhao, and Xudong Gong. Progressive layered extraction (PLE): A novel multi-task learning (mtl) model for personalized recommendations. In Proceedings of the 14th ACM Conference on Recommender Systems, pages 269–278, 2020.

Menghan Wang, Yujie Lin, Guli Lin, Keping Yang, and Xiao-ming Wu. M2GRL: A multi-task multi-view graph representation learning framework for web-scale recommender systems. In Proceedings of the 26th ACM SIGKDD international conference on knowledge discovery & data mining, pages 2349–2358, 2020a.

Ruoxi Wang, Rakesh Shivanna, Derek Cheng, Sagar Jain, Dong Lin, Lichan Hong, and Ed Chi. DCN v2: Improved deep & cross network and practical lessons for web-scale learning to rank systems. In Proceedings of the web conference 2021, pages 1785–1797, 2021.

Zhe Wang, Liqin Zhao, Biye Jiang, Guorui Zhou, Xiaoqiang Zhu, and Kun Gai. COLD: Towards the next generation of pre-ranking system. arXiv preprint arXiv:2007.16122, 2020b.

Xiaoyong Yang, Yadong Zhu, Yi Zhang, Xiaobo Wang, and Quan Yuan. Large scale product graph construction for recommendation in e-commerce. arXiv preprint arXiv:2010.05525, 2020.

Tiansheng Yao, Xinyang Yi, Derek Zhiyuan Cheng, Felix Yu, Ting Chen, Aditya Menon, Lichan Hong, Ed H Chi, Steve Tjoa, Jieqi Kang, et al. Self-supervised learning for large-scale item recommendations. In Proceedings of the 30th ACM International Conference on Information & Knowledge Management, pages 4321–4330, 2021.

Xinyang Yi, Ji Yang, Lichan Hong, Derek Zhiyuan Cheng, Lukasz Heldt, Aditee Kumthekar, Zhe Zhao, Li Wei, and Ed Chi. Sampling-bias-corrected neural modeling for large corpus item recommendations. In Proceedings of the 13th ACM Conference on Recommender Systems, pages 269–277, 2019.

Zhe Zhao, Lichan Hong, Li Wei, Jilin Chen, Aniruddh Nath, Shawn Andrews, Aditee Kumthekar, Maheswaran Sathiamoorthy, Xinyang Yi, and Ed Chi. Recommending what video to watch next: a multitask ranking system. In Proceedings of the 13th ACM Conference on Recommender Systems, pages 43–51, 2019.

Guorui Zhou, Xiaoqiang Zhu, Chenru Song, Ying Fan, Han Zhu, Xiao Ma, Yanghui Yan, Junqi Jin, Han Li, and Kun Gai. Deep interest network for click-through rate prediction. In Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining, pages 1059–1068, 2018.

Han Zhu, Xiang Li, Pengye Zhang, Guozheng Li, Jie He, Han Li, and Kun Gai. Learning tree-based deep model for recommender systems. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pages 1079–1088, 2018.
