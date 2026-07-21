# MV-DNN: A Multi-View Deep Learning Approach for Cross Domain User Modeling in Recommendation Systems

> Ali Elkahky, Yang Song, Xiaodong He | 哥伦比亚大学计算机科学系 & 微软研究院



本文提出了一种基于多视图深度学习的内容推荐系统，用于**跨领域用户建模**。核心内容：

- 利用丰富的用户浏览和搜索历史特征构建用户表示
- 提出Multi-View DNN（MV-DNN）模型，将DSSM扩展到**多领域联合学习**
- 通过降维和训练样本压缩技术实现系统可扩展性

关键发现：MV-DNN在三个**真实推荐系统**（Windows Apps推荐、新闻推荐、Movie/TV推荐）上的表现显著优于现有方法，对现有用户提升达49%，对新用户提升达115%。

---



## 摘要

最近的在线服务严重依赖自动个性化技术向大量用户推荐相关内容。这要求系统能够快速扩展以适应首次访问在线服务的新用户流。在这项工作中，我们提出了一个基于内容的推荐系统，以同时解决推荐质量和系统可扩展性问题。我们提出使用丰富的特征集来表示用户，这些特征来自用户的Web浏览历史和搜索查询。我们使用深度学习方法将用户和item映射到一个latent空间，在该空间中用户与其偏好的item之间的相似性被最大化。我们通过引入多视图深度学习模型，将该模型扩展到联合学习来自不同领域的item特征和用户特征。我们展示了如何通过降低输入维度和训练数据量，使这种基于丰富特征的用户表示具有可扩展性。丰富的用户特征表示允许模型学习相关的用户行为模式，并为那些虽与服务没有任何交互但拥有足够搜索和浏览历史的用户提供有用的推荐。将不同领域组合到单一学习模型中有助于提高所有领域的推荐质量，并得到更紧凑、语义更丰富的用户latent特征向量。我们在来自Microsoft产品不同来源的三个真实推荐系统上进行了实验：Windows Apps推荐、新闻推荐和Movie/TV推荐。结果表明，我们的方法显著优于现有算法（对现有用户提升高达49%，对新用户提升高达115%）。此外，在公开数据集上的实验也表明，与传统生成式主题模型相比，我们的方法在跨领域推荐系统建模方面具有优越性。可扩展性分析表明，我们的多视图DNN模型可以轻松扩展到覆盖数百万用户和数十亿item条目。**实验结果还证实，结合所有领域的特征比分别为每个领域构建单独模型产生更好的性能**。



## 通用术语

User Modeling



## 关键词

User Modeling; Recommendation System; Multi-View Learning; Deep Learning



## 1. 引言

推荐系统和内容个性化在现代在线Web服务中扮演着越来越重要的角色。许多最近的Web服务致力于为用户找到最相关的内容，以最大化用户与网站的互动，并最小化查找相关内容的时间。完成此任务的一种主要方法称为协同过滤（CF）[3, 19, 22, 21, 23, 6]，它利用用户之前在网站上的交互历史来预测最相关的内容进行推荐。另一种常见方法是基于内容的推荐[14, 15]，它使用关于item和/或用户的特征，基于特征之间的相似性向用户推荐新item。虽然这两种方法在许多实际应用[6]中工作良好，但它们通常面临一定的限制和挑战，尤其是在个性化和推荐质量需求日益增长的情况下。

具体来说，CF需要相当数量的先前交互历史才能给出高质量的推荐。这个问题被称为用户冷启动问题[24]。在新建立的在线服务中，这个问题变得更加严重，因为用户几乎没有与该服务的交互历史。因此，传统的CF方法通常无法为新用户产生高质量的推荐。另一方面，基于内容的推荐方法从每个用户和/或item中提取特征，并使用这些特征进行推荐。例如，如果两篇新闻文章$N_i$和$N_j$共享相同的主题，并且用户喜欢文章$N_i$，系统可能会向用户推荐文章$N_j$。类似地，如果两个用户$U_i$和$U_j$在某些相似性上（如位置、年龄或性别）有共同点，系统可能会向用户$U_j$推荐用户$U_i$之前喜欢的item。在实践中，研究表明基于内容的方法可以很好地处理新item的冷启动问题[32]。然而，其在应用于新用户推荐时的有效性是值得怀疑的，因为用户级别的特征通常更难获取，并且通常从用户在线个人资料的有限信息中生成，这些信息未能准确捕捉用户的真实兴趣。

为了解决这些局限性，我们提出了一个同时利用用户和item特征的推荐系统。为了构建用户特征，与许多基于用户画像的方法不同，我们提出从用户的浏览和搜索历史中提取丰富的特征来建模用户的兴趣。其基本假设是，用户的历史在线活动在很大程度上反映了用户的背景和偏好，因此提供了关于用户可能感兴趣的item和主题的精确洞察。例如，一个有许多与婴儿相关查询和网站访问（如toysrus.com）的用户可能表明她是一位新生婴儿的母亲。有了这些丰富的用户在线活动，推荐相关item可以更高效、更有效地实现。

在我们的工作中，我们提出了一种新颖的深度学习方法，从深度结构化语义模型（DSSM）[9]扩展而来，将用户和item映射到一个共享的语义空间，并推荐在映射空间中与用户具有最大相似性的item。为实现这一点，我们的模型将用户和item（每个都由丰富的特征集表示）通过非线性变换层投影到一个紧凑的共享latent语义空间，在该空间中，用户的映射与该用户喜欢的item的映射之间的相似性被最大化。这使得模型能够学习有趣的映射，例如访问fifa.com的人可能喜欢阅读关于世界杯的新闻文章，并在PC或Xbox上玩足球游戏。用户侧的丰富特征使得对用户行为的建模成为可能，从而克服了基于内容推荐的许多局限性。它还有效地解决了用户冷启动问题，因为该模型允许我们从查询中捕捉用户兴趣并推荐相关item（比如音乐），即使他们没有使用音乐服务的任何历史。我们的深度学习模型具有基于排序的目标函数，旨在将正例（用户喜欢的item）排在负例之上。这种基于排序的目标函数已被证明对推荐系统更有效[9]。

此外，我们将原始的DSSM模型（在本文中称为单视图DNN，因为它从来自单个领域的用户特征和item中学习）扩展到联合学习来自不同领域的item特征。我们将新模型命名为多视图深度神经网络（MV-DNN）。在文献中，多视图学习是一个研究充分的领域，它从不共享共同特征空间的数据中学习[27]。我们将MV-DNN视为多视图学习设置中的一种通用深度学习方法。具体来说，在我们的新闻、Apps和Movie/TV日志数据集中，我们不是为每个领域构建单独模型（这些模型天真地将用户特征映射到该领域内的item特征），而是构建一个新颖的多视图模型，该模型在latent空间中发现用户特征的单一映射，使得它与来自所有领域的item特征共同优化。MV-DNN允许我们学习一个更好的用户表示，该表示利用了跨领域的更多数据，并以原则性的方式通过利用所有领域的用户偏好数据来解决数据稀疏性问题。我们在实验中展示，这种多视图扩展同时提高了所有领域的推荐质量。此外，值得提到的是，深度学习模型中的非线性映射使我们能够在latent空间中找到一个紧凑的用户表示，这使得存储学习到的用户映射和在不同任务之间共享信息变得更加容易。

使用深度学习建模丰富用户特征的一个挑战是特征空间的高维性，这使得学习效率低下，并可能影响模型的泛化能力。我们提出了几种有效且可扩展的降维技术，可以在不损失太多信息的情况下将维度降低到合理的大小。

总的来说，本文的贡献是：

（1）使用丰富的用户特征构建通用推荐系统；

（2）提出了一种基于内容的推荐系统的深度学习方法，并研究了不同的系统扩展技术；

（3）引入新颖的多视图深度学习模型，通过组合来自多个领域的数据集来构建推荐系统；

（4）利用从多视图DNN模型学习到的语义特征映射，解决了文献中尚未充分研究的用户冷启动问题；

（5）使用四个真实世界的大规模数据集进行了严格的实验，并展示了所提出的系统以显著的优势优于现有方法。

本文的其余部分组织如下：首先在第2节中回顾推荐系统的主要方法，包括关注冷启动问题的论文；在第3节中，我们描述了所使用的数据集，并详细介绍了在每个领域中用于建模用户和item的特征类型；随后在第4节中回顾了基本的DSSM模型，并讨论了如何将其扩展到我们的设置；在第5节中，我们详细介绍了多视图深度学习模型并讨论了其优势；在第6节中，我们讨论了用于扩展模型的降维方法；在第7、8、9和10节中，我们呈现了全面的实证研究；最后在第11节中总结并提出几个未来工作方向。



## 2. 相关工作

关于推荐系统已有大量研究，出版物不计其数。在本节中，我们旨在回顾与我们的方法最相关的一组代表性方法。

总的来说，推荐系统可以分为协同推荐和基于内容的推荐。协同推荐系统如果相似用户喜欢某个item，则向用户推荐该item。这种技术的例子包括最近邻建模[3]、矩阵补全[19]、受限玻尔兹曼机[22]、贝叶斯矩阵分解[21]等。本质上，这些方法是用户协同过滤、item协同过滤或item和用户协同过滤。在用户协同过滤中，如[3]，算法基于用户喜欢的item计算用户之间的相似性。然后，通过组合相似用户对该item的评分来计算用户-item对的分数。基于item的协同过滤[23]基于喜欢两个item的用户来计算item之间的相似性，然后向用户推荐与他们之前喜欢的item相似的item。基于用户-item的协同过滤根据用户-item矩阵为item和用户找到一个公共空间，并组合item和用户表示来找到推荐。所有矩阵分解方法如[19]和[21]都是这种技术的例子。CF可以扩展到大规模设置，如[6]中所示。然而，CF通常无法处理新用户和新item，这个问题通常被称为冷启动问题。

推荐系统的第二种方法是基于内容的推荐。这种方法从item和/或用户画像中提取特征，并根据这些特征向用户推荐item。其基本假设是相似用户倾向于喜欢与他们之前喜欢的item相似的item。在[14]中，提出了一种方法，通过构造一个包含用户之前喜欢的item的某些特征的搜索查询来查找其他相关item进行推荐。另一个例子见于[15]，其中每个用户被建模为一个在新闻主题上的分布，该分布从她喜欢的文章构建，并使用通过所有共享相同位置的用户计算的主题偏好的先验分布。这种方法可以处理新item（新闻文章），但对于新用户，系统仅使用位置特征，这意味着新用户将看到其所在位置的最频繁主题。对于新闻推荐来说这可能是一个好的特征，但在其他领域，例如Apps推荐，仅使用位置信息可能无法作为用户偏好的良好先验。

最近，研究人员开发了结合协同推荐和基于内容推荐的方法。在[16]中，作者使用item特征在使用协同过滤之前平滑用户数据。在[7]中，作者使用受限玻尔兹曼机学习item之间的相似性，然后将其与协同过滤相结合。在[32]中开发了一种贝叶斯方法，用于联合学习item在不同组件（主题）上的分布和评分矩阵的分解。

推荐系统中的冷启动问题主要是针对新item（没有任何用户评分的item）进行研究的。正如我们之前提到的，所有基于内容的过滤都可以处理item的冷启动，并且有一些方法专门针对这个问题进行了开发和评估，如[24]和[7]。文献[18]研究了如何通过推荐那些能提供最多用户偏好信息同时最小化推荐不相关内容概率的item，来增量式地学习新用户的偏好。基于丰富特征的用户建模最近得到了大量研究。例如，已有研究表明用户搜索查询可以用于发现用户之间的相似性[25]。来自用户搜索历史的丰富特征也已用于个性化Web搜索[26]。对于推荐系统，作者在[2]中利用用户的历史搜索查询来构建个性化分类法用于推荐广告。另一方面，研究人员发现用户的社会行为也可以用于构建用户画像。在[1]中，作者使用Twitter数据中的用户推文来推荐新闻文章。

大多数传统的推荐系统研究专注于单一领域内的数据。最近，人们对跨领域推荐越来越感兴趣。解决跨领域推荐有不同的方法。一种方法是假设不同领域共享相似的用户集但不共享item，如[20]所述。在他们的工作中，作者通过从具有共同用户的数据集中获取电影和书籍的评分数据来增强数据。增强后的数据集随后用于执行协同过滤。他们表明这特别有助于用户在一个领域中的画像信息较少的情况（冷启动用户）。第二种方法解决了同一组item在不同领域（如用户点击或用户显式评分）共享不同类型反馈的场景。如[17]所示，作者引入了跨领域矩阵分解的坐标系统转换方法。在[12]中，作者研究了在领域之间不存在共享用户或item的情况下的跨领域推荐。他们开发了一个生成模型来发现不同领域之间的公共聚类。然而，他们的方法的一个挑战是由于计算成本，其扩展到中等规模以上数据集的能力。在[28]中引入了一种不同的方法用于作者合作推荐，他们构建了一个主题模型来推荐来自不同研究领域的作者进行合作。

对于许多推荐系统方法，目标函数是最小化用户-item矩阵重构上的均方根误差。最近，基于排序的目标函数已被证明在提供更好的推荐方面更有效，如[11]所示。

深度学习最近被提出用于构建协同和基于内容方法的推荐系统。在[22]中，RBM模型用于协同过滤。基于内容的深度学习推荐已在[30]中完成，其中深度学习用于学习音乐特征的embedding。然后使用此embedding来正则化协同过滤中的矩阵分解。



## 3. 数据集描述

本节介绍数据集。我们描述了数据收集过程、每个数据集的特征表示以及数据的一些基本统计信息。

本研究中使用的四个数据集来自Microsoft产品的用户日志，包括：

1）来自Bing Web垂直搜索引擎的搜索引擎日志，

2）来自Bing News垂直产品的新闻文章浏览历史，

3）来自Windows App Store的应用下载日志

4）来自Xbox的Movie/TV观看日志。

所有日志均在2013年12月至2014年6月期间收集，主要关注英语市场，包括美国、加拿大和英国。

**（用户特征）** 我们从Bing收集了用户的搜索查询及其点击的URL来构成用户特征。查询首先被标准化、词干提取，然后分割为unigram特征；URL被缩短为仅包含域名级别（例如www.linkedin.com）以降低特征维度。然后我们使用TF-IDF分数仅保留最流行和非平凡的特征。总体而言，我们选择了300万个unigram特征和50万个域名特征，得到一个总长度为350万的特征向量。

**（新闻特征）** 我们从Bing News垂直产品收集了新闻文章点击数据。每个新闻item由三部分特征表示。第一部分是使用字母三元组（letter tri-gram）表示编码的标题特征，这将在下一节中描述。其次，每条新闻的顶级类别（例如，娱乐）被编码为二元特征。最后，使用内部专有NLP解析器提取的文章中的命名实体，也使用字母三元组编码。这产生了一个100K的特征向量。

**（Apps特征）** 用户的App下载历史从Windows App Store日志中收集。每个App的标题使用字母三元组表示，结合其类别（例如，游戏）特征的二元格式。由于App描述不断变化的性质，我们决定不将其作为特征空间的一部分。这为Apps产生了一个50K的特征向量。

**（Movie/TV特征）** 从Xbox日志中，我们收集了每个Xbox用户的Movie/TV观看历史。每个item的标题和描述被合并为文本特征，然后使用字母三元组编码。流派也被用作二元特征。这为Movie/TV产生了一个50K的特征向量。

在我们的网络框架中，用户特征被映射到用户视图（user view），其余特征被映射到不同的item视图（item view）。出于训练目的，每个用户视图与一个包含完全相同用户集的item视图匹配。为实现这一点，我们从每个用户-item视图对中二次抽样了登录用户（即具有唯一匿名化和哈希处理的Microsoft用户ID的用户），并基于其ID执行内连接。这导致每个用户-item视图对的用户数量不同。表1描述了本文中使用数据的一些基本统计信息。

<img src="/Users/dazhang/PycharmProject/Papers/3-RecSys/.picture/image-20260717173758591.png" alt="image-20260717173758591" style="zoom:50%;" />

$$ \text{Table 1: Statistics of the four data sets used in this paper.} $$

| Type | DataSet | UserCnt | Feature Size | Joint Users |
|------|---------|---------|--------------|-------------|
| User View | Search | 20M | 3.5M | / |
| Item View | News | 5M | 100K | 1.5M |
| | Apps | 1M | 50K | 210K |
| | Movie/TV | 60K | 50K | 16K |

联合用户（Joint Users）列表示每个item视图与用户视图之间的共同用户数量。



## 4. 推荐系统中的DSSM用于用户建模

深度结构化语义模型（DSSM）在[9]中被引入，用于增强Web搜索上下文中的查询文档匹配。鉴于其与我们提出的多视图深度神经网络的密切关系，我们在此简要回顾DSSM。

<img src="/Users/dazhang/PycharmProject/Papers/3-RecSys/.picture/image-20260717173321175.png" alt="image-20260717173321175" style="zoom:50%;" />

DSSM的典型架构如图1所示。DNN的输入（原始文本特征）是一个高维的词项向量，例如查询或文档中词项的原始计数（未经标准化）。然后DSSM将其输入通过两个神经网络（分别针对两个不同的输入）传递，并将它们映射到共享语义空间中的语义向量。对于Web文档排序，DSSM计算查询和文档之间的相关性分数为它们相应语义向量的余弦相似度，并根据它们与查询的相似度分数对文档进行排序。

更形式化地，如果我们记$x$为输入词项向量，$y$为输出向量，$l_i, i=1,\dots,N-1$为中间隐藏层，$W_i$为第$i$个权重矩阵，$b_i$为第$i$个偏置项，我们有：

$$ l_1 = W_1 x $$

$$ l_i = f(W_i l_{i-1} + b_i), i = 2, \dots, N-1 $$

$$ y = f(W_N l_{N-1} + b_N) \qquad (1) $$

其中我们在输出层和隐藏层$l_i, i=2,\dots,N-1$中使用tanh函数作为激活函数：

$$ f(x) = \frac{1-e^{-2x}}{1+e^{-2x}} \qquad (2) $$

查询$Q$和文档$D$之间的语义相关性分数通过以下公式计算：

$$ R(Q,D) = \text{cosine}(y_Q, y_D) = \frac{y_Q^T y_D}{\|y_Q\| \|y_D\|} \qquad (3) $$

其中$y_Q$和$y_D$分别是查询和文档的语义向量。在Web搜索中，给定查询后，文档按其语义相关性分数排序。

传统上，每个单词$w$由一个one-hot词向量表示，该向量的维度是词汇表的大小。然而，在现实世界的Web搜索任务中，词汇量通常非常大，one-hot向量词表示使得模型学习非常昂贵。因此，DSSM使用词哈希层（word hashing layer）来用字母三元组向量表示一个词。例如，给定一个单词（如"web"），在添加词边界符号（如"#web#"）后，该词被分割为一系列字母n-gram（例如，字母三元组：#-w-e, w-e-b, e-b-#）。然后，该词被表示为一个字母三元组的计数向量。例如，"web"的字母三元组表示是：...在图1中，第一层矩阵$W_1$表示字母三元组矩阵，将词项向量转换为其字母三元组计数向量，这不需要学习。尽管英语单词的总数可能会变得非常大，但英语（或其他类似语言）中不同字母三元组的总数通常是有限的。因此，它可以泛化到训练数据中未见的新词。

在训练中，假设查询与其点击的文档相关，DSSM的参数（即权重矩阵$W_i$）使用此信号进行训练。即，首先通过softmax函数从查询和文档之间的语义相关性分数估计给定查询的文档后验概率：

$$ P(D|Q) = \frac{\exp(\gamma R(Q,D))}{\sum_{D' \in D} \exp(\gamma R(Q,D'))} \qquad (4) $$

其中$\gamma$是softmax函数中的平滑因子，在我们的实验中通常根据保留数据集经验性地设置。$D$表示待排序的候选文档集。理想情况下，$D$应包含所有可能的文档。在实践中，对于每个（查询，点击文档）对，记为$(Q, D^+)$，其中$Q$是查询，$D^+$是被点击的文档，我们通过包含$D^+$和$N$个随机选择的未点击文档（记为$\{D_j^-; j=1,\dots,N\}$）来近似$D$。

在训练中，模型参数被估计为最大化训练集上给定查询的被点击文档的似然：

$$ L(\Lambda) = -\log \prod_{(Q, D^+)} P(D^+|Q) \qquad (5) $$

其中$\Lambda$表示神经网络的参数集。



## 5. 多视图深度神经网络

DSSM可以被视为一个多学习框架，它将数据的两个不同视图映射到一个共享视图。从这个意义上说，它可以被视为一个更通用的设置，用于学习两个不同视图之间的共享映射。

在这项工作中，我们提出了DSSM的一个扩展，其中我们有超过两个数据视图，我们称之为多视图DNN（MV-DNN）。在这种设置中，我们有$v+1$个视图，一个主视图称为$X_u$，其他$v$个辅助视图$X_1$到$X_v$，每个$X_i$有其自身的输入域$X_i \in \mathbb{R}^{d_i}$。每个视图也有其自身的非线性映射层$f_i(X_i, W_i)$，将$X_i$转换到共享语义空间$Y_i$。训练数据包含一组样本。第$j$个样本有一个主视图的实例$X_{u,j}$和一个激活的辅助视图$X_{a,j}$，其中$a$是样本$j$中激活视图的索引。所有其他视图输入$X_{i:i \neq a}$被设置为0向量。

目标是找到每个视图的非线性映射，使得主视图映射$Y_u$与所有其他视图$Y_1,\dots,Y_v$的映射之间的相似性之和在语义空间中被最大化。形式上，我们有：

$$ p = \arg \max_{W_u, W_1, \dots, W_v} \sum_{j=1}^N \frac{e^{\alpha \cos(Y_u, Y_{a,j})}}{\sum_{X' \in R^{d_a}} e^{\alpha \cos(Y_u, f_a(X', W_a))}} \qquad (6) $$

MV-DNN的架构如图2所示。在我们的推荐系统设置中，我们将主视图$X_u$设置为用户特征，并为我们旨在推荐的每种不同类型的item创建辅助视图。

采用此目标函数的直觉是尝试找到用户特征的单一映射，即$W_u$，可以将用户特征转换到一个空间，该空间匹配用户在的不同视图/领域中喜欢的所有不同item。这种参数共享方式允许那些没有足够信息学习良好映射的领域，通过其他拥有更多数据的领域来学习。如果假设在新闻文章方面品味相似的用户在其他领域也有相似的品味，那么这些领域可以从新闻领域学习到的用户映射中受益，这个假设成立时，该方法应该效果良好。如果这个假设成立，那么来自任何领域的样本都将有助于在所有领域中更准确地对相似用户进行分组。实验结果表明，在我们实验的领域中，这个假设是合理的，我们将在实验部分详细阐述。

<img src="/Users/dazhang/PycharmProject/Papers/3-RecSys/.picture/image-20260717173400035.png" alt="image-20260717173400035" style="zoom:50%;" />

Figure 2: Multi-view DNN for multiple domain recommendation

图2：用于多领域推荐的多视图DNN。它使用DNN将高维稀疏特征（例如，用户、新闻、App的原始特征）映射到联合语义空间中的低维稠密特征。第一隐藏层有50k个单元，实现了词哈希。经过词哈希的特征随后通过多个非线性投影层进行投影。该DNN中最终层的神经活动构成了语义空间中的特征。请注意，此图中的输入特征维度$x$（5M和3M）是假设性的，实际上每个视图可以有任意数量的特征。详见正文。

### 5.1 训练MV-DNN

MV-DNN可以使用随机梯度下降（SGD）进行训练。在实践中，每个训练示例包含一对输入，一个用于用户视图，一个用于数据视图。因此，尽管在我们的模型中只存在一个用户视图，但通常更方便的做法是拥有$N$个用户特征文件，每个文件对应一个item特征文件，其中$N$是用户-item视图对的总数。在算法1中，我们概述了训练MV-DNN的高级过程。当对$W_i \in \{W_u, W_1, \dots, W_v\}$求导数时，我们最终得到只有两个非零导数$\frac{\partial p}{\partial W_u}$和$\frac{\partial p}{\partial W_a}$，这使我们能够应用与DSSM[9]相同的更新规则，只需将$q$替换为$X_u$，将$d$替换为$X_a$。

<img src="/Users/dazhang/PycharmProject/Papers/3-RecSys/.picture/image-20260717173737756.png" alt="image-20260717173737756" style="zoom:50%;" />

**Algorithm 1** Training Multi-View DNN

1: **Input:** $N$ = # of view pairs, $M$ = # of training iterations,
   $U_A$ = user view architecture,
   $I_A = \{I_{A1}, \dots I_{AN}\}$ item view architecture,
   $U_D = \{U_{D1}, \dots U_{DN}\}$ user input files,
   $I_D = \{I_{D1}, \dots I_{DN}\}$ item input files,
   $W_U$ = user view weight matrix,
   $W_I = \{W_{I1}, \dots W_{IN}\}$ item view weight matrices
2: **Initialization**
3: Initialize $W_U$ and $W_I$ using $U_A$ and $I_A$
4: **for** $m = 1$ **to** $M$
5:   **for** $v = 1$ **to** $N$
6:     $T_U \leftarrow U_{Dv}$
7:     $T_I \leftarrow I_{Dv}$
8:     train $W_U$ and $W_I$ using $T_U$ and $T_I$
9:   **end for**
10: **end for**
11: **Output:** $W_U$ = final user weight matrix,
    $W_I$ = final set of item view weight matrices

### 5.2 MV-DNN的优势

尽管MV-DNN从原始的DSSM框架扩展而来，但它具有几个独特的特性，使其优于其前身。首先，原始的DSSM模型对查询视图和文档视图使用相同大小的特征维度，并使用相同的表示方法（例如字母三元组）进行预处理。这对特征组合步骤造成了很大的限制。由于推荐系统的异构性，用户视图和item视图很可能具有不同的输入特征。此外，许多类型的特征无法使用字母三元组进行最优表示。例如，URL域名特征通常包含前缀和后缀，如www、com、org，如果应用字母三元组，这些将被映射到相同的特征。在实践中，我们发现字母三元组表示在输入原始文本较短的情况下（例如原始DSSM模型中的查询文本和文档标题）效果理想，但不适合建模通常包含大量查询和URL域名的用户级特征。通过去除这个约束，新的MV-DNN模型可以纳入分类特征（如电影流派和App类别）、地理空间特征（如国家和区域），以及使用unigram或bigram表示的用户输入的原始文本特征。

其次，MV-DNN具有扩展到许多不同领域的能力，而原始的DSSM框架无法做到。通过执行如算法1所述的每个用户-item视图对之间的成对训练，我们的模型可以在训练过程的任何阶段轻松采用新的视图对，这些视图对可能包含完全独立的用户和item集，例如，从Xbox游戏添加新的数据集。通过在每次训练迭代中交替用户-视图对，我们的模型最终可以收敛到一个通过所有item视图训练的用户视图的最优embedding。注意，尽管理论上我们可以在不同的item视图中拥有不同的用户集，但在我们的实验中，为了方便和特征标准化的易用性，我们选择在所有视图中保持相同的用户集。



## 6. 维度和数据降维

在实践中，所提出的深度学习方法通常需要在用户视图的高维特征空间中处理大量的训练样本。为了扩展系统，我们提出了几种降维技术来减少用户视图中的特征数量。然后，我们提出了一种压缩和汇总用户训练样本的想法，将训练数据量减少到与用户数量呈线性关系。

### 6.1 顶部特征（Top Features）

一种简单的用户特征降维方法是选择top-K最频繁的特征。我们根据出现概率（约0.001）选择这些特征。其基本原理是用户可以用一组相对较小的频繁特征来描述，这些特征解释了用户常见的在线行为。注意，如第3节所述，用户的原始特征已经使用TF-IDF分数进行了预处理，因此我们选择的顶部特征不再包含搜索查询中的常见停用词。这产生了83K个用户特征。

### 6.2 K-means

K-means [8] 是一种众所周知的聚类技术，旨在创建若干聚类，使得每个点与其最近聚类之间的距离之和最小化：

$$ \arg \min_{C_1,\dots,C_k} \sum_{i=1}^N \min_{C_j \in \{C_1,\dots,C_k\}} \text{distance}(X_i, C_j) \qquad (7) $$

其中$X_i$是数据点，$C_j$表示聚类质心。K-means在视觉领域已显示出作为学习无监督特征的方法的良好性能[5]。基本思想是将相似的特征分组到同一个聚类中，并生成一个新的特征向量$Y$，其大小等于聚类数$K$，并将特征在聚类$i$中的出现次数作为其第$i$个分量的值。为此，我们使用长度为$U$的向量$f_i$表示每个特征，其中$U$是训练数据中的用户数，$f_i(j)$等于用户$i$拥有特征$j$的次数。然后对$f_i$进行归一化以使其长度为1。运行K-means的一个重要方面是，这里相关的距离是不同特征向量之间的角度，可以使用余弦相似度计算。然后，为每个用户向量$f_i$创建降维后的向量$Y_i$，如下所示：记$1 \leq \text{Cls}(a) \leq K$为分配给特征$a$的聚类，我们可以计算$Y_i$：

$$ Y_i(j) = \sum_{a: X_i(a) > 0 \ \& \ \text{Cls}(a) = j} f_i(a) \qquad (8) $$

为了能够使用K-means提取合理数量的特征，我们需要有相对较多的聚类，因为少量的聚类（比如100）将导致大量特征落在同一聚类中（考虑到用户特征规模为350万）。这将因此产生难以学习有用模式的特征。为了缓解这个问题，我们将聚类数设置为10k。这意味着平均每个聚类有350个特征。大量的聚类和原始特征使得K-means的计算代价很高。在我们的实验中，我们使用了基于云的Map-Reduce实现[4]来运行K-means。

### 6.3 局部敏感哈希（LSH）

局部敏感哈希（LSH）[10]通过使用随机投影矩阵将数据投影到低维空间，使得原始空间中成对余弦距离在新区间中仍然得以保留。LSH需要一个变换矩阵$A \in \mathbb{R}^{d \times k}$，其中$d$是原始空间中的特征数量，$k$是使用的随机投影数量。这意味着$A$包含$k$个不同的投影，记为$A_i$，每个投影接收$X$向量并输出一个哈希值$Y_i$。LSH的输出向量$Y \in \mathbb{R}^k$可以通过串联所有不同的$Y_i$哈希值得到。具体来说，为了计算每个$Y_i$，我们使用以下方程：

$$ Y_i = \begin{cases} 1 & \text{if } A_i \cdot X \geq 0 \\ 0 & \text{else} \end{cases} \qquad (9) $$

两个向量$X_1, X_2 \in \mathbb{R}^d$之间的余弦相似度可以通过$\cos\left(\frac{H(Y_1, Y_2)}{k}\pi\right)$来近似，其中$H(Y_1, Y_2)$是输入向量的LSH之间的汉明距离。为了以更高的精度保持余弦相似度，我们需要增加使用的投影数$k$。我们使用了$k=10,000$，与K-means聚类使用的相同。虽然LSH可以独立地应用于每个向量，并且所有投影可以独立计算，这使得算法在Map-Reduce框架下高度可分布式，但该算法需要生成变换矩阵$A$，在我们的情况下它有$3.5M \times 10^4$个条目，从$N(0,1)$生成，需要约300GB的存储空间。此外，在计算LSH向量期间，$A$必须存储在每个节点上。这些问题使得LSH在Map-Reduce框架中代价过高。

已经提出了许多解决方案来解决这个问题。其中很多基于生成稀疏$A$ [13]。这里，我们使用了[31]中引入的池化技巧（pooling trick）。其思想是保留一个大小为$m$的随机数池$B$，这些随机数从$N(0,1)$生成，其中$m$通常远小于变换矩阵$A$的大小。要获取$A_{ij}$条目，只需应用$i,j$的一致哈希函数来获取$B$中的索引并查找该值。在我们的实验中，我们设置$m=1,000,000$，相比原始存储需求减少了10000倍以上，在Map-Reduce期间每个节点仅需10M存储即可轻松存储。

### 6.4 减少训练样本数量

每个视图的训练数据包含一组对$(User_i, Item_j)$，其中用户$i$喜欢item $j$。在实践中，一个用户可能喜欢很多item，这有时会导致训练数据非常庞大。例如，在我们的新闻推荐数据集中，对的数量超过10亿，即使在优化后的GPU实现上训练也非常缓慢。

为了缓解这个问题，我们压缩训练数据，使其每个视图每个用户只包含一个训练样本。具体来说，压缩后的训练样本包括用户特征与该视图中用户喜欢的所有item的平均特征配对。这将对每个视图的训练样本数量减少到训练数据中的用户数量，从而大大减小训练数据量。注意，这种技术的一个担忧是目标函数现在变成了最大化用户特征与用户喜欢的item的平均特征之间的相似性。在评估时会略有不同，因为在测试时每个用户只给一个单一item。然而，这种近似对于使系统良好扩展是必要的。此外，实验结果表明这种近似在实践中仍然产生了非常有前景的结果。



## 7. 实验设置

在本节中，我们解释了我们实证研究的过程，并简要回顾了我们作为基线进行比较的几种推荐算法。

对于每个数据集，我们旨在评估系统对已有该领域交互历史的用户（老用户）和没有任何先前交互但有一定搜索和浏览历史（可编码到用户视图中）的用户（新用户）的性能。为了评估，数据集使用以下标准划分为训练集和测试集：

首先，每个用户以0.9:0.1的概率比被随机分配"训练"或"测试"标签。然后，对于每个带有"测试"标签的用户，我们以0.8:0.2的概率比进一步将其标记为"老"或"新"用户。对于标记为"老"的用户，其50%的item用于训练，其余用于测试。另一方面，对于"新"用户，他们的item仅用于测试，因为用户-item对保证永远不会出现在训练过程中。这样，这些用户确实是系统完全的新用户。数据集划分的详细信息见表2。

![image-20260717173710490](/Users/dazhang/PycharmProject/Papers/3-RecSys/.picture/image-20260717173710490.png)

在性能评估方面，对于训练数据中的每个$(user_i, item_j)$对，我们选择9个其他随机item $item_{r1}, \dots, item_{r9}$，其中$r1$到$r9$是随机索引，并创建9个测试对$(user_i, item_{rk}), 1 \leq k \leq 9$，添加到测试数据集中。评估标准是衡量系统对同一用户的正确对$(user_i, item_j)$相对于其他随机item $(user_i, item_{rk})$的排名效果。因此，我们采用了两个指标：（1）平均倒数排名（MRR），计算正确item在其他item中的排名的倒数，并对整个测试数据求平均；（2）Precision@1（P@1），计算系统将正确item排在第一位的次数百分比。

我们与以下几种基线系统进行了比较：

**标准SVD矩阵分解：** 在此基线中，我们构建用户/item矩阵并使用SVD执行矩阵分解。这是协同过滤技术的标准基线，不使用item或用户的任何特征。在实践中，此基线仅在相对较小的数据集上计算可行（在我们的案例中是Apps数据）。此外，这种方法无法对新用户进行推荐，因为他们不在用户-item矩阵中。

**最频繁item：** 由于SVD无法处理新用户推荐，此方法被用作新用户的简单基线。它首先计算训练数据中每个item的频率，然后对于每个测试样本，根据item在训练集中的频率将$(user_i, item_j)$与其他随机item进行排序。

**典型相关分析（CCA）：** CCA [29]是一种传统的多视图学习技术，旨在为每个输入视图找到一对线性变换，使得变换后数据之间的相关性最大化。它与DSSM相似，但有两个主要区别：（1）CCA通常使用线性变换，尽管存在通过核版本的CCA实现的非线性变换，但这在大规模实验中通常计算代价过高；（2）CCA在一定的固定方差约束下最大化相关性，而DSSM最大化正确对的排名。基于排序的目标函数已被证明对推荐系统是更好的目标。在我们的CCA实验中，我们只使用了top-k用户特征，因为其他两种降维技术（K-means和LSH）产生非稀疏特征向量，使得相关矩阵过于稠密而无法高效计算。

**协同主题回归（CTR）：** CTR [32]是一种最近开发的推荐系统，它结合了贝叶斯矩阵分解和item特征来为item创建推荐。它在学术论文推荐中已被证明是成功的。在CTR模型中，有两个输入：一个协同矩阵和item特征（使用词袋表示）。该模型通过最小化协同矩阵的重构误差并利用item特征作为额外信号来将用户与item匹配。这有助于建模在训练数据中之前未出现的新item。对于我们的场景中的新用户推荐，我们将协同矩阵$A$的转置作为输入，并提使用用户特征代替item特征。

对于我们的方法，在Apps和News数据集上，我们首先运行了三组实验来训练单视图DNN模型，每对应第6节中的一种降维方法（SV-TopK、SV-Kmeans和SV-LSH）。然后我们为MV-DNN再运行了三组实验。前两组使用TopK和K-means用户特征组合Apps和News数据（MV-TopK和MV-Kmeans）。第三组实验使用TopK用户特征联合训练Apps、News和Movie/TV特征（MV-TopK w/Xbox）。



## 8. 结果与讨论

![image-20260717173543419](/Users/dazhang/PycharmProject/Papers/3-RecSys/.picture/image-20260717173543419.png)

![image-20260717173558306](/Users/dazhang/PycharmProject/Papers/3-RecSys/.picture/image-20260717173558306.png)

表3和表4分别显示了在Apps和News数据上不同方法获得的结果。我们将算法分为三种类型以便说明：类型I是基线算法；类型II是我们的单视图模型；类型III是多视图DNN模型。我们看到，朴素的最频繁item基线表现非常差，这证实了针对新用户的简单解决方案在我们的案例中不会效果良好。它还表明，标准SVD矩阵分解在此任务中即使对于在协同过滤矩阵中有条目的现有用户也不够好。令人惊讶的是，CCA模型在Apps数据上的表现不比随机猜测好，这表明在DSSM中使用非线性映射加上基于排序的目标对系统很重要。CTR模型[32]对现有用户表现尚可，但对新用户不够好。

对于单视图DNN（类型II），结果表明性能取决于所使用的降维方法。可以看出，对Apps和News数据都最好的方法是top-K特征维度方法，它以很大幅度优于其他两种方法。这可以看作是对用户可以通过一组相对较小的信息性特征进行建模的假设的确认。它也表明K-means和LSH在正确捕捉用户行为语义方面效果较差。作为我们的单视图模型（类型II）与传统推荐方法（类型I）的正面比较，我们最好的模型（SV-TopK）在所有用户上优于同样利用item特征进行推荐的最佳基线CTR [32] 11%（MRR分数0.497 vs 0.448），在新用户上优于36.7%（MRR分数0.436 vs 0.319）。对于P@1，我们看到了更大的提升：所有用户提升13%，新用户提升88.7%，这显示了我们的系统在推荐顶级item方面的有效性。

此外，对于MV-DNN，结果表明增加更多领域确实有助于同时改善所有领域。具体来说，通过结合新闻视图和App视图进行训练，我们在两个指标上看到News和Apps数据集的显著改善。具体而言，对于表3中显示的App数据，当与最佳单视图模型相比时，所有用户的MRR分数从0.497增加到0.517，相对于单视图模型增长了4%。更重要的是，我们在新用户上看到了更大的提升，其中一个视图新用户数据的缺乏可以通过来自其他视图的数据得到补偿。这由App数据集新用户的MRR相对提升7%（从0.436到0.466）和P@1相对提升11%（从0.268到0.297）所证明。因此，我们渴望知道：我们能否安全地推断，更多的视图确实可以提高系统性能？为了找到这个假设的答案，我们进一步将Xbox数据添加到框架中，并用三个用户-item视图对训练了一个MV-DNN模型。结果相当令人鼓舞：MRR分数在App数据上对所有用户进一步提高了6%，对新用户提高了8%。另一方面，通过与最优算法相比，我们最好的MV-DNN（包含Xbox视图）与top-K特征结合，在P@1上对所有用户比CTR模型好25.2%（从0.277到0.347），对新用户好115%（从0.142到0.306）。

在News数据上也可以观察到类似的结果，如表4所示，MV-DNN在所有用户上比CTR模型好49%，在新用户上好101%。注意，在此表中，CCA和SVD的结果缺失。由于训练数据规模极大，包含150万用户和超过10亿条条目，这两种传统算法无法处理如此大规模的数据。我们将在下一节详细讨论可扩展性，但这里非常明显的是，我们基于DNN的方法可以轻松扩展到十亿条目的计算能力，同时产生出色的推荐结果。

为了探索从系统学习到的模式的有效性，我们执行了以下实验来测试单特征输入下的推荐性能。具体来说，我们采用了性能最好的系统（带有top-k特征的MV-DNN），并仅开启一个领域特征来构建用户特征。因此，生成的用户特征只有一个值，即该领域的ID。然后我们运行我们的模型对其他视图进行预测，以在所有现有item中找到top匹配的新闻和App。表5显示了一些结果。可以看出，学习到的推荐系统确实非常有效。在第一个例子中，我们假设一个用户只访问了barackobama.com。top匹配的新闻显示了所有关于奥巴马总统和奥巴马医改的相关信息，所有这些都与该网站相关。另一方面，top匹配的Apps在这种情况下也与健康相关。在第二个例子中，我们有一个访问了www.spiegel.de的用户，这是一个主要的德国新闻网站，除了说明用户可以阅读德语之外，并没有透露太多关于用户的信息。系统为他们匹配了关于2014年FIFA世界杯的文章，这似乎是德国人在此时间段内的共同兴趣。在最后一个例子中，用户似乎对婴儿相关信息感兴趣，top匹配的新闻和Apps都与婴儿、怀孕等相关。

![image-20260717173516179](/Users/dazhang/PycharmProject/Papers/3-RecSys/.picture/image-20260717173516179.png)

![Table 5: Examples of learned mapping between URL domains, News articles and Apps]()

**Table 5:** 学习到的URL域名、新闻文章和Apps之间的映射示例。对于这些域名，只有其特征ID用于训练。底层域名名称对目标应用来说是未知的。



## 9. 公开数据实验

为了进一步展示我们的方法在跨领域用户建模方面的优势，我们在由文献[28]作者提供的公开数据上执行了一组实验。数据集包含来自不同研究领域的作者，目标是推荐来自另一个领域的作者进行跨领域合作。该数据包含来自五个领域（数据挖掘、理论等）的33,739名作者，其中每个数据条目指定了研究领域的名称、论文的标题和摘要、作者列表以及论文发表的年份。我们使用单视图DNN来建模这种跨领域合作（例如，数据挖掘和理论研究人员之间的合作）。在这种情况下，用户视图和item视图共享相同的特征表示。具体来说，我们使用作者在训练期间（1990年至2001年）发表的论文标题和摘要中的unigram单词作为特征，得到特征维度为31,932。类似于原始作者的评估方法，我们随机选择了一组在训练期间已有跨领域合作且在测试期间至少有五次跨领域合作的作者作为我们的评估集。对于每对不同领域之间的合作，我们训练一个单独的单视图DNN模型，迭代100次。

![image-20260717173615508](/Users/dazhang/PycharmProject/Papers/3-RecSys/.picture/image-20260717173615508.png)

![image-20260717173626059](/Users/dazhang/PycharmProject/Papers/3-RecSys/.picture/image-20260717173626059.png)

表6显示了结果。总体而言，我们的方法在所有四个跨领域数据集上始终优于CTL方法，除了P@20指标。特别地，我们在recall@100上获得了更高的值，在DM到理论推荐中最佳提升达到96%。结果表明，使用带有非线性深度神经模型的丰富用户特征确实可以捕获大量使用传统的基于单词的共现模型（如生成式主题模型）无法准确建模的语义。我们相信使用多视图DNN模型可以进一步提高性能，但将其留给未来的研究。

在效率方面，作者报告了CTL方法在整个数据集上的训练时间为12-15小时。我们的算法运行速度快得多，每个模型在GPU机器上对相同数量的数据完成100次迭代训练仅需5-7分钟。我们将在下一节详细介绍我们框架的可扩展性。

![Table 6: Recommendation performance on the cross-domain collaboration public data [28]]()

**Table 6:** 跨领域合作公开数据[28]上的推荐性能。



## 10. 算法可扩展性

本节我们比较各种算法在训练时间方面的性能。回顾前一节，我们提到（在表4中）对于新闻数据，SVD（CF）和CCA无法处理包含10亿条目的用户-item矩阵。这显示了我们的深度学习框架的优势之一，它使用SGD进行训练，因此能够通过分布式训练处理海量数据。性能的细节见表7。我们可以观察到，对于相对较小的Apps数据集，SVD和CCA完成得相对较快（约4小时，但产生了相当差的推荐性能）。单视图DNN模型（SV-TopK）在33小时内完成了100次训练迭代。然而，基于内容的CTR模型需要很长时间来训练。原因是CTR需要一个使用LDA模型训练的主题比例($\theta$)和主题分布($\beta$)的初始种子。CTR然后使用这些文件来优化用户和item特征之间的相关性。因此，在两个数据集上，训练CTR比我们的深度学习模型更昂贵。另一方面，我们看到SV-TopK和MV-TopK表现出与数据规模（次）线性关系的训练时间，因为通常SGD在更多数据可用时运行更少的epoch来收敛。

![image-20260717173646567](/Users/dazhang/PycharmProject/Papers/3-RecSys/.picture/image-20260717173646567.png)

![Figure 3: Training errors for two views in MV-TopK model]()

**图3：** MV-TopK模型中两个视图的训练误差。

图3显示了MV-TopK模型在News和Apps两个视图上每次迭代的训练误差。在我们的实验中，我们手动指定训练迭代次数为100。一个原因是我们继续看到所有视图性能的改善，尽管改善随着时间的推移变得越来越小。另一方面，我们发现实践中某些视图的收敛速度比其他视图快。例如，对于图3中的那个特定模型，新闻视图在20次迭代后快速收敛，而Apps视图需要约70次迭代才能达到收敛。由于训练期间交替用户-item视图对的过程以及不同视图的不同收敛速度，进行早停以进一步提高模型的可扩展性成为一项关键的未来工作。



## 11. 结论与未来工作

在这项工作中，我们提出了一个通用的推荐框架，使用深度学习将丰富的用户特征与item特征进行匹配。我们还展示了如何扩展此框架以结合来自不同领域的数据，从而进一步提高推荐质量。然后我们讨论了通过降维使这种方法适用于大规模数据集的不同方法。所提出的模型可以处理现有用户和新用户的推荐。在多个大规模真实世界数据集上的实验表明，所提出的方法以较大幅度优于其他系统。

作为一项初步研究，我们相信这项工作为使用来自多个数据源的深度学习构建推荐系统打开了一扇新的大门。尽管本文中的大多数评估使用了专有数据，但该框架应该能够在不增加太多额外工作的情况下泛化到其他数据源，如第9节中使用小型公开数据集所展示的那样。例如，根据用户的推文推荐音乐，根据Facebook状态更新推荐餐厅，或根据用户搜索查询推荐图像和视频。

对于未来的工作，我们旨在将更多的用户特征纳入用户视图。我们希望使我们的DNN学习更具可扩展性，以便最终可以使用全部用户特征进行训练而无需降维。我们还旨在将更多领域添加到我们的多视图框架中，并进一步详细分析其性能。另一个重要的方向是研究如何将协同过滤与我们的方法结合起来，目前它仅作为基于内容的过滤方法运行。



## 12. 致谢

我们非常感谢KDD'12论文[28]的作者公开他们的数据。特别感谢Sen Wu在数据集讨论中的大力支持。我们还要感谢匿名审稿人提供的宝贵反馈。



## 参考文献

[1] Fabian Abel, Qi Gao, Geert-Jan Houben, and Ke Tao. Twitter-based user modeling for news recommendations. In *IJCAI'13*, pages 2962–2966.

[2] Amr Ahmed, Abhimanyu Das, and Alexander J Smola. Scalable hierarchical multitask learning algorithms for conversion optimization in display advertising. In *WSDM'14*, pages 153–162.

[3] Robert M Bell and Yehuda Koren. Improved neighborhood-based collaborative filtering. In *KDD'13 CUP*, 2007.

[4] Cheng Chu, Sang Kyun Kim, Yi-An Lin, Yuan Yuan Yu, Gary Bradski, Andrew Y Ng, and Kunle Olukotun. Map-reduce for machine learning on multicore. 2007.

[5] Adam Coates, Andrew Y Ng, and Honglak Lee. An analysis of single-layer networks in unsupervised feature learning. In *AISTATS'11*, pages 215–223.

[6] Abhinandan S Das, Mayur Datar, Ashutosh Garg, and Shyam Rajaram. Google news personalization: scalable online collaborative filtering. In *WWW'07*, pages 271–280.

[7] Asela Gunawardana and Christopher Meek. Tied boltzmann machines for cold start recommendations. In *RecSys'08*, pages 19–26.

[8] John A Hartigan and Manchek A Wong. Algorithm as 136: A k-means clustering algorithm. *Applied statistics*, pages 100–108, 1979.

[9] Po-Sen Huang, Xiaodong He, Jianfeng Gao, Li Deng, Alex Acero, and Larry Heck. Learning deep structured semantic models for web search using clickthrough data. In *CIKM'13*, pages 2333–2338.

[10] Piotr Indyk and Rajeev Motwani. Approximate nearest neighbors: towards removing the curse of dimensionality. In *STOC'98*, pages 604–613.

[11] Joonseok Lee, Samy Bengio, Seungyeon Kim, Guy Lebanon, and Yoram Singer. Local collaborative ranking. In *WWW'14*, pages 85–96.

[12] Bin Li, Qiang Yang, and Xiangyang Xue. Transfer learning for collaborative filtering via a rating-matrix generative model. In *ICML'09*, pages 617–624.

[13] Ping Li, Trevor J Hastie, and Kenneth W Church. Very sparse random projections. In *KDD'06*, pages 287–296.

[14] Greg Linden, Brent Smith, and Jeremy York. Amazon.com recommendations: Item-to-item collaborative filtering. *Internet Computing, IEEE*, 7(1):76–80, 2003.

[15] Jiahui Liu, Peter Dolan, and Elin Rønby Pedersen. Personalized news recommendation based on click behavior. In *IUI'10*, pages 31–40.

[16] Prem Melville, Raymond J Mooney, and Ramadass Nagarajan. Content-boosted collaborative filtering for improved recommendations. In *AAAI'02*.

[17] Weike Pan, Evan Wei Xiang, Nathan Nan Liu, and Qiang Yang. Transfer learning in collaborative filtering for sparsity reduction. In *AAAI'10*, 2010.

[18] Al Mamunur Rashid, George Karypis, and John Riedl. Learning preferences of new users in recommender systems: an information theoretic approach. *ACM SIGKDD Explorations Newsletter*, 10(2):90–100, 2008.

[19] Jasson DM Rennie and Nathan Srebro. Fast maximum margin matrix factorization for collaborative prediction. In *ICML'05*, pages 713–719.

[20] Shaghayegh Sahebi and Peter Brusilovsky. Cross-domain collaborative recommendation in a cold-start context: The impact of user profile size on the quality of recommendation. In *User Modeling, Adaptation, and Personalization*, pages 289–295. Springer, 2013.

[21] Ruslan Salakhutdinov and Andriy Mnih. Bayesian probabilistic matrix factorization using markov chain monte carlo. In *ICML'08*, pages 880–887.

[22] Ruslan Salakhutdinov, Andriy Mnih, and Geoffrey Hinton. Restricted boltzmann machines for collaborative filtering. In *ICML'07*, pages 791–798.

[23] Badrul Sarwar, George Karypis, Joseph Konstan, and John Riedl. Item-based collaborative filtering recommendation algorithms. In *WWW'01*, pages 285–295. ACM, 2001.

[24] Andrew I Schein, Alexandrin Popescul, Lyle H Ungar, and David M Pennock. Methods and metrics for cold-start recommendations. In *SIGIR'02*, pages 253–260.

[25] Yang Song, Weiwei Cui, Shixia Liu, and Kuansan Wang. Online behavioral genome sequencing from usage logs: decoding the search behaviors. In *WWW'14*, pages 91–94.

[26] Yang Song, Hongning Wang, and Xiaodong He. Adapting deep ranknet for personalized search. In *WSDM'14*, pages 83–92.

[27] Shiliang Sun. A survey of multi-view machine learning. *Neural Computing and Applications*, 23(7-8):2031–2038, 2013.

[28] Jie Tang, Sen Wu, Jimeng Sun, and Hang Su. Cross-domain collaboration recommendation. In *KDD'12*, pages 1285–1293.

[29] Bruce Thompson. Canonical correlation analysis. *Encyclopedia of statistics in behavioral science*, 2005.

[30] Aaron Van den Oord, Sander Dieleman, and Benjamin Schrauwen. Deep content-based music recommendation. In *NIPS'13*, pages 2643–2651.

[31] Benjamin Van Durme and Ashwin Lall. Online generation of locality sensitive hash signatures. In *ACL'10 Short Papers*, pages 231–235.

[32] Chong Wang and David M Blei. Collaborative topic modeling for recommending scientific articles. In *KDD'11*, pages 448–456.
