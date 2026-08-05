# ItemCF: Amazon.com Recommendations: Item-to-Item Collaborative Filtering

> Greg Linden, Brent Smith, Jeremy York | Amazon.com
>
> IEEE Internet Computing, January/February 2003



本文分享了Amazon.com提出的item到item协同过滤（Item-to-Item Collaborative Filtering）推荐算法。核心内容：

- 离线构建item-item相似度矩阵，在线推荐仅需毫秒级计算，可扩展到 **数千万用户** 和 **数百万item**
- 通过共同购买/共同浏览行为计算**item间相似度**，替代传统用户-用户协同过滤
- 相比传统协同过滤、聚类模型和基于搜索的方法，在大规模数据上兼具可扩展性和推荐质量

关键发现：

- 与传统协同过滤不同，算法在线计算量 **与顾客数量和目录item数量无关**，可实时生成高质量推荐
- 算法在线部分 **仅依赖用户购买或评分的item数量**，即使数据有限（仅两三个item）也能生成高质量推荐

---



## 摘要

推荐算法最为人熟知的应用是电子商务网站[1]，它们利用顾客的兴趣输入来生成推荐item列表。许多应用仅使用顾客购买和明确评分的item来表征其兴趣，但也可以使用其他属性，包括浏览过的item、人口统计数据、主题兴趣和最喜欢的艺术家。

在Amazon.com，我们使用推荐算法为每位顾客个性化在线商店。商店会根据顾客兴趣发生根本性变化——向软件工程师展示编程类书籍，向新手妈妈展示婴儿玩具。**点击率和转化率（衡量网页和电子邮件广告效果的两个重要指标）远超非定向内容（如横幅广告和畅销榜单）的效果**。

电子商务推荐算法通常在充满挑战的环境中运行。例如：

- 大型零售商可能拥有海量数据，数千万顾客 和 数百万种不同的目录item。
- 许多应用要求结果集在实时内返回，**不超过半秒钟**，同时仍要生成**高质量的推荐**。
- 新顾客通常只有极为有限的信息，仅基于少数几次购买或产品评分。
- 老顾客可能拥有过多信息，基于数千次购买和评分。
- **顾客数据是动态变化的**：每次交互都提供有价值的顾客数据，算法必须立即响应新信息。

解决推荐问题有三种常见方法：传统协同过滤、聚类模型和基于搜索的方法。本文将比较这些方法与我们的算法——我们称之为**item到item协同过滤**。与传统协同过滤不同，我们的算法的在线计算量与顾客数量和产品目录中的item数量无关。该算法实时生成推荐，可扩展到海量数据集，并能生成高质量的推荐。



## 推荐算法（Recommendation Algorithms）

大多数推荐算法首先寻找一组顾客，其购买和评分的item 与 用户的购买和评分item存在重叠[2]。然后，算法聚合这些相似顾客的item，**排除用户已购买或评分的item**，并将剩余item推荐给用户。这类算法的两种流行版本是协同过滤和聚类模型。其他算法——包括基于搜索的方法 和 我们自己的item到item协同过滤——则专注于寻找相似item，而非相似顾客。对于用户购买和评分的每个item，算法尝试找到相似item，然后聚合这些相似item并进行推荐。



## 传统协同过滤（Traditional Collaborative Filtering）

传统协同过滤算法将顾客表示为item的 $N$ 维向量，其中 $N$ 是不同目录item的数量。向量的分量对于已购买或正面评分的item为正，对于负面评分的item为负。**为了补偿畅销item，算法通常将向量分量乘以逆频率（购买或评分该item的顾客数量的倒数），使得不太知名的item更加相关[3]**。对于几乎所有顾客来说，这个**向量都非常稀疏**。

该算法基于与用户最相似的少数几位顾客来生成推荐。它可以通过多种方式衡量两位顾客 $A$ 和 $B$ 的相似度；一种常见方法是测量两个向量之间夹角的余弦值[4]：

$$
\text{similarity}(\vec{A}, \vec{B}) = \cos(\vec{A}, \vec{B}) = \frac{\vec{A} \cdot \vec{B}}{|\vec{A}| \times |\vec{B}|}
$$

算法也可以使用多种方法从相似顾客的item中选择推荐；一种常用技术是根据有多少位相似顾客购买了该item来对每个item进行排序。

使用协同过滤生成推荐的计算开销很大。最坏情况下为 $O(MN)$ ，其中 $M$ 是顾客数量， $N$ 是产品目录item数量，因为它需要检查 $M$ 位顾客以及每位顾客最多 $N$ 个item。然而，由于平均顾客向量极其稀疏，算法的性能往往更接近 $O(M+N)$ 。扫描每位顾客约为 $O(M)$ 而非 $O(MN)$ ，因为几乎所有的顾客向量都只包含少量item，与目录大小无关。但少数顾客购买或评分的item占目录的很大比例，这需要 $O(N)$ 的处理时间。因此，算法的最终性能约为 $O(M+N)$ 。即便如此，对于非常大的数据集——例如1千万或更多顾客以及1百万或更多目录item——该算法会遇到严重的性能和扩展性问题。

通过减少数据规模可以部分解决这些扩展性问题[4]。我们可以通过随机抽样顾客或丢弃购买次数较少的顾客来减少 $M$ ，通过丢弃非常流行或非常不受欢迎的item来减少 $N$ 。还可以通过基于产品类别或主题分类划分item空间来以较小的常数因子减少需要检查的item数量。聚类和主成分分析等降维技术可以大幅减少 $M$ 或 $N$ [5]。

不幸的是，所有这些方法也会以多种方式降低推荐质量。第一，如果算法只检查少量顾客样本，所选顾客与用户的相似度将降低。第二，item空间划分将推荐限制在特定的产品或主题领域。第三，如果算法丢弃最流行或最不受欢迎的item，这些item将永远不会出现在推荐中，且仅购买过这些item的顾客将无法获得推荐。应用于item空间的降维技术通过消除低频item往往会产生同样的效果。应用于顾客空间的降维技术实际上将相似顾客分组到聚类中；如下所述，这种聚类也会降低推荐质量。



## 聚类模型（Cluster Models）

为了找到与用户相似的顾客，聚类模型**将顾客群划分为多个分片**（segment），并将任务视为分类问题。**算法的目标是将用户分配到包含最相似顾客的分片中，然后使用该分片中顾客的购买和评分来生成推荐**。

这些分片通常使用聚类或其他无监督学习算法创建，尽管有些应用使用人工确定的分片。聚类算法使用相似度度量将最相似的顾客分组在一起形成聚类或分片。由于在大数据集上进行最优聚类是不切实际的，大多数应用使用各种形式的贪心聚类生成。这些算法通常从一组初始分片开始——通常每个分片包含一个随机选择的顾客——然后反复将顾客匹配到现有分片，通常还会创建新分片或合并现有分片[6]。对于非常大的数据集——尤其是高维数据集——采样或降维也是必要的。

一旦算法生成了分片，它会计算用户与概括每个分片的向量之间的相似度，然后选择相似度最强的分片并相应地对用户进行分类。有些算法将用户分类到多个分片中，并描述每个关系的强度[7]。

聚类模型的在线可扩展性和性能优于协同过滤[3]，**因为它们将用户与可控数量的分片进行比较，而不是与整个顾客群进行比较**。复杂且昂贵的聚类计算在离线运行。然而，推荐质量较低[1]。聚类模型将众多顾客分组到一个分片中，将用户匹配到一个分片，然后将该分片中的所有顾客视为相似顾客以进行推荐。由于**聚类模型找到的相似顾客并非最相似的那些顾客，因此它们生成的推荐相关性较低**。通过使用大量细粒度分片可以提高质量，但此时在线用户-分片分类的代价几乎与使用协同过滤寻找相似顾客一样高。



## 基于搜索的方法（Search-Based Methods）

基于搜索或基于内容的方法将推荐问题视为相关item的搜索[8]。给定用户购买和评分的item，算法构造一个搜索查询，以查找同一作者、艺术家或导演的其他流行item，或具有相似关键词或主题的item。例如，如果顾客购买了《教父》DVD套装，系统可能推荐其他犯罪剧情片、其他由马龙·白兰度主演的影片，或其他由弗朗西斯·福特·科波拉导演的电影。

如果用户购买或评分较少，基于搜索的推荐算法具有良好的扩展性和性能。然而，对于拥有数千次购买的用户，将所有item作为查询依据是不切实际的。算法必须使用数据的子集或摘要，从而降低质量。在所有情况下，推荐质量都相对较差。推荐通常要么过于泛化（如畅销的剧情片DVD），要么过于狭窄（如同一位作者的所有书籍）。推荐应该帮助顾客发现新的、相关的和有趣的item。同一作者或同一主题类别的热门item无法实现这一目标。



## item到item协同过滤（Item-to-Item Collaborative Filtering）

Amazon.com在许多电子邮件营销活动和大多数网站页面（包括高流量的Amazon.com首页）上使用推荐作为定向营销工具。点击"您的推荐"链接可引导顾客进入一个区域，在这里他们可以按产品线和主题领域筛选推荐，对推荐产品进行评分，对以前的购买进行评分，并查看推荐item的原因（见图1）。

<img src="/Users/dazhang/PycharmProject/Papers/3-RecSys/.picture/image-20260805171546535.png" alt="image-20260805171546535" style="zoom:33%;" />

> 图1：Amazon.com首页上的"您的推荐"功能。使用该功能，顾客可以对推荐进行排序并添加自己的产品评分。

如图2所示，我们的购物车推荐功能根据顾客购物车中的item提供产品建议。该功能类似于超市结账通道中的冲动性购买item，但我们的冲动性购买item是针对每位顾客定向的。

<img src="/Users/dazhang/PycharmProject/Papers/3-RecSys/.picture/image-20260805171613255.png" alt="image-20260805171613255" style="zoom:33%;" />

> 图2：Amazon.com购物车推荐。推荐基于顾客购物车中的item：《程序员修炼之道》和《游戏开发物理学》。

Amazon.com广泛使用推荐算法来个性化其网站以满足每位顾客的兴趣。由于现有推荐算法无法扩展到Amazon.com数千万顾客和产品的规模，我们开发了自己的算法。我们的算法——**item到item协同过滤**——可扩展到海量数据集，并实时生成高质量的推荐。



## 工作原理（How It Works）

item到item协同过滤**不是将用户匹配到相似顾客**，而是**将用户购买和评分的每个item匹配到相似item**，然后将这些相似item组合成推荐列表[9]。

为了确定给定item的最相似匹配，算法通过**查找顾客倾向于一起购买的item来构建相似item表**。我们可以通过遍历所有item对并计算每对的相似度度量来构建产品到产品矩阵。然而，许多产品对没有共同顾客，因此该方法在处理时间和内存使用方面效率低下。以下迭代算法通过计算单个产品与所有相关产品之间的相似度提供了一种更好的方法：

$$
\begin{aligned}
&\textbf{for } \text{产品目录中的每个item } I_1 \textbf{ do} \\
&\quad \textbf{for } \text{每个购买过 } I_1 \text{ 的顾客 } C \textbf{ do} \\
&\quad\quad \textbf{for } \text{每个被顾客 } C \text{ 购买过的item } I_2 \textbf{ do} \\
&\quad\quad\quad \text{记录顾客同时购买了 } I_1 \text{ 和 } I_2 \\
&\quad\quad \textbf{end for} \\
&\quad \textbf{end for} \\
&\quad \textbf{for } \text{每个item } I_2 \textbf{ do} \\
&\quad\quad \text{计算 } I_1 \text{ 和 } I_2 \text{ 之间的相似度} \\
&\quad \textbf{end for} \\
&\textbf{end for}
\end{aligned}
$$

可以通过多种方式计算两个item之间的相似度，但一种常用方法是使用我们之前描述的余弦度量，其中每个向量对应一个item而非顾客，并且向量的 $M$ 个维度对应购买过该item的顾客。

这种相似item表的离线计算极其耗时，最坏情况下为 $O(N^2 M)$ 。然而，在实际中它更接近 $O(NM)$ ，因为大多数顾客的购买记录非常少。对购买畅销品的顾客进行采样可以进一步减少运行时间，且对质量的降低很小。

有了相似item表后，算法找到与用户每次购买和评分相似的item，聚合这些item，然后推荐最流行或相关性最高的item。这个计算非常快速，**仅取决于用户购买或评分的item数量**。



## 可扩展性：比较（Scalability: A Comparison）

Amazon.com拥有超过2900万顾客和数百万种目录item。其他大型零售商也有类似规模的数据源。虽然所有这些数据提供了机会，但也是一个诅咒——它压垮了为小三个数量级的数据集设计的算法。几乎所有现有算法都是在小数据集上评估的。例如，MovieLens数据集[4]包含35,000名顾客和3,000个item，EachMovie数据集[3]包含4,000名顾客和1,600个item。

**对于非常大的数据集，一个可扩展的推荐算法必须将最昂贵的计算在离线完成**。简要比较可以看出现有方法的不足：

- **传统协同过滤**很少或根本不进行离线计算，其在线计算量随顾客数量和目录item数量扩展。该算法在大数据集上不切实际，除非使用降维、采样或分区——所**有这些都会降低推荐质量**。
- **聚类模型**可以在离线完成大部分计算，但推荐质量相对较差。为提高质量，可以增加分片数量，但这会使在线用户-分片分类变得昂贵。
- **基于搜索的模型**在离线建立关键词、类别和作者索引，但无法提供有趣、有针对性标题的推荐。对于拥有大量购买和评分的顾客，它们的扩展性也很差。

item到item协同过滤的可扩展性和高性能关键在于它在离线创建昂贵的相似item表。该算法的在线部分——为用户购买和评分的item查找相似item——的复杂度与 目录大小 或 顾客总数无关；它**仅依赖于用户购买或评分的item数量**。因此，即使对于极大的数据集，算法也非常快速。由于算法推荐高度相关的相似item，推荐质量极佳[10]。与传统协同过滤不同，该算法在用户数据有限时也表现良好，仅基于两到三个item就能生成高质量的推荐。



## 结论（Conclusion）

推荐算法通过为每位顾客创建个性化购物体验，提供了一种有效的定向营销形式。对于像Amazon.com这样的大型零售商，好的推荐算法需要满足以下条件：能够在非常大的顾客群和产品目录上扩展；**生成在线推荐只需亚秒级处理时间；能够立即响应用户数据的变化；以及为所有用户（无论购买和评分的数量多少）生成有吸引力的推荐。与其他算法不同，item到item协同过滤能够应对这一挑战**。

未来，我们预计零售行业将更广泛地应用推荐算法进行线上和线下的定向营销。虽然电子商务企业拥有最便捷的个性化载体，但与传统的大规模营销方法相比，该技术更高的转化率也将使其对线下零售商具有吸引力，可用于邮寄广告、优惠券和其他形式的客户沟通。



## 参考文献（References）

[1] J.B. Schafer, J.A. Konstan, and J. Reidl, "E-Commerce Recommendation Applications," *Data Mining and Knowledge Discovery*, Kluwer Academic, 2001, pp. 115-153.

[2] P. Resnick et al., "GroupLens: An Open Architecture for Collaborative Filtering of Netnews," *Proc. ACM 1994 Conf. Computer Supported Cooperative Work*, ACM Press, 1994, pp. 175-186.

[3] J. Breese, D. Heckerman, and C. Kadie, "Empirical Analysis of Predictive Algorithms for Collaborative Filtering," *Proc. 14th Conf. Uncertainty in Artificial Intelligence*, Morgan Kaufmann, 1998, pp. 43-52.

[4] B.M. Sarwarm et al., "**Analysis of Recommendation Algorithms for E-Commerce**," *ACM Conf. Electronic Commerce*, ACM Press, 2000, pp.158-167.

[5] K. Goldberg et al., "Eigentaste: A Constant Time Collaborative Filtering Algorithm," *Information Retrieval J.*, vol. 4, no. 2, July 2001, pp. 133-151.

[6] P.S. Bradley, U.M. Fayyad, and C. Reina, "Scaling Clustering Algorithms to Large Databases," *Knowledge Discovery and Data Mining*, Kluwer Academic, 1998, pp. 9-15.

[7] L. Ungar and D. Foster, "Clustering Methods for Collaborative Filtering," *Proc. Workshop on Recommendation Systems*, AAAI Press, 1998.

[8] M. Balabanovic and Y. Shoham, "Content-Based Collaborative Recommendation," *Comm. ACM*, Mar. 1997, pp. 66-72.

[9] G.D. Linden, J.A. Jacobi, and E.A. Benson, *Collaborative Recommendations Using Item-to-Item Similarity Mappings*, US Patent 6,266,649 (to Amazon.com), Patent and Trademark Office, Washington, D.C., 2001.

[10] B.M. Sarwar et al., "**Item-Based Collaborative Filtering Recommendation Algorithms**," *10th Int'l World Wide Web Conference*, ACM Press, 2001, pp. 285-295.



## 作者简介

**Greg Linden** 曾任Amazon.com个性化组的联合创始人、研究员和高级经理，设计并开发了推荐算法。他目前是斯坦福大学商学院斯隆项目的管理学研究生。他的研究兴趣包括推荐系统、个性化、数据挖掘和人工智能。Linden在华盛顿大学获得计算机科学硕士学位。联系方式：Linden_Greg@gsb.stanford.edu。

**Brent Smith** 领导Amazon.com的自动营销团队。他的研究兴趣包括数据挖掘、机器学习和推荐系统。他在加州大学圣地亚哥分校获得数学学士学位，在华盛顿大学获得数学硕士学位（研究生期间从事微分几何研究）。联系方式：smithbr@amazon.com。

**Jeremy York** 领导Amazon.com的自动内容选择与交付团队。他的兴趣包括分类数据的统计模型、推荐系统和网站显示组件的最优选择。他在华盛顿大学获得统计学博士学位，其博士论文获得了Leonard J. Savage奖（应用贝叶斯计量经济学与统计学最佳论文奖）。联系方式：jeremy@amazon.com。
