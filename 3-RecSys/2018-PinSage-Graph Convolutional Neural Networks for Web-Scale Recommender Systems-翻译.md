# 面向Web规模推荐系统的图卷积神经网络


本文介绍了 面向Web规模推荐系统的图卷积神经网络。核心内容：


关键发现：

---


> Rex Ying\*†, Ruining He\*, Kaifeng Chen\*†, Pong Eksombatchai\*, William L. Hamilton†, Jure Leskovec\*†, , \*Pinterest, †Stanford University,  | {rhe,kaifengchen,pong}@pinterest.com, {rexying,wleif,jure}@stanford.edu


---

## 摘要

Recent advancements in deep neural networks for graph-structured data have led to state-of-the-art performance on recommender system benchmarks. However, making these methods practical and scalable to web-scale recommendation tasks with billions of items and hundreds of millions of users remains a challenge.

近年来，面向图结构的深度神经网络的进展在推荐系统基准测试上取得了最先进的性能。然而，如何使这些方法在拥有数十亿item和数亿用户的Web规模推荐任务中变得实用和可扩展，仍然是一个挑战。

Here we describe a large-scale deep recommendation engine that we developed and deployed at Pinterest. We develop a data-efficient Graph Convolutional Network (GCN) algorithm PinSage, which combines efficient random walks and graph convolutions to generate embeddings of nodes (i.e., items) that incorporate both graph structure as well as node feature information. Compared to prior GCN approaches, we develop a novel method based on highly efficient random walks to structure the convolutions and design a novel training strategy that relies on harder-and-harder training examples to improve robustness and convergence of the model.

本文描述了我们在Pinterest中开发和部署的大规模深度学习推荐引擎。我们提出了一种数据高效的图卷积网络（GCN）算法PinSage，它结合了高效随机游走和图卷积来生成节点（即item）的嵌入，该嵌入同时融合了图结构信息和节点特征信息。与先前的GCN方法相比，我们基于高效随机游走提出了一种新的卷积构建方法，并设计了一种依赖逐步变难的训练样本来提高模型鲁棒性和收敛性的新型训练策略。

We deploy PinSage at Pinterest and train it on 7.5 billion examples on a graph with 3 billion nodes representing pins and boards, and 18 billion edges. According to offline metrics, user studies and A/B tests, PinSage generates higher-quality recommendations than comparable deep learning and graph-based alternatives. To our knowledge, this is the largest application of deep graph embeddings to date and paves the way for a new generation of web-scale recommender systems based on graph convolutional architectures.

我们在Pinterest上部署了PinSage，在一个包含30亿个节点（代表Pin和画板）和180亿条边的图上进行了75亿个样本的训练。根据离线指标、用户研究和A/B测试，PinSage生成了比同类深度学习和基于图的替代方案更高质量的推荐。据我们所知，这是迄今为止最大规模的深度图嵌入应用，为基于图卷积架构的新一代Web规模推荐系统铺平了道路。

**ACM引用格式：**

Rex Ying\*†, Ruining He\*, Kaifeng Chen\*†, Pong Eksombatchai\*, William L. Hamilton†, Jure Leskovec\*†. 2018. Graph Convolutional Neural Networks for Web-Scale Recommender Systems. In KDD '18: The 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, August 19–23, 2018, London, United Kingdom. ACM, New York, NY, USA, 10 pages. https://doi.org/10.1145/3219819.3219890

---
本文介绍了 面向Web规模推荐系统的图卷积神经网络。核心内容：


关键发现：



## 1 引言

Deep learning methods have an increasingly critical role in recommender system applications, being used to learn useful low-dimensional embeddings of images, text, and even individual users [9, 12]. The representations learned using deep models can be used to complement, or even replace, traditional recommendation algorithms like collaborative filtering, and these learned representations have high utility because they can be re-used in various recommendation tasks. For example, item embeddings learned using a deep model can be used for item-item recommendation and also to recommend themed collections (e.g., playlists, or "feed" content). Recent years have seen significant developments in this space—especially the development of new deep learning methods that are capable of learning on graph-structured data, which is fundamental for recommendation applications (e.g., to exploit user-to-item interaction graphs as well as social graphs) [6, 19, 21, 24, 29, 30].

深度学习方法在推荐系统应用中扮演着越来越关键的角色，被用于学习图像、文本甚至单个用户的有用低维嵌入[9, 12]。使用深度模型学习到的表示可以用来补充甚至替代传统的推荐算法（如协同过滤），这些学习到的表示具有很高的实用性，因为它们可以在各种推荐任务中重复使用。例如，使用深度模型学习到的item嵌入可以用于item到item的推荐，也可以用于推荐主题集合（如播放列表或"动态"内容）。近年来，这一领域取得了显著进展——特别是能够在图结构数据上进行学习的新型深度学习方法的发展，这对于推荐应用（例如利用用户-item交互图以及社交图）至关重要[6, 19, 21, 24, 29, 30]。

Most prominent among these recent advancements is the success of deep learning architectures known as Graph Convolutional Networks (GCNs) [19, 21, 24, 29]. The core idea behind GCNs is to learn how to iteratively aggregate feature information from local graph neighborhoods using neural networks (Figure 1). Here a single "convolution" operation transforms and aggregates feature information from a node's one-hop graph neighborhood, and by stacking multiple such convolutions information can be propagated across far reaches of a graph. Unlike purely content-based deep models (e.g., recurrent neural networks [3]), GCNs leverage both content information as well as graph structure. GCN-based methods have set a new standard on countless recommender system benchmarks (see [19] for a survey). However, these gains on benchmark tasks have yet to be translated to gains in real-world production environments.

在这些最新进展中，最突出的当属被称为图卷积网络（GCN）的深度学习架构的成功[19, 21, 24, 29]。GCN的核心思想是学习如何通过神经网络从局部图邻域中迭代地聚合特征信息（图1）。单个"卷积"操作转换并聚合来自节点一跳图邻域的特征信息，通过堆叠多个这样的卷积，信息可以在图的远端传播。与纯基于内容的深度模型（如循环神经网络[3]）不同，GCN同时利用了内容信息和图结构。基于GCN的方法在无数推荐系统基准测试中树立了新标准（参见[19]的综述）。然而，这些在基准任务上的收益尚未转化为实际生产环境中的性能提升。

The main challenge is to scale both the training as well as inference of GCN-based node embeddings to graphs with billions of nodes and tens of billions of edges. Scaling up GCNs is difficult because many of the core assumptions underlying their design are violated when working in a big data environment. For example, all existing GCN-based recommender systems require operating on the full graph Laplacian during training—an assumption that is infeasible when the underlying graph has billions of nodes and whose structure is constantly evolving.

主要的挑战在于将基于GCN的节点嵌入的训练和推理扩展到拥有数十亿节点和数百亿边的图。扩展GCN是困难的，因为其设计所依赖的许多核心假设在大数据环境下都不再成立。例如，所有现有的基于GCN的推荐系统都要求在训练过程中操作完整的图拉普拉斯矩阵——当底层图拥有数十亿节点且其结构不断变化时，这一假设是不可行的。

**当前工作.** Here we present a highly-scalable GCN framework that we have developed and deployed in production at Pinterest. Our framework, a random-walk-based GCN named PinSage, operates on a massive graph with 3 billion nodes and 18 billion edges—a graph that is 10,000$\times$ larger than typical applications of GCNs. PinSage leverages several key insights to drastically improve the scalability of GCNs:

本文提出了一个高度可扩展的GCN框架，我们已在Pinterest中开发并部署到生产环境中。我们的框架——一个基于随机游走的GCN，名为PinSage——在一个包含30亿节点和180亿边的大规模图上运行，该图比GCN的典型应用大10,000倍。PinSage利用了几个关键洞察来大幅提高GCN的可扩展性：

- **On-the-fly convolutions:** Traditional GCN algorithms perform graph convolutions by multiplying feature matrices by powers of the full graph Laplacian. In contrast, our PinSage algorithm performs efficient, localized convolutions by sampling the neighborhood around a node and dynamically constructing a computation graph from this sampled neighborhood. These dynamically constructed computation graphs (Fig. 1) specify how to perform a localized convolution around a particular node, and alleviate the need to operate on the entire graph during training.

- **即时卷积：** 传统的GCN算法通过将特征矩阵乘以完整图拉普拉斯矩阵的幂来执行图卷积。相比之下，我们的PinSage算法通过对节点周围的邻域进行采样，并从这个采样的邻域动态构建计算图，来执行高效的局部卷积。这些动态构建的计算图（图1）指定了如何在特定节点周围执行局部卷积，避免了在训练期间操作整个图的需要。

- **Producer-consumer minibatch construction:** We develop a producer-consumer architecture for constructing minibatches that ensures maximal GPU utilization during model training. A large-memory, CPU-bound producer efficiently samples node network neighborhoods and fetches the necessary features to define local convolutions, while a GPU-bound TensorFlow model consumes these pre-defined computation graphs to efficiently run stochastic gradient descent.

- **生产者-消费者小批量构建：** 我们开发了一种用于构建小批量的生产者-消费者架构，确保在模型训练期间实现最大的GPU利用率。一个拥有大内存的CPU密集型生产者高效地采样节点网络邻域并获取定义局部卷积所需的特征，而一个GPU密集型的TensorFlow模型则消费这些预定义的计算图来高效地运行随机梯度下降。

- **Efficient MapReduce inference:** Given a fully-trained GCN model, we design an efficient MapReduce pipeline that can distribute the trained model to generate embeddings for billions of nodes, while minimizing repeated computations.

- **高效MapReduce推理：** 给定一个完全训练好的GCN模型，我们设计了一个高效的MapReduce流水线，可以分发训练好的模型来生成数十亿节点的嵌入，同时最小化重复计算。

In addition to these fundamental advancements in scalability, we also introduce new training techniques and algorithmic innovations. These innovations improve the quality of the representations learned by PinSage, leading to significant performance gains in downstream recommender system tasks:

除了这些在可扩展性方面的基本进展之外，我们还引入了新的训练技术和算法创新。这些创新提高了PinSage学习到的表示质量，从而在下游推荐系统任务中带来了显著的性能提升：

- **Constructing convolutions via random walks:** Taking full neighborhoods of nodes to perform convolutions (Fig. 1) would result in huge computation graphs, so we resort to sampling. However, random sampling is suboptimal, and we develop a new technique using short random walks to sample the computation graph. An additional benefit is that each node now has an importance score, which we use in the pooling/aggregation step.

- **通过随机游走构建卷积：** 采用节点的完整邻域来执行卷积（图1）会导致巨大的计算图，因此我们采用采样方法。然而，随机采样并非最优，我们开发了一种使用短随机游走来采样计算图的新技术。另一个好处是每个节点现在都有一个重要性分数，我们在池化/聚合步骤中使用该分数。

- **Importance pooling:** A core component of graph convolutions is the aggregation of feature information from local neighborhoods in the graph. We introduce a method to weigh the importance of node features in this aggregation based upon random-walk similarity measures, leading to a 46% performance gain in offline evaluation metrics.

- **重要性池化：** 图卷积的一个核心组件是从图中的局部邻域聚合特征信息。我们引入了一种方法，基于随机游走相似度度量来权衡节点特征在此聚合中的重要性，从而在离线评估指标中带来了46%的性能提升。

- **Curriculum training:** We design a curriculum training scheme, where the algorithm is fed harder-and-harder examples during training, resulting in a 12% performance gain.

- **课程训练：** 我们设计了一种课程训练方案，在训练过程中给算法喂入渐进变难的样本，从而带来了12%的性能提升。

We have deployed PinSage for a variety of recommendation tasks at Pinterest, a popular content discovery and curation application where users interact with pins, which are visual bookmarks to online content (e.g., recipes they want to cook, or clothes they want to purchase). Users organize these pins into boards, which contain collections of similar pins. Altogether, Pinterest is the world's largest user-curated graph of images, with over 2 billion unique pins collected into over 1 billion boards.

我们已在Pinterest上为多种推荐任务部署了PinSage。Pinterest是一款流行的内容发现和策展应用，用户在其中与Pin进行交互，Pin是在线内容的视觉书签（例如他们想烹饪的食谱或想购买的衣物）。用户将这些Pin组织到画板中，画板包含相似Pin的集合。总的来说，Pinterest是全球最大的用户策展图像图，拥有超过20亿个独特的Pin，被收集到超过10亿个画板中。

Through extensive offline metrics, controlled user studies, and A/B tests, we show that our approach achieves state-of-the-art performance compared to other scalable deep content-based recommendation algorithms, in both an item-item recommendation task (i.e., related-pin recommendation), as well as a "homefeed" recommendation task. In offline ranking metrics we improve over the best performing baseline by more than 40%, in head-to-head human evaluations our recommendations are preferred about 60% of the time, and the A/B tests show 30% to 100% improvements in user engagement across various settings.

通过广泛的离线指标、受控用户研究和A/B测试，我们展示了我们的方法在item到item推荐任务（即相关Pin推荐）和"首页动态"推荐任务中，均取得了比其他可扩展的深度内容推荐算法更先进的性能。在离线排名指标中，我们比最佳基线提升了40%以上；在人工对比评估中，我们的推荐在大约60%的情况下被偏好；A/B测试显示，在各种设置下用户参与度提升了30%到100%。

To our knowledge, this is the largest-ever application of deep graph embeddings and paves the way for a new generation of recommendation systems based on graph convolutional architectures.

据我们所知，这是迄今为止最大规模的深度图嵌入应用，为基于图卷积架构的新一代推荐系统铺平了道路。

---

## 2 相关工作

Our work builds upon a number of recent advancements in deep learning methods for graph-structured data.

我们的工作建立在图结构数据深度学习方法的一些最新进展之上。

The notion of neural networks for graph data was first outlined in Gori et al. (2005) [15] and further elaborated on in Scarselli et al. (2009) [27]. However, these initial approaches to deep learning on graphs required running expensive neural "message-passing" algorithms to convergence and were prohibitively expensive on large graphs. Some limitations were addressed by Gated Graph Sequence Neural Networks [22]—which employs modern recurrent neural architectures—but the approach remains computationally expensive and has mainly been used on graphs with <10,000 nodes.

面向图数据的神经网络概念最早由Gori等人（2005）[15]提出，并由Scarselli等人（2009）[27]进一步阐述。然而，这些早期的图深度学习方法需要运行昂贵的神经"消息传递"算法直至收敛，在大规模图上成本过高。门控图序列神经网络[22]采用了现代循环神经架构，解决了一些局限性，但该方法仍然计算成本高昂，主要被用于节点数少于10,000的图。

More recently, there has been a surge of methods that rely on the notion of "graph convolutions" or Graph Convolutional Networks (GCNs). This approach originated with the work of Bruna et al. (2013), which developed a version of graph convolutions based on spectral graph theory [7]. Following this work, a number of authors proposed improvements, extensions, and approximations of these spectral convolutions [6, 10, 11, 13, 18, 21, 24, 29, 31], leading to new state-of-the-art results on benchmarks such as node classification, link prediction, as well as recommender system tasks (e.g., the MovieLens benchmark [24]). These approaches have consistently outperformed techniques based upon matrix factorization or random walks (e.g., node2vec [17] and DeepWalk [26]), and their success has led to a surge of interest in applying GCN-based methods to applications ranging from recommender systems [24] to drug design [20, 31]. Hamilton et al. (2017b) [19] and Bronstein et al. (2017) [6] provide comprehensive surveys of recent advancements.

最近，涌现了大量依赖于"图卷积"或图卷积网络（GCN）概念的方法。这种方法起源于Bruna等人（2013）的工作，他们基于谱图理论开发了一种图卷积版本[7]。在这项工作之后，许多研究者提出了对这些谱卷积的改进、扩展和近似[6, 10, 11, 13, 18, 21, 24, 29, 31]，在节点分类、链接预测以及推荐系统任务（如MovieLens基准[24]）等基准测试中取得了新的最先进成果。这些方法一致地优于基于矩阵分解或随机游走的技术（如node2vec[17]和DeepWalk[26]），它们的成功引发了将基于GCN的方法应用于从推荐系统[24]到药物设计[20, 31]等各个领域的兴趣激增。Hamilton等人（2017b）[19]和Bronstein等人（2017）[6]提供了对最新进展的全面综述。

However, despite the successes of GCN algorithms, no previous works have managed to apply them to production-scale data with billions of nodes and edges—a limitation that is primarily due to the fact that traditional GCN methods require operating on the entire graph Laplacian during training. Here we fill this gap and show that GCNs can be scaled to operate in a production-scale recommender system setting involving billions of nodes/items. Our work also demonstrates the substantial impact that GCNs have on recommendation performance in a real-world environment.

然而，尽管GCN算法取得了成功，但此前没有工作能够将其应用于拥有数十亿节点和边的生产规模数据——这一限制主要是由于传统的GCN方法需要在训练过程中操作完整的图拉普拉斯矩阵。本文填补了这一空白，并展示了GCN可以被扩展到涉及数十亿节点/item的生产规模推荐系统环境中。我们的工作也展示了GCN在真实世界环境中对推荐性能的显著影响。

In terms of algorithm design, our work is most closely related to Hamilton et al. (2017a)'s GraphSAGE algorithm [18] and the closely related follow-up work of Chen et al. (2018) [8]. GraphSAGE is an inductive variant of GCNs that we modify to avoid operating on the entire graph Laplacian. We fundamentally improve upon GraphSAGE by removing the limitation that the whole graph be stored in GPU memory, using low-latency random walks to sample graph neighborhoods in a producer-consumer architecture. We also introduce a number of new training techniques to improve performance and a MapReduce inference pipeline to scale up to graphs with billions of nodes.

在算法设计方面，我们的工作与Hamilton等人（2017a）的GraphSAGE算法[18]以及Chen等人（2018）密切相关的后续工作[8]最为相似。GraphSAGE是GCN的一种归纳式变体，我们对其进行了修改以避免操作整个图拉普拉斯矩阵。我们从根本上改进了GraphSAGE，消除了将整个图存储在GPU内存中的限制，使用低延迟随机游走在生产者-消费者架构中对图邻域进行采样。我们还引入了一些新的训练技术以提高性能，以及一个MapReduce推理流水线来扩展到拥有数十亿节点的图。

Lastly, also note that graph embedding methods like node2vec [17] and DeepWalk [26] cannot be applied here. First, these are unsupervised methods. Second, they cannot include node feature information. Third, they directly learn embeddings of nodes and thus the number of model parameters is linear with the size of the graph, which is prohibitive for our setting.

最后，还需要注意的是像node2vec[17]和DeepWalk[26]这样的图嵌入方法无法在此应用。首先，这些是无监督方法。其次，它们无法包含节点特征信息。第三，它们直接学习节点的嵌入，因此模型参数的数量与图的大小呈线性关系，这在我们的大规模场景下是不可行的。

---

## 3 方法

In this section, we describe the technical details of the PinSage architecture and training, as well as a MapReduce pipeline to efficiently generate embeddings using a trained PinSage model.

在本节中，我们将描述PinSage架构和训练的技术细节，以及使用训练好的PinSage模型高效生成嵌入的MapReduce流水线。

The key computational workhorse of our approach is the notion of localized graph convolutions. To generate the embedding for a node (i.e., an item), we apply multiple convolutional modules that aggregate feature information (e.g., visual, textual features) from the node's local graph neighborhood (Figure 1). Each module learns how to aggregate information from a small graph neighborhood, and by stacking multiple such modules, our approach can gain information about the local network topology. Importantly, parameters of these localized convolutional modules are shared across all nodes, making the parameter complexity of our approach independent of the input graph size.

我们方法的关键计算引擎是局部图卷积的概念。为了生成节点（即item）的嵌入，我们应用多个卷积模块来从节点的局部图邻域聚合特征信息（例如视觉、文本特征）（图1）。每个模块学习如何从小型图邻域中聚合信息，通过堆叠多个这样的模块，我们的方法可以获得关于局部网络拓扑的信息。重要的是，这些局部卷积模块的参数在所有节点间共享，使得我们方法的参数复杂度与输入图的大小无关。

### 3.1 问题设定

Pinterest is a content discovery application where users interact with pins, which are visual bookmarks to online content (e.g., recipes they want to cook, or clothes they want to purchase). Users organize these pins into boards, which contain collections of pins that the user deems to be thematically related. Altogether, the Pinterest graph contains 2 billion pins, 1 billion boards, and over 18 billion edges (i.e., memberships of pins to their corresponding boards).

Pinterest是一款内容发现应用，用户在其中与Pin（在线内容的视觉书签，例如他们想烹饪的食谱或想购买的衣物）交互。用户将这些Pin组织到画板中，画板包含用户认为主题相关的Pin集合。总的来说，Pinterest图包含20亿个Pin、10亿个画板以及超过180亿条边（即Pin与其对应画板的隶属关系）。

Our task is to generate high-quality embeddings or representations of pins that can be used for recommendation (e.g., via nearest-neighbor lookup for related pin recommendation, or for use in a downstream re-ranking system). In order to learn these embeddings, we model the Pinterest environment as a bipartite graph consisting of nodes in two disjoint sets, I (containing pins) and C (containing boards). Note, however, that our approach is also naturally generalizable, with I being viewed as a set of items and C as a set of user-defined contexts or collections.

我们的任务是生成可用于推荐的Pin的高质量嵌入或表示（例如，通过最近邻查找进行相关Pin推荐，或用于下游重排序系统）。为了学习这些嵌入，我们将Pinterest环境建模为一个二分图，由两个不相交的节点集合组成：I（包含Pin）和C（包含画板）。但需要注意的是，我们的方法也可以自然地推广，将I视为item集合，将C视为用户定义的上下文或集合。

In addition to the graph structure, we also assume that the pins/items u \in I are associated with real-valued attributes, xu \in R^d. In general, these attributes may specify metadata or content information about an item, and in the case of Pinterest, we have that pins are associated with both rich text and image features. Our goal is to leverage both these input attributes as well as the structure of the bipartite graph to generate high-quality embeddings. These embeddings are then used for recommender system candidate generation via nearest neighbor lookup (i.e., given a pin, find related pins) or as features in machine learning systems for ranking the candidates.

除了图结构之外，我们还假设Pin/itemu \in I与实值属性xu \in R^d相关联。一般来说，这些属性可以指定item的元数据或内容信息，在Pinterest的情况下，Pin与丰富的文本和图像特征相关联。我们的目标是利用这些输入属性以及二分图的结构来生成高质量嵌入。这些嵌入随后用于推荐系统的候选生成（通过最近邻查找，即给定一个Pin，找到相关的Pin）或作为机器学习系统中用于对候选进行排序的特征。

For notational convenience and generality, when we describe the PinSage algorithm, we simply refer to the node set of the full graph with V = I \cup C and do not explicitly distinguish between pin and board nodes (unless strictly necessary), using the more general term "node" whenever possible.

为了符号上的便利和通用性，在描述PinSage算法时，我们简单地将整个图的节点集记为V = I \cup C，并且不明确区分Pin节点和画板节点（除非严格必要），尽可能使用更通用的术语"节点"。

### 3.2 模型架构

We use localized convolutional modules to generate embeddings for nodes. We start with input node features and then learn neural networks that transform and aggregate features over the graph to compute the node embeddings (Figure 1).

我们使用局部卷积模块来生成节点的嵌入。我们从输入节点特征开始，然后学习在图之上转换和聚合特征的神经网络，以计算节点嵌入（图1）。

**前向传播算法.** We consider the task of generating an embedding, zu for a node u, which depends on the node's input features and the graph structure around this node.

我们考虑为节点u生成嵌入zu的任务，该嵌入依赖于节点的输入特征和该节点周围的图结构。

**算法1: convolve（卷积）**

输入:
- 节点u的当前嵌入zu
- 邻居嵌入集合{zv | v \in N(u)}
- 邻居权重集合\alpha
- 对称向量函数\gamma(·)

输出: 节点u的新嵌入z^new_u

1: nu \leftarrow \gamma({ReLU(Qhv + q) | v \in N(u)}, \alpha)
2: z^new_u \leftarrow ReLU(W · concat(zu, nu) + w)
3: z^new_u \leftarrow z^new_u / ∥z^new_u∥_2

The core of our PinSage algorithm is a localized convolution operation, where we learn how to aggregate information from u's neighborhood (Figure 1). This procedure is detailed in Algorithm 1 convolve. The basic idea is that we transform the representations zv, \forallv \in N(u) of u's neighbors through a dense neural network and then apply an aggregator/pooling function (e.g., an element-wise mean or weighted sum, denoted as \gamma) on the resulting set of vectors (Line 1). This aggregation step provides a vector representation, nu, of u's local neighborhood, N(u). We then concatenate the aggregated neighborhood vector nu with u's current representation hu and transform the concatenated vector through another dense neural network layer (Line 2). Empirically we observe significant performance gains when using concatenation operation instead of the average operation as in [21]. Additionally, the normalization in Line 3 makes training more stable, and it is more efficient to perform approximate nearest neighbor search for normalized embeddings (Section 3.5). The output of the algorithm is a representation of u that incorporates both information about itself and its local graph neighborhood.

我们PinSage算法的核心是局部卷积操作，它学习如何从u的邻域聚合信息（图1）。这个过程在算法1 convolve中详细说明。基本思想是通过一个稠密神经网络转换u的邻居的表示zv（\forallv \in N(u)），然后对结果向量集应用一个聚合器/池化函数（例如逐元素均值或加权和，记为\gamma）（第1行）。这个聚合步骤提供了u的局部邻域N(u)的向量表示nu。然后我们将聚合后的邻域向量nu与u的当前表示hu进行拼接，并通过另一个稠密神经网络层来转换拼接后的向量（第2行）。根据经验，我们观察到使用拼接操作（而非[21]中的平均操作）可以带来显著的性能提升。此外，第3行的归一化使得训练更加稳定，并且对归一化后的嵌入执行近似最近邻搜索也更加高效（第3.5节）。算法的输出是u的一个表示，它融合了关于自身及其局部图邻域的信息。

**基于重要性的邻域.** An important innovation in our approach is how we define node neighborhoods N(u), i.e., how we select the set of neighbors to convolve over in Algorithm 1. Whereas previous GCN approaches simply examine k-hop graph neighborhoods, in PinSage we define importance-based neighborhoods, where the neighborhood of a node u is defined as the T nodes that exert the most influence on node u. Concretely, we simulate random walks starting from node u and compute the L1-normalized visit count of nodes visited by the random walk [14]. The neighborhood of u is then defined as the top T nodes with the highest normalized visit counts with respect to node u.

我们方法中的一个重要创新是定义节点邻域N(u)的方式，即在算法1中选择哪些邻居进行卷积。先前的GCN方法只是简单地考察k跳图邻域，而在PinSage中，我们定义了基于重要性的邻域，其中节点u的邻域被定义为对节点u影响最大的T个节点。具体来说，我们模拟从节点u开始的随机游走，并计算被随机游走访问的节点的L1归一化访问次数[14]。然后，节点u的邻域被定义为相对于节点u具有最高归一化访问次数的前T个节点。

The advantages of this importance-based neighborhood definition are two-fold. First, selecting a fixed number of nodes to aggregate from allows us to control the memory footprint of the algorithm during training [18]. Second, it allows Algorithm 1 to take into account the importance of neighbors when aggregating the vector representations of neighbors. In particular, we implement \gamma in Algorithm 1 as a weighted-mean, with weights defined according to the L1 normalized visit counts. We refer to this new approach as importance pooling.

这种基于重要性的邻域定义有两个优点。首先，选择固定数量的节点进行聚合使我们能够在训练期间控制算法的内存占用[18]。其次，它使算法1在聚合邻居的向量表示时能够考虑邻居的重要性。具体来说，我们将算法1中的\gamma实现为加权平均，权重根据L1归一化的访问次数定义。我们将这种新方法称为重要性池化。

**堆叠卷积.** Each time we apply the convolve operation (Algorithm 1) we get a new representation for a node, and we can stack multiple such convolutions on top of each other in order to gain more information about the local graph structure around node u. In particular, we use multiple layers of convolutions, where the inputs to the convolutions at layer k depend on the representations output from layer k - 1 (Figure 1) and where the initial (i.e., "layer 0") representations are equal to the input node features. Note that the model parameters in Algorithm 1 (Q, q, W, and w) are shared across the nodes but differ between layers.

每次应用convolve操作（算法1）时，我们都会得到节点的一个新表示，并且我们可以将多个这样的卷积彼此堆叠，以获得关于节点u周围局部图结构的更多信息。具体来说，我们使用多个卷积层，其中第k层卷积的输入依赖于第k-1层输出的表示（图1），且初始（即"第0层"）表示等于输入节点特征。注意，算法1中的模型参数（Q, q, W和w）在节点之间共享，但在不同层之间有所不同。

**算法2: minibatch（小批量）**

输入:
- 节点集合 M ⊂ V
- 深度参数 K
- 邻域函数 N: V \rightarrow 2^V

输出: 嵌入 zu, \forallu \in M

/* 采样小批量节点的邻域 */
1: S^(K) \leftarrow M
2: for k = K, ..., 1 do
3:   S^(k-1) \leftarrow S^(k)
4:   for u \in S^(k) do
5:     S^(k-1) \leftarrow S^(k-1) \cup N(u)
6:   end
7: end

/* 生成嵌入 */
8: h^(0)_u \leftarrow xu, \forallu \in S^(0)
9: for k = 1, ..., K do
10:   for u \in S^(k) do
11:     H \leftarrow {h^(k-1)_v, \forallv \in N(u)}
12:     h^(k)_u \leftarrow convolve^(k)(h^(k-1)_u, H)
13:   end
14: end
15: for u \in M do
16:   zu \leftarrow G2 · ReLU(G1 h^(K)_u + g)
17: end

Algorithm 2 details how stacked convolutions generate embeddings for a minibatch set of nodes, M. We first compute the neighborhoods of each node and then apply K convolutional iterations to generate the layer-K representations of the target nodes. The output of the final convolutional layer is then fed through a fully-connected neural network to generate the final output embeddings zu, \forallu \in M.

算法2详细说明了堆叠卷积如何为小批量节点集M生成嵌入。我们首先计算每个节点的邻域，然后应用K次卷积迭代来生成目标节点的第K层表示。然后，最后一个卷积层的输出通过一个全连接神经网络来生成最终输出嵌入zu（\forallu \in M）。

The full set of parameters of our model which we then learn is: the weight and bias parameters for each convolutional layer (Q^(k), q^(k), W^(k), w^(k), \forallk \in {1, ..., K}) as well as the parameters of the final dense neural network layer, G1, G2, and g. The output dimension of Line 1 in Algorithm 1 (i.e., the column-space dimension of Q) is set to be m at all layers. For simplicity, we set the output dimension of all convolutional layers (i.e., the output at Line 3 of Algorithm 1) to be equal, and we denote this size parameter by d. The final output dimension of the model (after applying line 18 of Algorithm 2) is also set to be d.

我们学习的模型的完整参数集包括：每个卷积层的权重和偏置参数（Q^(k), q^(k), W^(k), w^(k), \forallk \in {1, ..., K}），以及最终稠密神经网络层G1, G2, g的参数。算法1第1行的输出维度（即Q的列空间维度）在所有层中设置为m。为简单起见，我们设置所有卷积层的输出维度（即算法1第3行的输出）相等，并将这个大小参数记为d。模型的最终输出维度（在应用算法2第18行之后）也设置为d。

### 3.3 模型训练

We train PinSage in a supervised fashion using a max-margin ranking loss. In this setup, we assume that we have access to a set of labeled pairs of items L, where the pairs in the set, (q, i) \in L, are assumed to be related—i.e., we assume that if (q, i) \in L then item i is a good recommendation candidate for query item q. The goal of the training phase is to optimize the PinSage parameters so that the output embeddings of pairs (q, i) \in L in the labeled set are close together.

我们使用最大间隔排序损失以监督方式训练PinSage。在这个设置中，我们假设有一个标注的item对集合L，其中的对(q, i) \in L被认为是相关的——即，我们假设如果(q, i) \in L，那么itemi是查询itemq的一个好的推荐候选。训练阶段的目标是优化PinSage的参数，使得标注集中的对(q, i) \in L的输出嵌入在嵌入空间中彼此靠近。

We first describe our margin-based loss function in detail. Following this, we give an overview of several techniques we developed that lead to the computation efficiency and fast convergence rate of PinSage, allowing us to train on billion node graphs and billions of training examples. And finally, we describe our curriculum-training scheme, which improves the overall quality of the recommendations.

我们首先详细描述基于间隔的损失函数。然后，我们概述我们开发的几种技术，这些技术带来了PinSage的计算效率和快速收敛速度，使我们能够在数十亿节点图和数十亿训练样本上进行训练。最后，我们描述我们的课程训练方案，它提高了推荐的整体质量。

**损失函数.** In order to train the parameters of the model, we use a max-margin-based loss function. The basic idea is that we want to maximize the inner product of positive examples, i.e., the embedding of the query item and the corresponding related item. At the same time we want to ensure that the inner product of negative examples—i.e., the inner product between the embedding of the query item and an unrelated item—is smaller than that of the positive sample by some pre-defined margin. The loss function for a single pair of node embeddings (zq, zi) : (q, i) \in L is thus:

为了训练模型的参数，我们使用基于最大间隔的损失函数。基本思想是我们想要最大化正例的内积，即查询item的嵌入与对应的相关item的嵌入。同时，我们想要确保负例的内积——即查询item嵌入与不相关item嵌入之间的内积——比正样本的内积小某个预定义的间隔。因此，单对节点嵌入(zq, zi) : (q, i) \in L的损失函数为：

J_G(zq, zi) = $E_{nk ~ Pn(q)}$ max{0, zq · znk - zq · zi + $\Delta$}   (1)

where Pn(q) denotes the distribution of negative examples for item q, and $\Delta$ denotes the margin hyper-parameter. We shall explain the sampling of negative samples below.

其中Pn(q)表示itemq的负例分布，$\Delta$表示间隔超参数。我们将在下面解释负样本的采样。

**使用大批量的多GPU训练.** To make full use of multiple GPUs on a single machine for training, we run the forward and backward propagation in a multi-tower fashion. With multiple GPUs, we first divide each minibatch (Figure 1 bottom) into equal-sized portions. Each GPU takes one portion of the minibatch and performs the computations using the same set of parameters. After backward propagation, the gradients for each parameter across all GPUs are aggregated together, and a single step of synchronous SGD is performed. Due to the need to train on extremely large number of examples (on the scale of billions), we run our system with large batch sizes, ranging from 512 to 4096.

为了充分利用单台机器上的多个GPU进行训练，我们以多塔方式进行前向和反向传播。对于多个GPU，我们首先将每个小批量（图1底部）划分为大小相等的部分。每个GPU取小批量的一部分，使用同一组参数执行计算。在反向传播之后，所有GPU上每个参数的梯度被聚合在一起，并执行单步同步SGD。由于需要在极大量样本上训练（数十亿级别），我们使用大批量大小运行系统，范围从512到4096。

We use techniques similar to those proposed by Goyal et al. [16] to ensure fast convergence and maintain training and generalization accuracy when dealing with large batch sizes. We use a gradual warmup procedure that increases learning rate from small to a peak value in the first epoch according to the linear scaling rule. Afterwards the learning rate is decreased exponentially.

我们使用类似于Goyal等人[16]提出的技术，以确保在处理大批量时快速收敛并保持训练和泛化精度。我们使用一个渐进热身过程，根据线性缩放规则，在第一个epoch中将学习率从小到大增加到峰值。之后，学习率呈指数衰减。

**生产者-消费者小批量构建.** During training, the adjacency list and the feature matrix for billions of nodes are placed in CPU memory due to their large size. However, during the convolve step of PinSage, each GPU process needs access to the neighborhood and feature information of nodes in the neighborhood. Accessing the data in CPU memory from GPU is not efficient. To solve this problem, we use a re-indexing technique to create a sub-graph G' = (V', E') containing nodes and their neighborhood, which will be involved in the computation of the current minibatch. A small feature matrix containing only node features relevant to computation of the current minibatch is also extracted such that the order is consistent with the index of nodes in G'. The adjacency list of G' and the small feature matrix are fed into GPUs at the start of each minibatch iteration, so that no communication between the GPU and CPU is needed during the convolve step, greatly improving GPU utilization.

在训练期间，由于数十亿节点的邻接表和特征矩阵规模巨大，它们被放置在CPU内存中。然而，在PinSage的卷积步骤中，每个GPU进程需要访问邻域中节点的邻域和特征信息。从GPU访问CPU内存中的数据效率不高。为了解决这个问题，我们使用重索引技术创建一个包含节点及其邻域的子图G' = (V', E')，该子图将参与当前小批量的计算。还提取一个仅包含与当前小批量计算相关的节点特征的小型特征矩阵，其顺序与G'中节点的索引一致。在每个小批量迭代开始时，将G'的邻接表和这个小特征矩阵送入GPU，这样在卷积步骤期间无需GPU和CPU之间的通信，大大提高了GPU利用率。

The training procedure has alternating usage of CPUs and GPUs. The model computations are in GPUs, whereas extracting features, re-indexing, and negative sampling are computed on CPUs. In addition to parallelizing GPU computation with multi-tower training, and CPU computation using OpenMP [25], we design a producer-consumer pattern to run GPU computation at the current iteration and CPU computation at the next iteration in parallel. This further reduces the training time by almost a half.

训练过程交替使用CPU和GPU。模型计算在GPU上进行，而特征提取、重索引和负采样在CPU上计算。除了使用多塔训练并行化GPU计算和使用OpenMP[25]并行化CPU计算之外，我们还设计了一种生产者-消费者模式，同时并行运行当前迭代的GPU计算和下一次迭代的CPU计算。这进一步将训练时间减少了近一半。

**采样负样本.** Negative sampling is used in our loss function (Equation 1) as an approximation of the normalization factor of edge likelihood [23]. To improve efficiency when training with large batch sizes, we sample a set of 500 negative items to be shared by all training examples in each minibatch. This drastically saves the number of embeddings that need to be computed during each training step, compared to running negative sampling for each node independently. Empirically, we do not observe a difference between the performance of the two sampling schemes.

在我们的损失函数（公式1）中，负采样被用作边似然归一化因子的近似[23]。为了提高大批量训练时的效率，我们采样一组500个负item，由每个小批量中的所有训练样本共享。与为每个节点独立运行负采样相比，这大大节省了每个训练步骤中需要计算的嵌入数量。根据经验，我们没有观察到两种采样方案之间的性能差异。

In the simplest case, we could just uniformly sample negative examples from the entire set of items. However, ensuring that the inner product of the positive example (pair of items (q, i)) is larger than that of the q and each of the 500 negative items is too "easy" and does not provide fine enough "resolution" for the system to learn. In particular, our recommendation algorithm should be capable of finding 1,000 most relevant items to q among the catalog of over 2 billion items. In other words, our model should be able to distinguish/identify 1 item out of 2 million items. But with 500 random negative items, the model's resolution is only 1 out of 500. Thus, if we sample 500 random negative items out of 2 billion items, the chance of any of these items being even slightly related to the query item is small. Therefore, with large probability the learning will not make good parameter updates and will not be able to differentiate slightly related items from the very related ones.

在最简单的情况下，我们可以从整个item集中均匀采样负例。然而，确保正例（item对(q, i)）的内积大于q与每个500个负item的内积太"容易"了，无法为系统学习提供足够精细的"分辨率"。具体来说，我们的推荐算法应该能够在超过20亿item的目录中找出与q最相关的1000个item。换句话说，我们的模型应该能够从200万个item中区分/识别出1个item。但是，使用500个随机负item时，模型的分辨率仅为1/500。因此，如果我们从20亿item中采样500个随机负item，这些item中任何一个与查询item稍微相关的可能性都很小。因此，学习过程很有可能会无法做出好的参数更新，也无法区分略微相关的item和非常相关的item。

To solve the above problem, for each positive training example (i.e., item pair (q, i)), we add "hard" negative examples, i.e., items that are somewhat related to the query item q, but not as related as the positive item i. We call these "hard negative items". They are generated by ranking items in a graph according to their Personalized PageRank scores with respect to query item q [14]. Items ranked at 2000-5000 are randomly sampled as hard negative items. As illustrated in Figure 2, the hard negative examples are more similar to the query than random negative examples, and are thus challenging for the model to rank, forcing the model to learn to distinguish items at a finer granularity.

为了解决上述问题，对于每个正训练样本（即item对(q, i)），我们添加"困难"负例，即与查询itemq有些相关但不如正itemi那么相关的item。我们称之为"困难负item"。它们是通过对图中的item根据其相对于查询itemq的个性化PageRank分数进行排序生成的[14]。排名在2000-5000的item被随机采样为困难负item。如图2所示，困难负例比随机负例与查询更相似，因此对模型的排序更具挑战性，迫使模型学习以更细粒度区分item。

Using hard negative items throughout the training procedure doubles the number of epochs needed for the training to converge. To help with convergence, we develop a curriculum training scheme [4]. In the first epoch of training, no hard negative items are used, so that the algorithm quickly finds an area in the parameter space where the loss is relatively small. We then add hard negative items in subsequent epochs, focusing the model to learn how to distinguish highly related pins from only slightly related ones. At epoch n of the training, we add n - 1 hard negative items to the set of negative items for each item.

在整个训练过程中使用困难负item会使训练收敛所需的epoch数翻倍。为了帮助收敛，我们开发了一种课程训练方案[4]。在训练的第一个epoch中，不使用困难负item，这样算法可以快速找到参数空间中损失相对较小的区域。然后我们在后续的epoch中添加困难负item，使模型专注于学习如何区分高度相关的Pin和仅略微相关的Pin。在训练的第n个epoch中，我们为每个item的负item集添加n - 1个困难负item。

### 3.4 通过MapReduce生成节点嵌入

After the model is trained, it is still challenging to directly apply the trained model to generate embeddings for all items, including those that were not seen during training. Naively computing embeddings for nodes using Algorithm 2 leads to repeated computations caused by the overlap between K-hop neighborhoods of nodes. As illustrated in Figure 1, many nodes are repeatedly computed at multiple layers when generating the embeddings for different target nodes. To ensure efficient inference, we develop a MapReduce approach that runs model inference without repeated computations.

在模型训练完成后，直接应用训练好的模型来为所有item（包括训练期间未见过的item）生成嵌入仍然具有挑战性。使用算法2朴素地计算节点嵌入会导致由于节点的K跳邻域之间的重叠而引起的重复计算。如图1所示，在为不同目标节点生成嵌入时，许多节点在多个层中被重复计算。为了确保高效的推理，我们开发了一种MapReduce方法，可以无重复计算地运行模型推理。

We observe that inference of node embeddings very nicely lends itself to MapReduce computational model. Figure 3 details the data flow on the bipartite pin-to-board Pinterest graph, where we assume the input (i.e., "layer-0") nodes are pins/items (and the layer-1 nodes are boards/contexts). The MapReduce pipeline has two key parts:

我们观察到节点嵌入的推理非常适合MapReduce计算模型。图3详细描述了在Pinterest二分Pin-画板图上的数据流，其中我们假设输入（即"第0层"）节点是Pin/item（而第1层节点是画板/上下文）。MapReduce流水线有两个关键部分：

(1) One MapReduce job is used to project all pins to a low-dimensional latent space, where the aggregation operation will be performed (Algorithm 1, Line 1).
(2) Another MapReduce job is then used to join the resulting pin representations with the ids of the boards they occur in, and the board embedding is computed by pooling the features of its (sampled) neighbors.

(1) 一个MapReduce作业用于将所有Pin投影到一个低维潜在空间，聚合操作将在此空间中执行（算法1，第1行）。
(2) 然后使用另一个MapReduce作业将得到的Pin表示与它们所在画板的ID进行连接，并通过池化其（采样的）邻居的特征来计算画板嵌入。

Note that our approach avoids redundant computations and that the latent vector for each node is computed only once. After the embeddings of the boards are obtained, we use two more MapReduce jobs to compute the second-layer embeddings of pins, in a similar fashion as above, and this process can be iterated as necessary (up to K convolutional layers).

注意，我们的方法避免了冗余计算，每个节点的潜在向量只被计算一次。在获得画板的嵌入之后，我们使用另外两个MapReduce作业以与上述类似的方式计算Pin的第二层嵌入，并且这个过程可以根据需要迭代（最多K个卷积层）。

### 3.5 高效最近邻查找

The embeddings generated by PinSage can be used for a wide range of downstream recommendation tasks, and in many settings we can directly use these embeddings to make recommendations by performing nearest-neighbor lookups in the learned embedding space. That is, given a query item q, we can recommend items whose embeddings are the K-nearest neighbors of the query item's embedding. Approximate KNN can be obtained efficiently via locality sensitive hashing [2]. After the hash function is computed, retrieval of items can be implemented with a two-level retrieval process based on the Weak AND operator [5]. Given that the PinSage model is trained offline and all node embeddings are computed via MapReduce and saved in a database, the efficient nearest-neighbor lookup operation enables the system to serve recommendations in an online fashion.

PinSage生成的嵌入可以用于广泛的下游推荐任务，在许多场景下，我们可以直接使用这些嵌入，通过学习到的嵌入空间中的最近邻查找来进行推荐。也就是说，给定一个查询itemq，我们可以推荐其嵌入是查询item嵌入的K个最近邻的item。近似KNN可以通过局部敏感哈希[2]高效地获得。在计算哈希函数之后，item的检索可以通过基于弱AND操作符[5]的两级检索过程来实现。由于PinSage模型是离线训练的，所有节点嵌入都通过MapReduce计算并保存在数据库中，高效的最近邻查找操作使系统能够以在线方式提供推荐服务。

---

## 4 实验

To demonstrate the efficiency of PinSage and the quality of the embeddings it generates, we conduct a comprehensive suite of experiments on the entire Pinterest object graph, including offline experiments, production A/B tests as well as user studies.

为了展示PinSage的效率及其生成的嵌入的质量，我们在整个Pinterest对象图上进行了一系列全面的实验，包括离线实验、生产A/B测试以及用户研究。

### 4.1 实验设置

We evaluate the embeddings generated by PinSage in two tasks: recommending related pins and recommending pins in a user's home/news feed. To recommend related pins, we select the K nearest neighbors to the query pin in the embedding space. We evaluate performance on this related-pin recommendation task using both offline ranking measures as well as a controlled user study. For the homefeed recommendation task, we select the pins that are closest in the embedding space to one of the most recently pinned items by the user. We evaluate performance of a fully-deployed production system on this task using A/B tests to measure the overall impact on user engagement.

我们在两个任务中评估PinSage生成的嵌入：推荐相关Pin和推荐用户首页/动态流中的Pin。为了推荐相关Pin，我们在嵌入空间中选择与查询Pin最近的K个邻居。我们使用离线排序指标和受控用户研究来评估这个相关Pin推荐任务的性能。对于首页动态推荐任务，我们选择嵌入空间中最接近用户最近Pin过的item之一的Pin。我们使用A/B测试来评估完全部署的生产系统在此任务上的性能，以衡量对用户参与度的整体影响。

**训练细节和数据准备.** We define the set, L, of positive training examples (Equation (1)) using historical user engagement data. In particular, we use historical user engagement data to identify pairs of pins (q, i), where a user interacted with pin i immediately after she interacted with pin q. We use all other pins as negative items (and sample them as described in Section 3.3). Overall, we use 1.2 billion pairs of positive training examples (in addition to 500 negative examples per batch and 6 hard negative examples per pin). Thus in total we use 7.5 billion training examples.

我们使用历史用户行为数据定义正训练样本集L（公式(1)）。具体来说，我们使用历史用户行为数据来识别Pin对(q, i)，其中用户在交互Pin q之后立即交互了Pin i。我们将所有其他Pin用作负item（并按照第3.3节所述进行采样）。总体而言，我们使用了12亿对正训练样本（此外每批还有500个负样本和每个Pin 6个困难负样本）。因此，我们总共使用了75亿个训练样本。

Since PinSage can efficiently generate embeddings for unseen data, we only train on a subset of the Pinterest graph and then generate embeddings for the entire graph using the MapReduce pipeline described in Section 3.4. In particular, for training we use a randomly sampled subgraph of the entire graph, containing 20% of all boards (and all the pins touched by those boards) and 70% of the labeled examples. During hyperparameter tuning, a remaining 10% of the labeled examples are used. And, when testing, we run inference on the entire graph to compute embeddings for all 2 billion pins, and the remaining 20% of the labeled examples are used to test the recommendation performance of our PinSage in the offline evaluations. Note that training on a subset of the full graph drastically decreased training time, with a negligible impact on final performance. In total, the full datasets for training and evaluation are approximately 18TB in size with the full output embeddings being 4TB.

由于PinSage可以高效地为未见过的数据生成嵌入，我们仅在Pinterest图的一个子集上训练，然后使用第3.4节描述的MapReduce流水线为整个图生成嵌入。具体来说，对于训练，我们使用整个图的一个随机采样子图，包含所有画板的20%（以及这些画板涉及的所有Pin）和70%的标注样本。在超参数调优期间，使用剩余10%的标注样本。在测试时，我们在整个图上运行推理来计算所有20亿个Pin的嵌入，剩余20%的标注样本用于测试PinSage在离线评估中的推荐性能。注意，在全图的子集上训练大大减少了训练时间，而对最终性能的影响可以忽略不计。总的来说，用于训练和评估的完整数据集约为18TB，完整输出嵌入为4TB。

**用于学习的特征.** Each pin at Pinterest is associated with an image and a set of textual annotations (title, description). To generate feature representation xq for each pin q, we concatenate visual embeddings (4,096 dimensions), textual annotation embeddings (256 dimensions), and the log degree of the node/pin in the graph. The visual embeddings are the 6-th fully connected layer of a classification network using the VGG-16 architecture [28]. Textual annotation embeddings are trained using a Word2Vec-based model [23], where the context of an annotation consists of other annotations that are associated with each pin.

Pinterest上的每个Pin都与一张图像和一组文本注释（标题、描述）相关联。为了生成每个Pin q的特征表示xq，我们拼接了视觉嵌入（4096维）、文本注释嵌入（256维）以及图中节点/Pin的度数对数。视觉嵌入是使用VGG-16架构[28]的分类网络的第6个全连接层。文本注释嵌入使用基于Word2Vec的模型[23]训练，其中注释的上下文由与每个Pin相关联的其他注释组成。

**对比基线.** We evaluate the performance of PinSage against the following state-of-the-art content-based, graph-based and deep learning baselines that generate embeddings of pins:

我们评估了PinSage与以下最先进的基于内容、基于图和深度学习的Pin嵌入生成基线的性能对比：

(1) **Visual embeddings (Visual):** Uses nearest neighbors of deep visual embeddings for recommendations. The visual features are described above.

(1) **视觉嵌入（Visual）：** 使用深度视觉嵌入的最近邻进行推荐。视觉特征如上所述。

(2) **Annotation embeddings (Annotation):** Recommends based on nearest neighbors in terms of annotation embeddings. The annotation embeddings are described above.

(2) **注释嵌入（Annotation）：** 基于注释嵌入的最近邻进行推荐。注释嵌入如上所述。

(3) **Combined embeddings (Combined):** Recommends based on concatenating visual and annotation embeddings, and using a 2-layer multi-layer perceptron to compute embeddings that capture both visual and annotation features.

(3) **组合嵌入（Combined）：** 基于拼接视觉和注释嵌入进行推荐，并使用2层多层感知器计算同时捕捉视觉和注释特征的嵌入。

(4) **Graph-based method (Pixie):** This random-walk-based method [14] uses biased random walks to generate ranking scores by simulating random walks starting at query pin q. Items with top K scores are retrieved as recommendations. While this approach does not generate pin embeddings, it is currently the state-of-the-art at Pinterest for certain recommendation tasks [14] and thus an informative baseline.

(4) **基于图的方法（Pixie）：** 这种基于随机游走的方法[14]使用有偏随机游走，通过模拟从查询Pin q开始的随机游走来生成排序分数。得分最高的K个item被检索为推荐。虽然这种方法不生成Pin嵌入，但它目前是Pinterest上某些推荐任务的最先进方法[14]，因此是一个有参考价值的基线。

The visual and annotation embeddings are state-of-the-art deep learning content-based systems currently deployed at Pinterest to generate representations of pins. Note that we do not compare against other deep learning baselines from the literature simply due to the scale of our problem. We also do not consider non-deep learning approaches for generating item/content embeddings, since other works have already proven state-of-the-art performance of deep learning approaches for generating such embeddings [9, 12, 24].

视觉和注释嵌入是目前部署在Pinterest上用于生成Pin表示的最先进的深度学习基于内容的系统。注意，由于我们问题的规模，我们没有与文献中的其他深度学习基线进行比较。我们也没有考虑用于生成item/内容嵌入的非深度学习方法，因为其他工作已经证明了深度学习方法在生成此类嵌入方面的最先进性能[9, 12, 24]。

We also conduct ablation studies and consider several variants of PinSage when evaluating performance:

我们在评估性能时还进行了消融研究，并考虑了PinSage的几种变体：

- **max-pooling** uses the element-wise max as a symmetric aggregation function (i.e., \gamma = max) without hard negative samples;
- **mean-pooling** uses the element-wise mean as a symmetric aggregation function (i.e., \gamma = mean);
- **mean-pooling-xent** is the same as mean-pooling but uses the cross-entropy loss introduced in [18].
- **mean-pooling-hard** is the same as mean-pooling, except that it incorporates hard negative samples as detailed in Section 3.3.
- **PinSage** uses all optimizations presented in this paper, including the use of importance pooling in the convolution step.

- **max-pooling（最大池化）** 使用逐元素最大作为对称聚合函数（即\gamma = max），不使用困难负样本；
- **mean-pooling（均值池化）** 使用逐元素均值作为对称聚合函数（即\gamma = mean）；
- **mean-pooling-xent** 与mean-pooling相同，但使用[18]中引入的交叉熵损失。
- **mean-pooling-hard** 与mean-pooling相同，但加入了第3.3节详述的困难负样本。
- **PinSage** 使用本文提出的所有优化，包括在卷积步骤中使用重要性池化。

The max-pooling and cross-entropy settings are extensions of the best-performing GCN model from Hamilton et al. [18]—other variants (e.g., based on Kipf et al. [21]) performed significantly worse in development tests and are omitted for brevity. For all the above variants, we used K = 2, hidden dimension size m = 2048, and set the embedding dimension d to be 1024.

最大池化和交叉熵设置是对Hamilton等人[18]中性能最佳的GCN模型的扩展——其他变体（例如基于Kipf等人[21]的变体）在开发测试中表现明显较差，为了简洁起见省略。对于上述所有变体，我们使用K = 2，隐藏维度大小m = 2048，嵌入维度d设置为1024。

**计算资源.** Training of PinSage is implemented in TensorFlow [1] and run on a single machine with 32 cores and 16 Tesla K80 GPUs. To ensure fast fetching of item's visual and annotation features, we store them in main memory, together with the graph, using Linux HugePages to increase the size of virtual memory pages from 4KB to 2MB. The total amount of memory used in training is 500GB. Our MapReduce inference pipeline is run on a Hadoop2 cluster with 378 d2.8xlarge Amazon AWS nodes.

PinSage的训练使用TensorFlow[1]实现，并在具有32个CPU核心和16块Tesla K80 GPU的单台机器上运行。为了确保快速获取item的视觉和注释特征，我们将它们与图一起存储在主内存中，使用Linux HugePages将虚拟内存页大小从4KB增加到2MB。训练使用的总内存为500GB。我们的MapReduce推理流水线在具有378个d2.8xlarge Amazon AWS节点的Hadoop2集群上运行。

### 4.2 离线评估

To evaluate performance on the related pin recommendation task, we define the notion of hit-rate. For each positive pair of pins (q, i) in the test set, we use q as a query pin and then compute its top K nearest neighbors NNq from a sample of 5 million test pins. We then define the hit-rate as the fraction of queries q where i was ranked among the top K of the test sample (i.e., where i \in NNq). This metric directly measures the probability that recommendations made by the algorithm contain the items related to the query pin q. In our experiments K is set to be 500.

为了评估相关Pin推荐任务的性能，我们定义了命中率的概念。对于测试集中的每个正Pin对(q, i)，我们使用q作为查询Pin，然后从500万个测试Pin的样本中计算其前K个最近邻NNq。然后我们将命中率定义为i被排在测试样本前K名内（即i \in NNq）的查询q的占比。这个指标直接衡量算法做出的推荐包含与查询Pin q相关的item的概率。在我们的实验中，K设置为500。

We also evaluate the methods using Mean Reciprocal Rank (MRR), which takes into account of the rank of the item j among recommended items for query item q:

我们还使用平均倒数排名（MRR）来评估这些方法，它考虑了itemj在查询itemq的推荐item中的排名：

MRR = 1/n $\times$ \Sigma_{(q,i)\inL} 1 / ⌈$R_{i,q}$ / 100⌉   (2)

Due to the large pool of candidates (more than 2 billion), we use a scaled version of the MRR in Equation (2), where $R_{i,q}$ is the rank of item i among recommended items for query q, and n is the total number of labeled item pairs. The scaling factor 100 ensures that, for example, the difference between rank at 1,000 and rank at 2,000 is still noticeable, instead of being very close to 0.

由于候选池巨大（超过20亿），我们使用公式(2)中MRR的缩放版本，其中$R_{i,q}$是itemi在查询q的推荐item中的排名，n是标注item对的总数。缩放因子100确保，例如，排名在1000和排名在2000之间的差异仍然是可察觉的，而不是非常接近0。

**表1：PinSage与基于内容的深度学习基线的命中率和MRR对比**

| 方法 | 命中率 | MRR |
|------|--------|-----|
| Visual | 0.23 | 17% |
| Annotation | 0.19 | 14% |
| Combined | 0.37 | 27% |
| max-pooling | 0.37 | 39% |
| mean-pooling | 0.51 | 41% |
| mean-pooling-xent | 0.35 | 29% |
| mean-pooling-hard | 0.56 | 46% |
| **PinSage** | **0.59** | **67%** |

Table 1 compares the performance of the various approaches using the hit rate as well as the MRR. PinSage with our new importance-pooling aggregation and hard negative examples achieves the best performance at 67% hit-rate and 0.59 MRR, outperforming the top baseline by 40% absolute (150% relative) in terms of the hit rate and also 22% absolute (60% relative) in terms of MRR. We also observe that combining visual and textual information works much better than using either one alone (60% improvement of the combined approach over visual/annotation only).

表1比较了各种方法的命中率和MRR性能。使用我们新的重要性池化聚合和困难负样本的PinSage取得了最佳性能，命中率为67%，MRR为0.59，在命中率上比最佳基线高出40%（绝对值）（150%相对提升），在MRR上高出22%（绝对值）（60%相对提升）。我们还观察到，结合视觉和文本信息比单独使用其中之一效果要好得多（组合方法比仅使用视觉或注释的方法提升60%）。

**嵌入相似度分布.** Another indication of the effectiveness of the learned embeddings is that the distances between random pairs of item embeddings are widely distributed. If all items are at about the same distance (i.e., the distances are tightly clustered) then the embedding space does not have enough "resolution" to distinguish between items of different relevance. Figure 4 plots the distribution of cosine similarities between pairs of items using annotation, visual, and PinSage embeddings. This distribution of cosine similarity between random pairs of items demonstrates the effectiveness of PinSage, which has the most spread out distribution. In particular, the kurtosis of the cosine similarities of PinSage embeddings is 0.43, compared to 2.49 for annotation embeddings and 1.20 for visual embeddings.

学习到的嵌入有效性的另一个指标是随机item嵌入对之间的距离分布广泛。如果所有item之间的距离大致相同（即距离紧密聚集），那么嵌入空间就没有足够的"分辨率"来区分不同相关性的item。图4绘制了使用注释嵌入、视觉嵌入和PinSage嵌入的item对之间的余弦相似度分布。这种随机item对之间的余弦相似度分布展示了PinSage的有效性，它具有最分散的分布。具体来说，PinSage嵌入的余弦相似度的峰度为0.43，而注释嵌入为2.49，视觉嵌入为1.20。

Another important advantage of having such a wide-spread in the embeddings is that it reduces the collision probability of the subsequent LSH algorithm, thus increasing the efficiency of serving the nearest neighbor pins during recommendation.

嵌入具有如此广泛分布的另一个重要优点是，它降低了后续LSH算法的碰撞概率，从而提高了在推荐过程中提供最近邻Pin的服务的效率。

### 4.3 用户研究

We also investigate the effectiveness of PinSage by performing head-to-head comparison between different learned representations. In the user study, a user is presented with an image of the query pin, together with two pins retrieved by two different recommendation algorithms. The user is then asked to choose which of the two candidate pins is more related to the query pin. Users are instructed to find various correlations between the recommended items and the query item, in aspects such as visual appearance, object category and personal identity. If both recommended items seem equally related, users have the option to choose "equal". If no consensus is reached among 2/3 of users who rate the same question, we deem the result as inconclusive.

我们还通过在不同学习到的表示之间进行头对头比较来研究PinSage的有效性。在用户研究中，向用户展示一张查询Pin的图像，以及由两种不同推荐算法检索到的两个Pin。然后要求用户选择两个候选Pin中哪一个与查询Pin更相关。用户被指示在推荐item和查询item之间寻找各种相关性，例如视觉外观、物体类别和个人身份等方面。如果两个推荐item看起来同样相关，用户可以选择"相等"。如果对同一问题评分的用户中有2/3未能达成共识，我们将结果视为不确定。

**表2：图像与查询推荐图像相关性的头对头比较**

| 方法 | 胜 | 负 | 平 | 胜率 |
|------|----|----|-----|------|
| PinSage vs. Visual | 28.4% | 21.9% | 49.7% | 56.5% |
| PinSage vs. Annot. | 36.9% | 14.0% | 49.1% | 72.5% |
| PinSage vs. Combined | 22.6% | 15.1% | 57.5% | 60.0% |
| PinSage vs. Pixie | 32.5% | 19.6% | 46.4% | 62.4% |

Table 2 shows the results of the head-to-head comparison between PinSage and the 4 baselines. Among items for which the user has an opinion of which is more related, around 60% of the preferred items are recommended by PinSage. Figure 5 gives examples of recommendations and illustrates strengths and weaknesses of the different methods. The image to the left represents the query item. Each row to the right corresponds to the top recommendations made by the visual embedding baseline, annotation embedding baseline, Pixie, and PinSage. Although visual embeddings generally predict categories and visual similarity well, they occasionally make large mistakes in terms of image semantics. In this example, visual information confused plants with food, and tree logging with war photos, due to similar image style and appearance. The graph-based Pixie method, which uses the graph of pin-to-board relations, correctly understands that the category of query is "plants" and it recommends items in that general category. However, it does not find the most relevant items. Combining both visual/textual and graph information, PinSage is able to find relevant items that are both visually and topically similar to the query item.

表2显示了PinSage与4个基线的头对头比较结果。在用户有明确偏好的item中，约60%的偏好item由PinSage推荐。图5给出了推荐示例，并说明了不同方法的优势和劣势。左侧的图像代表查询item。右侧的每一行对应视觉嵌入基线、注释嵌入基线、Pixie和PinSage做出的前几名推荐。尽管视觉嵌入通常能很好地预测类别和视觉相似性，但它们偶尔会在图像语义方面犯大错误。在这个例子中，由于图像风格和外观相似，视觉信息将植物与食物混淆，将伐木与战争照片混淆。基于图的Pixie方法使用Pin到画板关系的图，它正确理解查询类别是"植物"，并推荐该大类别中的item。然而，它没有找到最相关的item。结合视觉/文本和图信息，PinSage能够找到在视觉上和主题上都与查询item相关的item。

In addition, we visualize the embedding space by randomly choosing 1000 items and compute the 2D t-SNE coordinates from the PinSage embedding, as shown in Figure 6. We observe that the proximity of the item embeddings corresponds well with the similarity of content, and that items of the same category are embedded into the same part of the space. Note that items that are visually different but have the same theme are also close to each other in the embedding space, as seen by the items depicting different fashion-related items on the bottom side of the plot.

此外，我们通过随机选择1000个item并计算PinSage嵌入的2D t-SNE坐标来可视化嵌入空间，如图6所示。我们观察到item嵌入的邻近性与内容的相似性很好地对应，同一类别的item被嵌入到空间的同一区域。值得注意的是，在视觉上不同但具有相同主题的item在嵌入空间中也彼此接近，从图底部描绘不同时尚相关item的示例可以看出。

### 4.4 生产A/B测试

Lastly, we also report on the production A/B test experiments, which compared the performance of PinSage to other deep learning content-based recommender systems at Pinterest on the task of homefeed recommendations. We evaluate the performance by observing the lift in user engagement. The metric of interest is repin rate, which measures the percentage of homefeed recommendations that have been saved by the users. A user saving a pin to a board is a high-value action that signifies deep engagement of the user. It means that a given pin presented to a user at a given time was relevant enough for the user to save that pin to one of their boards so that they can retrieve it later.

最后，我们还报告了生产A/B测试实验，该实验比较了PinSage与Pinterest上其他基于深度学习的基于内容的推荐系统在首页动态推荐任务上的性能。我们通过观察用户参与度的提升来评估性能。关注的指标是再Pin率，它衡量了被用户保存的首页动态推荐的百分比。用户将Pin保存到画板是一种高价值行为，标志着用户的深度参与。这意味着在特定时间呈现给用户的特定Pin足够相关，以至于用户将其保存到自己的某个画板以便日后检索。

We find that PinSage consistently recommends pins that are more likely to be re-pinned by the user than the alternative methods. Depending on the particular setting, we observe 10-30% improvements in repin rate over the Annotation and Visual embedding based recommendations.

我们发现PinSage始终如一地推荐比替代方法更有可能被用户再Pin的Pin。根据具体设置的不同，我们观察到与基于注释和视觉嵌入的推荐相比，再Pin率提升了10-30%。

### 4.5 训练与推理运行时间分析

One advantage of GCNs is that they can be made inductive [19]: at the inference (i.e., embedding generation) step, we are able to compute embeddings for items that were not in the training set. This allows us to train on a subgraph to obtain model parameters, and then make embeddings for nodes that have not been observed during training. Also note that it is easy to compute embeddings of new nodes that get added into the graph over time. This means that recommendations can be made on the full (and constantly growing) graph. Experiments on development data demonstrated that training on a subgraph containing 300 million items could achieve the best performance in terms of hit-rate (i.e., further increases in the training set size did not seem to help), reducing the runtime by a factor of 6 compared to training on the full graph.

GCN的一个优点是可以使其具有归纳性[19]：在推理（即嵌入生成）步骤中，我们能够为训练集中不存在的item计算嵌入。这使我们能够在一个子图上训练以获得模型参数，然后为训练期间未见过的节点生成嵌入。还要注意，为随着时间推移添加到图中的新节点计算嵌入是很容易的。这意味着可以在完整（且不断增长的）图上进行推荐。在开发数据上的实验表明，在包含3亿个item的子图上训练可以在命中率方面达到最佳性能（即进一步增加训练集大小似乎没有帮助），与在全图上训练相比，运行时间减少了6倍。

**表3：不同批量大小的运行时间对比**

| 批量大小 | 每次迭代(ms) | 迭代次数 | 总时间(h) |
|----------|-------------|---------|-----------|
| 512 | 590 | 390k | 63.9 |
| 1024 | 870 | 220k | 53.2 |
| 2048 | 1350 | 130k | 48.8 |
| 4096 | 2240 | 100k | 68.4 |

Table 3 shows the effect of batch size of the minibatch SGD on the runtime of PinSage training procedure, using the mean-pooling-hard variant. For varying batch sizes, the table shows: (1) the computation time, in milliseconds, for each minibatch, when varying batch size; (2) the number of iterations needed for the model to converge; and (3) the total estimated time for the training procedure. Experiments show that a batch size of 2048 makes training most efficient.

表3显示了使用mean-pooling-hard变体时，小批量SGD的批量大小对PinSage训练过程运行时间的影响。对于不同的批量大小，表格显示：(1) 不同批量大小下每个小批量的计算时间（毫秒）；(2) 模型收敛所需的迭代次数；以及 (3) 训练过程的估计总时间。实验表明，批量大小为2048时训练效率最高。

**表4：重要性池化的性能权衡**

| 邻居数 | 命中率 | MRR | 训练时间(h) |
|--------|--------|-----|------------|
| 10 | 60% | 0.51 | 20 |
| 20 | 63% | 0.54 | 33 |
| 50 | 67% | 0.59 | 78 |

When training the PinSage variant with importance pooling, another trade-off comes from choosing the size of neighborhood T. Table 4 shows the runtime and performance of PinSage when T = 10, 20 and 50. We observe a diminishing return as T increases, and find that a two-layer GCN with neighborhood size 50 can best capture the neighborhood information of nodes, while still being computationally efficient.

在训练使用重要性池化的PinSage变体时，另一个权衡来自选择邻域大小T。表4显示了T = 10、20和50时PinSage的运行时间和性能。我们观察到随着T的增加存在收益递减，并发现邻域大小为50的双层GCN能够最好地捕捉节点的邻域信息，同时仍然具有计算效率。

After training completes, due to the highly efficient MapReduce inference pipeline, the whole inference procedure to generate embeddings for 3 billion items can finish in less than 24 hours.

在训练完成后，由于高效的MapReduce推理流水线，为30亿item生成嵌入的整个推理过程可以在不到24小时内完成。

---

## 5 结论

We proposed PinSage, a random-walk graph convolutional network (GCN). PinSage is a highly-scalable GCN algorithm capable of learning embeddings for nodes in web-scale graphs containing billions of objects. In addition to new techniques that ensure scalability, we introduced the use of importance pooling and curriculum training that drastically improved embedding performance. We deployed PinSage at Pinterest and comprehensively evaluated the quality of the learned embeddings on a number of recommendation tasks, with offline metrics, user studies and A/B tests all demonstrating a substantial improvement in recommendation performance. Our work demonstrates the impact that graph convolutional methods can have in a production recommender system, and we believe that PinSage can be further extended in the future to tackle other graph representation learning problems at large scale, including knowledge graph reasoning and graph clustering.

我们提出了PinSage，一种随机游走图卷积网络（GCN）。PinSage是一种高度可扩展的GCN算法，能够为包含数十亿对象的Web规模图中的节点学习嵌入。除了确保可扩展性的新技术之外，我们引入了重要性池化和课程训练，极大地提高了嵌入性能。我们在Pinterest上部署了PinSage，并在多个推荐任务上全面评估了学习到的嵌入的质量，离线指标、用户研究和A/B测试都证明了推荐性能的显著提升。我们的工作展示了图卷积方法在生产推荐系统中的影响力，并且我们相信PinSage将来可以进一步扩展，以解决其他大规模图表示学习问题，包括知识图谱推理和图聚类。

**致谢：**

The authors acknowledge Raymond Hsu, Andrei Curelea and Ali Altaf for performing various A/B tests in production system, Jerry Zitao Liu for providing data used by Pixie [14], and Vitaliy Kulikov for help in nearest neighbor query of the item embeddings.

作者感谢Raymond Hsu、Andrei Curelea和Ali Altaf在生产系统中执行各种A/B测试，感谢Jerry Zitao Liu提供Pixie [14]所使用的数据，以及感谢Vitaliy Kulikov在item嵌入的最近邻查询方面的帮助。

---

## 参考文献

[1] M. Abadi, A. Agarwal, P. Barham, E. Brevdo, Z. Chen, C. Citro, G. S. Corrado, A. Davis, J. Dean, M. Devin, et al. 2016. Tensorflow: Large-scale machine learning on heterogeneous distributed systems. arXiv preprint arXiv:1603.04467 (2016).

[2] A. Andoni and P. Indyk. 2006. Near-optimal hashing algorithms for approximate nearest neighbor in high dimensions. In FOCS.

[3] T. Bansal, D. Belanger, and A. McCallum. 2016. Ask the GRU: Multi-task learning for deep text recommendations. In RecSys. ACM.

[4] Y. Bengio, J. Louradour, R. Collobert, and J. Weston. 2009. Curriculum learning. In ICML.

[5] A. Z. Broder, D. Carmel, M. Herscovici, A. Soffer, and J. Zien. 2003. Efficient query evaluation using a two-level retrieval process. In CIKM.

[6] M. M. Bronstein, J. Bruna, Y. LeCun, A. Szlam, and P. Vandergheynst. 2017. Geometric deep learning: Going beyond euclidean data. IEEE Signal Processing Magazine 34, 4 (2017).

[7] J. Bruna, W. Zaremba, A. Szlam, and Y. LeCun. 2014. Spectral networks and locally connected networks on graphs. In ICLR.

[8] J. Chen, T. Ma, and C. Xiao. 2018. FastGCN: Fast Learning with Graph Convolutional Networks via Importance Sampling. ICLR (2018).

[9] P. Covington, J. Adams, and E. Sargin. 2016. Deep neural networks for youtube recommendations. In RecSys. ACM.

[10] H. Dai, B. Dai, and L. Song. 2016. Discriminative Embeddings of Latent Variable Models for Structured Data. In ICML.

[11] M. Defferrard, X. Bresson, and P. Vandergheynst. 2016. Convolutional neural networks on graphs with fast localized spectral filtering. In NIPS.

[12] A. Van den Oord, S. Dieleman, and B. Schrauwen. 2013. Deep content-based music recommendation. In NIPS.

[13] D. Duvenaud, D. Maclaurin, J. Iparraguirre, R. Bombarell, T. Hirzel, A. Aspuru-Guzik, and R. P. Adams. 2015. Convolutional networks on graphs for learning molecular fingerprints. In NIPS.

[14] C. Eksombatchai, P. Jindal, J. Z. Liu, Y. Liu, R. Sharma, C. Sugnet, M. Ulrich, and J. Leskovec. 2018. Pixie: A System for Recommending 3+ Billion Items to 200+ Million Users in Real-Time. WWW (2018).

[15] M. Gori, G. Monfardini, and F. Scarselli. 2005. A new model for learning in graph domains. In IEEE International Joint Conference on Neural Networks.

[16] P. Goyal, P. Dollár, R. Girshick, P. Noordhuis, L. Wesolowski, A. Kyrola, A. Tulloch, Y. Jia, and K. He. 2017. Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour. arXiv preprint arXiv:1706.02677 (2017).

[17] A. Grover and J. Leskovec. 2016. node2vec: Scalable feature learning for networks. In KDD.

[18] W. L. Hamilton, R. Ying, and J. Leskovec. 2017. Inductive Representation Learning on Large Graphs. In NIPS.

[19] W. L. Hamilton, R. Ying, and J. Leskovec. 2017. Representation Learning on Graphs: Methods and Applications. IEEE Data Engineering Bulletin (2017).

[20] S. Kearnes, K. McCloskey, M. Berndl, V. Pande, and P. Riley. 2016. Molecular graph convolutions: moving beyond fingerprints. CAMD 30, 8.

[21] T. N. Kipf and M. Welling. 2017. Semi-supervised classification with graph convolutional networks. In ICLR.

[22] Y. Li, D. Tarlow, M. Brockschmidt, and R. Zemel. 2015. Gated graph sequence neural networks. In ICLR.

[23] T. Mikolov, I Sutskever, K. Chen, G. S. Corrado, and J. Dean. 2013. Distributed representations of words and phrases and their compositionality. In NIPS.

[24] F. Monti, M. M. Bronstein, and X. Bresson. 2017. Geometric matrix completion with recurrent multi-graph neural networks. In NIPS.

[25] OpenMP Architecture Review Board. 2015. OpenMP Application Program Interface Version 4.5. (2015).

[26] B. Perozzi, R. Al-Rfou, and S. Skiena. 2014. DeepWalk: Online learning of social representations. In KDD.

[27] F. Scarselli, M. Gori, A.C. Tsoi, M. Hagenbuchner, and G. Monfardini. 2009. The graph neural network model. IEEE Transactions on Neural Networks 20, 1 (2009), 61–80.

[28] K. Simonyan and A. Zisserman. 2014. Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556 (2014).

[29] R. van den Berg, T. N. Kipf, and M. Welling. 2017. Graph Convolutional Matrix Completion. arXiv preprint arXiv:1706.02263 (2017).

[30] J. You, R. Ying, X. Ren, W. L. Hamilton, and J. Leskovec. 2018. GraphRNN: Generating Realistic Graphs using Deep Auto-regressive Models. ICML (2018).

[31] M. Zitnik, M. Agrawal, and J. Leskovec. 2018. Modeling polypharmacy side effects with graph convolutional networks. Bioinformatics (2018).
