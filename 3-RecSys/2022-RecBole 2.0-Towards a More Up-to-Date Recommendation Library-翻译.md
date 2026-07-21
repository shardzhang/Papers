# RecBole 2.0：迈向更现代化的推荐库

> Wayne Xin Zhao, Yupeng Hou, Xingyu Pan, Chen Yang et al. | Renmin University of China

本文介绍了 RecBole 2.0：迈向更现代化的推荐库。核心内容：


关键发现：

---


Wayne Xin Zhao^1,2, Yupeng Hou^1,2,#, Xingyu Pan^1,3,#, Chen Yang^1,2
Zeyu Zhang^1, Zihan Lin^1,3, Jingsen Zhang^1,3, Shuqing Bian^1,3, Jiakai Tang^1
Wenqi Sun^1,2, Yushuo Chen^1,2, Lanling Xu^1, Gaowei Zhang^1, Zhen Tian^1
Changxin Tian^1,3, Shanlei Mu^1,3, Xinyan Fan^1,2, Xu Chen^1,2,∗ 和 Ji-Rong Wen^1,2,3

^1 北京大数据管理与分析方法重点实验室
^2 {高瓴人工智能学院, ^3 信息学院} 中国人民大学


---

## 摘要

为了支持推荐系统最新进展的研究，本文提出了一个扩展的推荐库，包含八个针对前沿主题和架构的包。首先，从数据的角度，我们考虑了与数据问题相关的三个重要主题（即稀疏性、偏差和分布偏移），并据此开发了五个包：元学习、数据增强、去偏、公平性和跨域推荐。此外，从模型的角度，我们分别开发了两个用于基于Transformer和基于图神经网络（GNN）模型的基准测试包。所有包（包含65个新模型）均基于流行的推荐框架RecBole开发，确保了实现和接口的统一。对于每个包，我们提供了从数据加载、实验设置、评估到算法实现的完整实现。该库为促进推荐系统的前沿研究提供了宝贵的资源。item发布链接：https://github.com/RUCAIBox/RecBole2.0。

## 关键词

推荐库；可复现性；评估

## 1 引言

如今，推荐系统已深刻改变了人们的日常生活，带来了巨大的商业价值并为信息搜寻提供了极大的便利。在学术界，基于不同的架构或方法，研究者提出了各种推荐算法[85]。尽管推荐系统取得了巨大进步，但人们对推荐算法的可复现性越来越关注[1, 87]。面对这一问题，许多开源推荐库已被发布，以促进所提出的推荐算法的可复现实现[1, 2, 4, 10, 28, 46, 58, 61, 87, 92]。这些库在很大程度上增强了推荐算法在研究目的上的可复现性。

然而，现有的推荐库主要关注经典模型，缺乏对推荐领域最新进展的考虑，包括新模型（例如图神经网络[73]）和新主题（例如去偏[6]和公平性[51]）。鉴于推荐系统的快速发展，我们认为需要一个更现代化的推荐库来支持新进展的研究。在早期阶段标准化这些正在进行的研究尤为重要，以防止非标准化的实现或不可靠的评估。

基于这一动机，本文对先前发布的推荐库RecBole[87]^1（在GitHub上获得广泛关注，拥有1.9K星标）进行了重要扩展，纳入了一系列面向最新进展的基准测试包。具体而言，我们的扩展从两个主要方面进行，即数据和模型。首先，交互数据本身的问题日益受到关注[7]，我们聚焦于与数据问题相关的三个重要研究主题，即数据稀疏性、数据偏差和数据分布偏移。针对这三个数据问题，我们开发了五个基准测试包，分别对应元学习、数据增强、去偏、公平性和跨域推荐。此外，从模型的角度，我们考虑为基于新兴模型架构的推荐算法提供更多支持，并开发了两个面向基于Transformer和基于图神经网络（GNN）模型的基准测试包。值得注意的是，这些包（除Transformer包外）提供了从数据加载、实验设置、评估到算法实现的完整实现，研究人员可以根据需求以很少的工作量快速产生某个包的基准测试结果。

^1 https://recbole.io/

这个扩展库的优点有三。第一，实现和接口统一：所有包完全基于推荐框架RecBole开发，尽可能重用主代码库中的现有函数或模块。第二，这是一次重要的扩展，包含65个新实现的算法或模型，对推荐系统的最新进展有很好的覆盖。通过这次扩展，RecBole已成为GitHub上公开研究项目中在算法（130+模型）和任务（11个任务或主题）方面最全面的推荐库。第三，使用灵活可靠：我们为每个包实现了完整的评估流程，并仔细进行了代码审查和测试。

在接下来的章节中，我们首先概述实现的包，然后详细说明每个包以阐述我们的设计原则并展示使用示例，最后讨论我们的库与RecBole相比的扩展。

## 2 概述

我们在图1中对扩展库进行了总体介绍。如图所示，我们的扩展包括八个包，分为数据和模型两大主要部分。对于数据包，我们聚焦三个关键问题，即数据稀疏性、数据偏差和数据分布偏移。为解决数据稀疏性问题，我们开发了两个包，即数据增强（生成额外的数据样本来优化推荐模型）和基于元学习的推荐（用元学习方法处理冷启动问题）。为缓解数据偏差，我们开发了两个包，即去偏推荐（减少数据偏差）和公平性推荐（增强推荐的公平性）。为克服数据分布偏移问题，我们实现了一个跨域推荐包。需要注意的是，跨域推荐并非新的研究主题，但我们的重点是纳入最新的跨域推荐模型。对于模型包，我们希望收集并实现最新的推荐模型。特别地，我们聚焦于图神经网络和基于Transformer的模型。此外，我们还实现了一个针对人岗匹配任务[12]的应用包，该任务已受到研究和工业界的广泛关注。

- 数据稀疏性 \rightarrow 数据增强、元推荐
- 数据偏差 \rightarrow 去偏推荐、公平性推荐
- 数据分布偏移 \rightarrow 跨域推荐
- 模型 \rightarrow 基于图的推荐、基于Transformer的推荐
- 其他 \rightarrow 人岗匹配

**图1：实现的包概览，分为数据和模型两部分。**

在研究社区中，我们的库中所有模型都提供了最优配置（通过在预定义参数范围内进行网格搜索获得），基于这些配置可以轻松产生每个包的基准测试结果。下面我们将详细介绍上述包。

## 3 包详情与使用

我们将包分为两组，即数据导向包和模型导向包，详细说明如下。

### 3.1 数据导向包

**数据增强（RecBole-DA）**。为缓解数据稀疏性问题，最近提出的一种技术是生成额外的样本来丰富原始的用户-item交互[64, 75]。根据[3, 44, 64]，我们实现了基于不同数据增强策略的三类模型，包括启发式方法（CL4SRec [75] 和 DuoRec [54]）、基于模型的方法（MMInfoRec [53] 和 CauseRec [84]）以及混合方法（CASR [64]、CCL [3] 和 CoSeRec [44]）。除了提供模型实现，我们还提供了易于使用的API来配置和组合不同的数据增强策略。

**元推荐（RecBole-MetaRec）**。元学习源于计算机视觉和机器学习领域，是处理少样本学习的一种原则性方法[14, 24, 29, 48]。我们实现了三种类型的元学习推荐模型：预测型（MeLU [37] 和 MAMO [11]）、参数化型（LWA [59]、NLBA [59] 和 TaNP [40]）以及嵌入型（MetaEmb [49] 和 MWUF [93]）。这些模型通过一系列通用模块（MetaDataset、MetaDataLoader、MetaRecommender、MetaTrainer、MetaCollector和MetaUtils）实现，便于扩展新模型。

**公平性推荐（RecBole-FairRec）**。第二个与数据偏差相关的包针对公平性推荐[83]，从用户角度考虑数据偏差（去偏推荐包主要关注item侧的偏差）。具体来说，我们在该包中实现了四个模型，包括FOCF [78]、PFCN [39]、FairGo [70] 和 NFCF [30]。除了模型，我们还实现了一系列公平性度量，这对公平性推荐尤为重要，包括基尼系数[18]、流行度比率[18]、差分公平性[30]、价值不公平性、绝对不公平性、低估不公平性、高估不公平性[78]以及非奇偶不公平性[32]。

**去偏推荐（RecBole-Debias）**。数据偏差普遍存在于推荐系统中观察到的用户-item交互数据中[7]。为纠正这些偏差，我们实现了六个去偏模型，考虑选择偏差（MF-IPS [56]）、流行度偏差（PDA [86]、MACR [65]、DICE [88] 和 CausE [5]）以及曝光偏差（Rel-MF [55]）。此外，我们还为三个主要的去偏数据集（Yahoo! R3 [56]、ML-100k [56] 和 KuaiRec [16]）实现了特定的数据加载器，以便方便地复现实验。

**跨域推荐（RecBole-CDR）**。数据分布偏移通常发生在跨域推荐中。为构建有效的跨域模型，我们考虑了三种代表性的方法类别：联合矩阵分解（CMF [57] 和 CLFM [17]）、表示共享或组合（DTCDR [90]、DeepAPF [77] 和 NATR [15]）以及迁移或映射（CoNet [27]、BiTGCF [43]、EMCDR [45]、SSCDR [33] 和 DCDCSR [91]）。跨域推荐实际上不是一个新主题，但很少有包能很好地覆盖这一研究方向上的各种代表性方法。

### 3.2 模型导向包

**基于GNN的推荐（RecBole-GNN）**。近年来，图神经网络（GNN）[20, 34, 60]已被证明能有效建模各种数据类型的图结构，例如推荐系统[73]。我们实现了针对不同任务定制的三类GNN模型，包括通用推荐（NGCF [62]、LightGCN [22]、SGL [69]、HMLET [35]、NCL [41] 和 SimGCL [82]）、序列推荐（SR-GNN [74]、GC-SAN [76]、NISER [19]、LESSR [9]、TAGNN [79]、GCE-GNN [63] 和 SGNN-HN [50]）以及社交推荐（DiffNet [72]、MHCN [81] 和 SEPT [80]）。遵循原始数据机制，我们设计了一种新的原子文件类型，后缀为`.net`，用于结构化数据（例如社交网络）的建模。

**基于Transformer的推荐（RecBole-TRM）**。我们实现的另一个主要模型架构是广泛使用的Transformer及其变体[21]。我们通过考虑两个主要任务来实现这个包，即序列推荐（TiSASRec [38]、SSE-PT [71]、LightSANs [13]、gMLP [42] 和 CORE [25]）和新闻推荐（NRMS [68]、NAML [66] 和 NPA [67]）。尽管这些任务中使用的基本Transformer架构相似，但它们的输入不同，这使得Transformer在序列建模中发挥不同的作用。具体来说，在序列推荐中，Transformer被用于捕获用户行为相关性；而在新闻推荐中，Transformer被用于提取文本语义。

**人岗推荐（RecBole-PJF）**。此外，我们还包含了一个专门为人岗匹配任务[12]定制的包。该任务是一个重要的应用，吸引了研究和工业界的广泛关注。该包包括三类模型：协同过滤模型（NeuMF [23]、LightGCN [22] 和 LFRR [47]）、基于内容的模型（PJFNN [89]、APJFNN [52]、BPJFNN [52] 以及双塔BERT模型）和混合模型（IPJF [36]、PJFFF [31] 和 SHPJF [26]）。该包设计了特殊的数据机制以支持额外数据类型的整合。

**包总结**。请注意，每个包都可以作为一个独立item运行（包括产生比较结果的完整流程），但基于RecBole统一实现。为清晰理解我们的库，我们在表1中总结了上述实现的包和模型。

**表1：包含的包及每个包中实现的模型**

| 模块 | 包 | 模型 |
|------|-----|------|
| 数据 | 数据增强 (RecBole-DA) | CL4SRec [75], DuoRec [54], MMInfoRe [53], CauseRec [84], CASR [64], CCL [3], CoSeRec [44] |
| 数据 | 元推荐 (RecBole-MetaRec) | MeLU [37], MAMO [11], LWA [59], NLBA [59], TaNP [40], MetaEmb [49], MWUF [93] |
| 数据 | 去偏推荐 (RecBole-Debias) | MF-IPS [56], PDA [86], MACR [65], DICE [88], CausE [5], Rel-MF [55] |
| 数据 | 公平性推荐 (RecBole-FairRec) | FOCF [78], PFCN [39], FairGo [70], NFCF [30] |
| 数据 | 跨域推荐 (RecBole-CDR) | CMF [57], CLFM [17], DTCDR [90], DeepAPF [77], NATR [15], CoNet [27], BiTGCF [43], EMCDR [45], SSCDR [33], DCDCSR [91] |
| 模型 | 基于图的推荐 (RecBole-GNN) | NGCF [62], LightGCN [22], SGL [69], HMLET [35], NCL [41], SimGCL [82], SR-GNN [74], GC-SAN [76], NISER [19], LESSR [9], TAGNN [79], GCE-GNN [63], SGNN-HN [50], DiffNet [72], MHCN [81], SEPT [80] |
| 模型 | 基于Transformer的推荐 (RecBole-TRM) | TiSASRec [38], SSE-PT [71], LightSANs [13], gMLP [42], CORE [25], NRMS [68], NAML [66], NPA [67] |
| 其他 | 人岗匹配 (RecBole-PJF) | PJFNN [89], APJFNN [52], BPJFNN [52], IPJF [36], PJFFF [31], SHPJF [26], LFRR [47] |

## 4 包的使用

在本节中，我们通过分别展示每个包的示例代码来介绍如何使用我们的库。

**数据增强**。在我们的库中，数据增强包通过一系列增强接口或命令提供。调用数据增强包需要遵循两个步骤：（1）指定配置文件，（2）基于现有或自定义模型运行增强命令。图2(a)展示了一个示例，演示如何使用名为"item_crop"的数据增强策略运行SASRec。

**元推荐**。要运行基于元学习的推荐器，可以使用快速启动封装器（见图2(b)），它将自动进行模型配置、训练器配置、数据集准备、模型训练和模型评估。要实现一个新的元学习模型，用户可以遵循三个步骤：（1）指定配置文件，（2）扩展MetaRecommender实现模型细节，（3）扩展MetaTrainer自定义训练过程。

**公平性推荐**。要运行已实现的公平性模型，应该（1）通过YAML文件指定关于环境、数据、训练器、评估和模型的参数，（2）指定模型和数据集，（3）使用快速启动脚本启动程序。图2(c)展示了一个训练和评估FOCF [78]的示例。为了实现一个新的公平感知推荐模型，有三个步骤：（1）扩展Trainer指定特定的训练过程，（2）扩展FairRecommender实现模型，（3）扩展AbstractMetric实现公平性度量。

**去偏推荐**。要运行已实现的去偏推荐模型，需要遵循两个步骤：（1）通过YAML文件指定模型、数据集、训练和评估过程的设置，（2）通过指定模型和数据集启动程序。图2(d)展示了基于Yahoo! R3运行PDA的示例。要实现一个模型，可以通过扩展DebiasedRecommender类来实现模型架构，并实现训练器和采样器。

**跨域推荐**。要运行跨域推荐模型，用户可以通过配置将格式化的数据集设置为源域或目标域，并通过简单命令运行模型。我们在图2(e)中展示了一个在我们的库中运行EMCDR [45]的示例，其中我们可以设置源域和目标域数据集，指定训练模式等。要实现新算法，可以首先扩展CrossdomainRecommender类，然后通过简单的配置指定训练模式。

**基于GNN的推荐**。要运行已实现的GNN模型，有两个主要准备步骤：（1）指定自定义配置值并将其存储在额外的YAML文件中；（2）指定模型和数据集并使用快速启动脚本启动。图2(f)展示了在Movielens-1M数据集上使用自定义配置训练和评估NCL [41]的示例。要实现新的基于GNN的推荐模型，可以重用或调整GNN层以快速复现。例如，我们可以在RecBole-GNN中重用LightGCNConv层来复现基于GNN的协同过滤模型，或将图卷积层替换为新的GNN组件（例如GCN2Conv [8]）以进行进一步探索。

**基于Transformer的推荐**。要运行已实现的基于Transformer的模型，可以遵循三个步骤：（1）通过YAML文件指定配置；（2）指定数据集和模型；（3）使用快速启动脚本运行模型。图2(g)展示了在Movielens-1M数据集上运行TiSASRec [38]的示例。要实现新的基于Transformer的推荐模型，用户可以重用RecBole实现的Transformer层，或在Transformer层中添加新的实现。

**人岗推荐**。该包主要遵循整体RecBole库的使用方式，并进行了若干调整或提供了新接口。开发新模型的步骤如下：创建一个新的模型文件（例如PJFNN.py），实现相应的函数并将超参数保存到配置文件中。特别地，我们引入了一个新的参数`multi-direction`来控制评估是否为双向的。如果设置为true，评估将在同一交互记录上从候选人和雇主两方面进行。

以上我们简要介绍了所实现包的使用方法。更详细的描述，请参考我们的项目链接：https://github.com/RUCAIBox/RecBole2.0。

## 5 讨论与结论

所包含的扩展是基于流行的推荐库RecBole开发的，该库最初包含70多个推荐模型，涵盖通用推荐、上下文感知推荐、序列推荐和基于知识的推荐等任务。自2020年以来，RecBole在GitHub上获得了广泛关注和使用，截至2022年6月1日拥有约1.9K星标和352个分支。在最初的RecBole中，我们专注于底层数据结构、通用评估流程和经典推荐模型的设计。

随着推荐系统的快速发展，我们收到了越来越多RecBole用户对支持最新进展（例如去偏、公平性和GNN）的请求。同时，我们的团队成员也在进行这些新兴主题或模型的研究。因此，我们开发并发布了这个扩展库，通过纳入对推荐系统最新进展的支持来增强RecBole。具体来说，在这次扩展中，我们发布了八个包含65个新实现模型的包，并提供了相应的数据准备、模型运行（使用经过良好调优的参数）和评估接口。我们相信这次扩展是对RecBole的重要贡献，是研究社区的宝贵资源。RecBole团队将持续改进该item，使其在研究方面保持最新、全面和灵活。

## 参考文献

[1] Vito Walter Anelli, Alejandro Bellogín, Antonio Ferrara, Daniele Malitesta, Felice Antonio Merra, Claudio Pomo, Francesco Maria Donini, and Tommaso Di Noia. 2021. Elliot: A Comprehensive and Rigorous Framework for Reproducible Recommender Systems Evaluation. In SIGIR. ACM, 2405–2414.

[2] Andreas Argyriou, Miguel González-Fierro, and Le Zhang. 2020. Microsoft Recommenders: Best Practices for Production-Ready Recommendation Systems. In Companion Proceedings of the Web Conference 2020. 50–51.

[3] Shuqing Bian, Wayne Xin Zhao, Kun Zhou, Jing Cai, Yancheng He, Cunxiang Yin, and Ji-Rong Wen. 2021. Contrastive Curriculum Learning for Sequential User Behavior Modeling via Data Augmentation. In CIKM 2021. 3737–3746.

[4] Xiangnan He, Xiang Wang, Bin Wu, Zhongchuan Sun and Jonathan Staniforth. 2017. NeuRec. https://github.com/wubinzzu/NeuRec (2017).

[5] Stephen Bonner and Flavian Vasile. 2018. Causal embeddings for recommendation. In Proceedings of the 12th ACM conference on recommender systems. 104–112.

[6] Jiawei Chen, Hande Dong, Xiang Wang, Fuli Feng, Meng Wang, and Xiangnan He. 2020. Bias and debias in recommender system: A survey and future directions. arXiv preprint arXiv:2010.03240 (2020).

[7] Jiawei Chen, Hande Dong, Xiang Wang, Fuli Feng, Meng Wang, and Xiangnan He. 2020. Bias and debias in recommender system: A survey and future directions. arXiv preprint arXiv:2010.03240 (2020).

[8] Ming Chen, Zhewei Wei, Zengfeng Huang, Bolin Ding, and Yaliang Li. 2020. Simple and Deep Graph Convolutional Networks. In ICML.

[9] Tianwen Chen and Raymond Chi-Wing Wong. 2020. Handling Information Loss of Graph Neural Networks for Session-based Recommendation. In KDD.

[10] Arthur da Costa, Eduardo Fressato, Fernando Neto, Marcelo Manzato, and Ricardo Campello. 2018. Case recommender: a flexible and extensible python framework for recommender systems. In Proceedings of the 12th ACM Conference on Recommender Systems. 494–495.

[11] Manqing Dong, Feng Yuan, Lina Yao, Xiwei Xu, and Liming Zhu. 2020. Mamo: Memory-augmented meta-optimization for cold-start recommendation. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 688–697.

[12] Jeffrey R Edwards. 1991. Person-job fit: A conceptual integration, literature review, and methodological critique. John Wiley & Sons.

[13] Xinyan Fan, Zheng Liu, Jianxun Lian, Wayne Xin Zhao, Xing Xie, and Ji-Rong Wen. 2021. Lighter and better: low-rank decomposed self-attention networks for next-item recommendation. In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval. 1733–1737.

[14] Chelsea Finn, Pieter Abbeel, and Sergey Levine. 2017. Model-agnostic meta-learning for fast adaptation of deep networks. In International conference on machine learning. PMLR, 1126–1135.

[15] Chen Gao, Xiangning Chen, Fuli Feng, Kai Zhao, Xiangnan He, Yong Li, and Depeng Jin. 2019. Cross-domain recommendation without sharing user-relevant data. In The Web Conf. 491–502.

[16] Chongming Gao, Shijun Li, Wenqiang Lei, Biao Li, Peng Jiang, Jiawei Chen, Xiangnan He, Jiaxin Mao, and Tat-Seng Chua. 2022. KuaiRec: A Fully-observed Dataset for Recommender Systems. arXiv preprint arXiv:2202.10842 (2022).

[17] Sheng Gao, Hao Luo, Da Chen, Shantao Li, Patrick Gallinari, and Jun Guo. 2013. Cross-domain recommendation via cluster-level latent factor model. In Joint European conference on machine learning and knowledge discovery in databases.

[18] Yingqiang Ge, Shuchang Liu, Ruoyuan Gao, Yikun Xian, Yunqi Li, Xiangyu Zhao, Changhua Pei, Fei Sun, Junfeng Ge, Wenwu Ou, and Yongfeng Zhang. 2021. Towards Long-Term Fairness in Recommendation. In Proceedings of the 14th ACM International Conference on Web Search and Data Mining. Association for Computing Machinery, 445–453.

[19] Priyanka Gupta, Diksha Garg, Pankaj Malhotra, Lovekesh Vig, and Gautam M Shroff. 2019. NISER: Normalized Item and Session Representations with Graph Neural Networks. arXiv preprint arXiv:1909.04276 (2019).

[20] William L. Hamilton, Zhitao Ying, and Jure Leskovec. 2017. Inductive Representation Learning on Large Graphs. In NIPS.

[21] Kai Han, Yunhe Wang, Hanting Chen, Xinghao Chen, Jianyuan Guo, Zhenhua Liu, Yehui Tang, An Xiao, Chunjing Xu, Yixing Xu, et al. 2020. A survey on visual transformer. arXiv e-prints (2020), arXiv–2012.

[22] Xiangnan He, Kuan Deng, Xiang Wang, Yan Li, Yongdong Zhang, and Meng Wang. 2020. Lightgcn: Simplifying and powering graph convolution network for recommendation. In SIGIR.

[23] Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu, and Tat-Seng Chua. 2017. Neural collaborative filtering. In WWW.

[24] Timothy Hospedales, Antreas Antoniou, Paul Micaelli, and Amos Storkey. 2020. Meta-learning in neural networks: A survey. arXiv preprint arXiv:2004.05439 (2020).

[25] Yupeng Hou, Binbin Hu, Zhiqiang Zhang, and Wayne Xin Zhao. 2022. CORE: Simple and Effective Session-based Recommendation within Consistent Representation Space. In SIGIR 2022.

[26] Yupeng Hou, Xingyu Pan, Wayne Xin Zhao, Shuqing Bian, Yang Song, Tao Zhang, and Ji-Rong Wen. 2022. Leveraging Search History for Improving Person-Job Fit. In Database Systems for Advanced Applications: 27th International Conference, DASFAA 2022, Virtual Event, April 11–14, 2022, Proceedings, Part I. 38–54.

[27] Guangneng Hu, Yu Zhang, and Qiang Yang. 2018. Conet: Collaborative cross networks for cross-domain recommendation. In CIKM.

[28] Nicolas Hug. 2020. Surprise: A Python library for recommender systems. Journal of Open Source Software 5, 52 (2020), 2174.

[29] Mike Huisman, Jan N Van Rijn, and Aske Plaat. 2021. A survey of deep meta-learning. Artificial Intelligence Review 54, 6 (2021), 4483–4541.

[30] Rashidul Islam, Kamrun Naher Keya, Ziqian Zeng, Shimei Pan, and James Foulds. 2021. Debiasing Career Recommendations with Neural Fair Collaborative Filtering. In Proceedings of the Web Conference 2021. 3779–3790.

[31] Junshu Jiang, Songyun Ye, Wei Wang, Jingran Xu, and Xiaosheng Luo. 2020. Learning Effective Representations for Person-Job Fit by Feature Fusion. In CIKM.

[32] Toshihiro Kamishima, Shotaro Akaho, and Jun Sakuma. 2011. Fairness-aware Learning through Regularization Approach. In 2011 IEEE 11th International Conference on Data Mining Workshops. 643–650.

[33] SeongKu Kang, Junyoung Hwang, Dongha Lee, and Hwanjo Yu. 2019. Semi-supervised learning for cross-domain recommendation to cold-start users. In CIKM.

[34] Thomas N. Kipf and Max Welling. 2017. Semi-Supervised Classification with Graph Convolutional Networks. In ICLR.

[35] Taeyong Kong, Taeri Kim, Jinsung Jeon, Jeongwhan Choi, Yeon-Chang Lee, Noseong Park, and Sang-Wook Kim. 2022. Linear, or Non-Linear, That is the Question!. In WSDM.

[36] Ran Le, Wenpeng Hu, Yang Song, Tao Zhang, Dongyan Zhao, and Rui Yan. 2019. Towards effective and interpretable person-job fitting. In CIKM.

[37] Hoyeop Lee, Jinbae Im, Seongwon Jang, Hyunsouk Cho, and Sehee Chung. 2019. Melu: Meta-learned user preference estimator for cold-start recommendation. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 1073–1082.

[38] Jiacheng Li, Yujie Wang, and Julian J. McAuley. 2020. Time Interval Aware Self-Attention for Sequential Recommendation. In WSDM'20: The Thirteenth ACM International Conference on Web Search and Data Mining, Houston, TX, USA, February 3-7, 2020, James Caverlee, Xia (Ben) Hu, Mounia Lalmas, and Wei Wang (Eds.). 322–330.

[39] Yunqi Li, Hanxiong Chen, Shuyuan Xu, Yingqiang Ge, and Yongfeng Zhang. 2021. Towards Personalized Fairness Based on Causal Notion. Association for Computing Machinery, 1054–1063.

[40] Xixun Lin, Jia Wu, Chuan Zhou, Shirui Pan, Yanan Cao, and Bin Wang. 2021. Task-adaptive neural process for user cold-start recommendation. In Proceedings of the Web Conference 2021. 1306–1316.

[41] Zihan Lin, Changxin Tian, Yupeng Hou, and Wayne Xin Zhao. 2022. Improving Graph Collaborative Filtering with Neighborhood-enriched Contrastive Learning. In The Web Conf.

[42] Hanxiao Liu, Zihang Dai, David R So, and Quoc V Le. 2021. Pay Attention to MLPs. arXiv preprint arXiv:2105.08050 (2021).

[43] Meng Liu, Jianjun Li, Guohui Li, and Peng Pan. 2020. Cross domain recommendation via bi-directional transfer graph collaborative filtering networks. In CIKM.

[44] Zhiwei Liu, Yongjun Chen, Jia Li, Philip S. Yu, Julian J. McAuley, and Caiming Xiong. 2021. Contrastive Self-supervised Sequential Recommendation with Robust Augmentation. CoRR abs/2108.06479 (2021).

[45] Tong Man, Huawei Shen, Xiaolong Jin, and Xueqi Cheng. 2017. Cross-domain recommendation: An embedding and mapping approach. In IJCAI.

[46] Zaiqiao Meng, Richard McCreadie, Craig Macdonald, Iadh Ounis, Siwei Liu, Yaxiong Wu, Xi Wang, Shangsong Liang, Yucheng Liang, Guangtao Zeng, et al. 2020. Beta-rec: Build, evaluate and tune automated recommender systems. In Fourteenth ACM conference on recommender systems. 588–590.

[47] James Neve and Ivan Palomares. 2019. Latent factor models and aggregation operators for collaborative filtering in reciprocal recommender systems. In RecSys.

[48] Alex Nichol, Joshua Achiam, and John Schulman. 2018. On first-order meta-learning algorithms. arXiv preprint arXiv:1803.02999 (2018).

[49] Feiyang Pan, Shuokai Li, Xiang Ao, Pingzhong Tang, and Qing He. 2019. Warm up cold-start advertisements: Improving ctr predictions via learning to learn id embeddings. In Proceedings of the 42nd International ACM SIGIR Conference on Research and Development in Information Retrieval. 695–704.

[50] Zhiqiang Pan, Fei Cai, Wanyu Chen, Honghui Chen, and Maarten de Rijke. 2020. Star Graph Neural Networks for Session-based Recommendation. In CIKM.

[51] Evaggelia Pitoura, Kostas Stefanidis, and Georgia Koutrika. 2022. Fairness in rankings and recommendations: an overview. VLDBJ. 31, 3 (2022), 431–458.

[52] Chuan Qin, Hengshu Zhu, Tong Xu, Chen Zhu, Liang Jiang, Enhong Chen, and Hui Xiong. 2018. Enhancing person-job fit for talent recruitment: An ability-aware neural network approach. In SIGIR.

[53] Ruihong Qiu, Zi Huang, and Hongzhi Yin. 2021. Memory Augmented Multi-Instance Contrastive Predictive Coding for Sequential Recommendation. In ICDM 2021. 519–528.

[54] Ruihong Qiu, Zi Huang, Hongzhi Yin, and Zijian Wang. 2022. Contrastive Learning for Representation Degeneration Problem in Sequential Recommendation. In WSDM 2022.

[55] Yuta Saito, Suguru Yaginuma, Yuta Nishino, Hayato Sakata, and Kazuhide Nakata. 2020. Unbiased recommender learning from missing-not-at-random implicit feedback. In Proceedings of the 13th International Conference on Web Search and Data Mining. 501–509.

[56] Tobias Schnabel, Adith Swaminathan, Ashudeep Singh, Navin Chandak, and Thorsten Joachims. 2016. Recommendations as treatments: Debiasing learning and evaluation. In international conference on machine learning. PMLR, 1670–1679.

[57] Ajit P Singh and Geoffrey J Gordon. 2008. Relational learning via collective matrix factorization. In Proceedings of the 14th ACM SIGKDD international conference on Knowledge discovery and data mining. 650–658.

[58] Zhu Sun, Di Yu, Hui Fang, Jie Yang, Xinghua Qu, Jie Zhang, and Cong Geng. 2020. Are We Evaluating Rigorously? Benchmarking Recommendation for Reproducible Evaluation and Fair Comparison. In RecSys. ACM, 23–32.

[59] Manasi Vartak, Arvind Thiagarajan, Conrado Miranda, Jeshua Bratman, and Hugo Larochelle. 2017. A meta-learning perspective on cold-start recommendations for items. Advances in neural information processing systems 30 (2017).

[60] Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, and Yoshua Bengio. 2018. Graph Attention Networks. In ICLR.

[61] Chenyang Wang, Min Zhang, Weizhi Ma, Yiqun Liu, and Shaoping Ma. 2020. Make It a Chorus: Knowledge-and Time-aware Item Modeling for Sequential Recommendation. In SIGIR. ACM, 109–118.

[62] Xiang Wang, Xiangnan He, Meng Wang, Fuli Feng, and Tat-Seng Chua. 2019. Neural graph collaborative filtering. In SIGIR.

[63] Ziyang Wang, Wei Wei, Gao Cong, Xiao-Li Li, Xianling Mao, and Minghui Qiu. 2020. Global Context Enhanced Graph Neural Networks for Session-based Recommendation. In SIGIR.

[64] Zhenlei Wang, Jingsen Zhang, Hongteng Xu, Xu Chen, Yongfeng Zhang, Wayne Xin Zhao, and Ji-Rong Wen. 2021. Counterfactual Data-Augmented Sequential Recommendation. In SIGIR 2021. 347–356.

[65] Tianxin Wei, Fuli Feng, Jiawei Chen, Ziwei Wu, Jinfeng Yi, and Xiangnan He. 2021. Model-agnostic counterfactual reasoning for eliminating popularity bias in recommender system. In Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining. 1791–1800.

[66] Chuhan Wu, Fangzhao Wu, Mingxiao An, Jianqiang Huang, Yongfeng Huang, and Xing Xie. 2019. Neural News Recommendation with Attentive Multi-View Learning. In Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence, IJCAI 2019, Macao, China, August 10-16, 2019. 3863–3869.

[67] Chuhan Wu, Fangzhao Wu, Mingxiao An, Jianqiang Huang, Yongfeng Huang, and Xing Xie. 2019. NPA: Neural News Recommendation with Personalized Attention. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, KDD 2019, Anchorage, AK, USA, August 4-8, 2019. 2576–2584.

[68] Chuhan Wu, Fangzhao Wu, Suyu Ge, Tao Qi, Yongfeng Huang, and Xing Xie. 2019. Neural News Recommendation with Multi-Head Self-Attention. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing, EMNLP-IJCNLP 2019, Hong Kong, China, November 3-7, 2019. 6388–6393.

[69] Jiancan Wu, Xiang Wang, Fuli Feng, Xiangnan He, Liang Chen, Jianxun Lian, and Xing Xie. 2021. Self-supervised graph learning for recommendation. In SIGIR.

[70] Le Wu, Lei Chen, Pengyang Shao, Richang Hong, Xiting Wang, and Meng Wang. 2021. Learning Fair Representations for Recommendation: A Graph-Based Perspective. In Proceedings of the Web Conference 2021. Association for Computing Machinery, New York, NY, USA, 2198–2208.

[71] Liwei Wu, Shuqing Li, Cho-Jui Hsieh, and James Sharpnack. 2020. SSE-PT: Sequential Recommendation Via Personalized Transformer. In RecSys 2020: Fourteenth ACM Conference on Recommender Systems, Virtual Event, Brazil, September 22-26, 2020. 328–337.

[72] Le Wu, Peijie Sun, Yanjie Fu, Richang Hong, Xiting Wang, and Meng Wang. 2019. A Neural Influence Diffusion Model for Social Recommendation. In SIGIR.

[73] Shiwen Wu, Fei Sun, Wentao Zhang, Xu Xie, and Bin Cui. 2022. Graph Neural Networks in Recommender Systems: A Survey. ACM Comput. Surv. (2022).

[74] Shu Wu, Yuyuan Tang, Yanqiao Zhu, Liang Wang, Xing Xie, and Tieniu Tan. 2019. Session-Based Recommendation with Graph Neural Networks. In AAAI.

[75] Xu Xie, Fei Sun, Bolin Ding, and Bin Cui. 2020. Contrastive Pre-training for Sequential Recommendation. CoRR abs/2010.14395 (2020).

[76] Chengfeng Xu, Pengpeng Zhao, Yanchi Liu, Victor S. Sheng, Jiajie Xu, Fuzhen Zhuang, Junhua Fang, and Xiaofang Zhou. 2019. Graph Contextualized Self-Attention Network for Session-based Recommendation. In IJCAI.

[77] Huan Yan, Xiangning Chen, Chen Gao, Yong Li, and Depeng Jin. 2019. Deepapf: Deep attentive probabilistic factorization for multi-site video recommendation. (2019).

[78] Sirui Yao and Bert Huang. 2017. Beyond Parity: Fairness Objectives for Collaborative Filtering. In Advances in Neural Information Processing Systems, Vol. 30.

[79] Feng Yu, Yanqiao Zhu, Qiang Liu, Shu Wu, Liang Wang, and Tieniu Tan. 2020. TAGNN: Target Attentive Graph Neural Networks for Session-based Recommendation. In SIGIR.

[80] Junliang Yu, Hongzhi Yin, Min Gao, Xin Xia, Xiangliang Zhang, and Nguyen Quoc Viet Hung. 2021. Socially-Aware Self-Supervised Tri-Training for Recommendation. In KDD.

[81] Junliang Yu, Hongzhi Yin, Jundong Li, Qinyong Wang, Nguyen Quoc Viet Hung, and Xiangliang Zhang. 2021. Self-Supervised Multi-Channel Hypergraph Convolutional Network for Social Recommendation. In WWW.

[82] Junliang Yu, Hongzhi Yin, Xin Xia, Tong Chen, Lizhen Cui, and Nguyen Quoc Viet Hung. 2022. Are Graph Augmentations Necessary? Simple Graph Contrastive Learning for Recommendation. In SIGIR.

[83] Meike Zehlike, Ke Yang, and Julia Stoyanovich. 2021. Fairness in ranking: A survey. arXiv preprint arXiv:2103.14000 (2021).

[84] Shengyu Zhang, Dong Yao, Zhou Zhao, Tat-Seng Chua, and Fei Wu. 2021. CauseRec: Counterfactual User Sequence Synthesis for Sequential Recommendation. In SIGIR 2021. 367–377.

[85] Shuai Zhang, Lina Yao, Aixin Sun, and Yi Tay. 2019. Deep Learning Based Recommender System: A Survey and New Perspectives. ACM Comput. Surv. 52, 1 (2019), 5:1–5:38.

[86] Yang Zhang, Fuli Feng, Xiangnan He, Tianxin Wei, Chonggang Song, Guohui Ling, and Yongdong Zhang. 2021. Causal intervention for leveraging popularity bias in recommendation. In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval. 11–20.

[87] Wayne Xin Zhao, Shanlei Mu, Yupeng Hou, Zihan Lin, Yushuo Chen, Xingyu Pan, Kaiyuan Li, Yujie Lu, Hui Wang, Changxin Tian, Yingqian Min, Zhichao Feng, Xinyan Fan, Xu Chen, Pengfei Wang, Wendi Ji, Yaliang Li, Xiaoling Wang, and Ji-Rong Wen. 2021. RecBole: Towards a Unified, Comprehensive and Efficient Framework for Recommendation Algorithms. In CIKM.

[88] Yu Zheng, Chen Gao, Xiang Li, Xiangnan He, Yong Li, and Depeng Jin. 2021. Disentangling user interest and conformity for recommendation with causal embedding. In Proceedings of the Web Conference 2021. 2980–2991.

[89] Chen Zhu, Hengshu Zhu, Hui Xiong, Chao Ma, Fang Xie, Pengliang Ding, and Pan Li. 2018. Person-job fit: Adapting the right talent for the right job with joint representation learning. TMIS (2018).

[90] Feng Zhu, Chaochao Chen, Yan Wang, Guanfeng Liu, and Xiaolin Zheng. 2019. Dtcdr: A framework for dual-target cross-domain recommendation. In Proceedings of the 28th ACM International Conference on Information and Knowledge Management. 1533–1542.

[91] Feng Zhu, Yan Wang, Chaochao Chen, Guanfeng Liu, Mehmet Orgun, and Jia Wu. 2018. A deep framework for cross-domain and cross-system recommendations. In IJCAI.

[92] Jieming Zhu, Kelong Mao, Quanyu Dai, Liangcai Su, Rong Ma, Jinyang Liu, Guohao Cai, Zhicheng Dou, Xi Xiao, and Rui Zhang. 2022. BARS: Towards Open Benchmarking for Recommender Systems. arXiv preprint arXiv:2205.09626 (2022).

[93] Yongchun Zhu, Ruobing Xie, Fuzhen Zhuang, Kaikai Ge, Ying Sun, Xu Zhang, Leyu Lin, and Juan Cao. 2021. Learning to warm up cold item embeddings for cold-start recommendation with meta scaling and shifting networks. In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval. 1167–1176.
