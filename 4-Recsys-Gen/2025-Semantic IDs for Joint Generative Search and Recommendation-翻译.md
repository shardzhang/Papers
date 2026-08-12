# Semantic IDs for Joint Generative Search and Recommendation

> Gustavo Penha, Edoardo D'Amico, Marco De Nadai, Enrico Palumbo, Alexandre Tamborrino, Ali Vardasbi, Max Lefarov, Shawn Lin, Timothy Heath, Francesco Fabbri, Hugues Bouchard | Spotify

由大语言模型（LLM）驱动的生成式模型正在成为同时支持推荐和搜索任务的统一解决方案。本文探讨了在联合生成式搜索与推荐模型中如何构建语义ID，比较了任务特定和跨任务策略。结果表明，使用在搜索和推荐任务上联合微调的双编码器模型获取item嵌入，并构建统一语义ID空间，提供了一种有效的权衡。

- 跨任务语义ID在联合搜索与推荐生成模型中优于任务特定ID
- 联合微调的双编码器（Multi-task）在搜索和推荐效能之间提供了最佳折衷
- RQ-KMeans量化器在此实验设置中优于RQ-VAE等学习方法
- 任务特定嵌入空间会在提升一个任务性能的同时牺牲另一任务

---

## 摘要

由大语言模型（LLM）驱动的生成式模型正在成为同时支持推荐和搜索任务的统一解决方案。这些模型的一个关键设计选择是如何表示item，传统上通过唯一标识符（ID），以及最近通过由离散码组成的语义ID（从嵌入中获得）。虽然针对特定任务的嵌入模型可以提升个别任务的性能，但它们可能在联合设置中泛化不佳。在本文中，我们探讨了如何构建在使用统一模型时在搜索和推荐中均表现良好的语义ID。我们比较了一系列构建语义ID的策略，研究了针对特定任务和跨任务的方法，以及在联合搜索和推荐生成模型中每个任务是否应拥有自己的语义ID token。我们的结果表明，使用在搜索和推荐任务上微调的双编码器模型来获取item嵌入，然后构建统一的语义ID空间，提供了一种有效的权衡，使得两个任务都能获得强劲性能。我们希望这些发现能激发关于可泛化、语义基础的ID方案的后续工作，并为下一代统一生成式推荐系统架构提供信息。

ACM引用格式：
Gustavo Penha, Edoardo D'Amico, Marco De Nadai, Enrico Palumbo, Alexandre Tamborrino, Ali Vardasbi, Max Lefarov, Shawn Lin, Timothy Heath, Francesco Fabbri, and Hugues Bouchard. 2025. Semantic IDs for Joint Generative Search and Recommendation. 收录于第十九届ACM推荐系统会议论文集（RecSys '25），2025年9月22–26日，捷克共和国，布拉格。ACM，美国纽约州纽约市，共6页。https://doi.org/10.1145/3705328.3759300

允许为个人或课堂教学制作或分发本作品的全部或部分数字或硬拷贝，无需付费，前提是拷贝不得用于盈利或商业优势，且拷贝需在第一页包含此声明和完整引用。第三方组件版权必须得到尊重。对于所有其他用途，请联系作者/所有者。

RecSys '25，捷克共和国，布拉格
© 2025 版权归作者/所有者所有。
ACM ISBN 979-8-4007-1364-4/2025/09

## 1 引言

由大语言模型（LLM）驱动的生成式模型正在改变我们处理推荐和搜索任务的方式。最近的工作不是构建针对特定任务的模型，而是探索了统一生成式框架，这些框架简化了系统设计并可能提升跨任务的泛化能力。

在生成式模型中，item表示必须映射到LLM可以消费和产生的离散token [4, 9, 12, 12, 15]。传统推荐系统通常使用添加到模型词汇表中的唯一item标识符（ID），例如SASRec [10]中那样，而其他系统则使用基于启发式的顺序ID（例如P5 [6]），或使用参与实体的标题等token效率较低的解决方案 [3]。这些方法大多在添加新的冷启动item时需要重新训练模型，并且在工业环境中存在不足。

为了解决这一问题，最近的工作提出了使用语义ID，即从预训练的item嵌入中生成的一组离散token [18, 21]。这些ID允许具有相似内容（嵌入）的item共享token，从而改进泛化能力并支持冷启动场景。然而，生成有效的语义ID关键取决于用于构建它们的嵌入空间。先前的工作已经表明，为特定任务（如推荐或搜索）微调嵌入，可以生成对该任务最有效的语义ID [16, 26]。然而，这引发了一个重要问题：

我们能否创建在联合生成模型中对搜索和推荐都表现良好的语义ID？

这一点尤为重要，因为联合搜索与推荐（Joint S\&R）方法已成为一种有前途的策略，可以减少工程开销并提升性能 [1, 14, 25, 27, 29]，它统一了历来被视为孤立隔离的数据源和模型。

在本文中，我们研究了在联合搜索和推荐生成模型背景下语义ID的构建。具体来说，我们研究了不同的嵌入来源（针对搜索、推荐或两者微调）如何影响模型的下游性能。我们还考虑了共享或为每个任务使用特定token来表示item是否更有利。

我们的发现证实了构建语义ID时存在一个基本权衡：针对一个任务（例如推荐）微调的嵌入会降低另一个任务（例如搜索）的性能，反之亦然。这质疑了在生成模型中使用针对特定任务的嵌入空间作为ID生成先验的日益增长的趋势 [16, 26]。

我们表明，在搜索和推荐上联合微调的双编码器模型提供了一个令人信服的折衷方案（见图1），生成的语义ID能够跨任务泛化，而对单个任务效能的影响极小。这挑战了传统观念，即最优性能需要为每个任务构建单独的ID。相反，我们的结果表明，共享表示空间可以简化生成式模型设计而不牺牲质量，特别是在多任务系统中。随着基于LLM的推荐持续发展，我们相信这项工作提供了一个及时的信号：更好的生成性能可能并非来自针对特定任务的专门化，而是来自构建良好、可泛化的语义ID。

## 2 动机

生成式模型承诺为搜索和推荐提供单一架构，然而先前的工作是在针对特定任务的空间中优化ID。我们问道：当这些独立的优化在统一模型中碰撞时会发生什么？为了回答这个问题，我们重现了三种ID构建流程：

**表1：为搜索和推荐微调用于语义ID构建的嵌入，在联合生成模型中是有效的。然而，选择其中一个微调嵌入空间是以牺牲另一个任务的效能为代价的。粗体表示最高效能，上标表示使用配对t检验和Bonferroni校正的统计显著性。**

| 嵌入空间 | 搜索 R@30 ( $\pm$ 标准差) | 推荐 R@30 ( $\pm$ 标准差) |
|---|---|---|
| 1 基于内容（例如DSI [21], TIGER [18]） | 0.013 ( $\pm$ 0.009) | 0.023 ( $\pm$ 0.017) |
| 2 基于搜索（例如RIPOR [26]） | **0.072 ( $\pm$ 0.028)** $^{13}$ | 0.026 ( $\pm$ 0.017) |
| 3 基于推荐（例如TokenRec [16]） | 0.004 ( $\pm$ 0.001) | **0.062 ( $\pm$ 0.015)** $^{12}$ |

(1) **基于内容**：文本嵌入，如TIGER [18]和DSI [21]，未针对搜索或推荐进行微调。
(2) **搜索微调ID**：为检索有效性微调的嵌入，类似于Zeng等人 [26]。
(3) **推荐微调ID**：为推荐微调的协同过滤模型嵌入，类似于Qu等人 [16]。

在计算目录中每个item的此类嵌入后，通过应用ID策略将每个item嵌入token化为一组离散token来获得语义ID。

表1显示了每种嵌入空间的语义ID在接入我们的Joint S\&R模型时的性能。模式很清晰：优化一个任务会牺牲另一个任务。搜索微调的ID将检索性能提升了约5倍，但将推荐性能降低了60%。为推荐微调的ID则效果相反。同期工作也报告了同样的紧张关系 [20]。

这些观察为我们的贡献奠定了基础：我们能否构建一组单一的语义ID来平衡双方？在下一节中，我们将介绍一个在两个任务上联合微调的双编码器，并展示其共享嵌入空间如何调和权衡，而不增加Joint S\&R模型的计算成本。

![图1：使用不同方法构建语义ID的联合生成模型对搜索和推荐的效能。我们研究了在表示item和构建语义ID时同时考虑搜索和推荐的嵌入。]()

## 3 语义ID构建

在本节中，我们定义了多种将item映射到ID的方法，分为：(i) 针对特定任务策略和 (ii) 跨任务策略。

### 3.1 针对特定任务方法

图2描绘了两种针对特定任务基线共享的流程。每种方法都在单个监督信号上训练嵌入模型，然后用量化器对该嵌入进行离散化。

**基于搜索。** 遵循基于相关性的ID构建 [26]，我们在搜索数据 $D_S$ 上训练一个双编码器模型，该数据由查询和相关item对组成，使用批次内交叉负样本。我们使用item的拼接元数据（例如标题、描述）作为其文档表示。得到的嵌入 $\mathbf{v}_{\text{search}}$ 随后被离散化以生成语义ID，这些ID同时用于搜索和推荐任务。

**基于推荐。** 类似地，遵循TokenRec [16]，我们在用户交互item的数据集 $D_R$ 上训练高效神经矩阵分解（ENMF）模型 [2]，以创建基于协同过滤的嵌入 $\mathbf{v}_{\text{rec}}$ 。每个item的嵌入被离散化并供给生成式模型，用于两个任务。

![图2：针对特定任务的联合生成式搜索与推荐模型语义ID构建方法。左侧我们展示了基于为搜索微调的双编码器模型（基于搜索）的语义ID，而右侧我们展示了基于为推荐微调的协同过滤模型嵌入（基于推荐）的语义ID。]()

### 3.2 跨任务方法

我们现在描述五种明确结合两个任务信息的方法。

**Token分离的ID。** 图3展示了Separate变体。我们简单地将任务标签前置到上述两个针对特定任务的ID前面，生成token： $\text{ID}^{\text{sep}}_i = \langle \text{SEARCH}:\text{ID}^{\text{search}}_i, \text{REC}:\text{ID}^{\text{rec}}_i \rangle$ 。这种策略使ID词汇表大小翻倍，但保持训练简单：搜索提示只能输出搜索token，而推荐提示只能输出推荐token。

**前缀共享ID。** 借鉴Shi等人 [20] 的思想，Prefix-share分配三个码本：一个共享码本（SHARED）加上上述两个针对特定任务的前缀。一个编码器接收拼接的嵌入 $[\mathbf{v}^{\text{search}}_i; \mathbf{v}^{\text{rec}}_i]$ ；两个解码器学习特定于码本的重建。最终的ID由共享token后跟特定任务token拼接而成。

![图3：Token分离的语义ID方法（Separate），其中每个任务拥有自己的一组从搜索和推荐特定嵌入构建的语义ID。]()

**嵌入组合的ID。** 图4展示了三种嵌入级融合策略。

- **Fused $_{\text{concat}}$ 。** 我们对 $\mathbf{v}^{\text{search}}_i$ 和 $\mathbf{v}^{\text{rec}}_i$ 进行 $\ell_2$ 归一化并拼接它们： $\mathbf{v}^{\text{concat}}_i = [\mathbf{v}^{\text{search}}_i; \mathbf{v}^{\text{rec}}_i]$ 。
- **Fused $_{\text{SVD}}$ 。** 我们再次对两个嵌入进行归一化，但首先用截断SVD降低高维空间，使两者具有相同的维度 $d$ 。然后逐元素相加： $\mathbf{v}^{\text{svd}}_i = \mathbf{v}^{\text{search}}_i + \mathbf{v}^{\text{rec}}_i$ 。
- **Multi-task。** 我们在两个监督信号上训练双编码器：来自 $D_S$ 的查询-item对和来自 $D_R$ 的共现item对。共享编码器用两个对比损失之和进行优化，生成携带检索和协同过滤线索的嵌入 $\mathbf{v}^{\text{mt}}_i$ 。

![图4：嵌入组合方法，其中两个任务都被考虑。左侧是Multi-task，它首先为搜索和推荐训练双编码器，然后生成语义ID。右侧是Fused，它组合来自搜索和推荐模型的两种嵌入，然后使用组合结果生成语义ID。]()

## 4 实验设置

**数据集。** 遵循Penha等人 [14]，我们从MovieLens25M [7] 构建了一个S\&R数据集。该数据集包含62,138部电影、124万条用户-item交互（按时间顺序拆分；每个用户的最后一次交互用于测试）以及每个item恰好20条自然语言查询（10条训练/10条测试），这些查询使用Gemini-2.0-flash生成 $^1$ 。我们注意到，每个item分配10/10条训练/测试查询消除了搜索数据集的热度偏差。由于我们不知道搜索热度的真实分布（即没有MovieLens数据的真实用户日志），我们决定使用均匀分布。这意味着搜索热度分布与推荐热度分布非常不同，因此在实际热度分布具有一定相似性的真实分布中，结果可能会更有利。使用item内容的方法（基于内容的、基于搜索的和跨任务方法）使用标题、年份、描述、流派、标签和基因组标签 [23] 来计算其嵌入。

**评估指标。** 鉴于我们关注检索任务，我们使用Recall@30。每个模型使用不同的随机种子运行五次，并报告不同运行的平均召回率。我们使用95%置信区间的配对Student t检验来评估结果的统计显著性。

**嵌入模型。** 我们使用以下模型生成item嵌入：

 $^1$ 我们使用以下提示："Your task is to return a list with 10 queries for a given movie (title of the movie, year and description and tags) After generating the initial set of queries, you should also generate a list of the same size with paraphrased of the first queries. The paraphrased queries should be similar to the original queries, but with different words, structure and slight variations in the meaning. The queries should be realistic things that a user would ask to find the movie. The queries should be diverse and cover different aspects of the movie. The queries should not include the title of the movie, but be broader descriptions of the movie and its content. The queries should also contain broad topics, themes and genres of the movie. Movie: {METADATA}"

- **基于内容的：** 我们使用来自sentence transformers [19] 的预训练all-mpnet-base-v2，基于数据集中描述的拼接元数据。
- **搜索：** 我们从相同的预训练模型出发，在搜索数据上进一步微调，使用批次内随机负样本（MultipleNegativesRankingLoss），使用sentence transformers进行5个epoch，batch 512，LR 2e-5，Adam。
- **推荐：** 我们通过RecBole [28] 训练ENMF [2]，30个epoch，batch 512，embedding 256，LR 0.001，Adam。

**生成式模型。** google/flan-t5-base [17]，在S\&R上联合训练3个epoch（LR 0.002，batch 128，AdamW，weight-decay 0.01）。为了增加所有生成式检索模型检索到的不同item数量，我们采用了多样化束搜索方法 [24]：beam 60，diversity penalty 0.25，30 groups。

**ID token化。** 除非另有说明，我们使用大小为256的两个码本（总共512个token）。某些跨任务方法需要额外的token。Separate在词汇表中增加了总共1024个新token，因为每个语义ID空间被单独处理。Prefix-share有256个共享token和512个特定任务token。

token使用RQ-KMeans（FAISS残差量化器 [5]）为所有模型构建。对于消融实验，我们还评估了来自Sklearn的MiniBatchDictionaryLearning [13]、ResidualLFQ和来自vector-quantize-pytorch $^3$ 的RQ-VAE。

 $^3$ https://github.com/lucidrains/vector-quantize-pytorch

## 5 结果

**面向联合搜索和推荐的语义ID。** 表2展示了不同语义ID构建方法在搜索和推荐上的结果。

**针对特定任务方法。** 前两行显示了每个任务的最佳表现者，如动机部分所述。然而，它们基于专用ID，无法为统一的S\&R生成式模型提供令人满意的权衡，因为每个语义ID都过度拟合了目标任务。有趣的是，我们看到对于Torso实体，基于搜索的嵌入是有效的，这表明每个实体的热度在推荐数据中起重要作用，而对于不太流行的item，更多地依赖内容是有效的。

**表2：使用不同语义ID构建方法（同时考虑搜索和推荐两个目标）的联合生成模型的R@30。Head表示训练集中最热门的前1% item的效能，Torso表示剩余的item集。搜索数据没有热度偏差，即所有item具有相同数量的查询。粗体表示最高分数，下划线表示第二高分。**

| 语义ID构建 | 搜索 R@30 (全部) | 推荐 R@30 (全部) | 推荐 R@30 (Head) | 推荐 R@30 (Torso) |
|---|---|---|---|---|
| **针对特定任务** | | | | |
| 基于搜索 | 0.072 | 0.026 | 0.090 | 0.070 |
| 基于推荐 | 0.004 | 0.062 | 0.170 | 0.035 |
| **跨任务** | | | | |
| Separate | 0.028 | 0.032 | 0.120 | 0.051 |
| Prefix-share | 0.007 | 0.021 | 0.058 | 0.010 |
| Fused $_{\text{concat}}$ | 0.048 | 0.018 | 0.045 | 0.041 |
| Fused $_{\text{SVD}}$ | 0.033 | 0.038 | 0.105 | 0.060 |
| Multi-task | 0.046 | 0.049 | 0.135 | 0.024 |

**跨任务方法。** 剩余的行是使用来自两个任务的嵌入的方法。Separate为每个任务使用不同的token，为生成式模型的词汇表添加更多token（针对特定任务的语义ID token）。这意味着从一个任务学到的知识不能用于另一个任务的特定任务token，从而否定了Penha等人 [14] 讨论的item表示中的正则化效应。

Prefix-share也表现不如其他方法，因为底层的量化方法在此处表现不佳（参见下一节关于token化方法的消融）。我们观察到，这两种方法都表现不如首先组合嵌入然后构建语义ID的方法（最后三行）。

融合方法的结果表明，如果其中一个嵌入空间大于另一个（双编码器模型为386维，而ENMF为256维），嵌入的拼接可能会有问题。具有更大维度的嵌入空间可能得到更多的表征。通过使用Fused $_{\text{SVD}}$ 使维度相同，我们提高了模型的推荐效能，同时相比Fused $_{\text{concat}}$ 降低了搜索效能 $^4$ 。

 $^4$ 我们未探索的另一种使嵌入空间大小相等的解决方案是使用Matryoshka目标 [11] 训练模型，并仅使用前几个维度。

最后，我们看到为搜索和推荐两个任务训练编码模型（Multi-task）提供了搜索和推荐效能的有效权衡。Vančura等人 [22] 提出了将协同过滤嵌入注入基于内容模型的类似解决方案。

我们相信这是一个有希望的方向，可以获得在搜索和推荐问题上都表现良好的item表示。

**Token化方法消融。** 表3显示了token化方法的消融结果，这些方法接收嵌入作为输入并输出离散token，同时保持来自Multi-task的嵌入空间固定（使用其他语义ID构建方法进行相同消融时发现了类似结果）。我们看到RQ-KMeans是为我们的数据集构建ID的最佳方法，优于RQ-VAE等常见方法。Hong等人 [8] 也发现RQ-VAE不稳定，并在他们的实验中选择使用分层k-means。我们将探究其在此实验设置中优于学习的自编码器方法的原因留待未来工作。

**表3：使用Multi-task方法的嵌入对不同token化方法的联合生成式搜索与推荐模型的R@30（使用其他语义ID构建方法进行相同消融时发现了类似结果）。**

| 方法 | 搜索 | 推荐 |
|---|---|---|
| RQ-KMeans | 0.046 | 0.049 |
| Dictionary encoding | 0.019 | 0.029 |
| ResidualLFQ | 0.018 | 0.023 |
| RQ-VAE | 0.002 | 0.024 |

## 6 结论

在本文中，我们展示了语义ID的构建方式对于服务于搜索和推荐任务的统一生成式模型的效能是一个决定性因素，系统地比较了针对特定任务的、token分离的和嵌入组合的构建方法。

我们发现，针对特定任务的语义ID仅在孤立情况下表现出色，而跨任务的语义ID则提供了平衡、高质量的解决方案，且不增加token预算。对离散化方法的消融进一步表明，轻量级的RQ-KMeans token化器优于VQ-VAE变体。

这些观察提供了早期的经验证据，表明统一item表示不仅是可行的，而且是有利的。随着基于LLM的检索和推荐系统在实践中趋于融合 [1]，这一见解至关重要。通过强调通往共享语义ID的实用路径，我们的研究提供了该领域发展方向的一个及时快照，并指出了表示学习、token效率和冷启动鲁棒性等方面的几个开放问题。我们希望这些发现能激发关于可泛化、语义基础的ID方案的后续工作，并为下一代统一生成式推荐系统架构提供信息。

## 参考文献

[1] Moumita Bhattacharya, Vito Ostuni, and Sudarshan Lamkhede. 2024. Joint Modeling of Search and Recommendations Via an Unified Contextual Recommender (UniCoRn). In *Proceedings of the 18th ACM Conference on Recommender Systems*. 793–795.

[2] Chong Chen, Min Zhang, Yongfeng Zhang, Yiqun Liu, and Shaoping Ma. 2020. Efficient neural matrix factorization without sampling for recommendation. *ACM Transactions on Information Systems (TOIS)* 38, 2 (2020), 1–28.

[3] Nicola De Cao, Gautier Izacard, Sebastian Riedel, and Fabio Petroni. 2020. Autoregressive entity retrieval. *arXiv preprint arXiv:2010.00904* (2020).

[4] Seungheon Doh, Keunwoo Choi, and Juhan Nam. 2025. TALKPLAY: Multimodal Music Recommendation with Large Language Models. *arXiv preprint arXiv:2502.13713* (2025).

[5] Matthijs Douze, Alexandr Guzhva, Chengqi Deng, Jeff Johnson, Gergely Szilvasy, Pierre-Emmanuel Mazaré, Maria Lomeli, Lucas Hosseini, and Hervé Jégou. 2024. The Faiss library. (2024). arXiv:2401.08281 [cs.LG]

[6] Shijie Geng, Shuchang Liu, Zuohui Fu, Yingqiang Ge, and Yongfeng Zhang. 2022. Recommendation as language processing (rlp): A unified pretrain, personalized prompt \& predict paradigm (p5). In *Proceedings of the 16th ACM conference on recommender systems*. 299–315.

[7] F Maxwell Harper and Joseph A Konstan. 2015. The movielens datasets: History and context. *ACM transactions on interactive intelligent systems (tiis)* 5, 4 (2015), 1–19.

[8] Minjie Hong, Yan Xia, Zehan Wang, Jieming Zhu, Ye Wang, Sihang Cai, Xiaoda Yang, Quanyu Dai, Zhenhua Dong, Zhimeng Zhang, et al. 2025. EAGER-LLM: Enhancing Large Language Models as Recommenders through Exogenous Behavior-Semantic Integration. In *Proceedings of the ACM on Web Conference 2025*. 2754–2762.

[9] Wenyue Hua, Shuyuan Xu, Yingqiang Ge, and Yongfeng Zhang. 2023. How to index item ids for recommendation foundation models. In *Proceedings of the Annual International ACM SIGIR Conference on Research and Development in Information Retrieval in the Asia Pacific Region*. 195–204.

[10] Wang-Cheng Kang and Julian McAuley. 2018. Self-attentive sequential recommendation. In *2018 IEEE international conference on data mining (ICDM)*. IEEE, 197–206.

[11] Aditya Kusupati, Gantavya Bhatt, Aniket Rege, Matthew Wallingford, Aditya Sinha, Vivek Ramanujan, William Howard-Snyder, Kaifeng Chen, Sham Kakade, Prateek Jain, et al. 2022. Matryoshka representation learning. *Advances in Neural Information Processing Systems* 35 (2022), 30233–30249.

[12] Enrico Palumbo, Gustavo Penha, Andreas Damianou, José Luis Redondo García, Timothy Christopher Heath, Alice Wang, Hugues Bouchard, and Mounia Lalmas. 2025. Text2Tracks: Prompt-based Music Recommendation via Generative Retrieval. *arXiv preprint arXiv:2503.24193* (2025).

[13] F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, and E. Duchesnay. 2011. Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research* 12 (2011), 2825–2830.

[14] Gustavo Penha, Ali Vardasbi, Enrico Palumbo, Marco De Nadai, and Hugues Bouchard. 2024. Bridging Search and Recommendation in Generative Retrieval: Does One Task Help the Other?. In *Proceedings of the 18th ACM Conference on Recommender Systems*. 340–349.

[15] Aleksandr V Petrov and Craig Macdonald. 2023. Generative sequential recommendation with gptrec. *arXiv preprint arXiv:2306.11114* (2023).

[16] Haohao Qu, Wenqi Fan, Zihuai Zhao, and Qing Li. 2024. TokenRec: Learning to Tokenize ID for LLM-based Generative Recommendation. arXiv:2406.10450 [cs.IR] https://arxiv.org/abs/2406.10450

[17] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. *Journal of machine learning research* 21, 140 (2020), 1–67.

[18] Shashank Rajput, Nikhil Mehta, Anima Singh, Raghunandan Hulikal Keshavan, Trung Vu, Lukasz Heldt, Lichan Hong, Yi Tay, Vinh Tran, Jonah Samost, et al. 2023. Recommender systems with generative retrieval. *Advances in Neural Information Processing Systems* 36 (2023), 10299–10315.

[19] Nils Reimers and Iryna Gurevych. 2019. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing*. Association for Computational Linguistics. https://arxiv.org/abs/1908.10084

[20] Teng Shi, Jun Xu, Xiao Zhang, Xiaoxue Zang, Kai Zheng, Yang Song, and Enyun Yu. 2025. Unified Generative Search and Recommendation. *arXiv preprint arXiv:2504.05730* (2025).

[21] Yi Tay, Vinh Tran, Mostafa Dehghani, Jianmo Ni, Dara Bahri, Harsh Mehta, Zhen Qin, Kai Hui, Zhe Zhao, Jai Gupta, et al. 2022. Transformer memory as a differentiable search index. *Advances in Neural Information Processing Systems* 35 (2022), 21831–21843.

[22] Vojtěch Vančura, Pavel Kordík, and Milan Straka. 2024. beeFormer: Bridging the Gap Between Semantic and Interaction Similarity in Recommender Systems. In *Proceedings of the 18th ACM Conference on Recommender Systems*. 1102–1107.

[23] Jesse Vig, Shilad Sen, and John Riedl. 2012. The tag genome: Encoding community knowledge to support novel interaction. *ACM Transactions on Interactive Intelligent Systems (TiiS)* 2, 3 (2012), 1–44.

[24] Ashwin K Vijayakumar, Michael Cogswell, Ramprasath R Selvaraju, Qing Sun, Stefan Lee, David Crandall, and Dhruv Batra. 2016. Diverse beam search: Decoding diverse solutions from neural sequence models. *arXiv preprint arXiv:1610.02424* (2016).

[25] Hamed Zamani and W Bruce Croft. 2018. Joint modeling and optimization of search and recommendation. *arXiv preprint arXiv:1807.05631* (2018).

[26] Hansi Zeng, Chen Luo, Bowen Jin, Sheikh Muhammad Sarwar, Tianxin Wei, and Hamed Zamani. 2024. Scalable and Effective Generative Information Retrieval. In *Proceedings of the ACM Web Conference 2024* (Singapore, Singapore) (WWW '24). Association for Computing Machinery, New York, NY, USA, 1441–1452. doi:10.1145/3589334.3645477

[27] Jujia Zhao, Wenjie Wang, Chen Xu, Xiuying Chen, Zhaochun Ren, and Suzan Verberne. 2025. Unifying Search and Recommendation: A Generative Paradigm Inspired by Information Theory. *arXiv preprint arXiv:2504.06714* (2025).

[28] Wayne Xin Zhao, Yupeng Hou, Xingyu Pan, Chen Yang, Zeyu Zhang, Zihan Lin, Jingsen Zhang, Shuqing Bian, Jiakai Tang, Wenqi Sun, Yushuo Chen, Lanling Xu, Gaowei Zhang, Zhen Tian, Changxin Tian, Shanlei Mu, Xinyan Fan, Xu Chen, and Ji-Rong Wen. 2022. RecBole 2.0: Towards a More Up-to-Date Recommendation Library. In *CIKM*. ACM, 4722–4726.

[29] Bowen Zheng, Yupeng Hou, Hongyu Lu, Yu Chen, Wayne Xin Zhao, Ming Chen, and Ji-Rong Wen. 2024. Adapting large language models by integrating collaborative semantics for recommendation. In *2024 IEEE 40th International Conference on Data Engineering (ICDE)*. IEEE, 1435–1448.
