# 大规模检索系统中利用迁移学习改进相关性预测

> Ruoxi Wang, Zhe Zhao, Xinyang Yi, Ji Yang, Derek Zhiyuan Cheng, Lichan Hong, Steve Tjoa, Jieqi Kang, Evan Ettinger, Ed H. Chi | Google Inc.

本文介绍了大规模检索系统中利用迁移学习改进相关性预测。核心内容：

- 提出一种新颖的迁移学习模型架构，利用丰富的用户隐式反馈（如点击）学习更好的查询和候选项表示
- 采用双塔深度神经网络（DNN）架构，引入共享底层（shared-bottom）结构联合优化显式反馈与隐式反馈两个训练目标
- 训练时先优化辅助的隐式反馈目标，再微调主要的显式反馈相关性目标，并在共享底层上应用梯度停止
- 在 Google 的工业大规模检索系统上评估所提出的模型

关键发现：

- 通过从丰富的隐式反馈迁移知识，显著提高了稀疏显式反馈的预测精度（相关性 RMSE 从 0.3464 降至 0.2673）
- 隐式反馈对模型起到正则化作用，防止过拟合

---

## 摘要

由机器学习构建的大规模检索系统需要大量训练数据来表示查询-item 相关性。然而，收集用户的显式反馈成本高昂。在本文中，我们提出利用用户日志和隐式反馈作为辅助目标来改善检索系统中的相关性建模。具体来说，我们采用双塔神经网络架构，结合协同信息和内容信息来建模查询-item 相关性。通过引入使用更丰富的隐式用户反馈数据训练的辅助任务，我们提高了查询和 item 学习表示的质量和区分度。将这些学习到的表示应用到工业检索系统中，已经带来了显著的改进。

## 1 引言

在本文中，我们提出了一种新颖的迁移学习模型架构，用于大规模检索系统。检索问题定义如下：给定一个查询和大量候选项集，检索出 top-$k$ 最相关的候选。检索系统在许多实际应用中非常有用，例如搜索[29]和推荐[7, 11, 32]。近年来构建大规模检索系统的研究主要集中在以下两个方面：

- **更好的表示学习。** 许多机器学习模型已被开发用于学习查询和候选项到嵌入空间的映射[15, 16]。这些模型利用各种特征，如协同信息和内容信息[30]。

- **高效的检索算法。** 在给定学习到的表示后，研究者提出了高效算法，根据与嵌入空间相关联的相似性（距离）度量来检索 top-$k$ 相关 item [4, 9]。

然而，设计和开发实际的大规模检索系统面临诸多挑战：

- **稀疏的相关性数据。** 收集用户对 item 相关性的真实意见成本高昂。通常，研究人员和工程师设计带有 Likert 量表问题的评估模板来评价相关性[6]，并通过众包平台（如 Amazon Mechanical Turk）征集反馈。

- **有噪声的反馈。** 此外，由于设计评估模板时的人为偏差以及提供反馈时的主观性，用户反馈往往高度主观且存在偏差。

- **多模态特征空间。** 我们需要在由多种模态生成的特征空间中学习相关性，例如查询内容特征、候选项内容特征、上下文特征以及来自查询与候选项之间连接的图特征[8, 22, 30]。

在本文中，我们提出利用用户对相关性的显式回答和用户的隐式反馈（如点击和其他类型的用户参与）来学习相关性。具体来说，我们开发了一个迁移学习框架，该框架首先使用大量用户隐式反馈学习有效的查询和候选项表示，然后使用从调查响应中收集的用户显式反馈来优化这些表示。所提出的模型架构如图 2 所示。

我们提出的模型基于大规模检索系统中常用的双塔深度神经网络（DNN，Deep Neural Network）[16]。如图 1 所示，该模型架构能够从多种模态的特征中学习有效的表示。这些表示随后可以使用高效的最近邻搜索系统进行服务[9]。

为了将隐式反馈中学到的知识迁移到显式反馈，我们扩展了双塔模型，采用了共享底层架构，该架构已在多任务学习场景中得到广泛应用[5]。具体来说，最终的损失包括隐式和显式反馈任务的训练目标。这两个任务共享一些隐藏层，每个任务都有自己独立的子塔。在服务时，仅使用和评估为显式反馈学习到的表示。

我们在工业大规模检索系统上的实验表明，通过从丰富的隐式反馈中迁移知识，我们可以显著提高稀疏相关性反馈的预测精度。

总之，我们的贡献如下：

- 我们提出了一个迁移学习框架，利用丰富的隐式反馈来学习更好的稀疏显式反馈表示。

- 我们设计了一种新颖的模型架构，该架构依次优化两个训练目标。

- 我们在真实的工业大规模检索系统上评估了我们的模型，并展示了显著的改进。

本文其余部分组织如下：第 2 节讨论了构建大规模检索系统的相关工作。第 3 节介绍了我们的问题和训练目标。第 4 节描述了我们的方法。第 5 节报告了大规模检索系统的实验结果。最后，在第 6 节中，我们总结了研究发现。

## 2 相关工作

在本节中，我们首先介绍一些最先进的工业级检索系统，然后讨论多任务学习和迁移学习技术在检索和推荐任务中的应用。

### 2.1 工业级检索系统

检索系统广泛应用于大规模应用，如搜索[29]和推荐[7, 11, 32]。近年来，工业界已从基于反向索引的方案[3]转向机器学习检索系统。协同过滤系统[1, 14]一直非常流行和成功，直到最近才被各种基于神经网络的检索模型所超越[2, 17, 32]。

检索系统涉及两个关键组件：表示学习和高效索引算法[20]。许多大规模工业检索系统已经成功使用双塔 DNN 模型来学习查询和候选项的独立表示[12, 16, 31]。

### 2.2 面向检索和推荐系统的多任务学习与迁移学习

现有研究也涉及基于张量分解的上下文感知检索应用中的多任务检索系统[33]。不幸的是，由于模型容量和服务时间限制的约束，该模型难以轻松适应从多个特征源学习复杂的特征表示。许多基于多任务 DNN 的推荐系统[7, 18]是为排序问题设计的，其中仅对少量高质量候选进行评分。这些完整的排序解决方案难以轻易应用于检索问题，在检索问题中，我们需要从拥有数百万到数亿候选 item 的大规模语料库中识别出数千个候选。

受这些工作的启发，我们提出了一个新颖的框架，结合了两方面的优势：（1）双塔模型架构的计算效率；（2）多任务 DNN 架构增强的模型能力[5]。这使我们能够将丰富的隐式反馈学习迁移到帮助稀疏的显式反馈任务。我们的工作与迁移学习[23, 24, 25, 28]和弱监督学习[10, 21, 26, 34]密切相关。

## 3 问题描述

在本节中，我们形式化定义检索问题，并介绍我们的训练数据和训练目标。

### 3.1 检索问题

检索问题定义如下：给定一个查询和一个候选项集语料库，返回 top-$k$ 最相关的 item。设 $\{x_i\}_{i=1}^{N} \subseteq X$ 和 $\{y_j\}_{j=1}^{M} \subseteq Y$ 分别为特征空间 $X$ 和 $Y$ 中查询和候选项的特征向量，其中 $N$ 和 $M$ 分别表示查询和候选项的数量。我们将检索系统建模为一个参数化评分函数 $s(\cdot, \cdot; \theta) : X \times Y \rightarrow \mathbb{R}$，其中 $\theta$ 表示模型参数。在推理时，对于给定的查询，选择具有 top-$k$ 得分 $s(x, y; \theta)$ 的 item。我们假设训练数据是一组查询-item 对 $\{(x_t, y_t)\}_{t=1}^{T}$，其中 $y_t$ 是与 $x_t$ 关联的候选项，具有显式或隐式的用户反馈，在实际中 $T \ll MN$。我们的目标是基于这些 $T$ 个样本拟合评分函数。

### 3.2 使用用户反馈进行训练

在训练基于机器学习的检索系统时，理想的方法是使用用户的显式反馈，这反映了 item 与查询的相关性。然而，获取用户的显式反馈成本高昂；因此，许多现有系统使用来自用户日志的隐式反馈，如点击。

在本文中，我们研究同时具有显式和隐式反馈的检索系统，其中隐式反馈丰富而显式反馈相对稀疏。

### 3.3 使用辅助目标进行联合优化

我们检索问题的目标是学习更好的查询和候选项表示，使得查询-候选项对之间的相似度能够紧密逼近相关性。因此，我们的主要训练目标是最小化预测相关性与真实相关性之间的差异。

为了促进表示学习，我们引入了一个辅助目标，用于捕获用户对 item 的参与度，例如 item 点击、购物检索中的产品购买、或电影推荐中的电影观看。

形式上，我们旨在同时学习两个目标 $s_{exp}(\cdot, \cdot; \theta)$ 和 $s_{imp}(\cdot, \cdot; \theta^{\prime})$，同时共享 $\theta$ 和 $\theta^{\prime}$ 之间的部分参数。我们假设一些样本 $(x_t, y_t)$ 在集合 $E$ 中带有显式反馈，而其他样本在集合 $I$ 中带有隐式反馈。此外，每个样本 $(x_t, y_t) \in E$ 关联一个标签 $l_t \in \mathbb{R}$，表示用户的显式反馈，例如对相关性调查的响应。注意 $E$ 和 $I$ 并非互斥，因为一些样本可以同时具有隐式和显式反馈。

我们使用回归损失来拟合集合 $E$ 中样本的用户显式反馈。一种示例损失是均方误差（MSE，Mean Squared Error）：

$$
L_{exp}(\theta; E) = \frac{1}{|E|} \sum_{(x_t, y_t) \in E} (s_{exp}(x_t, y_t; \theta) - l_t)^2 \qquad (1)
$$

其中 $|\cdot|$ 表示集合基数。另一方面，我们将隐式反馈的建模视为对整个 item 语料库的多类分类任务，并使用 softmax 公式来建模选择 item $y$ 的概率，即：

$$
P(y | x; \theta^{\prime}) = \frac{\exp(s_{imp}(x, y; \theta^{\prime}))}{\sum_{j=1}^{M} \exp(s_{imp}(x, y_j; \theta^{\prime}))}
$$

极大似然估计（MLE，Maximum Likelihood Estimation）可表述为：

$$
L_{imp}(\theta^{\prime}; I) = -\frac{1}{|I|} \sum_{(x_t, y_t) \in I} \log(P(y_t | x_t; \theta^{\prime})) \qquad (2)
$$

使用损失权重 $w$ 和 $w^{\prime}$，我们通过联合优化公式 (1) 和 (2) 中的损失：

$$
L(\theta, \theta^{\prime}) = w \cdot L_{exp}(\theta; E) + w^{\prime} \cdot L_{imp}(\theta^{\prime}; I)
$$

## 4 模型架构

在本节中，我们描述提出的用于学习大规模检索问题中相关性的框架。我们通过引入共享底层架构来扩展双塔模型架构。

### 4.1 双塔模型架构

图 1 提供了双塔 DNN 模型架构的高级图示。给定一对由特征向量 $x \in X$、$y \in Y$ 表示的查询和 item，左侧和右侧塔分别提供两个基于 DNN 的参数化嵌入函数 $u : X \times \mathbb{R}^{d} \rightarrow \mathbb{R}^{k}$、$v : Y \times \mathbb{R}^{d} \rightarrow \mathbb{R}^{k}$，它们将查询和 item 的特征编码到 $k$ 维嵌入空间。评分函数然后计算为查询嵌入和 item 嵌入在顶层的点积，即：

$$
s(x, y; \theta) = \langle u(x, \theta), v(y, \theta) \rangle
$$

### 4.2 用于迁移学习的共享底层架构

为了实现多任务学习，我们通过采用共享底层架构来扩展双塔模型。具体来说，我们在底层隐藏层之上引入了两个子塔，一个用于显式反馈任务，另一个用于隐式反馈任务。底层隐藏层的输出被并行输入到两个子塔。底层隐藏层在两个子塔之间共享[5]，并称为共享底层。最终的模型架构如图 2 所示。

### 4.3 训练和服务方案

在训练过程中，我们首先训练辅助用户参与度目标模型，该模型使用交叉熵损失。在学习到共享表示后，我们对模型进行微调以用于主要的相关性目标，该目标使用平方损失。为了防止稀疏相关性数据可能导致的过拟合，我们对相关性目标在共享底层上应用梯度停止。

在服务时，我们只需存储和提供两个相关性子塔的顶层来预测相关性。

## 5 评估

在本节中，我们描述在 Google 的一个大规模检索系统（用于相关 item 推荐，例如应用推荐）上对所提出框架进行的实验。

### 5.1 实验设置

我们的系统包含数百万个候选项。我们的训练数据包含来自相关性调查的数十万条显式反馈，以及来自用户日志的数十亿条隐式反馈。

我们随机将数据分为 90% 用于训练，10% 用于评估。模型性能通过评估集上相关性预测的均方根误差（RMSE，Root Mean Square Error）来衡量。该模型使用 TensorFlow 实现，其输出的查询和候选项的相关性嵌入用于检索服务。包括模型大小、学习率和训练步数在内的超参数经过仔细调整以获得最佳模型性能。

### 5.2 实验结果

我们研究了将迁移学习应用于相关性预测的效果。以下实验结果表明，迁移学习显著提高了稀疏相关性任务的预测质量，并有助于避免过拟合。

表 1 报告了不同训练目标和特征类型组合的相关性 RMSE（数值越低越好）。我们可以看到，与仅使用显式反馈相比，使用隐式反馈带来了显著的改进。此外，协同信息与内容信息结合使用的模型优于仅使用协同信息的模型。

| 训练目标 | 使用的特征 | 相关性 RMSE |
|---------|-----------|------------|
| 仅显式反馈 | 仅协同 | 0.3583 |
| 仅显式反馈 | 协同+内容 | 0.3464 |
| 显式+隐式反馈 | 仅协同 | 0.2837 |
| 显式+隐式反馈 | 协同+内容 | 0.2673 |

**表 1：不同训练目标和特征信息集上的相关性评估 RMSE。**

表 2 报告了两种训练目标集下不同模型大小的相关性 RMSE。作为模型大小的近似，我们报告了乘法运算次数。对于仅使用稀疏显式反馈训练的模型，增加模型大小会导致过拟合，从而降低模型性能。相比之下，对于使用隐式反馈训练的模型，增加模型大小会提高模型性能。这表明隐式反馈对模型起到了正则化作用，并防止了过拟合。

| 训练目标 | 乘法运算次数 | 相关性 RMSE |
|---------|------------|------------|
| 仅显式反馈 | 51K | 0.3447 |
| 仅显式反馈 | 68K | 0.3464 |
| 显式+隐式反馈 | 176K | 0.2775 |
| 显式+隐式反馈 | 802K | 0.2673 |

**表 2：不同模型大小上的相关性评估 RMSE。**

### 5.3 讨论与未来工作

迁移学习的成功取决于辅助任务和主要任务的合理参数化。一方面，我们需要足够的容量从大量辅助数据中学习高质量的表示。另一方面，我们希望限制主要任务的容量，以避免对其稀疏标签的过拟合。因此，我们提出的模型架构与传统的预训练微调模型略有不同[13]。除了共享层之外，每个任务都有自己不同容量的隐藏层。此外，我们应用了两阶段训练和梯度停止，以避免主要任务和辅助任务之间极端数据倾斜可能导致的潜在问题。

我们的经验激励我们在以下方向继续工作：

- 我们将考虑使用不同的多任务学习框架来处理多种类型的用户隐式反馈，例如多门专家混合（Multi-gate Mixture-of-Expert，MMoE）[18]和子网路由（Sub-Network Routing，SNR）[19]。我们将继续探索新的模型架构，将迁移学习与多任务学习相结合。

- 辅助任务需要超参数调优来为主任务学习最佳表示。我们将探索 AutoML（Automated Machine Learning，自动机器学习）[27]技术，以自动学习查询塔和候选塔跨任务的合理参数化。

## 6 结论

在本文中，我们提出了一种新颖的模型架构，通过迁移学习来学习更好的查询和候选项表示。我们扩展了双塔神经网络方法，通过利用具有丰富隐式反馈的辅助任务来增强稀疏任务学习。通过引入辅助目标并联合使用隐式反馈训练该模型，我们观察到在 Google 的一个大规模检索系统上，相关性预测有了显著改进。

## 参考文献

[1] Beutel, A., Chi, E. H., Cheng, Z., Pham, H., and Anderson, J. Beyond globally optimal: Focused learning for improved recommendations. In WWW, 2017.

[2] Beutel, A., Covington, P., Jain, S., Xu, C., Li, J., Gatto, V., and Chi, E. H. Latent cross: Making use of context in recurrent recommender systems. In WSDM, 2018.

[3] Brin, S. and Page, L. The anatomy of a large-scale hypertextual web search engine. Computer networks and ISDN systems, 30(1-7):107–117, 1998.

[4] Broder, A. Z. On the resemblance and containment of documents. In Proceedings. Compression and Complexity of SEQUENCES 1997, pp. 21–29. IEEE, 1997.

[5] Caruana, R. Multitask learning. Machine learning, 28(1):41–75, 1997.

[6] Chang, S., Dai, P., Chen, J., and hsin Chi, E. H. Got many labels?: Deriving topic labels from multiple sources for social media posts using crowdsourcing and ensemble learning. In WWW, 2015.

[7] Covington, P., Adams, J., and Sargin, E. Deep neural networks for youtube recommendations. In RecSys, 2016.

[8] Cui, B., Tung, A. K., Zhang, C., and Zhao, Z. Multiple feature fusion for social media applications. In SIGKDD, 2010.

[9] Guo, R., Kumar, S., Choromanski, K., and Simcha, D. Quantization based fast inner product search. In Artificial Intelligence and Statistics, pp. 482–490, 2016.

[10] Han, J., Zhang, D., Cheng, G., Guo, L., and Ren, J. Object detection in optical remote sensing images based on weakly supervised learning and high-level feature learning. IEEE Transactions on Geoscience and Remote Sensing, 53(6):3325–3337, 2014.

[11] He, X., Pan, J., Jin, O., Xu, T., Liu, B., Xu, T., Shi, Y., Atallah, A., Herbrich, R., Bowers, S., et al. Practical lessons from predicting clicks on ads at facebook. In Proceedings of the Eighth International Workshop on Data Mining for Online Advertising, pp. 1–9. ACM, 2014.

[12] He, X., Liao, L., Zhang, H., Nie, L., Hu, X., and Chua, T.-S. Neural collaborative filtering. In WWW, 2017.

[13] Hinton, G. E. and Salakhutdinov, R. R. Reducing the dimensionality of data with neural networks. Science, 313(5786):504–507, 2006.

[14] Hu, Y., Koren, Y., and Volinsky, C. Collaborative filtering for implicit feedback datasets. In ICDM, 2008.

[15] Koren, Y., Bell, R., and Volinsky, C. Matrix factorization techniques for recommender systems. Computer, (8):30–37, 2009.

[16] Krichene, W., Mayoraz, N., Rendle, S., Zhang, L., Yi, X., Hong, L., Chi, E., and Anderson, J. Efficient training on very large corpora via gramian estimation. In ICLR, 2019.

[17] Liu, D. C., Rogers, S., Shiau, R., Kislyuk, D., Ma, K. C., Zhong, Z., Liu, J., and Jing, Y. Related pins at pinterest: The evolution of a real-world recommender system. In WWW, 2017.

[18] Ma, J., Zhao, Z., Yi, X., Chen, J., Hong, L., and Chi, E. H. Modeling task relationships in multi-task learning with multi-gate mixture-of-experts. In KDD, 2018.

[19] Ma, J., Zhao, Z., Chen, J., Li, A., Hong, L., and Chi, E. Snr: Sub-network routing for flexible parameter sharing in multi-task learning. AAAI, 2019.

[20] Manning, C., Raghavan, P., and Schütze, H. Introduction to information retrieval. Natural Language Engineering, 16(1):100–103, 2010.

[21] Oquab, M., Bottou, L., Laptev, I., and Sivic, J. Is object localization for free?-weakly-supervised learning with convolutional neural networks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 685–694, 2015.

[22] Page, L., Brin, S., Motwani, R., and Winograd, T. The pagerank citation ranking: Bringing order to the web. Technical report, Stanford InfoLab, 1999.

[23] Pan, S. J. and Yang, Q. A survey on transfer learning. IEEE Transactions on knowledge and data engineering, 22(10):1345–1359, 2009.

[24] Pan, W., Xiang, E. W., Liu, N. N., and Yang, Q. Transfer learning in collaborative filtering for sparsity reduction. In Twenty-fourth AAAI conference on artificial intelligence, 2010.

[25] Pan, W., Liu, N. N., Xiang, E. W., and Yang, Q. Transfer learning to predict missing ratings via heterogeneous user feedbacks. In Twenty-Second International Joint Conference on Artificial Intelligence, 2011.

[26] Papandreou, G., Chen, L.-C., Murphy, K. P., and Yuille, A. L. Weakly-and semi-supervised learning of a deep convolutional network for semantic image segmentation. In Proceedings of the IEEE international conference on computer vision, pp. 1742–1750, 2015.

[27] Pham, H., Guan, M., Zoph, B., Le, Q., and Dean, J. Efficient neural architecture search via parameters sharing. In Dy, J. and Krause, A. (eds.), Proceedings of the 35th International Conference on Machine Learning, volume 80, pp. 4095–4104. PMLR, 10–15 Jul 2018.

[28] Raina, R., Battle, A., Lee, H., Packer, B., and Ng, A. Y. Self-taught learning: transfer learning from unlabeled data. In Proceedings of the 24th international conference on Machine learning, pp. 759–766. ACM, 2007.

[29] Shen, Y., He, X., Gao, J., Deng, L., and Mesnil, G. Learning semantic representations using convolutional neural networks for web search. In WWW, 2014.

[30] Wang, J., Zhao, Z., Zhou, J., Wang, H., Cui, B., and Qi, G. Recommending flickr groups with social topic model. Information retrieval, 15(3-4):278–295, 2012.

[31] Yang, Y., Yuan, S., Cer, D., Kong, S.-y., Constant, N., Pilar, P., Ge, H., Sung, Y.-H., Strope, B., and Kurzweil, R. Learning semantic textual similarity from conversations. arXiv preprint arXiv:1804.07754, 2018.

[32] Yi, X., Chen, Y.-F., Ramesh, S., Rajashekhar, V., Hong, L., Fiedel, N., Seshadri, N., Heldt, L., Wu, X., and Chi, H. Deep retrieval and distributed tensorflow serving. In SysML Conference, Feb 2018.

[33] Zhao, Z., Cheng, Z., Hong, L., and Chi, E. H. Improving user topic interest profiles by behavior factorization. In WWW, 2015.

[34] Zhou, Z.-H. A brief introduction to weakly supervised learning. National Science Review, 5(1):44–53, 2017.
