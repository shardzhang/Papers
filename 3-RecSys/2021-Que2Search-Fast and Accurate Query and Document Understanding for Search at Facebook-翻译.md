# Que2Search：面向Facebook搜索的快速准确查询与文档理解

> Yiqun Liu, Kaushik Rangadurai, Yunzhong He, Siddarth Malreddy, Xunlong Gui, Xiaoyi Liu, Fedor Borisyuk | Facebook Inc.
>
> Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '21), August 14–18, 2021, Virtual Event, Singapore

本文介绍了 Que2Search：面向Facebook搜索的快速准确查询与文档理解系统。核心内容：

- 提出Que2Search，一个已部署的面向搜索的查询与产品理解系统，利用多任务与多模态学习方法训练查询与产品表示
- 结合XLM/XLM-R多语言语言理解架构与多模态融合技术，在Facebook现有最先进产品理解系统基础上实现超过5%的绝对离线相关性改进
- 在Facebook规模下部署基于XLM的查询理解模型，实现CPU上P99延迟低于1.5毫秒

关键发现：

- 多阶段课程训练、批内难负采样、多任务学习与模态融合等建模技术在离线与在线A/B实验中均带来增益
- 在线A/B测试显示搜索产品参与度提升超过4%，其中约2.8%来自语义检索，1.24%来自排序
- 简单注意力融合优于拼接融合，分组特征梯度混合额外提供+0.91%的ROC AUC提升

---

## 摘要

在本文中，我们提出了Que2Search，一个已部署的面向搜索的查询与产品理解系统。Que2Search利用多任务和多模态学习方法训练查询与产品表示。通过将最新的多语言自然语言理解架构（如XLM和XLM-R）与多模态融合技术相结合，我们在Facebook现有最先进的产品理解系统基础上实现了超过5%的绝对离线相关性改进和超过4%的在线参与度增益。在本文中，我们描述了如何在Facebook规模下部署基于XLM的搜索查询理解模型，该模型在CPU上P99延迟低于1.5毫秒，这在该行业一直是一个重大挑战。我们还基于大量的离线与在线A/B实验，描述了哪些模型优化有效（以及哪些无效）。我们将Que2Search部署到Facebook Marketplace搜索中，并分享我们的生产部署经验和调优技巧，以在在线A/B实验中实现更高效率。Que2Search已在生产应用中展现出收益，并在Facebook规模下运行。

**CCS概念**

• 信息系统 $\rightarrow$ 多媒体与多模态检索；在线购物。

**关键词**

Product understanding, e-commerce, multi-modal learning, embedding, deep learning

**ACM引用格式：**

Yiqun Liu, Kaushik Rangadurai, Yunzhong He, Siddarth Malreddy, Xunlong Gui, Xiaoyi Liu, Fedor Borisyuk. 2021. Que2Search: Fast and Accurate Query and Document Understanding for Search at Facebook. In Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '21), August 14–18, 2021, Virtual Event, Singapore. ACM, New York, NY, USA, 9 pages. https://doi.org/10.1145/3447548.3467127

---

## 1 引言

Facebook Marketplace^1 是一个全球性平台，使全球各地的企业和买家能够买卖商品。数以亿计的产品被列出待售，Marketplace为买家提供了一个探索、发现和购买所需商品的生态系统。我们的目标是提升Marketplace搜索引擎的质量和召回率，以便我们能够根据用户的文本查询展示正确的产品集合。参见图1中针对查询"hats"的Marketplace搜索引擎输出。

**图1：Facebook Marketplace 搜索结果页的截图。**

^1 http://www.facebook.com/marketplace

传统上，搜索引擎基于各种词项匹配方法[1]。最近，Huang等人[14]将基于嵌入的检索引入Facebook搜索：[14]的作者使用字符n-gram稀疏特征来表示文本特征，并依赖宽深度神经网络为查询和文档构建嵌入表示[6]。这种方法缺乏对自然语言的表示能力，并且主要由于高计算需求而没有采用基于BERT（Bidirectional Encoder Representations from Transformers，来自Transformer的双向编码器表示）[10]的模型。在我们之前的工作中，我们为产品图像理解构建了最先进的图像识别系统（Bell等人[2]，Tang等人[25]）。除产品图像外，其他文本和类别型产品特征，如标题、产品描述、位置和产品类别，也提供了有价值的信息。

本文提出了Que2Search，一个查询到产品的相似度模型，该模型提供了一种考虑全面多模态特征并利用XLM（Cross-lingual Language Model，跨语言语言模型）编码器（Lample和Conneau[17]）处理文本特征的建模方法。Que2Search在弱监督数据集上训练，与Facebook之前的技术基线相比，在产品表示方面达到了最先进的性能。Que2Search已部署到Facebook Marketplace搜索中；它为查询到产品的基于嵌入的检索提供支持，并用作排序特征。Que2Search在Facebook Marketplace规模下运行，支持多语言，并满足严格的延迟约束。

在构建Que2Search过程中存在若干挑战，我们在此列出其中一些：

• **噪声产品描述**：卖家提供的产品描述质量差异很大。在许多情况下，市场产品的属性缺失或拼写错误。

• **国际化支持**：我们希望构建一个在Facebook Marketplace所启用的多种语言中都能表现良好的模型。

• **多模态的高效处理**：我们需要将所有多模态特征（如产品图像和文本信息）有效地整合到一个模型中。

• **严格的延迟约束**：我们需要满足搜索引擎的严格延迟约束，这尤其具有挑战性，因为我们使用了基于Transformer的语言模型，而这类模型以计算开销大而闻名。

最近在语言理解方面的最先进技术源于基于BERT[10]的Transformer语言模型。扩展BERT-based模型和改进推理延迟的挑战在整个行业都是众所周知的[21, 22]。推理时间对于搜索问题尤为关键，因为我们需要实时地为自由形式的文本查询生成嵌入表示。我们在查询端使用一个2层XLM编码器[17]来转换查询文本，在产品端，我们采用多语言方法，使用XLM-R（Cross-lingual Language Model RoBERTa，跨语言RoBERTa语言模型）[7]编码器对产品的文本字段进行编码。我们在§3.5中分享了在CPU上将XLM/XLM-R模型扩展到搜索场景并在1.5毫秒内运行的经验。

Que2Search通过引入多种建模技术在生产应用中展现了收益，包括多阶段课程学习、多模态处理以及联合优化查询到产品检索任务和产品分类任务的多任务学习。我们在§3中分享建模技术，在§5中分享消融研究，在§6中分享生产部署经验和在在线A/B实验中实现更高性能的调优技巧。

---

## 2 相关工作

**基于嵌入的检索。** 传统上，Siamese网络[3]被用于建模成对关系，并且由于它提供了一种独立于查询计算对象嵌入并将其存储在搜索索引中的方式，在生产应用的检索中十分有效。近年来行业内的一系列工作研究了在各种应用中使用Siamese网络与宽深度架构[6]组合进行大规模语义检索，其中每个塔使用一个网络。Yang等人[27]训练了一个双塔神经网络为商业移动应用商店的检索提供支持，并引入了混合负采样，从搜索索引中随机采样负例，此外还使用批内随机负例。Huang等人[14]引入了双塔宽深度网络方法用于Facebook搜索，他们使用一组基于字符n-gram的稀疏特征来建模文本特征，并且主要处理单一模态数据。近年来有一些工作在使用基于先进BERT[10]模型的搜索应用方面取得了进展，例如[11, 23]。Guo等人[11]提出在排序框架内使用深度网络，并采用了多种优化手段来减少计算问题，例如对文档使用更小的模型，以及在服务前预计算文档端嵌入。本文分享了我们在利用基于Transformer的语言模型处理文本特征方面的经验，并描述了融合文本、视觉和类别型模态的实用多模态训练技术。

**多模态建模。** 近年来，作者们[4, 15, 18]一直在研究跨文本和视觉领域的多模态表示解决方案，并在研究数据集上取得了最先进的结果。许多论文在两个塔之间采用早期融合，这排除了独立部署查询塔模型和文档塔模型的可能性。本文聚焦于采用晚期融合的双塔Siamese网络，其中两个塔可以在生产中独立计算。

**自然语言处理。** 随着BERT[10]的突破，自然语言理解达到了新的质量水平。由于Transformer的高计算需求，各家公司探索了BERT优化，包括[11, 21, 22]。我们应用了[11]中的一些类似方法，并讨论了在生产中为改进推理延迟而采取的额外方法。最近，XLM-R（Conneau等人[7]）将BERT方法适配到多语言应用中。我们使用XLM-R来转换文本特征，并使用搜索日志数据进行微调，我们在§3中分享我们的解决方案。

---

## 3 建模

在本节中，我们首先描述模型架构并介绍输入特征。然后我们描述如何训练和评估模型，包括一个课程训练方案，在该方案中我们通过使用不同的损失函数（不改变训练数据收集过程）向模型提供更难的负样本。我们还讨论了在多模态训练中面临的挑战以及我们的解决方案。最后，我们描述了如何使用ML可解释性技术来理解模型并识别改进领域。

### 3.1 模型架构

模型架构如图2所示。对于查询塔，我们使用三个输入特征——字符三-gram[14]多类别特征；搜索者所在国家，一个单类别特征；以及原始查询文本，这是一个文本特征。对于字符三-gram，我们应用一个哈希函数（例如MurmurHash）到查询的3个字符的滑动窗口上，从而获得查询的哈希id列表。然后我们将哈希id列表传递给一个EmbeddingBag^2[24]，它为每个哈希id学习一个嵌入并执行求和池化，为每个查询提供固定大小的表示。我们还应用EmbeddingBag为搜索者国家单类别特征学习国家表示。我们对原始查询文本应用一个2层XLM[17]编码器。我们将最后一层的[CLS]标记的表示作为编码器表示，并将其投影到目标维度以获得查询的XLM表示。然后我们基于注意力权重融合这三个表示，得到查询的最终表示。

^2 https://pytorch.org/docs/stable/generated/torch.nn.EmbeddingBag.html

**图2：Que2Search架构——Facebook的可扩展查询与产品理解模型。**

对于文档塔，我们拥有产品标题、描述和图像等输入特征。我们使用一个共享的6层XLM-R[7]编码器用于所有文本字段（如标题和描述）；共享编码器有助于在batch recall@1上取得改进（见表3）。我们还应用EmbeddingBag[24]分别对标题和描述的字符三-gram多类别特征进行编码，如前一段所述。每个文档附带可变数量的图像。我们为每张附带的图像取预训练的图像表示（Bell等人[2]），应用一个共享的MLP（Multilayer Perceptron，多层感知机）层和Deep Sets（Zaheer等人[29]）融合来获得图像通道表示。与查询塔相同，我们随后使用学习到的注意力权重融合来自不同特征通道的表示，得到文档的最终表示。

我们尝试了不同的晚期融合技术来融合不同特征通道的表示，例如拼接融合（即，将所有编码器输出拼接后接一个MLP层）和简单注意力融合（基于学习到的通道权重的通道表示加权求和，如公式1所示，其中 $\parallel$ 表示拼接）。与拼接融合相比，简单注意力融合显示出更好的指标性能。

$$
\begin{aligned}
& \varphi = \{\varphi_i\}_{i=1}^N \quad \text{representations of } N \text{ channels} \\
& \Phi = \varphi_1 \parallel ... \parallel \varphi_N \quad \text{concatenation} \\
& a = \text{Softmax}(\Phi W), \qquad W \in \mathbb{R}^{ND \times N} \\
& f = \sum_{i=1}^N a_i \odot \varphi_i \quad \text{final tower representation}
\end{aligned}
\qquad (1)
$$

我们使用PyTorch以及多个下游库，如Facebook的PyText、Fairseq和多模态框架（MMF，Multimodal Framework）来实现模型。我们使用的学习率为 $7e^{-4}$，批量大小为768，优化器为Adam。然而，由于我们使用预训练的XLM/XLM-R模型，我们对XLM模块实施了一个不同的学习率 $2e^{-4}$——这意味着XLM模块内的所有参数与其余参数相比将具有不同的学习率。这使得验证集的ROC AUC（Receiver Operating Characteristic - Area Under the Curve，受试者工作特征曲线下面积）提高了1%（在表4中称为varying XLM LR）。我们使用了标准的正则化技术，如dropout（dropout率为0.1）和梯度裁剪为1.0。我们使用验证集上的ROC AUC指标进行早停，耐心值为3个epoch。

#### 3.1.1 带分类的双塔

我们还将双塔模型结构扩展为在文档塔中增加一个额外的分类任务，如图2右上部分所示。对于每个文档，我们从去标识化和聚合的Marketplace搜索日志数据中收集一个估计与该文档相关的文本查询列表（定义见§3.2）。我们保留最常见的45k个查询。我们将该问题视为一个多标签多类分类任务，并使用多标签交叉熵，其中每个正目标设置为 $1/k$（如果该文档有 $k$ 个正标签）[19]。我们在§5中将这种多任务架构的结果与原始双塔模型进行比较。

### 3.2 训练

**训练数据。** 在传统的监督学习中，我们要求训练数据同时包含正例和负例。然而，在我们的训练设置中，我们只取正训练（查询，文档）对，并自动从查询到批内其他文档获得负例。正（查询，文档）对可以通过搜索日志收集。具体来说，我们使用与[2, 25]中类似的数据收集模式：我们通过过滤以下事件序列从Facebook Marketplace搜索日志中创建文档-查询对数据集：（1）用户搜索一个查询，（2）点击一个产品，（3）给Marketplace卖家发消息，以及（4）卖家回复。如果所有4个事件在短时间内发生（例如24小时内），则该用户很可能找到了他们想要的东西，因此该查询被认为与该文档相关。我们无法访问消息内容，只知道用户与卖家进行了交互。我们将此类用户交互序列称为in-conversation[25]。

**批内负例：** 我们使用了多种方式生成负例。为了解释批内负例如何工作，我们将解释在训练过程中一个批量大小为 $B$ 的批次内发生的情况。在每个批次中，我们获得维度为 $D$ 的查询嵌入张量 $\{q_i\}_{i=1}^B$，类似地，我们也获得相同大小的文档嵌入张量 $\{d_j\}_{j=1}^B$。然后我们计算一个余弦相似度矩阵 $\{\cos(q_i, d_j)\}_{i,j=1}^B$。该余弦相似度矩阵表示批次内所有可能的查询-文档对的相似度，矩阵的行属于查询，列属于文档。我们将其视为一个多类分类问题，其中类别数为 $B$，文档 $d_i$ 是查询 $q_i$ 的真实标签类（如图4中绿色网格所示），而其他文档 $d_j$（$j \neq i$）是 $q_i$ 的负例。我们使用一个缩放的多类交叉熵损失（公式2）来优化网络。在损失中使用缩放乘以余弦的思想也在Deng等人[9]中提及。在训练过程中，我们发现scale对于损失收敛很重要。在我们的用例中，我们选择的scale在15到20之间。

$$
\text{loss}_i = -\log \frac{\exp(s \cdot \cos(q_i, d_i))}{\sum_{j=1}^B \exp(s \cdot \cos(q_i, d_j))} \qquad (2)
$$

其中 $s$ 表示 scale。

我们还尝试了对称缩放交叉熵损失（公式3）。该损失函数同时优化查询到文档检索和文档到查询检索。虽然这一损失在我们本文讨论的查询到文档双塔模型中既未改善也未降低整体系统指标，但在一个不同的用例中——我们训练文档到文档的双塔模型——该对称损失在评估数据集上显示出2%的ROC AUC改进。

$$
-\frac{1}{2}\left(\log \frac{\exp(s \cdot \cos(q_i, d_i))}{\sum_j \exp(s \cdot \cos(q_i, d_j))} + \log \frac{\exp(s \cdot \cos(q_i, d_i))}{\sum_j \exp(s \cdot \cos(q_j, d_i))}\right) \qquad (3)
$$

### 3.3 课程训练

除了§3.2中描述的批内负采样外，我们设计了一个课程训练方案，其中双塔模型在第二阶段训练中接受更难负例的训练，并在验证集上实现了超过1%的绝对ROC AUC增益（见表4）。在第一阶段训练中，我们使用批内负例训练模型，并通过在验证集ROC AUC指标上的早停让模型收敛。我们希望向模型提供"更难"的负例用于第二阶段训练。传统上，这意味着运行一个单独的数据管道来寻找难负例并将该数据集提供给模型，而我们建议使用批次内的负例：我们照常获得形状为 $(B \times B)$ 的余弦相似度矩阵。对于每一行，我们现在提取除对角线分数外的最高分数作为难负样本；我们将该分数的列索引记为 $nq_i = \arg\max_{j \neq i}^{B} \cos(q_i, d_j)$。这将有助于以 $(q_i, d_i, d_{nq_i})$ 格式生成训练样本。我们使用这些三元组样本探索了缩放二元交叉熵损失（公式4）和边界排序损失（公式5），并观察到边界在0.1到0.2之间的边界排序损失效果最佳。批内难负采样还减少了寻找难负例所需的额外CPU使用和维持另一个离线数据管道的额外成本。

最初，使用更难负例的课程训练效果并不好。通过实验，我们发现确保第一阶段训练在启动第二阶段训练之前收敛至关重要。此外，我们观察到MarginRank损失^3中reduction=sum比reduction=mean表现更好。

^3 https://pytorch.org/docs/stable/generated/torch.nn.MarginRankingLoss.html

$$
\text{loss}_i = -\log(\sigma(s \cdot \cos(q_i, d_i))) + \log(1 - \sigma(s \cdot \cos(q_i, d_{nq_i}))) \qquad (4)
$$

$$
\text{loss}_i = \max(0, -[\cos(q_i, d_i) - \cos(q_i, d_{nq_i})] + \text{margin}) \qquad (5)
$$

图3是我们某个模型训练实例的多阶段训练ROC AUC评估指标曲线，我们在第二阶段训练中看到了ROC AUC的提升。

### 3.4 评估

在通过在线流量进行A/B测试之前，我们在离线状态下评估我们的候选模型，并基于batch recall@$K$、ROC AUC和KNN（K-Nearest Neighbors，K最近邻）Recall与Precision选择最佳模型。

**Batch recall@$K$：** 该指标衡量对角线元素 $\cos(q_i, d_i)$ 是否位于行 $\cos(q_i, d_j)$（$j \in [1, B]$）的前 $K$ 个分数中。该指标在训练期间易于计算，并且是模型优化的最接近指标，使我们能够快速迭代建模思路。

**图4：随机负例的生成。**

**ROC AUC：** 在Facebook Marketplace中，我们定期收集人工评分数据以评估搜索引擎质量。人工评分评估针对公开可见的产品进行，查询数据在评估前经过去标识化和聚合处理。给定一组预选的搜索查询（数量通常为数千），我们抓取搜索引擎结果并将其发送给人工评分员进行评估：1表示满足查询搜索意图的相关文档，0表示不相关。这为我们提供了一个带有标签的（查询，文档）对评估数据集。以推断出的查询和文档嵌入的余弦相似度 $\cos(q, d)$ 作为分数，我们计算ROC AUC。ROC AUC在训练期间用作早停的验证指标，并在模型训练后用作离线评估指标。人工评分数据帮助我们评估对相关性和搜索质量的潜在影响。

**KNN Recall@$K$：** 另一方面，我们通过KNN Recall和Precision评估对在线参与度的潜在影响。KNN Recall@$K$ 是一个离线评估指标，在模型训练后在评估数据集上计算。我们使用数周的在线参与度数据（§3.2中定义的in-conversation）训练模型，并在未来未见天的数据上进行评估：给定一个训练好的双塔模型和一个评估数据集，推断查询嵌入和文档嵌入。给定查询嵌入，我们使用Faiss（Johnson等人[16]）K近邻在文档嵌入空间中检索 $N$ 个相似文档，并检查原始查询的真实标签文档是否位于前 $K$ 个检索文档中。我们观察到KNN Recall@$K$ 指标与基于嵌入的检索的在线性能密切相关。

### 3.5 加速模型推理

在Facebook，我们的系统需要以极快速度处理大量QPS（Queries Per Second，每秒查询数），以服务我们庞大的用户群。搜索查询由用户实时发起，我们期望在几百毫秒内返回结果。查询端模型需要足够快以满足系统需求。因此，我们试验了多种Transformer架构以处理延迟和准确率之间的权衡。我们在查询塔中试验了两个XLM模型：一个具有2层、4个注意力头和128的句子嵌入维度，另一个具有3个编码器层、4个注意力头和256的句子嵌入维度。我们发现前者的P99（99th percentile，第99百分位）延迟为1.5毫秒，而后者的P99延迟为3.5毫秒，但性能增益极小。我们将查询序列长度上限设为Marketplace搜索中所有查询长度的第99百分位数。我们使用了0.1的dropout率和句子嵌入维度3倍的前馈嵌入维度。我们在所有实验中使用大小为150k的SentencePiece词汇表。

文档塔模型在执行速度方面的要求较低，因为文档端嵌入可以在离线状态下预计算，或在文档创建后近实时计算，然后被摄取到搜索系统后端。我们在§4中分享更多关于服务系统优化的细节。

我们还利用Torch JIT（Just-In-Time，即时）编译来加速推理。

### 3.6 不同模态的融合

我们尝试了多种方法来混合不同模态，例如文本模态和图像模态。默认方法是公式1中描述的简单注意力融合。我们将这种方法称为基线模态处理。

GB（Gradient Blending，梯度混合）[26]是一种优化技术，用于平衡不同模态之间的贡献，并在多模态设置中减少对主导模态的过拟合。在梯度混合中，我们训练 $M+1$ 个模型：$M$ 个单模态模型和一个完整的多模态模型，其中 $M$ 是模态数量。我们计算每个模型的损失，并使用估计的权重取损失的加权和作为最终损失。这背后的直觉是让模型即使在模态不存在时也能学习，使模型对稀疏输入特征具有鲁棒性。

我们将梯度混合扩展到双塔架构，每次只保持一个塔具有全部模态。例如在图2中，我们说查询塔有 $m = 2$ 种模态：查询和国家；文档塔有 $n = 3$ 种模态：标题、描述和图像。然后我们得到 $m + n + 1$ 个损失，公式如下，其中 $\{\varphi_i^1\}_{i=1}^m$ 和 $\{\varphi_j^2\}_{j=1}^n$ 分别是左塔和右塔的单模态深度网络，$\oplus$ 表示融合操作：

$$
\begin{aligned}
& L_{\text{multi}} = L(\cos(\varphi_1^1 \oplus ... \oplus \varphi_m^1, \varphi_1^2 \oplus ... \oplus \varphi_n^2)) \\
& L_i^1 = L(\cos(\varphi_i^1, \varphi_1^2 \oplus ... \oplus \varphi_n^2)), \quad i \in [1, m] \\
& L_j^2 = L(\cos(\cos(\varphi_1^1 \oplus ... \oplus \varphi_m^1, \varphi_j^2))), \quad j \in [1, n]
\end{aligned}
\qquad (6)
$$

模态是一个定义较为松散的概念。我们可以将每个输入特征通道视为一个单独的模态。在这种情况下，在图2中，查询塔有 $m = 3$ 种模态，文档塔有 $n = 5$ 种模态。我们也可以将相关特征分组视为一种模态。例如，如果产品描述特征缺失，那么描述原始文本和描述字符三-gram都不可用。我们可以将这两个特征分组在一起。这样在文档塔中，我们只有 $n = 3$ 种模态：标题、描述和图像。我们在表1中分享了关于梯度混合的实验结果。我们没有观察到通过使用个体特征通道的梯度混合在离线指标上取得统计显著的改善，但观察到分组特征梯度混合在ROC AUC中提供了+0.91%的性能提升。梯度混合仅在训练期间应用，推理期间我们照常运行完整的多模态模型。

**表1：在人工评分数据集上的模态融合评估**

| 技术 | ROC AUC%（相对） |
|---|---|
| 基线模态处理 | - |
| GB - 个体特征 | -1.2%（不显著） |
| GB - 分组特征 | +0.91% |

### 3.7 模型可解释性

#### 3.7.1 XLM编码器是否为查询塔增加价值？

一个常见的问题是，在查询塔中使用XLM编码器是否有用。有人可能会认为查询通常是短短语，并且对于Marketplace来说，查询是头部集中的（大部分流量由少量查询主导）。直观上，字符三-gram可能已经足够表示查询了。

我们在此节中使用注意力权重来探索答案。在查询塔中，我们在查询字符三-gram通道和查询XLM编码器通道之间有一个注意力机制。注意力权重（和为1）在公式1中表述。我们提取这些注意力权重并尝试解释模型更关注哪个特征。我们发现XLM编码器的平均注意力权重为0.64，而EmbeddingBagEncoder为0.36。当我们绘制注意力权重随查询长度变化的函数时，我们发现了一个有趣的观察结果。我们发现当查询长度小于5个字符时，模型更关注字符三-gram，但对于较长的查询，XLM编码器权重占主导。

#### 3.7.2 特征重要性

我们认为双塔模型更像是一个"搜索"问题而非"分类"问题。因此，我们提出了一种不同的方法来计算双塔模型的特征重要性，其中我们在保持另一个塔不变的情况下计算一个塔内的特征重要性。我们使用一种称为特征消融的特征重要性技术——其中每个输入特征被一个基线（一个零张量，或其他随机训练点的特征值）替换，并计算输出的变化。

我们在表2中展示了特征重要性结果。如我们所见，XLM在两个塔的所有文本特征中占主导地位。另一方面，字符三-gram特征也提供了不可忽视的增益。这与我们在第3.7节中的研究一致，即字符三-gram特征对短查询贡献最大。在文档塔中，我们使用来自GrokNet[2]的预训练图像嵌入，这是Facebook最先进的图像识别系统。从表2中我们观察到图像模态为模型提供了显著价值。

**表2：特征重要性计算**

| 特征 | 塔 | 重要性 |
|---|---|---|
| XLM | 查询 | 60% |
| 字符三-gram | 查询 | 40% |
| GrokNet | 文档 | 55% |
| 描述XLM | 文档 | 13% |
| 标题XLM | 文档 | 9% |
| 描述字符三-gram | 文档 | 1.5% |
| 标题字符三-gram | 文档 | 1.5% |
| 语言 | 文档 | 1% |
| 国家 | 文档 | 1% |

此外，我们对产品的多张图像使用Deep Sets[29]融合，这为产品（如庭院旧货出售，其中多个商品在一个列表中出售）提供了额外的好处。

---

## 4 系统架构

我们现在描述Que2Search（Facebook面向Marketplace的大规模产品理解系统）部署的系统架构。Que2Search已部署到生产中，旨在对实时创建的产品进行操作。模型推理在称为Predictor[12]的机器集群上计算。Predictor提供了部署模型的功能以及使用一组输入特征调用模型的API。Predictor是一个云服务，可以相应地进行扩展。在调用Predictor时，查询嵌入在P99下1.5毫秒内计算完成，文档嵌入在7毫秒内计算完成。我们使用Unicorn[8]将产品嵌入与其他产品索引一起存储，用于搜索检索和排序。

**图5：Que2Search 的系统架构。**

**文档端模型服务。** 在产品创建和更新时，将异步调用Predictor以生成其产品嵌入，嵌入向量将实时更新到搜索索引中，以准备产品用于检索和排序。换句话说，产品嵌入在搜索查询发出时已经预计算并索引，这使得跨大量产品候选计算查询-产品相似度变得可行。此外，Unicorn进一步对嵌入空间进行聚类和量化[16]以加速计算。我们的基础设施还支持对旧产品进行回填嵌入。在新模型部署后，通常需要不到24小时的时间将嵌入回填到足够多的文档到搜索索引中，以准备进行实验。

**查询时模型服务。** 在推理时，查询流程遵循调用我们的后端NLP（Natural Language Processing，自然语言处理）服务的路径，该服务实时调用Predictor服务。NLP服务将嵌入在内存中缓存一段固定的时间（例如8小时），以加速服务响应并节省Predictor计算。然后查询嵌入被传递到Unicorn搜索引擎后端进行产品检索。Unicorn搜索引擎能够进行带布尔约束的近似最近邻（ANN，Approximate Nearest Neighbor）检索，这意味着我们可以，例如，要求检索与查询最相关/最匹配的产品，同时要求产品在用户指定位置的50英里半径内，以及结合其他传统词项匹配表达式。然后结果被传回多个排序阶段。我们在所有排序阶段使用双塔模型产生的余弦相似度分数作为排序特征。

---

## 5 消融研究

我们进行了离线消融研究以确认每一步的有效性。我们的基线模型[14]使用深度宽架构并在产品的文本属性上应用字符三-gram特征；该模型此前已上线生产并取得了生产收益。在表3中，我们分享了消融研究的结果，比较了新模型和新增的特征。我们看到，添加基于XLM/XLM-R的原始文本特征转换器显著提高了batch recall@1（§3.4）。^4 类似的影响来自图像特征，其中我们对产品的多达五张图像应用Deep Sets融合[29]。文本转换器和图像特征各自提供了独立的价值；同时引入文本转换器和图像特征带来了+9.23%的相对改进。塔模态上的简单注意力融合（公式1）与拼接融合相比，提供了0.15%的改进；在产品的文本字段间使用共享的XLM-R编码器提供了额外0.18%的改进（见表3）。

**表3：特征与参数的消融研究。每种模态独立带来各自的增益。字符三-gram仍然有用。**

| 技术 | Batch Recall@1 |
|---|---|
| Wide & dense char trigram baseline [25] | 0.6788 |
| 字符三-gram + 原始文本特征 | 0.7292 |
| 字符三-gram + 图像特征 | 0.7237 |
| 字符三-gram + 原始文本 + 图像 | 0.7445 |
| 所有特征 + 两个塔上的注意力融合 | 0.7456 |
| 上述 + 标题和描述共享XLM | 0.7459 |

^4 我们在开发的初始阶段使用了batch recall@1，并在表3中报告了详细信息。在后续研究中，我们报告了在人工评分数据集上的ROC AUC改进。

Que2Search双塔模型还在人工评分数据集上大幅提高了ROC AUC。评估结果见表4。图像和文本特征都提供了额外价值，其中原始文本上的XLM编码器贡献了最大增益，ROC AUC绝对提升了2.1%。第二大贡献来自课程训练，ROC AUC绝对提升0.8%（§3.3）。XLM模块与模型其余部分使用不同的学习率（§3.1）进一步帮助提升了质量，相对提升1%。在文档塔中添加分类任务头（§3.1）在ROC AUC中提供了0.61%的相对提升。使用分组特征的梯度混合（§3.7）使模型对某种模态缺失的情况更具鲁棒性，并在ROC AUC中提供了额外0.91%的相对提升。总而言之，Que2Search在现有Facebook产品理解系统的基础上实现了超过5%的绝对ROC AUC改进。

**表4：在人工评分数据集上的评估。每种模态独立带来各自的增益。包含注意力融合和权重共享。**

| 技术 | ROC AUC |
|---|---|
| Wide & dense char trigram baseline [25] | 0.795 |
| 字符三-gram + 原始文本特征 | 0.816 |
| 字符三-gram + 图像特征 | 0.806 |
| 字符三-gram + 原始文本 + 图像 | 0.825 |
| 上述 + 课程训练 | 0.833 |
| 上述 + 可变XLM LR | 0.837 |
| 上述 + 分类头 | 0.842 |
| 上述 + GB-分组特征 | 0.849 |

---

## 6 部署经验

我们将Que2Search部署到Facebook Marketplace上，每天为数亿次产品搜索提供支持。此外，我们通过将其作为Facebook Marketplace的基于嵌入的检索和排序栈的一个有机组成部分进行集成，并通过在精度、召回率、CPU使用率和延迟方面的各种优化，实现了可扩展的部署。在线A/B测试显示，Que2Search带来了超过4%的搜索产品参与度提升，其中约2.8%来自语义检索，1.24%来自排序。表5展示了不同技术如何逐步贡献于搜索产品参与度增益的更详细分解。

**表5：不同技术带来的增量参与度增益的在线A/B测试结果。每种技术的相对改进假设其上面的技术已投入生产。模型使用了表4中提到的技术。**

| 技术 | 对话内参与度 |
|---|---|
| （基线）无基于嵌入的排序 | - |
| （EBR）仅字符三-gram模型 [25] | +3.3% |
| （EBR）Que2Search模型 | +2.80% |
| （排序）余弦相似度排序特征 | +1.24% |

### 6.1 基于嵌入的检索

随着Que2Search的引入，我们能够利用Unicorn对EBR（Embedding-based Retrieval，基于嵌入的检索）[14]的原生支持执行更多的语义检索。更具体地，如第4节所述，我们将查询塔模型部署到NLP服务以实时生成查询嵌入，并将产品塔模型部署到产品创建和更新时的Unicorn索引异步更新路径上。当用户发起产品搜索时，其搜索查询将被转换为一个嵌入向量，使用Unicorn的NN-operator在产品嵌入空间中进行ANN搜索，并且余弦距离在阈值范围内的产品将被检索为结果。

### 6.2 检索没有银弹

需要注意的是，我们将Que2Search部署为整个Marketplace搜索检索的一个有机组成部分，而不是将其用作唯一的搜索检索手段。整个检索过程的其他关键组成部分包括与ANN搜索语义平行的传统基于词元的检索，以及其他可能的约束，如位置、社交[13]和产品类别。这是通过Unicorn的查询语言[8]无缝完成的（即，ANN和基于词元的检索组织在OR之下，约束分组为高级AND）。

根据经验，我们发现将EBR视为解决语义匹配的一个组成部分，而非将其视为解决所有检索问题的银弹，是有益的。以下是一些原因：（1）尽管Que2Search具有位置感知能力，但在ANN搜索中指定明确的位置约束是困难的，而这对于像Facebook Marketplace这样的本地商业搜索引擎至关重要；（2）ANN搜索无法处理用户选择的过滤器，这是任何商业搜索引擎的关键功能；（3）由于性能问题，很难使ANN搜索穷举，而基于词元的检索提供了一种低成本的选择，可以更穷举地覆盖较简单的案例。

### 6.3 精度、召回率、CPU使用率和延迟

在部署Que2Search时，精度、召回率、CPU使用率和延迟是需要优化权衡的维度。对于Unicorn的NN-operator，它们可以转化为两个超参数——$n_{\text{prob}}$ 和 radius。由于Unicorn采用聚类和乘积量化方法[14, 16]，要扫描的聚类数量决定了ANN搜索的穷举程度，这由 $n_{\text{prob}}$ 控制。一般来说，扫描更多的聚类意味着更高的精度但更高的CPU使用率。当扫描一个产品时，将计算其与查询向量的余弦距离。然后，radius参数控制阈值，以决定是否检索该产品。

虽然可以使用在线A/B测试对超参数进行网格搜索，但我们希望尽量减少对用户的影响，尤其是当参数选择不当会降低用户体验时。因此，我们在建立在线A/B测试之前，先在一个黄金查询集上模拟检索，将其缩减为少数几个合理好的候选。黄金集由通过对搜索查询进行分层抽样从Marketplace搜索结果中抓取的搜索结果组成。我们还要求评分员判断结果是否相关。利用这个黄金集，我们可以使用不同的 radius 模拟检索，并通过检查返回结果的数量和相关结果的召回率来确定检索的激进程度。然而，在线A/B测试对于选择正确的参数仍然至关重要，因为除了需要衡量真实的用户参与度外，有时CPU使用率和延迟很难被准确模拟，因为它们与用户行为密切相关。

### 6.4 优化连续翻页

关于用户行为如何影响CPU使用率的一个关键观察与连续翻页有关，这指的是用户不断向下滚动搜索结果页面以触发更多的检索请求，从而使更多产品被展示。显然，如果用户对顶部结果（相当于第一个检索请求提供了高精度）感到满意，我们将节省CPU使用率，因为任何后续请求都不需要被触发。然而，我们也从经验上知道，用户对第一个请求的延迟很敏感——如果一段时间内没有结果显示，整个搜索可能会被放弃。这意味着我们不能为了追求精度而使第一个请求过于昂贵。总而言之，连续翻页可以用作一个杠杆来权衡精度、召回率和CPU使用率，但受限于延迟等约束。

基于这一观察，我们对第一个请求启用了EBR，采用保守的 $n_{\text{prob}}$ 和 radius 设置，以便保持较高的延迟标准，并使后续请求中的EBR更加昂贵以进一步优化参与度。基于在线A/B测试结果，我们发现这种动态ANN配置与全程保持相同配置相比，产生了相同的参与度收益，但显著减少了CPU使用周期和延迟。事实上，如果没有这一优化，我们将无法因延迟问题在生产中启用Que2Search。

### 6.5 搜索排序

基于表4，我们知道来自双塔模型的查询-产品余弦相似度分数是人工评分数据集的一个良好预测指标。鉴于该相似度分数在检索阶段即可获得，我们也将该分数暴露给搜索排序，以进一步利用Que2Search。

Facebook Marketplace的搜索排序遵循两阶段方案，其中一个轻量级的GBDT（Gradient Boosting Decision Tree，梯度提升决策树）排序器在单个索引服务器实例上运行，从每个分片中选择顶部结果，然后使用一个更昂贵的类DLRM（Deep Learning Recommendation Model，深度学习推荐模型）[20]模型进一步选择最佳结果。为了在排序中利用Que2Search，我们简单地将该分数作为排序特征添加到两个阶段中。我们检查了特征重要性，发现余弦距离特征在两个阶段中都是最重要的特征，在线A/B测试显示，这种简单的特征添加带来了统计显著的参与度改进，如表5所示。

### 6.6 失败教训

为获得更好的搜索质量而部署Que2Search是一个试错的过程，我们遇到了许多乍看起来可能反直觉的现象。

**精度很重要：** 精度在各种机器学习任务中显然非常重要，但为什么在搜索检索设置中不能仅使用召回率作为标准则不那么直观。换句话说，我们发现放宽ANN搜索的阈值可能导致更差的结果，尽管从理论上讲，这不应影响检索到的良好结果总数。进一步分析表明，这是由于检索模型和排序模型之间的不一致造成的——我们的排序模型无法处理通过更宽松阈值检索到的更嘈杂的结果。改善多阶段模型之间的一致性本身就是一个研究课题[5, 28]，我们在此不再详细展开。最终，我们只是通过在线和离线参数搜索来决定最佳阈值。

**相关性不够：** 改善检索模型和排序模型之间一致性的一种朴素方法是直接使用双塔模型的余弦相似度作为排序分数。这种方法的一个好处是它保证了因更宽松阈值而检索到的所有结果都可以被排在底部。实际上，我们通过将基于GBDT的第一阶段模型替换为余弦相似度来尝试了这个想法，但尽管我们发现以NDCG（Normalized Discounted Cumulative Gain，归一化折损累计增益）衡量的相关性有显著改善，它却大幅降低了在线参与度。这可能是因为双塔模型被训练为优化查询-产品相似度而非优化参与度，而GBDT模型则更注重参与度。虽然可以向双塔模型添加参与度特征，但这并非一个简单的扩展，因为我们的许多重要参与度特征是基于搜索者和产品信号的手工调优稠密特征，这违反了双塔模型的独立性/晚期融合属性。

---

## 7 结论

我们提出了构建一个名为Que2Search的全面查询与产品理解系统的方法。我们提出了关于多任务和多模态训练以学习查询和产品表示的创新思路。通过Que2Search，我们在Facebook现有最先进的产品理解系统基础上实现了超过5%的绝对离线相关性改进和超过4%的在线参与度增益。我们分享了针对搜索用例调优和部署基于BERT的查询理解模型的经验，并实现了第99百分位数下1.5毫秒的推理时间。我们分享了我们的部署故事，以及关于部署步骤的实用建议，以及如何将Que2Search组件集成到搜索语义检索和排序栈中。

---

## 8 致谢

作者感谢Shaoliang Nie、Liang Tan、Rao Bayyana、Hamed Firooz、ZJ Yin、Ashish Gandhe、Yinzhe Yu以及其他贡献、支持和与我们合作的人。

---

## 参考文献

[1] Ricardo Baeza-Yates and Berthier Ribeiro-Neto. 2011. Modern Information Retrieval: The Concepts and Technology behind Search.

[2] Sean Bell, Yiqun Liu, Sami Alsheikh, Yina Tang, Edward Pizzi, M. Henning, Karun Singh, Omkar Parkhi, and Fedor Borisyuk. 2020. GrokNet: Unified Computer Vision Model Trunk and Embeddings For Commerce. In KDD.

[3] Jane Bromley, Isabelle Guyon, Yann Lecun, Eduard Säckinger, and Roopak Shah. 1994. Signature Verification using a "Siamese" Time Delay Neural Network. In NeurIPS.

[4] Yen-Chun Chen, Linjie Li, Licheng Yu, Ahmed El Kholy, Faisal Ahmed, Zhe Gan, Yu Cheng, and Jingjing Liu. 2020. UNITER: UNiversal Image-TExt Representation Learning. In ECCV.

[5] Zhihong Chen, Rong Xiao, Chenliang Li, Gangfeng Ye, Haochuan Sun, and Hongbo Deng. 2020. ESAM: Discriminative Domain Adaptation with Non-Displayed Items to Improve Long-Tail Performance. arXiv preprint arXiv:2005.10545 (2020).

[6] Heng-Tze Cheng, Levent Koc, Jeremiah Harmsen, Tal Shaked, Tushar Chandra, Hrishi Aradhye, Glen Anderson, Greg Corrado, Wei Chai, Mustafa Ispir, Rohan Anil, Zakaria Haque, Lichan Hong, Vihan Jain, Xiaobing Liu, and Hemal Shah. 2016. Wide & Deep Learning for Recommender Systems. In RecSys.

[7] Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer, and Veselin Stoyanov. 2020. Unsupervised Cross-lingual Representation Learning at Scale. In ACL.

[8] Michael Curtiss, Iain Becker, Tudor Bosman, Sergey Doroshenko, Lucian Grijincu, Tom Jackson, Sandhya Kunnatur, Soren Lassen, Philip Pronin, Sriram Sankar, Guanghao Shen, Gintaras Woss, Chao Yang, and Ning Zhang. 2013. Unicorn: A System for Searching the Social Graph. VLDB (2013).

[9] Jiankang Deng, Jia Guo, Niannan Xue, and Stefanos Zafeiriou. 2019. ArcFace: Additive Angular Margin Loss for Deep Face Recognition. arXiv:cs.CV/1801.07698

[10] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. In ACL.

[11] Weiwei Guo, Xiaowei Liu, Sida Wang, Huiji Gao, Ananth Sankar, Zimeng Yang, Qi Guo, Liang Zhang, Bo Long, Bee-Chung Chen, and Deepak Agarwal. 2020. DeText: A Deep Text Ranking Framework with BERT. In CIKM.

[12] K. Hazelwood, S. Bird, D. Brooks, S. Chintala, U. Diril, D. Dzhulgakov, M. Fawzy, B. Jia, Y. Jia, A. Kalro, J. Law, K. Lee, J. Lu, P. Noordhuis, M. Smelyanskiy, L. Xiong, and X. Wang. 2018. Applied Machine Learning at Facebook: A Datacenter Infrastructure Perspective. In HPCA.

[13] Yunzhong He, Wenyuan Li, Liang-Wei Chen, Gabriel Forgues, Xunlong Gui, Sui Liang, and Bo Hou. 2020. A Social Search Model for Large Scale Social Networks. arXiv:cs.IR/2005.04356

[14] Jui-Ting Huang, Ashish Sharma, Shuying Sun, Li Xia, David Zhang, Philip Pronin, Janani Padmanabhan, Giuseppe Ottaviano, and Linjun Yang. 2020. Embedding-Based Retrieval in Facebook Search. In KDD.

[15] Zhicheng Huang, Zhaoyang Zeng, Bei Liu, Dongmei Fu, and Jianlong Fu. 2020. Pixel-BERT: Aligning Image Pixels with Text by Deep Multi-Modal Transformers. CoRR (2020).

[16] J. Johnson, M. Douze, and H. Jégou. 2019. Billion-scale similarity search with GPUs. IEEE Transactions on Big Data (2019).

[17] Guillaume Lample and Alexis Conneau. 2019. Cross-lingual Language Model Pretraining. arXiv:cs.CL/1901.07291

[18] Jiasen Lu, Dhruv Batra, Devi Parikh, and Stefan Lee. 2019. ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations for Vision-and-Language Tasks. In NeurIPS.

[19] Dhruv Mahajan, Ross B. Girshick, Vignesh Ramanathan, Kaiming He, Manohar Paluri, Yixuan Li, Ashwin Bharambe, and Laurens van der Maaten. 2018. Exploring the Limits of Weakly Supervised Pretraining. In ECCV.

[20] Maxim Naumov, Dheevatsa Mudigere, Hao-Jun Michael Shi, Jianyu Huang, Narayanan Sundaraman, Jongsoo Park, Xiaodong Wang, Udit Gupta, Carole-Jean Wu, Alisson G. Azzolini, Dmytro Dzhulgakov, Andrey Mallevich, Ilia Cherniavskii, Yinghai Lu, Raghuraman Krishnamoorthi, Ansha Yu, Volodymyr Kondratenko, Stephanie Pereira, Xianjie Chen, Wenlin Chen, Vijay Rao, Bill Jia, Liang Xiong, and Misha Smelyanskiy. 2019. Deep Learning Recommendation Model for Personalization and Recommendation Systems. arXiv:cs.IR/1906.00091

[21] Pandu Nayak. 2019. Understanding searches better than ever before. https://blog.google/products/search/search-language-understanding-bert/

[22] Emma Ning. 2019. Microsoft open sources breakthrough optimizations for transformer inference on GPU and CPU. https://cloudblogs.microsoft.com/opensource/2020/01/21/microsoft-onnx-open-source-optimizations-transformer-inference-gpu-cpu/

[23] Xichuan Niu, Bofang Li, Chenliang Li, Rong Xiao, Haochuan Sun, Hongbo Deng, and Zhenzhong Chen. 2020. A Dual Heterogeneous Graph Attention Network to Improve Long-Tail Performance for Shop Search in E-Commerce. In KDD.

[24] Adam Paszke, Sam Gross, Soumith Chintala, Gregory Chanan, Edward Yang, Zachary DeVito, Zeming Lin, Alban Desmaison, Luca Antiga, and Adam Lerer. 2017. Automatic differentiation in PyTorch. (2017).

[25] Yina Tang, Fedor Borisyuk, Siddarth Malreddy, Yixuan Li, Yiqun Liu, and Sergey Kirshner. 2019. MSURU: Large Scale E-commerce Image Classification with Weakly Supervised Search Data. In KDD.

[26] Weiyao Wang, Du Tran, and Matt Feiszli. 2019. What Makes Training Multi-Modal Networks Hard? CVPR (2019).

[27] Ji Yang, Xinyang Yi, Derek Zhiyuan Cheng, Lichan Hong, Yang Li, Simon Xiaoming Wang, Taibai Xu, and Ed H. Chi. 2020. Mixed Negative Sampling for Learning Two-Tower Neural Networks in Recommendations. In WWW.

[28] Bowen Yuan, Jui-Yang Hsia, Meng-Yuan Yang, Hong Zhu, Chih-Yao Chang, Zhenhua Dong, and Chih-Jen Lin. 2019. Improving ad click prediction by considering non-displayed events. In KDD.

[29] Manzil Zaheer, Satwik Kottur, Siamak Ravanbakhsh, Barnabás Póczos, Ruslan Salakhutdinov, and Alexander J. Smola. 2017. Deep Sets. CoRR abs/1703.06114 (2017). arXiv:1703.06114 http://arxiv.org/abs/1703.06114
