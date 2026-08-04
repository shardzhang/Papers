# 端到端用户行为检索在点击率预测模型中的应用

> Qiwei Chen, Changhua Pei, Shanshan Lv, Chao Li, Junfeng Ge, Wenwu Ou | Alibaba Group

本文介绍了端到端用户行为检索在点击率预测模型中的应用。核心内容：

- 提出ETA（End-to-end Target Attention，端到端目标注意力）模型：受Reformer启发，采用SimHash局部敏感哈希生成行为item指纹，将检索复杂度从 $O(L \times B \times d)$ 次乘法降至 $O(L \times B)$ 次汉明距离计算，实现长期用户行为序列的端到端训练与实时检索
- 揭示两阶段方法（如SIM、UBR4CTR）中检索目标与主CTR模型之间的信息鸿沟；ETA通过SimHash指纹与CTR嵌入同步更新，使检索部分无需预训练或离线倒排索引即可与主模型联合学习
- 在公开淘宝数据集与自建工业数据集（1420亿实例）上进行离线实验，并在大规模真实电商系统部署在线A/B测试

关键发现：

- 离线实验：工业数据集上ETA+timeinfo较SIM(hard)+timeinfo的AUC提升0.35%，淘宝数据集上提升0.38%；推理时间约19ms，与SIM相当
- 在线A/B测试：ETA+timeinfo在CTR上提升6.33%、GMV（Gross Merchandise Value，商品交易总额）上提升9.7%，较两阶段SOTA模型额外获得3.1%的GMV提升
- 消融实验表明，ETA基础版（v0，SimHash检索）相比直接对1024长度序列执行多头注意力（v4）牺牲约0.1%的AUC，但推理时间降低46%（35ms降至19ms）；相比内积检索（v3）AUC仅损失0.07%，推理时间远低于内积检索（19ms vs 32ms）

---

## 摘要

点击率（CTR，Click-Through Rate）预测是推荐系统（RS，Recommender System）中的核心任务之一。它预测每个用户-item对的个性化点击概率。最近，研究人员发现，考虑用户行为序列（尤其是长期用户行为序列）可以大幅提升CTR模型的性能。某电商网站的报告显示，23%的用户在过去5个月内的点击次数超过1000次。尽管已有大量工作致力于对序列用户行为进行建模，但由于现实世界系统中严格的推理时间约束，很少有工作能够处理长期用户行为序列。为了突破性能极限，研究者提出了两阶段方法。在第一阶段，设计一个辅助任务来从长期用户行为序列中检索top-k相似item。在第二阶段，在候选item与第一阶段选出的 $k$ 个item之间执行经典的注意力机制。然而，检索阶段与主CTR任务之间存在信息鸿沟。这种目标差异会大大削弱长期用户行为序列的性能增益。本文受Reformer启发，提出了一种名为ETA（End-to-end Target Attention，端到端目标注意力）的局部敏感哈希（LSH，Locality-sensitive Hashing）方法，该方法能大幅降低训练和推理成本，使得端到端训练长期用户行为序列成为可能。离线和在线实验均证实了我们模型的有效性。我们将ETA部署到大规模真实电商系统中，与两阶段长用户序列CTR模型相比，在GMV（Gross Merchandise Value，商品交易总额）上额外提升了3.1%。

\*两位作者对本文贡献相等。

允许以个人或课堂使用为目的制作或分发本作品的数字或硬拷贝，无需付费，前提是这些拷贝不得用于盈利或商业目的，且每份拷贝首页包含此声明和完整引用。本作品版权归ACM其他所有者所有，必须予以尊重。允许带有致谢的摘要转载。如需以其他方式复制、重新发布、上传到服务器或分发给列表，需事先获得特定许可和/或支付费用。请向permissions@acm.org申请许可。

Conference'17, 2017年7月, 美国华盛顿特区
© 2021 计算机协会（ACM）
ACM ISBN 978-x-xxxx-xxxx-x/YY/MM... $15.00$
https://doi.org/10.1145/nnnnnnn.nnnnnnn

## 关键词

recommendation, behavior sequence, real-time retrieval

## ACM引用格式

Qiwei Chen, Changhua Pei, Shanshan Lv, Chao Li, Junfeng Ge, and Wenwu Ou. 2021. End-to-End User Behavior Retrieval in Click-Through Rate Prediction Model. In Proceedings of ACM Conference (Conference'17). ACM, New York, NY, USA, 11 pages. https://doi.org/10.1145/nnnnnnn.nnnnnnn

## 1 引言

推荐系统（RS，Recommender System）被广泛用于解决信息过载问题。在RS使用的所有深度学习模型中，点击率（CTR，Click-Through Rate）预测模型是最重要的模型之一。工业界和学术界都非常关注提升CTR模型的AUC（Area Under the ROC Curve，ROC曲线下面积），以提高RS的在线性能。在过去十年中，CTR模型的性能得到了极大提升。其中一个重要的里程碑是引入了用户行为序列，特别是长期用户行为序列[20–22, 35, 36]。根据[24]的报告，某电商网站23%的用户在过去5个月内的点击次数超过1000次。如何有效利用海量且信息丰富的用户行为变得越来越重要，这也是本文的目标。

研究者提出了多种方法来对序列用户行为数据进行建模。早期方法，如求和/平均池化方法、基于RNN（Recurrent Neural Network，循环神经网络）的方法[4, 5, 11]、基于CNN（Convolutional Neural Network，卷积神经网络）的方法[14, 27]和基于自注意力的方法[12, 28]，将不同长度的用户行为序列编码为固定维度的隐藏向量。然而，在评分不同候选item时，它们无法捕捉用户的动态局部兴趣。这些方法通过对所有用户历史行为进行编码，也引入了噪声。为了克服全局池化方法的缺陷，DIN（Deep Interest Network，深度兴趣网络）[36]被提出，通过目标注意力机制根据不同的候选item生成不同的用户序列表示，其中目标候选item作为查询 $\boldsymbol{Q}$，序列中的每个item作为键 $\boldsymbol{K}$ 和值 $\boldsymbol{V}$。然而，由于昂贵的计算和存储资源，DIN只使用了最近的50个行为进行目标注意力计算，这忽略了长期用户行为序列中的丰富信息，显然不是最优的。

最近，SIM（Search-based User Interest Modeling，基于搜索的用户兴趣建模）[21]和UBR4CTR（User Behavior Retrieval for Click-Through Rate Prediction，面向点击率预测的用户行为检索）[22]等方法被提出用于从更长的用户行为序列中捕捉用户动态兴趣，并成为最先进的（SOTA，State-of-the-Art）方法。这些方法以两阶段方式运行。在第一阶段，设计一个辅助任务从长期用户行为序列中检索top-k相似item，从而预先准备好top-k相似item。在第二阶段，在目标item与第一阶段选出的 $k$ 个item之间执行目标注意力机制。然而，检索阶段使用的信息与主CTR模型存在差异或过时。例如，UBR4CTR[22]和SIM[21]使用类别等属性从用户行为序列中选择与目标候选item共享相同属性的item，这与CTR模型的目标不一致。SIM[21]还尝试基于预训练嵌入构建离线倒排索引。在训练和推理期间，模型可以搜索top-k"相似"item。但大多数CTR模型采用在线学习范式，嵌入向量持续更新。因此，离线倒排索引中的预训练嵌入相比于在线CTR模型中的嵌入已经过时。无论是目标差异还是过时的检索向量，都将阻碍长期用户行为序列被充分利用。

在本文中，我们提出了一种名为ETA的方法，实现了端到端的长期用户行为检索，以缓解上述CTR预测任务中的信息鸿沟（即目标差异和嵌入过时）。我们使用SimHash为用户行为序列中的每个item生成一个指纹。然后利用汉明距离帮助选择Top-k item进行目标注意力计算。我们的方法将检索复杂度从 $O(L \times B \times d)$ 次乘法降低到 $O(L \times B)$ 次汉明距离计算，其中 $L$ 是行为序列的长度， $B$ 是每次推荐中需要由CTR模型评分的候选item数量， $d$ 是item嵌入的维度。复杂度的降低帮助我们移除了离线辅助模型，并在训练和服务过程中实现了实时检索。与SOTA模型相比，这大幅提升了排序效果。本文的贡献可归纳为以下三个方面：

*   我们提出了一种用于CTR预测任务的端到端目标注意力方法，称为ETA。据我们所知，ETA是第一个以端到端方式利用CTR模型对长期用户行为序列进行建模的工作。
*   离线和在线A/B测试均表明，与SOTA模型相比，ETA取得了显著的性能提升。在将ETA部署到大规模真实电商平台后，与两阶段CTR模型相比，我们在GMV上额外获得了3.1%的提升。
*   我们进行了全面的消融研究，揭示了在推理时间约束下更好地对序列用户行为进行建模的实践经验。
*   我们的方法还可以扩展到其他需要处理超长序列的场景，例如长时间序列时间序列预测模型。

## 2 相关工作

CTR预测任务是推荐系统、在线广告和信息检索中的关键任务之一。CTR模型预测用户点击某个目标item的概率。输出的概率可作为下游排序任务的排序分数。CTR模型的准确性会极大地影响在线系统的线上性能。例如，在我们的在线RS中，CTR模型AUC提升0.1%就能带来数百万的真实点击和收入。大量工作致力于以不同方式提升CTR模型的准确性，可分为三类：特征交互、用户行为序列和长期用户行为序列。

**特征交互：** 特征交互的直觉是在特征空间中记忆与标签共现的模式。例如，特征 AND(user_installed_app = netflix, impression_app = pandora) 可以更好地捕捉用户点击或未点击某个推荐App的模式。已有系列工作致力于更有效地建模特征交互。代表性工作有FM[25]、FFM[19]、GBDT+LR[9]、Wide&Deep[3]、FNN[33]、AFM[30]、DeepCross[29]、DeepFM[8]、PNN[23]和xDeepFM[17]。任意两个模型之间存在各种差异，例如是否使用深度学习技术、权重和特征的嵌入是否共享、以及是否需要进行特征工程。

**用户行为序列：** 用户行为序列对每个用户高度个性化，并包含时空用户兴趣信息。将序列用户行为引入CTR模型是一个重要的里程碑。YouTube[6]在其模型中使用观看视频序列和搜索token来捕捉用户兴趣。为了更好地从用户行为序列中提取用户兴趣，研究者提出了多种模型，包括CNN[27, 32]、RNN[10, 35]、Attention[7, 26]和Capsule Network[15]。然而，上述模型学到的用户兴趣向量对于某个特定用户是全局的。DIN[36]提出了一种基于注意力的方法（称为目标注意力），以捕捉某个用户面对不同目标item时的多样化局部兴趣。

**长期用户行为序列：** 尽管具有捕捉多样化用户兴趣的强大能力，目标注意力的计算成本很高。严格的在线推理时间限制阻止了类似DIN的模型使用更长的用户序列。MIMN（Multi-channel user Interest Memory Network，多通道用户兴趣记忆网络）[20]通过将用户兴趣建模与CTR任务的其余部分解耦，可以处理长期用户行为序列。每当观察到新行为时，用户兴趣向量以异步方式离线更新。由于离线没有推理时间限制，MIMN理论上可以对任意长度的序列进行建模。然而，MIMN无法为不同的目标item学习不同的用户兴趣向量。SIM[21]和UBR4CTR[22]在CTR任务上超越了MIMN，成为SOTA模型。SIM和UBR4CTR均采用两阶段架构来建模长期用户行为序列。在第一阶段，设计一个辅助任务来从长期用户行为序列中检索top-k相似item。在第二阶段，在目标item与第一阶段选出的 $k$ 个item之间执行目标注意力。

除上述CTR预测任务的相关工作外，还有大量工作致力于提高Transformer的效率和有效性。Reformer[13]和Informer[37]是最相关的两项工作。然而，它们只关注经典Transformer的优化，由于大规模在线推荐系统的严格推理时间约束，无法直接用于CTR预测任务。

## 3 预备知识

在本节中，我们首先给出CTR预测任务的公式化表述。然后介绍如何通过SimHash机制生成 $d$ 维嵌入向量的指纹。

### 3.1 CTR预测任务的公式化表述

CTR预测任务广泛部署于在线广告、推荐系统和信息检索中。其目标是解决以下问题：

给定一次曝光 $j$ （其中一个item展示给一个用户），使用特征向量 $x_j$ 预测用户点击（标记为 $y_j$）的概率。

$$
p_j = P(y_j = 1 \mid x_j; \theta); \quad j \in I. \qquad (1)
$$

CTR任务通常被建模为一个二分类问题。对于每次曝光 $j \in I$，根据item是否被点击记录一个二值标签 $y_j$。然后以监督方式训练CTR模型以最小化交叉熵损失，如公式2所示，其中 $N$ 是曝光次数。 $x_j$ 和 $y_j$ 分别是曝光 $j$ 的特征向量和标签。 $\theta$ 代表CTR模型的可训练参数。为便于表述，我们在表1中列出了本文使用的符号。

$$
L_{CTR}(\theta) = -\frac{1}{N} \sum_{j=1}^{N} \left[ y_j \ast \log(p_j) + (1 - y_j) \ast \log(1 - p_j) \right]. \qquad (2)
$$

**表1：本文使用的符号。**

| 符号 | 描述 |
|---|---|
| $I$ | 曝光集合。 |
| $p_j$ | 某次曝光 $j$ 的点击概率。 |
| $y_j$ | 曝光 $j$ 上的点击标签。 |
| $x_j$ | 曝光 $j$ 上的特征向量。 |
| $\theta$ | CTR模型的参数。 |
| $d$ | item嵌入的维度。 |
| $\boldsymbol{e} \in \mathbb{R}^{d \times 1}$ | item嵌入向量。 |
| $\boldsymbol{h} \in \mathbb{R}$ | 嵌入向量 $\boldsymbol{e}$ 的哈希指纹。 |
| $\boldsymbol{sig}_i$ | 由第 $i$ 个哈希函数生成的位向量。 |
| $m$ | 哈希指纹 $\boldsymbol{h}$ 的位长度。 |
| $H_{lu}$ | 长期用户行为序列。 |
| $H_{su}$ | 短期用户行为序列。 |
| $x_u$ | 用户画像的原始特征。 |
| $x_c$ | 上下文信息的原始特征。 |
| $x_t$ | 目标item的原始特征。 |
| $L$ | 长期用户行为序列的长度。 |
| $B$ | 每次用户请求中待预测的候选item数量。 |
| $\boldsymbol{e}_u, \boldsymbol{e}_c, \boldsymbol{e}_t$ | 用户画像、上下文和目标item的嵌入向量。 |
| $\boldsymbol{E}_s \in \mathbb{R}^{L \times d}$ | 长期用户行为序列的嵌入矩阵。 |
| $\boldsymbol{E}_t \in \mathbb{R}^{1 \times d}$ | 目标item的嵌入矩阵。 |
| $f(\cdot)$ | 两个嵌入向量的相似度函数。 |

**算法1：SimHash算法的伪代码。**

**输入：** 一个 $d$ 维嵌入向量 $\boldsymbol{e}_k \in \mathbb{R}^{1 \times d}$；一个固定的随机哈希矩阵 $\boldsymbol{H} \in \mathbb{R}^{d \times m}$，其每一列可视为一个哈希函数。
**输出：** $\boldsymbol{e}_k$ 的二进制签名向量 $\boldsymbol{sig}_k \in \mathbb{R}^{1 \times m}$。

$$
\begin{aligned}
&\textbf{for } i \leftarrow 0 \textbf{ to } m-1 \textbf{ do} \\
&\quad \boldsymbol{sig}_k[i] = \sum_{j=1}^{d} \mathrm{sgn}(\boldsymbol{e}_k[j] \ast \boldsymbol{H}[j][i]) \\
&\quad \textbf{if } \boldsymbol{sig}_k[i] > 0 \textbf{ then} \\
&\quad\quad \boldsymbol{sig}_k[i] = 1 \\
&\quad \textbf{else} \\
&\quad\quad \boldsymbol{sig}_k[i] = 0 \\
&\quad \textbf{end if} \\
&\textbf{end for} \\
&\textbf{return } \boldsymbol{sig}_k
\end{aligned}
$$

### 3.2 SimHash

**图1：两个向量 $\boldsymbol{x}$ 和 $\boldsymbol{y}$ 的SimHash局部敏感示意图。** 详细实现见算法1。每个 $d$ 维向量被转换为一个 $m$ 长度的签名向量。这里的每次随机旋转可视为一个"哈希函数"。旋转通过乘以一个随机哈希列向量 $\boldsymbol{H}^{(i)}$ 来实现（算法1第2行）。随机旋转后，球面上的点被投影到带符号的轴上（算法1第3-7行）。这里我们使用4个哈希函数和两个投影轴，将每个向量映射为4个二进制位。可以观察到，只有彼此接近的向量才能共享更多相同的0/1位，如下图底部所示。

SimHash算法由[2]首次提出，其著名应用之一是[18]，该应用通过基于SimHash的指纹检测重复网页。SimHash函数以item的嵌入向量为输入，生成其二进制指纹。算法1展示了一种可能的SimHash实现的伪代码。SimHash满足局部敏感性质：如果输入向量彼此相似，则SimHash的输出也相似，如图1所示。图1中的每次随机旋转可视为一个"哈希函数"。旋转通过将输入嵌入向量与一个随机投影列向量 $\boldsymbol{H}^{(i)}$ 相乘来实现，如算法1第2行所示。随机旋转后，球面上的点被投影到带符号的轴上（算法1第3-7行）。在图1中，我们使用4个哈希函数和两个投影轴，将每个向量映射为包含4个元素的签名向量。签名向量中的每个元素为1或0。该向量可进一步解码为一个整数，以节省存储成本并加速后续的汉明距离计算。从图1可以观察到，相近的嵌入向量有很大概率获得相同的哈希签名（见图1底部与图1上部的对比）。这一观察结果就是所谓的"局部敏感"性质。利用局部敏感性质，嵌入向量之间的相似度可以用哈希签名之间的相似度代替。换句话说，两个向量之间的内积可以用汉明距离代替。值得注意的是，SimHash算法对每次旋转的"哈希函数"的选择并不敏感。任何固定的随机哈希向量都足够了（见算法1中的 $\boldsymbol{H}^{(i)}$）。它易于实现，并且可以轻松应用于批量嵌入向量。

## 4 模型

在本节中，我们首先介绍ETA（端到端目标注意力，End-to-end Target Attention）模型的详细架构。然后介绍ETA模型的不同子模块。最后，介绍部署ETA的实践经验。

**图2：我们的ETA（端到端目标注意力）模型示意图。** $\boldsymbol{e}_{k+1} \in \mathbb{R}^{d \times 1}$ 和 $\boldsymbol{e}_t \in \mathbb{R}^{d \times 1}$ 代表行为item $k+1$ 和目标item $t$ 的嵌入向量。 $d$ 是item嵌入的维度。 $\boldsymbol{h}_{k+1}$ 和 $\boldsymbol{h}_t$ 是由SimHash函数生成的item $k+1$ 和 $t$ 的指纹。一个 $d$ 维嵌入 $\boldsymbol{e}_{k+1}$ 可以被哈希为一个 $m$ 位整数。注意，为清晰起见，图中省略了嵌入向量 $\boldsymbol{e}_{k+1}$ 和 $\boldsymbol{e}_t$ 在SimHash之前的投影。其他符号可在表1中找到。

### 4.1 模型概述

如图2所示，我们的模型以用户/item侧特征作为输入，并输出特定用户-item对的点击概率。 $H_{lu}$、 $H_{su}$、 $x_u$、 $x_t$ 和 $x_c$ 是原始输入特征。 $\theta$ 代表可训练参数。利用这些特征，我们使用长期兴趣提取单元（第4.4节）、多头目标注意力（第4.3节）和嵌入层（第4.2节）分别将 $H_{lu}$、 $H_{su}$、 $x_u$、 $x_t$ 和 $x_c$ 转换为隐藏向量。然后将这些隐藏向量拼接在一起，送入MLP（Multi-Layer Perceptron，多层感知机）部分。在MLP的最后一层，使用sigmoid函数将隐藏向量映射为一个标量 $p(y_j \mid H_{lu}, H_{su}, x_u, x_t, x_c; \theta)$，该标量代表特定用户-item对的点击概率。此概率可用作下游任务的排序分数。

### 4.2 嵌入层

针对不同类型的特征，我们采用不同的嵌入技术。原始输入特征主要分为两类：类别特征和数值特征。在我们的模型中，对类别特征使用独热编码。对于数值特征，我们首先将特征划分为不同的数值分桶。然后应用独热编码来标识不同的分桶，这与[16]采用的方式类似。注意，由于存在数十亿的item ID，独热编码向量可能极度稀疏。因此，我们将所有独热嵌入向量映射为低维隐藏向量，以减少参数量。我们用 $\boldsymbol{e}_i \in \mathbb{R}^{d \times 1}$ 表示item $i$ 的嵌入向量。然后将所有用户行为item的嵌入向量打包成一个矩阵 $\boldsymbol{E}_s \in \mathbb{R}^{L \times d}$，如公式3所示。 $L$ 是用户行为序列的长度， $d$ 是嵌入大小。

$$
\boldsymbol{E}_s = \begin{bmatrix} \boldsymbol{e}_1^{\top} \\ \boldsymbol{e}_2^{\top} \\ \vdots \\ \boldsymbol{e}_L^{\top} \end{bmatrix}. \qquad (3)
$$

### 4.3 多头目标注意力

多头注意力由[28]首次提出，并广泛用于CTR预测任务[21, 22, 30, 31, 34]。在CTR预测任务中，目标item充当查询（$\boldsymbol{Q}$），用户行为序列中的每个item充当键（$\boldsymbol{K}$）和值（$\boldsymbol{V}$）。我们将这种多头注意力结构称为多头目标注意力，简称为TA（Target Attention，目标注意力）。TA的计算如公式4所示。TA的主要部分是点积注意力，如公式5所示。点积注意力包含两个步骤。首先，根据行为item和目标item的嵌入矩阵 $\boldsymbol{Q}$ 和 $\boldsymbol{K}$，计算每个行为item与目标item之间的相似度。其次，将归一化的相似度作为注意力权重，计算所有行为item的加权求和嵌入，其嵌入矩阵表示为 $\boldsymbol{V}$。

$$
\mathrm{TA}(\boldsymbol{E}_t, \boldsymbol{E}_s) = \mathrm{Concat}(head_1, \ldots, head_h)\boldsymbol{W}^O, \quad \text{where } head_i = \mathrm{Attention}(\boldsymbol{E}_t\boldsymbol{W}_i^Q, \boldsymbol{E}_s\boldsymbol{W}_i^K, \boldsymbol{E}_s\boldsymbol{W}_i^V), \qquad (4)
$$

$$
\mathrm{Attention}(\boldsymbol{Q}, \boldsymbol{K}, \boldsymbol{V}) = \mathrm{softmax}\left(\frac{\boldsymbol{Q}\boldsymbol{K}^T}{\sqrt{d_k}}\right)\boldsymbol{V}, \qquad (5)
$$

其中 $\boldsymbol{E}_t \in \mathbb{R}^{1 \times d}$ 和 $\boldsymbol{E}_s \in \mathbb{R}^{L \times d}$ 分别是目标item和行为序列的输入嵌入矩阵。 $L$ 是序列长度， $d$ 是每个行为item隐藏向量的嵌入大小。为清晰起见，我们只选择一个样本而非一批样本。矩阵 $\boldsymbol{Q}$、 $\boldsymbol{K}$、 $\boldsymbol{V}$ 分别代表查询、键和值。 $d_k$、 $d_q$、 $d_v$ 分别是 $\boldsymbol{K}$、 $\boldsymbol{Q}$、 $\boldsymbol{V}$ 中每行向量的嵌入大小。 $\sqrt{d_k}$ 用于避免内积值过大。softmax函数用于将内积值转换为值向量 $\boldsymbol{V}$ 的加权权重。 $\boldsymbol{W}_i^Q \in \mathbb{R}^{d \times d_k}$， $\boldsymbol{W}_i^K \in \mathbb{R}^{d \times d_k}$， $\boldsymbol{W}_i^V \in \mathbb{R}^{d \times d_v}$。 $\boldsymbol{W}^O \in \mathbb{R}^{hd_v \times d}$ 是投影矩阵。 $h$ 是头数。

### 4.4 长期兴趣提取单元

这部分是ETA模型的主要贡献。它将用户行为序列的编码长度从数十扩展到数千或更长，以捕捉长期用户兴趣。如前所述，多头目标注意力的复杂度为 $O(L \times B \times d)$，其中 $L$ 是用户序列的长度， $B$ 是候选item的数量， $d$ 是表示维度。在大规模在线系统中， $B$ 接近1000， $d$ 接近128。因此，直接对数千个长期用户行为进行多头目标注意力计算是不可行的。

根据公式5，softmax由最大的元素主导，因此对于每个查询，我们只需要关注与查询最接近的键，这一点也得到[13, 21, 22]的证实。因此，我们可以先从行为序列中检索top-k个item，然后对这 $k$ 个行为执行多头目标注意力。然而，一个好的检索算法应满足两个约束：1）检索部分的目标应与整个CTR模型保持一致。只有这样，检索到的top-k item才能对CTR模型贡献最大。2）检索时间应满足严格的推理时间限制，以确保算法能够应用于每秒服务数百万请求的真实世界系统。我们在表2中比较了不同的检索算法。SIM[21]和UBR4CTR[22]构建离线倒排索引以实现在训练和推理期间的快速搜索。然而，它们用于构建索引的输入是item的属性信息（例如类别）或预训练嵌入，这与CTR模型中使用的嵌入不同。这种差异违反了上述约束1），可能导致性能下降。如果我们直接使用CTR模型中的嵌入并通过内积搜索 $k$ 近邻，则需要 $O(L \times B \times d)$ 次乘法，推理时间将大幅增加。 $d$ 是嵌入向量的维度。 $L$ 和 $B$ 分别是行为item和目标item的数量。这将违反上述约束2），无法在线部署。我们的ETA使用SimHash将两个向量的内积计算转换为汉明距离计算，如图2所示。这使得在真实推荐系统中部署成为可能。此外，SimHash的局部敏感性质确保了指纹始终与CTR模型中的原始嵌入保持同步。第5节的评估表明，这种兼容性可以大幅提升性能。如何选择合适的哈希函数，以及检索部分与ETA其余部分的联合学习，将在第4.5.2节和第4.5.1节中说明。

**表2：不同检索算法的比较。** $d$ 是嵌入向量的维度。 $L$ 和 $B$ 分别是行为item和目标item的数量。 $m$ 是SimHash生成的指纹维度。 $M$ 是每个用户的属性倒排索引大小。在真实世界的CTR模型中， $L = 1024$、 $B = 1024$、 $d = 128$、 $m = 4$、 $M = 300$。值得注意的是，在线直接进行内积计算会违反时间约束，无法部署在大规模在线推荐系统中。

| 检索输入 | 检索方法 | 检索与CTR模型之间的目标差距 | 检索复杂度 | 代表 |
|---|---|---|---|---|
| 属性 | 离线倒排索引 | 大 | $O(B \times \log(M))$ | SIM(hard)[21]和UBR4CTR[22] |
| 预训练嵌入 | 离线倒排索引和欧氏距离 | 中 | $O(B \times M \times d)$ | SIM(soft)[21] |
| 基于SimHash的指纹 | 汉明距离 | 小 | $O(B \times L \times m)$ | ETA |
| CTR模型的嵌入 | 内积 | 无 | $O(B \times L \times d)$ | 无法部署 |

经过SimHash函数和汉明距离层后，从 $H_{lu}$ 中选择top-k个相似行为item，然后执行前述的多头目标注意力以生成隐藏向量。该向量作为长期用户兴趣的表示，与其他向量一起被馈入MLP（多层感知机，Multi-Layer Perceptron）层。长期兴趣单元的公式如下：

$$
\mathrm{LTI}(\boldsymbol{E}_t, \boldsymbol{E}_s) = \mathrm{TA}(\boldsymbol{E}_t, \boldsymbol{E}_s^{\prime}), \qquad (6)
$$

$$
\boldsymbol{E}_s^{\prime} = \begin{bmatrix} \boldsymbol{e}_1^{\top} \\ \boldsymbol{e}_i^{\top} \\ \vdots \\ \boldsymbol{e}_k^{\top} \end{bmatrix}, \qquad (7)
$$

$$
\boldsymbol{e}_i \in \mathrm{top}_k(\mathrm{HammingDistance}(\boldsymbol{h}_i, \boldsymbol{h}_t)), \qquad (8)
$$

$$
\boldsymbol{h}_i = \mathrm{SimHash}(\boldsymbol{e}_i), \quad \boldsymbol{h}_t = \mathrm{SimHash}(\boldsymbol{e}_t), \qquad (9)
$$

其中LTI（Long-term user Interest extraction unit，长期用户兴趣提取单元）和TA（Target Attention，目标注意力）分别是长期兴趣提取单元和多头目标注意力的缩写。 $\boldsymbol{E}_s \in \mathbb{R}^{|H_{lu}| \times d}$ 是长期用户行为序列 $H_{lu}$ 的嵌入矩阵。 $\boldsymbol{E}_s^{\prime} \in \mathbb{R}^{k \times d}$ 由从 $\boldsymbol{E}_s$ 中选择的与目标item $\boldsymbol{E}_t \in \mathbb{R}^{1 \times d}$ 汉明距离最大的前 $k$ 行组成。

**相似度函数。** 如图2所示，我们使用SimHash函数和汉明距离来计算两个嵌入向量的相似度，而不是内积。SimHash函数接收前述嵌入层的输出作为输入。对于每个输入嵌入向量，SimHash函数生成其压缩数字作为指纹。SimHash满足局部敏感性质：如果输入特征彼此相似，则哈希输出也相似。因此，嵌入向量之间的相似度可以用哈希指纹之间的相似度代替。一个 $d$ 维嵌入向量可以被编码为 $m$ 位数字。然后两个指纹之间的相似度可以通过汉明距离来衡量。

**Top-K检索。** 与基于内积的模型相比，top-k检索层可以通过汉明距离更高效地找到与目标item最相似的top-k个用户行为item。两个整数的汉明距离定义为对应位不同的位置数量。要计算两个 $m$ 位数字的汉明距离，我们首先进行XOR（exclusive OR，异或）操作，然后计算结果为1的位数。如果我们将乘法定义为原子操作，则两个 $m$ 位数字的汉明距离复杂度为 $O(1)$。基于汉明距离的top-k检索的总复杂度为 $O(L \times B \times 1)$，其中 $L$ 是序列长度， $B$ 是候选item数量。值得注意的是，每次执行SimHash函数时，哈希后的指纹可以存储在模型的嵌入表中。推理时，只需要进行嵌入查找，其复杂度可忽略不计。

### 4.5 部署

在本节中，我们展示ETA是如何与检索部分一起训练的。然后介绍如何选择SimHash算法中使用的"哈希函数"。最后介绍工程优化技巧。

#### 4.5.1 检索部分的联合学习

在训练阶段，检索部分不需要更新梯度。检索的目标是为后续的多头目标注意力部分选择与查询最近的邻居键。选择出与查询最近的top-k个键后，对这些top-k item的原始嵌入向量执行正常的注意力和反向传播。检索部分唯一要做的事情是在训练开始时初始化一个固定的随机矩阵 $\boldsymbol{H} \in \mathbb{R}^{d \times m}$ （见算法1）。只要输入嵌入向量 $\boldsymbol{e}_k \in \mathbb{R}^{1 \times d}$ 被更新，SimHash的签名就会相应更新。局部敏感性质确保了每次迭代中，使用CTR模型的最新嵌入无缝地选择与查询最近的top-k个键。因此，检索与CTR模型之间的目标差距远小于其他检索方法，例如表2中基于离线倒排索引的方法。从CTR模型的角度看，检索部分是透明的，但能确保模型使用最接近的item来进行多头注意力计算。评估部分（第5节）表明，这种无需任何预训练或离线倒排索引构建的端到端训练可以大幅提升CTR预测任务的性能。

#### 4.5.2 "哈希函数"的选择

SimHash是一种著名的局部敏感哈希（LSH，Locality-sensitive Hashing）[1]算法。SimHash的实现如算法1所示，我们使用固定的随机哈希向量作为"哈希函数"。任何将字符串哈希为随机整数的传统哈希函数也可以使用。然而，在我们的算法中，考虑到矩阵计算的可扩展性和效率，我们选择随机哈希向量和算法1的实现，这与Reformer[13]相同。局部敏感哈希通过随机旋转和投影实现。随机旋转是指嵌入向量与固定随机哈希向量 $\boldsymbol{H}^{(j)}$ 之间的乘法。这里可以使用任何随机 $d$ 维向量。值得注意的是，由于我们需要将内积的结果投影到两个有符号轴上以获得二进制签名， $\boldsymbol{H}^{(j)}$ 中的元素应在0附近随机生成。

#### 4.5.3 工程优化技巧

当模型在线部署时，SimHash的计算可以进一步简化。对于通过算法1计算的嵌入向量 $\boldsymbol{e}_k$ 的 $m$ 位签名向量 $\boldsymbol{sig}_k$，我们可以使用 $\log(m)$ 位整数来表示签名向量，因为 $\boldsymbol{sig}_k$ 中的每个元素要么是1要么是0。这可以大幅降低内存成本，并加速汉明距离的计算。两个整数的计算时间可以在 $O(1)$ 时间复杂度内完成，可以忽略不计。

## 5 实验

在本节中，我们进行实验来回答以下研究问题：

*   RQ1：我们的ETA模型是否优于基线模型？
*   RQ2：与基线模型相比，我们的ETA模型的推理时间如何？推理时间与性能同等重要，因为它决定了模型是否可以在线部署并提供服务。
*   RQ3：我们ETA模型的哪个部分对性能和推理时间的贡献最大？

在展示评估结果之前，我们首先描述数据集、基线模型、评估指标和实验设置。

### 5.1 数据集

为在ETA模型与基线模型之间进行全面比较，我们使用了公开数据集和工业数据集。同时进行了在线A/B测试。对于公开数据集，我们选择淘宝数据集，该数据集也被基线模型SIM[21]和UBR4CTR[22]采用。我们准备了一个工业数据集作为公开数据集的补充。表3给出了数据集的简要介绍。

**表3：本文使用的数据集语料库规模。**

| 数据集 | 用户数 | item数 | 类目数 | 实例数 |
|---|---|---|---|---|
| 淘宝 | 987,994 | 4,162,024 | 9,439 | 100,150,807 |
| 工业（自有） | 4亿 | 7亿 | 24,568 | 1420亿 |

**淘宝数据集1：** 该数据集由[38]首次发布，被广泛用作CTR预测任务和序列推荐任务的公开基准。它由淘宝移动App的用户行为日志组成。用户行为包括点击、收藏、加购和购买。该数据集包含1亿个实例。平均而言，每个用户约有101次交互，每个item获得超过24次交互。选择最近的16个行为作为短期用户行为序列，最近的256个行为作为长期用户行为序列。

**工业数据集2：** 该数据集收集自我们自己的在线RS，是我国顶级移动App之一。我们的工业数据集有三个优势：(i) 数据集包含曝光交互，这表明item展示给了用户但用户没有点击。曝光交互天然是CTR模型的负样本。因此，不需要棘手的负采样。(ii) 我们的工业数据集中用户行为序列长得多。超过1420亿个实例，平均长度达到938，是公开淘宝数据集的9倍。(iii) 我们的工业数据集拥有由多位软件工程师设计的更多特征，更接近真实世界的RS模型。选择最近的48个行为作为短期用户行为序列，最近的1024个行为作为长期用户行为序列。在消融研究中，我们还尝试了长度在 $\{256, 512, 2048\}$ 范围内的长期用户行为序列。

1 https://tianchi.aliyun.com/dataset/dataDetail?dataId=649&userId=1
2 该数据集将向公众发布，以帮助长期用户兴趣建模的研究。

### 5.2 基线与评估指标

**基线：** 我们将模型与以下主流的CTR预测基线进行比较。每个基线都旨在回答上述一个或多个相关研究问题。

*   **Avg-Pooling DNN（Deep Neural Network，深度神经网络）：** 利用用户行为序列的最简单方式是平均池化，它将不同长度的用户序列编码为固定大小的隐藏向量。该基线可以视为DIN的变体，将目标注意力替换为平均池化，类似于YouTube[6]。该基线主要用于在与DIN比较时展示目标注意力的必要性。
*   **DIN[36]：** DIN通过注意力机制（称为目标注意力）为不同目标item建模个性化用户兴趣。然而，DIN仅利用了短期用户行为序列。
*   **DIN（长序列）：** 是配备了长期用户行为序列 $H_{lu}$ 的DIN。 $H_{lu}$ 通过平均池化编码。该基线用于衡量长期用户行为序列本身与DIN相比的信息增益。
*   **SIM(hard)[21]：** SIM是一种CTR预测模型，提出以两阶段方式从长期用户行为序列中提取用户兴趣的搜索单元。SIM(hard)是在第一阶段通过品类ID搜索top-k行为item的SIM。
*   **UBR4CTR[22]：** UBR4CTR也是一种两阶段方法，在CTR预测任务中利用长期用户行为序列。在UBR4CTR中，通过特征选择模型准备一个查询来检索最相似的行为item。为在线使用准备了一个倒排索引。由于UBR4CTR和SIM几乎同时发表，它们之间没有相互比较。在本文中，我们首次同时比较了UBR4CTR和SIM。
*   **SIM(hard)/UBR4CTR + timeinfo：** 在编码用户行为序列时添加时间嵌入的SIM(hard)/UBR4CTR。

在[21]中，作者提出了SIM(soft)作为基础算法SIM(hard)的变体。他们最终采用SIM(hard)方法作为在线服务算法，并在线部署SIM(hard)+timeinfo服务于主要流量。这是因为SIM(hard)不需要预训练，对系统演进和维护更友好。此外，SIM(hard)+timeinfo可以获得与SIM(soft)相当的性能。因此，我们选择SIM(hard)和SIM(hard)+timeinfo作为强基线。

MIMN[20]由DIN的同一团队提出。MIMN提出了一个多轨道离线用户兴趣中心来提取长期用户兴趣。在其发布时，它通过利用长期用户行为序列达到了当时最先进的性能。然而，MIMN被来自同一团队的SIM[21]所击败。由于MIMN对我们的研究问题贡献不大，出于篇幅限制，我们省略了该基线。

**评估指标：** 对于离线实验，我们采用广泛使用的ROC曲线下面积（AUC）作为主要指标，推理时间作为补充指标。AUC适用于衡量二分类问题的配对排序性能。推理时间定义为对特定模型请求的一批item进行评分时的往返时间。我们通过在线部署模型来服务从生产环境复制的用户请求来衡量推理时间。为公平比较，机器和用户请求数量保持相同。

对于在线A/B测试，我们使用CLICK和CTR作为评估指标。CLICK定义为被点击item的总数。CTR用于衡量平台上用户的点击意愿。其定义为CLICK/PV，其中PV（Page View，页面浏览量）定义为展示item的总数。

### 5.3 实验设置

在本节中，我们首先介绍离线数据集的预处理。然后列出基线和模型的超参数。

淘宝数据集仅包含正向交互，例如浏览、点击、收藏、加购和购买。我们使用与MIMN[20]相同的数据预处理方法。首先，对于每个用户，选择最后一个行为作为正样本。然后为该用户随机采样一个与正样本同品类的新item作为负样本。其余行为item用作特征。根据样本的时间戳 $t$，将样本划分为训练集（80%）、验证集（10%）和测试集（10%）。

我们的工业数据集天然包含正样本和负样本，因为我们记录了每个用户的所有曝光。如果item被用户点击，则该曝光标记为正样本。否则，标记为负样本。我们使用过去两周的日志作为训练集，次日作为测试集，这与SIM[21]类似。

对于不同数据集上的每个模型，我们使用验证集调整超参数以获得最佳性能。学习率从 $1 \times 10^{-4}$ 到 $1 \times 10^{-2}$ 之间搜索。 $L_2$ 正则化项从 $1 \times 10^{-4}$ 到1之间搜索。所有模型使用Adam优化器。淘宝数据集和工业数据集的批大小分别为256和1024。

### 5.4 性能比较

**淘宝数据集：** 在淘宝数据集上的评估结果如表4所示。从表中我们发现，我们的ETA在所有基线上都有稳定的性能提升。ETA比SIM(hard)高出0.46%，比DIN（长序列）高出0.6%。添加时间嵌入后，ETA+timeinfo比SIM(hard)+timeinfo高出0.38%，比DIN（长序列）高出0.85%。在SIM(hard)和UBR4CTR上可以观察到类似的结果。可以观察到，DIN（长序列）相比DIN在AUC上带来了0.35%的提升，这表明对CTR预测建模长期用户行为序列的有效性。我们还发现UBR4CTR的表现不如DIN（长序列）。这是因为UBR4CTR的特征选择模型只选择那些特征（例如品类、星期几）与目标item相同的行为。这种过滤在UBR4CTR中帮助去除序列中的噪声，但也可能导致用户序列变短，当没有足够item进行top-k检索时是有害的。从表4中我们发现，DIN比Avg-Pooling DNN高出1.84%，证实了使用目标注意力编码用户序列可以大幅提升性能。

**表4：淘宝数据集上的实验结果**

| 方法 | AUC |
|---|---|
| Avg-Pooling DNN | 0.8442 |
| DIN | 0.8626 |
| DIN（长序列） | 0.8661 |
| UBR4CTR | 0.8651 |
| UBR4CTR+timeinfo | 0.8683 |
| SIM(hard) | 0.8675 |
| SIM(hard)+timeinfo | 0.8708 |
| ETA | 0.8721 |
| ETA+timeinfo | 0.8746 |

**工业数据集：** 在我们自己的工业数据集上的评估结果如表5所示。请注意，CTR模型AUC提升0.1%就能在我们的在线RS中带来数百万的真实点击和收入。我们的ETA在所有基线上取得了最佳性能。我们的基础ETA相比SIM(hard)和UBR4CTR分别提升了0.34%和0.43%。我们的ETA+timeinfo相比SIM(hard)+timeinfo和UBR4CTR+timeinfo分别提升了0.35%和0.42%。与公开数据集上的实验结果不同，SIM(hard)+timeinfo在工业数据集上成为最强的基线，比DIN（长序列）高出0.27%。这有两个原因。一方面，工业数据集的用户序列长度足够大，有利于基于长期用户序列的模型。工业数据集的平均长度达到938，是公开淘宝数据集的9倍。另一方面，DNN（长序列）使用平均池化编码整个序列而不加选择，与基于检索的模型（如UBR4CTR、SIM和ETA）相比可能引入噪声。

**表5：工业数据集上的实验结果**

| 方法 | AUC | 推理时间（ms） |
|---|---|---|
| Avg-Pooling DNN | 0.7216 | 8 |
| DIN | 0.7279 | 11 |
| DIN（长序列） | 0.7311 | 14 |
| UBR4CTR | 0.7318 | 41 |
| UBR4CTR+timeinfo | 0.7331 | 41 |
| SIM(hard) | 0.7327 | 21 |
| SIM(hard)+timeinfo | 0.7338 | 21 |
| ETA | 0.7361 | 19 |
| ETA+timeinfo | 0.7373 | 19 |

我们还可以发现另一个事实：SIM(hard)相比UBR4CTR有0.09%的性能提升。这主要是由对用户行为序列的不同处理方法造成的。在SIM(hard)中，用户序列被分割成两个独立的子序列，这与我们的ETA在图2中类似。短期用户行为序列 $H_{su}$ 由从item 1到item $k$ 的最近 $k$ 个用户行为组成。长期用户行为序列 $H_{lu}$ 由从item $k+1$ 到item $n$ 中选择的另外 $k$ 个行为组成。然而，UBR4CTR从item 1到item $n$ 中选择一个 $2 \times k$ 长度的行为序列。因此，最近的 $k$ 个item（图2中的 $\boldsymbol{e}_1$ 到 $\boldsymbol{e}_k$）在SIM(hard)中以100%的概率被选中，而在UBR4CTR中以由特征选择模型决定的概率 $p$ 被选中。然而，时间信息在用户兴趣建模中起着重要作用，因为用户兴趣是动态的且频繁变化。因此SIM(hard)的表现优于UBR4CTR。

**在线A/B测试：** 在线A/B测试的评估结果如表6所示。表6展示了相对于基于DIN方法的性能提升，其中基于DIN的方法没有长期用户行为序列。从表6我们发现，我们的ETA+timestamp在CTR上实现了6.33%的提升，与基于DIN的方法相比带来了9.7%的额外GMV。与最强基线SIM(hard)+timeinfo相比，我们的ETA+timeinfo在CTR上额外提升了1.8%，在GMV上额外提升了3.1%。请注意，GMV提升1%是显著的改进，因为它意味着为推荐系统带来了数百万的额外收入。

**表6：在线A/B测试中的相对性能改进和推理时间，与没有长期用户行为序列的基于DIN的方法相比。注意，GMV提升1%是显著的改进，因为它意味着为推荐系统带来数百万的额外收入。**

| 方法 | CTR | GMV | 推理时间 |
|---|---|---|---|
| SIM(hard)+timeinfo | 4.53% | 6.6% | 21ms |
| ETA+timeinfo | 6.33% | 9.7% | 19ms |

### 5.5 推理时间比较

尽管使用长期用户行为序列提升了CTR预测的性能，模型复杂度也随之增加。我们测量了不同模型的推理时间，如表4所示。Avg-Pooling DNN的推理时间最小，为8毫秒（ms）。它仅使用平均池化方法对最近的行为item进行编码。将平均池化替换为目标注意力后，推理时间增加了3ms（从8ms到11ms）。引入长期用户行为序列后，推理时间又增加了3ms（从11ms到14ms）。SIM和我们的ETA具有相当的推理时间，约为19-21ms。UBR4CTR的推理时间最长，因为在检索阶段之前使用了一个额外的特征选择模型，并且在线执行了相对耗时的基于IDF（Inverse Document Frequency，逆文档频率）和BM25的过程来获取top-k item。

### 5.6 消融研究

消融研究的结果如表7所示，用于回答研究问题RQ3。我们使用编码方式来区分ETA模型的不同版本（v0到v4）。注意，v0是ETA的基础版本。编码方式列在表7的第二列中，其中avg(·)和ta(·)分别表示通过平均池化和目标注意力对用户行为进行编码。ta(1024 -s- 48)表示对从1024个序列用户行为item中选择的top-48个用户行为执行目标注意力。ta(1024 -s- 48)中的符号s表示使用SimHash从1024个中选择top-48个。类似地，ta(1024 -i- 48)中的符号i表示使用内积从1024个中选择top-48个。

**表7：ETA模型在工业数据集上的消融研究。v0是ETA的基础版本。avg(·)和ta(·)分别表示通过平均池化和目标注意力对用户行为进行编码。ta(1024 -s- 48)表示对从1024个序列用户行为item中选择的top-48个用户行为执行目标注意力。ta(1024 -s- 48)中的符号s表示使用SimHash从1024个中选择top-48个。类似地，ta(1024 -i- 48)中的符号i表示使用内积从1024个中选择top-48个。**

| ETA版本 | 编码方式 | AUC | 推理时间（ms） |
|---|---|---|---|
| v0 | ta(1024 -s- 48) | 0.7361 | 19 |
| v1 | avg(1024) | 0.7311 | 14 |
| v2.1 | ta(256 -s- 48) | 0.7339 | 14 |
| v2.2 | ta(512 -s- 48) | 0.7348 | 16 |
| v2.3 | ta(2048 -s- 48) | 0.7394 | 23 |
| v3 | ta(1024 -i- 48) | 0.7368 | 32 |
| v4 | ta(1024) | 0.7371 | 35 |

从表7中，我们得到以下观察结果。(i) 直接在原始1024长度的用户序列上执行多头目标注意力（v4）可以获得最佳性能，但同时推理时间最高。与v4相比，我们的基础ETA（v0）选择top-k个行为进行注意力，牺牲了约0.1%的AUC，但将推理时间减少了46%。(ii) 比较v3与v0，在检索阶段将SimHash替换为内积在AUC上获得了0.07%的提升。然而，推理时间增加了68%，这不符合我们严格的在线SLA（Service Level Agreement，服务等级协议）。(iii) 当我们改变用户行为序列的长度时（v2.x与v0），可以观察到AUC与推理时间之间的权衡。可以根据在线推理时间的要求来决定合适的序列长度。我们还在图3中评估了SimHash生成的不同位长度的哈希指纹 $\boldsymbol{h}$ 下的性能。如第3.2节所述，指纹的位长度可以通过SimHash中使用的哈希函数数量来控制。我们发现，增加 $\boldsymbol{h}$ 的位长度可以提升AUC。然而，当 $\boldsymbol{h}$ 的位长度超过嵌入大小的2倍时，AUC的提升变得边际化。

**图3：不同位长度的哈希指纹下ETA的AUC。其他模型设置与表5中的ETA相同。**

## 6 结论

在本文中，我们提出了用于CTR预测任务的ETA模型。据我们所知，ETA是第一种能够以端到端方式将CTR与长期用户行为序列一起建模的方法。与SOTA的两阶段模型相比，端到端范式使得检索部分与CTR模型的主体部分无缝共享信息，从而显著提升了预测性能。此外，它对大规模在线RS中CTR模型的维护和演进也很友好。为实现端到端在线检索的目标，我们提出了一种基于SimHash的方法，将传统top-k检索的复杂度从 $O(L \times B \times d)$ 次乘法降低到 $O(L \times B)$ 次汉明距离计算，其中 $L$ 是用户序列的长度， $B$ 是每个用户请求的候选item数量， $d$ 是item嵌入的维度。离线和在线实验均证实了我们ETA的有效性。在线A/B测试中，与SOTA模型相比，总GMV提升了3.1%。ETA已在线部署，服务于主流流量。

## 参考文献

[1] Alexandr Andoni, Piotr Indyk, Thijs Laarhoven, Ilya Razenshteyn, and Ludwig Schmidt. 2015. Practical and optimal LSH for angular distance. arXiv preprint arXiv:1509.02897 (2015).

[2] Moses S Charikar. 2002. Similarity estimation techniques from rounding algorithms. In Proceedings of the thiry-fourth annual ACM symposium on Theory of computing. 380–388.

[3] Heng-Tze Cheng, Levent Koc, Jeremiah Harmsen, Tal Shaked, Tushar Chandra, Hrishi Aradhye, Glen Anderson, Greg Corrado, Wei Chai, Mustafa Ispir, et al. 2016. Wide & deep learning for recommender systems. In Proceedings of the 1st workshop on deep learning for recommender systems. 7–10.

[4] Kyunghyun Cho, Bart Van Merriënboer, Dzmitry Bahdanau, and Yoshua Bengio. 2014. On the properties of neural machine translation: Encoder-decoder approaches. arXiv preprint arXiv:1409.1259 (2014).

[5] Kyunghyun Cho, Bart Van Merriënboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. 2014. Learning phrase representations using RNN encoder-decoder for statistical machine translation. arXiv preprint arXiv:1406.1078 (2014).

[6] Paul Covington, Jay Adams, and Emre Sargin. 2016. Deep neural networks for youtube recommendations. In Proceedings of the 10th ACM conference on recommender systems. 191–198.

[7] Yufei Feng, Fuyu Lv, Weichen Shen, Menghan Wang, Fei Sun, Yu Zhu, and Keping Yang. 2019. Deep session interest network for click-through rate prediction. arXiv preprint arXiv:1905.06482 (2019).

[8] Huifeng Guo, Ruiming Tang, Yunming Ye, Zhenguo Li, and Xiuqiang He. 2017. DeepFM: a factorization-machine based neural network for CTR prediction. arXiv preprint arXiv:1703.04247 (2017).

[9] Xinran He, Junfeng Pan, Ou Jin, Tianbing Xu, Bo Liu, Tao Xu, Yanxin Shi, Antoine Atallah, Ralf Herbrich, Stuart Bowers, et al. 2014. Practical lessons from predicting clicks on ads at facebook. In Proceedings of the Eighth International Workshop on Data Mining for Online Advertising. 1–9.

[10] Balázs Hidasi, Alexandros Karatzoglou, Linas Baltrunas, and Domonkos Tikk. 2015. Session-based recommendations with recurrent neural networks. arXiv preprint arXiv:1511.06939 (2015).

[11] Sepp Hochreiter and Jürgen Schmidhuber. 1997. Long short-term memory. Neural computation 9, 8 (1997), 1735–1780.

[12] Wang-Cheng Kang and Julian McAuley. 2018. Self-attentive sequential recommendation. In 2018 IEEE International Conference on Data Mining (ICDM). IEEE, 197–206.

[13] Nikita Kitaev, Łukasz Kaiser, and Anselm Levskaya. 2020. Reformer: The efficient transformer. arXiv preprint arXiv:2001.04451 (2020).

[14] Yann LeCun, Bernhard Boser, John S Denker, Donnie Henderson, Richard E Howard, Wayne Hubbard, and Lawrence D Jackel. 1989. Backpropagation applied to handwritten zip code recognition. Neural computation 1, 4 (1989), 541–551.

[15] Chao Li, Zhiyuan Liu, Mengmeng Wu, Yuchi Xu, Huan Zhao, Pipei Huang, Guoliang Kang, Qiwei Chen, Wei Li, and Dik Lun Lee. 2019. Multi-interest network with dynamic routing for recommendation at Tmall. In Proceedings of the 28th ACM International Conference on Information and Knowledge Management. 2615–2623.

[16] Zeyu Li, Wei Cheng, Yang Chen, Haifeng Chen, and Wei Wang. 2020. Interpretable Click-Through Rate Prediction through Hierarchical Attention. In Proceedings of the Thirteenth ACM International Conference on Web Search and Data Mining. ACM.

[17] Jianxun Lian, Xiaohuan Zhou, Fuzheng Zhang, Zhongxia Chen, Xing Xie, and Guangzhong Sun. 2018. xdeepfm: Combining explicit and implicit feature interactions for recommender systems. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 1754–1763.

[18] Gurmeet Singh Manku, Arvind Jain, and Anish Das Sarma. 2007. Detecting near-duplicates for web crawling. In Proceedings of the 16th international conference on World Wide Web. 141–150.

[19] Junwei Pan, Jian Xu, Alfonso Lobos Ruiz, Wenliang Zhao, Shengjun Pan, Yu Sun, and Quan Lu. 2018. Field-weighted factorization machines for click-through rate prediction in display advertising. In Proceedings of the 2018 World Wide Web Conference. 1349–1357.

[20] Qi Pi, Weijie Bian, Guorui Zhou, Xiaoqiang Zhu, and Kun Gai. 2019. Practice on long sequential user behavior modeling for click-through rate prediction. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 2671–2679.

[21] Pi Qi, Xiaoqiang Zhu, Guorui Zhou, Yujing Zhang, Zhe Wang, Lejian Ren, Ying Fan, and Kun Gai. 2020. Search-based User Interest Modeling with Lifelong Sequential Behavior Data for Click-Through Rate Prediction. arXiv preprint arXiv:2006.05639 (2020).

[22] Jiarui Qin, Weinan Zhang, Xin Wu, Jiarui Jin, Yuchen Fang, and Yong Yu. 2020. User Behavior Retrieval for Click-Through Rate Prediction. arXiv preprint arXiv:2005.14171 (2020).

[23] Yanru Qu, Han Cai, Kan Ren, Weinan Zhang, Yong Yu, Ying Wen, and Jun Wang. 2016. Product-based neural networks for user response prediction. In 2016 IEEE 16th International Conference on Data Mining (ICDM). IEEE, 1149–1154.

[24] Kan Ren, Jiarui Qin, Yuchen Fang, Weinan Zhang, Lei Zheng, Weijie Bian, Guorui Zhou, Jian Xu, Yong Yu, Xiaoqiang Zhu, et al. 2019. Lifelong Sequential Modeling with Personalized Memorization for User Response Prediction. In Proceedings of the 42nd International ACM SIGIR Conference on Research and Development in Information Retrieval. 565–574.

[25] Steffen Rendle. 2010. Factorization machines. In 2010 IEEE International Conference on Data Mining. IEEE, 995–1000.

[26] Fei Sun, Jun Liu, Jian Wu, Changhua Pei, Xiao Lin, Wenwu Ou, and Peng Jiang. 2019. BERT4Rec: Sequential recommendation with bidirectional encoder representations from transformer. In Proceedings of the 28th ACM International Conference on Information and Knowledge Management. 1441–1450.

[27] Jiaxi Tang and Ke Wang. 2018. Personalized top-n sequential recommendation via convolutional sequence embedding. In Proceedings of the Eleventh ACM International Conference on Web Search and Data Mining. 565–573.

[28] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. Attention Is All You Need. CoRR abs/1706.03762 (2017). arXiv:1706.03762 http://arxiv.org/abs/1706.03762

[29] Ruoxi Wang, Bin Fu, Gang Fu, and Mingliang Wang. 2017. Deep & cross network for ad click predictions. In Proceedings of the ADKDD'17. 1–7.

[30] Jun Xiao, Hao Ye, Xiangnan He, Hanwang Zhang, Fei Wu, and Tat-Seng Chua. 2017. Attentional factorization machines: Learning the weight of feature interactions via attention networks. arXiv preprint arXiv:1708.04617 (2017).

[31] Weinan Xu, Hengxu He, Minshi Tan, Yunming Li, Jun Lang, and Dongbai Guo. 2020. Deep Interest with Hierarchical Attention Network for Click-Through Rate Prediction. arXiv preprint arXiv:2005.12981 (2020).

[32] Fajie Yuan, Alexandros Karatzoglou, Ioannis Arapakis, Joemon M Jose, and Xiangnan He. 2019. A simple convolutional generative network for next item recommendation. In Proceedings of the Twelfth ACM International Conference on Web Search and Data Mining. 582–590.

[33] Weinan Zhang, Tianming Du, and Jun Wang. 2016. Deep learning over multi-field categorical data. In European conference on information retrieval. Springer, 45–57.

[34] Guorui Zhou, Weijie Bian, Kailun Wu, Lejian Ren, Qi Pi, Yujing Zhang, Can Xiao, Xiang-Rong Sheng, Na Mou, Xinchen Luo, et al. 2020. CAN: Revisiting Feature Co-Action for Click-Through Rate Prediction. arXiv preprint arXiv:2011.05625 (2020).

[35] Guorui Zhou, Na Mou, Ying Fan, Qi Pi, Weijie Bian, Chang Zhou, Xiaoqiang Zhu, and Kun Gai. 2019. Deep interest evolution network for click-through rate prediction. In Proceedings of the AAAI conference on artificial intelligence, Vol. 33. 5941–5948.

[36] Guorui Zhou, Xiaoqiang Zhu, Chenru Song, Ying Fan, Han Zhu, Xiao Ma, Yanghui Yan, Junqi Jin, Han Li, and Kun Gai. 2018. Deep interest network for click-through rate prediction. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 1059–1068.

[37] Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai Zhang, Jianxin Li, Hui Xiong, and Wancai Zhang. 2020. Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting. arXiv preprint arXiv:2012.07436 (2020).

[38] Han Zhu, Daqing Chang, Ziru Xu, Pengye Zhang, Xiang Li, Jie He, Han Li, Jian Xu, and Kun Gai. 2019. Joint optimization of tree-based index and deep model for recommender systems. arXiv preprint arXiv:1902.07565 (2019).
