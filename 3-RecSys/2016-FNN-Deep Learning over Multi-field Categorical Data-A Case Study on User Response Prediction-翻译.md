# FNN：基于深度学习的多领域分类数据用户响应预测

> Weinan Zhang, Tianming Du, Jun Wang | University College London, London, United Kingdom; RayCloud Inc., Hangzhou, China

本文介绍了两种利用深度神经网络（DNN，Deep Neural Network）从大规模多领域分类特征中自动学习特征交互模式并进行用户广告点击预测的模型。核心内容：

- 提出分解机支持神经网络（FNN，Factorisation-machine supported Neural Network），利用分解机（FM，Factorisation Machine）预训练第一层嵌入，将高维稀疏特征高效映射为低维稠密表示，解决大规模分类特征空间的计算复杂度问题
- 提出采样神经网络（SNN，Sampling-based Neural Network），结合基于采样的受限玻尔兹曼机（SNN-RBM，Sampling-based Restricted Boltzmann Machine）和去噪自编码器（SNN-DAE，Sampling-based Denoising Auto-Encoder）进行无监督预训练，进一步降低计算开销
- 在真实大规模数据集上的实验表明，FNN 和 SNN 在 ROC 曲线下面积（AUC，Area Under the ROC Curve）指标上一致优于逻辑回归（LR，Logistic Regression）和分解机等基线模型

关键发现：

- 在 iPinYou 公开数据集上，FNN 在所有广告商数据和整体数据上均取得最高的 AUC，整体 AUC 达到 70.70%，显著优于 LR（68.81%）和 FM（68.18%）
- 菱形（diamond）网络架构（如 200-300-100 隐藏单元配置）在几乎所有层级设置下均优于等宽、递增和递减架构
- Dropout 正则化在所有设置下均优于 L2 正则化；FNN 最优 Dropout 率约为 0.8，SNN 约为 0.99

---

## 摘要

用户响应预测（如点击率（CTR，Click-Through Rate）和转化率（CVR，Conversion Rate））在许多网络应用中至关重要，包括网络搜索、个性化推荐和在线广告。与图像和音频领域中常见的连续原始特征不同，网络空间中的输入特征总是多领域的，且大多是离散和分类的，而它们之间的依赖关系鲜为人知。主流的用户响应预测模型要么局限于线性模型，要么需要手动构建高阶组合特征。前者缺乏探索特征交互的能力，后者在大规模特征空间中导致沉重的计算负担。为解决这一问题，我们提出了两种使用深度神经网络（DNN）自动学习分类特征交互有效模式并预测用户广告点击的模型。为了让 DNN 高效工作，我们提出利用三种特征转换方法：分解机（FM）、受限玻尔兹曼机（RBM，Restricted Boltzmann Machine）和去噪自编码器（DAE，Denoising Auto-Encoder）。本文介绍了我们模型的结构及其高效训练算法。大规模真实数据实验表明，我们的方法优于主流的最先进模型。

**关键词：** Deep Learning, Click-Through Rate, Factorisation Machine, Neural Network

---

## 1 引言

用户响应（如点击或转化）预测在许多网络应用中扮演关键角色，包括网络搜索、推荐系统、赞助搜索和展示广告。在在线广告中，针对特定用户的定向投放能力是相比传统线下广告的核心优势。所有这些定向技术本质上都依赖于系统预测特定用户是否认为潜在广告"相关"的功能，即在特定情境下用户点击给定广告的概率[6]。赞助搜索、上下文广告以及最近兴起的实时竞价（RTB，Real-Time Bidding）展示广告都严重依赖于学习到的模型预测广告点击率（CTR）的能力[32, 41]。当前应用的 CTR 估计模型大多是线性的，从逻辑回归（LR）[32]和朴素贝叶斯[14]到 FTRL 逻辑回归[28]和贝叶斯概率回归[12]，这些模型都基于大量使用独热编码[1]的稀疏特征。线性模型具有实现简单、学习高效的优点，但由于无法学习假设条件独立的原始特征之间的非平凡交互模式[12]，性能相对较低。另一方面，非线性模型能够利用不同的特征组合，因此有可能提升估计性能。例如，分解机（FM）[29]将用户和 item 的二值特征映射到低维连续空间，通过向量内积自动探索特征交互。梯度提升树[38]在构建每棵决策/回归树时自动学习特征组合。然而，这些模型无法利用所有可能的不同特征组合[20]。此外，许多模型需要特征工程来手动设计输入。主流广告 CTR 估计模型的另一个问题是，大多数预测模型具有浅层结构，在从复杂海量数据中建模底层模式方面表达能力有限[15]。因此，它们的数据建模和泛化能力仍然受到限制。

深度学习[25]在计算机视觉[22]、语音识别[13]和自然语言处理（NLP，Natural Language Processing）[19, 33]领域在过去五年取得了成功。由于视觉、听觉和文本信号在空间和/或时间上具有相关性，新引入的深度结构无监督训练[18]能够探索这种局部依赖性并建立特征空间的稠密表示，使神经网络模型能够直接从原始特征输入中学习高阶特征。凭借这种学习能力，深度学习是估计在线用户响应率（如广告 CTR）的良好候选方法。然而，CTR 估计中的大多数输入特征是多领域的离散分类特征，例如用户所在城市（伦敦、巴黎）、设备类型（PC、移动端）、广告类别（体育、电子）等，它们的局部依赖性（因此在特征空间中的稀疏性）是未知的。因此，深度学习如何通过学习此类大规模多领域离散分类特征的特征表示来提升 CTR 估计，具有重要的研究意义。据我们所知，此前尚无使用深度学习方法进行广告 CTR 估计的文献[†]。此外，在大规模输入特征空间上训练深度神经网络（DNN）需要调整大量参数，计算代价高昂。例如，与图像和音频不同，我们约有 100 万个二值输入特征和第一层 100 个隐藏单元；那么构建第一层神经网络就需要 1 亿个连接。

> [†] 尽管业界已有利用深度学习模型进行广告 CTR 估计的报道（如[42]），但没有模型或实现的详细信息。

在本文中，我们以广告 CTR 估计为实例，研究在大规模多领域分类特征空间上利用嵌入方法进行有监督和无监督的深度学习。我们介绍两种深度学习模型：分解机支持神经网络（FNN）和采样神经网络（SNN）。具体来说，FNN 使用分解机[31]作为有监督嵌入层，高效地将稀疏特征降维为稠密连续特征。第二种模型 SNN 是由基于采样的受限玻尔兹曼机（SNN-RBM）或基于采样的去噪自编码器（SNN-DAE）驱动的深度神经网络，并提出了负采样方法。基于嵌入层，我们构建具有全连接的多层网络来探索非平凡的数据模式。我们在多个真实广告商的广告点击数据上的实验表明，所提出的模型在 CTR 估计方面相比最先进模型具有持续的改进。

## 2 相关工作

点击率（CTR）定义为特定用户对展示广告的点击概率，在在线广告中至关重要[39]。为了最大化收入和用户满意度，在线广告平台必须预测每个展示广告的预期用户行为，并最大化用户点击的期望。当前大多数模型使用基于一组稀疏二值特征的逻辑回归，这些特征通过独热编码从原始分类特征转换而来[26, 32]。需要大量的工程工作来设计特征，如位置、高频词、组合特征等[15]。

将非常大的特征向量嵌入低维向量空间对预测任务非常有用，因为它降低了数据和模型复杂度，并提高了训练和预测的有效性和效率。已有各种嵌入架构方法被提出[37, 23]。分解机（FM）[31]最初为协同过滤推荐提出，被认为是最成功的嵌入模型之一。FM 天然具有通过将任意两个特征映射到低秩潜在空间中的向量来估计它们之间交互的能力。

深度学习[2]是人工智能研究的一个分支，旨在开发使计算机能够以高性能处理识别和预测等复杂任务的技术。深度神经网络（DNN）能够从训练数据中提取不同抽象层次的隐藏结构和内在模式。DNN 已成功应用于计算机视觉[40]、语音识别[8]和自然语言处理（NLP）[7, 19, 33]。此外，借助无监督预训练，我们可以获得良好的特征表示，引导学习走向支持更好泛化的吸引盆地[10]。通常，这些深度模型的学习有两个阶段[18]：第一阶段通过无监督学习（即受限玻尔兹曼机（RBM）或堆叠去噪自编码器（DAE））进行模型初始化，使模型捕获输入数据分布；第二阶段通过有监督的反向传播对初始化模型进行微调。我们的深度学习模型的新颖之处在于第一层初始化，其中输入原始特征是从原始分类特征转换而来的高维稀疏二值特征，这使得传统 DNN 在大规模上难以训练。与 NLP 中使用的词嵌入技术[19, 33]相比，我们的模型处理更一般的多领域分类特征，不依赖于任何假设的数据结构（如词对齐和字母 n-gram 等）。

## 3 给定分类特征的 DNN CTR 估计

在本节中，我们详细介绍两种提出的 DNN 架构：分解机支持神经网络（FNN）和采样神经网络（SNN）。输入分类特征按领域进行独热编码。对于每个领域（例如城市），有多个单元，每个单元代表该领域的一个特定值（例如 city=London），只有一个正例（1）单元，其余均为负例（0）。编码后的特征记为 $x$ ，是许多 CTR 估计模型[32, 26]以及我们 DNN 模型的输入，如图 1 底层所示。

> **图 1：4 层 FNN 模型结构。**

### 3.1 分解机支持神经网络（FNN）

我们的第一个模型 FNN 以分解机作为底层。网络结构如图 1 所示。自顶向下描述，输出单元是一个实数 $\hat{y} \in (0, 1)$ 作为预测的 CTR，即特定用户在特定情境下点击给定广告的概率：

$$
\hat{y} = \mathrm{sigmoid}(\mathbf{W}^3 \mathbf{l}_2 + \mathbf{b}^3) \qquad (1)
$$

其中 $\mathrm{sigmoid}(x) = 1/(1 + e^{-x})$ 是逻辑激活函数， $\mathbf{W}^3 \in \mathbb{R}^{1 \times L}$ ， $\mathbf{b}^3 \in \mathbb{R}$ 且 $\mathbf{l}_2 \in \mathbb{R}^L$ 作为该层的输入。 $\mathbf{l}_2$ 的计算为：

$$
\mathbf{l}_2 = \tanh(\mathbf{W}^2 \mathbf{l}_1 + \mathbf{b}^2) \qquad (2)
$$

其中 $\tanh(x) = (1 - e^{-2x})/(1 + e^{-2x})$ ， $\mathbf{W}^2 \in \mathbb{R}^{L \times M}$ ， $\mathbf{b}^2 \in \mathbb{R}^L$ 且 $\mathbf{l}_1 \in \mathbb{R}^M$ 。我们选择 $\tanh(\cdot)$ 是因为它比其他激活函数具有更好的经验学习性能，如 4.3 节所述。类似地，

$$
\mathbf{l}_1 = \tanh(\mathbf{W}^1 \mathbf{z} + \mathbf{b}^1) \qquad (3)
$$

其中 $\mathbf{W}^1 \in \mathbb{R}^{M \times J}$ ， $\mathbf{b}^1 \in \mathbb{R}^M$ 且 $\mathbf{z} \in \mathbb{R}^J$ 。

$$
\mathbf{z} = (w_0, \mathbf{z}_1, \mathbf{z}_2, \ldots, \mathbf{z}_i, \ldots, \mathbf{z}_n) \qquad (4)
$$

其中 $w_0 \in \mathbb{R}$ 是全局标量参数， $n$ 是领域总数。 $\mathbf{z}_i \in \mathbb{R}^{K+1}$ 是分解机中第 $i$ 个领域的参数向量：

$$
\mathbf{z}_i = \mathbf{W}_i^0 \cdot \mathbf{x}[\mathrm{start}_i : \mathrm{end}_i] = (w_i, v_{i1}, v_{i2}, \ldots, v_{iK}) \qquad (5)
$$

其中 $\mathrm{start}_i$ 和 $\mathrm{end}_i$ 是第 $i$ 个领域的起始和结束特征索引， $\mathbf{W}_i^0 \in \mathbb{R}^{(K+1) \times (\mathrm{end}_i - \mathrm{start}_i + 1)}$ 且 $\mathbf{x}$ 是如开头所述的输入向量。所有权重 $\mathbf{W}_i^0$ 分别用偏置项 $w_i$ 和向量 $\mathbf{v}_i$ 初始化（例如， $\mathbf{W}_i^0[0]$ 由 $w_i$ 初始化， $\mathbf{W}_i^0[1]$ 由 $v_{i1}$ 初始化， $\mathbf{W}_i^0[2]$ 由 $v_{i2}$ 初始化，依此类推）。通过这种方式，第一层的 $\mathbf{z}$ 向量通过训练分解机（FM）[31]初始化，如图 1 所示：

$$
y_{\mathrm{FM}}(\mathbf{x}) := \mathrm{sigmoid}\left(w_0 + \sum_{i=1}^{N} w_i x_i + \sum_{i=1}^{N} \sum_{j=i+1}^{N} \langle \mathbf{v}_i, \mathbf{v}_j \rangle x_i x_j\right) \qquad (6)
$$

其中每个特征 $i$ 被分配一个偏置权重 $w_i$ 和一个 $K$ 维向量 $\mathbf{v}_i$ ，特征交互通过它们向量的内积 $\langle \mathbf{v}_i, \mathbf{v}_j \rangle$ 建模。通过这种方式，上述神经网络可以更高效地从分解机表示中学习，从而自然地绕过了高维二值输入的计算复杂度问题。不同的隐藏层可以被视为捕获数据实例不同形式表示的不同内部函数。因此，该模型具有更强的捕获内在数据模式的能力，从而带来更好的性能。

使用 FM 作为底层的灵感来自卷积神经网络（CNN，Convolutional Neural Network）[11]，CNN 通过在相邻层的神经元之间强制建立局部连接模式来利用空间局部相关性。类似地，隐藏层 1 的输入连接到特定领域的输入单元。此外，底层不是全连接的，因为 FM 对独热稀疏编码输入执行按领域的训练，允许局部稀疏性，如图 1 中虚线所示。FM 在潜在空间中学习良好的结构化数据表示，有助于任何进一步模型的构建。然而，FM 的乘积规则和 DNN 的求和规则在组合方式上存在细微差异。不过，根据[21]，如果观测到的判别信息高度模糊（在我们的广告点击行为场景中确实如此），后验权重（来自 DNN）不会显著偏离先验（FM）。

此外，隐藏层（FM 层除外）的权重通过使用对比散度[17]的逐层 RBM 预训练[3]初始化，这有效地保留了输入数据集中的信息，详见[18, 16]。FM 的初始权重通过随机梯度下降（SGD，Stochastic Gradient Descent）训练，详见[31]。注意，我们只需要更新连接到正例输入单元的权重，这大大降低了计算复杂度。在 FM 和上层预训练之后，应用有监督微调（反向传播）来最小化交叉熵损失函数：

$$
L(y, \hat{y}) = -y \log \hat{y} - (1 - y) \log(1 - \hat{y}) \qquad (7)
$$

其中 $\hat{y}$ 是公式 (1) 中预测的 CTR， $y$ 是二值点击真实标签。使用反向传播的链式法则，可以高效更新 FNN 权重（包括 FM 权重）。例如，我们通过以下方式更新 FM 层权重：

$$
\frac{\partial L(y, \hat{y})}{\partial \mathbf{W}_i^0} = \frac{\partial L(y, \hat{y})}{\partial \mathbf{z}_i} \frac{\partial \mathbf{z}_i}{\partial \mathbf{W}_i^0} = \frac{\partial L(y, \hat{y})}{\partial \mathbf{z}_i} \mathbf{x}[\mathrm{start}_i : \mathrm{end}_i] \qquad (8)
$$

$$
\mathbf{W}_i^0 \leftarrow \mathbf{W}_i^0 - \eta \cdot \frac{\partial L(y, \hat{y})}{\partial \mathbf{z}_i} \mathbf{x}[\mathrm{start}_i : \mathrm{end}_i] \qquad (9)
$$

由于 $\mathbf{x}[\mathrm{start}_i : \mathrm{end}_i]$ 的大部分条目为 0，我们可以通过仅更新连接到正例单元的权重来加速微调。

### 3.2 采样神经网络（SNN）

第二个模型 SNN 的结构如图 2(a) 所示。SNN 与 FNN 的区别在于底层的结构和训练方法。SNN 的底层使用 sigmoid 激活函数进行全连接：

$$
\mathbf{z} = \mathrm{sigmoid}(\mathbf{W}^0 \mathbf{x} + \mathbf{b}^0) \qquad (10)
$$

为了初始化底层的权重，我们尝试了受限玻尔兹曼机（RBM）[16]和去噪自编码器（DAE）[4]进行预训练阶段的初始化。为了处理大规模稀疏独热编码数据的计算问题，我们提出了基于采样的 RBM（图 2(b)，记为 SNN-RBM）和基于采样的 DAE（图 2(c)，记为 SNN-DAE）来高效计算底层的初始权重。

> **图 2：4 层 SNN 架构及两种第一层预训练方法。**

不是为每个训练实例集建模整个特征集，而是对每个特征领域（例如城市），每个训练实例只有一个正例值特征（例如 city=London），我们采样 $m$ 个负例单元（例如当 $m = 1$ 时 city=Paris），随机取值为 0。图 2(b) 和 2(c) 中的黑色单元未被采样，因此在预训练该数据实例时被忽略。使用采样后的单元，我们可以通过对比散度[17]训练 RBM，通过 SGD 以无监督方式训练 DAE，以大幅降低数据维度并保持高恢复性能。得到的实值稠密向量用作 SNN 后续层的输入。

通过这种方式，计算复杂度可以大幅降低，初始权重可以快速计算，然后执行反向传播来微调 SNN 模型。

### 3.3 正则化

为了防止过拟合，广泛使用的 L2 正则化项被添加到损失函数中。例如，图 1 中 FNN 的 L2 正则化为：

$$
\Omega(w) = \|\mathbf{W}^0\|_2^2 + \sum_{l=1}^{3} \left(\|\mathbf{W}^l\|_2^2 + \|\mathbf{b}^l\|_2^2\right) \qquad (11)
$$

另一方面，Dropout[35]是近年来成为深度学习中流行且有效的正则化技术。我们也在实验中实现了该正则化并进行了比较。

## 4 实验

### 4.1 实验设置

**数据。** 我们基于 iPinYou 数据集[27]评估模型，这是一个公开的真实展示广告数据集，包含每次广告展示信息和相应的用户点击反馈。数据日志按不同广告商组织，采用逐行记录格式。共有 1950 万个数据实例，其中 14790 个正例标签（点击）。每个数据实例的特征均为分类特征。广告日志数据中的特征示例包括用户代理、部分遮蔽 IP、区域、城市、广告交换平台、域名、URL、广告位 ID、广告位可见性、广告位尺寸、广告位格式、创意 ID、用户标签等。经过独热编码后，整个数据集中的二值特征数量为 937670 个。我们将这些二值特征数据实例和用户点击（1）与未点击（0）反馈作为真实标签输入各比较模型。在实验中，我们分别使用广告商 1458、2259、2261、2997、3386 以及整个数据集的训练数据。

**模型。** 我们比较以下 CTR 估计模型的性能：

* **LR：** 逻辑回归（LR）[32]是一种线性模型，实现简单、训练速度快，广泛用于在线广告估计。
* **FM：** 分解机（FM）[31]是一种非线性模型，即使在高稀疏问题中也能估计特征交互。
* **FNN：** 分解机支持神经网络（FNN）是我们 3.1 节提出的模型。
* **SNN：** 采样神经网络（SNN）也是我们提出的模型，使用基于采样的 RBM 和 DAE 预训练方法初始化第一层，分别记为 SNN-RBM 和 SNN-DAE，见 3.2 节。

> **图 1：4 层 FNN 模型结构。**

> **图 2：4 层 SNN 架构及两种第一层预训练方法。**

> **图 3：不同架构下的 AUC 性能。**

> **图 4：不同正则化设置下的 AUC 性能。**

**表 1：整体 CTR 估计 AUC 性能。**

| 广告商 | LR | FM | FNN | SNN-DAE | SNN-RBM |
|--------|------|------|------|---------|---------|
| 1458 | 70.42% | 70.21% | 70.52% | 70.46% | 70.49% |
| 2259 | 69.66% | 69.73% | 69.74% | 68.08% | 68.34% |
| 2261 | 62.03% | 60.97% | 62.99% | 63.72% | 63.72% |
| 2997 | 60.77% | 60.87% | 61.41% | 61.58% | 61.45% |
| 3386 | 80.30% | 79.05% | 80.56% | 79.62% | 80.07% |
| 整体 | 68.81% | 68.18% | 70.70% | 69.15% | 69.15% |

我们的实验代码[‡] 使用 Theano[§] 实现。**评估指标。** 为了衡量每个模型的 CTR 估计性能，我们采用 ROC 曲线下面积（AUC）[¶] 指标。AUC[12]是广泛用于评估 CTR 性能的指标。

> [‡] 源代码及演示数据：https://github.com/wnzhang/deep-ctr
>
> [§] Theano：http://deeplearning.net/software/theano/
>
> [¶] 除了 AUC，还测试了均方根误差（RMSE）。然而，广告点击场景中正负样本极度不平衡，经验上最优的回归模型通常提供接近 0 的预测 CTR，导致 RMSE 值非常小，因此改进不易被捕捉。

### 4.2 性能比较

表 1 展示了 LR、FM、FNN 和 SNN（使用 RBM 和 DAE）在 5 个不同广告商和整个数据集上的比较结果。我们观察到 FM 并不显著优于 LR，这意味着二阶组合特征可能不足以捕获底层数据模式。所提出的 FNN 和 SNN 的 AUC 性能在所有测试数据集上均优于 LR 和 FM。基于 FM 学习的潜在结构，FNN 进一步学习了这些潜在特征之间的有效模式，并在 FM 基础上提供了一致的改进。SNN-DAE 和 SNN-RBM 的性能总体一致，即 SNN 结果的相对顺序几乎相同。

### 4.3 超参数调优

由于深度神经网络涉及许多实现细节，需要调整相当多的超参数，以下细节展示了我们如何实现模型并调整模型中的超参数。

我们使用随机梯度下降（SGD）来学习所有提出模型的大部分参数。关于训练轮数的选择，我们使用早停[30]，即当验证误差增加时停止训练。我们尝试了 1、0.1、0.01、0.001 到 0.0001 的不同学习率，并选择在验证数据集上性能最优的值。

对于 SNN-RBM 和 SNN-DAE 的负例单元采样，我们尝试了 3.2 节中描述的每个领域 $m = 1, 2$ 和 4 个负样本，发现 $m = 2$ 在大多数情况下产生最佳结果。对于两个模型隐藏层的激活函数（如公式 (3) 和 (2)），我们尝试了线性函数、sigmoid 函数和 $\tanh$ 函数，发现 $\tanh$ 函数的结果最优。这可能是因为双曲正切函数通常比 sigmoid 函数收敛更快。

### 4.4 架构选择

在我们的模型中，我们通过固定所有层大小来研究 3、4 和 5 个隐藏层的架构，发现具有 3 个隐藏层（即总共 5 层）的架构在 AUC 性能方面最优。然而，选择隐藏层大小的范围随隐藏层数量呈指数增长。假设一个具有 $L$ 个隐藏层的深度神经网络，每个隐藏层的隐藏单元数从 100 到 500 以 100 为增量进行训练，那么总共有 $5^L$ 个模型需要比较。

> **图 3：不同架构下的 AUC 性能。**

不是尝试所有隐藏单元的组合，我们在实验中使用另一种策略，即从所有三个隐藏层中具有相同数量隐藏单元[††] 的设置开始调优不同隐藏层大小，因为根据[24]，等宽隐藏层架构在经验上优于递增或递减宽度的架构。因此，我们从等宽隐藏层大小开始调优。事实上，除了递增、恒定和递减的层大小外，还有一种更有效的结构——菱形神经网络，如图 3(a) 所示。我们将菱形网络与其他三种形状的网络进行比较，并在两个不同数据集上调整总隐藏单元数，如图 3(b) 和 3(c) 所示。菱形架构在几乎所有层级设置下均优于其他架构。菱形架构之所以有效，可能是因为这种特殊形状的神经网络对网络容量有一定的约束，从而在测试集上提供更好的泛化能力。另一方面，菱形架构的性能在总隐藏单元数为 600 时达到峰值，即 (200, 300, 100) 的组合。这取决于训练数据的观测数量。过多的隐藏单元配以有限的数据集可能导致过拟合。

> [††] 一些先进的贝叶斯超参数调优方法[34]在本文中未被考虑，可能在未来工作中研究。

### 4.5 正则化比较

神经网络训练算法对过拟合问题非常敏感，因为深度网络具有多个非线性层，这使它们成为能够学习非常复杂函数的高表达力模型。对于 DNN 模型，我们比较了 L2 正则化（公式 (11)）和 Dropout[35] 在防止训练数据上复杂共适应方面的效果。本实验中实现的 Dropout 率指的是每个单元被激活的概率。

> **图 4：不同正则化设置下的 AUC 性能。**

图 4(a) 展示了 SNN-RBM 使用 L2 范数和 Dropout 正则化的 AUC 性能比较。显然，Dropout 在所有比较设置中均优于 L2。Dropout 更有效的原因在于，当输入每个训练样本时，每个隐藏单元以一定的 Dropout 率被随机排除在网络之外，即每个训练样本可以被视为一个新模型，这些模型被平均后作为 Bagging[5] 的特例，有效地提高了 DNN 模型的泛化能力。

### 4.6 参数分析

作为 4.4 和 4.5 节的总结，对于 FNN 和 SNN，有两个重要参数需要调优以使模型更有效：（i）层大小参数决定神经网络的架构；（ii）Dropout 率参数改变了与仅使用 L2 正则化的神经网络相比在所有数据集上的泛化能力。

图 4(b) 和 4(c) 展示了 FNN 和 SNN 中 AUC 性能随 Dropout 率增加的变化。我们可以发现，两个模型在开始时性能呈上升趋势，然后随 Dropout 率持续降低而急剧下降。两个模型的区别在于对 Dropout 的不同敏感度。从图 4(c) 可以看出，SNN 模型对 Dropout 率敏感。这可能是由于底层的连接性造成的。SNN 的底层与输入向量全连接，而 FNN 的底层是部分连接的，因此当某些隐藏单元被丢弃时，FNN 更具鲁棒性。此外，在 Dropout 方面，sigmoid 激活函数往往比线性激活函数更有效。因此，FNN 和 SNN 达到最佳性能时的 Dropout 率差异很大。FNN 的最优 Dropout 率约为 0.8，而 SNN 约为 0.99。

## 5 结论

在本文中，我们研究了训练深度神经网络（DNN）基于多领域分类特征预测用户广告点击响应的潜力。为了处理高维离散分类特征的计算复杂度问题，我们提出了两种 DNN 模型：使用有监督分解机预训练的按领域特征嵌入，以及使用按领域采样的 RBM 和 DAE 无监督预训练的全连接 DNN。这些架构和预训练算法使我们的 DNN 能够高效训练。在公开真实数据集上的综合实验验证了所提出的 DNN 模型成功学习了底层数据模式，并提供了优于其他比较模型的 CTR 估计性能。所提出的模型非常通用，可以支持广泛的未来工作。例如，可以通过动量方法改进模型性能，因为它足以处理 DNN 训练目标中的曲率问题，无需使用复杂的二阶方法[36]。此外，底层的部分连接可以扩展到更高的隐藏层，因为部分连接具有许多优势，如更低的复杂度、更高的泛化能力以及更接近人脑[9]。

---

## 参考文献

[1] Beck, J.E., Woolf, B.P.: High-level student modeling with machine learning. In: Intelligent tutoring systems. pp. 584–593. Springer (2000)

[2] Bengio, Y.: Learning deep architectures for AI. Foundations and trends in Machine Learning 2(1), 1–127 (2009)

[3] Bengio, Y., Lamblin, P., Popovici, D., Larochelle, H., et al.: Greedy layer-wise training of deep networks. NIPS 19, 153 (2007)

[4] Bengio, Y., Yao, L., Alain, G., Vincent, P.: Generalized denoising auto-encoders as generative models. In: NIPS. pp. 899–907 (2013)

[5] Breiman, L.: Bagging predictors. Machine Learning 24(2), 123–140 (1996)

[6] Broder, A.Z.: Computational advertising. In: SODA. vol. 8, pp. 992–992 (2008)

[7] Collobert, R., Weston, J., Bottou, L., Karlen, M., Kavukcuoglu, K., Kuksa, P.: Natural language processing (almost) from scratch. JMLR 12, 2493–2537 (2011)

[8] Deng, L., Abdel-Hamid, O., Yu, D.: A deep convolutional neural network using heterogeneous pooling for trading acoustic invariance with phonetic confusion. In: ICASSP. pp. 6669–6673. IEEE (2013)

[9] Elizondo, D., Fiesler, E.: A survey of partially connected neural networks. International Journal of Neural Systems 8(05n06), 535–558 (1997)

[10] Erhan, D., Bengio, Y., Courville, A., Manzagol, P.A., Vincent, P., Bengio, S.: Why does unsupervised pre-training help deep learning? JMLR 11 (2010)

[11] Fukushima, K.: Neocognitron: A self-organizing neural network model for a mechanism of pattern recognition unaffected by shift in position. Biological Cybernetics 36(4), 193–202 (1980)

[12] Graepel, T., Candela, J.Q., Borchert, T., Herbrich, R.: Web-scale bayesian click-through rate prediction for sponsored search advertising in microsoft's bing search engine. In: ICML. pp. 13–20 (2010)

[13] Graves, A., Mohamed, A.r., Hinton, G.: Speech recognition with deep recurrent neural networks. In: ICASSP. pp. 6645–6649. IEEE (2013)

[14] Hand, D.J., Yu, K.: Idiot's bayes — not so stupid after all? International Statistical Review 69(3), 385–398 (2001)

[15] He, X., Pan, J., Jin, O., Xu, T., Liu, B., Xu, T., Shi, Y., Atallah, A., Herbrich, R., Bowers, S., et al.: Practical lessons from predicting clicks on ads at facebook. In: ADKDD. pp. 1–9. ACM (2014)

[16] Hinton, G.: A practical guide to training restricted boltzmann machines. Momentum 9(1), 926 (2010)

[17] Hinton, G.E.: Training products of experts by minimizing contrastive divergence. Neural Computation 14(8), 1771–1800 (2002)

[18] Hinton, G.E., Salakhutdinov, R.R.: Reducing the dimensionality of data with neural networks. Science 313(5786), 504–507 (2006)

[19] Huang, P.S., He, X., Gao, J., Deng, L., Acero, A., Heck, L.: Learning deep structured semantic models for web search using clickthrough data. In: CIKM. pp. 2333–2338 (2013)

[20] Juan, Y.C., Zhuang, Y., Chin, W.S.: 3 idiots approach for display advertising challenge. In: Internet and Network Economics, pp. 254–265. Springer (2011)

[21] Kittler, J., Hatef, M., Duin, R.P., Matas, J.: On combining classifiers. PAMI 20(3), 226–239 (1998)

[22] Krizhevsky, A., Sutskever, I., Hinton, G.E.: Imagenet classification with deep convolutional neural networks. In: NIPS (2012)

[23] Kurashima, T., Iwata, T., Takaya, N., Sawada, H.: Probabilistic latent network visualization: inferring and embedding diffusion networks. In: KDD. pp. 1236–1245. ACM (2014)

[24] Larochelle, H., Bengio, Y., Louradour, J., Lamblin, P.: Exploring strategies for training deep neural networks. JMLR 10, 1–40 (2009)

[25] LeCun, Y., Bengio, Y., Hinton, G.: Deep learning. Nature 521(7553) (2015)

[26] Lee, K.c., Orten, B., Dasdan, A., Li, W.: Estimating conversion rate in display advertising from past performance data. In: KDD. pp. 768–776. ACM (2012)

[27] Liao, H., Peng, L., Liu, Z., Shen, X.: ipinyou global rtb bidding algorithm competition dataset. In: ADKDD. pp. 1–6. ACM (2014)

[28] McMahan, H.B., Holt, G., Sculley, D., Young, M., Ebner, D., Grady, J., Nie, L., Phillips, T., Davydov, E., Golovin, D., et al.: Ad click prediction: a view from the trenches. In: KDD. pp. 1222–1230. ACM (2013)

[29] Oentaryo, R.J., Lim, E.P., Low, D.J.W., Lo, D., Finegold, M.: Predicting response in mobile advertising with hierarchical importance-aware factorization machine. In: WSDM (2014)

[30] Prechelt, L.: Automatic early stopping using cross validation: quantifying the criteria. Neural Networks 11(4), 761–767 (1998)

[31] Rendle, S.: Factorization machines with libfm. ACM TIST 3(3), 57 (2012)

[32] Richardson, M., Dominowska, E., Ragno, R.: Predicting clicks: estimating the click-through rate for new ads. In: WWW. pp. 521–530. ACM (2007)

[33] Shen, Y., He, X., Gao, J., Deng, L., Mesnil, G.: A latent semantic model with convolutional-pooling structure for information retrieval. In: CIKM (2014)

[34] Snoek, J., Larochelle, H., Adams, R.P.: Practical bayesian optimization of machine learning algorithms. In: NIPS. pp. 2951–2959 (2012)

[35] Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., Salakhutdinov, R.: Dropout: A simple way to prevent neural networks from overfitting. JMLR 15(1), 1929–1958 (2014)

[36] Sutskever, I., Martens, J., Dahl, G., Hinton, G.: On the importance of initialization and momentum in deep learning. In: ICML. pp. 1139–1147 (2013)

[37] Tang, J., Qu, M., Wang, M., Zhang, M., Yan, J., Mei, Q.: Line: Large-scale information network embedding. In: WWW. pp. 1067–1077 (2015)

[38] Trofimov, I., Kornetova, A., Topinskiy, V.: Using boosted trees for click-through rate prediction for sponsored search. In: WINE. p. 2. ACM (2012)

[39] Wang, X., Li, W., Cui, Y., Zhang, R., Mao, J.: Click-through rate estimation for rare events in online advertising. Online Multimedia Advertising: Techniques and Technologies pp. 1–12 (2010)

[40] Zeiler, M.D., Taylor, G.W., Fergus, R.: Adaptive deconvolutional networks for mid and high level feature learning. In: ICCV. pp. 2018–2025. IEEE (2011)

[41] Zhang, W., Yuan, S., Wang, J.: Optimal real-time bidding for display advertising. In: KDD. pp. 1077–1086. ACM (2014)

[42] Zou, Y., Jin, X., Li, Y., Guo, Z., Wang, E., Xiao, B.: Mariana: Tencent deep learning platform and its applications. VLDB 7(13), 1772–1777 (2014)
