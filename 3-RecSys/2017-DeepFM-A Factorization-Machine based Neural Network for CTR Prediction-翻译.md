# DeepFM: A Factorization-Machine based Neural Network for CTR Prediction（中文翻译）

> 翻译说明：本文为 DeepFM 论文的完整中文翻译，保留原论文所有章节、公式、图表及参考文献。专业术语遵循约定译法。

---

## 摘要

学习用户行为背后复杂的特征交互，对于最大化推荐系统的点击率（CTR）至关重要。尽管已有诸多进展，现有方法仍存在偏向低阶或高阶交互的明显倾向，或依赖专业的特征工程。本文证明，可以构建一个同时强调低阶和高阶特征交互的端到端学习模型。所提出的模型 DeepFM 采用全新的神经网络架构，融合了因子分解机在推荐方面的能力和深度学习在特征学习方面的优势。与谷歌最新的 Wide & Deep 模型相比，DeepFM 的"宽"部分和"深"部分共享同一输入，除原始特征外无需任何特征工程。我们在基准数据集和商业数据上进行了全面的实验，以证明 DeepFM 在 CTR 预测任务上相较于现有模型的有效性和高效性。

## 1 引言

点击率（CTR）预估是推荐系统中的核心任务，其目标是估计用户点击推荐物品的概率。在许多推荐系统中，目标是最大化点击量，因此返回给用户的物品可以按预估 CTR 排序；而在在线广告等其他应用场景中，提升收益同样重要，因此排序策略可以调整为所有候选物品的 CTR×出价，其中"出价"是用户点击物品后系统获得的收益。无论哪种情况，关键都在于准确预估 CTR。

学习用户点击行为背后隐含的特征交互对于 CTR 预测至关重要。通过对主流应用市场的研究，我们发现用户常在用餐时间下载外卖类应用，这表明应用类别与时间戳之间的（二阶）交互可以作为 CTR 预测的信号。第二个观察是，男性青少年喜欢射击类和角色扮演类游戏，这意味着应用类别、用户性别和年龄的（三阶）交互是 CTR 的另一个信号。总的来说，用户点击行为背后的特征交互可能非常复杂，其中低阶和高阶特征交互都应发挥重要作用。根据谷歌 Wide & Deep 模型 [Cheng et al., 2016] 的见解，同时考虑低阶和高阶特征交互比单独考虑其中任何一种都能带来额外的改进。

关键挑战在于如何有效地建模特征交互。某些特征交互易于理解，因此可以由专家设计（如上文实例）。然而，大多数其他特征交互隐藏在数据中且难以先验地识别（例如经典的关联规则"尿布和啤酒"是从数据中挖掘出来的，而非专家发现），这些交互只能通过机器学习自动捕获。即使对于易于理解的交互，专家也很难穷尽地建模，尤其是在特征数量庞大的情况下。

尽管广义线性模型（如 FTRL [McMahan et al., 2013]）结构简单，但在实践中已展现出良好的性能。然而，线性模型缺乏学习特征交互的能力，通常的做法是在其特征向量中手动加入成对特征交互。这种方法难以泛化到建模高阶特征交互或那些在训练数据中从未出现或极少出现的交互 [Rendle, 2010]。因子分解机（FM）[Rendle, 2010] 将成对特征交互建模为特征之间隐向量的内积，并展现出非常有前景的结果。虽然原则上 FM 可以建模高阶特征交互，但由于高复杂度，实践中通常只考虑二阶特征交互。

作为一种学习特征表示的强大方法，深度神经网络具有学习复杂特征交互的潜力。一些研究将 CNN 和 RNN 扩展到 CTR 预测 [Liu et al., 2015; Zhang et al., 2014]，但基于 CNN 的模型偏向于相邻特征之间的交互，而基于 RNN 的模型更适合具有序列依赖性的点击数据。[Zhang et al., 2016] 研究了特征表示，并提出因子分解机支持的神经网络（FNN）。该模型在应用 DNN 之前预训练 FM，因此受限于 FM 的能力。[Qu et al., 2016] 通过在嵌入层和全连接层之间引入乘积层，提出了基于乘积的神经网络（PNN），从而研究了特征交互。正如 [Cheng et al., 2016] 所指出的，PNN 和 FNN 与其他深度模型一样，几乎不捕获对 CTR 预测同样重要的低阶特征交互。为了同时建模低阶和高阶特征交互，[Cheng et al., 2016] 提出了一种有趣的混合网络结构（Wide & Deep），结合了线性（"宽"）模型和深度模型。在该模型中，"宽部分"和"深部分"分别需要两种不同的输入，且"宽部分"的输入仍然依赖专业的特征工程。

可以看出，现有模型偏向低阶或高阶特征交互，或依赖特征工程。在本文中，我们证明可以构建一个能够以端到端方式学习所有阶次特征交互的学习模型，除原始特征外无需任何特征工程。我们的主要贡献总结如下：

- 我们提出了一种新的神经网络模型 DeepFM（图 1），它融合了 FM 和深度神经网络（DNN）的架构。它像 FM 一样建模低阶特征交互，像 DNN 一样建模高阶特征交互。与 Wide & Deep 模型 [Cheng et al., 2016] 不同，DeepFM 无需任何特征工程即可进行端到端训练。
- DeepFM 可以高效训练，因为其宽部分和深部分与 [Cheng et al., 2016] 不同，共享相同的输入和嵌入向量。在 [Cheng et al., 2016] 中，由于宽部分的输入向量中包含了手动设计的成对特征交互，输入向量可能非常巨大，这也极大地增加了其复杂度。
- 我们在基准数据和商业数据上评估了 DeepFM，结果显示其在 CTR 预测任务上持续优于现有模型。

## 2 我们的方法

假设训练数据集包含 n 个实例 (χ, y)，其中 χ 是一个 m 字段的数据记录，通常涉及一对用户和物品，y ∈ {0, 1} 是关联的标签，指示用户点击行为（y = 1 表示用户点击了物品，y = 0 表示未点击）。χ 可能包含类别字段（如性别、位置）和连续字段（如年龄）。每个类别字段表示为独热编码向量，每个连续字段表示为数值本身或离散化后的独热编码向量。然后，每个实例被转换为 (x, y)，其中 x = [x_{field1}, x_{field2}, ..., x_{fieldj}, ..., x_{fieldm}] 是一个 d 维向量，x_{fieldj} 是 χ 的第 j 个字段的向量表示。通常，x 是高维且极其稀疏的。CTR 预测的任务是构建一个预测模型 ŷ = CTR_model(x)，以估计用户在给定上下文中点击特定应用的概率。

### 2.1 DeepFM

我们的目标是同时学习低阶和高阶特征交互。为此，我们提出了一种基于因子分解机的神经网络（DeepFM）。如图 1 所示，DeepFM 由两个组件组成：FM 组件和深度组件，它们共享相同的输入。对于特征 i，标量 w_i 用于衡量其一阶重要性，隐向量 V_i 用于衡量其与其他特征交互的影响。V_i 被输入 FM 组件以建模二阶特征交互，并输入深度组件以建模高阶特征交互。所有参数，包括 w_i、V_i 以及网络参数（下文中的 W^(l)、b^(l)），针对组合预测模型进行联合训练：

$$ \hat{y} = \text{sigmoid}(y_{FM} + y_{DNN}) \quad (1) $$

其中 ŷ ∈ (0, 1) 是预测的 CTR，y_{FM} 是 FM 组件的输出，y_{DNN} 是深度组件的输出。

**FM 组件**

图 2：FM 的架构。

FM 组件是一个因子分解机，由 [Rendle, 2010] 提出用于学习推荐中的特征交互。除了特征之间的线性（一阶）交互外，FM 还将成对（二阶）特征交互建模为相应特征隐向量的内积。与以往方法相比，它能够更有效地捕获二阶特征交互，尤其是在数据集稀疏的情况下。在以往的方法中，特征 i 和 j 的交互参数仅在特征 i 和 j 同时出现在同一条数据记录中时才能被训练。而在 FM 中，它是通过隐向量 V_i 和 V_j 的内积来衡量的。得益于这种灵活的设计，FM 可以在 i（或 j）出现在数据记录中时训练隐向量 V_i（V_j）。因此，在训练数据中从未出现或极少出现的特征交互能被 FM 更好地学习。

如图 2 所示，FM 的输出是一个加法单元和多个内积单元的总和：

$$ y_{FM} = \langle w, x \rangle + \sum_{j_1=1}^{d} \sum_{j_2=j_1+1}^{d} \langle V_i, V_j \rangle x_{j_1} \cdot x_{j_2} \quad (2) $$

其中 w ∈ R^d，V_i ∈ R^k（k 为给定值）。加法单元（⟨w, x⟩）反映了一阶特征的重要性，内积单元代表了二阶特征交互的影响。

**深度组件**

图 3：DNN 的架构。

深度组件是一个前馈神经网络，用于学习高阶特征交互。如图 3 所示，数据记录（向量）被输入神经网络。与以图像 [He et al., 2016] 或音频 [Boulanger-Lewandowski et al., 2013] 数据为输入的神经网络相比（这些输入是纯连续且稠密的），CTR 预测的输入有很大不同，这需要新的网络架构设计。具体而言，CTR 预测的原始特征输入向量通常是高度稀疏、超高维、类别-连续混合且按字段分组的（如性别、位置、年龄）。这意味着在将输入向量送入第一个隐藏层之前，需要一个嵌入层将其压缩为低维、稠密的实值向量，否则网络训练将变得非常困难。

图 4 展示了从输入层到嵌入层的子网络结构。我们想指出该网络结构的两个有趣特点：1）不同输入字段向量的长度可以不同，但它们的嵌入具有相同的大小（k）；2）FM 中的隐特征向量（V）现在作为网络权重，通过学习和训练来将输入字段向量压缩为嵌入向量。在 [Zhang et al., 2016] 中，V 由 FM 预训练并用作初始化。在本文中，我们不是像 [Zhang et al., 2016] 那样使用 FM 的隐特征向量来初始化网络，而是将 FM 模型作为整个学习架构的一部分，与另一个 DNN 模型并列。因此，我们消除了 FM 预训练的需要，而是以端到端的方式联合训练整个网络。将嵌入层的输出记为：

$$ a^{(0)} = [e_1, e_2, ..., e_m] \quad (3) $$

其中 e_i 是第 i 个字段的嵌入，m 是字段数量。然后 a^{(0)} 被送入深度神经网络，前向过程为：

$$ a^{(l+1)} = \sigma(W^{(l)} a^{(l)} + b^{(l)}) \quad (4) $$

其中 l 是层深度，σ 是激活函数。a^{(l)}、W^{(l)}、b^{(l)} 分别是第 l 层的输出、模型权重和偏置。之后，生成一个稠密的实值特征向量，最终被送入 sigmoid 函数进行 CTR 预测：y_{DNN} = σ(W^{(|H|+1)} · a^H + b^{(|H|+1)})，其中 |H| 是隐藏层的数量。

值得指出的是，FM 组件和深度组件共享相同的特征嵌入，这带来了两个重要好处：1）它从原始特征中同时学习低阶和高阶特征交互；2）不需要像 Wide & Deep [Cheng et al., 2016] 那样对输入进行专业的特征工程。

### 2.2 与其他神经网络的关系

受深度学习在各种应用中巨大成功的启发，近年来针对 CTR 预测开发了几种深度模型。本节将提出的 DeepFM 与现有的 CTR 预测深度模型进行比较。

**FNN**：如图 5（左）所示，FNN 是一个由 FM 初始化的前馈神经网络 [Zhang et al., 2016]。FM 预训练策略导致两个局限性：1）嵌入参数可能受到 FM 的过度影响；2）预训练阶段引入的开销降低了效率。此外，FNN 仅捕获高阶特征交互。相比之下，DeepFM 不需要预训练，同时学习高阶和低阶特征交互。

**PNN**：为了捕获高阶特征交互，PNN 在嵌入层和第一个隐藏层之间引入了一个乘积层 [Qu et al., 2016]。根据乘积操作类型的不同，有三种变体：IPNN、OPNN 和 PNN∗，其中 IPNN 基于向量内积，OPNN 基于外积，PNN∗ 同时基于内积和外积。

图 5：用于 CTR 预测的现有深度模型的架构：FNN、PNN、Wide & Deep 模型

表 1：用于 CTR 预测的深度模型比较

| 模型 | 低阶特征 | 高阶特征 | 无需预训练 | 无需特征工程 |
|------|---------|---------|-----------|------------|
| FNN | × | √ | × | √ |
| PNN | × | √ | √ | √ |
| Wide & Deep | √ | √ | √ | × |
| DeepFM | √ | √ | √ | √ |

为了提高计算效率，作者提出了内积和外积的近似计算方法：1）内积通过消除部分神经元进行近似计算；2）外积通过将 m 个 k 维特征向量压缩为一个 k 维向量进行近似计算。然而，我们发现外积的可靠性低于内积，因为外积的近似计算丢失了大量信息，导致结果不稳定。尽管内积更可靠，它仍然面临高计算复杂度的困扰，因为乘积层的输出连接到第一个隐藏层的所有神经元。与 PNN 不同，DeepFM 中乘积层的输出仅连接到最终输出层（一个神经元）。与 FNN 一样，所有 PNN 都忽略了低阶特征交互。

**Wide & Deep**：Wide & Deep（图 5（右））由 Google 提出，用于同时建模低阶和高阶特征交互。如 [Cheng et al., 2016] 所示，"宽"部分的输入需要专业的特征工程（例如，在应用推荐中用户安装应用与曝光应用的交叉积）。相比之下，DeepFM 通过直接从输入原始特征学习，无需此类专业知识来处理输入。

该模型的一个直接扩展是用 FM 替换 LR（我们也在第 3 节中评估了这一扩展）。这种扩展与 DeepFM 类似，但 DeepFM 在 FM 和深度组件之间共享特征嵌入。特征嵌入的共享策略以反向传播的方式，通过低阶和高阶特征交互共同影响特征表示，从而使表示建模更加精确。

**总结**：概括而言，DeepFM 与其他深度模型在四个方面的关系如表 1 所示。可以看出，DeepFM 是唯一一个不需要预训练和特征工程，同时能捕获低阶和高阶特征交互的模型。

## 3 实验

在本节中，我们将提出的 DeepFM 与其他最先进模型进行实证比较。评估结果表明，我们提出的 DeepFM 比任何其他最先进模型都更有效，且 DeepFM 的效率可与其中最优者相媲美。

### 3.1 实验设置

**数据集**

我们在以下两个数据集上评估 DeepFM 的有效性和高效性。

1) **Criteo 数据集**：Criteo 数据集包含 4500 万用户的点击记录。其中有 13 个连续特征和 26 个类别特征。我们将数据集随机分为两部分：90% 用于训练，剩余 10% 用于测试。

2) **公司∗数据集**：为验证 DeepFM 在实际工业 CTR 预测中的性能，我们在公司∗数据集上进行了实验。我们从公司∗应用商店的游戏中心收集了连续 7 天的用户点击记录用于训练，接下 1 天的数据用于测试。整个收集的数据集中约有 10 亿条记录。该数据集中包含应用特征（如标识、类别等）、用户特征（如用户下载的应用等）和上下文特征（如操作时间等）。

**评估指标**

我们在实验中使用了两个评估指标：AUC（ROC 曲线下面积）和 Logloss（交叉熵损失）。

**模型对比**

我们在实验中比较了 9 个模型：LR、FM、FNN、PNN（三个变体）、Wide & Deep 和 DeepFM。在 Wide & Deep 模型中，为消除特征工程的工作量，我们还将原始的 Wide & Deep 模型进行了调整，用 FM 替换 LR 作为宽部分。为区分这两个 Wide & Deep 变体，我们分别称之为 LR & DNN 和 FM & DNN。

**参数设置**

为了在 Criteo 数据集上评估模型，我们遵循 [Qu et al., 2016] 中对 FNN 和 PNN 的参数设置：（1）dropout：0.5；（2）网络结构：400-400-400；（3）优化器：Adam；（4）激活函数：IPNN 使用 tanh，其他深度模型使用 relu。为公平起见，我们提出的 DeepFM 使用相同的设置。LR 和 FM 的优化器分别为 FTRL 和 Adam，FM 的隐向量维度为 10。为了在公司∗数据集上使每个模型达到最佳性能，我们进行了仔细的参数研究，详见第 3.3 节。

### 3.2 性能评估

在本节中，我们评估第 3.1 节中列出的模型在两个数据集上的效果和效率。

**效率对比**

深度学习模型的效率对实际应用非常重要。我们通过以下公式比较不同模型在 Criteo 数据集上的效率：|深度 CTR 模型的训练时间| / |LR 的训练时间|。结果如图 6 所示，包括在 CPU（左）和 GPU（右）上的测试，我们有以下观察：1）FNN 的预训练使其效率较低；2）尽管 IPNN 和 PNN∗ 在 GPU 上的加速比高于其他模型，但由于内积操作效率低下，它们仍然计算开销很大；3）DeepFM 在两种测试中几乎都达到了最高效率。

图 6：时间对比。

**效果对比**

不同模型在 Criteo 数据集和公司∗数据集上的 CTR 预测性能如表 2 所示，我们有以下观察：

- 学习特征交互提升了 CTR 预测模型的性能。这一观察来自这样一个事实：LR（唯一不考虑特征交互的模型）表现比其他模型差。作为最佳模型，DeepFM 在公司∗和 Criteo 数据集上以 AUC 计分别超过 LR 0.86% 和 4.18%（以 Logloss 计分别超过 1.15% 和 5.60%）。
- 同时且适当地学习高低阶特征交互提升了 CTR 预测模型的性能。DeepFM 优于仅学习低阶特征交互（即 FM）或仅学习高阶特征交互（即 FNN、IPNN、OPNN、PNN∗）的模型。与第二优模型相比，DeepFM 在公司∗和 Criteo 数据集上以 AUC 计分别提高了 0.37% 和 0.25% 以上（以 Logloss 计分别提高了 0.42% 和 0.29% 以上）。
- 同时学习高低阶特征交互，且为高低阶特征交互学习共享相同的特征嵌入，提升了 CTR 预测模型的性能。DeepFM 优于使用独立特征嵌入学习高低阶特征交互的模型（即 LR & DNN 和 FM & DNN）。与这两个模型相比，DeepFM 在公司∗和 Criteo 数据集上以 AUC 计分别超过 0.48% 和 0.33%（以 Logloss 计分别超过 0.61% 和 0.66%）。

表 2：CTR 预测性能

| 模型 | Criteo AUC | Criteo LogLoss | 公司∗ AUC | 公司∗ LogLoss |
|------|-----------|---------------|-----------|--------------|
| LR | 0.8640 | 0.02648 | 0.7686 | 0.47762 |
| FM | 0.8678 | 0.02633 | 0.7892 | 0.46077 |
| FNN | 0.8683 | 0.02629 | 0.7963 | 0.45738 |
| IPNN | 0.8664 | 0.02637 | 0.7972 | 0.45323 |
| OPNN | 0.8658 | 0.02641 | 0.7982 | 0.45256 |
| PNN∗ | 0.8672 | 0.02636 | 0.7987 | 0.45214 |
| LR & DNN | 0.8673 | 0.02634 | 0.7981 | 0.46772 |
| FM & DNN | 0.8661 | 0.02640 | 0.7850 | 0.45382 |
| DeepFM | **0.8715** | **0.02618** | **0.8007** | **0.45083** |

总体而言，我们提出的 DeepFM 模型在公司∗数据集上以 AUC 和 Logloss 计分别超过竞争对手 0.37% 和 0.42% 以上。事实上，离线 AUC 评估的微小改进很可能带来在线 CTR 的显著提升。如 [Cheng et al., 2016] 所报道，与 LR 相比，Wide & Deep 将 AUC 提高了 0.275%（离线），在线 CTR 的改进为 3.9%。公司∗应用商店的日成交额达数百万美元，因此 CTR 即使几个百分点的提升每年也能带来额外的数百万美元收入。

### 3.3 超参数研究

我们研究了不同深度模型的不同超参数在公司∗数据集上的影响。顺序为：1）激活函数；2）dropout 率；3）每层神经元数量；4）隐藏层数量；5）网络形状。

**激活函数**

根据 [Qu et al., 2016]，relu 和 tanh 比 sigmoid 更适合深度模型。在本文中，我们比较了深度模型在应用 relu 和 tanh 时的性能。如图 7 所示，除 IPNN 外，relu 比 tanh 更适合所有深度模型。可能的原因是 relu 引入了稀疏性。

图 7：激活函数的 AUC 和 Logloss 对比。

**Dropout**

Dropout [Srivastava et al., 2014] 指神经元在网络中被保留的概率。Dropout 是一种正则化技术，用于在神经网络的精度和复杂度之间取得平衡。我们将 dropout 设置为 1.0、0.9、0.8、0.7、0.6、0.5。如图 8 所示，当 dropout 设置适当时（从 0.6 到 0.9），所有模型都能达到自身的最佳性能。结果表明，向模型添加合理的随机性可以增强模型的鲁棒性。

图 8：Dropout 的 AUC 和 Logloss 对比。

**每层神经元数量**

其他因素保持不变时，增加每层神经元数量会增加模型复杂度。从图 9 中我们可以观察到，增加神经元数量并不总能带来益处。例如，DeepFM 在每层神经元数量从 400 增加到 800 时表现稳定；更糟糕的是，OPNN 在每层神经元数量从 400 增加到 800 时表现变差。这是因为过于复杂的模型容易过拟合。在我们的数据集中，每层 200 或 400 个神经元是较好的选择。

图 9：神经元数量的 AUC 和 Logloss 对比。

**隐藏层数量**

如图 10 所示，增加隐藏层数量最初提升了模型的性能，但如果继续增加隐藏层数量，模型性能反而下降。这一现象同样是由于过拟合所致。

图 10：层数量的 AUC 和 Logloss 对比。

**网络形状**

我们测试了四种不同的网络形状：常数型、递增型、递减型和钻石型。在改变网络形状时，我们固定了隐藏层数量和神经元总数。例如，当隐藏层数为 3、神经元总数为 600 时，四种不同形状为：常数型（200-200-200）、递增型（100-200-300）、递减型（300-200-100）和钻石型（150-300-150）。从图 11 可以看出，"常数型"网络形状在经验上优于其他三种选择，这与先前的研究 [Larochelle et al., 2009] 一致。

图 11：网络形状的 AUC 和 Logloss 对比。

## 4 相关工作

本文提出了一种用于 CTR 预测的新型深度神经网络。最相关的领域是 CTR 预测和推荐系统中的深度学习。在本节中，我们讨论这两个领域的相关工作。

CTR 预测在推荐系统中扮演着重要角色 [Richardson et al., 2007; Juan et al., 2016; McMahan et al., 2013]。除了广义线性模型和 FM 之外，还有一些其他模型被提出用于 CTR 预测，如基于树的模型 [He et al., 2014]、基于张量的模型 [Rendle and Schmidt-Thieme, 2010]、支持向量机 [Chang et al., 2010] 和贝叶斯模型 [Graepel et al., 2010]。

另一个相关领域是推荐系统中的深度学习。在第 1 节和第 2.2 节中，已经提到了几种用于 CTR 预测的深度学习模型，因此我们不再赘述。除 CTR 预测外，推荐任务中还提出了若干深度学习模型（例如 [Covington et al., 2016; Salakhutdinov et al., 2007; van den Oord et al., 2013; Wu et al., 2016; Zheng et al., 2016; Wu et al., 2017; Zheng et al., 2017]）。[Salakhutdinov et al., 2007; Sedhain et al., 2015; Wang et al., 2015] 提出通过深度学习改进协同过滤。[Wang and Wang, 2014; van den Oord et al., 2013] 利用深度学习提取内容特征以提升音乐推荐性能。[Chen et al., 2016] 设计了一个深度学习网络，同时考虑展示广告的图像特征和基本特征。[Covington et al., 2016] 为 YouTube 视频推荐开发了一个两阶段深度学习框架。

## 5 结论

本文中，我们提出了 DeepFM——一种基于因子分解机的 CTR 预测神经网络，以克服现有模型的缺点并实现更好的性能。DeepFM 联合训练一个深度组件和一个 FM 组件。它通过以下优势获得性能提升：1）无需任何预训练；2）同时学习高阶和低阶特征交互；3）引入特征嵌入共享策略以避免特征工程。我们在两个真实世界数据集（Criteo 数据集和商业应用商店数据集）上进行了大量实验，以比较 DeepFM 与最先进模型的有效性和高效性。我们的实验结果表明：1）DeepFM 在两个数据集上的 AUC 和 Logloss 指标均优于最先进模型；2）DeepFM 的效率与最先进的深度模型中最高效者相当。

未来研究有两个有趣的方向。一是探索一些策略（如引入池化层）以增强学习最有用的高阶特征交互的能力。二是在 GPU 集群上训练 DeepFM 以解决大规模问题。

## 参考文献

[Boulanger-Lewandowski et al., 2013] Nicolas Boulanger-Lewandowski, Yoshua Bengio, and Pascal Vincent. Audio chord recognition with recurrent neural networks. In *ISMIR*, pages 335–340, 2013.

[Chang et al., 2010] Yin-Wen Chang, Cho-Jui Hsieh, Kai-Wei Chang, Michael Ringgaard, and Chih-Jen Lin. Training and testing low-degree polynomial data mappings via linear SVM. *JMLR*, 11:1471–1490, 2010.

[Chen et al., 2016] Junxuan Chen, Baigui Sun, Hao Li, Hongtao Lu, and Xian-Sheng Hua. Deep CTR prediction in display advertising. In *MM*, 2016.

[Cheng et al., 2016] Heng-Tze Cheng, Levent Koc, Jeremiah Harmsen, Tal Shaked, Tushar Chandra, Hrishi Aradhye, Glen Anderson, Greg Corrado, Wei Chai, Mustafa Ispir, Rohan Anil, Zakaria Haque, Lichan Hong, Vihan Jain, Xiaobing Liu, and Hemal Shah. Wide & deep learning for recommender systems. *CoRR*, abs/1606.07792, 2016.

[Covington et al., 2016] Paul Covington, Jay Adams, and Emre Sargin. Deep neural networks for youtube recommendations. In *RecSys*, pages 191–198, 2016.

[Graepel et al., 2010] Thore Graepel, Joaquin Quiñonero Candela, Thomas Borchert, and Ralf Herbrich. Web-scale bayesian click-through rate prediction for sponsored search advertising in microsoft's bing search engine. In *ICML*, pages 13–20, 2010.

[He et al., 2014] Xinran He, Junfeng Pan, Ou Jin, Tianbing Xu, Bo Liu, Tao Xu, Yanxin Shi, Antoine Atallah, Ralf Herbrich, Stuart Bowers, and Joaquin Quiñonero Candela. Practical lessons from predicting clicks on ads at facebook. In *ADKDD*, pages 5:1–5:9, 2014.

[He et al., 2016] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In *CVPR*, pages 770–778, 2016.

[Juan et al., 2016] Yu-Chin Juan, Yong Zhuang, Wei-Sheng Chin, and Chih-Jen Lin. Field-aware factorization machines for CTR prediction. In *RecSys*, pages 43–50, 2016.

[Larochelle et al., 2009] Hugo Larochelle, Yoshua Bengio, Jérôme Louradour, and Pascal Lamblin. Exploring strategies for training deep neural networks. *JMLR*, 10:1–40, 2009.

[Liu et al., 2015] Qiang Liu, Feng Yu, Shu Wu, and Liang Wang. A convolutional click prediction model. In *CIKM*, 2015.

[McMahan et al., 2013] H. Brendan McMahan, Gary Holt, David Sculley, Michael Young, Dietmar Ebner, Julian Grady, Lan Nie, Todd Phillips, Eugene Davydov, Daniel Golovin, Sharat Chikkerur, Dan Liu, Martin Wattenberg, Arnar Mar Hrafnkelsson, Tom Boulos, and Jeremy Kubica. Ad click prediction: a view from the trenches. In *KDD*, 2013.

[Qu et al., 2016] Yanru Qu, Han Cai, Kan Ren, Weinan Zhang, Yong Yu, Ying Wen, and Jun Wang. Product-based neural networks for user response prediction. *CoRR*, abs/1611.00144, 2016.

[Rendle and Schmidt-Thieme, 2010] Steffen Rendle and Lars Schmidt-Thieme. Pairwise interaction tensor factorization for personalized tag recommendation. In *WSDM*, pages 81–90, 2010.

[Rendle, 2010] Steffen Rendle. Factorization machines. In *ICDM*, 2010.

[Richardson et al., 2007] Matthew Richardson, Ewa Dominowska, and Robert Ragno. Predicting clicks: estimating the click-through rate for new ads. In *WWW*, pages 521–530, 2007.

[Salakhutdinov et al., 2007] Ruslan Salakhutdinov, Andriy Mnih, and Geoffrey E. Hinton. Restricted boltzmann machines for collaborative filtering. In *ICML*, pages 791–798, 2007.

[Sedhain et al., 2015] Suvash Sedhain, Aditya Krishna Menon, Scott Sanner, and Lexing Xie. Autorec: Autoencoders meet collaborative filtering. In *WWW*, pages 111–112, 2015.

[Srivastava et al., 2014] Nitish Srivastava, Geoffrey E. Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov. Dropout: a simple way to prevent neural networks from overfitting. *JMLR*, 15(1):1929–1958, 2014.

[van den Oord et al., 2013] Aäron van den Oord, Sander Dieleman, and Benjamin Schrauwen. Deep content-based music recommendation. In *NIPS*, pages 2643–2651, 2013.

[Wang and Wang, 2014] Xinxi Wang and Ye Wang. Improving content-based and hybrid music recommendation using deep learning. In *ACM MM*, pages 627–636, 2014.

[Wang et al., 2015] Hao Wang, Naiyan Wang, and Dit-Yan Yeung. Collaborative deep learning for recommender systems. In *ACM SIGKDD*, pages 1235–1244, 2015.

[Wu et al., 2016] Yao Wu, Christopher DuBois, Alice X. Zheng, and Martin Ester. Collaborative denoising autoencoders for top-n recommender systems. In *ACM WSDM*, pages 153–162, 2016.

[Wu et al., 2017] Chao-Yuan Wu, Amr Ahmed, Alex Beutel, Alexander J. Smola, and How Jing. Recurrent recommender networks. In *WSDM*, pages 495–503, 2017.

[Zhang et al., 2014] Yuyu Zhang, Hanjun Dai, Chang Xu, Jun Feng, Taifeng Wang, Jiang Bian, Bin Wang, and Tie-Yan Liu. Sequential click prediction for sponsored search with recurrent neural networks. In *AAAI*, 2014.

[Zhang et al., 2016] Weinan Zhang, Tianming Du, and Jun Wang. Deep learning over multi-field categorical data - A case study on user response prediction. In *ECIR*, 2016.

[Zheng et al., 2016] Yin Zheng, Yu-Jin Zhang, and Hugo Larochelle. A deep and autoregressive approach for topic modeling of multimodal data. *IEEE Trans. Pattern Anal. Mach. Intell.*, 38(6):1056–1069, 2016.

[Zheng et al., 2017] Lei Zheng, Vahid Noroozi, and Philip S. Yu. Joint deep modeling of users and items using reviews for recommendation. In *WSDM*, pages 425–434, 2017.
