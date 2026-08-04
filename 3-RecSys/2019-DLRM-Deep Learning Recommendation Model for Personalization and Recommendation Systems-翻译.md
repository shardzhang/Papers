# DLRM：用于个性化和推荐系统的深度学习推荐模型

> Maxim Naumov, Dheevatsa Mudigere, Hao-Jun Michael Shi\*, Jianyu Huang, Narayanan Sundaraman, Jongsoo Park, Xiaodong Wang, Udit Gupta†, Carole-Jean Wu, Alisson G. Azzolini, Dmytro Dzhulgakov, Andrey Mallevich, Ilia Cherniavskii, Yinghai Lu, Raghuraman Krishnamoorthi, Ansha Yu, Volodymyr Kondratenko, Stephanie Pereira, Xianjie Chen, Wenlin Chen, Vijay Rao, Bill Jia, Liang Xiong and Misha Smelyanskiy, Facebook, 1 Hacker Way, Menlo Park, CA 94065 | {mnaumov, dheevatsa}@fb.com, \* 西北大学，† 哈佛大学，工作期间在Facebook完成。, 预印本。正在审稿。



本文分享了Facebook提出的深度学习推荐模型（DLRM），该模型融合了 协同过滤 和 预测分析 两类视角，利用嵌入（embedding）处理稀疏类别特征、MLP处理连续特征，并通过 **显式的二阶特征交互**（点积）进行预测。核心内容：

- **模型架构**：Embedding处理类别特征 \rightarrow 底层MLP处理连续特征 \rightarrow **所有特征向量两两点积交互** \rightarrow 顶层MLP \rightarrow Sigmoid输出
- **并行策略**：Embedding表采用模型并行（以缓解内存瓶颈），MLP层采用数据并行（以扩展计算能力）
- **数据方案**：支持随机/合成/公开（Criteo Ad Kaggle & Terabyte）三类数据集

关键发现：DLRM在**Criteo Ad Kaggle数据集**上取得了略高于DCN的训练和验证准确率。Embedding查找是主要计算瓶颈。DLRM核心优势在于**仅考虑嵌入向量间的二阶交叉项**，显著降低模型维度。


---



## 摘要

随着深度学习的出现，基于神经网络的推荐模型已成为解决个性化和推荐任务的重要工具。这些网络与其他深度学习网络有显著不同，因为它们需要**处理类别特征**，且尚未得到充分研究或理解。在本文中，我们开发了一个最先进的深度学习推荐模型（DLRM），并提供了其在PyTorch和Caffe2框架中的实现。**此外，我们设计了一种专门的并行化方案，在嵌入表上利用模型并行来缓解内存约束，同时利用数据并行来扩展全连接层的计算**。我们将DLRM与现有推荐模型进行比较，并在Big Basin AI平台上对其性能进行表征，展示了其作为未来算法实验 和 系统协同设计基准的实用性。



## 1 引言

个性化和推荐系统目前已被大型互联网公司部署用于多种任务，包括广告点击率（CTR）预测和排序。尽管这些方法已有悠久历史，但它们最近才引入神经网络。

有两个主要视角对用于个性化和推荐的深度学习模型的架构设计做出了贡献。

第一个来自推荐系统的视角。这些系统最初采用 **内容过滤**，由一组专家将产品分类到不同类别，用户选择他们偏好的类别并根据其偏好进行匹配[22]。该领域后来演变为使用**协同过滤，即基于用户过去的行为（例如对产品的历史评分）进行推荐**。后来成功部署了邻域方法[21]（通过将用户和产品分组来提供推荐）和潜在因子方法（通过矩阵分解技术[9, 17]用某些隐式因子表征用户和产品）。

第二个视角来自预测分析，它依赖于统计模型来根据给定数据对事件进行分类或预测其概率[5]。预测模型从使用线性回归和逻辑回归[26]等简单模型，转变为使用包含深度网络的模型。为了处理类别数据，这些模型采用了嵌入（embedding），将one-hot和multi-hot向量转换为抽象空间中的稠密表示[20]。这个抽象空间可以解释为推荐系统所发现的潜在因子空间。

---


## 2 模型设计与架构

在本节中，我们将描述DLRM的设计。我们将从网络的高层组件开始，解释它们如何 以及 为何以特定方式组合在一起，这对未来的模型设计具有启示意义；然后描述构成模型的底层算子和原语，这对未来的硬件和系统设计具有启示意义。

### 2.1 DLRM的组件

通过回顾早期模型，可以更容易地理解DLRM的高层组件。我们将避免完整的科学文献综述，而是聚焦于早期模型中使用的四种技术，它们可以解释为DLRM的**突出高层组件**。

#### 2.1.1 嵌入（Embeddings）

为了处理类别数据，嵌入将每个类别映射到抽象空间中的一个稠密表示。特别地，每个嵌入查找可以解释为使用一个one-hot向量 $e_i$ （第 $i$ 个位置为1，其他位置为0，其中索引 $i$ 对应第 $i$ 个类别）来获取嵌入表 $W \in \mathbb{R}^{m \times d}$ 的对应行向量，如下所示：

$$
w^T_i = e^T_i W. \qquad (1)
$$

在更复杂的场景中，**嵌入也可以表示多个item的加权组合**，使用一个multi-hot权重向量 $a^T = [0, ..., a_{i_1}, ..., a_{i_k}, ..., 0]$ ，其中元素 $a_i \neq 0$ 对应 $i = i_1, ..., i_k$ ，其他位置为0，这里 $i_1, ..., i_k$ 索引相应的item。注意，一个mini-batch共 $t$ 个嵌入查找因此可以写作：

$$
S = A^T W \qquad (2)
$$

其中稀疏矩阵 $A = [a_1, ..., a_t]$ [20]。

DLRM将利用嵌入表将类别特征映射为稠密表示。然而，即使在嵌入被有意义地设计之后，它们如何被利用来产生准确的预测呢？为了回答这个问题，我们回到潜在因子方法。

#### 2.1.2 矩阵分解（Matrix Factorization）

回顾一下推荐问题的典型设定：给定一个用户集合 $S$ ，他们对某些产品进行了评分。我们希望用向量 $w_i \in \mathbb{R}^d$ （ $i=1,...,n$ ）表示第 $i$ 个产品，用向量 $v_j \in \mathbb{R}^d$ （ $j=1,...,m$ ）表示第 $j$ 个用户，以找出所有评分，其中 $n$ 和 $m$ 分别表示产品和用户的总数。更严格地说，集合 $S$ 由元组 $(i, j)$ 组成，表示第 $i$ 个产品已被第 $j$ 个用户评分。

矩阵分解方法通过最小化下式来解决这个问题：

$$
\min \sum_{(i,j) \in S} \left( r_{ij} - w^T_i v_j \right)^2 \qquad (3)
$$

其中 $r_{ij} \in \mathbb{R}$ 是第 $j$ 个用户对第 $i$ 个产品的评分， $i=1,...,m$ ， $j=1,...,n$ 。然后，令 $W^T = [w_1, ..., w_m]$ 和 $V^T = [v_1, ..., v_n]$ ，我们可以将完整的评分矩阵 $R = [r_{ij}]$ 近似为矩阵乘积 $R \approx WV^T$ 。注意 $W$ 和 $V$ 可以解释为两个嵌入表，其中每一行在潜在因子空间中表示一个用户/产品[17]。**这些嵌入向量的点积可以产生对后续评分的有意义的预测，这是分解机和DLRM设计的关键观察**。

#### 2.1.3 分解机（Factorization Machine）

在分类问题中，我们希望定义一个预测函数 $\phi: \mathbb{R}^n \rightarrow \mathcal{T}$ ，将输入数据点 $x \in \mathbb{R}^n$ 映射到目标标签 $y \in \mathcal{T}$ 。例如，我们可以通过定义 $\mathcal{T} = \{+1, -1\}$ 来预测点击率，其中 $+1$ 表示有点击， $-1$ 表示无点击。

分解机（FM）通过定义一个如下形式的模型，将二阶交互纳入带有类别数据的线性模型中：

$$
\hat{y} = b + w^T x + x^T \text{upper}(VV^T)x \qquad (4)
$$

其中 $V \in \mathbb{R}^{n \times d}$ ， $w \in \mathbb{R}^n$ ， $b \in \mathbb{R}$ 是参数， $d \ll n$ ， $\text{upper}$ 选择矩阵的严格上三角部分[24]。

FM与使用 多项式核的支持向量机（SVM）[4]有明显区别，因为它们将 **二阶交互矩阵分解为其潜在因子（或嵌入向量）**，就像矩阵分解一样，这能更有效地处理稀疏数据。这通过仅捕获不同嵌入向量对之间的交互，显著降低了二阶交互的复杂度，达到了**线性计算复杂度**。

#### 2.1.4 多层感知机（Multilayer Perceptron）

与此同时，机器学习领域的许多近期成功归因于深度学习的兴起。其中最基本的模型是多层感知机（MLP），这是一个由交错的全连接（FC）层 和 **逐分量应用的激活函数** $\sigma: \mathbb{R} \rightarrow \mathbb{R}$ 组成的预测函数，如下所示：

$$
\hat{y} = W_k \sigma(W_{k-1} \sigma(...\sigma(W_1 x + b_1)...) + b_{k-1}) + b_k \qquad (5)
$$

其中权重矩阵 $W_l \in \mathbb{R}^{n_l \times n_{l-1}}$ ，偏置 $b_l \in \mathbb{R}^{n_l}$ ，层 $l = 1, ..., k$ 。

这些方法已被用于捕获更复杂的交互。例如，已有研究表明，**给定足够的参数，具有足够深度和宽度的MLP可以以任意精度拟合数据**[1]。这些方法的变体已被广泛应用于各种应用，包括计算机视觉和自然语言处理。一个具体的案例——神经协同过滤（NCF）[15, 25]，被用作MLPerf基准[19]的一部分，它使用MLP而非点积来计算矩阵分解中嵌入之间的交互。

### 2.2 DLRM架构

到目前为止，我们已经描述了推荐系统和预测分析中使用的不同模型。现在让我们结合它们的直觉来构建一个最先进的个性化模型。

假设用户和产品由许多连续特征和类别特征来描述。为了处理类别特征，每个类别特征将由一个相同维度的嵌入向量表示，这推广了矩阵分解（3）中使用的潜在因子概念。为了处理连续特征，连续特征将由一个MLP（我们称之为底层或稠密MLP）进行变换，该MLP将产生一个与嵌入向量长度相同的稠密表示（5）。

我们将按照FM（4）中处理稀疏数据的直觉，显式地计算不同特征的二阶交互，并可选择将其通过MLP。这通过计算所有嵌入向量对和处理后的稠密特征之间的点积来实现。这些点积与原始处理后的稠密特征拼接，然后经由另一个MLP（顶层或输出MLP）（5）进行后处理，并输入sigmoid函数以给出概率。

我们将得到的模型称为DLRM，如图1所示。我们在表1中展示了DLRM在PyTorch[23]和Caffe2[8]框架中使用的一些算子。

| | 嵌入（Embedding） | MLP | 交互（Interactions） | 损失（Loss） |
|---|---|---|---|---|
| PyTorch | nn.EmbeddingBag | nn.Linear/addmm | matmul/bmm | nn.CrossEntropyLoss |
| Caffe2 | SparseLengthSum | FC | BatchMatMul | CrossEntropy |

表1：各框架下的DLRM算子

### 2.3 与先前模型的比较

许多基于深度学习的推荐模型[3, 13, 27, 18, 28, 29]使用类似的基本思想来生成高阶项以处理稀疏特征。例如，Wide and Deep、Deep and Cross、DeepFM和xDeepFM网络设计了专门的网络来系统地构建高阶交互。这些网络然后将它们专门模型和MLP的结果相加，通过线性层和sigmoid激活以产生最终概率。DLRM以一种结构化的方式交互嵌入，模仿分解机，通过仅考虑最终MLP中嵌入对之间点积产生的交叉项，显著降低了模型的维度。我们认为，其他网络中的二阶以上高阶交互可能并不值得额外的计算/内存成本。

DLRM与其他网络之间的一个关键区别在于这些网络如何处理嵌入特征向量及其交叉项。具体来说，DLRM（以及xDeepFM[18]）将每个特征向量解释为表示单个类别的单一单元，而像Deep and Cross这样的网络则将特征向量中的每个元素视为应该产生不同交叉项的新单元。因此，Deep and Cross网络不仅会产生DLRM中通过点积在不同特征向量的元素之间的交叉项，还会产生同一特征向量内部元素之间的交叉项，从而导致更高的维度。

## 3 并行化

现代个性化和推荐系统需要大型复杂模型来利用海量数据。DLRM尤其包含非常多的参数，比其他常见的深度学习模型（如卷积神经网络（CNN）、Transformer和循环神经网络（RNN）以及生成对抗网络（GAN））高出多个数量级。这导致训练时间长达数周或更长。因此，高效地并行化这些模型对于在实际规模下解决这些问题至关重要。

如前一节所述，DLRM以一种耦合的方式同时处理类别特征（通过嵌入）和连续特征（通过底层MLP）。嵌入贡献了大部分参数，几个表每个都需要超过数GB的内存，这使得DLRM对内存容量和带宽有很高要求。嵌入的大小使得使用数据并行变得不可行，因为它需要在每个设备上复制大型嵌入。在许多情况下，这种内存约束要求将模型分布在多个设备上，以满足内存容量需求。

另一方面，MLP参数在内存中较小，但转化为可观的计算量。因此，MLP更倾向于使用数据并行，因为这可以在不同设备上并发处理样本，且仅在累积更新时需要通信。

我们的并行化DLRM将在嵌入上使用模型并行和在MLP上使用数据并行相结合，以缓解嵌入产生的内存瓶颈，同时对MLP进行前向和反向传播的并行化。模型并行和数据并行的结合是DLRM因其架构和大模型规模而独有的需求。Caffe2和PyTorch（以及其他流行的深度学习框架）都不支持这种组合并行，因此我们设计了一个自定义实现。我们计划在未来的工作中提供其详细的性能研究。

在我们的设置中，顶层MLP和交互算子需要访问来自底层MLP的部分mini-batch以及所有嵌入。由于模型并行已被用于将嵌入分布到不同设备上，这需要一种个性化的all-to-all通信[12]。在嵌入查找结束时，每个设备拥有该设备上驻留的嵌入表对应mini-batch中所有样本的向量，这些向量需要沿mini-batch维度分割并传输到适当的设备，如图2所示。PyTorch和Caffe2都不原生支持模型并行；因此，我们通过将嵌入算子（PyTorch的nn.EmbeddingBag，Caffe2的SparseLengthSum）显式映射到不同设备来实现。然后，个性化的all-to-all通信使用蝴蝶混洗算子实现，该算子适当切分生成的嵌入向量并将其传输到目标设备。在当前版本中，这些传输是显式拷贝，但我们打算利用可用的通信原语（如all-gather和send-recv）进一步优化。

我们注意到，对于数据并行的MLP，反向传播中的参数更新以同步方式通过allreduce³累积并应用到每个设备上的复制参数[12]，确保每次迭代前每个设备上的参数是一致的。在PyTorch中，数据并行通过nn.DistributedDataParallel和nn.DataParallel模块实现，这些模块在每个设备上复制模型并插入具有必要依赖关系的allreduce。在Caffe2中，我们在梯度更新前手动插入allreduce。

---

³ allreduce算子的优化实现包括Nvidia的NCCL[16]和Facebook的gloo[7]。



## 4 数据

为了衡量模型的准确性、测试其整体性能并表征各个算子，我们需要为我们的实现创建或获取数据集。我们当前的模型实现支持三种类型的数据集：随机、合成和公开数据集。前两种数据集在从系统角度对模型进行实验时很有用。特别地，它们允许我们通过即时生成数据来测试不同的硬件特性和瓶颈，同时消除对数据存储系统的依赖。后者允许我们在真实数据上执行实验并衡量模型的准确性。

### 4.1 随机

回顾DLRM接受连续特征和类别特征作为输入。前者可以通过使用numpy.random包的rand或randn调用（使用默认参数），利用均匀分布或正态（高斯）分布生成随机数向量来建模。然后可以通过生成矩阵来获得输入的mini-batch，其中每一行对应mini-batch中的一个元素。

为了生成类别特征，我们需要确定在给定的multi-hot向量中希望有多少个非零元素。基准允许这个数在范围 $[1, k]$ 内固定或随机⁴。然后，我们在范围 $[1, m]$ 内生成相应数量的整数索引，其中 $m$ 是公式（2）中嵌入 $W$ 的行数。最后，为了创建查找的mini-batch，我们连接上述索引并用lengths（SparseLengthsSum）或offsets（nn.EmbeddingBag）⁵分隔每个单独的查找。

### 4.2 合成

支持自定义生成对应类别特征的索引有许多原因。例如，如果我们的应用程序使用特定数据集，但我们出于隐私原因不想共享它，那么我们可以选择通过分布来表达类别特征。这可以作为联邦学习[2, 10]等应用中使用的隐私保护技术的潜在替代方案。此外，如果我们想测试系统组件（例如研究内存行为），我们可能希望捕获原始轨迹在合成轨迹中的基本访问局部性。

现在让我们说明如何使用合成数据集。假设我们有一个对应于单个类别特征的嵌入查找的索引轨迹（并对所有特征重复该过程）。我们可以记录该轨迹中的唯一访问和重复访问之间的距离频率（算法1），然后按照[14]中提出的方法生成合成轨迹（算法2）。

**算法1 分析（原始）轨迹**
1: 令tr为输入序列，s为距离栈，u为唯一访问列表，p为概率分布
2: 令s.position_from_the_top返回d=0（如果索引未找到），否则返回d>0
3: for i=0; i<length(tr); i++ do
4:     a = tr[i]
5:     d = s.position_from_the_top(a)
6:     if d == 0 then
7:         u.append(a)
8:     else
9:         s.remove_from_the_top_at_position(d)
10:    end if
11:    p[d] += 1.0/length(tr)
12:    s.push_to_the_top(a)
13: end for

**算法2 生成（合成）轨迹**
1: 令u为输入的唯一访问列表，p为距离的概率分布，tr为输出轨迹
2: for s=0, i=0; i<length; i++ do
3:     d = p.sample_from_distribution_with_support(0, s)
4:     if d == 0 then
5:         a = u.remove_from_front()
6:         s++
7:     else
8:         a = u.remove_from_the_back_at_position(d)
9:     end if
10:    u.append(a)
11:    tr[i] = a
12: end for

注意，我们只能生成最多到目前为止已看到的唯一访问数 $s$ 的栈距离，因此在算法2中使用 $s$ 来控制分布 $p$ 的支撑集。给定固定数量的唯一访问，较长的输入轨迹将导致算法1中分配给唯一访问的概率较低，这将导致算法2中达到完整分布支撑集的时间更长。为了解决这个问题，我们将唯一访问的概率增加到最小阈值，并在所有唯一访问都被看到后调整支撑集以将其移除。基于原始轨迹和合成轨迹的概率分布 $p$ 的视觉比较如图3所示。在我们的实验中，原始轨迹和调整后的合成轨迹产生了相似的缓存命中/未命中率。

算法1和2是为更精确的缓存模拟而设计的，但它们说明了一个通用思路，即如何使用概率分布来生成具有所需属性的合成轨迹。

---

⁴ 参见选项 --num-indices-per-lookup=k 和 --num-indices-per-lookup-fixed
⁵ 例如，为了表示三个嵌入查找，分别具有索引{0,2}、{0,1,5}和{3}，我们使用
   lengths/offsets = {2,3,1}/{0,2,5}
   indices = {0,2,0,1,5,3}
   注意这种格式类似于线性代数中常用于稀疏矩阵的压缩稀疏行（CSR）格式。

### 4.3 公开

目前可用的推荐和个性化系统的公开数据集很少。Criteo AI Labs Ad Kaggle⁶ 和 Terabyte⁷ 数据集是用于广告CTR预测的点击日志开源数据集。每个数据集包含13个连续特征和26个类别特征。通常连续特征使用简单的对数变换 $\log(1+x)$ 进行预处理。类别特征被映射到其对应的嵌入索引，未标记的类别特征或标签被映射到0或NULL。

Criteo Ad Kaggle 数据集包含约4500万个样本，覆盖7天。在实验中，通常将第7天分割为验证集和测试集，而前6天用作训练集。Criteo Ad Terabyte 数据集采样自24天，其中第24天被分割为验证集和测试集，前23天用作训练集。注意每天的样本数量大致相等。



## 5 实验

现在让我们展示DLRM的性能和准确性。该模型在PyTorch和Caffe2框架中实现，并在GitHub上可用⁸。它分别使用fp32浮点类型和int32（Caffe2）/int64（PyTorch）类型用于模型参数和索引。实验在Big Basin平台上进行，配备双路Intel Xeon 6138 CPU @ 2.00GHz和八块Nvidia Tesla V100 16GB GPU，可通过开放计算item⁹公开获得，如图4所示。

图4：Big Basin AI平台

### 5.1 在公开数据集上的模型准确性

我们在Criteo Ad Kaggle数据集上评估模型的准确性，并将DLRM与Deep and Cross网络（DCN）的性能进行比较，使用原样（as-is）配置，未进行大量调优[27]。我们选择与DCN比较，因为它是少数几个在同一数据集上有全面结果的模型之一。注意，在此情况下，模型大小被调整为适应数据集中的特征数量。具体来说，DLRM包含一个用于处理稠密特征的底层MLP（由三个隐藏层组成，节点数分别为512、256和64）和一个顶层MLP（由两个隐藏层组成，节点数分别为512和256）。另一方面，DCN由六个交叉层和一个深度网络（512和256个节点）组成。使用嵌入维度16。注意，这使得DLRM和DCN都约有5.4亿个参数。

我们绘制了两种模型在使用SGD和Adagrad优化器[6]时，一个完整训练epoch中的训练（实线）和验证（虚线）准确率。未使用正则化。在此实验中，DLRM获得了略高的训练和验证准确率，如图5所示。我们强调这是在未对模型超参数进行大量调优的情况下取得的。

图5：DLRM和DCN的训练（实线）和验证（虚线）准确率比较

### 5.2 在单个Socket/设备上的模型性能

为了分析模型在单个socket设备上的性能，我们考虑一个具有8个类别特征和512个连续特征的示例模型。每个类别特征通过一个包含100万个向量、向量维度为64的嵌入表进行处理，而连续特征被组合成一个维度为512的向量。设底层MLP有两层，顶层MLP有四层。我们在一个包含204.8万个随机生成样本（组织成1000个mini-batch¹⁰）的数据集上分析此模型。

图6：在单个socket/设备上的示例DLRM分析

此模型在Caffe2中的实现在CPU上运行约256秒，在GPU上运行约62秒，各算子的分析如图6所示。正如预期，大部分时间花在嵌入查找和全连接层上。在CPU上，全连接层占据了计算的显著部分，而在GPU上它们几乎可忽略。

---

⁶ https://www.kaggle.com/c/criteo-display-ad-challenge
⁷ https://labs.criteo.com/2013/12/download-terabyte-click-logs/
⁸ https://github.com/facebookresearch/dlrm
⁹ https://www.opencompute.org
¹⁰ 例如，此配置可以通过以下命令行参数实现：
   --arch-embedding-size=1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000
   --arch-sparse-feature-size=64 --arch-mlp-bot=512-512-64 --arch-mlp-top=1024-1024-1024-1
   --data-generation=random --mini-batch-size=2048 --num-batches=1000 --num-indices-per-lookup=100 [--use-gpu]
   [--enable-profiling]



## 6 结论

在本文中，我们提出并开源了一种新颖的基于深度学习的推荐模型，该模型有效利用了类别数据。尽管推荐和个性化系统至今仍在推动深度学习在工业界的许多实际成功，但这些网络在学术界仍然很少受到关注。通过提供最先进的推荐系统的详细描述及其开源实现，我们希望以易于理解的方式引起人们对这类网络所呈现的独特挑战的关注，以促进进一步的算法实验、建模、系统协同设计和基准测试。



## 致谢

作者感谢AI Systems Co-Design、Caffe2、PyTorch和AML团队成员的审阅帮助。



## 参考文献

[1] Christopher M. Bishop. *Neural Networks for Pattern Recognition*. The Oxford University Press, 1st edition, 1995.

[2] Keith Bonawitz, Hubert Eichner, Wolfgang Grieskamp, Dzmitry Huba, Alex Ingerman, Vladimir Ivanov, Chloé Kiddon, Jakub Konečný, Stefano Mazzocchi, Brendan McMahan, Timon Van Overveldt, David Petrou, Daniel Ramage, and Jason Roselander. Towards federated learning at scale: System design. In *Proc. 2nd Conference on Systems and Machine Learning (SysML)*, 2019.

[3] Heng-Tze Cheng, Levent Koc, Jeremiah Harmsen, Tal Shaked, Tushar Chandra, Hrishi Aradhye, Glen Anderson, Greg Corrado, Wei Chai, Mustafa Ispir, Rohan Anil, Zakaria Haque, Lichan Hong, Vihan Jain, Xiaobing Liu, and Hemal Shah. Wide & deep learning for recommender systems. In *Proc. 1st Workshop on Deep Learning for Recommender Systems*, pages 7–10, 2016.

[4] Corinna Cortes and Vladimir N. Vapnik. Support-vector networks. *Machine Learning*, 2:273–297, 1995.

[5] Luc Devroye, Laszlo Gyorfi, and Gabor Lugosi. *A Probabilistic Theory of Pattern Recognition*. New York, Springer-Verlag, 1996.

[6] John Duchi, Elad Hazan, and Yoram Singer. Adaptive subgradient methods for online learning and stochastic optimization. *Journal of Machine Learning Research*, 12:2121–2159, 2011.

[7] Facebook. Collective communications library with various primitives for multi-machine training (gloo), https://github.com/facebookincubator/gloo.

[8] Facebook. Caffe2, https://caffe2.ai, 2016.

[9] Evgeny Frolov and Ivan Oseledets. Tensor methods and recommender systems. *Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery*, 7(3):e1201, 2017.

[10] Craig Gentry. A fully homomorphic encryption scheme. PhD thesis, Stanford University, 2009.

[11] Gene H. Golub and Charles F. Van Loan. *Matrix Computations*. The John Hopkins University Press, 3rd edition, 1996.

[12] Ananth Grama, Vipin Kumar, Anshul Gupta, and George Karypis. *Introduction to Parallel Computing*. Pearson Education, 2003.

[13] Huifeng Guo, Ruiming Tang, Yunming Ye, Zhenguo Li, and Xiuqiang He. DeepFM: a factorization-machine based neural network for CTR prediction. *arXiv preprint arXiv:1703.04247*, 2017.

[14] Rahman Hassan, Antony Harris, Nigel Topham, and Aris Efthymiou. Synthetic trace-driven simulation of cache memory. In *Proc. 21st International Conference on Advanced Information Networking and Applications Workshops (AINAW'07)*, 2007.

[15] Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu, and Tat-Seng Chua. Neural collaborative filtering. In *Proc. 26th Int. Conf. World Wide Web*, pages 173–182, 2017.

[16] Sylvain Jeaugey. Nccl 2.0, 2017.

[17] Yehuda Koren, Robert Bell, and Chris Volinsky. Matrix factorization techniques for recommender systems. *Computer*, (8):30–37, 2009.

[18] Jianxun Lian, Xiaohuan Zhou, Fuzheng Zhang, Zhongxia Chen, Xing Xie, and Guangzhong Sun. xDeepFM: Combining explicit and implicit feature interactions for recommender systems. In *Proc. of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*, pages 1754–1763. ACM, 2018.

[19] MLPerf. https://mlperf.org/.

[20] Maxim Naumov. On the dimensionality of embeddings for sparse features and data. In *arXiv preprint arXiv:1901.02103*, 2019.

[21] Xia Ning, Christian Desrosiers, and George Karypis. A comprehensive survey of neighborhood-based recommendation methods. In *Recommender Systems Handbook*, 2015.

[22] Pandora. Music genome project https://www.pandora.com/about/mgp.

[23] Adam Paszke, Sam Gross, Soumith Chintala, and Gregory Chanan. PyTorch: Tensors and dynamic neural networks in python with strong GPU acceleration https://pytorch.org/, 2017.

[24] Steffen Rendle. Factorization machines. In *Proc. 2010 IEEE International Conference on Data Mining*, pages 995–1000, 2010.

[25] Suvash Sedhain, Aditya Krishna Menon, Scott Sanner, and Lexing Xie. Autorec: Autoencoders meet collaborative filtering. In *Proc. 24th Int. Conf. World Wide Web*, pages 111–112, 2015.

[26] Strother H. Walker and David B. Duncan. Estimation of the probability of an event as a function of several independent variables. *Biometrika*, 54:167–178, 1967.

[27] Ruoxi Wang, Bin Fu, Gang Fu, and Mingliang Wang. Deep & cross network for ad click predictions. In *Proc. ADKDD*, page 12, 2017.

[28] Guorui Zhou, Na Mou, Ying Fan, Qi Pi, Weijie Bian, Chang Zhou, Xiaoqiang Zhu, and Kun Gai. Deep interest evolution network for click-through rate prediction. *arXiv preprint arXiv:1809.03672*, 2018.

[29] Guorui Zhou, Xiaoqiang Zhu, Chenru Song, Ying Fan, Han Zhu, Xiao Ma, Yanghui Yan, Junqi Jin, Han Li, and Kun Gai. Deep interest network for click-through rate prediction. In *Proc. of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*, pages 1059–1068. ACM, 2018.
