# FiBiNET：结合特征重要性与双线性特征交互的点击率预测

> Tongwen Huang, Zhiqi Zhang, Junlin Zhang | Sina Weibo Inc.



本文介绍了 FiBiNET（Feature Importance and Bilinear feature Interaction NETwork，特征重要性与双线性特征交互网络）。核心内容：

- 提出 FiBiNET 模型，**通过 SENET 机制动态学习特征重要性**，并利用 **双线性函数细粒度地计算特征交互**
- 引入三种类型的 **双线性交互层**（Field-All、Field-Each、Field-Interaction）
- 将经典深度神经网络（DNN）组件 与 浅层模型结合为深度模型
- 在 Criteo 和 Avazu 两个真实数据集上进行大量实验评估

关键发现：

- 浅层 FiBiNET 优于 FM、FFM 等其他浅层模型；深度 FiBiNET 持续优于 DeepFM、xDeepFM 等最先进的深度模型
- SENET 层 与 双线性交互层对 FiBiNET 的性能**均不可或缺**

---



## 摘要

广告和Feed排序对于许多互联网公司（如Facebook和新浪微博）至关重要。在许多真实的广告和Feed排序系统中，点击率（CTR，Click-Through Rate）预测扮演着核心角色。该领域已有许多模型被提出，如逻辑回归、基于树的模型、基于因子分解机的模型以及基于深度学习的CTR模型。然而，当前许多方法**以简单方式（如Hadamard积和内积）计算特征交互**，并且**较少关注特征的重要**性。本文提出了一种名为FiBiNET（Feature Importance and Bilinear feature Interaction NETwork，特征重要性与双线性特征交互网络）的新模型，用于动态学习特征重要性和细粒度特征交互。一方面，FiBiNET通过SENET（Squeeze-and-Excitation Network，**压缩激励网络**）机制动态学习特征的重要性；另一方面，它能够通过双线性函数有效**学习特征交互**。我们在两个真实数据集上进行了大量实验，结果表明我们的浅层模型优于其他浅层模型，如因子分解机（FM，Factorization Machine）和场感知因子分解机（FFM，Field-aware Factorization Machine）。为了进一步提升性能，我们将经典深度神经网络（DNN，Deep Neural Network）组件与浅层模型结合为深度模型。深度FiBiNET持续优于其他最先进的深度模型，如DeepFM和xDeepFM（eXtreme Deep Factorization Machine，极端深度因子分解机）。

**CCS概念**：计算机系统组织 $\rightarrow$ 因子分解方法；计算理论 $\rightarrow$ 计算广告理论。

**关键词**：Display Advertising, CTR Prediction, Factorization Machines, Squeeze-Excitation network, Neural Network, Bilinear Function



## 1 引言

广告和Feed排序对于许多互联网公司（如Facebook和新浪微博）至关重要。这些任务背后的主要技术是点击率预测，即CTR（Click-Through Rate，点击率）。该领域已提出许多模型，如逻辑回归（LR，Logistic Regression）[17]、多项式-2（Poly2，Polynomial-2）[10]、基于树的模型[7]、基于张量的模型[13]、贝叶斯模型[3]以及基于因子分解机的模型[9, 10, 19, 20]。

随着深度学习在计算机视觉[5, 14]和自然语言处理[2, 18]等许多研究领域的巨大成功，近年来许多基于深度学习的CTR模型被提出[1, 4, 6, 15, 22, 23, 25, 26]。因此，基于深度学习的CTR预测也成为该领域的研究趋势。一些基于神经网络的模型已被提出并取得成功，如因子分解机支持的神经网络（FNN，Factorization-Machine Supported Neural Network）[25]、Wide & Deep模型（WDL，Wide & Deep Learning）[1]、注意力因子分解机（AFM，Attentional Factorization Machine）[23]、DeepFM[4]和xDeepFM[15]等。

本文提出了一种名为FiBiNET（Feature Importance and Bilinear feature Interaction NETwork，特征重要性与双线性特征交互网络）的新模型，用于动态学习特征重要性和细粒度特征交互。**据我们所知，不同的特征对于目标任务具有不同的重要性**。例如，当我们预测一个人的收入时，职业特征比爱好特征更重要。考虑到这一点，我们引入了SENET（Squeeze-and-Excitation Network，压缩激励网络）[8]来动态学习特征的权重。此外，特征交互是CTR预测领域的一个关键挑战，许多相关工作以简单方式（如Hadamard积和内积）计算特征交互。本文提出了一种新的细粒度方法，使用双线性函数计算特征交互。我们的主要贡献如下：

- 受SENET在计算机视觉领域成功的启发，我们使用SENET机制动态学习特征的权重。
- 我们引入了三种类型的双线性交互层（Bilinear-Interaction layer），以细粒度的方式学习特征交互。这与先前的工作[6, 9, 10, 19, 20, 23]形成对比，这些工作使用Hadamard积或内积计算特征交互。
- 结合SENET机制与双线性特征交互，我们的浅层模型在Criteo和Avazu数据集上达到了浅层模型中的最先进水平，如**优于FFM**。
- 为了进一步提升性能，我们将经典深度神经网络（DNN）组件与浅层模型结合为深度模型。深度FiBiNET在Criteo和Avazu数据集上持续优于其他最先进的深度模型。

本文其余部分组织如下：第2节回顾与我们所提模型相关的相关工作；第3节介绍我们所提模型；第4节展示在Criteo和Avazu数据集上的实验探索；最后，我们在第5节讨论实验结果并总结本文工作。



## 2 相关工作

### 2.1 因子分解机及其相关变体

因子分解机（FM，Factorization Machine）[19, 20]和场感知因子分解机（FFM，Field-aware Factorization Machine）[9, 10]是**两个最成功的CTR模型**。**FM使用因子化参数对变量之间的所有特征交互进行建模**。**它具有低时间复杂度和内存存储，并且能很好地处理大规模稀疏数据**。FFM引入了场感知隐向量，并赢得了Criteo和Avazu主办的两个竞赛[9]。然而，FFM受限于**大内存需求**，不易在互联网公司中使用。

### 2.2 基于深度学习的CTR模型

深度学习在计算机视觉[5, 14]和自然语言处理[2, 18]等许多研究领域取得了巨大成功。因此，近年来许多基于深度学习的CTR模型也被提出[1, 4, 6, 15, 22, 23, 25, 26]。如何有效建模特征交互是大多数基于神经网络的模型的关键因素。

因子分解机支持的神经网络（FNN，Factorization-Machine Supported Neural Network）[25]是一种**使用FM预训练嵌入层的前馈神经网络**。然而，**FNN只能捕获高阶特征交互**。Wide & Deep模型（WDL，Wide & Deep Learning）[1]最初是为Google Play中的应用推荐而引入的。WDL联合训练宽线性模型和深度神经网络，以结合记忆和泛化的优势用于推荐系统。然而，**WDL的宽部分输入仍然需要专家特征工程，这意味着交叉积转换也需要人工设计**。为了减轻特征工程中的人工工作，DeepFM[4]用FM替换了WDL的宽部分，并在FM和深度组件之间共享特征嵌入。DeepFM被认为是**CTR预估领域最先进的模型之一**。

Deep & Cross Network（DCN）[22]以显式方式高效捕获有界阶数的特征交互。类似地，极端深度因子分解机（xDeepFM）[15]通过提出新颖的**压缩交互网络**（CIN，Compressed Interaction Network）部分，以显式方式建模低阶和高阶特征交互。如[23]所述，FM对所有特征交互使用相同权重进行建模可能会受到限制，因为并非所有特征交互都同样有用和具有预测性。他们提出了注意力因子分解机（AFM，Attentional Factorization Machine）[23]模型，该模型使用注意力网络学习特征交互的权重。深度兴趣网络（DIN，Deep Interest Network）[26]使用兴趣分布表示用户多样化的兴趣，并设计类似注意力的网络结构，**根据候选广告 局部激活相关兴趣**。

### 2.3 SENET模块

Hu等人[8]提出了"Squeeze-and-Excitation Network"（SENET），通过**显式建模卷积特征通道之间的相互依赖关系**来提升网络的表示能力，用于各种图像分类任务。SENET被证明在图像分类任务中非常成功，并在ILSVRC 2017分类任务中获得第一名。

除图像分类外，SENET还有其他应用[12, 21, 24]。[21]为语义分割任务引入了SE模块的三种变体。胸部X光片上常见胸部疾病分类以及可疑病变区域的定位[24]是另一个应用领域。[16]将SENET模块与全局-局部注意力（GALA，Global-and-Local Attention）模块扩展，在ILSVRC上获得了最先进的准确率。



## 3 本文提出的模型

我们的目标是动态学习特征的重要性并以细粒度的方式学习特征交互。为此，我们提出了特征重要性与双线性特征交互网络（FiBiNET）用于CTR预测任务。

在本节中，我们将描述所提模型的架构，如图1所示。为清晰起见，我们省略了可以简单加入的逻辑回归部分。我们所提模型由以下部分组成：稀疏输入层、嵌入层、SENET层、双线性交互层、组合层、多个隐藏层和输出层。稀疏输入层和嵌入层与DeepFM[4]相同，采用稀疏表示处理输入特征，并将原始特征输入嵌入为稠密向量。SENET层可以将嵌入层转换为类似SENET的嵌入特征，有助于提升特征判别能力。随后的双线性交互层分别在原始嵌入和类似SENET的嵌入上建模二阶特征交互。然后，这些交叉特征由组合层拼接，该层合并双线性交互层的输出。最后，将交叉特征输入深度神经网络，网络输出预测得分。

<img src="/Users/dazhang/PycharmProject/Papers/3-RecSys/.picture/image-20260805040403645.png" alt="image-20260805040403645" style="zoom: 33%;" />

> 图1：我们所提 FiBiNET 的架构。

### 3.1 稀疏输入和嵌入层

稀疏输入层和嵌入层广泛用于基于深度学习的CTR模型，如DeepFM[4]和AFM[23]。稀疏输入层采用稀疏表示处理原始输入特征。嵌入层能够将稀疏特征嵌入到低维、稠密的实值向量中。嵌入层的输出是一个宽拼接的场嵌入[^1]向量：$E = [e_1, e_2, \ldots, e_i, \ldots, e_f]$，其中 $f$ 表示场的数量，$e_i \in \mathbb{R}^k$ 表示第 $i$ 个场的嵌入，$k$ 是嵌入层的维度。

### 3.2 SENET层

据我们所知，不同的特征对于目标任务具有不同的重要性。例如，当我们预测一个人的收入时，职业特征比爱好特征更重要。受SENET在计算机视觉领域成功的启发，我们引入SENET机制使模型更加关注特征重要性。对于特定的CTR预测任务，我们可以通过SENET机制动态增加重要特征的权重并减少无信息特征的权重。

使用特征嵌入作为输入，SENET生成场嵌入的权重向量 $A = \{a_1, \ldots, a_i, \ldots, a_f\}$，然后用向量 $A$ 重新缩放原始嵌入 $E$，得到新的嵌入（类似SENET的嵌入）$V = [v_1, \ldots, v_i, \ldots, v_f]$，其中 $a_i \in \mathbb{R}$ 是标量，表示第 $i$ 个场嵌入 $v_i$ 的权重，$v_i \in \mathbb{R}^k$ 表示第 $i$ 个场的类似SENET的嵌入，$i \in [1, 2, \ldots, f]$，$V \in \mathbb{R}^{f \times k}$，$k$ 是嵌入大小，$f$ 是场的数量。

如图2所示，SENET由三个步骤组成：压缩步骤、激励步骤和重新加权步骤。这些步骤的详细描述如下：

**图2：SENET层。**

**压缩（Squeeze）**：此步骤用于计算每个场嵌入的"汇总统计量"。具体来说，我们使用一些池化方法（如最大池化或均值池化）将原始嵌入 $E = [e_1, \ldots, e_f]$ 压缩为统计向量 $Z = [z_1, \ldots, z_i, \ldots, z_f]$，其中 $i \in [1, \ldots, f]$，$z_i$ 是一个标量值，表示第 $i$ 个特征表示的全局信息。$z_i$ 可以通过以下全局均值池化计算：

$$
z_i = F_{sq}(e_i) = \frac{1}{k} \sum_{t=1}^{k} e_i^{(t)} \qquad (1)
$$

原始的SENET论文[8]中的压缩函数是最大池化。然而，我们的实验结果表明均值池化优于最大池化。

**激励（Excitation）**：此步骤用于基于统计向量 $Z$ 学习每个场嵌入的权重。我们使用两个全连接（FC，Full Connected）层来学习权重。第一个FC层是降维层，参数为 $W_1$，降维比率 $r$ 是一个超参数，并使用 $\sigma_1$ 作为非线性函数。第二个FC层使用参数 $W_2$ 增加维度。形式上，场嵌入的权重可以计算如下：

$$
A = F_{ex}(Z) = \sigma_2(W_2 \sigma_1(W_1 Z)) \qquad (2)
$$

其中 $A \in \mathbb{R}^f$ 是一个向量，$\sigma_1$ 和 $\sigma_2$ 是激活函数，学习参数为 $W_1 \in \mathbb{R}^{f \times (f/r)}$，$W_2 \in \mathbb{R}^{(f/r) \times f}$，$r$ 是降维比率。

**重新加权（Re-Weight）**：SENET的最后一步是重新加权步骤，在原始论文[8]中称为重新缩放。它在原始场嵌入 $E$ 和场权重向量 $A$ 之间进行逐场相乘，并输出新的嵌入（类似SENET的嵌入）$V = \{v_1, \ldots, v_i, \ldots, v_f\}$。类似SENET的嵌入 $V$ 可以计算如下：

$$
V = F_{ReWeight}(A, E) = [a_1 \cdot e_1, \ldots, a_f \cdot e_f] = [v_1, \ldots, v_f] \qquad (3)
$$

其中 $a_i \in \mathbb{R}$，$e_i \in \mathbb{R}^k$，$v_i \in \mathbb{R}^k$。

简而言之，SENET使用两个全连接层动态学习特征的重要性。对于特定任务，它会增加重要特征的权重并减少无信息特征的权重。

### 3.3 双线性交互层

交互层是用于计算二阶特征交互的层。交互层中特征交互的经典方法是内积和Hadamard积。内积广泛用于浅层模型（如FM和FFM），而Hadamard积通常用于深度模型（如AFM和神经因子分解机（NFM，Neural Factorization Machine））。内积和Hadamard积的形式分别表示为 $\{(v_i \cdot v_j)x_i x_j\}_{(i,j) \in R_x}$ 和 $\{(v_i \odot v_j)x_i x_j\}_{(i,j) \in R_x}$，其中 $R_x = \{(i, j)\} i \in \{1, \ldots, f\}, j \in \{1, \ldots, f\}, j > i$，$v_i$ 是第 $i$ 个场嵌入向量，$\cdot$ 表示常规内积，$\odot$ 表示Hadamard积，例如 $[a_1, a_2, a_3] \odot [b_1, b_2, b_3] = [a_1b_1, a_2b_2, a_3b_3]$。交互层中的内积和Hadamard积过于简单，无法有效建模稀疏数据集中的特征交互。因此，我们提出了一种更细粒度的方法，结合内积和Hadamard积，使用额外的参数来学习特征交互。如图3.c所示，矩阵 $W$ 与向量 $v_i$ 之间使用内积，矩阵 $W$ 与向量 $v_j$ 之间使用Hadamard积。具体来说，我们在此层中提出了三种类型的双线性函数，并将该层称为双线性交互层。以第 $i$ 个场嵌入 $v_i$ 和第 $j$ 个场嵌入 $v_j$ 为例，特征交互的结果 $p_{ij}$ 可以计算如下：

**图3：计算特征交互的不同方法。(a)：内积。(b)：Hadamard积。(c)：我们提出的双线性交互。这里内积方法中的 $p_{ij}$ 是标量，而在Hadamard积和我们提出的双线性函数中它是向量。**

**a. Field-All类型**

$$
p_{ij} = v_i \cdot W \odot v_j \qquad (4)
$$

其中 $W \in \mathbb{R}^{k \times k}$，$v_i, v_j \in \mathbb{R}^k$ 是第 $i$ 个和第 $j$ 个场嵌入，$1 \le i \le f$，$i \le j \le f$。这里 $W$ 在所有 $(v_i, v_j)$ 场交互对之间共享，双线性交互层中共有 $k \times k$ 个参数，因此我们将此类型称为"Field-All"。

**b. Field-Each类型**

$$
p_{ij} = v_i \cdot W_i \odot v_j \qquad (5)
$$

其中 $W_i \in \mathbb{R}^{k \times k}$，$v_i, v_j \in \mathbb{R}^k$ 是第 $i$ 个和第 $j$ 个场嵌入，$1 \le i \le f$，$i \le j \le f$。这里 $W_i$ 是第 $i$ 个场的对应参数矩阵，由于我们有 $f$ 个不同的场，双线性交互层中共有 $f \times k \times k$ 个参数，因此我们将此类型称为"Field-Each"。

**c. Field-Interaction类型**

$$
p_{ij} = v_i \cdot W_{ij} \odot v_j \qquad (6)
$$

其中 $W_{ij} \in \mathbb{R}^{k \times k}$ 是场 $i$ 和场 $j$ 之间交互的对应参数矩阵，$1 \le i \le f$，$i \le j \le f$。此层中总的学习参数数量为 $n \times k \times k$，其中 $n$ 是场交互数量，等于 $\frac{f(f-1)}{2}$。因此我们将此类型称为"Field-Interaction"。

如图1所示，我们有两个嵌入（原始嵌入和类似SENET的嵌入），我们可以对任意嵌入采用双线性函数或Hadamard积作为特征交互操作。因此，在此层中有几种特征交互的组合。在第4.3节中，我们将详细讨论双线性函数和Hadamard积不同组合的性能。此外，我们有三种不同类型的所提特征交互方法（Field-All、Field-Each、Field-Interaction）应用于我们的模型，我们将在第4.4节讨论不同场类型的性能。

在本节中，双线性交互层可以从原始嵌入 $E$ 输出交互向量 $p = [p_1, \ldots, p_i, \ldots, p_n]$，从类似SENET的嵌入 $V$ 输出类似SENET的交互向量 $q = [q_1, \ldots, q_i, \ldots, q_n]$，其中 $p_i, q_i \in \mathbb{R}^k$ 是向量。

### 3.4 组合层

组合层拼接交互向量 $p$ 和 $q$，并将拼接后的向量馈入FiBiNET的后续层（标准神经网络层）。它可以表示为以下形式：

$$
c = F_{concat}(p, q) = [p_1, \ldots, p_n, q_1, \ldots, q_n] = [c_1, \ldots, c_{2n}] \qquad (7)
$$

如果我们将向量 $c$ 中的每个元素求和，然后使用sigmoid函数输出预测值，我们就得到了一个浅层CTR模型。为了进一步提升性能，我们将浅层组件和经典深度神经网络（DNN）（将在第3.5节中描述）结合为一个统一模型，形成深度网络结构，在本文中称为深度模型。

### 3.5 深度网络

深度网络由几个全连接层组成，隐式捕获高阶特征交互。如图1所示，深度网络的输入是组合层的输出。设 $a^{(0)} = [c_1, c_2, \ldots, c_{2n}]$ 表示组合层的输出，其中 $c_i \in \mathbb{R}^k$，$n$ 是场交互数量。然后，$a^{(0)}$ 被馈入深度神经网络，前馈过程为：

$$
a^{(l)} = \sigma(W^{(l)} a^{(l-1)} + b^{(l)}) \qquad (8)
$$

其中 $l$ 是层深度，$\sigma$ 是激活函数。$W^{(l)}$、$b^{(l)}$、$a^{(l)}$ 分别是第 $l$ 层的模型权重、偏置和输出。之后，生成稠密的实值特征向量，最终馈入sigmoid函数进行CTR预测：$y_d = \sigma(W^{|L|+1} a^{|L|} + b^{|L|+1})$，其中 $|L|$ 是DNN的深度。

### 3.6 输出层

总结起来，我们给出所提模型输出的整体公式如下：

$$
\hat{y} = \sigma\left(w_0 + \sum_{i=0}^{m} w_i x_i + y_d\right) \qquad (9)
$$

其中 $\hat{y} \in (0, 1)$ 是CTR的预测值，$\sigma$ 是sigmoid函数，$m$ 是特征数量，$x$ 是输入，$w_i$ 是线性部分中第 $i$ 个特征的权重。我们模型的参数为 $\theta = \{w_0, \{w_i\}_{i=1}^{m}, \{e_i\}_{i=1}^{m}, \{W_i\}_{i=1}^{2}, \{W^{(i)}\}_{i=1}^{|L|}\}$。学习过程旨在最小化以下目标函数（交叉熵）：

$$
loss = -\frac{1}{N} \sum_{i=1}^{N} (y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)) \qquad (10)
$$

其中 $y_i$ 是第 $i$ 个样本的真实标签，$\hat{y}_i$ 是预测的CTR，$N$ 是样本总数。

#### 3.6.1 与FM和FNN的关系

假设我们移除SENET层和双线性交互层，不难发现我们的模型将退化为FNN。当我们进一步移除DNN部分，同时使用常数求和，则浅层FiBiNET退化为传统的FM模型。



## 4 实验

在本节中，我们进行大量实验来回答以下问题：

- (RQ1) 与CTR预测的最先进方法相比，我们的模型表现如何？
- (RQ2) 双线性交互层中双线性函数和Hadamard函数的不同组合是否会影响其性能？
- (RQ3) 双线性交互层的不同场类型（Field-All、Field-Each和Field-Interaction）是否会影响其性能？
- (RQ4) 网络设置如何影响我们模型的性能？
- (RQ5) FiBiNET中最重要的组件是什么？

在介绍一些基础实验设置之后，我们将回答这些问题。

### 4.1 实验平台与设置

#### 4.1.1 数据集

1) **Criteo**：Criteo[^2]数据集广泛用于许多CTR模型评估。它包含4500万条数据实例的点击日志。Criteo数据集中有26个匿名类别字段和13个连续特征字段。我们将数据集随机分为两部分：90%用于训练，其余用于测试。

2) **Avazu**：Avazu[^3]数据集包含按时间顺序排列的几天广告点击数据。它包含4000万条数据实例的点击日志。每条点击数据有24个字段，表示单个广告展示的元素。我们将其随机分为两部分：80%用于训练，其余用于测试。

#### 4.1.2 评估指标

在我们的实验中，我们采用两个指标：AUC（Area Under ROC Curve，ROC曲线下面积）和Log损失。

**AUC**：ROC曲线下面积是评估分类问题中广泛使用的指标。此外，一些工作验证了AUC是CTR预测中的良好度量[3]。AUC对分类阈值和正样本比例不敏感。AUC的上界为1，越大越好。

**Log损失**：Log损失是二分类中广泛使用的指标，用于衡量两个分布之间的距离。Log损失的下界为0，表示两个分布完美匹配，值越小表示性能越好。

#### 4.1.3 基线方法

为了验证在浅层模型和深度模型中结合SENET层与双线性交互层的效果，我们将实验分为两组：浅层组和深层组。我们也将基线模型分为两部分：浅层基线模型和深度基线模型。浅层基线模型包括LR（逻辑回归）[17]、FM[19, 20]、FFM[9, 10]、AFM[23]；深度基线模型包括FNN[25]、DCN[22]、DeepFM[4]、xDeepFM[15]。

需要注意的是，AUC提升1‰通常被认为对CTR预测是显著的，因为如果公司拥有非常大的用户基础，这将带来公司收入的大幅增长。

#### 4.1.4 实现细节

我们在实验中均使用TensorFlow[^4]实现所有模型。对于嵌入层，Criteo数据集的嵌入维度设为10，Avazu数据集设为50。对于优化方法，我们对Criteo使用Adam[11]优化器，迷你批大小为1000；对Avazu数据集使用迷你批大小为500，学习率设为0.0001。对于所有深度模型，层深度设为3，所有激活函数为RELU，Criteo数据集每层神经元数为400，Avazu数据集为2000，dropout率设为0.5。对于SENET部分，两个全连接层的激活函数均为RELU函数，降维比率设为3。我们使用2块Tesla K40 GPU进行实验。

### 4.2 性能比较（RQ1）

在本小节中，我们总结了浅层模型和深度模型在Criteo和Avazu测试集上的整体性能，分别见表1和表2。

**表1：浅层模型在Criteo和Avazu数据集上的整体性能。SE-FM-ALL表示使用Field-All类型双线性交互层的浅层模型。**

| 模型 | Criteo AUC | Criteo Logloss | Avazu AUC | Avazu Logloss |
|------|-----------|---------------|----------|--------------|
| LR | 0.7808 | 0.4681 | 0.7633 | 0.3891 |
| FM | 0.7923 | 0.4584 | 0.7745 | 0.3832 |
| FFM | 0.8001 | 0.4525 | 0.7795 | 0.3810 |
| AFM | 0.7965 | 0.4541 | 0.7740 | 0.3839 |
| SE-FM-All | 0.8021 | 0.4495 | 0.7803 | 0.3800 |

表1显示了浅层模型在Criteo和Avazu数据集上的结果。我们发现我们的浅层SE-FM-All模型始终优于其他模型，如FM、FFM、AFM等。一方面，结果表明将SENET机制与稀疏特征上的双线性交互相结合是一种对许多真实数据集有效的方法；另一方面，对于经典浅层模型，最先进的模型是FFM，但它受限于大内存需求，不易在互联网公司中使用，而我们的浅层模型参数更少但性能仍然优于FFM。因此，它可以被视为FFM的替代解决方案。

**表2：深度模型在Criteo和Avazu数据集上的整体性能。DeepSE-FM-ALL表示使用Field-All类型双线性交互层的深度模型。**

| 模型 | Criteo AUC | Criteo Logloss | Avazu AUC | Avazu Logloss |
|------|-----------|---------------|----------|--------------|
| FNN | 0.8057 | 0.4464 | 0.7802 | 0.3800 |
| DeepFM | 0.8085 | 0.4445 | 0.7786 | 0.3810 |
| DCN | 0.7978 | 0.4617 | 0.7681 | 0.3940 |
| xDeepFM | 0.8091 | 0.4461 | 0.7808 | 0.3818 |
| DeepSE-FM-All | 0.8103 | 0.4423 | 0.7832 | 0.3786 |

为了进一步提升性能，我们将浅层部分和DNN结合为深度模型。深度模型的整体性能如表2所示，我们有以下观察：

- 将浅层部分和DNN结合为统一模型，浅层模型可以获得进一步的性能提升。从实验结果我们可以推断，隐式高阶特征交互有助于浅层模型获得更强的表达能力。
- 在所有比较方法中，我们提出的深度FiBiNET实现了最佳性能。我们的深度模型在Criteo和Avazu数据集上，AUC分别相对优于FNN 0.571%和0.386%（Log loss分别优于0.918%和0.4%），AUC分别优于DeepFM 0.222%和0.59%（Log loss分别优于0.494%和0.6%）。
- 结果表明，在DNN中结合SENET机制与双线性交互进行预测是有效的。一方面，SENET本质上是根据输入引入动态机制，有助于提升特征判别能力；另一方面，与内积或Hadamard积等其他方法相比，双线性函数是一种有效的特征交互建模方法，如第4.3节所述。

为了进一步提升性能，我们将在第4.3节讨论双线性交互层的不同组合，并在第4.4节讨论双线性交互层的场类型。

### 4.3 双线性交互层的组合（RQ2）

在本节中，我们将讨论双线性交互层中双线性函数和Hadamard积不同组合类型的影响。为方便起见，我们使用0和1表示双线性交互层中使用哪个函数。"1"表示使用双线性函数，"0"表示使用Hadamard积。我们有两个嵌入，因此使用两个数字。第一个数字表示在原始嵌入上使用的特征交互方法，第二个数字表示在类似SENET的嵌入上使用的特征交互方法。例如，"10"表示在原始嵌入上使用双线性函数作为特征交互方法，而在类似SENET的嵌入上使用Hadamard函数作为特征交互方法。类似地，我们在浅层和深度模型上进行实验，并将结果总结在表3中。

**表3：双线性交互层中双线性函数和Hadamard函数不同组合的性能。双线性交互层的场类型设为Field-Each。**

| 组合 | Criteo AUC | Criteo Logloss | Avazu AUC | Avazu Logloss |
|------|-----------|---------------|----------|--------------|
| SE-FM_00 | 0.7989 | 0.4525 | 0.7782 | 0.3818 |
| SE-FM_01 | 0.8018 | 0.4500 | 0.7797 | 0.3808 |
| SE-FM_10 | 0.8029 | 0.4488 | 0.7794 | 0.3807 |
| SE-FM_11 | 0.8037 | 0.4479 | 0.7770 | 0.3815 |
| DeepSE-FM-00 | 0.8105 | 0.4425 | 0.7828 | 0.3785 |
| DeepSE-FM-01 | 0.8104 | 0.4423 | 0.7833 | 0.3783 |
| DeepSE-FM-10 | 0.8100 | 0.4427 | 0.7810 | 0.3809 |
| DeepSE-FM-11 | 0.8099 | 0.4428 | 0.7805 | 0.3807 |

总体而言，我们无法得出明显的结论，但可以发现以下经验性观察：

- 在Criteo数据集上，浅层模型中组合"11"优于其他类型的组合。然而，在深度模型中组合"11"表现最差。
- 深度模型中的首选组合应为"01"。这种组合意味着双线性函数仅应用于类似SENET的嵌入层，这有助于在我们的模型中设计有效的网络架构。

### 4.4 双线性交互的场类型（RQ3）

**表4：双线性交互层不同场类型的性能。**

| 场类型 | Criteo AUC | Criteo Logloss | Avazu AUC | Avazu Logloss |
|--------|-----------|---------------|----------|--------------|
| SE-FM-All | 0.8021 | 0.4495 | 0.7804 | 0.3800 |
| SE-FM-Each | 0.8037 | 0.4479 | 0.7797 | 0.3812 |
| SE-FM-Interaction | 0.8059 | 0.4460 | 0.7785 | 0.3815 |
| DeepSE-FM-All | 0.8103 | 0.4423 | 0.7832 | 0.3786 |
| DeepSE-FM-Each | 0.8104 | 0.4423 | 0.7833 | 0.3783 |
| DeepSE-FM-Interaction | 0.8105 | 0.4421 | 0.7828 | 0.3788 |

在本节中，我们研究双线性交互层不同场类型（Field-All、Field-Each和Field-Interaction）的影响。我们首先固定双线性交互层中双线性函数和Hadamard积的组合。深度模型的双线性交互层组合设为"01"，浅层模型设为"11"。"01"和"11"的含义在第4.3节中说明。我们将实验结果总结在表4中，并有以下观察：

- 对于浅层模型，与Field-All类型（表1中）相比，Field-Interaction类型在Criteo数据集上AUC提升了0.382%（相对提升0.476%）。
- 对于深度模型，与Field-All类型（表2中）相比，Criteo数据集的Field-Interaction类型和Avazu数据集的Field-Each类型可以分别获得一些改进。
- 双线性交互层不同场类型的性能取决于数据集。在Criteo数据集上，性能排序为：Field-Interaction、Field-Each、Field-All。而在Avazu数据集上，我们无法得出明显结论。

### 4.5 超参数研究（RQ4）

在本小节中，我们将对模型中的一些超参数进行研究。我们关注FiBiNET中以下两个组件的超参数：嵌入部分和DNN部分。具体来说，我们更改以下超参数：（1）嵌入维度；（3）DNN每层神经元数量；（4）DNN深度。除非本文特别说明，网络默认参数如第4.1.4节所述。

**表5：不同嵌入大小在Criteo和Avazu数据集上的性能。**

| 嵌入大小 | Criteo AUC | Criteo Logloss | Avazu AUC | Avazu Logloss |
|---------|-----------|---------------|----------|--------------|
| 10 | 0.8104 | 0.4423 | 0.7809 | 0.3801 |
| 20 | 0.8093 | 0.4435 | 0.7810 | 0.3796 |
| 30 | 0.8071 | 0.4460 | 0.7812 | 0.3799 |
| 40 | 0.8071 | 0.4464 | 0.7824 | 0.3790 |
| 50 | 0.8072 | 0.4468 | 0.7833 | 0.3787 |

#### 4.5.1 嵌入部分

我们将嵌入大小从10更改为50，并将实验结果总结在表5中。我们可以发现以下观察：

- 当维度从10扩展到50时，我们的模型在Avazu数据集上可以获得显著提升。
- 在Criteo数据集上，增加嵌入大小时性能会下降。扩大嵌入大小意味着增加嵌入层和DNN部分的参数数量。我们猜测，可能是Criteo数据集比Avazu数据集有更多的特征，这导致了优化困难。

#### 4.5.2 DNN部分

在深度部分，我们可以改变每层神经元数量、DNN深度、激活函数和dropout率。为简洁起见，我们只研究DNN部分每层不同神经元数量和不同深度的影响。

事实上，增加层数可以增加模型复杂度。从图4可以看出，增加层数在开始时可以提高模型性能。然而，如果层数持续增加，性能会下降。这是因为过于复杂的模型容易过拟合。对于Avazu数据集和Criteo数据集，将隐藏层数量设为3是一个不错的选择。

**图4：DNN中不同层数的性能。**

同样，增加每层神经元数量会引入复杂度。在图5中，我们发现Criteo数据集每层设置400个神经元、Avazu数据集每层设置2000个神经元效果更好。

**图5：DNN中每层不同神经元数量的性能。**

### 4.6 消融研究（RQ5）

**表6：FiBiNET中不同组件的性能。**

| 模型 | Criteo AUC | Criteo Logloss | Avazu AUC | Avazu Logloss |
|------|-----------|---------------|----------|--------------|
| BASE（浅层） | 0.8037 | 0.4479 | 0.7797 | 0.3812 |
| NO-SE | 0.7962 | 0.4552 | 0.7763 | 0.3825 |
| NO-BI | 0.7986 | 0.4525 | 0.7754 | 0.3829 |
| FM | 0.7923 | 0.4584 | 0.7745 | 0.3832 |
| Deep-BASE | 0.8104 | 0.4423 | 0.7833 | 0.3783 |
| Deep-NO-SE | 0.8098 | 0.4427 | 0.7822 | 0.3790 |
| Deep-NO-BI | 0.8093 | 0.4435 | 0.7827 | 0.3785 |
| FNN | 0.8057 | 0.4464 | 0.7802 | 0.3800 |

尽管我们已经展示了强劲的实验结果，但迄今为止的结果尚未分离FiBiNET每个组件的具体贡献。在本节中，我们对FiBiNET进行消融实验，以便更好地理解它们的相对重要性。我们设定"DeepSE-FM-Interaction"作为基础模型，并进行以下操作：1) No BI：从FiBiNET中移除双线性交互层；2) No SE：从FiBiNET中移除SENET层。

如果我们移除SENET层和双线性交互层，我们的浅层FiBiNET和深度FiBiNET将分别退化为FM和FNN。我们可以在表6中发现以下观察：

- 双线性交互层和SENET层对于FiBiNET的性能都是必要的。我们可以看到，当我们移除任何组件时，性能都会明显下降。
- 双线性交互层在FiBiNET中与SENET层同等重要。



## 5 结论

受现有最先进模型缺点的启发，我们提出了一种名为FiBiNET（Feature Importance and Bilinear feature Interaction NETwork，特征重要性与双线性特征交互网络）的新模型，旨在动态学习特征重要性和细粒度特征交互。我们提出的FiBiNET在以下方面为提高性能做出了贡献：1) 对于CTR任务，SENET模块可以动态学习特征的重要性。它增强重要特征的权重并抑制不重要特征的权重。2) 我们引入了三种类型的双线性交互层来学习特征交互，而不是使用Hadamard积或内积计算特征交互。3) 在我们的浅层模型中结合SENET机制与双线性特征交互，优于其他浅层模型（如FM和FFM）。4) 为了进一步提升性能，我们将经典深度神经网络（DNN）组件与浅层模型结合为深度模型。深度FiBiNET持续优于其他最先进的深度模型，如DeepFM和xDeepFM。



## 参考文献

[1] Heng-Tze Cheng, Levent Koc, Jeremiah Harmsen, Tal Shaked, Tushar Chandra, Hrishi Aradhye, Glen Anderson, Greg Corrado, Wei Chai, Mustafa Ispir, et al. 2016. Wide & deep learning for recommender systems. In Proceedings of the 1st Workshop on Deep Learning for Recommender Systems. ACM, 7–10.

[2] Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. 2014. **Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation.** arXiv:cs.CL/1406.1078.

[3] Thore Graepel, Joaquin Quinonero Candela, Thomas Borchert, and Ralf Herbrich. 2010. **Web-scale bayesian click-through rate prediction for sponsored search advertising in microsoft's bing search engine**. Omnipress.

[4] Huifeng Guo, Ruiming Tang, Yunming Ye, Zhenguo Li, and Xiuqiang He. 2017. DeepFM: a factorization-machine based neural network for CTR prediction. arXiv preprint arXiv:1703.04247 (2017).

[5] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. 2016. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition. 770–778.

[6] Xiangnan He and Tat-Seng Chua. 2017. Neural Factorization Machines for Sparse Predictive Analytics. In Proceedings of the 40th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '17). ACM, New York, NY, USA, 355–364. https://doi.org/10.1145/3077136.3080777

[7] Xinran He, Junfeng Pan, Ou Jin, Tianbing Xu, Bo Liu, Tao Xu, Yanxin Shi, Antoine Atallah, Ralf Herbrich, Stuart Bowers, et al. 2014. Practical lessons from predicting clicks on ads at facebook. In Proceedings of the Eighth International Workshop on Data Mining for Online Advertising. ACM, 1–9.

[8] Jie Hu, Li Shen, and Gang Sun. 2017. **Squeeze-and-excitation networks**. arXiv preprint arXiv:1709.01507 7 (2017).

[9] Yuchin Juan, Damien Lefortier, and Olivier Chapelle. 2017. **Field-aware factorization machines in a real-world online advertising system**. In Proceedings of the 26th International Conference on World Wide Web Companion. International World Wide Web Conferences Steering Committee, 680–688.

[10] Yuchin Juan, Yong Zhuang, Wei-Sheng Chin, and Chih-Jen Lin. 2016. Field-aware factorization machines for CTR prediction. In Proceedings of the 10th ACM Conference on Recommender Systems. ACM, 43–50.

[11] Diederik P Kingma and Jimmy Ba. 2014. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980 (2014).

[12] Shunsuke Kitada and Hitoshi Iyatomi. 2018. Skin lesion classification with ensemble of squeeze-and-excitation networks and semi-supervised learning. arXiv preprint arXiv:1809.02568 (2018).

[13] Yehuda Koren, Robert Bell, and Chris Volinsky. 2009. Matrix factorization techniques for recommender systems. Computer 8 (2009), 30–37.

[14] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. 2012. Imagenet classification with deep convolutional neural networks. In Advances in neural information processing systems. 1097–1105.

[15] Jianxun Lian, Xiaohuan Zhou, Fuzheng Zhang, Zhongxia Chen, Xing Xie, and Guangzhong Sun. 2018. **xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems**. arXiv preprint arXiv:1803.05170 (2018).

[16] Drew Linsley, Dan Scheibler, Sven Eberhardt, and Thomas Serre. 2018. Global-and-local attention networks for visual recognition. arXiv preprint arXiv:1805.08819 (2018).

[17] H Brendan McMahan, Gary Holt, David Sculley, Michael Young, Dietmar Ebner, Julian Grady, Lan Nie, Todd Phillips, Eugene Davydov, Daniel Golovin, et al. 2013. Ad click prediction: a view from the trenches. In Proceedings of the 19th ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 1222–1230.

[18] Tomáš Mikolov, Martin Karafiát, Lukáš Burget, Jan Černockỳ, and Sanjeev Khudanpur. 2010. **Recurrent neural network based language model**. In Eleventh Annual Conference of the International Speech Communication Association.

[19] Steffen Rendle. 2010. Factorization machines. In Data Mining (ICDM), 2010 IEEE 10th International Conference on. IEEE, 995–1000.

[20] Steffen Rendle. 2012. Factorization machines with libfm. ACM Transactions on Intelligent Systems and Technology (TIST) 3, 3 (2012), 57.

[21] Abhijit Guha Roy, Nassir Navab, and Christian Wachinger. 2018. **Recalibrating Fully Convolutional Networks with Spatial and Channel 'Squeeze & Excitation' Blocks**. arXiv preprint arXiv:1808.08127 (2018).

[22] Ruoxi Wang, Bin Fu, Gang Fu, and Mingliang Wang. 2017. Deep & cross network for ad click predictions. In Proceedings of the ADKDD'17. ACM, 12.

[23] Jun Xiao, Hao Ye, Xiangnan He, Hanwang Zhang, Fei Wu, and Tat-Seng Chua. 2017. **Attentional factorization machines: Learning the weight of feature interactions via attention networks**. arXiv preprint arXiv:1708.04617 (2017).

[24] Chaochao Yan, Jiawen Yao, Ruoyu Li, Zheng Xu, and Junzhou Huang. 2018. Weakly Supervised Deep Learning for Thoracic Disease Classification and Localization on Chest X-rays. In Proceedings of the 2018 ACM International Conference on Bioinformatics, Computational Biology, and Health Informatics. ACM, 103–110.

[25] Weinan Zhang, Tianming Du, and Jun Wang. 2016. **Deep learning over multi-field categorical data**. In European conference on information retrieval. Springer, 45–57.

[26] Guorui Zhou, Xiaoqiang Zhu, Chenru Song, Ying Fan, Han Zhu, Xiao Ma, Yanghui Yan, Junqi Jin, Han Li, and Kun Gai. 2018. Deep interest network for click-through rate prediction. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 1059–1068.



[^1]: 场嵌入也称为特征嵌入。如果场是多值的，则使用特征嵌入之和作为场嵌入。为与先前文献保持一致，我们在某些术语中保留"feature"，例如特征交互（feature interaction）和特征表示（feature representation）。

[^2]: http://labs.criteo.com/downloads/download-terabyte-click-logs/

[^3]: http://www.kaggle.com/c/avazu-ctr-prediction

[^4]: https://www.tensorflow.org/
