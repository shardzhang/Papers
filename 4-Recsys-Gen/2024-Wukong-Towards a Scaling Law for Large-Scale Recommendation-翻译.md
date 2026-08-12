# Wukong: Towards a Scaling Law for Large-Scale Recommendation（中文翻译）


本文介绍了 Wukong: Towards a Scaling Law for Large-Scale Recommendation。核心内容：


关键发现：

---


> Buyun Zhang*、Liang Luo*、Yuxin Chen*、Jade Nie、Xi Liu、Daifeng Guo、Yanli Zhao、Shen Li、Yuchen Hao、Yantao Yao、Guna Lakshminarayanan、Ellie Dingqiao Wen、Jongsoo Park、Maxim Naumov、Wenlin Chen
> Meta AI
> *共同第一作者。通讯作者：Buyun Zhang <buyunz@meta.com>, Liang Luo <liangluo@meta.com>, Yuxin Chen <yuxinc@meta.com>
> ICML 2024

---

## 摘要

缩放定律在模型质量的可持续改进中发挥着重要作用。不幸的是，迄今为止的推荐模型并不展现出类似大型语言模型领域中观察到的那些定律，这是由于它们的上缩放机制的低效性。这一限制在使这些模型适应日益复杂的真实世界数据集方面带来了重大挑战。在本文中，我们提出了一种有效的、纯粹基于堆叠因子分解机的网络架构，以及一种协同的上缩放策略，统称为 **Wukong**，以在推荐领域建立缩放定律。Wukong 的独特设计使其能够仅通过更深和更宽的层来捕获多样化的、任意阶的交互。我们在六个公开数据集上进行了广泛评估，结果表明 Wukong 在质量上一致优于最先进的模型。此外，我们在一个内部大规模数据集上评估了 Wukong 的可扩展性。结果表明，Wukong 在质量上保持了对最先进模型的优越性，同时保持了在两个数量级的模型复杂度（超过 100 GFLOP/样本）上的缩放定律，而先前的工作在此方面有所不足。

**图 1：Wukong 在两个数量级的模型复杂度上优于现有最先进模型，展现了推荐领域的缩放定律，扩展到超过 100 GFLOP/样本。纵轴：相对 LogLoss（%），横轴：GFLOP/样本。Wukong 在最右侧持续下降，而其他模型趋于饱和。**

---

## 1 引言

基于深度学习的推荐系统（DLRS）为当今广泛的在线服务提供动力（Naumov et al., 2019; Wang et al., 2021a; Lian et al., 2021; Liu et al., 2022; Covington et al., 2016）。现代 DLRS 设计用于处理连续稠密特征（如日期）和分类稀疏特征（如用户点击过的帖子历史）的混合。每个稀疏特征通过可训练的嵌入查找表转换为稠密嵌入表示。然后将这些稠密嵌入馈送到交互组件中，该组件旨在捕获特征之间的复杂交互。

虽然现有模型在较小数据集上展示了有前景的精度，但它们适应显著更大数据集的规模和复杂性，以及在模型扩展时维持持续质量改进的能力仍然不太确定。这种可扩展性日益关键，因为现代数据集经历了指数级增长。例如，今天的生产数据集可能包含数千亿的训练样本（Wang et al., 2021a）。此外，基础模型（Bommasani et al., 2021）需要大规模运作以同时处理更大和多个复杂的输入源。因此，对能够同时有效上缩放和下缩放、适应不同数据集大小和计算约束的 DLRS 的需求至关重要。这种可扩展性被包含在所谓的"缩放定律"中（Kaplan et al., 2020）。

迄今为止，DLRS 上缩放的主要趋势是通过**稀疏缩放**，即扩展嵌入表的大小（更多的行和/或更高的维度），以减少碰撞并获得更好的表达能力。因此，DLRS 已达到数万亿的参数（Kang et al., 2020; Mudigere et al., 2021; Lian et al., 2021），其中嵌入表主导了参数计数。不幸的是，这种传统的上缩放方式有一些实际的缺点。仅仅扩展模型的稀疏组件并不能增强其捕获日益增长的特征之间复杂交互的能力。此外，这一趋势明显偏离了硬件发展的趋势，因为下一代加速器的大部分改进在于计算能力（Luo et al., 2018; 2017），而嵌入表查找无法利用这一点。因此，简单地扩展嵌入表会导致基础设施成本过高，且加速器利用率次优，特别是在分布式设置中（Luo et al., 2024）。

我们的工作旨在为推荐模型找到一种替代的缩放机制，可以建立类似 LLM 领域中建立的缩放定律。也就是说，我们希望设计一个统一的架构，其质量可以随着数据集大小、计算和参数预算的协同策略而持续改进。我们专注于上缩放交互组件，称为**密集缩放**，以缓解稀疏缩放在质量和效率方面的缺陷。然而，由于各种原因，现有模型无法从这一范式中受益。例如，DLRM 缺乏捕获高阶交互的能力；DCNv2 和 AutoInt+ 缺乏有效的上缩放策略，导致在扩展时收益迅速递减；更进一步，即使使用了现代技巧如残差连接（He et al., 2016）、层归一化（Ba et al., 2016）、梯度裁剪（Pascanu et al., 2013），上缩放现有模型仍容易出现训练稳定性问题（Tang et al., 2023）。

为建立推荐模型的缩放定律，我们提出了 **Wukong**，一个简单的交互架构，展现出有效的密集缩放属性。受二进制幂运算原理的启发，我们的关键创新是使用一系列堆叠的因子分解机（FM）来高效且可扩展地捕获任意阶的特征交互。在我们的设计中，每个 FM 负责捕获相对于其输入的二阶交互，然后这些 FM 的输出由 MLP 转换为新的嵌入，这些嵌入编码了交互结果并作为下一层的输入。

我们在六个公开数据集和一个内部大规模数据集上评估了 Wukong 的性能。结果表明，Wukong 在 AUC 方面在所有公开数据集上优于最先进的模型，表明 Wukong 架构的有效性及其在广泛推荐任务和数据集上泛化的能力。在我们的内部数据集上，Wukong 不仅在相当复杂度水平上在质量上显著优于现有模型，而且当在两个数量级的模型复杂度（超过 100 GFLOP/样本）上缩放时，显示出持续的质量增强，而先前的工作在此方面有所不足。

---

## 2 相关工作

**深度学习推荐系统（DLRS）。** 现有的 DLRS 共享相似的结构。一个典型的模型由稀疏和稠密组件组成。稀疏组件本质上是嵌入查找表，将稀疏分类特征转换为稠密嵌入，而稠密组件负责捕获这些嵌入之间的交互以生成预测。

**稠密交互架构。** 捕获特征之间的交互是 DLRS 有效性的关键，我们重点介绍一些先前的工作。AFN+（Cheng et al., 2020）将特征转换到对数空间以捕获任意阶交互；AutoInt+（Song et al., 2019）使用多头自注意力；DLRM 和 DeepFM（Naumov et al., 2019; Guo et al., 2017）利用因子分解机（FM）（Rendle, 2010）显式捕获二阶交互；HOFM（Blondel et al., 2016）优化 FM 以高效捕获更高阶交互；DCNv2（Wang et al., 2021a）使用 CrossNet，通过堆叠的特征交叉捕获交互，可以视为一种逐元素的输入注意力形式；FinalMLP（Mao et al., 2023）采用双线性融合来聚合来自两个 MLP 流的结果，每个流以流特定的门控特征为输入；MaskNet（Wang et al., 2021b）采用一系列 MaskBlock 进行交互捕获，将"输入注意力"应用于输入本身和 DNN 的中间激活；xDeepFM（Lian et al., 2018）将 DNN 与压缩交互网络结合，通过外积和逐元素求和压缩结果来捕获交互。

**缩放 DLRS。** Kang et al.（2020）、Mudigere et al.（2021）、Lian et al.（2021）提供了稀疏缩放的机制。Shin et al.（2023）专注于缩放用户表示模型，Zhang et al.（2023）旨在改进用户端的序列建模。此外，Ardani et al.（2022）研究了 DLRM 的缩放定律。Zhao et al.（2023b）提出了以用户为中心的排序形式化以提高可扩展性；Guo et al.（2023）提供了稀疏缩放的见解，展示了先前工作的局限，与我们的工作互补。此外，VIP5（Geng et al., 2023）利用 LLM 中现有的缩放定律将多模态 LLM 应用于推荐，然而 Lin et al.（2023）指出需要进一步研究验证在 LLM 驱动的推荐中"更大是否意味着更好"，而 Huang et al.（2024）建议需要在更多样化的数据集上进行评估才能得出结论。

---

## 3 Wukong 的设计

在设计 Wukong 的架构时，我们保持两个目标：（1）有效捕获复杂的高阶特征交互；（2）确保 Wukong 的质量随数据集大小、GFLOP/样本和参数预算优雅地缩放。

### 3.1 概述

在 Wukong 中，分类和稠密特征首先通过嵌入层（第 3.2 节），将这些输入转换为稠密嵌入。如图 2 所示，Wukong 随后采用**交互堆叠**（第 3.3 节），即一组统一的神经网络层堆叠，用于捕获嵌入之间的交互。交互堆叠的灵感来自二进制幂运算的概念，允许每个连续层捕获指数级更高阶的交互。交互堆叠中的每一层由一个**因子分解机块（FMB）**（第 3.4 节）和一个**线性压缩块（LCB）**（第 3.5 节）组成。FMB 和 LCB 独立地接收上一层的输入，它们的输出集成为当前层的输出。在交互堆叠之后是一个最终的多层感知器（MLP）层，将交互结果映射为预测。

**图 2：Wukong 采用交互堆叠来捕获特征交互。堆叠中的每一层由一个因子分解机块和一个线性压缩块组成。**

### 3.2 嵌入层

给定多热分类输入，嵌入表将其映射为稠密嵌入。该过程涉及一系列查找，每个对应输入中的一个"热"维度。查找结果然后通过池化操作（通常是求和）进行聚合。

在我们的设计中，嵌入层生成的所有嵌入的维度是标准化的，称为**全局嵌入维度 d**。为适应不同特征的不同重要性，对被认为重要的特征生成多个嵌入。不太重要的特征被分配较小的底层嵌入维度。这些较小的嵌入然后被集体分组、拼接，并通过 MLP 转换为 d 维嵌入。

稠密输入由 MLP 转换为与嵌入共享相同 d 维度的潜在嵌入，并与分类输入的嵌入输出合并。这产生一个形状为 $X_0$
$$
$
$$
 $\in
$$
$
$$
$ ℝⁿ
$$
$
$$
 $
$$
$
$$
$
$$
$
$$
 $
$$
$
$$
$
$$
$
$$
 $
$$
$
$$
$
$$
$
$$
 $
$$
$
$$
$
$$
$
$$
 $
$$
$
$$
$
$$
$
$$
 $
$$
$
$$
$
$$
$
$$
 $
$$
$
$$
$
$$
$
$$
 $
$$
$
$$
$
$$
$
$$
 $
$$
$
$$
$
$$
$
$$
 $
$$
$
$$
$
$$
\times
$$
 $
$$
$
$$
$
$$
$
$$
 $
$$
$
$$
$
$$
$
$$
 $
$$
$
$$
$
$$
$
$$
 $
$$
$
$$
$
$$
$
$$
 $
$$
$
$$
$
$$
$
$$
 $
$$
$
$$
$
$$
$
$$
 $
$$
$
$$
$
$$
$
$$
 $
$$
$
$$
$
$$
$
$$
 $
$$
$
$$
$
$$
$
$$
 $
$$
$
$$
$
$$
$
$$
ᵈ 的输出张量，其中 n 是来自稠密和稀疏部分的嵌入总数。然后 $X_0$ 准备好由交互堆叠进一步处理。注意，与 DCN 等传统方法不同（Wang et al., 2021a），我们将每个嵌入向量解释为一个完整的单元（稍后详细说明），因此我们的表示为 X
$$
\in
$$
 ℝⁿ
$$
\times
$$
ᵈ 而不是 X
$$
\in
$$
 ℝⁿᵈ。

### 3.3 交互堆叠

交互模块堆叠 l 个相同的交互层，其中每一层捕获渐进式更高阶的特征交互，使用因子分解机（FM）。

一个交互层有两个并行的块：**因子分解机块（FMB）**和**线性压缩块（LCB）**。FMB 计算输入嵌入之间的特征交互，LCB 简单地转发线性压缩后的输入嵌入。FMB 和 LCB 的输出然后被拼接。

对于堆叠中的第 i 层，其结果可以包含从 1 到 2ⁱ 的任意阶的特征交互。这可以通过归纳法简单证明。假设第 i 层的输入包含从 1 到 2^{i-1} 阶的交互，这对于第一层（即 i=1）成立。由于 FMB 给定 $o_1$ 和 $o_2$ 阶交互时生成 ( $o_1$ + $o_2$ ) 阶特征交互，那么我们立即得到第 i 层的输出包含从 1 到 2ⁱ 阶的交互，下界来自 LCB 的输出，上界由 FM 对输入中两个 2^{i-1} 阶交互进行交互得到。

为帮助稳定训练，我们还在层间采用残差连接，后接层归一化（LN）。综合起来，我们有：

 $X_{i+1}$ = LN(concat(FMB(X_i), LCB(X_i)) + X_i)

根据 FMB 和 LCB 的具体配置， $X_{i+1}$ 可能具有与 X_i 不同数量的嵌入，这通常发生在第一层。为处理这种情况，残差可以被线性压缩以匹配形状。

### 3.4 因子分解机块（FMB）

FMB 包含一个 FM 后接一个 MLP。FM 用于捕获输入嵌入的显式特征交互，输出是一个 2D 交互矩阵，其中每个元素表示一对嵌入之间的交互。这个交互矩阵被展平并通过 MLP 转换为形状为 (n_F
$$
\times
$$
 d) 的向量，然后重塑为 n_F 个嵌入供后续使用。

操作上，FMB 执行以下操作：

FMB(X_i) = reshape(MLP(LN(flatten(FM(X_i)))))

Wukong 的 FM 模块是完全可定制的：例如在最基本的版本中，我们遵循 DLRM（Naumov et al., 2019）中 FM 的设计，即取所有嵌入向量对之间的点积：FM(X) = XXᵀ。我们在第 3.6 节讨论更优化的 FM 设计。

### 3.5 线性压缩块（LCB）

LCB 简单地将嵌入进行线性重组而不增加交互阶数，这对于确保交互阶数的不变性在层间维持至关重要。具体来说，它保证了第 i 个交互层捕获从 1 到 2ⁱ 的交互阶数。LCB 执行的操作可以描述为：

LCB(X_i) = W_L X_i

其中 W_L
$$
\in
$$
 ℝ^{n_L
$$
\times
$$
 n_i} 是一个权重矩阵，n_L 是一个超参数，表示压缩嵌入的数量，n_i 是第 i 层的输入嵌入数。

### 3.6 优化的 FM

FM 的计算和存储复杂度随嵌入数量呈平方增长，因为在成对点积之后，这在具有数千个特征的真实世界数据集上很快变得不可行。

为允许有效的特征交互同时降低计算成本，我们采用了类似于 Sharma（2023）的方案，利用成对点积矩阵中的低秩性质，这在许多真实世界数据集中已被观察到（Wang et al., 2021a）。

当 d <= n 时，点积交互 XXᵀ 是一个秩为 d 的矩阵，这在特征数大于嵌入维度的大数据集上通常是成立的。因此，我们可以通过将 XXᵀ 与一个可学习的投影矩阵 Y（形状为 n
$$
\times
$$
k）相乘（即计算 XXᵀY），有效地将输出矩阵的大小从 n
$$
\times
$$
n 减少到 n
$$
\times
$$
k，其中 k 是一个超参数，理论上不会损失信息。这将存储交互矩阵的内存需求降低。然后我们可以利用结合律先计算 XᵀY，进一步将计算复杂度从 O(n²d) 降低到 O(nkd)，其中 k << n。

此外，为增强模型质量，可以通过将经过线性压缩的输入通过 MLP 处理，使投影矩阵 Y 对输入具有注意力机制。我们在以下实验中默认使用优化后的 FM，除非另有说明。

### 3.7 复杂度分析

我们假设交互堆叠中的每一层使用相同的超参数，并且 MLP 中的最大全连接层大小为 h。对于第一层，FMB 的时间复杂度是 FM 和 MLP 的和，即 O(nkd) $\approx$ O(ndh) 和 O(nkh + h² + n_F dh) $\approx$ O(ndh + h²)。LCB 的时间复杂度是 O(n n_L d) $\approx$ O(ndh)。对于后续层，时间复杂度是 O(n'dh + h²)，其中 n' = n_L + n_F。因此，Wukong 的总时间复杂度为 O(ndh + ln'dh + h²) $\approx$ O(ndh log n + h²)。

### 3.8 缩放 Wukong

我们现在总结与上缩放相关的主要超参数，然后描述我们相对于这些超参数上缩放 Wukong 的努力。

主要超参数：
- **l**：交互堆叠中的层数
- **n_F**：FMB 生成的嵌入数
- **n_L**：LCB 生成的嵌入数
- **k**：优化 FM 中压缩嵌入数
- **MLP**：FMB 中 MLP 的层数和全连接层大小

在上缩放过程中，我们最初专注于增加 **l** 以使模型能够捕获更高阶的交互。随后，我们扩大其他超参数以增强模型捕获更广泛交互的能力。

### 3.9 Wukong 增强有效性的直觉

与使用 FM 作为其主要交互架构的现有工作相比，Wukong 堆叠 FM 的创新方法极大地增强了传统 FM 的能力。这使得 Wukong 能够捕获任意阶的交互，使其对需要高阶推理的大规模复杂数据集非常有效。虽然有实现高阶 FM 的工作，但 Wukong 以指数速率捕获高阶交互提供了极高的效率，绕过了 HOFM 中看到的线性复杂度并避免了 xDeepInt 中昂贵的外积。

虽然 MLP 在隐式捕获交互方面显示出局限性（Beutel et al., 2018），但 Wukong 避开了依赖 MLP 进行交互捕获的方法。相反，Wukong 主要使用 MLP 将交互结果转换为嵌入表示，然后用于进一步的交互。这种对 MLP 的独特使用增强了模型处理和解释复杂异构特征的能力。

此外，Wukong 将每个嵌入视为一个单一单元，关注**嵌入级**交互。与捕获逐元素交互的架构相比，这种方法显著降低了计算需求。

---

## 4 实现

本节讨论在大规模数据集上有效训练高复杂度 Wukong 的实践。

为使得 Wukong 的训练可行，需要分布式训练。对于嵌入层，我们使用 Neo（Mudigere et al., 2021）和 NeuroShard（Zha et al., 2023）提供的按列分片的嵌入袋实现。在稠密部分，我们通过采用 FSDP（Zhao et al., 2023a）平衡性能与内存容量之间的权衡，调整分片因子以使模型适合内存而不产生过多冗余。

为提高训练效率，我们采用自动算子融合来提升训练性能。此外，我们积极应用量化以同时降低计算、内存和通信开销。具体来说，我们以 FP16 训练 Wukong 的嵌入表，在前向传播中以 FP16 通信嵌入查找结果，在后向传播中以 BF16 通信；在后向传播期间，我们在稠密参数的梯度传输中使用 BF16 量化。

---

## 5 评估概述

我们使用六个公开数据集和一个内部数据集评估 Wukong，其详细情况总结在表 1 中。评估结果分为两部分。

**表 1：我们评估数据集的统计信息。**

| 数据集 | #样本 | #特征 |
|--------|-------|-------|
| Frappe | 0.29M | 10 |
| MicroVideo | 1.7M | 7 |
| MovieLensLatest | 2M | 3 |
| KuaiVideo | 13M | 8 |
| TaobaoAds | 26M | 21 |
| CriteoTerabyte | 4B | 39 |
| **Internal** | **146B** | **720** |

在第 6 节中，我们在六个公开数据集上进行评估，重点展示 Wukong 在低复杂度领域的有效性。我们的结果表明，Wukong 在所有六个数据集上超过了先前的最先进方法，证明了其有效性。

在第 7 节中，我们在大规模内部数据集上评估，以展示 Wukong 的可扩展性。该数据集包含比最大数据集之一 Criteo 多 30 倍的样本和 20 倍的特征。我们的结果揭示了：（1）Wukong 在模型质量和运行速度方面一致优于所有基线模型，在所有复杂度规模上保持这一优越性；（2）Wukong 相比基线模型展现出更好的缩放趋势。我们还进行了消融研究，以理解 Wukong 内各组件的独立贡献和有效性。

---

## 6 公开数据集上的评估

在本节中，我们旨在展示 Wukong 在多种公开数据集上的有效性。除非另有说明，我们使用 BARS 基准（Zhu et al., 2022b）提供的预处理以保持与先前工作的一致性。

### 6.1 通用评估设置

#### 6.1.1 数据集

- **Frappe**（Baltrunas）：一个应用使用日志。该数据集预测用户是否在给定上下文中使用应用。
- **MicroVideo**（Chen et al., 2018）：一个基于内容理解的数据集，包含用户与微视频之间的交互。该日志包含多模态嵌入以及传统特征。
- **MovieLensLatest**（Harper & Konstan, 2015）：一个众所周知的数据集，包含用户对电影的评分。
- **KuaiVideo**（快手）：由快手发布的竞赛数据集。用于预测用户对新微视频的点击概率。该数据集也包含基于内容理解的嵌入以及其他分类和浮点特征。
- **TaobaoAds**（天池, 2018）：该数据集包含 8 天的淘宝广告点击率（CTR）预测数据。
- **CriteoTerabyte**（Criteo）：该数据集包含 24 天的广告点击反馈。我们使用最后一天的数据进行测试。

#### 6.1.2 基线

我们将 Wukong 与七个在学术界和工业界广泛认可的最先进模型进行基准比较，包括 AFN+（Cheng et al., 2020）、AutoInt+（Song et al., 2019）、DLRM（Naumov et al., 2019）、DCNv2（Wang et al., 2021a）、FinalMLP（Mao et al., 2023）、MaskNet（Wang et al., 2021b）和 xDeepFM（Lian et al., 2018）。

#### 6.1.3 指标

- **AUC**：曲线下面积（AUC）衡量模型在所有阈值下正确分类正负样本的能力。越高越好。我们使用 AUC 作为超参数调整的基础和报告的首要指标，遵循推荐惯例（Tien et al., 2014; Blondel et al., 2016; Song et al., 2019; Wang et al., 2021a; Zhu et al., 2022b; Mao et al., 2023）。
- **LogLoss**：对数损失量化了基于预测与真实标签差距的惩罚。越低越好。

### 6.2 特定模型设置

对于五个较小的数据集（Criteo 除外），我们采用公开的 BARS 评估框架（Zhu et al., 2022a；2021）。我们在可能的情况下直接使用 BARS 上搜索到的最佳模型配置，其余使用提供的模型默认超参数。除了框架中提供的默认嵌入维度外，我们进一步测试了嵌入维度为 128 的情况，并报告两个配置中较好的结果。

对于 Wukong，我们调整了 dropout 率和优化器设置以及 LCB 的压缩率，以适应特征数量。我们利用较大的 Criteo 数据集评估模型在真实在线推荐系统上的性能，采用单遍训练。鉴于新的训练设置，我们使用第 4 节中描述的系统对所有基线和 Wukong 进行了广泛的网格搜索，以促进公平比较。这一详尽的过程涉及近 3000 次独立运行。我们在附录 A 中提供了模型特定的搜索空间。搜索到的最佳模型超参数后来被用作第 7 节中基本配置。

### 6.3 结果

**表 2：六个公开数据集上的评估结果。每个数据集上具有最佳 AUC 和最佳 LogLoss 的模型已高亮显示。**

| 模型 | Frappe | MicroVideo | MovieLensL. | KuaiVideo | TaobaoAds | CriteoTB |
|------|-------|-------|-----------|----------|----------|----------|
| | AUC / LogLoss | AUC / LogLoss | AUC / LogLoss | AUC / LogLoss | AUC / LogLoss | AUC / LogLoss |
| **基线** | | | | | | |
| AFN+ | .9812/.2340 | .7220/.4142 | .9648/.3109 | .7348/.4372 | .6416/.1929 | .8023/.1242 |
| AutoInt+ | .9806/.1754 | .7155/.4203 | .9693/.2178 | .7297/.4376 | .6437/.1930 | .8073/.1233 |
| DCNv2 | .9774/.2325 | .7187/.4162 | .9683/.2169 | .7360/.4383 | .6457/.1926 | .8096/.1227 |
| DLRM | .9846/.1465 | .7173/.4179 | .9685/.2160 | .7357/.4382 | .6430/.1931 | .8076/.1232 |
| FinalMLP | .9868/.1280 | .7247/.4147 | .9723/.2211 | .7374/.4435 | .6434/.1928 | .8096/.1226 |
| MaskNet | .9816/.1701 | .7255/.4157 | .9676/.2383 | .7376/.4372 | .6433/.1927 | .8100/.1227 |
| xDeepFM | .9780/.2441 | .7167/.4172 | .9667/.2089 | .7118/.4565 | .6342/.1961 | .8084/.1229 |
| **Wukong** | **.9868**/.1757 | **.7292**/.4148 | **.9723**/**.1794** | **.7414**/.4367 | **.6488**/.1954 | **.8106**/.1225 |

总体而言，Wukong 在 AUC 方面能够在所有公开数据集上取得最先进的结果。这一结果证明了 Wukong 架构的有效性及其理解不同数据集和在广泛推荐任务上泛化的能力。

---

## 7 内部数据集上的评估

在本节中，我们使用大规模数据集展示 Wukong 的可扩展性，并深入理解 Wukong 不同组件对其有效性的贡献，该数据集能够实现在小型公开数据集中看不到的涌现特性研究。

### 7.1 评估设置

#### 7.1.1 数据集

该数据集共包含 146B 条目，具有 720 个不同的特征。每个特征描述item或用户的属性。与该数据集相关的有两个任务：（任务 1）预测用户是否对某个item表现出兴趣（例如点击）；（任务 2）是否发生了转化（例如点赞、关注）。

#### 7.1.2 指标

- **GFLOP/样本**：每样本千兆浮点运算次数（GFLOP/样本）量化了模型训练期间的计算复杂度。
- **PF-天**：总训练计算量，相当于以 1 PetaFLOP/s 运行一台机器 1 天。
- **#参数**：模型中的参数数量。稀疏嵌入表大小固定为 627B 参数。
- **相对 LogLoss**：相对于固定基线的 LogLoss 改进。我们选择使用基本配置的 DLRM 作为基线。在该数据集上，0.02% 的相对 LogLoss 改进被认为是显著的。我们报告在线训练期间最后 1B 窗口上的相对 LogLoss。

#### 7.1.3 基线

我们遵循与第 6.1.2 节中详述的相同的基线设置。然而，xDeepFM 未包含在报告的结果中，因为其昂贵的外积操作与大规模数据集不兼容，即使在最小设置中也持续导致内存不足问题。

#### 7.1.4 训练

我们在所有实验中使用在初步研究中发现的最佳优化器配置：稠密部分使用 Adam（lr=0.04, beta1=0.9, beta2=1），稀疏嵌入表使用 Rowwise Adagrad（lr=0.04）。模型以在线训练方式进行训练和评估。我们在所有运行中固定嵌入维度为 160。

我们将第 6 节中描述的 CriteoTerabyte 评估中搜索到的最佳配置超参数设置为起点，并逐步增加每个模型的参数计数。所有实验使用全局批大小为 262,144。每个实验根据模型大小在 128 或 256 块 H100 GPU 上运行。

### 7.2 结果

我们观察到两个任务的结果相当，并在正文中报告任务 1 的结果，任务 2 的详细结果在附录 C 中提供。

**质量 vs. 计算复杂度。** 在图 1 中，我们描绘了质量与计算复杂度之间的关系（经验上 y = −100 + 99.56x^0.00071）。结果表明，Wukong 在各种复杂度水平上一致优于所有基线，实现了超过 0.2% 的 LogLoss 改进。值得注意的是，Wukong 保持了缩放定律，而 AFN+、DLRM 和 FinalMLP 在达到一定复杂度水平后趋于饱和，AutoInt+、DCNv2 和 MaskNet 未能进一步提高质量。即使是最好的基线 DCNv2，也需要增加 40 倍的复杂度才能匹配 Wukong 的质量水平。

**质量 vs. 模型大小。** 在图 3 中，我们展示了模型质量与模型大小之间的相关性。与上述计算复杂度缩放中观察到的趋势一致，Wukong 在所有模型大小规模上一致优于所有基线约 0.2%，同时在超过 6370 亿参数时展现出稳定的改进趋势。

**图 3：Wukong 相对于 #参数的缩放性。横轴：稀疏+稠密参数量（B），纵轴：相对 LogLoss（%）。Wukong 持续优于所有基线。**

**特定模型缩放。** 在整个缩放过程中，我们为每个模型采用了不同的策略。每个运行的详细超参数设置在附录 C 中提供。每个模型的缩放过程总结如下：
- **Wukong**：通过调整第 3.8 节详述的超参数进行上缩放。
- **AFN+**：缩放 AFN 的隐藏层、集成 DNN 和对数神经元的数量。结果表明缩放 AFN 并不能提高模型质量。
- **AutoInt+**：缩放多头注意力和集成 DNN。该模型的模型质量最初比其他模型差，但在扩展时显著改善。
- **DLRM**：缩放顶部 MLP。结果表明质量在超过 31 GFLOP/样本后开始饱和。
- **DCNv2**：缩放 CrossNet 的深度和 MLP 的大小。
- **FinalMLP**：缩放其主干网络的两流 MLP。
- **MaskNet**：缩放 MaskBlock（注意：MaskNet 受到过量内存消耗的阻碍，妨碍了其扩展能力）。

---

## 8 结论

我们提出了 Wukong，一个展示推荐领域缩放定律的新型架构。通过堆叠因子分解机和协同上缩放策略，Wukong 在两个数量级的模型复杂度上实现了持续的性能提升。在六个公开数据集和一个大规模内部数据集上的广泛评估验证了 Wukong 的有效性和可扩展性。我们的工作表明，通过合适的架构设计和缩放策略，推荐模型可以像 LLM 一样建立缩放定律，为构建更大、更强大的推荐系统铺平了道路。

---

## 8 讨论

**实际服务上缩放模型。** 扩展到高复杂度为实时服务带来了显著挑战。潜在解决方案包括训练多任务基础模型以分摊成本：将大模型的知识蒸馏到小型高效模型中进行服务。

**局限性与未来工作。** 我们也注意到工作的局限性和注意事项，这些可以作为未来工作的目标。理解 Wukong 可扩展性的确切限制是一个重要的研究领域。由于巨大的计算需求，我们还未能达到该限制适用的复杂度水平。

虽然 Wukong 在各种评估中展现出优越的质量，但其基本原理的全面理论理解，特别是与共享堆叠点积结构的 Transformer 等架构的对比，仍然是一个需要进一步探索的领域。此外，Wukong 在推荐之外的可泛化性，特别是在涉及异构输入数据源（类似于推荐中不同特征）的领域，仍有待进一步探索和理解。

---

## 9 结论

我们提出了一个有效的网络架构，名为 **Wukong**。我们证明了 Wukong 在推荐领域建立了一个之前未观察到的缩放定律——Wukong 能够在两个数量级的计算复杂度上高效地上缩放和下缩放，同时保持相对于其他最先进模型的竞争优势，使其成为一个可扩展的架构，可以在广泛的任务和数据集上作为从小型垂直模型到大型基础模型的骨干网络。

**影响声明。** 本文提出的工作旨在推动机器学习领域的发展。我们的工作有许多潜在的社会影响，我们认为没有特别需要在此强调的。

---

## 参考文献

[1] Anonymous. Dot product matrix compression for machine learning. Technical Disclosure Commons, 2019.
[2] Ardalani, N., Wu, C.-J., Chen, Z., Bhushanam, B., and Aziz, A. Understanding scaling laws for recommendation models. arXiv:2208.08489, 2022.
[3] Ba, J. L., Kiros, J. R., and Hinton, G. E. Layer normalization. arXiv:1607.06450, 2016.
[4] Baltrunas, L. Frappe - mobile app usage dataset.
[5] Beutel, A. et al. Latent cross: Making use of context in recurrent recommender systems. WSDM 2018.
[6] Blondel, M. et al. Higher-order factorization machines. NeurIPS 2016.
[7] Bommasani, R. et al. On the opportunities and risks of foundation models. arXiv:2108.07258, 2021.
[8] Chen, X. et al. Temporal hierarchical attention for micro-video click prediction. MM 2018.
[9] Cheng, W. et al. Adaptive factorization network. AAAI 2020.
[10] Covington, P. et al. Deep neural networks for youtube recommendations. RecSys 2016.
[11] Criteo. Criteo 1TB click logs dataset.
[12] Geng, S. et al. VIP5: Towards multimodal foundation models for recommendation. arXiv:2305.14302, 2023.
[13] Gui, H. et al. HiFormer: Heterogeneous feature interactions learning with transformers. arXiv:2311.05884, 2023.
[14] Guo, H. et al. DeepFM. arXiv:1703.04247, 2017.
[15] Guo, X. et al. On the embedding collapse when scaling up recommendation models. arXiv:2310.04400, 2023.
[16] Harper, F. M. and Konstan, J. A. The MovieLens datasets. ACM TISS, 2015.
[17] He, K. et al. Deep residual learning for image recognition. CVPR 2016.
[18] Huang, C. et al. Foundation models for recommender systems: A survey. arXiv:2402.11143, 2024.
[19] Kang, W.-C. et al. Learning to embed categorical features without embedding tables. arXiv:2010.10784, 2020.
[20] Kaplan, J. et al. Scaling laws for neural language models. arXiv:2001.08361, 2020.
[21] Kuaishou. KuaiVideo dataset. https://www.kuaishou.com/activity/uimc
[22] Lian, J. et al. xDeepFM. KDD 2018.
[23] Lian, X. et al. Persia: An open, hybrid system scaling deep learning-based recommenders up to 100 trillion parameters. 2021.
[24] Lin, J. et al. How can recommender systems benefit from large language models: A survey. arXiv:2306.05817, 2023.
[25] Liu, Z. et al. Monolith: Real time recommendation system with collisionless embedding table. arXiv:2209.07663, 2022.
[26] Luo, L. et al. Parameter hub: a rack-scale parameter server. SoCC 2018.
[27] Luo, L. et al. Disaggregated multi-tower. arXiv:2403.00877, 2024.
[28] Mao, K. et al. FinalMLP. arXiv:2304.00902, 2023.
[29] Mudigere, D. et al. High-performance distributed training of large-scale deep learning recommendation models. arXiv:2104.05158, 2021.
[30] Naumov, M. et al. Deep learning recommendation model for personalization and recommendation systems. arXiv:1906.00091, 2019.
[31] Pascanu, R. et al. On the difficulty of training recurrent neural networks. ICML 2013.
[32] Rendle, S. Factorization machines. ICDM 2010.
[33] Sharma, S. Feature fusion for the uninitiated. Medium, 2023.
[34] Shin, K. et al. Scaling law for recommendation models: Towards general-purpose user representations. AAAI 2023.
[35] Song, W. et al. AutoInt. CIKM 2019.
[36] Tang, J. et al. Improving training stability for multitask ranking models. arXiv:2302.09178, 2023.
[37] Tianchi. Taobao display/click data. 2018.
[38] Tien, J.-B. et al. Criteo display advertising challenge. Kaggle, 2014.
[39] Wang, R. et al. DCNv2. WWW 2021a.
[40] Wang, Z. et al. MaskNet. arXiv:2102.07619, 2021b.
[41] Zha, D. et al. Pre-train and search: Efficient embedding table sharding. MLSys 2023.
[42] Zhang, G. et al. Scaling law of large sequential recommendation models. arXiv:2311.11351, 2023.
[43] Zhao, Y. et al. PyTorch FSDP. arXiv:2304.11277, 2023a.
[44] Zhao, Z. et al. Breaking the curse of quality with user-centric ranking. arXiv:2305.15333, 2023b.
[45] Zhu, J. et al. Open benchmarking for CTR prediction. CIKM 2021.
[46] Zhu, J. et al. BARS. SIGIR 2022a.
[47] Zhu, J. et al. BARS. SIGIR 2022b.

---

## 附录 A：在 Criteo 上的模型特定网格搜索空间

我们使用 Adam 进行稠密参数优化，使用 Rowwise AdaGrad 进行稀疏参数优化，在前 10% 的步骤中使用线性预热。我们使用 8
$$
\times
$$
16384 = 131,072 的全局批大小。所有模型使用 ReLU 激活。我们选择 128 作为嵌入维度，因为在初步实验中它对所有模型表现更好。所有运行使用 FP32。鉴于数据集规模和模型大小，我们使用 Neo（Mudigere et al., 2021）作为稀疏分布式训练框架，并使用数据并行进行稠密同步。

为促进公平比较，我们在 Criteo 数据集上进行了广泛的网格搜索（>3000 次运行），涵盖通用超参数和模型特定配置。

对于所有模型，稀疏和稠密学习率分别在 {1e−3, 1e−2, 1e−1} 中单独调整。对于所有模型中的 MLP，隐藏层数在 {1, 2, 3, 4} 中变化，层大小在 {512, 1024, 2048} 中变化。为减少过大的搜索空间，我们在优化器超参数上进行了初步实验，发现将稠密学习率设为 1e−3、稀疏学习率设为 1e−1 对所有模型效果最佳。我们在后续运行中固定了学习率。以下是各个模型特定的搜索空间：

**AFN+**：AFN 隐藏单元和 DNN 隐藏单元在所有运行中相同，遵循通用 MLP 搜索空间。对数神经元数量在 {128, 256, 512, 1024} 中变化。

**AutoInt+**：我们基于论文（Song et al., 2019）中报告的最佳配置创建搜索空间，每个超参数额外考虑更大的值。注意力层数在 {3, 4} 中变化，注意力维度在 {256, 512} 中变化。注意力头数在 {4, 8} 中变化。DNN 隐藏单元遵循通用 MLP 搜索空间。

**DCNv2**：交叉层数从 1 到 4 变化。秩设为全秩或 512。

**DLRM**：底部 MLP 层大小和数量设为 [512, 256]。

**FinalMLP**：我们遵循公开基准设置（Zhu et al., 2022a），将一个流的特征选择（FS）设为所有浮点特征，并在另一个流中搜索 8 个选定的稀疏特征之一。FS_MLP 设为 [800]。头数固定为 256。

**MaskNet**：我们测试了并行 MaskNet 和串行 MaskNet。对于并行变体，块数在 {1, 8, 16} 中考虑，块维度在 {64, 128} 中考虑。对于串行变体，层数在 {1, 4, 8} 中考虑，层大小在 {64, 256, 1024} 中考虑。两种变体的缩减率固定为 1。

**xDeepInt**：我们考虑压缩交互网络（CIN），层数在 {3, 4} 中变化，层维度在 {16, 32, 64} 中变化。

**Wukong**：底部 MLP 层大小和数量设为 [512, 256]。**l** 从 1 到 4 变化；**n_F** 和 **n_L** 设为相同值，在 {8, 16} 中变化。**k** 固定为 24。

---

## 附录 B：公开数据集上的模型复杂度/大小

请参见表 3 的详细信息。

**表 3：公开数据集上的模型复杂度和大小。**（完整表格数据见原文，涵盖 Frappe、MicroVideo、MovieLens、KuaiVideo、TaobaoAds、Criteo 六个数据集上每个模型的参数量（#Params）和 MFLOP。）

---

## 附录 C：特定模型上缩放的详细配置

请参见表 5 的详细信息。

**表 5：第 7 节中评估的每个运行的详细超参数、计算复杂度、模型质量和模型大小。**（完整表格见原文，涵盖 AFN+、AutoInt+、DCNv2、DLRM、FinalMLP、MaskNet、Wukong 从低复杂度到高复杂度的所有配置和指标。）

---

## 附录 D：Wukong 中高阶交互的分析

传统的因子分解机方法通过最小化以下目标来解决二阶交互问题（Naumov et al., 2019）：

min $\Sigma$ (r_ij − $X_1$ $X_1$ ᵀ)

其中 r_ij
$$
\in
$$
 ℝ 是第 j 个用户对第 i 个产品的评分，i=1,...,m，j=1,...,n；X 表示用户和item表示（嵌入），上标 1 表示嵌入包含一阶信息。这些嵌入向量的点积为二阶交互的后续评分提供了有意义的预测。

在 Wukong 中，这些有意义的交互然后通过 MLP 转换为二阶交互表示 $X_2$ 。在第 2 层 FMB 中，通过残差和 LCB 连接，( $X_1$ + $X_2$ )( $X_1$ + $X_2$ )ᵀ 的点积产生从一阶到四阶的有意义交互。以此类推，一个 l 层的 Wukong 通过最小化以下目标来解决问题：

min $\Sigma$ (r_ij − $\Sigma$ X_k X_kᵀ)
   i,j
$$
\in
$$
S   k
$$
\in
$$
1,2,...,2^l−1

因此，与传统的因子分解方法相比，Wukong 能够以更充分的交互阶数来解决推荐问题。

---

## 附录 E：训练数据量上的缩放定律

图 6 提供了 Wukong 性能与训练数据集大小（单遍）的总结。与在 LLM 上观察到的情况类似，我们发现更大的模型更数据高效，意味着它们需要更少的样本来实现相同的质量改进。此外，我们发现所有 Wukong 模型在 146B 数据结束时都一致地改善了模型质量，而更大的模型具有更陡峭的改进趋势。我们还注意到，我们研究的一个局限性是数据集大小对于大模型收敛仍然远远不够，这将是未来研究的领域之一。

**图 6：Wukong 的模型质量提升 vs. 训练数据量和训练计算量。**
- 横轴左图：#样本（示例数），右图：计算量（PF-天）
- 纵轴：相对 LogLoss（%）
- 颜色表示 GFLOP/样本
- 更大模型（更高 GFLOP/样本）数据效率更高，趋势更陡峭。

---

## 附录 F：与基于 Transformer 的方法的比较

我们强调差异并提供 Wukong 为何比基于 Transformer 的方法（如 AutoInt+（Song et al., 2019））更可扩展的直觉。虽然 Wukong 的结构类似于 Transformer 的结构，但我们注意到以下架构差异：首先，Wukong 中使用的投影是 MLP（逐位），在 FMB 和每一层中都是如此，而不是 Transformer 中的 FFN（逐嵌入/逐位置）；其次，Wukong 配置为金字塔形状，而不是 Transformer 中使用的均匀形状。

我们假设投影的差异在质量交付中起着重要作用。这些 MLP 在展平的输入嵌入上操作，本质上为每个特征提供了不同的投影矩阵。我们的直觉是，这有助于模型从异构输入特征中学习，与 LLM 中使用的单一嵌入空间形成对比。类似的直觉在（Gui et al., 2023）中也有讨论。

在效率方面，我们认为金字塔形状配置允许 Wukong 通过收缩每层使用的嵌入数量来排除不必要的计算。

为验证这些假设，我们进行了以下实验，将 Wukong 的独特组件应用于 AutoInt+，结果如下：
1. 使用逐位 MLP 代替 FFN 进行 V 投影：LogLoss 改善 0.34%
2. 在自注意力之后添加逐位 MLP：LogLoss 改善 0.65%
3. 结合以上两项并采用金字塔层形状（在第一层输出上使用 LCB）：LogLoss 改善 0.57%

相比上缩放的 AutoInt+，Wukong 实现了 0.08% 的质量提升，同时节省了 90% 的 FLOPs。结果总结在表 4 中。

**表 4：将 Wukong 的独特组件替换/添加到原始 AutoInt+ 中可改善模型质量。**

| 相对于 AutoInt+ 的更改 | 相对 LogLoss (%) | GFLOP/样本 |
|------------------------|-----------------|-----------|
| 原始 AutoInt+ | 0 | 8 |
| V=FFN() $\to$ V=MLP() | -0.34 | 13 |
| 上缩放的 AutoInt+ | -0.49 | 50 |
| V=FFN() $\to$ V=MLP() + 层FFN $\to$ 层MLP + 金字塔形状 | -0.57 | 5 |
| 层FFN $\to$ 层MLP | -0.65 | 36 |
