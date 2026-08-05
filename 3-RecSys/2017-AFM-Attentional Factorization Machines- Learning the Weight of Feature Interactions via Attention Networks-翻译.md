# 注意力分解机：通过注意力网络学习特征交互的权重

> Jun Xiao, Hao Ye, Xiangnan He, Hanwang Zhang, Fei Wu, Tat-Seng Chua | Zhejiang University; National University of Singapore

本文介绍了 AFM（Attentional Factorization Machine，注意力分解机）：一种通过注意力网络学习特征交互权重的新型模型。核心内容：

- 问题动机：FM（Factorization Machine，分解机）对所有特征交互赋予相同权重，而实际中不同交互对预测的贡献差异很大，与无关特征的交互甚至可能引入噪声
- 模型创新：提出成对交互层（Pair-wise Interaction Layer）将 $m$ 个向量扩展为 $m(m-1)/2$ 个交互向量，并通过注意力网络自动学习每个交互的重要性权重
- 过防过拟合：采用成对交互层上的 Dropout 和注意力网络上的 $L_2$ 正则化两种技术防止过拟合
- 实验验证：在 Frappe 和 MovieLens 两个数据集上，AFM 以更少的参数优于 Wide&Deep [8] 和 DeepCross [22]，并比 LibFM 提升 8.6% 的 RMSE

关键发现：

- 注意力机制有效区分特征交互的重要性：在 MovieLens 的微观分析中，item-tag 交互被赋予最高注意力分数，符合直觉判断
- AFM 收敛速度更快且泛化能力更强：在 Frappe 上训练和测试误差均远低于 FM，在 MovieLens 上测试误差更低
- 模型简洁高效：AFM 仅使用 1.45M 参数（Frappe），比 Wide&Deep（4.66M）和 DeepCross（8.93M）少得多，却取得最优性能

---

## 摘要

分解机（FM）是一种通过引入二阶特征交互来增强线性回归模型的监督学习方法。尽管 FM 具有有效性，但它对所有特征交互赋予相同权重的做法可能成为瓶颈，因为并非所有特征交互都同等有用且具有预测价值。例如，与无用特征的交互甚至可能引入噪声并损害性能。在本文中，我们通过区分不同特征交互的重要性来改进 FM。我们提出了一种名为注意力分解机（AFM）的新模型，它通过神经注意力网络从数据中学习每个特征交互的重要性。在两个真实世界数据集上的大量实验表明了 AFM 的有效性。在回归任务上，AFM 比 FM 取得了 8.6% 的相对提升，并以更简单的结构和更少的模型参数持续优于最先进的深度学习方法 Wide&Deep 和 DeepCross。我们的 AFM 实现已公开：https://github.com/hexiangnan/attentional_factorization_machine

## 关键词

Factorization Machines, Attention Mechanism, Feature Interactions, Sparse Data Prediction

## 1 引言

监督学习是机器学习（ML）和数据挖掘中的基本任务之一。其目标是推断一个函数，根据预测变量（即特征）作为输入来预测目标。例如，用于回归的实值目标和用于分类的类别标签。它具有广泛的应用，包括推荐系统 [2, 36]、在线广告 [22, 16] 和图像识别 [34, 25]。

在对类别预测变量进行监督学习时，考虑它们之间的交互非常重要 [14, 8]。作为一个例子，让我们考虑一个用三个类别变量预测客户收入的玩具问题：1）职业 = {银行家, 工程师,...}，2）级别 = {初级, 高级}，3）性别 = {男, 女}。虽然初级银行家的收入低于初级工程师，但对于高级客户来说情况可能相反——高级银行家的收入通常高于高级工程师。如果一个 ML 模型假设预测变量之间独立并忽略它们之间的交互，它将无法准确预测，例如线性回归为每个特征分配一个权重并将目标预测为所有特征的加权和。

为了利用特征之间的交互，一种常见解决方案是显式地用特征的乘积（即交叉特征）来增强特征向量，如多项式回归（PR）中也为每个交叉特征学习一个权重。然而，PR（以及其他基于交叉特征的解决方案，如 Wide&Deep [8] 的 wide 部分）的关键问题是，对于稀疏数据集，只有少数交叉特征被观察到，未观察到的交叉特征的参数无法被估计。

为了解决 PR 的泛化问题，提出了分解机（FM）[19]，它将交叉特征的权重参数化为组成特征的嵌入向量的内积。通过为每个特征学习一个嵌入向量，FM 可以估计任何交叉特征的权重。由于这种泛化能力，FM 已成功应用于各种应用，从推荐系统 [27, 4, 36, 2] 到自然语言处理 [18]。尽管前景广阔，我们认为 FM 可能因其对所有分解交互赋予相同权重的建模方式而受到阻碍。在实际应用中，不同的预测变量通常具有不同的预测能力，并非所有特征都包含用于估计目标的有用信号，例如前述示例中用于预测客户收入的性别变量。因此，与不太有用的特征的交互应被赋予较低的权重，因为它们对预测的贡献较小。然而，FM 缺乏区分特征交互重要性的能力，这可能导致次优预测。

在本文中，我们通过区分特征交互的重要性来改进 FM。我们设计了一种名为 AFM 的新模型，它利用神经网络建模的最新进展——注意力机制 [5, 6]——使特征交互对预测的贡献不同。更重要的是，特征交互的重要性是从数据中自动学习的，无需任何人工领域知识。我们在两个公开的上下文感知预测 [1] 和个性化标签推荐 [9] 基准数据集上进行了实验。大量实验表明，我们在 FM 上使用注意力机制有两个好处：它不仅带来更好的性能，还提供了关于哪些特征交互对预测贡献更大的洞察。这大大增强了 FM 的可解释性和透明度，允许从业者对其行为进行更深入的分析。

## 2 分解机

作为一种通用的监督学习 ML 模型，分解机最初是为协同推荐而提出的 [19, 20]。给定一个实值特征向量 $\mathbf{x} \in \mathbb{R}^n$，其中 $n$ 表示特征的数量，FM 通过建模每对特征之间的所有交互来估计目标：

$$
\hat{y}_{\text{FM}}(\mathbf{x}) = w_0 + \sum_{i=1}^{n} w_i x_i + \sum_{i=1}^{n}\sum_{j=i+1}^{n} \hat{w}_{ij} x_i x_j \qquad (1)
$$

其中 $w_0$ 是全局偏置，$w_i$ 表示第 $i$ 个特征的权重，$\hat{w}_{ij}$ 表示交叉特征 $x_i x_j$ 的权重，它被分解为：$\hat{w}_{ij} = \mathbf{v}_i^{\top} \mathbf{v}_j$，其中 $\mathbf{v}_i \in \mathbb{R}^k$ 表示特征 $i$ 的嵌入向量，$k$ 表示嵌入向量的大小。注意，由于系数 $x_i x_j$，只考虑非零特征之间的交互。

值得注意的是，FM 以相同的方式建模所有特征交互：首先，一个潜在向量 $\mathbf{v}_i$ 在估计第 $i$ 个特征涉及的所有特征交互时共享；其次，所有估计的特征交互 $\hat{w}_{ij}$ 具有统一的权重 1。在实践中，并非所有特征都与预测相关是很常见的。作为一个例子，考虑新闻分类问题，句子为"美国继续在对外支付透明度方面发挥主导作用"。很明显，除了"对外支付透明度"之外的词并不指示（金融）新闻的主题。那些涉及不相关特征的交互可以被视为对预测没有贡献的噪声。然而，FM 以相同的权重建模所有可能的特征交互，这可能不利地损害其泛化性能。

## 3 注意力分解机

### 3.1 模型

图 1 展示了我们提出的 AFM 模型的神经网络架构。为清晰起见，我们在图中省略了线性回归部分，它可以被简单地加入。输入层和嵌入层与 FM 相同，它对输入特征采用稀疏表示并将每个非零特征嵌入到一个密集向量中。接下来，我们详细介绍成对交互层和基于注意力的池化层，这是本文的主要贡献。

**成对交互层。** 受到 FM [19] 使用内积建模每对特征之间交互的启发，我们在神经网络建模中提出了一种新的成对交互层。它将 $m$ 个向量扩展为 $m(m-1)/2$ 个交互向量，其中每个交互向量是两个不同向量的逐元素乘积以编码它们的交互。形式上，令特征向量 $\mathbf{x}$ 中非零特征的集合为 $\mathcal{X}$，嵌入层的输出为 $E = \{\mathbf{v}_i x_i\}_{i \in \mathcal{X}}$。我们可以将成对交互层的输出表示为一个向量集合：

$$
f_{\text{PI}}(E) = \{(\mathbf{v}_i \odot \mathbf{v}_j) x_i x_j\}_{(i,j) \in \mathcal{R}_x} \qquad (2)
$$

其中 $\odot$ 表示两个向量的逐元素乘积，$\mathcal{R}_x = \{(i, j)\}_{i \in \mathcal{X}, j \in \mathcal{X}, j > i}$ 为简写。通过定义成对交互层，我们可以在神经网络架构下表达 FM。为了展示这一点，我们首先对 $f_{\text{PI}}(E)$ 进行求和池化，然后使用全连接层将其投影到预测分数：

$$
\hat{y} = \mathbf{p}^{\top} \sum_{(i,j) \in \mathcal{R}_x} (\mathbf{v}_i \odot \mathbf{v}_j) x_i x_j + b \qquad (3)
$$

其中 $\mathbf{p} \in \mathbb{R}^k$ 和 $b$ 分别表示预测层的权重和偏置。显然，通过将 $\mathbf{p}$ 固定为 1 且 $b$ 固定为 0，我们可以精确地恢复 FM 模型。请注意，我们最近的神经 FM 工作 [14] 提出了一种双线性交互池化操作，它可以被视为在成对交互层上使用求和池化。

**基于注意力的池化层。** 自从注意力机制被引入神经网络建模以来，它已在许多任务中被广泛使用，如推荐 [5]、信息检索 [29] 和计算机视觉 [6]。其思想是在将不同部分压缩为单一表示时允许它们有不同贡献。受到 FM 缺陷的启发，我们提出通过对交互向量进行加权求和来对特征交互应用注意力机制：

$$
f_{\text{Att}}(f_{\text{PI}}(E)) = \sum_{(i,j) \in \mathcal{R}_x} a_{ij} (\mathbf{v}_i \odot \mathbf{v}_j) x_i x_j \qquad (4)
$$

其中 $a_{ij}$ 是特征交互 $\hat{w}_{ij}$ 的注意力分数，可以解释为 $\hat{w}_{ij}$ 在预测目标中的重要性。为了估计 $a_{ij}$，一个直观的解决方案是通过最小化预测损失来直接学习它，这在技术上似乎也是可行的。然而，问题在于，对于在训练数据中从未共现的特征，它们交互的注意力分数无法被估计。为了解决泛化问题，我们进一步用多层感知机（MLP）参数化注意力分数，我们称之为注意力网络。注意力网络的输入是两个特征的交互向量，它在嵌入空间中编码了它们的交互信息。形式上，注意力网络定义为：

$$
a'_{ij} = \mathbf{h}^{\top} \text{ReLU}(\mathbf{W}(\mathbf{v}_i \odot \mathbf{v}_j) x_i x_j + \mathbf{b}) \qquad (5)
$$

$$
a_{ij} = \frac{\exp(a'_{ij})}{\sum_{(i,j) \in \mathcal{R}_x} \exp(a'_{ij})}
$$

其中 $\mathbf{W} \in \mathbb{R}^{t \times k}$，$\mathbf{b} \in \mathbb{R}^t$，$\mathbf{h} \in \mathbb{R}^t$ 是模型参数，$t$ 表示注意力网络的隐藏层大小，我们称之为注意力因子。注意力分数通过 softmax 函数进行归一化，这是先前工作的常见做法。我们使用整流器作为激活函数，经验上表现出良好性能。

基于注意力的池化层的输出是一个 $k$ 维向量，它通过区分特征交互的重要性来压缩嵌入空间中的所有特征交互。然后我们将其投影到预测分数。总结来说，我们给出 AFM 模型的完整公式：

$$
\hat{y}_{\text{AFM}}(\mathbf{x}) = w_0 + \sum_{i=1}^{n} w_i x_i + \mathbf{p}^{\top} \sum_{i=1}^{n}\sum_{j=i+1}^{n} a_{ij} (\mathbf{v}_i \odot \mathbf{v}_j) x_i x_j \qquad (6)
$$

其中 $a_{ij}$ 已在公式 (5) 中定义。模型参数为 $\Theta = \{w_0, \{w_i\}_{i=1}^{n}, \{\mathbf{v}_i\}_{i=1}^{n}, \mathbf{p}, \mathbf{W}, \mathbf{b}, \mathbf{h}\}$。

### 3.2 学习

由于 AFM 从数据建模的角度直接增强了 FM，它也可以应用于各种预测任务，包括回归、分类和排序。应使用不同的目标函数来为不同任务定制 AFM 模型学习。对于目标 $y(\mathbf{x})$ 为实值的回归任务，常见的目标函数是平方损失：

$$
\mathcal{L}_r = \sum_{\mathbf{x} \in \mathcal{T}} (\hat{y}_{\text{AFM}}(\mathbf{x}) - y(\mathbf{x}))^2 \qquad (7)
$$

其中 $\mathcal{T}$ 表示训练实例的集合。对于具有隐式反馈的二分类或推荐任务 [13]，我们可以最小化对数损失。在本文中，我们专注于回归任务并优化平方损失。

**过拟合预防。** 过拟合是优化 ML 模型的永恒问题。研究表明 FM 可能受到过拟合的影响 [20]，因此 $L_2$ 正则化是防止 FM 过拟合的重要组成部分。由于 AFM 比 FM 具有更强的表示能力，它可能更容易过拟合训练数据。这里我们考虑两种广泛用于神经网络模型的过拟合预防技术——Dropout 和 $L_2$ 正则化。

Dropout [24] 的思想是在训练期间随机丢弃一些神经元（及其连接）。它已被证明能够防止神经元在训练数据上的复杂协同适应。由于 AFM 建模了特征之间的所有成对交互，而并非所有交互都有用，成对交互层的神经元可能容易相互协同适应并导致过拟合。因此，我们在成对交互层上使用 Dropout 以避免协同适应。此外，由于 Dropout 在测试期间被禁用且整个网络用于预测，Dropout 还有另一个副作用，即与较小的神经网络进行模型集成，这可能潜在地提高性能。

对于作为单层 MLP 的注意力网络组件，我们对权重矩阵 $\mathbf{W}$ 应用 $L_2$ 正则化以防止可能的过拟合。也就是说，我们实际优化的目标函数是：

$$
\mathcal{L} = \sum_{\mathbf{x} \in \mathcal{T}} (\hat{y}_{\text{AFM}}(\mathbf{x}) - y(\mathbf{x}))^2 + \lambda \|\mathbf{W}\|^2 \qquad (8)
$$

其中 $\lambda$ 控制正则化强度。我们不在注意力网络上使用 Dropout，因为我们发现在交互层和注意力网络上同时使用 Dropout 会导致一些稳定性问题并降低性能。

为优化目标函数，我们采用随机梯度下降（SGD）——一种神经网络模型的通用求解器。实现 SGD 算法的关键是获得预测模型 $\hat{y}_{\text{AFM}}(\mathbf{x})$ 关于每个参数的导数。由于大多数现代深度学习工具包（如 Theano 和 TensorFlow）都提供了自动微分功能，我们在此省略导数的细节。

## 4 相关工作

FM [19] 主要用于稀疏设置下的监督学习；例如，在类别变量通过独热编码转换为稀疏特征向量的情况下。与图像和音频中发现的连续原始特征不同，Web 领域的输入特征大多是离散和类别的。对于此类稀疏数据的预测，对特征之间的交互进行建模至关重要。与仅建模两个实体之间交互的矩阵分解（MF）[11] 不同，FM 被设计为用于建模任意数量实体之间交互的通用机器学习器。通过指定输入特征向量 [21]，FM 可以涵盖许多特定的分解模型，如 MF、并行因子分析和 SVD++ [17] [17]。因此，FM 被认为是稀疏数据预测的最有效线性嵌入方法。已提出了许多 FM 变体，如在神经框架下深化 FM [14] 以学习高阶特征交互的神经 FM [14]，以及为特征关联多个嵌入向量以区分其与不同领域特征交互的场感知 FM [16] [16]。

在本文中，我们通过区分特征交互的重要性来贡献 FM 的改进。我们知道一个与我们提案相似的工作——GBFM [7]，它使用梯度提升选择"好"特征并仅建模好特征之间的交互。对于选定特征之间的交互，GBFM 像 FM 一样以相同的权重求和它们。因此，GBFM 本质上是一种特征选择算法，这与我们的 AFM 根本不同，后者可以学习每个特征交互的重要性。

沿着另一条线，深度神经网络（即深度学习）正变得越来越流行，最近已被用于稀疏设置下的预测。具体来说，Wide&Deep [8] 为应用推荐提出了 wide 和 deep 架构，其中 deep 部分是特征嵌入向量拼接上的 MLP 以学习特征交互；DeepCross [22] 为点击率预测提出了深度残差 MLP [10] 以学习交叉特征。我们指出，在这些方法中，特征交互是由深度神经网络隐式捕获的，而不是像 FM 那样将每个交互显式建模为两个特征的内积。因此，这些深度方法是不可解释的，因为每个特征交互的贡献是未知的。通过直接用注意力机制扩展 FM �学习每个特征交互的重要性，我们的 AFM [14] 更具可解释性，并在经验上展示了优于 Wide&Deep 和 DeepCross 的性能。

## 5 实验

我们进行实验以回答以下问题：

- **RQ1**：AFM 的关键超参数（即特征交互上的 Dropout 和注意力网络上的正则化）如何影响其性能？
- **RQ2**：注意力网络能否有效地学习特征交互的重要性？
- **RQ3**：与最先进的稀疏数据预测方法相比，AFM 的表现如何？

### 5.1 实验设置

**数据集。** 我们使用两个公开数据集进行实验：Frappe 和 MovieLens。Frappe [1] 数据集已用于上下文感知推荐，包含 96,203 条用户在不同上下文下的应用使用日志。八个上下文变量都是类别的，包括天气、城市、白天等。我们通过独热编码将每条日志（用户 ID、应用 ID 和上下文变量）转换为特征向量，获得 5,382 个特征。MovieLens [9] 数据已用于个性化标签推荐，包含 668,953 条用户对电影的标签应用。我们将每个标签应用（用户 ID、电影 ID 和标签）转换为特征向量，获得 90,445 个特征。

**评估协议。** 对于两个数据集，每条日志被赋予值为 1 的目标，表示用户已在该上下文中使用了应用或将标签应用于电影。我们随机为每条日志配对两个负样本并将其目标设为 -1。因此，Frappe 和 MovieLens 的最终实验数据分别包含 288,609 和 2,006,859 个实例。我们将每个数据集随机分为三部分：70% 用于训练，20% 用于验证，10% 用于测试。验证集仅用于调整超参数，性能比较在测试集上完成。为评估性能，我们采用均方根误差（RMSE），其中较低的分数表示更好的性能。

**基线方法。** 我们将 AFM 与以下为稀疏数据预测设计的竞争方法进行比较：

- **LibFM**：这是 FM [21] 的官方 C++ 实现。我们选择 SGD 求解器，因为其他方法都通过 SGD（或其变体）优化。
- **HOFM**：这是高阶 FM [3] 的 TensorFlow 实现。我们将阶数设为 3，因为 MovieLens 数据只有三种类型的预测变量（用户、item 和标签）。
- **Wide&Deep**：我们实现了该方法 [8]。由于深度神经网络的结构（例如每层的深度和大小）难以完全调整，我们使用与原文 [8] 报告相同的结构。wide 部分与 FM 的线性回归部分相同，deep 部分是层大小为 1024、512 和 256 的三层 MLP。
- **DeepCross**：我们使用与原文 [22] 相同的结构实现了该方法。它堆叠了 5 个残差单元（每个单元有两层），隐藏维度分别为 512、512、256、128 和 64。

所有模型通过优化平方损失进行公平比较。除 LibFM 外，所有方法通过小批量 Adagrad 学习。Frappe 和 MovieLens 的批大小分别设为 128 和 4096。所有方法的嵌入大小设为 256。除非另有说明，注意力因子也设为 256，与嵌入大小相同。我们仔细调整了 Frappe 和 HOFM 的 $L_2$ 正则化，以及 Wide&Deep 和 DeepCross 的 Dropout 比率。采用早停策略，基于验证集上的性能。对于 Wide&Deep [8]、DeepCross [22] 和 AFM，我们发现用 FM [19] 预训练其特征嵌入比随机初始化导致更低的 RMSE。因此，我们报告它们使用预训练的性能。

### 5.2 超参数调查（RQ1）

首先，我们探讨成对交互层上 Dropout 的影响。我们将 $\lambda$ 设为 0，使注意力网络上不使用 $L_2$ 正则化。我们还在去掉注意力组件的 AFM 实现上验证了 Dropout，即我们的 FM 实现。图 2 展示了 AFM 和 FM 在不同 Dropout 比率下的验证误差；LibFM 的结果也作为基准展示。我们有以下观察：

- 通过将 Dropout 比率设为适当值，AFM 和 FM 都可以显著改善。具体来说，对于 AFM，Frappe 和 MovieLens 上的最优 Dropout 比率分别为 0.2 和 0.5。这验证了在成对交互层上使用适当 Dropout 的有用性，它改善了 FM 和 AFM 的泛化能力。
- 我们的 FM 实现提供了比 LibFM 更好的性能。原因有两方面。首先，LibFM 使用原始 SGD 优化，对所有参数采用固定学习率；而我们使用 Adragrad 优化 FM，它根据每个参数的频率调整学习率（即对频繁参数进行较小更新，对不频繁参数进行较大更新）。其次，LibFM 通过 $L_2$ 正则化防止过拟合，而我们使用 Dropout，由于模型集成效果可能更有效。
- AFM 大幅优于 FM 和 LibFM。即使不使用 Dropout 且过拟合问题在一定程度上存在，AFM 也取得了显著优于 LibFM 和 FM 最优性能的结果。这证明了注意力网络在学习特征交互权重方面的好处。

然后我们研究注意力网络上的 $L_2$ 正则化是否对 AFM 有益。Dropout 比率设为前一实验中每个数据集的最优值。从图 3 可以看出，当 $\lambda$ 设为大于 0 的值时，AFM 得到改善。这意味着简单地在成对交互层上使用 Dropout 不足以防止过拟合，更重要的是，调整注意力网络的正则化可以进一步改善 AFM 的泛化能力。

### 5.3 注意力网络的影响（RQ2）

我们现在专注于分析注意力网络对 AFM 的影响。第一个要回答的问题是如何选择注意力因子？图 4 展示了 AFM 在不同注意力因子下的验证误差。请注意，$\lambda$ 已为每个注意力因子单独调整。我们可以观察到，对于两个数据集，AFM 的性能在注意力因子上相当稳定。具体来说，当注意力因子为 1 时，$\mathbf{W}$ 矩阵变为一个向量，注意力网络本质上退化为以交互向量（即 $\mathbf{v}_i \odot \mathbf{v}_j$）作为输入特征的线性回归模型。尽管注意力组件的模型能力如此受限，AFM 仍然非常强大，并显著改善了 FM。这证明了 AFM 设计的合理性，即基于交互向量估计特征交互的重要性分数，这是本文的关键发现。

图 5 比较了 AFM 和 FM 每个 epoch 的训练和测试误差。我们观察到 AFM 比 FM 收敛更快。在 Frappe 上，AFM 的训练和测试误差都远低于 FM，表明 AFM 可以更好地拟合数据并带来更准确的预测。在 MovieLens 上，虽然 AFM 的训练误差略高于 FM，但更低的测试误差表明 AFM 对未见数据的泛化能力更强。

**微观分析。** 除了改善的性能外，AFM 的另一个关键优势是通过解释每个特征交互的注意力分数来更具可解释性。为了展示这一点，我们进行了一些微观分析，通过调查 MovieLens 上每个特征交互的分数。为了对注意力网络进行专门分析，我们首先将 $a_{ij}$ 固定为统一数 $1/|\mathcal{R}_x|$，训练模型，这模拟了 FM。然后我们固定特征嵌入，仅训练注意力网络；收敛后，性能提升了约 3%，这证明了注意力网络的有效性。然后我们选择了三个目标值为 1 的测试实例，在表 1 中展示每个特征交互的注意力分数和交互分数。我们可以看到，在所有三个实例中，item-tag 交互是最重要的。然而，FM 对所有交互分配相同的重要性分数，导致较大的预测误差。通过用注意力网络增强 FM（参见 FM+A 行），item-tag 交互被赋予更高的重要性分数，预测误差降低。

### 5.4 性能比较（RQ3）

在这最后一个子节中，我们比较不同方法在测试集上的性能。表 2 总结了在嵌入大小 256 上获得的最佳性能和每种方法的可训练参数数量。

- 首先，我们看到 AFM 在所有方法中取得了最佳性能。具体来说，AFM 通过使用不到 0.1M 的额外参数，比 LibFM [21] 提升了 8.6% 的相对改善；AFM 以少得多的模型参数优于第二好的方法 Wide&Deep [8] 4.3%。这证明了 AFM 的有效性，尽管它是一种浅层模型，却取得了比深度学习方法更好的性能。
- 其次，HOFM [3] 改善了 FM [19]，这归因于其对高阶特征交互的建模。然而，这些微小的改善基于几乎双倍参数数量的昂贵成本，因为 HOFM 使用分离的嵌入集来建模每个阶数的特征交互。这指向了未来研究的一个有希望的方向——设计更有效的方法来捕获高阶特征交互。
- 最后，DeepCross [22] 表现最差，原因是严重的过拟合问题。我们发现 Dropout 对 DeepCross 效果不佳，这可能是由其使用批归一化引起的。考虑到 DeepCross 是最深的方法（在嵌入层之上堆叠了 10 层），在所有比较方法中，它提供了证据表明更深的学习并不总是有帮助，因为深度网络可能受到过拟合的影响，并且在实践中更难优化。

## 6 结论与未来工作

我们提出了一种简单而有效的监督学习模型 AFM。我们的 AFM 通过用注意力网络学习特征交互的重要性来增强 FM，这不仅提高了表示能力，还提高了模型的可解释性。这项工作与我们最近关于神经 FM [14] 的工作正交，后者开发了 FM 的深度变体来建模高阶特征交互，现在是将注意力机制引入分解机的时候了。

在未来，我们将通过在基于注意力的池化层之上堆叠多个非-linear 层来探索 AFM 的深度版本，看看它是否能进一步改善性能。由于 AFM 相对于非零特征数量具有较高的二次复杂度，我们将考虑提高其学习效率，例如使用 learning to hash [33, 23] 和数据采样 [28] 技术。另一个有希望的方向是为半监督和多视图学习开发 FM 变体，例如通过结合广泛使用的图拉普拉斯 [12, 26] 和协同正则化设计 [15, 31]。最后，我们将探索 AFM 在不同应用中建模其他类型数据，如用于问答 [35] 的文本和更丰富的语义多媒体内容 [32, 30]。

## 致谢

本工作得到中国国家自然科学基金（批准号 U1611461 和 61572431）、浙江省重点研发计划（批准号 2015C01027）、浙江省自然科学基金（批准号 LZ17F020001）的支持。NExT 研究得到新加坡国家研究基金会总理办公室 IRC@SG 资助计划的支持。

## 参考文献

[1] Linas Baltrunas, Karen Church, Alexandros Karatzoglou, and Nuria Oliver. Frappe: Understanding the usage and perception of mobile app recommendations in-the-wild. CoRR, abs/1505.03014, 2015.

[2] Immanuel Bayer, Xiangnan He, Bhargav Kanagal, and Steffen Rendle. A generic coordinate descent framework for learning from implicit feedback. In WWW, 2017.

[3] Mathieu Blondel, Akinori Fujino, Naonori Ueda, and Masakazu Ishihata. Higher-order factorization machines. In NIPS, 2016.

[4] Tao Chen, Xiangnan He, and Min-Yen Kan. Context-aware image tweet modelling and recommendation. In MM, 2016.

[5] Jingyuan Chen, Hanwang Zhang, Xiangnan He, Liqiang Nie, Wei Liu, and Tat-Seng Chua. Attentive collaborative filtering: Multimedia recommendation with feature- and item-level attention. In SIGIR, 2017.

[6] Long Chen, Hanwang Zhang, Jun Xiao, Liqiang Nie, Jian Shao, and Tat-Seng Chua. SCA-CNN: Spatial and channel-wise attention in convolutional networks for image captioning. In CVPR, 2017.

[7] Chen Cheng, Fen Xia, Tong Zhang, Irwin King, and Michael R Lyu. Gradient boosting factorization machines. In RecSys, 2014.

[8] Heng-Tze Cheng, Levent Koc, Jeremiah Harmsen, et al. Wide & deep learning for recommender systems. In DLRS, 2016.

[9] F. Maxwell Harper and Joseph A. Konstan. The movielens datasets: History and context. ACM TIIS, 2015.

[10] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In CVPR, 2016.

[11] Xiangnan He, Hanwang Zhang, Min-Yen Kan, and Tat-Seng Chua. Fast matrix factorization for online recommendation with implicit feedback. In SIGIR, 2016.

[12] Xiangnan He, Ming Gao, Min-Yen Kan, and Dingxian Wang. BiRank: Towards ranking on bipartite graphs. IEEE TKDE, 2017.

[13] Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu, and Tat-Seng Chua. Neural collaborative filtering. In WWW, 2017.

[14] Xiangnan He and Tat-Seng Chua. Neural factorization machines for sparse predictive analytics. In SIGIR, 2017.

[15] Xiangnan He, Min-Yen Kan, Peichu Xie, and Xiao Chen. Comment-based multi-view clustering of web 2.0 items. In WWW, 2014.

[16] Yuchin Juan, Yong Zhuang, Wei-Sheng Chin, and Chih-Jen Lin. Field-aware factorization machines for CTR prediction. In RecSys, 2016.

[17] Yehuda Koren. Factorization meets the neighborhood: A multifaceted collaborative filtering model. In KDD, 2008.

[18] Fabio Petroni, Luciano Del Corro, and Rainer Gemulla. Core: Context-aware open relation extraction with factorization machines. In EMNLP, 2015.

[19] Steffen Rendle. Factorization machines. In ICDM, 2010.

[20] Steffen Rendle, Zeno Gantner, Christoph Freudenthaler, and Lars Schmidt-Thieme. Fast context-aware recommendations with factorization machines. In SIGIR, 2011.

[21] Steffen Rendle. Factorization machines with libfm. ACM TIST, 2012.

[22] Ying Shan, T Ryan Hoens, Jian Jiao, Haijing Wang, Dong Yu, and JC Mao. Deep crossing: Web-scale modeling without manually crafted combinatorial features. In KDD, 2016.

[23] Fumin Shen, Chunhua Shen, Wei Liu, and Heng Tao Shen. Supervised discrete hashing. In CVPR, 2015.

[24] Nitish Srivastava, Geoffrey E Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov. Dropout: A simple way to prevent neural networks from overfitting. JMLR, 2014.

[25] Meng Wang, Xueliang Liu, and Xindong Wu. Visual classification by l1-hypergraph modeling. IEEE TKDE, 2015.

[26] Meng Wang, Weijie Fu, Shijie Hao, Dacheng Tao, and Xindong Wu. Scalable semi-supervised learning by efficient anchor graph regularization. IEEE TKDE, 2016.

[27] Xiang Wang, Xiangnan He, Liqiang Nie, and Tat-Seng Chua. Item silk road: Recommending items from information domains to social users. In SIGIR, 2017.

[28] Meng Wang, Weijie Fu, Shijie Hao, Hengchang Liu, and Xindong Wu. Learning on big graph: Label inference and regularization with anchor hierarchy. IEEE TKDE, 2017.

[29] Chenyan Xiong, Jimie Callan, and Tie-Yen Liu. Learning to attend and to rank with word-entity duets. In SIGIR, 2017.

[30] Yang Yang, Zheng-Jun Zha, Yue Gao, Xiaofeng Zhu, and Tat-Seng Chua. Exploiting web images for semantic video indexing via robust sample-specific loss. IEEE TMM, 2014.

[31] Yang Yang, Zhigang Ma, Yi Yang, Feiping Nie, and Heng Tao Shen. Multitask spectral clustering by exploring intertask correlation. IEEE TCYB, 2015.

[32] Hanwang Zhang, Xindi Shang, Huanbo Luan, Meng Wang, and Tat-Seng Chua. Learning from collective intelligence: Feature learning using social images and tags. TMM, 2016.

[33] Hanwang Zhang, Fumin Shen, Wei Liu, Xiangnan He, Huanbo Luan, and Tat-Seng Chua. Discrete collaborative filtering. In SIGIR, 2016.

[34] Hanwang Zhang, Zawlin Kyaw, Shih-Fu Chang, and Tat-Seng Chua. Visual translation embedding network for visual relation detection. In CVPR, 2017.

[35] Zhou Zhao, Lijun Zhang, Xiaofei He, and Wilfred Ng. Expert finding for question answering via graph regularized matrix completion. TKDE, 2015.

[36] Zhou Zhao, Hanqing Lu, Deng Cai, Xiaofei He, and Yueting Zhuang. User preference learning for online social recommendation. TKDE, 2016.
