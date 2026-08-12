# DCN V2: 改进的深度交叉网络及大规模学习排序系统的实践经验

> Ruoxi Wang, Rakesh Shivanna, Derek Z. Cheng, Sagar Jain, Dong Lin, Lichan Hong, Ed H. Chi | Google

本文提出DCN-V2模型，**通过改进交叉网络的表达能力并结合低秩混合专家架构，在保持计算效率的同时显著提升特征交互学习效果**。

核心内容：

- 原始DCN的交叉网络表达能力有限，多项式类仅由$O(\text{input size})$个参数刻画，限制了建模复杂特征交叉的灵活性
- 提出DCN-V2：改进交叉层公式，支持任意嵌入维度，显著增强显式特征交叉的表达能力
- 发现权重矩阵的低秩特性，提出低秩混合专家版本DCN-Mix，在性能和延迟间取得更优权衡
- 在Google多个大规模学习排序系统中成功部署，离线AUC和在线业务指标均显著提升

关键发现：

- DCN-V2在Criteo数据集上LogLoss达到**0.4406**，AUC达到**0.8115**，优于所有SOTA基线
- DCN-Mix在保持精度的同时**降低30%计算成本**，实现更优的质量/成本权衡
- 合成实验表明：交叉网络仅需5层即可准确捕获复杂多项式模式，而DNN即使更深更宽也难以拟合
- 生产部署中DCN-V2相比同规模ReLU层AUCLoss改进**0.6%**（0.1%即为显著改进）

---

## 摘要

学习有效的特征交叉是构建推荐系统的关键。然而，稀疏且大规模的特征空间需要穷举搜索来识别有效的交叉。深度交叉网络（DCN）被提出以自动且高效地学习有界度预测性特征交互。不幸的是，在服务数十亿训练样本的Web流量模型中，DCN在交叉网络中学习更具预测性的特征交互方面显示出有限的表达能力。尽管研究取得了显著进展，但许多生产中的深度学习模型仍然依赖传统的前馈神经网络来低效地学习特征交叉。

鉴于DCN和现有特征交互学习方法的优缺点，我们提出了一个改进框架DCN-V2，以使DCN在大规模工业环境中更加实用。在一项包含广泛超参数搜索和模型调优的综合实验研究中，我们观察到DCN-V2方法在流行的基准数据集上优于所有最先进的算法。改进的DCN-V2在特征交互学习方面更具表达能力，同时保持成本效率，特别是与低秩混合架构结合时。DCN-V2简单，可以轻松作为构建模块采用，并已在Google的许多大规模学习排序系统中带来了显著的离线精度和在线业务指标提升。

## 1 引言

学习排序（LTR）[4, 27]一直是现代机器学习和深度学习中最重要的问题之一。它在搜索、推荐系统[17, 39, 41]和计算广告[2, 3]中有广泛的应用。在LTR模型的关键组件中，学习有效的特征交叉继续吸引着学术界[26, 35, 46]和工业界[1, 6, 13, 34, 50]的大量关注。

有效的特征交叉对许多模型的成功至关重要。它们提供了超越单个特征的额外交互信息。例如，"国家"和"语言"的组合比其中任何一个都更具信息量。在线性模型时代，ML从业者依赖手动识别此类特征交叉[43]来增加模型的表达能力。不幸的是，这涉及一个组合搜索空间，在数据主要是类别的Web规模应用中，这个空间既大又稀疏。在这种设置下搜索是穷举的，通常需要领域专业知识，并且使模型更难泛化。

后来，嵌入技术被广泛采用，将特征从高维稀疏向量投影到低得多的密集向量。因子分解机（FM）[36, 37]利用嵌入技术，通过两个潜在向量的内积构建成对的特征交互。与线性模型中的传统特征交叉相比，FM带来了更强的泛化能力。

在过去的十年中，随着更多计算能力和大规模数据，工业中的LTR模型逐渐从线性模型和基于FM的模型迁移到深度神经网络（DNN）。这显著改善了搜索和推荐系统的模型性能[6, 13, 50]。人们通常将DNN视为通用函数逼近器，可能学习各种特征交互[31, 47, 49]。然而，最近的研究[1, 50]发现，DNN甚至在近似建模2阶或3阶特征交叉方面效率低下。

为了更准确地捕获有效的特征交叉，一个常见的补救措施是通过更宽或更深的网络进一步增加模型容量。这自然形成了一把双刃剑：我们在提高模型性能的同时使模型服务速度变慢。在许多生产环境中，这些模型处理极高的QPS，因此对实时推理有非常严格的延迟要求。可能服务系统已经被推到极限，无法承受更大的模型。此外，更深的模型通常引入可训练性问题，使模型更难训练。

这揭示了设计一个能够高效且有效地学习预测性特征交互模型的关键需求，特别是在处理来自数十亿用户实时流量的资源受限环境中。许多最近的工作[1, 6, 13, 26, 34, 35, 46, 50]试图解决这一挑战。共同主题是利用从DNN中学习的隐式高阶交叉，结合在线性模型中被发现有效的显式且有界度特征交叉。隐式交叉意味着交互是通过端到端函数学习的，没有任何显式公式建模此类交叉。显式交叉则通过具有可控交互阶数的显式公式建模。我们在第2节中详细讨论这些模型。

其中，深度交叉网络（DCN）[50]有效且优雅，然而在大规模工业系统中生产DCN面临许多挑战。其交叉网络的表达能力有限。交叉网络再现的多项式类仅由$O(\text{input size})$个参数刻画，在很大程度上限制了其建模随机交叉模式的灵活性。此外，交叉网络和DNN之间的容量分配不平衡。当将DCN应用于大规模生产数据时，这一差距显著增加。大部分参数将用于学习DNN中的隐式交叉。

在本文中，我们提出了一个新模型DCN-V2，改进了原始DCN模型。我们已经成功地在Google的多个学习排序系统中部署了DCN-V2，在离线模型精度和在线业务指标方面都取得了显著收益。本文的主要贡献有五个方面：

- 我们提出了一个新颖的模型——DCN-V2，用于学习有效的显式和隐式特征交叉。与现有方法相比，我们的模型更具表达能力，同时保持高效和简单。
- 观察到DCN-V2中学习矩阵的低秩特性，我们提出利用低秩技术在子空间中近似特征交叉，以获得更好的性能和延迟权衡。此外，我们提出了一种基于混合专家架构[19, 45]的技术，将矩阵进一步分解为多个更小的子空间，然后通过门控机制聚合这些子空间。
- 我们使用合成数据集进行了广泛研究，展示了传统ReLU神经网络在学习高阶特征交叉方面的低效性。
- 通过全面的实验分析，我们展示了提出的DCN-V2模型在Criteo和MovieLen-1M基准数据集上显著优于SOTA算法。
- 我们提供了一个案例研究，分享了在大规模工业排序系统中生产DCN-V2的经验，该系统带来了显著的离线和在线收益。

## 2 相关工作

最近特征交互学习工作的核心思想是利用显式和隐式（来自DNN）特征交叉。为了建模显式交叉，大多数最近的工作引入了乘法运算（$x_1 \times x_2$），这在DNN中效率低下，并设计了一个函数$f(x_1, x_2)$来高效且显式地建模特征$x_1$和$x_2$之间的成对交互。我们根据它们如何组合显式和隐式组件来组织工作。

**并行结构。** 一系列工作联合训练两个并行网络，灵感来自Wide & Deep模型[6]，其中wide组件以原始特征的交叉作为输入；deep组件是一个DNN模型。然而，为wide组件选择交叉特征又回到了线性模型的特征工程问题。尽管如此，Wide & Deep模型启发了许多工作采用这种并行架构并改进wide组件。

DeepFM[13]通过采用FM模型自动化wide组件中的特征交互学习。DCN[50]引入了一个交叉网络，自动且高效地学习显式且有界度的特征交互。xDeepFM[26]通过生成多个特征图增加了DCN的表达能力，每个特征图编码当前层和输入层之间所有成对的交互。此外，它还将每个特征嵌入$x_i$视为一个单元，而不是将每个元素$x_i$视为一个单元。不幸的是，其计算成本显著偏高（参数数量的10倍），使其不适用于工业规模的应用。此外，DeepFM和xDeepFM都要求所有特征嵌入大小相等，这在应用于词汇表大小（类别特征的大小）从$O(10)$到数百万不等的工业数据时又是一个限制。AutoInt[46]利用多头自注意力机制和残差连接。InterHAt[25]进一步采用层次化注意力。

**堆叠结构。** 另一系列工作在嵌入层和DNN模型之间引入了一个交互层——创建显式特征交叉。这个交互层在早期阶段捕获特征交互，并促进后续隐藏层的学习。基于乘积的神经网络（PNN）[35]引入内积（IPNN）和外积（OPNN）层作为成对交互层。OPNN的一个缺点在于其高计算成本。神经FM（NFM）[16]通过用Hadamard乘积替换内积来扩展FM；DLRM[34]遵循FM通过内积计算特征交叉；这些模型最多只能创建2阶显式交叉。AFN[7]将特征转换到对数空间并自适应地学习任意阶特征交互。与DeepFM和xDeepFM类似，它们只接受相等大小的嵌入。

尽管取得了许多进展，我们的综合实验（第7节）表明DCN仍然是一个强大的基线。我们将其归因于其简单结构促进了优化。然而，如前所述，其有限的表达能力阻碍了它在Web规模系统中学习更有效的特征交叉。在下文中，我们提出了一种新架构，它继承了DCN的简单结构，同时增加了其表达能力。

## 3 提出的架构：DCN-V2

本节描述了一种新颖的模型架构——DCN-V2——用于学习显式和隐式特征交互。DCN-V2从嵌入层开始，然后是一个包含多个交叉层的交叉网络来建模显式特征交互，最后与一个建模隐式特征交互的深度网络结合。DCN-V2所做的改进对于将DCN应用于高度优化的生产系统至关重要。DCN-V2显著提高了DCN[50]在Web规模生产数据中建模复杂显式交叉项的表达能力，同时保持其优雅的公式以便于部署。DCN-V2建模的函数类是DCN建模的严格超集。整体模型架构如图1所示，有两种方式将交叉网络与深度网络结合：（1）堆叠和（2）并行。此外，观察到交叉层的低秩特性，我们提出利用低秩交叉层的混合来实现模型性能和效率之间更优的权衡。

### 3.1 嵌入层

嵌入层以类别（稀疏）和密集特征的组合作为输入，输出$x_0 \in \mathbb{R}^d$。对于第$i$个类别特征，我们通过$x_{\text{embed},i} = W_{\text{embed},i} e_i$将其从高维稀疏空间投影到低维密集空间，其中$e_i \in \{0, 1\}^{v_i}$；$W \in \mathbb{R}^{e_i \times v_i}$是学习的投影矩阵；$x_{\text{embed},i} \in \mathbb{R}^{e_i}$是密集嵌入向量；$v_i$和$e_i$分别表示词汇表和嵌入大小。对于多值特征，我们使用嵌入向量的均值作为最终向量。

输出是所有嵌入向量和归一化密集特征的拼接：$x_0 = [x_{\text{embed},1}; \ldots; x_{\text{embed},n}; x_{\text{dense}}]$。与许多要求$e_i = e_j \forall i, j$的相关工作[13, 16, 26, 34, 35, 46]不同，我们的模型接受任意嵌入大小。这对于工业推荐系统尤为重要，因为词汇表大小从$O(10)$到$O(10^5)$不等。此外，我们的模型不限于上述描述的嵌入方法；可以采用任何其他嵌入技术，如哈希。

### 3.2 交叉网络

DCN-V2的核心在于创建显式特征交叉的交叉层。公式(1)展示了第$(l+1)$个交叉层：

$$
x_{l+1} = x_0 \odot (W_l x_l + b_l) + x_l \qquad (1)
$$

其中$x_0 \in \mathbb{R}^d$是包含原始1阶特征的基础层，通常设置为嵌入（输入）层。$x_l, x_{l+1} \in \mathbb{R}^d$分别表示第$(l+1)$个交叉层的输入和输出。$W_l \in \mathbb{R}^{d \times d}$和$b_l \in \mathbb{R}^d$是学习的权重矩阵和偏置向量。

对于一个$l$层的交叉网络，最高多项式阶数为$l+1$，网络包含所有直到最高阶的特征交叉。当$W = \mathbf{1} \times w^\top$时（其中$\mathbf{1}$表示全1向量），DCN-V2退化为DCN。交叉层只能再现有界度的多项式函数类；任何其他复杂的函数空间只能被近似。因此，我们引入一个深度网络来补充对数据固有分布的建模。

### 3.3 深度网络

第$l$个深度层的公式为$h_{l+1} = f(W_l h_l + b_l)$，其中$h_l \in \mathbb{R}^{d_l}$，$h_{l+1} \in \mathbb{R}^{d_{l+1}}$分别是第$l$个深度层的输入和输出；$W_l \in \mathbb{R}^{d_l \times d_{l+1}}$是权重矩阵，$b_l \in \mathbb{R}^{d_{l+1}}$是偏置向量；$f(\cdot)$是逐元素激活函数，我们设置为ReLU；任何其他激活函数也适用。

### 3.4 深度与交叉组合

我们寻求将交叉网络和深度网络结合的结构。最近的文献采用了两种结构：堆叠和并行。在实践中，我们发现哪种架构更好取决于数据。因此，我们同时展示两种结构：

**堆叠结构（图1a）：** 输入$x_0$先输入交叉网络，然后输入深度网络，最终层为$x_{\text{final}} = h_{L_d}$，$h_0 = x_{L_c}$，将数据建模为$f_{\text{deep}} \circ f_{\text{cross}}$。

**并行结构（图1b）：** 输入$x_0$并行输入交叉网络和深度网络；然后将输出$x_{L_c}$和$h_{L_d}$拼接创建最终输出层$x_{\text{final}} = [x_{L_c}; h_{L_d}]$。这种结构将数据建模为$f_{\text{cross}} + f_{\text{deep}}$。

最终，预测$\hat{y}_i$计算为：$\hat{y}_i = \sigma(w_{\text{logit}}^\top x_{\text{final}})$，其中$w_{\text{logit}}$是logit的权重向量，$\sigma(x) = 1/(1 + \exp(-x))$。对于最终损失，我们使用Log Loss，这在具有二分类标签（如点击）的学习排序系统中常用。注意，DCN-V2本身对预测任务和损失函数都是不可知的。

$$
\text{loss} = -\frac{1}{N} \sum_{i=1}^{N} y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) + \lambda \sum_l \|W_l\|_2^2
$$

其中$\hat{y}_i$是预测；$y_i$是真实标签；$N$是输入总数；$\lambda$是$L_2$正则化参数。

### 3.5 成本有效的低秩混合DCN

在实际生产模型中，模型容量通常受限于有限的服务资源和严格的延迟要求。通常情况下，我们必须寻求在保持精度的同时降低成本的方法。低秩技术[12]被广泛使用[5, 9, 14, 20, 51, 52]来降低计算成本。它通过两个高瘦矩阵$U, V \in \mathbb{R}^{d \times r}$来近似密集矩阵$M \in \mathbb{R}^{d \times d}$。当$r \leq d/2$时，成本将降低。然而，当矩阵在奇异值上显示出大的差距或快速的频谱衰减时，它们最有效。在许多设置中，我们确实观察到学习的矩阵在实践中是数值低秩的。

因此，对$W$施加低秩结构是有充分动机的。公式(2)展示了结果第$(l+1)$个低秩交叉层：

$$
x_{l+1} = x_0 \odot (U_l V_l^\top x_l + b_l) + x_l \qquad (2)
$$

其中$U_l, V_l \in \mathbb{R}^{d \times r}$且$r \ll d$。公式(2)有两种解释：1）我们在子空间中学习特征交叉；2）我们将输入$x$投影到低维$\mathbb{R}^r$，然后投影回$\mathbb{R}^d$。这两种解释启发了以下两种模型改进。

解释1启发我们采用混合专家（MoE）[10, 19, 30, 45]的思想。基于MoE的模型由两个组件组成：专家（通常是一个小网络）和门控（输入的函数）。在我们的情况下，我们不依赖单个专家（公式2）来学习特征交叉，而是利用多个这样的专家，每个在不同的子空间中学习特征交互，并使用依赖于输入$x$的门控机制自适应地组合学习的交叉。结果的低秩混合交叉层公式如公式(3)所示。

$$
x_{l+1} = \sum_{i=1}^{K} G_i(x_l) E_i(x_l) + x_l
$$
$$
E_i(x_l) = x_0 \odot (U_i_l V_i_l^\top x_l + b_l) \qquad (3)
$$

其中$K$是专家数量；$G_i(\cdot) : \mathbb{R}^d \rightarrow \mathbb{R}$是门控函数，常用的sigmoid或softmax；$E_i(\cdot) : \mathbb{R}^d \rightarrow \mathbb{R}^d$是第$i$个学习特征交叉的专家。$G(\cdot)$为输入$x$动态加权每个专家，当$G(\cdot) \equiv 1$时，公式(3)退化为公式(2)。

解释2启发我们利用投影空间的低维特性。我们不立即从维度$d'$投影回$d$（$d' \ll d$），而是在投影空间中进一步应用非线性变换来细化表示[11]。

$$
E_i(x_l) = x_0 \odot (U_i_l \cdot g(C_i_l \cdot g(V_i_l^\top x_l)) + b_l) \qquad (4)
$$

其中$g(\cdot)$表示任何非线性激活函数。

### 3.6 复杂度分析

设$d$表示嵌入大小，$L_c$表示交叉层数，$K$表示低秩DCN专家数量。进一步，为简单起见，我们假设每个专家具有相同的较小维度$r$（秩的上界）。交叉网络的时间和空间复杂度为$O(d^2 L_c)$，对于低秩混合DCN（DCN-Mix），当$rK \ll d$时效率为$O(2drKL_c)$。

## 4 模型分析

本节从多项式近似的角度分析DCN-V2，并与相关工作建立联系。

### 4.1 多项式近似

我们从两个多项式近似的角度分析DCN-V2——1）将每个元素（位）$x_i$视为一个单元，分析元素之间的交互（定理4.1）；2）将每个特征嵌入$x_i$视为一个单元，仅分析特征级别的交互（定理4.2）（证明见附录）。

**定理4.1（位级别）。** 假设$l$层交叉网络的输入为$x \in \mathbb{R}^d$，输出为$f_l(x) = \mathbf{1}^\top x_l$，第$i$层定义为$x_i = x \odot W^{(i-1)} x_{i-1} + x_{i-1}$。则多变量多项式$f_l(x)$再现以下类中的多项式：

$$
\left\{ \sum_{\boldsymbol{\alpha}} c_{\boldsymbol{\alpha}}(W^{(1)}, \ldots, W^{(l)}) x_1^{\alpha_1} x_2^{\alpha_2} \ldots x_d^{\alpha_d} \ \middle|\ 0 \leq |\boldsymbol{\alpha}| \leq l+1, \boldsymbol{\alpha} \in \mathbb{N}^d \right\}
$$

**定理4.2（特征级别）。** 在与定理4.1相同的设置下，我们进一步假设输入$x = [x_1; \ldots; x_k]$包含$k$个特征嵌入，并将每个$x_i$视为一个单元。则$l$层交叉网络的输出$x_l$创建所有直到$l+1$阶的特征交互。

从位级别和特征级别的角度来看，交叉网络能够为$l$层交叉网络创建所有直到$l+1$阶的特征交互。与DCN-V相比，DCN-V2以更多参数刻画相同的多项式类，更具表达能力。

### 4.2 与相关工作的联系

**DCN。** 我们的模型主要受DCN[50]启发。让我们采用DCN的高效投影视角，即它隐式生成所有成对交叉然后投影到低维空间；DCN-V2类似但具有不同的投影结构。

**DLRM和DeepFM。** 两者本质上是没有DNN组件的2阶FM（忽略微小差异）。因此，我们简化分析并与FM比较，其公式为$x^\top \beta + \sum_{i<j} w_{ij} \langle x_i, x_j \rangle$。这等价于具有结构化权重矩阵的1层DCN-V2（无残差项）。

**xDeepFM。** 第$k$层的第$h$个特征图为$x_{h,*}^k = \sum_{i=1}^{k-1} \sum_{j=1}^{m} w_{ij}^{k,h} (x_{i,*}^{k-1} \odot x_j)$。第1层的第$h$个特征图等价于1层DCN-V2（无残差项）。

**AutoInt。** 从高层视角看，AutoInt的第1层输出$\tilde{e}x = [\tilde{e}x_1; \tilde{e}x_2; \ldots; \tilde{e}x_k]$，其中$\tilde{e}x_i$编码所有与第$i$个特征的2阶特征交互。然后$\tilde{e}x$被输入第2层学习更高阶交互。这与DCN-V2相同。

**PNN。** 内积版本（IPNN）类似于FM。对于外积版本（OPNN），它首先显式创建所有$d^2$成对交互，然后使用$d' \times d^2$密集矩阵将它们投影到低维空间$d'$。不同的是，DCN-V2使用结构化矩阵隐式创建交互。

## 5 研究问题

我们有兴趣寻求以下研究问题的答案：

- **RQ1** 特征交互学习方法在何时会比基于ReLU的DNN更高效？
- **RQ2** 每个基线的特征交互组件在不与DNN集成时表现如何？
- **RQ3** 提出的DCN-V2方法与基线相比如何？我们能否通过DCN-V2和低秩混合DCN实现模型精度和成本之间更优的权衡？
- **RQ4** DCN-V2中的设置如何影响模型质量？
- **RQ5** DCN-V2是否捕获了重要的特征交叉？模型是否提供了良好的可理解性？

## 6 特征交叉技术的实证理解（RQ1）

许多最近的工作[1, 6, 13, 26, 34, 35, 50]提出建模无法从传统神经网络中高效学习的显式特征交叉。然而，大多数工作仅研究了具有未知交叉模式和噪声数据的公开数据集；很少有工作在已知真实模型的干净设置中进行研究。因此，理解以下问题很重要：1）在哪些情况下传统神经网络变得低效；2）DCN-V2交叉网络中每个组件的作用。

我们使用DCN模型中的交叉网络来表示那些特征交叉方法，并与ReLU进行比较，ReLU在工业推荐系统中被广泛使用。为了简化实验和便于理解，我们假设每个特征$x_i$的维度为一，单项式$x_1^{\alpha_1} x_2^{\alpha_2} \ldots x_d^{\alpha_d}$表示特征之间的$|\boldsymbol{\alpha}|$阶交互。

**难度递增的性能。** 考虑仅2阶特征交叉，真实模型为$f(x) = \sum_{|\boldsymbol{\alpha}|=2} w_{\boldsymbol{\alpha}} x_1^{\alpha_1} x_2^{\alpha_2} \ldots x_d^{\alpha_d}$。则学习$f(x)$的难度取决于：1）稀疏性（$w_{\boldsymbol{\alpha}} = 0$），交叉的数量；2）交叉模式的相似性（由$\text{Var}(w_{\boldsymbol{\alpha}})$表征），意味着一个特征的变化会同时以相似的量影响大多数特征交叉。

**表1：难度递增的多项式拟合的RMSE和模型大小（参数数量）**

| | DCN (1层) | DCN-V2 (1层) | DNN (1层) | DNN (大) |
|---|---|---|---|---|
| | RMSE | Size | RMSE | Size | RMSE | Size | RMSE | Size |
| $f_1$ | 8.9E-13 | 12 | 5.1E-13 | 24 | 2.7E-2 | 24 | 4.7E-3 | 41K |
| $f_2$ | 1.0E-01 | 9 | 4.5E-15 | 15 | 3.0E-2 | 15 | 1.4E-3 | 41K |
| $f_3$ | 2.6E+00 | 300 | 6.7E-07 | 10K | 2.7E-1 | 10K | 7.8E-2 | 758K |

当交叉模式简单时（$f_1$），DCN-V2和DCN都很高效。当模式变得更复杂时（$f_3$），DCN-V2保持准确而DCN退化。即使使用更宽更深的结构，DNN的性能仍然很差。这表明DNN在建模单项式模式方面的低效性。

**每个组件的作用。** 我们还对3阶和4阶齐次多项式进行了消融研究。显然，$x_0 \odot (W x_i)$在第$d-1$层建模$d$阶交叉，这通过3阶多项式的最佳性能在第2层实现得到验证。在其他层，性能显著下降。这就是偏置和残差项发挥作用的地方——它们创建并维护所有直到最高阶的交叉。

**表2：组合阶（1-4）多项式拟合**

| #Layers | 1 | 2 | 3 | 4 | 5 |
|---------|---|---|---|---|---|
| DCN-V2 | 1.43E-01 | 2.89E-02 | 9.82E-03 | 9.87E-03 | 9.92E-03 |
| DNN | 1.32E-01 | 1.03E-01 | 1.03E-01 | 1.09E-01 | 1.05E-01 |

总结：即使使用更深更大的网络，ReLU在捕获显式特征交叉（乘法关系）方面效率低下。当交叉模式变得更复杂时，精度显著下降。DCN准确捕获简单交叉模式但在更复杂的模式上失败。另一方面，DCN-V2对复杂交叉模式保持准确和高效。

## 7 实验结果（RQ2-RQ5）

本节通过3个数据集和2个平台上的实验验证了DCN-V2在特征交互学习方面的有效性，与SOTA进行比较。鉴于最近对已发表结果可重复性差的担忧[8, 33, 38]，我们进行了公平且全面的实验研究，包含广泛的超参数搜索以适当调优所有基线和提出的方法。此外，对于每个最优设置，我们使用不同的随机初始化训练5个模型，并报告均值和标准差。

### 7.1 实验设置

**数据集。** 表3列出了每个数据集的统计信息：

**表3：数据集**

| 数据 | 样本数 | 特征数 | 词汇表大小 |
|------|--------|--------|-----------|
| Criteo | 45M | 39 | 2.3M |
| MovieLen-1M | 740k | 7 | 3.5k |
| Production | > 100B | NA | NA |

**Criteo。** 最流行的点击率（CTR）预测基准数据集，包含7天的用户日志。我们遵循[46, 50]，使用前6天进行训练，并将最后一天的数据随机等分为验证集和测试集。

**MovieLen-1M。** 最流行的推荐系统研究数据集。每个训练样本包括一个$\langle\text{用户特征}, \text{电影特征}, \text{评分}\rangle$三元组。与AutoInt[46]类似，我们将任务形式化为回归问题。数据随机分为80%训练、10%验证和10%测试。

**基线。** 我们将提出的方法与6种SOTA特征交互学习算法进行比较。

**实现细节。** 所有基线和我们的方法都在TensorFlow v1中实现。为了公平比较，除了特征交互组件外，所有模型的实现都是相同的。

**嵌入。** 除DNN和DCN外，所有基线要求每个特征的嵌入大小相同。因此，我们将其固定为$\text{Avg}(\sum_{\text{vocab}} 6 \cdot (\text{vocab cardinality})^{1/4})$（Criteo为39，Movielen-1M为30）。

**优化。** 我们使用Adam[22]，批大小为512（MovieLen为128）。核使用He Normal[15]初始化，偏置初始化为0；梯度裁剪范数为10；对训练参数应用衰减为0.9999的指数移动平均。

**超参数调优和结果报告。** 对于所有基线，我们首先在超参数上进行粗粒度（大范围）网格搜索，然后进行更细粒度（小范围）搜索。为了确保可重复性和减少模型方差，对于每种方法和数据集，我们报告5次独立运行的最佳配置的均值和标准差。

### 7.2 特征交互组件单独性能（RQ2）

我们考虑每种模型的特征交互组件，不包含DNN组件。此外，我们仅考虑类别特征，因为密集特征在基线之间的处理方式不同。

**表5：每个模型特征交互组件的LogLoss（测试）。仅使用类别特征。**

| 模型 | LogLoss | 最佳设置 |
|------|---------|---------|
| **2阶方法** | | |
| PNN [35] | 0.4715 ± 4.430e-04 | OPNN, kernel=matrix |
| FM | 0.4736 ± 3.04E-04 | – |
| **>2阶方法** | | |
| CIN [26] | 0.4719 ± 9.41E-04 | l=3, cinLayerSize=100 |
| AutoInt [46] | 0.4711 ± 1.62E-04 | l=2, head=3, attEmbed=40 |
| DNN | 0.4704 ± 1.57E-04 | l=2, size=1024 |
| CrossNet | 0.4702 ± 3.80E-04 | l=2 |
| CrossNet-Mix | 0.4694 ± 4.35E-04 | l=5, expert=4 |

有两个主要观察：1）高阶方法展示了优于2阶方法的性能。这表明高阶交叉在这个数据集中是有意义的。2）在高阶方法中，交叉网络取得了最佳性能，与DNN相当或略好。

### 7.3 基线性能（RQ3）

本节以端到端方式比较DCN-V2方法与基线的性能。注意，报告的每个模型的最佳设置是在广泛的模型容量和超参数空间上搜索的。

**表6：Criteo和Movielen-1M上的LogLoss和AUC（测试）。指标在5次独立运行上平均，括号内为标准差。**

| 基线 | Criteo | | | MovieLens-1M | | |
|------|--------|---|---|-------------|---|---|
| | Logloss | AUC | Params | FLOPS | Logloss | AUC | Params | FLOPS |
| PNN | 0.4421 | 0.8099 | 3.1M | 6.1M | 0.3182 | 0.8955 | 54K | 110K |
| DeepFM | 0.4420 | 0.8099 | 1.4M | 2.8M | 0.3202 | 0.8932 | 46K | 93K |
| DLRM | 0.4427 | 0.8092 | 1.1M | 2.2M | 0.3245 | 0.8890 | 7.7K | 16K |
| xDeepFM | 0.4421 | 0.8099 | 3.7M | 32M | 0.3251 | 0.8923 | 160K | 990K |
| AutoInt+ | 0.4420 | 0.8101 | 4.2M | 8.7M | 0.3204 | 0.8928 | 260K | 500K |
| DCN | 0.4420 | 0.8099 | 2.1M | 4.2M | 0.3197 | 0.8935 | 110K | 220K |
| DNN | 0.4421 | 0.8098 | 3.2M | 6.3M | 0.3201 | 0.8929 | 46K | 92K |
| **DCN-V2** | **0.4406** | **0.8115** | 3.5M | 7.0M | 0.3170 | 0.8950 | 110K | 220K |
| **DCN-Mix** | 0.4408 | 0.8112 | 2.4M | 4.8M | **0.3160** | **0.8964** | 110K | 210K |
| CrossNet | 0.4413 | 0.8107 | 2.1M | 4.2M | 0.3185 | 0.8937 | 65K | 130K |

DCN-V2始终优于基线（包括DNN），并实现了良好的质量/成本权衡。

### 7.4 交叉层能否替代ReLU层？

DCN-V2方法的出色性能启发我们进一步研究其交叉层（CrossNet）在学习显式高阶特征交叉方面的效率。

**表7：固定内存预算下的Logloss和AUC（测试）**

| #Params | 7.9E+05 | 1.3E+06 | 2.1E+06 | 2.6E+06 |
|---------|---------|---------|---------|---------|
| **LogLoss** | | | | |
| CrossNet | 0.4424 | 0.4417 | 0.4416 | 0.4415 |
| DNN | 0.4427 | 0.4426 | 0.4423 | 0.4423 |
| **AUC** | | | | |
| CrossNet | 0.8096 | 0.8104 | 0.8105 | 0.8106 |
| DNN | 0.8091 | 0.8094 | 0.8096 | 0.80961 |

最佳性能由交叉网络（5层）实现，表明真实模型可以被多项式很好地近似。此外，每个内存限制下的最佳性能也由交叉网络实现，表明其既有效又高效。

### 7.5 超参数选择如何影响DCN-V2模型性能（RQ4）

**交叉层深度。** 图5a显示了在Criteo数据集上增加层深度时的测试LogLoss和AUC。随着更深的交叉网络，质量稳步提高，表明它能够捕获更有意义的交叉。然而，当使用更多层时，改进的速度减慢。

**矩阵秩。** 权重矩阵的秩控制参数数量以及通过交叉层的低频信号部分。图5b显示了矩阵秩$r$对测试LogLoss和AUC的影响。当$r$仅为4时，性能与其他基线相当。当$r$从4增加到64时，LogLoss几乎随$r$线性下降。当$r$从64进一步增加到完整时，LogLoss的改进减慢。

**专家数量。** 我们研究了低秩专家数量如何影响质量。我们观察到：1）最佳设置（专家数、门控、矩阵激活类型）取决于数据集和模型架构；2）每个设置的最佳模型产生了类似的结果。

### 7.6 模型理解（RQ5）

一个关键的研究问题是提出的方法是否确实学习了有意义的特征交叉。对学习的特征交叉的良好理解有助于提高模型的可理解性，对ML公平性和ML健康等领域特别重要。幸运的是，DCN-V2中的权重矩阵$W$恰好揭示了模型学习了哪些重要的特征交叉。

具体来说，我们假设每个输入$x = [x_1; x_2; \ldots; x_k]$包含$k$个特征，每个由嵌入$x_i$表示。然后，公式(7)中特征交叉组件的逐块视图（忽略偏置）表明第$i$个和第$j$个特征之间特征交互的重要性由$(i, j)$块$W_{i,j}$表征。

## 8 在Google生产化DCN-V2

本节提供一个案例研究，分享我们在Google大规模推荐系统中生产化DCN-V2的经验。我们通过DCN-V2在离线模型精度和在线关键业务指标方面都取得了显著收益。

**排序问题：** 给定一个用户和大量候选，我们的问题是返回用户最可能参与的top-$k$个item。

**生产数据和模型：** 生产数据是采样的用户日志，包含数千亿训练样本。稀疏特征的词汇表大小从2到数百万不等。基线模型是具有ReLU激活的全连接多层感知器（MLP）。

**与生产模型的比较：** 与生产模型相比，DCN-V2产生了0.6%的AUCLoss（1-AUC）改进。对于这个特定模型，AUCLoss上0.1%的收益被认为是显著改进。我们还在关键指标上观察到显著的在线性能提升。

**表8：DCN-V2与同规模ReLU的相对AUCLoss**

| | 1层ReLU | 2层ReLU | 1层DCN-V2 | 2层DCN-V2 |
|---|---------|---------|-----------|-----------|
| 相对改进 | 0% | -0.15% | -0.19% | -0.45% |

**实践经验。** 我们分享通过生产化DCN-V2学到的一些实践经验：

- 最好将交叉层插入DNN的输入层和隐藏层之间（也在[44]中观察到）。我们的假设是，随着远离输入层，特征表示及其交互的物理含义变弱。
- 通过堆叠或拼接1-2个交叉层，我们看到了一致的精度提升。超过2个交叉层后，收益开始趋于平稳。
- 我们观察到堆叠交叉层和拼接交叉层都工作良好。堆叠层学习更高阶的特征交互，而拼接层（类似于多头机制[48]）捕获互补交互。
- 我们观察到使用秩为（输入大小）/4的低秩DCN始终保留了全秩DCN-V2的精度。

## 9 结论与未来工作

在本文中，我们提出了一个新模型——DCN-V2——以表达性强且简单的方式建模显式交叉。观察到交叉网络中权重矩阵的低秩特性，我们还提出了低秩混合DCN（DCN-Mix）以实现模型性能和延迟之间更优的权衡。DCN-V2已在多个Web规模学习排序系统中成功部署，带来了显著的离线模型精度和在线业务指标提升。我们的实验结果也证明了DCN-V2相对于SOTA方法的有效性。

对于未来工作，我们有兴趣推进对以下方面的理解：1）DCN-V2与优化算法（如二阶方法）之间的交互；2）嵌入、DCN-V2及其矩阵秩之间的关系。此外，我们希望改进DCN-Mix中的门控机制。此外，观察到DCN-V2中的交叉层可能作为DNN中ReLU层的潜在替代，我们非常有兴趣在更复杂的模型架构（如RNN、CNN）中验证这一观察。

## 参考文献

[1] Alex Beutel, Paul Covington, Sagar Jain, Can Xu, Jia Li, Vince Gatto, and Ed H Chi. 2018. Latent cross: Making use of context in recurrent recommender systems. In Proceedings of the Eleventh ACM International Conference on Web Search and Data Mining. 46–54.

[2] Léon Bottou, Jonas Peters, Joaquin Quiñonero-Candela, Denis X Charles, Max D Chickering, Elon Portugaly, Dipankar Ray, Patrice Simard, and Ed Snelson. 2013. Counterfactual reasoning and learning systems: The example of computational advertising. The Journal of Machine Learning Research 14, 1 (2013), 3207–3260.

[3] Andrei Z Broder. 2008. Computational advertising and recommender systems. In Proceedings of the 2008 ACM conference on Recommender systems. 1–2.

[4] Zhe Cao, Tao Qin, Tie-Yan Liu, Ming-Feng Tsai, and Hang Li. 2007. Learning to rank: from pairwise approach to listwise approach. In Proceedings of the 24th international conference on Machine learning. 129–136.

[5] Ting Chen, Ji Lin, Tian Lin, Song Han, Chong Wang, and Denny Zhou. 2018. Adaptive mixture of low-rank factorizations for compact neural modeling. (2018).

[6] Heng-Tze Cheng, Levent Koc, Jeremiah Harmsen, Tal Shaked, Tushar Chandra, Hrishi Aradhye, Glen Anderson, Greg Corrado, Wei Chai, Mustafa Ispir, et al. 2016. Wide & Deep Learning for Recommender Systems. arXiv preprint arXiv:1606.07792 (2016).

[7] Weiyu Cheng, Yanyan Shen, and Linpeng Huang. 2019. Adaptive Factorization Network: Learning Adaptive-Order Feature Interactions. arXiv preprint arXiv:1909.03276 (2019).

[8] Maurizio Ferrari Dacrema, Paolo Cremonesi, and Dietmar Jannach. 2019. Are we really making much progress? A worrying analysis of recent neural recommendation approaches. In Proceedings of the 13th ACM Conference on Recommender Systems. 101–109.

[9] Petros Drineas and Michael W Mahoney. 2005. On the Nyström method for approximating a Gram matrix for improved kernel-based learning. journal of machine learning research 6, Dec (2005), 2153–2175.

[10] David Eigen, Marc'Aurelio Ranzato, and Ilya Sutskever. 2013. Learning factored representations in a deep mixture of experts. arXiv preprint arXiv:1312.4314 (2013).

[11] Yuwei Fan, Jordi Feliu-Faba, Lin Lin, Lexing Ying, and Leonardo Zepeda-Núnez. 2019. A multiscale neural network based on hierarchical nested bases. Research in the Mathematical Sciences 6, 2 (2019), 21.

[12] Gene H Golub and Charles F Van Loan. 1996. Matrix Computations Johns Hopkins University Press. Baltimore and London (1996).

[13] Huifeng Guo, Ruiming Tang, Yunming Ye, Zhenguo Li, and Xiuqiang He. 2017. DeepFM: a factorization-machine based neural network for CTR prediction. arXiv preprint arXiv:1703.04247 (2017).

[14] Nathan Halko, Per-Gunnar Martinsson, and Joel A Tropp. 2011. Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions. SIAM review 53, 2 (2011), 217–288.

[15] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. 2015. Delving deep into rectifiers: Surpassing human-level performance on imagenet classification. In Proceedings of the IEEE international conference on computer vision. 1026–1034.

[16] Xiangnan He and Tat-Seng Chua. 2017. Neural factorization machines for sparse predictive analytics. In Proceedings of the 40th International ACM SIGIR conference on Research and Development in Information Retrieval. 355–364.

[17] Jonathan L Herlocker, Joseph A Konstan, Loren G Terveen, and John T Riedl. 2004. Evaluating collaborative filtering recommender systems. ACM Transactions on Information Systems (TOIS) 22, 1 (2004), 5–53.

[18] Sepp Hochreiter and Jürgen Schmidhuber. 1997. Long short-term memory. Neural computation 9, 8 (1997), 1735–1780.

[19] Robert A Jacobs, Michael I Jordan, Steven J Nowlan, and Geoffrey E Hinton. 1991. Adaptive mixtures of local experts. Neural computation 3, 1 (1991), 79–87.

[20] Max Jaderberg, Andrea Vedaldi, and Andrew Zisserman. 2014. Speeding up convolutional neural networks with low rank expansions. arXiv preprint arXiv:1405.3866 (2014).

[21] Eric Jang, Shixiang Gu, and Ben Poole. 2016. Categorical reparameterization with gumbel-softmax. arXiv preprint arXiv:1611.01144 (2016).

[22] Diederik Kingma and Jimmy Ba. 2014. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980 (2014).

[23] Steve Lawrence, C Lee Giles, Ah Chung Tsoi, and Andrew D Back. 1997. Face recognition: A convolutional neural-network approach. IEEE transactions on neural networks 8, 1 (1997), 98–113.

[24] Yann LeCun, Bernhard Boser, John S Denker, Donnie Henderson, Richard E Howard, Wayne Hubbard, and Lawrence D Jackel. 1989. Backpropagation applied to handwritten zip code recognition. Neural computation 1, 4 (1989), 541–551.

[25] Zeyu Li, Wei Cheng, Yang Chen, Haifeng Chen, and Wei Wang. 2020. Interpretable Click-Through Rate Prediction through Hierarchical Attention. In Proceedings of the 13th International Conference on Web Search and Data Mining. 313–321.

[26] Jianxun Lian, Xiaohuan Zhou, Fuzheng Zhang, Zhongxia Chen, Xing Xie, and Guangzhong Sun. 2018. xdeepfm: Combining explicit and implicit feature interactions for recommender systems. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 1754–1763.

[27] Tie-Yan Liu. 2011. Learning to rank for information retrieval. Springer Science & Business Media.

[28] Christos Louizos, Max Welling, and Diederik P Kingma. 2017. Learning Sparse Neural Networks through $L_0$ Regularization. arXiv preprint arXiv:1712.01312 (2017).

[29] Jiaqi Ma, Zhe Zhao, Jilin Chen, Ang Li, Lichan Hong, and Ed H Chi. 2019. Snr: Sub-network routing for flexible parameter sharing in multi-task learning. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 33. 216–223.

[30] Jiaqi Ma, Zhe Zhao, Xinyang Yi, Jilin Chen, Lichan Hong, and Ed H Chi. 2018. Modeling task relationships in multi-task learning with multi-gate mixture-of-experts. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 1930–1939.

[31] Hrushikesh N Mhaskar. 1996. Neural networks for optimal approximation of smooth and analytic functions. Neural computation 8, 1 (1996), 164–171.

[32] Tomáš Mikolov, Stefan Kombrink, Lukáš Burget, Jan Černocký, and Sanjeev Khudanpur. 2011. Extensions of recurrent neural network language model. In 2011 IEEE international conference on acoustics, speech and signal processing (ICASSP). IEEE, 5528–5531.

[33] Kevin Musgrave, Serge Belongie, and Ser-Nam Lim. 2020. A metric learning reality check. arXiv preprint arXiv:2003.08505 (2020).

[34] Maxim Naumov, Dheevatsa Mudigere, Hao-Jun Michael Shi, Jianyu Huang, Narayanan Sundaraman, Jongsoo Park, Xiaodong Wang, Udit Gupta, Carole-Jean Wu, Alisson G Azzolini, et al. 2019. Deep learning recommendation model for personalization and recommendation systems. arXiv preprint arXiv:1906.00091 (2019).

[35] Yanru Qu, Han Cai, Kan Ren, Weinan Zhang, Yong Yu, Ying Wen, and Jun Wang. 2016. Product-based neural networks for user response prediction. In 2016 IEEE 16th International Conference on Data Mining (ICDM). IEEE, 1149–1154.

[36] Steffen Rendle. 2010. Factorization machines. In 2010 IEEE International Conference on Data Mining. IEEE, 995–1000.

[37] Steffen Rendle. 2012. Factorization Machines with libFM. ACM Trans. Intell. Syst. Technol. 3, 3, Article 57 (May 2012), 22 pages.

[38] Steffen Rendle, Walid Krichene, Li Zhang, and John Anderson. 2020. Neural Collaborative Filtering vs. Matrix Factorization Revisited. arXiv preprint arXiv:2005.09683 (2020).

[39] Paul Resnick and Hal R Varian. 1997. Recommender systems. Commun. ACM 40, 3 (1997), 56–58.

[40] David E Rumelhart, Geoffrey E Hinton, and Ronald J Williams. 1985. Learning internal representations by error propagation. Technical Report. California Univ San Diego La Jolla Inst for Cognitive Science.

[41] J Ben Schafer, Joseph Konstan, and John Riedl. 1999. Recommender systems in e-commerce. In Proceedings of the 1st ACM conference on Electronic commerce. 158–166.

[42] Jürgen Schmidhuber. 2015. Deep learning in neural networks: An overview. Neural networks 61 (2015), 85–117.

[43] Frank Seide, Gang Li, Xie Chen, and Dong Yu. 2011. Feature engineering in context-dependent deep neural networks for conversational speech transcription. In 2011 IEEE Workshop on Automatic Speech Recognition & Understanding. IEEE, 24–29.

[44] Ying Shan, T Ryan Hoens, Jian Jiao, Haijing Wang, Dong Yu, and JC Mao. 2016. Deep Crossing: Web-Scale Modeling without Manually Crafted Combinatorial Features. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. ACM, 255–262.

[45] Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. 2017. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. arXiv preprint arXiv:1701.06538 (2017).

[46] Weiping Song, Chence Shi, Zhiping Xiao, Zhijian Duan, Yewen Xu, Ming Zhang, and Jian Tang. 2019. Autoint: Automatic feature interaction learning via self-attentive neural networks. In Proceedings of the 28th ACM International Conference on Information and Knowledge Management. 1161–1170.

[47] Gregory Valiant. 2014. Learning polynomials with neural networks. (2014).

[48] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in neural information processing systems. 5998–6008.

[49] Andreas Veit, Michael J Wilber, and Serge Belongie. 2016. Residual Networks Behave Like Ensembles of Relatively Shallow Networks. In Advances in Neural Information Processing Systems 29, D. D. Lee, M. Sugiyama, U. V. Luxburg, I. Guyon, and R. Garnett (Eds.). Curran Associates, Inc., 550–558.

[50] Ruoxi Wang, Bin Fu, Gang Fu, and Mingliang Wang. 2017. Deep & Cross Network for Ad Click Predictions. In Proceedings of the ADKDD'17. 1–7.

[51] Ruoxi Wang, Yingzhou Li, Michael W Mahoney, and Eric Darve. 2019. Block Basis Factorization for Scalable Kernel Evaluation. SIAM J. Matrix Anal. Appl. 40, 4 (2019), 1497–1526.

[52] Xiyu Yu, Tongliang Liu, Xinchao Wang, and Dacheng Tao. 2017. On compressing deep models by low rank and sparse decomposition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 7370–7379.

---

## 附录

### 10 论文中报告的基线性能

表9列出了每篇论文中引用的基线的Logloss和AUC指标。

### 11 定理证明

#### 11.1 定理4.2的证明

**证明。** 我们从符号开始；然后通过归纳法证明。

**符号。** 设$[k] := \{1, \ldots, k\}$。将嵌入表示为$x = [x_1; x_2; \ldots; x_c]$，第$l$个交叉层的输出为$x_l = [x_l_1; x_l_2; \ldots; x_l_c]$，其中$x_i, x_l_i \in \mathbb{R}^{e_i}$，$e_i$是第$i$个特征的嵌入大小。为了简化符号，我们还定义有序集合$I$中特征之间的特征交互，其权重由有序集合$J$表征。

**命题。** 我们首先通过归纳法证明$x_l_i$具有以下公式：

$$
x_l_i = \sum_{p=2}^{l+1} \sum_{I \in S_i^p} \sum_{J \in C_l^{p-1}} g(I, J; x, W) + x_i \qquad (9)
$$

其中$S_i^p$是一个集合，表示从$[c]$中选择$p$个元素的所有组合，第一个元素固定为$i$；$C_l^{p-1}$是一个集合，表示从整数$[l]$中一次选择$p-1$个索引的组合。

**基础情况。** 当$l=1$时，$x_1_i = \sum_j W^1_{i,j} x_j + x_i$。

**归纳步骤。** 假设当$l=k$时公式成立，则对于$l=k+1$，我们有：

$$
x_{k+1}_i = x_i \odot \sum_{q=1}^{c} W^{k+1}_{i,q} x_k_q + x_k_i
$$

经过代数运算，我们得到：

$$
x_{k+1}_i = \sum_{p=2}^{k+2} \sum_{I \in S_i^p} \sum_{J \in C_{k+1}^{p-1}} g(I, J; x, W) + x_i
$$

**结论。** 由于基础情况和归纳步骤都成立，我们得出结论$\forall l \geq 1$，公式(9)成立。这完成了证明。$\square$

#### 11.2 定理4.1的证明

**证明。** 我们不将每个特征嵌入视为一个单元，而是将输入嵌入$x = [x_1, x_2, \ldots, x_d]$中的每个元素$x_i$视为一个单元。这是定理4.2的特殊情况，其中所有特征嵌入大小为1。在这种情况下，所有计算都是可交换的。

$$
x_l_i = \sum_{p=2}^{l+1} \sum_{I \in S_i^p} \sum_{J \in C_l^{p-1}} g(I, J; x, W) + x_i \qquad (10)
$$

为了简化证明和最终公式，假设$l$层交叉网络的最终logit为$\mathbf{1}^\top x_l$，则：

$$
\mathbf{1}^\top x_l = \sum_{\boldsymbol{\alpha}} \sum_{\mathbf{j} \in C_l^{|\boldsymbol{\alpha}|-1}} \sum_{\mathbf{i} \in P_{\boldsymbol{\alpha}}} \prod_{k=1}^{|\boldsymbol{\alpha}|-1} w^{(j_k)}_{i_k i_{k+1}} x_1^{\alpha_1} x_2^{\alpha_2} \cdots x_d^{\alpha_d} + \sum_{i=1}^{d} x_i
$$

其中$P_{\boldsymbol{\alpha}}$是$(1 \cdots 1 \cdots d \cdots d)$的所有排列的集合。$\square$
