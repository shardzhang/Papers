# 动态路由间的胶囊（Dynamic Routing Between Capsules）

> Sara Sabour, Nicholas Frosst, Geoffrey E. Hinton | Google Brain, Toronto | {sasabour, frosst, geoffhinton}@google.com

本文提出胶囊网络（CapsNet），用向量输出神经元替代传统标量输出特征检测器，并通过迭代路由协议（routing-by-agreement）替代最大池化，从而实现更优的特征绑定与分割能力。

- 胶囊（capsule）是一组神经元，其活动向量表示实体（如物体或物体部件）的实例化参数。向量长度表示实体存在概率，方向表示实例化属性
- 使用迭代动态路由机制：低层胶囊通过标量积（agreement）决定输出发送给哪个高层胶囊，替代最大池化的硬路由
- 在 MNIST 上达到 0.25% 测试错误率（3 层网络，无需集成），优于传统 CNN；在高度重叠数字分割任务（MultiMNIST）上达到 5.0% 错误率，与带注意力机制的序列模型持平但重叠度更高（80% vs <4%）
- 重构正则化增强了胶囊对姿态编码的鲁棒性；仿射变换泛化能力显著优于传统 CNN（79% vs 66% on affNIST）

关键发现：

- 动态路由的"解释消除"（explaining away）能力使模型能分割高度重叠物体，无需像素级分割
- 胶囊向量维度具有可解释性：不同维度编码宽度、倾斜、笔画粗细等独立变化因素
- 3 次路由迭代在收敛速度与泛化能力间取得最佳平衡
- 胶囊假设每个位置最多存在一个实体实例，通过分布式表示实现指数级更高效的编码

---

## 摘要

胶囊是一组神经元，其活动向量表示特定类型实体（如物体或物体部件）的实例化参数。我们使用活动向量的长度表示实体存在的概率，其方向表示实例化参数。某一层的活跃胶囊通过变换矩阵为高层胶囊的实例化参数生成预测。当多个预测达成一致时，高层胶囊被激活。我们证明，经过判别式训练的多层胶囊系统在 MNIST 上达到了最先进的性能，并且在识别高度重叠数字方面明显优于卷积网络。为实现这些结果，我们使用了一种迭代的"路由协议"机制：低层胶囊倾向于将其输出发送给那些活动向量与该低层胶囊预测具有较大标量积的高层胶囊。

---

## 1 引言

人类视觉通过精心确定的注视点序列来忽略无关细节，确保只有极小部分光学阵列在高分辨率下被处理。内省很难帮助我们理解场景知识有多少来自注视序列、又有多少来自单次注视，但在本文中，我们假设单次注视提供的信息远超单个已识别物体及其属性。我们假设多层视觉系统在每个注视点上构建一棵类似解析树的结构，并且我们暂不讨论这些单次注视解析树如何在多次注视间协调的问题。

解析树通常通过动态分配内存即时构建。然而，遵循 Hinton 等人 [9] 的思路，我们假设对于单次注视而言，解析树是从一个固定的多层神经网络中雕刻出来的，就像雕塑从岩石中雕出一样。每一层被划分为许多称为"胶囊"的神经元小组（Hinton 等人 [10]），解析树中的每个节点对应一个活跃胶囊。通过迭代路由过程，每个活跃胶囊选择上一层中的一个胶囊作为其在树中的父节点。对于视觉系统的更高层，这一迭代过程将解决将部件分配给整体的任务。

活跃胶囊内神经元的活动表示图像中存在的特定实体的各种属性。这些属性可包括多种不同类型的实例化参数，如姿态（位置、大小、方向）、形变、速度、反照率、色调、纹理等。一个非常特殊的属性是实例化实体在图像中的存在性。表示存在性的一种直观方式是使用单独的 logistic 单元，其输出为实体存在的概率。在本文中，我们探索了一种有趣的替代方案：使用实例化参数向量的整体长度来表示实体的存在性，并强制向量的方向表示实体的属性[^1]。我们通过应用一个非线性函数确保胶囊向量输出的长度不超过 1，该函数保持向量方向不变但缩放其幅度。

胶囊输出是向量这一事实使得我们可以使用强大的动态路由机制，确保胶囊的输出被发送到上一层中合适的父节点。初始时，输出被路由到所有可能的父节点，但通过耦合系数（总和为 1）进行缩放。对于每个可能的父节点，胶囊通过将其自身输出乘以权重矩阵来计算"预测向量"。如果该预测向量与某个可能父节点的输出具有较大的标量积，则存在自上而下的反馈，增加该父节点的耦合系数并减小其他父节点的耦合系数。这增加了胶囊对该父节点的贡献，从而进一步增大胶囊预测与父节点输出的标量积。这种"路由协议"机制应远优于最大池化实现的非常原始的路由形式——后者允许一层中的神经元忽略底层局部池中除最活跃特征检测器之外的所有神经元。我们证明，动态路由机制是实现分割高度重叠物体所需的"解释消除"（explaining away）的有效方法。

卷积神经网络（CNN）使用学习到的特征检测器的平移副本。这使它们能够将在图像某一位置学到的良好权重值知识迁移到其他位置。这在图像解释中被证明极为有用。尽管我们正在用向量输出胶囊替代 CNN 的标量输出特征检测器，并用路由协议替代最大池化，我们仍然希望跨空间复制学到的知识。为此，除最后一层胶囊外，我们使所有胶囊层均为卷积层。与 CNN 类似，我们使高层胶囊覆盖图像中更大的区域。然而，与最大池化不同，我们不会丢弃关于实体在区域内精确位置的信息。对于低层胶囊，位置信息通过哪个胶囊活跃来进行"位置编码"（place-coded）。随着层级上升，越来越多的位置信息被"比率编码"（rate-coded）到胶囊输出向量的实值分量中。这种从位置编码到比率编码的转变，加上高层胶囊表示具有更多自由度的更复杂实体这一事实，表明胶囊的维度应随层级上升而增加。

[^1]: 这在生物学上是合理的，因为它不使用大的活动量来获得可能不存在的物体的精确表示。

## 2 胶囊的向量输入和输出如何计算

实现胶囊这一通用思路有许多可能的方式。本文的目的并非探索整个空间，而是展示一个相当直接的实现方式效果良好，并且动态路由有所帮助。

我们希望胶囊输出向量的长度表示该胶囊所代表的实体在当前输入中存在的概率。因此，我们使用非线性"squashing"函数确保短向量被压缩到几乎零长度，长向量被压缩到略低于 1 的长度。我们让判别式学习来充分利用这一非线性。

$$
\mathbf{v}_j = \frac{||\mathbf{s}_j||^2}{1 + ||\mathbf{s}_j||^2} \frac{\mathbf{s}_j}{||\mathbf{s}_j||} \qquad (1)
$$

其中 $\mathbf{v}_j$ 是胶囊 $j$ 的向量输出， $\mathbf{s}_j$ 是其总输入。

对于除第一层之外的所有胶囊层，胶囊 $j$ 的总输入 $\mathbf{s}_j$ 是来自低层所有胶囊的"预测向量" $\hat{\mathbf{u}}_{j|i}$ 的加权和，通过将低层胶囊的输出 $\mathbf{u}_i$ 乘以权重矩阵 $\mathbf{W}_{ij}$ 得到：

$$
\mathbf{s}_j = \sum_i c_{ij} \hat{\mathbf{u}}_{j|i}, \quad \hat{\mathbf{u}}_{j|i} = \mathbf{W}_{ij} \mathbf{u}_i \qquad (2)
$$

其中 $c_{ij}$ 是由迭代动态路由过程确定的耦合系数。胶囊 $i$ 与上一层所有胶囊之间的耦合系数之和为 1，并由"路由 softmax"确定，其初始 logits $b_{ij}$ 是胶囊 $i$ 应与胶囊 $j$ 耦合的先验对数概率：

$$
c_{ij} = \frac{\exp(b_{ij})}{\sum_k \exp(b_{ik})} \qquad (3)
$$

对数先验可以与其他所有权重同时进行判别式学习。它们取决于两个胶囊的位置和类型，但不取决于当前输入图像[^2]。初始耦合系数随后通过测量上一层中每个胶囊 $j$ 的当前输出 $\mathbf{v}_j$ 与胶囊 $i$ 所做的预测 $\hat{\mathbf{u}}_{j|i}$ 之间的一致性来迭代优化。

一致性即为标量积 $a_{ij} = \mathbf{v}_j \cdot \hat{\mathbf{u}}_{j|i}$ 。该一致性被视为对数似然，并加到初始 logit $b_{ij}$ 中，之后再计算连接胶囊 $i$ 与高层胶囊的所有耦合系数的新值。

在卷积胶囊层中，每个胶囊向上一层中的每种胶囊输出一个局部向量网格，对网格中的每个成员以及每种胶囊使用不同的变换矩阵。

[^2]: 对于 MNIST，我们发现将所有先验设置为相等就足够了。

**算法 1 路由算法。**

```
1: 过程 ROUTING(û_{j|i}, r, l)
2:     对于层 l 中的所有胶囊 i 和层 (l+1) 中的所有胶囊 j：b_{ij} $\leftarrow$ 0
3:     重复 r 次迭代 do
4:         对于层 l 中的所有胶囊 i：c_i $\leftarrow$ softmax(b_i)         ▷ softmax 计算式 (3)
5:         对于层 (l+1) 中的所有胶囊 j：s_j $\leftarrow$ $\sum$_i c_{ij} û_{j|i}
6:         对于层 (l+1) 中的所有胶囊 j：v_j $\leftarrow$ squash(s_j)      ▷ squash 计算式 (1)
7:         对于层 l 中的所有胶囊 i 和层 (l+1) 中的所有胶囊 j：b_{ij} $\leftarrow$ b_{ij} + û_{j|i} · v_j
8:     return v_j
```

## 3 数字存在性的边际损失

我们使用实例化向量的长度来表示胶囊实体存在的概率。我们希望数字类别 $k$ 的顶层胶囊有长的实例化向量当且仅当该数字存在于图像中。为允许多个数字，我们为每个数字胶囊 $k$ 使用单独的边际损失 $L_k$ ：

$$
L_k = T_k \max(0, m^+ - ||\mathbf{v}_k||)^2 + \lambda (1 - T_k) \max(0, ||\mathbf{v}_k|| - m^-)^2 \qquad (4)
$$

其中 $T_k = 1$ 当且仅当类别 $k$ 的数字存在[^3]， $m^+ = 0.9$ ， $m^- = 0.1$ 。对不存在的数字类别损失进行 $\lambda$ 降权重处理，防止初始学习阶段将所有数字胶囊的活动向量长度压缩到零。我们使用 $\lambda = 0.5$ 。总损失即为所有数字胶囊损失之和。

[^3]: 我们不允许图像包含同一数字类别的两个实例。我们在讨论部分讨论了胶囊的这一缺陷。

## 4 CapsNet 架构

一个简单的 CapsNet 架构如图 1 所示。该架构较浅，仅有两个卷积层和一个全连接层。Conv1 有 256 个 $9 \times 9$ 卷积核，步长为 1，使用 ReLU 激活。该层将像素强度转换为局部特征检测器的活动，然后作为主胶囊（primary capsules）的输入。

主胶囊是最低层的多维实体，从逆向图形学的角度看，激活主胶囊对应于逆转渲染过程。这是一种与将实例化部件拼凑成熟悉整体截然不同的计算类型——而后者正是胶囊所擅长的。

第二层（PrimaryCapsules）是一个卷积胶囊层，有 32 通道的卷积 8D 胶囊（即每个主胶囊包含 8 个卷积单元，使用 $9 \times 9$ 卷积核，步长为 2）。每个主胶囊输出看到所有 $256 \times 81$ 个 Conv1 单元的输出，这些单元的感受野与胶囊中心位置重叠。PrimaryCapsules 总共有 $[32 \times 6 \times 6]$ 个胶囊输出（每个输出是一个 8D 向量）， $[6 \times 6]$ 网格中的每个胶囊共享权重。可以将 PrimaryCapsules 视为一个以式 (1) 为其块非线性的卷积层。最后一层（DigitCaps）每个数字类别有一个 16D 胶囊，这些胶囊中的每一个都接收来自低层所有胶囊的输入。

我们仅在两个连续的胶囊层之间进行路由（例如 PrimaryCapsules 和 DigitCaps）。由于 Conv1 的输出是 1D 的，其空间中没有可供达成一致的方向。因此，Conv1 和 PrimaryCapsules 之间不使用路由。所有路由 logits ( $b_{ij}$ ) 初始化为零。因此，初始时胶囊输出 ( $\mathbf{u}_i$ ) 以等概率 ( $c_{ij}$ ) 发送给所有父胶囊 ( $\mathbf{v}_0...\mathbf{v}_9$ )。

我们的实现基于 TensorFlow [1]，并使用 Adam 优化器 [12] 及其 TensorFlow 默认参数（包括指数衰减学习率）来最小化式 (4) 中边际损失之和。

**图 1：一个具有 3 层的简单 CapsNet。**该模型的结果与深度卷积网络（如 Chang 和 Chen [3]）相当。DigitCaps 层中每个胶囊的活动向量长度表示每个类别实例的存在性，并用于计算分类损失。 $\mathbf{W}_{ij}$ 是 PrimaryCapsules 中的每个 $\mathbf{u}_i$ , $i \in (1, 32 \times 6 \times 6)$ 与 $\mathbf{v}_j$ , $j \in (1, 10)$ 之间的权重矩阵。

![图1](figure1.png)

### 4.1 重构作为正则化方法

我们使用额外的重构损失来鼓励数字胶囊编码输入数字的实例化参数。在训练期间，我们屏蔽除正确数字胶囊活动向量之外的所有向量。然后使用该活动向量重构输入图像。数字胶囊的输出被送入由 3 个全连接层组成的解码器，这些全连接层按图 2 所述对像素强度进行建模。我们最小化 logistic 单元输出与像素强度之间的平方差之和。我们将该重构损失缩小 0.0005 倍，使其在训练期间不会主导边际损失。如图 3 所示，来自 CapsNet 16D 输出的重构具有鲁棒性，同时仅保留重要细节。

**图 2：从 DigitCaps 层表示重构数字的解码器结构。**训练期间最小化图像与 Sigmoid 层输出之间的欧氏距离。训练时我们使用真实标签作为重构目标。

![图2](figure2.png)

**图 3：具有 3 次路由迭代的 CapsNet 在 MNIST 测试集上的重构样例。** $(l, p, r)$ 分别表示标签、预测和重构目标。最右侧两列显示了一个失败样例的两次重构，展示了模型如何将这张图像中的 5 和 3 混淆。其他列来自正确分类，显示模型在平滑噪声的同时保留了许多细节。

![图3](figure3.png)

**表 1：CapsNet 分类测试准确率。**MNIST 的平均值和标准差来自 3 次试验。

| 方法 | 路由 | 重构 | MNIST (%) | MultiMNIST (%) |
|------|------|------|-----------|----------------|
| Baseline | - | - | 0.39 | 8.1 |
| CapsNet | 1 | 否 | $0.34 \pm 0.032$ | - |
| CapsNet | 1 | 是 | $0.29 \pm 0.011$ | 7.5 |
| CapsNet | 3 | 否 | $0.35 \pm 0.036$ | - |
| CapsNet | 3 | 是 | $0.25 \pm 0.005$ | 5.2 |

## 5 胶囊在 MNIST 上的实验

训练在 $28 \times 28$ 的 MNIST [13] 图像上进行，图像在每个方向上进行最多 2 像素的零填充平移。未使用其他数据增强/形变。数据集包含 6 万张训练图像和 1 万张测试图像。

我们使用单一模型进行测试，未使用任何模型平均。Wan 等人 [18] 通过集成以及旋转和缩放数据增强实现了 0.21% 的测试错误率，不使用这些方法时为 0.39%。我们在一个 3 层网络上获得了之前只有更深网络才能达到的低测试错误率（0.25%）。表 1 报告了不同 CapsNet 配置在 MNIST 上的测试错误率，展示了路由和重构正则化的重要性。添加重构正则化通过强制在胶囊向量中进行姿态编码来提升路由性能。

基线是一个标准 CNN，包含三个卷积层（通道数分别为 256、256、128），每层使用 $5 \times 5$ 卷积核和步长 1。最后一个卷积层后接两个全连接层（大小分别为 328 和 192）。最后一个全连接层通过 dropout 连接到 10 类 softmax 层，使用交叉熵损失。基线也在 2 像素平移的 MNIST 上使用 Adam 优化器进行训练。基线旨在在保持计算成本尽可能接近 CapsNet 的情况下达到 MNIST 上的最佳性能。参数数量方面，基线有 3540 万参数，而 CapsNet 有 820 万参数（不含重构子网络为 680 万参数）。

### 5.1 胶囊各维度的含义

由于我们只传入一个数字的编码并清零其他数字，数字胶囊的维度应该学会覆盖该类数字实例化方式的变化空间。这些变化包括笔画粗细、倾斜度和宽度。它们还包括特定数字的变化，例如数字 2 尾部长度。我们可以通过利用解码器网络来观察各个维度的含义。在计算正确数字胶囊的活动向量后，我们可以将该活动向量的扰动版本输入解码器网络，观察扰动如何影响重构。图 4 展示了这些扰动的示例。我们发现胶囊的某一维度（16 维中）几乎总是表示数字的宽度。虽然某些维度表示全局变化的组合，但还有其他维度表示数字局部区域的变化。

**图 4：维度扰动。**每一行显示了 DigitCaps 表示的 16 个维度中某一维在 $[-0.25, 0.25]$ 范围内以 0.05 为间隔微调时的重构结果。

![图4](figure4.png)

### 5.2 对仿射变换的鲁棒性

实验表明，每个 DigitCaps 胶囊比传统卷积网络为每个类别学习了更鲁棒的表示。由于手写数字在倾斜、旋转、风格等方面存在自然变化，训练好的 CapsNet 对训练数据的小幅仿射变换具有中等的鲁棒性。

为测试 CapsNet 对仿射变换的鲁棒性，我们在填充和平移后的 MNIST 训练集上训练了 CapsNet 和传统卷积网络（带最大池化和 DropOut），其中每个样例是一个随机放置在 $40 \times 40$ 像素黑色背景上的 MNIST 数字。然后我们在 affNIST[^4] 数据集上测试该网络，其中每个样例是一个带有随机小幅仿射变换的 MNIST 数字。我们的模型从未使用除平移和标准 MNIST 中自然变换之外的仿射变换进行训练。一个采用早停策略、在扩展 MNIST 测试集上达到 99.23% 准确率的欠训练 CapsNet，在 affNIST 测试集上达到了 79% 的准确率。而一个参数数量相近、在扩展 MNIST 测试集上达到类似准确率（99.22%）的传统卷积模型，在 affNIST 测试集上仅达到 66%。

[^4]: 可从 http://www.cs.toronto.edu/~tijmen/affNIST/ 获取。

## 6 分割高度重叠的数字

动态路由可以看作是一种并行注意力机制，允许某一层的每个胶囊关注低层的一些活跃胶囊并忽略其他胶囊。这应能使模型识别图像中的多个物体，即使物体存在重叠。Hinton 等人提出了分割和识别高度重叠数字的任务 [9]，其他研究者也在类似领域测试了他们的网络 [5, 2, 6]。路由协议机制应能利用物体形状的先验信息来辅助分割，并避免在像素域中做出高层分割决策。

### 6.1 MultiMNIST 数据集

我们通过将同一集合（训练集或测试集）中但不同类别的两个数字叠加来生成 MultiMNIST 训练和测试数据集。每个数字在每个方向平移最多 4 个像素，生成 $36 \times 36$ 的图像。考虑到 $28 \times 28$ 图像中的数字被限制在 $20 \times 20$ 的边界框内，两个数字的边界框平均有 80% 的重叠。对于 MNIST 数据集中的每个数字，我们生成 1000 个 MultiMNIST 样例。因此训练集大小为 6000 万，测试集大小为 1000 万。

**图 5：具有 3 次路由迭代的 CapsNet 在 MultiMNIST 测试集上的重构样例。**两个重构的数字以绿色和红色叠加显示在下方图像中。上方图像显示输入图像。 $L:(l_1, l_2)$ 表示图像中两个数字的标签， $R:(r_1, r_2)$ 表示用于重构的两个数字。最右侧两列显示了两个错误分类的样例，分别从标签和预测 (P) 进行重构。在 (2, 8) 样例中，模型将 8 误认为 7；在 (4, 9) 样例中，它将 9 误认为 0。其他列为正确分类，显示模型能够解释所有像素，同时能够在极端困难的情况下将一个像素分配给两个数字（第 1-4 列）。注意在数据集生成过程中，像素值被裁剪到 1。带 (*) 标记的两列显示从既非标签也非预测的数字进行的重构。这些列表明模型并非仅寻找图像中所有数字（包括不存在的数字）的最佳拟合。因此在 (5, 0) 的情况下，它无法重构出 7，因为它知道存在最匹配且解释了所有像素的 5 和 0。同样，在 (8, 1) 的情况下，8 的环没有触发 0，因为它已被 8 所解释。因此，如果某个数字没有任何其他支撑，模型不会将一个像素分配给两个数字。

![图5](figure5.png)

### 6.2 MultiMNIST 结果

我们的 3 层 CapsNet 模型在 MultiMNIST 训练数据上从头训练，达到了比基线卷积模型更高的测试分类准确率。我们在高度重叠的数字对上达到了 5.0% 的分类错误率，与 Ba 等人 [2] 的顺序注意力模型在一个重叠度低得多的更简单任务上的结果相同（我们的两个数字边界框重叠 80%，而 Ba 等人 [2] 的 <4%）。在由测试集图像对组成的测试图像上，我们将两个最活跃的数字胶囊视为胶囊网络产生的分类结果。在重构过程中，我们每次选择一个数字，使用所选数字胶囊的活动向量重构所选数字的图像（我们知道该图像，因为我们用它生成了合成图像）。与我们的 MNIST 模型唯一的区别是，我们将学习率衰减步长的周期增加了 10 倍，因为训练数据集更大。

图 5 所示的重构表明，CapsNet 能够将图像分割成两个原始数字。由于这种分割不是在像素级别进行的，我们观察到模型能够正确处理重叠（一个像素在两个数字中均被激活），同时解释所有像素。每个数字的位置和风格都编码在 DigitCaps 中。解码器已学会在给定编码的情况下重构数字。它能够无视重叠重构数字这一事实表明，每个数字胶囊可以从 PrimaryCapsules 层接收的投票中获取风格和位置信息。

表 1 强调了在该任务上使用带有路由的胶囊的重要性。作为 CapsNet 准确率分类的基线，我们训练了一个具有两个卷积层和两个全连接层的卷积网络。第一层有 512 个 $9 \times 9$ 卷积核，步长 1。第二层有 256 个 $5 \times 5$ 卷积核，步长 1。每个卷积层之后是 $2 \times 2$ 池化层，步长 2。第三层是 1024D 的全连接层。所有三层均使用 ReLU 非线性。最后一层 10 个单元为全连接。我们使用 TensorFlow 默认的 Adam 优化器 [12] 训练最后一层输出的 sigmoid 交叉熵损失。该模型有 2456 万参数，是 CapsNet（1136 万参数）的 2 倍多。我们从较小的 CNN（32 和 64 个 $5 \times 5$ 卷积核，步长 1，以及 512D 全连接层）开始，逐步增加网络宽度，直到在 MultiMNIST 数据的 1 万子集上达到最佳测试准确率。我们还在 1 万验证集上搜索了正确的衰减步长。

我们逐个解码两个最活跃的 DigitCaps 胶囊，得到两幅图像。然后，将任何具有非零强度的像素分配给每个数字，得到每个数字的分割结果。

## 7 其他数据集

我们在 CIFAR-10 上测试了胶囊模型，使用 7 个模型的集成达到了 10.6% 的错误率，每个模型在 $24 \times 24$ 的图像块上使用 3 次路由迭代进行训练。每个模型具有与我们在 MNIST 上使用的简单模型相同的架构，区别在于使用了三个颜色通道和 64 种不同类型的主胶囊。我们还发现，为路由 softmax 引入一个"以上皆非"类别有所帮助，因为我们不希望最后十层胶囊解释图像中的所有内容。10.6% 的测试错误率大致相当于标准卷积网络首次应用于 CIFAR-10 时的水平 [19]。

胶囊的一个缺点（与生成模型共有的）是它倾向于解释图像中的所有内容，因此当它能够对杂乱背景进行建模时表现更好，而在动态路由中仅使用额外的"孤儿"类别时表现较差。在 CIFAR-10 中，背景变化太大，无法在合理大小的网络中建模，这有助于解释其较差的性能。

我们还在 smallNORB [14] 上测试了与 MNIST 完全相同的架构，达到了 2.7% 的测试错误率，与最先进水平 [4] 相当。smallNORB 数据集由 $96 \times 96$ 的立体灰度图像组成。我们将图像调整为 $48 \times 48$ ，在训练期间处理随机的 $32 \times 32$ 裁剪块。测试时传入中心 $32 \times 32$ 块。

我们还在 SVHN [15] 的小训练集（仅 73257 张图像）上训练了一个较小的网络。我们将第一卷积层的通道数减少到 64，主胶囊层减少到 16 个 6D 胶囊，最终胶囊层为 8D，在测试集上达到了 4.3% 的错误率。

## 8 讨论与相关工作

三十年来，语音识别的最先进技术使用隐马尔可夫模型（HMM）和高斯混合输出分布。这些模型在小计算机上易于学习，但具有一个最终致命的表示局限性：它们使用的 one-of-n 表示与使用分布式表示的循环神经网络相比呈指数级低效。要使 HMM 记住的关于已生成字符串的信息量翻倍，需要将隐藏节点数平方。对于循环网络，我们只需要将隐藏神经元数量翻倍。

现在卷积神经网络已成为物体识别的主流方法，我们有必要问问是否存在任何可能导致其衰落的指数级低效。一个好的候选是卷积网络在泛化到新视角方面的困难。处理平移的能力是内置的，但对于仿射变换的其他维度，我们必须在以下两者之间做出选择：在一个随维度数量指数增长的网格上复制特征检测器，或者以类似的指数方式增加标注训练集的大小。

胶囊 [10] 通过将像素强度转换为已识别片段的实例化参数向量，然后将变换矩阵应用于这些片段以预测更大片段的实例化参数，从而避免了这些指数级低效。学会编码部分与整体之间内在空间关系的变换矩阵构成了视角不变的知识，能够自动泛化到新视角。Hinton 等人 [10] 提出了变换自编码器来生成 PrimaryCapsule 层的实例化参数，其系统需要外部提供变换矩阵。我们提出了一个完整的系统，该系统也回答了"如何通过活跃的低层胶囊预测的姿态的一致性来识别更大更复杂的视觉实体"这一问题。

胶囊做了一个非常强的表示假设：在图像中的每个位置，最多存在一个胶囊所代表的实体类型的实例。这一假设受到称为"拥挤"（crowding）的感知现象 [17] 的启发，消除了绑定问题 [7]，并允许胶囊使用分布式表示（其活动向量）来编码该位置处该类型实体的实例化参数。这种分布式表示比通过激活高维网格上的一个点来编码实例化参数的效率呈指数级提高，并且有了正确的分布式表示，胶囊可以充分利用空间关系可以通过矩阵乘法建模这一事实。

胶囊使用随视角变化而变化的神经活动，而不是试图从活动中消除视角变化。这使它们相对于"归一化"方法（如空间变换网络 [11]）具有优势：它们可以同时处理不同物体或物体部件的多个不同仿射变换。

胶囊也非常适合处理分割问题——这是视觉中最困难的问题之一——因为实例化参数向量使它们能够使用路由协议机制，正如我们在本文中所展示的。动态路由过程的重要性还得到了视觉皮层中不变模式识别的生物合理模型的支持。Hinton [8] 提出了动态连接和基于规范物体框架来生成可用于物体识别的形状描述。Olshausen 等人 [16] 改进了 Hinton [8] 的动态连接，提出了一个生物合理的、位置和尺度不变的物体表示模型。

胶囊研究现在处于与本世纪初循环神经网络用于语音识别研究相似的阶段。有根本性的表示论据表明它是一种更好的方法，但可能需要更多的小洞见才能超越高度发达的技术。一个简单的胶囊系统已经在分割重叠数字方面展现出无与伦比的性能，这是一个早期迹象，表明胶囊是值得探索的方向。

**致谢。** 在众多为我们提供建设性意见的人中，我们特别感谢 Robert Gens、Eric Langlois、Vincent Vanhoucke、Chris Williams 以及审稿人的富有成果的评论和修正。

## 参考文献

[1] Martín Abadi, Ashish Agarwal, Paul Barham, Eugene Brevdo, Zhifeng Chen, Craig Citro, Greg S Corrado, Andy Davis, Jeffrey Dean, Matthieu Devin, et al. Tensorflow: Large-scale machine learning on heterogeneous distributed systems. *arXiv preprint arXiv:1603.04467*, 2016.

[2] Jimmy Ba, Volodymyr Mnih, and Koray Kavukcuoglu. Multiple object recognition with visual attention. *arXiv preprint arXiv:1412.7755*, 2014.

[3] Jia-Ren Chang and Yong-Sheng Chen. Batch-normalized maxout network in network. *arXiv preprint arXiv:1511.02583*, 2015.

[4] Dan C Cireșan, Ueli Meier, Jonathan Masci, Luca M Gambardella, and Jürgen Schmidhuber. High-performance neural networks for visual object classification. *arXiv preprint arXiv:1102.0183*, 2011.

[5] Ian J Goodfellow, Yaroslav Bulatov, Julian Ibarz, Sacha Arnoud, and Vinay Shet. Multi-digit number recognition from street view imagery using deep convolutional neural networks. *arXiv preprint arXiv:1312.6082*, 2013.

[6] Klaus Greff, Antti Rasmus, Mathias Berglund, Tele Hao, Harri Valpola, and Jürgen Schmidhuber. Tagger: Deep unsupervised perceptual grouping. In *Advances in Neural Information Processing Systems*, pages 4484–4492, 2016.

[7] Geoffrey E Hinton. Shape representation in parallel systems. In *International Joint Conference on Artificial Intelligence Vol 2*, 1981a.

[8] Geoffrey E Hinton. A parallel computation that assigns canonical object-based frames of reference. In *Proceedings of the 7th international joint conference on Artificial intelligence-Volume 2*, pages 683–685. Morgan Kaufmann Publishers Inc., 1981b.

[9] Geoffrey E Hinton, Zoubin Ghahramani, and Yee Whye Teh. Learning to parse images. In *Advances in neural information processing systems*, pages 463–469, 2000.

[10] Geoffrey E Hinton, Alex Krizhevsky, and Sida D Wang. Transforming auto-encoders. In *International Conference on Artificial Neural Networks*, pages 44–51. Springer, 2011.

[11] Max Jaderberg, Karen Simonyan, Andrew Zisserman, and Koray Kavukcuoglu. Spatial transformer networks. In *Advances in Neural Information Processing Systems*, pages 2017–2025, 2015.

[12] Diederik Kingma and Jimmy Ba. Adam: A method for stochastic optimization. *arXiv preprint arXiv:1412.6980*, 2014.

[13] Yann LeCun, Corinna Cortes, and Christopher JC Burges. The mnist database of handwritten digits, 1998.

[14] Yann LeCun, Fu Jie Huang, and Leon Bottou. Learning methods for generic object recognition with invariance to pose and lighting. In *Computer Vision and Pattern Recognition, 2004. CVPR 2004. Proceedings of the 2004 IEEE Computer Society Conference on*, volume 2, pages II–104. IEEE, 2004.

[15] Yuval Netzer, Tao Wang, Adam Coates, Alessandro Bissacco, Bo Wu, and Andrew Y Ng. Reading digits in natural images with unsupervised feature learning. In *NIPS workshop on deep learning and unsupervised feature learning*, volume 2011, page 5, 2011.

[16] Bruno A Olshausen, Charles H Anderson, and David C Van Essen. A neurobiological model of visual attention and invariant pattern recognition based on dynamic routing of information. *Journal of Neuroscience*, 13(11):4700–4719, 1993.

[17] Denis G Pelli, Melanie Palomares, and Najib J Majaj. Crowding is unlike ordinary masking: Distinguishing feature integration from detection. *Journal of vision*, 4(12):12–12, 2004.

[18] Li Wan, Matthew D Zeiler, Sixin Zhang, Yann LeCun, and Rob Fergus. Regularization of neural networks using dropconnect. In *Proceedings of the 30th International Conference on Machine Learning (ICML-13)*, pages 1058–1066, 2013.

[19] Matthew D Zeiler and Rob Fergus. Stochastic pooling for regularization of deep convolutional neural networks. *arXiv preprint arXiv:1301.3557*, 2013.

---

## 附录 A 使用多少次路由迭代？

为了实验验证路由算法的收敛性，我们绘制了每次路由迭代中路由 logits 的平均变化。图 A.1 显示了每次路由迭代后 $b_{ij}$ 的平均变化。实验上我们观察到，从训练开始到第 5 次迭代，路由的变化可以忽略不计。第 2 轮路由的平均变化在训练 500 个 epoch 后稳定到 0.007，而路由迭代 5 次时，logits 平均仅变化 $1 \times 10^{-5}$ 。

**图 A.1：每次路由迭代中每个路由 logit ( $b_{ij}$ ) 的平均变化。**在 MNIST 上训练 500 个 epoch 后，平均变化趋于稳定，如右图所示，随着路由迭代次数的增加，其在对数尺度上几乎呈线性下降。

(a) 训练过程中。
(b) 最终差值的对数尺度。

![图A.1](figureA1.png)

我们观察到，通常更多的路由迭代会增加网络容量并倾向于在训练数据集上过拟合。图 A.2 展示了在 CIFAR-10 上使用 1 次路由迭代与 3 次路由迭代训练时的胶囊训练损失对比。受图 A.2 和图 A.1 的启发，我们建议在所有实验中使用 3 次路由迭代。

**图 A.2：CapsuleNet 在 CIFAR-10 数据集上的训练损失。**每个训练步的 batch size 为 128。使用 3 次路由迭代的 CapsuleNet 优化损失更快，并最终收敛到更低的损失。

![图A.2](figureA2.png)
