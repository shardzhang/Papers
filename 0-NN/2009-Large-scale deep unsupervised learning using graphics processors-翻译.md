# Large-scale Deep Unsupervised Learning using Graphics Processors

> Rajat Raina | Anand Madhavan | Andrew Y. Ng
> 斯坦福大学计算机科学系，美国加利福尼亚州斯坦福 94305

无监督学习方法的前景在于它们利用大量未标注数据来学习具有数百万自由参数的复杂高度非线性模型的潜力。我们考虑两种著名的无监督学习模型——深度信念网络（DBN）和稀疏编码（sparse coding），它们近期已被应用于大量机器学习应用中（Hinton & Salakhutdinov, 2006; Raina et al., 2007）。不幸的是，这两种模型当前的算法对于大规模应用而言过于缓慢，迫使研究人员专注于较小规模的模型或使用较少的训练样本。

在本文中，我们提出大规模并行方法来帮助解决这些问题。我们认为现代图形处理器（graphics processors）的计算能力远超多核CPU，并有可能彻底改变深度无监督学习方法的适用性。我们开发了使用图形处理器大规模并行化无监督学习任务的一般原则。我们表明这些原则可以成功应用于扩展DBN和稀疏编码的学习算法。对于大模型，我们的DBN学习实现相比双核CPU实现最多可加速70倍。例如，我们能够将学习一个具有1亿自由参数的四层DBN所需的时间从数周减少到大约一天。对于稀疏编码，我们开发了一种简单、内在并行的算法，相比先前方法实现了5到15倍的加速。

发表在第26届国际机器学习会议论文集，蒙特利尔，加拿大，2009。版权归作者所有。

---

## 摘要

无监督学习方法的前景在于它们利用大量未标注数据来学习具有数百万自由参数的复杂高度非线性模型的潜力。我们考虑两种著名的无监督学习模型——深度信念网络（DBN）和稀疏编码（sparse coding），它们近期已被应用于大量机器学习应用中。不幸的是，这两种模型当前的算法对于大规模应用而言过于缓慢。在本文中，我们提出大规模并行方法来帮助解决这些问题。我们开发了使用图形处理器（GPU）大规模并行化无监督学习任务的一般原则，并表明这些原则可以成功应用于扩展DBN和稀疏编码的学习算法。我们的DBN学习实现相比双核CPU实现最多可加速70倍。对于稀疏编码，我们开发了一种简单、内在并行的算法，相比先前方法实现了5到15倍的加速。

---

## 1. 引言

我们考虑两种著名的无监督学习模型——深度信念网络（DBN）和稀疏编码（sparse coding）——它们可以学习输入的分层表示（Olshausen & Field, 1996; Hinton & Salakhutdinov, 2006）。随着过去十年中日益高效的学习算法的发明，这些模型已被应用于许多机器学习应用，包括计算机视觉、文本建模和协同过滤等。这些模型特别适合具有高维输入的问题，能够学习具有许多潜在变量（latent variables）或层的丰富模型。当应用于图像时，这些模型可以轻松拥有数千万个自由参数，并且理想情况下，我们希望使用数百万个未标注训练样本来充分覆盖输入空间。不幸的是，使用当前算法，在单CPU上使用传统实现进行参数学习可能需要数周时间。部分由于如此严峻的计算需求，文献中考虑的DBN和稀疏编码的典型应用通常包含少得多的自由参数（例如，见表1），或者仅使用可用输入样本的一小部分进行训练。

在我们看来，如果目标是部署更好的机器学习应用，则学习大模型的困难是一个严重的限制。举一个具体的案例研究，在自然语言处理中两个被广泛研究的统计学习任务——语言建模（language modeling）和拼写纠正（spelling correction）——已经表明，简单的经典模型可以胜过更新、更复杂的模型，仅仅因为简单模型可以使用数量级更多的输入数据来可行地学习（Banko & Brill, 2001; Brants et al., 2007）。类似地，在我们看来，扩展现有的DBN和稀疏编码模型以使用更多参数或更多训练数据，可能会产生非常显著的性能提升。例如，已经表明，当模型很大时，稀疏编码会表现出一种称为"end-stopping"的定性不同且高度选择性的行为，但在模型不大时则不会（Lee et al., 2006）。

近期已有大量关于扩展DBN和稀疏编码算法的工作，有时整篇研究论文专门致力于为这些模型中的每一个设计巧妙的方法（Hinton et al., 2006; Bengio et al., 2006; Murray & Kreutz-Delgado, 2006; Lee et al., 2006; Kavukcuoglu et al., 2008）。与此同时，单CPU的原始时钟速度已开始触及硬件功率极限，处理能力增长越来越多地通过集成多个CPU核心而非加速单个核心来实现（Gelsinger, 2001; Frank, 2002）。近期工作表明，几种流行的学习算法如逻辑回归（logistic regression）、线性SVM等可以通过让每个核心对一部分输入样本执行所需计算然后将结果集中合并，在多核架构上轻松并行实现（Dean & Ghemawat, 2004; Chu et al., 2006）。然而，DBN和稀疏编码的标准算法难以用这种"数据并行"方案并行化，因为它们涉及迭代的随机参数更新（stochastic parameter updates），其中任何更新都依赖于之前的更新。这使得在粗粒度的数据并行层面上（例如，通过并行计算更新并集中求和）难以大规模并行化这些更新，而不会丢失更新的关键随机特性。看来细粒度并行性可能是成功并行化这些任务所必需的。

在本文中，我们利用现代图形处理器（GPU）的能力来可行地学习大型DBN和稀疏编码模型。当前台式机附带的典型图形卡包含超过一百个处理核心，其峰值内存带宽比现代CPU高出数倍。该硬件可以同时处理数千个线程，并且能够以非常小的开销在可用核心上调度这些线程。这种细粒度并行性使得GPU对于难以在其他分布式架构上并行化的通用计算越来越有吸引力。

当然，这存在一个权衡——这种并行性是通过将更多的晶体管用于数据处理而非缓存和控制流（如常规CPU核心中那样）而获得的。这对可以有效实现的指令类型和内存访问类型施加了限制。因此，成功将GPU应用于机器学习任务的主要挑战在于尽可能重新设计学习算法以满足这些约束。虽然对图形处理器架构的全面介绍超出了本文的范围，但我们现在回顾成功使用GPU进行计算的基本概念。

## 2. 使用图形处理器进行计算

我们使用Nvidia的CUDA编程模型来说明GPU计算的原则（Harris, 2008）。图1显示了典型Nvidia GPU的简化示意图。GPU硬件提供了两个层次的并行性：有多个多处理器（MPs），每个多处理器包含多个运行实际计算的流处理器（SPs）。计算被组织成称为"块"（blocks）的线程组，每个块被调度到一个多处理器上运行，在一个多处理器内部，每个线程被调度到一个流处理器上运行。

一个块内的所有线程（因此在同一个多处理器上执行）可以共享访问一小块（16 KB）非常快速的"共享内存"（shared memory），并且它们可以在执行的不同点彼此同步。所有线程还可以访问大得多的GPU全局"全局内存"（global memory）（目前最多4 GB），该内存比共享内存慢，但针对某些类型的同步访问模式（称为"合并"（coalesced）访问）进行了优化。

<img src="...">
图1. Nvidia GeForce GTX 280 图形卡的简化示意图，总共 240 个核心（30 个多处理器，每个有 8 个流处理器）。

简而言之，如果块中的线程按顺序访问内存（即第k个线程访问内存中第k个连续位置），则称这些线程的内存访问请求是合并的。[1]当内存访问合并时，硬件可以针对所有流处理器并行执行它们，有效访问速度（在流处理器和全局内存之间）比CPU和RAM之间的访问速度快数倍。

由于GPU计算和GPU内部内存访问本身是高度并行的，在许多算法中，主要瓶颈出现在RAM和GPU全局内存之间传输数据。例如，使用我们的GPU配置（以及供应商提供的线性代数包）将两个1000x1000矩阵相乘所需的总时间约为20毫秒，但实际计算仅占该时间的0.5%，其余时间用于进出全局内存的传输。部分解决方案是仅以大批量方式执行内存传输，将多次计算分组进行。在我们的例子中，如果我们进行25个不同的矩阵乘法，并且能够以大的块执行内存传输（通过将所有输入一起传输，并将所有输出一起传输），则多达25%的总时间用于计算。因此，高效利用GPU的并行性需要仔细考虑应用中的数据流。

[1]为简洁起见，我们忽略了某些在实践中易于遵守的其他技术条件。我们还省略了对另外两种内存——常量内存和纹理内存——的讨论，这两种内存针对我们应用中未使用的其他特定访问模式进行了优化。

## 3. 预备知识

我们现在介绍本文中考虑的无监督学习问题，并分析将GPU应用于这些问题时面临的具体问题。我们考虑一个无监督学习任务，其中给定一个大型未标注数据集 $\{x^{(1)}, x^{(2)}, \ldots, x^{(m)}\}$，每个输入 $x^{(i)} \in \mathbb{R}^k$。目标是学习输入 $x$ 的模型，然后将该模型应用于特定的机器学习任务。例如，每个未标注输入 $x^{(i)} \in \mathbb{R}^{900}$ 可能表示一个手写字符的30x30像素图像（表示为像素强度向量）。我们可能想要学习一个针对输入的这个复杂900维空间的模型，然后使用该模型仅用非常少的标注数据来分类新的手写字符。

### 3.1. 深度信念网络

DBN是多层神经网络模型，学习其输入数据的分层表示。Hinton等人（2006）提出了一种用于学习DBN的无监督算法，其中DBN从输入数据开始逐层贪婪地构建。每一层使用一种称为受限玻尔兹曼机（RBM）的概率模型来学习。简而言之，RBM包含一组随机隐藏单元 $h$，它们在一个无向模型中与一组随机可见单元 $x$ 完全连接。假设二元值单元，RBM定义以下联合分布：

$$P(x, h) \propto \exp\left( \sum_{i,j} x_i w_{ij} h_j + \sum_i c_i x_i + \sum_j b_j h_j \right)$$

其中权重 $w$ 以及偏置 $b$ 和 $c$ 是要调整的参数。条件分布可以解析计算：

$$P(h_j | x) = \text{sigmoid}(b_j + \sum_i w_{ij} x_i) \qquad (1)$$
$$P(x_i | h) = \text{sigmoid}(c_i + \sum_j w_{ij} h_j) \qquad (2)$$

RBM的最大似然参数学习可以通过对比散度（contrastive divergence）更新高效近似（Hinton, 2002），其中我们从未标注示例作为可见单元开始，使用Gibbs采样（方程1-2）交替采样隐藏单元 $h$ 和可见单元 $x$，并更新参数如下：

$$w_{ij} := w_{ij} + \eta \left( \langle x_i h_j \rangle_{\text{data}} - \langle x_i h_j \rangle_{\text{sample}} \right) \qquad (3)$$
$$c_i := c_i + \eta \left( \langle x_i \rangle_{\text{data}} - \langle x_i \rangle_{\text{sample}} \right) \qquad (4)$$
$$b_j := b_j + \eta \left( \langle h_j \rangle_{\text{data}} - \langle h_j \rangle_{\text{sample}} \right) \qquad (5)$$

其中 $\eta$ 是学习率，$\langle \cdot \rangle_{\text{data}}$ 表示可见单元与输入示例绑定时计算得到的期望，$\langle \cdot \rangle_{\text{sample}}$ 表示经过 $T \geq 1$ 次Gibbs采样迭代后的期望。由于每次更新都需要一次Gibbs采样操作，并且必须对大量未标注示例重复应用更新以达到收敛，无监督学习参数在现代CPU上可能需要数天才能完成。

### 3.2. 稀疏编码

稀疏编码（Sparse coding）是一种构建输入数据简洁表示的算法（Olshausen & Field, 1996）。使用我们之前的例子，如果每个输入 $x^{(i)} \in \mathbb{R}^{900}$ 表示一个手写字符图像，稀疏编码尝试学习每个手写字符仅由少数构建块组成，如笔画（而不是900个任意强度值）。这样的高层次表示随后可以应用于分类任务，即使在标注数据有限的情况下也能取得良好结果（Raina et al., 2007; Bradley & Bagnell, 2008）。

具体来说，给定输入 $x \in \mathbb{R}^k$，稀疏编码试图寻找基向量 $b = \{b_1, b_2, \ldots, b_n\}$，$b_j \in \mathbb{R}^k$，使得每个输入 $x$ 可以表示为少数基向量的线性组合：$x \approx \sum_j a_j b_j$，其中 $a_j \in \mathbb{R}$ 表示基 $b_j$ 的激活值（activation），并且大多数 $a_j$ 值为零（即向量 $a$ 是稀疏的）。基向量通过求解以下优化问题得到（Lee et al., 2006）：

$$\begin{aligned}
\underset{b, a}{\text{minimize}} \quad & \frac{1}{2} \sum_i \left\| x^{(i)} - \sum_j a_j^{(i)} b_j \right\|^2 + \beta \sum_{i,j} |a_j^{(i)}| \\
\text{s.t.} \quad & \|b_j\| \leq 1, \quad \forall j \in \{1, \ldots, n\}
\end{aligned}$$

其中目标函数的第一项鼓励良好的重构（$x^{(i)} \approx \sum_j b_j a_j^{(i)}$），第二项通过惩罚非零激活值（non-zero activations）来鼓励稀疏性（Tibshirani, 1996）。该优化问题在变量 $b$ 和 $a$ 上不是联合凸的，但在保持另一个固定时，对其中任何一个变量都是凸的。这提示了一种交替最小化算法，包含两个步骤：首先，固定 $b$，我们对 $a$ 进行优化，得到一个L1正则化最小二乘问题，可以使用定制设计的求解器求解（Efron et al., 2004; Lee et al., 2006; Andrew & Gao, 2007）。然后，固定 $a$，我们使用凸优化技术对 $b$ 进行优化（Lee et al., 2006）。对于具有高维输入和大量基向量的问题，第一步尤其耗时，因为它涉及不可微的目标函数，整个学习算法可能需要数天时间。

## 4. 用于无监督学习的GPU

上述两种算法都重复执行以下计算：选取少量未标注示例，计算一个更新（通过对比散度或求解一个凸优化问题），并将其应用于参数。为了成功将GPU应用于此类无监督学习算法，我们需要满足两个主要要求。首先，需要最小化RAM和GPU全局内存之间的内存传输，或将其分组为大块。对于机器学习应用，我们可以通过在训练期间将所有参数永久存储在GPU全局内存中来实现这一点。未标注示例通常无法全部存储在全局内存中，但它们应尽可能以大的块偶发地传输到全局内存中。当参数和未标注示例都在GPU全局内存中时，可以在没有任何内存传输操作的情况下计算更新，任何中间计算也存储在全局内存中。

第二个要求是，学习更新的实现应适配块和线程的两层层次结构，使得可以在可能的地方使用共享内存，并且全局内存访问可以被合并。通常，块可以利用数据并行性（例如，每个块可以处理一个单独的输入示例），而线程可以利用更细粒度的并行性，因为它们可以访问非常快速的共享内存并且可以同步（例如，每个线程可以处理分配给该块的输入示例的单个坐标）。此外，图形硬件可以通过在该时间段内调度一个准备运行的块来隐藏等待全局内存访问的块的内存延迟。为了充分利用这种延迟隐藏，使用大量独立执行的块是有利的。在某些情况下，如第6节中讨论的稀疏编码，我们可以完全重新设计更新使其内在并行且需要更少的线程间同步。

因此，我们得出以下将GPU应用于无监督学习任务的模板算法：

**算法1 GPU上的并行无监督学习**
1. 在全局内存中初始化参数。
2. 当收敛条件未满足时：
   a. 定期将大量未标注示例传输到全局内存中。
   b. 每次选取少数未标注示例，使用GPU的两层并行性（块和线程）并行计算更新。
3. 从全局内存中转移学习到的参数。

## 5. 学习大型深度信念网络

我们将算法1应用于使用方程(3-5)中的对比散度更新来学习大型DBN。所有DBN层的参数 $w$、$c$ 和 $b$ 在训练期间永久保存在全局内存中。更新需要重复使用方程(1-2)中的分布进行Gibbs采样。这些分布可以使用矩阵符号重写：

$$P(h | x) = \text{vectorSigmoid}(b + w^T x)$$
$$P(x | h) = \text{vectorSigmoid}(c + w h)$$

其中 $\text{vectorSigmoid}(\cdot)$ 表示逐元素的sigmoid函数，$x$、$h$ 是包含分别对应每个可见单元和隐藏单元的元素的向量。为了进一步提高效率，上述计算可以对多个示例批量进行。矩阵操作可以使用针对GPU优化的线性代数包并行执行，sigmoid计算和采样可以通过一个简单的并行化方案完成，其中每个块处理一个示例，块中的每个线程处理该示例的单个元素。最后，一旦生成了样本，更新可以再次使用线性代数包并行应用：例如，$w := w + \eta \left( \langle x^T h \rangle_{\text{data}} - \langle x^T h \rangle_{\text{sample}} \right)$。

我们将我们的方法扩展到学习具有"重叠块"（overlapping patches）的深度信念网络（图2）。这个模型最容易理解的方式是将隐藏单元和可见单元排列在二维阵列中（例如，当输入是图像且每个可见单元是一个像素时）。输入图像由等间距、等大小的块（或感受野）平铺，每个块完全连接到一组独特的隐藏单元。该模型中没有权重共享，每个连接由一个自由参数参数化。由于块的重叠，模型中的所有参数相互依赖，使得学习变得困难。然而，对于该模型，Gibbs采样仍然可以并行执行：每个可见单元依赖于许多不同位置的隐藏单元，但 $x|h$ 的采样操作可以使用仅合并的全局内存访问来实现（实现细节省略）。这些重叠块RBM可以相互堆叠，使得第二层RBM包含与第一层隐藏单元局部连接的隐藏单元，依此类推。由此产生的深度网络具有非常大量的单元，但只有稀疏的局部连接，这使得即使是具有超过1亿参数的模型也能可行地学习。

<img src="...">
图2. 重叠块模型的示意图。显示了输入图像中的两个块A和B，每个块连接到一组不同的隐藏单元。连接由它们自己的参数集 $w_A, b_A, c_A$ 和 $w_B, b_B, c_B$ 参数化。

**实验结果：** 我们将基于GPU的算法与基于CPU的方法进行比较，使用以下多核硬件：

- **GPU：** Nvidia GeForce GTX 280显卡，1GB内存。双核CPU @ 3.16GHz。报告的结果显示了总运行时间（包括所有计算、内存传输等）。
- **单CPU：** 单核 @ 3.16GHz。
- **双核CPU：** 两个核心，每个 @ 3.16GHz。（与GPU结果相同的机器）。

基于CPU的方法使用两个高度优化的多线程线性代数包实现：ATLAS BLAS（Whaley et al., 2001）和Goto BLAS（Goto & Van De Geijn, 2008）。与先前结果一致，我们发现Goto BLAS更快（Bengio, 2007），因此我们报告使用它的CPU结果。作为输入，我们使用了一个大型自然图像数据集（van Hateren & van der Schaaff, 1997），并通过随机提取所需大小的方形图像块来获得输入示例。遵循先前工作，我们使用了高斯可见单元和二元隐藏单元，并通过向目标添加额外惩罚项来训练稀疏RBM（Lee et al., 2007）——然而，这些修改不会显著影响运行时间结果。对于学习，我们使用大小为192个示例的小批量（mini-batch）执行单步对比散度更新。

表2显示了处理100万示例用于不同大小RBM（由可见单元数 \\times 隐藏单元数表示）的运行时间。GPU方法比最快的基于CPU的结果快12到72倍。对于大型RBM，加速比最高，在这些情况下计算涉及大矩阵，可以通过使用大量并发块（这允许图形硬件更好地隐藏内存延迟）更高效地并行化。表2中最大的模型有4500万个参数，我们的GPU方法可以在大约29分钟内使用100万个示例更新这些参数。相比之下，我们的多核CPU每100万示例需要超过一天。由于我们理想情况下希望使用数千万个训练样本来学习如此大的模型，CPU方法对于这类任务是不可行的。

表3显示了两个"重叠块"模型的类似运行时间比较（见表注了解详情）。GPU方法比双核CPU快约10倍。这个加速比略低于在全连接RBM中观察到的加速比（表2），因为重叠块模型中的Gibbs采样需要许多涉及小矩阵的操作（每个块一个权重矩阵），而不是仅涉及大矩阵的几个操作。使用重叠块模型，我们可以学习一个具有9600万个参数的四层DBN，在输入层和四个相继的隐藏层中分别有25600、82944、8192、4608和1024个单元。这类模型至少比先前发表的DBN工作大一个数量级。

最后，我们注意到重叠块模型可以修改为在所有块中共享参数，例如，使得图2中 $w_A = w_B$。如果重叠块以相距一个像素的方式平铺，则该模型与卷积RBM模型相同（Desjardins & Bengio, 2008; Lee et al., 2009）。该模型中的对比散度学习可以通过使用卷积执行Gibbs采样操作 $h|x$ 来实现。对于小到中等大小的滤波器（块）尺寸，空间卷积可以通过使每个块将一个滤波器读入共享内存，然后逐列将输入图像读入共享内存，最后聚合受该滤波器和该输入图像列影响的输出元素，从而使用GPU非常高效地实现。可以证明，通过这样排序操作，我们只使用快速的共享内存访问和合并的全局内存访问。[2]例如，在计算32个128x128图像与32个16x16滤波器的卷积时，我们的GPU空间卷积实现（包括将图像/滤波器传输到GPU内存的时间）比用C实现的空间卷积或基于FFT的Matlab卷积快100倍以上。

[2]对于更大的滤波器尺寸，基于FFT的卷积通常更好，可以使用GPU FFT包。

## 6. 并行稀疏编码

我们现在考虑第3.2节中讨论的稀疏编码优化问题。遵循算法1中的模板，我们将基参数 $b$ 永久保持在全局内存中，并定期以大批量将输入示例传输到GPU全局内存中。遵循交替最小化方法，每次更新本身包含两个步骤：更新的第一个较简单的部分是在给定固定 $a$ 的情况下对 $b$ 进行优化：

$$\begin{aligned}
\underset{b}{\text{minimize}} \quad & \sum_i \left\| x^{(i)} - \sum_j a_j^{(i)} b_j \right\|^2 \\
\text{s.t.} \quad & \|b_j\| \leq 1, \quad \forall j
\end{aligned}$$

我们使用投影梯度下降（projected gradient descent）解决这个问题，其中我们遵循二次目标函数的梯度，并在每一步投影到可行集。[3]该方保证收敛到最优 $b$，并且可以使用GPU线性代数包直接实现。

[3]投影操作特别简单：对于每个基向量 $b_j$，如果 $\|b_j\| > 1$，则将 $b_j$ 重新缩放为单位范数，否则保持 $b_j$ 不变。

更新的另一部分涉及在给定固定 $b$ 的情况下对 $a$ 进行优化。由于每个示例 $x^{(i)}$ 的激活值 $a^{(i)}$ 独立于其他示例的激活值，因此只需考虑单个输入示例 $x$ 的以下标准L1正则化最小二乘问题：

$$\underset{a}{\text{minimize}} \quad \frac{1}{2} \left\| x - \sum_j a_j b_j \right\|^2 + \beta \sum_j |a_j| \qquad (6)$$

由于第二项的存在，目标函数不可微。这个问题因其鲁棒的特征选择性质（Tibshirani, 1996; Ng, 2004）近年来受到广泛关注，并且已经设计了定制的算法来解决它（Efron et al., 2004; Lee et al., 2006; Andrew & Gao, 2007）。其中一些算法使用稀疏线性代数操作来实现效率。相反，我们提出了一种非常不同的算法，该算法本质上是并行的，因此更有效地利用GPU硬件。

### 6.1. 并行L1正则化最小二乘

我们的算法基于以下观察：在方程(6)的优化问题中，如果我们只改变其中一个激活值 $a_j$，同时保持其他激活值固定，则最优值 $a_j^*$ 可以轻松计算（Friedman et al., 2007）。令 $B$ 为以 $b_j$ 作为其第j列的矩阵，且 $r_j = b_j^T b_j$：

$$
a_j^* =
\begin{cases}
0 & \text{if } |g_j - r_j a_j| \leq \beta \\
(-g_j + r_j a_j + \beta) / r_j & \text{if } g_j - r_j a_j > \beta \\
(-g_j + r_j a_j - \beta) / r_j & \text{if } g_j - r_j a_j < -\beta
\end{cases}
$$

其中 $g = \nabla_a \frac{1}{2} \| x - \sum_j a_j b_j \|^2 = B^T B a - B^T x$。

更新可以通过让线程j仅计算一个坐标 $a_j^*$ 来高效并行执行。[4]此外，由于我们通常将几个示例一起批处理，我们可以预先并行计算矩阵 $B^T B$、向量 $B^T x$ 和向量 $r$，将结果存储在全局内存中，并仅执行高效访问来计算所有 $a_j^*$ 值。

[4]要理解原因，请注意计算 $a_j^*$ 需要线程j计算 $g_j - r_j a_j = \sum_t (B^T B)_{tj} a_t - (B^T x)_j - r_j a_j$。考虑线程j访问的元素：(i) $(B^T B)_{tj}$：如果 $B^T B$ 按行主序存储，访问可以合并。(ii) 通过将 $a$ 维护在共享内存中，所有线程可以同时访问 $a$ 的相同元素，以及访问每个线程不同的元素 $a_j$。（为有兴趣的读者补充，这避免了共享内存中的"bank冲突"。详情见CUDA参考手册。）(iii) $(B^T x)_j$ 和 $r_j$：可以合并，因为线程j访问第j个位置。

因此，我们提出以下迭代算法：在每次迭代中，从当前激活值 $a = \hat{a}$ 开始，我们如上所述并行计算所有最优坐标值 $a_j^*$。然后，我们在向量 $d = a^* - \hat{a}$ 的方向上执行线搜索（line search）。线搜索包括寻找步长 $t > 0$，使得目标函数在点 $a = \hat{a} + t d$ 处的值低于在 $a = \hat{a}$ 处的值。这个线搜索，包括函数评估，可以并行运行。[5]然后我们将 $a$ 移动到新点 $a = \hat{a} + t d$，并继续迭代。当目标值的减少量小于前一个目标值的 $10^{-6}$ 时，我们声明收敛。

[5]细节：通过将 $a = \hat{a} + t d$ 代入原始目标函数，线搜索简化为最小化形式为 $f(t) = \alpha_2 t^2 + \alpha_1 t + \alpha_0 + \beta \| \hat{a} + t d \|_1$ 的一维函数，其中 $\alpha_2, \alpha_1, \alpha_0$ 可以并行计算。对于 $f(t)$ 的一维线搜索，我们只需尝试一组固定的正步长，并选择能最大程度降低目标函数值的最大步长。

由于方向 $d$ 是沿坐标轴下降方向（descent directions）的非负线性组合：$d_j = a_j^* - \hat{a}_j$，$d$ 本身必须是目标函数的下降方向。因此，在每次迭代中，总能找到一个步长 $t > 0$ 来减小目标函数的值，并且整个算法保证收敛到最优解。

该算法通过让每个线程只计算解的一个坐标来使用细粒度并行性。这种高度多线程的执行特别适合图形处理器，因为硬件能够通过调度其他未阻塞在内存访问上的线程来隐藏（阻塞在内存访问上的线程的）内存延迟，并导致可用核心的高利用率。

**实验结果：** 我们再次将我们的方法与多核CPU基线（Lee et al., 2006）进行比较。我们使用了Lee等人提供的优化Matlab代码。对于CPU多核结果，我们在启用多线程的情况下执行相同的Matlab代码。

表4显示了应用稀疏编码基更新（包括基和激活优化）用于 $m = 5000$ 个示例、小批量为1000个示例时的运行时间。每个示例 $x \in \mathbb{R}^{1024}$ 通过随机采样的32x32像素自然图像块获得。我们使用了 $n = 1024$ 个基向量，随机初始化。Lee等人的方法中稀疏编码的大部分时间由激活学习步骤占用，特别是当许多激活值在最优解处非零时。通过有效并行化这一步，我们的GPU方法比双核实现快最多15倍。

## 7. 讨论

图形处理器能够利用比当前多核架构或分布式集群更细粒度的并行性。它们被设计为在任何时刻维护数千个活动线程，并以非常低的调度开销将线程调度到数百个核心上。Map-reduce框架（Dean & Ghemawat, 2004）已成功应用于并行化一类机器学习算法（Chu et al., 2006）。然而，该方法完全依赖于数据并行性——每个核心可以独立处理一组不同的输入示例——没有进一步的工作细分。相比之下，GPU提供的两层并行性强大多了：顶层的GPU块已经可以利用数据并行性，而GPU线程可以进一步细分每个块中的工作，通常只需处理输入示例的单个元素。

GPU已被应用于机器学习中的某些问题，包括SVM（Catanzaro et al., 2008）和卷积网络中的监督学习（Chellapilla et al., 2006）。为了继续这一工作路线，并鼓励深度信念网络和稀疏编码的进一步应用，我们将公开提供我们的源代码。

**致谢：** 我们衷心感谢Roger Grosse、Honglak Lee和匿名审稿人的有益评论，以及Ethan Dreyfuss、Ian Goodfellow和Quoc Le在组装硬件方面的帮助。这项工作得到了DARPA迁移学习项目合同号FA8750-05-2-0249以及海军研究办公室MURI N000140710747的资助。

## 参考文献

[1] Andrew, G., & Gao, J. (2007). Scalable training of L1-regularized log-linear models. *International Conference on Machine Learning* (pp. 33–40).

[2] Banko, M., & Brill, E. (2001). Scaling to very very large corpora for natural language disambiguation. *Annual Meeting of the Association for Computational Linguistics* (pp. 26–33).

[3] Bengio, Y. (2007). Speeding up stochastic gradient descent. *Neural Information Processing Systems Workshop on Efficient Machine Learning*.

[4] Bengio, Y., Lamblin, P., Popovici, D., & Larochelle, H. (2006). Greedy layer-wise training of deep networks. *Neural Information Processing Systems* (pp. 153–160).

[5] Bradley, D., & Bagnell, J. A. (2008). Differentiable sparse coding. *Neural Information Processing Systems* (pp. 113–120).

[6] Brants, T., Popat, A. C., Xu, P., Och, F. J., & Dean, J. (2007). Large language models in machine translation. *Conference on Empirical Methods in Natural Language Processing (EMNLP-CoNLL)*.

[7] Catanzaro, B. C., Sundaram, N., & Keutzer, K. (2008). Fast support vector machine training and classification on graphics processors. *International Conference on Machine Learning* (pp. 104–111).

[8] Chellapilla, K., Puri, S., & Simard, P. (2006). High performance convolutional neural networks for document processing. *International Workshop on Frontiers in Handwriting Recognition*.

[9] Chu, C. T., Kim, S. K., Lin, Y. A., Yu, Y., Bradski, G. R., Ng, A. Y., & Olukotun, K. (2006). Map-reduce for machine learning on multicore. *Neural Information Processing Systems* (pp. 281–288).

[10] Dean, J., & Ghemawat, S. (2004). Mapreduce: Simplified data processing on large clusters. *Operating System Design and Implementation* (pp. 137–150).

[11] Desjardins, G., & Bengio, Y. (2008). Empirical evaluation of convolutional RBMs for vision. *Tech Report*.

[12] Efron, B., Hastie, T., Johnstone, I., & Tibshirani, R. (2004). Least angle regression. *Ann. Stat.*, 32, 407.

[13] Frank, D. (2002). Power-constrained CMOS scaling limits. *IBM Jour. of Res. and Devel.*, 46, 235–244.

[14] Friedman, J., Hastie, T., Hfling, H., & Tibshirani, R. (2007). Pathwise coordinate optimization. *Ann. App. Stat.*, 2, 302–332.

[15] Gelsinger, P. (2001). Microprocessors for the new millennium: Challenges, opportunities and new frontiers. *ISSCC Tech. Digest*, 22–25.

[16] Goto, K., & Van De Geijn, R. (2008). High-performance implementation of the level-3 BLAS. *ACM Trans. Math. Softw.*, 35, 1–14.

[17] Harris, M. (2008). Many-core GPU computing with NVIDIA CUDA. *Int. Conf. Supercomputing* (p. 1).

[18] Hinton, G. E. (2002). Training products of experts by minimizing contrastive divergence. *Neural Computation*, 14, 1771–1800.

[19] Hinton, G. E., Osindero, S., & Teh, Y.-W. (2006). A fast learning algorithm for deep belief nets. *Neural Computation*, 18, 1527–1554.

[20] Hinton, G. E., & Salakhutdinov, R. R. (2006). Reducing the dimensionality of data with neural networks. *Science*, 313, 504–507.

[21] Kavukcuoglu, K., Ranzato, M., & LeCun, Y. (2008). Fast inference in sparse coding algorithms with applications to object recognition. *NYU Tech Report*.

[22] Lee, H., Battle, A., Raina, R., & Ng, A. Y. (2006). Efficient sparse coding algorithms. *Neural Information Processing Systems* (pp. 801–808).

[23] Lee, H., Chaitanya, E., & Ng, A. Y. (2007). Sparse deep belief net model for visual area V2. *Neural Information Processing Systems* (pp. 873–880).

[24] Lee, H., Grosse, R., Ranganath, R., & Ng, A. Y. (2009). Convolutional deep belief networks for scalable unsupervised learning of hierarchical representations. *International Conference on Machine Learning (to appear)*.

[25] Murray, J. F., & Kreutz-Delgado, K. (2006). Learning sparse overcomplete codes for images. *J. VLSI Signal Processing Systems*, 45, 97–110.

[26] Ng, A. Y. (2004). Feature selection, L1 vs. L2 regularization, and rotational invariance. *International Conference on Machine Learning* (pp. 78–85).

[27] Olshausen, B. A., & Field, D. J. (1996). Emergence of simple-cell receptive field properties by learning a sparse code for natural images. *Nature*, 381, 607–609.

[28] Raina, R., Battle, A., Lee, H., Packer, B., & Ng, A. Y. (2007). Self-taught learning: Transfer learning from unlabeled data. *International Conference on Machine Learning* (pp. 759–766).

[29] Ranzato, M. A., & Szummer, M. (2008). Semi-supervised learning of compact document representations with deep networks. *International Conference on Machine Learning* (pp. 792–799).

[30] Salakhutdinov, R., & Hinton, G. (2007). Semantic Hashing. *SIGIR Workshop on Information Retrieval and Applications of Graphical Models*.

[31] Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. *J. R. Stat. Soc. B.*, 58, 267–288.

[32] van Hateren, J. H., & van der Schaaff, A. (1997). Independent component filters of natural images compared with simple cells in primary visual cortex. *Royal Soc. Lond. B*, 265, 359–366.

[33] Whaley, R. C., Petitet, A., & Dongarra, J. J. (2001). Automated empirical optimization of software and the ATLAS project. *Parallel Computing*, 27, 3–35.
