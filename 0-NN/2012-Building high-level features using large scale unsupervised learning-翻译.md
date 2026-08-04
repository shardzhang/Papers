# Building High-level Features Using Large Scale Unsupervised Learning

> Quoc V. Le, Marc'Aurelio Ranzato, Rajat Monga, Matthieu Devin, Kai Chen, Greg S. Corrado, Jeff Dean, Andrew Y. Ng | Stanford University, Google

本文研究仅从无标注数据中构建高层、类别特定的特征检测器的问题。例如，是否可能仅使用无标注图像学习一个人脸检测器？为了回答这个问题，我们训练了一个9层的局部连接稀疏自编码器，包含池化(pooling)和局部对比度归一化，使用大规模图像数据集（模型有10亿连接，数据集包含从互联网下载的1000万张200x200像素图像）。我们使用模型并行化和异步SGD在1000台机器（16000核）的集群上训练了三天。与普遍直觉相反，我们的实验结果表明，无需将图像标注为包含人脸或不包含人脸，就可以训练出一个人脸检测器。对照实验表明，该特征检测器不仅对平移具有鲁棒性，对缩放和平面外旋转也具有鲁棒性。我们还发现同一网络对其他高层概念（如猫脸和人体）也敏感。从这些学习到的特征出发，我们训练网络在ImageNet的20000个物体类别识别中达到了15.8%的准确率，相比之前的最先进水平提升了70%。

Key findings:
- 仅使用无标注YouTube视频帧即可训练出人脸检测器，无需任何标注数据
- 9层局部连接稀疏自编码器包含10亿参数，在1000台机器上训练3天
- 学习到的特征检测器对平移、缩放和平面外旋转均具有鲁棒性
- 网络同时学习到猫脸和人体等高层概念
- 在ImageNet 20000类分类任务上达到15.8%准确率，相对提升70%

---

## 摘要

我们考虑仅从无标注数据中构建高层、类别特定的特征检测器的问题。例如，是否可能仅使用无标注图像学习一个人脸检测器？为了回答这个问题，我们训练了一个9层的局部连接稀疏自编码器，包含池化(pooling)和局部对比度归一化，使用大规模图像数据集（模型有10亿连接，数据集包含从互联网下载的1000万张200x200像素图像）。我们使用模型并行化和异步SGD在1000台机器（16000核）的集群上训练了三天。与普遍直觉相反，我们的实验结果表明，无需将图像标注为包含人脸或不包含人脸，就可以训练出一个人脸检测器。对照实验表明，该特征检测器不仅对平移具有鲁棒性，对缩放和平面外旋转也具有鲁棒性。我们还发现同一网络对其他高层概念（如猫脸和人体）也敏感。从这些学习到的特征出发，我们训练网络在ImageNet的20000个物体类别识别中达到了15.8%的准确率，相比之前的最先进水平提升了70%。

## 1. 引言

本文工作的重点是从无标注图像中构建高层、类别特定的特征检测器。例如，我们希望理解是否可能仅从无标注图像构建一个人脸检测器。这种方法受到神经科学猜想的启发，即人脑中存在高度类别特定的神经元，通常非正式地被称为"祖母神经元(grandmother neurons)"。关于大脑中神经元类别特异性的程度仍是一个活跃的研究领域，但当前的实验证据表明，颞叶皮层中的某些神经元可能对物体类别（如人脸或手）高度选择（Desimone et al., 1984）[1]，甚至可能对特定人也高度选择（Quiroga et al., 2005）[2]。

当代计算机视觉方法通常强调标注数据在获得这些类别特定特征检测器中的作用。例如，为了构建一个人脸检测器，需要大量标注为包含人脸的图像，通常还要包含人脸周围的边界框。对大规模标注数据集的需求给标注数据稀缺的问题带来了重大挑战。尽管利用廉价无标注数据的方法通常更受青睐，但尚未证明它们在构建高层特征方面效果良好。

本研究探讨了仅从无标注数据构建高层特征的可行性。若此问题得到肯定回答，将产生两个重要结果。从实践角度，这提供了一种从无标注数据开发特征的廉价方法。但或许更重要的是，它回答了一个有趣的问题："祖母神经元(grandmother neuron)"的特异性是否可能从无标注数据中学习得到。非正式地说，这将表明至少在原则上，婴儿可能因为看到过许多面孔而将面孔归为一类，而不是因为受到监督或奖励的引导。

无监督特征学习和深度学习已成为机器学习中从无标注数据构建特征的方法。使用现实世界中的无标注数据来学习特征是自学习框架（Raina et al., 2007）[3]的关键思想。成功的特征学习算法及其应用可以在最近的文献中找到，使用了多种方法，如RBM（Hinton et al., 2006）[4]、自编码器（Hinton & Salakhutdinov, 2006; Bengio et al., 2007）[5][6]、稀疏编码（Lee et al., 2007）[7]和K-means（Coates et al., 2011）[8]。到目前为止，这些算法大多只成功学习了低级特征，如"边缘"或"斑点"检测器。超越这些简单特征并捕捉复杂不变性是本文的研究主题。

最近的研究发现，训练深度学习算法以产生最先进的结果非常耗时（Ciresan et al., 2010）[9]。我们猜想，长时间的训练是文献中缺乏高层特征的部分原因。例如，研究人员通常为了在可行的时间内训练网络而减小数据集和模型的规模，而这些缩减损害了高层特征的学习。

我们通过扩展训练深度网络所涉及的核心组件来解决这个问题：数据集、模型和计算资源。

首先，我们使用一个大型数据集，该数据集通过从随机YouTube视频中采样随机帧生成[^1]。我们的输入数据是200x200图像，远大于深度学习和无监督特征学习中通常使用的32x32图像（Krizhevsky, 2009; Ciresan et al., 2010; Le et al., 2010; Coates et al., 2011）[10][9][11][8]。我们的模型是一个带有池化(pooling)和局部对比度归一化的深度自编码器，通过使用大型计算机集群扩展到这些大尺寸图像。为了在该集群上支持并行化，我们使用了局部感受野的想法，例如（Raina et al., 2009; Le et al., 2010; 2011b）[12][11][13]。这个想法减少了机器之间的通信开销，从而允许模型并行化（参数分布在多台机器上）。采用异步SGD来支持数据并行化。该模型在1000台机器（16000核）的集群上以分布式方式训练了三天。

使用分类和可视化的实验结果证实，确实可以从无标注数据构建高层特征。特别是，使用包含人脸和干扰物的保留测试集，我们发现了一个对人脸高度选择的特征。

该结果也通过数值优化的可视化得到验证。对照实验表明，学习到的检测器不仅对平移不变，对平面外旋转和缩放也是不变的。

类似实验揭示，该网络还学习了猫脸和人体的概念。

学习到的表示也具有判别性。使用学习到的特征，我们在ImageNet物体识别上取得了显著提升。例如，在包含20000个类别的ImageNet上，我们达到了15.8%的准确率，相对于最先进水平提升了70%。

[^1]: 这与（Lee et al., 2009）[14]的工作不同，后者是在来自单一类别的图像上训练模型。

## 2. 训练集构建

我们的训练数据集通过从1000万个YouTube视频中采样帧构建。为避免重复，每个视频只贡献一张图像给数据集。每个样本是一张200x200像素的彩色图像。部分训练图像子集见附录A。为了检查数据集中人脸的比例，我们在从数据集中随机采样的60x60 patch上运行OpenCV人脸检测器(http://opencv.willowgarage.com/wiki/)。该实验表明，被OpenCV人脸检测器检测为人脸的patch在10万个采样patch中占比不到3%。

## 3. 算法

本节中，我们描述用于从无标注训练集学习特征的算法。

### 3.1. 先前工作

我们的工作受到最近在无监督特征学习和深度学习方面成功算法的启发（Hinton et al., 2006; Bengio et al., 2007; Ranzato et al., 2007; Lee et al., 2007）[4][6][15][7]。它深受（Olshausen & Field, 1996）[16]关于稀疏编码研究的影响。根据他们的研究，稀疏编码可以在无标注的自然图像上训练，产生类似于V1简单细胞的感受野（Hubel & Wiesel, 1959）[17]。

早期方法（如稀疏编码（Olshausen & Field, 1996）[16]）的一个缺点是它们的架构较浅，通常只捕获低级概念（例如，边缘"Gabor"滤波器）和简单不变性。解决这个问题是最近深度学习工作的重点（Hinton et al., 2006; Bengio et al., 2007; Bengio & LeCun, 2007; Lee et al., 2008; 2009）[4][6][18][19][14]，它们构建了层次化的特征表示。特别是，Lee et al.（2008）[19]表明堆叠稀疏RBM可以对皮层V2区域的某些简单功能建模。他们还展示了在对齐的人脸图像上训练的卷积DBN（Lee et al., 2009）[14]可以学习一个人脸检测器。这个结果是令人感兴趣的，但不幸的是在数据集构建过程中需要一定程度的监督：他们的训练图像（即Caltech 101图像）是对齐的、同质的，且属于一个选定的类别。

![Figure 1](Figure 1 description: 网络一层中的架构和参数。整个网络将该结构重复三次。为简单起见，图像以一维表示。)

### 3.2. 架构

我们的算法建立在这些想法之上，可以看作是一个稀疏深度自编码器，具有三个重要组成部分：局部感受野、池化(pooling)和局部对比度归一化。首先，为了将自编码器扩展到大型图像，我们使用了一个简单的想法——局部感受野（LeCun et al., 1998; Raina et al., 2009; Lee et al., 2009; Le et al., 2010）[20][12][14][11]。这个受生物学启发的想法提出，自编码器中的每个特征只能连接到下层的一个小区域。接下来，为了实现局部形变的不变性，我们采用了局部L2池化(pooling)（Hyvärinen et al., 2009; Le et al., 2010）[21][11]和局部对比度归一化（Jarrett et al., 2009）[22]。特别是L2池化(pooling)允许学习不变特征（Hyvärinen et al., 2009; Le et al., 2010）[21][11]。

我们的深度自编码器通过将同一阶段（由局部滤波、局部池化和局部对比度归一化组成）重复三次构建而成。一个阶段的输出是下一个阶段的输入，整个模型可以解释为一个九层网络（见图1）。

第一和第二个子层通常分别称为滤波（或简单）层和池化（或复杂）层。第三个子层执行局部减法和除法归一化，其灵感来自生物学和计算模型（Pinto et al., 2008; Lyu & Simoncelli, 2008; Jarrett et al., 2009）[23][24][22][^2]。

如上所述，我们方法的核心是使用神经元之间的局部连接。在我们的实验中，第一个子层的感受野为18x18像素，第二个子层在5x5重叠的特征邻域上进行池化(pooling)（即池化大小）。第一个子层的神经元连接到所有输入通道（或map）的像素，而第二个子层的神经元仅连接到一个通道（或map）的像素[^3]。虽然第一个子层输出线性滤波响应，池化层输出其输入平方和平方根，因此被称为L2池化(pooling)。

我们堆叠一系列统一模块（在选择性和容忍度层之间切换）的风格，让人联想到Neocognitron和HMAX（Fukushima & Miyake, 1982; LeCun et al., 1998; Riesenhuber & Poggio, 1999）[25][20][26]。这也被认为是大脑采用的一种架构（DiCarlo et al., 2012）[27]。

尽管我们使用了局部感受野，但它们不是卷积的：参数在图像的不同位置之间不共享。这是我们方法与之前工作之间的明显区别（LeCun et al., 1998; Jarrett et al., 2009; Lee et al., 2009）[20][22][14]。除了更具生物学合理性之外，不共享权重允许学习除平移不变性之外的更多不变性（Le et al., 2010）[11]。

就规模而言，我们的网络可能是迄今为止已知的最大网络之一。它包含10亿个可训练参数，比文献中报道的其他大型网络（例如（Ciresan et al., 2010; Sermanet & LeCun, 2011）[9][28]约1000万参数）大一个数量级以上。值得注意的是，与人类的视觉皮层相比，我们的网络仍然很小，后者的神经元和突触数量要大 $10^6$ 倍（Pakkenberg et al., 2003）[29]。

[^2]: 减法归一化从当前神经元中移除相邻神经元的加权平均： $g_{i,j,k} = h_{i,j,k} - \sum_{iuv} G_{uv} h_{i,j+u,i+v}$ 。除法归一化计算 $y_{i,j,k} = g_{i,j,k} / \max\{c, (\sum_{iuv} G_{uv} g^2_{i,j+u,i+v})^{0.5}\}$ ，其中 $c$ 设为一个小的数0.01以防止数值错误。 $G$ 是一个高斯加权窗口。（Jarrett et al., 2009）[22]
[^3]: 关于连接模式和参数敏感性的更多细节，见附录B和E。

### 3.3. 学习与优化

**学习：** 在学习过程中，第二个子层（ $H$ ）的参数固定为均匀权重，而第一个子层的编码权重 $W_1$ 和解码权重 $W_2$ 通过以下优化问题调整：

$$
\minimize_{W_1,W_2} \sum_{i=1}^m \left( \| W_2 W_1^T x^{(i)} - x^{(i)} \|_2^2 + \lambda \sum_{j=1}^k \sqrt{\epsilon + H_j(W_1^T x^{(i)})^2} \right) \qquad (1)
$$

其中 $\lambda$ 是稀疏性和重构之间的权衡参数； $m$ 、 $k$ 分别是样本数量和一层中池化单元的数量； $H_j$ 是第 $j$ 个池化单元的权重向量。在我们的实验中，设置 $\lambda = 0.1$ 。

该优化问题也称为重构拓扑独立成分分析（Hyvärinen et al., 2009; Le et al., 2011a）[21][30][^4]。目标中的第一项确保表示编码了数据的重要信息，即它们可以重构输入数据；而第二项鼓励池化特征将相似特征分组在一起以实现不变性。

**优化：** 我们模型中的所有参数都联合训练，目标函数是三个层目标函数之和。

为了训练模型，我们实现了模型并行化，将局部权重 $W_1$ 、 $W_2$ 和 $H$ 分布到不同的机器上。模型的一个实例将神经元和权重分配到169台机器上（每台机器有16个CPU核）。一组共同组成模型单个副本的机器被称为"模型副本(model replica)"。我们构建了一个名为DistBelief的软件框架，用于管理模型副本内不同机器之间的所有必要通信，因此该框架的用户只需编写模型中神经元所需的向上和向下计算函数，而无需处理跨机器的底层数据通信。

我们通过使用多个核心模型副本实现异步SGD进一步扩展了训练。对于本文描述的实验，我们将训练集分为5部分，在每一部分上运行模型的一个副本。这些模型通过一组集中的"参数服务器(parameter server)"进行更新通信，该参数服务器在一组分区服务器中维护模型所有参数的当前状态（我们使用了256个参数服务器分区来训练本文描述的模型）。在最简单的实现中，在处理每个mini-batch之前，模型副本向集中的参数服务器请求其模型参数的更新副本。然后它处理一个mini-batch以计算参数梯度，并将参数梯度发送到相应的参数服务器，后者随后将每个梯度应用到模型参数的当前值。我们可以通过让每个模型副本每 $P$ 步请求更新参数，并每 $G$ 步（ $G$ 可能不等于 $P$ ）将更新后的梯度值发送给参数服务器来减少通信开销。我们的DistBelief软件框架自动管理模型分区和参数服务器之间的参数和梯度传输，使层函数的实现者无需处理这些问题。

异步SGD比标准（同步）SGD对故障更鲁棒。具体来说，对于同步SGD，如果一台机器宕机，整个训练过程都会被延迟；而对于异步SGD，如果一台机器宕机，只有一个SGD副本被延迟，而其余优化过程仍可继续。

在我们的训练中，在SGD的每一步，梯度在100个样本的minibatch上计算。我们在1000台机器的集群上训练网络三天。关于优化实现的更多细节，见附录B、C和D。

[^4]: 在（Bengio et al., 2007; Le et al., 2011a）[6][30]中，编码权重和解码权重是绑定的： $W_1 = W_2$ 。然而，为了更好的并行性和更好的特征，我们的实现不强制权重绑定。

## 4. 人脸实验

本节中，我们描述对学习表示在识别人脸方面的分析（"人脸检测器"），并呈现理解人脸检测器不变性性质的对照实验。其他概念的结果在下一节中呈现。

### 4.1. 测试集

测试集包含从两个数据集采样的37,000张图像：Labeled Faces In the Wild数据集（Huang et al., 2007）[31]和ImageNet数据集（Deng et al., 2009）[32]。其中有13,026张人脸来自非对齐的Labeled Faces In The Wild[^5]。其余是从ImageNet随机采样的干扰物图像。这些图像被调整为适合顶层神经元的可视区域。一些示例图像见附录A。

[^5]: http://vis-www.cs.umass.edu/lfw/lfw.tgz

### 4.2. 实验协议

训练完成后，我们使用该测试集测量每个神经元在分类人脸与干扰物方面的性能。对于每个神经元，我们找到其最大和最小激活值，然后在它们之间选取20个等间隔阈值。报告的准确率是20个阈值中的最佳分类准确率。

### 4.3. 识别

令人惊讶的是，网络中的最佳神经元在识别人脸方面表现非常好，尽管训练期间没有给出任何监督信号。网络中最佳神经元在人脸检测中达到了81.7%的准确率。测试集中有13,026张人脸，因此全部预测为负样本仅能达到64.8%的准确率。单层网络中的最佳神经元仅达到71%的准确率，而从训练集中随机采样的100,000个滤波器中选出的最佳线性滤波器仅达到74%。

为了理解它们的贡献，我们移除了局部对比度归一化子层并重新训练了网络。结果表明最佳神经元的准确率下降到78.5%。这与先前显示局部对比度归一化重要性的研究一致（Jarrett et al., 2009）[22]。

我们在图2中可视化了人脸图像和随机图像的激活值直方图。可以看到，即使完全使用无标注数据，神经元也能学习区分人脸和随机干扰物。具体来说，当输入一张人脸图像时，该神经元的输出值倾向于大于阈值0。相比之下，如果输入一张随机图像，该神经元的输出值倾向于小于0。

![Figure 2](Figure 2 description: 人脸（红色）与非人脸（蓝色）的直方图。测试集经过子采样使得人脸与非人脸的比例为1:1。)

### 4.4. 可视化

本节中，我们将呈现两种可视化技术来验证神经元的最优刺激是否确实是人脸。第一种方法是可视化测试集中最响应的刺激。由于测试集很大，该方法可以可靠地检测出被测试神经元的接近最优的刺激。第二种方法是执行数值优化以找到最优刺激（Berkes & Wiskott, 2005; Erhan et al., 2009; Le et al., 2010）[33][34][11]。特别地，我们找到使被测试神经元的输出 $f$ 最大化的范数有界输入 $x$ ，通过求解：

$$
x^* = \arg\min_x f(x; W, H), \quad \text{subject to } \|x\|_2 = 1
$$

这里 $f(x; W, H)$ 是在给定学习到的参数 $W$ 、 $H$ 和输入 $x$ 时被测试神经元的输出。在我们的实验中，这个带约束的优化问题通过带线搜索的投影梯度下降求解。

这些可视化方法有互补的优势和弱点。例如，可视化最响应的刺激可能受到过拟合噪声的影响。另一方面，数值优化方法可能容易陷入局部最小值。结果如图13所示，证实了被测试神经元确实学习了人脸的概念。

![Figure 3](Figure 3 description: 上：测试集中最佳神经元的前48个最刺激刺激。下：根据数值约束优化得到的最优刺激。)

### 4.5. 不变性性质

我们希望评估人脸检测器对常见物体变换（如平移、缩放和平面外旋转）的鲁棒性。首先，我们选择了一组10张人脸图像并对它们进行形变（如缩放和平移）。对于平面外旋转，我们使用了10张3D旋转人脸图像（"out-of-plane"）作为测试集。为了检查神经元的鲁棒性，我们绘制了其在小测试集上的平均响应随尺度变化、3D旋转变化（图4）和平移变化（图5）的情况[^6]。

![Figure 4](Figure 4 description: 最佳特征的尺度（左）和平面外（3D）旋转（右）不变性性质。)

![Figure 5](Figure 5 description: 最佳特征的平移不变性性质。x轴单位为像素。)

结果显示该神经元对复杂且难以硬编码的不变性（如平面外旋转和缩放）具有鲁棒性。

**关于不包含人脸的数据集的对照实验：** 如上所述，最佳神经元在分类人脸与随机干扰物方面达到了81.7%的准确率。如果我们从训练集中移除所有包含人脸的图像会怎样？

我们通过在OpenCV中运行人脸检测器并移除那些包含至少一张人脸的训练图像来执行对照实验。最佳神经元的识别准确率下降到72.5%，与第4.3节中报告的那些简单线性滤波器相当。

[^6]: 缩放和平移的人脸通过标准三次插值生成。对于3D旋转的人脸，我们使用了来自Sheffield Face Database的10个旋转人脸序列——http://www.sheffield.ac.uk/eee/research/iel/research/face。不同的序列记录了不同个体的旋转人脸。数据集仅包含最多90度的旋转人脸。见附录F的示例序列。

## 5. 猫和人体检测器

在获得了一个对人脸敏感的神经元之后，我们希望了解网络是否也能检测其他高层概念。

我们观察到YouTube数据集中最常见的物体是身体部位和宠物，因此怀疑网络也学习了这些概念。为了验证这一假设并量化网络对这些概念的选择性性质，我们构建了两个数据集：一个用于分类人体与随机背景，另一个用于分类猫脸与其他随机干扰物。

![Figure 6](Figure 6 description: 猫脸神经元（左）和人体神经元（右）的可视化。)

为便于解释，这些数据集的正负样本比例与人脸数据集相同。猫脸图像收集自（Zhang et al., 2008）[35]描述的数据集。该数据集中有10,000张正样本和18,409张负样本（使得正负比例与人脸情况相似）。负样本从ImageNet数据集中随机选择。

人体数据集的负样本和正样本从一个基准数据集（Keller et al., 2009）[36]中随机子采样。在原始数据集中，每个样本是一对立体黑白图像。但为简单起见，我们只保留左图。总体而言，与人脸情况类似，我们有13,026个正样本和23,974个负样本。

我们随后遵循与之前相同的实验协议。结果如图14所示，证实了网络不仅学习了人脸的概念，还学习了猫脸和人体的概念。

我们的高层检测器在识别率方面也优于标准基线，在猫和人体上分别达到74.8%和76.7%。相比之下，最佳线性滤波器（从训练集中采样）仅分别达到67.2%和68.1%。

在表1中，我们总结了所有先前的数值结果，将最佳神经元与其他基线（如线性滤波器和随机猜测）进行比较。为了理解训练的效果，我们还测量了相同网络在随机初始化时最佳神经元的性能。

在算法开发过程中，我们还尝试了几种其他算法，如深度自编码器（Hinton & Salakhutdinov, 2006; Bengio et al., 2007）[5][6]和K-means（Coates et al., 2011）[8]。在我们的实现中，深度自编码器也是局部连接的，并使用sigmoid激活函数。对于K-means，我们将图像下采样到40x40以降低计算成本。

我们还调整了自编码器和K-means的参数，并选择了在资源约束下最大化性能的参数。在我们的实验中，K-means使用了30,000个centroid。这些模型也以与本文描述类似的方式使用了并行化。它们同样使用了1000台机器训练三天。这些基线的结果见表1底部。

| 概念 | 随机猜测 | 相同架构随机权重 | 最佳线性滤波器 | 最佳第一层神经元 | 最佳神经元 | 无对比度归一化的最佳神经元 |
|------|---------|----------------|--------------|---------------|----------|------------------------|
| 人脸 | 64.8% | 67.0% | 74.0% | 71.0% | **81.7%** | 78.5% |
| 人体 | 64.8% | 66.5% | 68.1% | 67.2% | **76.8%** | 71.8% |
| 猫 | 64.8% | 66.0% | 67.8% | 67.1% | **74.6%** | 69.3% |

表1：我们的算法与其他基线的数值比较。上：我们的算法与简单基线对比。前三列是不需要训练的方法的结果：随机猜测、随机权重（网络初始化时的权重，未经任何训练）和从训练集中采样的100,000个样本中选出的最佳线性滤波器。后三列是需要训练的方法的结果：第一层最佳神经元、训练后最高层最佳神经元、移除对比度归一化层后网络中的最佳神经元。下：我们的算法与自编码器和K-means的对比。

| 概念 | 我们的网络 | 深度自编码器3层 | 深度自编码器6层 | K-means (40x40图像) |
|------|----------|--------------|--------------|-----------------|
| 人脸 | **81.7%** | 72.3% | 70.9% | 72.5% |
| 人体 | **76.7%** | 71.2% | 69.8% | 69.3% |
| 猫 | **74.8%** | 67.5% | 68.3% | 68.5% |

## 6. 基于ImageNet的物体识别

我们将特征学习方法应用于ImageNet数据集中的物体识别任务（Deng et al., 2009）[32]。在YouTube和ImageNet图像上进行无监督训练后，我们在最高层之上添加了一对其余逻辑分类器。我们首先训练逻辑分类器，然后微调网络。逻辑分类器中未使用正则化。整个训练在2000台机器上进行了一周。

我们遵循（Deng et al., 2010; Sanchez & Perronnin, 2011）[37][38]中指定的实验协议，将数据集随机分成两半用于训练和验证。我们在验证集上报告性能，并与表2中的最先进基线进行比较。注意，数据划分与先前工作不完全相同，但验证集性能在不同划分之间略有变化。

| 数据集版本 | 2009（约900万图像，约10K类别） | 2011（约1600万图像，约20K类别） |
|-----------|---------------------------|----------------------------|
| 最先进水平 | 16.7%（Sanchez & Perronnin, 2011）[38] | 9.3%（Weston et al., 2011）[39] |
| 我们的方法 | 16.1%（无预训练） / **19.2%（有预训练）** | 13.6%（无预训练） / **15.8%（有预训练）** |

表2：我们的方法在ImageNet上与其他最先进基线的分类准确率汇总。

结果表明，我们的方法从零开始（即原始像素）即优于许多使用手工设计特征的最先进基线。在包含10K类别的ImageNet上，我们的方法相比先前发布的最佳结果提升了15%。在包含20K类别的ImageNet上，我们的方法相比我们所知的最高结果（包括（Weston et al., 2011）[39]作者已知的未发表结果）实现了70%的相对提升。

## 7. 结论

在本工作中，我们使用无标注数据模拟了高层类别特定神经元。我们通过结合近期开发的算法中的思想（从无标注数据学习不变性）实现了这一目标。得益于模型并行化和异步SGD，我们的实现可扩展到包含数千台机器的集群。

我们的工作表明，使用完全无标注的数据训练对高层概念具有选择性的神经元是可能的。在我们的实验中，通过在YouTube视频的随机帧上进行训练，我们获得了充当人脸、人体和猫脸检测器的神经元。这些神经元自然地捕捉了如平面外和尺度不变性等复杂不变性。使用学习到的表示，我们在包含20,000个类别的ImageNet物体识别上获得了15.8%的准确率，相比最先进水平显著提升了70%。

**致谢：** 感谢Samy Bengio、Adam Coates、Tom Dean、Mark Mao、Peter Norvig、Paul Tucker、Andrew Saxe和Jon Shlens的有益讨论和建议。

## 参考文献

[1] Desimone, R., Albright, T., Gross, C., and Bruce, C. Stimulus-selective properties of inferior temporal neurons in the macaque. *The Journal of Neuroscience*, 1984.

[2] Quiroga, R. Q., Reddy, L., Kreiman, G., Koch, C., and Fried, I. Invariant visual representation by single neurons in the human brain. *Nature*, 2005.

[3] Raina, R., Battle, A., Lee, H., Packer, B., and Ng, A.Y. Self-taught learning: Transfer learning from unlabelled data. In *ICML*, 2007.

[4] Hinton, G. E., Osindero, S., and Teh, Y. W. A fast learning algorithm for deep belief nets. *Neural Computation*, 2006.

[5] Hinton, G. E. and Salakhutdinov, R.R. Reducing the dimensionality of data with neural networks. *Science*, 2006.

[6] Bengio, Y., Lamblin, P., Popovici, D., and Larochelle, H. Greedy layerwise training of deep networks. In *NIPS*, 2007.

[7] Lee, H., Battle, A., Raina, R., and Ng, Andrew Y. Efficient sparse coding algorithms. In *NIPS*, 2007.

[8] Coates, A., Lee, H., and Ng, A. Y. An analysis of single-layer networks in unsupervised feature learning. In *AISTATS 14*, 2011.

[9] Ciresan, D. C., Meier, U., Gambardella, L. M., and Schmidhuber, J. Deep big simple neural nets excel on handwritten digit recognition. *CoRR*, 2010.

[10] Krizhevsky, A. Learning multiple layers of features from tiny images. Technical report, University of Toronto, 2009.

[11] Le, Q. V., Ngiam, J., Chen, Z., Chia, D., Koh, P. W., and Ng, A. Y. Tiled convolutional neural networks. In *NIPS*, 2010.

[12] Raina, R., Madhavan, A., and Ng, A. Y. Large-scale deep unsupervised learning using graphics processors. In *ICML*, 2009.

[13] Le, Q.V., Ngiam, J., Coates, A., Lahiri, A., Prochnow, B., and Ng, A.Y. On optimization methods for deep learning. In *ICML*, 2011b.

[14] Lee, H., Grosse, R., Ranganath, R., and Ng, A.Y. Convolutional deep belief networks for scalable unsupervised learning of hierarchical representations. In *ICML*, 2009.

[15] Ranzato, M., Huang, F. J, Boureau, Y., and LeCun, Y. Unsupervised learning of invariant feature hierarchies with applications to object recognition. In *CVPR*, 2007.

[16] Olshausen, B. and Field, D. Emergence of simple-cell receptive field properties by learning a sparse code for natural images. *Nature*, 1996.

[17] Hubel, D. H. and Wiesel, T.N. Receptive fields of single neurons in the cat's visual cortex. *Journal of Physiology*, 1959.

[18] Bengio, Y. and LeCun, Y. Scaling learning algorithms towards AI. In *Large-Scale Kernel Machines*, 2007.

[19] Lee, H., Ekanadham, C., and Ng, A. Y. Sparse deep belief net model for visual area V2. In *NIPS*, 2008.

[20] LeCun, Y., Bottou, L., Bengio, Y., and Haffner, P. Gradient based learning applied to document recognition. *Proceedings of the IEEE*, 1998.

[21] Hyvärinen, A., Hurri, J., and Hoyer, P. O. *Natural Image Statistics*. Springer, 2009.

[22] Jarrett, K., Kavukcuoglu, K., Ranzato, M.A., and LeCun, Y. What is the best multi-stage architecture for object recognition? In *ICCV*, 2009.

[23] Pinto, N., Cox, D. D., and DiCarlo, J. J. Why is real-world visual object recognition hard? *PLoS Computational Biology*, 2008.

[24] Lyu, S. and Simoncelli, E. P. Nonlinear image representation using divisive normalization. In *CVPR*, 2008.

[25] Fukushima, K. and Miyake, S. Neocognitron: A new algorithm for pattern recognition tolerant of deformations and shifts in position. *Pattern Recognition*, 1982.

[26] Riesenhuber, M. and Poggio, T. Hierarchical models of object recognition in cortex. *Nature Neuroscience*, 1999.

[27] DiCarlo, J. J., Zoccolan, D., and Rust, N. C. How does the brain solve visual object recognition? *Neuron*, 2012.

[28] Sermanet, P. and LeCun, Y. Traffic sign recognition with multiscale convolutional neural networks. In *IJCNN*, 2011.

[29] Pakkenberg, B., Pelvig, D., Marner, L., Bundgaard, M. J., Gundersen, H. J. G., Nyengaard, J. R., and Regeur, L. Aging and the human neocortex. *Experimental Gerontology*, 2003.

[30] Le, Q. V., Karpenko, A., Ngiam, J., and Ng, A. Y. ICA with Reconstruction Cost for Efficient Overcomplete Feature Learning. In *NIPS*, 2011a.

[31] Huang, G. B., Ramesh, M., Berg, T., and Learned-Miller, E. Labeled faces in the wild: A database for studying face recognition in unconstrained environments. Technical Report 07-49, University of Massachusetts, Amherst, October 2007.

[32] Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., and Fei-Fei, L. ImageNet: A Large-Scale Hierarchical Image Database. In *CVPR*, 2009.

[33] Berkes, P. and Wiskott, L. Slow feature analysis yields a rich repertoire of complex cell properties. *Journal of Vision*, 2005.

[34] Erhan, D., Bengio, Y., Courville, A., and Vincent, P. Visualizing higher-layer features of deep networks. Technical report, University of Montreal, 2009.

[35] Zhang, W., Sun, J., and Tang, X. Cat head detection - how to effectively exploit shape and texture features. In *ECCV*, 2008.

[36] Keller, C., Enzweiler, M., and Gavrila, D. M. A new benchmark for stereo-based pedestrian detection. In *Proc. of the IEEE Intelligent Vehicles Symposium*, 2009.

[37] Deng, J., Berg, A., Li, K., and Fei-Fei, L. What does classifying more than 10,000 image categories tell us? In *ECCV*, 2010.

[38] Sanchez, J. and Perronnin, F. High-dimensional signature compression for large-scale image classification. In *CVPR*, 2011.

[39] Weston, J., Bengio, S., and Usunier, N. Wsabie: Scaling up to large vocabulary image annotation. In *IJCAI*, 2011.

---

## 附录A. 训练和测试图像

部分训练图像子集如图7所示。可以看出，数据集中人脸的位置、尺度、方向是多样的。用于识别脸神经元的测试图像子集如图8所示。

![Figure 7](Figure 7 description: 30张随机选择的训练图像（在美白步骤之前显示）。)

![Figure 8](Figure 8 description: 一些示例测试集图像（在美白步骤之前显示）。)

## 附录B. 模型

本文方法的核心是使用局部连接网络。在这些网络中，神经元只连接到下一层的局部区域。在图9中，我们展示了本文描述的神经网络架构的连接模式。实验中的实际图像是2D的，但为简单起见，我们可视化中的图像是1D的。

![Figure 9](Figure 9 description: 我们使用的网络示意图，包含更详细的连接模式。彩色箭头表示权重仅连接到一个map。深色箭头表示权重连接到所有map。池化神经元仅连接到一个map，而简单神经元和LCN神经元连接到所有map。)

## 附录C. 模型并行化

我们使用模型并行化将参数的存储和梯度计算分布到不同的机器上。在图10中，我们展示了权重如何被划分并存储在不同的"分区(partition)"（或更简单地，机器）中（另见（Krizhevsky, 2009）[10]）。

![Figure 10](Figure 10 description: 所使用网络架构的模型并行化。可以看到，权重根据图像的局部性进行划分，并存储在不同的机器上。具体来说，连接到图像左侧的权重存储在机器1（"分区1"）。连接到图像中央部分的权重存储在机器2（"分区2"）。连接到图像右侧的权重存储在机器3（"分区3"）。)

## 附录D. 进一步的多核并行化

我们集群中的机器有许多核，允许进一步的并行化。因此，我们将这些核拆分为执行不同的任务。在我们的实现中，这些核分为三组：读取数据、发送（或写入）数据以及执行算术计算。在每个时间点，这些组并行工作以加载数据、计算数值结果并发送到网络或写入磁盘。

## 附录E. 参数敏感性

网络的超参数选择为适应计算约束并优化我们算法的训练时间。这些参数可以改变，但需要更长的训练时间或更多的计算资源。例如，可以增加感受野的大小，但代价是使用更多内存、更多计算和每台机器更多的网络带宽；或者可以增加map的数量，但代价是使用更多机器和内存。

这些超参数也可能影响特征的性能。我们进行了对照实验以理解两个超参数的影响：感受野的大小和map的数量。通过改变这些参数中的每一个并观察测试集准确率，我们可以了解它们对人脸识别任务性能的影响程度。结果如图11所示，证实了结果对这些控制参数的变化仅略敏感。

![Figure 11](Figure 11 description: 左：感受野大小对测试集准确率的影响。右：map数量对测试集准确率的影响。)

## 附录F. 平面外旋转人脸序列示例

在图12中，我们展示了一个3D（平面外）旋转人脸的示例序列。注意，这些脸是黑白的，但在测试中被当作彩色图像处理。更多详细信息可在Sheffield Face Database数据集的网页上获得——http://www.sheffield.ac.uk/eee/research/iel/research/face

![Figure 12](Figure 12 description: 一个个体的一张3D（平面外）旋转人脸序列。数据集包含10个序列。)

## 附录G. 最佳线性滤波器

在本文中，我们进行了对照实验，将我们的特征与"最佳线性滤波器(best linear filters)"进行比较。该基线的工作方式如下。第一步是从训练集中采样100,000个随机patch（或滤波器）（每个patch的大小与测试集图像相同）。然后对每个patch，计算其与测试集图像之间的余弦距离。余弦距离被视为特征值。使用这些特征值，我们在20个阈值中搜索，以找到patch在分类人脸与干扰物方面的最佳准确率。每个patch为我们的测试集给出一个准确率。报告的准确率是从训练集中随机选择的100,000个patch中的最佳准确率。

## 附录H. 完整测试集上的直方图

这里，我们还展示了整个测试集上神经元的详细直方图。直方图对正样本和负样本图像具有区分性的事实表明网络已经学习了概念检测器。

![Figure 13](Figure 13 description: 测试集上最佳人脸神经元的激活值直方图。红色：人脸图像的直方图。蓝色：随机干扰物的直方图。)

![Figure 14](Figure 14 description: 测试集上最佳人体神经元的直方图。红色：人体图像的直方图。蓝色：随机干扰物的直方图。)

## 附录I. 猫和人体最响应的刺激

在图16中，我们展示了猫和人体神经元在测试集上最响应的刺激。注意，人体神经元的顶级刺激是黑白图像，因为测试集图像是黑白的（Keller et al., 2009）[36]。

![Figure 15](Figure 15 description: 测试集上最佳猫神经元的直方图。红色：猫图像的直方图。蓝色：随机干扰物的直方图。)

![Figure 16](Figure 16 description: 上：测试集中对猫神经元最响应的刺激。右：测试集中对人体神经元最响应的人体刺激。)
