# Deep Metric Learning Using Triplet Network

> Elad Hoffer | 电气工程系，以色列理工学院，ehoffer@tx.technion.ac.il
> Nir Ailon | 计算机科学系，以色列理工学院，nailon@cs.technion.ac.il

深度学习已被证明是一组成功的模型，可用于学习数据的有用语义表示。然而，这些表示大多是作为分类任务的一部分隐式学习的。在本文中，我们提出了三元组网络（triplet network）模型，旨在通过距离比较来学习有用的表示。Wang 等人（2014）定义了一个类似的模型，专门用于学习图像信息检索的排序。我们在此使用多个数据集证明，我们的模型学习到的表示优于其直接竞争对手——孪生网络（Siamese network）。我们还讨论了其作为无监督学习框架的未来可能用途。

关键发现：
- 三元组网络在多个数据集（MNIST、CIFAR-10、SVHN、STL-10）上的分类准确率均优于孪生网络
- STL-10 上取得了无数据增强条件下的最佳已知结果（70.67%）
- 学习到的表示具有稀疏性（约 25% 非零值），有利于后续分类任务
- 嵌入表示的 2D 可视化显示出显著的语义聚类效果

---

## 摘要

深度学习已被证明是一组成功的模型，可用于学习数据的有用语义表示。然而，这些表示大多是作为分类任务的一部分隐式学习的。在本文中，我们提出了三元组网络（triplet network）模型，旨在通过距离比较来学习有用的表示。Wang 等人（2014）定义了一个类似的模型，专门用于学习图像信息检索的排序。我们在此使用多个数据集证明，我们的模型学习到的表示优于其直接竞争对手——孪生网络（Siamese network）。我们还讨论了其作为无监督学习框架的未来可能用途。

## 1 引言

在过去的几年里，深度学习模型已被广泛用于解决各种机器学习任务。其中一个基本假设是，深层分层模型（如卷积网络）能够创建有用的数据表示（Bengio（2009）；Hinton（2007）），然后再用于区分可用类别。这一特性与需要从数据中提取人工特征然后在单独的学习方案中使用的传统方法形成对比。深度网络提取的特征也被证明能够提供有用的表示（Zeiler & Fergus（2013a）；Sermanet et al.（2013）），进而可成功用于其他任务（Razavian et al.（2014））。

尽管这些表示及其对应的诱导度量（induced metric）非常重要，但它们通常被视为分类任务的副产品，而非明确追求的目标。还有许多关于中间表示及其在解缠（disentangling）和解释数据中的作用的开放性问题（Bengio（2013））。值得注意的例外是孪生网络变体（Siamese Network variants）（Bromley et al.（1993）；Chopra et al.（2005）；Hadsell et al.（2006）），其中使用表示所诱导的度量上的对比损失（contrastive loss）来训练网络区分相似和不相似的样本对。对比损失倾向于使标记为相似的样本对之间的距离较小，而标记为不相似的样本对之间的距离较大。然而，这些模型学习到的表示在用作分类特征时，与其他深度学习模型（包括我们的模型）相比效果欠佳。孪生网络对校准（calibration）也很敏感，因为相似与不相似的概念需要上下文。例如，在提供随机对象数据集时，一个人可能与另一个人被认为是相似的，但在我们只想区分一个个体集合中的两个个体时，同一个个体相对于同一个其他人则可能被认为是不相似的。在我们的模型中，不需要这样的校准。事实上，在我们的实验中，我们亲身体验到了使用孪生网络的困难。

我们遵循与 Chechik et al.（2010）类似的任务。对于一组样本 $P$ 和通过训练 oracle 给定的粗略相似度度量 $r(x, x')$（例如，两个物体图像在语义上的接近程度），我们希望学习一个由赋范度量（normed metric）诱导的相似度函数 $S(x, x')$。与 Chechik et al.（2010）的工作不同，我们的标签形式为对于三元组 $x, x_1, x_2$ 有 $r(x, x_1) > r(x, x_2)$。相应地，我们尝试拟合一个度量嵌入（metric embedding）和相应的相似度函数，满足：

$$S(x, x_1) > S(x, x_2), \quad \forall x, x_1, x_2 \in P \text{ such that } r(x, x_1) > r(x, x_2).$$

在我们的实验中，我们尝试为多类标记数据集寻找度量嵌入。我们总是取 $x_1$ 与 $x$ 属于同一类别，$x_2$ 属于不同类别，尽管在一般情况下可以做出更复杂的选择。相应地，我们将使用符号 $x^+$ 和 $x^-$ 代替 $x_1, x_2$。我们专注于寻找 $L_2$ 嵌入，通过学习一个函数 $F(x)$，使得 $S(x, x') = \|F(x) - F(x')\|_2$。受深度学习近期成功的启发，我们将使用深度网络作为我们的嵌入函数 $F(x)$。

我们将我们的方法称为三元组网络（triplet network）。Wang et al.（2014）提出了类似的方法，用于学习图像检索的排序函数。与 Wang et al.（2014）中提出的单一应用相比，我们对三元组架构进行了全面研究，我们认为该架构本身是很有趣的。事实上，我们将在下面证明，三元组方法是对孪生方法（其最明显的竞争对手）的有力竞争者。

## 2 三元组网络

三元组网络（受"孪生网络"启发）由 3 个相同的前馈网络实例（共享参数）组成。当输入 3 个样本时，网络输出 2 个中间值——其中两个输入的嵌入表示与第三个输入的嵌入表示之间的 $L_2$ 距离。如果将 3 个输入记为 $x$、$x^+$ 和 $x^-$，网络的嵌入表示记为 $\operatorname{Net}(x)$，则倒数第二层的向量为：

$$\operatorname{TripletNet}(x, x^-, x^+) = \begin{bmatrix} \|\operatorname{Net}(x) - \operatorname{Net}(x^-)\|_2 \\ \|\operatorname{Net}(x) - \operatorname{Net}(x^+)\|_2 \end{bmatrix} \in \mathbb{R}_+^2.$$

换句话说，它编码了 $x^+$ 和 $x^-$ 各自相对于参考 $x$ 的距离对。

$$
\|\operatorname{Net}(x) - \operatorname{Net}(x^-)\|_2
$$
$$
\|\operatorname{Net}(x) - \operatorname{Net}(x^+)\|_2
$$

<img src="..." />

图 1：三元组网络结构

### 2.1 训练

训练通过向网络输入样本进行，如上所述，$x$ 和 $x^+$ 属于同一类别，$x^-$ 属于不同类别。网络架构允许将该任务表达为一个 2 类分类问题，其目标是正确分类 $x^+$ 和 $x^-$ 中哪一个与 $x$ 属于同一类别。我们强调，在更一般的设置中（目标可能是学习度量嵌入），标签决定哪个样本更接近 $x$。这里我们简单地将"接近度"解释为"共享相同标签"。为了从模型输出比较算子，在两个输出上应用 SoftMax 函数——实际上创建了一个比率度量。与传统卷积网络类似，训练通过简单的 SGD 在关于该 2 类问题的负对数似然损失上进行。我们后来发现，当损失函数替换为 soft-max 结果与 $(0, 1)$ 向量之间的简单 MSE 时，可以获得更好的结果，因此损失为：

$$\operatorname{Loss}(d_+, d_-) = \|(d_+, d_- - 1)\|_2^2 = \text{const} \cdot d_+^2$$

其中

$$d_+ = \frac{e^{\|\operatorname{Net}(x) - \operatorname{Net}(x^+)\|_2}}{e^{\|\operatorname{Net}(x) - \operatorname{Net}(x^+)\|_2} + e^{\|\operatorname{Net}(x) - \operatorname{Net}(x^-)\|_2}}$$

且

$$d_- = \frac{e^{\|\operatorname{Net}(x) - \operatorname{Net}(x^-)\|_2}}{e^{\|\operatorname{Net}(x) - \operatorname{Net}(x^+)\|_2} + e^{\|\operatorname{Net}(x) - \operatorname{Net}(x^-)\|_2}}.$$

我们注意到，$\operatorname{Loss}(d_+, d_-) \to 0$ 当且仅当 $\frac{\|\operatorname{Net}(x) - \operatorname{Net}(x^+)\|}{\|\operatorname{Net}(x) - \operatorname{Net}(x^-)\|} \to 0$，这正是所需的目标。通过使用相同的共享参数网络，我们允许反向传播算法同时针对所有三个样本更新模型。

## 3 测试与结果

三元组网络使用 Torch7 环境（Collobert et al.（2011））实现并训练。

### 3.1 数据集

我们试验了 4 个数据集。第一个是 Cifar10（Krizhevsky & Hinton（2009）），包含 60000 张 $32 \\times 32$ 的彩色图像，共 10 个类别（其中 50000 张仅用于训练，10000 张仅用于测试）。第二个数据集是原始的 MNIST（LeCun et al.（1998）），包含 60000 张 $28 \\times 28$ 的手写数字（0-9）灰度图像，以及对应的 10000 张测试图像。第三个是 Netzer 等人的 Street-View-House-Numbers（SVHN），包含 600000 张 $32 \\times 32$ 的门牌数字（0-9）彩色图像。第四个数据集是 Coates et al.（2011）的 STL10，与 Cifar10 类似，包含 10 个物体类别，但只有 5000 张训练图像（而 Cifar 为 50000 张），图像尺寸更大，为 $96 \\times 96$。

需要注意的是，没有应用数据增强或白化（whitening），唯一的预处理是全局归一化为零均值和单位方差。每个训练实例（对于所有四个数据集）都是一个均匀采样的 3 张图像集合，其中 2 张属于同一类别（$x$ 和 $x^+$），第三张（$x^-$）属于不同类别。每个训练周期包含 640000 个这样的实例（每个周期随机选择），并使用固定的 64000 个实例进行测试。我们强调，每个测试实例涉及来自测试图像集的 3 张图像，该测试集不参与训练。

### 3.2 嵌入网络

对于 Cifar10 和 SVHN，我们使用了卷积网络，包含 3 个卷积层和 $2 \\times 2$ 最大池化层，后接第四层卷积层。在两个连续层之间应用 ReLU 非线性。网络配置（从输入到输出排序）包括滤波器大小 $\{5, 3, 3, 2\}$ 和特征图维度 $\{3, 64, 128, 256, 128\}$，其中 128 维向量是网络的最终嵌入表示。通常在卷积网络中，后续会使用全连接层进行分类。在我们的网络中，这一层被移除，因为我们只关心特征嵌入。

用于 STL10 的网络是相同的，只是第一层的 stride=3，以适应更大的输入尺寸。用于 MNIST 的网络是一个较小的版本，特征图尺寸为 $\{1, 32, 64, 128\}$。

### 3.3 结果

所有数据集的训练均通过 SGD 进行，初始学习率为 0.5，并采用学习率衰减策略。我们使用的动量值为 0.9。我们还使用了 dropout 正则化技术，$p = 0.5$，以避免过拟合。在每个数据集上训练 10-30 个周期后，网络在三元组比较上达到了固定的误差。然后我们使用嵌入网络从完整数据集中提取特征，并在完整的 10 类分类任务上训练一个简单的 1 层网络模型（仅使用训练集表示）。随后在测试集上测量准确率。这些结果（图 2）与使用深度学习模型（未使用任何人工数据增强）的最先进结果相当（Zeiler & Fergus（2013b）；Goodfellow et al.（2013）；Lin et al.（2013））。值得关注的是 STL-10 数据集，TripletNet 在该数据集上取得了非增强数据条件下的最佳已知结果。我们推测数据增强技术（如平移、镜像和加噪）可能提供与先前工作中所述类似的收益。

我们还注意到，当使用线性 SVM 模型或 KNN 分类（与图 2 结果偏差不超过 0.5%）对嵌入表示进行分类时，也获得了类似的结果。另一个观察到的副作用是，表示似乎是稀疏的——约 25% 的非零值。这在后续用作分类特征时非常有帮助，无论是在计算上还是在准确性上，因为每个类别仅由少数非零元素表征。

| 数据集 | TripletNet | SiameseNet | 最佳已知结果（无数据增强） |
|--------|------------|------------|--------------------------|
| MNIST | 99.5$4 \pm 0$.08% | 97.$9 \pm 0$.1% | 99.61% Mairal et al.（2014）；Lee et al.（2014） |
| CIFAR10 | 87.1% | - | 90.22% Lee et al.（2014） |
| SVHN | 95.37% | - | 98.18% Lee et al.（2014） |
| STL10 | 70.67% | - | 67.9% Lin & Kung（2014） |

图 2：分类准确率（无数据增强）

### 3.4 特征的 2D 可视化

为了检验我们的主要前提（即网络将图像嵌入到具有有意义属性的表示中），我们使用 PCA 将嵌入投影到易于可视化的 2 维欧几里得空间（图 3、图 4、图 5）。我们可以看到显著的语义聚类，证实网络在根据图像内容将其嵌入到欧几里得空间方面是有用的。物体之间的相似性可以通过测量其嵌入之间的距离轻松找到，并且如结果所示，使用简单的后续线性分类器即可达到较高的分类准确率。

### 3.5 与孪生网络性能的比较

孪生网络是我们方法最明显的竞争对手。我们对孪生网络的实现使用了相同的嵌入网络，但使用了一对样本之间的对比损失（contrastive loss），而不是三个样本（如 Chopra et al.（2005）所述）。生成的特征随后使用与 TripletNet 方法类似的线性模型进行分类。我们在 MNIST 数据集上测量到的准确率低于使用 TripletNet 表示所获得的结果（图 2）。

我们也尝试了其他三个数据集的类似比较，但不幸的是，使用孪生网络无法获得任何有意义的结果。我们推测这可能与上述上下文问题有关，并将这一推测的验证留待未来工作。

<img src="..." />

图 3：CIFAR10 — 嵌入测试数据的欧几里得表示，投影到前两个奇异向量上

<img src="..." />

图 4：MNIST — 嵌入测试数据的欧几里得表示，投影到前两个奇异向量上

<img src="..." />

图 5：SVHN — 嵌入测试数据的欧几里得表示，投影到前两个奇异向量上

## 4 未来工作

由于三元组网络模型允许通过样本比较而非直接数据标签进行学习，因此可以用作无监督学习模型。未来的研究可以在以下几种场景中进行：

- **利用空间信息**。空间上邻近的物体和图像块在语义上也预期是相似的。因此，在无监督设置中，我们可以使用同一图像中不同块之间的几何距离作为粗略相似度 oracle $r(x, x')$。
- **利用时间信息**。同样适用于时间域，其中两个连续视频帧预期描述同一物体，而 10 分钟后拍摄的一帧则不太可能如此。我们的三元组网络可以提供更好的嵌入，并改进以往在无监督环境中解决分类任务的尝试，如 Mobahi et al.（2009）的工作。

众所周知，人类更擅长准确提供比较标签。我们的框架可以用于众包学习环境。这可以与 Tamuz et al.（2011）的工作进行比较，后者使用了不同的方法。此外，收集可用于三元组网络训练的数据可能更容易，因为基于相似度度量的比较更容易获得（在同一地点拍摄的照片、共享的标注等）。

## 5 结论

在这项工作中，我们介绍了三元组网络模型，这是一种使用深度网络显式学习有用表示的工具。在多个数据集上展示的结果提供了证据，证明所学表示对分类有用的程度与显式训练用于分类样本的网络相当。我们相信，对嵌入网络的改进，如 Network-in-Network 模型（Lin et al.（2013））、Inception 模型（Szegedy et al.（2014））等，可以使三元组网络受益，其方式与它们使其他分类任务受益的方式类似。考虑到该方法只需要知道三张图像中的两张来自同一类别，而非知道该类别是什么，我们认为这一点值得进一步探究，并可能为我们提供深度网络一般学习方式的理解。我们还展示了该模型如何仅使用比较度量而非标签进行学习，这在未来可用于利用那些明确标签未知或无意义（如层次化标签）的新数据源。

## 致谢

我们衷心感谢 NVIDIA 公司捐赠 Titan-Z GPU 用于本研究。

## 参考文献

[1] Bengio, Yoshua. Learning Deep Architectures for AI, 2009. ISSN 1935-8237.

[2] Bengio, Yoshua. Deep learning of representations: Looking forward. In *Lecture Notes in Computer Science (including subseries Lecture Notes in Artificial Intelligence and Lecture Notes in Bioinformatics)*, volume 7978 LNAI, pp. 1–37, 2013. ISBN 9783642395925. doi: 10.1007/978-3-642-39593-2_1.

[3] Bromley, Jane, Bentz, James W, Bottou, Léon, Guyon, Isabelle, LeCun, Yann, Moore, Cliff, Säckinger, Eduard, and Shah, Roopak. Signature verification using a siamese time delay neural network. *International Journal of Pattern Recognition and Artificial Intelligence*, 7(04):669–688, 1993.

[4] Chechik, Gal, Sharma, Varun, Shalit, Uri, and Bengio, Samy. Large scale online learning of image similarity through ranking. *The Journal of Machine Learning Research*, 11:1109–1135, 2010.

[5] Chopra, Sumit, Hadsell, Raia, and LeCun, Yann. Learning a similarity metric discriminatively, with application to face verification. In *Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition*, volume 1, pp. 539–546, 2005. ISBN 0769523722. doi: 10.1109/CVPR.2005.202.

[6] Coates, Adam, Ng, Andrew Y, and Lee, Honglak. An analysis of single-layer networks in unsupervised feature learning. In *International Conference on Artificial Intelligence and Statistics*, pp. 215–223, 2011.

[7] Collobert, Ronan, Kavukcuoglu, Koray, and Farabet, Clément. Torch7: A matlab-like environment for machine learning. In *BigLearn, NIPS Workshop*, number EPFL-CONF-192376, 2011.

[8] Goodfellow, Ian J, Warde-Farley, David, Mirza, Mehdi, Courville, Aaron, and Bengio, Yoshua. Maxout networks. *arXiv preprint arXiv:1302.4389*, 2013.

[9] Hadsell, Raia, Chopra, Sumit, and LeCun, Yann. Dimensionality reduction by learning an invariant mapping. In *Computer vision and pattern recognition, 2006 IEEE computer society conference on*, volume 2, pp. 1735–1742. IEEE, 2006.

[10] Hinton, Geoffrey E. Learning multiple layers of representation, 2007. ISSN 13646613.

[11] Krizhevsky, Alex and Hinton, Geoffrey. Learning multiple layers of features from tiny images. *Computer Science Department, University of Toronto, Tech. Rep*, 2009.

[12] LeCun, Yann, Bottou, Léon, Bengio, Yoshua, and Haffner, Patrick. Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11):2278–2324, 1998.

[13] Lee, Chen-Yu, Xie, Saining, Gallagher, Patrick, Zhang, Zhengyou, and Tu, Zhuowen. Deeply-supervised nets. *arXiv preprint arXiv:1409.5185*, 2014.

[14] Lin, Min, Chen, Qiang, and Yan, Shuicheng. Network in network. *CoRR*, abs/1312.4400, 2013. URL http://arxiv.org/abs/1312.4400.

[15] Lin, Tsung-Han and Kung, HT. Stable and efficient representation learning with nonnegativity constraints. In *Proceedings of the 31st International Conference on Machine Learning (ICML-14)*, pp. 1323–1331, 2014.

[16] Mairal, Julien, Koniusz, Piotr, Harchaoui, Zaid, and Schmid, Cordelia. Convolutional kernel networks. In *Advances in Neural Information Processing Systems*, pp. 2627–2635, 2014.

[17] Mobahi, Hossein, Collobert, Ronan, and Weston, Jason. Deep learning from temporal coherence in video. In *Proceedings of the 26th Annual International Conference on Machine Learning*, pp. 737–744. ACM, 2009.

[18] Netzer, Yuval, Wang, Tao, Coates, Adam, Bissacco, Alessandro, Wu, Bo, and Ng, Andrew Y. Reading digits in natural images with unsupervised feature learning.

[19] Razavian, Ali Sharif, Azizpour, Hossein, Sullivan, Josephine, and Carlsson, Stefan. CNN Features off-the-shelf: an Astounding Baseline for Recognition. *Arxiv*, 2014. URL http://arxiv.org/abs/1403.6382.

[20] Sermanet, Pierre, Eigen, David, Zhang, Xiang, Mathieu, Michael, Fergus, Rob, and LeCun, Yann. OverFeat: Integrated Recognition, Localization and Detection using Convolutional Networks. *arXiv preprint arXiv:1312.6229*, pp. 1–15, 2013. URL http://arxiv.org/abs/1312.6229.

[21] Szegedy, Christian, Liu, Wei, Jia, Yangqing, Sermanet, Pierre, Reed, Scott, Anguelov, Dragomir, Erhan, Dumitru, Vanhoucke, Vincent, and Rabinovich, Andrew. Going deeper with convolutions. *CoRR*, abs/1409.4842, 2014. URL http://arxiv.org/abs/1409.4842.

[22] Tamuz, Omer, Liu, Ce, Belongie, Serge, Shamir, Ohad, and Kalai, Adam. Adaptively learning the crowd kernel. In Getoor, Lise and Scheffer, Tobias (eds.), *Proceedings of the 28th International Conference on Machine Learning (ICML-11)*, ICML '11, pp. 673–680, New York, NY, USA, June 2011. ACM. ISBN 978-1-4503-0619-5.

[23] Wang, Jiang, Song, Yang, Leung, Thomas, Rosenberg, Chuck, Wang, Jingbin, Philbin, James, Chen, Bo, and Wu, Ying. Learning fine-grained image similarity with deep ranking. In *CVPR*, 2014.

[24] Zeiler, Matthew D and Fergus, Rob. Visualizing and Understanding Convolutional Networks. *arXiv preprint arXiv:1311.2901*, 2013a. URL http://arxiv.org/abs/1311.2901.

[25] Zeiler, Matthew D and Fergus, Rob. Stochastic pooling for regularization of deep convolutional neural networks. *arXiv preprint arXiv:1301.3557*, 2013b.
