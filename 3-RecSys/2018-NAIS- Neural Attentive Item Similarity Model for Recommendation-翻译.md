# NAIS: Neural Attentive Item Similarity Model for Recommendation（中文翻译）

> Xiangnan He, Zhankui He, Jingkuan Song, Zhenguang Liu, Yu-Gang Jiang, Tat-Seng Chua | NUS, UESTC, Zhejiang Univ., Fudan Univ.

**NAIS** 提出将**注意力机制**引入item相似度模型。核心创新：
- 在 FISM 的基础上，使用注意力网络学习item之间的关系权重
- 不同item对当前预测的重要性被自适应地加权
- 在多个数据集上显著优于 FISM 等传统item相似度方法

NAIS 验证了注意力机制在 Top-N 推荐item相似度建模中的有效性。

---
本文介绍了 NAIS: Neural Attentive Item Similarity Model for Recommendation。核心内容：


关键发现：




---

## 摘要

item到item的协同过滤（又称基于item的CF）由于其可解释性和在实时个性化中的高效性，长期以来一直被用于构建工业推荐系统。它通过用户历史交互过的item来构建用户的画像，推荐与用户画像相似的新item。因此，基于item的CF方法的关键在于item相似度的估计。早期方法使用统计度量如余弦相似度和皮尔逊系数来估计item相似度，这些方法由于缺乏针对推荐任务的定制优化而不够准确。近年来，一些工作尝试从数据中学习item相似度，通过将相似度表示为一个底层模型并优化一个推荐感知的目标函数来估计模型参数。尽管已经在使用浅层线性模型学习item相似度方面做出了大量努力，但探索非线性神经网络模型用于基于item的CF的工作相对较少。

在这项工作中，我们提出了一种名为神经注意力item相似度模型（NAIS）的神经网络模型，用于基于item的CF。我们设计的NAIS的关键是一个注意力网络，它能够区分用户画像中的哪些历史item对预测更为重要。与最先进的基于item的CF方法——因子化item相似度模型（FISM）[1]相比，我们的NAIS具有更强的表示能力，且仅增加了注意力网络带来的少量额外参数。在两个公开基准上的大量实验证明了NAIS的有效性。这项工作首次为基于item的CF设计了神经网络模型，为未来神经推荐系统的发展开辟了新的研究可能性。

**索引词—** 协同过滤，基于item的CF，神经推荐模型，注意力网络

---

## 1 引言

推荐系统是许多面向客户的在线服务的核心服务，用于增加流量并盈利，例如电子商务和社交媒体网站。例如，据报道，在YouTube中，推荐贡献了主页约60%的视频点击量[2]；在Netflix，推荐系统贡献了约80%的观看电影，并每年带来超过10亿美元的商业价值，正如其首席产品官Neil Hunt所指出的[3]。

在现代推荐系统中，协同过滤（CF）——一种仅从用户-item交互中预测用户个性化偏好的技术——发挥着核心作用，尤其是在候选生成阶段[4, 5]。受Netflix Prize的推动，矩阵分解（MF）方法已成为学术界最流行的推荐方法，并在文献中被广泛研究[6, 7]。虽然MF方法在评分预测方面被证明比基于邻居的方法具有更高的准确性，但它们在实际工业应用中的使用却相对较少报道。一个可能的原因是由于MF的个性化方案——用户到item的CF，它用一个ID来刻画用户并将其与一个嵌入向量相关联。因此，为了根据用户的新交互来刷新推荐，用户的嵌入向量必须更新。然而，在大规模数据上重新训练MF模型很难实时实现，并且可能需要复杂的软件栈来支持在线学习，这使得该方法对工业环境的吸引力降低[8]。

另一方面，item到item的CF——它通过用户历史交互过的item来刻画用户，并推荐与用户画像相似的item——已在工业应用中得到大量使用[4, 2, 3, 9]。基于item的CF不仅提供更适合许多推荐场景的可解释预测，而且使实时个性化更容易实现。具体来说，估计item相似度的主要计算可以离线完成，而在线推荐模块只需要执行一系列对相似item的查找，这可以轻松实时完成。

早期的基于item的CF方法使用统计度量如皮尔逊系数和余弦相似度来估计item相似度[10]。由于这种基于启发式的方法缺乏针对推荐的定制优化，它们在Top-K推荐准确性方面通常不如基于机器学习的方法[11, 6]。为了解决这个问题，Ning等人[12]采用了一种基于机器学习的视角来处理基于item的CF，通过优化一个推荐感知的目标函数来从数据中学习item相似度。虽然可以获得更好的准确性，但直接学习整个item-item相似度矩阵具有关于item数量的二次复杂度，使得它对于需要处理数百万甚至数十亿item的实际推荐器来说不可行。

为了解决基于学习的item到itemCF的低效问题，Kabbur等人[1]提出了因子化item相似度模型（FISM），它将item表示为一个嵌入向量，并将两个item之间的相似度建模为它们嵌入向量的内积。作为表示学习的一个萌芽[13, 14]，FISM提供了最先进的推荐准确性，并且非常适合在线推荐场景。然而，我们认为FISM的建模保真度可能受到其假设的限制，即用户画像中的所有历史item在估计用户画像与目标item之间的相似度时贡献相等。直观地说，一个用户过去与多个item交互过，但这些交互过的item可能并不都同样程度地反映用户的兴趣。例如，一个爱情电影的粉丝可能也会看一部恐怖片，仅仅是因为那部电影在当时很流行。另一个例子是，用户兴趣可能随着时间变化，因此最近交互的item应该更能反映用户未来的偏好。

在这项工作中，我们通过区分交互item在贡献用户偏好方面的不同重要性，提出了一个增强的item相似度模型。我们的NAIS模型建立在FISM之上，保留了FISM在在线预测方面的高效性，同时通过学习交互item的不同重要性，比FISM更具表达力。这是通过利用神经表示学习的最新进展——注意力机制[15, 16, 17]——来学习item到item的交互实现的。我们的一个关键发现是，标准的注意力机制无法从用户历史数据中学习，原因在于用户历史长度的巨大方差。为了解决这个问题，我们通过平滑用户历史来调整注意力设计。我们在两个公开基准上进行了全面的实验来评估Top-K推荐，结果表明我们的NAIS在NDCG方面比FISM有4.5%的相对提升，并达到了具有竞争力的性能。为了促进研究社区验证NAIS并在此基础上进行进一步开发，我们已在以下网址发布了我们的实现代码：https://github.com/AaronHeee/Neural-Attentive-Item-Similarity-Model。

本文的其余部分结构如下。在第2节介绍一些预备知识之后，我们在第3节详细阐述我们提出的方法。然后我们在第4节进行实验评估。我们在第5节讨论相关工作，最后在第6节总结全文。

---

## 2 预备知识

我们首先简要重述标准的基于item的CF技术[10]。然后我们介绍基于学习的基于item的CF方法[12]和FISM[1]，它们是我们提出的NAIS方法的构建模块。

### 2.1 标准基于item的CF

基于item的CF的思想是，用户u对目标itemi的预测取决于i与用户过去交互过的所有item的相似度。形式上，基于item的CF的预测模型为：

$$
ˆyui = \Sigma_{j\inR+_u} ruj sij     (1)
$$

其中R+_u表示用户u交互过的item集合，sij表示itemi与j之间的相似度，ruj是一个交互项，表示用户u对j的已知偏好——对于显式反馈（如评分），ruj可以是表示评分分数的实数值；对于隐式反馈（如购买），ruj可以是表示u是否与j交互过的二进制值1或0。

高效在线推荐的吸引人特性来自于其在计算预测分数时的组合性。首先，当item相似度已经离线获得后，在线推荐阶段只需要检索候选itemR+_u的顶部相似item，并用方程（1）对它们进行评分。其次，为了根据用户的新交互来刷新推荐，我们只需要考虑与新交互item相似的item。这种增量复杂性使得基于item的CF非常适合在线学习和实时个性化，正如[2, 8]中所证明的。

对于item相似度sij，一种直观的方法是将item表示为其交互过的用户，并应用相似度度量如余弦相似度和皮尔逊系数[10]。另一种常见的方法是在用户-item交互图上使用随机游走[4]。然而，这种基于启发式的估计item相似度的方法缺乏针对推荐的定制优化，因此可能产生次优性能。接下来，我们介绍基于学习的方法，这些方法旨在通过自适应地从数据中学习item相似度来提高基于item的CF的准确性。

### 2.2 基于学习的基于item的CF方法

在[12]中，作者提出了一种名为SLIM（稀疏线性方法）的方法，它通过优化一个推荐感知的目标函数来学习item相似度。其思想是最小化原始用户-item交互矩阵与从基于item的CF模型重建的矩阵之间的损失。形式上，要最小化的目标函数如下：

$$
L = 1/2 \Sigma_{u=1}^U \Sigma_{i=1}^I (rui - ˆyui)² + \beta||S||₂ + \gamma||S||₁    (2)

subject to S \geq 0, diag(S) = 0,
$$

其中U和I分别表示用户数和item数，S\inR^{I×I}表示item-item相似度矩阵，\beta控制L2正则化的强度以防止过拟合。注意在SLIM中有三个特意设计的关于S的约束，以确保有效的item相似度学习：1）由\gamma控制的L1正则化以强制S的稀疏性，因为实际上只有少数item与某个item特别相似；2）对S每个元素的非负约束，使其成为一个有意义的相似度度量；3）对S对角线元素的零约束，以消除目标item本身在估计预测中的影响。

尽管可以获得更好的推荐准确性，但SLIM有两个固有的局限性。首先，由于直接学习具有I²个元素的S（时间复杂度为O(I²)量级），对于大规模数据，离线训练过程可能非常耗时。其次，它只能学习之前被共同评分过的两个item之间的相似度，而无法捕捉item之间的传递关系。为了解决这些局限性，后来的工作[1]提出了FISM（因子化item相似度模型），它将item表示为一个低维嵌入向量；然后相似度分数sij被参数化为i和j的嵌入向量之间的内积。形式上，FISM的预测模型为：

$$
ˆyui = pT_i ( 1/|R+_u|^{\alpha} \Sigma_{j\inR+_u\{i}} qj )     (3)
         |_______________________|
             用户u的表示
$$

其中\alpha是一个控制归一化效果的超参数，pi和qj分别表示itemi和j的嵌入向量。符号\{i}对应于方程（2）中diag(S)=0的约束，以避免对目标item的自相似度建模。

从基于用户的CF的角度来看，括号中的项可以看作是用户u的表示，它是从u的历史item的嵌入聚合而来的。注意在FISM中，每个item有两个嵌入向量p和q，以区分其作为预测目标或历史交互的角色，这也可以增加模型的表达力；评分项ruj被省略，因为FISM关注隐式反馈，其中对于j\inR+_u，ruj=1。给定方程（3）的良好定义的预测模型，我们可以通过优化推荐的标准损失函数（即，不使用SLIM中使用的item相似度约束）来学习模型参数，例如逐点分类损失[5]和成对回归损失[18]。

虽然FISM提供了基于item的CF方法中最先进的性能，但我们认为其表示能力可能受到其在获取用户表示时对所有历史item平等处理的限制。正如前文引言中所提到的，这一假设对于真实世界的数据来说是反直觉的，可能会降低模型的保真度。我们提出的NAIS模型通过使用神经注意力网络区分历史item的重要性来解决FISM的这一局限性。

---

## 3 神经注意力item相似度模型

在本节中，我们介绍我们提出的NAIS方法。在介绍NAIS模型之前，我们首先讨论几种试图解决FISM局限性的注意力机制设计。然后我们详细阐述模型参数的优化。我们重点讨论使用隐式反馈优化NAIS，这是近期推荐研究的焦点，因为隐式反馈比显式评分更普遍且更容易收集。最后，我们讨论NAIS的几个特性，包括时间复杂度、对在线个性化的支持以及注意力函数的选择。

### 3.1 模型设计

**设计1.** 注意力的原始思想是模型的不同部分可以对最终预测有不同的贡献（即关注）[19]。在基于item的CF场景中，我们可以直观地允许历史item通过为每个item分配一个个性化权重来对用户表示有不同贡献：

$$
ˆyui = pT_i ( 1/|R+_u|^{\alpha} \Sigma_{j\inR+_u\{i}} aj qj )     (4)
$$

其中aj是一个可训练参数，表示itemj在贡献用户表示时的注意力权重。显然，这个模型包含了FISM（当所有item的aj固定为1时可以恢复为FISM）。虽然这个模型似乎能够区分历史item的重要性，但它忽略了目标item对历史item的影响。特别是，我们认为，无论要预测哪个item，为历史item分配一个全局权重是不合理的。例如，当预测用户对一部浪漫电影的偏好时，将一部恐怖片视为与另一部浪漫片同等重要是不可取的。从用户表示学习的角度来看，它假设用户有一个静态向量来表示其兴趣，这可能限制了模型的表示能力。

**设计2.** 为了解决设计1的局限性，一个直观的解决方案是将aj调整为感知目标item的，即为每个(i, j)对分配一个个性化权重：

$$
ˆyui = pT_i ( 1/|R+_u|^{\alpha} \Sigma_{j\inR+_u\{i}} aij qj )     (5)
$$

其中aij表示当预测u对目标itemi的偏好时，itemj在贡献用户u的表示时的注意力权重。虽然这个解决方案在技术上似乎是可行的，但问题是如果一对item(i, j)在训练数据中从未共同出现过（即没有用户同时与i和j交互过），其注意力权重aij就无法被估计，将是一个平凡的数字。

**设计3.** 为了解决设计2的泛化问题，我们考虑将aij与嵌入向量pi和qj关联起来。其原理是嵌入向量应该编码item的信息，因此它们可以用来确定交互(i, j)的权重。具体来说，我们将aij参数化为以pi和qj为输入的函数：

$$
aij = f(pi, qj)     (6)
$$

这种参数化的好处是，即使一对item(i, j)从未共同出现过，只要pi和qj已经从数据中可靠地学习到，它们仍然可以用来很好地估计注意力权重aij。为了实现这个目标，我们需要确保函数f具有较强的表示能力。受最近使用神经网络建模注意力权重的成功启发[16, 15]，我们同样使用多层感知器（MLP）来参数化注意力函数f。具体来说，我们考虑两种定义注意力网络的方式：

$$
1. f_concat(pi, qj) = h^T ReLU(W [pi; qj] + b)     (7)
2. f_prod(pi, qj) = h^T ReLU(W (pi \odot qj) + b)
$$

其中W和b分别是将输入投影到隐藏层的权重矩阵和偏置向量，h^T是将隐藏层投影到输出注意力权重的向量。我们将隐藏层的大小称为"注意力因子"，其值越大，注意力网络的表示能力越强。我们使用修正线性单元（ReLU）作为隐藏层的激活函数，它在神经注意力网络中已被证明具有良好的性能[15]。在后面的第3.3节中，我们讨论两种注意力函数f_concat和f_prod的优缺点。

遵循神经注意力网络的标准设置[20, 16]，我们可以将设计3的预测模型公式化如下：

$$
ˆyui = pT_i ( \Sigma_{j\inR+_u\{i}} aij qj )     (8)

aij = exp(f(pi, qj)) / \Sigma_{j\inR+_u\{i}} exp(f(pi, qj))
$$

其中系数1/|R+_u|^\alpha被合并到注意力权重aij中而不影响表示能力，softmax函数用于将注意力权重转换为概率分布。注意这是将注意力网络应用于交互历史的最自然和最直接的方式，与注意力CF模型[16]的历史建模部分相同。

不幸的是，我们发现这种标准的注意力解决方案在实践中效果不佳——它显著不如FISM，尽管它在理论上可以泛化FISM。在研究了注意力权重后，我们意外地发现问题源于softmax函数，这是神经注意力网络中的一个标准选择。理由如下。在注意力的传统使用场景中，如CV和NLP任务，注意力组件的数量变化不大，例如句子中的单词[21]和图像中的区域[22, 23]。因此，使用softmax可以正确地对注意力权重进行归一化，并具有良好的概率解释。然而，这种场景在用户历史数据中不再存在，因为用户的历史长度（即用户消费的历史item数量）可能变化很大。定性地说，softmax函数对注意力权重执行L1归一化，这可能会过度惩罚具有长历史记录的活跃用户的权重。

为了证明这一点，我们在图2中展示了实验所用的MovieLens和Pinterest数据集的用户历史长度分布。我们可以看到，对于这两个真实世界数据集，用户的历史长度变化很大；具体来说，MovieLens和Pinterest的用户历史长度的（均值，方差）分别为(166, 37145)和(27, 57)。以MovieLens数据的左子图为例，所有用户的平均长度为166，而最大长度为2313。这意味着最活跃用户的平均注意力权重为1/2313，比普通用户的平均注意力权重（即1/166）少约14倍。注意力权重的这种巨大方差将导致模型item嵌入的优化问题。

**NAIS模型.** 我们现在给出NAIS模型的最终设计。如上所述，设计3性能不佳来自softmax，它对注意力权重执行L1归一化，导致不同用户的注意力权重具有较大方差。为了解决这个问题，我们提出平滑softmax的分母，以减轻对活跃用户注意力权重的惩罚，同时减少注意力权重的方差。形式上，NAIS的预测模型如下：

$$
ˆyui = pT_i ( \Sigma_{j\inR+_u\{i}} aij qj )     (9)

aij = exp(f(pi, qj)) / [ \Sigma_{j\inR+_u\{i}} exp(f(pi, qj)) ]^\beta
$$

其中\beta是平滑指数，是一个设置在[0, 1]范围内的超参数。显然，当\beta设为1时，它恢复为softmax函数；当\beta小于1时，分母的值会被抑制，因此活跃用户的注意力权重不会受到过度惩罚。虽然当\beta<1时注意力网络的概率解释被破坏了，但我们在经验上发现，使用\beta<1的性能远好于使用标准softmax（见第4.4节的实验结果）。我们使用术语"NAIS-concat"和"NAIS-prod"分别表示使用f_concat和f_prod作为注意力函数的NAIS模型（参见方程(7)）。

此外，我们的NAIS模型可以被视为在最近提出的神经协同过滤（NCF）框架[5]下，如图1所示。与使用独热用户ID作为输入特征的基于用户的NCF模型不同，我们的NAIS模型使用多热交互item作为用户的输入特征。结合精心设计的注意力网络作为隐藏层，我们的NAIS模型可以被更直观地理解为执行item到item的CF。

### 3.2 优化

为了学习推荐模型，我们需要指定一个目标函数进行优化。由于我们处理的是隐式反馈，其中每个条目是二进制值1或0，我们可以将推荐模型的学习视为一个二分类任务。与神经CF的前期工作[5]类似，我们将观察到的用户-item交互视为正实例，从剩余的未观察交互中采样负实例。设R+和R-分别表示正实例和负实例的集合，我们最小化定义如下的正则化对数损失：

$$
L = -1/N ( \Sigma_{(u,i)\inR+} log \sigma(ˆyui) + \Sigma_{(u,i)\inR-} log(1-\sigma(ˆyui)) ) + \lambda||Θ||²    (10)
$$

其中N表示总训练实例数，\sigma是将预测ˆyui转换为表示u将与i交互的可能性概率值的sigmoid函数。超参数\lambda控制L2正则化的强度以防止过拟合，Θ = {{pi}, {qi}, W, b, h}表示所有可训练参数。我们知道还有其他目标函数的选择，例如逐点回归损失[6, 24]和成对排序损失[11, 20]，也可以用于学习NAIS的隐式反馈。由于本文的重点是展示NAIS的有效性，特别是与FISM相比的改进以证明注意力使用的合理性，我们将其他目标函数的探索留给未来的工作。

为了优化目标函数，我们采用Adagrad[25]，这是一种随机梯度下降（SGD）的变体，为每个参数应用自适应学习率。它从所有训练实例中随机抽取样本，沿其梯度的负方向更新相关参数。我们使用小批量版本的Adagrad来加速训练过程，小批量的生成在第4.1节实验设置中详述。在每个训练轮次中，我们首先生成所有负实例，然后将它们与正实例一起输入训练算法进行参数更新。这比在GPU平台上训练时动态采样负实例（如贝叶斯个性化排序[11]中所做的那样）快得多，因为它避免了GPU（用于参数更新）和CPU（用于负采样）之间的不必要切换。具体来说，对于每个正实例(u, i)，我们随机采样u从未交互过的X个item作为负实例。在我们的实验中，我们将X设为4，这是一个经验数字，已在神经CF方法中表现出良好的性能[5]。

**预训练.** 由于神经网络的非线性和目标函数的非凸性（相对于所有参数），使用SGD的优化很容易陷入性能较差的局部最小值。因此，模型参数的初始化在模型的最终性能中起着至关重要的作用。经验上，当我们尝试从随机初始化训练NAIS时，我们发现它收敛缓慢，且最终性能仅略优于FISM。我们假设这是由于同时优化注意力网络和item嵌入的困难。由于注意力网络的输出重新缩放item嵌入，联合训练它们可能导致共适应效应，从而减慢收敛速度。例如，一个训练轮次可能减少注意力权重aij但增加嵌入乘积pT_i qj，导致更新预测分数时只取得微小进展。

为了解决训练NAIS中的实际问题，我们使用FISM预训练NAIS，用FISM学习到的item嵌入来初始化NAIS的item嵌入。由于FISM没有共适应问题，它可以很好地学习item嵌入以编码item相似度。因此，使用FISM嵌入初始化NAIS可以极大地促进注意力网络的学习，从而加快收敛速度并提高性能。在这种有意义的item嵌入初始化下，我们可以简单地用随机高斯分布初始化注意力网络。

### 3.3 讨论

在本小节中，我们讨论NAIS的三个特性：时间复杂度、支持在线个性化的简便性以及注意力函数的两种选择。

**时间复杂度分析.** 我们分析NAIS预测模型（即方程(9)）的时间复杂度。这直接反映了NAIS在测试（或推荐）中的时间成本，训练的时间成本应与测试成正比。使用FISM（参见方程(3)）评估预测ˆyui的时间复杂度为O(k|R+_u|)，其中k表示嵌入大小，|R+_u|表示用户u的历史交互数。与FISM相比，评估NAIS预测的额外成本来自注意力网络。设a为注意力因子，则评估f(pi, qj)的时间复杂度可表示为O(ak)。由于softmax（以及我们提出的平滑变体）的分母需要遍历R+_u中的所有item，评估一个aij的时间复杂度为O(ak|R+_u|)。因此，NAIS模型的直接实现需要O(ak|R+_u|²)的时间，因为我们需要对|R+_u|中的每个j评估aij。然而，考虑到分母项在R+_u中所有item的计算中是共享的，我们只需要计算一次并将其缓存用于所有aij的评估（其中j在R+_u中）。因此，评估NAIS预测的总体时间复杂度可以降低到O(ak|R+_u|)，这是FISM的a倍。

**对在线个性化的支持.** 推荐模型的离线训练基于用户过去的历史提供个性化推荐。对于在线个性化，我们考虑用户有新的交互流式进入的实际场景，推荐模型需要即时刷新用户的Top-K推荐[6, 26]。由于实时执行模型重训练是不可行的，一种替代方案是基于新反馈仅对模型参数执行局部更新。这是基于用户的CF模型使用的常见策略，例如矩阵分解[6]。然而，我们认为即使在实践中也很难实现参数的局部更新。关键困难在于用户可能对一个item有并发交互。因此，在每个交互基础上分别执行局部更新会导致冲突，并且在分布式环境中实时解决冲突并非易事。

与通过更新模型参数来适应新交互不同，NAIS可以在不更新任何模型参数的情况下刷新用户的表示向量，从而降低了提供在线个性化服务的难度。这归功于基于item的CF机制，它通过用户的交互历史而非其ID来刻画用户。具体在NAIS中，用户的表示向量通过item嵌入的加权求和聚合而成，这允许对预测进行良好的可分解评估。例如，假设用户u在itemt上有一个新交互。为了刷新u对候选itemi的预测（即ˆyui），我们不需要从头计算ˆyui（即遵循方程(9)），而只需要评估ait pT_i qt的分数，然后将其与ˆyui的旧预测相加。通过缓存softmax的分母，ˆyui的刷新可以在O(ak)时间内完成。这比使用MF[6]进行局部更新（时间复杂度为O(k²+|R+_u|k)）高效得多，因为a通常是一个小数字（通常设置为与k相同）。

**注意力函数的选择.** 注意力函数的两种选择在输入的构造上有所不同：第一种选择f_concat简单地拼接pi和qj来学习注意力权重wij[19]，而第二种选择f_prod将pi和qj的逐元素乘积馈入注意力网络[15]。从分析上讲，由于注意力权重wij是对交互pT_i qj的评分，使用逐元素乘积pi\odotqj作为输入可以促进隐藏层学习注意力函数（因为pT_i qj = 1^T(pi\odotqj)）；作为缺点，它也可能无意中导致一些信息损失，因为pi和qj中编码的原始信息被丢弃了。相比之下，f_concat利用pi和qj中编码的原始信息来学习它们的交互权重，没有信息损失；然而，由于拼接[pi, qj]^T与逐元素乘积pi\odotqj之间的数值差距，它可能导致收敛较慢。我们将在实验部分经验性地比较这两种注意力函数的选择。

---

## 4 实验

在本节中，我们进行实验，旨在回答以下研究问题：

**RQ1** 我们提出的注意力网络是否有助于提供更准确的推荐？
**RQ2** 我们提出的NAIS方法与最先进的推荐方法相比表现如何？
**RQ3** NAIS的关键超参数是什么，它们如何影响NAIS的性能？

接下来，我们首先介绍实验设置，然后给出回答上述问题的结果。

### 4.1 实验设置

**表1：评估数据集的统计数据**

| 数据集 | 交互数# | 训练数# | item数# | 用户数# |
|--------|---------|---------|---------|---------|
| MovieLens | 1,000,209 | 4,970,845 | 3,706 | 6,040 |
| Pinterest | 1,500,809 | 7,228,110 | 9,916 | 55,187 |

**数据集和评估协议.** 我们采用与NCF论文[5]中使用的相同的MovieLens和Pinterest数据集。由于两个数据集都经过了一些预处理步骤，例如移除稀疏用户和训练-测试分割，我们直接对处理后的数据进行评估。表1总结了两个数据集的统计数据。关于两个数据集生成的更多细节已在[5]中详细阐述，因此我们不再重述。注意在训练期间，每个交互与4个负实例配对，因此训练实例的数量远多于交互数量。

我们采用留一评估协议[11, 5]，该协议将每个用户的最新交互作为测试数据，并使用剩余的交互进行训练。具体来说，每个测试实例与99个随机采样的负实例配对；然后每个方法输出100个实例（1个正例加99个负例）的预测分数，性能由位置10的命中率（HR）[27]和归一化折现累计增益（NDCG）[28]评判。这两个度量在信息检索文献中已被广泛用于评估Top-K推荐[1]和排序系统[29]。我们报告所有用户的平均分数，其中HR@10可以解释为基于召回率的度量，表示成功推荐的用户百分比（即正实例出现在前10名中），NDCG@10是基于精确率的度量，考虑了正实例的预测位置，越大越好。

**基线方法.** 我们将NAIS与以下item推荐方法进行比较：

- **Pop.** 这是一种非个性化方法，用于基准测试Top-K推荐任务的性能。它根据item的流行度（由item收到的交互数判断）对item进行排序。
- **ItemKNN [10].** 这是如方程(1)中公式化的标准基于item的CF方法。我们使用余弦相似度来衡量sij。我们实验了不同的最近邻item数，发现使用所有邻居能带来最佳结果。
- **FISM [1].** 这是如方程(3)中公式化的最先进的基于item的CF模型。我们从0到1以0.1的步长测试\alpha，发现在两个数据集上值为0时结果最佳（当\alpha小于0.6时方差实际上很小）。
- **MF-BPR [11].** MF-BPR通过优化成对贝叶斯个性化排序（BPR）损失来学习MF。该方法是从隐式反馈构建CF推荐器的流行选择。
- **MF-eALS [6].** 该方法也学习MF模型，但优化了一个不同的逐点回归损失，将所有缺失数据视为权重较小的负反馈。优化通过逐元素交替学习平方（eALS）算法完成。
- **MLP [5].** 该方法在用户和item嵌入之上应用多层感知器（MLP）来从数据中学习评分函数。我们采用3层MLP并优化相同的逐点对数损失，据报道该方法在两个数据集上表现良好。

我们特意选择了上述方法以涵盖多样化的推荐方法：ItemKNN和FISM是基于item的CF方法的代表，以验证我们注意力增强建模的有效性；MF-BPR和MF-eALS是有竞争力的基于用户的CF方法，以证明隐式反馈推荐的最先进性能；MLP是最近提出的基于深度神经网络的CF方法。注意我们关注的是单一CF模型的比较。因此，我们不再与NeuMF进行比较，后者达到了NCF论文中的最佳性能，因为NeuMF是在潜在空间中融合了MF和MLP的集成方法。

**参数设置.** 对于每个方法，我们首先在没有正则化的情况下训练它；如果观察到过拟合（即训练损失持续下降但性能变差），我们则在[10^{-6}, 10^{-5}, ..., 1]范围内调整正则化系数\lambda。验证集由每个用户随机抽取的一个交互组成。对于嵌入大小k，我们测试[8, 16, 32, 64]的值，并将注意力因子a在每个设置中设置为与嵌入大小相同。为了与FISM进行公平比较，我们使用相同的逐点对数损失和相同的Adagrad学习器对其进行优化。我们发现使用FISM学习到的item嵌入来初始化NAIS（即预训练步骤）可以带来略好的性能但更快的收敛速度。除非特别说明，我们报告NAIS在以下默认设置下的性能：1) \beta=0.5，2) k=a=16，3) \lambda=0，4) Adagrad学习率为0.01，5) 使用FISM嵌入预训练。

**实现细节.** 我们使用TensorFlow实现NAIS。由于在输入层中，一个item（用户）被表示为一个独热（多热）向量，其中大多数条目为零，出于效率和内存考虑，我们采用稀疏表示，仅存储非零条目的ID。这里的一个实现挑战是不同用户具有不同数量的非零条目，而TensorFlow要求批次中的所有训练实例必须具有相同的长度（与其他深度学习编程工具如Theano相同）。为了解决这一挑战，一个广泛采用的解决方案是使用掩码技巧，即添加掩码（即伪非零条目）以确保批次的所有实例具有相同的长度（即批次实例的最大长度）。然而，我们发现这个解决方案在CF数据集上非常耗时，因为一些活跃用户可能交互过数千个item，使得采样的小批量非常大。为了解决这个问题，我们创新地将小批量构建为一个随机采样的用户的所有训练实例，而不是随机采样固定数量的训练实例作为小批量。这种基于用户的小批量技巧有两个优点：1）不使用掩码，因此速度快得多（经验上比掩码技巧快3倍），2）不需要指定批次大小，免去了调整批次大小的痛苦。此外，根据我们的实验，推荐性能保持不变。

**表2：使用TensorFlow实现的方法每轮训练时间（秒）**

| 方法 | MovieLens | Pinterest |
|------|----------|----------|
| MF-BPR | 24.4 s | 17.3 s |
| MLP | 125.8 s | 155.8 s |
| FISM | 238.3 s | 353.3 s |
| NAIS-concat | 455.2 s | 525.6 s |
| NAIS-prod | 428.5 s | 485.2 s |

**训练时间.** 表2显示了NAIS和用TensorFlow实现的基线方法每轮训练时间。一个训练轮次定义为训练5|R+|个实例，因为负采样比率为4。运行环境是配备Intel Xeon CPU E5-2630 @ 2.20GHz和64GB内存的服务器。注意ItemKNN和MF-eALS的运行时间未显示，因为它们是用Java实现的，与其他方法不可比。我们可以看到基于item的CF方法（FISM和NAIS）比基于用户的CF方法（MF-BPR和MLP）花费更长的训练时间。这是合理的，因为基于用户的方法在输入层只用一个ID来表示用户，而基于item的方法使用交互过的item来表示用户。MLP比MF-BPR花费更多时间，因为它比MF-BPR多了三个隐藏层。此外，两种NAIS方法比FISM花费更长时间，这是由于额外使用了注意力网络。额外的时间成本是相当可接受的，大约是FISM训练时间的0.8倍。在两种NAIS方法中，NAIS-concat比NAIS-prod花费略长的时间，因为拼接增加了输入维度而乘积没有。

### 4.2 注意力网络的有效性（RQ1）

从技术上讲，我们的NAIS模型通过用一个由注意力网络学习的可变权重替换常数权重（即1/|R+_u|^\alpha）来增强FISM，该常数权重与估计的item-item相似度（即pT_i qj）相乘。为了证明我们设计的注意力网络的有效性，我们首先运行FISM直至收敛，然后使用FISM嵌入初始化NAIS以训练注意力网络。

图3显示了FISM的稳定性能以及我们的两种NAIS方法在嵌入大小为16时每个轮次的分数。我们可以清楚地看到使用注意力网络的有效性。具体来说，NAIS的初始化性能接近FISM，而通过训练注意力网络，两种NAIS方法显著优于FISM。这里我们只展示了50轮的性能，进一步训练NAIS可以带来更好的性能。在收敛时（结果可见表5），两种NAIS方法在MovieLens和Pinterest上分别在NDCG方面比FISM实现了6.3%和3.6%的相对提升。我们相信推荐准确性的改进源于NAIS强大的表示能力。此外，我们发现NAIS-prod比NAIS-concat收敛得更快（而它们的最终性能相近）。这印证了我们在第3.3节中的分析，通过提供经验证据表明将pi\odotqj馈入注意力网络可以促进学习pT_i qj的权重。

#### 4.2.1 定性分析

这里我们提供一些关于注意力权重的定性分析，以展示它们的可学习性和可解释性。

首先，观察注意力权重在训练过程中如何演化是很有趣的。然而，一个ˆyui的预测有|R+_u|个注意力权重，很难绘制所有预测的注意力权重。相反，我们记录注意力权重的统计量——均值和方差，注意力网络的有效学习由较大的方差来证明（注意FISM的方差为0）。图4显示了NAIS-prod在不同轮次学习到的注意力权重的散点图，其中每个散点表示Pinterest中一个测试实例的预测。我们可以看到，在训练的初始阶段（第1轮），点集中在x轴附近，即方差接近于零。随着更多训练轮次，点沿y轴变得更加分散，许多点开始获得高方差。结合图3显示更多训练轮次导致更好的性能，我们可以得出结论：注意力权重已经被适当训练，对历史item更具区分度。这揭示了NAIS优于FISM的原因，证明了本工作的关键论点：用户的历史item在预测中并非平等贡献。

**表3：Pinterest中目标item#1382上采样用户的注意力权重分解。该用户有四个历史item，分别显示在第1到4列，最后一列表示预测分数（经过sigmoid）。**

| itemID | FISM | NAIS-prod |
|-------|------|-----------|
| #131 | 0.25 | 0.03 |
| #894 | 0.25 | 0.52 |
| #1534 | 0.25 | 0.22 |
| #3157 | 0.25 | 0.23 |
| \sigma(ˆyui) | 0.17 | 0.81 |

其次，我们在表3中展示了一个采样用户预测的注意力权重案例研究。权重已经过L1归一化，以便与FISM进行清晰比较，FIS假设历史item上的均匀权重。在这个例子中，目标item#1382是测试集中的一个正例，应该得到更高的分数。我们可以看到，FISM对所有历史item（更准确地说，它们与目标item的交互）均匀加权，导致相对较小的预测分数。相比之下，NAIS-prod对item#894分配更高的权重，对item#131分配更低的权重，成功地将目标item#1382评分更高，这正是期望的结果。为了证明其合理性，我们进一步调查了这些item的内容（即Pinterest图像）。我们发现目标item#1382和注意力最高的item#894都是关于自然风景的，而注意力最低的item#131是一张家庭照片。这是符合预期的，因为当预测用户对目标item的偏好时，她同一类别的历史item应该比其他不太相关的item具有更大的影响。这很好地印证了我们引言中的动机示例，提供了注意力权重与item特征相关性的证据。

#### 4.2.2 预训练的效果

**表4：嵌入大小为16时，使用（w/）和不使用（w/o）FISM预训练的NAIS方法性能**

| 方法 | MovieLens HR | MovieLens NDCG | Pinterest HR | Pinterest NDCG |
|------|-------------|----------------|-------------|----------------|
| FISM | 66.47 | 39.49 | 87.40 | 55.22 |
| NAIS-concat w/o 预训练 | 67.77 | 40.41 | 87.90 | 56.23 |
| NAIS-concat w/ 预训练 | 69.72 | 41.96 | 88.44 | 57.20 |
| NAIS-prod w/o 预训练 | 68.04 | 40.55 | 87.90 | 56.04 |
| NAIS-prod w/ 预训练 | 69.69 | 41.94 | 88.44 | 57.22 |

为了证明预训练（即使用FISM学习到的嵌入作为模型初始化）的效果，我们在表4中展示了嵌入大小为16时NAIS在有和没有预训练的情况下的性能。注意没有预训练的NAIS的超参数已经被分别调整。可以看出，通过使用FISM嵌入预训练两种NAIS方法，两种方法都得到了显著改善。除了性能提升外，有预训练的NAIS方法比随机初始化的收敛速度更快。这指出了使用FISM嵌入初始化NAIS的积极效果。此外，从头训练NAIS也能获得比FISM更好的性能，这进一步验证了注意力网络的有用性。

### 4.3 性能比较（RQ2）

我们现在比较NAIS与其他item推荐方法的性能。对于这些基于嵌入的方法（MF、MLP、FISM和NAIS），嵌入大小控制着它们的建模能力；因此，为了公平比较，我们将所有方法的嵌入大小设为16。在后续第4.4节的超参数研究中，我们为每个方法变化嵌入大小。表5显示了总体推荐准确性。我们有以下几个主要观察结果。

**表5：嵌入大小为16时比较方法的推荐准确率分数（%）**

| 方法 | MovieLens HR | MovieLens NDCG | Pinterest HR | Pinterest NDCG |
|------|-------------|----------------|-------------|----------------|
| Pop | 45.36 | 25.43 | 27.39 | 14.09 |
| ItemKNN | 62.27 | 35.87 | 78.57 | 48.32 |
| MF-BPR | 66.64 | 39.73 | 86.90 | 54.01 |
| MF-eALS | 67.88 | 39.83 | 87.13 | 52.55 |
| MLP | 68.41 | 41.03 | 86.48 | 53.85 |
| FISM | 66.47 | 39.49 | 87.40 | 55.22 |
| NAIS-concat | 69.72 | 41.96 | 88.44 | 57.20 |
| NAIS-prod | 69.69 | 41.94 | 88.44 | 57.22 |

- 1. 两种NAIS方法在两个数据集上均获得了最高的NDCG和HR分数。它们达到了相同的性能水平，显著优于其他方法（单样本配对t检验p<10^{-3}）。我们相信这些收益归功于注意力网络在学习item到item交互方面的有效设计。
- 2. 基于学习的CF方法性能优于基于启发式的方法Pop和ItemKNN。特别是，FISM相比其对应方法ItemKNN有约10%的相对提升。考虑到这两种方法使用相同的预测模型而在估计item相似度的方式上不同，我们可以清楚地看到针对推荐的定制优化的积极效果。
- 3. 在基线方法中，基于用户的CF模型（MF、MLP）和基于item的CF模型（FISM）之间没有明显的赢家。具体来说，在MovieLens上，基于用户的模型优于FISM，而在Pinterest上，FISM优于基于用户的模型。由于Pinterest数据的用户交互更加稀疏，这表明基于item的CF可能对稀疏数据集更有优势，这与先前工作的发现一致[1]。

值得指出的是，表5中报告的NAIS性能使用了默认的超参数设置（见第4.1节）。通过调整超参数可以观察到进一步的改进，这将在下一小节中探讨。

### 4.4 超参数研究（RQ3）

通过引入注意力网络，NAIS有两个额外的超参数——注意力网络的隐藏层大小（又称注意力因子a）和平滑指数\beta。此外，作为基于嵌入的模型，嵌入大小是NAIS的另一个关键超参数。本小节研究了这三个超参数的影响。

**表6：嵌入大小为8、32和64时基于嵌入的方法的推荐准确率分数（%）。每个设置的最佳性能以粗体突出显示。**

| 方法 | 嵌入大小=8 MovieLens HR/NDCG | 嵌入大小=8 Pinterest HR/NDCG | 嵌入大小=32 MovieLens HR/NDCG | 嵌入大小=32 Pinterest HR/NDCG | 嵌入大小=64 MovieLens HR/NDCG | 嵌入大小=64 Pinterest HR/NDCG |
|------|-----|-----|-----|-----|-----|-----|
| MF-BPR | 62.86/36.08 | 85.85/53.26 | 68.54/41.14 | 86.34/54.54 | 68.97/41.91 | 85.80/54.58 |
| MF-eALS | 62.80/36.35 | 86.26/51.86 | 70.40/42.16 | 86.75/53.84 | 70.35/43.50 | 85.77/53.77 |
| MLP | 67.10/39.98 | 85.90/53.67 | 69.24/42.51 | 86.77/54.20 | 70.18/42.64 | 86.90/54.50 |
| FISM | 61.71/35.73 | 87.03/54.82 | 69.29/41.71 | 88.43/57.13 | 70.17/42.82 | 88.62/57.18 |
| NAIS-concat | 64.17/37.36 | 87.44/55.27 | 70.83/43.36 | 88.56/57.47 | 71.66/44.15 | 88.74/57.75 |
| NAIS-prod | 64.50/37.60 | 87.88/55.75 | 70.91/43.39 | 88.67/57.59 | 71.82/44.18 | 88.84/57.90 |

表6显示了嵌入大小为8、32和64时基于嵌入的方法的性能。我们可以看到性能趋势总体上与嵌入大小为16时的观察结果一致（在第4.3节中详细说明）。我们的NAIS方法在大多数情况下达到了最佳性能，唯一的例外是嵌入大小8时，MLP表现最佳。这是因为当嵌入大小较小时，线性模型受到小嵌入大小的限制，而非线性模型比线性模型更容易表达更强的表示能力。

图5显示了NAIS关于注意力因子的性能。我们可以看到，无论注意力因子的设置如何，两种NAIS方法都优于FISM。在这两种方法中，NAIS-prod在小注意力因子下优于NAIS-concat，证明了使用pi\odotqj作为注意力网络输入来学习pT_i qj权重的积极效果。此外，为NAIS-concat使用较大的注意力因子可以弥补与NAIS-prod的性能差距。这暗示了使用表达力强的模型来学习注意力权重的作用。

图6显示了NAIS关于\beta的性能。很明显，当\beta小于1时，两种NAIS方法都表现出良好的性能并优于FISM。然而，当\beta设为1时，NAIS的性能显著下降并差于FISM。注意将\beta设为1意味着使用softmax对注意力权重进行归一化，这是神经注意力网络的标准设置[19, 15, 16]。不幸的是，这种标准设置对CF数据集效果不佳。我们相信原因是用户历史长度的巨大方差。具体来说，在MovieLens和Pinterest上，用户历史长度的（均值，方差）分别为(166, 37145)和(27, 57)。注意力组件数量的这种巨大方差在NLP和CV任务中很少发生，这些任务处理的是句子（即对单词的注意力）和图像（即对区域的注意力）。这是本工作将注意力网络应用于用户行为数据的关键见解，据我们所知，这在之前从未被研究过。

---

## 5 相关工作

早期的CF工作大多处理显式反馈如用户评分，将其公式化为评分预测任务[10, 30]。目标是最小化观察到的评分与相应模型预测之间的误差。对于这种基于回归的CF任务，MF——一种线性潜在因子模型——被认为是最有效的方法。其基本思想是将每个用户和item与一个潜在向量（又称嵌入）关联起来，将其匹配分数建模为它们潜在向量之间的内积。已经提出了许多MF的变体，如SVD++[30]、局部MF[31]、层次MF[32]、社交感知MF[26]和跨平台MF[33]。SVD++模型在拟合评分方面表现出了强大的表示能力；特别是，它被报告为Netflix挑战中最佳的单一模型。在我们看来，这应归功于它在潜在因子模型下对基于用户的CF和基于item的CF的集成。而在SVD++[30]的原始论文中，作者声称通过引入隐式反馈来增强MF，隐式反馈部分的建模本质上是一个基于item的CF模型。

后来的CF研究已经转向从隐式反馈学习推荐器[11, 1, 6, 8, 5]。从本质上讲，隐式反馈是单类数据，其中只记录用户的交互行为，而他们对item的显式偏好（即喜欢或不喜欢）是未知的。与预测评分分数的早期CF方法不同，隐式反馈方面的工作通常将CF视为个性化排序任务，采用基于排序的评估协议进行Top-K推荐。显然，用基于排序的协议评估CF方法更具说服力和实用价值，因为推荐本质上对许多应用来说是一个Top-K排序任务。此外，有经验证据表明，评分预测误差较低的CF模型并不一定在Top-K推荐中具有更高的准确性[34]。

从技术上讲，评分预测方法和Top-K推荐方法之间的关键区别在于优化CF模型的方式[6]。具体来说，评分预测方法通常仅优化观察数据上的回归损失，而Top-K推荐方法需要考虑缺失数据（又称负反馈）[34]。因此，只需调整目标函数进行优化，从技术上讲就可以将评分预测CF方法调整为适用于隐式反馈。

为了从隐式反馈学习推荐模型，两种类型的学习排序（L2R）目标函数已经被普遍应用：逐点和成对。逐点L2R方法要么优化基于回归的平方损失[1, 35]，要么优化基于分类的对数损失[5]，通过从缺失数据中采样负反馈[36]或将所有缺失数据视为负反馈[6]。对于线性CF模型如MF及其变体（例如因子分解机），存在高效的坐标下降算法可以优化所有缺失数据上的平方损失[6, 8]。然而对于复杂的非线性CF模型如神经网路，只有基于SGD的优化方法适用，并且为了效率需要从缺失数据中采样负反馈。成对L2R方法考虑用户的一个正反馈和（采样的）负反馈对，最大化它们的预测分数之间的间隔，而不考虑它们的精确值[11, 20]。其基本假设是观察到的交互应该比未观察到的反馈更可能引起用户的兴趣。一项最先进的工作开发了对抗个性化排序[37]，它在成对学习上使用对抗训练来增强推荐模型的鲁棒性并提高其泛化性能。

近年来，使用深度神经网络（DNN，又称深度学习）进行推荐变得越来越流行。DNN具有从数据中学习复杂函数的强大能力，以从低级原始数据（如图像和音频）中提取高级特征而闻名[13]。现有关于DNN用于推荐的工作可以分为两种类型：1）使用DNN从辅助数据中进行特征提取，如图像和文本[38, 39]，以及2）使用DNN学习用户-item评分函数[5, 40, 41]。由于我们关注的是仅利用用户-item交互的CF，第二类工作与本文更相关。在[5]中，作者提出了一个通用的NCF框架，用于使用前馈神经网络执行CF，并设计了三种基于用户的CF模型。后来NCF被扩展以整合属性并优化成对排序损失[18]。神经因子分解机（NFM）[40]被提出来建模特征之间的高阶和非线性交互，适用于信息丰富的推荐场景，如基于属性和上下文的推荐。最近，Wang等人[41]结合了基于嵌入的和基于树的模型的优势，用于可解释推荐。

与我们的工作最相似的是注意力协同过滤（ACF）[16]，它为基于用户的CF开发了一个注意力网络。我们的NAIS与ACF及所有先前工作的不同在于为基于item的CF定制了注意力网络。我们发现使用标准注意力网络在用户交互历史上效果不佳，原因在于有问题的softmax处理可变长度历史。为了解决这个问题，我们提出平滑softmax函数的分母。这一见解对于为长度方差较大的序列数据开发注意力网络特别有用，据我们所知，这在之前从未被探索过。

---

## 6 结论

在这项工作中，我们开发了用于item到item协同过滤的神经网络方法。我们的关键论点是用户画像的历史item并非平等地贡献于预测用户对某个item的偏好。为了解决这个问题，我们首先从表示学习的角度重新审视了FISM方法，然后逐步设计了几个注意力机制来增强其表示能力。我们发现，由于用户历史长度的巨大方差，传统的神经注意力网络设计[19, 16, 15, 17]对基于item的CF效果不佳。我们提出了一个简单而有效的softmax变体来解决用户行为上的大方差问题。我们进行了实证研究来验证我们的NAIS方法的有效性。实验结果表明，NAIS显著优于FISM，在item推荐任务中取得了有竞争力的性能。据我们所知，这是第一项为基于item的CF设计神经网路模型的工作，为未来神经推荐模型的发展开辟了新的研究可能性。在未来，我们特别感兴趣的是探索NAIS方法的深度架构。目前，我们的NAIS设计考虑了成对相似度，即仅考虑item之间的二阶交互，这是出于保持模型在在线个性化中简洁性的考虑。这主要是出于推荐方法的实际考虑。为了进一步提高推荐准确性，自然可以通过在嵌入层之上放置全连接层或卷积层来扩展NAIS，这已被证明通过建模高阶和非线性特征交互是有帮助的[40]。从技术上讲，另一个值得探索的有趣方向是将深度神经网络与基于图的方法[42, 43]相结合，它们具有独特的优势，并且在排序方面也被广泛使用。此外，我们有兴趣在基于item的CF上探索最近的对抗个性化排序学习，以研究可能的性能改进[37]。最后，我们将研究推荐系统的可解释性，这是一个近期有前景的方向[44, 28, 45, 41]，可以通过在基于item的CF方法中引入注意力网络来促进。

---

## 致谢

NExT研究由新加坡国家研究基金会、总理公署根据其IRC@SG资助计划支持。本工作部分受中国国家重点研发计划item2017YFB1401304、新加坡国家研究基金会下的AI Singapore计划、Linksure Network Holding Pte Ltd和亚洲大数据协会（item号：AISG-100E-2018-002）资助。

---

## 参考文献

[1] S. Kabbur, X. Ning, and G. Karypis, "FISM: Factored item similarity models for top-n recommender systems," in KDD, 2013, pp. 659–667.

[2] J. Davidson, B. Liebald, J. Liu, P. Nandy, T. Van Vleet, U. Gargi, S. Gupta, Y. He, M. Lambert, B. Livingston, and D. Sampath, "The youtube video recommendation system," in RecSys, 2010, pp. 293–296.

[3] C. A. Gomez-Uribe and N. Hunt, "The netflix recommender system: Algorithms, business value, and innovation," ACM Transactions on Management Information Systems, vol. 6, no. 4, pp. 13:1–13:19, 2015.

[4] D. C. Liu, S. Rogers, R. Shiau, D. Kislyuk, K. C. Ma, Z. Zhong, J. Liu, and Y. Jing, "Related pins at pinterest: The evolution of a real-world recommender system," in WWW Companion, 2017, pp. 583–592.

[5] X. He, L. Liao, H. Zhang, L. Nie, X. Hu, and T.-S. Chua, "Neural collaborative filtering," in WWW, 2017, pp. 173–182.

[6] X. He, H. Zhang, M.-Y. Kan, and T.-S. Chua, "Fast matrix factorization for online recommendation with implicit feedback," in SIGIR, 2016, pp. 549–558.

[7] H. Zhang, F. Shen, W. Liu, X. He, H. Luan, and T.-S. Chua, "Discrete collaborative filtering," in SIGIR, 2016, pp. 325–334.

[8] I. Bayer, X. He, B. Kanagal, and S. Rendle, "A generic coordinate descent framework for learning from implicit feedback," in WWW, 2017, pp. 1341–1350.

[9] B. Smith and G. Linden, "Two decades of recommender systems at amazon. com," IEEE Internet Computing, vol. 21, no. 3, pp. 12–18, 2017.

[10] B. Sarwar, G. Karypis, J. Konstan, and J. Riedl, "Item-based collaborative filtering recommendation algorithms," in WWW, 2001, pp. 285–295.

[11] S. Rendle, C. Freudenthaler, Z. Gantner, and L. Schmidt-Thieme, "BPR: Bayesian personalized ranking from implicit feedback," in UAI, 2009, pp. 452–461.

[12] X. Ning and G. Karypis, "SLIM: Sparse linear methods for top-n recommender systems," in ICDM, 2011, pp. 497–506.

[13] B. Yoshua, C. Aaron, and V. Pascal, "Representation learning: A review and new perspectives," IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 35, no. 8, pp. 1798–1828, 2013.

[14] C. Luo, B. Ni, S. Yan, and M. Wang, "Image classification by selective regularized subspace learning," IEEE Transactions on Multimedia, vol. 18, no. 1, pp. 40–50, 2016.

[15] J. Xiao, H. Ye, X. He, H. Zhang, F. Wu, and T.-S. Chua, "Attentional factorization machines: Learning the weight of feature interactions via attention networks," in IJCAI, 2017, pp. 3119–3125.

[16] J. Chen, H. Zhang, X. He, L. Nie, W. Liu, and T.-S. Chua, "Attentive collaborative filtering: Multimedia recommendation with item- and component-level attention," in SIGIR, 2017, pp. 335–344.

[17] J. Li, P. Ren, Z. Chen, Z. Ren, T. Lian, and J. Ma, "Neural attentive session-based recommendation," in CIKM, 2017, pp. 1419–1428.

[18] X. Wang, X. He, L. Nie, and T.-S. Chua, "Item silk road: Recommending items from information domains to social users," in SIGIR, 2017, pp. 185–194.

[19] D. Bahdanau, K. Cho, and Y. Bengio, "Neural machine translation by jointly learning to align and translate," in ICLR, 2015.

[20] Z. Zhao, B. Gao, V. W. Zheng, D. Cai, X. He, and Y. Zhuang, "Link prediction via ranking metric dual-level attention network learning," in IJCAI, 2017, pp. 3525–3531.

[21] A. P. Parikh, O. Täckström, D. Das, and J. Uszkoreit, "A decomposable attention model for natural language inference," in EMNLP, 2016, pp. 2249–2255.

[22] L. Chen, H. Zhang, J. Xiao, L. Nie, J. Shao, W. Liu, and T. Chua, "SCA-CNN: spatial and channel-wise attention in convolutional networks for image captioning," in CVPR, 2017, pp. 6298–6306.

[23] M. Wang, X. Liu, and X. Wu, "Visual classification by l1-hypergraph modeling," IEEE Transactions on Knowledge and Data Engineering, vol. 27, no. 9, pp. 2564–2574, 2015.

[24] M. Wang, W. Fu, S. Hao, H. Liu, and X. Wu, "Learning on big graph: Label inference and regularization with anchor hierarchy," IEEE transactions on knowledge and data engineering, vol. 29, no. 5, pp. 1101–1114, 2017.

[25] J. Duchi, E. Hazan, and Y. Singer, "Adaptive subgradient methods for online learning and stochastic optimization," Journal of Machine Learning Research, vol. 12, no. Jul, pp. 2121–2159, 2011.

[26] Z. Zhao, H. Lu, D. Cai, X. He, and Y. Zhuang, "User preference learning for online social recommendation," IEEE Transactions on Knowledge and Data Engineering, vol. 28, no. 9, pp. 2522–2534, 2016.

[27] M. Deshpande and G. Karypis, "Item-based top-n recommendation algorithms," ACM Transactions on Information Systems, vol. 22, no. 1, pp. 143–177, 2004.

[28] X. He, T. Chen, M.-Y. Kan, and X. Chen, "Trirank: Review-aware explainable recommendation by modeling aspects," in CIKM, 2015, pp. 1661–1670.

[29] X. He, M. Gao, M.-Y. Kan, and D. Wang, "Birank: Towards ranking on bipartite graphs," IEEE Transactions on Knowledge and Data Engineering, vol. 29, no. 1, pp. 57–71, 2017.

[30] Y. Koren, "Factorization meets the neighborhood: A multifaceted collaborative filtering model," in KDD, 2008, pp. 426–434.

[31] Y. Zhang, M. Zhang, Y. Liu, S. Ma, and S. Feng, "Localized matrix factorization for recommendation based on matrix block diagonal forms," in WWW, 2013, pp. 1511–1520.

[32] S. Wang, J. Tang, Y. Wang, and H. Liu, "Exploring implicit hierarchical structures for recommender systems," in IJCAI, 2015, pp. 1813–1819.

[33] D. Cao, X. He, L. Nie, X. Wei, X. Hu, S. Wu, and T.-S. Chua, "Cross-platform app recommendation by jointly modeling ratings and texts," ACM Trans. Inf. Syst., vol. 35, no. 4, pp. 37:1–37:27, 2017.

[34] P. Cremonesi, Y. Koren, and R. Turrin, "Performance of recommender algorithms on top-n recommendation tasks," in RecSys, 2010, pp. 39–46.

[35] M. Wang, W. Fu, S. Hao, D. Tao, and X. Wu, "Scalable semi-supervised learning by efficient anchor graph regularization," IEEE Transactions on Knowledge and Data Engineering, vol. 28, no. 7, pp. 1864–1877, 2016.

[36] X. Chen, Y. Zhang, Q. Ai, H. Xu, J. Yan, and Z. Qin, "Personalized key frame recommendation," in SIGIR, 2017, pp. 315–324.

[37] X. He, Z. He, X. Du, and T.-S. Chua, "Adversarial personalized ranking for recommendation," in SIGIR, 2018.

[38] X. Geng, H. Zhang, J. Bian, and T.-S. Chua, "Learning image and user features for recommendation in social networks," in ICCV, 2015, pp. 4274–4282.

[39] S. Wang, Y. Wang, J. Tang, K. Shu, S. Ranganath, and H. Liu, "What your images reveal: Exploiting visual contents for point-of-interest recommendation," in WWW, 2017, pp. 391–400.

[40] X. He and T.-S. Chua, "Neural factorization machines for sparse predictive analytics," in SIGIR, 2017, pp. 355–364.

[41] X. Wang, X. He, F. Feng, L. Nie, and T.-S. Chua, "Tem: Tree-enhanced embedding model for explainable recommendation," in WWW, 2018, pp. 1543–1552.

[42] R. Hong, C. He, Y. Ge, M. Wang, and X. Wu, "User vitality ranking and prediction in social networking services: A dynamic network perspective," IEEE Transactions on Knowledge and Data Engineering, vol. 29, no. 6, pp. 1343–1356, 2017.

[43] F. Feng, X. He, Y. Liu, L. Nie, and T.-S. Chua, "Learning on partial-order hypergraphs," in WWW, 2018, pp. 1523–1532.

[44] Y. Zhang, G. Lai, M. Zhang, Y. Zhang, Y. Liu, and S. Ma, "Explicit factor models for explainable recommendation based on phrase-level sentiment analysis," in SIGIR, 2014, pp. 83–92.

[45] Z. Ren, S. Liang, P. Li, S. Wang, and M. de Rijke, "Social collaborative viewpoint regression with explainable recommendations," in WSDM, 2017, pp. 485–494.

---

## 作者简介

**Xiangnan He** 目前是新加坡国立大学（NUS）计算机学院的高级研究学者。他在NUS获得计算机科学博士学位。他的研究兴趣涵盖推荐系统、信息检索和多媒处理。他在多个顶级会议上发表了20多篇论文，如SIGIR、WWW、MM、CIKM和IJCAI，以及期刊包括TKDE、TOIS和TMM。他在推荐系统方面的工作获得了ACM SIGIR 2016最佳论文荣誉提名。此外，他曾担任知名会议的PC成员，包括SIGIR、WWW、MM、AAAI、IJCAI、WSDM、CIKM和EMNLP，以及知名期刊的常规审稿人，包括TKDE、TOIS、TKDD、TMM等。

**Zhankui He** 是中国复旦大学的一名本科生，也是新加坡国立大学（NUS）的交换生。他是NUS媒体搜索实验室和NExT研究中心的研究助理。他曾获得2016年上海市奖学金、2016年复旦大学优秀学生奖和2017年东方CJ奖学金。他的研究兴趣包括推荐系统和计算机视觉。

**Jingkuan Song** 目前是电子科技大学的教授。他曾在哥伦比亚大学担任博士后研究科学家（2016-2017），在特伦托大学担任研究学者（2014-2016）。他于2014年在澳大利亚昆士兰大学（UQ）获得信息技术博士学位。他的研究兴趣包括大规模多媒体检索、图像/视频分割以及使用哈希、图学习和深度学习技术的图像/视频标注。

**Tat-Seng Chua** 是新加坡国立大学（NUS）计算机学院的KITHCT讲座教授。他曾在1998-2000年担任学院的代理创始院长。Chua博士的主要研究兴趣是多媒体信息检索和社交媒体分析。特别是，他的研究专注于从网络和多个社交网络中提取、检索和问答文本和富媒体。他是NExT的联合主任，这是一个NUS与清华大学之间的联合中心，旨在开发实时社交媒体搜索技术。Chua博士是2015年ACM SIGMM杰出技术贡献奖的获得者。他是ACM国际多媒体检索会议（ICMR）和多媒体建模（MMM）系列会议指导委员会的主席。Chua博士还担任过ACM Multimedia 2005、ACM CIVR（现ACM ICMR）2005、ACM SIGIR 2008和ACM Web Science 2015的总联合主席。他在四个国际期刊的编辑委员会任职。Chua博士是新加坡两家科技创业公司的联合创始人。他持有英国利兹大学的博士学位。

**Zhenguang Liu** 目前是新加坡科技研究局（A* STAR）的研究学者。他于2015年至2017年5月在新加坡国立大学担任研究学者。他分别于2010年和2015年在中国浙江大学和山东大学获得博士学位和学士学位。他的研究兴趣包括多媒数据分析和数据挖掘。他的工作各部分已发表在包括TIP、AAAI、MM、TMM、TOMM在内的第一梯队 venues。Liu博士曾担任ACM MM和MMM等会议的技术程序委员会成员，以及IEEE Transactions on Visualization and Computer Graphics、ACM MM、IEEE Transactions on Multimedia、Multimedia Tools and Applications等的审稿人。

**Yu-Gang Jiang** 是中国复旦大学计算机科学教授和上海市视频技术与系统工程研究中心主任。他的大数据视频分析实验室研究从大视频数据中提取高级信息的各个方面，如视频事件识别、物体/场景识别和大规模视觉搜索。他是世界范围内竞赛（如年度美国NIST TRECVID评估）中一些性能最佳的视频分析系统的主要架构师。他的工作获得了许多奖项，包括首届ACM中国新星奖、2015年ACM SIGMM新星奖以及国家自然科学基金优秀青年科学基金item。他还入选了中国国家万人计划和教育部长江学者计划。他目前是ACM TOMM、Machine Vision and Applications（MVA）和Neurocomputing的副编辑。他在香港城市大学获得计算机科学博士学位，在2011年加入复旦大学之前，曾在哥伦比亚大学工作三年。
