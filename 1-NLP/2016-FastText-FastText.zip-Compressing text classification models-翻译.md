# FastText.zip：压缩文本分类模型

> Armand Joulin, Edouard Grave, Piotr Bojanowski, Matthijs Douze, Hervé Jégou, Tomas Mikolov | Facebook AI Research

本文提出一套将文本分类模型压缩至极小体积的完整方案，使模型能部署在内存受限的设备上。核心发现是——**通过乘积量化（Product Quantization）结合判别式剪枝和哈希技巧，模型体积可压缩两个数量级（\times1000），而准确率仅下降不到 2%**。

核心内容：
- 文本分类模型（如 fastText）需要存储大量词嵌入矩阵，在词表和输出空间较大时占用数 GB 内存，难以部署到智能手机等内存受限设备
- 提出基于乘积量化（PQ）的嵌入压缩方案，将向量拆分为子向量并独立量化，配合归一化和再训练策略恢复精度
- 进一步通过判别式剪枝（基于嵌入范数的贪心最大覆盖算法）缩减词表，结合哈希 trick 和 Bloom filter 压缩字典存储
- 在 8 个文本分类基准和大规模 FlickrTag 数据集上全面验证，模型可压缩至 100KB 以下

关键发现：
- **NPQ（归一化乘积量化）在 $k = d/2$ 时可实现 \times10 压缩率且几乎无精度损失，再训练后精度甚至略超原始模型**
- 极端压缩至 64KiB（\times1000-4000 压缩率）时，平均精度仅下降 0.8%-1.7%
- 与字符级 CNN 相比，fastText+NPQ 在相同甚至更小的模型体积下精度更高，且训练速度快数个数量级
- 最大覆盖剪枝在 FlickrTag（31.2 万类别）上保持 88.4% 测试集覆盖率，而朴素剪枝仅覆盖 60%-70%

---

## 摘要

我们研究如何构建紧凑的文本分类架构，使完整模型能装入有限的内存量中。在参考了哈希文献中的多种方案后，我们提出一种基于乘积量化存储词嵌入的方法。原始技术会导致精度下降，我们对其进行了改进以消除量化伪影。在多个基准上的实验表明，我们的方法通常比 fastText 少用两个数量级的内存，而准确率仅有轻微下降。因此，在内存使用与准确率的权衡上，该方法大幅超越了现有最优水平。

## 1 引言

文本分类是自然语言处理中的重要问题。现实应用场景包括垃圾邮件过滤和电子邮件分类。它也是搜索和排序等更复杂系统的核心组件。近年来，基于神经网络的深度学习技术在各种 NLP 应用中取得了最优结果。深度学习的主要成功之一归功于循环网络在语言建模中的有效性及其在语音识别和机器翻译中的应用 [Mikolov, 2012]。然而，在其他场景包括若干文本分类问题中，深度网络并未令人信服地击败先前的最优技术 [Wang & Manning, 2012; Joulin et al., 2016]。

尽管神经网络的训练速度通常比基于 $n$-gram 的传统技术慢数个数量级，但由于模型体积紧凑——尤其是基于字符的模型——神经网络常被视为有前景的替代方案。这对需要在内存受限系统（如智能手机）上运行的应用尤为重要。

本文专门针对分类准确率与模型体积之间的权衡问题。我们扩展了之前在 fastText 库 [1] 中实现的工作。该方法基于 $n$-gram 特征、降维和 softmax 分类器的快速近似 [Joulin et al., 2016]。我们展示了几个关键要素——特征剪枝、量化、哈希和再训练——使我们能够生成体积极小的文本分类模型，在多个流行数据集上训练后通常不到 100KB，且不会明显牺牲准确率或速度。

我们计划将复现结果所需的代码和脚本作为 fastText 库的扩展发布，从而为优化模型体积与准确率权衡的文本分类器提供强可复现的基线。我们希望这能帮助工程社区通过使用更高效的模型来改进现有应用。

本文结构如下：第 2 节介绍相关工作，第 3 节描述我们的文本分类模型并解释如何大幅缩小模型体积。第 4 节通过多个文本分类基准的实验展示我们方法的有效性。

[1] https://github.com/facebookresearch/fastText

## 2 相关工作

**文本分类模型。** 文本分类是一个植根于多种应用的问题，如网络搜索、信息检索和文档分类 [Deerwester et al., 1990; Pang & Lee, 2008]。线性分类器通常在保持可扩展性的同时获得最优性能 [Agarwal et al., 2014; Joachims, 1998; Joulin et al., 2016; McCallum & Nigam, 1998]。当与合适的特征结合时，它们尤其具有吸引力 [Wang & Manning, 2012]。这些方法通常需要存储词和 $n$-gram 的嵌入，导致内存效率低下。

**语言模型压缩。** 我们的工作与统计语言模型的压缩相关。经典方法包括基于熵的特征剪枝 [Stolcke, 2000] 和量化。剪枝旨在仅保留模型中最重要的 $n$-gram，剔除概率低于指定阈值的部分。此外，单个 $n$-gram 可以通过量化概率值来压缩，$n$-gram 本身的存储也可以比字符序列更高效。已有多种策略被开发出来，例如使用树结构或哈希函数，详见 [Talbot & Brants, 2008]。

**相似度估计与搜索的压缩。** 关于如何将一组向量压缩为紧凑编码以使两个编码的比较近似原始空间中的目标相似度，已有大量文献。这些方法的典型用例是：给定一个压缩向量的索引数据集和一个查询，我们要在索引集中找到最近邻。其中最流行的方法之一是 Charikar [2002] 提出的局部敏感哈希（LSH），这是一种基于随机投影的二值化技术，通过两个对应二值编码之间汉明距离的单调函数来近似两个向量之间的余弦相似度。在本文中，LSH 指的是这种二值化策略 [2]。许多后续工作改进了这一初始二值化技术，如谱哈希 [Weiss et al., 2009] 或迭代量化（ITQ）[Gong & Lazebnik, 2011]，后者学习一个旋转矩阵以最小化二值化的量化损失。读者可参阅 Wang et al. [2014] 和 Wang et al. [2015] 的两篇综述了解二值哈希文献。

除了这些二值化策略之外，源自 Jegou et al. [2011] 的更通用量化技术在内存与距离估计器近似之间提供了更好的权衡。乘积量化（PQ）方法通过在压缩域中计算量化近似之间的距离来近似距离。该方法在统计上保证以与量化误差直接相关的误差界保留向量之间的欧氏距离。原始 PQ 已被 Ge et al. [2013] 和 Norouzi & Fleet [2013] 同时改进，他们学习一个正交变换以最小化整体量化损失。在本文中，我们将考虑优化乘积量化（OPQ）变体 [Ge et al., 2013]。

**Softmax 近似。** 上述工作近似的是欧氏距离或余弦相似度（在单位范数向量的情况下两者等价）。然而，在 fastText 的场景中，我们特别关注近似 softmax 层中涉及的最大内积。近年来已有若干源自 LSH 的方法被提出以实现这一目标，如 Shrivastava & Li [2014] 的非对称 LSH，随后被 Neyshabur & Srebro [2015] 讨论。在我们的工作中，由于不受纯二值编码的约束，我们采用更传统的编码方式，对向量使用幅度/方向参数化。因此我们只需要编码/压缩一个单位 $d$ 维向量，这与上述 LSH 和 PQ 方法非常契合。

**神经网络压缩模型。** 近来，已有若干研究致力于压缩计算机视觉中架构的参数，即用于最先进的卷积神经网络（CNN）[Han et al., 2016; Lin et al., 2015]。一些使用向量量化 [Gong et al., 2014]，另一些则对网络进行二值化 [Courbariaux et al., 2016]。Denil et al. [2013] 表明此类分类模型很容易被压缩，因为它们是过度参数化的，这与 LeCun et al. [1990] 的早期观察一致。

[2] 在文献中，LSH 指的是与 Johnson-Lindenstrauss 引理相关的多种不同策略。例如，LSH 有时指一种基于随机投影的分区技术，通过单元探测实现亚线性搜索，参见 Datar et al. [2004] 的 E2 LSH 变体。

这些工作中的一些同时致力于减小模型体积和提升速度。而在我们的场景中，由于构建在其上的 fastText 分类器已经非常高效，我们主要关注在保持相当分类效率的同时减小模型体积。

## 3 提出的方法

### 3.1 文本分类

在线性文本分类的背景下，线性分类器 [Joulin et al., 2016] 与更复杂的深层模型相比仍具竞争力，且训练速度更快。在线性文本分类常用的标准技巧之上 [Agarwal et al., 2014; Wang & Manning, 2012; Weinberger et al., 2009]，Joulin et al. [2016] 使用低秩约束来减少计算负担，同时在不同类别之间共享信息。这在输出空间很大、稀有类别可能只有少量训练样本的情况下尤为有用。在本文中，我们关注类似的模型，即最小化 $N$ 个文档上的 softmax 损失 $\ell$：

$$
\sum_{n=1}^{N} \ell(y_n, B A x_n) \qquad (1)
$$

其中 $x_n$ 是 one-hot 向量的词袋，$y_n$ 是第 $n$ 个文档的标签。在词表和输出空间很大的情况下，矩阵 $A$ 和 $B$ 很大，可能需要数 GB 内存。下面我们描述如何减少这种内存使用。

### 3.2 自底向上乘积量化

乘积量化是压缩域近似最近邻搜索的常用方法 [Jegou et al., 2011]。作为一种压缩技术，它通过在预定义的结构化质心集合（称为码本）中找到最近向量来近似实值向量。这个码本不会被枚举，因为它极其庞大。相反，它通过其结构隐式定义：一个 $d$ 维向量 $x \in \mathbb{R}^d$ 被近似为

$$
\hat{x} = \sum_{i=1}^{k} q_i(x) \qquad (2)
$$

其中不同的子量化器 $q_i: x \mapsto q_i(x)$ 是互补的，即它们各自的质心位于不同的正交子空间中，即 $\forall i \neq j, \forall x, y, \langle q_i(x) | q_j(y) \rangle = 0$。在原始 PQ 中，子空间与自然轴对齐，而 OPQ 学习一个旋转，相当于放宽这一约束并使其不依赖于原始坐标系。另一种理解方式是将 PQ 视为将给定向量 $x$ 拆分为 $k$ 个子向量 $x_i$，$i = 1 \ldots k$，每个维度为 $d/k$：$x = [x_1 \ldots x_i \ldots x_k]$，并使用不同的 $k$-means 量化器分别量化每个子向量。每个子向量 $x_i$ 因此被映射到 $2^b$ 个质心中最近的一个，其中 $b$ 是存储子量化器量化索引所需的比特数，通常 $b = 8$。重建向量可取 $2^{kb}$ 个不同的复制值，以 $kb$ 比特存储。

PQ 在压缩域中估计内积为

$$
x^\top y \approx \hat{x}^\top y = \sum_{i=1}^{k} q_i(x_i)^\top y_i \qquad (3)
$$

这是 Jegou et al. [2011] 的平方 L2 距离估计的直接扩展。在实践中，向量估计 $\hat{x}$ 可以从编码（即量化索引）中通过拼接这些质心轻松重建。

PQ 涉及的两个参数——子量化器数量 $k$ 和每个量化索引的比特数 $b$——通常设为 $k \in [2, d/2]$，$b = 8$ 以确保字节对齐。

**讨论。** PQ 在我们的文本分类场景中提供了几个有趣的特性。首先，训练非常快，因为子量化器的质心数量很少，即 $b = 8$ 时为 256 个质心。其次，在测试时它允许以几乎零计算和内存开销重建向量。第三，它已在计算机视觉中成功应用，比二值编码提供了好得多的性能，这使其成为压缩相对较浅模型的自然选择。如 Sánchez & Perronnin [2011] 所观察，在最后一层之前使用 PQ 与支持向量机结合时，精度损失非常有限。

在文本分类的背景下，向量的范数分布范围很广，最大值与最小值之间的比率通常为 1000。因此 $k$-means 表现不佳，因为它优化绝对误差目标，会将所有低范数向量映射到 0。一个简单的解决方案是将向量的范数和角度分离并分别量化。这允许无性能损失的量化，但每个向量需要额外 $b$ 比特。

**自底向上策略：再训练。** 最早旨在压缩 CNN 模型的工作 [Gong et al., 2014] 使用现成 PQ 的重建，即不做任何再训练。然而，如 Sablayrolles et al. [2016] 所观察，当使用 PQ 等量化方法时，最好再训练量化之后的层，以便网络可以重新适应量化。有一个有力的论据支持这种再训练策略：对于任何满足 Lloyd 条件的量化器，向量的平方幅度平均减少量等于平均量化误差；详见 Jegou et al. [2011]。

这暗示了一种自底向上的学习策略：我们首先量化输入矩阵，然后重新训练并量化输出矩阵（输入矩阵被冻结）。第 4 节的实验表明采用这种策略是值得的。

**PQ 的内存节省。** 在实践中，自底向上 PQ 策略提供 \times10 的压缩率且无明显性能损失。不做再训练时，我们观察到 0.1% 到 0.5% 的精度下降，具体取决于数据集和设置；详见第 4 节和附录。

### 3.3 更多文本特定技巧

内存使用量强烈依赖于词表大小，在许多文本分类任务中词表可能很大。虽然很明显词表的大部分是无用或冗余的，但直接将词表缩减为最常见的词并不令人满意：大多数高频词如"the"或"is"不具有区分性，而一些稀有词反而有区分性，例如在标签预测的场景中。在本节中，我们讨论几种减少字典所占空间的启发式方法。它们带来了大幅内存缩减，在极端情况下可达 \times100。我们通过实验表明这种大幅缩减与 PQ 压缩方法是互补的，意味着两种策略的结合在某些数据集上可将模型体积缩减至 \times1000。

**词表剪枝。** 发现哪些词或 $n$-gram 必须保留以维持整体性能是一个特征选择问题。虽然已有许多方法被提出用于在训练期间选择变量组 [Bach et al., 2012; Meier et al., 2008]，但我们关注的是从预训练模型中选择一个固定的 $K$ 个词和 $n$-gram 的子集。这可以通过选择能最大程度保留模型的 $K$ 个嵌入来实现，可归结为选择与最高范数关联的 $K$ 个词和 $n$-gram。

虽然这种方法提供了大幅内存节省，但在某些特定情况下有一个缺点：某些文档可能不包含这 $K$ 个最佳特征中的任何一个，导致性能显著下降。因此，在保留 $K$ 个最佳特征的同时确保它们覆盖整个训练集是很重要的。更形式化地说，问题是在特征集 $V$ 中找到一个子集 $S$，在约束所有训练集 $D$ 中的文档都被覆盖的条件下最大化其范数之和 $w_s$：

$$
\max_{S \subseteq V} \sum_{s \in w_s} w_s \quad \text{s.t.} \quad |S| \leq K, \quad P \mathbf{1}_S \geq \mathbf{1}_D
$$

其中 $P$ 是一个矩阵，当第 $s$ 个特征出现在第 $d$ 个文档中时 $P_{ds} = 1$，否则为 0。这个问题与 NP 困难的集合覆盖问题直接相关 [Feige, 1998]。标准的贪心方法需要存储倒排索引或对数据集进行多次遍历，这在非常大的数据集上是不可行的 [Chierichetti et al., 2010]。该问题可以作为带有秩约束的在线子模最大化问题来处理 [Badanidiyuru et al., 2014; Bateni et al., 2010]。在我们的场景中，我们使用一种简单的在线可并行化贪心方法：对于每个文档，我们检查它是否已被保留的特征覆盖，如果没有，则将范数最高的特征添加到保留特征集中。如果特征数量低于 $K$，则添加尚未被选中的范数最高的特征。

**哈希 trick 和 Bloom filter。** 在小模型上，字典可能占据内存的很大一部分。我们不保存字典，而是将 Joulin et al. [2016] 中使用的哈希 trick 扩展到词和 $n$-gram。这种策略也被 Vowpal Wabbit [Agarwal et al., 2014] 在在线训练的场景中使用。这使我们节省约 1-2MB，测试时几乎零开销（仅需计算哈希函数的代价）。

在使用哈希 trick 的同时进行词表剪枝需要保留 $K$ 个剩余桶的索引列表。测试时需要对索引列表进行二分搜索，复杂度为 $O(\log(K))$，内存开销为几百 KB。使用 Bloom filter 可将测试时复杂度降为 $O(1)$ 并节省几百 KB。然而，在实践中它会降低性能。

## 4 实验

本节评估我们模型压缩流程的质量，并将其与其他压缩方法在不同文本分类问题上进行比较，同时也与其他紧凑文本分类器进行比较。

**评估协议和数据集。** 我们的实验流程如下：除非另有说明，我们使用 fastText 的默认设置训练模型。即 200 万个桶、学习率 0.1 和 10 个训练轮次。嵌入的维度 $d$ 设为 2 的幂次以避免可能使结果解读更困难的边界效应。作为基线，我们使用局部敏感哈希（LSH）[Charikar, 2002]、PQ [Jegou et al., 2011] 和 OPQ [Ge et al., 2013]（非参数变体）。注意我们使用了 LSH 的改进版本，用随机正交矩阵代替随机矩阵投影 [Jégou et al., 2008]。在第一组实验中，我们使用 Zhang et al. [2015] 的 8 个数据集和评估协议。这些数据集包含数百万个文档，最多有 10 个类别。我们还在一个输出空间极大的数据集上探索量化的极限，即从 YFCC100M 集合 [Thomee et al., 2016] [3] 中提取的标签数据集，在本文其余部分称为 FlickrTag。

[3] 数据可在 https://research.facebook.com/research/fasttext/ 获取。


![图1](/Users/dazhang/PycharmProject/Papers/1-NLP/.picture/2016-FastText-FastText.zip-Compressing text classification models-fig1.png)

图 1：在 Zhang et al. [2015] 的 3 个数据集上，准确率作为每个向量/嵌入内存的函数。注意，当我们显式编码范数（"norm"）时需要额外一个字节。

### 4.1 小数据集

**压缩技术。** 我们在 Zhang et al. [2015] 发布的数据集上比较了三种用于紧凑编码相似度估计的流行方法：LSH、PQ 和 OPQ。图 1 展示了准确率作为每个嵌入使用字节数的函数，在 PQ 和 OPQ 的情况下对应子向量数 $k$。更多结果见附录。如第 2 节所讨论，LSH 重现余弦相似度，因此不适合未归一化的数据。因此我们仅报告归一化的结果。归一化后，PQ 和 OPQ 即使仅使用 $k = 4$ 个子量化器（等效地，字节）也几乎是无损的。我们在实践中观察到使用 $k = d/2$（即嵌入维度的一半）效果良好。在本文其余部分，如无特别说明，我们关注这一设置。PQ 和 OPQ 的归一化版本之间差异有限且取决于数据集。因此我们在本研究的其余部分采用归一化 PQ（NPQ），因为它训练更快。

| 词 | 熵排名 | 范数排名 | 词 | 熵排名 | 范数排名 |
|---|---|---|---|---|---|
| . | 1 | 354 | mediocre | 1399 | 1 |
| , | 2 | 176 | disappointing | 454 | 2 |
| the | 3 | 179 | so-so | 2809 | 3 |
| and | 4 | 1639 | lacks | 1244 | 4 |
| i | 5 | 2374 | worthless | 1757 | 5 |
| a | 6 | 970 | dreadful | 4358 | 6 |
| to | 7 | 1775 | drm | 6395 | 7 |
| it | 8 | 1956 | poorly | 716 | 8 |
| of | 9 | 2815 | uninspired | 4245 | 9 |
| this | 10 | 3275 | worst | 402 | 10 |

表 1：Amazon 全量评论数据集上按熵（左）和范数（右）排名的最佳词。我们给出了两种标准的排名。范数排名过滤掉了携带少量信息的词。

**剪枝。** 图 2 展示了不同体积下我们模型的性能。我们固定 $k = d/2$ 并使用不同的剪枝阈值。NPQ 相比完整模型提供 \times10 的压缩率。随着剪枝更加激进，整体压缩率可提升至 \times1000，性能仅有小幅下降且测试时无额外开销。事实上，使用更小的字典使模型在测试时更快。我们还与字符级卷积神经网络（CNN）[Zhang et al., 2015; Xiao & Cho, 2016] 进行比较。它们是有吸引力的文本分类模型，因为以更少的内存使用达到相当的性能 [Xiao & Cho, 2016]。尽管 fastText 使用默认设置时占用更多内存，但 NPQ 已经与 CNN 的内存使用相当。注意 CNN 未被量化，值得研究它们在多少量化下可以不损失性能。这样的研究超出本文范围。我们的剪枝基于第 3.3 节指南中的嵌入范数。表 1 将范数获得的排名与使用熵获得的排名进行了比较，后者常用于无监督设置 [Stolcke, 2000]。


![图2](/Users/dazhang/PycharmProject/Papers/1-NLP/.picture/2016-FastText-FastText.zip-Compressing text classification models-fig2.png)

图 2：准确率损失作为模型体积的函数。我们将压缩模型与不同剪枝级别的 NPQ 和完整 fastText 模型进行比较。我们还与 Zhang et al. [2015] 和 Xiao & Cho [2016] 进行比较。注意体积为对数刻度。

**极端压缩。** 最后，在表 2 中，我们通过查看 64KB 以下模型获得的性能来探索量化模型的极限。令人惊讶的是，即使在 64KiB 和 32KiB 下，尽管压缩率高达 \times1000-4000，性能下降分别仅为约 0.8% 和 1.7%。

| 数据集 | 完整 | 64KiB | 32KiB | 16KiB |
|---|---|---|---|---|
| AG | 92.1 | 91.4 | 90.6 | 89.1 |
| Amazon full | 60.0 | 58.8 | 56.0 | 52.9 |
| Amazon pol. | 94.5 | 93.3 | 92.1 | 89.3 |
| DBPedia | 98.4 | 98.2 | 98.1 | 97.4 |
| Sogou | 96.4 | 96.4 | 96.3 | 95.5 |
| Yahoo | 72.1 | 70.0 | 69.0 | 69.2 |
| Yelp full | 63.8 | 63.2 | 62.4 | 58.7 |
| Yelp pol. | 95.7 | 95.3 | 94.9 | 93.2 |
| 平均差异 [%] | 0 | -0.8 | -1.7 | -3.5 |

表 2：极小模型上的性能。我们使用 $k = 1$ 的量化、哈希和极端剪枝。最后一行显示了不同体积的平均性能下降。

### 4.2 大数据集：FlickrTag

在本节中，我们探索压缩算法在非常大数据集上的极限。与 Joulin et al. [2016] 类似，我们考虑一个包含 312,116 个标签的标签预测数据集。我们将词的最小出现次数设为 10，得到 1,427,667 个词的字典。我们使用 1000 万个桶用于 $n$-gram 并使用层级 softmax。我们将此数据集称为 FlickrTag。

**输出编码。** 我们有兴趣了解当分类器也被量化（即公式 (1) 中的矩阵 $B$）且当剪枝处于覆盖全数据集所需最少特征的极限时，性能如何退化。

| 模型 | $k$ | 归一化 | 再训练 | 准确率 | 体积 |
|---|---|---|---|---|---|
| 完整（未压缩） | | | | 45.4 | 12 GiB |
| 输入 | 128 | | | 45.0 | 1.7 GiB |
| 输入 | 128 | \times | | 45.3 | 1.8 GiB |
| 输入 | 128 | \times | \times | 45.5 | 1.8 GiB |
| 输入+输出 | 128 | \times | | 45.2 | 1.5 GiB |
| 输入+输出 | 128 | \times | \times | 45.4 | 1.5 GiB |

表 3：FlickrTag：量化输出矩阵对性能的影响。我们使用 PQ 进行量化，可选归一化。我们还在量化输入矩阵后再训练输出矩阵。"归一化"指将幅度和角度分开编码，"再训练"指第 3.2 节描述的自底向上 PQ 再训练方法。

表 3 显示同时量化"输入"矩阵（即公式 (1) 中的 $A$）和"输出"矩阵（即 $B$）不会使性能相比完整模型下降。我们使用 $d = 256$ 维的嵌入并使用 $k = d/2$ 个子量化器。我们不使用任何文本特定技巧，这导致 \times8 的压缩率。注意即使输出矩阵未在嵌入上重新训练，性能也仅比完整模型低 0.2%。如附录所示，使用更少的子量化器会显著降低性能，而内存节省有限。

| 模型 | 嵌入数 | 完整 | 熵剪枝 | | 范数剪枝 | | 最大覆盖剪枝 | |
|---|---|---|---|---|---|---|---|---|
| | | | 2M | 1M | 2M | 1M | 2M | 1M |
| 内存 | 11.5M | 12GiB | 297MiB | 174MiB | 305MiB | 179MiB | 305MiB | 179MiB |
| 覆盖率 [%] | 88.4 | | 70.5 | 70.5 | 73.2 | 61.9 | 88.4 | 88.4 |
| 准确率 | 45.4 | | 32.1 | 30.5 | 41.6 | 35.8 | 45.5 | 43.9 |

表 4：FlickrTag：熵剪枝、范数剪枝和最大覆盖剪枝方法的比较。我们展示了每种方法在测试集上的覆盖率。

**剪枝。** 表 4 展示了性能如何随剪枝变化。我们在完全量化模型之上测量这种效果。完整模型因缺失词（某些文档仅由标签组成或仅有稀有词）而遗漏 11.6% 的测试集。有 312,116 个标签，因此保留百万量级的嵌入似乎是合理的。使用 100 万个特征的朴素剪枝遗漏约 30%-40% 的测试集，导致性能显著下降。另一方面，尽管最大覆盖剪枝方法是在训练集上设置的，但它在测试集上没有遭受任何覆盖率损失。这导致更小的性能下降。但如果剪枝过于激进，覆盖率会显著下降。

## 5 未来工作

未来可能获得进一步的模型体积缩减。一个想法是根据频率条件化向量的大小（对输入特征和标签均适用）[Chen et al., 2015; Grave et al., 2016]。例如，在 FlickrTag 数据集中，用完整的 256 维向量表示稀有标签可能不值得。因此，根据频率和范数条件化向量大小似乎是未来值得探索的方向。

我们也可以考虑结合熵和范数剪枝标准：不是仅基于频率或范数保留模型中的特征，而是同时使用两者来保留一组好的特征。这有助于保留既频繁又有区分性的特征，从而减少我们观察到的覆盖率问题。

此外，与其剪枝掉不太有用的特征，我们可以将它们分解为更小的单元 [Mikolov et al., 2012]。例如，这可以通过将每个非区分性词拆分为字符三元组序列来实现。这在训练和测试样本非常短（例如只有一个词）的情况下可能会有所帮助。

## 6 结论

在本文中，我们提出了几种简单技术，可在不牺牲准确率和速度的情况下，将某些文本分类器的内存复杂度降低数个数量级。这是通过应用判别式剪枝（旨在仅保留训练模型中的重要特征）以及对权重矩阵进行量化和对字典进行哈希来实现的。

我们将把代码作为 fastText 库的扩展发布。我们希望我们的工作能为研究社区提供一个基线，因为对给定参数数量下各种深度学习文本分类器性能比较的兴趣日益增长。总体而言，与基于卷积神经网络的近期工作相比，fastText.zip 通常更准确，同时在常见 CPU 上训练所需时间少数个数量级，且内存复杂度仅为一小部分。

## 参考文献

[1] Alekh Agarwal, Olivier Chapelle, Miroslav Dudık, and John Langford. A reliable effective terascale linear learning system. *Journal of Machine Learning Research*, 15(1):1111–1133, 2014.

[2] Francis Bach, Rodolphe Jenatton, Julien Mairal, and Guillaume Obozinski. Optimization with sparsity-inducing penalties. *Foundations and Trends R in Machine Learning*, 4(1):1–106, 2012.

[3] Ashwinkumar Badanidiyuru, Baharan Mirzasoleiman, Amin Karbasi, and Andreas Krause. Streaming submodular maximization: Massive data summarization on the fly. In *SIGKDD*, pp. 671–680. ACM, 2014.

[4] Mohammad Hossein Bateni, Mohammad Taghi Hajiaghayi, and Morteza Zadimoghaddam. Submodular secretary problem and extensions. In *Approximation, Randomization, and Combinatorial Optimization. Algorithms and Techniques*, pp. 39–52. Springer, 2010.

[5] Moses S. Charikar. Similarity estimation techniques from rounding algorithms. In *STOC*, pp. 380–388, May 2002.

[6] Welin Chen, David Grangier, and Michael Auli. Strategies for training large vocabulary neural language models. *arXiv preprint arXiv:1512.04906*, 2015.

[7] Flavio Chierichetti, Ravi Kumar, and Andrew Tomkins. Max-cover in map-reduce. In *International Conference on World Wide Web*, 2010.

[8] Matthieu Courbariaux, Itay Hubara, Daniel Soudry, Ran El-Yaniv, and Yoshua Bengio. Binarized neural networks: Training neural networks with weights and activations constrained to +1 or -1. *arXiv preprint arXiv:1602.02830*, 2016.

[9] M. Datar, N. Immorlica, P. Indyk, and V.S. Mirrokni. Locality-sensitive hashing scheme based on p-stable distributions. In *Proceedings of the Symposium on Computational Geometry*, pp. 253–262, 2004.

[10] Scott Deerwester, Susan T Dumais, George W Furnas, Thomas K Landauer, and Richard Harshman. Indexing by latent semantic analysis. *Journal of the American society for information science*, 1990.

[11] Misha Denil, Babak Shakibi, Laurent Dinh, Marc-Aurelio Ranzato, and Nando et all de Freitas. Predicting parameters in deep learning. In *NIPS*, pp. 2148–2156, 2013.

[12] Uriel Feige. A threshold of ln n for approximating set cover. *JACM*, 45(4):634–652, 1998.

[13] Tiezheng Ge, Kaiming He, Qifa Ke, and Jian Sun. Optimized product quantization for approximate nearest neighbor search. In *CVPR*, June 2013.

[14] Yunchao Gong and Svetlana Lazebnik. Iterative quantization: A procrustean approach to learning binary codes. In *CVPR*, June 2011.

[15] Yunchao Gong, Liu Liu, Ming Yang, and Lubomir Bourdev. Compressing deep convolutional networks using vector quantization. *arXiv preprint arXiv:1412.6115*, 2014.

[16] Edouard Grave, Armand Joulin, Moustapha Cissé, David Grangier, and Hervé Jégou. Efficient softmax approximation for gpus. *arXiv preprint arXiv:1609.04309*, 2016.

[17] Song Han, Huizi Mao, and William J Dally. Deep compression: Compressing deep neural networks with pruning, trained quantization and huffman coding. In *ICLR*, 2016.

[18] Hervé Jégou, Matthijs Douze, and Cordelia Schmid. Hamming embedding and weak geometric consistency for large scale image search. In *ECCV*, October 2008.

[19] Hervé Jegou, Matthijs Douze, and Cordelia Schmid. Product quantization for nearest neighbor search. *IEEE Trans. PAMI*, January 2011.

[20] Thorsten Joachims. Text categorization with support vector machines: Learning with many relevant features. *Springer*, 1998.

[21] Armand Joulin, Edouard Grave, Piotr Bojanowski, and Tomas Mikolov. Bag of tricks for efficient text classification. *arXiv preprint arXiv:1607.01759*, 2016.

[22] Yann LeCun, John S Denker, and Sara A Solla. Optimal brain damage. *NIPS*, 2:598–605, 1990.

[23] Zhouhan Lin, Matthieu Courbariaux, Roland Memisevic, and Yoshua Bengio. Neural networks with few multiplications. *arXiv preprint arXiv:1510.03009*, 2015.

[24] Andrew McCallum and Kamal Nigam. A comparison of event models for naive bayes text classification. In *AAAI workshop on learning for text categorization*, 1998.

[25] Lukas Meier, Sara Van De Geer, and Peter Bühlmann. The group lasso for logistic regression. *Journal of the Royal Statistical Society: Series B (Statistical Methodology)*, 70(1):53–71, 2008.

[26] Tomas Mikolov. Statistical language models based on neural networks. In *PhD thesis*. VUT Brno, 2012.

[27] Tomas Mikolov, Ilya Sutskever, Anoop Deoras, Hai-Son Le, Stefan Kombrink, and J Cernocky. Subword language modeling with neural networks. *preprint*, 2012.

[28] Behnam Neyshabur and Nathan Srebro. On symmetric and asymmetric lshs for inner product search. In *ICML*, pp. 1926–1934, 2015.

[29] Mohammad Norouzi and David Fleet. Cartesian k-means. In *CVPR*, June 2013.

[30] Bo Pang and Lillian Lee. Opinion mining and sentiment analysis. *Foundations and trends in information retrieval*, 2008.

[31] Alexandre Sablayrolles, Matthijs Douze, Hervé Jégou, and Nicolas Usunier. How should we evaluate supervised hashing? *arXiv preprint arXiv:1609.06753*, 2016.

[32] Jorge Sánchez and Florent Perronnin. High-dimensional signature compression for large-scale image classification. In *CVPR*, 2011.

[33] Anshumali Shrivastava and Ping Li. Asymmetric LSH for sublinear time maximum inner product search. In *NIPS*, pp. 2321–2329, 2014.

[34] Andreas Stolcke. Entropy-based pruning of backoff language models. *arXiv preprint cs/0006025*, 2000.

[35] David Talbot and Thorsten Brants. Randomized language models via perfect hash functions. In *ACL*, 2008.

[36] Bart Thomee, David A Shamma, Gerald Friedland, Benjamin Elizalde, Karl Ni, Douglas Poland, Damian Borth, and Li-Jia Li. Yfcc100m: The new data in multimedia research. In *Communications of the ACM*, 2016.

[37] Jingdong Wang, Heng Tao Shen, Jingkuan Song, and Jianqiu Ji. Hashing for similarity search: A survey. *arXiv preprint arXiv:1408.2927*, 2014.

[38] Jun Wang, Wei Liu, Sanjiv Kumar, and Shih-Fu Chang. Learning to hash for indexing big data - A survey. *CoRR*, abs/1509.05472, 2015.

[39] Sida Wang and Christopher D Manning. Baselines and bigrams: Simple, good sentiment and topic classification. In *ACL*, 2012.

[40] Kilian Q Weinberger, Anirban Dasgupta, John Langford, Alex Smola, and Josh Attenberg. Feature hashing for large scale multitask learning. In *ICML*, 2009.

[41] Yair Weiss, Antonio Torralba, and Rob Fergus. Spectral hashing. In *NIPS*, December 2009.

[42] Yijun Xiao and Kyunghyun Cho. Efficient character-level document classification by combining convolution and recurrent layers. *arXiv preprint arXiv:1602.00367*, 2016.

[43] Xiang Zhang, Junbo Zhao, and Yann LeCun. Character-level convolutional networks for text classification. In *NIPS*, 2015.

## 附录

在附录中，我们展示一些额外结果。这些实验中使用的模型仅有 100 万个 $n$-gram 桶。表 5 展示了 LSH、PQ 和 OPQ 在 8 个不同数据集上的详细比较。表 7 总结了与 CNN 在准确率和体积方面的比较。表 8 展示了哈希 trick 和 Bloom filter 的详细比较。

| 量化方法 | 归一化 | $k$ | AG | Amz. f. | Amz. p. | DBP | Sogou | Yah. | Yelp f. | Yelp p. |
|---|---|---|---|---|---|---|---|---|---|---|
| 完整 | | | 92.1 (36M) | 59.8 (97M) | 94.5 (104M) | 98.4 (67M) | 96.3 (47M) | 72 (120M) | 63.7 (56M) | 95.7 (53M) |
| 完整,无字典 | | | 92.1 (34M) | 59.9 (78M) | 94.5 (83M) | 98.4 (56M) | 96.3 (42M) | 72.2 (91M) | 63.6 (48M) | 95.6 (46M) |
| LSH | | 8 | 88.7 (8.5M) | 51.3 (20M) | 90.3 (21M) | 92.7 (14M) | 94.2 (11M) | 54.8 (23M) | 56.7 (12M) | 92.2 (12M) |
| PQ | | 8 | 91.7 (8.5M) | 59.3 (20M) | 94.4 (21M) | 97.4 (14M) | 96.1 (11M) | 71.3 (23M) | 62.8 (12M) | 95.4 (12M) |
| OPQ | | 8 | 91.9 (8.5M) | 59.3 (20M) | 94.4 (21M) | 96.9 (14M) | 95.8 (11M) | 71.4 (23M) | 62.5 (12M) | 95.4 (12M) |
| LSH | \times | 8 | 91.9 (9.5M) | 59.4 (22M) | 94.5 (24M) | 97.8 (16M) | 96.2 (12M) | 71.6 (26M) | 63.4 (14M) | 95.6 (13M) |
| PQ | \times | 8 | 92.0 (9.5M) | 59.8 (22M) | 94.5 (24M) | 98.4 (16M) | 96.3 (12M) | 72.1 (26M) | 63.7 (14M) | 95.6 (13M) |
| OPQ | \times | 8 | 92.1 (9.5M) | 59.9 (22M) | 94.5 (24M) | 98.4 (16M) | 96.3 (12M) | 72.2 (26M) | 63.6 (14M) | 95.6 (13M) |
| LSH | | 4 | 88.3 (4.3M) | 50.5 (9.7M) | 88.9 (11M) | 91.6 (7.0M) | 94.3 (5.3M) | 54.6 (12M) | 56.5 (6.0M) | 92.9 (5.7M) |
| PQ | | 4 | 91.6 (4.3M) | 59.2 (9.7M) | 94.4 (11M) | 96.3 (7.0M) | 96.1 (5.3M) | 71.0 (12M) | 62.2 (6.0M) | 95.4 (5.7M) |
| OPQ | | 4 | 91.7 (4.3M) | 59.0 (9.7M) | 94.4 (11M) | 96.9 (7.0M) | 95.6 (5.3M) | 71.2 (12M) | 62.6 (6.0M) | 95.4 (5.7M) |
| LSH | \times | 4 | 92.1 (5.3M) | 59.2 (13M) | 94.4 (13M) | 97.7 (8.8M) | 96.2 (6.6M) | 71.1 (15M) | 63.1 (7.4M) | 95.5 (7.2M) |
| PQ | \times | 4 | 92.1 (5.3M) | 59.8 (13M) | 94.5 (13M) | 98.4 (8.8M) | 96.3 (6.6M) | 72.0 (15M) | 63.6 (7.5M) | 95.6 (7.2M) |
| OPQ | \times | 4 | 92.2 (5.3M) | 59.8 (13M) | 94.5 (13M) | 98.3 (8.8M) | 96.3 (6.6M) | 72.1 (15M) | 63.7 (7.5M) | 95.6 (7.2M) |
| LSH | | 2 | 87.7 (2.2M) | 50.1 (4.9M) | 88.9 (5.2M) | 90.6 (3.5M) | 93.9 (2.7M) | 51.4 (5.7M) | 56.6 (3.0M) | 91.3 (2.9M) |
| PQ | | 2 | 91.1 (2.2M) | 58.7 (4.9M) | 94.4 (5.2M) | 87.1 (3.6M) | 95.3 (2.7M) | 69.5 (5.7M) | 62.1 (3.0M) | 95.4 (2.9M) |
| OPQ | | 2 | 91.4 (2.2M) | 58.2 (4.9M) | 94.3 (5.2M) | 91.6 (3.6M) | 94.2 (2.7M) | 69.6 (5.7M) | 62.1 (3.0M) | 95.4 (2.9M) |
| LSH | \times | 2 | 91.8 (3.2M) | 58.6 (7.3M) | 94.3 (7.8M) | 97.1 (5.3M) | 96.1 (4.0M) | 69.7 (8.6M) | 62.7 (4.5M) | 95.5 (4.3M) |
| PQ | \times | 2 | 91.9 (3.2M) | 59.6 (7.3M) | 94.5 (7.8M) | 98.1 (5.3M) | 96.3 (4.0M) | 71.3 (8.6M) | 63.4 (4.5M) | 95.6 (4.3M) |
| OPQ | \times | 2 | 92.1 (3.2M) | 59.5 (7.3M) | 94.5 (7.8M) | 98.1 (5.3M) | 96.2 (4.0M) | 71.5 (8.6M) | 63.4 (4.5M) | 95.6 (4.3M) |

表 5：标准量化方法之间的比较。原始模型维度为 8，200 万个桶。注意所有方法均无字典。

| $k$ | 剪枝截止 | AG | Amz. f. | Amz. p. | DBP | Sogou | Yah. | Yelp f. | Yelp p. |
|---|---|---|---|---|---|---|---|---|---|
| 完整,无字典 | | 92.1 (34M) | 59.8 (78M) | 94.5 (83M) | 98.4 (56M) | 96.3 (42M) | 72.2 (91M) | 63.7 (48M) | 95.6 (46M) |
| 8 | 完整 | 92.0 (9.5M) | 59.8 (22M) | 94.5 (24M) | 98.4 (16M) | 96.3 (12M) | 72.1 (26M) | 63.7 (14M) | 95.6 (13M) |
| 4 | 完整 | 92.1 (5.3M) | 59.8 (13M) | 94.5 (13M) | 98.4 (8.8M) | 96.3 (6.6M) | 72 (15M) | 63.6 (7.5M) | 95.6 (7.2M) |
| 2 | 完整 | 91.9 (3.2M) | 59.6 (7.3M) | 94.5 (7.8M) | 98.1 (5.3M) | 96.3 (4.0M) | 71.3 (8.6M) | 63.4 (4.5M) | 95.6 (4.3M) |
| 8 | 200K | 92.0 (2.5M) | 59.7 (2.5M) | 94.3 (2.5M) | 98.5 (2.5M) | 96.6 (2.5M) | 71.8 (2.5M) | 63.3 (2.5M) | 95.6 (2.5M) |
| 8 | 100K | 91.9 (1.3M) | 59.5 (1.3M) | 94.3 (1.3M) | 98.5 (1.3M) | 96.6 (1.3M) | 71.6 (1.3M) | 63.4 (1.3M) | 95.6 (1.3M) |
| 8 | 50K | 91.7 (645K) | 59.7 (645K) | 94.3 (644K) | 98.5 (645K) | 96.6 (645K) | 71.5 (645K) | 63.2 (645K) | 95.6 (644K) |
| 8 | 10K | 91.3 (137K) | 58.6 (137K) | 93.2 (137K) | 98.5 (137K) | 96.5 (137K) | 71.3 (137K) | 63.3 (137K) | 95.4 (137K) |
| 4 | 200K | 92.0 (1.8M) | 59.7 (1.8M) | 94.3 (1.8M) | 98.5 (1.8M) | 96.6 (1.8M) | 71.7 (1.8M) | 63.3 (1.8M) | 95.6 (1.8M) |
| 4 | 100K | 91.9 (889K) | 59.5 (889K) | 94.4 (889K) | 98.5 (889K) | 96.6 (889K) | 71.7 (889K) | 63.4 (889K) | 95.6 (889K) |
| 4 | 50K | 91.7 (449K) | 59.6 (449K) | 94.3 (449K) | 98.5 (450K) | 96.6 (449K) | 71.4 (450K) | 63.2 (449K) | 95.5 (449K) |
| 4 | 10K | 91.5 (98K) | 58.6 (98K) | 93.2 (98K) | 98.5 (98K) | 96.5 (98K) | 71.2 (98K) | 63.3 (98K) | 95.4 (98K) |
| 2 | 200K | 91.9 (1.4M) | 59.6 (1.4M) | 94.3 (1.4M) | 98.4 (1.4M) | 96.5 (1.4M) | 71.5 (1.4M) | 63.2 (1.4M) | 95.5 (1.4M) |
| 2 | 100K | 91.6 (693K) | 59.5 (693K) | 94.3 (693K) | 98.4 (694K) | 96.6 (693K) | 71.1 (694K) | 63.2 (693K) | 95.6 (693K) |
| 2 | 50K | 91.6 (352K) | 59.6 (352K) | 94.3 (352K) | 98.4 (352K) | 96.5 (352K) | 71.1 (352K) | 63.2 (352K) | 95.6 (352K) |
| 2 | 10K | 91.3 (78K) | 58.5 (78K) | 93.2 (78K) | 98.4 (79K) | 96.5 (78K) | 70.8 (78K) | 63.2 (78K) | 95.3 (78K) |

表 6：不同量化方法和剪枝级别的比较。"剪枝截止"是剪枝的截止参数。

| 数据集 | Zhang et al. (2015) | Xiao & Cho (2016) | fastText+PQ, $k = d/2$ |
|---|---|---|---|
| AG | 90.2 (108M) | 91.4 (80M) | 91.9 (889K) |
| Amz. f. | 59.5 (10.8M) | 59.2 (1.6M) | 59.6 (449K) |
| Amz. p. | 94.5 (10.8M) | 94.1 (1.6M) | 94.3 (449K) |
| DBP | 98.3 (108M) | 98.6 (1.2M) | 98.5 (98K) |
| Sogou | 95.1 (108M) | 95.2 (1.6M) | 96.5 (98K) |
| Yah. | 70.5 (108M) | 71.4 (80M) | 71.7 (889K) |
| Yelp f. | 61.6 (108M) | 61.8 (1.4M) | 63.3 (98K) |
| Yelp p. | 94.8 (108M) | 94.5 (1.2M) | 95.5 (449K) |

表 7：CNN 与 fastText 有无量化的比较。Zhang et al. [2015] 的数字来自 Xiao & Cho [2016]。注意对于 CNN，我们报告的模型体积基于 float32 存储的假设。对于 fastText(+PQ)，我们报告测试时 RAM 中使用的内存量。

| 量化方法 | Bloom | 剪枝截止 | AG | Amz. f. | Amz. p. | DBP | Sogou | Yah. | Yelp f. | Yelp p. |
|---|---|---|---|---|---|---|---|---|---|---|
| 完整,无字典 | | | 92.1 (34M) | 59.8 (78M) | 94.5 (83M) | 98.4 (56M) | 96.3 (42M) | 72.2 (91M) | 63.7 (48M) | 95.6 (46M) |
| NPQ | | 200K | 91.9 (1.4M) | 59.6 (1.4M) | 94.3 (1.4M) | 98.4 (1.4M) | 96.5 (1.4M) | 71.5 (1.4M) | 63.2 (1.4M) | 95.5 (1.4M) |
| NPQ | \times | 200K | 92.2 (830K) | 59.3 (830K) | 94.1 (830K) | 98.4 (830K) | 96.5 (830K) | 70.7 (830K) | 63.0 (830K) | 95.5 (830K) |
| NPQ | | 100K | 91.6 (693K) | 59.5 (693K) | 94.3 (693K) | 98.4 (694K) | 96.6 (693K) | 71.1 (694K) | 63.2 (693K) | 95.6 (693K) |
| NPQ | \times | 100K | 91.8 (420K) | 59.1 (420K) | 93.9 (420K) | 98.4 (420K) | 96.5 (420K) | 70.6 (420K) | 62.8 (420K) | 95.3 (420K) |
| NPQ | | 50K | 91.6 (352K) | 59.6 (352K) | 94.3 (352K) | 98.4 (352K) | 96.5 (352K) | 71.1 (352K) | 63.2 (352K) | 95.6 (352K) |
| NPQ | \times | 50K | 91.5 (215K) | 58.8 (215K) | 93.6 (215K) | 98.3 (215K) | 96.5 (215K) | 70.1 (215K) | 62.7 (215K) | 95.1 (215K) |
| NPQ | | 10K | 91.3 (78K) | 58.5 (78K) | 93.2 (78K) | 98.4 (79K) | 96.5 (78K) | 70.8 (78K) | 63.2 (78K) | 95.3 (78K) |
| NPQ | \times | 10K | 90.8 (51K) | 56.8 (51K) | 91.7 (51K) | 98.1 (51K) | 96.1 (51K) | 68.7 (51K) | 61.7 (51K) | 94.5 (51K) |

表 8：有无 Bloom filter 的比较。对于 NPQ，我们设 $d = 8$，$k = 2$。

| 模型 | $k$ | 归一化 | 再训练 | 准确率 | 体积 |
|---|---|---|---|---|---|
| 完整 | | | | 45.4 | 12G |
| 输入 | 128 | | | 45.0 | 1.7G |
| 输入 | 128 | \times | | 45.3 | 1.8G |
| 输入 | 128 | \times | \times | 45.5 | 1.8G |
| 输入+输出 | 128 | \times | | 45.2 | 1.5G |
| 输入+输出 | 128 | \times | \times | 45.4 | 1.5G |
| 输入+输出,截止=2M | 128 | \times | \times | 45.5 | 305M |
| 输入+输出,截止=1M | 128 | \times | \times | 43.9 | 179M |
| 输入 | 64 | | | 44.0 | 1.1G |
| 输入 | 64 | \times | | 44.7 | 1.1G |
| 输入 | 64 | \times | \times | 44.9 | 1.1G |
| 输入+输出 | 64 | \times | | 44.6 | 784M |
| 输入+输出 | 64 | \times | \times | 44.8 | 784M |
| 输入+输出,截止=2M | 64 | \times | | 42.5 | 183M |
| 输入+输出,截止=1M | 64 | \times | | 39.9 | 118M |
| 输入+输出,截止=2M | 64 | \times | \times | 45.0 | 183M |
| 输入+输出,截止=1M | 64 | \times | \times | 43.4 | 118M |
| 输入 | 32 | | | 40.5 | 690M |
| 输入 | 32 | \times | | 42.4 | 701M |
| 输入 | 32 | \times | \times | 42.9 | 701M |
| 输入+输出 | 32 | \times | | 42.3 | 435M |
| 输入+输出 | 32 | \times | \times | 42.8 | 435M |
| 输入+输出,截止=2M | 32 | \times | | 35.0 | 122M |
| 输入+输出,截止=1M | 32 | \times | | 32.6 | 88M |
| 输入+输出,截止=2M | 32 | \times | \times | 43.3 | 122M |
| 输入+输出,截止=1M | 32 | \times | \times | 41.6 | 88M |

表 9：FlickrTag：大数据集上 (i) 不同量化方法和参数、(ii) 有无再训练的比较。
