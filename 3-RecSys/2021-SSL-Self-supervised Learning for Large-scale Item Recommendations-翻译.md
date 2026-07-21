# 面向大规模item推荐的自监督学习

> Tiansheng Yao, Xinyang Yi, Derek Zhiyuan Cheng, Felix Yu, Ting Chen, Aditya Menon, Lichan Hong, Ed H. Chi, Steve Tjoa, Jieqi Kang, Evan Ettinger | Google

本文介绍了 面向大规模item推荐的自监督学习。核心内容：


关键发现：


{tyao,xinyang,zcheng,felixyu,iamtingchen,adityakmenon}@google.com
{lichan,edchi,stevetjoa,jaykang,eettinger}@google.com


---

## 摘要

大规模推荐模型从海量候选中找出最相关的item，它们在现代搜索和推荐系统中扮演着关键角色。为了对具有大词表类别型特征的输入空间进行建模，典型的推荐模型通过神经网络从用户反馈数据中为查询和item学习一个联合嵌入空间。然而，当语料库中包含数百万到数十亿个item时，用户往往只对其中非常小的一部分提供反馈，导致幂律分布。这使得长尾item的反馈数据极度稀疏。

受近期自监督表示学习在计算机视觉和自然语言理解领域取得成功的启发，我们提出了一个用于大规模item推荐的多任务自监督学习框架。该框架旨在通过更好地学习item特征的潜在关系来解决标签稀疏问题。具体而言，自监督学习改进了item表示学习，并作为额外的正则化手段提升了泛化能力。此外，我们在所提出的框架中提出了一种利用特征相关性的新颖数据增强方法。

我们使用两个分别包含5亿和10亿训练样本的真实世界数据集评估了我们的框架。实验结果证明了自监督学习正则化的有效性，并展示了其相对于最先进正则化技术的优越性能。我们还将所提出的技术部署到了一个网络级商业应用间推荐系统中，在线A/B实验在顶级的业务指标上取得了显著提升。我们的在线结果也验证了我们的假设：该框架在缺乏监督的数据切片上能更大程度地提升模型性能。

## CCS概念
• 信息系统 \rightarrow 信息检索。

## 关键词
推荐系统，自监督学习，神经网络，对比学习

## ACM引用格式
Tiansheng Yao, Xinyang Yi, Derek Zhiyuan Cheng, Felix Yu, Ting Chen, Aditya Menon and Lichan Hong, Ed H. Chi, Steve Tjoa, Jieqi (Jay) Kang, Evan Ettinger. 2021. 面向大规模item推荐的自监督学习. 发表于第30届ACM信息与知识管理国际会议(CIKM '21), 2021年11月1-5日, 澳大利亚线上会议. ACM, 纽约, NY, USA, 10页. https://doi.org/10.1145/3459637.3481952

允许个人或课堂使用复制本作品的部分或全部，无需付费，前提是复制件不以营利或商业优势为目的，且首页保留此声明和完整引用。第三方组件的版权须得到尊重。所有其他用途请联系作者/所有者。
CIKM '21, 2021年11月1–5日, 澳大利亚线上会议。
© 2021 版权归作者/所有者所有。
ACM ISBN 978-1-4503-8446-9/21/11.
https://doi.org/10.1145/3459637.3481952

图1: 模型架构：双塔DNN，包含查询和item表示。

## 1 引言

近年来，神经网络模型已成为现代推荐系统在工业界（参见例如[18, 31, 39, 42]）和学术界（[8, 32]）的主要舞台。与矩阵分解[1, 21, 22]、梯度提升决策树[4, 29]和基于逻辑回归的推荐器[19]等传统方法相比，这些深度模型能更有效地处理类别型特征。它们还支持更复杂的数据表示，并引入更多非线性以更好地拟合推荐器的复杂数据。

本文聚焦的一个特定推荐任务是从海量item候选中找出与给定查询最相关的item。这个大规模item推荐的一般性问题已在各种应用中被广泛采纳。根据查询的类型，推荐任务可以是：(i) 个性化推荐：当查询是一个用户时；(ii) item到item推荐：当查询也是一个item时；(iii) 搜索：当查询是一段自由文本时。为了对查询和item之间的交互进行建模，一种知名的方法利用基于嵌入的神经网络。推荐任务通常被形式化为一个极端分类问题[10]，其中每个item在输出空间中被表示为一个稠密向量。

本文聚焦于双塔DNN（见图1），这种结构在许多实际推荐系统中很流行（参见例如[23, 39]）。在这种架构中，神经网络将一组item特征编码为嵌入，从而使其即使对冷启动item也能进行索引。此外，双塔DNN架构通过将top-k最近邻搜索问题转化为可在亚线性复杂度下求解的最大内积搜索问题[9]，实现了对大型语料库item的实时高效服务。

基于嵌入的深度模型通常拥有大量参数，因为它们构建于高维嵌入之上，这些嵌入表示高基数的稀疏特征，如主题或itemID。在许多现有文献中，训练这些模型的损失函数被形式化为一个有监督学习问题。监督信号来自收集的标签（例如点击）。现代推荐系统从用户那里收集数十亿到数万亿的足迹，为构建深度模型提供了海量训练数据。然而，当需要对数百万（如歌曲和app[28]）甚至数十亿（如YouTube上的视频[10]）量级的海量item语料库进行建模时，数据在某些切片上仍然可能高度稀疏，原因在于：

• 高度倾斜的数据分布：查询和item之间的交互通常呈现高度倾斜的幂律分布[30]。因此，一小部分热门item占据了大部分交互。这始终使得长尾item的训练数据非常稀疏。

• 缺乏显式用户反馈：用户通常提供大量隐式正向反馈，如点击和点赞。然而，他们提供显式反馈（如item评分、用户满意度反馈和相关性评分）的可能性要小得多。

自监督学习通过未标注数据提供了改进深度表示学习的不同视角。其基本思想是通过各种数据增强来丰富训练数据，并将预测或重构原始样本的监督任务作为辅助任务。自监督学习已广泛应用于计算机视觉[15, 25, 33]和自然语言理解[12, 24]领域。例如，计算机视觉中的一项工作[25]提出随机旋转图像，并训练模型预测每个增强后的输入图像是如何旋转的。在自然语言理解中，BERT模型引入了掩码语言任务，以帮助改进语言模型的预训练。类似地，其他预训练任务，如预测周围句子和Wikipedia文章中的链接句子，也被用于改进自然语言理解中的双编码器类型模型[3]。与传统的监督学习相比，自监督学习提供了互补的目标，消除了手动收集标签的前提条件。此外，自监督学习通过利用输入特征的内部关系，能够自主发现良好的语义表示。

尽管自监督学习在计算机视觉和自然语言理解中被广泛采用，但其在推荐系统领域的应用研究较少。最接近的研究方向是一组正则化技术[17, 23, 41]，这些技术旨在强制不同样本的学习表示（即多层感知机的输出层（嵌入））彼此远离，并在整个潜在嵌入空间中分散开来。虽然与自监督学习精神相似，但这些技术并未显式构建自监督学习任务。与计算机视觉或自然语言理解应用中的模型不同，推荐模型的输入极度稀疏，其中高基数类别型特征采用独热（或多热）编码，例如itemID或item类别[31]。这些特征在深度模型中通常表示为可学习的嵌入向量。由于计算机视觉和自然语言理解中的大多数模型处理的是稠密输入，现有的创建自监督学习任务的方法不能直接适用于推荐系统中的稀疏模型。最近，一系列研究探索了利用自监督学习改进推荐中的序列用户建模[27, 37, 43]。与这些工作不同，本文聚焦于item表示学习，并展示了自监督学习如何在长尾item分布的背景下帮助改进泛化。此外，与在特定序列用户特征上使用自监督学习不同，我们设计了新的自监督学习任务，并证明了它们在一组异质类别型特征上的学习效果。我们认为这是一种对于其他类型的推荐模型（如多任务排序模型（例如[42]））更通用的设置。

在本文中，我们提出利用基于自监督学习的辅助任务来改进item表示，特别是在长尾分布和稀疏数据场景下。与计算机视觉或自然语言理解应用不同，推荐模型的输入空间高度稀疏，并由一组具有大基数的类别型特征（例如itemID）表示。针对这类稀疏模型，我们提出了一个新的自监督学习框架，其核心思想是：(i) 通过掩码输入信息来增强数据；(ii) 通过双塔DNN编码每一对增强样本；(iii) 应用对比损失来学习增强数据的表示。对比学习的目标是让来自同一样本的增强数据能够与其他样本区分开来。注意，用于对比学习的双塔DNN与用于编码查询和item的双塔DNN可以共享一定量的模型参数。详见第3节。

我们的贡献有四个方面：

• 自监督学习框架：我们提出了一种与模型架构无关的自监督学习框架，适用于推荐中的稀疏神经模型。辅助的自监督学习损失和主监督损失通过多任务学习框架联合优化。我们专注于使用该框架对大型语料库item进行高效评分，这在两阶段推荐器中也被称为item检索[10]。我们相信它也将为其他类型的模型（如排序模型[7]）设计自监督学习提供启示。

• 数据增强：我们提出了一种新颖的数据增强方法，该方法利用特征相关性，并针对推荐模型中常见的异质类别型特征进行了定制。

• 离线实验：在一个公开数据集和一个工业级推荐系统数据集上，我们证明了引入自监督学习作为辅助任务可以显著提升模型性能，尤其是在标签稀缺的情况下。与最先进的非自监督学习正则化技术[17, 23, 41]相比，我们证明了自监督学习持续表现更好，并且在非自监督学习正则化无法带来额外提升时，自监督学习仍能改进模型性能。

• 网络级推荐器上的在线实验：我们已将所提出的自监督学习技术部署到一个实际大规模系统中的强双塔app间推荐模型中。在线A/B测试显示顶级指标有显著提升。我们特别看到在缺乏监督的切片上改进更大。

## 2 相关工作

**自监督学习与预训练。** 计算机视觉社区已经研究了各种无监督和自监督学习任务。最接近的研究方向是SimCLR[5]，它也利用自监督学习和对比学习进行视觉表示学习。与SimCLR和视觉领域的其他工作[2, 6]不同，我们在此提出更适合推荐系统中类别型特征的增强方法，而非依赖图像特定的增强如裁剪、旋转和颜色失真。此外，所提出的框架不需要多阶段训练方案（如先预训练再微调）[20]。

在自然语言理解中，对于双编码器模型，[3]表明与最终任务更对齐的预训练任务比通用任务（如下一句预测和掩码语言模型）更有帮助。这些预训练任务旨在利用大规模公共自然语言理解内容，如Wikipedia。在本文中，我们也使用双编码器模型架构。与上述不同，我们提出的自监督任务不需要使用额外的数据源。

**分散正则化。** Zhang等人[41]和Wu等人[36]使用分散正则化来改进深度模型的泛化能力。具体而言，在[41]中，一种促进随机实例之间分离的正则化方法被证明可以改善训练稳定性和泛化能力。在[36]中，一个目标是将每个实例作为自己的类别来训练分类器，从而促进嵌入空间中大的实例分离。上述两种方法都是针对计算机视觉应用研究的。

**神经推荐器。** 深度学习在构建工业级推荐系统方面取得了许多成功，例如视频推荐[10]、新闻推荐[34]以及社交网络中的视觉发现[26, 40]。对于大规模item检索，具有独立查询塔和item塔的双塔模型因其高效的服务能力而被广泛使用。推荐通常通过查询嵌入和item嵌入之间的点积计算，从而使寻找top-kitem可以转化为具有亚线性时间复杂度的MIPS问题[9]。一种流行的分解结构是基于softmax的多类分类模型。[10]中的工作将检索任务视为一个极端多类分类问题，使用多层感知机模型通过采样softmax作为损失函数进行训练。这类模型仅使用itemID作为唯一的item特征，面临冷启动问题。最近，一系列研究[23, 39]考虑将双塔DNN应用于检索问题，这也被称为双编码器[16, 38]，其中item嵌入由多层感知机从ID和其他类别型元数据特征构建而成。所提出的自监督方法既适用于排序模型也适用于检索模型。在本文中，我们专注于将自监督学习用于检索模型，特别是改进双塔DNN中的item表示。

**序列推荐中的自监督学习。** 在推荐系统中，最近有一系列研究探索利用自监督学习进行序列推荐。自监督学习任务旨在捕获用户历史中的信息[43]，并在用户序列推荐中学习更鲁棒的去纠缠用户表示[27]。此外，Xin等人表明将自监督学习与强化学习结合能有效捕获序列推荐中的长期用户兴趣。与上述不同，我们提出的自监督学习框架专注于改进长尾分布下的item表示。所提出的自监督学习任务不需要建模序列信息，并且普遍适用于具有异质类别型特征的深度模型。

图2: 自监督学习框架示意图。对输入应用两种数据增强h和g；编码器H和G应用于增强后的样本y_i和y'_i。自监督损失L_self相对于z_i进行优化，目标是最大化与z'_i的相似度，同时最小化z_j和z'_j之间的相似度。

## 3 方法

我们提出了一个用于具有大词表类别型特征的推荐器深度神经网络模型的自监督学习框架。具体而言，第3.1节介绍了一个通用的自监督学习框架。在第3.2节中，我们提出了一种构建自监督学习任务的数据增强方法，并详细阐述了它们与分散正则化的联系。最后，在第3.3节中，我们描述了如何通过多任务学习框架使用自监督学习来改进分解模型（即图1所示的双塔DNN）。

### 3.1 框架

受用于视觉表示学习的SimCLR框架[5]启发，我们采用类似的对比学习算法来学习类别型特征的表示。基本思想有两方面：首先，我们对同一个训练样本应用不同的数据增强来学习表示；然后使用对比损失函数来鼓励为同一个训练样本学到的表示彼此相似。对比损失也被应用于训练双塔DNN（参见例如[23, 39]），尽管那里的目标是使正item与其对应的查询一致。

考虑一批N个item样本x_1, ..., x_N，其中x_i \in X表示样本i的一组特征。在推荐器的上下文中，一个样本表示一个查询、一个item或一个查询-item对。假设存在一对变换函数h, g: X \rightarrow X，分别将x_i增强为y_i和y'_i：

y_i \leftarrow h(x_i), y'_i \leftarrow g(x_i). (1)

对于相同的输入样本i，我们希望学习增强后不同的表示y_i和y'_i，以确保模型仍然能够识别y_i和y'_i代表相同的输入i。换句话说，对比损失学习最小化y_i和y'_i之间的差异。同时，对于不同的样本i和j，对比损失最大化经过不同数据增强后学到的表示y_i和y'_j之间的差异。令z_i, z'_i表示y_i, y'_i经过两个神经网络H, G: X \rightarrow R^d编码后的嵌入，即：

z_i \leftarrow H(y_i), z'_i \leftarrow G(y'_i). (2)

我们将(z_i, z'_i)视为正样本对，将(z_i, z'_j)视为负样本对（其中i \neq j）。令s(z_i, z'_j) = ⟨z_i, z'_j⟩/(||z_i|| · ||z'_j||)。为了鼓励上述性质，我们对一批N个样本{x_i}定义自监督损失为：

L_self({x_i}; H, G) := - (1/N) \Sigma_i log( exp(s(z_i, z'_i)/\tau) / \Sigma_j exp(s(z_i, z'_j)/\tau) ). (3)

其中\tau是一个可调的softmax温度超参数。上述损失函数学习一个鲁棒的嵌入空间，使得数据增强后相似的item彼此接近，而随机样本被推得更远。整体框架如图2所示。

**编码器架构。** 对于具有类别型特征的输入样本，H和G通常由一个输入层和其上的多层感知机构建而成。输入层通常是归一化稠密特征和多个稀疏特征嵌入的拼接，其中稀疏特征嵌入是存储在嵌入表中的学习表示（相比之下，计算机视觉和语言模型的输入层直接处理原始输入）。为了使自监督学习促进监督学习任务，我们为神经网络H和G共享稀疏特征的嵌入表。根据数据增强技术(h, g)的不同，H和G的多层感知机也可以完全或部分共享。

**与分散正则化的联系[41]。** 在特殊情况下，当(h, g)是恒等映射且H, G是同一个神经网络时，方程(3)中的损失函数简化为：

-N^{-1} \Sigma_i log( exp(1/\tau) / (exp(1/\tau) + \Sigma_{j\neqi} exp(s(z_i, z_j)/\tau)) )

这鼓励不同样本的学习表示具有小的余弦相似度。该损失类似于[41]中引入的分散正则化，区别在于原始提议使用平方损失，即N^{-1} \Sigma_i \Sigma_{j\neqi} ⟨z_i, z_j⟩^2，而不是softmax。分散正则化已被证明能改进大规模检索模型的泛化能力。在第4节中，我们表明通过引入特定的数据增强，使用基于自监督学习的正则化可以相比分散正则化进一步改进模型性能。

### 3.2 两阶段数据增强

我们介绍数据增强，即图2中的h和g。给定一组item特征，核心思想是通过掩码部分信息来创建两个增强样本。一个好的变换和数据增强应该对数据做出最少的假设，以便能普遍适用于各种任务和模型。掩码的想法受BERT[12]中的掩码语言建模启发。与序列化token不同，一组通用的特征没有序列顺序，这使得掩码模式的选择成为一个开放问题。我们试图通过探索特征相关性来设计掩码模式。我们提出了相关特征掩码，针对类别型特征进行了定制，并考虑了特征相关性。

在深入掩码细节之前，我们首先介绍两阶段增强算法。注意，在没有增强的情况下，输入层是通过拼接所有类别型特征的嵌入来创建的。两阶段增强包括：

• 掩码：对item特征集应用掩码模式。我们在输入层中使用一个默认嵌入来表示被掩码的特征。

• Dropout：对于具有多个值的类别型特征，我们以一定概率丢弃每个值。这进一步减少了输入信息并增加了自监督学习任务的难度。

掩码步骤可以解释为dropout的一个特例，即100%的dropout率。一种策略是互补掩码模式，即将特征集分成两个互斥的特征集，分别用于两个增强样本。具体而言，我们可以随机将特征集分成两个不相交的子集。我们将此方法称为随机特征掩码，并将其作为基线之一。我们现在介绍相关特征掩码，在其中我们进一步探索创建掩码模式时的特征相关性。

**类别型特征的互信息。** 如果被掩码的特征集是随机选择的，那么(h, g)本质上是从包含k个特征的整个特征集的2^k种不同掩码模式中采样得到的。不同的掩码模式自然会对自监督学习任务产生不同的效果。例如，自监督对比学习任务可能利用两个增强样本之间高度相关特征的捷径，使得自监督学习任务过于简单。为了解决这个问题，我们提出根据通过互信息度量的特征相关性来分割特征。两个类别型特征的互信息由下式给出：

MI(V_i, V_j) = \Sigma_{v_i\inV_i, v_j\inV_j} P(v_i, v_j) log( P(v_i, v_j) / (P(v_i)P(v_j)) ), (4)

其中V_i, V_j表示它们的词表集。所有特征对的互信息可以预先计算。

**相关特征掩码。** 利用预先计算的互信息，我们提出了相关特征掩码，其利用特征依赖模式来构建更有意义的自监督学习任务。对于被掩码的特征集F_m，我们寻求将高度相关的特征一起掩码。为此，我们首先从所有可用特征F = {f_1, ..., f_k}中均匀采样一个种子特征f_seed，然后根据它们与f_seed的互信息选择top-n个最相关的特征F_c = {$f_{c,1}$, ..., $f_{c,n}$}。最终的F_m将是种子特征和相关特征集的并集，即F_m = {f_seed, $f_{c,1}$, ..., $f_{c,n}$}。我们选择n = ⌊k/2⌋，使得被掩码和保留的特征集大小大致相同。我们每批次更换种子特征，以便自监督学习任务能在各种掩码模式上进行学习。

### 3.3 多任务训练

为了使自监督学习学到的表示有助于改进主要监督任务（如回归或分类），我们采用了一种多任务训练策略，其中主要监督任务和辅助自监督学习任务被联合优化。具体而言，令{(q_i, x_i)}是从训练数据分布D_train中采样的一批查询-item对，令{x_i}是从item分布D_item中采样的一批item。则联合损失为：

L = L_main({(q_i, x_i)}) + \alpha · L_self({x_i}), (5)

其中L_main是主要任务的损失函数，用于捕获查询和item之间的交互，\alpha是正则化强度。

**异质样本分布。** 来自D_train的边缘item分布通常遵循幂律分布。因此，对L_self使用训练item分布会导致学到的特征关系偏向头部item。相反，我们对L_self从语料库中均匀采样item。换句话说，D_item是均匀item分布。在实践中，我们发现为主任务和自监督学习任务使用异质分布对于自监督学习实现优越性能至关重要。

**主任务损失。** 根据目标的不同，主损失可以有多种选择。在本文中，我们考虑推荐器[39]和自然语言处理[16]中用于优化top-k准确率的批softmax损失。具体来说，令q_i, x_i分别为查询和item样本(q_i, x_i)经过两个神经网络编码后的嵌入，那么对于一批N对{(q_i, x_i)}和温度\tau，批softmax交叉熵损失为：

L_main = -(1/N) \Sigma_i log( exp(s(q_i, x_i)/\tau) / \Sigma_j exp(s(q_i, x_j)/\tau) ). (6)

**其他基线。** 如第2节所述，我们使用双塔DNN作为主任务的基线模型。与经典的矩阵分解和分类模型相比，双塔模型具有编码item特征的独特性质。而后两种方法虽然也适用于大规模item检索，但它们仅基于ID学习item嵌入，因此不符合我们使用自监督学习来利用item特征关系的提议。

## 4 离线实验

我们提供了经验结果来证明所提出的自监督框架在学术公共数据集和实际大规模推荐产品中的有效性。实验旨在回答以下研究问题。

• RQ1：所提出的自监督学习框架是否能改进用于推荐的深度模型？

• RQ2：自监督学习旨在通过引入对未标注样本的自监督学习任务来改进主要监督任务。训练数据量对自监督学习带来的改进有何影响？

• RQ3：自监督学习参数（即损失乘子\alpha和数据增强中的dropout率）如何影响模型质量？

• RQ4：随机特征掩码与相关特征掩码相比表现如何？在数据增强中利用特征相关性有什么好处？

上述问题将在第4.3-4.5节中依次解答。

### 4.1 数据集

我们在两个大规模数据集上进行实验，这两个数据集都带有丰富的item元数据特征。我们将它们的主要监督任务形式化为一个item到item推荐问题，以研究自监督学习对训练推荐器（在本例中为检索）模型的影响。

**Wikipedia [14]：** 第一个数据集聚焦于Wikipedia页面之间的链接预测问题。它由页面对(x, y) \in \chi $\times$ \chi组成，其中x表示源页面，y是从x链接的目标页面。目标是从整个网页语料库中预测可能链接到给定源页面的页面集。每个页面由一个特征向量x = (x_id, x_ngrams, x_cats)表示，其中所有特征都是类别型的。这里，x_id表示页面URL的独热编码，x_ngrams表示页面标题n-gram集合的词袋表示，x_cats表示页面所属类别的词袋表示。我们按照[23]和[39]中的相同处理方式，将数据集按(90%, 10%)的比例划分为训练集和评估集。

**应用间安装 (AAI)：** AAI数据集是从一个商业移动应用商店的应用落地页收集的。在特定应用（种子应用）的落地页上，收集了推荐应用部分的应用安装（候选应用）。每个训练样本表示一个种子-候选应用对(x_seed, x_candidate)及其元数据特征。目标是在给定种子应用的情况下推荐高度相似的应用。这也通过多类分类损失形式化为一个item到item推荐问题。注意，我们只收集正样本，即x_candidate是从x_seed落地页安装的应用。所有展示过但没有安装的推荐应用都被忽略，因为我们认为它们更像是弱正样本而非负样本，对于构建检索模型而言。每个item（应用）由包含以下特征的特征向量x表示：

• id：应用ID，作为独热类别型特征。
• developer_name：应用开发者的名称，作为独热类别型特征。
• categories：应用的语义类别，作为多热类别型特征。
• title_unigram：应用标题的一元组，作为多热类别型特征。

表1显示了Wikipedia和AAI数据集的一些基本统计信息。图4显示了两个数据集最频繁item的CDF，表明了高度倾斜的数据分布。例如，AAI数据集中前50个item在训练数据中总共出现了约10%。如果我们考虑一个朴素的基线（即TopPopular推荐器[11]），它为每个查询推荐最频繁的top-K个item，那么第K个频繁item的CDF本质上代表了该基线的Recall@K指标。这表明朴素的TopPopular推荐器在AAI上达到Recall@50 \approx 0.1，在Wikipedia上达到Recall@50 \approx 0.05。我们在第4节中展示所有提出的方法都大幅优于该基线。

### 4.2 实验设置

**骨干网络。** 对于预测给定查询的相关item的主任务，我们使用双塔DNN对查询和item特征进行编码（见图1）作为骨干网络。item到item推荐问题被形式化为一个多类分类问题，使用方程(6)中提出的批softmax损失作为损失函数。关于骨干网络选择的讨论，请读者参阅第2节和第3.3节的相关部分。

**超参数。** 对于骨干双塔DNN，我们搜索一组超参数，如学习率、softmax温度(\tau)和模型架构，以在验证集上获得最高的Recall@50。注意，批softmax中的训练批次大小对模型质量至关重要，因为它决定了每个正item使用的负样本数量。在本节中，我们对Wikipedia和AAI分别使用批次大小1024和4096。我们还为基础模型调整了隐藏层数量、隐藏层大小和softmax温度\tau。对于Wikipedia数据集，我们使用softmax温度\tau = 0.07，隐藏层大小为[1024, 128]。对于AAI，我们使用\tau = 0.06和隐藏层[1024, 256]。注意，最后一个隐藏层的维度也是最终查询和item嵌入的维度。所有模型均使用学习率0.01的Adagrad[13]优化器进行训练。

我们考虑两个自监督学习参数：1) 方程(5)中的自监督学习损失乘子\alpha，以及2) 数据增强第二阶段中的特征dropout率，记为dr（见第3.2节）。对于每种增强方法（例如CFM、RFM），我们在\alpha = [0.1, 0.3, 1.0, 3.0]和dr = [0.1, 0.2, ..., 0.9]范围内对这两个参数进行网格搜索，并报告最佳结果。

**评估。** 为了评估给定种子item时的推荐性能，我们从整个语料库中计算并找出具有最高余弦相似度的top K个item，并基于检索到的K个item评估质量。注意，考虑到数据集的稀疏性和语料库中大量的item，这是一个相对具有挑战性的任务。我们采用流行的标准指标Recall@K和平均准确率均值(MAP@K)来评估推荐性能[18]。对于每种实验配置，我们运行实验5次并报告平均值。

### 4.3 自监督学习结合相关特征掩码的有效性

为了回答RQ1，我们首先评估自监督学习对模型质量的影响。我们专注于使用CFM后接dropout作为数据增强技术。我们将在第4.5节展示CFM相对于其他变体的优越性能。

我们考虑三种基线方法：

• 基线：使用双塔DNN架构的原始骨干网络。

• 特征Dropout (FD) [35]：在监督学习任务中对item塔应用随机特征dropout的骨干模型。item特征上的特征dropout可以视为数据增强。FD与我们的方法相比没有额外的自监督学习正则化。

• 分散正则化 (SO) [41]：在item塔上应用分散正则化作为正则化的骨干模型。SO正则化与我们自监督学习框架中的对比损失相似。然而，它是在没有数据增强的原始样本上应用对比学习，因此与我们的方法不同。

选择后两种方法是因为它们(1) 是模型无关的，并且可扩展到工业级推荐系统；(2) 兼容类别型稀疏特征以改进泛化。此外，FD可以视为一项消融研究，以隔离对比学习的潜在改进。类似地，包含SO是为了隔离特征增强带来的改进。

我们观察到，在完整数据集上（见表2），CFM与无自监督学习正则化技术相比持续表现最佳。在AAI上，CFM相对次优方法提升了8.69%，在Wikipedia上提升了3.98%。这有助于回答RQ1：所提出的自监督学习框架和任务确实改进了推荐器的模型性能。通过比较CFM和SO，表明数据增强对于自监督学习正则化获得更好性能至关重要。如果没有任何数据增强，所提出的自监督学习方法就退化为SO。通过比较CFM和FD，我们发现特征增强在应用于自监督学习任务时比作为标准正则化技术应用于监督任务时更有效。注意，FD作为一种在某些情况下改善泛化的知名方法，是将特征增强与监督训练结合使用。

**头尾分析。** 为了理解自监督学习带来的增益，我们进一步按item流行度将整体性能分解到不同的item切片上。

我们根据真实item的流行度划分完整测试数据集。对于AAI测试数据集，头部数据集包含真实item属于前10%最频繁item的样本，其余测试样本被视为尾部。对于Wikipedia，我们遵循[23]中的数据划分，其中包含训练集中未出现item的测试样本被视为尾部，其余测试样本被视为头部。我们的假设是自监督学习通常有助于改进缺乏监督的切片（例如尾部item）上的性能。在尾部和头部测试集上评估的结果报告在表4中。我们观察到所提出的自监督学习方法在头部和尾部item推荐上都提升了性能，其中尾部item的增益更大。例如，在AAI中，CFM在尾部item上的Recall@10提升了超过51.5%，而头部item的提升为8.57%。

**自监督学习参数的影响 (RQ3)。** 图6总结了在Wikipedia和AAI数据集上关于正则化强度\alpha的Recall@50。它还显示了使用相同正则化参数的SO的结果。我们观察到，随着\alpha增加，超过某个阈值后，模型性能比基线模型（以虚线显示）更差。这是预期的，因为大的自监督学习权重\alpha导致多任务损失L在方程(5)中被\alpha·L_self主导。通过进一步将我们的方法与SO比较，我们展示了基于自监督学习的正则化在广泛的\alpha范围内优于SO。图7显示了不同dropout率dr下的模型性能。它还显示了使用相同参数的DO。随着dr增加，DO的模型性能持续恶化。对于大多数\alpha的取值（除了\alpha = 0.1），DO比基线更差。对于带有特征dropout的自监督学习任务，模型性能在dr = 0.3时达到峰值，然后随着进一步增加dropout率而恶化。当dr过大时，模型开始低于基线。这一观察与我们的预期一致，即当dropout率过大时，输入信息变得太少，无法通过自监督学习学到有意义的表示。

**item表示的可视化。** 除了更好的模型性能，我们期望使用自监督学习学到的表示比没有自监督学习的表示具有更好的质量。为了验证我们的假设，我们取在AAI数据集上训练的模型中学到的app嵌入，并使用t-SNE图在图5中绘制它们。不同类别的app以不同颜色绘制，如图5的图例所示。与图5a中的app相比，最佳自监督学习模型（图5b）中的app倾向于与同一类别中的相似app更好地聚集在一起，不同类别的分离看起来更加清晰。例如，我们可以看到在图5a中，"体育与娱乐"类app（红色）与"法律与政府"和"旅行"类app混在一起。而在图5b中，我们清楚地看到4个类别的app各自聚集在一起。这表明使用自监督学习学到的表示携带了更多语义信息，这也是自监督学习在我们的实验中带来更好模型性能的原因。

### 4.4 数据稀疏性

我们研究了CFM在稀疏数据下的有效性，以回答RQ2。我们均匀下采样10%的训练数据，并在相同的（完整）测试数据集上进行评估。实验结果报告在表3中。随着数据稀疏性的增加，CFM在Wikipedia和AAI上分别显示出更大的改进。具体而言，在完整Wikipedia数据集上的CFM相比基线改进了Recall@10的6.1%，而在10%数据集上的相对改进为20.6%。在AAI数据集上也观察到类似趋势（下采样数据集上为10.2% vs 25.7%）。值得注意的是，CFM持续优于FD，并且随着数据变得更稀疏，差距变得更大。这表明在自监督学习任务中使用dropout进行数据增强比在监督任务中直接应用dropout更有效。

总结来说，这些发现回答了RQ2中提出的研究问题：所提出的自监督学习框架在监督更少的情况下对模型性能的提升更大。

### 4.5 不同数据增强的比较

在本节中，我们将几种特征增强替代方案与CFM进行比较，以回答RQ4，研究：1) 在掩码中利用特征相关性的好处，以及2) 使用dropout作为增强的一部分的好处。具体来说，我们考虑以下替代方案：

• RFM：随机特征掩码。在这种方法中，随机掩码特征集，而不是像CFM中那样通过互信息指导。

• RFM_no_compl：随机特征掩码，不使用互补特征集。在这种方法中，随机独立掩码两个特征集，而不是像CFM中那样使用互补掩码对。

• CFM_no_dropout：相关特征掩码，不应用dropout。换句话说，在自监督学习任务中仅应用相关掩码作为增强。

• NoMasking：相关特征掩码但跳过增强中的掩码阶段。换句话说，我们仅对特征应用dropout作为增强。

我们在自监督学习框架中应用了这些特征增强函数，并在AAI数据集上报告了表5中的结果。

首先，我们观察到所有变体都比CFM差，但仍然优于基线模型。特别是，我们看到在掩码集选择中利用互信息对于模型改进至关重要，因为我们看到最大的性能下降来自RFM，其中掩码集是随机选择的。通过比较CFM与两种允许通过独立dropout在两个增强样本之间特征重叠的方法（RFM_no_compl和NoMasking）的结果，我们看到对比学习任务在具有互补信息时更有帮助，这可能避免了学习中的捷径。最后，通过比较CFM_no_dropout和CFM，我们看到第二阶段随机丢弃特征值也有帮助，这可能是因为在自监督学习任务中引入了更多的特征变体。

## 5 在线实验

在本节中，我们描述如何将所提出的自监督学习框架部署到一个网络级商业应用推荐系统中。具体而言，给定一个应用作为查询，系统识别与查询相似的应用。提供此推荐的一个模型是在第4.1节描述的AAI数据集上训练的，具有与图1中双塔DNN结构相同的骨干网络结构（进行了修改）。作为第4.3节对AAI实验进行的离线实验的自然延伸，我们进行了一项A/B实验，以调查将最佳基于自监督学习的模型部署在线的协同效应。虽然我们已经在该数据集上展示了改进的离线指标，但在许多现实系统中，离线研究可能与在线效果不一致，原因是：1) 缺乏隐式反馈，因为离线评估数据是通过基于生产系统的用户互动历史收集的；2) 未能捕获产品的多目标优化目标，推荐更具吸引力的应用很可能损害其他业务目标。因此，这个实验对于证明所提出框架在真实环境中的有效性至关重要。

在我们的在线A/B测试中，我们在生产环境中已有的经过良好调整的双塔DNN模型之上，添加了具有相同超参数集的最佳自监督学习任务。在14天的时间范围内，该模型显著改进了整体业务指标，关键用户参与度提升+0.67%（图9a），顶级业务指标提升+1.5%（图8a）。为了呼应第4.3节中的头尾分析和第4.4节中的数据稀疏性分析，我们看到了两个切片上的显著改进：1) 新应用的冷启动：该模型将新应用的用户参与度提升了+4.5%（图9b）；以及(2) 与主要市场相比训练数据更稀疏的国际国家：我们看到了显著的+5.47%顶级业务指标增益（图8b右）。再次，这两个结果验证了我们的假设：我们的自监督学习框架确实显著改进了缺乏监督的数据切片上的模型性能。基于这些结果，该自监督学习增强模型已成功部署到当前的生产系统中。

## 6 结论

在本文中，我们提出了一个与模型架构无关的自监督学习框架，用于大规模神经推荐模型。在自监督学习框架内，我们还引入了一种适用于异质类别型特征的新颖数据增强方法，并展示了其相对于其他变体的优越性能。

未来的工作，我们计划研究不同的训练方案如何影响模型质量。一个方向是首先在自监督学习任务上进行预训练以学习查询和item表示，然后在主要监督任务上进行微调。或者，将该技术扩展到搜索排序或pCTR预测等应用领域中的深度模型也将是很有趣的。

## 参考文献

[1] Alex Beutel, Ed H. Chi, Zhiyuan Cheng, Hubert Pham, and John Anderson. [n. d.]. Beyond Globally Optimal: Focused Learning for Improved Recommendations. In WWW 2017.

[2] L. Beyer, X. Zhai, A. Oliver, and A. Kolesnikov. [n. d.]. S4L: Self-Supervised Semi-Supervised Learning. In ICCV 2019.

[3] Wei-Cheng Chang, Felix X. Yu, Yin-Wen Chang, Yiming Yang, and Sanjiv Kumar. [n. d.]. Pre-training Tasks for Embedding-based Large-scale Retrieval. In ICLR 2020.

[4] Tianqi Chen and Carlos Guestrin. [n. d.]. XGBoost: A Scalable Tree Boosting System. In KDD 2016.

[5] Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey E. Hinton. 2020. A Simple Framework for Contrastive Learning of Visual Representations. https://arxiv.org/abs/2002.05709

[6] Ting Chen, Simon Kornblith, Kevin Swersky, Mohammad Norouzi, and Geoffrey Hinton. 2020. Big Self-Supervised Models are Strong Semi-Supervised Learners. arXiv preprint arXiv:2006.10029 (2020).

[7] Heng-Tze Cheng, Levent Koc, Jeremiah Harmsen, Tal Shaked, Tushar Chandra, Hrishi Aradhye, Glen Anderson, Greg Corrado, Wei Chai, Mustafa Ispir, Rohan Anil, Zakaria Haque, Lichan Hong, Vihan Jain, Xiaobing Liu, and Hemal Shah. [n. d.]. Wide & Deep Learning for Recommender Systems (DLRS 2016).

[8] Evangelia Christakopoulou and George Karypis. [n. d.]. Local Latent Space Models for Top-N Recommendation.

[9] Edith Cohen and David D. Lewis. [n. d.]. Approximating Matrix Multiplication for Pattern Recognition Tasks. In SODA 1997.

[10] Paul Covington, Jay Adams, and Emre Sargin. [n. d.]. Deep Neural Networks for YouTube Recommendations. In RecSys 2016.

[11] Maurizio Ferrari Dacrema, Paolo Cremonesi, and Dietmar Jannach. [n. d.]. Are We Really Making Much Progress? A Worrying Analysis of Recent Neural Recommendation Approaches. In RecSys 2019.

[12] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. [n. d.]. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. In NAACL-HLT 2019.

[13] John Duchi, Elad Hazan, and Yoram Singer. 2011. Adaptive Subgradient Methods for Online Learning and Stochastic Optimization. J. Mach. Learn. Res. 12, null (July 2011), 2121–2159.

[14] Wikimedia Foundation. [n. d.]. Wikimedia. https://dumps.wikimedia.org/

[15] Spyros Gidaris, Praveer Singh, and Nikos Komodakis. [n. d.]. Unsupervised Representation Learning by Predicting Image Rotations. In ICLR 2018.

[16] Daniel Gillick, Alessandro Presta, and Gaurav Singh Tomar. 2018. End-to-End Retrieval in Continuous Space. CoRR abs/1811.08008 (2018). http://arxiv.org/abs/1811.08008

[17] Chuan Guo, Ali Mousavi, Xiang Wu, Daniel N Holtmann-Rice, Satyen Kale, Sashank Reddi, and Sanjiv Kumar. 2019. Breaking the Glass Ceiling for Embedding-Based Classifiers for Large Output Spaces. In Neurips, H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and R. Garnett (Eds.).

[18] Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu, and Tat-Seng Chua. [n. d.]. Neural Collaborative Filtering. In WWW 2017.

[19] Xinran He, Junfeng Pan, Ou Jin, Tianbing Xu, Bo Liu, Tao Xu, Yanxin Shi, Antoine Atallah, Ralf Herbrich, Stuart Bowers, and Joaquin Quiñonero Candela. 2014. Practical Lessons from Predicting Clicks on Ads at Facebook. In Proceedings of the Eighth International Workshop on Data Mining for Online Advertising.

[20] Alexander Kolesnikov, Xiaohua Zhai, and Lucas Beyer. [n. d.]. Revisiting Self-Supervised Visual Representation Learning. In CVPR 2019.

[21] Yehuda Koren, Robert Bell, and Chris Volinsky. 2009. Matrix Factorization Techniques for Recommender Systems. Computer 42, 8 (Aug. 2009), 30–37.

[22] Yehuda Koren and Robert M. Bell. 2015. Advances in Collaborative Filtering. Springer, 77–118.

[23] Walid Krichene, Nicolas Mayoraz, Steffen Rendle, Li Zhang, Xinyang Yi, Lichan Hong, Ed Chi, and John Anderson. [n. d.]. Efficient Training on Very Large Corpora via Gramian Estimation. In ICLR 2019.

[24] Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, and Radu Soricut. [n. d.]. ALBERT: A Lite BERT for Self-supervised Learning of Language Representations. In ICLR 2020.

[25] Gustav Larsson, Michael Maire, and Gregory Shakhnarovich. [n. d.]. Learning Representations for Automatic Colorization. In ECCV 2016.

[26] David C. Liu, Stephanie Rogers, Raymond Shiau, Dmitry Kislyuk, Kevin C. Ma, Zhigang Zhong, Jenny Liu, and Yushi Jing. [n. d.]. Related Pins at Pinterest: The Evolution of a Real-World Recommender System. In WWW 2017.

[27] Jianxin Ma, Chang Zhou, Hongxia Yang, Peng Cui, Xin Wang, and Wenwu Zhu. [n. d.]. Disentangled Self-Supervision in Sequential Recommenders. In KDD 2020.

[28] Klaas Bosteels Mark Levy. [n. d.]. Music Recommendation and the Long Tail. In 1st Workshop On Music Recommendation And Discovery (WOMRAD), ACM RecSys, 2010.

[29] Rishabh Mehrotra, Mounia Lalmas, Doug Kenney, Thomas Lim-Meng, and Golli Hashemian. [n. d.]. Jointly Leveraging Intent and Interaction Signals to Predict User Satisfaction with Slate Recommendations. In WWW 2019.

[30] Staša Milojević. 2010. Power Law Distributions in Information Science: Making the Case for Logarithmic Binning. J. Am. Soc. Inf. Sci. Technol. 61, 12 (Dec. 2010), 2417–2425.

[31] Maxim Naumov, Dheevatsa Mudigere, Hao-Jun Michael Shi, Jianyu Huang, Narayanan Sundaraman, Jongsoo Park, Xiaodong Wang, Udit Gupta, Carole-Jean Wu, Alisson G. Azzolini, Dmytro Dzhulgakov, Andrey Mallevich, Ilia Cherniavskii, Yinghai Lu, Raghuraman Krishnamoorthi, Ansha Yu, Volodymyr Kondratenko, Stephanie Pereira, Xianjie Chen, Wenlin Chen, Vijay Rao, Bill Jia, Liang Xiong, and Misha Smelyanskiy. 2019. Deep Learning Recommendation Model for Personalization and Recommendation Systems. CoRR abs/1906.00091 (2019).

[32] Wei Niu, James Caverlee, and Haokai Lu. [n. d.]. Neural Personalized Ranking for Image Recommendation. In WSDM 2018.

[33] Mehdi Noroozi and Paolo Favaro. [n. d.]. Unsupervised Learning of Visual Representations by Solving Jigsaw Puzzles. In ECCV 2016.

[34] Shumpei Okura, Yukihiro Tagami, Shingo Ono, and Akira Tajima. [n. d.]. Embedding-Based News Recommendation for Millions of Users. In KDD 2017.

[35] Maksims Volkovs, Guangwei Yu, and Tomi Poutanen. [n. d.]. DropoutNet: Addressing Cold Start in Recommender Systems. In Neurips 2017.

[36] Zhirong Wu, Yuanjun Xiong, Stella Yu, and Dahua Lin. 2018. Unsupervised Feature Learning via Non-Parametric Instance-level Discrimination. CoRR abs/1805.01978 (2018). http://arxiv.org/abs/1805.01978

[37] Xin Xin, Alexandros Karatzoglou, I. Arapakis, and J. Jose. [n. d.]. Self-Supervised Reinforcement Learning for Recommender Systems. SIGIR 2020 ([n. d.]).

[38] Yinfei Yang, Steve Yuan, Daniel Cer, Sheng-yi Kong, Noah Constant, Petr Pilar, Heming Ge, Yun-Hsuan Sung, Brian Strope, and Ray Kurzweil. 2018. Learning Semantic Textual Similarity from Conversations. In Proceedings of The Third Workshop on Representation Learning for NLP. ACL, 164–174.

[39] Xinyang Yi, Ji Yang, Lichan Hong, Derek Zhiyuan Cheng, Lukasz Heldt, Aditee Kumthekar, Zhe Zhao, Li Wei, and Ed Chi. [n. d.]. Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations. In RecSys 2019.

[40] Andrew Zhai, Dmitry Kislyuk, Yushi Jing, Michael Feng, Eric Tzeng, Jeff Donahue, Yue Li Du, and Trevor Darrell. [n. d.]. Visual Discovery at Pinterest. In WWW 2017.

[41] Xu Zhang, Felix X. Yu, Sanjiv Kumar, and Shih-Fu Chang. [n. d.]. Learning Spread-Out Local Feature Descriptors. In ICCV 2017.

[42] Zhe Zhao, Lichan Hong, Li Wei, Jilin Chen, Aniruddh Nath, Shawn Andrews, Aditee Kumthekar, Maheswaran Sathiamoorthy, Xinyang Yi, and Ed Chi. [n. d.]. Recommending What Video to Watch next: A Multitask Ranking System. In RecSys 2019.

[43] Kun Zhou, Haibo Wang, Wayne Xin Zhao, Yutao Zhu, Sirui Wang, Fuzheng Zhang, Zhong yuan Wang, and Jirong Wen. [n. d.]. S3-Rec: Self-Supervised Learning for Sequential Recommendation with Mutual Information Maximization. CIKM 2020 ([n. d.]).
