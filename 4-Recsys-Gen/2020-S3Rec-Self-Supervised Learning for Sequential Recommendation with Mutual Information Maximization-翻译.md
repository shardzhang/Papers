# S3-Rec: 面向序列推荐的自监督学习与互信息最大化

> ¹School of Information, Renmin University of China
本文介绍了 S3-Rec: 面向序列推荐的自监督学习与互信息最大化。核心内容：
关键发现：
---
# S3-Rec: Self-Supervised Learning for Sequential Recommendation with Mutual Information Maximization（中文翻译）
## 作者与机构
Kun Zhou¹†, Hui Wang¹†, Wayne Xin Zhao²,³∗, Yutao Zhu⁵, Sirui Wang⁴,
Fuzheng Zhang⁴, Zhongyuan Wang⁴ and Ji-Rong Wen²,³
²Gaoling School of Artificial Intelligence, Renmin University of China
³Beijing Key Laboratory of Big Data Management and Analysis Methods
⁴Meituan-Dianping Group
⁵Université de Montréal, Montréal, Québec, Canada
francis_kun_zhou@163.com, {hui.wang, batmanfly, jrwen}@ruc.edu.cn, yutao.zhu@umontreal.ca,
wangsirui@meituan.com, zhfzhkris@outlook.com
---
## 摘要
ABSTRACT
最近，基于深度学习的序列推荐取得了显著进展。现有的神经序列推荐模型通常依赖item预测损失来学习模型参数或数据表示。然而，使用该损失训练的模型容易遭受数据稀疏问题。
Recently, significant progress has been made in sequential recommendation with deep learning. Existing neural sequential recommendation models usually rely on the item prediction loss to learn model parameters or data representations. However, the model trained with this loss is prone to suffer from data sparsity problem.
由于它过分强调最终性能，上下文数据和序列数据之间的关联或融合尚未被很好地捕获和利用用于序列推荐。
Since it overemphasizes the final performance, the association or fusion between context data and sequence data has not been well captured and utilized for sequential recommendation.
为解决这一问题，我们提出了S3-Rec模型，即面向序列推荐的自监督学习，基于自注意力神经架构。我们方法的主要思想是利用内在数据相关性来推导自监督信号，并通过预训练方法增强数据表示以改进序列推荐。
To tackle this problem, we propose the model S3-Rec, which stands for Self-Supervised learning for Sequential Recommendation, based on the self-attentive neural architecture. The main idea of our approach is to utilize the intrinsic data correlation to derive self-supervision signals and enhance the data representations via pre-training methods for improving sequential recommendation.
对于我们的任务，我们设计了四个辅助自监督目标来学习属性、item、子序列和序列之间的相关性，利用互信息最大化原理。
For our task, we devise four auxiliary self-supervised objectives to learn the correlations among attribute, item, subsequence, and sequence by utilizing the mutual information maximization (MIM) principle.
互信息最大化提供了一种统一的方式来表征不同类型数据之间的相关性，这特别适合我们的场景。
MIM provides a unified way to characterize the correlation between different types of data, which is particularly suitable in our scenario.
在六个真实世界数据集上进行的大量实验表明，我们提出的方法优于现有的最先进方法，特别是在仅有有限训练数据可用的情况下。
Extensive experiments conducted on six real-world datasets demonstrate the superiority of our proposed method over existing state-of-the-art methods, especially when only limited training data is available.
此外，我们将我们的自监督学习方法扩展到其他推荐模型，这也提升了它们的性能。
Besides, we extend our self-supervised learning method to other recommendation models, which also improve their performance.
## CCS概念
CCS CONCEPTS
• 信息系统 $\to$ 推荐系统。
• Information systems $\to$ Recommender systems.
†Equal contribution.
†共同贡献。
∗Corresponding author.
∗通讯作者。
Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than ACM must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.
允许制作本工作的全部或部分的数字或硬拷贝用于个人或课堂使用，无需付费，前提是拷贝不是为了盈利或商业优势，且拷贝带有本声明和首页的完整引用。必须尊重由他人拥有并为ACM所有的本工作的组件的版权。允许在注明出处的情况下进行摘要。如需以其他方式复制、重新发布、发布到服务器或重新分发到列表，需要事先获得特定许可和/或支付费用。请向permissions@acm.org请求许可。
CIKM '20, October 19–23, 2020, Virtual Event, Ireland
CIKM '20，2020年10月19–23日，爱尔兰，虚拟会议
© 2020 Association for Computing Machinery.
© 2020 美国计算机协会。
ACM ISBN 978-1-4503-6859-9/20/10. . . $15.00$
ACM ISBN 978-1-4503-6859-9/20/10. . . $15.00$
https://doi.org/10.1145/3340531.3411954
---
## 关键词
KEYWORDS
自监督学习，序列推荐，互信息最大化
Self-Supervised Learning, Sequential Recommendation, Mutual Information Maximization
## ACM引用格式
ACM Reference Format:
Kun Zhou¹†, Hui Wang¹†, Wayne Xin Zhao², ³∗, Yutao Zhu⁵, Sirui Wang⁴, and Fuzheng Zhang⁴, Zhongyuan Wang⁴ and Ji-Rong Wen², ³. 2020. S3-Rec: Self-Supervised Learning for Sequential Recommendation with Mutual Information Maximization. In The 29th ACM International Conference on Information and Knowledge Management (CIKM '20), October 19–23, 2020, Virtual Event, Ireland. ACM, New York, NY, USA, 10 pages. https://doi.org/10.1145/3340531.3411954
Kun Zhou¹†, Hui Wang¹†, Wayne Xin Zhao², ³∗, Yutao Zhu⁵, Sirui Wang⁴, and Fuzheng Zhang⁴, Zhongyuan Wang⁴ and Ji-Rong Wen², ³. 2020. S3-Rec: Self-Supervised Learning for Sequential Recommendation with Mutual Information Maximization. In The 29th ACM International Conference on Information and Knowledge Management (CIKM '20), October 19–23, 2020, Virtual Event, Ireland. ACM, New York, NY, USA, 10 pages. https://doi.org/10.1145/3340531.3411954
---
## 1 引言
1 INTRODUCTION
近年来，许多在线平台取得了巨大成功，例如Amazon和淘宝。在在线平台中，用户的行为是动态且随时间演变的。因此，捕获顺序用户行为的动态变化对于做出适当的推荐至关重要。为了准确刻画用户兴趣并提供高质量的推荐，序列推荐任务已在文献中得到广泛研究[3, 8, 20, 21, 24]。
Recent years have witnessed the great success of many online platforms, such as Amazon and Taobao. Within online platforms, users' behaviors are dynamic and evolving over time. Thus it is critical to capture the dynamics of sequential user behaviors for making appropriate recommendations. In order to accurately characterize user interests and provide high-quality recommendations, the task of sequential recommendation has been widely studied in the literature [3, 8, 20, 21, 24].
通常，序列推荐方法[3, 8, 21, 24]从用户的历史行为中捕获有用的序列模式。这种动机已经通过深度学习得到了广泛探索。使用循环神经网络、卷积神经网络和自注意力机制的各种方法已被提出来学习用户偏好的良好表示并刻画序列化的用户-item交互。
Typically, sequential recommendation methods [3, 8, 21, 24] capture useful sequential patterns from users' historical behaviors. Such motivation has been extensively explored with deep learning. Various methods using recurrent neural networks (RNNs) [3], convolutional neural networks (CNNs) [24], and self-attention mechanisms [8] have been proposed to learn good representations of user preference and characterize sequential user-item interactions.
此外，研究人员已将丰富的上下文信息（如item属性）引入神经序列推荐器[4, 6, 29]。已经证明，上下文信息对于提升序列推荐系统的性能是重要的。
Furthermore, researchers have incorporated rich contextual information (such as item attributes) to neural sequential recommenders [4, 6, 29]. It has been demonstrated that contextual information is important to consider for improving the performance of sequential recommender systems.
尽管现有方法在一定程度上已被证明是有效的，但存在两个可能影响推荐性能的主要缺点。首先，它们依赖item预测损失来学习整个模型。当引入上下文数据时，涉及的参数也通过唯一的优化目标来学习。已经发现，这种优化方式容易遭受数据稀疏等问题[21, 22]。
Although existing methods have been shown effective to some extent, there are two major shortcomings that are likely to affect the recommendation performance. First, they rely on the item prediction loss to learn the entire model. When context data is incorporated, the involved parameters are also learned through the only optimization objective. It has been found that such an optimization way is easy to suffer from issues such as data sparsity [21, 22].
其次，它们过分强调最终性能，而上下文数据和序列数据之间的关联或融合尚未在数据表示中被很好地捕获。正如来自各个领域的越来越多证据所示[1, 5, 10]，有效的数据表示（例如，预训练的上下文化嵌入）已成为改进现有模型或架构性能的关键因素。因此，需要重新思考学习范式，以开发更有效的序列推荐系统。
Second, they overemphasize the final performance, while the association or fusion between context data and sequence data has not been well captured in data representations. As shown in increasing evidence from various fields [1, 5, 10], effective data representation (e.g., pre-trained contextualized embedding) has been a key factor to improve the performance of existing models or architectures. Therefore, there is a need to rethink the learning paradigm to develop more effective sequential recommender systems.
为了解决上述问题，我们借鉴了自监督学习的思想来改进序列推荐。自监督学习[1, 15]是一种新兴的范式，旨在让模型从原始数据的内在结构中学习。自监督学习的一般框架是首先直接从原始数据构建训练信号，然后用额外设计的优化目标预训练模型参数。如前所述，有限的监督信号和无效的数据表示是现有神经序列方法面临的两个主要学习问题。幸运的是，自监督学习似乎为这两个问题提供了一个有前景的解决方案：它利用内在数据相关性来设计辅助训练目标，并通过带有丰富自监督信号的预训练方法来增强数据表示。然而，对于序列推荐，上下文信息以不同形式或具有不同的内在特性存在，包括item、属性、子序列或序列。开发一种统一的方法来表征这种数据相关性并不容易。对于这个问题，我们受到最近提出的互信息最大化方法的启发[5, 10, 11, 30]。它已被证明在通过最大化这些视图的编码表示之间的互信息来捕获原始输入的不同视图（或部分）之间的相关性方面特别有效。
To address the above issues, we borrow the idea of self-supervised learning for improving sequential recommendation. Self-supervised learning [1, 15] is a newly emerging paradigm, which aims to let the model learn from the intrinsic structure of the raw data. A general framework of self-supervised learning is to first construct training signals directly from the raw data and then pre-train the model parameters with additionally devised optimization objectives. As previously discussed, limited supervision signals and ineffective data representations are the two major learning issues with existing neural sequential methods. Fortunately, self-supervised learning seems to provide a promising solution to both problems: it utilizes the intrinsic data correlation to devise auxiliary training objectives and enhances the data representations via pre-trained methods with rich self-supervised signals. However, for sequential recommendation, the context information exists in different forms or with varying intrinsics, including item, attribute, subsequence, or sequence. It is not easy to develop a unified approach to characterizing such data correlations. For this problem, we are inspired by the recently proposed mutual information maximization (MIM) method [5, 10, 11, 30]. It has been shown to be particularly effective to capture the correlation between different views (or parts) of the original input by maximizing the mutual information between the encoded representations of these views.
为此，在本文中，我们提出了一种新颖的自监督学习方法，通过互信息最大化来改进序列推荐，称为S3-Rec。基于自注意力推荐器架构[8]，我们提出首先用自监督信号预训练序列推荐器，然后根据推荐任务微调模型参数。主要的创新在于预训练阶段。具体来说，我们精心设计了四个自监督优化目标，分别用于捕获item-属性、序列-item、序列-属性和序列-子序列的相关性。这些优化目标以互信息最大化的统一形式开发。因此，S3-Rec能够以通用方式表征不同粒度级别或不同形式之间的相关性。它也可以灵活适应新的数据类型或新的相关模式。通过这种预训练方法，我们可以有效融合各种上下文数据，并学习属性感知的上下文化数据表示。最后，学习到的数据表示被输入到神经推荐器中，该推荐器将根据推荐性能进行优化。
To this end, in this paper, we propose a novel Self-Supervised learning approach to improve Sequential Recommendation with MIM, which is called S3-Rec. Based on a self-attentive recommender architecture [8], we propose to first pre-train the sequential recommender with self-supervised signals and then fine-tune the model parameters according to the recommendation task. The major novelty lies in the pre-training stage. In particular, we carefully devise four self-supervised optimization objectives for capturing item-attribute, sequence-item, sequence-attribute and sequence-subsequence correlations, respectively. These optimization objectives are developed in a unified form of MIM. As such, S3-Rec is able to characterize the correlation in varying levels of granularity or between different forms in a general way. It is also flexible to adapt to new data types or new correlation patterns. Via such a pre-trained method, we can effectively fuse various kinds of context data, and learn attribute-aware contextualized data representations. Finally, the learned data representations are fed into the neural recommender, which will be optimized according to the recommendation performance.
为了验证我们提出的S3-Rec方法的有效性，我们在来自不同领域的六个真实世界推荐数据集上进行了大量实验。实验结果表明，S3-Rec相比多个竞争方法实现了最先进的性能，特别是在训练数据有限的情况下。
To validate the effectiveness of our proposed S3-Rec method, we conduct extensive experiments on six real-world recommendation datasets of different domains. Experimental results show that S3-Rec achieves state-of-the-art performance compared to a number of competitive methods, especially when training data is limited.
我们还展示了我们的S3-Rec能够有效地适应其他类别的神经架构，例如GRU和CNN。
We also show that our S3-Rec is effective to adapt to other classes of neural architectures, such as GRU and CNN.
我们的主要贡献总结如下：(1)据我们所知，这是首次将带有互信息最大化的自监督学习应用于改进序列推荐任务；(2)我们提出了四个自监督优化目标，以最大化不同形式或粒度的上下文信息的互信息；(3)在六个真实世界数据集上进行的大量实验证明了我们提出的方法的有效性。
Our main contributions are summarized as follows: (1) To the best of our knowledge, it is the first time that self-supervised learning with MIM has been applied to improve the sequential recommendation task; (2) We propose four self-supervised optimization objectives to maximize the mutual information of context information in different forms or granularities; (3) Extensive experiments conducted on six real-world datasets demonstrate the effectiveness of our proposed approach.
---
## 2 相关工作
2 RELATED WORK
### 2.1 序列推荐
2.1 Sequential Recommendation
序列推荐的早期工作基于马尔可夫链假设。基于马尔可夫链的方法[20]估计了一个item-item转移概率矩阵，并利用它在给定用户最后一次交互的情况下预测下一个item。一系列工作遵循这一方向并将其扩展到高阶马尔可夫链[4, 8, 24]。随着神经网络的发展，Hidasi等人[3]首次将门控循环单元引入基于会话的推荐，随后大量变体通过引入成对损失函数[4]、记忆网络[6, 7]、层次结构[17]、复制机制[18]和强化学习[27]等来修改该模型。也有研究利用其他架构[8, 23, 24]进行序列推荐。然而，这些方法忽略了关于item的丰富属性信息。为了解决这个问题，TransFM[16]利用因子分解机将任意实值特征引入序列推荐。FDSA[29]采用特征级自注意力块来利用用户历史中item的属性信息。尽管这些序列推荐模型取得了显著成功，但属性、item和序列之间的相关性仍然没有得到充分利用和充分建模。
Early works on sequential recommendation are based on the Markov Chain assumption. MC-based methods [20] estimated an item-item transition probability matrix and utilized it to predict the next item given the last interaction of a user. A series of works follow this line and extend it for high-order MCs [4, 8, 24]. With the development of the neural networks, Hidasi et al. [3] firstly introduced Gated Recurrent Units (GRU) to the session-based recommendation and a surge of following variants modified this model by introducing pair-wise loss functions [4], memory networks [6, 7], hierarchical structures [17], copy mechanism [18] and reinforcement learning [27], etc. There are also studies that leverage other architectures [8, 23, 24] for sequential recommendation. However, these approaches neglect the rich attribute information about items. To tackle this problem, TransFM [16] utilized Factorization Machines to incorporate arbitrary real-valued features to the sequential recommendation. FDSA [29] employed a feature-level self-attention block to leverage the attribute information about items in user history. Despite the remarkable success of these sequential recommendation models, the correlations among attribute, item, and sequence are still not utilized and modeled sufficiently.
### 2.2 自监督学习
2.2 Self-supervised Learning
自监督学习[1, 5, 15]旨在在一个辅助目标上训练网络，其中真实样本从原始数据中自动获取。一般框架是直接从原始数据内的相关性构建训练信号，并利用它们来训练模型。通过自监督学习学到的相关性信息随后可以轻松地用于惠及其他任务。已经引入了几个自监督目标，利用非视觉但内在相关的特征来指导视觉特征学习[5]。至于语言建模[1, 15]，这是自然语言处理中一个流行的自监督目标，其中模型学习在给定前序序列的情况下预测下一个单词或句子。学到的单词或序列的表示可以提升下游任务的性能，例如机器阅读理解[1]和自然语言理解[10]。
Self-supervised learning [1, 5, 15] aims at training a network on an auxiliary objective where the ground-truth samples are obtained from the raw data automatically. The general framework is to construct training signals directly from the correlation within the raw data and utilize them to train the model. The correlation information learned through self-supervised learning can then be easily utilized to benefit other tasks. Several self-supervised objectives have been introduced to use non-visual but intrinsically correlated features to guide the visual feature learning [5]. As for language modeling [1, 15], it is a popular self-supervised objective for natural language processing, where the model learns to predict the next word or sentence given the previous sequences. The learned representations of words or sequences can improve the performance of downstream tasks such as machine reading comprehension [1] and natural language understanding [10].
互信息最大化[5, 10, 11]是自监督学习的一个特殊分支。它受到InfoMax原理的启发[11]，并在多个领域取得了重要进展，例如计算机视觉[5]、音频处理[25]和自然语言理解[10]。该方法将输入数据拆分为多个（可能重叠的）视图，并最大化这些视图的表示之间的互信息。从其他输入派生的视图被用作负样本。
Mutual information maximization [5, 10, 11] is a special branch of the self-supervised learning. It is inspired by the InfoMax principle [11] and has made important progress in several domains such as computer vision [5], audio processing [25], and nature language understanding [10]. This method splits the input data into multiple (possibly overlapping) views and maximizes the mutual information between representations of these views. The views derived from other inputs are used as negative samples.
与上述方法不同，我们的工作是首个考虑将上下文信息内的相关性作为序列推荐中的自监督信号的工作。我们最大化属性、item和序列这些不同粒度的上下文信息视图之间的互信息。增强后的数据表示可以提升推荐性能。
Different from the above approaches, our work is the first to consider the correlations within the contextual information as the self-supervised signals in sequential recommendation. We maximize the mutual information among the views of the attribute, item, and sequence, which are in different levels of granularity of the contextual information. The enhanced data representations can improve recommendation performance.
---
## 3 预备知识
3 PRELIMINARIES
在本节中，我们首先形式化定义序列推荐问题，然后介绍互信息最大化技术。
In this section, we first formulate the sequential recommendation problem and then introduce the technique of mutual information maximization.
### 3.1 问题陈述
3.1 Problem Statement
假设我们有一组用户和一组item，分别用U和I表示，其中u
$$
\in
$$
U表示一个用户，i
$$
\in
$$
I表示一个item。用户和item的数量分别表示为|U|和|I|。通常，一个用户u有一个按时间顺序排列的item交互序列：{ $i_1$ , ..., $i_n$ }，其中n是交互次数， $i_t$ 是用户u交互过的第t个item。为方便起见，我们用 $i_j$ : $_k$ 来表示子序列，即 $i_j$ : $_k$ = { $i_j$ , ..., $i_k$ }，其中1
$$
\leq
$$
 j < k
$$
\leq
$$
 n。此外，每个itemi与几个属性 $A_i$ = { $a_1$ , ..., $a_m$ }相关联。例如，对于音乐推荐，一首歌曲通常具有辅助信息，如艺术家、专辑和流行度。所有属性构成一个属性集A，属性的数量记为|A|。
Assume that we have a set of users and items, denoted by U and I, respectively, where u
$$
\in
$$
 U denotes a user and i
$$
\in
$$
 I denotes an item. The numbers of users and items are denoted as |U| and |I|, respectively. Generally, a user u has a chronologically-ordered interaction sequence with items: { $i_1$ , ···, $i_n$ }, where n is the number of interactions and $i_t$ is the t-th item that the user u has interacted with. For convenience, we use $i_j$ : $_k$ to denote the subsequence, i.e., $i_j$ : $_k$ = { $i_j$ , ···, $i_k$ } where 1
$$
\leq
$$
 j < k
$$
\leq
$$
 n. Besides, each item i is associated with several attributes $A_i$ = { $a_1$ , ···, $a_m$ }. For example, a song is typical with auxiliary information such as artist, album, and popularity for music recommender. All attributes constitute an attribute set A, and the number of attributes is donated as |A|.
基于上述符号，我们现在定义序列推荐的任务。形式化地，给定一个用户的历史行为{ $i_1$ , ..., $i_n$ }和每个itemi的属性 $A_i$ ，序列推荐的任务是在第(n+1)步预测用户可能与之交互的下一个item。
Based on the above notations, we now define the task of sequential recommendation. Formally, given the historical behaviors of a user { $i_1$ , ···, $i_n$ } and the attributes $A_i$ of each item i, the task of sequential recommendation is to predict the next item that the user is likely to interact with at the (n+1)-th step.
### 3.2 互信息最大化
3.2 Mutual Information Maximization
我们方法中的一个重要技术是互信息最大化。它建立在互信息的核心概念之上，互信息度量随机变量之间的依赖关系。给定两个随机变量X和Y，它可以理解为知道X在多大程度上减少了Y的不确定性，反之亦然。形式上，X和Y之间的互信息为：
An important technique in our approach is the Mutual Information Maximization (MIM). It is developed on the core concept of mutual information, which measures dependencies between random variables. Given two random variables X and Y, it can be understood as how much knowing X reduces the uncertainty in Y or vice versa. Formally, the mutual information between X and Y is:
I(X, Y) = H(X) − H(X|Y) = H(Y) − H(Y|X). (1)
直接最大化互信息通常是难以处理的。因此，我们求助于I(X,Y)的一个下界。一个在实践中表现良好的特定下界是InfoNCE[10, 12, 25]，它基于噪声对比估计[2]。InfoNCE定义为：
Maximizing mutual information directly is usually intractable. Thus we resort to a lower bound on I(X,Y). One particular lower bound that has been shown to work well in practice is InfoNCE [10, 12, 25], which is based on Noise Contrastive Estimation (NCE) [2]. InfoNCE is defined as:
 $E_{p(X,Y)}$ [f_ $\theta$ (x,y) − $E_{q(Ỹ)}$ [log
$$
\sum
$$
_{ỹ
$$
\in
$$
Ỹ} exp f_ $\theta$ (x,ỹ)]] + log|Ỹ|, (2)
其中x和y是一个输入的不同视图，f_ $\theta$ 是一个由 $\theta$ 参数化的函数（例如，一个词及其上下文的编码表示之间的点积[10]，或一个图像与图像的局部区域之间的点积[5]），而Ỹ是从提议分布q(Ỹ)中抽取的一组样本，其中包含一个正样本y和|Ỹ|−1个负样本。
where x and y are different views of an input, and f_ $\theta$ is a function parameterized by $\theta$ (e.g., a dot product between encoded representations of a word and its context [10] or a dot product between encoded representations of an image and the local regions of the image [5]), and Ỹ is a set of samples drawn from a proposal distribution q(Ỹ), which contains a positive sample y and |Ỹ|−1 negative samples.
注意，InfoNCE与交叉熵相关。如果Ỹ总是包含随机变量Y的所有可能值（即Ỹ=Y）且它们均匀分布，那么最大化InfoNCE类似于最大化标准交叉熵损失：
Note that InfoNCE is related to the cross-entropy. If Ỹ always includes all possible values of the random variable Y (i.e., Ỹ=Y) and they are uniformly distributed, maximizing InfoNCE is analogous to maximize the standard cross-entropy loss:
 $E_{p(X,Y)}$ [f_ $\theta$ (x,y) − log
$$
\sum
$$
_{ỹ
$$
\in
$$
Y} exp f_ $\theta$ (x,ỹ)]. (3)
该公式表明InfoNCE与最大化p_ $\theta$ (y|x)相关，并通过负采样来近似对Y中元素的求和（即配分函数）。基于这一公式，我们可以利用特定的X,Y来最大化原始数据不同视图之间的互信息，例如一个item及其属性，或一个序列及其包含的item。
This equation shows that InfoNCE is related to maximize p_ $\theta$ (y|x), and it approximates the summation over elements in Y (i.e., the partition function) by negative sampling. Based on this formula, we can utilize specific X, Y to maximize the mutual information between different views of the raw data, e.g., an item and its attributes, or a sequence and the items that it contains.
---
## 4 方法
4 APPROACH
### 4.1 概述
4.1 Overview
现有研究[3, 4, 8, 24]主要仅使用item级优化目标来强调序列特征的效果。受互信息最大化最新进展的启发[5, 28]，我们采取不同的视角，通过最大化原始数据不同视图之间的互信息来开发神经序列推荐器。
Existing studies [3, 4, 8, 24] mainly emphasize the effect of sequential characteristics using an item-level optimization objective alone. Inspired by recent progress with MIM [5, 28], we take a different perspective to develop neural sequential recommenders by maximizing the mutual information among different views of the raw data.
我们方法的基本思想是引入几个精心设计的自监督学习目标来增强原始模型。为了设计这样的目标，我们利用输入的内在特征中反映的有效相关性信号。对于我们的任务，我们考虑不同粒度的信息，包括属性、item、片段（即子序列）和序列，它们被视为输入的不同视图。通过捕获多视图相关性，我们将这些自监督学习目标与语言建模中最近提出的预训练框架[1]统一起来。
The basic idea of our approach is to incorporate several elaborately designed self-supervised learning objectives for enhancing the original model. To develop such objectives, we leverage effective correlation signals reflected in the intrinsic characteristics of the input. For our task, we consider the information in different levels of granularity, including attribute, item, segment (i.e., subsequence), and sequence, which are considered as different views of the input. By capturing the multi-view correlation, we unify these self-supervised learning objectives with the recently proposed pre-training framework in language modeling [1].
S3-Rec的概览如图1所示。在接下来的小节中，我们首先介绍基于Transformer架构[8]开发的基础模型。然后，我们将描述如何利用属性、item、片段和序列之间的相关性信号，基于InfoNCE[10, 25]方法增强数据表示。最后，我们对我们的方法进行讨论。
The overview of S3-Rec is presented in Fig. 1. In the following sections, we first introduce the base model of our proposed approach that is developed on the Transformer architecture [8]. Then, we will describe how we utilize the correlation signals among attributes, items, segments, and sequences to enhance the data representations based on the InfoNCE [10, 25] method. Finally, we present the discussions on our approach.
### 4.2 基础模型
4.2 Base Model
我们通过堆叠嵌入层、自注意力块和预测层来开发序列推荐模型的基本框架。
We develop the basic framework for sequential recommendation model by stacking the embedding layer, self-attention blocks, and the prediction layer.
#### 4.2.1 嵌入层
4.2.1 Embedding Layer
在嵌入映射阶段，我们维护一个item嵌入矩阵Mᴵ
$$
\in
$$
ℝ^{|I|
$$
\times
$$
d}和一个属性嵌入矩阵Mᴬ
$$
\in
$$
ℝ^{|A|
$$
\times
$$
d}。这两个矩阵将item或属性的高维独热表示投影到低维稠密表示。给定一个长度为n的item序列，我们从Mᴵ中应用查找操作来形成输入嵌入矩阵E
$$
\in
$$
ℝ^{n
$$
\times
$$
d}。此外，我们引入一个可学习的位置编码矩阵P
$$
\in
$$
ℝ^{n
$$
\times
$$
d}来增强item序列的输入表示。通过这种方式，序列表示Eᴵ
$$
\in
$$
ℝ^{n
$$
\times
$$
d}可以通过求和两个嵌入矩阵得到：Eᴵ=E+P。由于我们的任务利用辅助上下文数据，我们还从整个属性嵌入矩阵Mᴬ中为每个item形成一个嵌入矩阵Eᴬ
$$
\in
$$
ℝ^{k
$$
\times
$$
d}，其中k是item属性的数量。
In the embedding mapping stage, we maintains an item embedding matrix Mᴵ
$$
\in
$$
ℝ^{|I|
$$
\times
$$
d} and an attribute embedding matrix Mᴬ
$$
\in
$$
ℝ^{|A|
$$
\times
$$
d}. The two matrices project the high-dimensional one-hot representation of an item or attribute to low-dimensional dense representations. Given a n-length item sequence, we apply a look-up operation from Mᴵ to form the input embedding matrix E
$$
\in
$$
ℝ^{n
$$
\times
$$
d}. Besides, we incorporate a learnable position encoding matrix P
$$
\in
$$
ℝ^{n
$$
\times
$$
d} to enhance the input representation of the item sequence. By this means, the sequence representation Eᴵ
$$
\in
$$
ℝ^{n
$$
\times
$$
d} can be obtained by summing two embedding matrices: Eᴵ=E+P. Since our task utilizes auxiliary context data, we also form an embedding matrix Eᴬ
$$
\in
$$
ℝ^{k
$$
\times
$$
d} for each item from the entire attribute embedding matrix Mᴬ, where k is the number of item attributes.
#### 4.2.2 自注意力块
4.2.2 Self-Attention Block
基于嵌入层，我们通过堆叠多个自注意力块来开发item编码器。一个自注意力块通常由两个子层组成，即多头自注意力层和逐点前馈网络。多头自注意力机制已被采用，用于从不同的表示子空间中选择性地有效提取信息。具体来说，多头自注意力定义为：
Based on the embedding layer, we develop the item encoder by stacking multiple self-attention blocks. A self-attention block generally consists of two sub-layers, i.e., a multi-head self-attention layer and a point-wise feed-forward network. The multi-head self-attention mechanism has been adopted for effectively extracting the information selectively from different representation subspaces. Specifically, the multi-head self-attention is defined as:
MultiHeadAttn( $F_l$ ) = [hea $d_1$ , hea $d_2$ , ..., $head_h$ ]Wᴼ,
hea $d_i$ = Attention( $F_l
$$
W_i$ Q, F$_{l}
$$
W_i$ K, F $_{l}
$$
W_i$ V),
(4) (5)
$\text{where } F_l$是第l层的输入。当l=0时，我们设$F_0$=Eᴵ，投影矩阵$W_i$ Q
$$
\in
$$
ℝ^{d
$$
\times
$$
d/h}, $W_i$ K
$$
\in
$$
ℝ^{d
$$
\times
$$
d/h}, $W_i$ V
$$
\in
$$
ℝ^{d
$$
\times
$$
d/h}, Wᴼ
$$
\in
$$
ℝ^{d
$$
\times
$$
d}是每个注意力头对应的可学习参数。注意力函数通过缩放点积操作实现：
where the $F_l$ is the input for the l-th layer. When l=0, we set $F_0$=Eᴵ, and the projection matrix $W_i$ Q
$$
\in
$$
ℝ^{d
$$
\times
$$
d/h}, $W_i$ K
$$
\in
$$
ℝ^{d
$$
\times
$$
d/h}, $W_i$ V
$$
\in
$$
ℝ^{d
$$
\times
$$
d/h} and Wᴼ
$$
\in
$$
ℝ^{d
$$
\times
$$
d} are the corresponding learnable parameters for each attention head. The attention function is implemented by scaled dot-product operation:
Attention(Q, K, V) = softmax(QKᵀ/$\sqrt{}$(d/h))V, (6)
其中Q=$F_l
$$
W_i$ Q, K=F $_{l}
$$
W_i$ K, V=F$_{l}
$$
W_i$ V是输入嵌入矩阵的线性变换， $\sqrt{}$ (d/h)是缩放因子，以避免内积值过大。
where Q= $F_l
$$
W_i$ Q, K=F$_{l}
$$
W_i$ K, and V=F $_{l}
$$
W_i$ V are the linear transformations of the input embedding matrix, and $\sqrt{}$(d/h) is the scale factor to avoid large values of the inner product.
由于多头注意力函数主要建立在线性投影之上。我们通过应用逐点前馈网络来赋予自注意力块非线性。计算定义为：
Since the multi-head attention function is mainly built on the linear projections. We endow the non-linearity of the self-attention block by applying a point-wise feed-forward network. The computation is defined as:
$F_l$ = [FFN($F_l$¹)ᵀ; ...; FFN($F_l$ⁿ)ᵀ], (7)
FFN(x) = (ReLU(x$W_1$+$b_1$))$W_2$+$b_2$, (8)
其中$W_1$, $b_1$, $W_2$, $b_2$是可训练参数。
where $W_1$, $b_1$, $W_2$, $b_2$ are trainable parameters.
在序列推荐中，只能利用当前时间步之前的信息，因此我们对多头自注意力函数的输出应用掩码操作，以移除$Q_i$和$K_j$之间的所有连接。受BERT[1]的启发，在预训练阶段，我们去除了掩码机制，以获得item序列中每个item的双向上下文感知表示。从两个方向结合上下文对于序列表示学习是有益的[1, 23]。
In sequential recommendation, only the information before the current time step can be utilized, thus we apply the mask operation for the output of the multi-head self-attention function to remove all connections between $Q_i$ and $K_j$. Inspired by BERT [1], at the pre-training stage, we remove the mask mechanism to acquire the bidirectional context-aware representation of each item in an item sequence. It is beneficial to incorporate context from both directions for sequence representation learning [1, 23].
#### 4.2.3 预测层
4.2.3 Prediction Layer
在S3-Rec的最后一层，我们根据用户历史上下文计算在第(t+1)步对itemi的用户偏好分数：
In the final layer of S3-Rec, we calculate the user's preference score for the item i in the step (t+1) under the context from user history as:
P($i_{t+1}$=i|$i_{1:t}$) = $e_i$ᵀ · F
$$
_{l}$ _t $, (9)
其中$ e_i $是来自item嵌入矩阵Mᴵ的itemi的表示，F
$$
_{l}$_t$是第L层自注意力块在第t步的输出，L是自注意力块的数量。
where $e_i$ is the representation of item i from item embedding matrix Mᴵ, F
$$
_{l}$ _t $ is the output of the L-layer self-attention block at step t and L is the number of self-attention blocks.
### 4.3 基于互信息最大化的自监督学习
4.3 Self-supervised Learning with MIM
基于上述自注意力模型，我们进一步引入额外的带有互信息最大化的自监督信号来增强输入数据的表示。我们采用预训练方式，基于多视图相关性构建不同的损失函数。
Based on the above self-attention model, we further incorporate additional self-supervised signals with MIM to enhance the representations of input data. We adopt a pre-training way to construct different loss functions based on the multi-view correlation.
#### 4.3.1 建模item-属性相关性
4.3.1 Modeling Item-Attribute Correlation
我们首先最大化item和属性之间的互信息。对于每个item，属性提供了关于它的细粒度信息。因此，我们旨在通过建模item-属性相关性来融合item级和属性级信息。通过这种方式，预期可以将有用的属性信息注入到item表示中。
We first maximize the mutual information between items and attributes. For each item, the attributes provide fine-grained information about it. Therefore, we aim to fuse item- and attribute-level information through modeling item-attribute correlation. In this way, it is expected to inject useful attribute information into item representations.
给定一个itemi和属性集$ A_i $={$ a_1 $,...,$ a_k $}，我们将item本身及其关联属性视为两个不同的视图。形式上，设$ e_i $表示通过嵌入层获得的item嵌入，$ e_{a_j} $表示第j个属性a_j
$$
\in
$$
$ A_i $的嵌入。我们通过对比学习框架设计了一个损失函数，最大化这两个视图之间的互信息。根据公式3，我们通过以下方式最小化关联属性预测损失：
Given an item i and the attribute set $ A_i $={$ a_1 $,...,$ a_k $}, we treat the item itself and its associated attributes as two different views. Formally, let $ e_i $ denote the item embedding obtained by the embedding layer, and $ e_{a_j} $ denote the embedding for the j-th attribute a_j
$$
\in
$$
$ A_i $. We design a loss function by the contrastive learning framework that maximizes the mutual information between the two views. Following Eq. 3, we minimize the Associated Attribute Prediction (AAP) loss by:
L_ AAP(i, $ A_i $) = $ E_{a_j
$$
$\in
$$
 $Ai}$ [f(i, a_j) − log
$$
\sum
$$
_{ã
$$
\in
$$
A\ $A_i$ } exp(f(i,ã))], (10)
where we sample negative attributes ã that enhance the association between the item i and the ground-truth attributes, "\" defines set subtraction operation. The function f(·,·) is implemented with a simple bilinear network:
其中我们采样负属性ã，以增强itemi与真实属性之间的关联，"\"定义集合减操作。函数f(·,·)通过一个简单的双线性网络实现：
f(i, a_j) = $\sigma$ ( $e_i$ ᵀ · W_AAP · $e_{a_j}$ ), (11)
其中W_AAP
$$
\in
$$
ℝ^{d
$$
\times
$$
d}是一个待学习的参数矩阵， $\sigma$ (·)是sigmoid函数。注意，为清晰起见，我们给出了单个item的损失定义L_AAP。在整个item集上定义该损失也很容易。
where W_AAP
$$
\in
$$
ℝ^{d
$$
\times
$$
d} is a parameter matrix to learn and $\sigma$ (·) is the sigmoid function. Note that for clarity, we give the loss definition L_AAP for a single item. It will be easy to define this loss over the entire item set.
#### 4.3.2 建模序列-item相关性
4.3.2 Modeling Sequence-Item Correlation
传统的序列推荐模型通常被训练来预测下一步的item。这种方法仅从左到右考虑item序列中的序列特征。然而，需要注意的是，在训练过程中模型确实观察到了完整的交互序列。受类似BERT[1]的掩码语言模型的启发，我们提出通过完形填空任务对item序列中的双向信息进行建模。对于我们的任务，完形填空设置如下描述：在每个训练步骤，我们随机掩码输入序列中的一定比例的item（即，将它们替换为特殊标记"[mask]"）。然后，我们根据周围的双向上下文从原始序列中预测被掩码的item。
Conventional sequential recommendation models are usually trained to predict the item at the next step. This approach only considers the sequential characteristics in an item sequence from left to right. While it is noted that the entire interaction sequence is indeed observed by the model in the training process. Inspired by the masked language model like BERT [1], we propose to model the bidirectional information in item sequence by a Cloze task. For our task, the Cloze setting is described as below: at each training step, we randomly mask a proportion of items in the input sequence (i.e., replace them with special tokens "[mask]"). Then we predict the masked items from the original sequence based on the surrounding context in both directions.
因此，我们考虑的第二个损失是从输入序列的双向上下文中恢复实际item。为此，我们准备了第4.2节中基础模型的预训练版本，这是一个双向Transformer架构。作为说明，假设我们掩码序列{ $i_1$ ,..., $i_t$ ,..., $i_n$ }中的第t个item $i_t$ 。我们将剩余序列{ $i_1$ ,...,mask,..., $i_n$ }视为 $i_t$ 的周围上下文，记为 $C_{it}$ 。给定周围上下文 $C_{it}$ 和被掩码的item $i_t$ ，我们将它们视为两个不同的视图以融合学习数据表示。根据公式3，我们通过以下方式最小化掩码item预测损失：
Therefore, the second loss we consider is to recover the actual item with the bidirectional context from the input sequences. For this purpose, we prepare a pre-trained version of the base model in Section 4.2, which is a bidirectional Transformer architecture. As illustration, let us mask the t-th item $i_t$ in a sequence { $i_1$ ,···, $i_t$ ,···, $i_n$ }. We treat the rest sequence { $i_1$ ,···,mask,···, $i_n$ } as the surrounding context for $i_t$ , denoted by $C_{it}$ . Given the surrounding context $C_{it}$ and the masked item $i_t$ , we treat them as two different views to fuse for learning data representations. Following Eq. 3, we minimize the Masked Item Prediction (MIP) loss by:
L_MIP( $C_{it}$ , $i_t$ ) = f( $C_{it}$ , $i_t$ ) − log[
$$
\sum
$$
_{ĩ
$$
\in
$$
I\{ $i_t$ }} f( $C_{it}$ , $i_t$ )], (12)
其中ĩ表示不相关item，f(·,·)根据以下公式实现：
where ĩ denotes an irrelevant item, and f(·,·) is implemented according to the following formula:
f( $C_{it}$ , $i_t$ ) = $\sigma$ ( $F_t$ ᵀ · W_MIP · $e_{it}$ ), (13)
其中W_MIP
$$
\in
$$
ℝ^{d
$$
\times
$$
d}是一个待学习的参数矩阵， $F_t$ 是使用双向Transformer架构获得的第t个位置的学习表示，其获取方式与公式7相同。
where W_MIP
$$
\in
$$
ℝ^{d
$$
\times
$$
d} is a parameter matrix to learn and $F_t$ is the learned representation for the t-th position using the bidirectional Transformer architecture obtained in the same way as Eq. 7.
#### 4.3.3 建模序列-属性相关性
4.3.3 Modeling Sequence-Attribute Correlation
在建模了item-属性和序列-item相关性之后，我们进一步考虑直接将属性信息与序列上下文融合。具体来说，我们采用与第4.3.2节类似的方式，基于周围上下文恢复被掩码item的属性。给定一个被掩码的item $i_t$ ，我们将其周围上下文 $C_{it}$ 及其属性集 $A_{it}$ 视为互信息最大化的两个不同视图。由此，我们可以通过以下方式开发掩码属性预测损失：
Having modeled both item-attribute and sequence-item correlations, we further consider directly fusing attribute information with sequential contexts. Specifically, we adopt a similar way as in Section 4.3.2 to recover the attributes of a masked item based on surrounding contexts. Given a masked item $i_t$ , we treat its surrounding context $C_{it}$ and its attribute set $A_{it}$ as two different views for MIM. As such, we can develop the following Masked Attribute Prediction (MAP) loss by:
L_MAP( $C_{it}$ , $A_{it}$ ) = $E_{a
$$
$\in
$$
$ A_{it} $}[f($ C_{it} $, a) − log
$$
\sum
$$
_{ã
$$
\in
$$
A\$ A_i $} exp(f($ C_{it} $, ã))], (14)
其中f(·,·)根据以下公式实现：
where f(·,·) is implemented according to the following formula:
f($ C_{it} $, a) = $ \sigma $($ F_t $ᵀ · W_MAP · e_a), (15)
其中W_MAP
$$
\in
$$
ℝ^{d
$$
\times
$$
d}是一个待学习的参数矩阵。注意，现有方法[4, 8, 24]很少直接建模序列上下文和属性信息之间的相关性。而我们希望显式地建模这种相关性，以导出更有意义的监督信号，这有助于改进多粒度信息的数据表示。
where W_MAP
$$
\in
$$
ℝ^{d
$$
\times
$$
d} is a parameter matrix to learn. Note that existing methods [4, 8, 24] seldom directly model the correlation between the sequential context and attribute information. While, we would like to explicitly model the correlation to derive more meaningful supervision signals, which is useful to improve the data representations for multi-granularity information.
#### 4.3.4 建模序列-片段相关性
4.3.4 Modeling Sequence-Segment Correlation
如上所示，完形填空学习策略在我们的预训练方法中在融合序列上下文与目标信息方面起着关键作用。然而，item序列与词序列之间的一个主要区别在于，单个目标item可能与周围上下文高度不相关。例如，用户购买某些产品仅仅是因为它们在打折。基于这一考虑，我们将完形填空策略从单个item扩展到item子序列（即称为片段）。显然，一个item片段比单个item反映了更清晰、更稳定的用户偏好。因此，我们遵循第4.3.2节中的类似策略，从周围上下文中恢复一个item子序列。这预期能够增强自监督学习信号并提升预训练性能。
As shown above, the Cloze learning strategy plays a key role in our pre-trained approach in fusing sequential contexts with target information. However, a major difference between item sequence with word sequence is that a single target item may not be highly related to surrounding contexts. For example, a user has bought some products just because they were on sale. Based on this concern, we extend the Cloze strategy from a single item to item subsequence (i.e., called segment). Apparently, an item segment reflects more clear, stable user preference than a single item. Therefore, we follow a similar strategy in Section 4.3.2 to recover an item subsequence from surrounding contexts. It is expected to enhance the self-supervised learning signal and improve the pre-trained performance.
令$ i_{j1:j2} $表示从item$ i_{j1} $到$ i_{j2} $的子序列，$ C_{i_{j1:j2} $}表示整个序列中$ i_{j1:j2} $的上下文。类似于公式12，我们可以用互信息最大化的公式恢复缺失的item片段，这称为片段预测损失：
Let $ i_{j1:j2} $ denote the subsequence from item $ i_{j1} $ to $ i_{j2} $, and $ C_{i_{j1:j2} $} denote the context for $ i_{j1:j2} $ within the entire sequence. Similar to Eq. 12, we can recover the missing item segment with a MIM formulation, which is so called the Segment Prediction (SP) loss as:
L_SP($ C_{i_{j1:j2} $}, $ i_{j1:j2} $) = f($ C_{i_{j1:j2} $}, $ i_{j1:j2} $) − log
$$
\sum
$$
_{ĩ_{$ j_1 $:$ j_2 $}} exp(f($ C_{i_{j1:j2} $}, ĩ_{$ j_1 $:$ j_2 $})), (16)
其中ĩ_{$ j_1 $,$ j_2 $}是损坏的负子序列，f(·,·)根据以下公式实现：
where ĩ_{$ j_1 $,$ j_2 $} is the corrupted negative subsequence and f(·,·) is implemented according to the following formula:
f($ C_{i_{j1:j2} $}, $ i_{j1:j2} $) = $ \sigma $(sᵀ · W_SP · s̃), (17)
其中W_SP
$$
\in
$$
ℝ^{d
$$
\times
$$
d}是一个待学习的参数矩阵，s和s̃分别是上下文$ C_{i_{j1:j2} $}和子序列$ i_{j1:j2} $的学习表示。为了学习s和s̃，我们应用双向Transformer来获取序列中最后一个位置的状态表示。
where W_SP
$$
\in
$$
ℝ^{d
$$
\times
$$
d} is a parameter matrix to learn, and s and s̃ are the learned representations for the contexts $ C_{i_{j1:j2} $} and subsequence $ i_{j1:j2} $, respectively. In order to learn s and s̃, we apply the bidirectional Transformer to obtain the state representations of the last position in a sequence.
### 4.4 学习与讨论
4.4 Learning and Discussion
在这一部分，我们介绍S3-Rec用于序列推荐的学习和相关讨论。
In this part, we present the learning and related discussions of our S3-Rec for sequential recommendation.
#### 4.4.1 学习
4.4.1 Learning
S3-Rec的整个过程包括两个重要阶段，即预训练阶段和微调阶段。我们分别在这两个阶段采用双向和单向Transformer[26]架构。在预训练阶段，我们通过考虑四种不同类型的相关性（公式10、公式12、公式14和公式16）来优化自监督学习目标；在微调阶段，我们利用从预训练阶段学到的参数来初始化单向Transformer的参数，然后利用从左到右的监督信号来训练网络。我们采用成对排序损失来优化其参数：
The entire procedure of S3-Rec consists of two important stages, namely pre-training and fine-tuning stages. We adopt bidirectional and unidirectional Transformer [26] architectures for the two stages, respectively. At the pre-trained stage, we optimize the self-supervised learning objectives by considering four different kinds of correlations (Eq. 10, Eq. 12, Eq. 14 and Eq. 16); at the fine-tuning stage, we utilize the learned parameters from the pre-trained stage to initialize the parameters of the unidirectional Transformer, and then utilize the left-to-right supervised signals to train the network. We adopt the pairwise rank loss to optimize its parameters as:
L_main = −
$$
\sum
$$
_{u
$$
\in
$$
U}
$$
\sum
$$
_{t=1}^{n} (log $ \sigma $(P($ i_{t+1} $|$ i_{1:t} $) − P(i⁻_{t+1}|$ i_{1:t} $))), (18)
where we pair each ground-truth item $ i_{t+1} $ with a negative item i⁻_{t+1} that is randomly sampled.
其中我们将每个真实item$ i_{t+1}$与一个随机采样的负itemi⁻_{t+1}配对。
#### 4.4.2 讨论
4.4.2 Discussion
我们的工作提供了一种新颖的自监督方法，通过预训练模型从输入中捕获内在数据相关性作为额外信号。这种方法非常通用，许多现有方法都可以包含在这个框架中。我们进行简要讨论如下。
Our work provides a novel self-supervised approach to capturing the intrinsic data correlation from the input as an additional signal through the pre-trained models. This approach is quite general so that many existing methods can be included in this framework. We make a brief discussion below.
基于特征的方法，如因子分解机[20]和AutoInt[22]，主要通过上下文特征的交互来学习数据表示。最终的预测是根据用户和item特征之间的实际交互结果进行的。在S3-Rec中，公式10中的关联属性预测损失L_AAP和公式14中的掩码属性预测损失L_MAP在特征交互方面具有类似的效果。然而，我们并没有显式地建模属性之间的交互。相反，我们专注于捕获属性信息与item/序列上下文之间的关联。我们工作的一个主要区别在于利用特征交互作为额外的监督信号来增强数据表示，而不是进行预测。
Feature-based approaches such as Factorization Machine [20] and AutoInt [22] mainly learn data representations through the interaction of context features. The final prediction is made according to the actual interaction results between the user and item features. In S3-Rec, the associated attribute prediction loss L_AAP in Eq. 10 and the masked attribute prediction loss L_MAP in Eq. 14 have the similar effect in feature interaction. However, we do not explicitly model the interaction between attributes. Instead, we focus on capturing the association between attribute information and item/sequential contexts. A major difference in our work is to utilize feature interaction as additional supervision signals to enhance data representations instead of making predictions.
序列模型如GRU4Rec[21]和SASRec[8]主要关注以从左到右的顺序建模上下文item与目标item之间的序列依赖关系。S3-Rec额外引入了一个预训练阶段，利用四种不同类型的自监督学习信号来增强数据表示。特别地，公式12中的掩码item预测损失L_MIP在捕获序列依赖关系方面与[8, 21]具有类似的效果，不同之处在于它还可以利用双向序列信息。
Sequential models such as GRU4Rec [21] and SASRec [8] mainly focus on modeling the sequential dependencies between contextual items and the target item in a left-to-right order. S3-Rec additionally incorporates a pre-trained stage that leverages four different kinds of self-supervised learning signals for enhancing data representations. In particular, the masked item prediction loss L_MIP in Eq. 12 has a similar effect to capture sequential dependencies as in [8, 21] except that it can also utilize bidirectional sequential information.
属性感知的序列模型如TransFM[16]和FDSA[29]利用上下文特征来改进序列推荐模型，其中这些特征被视为辅助信息来增强item或序列的表示。在我们的S3-Rec中，L_AAP损失和L_MAP损失旨在将属性与item或序列上下文融合，这能够达到与之前方法[16, 29]相同的效果。此外，预训练的数据表示也可以应用于改进现有方法。
Attribute-aware sequential models such as TransFM [16] and FDSA [29] leverage the contextual features to improve the sequential recommender models, in which these features are treated as auxiliary information to enhance the representation of items or sequences. In our S3-Rec, the L_AAP loss and L_MAP loss aim to fuse attribute with items or sequential contexts, which is able to achieve the same effect as previous methods [16, 29]. Besides, the pre-trained data representations can be also applied to improve existing methods.
---
## 5 实验
5 EXPERIMENT
### 5.1 实验设置
5.1 Experimental Setup
#### 5.1.1 数据集
5.1.1 Dataset
我们在从四个真实世界平台收集的六个数据集上进行实验，这些数据集具有不同的领域和稀疏程度。预处理后这些数据集的统计信息总结在表1中。
We conduct experiments on six datasets collected from four real-world platforms with varying domains and sparsity levels. The statistics of these datasets after preprocessing are summarized in Table 1.
Table 1: Statistics of the datasets after preprocessing.
表1：预处理后数据集的统计信息。
| Dataset | Meituan | Beauty | Sports | Toys | Yelp | LastFM |
|---------|---------|--------|--------|------|------|--------|
| # Users | 13,622 | 22,363 | 25,598 | 19,412 | 30,431 | 1,090 |
| # Items | 20,062 | 12,101 | 18,357 | 11,924 | 20,033 | 3,646 |
| # Avg. Actions / User | 54.9 | 8.9 | 16.4 | 8.6 | 10.4 | 48.2 |
| # Avg. Actions / Item | 37.3 | 16.4 | 10.4 | 14.1 | 15.8 | 14.4 |
| # Actions | 747,827 | 198,502 | 296,337 | 167,597 | 316,354 | 52,551 |
| Sparsity | 99.73% | 99.93% | 99.95% | 99.93% | 99.95% | 98.68% |
| # Attributes | 388 | 1,221 | 2,277 | 1,027 | 1,001 | 331 |
| # Avg. Attribute / Item | 31.5 | 5.1 | 6.0 | 4.3 | 4.8 | 8.8 |
(1) Meituan¹: this dataset consists of six-year (from Jan. 2014 to Jan. 2020) transaction records in Beijing on the Meituan platform. We select categories, locations, and the keywords extracted from customer reviews as attributes.
(1) 美团¹：该数据集包含美团平台上北京地区六年（从2014年1月到2020年1月）的交易记录。我们选择类别、位置以及从客户评论中提取的关键词作为属性。
(2) Amazon Beauty, Sports, and Toys: these three datasets are obtained from Amazon review datasets in [14]. In this work, we select three subcategories: "Beauty", "Sports and Outdoors", and "Toys and Games", and utilize the fine-grained categories and the brands of the goods as attributes.
(2) Amazon Beauty, Sports, and Toys：这三个数据集来自[14]中的Amazon评论数据集。在本工作中，我们选择了三个子类别："Beauty"、"Sports and Outdoors"和"Toys and Games"，并利用商品的细粒度类别和品牌作为属性。
(3) Yelp²: this is a popular dataset for business recommendation. As it is very large, we only use the transaction records after January 1st, 2019. We treat the categories of businesses as attributes.
(3) Yelp²：这是一个流行的商业推荐数据集。由于它非常大，我们只使用2019年1月1日之后的交易记录。我们将商业的类别作为属性。
(4) LastFM³: this is a music artist recommendation dataset and contains user tagging behaviors for artists. In this dataset, the tags of the artists given by the users are used as attributes.
(4) LastFM³：这是一个音乐艺术家推荐数据集，包含用户对艺术家的标签行为。在该数据集中，用户给出的艺术家标签被用作属性。
For all datasets, we group the interaction records by users and sort them by the interaction timestamps ascendingly. Following [21, 29], we only keep the 5-core datasets, and filter unpopular items and inactive users with fewer than five interaction records.
对于所有数据集，我们按用户分组交互记录，并按交互时间戳升序排序。遵循[21, 29]，我们只保留5-core数据集，并过滤掉交互记录少于五条的非热门item和非活跃用户。
¹https://www.meituan.com
²https://www.yelp.com/dataset
³https://grouplens.org/datasets/hetrec-2011/
#### 5.1.2 评估指标
5.1.2 Evaluation Metrics
我们采用top-k命中率、top-k归一化折损累计增益和平均倒数排名来评估性能，这些指标在相关工作中被广泛使用[21, 29]。由于HR@1等于NDCG@1，我们报告HR@{1,5,10}、NDCG@{5,10}和MRR的结果。遵循之前的工作[8, 19, 23]，我们应用留一法进行评估。具体来说，对于每个用户交互序列，最后一个item用作测试数据，倒数第二个item用作验证数据，剩余数据用于训练。由于item集很大，使用所有item作为测试候选非常耗时。遵循常见策略[7, 8]，我们将真实item与99个随机采样的用户未交互过的负item配对。我们根据item的排名计算所有指标，并报告所有测试用户的平均得分。
We employ top-k Hit Ratio (HR@k), top-k Normalized Discounted Cumulative Gain (NDCG@k), and Mean Reciprocal Rank (MRR) to evaluate the performance, which are widely used in related works [21, 29]. Since HR@1 is equal to NDCG@1, we report results on HR@{1,5,10}, NDCG@{5,10}, and MRR. Following previous works [8, 19, 23], we apply the leave-one-out strategy for evaluation. Concretely, for each user interaction sequence, the last item is used as the test data, the item before the last one is used as the validation data, and the remaining data is used for training. Since the item set is large, it is time-consuming to use all items as candidates for testing. Following the common strategy [7, 8], we pair the ground-truth item with 99 randomly sampled negative items that the user has not interacted with. We calculate all metrics according to the ranking of the items and report the average score over all test users.
#### 5.1.3 基线模型
5.1.3 Baseline Models
我们将我们提出的方法与以下十一种基线方法进行比较：
We compare our proposed approach with the following eleven baseline methods:
(1) PopRec is a non-personalized method that ranks items according to popularity measured by the number of interactions.
(1) PopRec是一种非个性化方法，根据交互次数衡量的流行度对item进行排名。
(2) FM [20] characterizes the pairwise interactions between variables using factorized model.
(2) FM [20]使用因子分解模型刻画变量之间的成对交互。
(3) AutoInt [22] utilizes the multi-head self-attentive neural network to learn the feature interaction.
(3) AutoInt [22]利用多头自注意力神经网络学习特征交互。
(4) GRU4Rec [3] applies GRU to model user click sequence for session-based recommendation. We represent the items using embedding vectors rather than one-hot vectors.
(4) GRU4Rec [3]应用GRU对用户点击序列进行建模，用于基于会话的推荐。我们使用嵌入向量而非独热向量来表示item。
(5) Caser [24] is a CNN-based method capturing high-order Markov Chains by applying horizontal and vertical convolutional operations for sequential recommendation.
(5) Caser [24]是一种基于CNN的方法，通过应用水平和垂直卷积操作为序列推荐捕获高阶马尔可夫链。
(6) SASRec [8] is a self-attention based sequential recommendation model, which uses the multi-head attention mechanism to recommend the next item.
(6) SASRec [8]是一种基于自注意力的序列推荐模型，它使用多头注意力机制来推荐下一个item。
(7) BERT4Rec [23] uses a Cloze objective loss for sequential recommendation by the bidirectional self-attention mechanism.
(7) BERT4Rec [23]通过双向自注意力机制使用完形填空目标损失进行序列推荐。
(8) HGN [13] is recently proposed and adopts hierarchical gating networks to capture long-term and short-term user interests.
(8) HGN [13]是最近提出的，采用层次门控网络来捕获长期和短期用户兴趣。
(9) GRU4RecF [4] is an improved version of GRU4Rec, which leverages attributes to improve the performance.
(9) GRU4RecF [4]是GRU4Rec的改进版本，利用属性来提升性能。
(10) SASRecF is our extension of SASRec, which concatenates the representations of item and attribute as the input to the model.
(10) SASRecF是我们对SASRec的扩展，它将item和属性的表示拼接作为模型的输入。
(11) FDSA [29] constructs a feature sequence and uses a feature-level self-attention block to model the feature transition patterns. This is the state-of-the-art model in sequential recommendation.
(11) FDSA [29]构建特征序列并使用特征级自注意力块来建模特征转移模式。这是序列推荐中的最先进模型。
#### 5.1.4 实现细节
5.1.4 Implementation Details
对于Caser和HGN，我们使用作者提供的源代码。对于其他方法，我们使用PyTorch实现它们。所有超参数都按照原始论文的建议进行设置。
For Caser and HGN, we use the source code provided by their authors. For other methods, we implement them by PyTorch. All hyper-parameters are set following the suggestions from the original papers.
对于我们提出的S3-Rec，我们将自注意力块和注意力头的数量设置为2。嵌入维度为64，最大序列长度为50（遵循[8]）。注意，我们的训练阶段包含两个阶段（即预训练和微调阶段），预训练阶段学到的参数用于初始化微调阶段中模型的嵌入层和自注意力层。
For our proposed S3-Rec, we set the number of the self-attention blocks and the attention heads as 2. The dimension of the embedding is 64, and the maximum sequence length is 50 (following [8]). Note that our training phase contains two stages (i.e., pre-training and fine-tuning stage), the learned parameters in the pre-training stage are used to initialize the embedding layers and self-attention layers of our model in the fine-tuning stage.
在预训练阶段，item的掩码比例设置为0.2，四个损失（即AAP、MIP、MAP和SP）的权重分别设置为0.2、1.0、1.0和0.5，基于我们的经验实验。我们使用Adam优化器[9]，学习率为0.001，预训练阶段的批量大小为200，微调阶段的批量大小为256。我们预训练模型100个epoch，然后在推荐任务上进行微调。代码和数据集可在链接获取：https://github.com/RUCAIBox/CIKM2020-S3Rec⁴。
In the pre-training stage, the mask proportion of item is set as 0.2 and the weights for the four losses (i.e., AAP, MIP, MAP, and SP) are set as 0.2, 1.0, 1.0, and 0.5, respectively, based on our empirical experiments. We use the Adam optimizer [9] with a learning rate of 0.001, where the batch size is set as 200 and 256 in the pre-training and the fine-tuning stage, respectively. We pre-train our model for 100 epochs and fine-tune it on the recommendation task. The code and data set are available at the link: https://github.com/RUCAIBox/CIKM2020-S3Rec⁴.
⁴To further verify the effectiveness of our method, we have performed the experiments that rank the ground-truth item with all the items as candidates. The complete results are shown on our project website at this link.
⁴为了进一步验证我们方法的有效性，我们进行了将真实item与所有item作为候选进行排名的实验。完整结果显示在我们的项目网站上。
### 5.2 实验结果
5.2 Experimental Results
不同方法在所有数据集上的结果如表2所示。基于这些结果，我们可以发现：
The results of different methods on all datasets are shown in Table 2. Based on the results, we can find:
对于三个非序列推荐基线，所有数据集上的性能排序一致，即PopRec > AutoInt > FM。由于产品采用中的"富者愈富"效应，PopRec是一个稳健的基线。AutoInt在大多数数据集上表现优于FM，因为多头自注意力机制具有更强的属性建模能力。然而，AutoInt在美团数据集上的性能比FM差。一个潜在的原因是多头自注意力可能从属性中引入更多噪声，因为这些属性是从美团平台上的评论中提取的关键词。总的来说，非序列推荐方法的性能比序列推荐方法差，因为序列模式在我们的任务中很重要。
For three non-sequential recommendation baselines, the performance order is consistent across all datasets, i.e., PopRec > AutoInt > FM. Due to the "rich-gets-richer" effect in product adoption, PopRec is a robust baseline. AutoInt performs better than FM on most datasets because the multi-head self-attention mechanism has a stronger capacity to model attributes. However, the performance of AutoInt is worse than that of FM on Meituan dataset. A potential reason is that the multi-head self-attention may incorporate more noise from the attributes since they are keywords extracted from the reviews on Meituan platform. In general, non-sequential recommendation methods perform worse than sequential recommendation methods, since the sequential pattern is important to consider in our task.
对于序列推荐基线方法，SASRec和BERT4Rec分别利用单向和双向自注意力机制，并且比GRU4Rec和Caser取得了更好的性能。这表明自注意力架构特别适合建模序列数据。然而，当使用传统的下一项预测损失进行训练时，它们的改进并不稳定。此外，HGN取得了与SASRec和BERT4Rec相当的性能。这表明层次门控网络可以很好地建模密切相关的item之间的关系。然而，当将属性信息直接注入GRU4Rec和SASRec（即GRU4RecF和SASRecF）时，性能改进并不一致。这种方法在Beauty、Sports、Toys和Yelp数据集上带来了改进，但在其他数据集上产生了负面影响。一个可能的原因是简单地拼接item表示及其属性表示不能有效地融合这两种信息。在大多数情况下，FDSA在所有基线中取得了最佳性能。这表明特征级自注意力块可以捕获有用的序列特征交互模式。
As for sequential recommendation baseline methods, SASRec and BERT4Rec utilize the unidirectional and bidirectional self-attention mechanism respectively, and achieve better performance than GRU4Rec and Caser. It indicates that self-attentive architecture is particularly suitable for modeling sequential data. However, their improvements are not stable when training with the conventional next-item prediction loss. Besides, HGN achieves comparable performance with SASRec and BERT4Rec. This indicates the hierarchical gating network can well model the relations between closely relevant items. However, when directly injecting the attribute information into GRU4Rec and SASRec (i.e., GRU4RecF and SASRecF), the performance improvement is not consistent. This method yields improvement on Beauty, Sports, Toys, and Yelp datasets, but has a negative influence on other datasets. One possible reason is that simply concatenating item representations and its attributes representations cannot effectively fuse the two kinds of information. In most cases, FDSA achieves the best performance among all baselines. This suggests that the feature-level self-attention blocks can capture useful sequential feature interaction patterns.
最后，通过将我们的方法与所有基线进行比较，可以清楚地看到S3-Rec在六个数据集上持续优于它们，且优势明显。与这些基线不同，我们采用自监督学习来增强属性、item和序列的表示以用于推荐任务，其中通过互信息最大化引入了四个预训练目标来建模多种数据相关性。这一结果也表明自监督方法对于提升自注意力架构在序列推荐中的性能是有效的。
Finally, by comparing our approach with all the baselines, it is clear to see that S3-Rec performs consistently better than them by a large margin on six datasets. Different from these baselines, we adopt the self-supervised learning to enhance the representations of the attribute, item, and sequence for the recommendation task, which incorporates four pre-training objectives to model multiple data correlations by MIM. This result also shows that the self-supervised approach is effective to improve the performance of the self-attention architecture for sequential recommendation.
### 5.3 进一步分析
5.3 Further Analysis
接下来，我们继续研究S3-Rec是否在更详细的分析中表现良好。
Next, we continue to study whether S3-Rec works well in more detailed analysis.
#### 5.3.1 消融研究
5.3.1 Ablation Study
我们提出的自监督方法S3-Rec基于互信息最大化设计了四个预训练目标。为了验证每个目标的有效性，我们在Meituan、Beauty、Sports和Toys数据集上进行了消融研究，分析每个目标的贡献。采用NDCG@10进行评估。还提供了最佳基线FDSA的结果以供比较。
Our proposed self-supervised approach S3-Rec designs four pre-training objectives based on MIM. To verify the effectiveness of each objective, we conduct the ablation study on Meituan, Beauty, Sports, and Toys datasets to analyze the contribution of each objective. NDCG@10 is adopted for this evaluation. The results from the best baseline FDSA are also provided for comparison.
从图2的结果中，我们可以观察到移除任何一个自监督目标都会导致性能下降。这表明所有目标对于提升推荐性能都是有用的。此外，这些目标的重要性在不同数据集上有所不同。总体而言，AAP和MAP比其他目标更重要。移除它们中的每一个都会在所有数据集上造成更大的性能下降。一个可能的原因是这两个目标利用属性信息增强了item和序列的表示。
From the results in Fig. 2, we can observe that removing any self-supervised objective would lead to the performance decrease. It indicates all the objectives are useful to improve the recommendation performance. Besides, the importance of these objectives is varying on different datasets. Overall, the AAP (Associated Attribute Prediction) and the MAP (Masked Attribute Prediction) are more important than the other objectives. Removing each of them yields a larger drop of performance on all datasets. One possible reason is that these two objectives enhance the representations of item and sequence with the attributes information.
可以清楚地看到，所有模型变体都优于仅用下一项预测损失训练的最佳基线FDSA。
It is clearly seen that all model variants are better than the best baseline FDSA, which is trained only with next-item predication loss.
#### 5.3.2 将自监督学习应用于其他模型
5.3.2 Applying Self-Supervised Learning to Other Models
由于自监督学习本身是一种学习范式，它可以普遍应用于各种模型。因此，在这一部分中，我们进行实验来检验我们的方法能否为其他模型带来改进。我们在Beauty和Toys数据集上使用自监督方法预训练一些基线模型。对于GRU4Rec、GRU4RecF、SASRec和SASRecF，我们直接应用我们的预训练目标来改进它们。值得注意的是，GRU4Rec和SASRec是单向模型，因此我们在预训练阶段保持单向编码器层。对于AutoInt和Caser，由于它们的架构不支持某些预训练目标⁵，我们仅利用预训练参数来初始化嵌入层的参数。
Since self-supervised learning itself is a learning paradigm, it can generally apply to various models. Thus, in this part, we conduct an experiment to examine whether our method can bring improvements to other models. We use the self-supervised approach to pre-training some baseline models on Beauty and Toys datasets. For GRU4Rec, GRU4RecF, SASRec, and SASRecF, we directly apply our pre-training objectives to improve them. It is worth noting that GRU4Rec and SASRec are unidirectional models, so we maintain the unidirectional encoder layer in the pre-training stage. For AutoInt and Caser, since their architectures do not support some of the pre-training objectives⁵, we only utilize the pre-trained parameters to initialize the parameters of the embedding layers.
⁵Because their base models do not support the mask operations.
⁵因为它们的基础模型不支持掩码操作。
Beauty和Toys数据集上NDCG@10的结果如图3所示。首先，经过我们的方法预训练后，所有基线都取得了更好的性能。这表明自监督学习也可以应用于提升它们的性能。其次，S3-Rec在预训练后优于所有基线。这是因为我们的模型在预训练阶段采用了双向Transformer编码器，这更适合我们的方法。第三，我们可以看到基于GRU的模型比其他模型取得的改进更少。一个可能的原因是RNN架构限制了自监督学习的潜力。
The results of NDCG@10 on Beauty and Toys datasets are shown in Fig. 3. First, after pre-training by our approach, all the baselines achieve better performance. This shows that self-supervised learning can also be applied to improve their performance. Second, S3-Rec outperforms all the baselines after pre-training. This is because our model adopts the bidirectional Transformer encoder in the pre-training stage, which is more suitable for our approach. Third, we can see the GRU-based models achieve less improvement than the other models. One possible reason is that RNN-based architecture limits the potential of self-supervised learning.
#### 5.3.3 关于训练数据量的性能比较
5.3.3 Performance Comparison w.r.t. the Amount of Training Data
传统的推荐系统需要大量的训练数据，因此它们在实际应用中容易遭受冷启动问题。我们的方法可以缓解这个问题，因为提出的自监督学习方法可以更好地利用输入中的数据相关性。我们通过使用完整数据集的不同比例（即20%、40%、60%、80%和100%）来模拟数据稀疏场景。
Conventional recommendation systems require a considerable amount of training data, thus they are likely to suffer from the cold start problem in real-world applications. This problem can be alleviated by our method because the proposed self-supervised learning approach can better utilize the data correlation from input. We simulate the data sparsity scenarios by using different proportions of the full dataset, i.e., 20%, 40%, 60%, 80%, and 100%.
图4展示了Sports和Yelp数据集上的评估结果。如我们所见，当使用较少的训练数据时，性能大幅下降。然而，在所有情况下，S3-Rec始终优于基线，特别是在极端稀疏水平（20%）下。这一观察结果表明S3-Rec能够通过自监督方法更好地利用数据，这在一定程度上减轻了数据稀疏问题对序列推荐的影响。
Fig. 4 shows the evaluation results on Sports and Yelp datasets. As we can see, the performance substantially drops when less training data is used. While, S3-Rec is consistently better than baselines in all cases, especially in an extreme sparsity level (20%). This observation implies that S3-Rec is able to make better use of the data with the self-supervised method, which alleviates the influence of data sparsity problem for sequential recommendation to some extent.
#### 5.3.4 关于预训练epoch数量的性能比较
5.3.4 Performance Comparison w.r.t. the Number of Pre-training Epochs
我们的方法包括一个预训练阶段和一个微调阶段。在预训练阶段，我们的模型可以学习属性、item、子序列和序列的增强表示以用于推荐任务。预训练epoch的数量影响推荐任务的性能。为了研究这一点，我们使用不同数量的epoch预训练模型，然后在推荐任务上进行微调。
Our approach consists of a pre-training stage and a fine-tuning stage. In the pre-training stage, our model can learn the enhanced representations of the attribute, item, subsequence, and sequence for the recommendation task. The number of pre-training epochs affects the performance of the recommendation task. To investigate this, we pre-train our model with a varying number of epochs and fine-tune it on the recommendation task.
图5展示了Beauty和Toys数据集上的结果。水平虚线表示没有预训练的性能。我们可以看到，我们的模型在最初的20个预训练epoch中受益最大。之后，性能略有提升。基于这一观察，我们可以得出结论，我们的自监督学习方法通过少量epoch的预训练就能很好地捕获不同视图（即属性、item、子序列和序列）之间的相关性。因此，增强的数据表示可以提升序列推荐的性能。
Fig. 5 presents the results on Beauty and Toys datasets. The horizontal dash lines represent the performance without pre-training. We can see that our model benefits mostly from the first 20 pre-training epochs. And after that, the performance improves slightly. Based on this observation, we can conclude that the correlations among different views (i.e., the attribute, item, subsequence, and sequence) can be well-captured by our self-supervised learning approach through pre-training within a small number of epochs. So that the enhanced data representations can improve the performance of sequential recommendation.
#### 5.3.5 收敛速度比较
5.3.5 Convergence Speed Comparison
在获得了属性、item和序列的增强表示之后，我们在推荐任务上微调模型。为了检验最终推荐任务上的收敛速度，我们逐步增加微调阶段的epoch数量，并比较我们的模型和其他基线的性能。
After obtaining the enhanced representations of the attribute, item, and sequence, we fine-tune our model on the recommendation task. To examine the convergence speed on the final recommendation task, we gradually increase the number of epochs for the fine-tuning stage and compare the performance of our model and other baselines.
图6展示了Beauty和Toys数据集上的结果。可以观察到，我们的模型收敛迅速，并在大约40个epoch后达到了最佳性能。相比之下，对比模型需要更多的epoch才能达到稳定的性能。这一结果表明，我们的方法可以利用预训练参数帮助模型更快收敛并取得更好的性能。
Fig. 6 shows the results on Beauty and Toys datasets. It can be observed that our model converges quickly and achieves the best performance after about 40 epochs. In contrast to our model, the comparison models need more epochs to achieve stable performance. This result shows that our approach can utilize pre-trained parameters to help the model converge faster and achieve better performance.
---
## 6 结论
6 CONCLUSION
在本文中，我们基于互信息最大化原理提出了一种自监督序列推荐模型S3-Rec。在我们的方法中，我们采用自注意力推荐器架构作为基础模型，并设计了四个自监督学习目标来学习原始数据内的相关性。基于互信息最大化，四个目标可以学习属性、item、片段和序列之间的相关性，从而增强序列推荐的数据表示。实验结果表明，我们的方法优于多个具有竞争力的基线。
In this paper, we proposed a self-supervised sequential recommendation model S3-Rec based on the mutual information maximization (MIM) principle. In our approach, we adopted the self-attentive recommender architecture as the base model and devised four self-supervised learning objectives to learn the correlations within the raw data. Based on MIM, the four objectives can learn the correlations among attribute, item, segment, and sequence, which enhances the data representations for sequential recommendation. Experimental results have shown that our approach outperforms several competitive baselines.
未来，我们将研究如何设计其他形式的自监督优化目标。我们还将考虑将我们的方法应用于更复杂的推荐任务，例如对话推荐和多模态推荐。
In the future, we will investigate how to design other forms of self-supervised optimization objectives. We will also consider applying our approach to more complex recommendation tasks, such as conversational recommendation and multimedia recommendation.
---
## 致谢
ACKNOWLEDGEMENT
This work was partially supported by the National Natural Science Foundation of China under Grant No. 61872369 and 61832017, Beijing Academy of Artificial Intelligence (BAAI) under Grant No. BAAI2020ZJ0301, and Beijing Outstanding Young Scientist Program under Grant No. BJJWZYJH012019100020098, the Fundamental Research Funds for the Central Universities, the Research Funds of Renmin University of China under Grant No.18XNLG22 and 19XNQ047. Xin Zhao is the corresponding author.
本工作部分受国家自然科学基金项目（批准号：61872369和61832017）、北京人工智能研究院项目（批准号：BAAI2020ZJ0301）、北京市杰出青年科学基金项目（批准号：BJJWZYJH012019100020098）、中央高校基本科研业务费专项资金、中国人民大学研究基金项目（批准号：18XNLG22和19XNQ047）资助。赵鑫为通讯作者。
---
## 参考文献
REFERENCES
[1] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova. 2019. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. In NAACL-HLT 2019. 4171–4186.
[1] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova. 2019. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. In NAACL-HLT 2019. 4171–4186.
[2] M. Gutmann and A. Hyvärinen. 2012. Noise-Contrastive Estimation of Unnormalized Statistical Models, with Applications to Natural Image Statistics. J. Mach. Learn. Res. 13 (2012), 307–361.
[2] M. Gutmann and A. Hyvärinen. 2012. Noise-Contrastive Estimation of Unnormalized Statistical Models, with Applications to Natural Image Statistics. J. Mach. Learn. Res. 13 (2012), 307–361.
[3] B. Hidasi, A. Karatzoglou, L. Baltrunas, and D. Tikk. 2016. Session-based Recommendations with Recurrent Neural Networks. In ICLR 2016.
[3] B. Hidasi, A. Karatzoglou, L. Baltrunas, and D. Tikk. 2016. Session-based Recommendations with Recurrent Neural Networks. In ICLR 2016.
[4] B. Hidasi, M. Quadrana, A. Karatzoglou, and D. Tikk. 2016. Parallel Recurrent Neural Network Architectures for Feature-rich Session-based Recommendations. In RecSys 2016. 241–248.
[4] B. Hidasi, M. Quadrana, A. Karatzoglou, and D. Tikk. 2016. Parallel Recurrent Neural Network Architectures for Feature-rich Session-based Recommendations. In RecSys 2016. 241–248.
[5] R. D. Hjelm, A. Fedorov, S. Lavoie-Marchildon, K. Grewal, P. Bachman, A. Trischler, and Y. Bengio. 2019. Learning deep representations by mutual information estimation and maximization. In ICLR 2019.
[5] R. D. Hjelm, A. Fedorov, S. Lavoie-Marchildon, K. Grewal, P. Bachman, A. Trischler, and Y. Bengio. 2019. Learning deep representations by mutual information estimation and maximization. In ICLR 2019.
[6] J. Huang, Z. Ren, W. X. Zhao, G. He, J.-R. Wen, and D. Dong. 2019. Taxonomy-Aware Multi-Hop Reasoning Networks for Sequential Recommendation. In WSDM 2019. 573–581.
[6] J. Huang, Z. Ren, W. X. Zhao, G. He, J.-R. Wen, and D. Dong. 2019. Taxonomy-Aware Multi-Hop Reasoning Networks for Sequential Recommendation. In WSDM 2019. 573–581.
[7] J. Huang, W. X. Zhao, H. Dou, J.-R. Wen, and E. Y. Chang. 2018. Improving Sequential Recommendation with Knowledge-Enhanced Memory Networks. In SIGIR 2018. 505–514.
[7] J. Huang, W. X. Zhao, H. Dou, J.-R. Wen, and E. Y. Chang. 2018. Improving Sequential Recommendation with Knowledge-Enhanced Memory Networks. In SIGIR 2018. 505–514.
[8] W.-C. Kang and J. J. McAuley. 2018. Self-Attentive Sequential Recommendation. In ICDM 2018. 197–206.
[8] W.-C. Kang and J. J. McAuley. 2018. Self-Attentive Sequential Recommendation. In ICDM 2018. 197–206.
[9] D. P. Kingma and J. Ba. 2015. Adam: A Method for Stochastic Optimization. In ICLR 2015.
[9] D. P. Kingma and J. Ba. 2015. Adam: A Method for Stochastic Optimization. In ICLR 2015.
[10] L. Kong, C. de Masson d'Autume, L. Yu, W. Ling, Z. Dai, and D. Yogatama. 2020. A Mutual Information Maximization Perspective of Language Representation Learning. In ICLR 2020.
[10] L. Kong, C. de Masson d'Autume, L. Yu, W. Ling, Z. Dai, and D. Yogatama. 2020. A Mutual Information Maximization Perspective of Language Representation Learning. In ICLR 2020.
[11] R. Linsker. 1988. Self-Organization in a Perceptual Network. IEEE Computer 21, 3 (1988), 105–117.
[11] R. Linsker. 1988. Self-Organization in a Perceptual Network. IEEE Computer 21, 3 (1988), 105–117.
[12] L. Logeswaran and H. Lee. 2018. An efficient framework for learning sentence representations. In ICLR 2018.
[12] L. Logeswaran and H. Lee. 2018. An efficient framework for learning sentence representations. In ICLR 2018.
[13] C. Ma, P. Kang, and X. Liu. 2019. Hierarchical Gating Networks for Sequential Recommendation. In KDD 2019. 825–833.
[13] C. Ma, P. Kang, and X. Liu. 2019. Hierarchical Gating Networks for Sequential Recommendation. In KDD 2019. 825–833.
[14] J. J. McAuley, C. Targett, Q. Shi, and A. van den Hengel. 2015. Image-Based Recommendations on Styles and Substitutes. In SIGIR 2015. 43–52.
[14] J. J. McAuley, C. Targett, Q. Shi, and A. van den Hengel. 2015. Image-Based Recommendations on Styles and Substitutes. In SIGIR 2015. 43–52.
[15] T. Mikolov, I. Sutskever, K. Chen, G. S. Corrado, and J. Dean. 2013. Distributed Representations of Words and Phrases and their Compositionality. In NeurIPS 2013. 3111–3119.
[15] T. Mikolov, I. Sutskever, K. Chen, G. S. Corrado, and J. Dean. 2013. Distributed Representations of Words and Phrases and their Compositionality. In NeurIPS 2013. 3111–3119.
[16] R. Pasricha and J. J. McAuley. 2018. Translation-based factorization machines for sequential recommendation. In RecSys 2018. 63–71.
[16] R. Pasricha and J. J. McAuley. 2018. Translation-based factorization machines for sequential recommendation. In RecSys 2018. 63–71.
[17] M. Quadrana, A. Karatzoglou, B. Hidasi, and P. Cremonesi. 2017. Personalizing Session-based Recommendations with Hierarchical Recurrent Neural Networks. In RecSys 2017. 130–137.
[17] M. Quadrana, A. Karatzoglou, B. Hidasi, and P. Cremonesi. 2017. Personalizing Session-based Recommendations with Hierarchical Recurrent Neural Networks. In RecSys 2017. 130–137.
[18] Pengjie Ren, Zhumin Chen, Jing Li, Zhaochun Ren, Jun Ma, and Maarten de Rijke. 2019. RepeatNet: A Repeat Aware Neural Recommendation Machine for Session-Based Recommendation. In AAAI 2019. 4806–4813.
[18] Pengjie Ren, Zhumin Chen, Jing Li, Zhaochun Ren, Jun Ma, and Maarten de Rijke. 2019. RepeatNet: A Repeat Aware Neural Recommendation Machine for Session-Based Recommendation. In AAAI 2019. 4806–4813.
[19] R. Ren, Z. Liu, Y. Li, W. X. Zhao, H. Wang, B. Ding, and J.-R. Wen. 2020. Sequential Recommendation with Self-Attentive Multi-Adversarial Network. In SIGIR 2020. 89–98.
[19] R. Ren, Z. Liu, Y. Li, W. X. Zhao, H. Wang, B. Ding, and J.-R. Wen. 2020. Sequential Recommendation with Self-Attentive Multi-Adversarial Network. In SIGIR 2020. 89–98.
[20] S. Rendle. 2010. Factorization Machines. In ICDM 2010. 995–1000.
[20] S. Rendle. 2010. Factorization Machines. In ICDM 2010. 995–1000.
[21] S. Rendle, C. Freudenthaler, and L. Schmidt-Thieme. 2010. Factorizing personalized Markov chains for next-basket recommendation. In WWW 2010. 811–820.
[21] S. Rendle, C. Freudenthaler, and L. Schmidt-Thieme. 2010. Factorizing personalized Markov chains for next-basket recommendation. In WWW 2010. 811–820.
[22] W. Song, C. Shi, Z. Xiao, Z. Duan, Y. Xu, M. Zhang, and J. Tang. 2019. AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks. In CIKM 2019. 1161–1170.
[22] W. Song, C. Shi, Z. Xiao, Z. Duan, Y. Xu, M. Zhang, and J. Tang. 2019. AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks. In CIKM 2019. 1161–1170.
[23] F. Sun, J. Liu, J. Wu, C. Pei, X. Lin, W. Ou, and P. Jiang. 2019. BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer. In CIKM 2019. 1441–1450.
[23] F. Sun, J. Liu, J. Wu, C. Pei, X. Lin, W. Ou, and P. Jiang. 2019. BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer. In CIKM 2019. 1441–1450.
[24] J. Tang and K. Wang. 2018. Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding. In WSDM 2018. 565–573.
[24] J. Tang and K. Wang. 2018. Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding. In WSDM 2018. 565–573.
[25] A. van den Oord, Y. Li, and O. Vinyals. 2018. Representation Learning with Contrastive Predictive Coding. CoRR abs/1807.03748 (2018). arXiv:1807.03748
[25] A. van den Oord, Y. Li, and O. Vinyals. 2018. Representation Learning with Contrastive Predictive Coding. CoRR abs/1807.03748 (2018). arXiv:1807.03748
[26] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin. 2017. Attention is All you Need. In NeurIPS 2017. 5998–6008.
[26] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin. 2017. Attention is All you Need. In NeurIPS 2017. 5998–6008.
[27] Xin Xin, Alexandros Karatzoglou, Ioannis Arapakis, and Joemon M. Jose. 2020. Self-Supervised Reinforcement Learning for Recommender Systems. In SIGIR 2020. 931–940.
[27] Xin Xin, Alexandros Karatzoglou, Ioannis Arapakis, and Joemon M. Jose. 2020. Self-Supervised Reinforcement Learning for Recommender Systems. In SIGIR 2020. 931–940.
[28] Y.-T. Yeh and Y.-N. Chen. 2019. QAInfomax: Learning Robust Question Answering System by Mutual Information Maximization. In EMNLP-IJCNLP 2019. 3368–3373.
[28] Y.-T. Yeh and Y.-N. Chen. 2019. QAInfomax: Learning Robust Question Answering System by Mutual Information Maximization. In EMNLP-IJCNLP 2019. 3368–3373.
[29] T. Zhang, P. Zhao, Y. Liu, V. S. Sheng, J. Xu, D. Wang, G. Liu, and X. Zhou. 2019. Feature-level Deeper Self-Attention Network for Sequential Recommendation. In IJCAI 2019. 4320–4326.
[29] T. Zhang, P. Zhao, Y. Liu, V. S. Sheng, J. Xu, D. Wang, G. Liu, and X. Zhou. 2019. Feature-level Deeper Self-Attention Network for Sequential Recommendation. In IJCAI 2019. 4320–4326.
[30] Kun Zhou, Wayne Xin Zhao, Shuqing Bian, Yuanhang Zhou, Ji-Rong Wen, and Jingsong Yu. 2020. Improving Conversational Recommender Systems via Knowledge Graph based Semantic Fusion. In KDD 2020.
[30] Kun Zhou, Wayne Xin Zhao, Shuqing Bian, Yuanhang Zhou, Ji-Rong Wen, and Jingsong Yu. 2020. Improving Conversational Recommender Systems via Knowledge Graph based Semantic Fusion. In KDD 2020.