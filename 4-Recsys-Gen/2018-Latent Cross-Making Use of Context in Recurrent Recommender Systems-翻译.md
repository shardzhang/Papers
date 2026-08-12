# 潜在交叉：在循环推荐系统中利用上下文


本文介绍了 潜在交叉：在循环推荐系统中利用上下文。核心内容：


关键发现：

---


Alex Beutel, Paul Covington, Sagar Jain, Can Xu, Jia Li\*, Vince Gatto, Ed H. Chi
> Google Inc.
Mountain View, California
{alexbeutel, pcovington, sagarj, canxu, vgatto, edchi}@google.com, vena900620@gmail.com

## 摘要

推荐系统的成功往往取决于其理解和利用推荐请求上下文的能力。大量研究关注时间、地点、界面以及大量其他上下文特征如何影响推荐。然而，在使用深度神经网络进行推荐系统时，研究人员常常忽略这些上下文，或者将它们作为普通特征纳入模型中。

在本文中，我们研究如何在神经推荐系统中有效处理上下文数据。我们首先对前馈推荐中将上下文作为特征的传统方法进行实证分析，并证明该方法在捕获常见特征交叉方面效率低下。我们将这一见解应用于设计一个最先进的RNN推荐系统。我们首先描述在YouTube中使用基于RNN的推荐系统。接着，我们提出"潜在交叉"（Latent Cross），一种易于使用的技术，通过先对上下文特征进行嵌入，然后将上下文嵌入与模型的隐藏状态进行逐元素乘积，从而将上下文数据纳入RNN中。我们通过在多种实验设置中使用这种潜在交叉技术，证明了性能的提升。

**ACM引用格式：**
Alex Beutel, Paul Covington, Sagar Jain, Can Xu, Jia Li, Vince Gatto, Ed H. Chi. 2018. Latent Cross: Making Use of Context in Recurrent Recommender Systems. In WSDM 2018: The Eleventh ACM International Conference on Web Search and Data Mining, February 5–9, 2018, Marina Del Rey, CA, USA. ACM, New York, NY, USA, 9 pages. https://doi.org/10.1145/3159652.3159727

## 1 引言

推荐系统长期以来一直被用于预测用户喜欢的内容。随着Facebook、Netflix、YouTube和Twitch等在线服务的持续增长，拥有高质量的推荐系统来帮助用户筛选不断扩展且日益多样化的内容变得越来越重要。

推荐系统中的大量研究集中在有效的机器学习技术上——即如何最好地从用户行为（如点击、购买、观看和评分）中学习。在这方面，有大量的关于协同过滤和推荐算法的研究，包括Netflix Prize期间的矩阵分解[24, 28, 30, 35]、局部聚焦模型[5, 9, 31]以及最近的深度学习[11, 36]。

与此同时，并且越来越突出的是，人们理解了建模推荐上下文的重要性——不仅仅是寻找视频的用户，还包括一天中的时间、地点、用户的设备等。许多这类模型已在分解设置中被提出，例如使用张量分解处理位置的[17]、为不同类型用户行为展开张量的[46]、或关于时间影响的人工设计特征的[29]。

随着深度学习的日益突出，如何将这些上下文特征纳入神经推荐系统却较少被直接探索。先前关于深度神经网络（DNN）推荐系统的工作主要依赖于将上下文建模为模型中的直接特征或采用多任务目标[11]。一个显著的例外是使用循环神经网络（RNN）来建模时间模式[25, 39, 43]。在本文中，我们弥合了上下文协同过滤文献和神经推荐文献之间的差距。我们探索如何在深度神经推荐（特别是RNN模型）中利用上下文数据，并证明现有技术遗漏了这些特征中的大量信息。

我们探索在YouTube使用的基于RNN的推荐系统中利用上下文数据的能力。与大多数生产环境一样，我们有大量重要的上下文数据需要包含：请求和观看时间、设备类型以及网站或移动应用上的页面。在本文中，首先，我们提供了关于将上下文建模为直接特征（特别是使用前馈神经网络作为示例基线DNN方法）局限性的理论解释。然后，我们提出了一种易于使用的技术来纳入这些特征，即使在更复杂的RNN模型中也能够提高预测准确性。

我们的贡献如下：
- **一阶挑战**：我们展示了一阶神经网络在建模低秩关系方面的挑战。
- **生产模型**：我们描述了我们如何为YouTube构建了一个大规模RNN推荐系统。
- **潜在交叉**：我们提出了一种简单的技术，称为"潜在交叉"（Latent Cross），以更具表现力的方式将上下文特征纳入我们的模型。具体来说，潜在交叉在上下文嵌入和神经网络隐藏状态之间执行逐元素乘积。
- **实证结果**：我们提供了实证结果，验证了我们的方法提高了推荐准确性。

## 2 相关工作

我们首先综述各种相关研究。表1给出了一个概览。

**上下文推荐。** 大量研究集中于在推荐过程中使用上下文数据。特别是，某些类型的上下文数据已被深入探索，而其他类型则被抽象处理。例如，推荐中的时间动态已被广泛探索[6]。在Netflix Prize期间[4]，Koren[29]发现了Netflix数据集中重要的长期时间动态，并在他的协同过滤（CF）模型中加入了时间特征来解释这些影响。研究人员还探索了偏好在更短时间尺度（如会话[39]）中的演化。更一般的抽象已被用于建模推荐的偏好演化，如点过程[15]和循环神经网络[43]。类似地，使用概率模型[2, 8]、矩阵分解[32]和张量分解[17]对用户行为与地理数据的建模已被广泛探索。各种方法基于矩阵和张量分解用于跨域学习[45, 46]。像分解机[34]和其他上下文推荐器[22, 37, 48]的方法提供了这些协同过滤方法的泛化。

**神经推荐系统。** 随着神经网络在计算机视觉和自然语言处理（NLP）任务中日益流行，推荐系统研究人员开始将DNN应用于推荐。早期的迭代直接应用协同过滤直觉到神经网络，如通过自编码器[36]或联合深度和CF模型[20]。更复杂的网络被设计用于纳入更广泛的输入特征[11]。Cheng等人[7]通过在线性模型中处理上下文特征之间的交互来解决这个问题，该线性模型位于模型的DNN部分之外。

最近，使用循环神经网络进行推荐的研究有所增长[21, 25, 39, 43]。[25, 43]在他们的模型中纳入了时间信息作为特征和监督信号，[41]纳入了通用上下文特征。然而，在这两种情况下，这些特征都是与输入拼接在一起的，我们证明这样做收益有限。同步且独立的研究[49]通过乘法方式纳入时间信息改进了LSTM，但并未将此推广到其他上下文数据。

**二阶神经网络。** 本文的一个主要方向是神经推荐中乘法关系的重要性。这些二阶单元在神经网络中有一些体现。循环单元，如LSTM[23]和GRU[10]，是常见的二阶单元，其门控机制使用逐元素乘法。关于循环网络更完整的教程可在[18]中找到。

此外，网络顶部的用于分类的softmax层明确是DNN生成的嵌入和标签类嵌入之间的双线性层。这一技术在多篇论文中被扩展，以在DNN顶部包含用户-item双线性层[20, 41, 43, 47]。

与本文描述的技术类似，有一系列关于乘法模型的工作[27, 44]。这些乘法结构最常见于自然语言处理中，如[14, 27]。NLP方法被应用于评论文本个性化建模[40]（具有略微不同的数学结构）。最近，[25]使用乘法技术不是针对上下文数据，而是直接针对用户，类似于张量分解。

PNN[33]和NFM[19]将这个想法推向极致，对输入的所有特征对进行乘法运算，然后在通过前馈网络之前对结果进行拼接或平均。这些模型的直觉与我们的相似，但不同之处在于我们关注上下文数据与用户行为之间的关系，我们的潜在交叉机制可以并且被应用到模型的各个部分，并且我们证明了即使在RNN推荐系统中这些交互的重要性。

更复杂的模型结构，如注意力模型[3]、记忆网络[38]和元学习[42]，也依赖于二阶关系，并且越来越流行。例如，注意力模型使用注意力向量，通过乘法来调制隐藏状态。然而，这些方法在结构上显著更加复杂，并且通常被发现在训练上更加困难。相比之下，本文提出的潜在交叉技术我们发现在实践中易于训练且有效。

## 3 建模预备知识

我们考虑一个推荐系统，其中我们有一个事件e的数据库E，这些事件是k元组。我们使用eℓ来表示元组中的第ℓ个值，使用eℓ̄来表示元组中的其他k−1个值。

例如，Netflix Prize设置可以用元组e ≡ (i, j, R)来描述，其中用户i对电影j给出了评分R。我们也可能有时间和设备等上下文，使得e ≡ (i, j, t, d)，其中用户i在时间t在设备类型d上观看了视频j。注意，每个值可以是离散类别变量（如存在N个用户，i
$$
\in
$$
 I），也可以是连续的（如t是Unix时间戳）。连续变量在预处理步骤中被离散化并不罕见，例如将t转换为事件发生的日期。

有了这个数据，我们可以将推荐系统框架化为试图在给定其他值的情况下预测事件的一个值。例如，Netflix Prize表示对于元组e = (i, j, R)，使用(i, j)来预测R。从机器学习的角度来看，我们可以将元组e拆分为特征x和标签y，使得x = (i, j)且标签y = R。

我们还可以进一步将推荐问题重新框架化为预测用户在给定时间将看什么视频，通过定义x = (i, t)和y = j。再次注意，根据标签是分类随机值（如视频ID）还是实数值（如评分），机器学习问题分别是分类问题或回归问题。

在分解模型中，所有输入值都被视为离散的，并进行嵌入和乘法。当我们"嵌入"一个离散值时，我们学习一个稠密的潜在表示，例如用户i由稠密的潜在向量ui描述，itemj由稠密的潜在向量vj描述。在矩阵分解模型中，预测通常基于ui·vj。在张量分解模型中， $预测基于$ \Sigma $_r$ $u_i$ , $_{r}$ $v_j$ , $_{r}$ $w_t$ , $_{r}$ ，其中 $w_t$ 是时间或其他上下文特征的稠密向量嵌入。参见分解机[34]以获得这些类型模型的清晰抽象。为简化记号，我们将使用⟨·⟩表示多维内积，即⟨ $u_i$ , $v_j$ , $w_t$ ⟩ = $\Sigma
$$
_{r}$ $u_i$,$_{r}$ $v_j$,$_{r}$ $w_t$,$_{r}$。

神经网络通常也对离散输入进行嵌入。也就是说，给定输入(i, j)，网络参数包括可训练（在神经网络中通过反向传播）的嵌入向量$u_i$和$v_j$。因此，我们考虑形如eℓ = f(eℓ̄)的神经网络，其中网络将元组中除一个值外的所有值作为输入，我们训练f来预测元组的最后一个值。我们稍后将扩展这个定义，允许模型也将相关的先前事件作为输入，就像在序列模型中一样。

**符号表**

| 符号 | 描述 |
|------|------|
| e | 描述观察事件的k元组 |
| eℓ | 元组中的元素ℓ |
| E | 所有观察到事件的集合 |
| $u_i$, $v_j$ | 用户i和itemj的可训练嵌入 |
| Xⁱ | 用户i的所有事件 |
| Xⁱ,$_t$ | 用户在时间t之前的所有事件 |
| e^($\tau$) | 特定序列中步骤$\tau$的事件 |
| ⟨·⟩ | k维内积 |
| ∗ | 逐元素乘积 |
| f(·) | 任意神经网络 |

## 4 动机：一阶DNN中的挑战

为了理解神经推荐器如何利用拼接特征，我们首先检查这些网络的典型构建块。如上所述，神经网络，特别是前馈DNN，通常建立在一阶操作之上。更精确地说，神经网络通常依赖于形如Wh的矩阵-向量乘法，其中W是学习到的权重矩阵，h是输入（可以是网络的输入，也可以是前一层的输出）。在前馈网络中，全连接层通常具有以下形式：

h_$\tau$ = g(W_$\tau$ $h_{$\tau$-1}$ + b_$\tau$)  (1)

其中g是逐元素操作如sigmoid或ReLU，$h_{$\tau$-1}$是上一层的输出，W_$\tau$和b_$\tau$是学习到的参数。我们将其视为一阶单元，因为$h_{$\tau$-1}$（一个k维向量）中不同的值只是根据W的权重加在一起，而从不相互相乘。

尽管具有这样层的神经网络已被证明能够逼近任何函数，但它们的核心计算在结构上与过去协同过滤的直觉显著不同。如上所述，矩阵分解模型采用一般形式ui·vj，使得模型学习不同类型输入（即用户、item、时间等）之间的低秩关系。鉴于低秩模型在推荐系统中取得了成功，我们提出以下问题：具有一阶单元的神经网络能在多大程度上建模低秩关系？

### 4.1 建模低秩关系

为了测试一阶神经网络是否能建模低秩关系，我们生成合成低秩数据，并研究不同大小的网络能多好地拟合该数据。更精确地说，我们考虑一个m阶张量，其中每个维度的大小为N。

我们生成随机向量u⁽ⁱ⁾和v⁽ʲ⁾，使得数据点x = [$u_i$; $v_j$]并且标签y = ⟨$u_i$, $v_j$⟩，其中嵌入：

$u_i$ ∼ N(0, (1/$\sqrt{}$m) I)  (2)

结果是我们的数据是一个秩为r的矩阵或张量，具有大致相同的尺度（均值为0，经验方差接近1）。作为示例，在m=3的情况下，我们可以使用这些嵌入来表示形式为(i, j, t, ⟨$u_i$, $u_j$, $u_t$⟩)的事件。

我们使用这个数据尝试拟合不同大小的模型。特别地，我们考虑一个模型，其中离散特征被嵌入并拼接作为输入。该模型具有一个带有ReLU激活函数的隐藏层（这在神经推荐系统中很常见），后跟一个最终的线性层。该模型在TensorFlow[1]中编程，使用均方误差损失（MSE）训练，优化器为Adagrad[16]，训练至收敛。我们通过训练数据和模型预测之间的皮尔逊相关系数（R）来衡量和报告模型的准确性。我们使用皮尔逊相关系数，使其对数据的微小方差变化不变。我们报告对训练数据的准确性，因为我们在测试这些模型结构能多好地拟合低秩模式（即，甚至不是它们是否能从中泛化）。

为了建模低秩关系，我们想看看模型能多好地逼近各个乘法，表示变量之间的交互。所有数据以N=100生成。当m=2时，我们检查隐藏层必须多大才能乘以两个标量；当m=3时，我们检查隐藏层必须多大才能乘以三个标量。我们使用r
$$
\in
$$
{1,2}来观察模型大小如何随着需要更多乘法而增长。我们将每个离散特征嵌入为20维向量，远大于r（但我们发现模型的准确性与该大小无关）。我们测试的隐藏层大小
$$
\in
$$
{1,2,5,10,20,30,50}。

**经验发现。** 从表3和图2可以看出，我们发现模型随着隐藏层大小的增长而持续更好地逼近数据。基于网络在逼近乘法的直觉，更宽的网络应能给出更好的逼近。其次，我们观察到，随着数据秩r从1增加到2，隐藏层大小大约翻倍才能获得相同的准确性。这也符合我们的直觉，因为增加r意味着有更多的交互需要相加——这是网络容易精确做到的。

更有趣的是，我们发现即使对于r=1和m=2，也需要大小为5的隐藏层才能获得"高"准确性的估计。考虑到协同过滤模型通常会发掘秩为200的关系[28]，这直观地表明现实世界的模型将需要非常宽的层才能学习单个双向关系。

此外，我们发现建模超过2路的关系增加了逼近该关系的难度。也就是说，当我们从m=2到m=3时，我们发现模型从需要宽度为5的隐藏层变为需要宽度为20的隐藏层才能获得约0.005的MSE或0.99的皮尔逊相关系数。

图2：ReLU层可以学习逼近低秩关系，但这样做效率低下。

总之，我们观察到ReLU层可以逼近乘法交互（交叉），但这样做效率相当低下。这激发了需要能够更容易表达和处理乘法关系的模型。我们现在将注意力转向使用RNN作为基线；这是一个更强的基线，因为它比前馈DNN能更好地表达乘法关系。

**表3：拟合低秩数据时不同宽度模型的皮尔逊相关系数**

| 隐藏层 | r=1, m=2 | r=1, m=3 | r=2, m=3 |
|--------|----------|----------|----------|
| 1 | 0.42601 | 0.27952 | 0.287817 |
| 2 | 0.601657 | 0.57222 | 0.472421 |
| 5 | 0.997436 | 0.854734 | 0.717233 |
| 10 | 0.999805 | 0.973214 | 0.805508 |
| 20 | 0.999938 | 0.996618 | 0.980821 |
| 30 | 0.999983 | 0.99931 | 0.975782 |
| 50 | 0.999993 | 0.999738 | 0.997821 |
| 100 | 0.999997 | 0.999928 | 0.99943 |

## 5 YouTube的循环推荐器

以上述分析为动机，我们现在描述对YouTube的RNN推荐系统的改进。RNN作为基线模型值得注意，因为它们已经是二阶神经网络，比上述探索的一阶模型复杂得多，并且处于动态推荐系统的前沿。

我们首先概述我们为YouTube构建的RNN推荐器，然后在第6节描述我们如何改进它以更好地利用上下文数据。

### 5.1 正式描述

在我们的设置中，我们观察到形式为"用户i在时间t观看了视频j（由用户$\psi$(j)上传）"的事件。（我们稍后将引入额外的上下文特征。）为了建模用户偏好和行为的演化，我们使用循环神经网络（RNN）模型，其中模型的输入是用户Xⁱ的事件集合 = {e = (i, j, $\psi$(j), t)
$$
\in
$$
 E | $e_0$ = i}。我们将使用Xⁱ,$_t$来表示用户Xⁱ在时间t之前的所有观看记录：

Xⁱ,$_t$ = {e = (i, j, t)
$$
\in
$$
 E | $e_0$ = i ∧ $e_3$ < t} ⊂ Xⁱ  (3)

模型被训练来产生序列预测Pr(j|i, t, Xⁱ,$_t$)，即用户i在给定时间t基于t之前的所有观看记录将要观看的视频j。为简单起见，我们将使用e^($\tau$)表示序列中的第$\tau$个事件，x^($\tau$)表示e^($\tau$)的转换后输入，y^($\tau$)表示要预测的第$\tau$个事件的标签。在上述例子中，如果e^($\tau$) = (i, j, $\psi$(j), t)且e^($\tau$+1) = (i, j', $\psi$(j^{\prime}), t^{\prime})，那么输入x^($\tau$) = [$v_j$; u_$\psi$(j); $w_t$]，用于预测y^($\tau$+1) = j'，其中$v_j$是视频嵌入，u_$\psi$(j)是上传者嵌入，$w_t$是上下文嵌入。

当预测y^($\tau$)时，我们当然不能使用对应事件e^($\tau$)的标签作为输入，但我们可以使用e^($\tau$)中的上下文，我们将其记为c^($\tau$)，例如c^($\tau$) = [$w_t$]。

### 5.2 基线RNN模型的结构

我们的RNN模型图可以在图1中看到，并在下面描述。循环神经网络对一系列动作进行建模。对于每个事件e^($\tau$)，模型前进一步，处理x^($\tau$)并更新隐藏状态向量z^($\tau$−1)。更精确地说，每个事件首先由一个神经网络h_0^($\tau$) = $f_i$(x^($\tau$))处理。在我们的设置中，这可以是恒等函数或全连接ReLU层。

网络的循环部分是一个函数h_1^($\tau$), z^($\tau$) = $f_r$(h_0^($\tau$), z^($\tau$−1))。也就是说，我们使用一个循环单元，如LSTM[23]或GRU[10]，它将前一步的状态和转换后的输入$f_i$(x^($\tau$))作为输入。

为了预测y^($\tau$)，我们使用 $f_o$(h_1^($\tau$−1), c^($\tau$))，这是另一个可训练的神经网络，它产生y^($\tau$)可能值上的概率分布。在我们的设置中，该网络将RNN的输出和即将进行的预测的上下文作为输入，最后以一个覆盖所有视频的softmax层结束。该网络可以包括多个全连接层。

### 5.3 上下文特征

该模型成功的关键在于整合超出观看视频序列之外的上下文数据。我们在下面讨论如何利用这些特征。

**时间差。** 在我们的系统中，有效纳入时间对于RNN的准确性非常有价值。历史上，时间上下文已以多种方式被纳入协同过滤模型。这里我们使用一种我们称为时间差的方法：

∆t^($\tau$) = log(t^($\tau$+1) − t^($\tau$))  (4)

也就是说，当考虑事件e^($\tau$)时，我们考虑距离下一个事件或预测还有多长时间。这基本上等同于[25]和[49]中描述的时间表示。

**软件客户端。** YouTube视频可以在各种设备上观看：浏览器、iOS、Android、Roku、Chromecast等。将这些上下文视作同等会错过相关的相关性。例如，用户可能在手机上比在Roku设备上更不可能观看长片电影。类似地，像预告片这样的短视频可能相对更可能在手机上观看。建模软件客户端，特别是它如何与观看决策交互，是很重要的。

**页面。** 我们还在系统中记录了观看发起的来源。例如，我们区分从首页开始的观看（即首页观看）和用户已经在观看视频时推荐为后续观看的观看（即下一个观看）。这很重要，因为来自首页的观看可能对新内容更开放，而前一次观看后的观看可能是由于用户想深入探索某个话题。

**前融合和后融合。** 我们可以以两种方式将这些上下文特征（我们统称为c^($\tau$)）作为直接输入。从图1可以看出，我们可以在网络的底部包含上下文作为输入，也可以与RNN单元的输出拼接在一起。我们将上下文特征在RNN之前纳入称为前融合（pre-fusion），将在RNN单元之后纳入上下文特征称为后融合（post-fusion）[12]。虽然这可能是一个微妙的点，但这个决策会对RNN产生显著影响。特别地，通过前融合包含一个特征，该特征将通过如何修改RNN的状态来影响预测。然而，通过后融合包含一个特征，该特征可以更直接地对当前步骤的预测产生影响。

为了管理这一点，当预测y^($\tau$)时，我们通常使用c^($\tau$)作为后融合特征，使用c^($\tau$−1)作为前融合特征。这意味着c^($\tau$−1)将影响RNN状态，但c^($\tau$)将用于预测y^($\tau$)。随后，在预测y^($\tau$+1)的下一步中，c^($\tau$)现在将成为前融合特征，影响从那时起RNN的状态。

### 5.4 实现与训练

我们的模型在TensorFlow[1]中实现，并在许多分布式工作节点和参数服务器上进行训练。训练使用可用的反向传播小批量随机梯度下降算法之一，要么是Adagrad[16]或ADAM[26]。在训练中，我们使用时间段($t_0$−7天, $t_0$]内最近的100次观看作为监督，其中$t_0$是训练时间。这通常优先考虑最近的观看，因为当学到的模型被应用于线上流量时，行为与预测任务更相似。

由于可用视频数量庞大，我们限制要预测的可能视频集以及建模的这些视频的上传者数量。在下面的实验中，这些集合的大小范围从500,000到2,000,000。Softmax层，其覆盖200万个视频的输出，使用采样softmax进行训练。我们在所有标签上使用交叉熵损失中的这个采样softmax。

## 6 基于潜在交叉的上下文建模

从以上对基线模型的描述应该可以清楚地看出，上下文特征的使用通常是通过简单的全连接层作为拼接输入来完成的。然而，正如我们在第4节中所解释的，神经网络在建模拼接输入特征之间的交互方面效率低下。这里我们提出一个简单的替代方案。

### 6.1 单一特征

我们从只有一个上下文特征要纳入的情况开始。为清楚起见，我们将使用时间作为示例上下文特征。我们不将该特征作为与其他相关特征拼接的另一个输入，而是在网络中间执行逐元素乘积。也就是说，我们执行：

h_0^($\tau$) = (1 + $w_t$) ∗ h_0^($\tau$)  (5)

其中我们通过0均值高斯分布初始化$w_t$（注意，w=0对应于恒等映射）。这可以解释为上下文提供了对隐藏状态的掩码或注意力机制。然而，这也使得输入的前一个观看和时间之间能够存在低秩关系。注意，我们也可以在RNN之后应用这个操作：

h_1^($\tau$) = (1 + $w_t$) ∗ h_1^($\tau$)  (6)

[27]中提供的技术可以被视为一个特例，其中乘法关系被包含在网络的顶层，与softmax函数一起，以改进NLP任务。在这种情况下，该操作可以被视为一个张量分解，其中一个模态的嵌入由神经网络产生。

### 6.2 使用多个特征

在许多情况下，我们有不止一个上下文特征要纳入。当包含多个上下文特征，比如时间t和设备d时，我们执行：

h^($\tau$) = (1 + $w_t$ + w_d) ∗ h^($\tau$)  (7)

我们出于几个不同原因使用这种形式：（1）通过用0均值高斯初始化$w_t$和w_d，乘法项的均值为1，因此可以类似地作为对隐藏状态的掩码/注意力机制。（2）通过将这些项相加，我们可以捕获隐藏状态与每个上下文特征之间的2路关系。这遵循了分解机[34]设计中的观点。（3）使用简单的加法函数易于训练。像$w_t$ ∗ w_d ∗ h^($\tau$)这样的更复杂函数会随着每个额外的上下文特征显著增加非凸性。类似地，我们发现学习一个函数f([$w_t$; w_d])更难以训练且结果更差。将特征纳入模型的概览可以见图1。

**效率。** 我们注意到，使用潜在交叉的一个显著优势是其简单性和计算效率。使用N个上下文特征和d维嵌入，潜在交叉可以在O(Nd)时间内计算，并且不会增加后续层的宽度。

## 7 实验

我们进行两组实验。第一组在一个受限数据集上进行，其中时间是唯一的上下文特征，我们比较了多个模型族。在第二组实验中，我们使用生产模型，并基于我们如何纳入上下文特征来探索相对改进。

### 7.1 比较分析

#### 7.1.1 设置

我们首先解释我们的实验设置。

**数据集和指标。** 我们使用一个包含数亿用户观看序列的数据集。用户被分为训练集、验证集和测试集，验证集和测试集各有数千万用户。观看被限制在一组500,000个流行视频中，所有用户在其序列中至少有50次观看。序列由观看过的视频列表和每个观看的时间戳给出。

任务是预测用户序列中的最后5次观看。为了衡量这一点，我们在测试集上使用k=1和k=20的平均精度均值（MAP@k）。

**模型。** 对于这组实验，我们使用带有LSTM循环单元的RNN。我们在RNN单元前后没有ReLU单元。模型在训练期间使用整个序列作为监督。模型使用反向传播和ADAM[26]进行训练。

由于时间是该数据集中唯一的上下文特征，我们使用视频嵌入$v_j$作为输入，并与时间差值w_∆t执行潜在交叉，使得LSTM被输入为$v_j$ ∗ w_∆t。这是前融合交叉的一个例子。我们将其称为RNN潜在交叉（RNN Latent Cross）。

**基线。** 我们将上述RNN潜在交叉模型与经过高度调优的替代形式模型进行比较：
- **RRN**：使用[$v_j$; w_∆t]作为RNN的输入；类似于[43]和[41]。
- **RNN**：直接在$v_j$上使用RNN（不含时间）；类似于[21]。
- **词袋（BOW）**：基于用户历史中的视频集和用户人口统计数据构建词袋模型。
- **BOW+Time**：3层前馈模型，输入为观看词袋（最近三次观看的拼接）、∆t和请求发生在一周中的时间。模型使用softmax（基于与最后一次观看共同观看最多的50个视频）进行训练。
- **段落向量（PV）**：使用[13]为每个用户学习无监督嵌入（基于用户人口统计数据和先前的观看）。将学到的嵌入以及最后一次观看的嵌入作为输入，输入到使用采样softmax训练的1层前馈分类器中。
- **共同观看（Cowatch）**：基于序列中的最后一次观看，预测最常见的共同观看视频。

除非另有说明，所有模型都具有层次化softmax。所有模型及其超参数都在建模竞赛过程中进行了调优。注意，只有词袋和段落向量使用了用户人口统计数据。

#### 7.1.2 结果

我们在表4中报告这组实验的结果。从那里可以看出，我们的模型使用带有∆t的RNN（∆t与观看进行潜在交叉）在Precision@1和MAP@20上都给出了最佳结果。可能更有趣的是这些模型的相对性能。我们在词袋模型和RNN模型中观察到建模时间的关键重要性。此外，观察到执行潜在交叉而不是仅仅拼接∆t所带来的改进，大于将∆t作为输入特征所带来的改进。

**表4：比较研究结果：带有潜在交叉的RNN表现最佳**

| 方法 | Precision@1 | MAP@20 |
|------|-------------|--------|
| RNN with ∆t Latent Cross | 0.1621 | 0.0828 |
| RRN (Concatenated ∆t) | 0.1465 | 0.0753 |
| RNN (Plain, no time) | 0.1345 | 0.0724 |
| Bag of Words | 0.1250 | 0.0707 |
| Bag of Words with time | 0.1550 | 0.0794 |
| Paragraph Vectors | 0.1123 | 0.0642 |
| Cowatch | 0.1204 | 0.0621 |

### 7.2 YouTube模型

其次，我们研究生产模型的多个变体，使用更大、更不受限的数据集。

#### 7.2.1 设置

这里，我们使用生产环境中的用户观看数据集，它比上述设置限制更少。我们的序列由观看的视频和创建视频的人（上传者）组成。我们使用一个更大的词汇表，数量级为数百万最近流行的上传视频和上传者。

我们基于用户和时间共同将数据集划分为训练集和测试集。首先，我们将用户分为两组：90%的用户在训练集中，10%在测试集中。其次，为了按时间划分，我们选择一个时间截止点$t_0$，在训练期间只考虑$t_0$之前的观看。在测试期间，我们考虑$t_0$+4小时之后的观看。同样，视频词汇表基于$t_0$之前的数据。

我们的模型包括嵌入和拼接上述所有定义的输入特征，然后是一个256维的ReLU层、一个256维的GRU单元、另一个256维的ReLU层，然后输入到softmax层。如前所述，我们使用时间段($t_0$−7天, $t_0$]内最近的100次观看作为监督。这里，我们使用Adagrad优化器[16]在多个工作节点和参数服务器上进行训练。

为了测试我们的模型，我们再次衡量平均精度均值（MAP@k）。对于不在我们词汇表中的观看，我们始终将预测标记为不正确。这里报告的评估MAP@k分数使用约45,000次观看进行测量。

#### 7.2.2 页面作为上下文的价值

我们首先分析以不同方式纳入页面带来的准确性改进。特别是，我们比较不使用页面、使用页面作为与其他输入拼接的输入、以及使用页面进行后融合潜在交叉。（注意，当我们将页面作为拼接特征时，它在前融合和后融合过程中都被拼接。）

从图3可以看出，使用具有潜在交叉的页面提供了最佳的准确性。此外，我们看到同时使用潜在交叉和拼接输入并没有带来额外的准确性提升，这表明潜在交叉足以捕获通过将特征作为直接输入所能获得的全部相关信息。

#### 7.2.3 总改进

最后，我们测试在完整生产模型之上添加潜在交叉如何影响准确性。在这种情况下，每次观看模型知道页面、设备类型、时间、视频被观看的时长（观看时间）、观看的新旧程度（观看年龄）以及上传者。特别地，我们的基线YouTube模型使用页面、设备、观看时间和时间差值作为前融合拼接特征，并且也使用页面、设备和观看年龄作为后融合拼接特征。

我们测试将时间差和页面作为前融合潜在交叉，以及设备类型和页面作为后融合潜在交叉。从图4可以看出，尽管所有这些特征已经通过拼接被包含在内，但将它们作为潜在交叉包含进来仍能提供相对于基线模型的准确性改进。这也证明了前融合和后融合与多个特征的协同工作能力，能够提供强大的准确性提升。

## 8 讨论

我们在下面探讨这项工作提出的若干问题以及对未来工作的启示。

### 8.1 DNN中的离散关系

虽然本文大部分内容集中在启用特征之间的乘法交互上，但我们发现神经网络也可以逼近离散交互，这是分解模型更困难的领域。例如，在[46]中，作者发现建模"用户i对itemj执行动作a"时的⟨u(i,a), $v_j$⟩比⟨$u_i$, $v_j$, $w_a$⟩具有更好的准确性。然而，发现将用户和动作一起索引效果更好是困难的，需要数据洞察。

与第4节中的实验类似，我们生成遵循模式$X_i$,$_j$,$_{a}$ = ⟨u(i,a), $v_j$⟩的合成数据，并测试不同的网络架构在仅将i、j和a作为独立输入拼接的情况下，预测$X_i$,$_j$,$_{a}$的效果如何。我们将u
$$
\in
$$
 R¹⁰⁰⁰⁰和v
$$
\in
$  $
 R¹⁰⁰初始化为向量，使得X是一个秩为1的矩阵。我们遵循与第4节相同的一般实验流程，测量具有不同数量隐藏层和不同宽度隐藏层的网络的皮尔逊相关系数（R）。（我们以0.01的学习率训练这些网络，比上面使用的学习率小十倍。）作为基线，我们还测量了不同秩的张量分解（⟨$ u_i $, $ v_j $, w$ _{a}$⟩）的皮尔逊相关系数。

从图5可以看出，深度模型在某些情况下能达到相当高的皮尔逊相关系数，表明它们实际上能够逼近离散交叉。同样有趣的是，学习这些交叉需要具有宽隐藏层的深层网络，相对于数据大小而言特别大。此外，我们发现这些网络难以训练。

这些数字相对于基线张量分解性能来说很有趣。我们观察到分解模型能相当好地逼近数据，但需要相对较高的秩。（注意，即使底层张量是满秩的，秩为100的分解就足以描述它。）然而，即使在这个高秩下，张量分解模型需要的参数比DNN少，并且更容易训练。因此，正如我们在第5节中的结果，DNN可以逼近这些模式，但这样做可能很困难，并且纳入低秩交互有助于提供易于训练的逼近。

**图5：足够大的DNN可以学习逼近离散交互。**

### 8.2 二阶DNN

阅读本文时自然要问的问题是，为什么不尝试更宽的层、让模型更深、或使用更多的二阶单元，如GRU和LSTM？所有这些都是合理的建模决策，但根据我们的经验，它们使模型的训练显著更加困难。这种方法的一个优势在于它易于实现和训练，同时仍能提供清晰的性能改进，即使在与其他二阶单元（如LSTM和GRU）结合使用时也是如此。

整个深度学习中日益增长的趋势似乎是使用更多的二阶交互。例如，这在注意力模型和记忆网络中很常见，如上所列。虽然这些模型甚至更难以训练，但我们相信这项工作展示了在这个方向上用于神经推荐系统的前景。

## 9 结论

在本文中，我们探讨了如何在YouTube的生产环境循环推荐系统中纳入上下文数据。特别是，本文做出了以下贡献：
- **一阶DNN的挑战**：我们发现前馈神经网络在建模特征之间的乘法关系（交叉）方面效率低下。
- **生产模型**：我们提供了在YouTube使用的基于RNN的推荐系统的详细描述。
- **潜在交叉**：我们提出了一种在DNN（包括RNN）中学习乘法关系的简单技术。
- **实证结果**：我们在多个设置和使用不同上下文特征的情况下证明，潜在交叉提高了推荐准确性，即使在复杂、最先进的RNN推荐器之上也是如此。

## 参考文献

[1] Martín Abadi, Paul Barham, Jianmin Chen, Zhifeng Chen, Andy Davis, Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Geoffrey Irving, Michael Isard, and others. 2016. TensorFlow: A system for large-scale machine learning. In Proceedings of the 12th USENIX Symposium on Operating Systems Design and Implementation (OSDI). Savannah, Georgia, USA.

[2] Amr Ahmed, Liangjie Hong, and Alexander J Smola. 2013. Hierarchical geographical modeling of user locations from social media posts. In Proceedings of the 22nd international conference on World Wide Web (WWW). ACM, 25–36.

[3] Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. 2014. Neural machine translation by jointly learning to align and translate. arXiv preprint arXiv:1409.0473 (2014).

[4] James Bennett, Stan Lanning, and others. 2007. The netflix prize. In Proceedings of KDD cup and workshop, Vol. 2007. New York, NY, USA, 35.

[5] Alex Beutel, Ed H Chi, Zhiyuan Cheng, Hubert Pham, and John Anderson. 2017. Beyond Globally Optimal: Focused Learning for Improved Recommendations. In Proceedings of the 26th International Conference on World Wide Web (WWW). ACM.

[6] Pedro G Campos, Fernando Díez, and Iván Cantador. 2014. Time-aware recommender systems: a comprehensive survey and analysis of existing evaluation protocols. User Modeling and User-Adapted Interaction 24, 1-2 (2014), 67–119.

[7] Heng-Tze Cheng, Levent Koc, Jeremiah Harmsen, Tal Shaked, Tushar Chandra, Hrishi Aradhye, Glen Anderson, Greg Corrado, Wei Chai, Mustafa Ispir, and others. 2016. Wide & deep learning for recommender systems. In Proceedings of the 1st Workshop on Deep Learning for Recommender Systems. ACM, 7–10.

[8] Zhiyuan Cheng, James Caverlee, and Kyumin Lee. 2010. You are where you tweet: a content-based approach to geo-locating twitter users. In Proceedings of the 19th ACM international conference on Information and knowledge management. ACM, 759–768.

[9] Evangelia Christakopoulou and George Karypis. 2016. Local Item-Item Models For Top-N Recommendation. In Proceedings of the 10th ACM Conference on Recommender Systems (RecSys). ACM, 67–74.

[10] Junyoung Chung, Caglar Gülçehre, Kyunghyun Cho, and Yoshua Bengio. 2015. Gated Feedback Recurrent Neural Networks.. In ICML. 2067–2075.

[11] Paul Covington, Jay Adams, and Emre Sargin. 2016. Deep Neural Networks for YouTube Recommendations. In Proceedings of the 10th ACM Conference on Recommender Systems (RecSys). ACM, 191–198.

[12] Bin Cui, Anthony KH Tung, Ce Zhang, and Zhe Zhao. 2010. Multiple feature fusion for social media applications. In Proceedings of the 2010 ACM SIGMOD International Conference on Management of data. ACM, 435–446.

[13] Andrew M Dai, Christopher Olah, and Quoc V Le. 2015. Document embedding with paragraph vectors. arXiv preprint arXiv:1507.07998 (2015).

[14] Yann N Dauphin, Angela Fan, Michael Auli, and David Grangier. 2016. Language modeling with gated convolutional networks. arXiv preprint arXiv:1612.08083 (2016).

[15] Nan Du, Yichen Wang, Niao He, Jimeng Sun, and Le Song. 2015. Time-sensitive recommendation from recurrent user activities. In Advances in Neural Information Processing Systems. 3492–3500.

[16] John Duchi, Elad Hazan, and Yoram Singer. 2011. Adaptive subgradient methods for online learning and stochastic optimization. Journal of Machine Learning Research 12, Jul (2011), 2121–2159.

[17] Hancheng Ge, James Caverlee, and Haokai Lu. 2016. TAPER: A contextual tensor-based approach for personalized expert recommendation. (2016).

[18] Alex Graves. 2013. Generating sequences with recurrent neural networks. arXiv preprint arXiv:1308.0850 (2013).

[19] Xiangnan He and Tat-Seng Chua. 2017. Neural Factorization Machines for Sparse Predictive Analytics. In Proceedings of the 40th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR'17). ACM, New York, NY, USA, 355–364. DOI: https://doi.org/10.1145/3077136.3080777

[20] Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu, and Tat-Seng Chua. 2017. Neural collaborative filtering. In Proceedings of the 26th International Conference on World Wide Web. International World Wide Web Conferences Steering Committee, 173–182.

[21] Balázs Hidasi, Alexandros Karatzoglou, Linas Baltrunas, and Domonkos Tikk. 2015. Session-based recommendations with recurrent neural networks. arXiv preprint arXiv:1511.06939 (2015).

[22] Balázs Hidasi and Domonkos Tikk. 2016. General factorization framework for context-aware recommendations. Data Mining and Knowledge Discovery 30, 2 (2016), 342–371.

[23] Sepp Hochreiter and Jürgen Schmidhuber. 1997. Long short-term memory. Neural computation 9, 8 (1997), 1735–1780.

[24] Yifan Hu, Yehuda Koren, and Chris Volinsky. 2008. Collaborative filtering for implicit feedback datasets. In ICDM.

[25] How Jing and Alexander J. Smola. 2017. Neural Survival Recommender. In Proceedings of the Tenth ACM International Conference on Web Search and Data Mining (WSDM). 515–524.

[26] Diederik Kingma and Jimmy Ba. 2014. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980 (2014).

[27] Ryan Kiros, Richard Zemel, and Ruslan R Salakhutdinov. 2014. A multiplicative model for learning distributed text-based attribute representations. In Advances in neural information processing systems. 2348–2356.

[28] Yehuda Koren. 2008. Factorization meets the neighborhood: a multifaceted collaborative filtering model. In KDD. ACM, 426–434.

[29] Yehuda Koren. 2010. Collaborative filtering with temporal dynamics. Commun. ACM 53, 4 (2010), 89–97.

[30] Yehuda Koren, Robert Bell, and Chris Volinsky. 2009. Matrix Factorization Techniques for Recommender Systems. Computer 42, 8 (Aug. 2009), 30–37. DOI: https://doi.org/10.1109/MC.2009.263

[31] Joonseok Lee, Seungyeon Kim, Guy Lebanon, and Yoram Singer. 2013. Local Low-Rank Matrix Approximation. In Proceedings of the 30th International Conference on Machine Learning (ICML). 82–90.

[32] Haokai Lu and James Caverlee. 2015. Exploiting geo-spatial preference for personalized expert recommendation. In Proceedings of the 9th ACM Conference on Recommender Systems (RecSys). ACM, 67–74.

[33] Yanru Qu, Han Cai, Kan Ren, Weinan Zhang, Yong Yu, Ying Wen, and Jun Wang. 2016. Product-based neural networks for user response prediction. In Data Mining (ICDM), 2016 IEEE 16th International Conference on. IEEE, 1149–1154.

[34] Steffen Rendle. 2012. Factorization Machines with libFM. ACM TIST 3, 3, Article 57 (May 2012), 22 pages.

[35] Ruslan Salakhutdinov and Andriy Mnih. 2008. Bayesian probabilistic matrix factorization using Markov chain Monte Carlo. In ICML. ACM, 880–887.

[36] Suvash Sedhain, Aditya Krishna Menon, Scott Sanner, and Lexing Xie. 2015. Autorec: Autoencoders meet collaborative filtering. In Proceedings of the 24th International Conference on World Wide Web (WWW). ACM, 111–112.

[37] Yue Shi, Alexandros Karatzoglou, Linas Baltrunas, Martha Larson, Alan Hanjalic, and Nuria Oliver. 2012. TFMAP: optimizing MAP for top-n context-aware recommendation. In Proceedings of the 35th international ACM SIGIR conference on Research and development in information retrieval. ACM, 155–164.

[38] Sainbayar Sukhbaatar, Jason Weston, Rob Fergus, and others. 2015. End-to-end memory networks. In Advances in neural information processing systems. 2440–2448.

[39] Yong Kiam Tan, Xinxing Xu, and Yong Liu. 2016. Improved recurrent neural networks for session-based recommendations. In Proceedings of the 1st Workshop on Deep Learning for Recommender Systems. ACM, 17–22.

[40] Duyu Tang, Bing Qin, Ting Liu, and Yuekui Yang. 2015. User Modeling with Neural Network for Review Rating Prediction.. In IJCAI. 1340–1346.

[41] Bartlomiej Twardowski. 2016. Modelling Contextual Information in Session-Aware Recommender Systems with Neural Networks.. In RecSys. 273–276.

[42] Manasi Vartak, Hugo Larochelle, and Arvind Thiagarajan. 2017. A Meta-Learning Perspective on Cold-Start Recommendations for Items. In Advances in Neural Information Processing Systems. 6888–6898.

[43] Chao-Yuan Wu, Amr Ahmed, Alex Beutel, Alexander J. Smola, and How Jing. 2017. Recurrent Recommender Networks. In Proceedings of the Tenth ACM International Conference on Web Search and Data Mining (WSDM). 495–503.

[44] Yuhuai Wu, Saizheng Zhang, Ying Zhang, Yoshua Bengio, and Ruslan R Salakhutdinov. 2016. On multiplicative integration with recurrent neural networks. In Advances in Neural Information Processing Systems. 2856–2864.

[45] Chunfeng Yang, Huan Yan, Donghan Yu, Yong Li, and Dah Ming Chiu. 2017. Multi-site User Behavior Modeling and Its Application in Video Recommendation. In Proceedings of the 40th International ACM SIGIR Conference on Research and Development in Information Retrieval. ACM, 175–184.

[46] Zhe Zhao, Zhiyuan Cheng, Lichan Hong, and Ed H Chi. 2015. Improving User Topic Interest Profiles by Behavior Factorization. In Proceedings of the 24th International Conference on World Wide Web (WWW). 1406–1416.

[47] Lei Zheng, Vahid Noroozi, and Philip S Yu. 2017. Joint deep modeling of users and items using reviews for recommendation. In Proceedings of the Tenth ACM International Conference on Web Search and Data Mining (WSDM). ACM, 425–434.

[48] Yong Zheng, Bamshad Mobasher, and Robin Burke. 2014. CSLIM: Contextual SLIM recommendation algorithms. In Proceedings of the 8th ACM Conference on Recommender Systems. ACM, 301–304.

[49] Yu Zhu, Hao Li, Yikang Liao, Beidou Wang, Ziyu Guan, Haifeng Liu, and Deng Cai. 2017. What to Do Next: Modeling User Behaviors by Time-LSTM. In Proceedings of the Twenty-Sixth International Joint Conference on Artificial Intelligence, IJCAI-17. 3602–3608. DOI: https://doi.org/10.24963/ijcai.2017/504
