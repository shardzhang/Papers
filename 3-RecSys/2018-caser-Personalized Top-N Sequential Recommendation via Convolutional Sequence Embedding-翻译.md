# 通过卷积序列嵌入实现个性化Top-N序列推荐

> Jiaxi Tang, Ke Wang | 西蒙弗雷泽大学计算科学学院

本文提出了卷积序列嵌入推荐模型（Caser）用于Top-N序列推荐。核心内容：

- 将用户建模为过去交互item的序列，预测用户在"近未来"可能交互的Top-N排序item
- 将最近L个item的嵌入表示为时间和隐空间中的"图像"，使用卷积滤波器学习序列模式作为图像的局部特征
- 使用水平和垂直卷积滤波器捕获点级、联合级序列模式以及跳跃行为

关键发现：

- Caser在四个真实数据集上的一致优于最先进序列推荐方法
- 水平卷积滤波器捕获联合级序列模式，垂直卷积滤波器捕获点级序列模式，两者结合效果最佳
- Caser泛化了矩阵分解（MF）、分解个性化马尔可夫链（FPMC）和Fossil等多个现有模型

---

## 摘要

Top-N序列推荐将每个用户建模为过去交互的item序列，旨在预测用户在"近未来"可能交互的Top-N排序item。交互的顺序意味着序列模式起着重要作用，其中序列中较新的item对下一个item有更大的影响。在本文中，我们提出卷积序列嵌入推荐模型（Caser，Convolutional Sequence Embedding Recommendation Model）作为解决方案来满足这一要求。其思想是将最近item的序列嵌入到时间和隐空间中的"图像"中，并使用卷积滤波器学习序列模式作为该图像的局部特征。这种方法提供了一个统一且灵活的网络结构来捕获一般偏好和序列模式。在公共数据集上的实验表明，Caser在多种常见评估指标上一致优于最先进的序列推荐方法。

## CCS 概念

- 信息系统 $\rightarrow$ 检索模型与排序

## 关键词

Recommender System; Sequential Prediction; Convolutional Neural Networks

## ACM 引用格式

Jiaxi Tang and Ke Wang. 2018. Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding. In Proceedings of Eleventh ACM International Conference on Web Search and Data Mining, February 5–9, 2018, Marina Del Rey, CA, USA, (WSDM 2018), 9 pages. DOI: 10.1145/3159652.3159656

允许以个人或课堂使用为目的制作或分发本作品的部分或全部数字或硬拷贝，前提是未以盈利或商业优势为目的进行分发，且副本承载本声明和首页的完整引用。本作品中由ACM以外其他人拥有的组件的版权必须予以尊重。允许以注明方式引用进行摘要。如需以其他方式复制、重新发布、发布到服务器或重新分发到列表，则需要事先获得特定许可和/或费用。如需许可，请联系 permissions@acm.org。

WSDM 2018，2018年2月5日至9日，美国加利福尼亚州玛丽安德尔湾

© 2018 ACM. 978-1-4503-5581-0/18/02...\$15.00

DOI: 10.1145/3159652.3159656

## 1 引言

推荐系统已成为许多应用中的核心技术。大多数系统，如Top-N推荐[9][19]，基于用户的一般偏好推荐item，而不关注item的新近度。

例如，某些用户始终偏好苹果产品而非三星产品。一般偏好代表用户的长期和静态行为。另一类用户行为是序列模式，其中下一个item或行动更可能依赖于用户最近参与的item或行动。序列模式代表用户的短期和动态行为，来自紧密时间范围内item之间的某种关系。例如，用户很可能在购买iPhone后不久购买手机配件，尽管一般情况下用户不购买手机配件。在这种情况下，仅考虑一般偏好系统将错失在销售iPhone后推荐手机配件的机会，因为购买手机配件不是长期用户行为。

### 1.1 Top-N序列推荐

为建模用户的序列模式，[17, 21]中的工作考虑了Top-N序列推荐，推荐用户在近未来可能交互的N个item。该问题假设一组用户 $U = \{u_1, u_2, \cdots, u_{|U|}\}$ 和一个item全域 $I = \{i_1, i_2, \cdots, i_{|I|}\}$。每个用户 $u$ 关联一个来自 $I$ 的item序列 $S_u = (S_u_1, \cdots, S_u_{|S_u|})$，其中 $S_u_i \in I$。$S_u_t$ 的索引 $t$ 表示行动在序列 $S_u$ 中发生的顺序，而非如[14, 31, 34]等时间推荐中的绝对时间戳。给定所有用户的序列 $S_u$，目标是推荐每个用户一个最大化其未来需求的item列表，同时考虑一般偏好和序列模式。与传统Top-N推荐不同，Top-N序列推荐将用户行为建模为item序列而非item集合。

### 1.2 先前工作的局限性

基于马尔可夫链的模型[2, 6, 21, 30]是Top-N序列推荐的早期方法，其中L阶马尔可夫链基于L个先前行动进行推荐。一阶马尔可夫链是使用最大似然估计学习的item到item转移矩阵。Rendle等人提出的分解个性化马尔可夫链（FPMC，Factorized Personalized Markov Chains）[21]及其变体[2]通过将该转移矩阵分解为两个潜在低秩子矩阵来改进该方法。He等人提出的基于item相似度模型的分解序列预测（Fossil，Factorized Sequential Prediction with Item Similarity ModeLs）[6]通过使用先前item潜在表示的加权和聚合将该方法推广到高阶马尔可夫链。然而，现有方法存在两个主要局限性：

**未能建模联合级序列模式。** 如图1a所示，马尔可夫链仅建模点级序列模式，其中每个先前行动（蓝色）单独而非集体影响目标行动（黄色）。FPMC和Fossil属于此类别。尽管Fossil考虑了高阶马尔可夫链，但整体影响是从一阶马尔可夫转移矩阵分解出的先前item潜在表示的加权和。这种点级影响的聚合不足以建模图1b所示的联合级影响，其中几个先前行动按顺序共同影响目标行动。例如，同时购买牛奶和黄油导致购买面粉的概率高于单独购买牛奶或黄油；同时购买内存和硬盘比仅购买其中一个组件更能表明接下来会购买操作系统。

**未能允许跳跃行为。** 现有模型未考虑图1c所示的序列模式的跳跃行为，其中过去行为的影响可能跳过几步仍然具有强度。例如，游客按顺序在机场、酒店、餐厅、酒吧和景点签到。虽然机场和酒店的签到不直接先于景点的签到，但它们与后者有强关联。另一方面，餐厅或酒吧的签到对景点签到影响很小（因为它们不一定发生）。L阶马尔可夫链未明确建模此类跳跃行为，因为它假设L个先前步骤对紧接的下一步有影响。

**图1：点级和联合级动态模式影响示例，马尔可夫链阶数 $L=3$。(a) 点级；(b) 联合级，无跳跃；(c) 联合级，跳过一次。**

为提供联合级影响和跳跃行为的证据，我们从两个真实数据集MovieLens和Gowalla（数据集详情见第4节）中挖掘以下形式的序列关联规则[1, 4]：

$$
(S_u_{t-L}, \cdots, S_u_{t-2}, S_u_{t-1}) \rightarrow S_u_t \qquad (1)
$$

对于上述形式的规则 $X \rightarrow Y$，支持计数 $\text{sup}(XY)$ 是 $X$ 和 $Y$ 按规则中顺序出现的序列数，置信度 $\frac{\text{sup}(XY)}{\text{sup}(X)}$ 是在 $X$ 出现的序列中 $Y$ 跟随 $X$ 的百分比。该规则表示 $X$ 中所有item对 $Y$ 的联合影响。通过将右侧更改为 $S_u_{t+1}$ 或 $S_u_{t+2}$，该规则也捕获了一步或两步跳跃的影响。图2总结了在最小支持计数为5和最小置信度为50%下发现的规则数与马尔可夫阶数 $L$ 和跳跃步数的关系（我们还尝试了10%、20%和30%的最小置信度，趋势类似）。大多数规则的阶数为 $L=2$ 和 $L=3$，且规则的置信度随 $L$ 增大而更高。该图还表明相当数量的规则有1或2步跳跃。这些发现支持了联合级影响和跳跃行为的存在。

**图2：关联规则数与 $L$ 和跳跃步数的关系。最小支持计数为5，最小置信度为50%。(a) MovieLens；(b) Gowalla。**

### 1.3 贡献

为解决上述现有工作的局限性，我们提出卷积序列嵌入推荐模型（Caser，ConvolutionAl Sequence Embedding Recommendation Model）作为Top-N序列推荐的解决方案。该模型利用卷积神经网络（CNN，Convolutional Neural Network）的卷积滤波器在图像识别[11, 16]和自然语言处理[12]中捕获局部特征的最新成功。Caser的新颖之处在于将先前L个item表示为 $L \times d$ 矩阵 $E$，其中 $d$ 是潜在维度数，行保留item的顺序。类似于[12]，我们将该嵌入矩阵视为L个item在潜在空间中的"图像"，并使用各种卷积滤波器搜索序列模式作为该"图像"的局部特征。然而，与图像识别不同，该"图像"不是在输入中给定的，必须与所有滤波器同时学习。

与现有方法相比，Caser提供了几个独特优势。(1) Caser使用水平和垂直卷积滤波器捕获点级、联合级和跳跃行为的序列模式。(2) Caser同时建模用户的一般偏好和序列模式，并在单一统一框架中泛化多个现有最先进方法。(3) Caser在真实数据集上优于Top-N序列推荐的最先进方法。在本文其余部分，我们在第2节讨论进一步的相关工作，第3节讨论Caser方法，第4节讨论实验研究。

## 2 进一步相关工作

传统推荐方法，如协同过滤[24]、矩阵分解（MF，Matrix Factorization）[15, 22]和Top-N推荐[9][19]，不适合捕获序列模式，因为它们不建模行动的顺序。序列模式挖掘的早期工作[1, 4]基于统计共现[17]发现显式序列关联规则。这种方法依赖于模式的显式表示，因此可能错失未观察状态中的模式。此外，它还面临潜在大搜索空间、对阈值设置敏感以及大量规则（多数冗余）的问题。

受限玻尔兹曼机（RBM，Restricted Boltzmann Machine）[23]是首个成功应用于推荐问题的2层神经网络。自编码器框架[25, 29]及其变体去噪自编码器[32]也产生了良好的推荐性能。卷积神经网络（CNN）[36]已被用于从用户评论中提取用户偏好。这些工作都不是用于序列推荐。

循环神经网络（RNN，Recurrent Neural Network）被用于基于会话的推荐[8, 10]。虽然RNN在建模序列方面展示了令人印象深刻的能力[18]，但其顺序连接的网络结构在序列推荐设置下可能效果不佳。因为在序列推荐问题中，并非所有相邻行动都有依赖关系（例如用户在 $i_1$ 之后购买 $i_2$ 仅因为她喜欢 $i_2$）。我们在第4节的实验结果验证了这一点：基于RNN的方法在数据集包含大量序列模式时表现更好。而我们提出的方法不将序列模式建模为相邻行动，而是采用CNN的卷积滤波器，将序列模式建模为先前item嵌入的局部特征。这种方法提供了在单一统一框架中建模点级和联合级序列模式以及跳跃行为的灵活性。事实上，我们将展示Caser泛化了多个最先进方法。

一个相关但不同的问题是时间推荐[26, 31, 34]。例如，时间推荐在早晨而非晚上推荐咖啡，而我们的Top-N序列推荐会在用户购买iPhone后不久推荐手机配件，与时间无关。显然，这两个问题不同且需要不同的解决方案。

## 3 提出的方法

所提模型——卷积序列嵌入推荐（Caser）——结合卷积神经网络（CNN）学习序列特征，以及隐因子模型（LFM，Latent Factor Model）学习用户特定特征。Caser网络设计的目标是多方面的：同时在联合级和点级捕获用户的一般偏好和序列模式，以及捕获跳跃行为，所有这些都在未观察空间中。如图3所示，Caser由三个组件组成：嵌入查找、卷积层和全连接层。为训练CNN，对于每个用户 $u$，我们从用户序列 $S_u$ 中提取每L个连续item作为输入，其后的T个item作为目标，如图3左侧所示。这是通过在用户序列上滑动大小为 $L+T$ 的窗口完成的，每个窗口为 $u$ 生成一个训练实例，表示为三元组（$u$，前L个item，后T个item）。

**图3：Caser的网络架构。矩形框代表用户序列中的item $S_u_1, \cdots, S_u_{|S_u|}$，而内部有圆圈的矩形框代表某个向量如用户嵌入 $P_u$。虚线矩形框是不同大小的卷积滤波器。卷积层中的红圈代表每个卷积结果中的最大值。这里我们使用前4个行动（$L=4$）来预测该用户在接下来2步（$T=2$）中将与哪些item交互。**

### 3.1 嵌入查找

Caser通过将前L个item的嵌入馈入神经网络来捕获潜在空间中的序列特征。item $i$ 的嵌入 $Q_i \in \mathbb{R}^d$ 是其隐因子的类似概念。这里 $d$ 是潜在维度数。嵌入查找操作检索前L个item的嵌入并将它们堆叠在一起，为用户 $u$ 在时间步 $t$ 生成矩阵：

$$
E_{(u,t)} = \begin{bmatrix} Q_{S_u_{t-L}} \\ \vdots \\ Q_{S_u_{t-2}} \\ Q_{S_u_{t-1}} \end{bmatrix} \qquad (2)
$$

除了item嵌入外，我们还有用户 $u$ 的嵌入 $P_u \in \mathbb{R}^d$，表示潜在空间中的用户特征。这些嵌入在图3的嵌入查找框中用蓝色和紫色圆圈表示。

### 3.2 卷积层

我们的方法利用CNN卷积滤波器在图像识别[11, 16]和自然语言处理[12]中捕获局部特征的最新成功。借鉴在文本分类[12]中使用CNN的思想，我们的方法将 $L \times d$ 矩阵 $E$ 视为前L个item在潜在空间中的"图像"，并将序列模式视为该"图像"的局部特征。这种方法使得可以使用卷积滤波器搜索序列模式。图4展示了两个"水平滤波器"捕获两个联合级序列模式。这些滤波器表示为 $h \times d$ 矩阵，高度 $h=2$，全宽度等于 $d$。它们通过在 $E$ 的行上滑动来拾取序列模式的信号。例如，第一个滤波器通过在机场和酒店具有较大值的潜在维度中具有较大值来拾取序列模式"(Airport, Hotel) $\rightarrow$ Great Wall"。类似地，"垂直滤波器"是 $L \times 1$ 矩阵，将在 $E$ 的列上滑动。更多细节如下解释。与图像识别不同，"图像" $E$ 不是给定的，因为所有item $i$ 的嵌入 $Q_i$ 必须与所有滤波器同时学习。

**图4：颜色越深表示值越大。第一个滤波器通过与机场和酒店的嵌入交互并跳过快餐和餐厅的嵌入来捕获"(Airport, Hotel) $\rightarrow$ Great Wall"。第二个滤波器捕获"(Fast Food, Restaurant) $\rightarrow$ Bar"。**

**水平卷积层。** 该层如图3第二组件的上部所示，有 $n$ 个水平滤波器 $F_k \in \mathbb{R}^{h \times d}$，$1 \leq k \leq n$。$h \in \{1, \cdots, L\}$ 是滤波器的高度。例如，如果 $L=4$，可以选择 $n=8$ 个滤波器，$h$ 为 $\{1, 2, 3, 4\}$ 中的每一个各两个。$F_k$ 将从上到下在 $E$ 上滑动，并与item $i$（$1 \leq i \leq L-h+1$）的 $E$ 的所有水平维度交互。交互的结果是第 $i$ 个卷积值：

$$
c_i^k = \phi_c(E_{i:i+h-1} \odot F_k) \qquad (3)
$$

其中符号 $\odot$ 表示内积运算符，$\phi_c(\cdot)$ 是卷积层的激活函数。该值是 $F_k$ 与由 $E$ 的第 $i$ 行到第 $i-h+1$ 行形成的子矩阵（记为 $E_{i:i+h-1}$）之间的内积。$F_k$ 的最终卷积结果是向量：

$$
c^k = [c_1^k \; c_2^k \; \cdots \; c_{L-h+1}^k] \qquad (4)
$$

然后我们对 $c^k$ 应用最大池化操作，从该特定滤波器产生的所有值中提取最大值。最大值捕获了滤波器提取的最显著特征。因此，对于该层中的 $n$ 个滤波器，输出值 $o \in \mathbb{R}^n$ 为：

$$
o = \{\max(c^1), \max(c^2), \cdots, \max(c^n)\} \qquad (5)
$$

水平滤波器通过嵌入 $E$ 与每组连续 $h$ 个item交互。嵌入和滤波器都被学习以最小化编码目标item预测误差的目标函数（详见第3.4节）。通过滑动各种高度的滤波器，无论位置如何都将拾取显著信号。因此，水平滤波器可以被训练来捕获具有多种联合大小的联合级模式。

**垂直卷积层。** 该层如图3第二组件的下部所示。我们使用波浪号（$\sim$）表示该层的符号。假设有 $\tilde{n}$ 个垂直滤波器 $\tilde{F}_k \in \mathbb{R}^{L \times 1}$，$1 \leq k \leq \tilde{n}$。每个滤波器 $\tilde{F}_k$ 通过从左到右在 $E$ 上滑动 $d$ 次与 $E$ 的列交互，产生垂直卷积结果 $\tilde{c}^k$：

$$
\tilde{c}^k = [\tilde{c}_1^k \; \tilde{c}_2^k \; \cdots \; \tilde{c}_d^k] \qquad (6)
$$

对于内积交互，容易验证该结果等于以 $\tilde{F}_k$ 为权重对 $E$ 的 $L$ 行的加权和：

$$
\tilde{c}^k = \sum_{l=1}^{L} \tilde{F}_{k_l} \cdot E_l \qquad (7)
$$

其中 $E_l$ 是 $E$ 的第 $l$ 行。因此，使用垂直滤波器我们可以学习聚合前L个item的嵌入，类似于Fossil [6]聚合前L个item潜在表示的加权和。区别在于每个滤波器 $\tilde{F}_k$ 充当不同的聚合器。因此，与Fossil类似，这些垂直滤波器通过对先前item潜在表示的加权和捕获点级序列模式。Fossil为每个用户使用单个加权和，而我们可以使用 $\tilde{n}$ 个全局垂直滤波器为所有用户产生 $\tilde{n}$ 个加权和 $\tilde{o} \in \mathbb{R}^{d\tilde{n}}$：

$$
\tilde{o} = [\tilde{c}^1 \; \tilde{c}^2 \; \cdots \; \tilde{c}^{\tilde{n}}] \qquad (8)
$$

由于其用途是聚合，垂直滤波器与水平滤波器有一些差异：(1) 每个垂直滤波器的大小固定为 $L \times 1$。这是因为 $E$ 的每一列对我们来说是潜在的，同时与多个连续列交互没有意义。(2) 无需对垂直卷积结果应用最大池化操作，因为我们希望保留每个潜在维度的聚合。因此，该层的输出为 $\tilde{o}$。

### 3.3 全连接层

我们将两个卷积层的输出拼接并馈入全连接神经网络层以获得更高级和抽象的特征：

$$
z = \phi_a(W \begin{bmatrix} o \\ \tilde{o} \end{bmatrix} + b) \qquad (9)
$$

其中 $W \in \mathbb{R}^{d \times (n + d\tilde{n})}$ 是将拼接层投影到 $d$ 维隐藏层的权重矩阵，$b \in \mathbb{R}^d$ 是对应的偏差项，$\phi_a(\cdot)$ 是全连接层的激活函数。$z \in \mathbb{R}^d$ 就是我们所说的卷积序列嵌入，编码了前L个item的各种序列特征。

为捕获用户的一般偏好，我们还查找用户嵌入 $P_u$ 并将两个 $d$ 维向量 $z$ 和 $P_u$ 拼接在一起，投影到具有 $|I|$ 个节点的输出层：

$$
y_{(u,t)} = W' \begin{bmatrix} z \\ P_u \end{bmatrix} + b' \qquad (10)
$$

其中 $b' \in \mathbb{R}^{|I|}$ 和 $W' \in \mathbb{R}^{|I| \times 2d}$ 分别是输出层的偏差项和权重矩阵。如第3.4节所述，输出层中的值 $y_{(u,t)}_i$ 与用户 $u$ 在时间步 $t$ 与item $i$ 交互的可能性相关。$z$ 旨在捕获短期序列模式，而用户嵌入 $P_u$ 捕获用户的长期一般偏好。我们将用户嵌入 $P_u$ 放在最后一个隐藏层有几个原因：(1) 如我们在第3.6节将看到的，它可以具有泛化其他模型的能力。(2) 我们可以用其他泛化模型的参数预训练模型参数。如[7]所述，此类预训练对模型性能至关重要。

### 3.4 网络训练

为训练网络，我们将输出层的值 $y_{(u,t)}$ 通过以下方式转换为概率：

$$
p(S_u_t \mid S_u_{t-1}, S_u_{t-2}, \cdots, S_u_{t-L}) = \sigma(y_{(u,t)}_{S_u_t}) \qquad (11)
$$

其中 $\sigma(x) = 1/(1 + e^{-x})$ 是sigmoid函数。令 $C_u = \{L+1, L+2, \ldots, |S_u|\}$ 为我们希望为用户 $u$ 进行预测的时间步集合。数据集中所有序列的似然为：

$$
p(S \mid \Theta) = \prod_u \prod_{t \in C_u} \sigma(y_{(u,t)}_{S_u_t}) \prod_{j, S_u_t} (1 - \sigma(y_{(u,t)}_j)) \qquad (12)
$$

为进一步捕获跳跃行为，我们可以一次考虑接下来的T个目标item $D_u_t = \{S_u_t, S_u_{t+1}, \ldots, S_u_{t+T}\}$，通过将上述方程中的紧接下一个item $S_u_t$ 替换为 $D_u_t$。取似然的负对数，我们得到目标函数，也称为二元交叉熵损失：

$$
\ell = \sum_u \sum_{t \in C_u} \sum_{i \in D_u_t} -\log(\sigma(y_{(u,t)}_i)) + \sum_{j, i} -\log(1 - \sigma(y_{(u,t)}_j)) \qquad (13)
$$

遵循先前工作[6, 21, 32]，对于每个目标item $i$，我们在第二项中随机采样几个（实验中为3个）负实例 $j$。

模型参数 $\Theta = \{P, Q, F, \tilde{F}, W, W', b, b'\}$ 通过在训练集上最小化公式(13)中的目标函数来学习，而超参数（如 $d, n, \tilde{n}, L, T$）通过在验证集上网格搜索来调整。我们采用随机梯度下降（SGD，Stochastic Gradient Descent）的变体——自适应矩估计（Adam，Adaptive Moment Estimation）[13]以加快收敛，批大小为100。为控制模型复杂度和避免过拟合，我们使用两种正则化方法：对所有模型参数应用L2范数，在全连接层上使用50%丢弃率的丢弃（Dropout）[27]技术。我们使用MatConvNet [28]实现了Caser。整个训练时间与训练实例数成正比。例如，在4核i7 CPU和32GB RAM机器上，MovieLens数据约需1小时，Gowalla数据约需2小时，Foursquare数据约需2小时，Tmall数据约需1小时。这些时间与Fossil [6]的运行时间相当，可通过使用GPU进一步减少。

### 3.5 推荐

获得训练好的神经网络后，为在时间步 $t$ 为用户 $u$ 进行推荐，我们取 $u$ 的潜在嵌入 $P_u$ 并提取其最后L个item的嵌入（由公式(2)给出）作为神经网络输入。我们推荐在输出层 $y$ 中具有最高值的N个item。为所有用户进行推荐的复杂度为 $O(|U||I|d)$，其中忽略了卷积操作的复杂度。注意，目标item数 $T$ 是模型训练期间使用的超参数，而 $N$ 是模型训练后推荐的item数。

### 3.6 与现有模型的关系

我们展示Caser是多个先前模型的泛化。

**Caser vs. MF。** 通过丢弃所有卷积层和所有偏差项，我们的模型变为使用用户嵌入作为用户隐因子及其关联权重作为item隐因子的朴素隐因子模型（LFM）。MF通常包含偏差项¹，在我们的模型中即 $b'$。丢弃所有卷积层后，结果模型与MF相同：

$$
y_i^u = W'_i \begin{bmatrix} 0 \\ P_u \end{bmatrix} + b'_i \qquad (14)
$$

**Caser vs. FPMC。** FPMC将分解的一阶马尔可夫链与LFM融合，并通过贝叶斯个性化排序（BPR，Bayesian Personalized Ranking）优化。虽然Caser使用不同的优化准则（即交叉熵），但它能够通过将前一个item的嵌入复制到隐藏层 $z$ 且不使用任何偏差项来泛化FPMC：

$$
y_{(u,t)}_i = W'_i \begin{bmatrix} Q_{S_u_{t-1}} \\ P_u \end{bmatrix} \qquad (15)
$$

由于FPMC使用BPR作为准则，我们的模型与FPMC不完全相同。然而，BPR限制为在每个时间步只有1个目标和负样本。我们的交叉熵损失没有这些限制。

**Caser vs. Fossil。** 通过省略水平卷积层、使用一个垂直滤波器并将垂直卷积结果 $\tilde{c}$ 复制到隐藏层 $z$，我们得到：

$$
y_{(u,t)}_i = W'_i \begin{bmatrix} \tilde{c} \\ P_u \end{bmatrix} + b'_i \qquad (16)
$$

如公式(7)的讨论，该垂直滤波器充当前L个item嵌入的加权和，与Fossil中类似，尽管Fossil使用相似度模型而非LFM，并在与马尔可夫模型相同的潜在空间中分解它。另一个区别是Fossil为每个用户使用一个局部加权，而我们通过垂直滤波器使用多个全局加权。

¹ Top-N推荐为每个用户单独排序item，对用户偏差和全局偏差不变。

## 4 实验

我们将Caser与最先进方法进行比较。Caser的源代码和处理后的数据集可在线获取²。

² https://github.com/graytowne/caser

### 4.1 实验设置

**数据集。** 仅当数据集包含序列模式时，序列推荐才有意义。为识别此类数据集，我们对几个公共数据集应用序列关联规则挖掘，并计算其序列强度（SI，Sequential Intensity）：

$$
\text{SI} = \frac{\text{#rules}}{\text{#users}} \qquad (17)
$$

分子是使用支持（即5）和置信度（即50%）的最小阈值找到的公式(1)形式的总规则数，马尔可夫阶数 $L$ 范围从1到5。分母是用户总数。我们使用SI估计数据集中序列信号的强度。

四个数据集及其SI在表1中描述。MovieLens³是广泛使用的电影评分数据。由[3]构建的Gowalla⁴和从[33]获得的Foursquare包含通过用户场所签到的隐式反馈。Tmall是中国最大的B2C平台，是从IJCAI 2015竞赛⁵获得的用户购买数据，旨在预测重复买家。遵循先前工作[6, 20, 32]，我们将所有数值评分转换为1的隐式反馈。我们还移除了少于 $n$ 次反馈的冷启动用户和item，因为处理冷启动推荐通常在文献中被视为单独问题[6, 7, 21, 32]。MovieLens、Gowalla、Foursquare和Tmall的 $n$ 分别为5、15、10、10。之前在[5, 6]中使用的Amazon数据由于其SI未被使用（"Office Products"类别为0.0026，"Clothing, Shoes, Jewelry"和"Video Games"类别为0.0019），换句话说，其序列信号远弱于上述数据集。

³ https://grouplens.org/datasets/movielens/1m/
⁴ https://snap.stanford.edu/data/loc-gowalla.html
⁵ https://ijcai-15.org/index.php/repeat-buyers-prediction-competition

遵循[17, 33, 35]，我们将每个用户序列中前70%的行动作为训练集，使用接下来的10%行动作为验证集来搜索所有模型的最优超参数设置。每个用户序列中剩余的20%行动用作评估模型性能的测试集。

**评估指标。** 如[19, 21, 29, 32]，我们通过Precision@N、Recall@N和平均精度均值（MAP，Mean Average Precision）评估模型。给定用户的前N个预测item列表（记为 $\hat{R}_{1:N}$）和其序列中最后20%的行动（记为 $R$，即测试集），Precision@N和Recall@N计算如下：

$$
\text{Prec@N} = \frac{|R \cap \hat{R}_{1:N}|}{N}, \quad \text{Recall@N} = \frac{|R \cap \hat{R}_{1:N}|}{|R|} \qquad (18)
$$

我们报告所有用户这些值的平均值。$N \in \{1, 5, 10\}$。平均精度（AP，Average Precision）定义为：

$$
\text{AP} = \frac{\sum_{N=1}^{|\hat{R}|} \text{Prec@N} \times \text{rel}(N)}{|\hat{R}|} \qquad (19)
$$

其中如果 $\hat{R}$ 中第N个item在 $R$ 中，则 $\text{rel}(N) = 1$。平均精度均值（MAP）是所有用户AP的平均值。

**表1：数据集统计信息。**

| 数据集 | 序列强度 | 用户数 | item数 | 平均行动数/用户 | 稀疏度 |
|--------|---------|--------|--------|----------------|--------|
| MovieLens | 0.3265 | 6.0k | 3.4k | 165.50 | 95.16% |
| Gowalla | 0.0748 | 13.1k | 14.0k | 40.74 | 99.71% |
| Foursquare | 0.0378 | 10.1k | 23.4k | 30.16 | 99.87% |
| Tmall | 0.0104 | 23.8k | 12.2k | 13.93 | 99.89% |

### 4.2 性能比较

我们将第3节提出的Caser与以下基线进行比较。

- **POP。** 所有item按其在所有用户序列中的流行度排序，流行度由交互次数决定。

- **BPR。** 结合矩阵分解模型，贝叶斯个性化排序（BPR）[20]是隐式反馈数据上非序列item推荐的最先进方法。

- **FMC和FPMC。** 如[21]中介绍，分解马尔可夫链（FMC，Factorized Markov Chain）将一阶马尔可夫转移矩阵分解为两个低维子矩阵，FPMC是FMC和LFM的融合。这些是最先进的序列推荐方法。FPMC在每步允许多个item的篮子。对于我们的序列推荐问题，每个篮子只有一个item。

- **Fossil。** Fossil [6]建模高阶马尔可夫链，并使用相似度模型而非LFM来建模一般用户偏好。

- **GRU4Rec。** 这是[8]提出的基于会话的推荐。该模型使用RNN捕获序列依赖并进行预测。

对于每种方法，应用网格搜索使用验证集找到超参数的最优设置。包括潜在维度 $d$ 从 $\{5, 10, 20, 30, 50, 100\}$ 中选择，正则化超参数，以及学习率从 $\{1, 10^{-1}, \ldots, 10^{-4}\}$ 中选择。对于Fossil、Caser和GRU4Rec，马尔可夫阶数 $L$ 从 $\{1, \cdots, 9\}$ 中选择。对于Caser本身，水平滤波器的高度 $h$ 从 $\{1, \cdots, L\}$ 中选择，目标数 $T$ 从 $\{1, 2, 3\}$ 中选择，激活函数 $\phi_a$ 和 $\phi_c$ 从 $\{\text{identity}, \text{sigmoid}, \text{tanh}, \text{relu}\}$ 中选择。对于每个高度 $h$，水平滤波器的数量从 $\{4, 8, 16, 32, 64\}$ 中选择。垂直滤波器的数量从 $\{1, 2, 4, 8, 16\}$ 中选择。我们报告每种方法在其最优超参数设置下的结果。

**表2：四个数据集上的性能比较。**

六种基线和Caser的最佳结果总结在表2中。每行的最佳表现者以粗体显示。最后一列是Caser相对于最佳基线的改进，定义为 $\frac{\text{Caser} - \text{baseline}}{\text{baseline}}$。除MovieLens外，Caser在所有测试的N上以较大优势改进了最佳基线的三个指标。在基线方法中，序列推荐器（如FPMC和Fossil）通常在所有数据集上优于非序列推荐器（即BPR），表明考虑序列信息的重要性。FPMC和Fossil在所有数据集上优于FMC，表明个性化的有效性。在MovieLens上，GRU4Rec取得了接近Caser的性能，但在其他三个数据集上表现差得多。事实上，MovieLens比其他三个数据集具有更多序列信号，因此基于RNN的GRU4Rec可以在MovieLens上表现良好，但容易在其他三个数据集的训练集上产生偏差（尽管使用了[8]中描述的正则化和丢弃）。此外，GRU4Rec的推荐是基于会话的而非个性化的，这在一定程度上放大了泛化误差。

在以下研究中，我们通过保持其余超参数在其最优设置来逐一检查超参数 $d$、$L$、$T$ 的影响。我们聚焦于MAP，因为它是整体性能指标且与其他指标一致。

#### 4.2.1 潜在维度 $d$ 的影响

图5展示了在保持其他最优超参数不变的情况下不同 $d$ 的MAP。在较密集的MovieLens上，较大的 $d$ 并不总能带来更好的模型性能。当 $d$ 选择适当时模型达到最佳性能，对于较大的 $d$ 由于过拟合而变差。但对于其他三个较稀疏的数据集，每个模型需要更多潜在维度来达到最佳结果。对于所有数据集，Caser使用相对较少的潜在维度即可超越最强基线性能。

**图5：MAP（y轴）与潜在维度数 $d$（x轴）的关系。**

#### 4.2.2 马尔可夫阶数 $L$ 和目标数 $T$ 的影响

我们改变 $L$ 以探索Fossil、GRU4Rec和Caser能从高阶信息中获得多少收益，同时保持其他最优超参数不变。Caser-1、Caser-2和Caser-3表示目标数 $T$ 分别为1、2、3的Caser，以研究跳跃行为的效果。结果如图6所示。在密集的MovieLens上，Caser最好地利用了较大 $L$ 提供的额外信息，Caser-3表现最佳，表明跳跃步骤的收益。然而，对于较稀疏的数据集，所有模型未从较大的 $L$ 中一致获益。这是合理的，因为对于稀疏数据集，高阶马尔可夫链倾向于引入额外信息和更多噪声。在大多数情况下，Caser-2在这三个数据集上略微优于其他模型。

**图6：MAP（y轴）与马尔可夫阶数 $L$（x轴）的关系。Caser-1、Caser-2和Caser-3表示目标数 $T$ 分别设置为1、2、3的Caser。**

#### 4.2.3 Caser组件分析

最后，我们在保持所有超参数在其最优设置的同时评估Caser各组件——水平卷积层（即 $o$）、垂直卷积层（即 $\tilde{o}$）和个性化（即 $P_u$）——对整体性能的贡献。MovieLens和Gowalla的结果如表3所示；其他两个数据集的结果类似。对于 $x \in \{p, h, v, vh, ph, pv, pvh\}$，Caser-x表示启用组件 $x$ 的Caser。$h$ 表示水平卷积层；$v$ 表示垂直卷积层；$p$ 表示个性化，类似于BPR仅使用LFM。缺失的组件通过将其对应的 $o$、$\tilde{o}$ 和 $P_u$ 设置为零来表示。例如，$vh$ 表示通过将 $P_u$ 设置为全零同时使用垂直和水平卷积层，$pv$ 表示通过将 $o$ 设置为全零同时使用垂直卷积层和个性化。Caser-p表现最差，而Caser-h、Caser-v和Caser-vh显著提高了性能，表明将Top-N序列推荐视为传统Top-N推荐会丢失有用信息，建模联合级和点级的序列模式对提高预测有用。对于两个数据集，通过联合使用Caser的所有部分（即Caser-pvh）达到最佳性能。

**表3：MAP与Caser组件的关系。**

| 组件 | MovieLens | Gowalla |
|------|-----------|---------|
| Caser-p | 0.0935 | 0.0777 |
| Caser-h | 0.1304 | 0.0805 |
| Caser-v | 0.1403 | 0.0841 |
| Caser-vh | 0.1448 | 0.0856 |
| Caser-ph | 0.1372 | 0.0911 |
| Caser-pv | 0.1494 | 0.0921 |
| Caser-pvh | 0.1507 | 0.0928 |

### 4.3 网络可视化

我们更仔细地查看一些训练好的网络和预测。图7展示了在MovieLens上以 $L=9$ 训练Caser后四个垂直卷积滤波器的值。从微观角度看，四个滤波器被训练为多样化，但从宏观角度看，它们遵循从过去位置到最近位置的上升趋势。每个垂直滤波器作为对先前行动嵌入加权的方式（见第3节的相关讨论），这一趋势表明Caser更强调最近行动，展示了与传统Top-N推荐的主要区别。

**图7：在MovieLens数据上以 $L=9$ 训练的模型的四个垂直卷积滤波器的可视化。**

为查看水平滤波器的有效性，图8(a)展示了Caser推荐的Top $N=3$ 排序电影，即 $\hat{R}_1$（Mad Max）、$\hat{R}_2$（Star War）、$\hat{R}_3$（Star Trek）按此顺序，对于一个有 $L=5$ 个先前电影（$S_1$（13th Warrior）、$S_2$（American Beauty）、$S_3$（Star Trek）、$S_4$（Star Trek III）、$S_5$（Star Trek IV））的用户。$\hat{R}_3$ 是真实标签（即用户序列中的下一个电影）。注意 $\hat{R}_1$ 和 $\hat{R}_2$ 与 $\hat{R}_3$ 非常相似，即都是动作和科幻电影，因此也被推荐给用户。图8(b)展示了在训练好的网络中通过将某些先前L个电影的item嵌入设置为零来屏蔽后 $\hat{R}_3$ 的新排名。屏蔽 $S_1$ 和 $S_2$ 实际上将 $\hat{R}_3$ 的排名从3提高到2；事实上，$S_1$ 和 $S_2$ 是历史或浪漫电影，对推荐 $\hat{R}_3$ 充当噪声。屏蔽 $S_3$、$S_4$ 和 $S_5$ 中的每一个都降低了 $\hat{R}_3$ 的排名，因为这些电影与 $\hat{R}_3$ 属于同一类别。屏蔽 $S_3$、$S_4$ 和 $S_5$ 全部后下降最多。该研究清楚表明我们的模型正确捕获了 $\hat{R}_3$ 对相关 $\{S_3, S_4, S_5\}$ 作为联合级序列特征来推荐 $\hat{R}_3$ 的依赖。

**图8：水平卷积滤波器在MovieLens数据上捕获联合级序列模式的有效性。(a) 预测；(b) 屏蔽item后 $\hat{R}_3$ 的新排名。**

## 5 结论

Caser是Top-N序列推荐的新颖解决方案，通过将最近行动建模为时间和潜在维度间的"图像"，并使用卷积滤波器学习序列模式。这种方法提供了一个统一且灵活的网络结构来捕获序列推荐的许多重要特征，即点级和联合级序列模式、跳跃行为以及长期用户偏好。我们在公共真实数据集上的实验和案例研究表明，Caser优于Top-N序列推荐的最先进方法。

致谢：第二作者的工作部分由加拿大自然科学与工程研究委员会的发现基金支持。

## 参考文献

[1] Rakesh Agrawal and Ramakrishnan Srikant. 1995. Mining sequential patterns. In International Conference on Data Engineering. IEEE, 3–14.

[2] Chen Cheng, Haiqin Yang, Michael R Lyu, and Irwin King. 2013. Where You Like to Go Next: Successive Point-of-Interest Recommendation. In International Joint Conference on Artificial Intelligence. 2605–2611.

[3] Eunjoon Cho, Seth A Myers, and Jure Leskovec. 2011. Friendship and mobility: user movement in location-based social networks. In International Conference on Knowledge Discovery and Data Mining. ACM, 1082–1090.

[4] Jiawei Han, Jian Pei, and Micheline Kamber. 2011. Data mining: concepts and techniques. Elsevier.

[5] R. He, W.-C. Kang, and J. McAuley. 2017. Translation-based recommendation. In ACM Conference on Recommender systems.

[6] R. He and J. McAuley. 2016. Fusing Similarity Models with Markov Chains for Sparse Sequential Recommendation. In International Conference on Data Mining. IEEE.

[7] Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu, and Tat-Seng Chua. 2017. Neural collaborative filtering. In International Conference on World Wide Web. ACM, 173–182.

[8] Balázs Hidasi, Alexandros Karatzoglou, Linas Baltrunas, and Domonkos Tikk. 2015. Session-based recommendations with recurrent neural networks. arXiv preprint arXiv:1511.06939 (2015).

[9] Yifan Hu, Yehuda Koren, and Chris Volinsky. 2008. Collaborative filtering for implicit feedback datasets. In International Conference on Data Mining. IEEE, 263–272.

[10] Dietmar Jannach and Malte Ludewig. 2017. When Recurrent Neural Networks meet the Neighborhood for Session-Based Recommendation. In ACM Conference on Recommender systems. ACM, 306–310.

[11] Andrej Karpathy, George Toderici, Sanketh Shetty, Thomas Leung, Rahul Sukthankar, and Li Fei-Fei. 2014. Large-scale video classification with convolutional neural networks. In IEEE conference on Computer Vision and Pattern Recognition. 1725–1732.

[12] Yoon Kim. 2014. Convolutional Neural Networks for Sentence Classification. In Conference on Empirical Methods on Natural Language Processing. ACL, 1756–1751.

[13] Diederik Kingma and Jimmy Ba. 2014. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980 (2014).

[14] Yehuda Koren. 2010. Collaborative filtering with temporal dynamics. Commun. ACM 53, 4 (2010), 89–97.

[15] Yehuda Koren, Robert Bell, and Chris Volinsky. 2009. Matrix factorization techniques for recommender systems. Computer 42, 8 (2009).

[16] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. 2012. Imagenet classification with deep convolutional neural networks. In Advances in Neural Information Processing Systems. 1097–1105.

[17] Duen-Ren Liu, Chin-Hui Lai, and Wang-Jung Lee. 2009. A hybrid of sequential rules and collaborative filtering for product recommendation. Information Sciences 179, 20 (2009), 3505–3519.

[18] Tomas Mikolov, Martin Karaﬁát, Lukas Burget, Jan Cernocký, and Sanjeev Khudanpur. 2010. Recurrent neural network based language model. In Interspeech, Vol. 2. 3.

[19] Rong Pan, Yunhong Zhou, Bin Cao, Nathan N Liu, Rajan Lukose, Martin Scholz, and Qiang Yang. 2008. One-class collaborative filtering. In International Conference on Data Mining. IEEE, 502–511.

[20] Steffen Rendle, Christoph Freudenthaler, Zeno Gantner, and Lars Schmidt-Thieme. 2009. BPR: Bayesian personalized ranking from implicit feedback. In Conference on Uncertainty in Artificial Intelligence. AUAI Press, 452–461.

[21] Steffen Rendle, Christoph Freudenthaler, and Lars Schmidt-Thieme. 2010. Factorizing personalized markov chains for next-basket recommendation. In International Conference on World Wide Web. ACM, 811–820.

[22] Ruslan Salakhutdinov and Andriy Mnih. 2007. Probabilistic Matrix Factorization. In Advances in Neural Information Processing Systems, Vol. 1. 2–1.

[23] Ruslan Salakhutdinov, Andriy Mnih, and Geoffrey Hinton. 2007. Restricted Boltzmann machines for collaborative filtering. In International Conference on Machine learning. ACM, 791–798.

[24] Badrul Sarwar, George Karypis, Joseph Konstan, and John Riedl. 2001. Item-based collaborative filtering recommendation algorithms. In International Conference on World Wide Web. ACM, 285–295.

[25] Suvash Sedhain, Aditya Krishna Menon, Scott Sanner, and Lexing Xie. 2015. Autorec: Autoencoders meet collaborative filtering. In International Conference on World Wide Web. ACM, 111–112.

[26] Yang Song, Ali Mamdouh Elkahky, and Xiaodong He. 2016. Multi-rate deep learning for temporal recommendation. In International ACM SIGIR conference on Research and Development in Information Retrieval. ACM, 909–912.

[27] Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov. 2014. Dropout: A Simple Way to Prevent Neural Networks from Overfitting. Journal of Machine Learning Research 15, 1 (2014), 1929–1958.

[28] Andrea Vedaldi and Karel Lenc. 2015. Matconvnet: Convolutional neural networks for matlab. In International conference on Multimedia. ACM, 689–692.

[29] Hao Wang, Naiyan Wang, and Dit-Yan Yeung. 2015. Collaborative deep learning for recommender systems. In International Conference on Knowledge Discovery and Data Mining. ACM, 1235–1244.

[30] Pengfei Wang, Jiafeng Guo, Yanyan Lan, Jun Xu, Shengxian Wan, and Xueqi Cheng. 2015. Learning hierarchical representation model for nextbasket recommendation. In International ACM SIGIR conference on Research and Development in Information Retrieval. ACM, 403–412.

[31] Chao-Yuan Wu, Amr Ahmed, Alex Beutel, Alexander J. Smola, and How Jing. 2017. Recurrent Recommender Networks. In International Conference on Web Search and Data Mining. ACM, 495–503.

[32] Yao Wu, Christopher DuBois, Alice X Zheng, and Martin Ester. 2016. Collaborative denoising auto-encoders for top-n recommender systems. In International Conference on Web Search and Data Mining. ACM, 153–162.

[33] Quan Yuan, Gao Cong, and Aixin Sun. 2014. Graph-based point-of-interest recommendation with geographical and temporal inﬂuences. In International Conference on Information and Knowledge Management. ACM, 659–668.

[34] Chenyi Zhang, Ke Wang, Hongkun Yu, Jianling Sun, and Ee-Peng Lim. 2014. Latent factor transition for dynamic collaborative ﬁltering. In SIAM International Conference on Data Mining. SIAM, 452–460.

[35] Shenglin Zhao, Tong Zhao, Haiqin Yang, Michael R Lyu, and Irwin King. 2016. Stellar: spatial-temporal latent ranking for successive point-of-interest recommendation. In AAAI Conference on Artiﬁcial Intelligence. AAAI Press, 315–321.

[36] Lei Zheng, Vahid Noroozi, and Philip S. Yu. 2017. Joint Deep Modeling of Users and Items Using Reviews for Recommendation. In International Conference on Web Search and Data Mining. ACM, 425–434.
