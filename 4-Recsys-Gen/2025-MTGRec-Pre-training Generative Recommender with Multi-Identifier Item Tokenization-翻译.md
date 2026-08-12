# 多标识符item标记化的预训练生成式推荐器

> bwzheng0324@ruc.edu.cn  enzeliu@ruc.edu.cn

本文介绍了 多标识符item标记化的预训练生成式推荐器。核心内容：
关键发现：
---
郑博文\*，刘恩泽\*，陈忠富，马中瑞
高瓴人工智能学院，中国人民大学，北京，中国
高瓴人工智能学院，中国人民大学，北京，中国
PoissonLab，华为，北京，中国
PoissonLab，华为，北京，中国
chenzhongfu3@huawei.com  zhongrui.ma@huawei.com
王悦，赵鑫，文继荣
PoissonLab，华为，北京，中国
高瓴人工智能学院，中国人民大学，北京，中国
wangyue262@huawei.com
batmanfly@gmail.com  jrwen@ruc.edu.cn
## 摘要
生成式推荐已成为一种有前景的范式，它通过自回归生成item的标识符来推荐潜在item。大多数现有方法采用严格的一对一映射策略，其中每个item由单个标识符表示。然而，这种僵化的标记化方案带来了问题，例如低频item的次优语义建模以及标记序列数据多样性有限。
为克服这些限制，我们提出MTGRec，它利用多标识符item标记化（Multi-identifier Item Tokenization）来增强用于生成式推荐器预训练的标记序列数据。我们的方法基于两项核心创新：多标识符item标记化和课程式推荐器预训练。对于多标识符item标记化，我们采用残差量化变分自编码器（RQ-VAE）作为标记器主干，并将相邻训练轮次的模型检查点视为语义相关的标记器。这使得每个item可以与多个标识符关联，从而将单个用户交互序列转换为多个标记序列作为不同的数据组。对于课程式推荐器预训练，我们引入了一种基于数据影响力估计的课程学习方案。具体来说，我们使用一阶梯度近似估计每个标记器数据的影响力，并在推荐器预训练期间动态调整每个数据组的采样概率。预训练后，我们使用单个标记器对模型进行微调，以确保推荐时准确的item识别。在三个公开基准数据集上的大量实验表明，MTGRec在有效性和可扩展性方面均显著优于传统和生成式推荐基线。我们的代码可在 https://github.com/RUCAIBox/MTGRec 获取。
**CCS概念：** • 信息系统 $\to$ 推荐系统。
**关键词：** 生成式推荐，item标记化
**ACM引用格式：**
Bowen Zheng\*, Enze Liu\*, Zhongfu Chen, Zhongrui Ma, Yue Wang, Wayne Xin Zhao (cid:0), and Ji-Rong Wen. 2018. Pre-training Generative Recommender with Multi-Identifier Item Tokenization. 收录于：Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '25). ACM, 纽约, 美国, 12页.
https://doi.org/XXXXXXX.XXXXXXX
---
## 1 引言
现今，序列推荐系统[10, 33]已广泛应用于各类在线平台，旨在基于用户的历史交互行为捕捉用户的个性化偏好。传统的序列推荐方法[10, 16, 36, 39]为每个item分配唯一ID，并通过近似最近邻（ANN）算法衡量用户偏好与候选item之间的相似度来预测下一个item。最近，受大语言模型（LLMs）[57]和生成式检索方法[37, 40, 43, 51]的巨大潜力驱动，若干研究提出采用生成式范式作为推荐系统中ANN的替代方案[15, 23, 27, 32, 58]。
生成式推荐的核心思想在于使用一个标记列表（即标记序列）作为item表示的标识符，而非单一的原始ID。因此，下一item预测被重新表述为一个序列到序列的问题，旨在自回归地生成目标item的标识符。
典型的生成式推荐框架由两个关键组件组成，即item标记器和生成式推荐器。item标记器被设计用于将每个item与一个蕴含语义知识的标记列表关联起来。其优点在于，item之间共享的标记反映了底层的语义相似性。现有方法利用多种技术来开发，例如共现矩阵分解[15, 27]、层次聚类[34, 45]和多级码本[29, 32, 42]。其中，残差量化变分自编码器（RQ-VAE）[53]是最常用的item标记器。最近的研究尝试通过整合协同信号[22, 42]或多行为信息[23]来进一步提高item标识符的质量。生成式推荐器用于自回归生成目标标记序列，通常采用仅解码器（如GPT[1, 30]）或编码器-解码器（如T5[31]）架构。此外，一些研究专注于通过使用双解码器[45]或引入对比学习[34]来增强生成式推荐器。
尽管取得了显著进展，以往的方法通常用单个标识符表示每个item，对item标记化采用严格的一对一映射，这导致了以下两个潜在问题。首先，标记序列数据继承了交互数据的长尾分布和数据稀疏性问题[46, 59]。因此，与长尾item关联的标记是低频的且缺乏监督信号，使得有效学习其语义变得困难。其次，一对一映射限制了序列数据的多样性。与所有可能的标记排列相比，以一对一的方式将观察到的item序列映射到标记序列会导致缺乏多样性。此外，这些限制阻碍了通过模型扩展来提升性能的潜力，正如在LLMs中所观察到的[4, 17, 57]。
鉴于这些问题，我们的想法是将一个item与多个标识符关联，这通过整合多个具有语义相关性的item标记器来实现。这种多标识符方案的优势有两方面。首先，将每个item与更多标记关联增加了标记的暴露频率并促进了item间的标记共享，从而有助于标记语义的有效学习。其次，一个item交互序列可以被标记化为多个标记序列，从而丰富了训练数据的多样性。此外，增加的标记序列数据量和多样性使我们能够通过模型扩展实现性能提升。为了开发我们的方法，我们关注两个关键挑战：(i) 学习语义相关而非无关的多个item标记器；以及 (ii) 基于所提出的多标识符item标记化有效训练生成式推荐器。
为此，我们提出了一个名为MTGRec的新框架，它整合了多个item标记器以提高生成式推荐器的有效性和可扩展性。总体而言，我们的方法将每个item与多个标识符关联，并将一个item序列扩充为多个标记序列，作为生成式推荐器预训练的训练数据。具体来说，我们关注两个关键方面，即多标识符item标记化和课程式推荐器预训练。对于多标识符item标记化，我们采用可学习的RQ-VAE作为主干，并将相邻轮次对应的模型检查点视为语义相关的item标记器。通过这种方式，我们可以将每个item标记化为多个标识符，并构建多个组的标记序列数据。每组具有来自不同item标记器的相关但不同的分布。对于课程式推荐器预训练，我们设计了一种数据课程方案，在模型预训练期间自适应地调度这些数据组的比例。作为具体技术，我们使用一阶梯度近似衡量来自每个item标记器的数据的影响力，并相应地调整每个数据组的采样概率。最后，我们使用单个item标识符微调预训练模型，以确保推荐时准确的item识别。
总之，我们的主要贡献如下：
- 我们提出了一个新颖的框架MTGRec，它学习多个item标记器用于课程式推荐器预训练，以改进生成式推荐。
- 我们开发了一种多标识符item标记化方法用于标记序列增强，并引入了一种基于数据影响力估计的数据课程方案来增强推荐器训练。
- 我们在三个公开数据集上进行了大量实验，展示了我们提出的框架在有效性和可扩展性方面优于传统和生成式推荐基线。
---
## 2 方法
在本节中，我们介绍我们提出的生成式推荐器MTGRec，它使用多标识符item标记化来增强用于生成式推荐器预训练的标记序列。
### 2.1 概述
#### 2.1.1 问题形式化
给定item集合 $V$ ，设 $S = [v_1, ..., v_t]$ 表示用户按时间顺序的历史交互item。序列推荐旨在捕捉item序列中隐含的用户偏好，并预测下一个潜在item $v_{t+1}$ 。生成式推荐将传统的序列推荐任务重新表述为一个序列到序列的问题。在这种范式中，学习一个item标记器 $T$ ，用标记序列作为其标识符来表示每个item。形式上，我们将上述过程称为item标记化，记为 $[c_1, ..., c_H] = T(v)$ ，其中 $c_h$ 表示 $v$ 的第 $h$ 个标记， $H$ 是标识符的长度。然后，交互item序列 $S$ 和目标item $v_{t+1}$ 被标记化为标记序列 $X = T(S) = [c_1^1, c_2^1, ..., c_H^{t-1}, c_H^t]$ 和 $Y = T(v_{t+1}) = [c_1^{t+1}, ..., c_H^{t+1}]$ ，其中每个item由 $H$ 个标记表示。最后，下一item预测通过自回归生成目标item的标识符（即 $Y$ ）来实现。形式上，该任务可以写作：
$$
P(Y|X) = \pro$d_{h=1}$^{H} P(c_h^{t+1} | X, c_1^{t+1}, ..., $c_{h-1}$^{t+1}). \qquad (1)
$$
#### 2.1.2 方法概述
与之前建立每个item与其标识符之间一对一映射的工作[32, 42]不同，我们的想法是将一个item与多个标识符关联，以为推荐器预训练构建更大量和更多样化的标记序列数据。为此，我们在以下两个方面做出努力：
- **多标识符item标记化（第2.2节）：** 我们采用可学习的RQ-VAE[53]作为item标记器的主干。为了获得具有语义相关性的item标记器，我们选择训练过程中相邻轮次对应的多个RQ-VAE检查点。应用这些标记器，每个item与多个标识符关联，使得单个item序列可以被标记化为多个标记序列。这些由语义相关标记器生成的标记序列，封装了相关但不同的语义知识。
- **课程式推荐器预训练（第2.3节）：** 鉴于来自不同item标记器的混合数据，我们提出一种课程学习方法，在模型预训练期间自适应地调度这些数据组的比例。具体来说，我们设计了一种数据课程方案，增加有用数据的比例同时减少低质量数据的比例。为此，我们利用一阶梯度近似来估计数据影响力，衡量数据是否"有用"，并动态调整不同数据组的采样概率。最后，我们基于单个item标识符微调预训练模型，以确保准确的item识别。
所提出方法的总体框架如图1所示。接下来，我们将介绍我们方法的细节。
---
**图1：MTGRec的总体框架，包含两项关键技术。(i) 我们利用相邻轮次的RQ-VAE检查点作为语义相关的item标记器，并将一个item序列标记化为多个标记序列。(ii) 我们提出了一种基于数据影响力估计的数据课程方案，通过一阶梯度近似实现。**
### 2.2 多标识符item标记化
如上所述，我们提出了一种多标识符方案，将一个item序列标记化为多个标记序列：(i) 采用可学习的RQ-VAE作为item标记器的主干（第2.2.1节），(ii) 从相邻轮次中选择语义相关的标记器（第2.2.2节），以及 (iii) 通过多个item标记器标记化item序列（第2.2.3节）。
#### 2.2.1 标记器主干
在实践中，我们将item标记器实现为RQ-VAE[53]，由于其在对item语义建模和缓解长度偏差方面的有效性而具有优势[34, 42]。最初，RQ-VAE将item语义嵌入 $\boldsymbol{z}$ （例如，由预训练语言模型编码的文本嵌入）作为输入，并将其编码为潜在表示 $\boldsymbol{r}$ 。然后， $\boldsymbol{r}$ 通过 $H$ 级残差量化从粗到细地被量化为序列化编码（称为标记）。每个级别的码本记为 $\mathcal{C}_h = \{\boldsymbol{e}_k^h\}_{k=1}^K$ ，其中 $\boldsymbol{e}_k^h$ 是可学习的聚类中心， $K$ 是码本大小。最后，对 $\boldsymbol{r}$ 应用残差量化：
$$
c_h = \arg\min_k \|\boldsymbol{r}_h - \boldsymbol{e}_k^h\|_2^2, \qquad (2)
$$
$$
\boldsymbol{r}_{h+1} = \boldsymbol{r}_h - \boldsymbol{e}_{c_h}^h, \qquad (3)
$$
其中 $\boldsymbol{r}_h$ 是第 $h$ 个RQ级别的残差向量，且 $\boldsymbol{r}_1 = \boldsymbol{r}$ 。此后，我们获得item量化表示 $\tilde{\boldsymbol{r}} = \sum_{h=1}^H \boldsymbol{e}_{c_h}^h$ 并将其送入解码器以重建item语义嵌入。总体而言，RQ-VAE的损失为 $\mathcal{L}_T = \mathcal{L}_{\text{recon}} + \mathcal{L}_{\text{rq}}$ ，其中 $\mathcal{L}_{\text{recon}} = \|\boldsymbol{z} - \hat{\boldsymbol{z}}\|_2^2$ ，且 $\mathcal{L}_{\text{rq}} = \sum_{h=1}^H \|\text{sg}[\boldsymbol{r}_h] - \boldsymbol{e}_{c_h}^h\|_2^2 + \beta \|\boldsymbol{r}_h - \text{sg}[\boldsymbol{e}_{c_h}^h]\|_2^2$ 。 $\hat{\boldsymbol{z}}$ 是重建的item嵌入， $\text{sg}[\cdot]$ 表示停止梯度操作。 $\beta$ 用于平衡编码器和码本之间的优化，通常设为0.25。
#### 2.2.2 语义相关的标记器
为了获得将每个item与多个标识符关联的多个item标记器，一个朴素的方法涉及训练多个RQ-VAE模型，每个模型使用不同的随机参数初始化。然而，这样独立训练得到的模型导致由它们标记化的标记序列之间是无关的。因此，由这些item标记器构建的多组标记序列数据之间缺乏关联和同质知识，甚至可能导致严重的语义冲突。
相反，我们提出将一次训练过程中相邻轮次对应的模型检查点视为多个语义相关的item标记器。这些检查点源自相同初始化参数的迭代梯度下降，确保了相邻轮次之间码本的差异保持在最小。由这些item标记器生成的标记序列封装了相关但不同的语义知识。形式上，学习到的多个语义相关item标记器写作：
$$
\mathcal{T} = \{T_1, T_2, ..., T_n\} \qquad (4)
$$
$$
= \{$T_{\phi_{N-n+1}$}, $T_{\phi_{N-n+2}$}, ..., $T_{\phi_N}$\}, \qquad (5)
$$
其中 $\mathcal{T}$ 表示一组item标记器， $n$ 是标记器的数量。 $\phi_i$ 表示第 $i$ 轮对应的RQ-VAE参数。 $N$ 表示最大轮次数。
#### 2.2.3 将一个item序列标记化为多个标记序列
利用学习到的语义相关item标记器，历史item序列 $S$ 和目标item $v_{t+1}$ 可以通过不同的标记器被标记化为多个标记序列：
$$
X^1, X^2, ..., X^n = T_1(S), T_2(S), ..., T_n(S), \qquad (6)
$$
$$
Y^1, Y^2, ..., Y^n = T_1($v_{t+1}$), T_2($v_{t+1}$), ..., T_n($v_{t+1}$), \qquad (7)
$$
其中 $X^i$ 和 $Y^i$ 分别表示由 $T_i$ 标记化的标记序列和目标item标识符。值得注意的是，我们并不直接将所有增强的标记序列用于模型预训练。原因是当 $n$ 很大时，产生的数据量会变得难以管理，并且自适应调整不同数据组的比例是不可行的。相反，我们每次仅采样一个标记序列用于模型优化，这通过多次采样大致等效于使用所有数据。在接下来的章节中，我们将详细介绍如何根据它们对数据组的采样概率进行调整。
### 2.3 课程式推荐器预训练
基于多标识符item标记化，我们获得了一个包含多组标记序列的数据混合体，从中我们选择实例用于生成式推荐器预训练。这提出了一个类似于LLM预训练中的关键挑战，即如何在预训练期间自适应地调整不同数据组的比例[57]。受LLM预训练中提出并广泛使用的数据课程[2, 49]启发，我们在MTGRec中设计了一种基于数据影响力估计的课程预训练方案。具体来说，我们通过一阶梯度近似估计与多个item标记器对应的数据影响力（第2.3.1节）。然后，我们根据估计的数据影响力动态调整不同数据组的采样概率，用于推荐器预训练（第2.3.2节）。
#### 2.3.1 估计数据影响力
为了更有效地利用来自多个item标记器的数据，我们的想法是增加有用数据的比例同时减少低质量数据的比例。为了以合理的方式衡量数据是否"有用"，我们将训练数据对验证损失的贡献定义为数据影响力[7, 28, 48]，并基于梯度信息对其进行估计。形式上，使用一阶泰勒展开，验证损失可以表示如下：
$$
\mathcal{L}($D_{val}$; \thet$a_{t+1}$) = \mathcal{L}($D_{val}$; \theta_t) + \nabla \mathcal{L}($D_{val}$; \theta_t) \cdot (\thet$a_{t+1}$ - \theta_t), \qquad (8)
$$
其中 $D_{val}$ 表示用于验证的留存数据， $\theta_t$ 是时间步 $t$ 的推荐器参数。方程的第一项表示时间步 $t$ 的验证损失，而第二项是泰勒展开中的一阶导数。那么验证损失的变化为：
$$
\mathcal{L}($D_{val}$; \thet$a_{t+1}$) - \mathcal{L}($D_{val}$; \theta_t) = \nabla \mathcal{L}($D_{val}$; \theta_t) \cdot (\thet$a_{t+1}$ - \theta_t). \qquad (9)
$$
**计算验证数据的梯度。** 具体到本文讨论的序列推荐场景，验证数据通过留一法获得。在使用不同标记器进行item标记化之后，多组标记序列数据被混合到 $D_{val}$ 中。项 $\mathcal{L}(D_{val}; \theta)$ 和 $\nabla \mathcal{L}(D_{val}; \theta)$ 分别表示所有验证数据上的平均损失和累积梯度，可以形式化为：
$$
\mathcal{L}($D_{val}$; \theta) = \frac{1}{|$D_{val}$|} \su$m_{X,Y \in D_{val}$} \mathcal{L}(X, Y; \theta), \qquad (10)
$$
$$
\nabla \mathcal{L}($D_{val}$; \theta) = \frac{1}{|$D_{val}$|} \su$m_{X,Y \in D_{val}$} \nabla \mathcal{L}(X, Y; \theta), \qquad (11)
$$
其中 $X, Y$ 表示对应于历史交互item和目标item的一对标记序列。 $\mathcal{L}(\cdot, \cdot; \theta)$ 是方程(22)中的负对数似然损失。
**计算训练数据的Adam梯度。** 由于生成式推荐器通常使用Adam优化器[18]训练，方程(8)中的参数更新 $\theta_{t+1} - \theta_t$ 可以如下计算：
$$
\thet$a_{t+1}$ - \theta_t = -\eta_t \Gamma($D_{train}$^i; \theta_t), \qquad (12)
$$
$$
\Gamma($D_{train}$^i; \theta_t) = \frac{\boldsymbol{m}_{t+1}}{\sqrt{\boldsymbol{v}_{t+1}} + \epsilon}, \qquad (13)
$$
$$
\boldsymbol{m}_{t+1} = (\beta_1 \boldsymbol{m}_t + (1 - \beta_1) \nabla \mathcal{L}($D_{train}$^i; \theta_t)) / (1 - \beta_1^t), \qquad (14)
$$
$$
\boldsymbol{v}_{t+1} = (\beta_2 \boldsymbol{v}_t + (1 - \beta_2) \nabla \mathcal{L}($D_{train}$^i; \theta_t)^2) / (1 - \beta_2^t), \qquad (15)
$$
其中 $D_{train}^i$ 表示由item标记器 $T_i$ 标记化的训练标记序列数据。 $\beta_1$ 和 $\beta_2$ 是Adam中一阶和二阶动量的超参数，通常分别设为0.9和0.999。 $\eta_t$ 是时间步 $t$ 的学习率。在我们的上下文中，我们不估计每个单独数据实例的影响力，如先前研究[7, 28, 48]中所做的那样。相反，我们将来自每个item标记器的一组数据作为一个整体进行分析。每组数据的梯度（即 $\nabla \mathcal{L}(D_{train}^i; \theta)$ ）通过与方程(11)类似的梯度累积来计算，这等效于将 $D_{train}^i$ 视为一批数据。
**计算影响力。** 基于以上分析，我们将时间步 $t$ 每个item标记器的数据影响力定义为：
$$
I(T_i; \theta_t) = \eta_t \nabla \mathcal{L}($D_{val}$; \theta_t) \cdot \Gamma($D_{train}$^i, \theta_t), \qquad (16)
$$
其中 $I(T_i; \theta_t)$ 表示与标记器 $T_i$ 关联的数据组的影响力。最后，鉴于训练过程跨越多个时间步，我们基于多个模型检查点计算累积影响力为 $\tilde{I}(T_i) = \sum_{k=1}^K I(T_i; \theta_k)$ ，其中 $\theta_k$ 指示时间步 $t_k$ 的第 $k$ 个检查点， $K$ 是检查点的总数。
#### 2.3.2 课程式预训练
在阐述了如何估计来自每个item标记器的数据的影响力之后，我们现在通过动态调整不同数据组的采样概率来制定用于模型预训练的数据课程方案。具体来说，我们将训练过程划分为多个阶段，每个阶段包含特定数量的轮次。在每个阶段结束时，我们根据最新的数据影响力更新数据采样概率，该影响力由当前模型检查点确定。形式上，给定当前模型检查点 $\theta_k$ 和前一阶段的累积数据影响力 $\tilde{I}_{k-1}(T_i)$ ，采样概率更新如下：
$$
\tilde{I}_k(T_i) = \tilde{I}_{k-1}(T_i) + I(T_i; \theta_k), \qquad (17)
$$
$$
p_i^k = \frac{e^{\tilde{I}_k(T_i) / \tau}}{\su$m_{j=1}$^n e^{\tilde{I}_k(T_j) / \tau}}, \qquad (18)
$$
其中 $\tau$ 是用于控制分布平滑度的温度系数， $p_i^k$ 是在后续阶段中item标记器 $T_i$ 的采样概率。初始时，每个数据组以等概率被采样。然后，阶段 $k+1$ 的数据采样策略定义如下：
$$
T \sim \mathcal{T} = \{T_1, T_2, ..., T_n\}, \quad P(T = T_i) = p_i^k, \qquad (19)
$$
$$
X = T(S), \quad Y = T($v_{t+1}$). \qquad (20)
$$
最后，采样的标记序列数据 $X$ 和 $Y$ 随后被送入生成式推荐器，使用负对数似然损失进行模型优化：
$$
\mathcal{L}(X, Y) = -\su$m_{h=1}$^H \log P(c_h^{t+1} | X, c_1^{t+1}, ..., $c_{h-1}$^{t+1}). \qquad (22)
$$
### 2.4 微调与推理
**用于item识别的微调。** 在实际应用中，推荐器生成的标记序列应能够识别对应的item。也就是说，item及其标识符在推荐系统内应满足一一映射。然而，在我们提出的多item标记器课程预训练过程中，无法识别item，因为可能存在多个标识符对应同一个item（即 $T_1(v), ..., T_n(v) \mapsto v$ ）。因此，我们进一步基于每个item标记器分别微调预训练的生成式推荐器，并选择具有最优验证性能的模型用于实际部署和测试。
**推理。** 我们在推理阶段的目标是从整个item集中生成 top- $K$ item以进行推荐。为此，我们采用束搜索（beam search）解码 $K$ 个标记序列，并将其映射到对应的item。与一些先前的工作[15, 42]不同，我们不引入前缀树来约束搜索过程，因为这会妨碍并行解码并降低效率。至于无效标识符，它们发生频率很低[32]，被简单地忽略。
---
## 3 实验
在本节中，我们在三个公开数据集上进行实证实验和深入分析，以证明我们提出的MTGRec的有效性。
### 3.1 实验设置
#### 3.1.1 数据集
我们评估了先前工作中使用的经过预处理的数据集[12, 59]，即"Musical Instruments"、"Industrial and Scientific"和"VideoGames"。这些数据集包含从1996年5月到2023年9月的用户评论数据。根据先前研究[32, 59]中概述的预处理步骤，我们过滤掉交互记录少于五条的低活跃度用户和item。随后，我们按用户对历史item序列进行分组，并按时间顺序排序，最大序列长度限制为20个item。预处理数据集的详细统计信息见表1。
**表1：预处理数据集的统计信息。Avg.len表示item序列的平均长度。**
| 数据集 | #用户 | #item | #交互 | 稀疏度 | 平均长度 |
|:-----------|:--------:|:--------:|:-----------:|:--------:|:-----------:|
| Instrument | 57,439 | 24,587 | 511,836 | 99.964% | 8.91 |
| Scientific | 50,985 | 25,848 | 412,947 | 99.969% | 8.10 |
| Game | 94,762 | 25,612 | 814,586 | 99.966% | 8.60 |
#### 3.1.2 基线模型
为便于全面比较，我们将基线模型分为以下两组：
**(1) 传统序列推荐模型：**
- **Caser** [39]：利用卷积神经网络捕捉用户行为序列中的空间和位置模式。
- **HGN** [24]：使用特征级别和实例级别的门控机制建模用户偏好。
- **GRU4Rec** [10]：采用GRU捕捉用户交互中的序列模式。
- **BERT4Rec** [36]：利用双向自注意力模型，以掩码预测目标进行序列建模。
- **SASRec** [16]：采用单向自注意力网络进行用户行为建模。
- **FMLP-Rec** [60]：提出一种全MLP模型，使用可学习滤波器减少噪声并建模用户偏好。
- **HSTU** [54]：将用户动作和时间戳纳入下一item预测，并提出具有显著可扩展性的层次序列转换器。注意它仍然是基于ID的方法。
- **FDSA** [55]：引入双流自注意力框架，独立建模item级别和特征级别的序列以进行推荐。
- **S3-Rec** [59]：通过利用特征-item相关性作为自监督信号来增强序列推荐模型。
**(2) 生成式推荐模型：**
- **TIGER** [32]：采用RQ-VAE将item嵌入量化为语义ID，作为item标识符，并采用生成式检索范式进行序列推荐。
- **LETTER** [42]：通过将协同和多样性正则化整合到RQ-VAE中扩展TIGER。
- **TIGER++** [32]：采用表示白化和指数移动平均（EMA）技术来增强码本学习并提高语义ID的质量。实现细节请参考第3.1.4节。
#### 3.1.3 评估设置
我们采用 top- $K$ 召回率（Recall）和归一化折损累积增益（NDCG）， $K$ 设为5和10，来评估序列推荐中的模型性能。遵循先前研究[16, 32, 59]，我们应用留一法划分训练集、验证集和测试集。对于每个用户的交互序列，他/她交互的最后一个item用作测试数据，倒数第二个item用作验证数据，所有其他item用于训练。为了进行严格的比较，我们对整个item集进行全排序评估，而不是基于采样的评估。此外，所有生成式推荐模型的自回归解码束大小设为50。
#### 3.1.4 实现细节
在本部分，我们分别介绍MTGRec中item标记器和生成式推荐器的实现细节。
**item标记器。** 根据TIGER[32]，我们利用Sentence-T5[25]将与每个item关联的文本信息编码为其语义嵌入。随后，我们学习一个具有3个大小为256的码本和一个用于冲突处理的额外码本的RQ-VAE模型。此外，引入以下三种技术来增强码本学习：(i) 应用带有表示白化的PCA[35]来提高item语义嵌入的质量。(ii) 如先前研究[23, 58]，在RQ-VAE中使用隐藏层大小为[2048, 1024, 512, 256]的更深层MLP作为编码器和解码器。码本维度设为128。(iii) 应用指数移动平均（EMA）而非梯度下降进行码本学习，这更稳定且有效[41]。我们将应用这些技术后的方法记为TIGER++，并在我们的MTGRec中使用相同的技术学习RQ-VAE。模型使用Adagrad优化器在10K轮次上优化，学习率为0.001，批大小为2048。我们选择最后 $n$ 轮的RQ-VAE检查点作为我们提出的方法中的语义相关item标记器， $n$ 在5到30之间以5为间隔调优。
**生成式推荐器。** 我们采用T5[31]作为推荐器的主干，其模型维度为128，内部维度为512，4个注意力头维度为64，并使用ReLU激活函数。我们在 $\{1,2,3,4,5,6,7,8\}$ 范围内调优模型层数 $L$ ，编码器和解码器的层数均设为 $L$ 。我们将每个GPU上的批大小设为256，并使用4个GPU在所有数据集上预训练模型200轮。至于课程预训练中每个阶段的轮次数，我们先训练60轮用于梯度特征预热，然后每20轮执行一次采样概率更新。温度系数 $\tau$ 在 $\{0.1, 0.3, 1.0, 3.0, 5.0, 10.0\}$ 中调优。AdamW优化器用于预训练和微调，学习率分别设为0.005和0.0002。此外，使用余弦调度器调整学习率。
我们基于RecBole[56]实现所有传统序列推荐模型，RecBole是一个用户友好的开源推荐系统库。为公平比较，我们将所有模型的嵌入维度设为128，并通过超参数网格搜索获得最佳性能。对于所有生成式基线模型，我们使用与我们的MTGRec相同的模型架构，并在1到8之间调优 $L$ 。
### 3.2 整体性能
我们将MTGRec与传统的和生成式的基线模型在三个公开推荐基准上进行比较。整体结果见表2。从这些结果中，我们可以发现：
对于传统序列推荐模型，FMLP-Rec和HSTU通过引入更先进的模型架构取得了比SASRec更好的结果。S3-Rec整合了辅助特征进行自监督预训练，在Game数据集上取得了出色结果。此外，FDSA在三个数据集上相比仅涉及itemID和协同信息的其他模型（即Caser、HGN、GRU4Rec、BERT4Rec、SASRec、FMLP-Rec、HSTU）展示了优越的性能。这一观察结果表明，将item文本特征作为补充信息纳入可以显著提升推荐效果。
对于生成式推荐模型，它们通常优于传统序列推荐模型，这得益于蕴含语义的item标识符和生成式范式。其中，LETTER和TIGER++比TIGER表现出更好的性能，这归因于它们在item标记器上的改进。LETTER引入了协同和多样性正则化，以整合协同信号并缓解RQ-VAE的编码分配偏差。TIGER++应用表示白化和EMA技术，在item嵌入质量和模型优化方面改进了item标记器。
最后，我们提出的MTGRec在所有情况下都始终保持最优性能，在传统和生成式基线模型上均取得了显著的改进。与先前的生成式推荐模型不同，我们引入了多个语义相关的item标记器用于标记序列增强，并设计了带有数据课程的模型预训练方法。通过在来自多个item标记器的更大、更多样化的序列数据上预训练生成式推荐器，我们显著提高了模型的可扩展性和有效性。
**表2：不同基线方法与MTGRec之间的整体性能比较。最佳和第二佳结果分别以粗体和下划线突出显示。**
| 方法 | Instrument | | | | Scientific | | | | Game | | | |
|:----------|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
| | R@5 | R@10 | N@5 | N@10 | R@5 | R@10 | N@5 | N@10 | R@5 | R@10 | N@5 | N@10 |
| Caser | 0.0241 | 0.0386 | 0.0151 | 0.0197 | 0.0159 | 0.0257 | 0.0101 | 0.0132 | 0.0330 | 0.0553 | 0.0209 | 0.0281 |
| HGN | 0.0321 | 0.0517 | 0.0202 | 0.0265 | 0.0212 | 0.0351 | 0.0131 | 0.0176 | 0.0424 | 0.0687 | 0.0271 | 0.0356 |
| GRU4Rec | 0.0324 | 0.0501 | 0.0209 | 0.0266 | 0.0202 | 0.0338 | 0.0129 | 0.0173 | 0.0499 | 0.0799 | 0.0320 | 0.0416 |
| BERT4Rec | 0.0307 | 0.0485 | 0.0195 | 0.0252 | 0.0186 | 0.0296 | 0.0119 | 0.0155 | 0.0460 | 0.0735 | 0.0298 | 0.0386 |
| SASRec | 0.0333 | 0.0523 | 0.0213 | 0.0274 | 0.0259 | 0.0412 | 0.0150 | 0.0199 | 0.0535 | 0.0847 | 0.0331 | 0.0438 |
| FMLP-Rec | 0.0339 | 0.0536 | 0.0218 | 0.0282 | 0.0269 | 0.0422 | 0.0155 | 0.0204 | 0.0528 | 0.0857 | 0.0338 | 0.0444 |
| HSTU | 0.0343 | 0.0577 | 0.0191 | 0.0271 | 0.0271 | 0.0429 | 0.0147 | 0.0198 | 0.0578 | 0.0903 | 0.0334 | 0.0442 |
| FDSA | 0.0347 | 0.0545 | 0.0230 | 0.0293 | 0.0262 | 0.0421 | 0.0169 | 0.0213 | 0.0544 | 0.0852 | 0.0361 | 0.0448 |
| S3-Rec | 0.0317 | 0.0496 | 0.0199 | 0.0257 | 0.0263 | 0.0418 | 0.0171 | 0.0219 | 0.0485 | 0.0769 | 0.0315 | 0.0406 |
| TIGER | 0.0370 | 0.0564 | 0.0244 | 0.0306 | 0.0264 | 0.0422 | 0.0175 | 0.0226 | 0.0559 | 0.0868 | 0.0366 | 0.0467 |
| LETTER | 0.0372 | 0.0580 | 0.0246 | 0.0313 | 0.0279 | 0.0435 | 0.0182 | 0.0232 | 0.0563 | 0.0877 | 0.0372 | 0.0473 |
| TIGER++ | 0.0380 | 0.0588 | 0.0249 | 0.0316 | 0.0289 | 0.0450 | 0.0190 | 0.0241 | 0.0580 | 0.0914 | 0.0377 | 0.0485 |
| **MTGRec** | **0.0413** | **0.0635** | **0.0275** | **0.0346** | **0.0322** | **0.0506** | **0.0212** | **0.0271** | **0.0621** | **0.0956** | **0.0410** | **0.0517** |
| 提升 | +8.68% | +7.99% | +10.44% | +9.49% | +11.42% | +12.44% | +11.58% | +12.45% | +7.07% | +4.60% | +8.75% | +6.60% |
### 3.3 消融研究
为了研究MTGRec中包含的各种技术的贡献，我们在Instrument和Scientific数据集上进行了消融研究，结果如表3所示。具体来说，我们将MTGRec与以下三种变体进行比较：
**(1) w/o Data curriculum**：不使用基于影响力估计的数据课程，以等概率从不同item标记器采样数据。我们可以看到这种变体在所有数据集上均表现比MTGRec差，这表明将课程学习引入生成式推荐器预训练可以有效提升性能。
**(2) w/o Relevant tokenizers**：学习使用不同随机参数初始化的多个item标记器进行item标记化，这些标记器是不相关且无关的。无关的标记器在预训练期间会导致严重的语义冲突，导致模型学习崩溃并造成显著的性能下降。这一观察结果强调了在训练过程中选择相邻轮次的RQ-VAE检查点作为语义相关item标记器的本质重要性。
**(3) w/o Pre-training**：不在来自多个语义相关item标记器的增强序列数据上进行预训练，生成式推荐器基于单个item标记器（即TIGER++）学习。结果表明，基于多个item标记器的预训练是我们框架有效性的关键要素。
**表3：我们的方法在Instrument和Scientific数据集上的消融研究。**
| 方法 | Instrument | | | | Scientific | | | |
|:--------------------|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
| | R@5 | R@10 | N@5 | N@10 | R@5 | R@10 | N@5 | N@10 |
| (0) **MTGRec** | **0.0413** | **0.0635** | **0.0275** | **0.0346** | **0.0322** | **0.0506** | **0.0212** | **0.0271** |
| (1) w/o Data curriculum | 0.0406 | 0.0618 | 0.0268 | 0.0338 | 0.0312 | 0.0487 | 0.0205 | 0.0263 |
| (2) w/o Relevant tokenizers | 0.0350 | 0.0548 | 0.0226 | 0.0290 | 0.0249 | 0.0404 | 0.0158 | 0.0208 |
| (3) w/o Pre-training | 0.0380 | 0.0571 | 0.0247 | 0.0309 | 0.0285 | 0.0443 | 0.0181 | 0.0236 |
### 3.4 进一步分析
#### 3.4.1 关于模型规模的性能比较
在我们的框架中，通过多个item标记器进行的序列增强为我们提供了更大量和更多样化的数据。大规模数据激励我们追求类似于LLMs中通过模型扩展来提升性能。因此，在本节中，我们努力探索不同模型规模对推荐性能的影响。具体来说，我们从单层开始，逐步将生成式推荐器的编码器和解码器层数增加到8层。对于不同规模的模型，我们尝试不同数量的标记器进行预训练以获得最优性能，结果如图2所示。显然，MTGRec在所有情况下均优于基线模型（即TIGER、TIGER++）。此外，基线模型的性能仅在浅层时与模型规模正相关；随着模型规模略微增加（例如4或5层），性能可能因过拟合而下降。相比之下，MTGRec的性能随模型规模总体呈上升趋势。然而，我们承认这种正相关是受限的，不像LLMs中即使模型规模达到100B仍有改进空间[4, 17, 57]。这种限制可能源于标记序列数据本质上是基于有限的观察到的用户交互构建的。由于更大的模型需要更多数据以实现有效优化，通过多个item标记器在增强数据的质量和数量之间取得平衡变得具有挑战性。具体来说，该方法在生成足够的标记序列的同时，难以保持来自间隔许多训练轮次的RQ-VAE检查点数据的语义相关性。我们将解决此类问题留给未来的工作。
**图2：关于模型规模的性能比较。x轴坐标是生成式推荐器中编码器和解码器的层数。MTGRec报告的所有结果均为使用各种数量标记器获得的最佳结果。**
（图中显示MTGRec、TIGER++和TIGER在Instrument、Scientific和Game数据集上的Recall@10对比层数的折线图，MTGRec随层数增加持续上升，而基线趋于平缓或下降。）
#### 3.4.2 关于标记器数量的性能比较
除了模型规模，我们进一步研究了用于模型预训练的item牌记器数量如何影响推荐性能。具体来说，我们实验了两种规模的生成式推荐器（即3层和6层模型），在由5到30个item标记器构建的数据集上进行预训练。如图3所示，使用较少的标记器进行预训练仅带来边际改进，这可归因于对深度模型优化而言序列数据的多样性和数量不足。此外，过多的item标记器也会导致次优性能。我们推测，当item标记器之间的间隔跨越太多轮次时，标记器之间的语义相关性会减弱甚至冲突，从而妨碍有效的模型学习。因此，MTGRec选择适当数量的item标记器以在数据量和语义相关性之间取得平衡至关重要。此外，我们观察到最佳标记器数量随模型规模增加（即对于6层模型），这表明更大的模型受益于更广泛和更多样化的序列数据进行有效训练。
**图3：在Instrument和Scientific数据集上关于标记器数量的性能比较。**
（图中显示3层和6层模型在Instrument和Scientific数据集上的Recall@10随标记器数量（5到30）变化的折线图。）
#### 3.4.3 关于温度系数的性能比较
在方程(18)中定义的不同数据组的采样概率中，温度系数 $\tau$ 用于调节分布的平滑度。为了评估 $\tau$ 的影响，我们将其值从0.1变化到10，并在图4中报告结果。结果表明，适当的 $\tau$ 可以显著提升MTGRec的性能。具体来说，在Instrument和Scientific数据集上 $\tau$ 的最优值分别为3和1。较小的 $\tau$ 使模型更倾向于高概率的item标记器，而较大的 $\tau$ 则导致数据课程退化为均匀采样。两种极端情况都不利于课程预训练的有效性。
**图4：在Instrument和Scientific数据集上关于温度系数的性能比较。**
（图中显示Instrument和Scientific数据集上NDCG@10和Recall@10随温度系数 $\tau$ （0.1到10.0）变化的折线图。）
#### 3.4.4 关于长尾item的性能比较
开发基于多个item标记器的预训练方法的关键动机之一是增强生成式推荐器的泛化能力，并防止其忽视长尾item。为了验证我们的方法在涉及长尾item的推荐中的优势，我们在具有不同交互次数的item组上评估MTGRec。具体来说，遵循先前工作[13]，我们根据目标item的流行度将测试数据划分为不同的组，并在图5中展示相对于TIGER的Recall@10改进。我们可以看到MTGRec在所有item组上始终优于基线模型。特别是当目标item不流行时，例如组[0, 20)，MTGRec展现出优越的性能和比TIGER和TIGER++更显著的改进。这一现象表明，长尾item可以从多item标记器的预训练中受益，因为这种方法提供了更多的曝光并整合了来自共享标记的更多知识。
**图5：在Instrument和Scientific数据集上关于长尾item的性能比较。柱状图显示了每个组中测试数据的交互数量，折线图显示了相对于TIGER的Recall@10提升比率。**
（图中显示了按目标item交互次数分组的测试数据分布以及各组上Recall@10相对于TIGER的提升百分比。MTGRec在低频组上提升显著。）
#### 3.4.5 将MTGRec应用于其他生成式推荐方法
此外，我们提出的方法可以无缝整合到其他生成式推荐方法中，例如原始的TIGER和LETTER，唯一的先决条件是具有可训练的item标记器。为了评估其通用适用性，我们将MTGRec应用于Instrument和Scientific数据集上的额外生成式推荐方法。如表4所示，结果表明我们提出的方法可以持续提升基础模型的性能，进一步验证了其有效性。这证实了选择相邻轮次的模型检查点作为语义相关的item标记器来生成具有跨多种方法同质知识的序列数据的方法的有效性。
**表4：在其他生成式推荐方法上的性能比较。我们的MTGRec显著提升了所有模型的性能。**
| 方法 | Instrument | | Scientific | |
|:---------------|:----------:|:----------:|:----------:|:----------:|
| | R@10 | N@10 | R@10 | N@10 |
| TIGER | 0.0568 | 0.0307 | 0.0423 | 0.0225 |
| +MTGRec | 0.0598 | 0.0329 | 0.0465 | 0.0245 |
| LETTER | 0.0580 | 0.0313 | 0.0435 | 0.0232 |
| +MTGRec | 0.0614 | 0.0335 | 0.0481 | 0.0255 |
| TIGER++ | 0.0588 | 0.0316 | 0.045 | 0.0241 |
| +MTGRec | 0.0635 | 0.0346 | 0.0506 | 0.0271 |
#### 3.4.6 多重标识符差异分析
在本节中，我们分析了由不同训练轮次的item标记器生成的item标识符的相关性和差异。具体来说，给定由两个item标记器生成的两组item标识符，我们计算两个指标：(1) 第一个标记发生变化的item比例，以及 (2) 标识符中任何标记发生变化的item比例。结果如表5所示。我们观察到，对于两个相邻的标记器（即轮次间隔为1），三个数据集上的item标识符变化最小，表明强烈的语义一致性。随着标记器之间的轮次间隔增加，更多的item标识符发生变化（例如，Game数据集上为59.95%），导致更高的语义冲突风险。值得注意的是，即使当间隔很大且许多标识符不同时，第一个标记的变化比例通常保持在1%以下，从而保留了核心语义信息。
**表5：不同间隔下的item标识符差异。**
| 间隔 | Instrument | | Scientific | | Game | |
|:--------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
| | 首个变化 | 任意变化 | 首个变化 | 任意变化 | 首个变化 | 任意变化 |
| 1 | 0.39% | 13.58% | 0.27% | 11.4% | 0.36% | 9.36% |
| 5 | 0.44% | 21.26% | 0.58% | 22.54% | 0.58% | 21.22% |
| 10 | 0.51% | 29.75% | 0.51% | 30.68% | 0.57% | 30.43% |
| 20 | 0.75% | 44.09% | 0.71% | 47.33% | 0.79% | 47.42% |
| 30 | 0.87% | 54.94% | 0.85% | 58.29% | 1.14% | 59.95% |
#### 3.4.7 效率分析
在本节中，我们进一步研究所提出方法的效率。如表6所示，我们测量了在相同设置和相同硬件环境下不同方法所需的训练时间和收敛轮次数。我们的方法MTGRec包含一个200轮的预训练阶段，随后是一个低成本的微调阶段。结果表明，与基线方法相比，我们的多标识符预训练策略没有引入过多的训练时间成本，同时取得了显著的性能提升。此外，提出的课程学习方案在Game数据集上加速了模型收敛。
**表6：不同方法的效率比较。**
| 方法 | Instrument | | Scientific | | Game | |
|:-----------|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
| | 时间 | 轮次 | 时间 | 轮次 | 时间 | 轮次 |
| TIGER | 1.33h | 186 | 1.04h | 184 | 2.19h | 253 |
| TIGER++ | 1.22h | 178 | 1.02h | 187 | 2.23h | 264 |
| MTGRec | 1.41h | 209 | 1.21h | 217 | 2.11h | 248 |
---
## 4 相关工作
在本节中，我们从两个方面回顾相关工作，即序列推荐和生成式推荐。
### 4.1 序列推荐
序列推荐[10, 16]旨在基于历史行为序列捕捉用户偏好，并预测用户最有可能交互的下一个item。早期研究[9, 33]采用马尔可夫链对item序列进行建模，并学习item之间的转换关系。随着深度学习的快速发展，深度神经网络已成为序列建模的强大工具。因此，最近的工作提出了各种基于神经网络的序列推荐器，包括卷积神经网络（CNN）[39]、循环神经网络（RNN）[10, 38]、图神经网络（GNN）[47, 50]和Transformer[16, 36]。然而，这些方法主要基于itemID和协同过滤关系开发，而忽略了嵌入在item内容（即标题、描述、类别）中的丰富信息。最近，有一些尝试[55, 59]利用与item相关的额外信息来增强ID序列建模。此外，预训练语言模型已被广泛用于将item文本特征编码为语义嵌入，以提高性能和泛化能力[11, 13, 20]。
### 4.2 生成式推荐
生成式推荐[3, 19, 26, 32, 52]，作为一种新兴且有前景的范式，在序列推荐任务上已展现出优于传统推荐系统的性能。在生成式范式中，每个item都用由一系列标记表示的标识符进行索引。这一过程称为item标记化，在生成式推荐中起着关键作用。现有的item标记化方法可以大致分为三类：启发式方法、基于文本的方法和基于码本的方法。启发式方法主要依赖于手动定义的规则或技术，如时间顺序[15]、item聚类[34, 45]和矩阵分解[15, 27]来构建item标识符。虽然这些方法易于实现，但它们往往无法捕捉item之间的隐式关系，限制了其有效性。基于文本的方法直接利用item属性，如标题、特征和描述作为标识符[5, 8, 14, 21]。这些方法通常旨在利用预训练语言模型的内部知识来提升推荐性能。然而，它们存在长度不一致、语义歧义和缺乏协同信息等问题。相比之下，基于码本的方法[6, 29, 32, 44]采用可学习的码本对item嵌入进行量化，从而构建固定长度、语义丰富的item标识符。此外，最近的研究集中于增强码本学习以更好地适应推荐系统。值得注意的例子包括引入协同和多样性正则化[42]，以及item标记器和生成式推荐器之间的对齐[22]。
回顾现有的生成式推荐方法，大多数方法在item及其标识符之间建立一一映射，这导致了诸如长尾分布、数据稀疏性和标记序列数据多样性不足等挑战。相比之下，在本文中，我们引入多个语义相关的item标记器来构建更大量和更多样化的数据用于生成式推荐器预训练，旨在提高模型的可扩展性和性能。
---
## 5 结论
在本文中，我们介绍了MTGRec，一个利用多标识符item标记化进行生成式推荐器预训练的框架。与以往在item及其标识符之间建立一一映射的方法相比，MTGRec引入了多个item标记器，将每个item与若干个标识符关联。具体来说，我们首先详细阐述了多标识符item标记化的概念，然后通过课程预训练增强了生成式推荐器。对于多标识符item标记化，我们提出使用相邻轮次对应的RQ-VAE检查点作为语义相关的item标记器。这些标记器能够将item序列数据增强为多组标记序列，每组具有相关但不同的语义分布。对于课程式推荐器预训练，我们设计了基于数据影响力估计的数据课程方案，以动态调整不同数据组的采样概率。最后，为了确保推荐时准确的item识别，我们在每个item标记器上微调预训练模型，并选择最佳模型进行部署和测试。在三个公开数据集上的大量实验和深入分析证明了我们提出的框架优于传统和生成式推荐基线。
对于未来的工作，我们将把多标识符item标记化适配到更通用的推荐场景中，例如可迁移推荐和多领域推荐。此外，我们将尝试进一步将模型参数扩展到十亿级别，并研究模型参数增加时的扩展效应。
---
## 参考文献
[1] Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. 2020. Language Models are Few-Shot Learners. 收录于：Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual.
[2] Mayee F. Chen, Nicholas Roberts, Kush Bhatia, Jue Wang, Ce Zhang, Frederic Sala, and Christopher Ré. 2023. Skill-it! A data-driven skills framework for understanding and training language models. 收录于：Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10-16, 2023.
[3] Runjin Chen, Mingxuan Ju, Ngoc Bui, Dimosthenis Antypas, Stanley Cai, Xiaopeng Wu, Leonardo Neves, Zhangyang Wang, Neil Shah, and Tong Zhao. 2024. Enhancing Item Tokenization for Generative Recommendation through Self-Improvement. arXiv:2412.17171 [cs.LG].
[4] Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, Albert Webson, Shixiang Shane Gu, Zhuyun Dai, Mirac Suzgun, Xinyun Chen, Aakanksha Chowdhery, Alex Castro-Ros, Marie Pellat, Kevin Robinson, Dasha Valter, Sharan Narang, Gaurav Mishra, Adams Yu, Vincent Y. Zhao, Yanping Huang, Andrew M. Dai, Hongkun Yu, Slav Petrov, Ed H. Chi, Jeff Dean, Jacob Devlin, Adam Roberts, Denny Zhou, Quoc V. Le, and Jason Wei. 2024. Scaling Instruction-Finetuned Language Models. J. Mach. Learn. Res. 25 (2024), 70:1–70:53.
[5] Dario Di Palma. 2023. Retrieval-augmented Recommender System: Enhancing Recommender Systems with Large Language Models. 收录于：Proceedings of the 17th ACM Conference on Recommender Systems, RecSys 2023, Singapore, Singapore, September 18-22, 2023. ACM, 1369–1373.
[6] Yijie Ding, Yupeng Hou, Jiacheng Li, and Julian McAuley. 2024. Inductive Generative Recommendation via Retrieval-based Speculation. arXiv preprint arXiv:2410.02939 (2024).
[7] Xiaochuang Han, Daniel Simig, Todor Mihaylov, Yulia Tsvetkov, Asli Celikyilmaz, and Tianlu Wang. 2023. Understanding In-Context Learning via Supportive Pretraining Data. 收录于：Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2023, Toronto, Canada, July 9-14, 2023. Association for Computational Linguistics, 12660–12673.
[8] Jesse Harte, Wouter Zorgdrager, Panos Louridas, Asterios Katsifodimos, Dietmar Jannach, and Marios Fragkoulis. 2023. Leveraging Large Language Models for Sequential Recommendation. 收录于：Proceedings of the 17th ACM Conference on Recommender Systems, RecSys 2023, Singapore, Singapore, September 18-22, 2023. ACM, 1096–1102.
[9] Ruining He and Julian J. McAuley. 2016. Fusing Similarity Models with Markov Chains for Sparse Sequential Recommendation. 收录于：IEEE 16th International Conference on Data Mining, ICDM 2016, December 12-15, 2016, Barcelona, Spain. IEEE Computer Society, 191–200.
[10] Balázs Hidasi, Alexandros Karatzoglou, Linas Baltrunas, and Domonkos Tikk. 2016. Session-based Recommendations with Recurrent Neural Networks. 收录于：4th International Conference on Learning Representations, ICLR 2016, San Juan, Puerto Rico, May 2-4, 2016, Conference Track Proceedings.
[11] Yupeng Hou, Zhankui He, Julian J. McAuley, and Wayne Xin Zhao. 2023. Learning Vector-Quantized Item Representation for Transferable Sequential Recommenders. 收录于：Proceedings of the ACM Web Conference 2023, WWW 2023, Austin, TX, USA, 30 April 2023 - 4 May 2023. ACM, 1162–1171.
[12] Yupeng Hou, Jiacheng Li, Zhankui He, An Yan, Xiusi Chen, and Julian J. McAuley. 2024. Bridging Language and Items for Retrieval and Recommendation. CoRR abs/2403.03952 (2024).
[13] Yupeng Hou, Shanlei Mu, Wayne Xin Zhao, Yaliang Li, Bolin Ding, and Ji-Rong Wen. 2022. Towards Universal Sequence Representation Learning for Recommender Systems. 收录于：KDD '22: The 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, Washington, DC, USA, August 14-18, 2022. ACM, 585–593.
[14] Yupeng Hou, Jianmo Ni, Zhankui He, Noveen Sachdeva, Wang-Cheng Kang, Ed H. Chi, Julian J. McAuley, and Derek Zhiyuan Cheng. 2025. ActionPiece: Contextually Tokenizing Action Sequences for Generative Recommendation. CoRR abs/2502.13581 (2025).
[15] Wenyue Hua, Shuyuan Xu, Yingqiang Ge, and Yongfeng Zhang. 2023. How to Index Item IDs for Recommendation Foundation Models. 收录于：Annual International ACM SIGIR Conference on Research and Development in Information Retrieval in the Asia Pacific Region, SIGIR-AP 2023, Beijing, China, November 26-28, 2023. ACM, 195–204.
[16] Wang-Cheng Kang and Julian J. McAuley. 2018. Self-Attentive Sequential Recommendation. 收录于：IEEE International Conference on Data Mining, ICDM 2018, Singapore, November 17-20, 2018. IEEE Computer Society, 197–206.
[17] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. 2020. Scaling Laws for Neural Language Models. CoRR abs/2001.08361 (2020).
[18] Diederik P. Kingma and Jimmy Ba. 2015. Adam: A Method for Stochastic Optimization. 收录于：3rd International Conference on Learning Representations, ICLR 2015, San Diego, CA, USA, May 7-9, 2015, Conference Track Proceedings.
[19] Guanghan Li, Xun Zhang, Yufei Zhang, Yifan Yin, Guojun Yin, and Wei Lin. 2024. Semantic Convergence: Harmonizing Recommender Systems via Two-Stage Alignment and Behavioral Semantic Tokenization. arXiv:2412.13771 [cs.IR].
[20] Jiacheng Li, Ming Wang, Jin Li, Jinmiao Fu, Xin Shen, Jingbo Shang, and Julian J. McAuley. 2023. Text Is All You Need: Learning Language Representations for Sequential Recommendation. 收录于：Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, KDD 2023, Long Beach, CA, USA, August 6-10, 2023. ACM, 1258–1267.
[21] Jinming Li, Wentao Zhang, Tian Wang, Guanglei Xiong, Alan Lu, and Gerard Medioni. 2023. GPT4Rec: A Generative Framework for Personalized Recommendation and User Interests Interpretation. 收录于：Proceedings of the 2023 SIGIR Workshop on eCommerce co-located with the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR 2023), Taipei, Taiwan, July 27, 2023 (CEUR Workshop Proceedings, Vol. 3589). CEUR-WS.org.
[22] Enze Liu, Bowen Zheng, Cheng Ling, Lantao Hu, Han Li, and Wayne Xin Zhao. 2024. End-to-End Learnable Item Tokenization for Generative Recommendation. CoRR abs/2409.05546 (2024).
[23] Zihan Liu, Yupeng Hou, and Julian J. McAuley. 2024. Multi-Behavior Generative Recommendation. 收录于：Proceedings of the 33rd ACM International Conference on Information and Knowledge Management, CIKM 2024, Boise, ID, USA, October 21-25, 2024. ACM, 1575–1585.
[24] Chen Ma, Peng Kang, and Xue Liu. 2019. Hierarchical Gating Networks for Sequential Recommendation. 收录于：Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, KDD 2019, Anchorage, AK, USA, August 4-8, 2019. ACM, 825–833.
[25] Jianmo Ni, Gustavo Hernández Ábrego, Noah Constant, Ji Ma, Keith B. Hall, Daniel Cer, and Yinfei Yang. 2022. Sentence-T5: Scalable Sentence Encoders from Pre-trained Text-to-Text Models. 收录于：Findings of the Association for Computational Linguistics: ACL 2022, Dublin, Ireland, May 22-27, 2022. Association for Computational Linguistics, 1864–1874.
[26] Fabian Paischer, Liu Yang, Linfeng Liu, Shuai Shao, Kaveh Hassani, Jiacheng Li, Ricky Chen, Zhang Gabriel Li, Xialo Gao, Wei Shao, Xue Feng, Nima Noorshams, Sem Park, Bo Long, and Hamid Eghbalzadeh. 2024. Preference Discerning with LLM-Enhanced Generative Retrieval. CoRR abs/2412.08604 (2024).
[27] Aleksandr V. Petrov and Craig Macdonald. 2023. Generative Sequential Recommendation with GPTRec. CoRR abs/2306.11114 (2023).
[28] Garima Pruthi, Frederick Liu, Satyen Kale, and Mukund Sundararajan. 2020. Estimating Training Data Influence by Tracing Gradient Descent. 收录于：Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual.
[29] Haohao Qu, Wenqi Fan, Zihuai Zhao, and Qing Li. 2024. TokenRec: Learning to Tokenize ID for LLM-based Generative Recommendation. CoRR abs/2406.10450 (2024).
[30] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. 2019. Language models are unsupervised multitask learners. OpenAI blog 1, 8 (2019), 9.
[31] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. 2020. Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer. J. Mach. Learn. Res. 21 (2020), 140:1–140:67.
[32] Shashank Rajput, Nikhil Mehta, Anima Singh, Raghunandan Hulikal Keshavan, Trung Vu, Lukasz Heldt, Lichan Hong, Yi Tay, Vinh Q. Tran, Jonah Samost, Maciej Kula, Ed H. Chi, and Mahesh Sathiamoorthy. 2023. Recommender Systems with Generative Retrieval. 收录于：Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10-16, 2023.
[33] Steffen Rendle, Christoph Freudenthaler, and Lars Schmidt-Thieme. 2010. Factorizing personalized Markov chains for next-basket recommendation. 收录于：Proceedings of the 19th International Conference on World Wide Web, WWW 2010, Raleigh, North Carolina, USA, April 26-30, 2010. ACM, 811–820.
[34] Zihua Si, Zhongxiang Sun, Jiale Chen, Guozhang Chen, Xiaoxue Zang, Kai Zheng, Yang Song, Xiao Zhang, and Jun Xu. 2023. Generative Retrieval with Semantic Tree-Structured Item Identifiers via Contrastive Learning. CoRR abs/2309.13375 (2023).
[35] Jianlin Su, Jiarun Cao, Weijie Liu, and Yangyiwen Ou. 2021. Whitening Sentence Representations for Better Semantics and Faster Retrieval. CoRR abs/2103.15316 (2021).
[36] Fei Sun, Jun Liu, Jian Wu, Changhua Pei, Xiao Lin, Wenwu Ou, and Peng Jiang. 2019. BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer. 收录于：Proceedings of the 28th ACM International Conference on Information and Knowledge Management, CIKM 2019, Beijing, China, November 3-7, 2019. ACM, 1441–1450.
[37] Weiwei Sun, Lingyong Yan, Zheng Chen, Shuaiqiang Wang, Haichao Zhu, Pengjie Ren, Zhumin Chen, Dawei Yin, Maarten de Rijke, and Zhaochun Ren. 2023. Learning to Tokenize for Generative Retrieval. 收录于：Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10-16, 2023.
[38] Yong Kiam Tan, Xinxing Xu, and Yong Liu. 2016. Improved Recurrent Neural Networks for Session-based Recommendations. 收录于：Proceedings of the 1st Workshop on Deep Learning for Recommender Systems, DLRS@RecSys 2016, Boston, MA, USA, September 15, 2016. ACM, 17–22.
[39] Jiaxi Tang and Ke Wang. 2018. Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding. 收录于：Proceedings of the Eleventh ACM International Conference on Web Search and Data Mining, WSDM 2018, Marina Del Rey, CA, USA, February 5-9, 2018. ACM, 565–573.
[40] Yi Tay, Vinh Tran, Mostafa Dehghani, Jianmo Ni, Dara Bahri, Harsh Mehta, Zhen Qin, Kai Hui, Zhe Zhao, Jai Prakash Gupta, Tal Schuster, William W. Cohen, and Donald Metzler. 2022. Transformer Memory as a Differentiable Search Index. 收录于：Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28-December 9, 2022.
[41] Aäron van den Oord, Oriol Vinyals, and Koray Kavukcuoglu. 2017. Neural Discrete Representation Learning. 收录于：Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, December 4-9, 2017, Long Beach, CA, USA. 6306–6315.
[42] Wenjie Wang, Honghui Bao, Xinyu Lin, Jizhi Zhang, Yongqi Li, Fuli Feng, See-Kiong Ng, and Tat-Seng Chua. 2024. Learnable Item Tokenization for Generative Recommendation. 收录于：International Conference on Information and Knowledge Management.
[43] Yujing Wang, Yingyan Hou, Haonan Wang, Ziming Miao, Shibin Wu, Qi Chen, Yuqing Xia, Chengmin Chi, Guoshuai Zhao, Zheng Liu, Xing Xie, Hao Sun, Weiwei Deng, Qi Zhang, and Mao Yang. 2022. A Neural Corpus Indexer for Document Retrieval. 收录于：Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022.
[44] Yidan Wang, Zhaochun Ren, Weiwei Sun, Jiyuan Yang, Zhixiang Liang, Xin Chen, Ruobing Xie, Su Yan, Xu Zhang, Pengjie Ren, Zhumin Chen, and Xin Xin. 2024. Enhanced Generative Recommendation via Content and Collaboration Integration. CoRR abs/2403.18480 (2024).
[45] Ye Wang, Jiahao Xun, Mingjie Hong, Jieming Zhu, Tao Jin, Wang Lin, Haoyuan Li, Linjun Li, Yan Xia, Zhou Zhao, and Zhenhua Dong. 2024. EAGER: Two-Stream Generative Recommender with Behavior-Semantic Collaboration. CoRR abs/2406.14017 (2024).
[46] Jiancan Wu, Xiang Wang, Fuli Feng, Xiangnan He, Liang Chen, Jianxun Lian, and Xing Xie. 2021. Self-supervised Graph Learning for Recommendation. 收录于：SIGIR '21: The 44th International ACM SIGIR Conference on Research and Development in Information Retrieval, Virtual Event, Canada, July 11-15, 2021. ACM, 726–735.
[47] Shu Wu, Yuyuan Tang, Yanqiao Zhu, Liang Wang, Xing Xie, and Tieniu Tan. 2019. Session-Based Recommendation with Graph Neural Networks. 收录于：The Thirty-Third AAAI Conference on Artificial Intelligence, AAAI 2019. AAAI Press, 346–353.
[48] Mengzhou Xia, Sadhika Malladi, Suchin Gururangan, Sanjeev Arora, and Danqi Chen. 2024. LESS: Selecting Influential Data for Targeted Instruction Tuning. 收录于：Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024. OpenReview.net.
[49] Canwen Xu, Corby Rosset, Luciano Del Corro, Shweti Mahajan, Julian J. McAuley, Jennifer Neville, Ahmed Hassan Awadallah, and Nikhil Rao. 2023. Contrastive Post-training Large Language Models on Data Curriculum. CoRR abs/2310.02263 (2023).
[50] Chengfeng Xu, Pengpeng Zhao, Yanchi Liu, Victor S. Sheng, Jiajie Xu, Fuzhen Zhuang, Junhua Fang, and Xiaofang Zhou. 2019. Graph Contextualized Self-Attention Network for Session-based Recommendation. 收录于：Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence, IJCAI 2019, Macao, China, August 10-16, 2019. ijcai.org, 3940–3946.
[51] Tianchi Yang, Minghui Song, Zihan Zhang, Haizhen Huang, Weiwei Deng, Feng Sun, and Qi Zhang. 2023. AutoSearchIndexer for End-to-End Document Retrieval. 收录于：Findings of the Association for Computational Linguistics: EMNLP 2023, Singapore, December 6-10, 2023. Association for Computational Linguistics, 6955–6970.
[52] Jun Yin, Zhengxin Zeng, Mingzheng Li, Hao Yan, Chaozhuo Li, Weihao Han, Jianjin Zhang, Ruochen Liu, Allen Sun, Denvy Deng, Feng Sun, Qi Zhang, Shirui Pan, and Senzhang Wang. 2024. Unleash LLMs Potential for Recommendation by Coordinating Twin-Tower Dynamic Semantic Token Generator. CoRR abs/2409.09253 (2024).
[53] Neil Zeghidour, Alejandro Luebs, Ahmed Omran, Jan Skoglund, and Marco Tagliasacchi. 2022. SoundStream: An End-to-End Neural Audio Codec. IEEE ACM Trans. Audio Speech Lang. Process. 30 (2022), 495–507.
[54] Jiaqi Zhai, Lucy Liao, Xing Liu, Yueming Wang, Rui Li, Xuan Cao, Leon Gao, Zhaojie Gong, Fangda Gu, Jiayuan He, Yinghai Lu, and Yu Shi. 2024. Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations. 收录于：Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024. OpenReview.net.
[55] Tingting Zhang, Pengpeng Zhao, Yanchi Liu, Victor S. Sheng, Jiajie Xu, Deqing Wang, Guanfeng Liu, and Xiaofang Zhou. 2019. Feature-level Deeper Self-Attention Network for Sequential Recommendation. 收录于：Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence, IJCAI 2019, Macao, China, August 10-16, 2019. ijcai.org, 4320–4326.
[56] Wayne Xin Zhao, Shanlei Mu, Yupeng Hou, Zihan Lin, Yushuo Chen, Xingyu Pan, Kaiyuan Li, Yujie Lu, Hui Wang, Changxin Tian, Yingqian Min, Zhichao Feng, Xinyan Fan, Xu Chen, Pengfei Wang, Wendi Ji, Yaliang Li, Xiaoling Wang, and Ji-Rong Wen. 2021. RecBole: Towards a Unified, Comprehensive and Efficient Framework for Recommendation Algorithms. 收录于：CIKM '21: The 30th ACM International Conference on Information and Knowledge Management, Virtual Event, Queensland, Australia, November 1-5, 2021. ACM, 4653–4664.
[57] Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, Yifan Du, Chen Yang, Yushuo Chen, Zhipeng Chen, Jinhao Jiang, Ruiyang Ren, Yifan Li, Xinyu Tang, Zikang Liu, Peiyu Liu, Jian-Yun Nie, and Ji-Rong Wen. 2023. A Survey of Large Language Models. CoRR abs/2303.18223 (2023).
[58] Bowen Zheng, Yupeng Hou, Hongyu Lu, Yu Chen, Wayne Xin Zhao, Ming Chen, and Ji-Rong Wen. 2024. Adapting Large Language Models by Integrating Collaborative Semantics for Recommendation. 收录于：40th IEEE International Conference on Data Engineering, ICDE 2024, Utrecht, The Netherlands, May 13-16, 2024. IEEE, 1435–1448.
[59] Kun Zhou, Hui Wang, Wayne Xin Zhao, Yutao Zhu, Sirui Wang, Fuzheng Zhang, Zhongyuan Wang, and Ji-Rong Wen. 2020. S3-Rec: Self-Supervised Learning for Sequential Recommendation with Mutual Information Maximization. 收录于：CIKM '20: The 29th ACM International Conference on Information and Knowledge Management, Virtual Event, Ireland, October 19-23, 2020. ACM, 1893–1902.
[60] Kun Zhou, Hui Yu, Wayne Xin Zhao, and Ji-Rong Wen. 2022. Filter-enhanced MLP is All You Need for Sequential Recommendation. 收录于：WWW '22: The ACM Web Conference 2022, Virtual Event, Lyon, France, April 25-29, 2022. ACM, 2388–2399.