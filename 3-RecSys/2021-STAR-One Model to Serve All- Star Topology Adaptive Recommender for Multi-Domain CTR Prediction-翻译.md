# STAR：面向多领域点击率预测的星型拓扑自适应推荐器

> Xiang-Rong Sheng, Liqin Zhao, Guorui Zhou, Xinyao Ding, Binding Dai, Qiang Luo, Siran Yang, Jingshan Lv, Chi Zhang, Hongbo Deng, Xiaoqiang Zhu | Alibaba Group, Beijing, China

本文介绍了 STAR（Star Topology Adaptive Recommender，星型拓扑自适应推荐器），一种通过组合所有领域共享的中心网络与每个领域特有的领域特定网络、使用单一模型同时服务多个业务领域的多领域点击率（CTR，Click-Through Rate）预测模型。核心内容：

- 提出星型拓扑自适应推荐器（STAR），将每个领域的网络分解为中心网络（共享）与领域特定网络，通过权重的逐元素相乘组合生成统一网络，实现参数高效的多领域 CTR 预测
- 提出分区归一化（PN，Partitioned Normalization），为不同领域的样本私有化归一化统计量与参数，解决多领域场景下批归一化统计量不准确的问题
- 提出辅助网络，将领域指示器直接作为 ID 特征输入并学习其嵌入，使领域信息直接且简单地对最终预测产生影响

关键发现：

- 在阿里巴巴 19 个业务领域的生产数据集上，STAR 的总体 AUC（AUC，Area Under the ROC Curve，ROC 曲线下面积）一致优于 Base、Shared Bottom、多门混合专家（MMoE，Multi-gate Mixture-of-Experts）、Cross-Stitch 等基线模型，且在所有领域上都优于 Base 模型
- 自 2020 年底 STAR 已部署于阿里巴巴展示广告系统并服务超过 60 个业务领域，在线 A/B 测试带来总体 CTR 提升 8.0%、总体千次展示收入（RPM，Revenue Per Mille）提升 6.0%

---

## 摘要

传统的工业推荐系统通常使用单一领域的数据训练模型，然后为该领域提供服务。然而，大型商业平台通常包含多个领域，其推荐系统通常需要为多个领域进行点击率（CTR，Click-Through Rate）预测。一般来说，不同领域可能共享一些共同的用户群体和item，同时每个领域可能拥有自己独特的用户群体和item。此外，即使是同一个用户，在不同领域也可能有不同的行为。为了充分利用来自不同领域的所有数据，可以训练一个单一模型来为所有领域提供服务。然而，一个单一模型很难捕捉各个领域的特征并为所有领域提供良好服务。另一方面，为每个领域单独训练一个模型又不能充分利用所有领域的数据。在本文中，我们提出了**星型拓扑自适应推荐器（STAR，Star Topology Adaptive Recommender）**模型，通过同时利用所有领域的数据来训练一个单一模型为所有领域提供服务，捕捉每个领域的特征，并建模不同领域之间的共性。本质上，每个领域的网络由两个分解网络组成：一个由所有领域共享的中心网络和一个为每个领域定制的领域特定网络。对于每个领域，我们通过将共享网络的权重与领域特定网络的权重进行逐元素相乘来组合这两个分解网络，生成一个统一的网络（尽管这两个分解网络也可以使用其他函数进行组合，这有待进一步研究）。最重要的是，STAR可以从所有数据中学习共享网络，并根据每个领域的特征自适应地调整领域特定参数。生产数据的实验结果验证了所提出的STAR模型的优越性。自2020年底以来，STAR已部署在阿里巴巴的展示广告系统中，CTR提升了8.0%，千次展示收入（RPM，Revenue Per Mille）提升了6.0%。

**关键词：** Multi-Domain Learning, Recommender System, Display Advertising

---

## 1 引言

传统的CTR预测模型[6, 13, 32, 43, 44]关注单一领域预测，即CTR模型在使用从该领域收集的样本进行训练后，为单个业务领域提供服务。每个业务领域是item在移动应用或PC网站上呈现给用户的具体位置。在像阿里巴巴和亚马逊这样的大型商业公司中，通常有许多需要CTR预测的业务领域，以提升用户满意度和提高业务收入。例如，在阿里巴巴中，业务领域范围从淘宝App首页的"猜你喜欢"、淘宝App首页的Banner到其他领域[46]。图1展示了阿里巴巴中两个代表性的业务领域。

* **Banner（横幅广告）：** 在Banner中，推荐item出现在淘宝首页的顶部横幅中。item可以是单个商品、店铺或品牌。
* **猜你喜欢：** 在猜你喜欢中，item均为单个商品，并在左侧或右侧栏中展示给用户。

由于不同业务领域拥有重叠的用户群体和item，这些领域之间存在共性。实现信息共享有助于学习每个领域的CTR模型。然而，特定的用户群体可能不同，用户在不同领域中的行为也会发生变化。这些差异导致了领域特定的数据分布。简单地混合所有数据并训练一个单一的共享CTR模型无法在所有领域上良好工作。

除了混合数据并训练共享模型，另一个简单的解决方案是为每个业务领域构建一个单独的模型。这种策略也有一些缺点：（1）某些业务领域的数据远少于其他领域。拆分数据忽略了领域的共性，导致训练数据大大减少，使模型难以学习。（2）维护多个模型会导致巨大的资源消耗，并需要更多的人力成本。当业务领域数量达到数百个时，这将变得异常繁重。本文旨在学习一个有效且高效的CTR模型来同时处理多个领域。我们将多领域CTR预测形式化为推荐器需要同时为 $M$ 个业务领域 $D_1, D_2, \ldots, D_M$ 进行CTR预测的问题。模型输入为 $(x, y, p)$ ，其中 $x$ 是多个业务领域使用的共同特征，如用户历史行为、用户画像特征、item特征和上下文特征。 $y \in \{0, 1\}$ 是点击标签， $p$ 是领域指示器，表示该样本来自哪个领域。注意 $(x, y)$ 从领域特定分布 $D_p$ 中抽取，且不同领域的分布不同。多领域CTR预测旨在构建一个有效且高效的模型，为每个领域提供准确的CTR预测，同时资源消耗成本极低。为实现这一目标，模型应充分利用领域共性并捕捉领域差异。

改进多领域学习的一个可能策略是多任务学习[5, 25, 33]。如图3所示，多领域CTR预测与多任务学习的区别在于，多领域CTR预测是在不同领域上解决相同的任务，即CTR预测，其中不同领域的标签空间相同而数据分布不同。相比之下，大多数多任务学习方法[24–27, 36]解决同一领域中的不同任务，其中标签空间可能不同，例如联合估计CTR和转化率（CVR，Conversion Rate）[26, 39]。由于任务的异质性，现有的多任务学习方法侧重于在底层共享信息，但保持独立的特定任务输出层[33]。直接将多任务方法应用于多领域CTR预测不能充分利用标签空间中的领域关系，并且忽略了不同领域的不同数据分布。

为了充分利用领域关系，我们提出了用于多领域CTR预测的**星型拓扑自适应推荐器（STAR）**。所提出的STAR模型具有星型拓扑结构，如图4所示。STAR由共享的中心参数和多组领域特定参数组成。每个领域的最终模型通过组合共享的中心参数和领域特定参数得到。中心参数用于学习所有领域中的通用行为，其中的通用知识可以在所有领域之间学习和迁移。领域特定参数捕捉不同领域中的特定行为，以促进更精细的CTR预测。星型拓扑结构促进了跨多个领域的有效信息转换，以学习领域共性和差异。

本文通过逐元素权重乘积作为组合策略实现了STAR模型。由于嵌入层贡献了工业推荐器中大部分参数，新增的领域特定参数相对于总参数量可以忽略不计。因此，使用STAR模型服务多个领域仅增加很少的计算和内存成本，同时产生更好的性能。

**本文的主要贡献总结如下：**

* 我们提出了**星型拓扑自适应推荐器（STAR）**来解决多领域CTR预测问题。星型拓扑促进了跨多个领域的有效信息转换，以学习领域共性同时捕捉领域差异。
* 不同领域具有不同的数据分布，这导致使用批归一化时产生不准确的统计量。我们提出了**分区归一化（PN，Partitioned Normalization）**，为来自不同领域的样本私有化归一化过程，以解决这一问题。PN可以在领域内产生更准确的矩，从而提升模型性能。
* 在多领域CTR预测中，描述领域信息的特征非常重要。我们提出了一个**辅助网络**，将领域指示器直接作为输入并学习其嵌入以描述领域特征。然后将这些嵌入输入到辅助网络中，该网络比原始网络简单得多。这使得领域指示器以直接的方式影响最终预测。
* 我们在工业生产数据集上评估了STAR，并于2020年将其部署在阿里巴巴的展示广告系统中。一致的优势验证了STAR的有效性。截至目前，STAR的部署带来了6%的CTR提升和8%的RPM提升。我们相信在部署中获得的经验可以推广到其他场景，因此对研究人员和工业从业者都具有参考价值。

---

## 2 相关工作

我们的工作与传统的单领域CTR预测密切相关，其中推荐器在单个业务领域上训练，然后为该业务领域提供服务。此外，我们的工作还与多任务学习和多领域学习相关。在本节中，我们进行简要介绍。

### 2.1 单领域CTR预测

受深度学习成功的启发，最近的CTR预测模型已经从传统的浅层方法[11, 19, 20, 32, 45]过渡到现代深度方法[6, 13, 28, 30, 43, 44]。大多数深度CTR模型遵循嵌入和MLP（Multi-Layer Perceptron，多层感知机）范式。Wide & Deep[6]和DeepFM[13]结合低阶和高阶特征以提高模型的表达能力。PNN（Product-based Neural Network，基于乘积的神经网络）[30]引入乘积层来捕捉跨域类别之间的交互模式。在这些模型中，用户历史行为在嵌入和池化后被转换为低维向量。DIN（Deep Interest Network，深度兴趣网络）[44]利用注意力机制相对于给定的目标item局部激活历史行为，并成功捕捉了用户兴趣的多样性特征。DIEN（Deep Interest Evolution Network，深度兴趣演化网络）[43]进一步提出了一个辅助损失，从历史行为中捕捉潜在兴趣。此外，DIEN将注意力机制与GRU（Gated Recurrent Unit，门控循环单元）集成，以建模用户兴趣的动态演化。MIND（Multi-Interest Network with Dynamic Routing，动态路由多兴趣网络）[21]和DMIN（Deep Multi-Interest Network，深度多兴趣网络）[40]认为单个向量可能不足以捕捉用户和item中的复杂模式。MIND中引入了胶囊网络和动态路由机制来学习多个表示以聚合原始特征。此外，受自注意力架构在序列到序列学习任务[37]中取得成功的启发，Transformer在[10]中被引入用于特征聚合。MIMN（Multi-channel user Interest Memory Network，多通道用户兴趣记忆网络）[28]提出了一种基于记忆的架构来聚合特征并应对长期用户兴趣建模的挑战。SIM（Search-based Interest Modeling，基于搜索的兴趣建模）[29]利用两个级联搜索单元提取用户兴趣，在可扩展性和准确性方面实现了更好的终身序列行为数据建模能力。

### 2.2 多任务学习

多任务学习（MTL，Multi-Task Learning）[5, 33]旨在通过在多个相关任务之间共享知识来改善泛化性能。利用共享知识和任务特定知识来促进每个任务的学习。多任务学习已成功应用于多个应用领域，包括自然语言处理[7]、语音识别[8]、推荐系统[42]和计算机视觉[17]。在针对线性模型的MTL早期文献中，Argyriou等人[1]提出了一种学习跨多个任务共享的稀疏表示的方法。在深度学习背景下，多任务学习通常通过隐藏层的参数共享来实现[5, 26]。Misra等人[27]提出了十字绣单元来学习每个任务的特定任务隐藏层的独特组合。Ma等人[25]提出了多门混合专家（MMoE，Multi-gate Mixture-of-Experts）模型，通过在所有任务之间共享专家子模型来建模任务关系，同时训练一个门控网络来优化每个任务。Kendall等人[17]提出了一种基于原则的多任务深度学习方法，通过考虑每个任务的同方差不确定性来权衡多个损失函数。在多任务学习中，不同任务可能相互冲突，需要权衡；优化最小化每个任务损失的加权线性组合的代理目标可能不是最优的。为解决此问题，Sener和Koltun[35]明确地将多任务学习视为多目标优化，总体目标是找到帕累托最优解。注意，[17, 35]与本文互补，可以结合以获得更好的性能。

### 2.3 多领域学习

在现实应用中，数据通常来自多个领域[9, 16, 22]。多领域学习使领域之间的知识迁移成为可能，以改善学习。因此，它与领域自适应（DA，Domain Adaptation）问题[3, 4]形成对比，后者知识迁移只是单向的，即从源领域到目标领域。Wang等人[38]提出了可迁移归一化来代替现有的归一化技术用于领域自适应，并揭示了批归一化（BN，Batch Normalization）[14]是迁移性的约束。

多领域CTR预测可以看作是多领域学习问题的一种特殊形式，其中每个领域对应一个业务领域，任务是CTR预测。与传统的多领域学习相比，我们的工作聚焦于CTR预测。所提出的模型充分利用了领域指示器，将其直接作为ID特征输入并学习其语义嵌入以促进模型学习，这一点被之前的文献所忽略。多领域学习和多任务学习的区别在于，多领域学习为多个领域进行预测，解决相同的问题（例如CTR预测），其中标签空间相同。相比之下，多任务学习侧重于解决不同的问题[41]。例如，在视频推荐领域，多任务学习问题可以是同时预测单个业务领域的CTR和期望观看时间，而多领域CTR预测是为多个业务领域（例如多个视频平台）进行CTR预测。

---

## 3 所提出的方法

在本节中，我们首先简要介绍多领域CTR预测的背景。接下来是所提出的方法——用于多领域CTR预测的星型拓扑自适应推荐器（STAR）的架构概述。然后我们详细介绍STAR，包括所提出的星型拓扑网络、分区归一化和辅助网络。

### 3.1 多领域CTR预测

在序列推荐系统中，模型输入为用户历史行为、用户画像特征、目标item特征以及其他特征（如上下文特征）。用户 $u$ 点击 item $m$ 的预测 CTR $\hat{y}$ 计算如下：

$$
\hat{y} = f(E(u_1), \ldots, E(u_i); E(m_1), \ldots, E(m_j); E(c_1), \ldots, E(c_k))
$$

其中 $\{u_1, \ldots, u_i\}$ 是用户特征集，包括用户历史行为和用户画像特征； $\{m_1, \ldots, m_j\}$ 是目标item特征集； $\{c_1, \ldots, c_k\}$ 是其他特征集。 $E(\cdot) \in \mathbb{R}^d$ 表示嵌入层，将稀疏ID映射为可学习的稠密向量。

在将原始特征映射为低维嵌入后，通常的做法是聚合这些嵌入以获得固定长度的向量。可以使用不同种类的聚合方法（如[43, 44]）来聚合这些嵌入，提取用户兴趣并获得固定长度的表示。获得的表示随后被输入到后续的深度神经网络（例如多层全连接网络）中，以得到最终的CTR预测。

传统的CTR模型[6, 13, 23, 43, 44]通常在单个业务领域的数据上训练。然而，现实世界的推荐器通常需要处理多个业务领域。具体来说，推荐器需要同时为 $M$ 个领域 $D_1, D_2, \ldots, D_M$ 进行CTR预测。模型输入为 $(x, y, p)$ ，其中 $x$ 是多个领域使用的共同特征，如用户历史行为和用户画像特征、目标item特征等。 $y \in \{0, 1\}$ 是点击标签， $p \in \{1, 2, \ldots, M\}$ 是领域指示器，表示该样本来自哪个领域。注意 $(x, y)$ 从领域特定分布 $D_p$ 中抽取，且不同领域的分布不同。多领域CTR预测的目标是构建一个单一的CTR模型，能够以低资源消耗和人力成本为所有领域提供准确的CTR预测。

### 3.2 架构概述

如上所述，忽略领域指示器 $p$ 并学习单一共享CTR模型会忽略领域差异，导致模型性能不佳。另一方面，为每个领域训练单独的模型效果更差，因为拆分领域使得每个模型的训练数据大大减少。此外，由于资源消耗和人力成本，在生产环境中为每个领域维护一个单独的模型是不可行的。

为此，我们提出了用于多领域CTR预测的**星型拓扑自适应推荐器（STAR）**，以更好地利用不同领域之间的相似性，同时捕捉领域差异。如图4所示，STAR由三个主要组件组成：（1）**分区归一化（PN，Partitioned Normalization）**，为来自不同领域的样本私有化归一化过程；（2）**星型拓扑全连接神经网络（星型拓扑FCN，Star Topology Fully-Connected Neural Network）**；（3）**辅助网络**，将领域指示器直接作为输入特征并学习其语义嵌入以捕捉领域差异。

在训练期间，首先采样一个领域指示器 $p$ ，然后从该领域采样一个包含 $B$ 个实例的小批量数据：

$$
(x_1, p), (x_2, p), \ldots, (x_B, p)
$$

STAR首先通过嵌入层将这些输入特征嵌入为低维向量。在工业推荐器中，模型通常使用数十亿特征[15]进行训练，嵌入参数通常远多于模型的其他部分。这使得不同领域难以用有限的数据学习领域特定的嵌入。例如，在我们日常任务使用的模型中，嵌入参数比全连接层的参数多10000倍[15]。因此，在提出的STAR模型中，我们让所有业务领域共享相同的嵌入层，即不同领域中的相同ID特征共享相同的嵌入。跨多个领域共享嵌入层可以显著减少计算和内存成本。

嵌入随后经过池化和拼接，获得 $B$ 个固定长度的表示。之后，这 $B$ 个提取的表示由提出的分区归一化（PN）层处理，该层为不同领域私有化归一化统计量。归一化后的向量随后被输入到提出的星型拓扑FCN中，以得到输出。星型拓扑FCN由共享的中心FCN和多个领域特定FCN组成。每个领域的最终模型通过组合共享的中心FCN和领域特定FCN得到。

在多领域CTR预测中，描述领域信息的特征非常重要。在STAR模型中，辅助网络将领域指示器作为输入，并与其他描述领域的特征一起输入到辅助网络中。辅助网络的输出与星型拓扑FCN的输出相加，得到最终预测。我们将辅助网络设计得比星型拓扑FCN简单得多，以使模型以直接且简单的方式捕捉领域差异。下面我们将详细描述这些组件。

### 3.3 分区归一化

如上所述，原始特征首先被转换为低维嵌入，然后经过池化和聚合得到中间表示。将一个实例的中间表示记为 $z$ 。为了快速稳定地训练深度网络，标准做法是对中间表示 $z$ 应用归一化层。在所有归一化方法中，批归一化（BN）[14]是一种代表性方法，被证明对于成功训练非常深的神经网络至关重要[14, 31]。BN对所有样本使用全局归一化，累积归一化统计量并在所有样本上学习共享参数。具体而言，训练期间的BN归一化可以表示为：

$$
z^{\prime} = \gamma \frac{z - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta \qquad (1)
$$

其中 $z^{\prime}$ 是输出， $\gamma$ 、 $\beta$ 是可学习的缩放和偏置参数， $\mu$ 、 $\sigma^2$ 是当前小批量数据的均值和方差。测试时，使用所有样本的移动平均统计量（均值 $E$ 和方差 $Var$ ）代替：

$$
z^{\prime} = \gamma \frac{z - E}{\sqrt{Var + \epsilon}} + \beta \qquad (2)
$$

换句话说，BN假设所有样本是独立同分布的，并使用跨所有训练样本的共享统计量。

然而，在多领域CTR预测中，样本仅在特定领域内被假设为局部独立同分布。因此，来自不同领域的数据具有不同的归一化统计量。在测试期间共享BN层的全局统计量和参数会模糊领域差异，导致模型性能下降。为了捕捉每个领域的独特数据特征，我们提出了**分区归一化（PN）**，为不同领域私有化归一化统计量和参数。具体而言，在训练期间，假设当前小批量数据来自第 $p$ 个领域，我们计算当前小批量的均值和方差，并对特征进行归一化：

$$
z^{\prime} = (\gamma \cdot \gamma_p) \frac{z - \mu}{\sqrt{\sigma^2 + \epsilon}} + (\beta + \beta_p) \qquad (3)
$$

其中 $\gamma$ 、 $\beta$ 是全局缩放和偏置， $\gamma_p$ 、 $\beta_p$ 是领域特定的缩放和偏置参数。对于每个小批量，PN通过将共享 $\gamma$ 与领域特定 $\gamma_p$ 逐元素相乘来获得最终缩放，即PN根据领域指示器自适应地缩放表示。类似地，PN的偏置也是条件于领域的自适应，通过全局偏置 $\beta$ 和领域特定偏置 $\beta_p$ 相加实现。注意，与BN相比，PN在训练期间也使用当前小批量的统计量，但PN引入了领域特定的缩放和偏置 $\gamma_p$ 、 $\beta_p$ 来捕捉领域差异。

除了修改缩放和偏置，PN还让不同领域累积领域特定的移动平均均值 $E_p$ 和方差 $Var_p$ 。在测试期间，PN对来自第 $p$ 个领域的实例 $z$ 进行如下变换：

$$
z^{\prime} = (\gamma \cdot \gamma_p) \frac{z - E_p}{\sqrt{Var_p + \epsilon}} + (\beta + \beta_p) \qquad (4)
$$

从公式4可以看出，PN使用领域特定的均值 $E_p$ 和方差 $Var_p$ 来归一化中间表示 $z$ 。因此，PN根据领域指示器自适应地改变中间表示，以捕捉独特的特征。

### 3.4 星型拓扑FCN

经过PN层后，表示 $z^{\prime}$ 被输入到后续的星型拓扑多层全连接神经网络（星型拓扑FCN）中。如图5所示，所提出的星型拓扑FCN由一个共享的中心FCN和每个领域独立的FCN组成，因此FCN总数为 $M + 1$ 。第 $p$ 个领域的最终模型通过组合共享的中心FCN和领域特定FCN得到，其中中心参数学习所有领域的通用行为，领域特定参数捕捉不同领域的特定行为以促进更精细的CTR预测。

具体来说，对于共享FCN，记 $W$ 为权重， $b$ 为神经网络层中的偏置。对于第 $p$ 个领域的特定FCN，记 $W_p$ 为权重， $b_p$ 为相应层中的偏置。设输入维度为 $c$ ，输出维度为 $d$ ，即 $W, W_p \in \mathbb{R}^{c \times d}$ ， $b, b_p \in \mathbb{R}^d$ 。第 $p$ 个领域的最终权重 $W_p^{\star}$ 和偏置 $b_p^{\star}$ 通过以下方式得到：

$$
W_p^{\star} = W_p \otimes W, \quad b_p^{\star} = b_p + b \qquad (5)
$$

其中 $\otimes$ 表示逐元素乘法。记 $in_p \in \mathbb{R}^{c \times 1}$ 为来自第 $p$ 个领域的神经网络层输入，最终输出 $out_p \in \mathbb{R}^{d \times 1}$ 由下式给出：

$$
out_p = \phi((W_p^{\star})^{\top} in_p + b_p^{\star}) \qquad (6)
$$

其中 $\phi$ 表示该层的激活函数。在所有层中均采用共享参数与领域特定参数的组合。通过这种方式，STAR可以根据领域条件调整其参数。

注意，我们通过权重的逐元素乘法和偏置的逐元素加法来实现共享中心FCN和领域特定FCN的组合策略，也可以研究其他策略以获得更好的性能。共享参数通过所有样本的梯度进行更新，而领域特定参数仅通过该领域内的样本进行更新。这有助于在通过共享中心参数学习领域共性的同时，捕捉领域差异以实现更精细的CTR预测。如上所述，工业推荐器中的大部分参数由嵌入层贡献，增加的 $M$ 个FCN相对于总参数量可以忽略不计。因此，STAR以参数高效和内存友好的方式使用一个模型有效地为所有业务领域提供服务。

### 3.5 辅助网络

在传统的CTR建模中，所有特征被平等对待并输入到复杂的模型中。然而，在多领域CTR预测中，模型可能难以自动学习领域差异。我们认为一个好的多领域CTR模型应具有以下特点：（1）具有关于领域特征的信息丰富的特征；（2）使这些特征能够轻松且直接地影响最终的CTR预测。其直觉是描述领域信息的特征非常重要，因为它们可以降低模型捕捉领域间差异的难度。

为此，我们提出了一个**辅助网络**来学习领域差异。为了增强关于领域特征的丰富信息，我们将领域指示器直接作为ID特征输入处理。领域指示器首先被映射为嵌入向量，然后与其他特征拼接。辅助网络随后对拼接后的特征进行前向计算，得到一维输出。将星型拓扑FCN的一维输出记为 $s_m$ ，辅助网络的输出记为 $s_a$ 。 $s_m$ 和 $s_a$ 相加得到最终的logit。然后应用Sigmoid函数得到CTR预测：

$$
\mathrm{Sigmoid}(s_m + s_a) \qquad (7)
$$

在我们的实现中，辅助网络比主网络简单得多，是一个两层全连接神经网络。这种简单的架构使领域特征能够直接影响最终预测。

记 $\hat{y}_i^p$ 为第 $p$ 个领域中第 $i$ 个实例的预测概率， $y_i^p \in \{0, 1\}$ 为真实标签。我们最小化所有领域上的交叉熵损失函数：

$$
\min \sum_{p=1}^{M} \sum_{i=1}^{N_p} \left[ -y_i^p \log(\hat{y}_i^p) - (1 - y_i^p) \log(1 - \hat{y}_i^p) \right] \qquad (8)
$$

---

## 4 实验

在本节中，我们评估STAR的有效性。我们首先在4.1节中介绍实验设置，包括使用的生产数据集、比较方法和实现细节。结果和讨论在4.2节中详细阐述。我们还在4.3节中进行了深入的消融研究。生产环境中的实验结果在4.4节中展示。

### 4.1 实验设置

**数据集。** 由于缺乏公开的多领域CTR预测数据集，我们使用阿里巴巴在19个业务领域上的用户点击行为生产数据进行离线评估。训练数据来自阿里巴巴在线展示广告系统的流量日志。使用来自19个业务领域的一天数据进行训练，次日的数据用于测试。训练数据集包含数十亿个样本。表1显示了训练集中每个领域的样本百分比和平均CTR（点击量/展示量，即正样本比例）。如表1所示，不同领域具有不同的领域特定数据分布，这可以从不同的CTR中反映出来。可以看出，CTR最高的领域（领域#15）为12.03%，而CTR最低的领域（领域#13）仅为1.27%。在该数据集中，大多数item在大多数业务领域中可用，但只有部分用户重叠，例如领域#1和领域#2具有相同的item集，但只有8.52%的用户重叠。

**表1：每个领域的样本百分比和平均点击率（CTR）。**
| 领域 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 |
|------|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 百分比 | 0.99% | 1.61% | 3.40% | 3.85% | 2.79% | 0.56% | 4.27% | 16.76% | 10.00% | 12.16% | 0.76% | 1.31% | 3.34% | 28.76% | 1.17% | 0.46% | 1.05% | 0.91% | 5.85% |
| CTR | 2.14% | 2.69% | 2.97% | 3.63% | 2.77% | 3.45% | 3.59% | 3.24% | 3.23% | 2.08% | 12.05% | 3.52% | 1.27% | 3.75% | 12.03% | 4.02% | 1.63% | 4.64% | 1.42% |

**比较模型。** 为了验证所提出方法的有效性，我们将STAR与以下模型进行比较：

* **Base（基础模型）。** 我们将Base称为由嵌入层、池化与拼接层、批归一化和7层全连接网络组成的模型。具体而言，池化与拼接层基于DIEN[43]，在嵌入层后提取用户兴趣。我们混合来自不同领域的所有样本并训练基础模型。
* **Shared Bottom（共享底层）。** Shared Bottom模型是一种多任务模型，共享底层的参数。在我们的实现中，Shared Bottom共享嵌入层。每个领域还有一个不共享的7层全连接网络。
* **MulANN。** MulANN[34]为基础模型添加了领域判别器模块。领域判别器对样本来自哪个领域进行分类。MulANN采用对抗性损失使领域判别器无法区分领域之间的偏移。
* **MMoE。** MMoE[25]隐式建模多任务学习的任务关系，其中不同任务可能具有不同的标签空间。这里我们将MMoE适配到多领域CTR预测，其中每个专家是一个7层全连接网络。专家数量等于领域数量。此外，MMoE还学习了每个领域的门控网络，该网络接收输入特征并输出softmax门控，以不同权重组合专家。
* **Cross-Stitch（十字绣）。** Cross-Stitch[27]使用线性十字绣单元学习任务特定表示的最优组合。在Cross-Stitch方法中，每个域有一个7层全连接网络，在每个隐藏层中添加十字绣单元以学习任务特定表示。

为了保证公平比较，所有比较方法和STAR模型均在4.2节中使用提出的辅助网络进行训练。关于辅助网络的消融研究在4.3节中进行。

**实现细节。** 所有模型均使用Adam[18]优化器进行训练，学习率设置为0.001，批量大小为2000。我们最小化来自所有领域样本的交叉熵损失来训练模型。

**评估指标。** ROC曲线下面积（AUC，Area Under the ROC Curve）是评估CTR预测性能的常用指标。用户加权AUC[44]通过计算用户AUC的平均值来衡量用户内部排序的质量，在推荐系统中被证明与在线性能更相关。其计算方式如下：

$$
\mathrm{AUC} = \frac{\sum_{i} \#\mathrm{impression}_i \times \mathrm{AUC}_i}{\sum_{i} \#\mathrm{impression}_i} \qquad (9)
$$

其中 $n$ 是用户数， $\#\mathrm{impression}_i$ 和 $\mathrm{AUC}_i$ 分别是第 $i$ 个用户的展示数量和AUC。我们使用这个加权AUC作为评估指标，为简洁起见仍称其为AUC。具体地，我们使用每个领域的AUC和总体AUC（混合所有领域的样本计算总体AUC）作为指标。

### 4.2 结果

我们在阿里巴巴生产数据集上评估了所有方法。为了保证公平比较，所有比较方法和STAR模型均使用提出的辅助网络进行训练。如表2所示，一致的改进验证了STAR的有效性。注意，MulANN的性能比Base模型差，这证明了模糊领域差异会损害多领域CTR预测的建模。此外，Shared Bottom、MMoE、Cross-Stitch和STAR都取得了比Base模型更好的整体性能。这证明了利用领域关系和捕捉领域差异对于提升预测性能的重要性。

**表2：不同方法在阿里巴巴离线生产数据集上的结果。**
| 领域 | Base | Shared Bottom | MulANN | MMoE | Cross-Stitch | STAR |
|------|------|---------------|--------|------|--------------|------|
| #1 | 0.6134 | 0.6186 | 0.6143 | 0.6143 | 0.6183 | 0.6306 |
| #2 | 0.6321 | 0.6320 | 0.6321 | 0.6355 | 0.6337 | 0.6417 |
| #3 | 0.6281 | 0.6293 | 0.6282 | 0.6311 | 0.6307 | 0.6372 |
| #4 | 0.6326 | 0.6361 | 0.6333 | 0.6373 | 0.6372 | 0.6451 |
| #5 | 0.6308 | 0.6292 | 0.6302 | 0.6336 | 0.6322 | 0.6388 |
| #6 | 0.6378 | 0.6383 | 0.6336 | 0.6412 | 0.6368 | 0.6494 |
| #7 | 0.6305 | 0.6329 | 0.6310 | 0.6340 | 0.6352 | 0.6410 |
| #8 | 0.6297 | 0.6278 | 0.6297 | 0.6330 | 0.6328 | 0.6411 |
| #9 | 0.6264 | 0.6283 | 0.6258 | 0.6292 | 0.6278 | 0.6368 |
| #10 | 0.6392 | 0.6434 | 0.6375 | 0.6431 | 0.6278 | 0.6577 |
| #11 | 0.6469 | 0.6529 | 0.6445 | 0.6508 | 0.6548 | 0.6719 |
| #12 | 0.6506 | 0.6575 | 0.6498 | 0.6518 | 0.6570 | 0.6676 |
| #13 | 0.6558 | 0.6612 | 0.6538 | 0.6603 | 0.6637 | 0.6739 |
| #14 | 0.6362 | 0.6405 | 0.6371 | 0.6412 | 0.6411 | 0.6486 |
| #15 | 0.6745 | 0.6888 | 0.6710 | 0.6787 | 0.6819 | 0.7021 |
| #16 | 0.6638 | 0.6627 | 0.6517 | 0.6634 | 0.6727 | 0.6901 |
| #17 | 0.6524 | 0.6658 | 0.6499 | 0.6519 | 0.6575 | 0.6715 |
| #18 | 0.6493 | 0.6480 | 0.6375 | 0.6500 | 0.6610 | 0.6754 |
| #19 | 0.6330 | 0.6375 | 0.6306 | 0.6374 | 0.6381 | 0.6476 |
| **总体AUC** | **0.6364** | **0.6398** | **0.6353** | **0.6403** | **0.6415** | **0.6506** |

尽管Shared Bottom、MMoE和Cross-Stitch取得了比Base模型更好的整体性能，但值得注意的是，在某些领域（如#5、#6和#16）中，Shared Bottom、MMoE和Cross-Stitch的AUC低于Base模型。我们假设这是因为这些模型在不同领域中的学习存在冲突。相比之下，STAR通过其星型拓扑避免了这一问题，其中领域特定参数仅通过该领域内的样本进行更新。所提出的STAR模型在所有领域上都表现出优于Base模型的性能。STAR相比Shared Bottom也取得了一致的改进，这证明了在顶层（所有领域共享相同标签空间的情况下）进行信息共享对多领域学习的重要性。STAR还优于MMoE和Cross-Stitch，这表明与通过门控网络或十字绣单元隐式建模领域关系相比，显式建模领域关系具有优越性。

### 4.3 消融研究

为了研究每个组件的影响，我们进行了多项消融研究。

**表3：分区归一化（PN）和星型拓扑全连接神经网络（STAR FCN）的消融研究。所有模型均使用提出的辅助网络进行训练。**
| 模型 | 总体AUC |
|------|---------|
| Base (BN) | 0.6364 |
| Base (PN) | 0.6485 |
| STAR FCN (BN) | 0.6455 |
| STAR FCN (PN) | 0.6506 |

**表4：多领域CTR预测中归一化方法的消融研究。STAR FCN分别使用BN、LN和PN进行训练。**
| 模型 | 总体AUC |
|------|---------|
| STAR FCN (BN) | 0.6455 |
| STAR FCN (LN) | 0.6463 |
| STAR FCN (PN) | 0.6506 |

#### 4.3.1 星型拓扑FCN和PN

我们分析了STAR不同组件的影响。具体来说，研究了星型拓扑FCN和PN的各自效果。我们比较了（a）使用BN训练的Base模型、（b）使用PN训练的Base模型、（c）使用BN的STAR FCN以及（d）STAR模型（STAR FCN + PN）。结果报告在表3中。我们观察到单独使用星型拓扑FCN和PN都可以优于Base模型。将两者结合可以进一步提升性能。结果验证了星型拓扑FCN和PN的效果。

#### 4.3.2 归一化方法

归一化方法是深度学习中非常有效的组件，许多实践表明它们可以简化优化并使非常深的网络能够收敛。我们分析了不同归一化方法的效果，包括批归一化（BN）[14]、层归一化（LN，Layer Normalization）[2]和所提出的分区归一化（PN）在多领域CTR预测上的表现。BN累积所有领域样本的全局统计量并学习全局参数。LN是一种代表性的基于实例的归一化方法，沿通道维度操作，避免了混合来自不同领域的样本的统计量。

结果如表4所示。我们首先观察到LN和PN均优于BN。这一观察验证了不同领域的数据具有不同的分布，需要特定的归一化。使用全局归一化模糊了领域差异，会损害多领域CTR预测的性能。我们还观察到PN优于LN，这验证了领域特定归一化优于实例特定归一化，因为PN可以在领域内产生更准确的矩。

#### 4.3.3 辅助网络

我们进行了实验来评估辅助网络对不同模型的效果。所有方法分别在有和没有提出的辅助网络的条件下进行训练。结果如图6所示。我们观察到辅助网络一致地提升了所有方法。结果验证了充分利用领域特征并使用它来捕捉领域差异的重要性。我们还观察到MulANN的辅助网络改进略弱于其他方法。原因可能在于模糊领域差异的对抗性损失与捕捉领域差异的领域特征相矛盾。

#### 4.3.4 捕捉领域差异的能力

按点击付费（CPC，Cost Per Click）是展示广告中广泛使用的基于性能的付费模式，广告主对点击进行竞价。在CPC中，展示系统将有效千次展示成本（eCPM，effective Cost Per Mille）计算为出价乘以CTR。系统按照eCPM的降序分配展示。在CPC中，CTR模型需要良好校准[12]以实现具有竞争力的广告系统，即预测CTR应尽可能接近实际CTR。

我们展示了STAR更好地校准且能够捕捉领域差异。我们计算了每个领域的预测CTR与实际CTR之比（PCOC，Predicted CTR over CTR）。注意，PCOC越接近1.0，CTR预测越准确。为简化说明，我们在图7中展示了Base模型和STAR的PCOC。可以看出，与Base模型相比，STAR在不同领域的PCOC更紧凑且集中在1.0附近。结果验证了STAR捕捉领域差异的能力。

### 4.4 生产环境

**在线服务与挑战。** 工业推荐器中的挑战之一是特征和CTR的分布随时间发生较大偏移。为了实时捕捉数据的动态变化，使用实时样本持续更新CTR模型以防止其过时非常重要。然而，对于多领域CTR预测，每个领域的样本比例会随时间变化。例如，某些业务领域在早晨出现流量高峰，而另一些在晚上出现流量高峰。如果我们直接按时间顺序训练模型，数据比例随时间的变化会导致模型学习不稳定。为解决此问题，我们重新设计了数据管道，并维护了一个缓冲区，存储历史样本的滑动窗口，以避免样本百分比的突然变化。具体来说，首先对缓冲区中的样本进行洗牌，然后采样以构建一个小批量。这些样本在输入模型后从缓冲区中移除，并将新到达的数据添加到缓冲区中。我们凭经验发现，这种训练方式比传统的在线更新方式更稳定。

注意，在服务期间，每个领域的FCN权重被预先计算以实现更快的推理。通过这种方式，STAR的计算时间与Shared Bottom模型相同。这种系统优化使STAR能够稳定地为多个业务领域的主要流量提供服务。自2020年以来，STAR已在阿里巴巴展示广告系统中部署并为超过60个业务领域提供服务。我们计算了所有领域的整体改进。表5显示了STAR相对于之前的生产模型（Base模型）的改进。STAR的引入在我们的在线A/B测试中带来了总体CTR提升8.0%和总体RPM提升6.0%。

**表5：阿里巴巴在线展示广告系统中的CTR和RPM增益。**
| | CTR | RPM |
|---|-----|-----|
| 总体 | +8.0% | +6.0% |

---

## 5 结论

在本文中，我们提出了**星型拓扑自适应推荐器（STAR）**来解决多领域CTR预测问题。不同于为不同领域保持独立的模型或简单地混合所有样本并维护一个共享模型，STAR采用星型拓扑结构，由共享的中心参数和领域特定参数组成。共享参数通过所有样本更新，学习共性；领域特定参数使用特定领域内的样本学习，捕捉领域差异以实现更精细的预测。通过这种方式，STAR可以根据领域自适应地调整其参数，实现更精细的预测。实验证明了STAR在多领域CTR预测上的优越性。自2020年以来，STAR已部署在阿里巴巴的广告系统中，CTR提升8.0%，RPM提升6.0%。

---

## 参考文献

[1] Andreas Argyriou, Theodoros Evgeniou, and Massimiliano Pontil. 2008. Convex multi-task feature learning. Machine Learning 73, 3 (2008), 243–272.

[2] Lei Jimmy Ba, Jamie Ryan Kiros, and Geoffrey E. Hinton. 2016. Layer Normalization. CoRR abs/1607.06450 (2016).

[3] Shai Ben-David, John Blitzer, Koby Crammer, Alex Kulesza, Fernando Pereira, and Jennifer Wortman Vaughan. 2010. A theory of learning from different domains. Machine Learning 79, 1-2 (2010), 151–175.

[4] Steffen Bickel, Michael Brückner, and Tobias Scheffer. 2007. Discriminative learning for differing training and test distributions. In Proceedings of the 24th International Conference on Machine Learning, Vol. 227. 81–88.

[5] Rich Caruana. 1998. Multitask Learning. In Learning to Learn. 95–133.

[6] Heng-Tze Cheng, Levent Koc, Jeremiah Harmsen, Tal Shaked, Tushar Chandra, Hrishi Aradhye, Glen Anderson, Greg Corrado, Wei Chai, Mustafa Ispir, et al. 2016. Wide & deep learning for recommender systems. In Proceedings of the 1st Workshop on Deep Learning for Recommender Systems. ACM, 7–10.

[7] Ronan Collobert and Jason Weston. 2008. A Unified Architecture for Natural Language Processing: Deep Neural Networks with Multitask Learning. In Proceedings of the 25-th International Conference on Machine Learning, Vol. 307. 160–167.

[8] Li Deng, Geoffrey E. Hinton, and Brian Kingsbury. 2013. New Types of Deep Neural Network Learning for Speech Recognition and Related Applications: An Overview. In Proceedings of the 2013 IEEE International Conference on Acoustics, Speech and Signal Processing. 8599–8603.

[9] Mark Dredze, Alex Kulesza, and Koby Crammer. 2010. Multi-Domain Learning by Confidence-Weighted Parameter Combination. Machine Learning 79, 1-2 (2010), 123–149.

[10] Yufei Feng, Fuyu Lv, Weichen Shen, Menghan Wang, Fei Sun, Yu Zhu, and Keping Yang. 2019. Deep Session Interest Network for Click-Through Rate Prediction. In Proceedings of the 28th International Joint Conference on Artificial Intelligence. 2301–2307.

[11] Jerome H Friedman. 2001. Greedy function approximation: a gradient boosting machine. Annals of statistics (2001), 1189–1232.

[12] Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger. 2017. On Calibration of Modern Neural Networks. In Proceedings of the 34th International Conference on Machine Learning, Vol. 70. 1321–1330.

[13] Huifeng Guo, Ruiming Tang, Yunming Ye, Zhenguo Li, and Xiuqiang He. 2017. DeepFM: a factorization-machine based neural network for CTR prediction. In Proceedings of the 26th International Joint Conference on Artificial Intelligence. 2782–2788.

[14] Sergey Ioffe and Christian Szegedy. 2015. Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. In Proceedings of the 32nd International Conference on Machine Learning, Vol. 37. 448–456.

[15] Biye Jiang, Chao Deng, Huimin Yi, Zelin Hu, Guorui Zhou, Yang Zheng, Sui Huang, Xinyang Guo, Dongyue Wang, Yue Song, et al. 2019. XDL: An Industrial Deep Learning Framework for High-Dimensional Sparse Data. In Proceedings of the 1st International Workshop on Deep Learning Practice for High-Dimensional Sparse Data. 1–9.

[16] Mahesh Joshi, Mark Dredze, William W. Cohen, and Carolyn Penstein Rosé. 2012. Multi-Domain Learning: When Do Domains Matter?. In Proceedings of the 2012 Joint Conference on Empirical Methods in Natural Language Processing and Computational Natural Language Learning. 1302–1312.

[17] Alex Kendall, Yarin Gal, and Roberto Cipolla. 2018. Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. 7482–7491.

[18] Diederik P. Kingma and Jimmy Ba. 2015. Adam: A Method for Stochastic Optimization. In Proceedings of the 3rd International Conference on Learning Representations.

[19] Yehuda Koren. 2008. Factorization meets the neighborhood: a multifaceted collaborative filtering model. In Proceedings of the 14th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. 426–434.

[20] Yehuda Koren, Robert M. Bell, and Chris Volinsky. 2009. Matrix Factorization Techniques for Recommender Systems. IEEE Computer 42, 8 (2009), 30–37.

[21] Chao Li, Zhiyuan Liu, Mengmeng Wu, Yuchi Xu, Huan Zhao, Pipei Huang, Guoliang Kang, Qiwei Chen, Wei Li, and Dik Lun Lee. 2019. Multi-Interest Network with Dynamic Routing for Recommendation at Tmall. In Proceedings of the 28th ACM International Conference on Information and Knowledge Management. 2615–2623.

[22] Pengcheng Li, Runze Li, Qing Da, Anxiang Zeng, and Lijun Zhang. 2020. Improving Multi-Scenario Learning to Rank in E-commerce by Exploiting Task Relationships in the Label Space. In Proceedings of The 29th ACM International Conference on Information and Knowledge Management. 2605–2612.

[23] Jianxun Lian, Xiaohuan Zhou, Fuzheng Zhang, Zhongxia Chen, Xing Xie, and Guangzhong Sun. 2018. xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining.

[24] Jiaqi Ma, Zhe Zhao, Jilin Chen, Ang Li, Lichan Hong, and Ed H. Chi. 2019. SNR: Sub-Network Routing for Flexible Parameter Sharing in Multi-Task Learning. In Proceedings of The 33rd AAAI Conference on Artificial Intelligence. 216–223.

[25] Jiaqi Ma, Zhe Zhao, Xinyang Yi, Jilin Chen, Lichan Hong, and Ed H. Chi. 2018. Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 1930–1939.

[26] Xiao Ma, Liqin Zhao, Guan Huang, Zhi Wang, Zelin Hu, Xiaoqiang Zhu, and Kun Gai. 2018. Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate. In The 41st International ACM SIGIR Conference on Research & Development in Information Retrieval. 1137–1140.

[27] Ishan Misra, Abhinav Shrivastava, Abhinav Gupta, and Martial Hebert. 2016. Cross-Stitch Networks for Multi-task Learning. 3994–4003.

[28] Qi Pi, Weijie Bian, Guorui Zhou, Xiaoqiang Zhu, and Kun Gai. 2019. Practice on Long Sequential User Behavior Modeling for Click-through Rate Prediction. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 1059–1068.

[29] Qi Pi, Guorui Zhou, Yujing Zhang, Zhe Wang, Lejian Ren, Ying Fan, Xiaoqiang Zhu, and Kun Gai. 2020. Search-based User Interest Modeling with Lifelong Sequential Behavior Data for Click-Through Rate Prediction. In Proceeding of The 29th ACM International Conference on Information and Knowledge Management. 2685–2692.

[30] Yanru Qu, Han Cai, Kan Ren, Weinan Zhang, Yong Yu, Ying Wen, and Jun Wang. 2016. Product-based neural networks for user response prediction. In Proceedings of the 16th International Conference on Data Mining. IEEE, 1149–1154.

[31] Alec Radford, Luke Metz, and Soumith Chintala. 2016. Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks. In Proceedings of the 4th International Conference on Learning Representations.

[32] Steffen Rendle. 2010. Factorization machines. In Proceedings of the 10th International Conference on Data Mining. IEEE, 995–1000.

[33] Sebastian Ruder. 2017. An Overview of Multi-Task Learning in Deep Neural Networks. CoRR abs/1706.05098 (2017).

[34] Alice Schoenauer Sebag, Louise Heinrich, Marc Schoenauer, Michèle Sebag, Lani F. Wu, and Steven J. Altschuler. 2019. Multi-Domain Adversarial Learning. In Proceedings of the 7th International Conference on Learning Representations.

[35] Ozan Sener and Vladlen Koltun. 2018. Multi-Task Learning as Multi-Objective Optimization. In Advances in Neural Information Processing Systems 31. 525–536.

[36] Hongyan Tang, Junning Liu, Ming Zhao, and Xudong Gong. 2020. Progressive Layered Extraction (PLE): A Novel Multi-Task Learning (MTL) Model for Personalized Recommendations. In Proceedings of the 14th ACM Conference on Recommender Systems. 269–278.

[37] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is All you Need. In Advances in Neural Information Processing Systems 30. 5998–6008.

[38] Ximei Wang, Ying Jin, Mingsheng Long, Jianmin Wang, and Michael I. Jordan. 2019. Transferable Normalization: Towards Improving Transferability of Deep Neural Networks. In Advances in Neural Information Processing Systems 32. 1951–1961.

[39] Hong Wen, Jing Zhang, Yuan Wang, Fuyu Lv, Wentian Bao, Quan Lin, and Keping Yang. 2020. Entire Space Multi-Task Modeling via Post-Click Behavior Decomposition for Conversion Rate Prediction. In Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval. 2377–2386.

[40] Zhibo Xiao, Luwei Yang, Wen Jiang, Yi Wei, Yi Hu, and Hao Wang. 2020. Deep Multi-Interest Network for Click-through Rate Prediction. In Proceedings of the 29th ACM International Conference on Information and Knowledge Management. 2265–2268.

[41] Yongxin Yang and Timothy M. Hospedales. 2015. A Unified Perspective on Multi-Domain and Multi-Task Learning. In Proceeding of the 3rd International Conference on Learning Representations.

[42] Fajie Yuan, Guoxiao Zhang, Alexandros Karatzoglou, Xiangnan He, Joemon Jose, Beibei Kong, and Yudong Li. 2020. One Person, One Model, One World: Learning Continual User Representation without Forgetting. CoRR abs/2009.13724 (2020).

[43] Guorui Zhou, Na Mou, Ying Fan, Qi Pi, Weijie Bian, Chang Zhou, Xiaoqiang Zhu, and Kun Gai. 2019. Deep Interest Evolution Network for Click-Through Rate Prediction. In Proceedings of the 33rd AAAI Conference on Artificial Intelligence. 5941–5948.

[44] Guorui Zhou, Xiaoqiang Zhu, Chenru Song, Ying Fan, Han Zhu, Xiao Ma, Yanghui Yan, Junqi Jin, Han Li, and Kun Gai. 2018. Deep interest network for click-through rate prediction. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 1059–1068.

[45] Yunhong Zhou, Dennis Wilkinson, Robert Schreiber, and Rong Pan. 2008. Large-scale parallel collaborative filtering for the netflix prize. In Proceedings of the International Conference on Algorithmic Applications in Management. Springer, 337–348.

[46] Han Zhu, Junqi Jin, Chang Tan, Fei Pan, Yifan Zeng, Han Li, and Kun Gai. 2017. Optimized Cost per Click in Taobao Display Advertising. In Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. 2191–2200.
