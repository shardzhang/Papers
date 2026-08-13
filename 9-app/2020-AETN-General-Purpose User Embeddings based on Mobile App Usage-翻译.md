# AETN: 基于移动应用使用的通用用户嵌入

> Junqi Zhang¹\*、Bing Bai¹\*、Ye Lin¹、Jian Liang¹、Kun Bai¹、Fei Wang² | ¹腾讯云与智慧产业事业群、²美国康奈尔大学
>
> \*两位作者对本研究贡献相同。



本文提出了一种 基于移动应用 **使用行为** 的**无监督用户建模方案** AETN（AutoEncoder-coupled Transformer Network），将用户的应用**保留、安装、卸载三种异构行为统一建模为通用用户嵌入**。核心发现是——**AETN在腾讯多个下游业务中相比基线模型实现2%~8%的线上指标提升，且训练成本可控、已大规模部署**。

核心内容：

- 移动应用使用行为（保留/安装/卸载）蕴含丰富的**用户长期和短期兴趣信息**，但传统方法依赖人工特征工程，效率低且效果受限于领域专家
- 异构行为需要统一建模、行为在时间上分布不均、长尾应用严重稀疏——这三大挑战催生了AETN的设计
- AETN由保留自编码器、Transformer编码器和Transformer解码器三部分组成，通过参数共享和**多目标联合训练**实现端到端学习
- 在腾讯超过 2000万 用户数据上训练，已在 腾讯手机管家 和 腾讯WiFi管家的推荐场景中部署上线

关键发现：

- **AETN在下周安装预测任务中，四个长尾应用类别的平均AUC相比DAE基线提升+0.0631（从0.7392到0.8023）**
- 线上A/B测试中，AETN嵌入在"发现"标签页带来PV CTR +5.96%、用户停留时间+8.14%的显著提升
- **遮蔽应用预测任务**（类似BERT的Masked LM）为 **嵌入质量** 贡献+0.0115 AUC增益，同时带来**数据增强效果**
- 改进的多头自注意力机制（将保留/瓶颈信息直接拼接到key和value）在安装预测上贡献+0.0069 AUC提升
- 未来方向：探索对Transformer编码器进行微调以学习任务特定的用户嵌入

---



## 摘要

在本文中，我们报告了腾讯在**基于移动应用使用行为的用户建模方面的最新实践**。移动应用使用中的用户行为，包括保留（retention，当前安装在手机上的应用）、安装（installation，最近安装了哪些应用及时间）和卸载（uninstallation，最近卸载了哪些应用及时间），能够很好地反映用户的长期和短期兴趣。例如，如果一个用户最近安装了Snapseed，她可能对摄影产生了日益浓厚的兴趣。这类信息对于包括广告、推荐在内的众多下游应用非常有价值。传统上，基于移动应用使用的用户建模严重依赖人工特征工程，针对不同的下游应用需要大量人力投入，且在缺乏领域专家时可能效果欠佳。然而，基于移动应用使用的自动用户建模面临着独特的挑战，包括：（1）保留、安装和卸载是异构的，但需要统一建模；（2）用户行为在时间上分布不均；（3）许多长尾应用面临严重的稀疏性问题。在本文中，我们提出了一种定制的自编码器耦合Transformer网络（AETN），通过该模型克服了这些挑战，实现了减少人工投入和提升性能的目标。我们已将该模型部署在腾讯，来自多个下游应用领域的线上和线下实验都证明了**输出用户嵌入**的有效性。

**关键词：** user modeling; embeddings; autoencoder; transformer; app usage



## 1 引言

个性化移动业务，例如推荐和广告，通常需要**有效的用户表示**。为了获得更好的性能，工业应用中的用户建模通常会考虑尽可能多的信息，包括但不限于性别、位置、兴趣标签、订阅的账号和购物兴趣[25]。其中，移动应用使用中的用户行为，包括*保留*（当前安装在手机上的应用）、*安装*（最近安装了哪些应用及时间）和*卸载*（最近卸载了哪些应用及时间），蕴含了丰富的用户长期和短期兴趣信息[26]。例如，如果一个用户安装了Google Photos、Snapseed和Instagram，她很有可能是移动摄影爱好者。如果一个用户最近安装了热门游戏《王者荣耀》，她可能是一个新手玩家，正在琢磨如何玩得更好。这类信息对各种下游应用都很有价值，如何更好地利用它们是一个值得解决的令人兴奋的问题。

传统上，从移动应用使用中挖掘信息依赖于特定任务的人工特征。例如，向已安装类似游戏的用户推荐新的游戏应用，可以避免向非游戏用户推荐。然而，人工特征工程通常需要大量人力投入，在缺乏领域专家时可能效果欠佳。为了提高效率和效果，需要**从移动应用使用中的用户行为自动生成通用用户表示**。

自2019年年中以来，我们一直在朝着这个目标努力，并已部署了多个版本的模型。在本文中，我们概述了腾讯的最新实践。构建面向多个下游应用的**通用用户表示**所面临的主要挑战包括：


- 保留、安装和卸载需要统一建模。它们从不同方面代表用户的偏好，分别**为三部分构建表示然后拼接可能会限制性能**。例如，对于已安装多个游戏的用户，卸载一个游戏应用可能只是表明她已经通关，想要开始新的游戏。而对于没有安装其他游戏的用户，安装后立即卸载可能表明她根本不喜欢这类游戏。使用传统的循环神经网络（RNN）来建模这种复杂关系具有挑战性。
- （卸载）安装应用的操作是**低频的，且在时间上分布不均**。图1展示了一个用户的应用安装和卸载记录示例。随着对新手机的兴奋感消退，大多数用户只在需要时才安装或卸载应用。此外，**用户通常一个月都不操作，但可能在某一天突然安装或卸载多个应用。在这种情况下，每两个行为之间的各种间隔不可忽略**。尽管基于RNN的模型在分析用户活动方面已经取得了成功[16,21]，但这些场景中的行为**通常具有显著更高的频率和几乎均匀的时间分布**。因此，传统RNN可能无法很好地完成此任务。
- 许多长尾应用面临严重的稀疏性。像微信和支付宝这样的热门应用已经安装在中国几乎所有的智能手机上，而长尾应用在一百万用户中可能只有几百次安装。然而，**用户在长尾应用上的行为通常能更好地反映其个性化兴趣。构建有效的用户表示需要利用长尾应用的信息，同时不受严重稀疏性的影响。**

<img src="/Users/dazhang/PycharmProject/Papers/9-app/.picture/image-20260813113306641.png" alt="image-20260813113306641" style="zoom: 33%;" />

> 图1. 保留、安装和卸载的示意图。（卸载）安装操作是低频的，且在时间上分布不均。

为实现这一目标，我们设计了一种定制的自编码器耦合Transformer网络（AETN）来分析移动应用使用中的用户行为。**该模型遵循经典的编码器-解码器框架，通过瓶颈层进行用户表示学习，并利用多目标联合训练方案进行参数学习**。图2展示了整体框架。该模型主要由三部分组成，即保留自编码器部分、（堆叠的）Transformer编码器部分和（堆叠的）Transformer解码器部分。这三部分通过参数共享紧密耦合并联合训练。所提出的模型完全无监督，并经过精心优化以从移动应用使用中学习用户嵌入。

![image-20260813113438708](/Users/dazhang/PycharmProject/Papers/9-app/.picture/image-20260813113438708.png)

> 图2. 所提出的AETN模型概览。该模型专为从移动应用使用中无监督学习有效的用户嵌入而设计。保留自编码器部分旨在基于共现关系学习应用和用户保留的良好表示。Transformer编码器部分统一建模保留、安装和卸载信息，并在瓶颈层将用户映射为一个嵌入。用户嵌入被Transformer解码器部分强制保留尽可能多的信息，用于**重建安装和卸载序列**。此外，**用户嵌入还需要能够重建保留**。

保留自编码器作为AETN的基础部分。从保留数据中应用的共现关系中，它学习并与Transformer网络共享有效的应用嵌入。作为**缓解稀疏性问题**的设计之一，我们同时使用应用ID 和 其对应的**类别ID**来建模应用嵌入。因此，如果某个应用的使用极其稀疏，至少其类别ID可以提供一些信息。另一个设计是编码器和解码器之间的权重绑定。请注意，我们只绑定自编码器的第一层和最后一层，以保留足够的灵活性。权重绑定可以显著**减少自由参数的数量，并加速收敛**[15]。与应用嵌入一起，有效的用户保留表示被获取并提供给Transformer部分。

另一方面，Transformer部分统一建模保留、安装和卸载，并输出最终的用户嵌入。Transformer网络已被证明在自然语言处理中对建模（多个）序列和获取上下文表示是有效的[10]。受BERT[10]的启发，在本文中，我们使用堆叠的Transformer网络来**整合不同类型的信息**。

Transformer编码器部分接收用户保留、共享应用嵌入、日期嵌入和行为类型嵌入（保留、安装和卸载）作为输入。因此，输入包含了用户是否安装或卸载了什么应用以及何时操作的完整信息，以及他们**当前的应用使用状态**。**日期嵌入使Transformer适合建模低频且在时间上分布不均的用户行为**。此外，我们还**引入了一个类似BERT[10]的*遮蔽应用预测*任务，以帮助更高效地提取信息。**

在瓶颈层压缩所有输入信息后，（堆叠的）Transformer解码器部分尝试重建（卸载）安装序列。重建遵循类似于非自回归翻译[13]的方式。日期嵌入 和 行为类型嵌入被用作查询。我们还使用多层感知机网络从瓶颈层重建保留数据。重建过程迫使瓶颈层通过Transformer编码器从原始输入中保留尽可能多的信息。

此外，我们在Transformer编码器和解码器的输出层中使用权重绑定。更重要的是，为了更好地促进Transformer网络内部的信息交互，我们提出了一种改进的多头自注意力机制，在每一步注意力计算中更直接地将保留或瓶颈的表示输入到注意力机制中。上述所有组件都在腾讯数千万用户的数据上联合训练。来自Transformer网络瓶颈层的表示被用作通用用户嵌入，可以丰富许多需要用户表示的下游应用。我们工作的主要贡献总结如下：

- 我们介绍了我们最近在基于移动应用使用的通用用户嵌入学习方面的实践，用于多个下游应用。
- 我们提出了一种定制的模型AETN来实现这一目标。通过精心设计的神经网络结构，自编码器耦合Transformer网络**克服了长尾应用的严重稀疏性和活动的不均匀分布**，并统一建模移动应用使用中的用户行为。我们的代码已公开发布[1]。
- 在实际应用场景中，模型训练的成本是可接受的。大量的线上和线下实验验证了所提出模型的有效性，该模型已在腾讯的真实系统中部署，并在日常业务中取得了性能提升。

本文其余部分的组织如下。我们在第2节介绍背景。第3节和第4节分别描述我们的高层系统和AETN的详细设计。我们在第5节和第6节分别介绍线下实验和线上A/B测试。模型部署的细节在第7节介绍。相关工作在第8节讨论，第9节得出结论。



## 2 背景

腾讯手机管家（Tencent Mobile Manager）目前是中国最流行的移动安全和管理应用，为近十亿用户提供服务。我们提供各种辅助功能，包括新闻推荐、短视频推荐、应用推荐等。例如，用户可以从腾讯手机管家的"早报"标签页以及腾讯WiFi管家（腾讯手机管家的配套应用）的"发现"标签页获取个性化内容流，包括新闻、文章和短视频。

我们已经建立了一个数据中心来支持各种下游应用。传统的人工特征工程和浅层模型可能无法最大化数据的价值，因此，就移动应用使用中的用户行为而言，通用用户表示是迫切需要的。



## 3 系统概览

在本节中，我们从高层视角介绍基于AETN的系统，并回顾其数据处理、模型训练和服务组件。

### 3.1 数据预处理

我们需要将用户数据预处理为适合后续模型处理的格式，同时减少数据中的噪声。经过数据预处理，每个用户由其"保留"和四个序列表示。"保留"是当前安装在用户手机上的应用集合。其中两个序列代表最近的"安装"操作，**由已安装的应用和对应日期组成**。**另外两个序列代表最近的"卸载"操作**。为了**减少用户行为中的噪声，我们为每个用户保留一周内最近10次安装或卸载操作**。

我们使用以下标准来选择模型中考虑的应用：

- 我们**手动排除了一些排名靠前的应用**，这些应用几乎安装在每一部智能手机上，很难代表用户兴趣，例如微信。同时，我们保留了像《王者荣耀》这样的热门应用，因为它们仍然可以代表用户的个性化兴趣。
- 我们**排除了手机厂商预装的应用**。
- 我们还**排除了安装量低于某个阈值的小众应用**。

此外，一个应用可能有多个包名（package_name）对应不同品牌和型号的智能手机。它们都被合并以避免重复。对于应用类别，我们考虑相对更细粒度的应用类别，例如，我们区分"游戏"应用的不同子类别。

### 3.2 模型训练与服务

预处理数据后，我们按以下步骤训练模型并生成用户嵌入：

- 步骤1：模型训练。我们使用数千万用户训练AETN。
- 步骤2：推理。我们为所有用户提取用户嵌入，并将嵌入推送到DCache系统[2]以供服务使用。
- 步骤3：服务。下游应用可以**使用 特征ID 和 用户ID 检索用户嵌入**。梯度提升决策树（GBDT）和神经网络（NN）通常被用作下游模型。

关于部署的更多细节在第7节。



## 4 自编码器耦合Transformer网络

在本节中，我们首先定义用户行为的符号表示，然后介绍所提出网络的详细结构。接着，我们详细阐述 **缓解稀疏性问题** 的设计以及 对**原始Transformer的修改**。最后，我们介绍用于模型优化的**多目标联合训练方案**。

### 4.1 用户行为的符号表示

如第3.1节所述，每个用户的行为**被预处理为其"保留" 和 四个序列**，定义如下。

保留。用户 $u$ 的保留可以用一个多热向量 $\boldsymbol{x}_{u} \in \mathbb{R}^{M}$ 表示，当应用 $m$ 已安装时 $x_{um} = 1$，其中 $M$ 是所考虑的应用数量。

安装和卸载。代表用户 $u$ 最近 $I$ 次安装或卸载应用操作的四个序列表示为 $\mathcal{S}_{u}$：

$$
\mathcal{S}_{u} = \Big\{ [a^{n}_{u,1},\dots,a^{n}_{u,i},\dots,a^{n}_{u,I}], [d^{n}_{u,1},\dots,d^{n}_{u,i},\dots,d^{n}_{u,I}], [a^{l}_{u,1},\dots,a^{l}_{u,i},\dots,a^{l}_{u,I}], [d^{l}_{u,1},\dots,d^{l}_{u,i},\dots,d^{l}_{u,I}] \Big\}
$$

具体而言，$a^{n}_{u,i}$ 表示 $u$ 第 $i$ 次新安装应用的ID，$d^{n}_{u,i}$ 是对应的日期。$a^{l}_{u,i}$ 和 $d^{l}_{u,i}$ 是卸载的对应项。此外，$1 \leq a^{n}_{u,i}, a^{l}_{u,i} \leq M$，所有操作都发生在最近 $T$ 个时间间隔内。

请注意，在本文其余部分，我们在大多数符号中省略了表示用户的下标 $u$ 以简化表示。

### 4.2 网络结构

如图2所示，保留自编码器、Transformer编码器和Transformer解码器是所提出模型的三个主要部分。我们**通过瓶颈层连接后两个部分**。Transformer编码器有一个嵌入层，解码器也有一个嵌入层。网络结构的详细信息如下。

保留自编码器。AETN采用一个具有三个隐藏层的自编码器来重建和编码用户的保留。自编码器可以用三元组 $(f^{(p)}, \mathbf{W}^{(p)}, \boldsymbol{b}^{(p)})$ 描述，其中 $p \in \{1,2,3,4\}$。$\mathbf{W}^{(p)}$ 和 $\boldsymbol{b}^{(p)}$ 是第 $p$ 层的权重和偏置，$f^{(p)}$ 表示对应的激活函数。我们选择常用的LeakyReLU[28]作为前三层的激活函数，$f^{(4)}$ 是sigmoid函数[27]。令 $\boldsymbol{x}^{(p)}$ 表示各层的输出，计算如下：

$$
\boldsymbol{x}^{(p)} = f^{(p)}(\boldsymbol{x}^{(p-1)}\mathbf{W}^{(p)} + \boldsymbol{b}^{(p)}), \quad p \in \{1,2,3,4\} \tag{1}
$$

其中 $\boldsymbol{x}^{(0)}$ 是从**用户保留 $\boldsymbol{x}$ 使用 $\ell_2$ 范数归一化**得到的。

这个自编码器的作用是双重的。首先，它有助于从应用的共现关系中学习高质量的应用嵌入。第一个隐藏层的权重矩阵 $\mathbf{W}^{(1)}$ 作为整个网络的**共享应用嵌入矩阵** $\mathbf{W}^{a}$，即：

$$
\mathbf{W}^{a} = \mathbf{W}^{(1)} \in \mathbb{R}^{M \times d_{model}} \tag{2}
$$

为了进一步缓解稀疏性问题，共享应用嵌入矩阵经过精心设计并与一些其他权重矩阵绑定。更多细节在第4.3节中提供。

其次，这个自编码器为Transformer部分提供有效的用户保留表示。**Transformer编码器部分需要接收保留信息，以便将长期兴趣压缩到用户嵌入中**。然而，保留最初是高维稀疏特征的形式。这个自编码器将保留编码到第一个隐藏层 $\boldsymbol{x}^{(1)} \in \mathbb{R}^{d_{model}}$。作为低维密集编码，$\boldsymbol{x}^{(1)}$ 在Transformer编码器部分中起着重要作用。

Transformer编码器及其嵌入层。Transformer编码器是AETN整合和压缩所有信息的核心部分，没有合适的嵌入层它就无法工作。受位置编码[34]的启发，我们基于共享应用嵌入、日期嵌入 和 行为类型嵌入为Transformer编码器设计了一个嵌入层，如图3所示[33]。

<img src="/Users/dazhang/PycharmProject/Papers/9-app/.picture/image-20260813132243461.png" alt="image-20260813132243461" style="zoom: 33%;" />

>  图3. Transformer编码器的嵌入层。

**日期嵌入是使整个网络适合建模低频且在时间上分布不均的用户行为的关键**。通过日期嵌入，后续的Transformer编码器直接接收关于行为何时发生的信息，而不是从行为的顺序中推断。我们将日期嵌入矩阵表示为 $\mathbf{W}^{d} \in \mathbb{R}^{T \times d_{model}}$，日期 $t$ 由 $\boldsymbol{w}^{d}_{t} \in \mathbb{R}^{d_{model}}$ 表示。

> [!NOTE]
>
> TODO：日期嵌入这里没将清楚

**行为类型嵌入帮助模型在整合所有用户行为时区分不同类型**。对于三种用户行为类型（保留、安装和卸载），嵌入分别为 $\boldsymbol{w}^{x}, \boldsymbol{w}^{n}, \boldsymbol{w}^{l} \in \mathbb{R}^{d_{model}}$。

通过这个嵌入层，我们为Transformer编码器构建输入表示，输入包含了**移动应用使用的完整信息**。

我们的编码器块与原始Transformer编码器[34]共享相似的基本结构，为了**促进不同类型行为之间的信息交互**，我们对多头自注意力机制进行了小幅修改。更多细节在第4.4节中介绍。为了更好地从用户行为中提取信息，受BERT[10]中*遮蔽语言模型*任务的启发，我们对安装和卸载序列应用了*遮蔽应用预测*任务。输出softmax的权重矩阵表示为 $\mathbf{W}^{\Omega} \in \mathbb{R}^{d_{model} \times M}$。关于这个训练任务的更多细节在第4.5节中提供。

瓶颈层。瓶颈层是生成（低维）用户嵌入 $\boldsymbol{\widetilde{e}}$ 的地方。由于编码器和解码器在这一层融合，**来自原始输入的压缩信息成为重建任务的信息来源。**

在Transformer编码器输出的最终隐藏向量 $\boldsymbol{e}^{\Omega}_{x}$（即与保留对应的表示）之上，我们使用一个单隐藏层自编码器进一步将维度从 $d_{model}$ 降低到 $d_{emb}$。瓶颈的激活函数是tanh。然后这个自编码器的重建输入被送入Transformer解码器部分。

> [!NOTE]
>
> TODO：这里没说清楚

在训练方案中，我们使用多层感知机网络和sigmoid激活函数从用户嵌入重建其保留。输出层的权重矩阵表示为 $\mathbf{W}^{\Theta} \in \mathbb{R}^{d_{model} \times M}$。

Transformer解码器及其嵌入层。Transformer解码器服务于我们以**非自回归方式[13]重建安装和卸载的目的**[12]。更具体地说，我们使用日期和行为类型作为查询，从用户嵌入中搜索有价值的信息以重建对应的已安装或已卸载应用。为此，我们为Transformer解码器设计了一个新的嵌入层，与编码器的嵌入层共享日期嵌入和行为类型嵌入。图4展示了这个嵌入层和解码器输入的细节。

<img src="/Users/dazhang/PycharmProject/Papers/9-app/.picture/image-20260813132305383.png" alt="image-20260813132305383" style="zoom: 33%;" />

>  图4. Transformer解码器的嵌入层。

为了完成重建整个安装和卸载序列的任务，我们将解码器中与安装和卸载对应的所有隐藏向量送入一个输出softmax层。该层的权重矩阵表示为 $\mathbf{W}^{\Phi} \in \mathbb{R}^{d_{model} \times M}$。

### 4.3 权重矩阵设置

我们为模型的几个部分精心设计了权重矩阵，这有助于解决稀疏性问题，并将自编码器部分和Transformer部分紧密耦合。如图5所示，应用嵌入基于应用ID和其对应的类别ID构建。即使某个应用的使用极其稀疏，其类别仍然可以提供有效信息。这种设置有助于克服稀疏性问题。

<img src="/Users/dazhang/PycharmProject/Papers/9-app/.picture/image-20260813132323382.png" alt="image-20260813132323382" style="zoom:50%;" />

> 图5. 应用嵌入示意图。

如前所述，我们重复使用 $M \times d_{model}$ 的应用嵌入矩阵，即在保留自编码器的输入和输出、Transformer编码器的输入、遮蔽应用预测的输出、Transformer解码器的输出，以及从用户嵌入（瓶颈）重建保留的部分。我们将所有这些部分的权重矩阵绑定在一起，即：

$$
\mathbf{W}^{\Omega} = \mathbf{W}^{\Theta} = \mathbf{W}^{\Phi} = \mathbf{W}^{(4)} = {\mathbf{W}^{a}}^{\mathrm{T}} \tag{3}
$$

通过绑定上述层的权重矩阵，我们减少了总参数量，这有利于克服稀疏性问题。此外，权重绑定有利于梯度的反向传播并加速收敛。

### 4.4 改进的多头自注意力

在我们的场景中，保留、瓶颈（用户嵌入）、安装和卸载是异构的。每次安装或卸载代表一个单独的操作，但保留或瓶颈是所有安装和卸载操作的累积。因此，为了更好地促进保留、瓶颈和（卸载）安装之间的信息交互，多头自注意力被修改，如图6所示。

<img src="/Users/dazhang/PycharmProject/Papers/9-app/.picture/image-20260813132344768.png" alt="image-20260813132344768" style="zoom: 33%;" />

>  图6. 改进的多头自注意力。它同时应用于Transformer编码器和解码器。

通过将保留（对于Transformer编码器部分）或瓶颈（对于Transformer解码器部分）拼接到缩放点积注意力的每个key和value，我们在每一步注意力计算中强制与保留或瓶颈进行信息交互。这样，Transformer编码器更高效地融合来自保留和（卸载）安装的信息，解码器更好地从瓶颈中提取信息用于重建任务。如实验结果所示，这种修改提高了用户嵌入的质量。

### 4.5 多目标联合训练方案

对于模型训练，我们采用包含三个任务的联合训练方案，即：

任务#1：主重建。为了基于用户在移动应用使用上的行为生成通用用户嵌入，我们训练所提出的模型从用户嵌入重建所有保留、安装和卸载信息。这个任务在联合训练方案中是不可或缺的，可以分为两个子任务：（1）通过多层感知机网络从用户嵌入（瓶颈层）重建保留数据。我们选择sigmoid交叉熵作为损失函数。（2）通过Transformer解码器重建安装和卸载序列。我们通过平均每次（卸载）安装的softmax交叉熵损失来计算此子任务的损失。这两个子任务的损失之和就是这个主重建任务的损失，我们将损失表示为 $\mathcal{L}_{main}$。

任务#2：辅助保留重建。这个辅助任务是为自编码器部分设计的。我们同样选择sigmoid交叉熵作为损失函数，表示为 $\mathcal{L}_{aux}$。

任务#3：遮蔽应用预测。这个任务类似于BERT[10]中的"Masked LM"任务。我们随机遮蔽安装和卸载中的应用，但保留对应的日期和行为类型。Transformer编码器仅被训练来预测被遮蔽的应用。为了简单起见，我们直接沿用BERT的遮蔽率，放弃"随机替换或保留"。我们通过平均每个被遮蔽应用的softmax交叉熵损失来计算这个任务的损失，表示为 $\mathcal{L}_{mask}$。

我们模型的最终损失函数是上述三个任务的损失以及正则化损失之和，即 $\mathcal{L} = \mathcal{L}_{main} + \mathcal{L}_{aux} + \mathcal{L}_{mask} + \mathcal{L}_{reg}$。$\mathcal{L}_{reg}$ 是所有可训练参数的 $\ell_2$ 范数正则化损失。



## 5 线下实验

在本节中，我们展示AETN在生成通用用户嵌入方面的线下性能。我们将基线与AETN的四个不同版本在**三个典型的下游线下实验**中进行比较。然后我们表明，**自编码器部分的辅助保留重建任务**可以帮助Transformer部分的收敛。最后，我们直观地比较基线和AETN生成的用户嵌入。

### 5.1 数据集

我们使用来自**腾讯的真实工业数据**进行模型训练。按照第3.1节中介绍的规则，我们考虑了超过1万个应用。然后我们采样了2000万用户和5亿条2019年7月至2019年12月的安装和卸载记录。我们随机分出约500万用户用于验证。

### 5.2 模型

我们训练和评估了5个模型，包括一个基线和AETN的四个不同版本，如下所示。

- DAE。去噪自编码器[35,36]被广泛应用于无监督表示学习。我们**基于用户保留数据训练生成用户嵌入**。
- AETN w/o $\mathcal{L}_{mask}$。AETN的退化版本，**不包含遮蔽应用预测任务**。
- AETN w/o $\mathcal{L}_{aux}$。AETN的另一个退化版本，**不包含辅助保留重建任务**。
- V-AETN。使用Vaswani等人[34]提出的**原始多头自注意力的AETN**。
- AETN。第4节中介绍的模型完整版本。

模型设置和超参数配置的详细信息列在附录A.1中。基于RNN的模型未被纳入。除了用户行为的不均匀分布外，**训练效率低下**也使它们在我们的场景中不可行。

### 5.3 线下评估测试

我们在三个典型的下游应用上进行线下实验，包括来自相关领域和不同领域的应用。评估任务如下：

测试#1：下周安装预测。这个任务是预测哪些**用户将在下周安装特定（小众）类别的应用**。我们从约500万用户中收集数据，然后按3:1:1的比例分为训练集、验证集和测试集。生成用户嵌入后，我们**训练多层感知机网络**来预测用户是否会在下周安装四个类别的应用。

测试#2：相似人群扩展。这是计算广告中的常见任务[39,30]。我们使用一个包含约50万用户的数据集，其中**约10%是某个词表外小众应用的种子用户**。按照常见做法，我们训练XGBoost[6]相似人群模型来评估不同的用户嵌入，并报告10折交叉验证的结果。

> [!NOTE]
>
> TODO：这是什么意思？

测试#3：信息流推荐。为了在跨域场景中评估通用用户嵌入，我们使用来自腾讯WiFi管家"发现"标签页的信息流推荐数据。我们选择约120万用户并提取他们8天的行为，然后使用前7天的数据进行训练，最后一天的数据用于验证和测试。训练集包含约2700万条记录，验证集和测试集各包含约200万条记录。我们基于生成的用户嵌入以及其他特征训练Deep & Cross Networks[37]用于信息流推荐[36]。

在所有三个任务中，我们使用ROC曲线下面积（AUC）作为指标。每个测试运行5次并报告平均值。

### 5.4 线下评估结果

表1. 用户嵌入的线下评估结果。

<img src="/Users/dazhang/PycharmProject/Papers/9-app/.picture/image-20260813132452821.png" alt="image-20260813132452821" style="zoom: 25%;" />

表1展示了所有三个下游实验的结果。我们可以从结果中得出以下结论。

所有版本的AETN都比DAE表现更好。在下周安装预测中，AETN为四个类别带来了平均AUC 0.0631的提升。其余两个应用分别享受了0.0134和0.0048的提升。**对于工业应用来说，0.1%的AUC提升都是显著的[**27]，这是一个重大改进。这种改进证实了两个假设。首先，安装和卸载中包含的短期用户兴趣对各种下游应用都有价值，只是程度不同。其次，所提出的AETN能够从所有类型的用户行为中提取长期和短期用户兴趣，并将它们**压缩到用户嵌入**中。

遮蔽应用预测任务对提高用户嵌入质量做出了重要贡献。它在下周安装预测中带来了平均AUC 0.0115的提升。即使对于相似人群扩展和信息流推荐，该任务带来的AUC提升也超过0.0010。我们将此归因于遮蔽应用预测不仅帮助Transformer编码器更高效地提取信息，还在训练过程中带来了**数据增强效果**。

改进的多头自注意力比原始版本表现更好。这个简单的修改——促进保留、瓶颈和（卸载）安装之间的信息交互——为下周安装预测贡献了0.0069的AUC增益。

辅助保留重建也有利于生成的用户嵌入的质量。没有自编码器部分的这个辅助任务，下周安装预测中的AUC下降了0.0023。除了用户嵌入的改进外，我们还发现训练效率也因辅助保留重建而提高。

### 5.5 训练效率比较

在训练AETN和AETN w/o $\mathcal{L}_{aux}$时，我们监控验证数据集上 $\mathcal{L}_{main}$ 和 $\mathcal{L}_{mask}$ 的总和，以确认辅助保留重建带来的训练效率提升。如图7a所示，辅助任务使AETN中的Transformer部分收敛更快。通过自编码器和权重绑定，来自输出层的梯度可以通过更少的层传递到应用嵌入矩阵。此外，完整版本的AETN在两个模型最终收敛时也实现了更低的损失。

<img src="/Users/dazhang/PycharmProject/Papers/9-app/.picture/image-20260813132421212.png" alt="image-20260813132421212" style="zoom: 33%;" />

图7. (a) 验证数据集上 $\mathcal{L}_{main}$ 和 $\mathcal{L}_{mask}$ 总和的记录。两个模型除了辅助保留重建外使用相同的设置训练。(b) 1万对邻居用户之间的应用重叠率。

### 5.6 邻居用户之间的应用重叠

**为了直观地比较AETN和DAE输出的嵌入，我们基于嵌入测量邻居用户对之间的应用重叠率**。对于每个用户，我们根据嵌入选择欧氏距离最小的对应用户作为邻居。我们随机采样1万用户，并从100万随机选择的用户中找到他们的邻居。对于每对邻居，我们计算保留、安装和卸载中的应用重叠率。图7b展示了AETN嵌入和DAE嵌入的所有邻居对的平均结果。结果表明，AETN成功地将安装和卸载的信息注入到用户嵌入中，同时保留了大部分保留信息。与此同时，仅基于保留信息提取的DAE嵌入无法提供太多关于安装和卸载的信息。

> [!NOTE]
>
> TODO：没看懂



## 6 线上A/B测试

为了进一步验证输出用户嵌入的有效性，我们在2020年2月1日至2020年2月10日期间，在腾讯手机管家的"早报"标签页和腾讯WiFi管家的"发现"标签页进行了线上信息流推荐A/B测试。我们按用户ID均匀分配线上A/B测试流量。我们评估了基础模型、带有DAE嵌入的模型和带有AETN嵌入的模型。与基础模型相比的改进结果报告在表2中。

表2. 带有DAE嵌入和AETN嵌入的模型的线上评估结果。

<img src="/Users/dazhang/PycharmProject/Papers/9-app/.picture/image-20260813132517239.png" alt="image-20260813132517239" style="zoom:50%;" />

我们主要考虑以下指标。UV CTR衡量用户维度的点击率，PV CTR衡量页面维度的点击率。用户停留时间衡量每个用户的平均停留时间。阅读文章数衡量每个用户阅读的平均文章数。从表中我们可以发现，与基础模型相比，AETN嵌入使所有指标都得到了2%到8%的改进。与DAE嵌入相比，PV CTR和用户停留时间从AETN嵌入中获得了更显著的改进，我们假设AETN引入了安装和卸载信息，因此除了来自保留的长期兴趣外，还能捕捉用户的短期兴趣，而这些信息对PV CTR和用户停留时间更为关键。比较"早报"标签页和"发现"标签页的结果，我们可以发现"发现"标签页的改进更为显著。这可能是因为用户倾向于全天在"发现"标签页阅读文章，而"早报"标签页的用户主要在早上起床后阅读新闻。"发现"标签页每用户的曝光量显著更多。因此，对用户兴趣进行更好的建模更为关键。



## 7 部署

我们使用TensorFlow[2]实现该模型。使用4块NVIDIA Tesla M40 GPU训练大约需要60小时。由于嵌入同时代表用户的长期和短期兴趣，持续更新嵌入以获得最佳性能至关重要。然而，大量用户给频繁更新带来了挑战。通常，我们有两种更新策略：

- 模型更新。我们可以更新模型以获得最佳性能。这种方法考虑了新兴应用和数据分布的完全最新应用列表。然而，更新模型会完全改变用户嵌入的语义结构。因此，我们需要同时更新所有下游模型。
- 特征更新。我们也可以保持模型固定，只更新用户的特征。这样我们考虑了用户的最新行为，更新后的嵌入仍然可以在相同的语义空间中。这种策略使更新成本更低。

在实践中，我们发现特征更新对下游应用更具成本效益，这是因为在几个月内应用通常不会发生剧烈变化。然而，为十亿级用户更新嵌入仍然具有挑战性。为了减少计算量，我们每天只更新下游应用的活跃用户的表示。这种策略可以将每次需要更新的用户数量减少到百万级。模型可以更不频繁地更新。一旦模型更新，我们使用新的特征ID以防止混淆。



## 8 相关工作

我们在三个领域总结相关工作，包括应用行为数据的应用、无监督特征提取和Transformer网络。

### 8.1 应用使用数据的应用

移动应用使用中的用户行为包含丰富的偏好信息，已被用于各种应用[26]。其中最重要的是应用安装广告[12,20]和移动应用推荐[39]。Yahoo发布了一个大规模的应用安装广告预测引擎，基于考虑了从应用行为生成的用户特征的两步逻辑回归模型[5][4]。为了减少稀疏性，Yahoo还在理解应用使用模式时将应用分类到预定义的兴趣分类法中[32]。应用使用模式通过深度记忆网络学习用于应用购买推荐[11]。除了应用安装广告外，用户的应用安装行为还被用于新闻推荐[23]，其中冷启动用户的邻域知识从应用领域转移到新领域。一项关于全球主要应用市场移动应用用户行为的大型调查被进行，以指导跨国应用竞争并分析软件工程面临的挑战[22][21]。

在本文中，我们解决了基于应用使用行为的通用用户嵌入的现实需求。用户嵌入可以用于各种下游应用。

### 8.2 无监督表示学习

无监督表示学习是一个长期存在的问题[4,38]，自编码器已在许多现实应用中成功部署[3]。它遵循编码器-解码器结构，试图通过瓶颈层重建输入。稀疏自编码器[24]、去噪自编码器[35,36]、变分自编码器[31]、对抗自编码器[29]等已被提出作为扩展[30]。最近，更先进的无监督表示学习已被提出，包括用于自然语言处理的BERT[10]和用于计算机视觉的MoCo[14]。通过大量数据和深度模型，无监督表示学习能够在比传统监督学习更少标注的情况下达到相当甚至更好的性能[10,14]。

在本文中，我们介绍了一种从移动应用用户行为中进行无监督表示学习的方法。我们通过定制的自编码器耦合Transformer网络解决了这个问题的独特挑战，并证明了其有效性。

### 8.3 Transformer网络

Transformer模型最初在[34]中提出，并已被广泛用于自然语言处理任务中的序列建模[10]、推荐[33,7]和音乐生成[18]。Transformer可以通过自注意力机制同时关注输入序列的每个token，并且已证明具有足够头数的多头自注意力层至少与任何卷积层一样具有表达能力[8][7]。与循环神经网络如长短期记忆网络（LSTM）[17]相比，Transformer更具可并行性，在大型数据集上训练所需时间显著更少[34][16]。Transformer-XL[9]和Reformer[19]被提出以进一步降低序列长度很长（例如长度为10,000的序列）时的复杂度。

在本文中，我们将Transformer网络与自编码器耦合，以统一建模保留、安装和卸载。我们修改了原始Transformer，以便在建模安装和卸载时强调保留状态或用户嵌入。



## 9 结论

在本文中，我们介绍了我们最近在基于移动应用使用的 **无监督用户嵌入学习** 方面的实践。为了解决真实系统中这个问题的独特挑战，我们提出了一种称为自编码器耦合Transformer网络（AETN）的定制模型。大量的线上和线下实验结果证明了所提出模型的有效性。我们还介绍了部署的细节。输出的通用用户嵌入可以丰富腾讯中**需要用户表示的多个下游应用**。目前，输出嵌入已在**腾讯手机管家和腾讯WiFi管家的信息流推荐场景**中提供服务。未来，我们计划探索对Transformer编码器部分进行微调以学习任务特定的用户嵌入。



###### 致谢

作者感谢腾讯BlueWhale项目团队对我们研究的支持。



## 参考文献

[1] Martín Abadi, Paul Barham, Jianmin Chen, Zhifeng Chen, Andy Davis, Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Geoffrey Irving, Michael Isard, et al. 2016. Tensorflow: A system for large-scale machine learning. In *12th USENIX Symposium on Operating Systems Design and Implementation*. 265–283.

[2] Pierre Baldi. 2012. **Autoencoders, unsupervised learning, and deep architectures**. In *Proceedings of ICML workshop on unsupervised and transfer learning*. 37–49.

[3] Yoshua Bengio, Aaron Courville, and Pascal Vincent. 2013. **Representation learning: A review and new perspectives**. *IEEE transactions on pattern analysis and machine intelligence* 35, 8 (2013), 1798–1828.

[4] Narayan Bhamidipati, Ravi Kant, and Shaunak Mishra. 2017. **A large scale prediction engine for app install clicks and conversions**. In *Proceedings of the 2017 ACM on Conference on Information and Knowledge Management*. ACM, 167–175.

[5] Tianqi Chen and Carlos Guestrin. 2016. **Xgboost: A scalable tree boosting system**. In *Proceedings of the 22nd acm sigkdd international conference on knowledge discovery and data mining*. 785–794.

[6] Xusong Chen, Dong Liu, Chenyi Lei, Rui Li, Zheng-Jun Zha, and Zhiwei Xiong. 2019. **BERT4SessRec: Content-Based Video Relevance Prediction with Bidirectional Encoder Representations from Transformer**. In *Proceedings of the 27th ACM International Conference on Multimedia*. 2597–2601.

[7] Jean-Baptiste Cordonnier, Andreas Loukas, and Martin Jaggi. 2020. On the Relationship between Self-Attention and Convolutional Layers. In *International Conference on Learning Representations*.

[8] Zihang Dai, Zhilin Yang, Yiming Yang, Jaime G Carbonell, Quoc Le, and Ruslan Salakhutdinov. 2019. **Transformer-XL: Attentive Language Models beyond a Fixed-Length Context**. In *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics*. 2978–2988.

[9] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. Bert: Pre-training of deep bidirectional transformers for language understanding. *2019 Annual Conference of the North American Chapter of the Association for Computational Linguistics* (2019).

[10] Djordje Gligorijevic, Jelena Gligorijevic, Aravindan Raghuveer, Mihajlo Grbovic, and Zoran Obradovic. 2018. Modeling mobile user actions for purchase recommendation using deep memory networks. In *The 41st International ACM SIGIR Conference on Research & Development in Information Retrieval*. ACM, 1021–1024.

[11] Jillian Gogel. 2018. *AppsFlyer Forecasts Global App Install Ad Spend to Reach $64B by 2020*. Retrieved October 22, 2019 from https://www.appsflyer.com/blog/app-install-ad-spend-predictions-2017-2020/

[12] Jiatao Gu, James Bradbury, Caiming Xiong, Victor OK Li, and Richard Socher. 2018. Non-autoregressive neural machine translation. In *International Conference on Learning Representations*.

[13] Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. 2019. Momentum contrast for unsupervised visual representation learning. *arXiv preprint arXiv:1911.05722* (2019).

[14] Balázs Hidasi and Alexandros Karatzoglou. 2018. Recurrent neural networks with top-k gains for session-based recommendations. In *Proceedings of the 27th ACM International Conference on Information and Knowledge Management*. ACM, 843–852.

[15] Balázs Hidasi, Alexandros Karatzoglou, Linas Baltrunas, and D Tikk. 2016. Session-based recommendations with recurrent neural networks. In *International Conference on Learning Representations*.

[16] Sepp Hochreiter and Jürgen Schmidhuber. 1997. **Long short-term memory**. *Neural computation* 9, 8 (1997), 1735–1780.

[17] Cheng-Zhi Anna Huang, Ashish Vaswani Jakob Uszkoreit Noam Shazeer, and Monica Dinculescu Douglas Eck. 2019. Music transformer: Generating music with long-term structure. In *International Conference on Learning Representations*.

[18] Nikita Kitaev, Łukasz Kaiser, and Anselm Levskaya. 2020. **Reformer: The Efficient Transformer.** In *International Conference on Learning Representations*.

[19] Joowon Lee and Dong-Hee Shin. 2016. Targeting potential active users for mobile app install advertising: An exploratory study. *International Journal of Human–Computer Interaction* 32, 11 (2016), 827–834.

[20] Jing Li, Pengjie Ren, Zhumin Chen, Zhaochun Ren, Tao Lian, and Jun Ma. 2017. **Neural attentive session-based recommendation**. In *Proceedings of the 2017 ACM on Conference on Information and Knowledge Management*. ACM, 1419–1428.

[21] Soo Ling Lim, Peter J Bentley, Natalie Kanakam, Fuyuki Ishikawa, and Shinichi Honiden. 2014. Investigating country differences in mobile app user behavior and challenges for software engineering. *IEEE Transactions on Software Engineering* 41, 1 (2014), 40–64.

[22] Jixiong Liu, Jiakun Shi, Wanling Cai, Bo Liu, Weike Pan, Qiang Yang, and Zhong Ming. 2017. **Transfer Learning from APP Domain to News Domain for Dual Cold-Start Recommendation**.. In *RecSysKTL*. 38–41.

[23] Weifeng Liu, Tengzhou Ma, Dapeng Tao, and Jane You. 2016. HSAE: A Hessian regularized sparse auto-encoders. *Neurocomputing* 187 (2016), 59–65.

[24] Yudan Liu, Kaikai Ge, Xu Zhang, and Leyu Lin. 2019. Real-time Attention Based Look-alike Model for Recommender System. In *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*. ACM, 2765–2773.

[25] Eric Hsueh-Chan Lu, Yi-Wei Lin, and Jing-Bin Ciou. 2014. **Mining mobile application sequential patterns for usage prediction**. In *2014 IEEE International Conference on Granular Computing (GrC)*. IEEE, 185–190.

[26] Xiao Ma, Liqin Zhao, Guan Huang, Zhi Wang, Zelin Hu, Xiaoqiang Zhu, and Kun Gai. 2018. Entire space multi-task model: An effective approach for estimating post-click conversion rate. In *The 41st International ACM SIGIR Conference on Research & Development in Information Retrieval*. 1137–1140.

[27] Andrew L Maas, Awni Y Hannun, and Andrew Y Ng. 2013. Rectifier nonlinearities improve neural network acoustic models. In *Proc. icml*, Vol. 30. 3.

[28] Alireza Makhzani, Jonathon Shlens, Navdeep Jaitly, Ian Goodfellow, and Brendan Frey. 2015. Adversarial autoencoders. *arXiv preprint arXiv:1511.05644* (2015).

[29] Ashish Mangalampalli, Adwait Ratnaparkhi, Andrew O Hatch, Abraham Bagherjeiran, Rajesh Parekh, and Vikram Pudi. 2011. A feature-pair-based associative classification approach to look-alike modeling for conversion-oriented user-targeting in tail campaigns. In *Proceedings of the 20th international conference companion on World wide web*. 85–86.

[30] Yunchen Pu, Zhe Gan, Ricardo Henao, Xin Yuan, Chunyuan Li, Andrew Stevens, and Lawrence Carin. 2016. Variational autoencoder for deep learning of images, labels and captions. In *Advances in neural information processing systems*. 2352–2360.

[31] Vladan Radosavljevic, Mihajlo Grbovic, Nemanja Djuric, Narayan Bhamidipati, Daneo Zhang, Jack Wang, Jiankai Dang, Haiying Huang, Ananth Nagarajan, and Peiji Chen. 2016. **Smartphone app categorization for interest targeting in advertising marketplace**. In *Proceedings of the 25th International Conference Companion on World Wide Web*. International World Wide Web Conferences Steering Committee, 93–94.

[32] Fei Sun, Jun Liu, Jian Wu, Changhua Pei, Xiao Lin, Wenwu Ou, and Peng Jiang. 2019. **BERT4Rec: Sequential recommendation with bidirectional encoder representations from transformer**. In *Proceedings of the 28th ACM International Conference on Information and Knowledge Management*. 1441–1450.

[33] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In *Advances in neural information processing systems*. 5998–6008.

[34] Pascal Vincent, Hugo Larochelle, Yoshua Bengio, and Pierre-Antoine Manzagol. 2008. Extracting and composing robust features with denoising autoencoders. In *Proceedings of the 25th international conference on Machine learning*. 1096–1103.

[35] Pascal Vincent, Hugo Larochelle, Isabelle Lajoie, Yoshua Bengio, and Pierre-Antoine Manzagol. 2010. Stacked denoising autoencoders: Learning useful representations in a deep network with a local denoising criterion. *Journal of machine learning research* 11, Dec (2010), 3371–3408.

[36] Ruoxi Wang, Bin Fu, Gang Fu, and Mingliang Wang. 2017. Deep & cross network for ad click predictions. In *Proceedings of the ADKDD'17*. 1–7.

[37] Daokun Zhang, Jie Yin, Xingquan Zhu, and Chengqi Zhang. 2018. **Network representation learning: A survey**. *IEEE transactions on Big Data* (2018).

[38] Weinan Zhang, Lingxi Chen, and Jun Wang. 2016. Implicit Look-Alike Modelling in Display Ads. In *European Conference on Information Retrieval*. Springer, 589–601.

[39] Hengshu Zhu, Hui Xiong, Yong Ge, and Enhong Chen. 2014. **Mobile app recommendations with security and privacy awareness**. In *Proceedings of the 20th ACM SIGKDD international conference on Knowledge discovery and data mining*. 951–960.



## 附录A 超参数和实验设置的详细信息

本附录提供了模型设置、超参数配置以及实验设置的详细补充信息。读者可以参考公开的代码获取更多实现细节。

### A.1 模型设置和超参数配置

模型和训练过程有许多设置和超参数。为了平衡效率和性能，我们根据之前的经验直接确定了其中一些，并根据验证数据集上的主重建损失和生成的用户嵌入在下游实验上的表现来寻找其他参数的最优设置。

就AETN的基本结构而言，我们选择使用一个具有三个隐藏层的自编码器，以及两个Transformer编码器层和一个Transformer解码器层。自编码器第一个隐藏层的维度和Transformer的隐藏大小，即 $d_{model}$，设置为512。我们将Transformer中逐位置前馈网络的隐藏大小设置为1024，自注意力头的数量为8。考虑到数据存储、计算复杂度和下游应用延迟的限制，瓶颈层的大小（即用户嵌入的维度）为128。

就正则化而言，用户保留输入层的dropout率设置为0.05，多头自注意力机制和逐位置前馈网络的dropout率设置为0.1。

通过监控验证数据集上主重建任务的损失，我们选择使用Adam作为优化器，mini-batch大小为1000。我们还选择应用指数衰减，学习率从0.0001开始，衰减率为每个epoch 0.8。对于 $\ell_2$ 范数正则化，我们将因子设置为 $1.5 \times 10^{-7}$。

安装或卸载序列的长度是另一个影响用户嵌入质量的重要超参数。我们在长度设置为15、20、25、30和35时训练了几个模型。然后我们在信息流推荐测试中生成和评估不同版本的用户嵌入，并确定最优长度为25。

在使用遮蔽应用预测任务训练所提出的AETN时，我们沿用BERT的遮蔽率。因此，我们在训练过程中遮蔽安装序列中的3个应用和卸载序列中的另外3个应用。请注意，我们**只在训练模型时遮蔽应用，安装和卸载的完整序列在验证和预测时保留**。

线下评估测试中的基线模型DAE与AETN中的自编码器共享相同的结构。DAE的瓶颈层维度也设置为128。

### A.2 线下评估数据集的额外细节

我们在下周预测中选择的四个类别是四个典型的小众类别，需要应用广告来扩大其用户群。这些类别的应用也是面临严重稀疏性的长尾应用。这四个类别的平均安装率分别约为每百万人600、400、25和300。

### A.3 应用重叠的测量

我们展示了邻居用户对之间应用重叠的详细测量方法。对于用户 $U$，我们找到她的邻居 $V$ 并获取他们的保留、安装和卸载信息。用户保留中应用重叠率的计算方法是将 $U$ 保留中的应用数量除以 $U$ 和 $V$ 保留交集中的应用数量。就安装或卸载中的应用重叠而言，操作的日期以及对同一应用的重复操作不被考虑。因此，我们首先将应用序列转换为应用集合。然后（卸载）安装中的应用重叠率的计算方法是将 $U$ 的（卸载）安装集合的大小除以 $U$ 和 $V$ 的（卸载）安装集合交集的大小。
