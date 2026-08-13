# 优化智能手机应用使用预测：一种点击率排序方法

> Yuqi Zhang, Meiying Kang, Xiucheng Li, Yu Qiu, Zhijun Li | 哈尔滨工业大学、苏州大学

本文提出了一种基于点击率（CTR）排序的应用使用预测方法 CTR-RAD，将传统的分类问题转化为 CTR 排序问题，解决了应用使用不平衡和部署期间分布外（OOD）的挑战。核心发现是——**CTR-RAD 在 Top-3 准确率上提升 4.93%，Top-5 准确率上提升 6.64%，在低频应用预测上达到基线方法约两倍的准确率**。

核心内容：
- 现有模型将应用使用预测视为分类问题，面临应用使用不平衡和部署期间分布外（OOD）两大挑战
- 提出 CTR-RAD：将分类问题转化为 CTR 排序问题，生成应用点击序列和三类判别特征
- 使用 DIEN（Deep Interest Evolution Network，深度兴趣演化网络）在云端训练 CTR 估计模型，部署到边缘设备
- 在大规模应用使用数据上验证，已成功部署于领先智能手机制造商的应用推荐系统

关键发现：
- **CTR-RAD 在 V2-data 上 Top-3 准确率达 81.78%，Top-5 准确率达 89.23%，分别超越最先进方法 4.93% 和 6.64%**
- 在低频应用预测上，CTR-RAD 达到分类模型约两倍的准确率，实现了全频谱均衡预测
- 亲和度特征对 Top-3 和 Top-5 提升显著，序列特征对 Top-1 提升显著，屏幕特征与其他特征组合效果最佳
- 未来方向：融合 POI 数据和物理活动事件，与其他先进 CTR 估计模型联合

---

## 摘要

在过去十年中，智能手机已成为不可或缺的个人移动设备，软件应用数量经历了显著增长。这些应用使用户能够无缝连接各种互联网服务，如社交通信和在线购物。准确预测智能手机应用使用可以有效改善用户体验并优化资源利用率。然而，现有模型通常将应用使用预测视为分类问题，这受到应用使用不平衡和部署期间分布外（OOD）问题的困扰。为应对这些挑战，本文提出了一种新颖的基于点击率（CTR）排序的应用使用预测方法。通过将分类问题转化为 CTR 问题，我们可以消除应用使用不平衡问题的负面影响。为解决部署期间的 OOD 问题，我们生成应用点击序列和三类判别特征，使其能够在未见过的应用上实现泛化。应用点击序列和三类特征作为输入，在云端训练 CTR 估计模型，然后将训练好的模型部署到用户的智能手机上，为每个已安装的应用预测 CTR。决策过程涉及对这些 CTR 值进行排序，并选择 CTR 最高的应用作为最终预测。我们的方法已在大规模应用使用数据上进行了广泛测试。结果表明，我们的方法能够超越最先进方法，在 Top-3 准确率上提升 4.93%，在 Top-5 准确率上提升 6.64%。与基线方法相比，在预测低使用频率应用方面实现了约两倍的准确率。我们的方法已成功部署在领先智能手机制造商的应用推荐系统上。

**关键词：** app usage prediction; click-through rate; ranking; deep interest evolution network; feature engineering

## 1 引言

智能手机应用使用预测[8]旨在根据用户历史应用使用行为预测其下一个可能使用的应用，在个性化服务中发挥着关键作用[2,4,12,20]。准确预测应用使用可以有效改善用户体验并优化资源利用率[3,10]。例如，智能手机制造商可以根据预测结果向用户推荐应用，从而提高用户满意度[11,17]。此外，准确的预测可以帮助操作系统预加载应用，减少启动延迟并优化资源分配[15]。

传统上，研究人员将应用使用预测视为分类问题[13,14,17]。具体而言，给定用户的历史应用点击序列，模型预测下一个可能使用的应用。然而，这种范式存在两个主要问题：

1. **应用使用不平衡**：在大量可用应用中，只有少数应用被非常频繁地使用，大多数应用使用频率适中或较低。因此，训练样本中目标应用的分布存在严重不平衡。分类模型在高频应用预测上表现出色，但在中低频应用预测上表现挣扎。在实践中，智能手机制造商对模型能够准确预测各种使用频率的应用更感兴趣，以提供个性化应用推荐。

2. **部署期间分布外（OOD）**：智能手机制造商通常在云端使用所有可用的应用标签训练应用使用预测模型，然后将其部署到边缘设备。然而，在部署期间可能会出现未见过的应用。

为解决这些问题，本文提出了一种新颖的方法——基于点击率排序的应用使用预测（CTR-RAD，Click-Through Rate Ranking-based Application Usage Prediction），将分类问题转化为点击率（CTR）排序问题。该方法涉及生成应用点击序列和三类判别特征作为输入，在云端训练 CTR 估计模型。然后将训练好的模型部署到用户的智能手机上，为每个已安装的应用预测 CTR。决策过程包括对这些 CTR 值进行排序，最终指定 CTR 最高的应用作为最终预测。值得注意的是， CTR 估计模型可以替换为任何基于用户行为序列的 CTR 估计模型。这种可互换性归因于所创建的应用点击序列和三类特征（序列特征、亲和度特征和上下文特征）在这些模型中的通用性。CTR-RAD 展示了以下两个优势：

1. **全频谱均衡性能**：与使用严重不平衡的应用标签训练的分类模型不同，CTR 估计模型仅使用两个标签（点击或未点击）进行训练，从而减轻了应用使用不平衡问题的负面影响。此外，我们引入了高频应用抑制策略，进一步减轻目标应用之间的不平衡。

2. **部署期间泛化到未见过的应用**：我们定义了可跨不同应用共享的应用点击序列和三类特征，使其能够在未见过的应用上实现泛化。此外，CTR-RAD 不依赖于特定用户，对新用户具有适应性。

总之，我们的主要贡献如下：
1. 我们提出了一种基于点击率排序的创新应用使用预测方法，解决了传统分类模型遇到的挑战。
2. 现有 CTR 估计模型通常针对产品推荐量身定制，在应用使用预测领域存在研究空白。本文通过引入通用的应用点击序列和三类适用于任何用户行为序列的特征来填补这一空白。
3. 通过使用应用使用数据进行广泛评估，结果表明 CTR-RAD 优于最先进方法，在 Top-3 准确率上提升 4.93%，在 Top-5 准确率上提升 6.64%。此外，与基线方法相比，CTR-RAD 在预测低使用频率应用方面实现了约两倍的准确率。值得注意的是，CTR-RAD 已成功部署在领先智能手机制造商的应用推荐系统上。

## 2 相关工作

本节简要概述了应用使用预测领域的现有文献。预测应用使用的基本基准方法通常包括识别最常用（MFU，Most Frequently Used）和最近使用（MRU，Most Recently Used）的应用[12]。在研究初期，概率模型如马尔可夫模型[9]和贝叶斯模型[2,21]通常用于建模应用点击序列。同时，早期研究倾向于通过结合来自传感器的多样化上下文信息来提升预测性能[4,12,18,20]。Shin 等人[12]对与应用使用相关的各种传感器上下文特征进行了全面分析。Do 等人[4]通过利用智能手机传感器的丰富上下文信息进一步预测用户未来的位置和应用使用。Zhu 等人[20]提出利用上下文日志挖掘用户的个人上下文感知偏好。此外，Zhao 等人[18]从用户轨迹中提取与人类移动性相关的特征，并将其与应用使用序列结合来训练分类器以预测应用使用。因此，应用使用预测在早期主要依赖于各种概率模型和丰富的上下文信息。

为了深入探索这一领域，研究人员采用了一系列基于深度学习的模型[3,7,10,11,13-17]来建模应用使用数据。AppUsage2Vec[17]作为一个经典的基于深度学习的应用使用预测模型，结合了应用注意力机制来量化每个应用对目标应用的影响，以及一个双 DNN 模块用于预测应用使用。随后基于深度学习预测应用使用的模型大致可分为两个主要类别。第一类方法称为基于序列的方法，采用各种形式的 RNN[7,14,15]来捕获应用点击序列中的时间模式。Lee 等人[7]引入了一个基于 GRU 的多任务学习框架，结合时间和位置上下文来增强应用使用预测。类似地，Xu 等人[15]和 Xia 等人[14]使用基于 LSTM 的网络来建模应用点击序列，同时也考虑时间和位置上下文。

另一类工作称为基于图的方法，利用图嵌入技术来捕获应用、位置和时间之间的相关性[3,10,11,13,16]。Chen 等人[3]构建三个二分图来表示各种关系（应用-位置、应用-时间和应用-应用类型）。然后他们引入一种异构图嵌入算法将这些图映射到共享的 latent 空间。然而，这种方法忽视了应用使用的动态性质。为克服这一局限性，Yu 等人[16]设计了一个图，其中节点代表应用、时间和位置，边封装了它们的共现关系。随后他们利用基于 GCN 的模型学习语义感知的时空表示。同时，为了捕获用户兴趣随时间的动态变化，Ouyang 等人[10]将用户应用使用行为建模为动态图。他们提出了一种动态使用图网络来获取该动态图内的有效嵌入。此外，Shen 等人[11]构建了一个属性感知有向图，并开发了一种基于注意力的聚合模型来表征应用使用模式。

最近的研究呈现出将基于序列的方法与基于图的方法相结合的趋势。Wang 等人[13]提出了 SGFNN，这是一种开创性的模型，将这两个领域无缝集成并以端到端方式进行训练。

总之，现有方法优先探索概率模型和深度学习技术，同时努力结合多样化的上下文信息来预测应用使用。然而，这些方法将应用使用预测视为分类问题，这受到应用使用不平衡和部署期间分布外问题的困扰。

## 3 方法

CTR-RAD 的整体框架如图 1 所示。本节首先在第 3.1 节解释 CTR 估计模型输入数据的准备。随后，第 3.2 节介绍使用构建的数据训练点击率估计模型。这些数据可以适用于任何基于用户行为序列的 CTR 估计模型。为了举例说明，我们采用了一个经典模型——深度兴趣演化网络（DIEN，Deep Interest Evolution Network）[19]。最后，第 3.3 节以 CTR-RAD 的预测过程结束。



![图1](/Users/dazhang/PycharmProject/Papers/9-app/.picture/2024-CTR-RAD-Optimizing Smartphone App Usage Prediction-fig1.png)

图 1. CTR-RAD 的整体框架。

### 3.1 数据准备

我们采用五种类型的事件，即 App Click（应用点击）、Screen（屏幕）、Headset（耳机）、WiFi 和 Install（安装），来生成输入数据。每个事件是一个由（EventType, Timestamp, Value, Date, UserID）组成的元组。表 1 展示了五个不同类型的事件实例。原始数据经过预处理，构建应用点击序列和三类不同的特征，作为第 3.2 节中点击率估计模型的输入。

表 1：事件示例

| EventType | Timestamp | Value | Date | UserID |
|-----------|-----------|-------|------|--------|
| App Click | 1648893782448 | {app_name} | 20220402 | 42839 |
| Screen | 1648891954527 | {screen_on; screen_off} | 20220402 | 42839 |
| Headset | 1648950513848 | {headset_connected; headset_disconnected} | 20220403 | 141 |
| WiFi | 1648803122971 | {WiFi_connected; WiFi_disconnected} | 20220401 | 23 |
| Install | 1648803122971 | {install; uninstall; update} | 20220401 | 27955 |

#### 3.1.1 应用点击序列

基于 App Click 事件，我们首先按时间戳升序收集应用点击序列。然后，我们使用窗口大小 $M$ 和持续时间限制 $D$ 生成训练样本。这意味着每个训练样本由 $M$ 个应用组成，且这些应用在窗口内的累计持续时间不超过 $D$。在每个训练样本中，记为 $a_{1:n+1} = (a_1, \ldots, a_{n+1})$，$a_{n+1}$ 代表最终点击的应用，作为模型的目标，而 $a_{1:n}$ 代表所有先前点击的应用，被视为输入。每个 $a_{1:n+1}$ 都伴随着相应的点击时间序列 $t_{1:n+1} = (t_1, \ldots, t_{n+1})$，其中 $t_{n+1}$ 表示预测时间。

#### 3.1.2 序列特征

序列特征指的是与应用点击顺序相关的特征。我们通过实验确定了两个有价值的特征：

1) **时间间隔特征**，捕获应用点击与预测时间之间的持续时间。对于序列 $a_{1:n+1}$ 中的每个应用点击，定义相应的时间间隔特征为 $g_{1:n+1} = (g_1, \ldots, g_{n+1})$。每个时间间隔特征计算如下：

$$
g_i = t_{n+1} - t_i, \quad 1 \leq i \leq n+1 \tag{1}
$$

2) **屏幕特征**，捕获应用在屏幕激活后的点击序列中的位置。对于序列 $a_{1:n+1}$ 中的每个应用，定义相应的屏幕特征为 $s_{1:n+1} = (s_1, \ldots, s_{n+1})$。为了确定 $a_{1:n+1}$ 中每个 $a_i$ 的屏幕特征，我们首先识别 Screen 事件值为 screen_on 的时间 $t_{\text{screen\_on}}$，该时间在 $t_i$ 或 $t_i$ 之前最近的时间出现：

$$
t_{\text{screen\_on}} = \max\{t | t \leq t_i \text{ and Screen}(t) = \text{screen\_on}\} \tag{2}
$$

接下来，我们获取在时间间隔 $[t_{\text{screen\_on}}, t_i]$ 内发生的应用点击序列。因此，$s_i$ 等价于 $[t_{\text{screen\_on}}, t_i]$ 内序列的长度：

$$
s_i = |\{a_j | t_{\text{screen\_on}} \leq t_j \leq t_i\}| \tag{3}
$$

#### 3.1.3 上下文特征

上下文特征指的是对应于预测时间 $t_{n+1}$ 的某些上下文信息。我们通过实验确定了三个有价值的特征：

1) **耳机特征**，记为 $H(t_{n+1})$，表示时间 $t_{n+1}$ 时耳机的连接状态。该状态由两个值表征：headset_connected 和 headset_disconnected，如表 1 所述。我们首先识别 Headset 事件值出现的时间 $t_{\text{headset}}$，该时间在 $t_{n+1}$ 或 $t_{n+1}$ 之前最近的时间出现：

$$
t_{\text{headset}} = \max\{t | t \leq t_{n+1} \text{ and Headset}(t) \text{ exists value}\} \tag{4}
$$

然后，时间 $t_{\text{headset}}$ 处 Headset 事件的值被定义为 $t_{n+1}$ 处的耳机特征 $H(t_{n+1})$。

2) **WiFi 特征**，记为 $W(t_{n+1})$，表示时间 $t_{n+1}$ 时 WiFi 的连接状态。该状态包含两个不同的值：WiFi_connected 和 WiFi_disconnected，如表 1 所示。我们首先识别 WiFi 事件值出现的时间 $t_{\text{WiFi}}$，该时间在 $t_{n+1}$ 或 $t_{n+1}$ 之前最近的时间出现：

$$
t_{\text{WiFi}} = \max\{t | t \leq t_{n+1} \text{ and WiFi}(t) \text{ exists value}\} \tag{5}
$$

然后，时间 $t_{\text{WiFi}}$ 处 WiFi 事件的值被定义为 $t_{n+1}$ 处的 WiFi 特征 $W(t_{n+1})$。

3) **安装特征**，记为 $L(t_{n+1})$，表示时间 $t_{n+1}$ 处 Install 事件的值。我们为此事件选择两个有效值：install 和 update，如表 1 所述。首先，我们识别 Install 事件值出现的时间 $t_{\text{install}}$，该时间在 $t_{n+1}$ 或 $t_{n+1}$ 之前最近的时间出现：

$$
t_{\text{install}} = \max\{t | t \leq t_{n+1} \text{ and Install}(t) \text{ exists value}\} \tag{6}
$$

然后，时间 $t_{\text{install}}$ 处 Install 事件的值被定义为 $t_{n+1}$ 处的安装特征 $L(t_{n+1})$。

#### 3.1.4 亲和度特征

亲和度特征，描述为 $F(a_{n+1})$，反映用户对目标应用 $a_{n+1}$ 的倾向。由于用户对不同应用表现出不同的偏好，我们通过使用与该应用相关的历史点击次数（Click_counts($a_{n+1}$)）的以 2 为底的对数（$\log_2$）来量化用户对目标应用 $a_{n+1}$ 的亲和度：

$$
F(a_{n+1}) = \lfloor \log_2(\text{Click\_counts}(a_{n+1})) \rfloor \tag{7}
$$

### 3.2 学习

在本节中，我们介绍 DIEN[19] 的训练过程，这是一个经典的基于用户行为序列的 CTR 估计模型。使用构建的应用点击序列和三类特征的 DIEN 框架如图 2 所示。DIEN 的输入包括序列数据 ($a_{1:n}$, $g_{1:n}$, $s_{1:n}$)、目标数据 ($a_{n+1}$, $g_{n+1}$, $s_{n+1}$)、亲和度特征 ($F(a_{n+1})$)、预测时间 ($t_{n+1}$) 和上下文特征 ($H(t_{n+1})$, $W(t_{n+1})$, $L(t_{n+1})$)，输出为目标应用 $a_{n+1}$ 的点击率。



![图2](/Users/dazhang/PycharmProject/Papers/9-app/.picture/2024-CTR-RAD-Optimizing Smartphone App Usage Prediction-fig2.png)

图 2. 使用应用点击序列和三类特征的 DIEN 框架。

#### 3.2.1 高频应用抑制

在众多可用应用中，只有少数应用被非常频繁地使用，大多数应用使用频率适中或较低。因此，训练样本中目标应用 $a_{n+1}$ 的分布存在严重不平衡。在某种程度上，CTR-RAD 部分减轻了应用分布不平衡的负面影响，因为 CTR 估计模型仅使用两个标签（点击或未点击）进行训练。为了进一步缓解这一问题，我们提出了一种策略来抑制目标应用具有高频使用率的训练样本数量。这种抑制可以通过公式 8 实现：

$$
P(a_{n+1}) = \left(\sqrt{\frac{w(a_{n+1})}{\tau} + 1}\right) \cdot \frac{\tau}{w(a_{n+1})} \tag{8}
$$

其中 $P(a_{n+1})$ 表示选择目标应用为 $a_{n+1}$ 的样本作为正样本的概率。$w(a_{n+1})$ 表示目标应用 $a_{n+1}$ 的使用频率，使用公式 9 计算：

$$
w(a_{n+1}) = \frac{\text{Click\_counts}(a_{n+1})}{\text{All\_click\_counts}} \tag{9}
$$

其中 $\text{Click\_counts}(a_{n+1})$ 表示 $a_{n+1}$ 的历史点击次数，$\text{All\_click\_counts}$ 表示同一历史时期所有应用的总点击次数。总体而言，概率 $P(a_{n+1})$ 与目标应用的使用频率 $w(a_{n+1})$ 呈负相关。

#### 3.2.2 训练

如图 2 所示，嵌入层处理序列数据 ($a_{1:n}$, $g_{1:n}$, $s_{1:n}$) 以产生密集的嵌入表示，记为 $\mathbf{e}(1:n)_{n \times 3d}$，其中 $d$ 表示嵌入维度。同时，嵌入层处理上下文特征 ($H(t_{n+1})$, $W(t_{n+1})$, $L(t_{n+1})$)，生成嵌入表示 $\mathbf{e}(c)_{3d}$。类似地，嵌入层将目标数据 ($a_{n+1}$, $g_{n+1}$, $s_{n+1}$) 编码为嵌入表示 $\mathbf{e}(n+1)_{3d}$，而 $F(a_{n+1})$ 被编码为嵌入表示 $\mathbf{e}(f)_d$，$t_{n+1}$ 被表示为 $\mathbf{e}(t)_d$。

首先，序列数据嵌入 $\mathbf{e}(1:n)_{n \times 3d}$ 作为基于用户行为序列的 CTR 估计模型中序列网络的输入。在 DIEN 中，序列数据嵌入被馈送到 GRU 网络，该网络产生隐藏状态 $\mathbf{h}(1:n)_{n \times n_h}$（其中 $n_h$ 表示隐藏大小）。随后，隐藏状态 $\mathbf{h}(1:n)_{n \times n_h}$ 和目标数据嵌入 $\mathbf{e}(n+1)_{3d}$ 被馈送到注意力单元，该单元计算注意力分数 $\alpha(1:n)$。这些分数反映了目标数据与序列数据之间的关系，分数越高表示相关性越强。

接下来，隐藏状态 $\mathbf{h}(1:n)_{n \times n_h}$ 和注意力分数 $\alpha(1:n)$ 被输入到带有注意力更新门的 GRU（AUGRU，Attentional Update Gate GRU）[19]。AUGRU 将注意力机制与 GRU 无缝集成。AUGRU 的最后一个隐藏状态记为 $\mathbf{h}^{\prime}(n)$。

最后，向量 $\mathbf{h}^{\prime}(n)$、$\mathbf{e}(n+1)$、$\mathbf{e}(f)$、$\mathbf{e}(t)$ 和 $\mathbf{e}(c)$ 被连接形成 $\mathbf{x}$。连接后的向量 $\mathbf{x}$ 被馈送到 MLP 层进行最终预测。每个训练样本被标记为 $y \in \{0, 1\}$，其中 0 对应负样本，1 对应正样本。CTR 估计模型中常用的损失函数是负对数似然函数：

$$
\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} (y \log p(\mathbf{x}) + (1 - y) \log(1 - p(\mathbf{x}))) \tag{10}
$$

其中 $N$ 是训练样本数量，$p(\mathbf{x})$ 称为点击率，表示网络的最终输出，表示用户点击目标应用的预测概率。

### 3.3 预测

如图 1 所示，当第 3.2 节中训练好的模型部署到用户智能手机上时，它会预测每个已安装应用（$A = \{A_1, \ldots, A_T\}$，其中 $T$ 是用户智能手机上的应用总数）的点击率（$C = \{CTR(A_1), \ldots, CTR(A_T)\}$）。然后这些 CTR 值按降序排序，排名前 K 的 CTR 值（TopK($C$)）决定最终预测：

$$
PA = \{A_i | 1 \leq i \leq T \text{ and } CTR(A_i) \in \text{TopK}(C)\} \tag{11}
$$

其中 $PA$ 表示预测的前 K 个应用。

## 4 实验

在本节中，我们首先在第 4.1 节概述数据集、训练设置、评估指标和基线方法。然后，我们在第 4.2 节使用广泛的应用使用数据对 CTR-RAD 进行全面评估。

### 4.1 数据集和实现细节

#### 4.1.1 数据集

应用使用数据包含两个版本的数据集，分别在不同时间段收集，记为 V1-data 和 V2-data。这两个版本数据集的详细信息如表 2 所示。数据集中的每个应用被归类为 18 个特定类别之一，包括教育、游戏、工具、娱乐、健康、新闻、导航、社交、购物、音乐、金融、生活、体育、视频、工具、商务、旅行和摄影。值得注意的是，这些数据集排除了任何个人可识别信息。用户 ID 已被匿名化，所有用户元数据已被移除。

表 2：两个版本数据集概要

| 版本 | 时间段 | 用户数 | 应用数 |
|------|--------|--------|--------|
| V1-data | 2022/03/02-2022/04/27 | 50,000 | 76,460 |
| V2-data | 2023/02/05-2023/05/05 | 200,000 | 180,671 |

#### 4.1.2 训练设置

隐藏状态大小设置为 32，嵌入大小设置为 16。模型使用 Adam 优化器[6]训练 100 个 epoch，学习率为 0.001，批量大小为 200。

#### 4.1.3 评估指标

我们采用 top-K 准确率作为评估指标，这是应用使用预测领域广泛采用的度量。这里，K 表示预测应用（$PA$）的数量，详见第 3.3 节。top-K 准确率计算如下：

$$
\text{top-K accuracy} = \frac{N(a_{n+1} \in PA)}{N_{\text{ALL}}} \tag{12}
$$

其中 $N_{\text{ALL}}$ 表示测试样本总数，分子表示正确预测的测试样本数量。如果目标应用 $a_{n+1}$ 在 $PA$ 内，则测试样本被视为正确预测。在我们后续的实验中，我们主要关注 top-1 准确率（Top1）、top-3 准确率（Top3）和 top-5 准确率（Top5）。值得注意的是，智能手机制造商优先考虑 Top3 和 Top5 指标，因为向用户推荐 3 个或 5 个应用更有意义。为确保公平评估，我们进行了五次实验，然后计算平均值作为最终结果。

#### 4.1.4 基线方法

我们将 CTR-RAD 与以下基线方法进行比较：
1) MFU（Most Frequently Used，最常用），识别用户最常使用的应用。
2) MRU（Most Recently Used，最近使用），识别用户最近使用的应用。
3) DNN[17]，使用两个隐藏层预测目标应用。
4) RNN-Attention[17]，采用注意力机制学习每个时间步隐藏状态的权重。然后对这些隐藏状态的加权和应用 softmax 激活，并连接到全连接层以预测目标应用。
5) AppUsage2Vec[17]，这是一个经典的应用使用预测模型，结合了应用注意力机制和时间上下文。
6) SGFNN[13]，这是一个开创性的模型，将基于序列的方法与基于图的方法集成。

### 4.2 结果

#### 4.2.1 超参数研究

如表 3 所示，我们进行了一系列实验来确定超参数 $P = \{\tau, \text{t-slot}, \text{g-slot}, D_{\text{train}}, D_{F(a_{n+1})}, D, M\}$ 的最优值。我们为每个超参数探索不同的值（$V$），后续实验中选择的值在表 3 中以粗体突出显示。每个超参数的详细信息如下：

1) $\tau$：它是公式 8 的超参数，调整对高频目标应用训练样本的抑制程度。较大的 $\tau$ 表示抑制程度较低。我们选择值 0.1。

2) t-slot：它表示用于离散化预测时间戳的粒度，以分钟为单位。我们选择值 10 分钟。

3) g-slot：它表示用于离散化时间间隔特征的粒度，以分钟为单位。我们选择值 2 分钟。

4) $D_{\text{train}}$：它指的是用于训练的应用使用数据天数。我们观察到当应用使用数据天数增加到 14 天时结果趋于平稳。因此，我们使用 7 天的应用使用数据进行训练。具体而言，我们使用 2022 年 4 月 1 日至 4 月 7 日的数据进行训练，使用 2022 年 4 月 8 日的数据进行验证。为了评估模型的泛化能力，我们从 V1-data 和 V2-data 中各随机选择一天的测试数据。

5) $D_{F(a_{n+1})}$：它表示用于获取亲和度特征的应用使用数据历史天数。我们可以观察到当历史天数延长到 14 天时结果趋于平稳。因此，我们使用 2022 年 3 月 25 日至 3 月 31 日的 7 天历史应用使用数据来获取亲和度特征。在所有后续实验中，用户 ID 设置为 0。

6) $D$ 和 $M$：我们通过考虑窗口大小 $M$ 和持续时间限制 $D$（以分钟为单位）来生成训练样本。每个训练样本由总共 $M$ 个应用组成，确保窗口内应用的累计持续时间不超过 $D$。通过实验，我们发现将 $M$ 设置为 16、$D$ 设置为 15 可以最大化训练样本数量，产生最优结果。

表 3：超参数研究

| P | V | V1-data Top1 | V1-data Top3 | V1-data Top5 | V2-data Top1 | V2-data Top3 | V2-data Top5 |
|---|---|---|---|---|---|---|---|
| $\tau$ | 0.01 | 47.28 | 78.21 | 85.11 | 47.66 | 78.05 | 84.71 |
| | 0.05 | 48.28 | 78.65 | 85.35 | 48.63 | 78.48 | 84.89 |
| | **0.1** | **49.07** | **78.72** | **85.43** | **49.28** | **78.57** | **84.99** |
| t-slot | 15 | 48.84 | 78.73 | 85.43 | 49.12 | 78.62 | 85.04 |
| | **10** | **48.82** | **78.73** | **85.46** | **49.07** | **78.59** | **85.04** |
| | 5 | 48.93 | 78.73 | 85.43 | 49.15 | 78.59 | 85.01 |
| g-slot | 5 | 49.45 | 78.83 | 85.47 | 49.61 | 78.67 | 85.07 |
| | **2** | **49.85** | **79.01** | **85.59** | **50.03** | **78.80** | **85.14** |
| | 1 | 49.81 | 79.00 | 85.61 | 49.95 | 78.82 | 85.17 |
| $D_{\text{train}}$ | 1 | 45.94 | 77.84 | 84.78 | 46.39 | 77.91 | 84.52 |
| | **7** | **49.07** | **78.72** | **85.43** | **49.28** | **78.57** | **84.99** |
| | 14 | 49.97 | 78.78 | 85.45 | 49.96 | 78.70 | 85.07 |
| $D_{F(a_{n+1})}$ | 1 | 49.47 | 80.53 | 88.20 | 49.89 | 81.04 | 88.63 |
| | **7** | **49.55** | **80.76** | **88.59** | **49.90** | **81.19** | **88.89** |
| | 14 | 49.43 | 80.78 | 88.59 | 49.90 | 81.19 | 88.85 |
| $D$ | 8 | 49.48 | 78.98 | 85.54 | 49.39 | 78.42 | 84.73 |
| | **15** | **49.07** | **78.72** | **85.43** | **49.28** | **78.57** | **84.99** |
| | 30 | 47.90 | 78.36 | 85.23 | 48.55 | 78.44 | 84.89 |
| | 60 | 47.13 | 77.99 | 85.05 | 47.81 | 78.14 | 84.81 |
| $M$ | 8 | 48.67 | 78.61 | 85.21 | 48.78 | 78.33 | 84.66 |
| | **16** | **49.07** | **78.72** | **85.43** | **49.28** | **78.57** | **84.99** |
| | 32 | 48.88 | 78.64 | 85.37 | 49.12 | 78.51 | 84.94 |

#### 4.2.2 消融研究

为了突出不同特征的影响，我们将 CTR-RAD 与其变体进行比较，结果如表 4 所示。此外，图 3 直观地描绘了使用纯应用点击序列输入（$a_{1:n}$）与结合每个特征的输入之间的增益差异。两个结果都表明每个特征在不同程度上提升了性能。具体而言，上下文特征通常影响较弱，而序列特征对 Top1 有积极影响，亲和度特征显著增强了 Top3 和 Top5。此外，所有特征组合作为输入在所有指标上表现最佳。

表 4：不同特征的结果

| 特征 | V1-data Top1 | V1-data Top3 | V1-data Top5 | V2-data Top1 | V2-data Top3 | V2-data Top5 |
|------|---|---|---|---|---|---|
| $a_{1:n}$ | 48.87 | 78.69 | 85.42 | 49.11 | 78.57 | 85.02 |
| +headset | 48.98 | 78.74 | 85.49 | 49.21 | 78.61 | 85.08 |
| +WiFi | 48.92 | 78.77 | 85.46 | 49.13 | 78.63 | 85.04 |
| +install | 48.95 | 78.69 | 85.43 | 49.15 | 78.57 | 85.04 |
| +screen | 49.93 | 78.87 | 85.56 | 50.13 | 78.70 | 85.12 |
| +time gap | 49.95 | 79.00 | 85.58 | 50.11 | 78.82 | 85.15 |
| +sequence | 50.76 | 79.17 | 85.71 | 50.88 | 78.94 | 85.27 |
| +context | 49.11 | 78.74 | 85.50 | 49.33 | 78.62 | 85.12 |
| +affinity | 49.50 | 80.74 | 88.59 | 49.90 | 81.24 | 88.94 |
| CTR-RAD | 51.59 | 81.37 | 88.95 | 51.90 | 81.78 | 89.23 |



![图3](/Users/dazhang/PycharmProject/Papers/9-app/.picture/2024-CTR-RAD-Optimizing Smartphone App Usage Prediction-fig3.png)

图 3. 不同特征的增益差异。

为了进一步研究特征交互对性能的贡献，我们研究了使用不同特征组合时模型性能的变化。表 6 总结了结果：'0' 表示性能不如单个特征，'1' 表示性能优于单个特征，'2' 表示性能优于两个单个特征。我们发现，将屏幕特征与除安装特征外的其他特征组合始终产生优越的结果。此外，安装和亲和度特征的组合表现出增强的性能，可能是因为安装特征弥补了新安装应用历史偏好数据的不足。然而，将安装特征与不相关的 WiFi 特征组合会产生更差的结果。

表 6：不同特征组合的性能

| 特征 | headset | WiFi | install | screen | time gap | affinity |
|------|---------|------|---------|--------|----------|----------|
| headset | - | 2 | 1 | 2 | 1 | 1 |
| WiFi | 2 | - | 0 | 2 | 1 | 2 |
| install | 1 | 0 | - | 1 | 2 | 2 |
| screen | 2 | 2 | 1 | - | 2 | 2 |
| time gap | 1 | 1 | 2 | 2 | - | 2 |
| affinity | 1 | 2 | 2 | 2 | 2 | - |

#### 4.2.3 与基线方法的比较结果

在表 5 中，我们将 CTR-RAD 与基线方法在两个数据集上的性能进行了比较。结果证明了 CTR-RAD 在 Top1、Top3 和 Top5 指标上的优越性能，在 Top3 和 Top5 指标上的改进尤为显著。具体而言，CTR-RAD 在 Top3 上超越最先进方法 4.93%，在 Top5 上超越 6.64%。值得注意的是，仅依赖应用点击序列输入的 CTR-RAD（无特征）在 Top3 和 Top5 上仍然优于包含额外特征的基线方法。

表 5：与基线方法的性能比较

| 方法 | V1-data Top1 | V1-data Top3 | V1-data Top5 | V2-data Top1 | V2-data Top3 | V2-data Top5 |
|------|---|---|---|---|---|---|
| MFU | 36.18 | 70.97 | 76.80 | 36.33 | 70.62 | 76.09 |
| MRU | 22.29 | 66.45 | 74.13 | 22.71 | 65.85 | 73.56 |
| DNN [17] | 48.30 | 76.70 | 82.58 | 48.48 | 76.26 | 82.00 |
| RNN-Attention [17] | 50.39 | 78.51 | 84.40 | 50.33 | 77.94 | 83.68 |
| AppUsage2Vec [17] | 42.27 | 74.95 | 81.90 | 42.65 | 74.56 | 81.22 |
| SGFNN [13] | 49.59 | 77.39 | 83.51 | 49.89 | 76.99 | 82.86 |
| CTR-RAD（无特征） | 48.87 | 78.69 | 85.42 | 49.11 | 78.57 | 85.02 |
| CTR-RAD | 51.59 | 81.37 | 88.95 | 51.90 | 81.78 | 89.23 |
| 提升（%） | 2.37 | 3.65 | 5.40 | 3.12 | 4.93 | 6.64 |

此外，我们在不同使用频率的应用上对 CTR-RAD 和基线方法进行了比较分析，如图 4 所示。x 轴表示应用使用频率的以 2 为底的对数（$\log_2$），通过公式 9 计算，值越大表示应用使用频率越高。通过比较柱状图，可以明显看出 CTR-RAD 超越了分类模型，在预测低频应用方面实现了约两倍的准确率。通过比较折线图，可以观察到随着应用频率的增加，CTR-RAD 的 top-5 准确率在所有频率上保持相对稳定。总之，分类模型在预测高频应用方面表现出色，但在预测中低频应用方面面临挑战。然而，CTR-RAD 在整个频率范围内表现出均衡的性能。



![图4](/Users/dazhang/PycharmProject/Papers/9-app/.picture/2024-CTR-RAD-Optimizing Smartphone App Usage Prediction-fig4.png)

图 4. 不同应用使用频率下的性能比较。

#### 4.2.4 案例研究

在本节中，我们对每个特征进行全面分析，强调每个特征如何影响 CTR-RAD 预测哪些应用、类别和应用使用频率。我们将结合上下文特征 $H(t_{n+1})$、$W(t_{n+1})$ 和 $L(t_{n+1})$ 的 top 1 预测结果与仅从应用点击序列输入获得的结果进行比较。

图 5-(a) 说明了通过结合 $H(t_{n+1})$ 特征准确预测的应用百分比，而仅使用应用点击序列输入预测不准确的应用百分比。可以观察到，当结合耳机特征时，音乐或视频应用（例如酷狗音乐）可以被更准确地预测。这一发现符合用户的日常习惯，表明他们在使用这些应用时通常佩戴耳机。

图 5-(b) 展示了通过结合 $W(t_{n+1})$ 特征准确预测的应用百分比，而仅依赖应用点击序列输入预测不准确的应用百分比。可以发现，当结合 WiFi 特征时，视频应用（例如抖音）可以被更准确地预测。这符合用户的习惯，因为他们通常使用 WiFi 运行资源密集型应用。

图 5-(c) 显示了通过结合 $L(t_{n+1})$ 特征准确预测的应用百分比，但仅依赖应用点击序列输入预测不准确的应用百分比。结果表明，当结合安装特征时，工具或购物应用（例如拼多多、钉钉）可以被更准确地预测。这与我们的日常习惯一致，因为用户通常在安装后立即点击这些应用。



![图5](/Users/dazhang/PycharmProject/Papers/9-app/.picture/2024-CTR-RAD-Optimizing Smartphone App Usage Prediction-fig5.png)

图 5. 上下文特征分析。

图 3 表明亲和度特征对 Top3 和 Top5 有显著影响。因此，我们将结合亲和度特征 $F(a_{n+1})$ 的 top 5 预测结果与仅从应用点击序列输入获得的结果进行比较。图 6 展示了纯应用点击序列输入与结合亲和度特征输入之间的 top-5 准确率差异。它表明当结合亲和度特征时，新闻和游戏应用可以被更准确地预测。这些类别体现了用户的偏好；例如，某些用户群体更喜欢游戏应用，而其他用户群体则更喜欢新闻相关应用。



![图6](/Users/dazhang/PycharmProject/Papers/9-app/.picture/2024-CTR-RAD-Optimizing Smartphone App Usage Prediction-fig6.png)

图 6. 亲和度特征的 Top5 增益差异。

同时，如图 3 所示，序列特征对 Top1 的影响相对较大， compared to Top3 and Top5。因此，我们将结合序列特征 $s_{1:n}$ 和 $g_{1:n}$ 的 top 1 预测结果与仅从应用点击序列输入获得的结果进行比较。图 7 展示了在不同应用使用频率下，纯应用点击序列输入与结合序列特征输入之间的 top-1 准确率差异。它表明序列特征在预测中频应用方面带来了稳定的改进。



![图7](/Users/dazhang/PycharmProject/Papers/9-app/.picture/2024-CTR-RAD-Optimizing Smartphone App Usage Prediction-fig7.png)

图 7. 序列特征的 Top1 增益差异。

#### 4.2.5 复杂度分析

CTR-RAD 的算法复杂度约为 $O(RN)$，其中 $R$ 表示记录数量，$N$ 表示输入应用数量。通过实证分析，如表 7 和表 3 所示，我们比较了 $D_{\text{train}}=7$ 和 $D_{\text{train}}=14$ 之间的性能。尽管 $D_{\text{train}}=14$ 时模型大小和训练时间都有显著增加，但我们观察到性能仅有轻微提升。这表明 $D_{\text{train}}=7$ 时模型性能和训练成本之间存在有利的权衡。

表 7：不同数据集大小的成本

| $D_{\text{train}}$ | 记录数 | 模型大小（MB） | 训练时间（h） |
|---|---|---|---|
| 1 | 660,506 | 15.6 | 0.92 |
| 7 | 4,349,482 | 15.7 | 2.28 |
| 14 | 8,827,062 | 16.0 | 3.95 |

图 8 说明了不同模型的实证推理时间（对于 10,000 条记录）。我们的方法专门预测目标应用的 CTR，与基线方法相比表现出明显更低的推理时间。此外，尽管需要预测用户手机上所有已安装应用（平均约 60 个）的 CTR，但我们可以通过并行执行预测来提高效率。



![图8](/Users/dazhang/PycharmProject/Papers/9-app/.picture/2024-CTR-RAD-Optimizing Smartphone App Usage Prediction-fig8.png)

图 8. 不同模型的推理时间。

#### 4.2.6 未见过的应用预测

在实际部署期间，我们为潜在新引入的应用分配 300 个特征值。根据我们的配置经验，每个用户引入的新应用数量通常保持在该阈值内。然而，尽管为分类模型分配了相同数量的特征值，它们仍然无法预测未见过的应用。我们的模型使用目标应用本身作为输入来预测目标应用的 CTR。如果目标应用是新引入的，我们提出的方法仍然可以基于输入应用点击序列（例如，如果新应用在用户历史中被点击过）和特征（例如，如果新应用在屏幕点亮后立即被点击）来预测其 CTR。

表 8 展示了 CTR-RAD（在 V1-data 上训练）在预测 V2-data 中目标应用为新引入应用的记录（共 24,086 条）时的性能。结果表明，我们的方法仅使用应用点击序列作为输入就能准确预测这些应用，并且通过结合额外特征进一步提升了性能。

表 8：未见过的应用预测结果

| 包含未见目标应用的记录 | Top1 | Top3 | Top5 |
|---|---|---|---|
| CTR-RAD（无特征） | 24.07 | 59.05 | 67.96 |
| CTR-RAD | 27.44 | 72.06 | 83.35 |

#### 4.2.7 局限性分析

CTR-RAD 的开发依赖于从原始用户行为数据中提取和构建的初始特征。我们提出的方法的两个潜在局限性是：1) 当前的特征工程严重依赖专业人员的专业知识；2) 部分提取的特征是特定于平台的，这可能会增加将模型扩展到新应用平台的成本。

## 5 结论

在本文中，我们得出结论：现有研究通常将应用使用预测视为分类问题，遇到了应用使用不平衡和部署期间分布外问题等挑战。为解决这些局限性，我们引入了一种创新的基于点击率排序的应用使用预测方法。我们的方法涉及使用应用点击序列和三类不同的特征在云端训练 CTR 估计模型。然后使用训练好的模型预测用户智能手机上每个已安装应用的 CTR。决策过程包括对这些 CTR 值进行排序，并选择点击率最高的应用作为最终预测。广泛的实验和分析表明，我们提出的方法优于最先进方法，在预测低使用频率应用方面实现了约两倍的准确率。此外，该方法已成功部署在领先智能手机制造商的应用推荐系统中。未来，我们将通过结合额外数据（如兴趣点（POI）数据和物理活动事件（例如跑步、步行））来进一步增强我们的特征工程。我们还旨在将我们的方法与其他先进的点击率估计模型联合，以获得更好的预测结果。

## 致谢

本工作部分得到国家自然科学基金资助（项目编号 62206074 和 62072137）、深圳市高校稳定支持计划资助（项目编号 GXWD20220811173233001）以及国家重点研发计划资助（项目编号 2023YFB4503100）。

## 参考文献

[1] 2024. Number of available applications in the Google Play Store from December 2009 to December 2023. Retrieved June 9, 2024 from https://www.statista.com/statistics/266210/number-of-available-applications-in-the-google-play-store/

[2] Ricardo Baeza-Yates, Di Jiang, Fabrizio Silvestri, and Beverly Harrison. 2015. Predicting the next app that you are going to use. In ACM International Conference on Web Search and Data Mining (WSDM). 285–294.

[3] Xinlei Chen, Yu Wang, Jiayou He, Shijia Pan, Yong Li, and Pei Zhang. 2019. CAP: Context-aware app usage prediction with heterogeneous graph embedding. ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies (IMWUT) 3, 1 (2019), 1–25.

[4] Trinh Minh Tri Do and Daniel Gatica-Perez. 2014. Where and what: Using smartphones to predict next locations and applications in daily life. Pervasive and Mobile Computing (PMC) 12 (2014), 79–91.

[5] Yonchanok Khaokaew, Mohammad Saiedur Rahaman, Ryen W White, and Flora D Salim. 2021. Cosem: Contextual and semantic embedding for app usage prediction. In ACM International Conference on Information and Knowledge Management (CIKM). 3137–3141.

[6] Diederik P Kingma and Jimmy Ba. 2015. Adam: A method for stochastic optimization. International Conference on Learning Representations (ICLR) (2015).

[7] Younghoon Lee, Sungzoon Cho, and Jinhae Choi. 2019. App usage prediction for dual display device via two-phase sequence modeling. Pervasive and Mobile Computing (PMC) 58 (2019), 101025.

[8] Tong Li, Tong Xia, Huandong Wang, Zhen Tu, Sasu Tarkoma, Zhu Han, and Pan Hui. 2022. Smartphone app usage analysis: Datasets, methods, and applications. IEEE Communications Surveys and Tutorials (COMST) 24, 2 (2022), 937–966.

[9] Nagarajan Natarajan, Donghyuk Shin, and Inderjit S Dhillon. 2013. Which app will you use next? collaborative filtering with interactional context. In ACM Conference on Recommender Systems (RecSys). 201–208.

[10] Yi Ouyang, Bin Guo, Qianru Wang, Yunji Liang, and Zhiwen Yu. 2022. Learning dynamic app usage graph for next mobile app recommendation. IEEE Transactions on Mobile Computing (TMC) (2022).

[11] Zhihao Shen, Xi Zhao, and Jianhua Zou. 2023. GinApp: An Inductive Graph Learning based Framework for Mobile Application Usage Prediction. In IEEE International Conference on Computer Communications (INFOCOM). IEEE, 1–10.

[12] Choonsung Shin, Jin-Hyuk Hong, and Anind K Dey. 2012. Understanding and prediction of mobile application usage for smart phones. In ACM Conference on Ubiquitous Computing (UbiComp). 173–182.

[13] Yizhuo Wang, Renhe Jiang, Hangchen Liu, Du Yin, and Xuan Song. 2023. Sequence-Graph Fusion Neural Network for User Mobile App Behavior Prediction. In Joint European Conference on Machine Learning and Knowledge Discovery in Databases (ECML PKDD). Springer, 105–121.

[14] Tong Xia, Yong Li, Jie Feng, Depeng Jin, Qing Zhang, Hengliang Luo, and Qingmin Liao. 2020. DeepApp: Predicting personalized smartphone app usage via context-aware multi-task learning. ACM Transactions on Intelligent Systems and Technology (TIST) 11, 6 (2020), 1–12.

[15] Shijian Xu, Wenzhong Li, Xiao Zhang, Songcheng Gao, Tong Zhan, and Sanglu Lu. 2020. Predicting and recommending the next smartphone apps based on recurrent neural network. CCF Transactions on Pervasive Computing and Interaction (CCF TPCI) 2, 4 (2020), 314–328.

[16] Yue Yu, Tong Xia, Huandong Wang, Jie Feng, and Yong Li. 2020. Semantic-aware spatio-temporal app usage representation via graph convolutional network. ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies (IMWUT) 4, 3 (2020), 1–24.

[17] Sha Zhao, Zhiling Luo, Ziwen Jiang, Haiyan Wang, Feng Xu, Shijian Li, Jianwei Yin, and Gang Pan. 2019. AppUsage2Vec: Modeling smartphone app usage for prediction. In IEEE International Conference on Data Engineering (ICDE). IEEE, 1322–1333.

[18] Xiaoxing Zhao, Yuanyuan Qiao, Zhongwei Si, Jie Yang, and Anders Lindgren. 2016. Prediction of user app usage behavior from geo-spatial data. In International ACM SIGMOD Workshop on Managing and Mining Enriched Geo-Spatial Data. 1–6.

[19] Guorui Zhou, Na Mou, Ying Fan, Qi Pi, Weijie Bian, Chang Zhou, Xiaoqiang Zhu, and Kun Gai. 2019. Deep interest evolution network for click-through rate prediction. In AAAI Conference on Artificial Intelligence (AAAI), Vol. 33. 5941–5948.

[20] Hengshu Zhu, Enhong Chen, Hui Xiong, Kuifei Yu, Huanhuan Cao, and Jilei Tian. 2014. Mining mobile user preferences for personalized context-aware recommendation. ACM Transactions on Intelligent Systems and Technology (TIST) 5, 4 (2014), 1–27.

[21] Xun Zou, Wangsheng Zhang, Shijian Li, and Gang Pan. 2013. Prophet: What app you wish to use next. In ACM Conference on Pervasive and Ubiquitous Computing Adjunct Publication (UbiComp Adjunct). 167–170.
