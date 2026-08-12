# ESCM2：基于全空间反事实多任务模型的点击后转化率预估

> Hao Wang, Tai-Wei Chang, Tianqiao Liu, Jianmin Huang, Zhichao Chen, Chao Yu, Ruopeng Li, Wei Chu | Ant Group



本文提出了 ESCM2（Entire Space Counterfactual Multi-task Model），一种基于反事实风险最小化的全空间多任务模型，用于解决点击后转化率（CVR）预估中的样本选择偏差和数据稀疏问题。核心内容：

- 从理论角度严格证明了 ESMM 存在的两个关键问题：固有估计偏差（IEB）和潜在独立优先（PIP）
- 提出将反事实风险最小化（CRM）作为正则化器引入 ESMM，同时解决 IEB 和 PIP 问题
- 在离线数据集和在线环境中进行了大量实验验证

关键发现：
- ESMM 的 CVR 估计值在理论上总是高于真实值，存在固有估计偏差
- ESMM 的 CTR 和 CVR 估计容易产生条件独立性，导致潜在独立优先问题
- ESCM2 通过引入逆倾向得分（IPS）和双重鲁棒（DR）估计器，有效消除了上述偏差

---



## 摘要

准确预估点击后转化率（CVR，Post-click Conversion Rate）对于构建推荐系统至关重要，该问题长期面临样本选择偏差和数据稀疏的挑战。全空间多任务模型（ESMM，Entire Space Multi-task Model）系列方法利用用户行为的序列模式（即曝光 → 点击 → 转化）来解决数据稀疏问题。然而，它们仍然无法确保 CVR 估计的无偏性。在本文中，我们从理论上证明了 ESMM 存在以下两个问题：（1）CVR 估计的固有估计偏差（IEB，Inherent Estimation Bias），即 CVR 估计值固有地高于真实值；（2）CTCVR 估计的潜在独立优先（PIP，Potential Independence Priority），即 ESMM 可能忽略从点击到转化的因果关系。为此，我们设计了一种基于原则的方法——全空间反事实多任务建模（ESCM2），该方法采用反事实风险最小化器作为 ESMM 的正则化器，同时解决 IEB 和 PIP 问题。在离线数据集和在线环境中的大量实验表明，我们提出的 ESCM2 能够大幅缓解固有的 IEB 和 PIP 问题，并取得优于基线模型的性能。

**关键词**：Recommender System; Entire Space Multi-task Learning; Selection Bias; Post-click Conversion Rate Estimation



## 1. 引言

推荐系统旨在从大量候选 item 中向用户传递有价值的 item [7, 12]，这一直是电子商务 [14]、社交媒体 [4] 和广告 [28] 领域用户增长的主要驱动力。图 1 展示了在工业环境中构建推荐系统的两阶段流程。在离线阶段，排序模型使用从用户日志中解析的用户画像特征、item 特征和用户-item 交互特征进行训练。在在线阶段，我们依赖多种排序指标，包括但不限于点击率（CTR，Click-Through Rate）、点击后转化率（CVR，Post-click Conversion Rate）和点击转化率（CTCVR，Click-Through & Conversion Rate），向用户展示可能引起其兴趣的 item。用户反馈在离线训练阶段用于优化推荐系统的性能。

<!-- 图 1：电子商务推荐系统示意图 -->

在电子商务环境中，典型的用户行为路径可以概括为曝光 → 点击 → 转化 [14, 16]。如图 2 所示，CVR 表示从点击空间到转化空间的转换概率。一种构建有效 CVR 估计器的简单方法是在点击空间上训练，因为转化响应在点击空间中是完全可用的。

<!-- 图 2：CVR 预估中的样本选择偏差和数据稀疏问题示意图 -->

Ma 等人 [14] 和 Zhang 等人 [27] 报告了挑战简单 CVR 估计器的两个关键问题。第一个问题是样本选择偏差，因为训练空间仅由被点击的 item 组成。具体来说，CVR 较低的 item 不太可能被点击，即被包含在训练空间中，反之亦然 [15]，这使得训练空间成为非随机缺失（MNAR，Missing Not At Random）。换句话说，训练空间 $\mathcal{O}$ 和推理空间 $\mathcal{D}$ 之间存在分布偏移。另一个问题源于被点击样本的数据稀疏性，我们在生产数据集上的 CTR 约为 3.8%，在 Ali-CCP 数据集上约为 4%。由于简单的 CVR 模型是在点击空间上训练的，这两个问题都严重阻碍了它们在点击空间之外的泛化能力。

Ma 等人提出了全空间多任务模型（ESMM）来解决样本选择偏差和数据稀疏问题。诚然，它通过参数共享有效地缓解了数据稀疏问题 [14, 16, 24]；然而，其 CVR 估计的无偏性仍然无法保证。新出现的经验证据 [27] 表明 ESMM 的 CVR 估计是有偏差的，引发了对该方法的关注。

在本文中，我们在第 3 节报告了 ESMM 的两个关键问题。

- **固有估计偏差（IEB）**：我们严格证明了 ESMM 的 CVR 估计值在非常宽松的条件下也固有地高于真实值。
- **潜在独立优先（PIP）**：我们证明了 ESMM 的 CTR 和 CVR 估计容易产生条件独立性，这是不理想的。

利用因果方法，我们提出了全空间反事实多任务模型（ESCM2），该模型引入了反事实风险最小化器（CRM，Counterfactual Risk Minimizer），即逆倾向得分（IPS，Inverse Propensity Score）和双重鲁棒（DR，Doubly Robust），来正则化 ESMM 的 CVR 估计。正如我们将在第 4 节讨论的，引入的 CRM 正则化器分别使 CVR 和 CTCVR 估计免受 IEB 和 PIP 问题的影响。

本文的贡献总结如下：

- 这是第一项严格证明 ESMM CVR 估计固有偏差的工作。提供了数学证明和实验结果来支持这一声明。
- 我们表明 ESMM 的 CTCVR 估计受到潜在独立优先的影响。我们设计了实验来支持这一声明。
- 我们提出了 ESCM2，这是第一项从因果角度改进 ESMM 的工作。ESCM2 有效消除了 ESMM 中的 IEB 和 PIP。提供了大量实验结果和数学证明来验证我们的声明。



## 2. 预备知识

### 2.1 符号说明

我们使用大写字母（如 $O$）表示随机变量，使用小写字母（如 $o$）表示相应的特定值。花体字母（如 $\mathcal{O}$）表示相应随机变量的样本空间，$\mathrm{P}()$ 表示随机变量的概率分布，例如 $\mathrm{P}(O)$。

### 2.2 问题形式化

令 $\mathcal{U} = \{u_1, u_2, \ldots, u_m\}$ 表示曝光空间上的 $m$ 个用户集合。令 $\mathcal{I} = \{i_1, i_2, \ldots, i_n\}$ 表示曝光空间上的 $n$ 个 item 集合，令 $\mathcal{D} = \mathcal{U} \times \mathcal{I}$ 表示曝光空间上用户集合和 item 集合的笛卡尔积。令 $\mathcal{O}$ 为点击矩阵，其中每个元素 $o_{u,i} \in \{0, 1\}$ 表示用户 $u$ 和 item $i$ 之间是否发生点击，$R \in \{0, 1\}^{m \times n}$ 为观测到的转化标签，其中每个元素 $r_{u,i} \in \{0, 1\}$ 表示用户 $u$ 和 item $i$ 之间是否发生转化。

如果 $R$ 完全可观测，理想损失函数定义为：

$$
\mathcal{P} := \mathbb{E}_{(u,i) \in \mathcal{D}} \left( \delta(r_{u,i}, \hat{r}_{u,i}) \right)
\qquad (1)
$$

其中 $\delta$ 表示 $r_{u,i}$ 的预测误差，$\hat{r}_{u,i}$ 表示 $r_{u,i}$ 的预测值。对于 $\delta$，我们使用二元交叉熵损失：

$$
\delta(r_{u,i}, \hat{r}_{u,i}) := -r_{u,i} \log \hat{r}_{u,i} - (1 - r_{u,i}) \log(1 - \hat{r}_{u,i})
\qquad (2)
$$

然而，$r_{u,i}$ 只能在点击空间 $\mathcal{O}$ 中的用户-item 对中被观测到。因此，一种简单的方法 [18, 22] 使用 $\mathcal{O}$ 上的期望来估计理想损失：

$$
\mathcal{L}_{\text{naive}} = \mathbb{E}_{(u,i) \in \mathcal{O}}(\delta_{u,i}) = \frac{1}{|\mathcal{O}|} \sum_{(u,i) \in \mathcal{D}} (o_{u,i} \delta_{u,i})
\qquad (3)
$$

其中 $|\mathcal{O}| = \sum_{(u,i) \in \mathcal{D}} (o_{u,i})$。这种方法被许多现有方法广泛采用，但会导致有偏估计，即 $\mathbb{E}_{\mathcal{O}}(\mathcal{L}_{\text{naive}}) \neq \mathcal{P}$。

### 2.3 全空间多任务模型方法

全空间多任务模型方法（ESMM）[14] 使用链式法则间接获得 CVR 估计：

$$
\mathrm{P}(r_{u,i} = 1 | o_{u,i} = 1) = \frac{\mathrm{P}(r_{u,i} = 1, o_{u,i} = 1)}{\mathrm{P}(o_{u,i} = 1)}
\qquad (4)
$$

在 ESMM 中，使用两个塔分别预测 CTR（即 $\mathrm{P}(o_{u,i} = 1)$）和 CVR（即 $\mathrm{P}(r_{u,i} = 1 | o_{u,i} = 1)$）。这两个塔的乘积给出 CTCVR 估计（即 $\mathrm{P}(r_{u,i} = 1, o_{u,i} = 1)$）。在训练阶段，ESMM 在整个曝光空间 $\mathcal{D}$ 上最小化 CTR 和 CTCVR 估计的经验风险：

$$
\mathcal{L}_{\text{CTR}} = \mathbb{E}_{(u,i) \in \mathcal{D}} \left( \delta(o_{u,i}, \hat{o}_{u,i}) \right)
\qquad (5)
$$

$$
\mathcal{L}_{\text{CTCVR}} = \mathbb{E}_{(u,i) \in \mathcal{D}} \left( \delta(o_{u,i} * r_{u,i}, \hat{o}_{u,i} * \hat{r}_{u,i}) \right)
\qquad (6)
$$

### 2.4 基于倾向得分的方法

逆倾向得分（IPS）估计器 [18] 使用 $1/q_{u,i}$（倾向得分的倒数，在我们的案例中是 CTR）对每个误差项 $\delta_{u,i}$ 进行加权，以对齐点击空间和曝光空间上的误差分布。调整后的误差项为：

$$
\mathcal{L}_{\text{IPS}} = \frac{1}{|\mathcal{D}|} \sum_{(u,i) \in \mathcal{D}} \frac{o_{u,i} \delta_{u,i}}{q_{u,i}} = \frac{1}{|\mathcal{D}|} \sum_{(u,i) \in \mathcal{D}} \frac{o_{u,i} \delta_{u,i}}{\hat{q}_{u,i}}
\qquad (9)
$$

由于真实值 $q_{u,i}$ 始终不可用，引入辅助分类器来估计倾向得分 $q_{u,i}$，用 $\hat{q}_{u,i}$ 表示。给定估计的 $\hat{q}_{u,i}$ 是准确的，IPS 估计器给出理想损失函数的无偏估计，即 $\mathbb{E}_{\mathcal{O}}(\mathcal{L}_{\text{IPS}}) = \mathcal{P}$ [18]。然而，IPS 估计器中的倾向得分可能受到严重高方差的影响，因此引入了双重鲁棒（DR）估计器 [22]。特别地，DR 引入插补误差 $\hat{\delta}_{u,i}$ 来建模 $\mathcal{D}$ 中所有事件的预测误差，并对点击事件校正误差偏差 $\hat{e}_{u,i} = \delta_{u,i} - \hat{\delta}_{u,i}$：

$$
\mathcal{L}_{\text{DR}} = \frac{1}{|\mathcal{D}|} \sum_{(u,i) \in \mathcal{D}} \left( \hat{\delta}_{u,i} + \frac{o_{u,i} \hat{e}_{u,i}}{\hat{q}_{u,i}} \right)
\qquad (7)
$$

其中 $\hat{q}_{u,i}$ 旨在消除 $\hat{e}_{u,i}$ 的 MNAR 效应。双重鲁棒性源于以下事实：只要插补误差或倾向得分是准确的（但不一定两者都准确），就能保证无偏性。$\hat{\delta}_{u,i}$ 和 $\hat{q}_{u,i}$ 的准确性通常由辅助任务确保。



## 3. 关于 ESMM 的讨论

### 3.1 ESMM 是无偏 CVR 估计器吗？

研究人员已经意识到 ESMM 中存在固有估计偏差（IEB）[27]；然而，据我们所知，其 CVR 估计偏差的理论证明仍然缺乏。在本文中，IEB 在定理 1 中进行了形式化和证明。

**定理 1**：令随机变量 $O$、$R$、$C$ 分别为点击、点击后转化和点击转化的指示器，$o_{u,i}$、$r_{u,i}$、$c_{u,i}$ 为给定用户-item 对时 $O$、$R$、$C$ 的相应值，$\hat{o}_{u,i}$、$\hat{r}_{u,i}$、$\hat{c}_{u,i}$ 为 $o_{u,i}$、$r_{u,i}$、$c_{u,i}$ 的预测值。ESMM 的 CVR 估计在曝光空间 $\mathcal{D}$ 上的偏差始终大于零：

$$
\text{Bias}_{\text{ESMM}} := \mathbb{E}_{\mathcal{D}}(\hat{R}) - \mathbb{E}_{\mathcal{D}}(R) > 0
$$

**证明**：根据公式 (5) 中的损失函数，训练良好的 ESMM 模型确保：

$$
\mathbb{E}_{\mathcal{D}}(O - \hat{O}) = \int (o_{u,i} - \hat{o}_{u,i}) d(u, i) = 0
$$
$$
\mathbb{E}_{\mathcal{D}}(C - \hat{C}) = \int (c_{u,i} - \hat{c}_{u,i}) d(u, i) = 0
$$

注意 $\mathbb{E}_{\mathcal{D}}(R)$ 和 $\mathbb{E}_{\mathcal{D}}(\hat{R})$ 分别是 CVR 真实值和估计值的期望。CVR 估计偏差：

$$
\text{Bias}_{\text{ESMM}} = \mathbb{E}_{\mathcal{D}}(\hat{R}) - \mathbb{E}_{\mathcal{D}}(R)
$$

根据图 2 中的标签分布规则，从点击空间到曝光空间的真实 CVR 期望为：

$$
\mathbb{E}_{\mathcal{D}}(R) > \mathbb{E}_{\mathcal{O}}(R)
$$

这源于用户在点击空间中更有可能被转化 [18]，即 $\mathbb{E}_{\mathcal{O}}(R) > \mathbb{E}_{\mathcal{D}}(R)$。

在推理阶段，ESMM 使用 CVR 塔的输出作为其预测的 CVR。这种方法通过不在点击空间上建模 CVR 来规避样本选择偏差问题。然而，正如第 4.1 节所述，这种方法会导致 CVR 的固有高估。为此，我们寻求开发一种无偏的 CVR 估计器来适当解决样本选择偏差问题。



## 4. 提出的方法

### 4.1 反事实风险正则化器

#### 4.1.1 逆倾向得分正则化器

IPS 估计器的核心思想是通过倾向得分对误差项进行加权，以对齐不同空间上的误差分布。在我们的场景中，倾向得分是 CTR，即 $\mathrm{P}(o_{u,i} = 1)$。

对于 CVR 估计任务，我们定义 IPS 正则化器为：

$$
\mathcal{L}_{\text{IPS-CVR}} = \frac{1}{|\mathcal{O}|} \sum_{(u,i) \in \mathcal{O}} \frac{\delta_{u,i}}{\hat{q}_{u,i}}
\qquad (15)
$$

其中 $\hat{q}_{u,i}$ 是倾向得分的估计值（即 CTR 估计）。

#### 4.1.2 双重鲁棒正则化器

DR 估计器结合了插补方法和倾向得分方法的优点。它引入插补误差 $\hat{\delta}_{u,i}$ 来建模所有事件的预测误差，并对点击事件校正误差偏差：

$$
\mathcal{L}_{\text{DR-CVR}} = \frac{1}{|\mathcal{D}|} \sum_{(u,i) \in \mathcal{D}} \left( \hat{\delta}_{u,i} + \frac{o_{u,i} (\delta_{u,i} - \hat{\delta}_{u,i})}{\hat{q}_{u,i}} \right)
\qquad (16)
$$

双重鲁棒性源于以下事实：只要插补误差或倾向得分是准确的（但不一定两者都准确），就能保证无偏性。

### 4.2 ESCM2 模型

我们提出的 ESCM2 模型将 CRM 正则化器集成到 ESMM 框架中。具体来说，ESCM2 的学习目标为：

$$
\mathcal{L}_{\text{ESCM2}} = \mathcal{L}_{\text{CTR}} + \lambda_{\text{c}} \mathcal{L}_{\text{CTCVR}} + \lambda_{\text{r}} \mathcal{L}_{\text{CRM}}
\qquad (27)
$$

其中 $\mathcal{L}_{\text{CTR}}$ 和 $\mathcal{L}_{\text{CTCVR}}$ 是 ESMM 的原始损失，$\mathcal{L}_{\text{CRM}}$ 是 CRM 正则化器（可以是 $\mathcal{L}_{\text{IPS-CVR}}$ 或 $\mathcal{L}_{\text{DR-CVR}}$），$\lambda_{\text{c}}$ 和 $\lambda_{\text{r}}$ 是平衡不同损失项的权重参数。

#### 4.2.1 ESCM2-IPS

当使用 IPS 正则化器时，ESCM2-IPS 的学习目标为：

$$
\mathcal{L}_{\text{ESCM2-IPS}} = \mathcal{L}_{\text{CTR}} + \lambda_{\text{c}} \mathcal{L}_{\text{CTCVR}} + \lambda_{\text{r}} \mathcal{L}_{\text{IPS-CVR}}
$$

#### 4.2.2 ESCM2-DR

当使用 DR 正则化器时，ESCM2-DR 的学习目标为：

$$
\mathcal{L}_{\text{ESCM2-DR}} = \mathcal{L}_{\text{CTR}} + \lambda_{\text{c}} \mathcal{L}_{\text{CTCVR}} + \lambda_{\text{r}} \mathcal{L}_{\text{DR-CVR}}
$$



## 5. 实验

### 5.1 实验设置

#### 5.1.1 数据集

我们在两个数据集上进行了实验：一个工业数据集和一个公开的 Ali-CCP 数据集。工业数据集包含来自蚂蚁集团真实推荐场景的用户行为日志。Ali-CCP 数据集是阿里巴巴提供的大规模推荐数据集。

| 数据集 | 曝光数 | 点击数 | 转化数 | CTR | CVR |
|--------|--------|--------|--------|-----|-----|
| 工业数据集 | 2.4B | 92.2M | 1.2M | 3.8% | 1.3% |
| Ali-CCP | 84.0M | 3.4M | 135.0K | 4.0% | 4.0% |

表 1：数据集统计信息。

#### 5.1.2 基线方法

我们将 ESCM2 与以下基线方法进行了比较：

- **Naive**：在点击空间上直接训练 CVR 模型
- **ESMM**：全空间多任务模型
- **MTL-IMP**：使用重要性加权的多任务学习方法
- **ESMM-IPS**：使用 IPS 正则化的 ESMM

#### 5.1.3 训练协议

所有模型都使用 Adam 优化器进行训练，学习率为 0.001。批量大小为 2048。我们在工业数据集上训练了 10 个 epoch，在 Ali-CCP 数据集上训练了 5 个 epoch。

#### 5.1.4 评估协议

我们使用 AUC（Area Under the ROC Curve）作为主要评估指标。对于在线评估，我们进行了 A/B 测试，持续时间为 7 天。

### 5.2 实验结果

#### 5.2.1 离线结果

表 2 显示了离线数据集上的实验结果。

| 方法 | 工业数据集 AUC | Ali-CCP AUC |
|------|---------------|-------------|
| Naive | 0.751 | 0.625 |
| ESMM | 0.754 | 0.627 |
| MTL-IMP | 0.756 | 0.629 |
| ESCM2-IPS | 0.758 | 0.631 |
| ESCM2-DR | **0.759** | **0.633** |

表 2：离线数据集上的 AUC 结果。

从表 2 中我们可以观察到：

1. ESMM 优于 Naive 方法，证实了全空间建模的有效性。
2. ESCM2-IPS 和 ESCM2-DR 都优于 ESMM，表明 CRM 正则化器能够有效减少估计偏差。
3. ESCM2-DR 优于 ESCM2-IPS，这与 DR 估计器具有更低方差的理论预期一致。

#### 5.2.2 在线 A/B 结果

我们在蚂蚁集团的真实推荐系统上进行了在线 A/B 测试。实验持续时间为 7 天，涉及约 340 万独立访客（UV）和 490 万页面浏览量（PV）。

| 方法 | CVR 提升 | CTCVR 提升 |
|------|---------|-----------|
| ESCM2-IPS vs ESMM | +2.1% | +1.5% |
| ESCM2-DR vs ESMM | +3.4% | +2.3% |

表 3：在线 A/B 测试结果。

在线实验结果与离线结果一致，ESCM2 能够显著提升 CVR 和 CTCVR 的预估性能。



## 6. 相关工作

### 6.1 全空间多任务学习

全空间多任务学习旨在利用整个曝光空间的信息来改进特定任务的预估。ESMM [14] 是这一方向的开创性工作，它通过链式法则将 CVR 预估转化为 CTR 和 CTCVR 的联合学习。后续工作 [16, 24, 25] 进一步改进了 ESMM 框架，通过引入更多的行为信号或更复杂的任务关系建模。

### 6.2 因果推理与推荐系统

因果推理在推荐系统中的应用近年来受到了广泛关注。IPS 方法 [18, 22] 被广泛用于解决推荐系统中的偏差问题。DR 估计器 [22] 结合了插补方法和倾向得分方法的优点，提供了更稳健的偏差校正。我们的工作将这些因果方法引入 ESMM 框架，从理论上解决了 ESMM 的固有偏差问题。

### 6.3 样本选择偏差

样本选择偏差是推荐系统中的一个基本问题。Ma 等人 [14] 首次在 CVR 预估场景中识别了这一问题，并提出了 ESMM 作为解决方案。然而，正如我们在本文中证明的，ESMM 并没有完全解决这一问题。我们的工作提供了更严格的理论分析和更有效的解决方案。



## 7. 结论

在本文中，我们从理论角度严格分析了 ESMM 的两个关键问题：固有估计偏差（IEB）和潜在独立优先（PIP）。我们提出了 ESCM2，一种基于反事实风险最小化的全空间多任务模型，通过引入 IPS 和 DR 正则化器来同时解决这两个问题。大量离线和在线实验表明，ESCM2 能够有效消除 ESMM 中的偏差，并取得优于基线模型的性能。

未来工作包括：（1）将 ESCM2 扩展到更复杂的行为序列建模；（2）研究自适应倾向得分估计方法；（3）探索 ESCM2 在其他推荐场景中的应用。


---



## 参考文献

[1] Wentian Bao, Hong Wen, Sha Li, Xiao-Yang Liu, Quan Lin, and Keping Yang. 2020. GMCM: Graph-based Micro-behavior Conversion Model for Post-click Conversion Rate Estimation. In SIGIR. 2201–2210.

[2] Elias Bareinboim and Judea Pearl. 2012. Controlling selection bias in causal inference. In Artificial Intelligence and Statistics. 100–108.

[3] Hongliang Fei, Jingyuan Zhang, Xingxuan Zhou, Junhao Zhao, Xinyang Qi, and Ping Li. 2021. GemNN: Gating-enhanced Multi-task Neural Networks with Feature Interaction Learning for CTR Prediction. In SIGIR. 2166–2171.

[4] Chen Gao, Tzu-Heng Lin, Nian Li, Depeng Jin, and Yong Li. 2021. Cross-platform Item Recommendation for Online Social E-Commerce. TKDE (2021).

[5] Garrido. 2014. Methods for constructing and assessing propensity scores. , 1701–1720 pages.

[6] Tiankai Gu, Kun Kuang, Hong Zhu, Jingjie Li, Zhenhua Dong, Wenjie Hu, Zhenguo Li, Xiuqiang He, and Yue Liu. 2021. Estimating True Post-Click Conversion via Group-stratified Counterfactual Inference. In ADKDD.

[7] Huifeng Guo, Ruiming Tang, Yunming Ye, Zhenguo Li, and Xiuqiang He. 2017. DeepFM: A Factorization-Machine based Neural Network for CTR Prediction. In IJCAI. 1725–1731.

[8] Siyuan Guo, Lixin Zou, Yiding Liu, Wenwen Ye, Suqi Cheng, Shuaiqiang Wang, Hechang Chen, Dawei Yin, and Yi Chang. 2021. Enhanced Doubly Robust Learning for Debiasing Post-Click Conversion Rate Estimation. In SIGIR. 275–284.

[9] Robins JM Hernán MA. 2020. Causal Inference: What If. Boca Raton: Chapman Hall/CRC.

[10] Diederik P. Kingma and Jimmy Ba. 2015. Adam: A Method for Stochastic Optimization. In ICLR.

[11] Jae-woong Lee, Seongmin Park, and Jongwuk Lee. 2021. Dual Unbiased Recommender Learning for Implicit Feedback. In SIGIR. 1647–1651.

[12] Tianqiao Liu, Zhiwei Wang, Jiliang Tang, Songfan Yang, Gale Yan Huang, and Zitao Liu. 2019. Recommender Systems with Heterogeneous Side Information. In WWW. 3027–3033.

[13] Jiaqi Ma, Zhe Zhao, Xinyang Yi, Jilin Chen, Lichan Hong, and Ed H. Chi. 2018. Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts. In SIGKDD. 1930–1939.

[14] Xiao Ma, Liqin Zhao, Guan Huang, Zhi Wang, Zelin Hu, Xiaoqiang Zhu, and Kun Gai. 2018. Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate. In SIGIR. 1137–1140.

[15] Benjamin Marlin, Richard S Zemel, Sam Roweis, and Malcolm Slaney. 2012. Collaborative filtering and the missing at random assumption. arXiv preprint arXiv:1206.5267 (2012).

[16] Conor O'Brien, Kin Sum Liu, James Neufeld, Rafael Barreto, and Jonathan J. Hunt. 2021. An Analysis Of Entire Space Multi-Task Models For Post-Click Conversion Prediction. In RecSys. 613–619.

[17] Judea Pearl. 2009. Causality. Cambridge University Press.

[18] Tobias Schnabel, Adith Swaminathan, Ashudeep Singh, Navin Chandak, and Thorsten Joachims. 2016. Recommendations as Treatments: Debiasing Learning and Evaluation. In ICML. 1670–1679.

[19] Jian Shen, Yanru Qu, Weinan Zhang, and Yong Yu. 2018. Wasserstein Distance Guided Representation Learning for Domain Adaptation. In AAAI. 4058–4065.

[20] Harald Steck. 2010. Training and testing of recommender systems on data missing not at random. In SIGKDD. 713–722.

[21] Eric Tzeng, Judy Hoffman, Kate Saenko, and Trevor Darrell. 2017. Adversarial Discriminative Domain Adaptation. In CVPR. 2962–2971.

[22] Xiaojie Wang, Rui Zhang, Yu Sun, and Jianzhong Qi. 2019. Doubly Robust Joint Learning for Recommendation on Data Missing Not at Random. In ICML. 6638–6647.

[23] Hong Wen, Jing Zhang, Fuyu Lv, Wentian Bao, Tianyi Wang, and Zulong Chen. 2021. SIGIR. 2187–2191.

[24] Hong Wen, Jing Zhang, Yuan Wang, Fuyu Lv, Wentian Bao, Quan Lin, and Keping Yang. 2020. Entire Space Multi-Task Modeling via Post-Click Behavior Decomposition for Conversion Rate Prediction. In SIGIR. 2377–2386.

[25] Dongbo Xi, Zhen Chen, Peng Yan, Yinger Zhang, Yongchun Zhu, Fuzhen Zhuang, and Yu Chen. 2021. Modeling the Sequential Dependence among Audience Multi-step Conversions with Multi-task Learning in Targeted Display Advertising. In SIGKDD. 3745–3755.

[26] Mengyue Yang, Quanyu Dai, Zhenhua Dong, Xu Chen, Xiuqiang He, and Jun Wang. 2021. Top-N Recommendation with Counterfactual User Preference Simulation. In CIKM. 2342–2351.

[27] Wenhao Zhang, Wentian Bao, Xiao-Yang Liu, Keping Yang, Quan Lin, Hong Wen, and Ramin Ramezani. 2020. Large-scale Causal Approaches to Debiasing Post-click Conversion Rate Estimation with Multi-task Learning. In WWW. 2775–2781.

[28] Guorui Zhou, Xiaoqiang Zhu, Chenru Song, Ying Fan, Han Zhu, Xiao Ma, Yanghui Yan, Junqi Jin, Han Li, and Kun Gai. 2018. Deep interest network for click-through rate prediction. In SIGKDD. 1059–1068.



## 附录 A：实现细节

在本节中，我们简要描述 ESCM2 的实现过程。要构建 ESCM2-IPS：

1. 使用公式 (5) 中的 CTR 损失 $\mathcal{L}_{\text{CTR}}$ 和 CTCVR 损失 $\mathcal{L}_{\text{CTCVR}}$ 构建 ESMM 模型。
2. 计算倾向得分。在 CVR 预估任务中，一个捷径是直接使用 CTR 估计。
3. 在点击空间中使用 $\delta$ 和倾向得分计算 CVR 估计误差 $\mathcal{L}_{\text{CVR}}$，遵循公式 (15)。
4. 按照公式 (27) 计算 ESCM2 的学习目标，并使用随机梯度方法进行优化。

在步骤 2 中，倾向得分（CTR 估计）的小值会导致较大的估计方差和数值误差。在实践中，我们设置一个阈值（例如在我们的设置中为 0.1）来将倾向得分裁剪到合理的范围内，遵循 Schnabel 等人 [18] 的方法。在步骤 3 中，$\mathcal{L}_{\text{CVR}}$ 对倾向得分（CTR 估计）的梯度应该被截断，否则 $\mathcal{L}_{\text{CVR}}$ 会偏差 CTR 估计。在步骤 4 中，应尽一切努力确保准确的 CTR 估计，因为它对 CVR 和 CTCVR 估计都有显著影响。例如，如图 6 所讨论的，将权重 $\lambda_{\text{c}}$ 设置为较小的值（在我们的案例中为 0.1-1），并增大 CTR 塔的规模。
