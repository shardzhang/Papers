# 面向真实在线广告系统的场感知分解机

> Yuchin Juan（Criteo Research）| Damien Lefortier（Facebook）| Olivier Chapelle（Google）

本文将场感知分解机（FFM）应用于 Criteo 的生产级广告点击率和转化率预测系统。核心内容：

- 在生产系统中离线和在线（A/B 测试）评估了 FFM 相对于逻辑回归的性能提升，尤其在小型广告主上表现显著
- 提出了基于迭代参数混合（IPM）的分布式训练方案，并通过 AdaGrad 梯度聚合改进收敛速度，实现了约 12 倍加速
- 提出了"未成熟热启动"（pre-mature warm-start）方法，解决了 FFM 因缺乏正则化而依赖早停法所带来的热启动难题

关键发现：

- FFM 在离线 NLL 和 Utility 指标上显著优于逻辑回归，在线 A/B 测试中投资回报率（ROI）提升 $+0.97\%$（全体广告主）和 $+2.61\%$（小型广告主）
- 分布式训练结合改进的 AdaGrad 聚合可将训练时间加速约 12 倍
- 未成熟热启动结合小训练集可将训练时间缩短至基线的 1/20，同时不损失预测精度

---

## 摘要

预测用户响应是计算广告领域核心的机器学习任务之一。场感知分解机（FFM，Field-aware Factorization Machines）最近被确立为该问题的最先进方法，并特别赢得了两项 Kaggle 竞赛。本文展示了在预测展示广告点击率和转化率的生产系统中实施该方法的一些结果，表明该方法不仅在竞赛中有效，在真实世界的预测系统中同样具有价值。我们还讨论了一些具体的挑战和降低训练时间的解决方案，包括一种创新的种子初始化算法和分布式学习机制。

## 1. 引言

在线广告是互联网公司的主要业务之一，该领域的核心问题之一是在正确的时间将正确的广告匹配给正确的用户。准确的点击率（CTR，Click-Through Rate）预测对于解决这一问题至关重要，无论是搜索广告 [11, 20] 还是展示广告 [5, 14]，这都已成为广泛研究的主题。效果类广告主不仅根据点击量来衡量其广告活动的效果，还根据转化量——定义为用户在网站上的操作（如购买）——来衡量，因此已开发了专门用于转化预测的机器学习模型 [15, 23, 3, 26]。

这些预测问题的一个重要模型是带有交叉特征的逻辑回归 [20, 5]。当添加所有交叉特征时，得到的模型等价于 2 阶多项式核 [2]。Criteo 在 2014 年举办了一场 Kaggle 竞赛来比较 CTR 预测算法。带有交叉特征的逻辑回归在该竞赛中确实相当成功：第 3 名的获胜方案正是基于该技术 [24]。但获胜方案是分解机（Factorization Machines）[22] 的一个变体，称为场感知分解机（FFM）[14]。FFM 的出色表现促使我们将其实施并作为生产系统的一部分进行测试。

**FFM。** 考虑分类特征的情况——广告系统中的大多数特征要么是分类的，要么可以通过离散化变为分类的。令 $F$ 为特征（或场）的数量，$v_1, \ldots, v_F$ 为给定样本中这些特征的值。该样本上的 FFM 预测可以写为：

$$
\hat{y} = \sum_{f_1=1}^{F} \sum_{f_2=f_1+1}^{F} \mathbf{w}_{i_1} \cdot \mathbf{w}_{i_2} \qquad (1)
$$

其中 $i_1 = \Phi(v_{f_1}, f_1, f_2)$，$i_2 = \Phi(v_{f_2}, f_2, f_1)$，

这里 $\mathbf{w} \in \mathbb{R}^{d \times k}$ 是权重矩阵，$\mathbf{w}_i \in \mathbb{R}^k$ 表示第 $i$ 个条目的嵌入向量。映射 $\Phi(v, f_1, f_2)$ 将特征 $f_1$ 的值 $v$ 在特征 $f_2$ 的上下文中映射到 $1$ 到 $d$ 之间的索引。这可以是任何哈希函数或基于字典。在后一种情况下，$d$ 等于 $\sum_{f=1}^{F} c_f$，其中 $c_f$ 是第 $f$ 个特征的基数。

在常规分解机中，给定特征值只有一个嵌入向量；换句话说，公式 (1) 中 FM 的索引为 $i_1 = \Phi(v_{f_1}, f_1)$ 和 $i_2 = \Phi(v_{f_2}, f_2)$。但在场感知 FM 中，根据点积中另一个特征的不同，嵌入向量也不同。正如 [14] 中所论述的，这提供了额外的建模灵活性。

**相关工作。** AdRoll 在一篇博客文章中报告了与我们类似的工作：作者报告了在其 CTR 预测系统中部署 FM 后获得了显著的性能提升。Google [20] 和 Facebook [12] 可能未使用 FM，但报告了在将其大规模 CTR 预测系统投入生产时遇到的一些具体挑战，这些挑战与 FFM 的生产化相关。分解机支持的神经网络（FNN，Factorisation Machine supported Neural Network）和基于采样的神经网络（SNN，Sampling-based Neural Network）[28] 是两种与 FM 相关的学习算法，也被应用于 CTR 预测任务。它们都是深度神经网络，但在嵌入层有所不同：SNN 使用常规嵌入层，而 FNN 使用分解机的结果进行初始化。近期对分解机的兴趣促进了分布式求解器 [18] 的开发。此外，[21] 引入了分解机的层次化版本。

尽管 FFM 已通过赢得两项 Kaggle 竞赛被证明是计算广告的最先进方法，但其是否适合生产环境尚不明确。Netflix 竞赛提醒我们，生产系统具有一系列不同于学术竞赛的特定约束和目标：Netflix 最终决定不使用获胜方案。本文讨论了我们尝试在预测展示广告点击率和转化率的生产系统中实施 FFM 的工作。第 2 节展示了离线和在线（A/B 测试）结果，并提供了该方法相对于标准逻辑回归的收益以及在生产系统中使用 FFM 的挑战的一些见解。这些积极的结果进一步促使我们解决 FFM 实施中遇到的主要瓶颈之一：训练速度。第 3 节研究了如何在分布式环境中训练 FFM。第 4 节提出了一种创新的模型种子初始化程序来进一步解决该问题，以更短的训练时间和更少的计算资源获得更准确的模型。最后，第 5 节给出了结论和未来工作。

## 2. 生产系统中的 FFM

在本节中，我们描述如何在生产系统中使用 FFM，展示我们的离线和在线结果，并讨论在此设置中使用 FFM 的收益和挑战。

### 2.1 基线

如第 1 节所述，最先进的广告系统基于点击率和转化率预测模型。在本文中，我们考虑用于实时竞价的 CTR 和 CR 预测模型（参见例如 [5, 26]）。为了预测给定展示后的成交概率，我们使用乘法模型，即给定展示后的点击概率模型与给定点击后的成交概率模型的乘积，如 [3] 中所讨论。因此，在本文的其余部分中，我们称这两个模型为 CTR 和 CR。

我们用于训练这些模型的基线系统基于之前的工作 [1, 5, 26]。遵循 [1, 5]，我们使用哈希技巧 [27] 来降低数据的维度，从而减少需要拟合的参数数量。我们使用带有交叉特征的逻辑回归（LR），通过 SGD 进行 L-BFGS 热启动来拟合 [1, 5]。遵循 [26]，我们还对 CR 模型使用代价敏感学习，根据销售对广告主的价值对每次销售进行加权，因为这已被证明可以提高 CR 模型的离线和在线性能。我们使用 Hadoop AllReduce 来分布式学习我们的模型 [1]。

以下，我们研究用 FFM 代替 LR 来训练 CTR 和 CR 预测模型。我们仍然使用哈希技巧。因此，公式 (1) 中的映射 $\Phi(v, f_1, f_2)$ 基于一个固定哈希空间（数量级为数千万）的哈希。

### 2.2 离线比较

我们现在展示 FFM 与最先进基线在离线数据集上的比较结果。

**离线指标。** 我们使用两个离线指标。第一个是归一化对数损失（NLL，Normalized Log Loss）。该指标展示了待评估模型相对于基线预测器在对数损失上的相对改善，在我们的例子中基线预测器是数据集的平均经验 CTR 或 CR，类似于 [12, 16, 26] 中的归一化。该指标对任何预测 $p$ 的正式定义如下，其中 $\bar{p}$ 表示测试集上的最佳常数预测器，$N$ 表示数据集中的展示数量。

$$
LL(p) = -\sum_{i=1}^{N} \left[ y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \right] \qquad (2)
$$

$$
NLL(p) = \frac{LL(\bar{p}) - LL(p)}{LL(\bar{p})} \qquad (3)
$$

我们还使用 Utility 5 指标 [4, 26]，它可以离线建模由于预测模型变化而导致的潜在利润变化。由于历史数据中的观测利润是固定的，该指标假设展示成本由第二价格拍卖中的最高第二出价决定，并且这些成本根据观测展示成本的条件分布生成。该指标定义如下，其中 $v_i$ 是第 $i$ 次展示的奖励。

$$
Utility = \sum_i \int_0^{p(x_i)} (y_i \cdot v_i - \tilde{c}) \Pr(\tilde{c} | c_i) d\tilde{c} \qquad (4)
$$

分布 $\Pr(\tilde{c} | c)$ 指定了在观测成本 $c$ 的情况下可能的第二价格是什么；[4] 建议使用参数 $\alpha = \beta c + 1$ 和自由参数 $\beta$ 的 Gamma 分布。选择该分布的动机在于它能在两个极限分布之间进行良好的插值：集中在 $c$ 处的 Dirac 分布（当 $\beta \to +\infty$）和均匀分布（当 $\beta \to 0$）。可以证明，使用均匀分布的 Utility 等价于加权平方误差 [13]。

**实验设置。** 我们使用 Criteo 的内部数据进行实验。然而，如第 1 节所讨论，FFM 已在许多公共数据集上被证明优于现有方法 [14]。此外，本节的目标是展示我们可以在使用自有数据的真实在线广告系统中使用 FFM 来改进基线。我们需要离线实验来确保 FFM 在我们的系统中（在预测性能和可扩展性方面）表现良好并进行参数调优，然后才能进行线上实验（A/B 测试）。

我们使用渐进式验证（progressive validation）的变体进行实验，类似于 [20]。训练期之后的第二天作为验证集。如图 1 所示，该过程重复 $N$ 次，每次将学习期（标记为"tr"）向前移动 1 天。最终结果是所有测试集（标记为"te"）上的平均指标。

参数调优在与最终实验所用数据不同的时间切片上进行。遵循 [14]，调优以下参数：正则化参数、学习率和 latent 因子数量。我们使用早停法来避免过拟合。

**延迟和内存消耗。** 在生产系统中使用 FFM 的一个潜在缺点是它们在推理时需要更多的 CPU 时间 [14]。这可能导致在响应竞价请求时在线延迟增加，从而导致更多的超时。FFM 还需要更多的内存来存储模型，因为 latent 因子数量和/或场数量的增加，这可能导致比 LR 大得多的内存消耗。

为解决内存问题，我们提出减小 FFM 模型的哈希空间大小（与基线相比），使 FFM 模型与 LR 模型具有相同的大小（具体值取决于场的数量和 latent 因子的数量）。注意，如果我们没有减小哈希空间大小而是保持不变，FFM 模型的大小将比我们的基线大 100 倍以上，这将使其不切实际。因此，在以下结果中，FFM 和 LR 模型具有相同数量的参数（与 [14] 不同）。

为解决延迟问题，我们提出在不显著降低 FFM 性能的情况下尽可能减少 latent 因子的数量。通过这两种解决方案，FFM 和 LR 消耗相同数量的内存，我们可以将延迟影响限制在满足生产系统要求的范围内。

**离线结果。** 我们在 CTR 和 CR 预测任务上比较了 LR 和 FFM 的 NLL（表 1）和 Utility（表 2）。

FFM 在 NLL 和 Utility 方面均取得了显著优于 LR 的结果，具有较大的效应量，这在我们的 CTR 模型上得到了验证，从而确认了 [14] 的结果在我们的数据上成立。我们还在 CR 模型上观察到了较大的增益，从而将 [14] 的结果扩展到了 CR 模型的所有离线指标上。

我们还观察到，小型广告主（占我们流量的很大一部分）的改进更大，这在 CTR 和 CR 模型的所有指标上都有体现。我们解释这些结果的假设与稀疏数据和未观测的交叉特征有关：LR 无法预测不属于训练数据的交叉特征所关联的值；而 FFM 能够通过其 latent 表示更好地泛化（详见 [14, 第 2 节] 的详细解释和示例）。对于大型广告主，LR 有足够的数据来学习好的模型，但对于小型广告主，FFM 更好地处理了数据稀疏性问题。

在超参数调优过程中，我们观察到与 [14] 非常相似的各超参数性能。最重要的参数是 epoch 数量，我们使用早停法来自动调优它。

我们还研究了 FFM 相对于基线模型的预测时间，尽管我们将 FFM 模型限制为与基线相同的大小，预计预测时间仍会增加。这是因为计算预测公式 (1) 的操作数量为 $O(F^2 k)$，而带有所有交叉特征的 LR 仅需要 $O(F^2)$ 次操作。我们观察到 FFM 的减速确实与 latent 因子数量 $k$ 成正比。结果表明 $k=2$ 是一个良好的权衡：与上述使用 4 个 latent 因子获得的结果相比，NLL 仅下降 0.1%，而 2 倍的预测时间在我们的系统中是可接受的，因为预测不是处理请求中最耗时的部分（与提取原始特征、预处理等相比）。

**表 1：逻辑回归（基线）和 FFM 在 CTR 和 CR 预测任务上的离线相对比较（NLL 指标）。** 我们展示了所有广告主和小型广告主——定义为平均每天销售少于 30 次的广告主——的结果。CTR（或 CR）模型的 NLL 是给定展示后点击（或成交）概率的 NLL。统计显著性用 N 标注。

| 预测模型 | 所有广告主 NLL | 小型广告主 NLL |
|:---|:---:|:---:|
| CTR（FFM） | +3.71%N | +5.9%N |
| CTR + CR（FFM） | +1.21%N | +6.2%N |

**表 2：逻辑回归（基线）和 FFM 在 CTR 和 CR 预测任务上的离线相对比较（Utility 指标）。** 我们报告模型在给定展示后预期销售数量方面的 Utility，该指标使用 CTR 和 CR 模型作为子模型。统计显著性用 N 标注。

| 预测模型 | $\text{Utility}_{\beta=10}$ 所有 | $\text{Utility}_{\beta=10}$ 小型 | $\text{Utility}_{\beta=1000}$ 所有 | $\text{Utility}_{\beta=1000}$ 小型 |
|:---|:---:|:---:|:---:|:---:|
| CTR（FFM） | +6.29%N | +9.70%N | +2.22%N | +4.39%N |
| CTR + CR（FFM） | +11.42%N | +38.44%N | +5.43%N | +18.34%N |

### 2.3 在线比较

由于离线结果相当有前景，我们决定对 CTR 和 CR 预测模型都使用 FFM 进行 A/B 测试。尽管 FFM 需要更多的推理时间（见上文），但我们在服务在线流量时未观察到任何对超时的显著影响。因此，我们能够在大部分在线流量上对 FFM 进行 A/B 测试。该 A/B 测试服务了约 50 亿次展示（每个群体约 25 亿次）。

在 A/B 测试期间，我们确保基线模型和 FFM 同步在线刷新，因为不同的刷新率可能会偏差结果。即使使用多线程，FFM 的学习时间也确实远高于我们的分布式优化基线。在第 3 节和第 4 节中，我们将看到如何减少这个学习时间，但现在我们只关注使用 FFM 可以在线获得的性能改进。

结果如表 3 所示。我们观察到展示数量增加了（+4.59%），而总体展示成本几乎保持不变。我们观察到点击量减少，但转化量增加，从而在相同成本下为广告主带来了更多价值。因此，我们的更改产生了显著的正向影响：投资回报率（ROI，Return On Investment）提升了 +0.97%，即广告主价值与成本之比，这是相当可观的。

我们还观察到小型广告主（定义为每天销售少于 30 次的广告主，占我们流量的很大一部分）的改进更大。在小型广告主上，我们观察到展示数量增加了（+4.85%），而总体展示成本也几乎保持不变。我们还观察到点击量减少，但转化量更多，从而在相同成本下为广告主带来了更多价值，ROI 提升了 +2.61%，这是非常显著的。

这证实了我们的离线结果，并表明 FFM 的优势之一确实是通过使用 latent 表示比逻辑回归更好地泛化的能力。

**表 3：逻辑回归（基线）和 FFM 在 CTR 和 CR 预测模型上的在线相对比较（ROI 指标）。** 在 A/B 测试期间的广告主价值与成本之比。统计显著性用 N 标注。

| 预测模型 | 所有广告主 ROI | 小型广告主 ROI |
|:---|:---:|:---:|
| CTR + CR（FFM） | +0.97%N | +2.61%N |

### 2.4 讨论

我们积极的在线结果促使我们使用 FFM 代替 LR 投入生产。为此，如果 SGD 已经可用，代码更改相当小。然而，在生产系统中使用 FFM 代替 LR 时需要记住一些挑战。

推出 FFM 的主要顾虑是学习时间，如前所述，它远高于基线。这意味着使用 FFM 时模型的刷新频率会降低，代价是降低系统性能。我们所有用于改进模型的离线实验也会花费更长时间。这是不可接受的，我们将在接下来的两节中讨论如何通过在多台机器上分布式学习来解决这个问题以应对大规模生产系统的规模。

还有其他挑战。上面我们讨论了内存消耗和预测延迟问题，并展示了如何管理它们。另一个潜在问题是 FFM 目标函数的非凸性，由于局部最小值可能导致 FFM 性能的不稳定性。为了调查这一点，我们在相同的数据集上使用随机权重初始化学习了多个 FFM，如 [14] 中所述。我们观察到所有模型尽管初始化不同，但具有相似的性能（NLL ±0.05%）。因此局部最小值问题不是主要顾虑。

我们还看到 FFM 的超参数数量多于 LR，增加了学习率（因为我们使用 L-BFGS 训练 LR 模型）和 latent 因子数量，而 LR 只需要调优正则化参数。这意味着改进模型时调优需要更多时间。然而，如 [14] 中所讨论，这不是主要问题，原因如下。首先，性能对 latent 因子数量和正则化参数不太敏感，而学习率的合适值容易找到。我们还发现 FFM 的性能随时间对超参数保持稳定（无需持续重新调优）。

由于我们未能为 FFM 找到令人满意的正则化器，我们使用早停法来避免过拟合 [14]——这是我们唯一的解决方案。因此，应添加一些监控以确保我们不会欠拟合或过拟合，即使使用了早停法（例如，用于测试和决定何时停止的少量数据不具有代表性的情况）。

最后注意，为了高效的回归测试 [16]，我们需要固定用于随机化初始权重的种子 [14]。

## 3. 一种简单的分布式设置

在上一节中，我们讨论了 FFM 的训练时间太慢，无法满足我们的生产要求，即使在多核机器上应用了 [14] 中提到的并行化方法之后也是如此。

为了获得更多的加速，一个自然的选择是在分布式系统上训练 FFM。一般来说，对于顺序算法（如 SGD 或对偶坐标下降），其并行化的收敛速度取决于每个 worker 访问模型的频率。在共享内存系统中，由于每个线程可以实时访问模型，收敛速度可能保持不变，如 [14] 中所示。然而，在需要通过网络进行通信的分布式系统中，我们无法再实时在机器之间共享模型（由于网络开销）。分布随机梯度算法主要有两种方式：同步和异步。在两种情况下，每台机器拥有数据的一个子集和自己的本地模型，在处理一批数据点后更新全局模型。异步训练通常被称为参数服务器方法 [17, 18, 7]：一些机器专门用于存储全局模型，worker 持续读取并用其本地模型更新该模型。另一方面，同步训练被称为迭代参数混合（IPM，Iterative Parameter Mixing）[19, 29, 1]：所有模型在处理一定量数据后进行平均（例如每个 epoch）。从工程角度来看，简单性是我们选择算法时考虑的最重要因素之一。复杂的算法需要更多的开发时间，更难维护，更容易引入 bug。因此，实际上，如果一个更简单的算法可以解决我们的问题，我们不会选择更复杂的算法。正如我们将看到的，使用 IPM 我们已经能够将 32 台机器的训练时间加速 12 倍。这已经满足了我们的需求，因此本文不研究参数服务器方法。AdaGrad 学习算法 [9] 的 IPM 如算法 1 所述。

分布式算法的加速可以用以下方程建模：

$$
\text{speed-up} = \#\text{machines} \times \frac{\#\text{epochs with multiple machines}}{\#\text{epochs with one machine}}
$$

该方程基于两个假设：
1. 每台机器几乎在同一时间完成计算。
2. 机器之间的通信成本可以忽略不计。

在我们的情况下，两个假设都成立。第一个假设成立，因为我们均匀地将训练数据分配给所有机器，并确保每台机器具有相似的计算能力。第二个假设成立，因为 IPM 仅在每个 epoch 结束时需要同步，使得同步时间远小于计算时间。

"真正的"分布式算法嵌入在我们的内部系统中并运行在我们的内部数据集上，因此我们无法发布它。为了实验可重复性，在本文中我们使用多线程来模拟机器，并使用从 Criteo 的 CTR 预测挑战赛中获得的数据集（第 1 节中描述）。这种模拟接近现实，因为分布式算法的加速仅取决于使用的机器数量和收敛速度的减慢，这可以通过多线程精确模拟。

如果我们直接应用 IPM，那么随着不断添加机器，收敛会越来越慢。实验结果如表 4 所示。假设我们使用 32 台机器而不是 1 台，虽然计算速度快了 32 倍，但还需要 20 倍的 epoch。因此加速仅为 $32/(157/8) \approx 1.6$。

**表 4：达到最佳对数损失所需的不同机器数量的 epoch 数。** 应用了算法 1。学习率 $\eta$ 为 0.2。

| 机器数量 | 1 | 2 | 4 | 8 | 16 | 32 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| epoch 数 | 8 | 15 | 29 | 47 | 100 | 157 |
| 对数损失 | 0.44552 | 0.44548 | 0.44549 | 0.44560 | 0.44554 | 0.44585 |

使收敛更快的一个自然方法是增大学习率 $\eta$。虽然增大学习率确实使算法收敛更快，但也使对数损失变差。该结果如表 5a 所示。

**表 5：使用 32 台机器时，达到最佳对数损失所需的不同学习率的 epoch 数。**

**(a) 算法 1**

| $\eta$ | 0.2 | 0.5 | 1.0 | 2.0 | 3.0 | 4.0 | 5.0 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| epoch 数 | 157 | 70 | 37 | 26 | 21 | 19 | 18 |
| 对数损失 | 0.44585 | 0.44569 | 0.44590 | 0.44622 | 0.44654 | 0.44688 | 0.44721 |

**(b) 算法 2**

| $\eta$ | 0.2 | 0.5 | 1.0 | 2.0 | 3.0 | 4.0 | 5.0 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| epoch 数 | 200 | 130 | 55 | 31 | 22 | 18 | 16 |
| 对数损失 | 0.44819 | 0.44600 | 0.44578 | 0.44565 | 0.44577 | 0.44592 | 0.44608 |

我们提出以下方法来解决这个问题。回顾一下，遵循 [14]，我们使用 AdaGrad [9] 来提升 SGD 的性能。AdaGrad 记录梯度平方和（$G$）以动态调整每个维度的学习率。在算法 1 中，$G$ 在机器之间不同步。这可能使每台机器上的 $G$ 非常小，从而使有效学习率过大。基于与 [1] 类似的思想，我们在每个 epoch 结束时在每台机器之间聚合 $G$。该新算法如算法 2 所述。实验结果如表 5b 所示。当使用较大学习率时，对数损失好得多。在此设置下，如果我们选择 $\eta = 3.0$，可以实现的加速为 $32 \times (8/22) \approx 12$。确实，在我们的系统中应用此设置后，我们观察到了类似的加速，使我们能够像当前系统一样快速地训练模型。

**算法 1：AdaGrad 的迭代参数混合（IPM）**

$$
\begin{aligned}
&1: \text{将 } m \text{ 个数据点分配到 } k \text{ 台机器上} \\
&2: \text{初始化 } \mathbf{w} \\
&3: \text{初始化 } \mathbf{G}_i \leftarrow \mathbf{I} \quad \forall i \in \{1, \ldots, k\} \\
&4: \textbf{for } t \in \{1, \ldots, T\} \textbf{ do} \quad \text{\# } T: \text{epoch 数量} \\
&\quad 5: \quad \text{令 } \mathbf{w}_i \leftarrow \mathbf{w} \quad \forall i \in \{1, \ldots, k\} \\
&\quad 6: \quad \textbf{for } i \in \{1, \ldots, k\} \textbf{ parallel do} \\
&\quad\quad 7: \quad\quad \textbf{for } \text{每个数据点 } \textbf{do} \\
&\quad\quad\quad 8: \quad\quad\quad \text{计算梯度 } \mathbf{g} \\
&\quad\quad\quad 9: \quad\quad\quad \text{更新 } \mathbf{G}_i: \mathbf{G}_i \leftarrow \mathbf{G}_i + \text{diag}(\mathbf{g} \mathbf{g}^T) \\
&\quad\quad\quad 10: \quad\quad\quad \text{更新 } \mathbf{w}_i: \mathbf{w}_i \leftarrow \mathbf{w}_i - \eta \mathbf{G}_i^{-1/2} \mathbf{g} \\
&\quad\quad 11: \quad\quad \mathbf{w} \leftarrow \sum_{i=1}^{k} \mathbf{w}_i / k \\
&\quad 12: \textbf{end for} \\
&13: \textbf{end for}
\end{aligned}
$$

**算法 2：改进的 AdaGrad IPM**

$$
\begin{aligned}
&1: \text{将 } m \text{ 个数据点分配到 } k \text{ 台机器上} \\
&2: \text{初始化 } \mathbf{w} \\
&3: \text{初始化 } \mathbf{G} \leftarrow \mathbf{I} \\
&4: \textbf{for } t \in \{1, \ldots, T\} \textbf{ do} \quad \text{\# } T: \text{epoch 数量} \\
&\quad 5: \quad \text{令 } \mathbf{w}_i \leftarrow \mathbf{w} \quad \forall i \in \{1, \ldots, k\} \\
&\quad 6: \quad \text{令 } \mathbf{G}_i \leftarrow \mathbf{G} \quad \forall i \in \{1, \ldots, k\} \\
&\quad 7: \quad \textbf{for } i \in \{1, \ldots, k\} \textbf{ parallel do} \\
&\quad\quad 8: \quad\quad \textbf{for } \text{每个数据点 } \textbf{do} \\
&\quad\quad\quad 9: \quad\quad\quad \text{计算梯度 } \mathbf{g} \\
&\quad\quad\quad 10: \quad\quad\quad \text{更新 } \mathbf{G}_i: \mathbf{G}_i \leftarrow \mathbf{G}_i + \text{diag}(\mathbf{g} \mathbf{g}^T) \\
&\quad\quad\quad 11: \quad\quad\quad \text{更新 } \mathbf{w}_i: \mathbf{w}_i \leftarrow \mathbf{w}_i - \eta \mathbf{G}_i^{-1/2} \mathbf{g} \\
&\quad\quad 12: \quad\quad \mathbf{w} \leftarrow \sum_{i=1}^{k} \mathbf{w}_i / k \\
&\quad 13: \quad \mathbf{G} \leftarrow \sum_{i=1}^{k} \mathbf{G}_i \\
&14: \textbf{end for}
\end{aligned}
$$

## 4. 热启动

如第 2 节所述，我们定期重新训练模型。在图 1 中，假设每个训练集包含若干天的数据，我们在每一步向前移动几个小时，则训练集 #1 和 #2 之间会有大量重叠。这意味着从 #1 获得的模型可能与从 #2 获得的模型非常相似。对于逻辑回归，通过使用模型 #1 初始化模型 #2，获得模型 #2 的训练时间可以显著减少。该技术被称为热启动（warm-start）[6, 25, 8]。

对于逻辑回归——一个凸优化问题——无论是否使用热启动，模型最终都会收敛到全局最优。热启动仅影响收敛速度。然而，FFM 并非如此。为了解释原因，我们首先回顾 [14] 中已经研究过的 FFM 的一个不理想性质——我们没有一个好的 FFM 正则化方法，因此需要依赖早停法来防止过拟合。我们在图 2 中可视化了这一性质。为了获得最佳测试精度，必须仔细选择 epoch 数量——epoch 不足时模型可能欠拟合；epoch 过多时模型可能过拟合。为了确定最佳 epoch 数量，我们通常使用验证集来监控模型在每个 epoch 的性能。一旦验证损失上升，我们就停止训练过程。我们定义三个阶段来表示模型的"成熟度"：

- **未成熟（pre-mature）**：模型训练的 epoch 太少
- **成熟（mature）**：模型训练了足够的 epoch
- **过成熟（post-mature）**：模型训练的 epoch 太多

然而，使用早停法使得热启动难以应用。如果我们将一个成熟模型种子化到下一步并继续训练，那么新模型可能会过成熟。这个问题可以通过以下实验来证明。

我们再次使用 Criteo 的 CTR 预测挑战赛数据集以确保可重复性。我们将数据集分成 90 个块，每步使用 44 个块进行训练，1 个块进行验证，1 个块进行测试。因此，整个实验从第 46 个块（作为测试集）开始，每步向前移动一个块，到第 90 个块（作为测试集）结束。验证集用于确定 epoch 数量。我们首先比较了不使用任何热启动方法的基线设置与算法 3 中描述的朴素热启动——它简单地将每步结束时获得的模型种子化到下一步。

实验结果如图 3 所示，表明过成熟问题确实严重发生——随着实验向前推进，测试精度越来越差。同样注意，热启动技术的目标是在保持模型相同预测能力的同时减少训练时间。显然，通过使用 FFM 的朴素热启动，这一目标并未实现。

**算法 3：朴素热启动**

$$
\begin{aligned}
&\textbf{Require: } \text{初始模型 } \mathbf{w}_0 \\
&\mathbf{w} \leftarrow \mathbf{w}_0 \\
&\text{计算验证损失 } L_0 \\
&\textbf{for } t \in \{1, \ldots, T\} \textbf{ do} \\
&\quad \text{更新 } \mathbf{w} \\
&\quad \mathbf{w}_t \leftarrow \mathbf{w} \\
&\quad \text{计算验证损失 } L_t \\
&\quad \textbf{if } L_t > L_{t-1} \textbf{ then} \\
&\quad\quad \textbf{return } \mathbf{w}_{t-1} \\
&\textbf{end for}
\end{aligned}
$$

在本文中，我们提出了一种新的热启动方法，称为未成熟热启动。其思想是不将成熟模型种子化到下一步，而是使用一个未成熟模型作为种子。在每一步，由于新模型是用未成熟模型初始化的，它可能能够从新数据中学习而不会过拟合旧数据。例如，如果成熟模型出现在第 6 个 epoch，那么该模型将用于预测，但第 5 个 epoch 获得的模型将被种子化到下一步。

未成熟热启动的算法如算法 4 所述。这里，$\mathbf{w}_{t-1}$ 用于预测，$\mathbf{w}_{t-2}$ 被种子化。

**算法 4：我们提出的"未成熟"热启动**

$$
\begin{aligned}
&\textbf{Require: } \text{初始模型 } \mathbf{w}_{-1} \\
&\mathbf{w} \leftarrow \mathbf{w}_0 \leftarrow \mathbf{w}_{-1} \\
&\text{计算验证损失 } L_0 \\
&\textbf{for } t \in \{1, \ldots, T\} \textbf{ do} \\
&\quad \text{更新 } \mathbf{w} \\
&\quad \mathbf{w}_t \leftarrow \mathbf{w} \\
&\quad \text{计算验证损失 } L_t \\
&\quad \textbf{if } L_t > L_{t-1} \textbf{ then} \\
&\quad\quad \textbf{return } (\mathbf{w}_{t-1}, \mathbf{w}_{t-2}) \\
&\textbf{end for}
\end{aligned}
$$

### 4.1 离线结果

图 3 和 4 的实验结果表明，使用未成熟热启动后，测试性能不再差于基线，所需的 epoch 数量显著减少。

值得注意的是，使用热启动的 FFM 的对数损失随着实验的推进而降低。这表明 FFM 可能具有记忆过去所学信息的某种能力。受此观察的启发，我们尝试减小训练集的大小。图 5 展示了使用未成熟热启动的不同训练集大小的比较。我们看到，在足够多的步骤之后，仅使用 4 个块训练集的未成熟方法仍然优于使用 44 个块的基线。通过使用更小的训练集，训练变得更快。训练时间的比较如表 6 所示。如果我们使用 4 个块进行训练，则比基线快 20 倍。

**表 6：整个实验的总 epoch 数、平均每个 epoch 的时间和总训练时间。** 基线和未成熟都使用 44 个块作为训练数据。

| 设置 | 每 epoch 时间 | 总时间 |
|:---|:---:|:---:|
| 基线 | 236s | 20.6hr |
| 未成熟 | 236s | 6.8hr |
| 未成熟（8 块） | 47s | 1.5hr |
| 未成熟（4 块） | 26s | 0.9hr |
| 未成熟（2 块） | 13s | 0.5hr |

一个极端情况是将训练集大小减小到仅一个块。在这种情况下，由于两个连续步骤之间没有重叠，我们不必使用未成熟热启动。（未成熟热启动的目的是防止过拟合旧数据。）我们将此设置称为在线（online）。

图 5 显示在此设置下 FFM 仍然可以记忆信息，因为它仍然优于基线。然而，我们不使用此设置，原因有二。首先，我们提出的方法可以获得更好的对数损失。其次，从概念上讲，如果我们只使用非常小的一部分数据进行训练，模型可能对这小组数据的质量非常敏感。实际上，例如在图 5 的第 36 个 epoch，我们看到在线方法比基线差，而我们提出的方法仍然优于基线。

### 4.2 讨论

我们提出了两种减少训练时间的不同方法。分布式学习通过添加更多机器来减少训练时间，但同时也增加了计算量。（在我们之前的实验中，使用 32 台机器时，我们需要大约 3 倍的 epoch。）另一方面，热启动通过明智地初始化模型来减少训练时间，并需要更少的训练 epoch，这意味着计算量减少了。从某种意义上说，热启动似乎比分布式学习更好的方法。然而，我们无法完全用热启动替代分布式学习，因为有时需要冷启动，这意味着我们需要训练一个全新的模型。在实践中，当代码更新或系统遇到意外错误时会发生这种情况。在冷启动场景中，我们仍然需要依赖分布式学习来确保我们可以按时学习模型。

## 5. 结论

在本文中，我们展示了场感知分解机可以成功部署在大规模广告系统中，并显著改善业务指标，特别是对于小型广告主。FFM 的优势之一确实是通过使用 latent 表示比逻辑回归更好地泛化的能力。

此外，我们提出了两种加速 FFM 训练的方法：分布式学习和热启动。第 3 节和第 4 节中实验的代码可在线获取。作为未来工作，我们计划将热启动方法应用于我们的其他难以正则化的非凸问题，如深度神经网络。

## 6. 参考文献

[1] A. Agarwal, O. Chapelle, M. Dudík, and J. Langford. A reliable effective terascale linear learning system. The Journal of Machine Learning Research, 15(1):1111–1133, 2014.

[2] Y.-W. Chang, C.-J. Hsieh, K.-W. Chang, M. Ringgaard, and C.-J. Lin. Training and testing low-degree polynomial data mappings via linear svm. Journal of Machine Learning Research, 11(Apr):1471–1490, 2010.

[3] O. Chapelle. Modeling delayed feedback in display advertising. In Proceedings of the 20th ACM SIGKDD international conference on Knowledge discovery and data mining, pages 1097–1105. ACM, 2014.

[4] O. Chapelle. Offline evaluation of response prediction in online advertising auctions. In Proceedings of the 24th International Conference on World Wide Web Companion, pages 919–922. International World Wide Web Conferences Steering Committee, 2015.

[5] O. Chapelle, E. Manavoglu, and R. Rosales. Simple and scalable response prediction for display advertising. ACM Transactions on Intelligent Systems and Technology (TIST), 5(4):61, 2014.

[6] B.-Y. Chu, C.-H. Ho, C.-H. Tsai, C.-Y. Lin, and C.-J. Lin. Warm start for parameter selection of linear classifiers. In KDD, 2015.

[7] J. Dean, G. S. Corrado, R. Monga, K. Chen, M. Devin, Q. V. Le, M. Z. Mao, M. Ranzato, A. Senior, P. Tucker, K. Yang, and A. Y. Ng. Large scale distributed deep networks. In NIPS, 2012.

[8] D. DeCoste and K. Wagstaff. Alpha seeding for support vector machines. In KDD, 2000.

[9] J. Duchi, E. Hazan, and Y. Singer. Adaptive subgradient methods for online learning and stochastic optimization. JMLR, 12:2121–2159, 2011.

[10] B. Efron and R. J. Tibshirani. An introduction to the bootstrap. CRC press, 1994.

[11] T. Graepel, J. Q. Candela, T. Borchert, and R. Herbrich. Web-scale bayesian click-through rate prediction for sponsored search advertising in microsoft's bing search engine. In Proceedings of the 27th International Conference on Machine Learning (ICML-10), pages 13–20, 2010.

[12] X. He, J. Pan, O. Jin, T. Xu, B. Liu, T. Xu, Y. Shi, A. Atallah, R. Herbrich, S. Bowers, et al. Practical lessons from predicting clicks on ads at facebook. In Proceedings of 20th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pages 1–9. ACM, 2014.

[13] P. Hummel and R. P. McAfee. Loss functions for predicted click through rates in auctions for online advertising. Preprint, Google Inc, 2013.

[14] Y. Juan, Y. Zhaung, W.-S. Chin, and C.-J. Lin. Field-aware factorization machines for CTR prediction. In RecSys, 2016.

[15] K.-c. Lee, B. Orten, A. Dasdan, and W. Li. Estimating conversion rate in display advertising from past erformance data. In Proceedings of the 18th ACM SIGKDD international conference on Knowledge discovery and data mining, pages 768–776. ACM, 2012.

[16] D. Lefortier, A. Truchet, and M. de Rijke. Sources of variability in large-scale machine learning systems. In Machine Learning Systems (NIPS 2015 Workshop), 2015.

[17] M. Li, D. G. Andersen, A. Smola, and K. Yu. Communication efficient distributed machine learning with the parameter server. In Proceedings of the 27th International Conference on Neural Information Processing Systems, NIPS'14, pages 19–27, Cambridge, MA, USA, 2014. MIT Press.

[18] M. Li, Z. Liu, A. J. Smola, and Y.-X. Wang. Difacto: Distributed factorization machines. In Proceedings of the Ninth ACM International Conference on Web Search and Data Mining, pages 377–386. ACM, 2016.

[19] R. McDonald, K. Hall, and G. Mann. Distributed training strategies for the structured perceptron. In Human Language Technologies: The 2010 Annual Conference of the North American Chapter of the Association for Computational Linguistics, HLT '10, pages 456–464, Stroudsburg, PA, USA, 2010. Association for Computational Linguistics.

[20] H. B. McMahan, G. Holt, D. Sculley, M. Young, D. Ebner, J. Grady, L. Nie, T. Phillips, E. Davydov, D. Golovin, et al. Ad click prediction: a view from the trenches. In Proceedings of the 19th ACM SIGKDD international conference on Knowledge discovery and data mining, pages 1222–1230. ACM, 2013.

[21] R. J. Oentaryo, E.-P. Lim, J.-W. Low, D. Lo, and M. Finegold. Predicting response in mobile advertising with hierarchical importance-aware factorization machine. In Proceedings of the 7th ACM international conference on Web search and data mining, pages 123–132. ACM, 2014.

[22] S. Rendle. Factorization machines with libFM. ACM Transactions on Intelligent Systems and Technology (TIST), 3(3):57, 2012.

[23] R. Rosales, H. Cheng, and E. Manavoglu. Post-click conversion modeling and analysis for non-guaranteed delivery display advertising. In Proceedings of the fifth ACM international conference on Web search and data mining, pages 293–302. ACM, 2012.

[24] G. Song. Criteo display advertising challenge. Available at https://www.kaggle.com/c/criteo-display-ad-challenge/forums/t/10547/document-and-code-for-the-3rd-place-finish, 2014.

[25] C.-H. Tsai, C.-Y. Lin, and C.-J. Lin. Incremental and decremental training for linear classification. In KDD, 2014.

[26] F. Vasile, D. Lefortier, and O. Chapelle. Cost-sensitive learning for utility optimization in online advertising auctions. arXiv preprint arXiv:1603.03713, 2016.

[27] K. Weinberger, A. Dasgupta, J. Langford, A. Smola, and J. Attenberg. Feature hashing for large scale multitask learning. In Proceedings of the 26th Annual International Conference on Machine Learning, pages 1113–1120. ACM, 2009.

[28] W. Zhang, T. Du, and J. Wang. Deep learning over multi-field categorical data. In European Conference on Information Retrieval, pages 45–57. Springer, 2016.

[29] M. Zinkevich, M. Weimer, L. Li, and A. J. Smola. Parallelized stochastic gradient descent. In J. D. Lafferty, C. K. I. Williams, J. Shawe-Taylor, R. S. Zemel, and A. Culotta, editors, Advances in Neural Information Processing Systems 23, pages 2595–2603. Curran Associates, Inc., 2010.
