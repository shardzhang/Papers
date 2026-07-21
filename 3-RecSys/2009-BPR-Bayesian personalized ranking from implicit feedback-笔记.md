

## 笔记

业务场景：无，学术型论文

业务指标：无

离线指标：AUC

数据集：公开数据集

- **Online shopping: Rossmann**：电商场景。Rossmann：德国大型日化连锁超市，主营洗护、食品、药妆，同时拥有线上商城。

- **Video rental: Netflix**：流媒体。影视租赁：网飞，全球流媒体平台，早期以 DVD 影片租赁起家，现主营影视剧线上点播服务。

模型类型：rank，不是点击率预估

创新点：

- 样本构造：对于隐式评分数据集，将所有评分样本作为正样本，再从用户无评分item中随机采样作为负样本

个人收获：

- 处理Netflix隐式评分数据集的逻辑可以借鉴。

疑问：

- MF和KNN中对于梯度的更新，是交给pytorch自动计算梯度，还是手动应用梯度？





## 公式推导

### 1. kNN 模型的梯度推导

在 kNN 模型中，预测分数定义为：

$$
\hat{x}_{ui} = \sum_{l \in I^{+}_u, l \neq i} c_{il}
$$

其中 $C$ 是对称的 item-item 相似度矩阵，$c_{il}$ 表示 item $i$ 和 item $l$ 之间的相似度。

而 $\hat{x}_{uij} = \hat{x}_{ui} - \hat{x}_{uj}$，代入后得到：

$$
\hat{x}_{uij} = \sum_{l \in I^{+}_u, l \neq i} c_{il} - \sum_{l \in I^{+}_u, l \neq j} c_{jl}
$$

#### 梯度公式

$$
\frac{\partial \hat{x}_{uij}}{\partial \theta} =
\begin{cases}
+1, & \text{if } \theta \in \{c_{il}, c_{li}\} \land l \in I^{+}_u \land l \neq i,\\
-1, & \text{if } \theta \in \{c_{jl}, c_{lj}\} \land l \in I^{+}_u \land l \neq j,\\
0, & \text{otherwise}
\end{cases}
$$

#### 三种情况分解

| 条件 | 梯度 | 含义 |
|------|------|------|
| $\theta = c_{il}$ 或 $c_{li}$，且 $l$ 是用户 $u$ 交互过的非 $i$ item | **$+1$** | 增加 $c_{il}$ 会提高 $\hat{x}_{ui}$，从而**提高** $\hat{x}_{uij}$ |
| $\theta = c_{jl}$ 或 $c_{lj}$，且 $l$ 是用户 $u$ 交互过的非 $j$ item | **$-1$** | 增加 $c_{jl}$ 会提高 $\hat{x}_{uj}$，从而**降低** $\hat{x}_{uij}$ |
| $\theta$ 是其他参数 | **$0$** | 不影响当前三元组的排序分数差 |

#### $c_{il}$ 和 $c_{li}$ 同时出现的原因

由于 $C$ 是对称矩阵（$c_{il} = c_{li}$），两者是同一个参数。BPR 同时列出只是为了明确对称性，实际梯度贡献相同。

---



### 2. MF 模型的梯度推导

在 BPR-MF 中，预测分数定义为：

$$
\hat{x}_{ui} = \langle w_u, h_i \rangle = \sum_{f=1}^{k} w_{uf} \cdot h_{if}
$$

其中 $w_u$ 是用户 $u$ 的潜在因子向量，$h_i$ 是 item $i$ 的潜在因子向量。而 $\hat{x}_{uij} = \hat{x}_{ui} - \hat{x}_{uj}$。

#### 梯度公式

$$
\frac{\partial \hat{x}_{uij}}{\partial \theta} =
\begin{cases}
(h_{if} - h_{jf}), & \text{if } \theta = w_{uf},\\
w_{uf}, & \text{if } \theta = h_{if},\\
-w_{uf}, & \text{if } \theta = h_{jf},\\
0, & \text{else}
\end{cases}
$$

#### 三种情况的含义

| $\theta$ | 梯度 | 直观理解 |
|----------|------|---------|
| $w_{uf}$（用户因子） | $(h_{if} - h_{jf})$ | 更新用户向量时，同时考虑正负样本 item 的差距方向 |
| $h_{if}$（正样本 item 因子） | $+w_{uf}$ | 提高正样本的分数 → 沿用户向量方向移动 $h_i$ |
| $h_{jf}$（负样本 item 因子） | $-w_{uf}$ | 降低负样本的分数 → 沿用户向量反方向移动 $h_j$ |

#### 梯度如何驱动学习

以 BPR 的 SGD 更新规则为例：

$$
\Theta \leftarrow \Theta + \alpha \left( \frac{e^{-\hat{x}_{uij}}}{1 + e^{-\hat{x}_{uij}}} \cdot \frac{\partial}{\partial \Theta} \hat{x}_{uij} + \lambda_\Theta \Theta \right)
$$

将梯度代入：

| 参数 | 更新方向 | 效果 |
|------|---------|------|
| $w_{uf}$ | $\propto (h_{if} - h_{jf})$ | 当 $h_{if} > h_{jf}$ 时（正样本已优于负样本），梯度为正，增大 $w_{uf}$；反之减小 |
| $h_{if}$（正样本） | $\propto +w_{uf}$ | 始终沿 $w_u$ 方向推动 $h_i$，提高正样本分数 |
| $h_{jf}$（负样本） | $\propto -w_{uf}$ | 始终沿 $w_u$ 反方向推动 $h_j$，降低负样本分数 |

---



### 3. MF 与 kNN 的梯度和对比

#### 梯度对比表

| 模型 | 正样本参数梯度 | 负样本参数梯度 | 用户参数梯度 |
|------|--------------|--------------|-------------|
| **MF** | $+w_{uf}$（对 $h_{if}$） | $-w_{uf}$（对 $h_{jf}$） | $(h_{if} - h_{jf})$（对 $w_{uf}$） |
| **kNN** | $+1$（对 $c_{il}$） | $-1$（对 $c_{jl}$） | 无用户参数 |

两种模型的梯度结构一致：**对正样本相关参数梯度为正，对负样本相关参数梯度为负**，这反映了 BPR-Opt 优化准则的统一思想——拉大正负样本之间的分数差距。

#### MF 用户因子 $w_u$ 的协同机制

MF 相比 kNN 多了一个用户因子 $w_u$，其梯度 $(h_{if} - h_{jf})$ 由正负样本 item 因子的差值决定。这里需要正确理解 $w_u$ 的角色：

**三者协同效果**（$w_u$ 作为锚点）：

| 参数                                            | 梯度方向        | 效果                                       |
| ----------------------------------------------- | --------------- | ------------------------------------------ |
| $w_u \leftarrow w_u + \alpha \cdot (h_i - h_j)$ | **$h_i - h_j$** | $w_u$ 被拉向正样本 $h_i$，推离负样本 $h_j$ |
| $h_i \leftarrow h_i + \alpha \cdot w_u$         | **$+w_u$**      | 正样本被拉向 $w_u$                         |
| $h_j \leftarrow h_j - \alpha \cdot w_u$         | **$-w_u$**      | 负样本被推离 $w_u$                         |

$w_u + (h_i - h_j)$ 的含义：以 BPR 的 SGD 更新为例（忽略正则化和系数）：

$$
w_u \leftarrow w_u + \alpha \cdot \sigma(-\hat{x}_{uij}) \cdot (h_i - h_j)
$$
直观类比——$w_u$ 是用户 $u$ 的"兴趣锚点"：

```
h_j ←--- w_u ---→ h_i
       ↑
    w_u 向 h_i 移动（被正样本吸引）
    w_u 远离 h_j（被负样本排斥）
```

- $h_i$（正样本）**被拉向 $w_u$** → 提高正样本分数 $\hat{x}_{ui} = \langle w_u, h_i \rangle$
- $h_j$（负样本）**被推离 $w_u$** → 降低负样本分数 $\hat{x}_{uj} = \langle w_u, h_j \rangle$
- $w_u$ 作为参考系中心，其自身的运动方向由 $h_i$ 和 $h_j$ 的差值决定

三者协同，结果是 $\hat{x}_{ui} - \hat{x}_{uj}$ 的间隔被放大。

kNN 中没有用户参数 $w_u$，不存在这种"平衡"——kNN 的梯度只单纯地给正样本相似度 $+1$、负样本 $-1$。这也是 MF 比 kNN 更具表达力的原因之一：**用户向量可以在潜在空间中自适应地调整位置，而不只是靠固定的相似度度量。**



### 4. BPR学习算法 VS 随机梯度下降算法

本质区别：BPR 的 LearnBPR 不是"改良的 SGD"，而是"采样策略 + SGD"

两者的核心差异在**训练三元组的采样方式**，而非优化算法本身。

#### 标准 SGD（用于 BPR 的朴素方式）

按用户或按 item 顺序遍历三元组 $(u, i, j)$：

```py
for user u in U:
    for item i in I⁺_u:
        for item j in I \ I⁺_u:
            SGD_update(u, i, j)
```

**问题**：同一个 $(u, i)$ 会连续更新 $|I \setminus I⁺_u|$ 次（因为要遍历所有负样本 $j$），导致梯度被热门 item 主导、收敛缓慢。BPR 论文第 4.2 节指出这种偏斜导致需要极小的学习率。

#### LearnBPR（BPR 提出的方法）

**用带放回的自助采样（bootstrap sampling）替代顺序遍历**：

```py
for step in 1..T:
    (u, i, j) ∼ uniform(D_S)  ← 从所有三元组中均匀随机采样
    SGD_update(u, i, j)
```

| 维度           | 标准 SGD（顺序遍历）   | LearnBPR（自助采样）             |
| -------------- | ---------------------- | -------------------------------- |
| 数据访问方式   | 按用户/item 顺序遍历   | **均匀随机采样**，带放回         |
| 热门 item 影响 | 梯度被热门 item 主导   | 随机采样削弱了偏斜               |
| 收敛步数       | 需要完整遍历所有三元组 | **无需完整遍历**，少量步即可收敛 |
| 停止条件       | 遍历完所有数据         | **可随时停止**（因为带放回）     |

#### 关键洞察

LearnBPR 不是新的优化器，而是 **SGD + 一种特定的采样分布**。BPR 论文的核心贡献在于指出了**在成对排序损失下，采样策略比优化器本身更重要**——均匀随机采样 + 带放回 + 可随时停止，这三者组合解决了三元组数据固有偏斜问题，让 SGD 能在成对排序场景中高效工作。
