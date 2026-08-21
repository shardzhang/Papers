重要意义：13年KDD最佳论文。Google工程团队分享了广告点击率预估系统在实际部署中的经验教训。该论文让FTRL成为业界处理稀疏大规模在线学习问题的经典算法。

业务场景：点击率预估

业务指标：auc

离线指标：

数据集：

模型类型：

创新点：

个人收获：

- FTRL为什么能够成功？
  - OGD在线梯度下降算法：使用简单的L1正则进行稀疏化。由于对重要特征没有保护，如果正则化系数 $\lambda_1$ 设置太大会导模型整体参数量变小，但模型精度下降问题。为了兼顾效果，一般使用较小的 $\lambda_1$ 导致模型稀疏性较弱。
  - FTRL：主要三点改进，一是考虑历史累计梯度的学习率自适应，对高频重要特征使用小学习率缓慢更新，对低频不重要特征使用大学习率快速更新。同时使用近端正则化项（包括了L2正则化项），防止梯度出现漂移，稳住参数更新方向。三是使用L1正则进行稀疏化。L2正则转移到计算历史累计梯度，对于单个样本的梯度波动具有较好平滑效果，重要特征历史累计梯度较大，不会被L1正则置零。不重要特征通过L1正则置零。L1和L2职责拆分，L1负责参数置零，L2负责保护重要特征。因此在稀疏性和精度上都有较好的平衡。

英文：第4节不理解



## FQA

Q：OGD 本质上与随机梯度下降相同；名称"在线"强调我们不是在解决**批处理问题**，而是对可能 **不是 IID 的样本序列** 进行预测。

这句话澄清了两个概念的区别：

传统 SGD（随机梯度下降）

- 假设数据是 **IID（独立同分布）** 的
- 目标是优化一个**固定的目标函数** $\min_\mathbf{w} \frac{1}{N}\sum \ell(\mathbf{w}, \mathbf{x}_i, y_i)$
- 数据可以**打乱后反复遍历**（epoch）

OGD（在线梯度下降）

- 数据**不是 IID** 的——样本序列可能有**时序相关性**（比如用户搜索行为）
- 没有"训练集"的概念，数据是**流式到达**的，每个样本**只处理一次就被丢弃**
- 不假设存在一个固定的目标函数——环境可能在变化（概念漂移）

一句话总结

**SGD 和 OGD 的更新公式完全一样（$\mathbf{w} \leftarrow \mathbf{w} - \eta \mathbf{g}$），但背后的假设不同：**SGD 假设数据是静态的、可重复使用的；OGD 面对的是数据流，需要在新数据到来时持续适应，一次处理完就扔掉。论文用"在线"这个名称是为了强调后者。



Q：在线学习与稀疏性— FTRL-Proximal 算法。这是论文最核心的算法贡献。

问题背景

- 广告点击率预测是**大规模线性模型**（逻辑回归）问题
- 特征空间极其稀疏：**数十亿维特征，但每个样本只有几百个非零值**
- 需要在线学习：数据流式到达，每个样本只处理一次

核心矛盾

- OGD（在线梯度下降）：预测精度好，但不能产生稀疏模型（所有系数非零）
- RDA（正则化对偶平均）：稀疏性好，但精度不如OGD
- FOBOS：介于两者之间但都不够好



#### Q：对于大规模学习，广义线性模型（如逻辑回归）的在线算法具有许多优势。尽管特征向量 $x$ 可能有数十亿维度，但通常每个实例只有数百个非零值。这使得通过从磁盘或网络流式传输样本即可在大规模数据集上进行高效训练成为可能 [3]，因为每个训练样本只需要被处理一次。 这段话是什么意思？

这段话解释的是 **广义线性模型 在工业级规模下的实用性优势**：

1. **特征极高维但稀疏** — 广告点击预估中特征向量可能有数十亿维（如用户 ID × 广告 ID 交叉），但单个样本通常只有几百个非零特征（绝大多数是 0）。这种**天然稀疏性**使得模型只需存储和处理非零特征，计算量不随总维度增长。

2. **Online 学习 = 一次遍历** — 样本流式读取，处理完即可丢弃，不需要像 batch 方法那样**反复扫描数据集**。对于 PB 级日志数据，"只过一遍"是唯一可行方案（内存/磁盘都存不下全部数据）。

3. **线性模型计算量极低** — 逻辑回归的梯度计算 $\nabla w = (p - y)x$，对于稀疏 $x$ 只需更新非零权重，一步 O(非零特征数)。深度网络则需前向+反向传播全连接，计算量高几个数量级，无法在同等数据量下以低成本训练。

一句话总结：**特征稀疏 + 线性模型 + online 更新 = 单次扫描即可训练，内存和计算都只跟 非零特征数 相关，与总维度无关**，这让数十亿维度、万亿样本的训练变得可行。



#### Q：batch_size越大越好么？还是越小越好？

都不是，存在 tradeoff：是梯度估计准确性、训练速度、显存开销/利用率之间的权衡

batch size 太小（如 1）：

- GPU现存开销小
- GPU硬件利用率低
- 梯度噪声大，收敛不稳定
- 训练速度慢。训练单个epoch需要的step=样本条数/batch_size

batch size 太大：

- 梯度估计准确，收敛稳定
- 内存/显存占用高，对于大内存显卡的利用率高
- 训练速度快



#### Q：Algorithm 1 详解

$$
\eta_{t,i} = \frac{\alpha}{\beta + \sqrt{\sum_{s=1}^t g_{s,i}^2}} \qquad (2)
$$

这是 FTRL-Proximal 算法用于逻辑回归的每坐标实现，结合 L1 + L2 正则化。



#### Q：FTRL训练LR的完整流程

**Algorithm 1** Per-Coordinate FTRL-Proximal with L1 and L2 Regularization for Logistic Regression
$$
\begin{aligned}
&\text{\# With per-coordinate learning rates of Eq. (2).} \\
&\textbf{Input: } \text{parameters } \alpha, \beta, \lambda_1, \lambda_2 \\
&(\forall i \in \{1, \dots, d\}),\ \text{initialize } z_i = 0 \text{ and } n_i = 0 \\
&\textbf{for } t = 1 \textbf{ to } T \textbf{ do} \\
&\quad \text{Receive feature vector } \mathbf{x}_t \text{ and let } I = \{i \mid x_i \neq 0\} \\
&\quad \textbf{For } i \in I \textbf{ compute} \\
&\quad\quad w_{t,i} =
\begin{cases}
0 & \text{if } |z_i| \leq \lambda_1 \\[6pt]
-\left( \dfrac{\beta + \sqrt{n_i}}{\alpha} + \lambda_2 \right)^{-1} (z_i - \operatorname{sgn}(z_i) \lambda_1) & \text{otherwise.}
\end{cases} \\
&\quad \text{Predict } p_t = \sigma(\mathbf{x}_t \cdot \mathbf{w}) \text{ using the } w_{t,i} \text{ computed above} \\
&\quad \text{Observe label } y_t \in \{0, 1\} \\
&\quad \textbf{for all } i \in I \textbf{ do} \\
&\quad\quad g_i = (p_t - y_t) x_i \quad \text{\# gradient of loss w.r.t. } w_i \\
&\quad\quad \sigma_i = \dfrac{\sqrt{n_i + g_i^2} - \sqrt{n_i}}{\alpha} \\
&\quad\quad z_i \leftarrow z_i + g_i - \sigma_i w_{t,i} \\
&\quad\quad n_i \leftarrow n_i + g_i^2 \\
&\quad \textbf{end for} \\
&\textbf{end for}
\end{aligned}
$$

**关键设计要点**：

- **每坐标学习率** $\eta_{t,i} = \frac{\alpha}{\beta + \sqrt{n_i}}$：$n_i$ 累积 $g_i^2$，频繁特征 $n_i$ 大 → $\eta_{t,i}$ 小 → 权重稳定
- **惰性表示** $z_i$：不直接存 $w$，而是在需要时从 $z$ 按闭式解求出 $w$。$|z_i| \leq \lambda_1$ 时 $w=0$，实现精确 L1 稀疏化
- **$\sigma_i$ 的作用**：$g_i - \sigma_i w_{t,i}$ 等价于 OGD 的 $g_i$ 加上"学习率衰减权重修正"。该修正使得即使学习率随时间下降，历史梯度也不会被过度放大
- **$\lambda_2$ 在分母中**：直接缩小非零 $w$，等价于标准 L2 正则化；$\lambda_2 = 0$ 退化为纯 L1 截断



#### Q：为什么比 OGD 更好？

OGD 更新 $w_{t+1} = w_t - \eta_t g_t$，在 L1 正则化下难以产生精确的零值。FTRL-Proximal 改用**惰性表示**（存储 $z$ 而非 $w$），在 $w$ 上施加显式 L1 截断门限，每次更新时直接做 **soft thresholding**：$|z_i| \leq \lambda_1 \Rightarrow w_i = 0$，从而获得高稀疏性且不牺牲精度。



#### Q：在线梯度下降的标准理论建议使用全局学习率调度 $\eta_t = 1/\sqrt{t}$，这**对所有坐标都是通用的** [34]。 「每坐标」是什么意思？

"每坐标"就是**每个特征维度独立拥有自己的学习率**，而非所有特征共用一个全局学习率。

用论文里的抛硬币例子解释：

- **全局学习率** $\eta_t = 1/\sqrt{t}$：不管第 $t$ 步更新的是哪个特征，所有 $d$ 个维度的学习率都按同一个时间表下降。即使特征 $i$ 这轮没被更新，它的学习率也在降低。
- **每坐标学习率** $\eta_{t,i} = \alpha/(\beta + \sqrt{n_{t,i}})$：每个特征 $i$ 根据**自己**被更新的次数 $n_i$ 独立计算学习率。$n_i$ 大（出现频繁）→ 学习率小（已充分学习）；$n_i$ 小（罕见特征）→ 学习率大（需要快速适应）。

**直观类比**：全局学习率像一个班级按统一进度上课；每坐标学习率像每个学生按自己的掌握速度独立学习——频繁出现的特征（学了很多遍）放慢节奏，罕见特征（刚遇到）保持高学习率快速吸收。



#### Q：在线学习与稀疏性。这部分的公式我是一点没看懂

好的，我用人话逐条解释这些公式到底在做什么。

$$
w_{t+1} = \arg\min_w \left( g_{1:t} \cdot w + \frac{1}{2} \sum_{s=1}^t \sigma_s \|w - w_s\|^2 + \lambda_1 \|w\|_1 \right)
$$

这个式子看起来很吓人，实际上拆开看：

| 部分                                     | 人话                                                         |
| ---------------------------------------- | ------------------------------------------------------------ |
| $g_{1:t} \cdot w$                        | 新的 $w$ 在所有历史数据上表现怎样（最小化累积损失）。TODO：为什么是这个意思？ |
| $\frac{1}{2}\sum \sigma_s \|w - w_s\|^2$ | **"别跑太远"**：新权重不要和每步的旧权重偏离太多（陡坡稳定） |
| $\lambda_1 \|w\|_1$                      | **L1 惩罚**：鼓励权重等于精确的 0（产生稀疏性，节省内存）    |

每一轮它做的事就是：**"在历史数据上拟合得好的前提下，权重尽量不剧烈变化，同时尽可能把不重要的权重压成 0。"**

---

**公式 — 简化版 $w$ 更新（每坐标）**

$$
w_{t+1,i} =
\begin{cases}
0 & \text{if } |z_{t,i}| \leq \lambda_1 \\
-\eta_t (z_{t,i} - \operatorname{sgn}(z_{t,i}) \lambda_1) & \text{otherwise}
\end{cases}
$$

这其实就是上面那个又长又吓人的公式解出来的**闭式解**，实际代码里算的就是这个。

- $z_{t,i}$ 是一个累积的"中间变量"（可以理解为**带衰减的负梯度累积和**）
- 如果 $|z_i|$ 很小（小于 $\lambda_1$），那么 $w_i$ **直接设成 0** → 这就是**产生稀疏性的核心机制**
- 否则 $w_i$ 取非零值，并且从 $z_i$ 中减去 $\lambda_1$（这叫 **soft-thresholding**，软阈值）

**类比**：想象 $z_i$ 是"这个特征有用程度的证据"。证据不够强（$|z_i| \leq \lambda_1$）就认为它没用，权重砍到 0 彻底丢弃它。



Q：FTRL-Proximal中的Proximal是什么含义？

**"Proximal"（近端）来源于优化理论中的近端算子（proximal operator）。**

在 FTRL-Proximal 中，它体现在更新公式中的**近端正则项**：

$$
\mathbf{w}_{t+1} = \arg\min_{\mathbf{w}} \left( \mathbf{g}_{1:t} \cdot \mathbf{w} + \frac{1}{2} \sum_{s=1}^t \sigma_s \|\mathbf{w} - \mathbf{w}_s\|^2 + \lambda_1 \|\mathbf{w}\|_1 \right)
$$

关键在第二项 $\frac{1}{2} \sum_{s=1}^t \sigma_s \|\mathbf{w} - \mathbf{w}_s\|^2$：

直觉理解

- **"近端"的含义**：每次更新时，新的权重 $\mathbf{w}_{t+1}$ 不能偏离之前所有权重 $\mathbf{w}_s$（$s=1..t$）太远。这一项像一个"锚"，把新解**拉近**到历史解附近。

- 它与 OGD 的关系：当 $\lambda_1=0$ 时，这个看起来很复杂的更新**等价于标准梯度下降**。论文中的一段话很关键：

  > "表面上这些更新看起来非常不同，但事实上当取 $\lambda_1=0$ 时，它们会产生完全相同的系数向量序列。"





#### Q：为什么需要"近端"结构？

论文通过 **近端形式的改写**，把更新重写成：

$$
\left( \mathbf{g}_{1:t} - \sum_{s=1}^t \sigma_s \mathbf{w}_s \right) \cdot \mathbf{w} + \frac{1}{\eta_t} \|\mathbf{w}\|_2^2 + \lambda_1 \|\mathbf{w}\|_1
$$

这样 L1 正则化项 $\lambda_1 \|\mathbf{w}\|_1$ 就可以在**每坐标基础上通过闭式解**直接求解——即那个分段函数：

$$
w_{t+1,i} = 
\begin{cases}
0 & \text{if } |z_{t,i}| \leq \lambda_1 \\
-\eta_t (z_{t,i} - \operatorname{sgn}(z_{t,i})\lambda_1) & \text{otherwise}
\end{cases}
$$

这就是产生**精确零系数**（稀疏性）的根源。

一句话总结

**Proximal = 用近端正则项约束新权重靠近历史权重，这种结构使得L1正则化可以在每坐标上闭式求解，从而同时获得梯度下降的精度和RDA的稀疏性。**





#### Q：在没有正则化的情况下，该算法 与 标准在线梯度下降相同，但由于它使用了模型系数 $\mathbf{w}$ 的替代惰性表示，L1正则化可以更有效地实现。 

这里的「惰性表示」是什么意思？

"惰性表示"（lazy representation）指的是 FTRL-Proximal **不直接存 $\mathbf{w}$，而是存一个辅助变量 $\mathbf{z}_t$**，$\mathbf{w}$ 仅在需要时才从这个中间表示计算出来。

具体来说：

| 算法          | 存储的变量                                                   |
| ------------- | ------------------------------------------------------------ |
| OGD           | $\mathbf{w}_t$（直接存系数）                                 |
| FTRL-Proximal | $\mathbf{z}_t = \mathbf{g}_{1:t-1} - \sum_{s=1}^{t-1} \sigma_s \mathbf{w}_s$（存的是"经过变换的梯度累积"） |

为什么叫"惰性"？

因为从 $\mathbf{z}_t$ 计算 $\mathbf{w}_{t+1}$ 的规则是：

$$
w_{t+1,i} = 
\begin{cases}
0 & \text{if } |z_{t,i}| \leq \lambda_1 \\
-\eta_t (z_{t,i} - \operatorname{sgn}(z_{t,i})\lambda_1) & \text{otherwise}
\end{cases}
$$

这里有个关键点：**当 $|z_{t,i}| \leq \lambda_1$ 时，$w_{t+1,i} = 0$**。在 OGD 中，即使系数最终趋近于 0，每次更新也要做完整的加减运算。而 FTRL 通过这种间接表示，**只有到真正需要 $w$ 时才去计算，而且可以直接"跳过"那些被 L1 阈值压到零的坐标**。

对比理解

- **OGD（直接表示）**：$\mathbf{w} \leftarrow \mathbf{w} - \eta \mathbf{g}$。每一步必须更新所有出现过非零值的坐标，即使它们最终会接近 0。
- **FTRL（惰性表示）**：$\mathbf{z} \leftarrow \mathbf{z} + \mathbf{g} + (\frac{1}{\eta_t} - \frac{1}{\eta_{t-1}})\mathbf{w}$。只在需要预测时才从 $\mathbf{z}$ **按需解出** $\mathbf{w}$，而那些被 L1 阈值截止的坐标直接解出 0，**不需要额外存储或操作**。

这就是文中说的"替代惰性表示"——**不是直接存储模型系数本身，而是存储一个可以延迟推导出系数的中间量**。这使得稀疏化（产生大量零系数）变得非常高效。



#### Q：表面上这些更新看起来非常不同，但事实上当取 $\lambda_1=0$ 时，它们会产生完全相同的系数向量序列。

为什么？如何证明？

这是一个纯粹的代数推导。分两步来证明：

---

1. 常数学习率的情况（$\eta_t = \eta$）

**OGD** 展开：

$$\mathbf{w}_2 = \mathbf{w}_1 - \eta \mathbf{g}_1$$
$$\mathbf{w}_3 = \mathbf{w}_2 - \eta \mathbf{g}_2 = \mathbf{w}_1 - \eta(\mathbf{g}_1 + \mathbf{g}_2)$$
$$\vdots$$
$$\mathbf{w}_{t+1} = \mathbf{w}_1 - \eta \sum_{s=1}^t \mathbf{g}_s$$

若 $\mathbf{w}_1 = 0$：$$\mathbf{w}_{t+1} = -\eta \sum_{s=1}^t \mathbf{g}_s$$



**FTRL-Proximal**（$\lambda_1=0$）：

$$\mathbf{w}_{t+1} = \arg\min_{\mathbf{w}} \left( \mathbf{g}_{1:t} \cdot \mathbf{w} + \frac{1}{2} \sum_{s=1}^t \sigma_s \|\mathbf{w} - \mathbf{w}_s\|^2 \right)$$

其中 $\sigma_{1:t} = \sum_{s=1}^t \sigma_s = 1/\eta_t$。

当 $\eta_t = \eta$（常数）：$\sigma_1 = 1/\eta$，$\sigma_2 = \sigma_3 = \cdots = 0$。（TODO：为什么这样假设？）

于是优化问题简化为：

$$\mathbf{w}_{t+1} = \arg\min_{\mathbf{w}} \left( \sum_{s=1}^t \mathbf{g}_s \cdot \mathbf{w} + \frac{1}{2\eta} \|\mathbf{w} - \mathbf{w}_1\|^2 \right)$$

**对 $\mathbf{w}$ 求梯度并设为零**（这是一个凸优化，极值点即最优解）：

$$\sum_{s=1}^t \mathbf{g}_s + \frac{1}{\eta} (\mathbf{w} - \mathbf{w}_1) = 0$$

解得：

$$\mathbf{w}_{t+1} = \mathbf{w}_1 - \eta \sum_{s=1}^t \mathbf{g}_s$$

若 $\mathbf{w}_1 = 0$：$$\mathbf{w}_{t+1} = -\eta \sum_{s=1}^t \mathbf{g}_s$$

**与 OGD 完全相同。** ✓

---

2. 非恒定学习率的情况

这时等价性依赖于 $\sigma_s$ 的**精确定义**。给定学习率序列 $\eta_t$，定义：

$$\sigma_t = \frac{1}{\eta_t} - \frac{1}{\eta_{t-1}} \quad (\text{约定 } 1/\eta_0 = 0)$$

则 $\sigma_{1:t} = \sum_{s=1}^t \sigma_s = 1/\eta_t$，满足条件。

FTRL 的优化目标展开：

$$\mathbf{g}_{1:t} \cdot \mathbf{w} + \frac{1}{2} \sum_{s=1}^t \sigma_s (\mathbf{w}^T\mathbf{w} - 2\mathbf{w}^T\mathbf{w}_s + \|\mathbf{w}_s\|^2)$$

展开平方项后，系数为：

$$= \underbrace{\frac{1}{2}\left(\sum_{s=1}^t \sigma_s\right)}_{=1/2\eta_t} \|\mathbf{w}\|_2^2 + \left( \mathbf{g}_{1:t} - \sum_{s=1}^t \sigma_s \mathbf{w}_s \right) \cdot \mathbf{w} + \text{常数}$$

这是 $\mathbf{w}$ 的**二次函数**，梯度为零给出闭式解：

$$\frac{1}{\eta_t} \mathbf{w}_{t+1} + \mathbf{g}_{1:t} - \sum_{s=1}^t \sigma_s \mathbf{w}_s = 0$$

$$\mathbf{w}_{t+1} = \eta_t \left( \sum_{s=1}^t \sigma_s \mathbf{w}_s - \mathbf{g}_{1:t} \right)$$

而 **OGD 的多步展开是**：

$$\mathbf{w}_{t+1} = \mathbf{w}_t - \eta_t \mathbf{g}_t = \mathbf{w}_1 - \sum_{s=1}^t \eta_s \mathbf{g}_s \qquad (\text{设 } \mathbf{w}_1 = 0)$$

这两个式子**形式不同**。关键的洞察在论文后面：

> 注意当 $\eta_t$ 是常数 $\eta$ 且 $\lambda_1=0$ 时，很容易看出与在线梯度下降的等价性

也就是说，**只在常数学习率时两者表达式完全一样**。对于自适应学习率，FTRL 的更新公式虽然与 OGD 的展开形式不同，但**通过惰性表示 $\mathbf{z}_t$ 的递推更新**，可以实现同等的学习效果。两者的等价性是在**算法层面**（通过替代表示实现相同的系数序列），而非简单的公式展开相等。

如果你想看更严格的证明，可以参考 McMahan 2011 年的 AISTATS 论文 *"Follow-the-regularized-leader and mirror descent: Equivalence theorems and L1 regularization"*。



Q：$\sigma_s$ 这是什么？

$\sigma_s$ 是 FTRL-Proximal 算法中用来**连接学习率 $\eta_t$ 和近端正则项**的一个辅助量。

论文中定义：

$$
\sigma_s = \frac{1}{\eta_s} - \frac{1}{\eta_{s-1}} \quad (\text{约定 } 1/\eta_0 = 0)
$$

使得：

$$
\sigma_{1:t} = \sum_{s=1}^t \sigma_s = \frac{1}{\eta_t}
$$

直觉理解

OGD 的更新是 $\mathbf{w}_{t+1} = \mathbf{w}_t - \eta_t \mathbf{g}_t$，FTRL 的近端项 $\frac{1}{2}\sum_{s=1}^t \sigma_s \|\mathbf{w} - \mathbf{w}_s\|^2$ 的作用是通过 $\sigma_s$ **对不同时间步的 $\mathbf{w}_s$ 分配不同的惩罚权重**。

- $\sigma_1 = 1/\eta_1$ → 对 $\mathbf{w}_1$ 的偏离惩罚最重
- 后续 $\sigma_s$ 逐次递减 → 对较新的 $\mathbf{w}_s$ 约束更轻

这样累加后 $\sum\sigma_s = 1/\eta_t$ 刚好匹配当前学习率，使得 FTRL 在没有正则化时退化为 OGD。

关键词

$\sigma_s$ **不是超参数**——它完全由学习率序列 $\eta_t$ 衍生而来，是连接标准梯度下降与近端形式的桥梁。





#### Q：其中我们**根据学习率调度定义 $\sigma_s$，使得 $\sigma_{1:t} = 1/\eta_t$**。为什么？

因为只有这样定义，FTRL-Proximal 才能在 $\lambda_1 = 0$ 时退化为 OGD。下面推导给你看为什么这个定义是**必要且精妙的**。

---

从目标函数出发

FTRL-Proximal 的更新规则：

$$
\mathbf{w}_{t+1} = \arg\min_{\mathbf{w}} \left( \mathbf{g}_{1:t} \cdot \mathbf{w} + \frac{1}{2} \sum_{s=1}^t \sigma_s \|\mathbf{w} - \mathbf{w}_s\|^2 \right)
$$

这是一个关于 $\mathbf{w}$ 的**凸二次函数**。展开平方项：

$$
\mathbf{g}_{1:t} \cdot \mathbf{w} + \frac{1}{2} \sum_{s=1}^t \sigma_s (\mathbf{w}^T\mathbf{w} - 2\mathbf{w}^T\mathbf{w}_s + \|\mathbf{w}_s\|^2)
$$

合并 $\mathbf{w}^T\mathbf{w}$ 项：

$$
= \frac{1}{2} \underbrace{\left( \sum_{s=1}^t \sigma_s \right)}_{= \sigma_{1:t}} \|\mathbf{w}\|^2 + \left( \mathbf{g}_{1:t} - \sum_{s=1}^t \sigma_s \mathbf{w}_s \right) \cdot \mathbf{w} + \text{常数}
$$

**对 $\mathbf{w}$ 求梯度并设为零：**

$$
\sigma_{1:t} \, \mathbf{w} + \mathbf{g}_{1:t} - \sum_{s=1}^t \sigma_s \mathbf{w}_s = 0
$$

解得：

$$
\mathbf{w}_{t+1} = \frac{1}{\sigma_{1:t}} \left( \sum_{s=1}^t \sigma_s \mathbf{w}_s - \mathbf{g}_{1:t} \right)
$$

目标：让这个等于 OGD

OGD（设 $\mathbf{w}_1 = 0$）的多步累积形式是：

$$
\mathbf{w}_{t+1} = -\sum_{s=1}^t \eta_s \mathbf{g}_s
$$

要让 $\mathbf{w}_{t+1}$ 的 FTRL 解等于这个结果，你必须让 $\sigma_{1:t}$ 等于某个合理的值。**论文设计者的思路是：**

做一次"逆向工程"——

如果我们要求 FTRL 的解恰好是 OGD 的解的某种形式，需要让：

$$
\frac{1}{\sigma_{1:t}} = \eta_t \quad \Longrightarrow \quad \sigma_{1:t} = \frac{1}{\eta_t}
$$

为什是 $1/\eta_t$ 而不是别的？

仔细看 OGD 的 Step-by-Step：

$$
\mathbf{w}_2 = \mathbf{w}_1 - \eta_1 \mathbf{g}_1
$$
$$
\mathbf{w}_3 = \mathbf{w}_2 - \eta_2 \mathbf{g}_2 = \mathbf{w}_1 - \eta_1 \mathbf{g}_1 - \eta_2 \mathbf{g}_2
$$

而 FTRL 如果设 $\sigma_{1:t} = 1/\eta_t$，代入上式：

$$
\mathbf{w}_{t+1} = \eta_t \left( \sum_{s=1}^t \sigma_s \mathbf{w}_s - \mathbf{g}_{1:t} \right)
$$

这个形式看起来和 OGD 不同，但**通过递推可以证明等价**。最关键的是，**当 $\eta_t$ 为常数时**：

- $\sigma_1 = 1/\eta$，$\sigma_2 = \sigma_3 = \cdots = 0$
- 近端项只剩 $\frac{1}{2\eta}\|\mathbf{w} - \mathbf{w}_1\|^2$
- 求导得 $\mathbf{w}_{t+1} = \mathbf{w}_1 - \eta \sum_{s=1}^t \mathbf{g}_s$，**完全等于 OGD**

一句话回答

**因为 $\sigma_{1:t} = 1/\eta_t$ 是使 FTRL 在没有正则化时与 OGD 保持一致的唯一条件**——它让近端项的曲率（二次系数）正好等于学习率的倒数，从而让两个算法的解空间对齐。





#### Q：逐项拆解这个公式：

$$
\mathbf{w}_{t+1} = \arg\min_{\mathbf{w}} \left( \mathbf{g}_{1:t} \cdot \mathbf{w} + \frac{1}{2} \sum_{s=1}^t \sigma_s \|\mathbf{w} - \mathbf{w}_s\|^2 + \lambda_1 \|\mathbf{w}\|_1 \right)
$$

---

第一项：$\mathbf{g}_{1:t} \cdot \mathbf{w}$

- $\mathbf{g}_{1:t} = \sum_{s=1}^t \mathbf{g}_s$，即到第 $t$ 步为止所有梯度的**累积和**
- $\mathbf{w}$ 是我们要求解的变量
- 这一项本质上是 $\sum_{s=1}^t \mathbf{g}_s \cdot \mathbf{w}$，它**迫使 $\mathbf{w}$ 沿着历史梯度的反方向移动**（因为优化是求最小值）

与 OGD 的类比：OGD 的展开式 $\mathbf{w}_{t+1} = \mathbf{w}_1 - \eta \sum \mathbf{g}_s$ 中，$\sum \mathbf{g}_s$ 也出现在同样的位置，只是多了一个学习率缩放。

---

第二项：$\frac{1}{2} \sum_{s=1}^t \sigma_s \|\mathbf{w} - \mathbf{w}_s\|^2$

这是**近端正则化项**，分解来看：

- **$\|\mathbf{w} - \mathbf{w}_s\|^2$**：度量候选 $\mathbf{w}$ 与历史权重 $\mathbf{w}_s$ 之间的欧氏距离平方
- **$\sigma_s$**：每个历史步的权重，$\sigma_s = \frac{1}{\eta_s} - \frac{1}{\eta_{s-1}}$
  - 由于 $\sum\sigma_s = 1/\eta_t$，早期步的 $\sigma_s$ 较大（因为 $\eta_s$ 小，倒数大）
  - 这意味**对早期的 $\mathbf{w}_s$ 约束更紧**（早期模型不稳定，不希望 $\mathbf{w}$ 偏离太远）
- **求和 $\sum_{s=1}^t$**：把 $\mathbf{w}$ 同时拉向所有历史 $\mathbf{w}_s$，是一种**"集体近端"约束**

展开平方项：

$$
\frac{1}{2} \sum_{s=1}^t \sigma_s (\|\mathbf{w}\|^2 - 2\mathbf{w}\cdot\mathbf{w}_s + \|\mathbf{w}_s\|^2) = \frac{1}{2} \sigma_{1:t} \|\mathbf{w}\|^2 - \left( \sum_{s=1}^t \sigma_s \mathbf{w}_s \right) \cdot \mathbf{w} + \text{const}
$$

其中 $\sigma_{1:t} = 1/\eta_t$。**核心效果**：对 $\|\mathbf{w}\|^2$ 施加了 $1/2\eta_t$ 的 **L2 正则化强度**——系数大小受控，且随时间递减（**学习率退火**）。

---

第三项：$\lambda_1 \|\mathbf{w}\|_1$

L1 正则化，作用是**产生稀疏性**。





#### Q：详细解释这个公式是如何推导出来的？


$$
w_{t+1,i} =
\begin{cases}
0 & \text{if } |z_{t,i}| \leq \lambda_1 \\
-\eta_t (z_{t,i} - \operatorname{sgn}(z_{t,i}) \lambda_1) & \text{otherwise.}
\end{cases}
$$

这是从目标函数到闭式解的完整推导过程。

---

起点：每坐标的优化问题

从 FTRL-Proximal 的完整目标函数出发：

$$
\mathbf{w}_{t+1} = \arg\min_{\mathbf{w}} \left( \mathbf{g}_{1:t} \cdot \mathbf{w} + \frac{1}{2} \sum_{s=1}^t \sigma_s \|\mathbf{w} - \mathbf{w}_s\|^2 + \lambda_1 \|\mathbf{w}\|_1 \right)
$$

展开平方项并将与 $\mathbf{w}$ 有关的项提取出来：

$$
\mathbf{g}_{1:t} \cdot \mathbf{w} + \frac{1}{2} \sigma_{1:t} \|\mathbf{w}\|^2 - \left( \sum_{s=1}^t \sigma_s \mathbf{w}_s \right) \cdot \mathbf{w} + \lambda_1 \|\mathbf{w}\|_1 + \text{const}
$$

其中 $\sigma_{1:t} = \sum \sigma_s = 1/\eta_t$。合并前两项和第三项：

$$
\underbrace{\left( \mathbf{g}_{1:t} - \sum_{s=1}^t \sigma_s \mathbf{w}_s \right)}_{\mathbf{z}_t} \cdot \mathbf{w} + \frac{1}{2\eta_t} \|\mathbf{w}\|^2 + \lambda_1 \|\mathbf{w}\|_1 + \text{const}
$$

这个目标函数对 $\mathbf{w}$ 的**每个坐标是可分离的**（L1 和 L2 都没有跨坐标耦合），所以可以逐坐标求解：

设 $z = z_{t,i}$，$\eta = \eta_t$，$\lambda = \lambda_1$，我们要解：

$$
f(w) = z w + \frac{1}{2\eta} w^2 + \lambda |w|
$$

其中 $w \in \mathbb{R}$。

分段处理绝对值

因为 $|w|$ 在 $w=0$ 处不可导，需要分 $w>0$ 和 $w<0$ 两个区间讨论。

情况 1：$w > 0$

此时 $|w| = w$：

$$
f_+(w) = z w + \frac{1}{2\eta} w^2 + \lambda w = \frac{1}{2\eta} w^2 + (z + \lambda) w
$$

这是开口向上的二次函数（$\frac{1}{2\eta} > 0$），极值点通过求导为零得到：

$$
f_+'(w) = \frac{1}{\eta} w + (z + \lambda) = 0
$$

$$
\Rightarrow w = -\eta(z + \lambda)
$$

但我们必须检查这个解是否满足 $w > 0$ 的假设。因为 $\eta > 0$，$\lambda > 0$：

- 如果 $-\eta(z + \lambda) > 0 \Rightarrow z + \lambda < 0 \Rightarrow z < -\lambda$
- 此时 $w = -\eta(z + \lambda) > 0$，有效



情况 2：$w < 0$

此时 $|w| = -w$：

$$
f_-(w) = z w + \frac{1}{2\eta} w^2 - \lambda w = \frac{1}{2\eta} w^2 + (z - \lambda) w
$$

求导：

$$
f_-'(w) = \frac{1}{\eta} w + (z - \lambda) = 0
$$

$$
\Rightarrow w = -\eta(z - \lambda)
$$

检查 $w < 0$ 条件：

- 如果 $-\eta(z - \lambda) < 0 \Rightarrow z - \lambda > 0 \Rightarrow z > \lambda$
- 此时 $w = -\eta(z - \lambda) < 0$，有效



情况 3：$z \in [-\lambda, \lambda]$

此时无论往正方向还是负方向走，极值点都不落在定义域内，意味着目标函数在 $w=0$ 处取最小值。

验证：计算 $f(w)$ 在 $w=0$ 处的次梯度（subgradient）：

$$
\partial f(0) = z + \lambda \cdot \partial|0| = z + \lambda[-1, 1] = [z - \lambda, z + \lambda]
$$

当 $z \in [-\lambda, \lambda]$ 时，$0 \in \partial f(0)$，所以 $w=0$ 是最优解。

---



整合分段函数

将三种情况合并：

$$
w_{t+1,i} =
\begin{cases}
0 & \text{if } |z_{t,i}| \leq \lambda_1 \\
-\eta_t (z_{t,i} - \lambda_1) & \text{if } z_{t,i} > \lambda_1 \ (\Rightarrow w < 0) \\
-\eta_t (z_{t,i} + \lambda_1) & \text{if } z_{t,i} < -\lambda_1 \ (\Rightarrow w > 0)
\end{cases}
$$

用 $\operatorname{sgn}$ 统一表达 $z_{t,i} > \lambda_1$ 和 $z_{t,i} < -\lambda_1$ 两种情况：

$$
w_{t+1,i} =
\begin{cases}
0 & \text{if } |z_{t,i}| \leq \lambda_1 \\
-\eta_t (z_{t,i} - \operatorname{sgn}(z_{t,i}) \lambda_1) & \text{otherwise}
\end{cases}
$$

---

直观理解

这个公式在做一件事：**Soft Thresholding（软阈值）**。

把 $-\eta_t z_{t,i}$ 看作没有 L1 时的普通 OGD 解（回想 OGD 展开式 $\mathbf{w}_{t+1} = -\eta_t \sum \mathbf{g}_s$，与 $\mathbf{z}_t$ 只差一个 $\sum\sigma_s\mathbf{w}_s$ 的偏移）。L1 正则化的效果就是：

- 如果 $|z_{t,i}| \leq \lambda_1$ → 直接砍到 0
- 如果 $z_{t,i} > \lambda_1$ → 从 $-\eta_t z_{t,i}$ 中减去一个 $\eta_t \lambda_1$（向右拉近 0）
- 如果 $z_{t,i} < -\lambda_1$ → 从 $-\eta_t z_{t,i}$ 中加上一个 $\eta_t \lambda_1$（向左拉近 0）

**图示**：横轴是 $-\eta_t z_{t,i}$（无正则化解），纵轴是最终 $w_{t+1,i}$：

```
    w_out
     ↑
     │  斜率 1
     │
─────┼─────→  -ηz
     │     (软阈值: (-λ, λ) 区间被压缩到 0)
```

这就是为什么 FTRL-Proximal 能产生**精确的零系数**——L1 正则化将一定范围内的值直接截止到零，而不是像 L2 正则化那样**只是缩小但不归零**。





Q：为什么是 我们通过令 $\mathbf{z}_t = \mathbf{z}_{t-1} + \mathbf{g}_t + (1/\eta_t - 1/\eta_{t-1}) \mathbf{w}_t$ 来更新？

这个更新公式直接从 $\mathbf{z}_t$ 的定义推导出来。一步步来看：

$\mathbf{z}_t$ 的定义
$$
\mathbf{z}_{t-1} = \mathbf{g}_{1:t-1} - \sum_{s=1}^{t-1} \sigma_s \mathbf{w}_s
$$

$$
\mathbf{z}_t = \mathbf{g}_{1:t} - \sum_{s=1}^{t} \sigma_s \mathbf{w}_s
$$

展开 $\mathbf{z}_t$
$$
\begin{aligned}
\mathbf{z}_t &= (\mathbf{g}_{1:t-1} + \mathbf{g}_t) - \left( \sum_{s=1}^{t-1} \sigma_s \mathbf{w}_s + \sigma_t \mathbf{w}_t \right) \\
&= \underbrace{(\mathbf{g}_{1:t-1} - \sum_{s=1}^{t-1} \sigma_s \mathbf{w}_s)}_{\mathbf{z}_{t-1}} + \mathbf{g}_t - \sigma_t \mathbf{w}_t
\end{aligned}
$$

$\sigma_t$ 的定义 $\sigma_{1:t} = 1/\eta_t$，所以：
$$
\sigma_t = \sigma_{1:t} - \sigma_{1:t-1} = \frac{1}{\eta_t} - \frac{1}{\eta_{t-1}}
$$

代入
$$
\mathbf{z}_t = \mathbf{z}_{t-1} + \mathbf{g}_t - \left( \frac{1}{\eta_t} - \frac{1}{\eta_{t-1}} \right) \mathbf{w}_t
$$

但原文写的是 $+$ 而不是 $-$？

你这里发现了一个关键问题。论文原文的算法伪代码（Algorithm 1）中写的是 **减号**：

> $z_i \leftarrow z_i + g_i - \sigma_i w_i$

而正文推导中写的是 **加号**。这是原文的一个**符号约定不一致**——取决于 $\mathbf{z}_t$ 的定义是 $\mathbf{g}_{1:t} - \sum\sigma_s\mathbf{w}_s$ 还是 $\sum\sigma_s\mathbf{w}_s - \mathbf{g}_{1:t}$。两种定义只是差一个负号，不影响最终 $\mathbf{w}_{t+1}$ 的值，因为解 $\mathbf{w}$ 的公式也相应地调整了符号。

具体来说：

| 定义                               | $z_t$ 更新                           | $w_{t+1,i}$ 解                                               |
| ---------------------------------- | ------------------------------------ | ------------------------------------------------------------ |
| $z_t = g_{1:t} - \sum\sigma_s w_s$ | $z_t = z_{t-1} + g_t - \sigma_t w_t$ | $w_{t+1,i} = -\eta_t(z_{t,i} - \text{sgn}(z_{t,i})\lambda_1)$ |
| $z_t = \sum\sigma_s w_s - g_{1:t}$ | $z_t = z_{t-1} - g_t + \sigma_t w_t$ | $w_{t+1,i} = \eta_t(z_{t,i} - \text{sgn}(z_{t,i})\lambda_1)$ |

**翻译文本里的 $+$ 很可能是一个笔误**，正确的实现应该用 Algorithm 1 中的 **减号** 版本。





#### 4.1 概率特征包含
- 问题：一半的独特特征在整个训练集中只出现一次，跟踪它们浪费内存
- **布隆过滤器方法**：用计数布隆过滤器跟踪特征出现次数，超过阈值才加入模型
- 效果：**节省66% RAM，仅损失0.008%精度**——这是最佳权衡

---



Q：详细解释 4.1 概率特征包含

痛点：罕见特征浪费内存

在 CTR 预测中，特征空间极度稀疏。原文说：

> 一半的独特特征在包含数十亿样本的整个训练集中**只出现一次**

为标准梯度下降风格的算法维护这些特征的状态需要：

- 在哈希表中存储特征键（字符串或哈希值）
- 存储系数 $w_i$
- 存储梯度累积值（如 $\sum g_{t,i}^2$）
- 存储计数

对于只出现一次的特征，这些存储完全是浪费——**你根本没有足够的数据来学习它的系数**。

直觉目标

> 能不能在训练过程中**即时决定"这个特征值不值得跟踪"**，而不需要事先做离线预处理？

---



Q：为什么不做离线预处理？

典型的离线方案：先扫描一遍数据，统计每个特征出现次数。再扫描一遍数据，丢弃出现次数小于 $k$ 的特征，然后用剩下的特征训练。

但在这个系统中**不行**：

1. 数据是流式的，做两遍扫描需要额外的数据读取/写入，**非常昂贵**
2. 一旦丢弃了某个特征，就无法再用它来训练——你无法估计"如果我不丢弃它，精度会提高多少"



两种概率方法

方法 1：泊松包含（Poisson Inclusion）

**规则**：当一个新特征首次出现时，**以概率 $p$ 加入模型**。一旦加入，后续正常训练。

**效果**：一个特征在被加入前需要被观察到的次数服从**几何分布**，期望值为 $1/p$。

TODO：这里的「几何分布」如何理解？

例如 $p = 0.1$：

- 期望需要被看到 10 次才会被加入
- 但有些特征可能在第一次出现就被加入（运气好）
- 有些特征出现 30 次可能还没被加入（运气差）

**优点**：实现极其简单（只需要一轮随机数）
**缺点**：控制力弱——你无法精确控制"至少出现 N 次才加入"



方法 2：布隆过滤器包含（Bloom Filter Inclusion）

**规则**：使用一组**滚动的计数布隆过滤器**来检测每个特征在训练中出现的次数。当某个特征出现次数超过阈值 $n$ 后，将其加入模型。

**布隆过滤器要点**：

- 是一种概率性数据结构，**用很小的空间近似计数**
- 有**假阳性**（可能把出现次数 < $n$ 的特征误判为 ≥ $n$）
- **没有假阴性**（不会漏掉真正 ≥ $n$ 的特征）
- "滚动"（rolling）意味着用多个时间窗口的过滤器组合，适应数据流特性

**优点**：

- 内存极省（布隆过滤器本身很小）
- 控制精确（设 $n=2$ 就确保大部分特征在出现 ≥2 次后才加入）

---

实验结果对比

| 方法             | RAM 节省 | AucLoss 损害 |
| ---------------- | -------- | ------------ |
| Bloom (n=2)      | **66%**  | **0.008%**   |
| Bloom (n=1)      | 55%      | 0.003%       |
| Poisson (p=0.03) | 60%      | 0.020%       |
| Poisson (p=0.1)  | 40%      | 0.006%       |

**解读**：

- Bloom (n=2) 是最优选择：**砍掉 2/3 内存，精度损失几乎为零**
- 更大的 $n$ 能节省更多内存，但 $n=2$ 已经是 sweet spot（$n=1$ 节省更少，$n>2$ 可能开始损失精度）
- 泊松包含的效果也可接受，但在相同的 RAM 节省下精度损失更大

---

核心思想总结

概率特征包含的巧妙之处在于：

> 不是问"这个特征是否重要"，而是问"这个特征出现的频率够不够高，值得我去跟踪它"

它用概率方法**以极低的代价做了近似特征选择**，而且不需要离线预处理，完全在在线流中完成。布隆过滤器的版本更是用空间换时间的经典案例——**用少量的假阳性（可能多跟踪了几个本不该跟踪的特征）换取了巨大的内存节省**。







### 4.2 使用更少位编码值

- 64位浮点 -> **q2.13定点编码**（16位）
- 范围 $[-4, 4)$，精度 $2^{-13}$
- 用**随机舍入**解决累积舍入误差（确保离散化误差均值为零）
- 效果：**节省75%系数存储RAM，无精度损失**



### 4.3 训练多个相似模型

- 多个模型变体共享哈希表（键、计数），只存各自的系数值
- 实质上是**分摊元数据存储成本**



### 4.5 使用计数计算学习率

- 不需存储完整的梯度平方和 $\sum g_{t,i}^2$
- 只存正负事件计数 $P$ 和 $N$，用公式 $\frac{PN}{N+P}$ 近似
- 经验上效果一样好，存储更少





### 4.6 训练数据子采样

- CTR很低（通常<2%），正样本更珍贵
- **保留所有有点击的查询，无点击的查询按比例 $r$ 采样**
- 用重要性权重 $\omega_t$ 校正：有点击的权重=1，无点击的权重=$1/r$
- 目标函数期望不变：这个设计很优雅

---



## 5. 评估模型性能

### 5.1 渐进验证（Progressive Validation）
- 不保留验证集，而是**用每次预测时的损失作为在线评估**
- 优点：**100%数据既用于训练又用于测试**，统计效率更高
- 只看**相对变化**而非绝对值，因为不同国家的CTR基准不同



### 5.2 GridViz 可视化工具

- 大规模学习中聚合指标可能掩盖子群体差异
- GridViz：按国家、主题等多种维度切片，彩色单元格显示模型对比
- 能快速发现"整体指标提升但某些子集下降"的问题

---



## 6. 置信度估计

### 问题
- 不仅需要预测CTR，还需要知道预测的**可信度**
- 传统统计方法（求逆 $n \times n$ 矩阵）在数十亿规模上不可行

### 不确定性分数
- **核心洞察**：每特征计数器 $n_{t,i}$ 本身就编码了不确定性信息
  - $n_i$ 大的特征 → 学习率小 → 系数可靠
  - $n_i$ 小的特征 → 学习率大 → 系数仍不确定
- 不确定性分数 $u(\mathbf{x}) = \alpha \boldsymbol{\eta} \cdot \mathbf{x}$
- **计算成本几乎为零**：一次稀疏点积，和预测本身一样快

---



## 7. 校准预测

### 问题
- 预测CTR和实际CTR可能存在系统性偏差
- 偏差来源：不准确的建模假设、学习算法缺陷、隐藏特征

### 解决方法
- 学习等张（单调递增）校正函数 $\tau(p)$
- 用**等张回归**（isotonic regression）找到分段线性校正
- 显著降低了预测在高低两端的偏差

---



## 8. 自动化特征管理

- 大规模系统中多个团队并行开发特征
- 部署**元数据索引**：管理数百个模型对数千个输入信号的使用
- 自动检测：废弃信号、平台兼容性、特征替代跟踪
- 新信号自动测试和加入白名单

---



## 9. 不成功的实验—— 同样重要

### 9.1 激进特征哈希
- 文献报道用哈希到 $2^{24}$ 特征空间效果很好
- 但Google无法在不损失精度的情况下降到数十亿以下
- 结论：**他们的场景可能需要更大特征空间，且可解释性很重要**

### 9.2 Dropout
- 在CV领域效果很好，但在CTR预测中**无益处甚至有害**
- 原因分析：CV特征是密集的，dropout帮助解耦强相关特征
- CTR特征是**稀疏+标签有噪声**，dropout只是减少了可用数据量

### 9.3 特征Bagging
- 多个模型在不同特征子集上训练后平均
- 实际上**略微降低了预测质量**（AucLoss增加0.1%-0.6%）

### 9.4 特征向量归一化
- 不同样本的非零特征数量差异大
- 尝试多种归一化方式，**无法转化为正向指标**
- 可能与每坐标学习率和正则化的交互有关

---



## 总结与启示

| 主题               | 关键启示                                             |
| ------------------ | ---------------------------------------------------- |
| **算法**           | FTRL-Proximal 是精度和稀疏性的最佳平衡               |
| **学习率**         | 每坐标学习率非常有效（AucLoss↓11.2%）                |
| **内存**           | 布隆过滤器 + 定点编码 是实用技巧                     |
| **子采样**         | 用重要性权重校正，理论保证期望不变                   |
| **置信度**         | 利用学习算法本身的计数器做不确定性估计               |
| **成功 vs 不成功** | Dropout/hashing/bagging 并非通用方案，取决于数据特征 |

**这篇论文最大的贡献是把"实际工程经验"系统化了**——它告诉我们：在工业环境中，内存、速度、可解释性和部署约束与模型精度同等重要。理论方法需要在真实系统中验证，而且结果常常出乎意料（如dropout无效）。









Q：FTRL中对线性模型的参数做稀疏化的目的是？

**FTRL 中对线性模型参数做稀疏化的主要目的**：

1. 减少模型体积，便于线上部署

线性模型参数量 = 特征维度。当特征维度达到**百亿级**时（如广告 CTR 预测），稀疏化使大部分参数为 0，模型体积大幅缩小，节省内存和存储。

2. 加速线上推理

稀疏参数意味着**大量特征可以直接跳过**，只需计算非零参数对应的特征，推理速度更快。

3. 自动特征选择

稀疏化自动**筛掉无效特征**，保留真正有预测能力的特征，等效于特征选择（Feature Selection），提升模型泛化能力。

4. FTRL-Proximal 的核心设计

FTRL-Proximal（Google 2011年论文）的核心就是通过 **L1 正则化**实现稀疏：

$$
w_{t+1} = \arg\min_w \left( \sum_{s=1}^t g_s \cdot w + \frac{1}{2} \sum_{s=1}^t \sigma_s \|w - w_s\|^2 + \lambda_1 \|w\|_1 \right)
$$

其中 $\lambda_1 \|w\|_1$ 项直接驱动参数趋于 0。

实际意义

| 场景     | 无稀疏化           | 有稀疏化             |
| -------- | ------------------ | -------------------- |
| 模型大小 | 数百 GB            | 数 GB 甚至更小       |
| 线上推理 | 慢（全量特征计算） | 快（仅计算非零参数） |
| 特征覆盖 | 包含噪声特征       | 保留关键特征         |

**一句话总结**：稀疏化 = **用更少的参数表达同样好的模型**，核心是为了线上工业部署的效率和可维护性。

