# SLIM: Sparse Linear Methods for Top-N Recommender Systems





重要意义：

业务场景：无

业务指标：无

离线指标：HitRate@10，ARHR（平均倒数排名）

数据集：包括MovieLens-10M在内的8个公开数据集

模型类型：评分矩阵预测，MSE回归模型或者01点击率预测

创新点：

- 详见翻译

个人收获：

- 文中对于稀疏性的可视化让人眼前一亮
- 长尾分布的图很漂亮和直观
- 坐标下降求解（闭式解 + L1软阈值），不是典型的batch_size + SGD的训练方式
- 所谓弹性网络，就是同时包括L1正则和L2正则的回归损失

疑问：

- TODO：弹性网络的坐标下降算法中用到的闭式解，推导还没看懂。本质是传统机器学习针对特定方法构造的损失函数，一般不能直接通过SGD类方法求解（因为非凸或者非光滑），往往需要自己推导参数更新算法。
- TODO：对应的SLIM实现代码没有看懂





## FQA

Q：弹性网络问题（Elastic Net Problem）

一、定义

**弹性网络**是带混合L1+L2正则的线性优化问题，由Zou & Hastie提出，用来弥补**Lasso（L1）**和**岭回归Ridge（L2）**各自的缺陷。
本质：最小化拟合损失 + L1、L2混合惩罚项。

标准目标函数（线性回归形式）
$$
\mathcal{L}(\boldsymbol{w})=\frac{1}{2}\|\boldsymbol{y}-\boldsymbol{X}\boldsymbol{w}\|_2^2 + \lambda\left[\rho\|\boldsymbol{w}\|_1+\frac{1-\rho}{2}\|\boldsymbol{w}\|_2^2\right]
$$
参数说明：
1. $\lambda\ge0$：正则整体强度；$\lambda$越大，惩罚越强
2. $\rho \in [0,1]$（sklearn中叫`l1_ratio`）：L1/L2混合比例
   - $\rho=1$ → **Lasso**（只有L1）
   - $\rho=0$ → **Ridge岭回归**（只有L2）
   - $0<\rho<1$ → 标准弹性网络

> 逻辑直观理解：
> - L1范数 $\|\boldsymbol{w}\|_1$：产生稀疏解，把部分系数压缩到0，实现**自动特征选择**
> - L2范数 $\|\boldsymbol{w}\|_2^2$：稳定系数，解决**多重共线性**，防止系数震荡



二、为什么需要弹性网络？Lasso存在两大痛点

1. **高度相关特征场景**
一组强相关特征（比如同一类统计指标），Lasso会**随机挑选其中一个，其余置零**，结果不稳定；
弹性网络拥有**分组效应**：倾向同时保留整组相关特征，估计更稳定。

2. **高维场景 $p\gg n$（特征数＞样本数）**
Lasso理论上最多只能选出 $n$ 个非零特征；弹性网络不受该限制。

三、核心优缺点

✅ **优点**
1. 同时具备**稀疏特征选择**（L1）+ **抵抗多重共线性**（L2）
2. 系数估计相比Lasso更加稳定
3. 适合高维、存在大量相关特征数据集（推荐CTR、基因数据、金融多因子）

❌ **缺点**
1. 需要调两个超参数 $(\lambda,\rho)$，调参成本高于Lasso/Ridge
2. 依然是线性模型，无法捕捉非线性交互

四、求解方式

没有**解析闭式解**，工业界主流：
- **坐标下降法 Coordinate Descent**（glmnet、sklearn默认）
- 在线学习拓展：**FTRL-Proximal**（广告CTR大规模稀疏LR，本质就是在线弹性网络）

五、三者对比速览

| 方法 | 正则 | 能否特征选择 | 多重共线性处理 |
|------|------|--------------|----------------|
| Ridge | L2 | ❌ 系数不会归零 | ✅ 稳定 |
| Lasso | L1 | ✅ 稀疏 | ❌ 相关特征随机取舍 |
| ElasticNet | L1+L2 | ✅ 稀疏 | ✅ 分组保留、稳定 |

六、工程常用场景（你做推荐系统CTR一定会碰到）

大规模稀疏逻辑回归（LR）常用**弹性网络正则**：
- L1带来稀疏：大量特征权重归零，线上推理模型体积更小、延迟更低
- L2防止特征共线性、权重剧烈波动；
业界FTRL算法就是在线版本的弹性网络优化。

七、补充：弹性网络逻辑回归

分类任务（CTR预估）把MSE损失替换成交叉熵，惩罚项不变，依然称为弹性网络问题：
$$
\mathcal{L} = \text{LogLoss}(\boldsymbol{y},\hat{\boldsymbol{y}}) + \lambda\left[\rho\|\boldsymbol{w}\|_1+\frac{1-\rho}{2}\|\boldsymbol{w}\|_2^2\right]
$$

如果你需要，我可以给你：
1）坐标下降法完整迭代更新公式；
2）sklearn ElasticNet代码示例；
3）几何约束空间示意图（L1菱形、L2圆形、弹性网圆角菱形）。





Q：闭式解

问题定义

给定 item $j$，其权重向量 $\mathbf{w}_j \in \mathbb{R}^n$（$w_{jj}=0$）：

$$
\min_{\mathbf{w}_j} \;\; \frac{1}{2} \|\mathbf{a}_j - A\mathbf{w}_j\|_2^2 + \frac{\beta}{2} \|\mathbf{w}_j\|_2^2 + \lambda \|\mathbf{w}_j\|_1
\quad \text{s.t.} \;\; w_{ij} \ge 0,\; w_{jj} = 0
$$

- $\mathbf{a}_j \in \mathbb{R}^m$：$A$ 的第 $j$ 列
- $A \in \mathbb{R}^{m \times n}$：交互矩阵
- $\beta$：L2 系数，$\lambda$：L1 系数

---

坐标下降：只更新 $w_{kj}$

固定其他坐标，只把 $w_{kj}$ 看作变量。排除 $k$ 后的线性组合：

$$
\mathbf{r}_{-k} = \mathbf{a}_j - \underbrace{(A\mathbf{w}_j - A_k w_{kj})}_{\text{rest}} = \mathbf{a}_j - A\mathbf{w}_j + A_k w_{kj}
$$

其中 $A_k \in \mathbb{R}^m$ 是 $A$ 的第 $k$ 列，$\text{rest} = \sum_{i \neq k} A_i w_{ij}$ 与 $w_{kj}$ 无关。

预测误差写为：

$$
\mathbf{a}_j - A\mathbf{w}_j = \mathbf{a}_j - (A_k w_{kj} + \text{rest}) = \mathbf{r}_{-k} - A_k w_{kj}
$$

---

代入损失函数（忽略与 $w_{kj}$ 无关的常数项）

先忽略 L1 项（软阈值单独处理），只看平滑部分：

$$
f(w_{kj}) = \frac{1}{2} \|\mathbf{r}_{-k} - A_k w_{kj}\|_2^2 + \frac{\beta}{2} w_{kj}^2 + \text{const}
$$

展开平方项：

$$
\begin{aligned}
f(w_{kj}) &= \frac{1}{2} \left( \mathbf{r}_{-k}^\top \mathbf{r}_{-k} - 2 w_{kj} A_k^\top \mathbf{r}_{-k} + w_{kj}^2 A_k^\top A_k \right) + \frac{\beta}{2} w_{kj}^2 + \text{const} \\[4pt]
&= \frac{1}{2} A_k^\top A_k \cdot w_{kj}^2 - A_k^\top \mathbf{r}_{-k} \cdot w_{kj} + \frac{\beta}{2} w_{kj}^2 + \text{const}
\end{aligned}
$$

---

求导为零
$$
\frac{\partial f}{\partial w_{kj}} = -A_k^\top \mathbf{r}_{-k} + w_{kj} \cdot A_k^\top A_k + \beta w_{kj} = 0
$$

$$
w_{kj} \cdot (A_k^\top A_k + \beta) = A_k^\top \mathbf{r}_{-k}
$$

---

无约束闭式解
$$
w_{kj}^* = \frac{A_k^\top \mathbf{r}_{-k}}{A_k^\top A_k + \beta}
= \frac{\langle A_k,\; \mathbf{r}_{-k} \rangle}{\|A_k\|_2^2 + \beta}
$$

---

加入 L1 + 非负约束

L1 惩罚产生软阈值（proximal operator for $\ell_1$）：

$$
\tilde{w}_{kj} = \text{sign}(w_{kj}^*) \cdot \max\left( |w_{kj}^*| - \gamma,\; 0 \right)
\quad \text{where} \quad
\gamma = \frac{\lambda}{\|A_k\|_2^2 + \beta}
$$

非负约束：

$$
w_{kj}^{\text{new}} = \max(\tilde{w}_{kj},\; 0)
$$

---

对应代码

| 数学                                                         | 代码                                    |
| ------------------------------------------------------------ | --------------------------------------- |
| $\mathbf{r}_{-k} = \mathbf{a}_j - A\mathbf{w}_j + A_k w_{kj}$ | `residual = a_j - pred + A_k * w_kj`    |
| $A_k^\top \mathbf{r}_{-k}$                                   | `numerator = np.dot(A_k, residual)`     |
| $\|A_k\|_2^2 + \beta$                                        | `denominator = col_norm_sq[k] + beta`   |
| $\gamma = \lambda / (\|A_k\|_2^2 + \beta)$                   | `gamma = lambda_ / denominator`         |
| 软阈值                                                       | `w_new = _soft_threshold(w_new, gamma)` |
| 非负                                                         | `if w_new < 0: w_new = 0.0`             |




Q：这个公式怎么来的？

$\mathbf{r}_{-k} = \mathbf{a}_j - A\mathbf{w}_j + A_k w_{kj}$

从定义出发：

当前用所有特征预测的误差是 $\mathbf{a}_j - A\mathbf{w}_j$。

但更新 $w_{kj}$ 时需要**排除 $k$ 后再算误差**（避免当前 $w_{kj}$ 的自影响）。当前预测拆开：

$$
A\mathbf{w}_j = \underbrace{A_k w_{kj}}_{\text{特征 }k} \;+\; \underbrace{\sum_{i \neq k} A_i w_{ij}}_{\text{rest}}
$$

所以排除 $k$ 后的预测是 $\text{rest} = A\mathbf{w}_j - A_k w_{kj}$，对应的误差为：

$$
\begin{aligned}
\mathbf{r}_{-k} &= \mathbf{a}_j - \text{rest} \\
&= \mathbf{a}_j - (A\mathbf{w}_j - A_k w_{kj}) \\
&= \mathbf{a}_j - A\mathbf{w}_j + A_k w_{kj}
\end{aligned}
$$

**通俗讲**：先算出当前整体误差 $\mathbf{a}_j - A\mathbf{w}_j$，再把特征 $k$ 的贡献加回来（相当于抹掉 $k$ 的影响），得到"没有 $k$ 时的误差"。用这个残差去更新 $w_{kj}$，才能得到正确的闭式解。



Q：

这是 **软阈值（soft-thresholding）** 算子，是 L1 正则项的**近端算子（proximal operator）**。

---

起源

L1 正则损失是两部分的加和：

$$
f(w) = \underbrace{\frac{1}{2}(w - w^*)^2}_{\text{来自 } \ell_2} \;+\; \underbrace{\gamma |w|}_{\text{来自 } \lambda\|w\|_1}
$$

L1 项在 $w=0$ 处不可导，没法直接求导为零。所以用**近端梯度**：在 $w^*$ 处做一步近端映射。

---

公式拆解
$$
\tilde{w} = \text{sign}(w^*) \cdot \max\left( |w^*| - \gamma,\; 0 \right)
$$

用表格理解 $w^*$ 在三个区间的行为：

| $w^*$ 范围           | $\|w^*\| - \gamma$ | $\max(\dots, 0)$   | $\text{sign}(w^*)$ | 结果 $\tilde{w}$ |
| -------------------- | ------------------ | ------------------ | ------------------ | ---------------- |
| $w^* > \gamma$       | $\> 0$             | $\|w^*\| - \gamma$ | $+1$               | $w^* - \gamma$   |
| $\|w^*\| \le \gamma$ | $\le 0$            | $0$                | —                  | $0$              |
| $w^* < -\gamma$      | $\> 0$             | $\|w^*\| - \gamma$ | $-1$               | $w^* + \gamma$   |

通俗就是**砍掉 $|w^*|$ 中 $\gamma$ 那么长的尾巴，剩下的往零缩**。

---

为什么 $\gamma = \lambda / (\|A_k\|^2 + \beta)$？

因为 L1 项 $\lambda\|w\|_1$ 在损失函数里，求近端算子时**除以了二次项的系数** $\frac{1}{2}(\|A_k\|^2 + \beta)$，所以阈值也要按比例缩放：

$$
\gamma = \frac{\lambda}{\|A_k\|^2 + \beta}
$$

如果二次项的曲率（分母）大，说明数据对该参数的信息量多，L1 的影响相对小，阈值就小；反之阈值大，更容易把 $w$ 压到零。



#### Q：坐标下降和软阈值

论文中提到了坐标下降和软阈值，但**没有给出我推导的那些完整公式细节**。

论文中的相关内容：

1. **Section IV-B-2, line 228-251** — 说明 W 可按列独立求解为式 (4)：

   > "The optimization problem of Equation 4 can be solved using coordinate descent and soft thresholding [14]."

2. **Section VIII-A, line 1786-1788** — 提到坐标下降更新公式但略过推导：

   > "The above can be shown theoretically by the coordinate descent updates (proof omitted here)."

3. 坐标准则（式 4 的求解）的完整推导来自其引用的 **[14] Friedman et al., "Regularization Paths for Generalized Linear Models via Coordinate Descent"**，不是 SLIM 论文原创的。

**结论**：我之前给的公式推导（$\mathbf{r}_{-k}$ 残差、闭式解 $w_{kj}^* = A_k^\top \mathbf{r}_{-k} / (\|A_k\|^2 + \beta)$、软阈值等）SLIM 论文里**没有具体写**。论文只是说了"用坐标下降 + 软阈值解"，完整推导在引用文献 [14]（Friedman 的 glmnet 论文）里。



#### Q：Coordinate Descent是什么？

**Coordinate Descent（坐标下降）** 是一种优化方法：每次只更新一个变量（坐标），固定其他所有变量，反复迭代直到收敛。

类比：调房间的旋钮

想象一个控制面板上有 n 个旋钮（= 参数 `w₁, w₂, ..., wₙ`），目标是最小化某个误差读数。

**梯度下降**：同时微调所有旋钮，每个旋钮转多少看梯度方向。

**坐标下降**：一次只拧**一个**旋钮，找到这个旋钮的最优位置，固定它，再拧下一个，循环往复。

SLIM 中的应用

要优化的问题：对每个 item j，有 n 个参数 `w₀ⱼ, w₁ⱼ, ..., wₙ₋₁,ⱼ`，加上约束 `wⱼⱼ = 0` 和 `w ≥ 0`。

坐标下降把所有参数排好队，**一次只更新一个 `w_kj`**：

```
固定 w₀ⱼ, w₂ⱼ, ..., wₙ₋₁,ⱼ   → 求 w₁ⱼ 的最优值（闭式解）
固定 w₀ⱼ, w₁ⱼ, w₃ⱼ, ...,     → 求 w₂ⱼ 的最优值
...
```

因为固定其他变量后，剩下一个变量的优化问题就是**一元二次函数**，可以直接求导得闭式解，不需要梯度下降式的小步迭代：

```
∂/∂w_kj  ½‖a_j − A·w_j‖²  →  w_kj* = A_k·r_{-k} / (‖A_k‖² + β)
```

为什么用坐标下降而不是梯度下降？

- **闭式解**：每次更新一步到位，不需要学习率
- **稀疏性**：soft-thresholding 直接把小权重砍成 0，天然产生稀疏解
- **适合 L1**：L1 不可导，坐标下降用软阈值处理，梯度下降需要近端算子