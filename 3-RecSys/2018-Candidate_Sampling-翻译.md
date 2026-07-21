# 候选采样（Candidate Sampling）学习笔记

---

本文分享了候选采样（Candidate Sampling）的核心概念与方法。核心内容：
- 从动机出发，说明正负样本数量悬殊时采样方法的必要性
- 介绍几率（Odds）定义及其与逻辑回归的关系
- 推导采样逻辑回归（Sampled Logistic）与采样Softmax（Sampled Softmax）
- 讲解重要性采样（Importance Sampling）与噪声对比估计（NCE）的原理
- 给出候选采样的形式化定义与算法框架

关键发现：
- 采样逻辑回归通过对负样本进行采样分布 Q(y|x) 下的采样，修正权重以逼近真实分布 P(y|x)
- 采样Softmax通过重要性权重修正梯度中的期望估计，避免遍历全量类别
- NCE通过将密度估计转化为二分类判别问题，绕过分区函数的计算
- 候选采样将目标类别与随机采样类别合并为候选集，大幅降低计算开销

# 候选采样

董冰峰

http://ml.dongbingfeng.cn

## 动机

- 在有限的计算资源下，正样本与负样本之间存在巨大的数量悬殊。
- CTR/CVR 预测模型。
- 疾病判别模型。

- 具有大量分类类别的 Softmax 模型。

- NLP 中的词语预测模型。
- DNN 信息检索模型。

## 几率

- 定义

- 几率与逻辑回归

$$odds(p\ vs\ q) = \frac{p}{q}$$

$$= log\_odds(p)$$

$$p = \frac{1}{1 + \exp(\boldsymbol{w}^T\boldsymbol{x})}$$

$$log\_odds(p) = logit(p) = \boldsymbol{w}^T\boldsymbol{x}$$

## 采样逻辑回归

- 我们以分布 $Q(y|x)$ 采样得到负样本，并希望获得 $P(y|x)$

$$\boldsymbol{w}^T\boldsymbol{x} = log\_odds(y\ came\ from\ Positive\ vs\ Negative\ |\ \boldsymbol{x})$$

$$= \log\frac{P(y|\boldsymbol{x})}{Q(y|\boldsymbol{x})(1 - P(y|\boldsymbol{x}))}$$

$$= \log\frac{P(y|\boldsymbol{x})}{1 - P(y|\boldsymbol{x})} - \log Q(y|\boldsymbol{x})$$

当采样比例为 $1/r$ 时，$Q(y|x) = 1/r$

$$F(\boldsymbol{x}, y) = F'(\boldsymbol{x}, y) + \log Q(y|\boldsymbol{x}) = \boldsymbol{w}^T\boldsymbol{x} - \log(r)$$

## 采样Softmax

- Softmax 训练过程

$$p(y|\mathbf{x}) = softmax(\mathbf{x}) = \frac{1}{Z}\exp(\mathbf{w}_y^T\mathbf{x}), \quad 其中\ Z = \sum_y \exp(\mathbf{w}_y^T\mathbf{x})$$

$$\nabla\log p(y|\mathbf{x}) = \nabla\mathcal{E}(y) - \sum_{y_k} p(y_k|\mathbf{x})\nabla\mathcal{E}(y_k), \quad 其中\ \mathcal{E}(y_k) = \mathbf{w}_y^T\mathbf{x}$$

$$\sum_{y_i} p(y_i|\mathbf{x})\nabla\mathcal{E}(y_i) = \mathbb{E}_{y_i \sim P}[\nabla\mathcal{E}(y_i)] \simeq \sum_{y_k} \hat{w}(y_k)\nabla\mathcal{E}(y_k)$$

$$其中\ \hat{w}(y_k) = \frac{\tilde{w}(y_k)}{\sum_{j=1}^m \tilde{w}(y_j)}, \quad \tilde{w}(y_k) = \hat{p}(y_k) / \hat{q}(y_k) = \mathbf{w}_{y_k}^T\mathbf{x} / \hat{q}(y_k)$$

## 重要性采样

- 我们需要得到

$$\mathbb{E}_{x \sim p}[f(x)] = \int f(x)p(x)dx$$

$$= \int f(x)\frac{p(x)}{q(x)}q(x)dx = \int f(x)w(x)q(x)dx$$

其中 $w(x) = \frac{p(x)}{q(x)}$ 为重要性权重。

## 噪声对比估计（NCE）

- 对于一个通过未归一化的概率密度函数 $p_m^0(\cdot; \alpha)$ 指定的统计模型，我们将归一化常数作为模型的一个额外参数 $c$。
即

$$\ln p_m(\cdot; \theta) = \ln p_m^0(\cdot; \alpha) + c$$

$X = \{\mathbf{x}_1, \ldots, \mathbf{x}_T\}$ 是观测数据集；$Y = \{\mathbf{y}_1, \ldots, \mathbf{y}_T\}$ 是人工生成的服从分布 $p_n(\cdot)$ 的噪声数据集。

$$\hat{\theta}_T = \arg\max_\theta J_T(\theta)$$

其中 $J_T(\theta)$ 为对比目标函数。

## NCE 理论

- 定义正样本的概率密度：$p_d(x_i) = p_{di}$；负样本的概率密度：$p_n(x_i) = p_{ni}$；带有参数 $\theta$ 的模型概率密度：$p_m(x_i) = p_{di}$；$N$ 为总样本数。

- 观测概率

$$e^{l(\theta)} = \prod_i \left(\frac{p_{mi}}{p_{mi} + p_{ni}}\right)^{N \cdot p_{di}} \left(\frac{p_{ni}}{p_{mi} + p_{ni}}\right)^{N \cdot p_{ni}}$$

$$\arg\max_\theta l(\theta) = \arg\max_\theta \sum_i \left[ p_{di} \ln\left(\frac{p_{mi}}{p_{mi} + p_{ni}}\right) + p_{ni} \ln\left(\frac{p_{ni}}{p_{mi} + p_{ni}}\right) \right]$$

- 对 $l(\theta)$ 关于 $p_{mi}$ 求导，当 $l(\theta)$ 最大化时可得 $p_{di} = p_{mi}$。
- 注意不存在 $\sum p_{mi} = 1$ 的约束，因此 $p_{mi}$ 默认是归一化的。

## NCE 理论

- 定理：如果满足条件 (a) 到 (c)，则 $\hat{\theta}_T$ 依概率收敛到 $\theta^*$，即 $\hat{\theta}_T \xrightarrow{P} \theta^*$。

(a) 当 $p_d$ 非零时 $p_n(\cdot)$ 非零。
(b) $\sup_\theta |J_T(\theta) - J(\theta)| \xrightarrow{P} 0$。
(c) 矩阵具有满秩，其中矩阵为 Fisher 信息矩阵。

- 对比噪声分布的选择：例如高斯分布或均匀分布、高斯混合分布或 ICA 分布。

## 形式化定义

- 我们面对一个多分类问题，每个训练样本 $(x_i, T_i)$ 由一个上下文 $x_i$ 和一个较小的目标类别集合 $T_i$ 组成，这些目标类别来自一个较大的候选类别全集 $L$。

- 我们希望学习一个兼容性函数 $F(x, y)$，该函数描述类别 $y$ 与上下文 $x$ 之间的兼容程度。

- 候选采样训练方法涉及为每个训练样本 $(x_i, T_i)$ 构建一个训练任务，在该任务中我们只需要对一小部分候选类别 $C_i \subset L$ 计算 $F(x, y)$，其中 $L = \{T_i\}$。通常，候选集 $C_i$ 是目标类别与随机采样的其他类别集合 $S_i \subset L$ 的并集：

$$C_i = T_i \cup S_i$$

$S_i$ 的随机选择可能依赖于也可能不依赖于 $x_i$ 和/或 $T_i$。

## 候选采样算法

- **全Softmax**：$C_i = L$，计算所有类别的完整Softmax。
- **负采样**：$C_i = T_i \cup S_i$，其中 $S_i$ 为从噪声分布中采样的负样本。
- **采样Softmax**：$C_i = T_i \cup S_i$，使用重要性权重修正采样偏差。
- **NCE**：$C_i = T_i \cup S_i$，将问题转化为二分类判别任务，区分正样本与噪声。
- **重要性采样**：使用提议分布 $Q(y|x)$ 计算权重，修正来自非目标分布的期望估计。

## 候选采样算法

- 候选采样方法的通用训练流程：

1. 给定训练样本 $(x_i, T_i)$，构造候选集 $C_i = T_i \cup S_i$。
2. 仅对 $y \in C_i$ 计算兼容性函数 $F(x_i, y)$。
3. 根据不同的候选采样算法（NCE、负采样、采样Softmax等）定义相应的损失函数。
4. 通过反向传播更新模型参数 $\theta$。

- 各类方法的差异在于损失函数的定义方式以及对采样分布 $Q(y|x)$ 的处理方式：

  - **负采样**：直接使用逻辑回归损失区分正样本与采样负样本。
  - **NCE**：将期望的模型分布 $p_m$ 与噪声分布 $p_n$ 进行对比。
  - **采样Softmax**：在Softmax中对候选集做归一化，并通过重要性权重修正梯度。
    - **重要性采样**：通过权重调整修正期望估计中的分布偏差。

## 参考文献



