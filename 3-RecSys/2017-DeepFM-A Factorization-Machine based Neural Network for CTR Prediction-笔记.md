# DeepFM- A Factorization-Machine based Neural Network for CTR Prediction

> 2026.06.25



业务场景：

任务：CTR点击率预估

数据集：开源+内部闭源

评估指标：





#### 摘要

挖掘用户行为背后复杂的特征交互关系，是提升推荐系统点击率（CTR）预估效果的关键。现有相关算法虽取得不少进展，但普遍存在偏向低阶交互或高阶交互的短板，或是依赖人工专业特征工程。

本文提出一种端到端训练模型，能够同时兼顾低阶与高阶特征交互的学习。所设计的 DeepFM 模型采用全新网络结构，融合了因子分解机（FM）的**推荐建模能力与深度学习的特征学习能力**。

对比谷歌提出的 Wide & Deep 模型，DeepFM 的线性分支（wide）与深度分支（deep）共享输入特征，**仅使用原始特征即可训练，无需额外人工构造交叉特征。**

本文在公开基准数据集与商业业务数据上开展大量对比实验，验证了 DeepFM 在点击率预估任务上，相比现有模型具备更优的效果与训练效率。





![image-20260629161129976](/Users/dazhang/Library/Application Support/typora-user-images/image-20260629161129976.png)

核心思想：从原始特征中直接学习低阶和高阶的特征交叉。





#### 1 引言

点击率（CTR）预估是推荐系统中的核心任务，该任务旨在预测用户点击推荐物品的概率。多数推荐系统**以最大化点击量为目标**，会依据预估点击率对候选物品排序；而在线广告等场景还**需要兼顾收益**，排序依据可调整为全体候选的「点击率 × 出价」，其中出价代表用户点击后平台获得的收益。无论哪种场景，精准预估点击率都是重中之重。

挖掘用户点击行为中隐含的特征交叉关系，对点击率预估至关重要。我们基于**主流应用商店的数据开展研究**，发现用户常在饭点下载外卖类 App，这说明**应用分类与时间戳**构成的二阶特征交叉，可作为点击率预测的有效信号。

另一项观察显示，青少年男性群体偏爱射击、角色扮演类游戏，这代表**应用分类、用户性别、年龄**组成的三阶特征交叉同样具备预测价值。

整体而言，用户点击行为背后的特征交叉模式十分复杂，低阶、高阶特征交叉均发挥关键作用。谷歌提出的 Wide & Deep 模型（2016）也证实：同时建模低阶与高阶特征交叉，效果优于仅单独学习其中一类。







## 实践

#### DeepFM 模型结构

#### 总览

DeepFM = FM（因子分解机） + Deep（DNN），两者共享同一组 feature embeddings。

最终输出为 sigmoid 后的 CTR 预估值。

```bash
输入: feature_bags (每个 field 的 sparse indices)
    │
    ├── Linear 部分 ──→ Linear_emb(vocab→1) ──→ sum ──→ bias ──→ 线性logit
    │
    ├── FM 二阶部分 ──→ Feature_emb(vocab→k) ──→ 成对交互 ──→ FM logit
    │
    └── Deep 部分 ──→ Feature_emb(vocab→k) ──→ concat ──→ MLP ──→ Deep logit
    │
┌────── logits ──────┐
│  sum + FM + Deep   │
└───────┬────────────┘
   sigmoid → (0,1)
```



1. Linear 部分（一阶特征）

```bash
logits = self.bias.expand(batch_size)           # 全局偏置
for fn in self.field_names:
    linear_emb = self.linear_embeddings[fn](...)  # EmbeddingBag(vocab→1)
    logits = logits + linear_emb.squeeze(-1)      # 每个 field 的线性贡献累加
```

- 每个 field 对应一个 EmbeddingBag(vocab_size, 1) — 将稀疏特征映射为标量权重
- 等价于 LR 中的 w_i * x_i，但用 embedding lookup 实现
- logits shape: [batch_size]



2. FM 二阶部分（特征交互）

```bash
# 所有 field 的 embedding (k 维) 堆叠
stacked = torch.stack(feature_emb_list, dim=1)     # [batch, num_fields, k]

# FM 公式: 0.5 * Σ((Σ f)^2 - Σ(f^2))
summed = stacked.sum(dim=1)                         # [batch, k]
sum_of_squares = (stacked * stacked).sum(dim=1)     # [batch, k]
logits += 0.5 * (summed * summed - sum_of_squares).sum(dim=1)  # [batch]
```
数学公式：

$$\text{FM} = \frac{1}{2}\sum_{f=1}^{k}\left((\sum_{i=1}^{n} v_{i,f})^2 - \sum_{i=1}^{n} v_{i,f}^2\right)$$

这等价于所有 field embedding 两两之间做 dot product：

$$\text{FM} = \sum_{i=1}^{n}\sum_{j=i+1}^{n} \langle v_i, v_j \rangle$$

表示两个 field 特征组合对预测的贡献。



3. Deep 部分（高阶非线性交互）

```bash
deep_input = torch.cat(feature_emb_list, dim=-1)   # [batch, n * k]
logits += self.deep_head(self.deep_network(deep_input)).squeeze(-1)
```

- 所有 field embedding 拼成一个大向量 [n * k]
- 通过 MLP（FullyConnectedLayer）学习高阶非线性交互
- MLP 最后一层通过 Linear → 1 输出标量



4. 最终输出

```bash
return torch.sigmoid(logits)    # [batch_size]，CTR 预估值
```

三个部分的 logit 相加后过 sigmoid，得到 (0, 1) 之间的点击概率。
参数量和计算量



#### FQA

#### DeepFM模型。哪些特征应该输入一阶和二阶项（FM侧），哪些特征应该输入Deep侧？

一、核心结论（工业落地标准方案）

1）FM 部分（一阶 LR + 二阶交叉项）

**全部离散类别特征（categorical feature）都进 FM**

- 用户 ID、物品 ID、类目、性别、城市、设备、标签、历史行为序列离散 ID
- 所有需要 Embedding 的稀疏 ID 类特征

> FM 作用：自动做任意两个特征的两两交叉，捕捉低阶组合特征（user+item，gender+category 等）。

2）Deep 部分（DNN）

两部分拼接送入 MLP：

1. **所有离散特征的 Embedding 向量（和 FM 共享 Embedding 矩阵！DeepFM 核心设计：Embedding 共享）**
2. **连续数值特征（numerical feature）**：年龄、点击率、时长、价格、曝光次数、统计类特征

------

二、严格拆分清单

① 送入 FM（1 阶 + 2 阶交互）

仅：**离散稀疏特征（类别特征）**

✅ 用户侧：user_id、gender、age_bin、city、occupation

✅ 物品侧：item_id、category、brand、shop_id

✅ 上下文：weekday、hour、device_type

连续数值**不要放进 FM 二阶交叉**，只给 Deep。

------

② 送入 Deep 网络的输入向量

输入 = [所有离散特征 Embedding 拼接] + [归一化后的连续数值特征]

1. 离散特征：和 FM 共用同一套 Embedding，参数共享，这是 DeepFM 最重要的设计，防止两套 Embedding 造成信息割裂。
2. 连续数值特征：
   - 数值型统计特征：ctr、浏览次数、商品价格、收入、时长
   - 必须做归一化（MinMax / StandardScaler）直接拼在向量尾部
3. DNN 负责学习**高阶非线性高阶组合特征**（三阶、四阶复杂特征交互），弥补 FM 只能做到二阶交叉的短板。
