# AETN: 基于移动应用使用的通用用户嵌入-笔记

> Junqi Zhang¹\*、Bing Bai¹\*、Ye Lin¹、Jian Liang¹、Kun Bai¹、Fei Wang² | ¹腾讯云与智慧产业事业群、²美国康奈尔大学
>
> \*两位作者对本研究贡献相同。



建模过于复杂了。还需要再重新阅读！TODO



重要意义：根据用户在腾讯app平台上的保留、安装和卸载行为序列，学习通用的用户嵌入表示，用于下游任务

业务场景：腾讯手机管家和腾讯WiFi管家的信息流推荐场景

业务指标：

离线指标：

数据集：

模型类型：

创新点：

个人收获：

疑问：

- app应该是系统层面的，能够获取系统权限
- User Embeddings (Bottleneck Layer) 这个部分在整个模型结构中的作用是？





## FQA

#### Q：User Embeddings (Bottleneck Layer) 这个部分在整个模型结构中的作用是？

根据 AETN 模型的描述，**瓶颈层（Bottleneck Layer）** 的核心作用是：

1. 生成低维用户嵌入

- 将 Transformer 编码器输出的高维隐藏向量 $\boldsymbol{e}^{\Omega}_{x}$（维度 $d_{model}$）压缩为低维用户嵌入 $\boldsymbol{\widetilde{e}}$（维度 $d_{emb}$）
- 这是最终的用户表示，可用于下游任务

2. 信息压缩与瓶颈效应

- 作为编码器和解码器的**融合点**
- 强制模型将所有用户行为信息压缩到一个紧凑的向量中
- 来自原始输入的压缩信息成为**重建任务的唯一信息来源**

3. 架构位置

```
用户行为序列 → [保留自编码器] → [Transformer编码器] → [瓶颈层] → [Transformer解码器] → 重建
                                           ↓
                                      用户嵌入 (输出)
```

4. 技术实现

- 使用**单隐藏层自编码器**
- 激活函数：tanh
- 维度变化：$d_{model} \rightarrow d_{emb}$（通常 $d_{emb} \ll d_{model}$）

5. 为什么重要？

- **避免信息泄露**：确保嵌入只包含压缩后的信息
- **提升泛化能力**：低维表示更鲁棒
- **便于下游应用**：生成的用户嵌入可直接用于推荐、分类等任务



#### Q：User Embeddings是AutoEncoder部分输出的，还是Transformer Encoder部分输出的？
## User Embeddings 的来源

**User Embeddings 是由 Transformer Encoder 部分输出的，不是 AutoEncoder 部分。**

### 数据流

```
用户保留 (Retention)
    ↓
[保留自编码器] → 输出 x⁽¹⁾ (作为 Transformer 的输入)
    ↓
[Transformer Encoder] → 输出 e^Ω_x
    ↓
[瓶颈层 (简单自编码器)] → 输出用户嵌入 ẽ
    ↓
[Transformer 解码器] → 重建
```

### 关键区分

| 组件 | 作用 | 输出 |
|------|------|------|
| **保留自编码器** | 将高维稀疏保留向量编码为低维密集表示 | $\boldsymbol{x}^{(1)}$ (作为 Transformer 输入) |
| **Transformer Encoder** | 整合所有行为信息（保留、安装、卸载） | $\boldsymbol{e}^{\Omega}_{x}$ |
| **瓶颈层** | 将 $\boldsymbol{e}^{\Omega}_{x}$ 进一步压缩为用户嵌入 | $\boldsymbol{\widetilde{e}}$ (最终 User Embedding) |

### 总结

- **保留自编码器**：只是预处理，为 Transformer 提供输入
- **用户嵌入**：来自 Transformer Encoder 的输出，经瓶颈层压缩后得到





## 瓶颈层的作用

从原文（第175-177行）可以找到：

> 瓶颈层。瓶颈层是生成（低维）用户嵌入 $\boldsymbol{\widetilde{e}}$ 的地方。由于编码器和解码器在这一层融合，**来自原始输入的压缩信息成为重建任务的信息来源。**
>
> 在Transformer编码器输出的最终隐藏向量 $\boldsymbol{e}^{\Omega}_{x}$ 之上，我们使用一个**单隐藏层自编码器**进一步将维度从 $d_{model}$ 降低到 $d_{emb}$。瓶颈的激活函数是tanh。

### 简单解释

```
Transformer Encoder 输出 e^Ω_x (维度: d_model)
         ↓
    [单隐藏层自编码器]
    输入层: d_model → 隐藏层: d_emb → 输出层: d_model
         ↓
    隐藏层输出 = 用户嵌入 ẽ (维度: d_emb)
```

### 这个"简单自编码器"的功能

1. **降维**：将 $d_{model}$（如128）压缩到 $d_{emb}$（如16）
2. **信息压缩**：只保留最核心的用户特征
3. **生成最终嵌入**：输出的 $\boldsymbol{\widetilde{e}}$ 就是最终的用户表示

### 与"保留自编码器"的区别

| 组件 | 目的 |
|------|------|
| 保留自编码器 | 将稀疏保留向量编码为密集表示，作为 Transformer 输入 |
| 瓶颈层（简单自编码器） | 将 Transformer 输出压缩为最终用户嵌入 |

**瓶颈层本质上是一个维度压缩器**，输出的 $\boldsymbol{\widetilde{e}}$ 就是整个模型的核心产出——用户嵌入。



#### Q：为什么要把Retention Autoencoder Part 和 User Embeddings (Bottleneck Layer) 用虚线连起来？





## 瓶颈层代码实现分析

根据 `modules/models.py` 中的代码，瓶颈层的实现如下：

### 1. 瓶颈层维度定义（第44行）

```python
self.bottleneck_dims = [self.dae_dims[1], self.u_dims, self.dae_dims[1]]
# 例如：[512, 128, 512]
```

这是一个**3层结构**：输入层 → 隐藏层（用户嵌入） → 输出层

### 2. 权重初始化（第139-148行）

```python
for i, (d_in, d_out) in enumerate(zip(self.bottleneck_dims[:-1], self.bottleneck_dims[1:])):
    weight_key = "bottleneck_weight_{}to{}".format(i,i+1)
    bias_key = "bottleneck_bias_{}to{}".format(i,i+1)
    self.bottleneck_weights.append(tf.Variable(
        tf.initializers.glorot_uniform()([d_in, d_out]),
        name=weight_key))
    self.bottleneck_biases.append(tf.Variable(
        tf.initializers.truncated_normal(stddev=0.001)([d_out]),
        name=bias_key))
```

创建了：
- `bottleneck_weights[0]`: 输入层 → 隐藏层 (512 × 128)
- `bottleneck_weights[1]`: 隐藏层 → 输出层 (128 × 512)

### 3. 编码过程 - 生成用户嵌入（第332-334行）

```python
# Transformer Encoder 输出 img_emb: (N, 1, 512)
user_embeddings = tf.matmul(tf.squeeze(img_emb, [1]), 
                           self.bottleneck_weights[0]) + self.bottleneck_biases[0]  # (N, 128)
user_embeddings = get_activation('tanh')(user_embeddings)  # (N, 128)
model_outputs.append(user_embeddings)
```

**这就是用户嵌入的生成过程**：
- 输入：Transformer Encoder 输出 `img_emb` (维度 512)
- 线性变换：`img_emb @ W + b` (512 → 128)
- 激活函数：`tanh`
- 输出：用户嵌入 `user_embeddings` (维度 128)

### 4. 解码过程 - 重建保留（第343-346行）

```python
img_emb = tf.matmul(user_embeddings, 
                    self.bottleneck_weights[1]) + self.bottleneck_biases[1]  # (N, 512)
img_emb = get_activation('leakyrelu')(img_emb)  # (N, 512)
logits_image = tf.matmul(img_emb, 
                         self.dae_weights[-1]) + self.dae_biases[-1]  # (N, num_softid)
```

**重建过程**：
- 输入：用户嵌入 `user_embeddings` (维度 128)
- 线性变换：128 → 512
- 激活函数：`leakyrelu`
- 最终重建：512 → 应用数量

### 5. 数据流总结

```
保留向量 (M)
    ↓
[保留自编码器] → img_emb (512)
    ↓
[Transformer Encoder] → img_emb (512)
    ↓
[瓶颈层编码] img_emb @ W[0] + b[0] → user_embeddings (128) ← 最终用户嵌入
    ↓
[瓶颈层解码] user_embeddings @ W[1] + b[1] → 重建保留
```

### 6. 关键点

| 组件 | 维度 | 激活函数 |
|------|------|----------|
| 瓶颈层编码 | 512 → 128 | tanh |
| 瓶颈层解码 | 128 → 512 | leakyrelu |

**用户嵌入就是瓶颈层隐藏层的输出**，维度为 128（由 `u_dims` 参数控制）。





#### Q：保留自编码器 和 瓶颈层（简单自编码器）	是同一个东西么？

**不是同一个东西。**

从代码来看，它们是两个独立的组件：

## 保留自编码器 (Retention Autoencoder)

```python
# 第40行
self.dae_dims = params['q_dims'] + params['p_dims'][1:]
# 例如：dae_dims = [512, 256, 128, 256, 512]
```

**作用**：
- 编码保留向量为密集表示
- 第一层输出 `img_emb` 作为 Transformer 的输入
- 权重在 `self.dae_weights`

## 瓶颈层 (Bottleneck Layer)

```python
# 第44行
self.bottleneck_dims = [self.dae_dims[1], self.u_dims, self.dae_dims[1]]
# 例如：bottleneck_dims = [256, 128, 256]
```

**作用**：
- 将 Transformer 输出压缩为用户嵌入
- 权重在 `self.bottleneck_weights`

## 对比

| 组件         | 结构                | 权重                 | 作用         |
| ------------ | ------------------- | -------------------- | ------------ |
| 保留自编码器 | 512→256→128→256→512 | `dae_weights`        | 重建保留     |
| 瓶颈层       | 256→128→256         | `bottleneck_weights` | 生成用户嵌入 |

## 数据流

```
保留向量 (512)
    ↓
[保留自编码器] → img_emb (256) ← 这是 Transformer 的输入
    ↓
[Transformer Encoder] → (256)
    ↓
[瓶颈层] → user_embeddings (128) ← 最终用户嵌入
```

**总结**：保留自编码器是预处理模块，瓶颈层是生成用户嵌入的模块。两者独立，不共享权重。





#### Q：瓶颈层 仅仅用于生成 用户emb么？瓶颈层在训练过程中是否更新参数？

## 瓶颈层的作用

### 1. 不仅仅用于生成用户嵌入

瓶颈层有**3个作用**：

| 作用                      | 代码位置    | 说明                       |
| ------------------------- | ----------- | -------------------------- |
| **编码**                  | 第332-333行 | Transformer输出 → 用户嵌入 |
| **解码重建保留**          | 第343-345行 | 用户嵌入 → 重建保留向量    |
| **Transformer解码器输入** | 第348-359行 | 用户嵌入作为解码器的查询   |

### 2. 参数是否更新？

**是的，瓶颈层参数会更新。**

从代码可以看到：

```python
# 第143-148行：瓶颈层权重是 tf.Variable（可训练）
self.bottleneck_weights.append(tf.Variable(...))
self.bottleneck_biases.append(tf.Variable(...))
```

```python
# model_fn.py 第159行：瓶颈层L2正则化包含在损失函数中
loss = alpha_main * (loss_image + loss_new + loss_loss) + \
        model.compute_bottleneck_L2() + model.compute_transformer_L2()
```

### 3. 总结

| 问题                       | 答案                                                  |
| -------------------------- | ----------------------------------------------------- |
| 瓶颈层仅用于生成用户嵌入？ | **否**，还用于重建保留和Transformer解码器             |
| 参数是否更新？             | **是**，作为tf.Variable可训练，且L2正则化包含在损失中 |

**瓶颈层是连接编码器和解码器的关键组件，既压缩信息生成用户嵌入，又参与重建任务的训练。**







#### Q：保留自编码器 不生产用户emb？ 那保留自编码器的作用是什么？

## 保留自编码器的作用

从代码来看，**保留自编码器不直接生产用户嵌入**。它的作用是：

### 1. 编码保留向量为密集表示

```python
# 第297-311行
for i, (w, b) in enumerate(zip(self.dae_weights, self.dae_biases)):
    if i == 0:
        img = tf.matmul(img,w)+b
        img = get_activation('leakyrelu')(img)
        img_emb = tf.expand_dims(img, 1) # (N, 1, d_model) ← 这是 Transformer 的输入
    if i == 1:
        img = tf.matmul(img,w)+b
        img = get_activation('leakyrelu')(img)
    if i == 2:
        img = tf.matmul(img,w)+b
        img = get_activation('leakyrelu')(img)
    if i == 3:
        logits_dae = tf.matmul(img,w)+b  # ← 重建保留
        model_outputs.append(logits_dae)
```

### 2. 两个输出

| 输出         | 维度        | 用途                           |
| ------------ | ----------- | ------------------------------ |
| `img_emb`    | (N, 1, 512) | **Transformer Encoder 的输入** |
| `logits_dae` | (N, 512)    | 重建保留的损失计算             |

### 3. 数据流对比

```
保留向量 (512)
    ↓
[保留自编码器] → img_emb (512) → [Transformer Encoder]
    ↓
logits_dae (512) → 重建损失

---

Transformer Encoder 输出 (512)
    ↓
[瓶颈层] → user_embeddings (128) ← 最终用户嵌入
    ↓
logits_image → 重建损失
```

### 4. 总结

| 组件         | 输入             | 输出                               | 作用                            |
| ------------ | ---------------- | ---------------------------------- | ------------------------------- |
| 保留自编码器 | 保留向量         | `img_emb` + `logits_dae`           | 预处理，为 Transformer 提供输入 |
| 瓶颈层       | Transformer 输出 | `user_embeddings` + `logits_image` | 生成用户嵌入                    |

**保留自编码器是预处理模块，瓶颈层才是生成用户嵌入的模块。**











抱歉之前说错了。让我根据代码确认实际维度：

## 实际维度（从 train.py 第109-111行）

```python
'p_dims': [256, 512, NUM_SOFTID],  # 编码器结构
'q_dims': None,  # 解码器结构（自动反转）
'u_dims': 128,   # 用户嵌入维度
```

## 计算后的维度

```python
# q_dims = p_dims[::-1] = [NUM_SOFTID, 512, 256]
# dae_dims = q_dims + p_dims[1:] = [NUM_SOFTID, 512, 256, 512, NUM_SOFTID]
```

| 组件             | 维度                                      |
| ---------------- | ----------------------------------------- |
| **保留自编码器** | NUM_SOFTID → 512 → 256 → 512 → NUM_SOFTID |
| **瓶颈层**       | 512 → 128 → 512                           |

## 正确的数据流

```
保留向量 (NUM_SOFTID)
    ↓
[保留自编码器] → img_emb (512) → [Transformer Encoder]
    ↓
logits_dae (NUM_SOFTID) → 重建损失

---

Transformer Encoder 输出 (512)
    ↓
[瓶颈层] → user_embeddings (128) ← 最终用户嵌入
    ↓
logits_image (NUM_SOFTID) → 重建损失
```

## 总结

- **保留自编码器输出 `img_emb` 维度是 512**（不是256）
- **瓶颈层输入维度是 512**
- **用户嵌入维度是 128**





#### Q：logits_dae 和 logits_image 有区别么？ 分别怎么用？

## logits_dae vs logits_image

两者都是**重建保留向量**，但路径不同：

### 1. 生成路径对比

| 输出           | 路径                                             | 来源                 |
| -------------- | ------------------------------------------------ | -------------------- |
| `logits_dae`   | 保留 → DAE → logits_dae                          | 保留自编码器直接输出 |
| `logits_image` | 保留 → DAE → Transformer → 瓶颈层 → logits_image | 经过完整管线         |

### 2. 代码位置

**logits_dae**（第310-311行）：
```python
# 保留自编码器直接输出
if i == 3:
    logits_dae = tf.matmul(img,w)+b
    model_outputs.append(logits_dae)
```

**logits_image**（第343-346行）：
```python
# 从用户嵌入重建保留
img_emb = tf.matmul(user_embeddings, self.bottleneck_weights[1]) + self.bottleneck_biases[1]
img_emb = get_activation('leakyrelu')(img_emb)
logits_image = tf.matmul(img_emb, self.dae_weights[-1]) + self.dae_biases[-1]
model_outputs.append(logits_image)
```

### 3. 损失函数用途（model_fn.py）

```python
# 第137-138行
loss_dae = get_loss(...)(inputs[0], logits_dae, weight_positive_dae)      # 辅助损失
loss_image = get_loss(...)(inputs[0], logits_image, weight_positive_image) # 主损失
```

### 4. 总结

| 项目     | logits_dae      | logits_image               |
| -------- | --------------- | -------------------------- |
| **作用** | 辅助重建保留    | 主重建保留                 |
| **路径** | DAE 直接输出    | DAE → Transformer → 瓶颈层 |
| **意义** | 训练 DAE 编码器 | 确保用户嵌入能重建保留     |

**简言之：`logits_dae` 是辅助任务，`logits_image` 是主任务。**





## img_emb 的后续使用

从代码（第343-346行）可以看到：

```python
# main reconstruction of retention
img_emb = tf.matmul(user_embeddings,self.bottleneck_weights[1]) + self.bottleneck_biases[1] # (N, d_model)
img_emb = get_activation('leakyrelu')(img_emb) # (N, d_model)
logits_image = tf.matmul(img_emb, self.dae_weights[-1]) + self.dae_biases[-1] # (N, num_softid)
model_outputs.append(logits_image)
```

### 数据流

```
user_embeddings (128)
    ↓ (bottleneck_weights[1]: 128×512)
img_emb (512) ← 中间结果
    ↓ (daeweights[-1]: 512×NUM_SOFTID)
logits_image (NUM_SOFTID) → 计算重建损失
```

### 总结

`img_emb` 是**中间结果**，用于：
1. 从用户嵌入（128维）解码回保留表示（512维）
2. 再通过 `daeweights[-1]` 映射回应用空间（NUM_SOFTID维）
3. 最终计算 `loss_image` 重建损失

**`img_emb` 本身不直接使用，只是计算 `logits_image` 的中间步骤。**



#### Q：损失是什么？

## 实际损失函数

代码中的实际实现是：

```python
# 第157-169行
if alpha_main > 0:
    loss = alpha_main * (loss_image + loss_new + loss_loss) + \
            model.compute_bottleneck_L2() + model.compute_transformer_L2()
    if alpha_dae > 0:
        loss += alpha_dae * loss_dae
        loss += model.compute_dae_L2()
    else:
        loss += model.compute_dae_L2(only_part=True)
    if train_mlm:
        loss += train_mlm * (loss_bert_new + loss_bert_loss)
    if alpha_classifier > 0:
        loss += alpha_classifier * loss_classifier
        loss += model.compute_classifier_L2()
```

## 两种模式

| 条件             | 损失函数                                                     |
| ---------------- | ------------------------------------------------------------ |
| `alpha_main > 0` | 主损失 + 辅助损失（可选） + MLM损失（可选） + 分类损失（可选） |
| `alpha_main = 0` | 仅辅助损失 + MLM损失（可选） + 分类损失（可选）              |

## 完整展开（当 alpha_main > 0 时）

```
loss = alpha_main * (loss_image + loss_new + loss_loss)  # 主损失
     + bottleneck_L2 + transformer_L2                    # L2正则化
     + alpha_dae * loss_dae                              # 辅助损失（可选）
     + dae_L2                                            # DAE正则化
     + train_mlm * (loss_bert_new + loss_bert_loss)     # MLM损失（可选）
     + alpha_classifier * loss_classifier                # 分类损失（可选）
     + classifier_L2                                     # 分类器正则化
```

## 总结

我之前简化了，实际代码更复杂。**两个公式都是对的**，只是我表述的方式不同。





#### Q：Transformer Decoder的输入是什么？

## Transformer Decoder 的输入

从代码（第348-365行）可以看到：

### 1. 输入构建过程

```python
use_emb = tf.expand_dims(user_embeddings, 1)  # (N, 1, 128)

# 安装序列：用户嵌入 + 安装位置嵌入
img_new_emb = tf.concat([tf.tile(use_emb, [1, tf.shape(new_emb)[1], 1]), new_pos_emb], 2)  # (N, length_his, 128+d_model)

# 卸载序列：用户嵌入 + 卸载位置嵌入
img_loss_emb = tf.concat([tf.tile(use_emb, [1, tf.shape(loss_emb)[1], 1]), loss_pos_emb], 2)  # (N, length_his, 128+d_model)

# 降维
img_new_emb = reduce_dim_layer(img_new_emb)  # (N, length_his, d_model)
img_loss_emb = reduce_dim_layer(img_loss_emb)  # (N, length_his, d_model)
```

### 2. Transformer Decoder 的输入

```python
queries = tf.concat([img_new_emb, img_loss_emb], 1)  # (N, 2*length_his, d_model)
keys = tf.concat([img_new_emb, img_loss_emb], 1)     # (N, 2*length_his, d_model)
values = tf.concat([img_new_emb, img_loss_emb], 1)   # (N, 2*length_his, d_model)
```

### 3. 数据流总结

| 组件           | 输入                    | 维度                       |
| -------------- | ----------------------- | -------------------------- |
| `img_new_emb`  | 用户嵌入 + 安装位置嵌入 | (N, length_his, d_model)   |
| `img_loss_emb` | 用户嵌入 + 卸载位置嵌入 | (N, length_his, d_model)   |
| **最终输入**   | 拼接两者                | (N, 2*length_his, d_model) |

### 4. 输出

```python
logits_new = ...  # 重建安装序列
logits_loss = ...  # 重建卸载序列
```

**Transformer Decoder 的输入是用户嵌入与位置嵌入的组合，用于重建安装和卸载序列。**