# word2vec 详解：推导 Mikolov 等人的负采样词嵌入方法

> Yoav Goldberg, Omer Levy | Bar-Ilan University



本文详细推导了 Mikolov 等人 word2vec 模型中的负采样方法。核心内容：

- Skip-gram 模型参数化：使用 softmax 建模条件概率 $p(c|w)$
- 负采样推导：逐步推导目标函数从 softmax 到负采样的等价变换
- SGNS 等价性：证明 skip-gram with negative sampling 隐式分解了词-上下文矩阵的 PMI

关键发现：负采样实质上是**隐式矩阵分解**，目标函数等价于点式互信息（PMI）矩阵的分解。

---



## 摘要

Tomas Mikolov 及其同事的 word2vec 软件[1] 近来获得了广泛关注，并提供了最先进的词嵌入。

该软件背后的学习模型在两篇研究论文 [1, 2] 中进行了描述。

我们发现这些论文中对模型的描述有些晦涩难懂。

虽然其动机和表述对神经网络语言建模领域的人来说可能显而易见，但我们却需要相当努力才能弄清楚这些方程背后的原理。

本文试图解释 Tomas Mikolov、Ilya Sutskever、Kai Chen、Greg Corrado 和 Jeffrey Dean 所著《Distributed Representations of Words and Phrases and their Compositionality》[2] 中的方程 (4)（负采样）。



## 1. Skip-gram 模型

本文的出发点是 skip-gram 模型。

在该模型中，我们给定一个包含词 $w$ 及其上下文 $c$ 的语料库。

我们考虑条件概率 $p(c|w)$，并且给定语料库 $\mathrm{Text}$，目标是设置 $p(c|w; \theta)$ 的参数 $\theta$，以最大化语料库概率：

$$
\arg\max_{\theta} \prod_{w\in \mathrm{Text}} [ \prod_{c\in C(w)} p(c|w; \theta) ] \qquad (1)
$$

在此方程中，$C(w)$ 是词 $w$ 的上下文集合。

或者：

$$
\arg\max_{\theta} \prod_{(w,c)\in D} p(c|w; \theta) \qquad (2)
$$

这里 $D$ 是我们从文本中提取的所有词-上下文对的集合。

1https://code.google.com/p/word2vec/

### 1.1 Skip-gram 模型的参数化

参数化 skip-gram 模型的一种方法遵循神经网络语言模型文献，并使用 softmax 对条件概率 $p(c|w; \theta)$ 进行建模：

$$
p(c|w; \theta) = \frac{e^{v_c\cdot v_w}}{\sum_{c'\in C} e^{v_{c'}\cdot v_w}} \qquad (3)
$$

其中 $v_c$ 和 $v_w$ $\in$ $\mathbb{R}^d$ 分别是对 $c$ 和 $w$ 的向量表示，$C$ 是所有可用上下文的集合。[2]

参数 $\theta$ 是 $v_{c_i}$, $v_{w_i}$，其中 $w$ $\in$ $V$, $c$ $\in$ $C$, $i$ $\in$ $1, \cdots, d$（总共 $|C|$ $\times$ $|V|$ $\times$ $d$ 个参数）。

我们希望设置参数使得乘积 (2) 最大化。

现在是对数化并将乘积转换为求和的好时机：

$$
\arg\max_{\theta} \sum_{(w,c)\in D} \log p(c|w) = \sum_{(w,c)\in D} (\log e^{v_c\cdot v_w} - \log \sum_{c'} e^{v_{c'}\cdot v_w}) \qquad (4)
$$

嵌入过程基于以下假设：

假设 最大化目标 (4) 将产生良好的嵌入 $v_w$ $\forall$ $w$ $\in$ $V$，即相似词将具有相似的向量。

目前我们不清楚为什么这个假设成立。

虽然目标 (4) 理论上可以计算，但计算成本非常高，因为项 $p(c|w; \theta)$ 需要对所有上下文 $c'$ 求和 $\sum_{c'\in C} e^{v_{c'}\cdot v_w}$（可能有数十万个）。

使计算更易处理的一种方法是用层次 softmax 替换 softmax。

我们将不详细讨论这个方向。[2] 在本文中，我们假设词和上下文来自不同的词汇表，因此，例如，与词 dog 相关联的向量将不同于与上下文 dog 相关联的向量。

这一假设沿用了文献中的做法，而文献中并未说明其动机。

做出这一假设的一个动机如下：考虑词 dog 和上下文 dog 共享相同向量 $v$ 的情况。

词几乎不会出现在自身的上下文中，因此模型应为 $p(dog|dog)$ 分配低概率，这意味着需要为 $v_c\cdot v_w$ 分配低值，但这是不可能的。



## 2. 负采样

Mikolov 等人 [2] 提出了负采样方法，作为推导词嵌入的更有效方式。

虽然负采样基于 skip-gram 模型，但它实际上是在优化一个不同的目标。

以下是负采样目标的推导。

考虑一个词-上下文对 $(w, c)$。

这对是否来自训练数据？

我们用 $p(D = 1|w, c)$ 表示 $(w, c)$ 来自语料库数据的概率。

相应地，$p(D = 0|w, c) = 1 - p(D = 1|w, c)$ 表示 $(w, c)$ 不来自语料库数据的概率。

与之前一样，假设存在控制分布的参数 $\theta$：$p(D = 1|w, c; \theta)$。

我们的目标现在是找到参数，以最大化所有观测确实来自数据的概率：
$$
\arg\max_{\theta} \prod_{(w,c)\in D} p(D = 1|w, c; \theta)
$$
$$
= \arg\max_{\theta} \log \prod_{(w,c)\in D} p(D = 1|w, c; \theta)
$$
$$
= \arg\max_{\theta} \sum_{(w,c)\in D} \log p(D = 1|w, c; \theta)
$$
$p(D = 1|c, w; \theta)$ 可以使用 softmax 定义：

$$
p(D = 1|w, c; \theta) = \frac{1}{1 + e^{-v_c\cdot v_w}}
$$

得到目标：
$$
\arg\max_{\theta} \sum_{(w,c)\in D} \log 1/(1 + e^{-v_c\cdot v_w})
$$
如果我们设置 $\theta$ 使得对于每一对 $(w, c)$ 都有 $p(D = 1|w, c; \theta) = 1$，这个目标有一个平凡解。

这可以通过设置 $\theta$ 使得对于所有 $v_c$, $v_w$，有 $v_c = v_w$ 且 $v_c\cdot v_w = K$ 来轻松实现，其中 $K$ 是足够大的数（实际上，当 $K \approx 40$ 时，我们就能得到概率为 1）。

我们需要一种机制通过禁止某些 $(w, c)$ 组合来防止所有向量具有相同的值。

一种方法是向模型提供一些 $p(D = 1|w, c; \theta)$ 必须低的 $(w, c)$ 对，即不在数据中的对。

这通过生成随机 $(w, c)$ 对的集合 $D'$ 来实现，假设它们都是不正确的（"负采样"这个名称源于随机采样的负例集合 $D'$）。

优化目标现在变为：
$$
\arg\max_{\theta} \prod_{(w,c)\in D} p(D = 1|c, w; \theta) \prod_{(w,c)\in D'} p(D = 0|c, w; \theta)
$$
$$
= \arg\max_{\theta} \prod_{(w,c)\in D} p(D = 1|c, w; \theta) \prod_{(w,c)\in D'} (1 - p(D = 1|c, w; \theta))
$$
$$
= \arg\max_{\theta} \sum_{(w,c)\in D} \log p(D = 1|c, w; \theta) + \sum_{(w,c)\in D'} \log(1 - p(D = 1|w, c; \theta))
$$
$$
= \arg\max_{\theta} \sum_{(w,c)\in D} \log 1/(1 + e^{-v_c\cdot v_w}) + \sum_{(w,c)\in D'} \log(1 - 1/(1 + e^{-v_c\cdot v_w}))
$$
$$
= \arg\max_{\theta} \sum_{(w,c)\in D} \log 1/(1 + e^{-v_c\cdot v_w}) + \sum_{(w,c)\in D'} \log(1/(1 + e^{v_c\cdot v_w}))
$$
如果我们设 $\sigma(x) = 1/(1 + e^{-x})$，我们得到：
$$
\arg\max_{\theta} \sum_{(w,c)\in D} \log \sigma(v_c\cdot v_w) + \sum_{(w,c)\in D'} \log \sigma(-v_c\cdot v_w)
$$
这几乎就是 Mikolov 等人 ([2]) 中的方程 (4)。

与 Mikolov 等人的不同之处在于，这里我们给出了整个语料库 $D$ $\cup$ $D'$ 的目标，而他们则针对一个样本 $(w, c)$ $\in$ $D$ 和 $k$ 个样本 $(w, c_j)$ $\in$ $D'$ 给出，遵循了一种特定的构造 $D'$ 的方式。

具体来说，在使用 $k$ 个负样本的情况下，Mikolov 等人构造的 $D'$ 比 $D$ 大 $k$ 倍，并且对于每个 $(w, c)$ $\in$ $D$，我们构造 $k$ 个样本 $(w, c_1), \ldots, (w, c_k)$，其中每个 $c_j$ 根据其一元分布（unigram distribution）的 $3/4$ 次方进行抽样。

这等价于从分布 $(w, c) \sim \frac{p_{words}(w) \, p_{contexts}(c)^{3/4}}{Z}$ 中抽取 $D'$ 中的样本 $(w, c)$，其中 $p_{words}(w)$ 和 $p_{contexts}(c)$ 分别是词和上下文的一元分布，$Z$ 是归一化常数。

在 Mikolov 等人的工作中，每个上下文都是一个词（并且所有词都作为上下文出现），因此 $p_{context}(x) = p_{words}(x) = \frac{\mathrm{count}(x)}{|\mathrm{Text}|}$。



### 2.1 备注

- 与上述 Skip-gram 模型不同，本节中的公式不是对 $p(c|w)$ 建模，而是对与 $w$ 和 $c$ 的联合分布相关的量进行建模。

- 如果我们固定词表示而仅学习上下文表示，或者固定上下文表示而仅学习词表示，则该模型简化为逻辑回归，并且是凸的。

- 然而，在该模型中，词和上下文表示是联合学习的，这使得模型是非凸的。



## 3. 上下文定义

本节列举了 word2vec 软件中使用的上下文的一些特性，这些特性反映在代码中。

一般来说，对于一个包含 $n$ 个词 $w_1, \ldots, w_n$ 的句子，词 $w_i$ 的上下文来自其周围大小为 $k$ 的窗口：$C(w) = w_{i-k}, \ldots, w_{i-1}, w_{i+1}, \ldots, w_{i+k}$，其中 $k$ 是一个参数。

然而，有两个微妙之处：

**动态窗口大小** 所使用的窗口大小是动态的——参数 $k$ 表示最大窗口大小。

对于语料库中的每个词，从 $1, \ldots, k$ 中均匀采样一个窗口大小 $k'$。

**子采样和稀有词剪枝的效果** word2vec 有两个额外的参数用于丢弃某些输入词：出现次数少于 min-count 的词不作为词或上下文考虑，此外，频繁词（由 sample 参数定义）被下采样。

重要的是，这些词在生成上下文之前会从文本中移除。

这具有增加某些词的有效窗口大小的效果。

根据 Mikolov 等人 [2] 的说法，频繁词的子采样提高了在某些基准测试上所得嵌入的质量。

子采样的原始动机是频繁词信息量较少。

这里我们看到了其有效性的另一种解释：有效窗口大小增大，包含了既有内容含量又在线性距离上远离焦点词的上下文词，从而使相似性更具主题性。



## 4. 为什么这能产生好的词表示？

好问题。我们其实不知道。

分布假设指出，在相似上下文中出现的词具有相似的含义。

上述目标显然试图增加好的词-上下文对的 $v_w\cdot v_c$ 值，并降低差的词-上下文对的该值。

直观地说，这意味着共享许多上下文的词将彼此相似（同时也要注意，共享许多词的上下文也将彼此相似）。

然而，这非常含糊其辞。

我们能否使这种直觉更加精确？我们真的很希望看到更形式化的内容。



## 参考文献

[1] Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. Efficient estimation of word representations in vector space. CoRR, abs/1301.3781, 2013.

[2] Tomas Mikolov, Ilya Sutskever, Kai Chen, Gregory S. Corrado, and Jeffrey Dean. Distributed representations of words and phrases and their compositionality. In Advances in Neural Information Processing Systems 26: 27th Annual Conference on Neural Information Processing Systems 2013. Proceedings of a meeting held December 5-8, 2013, Lake Tahoe, Nevada, United States, pages 3111–3119, 2013.
