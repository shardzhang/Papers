# word2vec Explained: Deriving Mikolov et al.'s Negative-Sampling Word-Embedding Method
# word2vec 详解：推导 Mikolov 等人的负采样词嵌入方法

Yoav Goldberg and Omer Levy
{yoav.goldberg,omerlevy}@gmail.com

February 14, 2014
2014 年 2 月 14 日

## Abstract 摘要

The word2vec software of Tomas Mikolov and colleagues[1] has gained a lot of traction lately, and provides state-of-the-art word embeddings.
Tomas Mikolov 及其同事的 word2vec 软件[1] 近来获得了广泛关注，并提供了最先进的词嵌入。

The learning models behind the software are described in two research papers [1, 2].
该软件背后的学习模型在两篇研究论文 [1, 2] 中进行了描述。

We found the description of the models in these papers to be somewhat cryptic and hard to follow.
我们发现这些论文中对模型的描述有些晦涩难懂。

While the motivations and presentation may be obvious to the neural-networks language-modeling crowd, we had to struggle quite a bit to figure out the rationale behind the equations.
虽然其动机和表述对神经网络语言建模领域的人来说可能显而易见，但我们却需要相当努力才能弄清楚这些方程背后的原理。

This note is an attempt to explain equation (4) (negative sampling) in "Distributed Representations of Words and Phrases and their Compositionality" by Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado and Jeffrey Dean [2].
本文试图解释 Tomas Mikolov、Ilya Sutskever、Kai Chen、Greg Corrado 和 Jeffrey Dean 所著《Distributed Representations of Words and Phrases and their Compositionality》[2] 中的方程 (4)（负采样）。

## 1 The skip-gram model
## 1. Skip-gram 模型

The departure point of the paper is the skip-gram model.
本文的出发点是 skip-gram 模型。

In this model we are given a corpus of words w and their contexts c.
在该模型中，我们给定一个包含词 w 及其上下文 c 的语料库。

We consider the conditional probabilities p(c|w), and given a corpus Text, the goal is to set the parameters θ of p(c|w; θ) so as to maximize the corpus probability:
我们考虑条件概率 p(c|w)，并且给定语料库 Text，目标是设置 p(c|w; θ) 的参数 θ，以最大化语料库概率：

(1) arg max θ ∏_{w∈Text} [ ∏_{c∈C(w)} p(c|w; θ) ]

in this equation, C(w) is the set of contexts of word w.
在此方程中，C(w) 是词 w 的上下文集合。

Alternatively:
或者：

(2) arg max θ ∏_{(w,c)∈D} p(c|w; θ)

here D is the set of all word and context pairs we extract from the text.
这里 D 是我们从文本中提取的所有词-上下文对的集合。

1https://code.google.com/p/word2vec/

### 1.1 Parameterization of the skip-gram model
### 1.1 Skip-gram 模型的参数化

One approach for parameterizing the skip-gram model follows the neural-network language models literature, and models the conditional probability p(c|w; θ) using soft-max:
参数化 skip-gram 模型的一种方法遵循神经网络语言模型文献，并使用 soft-max 对条件概率 p(c|w; θ) 进行建模：

(3) p(c|w; θ) = e^{v_c·v_w} / ∑_{c'∈C} e^{v_{c'}·v_w}

where v_c and v_w ∈ R^d are vector representations for c and w respectively, and C is the set of all available contexts.[2]
其中 v_c 和 v_w ∈ R^d 分别是对 c 和 w 的向量表示，C 是所有可用上下文的集合。[2]

The parameters θ are v_{c_i}, v_{w_i} for w ∈ V, c ∈ C, i ∈ 1, ···, d (a total of |C| × |V| × d parameters).
参数 θ 是 v_{c_i}, v_{w_i}，其中 w ∈ V, c ∈ C, i ∈ 1, ···, d（总共 |C| × |V| × d 个参数）。

We would like to set the parameters such that the product (2) is maximized.
我们希望设置参数使得乘积 (2) 最大化。

Now will be a good time to take the log and switch from product to sum:
现在是对数化并将乘积转换为求和的好时机：

(4) arg max θ ∑_{(w,c)∈D} log p(c|w) = ∑_{(w,c)∈D} (log e^{v_c·v_w} - log ∑_{c'} e^{v_{c'}·v_w})

An assumption underlying the embedding process is the following:
嵌入过程基于以下假设：

Assumption maximizing objective 4 will result in good embeddings v_w ∀ w ∈ V, in the sense that similar words will have similar vectors.
假设 最大化目标 (4) 将产生良好的嵌入 v_w ∀ w ∈ V，即相似词将具有相似的向量。

It is not clear to us at this point why this assumption holds.
目前我们不清楚为什么这个假设成立。

While objective (4) can be computed, it is computationally expensive to do so, because the term p(c|w; θ) is very expensive to compute due to the summation ∑_{c'∈C} e^{v_{c'}·v_w} over all the contexts c' (there can be hundreds of thousands of them).
虽然目标 (4) 理论上可以计算，但计算成本非常高，因为项 p(c|w; θ) 需要对所有上下文 c' 求和 ∑_{c'∈C} e^{v_{c'}·v_w}（可能有数十万个）。

One way of making the computation more tractable is to replace the softmax with an hierarchical softmax.
使计算更易处理的一种方法是用层次 softmax 替换 softmax。

We will not elaborate on this direction.
我们将不详细讨论这个方向。

[2] Throughout this note, we assume that the words and the contexts come from distinct vocabularies, so that, for example, the vector associated with the word dog will be different from the vector associated with the context dog.
[2] 在本文中，我们假设词和上下文来自不同的词汇表，因此，例如，与词 dog 相关联的向量将不同于与上下文 dog 相关联的向量。

This assumption follows the literature, where it is not motivated.
这一假设沿用了文献中的做法，而文献中并未说明其动机。

One motivation for making this assumption is the following: consider the case where both the word dog and the context dog share the same vector v.
做出这一假设的一个动机如下：考虑词 dog 和上下文 dog 共享相同向量 v 的情况。

Words hardly appear in the contexts of themselves, and so the model should assign a low probability to p(dog|dog), which entails assigning a low value to v·v which is impossible.
词几乎不会出现在自身的上下文中，因此模型应为 p(dog|dog) 分配低概率，这意味着需要为 v·v 分配低值，但这是不可能的。

## 2 Negative Sampling
## 2. 负采样

Mikolov et al. [2] present the negative-sampling approach as a more efficient way of deriving word embeddings.
Mikolov 等人 [2] 提出了负采样方法，作为推导词嵌入的更有效方式。

While negative-sampling is based on the skip-gram model, it is in fact optimizing a different objective.
虽然负采样基于 skip-gram 模型，但它实际上是在优化一个不同的目标。

What follows is the derivation of the negative-sampling objective.
以下是负采样目标的推导。

Consider a pair (w, c) of word and context.
考虑一个词-上下文对 (w, c)。

Did this pair come from the training data?
这对是否来自训练数据？

Let's denote by p(D = 1|w, c) the probability that (w, c) came from the corpus data.
我们用 p(D = 1|w, c) 表示 (w, c) 来自语料库数据的概率。

Correspondingly, p(D = 0|w, c) = 1 - p(D = 1|w, c) will be the probability that (w, c) did not come from the corpus data.
相应地，p(D = 0|w, c) = 1 - p(D = 1|w, c) 表示 (w, c) 不来自语料库数据的概率。

As before, assume there are parameters θ controlling the distribution: p(D = 1|w, c; θ).
与之前一样，假设存在控制分布的参数 θ：p(D = 1|w, c; θ)。

Our goal is now to find parameters to maximize the probabilities that all of the observations indeed came from the data:
我们的目标现在是找到参数，以最大化所有观测确实来自数据的概率：

arg max θ ∏_{(w,c)∈D} p(D = 1|w, c; θ)

= arg max θ log ∏_{(w,c)∈D} p(D = 1|w, c; θ)

= arg max θ ∑_{(w,c)∈D} log p(D = 1|w, c; θ)

The quantity p(D = 1|c, w; θ) can be defined using softmax:
p(D = 1|c, w; θ) 可以使用 softmax 定义：

p(D = 1|w, c; θ) = 1 / (1 + e^{-v_c·v_w})

Leading to the objective:
得到目标：

arg max θ ∑_{(w,c)∈D} log 1/(1 + e^{-v_c·v_w})

This objective has a trivial solution if we set θ such that p(D = 1|w, c; θ) = 1 for every pair (w, c).
如果我们设置 θ 使得对于每一对 (w, c) 都有 p(D = 1|w, c; θ) = 1，这个目标有一个平凡解。

This can be easily achieved by setting θ such that v_c = v_w and v_c·v_w = K for all v_c, v_w, where K is large enough number (practically, we get a probability of 1 as soon as K ≈ 40).
这可以通过设置 θ 使得对于所有 v_c, v_w，有 v_c = v_w 且 v_c·v_w = K 来轻松实现，其中 K 是足够大的数（实际上，当 K ≈ 40 时，我们就能得到概率为 1）。

We need a mechanism that prevents all the vectors from having the same value, by disallowing some (w, c) combinations.
我们需要一种机制通过禁止某些 (w, c) 组合来防止所有向量具有相同的值。

One way to do so, is to present the model with some (w, c) pairs for which p(D = 1|w, c; θ) must be low, i.e. pairs which are not in the data.
一种方法是向模型提供一些 p(D = 1|w, c; θ) 必须低的 (w, c) 对，即不在数据中的对。

This is achieved by generating the set D' of random (w, c) pairs, assuming they are all incorrect (the name "negative-sampling" stems from the set D' of randomly sampled negative examples).
这通过生成随机 (w, c) 对的集合 D' 来实现，假设它们都是不正确的（"负采样"这个名称源于随机采样的负例集合 D'）。

The optimization objective now becomes:
优化目标现在变为：

arg max θ ∏_{(w,c)∈D} p(D = 1|c, w; θ) ∏_{(w,c)∈D'} p(D = 0|c, w; θ)

= arg max θ ∏_{(w,c)∈D} p(D = 1|c, w; θ) ∏_{(w,c)∈D'} (1 - p(D = 1|c, w; θ))

= arg max θ ∑_{(w,c)∈D} log p(D = 1|c, w; θ) + ∑_{(w,c)∈D'} log(1 - p(D = 1|w, c; θ))

= arg max θ ∑_{(w,c)∈D} log 1/(1 + e^{-v_c·v_w}) + ∑_{(w,c)∈D'} log(1 - 1/(1 + e^{-v_c·v_w}))

= arg max θ ∑_{(w,c)∈D} log 1/(1 + e^{-v_c·v_w}) + ∑_{(w,c)∈D'} log(1/(1 + e^{v_c·v_w}))

If we let σ(x) = 1/(1 + e^{-x}) we get:
如果我们设 σ(x) = 1/(1 + e^{-x})，我们得到：

arg max θ ∑_{(w,c)∈D} log σ(v_c·v_w) + ∑_{(w,c)∈D'} log σ(-v_c·v_w)

which is almost equation (4) in Mikolov et al ([2]).
这几乎就是 Mikolov 等人 ([2]) 中的方程 (4)。

The difference from Mikolov et al. is that here we present the objective for the entire corpus D ∪ D', while they present it for one example (w, c) ∈ D and k examples (w, c_j) ∈ D', following a particular way of constructing D'.
与 Mikolov 等人的不同之处在于，这里我们给出了整个语料库 D ∪ D' 的目标，而他们则针对一个样本 (w, c) ∈ D 和 k 个样本 (w, c_j) ∈ D' 给出，遵循了一种特定的构造 D' 的方式。

Specifically, with negative sampling of k, Mikolov et al.'s constructed D' is k times larger than D, and for each (w, c) ∈ D we construct k samples (w, c_1), ..., (w, c_k), where each c_j is drawn according to its unigram distribution raised to the 3/4 power.
具体来说，在使用 k 个负样本的情况下，Mikolov 等人构造的 D' 比 D 大 k 倍，并且对于每个 (w, c) ∈ D，我们构造 k 个样本 (w, c_1), ..., (w, c_k)，其中每个 c_j 根据其一元分布（unigram distribution）的 3/4 次方进行抽样。

This is equivalent to drawing the samples (w, c) in D' from the distribution (w, c) ∼ p_{words}(w) p_{contexts}(c)^{3/4} / Z, where p_{words}(w) and p_{contexts}(c) are the unigram distributions of words and contexts respectively, and Z is a normalization constant.
这等价于从分布 (w, c) ∼ p_{words}(w) p_{contexts}(c)^{3/4} / Z 中抽取 D' 中的样本 (w, c)，其中 p_{words}(w) 和 p_{contexts}(c) 分别是词和上下文的一元分布，Z 是归一化常数。

In the work of Mikolov et al. each context is a word (and all words appear as contexts), and so p_{context}(x) = p_{words}(x) = count(x)/|Text|.
在 Mikolov 等人的工作中，每个上下文都是一个词（并且所有词都作为上下文出现），因此 p_{context}(x) = p_{words}(x) = count(x)/|Text|。

### 2.1 Remarks
### 2.1 备注

- Unlike the Skip-gram model described above, the formulation in this section does not model p(c|w) but instead models a quantity related to the joint distribution of w and c.
- 与上述 Skip-gram 模型不同，本节中的公式不是对 p(c|w) 建模，而是对与 w 和 c 的联合分布相关的量进行建模。

- If we fix the words representation and learn only the contexts representation, or fix the contexts representation and learn only the word representations, the model reduces to logistic regression, and is convex.
- 如果我们固定词表示而仅学习上下文表示，或者固定上下文表示而仅学习词表示，则该模型简化为逻辑回归，并且是凸的。

- However, in this model the words and contexts representations are learned jointly, making the model non-convex.
- 然而，在该模型中，词和上下文表示是联合学习的，这使得模型是非凸的。

## 3 Context definitions
## 3. 上下文定义

This section lists some peculiarities of the contexts used in the word2vec software, as reflected in the code.
本节列举了 word2vec 软件中使用的上下文的一些特性，这些特性反映在代码中。

Generally speaking, for a sentence of n words w_1, ..., w_n, contexts of a word w_i comes from a window of size k around the word: C(w) = w_{i-k}, ..., w_{i-1}, w_{i+1}, ..., w_{i+k}, where k is a parameter.
一般来说，对于一个包含 n 个词 w_1, ..., w_n 的句子，词 w_i 的上下文来自其周围大小为 k 的窗口：C(w) = w_{i-k}, ..., w_{i-1}, w_{i+1}, ..., w_{i+k}，其中 k 是一个参数。

However, there are two subtleties:
然而，有两个微妙之处：

**Dynamic window size** the window size that is being used is dynamic – the parameter k denotes the maximal window size.
**动态窗口大小** 所使用的窗口大小是动态的——参数 k 表示最大窗口大小。

For each word in the corpus, a window size k' is sampled uniformly from 1, ..., k.
对于语料库中的每个词，从 1, ..., k 中均匀采样一个窗口大小 k'。

**Effect of subsampling and rare-word pruning** word2vec has two additional parameters for discarding some of the input words: words appearing less than min-count times are not considered as either words or contexts, an in addition frequent words (as defined by the sample parameter) are down-sampled.
**子采样和稀有词剪枝的效果** word2vec 有两个额外的参数用于丢弃某些输入词：出现次数少于 min-count 的词不作为词或上下文考虑，此外，频繁词（由 sample 参数定义）被下采样。

Importantly, these words are removed from the text before generating the contexts.
重要的是，这些词在生成上下文之前会从文本中移除。

This has the effect of increasing the effective window size for certain words.
这具有增加某些词的有效窗口大小的效果。

According to Mikolov et al. [2], sub-sampling of frequent words improves the quality of the resulting embedding on some benchmarks.
根据 Mikolov 等人 [2] 的说法，频繁词的子采样提高了在某些基准测试上所得嵌入的质量。

The original motivation for sub-sampling was that frequent words are less informative.
子采样的原始动机是频繁词信息量较少。

Here we see another explanation for its effectiveness: the effective window size grows, including context-words which are both content-full and linearly far away from the focus word, thus making the similarities more topical.
这里我们看到了其有效性的另一种解释：有效窗口大小增大，包含了既有内容含量又在线性距离上远离焦点词的上下文词，从而使相似性更具主题性。

## 4 Why does this produce good word representations?
## 4. 为什么这能产生好的词表示？

Good question. We don't really know.
好问题。我们其实不知道。

The distributional hypothesis states that words in similar contexts have similar meanings.
分布假设指出，在相似上下文中出现的词具有相似的含义。

The objective above clearly tries to increase the quantity v_w·v_c for good word-context pairs, and decrease it for bad ones.
上述目标显然试图增加好的词-上下文对的 v_w·v_c 值，并降低差的词-上下文对的该值。

Intuitively, this means that words that share many contexts will be similar to each other (note also that contexts sharing many words will also be similar to each other).
直观地说，这意味着共享许多上下文的词将彼此相似（同时也要注意，共享许多词的上下文也将彼此相似）。

This is, however, very hand-wavy.
然而，这非常含糊其辞。

Can we make this intuition more precise? We'd really like to see something more formal.
我们能否使这种直觉更加精确？我们真的很希望看到更形式化的内容。

## References
## 参考文献

[1] Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. Efficient estimation of word representations in vector space. CoRR, abs/1301.3781, 2013.
[1] Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. Efficient estimation of word representations in vector space. CoRR, abs/1301.3781, 2013.

[2] Tomas Mikolov, Ilya Sutskever, Kai Chen, Gregory S. Corrado, and Jeffrey Dean. Distributed representations of words and phrases and their compositionality. In Advances in Neural Information Processing Systems 26: 27th Annual Conference on Neural Information Processing Systems 2013. Proceedings of a meeting held December 5-8, 2013, Lake Tahoe, Nevada, United States, pages 3111–3119, 2013.
[2] Tomas Mikolov, Ilya Sutskever, Kai Chen, Gregory S. Corrado, and Jeffrey Dean. Distributed representations of words and phrases and their compositionality. In Advances in Neural Information Processing Systems 26: 27th Annual Conference on Neural Information Processing Systems 2013. Proceedings of a meeting held December 5-8, 2013, Lake Tahoe, Nevada, United States, pages 3111–3119, 2013.
