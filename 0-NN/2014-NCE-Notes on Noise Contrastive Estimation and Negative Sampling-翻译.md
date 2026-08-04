# Notes on Noise Contrastive Estimation and Negative Sampling——噪声对比估计与负采样笔记

> Chris Dyer | 卡内基梅隆大学计算机科学学院

本文分享了一篇对噪声对比估计（Noise Contrastive Estimation, NCE）和负采样（Negative Sampling）进行深入比较的技术笔记。文章形式化了两者作为概率模型参数估计方法的定义，指出尽管表面上相似，NCE是一种渐进无偏的通用估计技术，而负采样本质上是一类用于学习词表示的二元分类模型，并非通用估计器。

- NCE 将语言模型估计问题转化为二元分类问题，使用噪声分布生成负样本，并通过固定配分函数避免对整个词汇表的求和
- 负采样是 NCE 在 $k=|V|$ 且 $q$ 为均匀分布时的特例，但其条件概率与语言模型概率不一致
- NCE 的梯度在 $k \to \infty$ 时趋近于对数似然梯度，具有渐进一致性保证

关键发现：

- NCE 是渐进无偏的通用参数估计方法，适用于任意局部归一化语言模型
- 负采样适用于词表示学习，但不适用于语言模型的参数估计
- 如目标是语言建模，应使用 NCE；如目标是学习词表示，NCE 和负采样均可考虑

---

## 摘要

估计概率语言模型（如最大熵模型和概率神经模型）的参数在计算上非常困难，因为需要通过对整个词汇表（可能包含数百万个词类型）求和来评估配分函数。两种密切相关的策略——噪声对比估计（Mnih 和 Teh, 2012 [7]；Mnih 和 Kavukcuoglu, 2013 [6]；Vaswani 等人, 2013 [8]）和负采样（Mikolov 等人, 2012 [5]；Goldberg 和 Levy, 2014 [3]）——已成为解决这一计算问题的流行方法，但关于何时使用哪种方法仍存在一些混淆。本文阐述了它们彼此之间以及与其他估计技术之间的关系。分析表明，尽管它们表面上相似，但 NCE 是一种渐进无偏的通用参数估计技术，而负采样最好被理解为一类对学习词表示有用、但并非通用估计器的二元分类模型。

---

## 1 引言

假设以下语言模型基于给定的上下文 $c$ 预测词汇表 $V$ 中的词 $w$ [^1]：

$$
p_\theta(w \mid c) = \frac{u_\theta(w, c)}{\sum_{w' \in V} u_\theta(w', c)} = \frac{u_\theta(w, c)}{Z_\theta(c)} \qquad (1)
$$

其中 $u_\theta(w, c) = \exp s_\theta(w, c)$ 为上下文中的词分配一个得分， $Z(c)$ 是将其归一化为概率分布的配分函数， $s_\theta(w, c)$ 关于 $\theta$ 可微。标准的学習过程是最大化训练数据样本的似然。不幸的是，计算该概率（及其导数）非常昂贵，因为需要对 $V$ 中所有词求和，而 $V$ 通常非常庞大。

如何解决？由于对数似然的导数包含模型分布下参数的期望值项，经典策略是使用重要性采样和相关的蒙特卡洛技术来近似这些期望（Bengio 和 Senécal, 2003 [1]）。噪声对比估计和负采样代表了这些技术的演进。它们通过将计算昂贵的學習问题转化为一个使用相同参数但所需统计量更容易计算的二元分类代理问题来工作。

[^1]: 这里所说的语言模型是指每次生成一个词、以其他环境上下文（如先前生成或周围的词、主题标签、另一种语言的文本等）为条件的模型。不包括所谓的"整句"或"全局归一化"语言模型。虽然原则上这些模型也可以使用本文所述技术进行学习，但本文聚焦于一次预测单个词的模型。

### 1.1 经验分布、噪声分布和模型分布

我将 $\tilde{p}(w \mid c)$ 和 $\tilde{p}(c)$ 称为经验分布。我们的任务是找到模型 $p_\theta(w \mid c)$ 的参数 $\theta$ ，使其在最小交叉熵意义上尽可能接近经验分布。为避免昂贵的求和，引入"噪声"分布 $q(w)$ 。在实践中， $q$ 可以是均匀分布、经验 unigram 分布，或"展平"的经验 unigram 分布（将每个概率取 $\alpha$ 次幂后重新归一化，其中 $0 < \alpha < 1$ ）。

## 2 噪声对比估计（NCE）

NCE 将语言模型估计问题简化为估计一个概率二元分类器参数的问题，该分类器使用相同的参数来区分来自经验分布的样本和由噪声分布生成的样本（Gutmann 和 Hyvärinen, 2010 [4]）。两类训练数据生成如下：从 $\tilde{p}(c)$ 采样一个 $c$ ，然后从 $\tilde{p}(w \mid c)$ 采样一个"真实"样本（附带标签 $D=1$ ，表示该数据点来自真实分布），以及从 $q$ 采样 $k$ 个"噪声"样本（附带标签 $D=0$ ，表示这些数据点是噪声）。因此，给定 $c$ ，两类数据中 $(d, w)$ 的联合概率具有两个分布混合的形式：

$$
p(d, w \mid c) =
\begin{cases}
\frac{k}{1+k} \times q(w) & \text{if } d = 0 \\
\frac{1}{1+k} \times \tilde{p}(w \mid c) & \text{if } d = 1
\end{cases}
$$

利用条件概率的定义，可以将其转化为在观察到 $w$ 和 $c$ 后 $d$ 的条件概率：

$$
p(D=0 \mid c, w) = \frac{\frac{k}{1+k} \times q(w)}{\frac{1}{1+k} \times \tilde{p}(w \mid c) + \frac{k}{1+k} \times q(w)} = \frac{k \times q(w)}{\tilde{p}(w \mid c) + k \times q(w)}
$$

$$
p(D=1 \mid c, w) = \frac{\tilde{p}(w \mid c)}{\tilde{p}(w \mid c) + k \times q(w)}
$$

注意这些概率是用经验分布表示的。

NCE 将经验分布 $\tilde{p}(w \mid c)$ 替换为模型分布 $p_\theta(w \mid c)$ ，并通过最大化上述创建的"代理语料库"的条件似然来选择 $\theta$ 。但到目前为止，我们还没有解决任何计算问题： $p_\theta(w \mid c)$ 仍然需要评估配分函数——我们所做的只是通过添加一些噪声来变换目标函数。为了避免评估配分函数的开销，NCE 做了两个进一步的假设。首先，它提出将配分函数值 $Z(c)$ 估计为参数 $z_c$ （因此，对于每个经验的 $c$ ，经典 NCE 引入一个参数）。其次，对于具有大量参数的神经网络，固定所有 $c$ 的 $z_c=1$ 是有效的（Mnih 和 Teh, 2012 [7]）。后一个假设既减少了参数数量，又鼓励模型具有"自归一化"输出（即 $Z(c) \approx 1$ ）。基于这些假设，我们现在可以用 $\theta$ 写出成为噪声样本或真实分布样本的条件似然：

$$
p(D=0 \mid c, w) = \frac{k \times q(w)}{u_\theta(w, c) + k \times q(w)}
$$

$$
p(D=1 \mid c, w) = \frac{u_\theta(w, c)}{u_\theta(w, c) + k \times q(w)}
$$

现在我们有了一个参数为 $\theta$ 的二元分类问题，可以通过最大化 $D$ 的条件对数似然来训练，其中选择了 $k$ 个负样本：

$$
\mathcal{L}_{\text{NCE}}^k = \sum_{(w,c) \in \mathcal{D}} \left( \log p(D=1 \mid c, w) + k \mathbb{E}_{w \sim q} \log p(D=0 \mid c, w) \right) \qquad (2)
$$

不幸的是，该求和中第二项的期望仍然是一个难以计算的求和——它是 $k$ 乘以在上下文 $c$ 下、噪声分布覆盖所有 $V$ 中词时生成负标签的期望对数概率（根据当前模型）。我们仍然需要遍历整个词汇表。因此，最后一步是将该期望替换为蒙特卡洛近似：

$$
\begin{aligned}
\mathcal{L}_{\text{MC-NCE}}^k &= \sum_{(w,c) \in \mathcal{D}} \left( \log p(D=1 \mid c, w) + k \times \sum_{i=1, w \sim q}^k \frac{1}{k} \times \log p(D=0 \mid c, w) \right) \\
&= \sum_{(w,c) \in \mathcal{D}} \left( \log p(D=1 \mid c, w) + \sum_{i=1, w \sim q}^k \log p(D=0 \mid c, w) \right) \qquad (3)
\end{aligned}
$$

### 2.1 渐进分析

尽管目标函数 $\mathcal{L}_{\text{NCE}}^k$ 是难解的，但其导数揭示了 NCE 有效的原因。该导数可以写为：

$$
\frac{\partial}{\partial \theta} \mathcal{L}_{\text{NCE}}^k = \sum_{(w',c) \in \mathcal{D}} \left( \sum_{w \in V} \frac{k \times q(w)}{u_\theta(w \mid c) + k \times q(w)} \times (\tilde{p}(w \mid c) - u_\theta(w \mid c)) \frac{\partial}{\partial \theta} \log u_\theta(w \mid c) \right) \qquad (4)
$$

容易看出，在极限情况 $k \to \infty$ 下，该导数趋近于 $\mathcal{D}$ 在 $p_\theta$ 下的对数似然梯度（此外， $\mathcal{L}_{\text{MC-NCE}}^k \to \mathcal{L}_{\text{NCE}}^k$ ）。也就是说，当模型分布与经验分布匹配时，梯度为零。

## 3 负采样

负采样是流行工具 word2vec 使用的 NCE 变体，它也生成一个代理语料库并将 $\theta$ 学习为一个二元分类问题，但它对给定 $(w, c)$ 的条件概率定义不同：

$$
p(D=0 \mid c, w) = \frac{1}{u_\theta(w, c) + 1}
$$

$$
p(D=1 \mid c, w) = \frac{u_\theta(w, c)}{u_\theta(w, c) + 1}
$$

该目标函数可以从几个角度理解。首先，它等价于 $k = |V|$ 且 $q$ 为均匀分布时的 NCE。其次，它可以理解为 Collobert 等人（2011 [2]）的 hinge 目标函数，其中 max 函数被替换为 softmax。因此，除了 $k=|V|$ 且 $q$ 均匀的情况外，给定 $(w,c)$ 时 $D$ 的条件概率与 $(w,c)$ 的语言模型概率不一致，因此使用该目标函数估计的 $\theta$ 不会优化公式 1 中语言模型的似然。因此，虽然负采样可能适用于词表示学习，但它不具备 NCE 所具有的渐进一致性保证。

## 4 结论

NCE 是一种为任意局部归一化语言模型学习参数的有效方法。然而，负采样应被视为一种为其他任务生成词表示的替代任务，其本身并非语言生成模型中参数的学习方法。因此，如果你的目标是语言建模，应使用 NCE；如果你的目标是词表示学习，NCE 和负采样都可以考虑。

## 参考文献

[1] Yoshua Bengio and Jean-Sébastien Senécal. 2003. Quick training of probabilistic neural nets by importance sampling. In *Proc. AISTATS*.

[2] Ronan Collobert, Jason Weston, Léon Bottou, Michael Karlen, Koray Kavukcuoglu, and Pavel Kuksa. 2011. Natural language processing (almost) from scratch. *JMLR*.

[3] Yoav Goldberg and Omer Levy. 2014. word2vec explained: Deriving Mikolov et al.'s negative-sampling word-embedding method.

[4] Michael Gutmann and Aapo Hyvärinen. 2010. Noise-contrastive estimation: A new estimation principle for unnormalized statistical models. In *Proc. AISTATS*.

[5] Tomas Mikolov, Ilya Sutskeve, Kai Chen, Greg Corrado, and Jeffrey Dean. 2012. Distributed representations of words and phrases and their compositionality. In *Proc. NIPS*.

[6] Andriy Mnih and Koray Kavukcuoglu. 2013. Learning word embeddings efficiently with noise-contrastive estimation. In *Proc. NIPS*.

[7] Andriy Mnih and Yee Whye Teh. 2012. A fast and simple algorithm for training neural probabilistic language models. In *Proc. ICML*.

[8] Ashish Vaswani, Yinggong Zhao, Victoria Fossum, and David Chiang. 2013. Decoding with large-scale neural language models improves translation. In *Proc. EMNLP*.
