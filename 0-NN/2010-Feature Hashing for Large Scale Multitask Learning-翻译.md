# 2010-面向大规模多任务学习的特征哈希

> Kilian Weinberger, Anirban Dasgupta, Josh Attenberg, John Langford, Alex Smola | Yahoo! Research, 2821 Mission College Blvd., Santa Clara, CA 95051 USA


本文分享了特征哈希（Feature Hashing）用于大规模多任务学习的理论与应用，即著名的"哈希技巧"（hashing-trick）论文。核心内容：

- 提出带符号的哈希函数变体，消除原始哈希核的偏差，并证明哈希核内积的无偏性及其方差界
- 首次为哈希内积的失真提供**指数尾界**，证明在哈希映射下 **向量长度以高概率保持**
- 证明独立哈希的子空间之间的干扰以高概率可忽略，使数十万任务可共享同一高度压缩的特征空间
- 将协同式电子邮件垃圾邮件过滤作为哈希表示的新应用，在大规模真实数据上实验验证

关键发现：在 43 万余用户、320 万封邮件的垃圾邮件过滤任务上，个性化哈希分类器平均减少 30% 漏检垃圾邮件；训练数据充足的用户最高减少 65%，即使从未贡献训练数据的用户也获得近 20% 的改善；**哈希维度为 $2^{22}$ 时即可达到与无哈希基线相当的性能**。

---


## 摘要

**经验证据**表明，**哈希是降维和实用非参数估计的一种有效策略**。在本文中，我们为特征哈希提供了指数尾界，并表明随机子空间之间的交互作用以高概率可以忽略不计。我们通过一个新的应用场景——具有数十万任务的多任务学习——的实验结果展示了该方法的可行性。

**关键词**：核方法、集中不等式、文档分类、分类器个性化、多任务学习[^1]

[^1]: 初步工作。正在国际机器学习会议（ICML）审稿中。请勿传播。


## 1. 引言

核方法使用内积作为比较对象的基本工具。也就是说，给定某个域 $\mathcal{X}$ 中的对象 $x_1, \ldots, x_n \in \mathcal{X}$ ，它们依赖于

$$
k(x_i, x_j) := \langle \phi(x_i), \phi(x_j) \rangle \qquad (1)
$$

来分别比较 $x_i$ 的特征 $\phi(x_i)$ 和 $x_j$ 的特征 $\phi(x_j)$ 。

式 (1) 通常被著名地称为核技巧（kernel-trick）。它允许通过 **半正定核矩阵 $k$ 的定义**，隐式地使用非常高维的特征向量 $\phi(x_i)$ 和 $\phi(x_j)$ 之间的内积，而永远不必直接计算向量 $\phi(x_i)$ 。这在分类设置中可能特别强大，因为原始输入表示具有非线性的决策边界。通常，在高维特征空间 $\phi(x_i)$ 中可以实现线性可分。

然而在实践中，例如在文本分类中，研究者经常遇到相反的问题：原始输入空间几乎是线性可分的（通常是由于存在手工构造的非线性特征），然而训练集的规模可能过大且维度非常高。在这种情况下，无需将输入向量映射到更高维的特征空间。相反，有限的内存使得存储核矩阵变得不可行。

对于这种常见场景，最近有几位作者提出了核技巧的一种替代但高度互补的变体，我们称之为哈希技巧（hashing-trick）：将高维输入向量 $x$ 哈希到低维特征空间 $\mathbb{R}^m$ 中，映射为 $\phi: \mathcal{X} \rightarrow \mathbb{R}^m$ （[11, 18]）。因此，分类器的参数向量可以存在于 $\mathbb{R}^m$ 中，而不是在使用核矩阵时存在于 $\mathbb{R}^n$ 中，或存在于原始输入空间 $\mathbb{R}^d$ 中，其中 $m \ll n$ 且 $m \ll d$ 。与随机投影不同，哈希技巧保留稀疏性，并且不会引入存储投影矩阵的额外开销。

据我们所知，我们是第一个为这些哈希内积的典范失真提供指数尾界的工作。我们还表明，哈希技巧在多任务学习场景中可能特别强大，其中原始特征空间是数据 $\mathcal{X}$ 与任务集合 $U$ 的叉积。我们表明，可以为每个任务使用不同的哈希函数 $\phi_1, \ldots, \phi_{|U|}$ ，将数据映射到一个联合空间中，且干扰很小。

虽然哈希技巧存在许多潜在应用，但作为具体案例研究，我们专注于协同式电子邮件垃圾邮件过滤。在这种场景中，数十万用户共同将电子邮件标记为垃圾邮件或非垃圾邮件，并且每个用户都期望获得反映其特定偏好的个性化分类器。在这里，任务集合 $U$ 是电子邮件用户的数量（对于 Yahoo Mail™ 或 Gmail™ 这样的开放系统，这可能非常大），特征空间跨越多种语言的词表的并集。

本文有四个主要贡献：1. 在第 2 节中，我们引入了具有无偏内积的专用哈希函数，可直接应用于各种各样的核方法。2. 在第 3 节中，我们提供了指数尾界，有助于解释为什么哈希特征向量反复地、有时令人惊讶地产生了强大的经验结果。3. 同样在第 3 节中，我们表明独立哈希的子空间之间的干扰以高概率可以忽略不计，这允许在高度压缩的空间中进行大规模多任务学习。4. 在第 5 节中，我们引入协同式电子邮件垃圾邮件过滤作为哈希表示的一个新应用，并在大规模真实世界的垃圾邮件数据集上提供实验结果。


## 2. 哈希函数

我们引入了 [18] 提出的哈希核的一个变体。该方案通过引入哈希特征的带符号求和进行了修改，而原始的哈希核使用无符号求和。这一修改产生了无偏估计，我们将在下一节中展示并进一步利用这一点。

**定义 1** 用 $h$ 表示一个哈希函数 $h: \mathbb{N} \rightarrow \{1, \ldots, m\}$ 。此外，用 $\xi$ 表示一个哈希函数 $\xi: \mathbb{N} \rightarrow \{\pm 1\}$ 。那么对于向量 $x, x' \in \ell_2$ ，我们定义哈希特征映射 $\phi$ 及相应的内积为

$$
\phi^{(h,\xi)}_i(x) = \sum_{j:h(j)=i} \xi(j) x_j \qquad (2)
$$

以及

$$
\langle x, x' \rangle_\phi := \left\langle \phi^{(h,\xi)}(x), \phi^{(h,\xi)}(x^{\prime}) \right\rangle. \qquad (3)
$$

尽管定义 1 中的哈希函数定义在自然数 $\mathbb{N}$ 上，但在实践中我们经常考虑任意字符串上的哈希函数。这两者是等价的，因为每个有限长度的字符串都可以用唯一的自然数表示。

通常，我们将记号 $\phi^{(h,\xi)}(\cdot)$ 简写为 $\phi(\cdot)$ 。当 $\phi = \phi^{(h,\xi)}$ 且 $\phi' = \phi^{(h',\xi^{\prime})}$ 满足 $h' \neq h$ 或 $\xi \neq \xi'$ 时，两个哈希函数 $\phi$ 和 $\phi'$ 是不同的。二元哈希 $\xi$ 的目的是消除 [18] 的哈希核中固有的偏差。

在多任务设置中，我们获得与任务组合的实例 $(x, u) \in \mathcal{X} \times U$ 。我们可以自然地将定义 1 扩展到哈希对，并将写作 $\phi_u(x) = \phi(x, u)$ 。


## 3. 分析

本节致力于哈希核及其应用的理论分析。从这个意义上说，本文接续了 [18] 未尽的工作：我们证明指数尾界。这些界适用于一般的哈希核，我们稍后将其应用于展示哈希如何使我们能够高效地进行大规模多任务学习。我们从一个关于哈希核的偏差和方差的简单引理开始。该引理的证明见附录 A。

**引理 2** 哈希核是无偏的，即 $E_\phi[\langle x, x' \rangle_\phi] = \langle x, x' \rangle$ 。此外，方差为 $\sigma^2_{x,x'} = \frac{1}{m}\left(\sum_{i \neq j} x_i^2 x_j'^2 + x_i x'_i x_j x'_j\right)$ ，因此，对于 $\|x\|_2 = \|x'\|_2 = 1$ ，有 $\sigma^2_{x,x'} = O\left(\frac{1}{m}\right)$ 。

这表明哈希核的典型值应集中在目标值的 $O(\frac{1}{\sqrt{m}})$ 范围内。我们使用切比雪夫不等式来证明一半的观测值落在 $\sqrt{2}\sigma$ 的范围内。这一点，再加上通过 [14] 的结果对 Talagrand 凸距离不等式的间接应用，使我们能够构造指数尾界。

### 3.1. 集中不等式界

在本小节中，我们证明在哈希特征映射下，每个向量的长度以高概率保持不变。Talagrand 不等式 [12] 是证明以下定理的关键工具（详见附录 B）。

**定理 3** 设 $\epsilon < 1$ 为固定常数， $x$ 为满足 $\|x\|_2 = 1$ 的给定实例。如果 $m \geq 72\log(1/\delta)/\epsilon^2$ 且 $\|x\|_\infty \leq \frac{\epsilon}{18\sqrt{\log(1/\delta)\log(m/\delta)}}$ ，则我们有

$$
\Pr\left[\left|\|x\|_\phi^2 - 1\right| \geq \epsilon\right] \leq 2\delta. \qquad (4)
$$

注意，类似的结果对 [18] 的原始哈希核也成立，唯一的修改是相关的偏差项。上述结果还可用于证明两个一般向量 $x$ 和 $x'$ 之间内积的集中界。

**推论 4** 对于两个向量 $x$ 和 $x'$ ，我们定义

$$
\sigma := \max(\sigma_{x,x}, \sigma_{x',x'}, \sigma_{x-x',x-x'})
$$

$$
\eta := \max\left(\frac{\|x\|_\infty}{\|x\|_2}, \frac{\|x'\|_\infty}{\|x'\|_2}, \frac{\|x-x'\|_\infty}{\|x-x'\|_2}\right).
$$

另外设 $\Delta = \|x\|_2 + \|x'\|_2 + \|x - x'\|_2$ 。如果 $m \geq \Omega\left(\frac{1}{\epsilon^2}\log(1/\delta)\right)$ 且 $\eta = O\left(\frac{\epsilon}{\log(m/\delta)}\right)$ ，则我们有

$$
\Pr\left[\left|\langle x, x'\rangle_\phi - \langle x, x'\rangle\right| > \epsilon\Delta/2\right] < \delta.
$$

该推论的证明见附录 C。我们还可以将定理 3 中的界扩展为向量之间大距离集合上的最大典范失真，如下所示：

**推论 5** 如果 $m \geq \Omega\left(\frac{1}{\epsilon^2}\log(n/\delta)\right)$ 且 $\eta = O\left(\frac{\epsilon}{\log(m/\delta)}\right)$ 。用 $X = \{x_1, \ldots, x_n\}$ 表示一组向量，对所有 $i, j$ 对满足 $\|x_i - x_j\|_\infty \leq \eta\|x_i - x_j\|_2$ 。在这种情况下，以概率 $1 - \delta$ ，我们对所有 $i, j$ 有

$$
\frac{\left|\|x_i - x_j\|_\phi^2 - \|x_i - x_j\|_2^2\right|}{\|x_i - x_j\|_2^2} \leq \epsilon.
$$

这意味着观测数量 $n$ （或相应地，未哈希核矩阵的大小）在分析中仅以对数形式出现。

**证明** 我们将定理 3 的界单独应用于每个距离。注意每个向量 $x_i - x_j$ 满足定理的条件，因此对于每个向量 $x_i - x_j$ ，我们以概率 $1 - \frac{\delta}{n^2}$ 将距离保持在 $(1 \pm \epsilon)$ 因子以内。对所有对取并集界即得结果。

### 3.2. 多重哈希

注意，推论 5 中并集界的紧性关键取决于 $\eta$ 的大小。换句话说，对于较大的 $\eta$ 值，即每当 $x$ 中的某些项非常大时，即使单次碰撞也已经可能导致嵌入的显著失真。这个问题可以通过用稀疏性换取方差来改善。一个单位长度向量可以写成 $(1, 0, 0, 0, \ldots)$ ，或 $\left(\frac{1}{\sqrt{2}}, \frac{1}{\sqrt{2}}, 0, \ldots\right)$ ，或更一般地写成具有 $c$ 个幅度为 $c^{-\frac{1}{2}}$ 的非零项的向量。这在某些情况下是相关的，例如当 $x$ 的幅度遵循已知模式时，比如将文档表示为词袋时，因为我们可以简单地对频繁词进行多次哈希。以下推论给出了置信界如何随复制次数缩放的直观说明：

**引理 6** 如果令 $x' = \frac{1}{\sqrt{c}}(x, \ldots, x)$ ，则：

1. 它是保范数的： $\|x\|_2 = \|x'\|_2$ 。
2. 它将分量幅度降低 $\frac{1}{\sqrt{c}} = \frac{\|x'\|_\infty}{\|x\|_\infty}$ 。
3. 方差增加至 $\sigma^2_{x',x'} = \frac{1}{c}\sigma^2_{x,x} + \frac{c-1}{c}2\|x\|_2^4$ 。

将引理 6 应用于定理 3，可以以增加方差为代价来降低大幅度值。

### 3.3. 近似正交性

对于多任务学习，我们必须为每个相关任务学习一个不同的参数向量。当映射到相同的哈希特征空间时，我们希望确保不同参数向量之间几乎没有交互。设 $U$ 为不同任务的集合， $u \in U$ 为其中一个特定任务。设 $w$ 为 $U \setminus \{u\}$ 中任务的参数向量的组合。我们证明，对于任务 $u$ 的任何观测 $x$ ， $w$ 与 $x$ 在哈希特征空间中的交互是最小的。对每个 $x$ ，设 $x$ 在任务 $u$ 的哈希特征映射下的像记为 $\phi_u(x) = \phi^{(\xi,h)}((x, u))$ 。

**定理 7** 设 $w \in \mathbb{R}^m$ 为 $U \setminus \{u\}$ 中任务的参数向量。在这种情况下，内积 $\langle w, \phi_u(x) \rangle$ 的值受下式约束

$$
\Pr\left\{|\langle w, \phi_u(x)\rangle| > \epsilon\right\} \leq 2e^{-\frac{\epsilon^2/2}{m^{-1}\|w\|_2^2\|x\|_2^2 + \epsilon\|w\|_\infty\|x\|_\infty/3}}
$$

**证明** 我们使用 Bernstein 不等式 [6]，该不等式指出，对于独立随机变量 $X_j$ ，若 $E[X_j] = 0$ ，且 $C > 0$ 使得 $|X_j| \leq C$ ，则

$$
\Pr\left[\sum_{j=1}^n X_j > t\right] \leq \exp\left(-\frac{t^2/2}{\sum_{j=1}^n E\left[X_j^2\right] + Ct/3}\right). \qquad (5)
$$

我们需要计算 $\langle w, \phi_u(x)\rangle = \sum_j x_j \xi(j) w_{h(j)}$ 的集中性质。设 $X_j = x_j \xi(j) w_{h(j)}$ 。根据 $h$ 和 $\xi$ 的定义， $X_j$ 是独立的。此外，对每个 $j$ ，由于 $w$ 仅依赖于 $U \setminus \{u\}$ 的哈希函数， $w_{h(j)}$ 与 $\xi(j)$ 独立。因此， $E[X_j] = E_{(\xi,h)}\left[x_j \xi(j) w_{h(j)}\right] = 0$ 。对每个 $j$ ，我们还有 $|X_j| < \|x\|_\infty \|w\|_\infty =: C$ 。最后， $\sum_j E[X_j^2]$ 由下式给出

$$
E\left[\sum_j \left(x_j \xi(j) w_{h(j)}\right)^2\right] = \frac{1}{m}\sum_{j,\ell} x_j^2 w_\ell^2 = \frac{1}{m}\|x\|_2^2 \|w\|_2^2
$$

将这两项和 $C$ 代入 Bernstein 不等式 (5) 即得结论。

定理 7 约束了不相关任务对任何特定实例的影响。在第 5 节中，我们通过大规模多任务学习问题上的实验结果展示了其在现实世界中的适用性。


## 4. 应用

特征哈希的优势在于它允许对参数向量进行显著的存储压缩：当 $w \in \mathbb{R}^d$ 时，在原始特征空间存储 $w$ 天真地需要 $O(d)$ 个数。通过哈希，我们能够将其减少到 $O(m)$ 个数，同时避免局部敏感哈希（Locally Sensitive Hashing）中常见的昂贵矩阵-向量乘法。此外，所得向量的稀疏性得以保留。

哈希技巧的好处使其在机器学习及其之外的几乎所有领域都有应用。特别是，当需要在有限内存容量内存储大量带冗余的参数时，特征哈希极其有用。

**个性化** 特征哈希的一个强大应用见于多任务学习。定理 7 允许我们将不同任务的多个分类器哈希到一个特征空间中，且几乎没有交互。为了说明这一点，我们在垃圾邮件分类器个性化的背景下探索这种设置。

假设我们有数千个用户 $U$ ，并希望为他们中的每一个执行相关但不完全相同的分类任务。用户通过将电子邮件标记为垃圾邮件或非垃圾邮件来提供标注数据。理想情况下，对于每个用户 $u \in U$ ，我们希望仅基于该用户的数据学习一个预测器 $w_u$ 。然而，网络邮箱用户在标注电子邮件方面出了名的懒惰，甚至那些不贡献训练数据的用户也期望有一个能用的垃圾邮件过滤器。因此，我们还需要学习一个额外的全局预测器 $w_0$ ，以允许在所有用户之间共享数据。

存储所有预测器 $w_i$ 需要 $O(d \times (|U| + 1))$ 的内存。在协同垃圾邮件过滤这样的任务中， $|U|$ （用户数量）可能达到数十万，词表大小通常为数百万量级。处理这个问题的朴素方法是消除所有不频繁的 token。然而，垃圾邮件发送者会针对这种内存脆弱性，通过恶意拼错单词，从而创造出高度不频繁但具有垃圾邮件典型特征的 token，使其"逃过"传统分类器的"雷达"。相反，如果所有单词都被哈希到一个固定大小的特征向量中，不频繁但具有类别指示性的 token 就有机会对分类结果做出贡献。此外，大规模垃圾邮件过滤器（例如 Yahoo Mail™ 或 GMail™）通常有严格的内存和时间限制，因为它们每天必须处理数十亿封电子邮件。为了保证有限大小的内存占用，我们用不同的哈希函数 $\phi_0, \ldots, \phi_{|U|}$ 将所有权重向量 $w_0, \ldots, w_{|U|}$ 哈希到一个联合的、显著更小的特征空间 $\mathbb{R}^m$ 中。所得的哈希权重向量 $w_h \in \mathbb{R}^m$ 可以写成：

$$
w_h = \phi_0(w_0) + \sum_{u \in U} \phi_u(w_u). \qquad (6)
$$

注意，在实践中权重向量 $w_h$ 可以直接在哈希空间中学习。所有未哈希的权重向量永远不需要被计算。给定用户 $u \in U$ 的新文档/电子邮件 $x$ ，预测任务现在包括计算 $\langle \phi_0(x) + \phi_u(x), w_h \rangle$ 。由于哈希，我们有两个误差来源——哈希内积的失真 $\epsilon_d$ 和与其他哈希权重向量的干扰 $\epsilon_i$ 。更精确地说：

$$
\langle \phi_0(x) + \phi_u(x), w_h \rangle = \langle x, w_0 + w_u \rangle + \epsilon_d + \epsilon_i. \qquad (7)
$$

干扰误差由 $\phi_0(x)$ 或 $\phi_u(x)$ 与其他用户的哈希函数之间的所有碰撞组成，

$$
\epsilon_i = \sum_{v \in U, v \neq 0} \langle \phi_0(x), \phi_v(w_v) \rangle + \sum_{v \in U, v \neq u} \langle \phi_u(x), \phi_v(w_v) \rangle. \qquad (8)
$$

为了证明 $\epsilon_i$ 以高概率很小，我们可以两次应用定理 7，对式 (8) 的每一项各应用一次。我们将每个用户的分类视为一个单独的任务，并且由于 $\sum_{v \in U, v \neq 0} w_v$ 独立于哈希函数 $\phi_0$ ，定理 7 的条件在 $w = \sum_{v \neq 0} w_v$ 时成立，我们可以用它来控制第二项 $\sum_{v \in U, v \neq 0} \langle \phi_u(x), \phi_u(w_v) \rangle$ 。第二次应用是相同的，只是将所有下标 "0" 替换为 "u"。由于篇幅限制，我们不推导确切的界。

失真误差的出现是因为用户 $u$ 所使用的每个哈希函数都可能发生自碰撞：

$$
\epsilon_d = \sum_{v \in \{u, 0\}} \left| \langle\phi_v(x), \phi_v(w_v)\rangle - \langle x, w_v\rangle \right|. \qquad (9)
$$

为了证明 $\epsilon_d$ 以高概率很小，我们对 $v$ 的每个可能值应用一次推论 4。

在第 5 节中，我们展示了该设置的实验结果。经验结果比本小节推导的理论界更强——我们的技术在数十万用户上优于单一的全局分类器。我们在第 5 节讨论一个直观的解释。

**大规模多类别估计** 我们还可以将大规模多类别分类视为一个多任务问题，并以类似于个性化设置的方式应用特征哈希。我们不是为每个用户使用不同的哈希函数，而是为每个类别使用不同的哈希函数。

[18] 将特征哈希应用于具有大量类别的问题。他们经验性地表明，对于具有数百万特征和数千类别的问题，可以高效地实现特征向量 $\phi(x, y)$ 的联合哈希。

**协同过滤** 假设我们有一个非常大的稀疏矩阵 $M$ ，其中条目 $M_{ij}$ 表示用户 $i$ 对实例 $j$ 采取的操作。操作和实例的一个常见示例是用户对电影的评分 [5]。一种成功的在用户和实例之间寻找共同因子以预测未观察操作的方法是将 $M$ 分解为 $M = U^\top W$ 。如果我们有数百万用户执行数百万操作，将 $U$ 和 $W$ 存储在内存中很快就会变得不可行。相反，我们可以选择使用哈希来压缩矩阵 $U$ 和 $W$ 。对于 $U, W \in \mathbb{R}^{n \times d}$ ，用 $u, w \in \mathbb{R}^m$ 表示满足下式的向量

$$
u_i = \sum_{j,k:h(j,k)=i} \xi(j, k) U_{jk} \quad \text{and} \quad w_i = \sum_{j,k:h'(j,k)=i} \xi'(j, k) W_{jk}.
$$

其中 $(h, \xi)$ 和 $(h', \xi^{\prime})$ 是独立选择的哈希函数。这允许我们通过下式近似矩阵元素 $M_{ij} = [U^\top W]_{ij}$

$$
M^\phi_{ij} := \sum_{k} \xi(k, i)\xi'(k, j) u_{h(k,i)} w_{h'(k,j)}.
$$

这给出了 $M$ 的一种可以高效存储的压缩向量表示。


## 5. 结果

我们在个性化设置中评估了我们的算法。作为数据集，我们使用了一个专有的电子邮件垃圾邮件分类任务，包含 $n = 320$ 万封电子邮件，经过适当匿名化，收集自 $|U| = 433167$ 个用户。每封电子邮件由 $U$ 中的一个用户标记为垃圾邮件或非垃圾邮件。分词后，数据集包含 4000 万个唯一单词。

对于本文中的所有实验，我们使用了 Vowpal Wabbit 实现的平方损失随机梯度下降[^2]。在邮件垃圾邮件文献中，将非垃圾邮件误分类被认为比将垃圾邮件误分类危害大得多。因此，我们遵循惯例，在测试时设置分类阈值，使得恰好 1% 的非垃圾邮件测试数据被分类为垃圾邮件。我们的个性化哈希函数的实现如图 1 所示。为了获得用户 $u$ 的个性化哈希函数 $\phi_u$ ，我们将唯一的用户 ID 连接到电子邮件中的每个单词，然后使用相同的全局哈希函数对新生成的 token 进行哈希。

[^2]: http://hunch.net/~vw/

<img src="/Users/dazhang/PycharmProject/Papers/3-RecSys/.picture/image-20260721000000001.png" alt="image-20260721000000001" style="zoom:50%;" />

图 1. 以示意图布局总结的哈希个性化。每个 token 被复制，其中一份被个性化（例如通过将每个单词与唯一的用户标识符连接）。然后，全局哈希函数将所有 token 映射到低维特征空间中，在那里对文档进行分类。

数据集收集时间跨度为 14 天。我们使用前 10 天进行训练，剩余 4 天进行测试。作为基线，我们选择了在所有用户上训练并哈希到 $2^{26}$ 维空间的纯全局分类器。由于 $2^{26}$ 远远超过唯一单词的总数，我们可以将基线视为代表无哈希的分类。所有结果均报告为相对于此基线的未被检测到的垃圾邮件数量（例如，值 0.80 表示该用户的垃圾邮件减少 20%）[^3]。

[^3]: 作为我们数据共享协议的一部分，我们同意不包含绝对的分类错误率。

<img src="/Users/dazhang/PycharmProject/Papers/3-RecSys/.picture/image-20260721000000002.png" alt="image-20260721000000002" style="zoom:50%;" />

图 2. 相对于基线分类器、对所有用户平均的未捕获垃圾邮件的下降。分类阈值的选择使非垃圾邮件误分类固定为 1%。哈希全局分类器（global-hashed）相对较快地收敛，表明失真误差 $\epsilon_d$ 消失。个性化分类器带来了平均高达 30% 的改善。

图 2 显示了用户收件箱中垃圾邮件的平均数量作为哈希键数 $m$ 的函数，相对于上述基线。除基线外，我们还评估了两种不同的设置。global-hashed 曲线表示哈希后全局分类器的相对垃圾邮件捕获率 $\langle\phi_0(w_0), \phi_0(x)\rangle$ 。在 $m = 2^{26}$ 时，这与基线相同。在 $m = 2^{22}$ 时的早期收敛表明，此时哈希碰撞对分类误差没有影响，基线确实等同于不进行哈希所能获得的结果。

在个性化设置中，每个用户 $u \in U$ 都获得自己的分类器 $\phi_u(w_u)$ 以及全局分类器 $\phi_0(w_0)$ 。如果不进行哈希，特征空间会爆炸，因为 $u = 40$ 万用户和 $n = 4000$ 万 token 的叉乘会产生 16 万亿个可能的唯一个性化特征。图 2 显示，尽管进行了激进的哈希，一旦哈希表由 22 位索引，个性化就能减少 30% 的垃圾邮件。

<img src="/Users/dazhang/PycharmProject/Papers/3-RecSys/.picture/image-20260721000000003.png" alt="image-20260721000000003" style="zoom:50%;" />

图 3. 按训练电子邮件数量聚类的用户的结果。例如，桶 $[8, 15]$ 由所有具有八到十五封训练电子邮件的用户组成。尽管拥有大量训练数据的桶中的用户确实从个性化分类器中受益更多（垃圾邮件减少高达 65%），但即使是完全没有为训练语料做出贡献的用户也获得了近 20% 的垃圾邮件减少。

**用户聚类** 图 2 中强结果的一个假设可能源于用户投票的非均匀分布——有可能使用个性化和特征哈希使我们受益于少数标注了许多电子邮件的用户，同时在此过程中降低了大多数用户（标注很少或没有标注电子邮件的用户）的性能。事实上，在现实生活中，大部分电子邮件用户根本不为训练语料做出贡献，只在测试时与分类器交互。测试电子邮件的个性化版本 $\Phi_u(x_u)$ 随后被哈希到其他 token 的桶中，只会给分类增加干扰噪声 $\epsilon_i$ 。

为了证明我们改善了大多数用户的性能，因此重要的是，我们不仅报告所有电子邮件的平均结果，而且明确考察个性化分类器对取决于其对训练集贡献的用户的影响。为此，我们根据用户的训练电子邮件数量将用户放入指数增长的桶中，并分别计算每个桶的未捕获垃圾邮件的相对减少量。图 3 显示了每个桶的结果。我们不与没有全局组件的纯局部方法进行比较，因为对于大部分用户——那些没有训练数据的用户——这种方法无法超越随机猜测。

没有或很少有训练电子邮件的桶中的用户（桶 $[0]$ 的线与桶 $[1]$ 相同）也从个性化中受益，这可能看起来相当令人惊讶。毕竟，他们的个性化分类器从未被训练过，在测试时只能增加噪声。这个桶的分类器改进可以用垃圾邮件和非垃圾邮件的主观定义来解释。在个性化设置中，用户标注的个体成分被局部分类器吸收，而全局分类器代表垃圾邮件和非垃圾邮件的共同定义。换句话说，个性化分类器的全局部分获得了更好的泛化性能，使所有用户受益。


## 6. 相关工作

许多研究者已经解决了相关但不同的问题。

[16] 使用 Bochner 定理和采样来获得径向基函数核的近似内积。[17] 将此扩展到基函数加权组合的稀疏近似。这对许多函数空间来说计算上是高效的。注意，该表示是稠密的。[13] 采取了互补的方法：对于稀疏特征向量 $\phi(x)$ ，他们设计了一种进一步减少非零项数量的方案。虽然这在原则上是可取的，但它并没有解决 $\phi(x)$ 高维的问题。更简明地说，有必要用对偶表示来表达函数，而不是将 $f$ 表示为线性函数，其中 $w$ 不太可能被紧凑表示： $f(x) = \langle\phi(x), w\rangle$ 。

[1] 为降维提供了计算高效的随机化方案。[10] 的算法需要通过稠密的 $d \cdot m$ 维矩阵向量乘法将维度为 $d$ 的向量降至维度 $m$ ，而他通过设计一个仅由 $\{-1, 0, 1\}$ 条目组成的矩阵，只需要其 $\frac{1}{3}$ 的计算量。由 [2] 开创，已有一系列工作 [3, 15] 通过使用各种编码矩阵对输入向量进行预处理来改进随机投影的复杂性。我们的一些理论界可以从 [14] 的界推导出来。

一个相关的构造是 [7] 的 CountMin 草图，它在哈希表的多个副本中存储计数。这为范围查询和点查询产生了良好的集中不等式。

[18] 提出了一个哈希核，用一个非常简单的算法来处理计算效率问题：通过将所有具有相同哈希值的坐标相加来压缩高维向量——只需要执行与向量中非零项一样多的计算。与局部敏感哈希 [1, 10] 相比，这节省了大量计算。

还有几项工作为研究哈希表示提供了动机。例如，[9] 提供了经验证据，表明哈希技巧可以通过移除字典，在许多稀疏学习问题上有效地将内存占用减少一个数量级。我们的实验结果验证了这一点，并表明可以实现更激进的压缩水平。此外，[11] 发布了 Vowpal Wabbit 快速在线学习软件，它使用了与本文讨论的类似的哈希表示。


## 7. 结论

在本文中，我们从理论和经验上分析了**用于降维的哈希技巧**。作为理论分析的一部分，我们引入了无偏哈希函数，并为哈希核提供了指数尾界。这些结果让我们进一步洞察哈希空间，并解释了先前得到的经验观察。我们还推导出哈希空间的随机子空间很可能不相互作用，这使得具有许多任务的多任务学习成为可能。我们的实验结果在垃圾邮件过滤背景下的一个真实世界应用上验证了这一点。在这里，我们证明即使有非常多的任务和特征，全部映射到一个联合的低维哈希空间中，也可以在有限内存保证下获得令人印象深刻的分类结果。


## 参考文献

[1] Achlioptas, D. (2003). Database-friendly random projections: Johnson-lindenstrauss with binary coins. Journal of Computer and System Sciences, 66, 671–687.

[2] Ailon, N., & Chazelle, B. (2006). Approximate nearest neighbors and the fast Johnson-Lindenstrauss transform. Proc. 38th Annual ACM Symposium on Theory of Computing (pp. 557–563).

[3] Ailon, N., & Liberty, E. (2008). Fast dimension reduction using Rademacher series on dual BCH codes. Proc. 19th Annual ACM-SIAM Symposium on Discrete algorithms (pp. 1–9).

[4] Alon, N. (2003). Problems and results in extremal combinatorics, Part I. Discrete Math, 273, 31–53.

[5] Bennett, J., & Lanning, S. The Netflix Prize. Proceedings of KDD Cup and Workshop 2007.

[6] Bernstein, S. (1946). The theory of probabilities. Moscow: Gastehizdat Publishing House.

[7] Cormode, G., & Muthukrishnan, M. (2004). An improved data stream summary: The count-min sketch and its applications. LATIN: Latin American Symposium on Theoretical Informatics.

[8] Dasgupta, A., Sarlos, T., & Kumar, R. (2010). A Sparse Johnson Lindenstrauss Transform. Submitted.

[9] Ganchev, K., & Dredze, M. (2008). Small statistical models by random feature mixing. Workshop on Mobile Language Processing, Annual Meeting of the Association for Computational Linguistics.

[10] Gionis, A., Indyk, P., & Motwani, R. (1999). Similarity search in high dimensions via hashing. Proceedings of the 25th VLDB Conference (pp. 518–529). Edinburgh, Scotland: Morgan Kaufmann.

[11] Langford, J., Li, L., & Strehl, A. (2007). Vowpal wabbit online learning project (Technical Report). http://hunch.net/?p=309.

[12] Ledoux, M. (2001). The concentration of measure phenomenon. Providence, RI: AMS.

[13] Li, P., Church, K., & Hastie, T. (2007). Conditional random sampling: A sketch-based sampling technique for sparse data. In B. Schölkopf, J. Platt and T. Hoffman (Eds.), Advances in neural information processing systems 19, 873–880. Cambridge, MA: MIT Press.

[14] Liberty, E., Ailon, N., & Singer, A. (2008). Dense fast random projections and lean Walsh transforms. Proc. 12th International Workshop on Randomization and Approximation Techniques in Computer Science (pp. 512–522).

[15] Matousek, J. (2008). On variants of the Johnson–Lindenstrauss lemma. Random Structures and Algorithms, 33, 142–156.

[16] Rahimi, A., & Recht, B. (2008). Random features for large-scale kernel machines. In J. Platt, D. Koller, Y. Singer and S. Roweis (Eds.), Advances in neural information processing systems 20. Cambridge, MA: MIT Press.

[17] Rahimi, A., & Recht, B. (2009). Randomized kitchen sinks. In L. Bottou, Y. Bengio, D. Schuurmans and D. Koller (Eds.), Advances in neural information processing systems 21. Cambridge, MA: MIT Press.

[18] Shi, Q., Petterson, J., Dror, G., Langford, J., Smola, A., Strehl, A., & Vishwanathan, V. (2009). Hash kernels. AISTATS 12.

[19] Weinberger, K., Dasgupta, A., Attenberg, J., Langford, J., & Smola, A. (2009). Feature hashing for large scale multitask learning. 26th International Conference on Machine Learning (p. 140).


## 附录

### A. 均值与方差

**证明**［引理 2］为了计算期望，我们展开

$$
\langle x, x' \rangle_\phi = \sum_{i,j} \xi(i)\xi(j) x_i x'_j \delta_{h(i),h(j)}. \qquad (10)
$$

由于 $E_\phi[\langle x, x'\rangle_\phi] = E_h[E_\xi[\langle x, x'\rangle_\phi]]$ ，对 $\xi$ 取期望，我们看到只有 $i = j$ 的项具有非零值，这证明了第一个论断。对于方差，我们计算 $E_\phi[\langle x, x'\rangle_\phi^2]$ 。将其展开，得到：

$$
\langle x, x'\rangle_\phi^2 = \sum_{i,j,k,l} \xi(i)\xi(j)\xi(k)\xi(l) x_i x'_j x_k x'_l \delta_{h(i),h(j)}\delta_{h(k),h(l)}.
$$

该表达式可以通过注意到下式来简化：

$$
E_\xi[\xi(i)\xi(j)\xi(k)\xi(l)] = \delta_{ij}\delta_{kl} + [1 - \delta_{ijkl}](\delta_{ik}\delta_{jl} + \delta_{il}\delta_{jk}).
$$

将对 $\xi$ 的期望传入求和，这使我们可以将方差的展开分解为两项。

$$
E_\phi[\langle x, x'\rangle_\phi^2] = \sum_{i,k} x_i x'_i x_k x'_k + \sum_{i \neq j} x_i^2 x_j'^2 E_h\left[\delta_{h(i),h(j)}\right] + \sum_{i \neq j} x_i x'_i x_j x'_j E_h\left[\delta_{h(i),h(j)}\right]
$$

$$
= \langle x, x'\rangle^2 + \frac{1}{m}\left(\sum_{i \neq j} x_i^2 x_j'^2 + \sum_{i \neq j} x_i x'_i x_j x'_j\right)
$$

其中注意到对 $i \neq j$ 有 $E_h\left[\delta_{h(i),h(j)}\right] = \frac{1}{m}$ 。利用 $\sigma^2 = E_\phi[\langle x, x'\rangle_\phi^2] - E_\phi[\langle x, x'\rangle_\phi]^2$ 即证得结论。

### B. 集中不等式

我们使用 Liberty、Ailon 和 Singer 在 [14] 中推导的集中结果。Liberty 等人通过将一个精心构造的确定性矩阵 $A$ 与随机对角矩阵组合，创建了 Johnson-Lindenstrauss 随机投影矩阵。为完整性起见，我们重述相关引理。设 $i$ 遍历哈希桶。设 $m = c\log(1/\delta)/\epsilon^2$ ，其中 $c$ 为足够大的常数。对于给定的向量 $x$ ，定义对角矩阵 $D_x$ 为 $(D_x)_{jj} = x_j$ 。对于任意矩阵 $A \in \mathbb{R}^{m \times d}$ ，定义 $\|x\|_A \equiv \max_{y:\|y\|_2=1} \|AD_x y\|_2$ 。

**引理 2（Liberty 等人，2008 [14]）。** 对于任何列归一化矩阵 $A$ 、满足 $\|x\|_2 = 1$ 的向量 $x$ 以及独立同分布的随机 $\pm 1$ 对角矩阵 $D_s$ ，以下成立： $\forall x$ ，如果 $\|x\|_A \leq \frac{\epsilon}{6\sqrt{\log(1/\delta)}}$ ，则 $\Pr\left[|\|AD_s x\|^2 - 1| > \epsilon\right] \leq \delta$ 。

我们还需要以下形式的加权球与桶不等式——该引理的陈述及证明遵循 [8] 的引理 6。由于某些参数值不同，我们仍概述其证明。

**引理 8** 设 $m$ 为哈希函数值域的大小，设 $\eta = \frac{1}{2\sqrt{m\log(m/\delta)}}$ 。如果 $x$ 满足 $\|x\|_2 = 1$ 且 $\|x\|_\infty \leq \eta$ ，则定义 $\sigma_*^2 = \max_i \sum_{j=1}^d x_j^2 \delta_{ih(j)}$ ，其中 $i$ 遍历所有哈希桶。我们有，以概率 $1 - \delta$ ，

$$
\sigma_*^2 \leq \frac{2}{m}
$$

**证明** 我们概述证明步骤。由于这些桶具有相同的分布，我们只考察第 1 个桶，即 $i = 1$ 的情况，并约束 $\sum_{j:h(j)=1} x_j^2$ 。定义 $X_j = x_j^2\left(\delta_{1h(j)} - \frac{1}{m}\right)$ 。则 $E_h[X_j] = 0$ 且 $E_h[X_j^2] = x_j^4\left(\frac{1}{m} - \frac{1}{m^2}\right) \leq \frac{x_j^4}{m} \leq \frac{x_j^2\eta^2}{m}$ ，其中使用了 $\|x\|_\infty \leq \eta$ 。因此 $\sum_j E_h[X_j^2] \leq \frac{\eta^2}{m}$ 。还要注意 $\sum_j X_j = \sum_{j:h(j)=1} x_j^2 - \frac{1}{m}$ 。将其代入 Bernstein 不等式，即式 (5)，我们有

$$
\Pr\left[\sum_j X_j > \frac{1}{m}\right] \leq \exp\left(-\frac{1/2m^2}{\eta^2/m + \eta^2/3m}\right) = \exp\left(-\frac{3}{8m\eta^2}\right) \leq \exp(-\log(m/\delta)) \leq \delta/m
$$

对所有 $m$ 个桶取并集界，我们得到上述结果。

**证明**［定理 3］给定函数 $\phi = (h, r)$ ，定义矩阵 $A$ 为 $A_{ij} = \delta_{ih(j)}$ ， $D_s$ 为 $(D_s)_{jj} = r_j$ 。设 $x$ 满足给定条件，即 $\|x\|_2 = 1$ 且 $\|x\|_\infty \leq \eta$ 。注意 $\|x\|_\phi = \|AD_s x\|_2$ 。设 $y \in \mathbb{R}^d$ 满足 $\|y\|_2 = 1$ 。因此

$$
\|AD_x y\|_2^2 = \sum_{i=1}^m \left(\sum_{j=1}^d y_j \delta_{ih(j)} x_j\right)^2 \leq \sum_{i=1}^m \left(\sum_{j=1}^d y_j^2 \delta_{ih(j)}\right)\left(\sum_{j=1}^d x_j^2 \delta_{ih(j)}\right) \leq \sum_{i=1}^m \left(\sum_{j=1}^d y_j^2 \delta_{ih(j)}\right)\sigma_*^2 \leq \sigma_*^2.
$$

上式通过应用柯西-施瓦茨不等式，并使用 $\sigma_*$ 的定义得到。因此， $\|x\|_A = \max_{y:\|y\|_2=1} \|AD_x y\|_2 \leq \sigma_* \leq \sqrt{2}m^{-1/2}$ 。

如果 $m \geq \frac{72}{\epsilon^2}\log(1/\delta)$ ，我们有 $\|x\|_A \leq \frac{\epsilon}{6\sqrt{\log(1/\delta)}}$ ，这满足 [14] 引理 2 的条件。因此将 [14] 引理 2 的上述结果应用于 $x$ ，并使用引理 8，我们有 $\Pr[|\|AD_s x\|_2^2 - 1| \geq \epsilon] \leq \delta$ ，因此

$$
\Pr\left[|\|x\|_\phi^2 - 1| \geq \epsilon\right] \leq \delta
$$

对引理 2 和引理 8 的两个错误概率取并集，我们得到结果。

### C. 内积

**证明**［推论 4］我们有 $2\langle x, x'\rangle_\phi = \|x\|_\phi^2 + \|x'\|_\phi^2 - \|x - x'\|_\phi^2$ 。取期望，我们得到标准的内积不等式。因此，

$$
|2\langle x, x'\rangle_\phi - 2\langle x, x'\rangle| \leq \left|\|x\|_\phi^2 - \|x\|_2^2\right| + \left|\|x'\|_\phi^2 - \|x'\|_2^2\right| + \left|\|x - x'\|_\phi^2 - \|x - x'\|_2^2\right|
$$

使用并集界，以概率 $1 - 3\delta$ ，上述每一项都由定理 3 约束。因此，将这些界放在一起，我们有，以概率 $1 - 3\delta$ ，

$$
|2\langle\phi_u(x), \phi_u(x)\rangle - 2\langle x, x\rangle| \leq \epsilon(\|x\|_2 + \|x'\|_2 + \|x - x'\|_2)
$$

### D. 对先前错误证明的反驳

本文的前一个版本 [19] 中存在几个错误。我们现在详细说明每一个错误并解释为什么它是错误的。当前的结果表明，使用哈希我们可以创建一个投影矩阵，对于具有有界 $\|x\|_\infty/\|x\|_2$ 比率的向量，可以将距离保持在 $(1 \pm \epsilon)$ 因子以内。对输入向量的约束可以通过多重哈希来规避，如第 3.2 节所述，但这需要哈希 $O(\frac{1}{\epsilon^2})$ 次。最近的工作 [8] 表明，对于这一构造可以证明更好的理论界。我们感谢 Tamas Sarlos 和 Ravi Kumar 就这些错误撰写的以下说明，并感谢他们提出了附录 B 中的新证明。

1. Weinberger 等人（[19]，定理 3）的主要定理的陈述是错误的，因为它与 Alon（[4]）的下界相矛盾。缺陷在于（[19]，定理 3）中的错误概率，其被声称是 $\exp(-\frac{\sqrt{\epsilon}}{4\eta})$ 。这个错误可以不通过增加嵌入维度 $m$ 、而通过减小 $\eta = \frac{\|x\|_\infty}{\|x\|_2}$ 来任意缩小，而这又可以通过预处理输入向量 $x$ 来实现。然而，这与 Alon 关于嵌入维度的下界相矛盾。这一矛盾的细节最好通过（[19]，推论 5）来呈现，如下所示。

   设 $m = 128$ 且 $\delta = 1/2$ ，考虑 $\mathbb{R}^{n+1}$ 中 $n$ 维单纯形的顶点，即 $x_1 = (1, 0, ..., 0)$ ， $x_2 = (0, 1, 0, ..., 0)$ ，……。设 $P \in \mathbb{R}^{(n+1)c \times (n+1)}$ 为朴素的、基于复制的预处理器，复制参数 $c = 512\log^2 n$ ，如我们投稿的第 2 节或（[19]，第 3.2 节）所定义。

   因此，对所有 $i \neq j$ 对，我们有 $\|Px_i - Px_j\|_\infty = 1/\sqrt{c}$ 且 $\|Px_i - Px_j\|_2 = \sqrt{2}$ 。因此我们可以将（[19]，推论 5）应用于向量集合 $Px_i$ ，其中 $\eta = 1/\sqrt{2c} = 1/(32\log n)$ ；则声称的近似误差为 $\sqrt{\frac{2}{m}} + \frac{64\eta^2\log^2 n}{2\delta} = \frac{1}{8} + \frac{1}{16} \leq \frac{1}{4}$ 。如果推论 5 为真，那么可以得出：以至少 $1/2$ 的概率，线性变换 $A = \phi \cdot P: \mathbb{R}^{n+1} \rightarrow \mathbb{R}^m$ 将上述 $n + 1$ 个向量的成对距离最多扭曲 $1 \pm 1/4$ 的乘性因子。另一方面，Alon 的下界表明，任何这样的变换 $A$ 必须映射到 $\Omega(\log n)$ 维；参见（[4]）中定理 9.3 之后的评论，并在那里设 $\epsilon = 1/4$ 。这显然与上面的 $m = 128$ 相矛盾。

2. 定理 3 的证明包含一个致命的、无法修复的错误。回顾 $\delta_{ij}$ 表示通常的 Kronecker 符号， $h$ 和 $h'$ 是哈希函数。Weinberger 等人在附录 B 第 8 页证明的式 (13) 之后做了以下观察：

   "首先注意 $\sum_i \sum_j \delta_{h(j)i} + \delta_{h'(j)i}$ 至多为 $2t$ ，其中 $t = |\{j: h(j) \neq h'(j)\}|$ 。"

   引用的观察是错误的。设 $d$ 表示输入的维度。那么， $\sum_i \sum_j \delta_{h(j)i} + \delta_{h'(j)i} = \sum_j\left(\sum_i \delta_{h(j)i} + \delta_{h'(j)i}\right) = \sum_j 2 = 2d$ ，与哈希函数的选择无关。注意， $t$ 在（[19]）的证明中起了关键作用，它将降维的欧几里得近似误差与定义在哈希函数集合上的 Talagrand 凸距离联系起来。尽管这个错误是初等的，但我们看不出如何修复其在（[19]）中的后果，即使该论断具有正确的形式。

3. （[19]）中定理 3 的证明还包含一个次要的、可修复的错误。为了看到这一点，考虑（[19]）中定理 3 证明末尾的句子，其中 $0 < \epsilon < 1$ 且 $\beta = \beta(x) \geq 1$ 。

   "注意到 $s^2 = (\sqrt{\beta^2 + \epsilon} - \beta)/4\|x\|_\infty \geq \sqrt{\epsilon}/4\|x\|_\infty$ ，……"

   在这里，作者错误地假设 $\sqrt{\beta^2 + \epsilon} - \beta \geq \sqrt{\epsilon}$ 成立，而事实是 $\sqrt{\beta^2 + \epsilon} - \beta \leq \sqrt{\epsilon}$ 总是成立。

   观察到这个小故障很容易在局部修复，然而这种修改是次要的，修改后的论断仍然是错误的。

   由于对所有 $0 \leq y \leq 1$ 我们有 $\sqrt{1 + y} \geq 1 + y/3$ ，从 $\beta \geq 1$ 可以得出 $\sqrt{\beta^2 + \epsilon} - \beta \geq \epsilon/3$ 。将后一个估计代入定理 3 的"证明"将产生一个修改后的论断，其中原来的错误概率 $\exp(-\frac{\sqrt{\epsilon}}{4\eta})$ 被替换为 $\exp(-\frac{\epsilon}{12\eta})$ 。更新本说明第一部分中的数值常数将表明，新论断仍然与 Alon 的下界相矛盾。为了说明这一点，观察到反例是基于常数 $\epsilon$ 的，而修改后的论断在其目标维度中仍然缺少必要的 $\Omega(\log n)$ 依赖性。
