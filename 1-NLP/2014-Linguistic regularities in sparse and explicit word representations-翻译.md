# 稀疏和显式词表示中的语言规律（Linguistic Regularities in Sparse and Explicit Word Representations）

> Omer Levy, Yoav Goldberg | 巴伊兰大学计算机科学系 | CoNLL 2014

本文证明 Mikolov 等人的向量算术类比方法（先加后减再搜索）等价于最大化三个两两词相似度的线性组合，并提出改进目标函数 3COSMUL。核心发现是——**关系相似度的恢复并非神经词嵌入的专利，传统分布式（显式）词表示也能恢复同等数量的关系相似度**。

核心内容：

- 将 3COSADD 目标分解为"两个吸引项 + 一个排斥项"：寻找与 $b$ 和 $a^{*}$ 相似但与 $a$ 不同的词
- 揭示加法目标存在"软或"行为——单个足够大的项会主导表达式，例如地理/文化方面会压倒王室方面的信号
- 提出 3COSMUL 乘法目标：对每项取对数后求和，放大小数量差异、缩小大数量差异，从而平衡不同相似度方面
- 显式表示用 PPMI 加权的线性上下文（窗口 2+2，区分位置），神经表示用 word2vec skip-gram（600 维、NEG-15）
- 在 MSR（8000 题）、GOOGLE（19544 题）开放词汇与 SEMEVAL 封闭词汇任务上对比两种表示

关键发现：

- 3COSMUL 在 GOOGLE 上显式表示达 **68.24%**、嵌入表示 66.72%，均超过 3COSADD 的 45.05%/62.70%（绝对提升 20+ 个百分点）
- MSR 上显式 56.83% vs 嵌入 59.09%，两者的差距从 3COSADD 的 25 个百分点骤降到约 2 个百分点
- SEMEVAL 上两种表示准确率几乎相同（38.37% vs 38.67%）
- 神谕设置（任一表示答对即算对）下 MSR 达 71.9%、GOOGLE 达 77.8%，说明两种表示错误互补
- 词"Fresno"在嵌入表示中被默认返回 82 次作为 city-in-state 的错误答案，显式表示中高频功能词（who/and/be）成为"枢纽"混淆模型

---

## 摘要

最近的工作表明，神经嵌入词表示捕获了许多关系相似度，这些相似度可以通过嵌入空间中的向量算术恢复。我们证明 Mikolov 等人的方法——先加后减词向量，然后搜索与结果相似的词——等价于搜索最大化三个两两词相似度线性组合的词。基于这一观察，我们提出了一种恢复关系相似度的改进方法，在两个最近的词类比数据集上提高了最先进的成绩。此外，我们证明类比恢复并不局限于神经词嵌入，传统分布式词表示也能恢复同等数量的关系相似度。

## 1 引言

用于语言处理的深度学习方法很大程度上归功于神经网络语言模型，其中词被表示为 $\mathbb{R}^d$ 中的稠密实值向量。此类表示被称为分布式词表示或词嵌入，因为它们将整个词汇表嵌入到一个相对低维的线性空间中，其维度是潜在（latent）连续特征。嵌入词向量使用神经网络的变体在大量文本集合上训练（Bengio et al.，2003；Collobert and Weston，2008；Mnih and Hinton，2008；Mikolov et al.，2011；Mikolov et al.，2013b）[5, 9, 22, 17, 19]。[^1]

词嵌入旨在捕获 Turney（2006）[32] 所称的词汇 item 之间的归因相似度：出现在相似上下文中的词在投影空间中会彼此接近。其效果是将共享语义（"dog cat cow"、"eat devour"）或句法（"cars hats days"、"emptied carried danced"）属性的词分组，并且已被证明作为各种 NLP 任务的特征是有效的（Turian et al.，2010；Collobert et al.，2011；Socher et al.，2011；Al-Rfou et al.，2013）[28, 10, 27, 1]。我们将此类词表示称为神经嵌入或简称嵌入。

最近，Mikolov et al.（2013c）[20] 证明递归神经网络（RNN，Recurrent Neural Network）创建的嵌入不仅编码词之间的归因相似度，还编码词对之间的相似度。此类相似度被 Mikolov 等人称为语言规律，被 Turney（2006）[32] 称为关系相似度。它们捕获例如"man:woman"、"king:queen"词对展示的性别关系，"france:french"、"mexico:spanish"中的语言使用关系，以及"capture:captured"、"go:went"中的过去时关系。值得注意的是，Mikolov 等人表明此类关系反映在词对之间的向量偏移中（apples − apple ≈ cars − car），并且通过使用简单的向量算术可以应用该关系并解决"a 之于 a* 正如 b 之于 —"形式的类比问题，其中关系的性质是隐藏的。也许最著名的例子是，词 queen 的嵌入表示可以大致从 king、man 和 woman 的表示中恢复：

queen ≈ king − man + woman

使用 RNN 嵌入向量上的向量算术恢复关系相似度在许多关系上进行了评估，在关系相似度识别任务中取得了最先进的结果（Mikolov et al.，2013c；Zhila et al.，2013）[20, 35]。随后证明，用不同架构训练的嵌入也可以以类似方式恢复关系相似度（Mikolov et al.，2013a；Mikolov et al.，2013b）[18, 19]。

这个迷人的结果提出了一个问题：关系语义属性在多大程度上是嵌入过程的产物？（Mikolov et al.，2013c）[20] 中的实验表明，基于 RNN 的嵌入优于其他稠密表示，但表示必须是稠密且低维这一点到底有多关键？

表示词向量的另一种方法是分布式相似度表示，即上下文词袋。在这种表示中，每个词与一个非常高维但稀疏的向量相关联，捕获词出现的上下文。我们称此类向量表示为显式的，因为每个维度直接对应于一个特定上下文。这些显式向量空间表示在 NLP 文献中被广泛研究（见（Turney and Pantel，2010；Baroni and Lenci，2010）[30, 3] 及其参考文献），并且已知展示了大程度的归因相似度（Pereira et al.，1993；Lin，1998；Lin and Pantel，2001；Sahlgren，2006；Kotlerman et al.，2010）[25, 16, 15, 26, 13]。

在这项研究中，我们表明与神经嵌入空间类似，显式向量空间也编码了大量可以以类似方式恢复的关系相似度，这表明显式向量空间表示可以作为神经嵌入进一步工作的有竞争力的基线。此外，这一结果意味着神经嵌入过程并未发现新的模式，而是在出色地保留词-上下文共现矩阵中固有的模式。

这项工作的一个关键洞见是，向量算术方法可以分解为三个两两相似度的线性组合（第 3 节）。虽然在数学上等价，但我们发现用分解的公式来思考该方法不那么令人困惑，并为为什么我们期望该方法在类比恢复任务上表现良好提供了更好的直觉。此外，分解形式引导我们提出一个改进的优化目标（第 6 节），它在两种表示下恢复关系相似度方面都优于最先进水平。

[^1]: 作者得到欧盟第七框架计划（FP7/2007-2013）根据协议号 287923（EXCITEMENT）的支持。

## 2 显式向量空间表示

我们采用分布式相似度文献中使用的传统词表示（Turney and Pantel，2010）[30]。每个词与一个捕获其出现上下文的稀疏向量相关联。我们称这种表示为显式的，因为每个维度对应于一个特定上下文。

对于词汇表 $V$ 和上下文集合 $C$，结果是 $|V| \times |C|$ 稀疏矩阵 $S$，其中 $S_{ij}$ 对应词 $i$ 和上下文 $j$ 之间的关联强度。词 $w \in V$ 和上下文 $c \in C$ 之间的关联强度可以采取多种形式。我们选择使用流行的正逐点互信息（PPMI，Positive Pointwise Mutual Information）度量：

$$
S_{ij} = \text{PPMI}(w_i, c_j)
$$

$$
\text{PPMI}(w, c) = \begin{cases} 0 & \text{PMI}(w, c) < 0 \\ \text{PMI}(w, c) & \text{otherwise} \end{cases}
$$

$$
\text{PMI}(w, c) = \log \frac{P(w, c)}{P(w)P(c)} = \log \frac{\text{freq}(w, c)|\text{corpus}|}{\text{freq}(w)\text{freq}(c)}
$$

其中 $|\text{corpus}|$ 是语料库中的 item 数量，$\text{freq}(w, c)$ 是词 $w$ 在语料库中出现在上下文 $c$ 中的次数，$\text{freq}(w)$、$\text{freq}(c)$ 分别是词和上下文的语料库频率。

在分布式相似度模型中使用 PMI 由 Church and Hanks（1990）[8] 引入并被广泛采用（Dagan et al.，1994；Turney，2001）[11, 31]。PPMI 变体至少可以追溯到（Niwa and Nitta，1994）[23]，并被证明在 Bullinaria and Levy（2007）[7] 中表现非常好。

在这项工作中，我们采用词出现的线性上下文。我们将目标词 $w$ 周围每侧窗口 2 内的每个词视为一个上下文，区分不同的顺序位置。例如，在句子 a b c d e 中，词 c 的上下文是 a−2、b−1、d+1 和 e+2。因此每个向量的维度是 $|C| \approx 4|V|$。经验上，我们语料库中词汇 item 的非零维度数介于 3（对于某些稀有 token）和 474,234（对于词"and"）之间，平均 1,595，中位数 415。

另一种流行的上下文选择是词参与的句法关系（Lin，1998；Padó and Lapata，2007；Levy and Goldberg，2014）[16, 24, 14]。在本文中，我们选择顺序上下文，因为它与我们比较的最先进神经嵌入方法可用的信息兼容。

## 3 类比与向量算术

Mikolov 等人证明向量空间表示编码了各种关系相似度，这些相似度可以使用向量算术恢复并用于解决词类比任务。

### 3.1 类比问题

在词类比任务中，我们得到共享一个关系的两对词（如"man:woman"、"king:queen"）。第四个词（"queen"）的身份是隐藏的，我们需要基于其他三个词推断它（例如回答这个问题："man 之于 woman 正如 king 之于 —？"）。在本文的其余部分，我们将这四个词称为 a:a\*、b:b\*。注意问题的类型没有在问题中显式提供，正确解答问题（由人类）涉及先推断关系，然后将其应用于第三个词（b）。

### 3.2 向量算术

Mikolov 等人表明，词之间的关系在很大程度上反映在它们向量嵌入之间的偏移中（queen − king ≈ woman − man），因此隐藏词 b\* 的向量将类似于向量 b − a + a\*，这表明类比问题可以通过优化下式来解决：

$$
\arg \max_{b^{*} \in V} \left( \text{sim}(b^{*}, b - a + a^{*}) \right)
$$

其中 $V$ 是排除问题词 $b$、$a$ 和 $a^{*}$ 的词汇表，$\text{sim}$ 是相似度度量。具体来说，他们使用余弦相似度度量，定义为：

$$
\cos(u, v) = \frac{u \cdot v}{\|u\| \|v\|}
$$

得到：

$$
\arg \max_{b^{*} \in V} \left( \cos(b^{*}, b - a + a^{*}) \right) \qquad (1)
$$

由于余弦与角度成反比，高余弦相似度（接近 1）意味着向量共享非常相似的方向。注意这个度量归一化（从而忽略）了向量的长度，这与它们之间的欧氏距离不同。出于稍后会清楚的原因，我们将 (1) 称为 3COSADD 方法。

3COSADD 的一个替代方案是要求变换方向被保持：

$$
\arg \max_{b^{*} \in V} \left( \cos(b^{*} - b, a^{*} - a) \right) \qquad (2)
$$

这基本上意味着 $b^{*} - b$ 与 $a^{*} - a$ 共享相同方向，忽略距离。我们将此方法称为 PAIRDIRECTION。虽然论文中没有提到，但 Mikolov et al.（2013c）[20] 使用 PAIRDIRECTION 解决 SemEval 任务的语义类比，使用 3COSADD 解决句法类比。[^2]

[^2]: 这一点由我们独立试验和与作者通信两方面确认。

### 3.3 重新解释向量算术

在 Mikolov 等人的实验中，所有词向量都被归一化为单位长度。在此归一化下，(1) 中的 arg max 在数学上等价于（用基础代数推导）：

$$
\arg \max_{b^{*} \in V} \left( \cos(b^{*}, b) - \cos(b^{*}, a) + \cos(b^{*}, a^{*}) \right) \qquad (3)
$$

这意味着用向量算术解决类比问题在数学上等价于寻找一个与 $b$ 和 $a^{*}$ 相似但与 $a$ 不同的词（$b^{*}$）。关系相似度因此被表示为归因相似度之和。虽然 (1) 和 (3) 是相等的，但我们发现 (3) 为何应该能找到类比的直觉更清晰。

## 4 实证设置

我们推导显式和神经嵌入向量表示，并使用目标 3COSADD（等式 3）和 PAIRDIRECTION（等式 2）比较它们恢复关系相似度的能力。

**底层语料库与预处理** 先前在词类比任务上使用向量算术的报告结果是用专有语料库获得的。为使我们的实验可复现，我们选择了一个开放且广泛可访问的语料库——英语 Wikipedia。我们从文章正文（排除标题、信息框、说明文字等）中提取所有句子，并过滤非字母数字 token，允许撇号、连字符、逗号和句号等 token 中间符号。所有文本被小写化。然后移除重复项和少于 5 个 token 的句子。总体而言，我们保留了一个约 15 亿 token、7,750 万个句子的语料库。

**词表示** 为创建嵌入和稀疏表示的上下文，我们使用每侧两个 token 的窗口（总共 5-gram），忽略在语料库中出现少于 100 次的词。[^3] 过滤后的词汇表包含 189,533 个词条。[^4]

显式向量表示如第 2 节所述创建。神经嵌入使用 (Mikolov et al.，2013b) [19] 附带的 word2vec 软件创建。[^4] 我们将词汇表嵌入到 600 维空间中，使用最先进的 skip-gram 架构、带 15 个负样本的负训练方法（NEG-15），以及参数为 $10^{-5}$ 的频繁词子采样。参数设置遵循（Mikolov et al.，2013b）[19]。

[^3]: 不同窗口大小和截止值的初始实验显示出相似趋势。
[^4]: http://code.google.com/p/word2vec
### 4.1 评估条件

我们使用先前工作使用的三个数据集评估不同的词表示。其中两个（MSR 和 GOOGLE）包含类比问题，而第三个（SEMEVAL）要求根据候选词对与一组给定词对的关系相似度对候选词对进行排序。

**开放词汇** 开放词汇数据集（MSR 和 GOOGLE）提出"a 之于 a\* 正如 b 之于 b\*"形式的问题，其中 b\* 是隐藏的，必须从整个词汇表中猜测。这些数据集的性能通过微平均准确率衡量。

MSR 数据集[^5]（Mikolov et al.，2013c）[20] 包含 8,000 个类比问题。这些问题描绘的关系是形态句法的，可以根据词性分类——形容词、名词和动词。形容词关系包括比较级和最高级（good 之于 best 正如 smart 之于 smartest）。名词关系包括单数和复数、所有格和非所有格（dog 之于 dog's 正如 cat 之于 cat's）。动词关系是时态修改（work 之于 worked 正如 accept 之于 accepted）。

GOOGLE 数据集[^6]（Mikolov et al.，2013a）[18] 包含 19,544 个问题。它涵盖 14 种关系类型，其中 7 种本质上是语义的，7 种是形态句法的（在第 8 节中列举）。该数据集通过手动构建每种关系的示例词对，并提供（每种关系类型内的）所有词对组合作为类比问题来创建。词汇表外（OOV，Out-of-Vocabulary）词[^7] 从两个测试集中移除。

[^5]: research.microsoft.com/en-us/projects/rnn/
[^6]: code.google.com/p/word2vec/source/browse/trunk/questions-words.txt
[^7]: 即出现在英语 Wikipedia 中少于 100 次的词。这从 MSR 数据集中移除了 882 个实例，从 GOOGLE 中移除了 286 个实例。

**封闭词汇** SEMEVAL 数据集包含出现在 SemEval 2012 任务 2：度量关系相似度（Jurgens et al.，2012）[12] 中的 79 个语义关系的集合。每个关系由几个（通常 3 个）特征词对示例化。给定一组几十个目标词对（据称具有相同关系），任务是根据该关系成立的程度对目标对排序。这可以以下列方式表述为类比问题：例如，取 Recipient:Instrument 关系，其原型词对为 king:crown 和 police:badge。要度量目标词对 wife:ring 具有相同关系的程度，我们形成两个类比问题"king 之于 crown 正如 wife 之于 ring"和"police 之于 badge 正如 wife 之于 ring"。我们计算每个类比的得分，并平均结果。注意与前两个测试集相反，这个不需要在整个词汇表中搜索最适合的词，而是对现有词对列表排序。

遵循先前工作，SEMEVAL 上的性能用准确率衡量，在所有关系上宏平均。

## 5 初步结果

我们的第一个实验使用 3COSADD（第 3 节中的方法 (3)）来度量每种表示中语言规律的普遍性。

**表 1：使用显式和神经嵌入表示的 3COSADD 在不同任务上的性能。**

| 表示 | MSR | GOOGLE | SEMEVAL |
|---|---|---|---|
| 嵌入 | 53.98% | 62.70% | 38.49% |
| 显式 | 29.04% | 45.05% | 38.54% |

表 1 的结果表明，两种表示都可以恢复大量关系相似度。事实上，两种表示在 SEMEVAL 任务上取得相同的准确率。然而，在开放词汇的 MSR 和 GOOGLE 任务中，存在有利于神经嵌入的巨大性能差距。

接下来，我们用 PAIRDIRECTION（第 3 节中的方法 (2)）运行相同的实验。

**表 2：使用显式和神经嵌入表示的 PAIRDIRECTION 在不同任务上的性能。**

| 表示 | MSR | GOOGLE | SEMEVAL |
|---|---|---|---|
| 嵌入 | 9.26% | 14.51% | 44.77% |
| 显式 | 0.66% | 0.75% | 45.19% |

表 2 的结果表明，PAIRDIRECTION 方法在受限词汇的 SEMEVAL 任务上优于 3COSADD（准确率从 38% 跳到 45%），但在 GOOGLE 和 MSR 的开放词汇问题上失败。当方法有效时，显式和嵌入表示的数值再次彼此相当。

为什么 PAIRDIRECTION 在 SEMEVAL 任务上表现如此之好，在其他任务上却如此之差？回想 PAIRDIRECTION 目标关注 $b^{*} - b$ 和 $a^{*} - a$ 的相似度，但不考虑单个向量之间的空间距离。仅依赖方向而忽略空间距离，在将整个词汇表作为候选时（如 MSR 和 GOOGLE 任务所要求的）是有问题的。我们很可能找到与 $b$ 具有 $a - a^{*}$ 所反映的相同关系但不一定与 $b$ 相似的候选 $b^{*}$。举个具体的例子，在 man:woman、king:? 中，我们很可能恢复阴性实体，但不一定是王室实体。另一方面，SEMEVAL 测试集已经提供了相关（因此在几何上接近）的候选，主要留下方向需要推理。

## 6 改进目标函数

3COSADD 目标（如 (3) 中所表达）揭示了两个吸引项和一个排斥项之间的"平衡动作"，即我们希望最大化的两个项和一个需要最小化的项：

$$
\arg \max_{b^{*} \in V} \left( \cos(b^{*}, b) - \cos(b^{*}, a) + \cos(b^{*}, a^{*}) \right)
$$

此类线性目标的一个已知性质是它们表现出"软或"行为，允许一个足够大的项主导整个表达式。这种行为在我们的设置中是有问题的，因为每个项反映相似度的不同方面，而不同的方面有不同的尺度。例如，king 比它是阳性更重要的是它是王室的，因此会遮蔽类比的性别方面。在显式向量表示的情况下尤其如此，因为相似度的每个方面由一组大小和权重不同的特征来体现。

一个典型的例子是类比问题"London 之于 England 正如 Baghdad 之于 —？"，我们用下式回答：

$$
\arg \max_{x \in V} \left( \cos(x, en) - \cos(x, lo) + \cos(x, ba) \right)
$$

我们寻找一个与 England 相似（都是国家）、与 Baghdad 相似（相似地理/文化）且与 London 不相似（不同地理/文化）的词（Iraq）。最大化这个和得到错误答案（两种表示下都是）：Mosul，一个伊拉克大城市。查看显式向量表示中计算的相似度，我们看到 Mosul 和 Iraq 都非常接近 Baghdad，而离 England 和 London 相当远：

（EXP）

| | ↑England | ↓London | ↑Baghdad | 和 |
|---|---|---|---|---|
| Mosul | 0.031 | 0.031 | 0.244 | 0.244 |
| Iraq | 0.049 | 0.038 | 0.206 | 0.217 |

同样的趋势出现在神经嵌入向量中，尽管相似度得分不同：

（EMB）

| | ↑England | ↓London | ↑Baghdad | 和 |
|---|---|---|---|---|
| Mosul | 0.130 | 0.141 | 0.755 | 0.748 |
| Iraq | 0.153 | 0.130 | 0.631 | 0.655 |

虽然 Iraq 比 Mosul 与 England 相似得多（都是国家），但两个相似度（显式 0.049 和 0.031，嵌入 0.130 和 0.153）都很小，和由类比的地理和文化方面主导：Mosul 和 Iraq 与 Baghdad 的相似度（显式 0.24 和 0.20，嵌入 0.75 和 0.63）。

为了在相似度的不同方面之间实现更好的平衡，我们建议从加法组合切换到乘法组合：

$$
\arg \max_{b^{*} \in V} \frac{\cos(b^{*}, b) \cos(b^{*}, a^{*})}{\cos(b^{*}, a) + \varepsilon} \qquad (4)
$$

（$\varepsilon = 0.001$ 用于防止除以零）

这等价于在求和之前对每项取对数，从而放大小数量之间的差异并减小较大数量之间的差异。使用这个目标，Iraq 的得分高于 Mosul（0.259 vs 0.236，0.736 vs 0.691）。我们将目标 (4) 称为 3COSMUL。[^8]

[^8]: 3COSMUL 要求所有相似度非负，这对显式表示平凡成立。对于嵌入，我们在计算 (4) 之前用 $(x + 1)/2$ 将余弦相似度变换到 $[0, 1]$。

## 7 主要结果

我们重复了实验，这次使用 3COSMUL 方法。表 3 展示了结果，表明乘法目标在两种表示中都恢复了更多关系相似度。显式表示取得的改进尤其显著，在 MSR 和 GOOGLE 数据集上正确识别的关系绝对增加超过 20%。

**表 3：3COSADD 和 3COSMUL 的比较。**

| 目标 | 表示 | MSR | GOOGLE |
|---|---|---|---|
| 3COSADD | 嵌入 | 53.98% | 62.70% |
| 3COSADD | 显式 | 29.04% | 45.05% |
| 3COSMUL | 嵌入 | 59.09% | 66.72% |
| 3COSMUL | 显式 | 56.83% | 68.24% |

3COSMUL 在这两个数据集上优于最先进水平（3COSADD）。此外，结果表明两种表示都能恢复可比数量的关系相似度。这表明神经嵌入中明显的语言规律不是嵌入过程的产物，而是被它很好地保留了。

在 SEMEVAL 上，3COSMUL 与 3COSADD 表现相当，用显式和神经表示恢复类似数量的类比（分别为 38.37% 和 38.67%）。

## 8 错误分析

使用 3COSMUL，显式向量和神经嵌入都恢复类似数量的类比，但它们是相同的模式，还是也许不同类型的关系相似度？

### 8.1 表示之间的一致性

考虑开放词汇任务（MSR 和 GOOGLE），我们统计两种表示都猜对、都猜错、以及一种表示得出正确答案而另一种没有的次数（表 4）。虽然表示之间有大量一致，但也有不可忽视数量的互补情况。如果我们运行一个神谕设置，其中答案在任一表示中正确就算正确，我们将在 MSR 数据集上达到 71.9% 的准确率，在 GOOGLE 上达到 77.8%。

**表 4：开放词汇任务上表示之间的一致性。**

| | 都正确 | 都错误 | 仅嵌入正确 | 仅显式正确 |
|---|---|---|---|---|
| MSR | 43.97% | 28.06% | 15.12% | 12.85% |
| GOOGLE | 57.12% | 22.17% | 9.59% | 11.12% |
| 全部 | 53.58% | 23.76% | 11.08% | 11.59% |

### 8.2 按关系类型分解

表 5 展示了每种表示中发现的不同关系类型的类比数量。出现了一些趋势：显式表示在更语义性的任务中表现优越，尤其是与地理相关的任务，以及最高级和名词。然而，神经嵌入在大多数动词屈折、比较级和家庭（性别）关系上占上风。有些关系（货币、形容词到副词、反义词）对两种表示都构成挑战，尽管嵌入表示处理得稍好。最后，国籍形容词和现在分词由两种表示同等处理。

**表 5：使用 3COSMUL 时，每种表示中按关系类型分类的关系相似度分解。**

| 关系 | 嵌入 | 显式 |
|---|---|---|
| GOOGLE | | |
| capital-common-countries | 90.51% | 99.41% |
| capital-world | 77.61% | 92.73% |
| city-in-state | 56.95% | 64.69% |
| currency | 14.55% | 10.53% |
| family (gender inflections) | 76.48% | 60.08% |
| gram1-adjective-to-adverb | 24.29% | 14.01% |
| gram2-opposite | 37.07% | 28.94% |
| gram3-comparative | 86.11% | 77.85% |
| gram4-superlative | 56.72% | 63.45% |
| gram5-present-participle | 63.35% | 65.06% |
| gram6-nationality-adjective | 89.37% | 90.56% |
| gram7-past-tense | 65.83% | 48.85% |
| gram8-plural (nouns) | 72.15% | 76.05% |
| gram9-plural-verbs | 71.15% | 55.75% |
| MSR | | |
| adjectives | 45.88% | 56.46% |
| nouns | 56.96% | 63.07% |
| verbs | 69.90% | 52.97% |

### 8.3 默认行为错误

两种表示下最常见的错误模式是"默认行为"，即一个核心代表词被提供作为许多同类问题的答案。例如，在嵌入表示中，词"Fresno"在 city-in-state 关系中被返回 82 次作为错误答案，在显式表示中，词"daughter"在 family 关系中被返回 47 次作为错误答案。粗略地说，"Fresno"被嵌入表示识别为典型地点，而"daughter"被显式表示识别为典型女性。在默认行为错误的定义是某个特定关系返回相同错误答案 10 次或更多的情况下，此类错误占显式表示错误的 49%，占嵌入表示错误的 39%。

表 6 列出了两种表示下最常见的 15 个默认错误。在大多数默认错误中，默认词的类别与类比问题密切相关，与正确答案或（如"Fresno"的情况）问题词共享类别。值得注意的例外是词"who"、"and"、"be"和"smith"，它们在显式表示中被作为默认答案返回，而这些词与预期关系相去甚远。似乎在显式表示中，一些非常频繁的功能词充当"枢纽"并混淆模型。事实上，过去时和复数动词关系中表示之间的性能差距可以特别归因于此类功能词错误：过去时关系中 23.4% 的错误是由于显式表示默认回答"who"或"and"，而复数动词关系中 19% 的错误是由于"is/and/that/who"的默认答案。

**表 6：两种表示下常见的默认行为错误。EMB / EXP：在嵌入或显式表示下，该词被返回为给定关系的错误答案的次数。**

| 关系 | 词 | EMB | EXP |
|---|---|---|---|
| gram7-past-tense | who | 0 | 138 |
| city-in-state | fresno | 82 | 24 |
| gram6-nationality-adjective | slovak | 39 | 39 |
| gram6-nationality-adjective | argentine | 37 | 39 |
| gram6-nationality-adjective | belarusian | 37 | 39 |
| gram8-plural (nouns) | colour | 36 | 35 |
| gram3-comparative | higher | 34 | 35 |
| city-in-state | smith | 1 | 61 |
| gram7-past-tense | and | 0 | 49 |
| gram1-adjective-to-adverb | be | 0 | 47 |
| family (gender inflections) | daughter | 8 | 47 |
| city-in-state | illinois | 3 | 40 |
| currency | currency | 5 | 40 |
| gram1-adjective-to-adverb | and | 0 | 39 |
| gram7-past-tense | enhance | 39 | 20 |

### 8.4 动词屈折错误

形态类比问题的正确解需要恢复正确的屈折（要求句法相似度）和正确的基础词（要求语义相似度）。我们观察到，从语言学上讲，形态区别和相似性往往依赖少数常见的词形（例如，"walk:walking"关系以"will"等情态动词出现在"walk"之前而从不出现在"walking"之前、be 动词出现在 walking 之前而从不出现在 walk 之前为特征），而语义关系的支持则分布在更多的 item 上。我们假设动词中的形态区别比语义更难捕获。事实上，在两种表示下，所选词具有正确词形但错误屈折的错误，比所选词具有正确屈折但错误基础词形式的错误，可能性高十倍以上。

## 9 解释关系相似度

通过向量（或相似度）算术捕获关系相似度的能力是卓越的。在本节中，我们尝试提供为什么它有效的直觉。

考虑词"king"；它有多个方面，即它蕴含的高层属性，如王室或（男性）性别，它与其他词的归因相似度基于这些方面的混合；例如，king 在王室和人类轴上与 queen 相关，并与 man 共享性别和人类方面。关系相似度可以被视为归因相似度的组合，每个反映一个不同方面。在"man 之于 woman 正如 king 之于 queen"中，两个主要方面是性别和王室。解决类比问题涉及识别相关方面，并尝试在保持另一个的同时改变其中一个。

像性别、王室或"城市性"这样的概念在向量空间中如何表示？虽然神经嵌入大多是难以解读的，但显式向量表示的一个吸引人属性是我们能够读取和理解向量的特征。例如，king 在我们的显式向量空间中由 51,409 个上下文表示，其中前 3 个是 tut+1、jeongjo+1、adulyadej+2——都是君主名字。显式表示让我们得以一窥不同方面被表示的方式。为此，我们选择一对共享一个方面的代表词，取它们向量的交集，并检查交集得分最高的特征。表 7 展示了每个方面的最高（最有影响力的）特征。

**表 7：通过对共享该方面的词进行逐点乘法恢复的每个方面的顶级特征。逐点乘法的结果是一个"方面向量"，其中两个词共有的、刻画该关系的特征得分最高。特征得分（未显示）对应特征对向量间余弦相似度的贡献权重。上标标记特征相对于目标词的位置。**

| 方面 | 示例 | 顶级特征 |
|---|---|---|
| 女性 | woman ⊙ queen | estrid+1 ketevan+1 adeliza+1 nzinga+1 gunnhild+1 impregnate−2 hippolyta+1 |
| 王室 | queen ⊙ king | savang+1 uncrowned−1 pmare+1 sisowath+1 nzinga+1 tupou+1 uvea+2 majesty−1 |
| 货币 | yen ⊙ ruble | devalue−2 banknote+1 denominated+1 billion−1 banknotes+1 pegged+2 coin+1 |
| 国家 | germany ⊙ australia | emigrates−2 1943-45+2 pentathletes−2 emigrated−2 emigrate−2 hong-kong−1 |
| 首都 | berlin ⊙ canberra | hotshots−1 embassy−2 1925-26+2 consulate-general+2 meetups−2 nunciature−2 |
| 最高级 | sweetest ⊙ tallest | freshest+2 asia's−1 cleveland's−2 smartest+1 world's−1 city's−1 america's−1 |
| 高度 | taller ⊙ tallest | regnans−2 skyscraper+1 skyscrapers+1 6'4+2 windsor's−1 smokestacks+1 burj+2 |

这些特征中有许多是在我们的语料库中很少出现的姓名或地名（如 Adeliza，一位历史女王，以及 Nzinga，一个王室家族），但尽管如此，它们对共享概念极具指示性。稀有词的普遍性源于 PMI 给它们更多权重，以及 woman 和 queen 等词密切相关（queen 是女人），因此共享许多特征。按出现频率排序 woman ⊙ queen 的特征，揭示了女性代词（"she"、"her"）和一长串常见女性名字，反映了 woman 和 queen 共享的预期方面。共享更具体方面的词对，如首都城市或国家，展示对其共享方面特有的特征（例如，首都城市有大使馆和聚会，而移民与国家相关）。观察相对句法的"最高级"方面如何被许多地区性所有格（"america's"、"asia's"、"world's"）捕获也很有趣。

## 10 相关工作

关系相似度（和回答类比问题）以前用显式表示处理过。以前的方法使用任务特定信息，要么依赖（词对、连接词）矩阵而不是标准的（词、上下文）矩阵（Turney and Littman，2005；Turney，2006）[29, 32]，要么将类比检测视为监督学习任务（Baroni and Lenci，2009；Jurgens et al.，2012；Turney，2013）[2, 12, 34]。相比之下，这里遵循的向量算术方法是无监督的，作用于通用的单词表示。即使训练过程对类比检测任务不知情，得到的表示也能相当准确地检测它们。Turney（2012）[33] 假设了类似的设置，但使用两种类型的词相似度，并用乘积和比率（类似 3COSMUL）组合它们以恢复各种语义关系，包括类比。

显式词向量的算术组合在组合语义的背景下被广泛研究（Mitchell and Lapata，2010）[21]，其中由两个或多个词组成的短语由单个向量表示，由分量词向量的函数计算。Blacoe and Lapata（2012）[6] 在一系列组合性基准上比较了多种表示（包括嵌入）的不同算术函数。据我们所知，此类词向量算术方法尚未被探索用于恢复显式表示中的关系相似度。

## 11 讨论

Mikolov 等人展示了无监督神经网络如何在"自然地"以向量偏移形式编码关系相似度的空间中表示词。这项研究表明，通过向量算术寻找类比实际上是一种平衡词相似度的形式，并且与 Baroni et al.（2014）[4] 最近的发现相反，在某些条件下，显式表示诱导的传统词相似度在此任务上可以表现得与神经嵌入一样好。

学习表示词是一个迷人且重要的挑战，对当前大多数 NLP 工作都有影响，神经嵌入尤其是一个有前景的研究方向。我们相信，要改进这些表示，我们应该理解它们如何工作，并希望这项工作提供的方法和洞见将有助于加深我们对当前和未来词表示研究的把握。

## 参考文献

[1] Rami Al-Rfou, Bryan Perozzi, and Steven Skiena. 2013. Polyglot: Distributed word representations for multilingual nlp. In Proc. of CoNLL 2013.

[2] Marco Baroni and Alessandro Lenci. 2009. One distributional memory, many semantic spaces. In Proceedings of the Workshop on Geometrical Models of Natural Language Semantics, pages 1–8, Athens, Greece, March. Association for Computational Linguistics.

[3] Marco Baroni and Alessandro Lenci. 2010. Distributional memory: A general framework for corpus-based semantics. Computational Linguistics, 36(4):673–721.

[4] Marco Baroni, Georgiana Dinu, and German Kruszewski. 2014. Dont count, predict! a systematic comparison of context-counting vs. context-predicting semantic vectors. In Proceedings of the 52nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), Baltimore, Maryland, USA, June. Association for Computational Linguistics.

[5] Yoshua Bengio, Réjean Ducharme, Pascal Vincent, and Christian Jauvin. 2003. A neural probabilistic language model. Journal of Machine Learning Research, 3:1137–1155.

[6] William Blacoe and Mirella Lapata. 2012. A comparison of vector-based representations for semantic composition. In Proceedings of the 2012 Joint Conference on Empirical Methods in Natural Language Processing and Computational Natural Language Learning, pages 546–556, Jeju Island, Korea, July. Association for Computational Linguistics.

[7] John A. Bullinaria and Joseph P. Levy. 2007. Extracting semantic representations from word co-occurrence statistics: A computational study. Behavior Research Methods, 39(3):510–526.

[8] Kenneth Ward Church and Patrick Hanks. 1990. Word association norms, mutual information, and lexicography. Computational linguistics, 16(1):22–29.

[9] Ronan Collobert and Jason Weston. 2008. A unified architecture for natural language processing: Deep neural networks with multitask learning. In Proceedings of the 25th International Conference on Machine Learning, pages 160–167.

[10] Ronan Collobert, Jason Weston, Léon Bottou, Michael Karlen, Koray Kavukcuoglu, and Pavel Kuksa. 2011. Natural language processing (almost) from scratch. The Journal of Machine Learning Research, 12:2493–2537.

[11] Ido Dagan, Fernando Pereira, and Lillian Lee. 1994. Similarity-based estimation of word cooccurrence probabilities. In Proceedings of the 32nd annual meeting on Association for Computational Linguistics, pages 272–278. Association for Computational Linguistics.

[12] David A Jurgens, Peter D Turney, Saif M Mohammad, and Keith J Holyoak. 2012. Semeval-2012 task 2: Measuring degrees of relational similarity. In Proceedings of the First Joint Conference on Lexical and Computational Semantics, pages 356–364. Association for Computational Linguistics.

[13] Lili Kotlerman, Ido Dagan, Idan Szpektor, and Maayan Zhitomirsky-Geffet. 2010. Directional distributional similarity for lexical inference. Natural Language Engineering, 16(4):359–389.

[14] Omer Levy and Yoav Goldberg. 2014. Dependency-based word embeddings. In Proceedings of the 52nd Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), Baltimore, Maryland, USA, June. Association for Computational Linguistics.

[15] Dekang Lin and Patrick Pantel. 2001. Dirt: discovery of inference rules from text. In KDD, pages 323–328.

[16] Dekang Lin. 1998. Automatic retrieval and clustering of similar words. In Proceedings of the 36th Annual Meeting of the Association for Computational Linguistics and 17th International Conference on Computational Linguistics - Volume 2, ACL '98, pages 768–774, Stroudsburg, PA, USA. Association for Computational Linguistics.

[17] Tomas Mikolov, Stefan Kombrink, Lukas Burget, JH Cernocky, and Sanjeev Khudanpur. 2011. Extensions of recurrent neural network language model. In Acoustics, Speech and Signal Processing (ICASSP), 2011 IEEE International Conference on, pages 5528–5531. IEEE.

[18] Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. 2013a. Efficient estimation of word representations in vector space. CoRR, abs/1301.3781.

[19] Tomas Mikolov, Ilya Sutskever, Kai Chen, Gregory S. Corrado, and Jeffrey Dean. 2013b. Distributed representations of words and phrases and their compositionality. In Advances in Neural Information Processing Systems 26: 27th Annual Conference on Neural Information Processing Systems 2013. Proceedings of a meeting held December 5-8, 2013, Lake Tahoe, Nevada, United States, pages 3111–3119.

[20] Tomas Mikolov, Wen-tau Yih, and Geoffrey Zweig. 2013c. Linguistic regularities in continuous space word representations. In Proceedings of the 2013 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 746–751, Atlanta, Georgia, June. Association for Computational Linguistics.

[21] Jeff Mitchell and Mirella Lapata. 2010. Composition in distributional models of semantics. Cognitive Science, 34(8):1388–1439.

[22] Andriy Mnih and Geoffrey E Hinton. 2008. A scalable hierarchical distributed language model. In Advances in Neural Information Processing Systems, pages 1081–1088.

[23] Yoshiki Niwa and Yoshihiko Nitta. 1994. Co-occurrence vectors from corpora vs. distance vectors from dictionaries. In Proceedings of the 15th conference on Computational linguistics-Volume 1, pages 304–309. Association for Computational Linguistics.

[24] Sebastian Padó and Mirella Lapata. 2007. Dependency-based construction of semantic space models. Computational Linguistics, 33(2):161–199.

[25] Fernando Pereira, Naftali Tishby, and Lillian Lee. 1993. Distributional clustering of english words. In Proceedings of the 31st annual meeting on Association for Computational Linguistics, pages 183–190. Association for Computational Linguistics.

[26] Magnus Sahlgren. 2006. The Word-Space Model: Using distributional analysis to represent syntagmatic and paradigmatic relations between words in high-dimensional vector spaces. Ph.D. thesis, Stockholm.

[27] Richard Socher, Jeffrey Pennington, Eric H Huang, Andrew Y Ng, and Christopher D Manning. 2011. Semi-supervised recursive autoencoders for predicting sentiment distributions. In Proceedings of the Conference on Empirical Methods in Natural Language Processing, pages 151–161. Association for Computational Linguistics.

[28] Joseph Turian, Lev Ratinov, and Yoshua Bengio. 2010. Word representations: a simple and general method for semi-supervised learning. In Proceedings of the 48th Annual Meeting of the Association for Computational Linguistics, pages 384–394. Association for Computational Linguistics.

[29] Peter D. Turney and Michael L. Littman. 2005. Corpus-based learning of analogies and semantic relations. Machine Learning, 60(1-3):251–278.

[30] Peter D. Turney and Patrick Pantel. 2010. From frequency to meaning: Vector space models of semantics. Journal of Artificial Intelligence Research, 37(1):141–188.

[31] Peter D. Turney. 2001. Mining the web for synonyms: Pmi-ir versus lsa on toefl. In Proceedings of the 12th European Conference on Machine Learning, pages 491–502. Springer-Verlag.

[32] Peter D. Turney. 2006. Similarity of semantic relations. Computational Linguistics, 32(3):379–416.

[33] Peter D. Turney. 2012. Domain and function: A dual-space model of semantic relations and compositions. Journal of Artificial Intelligence Research, 44:533–585.

[34] Peter D. Turney. 2013. Distributional semantics beyond words: Supervised learning of analogy and paraphrase. CoRR, abs/1310.5042.

[35] Alisa Zhila, Wen-tau Yih, Christopher Meek, Geoffrey Zweig, and Tomas Mikolov. 2013. Combining heterogeneous models for measuring relational similarity. In Proceedings of the 2013 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 1000–1009, Atlanta, Georgia, June. Association for Computational Linguistics.
