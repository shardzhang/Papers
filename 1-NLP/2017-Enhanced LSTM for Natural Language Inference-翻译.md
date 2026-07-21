# 用于自然语言推理的增强型LSTM

> Qian Chen, Xiaodan Zhu, Zhenhua Ling, Si Wei, Hui Jiang, Diana Inkpen | 中国科学技术大学, 加拿大国家研究委员会, 科大讯飞, 约克大学, 渥太华大学

本文提出了增强型序列推理模型（ESIM）和混合推理模型（HIM），在SNLI数据集上达到了88.6%的准确率，证明了精心设计的链式LSTM模型可以超越所有先前模型。

---

## 摘要

推理和推断是人工智能的核心。对人类语言中的推理进行建模极具挑战性。随着大型标注数据[1]的可用，训练基于神经网络的推理模型最近变得可行，并且已被证明非常有效。在本文中，我们提出了一个新的最先进结果，在斯坦福自然语言推理数据集上达到了88.6%的准确率。与先前使用非常复杂网络架构的最优模型不同，我们首先证明，精心设计基于链式LSTM的序列推理模型可以超越所有先前模型。在此基础上，我们进一步表明，在局部推理建模和推理组合中显式考虑递归架构，我们可以获得额外的改进。特别是，即使添加到已经非常强大的模型中，融入句法解析信息也有助于我们取得最佳结果。

## 1 引言

推理和推断是人类和人工智能的核心。对人类语言中的推理进行建模是出了名的困难，但这是通往真正自然语言理解的基本问题，正如MacCartney和Manning[2]所指出的，"真正的自然语言理解的一个必要条件（如果不是充分条件）是掌握开放域自然语言推理。"先前的工作包括了对文本蕴含识别的广泛研究。

具体来说，自然语言推理（NLI）涉及确定是否可以从前提$p$推断出自然语言假设$h$，如下面的例子所示[3]，其中假设被认为是从前提蕴含的。

$p$: Several airlines polled saw costs grow more than expected, even after adjusting for inflation.
$h$: Some of the companies in the poll reported cost increases.

近年来，自然语言推理建模取得了进展。一个重要贡献是创建了更大的标注数据集——斯坦福自然语言推理（SNLI）数据集[1]。该语料库包含57万个人工编写的英文句子对，由多个人类标注者手动标注。这使得训练更复杂的推理模型成为可能。神经网络模型通常需要相对较大的标注数据来估计其参数，已被证明可以在SNLI上达到最先进的性能[1, 4, 5, 6, 7, 8]。

虽然一些先前的最优模型使用相当复杂的网络架构来实现最先进的结果[5]，但我们在本文中证明，增强基于链式LSTM的序列推理模型可以超越所有先前结果，表明这类序列推理方法的潜力尚未被充分发掘。更具体地说，我们展示了序列推理模型在SNLI基准上达到了88.0%的准确率。

探索NLI的句法信息对我们极具吸引力。在许多问题中，句法和语义密切交互，包括语义组合等[9]。像自然语言推理这样的复杂任务很可能涉及两者，这在文本蕴含识别（RTE）的背景下已被讨论过[10, 11]。在本文中，我们有兴趣在有相对较大训练数据的情况下，在神经网络框架内对此进行探索。我们表明，通过在局部推理建模和推理组合中使用递归网络显式编码解析信息，并将其融入我们的框架，我们获得了额外的提升，将性能提升到新的最先进水平，准确率达88.6%。

## 2 相关工作

自然语言推理的早期工作是在相当小的数据集上使用更传统的方法进行的（参见[3]的文献综述），包括大量关于文本蕴含识别的工作[12, 13]等。最近，Bowman等人[1]发布了SNLI数据集，包含57万个人工标注的句子对。他们还实验了简单的分类模型以及独立编码前提和假设的简单神经网络。Rocktäschel等人[14]提出了基于神经注意力的NLI模型，捕获了注意力信息。一般来说，基于注意力的模型已被证明在广泛的任务中有效，包括机器翻译[15]、语音识别[16, 17]、图像描述[18]和文本摘要[19, 20]等。对于NLI，这一思想允许神经模型关注句子的特定区域。

此后，各种更先进的网络被开发出来[4, 21, 22, 23, 5, 14, 24, 25, 6, 7, 26, 8]。其中，与我们的工作更相关的是Parikh等人[6]和Munkhdalai与Yu[5]提出的方法，它们属于性能最佳的模型。

Parikh等人[6]提出了一个相对简单但非常有效的可分解模型。该模型将NLI问题分解为可以单独解决的子问题。另一方面，Munkhdalai和Yu[5]提出了更复杂的网络，考虑了基于序列LSTM的编码、递归网络以及注意力模型的复杂组合，相比于Parikh等人[6]报告的结果，提升了约0.5%。

然而，序列推理网络在NLI方面的潜力是否已被充分利用尚不清楚。在本文中，我们首先重新审视这个问题，并证明增强基于链式网络的序列推理模型实际上可以超越所有先前结果。我们进一步表明，显式考虑递归架构来编码NLI的句法解析信息可以进一步提高性能。

## 3 混合神经推理模型

我们在此提出的自然语言推理网络由以下主要组件构成：输入编码、局部推理建模和推理组合。图1显示了架构的高层视图。垂直方向，该图描述了三个主要组件；水平方向，图的左侧表示我们名为ESIM的序列NLI模型，右侧表示在树LSTM中融入句法解析信息的网络。

在我们的符号中，我们有两个句子$a = (a_1, \ldots, a_{\ell_a})$和$b = (b_1, \ldots, b_{\ell_b})$，其中$a$是前提，$b$是假设。$a_i$或$b_j \in \mathbb{R}^l$是$l$维向量的嵌入，可以用一些预训练的词嵌入初始化，并用解析树组织。目标是预测表示$a$和$b$之间逻辑关系的标签$y$。

### 3.1 输入编码

我们采用双向LSTM（BiLSTM）作为NLI的基本构建块之一。我们首先使用它来编码输入前提和假设（公式(1)和(2)）。这里BiLSTM学习表示一个词（如$a_i$）及其上下文。之后，我们还将使用BiLSTM进行推理组合以构建最终预测，其中BiLSTM编码局部推理信息及其交互。为便于后续引用，我们将$a$的BiLSTM在时刻$i$生成的隐藏（输出）状态记为$\bar{a}_i$。同样适用于$\bar{b}_j$：

$$
\bar{a}_i = \text{BiLSTM}(a, i), \quad \forall i \in [1, \ldots, \ell_a] \qquad (1)
$$

$$
\bar{b}_j = \text{BiLSTM}(b, j), \quad \forall j \in [1, \ldots, \ell_b] \qquad (2)
$$

由于篇幅限制，我们跳过基本链式LSTM的描述，读者可参考[27]了解详情。简而言之，在建模序列时，LSTM使用一组软门以及一个记忆单元来控制信息流，从而有效地建模序列中的长距离信息/依赖关系。

双向LSTM在序列上分别从左端和右端运行前向和后向LSTM。这两个LSTM在每个时间步生成的隐藏状态被拼接在一起，以表示该时间步及其上下文。注意，我们在模型中使用的是LSTM记忆块。我们检查了其他循环记忆块如GRU[28]，但在NLI任务的验证集上它们劣于LSTM。

如上所述，探索句法对自然语言推理的有效性是有趣的；例如，即使将其融入性能最佳的模型中，它是否有用。为此，我们还将通过树LSTM[29, 30, 31]编码前提和假设的句法解析树，它将链式LSTM扩展为递归网络[32]。

具体来说，给定前提或假设的解析树，一个树节点部署一个如图2所示的树LSTM记忆块，并通过公式(3-10)计算。简而言之，在每个节点，输入向量$x_t$和其两个子节点（左子节点$h_{t-1}^L$和右子节点$h_{t-1}^R$）的隐藏向量被作为输入来计算当前节点的隐藏向量$h_t$。

**图2: 树LSTM记忆块。**

我们通过公式(3)在高层描述节点的更新以方便后续引用，详细计算见(4-10)。具体来说，节点的输入用于配置四个门：输入门$i_t$、输出门$o_t$以及两个遗忘门$f_t^L$和$f_t^R$。记忆单元$c_t$考虑每个子节点的单元向量$c_{t-1}^L$和$c_{t-1}^R$，它们分别由左遗忘门$f_t^L$和右遗忘门$f_t^R$门控。

$$
h_t = \text{TrLSTM}(x_t, h_{t-1}^L, h_{t-1}^R) \qquad (3)
$$

$$
h_t = o_t \odot \tanh(c_t) \qquad (4)
$$

$$
o_t = \sigma(W_o x_t + U_o^L h_{t-1}^L + U_o^R h_{t-1}^R) \qquad (5)
$$

$$
c_t = f_t^L \odot c_{t-1}^L + f_t^R \odot c_{t-1}^R + i_t \odot u_t \qquad (6)
$$

$$
f_t^L = \sigma(W_f x_t + U_f^{LL} h_{t-1}^L + U_f^{LR} h_{t-1}^R) \qquad (7)
$$

$$
f_t^R = \sigma(W_f x_t + U_f^{RL} h_{t-1}^L + U_f^{RR} h_{t-1}^R) \qquad (8)
$$

$$
i_t = \sigma(W_i x_t + U_i^L h_{t-1}^L + U_i^R h_{t-1}^R) \qquad (9)
$$

$$
u_t = \tanh(W_c x_t + U_c^L h_{t-1}^L + U_c^R h_{t-1}^R) \qquad (10)
$$

其中$\sigma$是sigmoid函数，$\odot$是两个向量的逐元素乘法，所有$W \in \mathbb{R}^{d \times l}$、$U \in \mathbb{R}^{d \times d}$是要学习的权重矩阵。在当前输入编码层中，$x_t$用于对叶子节点编码词嵌入。由于非叶子节点不对应特定词，我们使用一个特殊的向量$x'_t$作为其输入，类似于未知词。然而，在我们稍后讨论的推理组合层中，使用树LSTM的目标非常不同；输入$x_t$也将非常不同——它将编码局部推理信息，并且在所有树节点上都有值。

### 3.2 局部推理建模

建模前提和假设之间的局部子句推理是确定这两个陈述之间整体推理的基本组件。为了仔细检查局部推理，我们探索了上述的序列模型和句法树模型。前者有助于收集词及其上下文的局部推理，而树LSTM有助于收集（语言）短语和从句之间的局部信息。

**推理的局部性。** 建模局部推理需要采用某种形式的硬对齐或软对齐来关联前提和假设之间的相关子组件。这包括来自传统自动机器翻译中受对齐启发的早期方法[3]。在神经网络模型中，这通常通过软注意力实现。

Parikh等人[6]分解了这一过程：前提（或假设）的词序列被视为词袋嵌入向量，并单独计算句子间"对齐"（或注意力），以软对齐每个词到假设（或前提）的内容。虽然他们的基本框架非常有效，取得了先前的最佳结果之一，但仅使用预训练的词嵌入本身并未自动考虑NLI中词周围的上下文。Parikh等人[6]确实通过可选的句子内距离敏感注意力考虑了词序和上下文信息。

在本文中，我们主张在前述的双向序列编码上利用注意力。我们将证明，这在实现我们的最佳结果中起着重要作用，而Parikh等人[6]使用的句子内注意力实际上并未对我们模型带来进一步提升，尽管他们提出的整体框架非常有效。

我们的软对齐层使用公式(11)计算前提和假设之间隐藏状态元组$<\bar{a}_i, \bar{b}_j>$的相似度作为注意力权重。我们确实研究了使用多层感知机的$\bar{a}_i$和$\bar{b}_j$之间更复杂的关系，但在验证数据上没有观察到进一步的提升。

$$
e_{ij} = \bar{a}_i^\top \bar{b}_j \qquad (11)
$$

在该公式中，$\bar{a}_i$和$\bar{b}_j$在前面公式(1)和(2)中计算，或在使用树LSTM时通过公式(3)计算。再次，如上所述，我们将分别使用双向LSTM和树LSTM来编码前提和假设。在我们的序列推理模型中，与Parikh等人[6]提出使用函数$F(\bar{a}_i)$（即前馈神经网络）来映射原始词表示以计算$e_{ij}$不同，我们主张使用BiLSTM，它能很好地编码前提和假设中的信息，并在实验部分中表现出更好的性能。我们尝试在计算$e_{ij}$之前对我们的隐藏状态应用$F(\cdot)$函数，但这并未进一步帮助我们的模型。

**在序列上收集局部推理。** 局部推理由上述计算的注意力权重$e_{ij}$确定，用于获取前提和假设之间的局部相关性。对于前提中一个词的隐藏状态$\bar{a}_i$（已编码词本身及其上下文），使用$e_{ij}$识别并组合假设中的相关语义，具体通过公式(12)。

$$
\tilde{a}_i = \sum_{j=1}^{\ell_b} \frac{\exp(e_{ij})}{\sum_{k=1}^{\ell_b} \exp(e_{ik})} \bar{b}_j, \quad \forall i \in [1, \ldots, \ell_a] \qquad (12)
$$

$$
\tilde{b}_j = \sum_{i=1}^{\ell_a} \frac{\exp(e_{ij})}{\sum_{k=1}^{\ell_a} \exp(e_{kj})} \bar{a}_i, \quad \forall j \in [1, \ldots, \ell_b] \qquad (13)
$$

其中$\tilde{a}_i$是$\{\bar{b}_j\}_{j=1}^{\ell_b}$的加权求和。直观上，$\{\bar{b}_j\}_{j=1}^{\ell_b}$中与$\bar{a}_i$相关的内容将被选择并表示为$\tilde{a}_i$。假设中的每个词也通过公式(13)进行同样的操作。

**在解析树上收集局部推理。** 我们使用树模型帮助在此层收集语言短语和从句上的局部推理信息。前提和假设的树结构由成分解析器生成。一旦树的所有隐藏状态都通过公式(3)计算完成，我们将所有树节点同等对待，因为没有进一步的启发式方法区分它们，而是让注意力权重来找出它们之间的关系。因此，我们使用公式(11)计算前提和假设之间所有节点对的注意力权重。这连接了前提和假设之间的所有词、成分短语和从句。然后我们使用公式(12)和(13)收集所有对之间的信息，并将其输入下一层。

**局部推理信息的增强。** 在我们的模型中，我们进一步增强收集到的局部推理信息。我们计算元组$<\bar{a}, \tilde{a}>$以及$<\bar{b}, \tilde{b}>$的差和逐元素积。我们期望这些操作可以帮助锐化元组中元素之间的局部推理信息，并捕获诸如矛盾之类的推理关系。差和逐元素积随后与原始向量$\bar{a}$和$\tilde{a}$（或$\bar{b}$和$\tilde{b}$）拼接[22, 33]。序列模型和树模型都进行这种增强。

$$
m_a = [\bar{a}; \tilde{a}; \bar{a} - \tilde{a}; \bar{a} \odot \tilde{a}] \qquad (14)
$$

$$
m_b = [\bar{b}; \tilde{b}; \bar{b} - \tilde{b}; \bar{b} \odot \tilde{b}] \qquad (15)
$$

这个过程可以被视为对元组元素之间的某些高阶交互进行建模的一种特例。沿着这个方向，我们还尝试通过将元组输入前馈神经网络，并将顶层隐藏状态添加到上述拼接中来进一步建模交互。我们发现这对验证数据集上的推理准确性没有进一步的帮助。

### 3.3 推理组合

为了确定前提和假设之间的整体推理关系，我们探索了一个组合层，用于组合增强后的局部推理信息$m_a$和$m_b$。我们分别使用BiLSTM和树LSTM在序列或其解析上下文中进行组合。

**组合层。** 在我们的序列推理模型中，我们持续使用BiLSTM来序列地组合局部推理信息。BiLSTM的公式在形式上与公式(1)和(2)相似，因此我们跳过了细节，但其目的在此处非常不同——它们在此用于捕获局部推理信息$m_a$和$m_b$及其上下文以进行推理组合。

在树组合中，树节点更新以组合局部推理的高级公式如下：

$$
v_{a,t} = \text{TrLSTM}(F(m_{a,t}), h_{t-1}^L, h_{t-1}^R) \qquad (16)
$$

$$
v_{b,t} = \text{TrLSTM}(F(m_{b,t}), h_{t-1}^L, h_{t-1}^R) \qquad (17)
$$

我们提出在此层控制模型复杂度，因为计算$m_a$和$m_b$的拼接可能显著增加整体参数量，可能导致过拟合。我们建议使用映射$F$，如公式(16)和(17)所示。更具体地说，我们使用一个带ReLU激活的单层前馈神经网络。该函数也应用于我们的序列推理组合中的BiLSTM。

**池化。** 我们的推理模型将上述得到的向量通过池化转换为固定长度的向量，并将其输入最终分类器以确定整体推理关系。

我们认为求和[6]可能对序列长度敏感，因此不够鲁棒。我们建议采用以下策略：同时计算平均池化和最大池化，并将所有这些向量拼接形成最终的固定长度向量$v$。实验表明，这导致比求和显著更好的结果。最终固定长度向量$v$计算如下：

$$
v_{a,ave} = \frac{\sum_{i=1}^{\ell_a} v_{a,i}}{\ell_a}, \quad v_{a,max} = \max_{i=1}^{\ell_a} v_{a,i} \qquad (18)
$$

$$
v_{b,ave} = \frac{\sum_{j=1}^{\ell_b} v_{b,j}}{\ell_b}, \quad v_{b,max} = \max_{j=1}^{\ell_b} v_{b,j} \qquad (19)
$$

$$
v = [v_{a,ave}; v_{a,max}; v_{b,ave}; v_{b,max}] \qquad (20)
$$

注意，对于树组合，公式(20)与序列组合略有不同。我们的树组合还会拼接通过公式(16)和(17)计算的根节点的隐藏状态，此处未显示。

然后我们将$v$输入最终的多层感知机（MLP）分类器。在我们的实验中，MLP有一个带tanh激活的隐藏层和softmax输出层。整个模型（上述三个组件）进行端到端训练。训练时，我们使用多类交叉熵损失。

**整体推理模型。** 我们的模型可以仅基于序列网络，移除所有树组件，我们称之为增强型序列推理模型（ESIM）（见图1左侧）。我们将证明ESIM超越了所有先前的结果。我们还将如所述地在多个层中使用树LSTM编码解析信息（见图1右侧）。我们训练这个模型，并通过平均预测概率将其融入ESIM中，以获得前提-假设对的最终标签。我们将证明解析信息与ESIM很好地互补并进一步提高性能，我们将最终模型称为混合推理模型（HIM）。

## 4 实验设置

**数据。** 斯坦福自然语言推理（SNLI）语料库[1]关注前提和潜在假设之间的三种基本关系：前提蕴含假设（蕴含）、彼此矛盾（矛盾）、或不相关（中性）。原始SNLI语料库还包含"其他"类别，包括多位人工标注者缺乏共识的句子对。与相关工作一致，我们移除了这一类别。我们使用了与[1]和其他先前工作相同的划分。

本文使用斯坦福PCFG解析器3.5.3[34]生成的解析树，作为SNLI语料库的一部分提供。我们使用分类准确率作为评估指标，与相关工作一致。

**训练。** 我们使用开发集来选择用于测试的模型。为帮助复现我们的结果，我们公开了代码¹。下面列出我们的训练细节。我们使用Adam方法[35]进行优化。一阶动量设为0.9，二阶动量设为0.999。初始学习率为0.0004，批大小为32。LSTM、树LSTM和词嵌入的所有隐藏状态均为300维。

我们使用丢弃率为0.5的dropout，应用于所有前馈连接。我们使用预训练的300维Glove 840B向量[36]初始化词嵌入。词汇表外（OOV）词使用高斯样本随机初始化。所有向量（包括词嵌入）在训练期间更新。

## 5 结果

**整体性能。** 表1展示了不同模型的结果。第一行是Bowman等人[1]提出的基线分类器，考虑了手工特征，如假设相对于前提的BLEU分数、重叠词以及它们之间的长度差等。

下一组模型(2)-(7)基于句子编码。[4]的模型用两个不同的LSTM编码前提和假设。[21]的模型在GRU编码器中使用无监督"skip-thoughts"预训练。[22]的方法考虑基于树的CNN捕获句子级语义，而[4]的模型引入了栈增强解析器-解释器神经网络（SPINN），这是一种树-序列混合模型。[23]的工作使用BiLSTM生成句子表示，然后用内部注意力替换平均池化。[26]的方法提出了一种记忆增强神经网络——神经语义编码器（NSE）来编码句子。

表中的下一组方法(8)-(15)是基于句子间注意力的模型。[14]的模型是强制所谓逐词注意力的LSTM。[24]的模型扩展了这一思想，在假设和前提之间显式强制逐词匹配。具有深度注意力融合的长短期记忆网络（LSTMN）[25]将当前词与存储在记忆中的先前词链接起来。[6]提出了一种不依赖任何词序信息的可分解注意力模型。一般来说，添加句子内注意力带来了进一步的改进，这并不令人惊讶，因为它有助于对齐前提和假设之间的相关文本片段。[5]的模型将[24]的框架扩展到完整的n元树模型，并取得了进一步的改进。[7]提出了一种特殊的LSTM变体，将另一句子的注意力向量视为LSTM的内部状态。[8]使用带有完整二叉树LSTM编码器（无句法信息）的神经架构。

**表1: 各模型在SNLI上的准确率。** 我们的最终模型达到了88.6%的准确率，这是SNLI上观察到的最佳结果，而我们的增强型序列编码模型达到了88.0%的准确率，同样超越了先前模型。

| 模型 | #参数 | 训练 | 测试 |
|------|-------|------|------|
| (1) 手工特征[1] | - | 99.7 | 78.2 |
| (2) 300D LSTM编码器[4] | 3.0M | 83.9 | 80.6 |
| (3) 1024D预训练GRU编码器[21] | 15M | 98.8 | 81.4 |
| (4) 300D 基于树的CNN编码器[22] | 3.5M | 83.3 | 82.1 |
| (5) 300D SPINN-PI编码器[4] | 3.7M | 89.2 | 83.2 |
| (6) 600D BiLSTM内部注意力编码器[23] | 2.8M | 84.5 | 84.2 |
| (7) 300D NSE编码器[26] | 3.0M | 86.2 | 84.6 |
| (8) 100D 注意力LSTM[14] | 250K | 85.3 | 83.5 |
| (9) 300D mLSTM[24] | 1.9M | 92.0 | 86.1 |
| (10) 450D 深度注意力融合LSTMN[25] | 3.4M | 88.5 | 86.3 |
| (11) 200D 可分解注意力模型[6] | 380K | 89.5 | 86.3 |
| (12) 句子内注意力 + (11)[6] | 580K | 90.5 | 86.8 |
| (13) 300D NTI-SLSTM-LSTM[5] | 3.2M | 88.5 | 87.3 |
| (14) 300D 重读LSTM[7] | 2.0M | 90.7 | 87.5 |
| (15) 300D btree-LSTM编码器[8] | 2.0M | 88.6 | 87.6 |
| **(16) 600D ESIM** | 4.3M | **92.6** | **88.0** |
| **(17) HIM (600D ESIM + 300D 句法树LSTM)** | 7.7M | **93.5** | **88.6** |

表格显示，我们的ESIM模型达到了88.0%的准确率，已超越所有先前模型，包括那些使用更复杂网络架构的模型[5]。

我们将ESIM模型与基于句法解析树的句法树LSTM[29]集成，在我们最好的序列编码模型ESIM之上获得了显著提升，达到了88.6%的准确率。这表明句法树LSTM与ESIM很好地互补。

**表2: 模型消融性能。**

| 模型 | 训练 | 测试 |
|------|------|------|
| (17) HIM (ESIM + 句法树) | 93.5 | 88.6 |
| (18) ESIM + tree | 91.9 | 88.2 |
| (16) ESIM | 92.6 | 88.0 |
| (19) ESIM - ave./max | 92.9 | 87.1 |
| (20) ESIM - diff./prod. | 91.5 | 87.0 |
| (21) ESIM - 推理BiLSTM | 91.3 | 87.3 |
| (22) ESIM - 编码BiLSTM | 88.7 | 86.3 |
| (23) ESIM - 基于前提的注意力 | 91.6 | 87.2 |
| (24) ESIM - 基于假设的注意力 | 91.4 | 86.5 |
| (25) 句法树 | 92.9 | 87.8 |

**消融分析。** 我们进一步分析帮助取得良好性能的主要组件。从最佳模型开始，我们首先将句法树LSTM替换为不编码句法解析信息的全树LSTM。更具体地说，句子中相邻的两个词被合并形成一个父节点，这个过程持续进行，最终形成一个完整的二叉树，当没有足够的叶子形成完整树时插入填充节点。每个树节点使用与模型(17)相同的树LSTM块[29]实现。表2显示，此替换后性能下降至88.2%。

此外，我们注意到第3.2节中增强局部推理信息的层和第3.3节中推理组合中的池化层的重要性。表2表明NLI任务对这些层非常敏感。如果我们移除推理组合中的池化层并将其替换为如[6]的求和，准确率降至87.1%。如果从局部推理增强层中移除差和逐元素积，准确率降至87.0%。为了与[6]进行详细比较，将推理组合中的双向LSTM以及输入编码替换为前馈神经网络，准确率分别降至87.3%和86.3%。

ESIM与表2中列出的其他每个模型之间的差异在99%置信水平下的单尾配对t检验中具有统计显著性。模型(17)和(18)之间的差异在同一水平上也是显著的。注意，我们无法对我们的模型与表1中列出的其他模型进行显著性检验，因为我们没有其他模型的输出。

如果我们从ESIM中移除基于前提的注意力（模型23），测试集准确率降至87.2%。基于前提的注意力是指系统在读取前提中的词时，使用软注意力考虑假设中的所有相关词。移除基于假设的注意力（模型24）将准确率降至86.5%，其中基于假设的注意力是对句子对另一个方向执行的注意力。结果表明，移除基于假设的注意力对我们模型性能的影响更大，但移除另一个方向的注意力也会损害性能。

独立的句法树LSTM模型达到了87.8%的准确率，与ESIM相当。我们还计算了融合句法树LSTM和ESIM的oracle分数，即如果任一模型正确则选择正确答案。该oracle/上界在测试集上的准确率为91.7%，这说明了树LSTM和ESIM在理想情况下能互补的程度。就速度而言，在Nvidia-Tesla K40M上训练树LSTM约需40小时，ESIM约需6小时，这可以轻松扩展到更大规模的数据。

**进一步分析。** 我们证明了编码句法解析信息有助于识别自然语言推理——它进一步改进了已经在很强的系统。图3展示了一个树LSTM做出不同且正确决策的例子。在子图(d)中，节点9和10上输入门的值较大，表明这些节点在做出最终决策时很重要。我们观察到，在子图(c)中，节点9和10与前提中的节点29对齐。这类信息帮助系统判断这对是矛盾关系。相应地，在序列BiLSTM的子图(e)中，词sitting和down在最终决策中并不起重要作用。子图(f)显示sitting与reading和standing的注意力相等，而down的对齐并不那么有用。

**图3: 分析示例。** (a)和(b)分别是前提和假设的成分解析树。"-"表示非叶子或空节点。(c)和(f)分别是树模型和ESIM的注意力可视化。颜色越深，值越大。前提在x轴上，假设在y轴上。(d)和(e)分别是推理组合中树LSTM和BiLSTM输入门的l2范数。

## 6 结论与未来工作

我们提出了用于自然语言推理的神经网络模型，在SNLI基准上取得了最佳报告结果。这些结果首先是通过我们增强的序列推理模型实现的，该模型超越了先前模型，包括那些使用更复杂网络架构的模型，表明序列推理模型的潜力尚未被充分发掘。在此基础上，我们进一步表明，通过在局部推理建模和推理组合中显式考虑递归架构，我们获得了额外的改进。特别是，融入句法解析信息有助于我们取得最佳结果：即使添加到已经非常强大的模型中，它也能进一步提升性能。

未来的有趣工作包括探索外部资源（如WordNet和对比含义嵌入[37]）以帮助增加词级推理关系的覆盖面。在神经网络框架内更仔细地建模否定[38, 39]可能有助于矛盾检测。

## 致谢

本文第一和第三作者部分得到安徽省科技发展计划（项目编号2014z02006）、中央高校基本科研业务费专项资金（项目编号WK2350000001）和中国科学院战略性先导科技专项（项目编号XDB02070006）的支持。

## 参考文献

[1] Samuel Bowman, Gabor Angeli, Christopher Potts, and D. Christopher Manning. A large annotated corpus for learning natural language inference. In *Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing*, pages 632–642, 2015.

[2] Bill MacCartney and Christopher D. Manning. Modeling semantic containment and exclusion in natural language inference. In *Proceedings of the 22nd International Conference on Computational Linguistics - Volume 1*, pages 521–528, 2008.

[3] Bill MacCartney. Natural Language Inference. PhD thesis, Stanford University, 2009.

[4] Samuel Bowman, Jon Gauthier, Abhinav Rastogi, Raghav Gupta, D. Christopher Manning, and Christopher Potts. A fast unified model for parsing and sentence understanding. In *Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 1466–1477, 2016.

[5] Tsendsuren Munkhdalai and Hong Yu. Neural tree indexers for text understanding. *CoRR*, abs/1607.04492, 2016.

[6] Ankur Parikh, Oscar Täckström, Dipanjan Das, and Jakob Uszkoreit. A decomposable attention model for natural language inference. In *Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing*, pages 2249–2255, 2016.

[7] Lei Sha, Baobao Chang, Zhifang Sui, and Sujian Li. Reading and thinking: Re-read LSTM unit for textual entailment recognition. In *Proceedings of COLING 2016*, pages 2870–2879, 2016.

[8] Biswajit Paria, K. M. Annervaz, Ambedkar Dukkipati, Ankush Chatterjee, and Sanjay Podder. A neural architecture mimicking humans end-to-end for natural language inference. *CoRR*, abs/1611.04741, 2016.

[9] Barbara Partee. Lexical semantics and compositionality. *Invitation to Cognitive Science*, 1:311–360, 1995.

[10] Yashar Mehdad, Alessandro Moschitti, and Massimo Fabio Zanzotto. Syntactic/semantic structures for textual entailment recognition. In *HLT-NAACL*, pages 1020–1028, 2010.

[11] Lorenzo Ferrone and Massimo Fabio Zanzotto. Towards syntax-aware compositional distributional semantic models. In *Proceedings of COLING 2014*, pages 721–730, 2014.

[12] Ido Dagan, Oren Glickman, and Bernardo Magnini. The PASCAL recognising textual entailment challenge. In *MLCW*, pages 177–190, 2005.

[13] Adrian Iftene and Alexandra Balahur-Dobrescu. Hypothesis Transformation and Semantic Variability Rules Used in Recognizing Textual Entailment. In *Proceedings of the ACL-PASCAL Workshop on Textual Entailment and Paraphrasing*, pages 125–130, 2007.

[14] Tim Rocktäschel, Edward Grefenstette, Karl Moritz Hermann, Tomás Kociský, and Phil Blunsom. Reasoning about entailment with neural attention. *CoRR*, abs/1509.06664, 2015.

[15] Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural machine translation by jointly learning to align and translate. *CoRR*, abs/1409.0473, 2014.

[16] Jan Chorowski, Dzmitry Bahdanau, Dmitriy Serdyuk, Kyunghyun Cho, and Yoshua Bengio. Attention-based models for speech recognition. In *NIPS*, pages 577–585, 2015.

[17] William Chan, Navdeep Jaitly, Quoc V. Le, and Oriol Vinyals. Listen, attend and spell: A neural network for large vocabulary conversational speech recognition. In *ICASSP*, pages 4960–4964, 2016.

[18] Kelvin Xu, Jimmy Ba, Ryan Kiros, Kyunghyun Cho, Aaron C. Courville, Ruslan Salakhutdinov, Richard S. Zemel, and Yoshua Bengio. Show, attend and tell: Neural image caption generation with visual attention. In *ICML*, pages 2048–2057, 2015.

[19] Alexander Rush, Sumit Chopra, and Jason Weston. A neural attention model for abstractive sentence summarization. In *EMNLP*, pages 379–389, 2015.

[20] Qian Chen, Xiaodan Zhu, Zhenhua Ling, Si Wei, and Hui Jiang. Distraction-based neural networks for modeling document. In *IJCAI*, pages 2754–2760, 2016.

[21] Ivan Vendrov, Ryan Kiros, Sanja Fidler, and Raquel Urtasun. Order-embeddings of images and language. *CoRR*, abs/1511.06361, 2015.

[22] Lili Mou, Rui Men, Ge Li, Yan Xu, Lu Zhang, Rui Yan, and Zhi Jin. Natural language inference by tree-based convolution and heuristic matching. In *ACL (Volume 2: Short Papers)*, pages 130–136, 2016.

[23] Yang Liu, Chengjie Sun, Lei Lin, and Xiaolong Wang. Learning natural language inference using bidirectional LSTM model and inner-attention. *CoRR*, abs/1605.09090, 2016.

[24] Shuohang Wang and Jing Jiang. Learning natural language inference with LSTM. In *NAACL-HLT*, pages 1442–1451, 2016.

[25] Jianpeng Cheng, Li Dong, and Mirella Lapata. Long short-term memory-networks for machine reading. In *EMNLP*, pages 551–561, 2016.

[26] Tsendsuren Munkhdalai and Hong Yu. Neural semantic encoders. *CoRR*, abs/1607.04315, 2016.

[27] Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. *Neural Computation*, 9(8):1735–1780, 1997.

[28] Kyunghyun Cho, Bart van Merrienboer, Dzmitry Bahdanau, and Yoshua Bengio. On the properties of neural machine translation: Encoder-decoder approaches. In *SSST@EMNLP*, pages 103–111, 2014.

[29] Xiaodan Zhu, Parinaz Sobhani, and Hongyu Guo. Long short-term memory over recursive structures. In *ICML*, pages 1604–1612, 2015.

[30] Sheng Kai Tai, Richard Socher, and D. Christopher Manning. Improved semantic representations from tree-structured long short-term memory networks. In *ACL (Volume 1: Long Papers)*, pages 1556–1566, 2015.

[31] Phong Le and Willem Zuidema. Compositional distributional semantics with long short term memory. In *Proceedings of the Fourth Joint Conference on Lexical and Computational Semantics*, pages 10–19, 2015.

[32] Richard Socher, Cliff Chiung-Yu Lin, Andrew Y. Ng, and Christopher D. Manning. Parsing natural scenes and natural language with recursive neural networks. In *ICML*, pages 129–136, 2011.

[33] Junbei Zhang, Xiaodan Zhu, Qian Chen, Lirong Dai, Si Wei, and Hui Jiang. Exploring question understanding and adaptation in neural-network-based question answering. *CoRR*, abs/1703.04617, 2017.

[34] Dan Klein and Christopher D. Manning. Accurate unlexicalized parsing. In *ACL*, 2003.

[35] Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. *CoRR*, abs/1412.6980, 2014.

[36] Jeffrey Pennington, Richard Socher, and Christopher Manning. GloVe: Global vectors for word representation. In *EMNLP*, pages 1532–1543, 2014.

[37] Zhigang Chen, Wei Lin, Qian Chen, Xiaoping Chen, Si Wei, Hui Jiang, and Xiaodan Zhu. Revisiting word embedding for contrasting meaning. In *ACL (Volume 1: Long Papers)*, pages 106–115, 2015.

[38] Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, D. Christopher Manning, Andrew Ng, and Christopher Potts. Recursive deep models for semantic compositionality over a sentiment treebank. In *EMNLP*, pages 1631–1642, 2013.

[39] Xiaodan Zhu, Hongyu Guo, Saif Mohammad, and Svetlana Kiritchenko. An empirical study on the effect of negation words on sentiment. In *ACL (Volume 1: Long Papers)*, pages 304–313, 2014.
