# Improving Language Understanding by Generative Pre-Training

> Alec Radford | OpenAI | alec@openai.com
> Karthik Narasimhan | OpenAI | karthikn@openai.com
> Tim Salimans | OpenAI | tim@openai.com
> Ilya Sutskever | OpenAI | ilyasu@openai.com

自然语言理解包含广泛且多样的任务，如文本蕴含、问答、语义相似度评估和文档分类。尽管大量无标注文本语料库很丰富，但用于学习这些特定任务的标注数据却很稀缺，这使得判别式训练的模型难以充分执行。我们证明，通过在多样化的无标注文本语料库上对语言模型进行生成式预训练，然后在每个特定任务上进行判别式微调，可以在这些任务上实现显著的提升。与之前的方法相比，我们在微调期间使用任务感知的输入转换，以在需要对模型架构进行最小更改的情况下实现有效的迁移。我们在广泛的自然语言理解基准上证明了我们方法的有效性。我们通用的、与任务无关的模型优于使用为每个任务专门设计的架构的判别式训练模型，在所研究的12个任务中有9个显著提升了最先进水平。例如，我们在常识推理（Stories Cloze Test）上取得了8.9%的绝对提升，在问答（RACE）上取得了5.7%的绝对提升，在文本蕴含（MultiNLI）上取得了1.5%的绝对提升。

---

## 摘要

自然语言理解包含广泛且多样的任务，如文本蕴含、问答、语义相似度评估和文档分类。尽管大量无标注文本语料库很丰富，但用于学习这些特定任务的标注数据却很稀缺，这使得判别式训练的模型难以充分执行。我们证明，通过在多样化的无标注文本语料库上对语言模型进行生成式预训练，然后在每个特定任务上进行判别式微调，可以在这些任务上实现显著的提升。与之前的方法相比，我们在微调期间使用任务感知的输入转换，以在需要对模型架构进行最小更改的情况下实现有效的迁移。我们在广泛的自然语言理解基准上证明了我们方法的有效性。我们通用的、与任务无关的模型优于使用为每个任务专门设计的架构的判别式训练模型，在所研究的12个任务中有9个显著提升了最先进水平。例如，我们在常识推理（Stories Cloze Test）上取得了8.9%的绝对提升，在问答（RACE）上取得了5.7%的绝对提升，在文本蕴含（MultiNLI）上取得了1.5%的绝对提升。

## 1 引言

从原始文本中有效学习的能力对于减轻自然语言处理（NLP）中对监督学习的依赖至关重要。大多数深度学习方法需要大量的人工标注数据，这限制了它们在许多缺乏标注资源的领域中的适用性[61]。在这些情况下，能够利用无标注数据中的语言信息的模型为收集更多标注——这既耗时又昂贵——提供了有价值的替代方案。此外，即使在有大量监督可用的情况下，以无监督方式学习良好的表示也能提供显著的性能提升。迄今为止，最令人信服的证据是广泛使用预训练的词嵌入（word embeddings）[10, 39, 42]来提高一系列NLP任务的性能[8, 11, 26, 45]。

然而，利用无标注文本中超过词级别的信息面临两个主要挑战。首先，尚不清楚哪种类型的优化目标最有效地学习对迁移有用的文本表示。最近的研究关注了各种目标，如语言建模[44]、机器翻译[38]和篇章连贯性[22]，每种方法在不同任务上优于其他方法。其次，对于将这些学到的表示迁移到目标任务的最有效方式尚未达成共识。现有技术包括对模型架构进行特定于任务的更改[43, 44]、使用复杂的学习方案[21]以及添加辅助学习目标[50]。这些不确定性使得为语言处理开发有效的半监督学习方法变得困难。

在本文中，我们探索了一种结合无监督预训练和监督微调的语言理解任务半监督方法。我们的目标是学习一种通用表示，只需很少的调整就能迁移到广泛的任务中。我们假设可以访问一个大型的无标注文本语料库以及几个带人工标注训练示例的数据集（目标任务）。我们的设置不要求这些目标任务与无标注语料库处于同一领域。我们采用两阶段训练过程。首先，我们在无标注数据上使用语言建模目标来学习神经网络模型的初始参数。随后，我们使用相应的监督目标将这些参数调整到目标任务。

对于我们的模型架构，我们使用Transformer [62]，该架构在机器翻译[62]、文档生成[34]和句法分析[29]等各种任务上表现出了强大的性能。与递归网络等替代方案相比，这种模型选择为我们提供了更结构化的记忆来处理文本中的长距离依赖，从而在多样化任务中产生稳健的迁移性能。在迁移期间，我们利用源自遍历式方法（traversal-style approaches）[52]的特定于任务的输入适配，将结构化文本输入处理为单个连续的token序列。正如我们在实验中所展示的，这些适配使我们能够在最小改变预训练模型架构的情况下进行有效的微调。

我们在四类语言理解任务上评估我们的方法：自然语言推理、问答、语义相似度和文本分类。我们通用的、与任务无关的模型优于使用为每个任务专门设计的架构的判别式训练模型，在所研究的12个任务中有9个显著提升了最先进水平。例如，我们在常识推理（Stories Cloze Test）[40]上取得了8.9%的绝对提升，在问答（RACE）[30]上取得了5.7%的绝对提升，在文本蕴含（MultiNLI）[66]上取得了1.5%的绝对提升，在最近引入的GLUE多任务基准[64]上取得了5.5%的绝对提升。我们还分析了预训练模型在四种不同设置下的零样本行为，并证明它获得了对下游任务有用的语言知识。

## 2 相关工作

### 用于NLP的半监督学习

我们的工作广泛属于自然语言半监督学习的范畴。这一范式吸引了大量关注，其应用包括序列标注[24, 33, 57]或文本分类[41, 70]等任务。最早的方法使用无标注数据来计算词级或短语级统计量，然后将其用作监督模型中的特征[33]。在过去的几年里，研究人员证明了使用词嵌入[11, 39, 42]（在无标注语料库上训练）来改善各种任务性能的好处[8, 11, 26, 45]。然而，这些方法主要迁移词级信息，而我们旨在捕获更高层次的语义。

最近的方法研究了从无标注数据中学习和利用超越词级别的语义。短语级或句子级嵌入，可以在无标注语料库上训练，已被用于将文本编码为适合各种目标任务的向量表示[28, 32, 1, 36, 22, 12, 56, 31]。

### 无监督预训练

无监督预训练是半监督学习的一个特例，其目标是找到一个好的初始化点，而不是修改监督学习目标。早期的工作探索了该技术在图像分类[20, 49, 63]和回归任务[3]中的应用。随后的研究[15]证明，预训练作为一种正则化方案，能够在深度神经网络中实现更好的泛化。在最近的工作中，该方法已被用于帮助训练各种任务的深度神经网络，如图像分类[69]、语音识别[68]、实体消歧[17]和机器翻译[48]。

与我们最接近的工作涉及使用语言建模目标预训练神经网络，然后在目标任务上进行有监督微调。Dai等人[13]以及Howard和Ruder [21]遵循这种方法来改进文本分类。然而，尽管预训练阶段有助于捕获一些语言信息，但他们使用LSTM模型限制了其预测能力到短距离。相比之下，我们选择的Transformer网络使我们能够捕获更长距离的语言结构，正如我们的实验所证明的那样。此外，我们还证明了我们的模型在更广泛的任务上的有效性，包括自然语言推理、 paraphrase检测和故事补全。其他方法[43, 44, 38]在目标任务上训练监督模型时，使用预训练语言或机器翻译模型的隐藏表示作为辅助特征。这需要为每个单独的目标任务引入大量新参数，而我们在迁移期间只需要对模型架构进行最小程度的更改。

### 辅助训练目标

添加辅助的无监督训练目标是半监督学习的另一种形式。Collobert和Weston [10]的早期工作使用了各种辅助NLP任务，如词性标注、组块分析、命名实体识别和语言建模来改善语义角色标注。最近，Rei [50]在目标任务目标中添加了辅助语言建模目标，并证明了在序列标注任务上的性能提升。我们的实验也使用辅助目标，但正如我们所示，无监督预训练已经学习了与目标任务相关的几个语言方面。

## 3 框架

我们的训练过程包括两个阶段。第一阶段是在大型文本语料库上学习高容量语言模型。随后是微调阶段，在此阶段我们将模型适应于带有标注数据的判别式任务。

### 3.1 无监督预训练

给定一个无监督的token语料库 $U = \{u_1, \ldots, u_n\}$，我们使用标准的语言建模目标来最大化以下似然：

$$L_1(U) = \sum_i \log P(u_i|u_{i-k}, \ldots, u_{i-1}; \Theta) \qquad (1)$$

其中 $k$ 是上下文窗口的大小，条件概率 $P$ 使用参数为 $\Theta$ 的神经网络建模。这些参数使用随机梯度下降[51]进行训练。

在我们的实验中，我们使用多层Transformer解码器[34]作为语言模型，它是Transformer [62]的一个变体。该模型对输入上下文token应用多头自注意力操作，然后接位置逐位的前馈层，以生成目标token上的输出分布：

$$
\begin{aligned}
h_0 &= UW_e + W_p \\
h_l &= \text{transformer\_block}(h_{l-1}) \quad \forall i \in [1, n] \\
P(u) &= \text{softmax}(h_n W_e^T)
\end{aligned} \qquad (2)$$

其中 $U = (u_{-k}, \ldots, u_{-1})$ 是上下文的token向量，$n$ 是层数，$W_e$ 是token嵌入矩阵，$W_p$ 是位置嵌入矩阵。

### 3.2 监督微调

在使用式(1)中的目标训练模型之后，我们将参数调整到监督目标任务。我们假设有一个标注数据集 $C$，其中每个实例由一个输入token序列 $x_1, \ldots, x_m$ 和一个标签 $y$ 组成。输入通过我们预训练的模型，获得最终Transformer块的激活 $h_l^m$，然后将其输入到一个新增的、参数为 $W_y$ 的线性输出层以预测 $y$：

$$P(y|x_1, \ldots, x_m) = \text{softmax}(h_l^m W_y) \qquad (3)$$

这给了我们以下要最大化的目标：

$$L_2(C) = \sum_{(x, y)} \log P(y|x_1, \ldots, x_m) \qquad (4)$$

我们还发现，在微调中加入语言建模作为辅助目标有助于学习，其方式为：(a) 提高监督模型的泛化能力，以及 (b) 加速收敛。这与之前的工作[50, 43]一致，他们也观察到使用这种辅助目标能改善性能。具体来说，我们优化以下目标（权重为 $\lambda$）：

$$L_3(C) = L_2(C) + \lambda * L_1(C) \qquad (5)$$

总的来说，我们在微调期间需要的唯一额外参数是 $W_y$ 以及分隔token的嵌入（下面第3.3节描述）。

![图1](figure1.png)

**图1：(左)** 本文使用的Transformer架构和训练目标。**(右)** 针对不同任务微调的输入转换。我们将所有结构化输入转换为token序列，由我们的预训练模型处理，然后接一个线性+softmax层。

### 3.3 特定于任务的输入转换

对于某些任务，如文本分类，我们可以直接按照上述方式微调我们的模型。其他某些任务，如问答或文本蕴含，具有结构化输入，例如有序句子对，或文档、问题和答案的三元组。由于我们的预训练模型是在连续的文本序列上训练的，我们需要进行一些修改才能将其应用于这些任务。之前的工作提出在迁移表示之上学习特定于任务的架构[44]。这种方法重新引入了大量特定于任务的定制化，并且没有对这些额外的架构组件使用迁移学习。相反，我们使用遍历式方法（traversal-style approach）[52]，将结构化输入转换为我们的预训练模型可以处理的有序序列。这些输入转换使我们避免了跨任务对架构进行大量更改。下面我们简要描述这些输入转换，图1提供了可视化说明。所有转换都包括添加随机初始化的起始和结束token（$\langle s \rangle$, $\langle e \rangle$）。

**文本蕴含** 对于蕴含任务，我们将前提 $p$ 和假设 $h$ 的token序列连接起来，中间用一个分隔token（$）隔开。

**相似度** 对于相似度任务，被比较的两个句子没有固有的顺序。为了反映这一点，我们修改输入序列以包含两种可能的句子顺序（中间用分隔符），并独立处理每个序列以产生两个序列表示 $h_l^m$，它们在送入线性输出层之前进行逐元素相加。

**问答和常识推理** 对于这些任务，我们给定一个上下文文档 $z$、一个问题 $q$ 和一组可能的答案 $\{a_k\}$。我们将文档上下文和问题与每个可能的答案连接起来，中间添加一个分隔token，得到 $[z; q; \$; a_k]$。这些序列各自独立地通过我们的模型处理，然后通过softmax层进行归一化，以产生在可能的答案上的输出分布。

## 4 实验

### 4.1 设置

**无监督预训练** 我们使用BooksCorpus数据集[71]来训练语言模型。它包含来自各种体裁（包括冒险、奇幻和浪漫）的超过7,000本独特的未出版书籍。至关重要的是，它包含长段的连续文本，这使得生成模型能够学习基于长距离信息进行条件建模。另一种类似方法ELMo [44]使用的替代数据集——1B Word Benchmark——大小大致相同，但按句子级别进行了混洗，破坏了长距离结构。我们的语言模型在此语料库上实现了非常低的token级困惑度18.4。

**模型规格** 我们的模型很大程度上遵循原始的Transformer工作[62]。我们训练了一个12层仅解码器的Transformer，带有 masked 自注意力头（768维状态和12个注意力头）。对于位置逐位的前馈网络，我们使用了3072维的内部状态。我们使用了Adam优化方案[27]，最大学习率为2.5e-4。学习率在前2000次更新中从零线性增加，然后使用余弦调度退火至0。我们在大小为64的随机采样连续序列（512个token）的小批量上训练100个epoch。由于层归一化（layernorm）[2]在整个模型中广泛使用，简单的权重初始化 $N(0, 0.02)$ 就足够了。我们使用了具有40,000次合并的字节对编码（BPE）词汇表[53]，以及残差、嵌入和注意力dropout，其比率为0.1用于正则化。我们还采用了[37]中提出的L2正则化的修改版本，在所有非偏置或增益权重上设置 $w = 0.01$。对于激活函数，我们使用了高斯误差线性单元（GELU）[18]。我们使用了学习得到的位置嵌入，而不是原始工作中提出的正弦版本。我们使用ftfy库清理BooksCorpus中的原始文本，标准化一些标点和空白，并使用spaCy分词器。

**微调细节** 除非另有说明，我们重用无监督预训练的超参数设置。我们向分类器添加dropout，比率为0.1。对于大多数任务，我们使用学习率6.25e-5和批量大小32。我们的模型微调很快，对于大多数情况，3个epoch的训练就足够了。我们使用线性学习率衰减调度，预热占训练的0.2%。$\lambda$ 设置为0.5。

**表1：** 我们实验中使用的不同任务和数据集列表。

| 任务 | 数据集 |
|------|--------|
| 自然语言推理 | SNLI [5], MultiNLI [66], Question NLI [64], RTE [4], SciTail [25] |
| 问答 | RACE [30], Story Cloze [40] |
| 句子相似度 | MSR Paraphrase Corpus [14], Quora Question Pairs [9], STS Benchmark [6] |
| 分类 | Stanford Sentiment Treebank-2 [54], CoLA [65] |

### 4.2 监督微调

我们在各种监督任务上进行实验，包括自然语言推理、问答、语义相似度和文本分类。其中一些任务是最近发布的GLUE多任务基准[64]的一部分，我们使用到了它。图1提供了所有任务和数据集的概述。

**自然语言推理** 自然语言推理（NLI）任务，也称为识别文本蕴含，涉及阅读一对句子并判断它们之间的关系，从蕴含、矛盾或中性中选择。尽管最近有很多关注[58, 35, 44]，但由于存在各种现象，如词汇蕴含、共指、以及词汇和句法歧义，该任务仍然具有挑战性。我们在五个来源多样的数据集上进行评估，包括图像标题（SNLI）、转述演讲、流行小说和政府报告（MNLI）、维基百科文章（QNLI）、科学考试（SciTail）或新闻文章（RTE）。

**表2：** 自然语言推理任务的实验结果，将我们的模型与当前最先进方法进行比较。5x表示5个模型的集成。所有数据集使用准确率作为评估指标。

| 方法 | MNLI-m | MNLI-mm | SNLI | SciTail | QNLI | RTE |
|------|--------|---------|------|---------|------|-----|
| ESIM + ELMo [44] (5x) | - | - | 89.3 | - | - | - |
| CAFE [58] (5x) | 80.2 | 79.0 | 89.3 | - | - | - |
| Stochastic Answer Network [35] (3x) | 80.6 | 80.1 | - | - | - | - |
| CAFE [58] | 78.7 | 77.9 | 88.5 | 83.3 | - | - |
| GenSen [64] | 71.4 | 71.3 | - | - | 82.3 | 59.2 |
| Multi-task BiLSTM + Attn [64] | 72.2 | 72.1 | - | - | 82.1 | 61.7 |
| Finetuned Transformer LM (ours) | **82.1** | **81.4** | **89.9** | **88.3** | **88.1** | 56.0 |

我们的方法显著优于五个数据集中四个的基线，在MNLI上取得了高达1.5%的绝对提升，在SciTail上为5%，在QNLI上为5.8%，在SNLI上为0.6%，超过了之前的最佳结果。这证明了我们的模型更好地推理多个句子和处理语言歧义方面的能力。在RTE上，这是我们评估的较小数据集之一（2490个示例），我们获得了56%的准确率，低于多任务biLSTM模型报告的61.7%。鉴于我们的方法在更大NLI数据集上的强劲性能，我们的模型很可能也会从多任务训练中受益，但我们目前尚未探索这一点。

**问答和常识推理** 另一个需要单句和多句推理方面的任务是问答。我们使用最近发布的RACE数据集[30]，其中包含来自中学和高中考试的英语段落及相关问题。该语料库已被证明包含比其他数据集（如CNN [19]或SQuaD [47]）更多推理类型的问题，为我们训练用于处理长距离上下文的模型提供了完美的评估。此外，我们在Story Cloze Test [40]上进行了评估，该测试涉及从两个选项中选择多句故事的正确结尾。在这些任务上，我们的模型再次以显著的优势超越了之前的最佳结果——在Story Cloze上高达8.9%，在RACE上整体达到5.7%。这证明了我们的模型有效处理长距离上下文的能力。

**表3：** 问答和常识推理的结果，将我们的模型与当前最先进方法进行比较。9x表示9个模型的集成。

| 方法 | Story Cloze | RACE-m | RACE-h | RACE |
|------|-------------|--------|--------|------|
| val-LS-skip [55] | 76.5 | - | - | - |
| Hidden Coherence Model [7] | 77.6 | - | - | - |
| Dynamic Fusion Net [67] (9x) | - | 55.6 | 49.4 | 51.2 |
| BiAttention MRU [59] (9x) | - | 60.2 | 50.3 | 53.3 |
| Finetuned Transformer LM (ours) | **86.5** | **62.9** | **57.4** | **59.0** |

**语义相似度** 语义相似度（或paraphrase检测）任务涉及预测两个句子在语义上是否等价。挑战在于识别概念的改写、理解否定以及处理句法歧义。我们使用三个数据集进行此任务：Microsoft Paraphrase Corpus (MRPC) [14]（收集自新闻来源）、Quora Question Pairs (QQP) 数据集[9]和Semantic Textual Similarity基准 (STS-B) [6]。我们在三个语义相似度任务中的两个上取得了最先进的结果（表4），在STS-B上获得1个绝对点的提升。在QQP上的性能差异显著，相比于单任务BiLSTM + ELMo + Attn取得了4.2%的绝对提升。

**表4：** 语义相似度和分类结果，将我们的模型与当前最先进方法进行比较。本表中的所有任务评估均使用GLUE基准进行。(mc = Mathews相关系数，acc = 准确率，pc = Pearson相关系数)

| 方法 | 分类 | | 语义相似度 | | | GLUE |
|------|------|---|-----------|----|---|------|
| | CoLA (mc) | SST2 (acc) | MRPC (F1) | STSB (pc) | QQP (F1) | |
| Sparse byte mLSTM [16] | - | 93.2 | - | - | - | - |
| TF-KLD [23] | - | - | 86.0 | - | - | - |
| ECNU (mixed ensemble) [60] | - | - | - | 81.0 | - | - |
| Single-task BiLSTM + ELMo + Attn [64] | 35.0 | 90.2 | 80.2 | 55.5 | 66.1 | 64.8 |
| Multi-task BiLSTM + ELMo + Attn [64] | 18.9 | 91.6 | 83.5 | 72.8 | 63.3 | 68.9 |
| Finetuned Transformer LM (ours) | **45.4** | 91.3 | 82.3 | **82.0** | **70.3** | **72.8** |

**分类** 最后，我们还评估了两个不同的文本分类任务。语言可接受性语料库（CoLA）[65]包含专家对句子是否合乎语法的判断，并测试训练模型的固有语言偏差。另一方面，Stanford Sentiment Treebank (SST-2) [54]是一个标准的二分类任务。我们的模型在CoLA上获得了45.4的得分，相比之前的最佳结果35.0有特别大的飞跃，展示了我们的模型学到的固有语言偏差。该模型在SST-2上也达到了91.3%的准确率，与最先进的结果具有竞争力。我们在GLUE基准上还获得了72.8的总体得分，显著优于之前的最佳68.9。

总的来说，我们的方法在我们评估的12个数据集中有9个达到了新的最先进结果，在许多情况下甚至超越了集成模型。我们的结果还表明，我们的方法在不同规模的数据集上都能良好工作，从较小的数据集如STS-B（约5.7k训练示例）到最大的SNLI（约550k训练示例）。

## 5 分析

**迁移层数的影响** 我们观察了从无监督预训练到监督目标任务迁移可变数量层的影响。图2（左）展示了我们的方法在MultiNLI和RACE上作为迁移层数函数的性能。我们观察到标准结果：迁移嵌入可以提高性能，并且每个Transformer层提供进一步的收益，在MultiNLI上完整迁移时高达9%。这表明预训练模型中的每一层都包含解决目标任务的有用功能。

![图2](figure2.png)

**图2：(左)** 从预训练语言模型向RACE和MultiNLI迁移递增层数的效果。**(右)** 不同任务上零样本性能作为LM预训练更新次数的函数的演化图。每个任务的性能在随机猜测基线和当前单模型最先进水平之间进行了归一化。

**零样本行为** 我们想更好地理解为什么Transformer的语言模型预训练是有效的。一个假设是，底层生成模型学习执行我们评估的许多任务以提高其语言建模能力，并且Transformer的注意力记忆（attentional memory）相比LSTM有助于迁移。我们设计了一系列启发式解决方案，使用底层生成模型在没有监督微调的情况下执行任务。我们在图2（右）中可视化了这些启发式解决方案在生成式预训练过程中的有效性。我们观察到这些启发式的性能是稳定的，并且随着训练稳步增加，这表明生成式预训练支持学习各种与任务相关的功能。我们还观察到LSTM在其零样本性能中表现出更高的方差，这表明Transformer架构的归纳偏置有助于迁移。

对于CoLA（语言可接受性），示例被评分为生成模型分配的平均token对数概率，并通过阈值化进行预测。对于SST-2（情感分析），我们在每个示例后附加token `very`，并将语言模型的输出分布限制为仅 `positive` 和 `negative` 两个词，并猜测它分配更高概率的token作为预测。对于RACE（问答），我们选择生成模型在给定文档和问题条件下分配最高平均token对数概率的答案。对于DPRD [46]（Winograd模式），我们用两个可能的指代替换定指代词，并预测生成模型在替换后对序列其余部分分配更高平均token对数概率的消解结果。

**消融研究** 我们进行了三项不同的消融研究（表5）。首先，我们检查了我们的方法在微调期间不使用辅助LM目标的性能。我们观察到辅助目标对NLI任务和QQP有帮助。总体而言，趋势表明较大的数据集受益于辅助目标，而较小的数据集则不然。其次，我们通过将Transformer与使用相同框架的单层2048单元LSTM进行比较来分析Transformer的效果。我们观察到当使用LSTM代替Transformer时，平均得分下降了5.6。LSTM仅在一个数据集（MRPC）上优于Transformer。最后，我们还将我们的Transformer架构与直接在有监督目标任务上训练（无预训练）进行了比较。我们观察到缺乏预训练会损害所有任务的性能，与我们的完整模型相比下降了14.8%。

**表5：** 不同任务上各种模型消融的分析。平均分是所有结果的未加权平均值。(mc = Mathews相关系数，acc = 准确率，pc = Pearson相关系数)

| 方法 | 平均分 | CoLA (mc) | SST2 (acc) | MRPC (F1) | STSB (pc) | QQP (F1) | MNLI (acc) | QNLI (acc) | RTE (acc) |
|------|--------|-----------|------------|-----------|----------|-------|-----------|-----------|--------|
| Transformer w/ aux LM (完整) | 74.7 | 45.4 | 91.3 | 82.3 | 82.0 | 70.3 | 81.8 | 88.1 | 56.0 |
| Transformer w/o pre-training | 59.9 | 18.9 | 84.0 | 79.4 | 30.9 | 65.5 | 75.7 | 71.2 | 53.8 |
| Transformer w/o aux LM | 75.0 | 47.9 | 92.0 | 84.9 | 83.2 | 69.8 | 81.1 | 86.9 | 54.4 |
| LSTM w/ aux LM | 69.1 | 30.3 | 90.5 | 83.2 | 71.8 | 68.1 | 73.7 | 81.1 | 54.6 |

## 6 结论

我们提出了一个框架，通过生成式预训练和判别式微调，使用单个与任务无关的模型实现了强大的自然语言理解。通过在包含长段连续文本的多样化语料库上进行预训练，我们的模型获得了显著的 world knowledge 和处理长距离依赖的能力，这些能力随后被成功迁移到解决判别式任务中，如问答、语义相似度评估、蕴含判定和文本分类，在我们研究的12个数据集中改进了9个的最先进水平。使用无监督（预）训练来提高判别式任务的性能长期以来一直是机器学习研究的重要目标。我们的工作表明，实现显著的性能提升确实是可能的，并提供了关于哪些模型（Transformers）和数据集（具有长距离依赖的文本）最适合这种方法的线索。我们希望这将有助于推动新的无监督学习研究，无论是对于自然语言理解还是其他领域，进一步提高我们对无监督学习如何以及何时有效的理解。

## 参考文献

[1] S. Arora, Y. Liang, and T. Ma. A simple but tough-to-beat baseline for sentence embeddings. 2016.

[2] J. L. Ba, J. R. Kiros, and G. E. Hinton. Layer normalization. *arXiv preprint arXiv:1607.06450*, 2016.

[3] Y. Bengio, P. Lamblin, D. Popovici, and H. Larochelle. Greedy layer-wise training of deep networks. In *Advances in neural information processing systems*, pages 153–160, 2007.

[4] L. Bentivogli, P. Clark, I. Dagan, and D. Giampiccolo. The fifth pascal recognizing textual entailment challenge. In *TAC*, 2009.

[5] S. R. Bowman, G. Angeli, C. Potts, and C. D. Manning. A large annotated corpus for learning natural language inference. *EMNLP*, 2015.

[6] D. Cer, M. Diab, E. Agirre, I. Lopez-Gazpio, and L. Specia. Semeval-2017 task 1: Semantic textual similarity-multilingual and cross-lingual focused evaluation. *arXiv preprint arXiv:1708.00055*, 2017.

[7] S. Chaturvedi, H. Peng, and D. Roth. Story comprehension for predicting what happens next. In *Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing*, pages 1603–1614, 2017.

[8] D. Chen and C. Manning. A fast and accurate dependency parser using neural networks. In *Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP)*, pages 740–750, 2014.

[9] Z. Chen, H. Zhang, X. Zhang, and L. Zhao. Quora question pairs. https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs, 2018.

[10] R. Collobert and J. Weston. A unified architecture for natural language processing: Deep neural networks with multitask learning. In *Proceedings of the 25th international conference on Machine learning*, pages 160–167. ACM, 2008.

[11] R. Collobert, J. Weston, L. Bottou, M. Karlen, K. Kavukcuoglu, and P. Kuksa. Natural language processing (almost) from scratch. *Journal of Machine Learning Research*, 12(Aug):2493–2537, 2011.

[12] A. Conneau, D. Kiela, H. Schwenk, L. Barrault, and A. Bordes. Supervised learning of universal sentence representations from natural language inference data. *EMNLP*, 2017.

[13] A. M. Dai and Q. V. Le. Semi-supervised sequence learning. In *Advances in Neural Information Processing Systems*, pages 3079–3087, 2015.

[14] W. B. Dolan and C. Brockett. Automatically constructing a corpus of sentential paraphrases. In *Proceedings of the Third International Workshop on Paraphrasing (IWP2005)*, 2005.

[15] D. Erhan, Y. Bengio, A. Courville, P.-A. Manzagol, P. Vincent, and S. Bengio. Why does unsupervised pre-training help deep learning? *Journal of Machine Learning Research*, 11(Feb):625–660, 2010.

[16] S. Gray, A. Radford, and K. P. Diederik. Gpu kernels for block-sparse weights. 2017.

[17] Z. He, S. Liu, M. Li, M. Zhou, L. Zhang, and H. Wang. Learning entity representation for entity disambiguation. In *Proceedings of the 51st Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)*, volume 2, pages 30–34, 2013.

[18] D. Hendrycks and K. Gimpel. Bridging nonlinearities and stochastic regularizers with gaussian error linear units. *arXiv preprint arXiv:1606.08415*, 2016.

[19] K. M. Hermann, T. Kocisky, E. Grefenstette, L. Espeholt, W. Kay, M. Suleyman, and P. Blunsom. Teaching machines to read and comprehend. In *Advances in Neural Information Processing Systems*, pages 1693–1701, 2015.

[20] G. E. Hinton, S. Osindero, and Y.-W. Teh. A fast learning algorithm for deep belief nets. *Neural computation*, 18(7):1527–1554, 2006.

[21] J. Howard and S. Ruder. Universal language model fine-tuning for text classification. *Association for Computational Linguistics (ACL)*, 2018.

[22] Y. Jernite, S. R. Bowman, and D. Sontag. Discourse-based objectives for fast unsupervised sentence representation learning. *arXiv preprint arXiv:1705.00557*, 2017.

[23] Y. Ji and J. Eisenstein. Discriminative improvements to distributional sentence similarity. In *Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing*, pages 891–896, 2013.

[24] F. Jiao, S. Wang, C.-H. Lee, R. Greiner, and D. Schuurmans. Semi-supervised conditional random fields for improved sequence segmentation and labeling. In *Proceedings of the 21st International Conference on Computational Linguistics and the 44th annual meeting of the Association for Computational Linguistics*, pages 209–216. Association for Computational Linguistics, 2006.

[25] T. Khot, A. Sabharwal, and P. Clark. Scitail: A textual entailment dataset from science question answering. In *Proceedings of AAAI*, 2018.

[26] Y. Kim. Convolutional neural networks for sentence classification. *EMNLP*, 2014.

[27] D. P. Kingma and J. Ba. Adam: A method for stochastic optimization. *arXiv preprint arXiv:1412.6980*, 2014.

[28] R. Kiros, Y. Zhu, R. R. Salakhutdinov, R. Zemel, R. Urtasun, A. Torralba, and S. Fidler. Skip-thought vectors. In *Advances in neural information processing systems*, pages 3294–3302, 2015.

[29] N. Kitaev and D. Klein. Constituency parsing with a self-attentive encoder. *ACL*, 2018.

[30] G. Lai, Q. Xie, H. Liu, Y. Yang, and E. Hovy. Race: Large-scale reading comprehension dataset from examinations. *EMNLP*, 2017.

[31] G. Lample, L. Denoyer, and M. Ranzato. Unsupervised machine translation using monolingual corpora only. *ICLR*, 2018.

[32] Q. Le and T. Mikolov. Distributed representations of sentences and documents. In *International Conference on Machine Learning*, pages 1188–1196, 2014.

[33] P. Liang. Semi-supervised learning for natural language. PhD thesis, Massachusetts Institute of Technology, 2005.

[34] P. J. Liu, M. Saleh, E. Pot, B. Goodrich, R. Sepassi, L. Kaiser, and N. Shazeer. Generating wikipedia by summarizing long sequences. *ICLR*, 2018.

[35] X. Liu, K. Duh, and J. Gao. Stochastic answer networks for natural language inference. *arXiv preprint arXiv:1804.07888*, 2018.

[36] L. Logeswaran and H. Lee. An efficient framework for learning sentence representations. *ICLR*, 2018.

[37] I. Loshchilov and F. Hutter. Fixing weight decay regularization in adam. *arXiv preprint arXiv:1711.05101*, 2017.

[38] B. McCann, J. Bradbury, C. Xiong, and R. Socher. Learned in translation: Contextualized word vectors. In *Advances in Neural Information Processing Systems*, pages 6297–6308, 2017.

[39] T. Mikolov, I. Sutskever, K. Chen, G. S. Corrado, and J. Dean. Distributed representations of words and phrases and their compositionality. In *Advances in neural information processing systems*, pages 3111–3119, 2013.

[40] N. Mostafazadeh, M. Roth, A. Louis, N. Chambers, and J. Allen. Lsdsem 2017 shared task: The story cloze test. In *Proceedings of the 2nd Workshop on Linking Models of Lexical, Sentential and Discourse-level Semantics*, pages 46–51, 2017.

[41] K. Nigam, A. McCallum, and T. Mitchell. Semi-supervised text classification using em. *Semi-Supervised Learning*, pages 33–56, 2006.

[42] J. Pennington, R. Socher, and C. Manning. Glove: Global vectors for word representation. In *Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP)*, pages 1532–1543, 2014.

[43] M. E. Peters, W. Ammar, C. Bhagavatula, and R. Power. Semi-supervised sequence tagging with bidirectional language models. *ACL*, 2017.

[44] M. E. Peters, M. Neumann, M. Iyyer, M. Gardner, C. Clark, K. Lee, and L. Zettlemoyer. Deep contextualized word representations. *NAACL*, 2018.

[45] Y. Qi, D. S. Sachan, M. Felix, S. J. Padmanabhan, and G. Neubig. When and why are pre-trained word embeddings useful for neural machine translation? *NAACL*, 2018.

[46] A. Rahman and V. Ng. Resolving complex cases of definite pronouns: the winograd schema challenge. In *Proceedings of the 2012 Joint Conference on Empirical Methods in Natural Language Processing and Computational Natural Language Learning*, pages 777–789. Association for Computational Linguistics, 2012.

[47] P. Rajpurkar, J. Zhang, K. Lopyrev, and P. Liang. Squad: 100,000+ questions for machine comprehension of text. *EMNLP*, 2016.

[48] P. Ramachandran, P. J. Liu, and Q. V. Le. Unsupervised pretraining for sequence to sequence learning. *arXiv preprint arXiv:1611.02683*, 2016.

[49] M. Ranzato, C. Poultney, S. Chopra, and Y. LeCun. Efficient learning of sparse representations with an energy-based model. In *Advances in neural information processing systems*, pages 1137–1144, 2007.

[50] M. Rei. Semi-supervised multitask learning for sequence labeling. *ACL*, 2017.

[51] H. Robbins and S. Monro. A stochastic approximation method. *The annals of mathematical statistics*, pages 400–407, 1951.

[52] T. Rocktäschel, E. Grefenstette, K. M. Hermann, T. Kočiský, and P. Blunsom. Reasoning about entailment with neural attention. *arXiv preprint arXiv:1509.06664*, 2015.

[53] R. Sennrich, B. Haddow, and A. Birch. Neural machine translation of rare words with subword units. *arXiv preprint arXiv:1508.07909*, 2015.

[54] R. Socher, A. Perelygin, J. Wu, J. Chuang, C. D. Manning, A. Ng, and C. Potts. Recursive deep models for semantic compositionality over a sentiment treebank. In *Proceedings of the 2013 conference on empirical methods in natural language processing*, pages 1631–1642, 2013.

[55] S. Srinivasan, R. Arora, and M. Riedl. A simple and effective approach to the story cloze test. *arXiv preprint arXiv:1803.05547*, 2018.

[56] S. Subramanian, A. Trischler, Y. Bengio, and C. J. Pal. Learning general purpose distributed sentence representations via large scale multi-task learning. *arXiv preprint arXiv:1804.00079*, 2018.

[57] J. Suzuki and H. Isozaki. Semi-supervised sequential labeling and segmentation using giga-word scale unlabeled data. *Proceedings of ACL-08: HLT*, pages 665–673, 2008.

[58] Y. Tay, L. A. Tuan, and S. C. Hui. A compare-propagate architecture with alignment factorization for natural language inference. *arXiv preprint arXiv:1801.00102*, 2017.

[59] Y. Tay, L. A. Tuan, and S. C. Hui. Multi-range reasoning for machine comprehension. *arXiv preprint arXiv:1803.09074*, 2018.

[60] J. Tian, Z. Zhou, M. Lan, and Y. Wu. Ecnu at semeval-2017 task 1: Leverage kernel-based traditional nlp features and neural networks to build a universal model for multilingual and cross-lingual semantic textual similarity. In *Proceedings of the 11th International Workshop on Semantic Evaluation (SemEval-2017)*, pages 191–197, 2017.

[61] Y. Tsvetkov. Opportunities and challenges in working with low-resource languages. CMU, 2017.

[62] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin. Attention is all you need. In *Advances in Neural Information Processing Systems*, pages 6000–6010, 2017.

[63] P. Vincent, H. Larochelle, Y. Bengio, and P.-A. Manzagol. Extracting and composing robust features with denoising autoencoders. In *Proceedings of the 25th international conference on Machine learning*, pages 1096–1103. ACM, 2008.

[64] A. Wang, A. Singh, J. Michael, F. Hill, O. Levy, and S. R. Bowman. Glue: A multi-task benchmark and analysis platform for natural language understanding. *arXiv preprint arXiv:1804.07461*, 2018.

[65] A. Warstadt, A. Singh, and S. R. Bowman. Corpus of linguistic acceptability. http://nyu-mll.github.io/cola, 2018.

[66] A. Williams, N. Nangia, and S. R. Bowman. A broad-coverage challenge corpus for sentence understanding through inference. *NAACL*, 2018.

[67] Y. Xu, J. Liu, J. Gao, Y. Shen, and X. Liu. Towards human-level machine reading comprehension: Reasoning and inference with multiple strategies. *arXiv preprint arXiv:1711.04964*, 2017.

[68] D. Yu, L. Deng, and G. Dahl. Roles of pre-training and fine-tuning in context-dependent dbn-hmms for real-world speech recognition. In *Proc. NIPS Workshop on Deep Learning and Unsupervised Feature Learning*, 2010.

[69] R. Zhang, P. Isola, and A. A. Efros. Split-brain autoencoders: Unsupervised learning by cross-channel prediction. In *CVPR*, volume 1, page 6, 2017.

[70] X. Zhu. Semi-supervised learning literature survey. 2005.

[71] Y. Zhu, R. Kiros, R. Zemel, R. Salakhutdinov, R. Urtasun, A. Torralba, and S. Fidler. Aligning books and movies: Towards story-like visual explanations by watching movies and reading books. In *Proceedings of the IEEE international conference on computer vision*, pages 19–27, 2015.
