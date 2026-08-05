# BERT：深度双向Transformer预训练用于语言理解

> Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova | Google AI Language

本文介绍了 BERT（Bidirectional Encoder Representations from Transformers）：一种新的语言表示模型。核心内容：

- 双向预训练：提出掩码语言模型（MLM）目标，随机掩码输入 token 并预测原始词汇，使表示能融合左、右上下文，突破标准语言模型的单向限制
- 下一句预测（NSP）：联合预训练文本对表示，通过判断句子 B 是否为句子 A 的真实下一句来学习句子间关系
- 统一架构：预训练架构与下游任务架构差异极小，只需一个额外输出层即可适配问答、蕴含、情感分析等多种任务
- 全面超越：在 11 个 NLP 任务上取得新的最先进结果，GLUE 80.5%、MultiNLI 86.7%、SQuAD v1.1 Test F1 93.2、SQuAD v2.0 Test F1 83.1

关键发现：

- 深度双向表示严格优于单向或浅层拼接：BERT 的 MLM 预训练显著优于 OpenAI GPT 的从左到右架构和 ELMo 的独立双向 LSTM 拼接
- 预训练表示减少了任务特定架构需求：BERT 是第一个基于微调的表示模型，在句子级和 token 级任务上均取得最先进性能，优于许多任务特定架构
- 双向性对 token 级任务至关重要：在 SQuAD 问答等需要融合双向上下文的任务中，BERT 相比单向模型优势尤为明显

BERT（Bidirectional Encoder Representations from Transformers）是一种新的语言表示模型。与最近的语言表示模型不同，BERT被设计为通过在所有层中同时联合调节左、右上下文，从未标记文本中预训练深度双向表示。因此，预训练的BERT模型只需一个额外的输出层进行微调，即可为广泛的任务（如问答和语言推理）创建最先进的模型，而无需大量的任务特定架构修改。BERT在概念上简单且经验上强大。它在十一个自然语言处理任务上取得了新的最先进结果，包括将GLUE分数推至80.5%（绝对提升7.7个百分点）、MultiNLI准确率达到86.7%（绝对提升4.6%）、SQuAD v1.1问答Test F1达到93.2（绝对提升1.5点）以及SQuAD v2.0 Test F1达到83.1（绝对提升5.1点）。

---

## 摘要

我们引入了一种名为BERT的新语言表示模型，它代表Bidirectional Encoder Representations from Transformers。与最近的语言表示模型（[36]；[38]）不同，BERT被设计为通过在所有层中联合调节左、右上下文，从未标记文本中预训练深度双向表示。因此，预训练的BERT模型只需一个额外的输出层进行微调，即可为广泛的任务（如问答和语言推理）创建最先进的模型，而无需大量的任务特定架构修改。BERT在概念上简单且经验上强大。它在十一个自然语言处理任务上取得了新的最先进结果，包括将GLUE分数推至80.5%（绝对提升7.7个百分点）、MultiNLI准确率达到86.7%（绝对提升4.6%）、SQuAD v1.1问答Test F1达到93.2（绝对提升1.5点）以及SQuAD v2.0 Test F1达到83.1（绝对提升5.1点）。

## 关键词

Pre-training, Language Representation, Bidirectional Transformers, Transfer Learning, Natural Language Understanding

## 1 引言

语言模型预训练已被证明对改进许多自然语言处理任务是有效的（[15]；[36]；[38]；[21]）。这些任务包括句子级任务（如自然语言推理（[6]；[51]）和释义[17]），旨在通过整体分析句子来预测句子间的关系，以及token级任务（如命名实体识别和问答），其中模型需要在token级别生成细粒度输出（Tjong Kim Sang和De [44]；[39]）。

将预训练语言表示应用于下游任务有两种现有策略：基于特征的和微调的。基于特征的方法（如ELMo[36]）使用包含预训练表示作为额外特征的任务特定架构。微调方法（如Generative Pre-trained Transformer（OpenAI GPT）[38]）引入最少的任务特定参数，并通过简单微调所有预训练参数来在下游任务上训练。这两种方法在预训练期间共享相同的目标函数，即它们使用单向语言模型来学习通用语言表示。

我们认为当前技术限制了预训练表示的能力，特别是对于微调方法。主要的限制是标准语言模型是单向的，这限制了在预训练期间可以使用的架构选择。例如，在OpenAI GPT中，作者使用从左到右的架构，其中每个token在Transformer的自注意力层[46]中只能关注之前的token。这种限制对于句子级任务来说是次优的，并且在将基于微调的方法应用于token级任务（如问答，其中从两个方向融合上下文至关重要）时可能非常有害。

在本文中，我们通过提出BERT：Bidirectional Encoder Representations from Transformers来改进基于微调的方法。BERT通过使用受Cloze任务[43]启发的"掩码语言模型"（MLM）预训练目标来缓解前述的单向性约束。掩码语言模型随机掩码输入中的一些token，目标是仅基于其上下文预测掩码token的原始词汇ID。与从左到右的语言模型预训练不同，MLM目标使得表示能够融合左、右上下文，这允许我们预训练深度双向Transformer。除了掩码语言模型，我们还使用"下一句预测"任务来联合预训练文本对表示。我们论文的贡献如下：

- 我们证明了双向预训练对语言表示的重要性。与使用单向语言模型进行预训练的Radford等人（2018）不同，BERT使用掩码语言模型来实现预训练的深度双向表示。这也与[36]形成对比，后者使用了独立训练的从左到右和从右到左LM的浅层拼接。

- 我们展示了预训练表示减少了对大量精心设计的任务特定架构的需求。BERT是第一个基于微调的表示模型，在一系列句子级和token级任务上取得最先进性能，优于许多任务特定架构。

- BERT推进了十一个NLP任务的最新水平。代码和预训练模型可在 https://github.com/google-research/bert 获取。

## 2 相关工作

预训练通用语言表示有悠久的历史，我们在本节简要回顾最广泛使用的方法。

### 2.1 无监督基于特征的方法

学习广泛适用的词语表示几十年来一直是一个活跃的研究领域，包括非神经（[7]；[3]；[5]）和神经（[31]；[34]）方法。预训练的词嵌入是现代NLP系统的一个组成部分，提供了比从头学习的嵌入显著的改进[45]。为了预训练词嵌入向量，已经使用了从左到右的语言建模目标[32]，以及在左、右上下文中区分正确和错误词语的目标[31]。

这些方法已经被泛化到更粗的粒度，如句子嵌入（[25]；[28]）或段落嵌入[26]。为了训练句子表示，先前的工作使用了排序候选下一句的目标（[23]；[28]）、给定前一个句子表示的情况下从左到右生成下一句词语[25]，或去噪自编码器派生目标[20]。

ELMo及其前身（[35]，2018a）沿不同方向推广了传统的词嵌入研究。它们从左到右和从右到左的语言模型中提取上下文敏感特征。每个token的上下文表示是左到右和右到左表示的拼接。当将上下文词嵌入与现有的任务特定架构集成时，ELMo推进了几个主要NLP基准的最先进水平[36]，包括问答[39]、情感分析[41]和命名实体识别（Tjong Kim Sang和De [44]）。[30]提出了通过使用LSTM从左、右上下文预测单个词的任务来学习上下文表示。与ELMo类似，他们的模型是基于特征的，而不是深度双向的。[18]表明Cloze任务可用于改进文本生成模型的鲁棒性。

### 2.2 无监督微调方法

与基于特征的方法一样，这一方向的第一批工作仅从无标记文本预训练词嵌入参数[13]。最近，生成上下文token表示的句子或文档编码器已从无标记文本预训练，并针对监督下游任务进行微调（[15]；[21]；[38]）。这些方法的优点是需要从头学习的参数很少。至少部分由于这一优势，OpenAI GPT[38]在GLUE基准[48]的许多句子级任务上取得了先前最先进的结果。从左到右的语言建模和自编码器目标已被用于预训练此类模型（[21]；[38]；[15]）。

### 2.3 从监督数据迁移学习

也有研究表明从具有大数据集的有监督任务进行有效迁移，如自然语言推理[14]和机器翻译（Mc[29]）。计算机视觉研究也证明了从大型预训练模型进行迁移学习的重要性，其中一种有效的方法是对使用ImageNet（[16]；[53]）预训练的模型进行微调。

## 3 BERT

我们在本节中介绍BERT及其详细实现。我们的框架有两个步骤：预训练和微调。在预训练期间，模型在不同的预训练任务上对无标记数据进行训练。在微调时，BERT模型首先用预训练参数初始化，然后所有参数使用下游任务的标记数据进行微调。每个下游任务都有独立的微调模型，尽管它们用相同的预训练参数初始化。图1中的问答示例将作为本节的一个贯穿示例。

BERT的一个独特之处在于其跨不同任务的统一架构。预训练架构和最终下游架构之间的差异极小。

**模型架构** BERT的模型架构是一个多层双向Transformer编码器，基于Vaswani等人（2017）中描述的原始实现并在tensor2tensor库中发布[1]。由于Transformer的使用已经很普遍，且我们的实现几乎与原始实现相同，我们将省略对模型架构的详尽背景描述，读者可参考Vaswani等人（2017）以及优秀的指南如"The Annotated Transformer"[2]。

在这项工作中，我们将层数（即Transformer块）记为 $L$ ，隐藏层大小记为 $H$ ，自注意力头数记为 $A$ [3]。我们主要报告两种模型大小的结果：BERT $_{\text{BASE}}$ （ $L=12$ ， $H=768$ ， $A=12$ ，总参数量=110M）和BERT $_{\text{LARGE}}$ （ $L=24$ ， $H=1024$ ， $A=16$ ，总参数量=340M）。BERT $_{\text{BASE}}$ 被选择为具有与OpenAI GPT相同的模型大小以便比较。然而关键的是，BERT Transformer使用双向自注意力，而GPT Transformer使用受限的自注意力，其中每个token只能关注其左侧的上下文[4]。

**输入/输出表示** 为了使BERT能够处理各种下游任务，我们的输入表示能够在一个token序列中无歧义地表示单个句子和句子对（例如， $\langle\text{Question}, \text{Answer}\rangle$ ）。在这项工作中，"句子"可以是连续的任意文本片段，而不是实际的语言学句子。"序列"指的是BERT的输入token序列，可以是单个句子或两个句子打包在一起。

我们使用包含30,000个token词汇表的WordPiece嵌入[52]。每个序列的第一个token始终是一个特殊的分类token（`[CLS]`）。对应于该token的最终隐藏状态被用作分类任务的聚合序列表示。句子对被打包成一个序列。我们通过两种方式区分句子。首先，我们使用特殊token（`[SEP]`）将它们分开。其次，我们为每个token添加一个学习到的嵌入，指示它属于句子A还是句子B。如图1所示，我们将输入嵌入记为 $E$ ，特殊`[CLS]`token的最终隐藏向量记为 $C \in \mathbb{R}^H$ ，第 $i$ 个输入token的最终隐藏向量记为 $T_i \in \mathbb{R}^H$ 。

对于给定的token，其输入表示由相应的token、segment和position嵌入求和构成。这种构建的可视化见图2。

### 3.1 预训练BERT

与[36]和Radford等人（2018）不同，我们不使用传统的从左到右或从右到左的语言模型来预训练BERT。相反，我们使用两个无监督任务来预训练BERT，本节将进行描述。这一步骤在图1的左侧部分展示。

**任务#1：掩码LM** 直观上，有理由相信深度双向模型严格比从左到右模型或左到右和右到左模型的浅层拼接更强大。不幸的是，标准条件语言模型只能从左到右或从右到左训练，因为双向条件化会让每个词间接地"看到自己"，从而模型可以轻易地在多层上下文中预测目标词。

为了训练深度双向表示，我们简单地随机掩码一定百分比的输入token，然后预测这些掩码token。我们将此过程称为"掩码LM"（MLM），尽管在文献中它通常被称为Cloze任务[43]。在这种情况下，对应于掩码token的最终隐藏向量被馈送到词汇表上的输出softmax，如同标准LM一样。在我们所有的实验中，我们随机掩码每个序列中15%的WordPiece token。与去噪自编码器[47]不同，我们只预测掩码词，而不是重构整个输入。

尽管这使我们能够获得双向预训练模型，但一个缺点是我们在预训练和微调之间造成了一个不匹配，因为`[MASK]`token在微调期间不会出现。为了缓解这个问题，我们并不总是用实际的`[MASK]`token替换"掩码"词。训练数据生成器随机选择15%的token位置进行预测。如果第 $i$ 个token被选中，我们用以下方式替换第 $i$ 个token：（1）80%的情况下替换为`[MASK]`token；（2）10%的情况下替换为随机token；（3）10%的情况下保持不变的第 $i$ 个token。然后， $T_i$ 将用于使用交叉熵损失预测原始token。我们在附录C.2中比较了这个过程的变体。

**任务#2：下一句预测（NSP）** 许多重要的下游任务，如问答（QA）和自然语言推理（NLI），都基于理解两个句子之间的关系，而这无法直接通过语言建模捕捉。为了训练一个理解句子关系的模型，我们预训练一个二值化的下一句预测任务，该任务可以轻易地从任何单语语料库生成。具体来说，在为每个预训练示例选择句子A和B时，50%的情况下B是A之后实际的下一句（标记为IsNext），50%的情况下B是语料库中的随机句子（标记为NotNext）。如图1所示， $C$ 用于下一句预测（NSP）[5]。尽管其简单性，我们在第5.1节中证明，针对该任务的预训练对QA和NLI都非常有益[6]。

NSP任务与Jernite等人（2017）和Logeswaran和Lee（2018）中使用的表示学习目标密切相关。然而，在先前工作中，只有句子嵌入被迁移到下游任务，而BERT将所有权重参数迁移以初始化最终任务模型参数。

**预训练数据** 预训练过程在很大程度上遵循语言模型预训练的现有文献。对于预训练语料库，我们使用BooksCorpus（8亿词）[56]和英文维基百科（25亿词）。对于维基百科，我们只提取文本段落，忽略列表、表格和标题。使用文档级语料库至关重要，而非像Billion Word Benchmark[9]这样的打乱句子级语料库，以便提取长的连续序列。

### 3.2 微调BERT

微调很直接，因为Transformer中的自注意力机制允许BERT对许多下游任务进行建模——无论是涉及单文本还是文本对——只需替换适当的输入和输出。对于涉及文本对的应用，一种常见模式是在应用双向交叉注意力之前独立编码文本对，如[33]和Seo等人（2017）所做的那样。相反，BERT使用自注意力机制来统一这两个阶段，因为用自注意力编码拼接的文本对有效地包含了两个句子之间的双向交叉注意力。

对于每个任务，我们只需将任务特定的输入和输出插入BERT，并端到端地微调所有权重参数。在输入方面，预训练中的句子A和句子B类似于：（1）释义中的句子对；（2）蕴含中的假设-前提对；（3）问答中的问题-段落对；（4）文本分类或序列标注中的退化文本- $\emptyset$ 对。在输出方面，token表示被馈送到输出层以用于token级任务（如序列标注或问答），而`[CLS]`表示被馈送到输出层以用于分类任务（如蕴含或情感分析）。

与预训练相比，微调相对廉价。本文中的所有结果都可以在单块Cloud TPU上最多1小时内或在GPU上几小时内从完全相同的预训练模型复制[7]。我们在第4节的相应子节中描述任务特定的细节。更多细节可在附录A.5中找到。

## 4 实验

在本节中，我们展示BERT在11个NLP任务上的微调结果。

### 4.1 GLUE

通用语言理解评估（GLUE）基准[48]是一个多样化自然语言理解任务的集合。GLUE数据集的详细描述包含在附录B.1中。

为了在GLUE上微调，我们将输入序列（对于单个句子或句子对）表示为第3节所述，并使用对应于第一个输入token（`[CLS]`）的最终隐藏向量 $C \in \mathbb{R}^H$ 作为聚合表示。微调期间引入的唯一新参数是分类层权重 $W \in \mathbb{R}^{K \times H}$ ，其中 $K$ 是标签数量。我们用 $C$ 和 $W$ 计算标准分类损失，即 $\log(\text{softmax}(CW^\top))$ 。

我们对所有GLUE任务使用32的批量大小并微调3个epoch。对于每个任务，我们在Dev集上选择最佳微调学习率（从5e-5、4e-5、3e-5和2e-5中）。此外，对于BERT $_{\text{LARGE}}$ ，我们发现微调在小型数据集上有时不稳定，因此我们运行多次随机重启并在Dev集上选择最佳模型。对于随机重启，我们使用相同的预训练检查点，但进行不同的微调数据打乱和分类器层初始化[9]。

结果呈现在表1中。BERT $_{\text{BASE}}$ 和BERT $_{\text{LARGE}}$ 都以显著优势在所有任务上优于所有系统，相较于先前最先进水平分别获得了4.5%和7.0%的平均准确率提升。注意，BERT $_{\text{BASE}}$ 和OpenAI GPT在模型架构上除了注意力掩码之外几乎相同。对于最大且最广泛报告的GLUE任务MNLI，BERT获得了4.6%的绝对准确率提升。在官方GLUE排行榜[10]上，截至撰写本文时，BERT $_{\text{LARGE}}$ 获得80.5分，而OpenAI GPT获得72.8分。

我们发现BERT $_{\text{LARGE}}$ 在所有任务上显著优于BERT $_{\text{BASE}}$ ，尤其是那些训练数据非常少的任务。模型大小的影响在第5.2节中更深入地探讨。

### 4.2 SQuAD v1.1

斯坦福问答数据集（SQuAD v1.1）是一个包含10万个众包问答对的集合[39]。给定一个问题和一个包含答案的维基百科段落，任务是预测段落中的答案文本跨度。

如图1所示，在问答任务中，我们将输入问题和段落表示为一个打包序列，问题使用A嵌入，段落使用B嵌入。我们在微调期间只引入起始向量 $S \in \mathbb{R}^H$ 和结束向量 $E \in \mathbb{R}^H$ 。词 $i$ 是答案跨度起始的概率计算为 $T_i$ 与 $S$ 的点积，然后对段落中所有词进行softmax：

$$
P_i = \frac{e^{S \cdot T_i}}{\sum_j e^{S \cdot T_j}} \qquad (1)
$$

用于答案跨度结束的公式类似。从位置 $i$ 到位置 $j$ 的候选跨度分数定义为 $S \cdot T_i + E \cdot T_j$ ，分数最高的 $j \geq i$ 的跨度被用作预测。训练目标是正确起始和结束位置的对数似然之和。我们以5e-5的学习率和32的批量大小微调3个epoch。

表2显示了排行榜顶尖的条目以及顶尖已发表系统的结果（[40]；[11]；[36]；[22]）。SQuAD排行榜上的顶尖结果没有最新的公开系统描述可用[11]，并且允许在训练其系统时使用任何公开数据。因此，我们在系统中使用适度的数据增强，首先在TriviaQA[24]上微调，然后在SQuAD上微调。

我们表现最佳的系统在集成方面超越了顶尖排行榜系统+1.5 F1，作为单一系统超越了+1.3 F1。事实上，我们的单一BERT模型在F1分数上超过了顶尖集成系统。没有TriviaQA微调数据，我们只损失0.1-0.4 F1，仍然以大幅优势超越所有现有系统[12]。

### 4.3 SQuAD v2.0

SQuAD 2.0任务扩展了SQuAD 1.1的问题定义，允许所提供的段落中可能不存在简短答案，使问题更加现实。

我们使用一种简单的方法将SQuAD v1.1 BERT模型扩展到此任务。我们将没有答案的问题视为答案跨度起始和结束在`[CLS]`token处。起始和结束答案跨度位置的概率空间被扩展以包含`[CLS]`token的位置。对于预测，我们比较无答案跨度的分数 $s_{\text{null}} = S \cdot C + E \cdot C$ 与最佳非空跨度分数 $\hat{s}_{i,j} = \max_{j \geq i} S \cdot T_i + E \cdot T_j$ 。当 $\hat{s}_{i,j} > s_{\text{null}} + \tau$ 时我们预测非空答案，其中阈值 $\tau$ 在dev集上选择以最大化F1。我们没有对此模型使用TriviaQA数据。我们以5e-5的学习率和48的批量大小微调2个epoch。

结果与先前排行榜条目和顶尖已发表工作（[42]；[48]）的比较在表3中显示，排除了使用BERT作为其组件之一的系统。我们观察到相较于先前最佳系统有+5.1 F1的提升。

### 4.4 SWAG

Situations With Adversarial Generations（SWAG）数据集包含113k个句子对补全示例，用于评估基于上下文的常识推理[55]。给定一个句子，任务是在四个选项中选出最合理的延续项。

在SWAG数据集上微调时，我们构造四个输入序列，每个序列包含给定句子（句子A）和一个可能的延续项（句子B）的拼接。引入的唯一任务特定参数是一个向量，其与`[CLS]`token表示 $C$ 的点积表示每个选项的分数，该分数通过softmax层归一化。

我们以2e-5的学习率和16的批量大小微调模型3个epoch。结果呈现在表4中。BERT $_{\text{LARGE}}$ 优于作者的基线ESIM+ELMo系统+27.1%，优于OpenAI GPT 8.3%。

## 5 消融研究

在本节中，我们对BERT的多个方面进行消融实验，以更好地理解它们的相对重要性。额外的消融研究可在附录C中找到。

### 5.1 预训练任务的影响

我们通过评估两个预训练目标来证明BERT深度双向性的重要性，使用与BERT $_{\text{BASE}}$ 完全相同的预训练数据、微调方案和超参数：

- **No NSP**：使用"掩码LM"（MLM）训练但不使用"下一句预测"（NSP）任务的双向模型。
- **LTR & No NSP**：使用标准从左到右（LTR）LM而非MLM训练的仅左侧上下文模型。左向约束在微调时也适用，因为移除它会引入损害下游性能的预训练/微调不匹配。此外，该模型在没有NSP任务的情况下进行了预训练。这直接与OpenAI GPT可比，但使用了我们更大的训练数据集、输入表示和微调方案。

我们首先检查NSP任务带来的影响。在表5中，我们显示移除NSP会显著损害QNLI、MNLI和SQuAD 1.1上的性能。接下来，我们通过比较"No NSP"和"LTR & No NSP"来评估训练双向表示的影响。LTR模型在所有任务上表现逊于MLM模型，在MRPC和SQuAD上有大幅下降。

对于SQuAD，直觉上很清楚LTR模型在token预测上表现不佳，因为token级别的隐藏状态没有右侧上下文。为了真诚地尝试加强LTR系统，我们在其上添加了一个随机初始化的BiLSTM。这确实显著改善了SQuAD的结果，但结果仍然远逊于预训练的双向模型。BiLSTM损害了GLUE任务上的性能。

我们认识到也可以像ELMo那样训练独立的LTR和RTL模型，并将每个token表示为两个模型的拼接。然而：（a）这比单个双向模型贵两倍；（b）这对于QA等任务来说不直观，因为RTL模型无法根据问题条件化答案；（c）这严格弱于深度双向模型，因为深度双向模型可以在每一层使用左、右两种上下文。

### 5.2 模型大小的影响

在本节中，我们探讨模型大小对微调任务准确率的影响。我们训练了一系列具有不同层数、隐藏单元数和注意力头数的BERT模型，同时使用与之前描述相同的超参数和训练过程。

在选定的GLUE任务上的结果显示在表6中。在该表中，我们报告了5次随机重启微调的平均Dev集准确率。我们可以看到更大的模型在所有四个数据集上都带来了严格的准确率提升，即使是仅有3,600个标记训练样本且与预训练任务有显著差异的MRPC。也许同样令人惊讶的是，我们能够在相对于现有文献已经相当大的模型之上取得如此显著的改进。例如，Vaswani等人（2017）中探索的最大Transformer是（ $L=6$ ， $H=1024$ ， $A=16$ ），编码器有1亿参数，而我们在文献中找到的最大Transformer是（ $L=64$ ， $H=512$ ， $A=2$ ），有2.35亿参数（Al-[2]）。相比之下，BERT $_{\text{BASE}}$ 包含1.1亿参数，BERT $_{\text{LARGE}}$ 包含3.4亿参数。

长期以来已知增加模型规模会持续改进大规模任务（如机器翻译和语言建模），这由表6中展示的留出训练数据的LM困惑度所证明。然而，我们相信这是第一次令人信服地证明扩展到极端模型大小也会在非常小规模的任务上带来大幅改进，前提是模型已经得到了充分的预训练。[37]展示了将预训练bi-LM规模从两层增加到四层在下游任务影响上的混合结果，[30]顺带提到将隐藏维度大小从200增加到600有所帮助，但进一步增加到1000并未带来进一步改进。这两项先前工作都使用了基于特征的方法——我们假设，当模型直接在下游任务上微调且只使用非常少量的随机初始化额外参数时，任务特定模型可以从更大、更具表现力的预训练表示中受益，即使下游任务数据非常少。

### 5.3 基于特征的BERT方法

目前展示的所有BERT结果都使用了微调方法，即在预训练模型上添加一个简单的分类层，并联合微调所有参数。然而，从预训练模型中提取固定特征的基于特征的方法也有其优势。首先，并非所有任务都能被Transformer编码器架构轻松表示，因此需要添加任务特定的模型架构。其次，预计算训练数据的昂贵表示一次，然后在此表示基础上用更便宜的模型进行多次实验，具有显著的计算优势。

在本节中，我们通过将BERT应用于CoNLL-2003命名实体识别（NER）任务（Tjong Kim Sang和De [44]）来比较这两种方法。在BERT的输入中，我们使用保留大小写的WordPiece模型，并包含数据提供的最大文档上下文。按照标准做法，我们将其规约为标注任务，但不在输出中使用CRF层。我们使用第一个子token的表示作为NER标签集上的token级分类器的输入。

为了与微调方法进行消融对比，我们应用基于特征的方法，提取一个或多个层的激活，而不微调BERT的任何参数。这些上下文嵌入在被馈送到分类层之前，用作随机初始化的两层768维BiLSTM的输入。

结果呈现在表7中。BERT $_{\text{LARGE}}$ 的性能与最先进方法相当。表现最佳的方法拼接预训练Transformer顶部四个隐藏层的token表示，这仅比微调整个模型落后0.3 F1。这证明了BERT对于微调和基于特征的方法都是有效的。

## 6 结论

最近由于使用语言模型进行迁移学习带来的实证改进表明，丰富的无监督预训练是许多语言理解系统不可或缺的部分。特别是，这些结果使得即使是低资源任务也能从深度单向架构中受益。我们的主要贡献是将这些发现进一步泛化到深度双向架构，使得相同的预训练模型能够成功处理广泛的NLP任务。

## 参考文献

[1] Alan Akbik, Duncan Blythe, and Roland Vollgraf. 2018. Contextual string embeddings for sequence labeling. In *Proceedings of the 27th International Conference on Computational Linguistics*, pages 1638–1649.

[2] Rami Al-Rfou, Dokook Choe, Noah Constant, Mandy Guo, and Llion Jones. 2018. Character-level language modeling with deeper self-attention. *arXiv preprint arXiv:1808.04444*.

[3] Rie Kubota Ando and Tong Zhang. 2005. A framework for learning predictive structures from multiple tasks and unlabeled data. *Journal of Machine Learning Research*, 6(Nov):1817–1853.

[4] Luisa Bentivogli, Bernardo Magnini, Ido Dagan, Hoa Trang Dang, and Danilo Giampiccolo. 2009. The fifth PASCAL recognizing textual entailment challenge. In *TAC*. NIST.

[5] John Blitzer, Ryan McDonald, and Fernando Pereira. 2006. Domain adaptation with structural correspondence learning. In *Proceedings of the 2006 conference on empirical methods in natural language processing*, pages 120–128. Association for Computational Linguistics.

[6] Samuel R. Bowman, Gabor Angeli, Christopher Potts, and Christopher D. Manning. 2015. A large annotated corpus for learning natural language inference. In *EMNLP*. Association for Computational Linguistics.

[7] Peter F Brown, Peter V Desouza, Robert L Mercer, Vincent J Della Pietra, and Jenifer C Lai. 1992. Class-based n-gram models of natural language. *Computational linguistics*, 18(4):467–479.

[8] Daniel Cer, Mona Diab, Eneko Agirre, Inigo Lopez-Gazpio, and Lucia Specia. 2017. SemEval-2017 task 1: Semantic textual similarity multilingual and crosslingual focused evaluation. In *Proceedings of the 11th International Workshop on Semantic Evaluation (SemEval-2017)*, pages 1–14, Vancouver, Canada. Association for Computational Linguistics.

[9] Ciprian Chelba, Tomas Mikolov, Mike Schuster, Qi Ge, Thorsten Brants, Phillipp Koehn, and Tony Robinson. 2013. One billion word benchmark for measuring progress in statistical language modeling. *arXiv preprint arXiv:1312.3005*.

[10] Z. Chen, H. Zhang, X. Zhang, and L. Zhao. 2018. Quora question pairs.

[11] Christopher Clark and Matt Gardner. 2018. Simple and effective multi-paragraph reading comprehension. In *ACL*.

[12] Kevin Clark, Minh-Thang Luong, Christopher D Manning, and Quoc Le. 2018. Semi-supervised sequence modeling with cross-view training. In *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing*, pages 1914–1925.

[13] Ronan Collobert and Jason Weston. 2008. A unified architecture for natural language processing: Deep neural networks with multitask learning. In *Proceedings of the 25th international conference on Machine learning*, pages 160–167. ACM.

[14] Alexis Conneau, Douwe Kiela, Holger Schwenk, Loïc Barrault, and Antoine Bordes. 2017. Supervised learning of universal sentence representations from natural language inference data. In *Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing*, pages 670–680, Copenhagen, Denmark. Association for Computational Linguistics.

[15] Andrew M Dai and Quoc V Le. 2015. Semi-supervised sequence learning. In *Advances in neural information processing systems*, pages 3079–3087.

[16] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei. 2009. ImageNet: A large-scale hierarchical image database. In *CVPR09*.

[17] William B Dolan and Chris Brockett. 2005. Automatically constructing a corpus of sentential paraphrases. In *Proceedings of the Third International Workshop on Paraphrasing (IWP2005)*.

[18] William Fedus, Ian Goodfellow, and Andrew M Dai. 2018. MaskGAN: Better text generation via filling in the \_. *arXiv preprint arXiv:1801.07736*.

[19] Dan Hendrycks and Kevin Gimpel. 2016. Bridging nonlinearities and stochastic regularizers with gaussian error linear units. *CoRR*, abs/1606.08415.

[20] Felix Hill, Kyunghyun Cho, and Anna Korhonen. 2016. Learning distributed representations of sentences from unlabelled data. In *Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*. Association for Computational Linguistics.

[21] Jeremy Howard and Sebastian Ruder. 2018. Universal language model fine-tuning for text classification. In *ACL*. Association for Computational Linguistics.

[22] Minghao Hu, Yuxing Peng, Zhen Huang, Xipeng Qiu, Furu Wei, and Ming Zhou. 2018. Reinforced mnemonic reader for machine reading comprehension. In *IJCAI*.

[23] Yacine Jernite, Samuel R. Bowman, and David Sontag. 2017. Discourse-based objectives for fast unsupervised sentence representation learning. *CoRR*, abs/1705.00557.

[24] Mandar Joshi, Eunsol Choi, Daniel S Weld, and Luke Zettlemoyer. 2017. TriviaQA: A large scale distantly supervised challenge dataset for reading comprehension. In *ACL*.

[25] Ryan Kiros, Yukun Zhu, Ruslan R Salakhutdinov, Richard Zemel, Raquel Urtasun, Antonio Torralba, and Sanja Fidler. 2015. Skip-thought vectors. In *Advances in neural information processing systems*, pages 3294–3302.

[26] Quoc Le and Tomas Mikolov. 2014. Distributed representations of sentences and documents. In *International Conference on Machine Learning*, pages 1188–1196.

[27] Hector J Levesque, Ernest Davis, and Leora Morgenstern. 2011. The winograd schema challenge. In *AAAI spring symposium: Logical formalizations of commonsense reasoning*, volume 46, page 47.

[28] Lajanugen Logeswaran and Honglak Lee. 2018. An efficient framework for learning sentence representations. In *International Conference on Learning Representations*.

[29] Bryan McCann, James Bradbury, Caiming Xiong, and Richard Socher. 2017. Learned in translation: Contextualized word vectors. In *NIPS*.

[30] Oren Melamud, Jacob Goldberger, and Ido Dagan. 2016. context2vec: Learning generic context embedding with bidirectional LSTM. In *CoNLL*.

[31] Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg S Corrado, and Jeff Dean. 2013. Distributed representations of words and phrases and their compositionality. In *Advances in Neural Information Processing Systems 26*, pages 3111–3119. Curran Associates, Inc.

[32] Andriy Mnih and Geoffrey E Hinton. 2009. A scalable hierarchical distributed language model. In D. Koller, D. Schuurmans, Y. Bengio, and L. Bottou, editors, *Advances in Neural Information Processing Systems 21*, pages 1081–1088. Curran Associates, Inc.

[33] Ankur P Parikh, Oscar Täckström, Dipanjan Das, and Jakob Uszkoreit. 2016. A decomposable attention model for natural language inference. In *EMNLP*.

[34] Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. GloVe: Global vectors for word representation. In *Empirical Methods in Natural Language Processing (EMNLP)*, pages 1532–1543.

[35] Matthew Peters, Waleed Ammar, Chandra Bhagavatula, and Russell Power. 2017. Semi-supervised sequence tagging with bidirectional language models. In *ACL*.

[36] Matthew Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, and Luke Zettlemoyer. 2018a. Deep contextualized word representations. In *NAACL*.

[37] Matthew Peters, Mark Neumann, Luke Zettlemoyer, and Wen-tau Yih. 2018b. Dissecting contextual word embeddings: Architecture and representation. In *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing*, pages 1499–1509.

[38] Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever. 2018. Improving language understanding with unsupervised learning. Technical report, OpenAI.

[39] Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. 2016. SQuAD: 100,000+ questions for machine comprehension of text. In *Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing*, pages 2383–2392.

[40] Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, and Hannaneh Hajishirzi. 2017. Bidirectional attention flow for machine comprehension. In *ICLR*.

[41] Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D Manning, Andrew Ng, and Christopher Potts. 2013. Recursive deep models for semantic compositionality over a sentiment treebank. In *Proceedings of the 2013 conference on empirical methods in natural language processing*, pages 1631–1642.

[42] Fu Sun, Linyang Li, Xipeng Qiu, and Yang Liu. 2018. U-net: Machine reading comprehension with unanswerable questions. *arXiv preprint arXiv:1810.06638*.

[43] Wilson L Taylor. 1953. Cloze procedure: A new tool for measuring readability. *Journalism Bulletin*, 30(4):415–433.

[44] Erik F Tjong Kim Sang and Fien De Meulder. 2003. Introduction to the CoNLL-2003 shared task: Language-independent named entity recognition. In *CoNLL*.

[45] Joseph Turian, Lev Ratinov, and Yoshua Bengio. 2010. Word representations: A simple and general method for semi-supervised learning. In *Proceedings of the 48th Annual Meeting of the Association for Computational Linguistics*, ACL '10, pages 384–394.

[46] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In *Advances in Neural Information Processing Systems*, pages 6000–6010.

[47] Pascal Vincent, Hugo Larochelle, Yoshua Bengio, and Pierre-Antoine Manzagol. 2008. Extracting and composing robust features with denoising autoencoders. In *Proceedings of the 25th international conference on Machine learning*, pages 1096–1103. ACM.

[48] Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel Bowman. 2018a. GLUE: A multi-task benchmark and analysis platform for natural language understanding. In *Proceedings of the 2018 EMNLP Workshop BlackboxNLP: Analyzing and Interpreting Neural Networks for NLP*, pages 353–355.

[49] Wei Wang, Ming Yan, and Chen Wu. 2018b. Multi-granularity hierarchical attention fusion networks for reading comprehension and question answering. In *Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*. Association for Computational Linguistics.

[50] Alex Warstadt, Amanpreet Singh, and Samuel R Bowman. 2018. Neural network acceptability judgments. *arXiv preprint arXiv:1805.12471*.

[51] Adina Williams, Nikita Nangia, and Samuel R Bowman. 2018. A broad-coverage challenge corpus for sentence understanding through inference. In *NAACL*.

[52] Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Wolfgang Macherey, Maxim Krikun, Yuan Cao, Qin Gao, Klaus Macherey, et al. 2016. Google's neural machine translation system: Bridging the gap between human and machine translation. *arXiv preprint arXiv:1609.08144*.

[53] Jason Yosinski, Jeff Clune, Yoshua Bengio, and Hod Lipson. 2014. How transferable are features in deep neural networks? In *Advances in neural information processing systems*, pages 3320–3328.

[54] Adams Wei Yu, David Dohan, Minh-Thang Luong, Rui Zhao, Kai Chen, Mohammad Norouzi, and Quoc V Le. 2018. QANet: Combining local convolution with global self-attention for reading comprehension. In *ICLR*.

[55] Rowan Zellers, Yonatan Bisk, Roy Schwartz, and Yejin Choi. 2018. SWAG: A large-scale adversarial dataset for grounded commonsense inference. In *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing (EMNLP)*.

[56] Yukun Zhu, Ryan Kiros, Rich Zemel, Ruslan Salakhutdinov, Raquel Urtasun, Antonio Torralba, and Sanja Fidler. 2015. Aligning books and movies: Towards story-like visual explanations by watching movies and reading books. In *Proceedings of the IEEE international conference on computer vision*, pages 19–27.

---

## BERT：深度双向Transformer预训练用于语言理解（附录）

我们将附录组织为三部分：
- BERT的额外实现细节在附录A中呈现；
- 实验的额外细节在附录B中呈现；
- 额外的消融研究在附录C中呈现。

我们为BERT呈现额外的消融研究，包括：
- 训练步数的影响；以及
- 不同掩码程序的消融。

## A BERT的额外细节

### A.1 预训练任务说明

我们在下面提供预训练任务的示例。

**掩码LM和掩码程序** 假设未标记句子是"my dog is hairy"，在随机掩码过程中我们选择了第4个token（对应于hairy），我们的掩码程序可以进一步说明为：

- 80%的情况下：用`[MASK]`token替换该词，例如，"my dog is hairy" \to "my dog is [MASK]"
- 10%的情况下：用随机词替换该词，例如，"my dog is hairy" \to "my dog is apple"
- 10%的情况下：保持该词不变，例如，"my dog is hairy" \to "my dog is hairy"。这样做的目的是使表示偏向实际观察到的词。

这个过程的好处是Transformer编码器不知道哪些词将被要求预测，哪些已被随机词替换，因此它被迫保持每个输入token的分布式上下文表示。此外，由于随机替换只发生在所有token的1.5%（即15%的10%），这似乎不会损害模型的语言理解能力。在第C.2节中，我们评估了这一程序的影响。

与标准语言模型训练相比，掩码LM只对每个批次中15%的token进行预测，这表明模型可能需要更多的预训练步数才能收敛。在第C.1节中，我们证明MLM确实比从左到右模型（预测每个token）收敛稍慢，但MLM模型的实证改进远远超过增加的训练成本。

**下一句预测** 下一句预测任务可以通过以下示例说明：

$$
\begin{aligned}
&\text{Input} = [\text{CLS}]\ \text{the man went to}\ [\text{MASK}]\ \text{store}\ [\text{SEP}]\ \text{he bought a gallon}\ [\text{MASK}]\ \text{milk}\ [\text{SEP}] \\
&\text{Label} = \text{IsNext} \\
& \\
&\text{Input} = [\text{CLS}]\ \text{the man}\ [\text{MASK}]\ \text{to the store}\ [\text{SEP}]\ \text{penguin}\ [\text{MASK}]\ \text{are flight \#\#less birds}\ [\text{SEP}] \\
&\text{Label} = \text{NotNext}
\end{aligned}
$$

### A.2 预训练过程

为了生成每个训练输入序列，我们从语料库中采样两个文本跨度，我们称之为"句子"，即使它们通常比单个句子长得多（但也可能更短）。第一个句子接收A嵌入，第二个句子接收B嵌入。50%的情况下B是A之后实际的下一句，50%的情况下是随机句子，这是为了"下一句预测"任务。它们被采样使得组合长度 $\leq 512$ 个token。LM掩码在WordPiece token化后应用，统一掩码率为15%，对部分词片没有特殊处理。

我们使用256个序列的批量大小（256序列 \times 512 token = 128,000 token/批次）训练1,000,000步，这大约是在33亿词语料库上的40个epoch。我们使用Adam，学习率为1e-4， $\beta_1 = 0.9$ ， $\beta_2 = 0.999$ ，L2权重衰减为0.01，前10,000步学习率预热，然后线性衰减学习率。我们在所有层上使用0.1的dropout概率。遵循OpenAI GPT，我们使用GELU [19] 激活而非标准relu。训练损失是平均掩码LM似然和平均下一句预测似然之和。

BERT $_{\text{BASE}}$ 的训练在Pod配置下的4块Cloud TPU（共16个TPU芯片）上进行[13]。BERT $_{\text{LARGE}}$ 的训练在16块Cloud TPU（共64个TPU芯片）上进行。每次预训练需要4天完成。

较长的序列由于注意力机制与序列长度的二次关系而不成比例地昂贵。为了加速预训练，我们在90%的步骤中以序列长度128预训练模型。然后，我们在剩余的10%步骤中以序列长度512进行训练以学习位置嵌入。

### A.3 微调过程

对于微调，大多数模型超参数与预训练相同，但批量大小、学习率和训练epoch数除外。dropout概率始终保持在0.1。最优超参数值是任务特定的，但我们发现以下可能的值范围在所有任务上表现良好：

- 批量大小：16, 32
- 学习率（Adam）：5e-5, 3e-5, 2e-5
- epoch数：2, 3, 4

我们还观察到大数据集（如10万+标记训练样本）对超参数选择的敏感性远低于小数据集。微调通常非常快，因此简单地对上述参数进行穷举搜索并选择在开发集上表现最佳的模型是合理的。

### A.4 BERT、ELMo和OpenAI GPT的比较

这里我们研究近期流行的表示学习模型（包括ELMo、OpenAI GPT和BERT）的差异。模型架构之间的比较在图3中以可视化方式展示。注意，除了架构差异外，BERT和OpenAI GPT是微调方法，而ELMo是基于特征的方法。

与BERT最可比的现有预训练方法是OpenAI GPT，它在大型文本语料库上训练从左到右的Transformer LM。事实上，BERT中的许多设计决策都是有意使其尽可能接近GPT，以便两种方法可以最小化地比较。这项工作的核心论点是双向性和第3.1节中呈现的两个预训练任务解释了大部分实证改进，但我们注意到BERT和GPT的训练方式之间还有几个其他差异：

- GPT在BooksCorpus（8亿词）上训练；BERT在BooksCorpus（8亿词）和维基百科（25亿词）上训练。
- GPT使用句子分隔符（`[SEP]`）和分类器token（`[CLS]`），这些只在微调时引入；BERT在预训练期间学习`[SEP]`、`[CLS]`和句子A/B嵌入。
- GPT训练了1M步，批量大小为32,000词；BERT训练了1M步，批量大小为128,000词。
- GPT对所有微调实验使用相同的学习率5e-5；BERT选择在开发集上表现最佳的任务特定微调学习率。

为了隔离这些差异的影响，我们在第5.1节中进行消融实验，证明大部分改进实际上来自两个预训练任务及其实现的单向性。

### A.5 不同任务上微调的说明

BERT在不同任务上的微调说明可见图4。我们的任务特定模型通过将BERT与一个额外的输出层结合形成，因此需要从头学习的参数数量最少。在这些任务中，（a）和（b）是序列级任务，而（c）和（d）是token级任务。在图中， $E$ 表示输入嵌入， $T_i$ 表示token $i$ 的上下文表示，`[CLS]`是用于分类输出的特殊符号，`[SEP]`是用于分隔非连续token序列的特殊符号。

## B 详细实验设置

### B.1 GLUE基准实验的详细描述

我们在表1中的GLUE结果来自 https://gluebenchmark.com/leaderboard 和 https://blog.openai.com/language-unsupervised 。GLUE基准包括以下数据集，其描述最初总结于Wang等人（2018a）：

**MNLI** 多体裁自然语言推理是一个大规模、众包的蕴含分类任务[51]。给定一对句子，目标是预测第二个句子相对于第一个句子是蕴含、矛盾还是中性。

**QQP** Quora问题对是一个二分类任务，目标是确定Quora上提出的两个问题是否语义等价（Chen等人，2018）。

**QNLI** 问题自然语言推理是斯坦福问答数据集[39]的一个版本，已被转换为二分类任务[48]。正例是包含正确答案的（问题，句子）对，负例是来自同一段落但不包含答案的（问题，句子）对。

**SST-2** 斯坦福情感树库是一个二值单句分类任务，由从电影评论中提取的句子及其情感的人类标注组成[41]。

**CoLA** 语言可接受性语料库是一个二值单句分类任务，目标是预测一个英语句子在语言上是否"可接受"[50]。

**STS-B** 语义文本相似度基准是一个从新闻标题和其他来源抽取的句子对集合[8]。它们用1到5的分数标注，表示两个句子在语义意义上的相似程度。

**MRPC** 微软研究院释义语料库由从在线新闻来源自动提取的句子对组成，带有句子对是否语义等价的人类标注[17]。

**RTE** 识别文本蕴含是一个与MNLI类似的二值蕴含任务，但训练数据少得多[4][14]。

**WNLI** Winograd NLI是一个小型自然语言推理数据集[27]。GLUE网页指出该数据集的构建存在问题[15]，每个提交到GLUE的训练系统都表现差于预测多数类的65.1基线准确率。因此，为了公平对待OpenAI GPT，我们排除了这个集合。对于我们的GLUE提交，我们总是预测多数类[14]。

## C 额外消融研究

### C.1 训练步数的影响

图5展示了从预训练了 $k$ 步的检查点微调后的MNLI Dev准确率。这使我们能够回答以下问题：

1. **问题：** BERT真的需要如此大量的预训练（128,000词/批次 \times 1,000,000步）才能达到高微调准确率吗？**回答：** 是的，BERT $_{\text{BASE}}$ 在1M步训练时相比500k步在MNLI上获得了几乎1.0%的额外准确率。

2. **问题：** MLM预训练是否比LTR预训练收敛更慢，因为每个批次中只有15%的词被预测而不是每个词？**回答：** MLM模型确实比LTR模型收敛稍慢。然而，在绝对准确率方面，MLM模型几乎立即开始优于LTR模型。

### C.2 不同掩码程序的消融

在第3.1节中，我们提到BERT在使用掩码语言模型（MLM）目标预训练时使用混合策略来掩码目标token。以下是评估不同掩码策略效果的消融研究。

注意，掩码策略的目的是减少预训练和微调之间的不匹配，因为`[MASK]`符号在微调阶段从未出现。我们报告MNLI和NER的Dev结果。对于NER，我们报告微调和基于特征的方法，因为我们预计不匹配会因基于特征的方法而被放大，因为模型将没有机会调整表示。

**表8：不同掩码策略的消融。** 结果呈现在表8中。在表中，MASK表示用`[MASK]`符号替换目标token以用于MLM；SAME表示保持目标token不变；RND表示用另一个随机token替换目标token。表格左侧的数字表示MLM预训练期间使用的特定策略的概率（BERT使用80%、10%、10%）。右侧部分表示Dev集结果。对于基于特征的方法，我们拼接BERT的最后4层作为特征，这被证明是第5.3节中的最佳方法。从表中可以看出，微调对不同掩码策略出奇地鲁棒。然而，正如预期的那样，仅使用MASK策略在将基于特征的方法应用于NER时存在问题。有趣的是，仅使用RND策略的表现也远差于我们的策略。

---

![图1](图1未嵌入文本中，显示为ASCII示意图)
**图1：BERT的总体预训练和微调过程。** 除了输出层外，预训练和微调使用相同的架构。相同的预训练模型参数用于初始化不同下游任务的模型。在微调期间，所有参数都被微调。`[CLS]`是添加在每个输入示例前的特殊符号，`[SEP]`是特殊分隔符token（例如，分隔问题/答案）。

![图2](图2未嵌入文本中，显示为ASCII示意图)
**图2：BERT输入表示。** 输入嵌入是token嵌入、segment嵌入和position嵌入之和。

![图3](图3未嵌入文本中，显示为ASCII示意图)
**图3：预训练模型架构的差异。** BERT使用双向Transformer。OpenAI GPT使用从左到右的Transformer。ELMo使用独立训练的从左到右和从右到左LSTM的拼接来为下游任务生成特征。三者中，只有BERT表示在所有层中同时以左、右上下文为条件。除了架构差异外，BERT和OpenAI GPT是微调方法，而ELMo是基于特征的方法。

![图4](图4未嵌入文本中，显示为ASCII示意图)
**图4：BERT在不同任务上微调的说明。**

![图5](图5未嵌入文本中，显示为ASCII示意图)
**图5：训练步数的消融。** 显示了从预训练了 $k$ 步的模型参数微调后的MNLI准确率。x轴是 $k$ 的值。

---

**表1：GLUE测试结果**，由评估服务器（https://gluebenchmark.com/leaderboard ）评分。每个任务下方的数字表示训练样本数。"Average"列与官方GLUE分数略有不同，因为我们排除了有问题的 WNLI [27] 集[8]。BERT和OpenAI GPT是单模型、单任务。QQP和MRPC报告F1分数，STS-B [8] 报告 Spearman 相关性，其他任务报告准确率。我们排除了将BERT作为其组件之一的条目。

**表2：SQuAD 1.1结果。** BERT集成是7个使用不同预训练检查点和微调种子的系统。

**表3：SQuAD 2.0结果。** 我们排除了将BERT作为其组件之一的条目。

**表4：SWAG Dev和Test准确率。** †人类性能用100个样本测量，如SWAG论文中报告。

**表5：使用BERT $_{\text{BASE}}$ 架构对预训练任务进行消融。** "No NSP"在没有下一句预测任务的情况下训练。"LTR & No NSP"像OpenAI GPT一样作为从左到右LM训练，没有下一句预测。"+ BiLSTM"在微调期间在"LTR + No NSP"模型之上添加了随机初始化的BiLSTM。

**表6：BERT模型大小的消融。** #L = 层数；#H = 隐藏层大小；#A = 注意力头数。"LM (ppl)"是留出训练数据的掩码LM困惑度。

**表7：CoNLL-2003命名实体识别结果。** 超参数使用Dev集选择。报告的Dev和Test分数是使用这些超参数在5次随机重启上的平均值。

**表8：不同掩码策略的消融。**

---

> [1] https://github.com/tensorflow/tensor2tensor
> [2] http://nlp.seas.harvard.edu/2018/04/03/attention.html
> [3] 在所有情况下，我们将前馈/滤波器大小设置为 $4H$ ，即 $H=768$ 时为3072， $H=1024$ 时为4096。
> [4] 我们注意到在文献中，双向Transformer通常被称为"Transformer encoder"，而仅左侧上下文的版本被称为"Transformer decoder"，因为它可用于文本生成。
> [5] 最终模型在NSP上达到97%-98%的准确率。
> [6] 向量 $C$ 在没有微调的情况下不是一个有意义的句子表示，因为它是用NSP训练的。
> [7] 例如，BERT SQuAD模型可以在单个Cloud TPU上约30分钟内训练到Dev F1分数91.0%。
> [8] 见 https://gluebenchmark.com/faq 中的(10)。
> [9] GLUE数据集分布不包括Test标签，我们仅为BERT $_{\text{BASE}}$ 和BERT $_{\text{LARGE}}$ 各提交了一次GLUE评估服务器。
> [10] https://gluebenchmark.com/leaderboard
> [11] QANet [54] 有描述，但该系统的性能在发表后已大幅提升。
> [12] 我们使用的TriviaQA数据由TriviaQA-Wiki中的段落组成，这些段落由文档中的前400个token构成，且包含至少一个提供的可能答案。
> [13] https://cloudplatform.googleblog.com/2018/06/Cloud-TPU-now-offers-preemptible-pricing-and-global-availability.html
> [14] 注意，我们在本文中只报告单任务微调结果。多任务微调方法可能将性能推得更高。例如，我们确实观察到通过MNLI的多任务训练在RTE上获得了显著改进。
> [15] https://gluebenchmark.com/faq
