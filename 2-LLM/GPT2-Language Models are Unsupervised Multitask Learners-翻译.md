# Language Models are Unsupervised Multitask Learners（语言模型是无监督多任务学习者）

> Alec Radford\*¹, Jeffrey Wu\*¹, Rewon Child¹, David Luan¹, Dario Amodei\*\*¹, Ilya Sutskever\*\*¹
> ¹OpenAI, San Francisco, California, United States
> 通讯作者：Alec Radford \<alec@openai.com\>

本文证明，当语言模型在一个名为 WebText 的数百万网页新数据集上训练时，无需任何显式监督即可开始学习这些任务。当以文档加问题为条件时，语言模型生成的答案在 CoQA 数据集上达到 55 F1——在不使用 127,000+ 训练样本的情况下匹配或超过 4 个基线系统中的 3 个。语言模型的容量对于零样本任务迁移的成功至关重要，增加容量会在所有任务上以对数线性方式提升性能。我们最大的模型 GPT-2 是一个 15 亿参数的 Transformer，在零样本设置下，在 8 个测试的语言建模数据集中有 7 个达到了最先进结果，但它仍然欠拟合 WebText。该模型的样本反映了这些改进，并包含连贯的文本段落。这些发现为构建能够从自然发生的演示中学习执行任务的语言处理系统指明了一条有希望的路径。

---

## 摘要

自然语言处理任务，如问答、机器翻译、阅读理解和摘要，通常通过在有监督的任务特定数据集上进行学习来处理。我们证明，当语言模型在一个名为 WebText 的数百万网页新数据集上训练时，无需任何显式监督即可开始学习这些任务。当以文档加问题为条件时，语言模型生成的答案在 CoQA 数据集上达到 55 F1——在不使用 127,000+ 训练样本的情况下匹配或超过 4 个基线系统中的 3 个。语言模型的容量对于零样本任务迁移的成功至关重要，增加容量会在所有任务上以对数线性方式提升性能。我们最大的模型 GPT-2 是一个 15 亿参数的 Transformer，在零样本设置下，在 8 个测试的语言建模数据集中有 7 个达到了最先进结果，但它仍然欠拟合 WebText。该模型的样本反映了这些改进，并包含连贯的文本段落。这些发现为构建能够从自然发生的演示中学习执行任务的语言处理系统指明了一条有希望的路径。

## 1. 引言

机器学习系统目前通过结合大规模数据集、高容量模型和监督学习 [1][2][3]，在它们所训练的任务上表现出色（在期望意义下）。然而，这些系统是脆弱的，对数据分布 [4] 和任务规范 [5] 的微小变化都很敏感。当前系统更适合被描述为狭隘的专家而非有能力的通才。我们希望朝着更通用的系统迈进——这些系统最终无需为每个任务手动创建和标注训练数据集。

构建机器学习系统的主流方法是：为目标任务收集一个包含正确行为训练样本的数据集，训练系统模仿这些行为，然后在独立同分布（IID）的留出样本上测试其性能。这种方法在推动狭隘专家系统的进步方面是有效的。但字幕模型 [6]、阅读理解系统 [7] 和图像分类器 [8] 在面对多样化的输入时经常出现不稳定的行为，这凸显了这种方法的缺陷。

我们怀疑，在单领域数据集上进行单任务训练的普遍性是当前系统缺乏泛化能力的主要原因。要在当前架构下构建鲁棒系统，可能需要在广泛的领域和任务上进行训练和性能度量。最近，一些基准测试被提出，如 GLUE [9] 和 decaNLP [10]，以开始研究这一问题。

多任务学习 [11] 是一种有希望的提高通用性能的框架。然而，NLP 中的多任务训练仍处于起步阶段。最近的工作报告了适度的性能提升 [12]，而迄今为止最雄心勃勃的两项工作分别在总共 10 和 17 个（数据集，目标）对上进行了训练 [10][13]。从元学习的角度来看，每个（数据集，目标）对是从数据集和目标的分布中采样得到的单个训练样本。当前的机器学习系统需要成百上千个样本才能归纳出泛化良好的函数。这表明，多任务训练可能需要同样多的有效训练对才能用当前方法实现其前景。用现有技术强行推进到那里可能需要不断扩大数据集的创建和目标的设计，这将非常困难。这促使我们探索执行多任务学习的其他设置。

当前语言任务上表现最佳的系统利用了预训练和监督微调的组合。这种方法有着悠久的历史，并趋向于更灵活的迁移形式。首先，词向量被学习并用作任务特定架构的输入 [14][15]；然后，循环网络的上下文表示被迁移 [16][17]；最近的工作表明，任务特定架构不再必要，迁移多个自注意力块就足够了 [18][19]。这些方法仍然需要监督训练才能执行任务。当只有极少或没有监督数据可用时，另一条研究路线证明了语言模型在执行特定任务（如常识推理 [20] 和情感分析 [21]）方面的潜力。

在本文中，我们连接了这两条研究路线，并延续了更通用迁移方法的趋势。我们证明语言模型可以在零样本设置下执行下游任务——无需任何参数或架构修改。我们通过突出语言模型在零样本设置下执行广泛任务的能力来展示这种方法的潜力。根据任务的不同，我们取得了有希望的、有竞争力的和最先进的结果。

## 2. 方法

我们方法的核心是语言建模。语言建模通常被表述为从一组样本 $(x_1, x_2, \ldots, x_n)$ 中进行无监督分布估计，每个样本由可变长度的符号序列 $(s_1, s_2, \ldots, s_n)$ 组成。由于语言具有自然的顺序性，通常将符号上的联合概率分解为条件概率的乘积 [22][23]：

$$
p(x) = \prod_{i=1}^{n} p(s_n \mid s_1, \ldots, s_{n-1}) \qquad (1)
$$

这种方法允许对 $p(x)$ 以及形如 $p(s_{n-k}, \ldots, s_n \mid s_1, \ldots, s_{n-k-1})$ 的任何条件概率进行可处理的采样和估计。近年来，能够计算这些条件概率的模型的表达能力有了显著提高，例如自注意力架构 Transformer [24]。

学习执行单个任务可以在概率框架中表示为估计条件分布 $p(\text{output} \mid \text{input})$ 。由于通用系统应该能够执行许多不同的任务，即使对于相同的输入，它也应该不仅以输入为条件，还要以要执行的任务为条件。也就是说，它应该建模 $p(\text{output} \mid \text{input}, \text{task})$ 。这在多任务和元学习设置中有多种形式化方式。任务条件通常在架构层面实现，例如 [25] 中的任务特定编码器和解码器，或在算法层面实现，例如 MAML [26] 的内外循环优化框架。但正如 McCann 等人 [10] 所展示的，语言提供了一种灵活的方式来将任务、输入和输出全部指定为符号序列。例如，一个翻译训练样本可以写成序列 `(translate to french, english text, french text)`。同样，一个阅读理解训练样本可以写成 `(answer the question, document, question, answer)`。McCann 等人 [10] 证明，可以训练单个模型 MQAN 来推断并执行这种格式的多个不同任务。

语言模型原则上也能够学习 McCann 等人 [10] 的任务，而无需对哪些符号是要预测的输出进行显式监督。由于监督目标与无监督目标相同，但仅在序列的子集上进行评估，因此无监督目标的全局最小值也是监督目标的全局最小值。在这个略显玩具化的设置中，[27] 中讨论的关于密度估计作为原则性训练目标的担忧被绕过了。问题反而变成了我们是否能够在实践中优化无监督目标直至收敛。初步实验证实，足够大的语言模型能够在这种玩具式设置中执行多任务学习，但学习速度比显式监督方法慢得多。

虽然从上述良好定义的设置到"野外语言"的混乱是一个巨大的跨越，但 Weston [28] 在对话的背景下论证了开发能够直接从自然语言学习的系统的必要性，并展示了一个概念验证——通过使用教师输出的前向预测，在没有奖励信号的情况下学习问答任务。虽然对话是一种有吸引力的方法，但我们担心它过于局限。互联网包含大量被动可用的信息，无需交互式通信。我们推测，具有足够容量的语言模型将开始学习推断和执行自然语言序列中演示的任务，以便更好地预测它们，无论这些序列是如何获取的。如果语言模型能够做到这一点，它实际上就是在执行无监督多任务学习。我们通过分析语言模型在多种任务上的零样本设置性能来检验这种情况是否成立。

### 2.1. 训练数据集

大多数先前的工作在单一文本领域上训练语言模型，如新闻文章 [29]、维基百科 [30] 或虚构书籍 [31]。我们的方法促使我们构建尽可能大和多样化的数据集，以便在最多样化的领域和上下文中收集任务的自然语言演示。

多样且几乎无限的文本的一个有希望的来源是网络爬取，如 Common Crawl。虽然这些存档比当前的语言建模数据集大数个数量级，但它们存在显著的数据质量问题。Trinh & Le [32] 在他们关于常识推理的工作中使用了 Common Crawl，但注意到大量文档"内容大多难以理解"。我们在用 Common Crawl 进行的初步实验中也观察到了类似的数据问题。Trinh & Le [32] 的最佳结果是使用 Common Crawl 的一个小子集实现的，该子集仅包含与其目标数据集（Winograd Schema Challenge）最相似的文档。虽然这是提高特定任务性能的实用方法，但我们希望避免预先假设要执行的任务。

表 1：在整个 WebText 训练集中发现的英语到法语和法语到英语翻译的自然发生演示示例。

因此，我们创建了一个新的网络爬取，强调文档质量。为此，我们只爬取了经过人工策划/过滤的网页。手动过滤完整的网络爬取将异常昂贵，因此作为起点，我们爬取了 Reddit（一个社交媒体平台）上所有获得至少 3 个 karma 的外部链接。这可以视为一个启发式指标，用于判断其他用户是否认为该链接有趣、有教育意义或只是有趣。

由此产生的数据集 WebText 包含这些 4500 万个链接的文本子集。为了从 HTML 响应中提取文本，我们结合使用了 Dragnet [33] 和 Newspaper¹ 内容提取器。本文呈现的所有结果均使用 WebText 的初步版本，该版本不包含 2017 年 12 月之后创建的链接，并且经过去重和基于启发式的清洗后，略超过 800 万篇文档，总计 40 GB 文本。我们从 WebText 中移除了所有维基百科文档，因为它是其他数据集的常见数据源，并且可能因训练数据与测试评估任务重叠而使分析复杂化。

¹ https://github.com/codelucas/newspaper

### 2.2. 输入表示

通用语言模型（LM）应该能够计算（并生成）任何字符串的概率。当前的大规模 LM 包含预处理步骤，如小写化、tokenization 和词表外 token，这些限制了可建模字符串的空间。虽然将 Unicode 字符串作为 UTF-8 字节序列处理可以优雅地满足这一要求，如 Gillick 等人 [34] 的工作所示，但当前的字节级 LM 在大规模数据集（如 One Billion Word Benchmark）上无法与词级 LM 竞争 [35]。我们在自己尝试在 WebText 上训练标准字节级 LM 时也观察到了类似的性能差距。

字节对编码（BPE）[36] 是字符级和词级语言建模之间的一个实用中间地带，它有效地在频繁符号序列的词级输入和不频繁符号序列的字符级输入之间进行插值。尽管名称如此，参考 BPE 实现通常操作于 Unicode 码点而非字节序列。这些实现需要包含完整的 Unicode 符号空间才能建模所有 Unicode 字符串。这将导致在添加任何多符号 token 之前就拥有超过 130,000 的基础词汇量。与 BPE 通常使用的 32,000 到 64,000 token 词表相比，这过于庞大。相比之下，字节级版本的 BPE 只需要大小为 256 的基础词汇量。然而，直接将 BPE 应用于字节序列会导致次优的合并，因为 BPE 使用基于贪心频率的启发式方法来构建 token 词表。我们观察到 BPE 包含了许多常见词的多个版本，如 `dog`、`dog.`、`dog!`、`dog?`，因为它们以多种变体出现。这导致了有限的词表槽位和模型容量的次优分配。为了避免这种情况，我们阻止 BPE 在任何字节序列的字符类别之间进行合并。我们对空格添加了一个例外，这显著提高了压缩效率，同时仅增加了极少的词跨多个词表 token 的碎片化。

这种输入表示使我们能够结合词级 LM 的经验性优势与字节级方法的通用性。由于我们的方法可以为任何 Unicode 字符串分配概率，这使我们能够在任何数据集上评估我们的 LM，无论其预处理、tokenization 或词表大小如何。

### 2.3. 模型

我们使用基于 Transformer [24] 的架构作为我们的 LM。该模型很大程度上遵循了 OpenAI GPT 模型 [18] 的细节，并进行了一些修改。层归一化 [37] 被移到每个子块的输入，类似于预激活残差网络 [38]，并在最终的自注意力块之后添加了一个额外的层归一化。我们使用了修改后的初始化，考虑了残差路径上随模型深度的累积。我们在初始化时将残差层的权重按 $1 / \sqrt{N}$ 缩放，其中 $N$ 是残差层的数量。词表扩展到 50,257。我们还将上下文大小从 512 增加到 1024 个 token，并使用更大的批大小 512。

表 2：4 种模型规模的架构超参数。

| 参数 | 层数 | $d_{\text{model}}$ |
|------|------|-------------------|
| 117M | 12   | 768               |
| 345M | 24   | 1024              |
| 762M | 36   | 1280              |
| 1542M | 48  | 1600              |

## 3. 实验

我们训练并评测了四个大约对数均匀间隔规模的 LM。架构总结在表 2 中。最小的模型等同于原始 GPT，第二小的等同于 BERT [19] 的最大模型。我们最大的模型称为 GPT-2，其参数比 GPT 多一个数量级以上。每个模型的学习率经过手动调整，以在 5% 的 WebText 留出样本上获得最佳困惑度。所有模型仍然欠拟合 WebText，并且迄今留出困惑度随着更多训练时间而持续改善。

### 3.1. 语言建模

作为向零样本任务迁移迈出的第一步，我们有兴趣了解 WebText LM 在它们所训练的主要任务——语言建模上的零样本领域迁移表现如何。由于我们的模型在字节级别上运行，不需要有损预处理或 tokenization，我们可以在任何语言模型基准上进行评估。语言建模数据集的结果通常以每个规范预测单元（通常是一个字符、一个字节或一个词）的平均负对数概率的缩放或指数化形式报告。我们通过根据 WebText LM 计算数据集的 log 概率并除以规范单元的数量来评估相同的量。对于其中许多数据集，WebText LM 将在显著分布外的情况下进行测试，需要预测经过激进标准化的文本、tokenization 伪影（如断开的标点和缩写）、打乱的句子，甚至是在 WebText 中极其罕见的 `<UNK>` 字符串——在 400 亿字节中仅出现 26 次。我们在表 3 中使用可逆的 de-tokenizer 报告主要结果，这些 de-tokenizer 尽可能多地移除这些 tokenization/预处理伪影。由于这些 de-tokenizer 是可逆的，我们仍然可以计算数据集的 log 概率，它们可以被视为一种简单的领域适应形式。我们观察到使用这些 de-tokenizer 后 GPT-2 的困惑度提升了 2.5 到 5。

WebText LM 在领域和数据集之间迁移良好，在零样本设置下将 8 个数据集中的 7 个的最先进水平提高了。在小型数据集如 Penn Treebank 和 WikiText-2（仅有 100 到 200 万训练 token）上观察到大幅改进。在旨在测量长期依赖关系的数据集如 LAMBADA [39] 和 Children's Book Test [40] 上也观察到了大幅改进。我们的模型在 One Billion Word Benchmark [41] 上仍然明显差于先前的工作。这可能是由于它既是最大的数据集，又具有最具破坏性的预处理——1BW 的句子级打乱移除了所有长程结构。

表 3：多个数据集上的零样本结果。这些结果均未进行任何训练或微调。PTB 和 WikiText-2 的结果来自 [42]。CBT 的结果来自 [43]。LAMBADA 准确率结果来自 [44]，LAMBADA 困惑度结果来自 [45]。其他结果来自 [46]。

| 数据集 | 指标 | SOTA | 117M | 345M | 762M | 1542M |
|--------|------|------|------|------|------|-------|
| LAMBADA | (PPL) | 99.8 | 35.13 | 15.60 | 10.87 | 8.63 |
| LAMBADA | (ACC) | 59.23 | 45.99 | 55.48 | 60.12 | 63.24 |
| CBT-CN | (ACC) | 85.7 | 87.65 | 92.35 | 93.45 | 93.30 |
| CBT-NE | (ACC) | 82.3 | 83.4 | 87.1 | 88.0 | 89.05 |
| WikiText2 | (PPL) | 39.14 | 29.41 | 22.76 | 19.93 | 18.34 |
| PTB | (PPL) | 46.54 | 65.85 | 47.33 | 40.31 | 35.76 |
| enwik8 | (BPB) | 0.99 | 1.16 | 1.01 | 0.97 | 0.93 |
| text8 | (BPC) | 1.08 | 1.17 | 1.06 | 1.02 | 0.98 |
| WikiText103 | (PPL) | 18.3 | 37.50 | 26.37 | 22.05 | 17.48 |
| 1BW | (PPL) | 21.8 | 75.20 | 55.72 | 44.57 | 42.16 |

### 3.2. Children's Book Test

图 2：Children's Book Test 上作为模型容量函数的性能。人类表现来自 [43]，而非原始论文中低得多的估计值。

Children's Book Test（CBT）[40] 旨在检验 LM 在不同类别词上的表现：命名实体、名词、动词和介词。CBT 不报告困惑度作为评估指标，而是在自动构建的完形填空测试上报告准确率，其任务是预测 10 个可能的被省略词选项中哪一个是正确的。遵循原始论文中介绍的 LM 方法，我们根据 LM 计算每个选项以及句子其余部分在该选项条件下的概率，并预测概率最高的选项。如图 2 所示，性能随着模型规模的增大而稳步提升，并在该测试上缩小了与人类表现的大部分差距。数据重叠分析显示，CBT 测试集书籍之一——Rudyard Kipling 的《The Jungle Book》——存在于 WebText 中，因此我们报告验证集上的结果（该验证集没有显著重叠）。GPT-2 在普通名词上达到了 93.3% 的新最先进结果，在命名实体上达到了 89.1%。我们应用了 de-tokenizer 以从 CBT 中移除 PTB 风格的 tokenization 伪影。

### 3.3. LAMBADA

LAMBADA 数据集 [39] 测试系统建模文本中长期依赖关系的能力。任务是预测句子的最后一个词，该词需要至少 50 个 token 的上下文才能被人类成功预测。GPT-2 将最先进水平从 99.8 困惑度 [45] 提升到 8.6 困惑度，并将 LM 在该测试上的准确率从 19% [47] 提高到 52.66%。调查 GPT-2 的错误发现，大多数预测是句子的有效延续，但不是有效的最后一个词。这表明 LM 没有使用"该词必须是句子最后一个词"这一额外有用的约束。添加一个停用词过滤器作为对此的近似，进一步将准确率提高到 63.24%，在该任务上整体最先进水平提高了 4%。先前的最高水平 [44] 使用了一种不同的受限预测设置，将模型的输出限制为仅出现在上下文中的词。对于 GPT-2，这种限制是有害的而非有益的，因为 19% 的答案不在上下文之中。我们使用未经预处理的版本的数据集。

### 3.4. Winograd Schema Challenge

图 3：Winograd Schema Challenge 上作为模型容量函数的性能。

Winograd Schema Challenge [48] 旨在通过衡量系统解析文本中歧义的能力来评估系统执行常识推理的能力。最近，Trinh & Le [32] 使用 LM 在该挑战上取得了显著进展，方法是以更高概率预测歧义的消解。我们遵循他们的问题表述，并在图 3 中展示了我们模型在完整评分和部分评分技术下的性能。GPT-2 将最先进准确率提高了 7%，达到 70.70%。该数据集相当小，只有 273 个样本，因此我们建议阅读 Trichelair 等人 [49] 以帮助理解这一结果。

### 3.5. 阅读理解

Conversation Question Answering 数据集（CoQA）[50] 包含来自 7 个不同领域的文档，以及提问者和回答者之间关于文档的自然语言对话。CoQA 测试阅读理解能力，以及模型回答依赖于对话历史（如"为什么？"）的问题的能力。

当以文档、相关对话历史以及最终 token `A:` 为条件时，GPT-2 的贪心解码在开发集上达到 55 F1。这匹配或超过了 4 个基线系统中的 3 个的性能，而无需使用这些基线所训练的 127,000+ 手动收集的问答对。监督 SOTA——一个基于 BERT 的系统 [19]——正接近人类 89 F1 的表现。虽然 GPT-2 的表现对于一个没有监督训练的系统来说是令人兴奋的，但对其答案和错误的一些检查表明，GPT-2 经常使用简单的基于检索的启发式方法，例如用文档中的名字回答"谁"的问题。

表 4：CNN 和 Daily Mail 数据集上由 ROUGE F1 指标衡量的摘要性能。Bottom-Up Sum 是来自 [51] 的 SOTA 模型。

| 模型 | R-1 | R-2 | R-L | R-AVG |
|------|-----|-----|-----|-------|
| Bottom-Up Sum | 41.22 | 18.68 | 38.34 | 32.75 |
| Lede-3 | 40.38 | 17.66 | 36.62 | 31.55 |
| Seq2Seq + Attn | 31.33 | 11.81 | 28.83 | 23.99 |
| GPT-2 TL;DR: | 29.34 | 8.27 | 26.58 | 21.40 |
| Random-3 | 28.78 | 8.63 | 25.52 | 20.98 |
| GPT-2 no hint | 21.58 | 4.03 | 19.47 | 15.03 |

### 3.6. 摘要

我们在 CNN 和 Daily Mail 数据集 [52] 上测试 GPT-2 执行摘要的能力。为了引导摘要行为，我们在文章后添加文本 `TL;DR:`，并使用 $k = 2$ 的 Top-k 随机采样 [53] 生成 100 个 token，这减少了重复并鼓励比贪心解码更具抽象性的摘要。我们使用这 100 个 token 中前 3 个生成的句子作为摘要。虽然从定性上看，生成的文本类似于摘要（如表 14 所示），但它们通常关注文章中的近期内容或混淆具体细节（如事故涉及多少辆车，或者标志是在帽子上还是衬衫上）。在常用的 ROUGE 1,2,L 指标上，生成的摘要仅开始接近经典神经基线的性能，并且勉强优于从文章中随机选择 3 个句子。当移除任务提示时，GPT-2 的性能在聚合指标上下降了 6.4 分，这证明了在语言模型中使用自然语言调用特定任务行为的能力。

### 3.7. 翻译

我们测试 GPT-2 是否已经开始学习如何在不同语言之间进行翻译。为了帮助它推断这是期望的任务，我们以 `english sentence = french sentence` 格式的示例对上下文为条件，然后在最后的提示 `english sentence =` 之后，我们使用贪心解码从模型中采样，并使用第一个生成的句子作为翻译。在 WMT-14 英法测试集上，GPT-2 获得了 5 BLEU，略低于使用先前无监督词翻译工作中推断的双语词典进行的逐词替换 [54]。在 WMT-14 法英测试集上，GPT-2 能够利用其非常强大的英语语言模型表现得更好，达到 11.5 BLEU。这优于 [55] 和 [56] 的几个无监督机器翻译基线，但仍然远低于当前最佳无监督机器翻译方法 [57] 的 33.5 BLEU。这个任务上的表现令我们惊讶，因为我们故意从 WebText 中移除非英语网页作为过滤步骤。为了确认这一点，我们在 WebText 上运行了一个字节级语言检测器²，它仅检测到 10 MB 的法语数据，这比先前无监督机器翻译研究中常用的单语法语语料库小大约 500 倍。

² https://github.com/CLD2Owners/cld2

### 3.8. 问答

测试语言模型包含什么信息的潜在方法是评估它生成事实型问题正确答案的频率。之前关于神经系统中所有信息存储在参数中的行为展示，如 A Neural Conversational Model [58]，由于缺乏高质量的评估数据集而报告了定性结果。最近引入的 Natural Questions 数据集 [59] 是更定量地测试这一点的有希望资源。与翻译类似，语言模型的上下文被植入示例问答对，这有助于模型推断数据集的简短答案风格。当通过常用于阅读理解数据集（如 SQUAD³）的精确匹配指标评估时，GPT-2 正确回答了 4.1% 的问题。作为比较点，最小的模型没有超过一个极其简单的基线（为每个问题类型（谁、什么、哪里等）返回最常见的答案）的 1.0% 准确率。GPT-2 正确回答的问题数量是其 5.3 倍，这表明模型容量一直是神经系统中此类任务表现不佳的主要因素。GPT-2 分配给其生成答案的概率校准良好，在其最有信心的 1% 的问题上准确率达到 63.1%。GPT-2 在开发集问题上生成的最有信心的 30 个答案如表 5 所示。GPT-2 的表现仍然远低于混合信息检索和抽取式文档问答的开放域问答系统 30% 到 50% 的范围 [60]。

³ Alec，以前自认为擅长随机冷知识，在与 GPT-2 相同的设置下测试时正确回答了 100 个随机样本中的 17 个。他实际上只答对了 14 个，但他应该得到另外那 3 个。

表 5：GPT-2 在 Natural Questions 开发集上生成的按 GPT-2 概率排序的 30 个最有信心的答案。根据第 4 节所述的程序，这些问题均未出现在 WebText 中。

| 问题 | 生成的答案 | 正确 | 概率 |
|------|-----------|------|------|
| Who wrote the book the origin of species? | Charles Darwin | ✓ | 83.4% |
| Who is the founder of the ubuntu project? | Mark Shuttleworth | ✓ | 82.0% |
| Who is the quarterback for the green bay packers? | Aaron Rodgers | ✓ | 81.1% |
| Panda is a national animal of which country? | China | ✓ | 76.8% |
| Who came up with the theory of relativity? | Albert Einstein | ✓ | 76.4% |
| When was the first star wars film released? | 1977 | ✓ | 71.4% |
| What is the most common blood type in sweden? | A | ✗ | 70.6% |
| Who is regarded as the founder of psychoanalysis? | Sigmund Freud | ✓ | 69.3% |
| Who took the first steps on the moon in 1969? | Neil Armstrong | ✓ | 66.8% |
| Who is the largest supermarket chain in the uk? | Tesco | ✓ | 65.3% |
| What is the meaning of shalom in english? | peace | ✓ | 64.0% |
| Who was the author of the art of war? | Sun Tzu | ✓ | 59.6% |
| Largest state in the us by land mass? | California | ✗ | 59.2% |
| Green algae is an example of which type of reproduction? | parthenogenesis | ✗ | 56.5% |
| Vikram samvat calender is official in which country? | India | ✓ | 55.6% |
| Who is mostly responsible for writing the declaration of independence? | Thomas Jefferson | ✓ | 53.3% |
| What us state forms the western boundary of montana? | Montana | ✗ | 52.3% |
| Who plays ser davos in game of thrones? | Peter Dinklage | ✗ | 52.1% |
| Who appoints the chair of the federal reserve system? | Janet Yellen | ✗ | 51.5% |
| State the process that divides one nucleus into two genetically identical nuclei? | mitosis | ✓ | 50.7% |
| Who won the most mvp awards in the nba? | Michael Jordan | ✗ | 50.2% |
| What river is associated with the city of rome? | the Tiber | ✓ | 48.6% |
| Who is the first president to be impeached? | Andrew Johnson | ✓ | 48.3% |
| Who is the head of the department of homeland security 2017? | John Kelly | ✓ | 47.0% |
| What is the name given to the common currency to the european union? | Euro | ✓ | 46.8% |
| What was the emperor name in star wars? | Palpatine | ✓ | 46.5% |
| Do you have to have a gun permit to shoot at a range? | No | ✓ | 46.4% |
| Who proposed evolution in 1859 as the basis of biological development? | Charles Darwin | ✓ | 45.7% |
| Nuclear power plant that blew up in russia? | Chernobyl | ✓ | 45.7% |
| Who played john connor in the original terminator? | Arnold Schwarzenegger | ✗ | 45.2% |

## 4. 泛化 vs 记忆

最近在计算机视觉领域的研究表明，常见的图像数据集包含非平凡数量的近似重复图像。例如，CIFAR-10 在训练和测试图像之间有 3.3% 的重叠 [61]。这导致机器学习系统的泛化性能被高估。随着数据集规模的增加，这个问题变得越来越可能，这表明类似的现象可能也发生在 WebText 上。因此，分析有多少测试数据也出现在训练数据中是很重要的。

为了研究这一点，我们创建了包含 WebText 训练集 token 的 8-gram 的 Bloom 过滤器。为了提高召回率，字符串被规范化为仅包含小写字母数字词，使用单个空格作为分隔符。Bloom 过滤器的构建使得假阳性率上限为 $1/10^8$ 。我们通过生成 100 万字符串进一步验证了低假阳性率，其中没有被过滤器发现的。

这些 Bloom 过滤器让我们能够计算，给定一个数据集，该数据集的 8-gram 有多少百分比也出现在 WebText 训练集中。表 6 显示了常见 LM 基准测试集的这种重叠分析。常见 LM 数据集的测试集与 WebText 训练集的重叠在 1% 到 6% 之间，平均重叠为 3.2%。有些令人惊讶的是，许多数据集与自己的训练集有更大的重叠，平均为 5.9%。

表 6：测试集 8-gram 与训练集重叠的百分比。

| 数据集 | 测试集与自身训练集重叠 | 测试集与 WebText 训练集重叠 |
|--------|-----------------------|---------------------------|
| PTB | 2.67% | 0.88% |
| WikiText-2 | 0.66% | 1.63% |
| enwik8 | 7.50% | 6.31% |
| text8 | 2.34% | 3.94% |
| Wikitext-103 | 9.09% | 2.42% |
| 1BW | 13.19% | 3.75% |

我们的方法优化了召回率，虽然对重叠部分的手动检查显示了许多常见短语，但也有许多较长的匹配是由于重复数据造成的。这不是 WebText 独有的。例如，我们发现 WikiText-103 的测试集中有一篇文章也出现在训练数据集中。由于测试集中只有 60 篇文章，因此重叠至少为 1.6%⁴。可能更令人担忧的是，根据我们的程序，1BW 与自己的训练集有近 13.2% 的重叠。

对于 Winograd Schema Challenge，我们只发现了 10 个模式与 WebText 训练集有任何 8-gram 重叠。其中，2 个是虚假匹配。在剩下的 8 个中，只有 1 个模式出现在任何泄露答案的上下文中。

对于 CoQA，新闻领域约 15% 的文档已经存在于 WebText 中，模型在这些文档上的表现提高了约 3 F1。CoQA 的开发集指标报告了 5 个不同领域的平均表现，我们测量到由于各领域的重叠，增益约为 0.5-1.0 F1。然而，没有实际的训练问题或答案出现在 WebText 中，因为 CoQA 是在 WebText 链接截止日期之后发布的。

在 LAMBADA 上，平均重叠为 1.2%。GPT-2 在重叠超过 15% 的样本上表现约好 2 个困惑度。当排除所有有任何重叠的样本重新计算指标时，困惑度从 8.6 变为 8.7，准确率从 63.2% 降至 62.9%。整体结果的这种非常小的变化很可能是由于每 200 个样本中只有 1 个有显著重叠。

总的来说，我们的分析表明，WebText 训练数据与特定评估数据集之间的数据重叠对报告的结果提供了微小但一致的益处。然而，对于大多数数据集，我们没有观察到比标准训练集和测试集之间已经存在的重叠显著更大的重叠，如表 6 所强调的。

理解和量化高度相似的文本如何影响性能是一个重要的研究问题。更好的去重技术，如可扩展的模糊匹配，也有助于更好地回答这些问题。目前，我们建议在创建新的 NLP 数据集的训练和测试划分时，使用基于 n-gram 重叠的去重作为重要的验证步骤和合理性检查。

确定 WebText LM 的性能是否可归因于记忆的另一种潜在方法是检查它们在自己留出集上的表现。如图 4 所示，WebText 训练集和测试集上的表现相似，并且随着模型规模的增加而一起改善。这表明即使是 GPT-2 在许多方面仍然欠拟合 WebText。

GPT-2 也能够撰写关于发现会说话的独角兽的新闻文章。表 13 中提供了一个示例。

⁴ 额外重叠的很大一部分是由于编辑在共享同一主题的多个文章中重复使用某些段落，例如朝鲜战争中的各种战役。

## 5. 相关工作

这项工作的很大一部分测量了在更大数据集上训练的更大语言模型的性能。这与 Jozefowicz 等人 [29] 的工作类似，他们在 1 Billion Word Benchmark 上扩展了基于 RNN 的语言模型。Bajgar 等人 [43] 之前也通过在标准训练数据集之外从 Project Gutenberg 创建更大的训练数据集，改进了 Children's Book Test 上的结果。Hestness 等人 [62] 对各种深度学习模型的性能如何随模型容量和数据集大小而变化进行了彻底分析。我们的实验虽然在任务间噪声更大，但表明类似的趋势对于目标子任务成立，并且延续到 1B+ 参数规模。

生成模型中有趣的学习功能此前已有记录，例如 RNN 语言模型中的细胞执行行宽跟踪和引文/注释检测 [63]。对我们的工作更具启发性的是 Liu 等人 [64] 的观察：一个训练用于生成维基百科文章的模型也学会了在语言之间翻译名称。

先前的工作探索了过滤和构建大型网页文本语料库的替代方法，例如 iWeb 语料库 [65]。

对于语言任务的预训练方法已有广泛的研究。除了引言中提到的那些，GloVe [66] 将词向量表示学习扩展到整个 Common Crawl。早期关于文本深度表示学习的有影响力工作是 Skip-thought Vectors [31]。McCann 等人 [67] 探索了使用从机器翻译模型导出的表示，Howard & Ruder [68] 改进了基于 RNN 的微调方法 [16]。Conneau 等人 [69] 研究了自然语言推理模型所学表示的迁移性能，Subramanian 等人 [70] 探索了大规模多任务训练。

Ramachandran 等人 [71] 证明了 seq2seq 模型从使用预训练语言模型作为编码器和解码器初始化中受益。最近的工作表明，LM 预训练在微调用于困难生成任务（如闲聊对话和基于对话的问答系统）时也是有帮助的 [72][73]。

## 6. 讨论

大量研究致力于学习 [74]、理解 [75] 和批判性评估 [76] 监督和无监督预训练方法的表示。我们的结果表明，无监督任务学习是另一个有希望探索的研究领域。这些发现可能有助于解释预训练技术在下游 NLP 任务中广泛成功的原因，因为我们表明，在极限情况下，这些预训练技术之一开始直接学习执行任务，而无需监督适应或修改。

在阅读理解方面，GPT-2 在零样本设置下的性能与监督基线具有竞争力。然而，在其他任务（如摘要）上，虽然它在定性上执行了任务，但根据定量指标，其性能仍然只是初级的。虽然作为研究结果具有提示意义，但就实际应用而言，GPT-2 的零样本性能仍远未达到可用水平。

我们研究了 WebText LM 在许多经典 NLP 任务上的零样本性能，但还有许多额外的任务可以被评估。毫无疑问，在许多实际任务中，GPT-2 的性能仍然不比随机好。即使在我们评估的常见任务（如问答和翻译）上，语言模型也只有在具有足够容量时才开始优于简单的基线。

虽然零样本性能确立了 GPT-2 在许多任务上的潜在性能基线，但尚不清楚微调上限在哪里。在某些任务上，GPT-2 完全抽象的输出与当前在许多问答和阅读理解数据集上最先进的抽取式指针网络 [77] 输出有显著不同。

鉴于先前微调 GPT 的成功，我们计划在 decaNLP 和 GLUE 等基准上研究微调，特别是因为尚不清楚 GPT-2 的额外训练数据和容量是否足以克服 BERT [19] 所展示的单向表示的低效性。

## 7. 结论

当大型语言模型在足够大和多样化的数据集上训练时，它能够在许多领域和数据集上表现良好。GPT-2 在 8 个测试的语言建模数据集中以零样本方式达到 7 个的最先进性能。该模型能够在零样本设置下执行的任务的多样性表明，训练用于最大化足够多样化文本语料库似然的高容量模型开始学习执行大量任务，而无需显式监督⁵。

## 致谢

感谢所有撰写文本、分享链接和点赞 WebText 内容的人。数以百万计的人参与了创建 GPT-2 所训练的数据。还要感谢所有帮助我们的 Google 同事，包括 Zak Stone、JS Riehl、Jonathan Hseu、Russell Power、Youlong Cheng、Noam Shazeer、Solomon Boulos、Michael Banfield、Aman Gupta、Daniel Sohn 和许多其他人。最后感谢对论文草稿提供反馈的人：Jacob Steinhardt、Sam Bowman、Geoffrey Irving 和 Madison May。

⁵ 下载和使用小模型的初步代码可在 https://github.com/openai/gpt-2 获取。

## 参考文献

[1] Krizhevsky, A., Sutskever, I., and Hinton, G. E. Imagenet classification with deep convolutional neural networks. In *Advances in neural information processing systems*, pp. 1097–1105, 2012.

[2] Sutskever, I., Vinyals, O., and Le, Q. V. Sequence to sequence learning with neural networks. In *Advances in neural information processing systems*, pp. 3104–3112, 2014.

[3] Amodei, D., Ananthanarayanan, S., Anubhai, R., Bai, J., Battenberg, E., Case, C., Casper, J., Catanzaro, B., Cheng, Q., Chen, G., et al. Deep speech 2: End-to-end speech recognition in english and mandarin. In *International Conference on Machine Learning*, pp. 173–182, 2016.

[4] Recht, B., Roelofs, R., Schmidt, L., and Shankar, V. Do cifar-10 classifiers generalize to cifar-10? arXiv preprint arXiv:1806.00451, 2018.

[5] Kirkpatrick, J., Pascanu, R., Rabinowitz, N., Veness, J., Desjardins, G., Rusu, A. A., Milan, K., Quan, J., Ramalho, T., Grabska-Barwinska, A., et al. Overcoming catastrophic forgetting in neural networks. *Proceedings of the national academy of sciences*, pp. 201611835, 2017.

[6] Lake, B. M., Ullman, T. D., Tenenbaum, J. B., and Gershman, S. J. Building machines that learn and think like people. *Behavioral and Brain Sciences*, 40, 2017.

[7] Jia, R. and Liang, P. Adversarial examples for evaluating reading comprehension systems. arXiv preprint arXiv:1707.07328, 2017.

[8] Alcorn, M. A., Li, Q., Gong, Z., Wang, C., Mai, L., Ku, W.-S., and Nguyen, A. Strike (with) a pose: Neural networks are easily fooled by strange poses of familiar objects. arXiv preprint arXiv:1811.11553, 2018.

[9] Wang, A., Singh, A., Michael, J., Hill, F., Levy, O., and Bowman, S. R. Glue: A multi-task benchmark and analysis platform for natural language understanding. arXiv preprint arXiv:1804.07461, 2018.

[10] McCann, B., Keskar, N. S., Xiong, C., and Socher, R. The natural language decathlon: Multitask learning as question answering. arXiv preprint arXiv:1806.08730, 2018.

[11] Caruana, R. Multitask learning. *Machine learning*, 28(1):41–75, 1997.

[12] Yogatama, D., d'Autume, C. d. M., Connor, J., Kocisky, T., Chrzanowski, M., Kong, L., Lazaridou, A., Ling, W., Yu, L., Dyer, C., et al. Learning and evaluating general linguistic intelligence. arXiv preprint arXiv:1901.11373, 2019.

[13] Bowman, S. R., Pavlick, E., Grave, E., Van Durme, B., Wang, A., Hula, J., Xia, P., Pappagari, R., McCoy, R. T., Patel, R., et al. Looking for elmo's friends: Sentence-level pretraining beyond language modeling. arXiv preprint arXiv:1812.10860, 2018.

[14] Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., and Dean, J. Distributed representations of words and phrases and their compositionality. In *Advances in neural information processing systems*, pp. 3111–3119, 2013.

[15] Collobert, R., Weston, J., Bottou, L., Karlen, M., Kavukcuoglu, K., and Kuksa, P. Natural language processing (almost) from scratch. *Journal of Machine Learning Research*, 12(Aug):2493–2537, 2011.

[16] Dai, A. M. and Le, Q. V. Semi-supervised sequence learning. In *Advances in neural information processing systems*, pp. 3079–3087, 2015.

[17] Peters, M. E., Neumann, M., Iyyer, M., Gardner, M., Clark, C., Lee, K., and Zettlemoyer, L. Deep contextualized word representations. arXiv preprint arXiv:1802.05365, 2018.

[18] Radford, A., Narasimhan, K., Salimans, T., and Sutskever, I. Improving language understanding by generative pre-training. 2018.

[19] Devlin, J., Chang, M.-W., Lee, K., and Toutanova, K. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018.

[20] Schwartz, R., Sap, M., Konstas, I., Zilles, L., Choi, Y., and Smith, N. A. Story cloze task: Uw nlp system. In *Proceedings of the 2nd Workshop on Linking Models of Lexical, Sentential and Discourse-level Semantics*, pp. 52–55, 2017.

[21] Radford, A., Jozefowicz, R., and Sutskever, I. Learning to generate reviews and discovering sentiment. arXiv preprint arXiv:1704.01444, 2017.

[22] Jelinek, F. and Mercer, R. L. Interpolated estimation of markov source parameters from sparse data. In *Proceedings of the Workshop on Pattern Recognition in Practice*, Amsterdam, The Netherlands: North-Holland, May., 1980.

[23] Bengio, Y., Ducharme, R., Vincent, P., and Jauvin, C. A neural probabilistic language model. *Journal of machine learning research*, 3(Feb):1137–1155, 2003.

[24] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., and Polosukhin, I. Attention is all you need. In *Advances in Neural Information Processing Systems*, pp. 5998–6008, 2017.

[25] Kaiser, L., Gomez, A. N., Shazeer, N., Vaswani, A., Parmar, N., Jones, L., and Uszkoreit, J. One model to learn them all. arXiv preprint arXiv:1706.05137, 2017.

[26] Finn, C., Abbeel, P., and Levine, S. Model-agnostic meta-learning for fast adaptation of deep networks. arXiv preprint arXiv:1703.03400, 2017.

[27] Sutskever, I., Jozefowicz, R., Gregor, K., Rezende, D., Lillicrap, T., and Vinyals, O. Towards principled unsupervised learning. arXiv preprint arXiv:1511.06440, 2015.

[28] Weston, J. E. Dialog-based language learning. In *Advances in Neural Information Processing Systems*, pp. 829–837, 2016.

[29] Jozefowicz, R., Vinyals, O., Schuster, M., Shazeer, N., and Wu, Y. Exploring the limits of language modeling. arXiv preprint arXiv:1602.02410, 2016.

[30] Merity, S., Xiong, C., Bradbury, J., and Socher, R. Pointer sentinel mixture models. arXiv preprint arXiv:1609.07843, 2016.

[31] Kiros, R., Zhu, Y., Salakhutdinov, R. R., Zemel, R., Urtasun, R., Torralba, A., and Fidler, S. Skip-thought vectors. In *Advances in neural information processing systems*, pp. 3294–3302, 2015.

[32] Trinh, T. H. and Le, Q. V. A simple method for commonsense reasoning. arXiv preprint arXiv:1806.02847, 2018.

[33] Peters, M. E. and Lecocq, D. Content extraction using diverse feature sets. In *Proceedings of the 22nd International Conference on World Wide Web*, pp. 89–90. ACM, 2013.

[34] Gillick, D., Brunk, C., Vinyals, O., and Subramanya, A. Multilingual language processing from bytes. arXiv preprint arXiv:1512.00103, 2015.

[35] Al-Rfou, R., Choe, D., Constant, N., Guo, M., and Jones, L. Character-level language modeling with deeper self-attention. arXiv preprint arXiv:1808.04444, 2018.

[36] Sennrich, R., Haddow, B., and Birch, A. Neural machine translation of rare words with subword units. arXiv preprint arXiv:1508.07909, 2015.

[37] Ba, J. L., Kiros, J. R., and Hinton, G. E. Layer normalization. arXiv preprint arXiv:1607.06450, 2016.

[38] He, K., Zhang, X., Ren, S., and Sun, J. Identity mappings in deep residual networks. In *European conference on computer vision*, pp. 630–645. Springer, 2016.

[39] Paperno, D., Kruszewski, G., Lazaridou, A., Pham, Q. N., Bernardi, R., Pezzelle, S., Baroni, M., Boleda, G., and Fernández, R. The lambada dataset: Word prediction requiring a broad discourse context. arXiv preprint arXiv:1606.06031, 2016.

[40] Hill, F., Bordes, A., Chopra, S., and Weston, J. The goldilocks principle: Reading children's books with explicit memory representations. arXiv preprint arXiv:1511.02301, 2015.

[41] Chelba, C., Mikolov, T., Schuster, M., Ge, Q., Brants, T., Koehn, P., and Robinson, T. One billion word benchmark for measuring progress in statistical language modeling. arXiv preprint arXiv:1312.3005, 2013.

[42] Gong, C., He, D., Tan, X., Qin, T., Wang, L., and Liu, T.-Y. Frage: frequency-agnostic word representation. In *Advances in Neural Information Processing Systems*, pp. 1341–1352, 2018.

[43] Bajgar, O., Kadlec, R., and Kleindienst, J. Embracing data abundance: Booktest dataset for reading comprehension. arXiv preprint arXiv:1610.00956, 2016.

[44] Hoang, L., Wiseman, S., and Rush, A. M. Entity tracking improves cloze-style reading comprehension. arXiv preprint arXiv:1810.02891, 2018.

[45] Grave, E., Joulin, A., and Usunier, N. Improving neural language models with a continuous cache. arXiv preprint arXiv:1612.04426, 2016.

[46] Dai, Z., Yang, Z., Yang, Y., Cohen, W. W., Carbonell, J., Le, Q. V., and Salakhutdinov, R. Transformer-xl: Attentive language models beyond a fixed-length context. arXiv preprint arXiv:1901.02860, 2019.

[47] Dehghani, M., Gouws, S., Vinyals, O., Uszkoreit, J., and Kaiser, Ł. Universal transformers. arXiv preprint arXiv:1807.03819, 2018.

[48] Levesque, H., Davis, E., and Morgenstern, L. The winograd schema challenge. In *Thirteenth International Conference on the Principles of Knowledge Representation and Reasoning*, 2012.

[49] Trichelair, P., Emami, A., Cheung, J. C. K., Trischler, A., Suleman, K., and Diaz, F. On the evaluation of common-sense reasoning in natural language understanding. arXiv preprint arXiv:1811.01778, 2018.

[50] Reddy, S., Chen, D., and Manning, C. D. Coqa: A conversational question answering challenge. arXiv preprint arXiv:1808.07042, 2018.

[51] Gehrmann, S., Deng, Y., and Rush, A. M. Bottom-up abstractive summarization. arXiv preprint arXiv:1808.10792, 2018.

[52] Nallapati, R., Zhou, B., Gulcehre, C., Xiang, B., et al. Abstractive text summarization using sequence-to-sequence rnns and beyond. arXiv preprint arXiv:1602.06023, 2016.

[53] Fan, A., Lewis, M., and Dauphin, Y. Hierarchical neural story generation. arXiv preprint arXiv:1805.04833, 2018.

[54] Conneau, A., Lample, G., Ranzato, M., Denoyer, L., and Jégou, H. Word translation without parallel data. arXiv preprint arXiv:1710.04087, 2017b.

[55] Artetxe, M., Labaka, G., Agirre, E., and Cho, K. Unsupervised neural machine translation. arXiv preprint arXiv:1710.11041, 2017.

[56] Lample, G., Conneau, A., Denoyer, L., and Ranzato, M. Unsupervised machine translation using monolingual corpora only. arXiv preprint arXiv:1711.00043, 2017.

[57] Artetxe, M., Labaka, G., and Agirre, E. An effective approach to unsupervised machine translation. arXiv preprint arXiv:1902.01313, 2019.

[58] Vinyals, O. and Le, Q. A neural conversational model. arXiv preprint arXiv:1506.05869, 2015.

[59] Kwiatkowski, T., Palomaki, J., Rhinehart, O., Collins, M., Parikh, A., Alberti, C., Epstein, D., Polosukhin, I., Kelcey, M., Devlin, J., et al. Natural questions: a benchmark for question answering research. 2019.

[60] Alberti, C., Lee, K., and Collins, M. A bert baseline for the natural questions. arXiv preprint arXiv:1901.08634, 2019.

[61] Barz, B. and Denzler, J. Do we train on test data? purging cifar of near-duplicates. arXiv preprint arXiv:1902.00423, 2019.

[62] Hestness, J., Narang, S., Ardalani, N., Diamos, G., Jun, H., Kianinejad, H., Patwary, M., Ali, M., Yang, Y., and Zhou, Y. Deep learning scaling is predictable, empirically. arXiv preprint arXiv:1712.00409, 2017.

[63] Karpathy, A., Johnson, J., and Fei-Fei, L. Visualizing and understanding recurrent networks. arXiv preprint arXiv:1506.02078, 2015.

[64] Liu, P. J., Saleh, M., Pot, E., Goodrich, B., Sepassi, R., Kaiser, L., and Shazeer, N. Generating wikipedia by summarizing long sequences. arXiv preprint arXiv:1801.10198, 2018.

[65] Davies, M. The 14 billion word iweb corpus. https://corpus.byu.edu/iWeb/, 2018.

[66] Pennington, J., Socher, R., and Manning, C. Glove: Global vectors for word representation. In *Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP)*, pp. 1532–1543, 2014.

[67] McCann, B., Bradbury, J., Xiong, C., and Socher, R. Learned in translation: Contextualized word vectors. In *Advances in Neural Information Processing Systems*, pp. 6294–6305, 2017.

[68] Howard, J. and Ruder, S. Universal language model fine-tuning for text classification. In *Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, volume 1, pp. 328–339, 2018.

[69] Conneau, A., Kiela, D., Schwenk, H., Barrault, L., and Bordes, A. Supervised learning of universal sentence representations from natural language inference data. arXiv preprint arXiv:1705.02364, 2017a.

[70] Subramanian, S., Trischler, A., Bengio, Y., and Pal, C. J. Learning general purpose distributed sentence representations via large scale multi-task learning. arXiv preprint arXiv:1804.00079, 2018.

[71] Ramachandran, P., Liu, P. J., and Le, Q. V. Unsupervised pre-training for sequence to sequence learning. arXiv preprint arXiv:1611.02683, 2016.

[72] Wolf, T., Sanh, V., Chaumond, J., and Delangue, C. Transfertransfo: A transfer learning approach for neural network based conversational agents. arXiv preprint arXiv:1901.08149, 2019.

[73] Dinan, E., Roller, S., Shuster, K., Fan, A., Auli, M., and Weston, J. Wizard of wikipedia: Knowledge-powered conversational agents. arXiv preprint arXiv:1811.01241, 2018.

[74] Hill, F., Cho, K., and Korhonen, A. Learning distributed representations of sentences from unlabelled data. arXiv preprint arXiv:1602.03483, 2016.

[75] Levy, O. and Goldberg, Y. Neural word embedding as implicit matrix factorization. In *Advances in neural information processing systems*, pp. 2177–2185, 2014.

[76] Wieting, J. and Kiela, D. No training required: Exploring random encoders for sentence classification. arXiv preprint arXiv:1901.10444, 2019.

[77] Vinyals, O., Fortunato, M., and Jaitly, N. Pointer networks. In *Advances in Neural Information Processing Systems*, pp. 2692–2700, 2015.

## 附录 A：样本

### A.1. 模型容量

为了补充图 4 中显示的更大 LM 在 WebText 上报告的困惑度提升，表 7 至 11 展示了小型 WebText LM 和 GPT-2 在随机未见过的 WebText 测试集文章上的并排补全结果。

### A.2. 文本记忆化

我们观察到 GPT-2 在数据集中重复多次的较长字符串（如著名引文或演讲）上存在一些记忆行为。例如，当以葛底斯堡演说（在 WebText 中出现了大约 40 次）的前一句话和半句话为条件时，GPT-2 的 argmax 解码恢复了该演讲。即使在使用无截断采样时，我们发现在漂移之前，模型会复制演讲一段时间，尽管风格相似。它通常在 100-200 个 token 内漂移，并且一旦漂移就显示出越来越大的多样性。

为了量化精确记忆在样本中出现的频率，我们生成了以 WebText 测试集文章为条件的 GPT-2 样本，并比较了 GPT-2 生成内容与真实补全内容的重叠率。该分析的结果如下所示，表明 GPT-2 复述训练集文本的频率低于留出文章的基础比率。

图 5：WebText 训练集 8-gram 重叠百分比的 CDF，针对 WebText 测试集和样本（以 WebText 测试集为条件，使用 $k = 40$ 的 top-k 截断随机采样）。大多数样本的重叠率小于 1%，超过 30% 的样本完全没有重叠，而测试集的中位数为 2.6%。

### A.3. 多样性

表 12 显示了相同随机 WebText 测试集上下文的多个补全结果，展示了在标准采样设置下的补全多样性。

### A.4. 鲁棒性

表 13 显示了前面提到的会说话的独角兽新闻文章。我们发现模型能够处理分布外上下文，但这些样本的质量通常较低。

（注：附录中的表格 7-17 包含大量生成样本，为保持译文简洁，此处不逐表翻译表格内的具体生成内容。如需完整翻译所有表格内容，请指示。）
