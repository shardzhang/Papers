# GLUE：自然语言理解的多任务基准与分析平台

> Alex Wang¹, Amanpreet Singh¹, Julian Michael², Felix Hill³, Omer Levy² & Samuel R. Bowman¹
> ¹纽约大学库朗数学科学研究所 | ²华盛顿大学保罗·G·艾伦计算机科学与工程学院 | ³DeepMind
> {alexwang,amanpreet,bowman}@nyu.edu | {julianjm,omerlevy}@cs.washington.edu | felixhill@google.com

GLUE基准是一个用于评估模型在一组多样化现有NLU任务上性能的工具集合。通过包含训练数据有限的任务，GLUE旨在偏好和鼓励跨任务共享通用语言知识的模型。GLUE还包含一个手工构建的诊断测试套件，能够对模型进行细粒度的语言分析。我们评估了基于当前迁移和表示学习方法的基线，发现对所有任务进行多任务训练优于为每个任务单独训练模型。然而，我们最佳模型的绝对性能仍然较低，表明需要改进通用的NLU系统。

---

## 摘要

为了使自然语言理解（NLU）技术发挥最大效用，它必须能够以不局限于单一任务、体裁或数据集的方式处理语言。为了实现这一目标，我们引入了通用语言理解评估（GLUE）基准，这是一个用于评估模型在一组多样化现有NLU任务上性能的工具集合。通过包含训练数据有限的任务，GLUE旨在偏好和鼓励跨任务共享通用语言知识的模型。GLUE还包含一个手工构建的诊断测试套件，能够对模型进行细粒度的语言分析。我们评估了基于当前迁移和表示学习方法的基线，发现对所有任务进行多任务训练优于为每个任务单独训练模型。然而，我们最佳模型的绝对性能仍然较低，表明需要改进通用的NLU系统。

## 1 引言

人类理解语言的能力是通用、灵活且鲁棒的。相比之下，大多数词汇级别以上的NLU模型是为特定任务设计的，并且在域外数据上表现不佳。如果我们希望开发能够超越检测输入与输出之间表面对应关系的理解能力的模型，那么开发一个更统一的模型是至关重要的，该模型能够学习在不同领域中执行一系列不同的语言任务。

为了促进这一方向的研究，我们提出了通用语言理解评估（GLUE）基准：一组NLU任务集合，包括问答、情感分析和文本蕴含，以及一个用于模型评估、比较和分析的在线平台。GLUE对模型架构没有施加任何限制，除了能够处理单句和句子对输入并做出相应预测。对于某些GLUE任务，训练数据是充足的，但对于其他任务，训练数据有限或无法匹配测试集的体裁。因此，GLUE偏好那些能够以促进样本高效学习和跨任务有效知识迁移的方式表示语言知识的模型。GLUE中的所有数据集都不是从头为基准创建的；我们依赖已有的数据集，因为它们已被NLP社区隐含地认为是具有挑战性和趣味性的。其中四个数据集采用私有测试数据，这将用于确保基准被公平使用。

为了理解模型学到的知识类型并鼓励有语言学意义的解决策略，GLUE还包含一组手工构建的分析示例，用于探针训练好的模型。该数据集旨在突出常见的挑战，例如使用世界知识和逻辑运算符，我们期望模型必须处理这些挑战才能鲁棒地解决任务。

为了更好地理解GLUE带来的挑战，我们使用简单的基线和最先进的句子表示模型进行了实验。我们发现统一的、经过多任务训练的模型略优于在每个任务上分别训练的同类模型。我们最佳的多任务模型使用了最近提出的预训练技术ELMo。然而，该模型仍然取得了相当低的绝对分数。使用我们的诊断数据集进行分析表明，我们的基线模型能够很好地处理强词汇信号，但在更深的逻辑结构上表现困难。

总之，我们提供：
(i) 一组九个句子或句子对NLU任务，建立在已有的标注数据集之上，并经过选择以覆盖多样化的文本体裁、数据集大小和难度级别。
(ii) 一个基于私有测试数据的在线评估平台和排行榜。该平台与模型无关，可以评估任何能够在全部九个任务上产生结果的方法。
(iii) 一个专家构建的诊断评估数据集。
(iv) 若干种主要的现有句子表示学习方法的基线结果。

## 2 相关工作

Collobert等人[2011]使用了一个具有共享句子理解组件的多任务模型来联合学习词性标注、组块分析、命名实体识别和语义角色标注。最近的工作探索了使用核心NLP任务的标签来监督深度神经网络较低层的训练[Søgaard & Goldberg, 2016; Hashimoto等人, 2017]以及自动学习多任务学习的跨任务共享机制[Ruder等人, 2017]。

除了多任务学习，通用NLU系统的开发还集中在句子到向量编码器[Le & Mikolov, 2014; Kiros等人, 2015, 等]，利用无标签数据[Hill等人, 2016; Peters等人, 2018]，有标签数据[Conneau & Kiela, 2018; McCann等人, 2017]，以及这些方法的组合[Collobert等人, 2011; Subramanian等人, 2018]。在这一系列工作中，一种标准的评估实践已经出现，最近被编入SentEval[Conneau等人, 2017; Conneau & Kiela, 2018]。与GLUE类似，SentEval依赖于一组涉及一个或两个句子作为输入的现有分类任务。与GLUE不同，SentEval只评估句子到向量编码器，使其非常适合在涉及孤立句子的任务上评估模型。然而，跨句上下文化和对齐在机器翻译[Bahdanau等人, 2015; Vaswani等人, 2017]、问答[Seo等人, 2017]和自然语言推理[Rocktäschel等人, 2016]等任务上取得最先进性能方面至关重要。GLUE旨在促进这些方法的发展：它与模型无关，允许任何类型的表示或上下文化，包括那些对句子完全不使用显式向量或符号表示的模型。

GLUE在评估任务的选择上也与SentEval有所不同。许多SentEval任务与情感分析密切相关，例如MR[Pang & Lee, 2005]、SST[Socher等人, 2013]、CR[Hu & Liu, 2004]和SUBJ[Pang & Lee, 2004]。其他任务则接近已被解决，对其评估相对无信息量，例如MPQA[Wiebe等人, 2005]和TREC问题分类[Voorhees等人, 1999]。在GLUE中，我们试图构建一个既多样化又具有挑战性的基准。

McCann等人[2018]引入了decaNLP，它也根据模型在多个数据集上的表现对NLP系统进行评分。该基准将十个评估任务重新定义为问答，使用自动转换将摘要和文本到SQL语义解析等任务转换为问答。该基准缺少GLUE的排行榜和错误分析工具包，但更重要的是，我们认为它追求的是一个更雄心勃勃但不够立即可行的目标：GLUE奖励那些在有限的任务集上使用类似这些任务当前使用的方法取得良好性能的方法，而他们的基准则奖励那些在将所有NLU统一在问答框架下这一目标上取得进展的系统。

## 3 任务

GLUE以九个英语句子理解任务为中心，涵盖了广泛的领域、数据量和难度。由于GLUE的目标是推动可泛化NLU系统的发展，我们设计的基准要求良好的性能需要模型在所有任务之间共享大量知识（例如，训练好的参数），同时仍然保持一些任务特定的组件。尽管可以为每个任务训练一个没有预训练或其他外部知识来源的单一模型，并在该基准上评估得到的一组模型，但我们预计包含多个数据稀缺任务最终将使这种方法失去竞争力。下面我们在表1中描述这些任务。附录A包含更多细节。除非另有说明，任务都使用准确率进行评估并且在各类别之间是平衡的。

| 语料库 | \|Train\| | \|Test\| | 任务 | 指标 | 领域 |
|--------|----------|---------|------|------|------|
| **单句任务** |
| CoLA | 8.5k | 1k | 可接受性 | Matthews相关 | 杂项 |
| SST-2 | 67k | 1.8k | 情感 | 准确率 | 影评 |
| **相似性和释义任务** |
| MRPC | 3.7k | 1.7k | 释义 | 准确率/F1 | 新闻 |
| STS-B | 7k | 1.4k | 句子相似性 | Pearson/Spearman相关 | 杂项 |
| QQP | 364k | 391k | 释义 | 准确率/F1 | 社交问答问题 |
| **推理任务** |
| MNLI | 393k | 20k | NLI | 匹配准确率/不匹配准确率 | 杂项 |
| QNLI | 105k | 5.4k | QA/NLI | 准确率 | 维基百科 |
| RTE | 2.5k | 3k | NLI | 准确率 | 新闻，维基百科 |
| WNLI | 634 | 146 | 共指/NLI | 准确率 | 小说 |

> 表1：任务描述和统计。除STS-B为回归任务外，所有任务均为单句或句子对分类。MNLI有三个类别；所有其他分类任务均有两个类别。粗体显示的测试集使用从未以任何形式公开的标签。

### 3.1 单句任务

**CoLA**

语言可接受性语料库（CoLA）[Warstadt等人, 2018]由从语言学理论的书籍和期刊文章中提取的英语可接受性判断组成。每个示例是一个单词序列，标注了它是否是一个符合语法的英语句子。遵循作者的做法，我们使用Matthews相关系数[Matthews, 1975]作为评估指标，该指标衡量不均衡二分类的性能，范围从-1到1，其中0表示无信息猜测的性能。我们使用标准的测试集，从作者处获得了私有标签。我们报告测试集域内和域外部分的组合单一性能数据。

**SST-2**

斯坦福情感树库（SST）[Socher等人, 2013]由电影评论中的句子及其情感的人工标注组成。任务是预测给定句子的情感。我们使用二类（正面/负面）分类划分，并且只使用句子级别的标签。

### 3.2 相似性和释义任务

**MRPC**

微软研究释义语料库（MRPC）[Dolan & Brockett, 2005]是一个从在线新闻源自动提取的句子对语料库，带有关于句子对中的句子是否语义等价的人工标注。由于类别不均衡（68%正面），我们遵循常见做法报告准确率和F1分数。

**QQP**

Quora问题对数据集²是一组来自社区问答网站Quora的问题对。任务是判断一对问题是否语义等价。与MRPC类似，QQP中的类别分布是不均衡的（63%负面），因此我们报告准确率和F1分数。我们使用标准测试集，从作者处获得了私有标签。我们观察到测试集的标签分布与训练集不同。

**STS-B**

语义文本相似性基准（STS-B）[Cer等人, 2017]是一个从新闻标题、视频和图像标题以及自然语言推理数据中提取的句子对集合。每对人工程度赋予1到5的相似性分数；任务是预测这些分数。遵循常见做法，我们使用Pearson和Spearman相关系数进行评估。

### 3.3 推理任务

**MNLI**

多体裁自然语言推理语料库（MNLI）[Williams等人, 2018]是一个带有文本蕴含标注的众包句子对集合。给定一个前提句子和一个假设句子，任务是预测前提是否蕴涵假设（蕴涵）、与假设矛盾（矛盾）或两者都不是（中立）。前提句子来自十个不同的来源，包括转录的语音、小说和政府报告。我们使用标准的测试集，从作者处获取了私有标签，并在匹配（域内）和不匹配（跨域）两部分上进行评估。我们还使用并推荐使用SNLI语料库[Bowman等人, 2015]中的55万示例作为辅助训练数据。

**QNLI**

斯坦福问答数据集（SQuAD）[Rajpurkar等人, 2016]是一个问答数据集，由问题-段落对组成，其中段落中有一个句子（摘自维基百科）包含对应问题（由标注者编写）的答案。我们通过在每个问题与对应上下文中的每个句子之间形成一对，并过滤掉问题和上下文句子之间的词汇重叠较低的句子对，将任务转换为句子对分类。任务是判断上下文句子是否包含问题的答案。这种修改后的版本消除了对模型选择确切答案的要求，但也消除了答案始终存在于输入中以及词汇重叠是可靠线索的简化假设。这种将现有数据集重构为NLI的过程类似于White等人[2017]介绍并由Demszky等人[2018]扩展的方法。我们将转换后的数据集称为QNLI（问答NLI）。³

**RTE**

识别文本蕴含（RTE）数据集来自一系列年度文本蕴含挑战赛。我们整合了RTE1[Dagan等人, 2006]、RTE2[Bar Haim等人, 2006]、RTE3[Giampiccolo等人, 2007]和RTE5[Bentivogli等人, 2009]的数据。⁴示例基于新闻和维基百科文本构建。我们将所有数据集转换为二类分类，对于三类数据集，为了保持一致性，我们将中立和矛盾合并为不蕴涵。

**WNLI**

Winograd模式挑战[Levesque等人, 2011]是一个阅读理解任务，系统必须阅读一个带有代词的句子，并从一组选项中选择该代词的指代对象。这些示例是手动构建的，以挫败简单的统计方法：每个示例都依赖于句子中单个单词或短语提供的上下文信息。为了将问题转化为句子对分类，我们通过用每个可能的指代对象替换歧义代词来构建句子对。任务是判断替换了代词的句子是否被原句蕴含。我们使用一个小的评估集，该集包含来自小说的新示例，由原始语料库的作者私下分享。虽然包含的训练集在两个类别之间是平衡的，但测试集在它们之间是不均衡的（65%不蕴涵）。此外，由于数据异常，开发集是对抗性的：假设有时在训练和开发示例之间共享，因此如果模型记住了训练示例，它们将在相应的开发集示例上预测错误的标签。与QNLI一样，每个示例是单独评估的，因此模型在此任务上的分数与其在未转换的原始任务上的分数之间不存在系统性对应关系。我们将转换后的数据集称为WNLI（Winograd NLI）。

### 3.4 评估

GLUE基准遵循与SemEval和Kaggle相同的评估模型。要在基准上评估一个系统，需要在该任务提供的测试数据上运行系统，然后将结果上传到网站gluebenchmark.com进行评分。基准网站显示每个任务的分数以及这些分数的宏平均值，以确定系统在排行榜上的位置。对于具有多个指标的任务（例如准确率和F1），我们使用指标的未加权平均值作为计算总体宏平均值时的任务分数。该网站还提供诊断数据集的细粒度和粗粒度结果。详见附录D。

## 4 诊断数据集

受FraCaS套件[Cooper等人, 1996]和最近的Build-It-Break-It竞赛[Ettinger等人, 2017]的启发，我们包含了一个小型、手动策划的测试集，用于分析系统性能。虽然主要基准主要反映应用驱动的示例分布，但我们的诊断数据集突显了一组我们认为是模型捕获的重要且有趣的现象。我们在表2中展示了全部现象。

> 表2：诊断数据集中标注的语言现象类型，分为四个主要类别。每个现象的描述见附录E。

| 粗粒度类别 | 细粒度类别 |
|-----------|-----------|
| 词汇语义 | 词汇蕴含、形态学否定、事实性、对称性/集合性、冗余、命名实体、量词 |
| 谓词-论元结构 | 核心论元、介词短语、省略/隐含、照应/共指、主动/被动、名词化、所有格/部分格、与格、关系从句、并列范围、交集性、限制性 |
| 逻辑 | 否定、双重否定、区间/数字、合取、析取、条件句、全称、存在、时序、向上单调、向下单调、非单调 |
| 知识 | 常识、世界知识 |

每个诊断示例是一个NLI句子对，带有标注的现象标签。NLI任务非常适合这种分析，因为它可以轻松评估（非接地）句子理解所涉及的全部技能，从解决句法歧义到使用世界知识的语用推理。我们通过为各种语言现象生成示例，并将我们的示例基于来自多个领域（新闻、Reddit、维基百科、学术论文）的自然发生的句子，确保数据足够多样化。这种方法不同于FraCaS，后者旨在用最小且统一的示例集测试语言学理论。

我们在表3中展示了一个数据样本。

> 表3：诊断集中的示例。Fwd（相应地Bwd）表示当句子1（相应地句子2）为前提时的标签。标签为蕴涵（E）、中立（N）或矛盾（C）。示例标注了它们所展示的现象，每个现象属于四个大类之一（括号内）。

| 标签 | 句子1 | 句子2 | Fwd | Bwd |
|------|-------|-------|-----|-----|
| 词汇蕴含（词汇语义），向下单调（逻辑） | 会议的时间尚未确定，据一位星巴克发言人称。 | 会议的时间尚未被考虑，据一位星巴克发言人称。 | N | E |
| 全称量词（逻辑） | 我们最深切的同情与所有受此事故影响的人同在。 | 我们最深切的同情与一位受此事故影响的受害者同在。 | E | N |
| 量词（词汇语义），双重否定（逻辑） | 我从未见过蜂鸟不飞。 | 我从未见过蜂鸟。 | N | E |

**标注过程**

我们从一个目标现象集开始，大致基于FraCaS套件[Cooper等人, 1996]中使用的现象。我们通过找到一个可以轻松地使其展示目标现象的句子，然后以两种方式进行编辑以产生合适的句子对来构建每个示例。我们做最小的修改以保持每个句子对内的高词汇和结构重叠，并限制表面线索。然后我们标注句子之间的推理关系，交替考虑每个句子作为前提，为每个对产生两个标注示例（总共1100个）。在可能的情况下，我们为单个源句子生成几个具有不同标签的对，以获得在词汇和结构上非常相似但对应不同蕴涵关系的极小句子对集合。最终的标签分布为42%蕴涵、35%中立和23%矛盾。

**评估**

由于诊断集内的类别分布不均衡，我们使用R³[Gorodkin, 2004]（Matthews相关系数的三类别推广）进行评估。

鉴于最近的研究表明众包数据通常包含可以被利用来在不解决预期任务的情况下取得好表现的人为伪影[Schwartz等人, 2017; Poliak等人, 2018; Tsuchiya, 2018, 等]，我们对此类伪影进行了审查。我们再现了Gururangan等人[2018]的方法，训练两个fastText分类器[Joulin等人, 2016]仅使用假设作为输入来预测SNLI和MNLI上的蕴涵标签。这些模型在我们的诊断数据上分别获得了接近随机的准确率32.7%和36.4%，表明该数据没有受到此类伪影的影响。

为了建立诊断集的人类基线性能，我们让六位NLP研究人员标注了从诊断集中随机抽样的50个句子对（100个蕴含示例）。标注者间一致性很高，Fleiss的\kappa值为0.73。标注者之间的平均R³分数为0.80，远高于第5节中描述的任何基线系统。

**预期用途**

诊断示例是精心挑选以针对特定现象的，而NLI是一个没有自然输入分布的任务，因此我们不期望诊断集的性能反映下游应用中的整体性能或泛化能力。分析集上的性能应在模型之间进行比较，而不是在类别之间进行比较。该集的提供不是作为一个基准，而是作为错误分析、定性模型比较和对抗性示例开发的分析工具。

## 5 基线

对于基线，我们评估了一个在GLUE任务上训练的多任务学习模型，以及基于最近预训练方法的几个变体。我们在此简要描述它们。详见附录B。我们在AllenNLP库[Gardner等人, 2017]中实现了我们的模型。基线的原始代码可从https://github.com/nyu-mll/GLUE-baselines 获得，新版本可从https://github.com/jsalt18-sentence-repl/jiant 获得。

**架构**

我们最简单的基线架构基于句子到向量编码器，暂不考虑GLUE评估具有更复杂结构的模型的能力。受Conneau等人[2017]的启发，该模型使用了一个两层、1500D（每方向）的BiLSTM，带有最大池化和300D的GloVe词嵌入（840B Common Crawl版本；Pennington等人, 2014）。对于单句任务，我们对句子进行编码并将结果向量传递给分类器。对于句子对任务，我们独立编码句子以生成向量 $u$ 和 $v$ ，然后将 $[u; v; |u - v|; u * v]$ 传递给分类器。分类器是一个具有512D隐藏层的MLP。

我们还考虑了一个模型变体，该变体在句子对任务中使用受Seo等人[2017]启发的注意力机制处理所有词对之间，然后是第二个带最大池化的BiLSTM。通过显式建模句子之间的交互，这些模型不属于句子到向量范式。

**预训练**

我们使用两种最近的预训练方法增强了我们的基础模型：ELMo和CoVe。我们两者都使用现有的训练好的模型。

ELMo使用了一对在Billion Word Benchmark [Chelba等人, 2013]上训练的双层神经语言模型。每个单词由一个上下文嵌入表示，该嵌入通过对两个模型每层对应隐藏状态的线性组合产生。我们遵循作者的建议⁶，在嵌入中优先使用ELMo嵌入而不是任何其他嵌入。

CoVe[McCann等人, 2017]使用了一个最初为英德翻译训练的双层BiLSTM编码器。一个词的CoVe向量是顶层的LSTM对应的隐藏状态。与原始工作一样，我们将CoVe向量与GloVe词嵌入连接起来。

**训练**

我们训练具有共享的BiLSTM句子编码器和注意力后BiLSTM的模型，这些编码器跨任务共享，而分类器则为每个任务单独训练。对于每次训练更新，我们根据每个任务的训练示例数量按比例采样一个任务来训练。我们训练模型使用Adam优化器[Kingma & Ba, 2015]，初始学习率为 $10^{-4}$ ，批大小为128。我们使用宏平均分数作为验证指标，并在学习率降至 $10^{-5}$ 以下或在5次验证检查后性能没有改善时停止训练。

我们还训练了一组单任务模型，它们配置和训练相同，但不共享任何参数。为了与多任务类似模型进行公平比较，我们没有为每个任务调整参数或训练设置，因此这些单任务模型通常不代表每个任务的最先进水平。

**句子表示模型**

最后，我们使用我们的基准评估了以下训练好的句子到向量编码器模型：使用GloVe嵌入的平均词袋（CBoW）、Skip-Thought[Kiros等人, 2015]、InferSent[Conneau等人, 2017]、DisSent[Nie等人, 2017]和GenSen[Subramanian等人, 2018]。对于这些模型，我们仅在其产生的表示之上训练任务特定的分类器。

⁶ github.com/allenai/allennlp/blob/master/tutorials/how to/elmo.md

> 表4：GLUE任务测试集上的基线性能。对于MNLI，我们报告匹配和不匹配测试集上的准确率。对于MRPC和QQP，我们报告准确率和F1。对于STS-B，我们报告Pearson和Spearman相关。对于CoLA，我们报告Matthews相关。对于所有其他任务，我们报告准确率。所有值均乘以100。在线平台上呈现了类似的表格。

| 模型 | Avg | CoLA | SST-2 | MRPC | QQP | STS-B | MNLI | QNLI | RTE | WNLI |
|------|-----|-------|-------|------|-----|-------|-------|------|-----|------|
| **单任务训练** |
| BiLSTM | 63.9 | 15.7 | 85.9 | 69.3/79.4 | 81.7/61.4 | 66.0/62.8 | 70.3/70.8 | 75.7 | 52.8 | 65.1 |
| +ELMo | 66.4 | 35.0 | 90.2 | 69.0/80.8 | 85.7/65.6 | 64.0/60.2 | 72.9/73.4 | 71.7 | 50.1 | 65.1 |
| +CoVe | 64.0 | 14.5 | 88.5 | 73.4/81.4 | 83.3/59.4 | 67.2/64.1 | 64.5/64.8 | 75.4 | 53.5 | 65.1 |
| +Attn | 63.9 | 15.7 | 85.9 | 68.5/80.3 | 83.5/62.9 | 59.3/55.8 | 74.2/73.8 | 77.2 | 51.9 | 65.1 |
| +Attn, ELMo | 66.5 | 35.0 | 90.2 | 68.8/80.2 | 86.5/66.1 | 55.5/52.5 | 76.9/76.7 | 76.7 | 50.4 | 65.1 |
| +Attn, CoVe | 63.2 | 14.5 | 88.5 | 68.6/79.7 | 84.1/60.1 | 57.2/53.6 | 71.6/71.5 | 74.5 | 52.7 | 65.1 |
| **多任务训练** |
| BiLSTM | 64.2 | 11.6 | 82.8 | 74.3/81.8 | 84.2/62.5 | 70.3/67.8 | 65.4/66.1 | 74.6 | 57.4 | 65.1 |
| +ELMo | 67.7 | 32.1 | 89.3 | 78.0/84.7 | 82.6/61.1 | 67.2/67.9 | 70.3/67.8 | 75.5 | 57.4 | 65.1 |
| +CoVe | 62.9 | 18.5 | 81.9 | 71.5/78.7 | 84.9/60.6 | 64.4/62.7 | 65.4/65.7 | 70.8 | 52.7 | 65.1 |
| +Attn | 65.6 | 18.6 | 83.0 | 76.2/83.9 | 82.4/60.1 | 72.8/70.5 | 67.6/68.3 | 74.3 | 58.4 | 65.1 |
| +Attn, ELMo | 70.0 | 33.6 | 90.4 | 78.0/84.4 | 84.3/63.1 | 74.2/72.3 | 74.1/74.5 | 79.8 | 58.9 | 65.1 |
| +Attn, CoVe | 63.1 | 8.3 | 80.7 | 71.8/80.0 | 83.4/60.5 | 69.8/68.4 | 68.1/68.6 | 72.9 | 56.0 | 65.1 |
| **预训练句子表示模型** |
| CBoW | 58.9 | 0.0 | 80.0 | 73.4/81.5 | 79.1/51.4 | 61.2/58.7 | 56.0/56.4 | 72.1 | 54.1 | 65.1 |
| Skip-Thought | 61.3 | 0.0 | 81.8 | 71.7/80.8 | 82.2/56.4 | 71.8/69.7 | 62.9/62.8 | 72.9 | 53.1 | 65.1 |
| InferSent | 63.9 | 4.5 | 85.1 | 74.1/81.2 | 81.7/59.1 | 75.9/75.3 | 66.1/65.7 | 72.7 | 58.0 | 65.1 |
| DisSent | 62.0 | 4.9 | 83.7 | 74.1/81.7 | 82.6/59.5 | 66.1/64.8 | 58.7/59.1 | 73.9 | 56.4 | 65.1 |
| GenSen | 66.2 | 7.7 | 83.1 | 76.6/83.0 | 82.9/59.8 | 79.3/79.2 | 71.4/71.3 | 78.6 | 59.2 | 65.1 |

## 6 基准结果

我们训练了每个模型的三次运行，并评估具有最佳宏平均开发集性能的运行（参见附录C中的表6）。对于单任务和句子表示模型，我们评估每个单独任务的最佳运行。我们在表4中展示了主基准任务的性能。

我们发现，在使用注意力或ELMo的模型中，多任务训练相比单任务训练产生了更好的总体分数。注意力在单任务训练中普遍具有可忽略或负面的聚合效应，但在多任务训练中有帮助。我们观察到使用ELMo嵌入替代GloVe或CoVe嵌入的一致性改进，特别是对于单句任务。与仅使用GloVe相比，使用CoVe效果有好有坏。

在预训练的句子表示模型中，我们观察到从CBoW到Skip-Thought再到InferSent和GenSen有相当一致的增益。相对于直接在GLUE任务上训练的模型，InferSent具有竞争力，而GenSen优于所有模型除了最好的两个。

观察每个任务的结果，我们发现句子表示模型在CoLA上的表现显著低于直接在该任务上训练的模型。另一方面，对于STS-B，直接在该任务上训练的模型明显落后于最佳句子表示模型的性能。最后，有些任务没有任何模型表现特别好。在WNLI上，没有模型超过最常见类别猜测（65.1%），我们将模型预测替换为最常见基线。在RTE上以及总体上，即使我们最好的基线也还有改进空间。这些早期结果表明解决GLUE超出了当前模型和方法的能力。

## 7 分析

我们通过在诊断集上评估每个模型的MNLI分类器来分析基线，以更好地了解它们的语言能力。结果呈现在表5中。

> 表5：诊断集上的结果。我们报告真实标签与预测标签之间的R³系数，已乘以100。粗粒度类别为词汇语义（LS）、谓词-论元结构（PAS）、逻辑（L）以及知识和常识（K）。我们示例性的细粒度类别为全称量化（UQuant）、形态学否定（MNeg）、双重否定（2Neg）、照应/共指（Coref）、限制性（Restr）和向下单调（Down）。

| 模型 | All | LS | PAS | L | K | UQuant | MNeg | 2Neg | Coref | Restr | Down |
|------|-----|-----|-----|----|----|--------|-------|-------|-------|-------|------|
| **单任务训练** |
| BiLSTM | 21 | 25 | 24 | 16 | 16 | 70 | 53 | 4 | 21 | -15 | 12 |
| +ELMo | 20 | 20 | 21 | 14 | 17 | 70 | 20 | 42 | 33 | -26 | -3 |
| +CoVe | 21 | 19 | 23 | 20 | 18 | 71 | 47 | -1 | 33 | -15 | 8 |
| +Attn | 25 | 24 | 30 | 20 | 14 | 50 | 47 | 21 | 38 | -8 | -3 |
| +Attn, ELMo | 28 | 30 | 35 | 23 | 14 | 85 | 20 | 42 | 33 | -26 | -3 |
| +Attn, CoVe | 24 | 29 | 29 | 18 | 12 | 77 | 50 | 1 | 18 | -1 | 12 |
| **多任务训练** |
| BiLSTM | 20 | 13 | 24 | 14 | 22 | 71 | 17 | -8 | 31 | -15 | 8 |
| +ELMo | 21 | 20 | 21 | 19 | 21 | 71 | 60 | 2 | 22 | 0 | 12 |
| +CoVe | 18 | 15 | 11 | 18 | 27 | 71 | 40 | 7 | 40 | 0 | 8 |
| +Attn | 18 | 13 | 24 | 11 | 16 | 71 | 1 | -12 | 31 | -15 | 8 |
| +Attn, ELMo | 22 | 18 | 26 | 13 | 19 | 70 | 27 | 5 | 31 | -26 | -3 |
| +Attn, CoVe | 18 | 16 | 25 | 16 | 13 | 71 | 26 | -8 | 33 | 9 | 8 |
| **预训练句子表示模型** |
| CBoW | 9 | 6 | 13 | 5 | 10 | 3 | 0 | 13 | 28 | -15 | -11 |
| Skip-Thought | 12 | 2 | 23 | 11 | 9 | 61 | 6 | -2 | 30 | -15 | 0 |
| InferSent | 18 | 20 | 20 | 15 | 14 | 77 | 50 | -20 | 15 | -15 | -9 |
| DisSent | 16 | 16 | 19 | 13 | 15 | 70 | 43 | -11 | 20 | -36 | -09 |
| GenSen | 20 | 28 | 26 | 14 | 12 | 78 | 57 | 2 | 21 | -15 | 12 |

**粗粒度类别**

所有模型的总体性能都很低：最高总分28仍然表示较差的绝对性能。在谓词-论元结构上的表现往往较高，在逻辑上较低，尽管这些数字在类别之间不具有直接可比性。与主要基准不同，多任务模型几乎总是被其单任务对应模型超越。这也许并不令人意外，因为在我们简单的多任务训练机制下，MNLI与其他任务之间可能存在一些破坏性干扰。在GLUE任务上训练的模型在很大程度上优于预训练的句子表示模型，除了GenSen。使用注意力对诊断分数的影响比使用ELMo或CoVe更大，我们认为这表明注意力对于NLI中的泛化尤为重要。

**细粒度子类别**

大多数模型能相对较好地处理全称量化。查看相关示例，依赖像"all"这样的词汇线索通常足以获得良好的表现。类似地，词汇线索通常在形态学否定示例中提供了良好的信号。我们观察到模型之间的不同弱点。双重否定对于仅使用GloVe嵌入的GLUE训练模型尤其困难。这一点通过ELMo以及一定程度上通过CoVe得到了改善。此外，注意力对总体结果有混合影响，带有注意力的模型往往在向下单调性上表现困难。检查它们的预测，我们发现模型对超义/下义替换和单词删除作为蕴含信号很敏感，但以错误的方向进行预测（仿佛替换/删除的单词处于向上单调的上下文中）。这与McCoy & Linzen [2019]最近的发现一致，即这些系统使用前提和假设之间的子序列关系作为启发式捷径。限制性示例，通常依赖于量词范围的细微差别，对几乎所有的模型都特别困难。

总体而言，有证据表明，超越句子到向量表示（例如使用注意力机制）可能有助于域外数据上的性能，并且像ELMo和CoVe这样的迁移方法编码了其监督信号特有的语言信息。然而，增加的表示容量可能导致过拟合，例如注意力模型在向下单调上下文中的失败。我们期望我们的平台和诊断数据集将来能用于类似的分析，以便模型设计者能够更好地理解其模型的泛化行为和隐含知识。

## 8 结论

我们介绍了GLUE，一个用于评估和分析自然语言理解系统的平台和资源集合。我们发现，总体上，在我们的任务上联合训练的模型比每个任务单独训练的模型的组合性能更好。我们确认了注意力机制和迁移学习方法（如ELMo）在NLU系统中的效用，这些方法结合后优于GLUE基准上最佳的句子表示模型，但仍然留有改进空间。当在我们的诊断数据集上评估这些模型时，我们发现它们在许多语言现象上失败了（通常是惨败），这暗示了未来工作的可能方向。总之，如何设计通用NLU模型的问题仍然没有答案，我们相信GLUE可以为解决这一挑战提供肥沃的土壤。

## 致谢

我们感谢Ellie Pavlick、Tal Linzen、Kyunghyun Cho和Nikita Nangia在项目早期阶段给出的意见，感谢Ernie Davis、Alex Warstadt以及Quora的Nikhil Dandekar和Kornel Csernai提供私有评估数据的访问权限。本项目受益于以下机构对SB的财务支持：Google、腾讯控股和三星研究，以及AdeptMind和NSF研究生研究奖学金对AW的支持。

## 参考文献

[1] Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural machine translation by jointly learning to align and translate. In *Proceedings of the International Conference on Learning Representations*, 2015.

[2] Roy Bar Haim, Ido Dagan, Bill Dolan, Lisa Ferro, Danilo Giampiccolo, Bernardo Magnini, and Idan Szpektor. The second PASCAL recognising textual entailment challenge. 2006.

[3] Luisa Bentivogli, Ido Dagan, Hoa Trang Dang, Danilo Giampiccolo, and Bernardo Magnini. The fifth PASCAL recognizing textual entailment challenge. 2009.

[4] Samuel R. Bowman, Gabor Angeli, Christopher Potts, and Christopher D. Manning. A large annotated corpus for learning natural language inference. In *Proceedings of the Conference on Empirical Methods in Natural Language Processing*, pp. 632–642. Association for Computational Linguistics, 2015.

[5] Daniel Cer, Mona Diab, Eneko Agirre, Inigo Lopez-Gazpio, and Lucia Specia. Semeval-2017 task 1: Semantic textual similarity-multilingual and cross-lingual focused evaluation. In *Eleventh International Workshop on Semantic Evaluations*, 2017.

[6] Ciprian Chelba, Tomas Mikolov, Mike Schuster, Qi Ge, Thorsten Brants, Phillipp Koehn, and Tony Robinson. One billion word benchmark for measuring progress in statistical language modeling. *arXiv preprint 1312.3005*, 2013.

[7] Ronan Collobert, Jason Weston, Léon Bottou, Michael Karlen, Koray Kavukcuoglu, and Pavel Kuksa. Natural language processing (almost) from scratch. *Journal of Machine Learning Research*, 12(Aug):2493–2537, 2011.

[8] Alexis Conneau and Douwe Kiela. SentEval: An evaluation toolkit for universal sentence representations. In *Proceedings of the Eleventh International Conference on Language Resources and Evaluation*, 2018.

[9] Alexis Conneau, Douwe Kiela, Holger Schwenk, Loïc Barrault, and Antoine Bordes. Supervised learning of universal sentence representations from natural language inference data. In *Proceedings of the Conference on Empirical Methods in Natural Language Processing*, Copenhagen, Denmark, September 9-11, 2017, pp. 681–691, 2017.

[10] Robin Cooper, Dick Crouch, Jan Van Eijck, Chris Fox, Josef Van Genabith, Jan Jaspars, Hans Kamp, David Milward, Manfred Pinkal, Massimo Poesio, Steve Pulman, Ted Briscoe, Holger Maier, and Karsten Konrad. Using the framework. Technical report, The FraCaS Consortium, 1996.

[11] Ido Dagan, Oren Glickman, and Bernardo Magnini. The PASCAL recognising textual entailment challenge. In *Machine learning challenges. evaluating predictive uncertainty, visual object classification, and recognising tectual entailment*, pp. 177–190. Springer, 2006.

[12] Dorottya Demszky, Kelvin Guu, and Percy Liang. Transforming question answering datasets into natural language inference datasets. *arXiv preprint 1809.02922*, 2018.

[13] William B Dolan and Chris Brockett. Automatically constructing a corpus of sentential paraphrases. In *Proceedings of the International Workshop on Paraphrasing*, 2005.

[14] Allyson Ettinger, Sudha Rao, Hal Daumé III, and Emily M Bender. Towards linguistically generalizable NLP systems: A workshop and shared task. In *First Workshop on Building Linguistically Generalizable NLP Systems*, 2017.

[15] Matt Gardner, Joel Grus, Mark Neumann, Oyvind Tafjord, Pradeep Dasigi, Nelson F. Liu, Matthew Peters, Michael Schmitz, and Luke S. Zettlemoyer. AllenNLP: A deep semantic natural language processing platform. 2017.

[16] Danilo Giampiccolo, Bernardo Magnini, Ido Dagan, and Bill Dolan. The third PASCAL recognizing textual entailment challenge. In *Proceedings of the ACL-PASCAL workshop on textual entailment and paraphrasing*, pp. 1–9. Association for Computational Linguistics, 2007.

[17] Jan Gorodkin. Comparing two k-category assignments by a k-category correlation coefficient. *Comput. Biol. Chem.*, 28(5-6):367–374, December 2004.

[18] Suchin Gururangan, Swabha Swayamdipta, Omer Levy, Roy Schwartz, Samuel R. Bowman, and Noah A. Smith. Annotation artifacts in natural language inference data. In *Proceedings of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, 2018.

[19] Kazuma Hashimoto, Caiming Xiong, Yoshimasa Tsuruoka, and Richard Socher. A joint many-task model: Growing a neural network for multiple nlp tasks. In *Proceedings of the Conference on Empirical Methods in Natural Language Processing*, 2017.

[20] Felix Hill, Kyunghyun Cho, and Anna Korhonen. Learning distributed representations of sentences from unlabelled data. In *Proceedings of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, 2016.

[21] Minqing Hu and Bing Liu. Mining and summarizing customer reviews. In *Proceedings of the tenth ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, pp. 168–177. ACM, 2004.

[22] Armand Joulin, Edouard Grave, Piotr Bojanowski, and Tomas Mikolov. Bag of tricks for efficient text classification. *arXiv preprint 1607.01759*, 2016.

[23] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In *Proceedings of the International Conference on Learning Representations*, 2015.

[24] Ryan Kiros, Yukun Zhu, Ruslan R Salakhutdinov, Richard Zemel, Raquel Urtasun, Antonio Torralba, and Sanja Fidler. Skip-Thought vectors. In *Advances in Neural Information Processing Systems*, pp. 3294–3302, 2015.

[25] Quoc Le and Tomas Mikolov. Distributed representations of sentences and documents. In Eric P. Xing and Tony Jebara (eds.), *Proceedings of the 31st International Conference on Machine Learning*, volume 32 of *Proceedings of Machine Learning Research*, pp. 1188–1196, Bejing, China, 22–24 Jun 2014. PMLR.

[26] Hector J Levesque, Ernest Davis, and Leora Morgenstern. The Winograd schema challenge. In *AAAI Spring Symposium: Logical Formalizations of Commonsense Reasoning*, volume 46, pp. 47, 2011.

[27] Brian W Matthews. Comparison of the predicted and observed secondary structure of t4 phage lysozyme. *Biochimica et Biophysica Acta (BBA)-Protein Structure*, 405(2):442–451, 1975.

[28] Bryan McCann, James Bradbury, Caiming Xiong, and Richard Socher. Learned in translation: Contextualized word vectors. In *Advances in Neural Information Processing Systems*, pp. 6297–6308, 2017.

[29] Bryan McCann, Nitish Shirish Keskar, Caiming Xiong, and Richard Socher. The natural language decathlon: Multitask learning as question answering. *arXiv preprint 1806.08730*, 2018.

[30] R. Thomas McCoy and Tal Linzen. Non-entailed subsequences as a challenge for natural language inference. In *Proceedings of the Society for Computation in Linguistics*, volume 2, pp. 357–360, 2019.

[31] Allen Nie, Erin D Bennett, and Noah D Goodman. Dissent: Sentence representation learning from explicit discourse relations. *arXiv preprint 1710.04334*, 2017.

[32] Bo Pang and Lillian Lee. A sentimental education: Sentiment analysis using subjectivity summarization based on minimum cuts. In *Proceedings of the 42nd Annual Meeting on Association for Computational Linguistics*, pp. 271. Association for Computational Linguistics, 2004.

[33] Bo Pang and Lillian Lee. Seeing stars: Exploiting class relationships for sentiment categorization with respect to rating scales. In *Proceedings of the 43rd Annual Meeting on Association for Computational Linguistics*, pp. 115–124. Association for Computational Linguistics, 2005.

[34] Jeffrey Pennington, Richard Socher, and Christopher Manning. GloVe: Global vectors for word representation. In *Proceedings of the Conference on Empirical Methods in Natural Language processing*, pp. 1532–1543, 2014.

[35] Matthew E Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, and Luke Zettlemoyer. Deep contextualized word representations. In *Proceedings of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, 2018.

[36] Adam Poliak, Jason Naradowsky, Aparajita Haldar, Rachel Rudinger, and Benjamin Van Durme. Hypothesis only baselines in natural language inference. In *SEM@NAACL-HLT*, 2018.

[37] Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. SQuAD: 100,000+ questions for machine comprehension of text. In *Proceedings of the Conference on Empirical Methods in Natural Language Processing*, pp. 2383–2392. Association for Computational Linguistics, 2016.

[38] Tim Rocktäschel, Edward Grefenstette, Moritz Hermann, Karl, Tomáš Kočiský, and Phil Blunsom. Reasoning about entailment with neural attention. In *Proceedings of the International Conference on Learning Representations*, 2016.

[39] Sebastian Ruder, Joachim Bingel, Isabelle Augenstein, and Anders Søgaard. Sluice networks: Learning what to share between loosely related tasks. *arXiv preprint 1705.08142*, 2017.

[40] Roy Schwartz, Maarten Sap, Ioannis Konstas, Li Zilles, Yejin Choi, and Noah A. Smith. The effect of different writing tasks on linguistic style: A case study of the ROC story cloze task. In *Proceedings of CoNLL*, 2017.

[41] Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, and Hannaneh Hajishirzi. Bidirectional attention flow for machine comprehension. In *Proceedings of the International Conference of Learning Representations*, 2017.

[42] Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D Manning, Andrew Ng, and Christopher Potts. Recursive deep models for semantic compositionality over a sentiment treebank. In *Proceedings of the Conference on Empirical Methods in Natural Language Processing*, pp. 1631–1642, 2013.

[43] Anders Søgaard and Yoav Goldberg. Deep multi-task learning with low level tasks supervised at lower layers. In *Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)*, volume 2, pp. 231–235, 2016.

[44] Sandeep Subramanian, Adam Trischler, Yoshua Bengio, and Christopher J. Pal. Learning general purpose distributed sentence representations via large scale multi-task learning. In *Proceedings of the International Conference on Learning Representations*, 2018.

[45] Masatoshi Tsuchiya. Performance Impact Caused by Hidden Bias of Training Data for Recognizing Textual Entailment. In *Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018)*, Miyazaki, Japan, May 7-12, 2018 2018. European Language Resources Association (ELRA).

[46] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In *Advances in Neural Information Processing Systems*, pp. 6000–6010, 2017.

[47] Ellen M Voorhees et al. The TREC-8 question answering track report. In *TREC*, volume 99, pp. 77–82, 1999.

[48] Alex Warstadt, Amanpreet Singh, and Samuel R Bowman. Neural network acceptability judgments. *arXiv preprint 1805.12471*, 2018.

[49] Aaron Steven White, Pushpendre Rastogi, Kevin Duh, and Benjamin Van Durme. Inference is everything: Recasting semantic resources into a unified evaluation framework. In *Proceedings of the Eighth International Joint Conference on Natural Language Processing (Volume 1: Long Papers)*, volume 1, pp. 996–1005, 2017.

[50] Janyce Wiebe, Theresa Wilson, and Claire Cardie. Annotating expressions of opinions and emotions in language. In *Proceedings of the International Conference on Language Resources and Evaluation*, volume 39, pp. 165–210. Springer, 2005.

[51] Adina Williams, Nikita Nangia, and Samuel R. Bowman. A broad-coverage challenge corpus for sentence understanding through inference. In *Proceedings of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, 2018.

[52] Yukun Zhu, Ryan Kiros, Rich Zemel, Ruslan Salakhutdinov, Raquel Urtasun, Antonio Torralba, and Sanja Fidler. Aligning books and movies: Towards story-like visual explanations by watching movies and reading books. In *Proceedings of the International Conference on Computer Vision*, pp. 19–27, 2015.

## 附录A 额外基准细节

**QNLI**

为了构建一个平衡的数据集，我们选择了所有与问题最相似的句子不是答案句的句子对，以及同等数量的正确句子与问题最相似但另一个干扰句子紧随其后的情况。我们的相似度度量基于带有预训练GloVe嵌入的CBoW表示。这种将已有数据集转换为NLI格式的方法与White等人[2017]的近期工作以及Dagan等人[2006]最初提出的文本蕴含动机密切相关。两者都认为许多NLP任务可以有效地归结为文本蕴含。

## 附录B 额外基线细节

### B.1 注意力机制

我们如下实现注意力机制：给定两个隐藏状态序列 $u_1, u_2, \ldots, u_M$ 和 $v_1, v_2, \ldots, v_N$ ，我们首先计算矩阵 $H$ ，其中 $H_{ij} = u_i \cdot v_j$ 。对于每个 $u_i$ ，我们通过对 $H$ 的第 $i$ 行取softmax得到注意力权重 $\alpha_i$ ，并通过将 $\alpha_i$ 作为权重的注意力加权求和得到对应的上下文向量 $\tilde{v}_i = \sum_j \alpha_{ij} v_j$ 。我们在序列 $[u_1; \tilde{v}_1], \ldots, [u_M; \tilde{v}_M]$ 上传递一个带最大池化的第二BiLSTM以产生 $u'$ 。我们类似地处理 $v_j$ 向量以得到 $v'$ 。最后，我们将 $[u'; v'; |u' - v'|; u' * v']$ 送入分类器。

### B.2 训练

我们训练具有共享的BiLSTM句子编码器和注意力后BiLSTM的模型，这些编码器跨任务共享，而分类器则为每个任务单独训练。对于每次训练更新，我们根据每个任务的训练示例数量按比例采样一个任务来训练。我们按每个任务的示例数反比例缩放每个任务的损失，我们发现这可以提高整体性能。我们使用Adam优化器[Kingma & Ba, 2015]训练模型，初始学习率为 $10^{-3}$ ，批大小为128，并使用梯度裁剪。我们使用所有任务的宏平均分数作为验证指标，每10k次更新进行一次验证检查。每当验证性能没有改善时，我们将学习率除以5。当学习率降至 $10^{-5}$ 以下或经过5次验证检查性能没有改善时，我们停止训练。

### B.3 句子表示模型

我们评估了以下句子表示模型：

1. **CBoW**：句子中token的GloVe嵌入的平均值。
2. **Skip-Thought**[Kiros等人, 2015]：一个序列到序列模型，训练用于在给定中间句子的情况下生成前一个和下一个句子。我们使用在Toronto Book Corpus [Zhu等人, 2015, TBC]的句子序列上训练的原始预训练模型。⁷
3. **InferSent**[Conneau等人, 2017]：一个在MNLI和SNLI上训练的带最大池化的BiLSTM。
4. **DisSent**[Nie等人, 2017]：一个带最大池化的BiLSTM，训练用于在源自TBC的数据上预测连接两个句子的话语标记（因为、所以等）。我们使用为八路分类训练的变体。
5. **GenSen**[Subramanian等人, 2018]：一个在各种监督和非监督目标上训练的序列到序列模型。我们使用同时在MNLI和SNLI上训练、在TBC上训练Skip-Thought目标、以及在Billion Word Benchmark上训练成分分析目标的模型变体。

我们在冻结的句子编码器之上训练任务特定的分类器，使用SentEval的默认参数。详情和代码参见 https://github.com/nyu-mll/SentEval。

⁷ github.com/ryankiros/skip-thoughts

## 附录C 开发集结果

GLUE网站限制用户每天提交两次，以避免对私有测试数据过拟合。为了为将来GLUE上的工作提供参考，我们在表6中呈现了基线所取得的最佳开发集结果。

> 表6：GLUE任务开发集上的基线性能。对于MNLI，我们报告匹配和不匹配测试集上平均的准确率。对于MRPC和QQP，我们报告准确率和F1。对于STS-B，我们报告Pearson和Spearman相关。对于CoLA，我们报告Matthews相关。对于所有其他任务，我们报告准确率。所有值均乘以100。

| 模型 | Avg | CoLA | SST-2 | MRPC | QQP | STS-B | MNLI | QNLI | RTE | WNLI |
|------|-----|-------|-------|------|-----|-------|-------|------|-----|------|
| **单任务训练** |
| BiLSTM | 66.7 | 17.6 | 87.5 | 77.9/85.1 | 85.3/82.0 | 71.6/72.0 | 66.7 | 77.0 | 58.5 | 56.3 |
| +ELMo | 68.7 | 44.1 | 91.5 | 70.8/82.3 | 88.0/84.3 | 70.3/70.5 | 68.6 | 71.2 | 53.4 | 56.3 |
| +CoVe | 66.8 | 25.1 | 89.2 | 76.5/83.4 | 86.2/81.8 | 70.7/70.8 | 62.4 | 74.4 | 59.6 | 54.9 |
| +Attn | 66.9 | 17.6 | 87.5 | 72.8/82.9 | 87.7/83.9 | 66.6/66.7 | 70.0 | 77.2 | 58.5 | 60.6 |
| +Attn, ELMo | 67.9 | 44.1 | 91.5 | 71.1/82.1 | 87.8/83.6 | 57.9/56.1 | 72.4 | 75.2 | 52.7 | 56.3 |
| +Attn, CoVe | 65.6 | 25.1 | 89.2 | 72.8/82.4 | 86.1/81.3 | 59.4/58.0 | 67.9 | 72.5 | 58.1 | 57.7 |
| **多任务训练** |
| BiLSTM | 60.0 | 18.6 | 82.3 | 75.0/82.7 | 84.4/79.3 | 69.0/66.9 | 65.6 | 74.9 | 59.9 | 9.9 |
| +ELMo | 63.1 | 26.4 | 90.9 | 80.2/86.7 | 84.2/79.7 | 72.9/71.5 | 67.4 | 76.0 | 55.6 | 14.1 |
| +CoVe | 59.3 | 9.8 | 82.0 | 73.8/81.0 | 83.4/76.6 | 64.5/61.9 | 65.5 | 70.4 | 52.7 | 32.4 |
| +Attn | 60.5 | 15.2 | 83.1 | 77.5/85.1 | 82.6/77.2 | 72.4/70.5 | 68.0 | 73.7 | 61.7 | 9.9 |
| +Attn, ELMo | 67.3 | 36.7 | 91.1 | 80.6/86.6 | 84.6/79.6 | 74.4/72.9 | 74.6 | 80.4 | 61.4 | 22.5 |
| +Attn, CoVe | 61.4 | 17.4 | 82.1 | 71.3/80.1 | 83.4/77.7 | 68.6/66.7 | 68.2 | 73.2 | 58.5 | 29.6 |
| **预训练句子表示模型** |
| CBoW | 61.4 | 4.6 | 79.5 | 75.0/83.7 | 75.0/65.5 | 70.6/71.1 | 57.1 | 62.5 | 71.9 | 56.3 |
| Skip-Thought | 61.8 | 0.0 | 82.0 | 76.2/84.3 | 78.9/70.7 | 74.8/74.8 | 63.4 | 58.5 | 73.4 | 49.3 |
| InferSent | 65.7 | 8.6 | 83.9 | 76.5/84.1 | 81.7/75.9 | 80.2/80.4 | 67.8 | 63.5 | 71.5 | 56.3 |
| DisSent | 63.8 | 11.7 | 82.5 | 77.0/84.4 | 81.8/75.6 | 68.9/69.0 | 61.2 | 59.9 | 73.9 | 56.3 |
| GenSen | 67.8 | 10.3 | 87.2 | 80.4/86.2 | 82.6/76.6 | 81.3/81.8 | 71.4 | 62.5 | 78.4 | 56.3 |

## 附录D 基准网站详情

GLUE的在线平台使用React、Redux和TypeScript构建。我们使用Google Firebase进行数据存储，并在提交时使用Google Cloud Functions托管和运行评分脚本。图1显示了排行榜上我们的基线的可视化展示。

> 图1：基准网站排行榜。展开视图显示每个提交的额外详细信息，包括简要的文字描述和参数数量。

> 表7：按粗粒度类别划分的诊断数据集统计信息。注意，某些示例可能被标注了属于多个类别的现象。

| 类别 | 计数 | % 中立 | % 矛盾 | % 蕴涵 |
|------|------|--------|--------|--------|
| 词汇语义 | 368 | 31.0 | 27.2 | 41.8 |
| 谓词-论元结构 | 424 | 37.0 | 13.7 | 49.3 |
| 逻辑 | 364 | 37.6 | 26.9 | 35.4 |
| 知识 | 284 | 26.4 | 31.7 | 41.9 |

## 附录E 额外诊断数据详情

该数据集旨在允许分析自然语言理解的多个层次，从词义和句子结构到高层推理和世界知识的应用。为了使这种分析可行，我们首先确定了四大类现象：词汇语义、谓词-论元结构、逻辑和知识。然而，由于这些类别较为模糊，我们将每个类别细分为一组更大的细粒度子类别。所有细粒度类别的描述在本节剩余部分给出。这些类别只是可以用来理解语言现象和蕴含的一个视角，当然关于示例应如何分类、类别应该是什么等问题还有讨论空间。这些类别并非基于任何特定的语言学理论，而是大致基于语言学家在句法和语义研究中经常识别和建模的问题。

该数据集的提供不是作为一个基准，而是作为一个分析工具，用以粗略描绘模型可能捕获或未捕获的现象类型，并提供一组示例，可用于错误分析、定性模型比较以及开发暴露模型弱点的对抗性示例。由于语言的分布有些任意，比较同一模型在不同类别上的表现将无助于分析。相反，我们建议比较不同模型在同一类别上的表现，或使用报告分数作为错误分析的指南。

### E.1 词汇语义

这些现象围绕词义的各个方面。

**词汇蕴含**

蕴含不仅可以在句子级别应用，也可以在词汇级别应用。例如，我们说"dog"词汇上蕴含"animal"，因为任何是狗的东西也是动物；而"dog"词汇上与"cat"矛盾，因为不可能同时是两者。这种关系适用于许多类型的词（名词、形容词、动词、许多介词等），并且词汇蕴含与句子蕴含之间的关系已被深入探索，例如在自然逻辑系统中。这种联系通常取决于语言中的单调性，因此许多词汇蕴含示例也将被标注为单调性类别之一，尽管我们并非在所有情况下都这样做（参见逻辑下的单调性）。

**形态学否定**

这是词汇矛盾的一种特殊情况，其中一个词从另一个词派生而来：从"affordable"到"unaffordable"，从"agree"到"disagree"等。我们还包括如"ever"和"never"等例子。我们也将这些例子标记为否定或双重否定，因为它们可以被视为涉及词汇层面的逻辑否定。

**事实性**

句子中出现的命题可能与整个句子处于任何蕴含关系中，具体取决于它们出现的上下文。在许多情况下，这由句子中的词汇触发词（通常是动词或副词）决定。例如：

- "I recognize that X" 蕴含 "X"。
- "I did not recognize that X" 蕴含 "X"。
- "I believe that X" 不蕴含 "X"。
- "I am refusing to do X" 与 "I am doing X" 矛盾。
- "I am not refusing to do X" 不与 "I am doing X" 矛盾。
- "I almost finished X" 与 "I finished X" 矛盾。
- "I barely finished X" 蕴含 "I finished X"。

像"I recognize that X"这样的结构通常被称为事实性的，因为蕴含（上面的X，被视为预设）即使在否定下也持续存在。像上面的"I am refusing to do X"这样的结构通常被称为蕴含性的，并且对否定敏感。还有一些情况，句子（不）蕴含其中提到的实体的存在，例如"I have found a unicorn"蕴含"A unicorn exists"，而"I am looking for a unicorn"不一定蕴含"A unicorn exists"。实体不一定存在的解读通常被称为内涵解读，因为它们似乎处理由描述所表示的属性（其内涵），而不是可以归结为匹配描述的实体集合（其外延，在不存在的情况下将是空的）。

我们将涉及这些现象的所有示例都归入事实性标签下。虽然判断嵌套命题或实体的存在是否被整体陈述所蕴含通常取决于上下文，但在很多情况下它严重依赖于词汇触发词，因此我们将此类别置于词汇语义下。

**对称性/集合性**

有些命题表示对称关系，而另一些则不是。例如，"John married Gary"蕴含"Gary married John"，但"John likes Gary"不蕴含"Gary likes John"。对称关系通常可以通过将两个论元收集到主语中进行改写："John met Gary"蕴含"John and Gary met"。一个关系是否是对称的，或允许将其论元收集到主语中，通常由其中心词（例如"like"、"marry"或"meet"）决定，因此我们将其归类于词汇语义。

**冗余**

如果一个词可以从句子中移除而不改变其含义，这意味着该词的含义或多或少已经由句子充分表达了；因此，识别这些情况反映了对词汇语义和句子语义的理解。

**命名实体**

词通常命名存在于世界中的实体。我们可能希望理解关于这些名称的许多不同类型的信息，包括它们的组合结构（例如，"Baltimore Police"与"Police of the City of Baltimore"是一样的）或它们的现实世界指代和首字母缩略词扩展（例如，"SNL"是"Saturday Night Live"）。这个类别与世界知识密切相关，但侧重于名称作为词汇项的语义，而不是关于其所指实体的背景知识。

**量词**

自然语言中的逻辑量化通常通过词汇触发词表达，如"every"、"most"、"some"和"no"。虽然我们将量化和单调性中的类别保留用于涉及这些量词及其论元的操作的蕴含，但我们选择将量词的可互换性（例如，在许多情况下"most"蕴含"many"）视为词汇语义的问题。

### E.2 谓词-论元结构

理解句子含义的一个重要组成部分是理解其各部分如何组合成一个整体。在这个类别中，我们处理从句法歧义到语义角色和共指的全方位问题。

**句法歧义：关系从句、并列范围**

这两个类别纯粹涉及解决句法歧义。关系从句和并列范围都是英语中大量歧义的来源。

**介词短语**

介词短语附着是NLP系统中的句法分析器继续努力应对的一个特别困难的问题。我们将其视为句法和语义的问题，因为介词短语可以表达各种各样的语义角色，并且通常在语义上超越其直接的句法附着。

**核心论元**

动词选择特定的论元，特别是主语和宾语，这些可能根据上下文或表面形式而互换。一个例子是作格交替："Jake broke the vase"蕴含"the vase broke"，但"Jake broke the vase"不蕴含"Jake broke"。核心论元的其他重排，例如在对称性/集合性中看到的，也属于核心论元标签。

**交替：主动/被动、所有格/部分格、名词化、与格**

这四个类别对应于已知在英语中遵循特定模式的句法交替：

- 主动/被动："I saw him"等价于"He was seen by me"并蕴含"He was seen"。
- 所有格/部分格："the elephant's foot"与"the foot of the elephant"相同。
- 名词化："I caused him to submit his resignation"蕴含"I caused the submission of his resignation"。
- 与格："I baked him a cake"蕴含"I baked a cake for him"和"I baked a cake"，但不蕴含"I baked him"。

**省略/隐含**

动词或其他谓词的论元常常在文本中被省略（缺省），由读者填补空缺。我们可以通过用正确或不正确的指代显式填补空缺来构建蕴含示例。例如，前提"Putin is so entrenched within Russia's ruling system that many of its members can imagine no other leader"蕴含"Putin is so entrenched within Russia's ruling system that many of its members can imagine no other leader than Putin"，并与"Putin is so entrenched within Russia's ruling system that many of its members can imagine no other leader than themselves"矛盾。

这通常被视为照应的一个特例，但我们决定将这些情况与显式照应分开，后者通常也被视为共指的一种情况（并在现代共指消解系统中得到了一定程度的处理）。

**照应/共指**

共指指的是多个表达式指代同一实体或事件的情况。它与照应密切相关，后者是一个表达式的含义依赖于上下文中另一个（先行词）表达式的情况。这两种现象有显著重叠；例如，代词（"she"、"we"、"it"）是与先行词共指的照应语。然而，它们也可能独立发生，例如两个定指名词短语之间的共指（如"Theresa May"和"the British Prime Minister"指代同一实体），或者像"other"这样的词的照应用法，它需要一个先行词来区分某事物。在本类别中，我们只包含存在与先行词或其他短语共指的显式短语（照应与否）的情况。我们以与省略/隐含大致相同的方式构建这些示例。

**交集性**

许多修饰语，尤其是形容词，允许非交集性用法，这会影响它们的蕴含行为。例如：

- 交集性："He is a violinist and an old surgeon"蕴含"He is an old violinist"和"He is a surgeon"。
- 非交集性："He is a violinist and a skilled surgeon"不蕴含"He is a skilled violinist"。
- 非交集性："He is a fake surgeon"不蕴含"He is a surgeon"。

一般来说，修饰语的交集性用法，如"old men"中的"old"，可以被解释为指代具有两种属性的实体集合（他们年老且他们是男人）。语言学家通常使用集合交来形式化这一点，因此得名。

交集性与事实性有关。例如，"fake"可以被视为反蕴含性修饰语，这些示例将相应地标注。然而，我们选择将交集性归类于谓词-论元结构而不是词汇语义，因为通常同一个词允许交集性和非交集性两种用法，因此它可以被视为论元结构的歧义。

**限制性**

限制性通常用于指代名词修饰语用法的属性。具体来说，修饰语的限制性用法用于识别所描述的实体，而非限制性用法则为已识别的实体添加额外细节。这种区别通常可以通过蕴含来突出：

- 限制性："I finished all of my homework due today"不蕴含"I finished all of my homework"。
- 非限制性："I got rid of all those pesky bedbugs"蕴含"I got rid of all those bedbugs"。

通常用作非限制性的修饰语包括同位语、以"which"或"who"开头的关系从句以及感叹词（例如"pesky"）。非限制性用法可以有多种形式。

### E.3 逻辑

理解了句子的结构后，通常可以使用逻辑算子推导出一组基线浅层结论，并通常可以使用逻辑的数学工具进行建模。事实上，数学逻辑的发展最初是由关于自然语言含义的问题引导的，从亚里士多德的三段论到弗雷格的符号。蕴含的概念也是从数学逻辑借用而来的。

**命题结构：否定、双重否定、合取、析取、条件句**

命题逻辑的所有基本运算都出现在自然语言中，我们在它们与我们的示例相关之处进行标注：

- 否定："The cat sat on the mat"与"The cat did not sit on the mat"矛盾。
- 双重否定："The market is not impossible to navigate"蕴含"The market is possible to navigate"。
- 合取："Temperature and snow consistency must be just right"蕴含"Temperature must be just right"。
- 析取："Life is either a daring adventure or nothing at all"不蕴含"Life is a daring adventure"但被其蕴含。
- 条件句："If both apply, they are essentially impossible"不蕴含"They are essentially impossible"。

条件句更为复杂，因为它们在语言中的使用并不总是反映它们在逻辑中的含义。例如，它们可能用于高于话题断言的层面："If you think about it, it's the perfect reverse psychology tactic"蕴含"It's the perfect reverse psychology tactic"。

**量化：全称、存在**

量词通常由诸如"all"、"some"、"many"和"no"等词触发。有大量工作用广义量词在数学逻辑中建模它们的含义。在这两个类别中，我们关注自然语言中全称和存在量词对应的直接推理：

- 全称："All parakeets have two wings"蕴含"My parakeet has two wings"但不被其蕴含。
- 存在："Some parakeets have two wings"不蕴含"My parakeet has two wings"但被其蕴含。

**单调性：向上单调、向下单调、非单调**

单调性是某些逻辑系统中论元位置的一种属性。一般来说，它提供了一种推导仅在单个子表达式上有所不同的表达式之间的蕴含关系的方法。在语言中，它可以解释一些蕴含如何通过逻辑算子和量词传播。

例如，"pet"蕴含"pet squirrel"，后者进一步蕴含"happy pet squirrel"。我们可以展示量词"a"、"no"和"exactly one"在单调性方面的差异：

- "I have a pet squirrel"蕴含"I have a pet"，但不蕴含"I have a happy pet squirrel"。
- "I have no pet squirrels"不蕴含"I have no pets"，但蕴含"I have no happy pet squirrels"。
- "I have exactly one pet squirrel"既不蕴含"I have exactly one pet"也不蕴含"I have exactly one happy pet squirrel"。

在所有这些例子中，"pet squirrel"出现在我们称为量词的限制域位置。我们说：

- "a"在其限制域中是向上单调的：限制域中的蕴含产生整体陈述的蕴含。
- "no"在其限制域中是向下单调的：限制域中的蕴含产生整体陈述的反方向蕴含。
- "exactly one"在其限制域中是非单调的：限制域中的蕴含不产生整体陈述的蕴含。

通过这种方式，建立在子短语蕴含之上的句子之间的蕴含几乎总是依赖于单调性判断；例如，参见词汇蕴含。然而，由于这是一个如此通用的句子对类别，为了保持逻辑类别的有意义性，我们并不总是标注这些示例的单调性。

**更丰富的逻辑结构：区间/数字、时序**

推理中一些更高层次的方面传统上已经使用逻辑建模，例如实际的数学推理（基于数字的蕴含）和时序推理（通常被建模为关于数学时间线的推理）。

- 区间/数字："I have had more than 2 drinks tonight"蕴含"I have had more than 1 drink tonight"。
- 时序："Mary left before John entered"蕴含"John entered after Mary left"。

### E.4 知识

严格来说，世界知识和常识在语言理解的每个层面都需要用于消解词义、句法结构、照应等。因此，我们的整个套件（以及任何蕴含测试）确实在一定程度上测试了这些特征。然而，在这些类别中，我们收集的示例中蕴含不仅仅依赖于句子的正确消歧，还依赖于额外知识的应用，无论是关于世界事务的具体知识，还是关于词义或社会、物理动态的更常识性知识。

**世界知识**

在这个类别中，我们专注于可以清晰表达为事实的知识，以及更广泛且不太常见的地理、法律、政治、技术或文化知识。示例：

- "This is the most oniony article I've seen on the entire internet"蕴含"This article reads like satire"。
- "The reaction was strongly exothermic"蕴含"The reaction media got very hot"。
- "There are amazing hikes around Mt. Fuji"蕴含"There are amazing hikes in Japan"但不蕴含"There are amazing hikes in Nepal"。

**常识**

在这个类别中，我们专注于那些更难表达为事实的知识，并且我们期望大多数人独立于文化或教育背景而拥有这些知识。这包括对物理和社会动态的基本理解，以及词汇含义（超越简单的词汇蕴含或逻辑关系）。示例：

- "The announcement of Tillerson's departure sent shock waves across the globe"与"People across the globe were prepared for Tillerson's departure"矛盾。
- "Marc Sims has been seeing his barber once a week, for several years"蕴含"Marc Sims has been getting his hair cut once a week, for several years"。
- "Hummingbirds are really attracted to bright orange and red (hence why the feeders are usually these colours)"蕴含"The feeders are usually coloured so as to attract hummingbirds"。
