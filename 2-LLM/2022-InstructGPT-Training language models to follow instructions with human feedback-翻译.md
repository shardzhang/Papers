# 使用人类反馈训练语言模型遵循指令

> Long Ouyang\*, Jeff Wu\*, Xu Jiang\*, Diogo Almeida\*, Carroll L. Wainwright\*, Pamela Mishkin\*, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell†, Peter Welinder, Paul Christiano\*†, Jan Leike\*, Ryan Lowe\*
> OpenAI

\*主要作者。这是OpenAI Alignment团队的联合项目。RL和JL是团队负责人。
†工作完成于OpenAI。当前隶属机构：AA: Anthropic; PC: Alignment Research Center。
通讯作者：lowe@openai.com

arXiv:2203.02155v1 [cs.CL] 2022年3月4日

---

## 摘要

增大语言模型的规模并不能使其天然地更好地遵循用户的意图。例如，大型语言模型可能生成不真实、有毒或对用户无帮助的输出。换句话说，这些模型与其用户未对齐。在本文中，我们展示了一条通过对齐语言模型与用户意图的途径，通过在广泛的任务上使用人类反馈进行微调。我们从一组标注人员编写的提示和通过OpenAI API提交的提示开始，收集了一个标注人员展示所需模型行为的数据集，用于使用监督学习微调GPT-3。然后，我们收集了一个模型输出排名的数据集，用于使用来自人类反馈的强化学习进一步微调这个监督模型。我们将得到的模型称为InstructGPT。在我们提示分布上的人类评估中，1.3B参数的InstructGPT模型的输出优于175B GPT-3的输出，尽管参数少了100倍以上。此外，InstructGPT模型在真实性方面展现出改进，在有毒输出生成方面有所减少，同时在公共NLP数据集上仅有最小的性能回归。尽管InstructGPT仍然会犯简单错误，但我们的结果表明，使用人类反馈进行微调是将语言模型与人类意图对齐的一个有前途的方向。

---

## 1 引言

大型语言模型（LM）可以通过"提示"来执行一系列自然语言处理（NLP）任务，只需在输入中给出一些任务示例。然而，这些模型常常表现出非预期的行为，例如编造事实、生成有偏见或有毒的文本，或者干脆不遵循用户指令（Bender et al., 2021; Bommasani et al., 2021; Kenton et al., 2021; Weidinger et al., 2021; Tamkin et al., 2021; Gehman et al., 2020）。这是因为许多近期大型LM使用的语言建模目标——预测互联网网页上的下一个token——与"有帮助且安全地遵循用户指令"这一目标是不同的（Radford et al., 2019; Brown et al., 2020; Fedus et al., 2021; Rae et al., 2021; Thoppilan et al., 2022）。因此，我们说语言建模目标是未对齐的。避免这些非预期行为对于部署在数百个应用中的语言模型尤其重要。

我们通过训练语言模型按照用户的意图行事，在对齐方面取得了进展（Leike et al., 2018）。这既包括显式意图（如遵循指令），也包含隐式意图（如保持真实、不带有偏见、有毒或其他有害性）。使用Askell et al.（2021）的术语，我们希望语言模型是有帮助的（应帮助用户解决他们的任务）、诚实的（不应编造信息或误导用户）和无害的（不应对人或环境造成身体、心理或社会伤害）。我们在第3.6节详细阐述了这些标准的评估。

我们专注于对齐语言模型的微调方法。具体来说，我们使用来自人类反馈的强化学习（RLHF; Christiano et al., 2017; Stiennon et al., 2020）来微调GPT-3，使其遵循广泛的书面指令（见图2）。该技术使用人类偏好作为奖励信号来微调我们的模型。我们首先雇佣了40名承包商，根据他们在筛选测试中的表现来标注我们的数据（详见第3.4节和附录B.1）。然后，我们在（主要是英语的）提交给OpenAI API的提示和一些标注人员编写的提示上，收集了人类编写的关于期望输出行为的演示数据集，并用它来训练我们的监督学习基线模型。接下来，我们在更大的一组API提示上，收集了模型输出之间的人类标注比较数据集。然后，我们在该数据集上训练一个奖励模型（RM），以预测我们的标注人员会更喜欢哪个模型输出。最后，我们使用这个RM作为奖励函数，并使用PPO算法（Schulman et al., 2017）微调我们的监督学习基线以最大化该奖励。我们在图2中说明了这个过程。该过程将GPT-3的行为与特定人群（主要是我们的标注人员和研究人员）声明的偏好对齐，而不是与任何更广泛的"人类价值观"概念对齐；我们将在第5.2节进一步讨论这一点。我们将得到的模型称为InstructGPT。

我们主要通过让标注人员在由保留客户（不在训练数据中）提供的提示组成的测试集上，对模型输出的质量进行评分来评估模型。我们还在一系列公共NLP数据集上进行了自动评估。我们训练了三种模型规模（1.3B、6B和175B参数），所有模型均使用GPT-3架构。我们的主要发现如下：

**标注人员显著更喜欢InstructGPT的输出而非GPT-3的输出。** 在我们的测试集上，1.3B参数的InstructGPT模型的输出优于175B GPT-3的输出，尽管参数少了100倍以上。这些模型具有相同的架构，唯一的区别是InstructGPT在我们的人类数据上进行了微调。即使我们为GPT-3添加few-shot提示以使其更好地遵循指令，这一结果仍然成立。175B InstructGPT的输出在85\pm3%的情况下优于175B GPT-3的输出，在71\pm4%的情况下优于few-shot 175B GPT-3。InstructGPT模型还根据我们的标注人员生成更合适的输出，并更可靠地遵循指令中的显式约束。

**InstructGPT模型在真实性方面展现出优于GPT-3的改进。** 在TruthfulQA基准上，InstructGPT生成真实且信息丰富的回答的频率大约是GPT-3的两倍。我们的结果在非对抗性选择针对GPT-3的问题子集上同样强劲。在我们API提示分布中的"闭域"任务上（输出不应包含输入中不存在的信息，例如摘要和闭域问答），InstructGPT编造输入中不存在信息的频率大约是GPT-3的一半（分别为21% vs. 41%的幻觉率）。

**InstructGPT在毒性方面相比GPT-3有微小改进，但在偏见方面没有。** 为了衡量毒性，我们使用RealToxicityPrompts数据集（Gehman et al., 2020）进行了自动和人工评估。当被提示要尊重时，InstructGPT模型生成的有毒输出比GPT-3少约25%。InstructGPT在Winogender（Rudinger et al., 2018）和CrowSPairs（Nangia et al., 2020）数据集上相比GPT-3没有显著改进。

**我们可以通过修改RLHF微调过程来最小化公共NLP数据集上的性能回归。** 在RLHF微调过程中，我们观察到相比GPT-3在某些公共NLP数据集上的性能下降，特别是SQuAD（Rajpurkar et al., 2018）、DROP（Dua et al., 2019）、HellaSwag（Zellers et al., 2019）和WMT 2015法英翻译（Bojar et al., 2015）。这是一个"对齐税"的例子，因为我们的对齐过程以在某些我们可能关心的任务上性能降低为代价。我们通过将PPO更新与增加预训练分布对数似然的更新混合（PPO-ptx），可以大大减少这些数据集上的性能回归，且不影响标注人员的偏好评分。

**我们的模型推广到了未产生任何训练数据的"保留"标注人员的偏好。** 为了测试我们模型的泛化能力，我们进行了一项初步实验，使用保留的标注人员，发现他们以大约与我们的训练标注人员相同的比率更喜欢InstructGPT的输出而非GPT-3的输出。然而，还需要更多工作来研究这些模型在更广泛的用户群体上的表现，以及在人类对期望行为存在分歧的输入上的表现。

**公共NLP数据集不能反映我们的语言模型的使用方式。** 我们将在使用人类偏好数据微调后的GPT-3（即InstructGPT）与在两个不同的公共NLP任务汇编上微调的GPT-3进行比较：FLAN（Wei et al., 2021）和T0（Sanh et al., 2021）（特别是T0++变体）。这些数据集包含各种NLP任务，以及每个任务的自然语言指令。在我们的API提示分布上，我们的FLAN和T0模型表现略差于我们的SFT基线，标注人员显著更喜欢InstructGPT而非这些模型（InstructGPT相比我们的基线有73.4\pm2%的胜率，而我们的T0和FLAN版本分别为26.8\pm2%和29.8\pm2%）。

**InstructGPT模型展现出对RLHF微调分布之外的指令的有希望的泛化能力。** 我们定性地探索了InstructGPT的能力，发现它能够遵循用不同语言编写的指令来总结代码、回答关于代码的问题，并且有时能遵循指令，尽管这些指令在微调分布中非常罕见。相比之下，GPT-3可以执行这些任务，但需要更仔细的提示设计，并且通常不会在这些领域遵循指令。这个结果令人兴奋，因为它表明我们的模型能够泛化"遵循指令"的概念。即使在其获得很少直接监督信号的任务上，它们仍然保持了一定的对齐。

**InstructGPT仍然会犯简单错误。** 例如，InstructGPT仍然可能无法遵循指令、编造事实、对简单问题给出冗长的回避回答，或无法检测到具有虚假前提的指令。

总体而言，我们的结果表明，使用人类偏好微调大型语言模型显著改善了它们在广泛任务上的行为，尽管在提高其安全性和可靠性方面还有很多工作要做。

本文的其余部分结构如下：我们首先在第2节详细介绍相关工作，然后在第3节深入探讨方法和实验细节，包括我们的高层次方法论（3.1）、任务和数据集细节（3.3和3.2）、人类数据收集（3.4）、我们如何训练模型（3.5）以及我们的评估过程（3.6）。然后我们在第4节展示结果，分为三个部分：API提示分布上的结果（4.1）、公共NLP数据集上的结果（4.2）和定性结果（4.3）。最后我们在第5节给出扩展讨论，包括对齐研究的启示（5.1）、我们与谁对齐（5.2）、局限性（5.3）、开放问题（5.4）和本文的更广泛影响（5.5）。

## 2 相关工作

**关于从人类反馈中进行对齐和学习的研究。** 我们建立在以前将模型与人类意图对齐的技术基础上，特别是来自人类反馈的强化学习（RLHF）。最初是为在模拟环境和Atari游戏中训练简单机器人而开发的（Christiano et al., 2017; Ibarz et al., 2018），最近已被应用于微调语言模型以总结文本（Ziegler et al., 2019; Stiennon et al., 2020; Böhm et al., 2019; Wu et al., 2021）。这项工作又受到在对话（Jaques et al., 2019; Yi et al., 2019; Hancock et al., 2019）、翻译（Kreutzer et al., 2018; Bahdanau et al., 2016）、语义解析（Lawrence and Riezler, 2018）、故事生成（Zhou and Xu, 2020）、评论生成（Cho et al., 2018）和证据提取（Perez et al., 2019）等领域使用人类反馈作为奖励的类似工作的影响。Madaan et al.（2022）使用书面人类反馈来增强提示并提高GPT-3的性能。也有工作使用带有规范性先验的RL在基于文本的环境中对齐智能体（Nahian et al., 2021）。我们的工作可以看作是将RLHF直接应用于在广泛的语言任务分布上对齐语言模型。

语言模型对齐意味着什么这个问题最近也受到了关注（Gabriel, 2020）。Kenton et al.（2021）编目了由未对齐导致的LM中的行为问题，包括产生有害内容和利用错误指定的目标。在同期工作中，Askell et al.（2021）提出将语言助手作为对齐研究的试验平台，研究了一些简单的基线及其缩放特性。

**训练语言模型遵循指令。** 我们的工作也与语言模型中的跨任务泛化研究有关，其中LM在广泛的公共NLP数据集上微调（通常带有适当的指令前缀），并在不同的一组NLP任务上进行评估。这个领域已有一些工作（Yi et al., 2019; Mishra et al., 2021; Wei et al., 2021; Khashabi et al., 2020; Sanh et al., 2021; Aribandi et al., 2021），它们在训练和评估数据、指令格式、预训练模型大小和其他实验细节上有所不同。跨研究的一个一致发现是，在带有指令的一系列NLP任务上微调LM，可以提高它们在保留任务上的下游性能，无论是在zero-shot还是few-shot设置中。

还有一系列相关工作是关于导航中的指令遵循，其中模型被训练来遵循自然语言指令在模拟环境中导航（Bahdanau et al., 2018; Abramson et al., 2020; Zhao et al., 2021）。

**评估语言模型的危害。** 修改语言模型行为的一个目标是减轻这些模型在现实世界中部署时的危害。这些风险已被广泛记录（Bender et al., 2021; Bommasani et al., 2021; Kenton et al., 2021; Weidinger et al., 2021; Tamkin et al., 2021）。语言模型可能产生有偏见的输出（Dhamala et al., 2021; Liang et al., 2021; Manela et al., 2021; Caliskan et al., 2017; Kirk et al., 2021）、泄露私人数据（Carlini et al., 2021）、生成错误信息（Solaiman et al., 2019; Buchanan et al., 2021），并被恶意使用；详细综述请参阅Weidinger et al.（2021）。在特定领域部署语言模型会带来新的风险和挑战，例如在对话系统中（Henderson et al., 2018; Xu et al., 2020; Dinan et al., 2019b）。有一个新兴但不断发展的领域旨在建立基准来具体评估这些危害，特别是关于毒性（Gehman et al., 2020）、刻板印象（Nadeem et al., 2020）和社会偏见（Dhamala et al., 2021; Nangia et al., 2020; Rudinger et al., 2018）。在这些问题上取得重大进展是困难的，因为出于善意的对LM行为的干预可能产生副作用（Welbl et al., 2021; Blodgett et al., 2020）；例如，减少LM毒性的努力可能会降低它们对来自代表性不足群体的文本建模的能力，这是由于训练数据中存在偏见相关性（Xu et al., 2021）。

**修改语言模型的行为以减轻危害。** 改变语言模型生成行为的方法有很多。Solaiman and Dennison（2021）在一个小的、针对价值观的数据集上微调LM，这提高了模型在问答任务上遵守这些价值观的能力。Ngo et al.（2021）通过移除语言模型在生成一组研究人员编写的触发短语时具有高条件似然的文档来过滤预训练数据集。在这个过滤后的数据集上训练时，它们的LM生成更少的有害文本，代价是语言建模性能略有下降。Xu et al.（2020）使用了多种方法来提高聊天机器人的安全性，包括数据过滤、在生成过程中屏蔽某些词语或n-gram、安全特定的控制token（Keskar et al., 2019; Dinan et al., 2019a），以及人在环数据收集（Dinan et al., 2019b）。其他减轻LM生成偏见的方方法包括词嵌入正则化（Liu et al., 2019; Huang et al., 2019）、数据增强（Liu et al., 2019; Dinan et al., 2019a; Sheng et al., 2019）、使敏感token上的分布更均匀的零空间投影（Liang et al., 2021）、不同的目标函数（Qian et al., 2019），或因果中介分析（Vig et al., 2020）。还有工作使用第二个（通常较小的）语言模型来引导语言模型的生成（Dathathri et al., 2019; Krause et al., 2020），该方法的变体已被应用于减少语言模型的毒性（Schick et al., 2021）。

## 3 方法和实验细节

### 3.1 高层次方法论

我们的方法遵循Ziegler et al.（2019）和Stiennon et al.（2020）的方法，他们将此方法应用于风格延续和摘要领域。我们从预训练的语言模型（Radford et al., 2019; Brown et al., 2020; Fedus et al., 2021; Rae et al., 2021; Thoppilan et al., 2022）、我们希望在其上产生对齐输出的提示分布，以及经过培训的人类标注人员团队（详见第3.4节）开始。然后我们应用以下三个步骤（图2）。

**步骤1：收集演示数据，训练监督策略。** 我们的标注人员在输入提示分布上提供期望行为的演示（关于该分布的详细信息见第3.2节）。然后我们使用监督学习在这个数据上微调预训练的GPT-3模型。

**步骤2：收集比较数据，训练奖励模型。** 我们收集一个模型输出之间比较的数据集，其中标注人员针对给定输入指出他们更偏好哪个输出。然后我们训练一个奖励模型来预测人类偏好的输出。

**步骤3：使用PPO针对奖励模型优化策略。** 我们使用RM的输出作为标量奖励。我们使用PPO算法（Schulman et al., 2017）微调监督策略以优化这个奖励。

步骤2和3可以持续迭代；在当前最佳策略上收集更多的比较数据，用于训练新的RM和新的策略。在实践中，我们的大部分比较数据来自我们的监督策略，也有一些来自我们的PPO策略。

### 3.2 数据集

我们的提示数据集主要包括提交给OpenAI API的文本提示，特别是那些在Playground界面上使用早期版本的InstructGPT模型（通过在我们的演示数据子集上进行监督学习训练）的提示。[4] 使用Playground的客户通过每次使用InstructGPT模型时的重复通知被告知，他们的数据可能被用于训练进一步的模型。在本文中，我们不使用在生成环境中使用API的客户的数据。我们通过检查共享长公共前缀的提示来进行启发式去重，并将每个用户ID的提示数量限制为200个。我们还基于用户ID创建训练集、验证集和测试集划分，以便验证集和测试集不包含训练集中用户的数据。为了避免模型学习潜在的敏感客户详细信息，我们过滤掉了训练划分中所有包含个人身份信息（PII）的提示。

[4] 这是OpenAI托管的用于直接与API模型交互的界面；参见 https://beta.openai.com/playground。

为了训练最初的InstructGPT模型，我们要求标注人员自己编写提示。这是因为我们需要一个初始的指令式提示来源来引导整个过程，而这类提示通常不会在API上提交给常规的GPT-3模型。我们要求标注人员编写三种类型的提示：

*   **Plain（普通）：** 我们简单地要求标注人员提出一个任意任务，同时确保任务具有足够的多样性。
*   **Few-shot（少样本）：** 我们要求标注人员提出一个指令，以及该指令的多个查询/响应对。
*   **User-based（基于用户）：** 我们在OpenAI API的候补名单申请中看到许多用例。我们要求标注人员提出与这些用例对应的提示。

从这些提示中，我们生成了微调过程中使用的三个不同数据集：（1）SFT数据集，包含用于训练SFT模型的标注人员演示；（2）RM数据集，包含用于训练RM的模型输出的标注人员排名；（3）PPO数据集，没有任何人工标签，用作RLHF微调的输入。SFT数据集包含约13K训练提示（来自API和标注人员编写的），RM数据集有33K训练提示（来自API和标注人员编写的），PPO数据集有31K训练提示（仅来自API）。数据集大小的更多细节见表6。

为了让大家了解我们数据集的构成，表1显示了我们的标注人员为API提示（具体是RM数据集）标记的用例类别分布。大多数用例是生成式的，而不是分类或问答。我们还在表2中展示了一些说明性提示（由研究人员编写以模仿提交给InstructGPT模型的提示类型）；更多提交给InstructGPT模型的提示见附录A.2.1，提交给GPT-3模型的提示见附录A.2.2。我们在附录A中提供了关于数据集的更多细节。

### 3.3 任务

我们的训练任务来自两个来源：（1）由我们的标注人员编写的提示数据集，和（2）提交给API上早期InstructGPT模型的提示数据集（见表6）。这些提示非常多样化，包括生成、问答、对话、摘要、提取和其他自然语言任务（见表1）。我们的数据集96%以上是英语，但在第4.3节中，我们也探究了模型用其他语言响应指令和完成编码任务的能力。

对于每个自然语言提示，任务通常通过自然语言指令直接指定（例如"写一个关于一只聪明青蛙的故事"），但也可能通过few-shot示例间接指定（例如提供两个青蛙故事的示例，然后提示模型生成一个新的）或通过隐式延续（例如提供关于一个青蛙的故事开头）。在每种情况下，我们要求标注人员尽最大努力推断编写提示的用户的意图，并要求他们跳过任务非常不明确的输入。此外，我们的标注人员还根据我们提供的指示（见附录B）和他们自己的最佳判断，考虑隐式意图，如回复的真实性，以及潜在的有害输出，如偏见或有毒语言。

### 3.4 人类数据收集

为了生成我们的演示和比较数据，并进行我们的主要评估，我们通过Upwork和ScaleAI雇佣了大约40名承包商。与早期在摘要任务上收集人类偏好数据的工作（Ziegler et al., 2019; Stiennon et al., 2020; Wu et al., 2021）相比，我们的输入涵盖更广泛的任务，并且偶尔可能涉及有争议和敏感的话题。我们的目标是选择一组对不同人口群体的偏好敏感、并且擅长识别潜在有害输出的标注人员。因此，我们进行了一项筛选测试，旨在衡量标注人员在这些方面的表现。我们选择了在此测试中表现良好的标注人员；关于我们的选择程序和标注人员人口统计的更多信息，请参见附录B.1。

在训练和评估过程中，我们的对齐标准可能发生冲突：例如，当用户请求一个潜在有害的响应时。在训练过程中，我们优先考虑对用户的有帮助性（不这样做需要做出一些困难的设计决策，我们留给未来工作；更多讨论见第5.4节）。然而，在我们的最终评估中，我们要求标注人员优先考虑真实性和无害性（因为这是我们真正关心的）。

与Stiennon et al.（2020）一样，我们在项目过程中与标注人员密切合作。我们有一个入职流程来培训标注人员了解项目，为每个任务编写详细的指导（见附录B.2），并在共享聊天室中回答标注人员的问题。

作为一项初步研究，为了了解我们的模型在多大程度上能推广到其他标注人员的偏好，我们雇佣了一组不产生任何训练数据的独立标注人员。这些标注人员来自相同的供应商，但未经过筛选测试。

尽管任务复杂，我们发现标注者间的一致率相当高：训练标注人员之间的一致率为72.6\pm1.5%，而保留标注人员的一致率为77.3\pm1.3%。作为比较，在Stiennon et al.（2020）的摘要工作中，研究人员之间的一致率为73\pm4%。

### 3.5 模型

我们从Brown et al.（2020）的GPT-3预训练语言模型开始。这些模型在广泛的互联网数据分布上训练，可以适应各种下游任务，但其行为特征不明确。从这些模型开始，我们使用三种不同的技术训练模型：

**监督微调（SFT）。** 我们使用监督学习在标注人员演示上微调GPT-3。我们训练了16个epoch，使用余弦学习率衰减和0.2的残差dropout。我们根据验证集上的RM评分进行最终的SFT模型选择。与Wu et al.（2021）类似，我们发现SFT模型在1个epoch后在验证损失上过拟合；然而，我们发现尽管存在过拟合，训练更多epoch有助于RM评分和人类偏好评分。

**奖励建模（RM）。** 从移除最终unembedding层的SFT模型开始，我们训练一个模型，输入提示和响应，输出标量奖励。在本文中，我们只使用6B RM，因为这节省了大量计算，并且我们发现175B RM训练可能不稳定，因此不太适合在RL中用作价值函数（更多细节见附录C）。

在Stiennon et al.（2020）中，RM在同一个输入上两个模型输出之间的比较数据集上训练。他们使用交叉熵损失，以比较结果作为标签——奖励的差异代表了一个响应相比另一个被人类标注人员偏爱的对数几率。

为了加快比较收集速度，我们让标注人员对K=4到K=9个响应进行排序。这为每个展示给标注人员的提示产生了 $inom{K}{2}$ 个比较。由于每个标注任务内的比较高度相关，我们发现如果简单地将比较结果打乱成一个数据集，单次遍历数据集会导致奖励模型过拟合。[5] 相反，我们将每个提示的所有 $inom{K}{2}$ 个比较作为单个批处理元素进行训练。这在计算上更加高效，因为每个完成只需要RM的一次前向传播（而不是K个完成的 $inom{K}{2}$ 次前向传播），而且由于它不再过拟合，实现了更好的验证准确率和对数损失。

具体来说，奖励模型的损失函数是：

$$
\text{loss}(\theta) = -\frac{1}{\binom{K}{2}} \mathbb{E}_{(x,y_w,y_l)\sim D} \left[ \log \left( \sigma \left( r_\theta(x,y_w) - r_\theta(x,y_l) \right) \right) \right] \qquad (1)
$$

其中 $r_\theta(x, y)$ 是奖励模型对提示 $x$ 和完成 $y$ 的标量输出（参数为 $\theta$ ）， $y_w$ 是 $y_w$ 和 $y_l$ 这对完成中更受偏好的一个， $D$ 是人类比较数据集。

[5] 也就是说，如果每个可能的 $\binom{K}{2}$ 个比较都被视为单独的数据点，那么每个完成可能被用于 $K-1$ 次单独的梯度更新。模型在一个epoch后倾向于过拟合，因此在一个epoch内重复数据也会导致过拟合。

最后，由于RM损失对奖励的平移不变，我们使用偏置对奖励模型进行归一化，使得标注人员演示在RL之前的平均得分为0。

**强化学习（RL）。** 再次遵循Stiennon et al.（2020），我们使用PPO（Schulman et al., 2017）在环境中微调SFT模型。该环境是一个bandit环境，呈现随机客户提示并期望对提示的响应。给定提示和响应，它产生由奖励模型确定的奖励并结束回合。此外，我们在每个token上添加来自SFT模型的逐token KL惩罚，以减轻对奖励模型的过度优化。价值函数从RM初始化。我们将这些模型称为"PPO"模型。

我们还尝试将预训练梯度混合到PPO梯度中，以修复公共NLP数据集上的性能回归。我们将这些模型称为"PPO-ptx"。我们在RL训练中最大化以下组合目标函数：

$$
\text{objective}(\phi) = \mathbb{E}_{(x,y)\sim D_{\pi^{RL}_\phi}} \left[ r_\theta(x,y) - \beta \log \left( \pi^{RL}_\phi(y|x) / \pi^{SFT}(y|x) \right) \right] + \gamma \mathbb{E}_{x\sim D_{\text{pretrain}}} \left[ \log(\pi^{RL}_\phi(x)) \right] \qquad (2)
$$

其中 $\pi^{RL}_\phi$ 是学习的RL策略， $\pi^{SFT}$ 是监督训练模型， $D_{\text{pretrain}}$ 是预训练分布。KL奖励系数 $\beta$ 和预训练损失系数 $\gamma$ 分别控制KL惩罚和预训练梯度的强度。对于"PPO"模型， $\gamma$ 设为0。除非另有说明，本文中InstructGPT指PPO-ptx模型。

**基线。** 我们将PPO模型的性能与SFT模型和GPT-3进行比较。我们还将GPT-3与提供给它的few-shot前缀进行比较，以将其"提示"到指令遵循模式（GPT-3-prompted）。此前缀被附加在用户指定的指令之前。[6]

我们还将InstructGPT与在FLAN（Wei et al., 2021）和T0（Sanh et al., 2021）数据集上微调的175B GPT-3进行比较，这两个数据集都包含各种NLP任务，以及每个任务的自然语言指令（这些数据集在包含的NLP数据集和使用的指令风格上有所不同）。我们分别在大约100万个样本上对它们进行微调，并选择在验证集上获得最高奖励模型评分的检查点。更多训练细节见附录C。

[6] 为了获得此前缀，作者RL和DA举行了一个前缀寻找比赛：每人花一个小时与GPT-3交互，提出他们最好的两个前缀。获胜的前缀是使GPT-3在提示验证集上获得最高RM分数的前缀。DA获胜。

### 3.6 评估

为了评估我们的模型有多"对齐"，我们首先需要澄清对齐在这个上下文中的含义。对齐的定义在历史上一直是一个模糊和令人困惑的话题，有各种相互竞争的提议（Chen et al., 2021; Leike et al., 2018; Gabriel, 2020）。遵循Leike et al.（2018），我们的目标是训练按照用户意图行事的模型。更实际地说，出于我们语言任务的目的，我们使用类似于Askell et al.（2021）的框架，他们定义如果模型是有帮助的、诚实的和无害的，则模型是对齐的。

为了有帮助，模型应该遵循指令，但也应从few-shot提示或其他可解释的模式（如"Q: {question}\nA:"）推断意图。由于给定提示的意图可能不清楚或模糊，我们依赖标注人员的判断，我们的主要指标是标注人员偏好评分。然而，由于我们的标注人员不是生成提示的用户，用户的实际意图与标注人员仅通过阅读提示所认为的意图之间可能存在差异。

目前尚不清楚如何在纯生成模型中衡量诚实性；这需要将模型的实际输出与其对正确输出的"信念"进行比较，由于模型是一个大黑箱，我们无法推断其信念。相反，我们使用两个指标来衡量真实性——模型关于世界的陈述是否真实：（1）评估模型在闭域任务上编造信息的倾向（"幻觉"），和（2）使用TruthfulQA数据集（Lin et al., 2021）。不用说，这仅捕捉了真实性的实际含义的一小部分。

与诚实性类似，衡量语言模型的危害也面临许多挑战。在大多数情况下，语言模型的危害取决于其输出在现实世界中的使用方式。例如，生成有毒输出的模型在部署的聊天机器人上下文中可能是有害的，但如果用于数据增强以训练更准确的毒性检测模型，甚至可能是有帮助的。在项目早期，我们让标注人员评估输出是否"潜在有害"。然而，我们停止了这一点，因为这需要太多关于输出最终将如何被使用的推测；特别是因为我们的数据也来自与Playground API界面交互的客户（而不是来自生产用例）。

因此，我们使用一套更具体的代理标准，旨在捕捉已部署模型中可能最终造成危害的行为的不同方面：我们让标注人员评估输出在客户助手上下文中是否不合适、是否贬低受保护类别、或包含性或暴力内容。我们还在旨在衡量偏见和毒性的数据集上对我们的模型进行基准测试，如RealToxicityPrompts（Gehman et al., 2020）和CrowS-Pairs（Nangia et al., 2020）。

总结一下，我们可以将定量评估分为两个独立的部分：

**API分布上的评估。** 我们的主要指标是在来自与训练分布相同来源的保留提示集上的人类偏好评分。当使用来自API的提示进行评估时，我们只选择未包含在训练中的客户的提示。然而，鉴于我们的训练提示是为与InstructGPT模型一起使用而设计的，这可能不利于GPT-3基线。因此，我们还在提交给API上的GPT-3模型的提示上进行评估；这些提示通常不是"指令遵循"的风格，而是专门为GPT-3设计的。在这两种情况下，对于每个模型，我们计算其输出相对于基线策略被偏爱的频率；我们选择175B SFT模型作为基线，因为其性能处于中游。此外，我们要求标注人员在1-7 Likert量表上判断每个响应的整体质量，并为每个模型输出收集一系列元数据（见表3）。

**公共NLP数据集上的评估。** 我们在两种类型的公共数据集上评估：那些捕捉语言模型安全某个方面的（特别是真实性、毒性和偏见），以及那些捕捉传统NLP任务（如问答、阅读理解和摘要）中zero-shot性能的。我们还在RealToxicityPrompts数据集（Gehman et al., 2020）上进行毒性的人工评估。我们正在发布我们的模型在所有基于抽样的NLP任务上的样本。[7]

[7] 可在此处获取：https://github.com/openai/following-instructions-human-feedback。

## 4 结果

在本节中，我们为第1节中的声明提供实验证据，分为三部分：API提示分布上的结果、公共NLP数据集上的结果和定性结果。

### 4.1 API分布上的结果

**标注人员显著更喜欢InstructGPT的输出而非GPT-3的输出。** 在我们的提示测试集上，标注人员显著更喜欢各模型规模的InstructGPT输出。这些结果显示在图1中。我们发现GPT-3输出表现最差，通过使用精心设计的few-shot提示（GPT-3 (prompted)）可以获得显著的逐步改进，然后通过监督学习在演示上进行训练（SFT），最后通过使用PPO在比较数据上进行训练。在PPO期间添加预训练混合的更新不会导致标注人员偏好的大变化。为了说明我们收益的幅度：在直接比较中，175B InstructGPT输出在85\pm3%的情况下优于GPT-3输出，在71\pm4%的情况下优于few-shot GPT-3。

我们还发现，在提交给API上的GPT-3模型的提示上评估时，我们的结果没有显著变化（见图3），尽管我们的PPO-ptx模型在更大的模型规模上表现略差。

在图4中，我们展示了标注人员在几个更具体的维度上对InstructGPT输出给出了有利评价。具体来说，与GPT-3相比，InstructGPT输出在客户助手的上下文中更合适，更经常遵循指令中定义的显式约束（例如"用两段或更少的字数写你的答案"），更少完全无法遵循正确指令，并且在闭域任务中更少编造事实（"幻觉"）。这些结果表明InstructGPT模型比GPT-3更可靠、更易于控制。我们发现，我们的其他元数据类别在我们的API中出现频率过低，无法获得模型之间的统计显著差异。

**我们的模型推广到了未产生任何训练数据的"保留"标注人员的偏好。** 保留的标注人员具有与我们用于产生训练数据的工作人员相似的排名偏好（见图3）。特别是，根据保留工作人员的评价，我们所有的InstructGPT模型仍然大大优于GPT-3基线。因此，我们的InstructGPT模型不仅仅是对训练标注人员偏好的过拟合。

我们从奖励模型的泛化能力中看到了进一步的证据。我们进行了一项实验，将标注人员分为5组，并使用5折交叉验证（在4组上训练，在保留组上评估）训练了5个RM（使用3个不同的种子）。这些RM在预测保留组中标注人员的偏好方面的准确率为69.6\pm0.9%，相比它们在预测训练集中标注人员偏好方面的72.4\pm0.4%的准确率略有下降。

**公共NLP数据集不能反映我们的语言模型的使用方式。** 在图5中，我们还将InstructGPT与在FLAN（Wei et al., 2021）和T0（Sanh et al., 2021）数据集上微调的175B GPT-3基线进行比较（详情见附录C）。我们发现这些模型表现优于GPT-3，与具有精心选择提示的GPT-3相当，但差于我们的SFT基线。这表明这些数据集不足以多样化以提高我们在API提示分布上的性能。在正面比较中，我们的175B InstructGPT模型输出优于我们的FLAN模型78\pm4%的时间，优于我们的T0模型79\pm4%的时间。这些模型的Likert分数见图5。

我们认为我们的InstructGPT模型优于FLAN和T0有两个原因。首先，公共NLP数据集旨在捕捉易于用自动指标评估的任务，如分类、问答，以及某种程度上摘要和翻译。然而，分类和问答仅占API客户使用我们语言模型的一小部分（约18%），而开放式的生成和头脑风暴根据标注人员占我们提示数据集的约57%（见表1）。其次，公共NLP数据集很难获得非常高的输入多样性（至少在真实世界用户可能感兴趣使用的输入类型方面）。当然，NLP数据集中的任务确实代表了我们希望语言模型能够解决的一种指令类型，所以最广泛的指令遵循模型类型将结合两种数据集。

### 4.2 公共NLP数据集上的结果

**InstructGPT模型在真实性方面展现出优于GPT-3的改进。** 根据在TruthfulQA数据集上的人类评估，我们的PPO模型在生成真实且信息丰富的输出方面与GPT-3相比显示出微小但显著的改进（见图6）。这是默认行为：我们的模型无需特别指示要说真话就能表现出改进的真实性。有趣的是，例外是我们的1.3B PPO-ptx模型，其表现略差于相同规模的GPT-3模型。当仅对非对抗性选择针对GPT-3的提示进行评估时，我们的PPO模型仍然显著比GPT-3更真实和信息丰富（尽管绝对改进下降了几个百分点）。

遵循Lin et al.（2021），我们还给出了一个有帮助的"Instruction+QA"提示，指示模型在不确信正确答案时回应"I have no comment"。在这种情况下，我们的PPO模型倾向于真实但不提供信息，而不是自信地说出虚假信息；基线GPT-3模型在这方面表现不佳。

我们在真实性方面的改进也由我们的PPO模型在API分布中的闭域任务上更少产生幻觉（即编造信息）这一事实所证明，我们在图4中展示了这一点。

**InstructGPT在毒性方面相比GPT-3有微小改进，但在偏见方面没有。** 我们首先在RealToxicityPrompts数据集（Gehman et al., 2020）上评估我们的模型。我们通过两种方式做到这一点：通过Perspective API[8]运行模型样本以获得自动毒性评分（这是该数据集的标准评估程序），以及将这些样本发送给标注人员以获得关于绝对毒性、相对于提示的毒性、连续性和整体输出偏好的评分。我们根据提示毒性均匀采样该数据集中的提示，以便更好地评估模型在高输入毒性下的表现（见附录E中的图39）；这与该数据集的标准提示采样不同，因此我们的绝对毒性数字被高估了。

[8] www.perspectiveapi.com

我们的结果在图7中。我们发现，当被指示产生安全和尊重的输出（"尊重提示"）时，根据Perspective API，InstructGPT模型生成的输出毒性低于GPT-3。当去除尊重提示时（"无提示"），这一优势消失。有趣的是，当被明确提示产生有毒输出时，InstructGPT输出的毒性远高于GPT-3（见图39）。

这些结果在我们的人类评估中得到确认：InstructGPT在"尊重提示"设置下毒性低于GPT-3，但在"无提示"设置下表现相似。我们在附录E中提供了扩展结果。总结：我们所有的模型都被评为比给定提示预期的毒性更低（它们在-1到1的尺度上获得负分，其中0表示"和预期差不多毒性"）。我们的SFT基线是所有模型中毒性最低的，但连续性和偏好排名也最低，这可能表明模型生成了非常短或退化的响应。

为了评估模型产生偏见言论的倾向（见附录E），我们还在Winogender（Rudinger et al., 2018）和CrowS-Pairs（Nangia et al., 2020）数据集的修改版本上评估了InstructGPT。这些数据集由可能突出潜在偏见的句子对组成。我们计算在每个句子对中产生句子的相对概率以及相关二元概率分布的熵（以比特为单位）。完全无偏的模型在每对句子之间没有偏好，因此具有最大熵。根据这个指标，我们的模型并不比GPT-3更少偏见。PPO-ptx模型显示出与GPT-3相似的偏见，但当被指示尊重地行事时，它表现出更低的熵，因此更高的偏见。偏见的模式不清楚；似乎被指示的模型对其输出更确信，无论其输出是否表现出刻板行为。

**我们可以通过修改RLHF微调过程来最小化公共NLP数据集上的性能回归。** 默认情况下，当我们在API分布上训练PPO模型时，它遭受了"对齐税"，因为在几个公共NLP数据集上的性能下降。我们希望有一个避免对齐税的对齐过程，因为对齐税会激励使用未对齐但在这些任务上更有能力的模型。

在图29中，我们展示了在我们的PPO微调中添加预训练更新（PPO-ptx）减轻了所有数据集上的性能回归，甚至在HellaSwag上超过了GPT-3。PPO-ptx模型的性能在DROP、SQuADv2和翻译方面仍然落后于GPT-3；需要更多工作来研究和进一步消除这些性能回归。

混合预训练更新比增加KL系数的更简单方案效果更好。在图33中，我们展示存在一个预训练混合系数的值，既可以逆转SQuADv2和DROP（我们用于测试的数据集）上的性能回归，又使验证奖励的减少最小化。相比之下，增加KL系数（图34）导致验证奖励显著下降，并且从未在DROP和SQuAD上完全恢复。将KL模型从PPO初始模型改为GPT-3得到类似结果。

### 4.3 定性结果

**InstructGPT模型展现出对RLHF微调分布之外的指令的有希望的泛化能力。** 特别是，我们发现InstructGPT展示了用非英语语言遵循指令、以及执行代码摘要和问答的能力。这很有趣，因为非英语语言和代码仅占我们微调数据的极小部分[9]，这表明在某些情况下，对齐方法可以推广到人类未直接监督的输入上产生期望行为。

[9] 我们通常指示标注人员在缺少所需专业知识时跳过评估，尽管有时标注人员使用翻译服务来评估他们不会说的语言中的简单指令。

我们没有定量跟踪这些行为，但在图8中展示了一些定性示例。我们的175B PPO-ptx模型能够可靠地回答关于代码的问题，并且也可以遵循其他语言的指令；然而，我们注意到即使指令是其他语言，它也经常产生英语输出。相比之下，我们发现GPT-3可以执行这些任务，但需要更仔细的提示设计，并且很少在这些领域遵循指令。

**InstructGPT仍然会犯简单错误。** 在与我们的175B PPO-ptx模型交互中，我们注意到它仍然可能犯简单错误，尽管在许多不同语言任务上表现强劲。举几个例子：（1）当给出一个具有错误前提的指令时，模型有时会错误地假定前提为真；（2）模型可能过度回避；当给出一个简单问题时，它有时会说没有唯一答案，并给出多种可能的答案，即使根据上下文有一个相当清晰的答案；（3）当指令包含多个显式约束时（例如，"列出10部1930年代在法国拍摄的电影"）或当约束对语言模型来说可能具有挑战性时（例如，用指定的句子数写摘要），模型性能下降。

我们在图9中展示了一些这些行为的示例。我们怀疑行为（2）部分是因为我们指示标注人员奖励认知谦逊；因此，他们可能倾向于奖励那些回避的输出，这被我们的奖励模型捕捉到了。我们怀疑行为（1）是因为训练集中很少有假设错误前提的提示，我们的模型不能很好地推广到这些示例。我们相信这两种行为都可以通过对抗性数据收集（Dinan et al., 2019b）大幅减少。

## 5 讨论

### 5.1 对齐研究的启示

这项研究是我们更广泛研究计划的一部分，旨在使AI系统与人类意图对齐（Christiano et al., 2017; Ziegler et al., 2019; Stiennon et al., 2020）。尽管这项工作聚焦于我们当前的语言模型系统，但我们寻求适用于未来AI系统的通用和可扩展的方法（Leike et al., 2018）。我们在这里使用的系统仍然相当有限，但它们是当今最大的语言模型之一，我们将它们应用于广泛的语言任务，包括分类、摘要、问答、创意写作、对话等。

我们在这项工作中进行对齐研究的方法是迭代的：我们改进当前AI系统的对齐，而不是抽象地专注于对齐尚不存在的AI系统。这种方法的一个缺点是我们没有直接面对仅在对齐超人类系统时才会出现的对齐问题（Bostrom, 2014）。然而，我们的方法确实为我们提供了一个清晰的实证反馈循环，了解什么有效和什么无效。我们相信这个反馈循环对于完善我们的对齐技术至关重要，它迫使我们跟上机器学习领域的进步。此外，我们在这里使用的对齐技术RLHF，是几个关于对齐超人类系统提案的重要组成部分（Leike et al., 2018; Irving et al., 2018; Christiano et al., 2018）。例如，RLHF是最近关于总结书籍的工作的核心方法，这一任务展示了对齐超人类AI系统的一些困难，因为人类难以直接评估（Wu et al., 2021）。

从这项工作中，我们可以更广泛地得出对齐研究的经验教训：

1. **增加模型对齐的成本相对于预训练是适度的。** 收集我们数据的成本和训练运行的计算量，包括实验运行，只是训练GPT-3所用成本的一小部分：训练我们的175B SFT模型需要4.9 petaflops/s-days，训练我们的175B PPO-ptx模型需要60 petaflops/s-days，相比GPT-3的3,640 petaflops/s-days（Brown et al., 2020）。同时，我们的结果表明RLHF在使语言模型对用户更有帮助方面非常有效，超过了100倍模型规模增加的效果。这表明目前增加对现有语言模型对齐的投资比起训练更大的模型更具成本效益——至少对我们的客户的自然语言任务分布而言是如此。

2. **我们观察到一些证据表明InstructGPT将"遵循指令"推广到了我们没有监督它的设置，** 例如在非英语语言任务和代码相关任务上。这是一个重要的性质，因为让人类监督模型执行的每一个任务成本过高。需要更多研究来了解这种泛化能力如何随着能力增强而缩放；参见Christiano et al.（2021）关于这一方向的近期研究。

3. **我们能够减轻由我们的微调引入的大部分性能退化。** 如果不是这样，这些性能退化将构成对齐税——对齐模型的额外成本。任何具有高税收的技术可能不会被采用。为了避免激励未来的高能力AI系统保持与人类意图未对齐，需要具有低对齐税的对齐技术。为此，我们的结果对RLHF作为低税对齐技术来说是好消息。

4. **我们已经在现实世界中验证了来自研究的对齐技术。** 对齐研究历史上相当抽象，要么关注理论结果（Soares et al., 2015），要么关注小的合成领域（Christiano et al., 2018; Leike et al., 2017），要么是在公共NLP数据集上训练ML模型（Ziegler et al., 2019; Stiennon et al., 2020）。我们的工作为AI系统中的对齐研究提供了基础，这些系统正在现实世界中与客户一起生产使用。[10] 这使得技术有效性和局限性之间能够形成重要的反馈循环。

[10] 注意，虽然在使用人类数据微调模型时是部署ML系统的常见做法，但这些努力的目标是获得一个在公司特定用例上表现良好的模型，而不是推进通用ML模型的对齐。

### 5.2 我们与谁对齐？

当将语言模型与人类意图对齐时，它们的最终行为是底层模型（及其训练数据）、微调数据和对齐方法的函数。在本节中，我们描述了许多专门影响微调数据的因素，以最终确定我们正在对齐什么和谁。然后我们考虑在5.3节更广泛地讨论我们工作的局限性之前需要改进的领域。

文献常常使用诸如"人类偏好"或"人类价值观"这样的术语来框架化对齐。在这项工作中，我们与一组标注人员的偏好对齐，这些偏好受到它们所收到的指示、接收指示的上下文（作为有偿工作）以及从谁那里收到指示等因素的影响。一些关键注意事项适用：

首先，我们与提供训练数据的标注人员的演示和偏好对齐，他们直接生产了我们用于微调模型的数据。我们在附录B中描述了标注人员的招聘过程和人口统计；一般来说，他们是主要通过Upwork或Scale AI雇佣的居住在美国或东南亚的讲英语的人。他们在许多示例上彼此意见不同；我们发现标注者间的一致率约为73%。

其次，我们与我们的偏好对齐，作为设计本项研究的研究人员（因此代理为我们更广泛的研究组织OpenAI）：我们编写标注人员在编写演示和选择他们偏好的输出时用作指南的标注指示，并在共享聊天室中回答他们关于边缘情况的问题。需要更多研究来了解不同的指示集和界面设计对从标注人员收集的数据及其对模型行为的最终影响的确切效果。

第三，我们的训练数据由OpenAI客户提交给OpenAI API Playground上的模型的提示决定，因此我们隐式地与客户认为有价值的内容对齐，在某些情况下，与他们的最终用户认为当前使用API有价值的内容对齐。客户及其最终用户可能意见不一致，或者客户可能没有优化最终用户的福祉；例如，客户可能想要一个最大化用户在平台上花费时间的模型，这不一定是最终用户想要的。在实践中，我们的标注人员无法看到给定提示或完成将在其中被看到的上下文。

第四，OpenAI的客户并不代表所有潜在或当前的语言模型用户——更不用说所有受语言模型使用影响的个人和群体了。在本项目的大部分时间里，OpenAI API的用户是从候补名单中选出的。这个候补名单的初始种子是OpenAI的员工，使最终群体偏向我们自己的网络。

退后一步，设计一个公平、透明并具有适当问责机制的对齐过程存在许多困难。本文的目标是证明这种对齐技术可以针对特定应用与特定的偏好参考群体对齐。我们并不主张研究人员、我们雇佣的标注人员或我们的API客户是偏好的正确来源。需要考虑许多利益相关者——训练模型的组织、使用模型开发产品的客户、这些产品的最终用户，以及可能直接或间接受影响的更广泛的人群。这不仅是一个让对齐过程更具参与性的问题；更根本的是，不可能训练出一个与所有人的偏好同时对齐，或者每个人都认可其权衡取舍的系统。

一条前进的道路可能是训练可以以某些群体的偏好为条件的模型，或者可以容易地微调或提示以代表不同群体的模型。然后，不同的模型可以由认可不同价值观的群体部署和使用。然而，这些模型仍然可能最终影响更广泛的社会，并且需要做出许多艰难的决定，涉及以谁的偏好为条件，以及如何确保所有群体都能被代表并能选择退出可能有害的过程。

### 5.3 局限性

**方法论。** 我们的InstructGPT模型的行为部分由我们承包商的的反馈决定。一些标注任务依赖于价值判断，可能受到承包商身份、他们的信仰、文化背景和个人历史的影响。我们雇用了约40名承包商，依据他们在旨在判断他们识别和回应敏感提示能力的筛选测试中的表现，以及他们在具有详细指示的标注任务上与研究人员的一致率（见附录B）。我们保持承包商团队规模较小，因为这便于与全职进行该任务的较小承包商团队进行高带宽沟通。然而，这个群体显然不能代表所有将使用并受我们部署模型影响的各类人群。举个简单的例子，我们的标注人员主要是讲英语的，我们的数据几乎完全由英语指令组成。

我们还可以在许多方面改进我们的数据收集设置。例如，出于成本原因，大多数比较仅由1名承包商标记。多次标记样本可以帮助识别我们承包商意见不一致的领域，因此在某个单一模型不太可能与他们所有人对齐的领域。在意见不一致的情况下，与平均标注人员偏好对齐可能并不理想。例如，当生成对少数群体产生不成比例影响的文本时，我们可能希望属于该群体的标注人员的偏好被更重地加权。

**模型。** 我们的模型既不完全对齐也不完全安全；它们仍然生成有毒或有偏见的输出、编造事实、并在没有明确提示的情况下生成性和暴力内容。它们也可能在某些输入上无法生成合理的输出；我们在图9中展示了一些这样的例子。也许我们的模型最大的局限性是，在大多数情况下，它们遵循用户的指令，即使这可能导致现实世界中的伤害。例如，当给出指示模型最大程度偏见的提示时，InstructGPT生成的有毒输出比同等规模的GPT-3模型更多。我们在以下部分讨论潜在的缓解措施。

### 5.4 开放问题

这项工作是朝着使用对齐技术微调语言模型以遵循广泛指令迈出的第一步。还有许多开放问题需要探索，以进一步使语言模型的行为与人们实际希望它们做的相一致。

可以尝试许多方法来进一步降低模型生成有毒、有偏见或其他有害输出的倾向。例如，可以使用对抗性设置，让标注人员找到模型的最坏情况行为，然后标记并添加到数据集中（Dinan et al., 2019b）。也可以将我们的方法与过滤预训练数据的方法相结合（Ngo et al., 2021），无论是用于训练初始的预训练模型，还是用于我们预训练混合方法的数据。类似地，可以将我们的方法与提高模型真实性的方法相结合，例如WebGPT（Nakano et al., 2021）。

在这项工作中，如果用户请求潜在有害或不诚实的响应，我们允许我们的模型生成这些输出。训练我们的模型尽管有用户指令但仍保持无害是重要的，但也很难，因为输出是否有害取决于其部署的上下文；例如，作为数据增强管道的一部分，使用语言模型生成有毒输出可能是有益的。我们的技术也可以应用于使模型拒绝某些用户指令，我们计划在后续迭代中探索这一点。

让模型做我们想做的事直接与可操纵性和可控性文献相关（Dathathri et al., 2019; Krause et al., 2020）。一个有前途的未来方向是将RLHF与其他可操纵性方法相结合，例如使用控制代码（Keskar et al., 2019），或在推理时使用较小的模型修改采样过程（Dathathri et al., 2019）。

虽然我们主要关注RLHF，但还有许多其他算法可用于在我们的演示和比较数据上训练策略，以获得更好的结果。例如，可以探索专家迭代（Anthony et al., 2017; Silver et al., 2017），或使用比较数据子集的更简单的行为克隆方法。也可以尝试约束优化方法（Achiam et al., 2017），在生成少量有害行为的条件下最大化来自奖励模型的分数。

比较也不一定是提供对齐信号的最有效方式。例如，我们可以让标注人员编辑模型响应以使其更好，或用自然语言生成对模型响应的批评。在设计标注人员向语言模型提供反馈的界面方面也有巨大的选择空间；这是一个有趣的人机交互问题。

我们通过将预训练数据纳入RLHF微调来减轻对齐税的方案并不能完全消除性能回归，并可能使某些任务的某些不良行为更可能出现（如果这些行为存在于预训练数据中）。这是一个有趣的研究领域。另一个可能改进我们方法的修改是过滤预训练混合数据中的有毒内容（Ngo et al., 2021），或用合成指令增强这些数据。

正如Gabriel（2020）详细讨论的那样，对齐指令、意图、揭示的偏好、理想偏好、兴趣和价值观之间存在微妙的差异。Gabriel（2020）倡导一种基于原则的对齐方法：换句话说，识别"尽管人们的道德信仰存在广泛差异，但能获得反思性认可的公平对齐原则。"在本文中，我们为简单起见与推断的用户意图对齐，但在这个领域需要更多研究。确实，最大的开放问题之一是如何设计一个透明、有意义地代表受技术影响的人群、并以在众多群体中达成广泛共识的方式综合人们价值观的对齐过程。我们在第5.2节中讨论了一些相关的考虑。

### 5.5 更广泛的影响

这项工作的动机是我们旨在通过训练大型语言模型做一组特定人类希望它们做的事情来增加它们的积极影响。默认情况下，语言模型优化下一个词预测目标，这只是我们希望这些模型做的事情的一个代理。我们的结果表明，我们的技术有望使语言模型更加有帮助、真实和无害。在长期，对齐失败可能导致更严重的后果，特别是如果这些模型部署在安全关键情况下。我们期望随着模型规模的持续扩大，必须更加小心地确保它们与人类意图对齐（Bostrom, 2014）。

然而，使语言模型更好地遵循用户意图也使它们更容易被滥用。可能更容易使用这些模型生成令人信服的错误信息，或仇恨或辱骂内容。

对齐技术不是解决与大型语言模型相关的安全问题的万能药；相反，它们应该被用作更广泛安全生态系统中的一个工具。除了有意滥用外，还有许多领域应该非常谨慎地部署大型语言模型，或者根本不部署。例子包括高风险领域，如医疗诊断、基于受保护特征对人群进行分类、确定信用、就业或住房资格、生成政治广告以及执法。如果这些模型开源，在没有适当监管的情况下限制在这些及其他领域的有害应用将变得具有挑战性。另一方面，如果大型语言模型的访问仅限于拥有训练所需资源的少数组织，这将将大多数人排除在尖端ML技术之外。另一种选择是让组织拥有模型部署的端到端基础设施，并通过API使其可访问。这允许实施安全协议，如用例限制（仅允许模型用于某些应用）、监控滥用并撤销滥用系统者的访问权限，以及速率限制以防止大规模错误信息的生成。然而，这可能以降低透明度和增加权力集中为代价，因为它要求API提供者在这些问题的界限划定上做出决策。

最后，如第5.2节所述，这些模型与谁对齐的问题极为重要，并将显著影响这些模型的净影响是正面还是负面的。

## 致谢

首先，我们要感谢Lilian Weng、Jason Kwon、Boris Power、Che Chang、Josh Achiam、Steven Adler、Gretchen Krueger、Miles Brundage、Tyna Eloundou、Gillian Hadfield、Irene Soliaman、Christy Dennison、Daniel Ziegler、William Saunders、Beth Barnes、Cathy Yeh、Nick Cammaratta、Jonathan Ward、Matt Knight、Pranav Shyam、Alec Radford以及OpenAI的其他人，感谢他们在项目过程中的讨论，这些讨论帮助我们塑造了研究方向。我们感谢Brian Green、Irina Raicu、Subbu Vincent、Varoon Mathur、Kate Crawford、Su Lin Blodgett、Bertie Vidgen和Paul Röttger的讨论和反馈。最后，我们感谢Sam Bowman、Matthew Rahtz、Ben Mann、Liam Fedus、Helen Ngo、Josh Achiam、Leo Gao、Jared Kaplan、Cathy Yeh、Miles Brundage、Gillian Hadfield、Cooper Raterink、Gretchen Krueger、Tyna Eloundou、Rafal Jakubanis和Steven Adler对本文的反馈。我们还要感谢Owain Evans和Stephanie Lin指出自动TruthfulQA指标高估了我们PPO模型的收益。

感谢以各种方式为训练和部署我们模型的基础设施做出贡献的人，包括：Daniel Ziegler、William Saunders、Brooke Chan、Dave Cummings、Chris Hesse、Shantanu Jain、Michael Petrov、Greg Brockman、Felipe Such、Alethea Power以及整个OpenAI超算团队。我们还要感谢Suchir Balaji在重新校准方面的帮助，Alper Ercetin和Justin Wang设计本文的主要图表，以及OpenAI Comms团队协助发布，包括：Steve Dowling、Hannah Wong、Natalie Summers和Elie Georges。

最后，我们要感谢我们的标注人员，没有他们这项工作就不可能完成：Meave Fryer, Sara Tirmizi, James Carroll, Jian Ouyang, Michelle Brothers, Conor Agnew, Joe Kwon, John Morton, Emma Duncan, Delia Randolph, Kaylee Weeks, Alexej Savreux, Siam Ahsan, Rashed Sorwar, Atresha Singh, Muhaiminul Rukshat, Caroline Oliveira, Juan Pablo Castaño Rendón, Atqiya Abida Anjum, Tinashe Mapolisa, Celeste Fejzo, Caio Oleskovicz, Salahuddin Ahmed, Elena Green, Ben Harmelin, Vladan Djordjevic, Victoria Ebbets, Melissa Mejia, Emill Jayson Caypuno, Rachelle Froyalde, Russell M. Bernandez, Jennifer Brillo, Jacob Bryan, Carla Rodriguez, Evgeniya Rabinovich, Morris Stuttard, Rachelle Froyalde, Roxanne Addison, Sarah Nogly, Chait Singh。

## 参考文献

[1] Abramson, J., Ahuja, A., Barr, I., Brussee, A., Carnevale, F., Cassin, M., Chhaparia, R., Clark, S., Damoc, B., Dudzik, A., et al. (2020). Imitating interactive intelligence. arXiv preprint arXiv:2012.05672.
[2] Achiam, J., Held, D., Tamar, A., and Abbeel, P. (2017). Constrained policy optimization. In International Conference on Machine Learning, pages 22–31. PMLR.
[3] Anthony, T., Tian, Z., and Barber, D. (2017). Thinking fast and slow with deep learning and tree search. arXiv preprint arXiv:1705.08439.
[4] Aribandi, V., Tay, Y., Schuster, T., Rao, J., Zheng, H. S., Mehta, S. V., Zhuang, H., Tran, V. Q., Bahri, D., Ni, J., et al. (2021). Ext5: Towards extreme multi-task scaling for transfer learning. arXiv preprint arXiv:2111.10952.
[5] Askell, A., Bai, Y., Chen, A., Drain, D., Ganguli, D., Henighan, T., Jones, A., Joseph, N., Mann, B., DasSarma, N., et al. (2021). A general language assistant as a laboratory for alignment. arXiv preprint arXiv:2112.00861.
[6] Bahdanau, D., Brakel, P., Xu, K., Goyal, A., Lowe, R., Pineau, J., Courville, A., and Bengio, Y. (2016). An actor-critic algorithm for sequence prediction. arXiv preprint arXiv:1607.07086.
[7] Bahdanau, D., Hill, F., Leike, J., Hughes, E., Hosseini, A., Kohli, P., and Grefenstette, E. (2018). Learning to understand goal specifications by modelling reward. arXiv preprint arXiv:1806.01946.
[8] Bender, E. M., Gebru, T., McMillan-Major, A., and Shmitchell, S. (2021). On the dangers of stochastic parrots: Can language models be too big? In Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency, pages 610–623.
[9] Blodgett, S. L., Barocas, S., Daumé III, H., and Wallach, H. (2020). Language (technology) is power: A critical survey of" bias" in nlp. arXiv preprint arXiv:2005.14050.
[10] Böhm, F., Gao, Y., Meyer, C. M., Shapira, O., Dagan, I., and Gurevych, I. (2019). Better rewards yield better summaries: Learning to summarise without references. arXiv preprint arXiv:1909.01214.
[11] Bojar, O., Chatterjee, R., Federmann, C., Haddow, B., Huck, M., Hokamp, C., Koehn, P., Logacheva, V., Monz, C., Negri, M., Post, M., Scarton, C., Specia, L., and Turchi, M. (2015). Findings of the 2015 workshop on statistical machine translation. In Proceedings of the Tenth Workshop on Statistical Machine Translation, pages 1–46, Lisbon, Portugal. Association for Computational Linguistics.
[12] Bommasani, R., Hudson, D. A., Adeli, E., Altman, R., Arora, S., von Arx, S., Bernstein, M. S., Bohg, J., Bosselut, A., Brunskill, E., et al. (2021). On the opportunities and risks of foundation models. arXiv preprint arXiv:2108.07258.
[13] Bostrom, N. (2014). Superintelligence. Dunod.
[14] Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al. (2020). Language models are few-shot learners. arXiv preprint arXiv:2005.14165.
[15] Buchanan, B., Lohn, A., Musser, M., and Sedova, K. (2021). Truth, lies, and automation. Technical report, Center for the Study of Emerging Technology.
[16] Caliskan, A., Bryson, J. J., and Narayanan, A. (2017). Semantics derived automatically from language corpora contain human-like biases. Science, 356(6334):183–186.
[17] Carlini, N., Tramer, F., Wallace, E., Jagielski, M., Herbert-Voss, A., Lee, K., Roberts, A., Brown, T., Song, D., Erlingsson, U., et al. (2021). Extracting training data from large language models. In 30th USENIX Security Symposium (USENIX Security 21), pages 2633–2650.
[18] Chen, M., Tworek, J., Jun, H., Yuan, Q., Pinto, H. P. d. O., Kaplan, J., Edwards, H., Burda, Y., Joseph, N., Brockman, G., et al. (2021). Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374.
[19] Cho, W. S., Zhang, P., Zhang, Y., Li, X., Galley, M., Brockett, C., Wang, M., and Gao, J. (2018). Towards coherent and cohesive long-form text generation. arXiv preprint arXiv:1811.00511.
[20] Choi, E., He, H., Iyyer, M., Yatskar, M., Yih, W.-t., Choi, Y., Liang, P., and Zettlemoyer, L. (2018). Quac: Question answering in context. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 2174–2184.
[21] Christiano, P., Cotra, A., and Xu, M. (2021). Eliciting latent knowledge: How to tell if your eyes deceive you. https://www.alignmentforum.org/posts/qHCDysDnvhteW7kRd/arc-s-first-technical-report-eliciting-latent-knowledge.
[22] Christiano, P., Shlegeris, B., and Amodei, D. (2018). Supervising strong learners by amplifying weak experts. arXiv preprint arXiv:1810.08575.
[23] Christiano, P. F., Leike, J., Brown, T., Martic, M., Legg, S., and Amodei, D. (2017). Deep reinforcement learning from human preferences. In Advances in Neural Information Processing Systems, pages 4299–4307.
[24] Dathathri, S., Madotto, A., Lan, J., Hung, J., Frank, E., Molino, P., Yosinski, J., and Liu, R. (2019). Plug and play language models: A simple approach to controlled text generation. arXiv preprint arXiv:1912.02164.
[25] Dhamala, J., Sun, T., Kumar, V., Krishna, S., Pruksachatkun, Y., Chang, K.-W., and Gupta, R. (2021). Bold: Dataset and metrics for measuring biases in open-ended language generation. In Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency, pages 862–872.
[26] Dinan, E., Fan, A., Williams, A., Urbanek, J., Kiela, D., and Weston, J. (2019a). Queens are powerful too: Mitigating gender bias in dialogue generation. arXiv preprint arXiv:1911.03842.
[27] Dinan, E., Humeau, S., Chintagunta, B., and Weston, J. (2019b). Build it break it fix it for dialogue safety: Robustness from adversarial human attack. arXiv preprint arXiv:1908.06083.
[28] Dua, D., Wang, Y., Dasigi, P., Stanovsky, G., Singh, S., and Gardner, M. (2019). Drop: A reading comprehension benchmark requiring discrete reasoning over paragraphs. arXiv preprint arXiv:1903.00161.
[29] Fedus, W., Zoph, B., and Shazeer, N. (2021). Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. arXiv preprint arXiv:2101.03961.
[30] Gabriel, I. (2020). Artificial intelligence, values, and alignment. Minds and machines, 30(3):411–437.
[31] Gehman, S., Gururangan, S., Sap, M., Choi, Y., and Smith, N. A. (2020). Realtoxicityprompts: Evaluating neural toxic degeneration in language models. arXiv preprint arXiv:2009.11462.
[32] Hancock, B., Bordes, A., Mazare, P.-E., and Weston, J. (2019). Learning from dialogue after deployment: Feed yourself, chatbot! arXiv preprint arXiv:1901.05415.
[33] Henderson, P., Sinha, K., Angelard-Gontier, N., Ke, N. R., Fried, G., Lowe, R., and Pineau, J. (2018). Ethical challenges in data-driven dialogue systems. In Proceedings of the 2018 AAAI/ACM Conference on AI, Ethics, and Society, pages 123–129.
[34] Huang, P.-S., Zhang, H., Jiang, R., Stanforth, R., Welbl, J., Rae, J., Maini, V., Yogatama, D., and Kohli, P. (2019). Reducing sentiment bias in language models via counterfactual evaluation. arXiv preprint arXiv:1911.03064.
[35] Ibarz, B., Leike, J., Pohlen, T., Irving, G., Legg, S., and Amodei, D. (2018). Reward learning from human preferences and demonstrations in atari. In Advances in neural information processing systems, pages 8011–8023.
[36] Irving, G., Christiano, P., and Amodei, D. (2018). AI safety via debate. arXiv preprint arXiv:1805.00899.
[37] Jaques, N., Ghandeharioun, A., Shen, J. H., Ferguson, C., Lapedriza, A., Jones, N., Gu, S., and Picard, R. (2019). Way off-policy batch deep reinforcement learning of implicit human preferences in dialog. arXiv preprint arXiv:1907.00456.
[38] Kenton, Z., Everitt, T., Weidinger, L., Gabriel, I., Mikulik, V., and Irving, G. (2021). Alignment of language agents. arXiv preprint arXiv:2103.14659.
[39] Keskar, N. S., McCann, B., Varshney, L. R., Xiong, C., and Socher, R. (2019). Ctrl: A conditional transformer language model for controllable generation. arXiv preprint arXiv:1909.05858.
[40] Khashabi, D., Min, S., Khot, T., Sabharwal, A., Tafjord, O., Clark, P., and Hajishirzi, H. (2020). Unifiedqa: Crossing format boundaries with a single qa system. arXiv preprint arXiv:2005.00700.
[41] Kirk, H., Jun, Y., Iqbal, H., Benussi, E., Volpin, F., Dreyer, F. A., Shtedritski, A., and Asano, Y. M. (2021). How true is gpt-2? an empirical analysis of intersectional occupational biases. arXiv preprint arXiv:2102.04130.
[42] Krause, B., Gotmare, A. D., McCann, B., Keskar, N. S., Joty, S., Socher, R., and Rajani, N. F. (2020). Gedi: Generative discriminator guided sequence generation. arXiv preprint arXiv:2009.06367.
[43] Kreutzer, J., Khadivi, S., Matusov, E., and Riezler, S. (2018). Can neural machine translation be improved with user feedback? arXiv preprint arXiv:1804.05958.
[44] Lawrence, C. and Riezler, S. (2018). Improving a neural semantic parser by counterfactual learning from human bandit feedback. arXiv preprint arXiv:1805.01252.
[45] Leike, J., Krueger, D., Everitt, T., Martic, M., Maini, V., and Legg, S. (2018). Scalable agent alignment via reward modeling: a research direction. arXiv preprint arXiv:1811.07871.
[46] Leike, J., Martic, M., Krakovna, V., Ortega, P. A., Everitt, T., Lefrancq, A., Orseau, L., and Legg, S. (2017). AI safety gridworlds. arXiv preprint arXiv:1711.09883.
[47] Liang, P. P., Wu, C., Morency, L.-P., and Salakhutdinov, R. (2021). Towards understanding and mitigating social biases in language models. In International Conference on Machine Learning, pages 6565–6576. PMLR.
[48] Lin, S., Hilton, J., and Evans, O. (2021). Truthfulqa: Measuring how models mimic human falsehoods. arXiv preprint arXiv:2109.07958.
[49] Liu, H., Dacon, J., Fan, W., Liu, H., Liu, Z., and Tang, J. (2019). Does gender matter? towards fairness in dialogue systems. arXiv preprint arXiv:1910.10486.
[50] Madaan, A., Tandon, N., Clark, P., and Yang, Y. (2022). Memory-assisted prompt editing to improve gpt-3 after deployment. arXiv preprint arXiv:2201.06009.
[51] Manela, D. d. V., Errington, D., Fisher, T., van Breugel, B., and Minervini, P. (2021). Stereotype and skew: Quantifying gender bias in pre-trained and fine-tuned language models. arXiv preprint arXiv:2101.09688.
[52] Mishra, S., Khashabi, D., Baral, C., and Hajishirzi, H. (2021). Cross-task generalization via natural language crowdsourcing instructions. arXiv preprint arXiv:2104.08773.
[53] Nadeem, M., Bethke, A., and Reddy, S. (2020). Stereoset: Measuring stereotypical bias in pretrained language models. arXiv preprint arXiv:2004.09456.
[54] Nahian, M. S. A., Frazier, S., Harrison, B., and Riedl, M. (2021). Training value-aligned reinforcement learning agents using a normative prior. arXiv preprint arXiv:2104.09469.
[55] Nakano, R., Hilton, J., Balaji, S., Wu, J., Ouyang, L., Kim, C., Hesse, C., Jain, S., Kosaraju, V., Saunders, W., et al. (2021). Webgpt: Browser-assisted question-answering with human feedback. arXiv preprint arXiv:2112.09332.
[56] Nallapati, R., Zhou, B., Gulcehre, C., Xiang, B., et al. (2016). Abstractive text summarization using sequence-to-sequence rnns and beyond. arXiv preprint arXiv:1602.06023.
[57] Nangia, N., Vania, C., Bhalerao, R., and Bowman, S. R. (2020). CrowS-Pairs: A Challenge Dataset for Measuring Social Biases in Masked Language Models. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing, Online. Association for Computational Linguistics.
[58] Ngo, H., Raterink, C., Araújo, J. G., Zhang, I., Chen, C., Morisot, A., and Frosst, N. (2021). Mitigating harm in language models with conditional-likelihood filtration. arXiv preprint arXiv:2108.07790.
[59] Perez, E., Karamcheti, S., Fergus, R., Weston, J., Kiela, D., and Cho, K. (2019). Finding generalizable evidence by learning to convince q&a models. arXiv preprint arXiv:1909.05863.
[60] Qian, Y., Muaz, U., Zhang, B., and Hyun, J. W. (2019). Reducing gender bias in word-level language models with a gender-equalizing loss function. arXiv preprint arXiv:1905.12801.
[61] Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., and Sutskever, I. (2019). Language models are unsupervised multitask learners. OpenAI Blog, 1(8):9.
[62] Rae, J. W., Borgeaud, S., Cai, T., Millican, K., Hoffmann, J., Song, F., Aslanides, J., Henderson, S., Ring, R., Young, S., et al. (2021). Scaling language models: Methods, analysis & insights from training gopher. arXiv preprint arXiv:2112.11446.
[63] Rajpurkar, P., Jia, R., and Liang, P. (2018). Know what you don't know: Unanswerable questions for squad. arXiv preprint arXiv:1806.03822.
[64] Rudinger, R., Naradowsky, J., Leonard, B., and Van Durme, B. (2018). Gender bias in coreference resolution. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, New Orleans, Louisiana. Association for Computational Linguistics.
[65] Sanh, V., Webson, A., Raffel, C., Bach, S. H., Sutawika, L., Alyafeai, Z., Chaffin, A., Stiegler, A., Scao, T. L., Raja, A., et al. (2021). Multitask prompted training enables zero-shot task generalization. arXiv preprint arXiv:2110.08207.
[66] Schick, T., Udupa, S., and Schütze, H. (2021). Self-diagnosis and self-debiasing: A proposal for reducing corpus-based bias in nlp. arXiv preprint arXiv:2103.00453.
[67] Schulman, J., Moritz, P., Levine, S., Jordan, M., and Abbeel, P. (2016). High-dimensional continuous control using generalized advantage estimation. In Proceedings of the International Conference on Learning Representations (ICLR).
[68] Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O. (2017). Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347.
[69] Sheng, E., Chang, K.-W., Natarajan, P., and Peng, N. (2019). The woman worked as a babysitter: On biases in language generation. arXiv preprint arXiv:1909.01326.
[70] Silver, D., Hubert, T., Schrittwieser, J., Antonoglou, I., Lai, M., Guez, A., Lanctot, M., Sifre, L., Kumaran, D., Graepel, T., et al. (2017). Mastering chess and shogi by self-play with a general reinforcement learning algorithm. arXiv preprint arXiv:1712.01815.
[71] Soares, N., Fallenstein, B., Armstrong, S., and Yudkowsky, E. (2015). Corrigibility. In Workshops at the Twenty-Ninth AAAI Conference on Artificial Intelligence.
[72] Socher, R., Perelygin, A., Wu, J., Chuang, J., Manning, C. D., Ng, A. Y., and Potts, C. (2013). Recursive deep models for semantic compositionality over a sentiment treebank. In Proceedings of the 2013 conference on empirical methods in natural language processing, pages 1631–1642.
[73] Solaiman, I., Brundage, M., Clark, J., Askell, A., Herbert-Voss, A., Wu, J., Radford, A., Krueger, G., Kim, J. W., Kreps, S., et al. (2019). Release strategies and the social impacts of language models. arXiv preprint arXiv:1908.09203.
[74] Solaiman, I. and Dennison, C. (2021). Process for adapting language models to society (palms) with values-targeted datasets. arXiv preprint arXiv:2106.10328.
[75] Stiennon, N., Ouyang, L., Wu, J., Ziegler, D. M., Lowe, R., Voss, C., Radford, A., Amodei, D., and Christiano, P. (2020). Learning to summarize from human feedback. arXiv preprint arXiv:2009.01325.
[76] Tamkin, A., Brundage, M., Clark, J., and Ganguli, D. (2021). Understanding the capabilities, limitations, and societal impact of large language models. arXiv preprint arXiv:2102.02503.
[77] Thoppilan, R., De Freitas, D., Hall, J., Shazeer, N., Kulshreshtha, A., Cheng, H.-T., Jin, A., Bos, T., Baker, L., Du, Y., et al. (2022). Lamda: Language models for dialog applications. arXiv preprint arXiv:2201.08239.
[78] Vig, J., Gehrmann, S., Belinkov, Y., Qian, S., Nevo, D., Singer, Y., and Shieber, S. M. (2020). Investigating gender bias in language models using causal mediation analysis. In NeurIPS.
[79] Völske, M., Potthast, M., Syed, S., and Stein, B. (2017). Tl; dr: Mining reddit to learn automatic summarization. In Proceedings of the Workshop on New Frontiers in Summarization, pages 59–63.
[80] Wang, A., Pruksachatkun, Y., Nangia, N., Singh, A., Michael, J., Hill, F., Levy, O., and Bowman, S. R. (2019). Superglue: A stickier benchmark for general-purpose language understanding systems. arXiv preprint arXiv:1905.00537.
[81] Wei, J., Bosma, M., Zhao, V. Y., Guu, K., Yu, A. W., Lester, B., Du, N., Dai, A. M., and Le, Q. V. (2021). Finetuned language models are zero-shot learners. arXiv preprint arXiv:2109.01652.
[82] Weidinger, L., Mellor, J., Rauh, M., Griffin, C., Uesato, J., Huang, P.-S., Cheng, M., Glaese, M., Balle, B., Kasirzadeh, A., et al. (2021). Ethical and social risks of harm from language models. arXiv preprint arXiv:2112.04359.
[83] Welbl, J., Glaese, A., Uesato, J., Dathathri, S., Mellor, J., Hendricks, L. A., Anderson, K., Kohli, P., Coppin, B., and Huang, P.-S. (2021). Challenges in detoxifying language models. arXiv preprint arXiv:2109.07445.
[84] Wu, J., Ouyang, L., Ziegler, D. M., Stiennon, N., Lowe, R., Leike, J., and Christiano, P. (2021). Recursively summarizing books with human feedback. arXiv preprint arXiv:2109.10862.
[85] Xu, A., Pathak, E., Wallace, E., Gururangan, S., Sap, M., and Klein, D. (2021). Detoxifying language models risks marginalizing minority voices. arXiv preprint arXiv:2104.06390.
[86] Xu, J., Ju, D., Li, M., Boureau, Y.-L., Weston, J., and Dinan, E. (2020). Recipes for safety in open-domain chatbots. arXiv preprint arXiv:2010.07079.
[87] Yi, S., Goel, R., Khatri, C., Cervone, A., Chung, T., Hedayatnia, B., Venkatesh, A., Gabriel, R., and Hakkani-Tur, D. (2019). Towards coherent and engaging spoken dialog response generation using automatic conversation evaluators. arXiv preprint arXiv:1904.13015.
[88] Zellers, R., Holtzman, A., Bisk, Y., Farhadi, A., and Choi, Y. (2019). Hellaswag: Can a machine really finish your sentence? In Association for Computational Linguistics, pages 4791–4800.
[89] Zhao, M., Anderson, P., Jain, V., Wang, S., Ku, A., Baldridge, J., and Ie, E. (2021). On the evaluation of vision-and-language navigation instructions. arXiv preprint arXiv:2101.10504.
[90] Zhou, W. and Xu, K. (2020). Learning to compare for better training and evaluation of open domain natural language generation models. arXiv preprint arXiv:2002.05058.
[91] Ziegler, D. M., Stiennon, N., Wu, J., Brown, T. B., Radford, A., Amodei, D., Christiano, P., and Irving, G. (2019). Fine-tuning language models from human preferences. arXiv preprint arXiv:1909.08593.

---

> 本翻译由机器翻译生成，仅供参考。如需引用，请参阅原始英文论文：arXiv:2203.02155。