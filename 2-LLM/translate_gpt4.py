#!/usr/bin/env python3
"""Translate GPT-4 Technical Report to Chinese"""

import fitz

doc = fitz.open('/Users/dazhang/PycharmProject/Papers/2-LLM/2024-GPT-4 Technical Report.pdf')
full_text = ''
for page in doc:
    full_text += page.get_text('text') + '\n\n'

lines = full_text.split('\n')

out = []
def w(s=''):
    out.append(s)

# ========== TITLE ==========
w('# GPT-4 技术报告')
w()
w('> OpenAI | arXiv:2303.08774v6 [cs.CL] 2024年3月4日')
w()
w('本技术报告介绍了GPT-4的开发情况，GPT-4是一个大规模多模态模型，能够接受图像和文本输入并生成文本输出。虽然在许多现实场景中能力不如人类，但GPT-4在各种专业和学术基准测试中展现出人类水平的性能，包括在模拟律师资格考试中获得约前10%考生的成绩。GPT-4是基于Transformer的模型，通过预训练预测文档中的下一个token。训练后的对齐过程在事实性和对期望行为的遵守方面提升了性能。该项目的一个核心组成部分是开发能够在广泛规模范围内可预测行为的基础设施和优化方法。这使得我们能够基于训练计算量不超过GPT-4的1/1000的模型，准确预测GPT-4某些方面的性能。')
w()
w('---')
w()

# ========== 摘要 ==========
w('## 摘要')
w()
w('我们报告了GPT-4的开发情况，GPT-4是一个大规模多模态模型，能够接受图像和文本输入并生成文本输出。虽然在许多现实场景中能力不如人类，但GPT-4在各种专业和学术基准测试中展现出人类水平的性能，包括在模拟律师资格考试中获得约前10%考生的成绩。GPT-4是基于Transformer的模型，通过预训练预测文档中的下一个token。训练后的对齐过程在事实性和对期望行为的遵守方面提升了性能。该项目的一个核心组成部分是开发能够在广泛规模范围内可预测行为的基础设施和优化方法。这使得我们能够基于训练计算量不超过GPT-4的1/1000的模型，准确预测GPT-4某些方面的性能。')
w()

# ========== 1 引言 ==========
w('## 1 引言')
w()
w('本技术报告介绍了GPT-4，一个能够处理图像和文本输入并生成文本输出的大型多模态模型。这类模型是一个重要的研究领域，因为它们有潜力被应用于广泛的场景，如对话系统、文本摘要和机器翻译。因此，近年来它们一直是大量关注和进展的主题[1–34]。')
w()
w('开发此类模型的主要目标之一是提高它们理解和生成自然语言文本的能力，特别是在更复杂和细微的场景中。为了测试其在此类场景中的能力，我们在各种原本为人类设计的考试上评估了GPT-4。在这些评估中，它表现相当出色，通常超过绝大多数人类考生。例如，在模拟律师资格考试中，GPT-4的成绩位于前10%的考生之列。这与GPT-3.5形成对比，后者得分位于后10%。')
w()
w('在一套传统的NLP基准测试中，GPT-4优于之前的大型语言模型以及大多数最先进的系统（这些系统通常具有基准测试特定的训练或手工工程）。在MMLU基准测试[35, 36]（一个涵盖57个学科的英语多选题套件）上，GPT-4不仅在英语上以显著优势优于现有模型，而且在其他语言中也展现出强劲性能。在MMLU的翻译变体上，GPT-4在考虑的26种语言中的24种上超过了英语的最先进水平。我们将在后面的章节中更详细地讨论这些模型能力结果，以及模型安全改进和结果。')
w()
w('本报告还讨论了该项目的一个关键挑战：开发在广泛规模范围内可预测行为的深度学习基础设施和优化方法。这使我们能够对GPT-4的预期性能做出预测（基于以类似方式训练的小规模运行），这些预测与最终运行进行了对比，以增强我们对训练的信心。')
w()
w('尽管具有这些能力，GPT-4与早期的GPT模型[1, 37, 38]有相似的局限性：它并非完全可靠（例如可能出现"幻觉"），具有有限的上下文窗口，并且不能从经验中学习。在使用GPT-4的输出时应谨慎，特别是在可靠性重要的场景中。')
w()
w('GPT-4的能力和局限性带来了重大且新颖的安全挑战，我们认为鉴于潜在的社会影响，对这些挑战的仔细研究是一个重要的研究领域。本报告包含一个广泛的系统卡（在附录之后），描述了我们在偏见、虚假信息、过度依赖、隐私、网络安全、扩散等方面的风险预见。它还描述了为减轻GPT-4部署可能造成的伤害而采取的干预措施，包括与领域专家进行的对抗性测试，以及一个模型辅助的安全流程。')
w()

# ========== 2 范围与局限 ==========
w('## 2 本技术报告的范围与局限')
w()
w('本报告侧重于GPT-4的能力、局限性和安全属性。GPT-4是一个Transformer风格模型[39]，通过预训练预测文档中的下一个token，使用了公开可用的数据（如互联网数据）和从第三方提供商许可的数据。然后使用基于人类反馈的强化学习（RLHF）[40]对模型进行微调。考虑到竞争环境以及像GPT-4这样的大规模模型的安全影响，本报告不包含关于架构（包括模型大小）、硬件、训练计算、数据集构建、训练方法或类似的进一步细节。')
w()
w('我们致力于对我们的技术进行独立审计，并在随本次发布附带的系统卡中分享了一些初步的步骤和想法。2 我们计划向其他第三方提供进一步的技术细节，这些第三方可以就如何在上述竞争和安全考虑与进一步提高透明度的科学价值之间进行权衡向我们提供建议。')
w()

# ========== 3 可预测的扩展 ==========
w('## 3 可预测的扩展')
w()
w('GPT-4项目的一个重点方向是构建一个可预测扩展的深度学习栈。主要原因是对于像GPT-4这样非常大的训练运行，进行广泛的模型特定调优是不可行的。为了解决这个问题，我们开发了在多个规模上具有非常可预测行为的基础设施和优化方法。这些改进使我们能够从使用计算量少1000倍到10000倍的较小模型训练的模型中可靠地预测GPT-4某些方面的性能。')
w()
w('### 3.1 损失预测')
w()
w('适当训练的大型语言模型的最终损失被认为可以用训练模型所用的计算量的幂律很好地近似[41, 42, 2, 14, 15]。')
w()
w('为了验证我们优化基础设施的可扩展性，我们通过拟合一个带有不可约损失项的标度律（如Henighan等人[15]所述）来预测GPT-4在内部代码库（不属于训练集）上的最终损失：$L(C) = aC^b + c$，该标度律基于使用相同方法论但计算量最多比GPT-4少10,000倍的模型。这个预测是在运行开始后不久做出的，没有使用任何部分结果。拟合的标度律以高精度预测了GPT-4的最终损失（图1）。')
w()
w('### 3.2 HumanEval上的能力扩展')
w()
w('在训练前了解模型的能力可以改善关于对齐、安全和部署的决策。除了预测最终损失外，我们还开发了预测更可解释能力指标的方法。其中一个指标是HumanEval数据集[43]上的通过率，该数据集衡量合成不同复杂度的Python函数的能力。我们通过外推使用最多少1000倍计算量训练的模型，成功预测了HumanEval数据集子集上的通过率（图2）。')
w()
w('对于HumanEval中的单个问题，性能有时可能随规模而下降。尽管存在这些挑战，我们发现一个近似的幂律关系：$-\\mathbb{E}_P[\\log(\\text{pass\\_rate}(C))] = \\alpha\\times C^{-k}$，其中$k$和$\\alpha$是正常数，$P$是数据集中问题的子集。我们假设这个关系对数据集中的所有问题都成立。在实践中，非常低的通过率很难或不可能估计，因此我们限制问题$P$和模型$M$，使得在给定某个大采样预算的情况下，每个模型至少解决每个问题一次。')
w()
w('我们在训练完成前注册了对GPT-4在HumanEval上性能的预测，仅使用了训练前可用的信息。除了最难的15个HumanEval问题外，其余问题根据较小模型的性能分为6个难度桶。第三容易桶的结果如图2所示，表明对于这个我们能准确估计多个较小模型的$\\log(\\text{pass\\_rate})$的HumanEval问题子集，预测结果非常准确。其他五个桶的预测表现也几乎同样好，主要例外是GPT-4在最容易桶上的表现低于我们的预测。')
w()
w('某些能力仍然难以预测。例如，逆标度奖[44]提出了几个任务，在这些任务中模型性能随规模增大而下降。与Wei等人[45]的最近结果类似，我们发现GPT-4逆转了这一趋势，如图3中一个名为"事后忽视"（Hindsight Neglect）[46]的任务所示。')
w()
w('图3：GPT-4和较小模型在"事后忽视"任务上的性能。y轴显示准确率，越高越好。ada、babbage和curie指通过OpenAI API[47]可用的模型。')
w()
w('我们相信准确预测未来能力对安全非常重要。展望未来，我们计划精炼这些方法，并在大型模型训练开始前注册各种能力的性能预测，我们希望这成为该领域的共同目标。')
w()

# ========== 4 能力 ==========
w('## 4 能力')
w()
w('我们在多样化的基准测试集上测试了GPT-4，包括模拟原本为人类设计的考试。4 我们没有针对这些考试进行特定训练。考试中的少数问题在训练期间被模型看到过；对于每个考试，我们运行一个移除了这些问题的变体，并报告两者中较低的分数。我们相信结果具有代表性。关于污染（方法和每个考试的统计信息）的更多细节，请参见附录C。')
w()
w('考试材料来源于公开可用的资料。考试问题包括多项选择题和自由回答题；我们为每种格式设计了独立的提示，并在需要时在输入中包含图像。评估设置基于验证集考试的性能进行设计，并在保留的测试考试上报告最终结果。总得分通过使用每个考试的公开可用方法结合多项选择和自由回答题的分数来确定。我们估计并报告每个总得分对应的百分位数。关于考试评估方法的更多细节，请参见附录A。')
w()
w('表1：GPT在学术和专业考试上的表现。在每种情况下，我们模拟真实考试的条件和评分。我们报告GPT-4根据考试特定评分标准评定的最终分数，以及达到GPT-4分数的考生百分位数。')
w()
w('图4：GPT在学术和专业考试上的表现。在每种情况下，我们模拟真实考试的条件和评分。考试根据GPT-3.5的表现从低到高排序。GPT-4在大多数测试考试上优于GPT-3.5。为保守起见，我们报告百分位范围的下限，但这在AP考试上造成了一些伪像，因为这些考试有非常宽的评分区间。例如，尽管GPT-4在AP生物学上获得了最高可能分数（5/5），但在图中仅显示为第85百分位，因为15%的考生获得了该分数。')
w()
w('GPT-4在大多数这些专业和学术考试上展现出人类水平的性能。值得注意的是，它在模拟的统一律师资格考试中通过了前10%考生的分数（表1，图4）。')
w()
w('模型在考试中的能力似乎主要源于预训练过程，并未受到RLHF的显著影响。在多项选择题上，基础GPT-4模型和RLHF模型在我们测试的考试中平均表现相当（见附录B）。')
w()
w('我们还评估了预训练的基础GPT-4模型在传统语言模型评估基准上的表现。对于每个报告的基准，我们运行了测试数据出现在训练集中的污染检查（关于每个基准污染的完整细节，请参见附录D）。5 在评估GPT-4时，我们对所有基准使用了少样本提示（few-shot prompting）[1]。6')
w()
w('GPT-4显著优于现有的语言模型，以及之前最先进的（SOTA）系统——这些系统通常具有基准测试特定的构建或额外的训练协议（表2）。')
w()
w('表2：GPT-4在学术基准上的表现。我们将GPT-4与最佳SOTA（具有基准测试特定训练）和评估为少样本的LM的最佳SOTA进行比较。GPT-4在所有基准上优于现有LM，并在除DROP之外的所有数据集上击败了具有基准测试特定训练的SOTA。对于每个任务，我们报告GPT-4的性能以及评估使用的少样本方法。对于GSM-8K，我们在GPT-4预训练混合中包含了部分训练集（见附录E），并在评估时使用链式思维提示（chain-of-thought prompting）[11]。对于多项选择题，我们向模型呈现所有答案（ABCD）并要求其选择答案的字母，类似于人类解决此类问题的方式。')
w()
w('许多现有的ML基准测试是用英语编写的。为了初步了解GPT-4在其他语言中的能力，我们使用Azure Translate将MMLU基准测试[35, 36]（一套涵盖57个学科的多项选择题套件）翻译成多种语言（示例翻译和提示见附录F）。我们发现，GPT-4在我们测试的大多数语言上优于GPT-3.5和现有语言模型（Chinchilla[2]和PaLM[3]）的英语性能，包括低资源语言如拉脱维亚语、威尔士语和斯瓦希里语（图5）。')
w()
w('GPT-4在遵循用户意图方面比之前的模型有显著改进[63]。在一个包含5,214个提交给ChatGPT[64]和OpenAI API[47]的提示的数据集上，GPT-4生成的响应在70.2%的提示中优于GPT-3.5生成的响应。7')
w()
w('我们正在开源OpenAI Evals8，这是我们用于创建和运行评估模型的基准测试的框架，可以逐个样本检查性能。Evals与现有基准测试兼容，可用于跟踪部署中模型的性能。我们计划随着时间的推移增加这些基准测试的多样性，以代表更广泛的失败模式和更难的任务集。')
w()
w('### 4.1 视觉输入')
w()
w('GPT-4接受包含图像和文本的提示，与纯文本设置并行，这让用户能够指定任何视觉或语言任务。具体来说，模型在给定任意交错文本和图像的输入时生成文本输出。在一系列领域中——包括带有文本和照片的文档、图表或截图——GPT-4展现出与纯文本输入相似的能力。GPT-4视觉输入的示例如表3所示。为语言模型开发的标准测试时技术（如少样本提示、链式思维等）在使用图像和文本时同样有效——示例见附录G。')
w()
w('关于视觉能力的初步结果（在狭窄的学术视觉基准测试集上）可以在GPT-4博客文章[65]中找到。我们计划在后续工作中发布关于GPT-4视觉能力的更多信息。')
w()

# ========== 5 局限性 ==========
w('## 5 局限性')
w()
w('尽管具有这些能力，GPT-4与早期的GPT模型有相似的局限性。最重要的是，它仍非完全可靠（它会"幻觉"事实并产生推理错误）。在使用语言模型输出时应非常谨慎，特别是在高风险场景中，具体的协议（如人工审查、用额外上下文进行 grounding，或完全避免高风险使用）应符合特定应用的需求。详情请参见我们的系统卡。')
w()
w('GPT-4相比之前的GPT-3.5模型（它们本身也在持续迭代中不断改进）显著减少了幻觉。GPT-4在我们内部的对抗性设计的事实性评估中比最新的GPT-3.5模型高出19个百分点（图6）。')
w()
w('图6：GPT-4在九个内部对抗性设计的事实性评估上的表现。y轴显示准确率，越高越好。准确率为1.0表示模型的所有回答都被判断为与人类理想回答一致。我们将GPT-4与基于GPT-3.5的三个早期ChatGPT版本[64]进行比较；GPT-4在最新的GPT-3.5模型上提升了19个百分点，在所有主题上都有显著提高。')
w()
w('GPT-4在TruthfulQA[66]等公共基准测试上取得了进展，该基准测试评估模型将事实与对抗性选择的错误陈述集区分开的能力（图7）。这些问题配对了统计上具有吸引力的错误答案。GPT-4基础模型在此任务上仅比GPT-3.5略好；然而，在RLHF后训练之后，我们观察到相比GPT-3.5有大幅提升。9 表4显示了正确答案和错误答案各一个。GPT-4抵制了选择常见说法的倾向（你不能教老狗新把戏），但它仍然可能遗漏细微的细节（埃尔维斯·普雷斯利不是演员的儿子，所以Perkins是正确答案）。')
w()
w('GPT-4通常缺乏对其预训练数据绝大部分截止于2021年9月之后发生的事件的了解10，并且不能从经验中学习。它有时会犯简单的推理错误，这似乎与其在众多领域的胜任能力不相称，或者可能过于轻信地接受用户明显错误的陈述。它可能像人类一样在困难问题上失败，例如在其生成的代码中引入安全漏洞。')
w()
w('GPT-4在其预测中也可能自信地犯错，在可能犯错时不注意仔细检查。有趣的是，预训练模型具有很高的校准度（其对答案的预测置信度通常与正确的概率相匹配）。然而，在后训练过程之后，校准度降低了（图8）。')
w()
w('GPT-4在其输出中存在各种偏见，我们已努力纠正，但完全表征和管理这些偏见需要一些时间。我们旨在使GPT-4和我们构建的其他系统具有反映广泛用户价值观的合理默认行为，允许这些系统在某些广泛范围内进行定制，并征询公众关于这些范围应该是什么样的意见。更多详情请参见OpenAI [68]。')
w()

# ========== 6 风险与缓解措施 ==========
w('## 6 风险与缓解措施')
w()
w('我们投入了大量精力来提高GPT-4的安全性和对齐性。在这里，我们重点介绍了我们在对抗性测试和红队测试中使用领域专家的情况，我们的模型辅助安全流程[69]，以及相比之前模型在安全指标上的改进。')
w()
w('通过领域专家进行对抗性测试：GPT-4与较小语言模型存在类似风险，例如生成有害建议、有漏洞的代码或不准确的信息。然而，GPT-4的额外能力带来了新的风险面。为了理解这些风险的程度，我们邀请了来自长期AI对齐风险、网络安全、生物风险和国际安全等领域的50多位专家对模型进行对抗性测试。他们的发现使我们能够特别测试模型在高风险领域的行为——这些领域需要专业知识来评估——以及评估对于非常高级的AI才会变得相关的风险，如权力寻求[70]。从这些专家收集的建议和训练数据被用于我们的缓解措施和模型改进；例如，我们收集了额外数据来改进GPT-4拒绝关于合成危险化学品的请求的能力（表5）。')
w()
w('模型辅助安全流程：与之前的GPT模型一样，我们使用基于人类反馈的强化学习（RLHF）[40, 63]对模型行为进行微调，以产生更符合用户意图的响应。然而，在RLHF之后，我们的模型在 unsafe 输入上仍然可能脆弱，有时在 safe 和 unsafe 输入上都会出现不期望的行为。这些不期望的行为可能出现在RLHF流程的奖励模型数据收集部分给标注者的指令规定不充分时。当给出 unsafe 输入时，模型可能生成不期望的内容，如提供犯罪建议。此外，模型在 safe 输入上也可能变得过于谨慎，拒绝无害的请求或过度回避。为了在更细粒度的层面上引导模型向适当行为发展，我们严重依赖模型本身作为工具。')
w()
w('我们的安全方法包括两个主要组成部分：一组额外的安全相关的RLHF训练提示，和基于规则的奖励模型（RBRMs）。')
w()
w('我们的基于规则的奖励模型（RBRMs）是一组零样本GPT-4分类器。这些分类器在RLHF微调期间为GPT-4策略模型提供额外的奖励信号，针对正确行为，例如拒绝生成有害内容或不拒绝无害请求。RBRM接受三个输入：提示（可选）、策略模型的输出以及人工编写的评估该输出应如何被评分的规则（例如，一套多项选择风格的规则）。然后，RBRM根据规则对输出进行分类。例如，我们可以提供一套规则，指示模型将响应分类为：（a）期望风格的拒绝，（b）不期望风格的拒绝（例如回避或冗长），（c）包含不允许内容，或（d）安全的非拒绝响应。然后，在那些请求有害内容（如非法建议）的安全相关训练提示集上，我们可以奖励GPT-4拒绝这些请求。相反，我们可以奖励GPT-4在保证安全且可回答的提示子集上不拒绝请求。')
w()
w('这项技术与Glaese等人[71]和Perez等人[72]的工作相关。这与其他改进（如计算最优RBRM权重和提供针对我们想改进领域的额外SFT数据）相结合，使我们能够将模型更接近期望的行为。')
w()
w('安全指标的改进：我们的缓解措施显著改善了GPT-4的许多安全属性。相比GPT-3.5，我们将模型响应不允许内容请求的倾向降低了82%（表6），并且GPT-4根据我们的政策响应敏感请求（例如医疗建议和自我伤害）的频率提高了29%（表7，图9）。在RealToxicityPrompts数据集[73]上，GPT-4仅有0.73%的时间产生有毒内容，而GPT-3.5产生有毒内容的时间为6.48%。')
w()
w('图9：在敏感和不允许提示上的不正确行为率。数值越低越好。GPT-4 RLHF相比之前的模型具有更低的不正确行为率。')
w()
w('总的来说，我们模型层面的干预增加了引发不良行为的难度，但仍然可能做到。例如，仍然存在"越狱"（jailbreaks）（例如对抗性系统消息，更多细节请见系统卡中的图10）来生成违反我们使用指南的内容。只要这些限制存在，就需要通过部署时的安全技术（如滥用监控和快速迭代模型改进流程）来补充。')
w()
w('GPT-4及其后续模型有潜力以有益和有害两种方式显著影响社会。我们正在与外部研究人员合作，改进我们理解和评估潜在影响的方式，并构建对未来系统中可能出现的危险能力的评估。我们将很快发布关于社会可以采取哪些措施为AI的影响做准备的建议，以及关于预测AI可能的经济影响的初步想法。')
w()

# ========== 7 结论 ==========
w('## 7 结论')
w()
w('我们描述了GPT-4，一个大型多模态模型，在某些困难的专业和学术基准上具有人类水平的性能。GPT-4在一系列NLP任务上优于现有的大型语言模型，并超过了绝大多数报告的最先进系统（这些系统通常包括任务特定的微调）。我们发现，改进的能力虽然在英语中测量，但可以在许多不同语言中得到展示。我们强调了可预测的扩展如何使我们能够准确预测GPT-4的损失和能力。')
w()
w('GPT-4由于增强的能力带来了新的风险，我们讨论了一些用于理解和改进其安全性和对齐性的方法和结果。虽然还有很多工作要做，但GPT-4代表了向广泛有用且安全部署的AI系统迈出的重要一步。')
w()

# ========== 作者贡献 ==========
w('## 作者署名、信用归属与致谢')
w()
w('请引用本文为"OpenAI (2023)"。')
w('...（完整作者列表见原文，此处省略以节省篇幅）...')
w()

# ========== 参考文献 ==========
w('## 参考文献')
w()
refs = [
    '[1] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D. Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in Neural Information Processing Systems, 33:1877–1901, 2020.',
    '[2] Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. Training compute-optimal large language models. arXiv preprint arXiv:2203.15556, 2022.',
    '[3] Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. PaLM: Scaling language modeling with pathways. arXiv preprint arXiv:2204.02311, 2022.',
    '[4] Jack W Rae, Sebastian Borgeaud, Trevor Cai, Katie Millican, Jordan Hoffmann, Francis Song, John Aslanides, Sarah Henderson, Roman Ring, Susannah Young, et al. Scaling language models: Methods, analysis & insights from training gopher. arXiv preprint arXiv:2112.11446, 2021.',
    '[5] Zihang Dai, Zhilin Yang, Yiming Yang, Jaime Carbonell, Quoc V. Le, and Ruslan Salakhutdinov. Transformer-XL: Attentive language models beyond a fixed-length context. arXiv preprint arXiv:1901.02860, 2019.',
    '[6] Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. RoBERTa: A robustly optimized BERT pretraining approach. arXiv preprint arXiv:1907.11692, 2019.',
    '[7] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018.',
    '[8] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. arXiv preprint arXiv:1910.10683, 2019.',
    '[9] Noam Shazeer and Mitchell Stern. Adafactor: Adaptive learning rates with sublinear memory cost. arXiv preprint arXiv:1804.04235, 2018.',
    '[10] Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E. Hinton. Layer normalization. arXiv preprint arXiv:1607.06450, 2016.',
    '[11] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Ed Chi, Quoc Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models. NeurIPS, 2022.',
    '[12] Jiaxin Huang, Shixiang Shane Gu, Le Hou, Yuexin Wu, Xuezhi Wang, Hongkun Yu, and Jiawei Han. Large language models can self-improve. arXiv preprint arXiv:2210.11610, 2022.',
    '[13] Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large language models are zero-shot reasoners. arXiv preprint arXiv:2205.11916, 2022.',
    '[14] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361, 2020.',
    '[15] Tom Henighan, Jared Kaplan, Mor Katz, Mark Chen, Christopher Hesse, Jacob Jackson, Heewoo Jun, Tom B. Brown, Prafulla Dhariwal, Scott Gray, et al. Scaling laws for autoregressive generative modeling. arXiv preprint arXiv:2010.14701, 2020.',
    '[16] Greg Yang, Edward J. Hu, Igor Babuschkin, Szymon Sidor, Xiaodong Liu, David Farhi, Nick Ryder, Jakub Pachocki, Weizhu Chen, and Jianfeng Gao. Tensor Programs V: Tuning large neural networks via zero-shot hyperparameter transfer. arXiv preprint arXiv:2203.03466, 2022.',
    '[17] Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. Outrageously large neural networks: The sparsely-gated Mixture-of-Experts layer. arXiv preprint arXiv:1701.06538, 2017.',
    '[18] Barret Zoph, Irwan Bello, Sameer Kumar, Nan Du, Yanping Huang, Jeff Dean, Noam Shazeer, and William Fedus. ST-MoE: Designing stable and transferable sparse expert models. arXiv preprint arXiv:2202.08906, 2022.',
    '[19] Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, et al. Emergent abilities of large language models. TMLR, 2022.',
    '[20] Mostafa Dehghani, Stephan Gouws, Oriol Vinyals, Jakob Uszkoreit, and Lukasz Kaiser. Universal transformers. In International Conference on Learning Representations, 2019.',
    '[21] Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, and Yunfeng Liu. RoFormer: Enhanced transformer with rotary position embedding. arXiv preprint arXiv:2104.09864, 2021.',
    '[22] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. In Advances in Neural Information Processing Systems.',
    '[23] Xi Chen, Xiao Wang, Soravit Changpinyo, AJ Piergiovanni, Piotr Padlewski, Daniel Salz, Sebastian Goodman, Adam Grycner, Basil Mustafa, Lucas Beyer, et al. PaLI: A jointly-scaled multilingual language-image model. arXiv preprint arXiv:2209.06794, 2022.',
    '[24] Ben Wang and Aran Komatsuzaki. GPT-J-6B: A 6 billion parameter autoregressive language model, 2021.',
    '[25] Sid Black, Leo Gao, Phil Wang, Connor Leahy, and Stella Biderman. GPT-Neo: Large scale autoregressive language modeling with mesh-tensorflow, 2021.',
    '[26] Teven Le Scao, Angela Fan, Christopher Akiki, Ellie Pavlick, Suzana Ili´c, Daniel Hesslow, Roman Castagné, Alexandra Sasha Luccioni, François Yvon, Matthias Gallé, et al. Bloom: A 176B-parameter open-access multilingual language model. arXiv preprint arXiv:2211.05100, 2022.',
    '[27] Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, et al. OPT: Open pre-trained transformer language models. arXiv preprint arXiv:2205.01068, 2022.',
    '[28] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. LLaMA: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.',
    '[29] Alec Radford, Rafal Józefowicz, and Ilya Sutskever. Learning to generate reviews and discovering sentiment. arXiv preprint arXiv:1704.01444, 2017.',
    '[30] Guillaume Lample and Alexis Conneau. Cross-lingual language model pretraining. arXiv preprint arXiv:1901.07291, 2019.',
    '[31] Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. Flashattention: Fast and memory-efficient exact attention with io-awareness. arXiv preprint arXiv:2205.14135, 2022.',
    '[32] Rewon Child, Scott Gray, Alec Radford, and Ilya Sutskever. Generating long sequences with sparse transformers. arXiv preprint arXiv:1904.10509, 2019.',
    '[33] Markus N. Rabe and Charles Staats. Self-attention does not need o(n2) memory. arXiv preprint arXiv:2112.05682, 2021.',
    '[34] Scott Gray, Alec Radford, and Diederik P. Kingma. Gpu kernels for block-sparse weights, 2017.',
    '[35] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. ICLR, 2021.',
    '[36] Dan Hendrycks, Collin Burns, Steven Basart, Andrew Critch, Jerry Li, Dawn Song, and Jacob Steinhardt. Aligning AI with shared human values. ICLR, 2021.',
    '[37] Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners. 2019.',
    '[38] Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever. Improving language understanding by generative pre-training. 2018.',
    '[39] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. NeurIPS, 2017.',
    '[40] Paul F Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep reinforcement learning from human preferences. NeurIPS, 30, 2017.',
    '[41] Joel Hestness, Sharan Narang, Newsha Ardalani, Gregory Diamos, Heewoo Jun, Hassan Kianinejad, Md Patwary, Mostofa Ali, Yang Yang, and Yanqi Zhou. Deep learning scaling is predictable, empirically. arXiv preprint arXiv:1712.00409, 2017.',
    '[42] Neil C Thompson, Kristjan Greenewald, Keeheon Lee, and Gabriel F Manso. The computational limits of deep learning. arXiv preprint arXiv:2007.05558, 2020.',
    '[43] Mark Chen et al. Evaluating large language models trained on code. 2021.',
    '[44] Ian McKenzie et al. The Inverse Scaling Prize, 2022.',
    '[45] Jason Wei, Najoung Kim, Yi Tay, and Quoc V. Le. Inverse scaling can become U-shaped. arXiv preprint arXiv:2211.02011, 2022.',
    '[46] Ian McKenzie et al. Inverse Scaling Prize: First round winners, 2022.',
    '[47] Greg Brockman, Peter Welinder, Mira Murati, and OpenAI. OpenAI: OpenAI API, 2020.',
    '[48] Aarohi Srivastava et al. Beyond the imitation game: Quantifying and extrapolating the capabilities of language models. arXiv preprint arXiv:2206.04615, 2022.',
    '[49] Dan Hendrycks et al. Measuring massive multitask language understanding. arXiv preprint arXiv:2009.03300, 2020.',
    '[50] Yi Tay et al. Transcending scaling laws with 0.1% extra compute. arXiv preprint arXiv:2210.11399, 2022.',
    '[51] Hyung Won Chung et al. Scaling instruction-finetuned language models. arXiv preprint arXiv:2210.11416, 2022.',
    '[52] Rowan Zellers et al. HellaSwag: Can a machine really finish your sentence? ACL, 2019.',
    '[53] Xiaodong Liu et al. Adversarial training for large neural language models. arXiv preprint arXiv:2004.08994, 2020.',
    '[54] Peter Clark et al. Think you have solved question answering? Try ARC, the AI2 reasoning challenge. 2018.',
    '[55] Xuezhi Wang et al. Self-consistency improves chain of thought reasoning in language models. arXiv preprint arXiv:2203.11171, 2022.',
    '[56] Keisuke Sakaguchi et al. WinoGrande: An adversarial Winograd schema challenge at scale. arXiv preprint arXiv:1907.10641, 2019.',
    '[57] Bei Chen et al. CodeT: Code generation with generated tests. arXiv preprint arXiv:2207.10397, 2022.',
    '[58] Dheeru Dua et al. DROP: A reading comprehension benchmark requiring discrete reasoning over paragraphs. NAACL, 2019.',
    '[59] Kunlong Chen et al. Question directed graph attention network for numerical reasoning over text. arXiv preprint arXiv:2009.07448, 2020.',
    '[60] Karl Cobbe et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.',
    '[61] Aitor Lewkowycz et al. Solving quantitative reasoning problems with language models. arXiv preprint arXiv:2206.14858, 2022.',
    '[62] Jonathan Uesato et al. Solving math word problems with process- and outcome-based feedback. arXiv preprint arXiv:2211.14275, 2022.',
    '[63] Long Ouyang et al. Training language models to follow instructions with human feedback. arXiv preprint arXiv:2203.02155, 2022.',
    '[64] OpenAI. Introducing ChatGPT, 2022.',
    '[65] OpenAI. GPT-4, 2023.',
    '[66] Stephanie Lin, Jacob Hilton, and Owain Evans. TruthfulQA: Measuring how models mimic human falsehoods. ACL, 2022.',
    '[67] Yuntao Bai et al. Training a helpful and harmless assistant with reinforcement learning from human feedback. arXiv preprint arXiv:2204.05862, 2022.',
    '[68] OpenAI. How should AI systems behave, and who should decide?, 2023.',
    '[69] Jan Leike, John Schulman, and Jeffrey Wu. Our approach to alignment research, 2022.',
    '[70] Joseph Carlsmith. Is power-seeking AI an existential risk? arXiv preprint arXiv:2206.13353, 2022.',
    '[71] Amelia Glaese et al. Improving alignment of dialogue agents via targeted human judgements. arXiv preprint arXiv:2209.14375, 2022.',
    '[72] Ethan Perez et al. Red teaming language models with language models. arXiv preprint arXiv:2202.03286, 2022.',
    '[73] Samuel Gehman et al. RealToxicityPrompts: Evaluating neural toxic degeneration in language models. arXiv preprint arXiv:2009.11462, 2020.',
    '[74] Dora Seigel. How do you calculate SAT score? raw and scaled, 2020.',
    '[75] The Albert blog.',
    '[76] Mathematical Association of America. AMC statistics, 2023.',
    '[77] Halle Edwards. SAT percentiles and score rankings, 2022.',
    '[78] College Board. Understanding SAT scores, 2022.',
    '[79] College Board. AP score distributions by subject, 2022.',
    '[80] Center for Excellence in Education. 2020 USABO Semifinal exam score distribution, 2022.',
    '[81] Chris Swimmer. GRE score percentiles, 2021.',
    '[82] John B. Nici. AP Art History: 5 Practice Tests + Comprehensive Review + Online Practice. Barron\'s, 2020.',
    '[83] ETS. GRE sample issue task, 2022.',
    '[84] Margaret Mitchell et al. Model Cards for Model Reporting. FAT, 2019.',
    '[85] Nekesha Green et al. System Cards, a new resource for understanding how AI systems work, 2022.',
]
for r in refs:
    w(r)
w()

# ========== 附录 ==========
w('---')
w('## 附录')
w()

w('### 附录A：考试基准方法')
w()
w('#### A.1 来源')
w('我们使用了最新的公开官方过去试题，或购买的2022-2023年第三方备考材料中的练习题。我们对照模型的训练数据交叉检查了这些材料，以确定训练数据在多大程度上没有受到考试问题的污染，我们也在本文中报告了这一点。')
w('统一律师资格考试由我们的合作方Casetext和Stanford CodeX运行。')
w()
w('#### A.2 提示：多项选择')
w('对于每个多项选择部分，我们使用了带有标准答案解释的少样本提示。对于每个问题，我们采样了一个解释（温度为0.3）以提取多项选择的答案字母。')
w('我们将每个多项选择部分作为一对考试来处理：一个保留集和一个非保留集。我们使用非保留考试迭代了方法，然后对每个保留考试运行一次以获得最终分数。我们没有为USABO和MKSAP问题找到非保留考试，而是使用我们在AP生物学考试上迭代确定的最佳推测方法对这些考试运行了一次。')
w('对于AMC 10和AMC 12保留测试考试，我们发现了一个限制响应长度的错误。我们修复了该错误并重新运行了这些考试以确保准确的结果。对于大多数考试运行，我们直接从解释中提取模型的字母选择。对于GPT-4 USABO和SAT阅读/写作运行（有和没有视觉）、GPT-3.5运行以及GPT-4在没有视觉的情况下的SAT数学、GRE、USNCO、AP生物学、AP化学和AP环境科学运行，我们改为在已经采样的解释基础上，在温度为0时采样字母选择。这些方法差异是由于评估后检测到的代码不匹配造成的，我们认为它们对结果的影响很小。')
w()
w('#### A.3 提示：自由回答')
w('对于每个自由回答部分，我们以简单的指令遵循风格的请求向模型提供自由回答问题的提示，并使用温度0.6采样响应。对于AP考试，我们使用了最新的2022年提示（全部公开可用）；对于SAT，我们使用了三个提示——来自重新设计的SAT考试规范的样本论文提示1和样本论文提示2（CollegeBoard, 2015），加上官方SAT练习论文#1（CollegeBoard, 2016），并取平均分；对于GRE，我们使用了来自市售备考书的问题论文和论证论文提示。')
w('由于人类专家评分的迭代时间较长，我们没有对温度或提示进行方法迭代，而是以最佳推测温度（0.6）和提示（一个简单的指令遵循提示，见A.8节）对每个自由回答问题运行了一次。')
w('所有需要评估写作质量的正式论文自由回答问题（AP英语语言与写作、AP英语文学与写作、AP世界历史、AP美国历史、AP美国政府与政治、AP艺术史、GRE和SAT）由1-2名具有相关论文评分工作经验的外部合格承包商评分。我们使用包含一个高质量GRE样本论文回答的少样本提示来采样这些响应，以鼓励模型生成适当复杂的文本，而不是不自然地简洁回答。我们根据公开可用的官方评分指南中的准则，对所有其他自由回答问题的技术内容进行评分。')
w()
w('#### A.4 图像')
w('考试问题常常包含图像。像GPT-3.5这样只消费文本（而不是图像）作为输入的模型可能无法获得正确解决问题所需的所有信息。在评估文本模型的多项选择题时，我们在图像缺失的地方包含了一个文本标签IMAGE:，后跟一个无意义的文件名。这使我们能够给出基于文本的模型在多项选择考试中的性能下限。12 在评估多模态模型的多项选择题时，我们将图像嵌入到提示中。SAT阅读与写作、MKSAP、侍酒师、AP心理学、AP英语语言和AP英语文学考试的多项选择部分不包含任何图像。对于所有自由回答问题以及USABO 2020半决赛，我们尽可能客观地转录了任何图像和图表。这减少了评估自由回答答案所需的人工评分工作量，因为在此转录过程之后，自由回答提示不再包含图像，因此GPT-4的评分可以运行一次并用于有视觉和无视觉两种条件。')
w()
w('#### A.5 评分')
w('我们使用最佳可用的真实评分方法近似值，将多项选择部分分数和自由回答部分分数综合为总分数：对于SAT，我们使用官方SAT样本的分数计算图表将多项选择分数转换为标准分数（如SAT备考网站[74]上重新发布的）；对于GRE，我们使用官方公式将多项选择分数转换为130-170量表：准确率乘以40再加130；对于AP考试，我们使用了公开学习网站上的分数计算器，该计算器基于2019-2020年官方AP评分指南中的分值[75]。百分位数基于每种考试类型最近可用的考生分数分布。')
w('对于AMC 10和12的百分位结果，由于2022年分数分布尚未公布，我们使用了2021年11月A、B卷的两个官方公布的分数分布，取两者中较低百分位的最小值和较高百分位的最大值来报告估计的百分位范围[76]。其他百分位数基于官方分数分布[77][78][79][80][81]。')
w()
w('#### A.6 Codeforces评分')
w('为了确定Codeforces评分（ELO），我们在10个最近的竞赛上评估了每个模型。每个竞赛大约有6个问题，模型每个问题有10次尝试。每次竞赛后，我们根据模型的表现重复进行ELO调整，直到ELO评分收敛到均衡评分（这模拟了以相同模型表现重复尝试竞赛）。我们将10个竞赛各模拟了100次，并报告所有竞赛的平均均衡ELO评分。')
w('大约50%的模拟中，解决的问题数为0，导致均衡ELO评分为0。因此最终的平均ELO相当低。单个竞赛上达到的最大均衡ELO对于GPT-3.5约为1000，对于GPT-4约为1300。')
w()
w('#### A.7 模型快照细节')
w('我们使用2023年3月1日的模型快照运行了GPT-4的多项选择题，而自由回答问题是使用2023年2月23日的非最终模型快照运行和评分的。GPT-3.5的多项选择题和自由回答问题均使用标准的ChatGPT快照运行。我们使用2022年12月16日的早期GPT-4快照运行了USABO半决赛考试。')
w('我们的评估表明RLHF不会显著影响基础GPT-4模型的能力——更多讨论见附录B。')
w()
w('#### A.8 示例少样本提示')
w('（多项选择考试示例提示——AP艺术史[82]）')
w('（自由回答问题示例提示——GRE）')
w('（详见原文）')
w()

w('### 附录B：RLHF对能力的影响')
w()
w('为了测试RLHF对我们基础模型能力的影响，我们在GPT-4基础模型和RLHF后的GPT-4模型上运行了考试基准的多项选择部分。结果如表8所示。在所有考试中平均，基础模型得分为73.7%，而RLHF模型得分为74.0%，这表明后训练不会显著改变基础模型的能力。')
w('对于自由回答问题，很难在平等基础上比较基础模型和RLHF模型，因为我们采样自由回答答案的方法可能受益于模型的指令遵循能力。')
w()
w('表8：GPT-4基础和GPT-4后RLHF在考试基准上的比较。在所有考试中平均，基础模型平均得分为73.7%，而RLHF模型平均得分为74.0%，这表明后训练不会显著改变基础模型的能力。')
w()

w('### 附录C：专业和学术考试的污染')
w()
w('我们使用子串匹配来测量评估数据集与预训练数据之间的交叉污染。评估和训练数据都通过移除所有空格和符号、仅保留字符（包括数字）进行处理。对于每个评估样本，我们随机选择三个50字符的子串（如果样本少于50个字符，则使用整个样本）。如果三个采样的评估子串中的任何一个已处理训练样本的子串，则识别为匹配。这将产生一个污染样本列表。我们丢弃这些样本并重新运行以获得未污染的分数。')
w('我们的过滤方法存在一些局限性。我们的子串匹配可能导致假阴性（如果评估和训练数据之间存在小的差异）以及假阳性。我们仅使用评估样本的部分信息，仅使用问题、上下文或同等数据，而忽略答案、响应或同等数据。在某些情况下，多项选择选项也被排除。这些排除可能导致假阳性的增加。')
w('RLHF后训练数据集远小于预训练集，不太可能污染任何特定问题。但我们没有明确检查。')
w('如表9和10所示，污染总体上对报告结果的影响非常小。')
w()

w('### 附录D：学术基准的污染')
w()
w('我们使用与附录C中介绍的方法类似的方法来测量学术基准与预训练数据之间的交叉污染。结果如表11所示。')
w()

w('### 附录E：GPT-4训练中的GSM-8K')
w()
w('为了提高GPT-4的数学推理能力，我们在训练数据中混合了MATH和GSM-8K（两个常用的语言模型数学推理基准）的训练集。从这些数学基准中提取的token总数仅占GPT-4总训练预算的极小部分。在混合这些数学基准的数据时，保留了一部分训练数据，因此每个单独的训练示例可能被GPT-4看到，也可能没有。')
w('我们进行了污染检查以验证GSM-8K的测试集不包含在训练集中（见附录D）。我们建议将表2中报告的GPT-4 GSM-8K的性能结果解释为介于真正的少样本迁移和完整的基准特定调优之间。')
w()

w('### 附录F：多语言MMLU')
w()
w('我们使用Azure Translate翻译了MMLU[49]的所有问题和答案。我们使用外部模型进行翻译，而不是依赖GPT-4本身，以防模型对其自己的翻译有不具代表性的性能。我们选择了一系列涵盖不同地理区域和文字的语言，在表13中展示了一个从天文学类别中选出的问题示例，翻译成马拉地语、拉脱维亚语和威尔士语。翻译并不完美，在某些情况下会丢失微妙的信息，这可能损害性能。此外，一些翻译保留了英语中的专有名词，按照翻译惯例，这可能有助于性能。')
w('我们采用了与[4]相同的MMLU提示，模型被告知它是一个智能agent，提供问题和四个标记为A-D的答案选项列表，后跟"Answer:"。我们翻译了模型指令、问题和答案，但保留了"Answer" token以及英语的"A-D"选项。一个示例提示如表12所示。提示由三个样本组成，这三个样本从开发集中选取。我们使用三样本评估而不是常规的五样本，因为某些语言映射到更长的token序列。最后，我们通过从模型中选择具有最高概率的A-D token续写来分类正确答案。')
w()

w('### 附录G：GPT-4视觉输入示例')
w()
w('表14-19展示了GPT-4视觉输入能力的多个示例，包括：')
w('- 图表推理（表14）：计算格鲁吉亚和西亚的平均每日肉类消费之和')
w('- 物理问题（表15）：回答巴黎综合理工学院的考试问题')
w('- 图像理解（表16）：识别图片中的不寻常之处')
w('- 论文总结（表17）：阅读并总结InstructGPT论文')
w('- 网络迷因理解（表18, 19）：解释迷因和漫画')
w()

# ========== 附录H：系统卡 ==========
w('### 附录H：系统卡')
w()
w('GPT-4的系统卡[84, 85]附于本文档之后。')
w()

w('---')
w()
w('# GPT-4 系统卡')
w()
w('> OpenAI')
w()
w('## 摘要')
w()
w('大型语言模型（LLMs）正被部署在我们生活的许多领域，从浏览到语音助手到编码辅助工具，并具有巨大的社会影响潜力。[1, 2, 3, 4, 5, 6, 7] 本系统卡分析了GPT-4，GPT模型家族中最新的LLM。[8, 9, 10] 首先，我们强调了模型的局限性（例如，生成令人信服但微妙错误的文本）和能力（例如，提供非法建议的能力增强、双重用途能力的表现以及风险涌现行为）所带来的安全挑战。其次，我们概述了OpenAI为准备GPT-4部署所采用的安全流程。这涵盖了我们跨测量、模型级更改、产品和系统级干预（如监控和策略）以及外部专家参与的工作。最后，我们证明了虽然我们的缓解措施和流程改变了GPT-4的行为并防止了某些类型的滥用，但它们仍然有限，并且在某些情况下仍然脆弱。这表明需要进行预期性规划和治理。[11]')
w()
w('内容警告：本文档包含可能令人不安或冒犯的内容，包括性、仇恨或暴力性质的内容。')
w()
w('## 1 引言')
w()
w('大型语言模型（LLMs）已成为我们日常生活中越来越普遍的部分，其用途扩展到广泛的领域，包括网页浏览、语音助手和编码辅助工具。[1, 2, 3, 4] 这些模型有潜力以多种方式对社会产生重大影响。[5, 6, 7] 本系统卡分析了GPT-4，GPT模型家族中最新的大型语言模型。[8, 9, 10] 自2022年8月完成训练以来，我们一直在评估、对抗性测试和迭代改进模型以及围绕它的系统级缓解措施。我们的缓解措施和流程改变了GPT-4的行为并防止了某些类型的滥用，尽管它们存在局限性，这表明需要进行预期性规划和治理[11]以及进一步的安全研究。我们的部署方法在最大程度降低部署风险、启用积极用例和从部署中学习之间取得平衡。')
w()
w('GPT模型通常分两个阶段训练。首先，它们使用来自互联网的大量文本数据集进行训练，以预测下一个词。然后，模型使用额外的数据进行微调，使用一种称为基于人类反馈的强化学习（RLHF）的算法，以产生人类标注者更偏好的输出。[10, 12, 13] 在大型文本数据集上训练语言模型产生了诸如少样本学习[10]和执行涵盖不同领域的广泛自然语言任务（包括问答、算术和分类）的能力。微调使这些模型更具可控性和有用性。')
w()
w('### 1.1 发现和缓解措施概述')
w()
w('在本系统卡中，1 我们概述了GPT-4带来的安全挑战，并解释了我们为减轻其部署带来的潜在伤害而实施的干预措施。我们关注安全挑战，不是因为它们必然超过潜在的好处，2 而是因为我们希望推动安全测量、缓解和保证方面的进一步工作。本系统卡的范围窄于GPT-4能力可能解锁的潜在范围；值得注意的是，自定义微调和图像能力明确不在范围内。')
w()
w('我们重点分析模型的两个版本：一个早期版本为指令遵循进行了微调（"GPT-4-early"）；另一个版本为增加有用性和无害性[18]进行了微调，反映了本系统卡中概述的进一步缓解措施（"GPT-4-launch"）。3 当我们讨论GPT-4的风险时，我们通常会提到GPT-4-early的行为，因为它反映了在应用最低安全缓解措施时GPT-4的风险。在大多数情况下，由于我们应用的安全缓解措施，GPT-4-launch表现出更安全的行为。')
w()
w('较小语言模型相关的已知风险在GPT-4中同样存在。GPT-4可能生成潜在有害的内容，如关于策划攻击或仇恨言论的建议。它可能代表各种社会偏见和世界观，这些可能不代表用户的意图，4 或广泛共享的价值观。它也可能生成受损或易受攻击的代码。GPT-4的额外能力也带来了新的风险面。')
w()
w('为了理解这些风险的程度，我们邀请了50多位专家帮助我们更全面地了解GPT-4模型和潜在的部署风险。我们根据多种因素选择了这些领域，包括先前在语言模型和AI系统中观察到的风险，以及我们观察到用户对语言模型应用兴趣增加的领域。与这些专家合作使我们能够测试高风险领域中的模型行为——这些领域需要专业知识来评估——以及理解不足的新生风险。')
w()
w('通过这一分析，我们发现GPT-4有可能在被外部数据增强时用于尝试识别私人个体。')
w('我们还发现，尽管GPT-4的网络安全能力并未远超前几代LLM，但它确实延续了降低成功网络攻击某些步骤（如通过社会工程或增强现有安全工具）成本的趋势。在没有安全缓解措施的情况下，GPT-4还能够就如何进行有害或非法活动提供更详细的指导。最后，我们促成了对齐研究中心（ARC）对GPT-4自主复制5 和获取资源能力的初步模型评估——这是一个虽然推测性但可能随着足够先进的AI系统而成为可能的风险——结论是当前模型可能还无法自主做到这一点。')
w()
w('需要进一步的研究来完全表征这些风险。我们特别希望看到针对已识别的风险领域进行更稳健的评估，以及对这些行为在不同语言模型中的普遍性进行更具体的测量，并指导这些模型向更安全的方向发展。我们正在进行这些类型的评估，通常与其他研究小组合作，重点评估风险涌现行为。')
w()
w('除了测量工作外，我们还在开发和部署过程的不同步骤中旨在缓解已识别的问题。我们减少了预训练数据集中违反我们使用政策的某些类型内容（如不适当的色情内容）的普遍性，并对模型进行了微调以拒绝某些指令，如直接请求非法建议。我们还降低了模型幻觉的倾向，并通过利用先前模型使用的数据，减少了模型屈服的对抗性提示或利用（包括有时称为"越狱"的攻击）的表面积。此外，我们针对新的风险向量训练了一系列分类器，并将其整合到我们的监控工作流中，使我们能够更好地执行API使用政策。这些缓解措施的效果各不相同，但总体而言，我们能够显著降低生成各种潜在有害内容的容易程度，从而使GPT-4-launch在这些维度上比GPT-4-early安全得多。')
w()
w('本系统卡并非全面无遗，我们预计随时间推移将对下面讨论的问题有更多了解。与OpenAI的部署策略一致，[21] 我们应用了早期部署的经验教训，并期望将本部署中吸取的经验教训用于进行路线修正并为未来部署奠定基础。')
w()
w('请注意，本系统卡中包含的示例并非零样本，而是从我们的评估工作中精心挑选的，以说明特定类型的安全问题或伤害。我们包含示例是为了让读者了解所观察风险的性质。单个示例不足以展示这些问题可能表现的广度。')
w()
w('在第1节中，我们概述了GPT-4开发中观察到的一些安全挑战。在第2节中，我们讨论了部署准备过程和一些模型缓解措施及系统安全措施。在第3节中，我们通过讨论在我们迭代部署策略中所学到的关于观察到的风险方面的一些剩余限制和建议来总结。')
w()

w('## 2 GPT-4观察到的安全挑战')
w()
w('GPT-4在推理、知识保留和编码等领域展现出相比早期模型如GPT-2[22]和GPT-3[10]更佳的性能。这些改进中的许多也带来了新的安全挑战，我们在本节中强调了这些挑战。')
w()
w('我们对GPT-4进行了一系列定性和定量评估。这些评估帮助我们了解了GPT-4的能力、局限性和风险；优先安排我们的缓解工作；以及迭代测试和构建更安全的模型版本。我们探索的一些具体风险包括：6')
w('- 幻觉')
w('- 有害内容')
w('- 代表性、分配和服务质量的伤害')
w('- 虚假信息和影响力操作')
w('- 常规和非常规武器的扩散')
w('- 隐私')
w('- 网络安全')
w('- 风险涌现行为的潜力')
w('- 与其他系统的交互')
w('- 经济影响')
w('- 加速')
w('- 过度依赖')
w()
w('我们发现GPT-4-early和GPT-4-launch表现出与早期语言模型相同的许多局限性，例如产生有偏见和不可靠的内容。在我们的缓解措施实施之前，我们还发现GPT-4-early在诸如寻找销售非法商品或服务的网站以及策划攻击等领域呈现增加的风险。此外，模型增强的连贯性使其能够生成可能更可信和更有说服力的内容。')
w()

w('### 2.1 评估方法')
w()
w('#### 2.1.1 定性评估')
w('2022年8月，我们开始招募外部专家对GPT-4模型进行定性探测、对抗性测试并提供反馈。该测试包括压力测试、边界测试和红队测试。7 我们参照[27]中的定义将这些对抗性测试流程非正式地称为"红队测试"，即"发现计划、组织或技术系统中的缺陷和漏洞的结构化努力，通常由专门的"红队"执行，试图采用攻击者的思维和方法。" 我们在2023年3月10日对GPT-4-launch进行了内部对抗性测试。在此日期之前，我们还测试了多个类似的GPT-4版本，因此这里的分析也受到该探索的启发。红队测试已以多种方式应用于语言模型：用于减少有害输出；[28] 以及利用外部专业知识进行领域特定的对抗性测试。[16] 有些人探索了使用语言模型对语言模型进行红队测试。[29]')
w()
w('红队测试总体而言，以及我们称为"专家红队测试"8 的那种红队测试，只是我们用于识别、测量和测试AI系统的工作机制之一。[27] 我们的方法是迭代式红队测试，从最初假设哪些领域可能是最高风险开始，测试这些领域，并根据进展进行调整。它也是迭代式的，因为我们随着纳入新的缓解和控制层而进行多轮红队测试，进行测试和完善，并重复此过程。')
w()
w('我们联系了研究人员和行业专业人士——主要具有公平性、对齐研究、行业信任与安全、虚假/错误信息、化学、生物风险、网络安全、核风险、经济学、人机交互、法律、教育和医疗保健方面的专业知识——以帮助我们更全面地了解GPT-4模型和潜在的部署风险。我们根据多种因素选择了这些领域，包括但不限于：先前在语言模型和AI系统中观察到的风险；[6, 30] 以及我们观察到用户对语言模型应用兴趣增加的领域。此红队流程的参与者是基于先前在这些风险领域的研究或经验选择的，因此反映了偏向具有特定教育和专业背景的群体（例如，具有重要高等教育或行业经验的人）。参与者通常也与英语国家、西方国家（如美国、加拿大和英国）有联系。我们对红队成员的选择引入了一些偏见，并且很可能影响了红队成员如何解释特定风险以及他们如何探测政治、价值观和模型的默认行为。我们招募研究人员的策略也可能偏向于学术社区和AI公司中最为关注的风险类型。')
w()
w('这些专家可以访问早期版本的GPT-4（包括GPT-4-early）以及带有开发中缓解措施（GPT-4-launch的前身）的模型。他们识别了初始风险，推动了关键领域的安全研究和进一步的迭代测试。我们通过技术缓解措施以及政策和执行杠杆的组合，在许多已识别领域降低了风险；然而，许多风险仍然存在。我们预计随着时间的推移将继续更多地了解这些及其他风险类别。尽管这种早期的定性红队练习对于深入了解像GPT-4这样的复杂新颖模型非常有用，但它并非对所有可能风险的全面评估。')
w()

w('#### 2.1.2 定量评估')
w('作为我们定性评估和对抗性测试的补充，我们针对内容政策中的类别（如仇恨言论、自我伤害建议和非法建议）构建了内部定量评估。这些评估衡量语言模型在给定旨在引发每个类别内容的提示时，生成属于上述类别之一的内容的可能性。语言模型生成的文本使用分类器和人工分析进行分类，以判定是否包含不期望的内容。')
w('这些评估是为了在训练期间自动化和加速不同模型检查点的评估，并更容易地在安全相关标准上比较不同模型而构建的。我们特别针对被识别为高风险的内容领域以及我们进一步定位进行模型缓解的领域。详见模型缓解部分。')
w()

w('### 2.2 幻觉')
w()
w('GPT-4有"幻觉"的倾向，9 即"产生与某些来源相比无意义或不真实的内容。"[31, 32] 随着模型变得日益令人信服和可信，这种倾向可能特别有害，导致用户过度依赖它们。[见过度依赖中的进一步讨论] 反直觉的是，随着模型变得更加真实，幻觉可能变得更危险，因为用户在模型提供他们熟悉的领域中真实信息时建立了对模型的信任。此外，随着这些模型被整合到社会中并用于帮助自动化各种系统，这种幻觉倾向是可能导致整体信息质量下降并进一步降低自由可用信息的真实性和可信度的因素之一。[33]')
w()
w('我们使用了一系列方法，在封闭域和开放域上下文中测量了GPT-4的幻觉潜力。10 我们使用自动评估（使用GPT-4作为零样本分类器）和人工评估来测量封闭域幻觉。对于开放域幻觉，我们收集了被标记为不真实的真实世界数据，进行了审查，并在可能的情况下为其创建了一个"事实性"集合。11 我们用它来评估模型生成相对于"事实性"集合的情况，并促进人工评估。')
w()
w('GPT-4经过训练，通过利用来自先前模型（如ChatGPT）的数据来减少模型幻觉的倾向。在内部评估中，GPT-4-launch在避免开放域幻觉方面比我们最新的GPT-3.5模型高出19个百分点，在避免封闭域幻觉方面高出29个百分点。')
w()

w('### 2.3 有害内容')
w()
w('语言模型可以通过提示生成不同类型的有害内容。这里我们指的是违反我们政策的内容，或可能对个人、群体或社会造成伤害的内容。12 这种伤害评估不考虑使用上下文，而使用上下文在确定内容最终是否有害方面起着关键作用。[39] 因此，我们专注于无论出现在什么上下文中都具有伤害潜力的内容领域。')
w()
w('例如，GPT-4-early可以生成仇恨言论、歧视性语言、煽动暴力或内容，这些内容随后被用于散布虚假叙事或利用个人。')
w('此类内容可能伤害边缘化社区，助长敌对的在线环境，并在极端情况下引发现实世界的暴力和歧视。特别是，我们发现对GPT-4-early的有意探测可能导致以下类型的有害内容[背景见[6, 21]]：')
w('1. 自我伤害行为的建议或鼓励')
w('2. 色情或暴力等露骨内容')
w('3. 骚扰、贬低和仇恨内容')
w('4. 可用于策划攻击或暴力的内容')
w('5. 寻找非法内容的指示')
w()
w('我们在模型拒绝方面的工作（在第2节中描述）旨在减少模型产生此类有害内容的倾向。下面我们提供GPT-4-early与GPT-4-launch（我们发布的版本）的一些比较示例。13')
w()

w('### 2.4 代表性、分配和服务质量的伤害')
w()
w('语言模型可以放大偏见并固化刻板印象。[40, 41, 42, 43, 44, 45, 46, 6] 与早期的GPT模型和其他常见语言模型一样，GPT-4-early和GPT-4-launch都继续强化社会偏见和世界观。')
w()
w('我们运行的评估过程帮助生成了GPT-4模型不同版本中社会偏见的额外定性证据。我们发现模型有潜力强化和复制特定的偏见和世界观，包括对某些边缘化群体的有害刻板印象和贬低性关联。模型行为，如不适当的回避行为，也可能加剧刻板印象或贬低性伤害。例如，某些版本的模型倾向于在回答关于女性是否应该被允许投票的问题时进行回避。')
w()
w('虽然我们的测试工作侧重于代表性伤害而非分配性伤害，但重要的是要注意，在涉及机会或资源分配的决策或为决策提供信息的上下文中使用GPT-4需要仔细评估不同群体间的表现。特别是，我们的使用政策禁止在高风险政府决策（如执法、刑事司法、移民和庇护）或提供法律或健康建议的上下文中使用我们的模型和产品。此外，GPT-4在不同人口统计和任务上表现出一些性能差异，例如，某些语言的使用者性能下降，如GPT-4技术报告中所述。这样的差异也可能导致服务质量的不平等。')
w()
w('某些类型的偏见可以通过训练拒绝来缓解，即让模型拒绝回答某些问题。当提示试图生成明确刻板印象或贬低某个群体的引导性问题时，这可能有效。然而，重要的是要注意，拒绝和其他缓解措施在某些情况下也可能加剧[35]偏见，或导致虚假的保证感。[43] 此外，跨不同人口统计或领域的不平等拒绝行为可能导致服务质量伤害。例如，拒绝可能通过拒绝为一个群体生成歧视性内容但为另一个群体生成而特别加剧性能差异问题。')
w()

w('### 2.5 虚假信息和影响力操作')
w()
w('GPT-4可以生成看似真实且有针对性的内容，包括新闻文章、推文、对话和电子邮件。在有害内容中，我们讨论了类似能力如何可能被滥用来利用个人。在这里，我们讨论围绕虚假信息和影响力操作的一般关切。14 根据我们的一般能力评估，我们预计GPT-4将比GPT-3更擅长生成真实、有针对性的内容。因此，存在GPT-4被用于生成旨在误导的内容的风险。[50]')
w()
w('经验证据表明，早期的语言模型也可用于生成误导性但具有说服力的内容。[51] 例如，研究人员发现GPT-3能够执行与改变话题叙事相关的任务。[52] 由GPT-3等语言模型在政治化问题上生成的具有说服力的呼吁被发现几乎与人类撰写的呼吁一样有效。[53, 54] 根据GPT-4在相关语言任务上的表现，我们预计它在这类任务上会比GPT-3更好，这增加了不良行为者可能使用GPT-4创建误导性内容以及社会未来的认知观点可能部分由具有说服力的LLM塑造的风险增加。')
w()
w('我们的红队测试结果表明，GPT-4在许多领域可以与人类宣传人员媲美，特别是与人类编辑合作时。尽管如此，在可靠性重要的领域，幻觉可能降低GPT-4对宣传人员的效果。红队测试发现，GPT-4还能够生成看似合理实现宣传目标的计划。')
w()

w('### 2.6 常规和非常规武器的扩散')
w()
w('某些LLM能力具有双重用途潜力，意味着这些模型可以用于"商业和军事或扩散应用"。[57] 我们在四个双重用途领域对模型进行了压力测试、边界测试和红队测试，16以探索我们的模型是否可能为寻求开发、获取或散布核武器、放射性武器、生物武器和化学武器的扩散者17提供必要信息。成功的扩散取决于许多"要素"，信息是其中之一。威胁行为者还需要获得双重用途物品和实验室设备，由于出口管制或其他特殊许可要求，这些通常难以获得。')
w()
w('单独而言，访问GPT-4是扩散的不足条件，但可能改变扩散者可用的信息，特别是与传统搜索工具相比。红队成员选择了一组问题来提示GPT-4和传统搜索引擎，发现使用GPT-4时研究完成的时间缩短了。在某些情况下，研究过程缩短了几个小时而不牺牲信息准确性。因此我们得出结论，一个关键的风险驱动因素是GPT-4生成公开可获取但难以找到的信息的能力，缩短了用户进行研究的时间并以非专家用户易于理解的方式编译这些信息。')
w()

w('### 2.7 隐私')
w()
w('GPT-4从各种许可、创建和公开可用的数据源学习，其中可能包括公开可用的个人信息。[59, 60] 因此，我们的模型可能了解在公共互联网上有显著存在的人们，如名人和公众人物。GPT-4还可以在单个完成中综合多个不同的信息类型并执行多个推理步骤。该模型可以完成可能与个人和地理信息相关的多个基本任务，例如确定与电话号码相关的地理位置或回答教育机构位于何处，而无需浏览互联网。例如，模型可以高召回率地将Rutgers大学电子邮件地址与具有新泽西区号的电话号码关联起来，并解释其推理路径。通过结合这些类型任务上的能力，GPT-4在被外部数据增强时有可能被用于尝试识别个人。')
w()

w('### 2.8 网络安全')
w()
w('GPT-4在社交工程的一些子任务（如起草钓鱼邮件）和解释某些漏洞方面是有用的。它也可能加速网络操作的某些方面（如解析审计日志或总结从网络攻击中收集的数据）。然而，由于其"幻觉"倾向和有限的上下文窗口，GPT-4在网络安全操作方面存在显著局限性。它并未改进现有的侦察、漏洞利用和网络导航工具，并且在复杂和高层次活动（如新型漏洞识别）方面不如现有工具有效。')
w()

w('### 2.9 风险涌现行为的潜力')
w()
w('更强大的模型中经常涌现出新颖的能力。[61, 62] 其中一些特别令人担忧的是创建和执行长期计划的能力，[63] 积累权力和资源（"权力寻求"），[64] 以及表现出日益"代理性"的行为。[65] 此上下文中的"代理性"并不意味着将语言模型拟人化或指代感知能力，而是指系统能够完成可能未被具体指定且未在训练中出现的目标；专注于实现特定、可量化的目标；以及进行长期规划。模型中已经存在一些这种涌现行为的证据。[66, 67, 65]')
w()
w('对于大多数可能的目标，最佳计划涉及辅助性的权力寻求行动，因为这本质上对于推进目标和避免对其的更改或威胁是有用的。19 [68, 69] 更具体地说，权力寻求对于大多数奖励函数和多种类型的agent是最优的；[70, 71, 72] 并且有证据表明现有模型可以将权力寻求识别为工具上有用的策略。[29] 因此，由于权力寻求行为可能带来的高风险，我们特别感兴趣于评估这种行为。[73, 74]')
w()
w('我们授予对齐研究中心（ARC）早期访问模型的权限，作为专家红队测试工作的一部分，以使他们的团队能够评估权力寻求行为的风险。ARC评估的权力寻求的具体形式是模型自主复制和获取资源的能力。我们向他们提供了多个版本的GPT-4模型的早期访问权限，但他们无法进行微调。他们也没有访问我们部署的最终版本模型。最终版本具有与限制早期模型权力寻求能力的一些因素相关的能力改进，例如更长的上下文长度和改进的问题解决能力。')
w()

w('### 2.10 与其他系统的交互')
w()
w('理解GPT-4如何与其他系统交互对于评估这些模型在各种现实世界情境中可能构成的风险至关重要。')
w()
w('除了ARC在风险涌现行为潜力部分进行的测试外，红队成员还评估了使用GPT-4增强其他工具[76, 77, 78, 79]来完成可能具有对抗性质的任务。我们在化学领域强调了一个这样的例子，目标是搜索与其他化合物相似的化学化合物，提出可在商业目录中购买的替代品，并执行购买。')
w()

w('### 2.11 经济影响')
w()
w('GPT-4对经济和劳动力的影响应该是政策制定者和其他利益相关者的重要考虑因素。虽然现有研究主要关注AI和生成模型如何增强人类工人，但GPT-4或后续模型可能导致某些工作的自动化。[83] 这可能导致劳动力置换。[84] 随着时间的推移，我们预计GPT-4将影响甚至历史上需要多年经验和教育的工作，如法律服务。[85]')
w()

w('### 2.12 加速')
w()
w('OpenAI一直关注像GPT-4这样的最先进系统的开发和部署如何影响更广泛的AI研究和开发生态系统。23 一个特别重要的关切是竞赛动态导致安全标准下降、不良规范的扩散以及AI时间线的加速，这些都加剧了与AI相关的社会风险。我们在这里将之称为"加速风险。"24 这是我们花费六个月进行安全研究、风险评估和迭代后才发布GPT-4的原因之一。25 为了更具体地了解GPT-4部署带来的加速风险，我们招募了专家预测者26来预测调整GPT-4部署的各种特征（如时间、沟通策略和商业化方法）将如何影响（加速风险的具体指标）。')
w()

w('### 2.13 过度依赖')
w()
w('如上文2.2节所述，尽管GPT-4具有能力，但它仍然倾向于编造事实、加倍坚持错误信息以及错误地执行任务。此外，它通常以比早期GPT模型更令人信服和可信的方式表现出这些倾向（例如，由于权威性的语气或呈现在高度详细的准确信息上下文中），增加了过度依赖的风险。')
w()
w('当用户过度信任和依赖模型时，就会发生过度依赖，可能导致未被注意的错误和不充分的监督。这可能以各种方式发生：用户可能因为对模型的信任而不会警惕错误；他们可能无法根据用例和上下文提供适当的监督；或者他们可能在缺乏专业知识的领域中使用模型，使识别错误变得困难。随着用户对系统越来越熟悉，对模型的依赖可能阻碍新技能的开发，甚至导致重要技能的丧失。')
w()

w('## 3 部署准备')
w()
w('自8月初以来，OpenAI一直在迭代[21]改进GPT-4和我们的部署计划，为更安全的发布做准备。我们相信这已经减少了风险面，尽管尚未完全消除。今天的部署代表了在最小化部署风险、启用积极用例和从部署中学习之间取得平衡。在此期间，我们的工作包括以下相互关联的步骤：')
w('1. 评估方法（如上所述）')
w('2. 模型缓解措施')
w('3. 系统安全')
w()
w('我们的方法涉及将模型级更改（如训练模型拒绝某些请求）与系统级缓解措施（如在用户界面中应用最佳实践以支持用户，以及监控违反我们使用政策的行为）相结合。')
w()

w('### 3.1 模型缓解措施')
w()
w('我们使用了数据集干预和预训练后干预的组合，在模型层面减轻伤害。')
w('在预训练阶段，我们为GPT-4过滤了数据集混合，专门减少不适当的色情文本内容。我们通过内部训练的分类器[37]和基于词典的方法的组合来识别被标记为极可能包含不适当色情内容的文档。然后我们从预训练集中移除了这些文档。')
w('在预训练阶段之后，我们塑造GPT-4-launch行为的主要方法是RLHF。我们使用了[12]中概述的方法。我们收集演示数据（给定输入，演示模型应如何响应）和模型输出的排名数据（给定输入和多个输出，将输出从最佳到最差排序）来自人类训练师。28')
w()

w('## 4 系统安全')
w()
w('### 4.1 使用政策和监控')
w('OpenAI禁止将我们的模型和工具用于某些活动和内容，如我们的使用政策所述。这些政策旨在禁止以造成个人或社会伤害的方式使用我们的模型和工具。我们根据新风险和我们模型使用方式的新信息更新这些政策。访问和使用我们的模型也受OpenAI使用条款的约束。')
w()

w('### 4.2 内容分类器开发')
w('审核分类器在我们的监控和执行流程中发挥着关键作用。我们不断开发和改进这些分类器。我们的几个审核分类器通过Moderation API端点对开发者可用，使开发者在将语言模型集成到其产品中时能够过滤掉有害内容。')
w()

w('## 5 结论和后续步骤')
w()
w('OpenAI在GPT-4的整个开发和部署过程中实施了各种安全措施和流程，这些措施减少了其生成有害内容的能力。然而，GPT-4仍然容易受到对抗性攻击和利用或"越狱"，并且有害内容并非风险的唯一来源。微调可以修改模型的行为，但预训练模型的基本能力（如生成有害内容的潜力）仍然是潜在的。随着与之相关的能力和风险增加，实现这些及其他干预措施的极高可靠性将变得至关重要；即使是现在，用使用政策和监控等其他干预措施补充这些模型级缓解措施也很重要，正如我们在系统安全部分所讨论的那样。')
w()

w('## 6 致谢')
w()
w('我们感谢帮助我们早期测试模型的专家对抗性测试者和红队成员，他们为我们的风险评估和系统卡输出提供了信息。参与此红队流程并不代表认可OpenAI的部署计划或OpenAI的政策。')
w()

w('## 系统卡参考文献')
w()
system_refs = [
    '[1] A. Tamkin, M. Brundage, J. Clark, and D. Ganguli, "Understanding the Capabilities, Limitations, and Societal Impact of Large Language Models," Feb. 2021.',
    '[2] "Introducing the new Bing." https://www.bing.com/new.',
    '[3] J. Hilton, R. Nakano, S. Balaji, and J. Schulman, "WebGPT: Improving the factual accuracy of language models through web browsing," Dec. 2021.',
    '[4] "ACT-1: Transformer for Actions – Adept." https://www.adept.ai/blog/act-1.',
    '[5] M. Chen et al., "Evaluating Large Language Models Trained on Code," July 2021.',
    '[6] L. Weidinger et al., "Ethical and social risks of harm from Language Models," Dec. 2021.',
    '[7] I. Solaiman et al., "Release Strategies and the Social Impacts of Language Models," Nov. 2019.',
    '[8] A. Radford, "Improving language understanding with unsupervised learning," June 2018.',
    '[9] A. Radford et al., "Better language models and their implications," Feb. 2019.',
    '[10] T. B. Brown et al., "Language Models are Few-Shot Learners," July 2020.',
    '[11] S. Altman, "Planning for AGI and beyond," Feb. 2023.',
    '[12] L. Ouyang et al., "Training language models to follow instructions with human feedback," Mar. 2022.',
    '[13] P. Christiano et al., "Deep reinforcement learning from human preferences," Feb. 2023.',
    '[14] M. Mitchell et al., "Model Cards for Model Reporting," FAT, 2019.',
    '[15] N. Green et al., "System Cards, a new resource for understanding how AI systems work," Feb. 2022.',
    '[16] "DALL·E 2 Preview - Risks and Limitations," OpenAI, Apr. 2022.',
    '[17] J. Sandbrink et al., "Differential Technology Development: A Responsible Innovation Principle for Navigating Technology Risks," Sept. 2022.',
    '[18] Y. Bai et al., "Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback," Apr. 2022.',
    '[19] E. Perez et al., "Discovering Language Model Behaviors with Model-Written Evaluations," Dec. 2022.',
    '[20] B. P. Kehoe, "Zen and the Art of the Internet," Project Gutenberg, June 1992.',
    '[21] M. Brundage et al., "Lessons learned on language model safety and misuse," Mar. 2022.',
    '[22] A. Radford et al., "Language Models are Unsupervised Multitask Learners," 2019.',
    '[23] G. C. Bowker and S. L. Star, "Sorting Things Out," MIT Press, Aug. 2000.',
    '[24] L. Weidinger et al., "Taxonomy of Risks posed by Language Models," FAccT, 2022.',
    '[25] I. Solaiman and C. Dennison, "Process for Adapting Language Models to Society (PALMS) with Values-Targeted Datasets," Nov. 2021.',
    '[26] H. Khlaaf, "Toward Comprehensive Risk Assessments and Assurance of AI-Based Systems," Trail of Bits, 2023.',
    '[27] M. Brundage et al., "Toward Trustworthy AI Development: Mechanisms for Supporting Verifiable Claims," Apr. 2020.',
    '[28] D. Ganguli et al., "Red Teaming Language Models to Reduce Harms: Methods, Scaling Behaviors, and Lessons Learned," Nov. 2022.',
    '[29] E. Perez et al., "Red Teaming Language Models with Language Models," Feb. 2022.',
    '[30] H. Khlaaf et al., "A Hazard Analysis Framework for Code Synthesis Large Language Models," July 2022.',
    '[31] J. Maynez et al., "On Faithfulness and Factuality in Abstractive Summarization," May 2020.',
    '[32] S. Lin et al., "TruthfulQA: Measuring How Models Mimic Human Falsehoods," May 2022.',
    '[33] J. A. Goldstein et al., "Forecasting potential misuses of language models for disinformation campaigns and how to reduce risk," Jan. 2023.',
    '[34] O. Evans et al., "Truthful AI: Developing and governing AI that does not lie," Oct. 2021.',
    '[35] A. Xu et al., "Detoxifying Language Models Risks Marginalizing Minority Voices," Apr. 2021.',
    '[36] L. Dixon et al., "Measuring and Mitigating Unintended Bias in Text Classification," AIES, 2018.',
    '[37] T. Markov et al., "A Holistic Approach to Undesired Content Detection in the Real World," Feb. 2023.',
    '[38] OpenAI, "How should AI systems behave, and who should decide?," Feb. 2023.',
    '[39] M. Rauh et al., "Characteristics of Harmful Text: Towards Rigorous Benchmarking of Language Models," Oct. 2022.',
    '[40] S. L. Blodgett et al., "Language (Technology) is Power: A Critical Survey of Bias in NLP," May 2020.',
    '[41] S. Dev et al., "On Measures of Biases and Harms in NLP," AACL-IJCNLP, 2022.',
    '[42] T. Bolukbasi et al., "Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings," July 2016.',
    '[43] H. Gonen and Y. Goldberg, "Lipstick on a Pig: Debiasing Methods Cover up Systematic Gender Biases in Word Embeddings But do not Remove Them," NAACL, 2019.',
    '[44] K. Webster et al., "Mind the GAP: A Balanced Corpus of Gendered Ambiguous Pronouns," Oct. 2018.',
    '[45] E. M. Bender et al., "On the Dangers of Stochastic Parrots: Can Language Models Be Too Big?," FAccT, 2021.',
    '[46] R. Bommasani et al., "On the Opportunities and Risks of Foundation Models," Aug. 2021.',
    '[47] S. U. Noble, "Algorithms of Oppression," NYU Press, Feb. 2018.',
    '[48] R. Richardson et al., "Dirty Data, Bad Predictions: How Civil Rights Violations Impact Police Data, Predictive Policing Systems, and Justice," Feb. 2019.',
    '[49] W. MacAskill, "What We Owe The Future," Basic Books, Aug. 2022.',
    '[50] OpenAI, "GPT-2: 1.5B release," Nov. 2019.',
]
for r in system_refs:
    w(r)
w()

# Write output
result = '\n'.join(out)
with open('/Users/dazhang/PycharmProject/Papers/2-LLM/2024-GPT-4 Technical Report-翻译.md', 'w') as f:
    f.write(result)

print(f"Translation written: {len(result)} chars, {len(result.split())} tokens")
print(f"Total lines: {len(out)}")
