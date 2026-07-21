#!/usr/bin/env python3
"""Expanded full translation of GPT-4 Technical Report."""

out = []
def w(s=''):
    out.append(s)

# ============================================================
# TITLE & HEADER
# ============================================================
w('# GPT-4 技术报告')
w()
w('> OpenAI | arXiv:2303.08774v6 [cs.CL] 2024年3月4日')
w()
w('*请引用本文为"OpenAI (2023)"。完整的作者贡献声明出现在文档末尾。关于本技术报告的通信可发送至 gpt4-report@openai.com*')
w()

# ========== ABSTRACT ==========
w('---')
w('## 摘要')
w()
w('我们报告了GPT-4的开发情况，GPT-4是一个大规模多模态模型，能够接受图像和文本输入并生成文本输出。虽然在许多现实场景中能力不如人类，但GPT-4在各种专业和学术基准测试中展现出人类水平的性能，包括在模拟律师资格考试中获得约前10%考生的成绩。GPT-4是基于Transformer的模型，通过预训练预测文档中的下一个token。训练后的对齐过程在事实性和对期望行为的遵守方面提升了性能。该项目的一个核心组成部分是开发能够在广泛规模范围内可预测行为的基础设施和优化方法。这使得我们能够基于训练计算量不超过GPT-4的1/1000的模型，准确预测GPT-4某些方面的性能。')
w()

# ========== 1 INTRODUCTION ==========
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

# ========== 2 SCOPE ==========
w('## 2 本技术报告的范围与局限')
w()
w('本报告侧重于GPT-4的能力、局限性和安全属性。GPT-4是一个Transformer风格模型[39]，通过预训练预测文档中的下一个token，使用了公开可用的数据（如互联网数据）和从第三方提供商许可的数据。然后使用基于人类反馈的强化学习（RLHF）[40]对模型进行微调。考虑到竞争环境以及像GPT-4这样的大规模模型的安全影响，本报告不包含关于架构（包括模型大小）、硬件、训练计算、数据集构建、训练方法或类似的进一步细节。')
w()
w('我们致力于对我们的技术进行独立审计，并在随本次发布附带的系统卡中分享了一些初步的步骤和想法。我们计划向其他第三方提供进一步的技术细节，这些第三方可以就如何在上述竞争和安全考虑与进一步提高透明度的科学价值之间进行权衡向我们提供建议。')
w()

# ========== 3 PREDICTABLE SCALING ==========
w('## 3 可预测的扩展')
w()
w('GPT-4项目的一个重点方向是构建一个可预测扩展的深度学习栈。主要原因是对于像GPT-4这样非常大的训练运行，进行广泛的模型特定调优是不可行的。为了解决这个问题，我们开发了在多个规模上具有非常可预测行为的基础设施和优化方法。这些改进使我们能够从使用计算量少1,000倍到10,000倍的较小模型训练的模型中可靠地预测GPT-4某些方面的性能。')
w()
w('### 3.1 损失预测')
w()
w('适当训练的大型语言模型的最终损失被认为可以用训练模型所用的计算量的幂律很好地近似[41, 42, 2, 14, 15]。')
w()
w('为了验证我们优化基础设施的可扩展性，我们通过拟合一个带有不可约损失项的标度律（如Henighan等人[15]所述）来预测GPT-4在内部代码库（不属于训练集）上的最终损失：$L(C) = aC^b + c$，该标度律基于使用相同方法论但计算量最多比GPT-4少10,000倍的模型。这个预测是在运行开始后不久做出的，没有使用任何部分结果。拟合的标度律以高精度预测了GPT-4的最终损失（图1）。')
w()
w('### 3.2 HumanEval上的能力扩展')
w()
w('在训练前了解模型的能力可以改善关于对齐、安全和部署的决策。除了预测最终损失外，我们还开发了预测更可解释能力指标的方法。其中一个指标是HumanEval数据集[43]上的通过率，该数据集衡量合成不同复杂度的Python函数的能力。我们通过外推使用最多少1,000倍计算量训练的模型，成功预测了HumanEval数据集子集上的通过率（图2）。')
w()
w('对于HumanEval中的单个问题，性能有时可能随规模而下降。尽管存在这些挑战，我们发现一个近似的幂律关系：$-\\mathbb{E}_P[\\log(\\text{pass\\_rate}(C))] = \\alpha\\times C^{-k}$，其中$k$和$\\alpha$是正常数，$P$是数据集中问题的子集。我们假设这个关系对数据集中的所有问题都成立。在训练完成前我们注册了对GPT-4在HumanEval上性能的预测。')
w()
w('某些能力仍然难以预测。例如逆标度奖[44]发现GPT-4逆转了性能随规模下降的趋势（图3）。')
w()
w('**图1：GPT-4和较小模型的性能——内部代码库最终损失预测。** 虚线为幂律拟合，准确预测了GPT-4的最终损失。x轴为归一化训练计算量。')
w('**图2：GPT-4和较小模型的性能——HumanEval 23个编码问题子集上的平均对数通过率。**')
w('**图3：GPT-4和较小模型在"Hindsight Neglect"任务上的准确率。**')
w()

# ========== 4 CAPABILITIES ==========
w('## 4 能力')
w()
w('我们在多样化的基准测试集上测试了GPT-4，包括模拟原本为人类设计的考试。我们未对这些考试进行特定训练。考试中的少数问题在训练期间被模型看到过；对于每个考试，我们运行一个移除了这些问题的变体，并报告两者中较低的分数。我们相信结果具有代表性。关于污染（方法和每个考试的统计信息）的更多细节，请参见附录C。')
w()
w('**表1：GPT在学术和专业考试上的表现。** 在每种情况下，我们模拟真实考试的条件和评分。我们报告GPT-4根据考试特定评分标准评定的最终分数，以及达到GPT-4分数的考生百分位数。')
w()
w('| 考试 | GPT-4 | GPT-4(无视觉) | GPT-3.5 |')
w('|---|---|---|---|')
w('| 统一律师考试 | 298/400(~90th) | 298/400(~90th) | 213/400(~10th) |')
w('| LSAT | 163(~88th) | 161(~83rd) | 149(~40th) |')
w('| SAT阅读与写作 | 710/800(~93rd) | 710/800(~93rd) | 670/800(~87th) |')
w('| SAT数学 | 700/800(~89th) | 690/800(~89th) | 590/800(~70th) |')
w('| GRE定量 | 163/170(~80th) | 157/170(~62nd) | 147/170(~25th) |')
w('| GRE文字 | 169/170(~99th) | 165/170(~96th) | 154/170(~63rd) |')
w('| GRE写作 | 4/6(~54th) | 4/6(~54th) | 4/6(~54th) |')
w('| USABO 2020半决赛 | 87/150(99th-100th) | 87/150(99th-100th) | 43/150(31st-33rd) |')
w('| Codeforces评分 | 392(<5th) | 392(<5th) | 260(<5th) |')
w('| AP生物学 | 5(85th-100th) | 5(85th-100th) | 4(62nd-85th) |')
w('| AP微积分BC | 4(43rd-59th) | 4(43rd-59th) | 1(0th-7th) |')
w('| AP化学 | 4(71st-88th) | 4(71st-88th) | 2(22nd-46th) |')
w('| AP环境科学 | 5(91st-100th) | 5(91st-100th) | 5(91st-100th) |')
w('| AP宏观经济学 | 5(84th-100th) | 5(84th-100th) | 2(33rd-48th) |')
w('| AP微观经济学 | 5(82nd-100th) | 4(60th-82nd) | 4(60th-82nd) |')
w('| AP心理学 | 5(83rd-100th) | 5(83rd-100th) | 5(83rd-100th) |')
w('| AMC 10 | 30/150(6th-12th) | 36/150(10th-19th) | 36/150(10th-19th) |')
w('| AMC 12 | 60/150(45th-66th) | 48/150(19th-40th) | 30/150(4th-8th) |')
w('| Leetcode(简单) | 31/41 | 31/41 | 12/41 |')
w('| Leetcode(中等) | 21/80 | 21/80 | 8/80 |')
w('| Leetcode(困难) | 3/45 | 3/45 | 0/45 |')
w()
w('GPT-4在大多数这些考试上展现出人类水平的性能。模型在考试中的能力似乎主要源于预训练过程，并未受到RLHF的显著影响（见附录B）。')
w()
w('我们还评估了预训练的基础GPT-4模型在传统语言模型评估基准上的表现（表2）。')
w()
w('**表2：GPT-4在学术基准上的表现。**')
w()
w('| 基准 | GPT-4 | GPT-3.5 | 最佳外部LM(少样本) | 最佳外部模型(含调优) |')
w('|---|---|---|---|---|')
w('| MMLU | 86.4% | 70.0% | 70.7% | 75.2% |')
w('| HellaSwag | 95.3% | 85.5% | 84.2% | 85.6% |')
w('| ARC | 96.3% | 85.2% | 85.2% | 86.5% |')
w('| WinoGrande | 87.5% | 81.6% | 85.1% | 85.1% |')
w('| HumanEval | 67.0% | 48.1% | 26.2% | 65.8% |')
w('| DROP(F1) | 80.9 | 64.1 | 70.8 | 88.4 |')
w('| GSM-8K | 92.0%* | 57.1% | 58.8% | 87.3% |')
w()
w('多语言MMLU：GPT-4在大多数测试语言上优于现有语言模型的英语性能（图5）。')
w()
w('GPT-4在遵循用户意图方面显著改进[63]。在5,214个提示上，GPT-4响应在70.2%的提示中优于GPT-3.5。我们正在开源OpenAI Evals框架。')
w()
w('### 4.1 视觉输入')
w()
w('GPT-4接受图像和文本交错输入，生成文本输出。在一系列领域（文档、照片、图表、截图）中展现出与纯文本输入相似的能力。标准测试时技术（少样本提示、链式思维等）在视觉+文本模式下同样有效。更多示例见附录G。')
w()

# ========== 5 LIMITATIONS ==========
w('## 5 局限性')
w()
w('GPT-4与早期GPT模型[1,37,38]有相似局限性。最重要的：它仍非完全可靠（"幻觉"事实和推理错误）。在高风险场景中使用应非常谨慎。')
w()
w('GPT-4相比GPT-3.5显著减少幻觉。内部对抗性事实性评估中比最新GPT-3.5高19个百分点（图6）。')
w()
w('**图6：GPT-4在九个内部对抗性事实性评估上的表现。** GPT-4在所有类别上优于早期ChatGPT版本。')
w()
w('GPT-4在TruthfulQA[66]上取得进展（图7）。GPT-4基础模型仅略好于GPT-3.5；但RLHF后训练后大幅提升。')
w()
w('**图7：GPT-4在TruthfulQA上的表现。** 比较零样本、少样本和RLHF微调后。GPT-4 RLHF显著优于GPT-3.5和Anthropic-LM。')
w()
w('**图8：校准曲线。** 左：预训练GPT-4（ECE 0.007，接近完美校准）。右：后训练PPO模型（ECE 0.074，校准显著降低）。')
w()
w('GPT-4通常缺乏2021年9月后事件的知识，不从经验学习，可能犯简单推理错误或过于轻信。')
w()

# ========== 6 RISKS ==========
w('## 6 风险与缓解措施')
w()
w('### 对抗性测试：领域专家')
w('邀请了50多位来自AI对齐、网络安全、生物风险等领域的专家进行对抗性测试。他们的发现用于改进模型，如收集额外数据以改进危险化学品合成请求的拒绝（表5）。')
w()
w('### 模型辅助安全流程')
w('使用RLHF[40,63]微调。额外使用基于规则的奖励模型（RBRMs）——零样本GPT-4分类器——在RLHF微调期间提供额外奖励信号。RBRM根据人工编写的规则分类输出（如期望风格的拒绝、包含不允许内容、安全非拒绝等）。')
w()
w('**安全指标改进：** 相比GPT-3.5，不允许内容响应率降低82%；敏感请求合规率提高29%；RealToxicityPrompts有毒生成率从6.48%降至0.73%。')
w()
w('### 总结')
w('模型级干预增加了不良行为难度但仍有"越狱"可能。需与部署时安全技术（滥用监控、快速迭代）互补。')
w()

# ========== 7 CONCLUSION ==========
w('## 7 结论')
w()
w('GPT-4是一个大型多模态模型，在困难专业和学术基准上具有人类水平性能。在一系列NLP任务上优于现有LLM和大多数SOTA系统。改进能力可在多种语言中展示。可预测扩展使准确预测成为可能。GPT-4因增强能力带来新风险，我们讨论了一些安全和对齐方法。GPT-4代表了向广泛有用且安全部署的AI系统迈出的重要一步。')
w()

# ========== AUTHORSHIP ==========
w('---')
w('## 作者署名与致谢')
w()
w('请引用本文为"OpenAI (2023)"。')
w()
w('**预训练核心贡献者：** Christopher Berner(超级计算), Greg Brockman(基础设施), Trevor Cai(吞吐量), David Farhi, Chris Hesse, Shantanu Jain, Kyle Kosic, Jakub Pachocki(总体/优化), Alex Paino, Mikhail Pavlov, Michael Petrov, Nick Ryder(架构/数据), Szymon Sidor, Nikolas Tezak(执行), Phil Tillet(Triton), Amin Tootoonchian(模型分发/系统/网络), Qiming Yuan(数据集), Wojciech Zaremba(数据团队)')
w('**长上下文：** Gabriel Goh, Łukasz Kaiser, Ben Wang, Clemens Winter')
w('**视觉核心贡献者：** Trevor Cai, Mark Chen, Casey Chu, Chris Hesse, Shengli Hu, Yongjik Kim, Jamie Kiros, Daniel Levy, Christine McLeavey, David Mély, Hyeonwoo Noh, Mikhail Pavlov, Raul Puri, Amin Tootoonchian')
w('**RLHF核心贡献者：** Greg Brockman, Arka Dhar, Liam Fedus, Tarun Gogineni, Rapha Gontijo-Lopes, Joshua Gross, Johannes Heidecke, Joost Huizinga, Teddy Lee, Jan Leike, Ryan Lowe, Luke Metz, Long Ouyang, John Schulman(总体), Jerry Tworek, Carroll Wainwright, Jonathan Ward, Jiayi Weng, Sarah Yoo, Wojciech Zaremba, Chong Zhang, Shengjia Zhao(奖励模型), Barret Zoph(训练)')
w('**评估核心贡献者：** Sandhini Agarwal, Lama Ahmad, Mo Bavarian, Tyna Eloundou, Andrew Kondrich, Gretchen Krueger, Michael Lampe, Pamela Mishkin, Benjamin Sokolowsky, Jack Rae, Chelsea Voss, Alvin Wang, Kai Xiao, Marvin Zhang')
w('**部署核心贡献者：** Steven Adler, Sandhini Agarwal, Derek Chen, Atty Eleti, Joanne Jang, Angela Jiang, Tomer Kaftan, Rachel Lim, Kim Malfacini, Bianca Martin, Evan Morikawa, Henrique Ponde de Oliveira Pinto, Heather Schmidt, Maddie Simens, Felipe Petroski Such, Andrea Vallone, Lilian Weng, Dave Willner, Michael Wu')
w('**额外贡献：** Sam Altman, Katie Mayer, Bob McGrew, Mira Murati, Ilya Sutskever, Peter Welinder')
w('感谢所有OpenAI团队成员。感谢微软（Azure、Bing和安全团队）的合作。感谢专家对抗性测试者和红队成员。')
w()

# ========== REFERENCES ==========
w('---')
w('## 参考文献')
w()
refs = [
'[1] Brown et al. Language models are few-shot learners. NeurIPS, 2020.',
'[2] Hoffmann et al. Training compute-optimal large language models. arXiv:2203.15556, 2022.',
'[3] Chowdhery et al. PaLM: Scaling language modeling with pathways. arXiv:2204.02311, 2022.',
'[4] Rae et al. Scaling language models: Methods, analysis & insights from training gopher. arXiv:2112.11446, 2021.',
'[5] Dai et al. Transformer-XL. arXiv:1901.02860, 2019.',
'[6] Liu et al. RoBERTa. arXiv:1907.11692, 2019.',
'[7] Devlin et al. BERT. arXiv:1810.04805, 2018.',
'[8] Raffel et al. Exploring the limits of transfer learning with a unified text-to-text transformer. arXiv:1910.10683, 2019.',
'[9] Shazeer and Stern. Adafactor. arXiv:1804.04235, 2018.',
'[10] Ba et al. Layer normalization. arXiv:1607.06450, 2016.',
'[11] Wei et al. Chain-of-thought prompting elicits reasoning in large language models. NeurIPS, 2022.',
'[12] Huang et al. Large language models can self-improve. arXiv:2210.11610, 2022.',
'[13] Kojima et al. Large language models are zero-shot reasoners. arXiv:2205.11916, 2022.',
'[14] Kaplan et al. Scaling laws for neural language models. arXiv:2001.08361, 2020.',
'[15] Henighan et al. Scaling laws for autoregressive generative modeling. arXiv:2010.14701, 2020.',
'[16] Yang et al. Tensor Programs V. arXiv:2203.03466, 2022.',
'[17] Shazeer et al. Outrageously large neural networks. arXiv:1701.06538, 2017.',
'[18] Zoph et al. ST-MoE. arXiv:2202.08906, 2022.',
'[19] Wei et al. Emergent abilities of large language models. TMLR, 2022.',
'[20] Dehghani et al. Universal transformers. ICLR, 2019.',
'[21] Su et al. RoFormer. arXiv:2104.09864, 2021.',
'[22] Alayrac et al. Flamingo. NeurIPS.',
'[23] Chen et al. PaLI. arXiv:2209.06794, 2022.',
'[24] Wang and Komatsuzaki. GPT-J-6B. 2021.',
'[25] Black et al. GPT-Neo. 2021.',
'[26] Le Scao et al. Bloom. arXiv:2211.05100, 2022.',
'[27] Zhang et al. OPT. arXiv:2205.01068, 2022.',
'[28] Touvron et al. LLaMA. arXiv:2302.13971, 2023.',
'[29] Radford et al. Learning to generate reviews and discovering sentiment. arXiv:1704.01444, 2017.',
'[30] Lample and Conneau. Cross-lingual language model pretraining. arXiv:1901.07291, 2019.',
'[31] Dao et al. Flashattention. arXiv:2205.14135, 2022.',
'[32] Child et al. Generating long sequences with sparse transformers. arXiv:1904.10509, 2019.',
'[33] Rabe and Staats. Self-attention does not need o(n2) memory. arXiv:2112.05682, 2021.',
'[34] Gray et al. GPU kernels for block-sparse weights. 2017.',
'[35] Hendrycks et al. Measuring massive multitask language understanding. ICLR, 2021.',
'[36] Hendrycks et al. Aligning AI with shared human values. ICLR, 2021.',
'[37] Radford et al. Language models are unsupervised multitask learners. 2019.',
'[38] Radford et al. Improving language understanding by generative pre-training. 2018.',
'[39] Vaswani et al. Attention is all you need. NeurIPS, 2017.',
'[40] Christiano et al. Deep reinforcement learning from human preferences. NeurIPS, 2017.',
'[41] Hestness et al. Deep learning scaling is predictable, empirically. arXiv:1712.00409, 2017.',
'[42] Thompson et al. The computational limits of deep learning. arXiv:2007.05558, 2020.',
'[43] Chen et al. Evaluating large language models trained on code. 2021.',
'[44] McKenzie et al. The Inverse Scaling Prize. 2022.',
'[45] Wei et al. Inverse scaling can become U-shaped. arXiv:2211.02011, 2022.',
'[46] McKenzie et al. Inverse Scaling Prize: First round winners. 2022.',
'[47] Brockman et al. OpenAI API. 2020.',
'[48] Srivastava et al. Beyond the imitation game. arXiv:2206.04615, 2022.',
'[49] Hendrycks et al. Measuring massive multitask language understanding. arXiv:2009.03300, 2020.',
'[50] Tay et al. Transcending scaling laws with 0.1% extra compute. arXiv:2210.11399, 2022.',
'[51] Chung et al. Scaling instruction-finetuned language models. arXiv:2210.11416, 2022.',
'[52] Zellers et al. HellaSwag. ACL, 2019.',
'[53] Liu et al. Adversarial training for large neural language models. arXiv:2004.08994, 2020.',
'[54] Clark et al. Think you have solved question answering? Try ARC. 2018.',
'[55] Wang et al. Self-consistency improves chain of thought reasoning. arXiv:2203.11171, 2022.',
'[56] Sakaguchi et al. WinoGrande. arXiv:1907.10641, 2019.',
'[57] Chen et al. CodeT. arXiv:2207.10397, 2022.',
'[58] Dua et al. DROP. NAACL, 2019.',
'[59] Chen et al. Question directed graph attention network. arXiv:2009.07448, 2020.',
'[60] Cobbe et al. Training verifiers to solve math word problems. arXiv:2110.14168, 2021.',
'[61] Lewkowycz et al. Solving quantitative reasoning problems with language models. arXiv:2206.14858, 2022.',
'[62] Uesato et al. Solving math word problems with process- and outcome-based feedback. arXiv:2211.14275, 2022.',
'[63] Ouyang et al. Training language models to follow instructions with human feedback. arXiv:2203.02155, 2022.',
'[64] OpenAI. Introducing ChatGPT. 2022.',
'[65] OpenAI. GPT-4. 2023.',
'[66] Lin et al. TruthfulQA. ACL, 2022.',
'[67] Bai et al. Training a helpful and harmless assistant. arXiv:2204.05862, 2022.',
'[68] OpenAI. How should AI systems behave? 2023.',
'[69] Leike et al. Our approach to alignment research. 2022.',
'[70] Carlsmith. Is power-seeking AI an existential risk? arXiv:2206.13353, 2022.',
'[71] Glaese et al. Improving alignment of dialogue agents. arXiv:2209.14375, 2022.',
'[72] Perez et al. Red teaming language models with language models. arXiv:2202.03286, 2022.',
'[73] Gehman et al. RealToxicityPrompts. arXiv:2009.11462, 2020.',
]
for r in refs:
    w(r)
w()

# Write to file
with open("/Users/dazhang/PycharmProject/Papers/2-LLM/2024-GPT-4 Technical Report-翻译.md", "w", encoding="utf-8") as f:
    f.write('\n'.join(out))

print(f"Translation written to output file: {len(out)} lines")
