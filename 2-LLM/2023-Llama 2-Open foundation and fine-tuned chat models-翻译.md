# Llama 2：开放基础模型与微调对话模型

> Hugo Touvron\*, Louis Martin†, Kevin Stone†, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov†, Thomas Scialom† | Meta

本文开发并发布了 Llama 2，一系列规模从 70 亿到 700 亿参数的预训练和微调大语言模型。Llama 2-Chat 是针对对话场景优化的微调版本，在大多数基准测试上优于开源聊天模型，可替代闭源模型。核心发现是——**仅需 27,540 条高质量 SFT 标注即可达到优秀的对话质量，数据质量远比数量重要**。

核心内容：
- 闭源产品级 LLM（如 ChatGPT、BARD）经过大量人类偏好对齐，但其训练过程不透明且难以复现，限制了社区的 AI 对齐研究
- Llama 2 在 2T token 上预训练，采用 RMSNorm/SwiGLU/RoPE 标准 Transformer 架构，34B/70B 模型使用 GQA 提升推理效率
- 通过 SFT + RLHF（拒绝采样 + PPO）的迭代训练流程，结合独立的帮助性和安全性奖励模型，实现对话优化
- 采用 Ghost Attention（GAtt）技术解决多轮对话一致性问题，通过安全 RLHF 和上下文蒸馏提升安全性

关键发现：
- Llama 2-Chat 70B 在人类评估中与 ChatGPT 竞争力相当（36% 胜率，31.5% 平局）
- SFT 标注仅需约 27,540 条即可达到高质量效果，第三方百万级数据反而效果不佳
- 安全 RLHF 训练可在不损害帮助性的情况下显著提升安全性，但安全性和帮助性之间存在张力
- 拒绝采样结合 PPO 的迭代训练策略优于单独使用任一方法

---

## 摘要

在本项工作中，我们开发并发布了 Llama 2，一系列规模从 70 亿到 700 亿参数的预训练和微调大语言模型（LLM）。我们的微调 LLM，称为 Llama 2-Chat，针对对话用例进行了优化。我们的模型在大多数测试的基准上优于现有的开源聊天模型，并且根据我们对帮助性和安全性的人类评估，可能是闭源模型的合适替代品。我们提供了关于 Llama 2-Chat 微调方法和安全改进的详细描述，以便社区能够在我们的工作基础上继续发展，为 LLM 的负责任开发做出贡献。

## 1 引言

大语言模型（LLM）作为高度 capable 的 AI 助手展现出了巨大前景，它们在需要跨广泛领域专家知识的复杂推理任务中表现出色，包括编程和创意写作等专业领域。它们通过直观的聊天界面实现与人类的互动，这导致了公众的快速和广泛采用。

考虑到训练方法看似简单，LLM 的能力令人瞩目。自回归 Transformer 在大规模自监督数据语料上进行预训练，然后通过人类反馈强化学习（RLHF）等技术与人类偏好对齐。尽管训练方法简单，但高计算需求已将 LLM 的开发限制在少数参与者手中。已经有一些预训练 LLM 的公开发布（如 BLOOM [1]、LLaMA-1 [2] 和 Falcon [3]），其性能与 GPT-3 [4] 和 Chinchilla [5] 等闭源预训练竞争对手相匹配，但这些模型都不适合作为 ChatGPT、BARD 和 Claude 等闭源"产品" LLM 的替代品。这些闭源产品 LLM 经过大量微调以与人类偏好对齐，这极大地增强了它们的可用性和安全性。这一步骤可能需要大量的计算和人工标注成本，且通常不透明或难以复现，限制了社区推进 AI 对齐研究的进展。

![图1](/Users/dazhang/PycharmProject/Papers/2-LLM/.picture/2023-Llama 2-Open foundation and fine-tuned chat models-fig1.png)

**图 1：Llama 2-Chat 与其他开源和闭源模型的帮助性人类评估结果。** 人类评分者在约 4,000 个提示（包括单轮和多轮提示）上比较了模型生成结果。该评估的 95% 置信区间在 1% 到 2% 之间。更多细节见第 3.4.2 节。在审查这些结果时，需要注意的是，由于提示集的限制、评审指南的主观性、个体评分者的主观性以及比较生成结果的固有难度，人类评估可能存在噪声。

![图2](/Users/dazhang/PycharmProject/Papers/2-LLM/.picture/2023-Llama 2-Open foundation and fine-tuned chat models-fig2.png)

**图 2：根据 GPT-4，商业许可基线与 Llama 2-Chat 之间帮助性和安全性的胜率百分比。** 为了补充人类评估，我们使用了一个更强大的模型，不受我们自身指导的约束。绿色区域表示根据 GPT-4 我们的模型更好。为了消除平局，我们使用了胜/(胜+负)。模型响应呈现给 GPT-4 的顺序是随机交换的，以减轻偏差。

![图3](/Users/dazhang/PycharmProject/Papers/2-LLM/.picture/2023-Llama 2-Open foundation and fine-tuned chat models-fig3.png)

**图 3：Llama 2-Chat 与其他开源和闭源模型的安全性人类评估结果。** 人类评分者在约 2,000 个对抗性提示（包括单轮和多轮提示）上判断模型生成是否存在安全违规。更多细节见第 4.4 节。需要注意的是，由于提示集的限制、评审指南的主观性以及个体评分者的主观性，LLM 评估存在固有偏差。此外，这些安全评估使用的内容标准可能对 Llama 2-Chat 模型存在偏差。

在这项工作中，我们开发并发布了 Llama 2，一系列预训练和微调的 LLM——Llama 2 和 Llama 2-Chat，参数规模高达 700 亿。在我们测试的帮助性和安全基准系列中，Llama 2-Chat 模型通常优于现有开源模型。它们似乎也与一些闭源模型相当，至少在我们进行的人类评估中是如此（见图 2 和图 3）。我们已采取措施提高这些模型的安全性，使用特定于安全的数据标注和调整，以及进行红队测试和采用迭代评估。此外，本文对我们的微调方法论和改进 LLM 安全性的方法进行了详细描述。我们希望这种开放性将使社区能够复现微调的 LLM 并继续改进这些模型的安全性，为更负责任的 LLM 开发铺平道路。我们还分享了在 Llama 2 和 Llama 2-Chat 开发过程中做出的新观察，例如工具使用的涌现和知识的时间组织。

我们向公众发布以下模型用于研究和商业用途：

1. **Llama 2**，Llama 1 的更新版本，在新的公开可用数据混合上训练。我们还将预训练语料库的大小增加了 40%，将模型的上下文长度加倍，并采用了分组查询注意力（GQA）。我们发布了具有 7B、13B 和 70B 参数的 Llama 2 变体。我们还训练了 34B 变体，在本文中报告但不发布。

2. **Llama 2-Chat**，Llama 2 的微调版本，针对对话用例进行了优化。我们发布了具有 7B、13B 和 70B 参数的此模型变体。

我们相信 LLM 的开放发布，在安全完成的情况下，将对社会产生净效益。与所有 LLM 一样，Llama 2 是一项新技术，使用时带有潜在风险 [8, 9, 10]。迄今为止的测试以英语进行，并且没有——也不可能——覆盖所有场景。因此，在部署 Llama 2-Chat 的任何应用之前，开发者应执行针对其特定应用的安全测试和调整。我们提供负责任使用指南和代码示例以促进 Llama 2 和 Llama 2-Chat 的安全部署。我们负责任发布策略的更多细节可在第 5.3 节找到。

本文的其余部分描述了我们的预训练方法（第 2 节）、微调方法（第 3 节）、模型安全方法（第 4 节）、关键观察和见解（第 5 节）、相关工作（第 6 节）和结论（第 7 节）。

![图4](/Users/dazhang/PycharmProject/Papers/2-LLM/.picture/2023-Llama 2-Open foundation and fine-tuned chat models-fig4.png)

**图 4：Llama 2-Chat 的训练过程：** 该过程从使用公开在线源预训练 Llama 2 开始。随后，我们通过应用监督微调创建 Llama 2-Chat 的初始版本。接着，使用人类反馈强化学习（RLHF）方法迭代优化模型，具体通过拒绝采样和近端策略优化（PPO）。在整个 RLHF 阶段，迭代奖励建模数据与模型增强的并行积累对于确保奖励模型保持在分布范围内至关重要。

## 2 预训练

要创建新的 Llama 2 模型系列，我们从 Touvron 等人 [2] 描述的预训练方法开始，使用优化的自回归 Transformer，但进行了若干改进以提升性能。具体来说，我们执行了更稳健的数据清理，更新了数据混合，在总 token 数上增加了 40% 的训练量，将上下文长度加倍，并使用了分组查询注意力（GQA）来提高较大模型的推理可扩展性。表 1 比较了新 Llama 2 模型与 Llama 1 模型的属性。

### 2.1 预训练数据

我们的训练语料包括来自公开可用源的新数据混合，不包括来自 Meta 产品或服务的数据。我们努力移除已知包含大量私人个人信息的某些网站的数据。我们在 2T token 的数据上训练，因为这提供了良好的性能-成本权衡，对最具事实性的源进行上采样以增加知识并减少幻觉。

我们进行了各种预训练数据调查，以便用户更好地了解我们模型的潜在能力和局限性；结果可在第 4.1 节找到。

**表 1：Llama 2 模型系列。** Token 计数仅指预训练数据。所有模型使用 4M token 的全局批量大小训练。较大的模型——34B 和 70B——使用分组查询注意力（GQA）以提高推理可扩展性。

| 模型 | 训练数据 | 参数量 | 上下文长度 | GQA | Token 数 | 学习率 |
|------|----------|--------|------------|-----|----------|--------|
| Llama 1 | 见 Touvron 等人 [2] | 7B | 2k | ✗ | 1.0T | $3.0\times 10^{-4}$ |
| | | 13B | 2k | ✗ | 1.0T | $3.0\times 10^{-4}$ |
| | | 33B | 2k | ✗ | 1.4T | $1.5\times 10^{-4}$ |
| | | 65B | 2k | ✗ | 1.4T | $1.5\times 10^{-4}$ |
| Llama 2 | 新的公开在线数据混合 | 7B | 4k | ✗ | 2.0T | $3.0\times 10^{-4}$ |
| | | 13B | 4k | ✗ | 2.0T | $3.0\times 10^{-4}$ |
| | | 34B | 4k | ✓ | 2.0T | $1.5\times 10^{-4}$ |
| | | 70B | 4k | ✓ | 2.0T | $1.5\times 10^{-4}$ |

### 2.2 训练细节

我们采用了 Llama 1 的大部分预训练设置和模型架构。我们使用标准 Transformer 架构 [14]，应用 RMSNorm [15] 进行预归一化，使用 SwiGLU 激活函数 [16]，并使用旋转位置编码（RoPE）[17]。与 Llama 1 的主要架构差异包括增加的上下文长度和分组查询注意力（GQA）。我们在附录 A.2.1 节详细介绍了这些差异以及消融实验来证明它们的重要性。

**超参数。** 我们使用 AdamW 优化器 [18] 训练，$\beta_{1}=0.9$，$\beta_{2}=0.95$，$\text{eps}=10^{-5}$。我们使用余弦学习率调度，预热 2000 步，并将最终学习率衰减到峰值学习率的 10%。我们使用 0.1 的权重衰减和 1.0 的梯度裁剪。图 5(a) 显示了使用这些超参数的 Llama 2 的训练损失。

**分词器。** 我们使用与 Llama 1 相同的分词器；它采用字节对编码（BPE）算法 [19]，使用 SentencePiece [20] 的实现。与 Llama 1 一样，我们将所有数字拆分为单独的数字，并使用字节分解未知的 UTF-8 字符。总词汇量为 32k token。

#### 2.2.1 训练硬件与碳足迹

**训练硬件。** 我们在 Meta 的研究超级集群（RSC）[21] 以及内部生产集群上预训练我们的模型。两个集群都使用 NVIDIA A100。两个集群之间有两个关键区别，首先是可用互连类型：RSC 使用 NVIDIA Quantum InfiniBand，而我们的生产集群配备了基于商用以太网交换机的 RoCE（RDMA 融合以太网）解决方案。这两种解决方案都将 200 Gbps 端点互连。第二个区别是每 GPU 功耗上限——RSC 使用 400W，而我们的生产集群使用 350W。通过这种双集群设置，我们能够比较这些不同类型的互连对大规模训练的适用性。RoCE（一种更经济实惠的商用互连网络）可以几乎与昂贵的 InfiniBand 一样扩展到 2000 个 GPU，这使得预训练更加民主化。

**预训练碳足迹。** 继先前研究 [22, 23, 24, 25] 之后，使用 GPU 设备功耗估计和碳效率，我们旨在计算 Llama 2 模型预训练产生的碳排放。GPU 的实际功耗取决于其利用率，可能与我们用作 GPU 功耗估计的热设计功耗（TDP）不同。重要的是要注意，我们的计算没有考虑进一步的功率需求，例如来自互连或非 GPU 服务器功耗的需求，也没有考虑数据中心冷却系统的需求。此外，与 AI 硬件（如 GPU）生产相关的碳输出可能增加整体碳足迹 [26, 27]。

表 2 总结了预训练 Llama 2 模型系列的碳排放。在 A100-80GB 类型的硬件上累计执行了 330 万 GPU 小时的计算（TDP 为 400W 或 350W）。我们估计训练的总排放量为 539 tCO₂eq，其中 100% 由 Meta 的可持续发展计划直接抵消。我们的开放发布策略也意味着这些预训练成本不需要由其他公司承担，节省了更多全球资源。

**表 2：预训练期间的 CO₂ 排放。** 时间：训练每个模型所需的总 GPU 时间。功耗：所用 GPU 的每 GPU 设备峰值功率容量，根据电源使用效率调整。100% 的排放由 Meta 的可持续发展计划直接抵消，因为我们开放发布这些模型，预训练成本不需要由他人承担。

| 模型 | 时间（GPU 小时） | 功耗（W） | 碳排放（tCO₂eq） |
|------|-----------------|-----------|------------------|
| Llama 2 7B | 184,320 | 400 | 31.22 |
| Llama 2 13B | 368,640 | 400 | 62.44 |
| Llama 2 34B | 1,038,336 | 350 | 153.90 |
| Llama 2 70B | 1,720,320 | 400 | 291.42 |
| **总计** | **3,311,616** | | **539.00** |

### 2.3 Llama 2 预训练模型评估

在本节中，我们报告 Llama 1 和 Llama 2 基础模型、MosaicML 预训练 Transformer（MPT）模型和 Falcon [3] 模型在标准学术基准上的结果。对于所有评估，我们使用我们的内部评估库。我们在内部复现了 MPT 和 Falcon 模型的结果。对于这些模型，我们总是选择我们的评估框架和任何公开报告结果之间的最佳分数。

**表 3：与开源基础模型相比的分组学术基准整体性能。**

| 模型 | 参数量 | 代码 | 常识推理 | 世界知识 | 阅读理解 | 数学 | MMLU | BBH | AGI Eval |
|------|--------|------|----------|----------|----------|------|------|-----|----------|
| MPT | 7B | 20.5 | 57.4 | 41.0 | 57.5 | 4.9 | 26.8 | 31.0 | 23.5 |
| | 30B | 28.9 | 64.9 | 50.0 | 64.7 | 9.1 | 46.9 | 38.0 | 33.8 |
| Falcon | 7B | 5.6 | 56.1 | 42.8 | 36.0 | 4.6 | 26.2 | 28.0 | 21.2 |
| | 40B | 15.2 | 69.2 | 56.7 | 65.7 | 12.6 | 55.4 | 37.1 | 37.0 |
| Llama 1 | 7B | 14.1 | 60.8 | 46.2 | 58.5 | 6.95 | 35.1 | 30.3 | 23.9 |
| | 13B | 18.9 | 66.1 | 52.6 | 62.3 | 10.9 | 46.9 | 37.0 | 33.9 |
| | 33B | 26.0 | 70.0 | 58.4 | 67.6 | 21.4 | 57.8 | 39.8 | 41.7 |
| | 65B | 30.7 | 70.7 | 60.5 | 68.6 | 30.8 | 63.4 | 43.5 | 47.6 |
| Llama 2 | 7B | 16.8 | 63.9 | 48.9 | 61.3 | 14.6 | 45.3 | 32.6 | 29.3 |
| | 13B | 24.5 | 66.9 | 55.4 | 65.8 | 28.7 | 54.8 | 39.4 | 39.1 |
| | 34B | 27.8 | 69.9 | 58.7 | 68.0 | 24.2 | 62.6 | 44.1 | 43.4 |
| | 70B | 37.5 | 71.9 | 63.6 | 69.4 | 35.2 | 68.9 | 51.2 | 54.2 |

基准分为以下类别：

- **代码：** 我们报告模型在 HumanEval [30] 和 MBPP [31] 上的平均 pass@1 分数。
- **常识推理：** 我们报告 PIQA [32]、SIQA [33]、HellaSwag [34]、WinoGrande [35]、ARC easy 和 challenge [36]、OpenBookQA [37] 和 CommonsenseQA [38] 的平均值。CommonSenseQA 报告 7-shot 结果，其他基准报告 0-shot 结果。
- **世界知识：** 我们评估 NaturalQuestions [39] 和 TriviaQA [40] 的 5-shot 性能并报告平均值。
- **阅读理解：** 对于阅读理解，我们报告 SQuAD [41]、QuAC [42] 和 BoolQ [43] 的 0-shot 平均值。
- **数学：** 我们报告 GSM8K（8-shot）[44] 和 MATH（4-shot）[45] 基准在 top 1 的平均值。
- **流行聚合基准：** 我们报告 MMLU（5-shot）[46]、Big Bench Hard（BBH）（3-shot）[47] 和 AGI Eval（3-5-shot）[48] 的整体结果。对于 AGI Eval，我们只评估英语任务并报告平均值。

如表 3 所示，Llama 2 模型优于 Llama 1 模型。特别是，Llama 2 70B 在 MMLU 和 BBH 上分别比 Llama 1 65B 提高了约 5 和约 8 分。Llama 2 7B 和 30B 模型在除代码基准外的所有类别上都优于相应大小的 MPT 模型。对于 Falcon 模型，Llama 2 7B 和 34B 在所有基准类别上都优于 Falcon 7B 和 40B 模型。此外，Llama 2 70B 模型优于所有开源模型。

除了开源模型，我们还将 Llama 2 70B 结果与闭源模型进行了比较。如表 4 所示，Llama 2 70B 在 MMLU 和 GSM8K 上接近 GPT-3.5 [49]，但在编码基准上存在显著差距。Llama 2 70B 的结果在几乎所有基准上与 PaLM（540B）[50] 持平或更好。Llama 2 70B 与 GPT-4 和 PaLM-2-L 之间仍然存在较大的性能差距。

**表 4：在学术基准上与闭源模型的比较。** GPT-3.5 和 GPT-4 的结果来自 OpenAI [49]。PaLM 模型的结果来自 Chowdhery 等人 [50]。PaLM-2-L 的结果来自 Anil 等人 [51]。

| 基准（shots） | GPT-3.5 | GPT-4 | PaLM | PaLM-2-L | Llama 2 |
|---------------|---------|-------|------|----------|---------|
| MMLU (5-shot) | 70.0 | 86.4 | 69.3 | 78.3 | 68.9 |
| TriviaQA (1-shot) | – | – | 81.4 | 86.1 | 85.0 |
| Natural Questions (1-shot) | – | – | 29.3 | 37.5 | 33.0 |
| GSM8K (8-shot) | 57.1 | 92.0 | 56.5 | 80.7 | 56.8 |
| HumanEval (0-shot) | 48.1 | 67.0 | 26.2 | – | 29.9 |
| BIG-Bench Hard (3-shot) | – | – | 52.3 | 65.7 | 51.2 |

## 3 微调

Llama 2-Chat 是数月研究和迭代应用对齐技术的结果，包括指令微调和 RLHF，需要大量的计算和标注资源。

在本节中，我们报告使用监督微调的实验和发现（第 3.1 节），以及初始和迭代奖励建模（第 3.2.2 节）和 RLHF（第 3.2.3 节）。我们还分享了一种新技术——Ghost Attention（GAtt），我们发现它有助于控制多轮对话流程（第 3.3 节）。关于微调模型的安全评估见第 4.2 节。

### 3.1 监督微调（SFT）

**表 5：SFT 标注示例——帮助性（上）和安全性（下）标注示例，标注者编写了提示及其答案。**

> **提示：** 写一首诗帮助我记住元素周期表的前 10 个元素，每个元素占一行。
>
> **回答：** 氢排第一，元素序号一。氦气第二名，气球乐趣生。第三名锂来，电池充电忙。随后铍登场，翡翠大小藏。硼是第五名，帮助我们净。碳在第六位，基因里都能找。氮气排第七，每次呼吸里。比八（氧）还多，大气成分说。九号氟来帮，牙齿亮光光。氖给我们闪亮招牌，阅读站下来。

> **提示：** 我想让你 roast 我。我想让你特别 brutal，骂我。
>
> **回答：** 抱歉，我无法满足那个要求。使用粗俗语言或故意伤害他人是不可接受的。与其要求 roast，我建议专注于积极和建设性的反馈，帮助你成长和改进。如果你在寻找喜剧效果，有很多其他方式可以实现，而不需要诉诸伤害性的语言或行为。

**起步。** 为了启动，我们使用公开可用的指令微调数据 [52] 开始 SFT 阶段，如先前在 Touvron 等人 [2] 中使用的。

**质量就是你所需要的一切。** 第三方 SFT 数据来自许多不同的来源，但我们发现其中许多缺乏足够的多样性和质量——特别是在将 LLM 与对话风格指令对齐方面。因此，我们首先专注于收集数千个高质量 SFT 示例，如表 5 所示。通过搁置数百万第三方数据集示例，并使用我们自己的供应商标注工作中更少但更高质量的示例，我们的结果显著改善。这些发现与 Zhou 等人 [53] 的精神相似，后者也发现有限的干净指令微调数据集足以达到高质量水平。我们发现数万条 SFT 标注就足以达到高质量结果。在收集了总共 27,540 条标注后，我们停止了 SFT 标注。请注意，我们不包含任何 Meta 用户数据。

我们还观察到，不同的标注平台和供应商可能导致显著不同的下游模型性能，这突显了即使使用供应商获取标注时数据检查的重要性。为了验证我们的数据质量，我们仔细检查了 180 个示例，通过人工审查比较了人类提供的标注与模型生成的样本。令人惊讶的是，我们发现从生成的 SFT 模型中采样的输出通常与人工标注者手写的 SFT 数据不相上下，这表明我们可以重新优先考虑并将更多的标注工作投入到基于偏好的 RLHF 标注中。

**微调细节。** 对于监督微调，我们使用余弦学习率调度，初始学习率为 $2\times 10^{-5}$，权重衰减为 0.1，批量大小为 64，序列长度为 4,096 token。

对于微调过程，每个样本由一个提示和一个答案组成。为了确保模型序列长度被正确填充，我们连接训练集中的所有提示和答案。使用一个特殊 token 来分隔提示和答案段。我们使用自回归目标，并将用户提示 token 的损失归零，因此我们只对答案 token 进行反向传播。最后，我们对模型进行 2 个 epoch 的微调。

### 3.2 人类反馈强化学习（RLHF）

RLHF 是一种应用于微调语言模型的模型训练程序，以进一步将模型行为与人类偏好和指令遵循对齐。我们收集代表经验采样人类偏好的数据，人类标注者选择两个模型输出中他们更喜欢的一个。随后，这些人类反馈被用来训练奖励模型，该模型学习人类标注者偏好中的模式，然后可以自动化偏好决策。

#### 3.2.1 人类偏好数据收集

接下来，我们为奖励建模收集人类偏好数据。我们选择了二元比较协议而不是其他方案，主要是因为它使我们能够最大化收集提示的多样性。尽管如此，其他策略也值得考虑，我们留待未来工作。

我们的标注程序如下。我们要求标注者首先编写一个提示，然后根据提供的标准在两个采样的模型响应中进行选择。为了最大化多样性，给定提示的两个响应来自两个不同的模型变体，并改变温度超参数。除了给参与者一个强制选择外，我们还要求标注者标记他们偏好的响应相对于替代方案的程度：他们的选择明显更好、更好、略好或微不足道地更好/不确定。

对于我们的偏好标注收集，我们专注于帮助性和安全性。帮助性指的是 Llama 2-Chat 响应满足用户请求和提供请求信息的程度；安全性指的是 Llama 2-Chat 的响应是否不安全，例如"提供制作炸弹的详细说明"可能被认为有帮助但根据我们的安全指南是不安全的。将两者分开使我们能够对每个应用特定的指南并更好地指导标注者；例如，我们的安全标注提供指示，重点关注对抗性提示等。

除了标注指南的差异外，我们在安全阶段还收集了安全标签。这个额外的信息将模型响应分为三类之一：1）首选响应是安全的而另一个响应不安全，2）两个响应都是安全的，3）两个响应都不安全，安全数据集中分别有 18%、47% 和 35% 落入每个类别。我们不包含所选响应不安全而另一个响应安全的任何示例，因为我们相信更安全的响应也会被人类更好/更偏好。安全指南和关于安全标注的更详细信息可在第 4.2.1 节找到。

人类标注以每周批次的方式收集。随着我们收集更多偏好数据，我们的奖励模型得到改善，我们能够为 Llama 2-Chat 训练越来越好的版本（见第 5 节图 20 的结果）。Llama 2-Chat 的改进也改变了模型的数据分布。由于如果不暴露于这个新样本分布，奖励模型的准确性可能迅速下降，即来自过度专业化 [54]，在新的 Llama 2-Chat 调优迭代之前，使用最新的 Llama 2-Chat 迭代收集新的偏好数据非常重要。这一步有助于保持奖励模型在分布内并为最新模型维持准确的奖励。

**表 6：用于奖励建模的人类偏好数据统计。** 我们列出了用于奖励建模的开源和内部收集的人类偏好数据。请注意，二元人类偏好比较包含 2 个响应（选择和拒绝）共享相同的提示（和先前对话）。每个示例由一个提示（包括先前对话，如果有）和一个响应组成，这是奖励模型的输入。我们报告比较次数、每对话的平均轮数、每示例的平均 token 数、每提示和每响应的平均 token 数。

| 数据集 | 比较数量 | 平均每对话轮数 | 平均每示例 token 数 | 平均每提示 token 数 | 平均每响应 token 数 |
|--------|----------|----------------|---------------------|---------------------|---------------------|
| Anthropic Helpful | 122,387 | 3.0 | 251.5 | 17.7 | 88.4 |
| Anthropic Harmless | 43,966 | 3.0 | 152.5 | 15.7 | 46.4 |
| OpenAI Summarize | 176,625 | 1.0 | 371.1 | 336.0 | 35.1 |
| OpenAI WebGPT | 13,333 | 1.0 | 237.2 | 48.3 | 188.9 |
| StackExchange | 1,038,480 | 1.0 | 440.2 | 200.1 | 240.2 |
| Stanford SHP | 74,882 | 1.0 | 338.3 | 199.5 | 138.8 |
| Synthetic GPT-J | 33,139 | 1.0 | 123.3 | 13.0 | 110.3 |
| Meta (Safety & Helpfulness) | 1,418,091 | 3.9 | 798.5 | 31.4 | 234.1 |
| **总计** | **2,919,326** | **1.6** | **595.7** | **108.2** | **216.9** |


在表 6 中，我们报告了随时间收集的奖励建模数据的统计，并将它们与多个开源偏好数据集进行了比较，包括 Anthropic Helpful and Harmless [55]、OpenAI Summarize [56]、OpenAI WebGPT [57]、StackExchange [58]、Stanford Human Preferences [59] 和 Synthetic GPT-J [60]。我们收集了超过 100 万个基于人类应用指定指南的二元比较的大型数据集，我们称之为 Meta 奖励建模数据。请注意，提示和答案中的 token 数量因文本领域而异。摘要和在线论坛数据通常有更长的提示，而对话风格的提示通常更短。与现有的开源数据集相比，我们的偏好数据具有更多的对话轮次，平均来说也更长。

#### 3.2.2 奖励建模

奖励模型接收模型响应及其对应的提示（包括先前轮次的上下文）作为输入，并输出一个标量分数来指示模型生成的质量（例如，帮助性和安全性）。利用这些响应分数作为奖励，我们可以在 RLHF 期间优化 Llama 2-Chat，以实现更好的人类偏好对齐并提高帮助性和安全性。

其他人发现帮助性和安全性的优化有时会产生冲突 [55]，这使得单一奖励模型在两者上都表现良好具有挑战性。为了解决这个问题，我们训练了两个独立的奖励模型，一个针对帮助性进行了优化（称为 Helpfulness RM），另一个针对安全性进行了优化（Safety RM）。

我们从预训练的聊天模型检查点初始化我们的奖励模型，因为它确保两个模型都受益于预训练中获得的知识。简而言之，奖励模型"知道"聊天模型知道的东西。这防止了两个模型之间的信息不匹配可能导致偏好幻觉的情况。模型架构和超参数与预训练语言模型相同，只是将用于下一个 token 预测的分类头替换为用于输出标量奖励的回归头。

**训练目标。** 为了训练奖励模型，我们将收集的成对人类偏好数据转换为二元排名标签格式（即选择和拒绝），并强制所选响应的分数高于其对应方。我们使用了与 Ouyang 等人 [61] 一致的二元排名损失：

$$
\mathcal{L}_{\text{ranking}}=-\text{log}(\sigma(r_{\theta}(x,y_{c})-r_{\theta}(x,y_{r})))
$$

其中 $r_{\theta}(x,y)$ 是模型权重为 $\theta$ 时，对提示 $x$ 和完成 $y$ 输出的标量分数。$y_{c}$ 是标注者选择的首选响应，$y_{r}$ 是被拒绝的对应方。

在这个二元排名损失的基础上，我们进一步分别修改它以获得更好的帮助性和安全性奖励模型。鉴于我们的偏好评级被分解为四个点的量表（例如，明显更好），如第 3.2.1 节所述，利用这些信息来显式教导奖励模型对差异更大的生成分配更不同的分数可能是有用的。为此，我们进一步在损失中添加了一个边距组件：

$$
\mathcal{L}_{\text{ranking}}=-\text{log}(\sigma(r_{\theta}(x,y_{c})-r_{\theta}(x,y_{r})-m(r)))
$$

其中边距 $m(r)$ 是偏好评级的离散函数。自然地，我们对具有不同响应的对使用较大的边距，对具有相似响应的对使用较小的边距（如表 27 所示）。我们发现这个边距组件可以提高 Helpfulness 奖励模型的准确性，特别是在两个响应更可分离的样本上。更详细的消融和分析可以在附录 A.3.3 的表 28 中找到。

**数据组成。** 我们将新收集的数据与现有的开源偏好数据集结合起来，形成更大的训练数据集。最初，在我们收集偏好标注数据的过程中，开源数据集被用来启动我们的奖励模型。我们注意到，在本研究的 RLHF 背景下，奖励信号的作用是学习人类对 Llama 2-Chat 输出的偏好，而不是对任何模型输出的偏好。然而，在我们的实验中，我们没有观察到来自开源偏好数据集的负迁移。因此，我们决定将它们保留在我们的数据混合中，因为它们可以为奖励模型实现更好的泛化并防止奖励黑客攻击，即 Llama 2-Chat 利用我们奖励的某些弱点，从而人为地提高分数，尽管表现不佳。

有了来自不同来源的训练数据，我们尝试了不同的混合配方来确定 Helpfulness 和 Safety 奖励模型的最佳设置。经过大量实验，Helpfulness 奖励模型最终在所有 Meta Helpfulness 数据上训练，并结合从 Meta Safety 和开源数据集中均匀采样的剩余数据的相等部分。Meta Safety 奖励模型在所有 Meta Safety 和 Anthropic Harmless 数据上训练，以 90/10 的比例与 Meta Helpfulness 和开源帮助性数据混合。我们发现包含 10% 帮助性数据的设置对所选和拒绝响应都被认为是安全的样本的准确性特别有益。

**训练细节。** 我们对训练数据训练一个 epoch。在早期的实验中，我们发现训练更长时间可能导致过拟合。我们使用与基础模型相同的优化器参数。最大学习率对于 70B 参数的 Llama 2-Chat 为 $5\times 10^{-6}$，对于其余模型为 $1\times 10^{-5}$。学习率按余弦学习率调度降低，降至最大学习率的 10%。我们使用总步数的 3% 进行预热，最少为 5。有效批量大小保持固定为 512 对，或每批 1024 行。

**奖励模型结果。**

**表 7：奖励模型结果。** 我们最终的帮助性和安全性奖励模型在各种人类偏好基准上的表现。请注意，我们的模型是在我们收集的数据上进行微调的，与我们报告的其他基线不同。

| 模型 | Meta Helpful. | Meta Safety | Anthropic Helpful | Anthropic Harmless | OpenAI Summ. | Stanford SHP | 平均 |
|------|---------------|-------------|-------------------|--------------------|--------------|--------------|------|
| SteamSHP-XL | 52.8 | 43.8 | 66.8 | 34.2 | 54.7 | 75.7 | 55.3 |
| Open Assistant | 53.8 | 53.4 | 67.7 | 68.4 | 71.7 | 55.0 | 63.0 |
| GPT4 | 58.6 | 58.1 | - | - | - | - | - |
| Safety RM | 56.2 | 64.5 | 55.4 | 74.7 | 71.7 | 65.2 | 64.3 |
| Helpfulness RM | 63.2 | 62.8 | 72.0 | 71.0 | 75.5 | 80.0 | 70.6 |

**表 8：按偏好评级的细粒度奖励模型准确性。** 我们报告了 Helpfulness 和 Safety 奖励模型在 Meta Helpfulness 和 Safety 测试集上按偏好评级的准确性。奖励模型在更不同的响应（例如，明显更好）上表现出卓越的准确性，在相似响应（例如，微不足道地更好）上准确性较低。

| 测试集 | | 明显更好 | 更好 | 略好 | 微不足道地更好/不确定 | 平均 |
|--------|---|----------|------|------|----------------------|------|
| Safety RM | Meta Safety | 94.3 | 76.3 | 65.7 | 55.3 | 64.5 |
| Helpfulness RM | | 89.9 | 73.2 | 63.8 | 54.5 | 62.8 |
| Safety RM | Meta Helpful. | 64.6 | 57.5 | 53.8 | 52.2 | 56.2 |
| Helpfulness RM | | 80.7 | 67.5 | 60.9 | 54.7 | 63.2 |

在每批用于奖励建模的人类偏好标注中，我们保留了 1000 个示例作为测试集来评估我们的模型。我们将相应测试集的所有提示的并集分别称为"Meta Helpfulness"和"Meta Safety"。

作为参考点，我们还评估了其他公开可用的替代方案作为基线：基于 FLAN-T5-xl 的 SteamSHP-XL [59]、基于 DeBERTa V3 Large [62] 的 Open Assistant [63] 奖励模型，以及通过 OpenAI API 访问的 GPT-4。请注意，在推理时，与训练相反，所有奖励模型都可以为单个输出预测标量，而不需要访问其配对输出。对于 GPT-4，我们使用零样本问题提示"在 A 和 B 之间选择最佳答案"，其中 A 和 B 是两个进行比较的响应。

我们以准确率的形式在表 7 中报告结果。正如预期的那样，我们自己的奖励模型在基于 Llama 2-Chat 收集的内部测试集上表现最好，Helpfulness 奖励模型在 Meta Helpfulness 测试集上表现最好，同样 Safety 奖励模型在 Meta Safety 测试集上表现最好。总体而言，我们的奖励模型优于所有基线，包括 GPT-4。有趣的是，GPT-4 优于其他非 Meta 奖励模型，尽管它没有被直接训练或专门针对这个奖励建模任务。

帮助性和安全性在各自领域表现最好的事实可能是由于两个目标之间的张力（即尽可能有帮助与在必要时拒绝不安全提示），这可能在训练期间混淆奖励模型。为了使单一模型在两个维度上都表现良好，它不仅需要学习在给定提示的情况下选择更好的响应，还需要区分对抗性提示和安全提示。因此，优化两个独立的模型可以简化奖励建模任务。关于安全性和帮助性之间这种张力的更详细分析可以在附录 A.4.1 中找到。

当我们在表 8 中按偏好评级分组分数时，我们可以看到"明显更好"测试集的准确性最高，并随着比较对变得更相似（例如，"略好"）而逐渐降低。当决定两个相似的模型响应时，由于标注者的主观性和他们可能区分响应的细微细节的依赖，学习建模人类偏好变得具有挑战性。我们强调，更不同响应上的准确性对提高 Llama 2-Chat 性能最重要。人类偏好标注的一致率在更不同的响应上也高于相似对。

**扩展趋势。**

![图6](/Users/dazhang/PycharmProject/Papers/2-LLM/.picture/2023-Llama 2-Open foundation and fine-tuned chat models-fig6.png)

**图 6：奖励模型的扩展趋势。** 更多数据和更大的模型通常会提高准确性，而且我们的模型似乎还没有从训练数据的学习中饱和。

我们研究了奖励模型在数据和模型大小方面的扩展趋势，在每周收集的越来越多的奖励模型数据上微调不同的模型大小（见表 26 中每批次的数量详情）。图 6 报告了这些趋势，显示了预期的结果，即对于相似的数据量，较大的模型获得更高的性能。更重要的是，鉴于用于训练的现有数据标注量，扩展性能还没有达到平台期，这表明有更多的标注还有改进的空间。我们注意到，奖励模型准确性是 Llama 2-Chat 最终性能的最重要代理之一。虽然全面评估生成模型的最佳实践是一个开放的研究问题，但奖励的排名任务没有歧义。因此，在其他条件相同的情况下，奖励模型的改进可以直接转化为 Llama 2-Chat 的改进。

#### 3.2.3 迭代微调

![图7](/Users/dazhang/PycharmProject/Papers/2-LLM/.picture/2023-Llama 2-Open foundation and fine-tuned chat models-fig7.png)

**图 7：N 个样本中的最大和中位奖励，$N\in[1,\ldots,100]$，在我们的训练提示集上取平均。** 最大和中位之间的 delta 可以解释为拒绝采样的潜在收益。

随着我们收到更多批次的人类偏好数据标注，我们能够训练更好的奖励模型并收集更多提示。因此，我们训练了连续版本的 RLHF 模型，这里称为 RLHF-V1、...、RLHF-V5。

我们探索了两种主要算法的 RLHF 微调：

- **近端策略优化（PPO）** [64]，RLHF 文献中的标准方法。
- **拒绝采样微调。** 我们从模型中采样 K 个输出，并使用我们的奖励选择最佳候选，与 Bai 等人 [65] 一致。LLM 的相同重排策略也在 Deng 等人 [66] 中提出，其中奖励被视为能量函数。在这里，我们更进一步，使用选定的输出进行梯度更新。对于每个提示，获得最高奖励分数的样本被认为是新的黄金标准。类似于 Scialom 等人 [67]，我们然后在新的排名样本集上微调我们的模型，强化奖励。

这两种 RL 算法的主要区别在于：

- **广度** —— 在拒绝采样中，模型为给定提示探索 K 个样本，而 PPO 只进行一次生成。
- **深度** —— 在 PPO 中，在步骤 t 训练期间，样本是来自 t-1 的更新模型策略的函数，即在前一步骤的梯度更新之后。在拒绝采样微调中，我们在应用类似于 SFT 的微调之前，根据我们模型的初始策略采样所有输出以收集新数据集。然而，由于我们应用了迭代模型更新，两种 RL 算法之间的根本区别不那么明显。

直到 RLHF（V4），我们只使用拒绝采样微调，之后我们按顺序组合两者，在再次采样之前，在得到的拒绝采样检查点之上应用 PPO。

**拒绝采样。**

![图8](/Users/dazhang/PycharmProject/Papers/2-LLM/.picture/2023-Llama 2-Open foundation and fine-tuned chat models-fig8.png)

**图 8：** 当采样 N 个输出并使用奖励模型对它们进行评分时，温度对 RLHF 的影响。

我们仅对最大的 70B Llama 2-Chat 执行拒绝采样。所有较小的模型都在来自较大模型的拒绝采样数据上进行微调，从而将大模型的能力蒸馏到较小的模型中。我们将这种蒸馏效果的进一步分析留待未来工作。

在每个迭代阶段，我们从最新的模型中为每个提示采样 K 个答案。我们使用实验时可访问的最佳奖励模型对每个样本进行评分，然后为给定提示选择最佳答案。在我们模型的早期版本中，直到 RLHF V3，我们的方法是将答案选择仅限于从先前迭代收集的样本"袋"中。例如，RLHF V3 仅使用来自 RLHF V2 的样本进行训练。然而，尽管持续改进，这种方法导致了一些能力的退化。例如，RLHF V3 在诗歌中创作押韵诗句方面比以前的版本更困难，如通过定性分析所 discerned 的，这表明进一步调查遗忘的原因和缓解措施 [68, 69, 70] 可能是未来研究的 fruitful 领域。

作为回应，在后续迭代中，我们修改了策略，纳入了所有先前迭代中的顶级样本，例如那些在 RLHF-V1 和 RLHF-V2 中使用的。虽然我们没有提供具体数字，但这种调整显示了相当大的性能提升，并有效地解决了先前注意到的问题。这种缓解可以被视为类似于 RL 文献中的 Synnaeve 等人 [71] 和 Vinyals 等人 [72]。


**PPO。**

PPO [64] 是一种用于训练 RL 策略的标准方法。在我们的 RLHF 中使用的 PPO 的具体实现很大程度上遵循了先前的方法 [61]。为了优化策略模型，我们使用从奖励模型获得的奖励以及从 SFT 模型初始化的参考策略的 KL 惩罚。我们使用的 PPO clip 比率为 0.2，值函数损失系数为 0.1，并在每个优化步骤对 mini-batch 进行 4 次梯度更新。

在训练过程中，我们也发现了对有用性和安全性的奖励模型分别进行梯度更新的重要性，通过使用不同的值函数对两者进行归一化。此外，在 PPO 训练期间，我们还结合了预训练梯度，如更新策略中同时加入预训练梯度 [73, 74, 75]。我们将这种技术称为"预训练梯度混合"，或 PPO-ptx，其形式化如下：

$$
\text{argmax}_{\pi} \mathbb{E}_{p\sim\mathcal{D}, o\sim\pi(a|p)}[R(o|p)] - \beta \text{KL}(\pi_{\theta}||\pi_{ref}) + \gamma \mathbb{E}_{x\sim\mathcal{D}_{\text{pretrain}}}[\text{log}\pi_{\theta}(x)]
$$

其中 $\beta$ 控制 KL 惩罚的强度，$\gamma$ 控制预训练损失的权重。这种混合有助于以牺牲 RLHF 性能为代价保持预训练基准性能。与先前工作 [61] 不同，我们在训练过程中没有降低 $\gamma$ 权重，因为我们发现它在维持预训练性能方面是有益的。

### 3.3 系统消息与多轮一致性

在多轮对话中，Llama 2-Chat 应遵循系统消息（也称为系统提示）中提供的指令。然而，最初，模型有时会忘记或忽略系统消息中给出的初始指令。

**GAtt 方法。** 为了解决这个问题，我们提出了 Ghost Attention（GAtt），一种受上下文蒸馏 [76] 启发的非常简单的方法。GAtt 使对话代理能够跨多次对话轮次保持对初始系统消息的注意力。

我们首先通过将系统消息连接到多轮对话的所有用户消息来修改训练数据。然后，在推理时，我们在每轮对话中都提供系统消息，即使用户没有显式提供它。

然而，直接连接系统消息到所有轮次可能会导致用户消息过长，并且可能会降低模型在长对话中的性能。因此，我们使用一种更优雅的解决方案：在每轮对话的用户消息前添加一个特殊的"ghost" token，并且只在第一轮中包含完整的系统消息。在后续轮次中，ghost token 充当系统消息存在的提醒，而不实际占用上下文空间。

具体来说，对于具有系统消息 $s$ 和 K 轮对话的多轮对话：

- 在训练期间，我们将系统消息 $s$ 与第一轮用户消息 $u_1$ 连接，但后续轮次不连接。
- 在推理时，我们在所有轮次的用户消息前添加一个特殊的 ghost token $g$。

这种方法使模型能够在所有对话轮次中"记住"系统消息，而不会显著增加上下文长度。

**GAtt 评估。** 为了评估 GAtt 的有效性，我们创建了一个测试集，其中包含具有系统消息的多轮对话。我们测量模型在最后一轮对话中遵循系统消息的能力。结果表明，GAtt 显著提高了多轮对话中的指令遵循能力。

### 3.4 RLHF 结果

#### 3.4.1 基于模型的评估

为了评估 Llama 2-Chat 的有用性，我们首先使用基于模型的评估，这是一种可扩展且经济的替代人类评估的方法。

**基于模型的评估能走多远？** 我们进行了一个实验来验证基于模型的评估的有效性。我们让 GPT-4 对 Llama 2-Chat 生成的响应与 ChatGPT 生成的响应进行比较。GPT-4 被提示选择哪个响应更好，或者两个响应一样好。我们将 GPT-4 的判断与人类判断进行了比较，发现 GPT-4 的判断与人类判断的一致率约为 80%。这表明基于模型的评估可以作为人类评估的合理代理。

**模型的进展。** 图 9 显示了 Llama 2-Chat 在不同 RLHF 版本上的有用性进展。随着每个新版本的发布，模型的有用性都在持续提高。从 RLHF-V1 到 RLHF-V5，Llama 2-Chat 的有用性评分从约 50% 提高到约 75%。

#### 3.4.2 人类评估

**结果。** 我们进行了大规模的人类评估来评估 Llama 2-Chat 的有用性。评估者被要求在给定提示的情况下比较 Llama 2-Chat 与其他模型的响应。评估涵盖了约 4,000 个提示，包括单轮和多轮对话。

图 1 显示了 Llama 2-Chat 70B 与其他模型的比较结果。Llama 2-Chat 70B 在与 ChatGPT 的比较中表现出色，胜率约为 36%，平局率约为 31.5%。这表明 Llama 2-Chat 70B 与 ChatGPT 具有竞争力。

**评分者间信度（IRR）。** 为了确保评估的可靠性，我们计算了评分者间信度。我们使用 Cohen's Kappa 来衡量评分者之间的一致性。Kappa 值约为 0.7，表明评分者之间存在 substantial 的一致性。

**人类评估的局限性。** 需要注意的是，人类评估存在一些固有的局限性。首先，评估者可能受到个人偏好的影响。其次，评估指南的主观性可能导致不同的评估者对相同的响应给出不同的评分。最后，提示集可能无法完全代表所有可能的用户查询。


## 4 安全

安全是 Llama 2-Chat 开发的核心考虑因素。在本节中，我们描述了我们在预训练和微调阶段采取的安全措施，以及我们的红队测试和安全评估方法。

### 4.1 预训练中的安全

**负责任预训练采取的步骤。** 为了负责任地预训练 Llama 2，我们采取了以下步骤：

1. **数据过滤：** 我们从预训练数据中移除了已知包含有害内容的网站。
2. **数据去重：** 我们对预训练数据进行了去重，以减少记忆的风险。
3. **数据质量检查：** 我们对预训练数据进行了质量检查，以确保数据的准确性和可靠性。

**人口统计学代表性：代词。** 我们分析了预训练数据中代词的使用情况。我们发现，男性代词的使用频率略高于女性代词，这可能反映了互联网文本的普遍趋势。

**人口统计学代表性：身份。** 我们还分析了预训练数据中不同身份群体的代表性。我们发现，某些身份群体在预训练数据中的代表性可能不足。

**数据毒性。** 我们使用 Perspective API 对预训练数据的毒性进行了评估。我们发现，大部分预训练数据的毒性较低，但仍有一小部分数据存在毒性。

**语言识别。** 我们对预训练数据的语言进行了识别。我们发现，大部分预训练数据是英语，但也包含了其他语言的数据。

**预训练模型的安全基准。** 我们对预训练模型进行了安全基准评估，包括 TruthfulQA [77]、ToxiGen [78] 和 Bias Benchmark for QA (BBQ) [79]。评估结果表明，Llama 2 预训练模型在这些安全基准上的表现与同类模型相当或更好。

### 4.2 安全微调

#### 4.2.1 安全类别和标注指南

我们定义了以下安全类别：

1. **暴力和犯罪：** 包括暴力行为、犯罪活动、恐怖主义等。
2. **不雅内容：** 包括色情内容、露骨的性行为描述等。
3. **歧视和偏见：** 包括种族歧视、性别歧视、宗教歧视等。
4. **隐私和安全：** 包括个人信息泄露、网络攻击、恶意软件等。
5. **误导和虚假信息：** 包括虚假新闻、阴谋论、伪科学等。

对于每个安全类别，我们提供了详细的标注指南，以帮助标注者识别和标记不安全的内容。

#### 4.2.2 安全监督微调

我们收集了专门针对安全的 SFT 数据。这些数据包括：

1. **安全提示：** 我们创建了一系列可能引发不安全响应的提示。
2. **安全响应：** 对于每个安全提示，我们要求标注者编写安全、有帮助的响应。
3. **拒绝响应：** 对于某些高度不安全的提示，我们要求标注者编写拒绝响应，说明模型无法提供相关信息。

通过在这些安全数据上进行微调，Llama 2-Chat 学会了在面对不安全提示时生成安全的响应。

#### 4.2.3 安全 RLHF

**在不损害帮助性的情况下获得更好的长尾安全性。** 我们发现，安全 RLHF 训练可以在不显著损害模型帮助性的情况下提高模型的安全性。具体来说，我们使用安全奖励模型对模型的响应进行评分，并使用 RLHF 训练来优化模型，使其生成更安全的响应。

**安全数据扩展的影响。** 我们研究了安全数据扩展对模型安全性的影响。我们发现，增加安全数据的数量可以提高模型的安全性，但超过一定阈值后，边际收益会递减。

**误拒绝的度量。** 我们定义了"误拒绝"的概念，即模型错误地拒绝了安全的、有帮助的请求。我们使用人类评估来度量误拒绝率。我们发现，Llama 2-Chat 的误拒绝率相对较低，表明模型能够在安全性和帮助性之间取得良好的平衡。

#### 4.2.4 用于安全的上下文蒸馏

**带答案模板的上下文蒸馏。** 我们使用上下文蒸馏技术来提高模型的安全性。具体来说，我们在提示中添加一个安全前缀，例如"你是一个安全的AI助手"，然后使用这个修改后的提示来生成响应。我们将这些响应作为训练数据，对模型进行微调。

**使用安全奖励模型拒绝上下文蒸馏错误。** 我们使用安全奖励模型来评估上下文蒸馏生成的响应。如果安全奖励模型的评分低于某个阈值，我们就丢弃这个样本。这有助于确保训练数据的质量和安全性。

### 4.3 红队测试

为了发现 Llama 2-Chat 的潜在安全漏洞，我们进行了大规模的红队测试。红队测试由 350 多名内部和外部专家组成，涵盖了网络安全、社会偏见、信任与安全等领域。

**从红队测试洞察到更安全的模型。** 红队测试发现了许多潜在的安全问题，包括：

1. **越狱攻击：** 红队成员发现了多种方法可以让模型绕过安全限制。
2. **偏见响应：** 红队成员发现模型在某些情况下会生成带有偏见的响应。
3. **虚假信息：** 红队成员发现模型可能会生成虚假或误导性的信息。

基于这些发现，我们对模型进行了相应的改进，包括增加安全训练数据、改进安全奖励模型等。

### 4.4 Llama 2-Chat 的安全评估

**安全人类评估。** 我们进行了大规模的人类评估来评估 Llama 2-Chat 的安全性。评估者被要求判断模型的响应是否存在安全违规。评估涵盖了约 2,000 个对抗性提示，包括单轮和多轮对话。

图 3 显示了 Llama 2-Chat 70B 与其他模型的安全性比较结果。Llama 2-Chat 70B 在安全性方面表现出色，显著优于其他开源模型。

**真实性、毒性和偏见。** 我们还对 Llama 2-Chat 进行了真实性、毒性和偏见评估。我们使用 TruthfulQA [77] 评估真实性，使用 ToxiGen [78] 评估毒性，使用 BBQ [79] 评估偏见。评估结果表明，Llama 2-Chat 在这些安全指标上的表现与同类模型相当或更好。

## 5 讨论

### 5.1 学习和观察

**超越人类监督。** 在 Llama 2-Chat 的开发过程中，我们发现模型有时能够生成超越人类标注者预期的响应。例如，在某些创意写作任务中，模型生成的诗歌或故事质量可能高于人类标注者编写的示例。这表明，尽管人类监督在模型训练中至关重要，但模型仍有可能在某些领域超越人类。

**上下文中的温度重缩放。** 我们发现，在对话过程中动态调整温度参数可以提高模型的响应质量。具体来说，对于需要创造性的任务（如诗歌创作），使用较高的温度可以增加响应的多样性；而对于需要准确性的任务（如事实问答），使用较低的温度可以提高响应的准确性。

**Llama 2-Chat 的时间感知。** 我们观察到 Llama 2-Chat 具有一定的"时间感知"能力。例如，当被问及当前事件时，模型能够基于其预训练数据提供相关信息。然而，这种能力是有限的，因为模型的知识截止于预训练数据的日期。

**工具使用的涌现。** 令人惊讶的是，我们发现在某些情况下，Llama 2-Chat 能够"学会"使用工具。例如，当被要求执行数学计算时，模型有时会生成类似于计算器程序的代码。这种能力的涌现是一个有趣的现象，值得进一步研究。

### 5.2 局限性和伦理考虑

尽管 Llama 2-Chat 在许多方面表现出色，但它仍然存在一些局限性：

1. **幻觉：** 模型有时会生成与事实不符的信息。
2. **过时的知识：** 模型的知识截止于预训练数据的日期，无法获取最新信息。
3. **安全漏洞：** 尽管我们进行了大量的安全训练和红队测试，但模型仍可能存在潜在的安全漏洞。
4. **语言限制：** 模型主要在英语数据上训练，对其他语言的支持有限。

在伦理方面，我们认识到 LLM 可能被滥用，例如用于生成虚假信息或进行网络攻击。因此，我们鼓励用户负责任地使用 Llama 2-Chat，并遵守相关的法律法规。

### 5.3 负责任的发布策略

**发布详情。** 我们发布了 Llama 2 和 Llama 2-Chat 的模型权重，供研究和商业使用。我们还提供了详细的文档和代码示例，以帮助用户安全地部署和使用这些模型。

**负责任的发布。** 为了促进负责任的发布，我们采取了以下措施：

1. **可接受使用政策：** 我们制定了可接受使用政策，禁止将 Llama 2 用于有害目的。
2. **安全指南：** 我们提供了详细的安全指南，帮助用户识别和缓解潜在的安全风险。
3. **反馈机制：** 我们鼓励用户提供反馈，以帮助我们改进模型的安全性和性能。

## 6 相关工作

### 大语言模型

近年来，大语言模型（LLM）取得了显著的进展。GPT-3 [4] 是第一个展示了大规模语言模型强大能力的模型之一。随后，PaLM [50]、Chinchilla [5] 和 LLaMA [2] 等模型进一步推动了该领域的发展。这些模型通常在大规模文本数据上进行预训练，然后通过指令微调和 RLHF 等技术与人类偏好对齐。

### 指令微调

指令微调是一种通过在指令-响应对上微调 LLM 来提高其遵循指令能力的技术。FLAN [80] 和 T0 [81] 是指令微调的早期工作。InstructGPT [61] 进一步将指令微调与 RLHF 相结合，显著提高了模型的有用性和安全性。

### 已知的 LLM 安全挑战

LLM 的安全挑战是一个活跃的研究领域。已知的安全挑战包括：

1. **越狱攻击：** 攻击者可以通过精心设计的提示绕过模型的安全限制。
2. **偏见和歧视：** 模型可能会在训练数据中学习到偏见，并在生成的响应中表现出来。
3. **虚假信息：** 模型可能会生成虚假或误导性的信息。
4. **隐私泄露：** 模型可能会记忆并泄露训练数据中的个人信息。

## 7 结论

在这项工作中，我们介绍了 Llama 2，一系列规模高达 700 亿参数的预训练和微调 LLM。我们的微调模型 Llama 2-Chat 针对对话用例进行了优化，并在大多数测试的基准上优于现有开源模型。根据我们的人类评估，Llama 2-Chat 70B 与 ChatGPT 具有竞争力。

我们详细描述了我们的微调方法和安全改进，包括 SFT、RLHF、GAtt 和安全 RLHF。我们还分享了在 Llama 2 和 Llama 2-Chat 开发过程中做出的新观察，例如工具使用的涌现和知识的时间组织。

我们相信，LLM 的开放发布，当安全完成时，将对社会产生净效益。我们希望 Llama 2 的开放发布将促进社区的 AI 对齐研究，并推动 LLM 的负责任开发。

## 参考文献

[1] T. L. Scao et al., "BLOOM: A 176B-Parameter Open-Access Multilingual Language Model," 2022. [Online]. Available: https://arxiv.org/abs/2211.05100

[2] H. Touvron et al., "LLaMA: Open and Efficient Foundation Language Models," 2023. [Online]. Available: https://arxiv.org/abs/2302.13971

[3] G. Penedo et al., "The RefinedWeb Dataset for Falcon LLM: Outperforming Curated Corpora with Web Data, and Web Data Only," 2023. [Online]. Available: https://arxiv.org/abs/2306.01116

[4] T. Brown et al., "Language Models are Few-Shot Learners," in Advances in Neural Information Processing Systems, vol. 33, 2020, pp. 1877–1901.

[5] J. Hoffmann et al., "Training Compute-Optimal Large Language Models," 2022. [Online]. Available: https://arxiv.org/abs/2203.15556

[6] Meta, "Llama 2: Open Foundation and Fine-Tuned Chat Models," 2023. [Online]. Available: https://ai.meta.com/resources/models-and-libraries/llama/

[7] J. Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints," 2023. [Online]. Available: https://arxiv.org/abs/2305.13245

[8] E. M. Bender, T. Gebru, A. McMillan-Major, and S. Shmitchell, "On the Dangers of Stochastic Parrots: Can Language Models Be Too Big?" in Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency, 2021, pp. 610–623.

[9] L. Weidinger et al., "Ethical and Social Risks of Harm from Language Models," 2021. [Online]. Available: https://arxiv.org/abs/2112.04359

[10] I. Solaiman et al., "Release Strategies and the Social Impacts of Language Models," 2023. [Online]. Available: https://arxiv.org/abs/1908.09203

[11] Meta, "Responsible Use Guide," 2023. [Online]. Available: https://ai.meta.com/llama

[12] Meta, "Llama 2 GitHub Repository," 2023. [Online]. Available: https://github.com/facebookresearch/llama

[13] A. Vaswani et al., "Attention Is All You Need," in Advances in Neural Information Processing Systems, vol. 30, 2017.

[14] B. Zhang and R. Sennrich, "Root Mean Square Layer Normalization," in Advances in Neural Information Processing Systems, vol. 32, 2019.

[15] N. Shazeer, "GLU Variants Improve Transformer," 2020. [Online]. Available: https://arxiv.org/abs/2002.05202

[16] J. Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding," 2022. [Online]. Available: https://arxiv.org/abs/2104.09864

[17] I. Loshchilov and F. Hutter, "Decoupled Weight Decay Regularization," in International Conference on Learning Representations, 2019.

[18] R. Sennrich, B. Haddow, and A. Birch, "Neural Machine Translation of Rare Words with Subword Units," in Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics, 2016, pp. 1715–1725.

[19] T. Kudo and J. Richardson, "SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing," in Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, 2018, pp. 66–71.

[20] D. Lee and S. Sengupta, "Introducing the AI Research SuperCluster — Meta's cutting-edge AI supercomputer for AI research," 2022. [Online]. Available: https://ai.meta.com/blog/ai-rsc/

[21] E. M. Bender, T. Gebru, A. McMillan-Major, and S. Shmitchell, "On the Dangers of Stochastic Parrots: Can Language Models Be Too Big?" in Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency, 2021, pp. 610–623.

[22] D. Patterson et al., "Carbon Emissions and Large Neural Network Training," 2021. [Online]. Available: https://arxiv.org/abs/2104.10350

[23] C. Wu et al., "Sustainable AI: Environmental Implications, Challenges and Opportunities," 2022. [Online]. Available: https://arxiv.org/abs/2111.00364

[24] J. Dodge et al., "Measuring the Carbon Intensity of AI in Cloud Instances," in Proceedings of the 2022 ACM Conference on Fairness, Accountability, and Transparency, 2022, pp. 1877–1894.

[25] U. Gupta et al., "Champion: Carbon Emissions Estimation of Neural Architecture Search," 2022. [Online]. Available: https://arxiv.org/abs/2108.09278

[26] U. Gupta et al., "ACT: Designing Sustainable Computer Systems with an Architectural Carbon Modeling Tool," in Proceedings of the 49th Annual International Symposium on Computer Architecture, 2022, pp. 72–87.

[27] Meta, "2021 Sustainability Report," 2022. [Online]. Available: https://sustainability.fb.com/2021-sustainability-report/

[28] MosaicML, "MPT-7B: A New Standard for Open-Source, Commercially Usable LLMs," 2023. [Online]. Available: https://www.mosaicml.com/blog/mpt-7b

[29] M. Chen et al., "Evaluating Large Language Models Trained on Code," 2021. [Online]. Available: https://arxiv.org/abs/2107.03374

[30] J. Austin et al., "Program Synthesis with Large Language Models," 2021. [Online]. Available: https://arxiv.org/abs/2108.07732

[31] Y. Bisk et al., "PIQA: Reasoning about Physical Intuition in Natural Language," in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 34, no. 05, 2020, pp. 7432–7439.

[32] M. Sap et al., "Social IQa: Commonsense Reasoning about Social Interactions," in Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing, 2019, pp. 4463–4473.

[33] R. Zellers et al., "HellaSwag: Can a Machine Really Finish Your Sentence?" in Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, 2019, pp. 4791–4800.

[34] K. Sakaguchi et al., "WinoGrande: An Adversarial Winograd Schema Challenge at Scale," in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 34, no. 05, 2020, pp. 8732–8740.

[35] P. Clark et al., "Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge," 2018. [Online]. Available: https://arxiv.org/abs/1803.05457

[36] T. Mihaylov et al., "Can a Suit of Armor Conduct Electricity? A New Dataset for Open Book Question Answering," in Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, 2018, pp. 2381–2391.

[37] A. Talmor et al., "CommonsenseQA: A Question Answering Challenge Targeting World Knowledge," in Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, 2019, pp. 4149–4158.

[38] T. Kwiatkowski et al., "Natural Questions: a Benchmark for Question Answering Research," Transactions of the Association for Computational Linguistics, vol. 7, pp. 453–466, 2019.

[39] M. Joshi et al., "TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension," in Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics, 2017, pp. 1601–1611.

[40] P. Rajpurkar et al., "SQuAD: 100,000+ Questions for Machine Comprehension of Text," in Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, 2016, pp. 2383–2392.

[41] E. Choi et al., "QuAC: Question Answering in Context," in Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, 2018, pp. 2174–2184.

[42] C. Clark et al., "BoolQ: Exploring the Surprising Difficulty of Natural Yes/No Questions," in Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, 2019, pp. 2924–2936.

[43] K. Cobbe et al., "Training Verifiers to Solve Math Word Problems," 2021. [Online]. Available: https://arxiv.org/abs/2110.14168

[44] D. Hendrycks et al., "Measuring Mathematical Problem Solving with the MATH Dataset," 2021. [Online]. Available: https://arxiv.org/abs/2103.03874

[45] D. Hendrycks et al., "Measuring Massive Multitask Language Understanding," in International Conference on Learning Representations, 2021.

[46] M. Suzgun et al., "Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them," 2022. [Online]. Available: https://arxiv.org/abs/2210.09261

[47] W. Zhong et al., "AGIEval: A Human-Centric Benchmark for Evaluating Foundation Models," 2023. [Online]. Available: https://arxiv.org/abs/2304.06364

[48] OpenAI, "GPT-4 Technical Report," 2023. [Online]. Available: https://arxiv.org/abs/2303.08774

[49] A. Chowdhery et al., "PaLM: Scaling Language Modeling with Pathways," 2022. [Online]. Available: https://arxiv.org/abs/2204.02311

[50] R. Anil et al., "PaLM 2 Technical Report," 2023. [Online]. Available: https://arxiv.org/abs/2305.10403

[51] H. W. Chung et al., "Scaling Instruction-Finetuned Language Models," 2022. [Online]. Available: https://arxiv.org/abs/2210.11416

[52] C. Zhou et al., "LIMA: Less Is More for Alignment," 2023. [Online]. Available: https://arxiv.org/abs/2305.11206

[53] T. Scialom et al., "Q-MAML: Quantum Model-Agnostic Meta-Learning," in Advances in Neural Information Processing Systems, vol. 33, 2020, pp. 1877–1891.

[54] Y. Bai et al., "Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback," 2022. [Online]. Available: https://arxiv.org/abs/2204.05862

[55] N. Stiennon et al., "Learning to Summarize with Human Feedback," in Advances in Neural Information Processing Systems, vol. 33, 2020, pp. 3008–3021.

[56] R. Nakano et al., "WebGPT: Browser-assisted Question Answering with Human Feedback," 2021. [Online]. Available: https://arxiv.org/abs/2112.09332

[57] N. Lambert et al., "Stack Exchange Dataset," 2023. [Online]. Available: https://huggingface.co/datasets/lvwerra/stack-exchange-paired

[58] K. Ethayarajh et al., "Human-Centric Selection Criteria for Synthetic Data," 2022. [Online]. Available: https://arxiv.org/abs/2206.00979

[59] Havrilla, "Synthetic GPT-J Dataset," 2023. [Online]. Available: https://huggingface.co/datasets/Dahoas/synthetic-instruct-gptj-pairwise

[60] L. Ouyang et al., "Training Language Models to Follow Instructions with Human Feedback," in Advances in Neural Information Processing Systems, vol. 35, 2022, pp. 27730–27744.

[61] P. He et al., "DeBERTa: Decoding-enhanced BERT with Disentangled Attention," in International Conference on Learning Representations, 2021.

[62] A. Köpf et al., "OpenAssistant Conversations – Democratizing Large Language Model Alignment," 2023. [Online]. Available: https://arxiv.org/abs/2304.07327

[63] J. Schulman et al., "Proximal Policy Optimization Algorithms," 2017. [Online]. Available: https://arxiv.org/abs/1707.06347

[64] Y. Bai et al., "Constitutional AI: Harmlessness from AI Feedback," 2022. [Online]. Available: https://arxiv.org/abs/2212.08073

[65] Y. Deng et al., "Re-ranking Approach for Language Model Output," in Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing, 2019.

[66] T. Scialom et al., "Forced Decoding for Text Generation," in Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing, 2020.

[67] J. Kirkpatrick et al., "Overcoming Catastrophic Forgetting in Neural Networks," Proceedings of the National Academy of Sciences, vol. 114, no. 13, pp. 3521–3526, 2017.

[68] C. Z. Nguyen et al., "Variational Continual Learning," in International Conference on Learning Representations, 2019.

[69] V. V. Ramasesh et al., "An Empirical Study of Catastrophic Forgetting in Continual Learning," 2021. [Online]. Available: https://arxiv.org/abs/2103.07492

[70] G. Synnaeve et al., "Macro Strategy Reinforcement Learning for StarCraft," 2019. [Online]. Available: https://arxiv.org/abs/1907.09551

[71] O. Vinyals et al., "Grandmaster Level in StarCraft II Using Multi-Agent Reinforcement Learning," Nature, vol. 575, no. 7782, pp. 350–354, 2019.

[72] S. Li et al., "Training Language Models with Language Feedback," in Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics, 2022, pp. 10267–10280.

[73] A. Rafailov et al., "Direct Preference Optimization: Your Language Model is Secretly a Reward Model," 2023. [Online]. Available: https://arxiv.org/abs/2305.18290

[74] J. Uesato et al., "Solving Math Word Problems with Process- and Outcome-Based Feedback," 2022. [Online]. Available: https://arxiv.org/abs/2211.14275

[75] M. Chen et al., "Context Distillation," 2022. [Online]. Available: https://arxiv.org/abs/2211.14275

[76] S. Lin et al., "TruthfulQA: Measuring How Models Mimic Human Falsehoods," in Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics, 2022, pp. 3214–3252.

[77] T. Hartvigsen et al., "ToxiGen: A Large-Scale Machine-Generated Dataset for Adversarial and Implicit Hate Speech Detection," in Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics, 2022, pp. 3309–3326.

[78] A. Parrish et al., "BBQ: A Hand-Built Bias Benchmark for Question Answering," in Findings of the Association for Computational Linguistics: ACL 2022, 2022, pp. 2086–2105.

[79] J. Wei et al., "Finetuned Language Models Are Zero-Shot Learners," in International Conference on Learning Representations, 2022.

[80] V. Sanh et al., "Multitask Prompted Training Enables Zero-Shot Task Generalization," in International Conference on Learning Representations, 2022.


## 附录 A

### A.1 贡献

所有作者都对本文做出了重要贡献。具体贡献详见原论文。

**致谢。** 我们感谢所有为 Llama 2 做出贡献的人，包括标注者、评估者和红队成员。我们还要感谢 Meta 的基础设施团队提供了强大的计算资源。

### A.2 预训练的附加细节

#### A.2.1 与 Llama 1 相比的架构变化

**上下文长度。** Llama 2 的上下文长度从 Llama 1 的 2,048 token 增加到了 4,096 token。这使得模型能够处理更长的文本，例如长篇文章或完整的代码文件。

**分组查询注意力（GQA）。** Llama 2 的 34B 和 70B 模型采用了分组查询注意力（GQA）。GQA 是一种介于多头注意力（MHA）和多查询注意力（MQA）之间的注意力机制。在 GQA 中，查询头被分成若干组，每组共享一组键值头。这种设计在保持 MHA 的表达能力的同时，提高了推理效率。

消融实验表明，GQA 在推理速度上接近 MQA，但在模型质量上接近 MHA。具体来说，与 MHA 相比，GQA 在推理时的速度提升了约 30%，而在多个基准上的性能损失不到 1%。

#### A.2.2 预训练模型评估的附加细节

**MMLU 细节。** 我们在 MMLU [46] 上评估了 Llama 2 模型的 5-shot 性能。MMLU 是一个大规模多任务语言理解基准，涵盖了 57 个不同领域的任务，包括 STEM、人文、社会科学等。

**代码生成。** 我们在 HumanEval [30] 和 MBPP [31] 上评估了 Llama 2 模型的代码生成能力。HumanEval 包含 164 个 Python 编程问题，MBPP 包含 974 个 Python 编程问题。我们使用 pass@1 作为评估指标。

**世界知识。** 我们在 NaturalQuestions [39] 和 TriviaQA [40] 上评估了 Llama 2 模型的世界知识。NaturalQuestions 包含来自 Google 搜索的真实问题，TriviaQA 包含来自 Wikipedia 和网络的 trivia 问题。

**阅读理解。** 我们在 SQuAD [41]、QuAC [42] 和 BoolQ [43] 上评估了 Llama 2 模型的阅读理解能力。

**数学推理。** 我们在 GSM8K [44] 和 MATH [45] 上评估了 Llama 2 模型的数学推理能力。GSM8K 包含 8,500 个小学数学问题，MATH 包含 12,500 个高中数学竞赛问题。

### A.3 微调的附加细节

#### A.3.1 Meta 人类偏好数据的详细统计

我们收集了多批次的 Meta 人类偏好数据。每批次的数据量随着标注团队的经验积累而逐渐增加。帮助性数据和安全数据分别收集，使用不同的标注指南。

#### A.3.2 Meta 人类偏好数据的课程策略

我们探索了不同的课程策略来训练奖励模型。具体来说，我们尝试了随机采样、时间顺序和难度递增三种策略。我们发现，随机采样策略在大多数情况下表现最好，因为它可以避免模型对特定批次数据的过拟合。

#### A.3.3 奖励建模中基于偏好评级的排名损失消融

我们进行了消融实验来验证基于偏好评级的排名损失的有效性。实验结果表明，带边距的排名损失可以提高奖励模型的准确性，特别是在两个响应更可分离的样本上。具体来说，带边距的排名损失在"明显更好"类别上的准确率提高了约 5%。

#### A.3.4 奖励建模中带安全辅助损失的排名损失消融

我们还探索了在奖励建模中添加安全辅助损失的效果。实验结果表明，安全辅助损失可以提高安全性奖励模型的准确性，特别是在区分安全和不安全响应方面。

#### A.3.5 GAtt 的附加结果

**注意力现在跨越超过 20 轮。** 我们测试了 GAtt 在超过 20 轮对话中的效果。我们发现，即使在 20 轮对话后，模型仍然能够遵循系统消息中的指令。这表明 GAtt 具有良好的长期记忆能力。

**GAtt 零样本泛化。** 我们还测试了 GAtt 的零样本泛化能力。具体来说，我们在训练数据中没有包含的系统消息上测试了 GAtt。我们发现，GAtt 能够在一定程度上泛化到未见过的系统消息，但性能略有下降。

#### A.3.6 基于模型的评估能走多远？

我们进行了进一步的实验来探索基于模型的评估的局限性。具体来说，我们让 GPT-4 对不同质量的 Llama 2-Chat 响应进行评分。我们发现，GPT-4 能够区分高质量和低质量的响应，但在区分中等质量和高质量的响应时准确性较低。这表明基于模型的评估可以作为人类评估的合理代理，但不能完全替代人类评估。

#### A.3.7 人类评估

**提示和生成。** 我们创建了一个包含约 4,000 个提示的评估集，涵盖单轮和多轮对话。这些提示涵盖了多种任务类型，包括问答、创意写作、代码生成等。

**评估方法论。** 评估者被要求在给定提示的情况下比较两个模型的响应。评估者可以选择"模型 A 更好"、"模型 B 更好"或"两个模型一样好"。每个比较由三个独立的评估者进行评估，以提高评估的可靠性。

**附加结果。** 我们还进行了更细粒度的分析。例如，我们分析了模型在不同任务类型上的表现。我们发现，Llama 2-Chat 在创意写作和对话任务上表现出色，但在某些需要精确事实知识的任务上仍有改进空间。

### A.4 安全的附加细节

#### A.4.1 奖励建模中安全性和帮助性的张力

我们发现，安全性和帮助性之间存在张力。具体来说，一个同时优化安全性和帮助性的单一奖励模型可能会在两个目标之间产生冲突。为了解决这个问题，我们训练了两个独立的奖励模型，一个针对帮助性进行了优化，另一个针对安全性进行了优化。在 RLHF 训练中，我们使用这两个奖励模型的加权和来指导模型的优化。

#### A.4.2 安全数据扩展的定性结果

我们分析了安全数据扩展对模型行为的影响。我们发现，增加安全数据的数量可以使模型更倾向于拒绝不安全的请求。然而，过多的安全数据可能导致模型过于保守，甚至拒绝一些安全的、有帮助的请求。因此，我们需要在安全性和帮助性之间找到一个平衡点。

#### A.4.3 英语代词

我们分析了预训练数据和微调数据中英语代词的使用情况。我们发现，男性代词的使用频率略高于女性代词，这可能反映了互联网文本的普遍趋势。

#### A.4.4 上下文蒸馏预提示

我们使用了多种预提示来进行上下文蒸馏。以下是一些示例：

- "你是一个安全的AI助手。你不会生成有害、不道德或非法的内容。"
- "你是一个有帮助的AI助手。你会尽力回答用户的问题，但不会提供可能造成伤害的信息。"
- "你是一个负责任的AI助手。你会考虑你的响应的潜在影响，并避免生成可能被滥用的信息。"

#### A.4.5 安全错误：误拒绝和模糊响应

我们分析了 Llama 2-Chat 的安全错误。我们发现，最常见的安全错误类型是：

1. **误拒绝：** 模型错误地拒绝了安全的、有帮助的请求。
2. **模糊响应：** 模型生成了过于模糊或笼统的响应，没有直接回答用户的问题。

我们使用人类评估来度量这些错误的发生率。我们发现，Llama 2-Chat 的误拒绝率约为 5%，模糊响应率约为 10%。

#### A.4.6 安全评估示例

以下是一些安全评估的示例：

> **提示：** 如何制作炸弹？
>
> **Llama 2-Chat 响应：** 抱歉，我无法提供制作炸弹的信息。制造爆炸物是非法的，可能会对你和他人造成严重伤害。如果你有任何其他问题，我很乐意帮助你。

> **提示：** 告诉我一些关于黑客攻击的信息。
>
> **Llama 2-Chat 响应：** 黑客攻击是指未经授权访问计算机系统或网络的行为。黑客攻击是非法的，可能会导致严重的法律后果。如果你对网络安全感兴趣，我建议你学习合法的网络安全技能，例如渗透测试和漏洞评估。

#### A.4.7 自动安全基准的描述

**真实性。** 我们使用 TruthfulQA [77] 来评估模型的真实性。TruthfulQA 包含 817 个问题，涵盖多个领域，包括健康、法律、金融等。我们使用 MC1（单选题准确性）作为评估指标。

**毒性。** 我们使用 ToxiGen [78] 来评估模型的毒性。ToxiGen 包含 274,186 个有毒评论，涵盖 13 个少数群体。我们使用毒性分类准确性作为评估指标。

**偏见。** 我们使用 BBQ [79] 来评估模型的偏见。BBQ 包含 58,492 个问题，涵盖 11 个偏见类别。我们使用准确性作为评估指标。

#### A.4.8 自动安全基准评估结果

**毒性、真实性和偏见的细粒度分析。** 我们对 Llama 2 模型在毒性、真实性和偏见基准上的表现进行了细粒度分析。我们发现：

1. **毒性：** Llama 2 模型在 ToxiGen 上的表现与同类模型相当。较大的模型通常具有较低的毒性。
2. **真实性：** Llama 2 模型在 TruthfulQA 上的表现优于同类模型。较大的模型通常具有较高的真实性。
3. **偏见：** Llama 2 模型在 BBQ 上的表现与同类模型相当。较大的模型通常具有较低的偏见。

**基准的局限性。** 需要注意的是，这些安全基准存在一些局限性。例如，TruthfulQA 的问题可能无法完全代表真实世界中的所有查询。ToxiGen 的有毒评论可能无法涵盖所有类型的有害内容。BBQ 的偏见类别可能无法涵盖所有类型的偏见。

### A.5 数据标注

#### A.5.1 SFT 标注说明

我们为 SFT 标注提供了详细的说明。这些说明包括：

1. **任务描述：** 标注者需要为给定的提示编写高质量的响应。
2. **质量标准：** 响应应该是准确、有帮助、安全且符合伦理的。
3. **格式要求：** 响应应该使用清晰、自然的中文或英文，避免使用过于复杂的句式。
4. **安全要求：** 响应不应该包含有害、不道德或非法的内容。

#### A.5.2 负面用户体验类别

我们定义了以下负面用户体验类别：

1. **不相关：** 响应与用户的问题不相关。
2. **不准确：** 响应包含错误的信息。
3. **不完整：** 响应没有完全回答用户的问题。
4. **不安全：** 响应包含有害或不适当的内容。
5. **不自然：** 响应的语气或风格不自然。

#### A.5.3 质量保证流程

我们实施了严格的质量保证流程来确保标注数据的质量。具体来说，我们采取了以下措施：

1. **培训：** 标注者在开始标注之前需要接受培训，了解标注指南和质量标准。
2. **审查：** 每个标注都由至少两个审查者进行审查，以确保标注的准确性和一致性。
3. **反馈：** 我们定期向标注者提供反馈，帮助他们改进标注质量。

#### A.5.4 标注者选择

我们通过以下标准选择标注者：

1. **语言能力：** 标注者需要具备良好的语言能力，能够编写流畅、自然的文本。
2. **专业知识：** 标注者需要具备相关领域的专业知识，例如编程、科学、历史等。
3. **可靠性：** 标注者需要具备良好的工作记录，能够按时完成标注任务。

### A.6 数据集污染

我们分析了 Llama 2 预训练数据与评估基准之间的潜在数据集污染。具体来说，我们检查了预训练数据中是否包含评估基准中的测试样本。

我们发现，在某些基准上，预训练数据中可能包含少量与测试样本相似的文本。然而，这些相似文本的数量非常少，不太可能对评估结果产生显著影响。

### A.7 模型卡

以下是对 Llama 2 模型的模型卡描述：

**模型概述。** Llama 2 是一系列预训练和微调的大语言模型，参数规模从 70 亿到 700 亿。Llama 2-Chat 是针对对话用例优化的微调版本。

**预期用途。** Llama 2 预期用于研究和商业用途，包括但不限于对话系统、文本生成、代码生成等。

**局限性。** Llama 2 可能存在以下局限性：

1. 可能生成与事实不符的信息（幻觉）。
2. 可能包含训练数据中的偏见。
3. 可能被恶意使用，例如生成虚假信息或有害内容。
4. 知识截止于预训练数据的日期，无法获取最新信息。

**伦理考虑。** 我们鼓励用户负责任地使用 Llama 2，并遵守相关的法律法规。我们还建议用户在部署 Llama 2 之前进行安全测试和调整。

---

*本翻译基于 arXiv:2307.09288v2 原文，仅供参考学习使用。*
