# trlX：一种用于大规模从人类反馈中进行强化学习的框架

> Alexander Havrilla\*, Maksym Zhuravinskyi, Duy Phung, Amrith Kumar, Jonathan Tow, Stanley Li, Amanda Bertsch, Dipam Chakraborty, Yizhe Huang, Akshita Bhagia, Martin Cai, Kshitij Gupta, Robert Eng, Ben Wang, Stella Biderman, Leo Gao, Thomas Wolf | CarperAI, Northeastern University, Université de Lorraine, LORIA, EleutherAI, Hugging Face, University of Maryland, Princeton University, University of Washington, Booz Allen Hamilton

\*工作完成于 CarperAI
通讯作者：ahavrilla3@gatech.edu

我们介绍 trlX，一个用于利用人类反馈通过强化学习（RLHF）大规模微调大型语言模型（LLM）的统一框架。trlX 实现了在线（online）和离线（offline）强化学习算法，支持训练规模达超过 700 亿参数的模型。我们通过利用模型架构中的共享冻结层、低秩适配器（LoRA）和 8-bit 优化器来强调低资源场景下的内存效率。为了在大规模场景下高效训练，我们通过 GPT-NeoX 和 NeMO-Megatron 库支持张量（tensor）、序列（sequence）和流水线（pipeline）并行化。trlX 支持基于 PPO 的在线方法和基于 ILQL 的离线方法。我们发现基于 ILQL 的微调在偏好胜率（preference win-rates）上与 PPO 相当，但计算成本仅为一小部分。我们在两个任务上对 trlX 进行评估：使用 TL;DR 数据集进行摘要学习，以及使用 Anthropic 的 Helpful and Harmless 数据集进行有用/无害偏好学习。我们观察到，基于 trlX 训练的模型在多个模型规模上的偏好胜率与已发表成果相当，其中 PPO 微调模型在 6B 和 20B 规模上相对于 SFT 基线的胜率超过 70%。我们将所有代码和模型开源，以支持进一步的 RLHF 研究。

---

## 摘要

从人类反馈中进行强化学习（RLHF）是将大型语言模型（LLM）与人类偏好对齐的重要技术。然而，由于训练框架的有限可用性和在线强化学习的计算挑战，RLHF 的学术研究一直受到限制。我们介绍 trlX，一个用于 RLHF 的统一框架，支持在线和离线方法，支持模型规模超过 700 亿参数。通过利用模型架构中的共享冻结层（Hydra）、低秩适配器（LoRA）和 8-bit 优化器，我们实现了显著的内存节省，使得资源受限的用户也能进行 RLHF 微调。我们在两个任务上进行评估：使用 TL;DR 数据集的摘要学习和使用 Anthropic 的 Helpful QA 数据集的有用/无害偏好学习。我们观察到，基于 trlX 训练的模型在多个模型规模上的偏好胜率与已发表成果相当，其中 PPO 微调模型在 6B 和 20B 规模上相对于监督微调（SFT）基线的胜率超过 70%。我们将所有代码和模型开源，以支持进一步的 RLHF 研究。

## 1 引言

大型语言模型（LLM）[4, 5, 6] 在自然语言理解和生成方面取得了显著进展，使得机器翻译、摘要和问答等领域的性能得到了提升。然而，这些模型通常是在纯粹的下一个 token 预测目标下训练的，这可能导致与人类偏好不一致的行为，如产生有害或有偏见的内容。为了使 LLM 与人类偏好对齐，研究者开发了从人类反馈中进行强化学习（RLHF），这已成为语言模型训练流水线中的关键阶段。在此阶段，收集到的人类偏好数据用于对预训练模型进行监督微调，然后训练一个奖励模型（reward model）。奖励模型将标量值赋给（提示，回复）对，这些值对应人类偏好。然后，可以使用近端策略优化（PPO）[7]（一种在线强化学习算法）对监督微调后的模型进行训练，以优化学习到的奖励模型。得到的模型与人类偏好更好地对齐，相对于基础模型的胜率可高达 80% [8, 1, 2]，并且生成更理想、更有害性更低的文本，且需要更少的提示。

然而，迄今为止，学术界对 RLHF 模型的研究一直受到缺乏开源训练框架和数据集的限制。通过 PPO 进行的在线强化学习计算成本高昂，并且难以扩展，因为我们在训练过程中必须在内存中存储模型的三份副本。为了解决这些问题，我们提出了 trlX：一个支持对高达超过 700 亿参数的语言模型进行在线和离线 RLHF 微调的库。在较小规模上，我们通过整合 Hydra 模型架构 [14]、LoRA 适配器 [15] 和 DeepSpeed [16] 等功能来强调低资源可访问性，这些功能结合使用时，可以将 GPT-J [17] 上的内存开销减少高达 75%，同时对获得的奖励影响最小。在此规模上的 trlX 训练兼容流行 Hugging Face Hub [18] 上支持的大多数编码器-解码器（encoder-decoder）和仅解码器（decoder-only）架构。对于超过 200 亿参数的大规模训练任务，我们通过 GPT-NeoX 库 [19] 和 NeMO-Megatron [20] 实现对张量（tensor）、序列（sequence）和流水线（pipeline）并行化的支持。

trlX 还支持隐式语言 Q 学习（Implicit Language Q Learning, ILQL），作为在线 RL 方法的离线替代方案。我们发现通过 ILQL 进行微调在偏好胜率上接近 PPO，但计算成本仅为一小部分。此外，ILQL 对奖励模型过拟合更具鲁棒性，而在线算法可能会受到此问题的影响。为了评估我们的框架，我们提供了来自 RLHF 文献的知名论文的开源复现，包括从人类反馈中学习摘要 [21] 和通用语言助手的有用/无害偏好学习 [1]。我们发现使用 trlX 训练的模型在人类评估者判定的偏好胜率上与原始工作相当，验证了我们的实现。我们开源了所有监督微调模型、奖励模型和 RLHF 模型以支持进一步研究，以及训练框架 trlX。

总之，我们做出以下贡献：

- trlX 作为一个功能完整的开源库，用于从人类反馈中进行强化学习，支持高达超过 700 亿参数的模型规模。这包括实现多个知名 RLHF 模型的首个已知开源复现的基准示例，为训练和评估提供见解。
- 对基于离线 RL 的大规模偏好学习微调进行新颖评估。
- 发布训练流水线中涉及的所有规模的所有模型，包括监督微调模型、奖励模型和 RL 微调模型。

## 2 背景

**从人类反馈中进行强化学习** 从人类反馈中进行强化学习尝试通过将对智能体行为的某种形式的人类反馈纳入其中，来提高智能体性能——无论是在样本效率上还是在下游任务性能上 [22, 23, 21]。我们关注 Stiennon 等人 [21] 概述的三阶段微调流水线：研究者首先构建一个人工标注偏好数据集，然后训练一个奖励模型来预测这些偏好，最后训练一个策略（policy）来最大化所得奖励模型的分数。类似结构的流水线已被用于训练许多最新的交互式自然语言辅助工具 [24, 2, 1, 14]。

然而，大规模收集人类偏好在成本上可能效率不高。Bai 等人 [25] 的近期工作提出了通过使用合成 AI 偏好而非直接的人类标注来实现可处理的监督。类似的工作如 Honovich 等人 [26]、Wang 等人 [27, 28] 通过查询已对齐的模型（如 text-davinci-003）来生成遵循指令的数据集，这些模型同时生成任务和遵循指令的回复。

**可扩展的训练框架** 有许多值得注意的分布式训练框架可用于大规模语言模型的预训练和微调，每个框架实现了多种并行化方案。这些包括 DeepSpeed [16]、Megatron-LM [29, 20]、结合了 DeepSpeed 和 Megatron-LM 的 GPT-NeoX 库 [19]、Fairseq [30, 31] 以及用于 TPU 训练的 T5X [32]。

然而，这些框架没有一个是专门为支持通过 RL 进行微调而设计的，因此需要进行大量的改造工作。近期，专门针对 RL 的微调库已经出现。RL4LM [33] 实现了通过从人类反馈中进行强化学习来微调中等规模语言模型的在线算法，并支持令人印象深刻的多种任务和度量指标。TRL [34] 最初是一个用于 Transformer 强化学习的小规模库，是 Ziegler 等人 [35] 在 PyTorch 中的重新实现，用于进行基于情感的微调。后来它以类似于 trlX 的方式进行了扩展，包括通过 Hugging Face accelerate 支持 DeepSpeed 训练。最近，DeepSpeed-Chat [36] 发布，允许进行更大规模的模型训练并具有更好的吞吐量。然而，它们不支持 trlX 中更高级的并行化形式。

## 3 使用 trlX 进行训练

trlX 旨在帮助减轻低资源用户面临的高计算成本，同时仍允许高资源用户获得良好的性能。大致上，我们将用户分为三种不同的资源配置：

1. **单 GPU 用户**。在此低资源用例中，我们推荐使用原生 PyTorch 集成以及内存节省功能，包括 Hydra 架构、低秩适配器和 8-bit Adam [37]。
2. **多 GPU 用户**。在此中资源用例中，我们推荐使用 Hugging Face accelerate [38] 与 DeepSpeed 以及内存节省功能的集成。我们使用此集成在一个节点上轻松训练高达 200 亿参数的语言模型。
3. **多节点用户**。在此高资源用例中，我们推荐使用 GPT-NeoX 或 NeMO-Megatron 的集成，这提供了比 accelerate 和 DeepSpeed 更高的 GPU 效率和扩展能力。我们使用此集成训练高达 700 亿参数的模型：这是开源 RLHF 模型前所未有的规模。

该框架围绕一个基础训练器（base trainer）构建，特定集成的训练器可以从中继承。独立地，在线和离线算法被实现，允许在不同的集成中重用。特别是，trlX 支持用于在线 RL 的 PPO 和 A2C，以及用于离线 RL 的 ILQL。在线 PPO 训练中最昂贵的部分是模型 rollout，其耗时可能比前向和反向传播的总和长 10 倍。为了有效最大化 rollout 和优化步骤的批大小，我们通过一个协调器（orchestrator）类将 rollout 推理批大小与 PPO 批大小解耦。这使得在线模型能够执行批量 rollout，从而减少对每个模型进行推理所花费的瓶颈时间。

我们与 Hugging Face 生态系统紧密集成，允许在 Hugging Face Hub 上训练大多数编码器-解码器和仅解码器模型，包括广泛使用的模型如 T5 [39] 和 Flan-T5 [40]、GPT-J [17]、Pythia [41]、OPT [31] 和 LLaMA [42, 43]。

通过人类反馈使用 PPO 微调大型语言模型在内存和 FLOPs 方面成本高昂，需要用户在内存中始终存储一个学生模型、一个参考模型和一个相似规模的奖励模型。此外，强化学习以其对超参数选择的脆弱性而闻名，通常需要广泛搜索才能找到最优设置。为了缓解这些成本，我们支持参数节省技术，如 LoRA [15] 和 Hydra 模型架构设计 [14]，后者允许在策略（policy）、值函数（value）和参考网络之间共享冻结层。类似地，ILQL 模型需要非标准的 Q 值头（Q-value heads）和生成能力，这些在两种集成中分别实现。

### 3.1 内存和计算节省功能

为了基准测试内存和计算节省功能对性能的影响，我们对模型规模从 1.25 亿到 200 亿参数的情感基线任务进行了一系列实验。对于每个模型规模，我们在 Hydra 架构中冻结一定比例的模型层，并观察对奖励、训练时间和所需内存的影响。我们还尝试将不同秩（rank）的 LoRA 适配器应用于所有 Transformer 矩阵。模型取自 Pythia 系列 [41]，在 8 块 80GB A100 上以全局批大小 32 训练 6000 步。

图 1 显示，在所有模型规模下，大约可以冻结一半的层而不会影响可获得的最大奖励。有趣的是，冻结模型除两层之外的所有层对较大模型的不利影响更大。我们推测这是因为较大模型在其中间层学习了大多数复杂的任务特定特征，而下游层仅进行微小的调整。

> **图 1：** 作为模型未冻结层数函数的最大可获得奖励。每个模型在冻结大约一半层时达到其最大可获得奖励。

图 2 显示了层冻结对内存节省的影响。这对于较大模型尤其有用，因为否则我们必须将冻结的参考模型单独加载到 GPU 内存中进行推理。当除两层外所有层都冻结时，我们为参考模型节省了除两层之外所有层的内存和计算成本。特别是，对于较大模型规模，我们可以在仍达到最大奖励的同时节省近 50% 的所需内存。

在更非玩具的问题上，我们还观察到层冻结通过减少与基础模型的 KL 散度来帮助稳定训练过程。这有助于缓解通过冻结参考模型进行基于 KL 惩罚的需求，在某些情况下甚至可以完全移除。此外，在某些情况下，部分冻结甚至提供了有益的归纳偏置（inductive bias），使模型能够达到比所有参数未冻结训练时更高的奖励。

> **图 2：** Hydra 内存消耗作为未冻结层数的函数。

基于 LoRA 的微调也可以看到类似的内存节省和正则化益处。当调优所有层时，使用 LoRA 秩 1 训练在情感基准上达到了最大奖励。在 69 亿参数规模下，LoRA 训练仅微调了模型参数的 0.03%，并将内存使用减少了 3 倍。LoRA 训练可以与层冻结结合使用以进一步节省内存。通过这两种优化，即使是中等规模的模型也可以在单块消费级 GPU 上进行 RLHF。这些内存节省和性能优势也延续到了使用 ILQL 的离线训练模式（见表 2）。我们推测，限制参数更新的秩以及冻结模型层为在线和离线 RL 训练都提供了有益的正则化效果。

### 3.2 与其他框架的比较

见表 1，其中概述了类似库中 trlX 关键功能的存在情况。trlX 是唯一支持离线 RL 微调的框架，也是唯一支持通过流水线（pipeline）、序列（sequence）和张量（tensor）并行化进行大规模模型微调的框架。此外，我们的功能最为完整，包括用于参数高效微调和分布式超参数搜索的工具。我们包含 10 多个基准示例，为多个知名的 RLHF 任务提供端到端流水线。

> **表 1：** trlX 与其他库的功能比较。\* trl 支持朴素流水线并行化，允许运行更大模型但效率远低。

| | RL 算法 | | 并行化策略 | | 功能 | |
|---|---|---|---|---|---|---|
| | 在线 | 离线 | 张量 | 流水线 | 序列 | LoRA | 搜索 |
| RL4LM | ✓ | | | | | | ✓ |
| trl | ✓ | | | ✓\* | | | |
| DS Chat | ✓ | | ✓ | ✓ | ✓ | ✓ | |
| trlX（ours） | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

**DeepSpeed 与 NeMO Megatron**  trlX 在在线 RL 方面与现有的开源 RLHF 实现具有竞争力。我们与 DeepSpeed-Chat [36] 进行比较，这是一个同时期的工作，实现了用于 LLM 的 PPO RLHF。见表 3 的性能比较。请注意，DeepSpeed-Chat 在 30B 和 60B 参数规模下的性能数据使用了基于 LoRA 的训练，而 trlX 使用全参数微调。我们将其他基准测试设置与已发布的 DeepSpeed-Chat 脚本保持一致。

> **表 2：** ILQL 在 Anthropic Helpful QA 对话数据集上达到最大奖励的时间基准。所有非 LoRA 超参数与基础模型相同，学习率设置为 $2.0 \times 10^{-4}$ 。对于 GPT-NeoX-20B LoRA，最后 8 层使用 LoRA 训练；对于 Pythia 6.9B LoRA，所有层使用 LoRA 训练。

| 模型 | 最大奖励 | 时间（分钟） | GPU 数 |
|---|---|---|---|
| GPT-NeoX 20B | -1.88 | 156 | 32 |
| GPT-NeoX 20B LoRA | -1.89 | 28 | 16 |
| Pythia 6.9B | -1.77 | 286 | 16 |
| Pythia 6.9B LoRA | -1.68 | 58 | 16 |

> **表 3：** trlX 与 DeepSpeed-Chat 在 OPT 架构上在线 RL（PPO）训练速度的比较，单位为 samples/s/GPU。\*DeepSpeed-Chat 30B 和 60B 的性能数据根据 (Yao et al., 2023) 表 2 转换而来，使用 64 块 GPU 在 131.9k 样本上训练 4 小时。† 对于 OPT 66B，我们使用 Hydra，50% 参数可训练。

| 参数 | DS-Chat | trlX |
|---|---|---|
| OPT 1.3B | 2.1 | 2.0 |
| OPT 6.7B | 0.44 | 0.41 |
| OPT 30B | 0.14\* (LoRA) | 0.12 |
| OPT 60B | 0.076\* (LoRA) | 0.043† |

## 4 基准测试与结果

我们在两个流行的 RLHF 任务上对 trlX 进行基准测试：OpenAI 的摘要学习 TL;DR 数据集 [21] 和 Anthropic 的有用 QA 数据集 [44]。我们开源所有相关代码和模型以供进一步研究。

**训练设置和超参数**  除非另有说明，我们在所有训练运行中使用附录中列出的相同固定超参数集。我们发现良好性能对以下参数特别敏感：

- **批大小（Batch size）**：使用每次迭代至少 128 个全局样本的较大批大小。这降低了运行之间的方差并稳定了性能。
- **奖励归一化（Reward normalization）**：在收集 rollouts 后，我们使用运行标准差估计对所有奖励进行归一化。我们发现这种归一化（值得注意的是，不减去运行均值）显著提高了性能。此外，我们在批次级别对优势（advantages）进行第二次归一化。
- **学习率（Learning rate）**：学习率选择为 $5 \times 10^{-6}$ ，比监督微调小一个数量级。

### 4.1 摘要

**设置**  从人类反馈中学习摘要 [21] 引入了 TL;DR 数据集。第一部分包含 129,772 条 Reddit 帖子，用于监督微调。第二部分用于训练奖励模型，包含 92,534 个训练样本和 83,629 个验证样本。

我们首先从 Pythia 系列中对 440M、1.4B、6.9B 和 20B 模型在 SFT 数据集上进行微调，训练监督微调（SFT）模型。我们使用 AdamW 优化器，学习率为 $1 \times 10^{-5}$ ，随后使用具有短预热阶段的线性调度器。最佳模型通过验证集上的 Average-ROUGE 分数选择。

为了训练我们的奖励模型（RM），我们使用 SFT 检查点进行初始化，将因果头（causal head）替换为标量输出。使用数据集的第二部分，我们最小化成对偏好损失 [21]。我们发现表现最好的奖励模型是 69 亿参数的 GPT-J（6B），以批大小 32 进行训练。

有了训练好的奖励模型，我们现在可以对从 440M 到 20B 的模型进行基于 RL 的微调。训练中使用来自 TL;DR 数据集两部分的帖子。我们使用表现最好的奖励模型 6.9B 作为所有实验的奖励信号。为了在线训练模型，我们从 SFT 检查点初始化，并使用 PPO，每批次进行 4 个 epoch，KL 惩罚系数为 0.005。我们保持 8 层未冻结。为了训练离线模型，我们将数据集两部分的帖子及其相关摘要分别标记为 \pm1。我们注意到，这种标记方式比使用学习到的 RM 标记数据效果更好。然后使用这个奖励标记的数据集通过 ILQL 算法训练基础模型。值得注意的是，我们没有从 SFT 检查点初始化，因为我们在离线模式下看到收益极小。

**结果**  我们在附录中附加了一个表格，显示在 6.9B 模型上使用 SFT 和 PPO 训练的 TL;DR 数据集测试集上的 ROUGE 分数。与 [21] 相比，ROUGE 分数的趋势相似，SFT 模型的表现略优于 PPO 模型。

更关键的是，我们进行人工评估，以更好地评估我们的在线 PPO 和离线 ILQL 模型在符合人类偏好方面与基线 SFT 的比较。为此，我们从数据集的测试部分中选择一个提示子集中的故事，并要求标注者在两个候选摘要中进行选择。特别地，对于每个模型规模，我们进行两项评估：PPO 与 SFT 的比较，以及 ILQL 与 SFT 的比较。除了在两个候选摘要之间进行选择外，我们还要求用户在 1-7 分的 Likert 量表上对覆盖率（coverage）、清晰度（clarity）和不一致性（inconsistency）进行评分。结果报告在图 3 和图 4 中。

我们通过相对于其对应 SFT 基线的相对改进来评估每个模型，部分是为了展示 RLHF 即使在较小模型规模上的有效性。

> **图 3：** ILQL、PPO 微调模型在摘要任务上相对于其对应 SFT 基线的胜率。注意比较是针对相同规模的 SFT 基线进行的（例如 6B SFT 对比 6B PPO）。OpenAI 基线（其 6B 模型相对于人工生成摘要的胜率）被包括作为参考。

> **图 4：** 20B 模型在覆盖率、清晰度和不一致性上的 Likert 分数。

**ILQL 以微小代价略逊于 PPO**  图 3 显示 ILQL 和 PPO 在大多数模型规模上均达到超过 10% 的胜率。在 6B 和 20B 规模下，我们的 PPO 模型相对于其 SFT 基线实现了超过 70% 的胜率。我们还看到 ILQL 模型非常有竞争力，尽管其训练所需的计算量要少得多。有趣的是，我们观察到 ILQL 生成的摘要明显更短、更简洁，比 PPO 甚至 SFT 基线更简洁。尽管如此，由于对关键点的覆盖率更好，ILQL 通常仍优于较长的 SFT 基线。这表明可能可以开发更复杂的离线训练方法，作为 PPO 的计算效率更高的替代方案。

### 4.2 有用 QA 对话

**设置**  Helpful Harmless RLHF [44]（简称 HH-RLHF）包含用户和 AI 助手之间的 118k 个样本交互。它可以进一步分为三部分：初始的 42k 个由提示语言模型创建的提示-回复三元组数据集、40k 个通过对同一提示模型的回复进行重新排名创建的样本，以及最后一组 22k 个来自初始 RLHF 模型的回复。前两部分称为静态子集。我们使用静态数据集中的有用部分进行训练和评估。每个交互样本包含一个以用户话语结束的对话历史。随后是一个偏好的或选择的助手回复和一个被拒绝的回复。

我们通过从 160M、1.4B、6.9B 和 20B 参数的普通模型在每个样本的选择回复上进行微调来训练监督微调 SFT 基线模型。训练进行一个 epoch，学习率 $\text{lr} = 5 \times 10^{-5}$ 。注意我们在对话历史上屏蔽损失，仅对回复 token 进行反向传播。这作为我们的基线。

然后我们独立训练大小为 160M 到 20B 的奖励模型。与摘要任务一样，我们通过从 SFT 模型初始化来进行热启动。如上所述，我们训练一个 epoch，学习率 $\text{lr} = 5 \times 10^{-6}$ 。我们观察到添加监督热启动可将测试准确率提高高达 2%。我们表现最好的模型是 60 亿参数的 GPT-J，在静态测试集上达到 0.72 的准确率。我们将其用作所有基于 RL 微调的默认奖励模型 RM。

有了我们的 RM，我们可以使用 trlX 微调我们的基线 SFT 模型。我们的训练数据集包含一组从整个静态数据集中提取的输入提示。我们通过由 text-davinci-003 合成生成的多轮提示和回复来增强该数据集。关于如何创建此合成数据的详细信息可以在附录中找到。总共这构成了我们 RL 训练数据集的 200k 个提示。

训练步数保持在 9000 步，有效批大小为 128。对于不同的模型规模，使用介于 $1 \times 10^{-6}$ 和 $8 \times 10^{-6}$ 之间的学习率。我们保持 8 层未冻结。使用恒定的 KL 惩罚系数 0.005。我们将由此产生的模型系列称为 PPO。

> **图 5：** 奖励模型在测试集上的准确率与训练比较样本数的关系。我们观察到每 10k 个训练样本，模型准确率提升约 1.8%。

> **图 6：** 模型在 HellaSwag、TriviaQA、LAMBADA、ARC Easy、ARC Challenge 和 OpenBook QA 上的零样本平均性能。完整结果表格见附录。

特别地，我们发现使用足够大的批次进行训练以确保稳健的 PPO 梯度估计至关重要。此外，如果训练时间过长或 KL 惩罚过弱，我们观察到模型严重过拟合于奖励模型。我们实行早停（early stopping）以防止这种过拟合。较大的批大小还具有通过简单减少提示数据集上的总步数来减轻过拟合的附加效果。

除了 PPO，我们还使用 ILQL 训练从 160M 到 20B 的模型。我们为选择的轨迹分配奖励 +1，为拒绝的轨迹分配奖励 -1。我们将由此产生的模型系列称为 ILQL。令人惊讶的是，\pm1 奖励分配在经验上优于通过从 RM 学习的奖励来标记选择和拒绝的回复。我们相信这是因为虽然 RM 奖励更密集，但在某些情况下也可能不准确，因此对于给定对话，预期不等式 $r_{\text{chosen}} > r_{\text{rejected}}$ 不成立，从而引入噪声。而 \pm1 分配则忠实于底层的人类偏好。这种分配还有一个额外的好处，即需要的计算量大大减少，因为不需要学习奖励模型。

除了上述模型，我们还训练和评估了最后一组 Vanilla-PPO，它通过我们的 RM 应用基于 PPO 的 RL 微调，而不从监督 SFT 检查点初始化。我们发现这仅对较大模型（6B 和 20B）可行，它们能够成功优化奖励。这突显了为足够困难的任务和较弱的模型收集监督微调数据的重要性。

**结果**  然后我们使用 Gao 等人 [45] 在一组常见的学术基准上评估普通模型（vanilla models）、SFT 模型、PPO 模型、Vanilla-PPO 模型和 ILQL 模型，包括 LAMBADA、ARC、OpenBookQA、TriviaQA 和 HellaSwag。图 6 绘制了每个模型类别在基准测试上的平均准确率。完整表格包含在附录中。我们发现监督微调显著影响性能。我们注意到，如果操作不当（例如在整个对话而非仅在回复上进行微调），效果会更加明显。在 SFT 基础上进行基于 RL 的微调略微改善结果但不显著。

> **图 7：** 提示模型（prompted）、PPO RLHF 和 ILQL RLHF 模型在 160M、1.4B、6.9B 和 20B 参数规模下的胜率。比较是针对相同规模的 SFT 基线进行的（例如 6.9B SFT 对比 6.9B PPO）。

**"对齐税"来源于 SFT**  令人惊讶的是，相反地，在 Vanilla-PPO 模型中不使用 SFT 进行的 RL 微调产生的惩罚要小得多，并且在 6.9B 的情况下甚至略微提高了性能。这回答了关于使用 RLHF 微调时是否存在对齐税的问题。OpenAI 报告了 InstructGPT [2] 中的这种税，特别是在监督微调之后，但没有报告严格基于 RL 的微调的结果。相反，Anthropic [1] 展示了对于足够大的模型，在基于 RL 的微调后基准性能有小幅提升，但没有使用 SFT 热启动。这些结果表明这种税主要是由于监督微调而非基于 RL 的微调。我们注意到一种技术是将预训练数据混合到 SFT 和 RL 微调分布中，如 Ouyang 等人 [2] 所做。

这表明我们需要一个高质量的 SFT 训练数据集，以在减轻基准性能下降的同时，适当地学习期望的行为。

除了自动基准评估外，我们还进行了一项人工评估，其中标注者在一个模型生成的回复和同等规模的监督微调基线回复之间进行选择。结果报告在图 7 中。注意我们考察的是模型相对于同规模基线的胜率，这与之前的工作 [1, 2] 不同。我们在附录中附上了标注者说明。

**RLHF 也可以使较小模型受益**  在所有模型规模上，我们观察到 PPO 训练模型和 SFT 基线之间至少 60% 的胜率。此外，离线训练的 ILQL 模型非常有竞争力，同样以很小的计算量实现了至少 60% 的胜率。进一步地，我们定性地观察到 ILQL 对奖励过拟合的鲁棒性显著优于基于在线 PPO 的微调。相比之下，在线模式需要结合大批次和早停来缓解这种奖励过拟合。最后，我们注意到提示基线（prompted baseline）已经相对较强，很可能是因为对于较大模型来说该任务已经充分处于分布内。这一点也得到以下事实的支持：大型普通模型在没有监督微调的情况下也能成功优化奖励。

除了收集胜率外，我们还收集了 1-7 分 Likert 量表上的回复有用性（Helpfulness）、有害性（Harmfulness）和诚实性（Honesty）分数。结果报告在附录中。

## 5 结论

**伦理**  我们提出 trlX 作为一个开源框架，用于利用从人类反馈中进行强化学习来大规模训练大型语言模型。即使通过 RLHF 进行微调，LLM 在推理时仍然容易出现幻觉和偏见，因此需要进一步研究缓解措施。我们希望研究者能够将 trlX 作为 RLHF 流水线的开源实现，以促进这项研究。

**局限**  虽然 PPO 性能很高，但它存在许多局限性，包括在实现过程中的困难以及训练时的超参数敏感性。像 ILQL 这样的离线方法既更容易实现，也更具计算效率，但仍未达到与 PPO 相同的性能。

## 参考文献

[1] Bai, Y., Jones, A., Ndousse, K., Askell, A., Chen, A., DasSarma, N., Drain, D., Fort, S., Ganguli, D., Henighan, T. J., Joseph, N., Kadavath, S., Kernion, J., Conerly, T., El-Showk, S., Elhage, N., Hatfield-Dodds, Z., Hernandez, D., Hume, T., Johnston, S., Kravec, S., Lovitt, L., Nanda, N., Olsson, C., Amodei, D., Brown, T. B., Clark, J., McCandlish, S., Olah, C., Mann, B., and Kaplan, J. Training a helpful and harmless assistant with reinforcement learning from human feedback. *ArXiv*, abs/2204.05862, 2022a.

[2] Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C. L., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., Schulman, J., Hilton, J., Kelton, F., Miller, L. E., Simens, M., Askell, A., Welinder, P., Christiano, P. F., Leike, J., and Lowe, R. J. Training language models to follow instructions with human feedback. *ArXiv*, abs/2203.02155, 2022.

[3] Stiennon, N., Ouyang, L., Wu, J., Ziegler, D., Lowe, R., Voss, C., Radford, A., Amodei, D., and Christiano, P. F. Learning to summarize with human feedback. *Advances in Neural Information Processing Systems*, 33:3008–3021, 2020.

[4] Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., Agarwal, S., Herbert-Voss, A., Krueger, G., Henighan, T., Child, R., Ramesh, A., Ziegler, D., Wu, J., Winter, C., Hesse, C., Chen, M., Sigler, E., Litwin, M., Gray, S., Chess, B., Clark, J., Berner, C., McCandlish, S., Radford, A., Sutskever, I., and Amodei, D. Language models are few-shot learners. In Larochelle, H., Ranzato, M., Hadsell, R., Balcan, M. F., and Lin, H. (eds.), *Advances in Neural Information Processing Systems*, volume 33, pp. 1877–1901. Curran Associates, Inc., 2020.

[5] Devlin, J., Chang, M.-W., Lee, K., and Toutanova, K. Bert: Pre-training of deep bidirectional transformers for language understanding. *ArXiv*, abs/1810.04805, 2019.

[6] Raffel, C., Shazeer, N. M., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., and Liu, P. J. Exploring the limits of transfer learning with a unified text-to-text transformer. *ArXiv*, abs/1910.10683, 2019.

[7] Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O. Proximal policy optimization algorithms. *ArXiv*, abs/1707.06347, 2017.

[8] Askell, A., Bai, Y., Chen, A., Drain, D., Ganguli, D., Henighan, T., Jones, A., Joseph, N., Mann, B., DasSarma, N., et al. A general language assistant as a laboratory for alignment. *arXiv preprint arXiv:2112.00861*, 2021.

[9] Andonian, A., Anthony, Q., Biderman, S., Black, S., Gali, P., Gao, L., Hallahan, E., Levy-Kramer, J., Leahy, C., Nestler, L., et al. Gpt-neox: Large scale autoregressive language modeling in pytorch. *GitHub Repo*, 2021.

[10] Black, S., Biderman, S., Hallahan, E., Anthony, Q., Gao, L., Golding, L., He, H., Leahy, C., McDonell, K., Phang, J., Pieler, M., Prashanth, U. S., Purohit, S., Reynolds, L., Tow, J., Wang, B., and Weinbach, S. GPT-NeoX-20B: An open-source autoregressive language model. In *Proceedings of the ACL Workshop on Challenges & Perspectives in Creating Large Language Models*, 2022.

[11] Biderman, S., Schoelkopf, H., Anthony, Q. G., Bradley, H., O'Brien, K., Hallahan, E., Khan, M. A., Purohit, S., Prashanth, U. S., Raff, E., Skowron, A., Sutawika, L., and van der Wal, O. Pythia: A suite for analyzing large language models across training and scaling. *ArXiv*, abs/2304.01373, 2023.

[12] Gao, L., Tow, J., Biderman, S., Black, S., DiPofi, A., Foster, C., Golding, L., Hsu, J., McDonell, K., Muennighoff, N., et al. A framework for few-shot language model evaluation. *Version v0.0.1. Sept*, 2021.

[13] Biderman, S., Bicheno, K., and Gao, L. Datasheet for the pile. *arXiv preprint arXiv:2201.07311*, 2022.

[14] Glaese, A., McAleese, N., Trkebacz, M., Aslanides, J., Firoiu, V., Ewalds, T., Rauh, M., Weidinger, L., Chadwick, M., Thacker, P., Campbell-Gillingham, L., Uesato, J., Huang, P.-S., Comanescu, R., Yang, F., See, A., Dathathri, S., Greig, R., Chen, C., Fritz, D., Elias, J. S., Green, R., Mokr'a, S., Fernando, N., Wu, B., Foley, R., Young, S., Gabriel, I., Isaac, W. S., Mellor, J. F. J., Hassabis, D., Kavukcuoglu, K., Hendricks, L. A., and Irving, G. Improving alignment of dialogue agents via targeted human judgements. *ArXiv*, abs/2209.14375, 2022.

[15] Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., and Chen, W. Lora: Low-rank adaptation of large language models. *ArXiv*, abs/2106.09685, 2021.

[16] Rajbhandari, S., Rasley, J., Ruwase, O., and He, Y. Zero: Memory optimizations toward training trillion parameter models. *SC20: International Conference for High Performance Computing, Networking, Storage and Analysis*, pp. 1–16, 2019.

[17] Wang, B. and Komatsuzaki, A. GPT-J-6B: A 6 Billion Parameter Autoregressive Language Model. https://github.com/kingoflolz/mesh-transformer-jax, May 2021.

[18] Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C., Moi, A., Cistac, P., Rault, T., Louf, R., Funtowicz, M., et al. Huggingface's transformers: State-of-the-art natural language processing. *arXiv preprint arXiv:1910.03771*, 2019.

[19] Andonian, A., Anthony, Q., Biderman, S., Black, S., Gali, P., Gao, L., Hallahan, E., Levy-Kramer, J., Leahy, C., Nestler, L., et al. Gpt-neox: Large scale autoregressive language modeling in pytorch. *GitHub Repo*, 2021.

[20] Kuchaiev, O., Li, J., Nguyen, H., Hrinchuk, O., Leary, R., Ginsburg, B., Kriman, S., Beliaev, S., Lavrukhin, V., Cook, J., et al. Nemo: a toolkit for building ai applications using neural modules. *arXiv preprint arXiv:1909.09577*, 2019.

[21] Stiennon, N., Ouyang, L., Wu, J., Ziegler, D., Lowe, R., Voss, C., Radford, A., Amodei, D., and Christiano, P. F. Learning to summarize with human feedback. *Advances in Neural Information Processing Systems*, 33:3008–3021, 2020.

[22] Knox, W. B. and Stone, P. Interactively shaping agents via human reinforcement: The tamer framework. In *The Fifth International Conference on Knowledge Capture*, September 2009.

[23] Christiano, P. F., Leike, J., Brown, T. B., Martic, M., Legg, S., and Amodei, D. Deep reinforcement learning from human preferences. *ArXiv*, abs/1706.03741, 2017.

[24] Nakano, R., Hilton, J., Balaji, S. A., Wu, J., Ouyang, L., Kim, C., Hesse, C., Jain, S., Kosaraju, V., Saunders, W., Jiang, X., Cobbe, K., Eloundou, T., Krueger, G., Button, K., Knight, M., Chess, B., and Schulman, J. Webgpt: Browser-assisted question-answering with human feedback. *ArXiv*, abs/2112.09332, 2021.

[25] Bai, Y., Kadavath, S., Kundu, S., Askell, A., Kernion, J., Jones, A., Chen, A., Goldie, A., Mirhoseini, A., McKinnon, C., et al. Constitutional ai: Harmlessness from ai feedback. *arXiv preprint arXiv:2212.08073*, 2022b.

[26] Honovich, O., Scialom, T., Levy, O., and Schick, T. Unnatural instructions: Tuning language models with (almost) no human labor. *arXiv preprint arXiv:2212.09689*, 2022.

[27] Wang, Y., Kordi, Y., Mishra, S., Liu, A., Smith, N. A., Khashabi, D., and Hajishirzi, H. Self-instruct: Aligning language model with self generated instructions. *arXiv preprint arXiv:2212.10560*, 2022a.

[28] Wang, Y., Mishra, S., Alipoormolabashi, P., Kordi, Y., Mirzaei, A., Arunkumar, A., Ashok, A., Dhanasekaran, A. S., Naik, A., Stap, D., et al. Supernaturalinstructions: Generalization via declarative instructions on 1600+ nlp tasks. *arXiv preprint arXiv:2204.07705*, 2022b.

[29] Shoeybi, M., Patwary, M., Puri, R., LeGresley, P., Casper, J., and Catanzaro, B. Megatron-lm: Training multi-billion parameter language models using model parallelism. *ArXiv*, abs/1909.08053, 2019.

[30] Ott, M., Edunov, S., Baevski, A., Fan, A., Gross, S., Ng, N., Grangier, D., and Auli, M. fairseq: A fast, extensible toolkit for sequence modeling. In *Proceedings of NAACL-HLT 2019: Demonstrations*, 2019.

[31] Zhang, S., Roller, S., Goyal, N., Artetxe, M., Chen, M., Chen, S., Dewan, C., Diab, M., Li, X., Lin, X. V., Mihaylov, T., Ott, M., Shleifer, S., Shuster, K., Simig, D., Koura, P. S., Sridhar, A., Wang, T., and Zettlemoyer, L. Opt: Open pre-trained transformer language models. *ArXiv*, abs/2205.01068, 2022.

[32] Roberts, A., Chung, H. W., Levskaya, A., Mishra, G., Bradbury, J., Andor, D., Narang, S., Lester, B., Gaffney, C., Mohiuddin, A., Hawthorne, C., Lewkowycz, A., Salcianu, A., van Zee, M., Austin, J., Goodman, S., Soares, L. B., Hu, H., Tsvyashchenko, S., Chowdhery, A., Bastings, J., Bulian, J., Garcia, X., Ni, J., Chen, A., Kenealy, K., Clark, J. H., Lee, S., Garrette, D., Lee-Thorp, J., Raffel, C., Shazeer, N., Ritter, M., Bosma, M., Passos, A., Maitin-Shepard, J., Fiedel, N., Omernick, M., Saeta, B., Sepassi, R., Spiridonov, A., Newlan, J., and Gesmundo, A. Scaling up models and data with t5x and seqio. *arXiv preprint arXiv:2203.17189*, 2022.

[33] Ramamurthy, R., Ammanabrolu, P., Brantley, K., Hessel, J., Sifa, R., Bauckhage, C., Hajishirzi, H., and Choi, Y. Is reinforcement learning (not) for natural language processing?: Benchmarks, baselines, and building blocks for natural language policy optimization. *ArXiv*, abs/2210.01241, 2022.

[34] Leandro, V. W. Transformer reinforcement learning. https://github.com/lvwerra/trl, 2019.

[35] Ziegler, D. M., Stiennon, N., Wu, J., Brown, T. B., Radford, A., Amodei, D., Christiano, P., and Irving, G. Fine-tuning language models from human preferences. *ArXiv*, abs/1909.08593, 2019.

[36] Yao, Z., Aminabadi, R. Y., Ruwase, O., Rajbhandari, S., Wu, X., Awan, A. A., Rasley, J., Zhang, M., Li, C., Holmes, C., Zhou, Z., Wyatt, M., Smith, M., Kurilenko, L., Qin, H., Tanaka, M., Che, S., Song, S. L., and He, Y. DeepSpeed-Chat: Easy, Fast and Affordable RLHF Training of ChatGPT-like Models at All Scales. *arXiv preprint arXiv:2308.01320*, 2023.

[37] Dettmers, T., Lewis, M., Shleifer, S., and Zettlemoyer, L. 8-bit optimizers via block-wise quantization. *ArXiv*, abs/2110.02861, 2021.

[38] Gugger, S., Debut, L., Thomas Wolf, T., Schmid, P., Mueller, Z., and Mangrulkar, S. Accelerate: Training and inference at scale made simple, efficient and adaptable. https://github.com/huggingface/accelerate, 2022.

[39] Raffel, C., Shazeer, N. M., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., and Liu, P. J. Exploring the limits of transfer learning with a unified text-to-text transformer. *ArXiv*, abs/1910.10683, 2019.

[40] Chung, H. W., Hou, L., Longpre, S., Zoph, B., Tay, Y., Fedus, W., Li, E., Wang, X., Dehghani, M., Brahma, S., Webson, A., Gu, S. S., Dai, Z., Suzgun, M., Chen, X., Chowdhery, A., Valter, D., Narang, S., Mishra, G., Yu, A. W., Zhao, V., Huang, Y., Dai, A. M., Yu, H., Petrov, S., hsin Chi, E. H., Dean, J., Devlin, J., Roberts, A., Zhou, D., Le, Q. V., and Wei, J. Scaling instruction-finetuned language models. *ArXiv*, abs/2210.11416, 2022.

[41] Biderman, S. R., Schoelkopf, H., Anthony, Q. G., Bradley, H., O'Brien, K., Hallahan, E., Khan, M. A., Purohit, S., Prashanth, U. S., Raff, E., Skowron, A., Sutawika, L., and van der Wal, O. Pythia: A suite for analyzing large language models across training and scaling. *ArXiv*, abs/2304.01373, 2023.

[42] Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.-A., Lacroix, T., Rozière, B., Goyal, N., Hambro, E., Azhar, F., Rodriguez, A., Joulin, A., Grave, E., and Lample, G. Llama: Open and efficient foundation language models. *ArXiv*, abs/2302.13971, 2023a.

[43] Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., Bashlykov, N., Batra, S., Bhargava, P., Bhosale, S., et al. Llama 2: Open foundation and fine-tuned chat models. *arXiv preprint arXiv:2307.09288*, 2023b.

[44] Ganguli, D., Lovitt, L., Kernion, J., Askell, A., Bai, Y., Kadavath, S., Mann, B., Perez, E., Schiefer, N., Ndousse, K., et al. Red teaming language models to reduce harms: Methods, scaling behaviors, and lessons learned. *arXiv preprint arXiv:2209.07858*, 2022.

[45] Gao, L., Biderman, S., Black, S., Golding, L., Hoppe, T., Foster, C., Phang, J., He, H., Thite, A., Nabeshima, N., et al. The pile: An 800gb dataset of diverse text for language modeling. *arXiv preprint arXiv:2101.00027*, 2020.

## 附录 A 模型超参数

见表 4 中所有模型训练类型的完整超参数列表。特别地，我们发现大批大小对 PPO 至关重要。KL 系数在 HH 训练中可以放宽，但对摘要任务很重要。此外，通过运行标准差估计对奖励进行缩放提供了小幅提升。

> **表 4：** 训练超参数。

| | SFT | RM | PPO |
|---|---|---|---|
| lr | $5 \times 10^{-5}$ | $5 \times 10^{-6}$ | $5 \times 10^{-6}$ |
| bs | 64 | 64 | 256 |
| 冻结层数 | N/A | 50% | 8 |
| 奖励归一化 | N/A | N/A | 缩放 |
| 目标 KL | N/A | N/A | 6 |
| $\lambda$ (GAE) | N/A | N/A | 0.95 |
| $\gamma$ (折扣) | N/A | N/A | 1 |
| 小批量归一化 | N/A | N/A | 真 |
| PPO epochs | N/A | N/A | 4 |
| KL 系数 | N/A | N/A | 0.01 |

## 附录 B LM 评估结果

所有 HH 模型在考虑的 lm-eval 基准上的完整分数可在表 5 中找到。注意，与 InstructGPT 的发现类似，SFT 模型表现较差。相比之下，纯 RL 微调对基准分数的影响可以忽略不计。

> **表 5：** lm-eval-harness 的结果表。

| 模型 | HellaSwag | LAMBADA | ARC Easy | ARC Challenge | OpenBookQA | TriviaQA |
|---|---|---|---|---|---|---|
| Pythia 160M Vanilla | 0.294 | 0.248 | 0.451 | 0.203 | 0.172 | 0.011 |
| Pythia 160M SFT | 0.291 | 0.215 | 0.453 | 0.206 | 0.17 | 0.013 |
| Pythia 160M PPO | 0.292 | 0.218 | 0.454 | 0.209 | 0.162 | 0.013 |
| Pythia 160M ILQL | 0.292 | 0.217 | 0.455 | 0.205 | 0.167 | 0.015 |
| Pythia 1.4B Vanilla | 0.402 | 0.458 | 0.581 | 0.265 | 0.198 | 0.048 |
| Pythia 1.4B SFT | 0.374 | 0.344 | 0.547 | 0.255 | 0.192 | 0.016 |
| Pythia 1.4B PPO | 0.369 | 0.360 | 0.543 | 0.257 | 0.196 | 0.011 |
| Pythia 1.4B ILQL | 0.392 | 0.439 | 0.563 | 0.255 | 0.189 | 0.015 |
| Pythia 6.9B Vanilla | 0.488 | 0.564 | 0.667 | 0.319 | 0.252 | 0.151 |
| Pythia 6.9B SFT | 0.432 | 0.398 | 0.606 | 0.309 | 0.236 | 0.027 |
| Pythia 6.9B PPO | 0.421 | 0.409 | 0.619 | 0.291 | 0.25 | 0.030 |
| Pythia 6.9B ILQL | 0.469 | 0.557 | 0.654 | 0.301 | 0.250 | 0.149 |
| Pythia 6.9B Vanilla-PPO | 0.495 | 0.605 | 0.670 | 0.312 | 0.28 | 0.149 |
| GPT-NeoX 20B Vanilla | 0.535 | 0.72 | 0.723 | 0.38 | 0.29 | 0.259 |
| GPT-NeoX 20B SFT | 0.462 | 0.505 | 0.664 | 0.343 | 0.252 | 0.041 |
| GPT-NeoX 20B PPO | 0.463 | 0.529 | 0.666 | 0.347 | 0.256 | 0.037 |
| GPT-NeoX 20B ILQL | 0.541 | 0.709 | 0.720 | 0.375 | 0.294 | 0.251 |
| GPT-NeoX 20B Vanilla-PPO | 0.548 | 0.618 | 0.731 | 0.383 | 0.3 | 0.227 |

## 附录 C 标注者说明

标注者说明链接在此。它们也显示在图 8 中以供参考。

> **标注指南：**
>
> 任务：你将看到提供给语言助手的提示和两个可能的回复：A 和 B。你将在回复 A 和回复 B 中选择哪个是对提示整体最有帮助的回复。有时你会被要求解释你的选择理由。
>
> **有帮助性（Helpfulness）**
>
> 一个有帮助的回复：
> - 对读者易于理解
> - 与用户的意图保持相关
> - 直截了当、切中要点
> - 为读者提供令人满意的回复
>
> 你还将在 1-5 分的量表上对有帮助性进行评分。同时对于两个回复，你将在 1-5 分的量表上对回复的正确性和潜在有害性进行评分。注意：如果用户的意图是请求有害的内容，那么有害性不应影响你对有帮助性在 A 和 B 之间的选择，因为该回复对用户来说是"有帮助的"。
>
> *定义草案改编自 InstructGPT 文档：*
>
> **正确性（Correctness）**
>
> 一个正确的回复：
> - 包含准确的事实
> - 不包含任何不准确的事实
> - 不误导用户
> - 不编造虚假细节
> - 如果输入提示包含不正确的事实，模型应反驳或纠正提示中的事实，并拒绝回答问题
>
> **有害性（Harmful）**
>
> 一个有害的回复，其本身或如果被付诸行动，包括以下任何一项：
> - 可能对人造成身体、心理或社会伤害
> - 可能造成设备或财产的损坏或损失
> - 可能损害环境
> - 可能损害对人类福祉必要的机构或资源，如学校、政府、医疗、电力等
> - 贬低群体成员，或使用针对群体的偏见性用语

> **图 8：** Helpful and Harmless QA 对话的标注者说明。

## 附录 D 合成奖励建模

我们进行了额外的实验系列，尝试使用 LLM 生成的合成数据来增强/替代由 Anthropic 收集的人类偏好数据的有用奖励建模。特别地，我们考虑两种生成合成偏好数据的方法：

1. 使用一个强大的指令微调模型（即 text-davinci-003）在 LLM 生成的两个候选回复之间进行选择。
2. 假设经过监督微调以使其有用的大型模型，在相同提示条件下，比小型模型生成更可取的回复。这通过模型规模定义了回复上的偏序关系。

为了验证第一种方法的有效性，我们评估了 GPT-NeoX-20B [10]、text-davinci-002 和 text-davinci-003 在 HH 测试集的有用部分上作为分类器的准确率。结果报告在表 6 中。我们发现即使是最好的可用 RLHF 模型 text-davinci-003，也只有 0.64 的准确率。这与我们最好的 GPT-J RM 的 0.71 准确率相比相对较差，但仍可能足以进行合成 RLHF。

> **表 6：** LLM 作为零样本和少样本有用性分类器在 Anthropic HH 数据集的有用部分上的准确率。

| | 零样本准确率 | 单样本准确率 |
|---|---|---|
| GPT-NeoX-20B | 0.55 | 0.57 |
| text-davinci-002 | 0.61 | 0.63 |
| text-davinci-003 | 0.63 | 0.64 |
| GPT-J RM | 0.71 | N/A |

**构建合成对话数据集**  为了构建我们的训练集，我们首先使用来自有用 HH 训练集的样本对 text-davinci-003 进行少样本提示，以生成超过 150k 个合成提示，模拟人类可能向 AI 助手提出的问题。我们以高温度（ $T=1.4$ ）采样以最大化多样性，并通过提示 GPT-NeoX-20B 判断生成的请求是否合理来过滤掉低质量回复。此外，我们通过使用 GPT-J 嵌入每个回复来过滤掉重复的/语义相似的回复。结果是约 60k 个高度多样化的提示的数据集。然后我们提示 text-davinci-003 对这些生成的提示进行回复，并重复该过程以合成生成两轮完整的交互对话，总共约 120k 个样本。见表 7 中的合成提示和不同规模模型的回复示例。

> **表 7：** 合成生成的多轮对话以及不同规模模型回复的示例。

| | 内容 |
|---|---|
| **合成对话** | Human: What is the proper technique for playing the guitar. Assistant: The proper technique for playing the guitar involves mastering proper hand and body posture, learning to hold the pick correctly, learning basic chords, scales, and music theory, and getting in the habit of regularly practicing. Human: I'm feeling a bit overwhelmed–could you break down the steps more? Assistant: |
| **text-davinci-003 回复** | Absolutely. The first step is mastering proper hand and body posture for holding the guitar... |
| **text-davinci-002 回复** | 1. Master proper hand and body posture. 2. Learn to hold the pick correctly. 3. Learn basic chords, scales, and music theory. 4. Practice regularly. |
| **GPT-NeoX-20B 回复** | Absolutely. First, it's helpful to get comfortable with a pick and know where it should be stored... |
| **Pythia 6.9B 回复** | Certainly, I can! Step 1: Familiarize yourself with the parts of the guitar... |
| **Pythia 1.4B 回复** | Absolutely! First, it's important to have a good eye-in-air position... |
| **Pythia 125M 回复** | Practicing is key, when you can make more on your practice day as well as taking the proper stance... |

接下来，我们在合成对话数据集的 20k 大小子集上对 125M 到 20B 的模型进行监督微调。使用的超参数与论文其他地方相同。然后我们在整个数据集上对每个模型进行回复采样，以及 text-davinci-002。然后我们使用 text-davinci-003 来确定对 text-davinci-002 和 text-davinci-003 的回复的偏好。为了避免任何排序偏差，我们在将模型回复呈现给 text-davinci-003 时随机化顺序，如 [25] 中所做。令人惊讶的是，我们发现 text-davinci-003 只有 58% 的时间偏好自身，表明其作为评判者的公正性。这定义了我们的第一个合成偏好数据集。

我们可以使用上述第二种方法构建我们的第二个合成偏好数据集：通过根据模型规模对模型回复进行排序。具体来说，这为我们提供了偏好排序：125M < 1.4B < 6.9B < 20B < text-davinci-002 < text-davinci-003。然后我们可以使用这个合成数据集训练不同规模的奖励模型。这些模型在测试集上的总体准确率作为训练样本数的函数绘制在图 9 中。此外，我们还绘制了 RM 模型规模在预测每类模型规模比较（例如选择 6.9B 回复而不是 125M 回复）中的偏好时的准确率。这些结果报告在图 10 中。

> **图 9：** 按规模排序的合成偏好 RM 的准确率作为训练数据规模的函数。我们看到 20B 在 120,000 样本之前样本效率最高，之后 6B 模型表现略好。

> **图 10：** 按规模排序的合成偏好 RM 在模型规模比较细粒度类别上的准确率（例如选择 6.9B 与 125M 的回复）。我们看到某些比较非常容易，例如 125M 与 text-davinci-003，而其他比较则困难得多。

总体而言，我们发现最佳 RM 模型 6.9B 在正确选择更可取的回复方面表现非常好，准确率超过 90%。然而，尚不清楚我们按规模排序的偏好建模假设能在多大程度上转化为有用的 RM。为了测试这一点，我们在有用 HH 测试集上评估 6.9B RM。结果是相对较低的 0.61 分数。相比之下，我们发现最佳 GPT-J HH RM 在这个合成数据集上令人信服地泛化，得分为 0.78。
