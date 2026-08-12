# OneRec-V2 技术报告

> OneRec 团队
> 快手科技

本文介绍了快手提出的 OneRec-V2，一个工业级生成式推荐框架，通过惰性仅解码器架构和基于真实用户交互的偏好对齐实现端到端推荐。核心内容：

- **惰性仅解码器架构**：消除编码器瓶颈，总计算量减少94%，训练资源减少90%，成功扩展到80亿参数
- **基于真实用户交互的偏好对齐**：时长感知奖励塑造 + 自适应比率裁剪，利用真实用户反馈优化推荐

关键发现：OneRec-V2 在快手/快手极速版上应用停留时间提升0.467%/0.741%，平衡多目标推荐而无跷跷板效应。

---

## 摘要

生成式AI的最新突破通过实现端到端生成，从根本上改变了推荐系统。OneRec是一个工业级生成式推荐框架，它将推荐重新定义为自回归生成任务，从而能够直接优化最终目标并实现高模型FLOPs利用率（MFU）。虽然OneRec-V1在实际部署中已展现出显著的经验成功，但两个关键挑战阻碍了其可扩展性和性能：（1）编码器-解码器架构中的低效计算分配，其中97.66%的资源消耗在序列编码（上下文编码）而非生成上，这限制了模型的可扩展性；（2）仅依赖奖励模型的强化学习的局限性，包括低效的采样以及由于代理奖励信号导致的潜在奖励破解。为解决这些挑战，我们提出了OneRec-V2，其特色包括：

1. **惰性仅解码器架构（Lazy Decoder-Only Architecture）**：一种精简的、仅解码器的设计，消除了编码器瓶颈并简化了交叉注意力，总计算量减少94%，训练资源减少90%。这种效率使得模型成功扩展到80亿参数。
2. **基于真实用户交互的偏好对齐（Preference Alignment with Real-World User Interactions）**：一个用户反馈驱动的框架，包含（i）时长感知奖励塑造以缓解视频时长偏差，以及（ii）自适应比率裁剪以稳定策略优化。

在快手/快手极速版上的大量A/B测试证明了OneRec-V2的有效性，应用停留时间提升了0.467%/0.741%，同时平衡了多目标推荐而无跷跷板效应。

**图1 | 左：不同模型架构从0.1B到8B参数的扩展曲线。右：1B参数下OneRec-V1与OneRec-V2的对比。**

© 2025 快手。版权所有。

## 目录

1. **引言**
2. **惰性仅解码器架构**
   2.1 设计原则
   2. 2 整体架构
   2.3 实验结果
3. **基于真实用户交互的偏好对齐**
   3.1 基于用户反馈信号的强化学习
   3.2 用户反馈信号与奖励模型对比
4. **在线A/B测试**
5. **结论、局限性与未来方向**

**附录**
A. 贡献者
B. 不同架构的计算复杂度
C. 实验结果
D. 禁用缓存时的在线性能

---

## 1. 引言

生成式AI已在众多领域催化了范式转变（Achiam et al., 2023; Guo et al., 2025; Yang et al., 2025a）。虽然传统的级联推荐架构一直在持续演进，但它们仍然受到根本性瓶颈的制约：固有的多阶段设计导致计算资源碎片化和优化目标不对齐。生成式推荐通过将推荐重新定义为端到端的序列生成问题（Badrinath et al., 2025; Chen et al., 2024; Cui et al., 2022; Feng et al., 2022; Han et al., 2025; Kong et al., 2025; Rajput et al., 2023; Yang et al., 2025b; Zhai et al., 2024; Zhou et al., 2025），改变了这一范式。这种统一的方法能够直接优化最终目标，实现高模型FLOPs利用率（MFU），并促进推荐系统与大型基础模型社区之间的更紧密融合。

虽然OneRec-V1（Zhou et al., 2025）在工业部署中已展现出相当的成功，但仍有机会进一步释放其可扩展性和性能：

**（1）编码器-解码器架构中的低效计算分配。** OneRec-V1采用编码器-解码器框架，用户历史交互序列通过编码器处理，然后通过交叉注意力被解码器使用。尽管OneRec-V1的解码器参数多于编码器，但计算负载主要集中在编码器上，因为它处理了广泛的用户交互序列，而解码器的输入则要短得多。如第2.1节所示，在OneRec-V1的上下文长度为512的情况下，上下文编码消耗了总FLOPs的97.66%，而目标（目标item）的解码器生成仅占2.34%。这种不成比例的分配带来了可扩展性挑战，因为大部分计算预算都用于序列编码，而非制定推荐决策的关键生成过程。在相同的计算预算下，这种不平衡的资源分配可能会限制模型有效扩展到更大架构的潜力。

**（2）仅依赖奖励模型的强化学习的局限性。** 尽管OneRec-V1已经证明了基于奖励模型的强化学习在策略优化方面的有效性，但这种方法面临两个内在挑战。首先，采样效率有限，因为依赖奖励模型的方法需要额外的计算资源进行在线生成和评分。这限制了对用户进行采样以近似全局行为的能力。其次，存在潜在的奖励破解问题，即策略学会利用奖励模型中的特定模式或偏差，而这些模式或偏差并不能转化为实际的改进。整合真实用户反馈以解决这些固有问题，可以更好地使策略与用户偏好对齐，并带来改进的结果。此外，OneRec的大规模部署为其在持续反馈循环中通过策略优化实现自我改进提供了关键机会。

在本工作中，我们介绍了OneRec-V2，它通过惰性解码器架构和基于真实用户交互的偏好对齐来解决上述根本性限制。如图2所示，我们的主要贡献是：

1. **惰性仅解码器架构。** 我们提出了一种精简的仅解码器架构，消除了传统编码器-解码器设计的计算瓶颈。通过移除编码器组件并简化交叉注意力机制（消除K/V投影层），我们的惰性解码器在等效计算预算下，实现了94%的计算需求减少和90%的实际训练资源减少，同时支持16倍更大的模型参数（从0.5B扩展到8B）。如图1所示，这种架构不仅使仅解码器Transformer在工业级推荐系统中变得实用和高效，而且在模型大小和FLOPs方面展现出强大的扩展能力，为未来生成式推荐中的模型开发提供了有价值的指导。

**图2 | OneRec-V2的整体架构和训练后对齐框架。左侧面板展示了惰性仅解码器架构，右侧面板描绘了训练后偏好对齐过程。**

2. **基于真实用户交互的偏好对齐。** 我们引入了一个全面的训练后框架，直接利用真实世界的用户反馈信号来解决生成式推荐系统中奖励建模的根本性挑战。（i）**时长感知奖励塑造（Duration-Aware Reward Shaping）**，通过考虑视频时长变化来缓解原始观看时长信号中的固有偏差，确保奖励信号准确反映内容质量而不仅仅是时长；（ii）**自适应比率裁剪（Adaptive Ratio Clipping）**，在策略优化过程中有效降低训练方差，同时保留收敛保证。我们的实验表明，应用停留时间有显著提升。值得注意的是，当结合OneRec自身推荐的流量分布模式时，我们观察到在线性能的增强，这表明模型优化与真实用户行为分布之间的对齐得到了改善。

在拥有4亿日活用户的快手/快手极速版APP上进行的大量在线A/B测试表明，与OneRec-V1相比，OneRec-V2实现了显著改进，应用停留时间分别提升了0.467%和0.741%，同时有效平衡了多个推荐目标，且无跷跷板效应。

在本文的其余部分，我们首先详细阐述OneRec-V2的架构和预训练实验结果（第2节）。接下来，我们介绍训练后方法（第3节），然后是通过在线A/B测试进行的综合评估（第4节）。最后，我们总结了本工作，讨论了现有局限性，并提出了未来研究的潜在方向（第5节）。

---

## 2. 惰性仅解码器架构

在本节中，我们介绍基于惰性解码器的架构。第2.1节详细阐述了OneRec模型架构的演进路径和思路。在第2.2节中，我们介绍了OneRec-V2的惰性仅解码器架构，该架构在显著降低计算复杂度和内存消耗的同时，实现了更低的生成任务损失。最后，在第2.3节中，我们详细介绍了验证惰性仅解码器设计优越性的综合实验结果，并探索了生成式推荐系统的扩展规律。

**图3 | 朴素曝光组织：模式A $\to$ B在多次曝光中被冗余训练。以用户为中心的组织：在时间t3训练用户2的数据时，模型已经从用户1在t4的未来交互中学习了模式B $\to$ C。仅新曝光组织：仅训练最新的曝光。**

### 2.1. 设计原则

自回归模型已成为现代自然语言处理中的主导范式，为GPT（Brown et al., 2020; Radford et al., 2019）和LLaMA（Touvron et al., 2023a,b）等最先进的大型语言模型（LLM）提供动力。它们展现出卓越的可扩展性（Hoffmann et al., 2022; Kaplan et al., 2020），其成功源于优雅的简洁性：一个统一架构以自回归方式处理序列。结合大规模预训练能力（Devlin et al., 2019; Raffel et al., 2020），基于Transformer的自回归模型已成为生成式AI应用的事实标准。

为将这些架构适配到推荐系统，第一步是构建用于自回归训练的文档。传统上，推荐系统的训练样本按时间顺序的曝光（impression）组织。然而，当与标准的下一个词元预测目标结合时，会出现冗余，如图3.a所示。避免冗余的一种方法是使用以用户为中心的组织方式，其中每个训练样本包含一个用户的完整交互历史，如图3.b所示。然而，这带来了潜在的时间数据泄露（Ji et al., 2023）和流行度偏差的风险。已有众多研究（Gangwar and Jain, 2021; Gharahighehi et al., 2021; Huang et al., 2022; Klimashevskaia et al.; Zhu et al., 2021）致力于缓解这些问题。

为解决上述问题，我们提出按时间顺序组织数据，但仅对最新曝光的item应用训练损失，如图3.c所示，其中灰色的item被排除在下一次元预测之外。由于早先的item和最新曝光的item工作方式不同，我们在之前的OneRec-V1（Zhou et al., 2025）中选择了编码器-解码器架构。如表1所示，我们对计算细节进行了初步分析。计算可以分为两类：上下文编码（context encoding）和目标解码（target decoding）。

**定义1. 上下文编码（Context Encoding）**

处理和转换用户上下文特征的计算操作，具体包括：（i）编码器中执行的上下文变换操作，以及（ii）解码器中交叉注意力的上下文投影操作。

**定义2. 目标解码（Target Decoding）**

在解码器中处理和转换目标item的语义词元的计算操作，具体包括：（i）捕获语义词元间依赖关系的自注意力，（ii）应用非线性变换的前馈网络（FFN），以及（iii）交叉注意力中的查询和输出变换。

**表1 | 与损失相关的目标解码计算比例，针对1B参数模型计算。此处上下文表示不直接参与损失计算的用户特征词元。**

| 上下文长度 N | 512 | 3000 |
|---|---|---|
| **编码器-解码器 (0.5B:0.5B)** | | |
| 总计算量 (GFLOPs) | 346 | 1988 |
| 上下文编码 (GFLOPs) | 338 | 1980 |
| 目标解码 (GFLOPs) | 8.1 | 8.1 |
| 目标比例 | 2.34% | 0.41% |
| **朴素仅解码器 (1B)** | | |
| 总计算量 (GFLOPs) | 632 | 3618 |
| 上下文编码 (GFLOPs) | 614 | 3600 |
| 目标解码 (GFLOPs) | 18 | 18 |
| 目标比例 | 2.85% | 0.49% |
| **惰性仅解码器 (1B)** | | |
| 总计算量 (GFLOPs) | 18 | 18 |
| 目标比例 | $\approx$ 100% | $\approx$ 100% |

根据表1，在参数数量相同的情况下，编码器-解码器相比经典的仅解码器架构节省了近一半的计算量。然而，两种架构仍然存在计算效率低下的问题：大部分计算都分配给了不直接贡献于损失计算的词元。对于典型的上下文长度N=512（OneRec-V1），只有不到3%的总FLOPs贡献于损失计算，并且随着上下文长度的增加，这一比例变得越来越微不足道。详细的计算分析见附录B。为了将计算集中在目标item的语义词元上，从而有效扩展到更大的模型，我们提出了惰性仅解码器架构。

### 2.2. 整体架构

在本节中，我们介绍我们的新型架构，如图4所示。该架构通过两项关键创新从根本上重新构想了生成式推荐器的设计。

**图4 | 所提出的惰性仅解码器生成式推荐器的架构。上下文处理器（Context Processor）将异构的用户特征路径转换为统一的上下文表示，然后经过归一化，产生跨层共享的键值对用于交叉注意力。惰性解码器通过堆叠的Transformer块处理BOS词元和目标item的词元化语义ID。每个块包含：（1）无键值投影的惰性交叉注意力，支持分组查询注意力（GQA）；（2）因果自注意力；以及（3）前馈网络。最终表示被投影以预测下一个item推荐的语义ID。**

首先，我们提出了一种惰性仅解码器架构，它不同于传统的编码器-解码器模型和朴素的仅解码器方法。我们的设计将上下文视为仅通过交叉注意力访问的静态条件信息，消除冗余计算，同时保留模型捕获复杂用户-item交互的能力。

其次，我们引入了一种极其高效的惰性交叉注意力机制，无需键值投影。结合分组查询注意力（GQA）（Ainslie et al., 2023），这一设计显著减少了内存占用，从而能够高效处理广泛的用户历史。

#### 2.2.1. 上下文处理器（Context Processor）

为有效整合异构和多模态的用户行为信号，我们设计了一个统一模块，称为上下文处理器，使其能够与下游基于注意力的解码器块无缝集成。

具体而言，用户画像和行为等异构输入被拼接成一个统一的序列，即上下文。上下文中的每个item都被处理为相同的维度：

$$
d_{\text{context}} = S_{\text{kv}} \cdot L_{\text{kv}} \cdot G_{\text{kv}} \cdot d_{\text{head}}
$$
  (1)

其中， $d_{\text{head}}$ 表示注意力头维度， $G_{\text{kv}}$ 表示键值头组数， $S_{\text{kv}}$ 表示键值分裂系数， $L_{\text{kv}}$ 表示键值层层数。

上下文表示被转换为注意力机制的逐层键值对。我们沿着特征维度对上下文张量进行分割，以生成 $L_{\text{kv}}$ 组键值对：

$$
\text{Context} = [C_0, C_1, \ldots, C_{S_{\text{kv}} \cdot L_{\text{kv}} - 1}]
$$
  (2)

其中 $C_{S_{\text{kv}} \cdot L_{\text{kv}} - 1} \in \mathbb{R}^{G_{\text{kv}} \cdot d_{\text{head}}}$ 。这里为简单起见忽略了序列维度。

对于每一层 $l \in \{0, 1, \ldots, L_{\text{kv}} - 1\}$ ，我们计算归一化的键值对：

$$
k_l = \text{RMSNorm}_{k,l}(C_{l \cdot S_{\text{kv}}})
$$
  (3)

$$
v_l =
\begin{cases}
\text{RMSNorm}_{v,l}($C_{l \cdot S_{\text{kv}$}+1}), & \text{若 } $S_{\text{kv}$} = 2 \text{（分离键值）}\\
k_l, & \text{若 } $S_{\text{kv}$} = 1 \text{（共享表示）}
\end{cases}
$$
  (4)

上下文处理器的最终输出为 $\{(k_0, v_0), \ldots, (k_{L_{\text{kv}}-1}, v_{L_{\text{kv}}-1})\}$ 。

#### 2.2.2. 惰性解码器块（Lazy Decoder Block）

**词元化器（Tokenizer）**：对于每个目标item，我们使用一个语义词元化器，生成3个捕获item多方面特征的语义ID，与OneRec-V1（Zhou et al., 2025）相同。在训练期间，我们使用前2个ID，并在前面添加一个序列开始（BOS）词元以形成输入序列。这些词元索引然后通过嵌入表映射以获得初始隐藏表示：

$$
h^{(0)} = \text{Embed}([\text{BOS}, s_1, s_2]) \in \mathbb{R}^{3 \times d_{\text{model}}}
$$
  (5)

**块结构（Block Structure）**：惰性解码器由 $N_{\text{layer}}$ 个堆叠的Transformer块组成，每个块包含三个主要组件：交叉注意力、自注意力和前馈模块。对于第 $l$ 层，变换定义为：

$$
h_{\text{cross}}^{(l)} = h^{(l-1)} + \text{CrossAttn}\left(\text{RMSNorm}(h^{(l-1)}), k_{l_{\text{kv}}}, v_{l_{\text{kv}}}\right)
$$
  (6)

$$
h_{\text{self}}^{(l)} = h_{\text{cross}}^{(l)} + \text{SelfAttn}\left(\text{RMSNorm}(h_{\text{cross}}^{(l)})\right)
$$
  (7)

$$
h^{(l)} = h_{\text{self}}^{(l)} + \text{FFN}^{(l)}\left(\text{RMSNorm}(h_{\text{self}}^{(l)})\right)
$$
  (8)

其中 $\text{RMSNorm}$ 表示均方根层归一化，用于训练稳定性。

为在保持计算效率的同时增强模型容量，我们采用了一种混合架构，其中较深层的稠密前馈网络被替换为混合专家（MoE）模块。遵循DeepSeek-V3（Liu et al., 2024），我们采用了一种无辅助损失的负载均衡策略，以确保高效的专业化利用。

**惰性交叉注意力：KV共享**：为提高参数和计算效率，多个惰性解码器块共享来自上下文处理器生成的同一组键值对。对于当前层 $l$ ，我们确定对应的键值索引：

$$
l_{\text{kv}} = \left\lfloor \frac{l \cdot L_{\text{kv}}}{N_{\text{layer}}} \right\rfloor
$$
  (9)

其中 $N_{\text{layer}}$ 是惰性解码器块的总数。这一设计确保了每连续 $\frac{N_{\text{layer}}}{L_{\text{kv}}}$ 个块共享相同的上下文表示 $(k_{l_{\text{kv}}}, v_{l_{\text{kv}}})$ ，其中 $k_{l_{\text{kv}}}, v_{l_{\text{kv}}} \in \mathbb{R}^{(N_s + T_{\text{short}} + T_{\text{long}}) \times G_{\text{kv}} \times d_{\text{head}}}$ 。

我们通过采用统一的键值表示进一步提高参数效率，其中所有层 $v_l = k_l$ ，利用了绑定的键值投影可以在保持相当性能的同时减少模型内存占用这一观察。

**惰性交叉注意力：分组查询注意力**：查询投影保持 $H_q = d_{\text{model}} / d_{\text{head}}$ 个注意力头，而键值对仅使用 $G_{\text{kv}}$ 个头组，其中通常 $G_{\text{kv}} < H_q$ 。这一设计显著减少了上下文表示的内存占用和注意力计算期间的内存访问需求，从而能够有效扩展到更长的上下文和更大的批量大小。

**输出层（Output Layer）**：来自最后一个解码器块的最终隐藏表示经过位置特定的RMSNorm和线性层，以为每个语义ID生成预测。在训练期间，我们优化模型以最大化目标item的语义ID $[s_1, s_2, s_3]$ 的似然。

### 2.3. 实验结果

为验证惰性仅解码器架构的有效性，我们从多个维度进行了全面的实验评估。我们系统地将我们的方法与经典架构进行了比较，研究了关键架构创新的影响，并探索了稠密和稀疏模型变体的扩展特性。所有实验均使用来自快手的2025年8月10日至14日的曝光数据，以相同的采样比例和一致的全局批量大小进行流式训练。除非另有说明，我们设置 $L_{\text{kv}} = 1$ , $S_{\text{kv}} = 1$ , $d_{\text{head}} = d_{\text{model}} / N_{\text{head}}$ , $G_{\text{kv}} = N_{\text{head}}$ 且 $(N_s + T_{\text{short}} + T_{\text{long}}) \approx 512$ 。对于在线部署，我们使用1B参数模型，并将长期用户行为序列长度扩展到 $(N_s + T_{\text{short}} + T_{\text{long}}) \approx 3000$ 。

#### 2.3.1. 架构对比

我们比较了生成式推荐的三种架构范式：编码器-解码器架构（OneRec-V1）、朴素仅解码器架构以及我们提出的惰性仅解码器架构。对于每个模型，我们评估三个语义词元的平均生成损失：

$$
L_{\text{Gen}} = -\frac{1}{3} \sum_{i=1}^{3} \log p(s_i | \text{BOS}, s_{<i}, \text{Context})
$$
  (10)

其中 $s_i$ 表示目标item的第 $i$ 个语义ID，BOS表示句子开始词元，context是上下文处理器的输出，包括用户静态和行为特征。该损失与OneRec-V1不同，因为我们使用三个词元的平均值，而V1使用它们的和。

**表2 | 不同模型规模下各架构的对比。朴素仅解码器在0.5B和1B规模下的实验因计算资源限制未能进行。激活值数量基于批量大小512计算。**

| 架构 | 总参数量¹ | GFLOPs | 激活值 | 收敛损失 |
|---|---|---|---|---|
| **0.1B** | | | | |
| Enc:Dec=1:1 | 0.1B | 25.64 | 4.21B | 3.59 |
| Enc:Dec=1:2 | 0.1B | 17.72 | 2.92B | 3.55 |
| 朴素 Dec-Only | 0.1B | 63.78 | 7.52B | 3.54 |
| 惰性 Dec-Only | 0.1B | 1.98 | 0.31B | 3.57 |
| **0.5B** | | | | |
| Enc:Dec=1:1 | 0.5B | 142.73 | 10.79B | 3.35 |
| Enc:Dec=1:2 | 0.5B | 104.73 | 7.94B | 3.32 |
| 朴素 Dec-Only | 0.5B | 317.68 | 19.28B | * |
| 惰性 Dec-Only | 0.5B | 9.55 | 0.77B | 3.33 |
| **1B** | | | | |
| Enc:Dec=1:1 | 1B | 296.36 | 17.63B | 3.28 |
| Enc:Dec=1:2 | 1B | 204.21 | 12.20B | 3.26 |
| 朴素 Dec-Only | 1B | 634.83 | 31.53B | * |
| 惰性 Dec-Only | 1B | 18.89 | 1.24B | 3.27 |

注意：此表中的FLOPs和激活值基于特定模型配置计算，比表1中呈现的近似估计更为精确。

**图5 | 三种模型规模下不同架构的训练曲线。尽管实现了相似的损失，惰性仅解码器架构所需的FLOPs比经典架构少10倍。E1D1和E1D2分别表示编码器-解码器参数比为1:1和1:2。**

表2和图5展示了不同模型规模下的计算需求和收敛性能。尽管需要的FLOPs和激活内存显著更少，但我们的惰性仅解码器架构实现了与传统方法相当的损失。

#### 2.3.2. 键值共享

我们的上下文处理器引入了两个关键参数，可以灵活控制总体上下文维度： $L_{\text{kv}}$ 和 $S_{\text{kv}}$ 。参数 $L_{\text{kv}}$ 决定了跨层的不同上下文表示的数量，每 $N_{\text{layer}} / L_{\text{kv}}$ 个连续的解码器块共享相同的键值对。参数 $S_{\text{kv}}$ 进一步控制键和值是否共享相同的表示（ $S_{\text{kv}} = 1$ ）或保持独立的投影（ $S_{\text{kv}} = 2$ ）。这种设计在保持生成任务上相当性能的同时，降低了计算成本和激活内存。我们使用 $N_{\text{layer}} = 18$ 的1B参数稠密惰性解码器模型进行消融研究，以探究这些设计选择的影响。

**表3 | 键值共享策略对模型效率和性能的影响。激活值数量基于批量大小512计算。**

| Lkv | Skv | GFLOPs | 激活值 | 收敛损失 |
|---|---|---|---|---|
| 1 | 1 | 18.89 | 1.24B | 3.27 |
| 1 | 2 | 19.19 | 1.33B | 3.27 |
| 3 | 1 | 19.49 | 1.42B | 3.27 |
| 9 | 1 | 21.27 | 1.99B | 3.27 |
| 18 | 1 | 23.95 | 2.83B | 3.27 |

图10a表明，激进的键值共享在整个训练过程中保持了有竞争力的损失，验证了我们高效的上下文处理策略。

#### 2.3.3. 分组查询注意力

分组查询注意力（GQA）在多个查询头之间共享键值头。在我们的惰性解码器架构中，这种优化减少了交叉注意力操作中的激活内存和内存访问瓶颈，从而在几乎不影响模型质量的情况下提高了训练吞吐量。我们研究了不同数量的键值头组 $G_{\text{kv}} \in \{1, 2, 7\}$ 对具有14个注意力头的1B参数稠密惰性解码器模型的影响。

**表4 | 分组查询注意力对模型效率和性能的影响。激活值数量和交叉注意力中的键值大小基于批量大小512计算。**

| Gkv | GFLOPs | 激活值 | KV大小 | 收敛损失 |
|---|---|---|---|---|
| 14 | 18.89 | 1.24B | 94M | 3.27 |
| 7 | 18.74 | 1.19B | 47M | 3.28 |
| 2 | 18.64 | 1.16B | 13M | 3.28 |
| 1 | 18.62 | 1.15B | 7M | 3.27 |

表4和图10b中的结果表明，具有不同组数的GQA在显著减少内存需求的同时，实现了与完全注意力几乎相同的性能。

#### 2.3.4. 模型扩展

我们对惰性仅解码器架构进行了全面的扩展实验，研究了稠密和稀疏配置，以了解不同模型规模下的计算-性能权衡。

**稠密模型扩展。** 我们探索了从0.1B到8B参数的稠密惰性解码器模型的扩展特性。表5展示了每个模型配置的架构超参数和收敛性能。

**表5 | 模型扩展实验的超参数配置和收敛损失。**

| 模型 | 参数量 | d_model | n_layers | n_heads | embed_dim | 学习率 | 收敛损失 |
|---|---|---|---|---|---|---|---|
| **稠密** | | | | | | | |
| | 0.1B | 640 | 12 | 10 | 32 | 5.00e-4 | 3.57 |
| | 0.2B | 896 | 12 | 14 | 45 | 3.54e-4 | 3.46 |
| | 0.5B | 1408 | 14 | 11 | 70 | 2.24e-4 | 3.33 |
| | 1B | 1792 | 18 | 14 | 90 | 1.58e-4 | 3.27 |
| | 2B | 2304 | 22 | 18 | 115 | 1.12e-4 | 3.23 |
| | 4B | 2944 | 26 | 23 | 147 | 7.91e-5 | 3.20 |
| | 8B | 3584 | 34 | 28 | 179 | 5.59e-5 | 3.19 |
| **MoE** | | | | | | | |
| | 4B (0.5B激活) | 1408 | 14 | 11 | 70 | 2.24e-4 | 3.22 |

**稀疏混合专家（MoE）。** 为实现更高效的扩展，我们研究了一种混合专家（MoE）变体，将稠密前馈网络替换为稀疏专家路由。我们的MoE配置采用53个路由专家和1个共享专家，总参数为4B（每个词元激活0.5B）。该模型对每个词元使用top-3专家路由，MoE中间大小为1408。稀疏模型保持与0.5B稠密模型相同的基础架构，同时将前2个惰性解码器块之后的前馈层替换为MoE层。

**结果与分析。** 图6展示了不同模型配置下的训练动态。我们的实验揭示了惰性解码器架构在推荐系统中扩展行为的几个关键见解。我们还展示了不同规模模型随着训练预算增加而损失下降的情况，见图11。

**图6 | 不同模型规模下惰性解码器架构的训练动态。收敛损失从3.57（0.1B）下降到3.19（8B）。4B MoE变体（激活0.5B），图中表示为4BA0.5B，在保持计算效率的同时实现了有竞争力的性能。**

收敛损失随着模型规模的增大而持续改善，从0.1B模型的3.57下降到8B模型的3.19。最显著的改进发生在十亿以下参数范围内，从0.1B扩展到1B参数时损失减少了0.3。边际收益逐渐减小，4B模型相比2B模型仅进一步降低0.03，这表明超过2B参数的扩展仍然具有挑战性。

总参数量4B（激活0.5B）的MoE变体实现了3.22的收敛损失，优于2B稠密模型，同时保持了与0.5B稠密基线相当的计算需求。相比0.5B稠密模型，该配置实现了0.11的损失减少，证明了稀疏架构在推荐任务中的有效性。

这些结果表明，我们的惰性解码器架构能够有效扩展，其中MoE变体为工业级推荐系统的部署提供了特别有吸引力的权衡，因为在这些系统中计算效率直接影响服务成本和延迟。

---

## 3. 基于真实用户交互的偏好对齐

在本节中，我们介绍OneRec-V2的训练后阶段。监督微调（SFT）阶段与OneRec-V1相同，使用流式曝光数据执行在线 $L_{\text{Gen}}$ 损失训练，与预训练期间使用的损失一致。主要目的是捕获用户的实时兴趣变化，同时防止模型偏离预训练模型太远。在OneRec-V1中，强化学习阶段完全基于奖励模型。在OneRec-V2中，我们引入了基于用户反馈信号作为奖励的强化学习。

### 3.1. 基于用户反馈信号的强化学习

基于用户反馈定义奖励可以避免奖励破解的问题，并且不需要额外的模型计算开销。然而，它仍然面临诸如如何结合多个目标以及正标签稀疏性等挑战。在短视频推荐场景中，每个视频的播放时间是最密集的反馈信号，并且与最重要的在线指标（如APP停留时间和LT7（7日留存））密切相关。因此，我们设计了一个简单但有效的基于播放时间的奖励。

#### 3.1.1. 时长感知奖励塑造

**图7 | 时长感知奖励塑造示意图。用户观看历史中的视频根据时长进行分桶，对于目标视频，其播放时间在相应桶内的分位数被计算为用户偏好分数。**

虽然视频播放时间是用户满意度的有用指标，但它天然受到视频时长的影响。为解决这一偏差，我们提出了一种时长感知奖励塑造机制，如图7所示。该方法通过将播放时间与每个用户历史中具有可比时长的视频进行比较来进行归一化。由于视频时长遵循长尾分布，我们使用对数策略将历史视频划分为桶。该方法将时长分成指数级变宽的区间，从而产生更平衡、更有意义的比较组。映射由函数 $F(d)$ 给出，该函数将时长为 $d$ 的视频分配给一个离散的桶索引 $b \in \mathcal{B}$ 。形式上，分桶函数定义为：

$$
F(d) = \lfloor \log_{\beta}(d + \epsilon) \rfloor
$$
  (11)

其中 $\beta$ 是一个可配置的对数底数，控制桶的粒度； $\epsilon$ 是一个小常数（例如 $10^{-6}$ ），用于在处理极短时长时保证数值稳定性。

令 $H_u = \{(d_k, p_k)\}_{k=1}^{N}$ 表示用户 $u$ 的历史交互序列，其中 $d_k$ 是视频时长， $p_k$ 是观察到的播放时间。对于每个时长桶 $b$ ，我们将播放时间的经验分布定义为：

$$
P_{u,b} = \{p_j | (d_j, p_j) \in H_u, F(d_j) = b\}
$$
  (12)

给定目标视频 $i$ ，其时长为 $d_i$ ，播放时间为 $p_i$ ，我们首先确定其桶 $b = F(d_i)$ 。然后，时长归一化的参与度得分计算为 $p_i$ 在用户历史分布 $P_{u,b}$ 中的经验百分位排名：

$$
q_i = \frac{|\{p_j \in P_{u,b} | p_j \leq p_i\}|}{|P_{u,b}|}
$$
  (13)

我们基于此得分选择最有价值的样本作为正样本。在一个批次中，我们将 $q_i$ 按降序排序后，计算 $\tau_b$ 作为 $q_i$ 的第25百分位数（上四分位数）。对于具有明确负面反馈（如"不喜欢"动作， $neg_i = 1$ ）的样本，我们设置 $A_i = -1$ 。所有其他样本被过滤掉，等价于设置 $A_i = 0$ 。注意我们直接分配优势值而不进行归一化，因为我们对正样本和负样本的定义足够严格。进一步归一化可能会在优化中引入不一致性，从而降低性能。形式上，定义如下：

$$
A_i =
\begin{cases}
+1, & q_i > \tau_B \text{ 且 } neg_i = 0 \\
-1, & neg_i = 1 \\
0, & \text{otherwise}
\end{cases}
$$
  (14)

这种策略有效地筛选了高质量的正例，同时纳入了直接的负信号，从而产生更准确的用户偏好信号。

#### 3.1.2. 强化学习

**梯度有界策略优化（Gradient-Bounded Policy Optimization）** 近年来，强化学习的有效性和稳定性一直是LLM社区的主要研究焦点。一个关键挑战是在保持梯度稳定性的同时，增强探索以提升性能。在本节中，我们介绍我们新提出的强化学习方法GBPO（梯度有界策略优化）。

$$
J_{\text{GBPO}}(\theta) = -\mathbb{E}_{u \sim P(U), \{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}} \left[ \frac{1}{G} \sum_{i=1}^G \frac{\pi_{\theta}(o_i|u)}{\pi'_{\theta_{\text{old}}}(o_i|u)} \cdot A_i \right]
$$
  (15)

$$
\pi'_{\thet$a_{\text{old}$}}(o_i|u) =
\begin{cases}
\max(\p$i_{\theta_{\text{old}$}}, \text{sg}(\p$i_{\theta}$)), & A_i \geq 0 \\
\max(\p$i_{\theta_{\text{old}$}}, 1 - \text{sg}(\p$i_{\theta}$)), & A_i < 0
\end{cases}
$$
  (16)

从公式中可以看出，GBPO移除了对比率的裁剪操作，并引入了对 $\pi_{\theta_{\text{old}}}$ 的动态边界。总体而言，GBPO有两个主要优势：

- **全样本利用**：保留所有样本的梯度，鼓励模型进行更多样化的探索。
- **有界梯度稳定化**：用BCE（二元交叉熵）损失的梯度来界定RL的梯度，增强RL训练的稳定性。

**现有的基于裁剪的工作** 在详细阐述GBPO之前，我们首先简要回顾一下现有的LLM强化学习方法。GRPO/PPO（Schulman et al., 2017; Shao et al., 2024）通过裁剪操作丢弃策略比率过大或过小的样本，以防止过于激进的训练。DAPO（Yu et al., 2025）通过clip higher放松了样本限制，特别是纳入了更多低概率或高熵的词元，从而在提升强化学习性能的同时增加了多样性。这些研究表明，放宽裁剪约束以纳入更多样本可以鼓励更多样化的探索并提升性能。

然而，这些方法并未提供对梯度稳定性的完整和全面考虑。特别是对于负样本，策略比率缺少上限很容易导致梯度爆炸，造成模型性能崩溃。Dual-clip对负样本的策略比率应用了上界截断。虽然这提高了稳定性，但它丢弃了过多的负样本，从而减缓了收敛速度。在OneRec-V1（Zhou et al., 2025）中，我们提出了早期裁剪GRPO（ECPO），它截断了负样本的梯度上界，从而在增强稳定性的同时保留了更多样本。

$$
J_{\text{ECPO}}(\theta) = -\mathbb{E}_{u \sim P(U), \{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}} \left[ \frac{1}{G} \sum_{i=1}^G \min\left( \frac{\pi_{\theta}(o_i|u)}{\pi'_{\theta_{\text{old}}}(o_i|u)} A_i, \text{clip}\left( \frac{\pi_{\theta}(o_i|u)}{\pi'_{\theta_{\text{old}}}(o_i|u)}, 1-\epsilon, 1+\epsilon \right) A_i \right) \right]
$$
  (17)

$$
\pi'_{\theta_{\text{old}}}(o_i|u) = \max\left( \frac{\text{sg}(\pi_{\theta}(o_i|u))}{1+\epsilon+\delta}, \pi_{\theta_{\text{old}}}(o_i|u) \right), \quad \delta > 0
$$
  (18)

**梯度分析** 曝光样本既包括OneRec生成的样本，也包括来自传统管道的样本。对于OneRec生成的曝光样本，我们使用曝光时的生成概率作为 $\pi_{\text{old}}$ 。对于来自传统管道的样本，由于管道的复杂性，我们无法获得其概率；因此，我们将 $\pi_{\text{old}}$ 简化为OneRec模型当前的生成概率，即 $\pi_{\text{old}} = \text{sg}(\pi_{\theta})$ 。对于这些样本，策略比率始终为1。在传统的RL方法中，比率为1的样本被认为对训练是稳定的，不会进行截断。然而，在现实中，此类样本仍可能导致梯度爆炸，这是由负样本引起的，如图8所示。

**图8 | GBPO与传统比率裁剪方法的梯度对比。在负样本训练中，GBPO展现出显著更稳定的梯度。**

从梯度的角度来看，对于这些样本的特定词元 $i$ ，我们有：

$$
J_{\text{ECPO}}^i(\theta) = -A_i \cdot \frac{\pi_{\theta}}{\text{sg}(\pi_{\theta})}
$$
  (19)

$$
\frac{\partial J_{\text{ECPO}}^i(\theta)}{\partial \theta} = -A_i \cdot \frac{1}{\pi_{\theta}} \frac{\partial \pi_{\theta}}{\partial \theta}
$$
  (20)

这表明当前词元概率 $\pi_{\theta}$ 越小，梯度越大。对于正样本，概率越小意味着有更多的提升空间，因此梯度较大是合理的。然而，对于负样本，概率越小意味着抑制它的空间越小；如果梯度太大，很容易导致模型过拟合甚至崩溃。这一现象表明，传统的裁剪方法无法完全解决RL梯度不稳定的问题，因为它们无法避免比率为1时的梯度爆炸。在BCE损失中，对负样本同样有惩罚，但相比于RL损失，其梯度要稳定得多。

$$
\mathcal{L}_{\text{BCE}}(y, p_{\theta}) = -[y \cdot \log(p_{\theta}) + (1-y) \cdot \log(1-p_{\theta})]
$$
  (21)

$$
\frac{\partial \mathcal{L}_{\text{BCE}}}{\partial \theta} =
\begin{cases}
\frac{1}{$p_{\theta}$} \frac{\partial $p_{\theta}$}{\partial \theta}, & y = 1 \\
-\frac{1}{1-$p_{\theta}$} \frac{\partial $p_{\theta}$}{\partial \theta}, & y = 0
\end{cases}
$$
  (22)

对于负样本，当前模型概率越小，抑制时的梯度越小，从而使模型更加稳定。基于这一观察，我们提出了GBPO，它用来自BCE损失的更稳定的梯度来界定RL梯度。我们在图9中展示了差异。

**图9 | GBPO示意图。x轴为 $\pi_{\theta}/\pi_{\theta_{\text{old}}}$ ，y轴为裁剪后的 $\pi_{\theta}/\pi_{\theta_{\text{old}}}$ 。"//"表示"无梯度"。与传统比率裁剪方法相比，GBPO的主要区别在于：1. 它不丢弃任何样本的梯度。2. 对于负样本，比率的界定基于与 $\pi_{\theta}$ 相关的动态边界。**

#### 3.1.3. 实验

**实验设置** 在本节中，我们通过实验验证所定义的用户反馈信号的有效性。为快速验证，本节所有实验均在0.5B模型、上下文长度512的设置下进行。基线为OneRec-V1。在OneRec-V1的实验设置中，分配给实验组的在线流量仅占总流量的很小一部分，因此训练样本几乎完全来自传统推荐管道。在LLM领域，已有研究表明在自生成样本上训练可以实现自我改进（He et al., 2025）。随着OneRec现在服务于总流量的25%，我们有足够的数据来验证这一假设。因此，我们设计了两个实验组进行比较：

- **w/o OneRec样本**：仅使用传统推荐管道生成的样本进行强化学习，这与OneRec-V1的样本一致。
- **w/ OneRec样本**：纳入OneRec管道生成的样本，其中也包括当前模型实验组生成的样本。换句话说，此设置引入了在策略（on-policy）强化学习。

如前所述，用于强化学习的正样本被识别为按时长感知奖励得分排序的前25%分位数的视频，而负样本通过明确的负面反馈（例如"不喜欢"动作）来识别。注意，两组的训练样本总数基本保持相同。强化学习损失为GBPO（公式11）。所有结果见表6。

**结果分析** 从表6中，我们有以下观察结果。当仅使用传统管道的样本时（即与OneRec-V1相同的样本来源），引入基于用户反馈的强化学习显著改善了时长相关指标，如应用停留时间和观看时长，但替代了其他一些指标，如视频观看量。这表明我们的时长感知奖励确实与应用停留时间高度相关。在纳入OneRec管道的样本后，几乎所有指标都显著改善，其中视频观看量尤其由负转正。这表明基于用户反馈的强化学习实现了自我迭代优化，充分利用了用户反馈信号以增强用户体验。

**表6 | 基于用户反馈信号的强化学习的在线A/B测试结果。所有指标均显示相对于OneRec-V1基线的相对改进。**

| 场景 | 在线指标 | w/o OneRec样本 | w/ OneRec样本 |
|---|---|---|---|
| **快手** | | | |
| | 应用停留时间 | +0.165% | +0.227% |
| | 观看时长 | +1.054% | +0.648% |
| | 视频观看量 | -0.901% | +0.716% |
| | 点赞 | -0.186% | +2.897% |
| | 关注 | +2.274% | +3.661% |
| | 评论 | -4.982% | +6.392% |
| | 收藏 | -0.817% | +1.232% |
| | 转发 | -2.162% | +3.426% |
| **快手极速版** | | | |
| | 应用停留时间 | +0.159% | +0.353% |
| | 观看时长 | +0.396% | +0.104% |
| | 视频观看量 | -2.231% | +0.575% |
| | 点赞 | -0.534% | +4.956% |
| | 关注 | +1.809% | +4.800% |
| | 评论 | -4.860% | +5.067% |
| | 收藏 | -0.377% | +2.701% |
| | 转发 | +0.775% | +5.783% |

### 3.2. 用户反馈信号与奖励模型对比

#### 3.2.1. 奖励模型的局限性

在本节中，我们比较OneRec-V1中依赖奖励模型的强化学习与由用户反馈信号驱动的强化学习。尽管OneRec-V1通过大量实验证明了强化学习的有效性，但其性能受到有限采样概率的制约。由于资源限制，在策略（on-policy）roll-out只能对一小部分用户（1%）进行。此外，奖励模型容易受到奖励破解的影响。用户反馈信号直接反映真实的用户偏好，从而降低了奖励破解的风险。然而，在OneRec全面部署之前，无法获得生成样本上的大规模真实用户反馈。随着OneRec的全面部署，现在可以更有效地利用这些信号进行精确的自我迭代优化。在上一节中，我们证明了所提出的时长感知反馈信号的有效性。现在，我们比较用户反馈与奖励模型的性能。

**表7 | OneRec-V2强化学习训练的在线A/B测试结果。**

| 场景 | 在线指标 | 奖励模型 | 用户反馈信号 | 混合 |
|---|---|---|---|---|
| **快手** | | | | |
| | 应用停留时间 | +0.269% | +0.299% | +0.283% |
| | 观看时长 | +0.537% | +0.610% | +0.118% |
| | 视频观看量 | +0.505% | +0.647% | +0.647% |
| | 点赞 | +6.552% | +2.435% | +7.010% |
| | 关注 | +7.265% | +2.007% | +8.458% |
| | 评论 | +15.472% | +0.944% | +8.763% |
| | 收藏 | +1.856% | +1.401% | +9.739% |
| | 转发 | +12.024% | +0.803% | +5.270% |
| **快手极速版** | | | | |
| | 应用停留时间 | +0.163% | +0.213% | +0.207% |
| | 观看时长 | +0.503% | +0.172% | -0.398% |
| | 视频观看量 | +0.457% | +0.056% | +0.083% |
| | 点赞 | +7.798% | +4.008% | +6.267% |
| | 关注 | +12.242% | +4.421% | +11.705% |
| | 评论 | +11.284% | +3.958% | +7.002% |
| | 收藏 | +4.468% | +1.731% | +3.495% |
| | 转发 | +15.919% | +7.704% | +6.670% |

#### 3.2.2. 实验

**实验设置** 我们设置了三组实验进行比较，分别称为奖励模型、用户反馈信号和混合。模型设置与第3.1.3节相同。评估指标与之前的实验相同，包括基于时长的指标和基于交互的指标。应用停留时间是最重要的指标，而其他指标作为用户体验的参考值。表7中显示的结果表示每组相对于OneRec-V1的相对性能。

- **奖励模型**：引入基于奖励模型的强化学习，与OneRec-V1的主要区别在于预训练生成模型的架构。OneRec-V1使用编码器-解码器架构，而OneRec-V2采用所提出的惰性解码器。
- **用户反馈信号**：引入基于用户反馈的强化学习并纳入自生成样本，与上一节中的"w/ OneRec样本"设置相同。
- **混合**：同时引入奖励模型和用户反馈信号，两种样本独立：前者是通过模型自身roll-out采样获得的样本，而后者是之前已曝光给用户的样本。

**结果分析** 从表7中，我们可以总结出以下观察结果。

- 在奖励模型设置中，OneRec-V2的表现显著优于OneRec-V1，进一步证实了惰性解码器架构带来的优势。
- 无论是基于奖励模型还是用户反馈，强化学习在时长指标和交互指标上都带来了双重增益。然而，奖励模型倾向于偏好交互指标的改进，而真实用户反馈倾向于偏好应用停留时间的提升。这是因为奖励模型输出的奖励是多个推荐目标的融合，而我们基于用户反馈定义的奖励主要是从视频播放时间计算得出的。这也表明不同的奖励定义导致了模型的不同偏好，这与OneRec-V1中的结论一致。
- 当两者结合（混合）时，虽然在时长和交互方面的具体增益不如单个策略各自那么高，但性能损失很小，并且应用停留时间和交互指标之间的平衡得到了改善。这是因为两个单独策略带来的增益存在部分重叠。尽管将它们结合起来不能实现完美的叠加效果，但可以让它们相互补充。这也凸显了多样化奖励信号的重要性。我们将在未来对奖励信号的多样性和准确性进行进一步研究。

---

## 4. 在线A/B测试

我们将OneRec-V2部署在快手两个主要的短视频场景中：快手主站feed流和快手极速版feed流，这些是平台最高流量的环境，服务4亿日活用户。评估使用5%流量的实验组进行为期一周的观察。使用的模型为1B参数版本，上下文长度3000，束搜索大小为512。对于在线推理，系统使用了L20 GPU，实现了36ms的延迟和62%的MFU（模型FLOPs利用率）。为降低系统复杂度，此版本仅纳入了用户反馈信号。我们的主要评估指标是应用停留时间（衡量用户总参与时长）和LT7（7日用户留存）。如表8所示，OneRec-V2在两个平台上均实现了显著改进。此外，OneRec-V2在所有用户交互指标上均表现出显著提升，包括点赞、关注、评论和其他参与行为，展示了其引导多任务推荐系统达到更均衡状态的能力，同时有效缓解了竞争目标之间的跷跷板效应。

**表8 | 在线A/B测试中OneRec-V2相对于OneRec-V1的相对改进。**

| 场景 | 在线指标 | OneRec-V2 |
|---|---|---|
| **快手** | | |
| | 应用停留时间 | +0.467% |
| | LT7 | +0.069% |
| | 观看时长 | +1.367% |
| | 视频观看量 | +0.331% |
| | 点赞 | +3.924% |
| | 关注 | +4.730% |
| | 评论 | +5.394% |
| | 收藏 | +2.112% |
| | 转发 | +3.183% |
| **快手极速版** | | |
| | 应用停留时间 | +0.741% |
| | LT7 | +0.034% |
| | 观看时长 | +0.762% |
| | 视频观看量 | +0.259% |
| | 点赞 | +5.393% |
| | 关注 | +5.627% |
| | 评论 | +5.013% |
| | 收藏 | +3.202% |
| | 转发 | +7.958% |

---

## 5. 结论、局限性与未来方向

在本文中，我们介绍了OneRec-V2，它在OneRec-V1的基础上构建。我们深入探讨了其扩展和奖励系统的设计。关于扩展，我们发现虽然OneRec-V1模型使用MoE在解码器中分配了大量参数，但由于序列长度差异，上下文编码过程消耗了大部分计算资源，阻碍了进一步的可扩展性和性能。因此，我们重新思考了模型架构并提出了惰性仅解码器架构，将计算转移到解码阶段，从而允许进一步的模型扩展（目前已扩展到8B）。此外，我们开发了一种有效利用真实用户反馈来对齐用户偏好的方法。与V1仅依赖奖励模型进行对齐不同，我们纳入了真实的用户反馈信号，并通过创新设计，在短视频观看时间与长期满意度之间建立了关联。此外，使用GBPO，我们实现了高度稳定的训练。严格的A/B实验证明了该框架的有效性。然而，我们的系统仍有改进空间。例如：

1. **扩展**：虽然我们观察到随着模型从0.1B扩展到8B，损失持续下降，但下降趋势并未严格遵循扩展定律（Kaplan et al., 2020）。这表明模型扩展需要进一步探索，我们将继续在数据组织、模型架构和预训练方法等方面投入研究。
2. **奖励系统**：我们新近将真实用户反馈纳入了奖励系统，这已被证明是有效的。然而，我们当前的解决方案建立了连接短期和长期回报的规则，而非让模型直接优化其长期价值。我们将朝这个方向优化，使模型能够实现对长期价值的自我强化。

除了在快手平台视频推荐中实现盈利外，OneRec-V2已被部署到多个业务场景中，产生了可观的价值，例如（Wei et al., 2025）。我们相信，通过更多研究人员和工程师的迭代、验证和优化，该系统可以进一步改进。

---

## 参考文献

（以下参考文献保留原文，不做翻译）

J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L. Aleman, D. Almeida, J. Altenschmidt, S. Altman, S. Anadkat, et al. Gpt-4 technical report. *arXiv preprint arXiv:2303.08774*, 2023.

J. Ainslie, J. Lee-Thorp, M. De Jong, Y. Zemlyanskiy, F. Lebrón, and S. Sanghai. Gqa: Training generalized multi-query transformer models from multi-head checkpoints. *arXiv preprint arXiv:2305.13245*, 2023.

A. Badrinath, P. Agarwal, L. Bhasin, J. Yang, J. Xu, and C. Rosenberg. Pinrec: Outcome-conditioned, multi-token generative retrieval for industry-scale recommendation systems. *arXiv preprint arXiv:2504.10507*, 2025.

T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell, et al. Language models are few-shot learners. *Advances in neural information processing systems*, 33:1877–1901, 2020.

J. Chen, L. Chi, B. Peng, and Z. Yuan. Hllm: Enhancing sequential recommendations via hierarchical large language models for item and user modeling. *arXiv preprint arXiv:2409.12740*, 2024.

Z. Cui, J. Ma, C. Zhou, J. Zhou, and H. Yang. M6-rec: Generative pretrained language models are open-ended recommender systems. *arXiv preprint arXiv:2205.08084*, 2022.

J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In *Proceedings of the 2019 conference of the North American chapter of the association for computational linguistics: human language technologies, volume 1 (long and short papers)*, pages 4171–4186, 2019.

C. Feng, W. Li, D. Lian, Z. Liu, and E. Chen. Recommender forest for efficient retrieval. *Advances in Neural Information Processing Systems*, 35:38912–38924, 2022.

A. Gangwar and S. Jain. An adaptive boosting technique to mitigate popularity bias in recommender system. *arXiv preprint arXiv:2109.05677*, 2021.

A. Gharahighehi, C. Vens, and K. Pliakos. Fair multi-stakeholder news recommender system with hypergraph ranking. *Information Processing & Management*, 58(5):102663, 2021.

D. Guo, D. Yang, H. Zhang, J. Song, R. Zhang, R. Xu, Q. Zhu, S. Ma, P. Wang, X. Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. *arXiv preprint arXiv:2501.12948*, 2025.

R. Han, B. Yin, S. Chen, H. Jiang, F. Jiang, X. Li, C. Ma, M. Huang, X. Li, C. Jing, et al. Mtgr: Industrial-scale generative recommendation framework in meituan. *arXiv preprint arXiv:2505.18654*, 2025.

J. He, J. Liu, C. Y. Liu, R. Yan, C. Wang, P. Cheng, X. Zhang, F. Zhang, J. Xu, W. Shen, et al. Skywork open reasoner 1 technical report. *arXiv preprint arXiv:2505.22312*, 2025.

J. Hoffmann, S. Borgeaud, A. Mensch, E. Buchatskaya, T. Cai, E. Rutherford, D. d. L. Casas, L. A. Hendricks, J. Welbl, A. Clark, et al. Training compute-optimal large language models. *arXiv preprint arXiv:2203.15556*, 2022.

J. Huang, H. Oosterhuis, and M. De Rijke. It is different when items are older: Debiasing recommendations when selection bias and user preferences are dynamic. In *Proceedings of the fifteenth ACM international conference on web search and data mining*, pages 381–389, 2022.

Y. Ji, A. Sun, J. Zhang, and C. Li. A critical study on data leakage in recommender system offline evaluation. *ACM Transactions on Information Systems*, 41(3):1–27, 2023.

J. Kaplan, S. McCandlish, T. Henighan, T. B. Brown, B. Chess, R. Child, S. Gray, A. Radford, J. Wu, and D. Amodei. Scaling laws for neural language models. *arXiv preprint arXiv:2001.08361*, 2020.

A. Klimashevskaia, D. Jannach, M. Elahi, and C. Trattner. A survey on popularity bias in recommender systems (2023). *CoRR*, abs/2308.01118.

L. Kong, L. Wang, C. Peng, Z. Lin, C. Law, and J. Shao. Generative click-through rate prediction with applications to search advertising. *arXiv preprint arXiv:2507.11246*, 2025.

A. Liu, B. Feng, B. Xue, B. Wang, B. Wu, C. Lu, C. Zhao, C. Deng, C. Zhang, C. Ruan, et al. Deepseek-v3 technical report. *arXiv preprint arXiv:2412.19437*, 2024.

A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, I. Sutskever, et al. Language models are unsupervised multitask learners. *OpenAI blog*, 1(8):9, 2019.

C. Raffel, N. Shazeer, A. Roberts, K. Lee, S. Narang, M. Matena, Y. Zhou, W. Li, and P. J. Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. *Journal of machine learning research*, 21(140):1–67, 2020.

S. Rajput, N. Mehta, A. Singh, R. Hulikal Keshavan, T. Vu, L. Heldt, L. Hong, Y. Tay, V. Tran, J. Samost, et al. Recommender systems with generative retrieval. *Advances in Neural Information Processing Systems*, 36:10299–10315, 2023.

J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov. Proximal policy optimization algorithms. *arXiv preprint arXiv:1707.06347*, 2017.

Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, X. Bi, H. Zhang, M. Zhang, Y. Li, Y. Wu, et al. Deepseek-math: Pushing the limits of mathematical reasoning in open language models. *arXiv preprint arXiv:2402.03300*, 2024.

H. Touvron, T. Lavril, G. Izacard, X. Martinet, M.-A. Lachaux, T. Lacroix, B. Rozière, N. Goyal, E. Hambro, F. Azhar, et al. Llama: Open and efficient foundation language models. *arXiv preprint arXiv:2302.13971*, 2023a.

H. Touvron, L. Martin, K. Stone, P. Albert, A. Almahairi, Y. Babaei, N. Bashlykov, S. Batra, P. Bhargava, S. Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. *arXiv preprint arXiv:2307.09288*, 2023b.

A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin. Attention is all you need. *Advances in neural information processing systems*, 30, 2017.

Z. Wei, K. Cai, J. She, J. Chen, M. Chen, Y. Zeng, Q. Luo, W. Zeng, R. Tang, K. Gai, et al. Oneloc: Geo-aware generative recommender systems for local life service. *arXiv preprint arXiv:2508.14646*, 2025.

A. Yang, A. Li, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Gao, C. Huang, C. Lv, et al. Qwen3 technical report. *arXiv preprint arXiv:2505.09388*, 2025a.

Y. Yang, Z. Ji, Z. Li, Y. Li, Z. Mo, Y. Ding, K. Chen, Z. Zhang, J. Li, S. Li, et al. Sparse meets dense: Unified generative recommendations with cascaded sparse-dense representations. *arXiv preprint arXiv:2503.02453*, 2025b.

Q. Yu, Z. Zhang, R. Zhu, Y. Yuan, X. Zuo, Y. Yue, W. Dai, T. Fan, G. Liu, L. Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. *arXiv preprint arXiv:2503.14476*, 2025.

J. Zhai, L. Liao, X. Liu, Y. Wang, R. Li, X. Cao, L. Gao, Z. Gong, F. Gu, M. He, et al. Actions speak louder than words: Trillion-parameter sequential transducers for generative recommendations. *arXiv preprint arXiv:2402.17152*, 2024.

G. Zhou, J. Deng, J. Zhang, K. Cai, L. Ren, Q. Luo, Q. Wang, Q. Hu, R. Huang, S. Wang, et al. Onerec technical report. *arXiv preprint arXiv:2506.13695*, 2025.

Z. Zhu, Y. He, X. Zhao, and J. Caverlee. Popularity bias in dynamic recommendation. In *Proceedings of the 27th ACM SIGKDD conference on knowledge discovery & data mining*, pages 2439–2449, 2021.

---

## 附录A. 贡献者

在每个角色内，作者按名字字母顺序排列。

**核心贡献者**
郭瑞（Guorui Zhou）、胡恒瑞（Hengrui Hu）、程洪涛（Hongtao Cheng）、王欢杰（Huanjie Wang）、邓佳欣（Jiaxin Deng）、张景浩（Jinghao Zhang）、蔡阔（Kuo Cai）、任乐健（Lejian Ren）、任璐（Lu Ren）、于廖（Liao Yu）、郑鹏飞（Pengfei Zheng）、罗强（Qiang Luo）、王茜茜（Qianqian Wang）、胡启艮（Qigen Hu）、黄睿（Rui Huang）、唐瑞明（Ruiming Tang）、王诗瑶（Shiyao Wang）、杨舒杰（Shujie Yang）、吴涛（Tao Wu）、李武昌（Wuchao Li）、罗新辰（Xinchen Luo）、王星梅（Xingmei Wang）、苏羿（Yi Su）、吴云帆（Yunfan Wu）、程泽宣（Zexuan Cheng）、刘占宇（Zhanyu Liu）、张子兴（Zixing Zhang）

**贡献者**
张斌（Bin Zhang）、王博轩（Boxuan Wang）、马朝毅（Chaoyi Ma）、宋成儒（Chengru Song）、王晨辉（Chenhui Wang）、褚成龙（Chenglong Chu）、王迪（Di Wang）、孟冬雪（Dongxue Meng）、臧敦举（Dunju Zang）、杨帆（Fan Yang）、张方宇（Fangyu Zhang）、蒋峰（Feng Jiang）、张福兴（Fuxing Zhang）、王刚（Gang Wang）、张国旺（Guowang Zhang）、李涵（Han Li）、鲍红辉（Honghui Bao）、曹弘扬（Hongyang Cao）、黄嘉铭（Jiaming Huang）、陈嘉鹏（Jiapeng Chen）、刘佳强（Jiaqiang Liu）、贾静辉（Jinghui Jia）、盖坤（Kun Gai）、胡兰涛（Lantao Hu）、曾亮（Liang Zeng）、王强（Qiang Wang）、周启东（Qidong Zhou）、张荣洲（Rongzhou Zhang）、王圣哲（Shengzhe Wang）、何世辉（Shihui He）、杨双（Shuang Yang）、毛思扬（Siyang Mao）、黄穗（Sui Huang）、何甜甜（Tiantian He）、高婷婷（Tingting Gao）、苑伟（Wei Yuan）、梁潇（Xiao Liang）、许晓晓（Xiaoxiao Xu）、刘旭刚（Xugang Liu）、王岩（Yan Wang）、周阳（Yang Zhou）、王毅（Yi Wang）、刘义武（Yiwu Liu）、宋玥（Yue Song）、张宇飞（Yufei Zhang）、赵云峰（Yunfeng Zhao）、凌志新（Zhixin Ling）、李梓铭（Ziming Li）

---

## 附录B. 不同架构的计算复杂度

**预备知识。** 在实际的推荐系统中，多个item会同时曝光。一个关键的优化是公共上下文压缩：当向同一用户推荐k个item时，共享的上下文信息（用户画像、历史行为）只需处理一次，并且可以跨所有目标item复用。这将有效上下文长度从N减少到每个item约N/k个词元。在快手， $k = 5$ 。

Transformer块（Vaswani et al., 2017）中的主要计算组件包括：（1）前馈网络（FFN），（2）注意力投影（ $W_q$ , $W_k$ , $W_v$ , $W_o$ ），以及（3）注意力分数计算。它们的计算复杂度为：

- 注意力投影： $O(L \cdot 4d_{\text{model}}^2)$
- FFN： $O(L \cdot d_{\text{model}} \cdot d_{\text{ff}}) \approx O(L \cdot 4d_{\text{model}}^2)$
- 注意力分数： $O(L^2 \cdot d_{\text{model}})$

其中 $L$ 是这些模块处理的词元数量， $d_{\text{model}}$ 是模型的隐藏维度。值得注意的是，FFN和注意力投影都可以近似为 $O(L \cdot D)$ ，其中 $D$ 是相应模块的参数数量。

**编码器-解码器架构。** 我们分析了一个编码器和解码器组件中各拥有0.5B参数的编码器-解码器模型的计算需求。在使用压缩上下文长度 $N/5$ 进行训练时，浮点运算（FLOPs）分解如下：

- 上下文变换（编码器）： $6 \times 0.5\text{B} \times \frac{N}{5} = 0.6N$ GFLOPs
- 上下文投影（交叉注意力）： $6 \times 0.05\text{B} \times \frac{N}{5} = 0.06N$ GFLOPs
- 上下文解码： $0.6N + 0.06N = 0.66N$ GFLOPs
- 目标解码： $6 \times 0.45\text{B} \times 3 = 8.1$ GFLOPs
- 总计算量： $0.66N + 8.1$ GFLOPs

其中因子6同时考虑了乘加运算（贡献因子2）和正向-反向传播比率（贡献因子3）。交叉注意力机制中的上下文投影矩阵（ $W_k$ , $W_v$ ）位于解码器内，约占解码器参数的10%（0.05B）。

这里忽略了注意力分数的计算。考虑特定模型配置：9个编码器层和9个解码器层， $d_{\text{model}} = 1792$ 。注意力分数的计算为：编码器： $6 \times 9 \times (\frac{N}{5})^2 \times 1792 = 3.8N^2$ KFLOPs，解码器： $6 \times 9 \times 3 \times N \times 1792 = 290N$ KFLOPs。当 $N = 512$ 时，这些值比FFN和注意力投影小几个数量级。

**朴素仅解码器架构。** 对于处理 $N/5 + 3$ 个词元且带有因果注意力掩码的1B参数仅解码器模型：

- 上下文解码： $6 \times 1\text{B} \times \frac{N}{5} = 1.2N$ GFLOPs
- 目标解码： $6 \times 1\text{B} \times 3 = 18$ GFLOPs
- 总计算量： $1.2N + 18$ GFLOPs

---

## 附录C. 实验结果

我们进行实验以研究OneRec-V2模型的模型大小、计算预算和训练损失之间的关系。图11展示了各规模模型的平滑生成训练损失曲线，作为总计算量（以FLOPs衡量）的函数。具体而言，较大的模型需要更多的计算资源才能达到相同的损失值，但它们也收敛到更低的损失点，这与大型语言模型领域的观察结果一致。

**图10 | 不同交叉注意力配置的训练损失曲线。键值共享（左）和分组查询注意力（右）策略在显著提高计算效率的同时，对收敛损失的影响极小。**

**图11 | OneRec-V2模型的平滑生成训练损失曲线作为总计算量（以FLOPs衡量）的函数，展示了各模型规模下的扩展行为和收敛模式。橙色线连接了不同模型实现的最小损失点。**

---

## 附录D. 禁用缓存时的在线性能

如第4节所述，我们的实验组流量为5%，其中OneRec-V2应用于该组内25%的降级流量。为进行更严格的比较，我们另外划出一个禁用缓存的1%实验组（在此组中，所有流量请求OneRec-V2）。性能如表9所示。

当所有流量请求OneRec-V2时，我们观察到关键参与度指标（包括观看时长和用户交互指标）的显著改进。具体而言，点赞、关注、评论和转发等交互指标在不同平台上表现出从9.6%到29.2%的显著提升。然而，某些生态系统级指标呈现出令人担忧的趋势。值得注意的是，冷启动视频观看量出现显著下降（快手和快手极速版分别下降44.7%和36.7%），同时簇密度显著增加（分别为11.7%和7.9%）。这是一个关键挑战，需要在未来的方向中认真考虑。

**表9 | OneRec-V2相对于OneRec-V1的相对改进（在此组中，所有流量请求OneRec-V2）。**

| 场景 | 在线指标 | OneRec-V2 |
|---|---|---|
| **快手** | | |
| | 应用停留时间 | +0.405% |
| | 观看时长 | +0.513% |
| | 视频观看量 | +0.938% |
| | 点赞 | +15.024% |
| | 关注 | +15.755% |
| | 评论 | +29.249% |
| | 收藏 | +9.640% |
| | 转发 | +24.741% |
| | 冷启动视频观看量 | -44.704% |
| | 簇密度 | +11.692% |
| **快手极速版** | | |
| | 应用停留时间 | +0.958% |
| | 观看时长 | +2.456% |
| | 视频观看量 | -1.121% |
| | 点赞 | +12.783% |
| | 关注 | +21.376% |
| | 评论 | +16.975% |
| | 收藏 | +12.886% |
| | 转发 | +30.957% |
| | 冷启动视频观看量 | -36.730% |
| | 簇密度 | +7.933% |
