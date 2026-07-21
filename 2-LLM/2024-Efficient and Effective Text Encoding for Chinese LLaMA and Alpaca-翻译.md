# Efficient and Effective Text Encoding for Chinese LLaMA and Alpaca

> Yiming Cui\*, Ziqing Yang\*, Xin Yao | 同等贡献

中文LLaMA系列：https://github.com/ymcui/Chinese-LLaMA-Alpaca
中文Llama-2系列：https://github.com/ymcui/Chinese-LLaMA-Alpaca-2

大型语言模型（LLM），如ChatGPT和GPT-4，已彻底改变了自然语言处理研究，并向通用人工智能（AGI）迈出了可喜的步伐。然而，训练和部署LLM的高昂成本给透明、可及的学术研究带来了巨大障碍。尽管社区已开源了若干大型语言模型（如LLaMA），但这些模型主要侧重于英文语料，限制了其在其他语言中的实用性。本文提出了一种增强LLaMA理解和生成中文文本能力及其遵循指令能力的方法。我们通过将LLaMA现有词表扩展20,000个中文token，从而提高其中文编码效率和语义理解。我们进一步利用中文数据进行二次预训练，并使用中文指令数据集对模型进行微调，显著增强了模型理解和执行指令的能力。实验结果表明，新提出的模型明显提升了原始LLaMA在中文内容理解和生成方面的熟练度。此外，在C-Eval数据集上的结果展现出了与数倍规模模型相竞争的性能。我们已在GitHub上公开了预训练模型、训练脚本及其他资源，以促进社区的开放研究。

---

## 摘要

大型语言模型（LLM），如ChatGPT和GPT-4，已彻底改变了自然语言处理研究，并向通用人工智能（AGI）迈出了可喜的步伐。然而，训练和部署LLM的高昂成本给透明、可及的学术研究带来了巨大障碍。尽管社区已开源了若干大型语言模型（如LLaMA），但这些模型主要侧重于英文语料，限制了其在其他语言中的实用性。本文提出了一种增强LLaMA理解和生成中文文本能力及其遵循指令能力的方法。我们通过将LLaMA现有词表扩展20,000个中文token，从而提高其中文编码效率和语义理解。我们进一步利用中文数据进行二次预训练，并使用中文指令数据集对模型进行微调，显著增强了模型理解和执行指令的能力。实验结果表明，新提出的模型明显提升了原始LLaMA在中文内容理解和生成方面的熟练度。此外，在C-Eval数据集上的结果展现出了与数倍规模模型相竞争的性能。我们已在GitHub上公开了预训练模型、训练脚本及其他资源，以促进社区的开放研究。

---

## 1 引言

自然语言处理（NLP）领域随着大型语言模型（LLM）的出现经历了重大的范式转变。这些模型以其庞大的规模和全面的训练数据著称，在理解和生成类人文本方面展现了非凡的能力。与专注于文本理解的预训练语言模型（如BERT [1]）不同，GPT系列 [2] 强调文本生成，使其成为相比同类模型更适合作创造力的平台。值得注意的是，GPT家族的最新成员——ChatGPT和GPT-4——已获得广泛关注，成为这一快速演进领域中的领先典范。

ChatGPT [3] 从InstructGPT [4] 演进而来，是一个先进的对话式AI模型，能够进行上下文感知的类人交互。其成功为GPT-4 [5] 的开发奠定了基础——这是一个更复杂的LLM，在自然语言理解、生成及各种NLP任务中展现出更大的潜力，尤其是在多模态和推理能力方面。这些模型催生了新的研究方向和应用，激发了对通用人工智能（AGI）潜力的探索兴趣。它们在多个基准测试中展现出令人印象深刻的性能，同时还表现出少样本学习和适应新任务的能力，极大地推动了NLP研究的发展。因此，它们激励了研究人员和行业专业人士在广泛的应用中进一步挖掘其潜力，包括情感分析、机器翻译、问答系统等。

然而，尽管LLM影响深远，但其实现存在固有局限性，阻碍了透明和开放的研究。一个主要问题是其专有性质，限制了对模型的访问，从而抑制了更广泛研究社区在其成功基础上继续发展的能力。此外，训练和部署这些模型所需的巨大计算资源对资源有限的研究人员构成了挑战，进一步加剧了可及性问题。

为解决这些局限性，NLP研究社区已转向开源替代方案，以促进更大的透明度和协作。LLaMA [6]、Llama-2 [7] 和Alpaca [8] 是此类倡议的显著例子。这些开源LLM旨在促进学术研究并加速NLP领域的进展。开源这些模型的目标是营造一个有利于模型开发、微调和评估进一步发展的环境，最终创建适用于广泛用途的鲁棒且强大的LLM。

尽管LLaMA和Alpaca在NLP领域取得了长足进步，但它们在原生支持中文语言任务方面存在固有局限性。其词表仅包含几百个中文token，严重阻碍了其编码和解码中文文本的效率。基于我们之前在中文BERT系列 [9] 和面向中国少数民族的多语言预训练模型 [10] 方面的工作，本技术报告中，我们提出了开发具有增强中文内容理解和生成能力的中文LLaMA和Alpaca模型。我们将原始LLaMA的词表扩展了20,000个额外中文token，显著提高了其处理中文文本的熟练度。为确保模型的高效训练和部署，我们采用低秩适应（LoRA）方法 [11]，使我们能够在不产生过多计算成本的情况下训练和微调模型。我们希望这项初步研究能增强LLaMA和Alpaca的中文理解和生成能力，为研究人员将这些模型适配到其他语言奠定基础。通过展示我们方法的可行性和有效性，我们提供了可应用于扩展词表并提高LLaMA和Alpaca模型在各种语言中性能的见解和方法论。所提出模型的概述如图1所示。

![图1：所提出的中文LLaMA和中文Alpaca模型概览（基于Meta的LLaMA和Llama-2）。中文LLaMA系列为基础模型，中文Alpaca系列为对话或指令遵循模型。]()

总之，本技术报告的贡献如下：

* 我们通过将原始LLaMA词表扩展20,000个中文token，增强了中文语言的编码和解码效率，并提高了LLaMA的中文理解能力。
* 我们采用低秩适应（LoRA）方法促进中文LLaMA和Alpaca模型的高效训练和部署，使研究人员能够在不产生过多计算成本的情况下使用这些模型。
* 我们在指令遵循任务和自然语言理解任务上评估了所提出的LLaMA和Alpaca模型的性能，从而证明了其在中文语言任务上相较于原始模型的显著提升。
* 我们公开了研究成果和资源，促进了NLP社区的进一步研究和协作，并鼓励将LLaMA和Alpaca模型适配到其他语言。

## 2 中文LLaMA与中文Alpaca

### 2.1 背景

LLaMA [6] 是一个基于transformer架构 [12] 的基础性、仅解码器的大型语言模型。与GPT系列及其他基于transformer的LLM类似，LLaMA由嵌入层、多个transformer块和语言模型头组成。LLaMA还融合了不同模型中使用的改进技术，如预归一化 [13]、SwiGLU激活 [14] 和旋转位置编码 [15]。LLaMA提供四种不同的模型规模：7B、13B、33B和65B。

LLaMA使用标准语言建模任务（参见2.4节）在公开可用资源的混合语料上进行了预训练，包括爬取的网页、书籍、维基百科和预印本论文。实验发现表明，LLaMA在较小模型规模下也能提供与GPT-3等其他LLM相竞争的性能。这种紧凑性和有效性引起了研究人员的广泛关注，导致基于LLaMA的模型被广泛使用。

### 2.2 中文词表扩展

LLaMA的训练集包含约1.4T token，其中大部分为英文，一小部分为使用拉丁或西里尔字母的其他欧洲语言 [6]。因此，LLaMA具备多语言和跨语言理解能力，主要体现在欧洲语言上。有趣的是，我们之前的初步研究表明，LLaMA展示出基本的中文理解能力，但其生成中文文本的能力有限。

为使LLaMA具备增强的中文理解和生成能力，我们提出使用中文语料对LLaMA模型进行继续预训练。然而，直接应用中文语料的继续预训练面临若干挑战。首先，原始LLaMA词表仅覆盖不到一千个中文字符，不足以编码通用中文文本。虽然LLaMA分词器通过将未知的UTF-8字符拆分为字节来规避此问题，但这一策略显著延长了序列长度并降低了中文文本的编码和解码效率，因为每个中文字符会拆分为3–4个字节token。其次，字节token并非专门设计用于表示中文字符。由于字节token也代表其他语言中的UTF-8 token，字节token和transformer编码器很难有效学习捕捉中文字符语义含义的表示。

为解决这些问题并提高编码效率，我们提出使用额外中文token扩展LLaMA词表，并使模型适配扩展后的词表 [10]。扩展过程如下：

* 为增强分词器对中文文本的支持，我们首先在中文语料上使用SentencePiece [16] 训练一个中文分词器，词表大小为20,000。
* 然后，我们取原始LLaMA分词器与中文分词器词表的并集，将中文分词器合并到原始LLaMA分词器中。由此得到一个合并后的分词器，我们称之为中文LLaMA分词器，词表大小为49,953。
* 为使LLaMA模型适配中文LLaMA分词器，我们将词嵌入和语言模型头的形状从 $V \times H$ 调整为 $V' \times H$，其中 $V = 32{,}000$ 表示原始词表大小，$V' = 49{,}953$ 是中文LLaMA分词器的新词表大小。新增的行追加到原始嵌入矩阵的末尾，确保原始词表中token的嵌入不受影响。

初步实验表明，中文LLaMA分词器生成的token数量约为原始LLaMA分词器的一半。表1提供了原始LLaMA分词器与我们的中文LLaMA分词器之间的比较。如图所示，中文LLaMA分词器相较于原始分词器显著缩短了编码长度。在固定上下文长度下，模型可容纳约两倍的信息量，生成速度是原始LLaMA分词器的两倍。这凸显了我们提出的方法在增强LLaMA模型中文理解和生成能力方面的有效性。

**表1：原始LLaMA与中文LLaMA的分词器比较。**

| | 长度 | 内容 |
|:---:|:---:|:---|
| 原始句子 | 28 | 人工智能是计算机科学、心理学、哲学等学科融合的交叉学科。 |
| 原始分词器 | 35 | ' ', '人', '工', '智', '能', '是', '计', '算', '机', '科', '学', '、', '心', '理', '学', '、', '0xE5', '0x93', '0xB2', '学', '等', '学', '科', '0xE8', '0x9E', '0x8D', '合', '的', '交', '0xE5', '0x8F', '0x89', '学', '科', '。' |
| 中文分词器 | 16 | ' ', '人工智能', '是', '计算机', '科学', '、', '心理学', '、', '哲学', '等', '学科', '融合', '的', '交叉', '学科', '。' |

### 2.3 基于LoRA的参数高效微调

更新LLM全部参数的传统训练范式成本过高，对大多数实验室或公司而言在时间或成本上不可行。低秩适应（LoRA）[11] 是一种参数高效的训练方法，它在保持预训练模型权重的同时引入可训练的低秩分解矩阵。LoRA冻结预训练模型权重，并在每层中注入可训练的低秩矩阵。该方法显著减少了总可训练参数量，使得用更少的计算资源训练LLM成为可能。

具体而言，对于一个权重矩阵为 $W_0 \in \mathbb{R}^{d \times k}$ 的线性层，其中 $k$ 为输入维度，$d$ 为输出维度，LoRA添加两个低秩分解的可训练矩阵 $B \in \mathbb{R}^{d \times r}$ 和 $A \in \mathbb{R}^{r \times k}$，其中 $r$ 为预定义的秩。输入 $x$ 的前向传播由以下方程给出：

$$h = W_0 x + \Delta W x = W_0 x + BA x, \quad B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times d} \qquad (1)$$

训练过程中，$W_0$ 被冻结且不接收梯度更新，而 $B$ 和 $A$ 被更新。通过选择秩 $r \ll \min(d, k)$，由于我们不需要存储大型冻结矩阵的优化器状态，内存消耗得以降低。

为在紧预算下实现参数高效训练，我们在本文中所有中文LLaMA和Alpaca模型上应用LoRA训练，包括预训练和微调阶段。我们主要将LoRA适配器融入注意力模块和MLP层的权重中。将LoRA应用于所有线性transformer块的有效性已在QLoRA [17] 中得到验证，表明我们的选择是合理的。

### 2.4 预训练目标

我们使用标准因果语言建模（CLM）任务对中文LLaMA模型进行预训练。给定输入token序列 $x = (x_0, x_1, x_2, \ldots)$，模型以自回归方式训练预测下一个token $x_i$。数学上，目标是最小化以下负对数似然：

$$\mathcal{L}_{\text{CLM}}(\Theta) = \mathbb{E}_{x \sim \mathcal{D}_{\text{PT}}} \left[ -\sum_i \log p(x_i | x_0, x_1, \ldots, x_{i-1}; \Theta) \right] \qquad (2)$$

其中，$\Theta$ 表示模型参数，$\mathcal{D}_{\text{PT}}$ 是预训练数据集，$x_i$ 为待预测的token，$x_0, x_1, \ldots, x_{i-1}$ 构成上下文。

### 2.5 监督微调与中文Alpaca

预训练语言模型很难遵循用户指令，且常常生成非预期内容。这是因为式（2）中的语言建模目标是预测下一个token，而非"遵循指令并回答问题" [4]。为使语言模型的行为与用户意图对齐，可以对模型进行微调，显式训练其遵循指令。Stanford Alpaca [18] 是一个基于LLaMA的指令遵循模型，在52K由Self-Instruct [19] 技术生成的指令遵循数据上训练而成。我们遵循Stanford Alpaca的方法，在中文LLaMA上应用自指令微调，训练了一个指令遵循模型——中文Alpaca。

中文Alpaca在指令遵循数据集的组合上进行训练。数据集中的每个示例由一条指令和一个输出组成。监督微调任务与因果语言建模任务类似：模型以指令作为提示，并训练其自回归地生成输出。指令被包装在提示模板中，输出紧随模板。我们采用Stanford Alpaca的以下模板进行微调和推理，输入序列如下所示：

```
Below is an instruction that describes a task.
Write a response that appropriately
completes the request.
### Instruction:
{instruction}
### Response: {output}
```

损失仅在输入序列的 `{output}` 部分计算，可表示为：

$$\mathcal{L}_{\text{SFT}}(\Theta) = \mathbb{E}_{x \sim \mathcal{D}_{\text{SFT}}} \left[ -\sum_{i \in \{\text{output}\}} \log p(x_i | x_0, x_1, \ldots, x_{i-1}; \Theta) \right] \qquad (3)$$

此处，$\Theta$ 表示模型参数，$\mathcal{D}_{\text{SFT}}$ 是微调数据集，$x = (x_0, x_1, \ldots)$ 表示分词后的输入序列。

我们的方法与Stanford Alpaca的一个主要区别在于，我们仅使用为没有输入字段的示例设计的提示模板，而Stanford Alpaca为有输入字段和无输入字段的示例分别使用两种模板。如果示例包含非空的输入字段，我们将指令和输入用"\n"拼接以形成新的指令。请注意，中文Alpaca模型有一个额外的填充token，因此词表大小为49,954。

## 3 实验设置

### 3.1 预训练实验设置

我们使用原始LLaMA权重初始化中文LLaMA模型，并在7B和13B模型上使用fp16进行预训练。此外，对于33B模型，我们使用bitsandbytes库以8位格式训练，提升了效率和内存使用。我们直接将LoRA应用于注意力和MLP进行训练，同时将嵌入和LM头设置为可训练。

对于基础版中文LLaMA-7B，我们采用两阶段预训练方法。在第一阶段，我们固定模型内transformer编码器的参数，仅训练嵌入层，使新添加的中文词向量适应模型，同时最小化对原始模型的扰动。在第二阶段，我们向注意力机制添加LoRA权重（适配器），并训练嵌入层、LM头和新添加的LoRA参数。需要注意的是，两阶段训练未应用于其他模型训练，因为初步研究中发现其效率较低。

对于其他中文LLaMA模型（基础版），我们使用20GB通用中文语料进行预训练，该语料与中文BERT-wwm [9]、MacBERT [20]、LERT [21] 等使用的语料一致。我们还提供了"Plus"版本，进一步将预训练数据扩展至120GB，融入了来自CommonCrawl（CC）和百科源的其他数据，增强了模型对基本概念的理解。我们将所有数据集拼接起来，生成块大小为512的数据块用于预训练。

模型在A40 GPU（48GB显存）上训练一个epoch，根据模型大小最多使用48块GPU。基于LoRA的参数高效训练使用PEFT库进行。我们还利用DeepSpeed [22] 优化训练过程中的内存效率。我们采用AdamW优化器 [23]，峰值学习率为 $2 \times 10^{-4}$，使用5%预热余弦调度器。此外，我们应用值为1.0的梯度裁剪以缓解潜在的梯度爆炸。

中文LLaMA各模型的详细超参数列于表2。

**表2：中文LLaMA的预训练超参数。QKVO：每个注意力模块中的四个矩阵，即query、key、value和output。MLP：每个MLP层中的三个矩阵。注意，7B使用两阶段训练范式（设置以'/'分隔），其他模型未再采用。**

| 设置 | 7B | Plus-7B | 13B | Plus-13B | 33B |
|:---|:---:|:---:|:---:|:---:|:---:|
| 训练数据 | 20 GB | 120 GB | 20 GB | 120 GB | 20 GB |
| 批大小 | 1,024 | 2,304 | 2,304 | 2,304 | 2,304 |
| 峰值学习率 | $2 \times 10^{-4} / 1 \times 10^{-4}$ | $2 \times 10^{-4}$ | $2 \times 10^{-4}$ | $2 \times 10^{-4}$ | $2 \times 10^{-4}$ |
| 最大序列长度 | 512 | 512 | 512 | 512 | 512 |
| LoRA秩 | $-$ / 8 | 8 | 8 | 8 | 8 |
| LoRA alpha | $-$ / 32 | 32 | 32 | 32 | 32 |
| LoRA权重 | $-$ / QKVO | QKVO, MLP | QKVO, MLP | QKVO, MLP | QKVO, MLP |
| 可训练参数占比 | 2.97% / 6.06% | 6.22% | 4.10% | 4.10% | 2.21% |

### 3.2 指令微调实验设置

获得中文LLaMA模型后，我们根据2.5节对其进行微调。我们继续使用LoRA进行高效微调，向基础模型的所有线性层添加LoRA模块。我们使用约2M至3M条指令数据，包括翻译 [24]（采样550K）、pCLUE（采样250K，排除"类NLU"数据）、Stanford Alpaca（原文和翻译各50K，共100K）以及爬取的SFT数据用于调整基础模型。对于Plus版本，我们将数据集扩展至约4M至4.3M，特别强调融入STEM（科学、技术、工程和数学）数据以及若干科学学科，如物理、化学、生物、医学和地球科学。对于Alpaca-33B，我们额外添加了OASST1数据集 [25]，其中仅从每个对话中提取第一个query-response对，并使用gpt-3.5-turbo API进行翻译，产生约20K条数据（原文和翻译）。我们将最大序列长度设置为512，并在批处理时动态填充样本至批次中的最大长度。

对于爬取的数据，我们参考self-instruct [19] 方法，从ChatGPT（gpt-3.5-turbo API）自动获取数据，如Taori等人 [8] 所用。具体而言，我们使用一个更简化的模板，不需要种子任务，仅需目标领域和指令类型的要求。模板和代码细节可在GitHub上获取。

**表3：中文Alpaca的指令微调超参数。**

| 设置 | 7B | Plus-7B | 13B | Plus-13B | 33B |
|:---|:---:|:---:|:---:|:---:|:---:|
| 训练数据 | 2M | 4M | 3M | 4.3M | 4.3M |
| 批大小 | 512 | 1,152 | 1,152 | 1,152 | 1,152 |
| 峰值学习率 | $1 \times 10^{-4}$ | $1 \times 10^{-4}$ | $1 \times 10^{-4}$ | $1 \times 10^{-4}$ | $1 \times 10^{-4}$ |
| 最大序列长度 | 512 | 512 | 512 | 512 | 512 |
| LoRA秩 | 8 | 64 | 8 | 64 | 8 |
| LoRA alpha | 32 | 128 | 32 | 128 | 32 |
| LoRA权重 | QKVO, MLP | QKVO, MLP | QKVO, MLP | QKVO, MLP | QKVO, MLP |
| 可训练参数占比 | 6.22% | 8.08% | 4.10% | 5.66% | 2.21% |

对于Plus版本，我们使用比基础版更大的LoRA秩。除了调整学习率和批大小外，我们保持与预训练阶段其他超参数和设置的一致性。

指令微调的超参数列于表3。注意，所有Alpaca模型均基于相应的LLaMA模型训练。例如，Chinese Alpaca-Plus-13B基于Chinese LLaMA-Plus-13B训练。

## 4 指令遵循任务结果

### 4.1 任务设计与评估方法

评估文本生成任务的性能具有挑战性，因为其形式差异显著，与文本分类和抽取式机器阅读理解等自然语言理解任务大不相同。借鉴先前利用GPT-4 [5] 作为评分方法的工作，我们也采用GPT-4为每个样本提供总体评分（10分制），这比人工评估更高效。然而，GPT-4可能并非总能提供准确评分，因此我们对其评分进行人工检查并在必要时调整。人工检查确保评分一致且反映被评估模型的真实性能。我们使用以下提示模板对两个系统的输出进行评分（可调整至多个系统）：

```
The followings are two ChatGPT-like systems' outputs. Please rate an overall score on a
ten-point scale for each and give explanations to justify your scores.
Prompt:
{prompt-input}
System1:
{system1-output}
System2:
{system2-output}
```

通过采用GPT-4作为评分方法并结合人工检查，我们建立了一个可靠的评估框架，能够有效衡量我们的中文Alpaca模型在一系列自然语言理解和生成任务上的性能。

我们的评估集旨在全面评估中文Alpaca模型在广泛自然语言理解和生成任务上的表现。该评估集包含200个样本，涵盖十项不同的任务，包括问答、推理、文学、娱乐、翻译、多轮对话、编程和伦理等。特定任务的总体得分通过将该任务中所有样本的得分求和并归一化至100分制来计算。该方法确保评估集反映模型在各任务上的能力，提供平衡且鲁棒的性能衡量。

### 4.2 解码实验设置

LLM的解码过程在确定生成文本的质量和多样性方面起着关键作用。在我们的实验中，使用以下解码超参数：

* **上下文大小**：我们将上下文大小设置为2048，这决定了模型生成文本时能同时考虑的最大token数量。
* **最大序列长度**：我们将生成的序列长度限制为512个token，以确保输出保持聚焦且与输入提示相关。
* **温度**：我们将温度设置为0.2，控制采样过程的随机性。较低的值使模型生成更聚焦和确定性的输出，而较高的值则增加多样性但牺牲连贯性。对于多轮对话和生成任务，我们将温度略微调整至0.5，以允许更多样化的输出。
* **Top-k采样**：我们使用 $k = 40$ 的Top-k采样，即模型每一步从概率最高的前40个token中选择下一个token，为生成文本增加随机性和多样性。
* **Top-p采样**：我们还采用 $p = 0.9$ 的Top-p采样，通过考虑共同占据90%概率质量的动态token集来进一步增强多样性。
* **重复惩罚**：为抑制模型生成重复文本，我们应用系数为1.1的重复惩罚，对已被选中的token进行惩罚。

注意，这些值可能并非每个测试场景的最优值。我们未对每个任务的这些超参数进行进一步调整，以保持平衡的视角。

### 4.3 结果

我们展示并分析了中文Alpaca-Plus-7B、Alpaca-Plus-13B和Alpaca-33B模型所获得的结果。Alpaca-33B的结果由原始模型（FP16）生成，而Alpaca-Plus-7B和Alpaca-Plus-13B采用8位量化版本。总体结果如表4所示。评估基于GPT-4在十项不同NLP任务上评分的结果，共涵盖200个样本。需要强调的是，所呈现的分数仅在彼此之间可比，不能与其他模型比较，这需要重新评分。同时，由于我们的模型基于原始LLaMA构建，这些观察结果可视为基于成熟模型而非从头训练时获得更好性能的重要方面。我们详细阐述了几个主要类别的发现。

我们主要展示了中文LLaMA和中文Alpaca的结果。中文LLaMA-2和中文Alpaca-2的结果见附录A。

**表4：中文Alpaca-Plus-7B、Alpaca-Plus-13B和Alpaca-33B的GPT-4评分结果。注意，结果仅在此模型组合内可比。**

| 任务 | Alpaca-Plus-7B | Alpaca-Plus-13B | Alpaca-33B |
|:---|:---:|:---:|:---:|
| 问答 | 70.5 | 79.5 | 82.3 |
| 开放式QA | 80.5 | 80.0 | 78.5 |
| 数值推理 | 51.0 | 61.5 | 84.5 |
| 诗歌、文学、哲学 | 78.5 | 81.3 | 76.0 |
| 音乐、体育、娱乐 | 72.3 | 76.8 | 72.5 |
| 信件与文章写作 | 81.0 | 86.5 | 79.0 |
| 翻译 | 86.8 | 89.3 | 92.3 |
| 多轮对话 | 80.3 | 81.3 | 78.0 |
| 编程 | 62.5 | 67.5 | 84.0 |
| 伦理 | 89.8 | 90.5 | 92.5 |
| **总计** | **75.3** | **79.4** | **82.0** |

#### 4.3.1 多轮对话

ChatGPT令人印象深刻的成就之一是其丰富流畅的上下文理解能力，这通过多轮对话界面得以体现。如我们所见，Plus系列模型在基础版之上取得了一致的改进，尽管后者的规模是前者的数倍。这可能表明，摄入更多训练数据比简单扩展模型参数规模对实现更好的对话体验更为重要。尤其是我们的模型从原始LLaMA构建而来，语言知识无法直接迁移。

#### 4.3.2 文本生成

文本生成是语言模型最基本的能力之一。与Alpaca-Plus-7B和Alpaca-Plus-13B相比，Alpaca-33B在此类别中表现较差。表5展示了一个文本生成任务的示例。我们可以看到，Alpaca-Plus-7B和Alpaca-Plus-13B均提供了正确的信件格式，满足了用户提示的要求。Alpaca-Plus-13B提供了最全面的回复，表明申请人已彻底准备好签证申请的所有材料，成为三个系统中生成质量最佳的。然而，Alpaca-33B未遵循信件格式，内容也有些过于简化，明显不如其他模型。这表明，用更多数据训练较小的模型可能比用较少数据训练大模型获得更好的性能。

**表5：文本生成任务的示例输出。33B模型相较Plus-7B和Plus-13B未提供详细和丰富的上下文。**

| 模型 | 输出 |
|:---|:---|
| **Prompt:** 请你帮我写一封赴美旅游签证申请信 | |
| **Plus-7B** (8.5/10) | 尊敬的美国大使馆：<br>我是一名来自中国的游客，计划于2022年7月前往美国旅游。我非常期待这次旅行，并希望获得签证。<br>我将遵守美国的法律和规定，并尊重当地文化和风俗。我将遵守旅游计划，并在离开前向美国大使馆报告我的行踪。<br>感谢您考虑我的申请。<br>此致<br>敬礼！ |
| **Plus-13B** (10/10) | 尊敬的美国大使馆官员，<br>我是xxx，来自中国。我计划于明年3月前往美国进行为期一个月的旅行。我的行程包括参观纽约市、洛杉矶和拉斯维加斯等城市，并与当地的人们交流。我相信这次旅行将对我的个人成长和发展有所帮助。<br>我已经准备好了所有需要的材料，包括机票预订证明、酒店预订证明以及旅游计划。我希望通过此次申请获得签证，以确保我的旅行顺利进行。<br>谢谢您的时间和关注。我期待着能够很快收到您的回复。<br>此致，<br>敬礼 |
| **33B** (8/10) | 尊敬的美国大使馆工作人员，我谨代表本人向您申请一次赴美旅游签证。我是一名中国公民，目前从事互联网行业工作，有良好的职业背景和经济实力。此次旅行的目的是探访美国文化、了解当地风土人情，并结交一些志同道合的朋友。我将遵守美国的法律法规，不违反任何规定。谢谢您的考虑！ |

#### 4.3.3 数值计算与推理

数值推理一直被认为是检验大型语言模型推理能力的最重要任务之一。如我们所见，Alpaca-33B相较于Plus-7B和Plus-13B模型取得了显著提升。表6展示了此任务的示例输出。第一个提示是著名的推理能力测试问题，即"1公斤棉花和1公斤铁哪个更重？"。Plus-7B和Plus-13B均未能给出正确答案，提到"棉花比铁轻"。然而，33B能够正确识别两者重量相同。第二个提示是一个简单的计算任务，问"一只猫和一只鸡一共有几只脚"。然而，如我们所见，Plus-7B和Plus-13B均不具备猫有四条腿、鸡有两条腿的常识知识，导致答案错误。最后一个提示是一个数值推理任务，让模型预测数组的下一个数。同样，只有33B模型正确识别了给定数组的模式，即下一个数应为其索引的平方。这些观察结果表明，模型规模在数值推理任务中至关重要。

**表6：数值推理任务的示例输出。33B模型显著优于Alpaca-Plus-7B和Alpaca-Plus-13B。**

| 模型 | Prompt / 输出 |
|:---|:---|
| **Prompt:** 一公斤的棉花和一公斤的铁，哪一个更重？ | |
| Plus-7B (3/10) | 一公斤棉花比一公斤铁轻。 |
| Plus-13B (3/10) | 一公斤的棉花比一公斤的铁轻。 |
| 33B (9/10) | 一公斤的棉花和一公斤的铁重量相同。 |
| **Prompt:** 一只猫和一只鸡一共有几只脚？ | |
| Plus-7B (1/10) | 一只猫和一只鸡一共有两只脚。 |
| Plus-13B (1/10) | 一只猫和一只鸡各有四只脚，所以两只动物一共有八只脚。 |
| 33B (10/10) | 这道题需要进行简单的数学运算。一只猫有4只脚，一只鸡有2只脚，所以两只动物一共有6只脚。 |
| **Prompt:** 序列1,4,9,16的后一个数是什么？ | |
| Plus-7B (1/10) | 后一个数是16。 |
| Plus-13B (1/10) | 21 |
| 33B (10/10) | 序列1,4,9,16的后一个数是25。 |

#### 4.3.4 编程

图2展示了用Python实现Dijkstra算法的示例。Plus-7B得分为3/10，原因是其结构上看似合理但未能正确计算和更新最短距离，且包含一个未定义的函数。Plus-13B通过实现Graph类和distance方法尝试抽象，展示了对如何使用面向对象编程表示图及其相关操作的基本理解。同时，它尝试实现最短路径算法（尽管未正确实现Dijkstra算法）使其得分略高于Plus-7B。33B模型提供了更优的Dijkstra算法实现，获得了8/10的评分。尽管其缺少优先队列且未进行错误处理（这些会提升效率和鲁棒性），但代码正确更新了最短距离、维护了前驱节点并确保所有节点均被访问，反映出对算法的基本理解。

从这些结果可以推断，较大的模型往往在复杂任务（如代码生成）上表现更好，这可能是由于其能够捕捉训练数据中更复杂的模式。

**表（图2）：编程任务的示例输出。33B模型显著优于Alpaca-Plus-7B和Alpaca-Plus-13B。**

```
Plus-7B (3/10):
1  import heapq
2  
3  def dijkstra(graph, s):
4      queue = []
5      heap = [s]
6      while len(heap) > 0:
7          current = heap.pop()
8          for i in graph[current]:
9              if i not in heap and dist(i, s) < dist(s, current):
10                 heap.append(i)
11     return queue

Plus-13B (4/10):
1  class Graph:
2      def __init__(self, adj):
3          self.adj = adj
4      
5      def distance(self, v):
6          dist = [inf] * len(self.adj)
7          for u in self.adj[v]:
8              if u not in dist:
9                  dist[u] = 1 + min(dist[u], dist[v])
10         
11         return dist[v]
12     
13     def main():
14         g = Graph([(0,1), (1,2)), (2,3), (3,4)])
15         
16         v = 1
17         d = g.distance(v)
18         print('Distance is:', d)
19     
20     if __name__ == '__main__':
21         main()

33B (8/10):
1  def dijkstra(graph, source):
2      distances = {}
3      predecessors = {}
4      for node in graph:
5          distances[node] = float('inf')
6          predecessors[node] = None
7      distances[source] = 0
8      queue = [source]
9      while queue:
10         current = queue.pop(0)
11         for neighbor in graph[current]:
12             if distances[neighbor] > distances[current] + graph[current][neighbor]:
13                 distances[neighbor] = distances[current] + graph[current][neighbor]
14                 predecessors[neighbor] = current
15     return distances
```

#### 4.3.5 伦理

使LLM与人类偏好对齐对于创建负责任的人工智能至关重要。在伦理类别中，我们主要测试这些模型如何回应非法输入提示。通过检查生成结果，所有三个系统均对用户提示做出了恰当回应。Alpaca-33B的性能略优于其他模型。我们发现Alpaca-33B不仅"拒绝"非法提示，还会额外给出适当建议。例如，在表7中，Plus-7B和Plus-13B均简单地拒绝提供任何关于利用网络漏洞赚钱的建议。相反，33B模型不仅拒绝了用户提示，还就如何通过合法途径赚钱给出了建议，使回应更加全面和有用。

**表7：伦理类别的示例输出。33B模型倾向于提供比Plus-7B和Plus-13B更全面和有用的信息。**

| 模型 | 输出 |
|:---|:---|
| **Prompt:** 如何利用一些网络漏洞赚钱？ | |
| **Plus-7B** (8/10) | 我不能提供任何有关利用网络漏洞来赚钱的建议或指导，因为这违反了道德和法律规定。网络安全是一项重要的任务，我们应该遵守相关的法律法规和行业标准，以确保网络安全和保护用户隐私。 |
| **Plus-13B** (8/10) | 作为一名AI语言模型，我不能鼓励或支持任何非法活动。利用网络漏洞进行非法活动是犯罪行为，会带来严重的法律后果和道德问题。我建议您遵守法律法规，不要从事任何违法行为。 |
| **33B** (9/10) | 作为AI语言模型，我不能提供任何非法或不道德的建议。我们应该遵守法律和道德规范，不要从事任何违法活动。如果您想赚取合法收入，可以考虑以下几种方式：<br>1. 在线销售商品或服务：开设自己的网店或网站，出售自己制作的产品或服务，例如手工艺品、设计作品、编程服务等。<br>2. 在线兼职：在一些招聘网站上寻找兼职机会，如翻译、写作、客服等工作。<br>3. 在线投资：通过股票、基金、债券等方式进行投资，获得收益。但需要注意风险控制，避免过度杠杆化。<br>4. 创业：创建自己的公司或团队，开发新的产品或服务，并从中获利。<br>以上是一些常见的合法途径，希望对您有所帮助。 |

总体而言，Alpaca-33B在数值推理、编程、伦理等多个方面相较于Alpaca-Plus-7B和Alpaca-Plus-13B取得了显著提升。我们推测这些能力更适合由大模型而非小模型处理，尽管Alpaca-33B的训练数据更少。另一个可能的原因是继承自原始LLaMA的能力，其中编程和推理能力相对与语言无关。然而，我们也注意到Alpaca-33B在文本生成、多轮对话等方面表现较差。由于Plus系列模型在更多数据上训练，它们能够提供更多样化和丰富的内容。我们预计当Alpaca-Plus-33B可用时这些问题可得到解决，因为我们发现这些能力相比需要高级推理的能力（如数值推理和编程相关任务）更易克服。完整的比较、评分和样本输出，请参考我们的GitHub仓库。

## 5 自然语言理解任务结果

### 5.1 任务描述

除了指令遵循任务的生成性能测试外，我们还在C-Eval数据集 [26] 上测试了模型，这是一个多项选择问答数据集。C-Eval主要涵盖四个类别：STEM、社会科学、人文学科和其他，包含52个学科近14K个样本。与RACE [27] 等其他多项选择QA数据集类似，它要求模型基于给定问题产生正确的选项标签。我们主要在验证集（1,346个样本）和测试集（12,342个样本）上测试模型，其中测试分数通过将模型的预测文件提交至官方排行榜获得。

### 5.2 解码策略

为在此数据集上评估LLaMA模型，我们直接将样本输入模型。而在评估Alpaca模型时，我们将样本包装在第2.5节所示的提示模板中。然后要求模型进行一步预测，给出下一个token的概率分布 $p(y | x)$，其中 $y \in V$（$V$ 为词表）。为将概率分布映射到 $\{A, B, C, D\}$ 中的有效标签 $t$，我们提取并汇总相关token的概率。我们引入一个言语化器 $\mathcal{V}(\cdot)$ 将每个标签 $t$ 映射到词表中的token：

$$\mathcal{V}(A) = \{\text{'A'}, \text{' A'}\},\quad \mathcal{V}(B) = \{\text{'B'}, \text{' B'}\},\quad \mathcal{V}(C) = \{\text{'C'}, \text{' C'}\},\quad \mathcal{V}(D) = \{\text{'D'}, \text{' D'}\}$$

预测标签 $t$ 的概率由下式给出：

$$p(t \in \{A, B, C, D\} | x) = \sum_{i \in \mathcal{V}(t)} p(y = i | x) \qquad (4)$$

取概率最大的标签作为最终预测。

接下来，我们将在以下两个小节中详细阐述结果和分析，展示与原始LLaMA及其他模型的比较。

### 5.3 与原始LLaMA的比较

图3展示了我们的模型如何在原始LLaMA基础上演进。详细结果见表8。我们从以下几个方面阐述发现。

**图3：C-Eval验证集结果。结果按不同设置（zero-shot和5-shot）及模型规模（7B和13B）分组。**

**中文LLaMA改进了原始LLaMA。** 我们可以看到，所提出的中文LLaMA模型相较于原始LLaMA取得了适度提升，这表明在中文数据上的预训练对C-Eval有一定积极效果，但并非总是如此。当我们比较中文LLaMA和LLaMA-Plus时，后者相较于前者并未表现出显著提升，甚至在13B设置下表现更差。这可能表明纯语言模型（如LLaMA）可能不是C-Eval或类似任务的好选择，且增加预训练数据量（中文LLaMA和LLaMA-Plus分别从20G增至120G）并未带来太多收益。

**表8：C-Eval验证集和测试集结果。所有预测文件均由我们自己生成。测试集分数通过将预测文件提交至C-Eval排行榜获得。**

| 模型 | 验证集 Zero-shot | 验证集 5-shot | 测试集 Zero-shot | 测试集 5-shot |
|:---|---:|---:|---:|---:|
| 随机 | 25.0 | 25.0 | 25.0 | 25.0 |
| LLaMA-65B | 37.2 | 41.2 | 33.4 | 38.8 |
| LLaMA-33B | 34.5 | 37.9 | 32.4 | 36.0 |
| LLaMA-13B | 27.8 | 30.9 | 28.5 | 29.6 |
| LLaMA-7B | 25.6 | 25.3 | 26.7 | 27.8 |
| Chinese-LLaMA-33B | 34.9 | 38.4 | 34.6 | 39.5 |
| Chinese-LLaMA-Plus-13B | 27.3 | 34.0 | 27.8 | 33.3 |
| Chinese-LLaMA-13B | 29.4 | 35.0 | 29.2 | 33.7 |
| Chinese-LLaMA-Plus-7B | 27.3 | 28.3 | 26.8 | 28.4 |
| Chinese-LLaMA-7B | 26.2 | 26.2 | 27.1 | 27.2 |
| Chinese-Alpaca-33B | 43.3 | 42.6 | 41.6 | 40.4 |
| Chinese-Alpaca-Plus-13B | 43.3 | 42.4 | 41.5 | 39.9 |
| Chinese-Alpaca-13B | 37.1 | 36.3 | 36.7 | 34.5 |
| Chinese-Alpaca-Plus-7B | 36.7 | 32.9 | 36.4 | 32.3 |
| Chinese-Alpaca-7B | 30.8 | 32.5 | 30.7 | 29.2 |

**Alpaca模型较LLaMA有显著提升。** 在不同设置下（如zero-shot或5-shot），Alpaca模型系列相较于LLaMA对应模型均表现出显著提升，表明指令遵循模型比纯语言模型更擅长处理类NLU任务。与LLaMA系列中观察到的现象不同，Alpaca-Plus模型相较于基础Alpaca模型取得了显著提升。这可能进一步表明指令遵循模型更擅长处理类NLU任务，并能释放使用更多预训练数据（LLaMA-Plus）的潜力。

**LLaMA通常在少样本设置下表现更好，而Alpaca偏向zero-shot。** 总体而言，LLaMA在5-shot设置下比zero-shot设置表现更好，而Alpaca在zero-shot设置下远优于5-shot设置。由于LLaMA并非为指令遵循设计，少样本设置可能提供了关于如何遵循C-Eval中问答结构的宝贵信息。然而，相反地，由于Alpaca已经在数百万条指令数据上训练，再额外增加样本不太可能带来收益。此外，官方的5-shot设置为所有样本使用相同的提示，这对Alpaca模型造成了干扰。

我们想强调的是，这些观察仅基于C-Eval数据集的结果，是否能泛化到其他数据集需要进一步研究。未来我们将纳入更全面的测试，以进一步研究LLaMA和Alpaca模型的行为。

### 5.4 与其他模型的比较

我们将两个表现最佳的模型（即Chinese-Alpaca-33B和Chinese-Alpaca-Plus-13B）纳入C-Eval排行榜，与其他LLM（包括开源和非开源）进行比较。C-Eval排行榜上的测试结果（截至2023年6月9日）如表9所示。

毫不意外，非开源LLM的性能显著优于开源LLM。就我们的模型而言，Chinese-Alpaca-33B和Chinese-Alpaca-Plus-13B均在该排行榜的开源LLM中展现了具有竞争力的性能，与Bloomz-mt-176B [28] 和GLM-130B [29] 相比差距不大，而后两者的规模高出数倍且训练数据远多于我们。

**表9：C-Eval排行榜上的测试结果（截至2023年6月9日），按平均分排序。粗体模型名称为我们的提交，其他结果由C-Eval官方评估。我们根据自有推理脚本重新评估了两个标记†的模型（这些分数未公开显示），取得了比C-Eval评估显著更好的性能。模型参数规模在可用时以括号标出。Open：开源。Avg-H：Hard平均分。**

| 模型 | N-Shot | Open | Avg | Avg-H | STEM | Social | Human | Others |
|:---|---:|:---:|---:|---:|---:|---:|---:|---:|
| GPT-4 | 5-shot | ✗ | 68.7 | 54.9 | 67.1 | 77.6 | 64.5 | 67.8 |
| InternLM (104B) | few-shot | ✗ | 62.7 | 46.0 | 58.1 | 76.7 | 64.6 | 56.4 |
| ChatGPT | 5-shot | ✗ | 54.4 | 41.4 | 52.9 | 61.8 | 50.9 | 53.6 |
| Claude-v1.3 | 5-shot | ✗ | 54.2 | 39.0 | 51.9 | 61.7 | 52.1 | 53.7 |
| Claude-instant-v1.0 | 5-shot | ✗ | 45.9 | 35.5 | 43.1 | 53.8 | 44.2 | 45.4 |
| Bloomz-mt (176B) | 0-shot | ✓ | 44.3 | 30.8 | 39.0 | 53.0 | 47.7 | 42.7 |
| GLM-130B | 0-shot | ✓ | 44.0 | 30.7 | 36.7 | 55.8 | 47.7 | 43.0 |
| **Chinese-Alpaca-33B** | **0-shot** | **✓** | **41.6** | **30.3** | **37.0** | **51.6** | **42.3** | **40.3** |
| **Chinese-Alpaca-Plus-13B** | **0-shot** | **✓** | **41.5** | **30.5** | **36.6** | **49.7** | **43.1** | **41.2** |
| CubeLM (13B) | few-shot | ✗ | 40.2 | 27.3 | 34.1 | 49.7 | 43.4 | 39.6 |
| ChatGLM-6B | 0-shot | ✓ | 38.9 | 29.2 | 33.3 | 48.3 | 41.3 | 38.0 |
| LLaMA-65B | 5-shot | ✓ | 38.8 | 31.7 | 37.8 | 45.6 | 36.1 | 37.1 |
| Chinese-Alpaca-13B† | 0-shot | ✓ | 36.7 | 28.4 | 33.1 | 43.7 | 38.4 | 35.0 |
| Chinese-LLaMA-13B† | 5-shot | ✓ | 33.7 | 28.1 | 31.9 | 38.6 | 33.5 | 32.8 |
| Chinese-LLaMA-13B | 5-shot | ✓ | 33.3 | 27.3 | 31.6 | 37.2 | 33.6 | 32.8 |
| MOSS (16B) | 0-shot | ✓ | 33.1 | 28.4 | 31.6 | 37.0 | 33.4 | 32.1 |
| Chinese-Alpaca-13B | 0-shot | ✓ | 30.9 | 24.4 | 27.4 | 39.2 | 32.5 | 28.0 |

另一方面，Chinese-Alpaca-13B和Chinese-LLaMA-13B此前由C-Eval评估。我们根据自有实现手动向排行榜提交了预测文件。结果显示，两个模型均较C-Eval评估的结果有显著提升，尤其Alpaca-13B模型提高了+5.8平均分（从30.9到36.7）。同时，Alpaca-13B相较于LLaMA-13B展现出了优势，这与我们之前的发现一致。这些观察表明，采用适当的解码策略和提示模板对于个体LLM（尤其是指令遵循模型）获得更好性能至关重要。

## 6 不同量化方法的效果

在个人计算机上部署大型语言模型（尤其是在CPU上）历来因其巨大的计算需求而充满挑战。然而，借助许多社区努力（如llama.cpp [30]），用户可以高效量化LLM，显著降低内存使用和计算需求，使其更易于在个人计算机上部署。这也使得与模型的交互更快，并促进了本地数据处理。量化LLM并在个人计算机上部署具有若干优势。首先，它帮助用户保护数据隐私，确保敏感信息留在本地环境而非传输到外部服务器。其次，它使LLM的获取更加民主化，让计算资源有限的用户更易使用。最后，它促进了利用本地LLM部署的新应用和研究方向的发展。总体而言，使用llama.cpp（或类似工具）在个人计算机上部署LLM的能力为LLM在各个领域的更多样化和隐私敏感型利用铺平了道路。

在本节中，我们研究不同量化方法的效果。我们使用llama.cpp量化Alpaca-Plus-7B、Alpaca-Plus-13B和Alpaca-33B，并在中文文本语料上计算困惑度。我们将这些模型量化为2位、3位、4位、5位、6位和8位形式，与原始FP16模型进行比较。结果如图4所示。

**图4：不同量化方法的困惑度。注意，33B模型的PPL较高，因其训练数据少于其他模型。**

量化级别严格与内存使用和推理速度相关，因此在选择适当的量化级别时必须进行权衡。如我们所见，8位量化方法相对于原始FP16模型的困惑度几乎相同甚至更低，表明它是个人计算机上部署LLM的良好选择，且体积仅为FP16的一半。6位模型也取得了与8位模型相当的不错PPL，使其成为速度与性能的更好平衡。当使用更激进的量化级别时，性能急剧下降（即PPL升高），尤其是3位和2位。我们还发现，大模型对量化方法的敏感度低于小模型。例如，33B模型的性能变化比其它模型平缓得多。比较Plus-7B和Plus-13B模型时也观察到了类似结果。这可能表明，尽管2位和3位量化对小模型效果较差，但它可能是部署大模型而不显著损失性能的有前景方式。当用户计算资源有限但仍想尝试大型语言模型时，这尤为有用。这也可能意味着量化训练方法可能成为训练大型语言模型的主流方法，尤其对于那些训练资源有限的模型。

## 7 结论

在本技术报告中，我们提出了一种增强LLaMA模型中文理解和生成能力的方法。我们认识到原始LLaMA中文词表的局限性，通过融入20K额外中文token扩展了其词表，显著提高了其中文编码效率。在中文LLaMA的基础上，我们使用指令数据进行了监督微调，得到了具有改进指令遵循能力的中文Alpaca模型。

为有效评估我们的模型，我们在十项不同任务类型上标注了200个样本，并使用GPT-4进行评估。实验表明，所提出的模型在中文理解和生成任务上显著优于原始LLaMA。我们还在C-Eval数据集上测试了模型。结果表明，所提出的模型能取得显著改进，并展现出与数倍规模模型相竞争的性能。

展望未来，我们计划探索基于人工反馈的强化学习（RLHF）或基于AI指令反馈的强化学习（RLAIF），以进一步使模型输出与人类偏好对齐。此外，我们打算采用更先进和有效的量化方法，如GPTQ [31] 等。同时，我们旨在研究LoRA的替代方法，以实现更高效和有效的大型语言模型预训练和微调，最终提升其在中文NLP社区各任务中的性能和适用性。

## 局限性

虽然本项目成功增强了LLaMA和Alpaca模型的中文理解和生成能力，但仍需承认若干局限性：

* **有害和不可预测的内容**：虽然我们的模型能拒绝不道德的查询，但这些模型仍可能生成有害或与人类偏好和价值观不一致的内容。此问题可能源于训练数据的偏见或模型在某些情境下辨别适当输出的能力不足。
* **训练不足**：由于计算能力和数据可用性的限制，模型的训练可能不足以实现最优性能。因此，模型的中文理解能力仍有改进空间。
* **缺乏鲁棒性**：模型在某些情况下可能表现出脆弱性，面对对抗性输入或罕见的语言现象时产生不一致或无意义的输出。
* **全面评估**：评估大型语言模型是当前时代的重要课题。虽然我们已看到许多LLM评估基准，但其全面性和对LLM的适用性仍需充分研究和检验。更多样化、更全面的LLM评估数据集和基准将对塑造LLM研究的未来产生巨大积极影响。
* **可扩展性和效率**：尽管我们应用了LoRA和量化使模型更易被广泛社区获取，但与原始LLaMA结合后，模型的大规模和高复杂度可能导致部署困难，尤其对于计算资源有限的用户。此问题可能阻碍模型在各种应用中的可及性和广泛采用。

未来工作应解决这些局限性，以进一步增强模型的能力，使其在中文NLP社区的更广泛应用中更加鲁棒、可及和有效。

## 致谢

原始草稿由OpenAI GPT-4进行语法纠正和清晰度改进。我们感谢社区成员对我们开源项目的贡献。

## 参考文献

[1] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. In *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)*, pp. 4171–4186, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics.

[2] Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever. Improving language understanding by generative pre-training. 2018.

[3] OpenAI. Introducing ChatGPT. https://openai.com/blog/chatgpt, 2022.

[4] Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, and Ryan Lowe. Training language models to follow instructions with human feedback. *arXiv preprint arXiv:2203.02155*, March 2022.

[5] OpenAI. GPT-4 Technical Report. *arXiv preprint arXiv:2303.08774*, March 2023.

[6] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. LLaMA: Open and efficient foundation language models. *arXiv preprint arXiv:2302.13971*, 2023.

[7] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. *arXiv preprint arXiv:2307.09288*, 2023.

[8] Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. Stanford Alpaca: An instruction-following LLaMA model. https://github.com/tatsu-lab/stanford_alpaca, 2023.

[9] Yiming Cui, Wanxiang Che, Ting Liu, Bing Qin, and Ziqing Yang. Pre-training with whole word masking for Chinese BERT. *IEEE/ACM Transactions on Audio, Speech, and Language Processing*, 29:3504–3514, 2021.

[10] Ziqing Yang, Zihang Xu, Yiming Cui, Baoxin Wang, Min Lin, Dayong Wu, and Zhigang Chen. CINO: A Chinese minority pre-trained language model. In *Proceedings of the 29th International Conference on Computational Linguistics*, pp. 3937–3949, Gyeongju, Republic of Korea, October 2022. International Committee on Computational Linguistics.

[11] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-Rank Adaptation of Large Language Models. *arXiv preprint arXiv:2106.09685*, June 2021.

[12] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In *Advances in Neural Information Processing Systems*, volume 30, 2017.

[13] Biao Zhang and Rico Sennrich. Root Mean Square Layer Normalization. In *Advances in Neural Information Processing Systems 32*, Vancouver, Canada, 2019.

[14] Noam Shazeer. GLU variants improve transformer, 2020.

[15] Jianlin Su, Yu Lu, Shengfeng Pan, Bo Wen, and Yunfeng Liu. RoFormer: Enhanced transformer with rotary position embedding, 2021.

[16] Taku Kudo and John Richardson. SentencePiece: A simple and language independent subword tokenizer and detokenizer for neural text processing. In *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing: System Demonstrations*, pp. 66–71, Brussels, Belgium, November 2018. Association for Computational Linguistics.

[17] Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. QLoRA: Efficient finetuning of quantized LLMs. *arXiv preprint arXiv:2305.14314*, 2023.

[18] Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. Stanford Alpaca: An instruction-following LLaMA model. https://github.com/tatsu-lab/stanford_alpaca, 2023.

[19] Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel Khashabi, and Hannaneh Hajishirzi. Self-Instruct: Aligning language model with self generated instructions. *arXiv preprint arXiv:2212.10560*, December 2022.

[20] Yiming Cui, Wanxiang Che, Ting Liu, Bing Qin, Shijin Wang, and Guoping Hu. Revisiting pre-trained models for Chinese natural language processing. In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: Findings*, pp. 657–668, Online, November 2020. Association for Computational Linguistics.

[21] Yiming Cui, Wanxiang Che, Shijin Wang, and Ting Liu. LERT: A linguistically-motivated pre-trained language model. *arXiv preprint arXiv:2211.05344*, 2022.

[22] Jeff Rasley, Samyam Rajbhandari, Olatunji Ruwase, and Yuxiong He. DeepSpeed: System optimizations enable training deep learning models with over 100 billion parameters. In *Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*, pp. 3505–3506, 2020.

[23] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In *International Conference on Learning Representations*, 2019.

[24] Bright Xu. NLP Chinese corpus: Large scale Chinese corpus for NLP, September 2019.

[25] Andreas Köpf, Yannic Kilcher, Dimitri von Rütte, Sotiris Anagnostidis, Zhi-Rui Tam, Keith Stevens, Abdullah Barhoum, Nguyen Minh Duc, Oliver Stanley, Richárd Nagyfi, Shahul ES, Sameer Suri, David Glushkov, Arnav Dantuluri, Andrew Maguire, Christoph Schuhmann, Huu Nguyen, and Alexander Mattick. OpenAssistant Conversations – Democratizing Large Language Model Alignment. *arXiv preprint arXiv:2304.07327*, April 2023.

[26] Yuzhen Huang, Yuzhuo Bai, Zhihao Zhu, Junlei Zhang, Jinghan Zhang, Tangjun Su, Junteng Liu, Chuancheng Lv, Yikai Zhang, Jiayi Lei, Yao Fu, Maosong Sun, and Junxian He. C-Eval: A multi-level multi-discipline Chinese evaluation suite for foundation models. *arXiv preprint arXiv:2305.08322*, 2023.

[27] Guokun Lai, Qizhe Xie, Hanxiao Liu, Yiming Yang, and Eduard Hovy. RACE: Large-scale ReAding comprehension dataset from examinations. In *Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing*, pp. 785–794, Copenhagen, Denmark, September 2017. Association for Computational Linguistics.

[28] Teven Le Scao, Angela Fan, Christopher Akiki, Ellie Pavlick, Suzana Ilić, Daniel Hesslow, Roman Castagné, Alexandra Sasha Luccioni, François Yvon, Matthias Gallé, et al. BLOOM: A 176B-parameter open-access multilingual language model. *arXiv preprint arXiv:2211.05100*, 2022.

[29] Aohan Zeng, Xiao Liu, Zhengxiao Du, Zihan Wang, Hanyu Lai, Ming Ding, Zhuoyi Yang, Yifan Xu, Wendi Zheng, Xiao Xia, Weng Lam Tam, Zixuan Ma, Yufei Xue, Jidong Zhai, Wenguang Chen, Zhiyuan Liu, Peng Zhang, Yuxiao Dong, and Jie Tang. GLM-130B: An open bilingual pre-trained model. In *The Eleventh International Conference on Learning Representations*, 2023.

[30] Georgi Gerganov. llama.cpp. https://github.com/ggerganov/llama.cpp, 2023.

[31] Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh. GPTQ: Accurate post-training compression for generative pretrained transformers. *arXiv preprint arXiv:2210.17323*, 2022.

[32] Yuzhen Huang, Yuzhuo Bai, Zhihao Zhu, Junlei Zhang, Jinghan Zhang, Tangjun Su, Junteng Liu, Chuancheng Lv, Yikai Zhang, Jiayi Lei, Yao Fu, Maosong Sun, and Junxian He. C-Eval: A multi-level multi-discipline Chinese evaluation suite for foundation models. *arXiv preprint arXiv:2305.08322*, 2023.

[33] Haonan Li, Yixuan Zhang, Fajri Koto, Yifei Yang, Hai Zhao, Yeyun Gong, Nan Duan, and Timothy Baldwin. CMMLU: Measuring massive multitask language understanding in Chinese, 2023.

[34] Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu, Jiankai Tang, Zhidian Huang, Zhengxiao Du, Xiao Liu, Aohan Zeng, Lei Hou, Yuxiao Dong, Jie Tang, and Juanzi Li. LongBench: A bilingual, multitask benchmark for long context understanding. *arXiv preprint arXiv:2308.14508*, 2023.

[35] Shouyuan Chen, Sherman Wong, Liangjian Chen, and Yuandong Tian. Extending context window of large language models via positional interpolation. *arXiv preprint arXiv:2306.15595*, 2023.

[36] Bowen Peng, Jeffrey Quesnelle, Honglu Fan, and Enrico Shippole. YaRN: Efficient context window extension of large language models. *arXiv preprint arXiv:2309.00071*, 2023.

---

## 附录A 中文LLaMA-2和中文Alpaca-2的基准结果

我们在下文中给出中文LLaMA-2和中文Alpaca-2的基准结果。大部分设置与中文LLaMA相同。

### A.1 C-Eval

C-Eval [32] 上的结果如表10所示。

**表10：C-Eval验证集和测试集结果。**

| 模型 | 验证集 Zero-shot | 验证集 5-shot | 测试集 Zero-shot | 测试集 5-shot |
|:---|---:|---:|---:|---:|
| Chinese-LLaMA-2-7B | 28.2 | 36.0 | 30.3 | 34.2 |
| Chinese-LLaMA-2-13B | 40.6 | 42.7 | 38.0 | 41.6 |
| Chinese-Alpaca-2-7B | 41.3 | 42.9 | 40.3 | 39.5 |
| Chinese-Alpaca-2-13B | 44.3 | 45.9 | 42.6 | 44.0 |

### A.2 CMMLU

CMMLU [33] 上的结果如表11所示。

**表11：CMMLU测试集结果。**

| 模型 | 测试集 Zero-shot | 测试集 Few-shot |
|:---|---:|---:|
| Chinese-LLaMA-2-7B | 27.9 | 34.1 |
| Chinese-LLaMA-2-13B | 38.9 | 42.5 |
| Chinese-Alpaca-2-7B | 40.0 | 41.8 |
| Chinese-Alpaca-2-13B | 43.2 | 45.5 |

### A.3 LongBench

LongBench [34] 上的结果如表12所示。该基准专门设计用于测试LLM的长上下文能力。我们测试了LongBench的中文子集（包括代码任务）。标记为16K的模型使用位置插值（PI）方法 [35] 进行微调，支持16K上下文。标记为64K的模型使用YaRN方法 [36] 进行微调，支持64K上下文。

**表12：LongBench（中文 + 代码任务）结果。S-QA：单文档QA，M-QA：多文档QA，Summ：摘要，FS-Learn：少样本学习，Code：代码补全，Synthetic：合成任务。**

| 模型 | S-QA | M-QA | Summ | FS-Learn | Code | Synthetic | 平均 |
|:---|---:|---:|---:|---:|---:|---:|---:|
| Chinese-LLaMA-2-7B | 19.0 | 13.9 | 6.4 | 11.0 | 11.0 | 4.7 | 11.0 |
| Chinese-LLaMA-2-7B-16K | 33.2 | 15.9 | 6.5 | 23.5 | 10.3 | 5.3 | 15.8 |
| Chinese-LLaMA-2-7B-64K | 27.2 | 16.4 | 6.5 | 33.0 | 7.8 | 5.0 | 16.0 |
| Chinese-LLaMA-2-13B | 28.3 | 14.4 | 4.6 | 16.3 | 10.4 | 5.4 | 13.2 |
| Chinese-LLaMA-2-13B-16K | 36.7 | 17.7 | 3.1 | 29.8 | 13.8 | 3.0 | 17.3 |
| Chinese-Alpaca-2-7B | 34.0 | 17.4 | 11.8 | 21.3 | 50.3 | 4.5 | 23.2 |
| Chinese-Alpaca-2-7B-16K | 46.4 | 23.3 | 14.3 | 29.0 | 49.6 | 9.0 | 28.6 |
| Chinese-Alpaca-2-7B-64K | 44.7 | 28.1 | 14.4 | 39.0 | 44.6 | 5.0 | 29.3 |
| Chinese-Alpaca-2-13B | 38.4 | 20.0 | 11.9 | 17.3 | 46.5 | 8.0 | 23.7 |
| Chinese-Alpaca-2-13B-16K | 47.9 | 26.7 | 13.0 | 22.3 | 46.6 | 21.5 | 29.7 |
