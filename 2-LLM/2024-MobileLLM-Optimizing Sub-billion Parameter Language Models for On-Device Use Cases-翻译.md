# MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases

> Zechun Liu, Changsheng Zhao, Forrest Iandola, Chen Lai, Yuandong Tian, Igor Fedorov, Yunyang Xiong, Ernie Chang, Yangyang Shi, Raghuraman Krishnamoorthi, Liangzhen Lai, Vikas Chandra | Meta



本文聚焦于设计**低于十亿参数的高质量LLM**用于移动部署，核心发现是——**架构比数据量更重要，深而窄才是王道**。

核心内容：

- 移动设备内存有限（iPhone DRAM 6GB），一个app不能超过10%，7B模型光解码就耗电0.7J/token，亚十亿参数是移动端的务实选择
- 主流观点认为 数据量 和 参数量决定模型质量，但本文证明 **模型架构才是关键**：深而窄结构 + 嵌入共享 + 分组查询注意力 → MobileLLM 基线
- 提出**即时块级权重共享**：相邻层共享权重，不增加模型大小，仅增加极小延迟 → MobileLLM-LS
- 在对话和API调用等端侧任务上验证，小模型也能有大能力

关键发现：

- MobileLLM 在125M/350M规模上，零样本常识推理比前SOTA分别提升 **+2.7%/+4.3%**
- MobileLLM-LS 进一步提升 **+0.7%/+0.8%**，且无额外内存开销
- 对话任务上，MobileLLM-LS-350M 胜率达48.2%，接近GPT-3基线（50%）
- API调用任务上，MobileLLM-350M 精确匹配得分**接近 LLaMA-v2 7B**

---



## 摘要

本文解决了移动设备上高效大型语言模型（LLM）日益增长的需求，其驱动力来自不断上升的云端成本和延迟问题。我们专注于设计参数少于十亿的顶级质量LLM，这是移动部署的务实选择。**与强调数据和参数量在决定模型质量中起关键作用的流行观点相反，我们的研究强调了模型架构对亚十亿规模LLM的重要性**。通过利用深而窄的架构，结合嵌入共享和分组查询注意力机制，我们建立了一个强基线网络，称为MobileLLM，该网络在先前125M/350M最优模型基础上分别实现了2.7%/4.3%的显著准确率提升。此外，我们提出了一种即时块级权重共享方法，在不增加模型大小且仅带来微小延迟开销的情况下，进一步提升性能。得到的模型称为MobileLLM-LS，相比MobileLLM 125M/350M分别进一步提升了0.7%/0.8%的准确率。此外，MobileLLM模型家族在对话基准测试上相比先前亚十亿模型显示出显著改进，并在API调用任务中展现出与LLaMA-v2 7B接近的正确率，突显了**小模型在常见端侧用例中的能力**。

---



## 1. 引言

大型语言模型（LLM）正渗透到人类生活的各个方面，不仅影响着人们的沟通和工作方式，也在塑造日常娱乐体验。**当代LLM产品的典型例子，如ChatGPT和Perplexity AI，主要在云端环境中运行**。像ChatGPT4这样的领先模型参数超过1万亿[1]。然而，设想一个未来场景，人类在前后端都广泛依赖LLM——无论是前端对话界面还是**后端如推荐系统等操作**——相当于个人每日时间的约5%。在这种假设场景下，以50 tokens/s的处理速率使用GPT-4，需要部署大约一亿块H100 GPU[2]（每块可提供60 TFLOPs/s[3]）。这种计算规模，不包括通信和数据传输开销，相当于160个Meta规模的公司[4]。随之而来的**能源消耗和二氧化碳排放将带来惊人的环境挑战**。因此，我们必须对LLM进行小型化。

**此外，便携性和计算成本的考虑也推动着将LLM部署到智能手机和移动设备上。**在当前移动技术格局中，集成像LLaMA-v2 7B [5]这样具有8比特权重的LLM，由于主内存（DRAM）容量的限制而成本过高。图2展示了移动设备中常见的内存层次结构。由于iPhone 15的DRAM容量为6 GB，Google Pixel 8 Pro [6, 7]为12 GB，一个移动应用不应超过DRAM的10%，因为DRAM需要与操作系统和其他应用程序共享[8]。这推动了部署亚十亿参数LLM的需求。此外，考虑到LLM的能耗（每十亿模型参数每token 0.1 J [9, 8]），一个7B参数的LLM消耗0.7 J/token。**一个充满电的iPhone大约有50kJ的能量**，以10 tokens/s的速率只能维持该模型对话不到2小时，每64个token消耗0.2%的电量。这些需求汇聚成一个必然要求：采用紧凑模型进行端侧执行。通过使用亚十亿模型，例如一个350M 8-bit模型仅消耗0.035 J/token，一部iPhone可以支持一整天的对话使用。此外，解码速度可以显著提高，例如125M模型的基准测试结果可以达到50 tokens/s，而使用LLaMA 7B模型的最先进iPhone App MLC Chat仅为3~6 tokens/s[10]。

<img src="/Users/dazhang/PycharmProject/Papers/2-LLM/.picture/image-20260812172158672.png" alt="image-20260812172158672" style="zoom: 33%;" />

图1：

<img src="/Users/dazhang/PycharmProject/Papers/2-LLM/.picture/image-20260812175811093.png" alt="image-20260812175811093" style="zoom: 33%;" />

图

基于这些考虑，本文致力于设计和实现参数**少于10亿的LLM**。

我们做出了以下贡献，构建了迄今为止参数少于10亿的最准确LLM[11]：

*   **与缩放定律[12]相反，我们证明对于小型LLM，深度比宽度更重要。深而窄的模型结构在捕捉抽象概念方面表现出色，从而带来更优的最终性能。**
*   我们重新审视了小型LLM中的嵌入共享方法[13]，并实现了**分组查询注意力**[14]，以**最大化权重利用率**。
*   我们提出了**即时块级权重共享**。在内存移动成为延迟瓶颈的场景中，两个相邻块之间的权重共享避免了权重移动，只需计算该块两次，带来极小的延迟开销。
*   我们提出了一个新的模型家族MobileLLM，展示了最先进的性能。在一组零样本任务中，MobileLLM比之前的125M/350M最优模型分别高出2.7%/4.3%。
*   在下游任务中，如对话和API调用，MobileLLM模型家族显著优于同等规模的模型。在API调用任务中，MobileLLM-350M甚至达到了与更大规模LLaMA-v2 7B模型相当的精确匹配得分。
*   我们进一步证明了我们的设计理念可以有效地扩展到更大的模型，MobileLLM-600M/1B/1.5B的结果详见附录A。



## 2. 改进亚十亿规模LLM设计

在本节中，我们展示了从 基线亚十亿参数模型 到 新的最优模型（图3）的演进路径。我们同时探索了125M和350M模型，并在两种情况下都展示了一致的改进。对于**模型大小是主要约束的端侧用例**，如何有效分配有限的权重参数变得比以往任何时候都更加关键。我们首先通过测试四种对亚十亿规模LLM有益的模型设计技术，建立了一个强基线模型MobileLLM，包括：

(1) 采用SwiGLU FFN [15]；

(2) 采用深而窄的架构；

(3) 重新审视嵌入共享方法[13]；

(4) 利用分组查询注意力[16]

然后，我们开发了一种 **即时块级层共享方法**，在不引入任何额外内存开销且仅带来微小延迟开销的情况下，进一步提升了准确率。我们将带有**层共享**的模型称为MobileLLM-LS。

<img src="/Users/dazhang/PycharmProject/Papers/2-LLM/.picture/image-20260812175724437.png" alt="image-20260812175724437" style="zoom: 33%;" />

图3：

### 2.1. 训练设置

我们的实验在 **32块A100 GPU**上进行，每块GPU的批大小为32。我们使用 120k次 迭代在 0.25T token上进行了探索性实验。随后，表3和表4中报告的最佳模型在1T token上使用 480k次 迭代进行训练。

我们在零样本常识推理任务上评估预训练模型，包括ARC-easy、ARC-challenge [17]、BoolQ [18]、PIQA [19]、SIQA [20]、HellaSwag [21]、OBQA [22]、WinoGrande [23]，以及使用TQA [24]和RACE数据集[25]的**问答和阅读理解任务**。

### 2.2. 构建强基线

#### 2.2.1. 前馈网络选择

我们首先研究了前馈网络（FFN）中常用的激活函数，发现**最先进的SwiGLU** [15]对小型模型也有益。通过将标准FFN（FC $\to$ ReLU $\to$ FC）改为SwiGLU，125M模型在零样本推理任务上的平均性能从42.6提升到43.9。因此，我们在后续实验中使用FFN中的SwiGLU。

#### 2.2.2. 架构深度 vs 宽度

该领域的一个普遍信念[12]认为，**Transformer模型的性能主要取决于 参数数量、训练数据集的大小 和 训练迭代次数**。这种信念认为架构设计对Transformer模型性能的影响可以忽略不计。然而，我们的发现表明，**这对于较小的模型可能不成立**。

我们的实验结果，特别是对于模型容量有限的小模型，揭示了在性能提升方面，增加深度比增加宽度更为关键。我们进行了一项广泛研究，训练了19个模型，包括9个约125M参数的模型和10个约350M参数的模型。每个模型设计为相似的大小，但在深度和宽度上有所不同。我们在八个零样本常识推理任务以及问答和阅读理解基准上进行了实验。**我们的发现一致表明，更深更窄 的模型优于 更浅更宽 的模型**。图4 (a)和(b)展示了更深网络在大多数零样本推理任务（包括ARC-easy、ARC-challenge、PIQA、HellaSwag、OBQA、WinoGrande）上的优越性能。特别地，这种趋势在TQA和RACE数据集上更为明显，如图4 (c)-(f)所示。详细的模型配置和结果见附录。

我们的发现表明，对于约125M大小的Transformer模型，30层甚至42层的模型性能显著优于12层的模型。考虑到大多数先前125M模型[13, 26]的层数限制在12层，这一发现令人惊讶。

#### 2.2.3. 嵌入共享

在亚十亿规模的语言模型中，嵌入层占据了参数数量的很大一部分。例如，嵌入维度为512、词表大小为32k时，输入和输出嵌入层各包含1600万个参数。这些嵌入层合计占125M参数模型**总参数的20%以上**。相比之下，这一比例在更大的语言模型中要低得多。例如，输入和输出嵌入仅占LLaMA-7B模型[27]**总参数的3.7%**，在LLaMA-70B模型中**仅占0.7%**。这种差异可能解释了为什么 **嵌入共享 最初在OPT模型**[13]中被提出和实施，但在最近的LLM设计中被忽略了。

在开发亚十亿规模语言模型时，我们重新审视了 **输入-输出嵌入共享** 的概念。LLM模型中的输入嵌入将词表中的token ID映射到对应的token嵌入，维度为(vocab_size, embedding_dim)。相反，输出全连接层将嵌入维度映射回跨词表的logits预测，权重大小为(vocab_size, embedding_dim)。通过共享嵌入，我们将输入嵌入权重重用为输出全连接层权重，从而产生更高效、更紧凑的模型架构。

我们在一个30层125M模型上进行了实验。在表1中，我们展示了共享输入和输出嵌入减少了1600万个参数，约占总参数的11.8%，平均准确率仅下降0.2个点。**这种微小的准确率下降可以通过重新分配节省的参数来 增加更多层 来轻松恢复**。将深度增加到32层产生了0.4个点的准确率提升，同时仍比原始的135M模型少1000万个参数。在350M模型中也观察到了类似的结果。这些发现进一步表明，在有限的模型存储预算下，**嵌入共享是 最大化权重利用率 和 优化模型性能 的宝贵技术**。

#### 2.2.4. 头数和KV头数

我们现在研究小型Transformer模型的最优头大小。每个头维度中的更多语义与多个头的更多非线性组合之间的权衡是选择头大小时的关键考虑因素。此外，大多数先前研究在亚十亿参数语言模型中**通常使用与查询头相同数量的键值头**。相反，我们发现最初为减少LLM中键值缓存大小而设计的分组查询注意力[16, 14]，也可以有效减少小型LM中键值头的冗余。分组查询注意力可以视为权重重用的另一种形式，其中键值头的数量是查询头数量的 $1/n$，kv-heads在计算注意力分数和与查询一起生成输出时重复 $n$ 次。这里，$n \in Z⁺$ 表示一个正整数，查询头数量可被其整除。

> [!NOTE]
>
> 

为了为最先进的小型Transformer模型奠定坚实基础，我们在125M和350M模型上进行了实验，以确定理想的头大小。图5中的结果显示，使用16个查询头产生了最佳结果。此外，将kv-heads数量从16减少到4，在125M模型上产生了相当的准确率，在350M模型上仅下降了0.2个点的准确率，同时模型大小减少了近10%。这些结果为我们模型架构设计提供了指导。通过采用分组查询注意力（GQA）同时增加嵌入维度以维持模型大小，125M的准确率进一步提高了0.4个点，表明**GQA是进一步挖掘小模型潜力的有利方法**。


综上所述，我们测试了四种对小模型设计有益的最先进技术，包括采用SwiGLU的FFN、深而窄的架构、嵌入共享和分组查询注意力。结合这些技术，我们构建了一个强基线小型LLM，并将其命名为MobileLLM。

### 2.3. 层共享

第2.2.2节关于层深度与宽度影响的发现表明，更深的层对小型Transformer模型更有利。这促使我们研究层共享，将其作为在不增加模型存储成本的情况下增加隐藏层数量的一种策略。**这种方法在 模型大小 是主要约束的 端侧场景 中特别有用**。

令人惊讶的是，**实验结果表明，简单地复制Transformer块就可以实现准确率提升，而无需进行架构修改或扩大模型大小**。我们进一步研究了三种不同的权重共享策略，如图6所示。表2中的结果表明，在即时块级重复、整体重复和反向共享策略中，重复覆盖层共享策略产生了最佳性能。然而，考虑到硬件内存层次结构（图2），用于计算的SRAM通常限制在约20MB。这个容量通常只足以容纳单个Transformer块。因此，将共享权重放在缓存中并立即计算两次，可以避免在SRAM和DRAM之间传输权重，从而提高自回归推理的整体执行速度。因此，我们在模型设计中选择了**即时块级共享策略**。我们将带有层共享的模型称为MobileLLM-LS。

> [!NOTE]
>

表

![image-20260812172336770](/Users/dazhang/PycharmProject/Papers/2-LLM/.picture/image-20260812172336770.png)

表3：

![image-20260812172352789](/Users/dazhang/PycharmProject/Papers/2-LLM/.picture/image-20260812172352789.png)

表4：

<img src="/Users/dazhang/PycharmProject/Papers/2-LLM/.picture/image-20260812172408529.png" alt="image-20260812172408529" style="zoom: 50%;" />

表5：

<img src="/Users/dazhang/PycharmProject/Papers/2-LLM/.picture/image-20260812172429488.png" alt="image-20260812172429488" style="zoom:33%;" />

表6：

<img src="/Users/dazhang/PycharmProject/Papers/2-LLM/.picture/image-20260812172441248.png" alt="image-20260812172441248" style="zoom:50%;" />



## 3. 实验

### 3.1. 实验设置

我们从头开始训练MobileLLM，使用Adam优化器[28]，权重衰减为0.1。实验在32块A100 GPU上进行，每块GPU的批大小为32。初始学习率设置为2e-3，采用**余弦学习率衰减策略**。我们使用120k次迭代在0.25T token上进行快速探索实验，并使用480k次迭代在1T token上训练表3和表4中报告的最佳模型。

### 3.2. 主要结果

我们在零样本常识推理任务、问答和阅读理解任务上比较了最终性能。基线方法的结果使用其开源的Hugging Face模型进行评估，以确保评估程序的一致性。

**零样本常识推理** 表3展示了我们提出的模型MobileLLM与最先进的亚十亿参数模型之间的比较，包括早期开源LLM OPT [13]、BLOOM [29]，以及最近发布的Galactica [30]、Cerebras [31]、GPT-neo [26]以及LLM分析套件Pythia [32]和Transformer变体RWKV [33]在零样本常识推理任务上的表现。对于125M模型大小，MobileLLM以显著优势优于同模型大小的先前模型，如OPT、GPT-Neo和Galactica。此外，MobileLLM-125M比Pythia-160M和RWKV-169M分别高出3.8个点和2.7个点的准确率，同时模型大小分别小22%和26%。此外，在MobileLLM-LS-125M中引入层共享带来了额外0.7个点的准确率提升。值得注意的是，MobileLLM-LS-125M达到了与大多数先前350M模型相当甚至更高的结果。在350M模型大小类别中，MobileLLM以相当或更小的模型大小超越了先前最优模型超过4个点。为了在更广泛的内存约束下进一步验证我们的设计原则，我们将模型扩展到MobileLLM-600M、1B和1.5B配置。完整结果详见附录A。

**问答和阅读理解** 我们在TQA问答基准[24]和RACE阅读理解基准[25]上评估了预训练模型。我们遵循[27]的评估设置，并在表4中报告结果。比较125M大小的模型，MobileLLM-125M在TQA基准上相比其前身展示了超过4.3个点的显著改进。此外，MobileLLM-350M模型相比其他350M大小的模型表现出大约10个点的实质性性能提升。对于阅读理解任务，MobileLLM模型家族也显示出比先前亚十亿参数模型显著更高的分数。

### 3.3. 下游任务

为了验证亚十亿规模模型在端侧应用中的有效性，我们在两个关键的端侧任务上评估了其性能：对话 和 API调用。

#### 3.3.1. 对话

我们对MobileLLM模型以及从HuggingFace检查点获取的先前最优（SoTA）模型进行微调，用于对话任务，并在相同设置下进行评估以确保一致性。我们评估了两个基准：AlpacaEval [34]（单轮对话基准）和MT-Bench [35]（多轮对话基准）。表5中的结果显示，MobileLLM模型显著优于先前最优的亚十亿规模模型，甚至超过了拥有10亿参数的模型。值得注意的是，MobileLLM-LS-350M在与基线GPT-3模型（text-davinci-001）比较时实现了48.2%的显著胜率。考虑到GPT-3的自胜率为50%，MobileLLM-LS-350M获得了与该基线模型相当的对话性能。附录中的对话示例可视化也突显了MobileLLM模型生成响应的令人印象深刻的質量。

#### 3.3.2. API调用

API调用是一种常见的端侧应用，特别是在与语音转文本模型协作实现助手功能时。利用LLM进行API调用涉及将自然语言输入转换为JSON配置以调用相应的API [36]。例如，给定输入"Help me set an alarm at 7:30 AM"，模型输出{API: "alarm(time=\"7:30 am\")"}。此外，模型生成代理响应："Sure! Your alarm is set to 7:30 AM."

为了使LLM适应此任务，我们创建了一个包含5000个训练样本和2500个测试样本的合成数据集。每个样本平均涉及8轮对话。该数据集的详细示例见附录。预训练模型在训练集上进行4个epoch的微调，使用Adam优化器，线性衰减学习率从2e-5开始，权重衰减为0.01。

表6显示，MobileLLM-350M在意图和结构精确匹配分数上表现出与LLaMA-v2 7B相当的性能，其中高意图分数表示正确预测了用户意图调用的API，而结构精确匹配分数反映了预测API函数内内容的熟练程度。尽管MobileLLM-350M的Rouge分数低于7B模型，但需要指出的是，API调用更注重正确的API调用。结果表明，端侧应用中的某些常见场景并不特别具有挑战性，像MobileLLM-350M这样的较小模型可以熟练地处理它们。

### 3.4. 与量化的兼容性

我们进一步在MobileLLM和MobileLLM-LS模型上进行了per-token min-max训练后量化（PTQ）实验，模型大小分别为125M和350M，在0.25T token上训练。图7显示，采用W8A8 PTQ产生的准确率降低小于0.5个点，并且与层共享兼容。

<img src="/Users/dazhang/PycharmProject/Papers/2-LLM/.picture/image-20260812172506452.png" alt="image-20260812172506452" style="zoom:50%;" />

图7：

表

<img src="/Users/dazhang/PycharmProject/Papers/2-LLM/.picture/image-20260812172527796.png" alt="image-20260812172527796" style="zoom:50%;" />



### 3.5. 知识蒸馏

到目前为止，我们使用**下一个token作为硬标签从头开始训练紧凑模型**。我们探索了以 LLAMA-v2 7B为教师模型 的125M和350M模型的知识蒸馏（KD）[37]。不幸的是，KD增加了训练时间（减慢2.6–3.2倍），并且 与基于标签的训练相比表现出 **相当或较差的准确率**（详见附录）。

> [!NOTE]
>
> TODO：为什么？



### 3.6. 端侧性能分析

我们通过ExecuTorch [38]在iPhone 13（iOS 17.2.1）上，使用Metal Performance Shaders（MPS）后端[39]测量了MobileLLM-125M和MobileLLM-LS-125M **FP16模型的延迟**。模型加载、初始化和执行时间报告在表7中。具体来说，执行时间取50次迭代的平均值。

表7中的结果反映，通过权重共享和将层数加倍，MobileLLM-LS相比MobileLLM在加载和初始化时间上仅增加了2.2%，这归因于它们相似的模型大小。执行时间也仅有2.6%的开销，得益于**数据局部性**。相比之下，没有权重共享的加倍层数模型在加载和初始化时间上增加了143%，执行时间增加了86%。



## 4. 相关工作

LLM的卓越性能促进了其广泛应用。考虑到LLM的计算成本和能耗，一个新的研究方向已经出现，即缩小LLM以实现端侧推理。这些方法包括：

**模型压缩** 针对LLM开发了许多模型压缩方法，包括剪枝[40, 41, 42]、稀疏化[43, 44]和量化[45, 46, 47, 48, 49, 50]。我们的研究与这些技术互补。正如第3.4节也证实的，我们的方法与量化兼容。

**小型模型设计** 少数研究探索了紧凑模型架构，如TinyLLaMA [51]。然而，即使是最小的TinyLLaMA也超过了10亿参数，使得它们对于许多端侧应用仍然不可行。一些研究提出了大型模型架构及其模型家族中的较小LLM变体[13, 29, 26, 31]或包含小型LLM变体的分析套件[32]。然而，这些模型在亚十亿参数约束下并未优化，因此可能不是最优的。

**神经架构搜索** NAS在卷积神经网络领域引起了广泛关注，特别是在视觉任务方面[52, 53, 54, 55]。相比之下，在Transformer领域，主流观点认为只要参数总数一致，模型架构对准确率的影响很小[12]。只有少数研究开发了针对语言Transformer的NAS算法，主要针对BERT模型[56, 57, 58]。我们当前的研究聚焦于深度和宽度之间的相互作用，可以概念化为在深度空间中进行细致的网格搜索。该研究的结果挑战了关于缩放定律的主流正统观念，提出深而窄的架构对紧凑型LLM表现出更高的性能。

**权重共享** 权重共享是在固定参数约束下优化模型权重利用率的直观策略。虽然OPT家族[13]和后续工作[26]利用输入和输出嵌入之间的权重共享，但很少有研究探索**Transformer中间层的权重共享**[59, 60]。先前的努力通常需要对共享层进行专门设计。相比之下，我们的贡献强调了一种更直接但有效的方式，即**简单地重复Transformer块，在固定模型大小和最小延迟增加的情况下提高了准确率。**

> [!NOTE]
>
> 『简单地重复Transformer块』这是什么意思？是指 块级权重共享？

**高效注意力和实现** 在高效Transformer设计领域，许多研究专注于通过低秩近似[61, 62, 63]和稀疏注意力[64, 65]等方法优化注意力计算。另一条工作线探索硬件调度和权重移动，例如FlashAttention [66]和FlexGen [67]等工作。相比之下，我们的主要目标是在不引入新的注意力计算或高效硬件实现方法的情况下优化模型大小。



## 5. 结论

本研究聚焦于优化面向端侧应用的亚十亿规模模型。我们的发现表明，对于较小的模型，优先考虑深度而非宽度可以提升模型性能。此外，通过利用先进的权重共享技术，包括嵌入共享、分组查询注意力 和 **块级权重共享**，我们在存储受限的场景中实现了权重利用率的大幅提升。得到的MobileLLM模型在零样本常识推理、问答和阅读理解任务中相比先前的最优方法展现出实质性进步。最后但同样重要的是，我们展示了微调后的MobileLLM模型在**两种常见的端侧用例——对话和API调用**——中的有效性，强调了它们处理此类任务的熟练能力。



## 致谢

我们感谢Hansong Zhang在设置 **iOS延迟测量环境**方面的宝贵贡献，以及PyTorch边缘团队的全力支持。



## 影响声明

本文倡导在端侧应用中采用亚十亿规模的大型语言模型，旨在减轻LLM推理过程中的能源消耗。所提出的方法在降低LLM部署相关的计算成本方面具有前景。



## 参考文献

[1] GPT-4 has a trillion parameters. https://the-decoder.com/gpt-4-has-a-trillion-parameters/.

[2] Detailed calculation can be found in the appendix.

[3] https://www.nvidia.com/en-us/data-center/h100/.

[4] https://twitter.com/soumithchintala/status/1748074223187173724.

[5] Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., Bashlykov, N., Batra, S., Bhargava, P., Bhosale, S., et al. Llama 2: Open foundation and fine-tuned chat models, 2023b.

[6] Hristov, V. A16 Bionic explained: what's new in Apple's pro-grade mobile chip? https://www.phonearena.com/news/A16-Bionic-explained-whats-new_id142438, 2022.

[7] Google. Pixel 8 pro tech specs. https://store.google.com/gb/product/pixel_8_pro_specs, 2023.

[8] Malladi, K. T., Lee, B. C., Nothaft, F. A., Kozyrakis, C., Periyathambi, K., and Horowitz, M. Towards energy-proportional datacenter memory with mobile dram. ACM SIGARCH Computer Architecture News, 40(3):37–48, 2012.

[9] Han, S., Liu, X., Mao, H., Pu, J., Pedram, A., Horowitz, M. A., and Dally, W. J. Eie: Efficient inference engine on compressed deep neural network. ACM SIGARCH Computer Architecture News, 44(3):243–254, 2016.

[10] https://llm.mlc.ai.

[11] Our pre-training code is available at https://github.com/facebookresearch/MobileLLM.

[12] Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., Chess, B., Child, R., Gray, S., Radford, A., Wu, J., and Amodei, D. **Scaling laws for neural language models**. arXiv preprint arXiv:2001.08361, 2020.

[13] Zhang, S., Roller, S., Goyal, N., Artetxe, M., Chen, M., Chen, S., Dewan, C., Diab, M., Li, X., Lin, X. V., et al. Opt: Open pre-trained transformer language models. arXiv preprint arXiv:2205.01068, 2022.

[14] Ainslie, J., Lee-Thorp, J., de Jong, M., Zemlyanskiy, Y., Lebron, F., and Sanghai, S. GQA: Training generalized multi-query transformer models from multi-head checkpoints. In EMNLP, 2023.

[15] Dauphin, Y. N., Fan, A., Auli, M., and Grangier, D. Language modeling with gated convolutional networks. In International conference on machine learning, pp. 933–941. PMLR, 2017.

[16] Chowdhery, A., Narang, S., Devlin, J., Bosma, M., Mishra, G., Roberts, A., Barham, P., Chung, H. W., Sutton, C., Gehrmann, S., et al. Palm: Scaling language modeling with pathways. Journal of Machine Learning Research, 24(240):1–113, 2023.

[17] Clark, P., Cowhey, I., Etzioni, O., Khot, T., Sabharwal, A., Schoenick, C., and Tafjord, O. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv preprint arXiv:1803.05457, 2018.

[18] Clark, C., Lee, K., Chang, M.-W., Kwiatkowski, T., Collins, M., and Toutanova, K. Boolq: Exploring the surprising difficulty of natural yes/no questions. arXiv preprint arXiv:1905.10044, 2019.

[19] Bisk, Y., Zellers, R., Gao, J., Choi, Y., et al. Piqa: Reasoning about physical commonsense in natural language. In Proceedings of the AAAI conference on artificial intelligence, volume 34, pp. 7432–7439, 2020.

[20] Sap, M., Rashkin, H., Chen, D., LeBras, R., and Choi, Y. Socialiqa: Commonsense reasoning about social interactions. arXiv preprint arXiv:1904.09728, 2019.

[21] Zellers, R., Holtzman, A., Bisk, Y., Farhadi, A., and Choi, Y. Hellaswag: Can a machine really finish your sentence? arXiv preprint arXiv:1905.07830, 2019.

[22] Mihaylov, T., Clark, P., Khot, T., and Sabharwal, A. Can a suit of armor conduct electricity? a new dataset for open book question answering. arXiv preprint arXiv:1809.02789, 2018.

[23] Sakaguchi, K., Bras, R. L., Bhagavatula, C., and Choi, Y. Winogrande: An adversarial winograd schema challenge at scale. Communications of the ACM, 64(9):99–106, 2021.

[24] Joshi, M., Choi, E., Weld, D. S., and Zettlemoyer, L. Triviaqa: A large scale distantly supervised challenge dataset for reading comprehension. arXiv preprint arXiv:1705.03551, 2017.

[25] Lai, G., Xie, Q., Liu, H., Yang, Y., and Hovy, E. Race: Large-scale reading comprehension dataset from examinations. arXiv preprint arXiv:1704.04683, 2017.

[26] Black, S., Biderman, S., Hallahan, E., Anthony, Q., Gao, L., Golding, L., He, H., Leahy, C., McDonell, K., Phang, J., et al. Gpt-neox-20b: An open-source autoregressive language model. arXiv preprint arXiv:2204.06745, 2022.

[27] Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.-A., Lacroix, T., Rozière, B., Goyal, N., Hambro, E., Azhar, F., et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023a.

[28] Kingma, D. P. and Ba, J. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.

[29] Scao, T. L., Fan, A., Akiki, C., Pavlick, E., Ili´c, S., Hesslow, D., Castagné, R., Luccioni, A. S., Yvon, F., Gallé, M., et al. Bloom: A 176b-parameter open-access multilingual language model. arXiv preprint arXiv:2211.05100, 2022.

[30] Taylor, R., Kardas, M., Cucurull, G., Scialom, T., Hartshorn, A., Saravia, E., Poulton, A., Kerkez, V., and Stojnic, R. Galactica: A large language model for science. arXiv preprint arXiv:2211.09085, 2022.

[31] Dey, N., Gosal, G., Khachane, H., Marshall, W., Pathria, R., Tom, M., Hestness, J., et al. Cerebras-gpt: Open compute-optimal language models trained on the cerebras wafer-scale cluster. arXiv preprint arXiv:2304.03208, 2023.

[32] Biderman, S., Schoelkopf, H., Anthony, Q. G., Bradley, H., O'Brien, K., Hallahan, E., Khan, M. A., Purohit, S., Prashanth, U. S., Raff, E., et al. Pythia: A suite for analyzing large language models across training and scaling. In International Conference on Machine Learning, pp. 2397–2430. PMLR, 2023.

[33] Peng, B., Alcaide, E., Anthony, Q., Albalak, A., Arcadinho, S., Cao, H., Cheng, X., Chung, M., Grella, M., GV, K. K., et al. Rwkv: Reinventing rnns for the transformer era. arXiv preprint arXiv:2305.13048, 2023.

[34] Li, X., Zhang, T., Dubois, Y., Taori, R., Gulrajani, I., Guestrin, C., Liang, P., and Hashimoto, T. B. Alpacaeval: An automatic evaluator of instruction-following models. https://github.com/tatsu-lab/alpaca_eval, 2023.

[35] Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li, Z., Li, D., Xing, E., et al. Judging llm-as-a-judge with mt-bench and chatbot arena. arXiv preprint arXiv:2306.05685, 2023.

[36] https://platform.openai.com/docs/guides/function-calling.

[37] Hinton, G., Vinyals, O., Dean, J., et al. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531, 2(7), 2015.

[38] https://pytorch.org/executorch-overview.

[39] https://pytorch.org/executorch/stable/build-run-mps.html.

[40] Xia, M., Gao, T., Zeng, Z., and Chen, D. Sheared llama: Accelerating language model pre-training via structured pruning. arXiv preprint arXiv:2310.06694, 2023b.

[41] Sun, M., Liu, Z., Bair, A., and Kolter, J. Z. A simple and effective pruning approach for large language models. arXiv preprint arXiv:2306.11695, 2023.

[42] Frantar, E. and Alistarh, D. Sparsegpt: Massive language models can be accurately pruned in one-shot. In International Conference on Machine Learning, pp. 10323–10337. PMLR, 2023.

[43] Xia, H., Zheng, Z., Li, Y., Zhuang, D., Zhou, Z., Qiu, X., Li, Y., Lin, W., and Song, S. L. Flash-llm: Enabling cost-effective and highly-efficient large generative model inference with unstructured sparsity. arXiv preprint arXiv:2309.10285, 2023a.

[44] Frantar, E., Ashkboos, S., Hoefler, T., and Alistarh, D. Gptq: Accurate post-training quantization for generative pre-trained transformers. arXiv preprint arXiv:2210.17323, 2022.

[45] Liu, J., Gong, R., Wei, X., Dong, Z., Cai, J., and Zhuang, B. Qllm: Accurate and efficient low-bitwidth quantization for large language models. arXiv preprint arXiv:2310.08041, 2023a.

[46] Dettmers, T., Lewis, M., Belkada, Y., and Zettlemoyer, L. Llm. int8 (): 8-bit matrix multiplication for transformers at scale. arXiv preprint arXiv:2208.07339, 2022.

[47] Kim, S., Hooper, C., Gholami, A., Dong, Z., Li, X., Shen, S., Mahoney, M. W., and Keutzer, K. Squeezellm: Dense-and-sparse quantization. arXiv preprint arXiv:2306.07629, 2023.

[48] Xiao, G., Lin, J., Seznec, M., Wu, H., Demouth, J., and Han, S. Smoothquant: Accurate and efficient post-training quantization for large language models. In International Conference on Machine Learning, pp. 38087–38099. PMLR, 2023.

[49] Yao, Z., Yazdani Aminabadi, R., Zhang, M., Wu, X., Li, C., and He, Y. Zeroquant: Efficient and affordable post-training quantization for large-scale transformers. Advances in Neural Information Processing Systems, 35:27168–27183, 2022.

[50] Liu, Z., Oguz, B., Zhao, C., Chang, E., Stock, P., Mehdad, Y., Shi, Y., Krishnamoorthi, R., and Chandra, V. Llm-qat: Data-free quantization aware training for large language models. arXiv preprint arXiv:2305.17888, 2023c.

[51] Timiryasov, I. and Tastet, J.-L. Baby llama: knowledge distillation from an ensemble of teachers trained on a small dataset with no performance penalty. arXiv preprint arXiv:2308.02019, 2023.

[52] Tan, M. and Le, Q. Efficientnet: Rethinking model scaling for convolutional neural networks. In International conference on machine learning, pp. 6105–6114. PMLR, 2019.

[53] Zoph, B. and Le, Q. V. Neural architecture search with reinforcement learning. arXiv preprint arXiv:1611.01578, 2016.

[54] Wu, B., Dai, X., Zhang, P., Wang, Y., Sun, F., Wu, Y., Tian, Y., Vajda, P., Jia, Y., and Keutzer, K. Fbnet: Hardware-aware efficient convnet design via differentiable neural architecture search. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10734–10742, 2019.

[55] Guo, Z., Zhang, X., Mu, H., Heng, W., Liu, Z., Wei, Y., and Sun, J. Single path one-shot neural architecture search with uniform sampling. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XVI 16, pp. 544–560. Springer, 2020.

[56] Xu, J., Tan, X., Luo, R., Song, K., Li, J., Qin, T., and Liu, T.-Y. Nas-bert: task-agnostic and adaptive-size bert compression with neural architecture search. In Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining, pp. 1933–1943, 2021.

[57] Jawahar, G., Yang, H., Xiong, Y., Liu, Z., Wang, D., Sun, F., Li, M., Pappu, A., Oguz, B., Abdul-Mageed, M., et al. Mixture-of-supernets: Improving weight-sharing supernet training with architecture-routed mixture-of-experts. arXiv preprint arXiv:2306.04845, 2023.

[58] Ganesan, V., Ramesh, G., and Kumar, P. Supershaper: Task-agnostic super pre-training of bert models with variable hidden dimensions. arXiv preprint arXiv:2110.04711, 2021.

[59] Shen, Z., Liu, Z., and Xing, E. Sliced recursive transformer. In European Conference on Computer Vision, pp. 727–744. Springer, 2022.

[60] Reid, M., Marrese-Taylor, E., and Matsuo, Y. Subformer: Exploring weight sharing for parameter efficiency in generative transformers. arXiv preprint arXiv:2101.00234, 2021.

[61] Wang, S., Li, B. Z., Khabsa, M., Fang, H., and Ma, H. Linformer: Self-attention with linear complexity. arXiv preprint arXiv:2006.04768, 2020.

[62] Katharopoulos, A., Vyas, A., Pappas, N., and Fleuret, F. Transformers are rnns: Fast autoregressive transformers with linear attention. In International conference on machine learning, pp. 5156–5165. PMLR, 2020.

[63] Xiong, Y., Zeng, Z., Chakraborty, R., Tan, M., Fung, G., Li, Y., and Singh, V. Nyströmformer: A nyström-based algorithm for approximating self-attention. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, pp. 14138–14148, 2021.

[64] Kitaev, N., Kaiser, Ł., and Levskaya, A. Reformer: The efficient transformer. arXiv preprint arXiv:2001.04451, 2020.

[65] Roy, A., Saffar, M., Vaswani, A., and Grangier, D. Efficient content-based sparse attention with routing transformers. Transactions of the Association for Computational Linguistics, 9:53–68, 2021.

[66] Dao, T., Fu, D., Ermon, S., Rudra, A., and Ré, C. Flashattention: Fast and memory-efficient exact attention with io-awareness. Advances in Neural Information Processing Systems, 35:16344–16359, 2022.

[67] Sheng, Y., Zheng, L., Yuan, B., Li, Z., Ryabinin, M., Chen, B., Liang, P., Ré, C., Stoica, I., and Zhang, C. Flexgen: High-throughput generative inference of large language models with a single gpu. In International Conference on Machine Learning, pp. 31094–31116. PMLR, 2023.
[68] Almazrouei, E., Alobeidli, H., Alshamsi, A., Cappelli, A., Cojocaru, R., Debbah, M., Goffinet, É., Hesslow, D., Launay, J., Malartic, Q., et al. The falcon series of open language models. arXiv preprint arXiv:2311.16867, 2023.
[69] Zhang, P., Zeng, G., Wang, T., and Lu, W. Tinyllama: An open-source small language model. arXiv preprint arXiv:2401.02385, 2024.
[70] Wu, M., Waheed, A., Zhang, C., Abdul-Mageed, M., and Aji, A. F. Lamini-lm: A diverse herd of distilled models from large-scale instructions. arXiv preprint arXiv:2304.14402, 2023.
[71] Bai, J., Bai, S., Chu, Y., Cui, Z., Dang, K., Deng, X., Fan, Y., Ge, W., Han, Y., Huang, F., et al. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023.
[72] Thawakar, O., Vayani, A., Khan, S., Cholakal, H., Anwer, R. M., Felsberg, M., Baldwin, T., Xing, E. P., and Khan, F. S. Mobillama: Towards accurate and lightweight fully transparent gpt. arXiv preprint arXiv:2402.16840, 2024.

---



# 附录

## A. 扩展到更大的模型架构

在本文中，我们主要研究了两种模型大小：MobileLLM-125M 和 MobileLLM-350M。在本节中，我们将我们的设计原则——SwiGLU、更深架构、分组查询注意力和嵌入共享——扩展到更大的模型，预训练了MobileLLM-600M、1B和1.5B变体。这一扩展促进了跨不同内存约束的更广泛应用。表8将MobileLLM与各种通用预训练模型[10]进行了比较，包括OPT [13]、BLOOM [29]、GPT-neo [26]、Pythia [32]、Falcon [68]、TinyLlama [69]、Cerebras-GPT [31]、Galactica [30]、RWKV [33]、LaMini-GPT [70]、Qwen [71]以及在MobileLLM之后发布的最新模型MobiLlama [72]。

表8中的结果表明，MobileLLM持续超越先前类似规模的模型。值得注意的是，MobileLLM-1.5B在零样本常识推理任务上达到了59.4个点的平均准确率，超过了先前最先进的Qwen1.5-1.8B模型2.9个点，尽管后者具有更多参数。MobileLLM的详细架构规格见表9。

**表8: MobileLLM-600M、1B和1.5B在常识推理任务上的零样本性能。每个模型大小类别中最高和第二高的平均分数已突出显示。**

![image-20260812172556585](/Users/dazhang/PycharmProject/Papers/2-LLM/.picture/image-20260812172556585.png)

> [10] Models pre-trained for specific downstream tasks were excluded to ensure a fair comparison.

**表9: MobileLLM的详细架构规格。"Emb Dim"表示嵌入维度，"Hidden Dim"表示前馈网络内部的维度。**

![image-20260812172612097](/Users/dazhang/PycharmProject/Papers/2-LLM/.picture/image-20260812172612097.png)



## B. 每个设计选择的影响

本节展示了在125M和350M模型大小下改进亚十亿规模LLM设计实验的全面表格结果。查看表10的结果，从传统前馈网络（FC \to ReLU \to FC）过渡到SwiGLU，两种模型大小的准确率均提升了1.3%。进一步增加模型深度，125M/350M模型的准确率分别提升了0.9%/1.1%。然后，引入输入和输出嵌入共享实现了约10%的参数减少，同时125M模型仅有0.2%的微小准确率下降，350M模型下降0.6%。此外，根据第D节的发现，我们采用了头维度等于64、头数约为kv-head数的4倍的分组查询注意力，同时增加嵌入维度以保持模型大小。这一修改进一步使125M/350M模型的性能提升了0.4%/0.7%。结合这些技术，建立了一个强基线网络，称为MobileLLM。最后，即时块级权重共享技术为在0.25万亿token上训练的模型贡献了额外1.1%的准确率增益，得到了MobileLLM-LS模型。最终的MobileLLM和MobileLLM-LS模型使用1万亿token进行训练。

**表10: 每个设计选择对模型在零样本常识推理任务上准确率影响的消融研究。对应于图3中的柱状图。这里，L、H、HKV分别表示层数、头数、kv-heads数，dim表示嵌入维度。**

![image-20260812172631385](/Users/dazhang/PycharmProject/Papers/2-LLM/.picture/image-20260812172631385.png)



## C. 深度 vs 宽度

我们在表11中提供了网络深度与宽度在零样本推理任务上的探索结果，以及在表12中提供了问答和阅读理解任务上的结果。研究结果表明，层数少于10层的浅层架构在推理或处理理解任务方面表现不佳。10-20层的模型表现出改进的准确率，而进一步增加深度在所有三个任务上继续提供显著的益处。值得注意的是，对于亚十亿规模模型，最优深度大约为30层。

**表11: 架构设计中深度与宽度的消融研究，如图4 (a)(b)所示。对于紧凑型模型，通过零样本常识推理任务评估，优先考虑深度而非宽度可以获得更优的性能。**

![image-20260812172654460](/Users/dazhang/PycharmProject/Papers/2-LLM/.picture/image-20260812172654460.png)

**表12: 在TQA和RACE数据集上架构设计中深度与宽度的消融研究，如图4 (c-f)所示。**

![image-20260812172711575](/Users/dazhang/PycharmProject/Papers/2-LLM/.picture/image-20260812172711575.png)

## D. 头数和键值头数

我们在表13中提供了评估注意力头数和键值头数对零样本推理准确率影响的详细实验结果。我们的研究涉及两个基线架构：一个嵌入维度为896的8层125M模型，和一个嵌入维度为1280的15层350M模型。我们在{8, 16, 32}范围内进行头大小扫描。表13中显示的研究结果表明，使用16个头（头维度接近64）和4个键值头，可以产生最佳的准确率和内存权衡。这一设置作为我们模型架构设计中的指导原则。

**表13: 探究注意力头数和键值头数影响的消融研究。**

![image-20260812172727312](/Users/dazhang/PycharmProject/Papers/2-LLM/.picture/image-20260812172727312.png)



## E. 层共享数量的消融

我们扩展了研究以确定最优的层重复次数。实验涉及嵌入维度为896的8层125M模型和嵌入维度为1280的15层350M模型。表14中的结果表明，当我们将层数加倍且每两个Transformer块共享权重时，准确率提高了0.4-0.6%。然而，当我们进一步将层重复次数增加到三倍或四倍时，这种准确率提升效果减弱。因此，在我们的实验中，我们采用了每两个块共享权重的配置，有效地将总层数加倍。

**表14: 不同层重复次数影响的消融研究。**

![image-20260812172737569](/Users/dazhang/PycharmProject/Papers/2-LLM/.picture/image-20260812172737569.png)



## F. 与量化的兼容性

本节探讨了量化与所提出的模型架构和层共享的兼容性。我们采用直接的per-token min-max量化，使用训练后量化（PTQ）将权重和激活都量化到8-bit。实验在MobileLLM和MobileLLM-LS上进行，模型大小为125M和350M，在0.25T token上训练。表15中的结果表明，W8A8 PTQ导致的准确率下降在0.5%以内，并且与所提出的层共享方法兼容。

**表15: 消融研究：8-bit权重、8-bit激活训练后量化在零样本常识推理任务上的结果。量化模型与全精度BF16对应模型相比，准确率差距在0.5%以内。**

![image-20260812172806367](/Users/dazhang/PycharmProject/Papers/2-LLM/.picture/image-20260812172806367.png)



## G. 知识蒸馏

将知识蒸馏（KD）[37]集成到小模型预训练中的结果如表16所示。LLaMA-v2 7B模型作为教师，KD损失使用大型预训练教师模型（即LLaMA-v2 7B）和小型学生网络（即125M或350M模型）的logits之间的交叉熵计算：

$$
L_{CE} = -\frac{1}{n} \sum_{c} \sum_{i=1}^{n} p_c^T(X_i) \log(p_c^S(X_i)) \qquad (1)
$$

这里， $i$ 表示当前批次中的第 $i$ 个样本，批次中共有 $n$ 个样本， $c$ 表示类别数量，在我们的情况下等于词表大小。 $T$ 和 $S$ 分别是教师网络和学生网络。

表16中的结果表明，添加KD损失与仅使用下一个token作为标签相比，结果相当甚至更低。然而，值得注意的是，使用KD的训练时间比从头开始使用标签训练慢2.6–3.2倍。所有模型在32块A100 80G GPU上训练，批大小为32，共120k次迭代。因此，我们在实验中选择了使用标签。

**表16: 使用LLaMA-v2 7B教师输出作为软标签进行知识蒸馏（KD）的消融研究。结果表明，与仅使用硬标签相比，加入KD损失后性能略有下降。**

![image-20260812172821474](/Users/dazhang/PycharmProject/Papers/2-LLM/.picture/image-20260812172821474.png)



## H. 数据集和基准

MobileLLM在零样本常识推理（BoolQ、PIQA、SIQA、HellaSwag、Winogrande、ARC、OBQA）、问答（TriviaQA）和阅读理解（RACE）任务上进行了评估。此外，我们在MT-Bench和AlpacaEval基准上评估了对话模型。我们还生成了一个API调用数据集，用于微调和评估模型在此特定任务上的表现。

### H.1. 零样本常识推理任务

**BoolQ** [18]是一个阅读理解数据集，专注于自然出现的是/否问题。每个实例包括一个问题（Q）、一段文章摘录（P）和一个答案（A），并附带解释以提高清晰度。

**PIQA** [19]是Physical Interaction: Question Answering的缩写，作为评估和研究自然语言模型理解物理常识能力的基准。

**SIQA** [20]是Social Interaction Question Answering的缩写，旨在通过多项选择问答来衡量计算模型的社交和情感智能。

**HellaSwag** [21]作为物理情境化常识自然语言推理的基准。它包含四选一的多项选择问题，这些问题对人类来说被认为是简单的（>95%准确率），但对语言模型来说具有挑战性。

**WinoGrande** [23]是一个常识推理基准。它由273个由专家精心设计的代词消解问题组成，故意设计为无法被依赖选择偏好或词语关联的统计模型解决。

**ARC** [17]（AI2推理挑战）是一个包含7787个自然科学问题的集合。它分为挑战集和简单集，挑战集仅包含那些被基于检索的算法和词语共现算法都回答错误的问题。

**OBQA** [22]是一个由约6000个问题组成的数据集，专为开放书籍问答设计。该任务涉及将提供的科学事实语料库（开放书籍）与外部广泛的常识知识相结合。提供正确答案需要利用超出核心事实的简单常识知识。

### H.2. 问答任务

**TriviaQA** [24]是一个闭卷问答基准。它包含超过65万个问答证据三元组，由Trivia爱好者编写的95,000个问答对组合而成。每个问题平均有六个证据文档支持。

### H.3. 阅读理解任务

**RACE** [25]是一个大规模阅读理解数据集，包含超过28,000篇文章和近100,000个问题。该数据集收集自中国的英语考试，针对初中和高中学生设计。

### H.4. 对话基准

**AlpacaEval** [34]是一个单轮基准，包含805个涵盖各种主题的问题。在本文的评估中，模型生成响应并由GPT-4评分，最终指标是与基线模型（text-davinci-001）的成对胜率。

**MT-Bench** [35]是一个多轮基准，包含160个涵盖八个不同知识领域的问题。在此评估中，模型需要回答一个初始问题，然后对预定义的后续问题提供第二个响应。每个模型响应由GPT-4按1到10的等级评分，最终分数由两轮的平均值决定。

### H.5. API调用数据集

API调用数据集通过指导语言模型模拟人类与代理之间的对话生成。此过程指示语言模型在模拟对话中明确提及应使用指定API执行的任务。生成数据集中对话轮次的示例如下：

```
{"role": "USER", "message": "Can you help me send a SMS message: We are looking forward to seeing you"},
{"role": "PARSER", "message": "style_transfer(text = \"We are looking forward to seeing you\", style = [\"sms\"], contact = None)"},
{"role": "AGENT", "message": "Sure! Here's a possible SMS message: Hey [name], we are looking forward to seeing you! Wishing you all well."},
{"role": "USER", "message": "How did the S&P 500 perform last month?"},
{"role": "PARSER", "message": "get_stock_information(company_name = \"S&P 500\", ticker_symbol = \"GSPC\", date_time = \"last month\", financial_metrics = \"price\")"},
{"role": "AGENT", "message": "The S&P 500 closed at $2,977.94 last month, with a monthly return of 2.5%."},
{"role": "USER", "message": "Is the San Diego Zoo open during the winter months?"},
{"role": "PARSER", "message": "get_local_information(point_of_interest = \"San Diego Zoo\", attributes = \"hours of operation\")"},
{"role": "AGENT", "message": "The San Diego Zoo is open year-round, but the hours of operation vary depending on the time of year. During the winter months, the zoo is open from 9:00 AM to 5:00 PM."},
{"role": "USER", "message": "What is the most recent news concerning aliens?"},
{"role": "PARSER", "message": "news(query = \"latest alien news\")"},
{"role": "AGENT", "message": "According to recent news articles, NASA has announced the discovery of a new planet outside of our solar system that has the potential to support alien life. The planet, named Kepler-186f, is located in the constellation Cygnus and is the first known Earth-sized planet to be found in the habitable zone of a star..."},
{"role": "USER", "message": "Who won the last Super Bowl?"},
{"role": "PARSER", "message": "get_sports_information(match = \"\", sport_type = \"NFL\", metric = \"winner\")"},
{"role": "AGENT", "message": "The last Super Bowl was won by the Kansas City Chiefs."}
```



## I. 预期的未来GPU消耗近似计算

$$
\#\text{GPUs} = 7.88 \times 10^9 (\text{population size}) \times 5\% (\text{individual time}^{11}) \times 220 \times 10^9 (\text{approximating FLOPs/token for 220B GPT-4 model with one expert activated}^{12}) \times 50 (\text{token/s}) \times 24 (\text{h}) \times 3600 (\text{s}) / (60 \times 10^{12} (\text{FLOPs/s, H100 computation capacity}) \times 24 (\text{h}) \times 3600 (\text{s})) \approx 1 \times 10^8
$$

这假设序列长度较短（即几百个token，而非数万个token），此时计算由线性层主导。对于非常长的序列，自注意力计算将占主导地位，这将需要更复杂的公式。

> [11] 根据统计数据，YouTube用户平均每月花费23小时（来源），Instagram用户每月在应用上花费12小时（来源），我们有理由相信未来与LLM的互动可以轻易超过这些平台的累积参与度。
> [12] https://the-decoder.com/gpt-4-has-a-trillion-parameters/



## J. MobileLLM的对话生成

我们展示了MobileLLM在回答各种问题时生成的几个对话输出示例。

**输入：** Are there any particular physical benefits to mindful walking, such as improved posture or increased physical fitness?

**输出（MobileLLM-125M）：** [英文原文保留，略]

**输出（MobileLLM-350M）：** [英文原文保留]

**输出（llama-v2-7b-chat）：** [英文原文保留]

**输入：** How do I wrap a present neatly?

**输出（MobileLLM-125M）：** [英文原文保留]

**输出（MobileLLM-350M）：** [英文原文保留]

**输出（llama-v2-7b-chat）：** [英文原文保留]

**输入：** What are the best exercises for beginners?

**输出（MobileLLM-125M）：** [英文原文保留]

**输出（MobileLLM-350M）：** [英文原文保留]

**输出（llama-v2-7b-chat）：** [英文原文保留]
