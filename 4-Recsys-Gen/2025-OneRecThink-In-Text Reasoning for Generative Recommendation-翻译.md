# OneRec-Think：生成式推荐中的文本内推理

> {liuzhanyu,wangshiyao08,zhouguorui}@kuaishou.com

本文介绍了 OneRec-Think：生成式推荐中的文本内推理。核心内容：
关键发现：
---
**OneRec-Think: In-Text Reasoning for Generative Recommendation**
刘展羽\*，王诗瑶\*，王兴梅，张荣洲，邓佳欣，
鲍红辉，张景浩，李无愁，郑鹏飞，吴翔宇，
胡逸飞，胡奇根，罗鑫辰，任乐健，张子兴，
王倩倩，蔡阔，吴云帆，程鸿涛，程泽宪，
任璐，王焕杰，苏毅，唐瑞明，蔡坤，周国瑞†
快手科技有限公司，中国北京
## 摘要
大型语言模型（LLM）强大的生成能力已经在推荐系统中引发了范式转变。然而，现有的生成式模型（如OneRec）作为隐式预测器运行，严重缺乏显式、可控的推理能力——而这正是LLM的关键优势。为弥合这一差距，我们提出了OneRec-Think，一个统一对话、推理和个性化推荐的框架。OneRec-Think包含：(1) **item级对齐**（Itemic Alignment）：跨模态的语义对齐；(2) **推理激活**（Reasoning Activation）：推理脚手架（Reasoning Scaffolding）以激活LLM在推荐上下文中的推理能力；(3) **推理增强**（Reasoning Enhancement），其中我们设计了推荐专用的奖励函数，以考虑用户偏好的多效性（multi-validity）本质。跨公开基准的实验展示了最先进的性能。此外，我们提出的"Think-Ahead"架构在快手上实现了有效的工业部署，获得了0.159%的应用停留时间（APP Stay Time）提升，验证了模型显式推理能力的实际效果。
## 1 引言
大型语言模型（LLM）的快速发展从根本上重塑了推荐系统，开启了生成式检索范式（GR）(Peng et al., 2025; Zhang et al., 2025b; Deldjoo et al., 2024; Li et al., 2023; Wang et al., 2023)。该方法代表了从传统查询-候选匹配的深刻转变，利用基于Transformer的序列到序列模型来自回归解码目标候选项的标识符。基于此，一个主要的研究前沿是端到端生成式框架的开发，包括OneRec、OneLoc、OneSug和OneSearch (Deng et al., 2025; Zhou et al., 2025a,b; Wei et al., 2025; Guo et al., 2025b; Chen et al., 2025)。这些统一模型取代了传统的多级推荐漏斗（涉及独立的检索和排序阶段），实现了面向最终目标的整体优化，并集中计算资源以获得更好的工业扩展和性能。
虽然这些模型成功利用了LLM的输出生成能力，但它们从根本上缺乏定义现代LLM突破的显式、可验证的推理路径，如基于文本的思维链（CoT）(Rajput et al., 2023; Zheng et al., 2024; Wang et al., 2024b)。为弥合这一关键差距，我们提出OneRec-Think，一个将对话、推理和个性化生成式推荐集成在单个统一模型中的新颖框架。它能够生成高质量、可解释的文本推理路径，显著提高了推荐准确性和用户信任度。模型固有的对话性质进一步实现了根据特定用户约束动态定制推荐（如图1所示）。我们的方法通过三阶段框架实现：(1) **item级对齐**（Itemic Alignment），将item语义映射到LLM的文本嵌入空间，建立统一的表征连续体，从而释放模型的推理能力；(2) **推理激活**（Reasoning Activation），旨在直接在推荐系统上下文中诱发LLM固有的推理能力；(3) **推理增强**（Reasoning Enhancement），利用推荐专用的奖励函数来捕捉用户偏好的多效性（即多个有效选择）本质。此外，我们引入了OneRec-Think推理架构，以确保在大规模工业服务场景中的高效部署和实时响应能力。
我们的贡献总结如下：
* 我们引入了一个统一框架，弥合了离散推荐item与连续推理空间之间的语义鸿沟，使得个性化推荐能够无缝集成到LLM的自然语言理解中。
* 我们设计了一种新颖的推理范式，将多步推演与推荐优化相协调，通过协同训练实现了可解释且精度感知的个性化推荐。
* 所提出的方法在多个公开基准上取得了最先进的结果，同时我们部署友好的"Think-Ahead"架构带来了显著的工业影响，实现了0.159%的应用停留时间提升。
## 2 相关工作
### 2.1 大型语言模型中的推理
大型语言模型通过各种提示技术实现复杂推理，其中思维链（CoT）提示 (Wei et al., 2022) 是将问题分解为中间推理步骤的基础方法。这启发了众多扩展，包括零样本CoT (Kojima et al., 2022)、自一致性解码 (Wang et al., 2022) 和思维树 (Yao et al., 2023)。这些技术实现了测试时扩展，即在推理过程中使用额外的计算预算来提高性能 (Snell et al., 2024)。最近的工作已将焦点从提示转向通过使用强化学习等技术进行推理能力的训练后增强。包括DeepSeek-R1 (Guo et al., 2025a) 和Seed-1.5 (Seed et al., 2025) 在内的模型通过GRPO (Shao et al., 2024)、DAPO (Yu et al., 2025) 和VAPO (Yue et al., 2025) 等技术优化推理行为，这展示了该方向上有前景的进展。
### 2.2 基于推理的推荐
尽管诸如TIGER (Rajput et al., 2023)、HSTU (Zhai et al., 2024) 和OneRec (Zhou et al., 2025a) 等生成式推荐模型已经证明了有效性，但它们本质上缺乏推理能力。最近，基于推理的推荐系统旨在执行多步演绎，以获得更准确和可解释的推荐。现有方法分为两类：显式推理方法生成人类可读的推理过程，但局限于判别性任务 (Tsai et al., 2024; Bismay et al., 2024; Fang et al., 2025; Kim et al.)；而隐式推理方法 (Zhang et al., 2025a; Tang et al., 2025) 在没有文本可解释性的情况下进行潜在推理。我们的工作将显式推理引入生成式推荐，弥合了这一差距，同时实现了可解释的推理过程和可扩展的item生成。
## 3 预备知识
**item级token（Itemic Token）**：item级token是一个离散的、富含语义的item表示单元，类似于自然语言中的词token。遵循OneRec (Zhou et al., 2025a; Rajput et al., 2023)，我们将每个itemv映射到一个token序列 s_v = (s_v^1, ..., s_v^L)，这些token从item的多模态和协同内容生成。
**问题定义**：令U和V分别表示用户和item的集合。每个用户u
$$
\in
$$
 U有一个按时间顺序的交互历史V_u = (v_u^1, v_u^2, ..., v_u^{n_u})，长度为n_u。为简洁起见，我们省略用户下标u。通过使用item级token，用户的交互历史因此由序列S_u = ( $s_{v_1}$ , ..., $s_{v_n}$ )表示。
传统的生成式推荐器 (Rajput et al., 2023; Zhou et al., 2025a,b; Zheng et al., 2024) 将其任务定义为生成下一个item级token：
 $s_{v_{n+1}$ } ∼ P(· | $s_{v_1}$ , ..., $s_{v_n}$ ; $\theta$ ) (1)
相比之下，我们重新定义该任务以在单次自回归过程中统一推理和推荐。基于提示的用户历史，我们顺序生成token，从推理序列 $\tau$ = (r_1, ..., r_M)开始，以下一个item级token $s_{v_{n+1}$ }结束。这个端到端过程由下式描述：
 $\tau$ ∼ P(· | P( $s_{v_1}$ , ..., $s_{v_n}$ ); $\theta$ ) (2)
 $s_{v_{n+1}$ } ∼ P(· | P( $s_{v_1}$ , ..., $s_{v_n}$ ), $\tau$ ; $\theta$ ) (3)
其中P(·)表示针对推荐构建的有效提示。
## 4 方法
我们现在介绍OneRec-Think，一个用于端到端生成式推理推荐的可扩展框架。我们的方法包含三个核心组件：一个item级对齐（Itemic Alignment）阶段、一个推理激活（Reasoning Activation）阶段，以及一个用于工业部署的"Think-Ahead"架构。OneRec-Think的示意图如图2所示。
### 4.1 通过多任务预训练的item级对齐
为将推荐知识与LLM的语言空间对齐，我们设计了一个多任务预训练策略，通过在下一次token预测（Next Token Prediction）下的四个互补任务，实现对自然语言和item级token的无缝处理。
**交错式用户画像接地（Interleaved User Persona Grounding）**：与之前使用纯文本数据或孤立的item序列的工作不同，该任务将item级token与用户画像的文本token交错排列。它包括序列化的静态属性、主动搜索行为、交互序列以及总结的用户兴趣。这种组合创建了丰富的双模态训练实例，其中item级token基于其语义上下文。
**序列偏好建模（Sequential Preference Modeling）**：作为核心推荐任务，该任务构建数据，教会模型从按时间顺序的用户历史中预测后续item交互。
**item级密集描述（Itemic Dense Captioning）**：该任务要求模型从其item级token解码item的描述性内容。通过学习生成详细的文本描述，模型建立起对item组合所代表的语义特征的基本理解。
**通用语言建模（General Language Modeling）**：该任务继续在通用文本语料上预训练模型，在将模型应用于推荐场景时保持模型的基础语言能力。
为在保持模型语言能力的同时实现有效的知识集成，我们实施了一个两子阶段训练策略以确保稳定的对齐。token预热（Token Warm-up）子阶段在保持基础LLM冻结的同时，专门在交错式用户画像接地任务上训练item级token嵌入。随后的多任务集成（Multi-Task Integration）子阶段使用设计的比例在所有参数上联合优化组合任务（详见附录A.3）。
### 4.2 推理激活
尽管有强大的item级对齐，直接应用于工业推荐场景往往由于真实世界用户行为序列的噪声和冗长性而无法产生有效的CoT推理。为解决此问题，我们提出了一个有监督微调框架，首先从剪枝后的用户上下文中提取连贯的推理轨迹，然后利用这些轨迹指导在原始行为数据上的推理过程生成，从而实现在噪声工业环境中的有效上下文蒸馏（如图2(b)所示）。
**使用剪枝上下文的自举（Bootstrapping with Pruned Contexts）**：为自举推理能力，我们首先构建易于学习的实例，其中逻辑关系在序列剪枝后仍然保留。对每个用户，我们选择目标item $s_{v_{n+1}$ }并形成上下文-目标对⟨( $s_{v_1}$ , ..., $s_{v_n}$ ), $s_{v_{n+1}$ }⟩。然后我们使用相似度函数g(·,·)检索top-k个最相关历史item：
g(( $s_{v_1}$ , ..., $s_{v_n}$ ), $s_{v_{n+1}$ }) = ( $s_{w_1}$ , ..., $s_{w_k}$ ) (3)
使用这些相关item，我们查询预对齐模型以生成解释目标交互的推理过程 $\tau$ ：
 $\tau$ ∼ P(· | P_r(( $s_{w_1}$ , ..., $s_{w_k}$ ), $s_{v_{n+1}$ }); $\theta$ ) (4)
其中P_r(a,b)表示构建提示来查询为什么与item序列a交互的用户会与itemb交互。这个过程产生高质量的推理过程，既逻辑合理又目标对齐，为推理归纳提供了理想的训练信号。
**从噪声序列学习推理（Learning to Reason from Noisy Sequences）**：蒸馏得到的推理过程作为从原始序列学习的监督信号。训练目标最小化生成推理过程和目标item的负对数似然：
L_RA = −( $\Sigma$ _{i=1}^{M} log P(r_i | P( $s_{v_1}$ , ..., $s_{v_n}$ ), $r_{<i}$ ; $\theta$ ) + $\Sigma$ _{j=1}^{L} log P( $s_{v_{n+1}$ }^j | P( $s_{v_1}$ , ..., $s_{v_n}$ ), $\tau$ , $s_{v_{n+1}$ }^{<j}; $\theta$ ) ) (5)
其中 $\tau$ = {r_1, ..., r_M}表示推理过程token， $s_{v_{n+1}$ } = { $s_{v_{n+1}$ }^1, ..., $s_{v_{n+1}$ }^L}表示目标itemtoken。通过优化L_RA，模型学会从噪声序列内部蒸馏相关上下文，并生成将用户历史与目标交互联系起来的连贯推理过程，显著增强其在挑战性推荐场景中的CoT能力。
### 4.3 推理增强
在推理激活的CoT能力基础上，我们通过强化学习解决确保一致高质量推理的挑战。该阶段使用为生成式推荐量身定制的新颖奖励机制来优化推荐精度。
**波束候选奖励最大化（Beam Candidate Reward Maximization）**：标准的可验证pass奖励在推荐场景中面临显著的稀疏性挑战，因为大多数推理rollout未能命中目标item，从而产生相同的零奖励——这在GRPO (Shao et al., 2024)等算法中消除了群体优势。为克服此问题，我们引入了Rollout-Beam奖励，通过模型在约束波束内可达到的最佳性能来评估推理能力。我们的方法在推理轨迹生成后使用宽度为K的波束搜索探索多个生成候选：
 $R_{Rollout-Beam}$ = ma $x_{ŝ_{v_{n+1}$ }
$$
\in
$$
 B} $\Sigma$ _{l=1}^{L} I(ŝ_{ $v_{n+1}$ }^l = $s_{v_{n+1}$ }^l) (6)
其中波束搜索结果集定义为：
B = { (ŝ_{ $v_{n+1}$ }^{1,(j)}, ..., ŝ_{ $v_{n+1}$ }^{L,(j)}) }_{j=1}^{K} = BeamSearch(P( $s_{v_{n+1}$ } | H, $\tau$ ; $\theta$ ), K) (7)
其中包含在分布P( $s_{v_{n+1}$ } | H, $\tau$ ; $\theta$ )的波束搜索中具有top-K概率的item。H = P( $s_{v_1}$ , ..., $s_{v_n}$ )是历史序列的有效提示。BeamSearch(P, K)表示在分布P内波束搜索的top-K结果。随后，我们基于 $R_{Rollout-Beam}$ 使用GRPO (Shao et al., 2024)优化模型，这有效利用了用户偏好多效性带来的丰富奖励信号。
总的来说，这一设计通过将奖励计算与基于波束搜索的推理对齐，建立了训练-推理一致性，通过多路径评估提供了更密集的学习信号。
### 4.4 工业部署："Think-Ahead"架构
在工业推荐系统中部署OneRec-Think面临一个根本性挑战：协调多步推理的计算需求与实时用户交互的严格延迟要求。
为解决这一关键瓶颈，我们引入了一种新颖的Think-Ahead推理架构。我们的解决方案策略性地将模型推理解耦为两个阶段：在第一阶段，计算密集的推理路径和初始itemtoken（例如前两个item级token）由完整的OneRec-Think模型离线生成。这些初始token旨在捕捉用户的广泛意图或一般偏好背景。随后，第二阶段采用一个遵循OneRec (Zhou et al., 2025a)实时更新的模型进行在线最终化。它利用预先生成的itemtoken作为约束前缀，快速生成最终的item级token。这一设计通过利用当前上下文数据，确保了实时响应能力并实现了生产级性能。该架构的细节在附录A.3.4中。
## 5 实验
### 5.1 实验设置
**数据集和基线**：我们使用来自流行的Amazon评论基准¹的三个真实世界推荐数据集：Beauty、Toys和Sports。我们将OneRec-Think与两组竞争基线进行比较：(1) 经典序列方法，如BERT4Rec (Sun et al., 2019)、HGN (Ma et al., 2019)、GRU4Rec (Hidasi et al., 2016)和SASRec (Kang and McAuley, 2018)；(2) 生成式推荐模型，如TIGER (Rajput et al., 2023)、HSTU (Zhai et al., 2024)和ReaRec (Tang et al., 2025)。使用K=5和10的Top-K召回率（R@K）和NDCG（N@K）作为指标，遵循(Rajput et al., 2023)。实现细节在附录A.1中。
### 5.2 整体性能
结果如表1所示。我们可以观察到，利用强大基于推理架构的模型（ReaRec和我们的OneRec-Think）持续优于传统序列推荐器和生成式推荐器。这一稳健趋势证实了有效序列预测需要强大的推理和上下文推理能力。此外，OneRec-Think在所有基准上进一步取得了最佳性能。这一显著优势直接归因于item生成的显式、基于文本的推理能力，而先前工作中的生成机制则更为隐式、纯粹基于学习。
### 5.3 消融研究
我们在Beauty数据集上进行了消融研究，比较三种配置：在原始item级token序列上微调的Base模型、增强item级对齐（IA）的Base+IA模型，以及包含我们增强推理机制（R）的完整Base+IA+R模型，如表2所示。结果表明每个组件都不可或缺：item级对齐通过创建item级token的连贯语义表示提供了基础性提升，而推理机制则带来了进一步显著增益，证实了这两个组件协同解决了序列推荐中的核心挑战。
### 5.4 工业实验
#### 5.4.1 训练设置
我们采用Qwen-8B (Yang et al., 2025)作为我们的骨干模型，从公开可用的预训练权重初始化其参数。模型的词汇表扩展了24,576个新token，表示三级层次化item级token（每级8,192个token），加上两个特殊边界token<|item_begin|>和<|item_end|>。
对我们的生产环境，我们实现了一个每日增量训练流程。模型每天在一个由80块旗舰GPU组成的集群上更新，每天处理约200亿个token，以保持与新生成的用户交互数据同步。详情见附录A.3。
#### 5.4.2 结果
**在线A/B测试结果**：我们在快手（一个拥有数亿日活跃用户的短视频平台）上部署OneRec-Think。使用1.29%的流量实验组，我们将OneRec-Think与我们的在线模型进行了一周比较，并在表3中报告结果，其中主要指标是APP停留时间（反映总用户互动时间）。主要指标APP停留时间显示出显著提升，增加了0.159%。请注意，在工业推荐系统中，0.1%的提升被认为是实质性的。此外，互动指标如视频浏览（VideoView）和转发（Forward）显示出积极趋势，表明用户参与度增强。我们在不同时间进行了多次实验，一致观察到停留时间和相关互动指标的显著提升。
**工业基准上的item级对齐消融**：我们在工业用户理解和短视频理解基准上，使用BertScore (Zhang et al., 2019)评估了item级对齐阶段的token预热（TW）和多任务集成（MI）（详情见附录A.3.2）。表4的结果揭示了每个组件的不同作用。在文本密集的用户理解任务上，TW在强基础模型上的增益有限，因为LLM可以直接有效处理丰富的文本信息，而MI通过将对齐后的表示转化为可操作的洞察带来了显著提升。相比之下，在纯item级token的短视频理解任务上，它显示出TW和MI的渐进式增益，确认了它们在解释非文本item信息方面的必要性。这些结果验证了这两个子阶段对item级对齐最终性能都有贡献。
### 5.5 案例研究
我们的案例研究展示了模型在不同场景下的复杂推理能力。在对话场景中（图3），当用户表达负面情绪时，模型检测到这一情感信号并策略性地将推荐从一般兴趣转向放松和积极的内容，展示了其通过与用户交互主动优化观看体验的能力。在基于推理的短视频推荐中（图4），模型生成多样化的推理路径，捕捉细粒度的用户偏好，如特定游戏机制和叙事模式，实现了超越粗略主题匹配的更精确推荐。此外，我们的一致性分析（图5）显示，在中间推理步骤应用波束搜索时，推理文本x和推荐item之间具有强对齐，证实了推理过程真正指导推荐生成，而非事后证明。值得注意的是，我们的模型实现了item-文本交错推理路径（图6）。通过item级token的精确内容锚定和文本token的因果表达，交错式推理提供了超越孤立模态方法的增强推荐准确性和透明解释。这些结果共同验证了模型具有真实、多方面的推理能力，展示了其适应实时交互、捕捉细粒度偏好并在多样化推荐场景中保持语义一致性的能力。
## 6 结论
我们提出了OneRec-Think，一个通过三个关键创新桥接推理能力与生成式推荐的新颖框架：层次化item级token对齐、通过CoT有监督微调的推理激活，以及基于强化的推理优化。我们的方法从根本上将推荐系统从单纯的item预测器转变为生成可解释推理过程和高质量推荐的推理感知模型。大量实验表明，OneRec-Think不仅在多个基准上取得了最先进的性能，还转化为具体的工业影响，在主要指标如APP停留时间上提升了0.15%。未来工作将聚焦于探索用户长序列建模和用于更细粒度偏好建模的密集RL奖励，进一步桥接基于LLM的推理与工业推荐系统。
## 局限性
尽管取得了有前景的实证结果，当前的公开数据集通过其有限的行为序列长度和受限的item空间，展现出质量约束。这些限制阻碍了我们的推理激活和推理增强模块获得与从工业规模数据中学习相媲美的高质量推理能力。因此，我们简化和适配了我们的方法，以实现稳定但简化的推理能力，这在公开数据集上保持稳健。为解决这些问题，我们正在积极构建一个具有扩展行为轨迹和多样化item目录的大规模基准，以实现对基于推理的推荐模型推理能力的更全面评估。
## 伦理声明
在这项工作中，我们在两种设置下进行了实验：一种是开源基准数据集，另一种是工业场景。对于开源基准数据集的实验，所有数据集均来自先前的工作或公开API，同时保持匿名性。对于工业场景，我们利用从我们的平台收集的用户交互数据来训练推荐模型。所有数据收集和使用严格遵守我们平台的隐私政策和服务条款，用户已提供明确同意。重要的是，我们的训练过程仅对聚合的行为序列、文本内容和用户基本信息进行操作，不访问或处理任何个人身份信息。
## 参考文献
- Millennium Bismay, Xiangjue Dong, and James Caverlee. 2024. Reasoningrec: Bridging personalized recommendations and human-interpretable explanations through llm reasoning. *arXiv preprint arXiv:2410.23180*.
- Ben Chen, Xian Guo, Siyuan Wang, Zihan Liang, Yue Lv, Yufei Ma, Xinlong Xiao, Bowen Xue, Xuxin Zhang, Ying Yang, and 1 others. 2025. Onesearch: A preliminary exploration of the unified end-to-end generative framework for e-commerce search. *arXiv preprint arXiv:2509.03236*.
- Yashar Deldjoo, Zhankui He, Julian McAuley, Anton Korikov, Scott Sanner, Arnau Ramisa, René Vidal, Maheswaran Sathiamoorthy, Atoosa Kasirzadeh, and Silvia Milano. 2024. A review of modern recommender systems using generative models (genrecsys). In *Proceedings of the 30th ACM SIGKDD conference on Knowledge Discovery and Data Mining*, pages 6448–6458.
- Jiaxin Deng, Shiyao Wang, Kuo Cai, Lejian Ren, Qigen Hu, Weifeng Ding, Qiang Luo, and Guorui Zhou. 2025. Onerec: Unifying retrieve and rank with generative recommender and iterative preference alignment. *arXiv preprint arXiv:2502.18965*.
- Yi Fang, Wenjie Wang, Yang Zhang, Fengbin Zhu, Qifan Wang, Fuli Feng, and Xiangnan He. 2025. Reason4rec: Large language models for recommendation with deliberative user preference alignment. *CoRR*.
- Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, and 1 others. 2025a. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. *arXiv preprint arXiv:2501.12948*.
- Xian Guo, Ben Chen, Siyuan Wang, Ying Yang, Chenyi Lei, Yuqing Ding, and Han Li. 2025b. Onesug: The unified end-to-end generative framework for e-commerce query suggestion. *arXiv preprint arXiv:2506.06913*.
- Balázs Hidasi, Alexandros Karatzoglou, Linas Baltrunas, and Domonkos Tikk. 2016. Session-based recommendations with recurrent neural networks. In *ICLR*.
- Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, and 1 others. 2022. Lora: Low-rank adaptation of large language models. *ICLR*, 1(2):3.
- Wang-Cheng Kang and Julian McAuley. 2018. Self-attentive sequential recommendation. In *ICDM*, pages 197–206. IEEE.
- Jieyong Kim, Hyunseo Kim, Hyunjin Cho, SeongKu Kang, Buru Chang, Jinyoung Yeo, and Dongha Lee. Review-driven personalized preference reasoning with large language models for recommendation. corr, abs/2408.06276, 2024. doi:10.48550. *arXiv preprint ARXIV.2408.06276*.
- Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. 2022. Large language models are zero-shot reasoners. *Advances in neural information processing systems*, 35:22199–22213.
- Lei Li, Yongfeng Zhang, Dugang Liu, and Li Chen. 2023. Large language models for generative recommendation: A survey and visionary discussions. *arXiv preprint arXiv:2309.01157*.
- Chen Ma, Peng Kang, and Xue Liu. 2019. Hierarchical gating networks for sequential recommendation. In *KDD*. ACM.
- Qiyao Peng, Hongtao Liu, Hua Huang, Qing Yang, and Minglai Shao. 2025. A survey on llm-powered agents for recommender systems. *arXiv preprint arXiv:2502.10050*.
- Shashank Rajput, Nikhil Mehta, Anima Singh, Raghunandan Hulikal Keshavan, Trung Vu, Lukasz Heldt, Lichan Hong, Yi Tay, Vinh Tran, Jonah Samost, and 1 others. 2023. Recommender systems with generative retrieval. *Advances in Neural Information Processing Systems*, 36:10299–10315.
- ByteDance Seed, Jiaze Chen, Tiantian Fan, Xin Liu, Lingjun Liu, Zhiqi Lin, Mingxuan Wang, Chengyi Wang, Xiangpeng Wei, Wenyuan Xu, and 1 others. 2025. Seed1. 5-thinking: Advancing superb reasoning models with reinforcement learning. *arXiv preprint arXiv:2504.13914*.
- Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, and 1 others. 2024. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. *arXiv preprint arXiv:2402.03300*.
- Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. 2024. Hybridflow: A flexible and efficient rlhf framework. *arXiv preprint arXiv:2409.19256*.
- Charlie Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. 2024. Scaling llm test-time compute optimally can be more effective than scaling model parameters. *arXiv preprint arXiv:2408.03314*.
- Fei Sun, Jun Liu, Jian Wu, Changhua Pei, Xiao Lin, Wenwu Ou, and Peng Jiang. 2019. Bert4rec: Sequential recommendation with bidirectional encoder representations from transformer. In *CIKM*, pages 1441–1450. ACM.
- Jiakai Tang, Sunhao Dai, Teng Shi, Jun Xu, Xu Chen, Wen Chen, Jian Wu, and Yuning Jiang. 2025. Think before recommend: Unleashing the latent reasoning power for sequential recommendation. *arXiv preprint arXiv:2503.22675*.
- Alicia Tsai, Adam Kraft, Long Jin, Chenwei Cai, Anahita Hosseini, Taibai Xu, Zemin Zhang, Lichan Hong, Ed H. Chi, and Xinyang Yi. 2024. Leveraging LLM reasoning enhances personalized recommender systems. In *Findings of the Association for Computational Linguistics: ACL 2024*, pages 13176–13188, Bangkok, Thailand. Association for Computational Linguistics.
- Wenjie Wang, Honghui Bao, Xinyu Lin, Jizhi Zhang, Yongqi Li, Fuli Feng, See-Kiong Ng, and Tat-Seng Chua. 2024a. Learnable item tokenization for generative recommendation. In *CIKM*. ACM.
- Wenjie Wang, Xinyu Lin, Fuli Feng, Xiangnan He, and Tat-Seng Chua. 2023. Generative recommendation: Towards next-generation recommender paradigm. *arXiv preprint arXiv:2304.03516*.
- Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. 2022. Self-consistency improves chain of thought reasoning in language models. *arXiv preprint arXiv:2203.11171*.
- Yidan Wang, Zhaochun Ren, Weiwei Sun, Jiyuan Yang, Zhixiang Liang, Xin Chen, Ruobing Xie, Su Yan, Xu Zhang, Pengjie Ren, and 1 others. 2024b. Content-based collaborative generation for recommender systems. In *Proceedings of the 33rd ACM International Conference on Information and Knowledge Management*, pages 2420–2430.
- Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, and 1 others. 2022. Chain-of-thought prompting elicits reasoning in large language models. *Advances in neural information processing systems*, 35:24824–24837.
- Zhipeng Wei, Kuo Cai, Junda She, Jie Chen, Minghao Chen, Yang Zeng, Qiang Luo, Wencong Zeng, Ruiming Tang, Kun Gai, and 1 others. 2025. One-loc: Geo-aware generative recommender systems for local life service. *arXiv preprint arXiv:2508.14646*.
- An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, and 1 others. 2025. Qwen3 technical report. *arXiv preprint arXiv:2505.09388*.
- Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L Griffiths, Yuan Cao, and Karthik Narasimhan. 2023. Tree of thoughts: Deliberate problem solving with large language models, 2023. URL https://arxiv.org/abs/2305.10601.
- Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, and 1 others. 2025. Dapo: An open-source llm reinforcement learning system at scale. *arXiv preprint arXiv:2503.14476*.
- Yu Yue, Yufeng Yuan, Qiying Yu, Xiaochen Zuo, Ruofei Zhu, Wenyuan Xu, Jiaze Chen, Chengyi Wang, Tian Tian Fan, Zhengyin Du, and 1 others. 2025. Vapo: Efficient and reliable reinforcement learning for advanced reasoning tasks. *arXiv preprint arXiv:2504.05118*.
- Jiaqi Zhai, Lucy Liao, Xing Liu, Yueming Wang, Rui Li, Xuan Cao, Leon Gao, Zhaojie Gong, Fangda Gu, Michael He, and 1 others. 2024. Actions speak louder than words: Trillion-parameter sequential transducers for generative recommendations. *arXiv preprint arXiv:2402.17152*.
- Junjie Zhang, Beichen Zhang, Wenqi Sun, Hongyu Lu, Wayne Xin Zhao, Yu Chen, and Ji-Rong Wen. 2025a. Slow thinking for sequential recommendation. *arXiv preprint arXiv:2504.09627*.
- Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q Weinberger, and Yoav Artzi. 2019. Bertscore: Evaluating text generation with bert. *arXiv preprint arXiv:1904.09675*.
- Yu Zhang, Shutong Qiao, Jiaqi Zhang, Tzu-Heng Lin, Chen Gao, and Yong Li. 2025b. A survey of large language model empowered agents for recommendation and search: Towards next-generation information retrieval. *arXiv preprint arXiv:2503.05659*.
- Bowen Zheng, Yupeng Hou, Hongyu Lu, Yu Chen, Wayne Xin Zhao, and Ji-Rong Wen. 2024. Adapting large language models by integrating collaborative semantics for recommendation. In *ICDE*. IEEE.
- Guorui Zhou, Jiaxin Deng, Jinghao Zhang, Kuo Cai, Lejian Ren, Qiang Luo, Qianqian Wang, Qigen Hu, Rui Huang, Shiyao Wang, and 1 others. 2025a. Onerec technical report. *arXiv preprint arXiv:2506.13695*.
- Guorui Zhou, Hengrui Hu, Hongtao Cheng, Huanjie Wang, Jiaxin Deng, Jinghao Zhang, Kuo Cai, Lejian Ren, Lu Ren, Liao Yu, and 1 others. 2025b. Onerec-v2 technical report. *arXiv preprint arXiv:2508.20900*.
---
## 附录 A
### A.1 实验设置
**基线细节**：我们将OneRec-Think与两组工作中的竞争基线进行比较：传统推荐模型和生成式推荐模型。1) BERT4Rec (Sun et al., 2019) 利用BERT的预训练语言表示来捕捉语义用户-item关系。2) HGN (Ma et al., 2019) 利用图神经网络学习用户和item表示以预测用户-item交互。3) GRU4Rec (Hidasi et al., 2016) 是一个轻量级图卷积网络模型，专注于用户和item之间的高阶连接。4) SASRec (Kang and McAuley, 2018) 采用自注意力机制捕捉用户交互历史中的长期依赖关系。5) TIGER (Rajput et al., 2023) 通过RQ-VAE引入基于码本的标识符，将语义信息量化成代码序列，用于基于LLM的生成式推荐。6) HSTU (Zhai et al., 2024) 将推荐问题重新表述为生成式建模框架内的序列转导任务，并提出了一种用于流数据的新架构。7) ReaRec (Tang et al., 2025) 通过在推荐的计算框架（inference-time computing framework）内进行隐式多步推理来增强用户表示。
**评估指标**：我们使用两个指标：top-K召回率（R@K）和NDCG（N@K），K=5和10，遵循(Rajput et al., 2023)。
**开源数据集实验细节**：我们采用Qwen3-1.7B (Yang et al., 2025) 作为骨干模型。模型的词汇表扩展了1,024个新token，代表四级层次化语义ID（每级256个token），外加两个特殊边界token`<|item_begin|>`和`<|item_end|>`。所有模型在配备旗舰GPU的服务器上训练。为在评估期间生成top-K推荐，我们采用波束宽度为10的波束搜索策略。鉴于从公开基准典型的短而稀疏的item序列中推导出稳健推理路径 $\tau$ 的固有挑战，我们策略性地使用手动构建的基于类别的CoT作为推理激活的剪枝内容，以确保稳定的语义指导。对于训练数据，我们采用先前工作的预处理技术 (Rajput et al., 2023; Wang et al., 2024a)，丢弃交互少于5的稀疏用户和item。我们考虑序列推荐设置，并使用留一法策略 (Rajput et al., 2023; Zheng et al., 2024) 来划分数据集。对于训练，我们遵循 (Kang and McAuley, 2018) 将用户历史中的item数量限制为50。
### A.2 模型展示
**短视频推荐的推理案例**：在本部分，我们展示了另一位用户的短视频推荐案例。图7展示了由我们模型生成的两个推理路径，展示了其多步兴趣推断和潜在需求识别能力。在第一种情况下，模型将用户的游戏偏好与硬件比较行为联系起来，推断出未明言的性能优化需求，最终推荐显示器分析视频。第二种情况通过将运动/军事观看历史与青春期叛逆搜索相关联，揭示了更深层的心理需求，将育儿挑战识别为核心关注点。两个例子都展示了我模型超越表面主题匹配、对用户动机进行因果推理的能力，从而实现了既满足表达兴趣又满足潜在需求的推荐。
**语义理解验证**：在语义对齐之后，模型获得了理解和用自然语言阐述itemtoken语义含义的能力。为验证这一涌现能力，我们评估了模型在没有显式训练的情况下为item级token生成描述性标题的能力。当被提示解释item级token代表什么时，模型利用学到的语义对应关系生成连贯的文本描述，准确捕捉item的特征。这表明对齐过程成功建立了真正的语义理解而非表面模式匹配，因为模型现在可以在离散item级token和其丰富的自然语言语义之间进行双向翻译。下面是Beauty数据集中展示模型item语义token标题生成能力的案例：
**案例：item级token解释1**
问答：
```
<|im_start|>system
你是一个有用的助手。<|im_end|>
<|im_start|>user
你能解释一下<|item_begin|><item_a_219><item_b_189><item_c_49><item_d_227><|item_end|>的含义吗？<|im_end|>
<|im_start|>assistant
它的标题是Manuka Honey and Manuka Oil Skin Cream，其分类是Beauty > Skin Care > Face > Cleansers。
```
真实标注：
```
"title": "Manuka Honey and Manuka Oil Skin Cream"
"categories": "Beauty > Skin Care > Face > Creams & Moisturizers > Fluids & Lotions > Lotions"
"sid": "<|item_begin|><item_a_219><item_b_189><item_c_49><item_d_227><|item_end|>"
```
**案例：item级token解释2**
问答：
```
<|im_start|>system
你是一个有用的助手。<|im_end|>
<|im_start|>user
你能解释一下<|item_begin|><item_a_8><item_b_18><item_c_86><item_d_131><|item_end|>的含义吗？<|im_end|>
<|im_start|>assistant
它的标题是25 Color Professional Camouflage Makeup Palette，其分类是Beauty > Makeup > Face > Concealer & Base。
```
真实标注：
```
"title": "Professional 15 Color Concealer Camouflage Makeup Palette",
"categories": "Beauty > Makeup > Face > Concealers & Neutralizers",
"sid": "<|item_begin|><item_a_8><item_b_18><item_c_86><item_d_131><|item_end|>"
```
### A.3 实现细节
#### A.3.1 item级对齐
**任务细节**：在此，我们首先介绍前述四种类型的任务。
1. **交错式用户画像接地（Interleaved User Persona Grounding）**。这一部分包含与丰富的自然语言文本（从静态用户画像中提取）交错排列的item级token。这个过程迫使模型在item级token与其现实世界含义之间建立稳健的映射，该映射基于用户属性、表达的兴趣和历史行为。样本数据如下所示。
**交错式用户画像接地数据**：
```
# 用户画像叙述
用户是位于北京的25-30岁男性。
# 用户画像叙述
用户近期在平台上的搜索包括："最佳太空歌剧小说"和"最新NASA发现"。
# 直播行为
他最近在"科学与科技"类别的直播中评论了5次。
# 点赞行为
他最近点赞了视频<|item_begin|><item_a_1123><item_b_5813><item_c_4212><|item_end|>，标题为"用詹姆斯·韦伯望远镜探索仙女座星系"；以及视频<|item_begin|><item_a_3421><item_b_8812><item_c_1234><|item_end|>，标题为"时间旅行的十大悖论"。
...
# 评论行为
他最近在视频<|item_begin|><item_a_5813><item_b_1123><item_c_9876><|item_end|>（一部关于黑洞的纪录片）上评论"令人难以置信的画面！"；并在视频<|item_begin|><item_a_8812><item_b_3421><item_c_5432><|item_end|>（解释费米悖论）上评论"令人脑洞大开的概念！"
...
# 关注的创作者
他关注了平台上的多个领域创作者，包括科学普及者、书评人和专注于科幻的电影评论家。
# 用户总结
主要兴趣：该用户喜欢科学，尤其是天文学。该用户还参与王者荣耀内容，表明有休闲游戏兴趣。
次要兴趣：多样化的探索包括宠物（猫）视频、传统文化和本地美食内容。
```
2. **序列偏好建模（Sequential Preference Modeling）**。模型被给予用户近期交互的序列（最多256个item），任务为预测下一个item。损失仅计算在目标item级token的token上。样本数据如下所示。
**序列偏好建模数据**：
```
<|im_start|>system
你是一个序列推荐引擎。你的任务是分析提供的用户-item交互序列，并预测用户最可能与之互动的下一个item。<|im_end|>
<|im_start|>user
用户交互历史：<|item_begin|><item_a_1024><item_b_2048><item_c_4096><|item_end|>...<|item_begin|><item_a_1234><item_b_5678><item_c_5876><|item_end|>。
预测下一个item。<|im_end|>
<|im_start|>assistant
下一个推荐item是<|item_begin|><item_a_5555><item_b_6666><item_c_7777><|item_end|>。<|im_end|>
```
3. **item级密集描述（Itemic Dense Captioning）**。该任务要求模型为给定的item级token生成文本描述，确保item级token与自然语言之间的稳健映射。损失计算在目标文本描述的token上。样本数据如下所示。
**item级密集描述数据**：
```
<|im_start|>system
你是一个专家内容分析师。给定一个item级token，生成其内容的简洁准确文本描述。<|im_end|>
<|im_start|>user
请为item级token提供描述：<|item_begin|><item_a_1357><item_b_2468><item_c_7753><|item_end|>。<|im_end|>
<|im_start|>assistant
该视频展示了一个当地美食摊位，突出传统烹饪方法和招牌菜品，如烤串，通过宣扬地方美食文化来吸引顾客。<|im_end|>
```
4. **通用语言建模（General Language Modeling）**。这部分数据包含通用语料的预训练和指令微调数据，以维持语言模型的基础能力。
为准备我们的模型进行文本内推理推荐，我们设计了一个两阶段训练课程，首先建立基础的语义理解，然后发展复杂的推荐能力。初始子阶段，token预热（Token Warm-up），专注于将item级token与其丰富的文本含义对齐。第二个子阶段，多任务集成（Multi-Task Integration），在发展模型核心推荐能力的同时保持初始语义对齐。
**token预热（Token Warm-up）**：初始和基础步骤是赋予模型对每个item级token（s_v）所代表的语义内容的深入理解。在开源基准数据集中，数据集由完整训练数据的用户画像组成。在工业场景中，整个对齐过程利用一个包含60亿token的交错式用户画像接地任务语料库，确保全面的语义接地。
在此阶段，我们冻结LLM骨干的参数，仅训练新引入的item级token的嵌入。这种聚焦方法使随机初始化的嵌入能够高效收敛到模型现有语义空间中的有意义的位点，而不会破坏其预训练知识。我们使用较高的学习率5
$$
\times
$$
10⁻⁴来仅训练新的token嵌入，这使得随机初始化的嵌入能够快速高效地收敛。
**多任务集成（Multi-Task Integration）**：在语义item词汇成功对齐后，课程的第二子阶段旨在发展模型的核心推荐能力。仅在单一目标上训练，如下一个item预测，可能导致模型逐渐将item级token视为常规的、非语义的标识符，从而失去初始接地子阶段的收益。为防止这种情况并促进更全面的理解，我们采用了多任务集成。不同任务的混合（详见表5）旨在平衡协同模式的学习与语义理解和上下文推理的强化。
对于开源基准数据集，我们训练模型的全部参数。对于工业场景，我们解冻模型骨干并使用LoRA (Hu et al., 2022) 进行参数高效微调。我们采用更保守的学习率3
$$
\times
$$
10⁻⁴并对模型进行微调，这确保了模型内部机制对推荐任务的稳定适应，同时不破坏其基础知识。
**表5：推荐增强的数据细节**
| 任务类型 | 数据占比 |
|:---|:---|
| 交错式用户画像接地 | 24.30% |
| 序列偏好建模 | 65.73% |
| item级密集描述 | 4.94% |
| 通用语言建模 | 5.03% |
#### A.3.2 推理激活
推理激活阶段采用精心编排的两子阶段训练流程，旨在逐步发展模型的推理能力，从简化上下文到复杂的噪声工业序列。
**使用剪枝上下文的自举（Bootstrapping with Pruned Contexts）**：在此子阶段，我们通过策略性地剪枝用户行为序列，仅保留最相关的交互，来构建一个高质量的推理数据集。对每个训练实例，我们采用语义相似度函数g(·,·)来提取与目标交互最相关的top-k个item（在我们的实验中设置k=10）。相似度函数利用预训练的item嵌入来计算历史item与目标之间的余弦相似度分数。使用这些剪枝后的序列，我们提示语义对齐的模型生成逐步推理轨迹，解释为什么目标item与用户表现出的偏好一致。示例如下所示。
**从剪枝上下文生成推理过程（Rationale Generation from Pruned Context）**：
```
<|im_start|>system
你是一个专家推荐分析师。给定用户的相关交互历史，提供逐步推理过程，解释为什么特定目标item与其偏好一致。<|im_end|>
<|im_start|>user
用户的相关交互历史：
<|item_begin|><item_a_1111><item_b_2222><item_c_3333><|item_end|> - 酸面团面包制作入门指南
<|item_begin|><item_a_2947><item_b_8156><item_c_3402><|item_end|> - 专业厨师刀工技巧教程
<|item_begin|><item_a_5618><item_b_3927><item_c_7084><|item_end|> - 从零开始的意大利面食谱
<|item_begin|><item_a_7293><item_b_5471><item_c_1836><|item_end|> - 厨房组织和备餐技巧
<|item_begin|><item_a_9105><item_b_6248><item_c_4519><|item_end|> - 家用搅拌机对比测评
目标item：<|item_begin|><item_a_3672><item_b_9814><item_c_2657><|item_end|> - 法式高级糕点技巧和设备指南
请解释为什么此目标item与用户的兴趣匹配。<|im_end|>
<|im_start|>assistant
推理：用户的观看历史显示对高级烹饪技巧和厨房设备有强烈兴趣，参与了酸面团烘焙、手工意大利面和搅拌机对比。用户已从基本技能进展到耗时的烹饪项目，展示了掌握传统烹饪方法的承诺。法式糕点代表了自然的下一个步骤，结合了他们的烘焙经验与技术精度。因此，用户很可能会点击主题相似的候选视频<|item_begin|><item_a_3672><item_b_9814><item_c_2657><|item_end|>。<|im_end|>
```
**从噪声序列学习推理（Learning to Reason from Noisy Sequences）**：在蒸馏得到的推理模式基础上，此子阶段训练模型将其推理能力迁移到原始、未过滤的用户行为序列。训练数据由完整的用户历史与子阶段1生成的高质量推理过程配对组成。我们采用方程5中定义的联合优化目标，同时训练模型：(1) 通过基于噪声上下文预测推理过程token来生成连贯的推理轨迹，以及 (2) 基于上下文和生成的推理过程产生准确的目标item预测。损失函数对两个分量应用等权重，确保模型在保持推荐准确性的同时学习进行上下文蒸馏。示例如下所示。
**从噪声序列推理（Reasoning from Noisy Sequences）**：
```
<|im_start|>system
你是一个智能推荐系统。给定用户的完整交互历史，为你的推荐提供推理，然后建议最合适的下一个item。<|im_end|>
<|im_start|>user
用户的完整交互历史：
<|item_begin|><item_a_1847><item_b_6392><item_c_5104><|item_end|>, <|item_begin|><item_a_9263><item_b_4571><item_c_8026><|item_end|>, <|item_begin|><item_a_1111><item_b_2222><item_c_3333><|item_end|>, <|item_begin|><item_a_5419><item_b_8736><item_c_2948><|item_end|>, <|item_begin|><item_a_3082><item_b_7145><item_c_6379><|item_end|>, <|item_begin|><item_a_2947><item_b_8156><item_c_3402><|item_end|>, <|item_begin|><item_a_7651><item_b_2493><item_c_1087><|item_end|>, <|item_begin|><item_a_5618><item_b_3927><item_c_7084><|item_end|>, <|item_begin|><item_a_8304><item_b_9672><item_c_4215><|item_end|>, <|item_begin|><item_a_7293><item_b_5471><item_c_1836><|item_end|>, <|item_begin|><item_a_4926><item_b_1358><item_c_8740><|item_end|>, <|item_begin|><item_a_9105><item_b_6248><item_c_4519><|item_end|>, <|item_begin|><item_a_1537><item_b_8904><item_c_3261><|item_end|>
应该推荐什么下一个item？请先解释你的推理。<|im_end|>
<|im_start|>assistant
<think>
用户的观看历史显示对高级烹饪技巧和厨房设备有强烈兴趣，参与了酸面团烘焙、手工意大利面和搅拌机对比。用户已从基本技能进展到耗时的烹饪项目，展示了掌握传统烹饪方法的承诺。法式糕点代表了自然的下一个步骤，结合了他们的烘焙经验与技术精度。因此，用户很可能会点击主题相似的候选视频<|item_begin|><item_a_3672><item_b_9814><item_c_2657><|item_end|>。
</think>
推荐：<|item_begin|><item_a_3672><item_b_9814><item_c_2657><|item_end|>
<|im_end|>
```
同样，对于这些任务，我们训练模型的全部参数。对于工业场景，我们解冻模型骨干并使用LoRA进行参数高效微调。我们采用学习率3
$$
\times
$$
10⁻⁴并对模型进行微调。
为量化item级对齐阶段语义接地（SG）和推荐增强（RE）的影响，我们计算了OneRec-Think在用户理解和短视频理解基准上预测与真实值之间的BertScore (Zhang et al., 2019)。表4的用户和短视频理解案例展示如下，真实值来自我们在线多模态大语言模型的准确答案。
**短视频理解基准案例**：
```
<|im_start|>system
你是一个专家内容分析师。给定一个item级token，生成其内容的简洁准确文本描述。<|im_end|>
<|im_start|>user
请为item级token提供描述：<|item_begin|><item_a_1357><item_b_2468><item_c_7753><|item_end|>。<|im_end|>
<|im_start|>assistant
# 答案
```
**用户理解基准案例**：
```
<|im_start|>system
你是一个专家内容分析师。你的任务是分析提供的用户交互历史并生成结构化的、可操作的用户画像总结。<|im_end|>
<|im_start|>user
<|item_begin|><item_a_1111><item_b_2222><item_c_3333><|item_end|> - 酸面团面包制作入门指南
<|item_begin|><item_a_2947><item_b_8156><item_c_3402><|item_end|> - 专业厨师刀工技巧教程
<|item_begin|><item_a_5618><item_b_3927><item_c_7084><|item_end|> - 从零开始的意大利面食谱
<|item_begin|><item_a_7293><item_b_5471><item_c_1836><|item_end|> - 厨房组织和备餐技巧
<|item_begin|><item_a_9105><item_b_6248><item_c_4519><|item_end|> - 家用搅拌机对比测评
...
<|im_end|>
<|im_start|>assistant
# 答案
```
#### A.3.3 推理增强
在上一阶段建立的推理能力基础上，我们采用强化学习进一步优化推理连贯性和推荐准确性。我们利用VERL框架 (Sheng et al., 2024) 进行高效优化，为分布式训练和可扩展奖励计算提供了稳健的基础设施。我们的训练使用GRPO算法并在VERL中实现Rollout-Beam奖励。训练过程中，我们对每个路径采样 |G| = 16 个CoT路径并以宽度 K = 32 进行波束搜索以计算奖励，平衡探索广度与计算效率。我们训练2个epoch，学习率为1e-5，KL散度系数 $\beta$ = 0.001，裁剪比率 $\epsilon$ = 0.2 以确保策略更新的稳定性。
#### A.3.4 系统部署："Think-Ahead"架构
在此，我们展示"Think-Ahead"架构的细节。我们的"Think-Ahead"架构将推理解耦为离线阶段（生成推理路径和初始itemtoken以捕捉用户广泛意图）和在线阶段（利用这些token作为约束前缀进行实时最终化），确保生产级延迟。
**阶段1：推理引导的前缀生成（Reasoning-Guided Prefix Generation）**
**推理路径合成（Reasoning Path Synthesis）**：对于具有交互历史 H_u = P( $s_{v_1}$ , ..., $s_{v_n}$ ) 的用户 u，我们采样T条不同的推理路径：
 $\tau$ ^{(i)} ∼ P(· | H_u; $\theta$ ) (8)
其中每条路径 $\tau$ ^{(i)}（i
$$
\in
$$
 {1, ..., T}）封装了将用户行为模式连接到潜在兴趣的不同逻辑通路。
**约束前缀生成（Constrained Prefix Generation）**：给定具有L个标识符的层次化item级token，我们执行波束搜索以生成候选前缀。这里，在我们的工业设置中，我们使用L=3。对于每条推理路径 $\tau$ ^{(i)}，我们仅解码前两个item级token，得到：
A_u^{(i)} = BeamSearch(P(ŝ_{ $v_{n+1}$ }^1, ŝ_{ $v_{n+1}$ }^2 | H_u, $\tau$ ^{(i)}; $\theta$ ), m) (9)
其中 A_u^{(i)} 表示从路径 $\tau$ ^{(i)} 为用户 u 导出的m个候选item前缀的集合。
**语义空间实例化（Semantic Space Materialization）**：所有item前缀集合的并集形成了用户个性化候选空间：
C_u = ⋃_{i=1}^{T} A_u^{(i)} (10)
该集合包含 T
$$
\times
$$
 m 个高潜力item前缀，缓存在工业级分布式存储系统中，有效将模型的推理性推理实例化为可操作的语义假设。
**阶段2：前缀约束的最终化（Prefix-Constrained Finalization）**
在线阶段在用户请求到达时执行快速推理，同时利用阶段1的预计算语义先验。
**约束解码（Constrained Decoding）**：收到用户u的请求后，我们检索 C_u 并使用实时更新的OneRec（遵循(Zhou et al., 2025a)） $h_{online}$ 推断最后一个token。搜索空间被限制为候选目标item的前缀 (ŝ_{ $v_{n+1}$ }^1, ŝ_{ $v_{n+1}$ }^2) 必须在候选前缀集合中，即 (ŝ_{ $v_{n+1}$ }^1, ŝ_{ $v_{n+1}$ }^2)
$$
\in
$$
 C_u。因此，我们可以构建解码目标：
ŝ_{ $v_{n+1}$ } = argma $x_{s_{v_{n+1}$ }} $P_{h_{online}$ }( $s_{v_{n+1}$ } | $s_{v_1}$ , ..., $s_{v_n}$ )  s.t. (ŝ_{ $v_{n+1}$ }^1, ŝ_{ $v_{n+1}$ }^2)
$$
\in
$$
 C_u (11)
然后将具有top-K概率的item提供给用户。
总之，我们的Think-Ahead架构为工业推荐提出了一种新颖的范式，有效协调了复杂推理与严格延迟要求。
---
## 表格
**表1：基线方法与OneRec-Think在三个数据集上的整体性能比较。加粗结果突出最佳结果，次佳结果加下划线。**
| 数据集 | 方法 | BERT4Rec | HGN | GRU4Rec | SASRec | TIGER | HSTU | ReaRec | **OneRec-Think** |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| **Beauty** | R@5 | 0.0232 | 0.0319 | 0.0395 | 0.0402 | 0.0405 | 0.0424 | 0.0450 | **0.0563** |
| | R@10 | 0.0396 | 0.0536 | 0.0584 | 0.0607 | 0.0623 | 0.0652 | 0.0704 | **0.0791** |
| | N@5 | 0.0146 | 0.0196 | 0.0265 | 0.0254 | 0.0267 | 0.0280 | 0.0262 | **0.0398** |
| | N@10 | 0.0199 | 0.0266 | 0.0326 | 0.0320 | 0.0337 | 0.0353 | 0.0344 | **0.0471** |
| **Sports** | R@5 | 0.0102 | 0.0183 | 0.0190 | 0.0199 | 0.0215 | 0.0268 | 0.0214 | **0.0288** |
| | R@10 | 0.0175 | 0.0313 | 0.0312 | 0.0301 | 0.0347 | 0.0343 | 0.0332 | **0.0412** |
| | N@5 | 0.0065 | 0.0109 | 0.0122 | 0.0106 | 0.0137 | 0.0173 | 0.0116 | **0.0199** |
| | N@10 | 0.0088 | 0.0150 | 0.0161 | 0.0141 | 0.0179 | 0.0226 | 0.0154 | **0.0239** |
| **Toys** | R@5 | 0.0215 | 0.0326 | 0.0330 | 0.0448 | 0.0337 | 0.0366 | 0.0523 | **0.0579** |
| | R@10 | 0.0332 | 0.0517 | 0.0490 | 0.0626 | 0.0547 | 0.0566 | 0.0764 | **0.0797** |
| | N@5 | 0.0131 | 0.0192 | 0.0228 | 0.0300 | 0.0209 | 0.0245 | 0.0298 | **0.0412** |
| | N@10 | 0.0168 | 0.0254 | 0.0279 | 0.0358 | 0.0276 | 0.0309 | 0.0376 | **0.0482** |
**表2：OneRec-Think不同变体在Beauty数据集上的消融研究。**
| 训练方法 | R@5 | R@10 | N@5 | N@10 |
|:---|:---|:---|:---|:---|
| Base | 0.0460 | 0.0654 | 0.0314 | 0.0377 |
| Base+IA | 0.0532 | 0.0735 | 0.0342 | 0.0402 |
| Base+IA+R | **0.0563** | **0.0791** | **0.0398** | **0.0471** |
**表3：在线A/B测试在短视频推荐场景上的相对提升。**
| 在线指标 | OneRec-Think |
|:---|:---|
| APP停留时间 | +0.159% |
| 观看时长 | +0.169% |
| 视频浏览 | +0.150% |
| 关注 | +0.431% |
| 转发 | +0.758% |
| 点赞 | +0.019% |
| 收藏 | +0.098% |
**表4：用户理解和短视频理解基准的BertScore。**
| 基准 | Qwen3 | Qwen3+TW | Qwen3+TW+MI |
|:---|:---|:---|:---|
| 用户理解 | 0.6588 | 0.6492 | **0.7053** |
| 短视频理解 | 0.6031 | 0.6443 | **0.7300** |
---
## 图表描述
**图1：OneRec-Think统一对话、推理和推荐框架的示例。** 展示了模型如何根据用户对话、描述和推理进行交互式推荐，包括提示"描述视频"、"描述用户"和"思考与推荐"，以及不同rollout阶段的推理和推荐结果。
**图2：OneRec-Think的框架图。** 第一阶段，通过多任务预训练实现item级语义对齐。第二阶段，通过提示模型生成偏好推理过程来激活显式推理。第三阶段，通过基于推荐定制奖励的RL来细化推理路径。
**图3：上下文感知推荐适配的演示：** 我们的模型根据用户指令动态将推荐转向放松内容。用户说"我现在心情不好，请给我推一些轻松的内容"，模型通过推理后推荐了动画电影、宠物日常、自然风光等轻松视频。
**图4：细粒度兴趣推理的演示，** 展示了从用户行为分析到可解释推荐的端到端过程。三个案例分别展示了游戏、校园生活和二次元文化等不同兴趣的推理路径。
**图5：模型在不同推理步骤使用波束搜索时的推理过程演化，** 从广泛的兴趣匹配（左）到细粒度主题规约（中），推荐结果（右）显示与每个推理步骤的语义一致性。
**图6：item-文本交错推理的演示。** 展示了用户兴趣推理输出中itemtoken和文本token交错出现的模式，以及对应的英文推理内容。
**图7：短视频推荐场景的推理示例。** 第一个案例展示从游戏和硬件兴趣到显示器推荐的推理链；第二个案例展示从体育/军事内容到青春期叛逆沟通问题的深层需求推理。