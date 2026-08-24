# CIRS：通过反事实交互式推荐系统打破过滤气泡

> Chongming Gao, University of Science and Technology of China, China; Shiqi Wang, Chongqing University, China; Shijun Li, University of Science and Technology of China, China; Jiawei Chen*, Zhejiang University, China; Xiangnan He*, University of Science and Technology of China, China; Wenqiang Lei, Sichuan University, China; Biao Li, Kuaishou Technology Co., Ltd., China; Yuan Zhang, Kuaishou Technology Co., Ltd., China; Peng Jiang, Kuaishou Technology Co., Ltd., China

本文提出 CIRS（反事实交互式推荐系统），通过将因果推理融入离线强化学习来解决过滤气泡问题，**在交互式推荐中首次建模过度曝光效应对用户满意度的影响**。核心发现是——**CIRS 通过在追求高单轮满意度和维持长交互序列之间找到平衡，实现了最优的累积用户满意度**。

核心内容：

- **问题/痛点**：个性化推荐系统存在过滤气泡问题——持续推荐用户感兴趣的内容反而会导致用户厌倦和满意度下降；现有方法在静态推荐设置中难以捕捉过度曝光的动态效应
- **方案/创新点**：提出 CIRS 框架，将因果推理与离线强化学习结合；先学习因果用户模型捕捉过度曝光效应，再用该模型为 RL 策略提供反事实满意度作为奖励信号
- **技术细节**：因果用户模型将用户满意度分解为两条因果路径——内在兴趣路径和过度曝光效应路径；通过因果干预 $do(E_t = e_t^*)$ 在 RL 规划阶段计算反事实满意度
- **实验验证**：在 VirtualTaobao 和 KuaiEnv 两个环境中验证有效性；基于快手真实数据创建 KuaiEnv 环境，包含99.6%密度的用户-视频评分矩阵

关键发现：

- **CIRS 在两个环境中均实现最优累积满意度**，在 VirtualTaobao 上超越 CIRS w/o CI 约15%，在 KuaiEnv 上超越约20%
- **用户满意度确实会因过度曝光而下降**：快手实地研究显示，1小时内观看同类视频数量从5增加到50时，视频观看比率从0.70降至0.60
- **学习到的用户敏感度 $\alpha_u$ 与用户活跃度正相关**，物品不可忍耐度 $\beta_i$ 与物品流行度正相关
- **CIRS w/o CI（无因果推理）策略不稳定**，性能随训练轮次增加而退化，证明了因果推理模块的有效性

---

## 摘要

虽然个性化提高了推荐系统的效用，但也带来了过滤气泡的问题。例如，如果系统持续曝光和推荐用户感兴趣的物品，也可能使用户感到厌倦和不满意。现有工作在静态推荐中研究过滤气泡，其中过度曝光的效应难以捕捉。相比之下，我们认为在交互式推荐中研究这个问题并优化长期用户满意度更有意义。然而，由于高成本，在线训练模型是不现实的。因此，我们必须利用离线训练数据并分解对用户满意度的因果效应。

为了实现这一目标，我们提出了一个反事实交互式推荐系统（CIRS），它用因果推理增强了离线强化学习（离线 RL）。基本思想是首先在历史数据上学习因果用户模型，以捕捉物品对用户满意度的过度曝光效应。然后使用学习到的因果用户模型来帮助 RL 策略的规划。为了离线进行评估，我们创新性地基于真实世界完全观察到的用户评分数据集创建了一个真实的 RL 环境（KuaiEnv）。实验表明 CIRS 在打破过滤气泡和在交互式推荐中实现长期成功方面是有效的。CIRS 的实现可通过 https://github.com/chongminggao/CIRS-codes 获取。

**CCS Concepts**：Information systems → Recommender systems; Personalization; Theory of computation → Sequential decision making.

**Additional Key Words and Phrases**：Filter bubble, Interactive recommendation, Causal inference, Offline reinforcement learning

## 1. 引言

推荐系统已经深刻影响了我们的生活。它们改变了检索信息的方式，从费力搜索转变为通过精确个性化便捷获取。系统通常通过学习收集的用户行为数据并选择用户潜在喜欢的产品来实现个性化。随着时间推移和数据积累，推荐器逐渐成为反映每个用户兴趣的镜子，并将推荐列表缩小到用户兴趣最大的物品。然而，这是有代价的。在享受精确个性化推荐的同时，用户不得不面对信息种类萎缩的事实。当用户与偏离其主导偏好的信息隔离时，他们就陷入了所谓的"过滤气泡"。

过滤气泡在推荐系统中很常见。最近的研究在大规模推荐系统中进行了广泛实验，发现过滤气泡有两个主要原因。从用户角度来看，偏好多样性较低的用户更容易陷入气泡。从系统角度来看，学习会导致强调用户的主导兴趣。此外，系统通常假设用户满意度等于内在兴趣——即使用户喜欢的物品已经被过度曝光，它假设用户满意度保持不变，这是不恰当的。我们认为推荐算法的过度利用行为是过滤气泡的主要原因，我们将使用过度曝光效应作为评估过滤气泡的代理。

过度曝光物品对用户满意度有有害影响，即使用户对推荐的物品感兴趣。例如，一个喜欢跳舞的用户在收到关于跳舞歌曲的推荐时可能会感到满意。然而，在几十次不间断的跳舞歌曲推荐后，她可能会感到厌倦并因此拒绝选择它。因此，为推荐系统打破过滤气泡对于最大化用户满意度具有重要意义。关键是分解对用户满意度的因果效应，即建模用户的内在兴趣与过度曝光效应如何共同影响用户的最终满意度。

现有方法用两种策略解决过滤气泡：(1) 帮助用户提高对多样化社会观点的认识，(2) 使模型增强推荐结果的多样性、惊喜性或公平性。然而，这些策略大多是启发式的，专注于静态推荐设置。它们不能从根本上解决问题，因为它们没有通过建模过度曝光效应来直接考虑过滤气泡的主要原因。

在这项工作中，我们专注于动态环境中的交互式推荐系统（IRS）。IRS 被表述为一个序贯决策过程，允许跟踪和建模动态和实时的过度曝光效应。图1(a) 展示了交互式推荐的示例，其中模型基于交互上下文（即反映用户信息和交互历史的状态 $s$）向用户推荐物品（即做出动作 $a$），然后接收用户反馈（即代表用户满意度的奖励 $r$）。交互过程重复直到用户退出。模型将更新其策略 $\pi_\theta$，目标是最大化整个交互过程中的累积满意度。为了实现这一目标，IRS 应该避免过度曝光物品，因为用户会感到厌倦，他们的满意度在过滤气泡中会下降（图1(b)）。

尽管这个想法很有吸引力，但它面临一个不可避免的挑战：很难通过实时反馈在线学习 IRS。原因有两方面：(1) 对于模型，在线训练策略会增加学习时间和部署复杂性；(2) 对于用户，与不成熟的系统交互会损害满意度。因此，有必要在为用户在线服务之前，在历史日志上离线训练 IRS。为此，我们需要对离线数据进行因果推理，以分解用户兴趣和过度曝光的因果效应。这为在服务阶段回答反事实问题提供了机会："如果物品已经被过度曝光，用户还会对感兴趣的物品感到满意吗？"——如果答案是否定的，那么一旦推荐该物品就会出现过滤气泡，所以我们不应该推荐它。

我们提出了一个反事实交互式推荐系统（CIRS）来实现这一目标。它用因果推理增强了离线强化学习（离线 RL）。图1(c) 展示了其学习框架，包含三个循环步骤：1) 学习因果用户模型以捕捉用户兴趣和物品过度曝光效应，2) 使用学习到的用户模型提供反事实满意度（即由因果用户模型给出的奖励，而不是即时用户反馈）来规划 RL 策略，3) 在真实环境中评估 RL 策略的累积用户满意度。

此外，我们提出了一个交互式推荐环境 KuaiEnv 用于评估问题。它基于 KuaiRec 数据集创建，该数据集包含从快手 App 收集的完全填充的用户-物品矩阵（密度：99.6%）。在这个矩阵中，IRS 可以更忠实地被评估，因为没有缺失值。为了有效反映过滤气泡的效应，我们进一步在 VirtualTaobao 和 KuaiEnv 中添加了"感到厌倦然后退出"的退出机制来模拟真实用户的反应。实验表明，所提出的 CIRS 可以在离线学习中捕捉过度曝光效应，并与其他基线相比实现更大的累积满意度。

我们的贡献总结如下：
- 据我们所知，这是第一项在交互式推荐中解决过滤气泡的工作，其中过滤气泡可以通过建模过度曝光效应对用户满意度的影响来更自然地观察和评估。
- 我们通过将因果推理集成到离线 RL 中来解决问题。我们是第一个在交互式推荐系统中结合因果推理和离线 RL 的。
- 我们在快手 App 上进行实证研究，验证过度曝光物品确实会伤害用户体验，并展示了所提出的方法可以打破过滤气泡并增加累积满意度。

## 2. 相关工作

我们从三个角度简要回顾相关工作：过滤气泡、因果推理和推荐中的离线 RL。

### 2.1 推荐中的过滤气泡

"个性化过滤提供了一种看不见的自动宣传，用我们自己的想法灌输我们，放大我们对熟悉事物的欲望，让我们对潜伏在未知黑暗 territory 中的危险视而不见。"
——Eli Pariser, The Filter Bubble

术语"过滤气泡"由互联网活动家 Eli Pariser 创造，用于描述算法个性化过程导致的智力隔离状态。Bruns 广泛讨论了过滤气泡的原因和影响。他声称在某些情况下过滤气泡可能不是真实的，相反，它只是被用作替罪羊，通过责备技术来避免解决根深蒂固的问题。他还从心理学角度讨论了过滤气泡，并描述了其对社会的政治影响。

McKay 等人通过采访从各种来源招募的18人关于他们改变对重要问题看法的经历来调查过滤气泡。他们发现了一个意想不到的结果，用户更有可能接触不同意的信息，而不是与思想更封闭的人互动。这些用户通常是被动遇到而不是主动寻求不同意的信息。这强调了向用户分发信息的推荐系统的重要性。Flaxman 等人对在线新闻消费进行了广泛研究以调查过滤气泡。他们发现大多数消费行为模仿了传统阅读习惯，即人们喜欢访问主页并被动接收信息。

在本文中，我们专注于推荐系统背景下的过滤气泡，推荐系统决定了向用户曝光的内容和信息。在推荐器中，当用户陷入过滤气泡时，算法将向用户展示符合其先有信念或兴趣的有限信息。

在推荐中，许多工作系统分析了过滤气泡的原因。通常有两个原因。第一个原因可能是用户的个人意图。例如，Liu 等人对新闻推荐进行了模拟研究，发现偏好更极端的用户更容易陷入过滤气泡。此外，一些研究人员在 YouTube（最流行的视频共享平台）上调查用户如何以及为什么陷入错误信息过滤气泡，即相信不准确信息或具有争议话题的极端内容。Hussein 等人进行了大规模审计实验，调查个性化（即年龄、性别、地理位置或观看历史）是否有助于放大错误信息。他们的结果揭示了这些个性化人口统计不会显著放大新用户的气泡。然而，一旦这些用户发展了错误信息的观看历史，这些人口统计就会产生影响。这种现象将我们带到过滤气泡的第二个原因。

第二个原因是推荐算法的过度利用行为，导致某些物品子集的过度曝光。最近的实证研究验证了过滤气泡的形成主要是由于模型强调具有主导用户兴趣的某些物品。Herlocker 等人证明，推荐已经熟悉的物品会触发用户的不满意结果，因为用户喜欢新颖性和惊喜性。

改善推荐系统中的过滤气泡是必要的。旨在对抗过滤气泡的现有方法专注于提高用户对多样化社会认识、增强模型的多样性、惊喜性、推荐结果的公平性、通过观看辟谣内容纠正模型行为。然而，这些策略专注于静态推荐设置，其中过滤气泡的动态性质难以建模。相比之下，我们捕捉过度曝光效应（即过滤气泡的主要症状或代理），并在交互式推荐中解决问题。

### 2.2 因果增强推荐

最近，因果推理（CI）在自然语言处理（NLP）、计算机视觉（CV）和推荐系统（RS）中引起了广泛关注。CI 不是通过将数据馈送到黑盒神经网络来利用输入和输出之间的相关关系，而是显式建模变量之间的因果机制。在推荐系统中，CI 可以是解决数据或学习过程中各种偏差的强大工具，例如选择偏差和流行度偏差。估计缺失值的最朴素方法是直接方法（DM）或基于错误插补（EIB）的估计器，其思想是使用超参数 $\gamma$ 来插补所有缺失值。因此我们可以使用 $\gamma$ 而不是零来学习这些位置的方法。这是一个非常粗粒度的解决方案。或者，许多研究尝试基于逆概率评分（IPS）估计无偏用户偏好，这是一种有效的基于 CI 的方法。直觉上，当用户看到物品的概率较低时，我们应该为该样本分配更大的重要性，反之亦然。然而，基于 IPS 的因果方法存在高方差问题，且难以估计倾向分数。因此，有方法结合 EIB 和 IPS 并提出双重稳健估计器，然后展示了有效性。然而，这些方法有一个共同问题：很难获得精确的估计器，即 EIB 中的超参数 $\gamma$ 和 IPS 中的倾向分数。

最近，研究人员在推荐系统中遵循 Pearl 的因果推理框架。通常，他们将变量的关系组织为三角结构：原因、效应和混杂因子。由于混杂因子的存在，原因和效应之间存在虚假关系。因此，我们需要在推理阶段切断路径并移除混杂因子的效应。例如，Sato 等人将用户和物品的特征视为混杂因子，并通过根据特征重新加权样本来推导无偏推理。

另一项工作将推理问题表述为反事实学习框架，其中反事实变量指的是未观察数据的潜在结果。通过显式计算潜在结果，我们可以得出治疗是否有效，即与什么都不做相比，进行治疗是否产生巨大差异。

在这项工作中，我们通常将大多数工作的程序总结如下：1) 构建因果图，即描述相关变量之间因果关系的结构因果模型（SCM）的表示。2) 在学习阶段，基于提出的因果图在训练数据集上拟合无偏模型（例如实现为神经网络）。3) 在推理阶段，根据特定要求主动改变某些变量（称为干预），然后预测目标变量的无偏结果。在这项工作中，我们使用这个框架来显式建模推荐系统在过滤气泡问题中的过度曝光效应。

### 2.3 推荐中的离线 RL

推荐系统旨在在为在线用户服务时提高用户满意度或增加销售额。我们通常将推荐视为一个序贯决策过程，其中推荐策略根据先前的用户反馈决定进行推荐。关于静态推荐模型（例如 DeepFM 和 LightGCN）的常见研究只关注提高单轮推荐的性能。然而，这种解决方案通常假设推荐遵循独立同分布（I.I.D.）假设，即每个物品或推荐是独立同分布的，这在许多情况下是不正确的。例如，在 slate 推荐或捆绑推荐中，物品的转化率不仅仅取决于自身。如果它被类似但更昂贵的物品包围，转化率可能会增加。这种现象被称为诱饵效应。此外，许多物品具有未来影响而不是即时奖励，例如推荐高价物品可能不会导致即时消费行为，但它可以给用户留下印象，即该平台有高质量物品，当用户有能力消费昂贵物品时，这种印象可以在未来导致交易。过滤气泡的过度曝光效应也会对用户体验产生影响，但这种影响是有害的，即它可以负面影响用户继续使用和信任该推荐系统的意愿。

为了直接建模多轮决策问题中的长期成功，我们可以采用基于强化学习（RL）的交互式推荐系统（IRS）。强化学习的目标是最大化累积奖励 $J(\pi_\theta)$：

$$
J(\pi) = \mathbb{E}_{\tau \sim p_\pi(\tau)} \left[ \sum_{t=0}^{H} \gamma^t r(s_t, a_t) \right]
$$

其中 $r(s_t, a_t)$ 是当代理在状态 $s_t$ 做出动作 $a_t$ 时从环境给出的奖励，$\gamma \in (0, 1]$ 是标量折扣因子。轨迹 $\tau$ 是长度为 $H$ 的状态和动作序列，由 $\tau = (s_0, a_0, \ldots, s_H, a_H)$ 给出。$p_\pi(\tau)$ 是给定马尔可夫决策过程（MDP）的轨迹分布，策略 $\pi$ 写为：

$$
p_\pi(\tau) = d_0(s_0) \prod_{t=0}^{H} \pi(a_t | s_t) T(s_{t+1} | s_t, a_t)
$$

其中 $T(s_{t+1} | s_t, a_t)$ 是描述环境系统动态的状态转移概率。

尽管有效，但与在线用户学习 IRS 策略是不切实际的，因为它太慢且会伤害用户体验。另一方面，记录的推荐日志和离线用户反馈更容易获得。因此，很自然地想到在离线数据上学习策略，这是离线 RL 的核心思想。

推荐中常用的离线 RL 策略包括离策略评估（OPE）学习和基于模型的 RL 方法。基于 OPE 的方法使用来自日志策略 $\pi_\beta$ 的加权数据估计目标策略 $\pi_\theta$ 的累积奖励。经典的逆概率估计器如下：

$$
J(\pi_\theta) = \mathbb{E}_{\tau \sim \pi_\beta(\tau)} \left[ \frac{\pi_\theta(\tau)}{\pi_\beta(\tau)} \sum_{t=0}^{H} \gamma^t r(s, a) \right]
$$

然而，基于 OPE 的方法中的估计器通常由于 $w_t^i$ 中两个分布之间的差异而遭受高方差问题。

基于模型的方法尝试估计转移函数 $T(s_{t+1} | s_t, a_t)$ 和环境的奖励函数 $r_t$。从估计的模型中，RL 代理可以根据预测的状态和奖励相应地规划其轨迹，而不是严格地在历史轨迹上学习。基于模型的方法可以避免基于 OPE 的方法中的高方差问题，但它在估计模型时存在偏差问题。在推荐系统中，各种偏差被指定和研究。在这项工作中，我们将偏差指定为模型是否考虑用户在交互中感到厌倦。

如上所述，CI 是解决推荐系统中偏差问题的强大工具。因此，我们将基于模型的离线 RL 和 CI 技术相结合，开发了一个无偏 IRS，可以在离线数据上识别和解决过滤气泡问题。我们选择使用基于模型的离线 RL，因为它在样本效率方面具有强大优势，这在数据高度稀疏且收集成本高昂的推荐系统中至关重要。此外，我们可以通过因果增强模型直接在提取的转移概率和奖励函数中建模偏差，这也是这项工作的主要贡献。

在这里，我们在表1中简要总结了关于三个维度的六种推荐器类型：(1) 系统是否显式构建试图捕捉真实用户偏好的用户模型，(2) 系统是否考虑去偏差，(3) 系统是否有基于 RL 的策略。注意，还有另外两项工作与我们的 CIRS 属于同一类别，但它们不是为过滤气泡问题设计的。此外，它们的无偏性不是指在推荐问题中解决某种偏差效应的能力，而是使估计更准确。

## 3. 前提条件

在本节中，我们介绍问题定义和来自快手的真实数据的实证分析。

### 3.1 问题定义

我们将用户集表示为 $\mathcal{U}$，物品集表示为 $\mathcal{I}$。用户 $u \in \mathcal{U}$ 的所有交互序列集可以表示为 $\mathcal{D}_u = \{S_u^1, S_u^2, \cdots, S_u^{|\mathcal{D}_u|}\}$。每个 $S_u^k \in \mathcal{D}_u$ 是第 $k$ 个交互序列（即轨迹），记录完整的交互过程：$S_u^k = \{(u, i_l, t_l)\}_{1 \leq l \leq |S_u^k|}$，其中用户 $u$ 在时间 $t_1$ 开始与系统交互并在时间 $t_{|S_u^k|}$ 退出，$i_l \in \mathcal{I}$ 是在时间 $t_l$ 推荐的物品。令 $\mathbf{e}_u \in \mathbb{R}^{d_u}$ 和 $\mathbf{e}_i \in \mathbb{R}^{d_i}$ 分别是用户 $u$ 和物品 $i$ 的特征表示向量。对于系统，任务是根据用户偏好和交互历史向用户推荐物品。这个过程可以被表述为一个强化学习问题，其关键组件总结如下：

- **环境**：代理在环境中工作，状态和奖励在其中生成。这里，环境是用户（在线真实用户或模拟用户），可以选择对系统推荐的物品进行评分或退出交互。
- **状态**：系统在时间 $t$ 维护的状态 $s_t \in \mathbb{R}^{d_s}$ 被视为表示用户 $u$ 与系统在 $t$ 之前所有历史交互信息的向量。在本文中，我们使用 Transformer 模型获得 $s_t$。
- **动作**：系统在时间 $t$ 做出的动作 $a_t$ 是向用户 $u$ 推荐物品。令 $\mathbf{e}_{a_t} \in \mathbb{R}^{d_a}$ 表示动作 $a_t$ 的表示向量。在本文中，每个动作 $a_t$ 只推荐一个物品 $i$。因此我们有 $\mathbf{e}_{a_t} = \mathbf{e}_i$。
- **奖励**：用户 $u$ 返回反馈作为奖励分数 $r_t$，反映其在收到推荐物品 $i$ 后的满意度。奖励也可以是由因果用户模型 $\phi_M$ 预测的反事实满意度，而不是真实用户的反馈。
- **状态转移**：在代理做出动作 $a_t$ 且用户给出奖励 $r_t$ 后，状态 $s_t$ 将根据状态转移概率 $T(s_{t+1} | s_t, a_t)$ 更新为 $s_{t+1}$。在我们的工作中，这个转移概率由基于 Transformer 模型实现的状态跟踪器建模。
- **策略**：系统的关键任务是优化目标策略 $\pi_\theta = \pi_\theta(a_t | s_t)$，表示在状态 $s_t$ 条件下做出动作 $a_t$ 的概率。它决定如何生成物品 $i$ 进行推荐，通常实现为全连接神经网络。

为了充分利用历史数据，我们将 CIRS 方法基于离线 RL 框架。学习 CIRS 需要三个循环步骤（如图3中的三个色块所示）：

(1) 在历史交互数据 $\{(u, i, r)\}$ 上训练因果用户模型 $\phi_M$，以估计用户兴趣和物品过度曝光效应。

(2) 使用学习到的因果用户模型 $\phi_M$（而不是真实用户）训练策略 $\pi_\theta$。在每个交互循环中，$\phi_M$ 采样用户 $u$ 与 $\pi_\theta$ 交互。当 $\pi_\theta$ 做出动作 $a_t$（推荐 $i$）时，$\phi_M$ 提供反事实满意度作为奖励 $r$。直觉上，如果 $\pi_\theta$ 之前做过类似推荐，$\phi_M$ 会缩减奖励 $r$。

(3) 将学习到的策略 $\pi_\theta$ 服务给真实用户并在交互环境中评估结果。当交互结束时，我们保存日志 $\{u, i, r, t\}$ 到历史数据以用于未来学习。

这三个步骤可以重复进行以持续改进 $\phi_M$ 和 $\pi_\theta$。

### 3.2 过度曝光效应对用户满意度的实地研究

如上所述，除了用户的责任外，过滤气泡的原因是模型不断向用户推荐类似物品。我们认为这是有害的，因为用户在这种情况下可能不会感到舒适。因此，我们提出以下假设：

**假设1**：如果推荐物品（或类似物品）在短时间内被重复曝光和推荐，用户满意度将下降。

为了验证这一假设，我们在快手（一个视频共享移动 App）的真实数据上进行了实证研究。我们从平台的视频观看用户池中选择了7,176名用户。我们过滤并收集了他们从2020年8月5日到2020年8月11日的交互历史。总共有34,215,294次观看，观看了3,110,886个视频。每个物品至少有一个最多四个类别标签，总共有31个类别标签。

在观看视频时，用户可以选择随时退出观看，向下滚动到下一个视频或离开视频播放界面。每个视频都有评论区，用户可以通过点击"评论"按钮进入。如果用户对视频感兴趣，他们会停留更长时间观看或进入评论区。因此，我们设计了两个关键指标来反映用户满意度：一个是在评论区停留的时间，另一个是视频观看比率，即观看时间与视频总长度的比率。

为了展示物品曝光如何影响用户满意度，我们研究了两个指标如何随过度曝光效应程度变化。具体来说，我们根据以下标准对所有收集的观看进行分组：(a) 1小时内观看的相同标签视频数量，或 (b) 现在（观看此视频）与上次观看相同标签视频之间的时间间隔。然后我们计算每组中提到指标的平均值。结果如图2所示。发现了两个观察结果：

**观察1**：当系统在最近推荐中增加类似物品数量时，用户对推荐物品的满意度下降。

**观察2**：随着两个类似物品之间的时间间隔缩短，用户对推荐物品的满意度下降。

结果具有统计显著性，因为即使最小的组也包含足够的点：图2(a) 中 $x = 50$ 的组有31,277个点，图2(b) 中 $x = (7h \sim 8h)$ 的组有76,303个点。

在我们之前的工作中，实证分析也证明了类似现象：随着物品或类别重复率的增加，用户满意度下降。因此，假设1被实证证明。这表明过滤气泡（即推荐算法的过度曝光效应）确实对用户满意度有有害影响。因此，我们可以在构建的环境中设计"感到厌倦然后退出"退出机制作为用户行为的反映，这使我们能够有效模拟过滤气泡的效应。我们将在第5节中说明。

## 4. 提出的方法

在本节中，基于实地研究中获得的观察，我们提出了一个反事实交互式推荐系统（CIRS），它在离线 RL 中利用因果推理。我们首先介绍 CIRS 在离线 RL 框架中的三个主要模块，然后描述如何利用因果推理分解对用户满意度的因果效应。

### 4.1 基于离线 RL 的框架

我们将 CIRS 模型基于离线 RL，其中我们可以利用大量离线数据训练交互式推荐系统。CIRS 的框架如图3所示。它包含三个阶段（在三个色块中显示）：预学习阶段、RL 规划阶段和 RL 评估阶段。这三个阶段的功能对应于：(1) 通过监督学习预学习用户模型 $\phi_M$，(2) 使用学习到的用户模型 $\phi_M$ 通过提供反事实满意度作为奖励来学习 RL 策略 $\pi_\theta$，(3) 在真实环境中评估策略 $\pi_\theta$。特别是，真实环境可以是真实的在线环境或可以反映真实用户行为的模拟环境。接下来，我们分别介绍 CIRS 中的三个主要组件：因果用户模型 $\phi_M$、状态跟踪器模块和基于 RL 的交互策略 $\pi_\theta$。

#### 4.1.1 因果用户模型

因果用户模型 $\phi_M$ 基于历史数据学习用户兴趣，并为 RL 规划阶段提供反事实满意度 $r$。它旨在通过正确建模物品过度曝光对用户满意度的影响来显式分解因果效应。用户模型中有两个子模块：用于计算用户内在兴趣 $y$ 的兴趣估计模块，以及捕捉过度曝光效应如何影响用户满意度 $r$ 的反事实满意度估计模块。该组件的细节将在第4.2节中报告，我们将正式介绍推荐器的因果视图。

#### 4.1.2 基于 Transformer 的状态跟踪器

交互式推荐中的状态 $s_t$ 应包括时间 $t$ 用于策略 $\pi_\theta$ 做出决策的所有关键信息。这包括：用户 $u$ 的特征向量 $\mathbf{e}_u$、最近推荐物品的特征向量（即交互循环中系统的动作 $\{\mathbf{e}_{a_1}, \cdots, \mathbf{e}_{a_t}\}$）以及用户对它们的反馈 $\{r_1, \cdots, r_t\}$。为了自动从这些向量中提取关键信息，我们使用 Transformer 模型推导 $s_t$，如图3所示。Transformer 是一种具有注意力机制的最先进序列到序列模型，可以捕捉当前输入和先前输入序列之间的依赖关系。在这项工作中，我们只使用 Transformer 的两层编码器。由于输入是顺序生成的，我们需要添加掩码以防止未来信息泄漏到序列的每个状态。

我们进一步使用门机制过滤来自动作 $\mathbf{e}_{a_t}$ 和用户反馈 $r_t$ 的信息。因此，Transformer 在时间 $t$ 的输入是 $\mathbf{e}_{a_t}' := \mathbf{g}_t \odot \mathbf{e}_{a_t}$。其中 "$\odot$" 表示逐元素乘积，门向量 $\mathbf{g}_t$ 计算为：

$$
\mathbf{g}_t = \sigma \left( \mathbf{W} \cdot \text{Concat}(r_t, \mathbf{e}_{a_t}) + \mathbf{b} \right)
$$

其中 $\mathbf{W} \in \mathbb{R}^{d_s \times (1 + d_a)}$ 和 $\mathbf{b} \in \mathbb{R}^{d_s}$ 分别指权重矩阵和偏置向量。$\text{Concat}(\cdot)$ 是向量连接运算符。此外，我们使用前馈网络（FFN）将用户 $u$ 的表示向量从 $\mathbf{e}_u \in \mathbb{R}^{d_u}$ 转换为 $\mathbf{e}_u' \in \mathbb{R}^{d_s}$，使其与 $\mathbf{e}_{a_t}'$ 处于同一空间。

特征向量、Transformer 和门中的参数被随机初始化，并使用从 RL 模型传递的梯度以端到端方式进行训练。

值得注意的是，Transformer 模型和门机制可以被其他架构替代，因为有其他方法可以从输入序列中提取和组合信息，例如基于循环神经网络（RNN）的模型。此外，Transformer 和门机制中的参数在 RL 策略更新其参数期间是固定的。之后，它们由另一个 Adam 优化器更新，使用从 RL 策略反向传播的梯度。

#### 4.1.3 基于 RL 的交互式推荐策略

我们将交互式推荐策略 $\pi_\theta$ 实现为 PPO 算法。PPO 是一种强大的基于 actor-critic 框架和信任区域策略优化（TRPO）算法的同策略强化学习算法。它可以在离散和连续状态空间和动作空间中工作。我们的目标是最大化长期交互的累积用户满意度，这可以通过最大化 PPO 的目标函数来实现：

$$
\mathbb{E}_t \left[ \min \left( \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)} \hat{A}_t, \text{clip} \left( \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)}, 1 - \epsilon, 1 + \epsilon \right) \hat{A}_t \right) \right]
$$

其中 $\epsilon$ 是控制一次可以更新的最大百分比变化的超参数。函数 $\text{clip}(x, a, b)$ 将变量 $x$ 裁剪在 $[a, b]$ 范围内。$\theta_{\text{old}}$ 是更新前的策略，即交互数据在策略 $\theta_{\text{old}}$ 下生成。优势函数 $\hat{A}_t$ 实现为广义优势估计器（GAE）：

$$
\hat{A}_t := \hat{A}_t^{\text{GAE}(\gamma, \lambda)} := \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}^V
$$

其中 $\lambda \in [0, 1]$ 是在偏差和方差之间进行折衷的超参数。$\delta_t^V$ 定义为 $\delta_t^V = r_t + \gamma V(s_{t+1}) - V(s_t)$，即折扣为 $\gamma$ 的近似值函数 $V$ 的 TD 残差。值函数 $V$ 定义为：

$$
V(s_t) := V^{\pi_\theta, \gamma}(s_t) := \mathbb{E}_{s_{t+1:\infty}, a_{t:\infty}} \left[ \sum_{l=0}^{\infty} \gamma^l r_{t+l} \right]
$$

应该注意的是，$\hat{A}_t$ 中的奖励项 $r_t$ 是由因果用户模型 $\phi_M$ 给出的反事实满意度，而不是即时用户反馈。

最后，为了学习一个好的策略，我们需要 $\phi_M$ 给出的反事实满意度尽可能正确和有建设性。接下来，我们将介绍如何构建因果用户模型 $\phi_M$ 以捕捉过度曝光效应从而避免过滤气泡。

### 4.2 基于因果推理的用户满意度分解

我们在图4中展示了传统推荐系统和我们的 CIRS 模型的因果图。

- 节点 $U$ 代表某个用户 $u$，例如可以代表用户的 ID 或个人资料特征。
- 节点 $I$ 代表推荐给用户 $u$ 的物品 $i$。
- 节点 $R$ 代表用户 $u$ 对推荐物品 $i$ 的实时满意度。它是反馈，如点击或视频观看比率。
- 节点 $Y$ 代表用户的内在兴趣，无论物品过度曝光如何都是静态的。
- 节点 $E_t$ 和 $e_t^*$ 代表物品 $i$ 对用户 $u$ 的过度曝光效应。$E_t$ 是随机变量，$e_t^*$ 是在推理阶段（即 RL 规划阶段）计算的 $E_t$ 的值。

传统推荐系统将仅基于用户 $U$ 和物品 $I$ 的特征信息拟合用户满意度 $R$（图4(a)）。这假设用户满意度等于内在兴趣，如前所述这是不恰当的。

我们提出的 CIRS 模型创新性地考虑了过度曝光效应 $E_t$，并分解了对用户满意度 $R$ 的因果效应。具体来说，$R$ 由两条因果路径生成：

(1) $(U, I) \rightarrow Y \rightarrow R$：这条路径将用户 $u$ 和物品 $i$ 投影到它们对应的内在兴趣 $y_{ui}$。然后用户满意度 $r_{ui}$ 与 $y_{ui}$ 成正比。

(2) $I \rightarrow E_t \rightarrow R$：这条路径捕捉物品 $i$ 对用户 $u$ 满意度 $r$ 的实时过度曝光效应 $e_t(u, i)$。这种效应对用户满意度有负面影响。

**内在兴趣估计**：我们将用户的内在兴趣 $y_{ui}$ 估计为 $\hat{y}_{ui} = f_\theta(u, i)$。估计模型 $f_\theta(u, i)$ 可以通过几乎任何已建立的推荐模型实现，例如本工作中使用的 DeepFM。我们在图3的兴趣估计模块中展示了这部分。

**过度曝光效应定义**：考虑到过度曝光效应对用户满意度有负面影响，我们将时间 $t$ 向用户 $u$ 推荐物品 $i$ 的过度曝光效应 $e_t$ 定义为：

$$
e_t := e_t(u, i) := \alpha_u \beta_i \sum_{(u, i_l, t_l) \in S_u, t_l < t} \exp \left( -\frac{t - t_l}{\tau} \right) \times \text{dist}(i, i_l)
$$

其中 $\text{dist}(i, i_l)$ 是两个物品 $i$ 和 $i_l$ 之间的距离。$\alpha_u$ 代表用户 $u$ 对过度曝光效应的敏感度，例如具有较大 $\alpha_u$ 的用户在过度曝光于类似内容时更容易感到厌倦。类似地，$\beta_i$ 代表物品 $i$ 的不可忍耐度。例如，古典音乐可能比流行歌曲更耐听，因此古典音乐的 $\beta_i$ 较小。$\tau$ 是温度超参数。我们将在第5.5节展示学习到的 $\alpha_u$（$\beta$）与用户活跃度（物品流行度）之间的关系。直觉上，当推荐物品 $i$ 接近该用户之前消费的物品（例如 $i_l$）时，即 $\text{dist}(i, i_l)$ 较小，且其推荐时间 $t$ 接近之前物品的推荐时间（例如物品 $i_l$ 的 $t_l$），即项 $(t_l - t)$ 较小，那么过度曝光效应 $e_t(u, i)$ 将较大。这意味着推荐系统正在向用户引入过滤气泡。

**用户满意度估计**：通常，最近推荐的类似物品 $i_l$（即 $\text{dist}(i, i_l)$ 较小）对物品 $i$ 贡献更大的过度曝光效应。$e_t$ 是在时间 $t$ 之前推荐给用户 $u$ 的所有物品效应的总和。通过等式(8)获得过度曝光效应 $e_t$ 和通过 DeepFM 获得内在兴趣 $\hat{y}_{ui}$ 后，我们将用户 $u$ 对物品 $i$ 的满意度估计为：

$$
\hat{r}_t := \hat{r}_{ui}^t = \frac{\hat{y}_{ui}}{1 + e_t(u, i)}
$$

因此，即使内在兴趣 $y_{ui}$ 不变，较大的过度曝光效应 $e_t(u, i)$ 也会降低用户满意度 $\hat{r}_{ui}^t$。

在因果用户模型 $\phi_M$ 的训练阶段，我们最小化推荐模型中的目标函数。在实验中，我们对 VirtualTaobao 使用 MSE 损失，对 KuaiEnv 使用 BPR 损失：

$$
L_{\text{MSE}} = \sum_{(u,i,t) \in \mathcal{D}} (\hat{r}_{ui}^t - r_{ui}^t)^2, \quad L_{\text{BPR}} = -\sum_{(u,i,t) \in \mathcal{D}, j \sim p_n} \log \sigma(\hat{r}_{ui}^t - \hat{r}_{uj}^t)
$$

其中 $\sigma(x) = \frac{1}{1 + e^{-x}}$ 是 Sigmoid 函数。物品 $j$ 是从分布 $p_n$ 采样的负实例。

**反事实满意度估计**：在 RL 规划阶段，当学习到的因果用户模型 $\phi_M$ 与策略 $\pi_\theta$ 交互时，过度曝光效应 $e_t^*$ 现在与预学习阶段的 $e_t$ 不同。因此，我们通过切断路径 $I \rightarrow E_t \rightarrow R$ 来执行因果干预 $do(E_t = e_t^*)$，如图4(c)所示。与旨在移除混杂因子效应的传统因果方法不同，我们仍然需要在此阶段建模正确的过度曝光效应 $e_t^*$。注意，我们使用星号标记此干预阶段中的所有值。我们将 $e_t^*$ 计算为：

$$
e_t^*(u, i) = \gamma^* \cdot \alpha_u \beta_i \sum_{(u, i_l^*, t_l^*) \in S_u^*, t_l^* < t} \exp \left( -\frac{t - t_l^*}{\tau^*} \right) \times \text{dist}(i, i_l^*)
$$

其中 $S_u^*$ 是在 RL 规划阶段产生的新交互轨迹。$\gamma^*$ 是引入的超参数，用于调整过度曝光效应的规模。我们在整个实验中固定 $\gamma$ 为10。$\tau^*$ 是干预阶段的温度超参数，可以与等式(8)中的 $\tau$ 具有不同的值。我们将反事实满意度估计为：

$$
\hat{r}_{ui}^{t*} = \frac{\hat{y}_{ui}}{1 + e_t^*(u, i)}
$$

到现在，我们可以使用估计的反事实满意度作为奖励信号，通过优化等式(5)来更新 RL 策略。整个过程如图3中的预学习和 RL 规划阶段所示。我们使用 Adam 优化器学习因果用户模型 $\phi_M$、策略 $\pi_\theta$ 和状态跟踪器。

最后，通过创新性地用因果推理增强离线 RL 框架，我们获得了一个可以通过防止过滤气泡（即过度曝光物品）来保证大用户满意度的策略。

## 5. 实验

在本节中，我们进行实验来评估 IRS。我们旨在研究以下研究问题：
- (RQ1) 与最先进的静态推荐方法和基于 RL 的交互式推荐策略相比，CIRS 表现如何？
- (RQ2) CIRS 在有限交互轮次中表现如何？
- (RQ3) CIRS 在具有不同用户对过滤气泡容忍度的不同环境中表现如何？
- (RQ4) CIRS 中关键参数的效果是什么？

### 5.1 实验设置

我们介绍关于设置、环境、评估指标和最先进推荐方法的实验设置。

#### 5.1.1 交互式推荐设置中的评估

我们强调我们在交互式设置中评估所有方法，而不是传统的静态或顺序设置。图5说明了静态推荐、传统顺序推荐和交互式推荐如何评估模型。静态和顺序推荐都使用监督学习的理念，即通过将 top-$k$ 结果与测试集中的一组"正确"答案进行比较并计算 Precision、Recall、NDCG 和 Hit Rate 等指标来评估。相比之下，交互式推荐通过沿交互轨迹累积奖励来评估结果。交互式推荐中没有标准答案，这既有趣又具有挑战性。这种设置需要高质量的离线数据，这阻碍了相关研究。

在交互式推荐设置中研究过滤气泡是必要的。过滤气泡是在真实世界推荐场景中当推荐器向用户过度曝光类似内容时发生的现象。这意味着用户满意度可以动态变化，因此在传统的静态或顺序设置中研究过滤气泡是不恰当的，其中测试集中的用户偏好是固定的。

目前，交互式推荐设置尚未被广泛研究，因为很难在离线数据上评估模型。我们通过在 VirtualTaobao 和 KuaiEnv 环境中评估来克服这个问题。

#### 5.1.2 推荐环境

传统推荐数据集太稀疏，无法评估交互式推荐系统。我们使用两个推荐环境：VirtualTaobao 和 KuaiEnv。这两个环境可以扮演与在线真实用户相同的角色。对于推荐器，环境就像图3右上角所示的黑盒。

VirtualTaobao 是一个用于推荐的基准 RL 环境。它通过多代理对抗模仿学习（MAIL）方法模拟淘宝（最大的在线零售平台之一）上真实用户的行为创建。模拟用户可以模仿真实用户的行为并生成与淘宝平台上记录的相同统计数据。在 VirtualTaobao 环境中，用户表示为88维向量 $\mathbf{e}_u \in \{0, 1\}^{88}$，推荐表示为27维向量 $\mathbf{e}_i \in \mathbb{R}^{27}, 0 \leq \mathbf{e}_i \leq 1$。当模型推荐 $\mathbf{e}_i$ 时，环境将立即返回表示用户兴趣的奖励信号，即标量 $r \in \{0, 1, \cdots, 10\}$。

它提供100,000条日志交互用于训练离线 RL 策略。由于物品在 VirtualTaobao 中表示为连续向量，我们使用欧几里得距离计算两个物品之间的距离，即等式(8)和等式(11)中的 $\text{dist}(i, i_l)$ 项。

KuaiEnv 由我们在 KuaiRec 数据集上创建。KuaiRec 是一个包含完全观察到的用户-物品交互矩阵的真实世界数据集。"完全观察"意味着每个用户已经观看了整个集合中的每个视频，然后留下了反馈。因此，与通过在淘宝数据上训练模型模拟真实用户的 VirtualTaobao 不同（即代表用户偏好的奖励来自生成模型），KuaiEnv 使用每个用户-物品对的真实用户历史反馈，这更有说服力。我们将奖励信号定义为视频观看比率，即观看时间与视频总长度的比率。不失一般性，我们使用这个浮点数来表示用户的内在兴趣，即我们假设用户的偏好保持静态并等于记录的评分。我们使用完全观察矩阵（即小矩阵）来评估策略 $\pi_\theta$。对于预学习用户模型 $\phi_M$，我们使用大矩阵中额外的稀疏用户-视频交互。在 RL 规划阶段，我们通过使用用户模型 $\phi_M$ 提供的奖励来学习策略模型 $\pi_\theta$，而不利用离线数据。

KuaiRec 中的每个视频至少有一个不超过四个分类属性，例如 Food 或 Sports。因此我们使用汉明距离计算等式(8)和等式(11)中的 $\text{dist}(i, i_l)$ 项。

**退出机制**：到目前为止，VirtualTaobao 和 KuaiEnv 可以提供用户内在兴趣作为奖励信号。然而，它们无法反映用户对过度曝光效应的响应。为此，我们在两个环境中引入"感到厌倦然后退出"机制来惩罚评估中的过滤气泡。通常，环境将与推荐器重复交互。VirtualTaobao 只有一个朴素的机制来通过预先预测交互轨迹的长度来结束交互。它不受控制，我们通过考虑第3.2节中的观察来改变它。退出机制如图6所示。具体来说，我们计算推荐目标与最近 $N$ 个推荐物品之间的欧几里得距离。如果其中任何一个低于阈值 $d_Q$，环境将退出交互过程，因为真实用户在这种单调推荐下会感到厌倦并退出。在 KuaiEnv 中，类似地，对于最近 $N$ 个推荐物品，如果 $N$ 个物品中有超过 $n_Q$ 个物品至少具有当前推荐目标的一个属性，则此环境中的用户结束交互过程。直觉上，好的推荐器应该避免重复高度相似的物品以防止用户提前退出。

#### 5.1.3 评估指标

我们旨在评估模型在整个交互轨迹 $S$ 上的累积满意度性能，即 $\sum_{l=0}^{|S|} r_{t+l}$，其中 $r_{t+l}$ 是 VirtualTaobao 或 KuaiEnv 返回的奖励信号。注意，在此设置中，用户满意度设置为：

$$
\text{satisfaction} = \begin{cases} \text{interest}, & \text{if no filter bubble ever occurs} \\ 0, & \text{otherwise} \end{cases}
$$

即，如果推荐没有触发退出机制，我们可以累积奖励来表示内在兴趣。但每当过度曝光的物品触发退出机制时，交互被中断，不能再添加奖励。因此，系统不能重复推荐几个置信度最高的高质量物品。直觉上，为了追求长期成功，推荐策略必须在追求更高单轮满意度和维持更长交互序列之间找到折衷。

我们报告100个交互序列上的平均累积满意度。

#### 5.1.4 基线

我们使用常用的静态推荐模型加直接策略作为基线。对于具有丰富物品特征的 KuaiEnv，四个静态推荐基线是：

- **DeepFM**：一种强大的基于分解机的神经网络，包含宽和深部分，从低阶和高阶特征交互中提取知识。它在许多公司中作为推荐框架的强大骨干。
- **IPS**：一种著名的统计技术，通过重新加权收集数据中的每个样本来调整目标分布。在推荐中，它广泛用于建模观察概率以移除收集数据中的曝光偏差或选择偏差。它易于实现且存在高方差问题。
- **PD（流行度去混杂）**：一种基于因果推理的方法，将物品流行度建模为在曝光物品和用户偏好之间引入虚假相关的混杂因子。通过显式建模流行度，PD 可以在最终推荐阶段移除流行度偏差。
- **DICE**：尝试通过在所谓的因果嵌入中分别建模流行度和用户兴趣来解耦它们。因此，物品流行度或其他不需要的因素可以在推荐阶段被移除。

应该注意：(1) IPS、PD 和 DICE 是用于去偏差的技术，我们将它们的骨干网络实现为 DeepFM。(2) 所有这些方法都是确定性/静态模型，研究人员通常将具有最高预测分数的 top-1 或 top-$k$ 物品作为最终推荐。在我们的交互式设置中，这种方式会立即导致过度曝光。因此，我们通过从带有 Softmax 层的最终 logits 中采样来进行推荐。

我们还在 KuaiEnv 中实现了基本策略：
- **Random**：完全推荐随机物品。
- **$\epsilon$-greedy**：以概率 $\epsilon$ 输出随机结果，以概率 $1 - \epsilon$ 使用 DeepFM 模型的结果。
- **UCB** 为每个物品维护一个上置信度边界，并遵循不确定性下的乐观原则。这意味着如果我们对某个动作不确定，我们应该尝试它。UCB 可以在决策过程中平衡探索和利用。

对于 VirtualTaobao，由于用户和物品由特征向量给出，我们只能实现一个在 Softmax 结果上采样的多层感知器（MLP）和 $\epsilon$-greedy 策略。此外，我们在相同的离线 RL 框架上实现了强大的 RL 基线 PPO，即它通过与用户模型交互来学习，但没有因果推理模块。为了比较，我们将此基线表示为 CIRS w/o CI。

注意 PPO 是基于模型的离线 RL 中的一个组件，可以被其他 RL 模型（例如 DQN、Actor-Critic 或 DDPG）替代。CIRS w/o CI 可以被视为总结基于模型的离线 RL 方法的框架。例如，与 Huang 等人的区别是他们使用 MF 作为用户模型而我们使用 DeepFM；与 Huang 等人的区别是他们研究了许多顺序模型作为状态跟踪器而我们使用基于 Transformer 的模型。

### 5.2 整体性能比较

我们在两个环境中评估提出的 CIRS 和基线。我们使用网格搜索为所有方法调整最优参数。例如，在 VirtualTaobao 中，关键参数 $\tau^*$ 在 $\{0.001, 0.005, 0.01, 0.1, 0.5, 1.0, 5.0\}$ 中搜索，$\tau$ 在 $\{0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0\}$ 中搜索。

对于一般比较，我们不限制交互长度并将最大轮次设置得足够大（但可行）。我们将 VirtualTaobao 和 KuaiEnv 的最大轮次分别设置为50和100。对于环境设置中的参数，我们将窗口大小（即最近推荐的数量）设置为 $N = 5$，退出阈值 VirtualTaobao 为 $d_Q = 3.0$，KuaiEnv 为 $N = 1, n_Q = 1$。

结果如图7(A-B)所示。第一行显示累积满意度，这是评估推荐系统的全局指标。第二行和第三行显示用户满意度的细节，即交互轨迹长度和单轮满意度。从(A1)和(B1)中，我们可以看到提出的 CIRS 在 VirtualTaobao 和 KuaiEnv 中经过几个时期后都实现了最大的平均累积满意度。

在 VirtualTaobao 的前几个时期中，CIRS 和 CIRS w/o CI 的性能都在提高，因为 RL 策略逐渐找到正确的用户偏好，因此每轮满意度增加(A3)。有趣的是，单轮满意度的增加在开始时牺牲了轨迹长度(A2)。后来，长度逐渐变得稳定，CIRS 的策略最终在长度和单轮满意度之间找到平衡点，从而实现最大累积满意度。然而，没有因果推理模块（如 CIRS w/o CI 所示），策略变得不稳定，性能随着时期增加而退化。这种现象证明了因果推理在捕捉过度曝光效应从而避免重复推荐物品方面的有效性。

在 KuaiEnv 中，CIRS 在足够的时期后也实现了最大的累积满意度(B1)。与 VirtualTaobao 不同，性能提高主要是因为交互长度的增加。如(B2)所示，CIRS 和 CIRS w/o CI 的交互长度在大约160个时期后增加到最大长度100。此外，CIRS 的累积满意度在(B1)中大约180个时期后进一步增加，这是由于进一步提高单轮性能的可能性(B3)。对于 VirtualTaobao 和 KuaiEnv，CIRS 都击败了对应方法 CIRS w/o CI，这证明了 CIRS 中因果模块的有效性。

对于其他基线，我们可以看到除 Random 外的所有其他方法都可以实现更好的单轮性能(A3和B3)。然而，即使有基本策略（即随机采样、基于 Softmax 的采样和 $\epsilon$-greedy）引入的随机性，它们的推荐结果也太有限和狭窄。注意在 VirtualTaobao 中，即使随机采样也不能带来更长的交互序列，因为维度诅咒：动作空间有88维，因此任何两个随机点的欧几里得距离在统计上变得无法区分。IPS 的结果在单轮性能方面剧烈波动(B3)，这是由于广泛讨论的高方差问题。$\epsilon$-greedy 和 IPS 的交互长度比其他方法（即 DICE、PD 和 DeepFM）长(B2)。这是因为这两种方法在整个交互过程中有能力探索物品空间。与这两种朴素方法相比，UCB 是一种可以自动平衡探索和利用的策略，它在开始时有最佳性能。然而，经过几个时期的探索后，策略增强了对某些物品的信念，从而导致陷入过滤气泡。因此，UCB 最终具有如(B2)所示的最低交互长度，但具有(B3)中的最大单轮满意度。

总之，除了基于深度 RL 策略的方法（即 CIRS 和 CIRS w/o CI），具有启发式策略的静态推荐模型（即基于 Softmax 的采样、$\epsilon$-greedy 和 UCB）无法克服过度曝光效应，从而导致过滤气泡并导致低用户满意度。此外，通过比较 CIRS 和 CIRS w/o CI，我们展示了因果推理在离线 RL 框架中的有效性。

### 5.3 有限交互轮次的结果

在真实世界推荐场景中，用户精力有限，不会花太多轮次与推荐器交互。因此，我们将 VirtualTaobao 和 KuaiEnv 的最大轮次分别限制为10和30。我们旨在研究在这种情况下策略是否可以利用来提高单轮满意度。我们将 VirtualTaobao 中的退出阈值更改为 $d_Q = 1.0$ 以获得更好的展示。

从图7(C-D)的结果中，我们可以看到 CIRS 在 KuaiEnv 中优于 CIRS w/o CI(D1和D3)，并在 VirtualTaobao 中产生类似性能(C1和C3)。在 VirtualTaobao 中，两种策略在大约150个时期后都实现了相同水平的单轮性能（大于4.0）。性能甚至类似于静态方法(C3和A3)。在 KuaiEnv 中，与前一设置(B3)相比，两种策略都实现了更高的单轮性能(D3)。特别是，CIRS 在单轮性能方面有巨大改进(D3和B3)，这意味着它适合具有有限交互轮次的真实世界交互式推荐场景。实际上，在(B1和B3)中，我们还可以看到 CIRS 的性能在 $\text{epoch} = 200$ 时继续增加。这意味着即使在巨大轮次下，只要有足够的训练时间，CIRS 也有潜力。(C3和D3)中的结果表明知识（即正确的用户偏好）可以从因果用户模型蒸馏到 RL 策略。这进一步证明了我们基于模型的 RL 框架的有效性。再次，在此设置中，我们得出结论，通过与因果推理集成，我们让 CIRS 优于其对应物。

### 5.4 不同用户敏感度的结果

为了验证 CIRS 的通用性，我们改变参数：距离阈值 $d_Q$ 和窗口大小 $N$ 以分别说明它们在 VirtualTaobao 和 KuaiEnv 中的影响。较大的 $d_Q$ 或 $N$ 意味着用户对过滤气泡更敏感，更容易退出交互。图8中的结果表明，当用户不太敏感时（即 VirtualTaobao 中的小 $d_Q$ 和 KuaiEnv 中的小 $N$），CIRS 优于所有基线方法。CIRS 获得最佳累积满意度，因为它可以避免重复推荐高度相似的物品，从而可以维持长交互长度。

然而，当用户变得更敏感时，CIRS 的性能不可避免地下降，尽管它仍然可以击败其对应物 CIRS w/o CI。当 $d_Q \geq 4$ 或 $N \geq 4$ 时，CIRS 和 CIRS w/o CI 只能实现与其他基线相同甚至更差的性能。这意味着面对极其挑剔的用户，即使是因果推理增强的模型也无法缓解因过度曝光任何物品而导致的不满。当 $d_Q = 5$ 或 $N = 5$ 时，两种基于 RL 的模型甚至无法击败作为两种基于模型的 RL 方法中教师模型的 DeepFM 或 MLP。这是因为当用户太挑剔时，基于 RL 的方法没有机会执行其探索-利用理念。

同时，其他基线在不同用户敏感度下具有类似性能——推荐让用户感到厌倦并退出，即使用户对过滤气泡更容忍（即对物品过度曝光不太敏感）。这也证明它们不适合解决过滤气泡问题。

### 5.5 关键参数的效果

我们在 KuaiEnv 中研究 CIRS 关键参数的效果。我们将用户模型中学习到的 $\alpha_u$ 和 $\beta_i$ 与两个数据统计（即用户的活跃度和物品的流行度）进行比较，这些统计通过对大矩阵的行和列求和得到。我们在图9中展示结果。结果直观易懂：用户敏感度（即 $\alpha_u$）与用户活跃度成正比，即活跃用户在观看过度曝光视频时更容易感到厌倦，因为他/她之前可能已经看过许多类似内容。类似地，物品不可忍耐度（即 $\beta_i$）与物品流行度成正比，即流行视频在过度曝光时更不可忍耐。这也解释了为什么流行物品很快过时。

此外，我们研究了 $\tau$（等式(8)中）和 $\tau^*$（等式(11)中）的不同组合对累积满意度的影响。$\tau = 0, \tau^* = 0$ 的 CIRS 退化为 CIRS w/o CI，因为 $e_t(u, i)$ 和 $e_t^*(u, i)$ 都变为0，即用户满意度的建模不会考虑过度曝光效应。图10中的结果表明，合适的 $\tau$-$\tau^*$ 对确实比 CIRS w/o CI 提高了性能。注意 $\tau$ 轴和 $\tau^*$ 轴的数量级在 KuaiEnv 中差异很大，因为等式(8)和等式(11)中的时间单位不同。前者使用日志数据中的秒(s)，后者使用 RL 规划和评估阶段中的步(s)。

## 6. 结论与讨论

这项工作在交互式推荐设置中研究过滤气泡。与使用监督学习理念的静态和顺序推荐设置不同，交互式设置通过沿交互轨迹累积奖励来评估基于 RL 的策略。

交互式推荐设置提供了跟踪和估计过滤气泡的实用方法，这是在真实世界推荐场景中当推荐器向用户过度曝光类似内容时发生的现象。我们在音乐和视频推荐数据集上进行了实地研究，表明用户满意度会随着类似内容的增加而下降，这促使我们在推荐系统中移除过滤气泡。

我们提出了一个反事实交互式推荐系统（CIRS），在离线 RL 中利用因果推理来推断用户变化的满意度。CIRS 利用因果用户模型，可以从物品的过度曝光效应中解耦内在用户兴趣。因果用户模型为学习 RL 策略提供无偏反事实奖励。为了进行评估，我们创新性地基于真实世界完全观察到的用户评分数据集创建了一个忠实的 RL 环境 KuaiEnv。大量实验表明，所提出的方法可以打破过滤气泡并增加用户的累积满意度。实验表明，CIRS 可以通过在追求高单轮满意度和维持持久交互之间找到折衷来获得最优累积满意度。

我们的工作有几个值得注意的贡献。最重要的是，我们展示了在交互式推荐设置中评估基于 RL 方法的正确方式，即通过累积奖励评估决策者（即推荐策略）。在现实中，真实用户在使用推荐系统时心中没有任何标准答案；公司关心的是模型是否能让用户长期满意。因此，交互式推荐设置可以很好地描述真实世界推荐场景。然而，许多先前工作仍然通过静态或顺序设置评估基于 RL 的方法，即通过将 top-k 结果与测试集中的一组"正确"答案进行比较并计算 Precision、Recall、NDCG 和 Hit Rate 等指标来评估。我们理解他们为什么选择那种评估方式：在离线数据上评估 RL 是出了名的困难。为了克服这个问题，我们创建了 KuaiEnv 环境，其中每个用户对所有物品的偏好都是已知的。有了这个环境，研究人员可以进行忠实的评估，而不必在模拟用户-物品矩阵中合成用户偏好。

通过在交互式推荐设置中建模和缓解过滤气泡问题，我们展示了推荐社区中潜在的研究方向和可能的解决方案。在未来，我们相信交互式推荐将吸引大量研究关注。探索此设置中的其他类型偏差很有趣。结合因果推理和强化学习是有前途的，因为因果推理可以提供优化模型的显式指南，从而在 RL 中引入可解释性。

## 致谢

这项工作得到国家重点研发计划（2021ZD0111802）、国家自然科学基金（61972372, U19A2079, 62121002）和文旅部 CCCD 重点实验室的支持。

## 参考文献

[1] Guy Aridor, Duarte Goncalves, and Shan Sikdar. 2020. Deconstructing the Filter Bubble: User Decision-Making and Recommender Systems. In RecSys '20. 82–91.

[2] Xueying Bai, Jian Guan, and Hongning Wang. 2019. A Model-Based Reinforcement Learning with Adversarial Training for Online Recommendation. In NeurIPS '19.

[3] Elias Bareinboim. 2020. Causal Reinforcement Learning. In ICML 2020 Tutorial.

[4] Stephen Bonner and Flavian Vasile. 2018. Causal Embeddings for Recommendation. In RecSys '18. 104–112.

[5] Axel Bruns. 2019. Are Filter Bubbles Real? John Wiley & Sons.

[6] Qingpeng Cai, Shuchang Liu, Xueliang Wang, Tianyou Zuo, Wentao Xie, Bin Yang, Dong Zheng, Peng Jiang, and Kun Gai. 2023. Reinforcing User Retention in a Billion Scale Short Video Recommender System. arXiv preprint arXiv:2302.01724 (2023).

[7] Qingpeng Cai, Zhenghai Xue, Chi Zhang, Wanqi Xue, Shuchang Liu, Ruohan Zhan, Xueliang Wang, Tianyou Zuo, Wentao Xie, Dong Zheng, et al. 2023. Two-Stage Constrained Actor-Critic for Short Video Recommendation. arXiv preprint arXiv:2302.01680 (2023).

[8] Allison J. B. Chaney, Brandon M. Stewart, and Barbara E. Engelhardt. 2018. How Algorithmic Confounding in Recommendation Systems Increases Homogeneity and Decreases Utility. In RecSys '18. 224–232.

[9] Haokun Chen, Xinyi Dai, Han Cai, Weinan Zhang, Xuejian Wang, Ruiming Tang, Yuzhou Zhang, and Yong Yu. 2019. Large-scale Interactive Recommendation with Tree-structured Policy Gradient. In AAAI. 3312–3320.

[10] Jiawei Chen, Hande Dong, Xiang Wang, Fuli Feng, Meng Wang, and Xiangnan He. 2022. Bias and Debias in Recommender System: A Survey and Future Directions. ACM Transactions on Information Systems (2022).

[11] Minmin Chen, Alex Beutel, Paul Covington, Sagar Jain, Francois Belletti, and Ed H. Chi. 2019. Top-K Off-Policy Correction for a REINFORCE Recommender System. In WSDM '19. 456–464.

[12] Minmin Chen, Bo Chang, Can Xu, and Ed H. Chi. 2021. User Response Models to Improve a REINFORCE Recommender System. In WSDM '21. 121–129.

[13] Xinshi Chen, Shuang Li, Hui Li, Shaohua Jiang, Yuan Qi, and Le Song. 2019. Generative adversarial user model for reinforcement learning based recommendation system. In ICML '19. 1052–1061.

[14] Marc Peter Deisenroth, Dieter Fox, and Carl Edward Rasmussen. 2013. Gaussian Processes for Data-efficient Learning in Robotics and Control. IEEE Transactions on Pattern Analysis and Machine Intelligence 37, 2 (2013), 408–423.

[15] Tim Donkers and Jürgen Ziegler. 2021. The Dual Echo Chamber: Modeling Social Media Polarization for Interventional Recommending. In RecSys '21. 12–22.

[16] Amir Feder, Katherine A Keith, Emaad Manzoor, Reid Pryzant, Dhanya Sridhar, Zach Wood-Doughty, Jacob Eisenstein, Justin Grimmer, Roi Reichart, Margaret E Roberts, et al. 2021. Causal Inference in Natural Language Processing: Estimation, Prediction, Interpretation and Beyond. arXiv preprint arXiv:2109.00725 (2021).

[17] Seth Flaxman, Sharad Goel, and Justin M Rao. 2016. Filter Bubbles, Echo Chambers, and Online News Consumption. Public Opinion Quarterly 80, S1 (2016), 298–320.

[18] Chongming Gao, Kexin Huang, Jiawei Chen, Yuan Zhang, Biao Li, Peng Jiang, Shiqi Wang, Zhong Zhang, and Xiangnan He. 2023. Alleviating Matthew Effect of Offline Reinforcement Learning in Interactive Recommendation. In SIGIR '23.

[19] Chongming Gao, Wenqiang Lei, Xiangnan He, Maarten de Rijke, and Tat-Seng Chua. 2021. Advances and Challenges in Conversational Recommender Systems: A Survey. AI Open 2 (2021), 100–126.

[20] Chongming Gao, Shijun Li, Wenqiang Lei, Jiawei Chen, Biao Li, Peng Jiang, Xiangnan He, Jiaxin Mao, and Tat-Seng Chua. 2022. KuaiRec: A Fully-Observed Dataset and Insights for Evaluating Recommender Systems. In CIKM '22. 540–550.

[21] Chongming Gao, Shuai Yuan, Zhong Zhang, Hongzhi Yin, and Junming Shao. 2019. BLOMA: Explain Collaborative Filtering via Boosted Local Rank-One Matrix Approximation. In DASFAA '19. 487–490.

[22] Mingkun Gao, Hyo Jin Do, and Wai-Tat Fu. 2018. Burst Your Bubble! An Intelligent System for Improving Awareness of Diverse Social Opinions. In IUI '18. 371–383.

[23] Yingqiang Ge, Shuchang Liu, Ruoyuan Gao, Yikun Xian, Yunqi Li, Xiangyu Zhao, Changhua Pei, Fei Sun, Junfeng Ge, Wenwu Ou, and Yongfeng Zhang. 2021. Towards Long-Term Fairness in Recommendation. In WSDM '21. 445–453.

[24] Alexandre Gilotte, Clément Calauzènes, Thomas Nedelec, Alexandre Abraham, and Simon Dollé. 2018. Offline A/B Testing for Recommender Systems. In WSDM '18. 198–206.

[25] Huifeng Guo, Ruiming Tang, Yunming Ye, Zhenguo Li, and Xiuqiang He. 2017. DeepFM: A Factorization-Machine Based Neural Network for CTR Prediction. In IJCAI'17. 1725–1731.

[26] Xiangnan He, Kuan Deng, Xiang Wang, Yan Li, YongDong Zhang, and Meng Wang. 2020. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation. In SIGIR '20. 639–648.

[27] Jonathan L Herlocker, Joseph A Konstan, Loren G Terveen, and John T Riedl. 2004. Evaluating Collaborative Filtering Recommender Systems. ACM Transactions on Information Systems (TOIS) 22, 1 (2004), 5–53.

[28] MA Hernán and JM Robins. 2020. Causal Inference: What If. Boca Raton: Chapman & Hall/CRC.

[29] Jin Huang, Harrie Oosterhuis, Bunyamin Cetinkaya, Thijs Rood, and Maarten de Rijke. 2022. State Encoders in Reinforcement Learning for Recommendation: A Reproducibility Study. In SIGIR '22. 2738–2748.

[30] Jin Huang, Harrie Oosterhuis, Maarten de Rijke, and Herke van Hoof. 2020. Keeping Dataset Biases out of the Simulation: A Debiased Simulator for Reinforcement Learning Based Recommender Systems. In RecSys '20. 190–199.

[31] Eslam Hussein, Prerna Juneja, and Tanushree Mitra. 2020. Measuring Misinformation in Video Search Platforms: An Audit Study on YouTube. Proc. ACM Hum.-Comput. Interact. 4, CSCW1, Article 048 (May 2020).

[32] Eugene Ie, Vihan Jain, Jing Wang, Sanmit Narvekar, Ritesh Agarwal, Rui Wu, Heng-Tze Cheng, Morgane Lustman, Vince Gatto, Paul Covington, et al. 2019. Reinforcement learning for slate-based recommender systems: A tractable decomposition and practical methodology. arXiv preprint arXiv:1905.12767 (2019).

[33] Rolf Jagerman, Ilya Markov, and Maarten de Rijke. 2019. When People Change Their Mind: Off-Policy Evaluation in Non-Stationary Recommendation Environments. In WSDM '19. 447–455.

[34] Olivier Jeunen and Bart Goethals. 2021. Pessimistic Reward Models for Off-Policy Learning in Recommendation. In RecSys '21. 63–74.

[35] Diederik P Kingma and Jimmy Ba. 2014. Adam: A method for Stochastic Optimization. arXiv preprint arXiv:1412.6980 (2014).

[36] Haruka Kiyohara, Kosuke Kawakami, and Yuta Saito. 2021. Accelerating Offline Reinforcement Learning Application in Real-Time Bidding and Recommendation: Potential Use of Simulation. arXiv preprint arXiv:2109.08331 (2021).

[37] Vijay Konda and John Tsitsiklis. 1999. Actor-critic algorithms. Advances in neural information processing systems 12 (1999).

[38] Yehuda Koren. 2008. Factorization Meets the Neighborhood: A Multifaceted Collaborative Filtering Model. In KDD '08. 426–434.

[39] Wenqiang Lei, Chongming Gao, and Maarten de Rijke. 2021. RecSys 2021 Tutorial on Conversational Recommendation: Formulation, Methods, and Evaluation. In RecSys '21. 842–844.

[40] Sergey Levine, Aviral Kumar, George Tucker, and Justin Fu. 2020. Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems. arXiv preprint arXiv:2005.01643 (2020).

[41] Qian Li, Xiangmeng Wang, and Guandong Xu. 2021. Be Causal: De-biasing Social Network Confounding in Recommendation. arXiv preprint arXiv:2105.07775 (2021).

[42] Dawen Liang, Laurent Charlin, James McInerney, and David M. Blei. 2016. Modeling User Exposure in Recommendation. In WWW '16. 951–961.

[43] Timothy P Lillicrap, Jonathan J Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, and Daan Wierstra. 2015. Continuous control with deep reinforcement learning. arXiv preprint arXiv:1509.02971 (2015).

[44] Dugang Liu, Pengxiang Cheng, Zhenhua Dong, Xiuqiang He, Weike Pan, and Zhong Ming. 2020. A General Knowledge Distillation Framework for Counterfactual Recommendation via Uniform Data. In SIGIR '20. 831–840.

[45] Feng Liu, Huifeng Guo, Xutao Li, Ruiming Tang, Yunming Ye, and Xiuqiang He. 2020. End-to-End Deep Reinforcement Learning Based Recommendation with Supervised Embedding. In WSDM '20. 384–392.

[46] Feng Liu, Ruiming Tang, Xutao Li, Weinan Zhang, Yunming Ye, Haokun Chen, Huifeng Guo, and Yuzhou Zhang. 2018. Deep Reinforcement Learning based Recommendation with Explicit User-item Interactions Modeling. arXiv preprint arXiv:1810.12027 (2018).

[47] Ping Liu, Karthik Shivaram, Aron Culotta, Matthew A. Shapiro, and Mustafa Bilgic. 2021. The Interaction between Political Typology and Filter Bubbles in News Recommendation Algorithms. In WWW '21. 3791–3801.

[48] Shuchang Liu, Qingpeng Cai, Bowen Sun, Yuhao Wang, Ji Jiang, Dong Zheng, Kun Gai, Peng Jiang, Xiangyu Zhao, and Yongfeng Zhang. 2023. Exploration and Regularization of the Latent Action Space in Recommendation. arXiv preprint arXiv:2302.03431 (2023).

[49] Yong Liu, Yingtai Xiao, Qiong Wu, Chunyan Miao, Juyong Zhang, Binqiang Zhao, and Haihong Tang. 2020. Diversified Interactive Recommendation with Implicit Feedback. In AAAI '20. 4932–4939.

[50] David Lopez-Paz, Robert Nishihara, Soumith Chintala, Bernhard Scholkopf, and Léon Bottou. 2017. Discovering causal signals in images. In CVPR '17. 6979–6987.

[51] Jiaqi Ma, Zhe Zhao, Xinyang Yi, Ji Yang, Minmin Chen, Jiaxi Tang, Lichan Hong, and Ed H. Chi. 2020. Off-Policy Learning in Two-Stage Recommender Systems. In WWW '20. 463–473.

[52] Prashan Madumal, Tim Miller, Liz Sonenberg, and Frank Vetere. 2020. Explainable reinforcement learning through a causal lens. In AAAI. 2493–2500.

[53] Farzan Masrour, Tyler Wilson, Heng Yan, Pang-Ning Tan, and Abdol Esfahanian. 2020. Bursting the Filter Bubble: Fairness-aware Network Link Prediction. In AAAI '20. 841–848.

[54] James McInerney, Brian Brost, Praveen Chandar, Rishabh Mehrotra, and Benjamin Carterette. 2020. Counterfactual Evaluation of Slate Recommendations with Sequential Reward Interactions. In KDD '20. 1779–1788.

[55] Dana McKay, Kaipin Owyong, Stephann Makri, and Marisela Gutierrez Lopez. 2022. Turn and Face the Strange: Investigating Filter Bubble Bursting. In CHIIR '22. 233–242.

[56] Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, and Martin Riedmiller. 2013. Playing atari with deep reinforcement learning. arXiv preprint arXiv:1312.5602 (2013).

[57] Anusha Nagabandi, Gregory Kahn, Ronald S Fearing, and Sergey Levine. 2018. Neural Network Dynamics for Model-based Deep Reinforcement Learning with Model-free Fine-tuning. In ICRA. IEEE, 7559–7566.

[58] Tien T. Nguyen, Pik-Mai Hui, F. Maxwell Harper, Loren Terveen, and Joseph A. Konstan. 2014. Exploring the Filter Bubble: The Effect of Using Recommender Systems on Content Diversity. In WWW '14. 677–686.

[59] Zachary A. Pardos and Weijie Jiang. 2020. Designing for Serendipity in a University Course Recommendation System. In LAK '20. 350–359.

[60] Eli Pariser. 2011. The filter bubble: How the new personalized web is changing what we read and how we think. Penguin.

[61] Judea Pearl. 2009. Causality. Cambridge University Press.

[62] Doina Precup. 2000. Eligibility Traces for Off-policy Policy Evaluation. Computer Science Department Faculty Publication Series (2000), 80.

[63] Manoel Horta Ribeiro, Raphael Ottoni, Robert West, Virgílio A. F. Almeida, and Wagner Meira. 2020. Auditing Radicalization Pathways on YouTube. In FAT* '20. 131–141.

[64] Yuta Saito, Suguru Yaginuma, Yuta Nishino, Hayato Sakata, and Kazuhide Nakata. 2020. Unbiased Recommender Learning from Missing-Not-At-Random Implicit Feedback. In WSDM '20. 501–509.

[65] Masahiro Sato, Sho Takemori, Janmajay Singh, and Tomoko Ohkuma. 2020. Unbiased Learning for the Causal Effect of Recommendation. In RecSys. 378–387.

[66] Tobias Schnabel, Paul N. Bennett, Susan T. Dumais, and Thorsten Joachims. 2018. Short-Term Satisfaction and Long-Term Coverage: Understanding How Users Tolerate Algorithmic Exploration. In WSDM '18. 513–521.

[67] Tobias Schnabel, Adith Swaminathan, Ashudeep Singh, Navin Chandak, and Thorsten Joachims. 2016. Recommendations as Treatments: Debiasing Learning and Evaluation. In ICML '16. 1670–1679.

[68] John Schulman, Sergey Levine, Pieter Abbeel, Michael Jordan, and Philipp Moritz. 2015. Trust region policy optimization. In ICML '15. 1889–1897.

[69] John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, and Pieter Abbeel. 2015. High-dimensional Continuous Control Using Generalized Advantage Estimation. arXiv preprint arXiv:1506.02438 (2015).

[70] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. 2017. Proximal Policy Optimization Algorithms. arXiv preprint arXiv:1707.06347 (2017).

[71] Jing-Cheng Shi, Yang Yu, Qing Da, Shi-Yong Chen, and An-Xiang Zeng. 2019. Virtual-Taobao: Virtualizing Real-world Online Retail Environment for Reinforcement Learning. In AAAI '19. 4902–4909.

[72] Larissa Spinelli and Mark Crovella. 2020. How YouTube Leads Privacy-Seeking Users Away from Reliable Information. In UMAP '20 Adjunct. 244–251.

[73] Dusan Stamenkovic, Alexandros Karatzoglou, Ioannis Arapakis, Xin Xin, and Kleomenis Katevas. 2022. Choosing the Best of Both Worlds: Diverse and Novel Recommendations through Multi-Objective Reinforcement Learning. In WSDM '22. 957–965.

[74] Richard S Sutton and Andrew G Barto. 2018. Reinforcement Learning: An Introduction. MIT press.

[75] Adith Swaminathan and Thorsten Joachims. 2015. Counterfactual Risk Minimization: Learning from Logged Bandit Feedback. In ICML '15. 814–823.

[76] Erich Christian Teppan and Alexander Felfernig. 2012. Minimization of Decoy Effects in Recommender Result Sets. Web Intelligence and Agent Systems 10, 4 (2012), 385–395.

[77] Philip S. Thomas and Emma Brunskill. 2016. Data-Efficient off-Policy Policy Evaluation for Reinforcement Learning. In ICML '16. 2139–2148.

[78] Matus Tomlein, Branislav Pecher, Jakub Simko, Ivan Srba, Robert Moro, Elena Stefancova, Michal Kompan, Andrea Hrckova, Juraj Podrouzek, and Maria Bielikova. 2021. An Audit of Misinformation Filter Bubbles on YouTube: Bubble Bursting and Recent Behavior Changes. In RecSys '21. 1–11.

[79] Antonela Tommasel, Juan Manuel Rodriguez, and Daniela Godoy. 2021. I Want to Break Free! Recommending Friends from Outside the Echo Chamber. In RecSys '21. 23–33.

[80] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is All you Need. In NeurIPS '17.

[81] Shiqi Wang, Chongming Gao, Min Gao, Junliang Yu, Zongwei Wang, and Hongzhi Yin. 2022. Who Are the Best Adopters? User Selection Model for Free Trial Item Promotion. IEEE Transactions on Big Data (2022).

[82] Tan Wang, Jianqiang Huang, Hanwang Zhang, and Qianru Sun. 2020. Visual commonsense r-cnn. In CVPR '20. 10760–10770.

[83] Wenjie Wang, Fuli Feng, Xiangnan He, Xiang Wang, and Tat-Seng Chua. 2021. Deconfounded Recommendation for Alleviating Bias Amplification. In KDD '21. 1717–1725.

[84] Wenjie Wang, Fuli Feng, Xiangnan He, Hanwang Zhang, and Tat-Seng Chua. 2021. Clicks Can Be Cheating: Counterfactual Recommendation for Mitigating Clickbait Issue. In SIGIR '21. 1288–1297.

[85] Wenlin Wang, Hongteng Xu, Ruiyi Zhang, Wenqi Wang, Piyush Rai, and Lawrence Carin. 2021. Learning to recommend from sparse data via generative user feedback. In AAAI '21.

[86] Xiaojie Wang, Rui Zhang, Yu Sun, and Jianzhong Qi. 2019. Doubly Robust Joint Learning for Recommendation on Data Missing Not at Random. In ICML. PMLR, 6638–6647.

[87] Zifeng Wang, Xi Chen, Rui Wen, Shao-Lun Huang, Ercan Kuruoglu, and Yefeng Zheng. 2020. Information Theoretic Counterfactual Learning from Missing-not-at-random Feedback. Advances in Neural Information Processing Systems 33 (2020), 1854–1864.

[88] Zhenlei Wang, Jingsen Zhang, Hongteng Xu, Xu Chen, Yongfeng Zhang, Wayne Xin Zhao, and Ji-Rong Wen. 2021. Counterfactual Data-Augmented Sequential Recommendation. In SIGIR '21. 347–356.

[89] Teng Xiao and Donglin Wang. 2021. A General Offline Reinforcement Learning Framework for Interactive Recommendation. In AAAI '21.

[90] Xin Xin, Alexandros Karatzoglou, Ioannis Arapakis, and Joemon M. Jose. 2020. Self-Supervised Reinforcement Learning for Recommender Systems. In SIGIR '20. 931–940.

[91] Shuyuan Xu, Yingqiang Ge, Yunqi Li, Zuohui Fu, Xu Chen, and Yongfeng Zhang. 2021. Causal Collaborative Filtering. arXiv preprint arXiv:2102.01868 (2021).

[92] Ya Xu, Nanyu Chen, Addrian Fernandez, Omar Sinno, and Anmol Bhasin. 2015. From Infrastructure to Culture: A/B Testing Challenges in Large Scale Social Networks. In KDD '15. 2227–2236.

[93] Yuanbo Xu, Yongjian Yang, En Wang, Jiayu Han, Fuzhen Zhuang, Zhiwen Yu, and Hui Xiong. 2020. Neural Serendipity Recommendation: Exploring the Balance between Accuracy and Novelty with Sparse Explicit Feedback. ACM Trans. Knowl. Discov. Data 14, 4, Article 50 (June 2020).

[94] Wanqi Xue, Qingpeng Cai, Ruohan Zhan, Dong Zheng, Peng Jiang, and Bo An. 2023. ResAct: Reinforcing Long-term Engagement in Sequential Recommendation with Residual Actor. In ICLR '23.

[95] Mengyue Yang, Quanyu Dai, Zhenhua Dong, Xu Chen, Xiuqiang He, and Jun Wang. 2021. Top-N Recommendation with Counterfactual User Preference Simulation. In CIKM '21.

[96] Junliang Yu, Min Gao, Hongzhi Yin, Jundong Li, Chongming Gao, and Qinyong Wang. 2019. Generating Reliable Friends via Adversarial Training to Improve Social Recommendation. In ICDM '19. IEEE, 768–777.

[97] Ruiyi Zhang, Tong Yu, Yilin Shen, Hongxia Jin, Changyou Chen, and Lawrence Carin. 2019. Reward Constrained Interactive Recommendation with Natural Language Feedback. In NeurIPS '19.

[98] Yang Zhang, Fuli Feng, Xiangnan He, Tianxin Wei, Chonggang Song, Guohui Ling, and Yongdong Zhang. 2021. Causal Intervention for Leveraging Popularity Bias in Recommendation. In SIGIR '21. 11–20.

[99] Zhong Zhang, Nian Shao, Chongming Gao, Rui Miao, Qinli Yang, and Junming Shao. 2022. Mixhead: Breaking the Low-rank Bottleneck in Multi-head Attention Language Models. Knowledge-Based Systems (2022), 108075.

[100] Xiangyu Zhao, Long Xia, Lixin Zou, Hui Liu, Dawei Yin, and Jiliang Tang. 2020. Whole-Chain Recommendations. In CIKM '20. 1883–1891.

[101] Xiangyu Zhao, Long Xia, Lixin Zou, Hui Liu, Dawei Yin, and Jiliang Tang. 2021. UserSim: User Simulation via Supervised Generative Adversarial Network. In WWW '21. 3582–3589.

[102] Yu Zheng, Chen Gao, Xiang Li, Xiangnan He, Yong Li, and Depeng Jin. 2021. Disentangling User Interest and Conformity for Recommendation with Causal Embedding. In WWW '21. 2980–2991.

[103] Sijin Zhou, Xinyi Dai, Haokun Chen, Weinan Zhang, Kan Ren, Ruiming Tang, Xiuqiang He, and Yong Yu. 2020. Interactive Recommender System via Knowledge Graph-Enhanced Reinforcement Learning. In SIGIR '20. 179–188.

[104] Ziwei Zhu, Yun He, Xing Zhao, and James Caverlee. 2021. Popularity Bias in Dynamic Recommendation. In KDD '21. 2439–2449.

[105] Lixin Zou, Long Xia, Zhuoye Ding, Jiaxing Song, Weidong Liu, and Dawei Yin. 2019. Reinforcement Learning to Optimize Long-Term User Engagement in Recommender Systems. In KDD '19. 2810–2818.

[106] Lixin Zou, Long Xia, Pan Du, Zhuo Zhang, Ting Bai, Weidong Liu, Jian-Yun Nie, and Dawei Yin. 2020. Pseudo Dyna-Q: A Reinforcement Learning Framework for Interactive Recommendation. In WSDM '20. 816–824.

[107] Lixin Zou, Long Xia, Yulong Gu, Xiangyu Zhao, Weidong Liu, Jimmy Xiangji Huang, and Dawei Yin. 2020. Neural Interactive Collaborative Filtering. In SIGIR '20. 749–758.
