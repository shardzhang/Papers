# MUSE：一个简单而有效的基于多模态搜索的终身用户兴趣建模框架

> Bin Wu*, Feifan Yang*, Zhangming Chan, Yu-Ran Gu, Jiawei Feng, Chao Yi, Xiang-Rong Sheng, Han Zhu, Jian Xu, Mang Ye†, Bo Zheng† | 武汉大学、阿里巴巴集团

本文提出 MUSE，一个把"简单"用在召回（GSU）、把"复杂"用在精排（ESU）的终身用户兴趣建模框架。核心发现是——**在淘宝展示广告中以 100K 长度用户行为序列上线后，CTR 提升 +12.6%、ROI 提升 +11.4%，在线延迟开销几乎为零**。

核心内容：

- 现有终身兴趣建模几乎只靠 ID 特征，长尾 item 泛化差、语义表达力不足——而 MISS 等早期工作只在 GSU 里用多模态，ESU 仍只用 ID
- MUSE 的系统性实验揭示出两条设计原则：GSU 只需轻量的多模态余弦相似度（复杂检索机制无增益）；ESU 才需要显式多模态序列建模（SimTier）+ ID–语义融合（SA-TA）
- 两个阶段都用冻结的 SCL 多模态嵌入；SimTier 把目标与行为的相似度序列压成直方图，SA-TA 在 ID 注意力上叠加语义相似度加权
- 部署解耦 GSU 预取与排序关键路径，异步并行执行并把嵌入缓存在 GPU 显存中，支持 100K 长度序列

关键发现：

- **离线 GAUC：生产数据集上 MUSE 0.6377（+1.69%），全面超过 SIM-Hard/SIM-Soft/TWIN/MISS 等最强基线；开源数据集 0.6154（+1.18%）**
- **序列长度从 5K 扩展到 100K 带来 GAUC +0.38% 的提升；"GSU+ESU 都用多模态"相比"仅 GSU 用多模态"优势明显**
- 多模态表示质量对 ESU 影响远大于 GSU：SCL > I2I > OpenCLIP，语义最细的表示在 ESU 中收益最大
- 开源了第一个"超长行为序列 + 高质量多模态嵌入"的大规模数据（Taobao-MM），供社区研究与复现

---

## 摘要
终身用户兴趣建模对工业级推荐系统至关重要，然而现有方法主要依赖基于 ID 的特征，在长尾 item 上泛化能力差，且语义表达能力有限。虽然近期工作探索了在通用搜索单元（GSU，General Search Unit）中使用多模态表示进行行为检索，但这些工作往往忽视了精排建模阶段——精确搜索单元（ESU，Exact Search Unit）中的多模态融合。在本文中，我们对如何在两阶段终身建模框架的两个阶段中有效利用多模态信号进行了系统性分析。我们的关键洞察是：**在 GSU 中简单即足够**——轻量的余弦相似度配合高质量多模态嵌入即可超越复杂的检索机制。相反，**ESU 需要更丰富的多模态序列建模和有效的 ID–多模态融合**才能充分发挥其潜力。在这些原则的指导下，我们提出了 MUSE，一个简单而有效的基于多模态搜索的框架。MUSE 已在淘宝展示广告系统中部署，支持 100K 长度的用户行为序列建模，并以可忽略不计的在线延迟开销带来了显著的 top-line 指标提升。为促进社区研究，我们分享了工业部署实践，并开源了第一个"超长行为序列 + 高质量多模态嵌入"的大规模数据集。我们的代码和数据可在 https://taobao-mm.github.io 获取。
**CCS概念**：• 信息系统 $\to$ 推荐系统；
**关键词**：Click-Through Rate Prediction; Multimodal Recommendation; Long Sequential User Behavior; Recommender System
*Bin Wu 和 Feifan Yang 对本研究贡献相同。Bin Wu 在阿里巴巴集团实习期间完成了本工作。
†Mang Ye 和 Bo Zheng 为通讯作者。
Conference acronym 'XX, Woodstock, NY
© 2018 版权归作者/权利人所有。出版权授权给 ACM。
ACM ISBN 978-1-4503-XXXX-X/2018/06
https://doi.org/XXXXXXX.XXXXXXX
**ACM引用格式**：
Bin Wu, Feifan Yang, Zhangming Chan, Yu-Ran Gu, Jiawei Feng, Chao Yi, Xiang-Rong Sheng, Han Zhu, Jian Xu, Mang Ye, and Bo Zheng. 2018. MUSE: A Simple Yet Effective Multimodal Search-Based Framework for Lifelong User Interest Modeling. In Proceedings of Make sure to enter the correct conference title from your rights confirmation email (Conference acronym 'XX). ACM, New York, NY, USA, 10 pages. https://doi.org/XXXXXXX.XXXXXXX

## 1 引言
点击率（CTR，Click-Through Rate）预测[1, 7, 8, 31, 32]在工业级推荐系统中扮演着关键角色。随着用户行为日志随时间不断累积——通常每个用户超过 $10^5$ 条行为——其中蕴含着丰富且不断演化的兴趣模式。为了利用这样的终身序列，现代 CTR 预测模型采用两阶段架构[3, 19]：(1) 通用搜索单元（GSU，General Search Unit）从完整历史行为中检索出一段简短的相关子序列；(2) 精确搜索单元（ESU，Exact Search Unit）对该子序列进行细粒度的用户兴趣建模，用于最终预测。

![图1](/Users/dazhang/PycharmProject/Papers/3-RecSys/.picture/2025-MUSE-MUSE-A Simple Yet Effective Multimodal Search-Based Framework for Lifelong User Interest Modeling-fig1.png)

**图 1：** MUSE 总览。(a) 多模态 item 嵌入通过语义感知对比学习（SCL，Semantic-aware Contrastive Learning）预训练。在推荐阶段，(b) GSU 阶段利用轻量的多模态余弦相似度，从用户的终身历史中高效检索与目标 item 最相关的 top-$K$ 行为，大幅缩减后续处理的序列长度。(c) ESU 阶段通过两个组件建模细粒度用户兴趣：SimTier 模块将多模态相似度序列压缩为直方图，而语义感知目标注意力（SA-TA，Semantic-Aware Target Attention）模块赋予基于 ID 的注意力语义指导，以产生最终的终身用户兴趣表示。

然而，现有方法在两个阶段都几乎完全依赖基于 ID 的特征[3, 19]。这带来两个关键局限：(1) 长尾或过时 item 的 ID 嵌入学习不充分，损害了 GSU 的检索质量；(2) ESU 缺乏语义表达能力，因为它无法泛化到共现信号之外。虽然近期工作（如 MISS [10]）将多模态表示引入 GSU，但其 ESU 仍只建模 ID 特征，第二个局限仍未得到解决。

在本文中，我们系统性地研究了如何充分利用多模态信息来增强终身用户兴趣建模，重点是将多模态表示整合到 GSU 和 ESU 两个阶段中。通过在工业规模数据上的大量实验，我们得出了三个关键洞察：

- 在 GSU 阶段，多模态嵌入之间的简单余弦相似度即足以实现有效检索；更复杂的相似度评分机制（如注意力或 ID–多模态联合融合）带来的增益可以忽略不计。
- 在 ESU 阶段，显式的多模态序列建模（如通过 SimTier [25]）能显著提升性能，而将多模态语义融入基于 ID 的注意力（通过我们提出的 SA-TA）还能获得进一步增益。
- 在 ESU 中，表示质量比在 GSU 中更重要：捕捉细粒度 item 语义的多模态嵌入（如 SCL [25]）能产生最强的结果。

在这些洞察的指导下，我们提出了 MUSE（MUltimodal SEarch-based framework for lifelong user interest modeling，基于多模态搜索的终身用户兴趣建模框架）——一个简单而有效的范式，将我们的发现统一为一个可部署的系统（见图 1）。具体而言，MUSE 在两个阶段都采用冻结的基于 SCL 的多模态嵌入[25]：(1) 在 GSU 阶段，通过轻量的多模态余弦相似度检索最相关的 top-$K$ 行为；(2) 在 ESU 阶段，通过双路径架构设计融合 ID 与多模态信号进行终身用户兴趣建模——语义感知目标注意力（SA-TA）以高质量语义指导增强基于 ID 的注意力，而 SimTier 将多模态余弦相似度序列压缩为直方图以捕捉语义相关性。

除了算法贡献之外，我们还做出两项额外的实践贡献：首先，我们分享生产部署经验。MUSE 自 2025 年年中起已在淘宝展示广告系统中部署。集成到在线服务流水线后，它在高效处理长达 100K 的终身用户行为序列并带来可观的 top-line 指标提升的同时，产生的延迟开销可忽略不计。其次，为促进社区研究，我们开源了第一个大规模公开数据集，其中包含与高质量多模态嵌入配对的终身用户行为序列，数据采集自淘宝展示广告系统的真实用户流量。

总而言之，我们的贡献如下：
- 我们首次对终身用户兴趣建模中的多模态整合进行了系统性分析，揭示了 GSU（追求简单）和 ESU（追求丰富 + 融合）各自不同的设计原则。
- 我们提出 MUSE，一个践行这些原则的实用框架，在离线和在线评估中均达到最先进水平。
- 我们发布工业部署实践和一个大规模多模态终身行为数据集，以支持未来的研究。

## 2 预备知识
本节介绍本文中使用的关键概念和符号，重点是基于终身用户行为序列的 CTR 预测以及我们框架中采用的多模态表示。

**基于终身行为序列的 CTR 预测。** 点击率（CTR，Click-Through Rate）预测估计用户点击所展示目标 item 的概率，通常表述为二分类任务。在工业规模推荐系统中，用户往往拥有超长的历史行为序列（例如数百万次交互）。给定用户 $u$ 及其行为序列 $B_u = [b_1, \ldots, b_L]$（$L$ 可能非常大）、目标 item $a$ 以及其他特征 $o$，目标是预测：

$$
P(y = 1 \mid B_u, a, o), \qquad (1)
$$

其中 $y \in \{0, 1\}$ 表示点击标签。

现代架构（如 SIM [19] 和 TWIN [3]）通过两阶段框架设计来解决终身序列挑战：(1) 通用搜索单元（GSU，General Search Unit）检索出简短且与目标相关的子序列 $B^*_u \subset B_u$；(2) 精确搜索单元（ESU，Exact Search Unit）在 $B^*_u$ 上进行细粒度建模，用于最终 CTR 预测。我们的工作聚焦于用丰富的多模态信息增强 GSU 和 ESU 两个阶段。

**多模态 item 表示。** 为利用 item 的丰富多模态信息（如 item 图像和标题），我们考虑三种预训练的多模态嵌入方法，它们都将一个 item 映射为固定维度的向量。

- **OpenCLIP [27]**：一个开源的、面向中文适配的 CLIP 模型[21]，通过对比学习在 200M 图像–文本对上预训练。它提供强大的开箱即用的通用语义嵌入，但缺乏用户交互信号。
- **I2I（Item to Item）**：一个基于淘宝用户行为日志中 item 共现模式训练的对比模型。正样本对由频繁的 item–item 转移构成，负样本通过 MoCo [11] 采样。该方法有效地将协同信号注入多模态嵌入。
- **SCL（Semantic-aware Contrastive Learning，语义感知对比学习）[25]**：从用户"搜索–购买"行为（如查询图像 ↔ 购买 item 图像）构建正样本对。使用 InfoNCE 损失[17]训练，SCL 学到的嵌入既捕捉语义信号也捕捉行为相关性，并已被证明对 CTR 预测有效。

这三种方法都产生冻结的多模态嵌入，推理时通过查表访问。

## 3 简单在 GSU，复杂在 ESU：多模态终身建模的原则
在本节中，我们研究如何充分利用多模态信息来增强终身用户兴趣建模，系统性地聚焦于将多模态表示整合到 GSU 和 ESU 两个阶段。通过在工业规模数据上进行的大量实验，我们得出以下关键洞察：

**关键洞察**
- **GSU**：多模态相似度优于基于 ID 的检索，但增加复杂度几乎不带来增益——简单的内积即已足够。
- **ESU**：多模态序列建模显著有效，ID–多模态融合还能带来进一步增益。
- **设计原则**：简单的 GSU + 增强的 ESU 是最优配置；ESU 对表示质量的敏感度远高于 GSU。

### 3.1 简单的多模态相似度即足以实现有效的 GSU 检索
GSU 的主要作用是从终身历史序列中高效检索与目标 item 最相关的用户行为。

**表1：使用不同嵌入时的 GSU 性能。为保证公平比较，各方法的 ESU 均相同。**
| 嵌入类型 | GAUC | Δ |
|------|-----------|------|
| ID | 0.6356 | - |
| OpenCLIP | 0.6365 | +0.14% |
| I2I | 0.6370 | +0.22% |
| SCL | 0.6377 | +0.33% |

我们分析两个关键设计选择：(1) 用于相似度计算的表示类型，以及 (2) 检索机制的复杂度。

**基于多模态的 GSU vs. 基于 ID 的 GSU。** 我们将基于 ID 的 GSU 与使用多模态表示——即第 2 节中描述的 OpenCLIP、I2I 和 SCL——的变体进行对比。具体而言，我们使用各自的嵌入类型计算目标 item 与每条历史行为之间的余弦相似度，并检索最相似的 top-$K$ 行为作为 GSU 的输出。

如表 1 所示，基于多模态表示的 GSU 始终优于基于 ID 的基线，证明多模态嵌入中的语义信号比离散 ID 更能捕捉用户–item 相关性。

**表2：GSU 中检索复杂度的影响。所有变体均使用 SCL 嵌入进行多模态相似度计算。**
| 相关性度量 | GAUC | Δ |
|------|-----------|------|
| 多模态余弦 | 0.6377 | - |
| 多模态注意力分数 | 0.6369 | -0.13% |
| 多模态余弦 + ID 余弦 | 0.6379 | +0.03% |
| 多模态余弦 + ID 注意力分数 | 0.6376 | -0.01% |

增加 GSU 的建模复杂度会提升性能吗？我们探索两种增强策略：(1) 在相似度计算前应用可学习的 MLP 变换多模态嵌入（记为多模态注意力分数）；(2) 融合基于 ID 的分数（记为 ID 余弦或 ID 注意力分数）与多模态语义相似度进行联合检索。

结果总结在表 2 中。对于 (1)，我们发现注意力机制——尽管对基于 ID 的特征有效——并不能直接迁移到多模态相似度建模，反而导致性能明显下降。对于 (2)，一种朴素的融合方法（在 ID 和多模态空间中计算归一化余弦相似度并求和）几乎不带来增益。即使是使用可学习的融合权重来结合基于 ID 的注意力分数与多模态相似度，其性能也与最简单的多模态内积基线相当，却要付出更高的计算开销。

这些发现使我们得出结论：**当具备合适的多模态表示时，增加检索机制的复杂度并不必然带来性能增益**。相反，轻量的内积相似度即足以实现有效的 GSU 检索。

### 3.2 多模态表示显著增强 ESU
虽然先前的工作探索了利用多模态表示改进 GSU，但大多数方法在 ESU 中仍完全依赖基于 ID 的表示——通常通过目标注意力聚合历史行为——而忽视了多模态信号在细粒度序列建模中的潜力。在本文中，我们研究两个关键问题：(1) 多模态表示能否提升 ESU 的性能？(2) 在 ESU 中增加建模复杂度（如通过多模态–ID 融合）是否能带来额外增益，还是像在 GSU 中观察到的那样，简单设计即已足够？

为回答这些问题，我们分析两个关键设计选择：(1) 将仅 ID 的 ESU 替换为仅多模态的 ESU，以及 (2) 在 ESU 中进一步融合多模态与 ID 表示，类似于在 GSU 中探索的联合检索策略。

**表3：不同 ESU 设计的对比，评估多模态建模（SimTier）与 ID–多模态融合（SA-TA）的各自贡献。为保证公平比较，所有变体使用相同的 GSU 和 SCL 嵌入。**
| ESU 类型 | GAUC | Δ |
|------|-----------|------|
| 目标注意力 | 0.6301 | - |
| SimTier | 0.6345 | +0.70% |
| 目标注意力 + SimTier | 0.6361 | +0.95% |
| SA-TA | 0.6339 | +0.60% |
| SA-TA + SimTier | 0.6377 | +1.21% |

**基于多模态的 ESU vs. 基于 ID 的 ESU。** 对于 ESU 中的多模态序列建模，我们采用 SimTier [25] 作为代表性架构。如表 3 所示，我们评估了 SimTier 与标准基于 ID 的目标注意力模块各自以及组合的效果。结果表明 SimTier 带来了显著的性能增益——无论单独使用还是与目标注意力组合——凸显了在 ESU 中显式建模多模态语义的巨大价值。

**融合多模态与 ID 表示还能带来进一步增益。** 我们探索在 ESU 中融合多模态与 ID 表示，类似于在 GSU 中考察的联合检索策略。具体而言，我们在目标注意力机制中将基于 ID 的注意力分数与多模态语义相似度整合，产生统一的、语义感知的注意力权重。我们称这种增强方法为语义感知目标注意力（SA-TA，Semantic-Aware Target Attention），详见第 4.3 节。

如表 3 所示，我们提出的 SA-TA 在 ESU 中始终优于原始的仅 ID 目标注意力，证明多模态与 ID 表示提供了互补的信号，可以被有效融合以增强终身用户兴趣的细粒度建模。

![图2](/Users/dazhang/PycharmProject/Papers/3-RecSys/.picture/2025-MUSE-MUSE-A Simple Yet Effective Multimodal Search-Based Framework for Lifelong User Interest Modeling-fig2.png)

**图 2：** 不同多模态表示的性能。ESU 明显更青睐细粒度的表示。

### 3.3 多模态表示质量至关重要——尤其是在 ESU 中
捕捉细粒度 item 语义的高质量表示能带来更好的性能。如表 1 所示，模型性能遵循 SCL > I2I > OpenCLIP 的排序。这是因为 SCL 嵌入捕捉了更细粒度的多模态语义[25]，从而带来更准确的 GSU 检索和更强的 ESU 建模。我们在附录 A.1 中通过一个 GSU 案例研究可视化了这些表示之间的差异。

**表示质量对 ESU 的影响显著大于对 GSU 的影响。** 如图 2 所示，我们通过在两种设置下评估不同的多模态表示来量化这一效应：(1) 改变 GSU 中的表示，同时保持 ESU 中的 SCL 表示不变（GSU ONLY）；(2) 在 GSU 和 ESU 中联合替换表示（GSU & ESU）。当只改变 GSU 表示时，性能相对稳定。相反，改变 ESU 中的表示会导致显著的性能差异，细粒度的 SCL 表示带来明显的增益。这表明 **ESU 对多模态表示质量的敏感度远高于 GSU**，凸显了 ESU 中语义建模的重要性。

## 4 MUSE：一个简单而有效的多模态终身建模框架
在第 3 节洞察的指导下，我们提出 MUSE（MUltimodal SEarch-based framework for lifelong user interest modeling，基于多模态搜索的终身用户兴趣建模框架），它遵循三条关键原则：(1) 利用高质量的 SCL 表示，(2) 采用简单的、基于相似度的 GSU，(3) 设计一个多模态增强的 ESU，同时进行显式的多模态序列建模和有效的 ID–多模态融合。这一设计在效率与建模能力之间取得了有效平衡。总体架构如图 1 所示。

### 4.1 多模态表示的选择
如第 3 节所示，表示质量对性能有决定性影响——尤其是在 ESU 中。在候选者（OpenCLIP、I2I、SCL）中，SCL 凭借其细粒度语义感知产生了最强的结果。因此，MUSE 在 GSU 检索和 ESU 建模中都采用 SCL 嵌入。

### 4.2 基于多模态的 GSU
与我们关于"GSU 中简单内积相似度即已足够"的发现一致，MUSE 使用冻结的 SCL 嵌入，通过余弦相似度计算语义相关性。

具体而言，给定目标 item $a$ 和用户的终身历史行为序列 $B_u = [b_1, \ldots, b_L]$，我们首先通过查表将每个 item 映射为其预训练的多模态嵌入，得到 $v_a$ 和 $V_u = [v_1, \ldots, v_L]$。

GSU 计算目标 item 与每个行为 item 之间的相似度：

$$
r_i = \langle v_a, v_i \rangle. \qquad (2)
$$

然后我们选择具有最高 $r_i$ 分数的 top-$K$ 行为，构成子序列 $B^*_u$。这种轻量级的检索与我们"当具备高质量多模态嵌入时，复杂机制没有增益"的洞察相符。

### 4.3 多模态增强的 ESU
与 GSU 相反，我们的分析表明，ESU 需要显式的多模态序列建模和 ID 与语义信号的融合，才能充分利用终身行为数据。为此，MUSE 的 ESU 由两个互补的组件构成：(1) 一个多模态序列建模模块（SimTier），捕捉检索到的行为序列中的细粒度语义相关性模式；(2) 一个多模态语义感知 ID 注意力模块（SA-TA），用多模态语义指导增强传统的基于 ID 的注意力。

**通过 SimTier 进行显式多模态序列建模。** SimTier [25] 并不处理原始的多模态嵌入，而是对目标 item 与 GSU 检索到的行为 $B^*_u$ 之间的语义相似度序列 $R = [r_1, \ldots, r_K]$ 进行操作。这一设计与我们的洞察一致：ESU 受益于建模语义相关性的结构，而非单个嵌入。具体而言，相似度范围 $[-1, 1]$ 被均匀划分为 $N$ 个 tier（分箱）。SimTier 将相似度序列 $R$ 映射为一个 $N$ 维直方图 $h_{MM} \in \mathbb{R}^N$，其中每个条目统计落入对应 tier 的相似度分数个数。该直方图作为用户多模态兴趣分布的紧凑而富有表现力的表示：

$$
h_{MM} = \text{Histogram}(R). \qquad (3)
$$

这种方法构成了我们的显式多模态序列建模——它在单一向量中捕捉多层次语义模式（例如，有多少行为是高度/弱相关的）。

**基于 ID 注意力中的语义感知融合（SA-TA）。** 虽然 SimTier 有效地捕捉了多模态语义，但 ID 路径对于捕捉协同过滤信号仍然必不可少。然而，如第 3 节所述，仅 ID 的注意力在终身用户行为中的长尾 item 上泛化能力差。为解决这一问题，我们提出语义感知目标注意力（SA-TA，Semantic-Aware Target Attention），如图 3 所示，它将基于 ID 的注意力分数与多模态语义相似度信号相融合。给定 ID 嵌入 $e_a \in \mathbb{R}^D$（目标）和 $E_u \in \mathbb{R}^{K \times D}$（行为），标准目标注意力[32]的计算方式为：

$$
\alpha_{ID} = \frac{(E_u W_k)(e_a W_q)^\top}{\sqrt{D}}.
$$

![图3](/Users/dazhang/PycharmProject/Papers/3-RecSys/.picture/2025-MUSE-MUSE-A Simple Yet Effective Multimodal Search-Based Framework for Lifelong User Interest Modeling-fig3.png)

**图 3：** 语义感知目标注意力（SA-TA）通过融入多模态语义相似度来增强基于 ID 的注意力。

SA-TA 用多模态相似度向量 $R$（来自公式(2)）对该注意力进行扩充，形成融合的注意力分数：

$$
\alpha_{Fusion} = \gamma_1 \alpha_{ID} + \gamma_2 R + \gamma_3(\alpha_{ID} \odot R), \qquad (4)
$$

其中 $\gamma_1, \gamma_2, \gamma_3$ 是可学习的标量。最终的基于 ID 的终身用户兴趣表示由下式得到：

$$
u^{ID}_l = \text{Softmax}(\alpha_{Fusion})^\top (E_u W_v). \qquad (5)
$$

这一设计实现了我们的洞察：ID 和多模态信号是互补的——SA-TA 使用多模态语义相似度来增强 ID 注意力，尤其是对稀疏 item。

最后，完整的终身用户兴趣表示通过拼接两条路径得到：

$$
u_l = \left[ h_{MM}, u^{ID}_l \right],
$$

随后送入预测塔进行 CTR 预测。

总而言之，MUSE 的 ESU 直接实现了第 3 节的两个关键发现：(1) 显式的多模态序列建模相比仅 ID 的基线有显著提升，(2) 在注意力中融合 ID 和多模态信号以获得进一步增益。

## 5 系统部署
工业规模推荐系统必须在严格的延迟预算内——通常为几百毫秒——从数十亿 item 的语料库中返回个性化结果。为满足这一约束，生产系统通常采用多级流水线：候选生成（匹配）阶段检索出数量级为 $10^3$ 的可管理候选集，随后是计算密集的排序阶段，对候选进行细粒度建模。在淘宝展示广告系统中，我们在排序阶段部署 MUSE，以有效建模长达 100K 交互的终身用户行为序列。主要的性能瓶颈来自获取这些长行为序列及其 GSU 中嵌入所需的大量网络通信，这可能使系统超出延迟预算。请注意，这一瓶颈与候选 item 无关，因为一旦请求到达，用户就是固定的。为消除这一瓶颈，我们将 GSU 的特征/嵌入获取从排序关键路径中解耦，使其与匹配阶段异步并行执行，并将嵌入缓存在 GPU 显存中。在排序阶段，我们执行相似度计算、top-$K$ 选择和 ESU 建模：

![图4](/Users/dazhang/PycharmProject/Papers/3-RecSys/.picture/2025-MUSE-MUSE-A Simple Yet Effective Multimodal Search-Based Framework for Lifelong User Interest Modeling-fig4.png)

**图 4：** MUSE 在淘宝展示广告系统中的在线部署。GSU 与匹配阶段并行地异步预取用户行为序列和多模态嵌入，缓存的输出被 GSU top-$K$ 选择和 ESU 建模在排序时消费。

- **GSU 预取。** 与匹配并行，GSU 服务器从远程存储检索用户行为序列和多模态嵌入，并将其缓存在 GPU 显存中。预取通常比匹配完成得更快，因此其延迟被完全隐藏。
- **GSU 的 top-$K$ 选择：** 在排序阶段，模型使用缓存的嵌入计算相似度，并选择最相关的 top-$K$ 历史交互送入 ESU。请注意，相似度计算和 top-$K$ 选择可以与排序阶段的其他特征处理操作并行执行，产生的延迟可忽略不计。
- **ESU 建模。** 给定选出的 top-$K$ 子序列，排序模型在 ESU 中应用 SimTier 和 SA-TA，并产生最终的 CTR 预测。

借助上述异步设计，MUSE 在生产中带来的增量延迟可忽略不计。其高效的两阶段设计将资源消耗控制在预算之内。自 2025 年年中部署以来，MUSE 已承担大部分流量，带来稳定性能和可观增益。

## 6 数据集构建
MUSE 框架面向工业规模的推荐场景，要求通过超长行为序列和丰富的多模态 item 内容来建模终身用户兴趣。这些能力对淘宝展示广告平台等真实系统至关重要。

然而，现有的公开推荐数据集[9, 16, 28]通常缺少长用户行为序列或全面的多模态特征。为彻底评估和验证 MUSE 的有效性，我们在两个源自淘宝展示广告系统的自建数据集上开展实验，两者都具有长用户行为序列和多模态 item 表示。

第一个数据集在生产数据上封顶采样，完全反映真实流量，用于在工业环境中验证 MUSE 的性能。第二个是精心策划的学术数据集，经过采样和匿名化处理以供研究使用。值得注意的是，我们将开源这一高质量数据集，以促进社区对多模态推荐和终身用户兴趣建模的研究：

**表4：工业生产和开源学术数据集的统计信息。这里，Distinct Items 表示用户行为和目标 item 覆盖的唯一 item 数量。**
| 字段 | 生产数据集 | 开源数据集 |
|------|-----------|-----------|
| 样本数 | 3.71B | 99.0M |
| 用户数 | 0.19B | 8.79M |
| 去重 item 数 | 23.6B | 35.4M |
| 最大行为长度 | 100K | 1K |

### 6.1 工业生产数据集
工业数据集由淘宝展示广告系统中的曝光日志构建。在我们的实验中，我们使用一周的数据：前六天用于训练，最后一天用于测试。该数据集中的每个样本包含数百个精心设计的特征。如表 4 所总结，该数据集包含来自 0.19B 用户的 3.71B 个样本，每个样本带有 100K 条历史行为。该数据集使得对 MUSE 等终身建模方法进行严格的离线评估成为可能。

### 6.2 开源学术数据集
为促进可复现的研究，我们发布一个保护隐私的开源学术数据集。它包含一组精炼的关键特征以及长达数千次交互的用户行为序列。由于产品图像和标题的版权限制，我们不提供原始多模态内容。相反，我们为所有 item 提供基于 SCL 的多模态嵌入[25]——这些高质量表示同时捕捉语义和行为相关性，如第 3 节所验证。

该数据集包含以下组成部分：
- **用户特征**：匿名化的用户 ID、年龄、性别、城市和省份。
- **item 特征**：item ID、类别、item 城市和省份。
- **行为序列**：用户的历史序列由最多 1K 条行为组成，每条行为由其 item ID 标识。每个 item 仅用一个 128 维的 SCL 嵌入表示。
- **标签**：一个二值标签，指示点击（1）或未点击（0）。

如表 4 所示，开源数据集包含来自 8.86M 用户的 107M 个样本，覆盖 275M 个唯一 item。最大序列长度被限制在 1K，以在研究效用和分发可行性之间取得平衡。

我们相信，这个将长用户行为序列与高质量多模态嵌入结合起来的数据集，将推动多模态终身兴趣建模的系统性探索。该数据集连同预处理脚本和基线实现现已公开发布¹²。

## 7 实验
我们开展综合实验，在 CTR 预测的背景下评估 MUSE。

¹数据集可在 https://huggingface.co/datasets/TaoBao-MM/Taobao-MM 获取
²代码可在 https://github.com/TaoBao-MM/MUSE 获取

### 7.1 实验设置
#### 7.1.1 基线方法。我们将 MUSE 与以下最先进且被广泛采用的终身兴趣建模方法进行对比：
- **DIN [32]**：该方法引入目标注意力机制，从历史行为中建模用户兴趣，是短序列兴趣建模中应用最广泛的模型。
- **SIM-Hard [19]**：该方法采用两阶段方法进行长序列建模。在 GSU 阶段，检索与目标 item 类别相同的 top-$K$ 行为。在 ESU 阶段，应用 DIN 建模用户的细粒度兴趣。
- **SIM-Soft [19]**：该变体将 SIM-hard 在 GSU 阶段的基于类别的检索替换为基于嵌入的相似度搜索，同时在 ESU 阶段仍采用 DIN 进行用户细粒度兴趣建模。
- **TWIN [3]**：该方法提出一种高效的、可应用于 GSU 和 ESU 两个阶段的目标注意力机制，使得注意力检索在 GSU 阶段即可进行。
- **MISS [10]**：该方法采用混合检索策略：基于 ID 的 Co-GSU 和基于多模态的 MM-GSU。在 ESU 阶段，它使用基于 ID 的 DIN 进行用户兴趣建模。

#### 7.1.2 实现细节。为保证公平比较，除长期行为建模模块外，所有方法共享相同的网络架构，包括嵌入层和上层 MLP 塔。所有方法一致使用 SCL [25] 作为多模态表示。每个模型训练一个 epoch [30, 34]。对于 DIN，由于其处理长序列存在瓶颈，我们使用最近的 50 条行为。对于两阶段模型（SIM、TWIN、MISS、MUSE），GSU 从终身行为序列中检索 50 条行为供 ESU 使用。完整行为序列长度在生产数据集上限制为 5,000，在开源数据集上限制为 1,000。我们严格按照原论文复现每种方法的 GSU。对于 SIM-soft 和 SIM-hard，我们采用与 MUSE 相同的 ESU（SA-TA 和 SimTier）。考虑到 TWIN 使用一种特殊的目标注意力变体以提高计算效率，我们保留其原始设计，并将 SimTier 集成到 TWIN 的 ESU 中。对于 MISS，我们保留其原始的仅 ID ESU，以说明多模态增强 ESU 的显著效果。

在开源数据集上，我们对所有参数组使用 Adam 训练所有模型：嵌入层的学习率为 2.0e-3，DNN 参数的学习率为 2.0e-4。批大小为 8,000。

#### 7.1.3 评估指标。我们采用 Group AUC（GAUC，分组 AUC）——一种被证明与在线指标更为一致的度量[23, 24, 32]。

### 7.2 总体性能
表 5 显示了所有模型在生产数据集和开源数据集上的性能，MUSE 在两个数据集上均始终取得最佳结果。这些对比进一步验证了第 3 节确立的洞察，证实了我们方法的有效性。

**表5：生产数据集和开源数据集上的总体性能。这里，为加速实验，生产数据集中的用户行为序列从 100K 交互截断为 5K。Base 模型设为 DIN。**
| 方法 | Production-5k GAUC | Production-5k Δ | Open-Source-1k GAUC | Open-Source-1k Δ |
|------|-----------|-----------|-----------|-----------|
| Base | 0.6271 | - | 0.6082 | - |
| SIM-Hard | 0.6351 | +1.27% | 0.6139 | +0.94% |
| SIM-Soft | 0.6356 | +1.35% | 0.6135 | +0.87% |
| TWIN | 0.6304 | +0.53% | 0.6098 | +0.26% |
| MISS | 0.6308 | +0.59% | 0.6097 | +0.25% |
| MUSE | 0.6377 | +1.69% | 0.6154 | +1.18% |

首先，基于多模态的 GSU 优于基于 ID 的 GSU。在与相同 ESU 配合时，MUSE 的性能优于依赖基于 ID 相似度搜索的 SIM-hard 和 SIM-soft。其次，复杂的 GSU 机制几乎不带来改进。TWIN 采用注意力机制建模目标–行为相关性，但性能有限，部分原因在于基于 ID 嵌入内积的注意力分数并不可靠，尤其是对长尾 item。回顾第 3.1 节，我们也在 GSU 中使用了 SA-TA 的融合注意力分数，这带来了额外的计算开销，却没有超越 MUSE 更简单的多模态语义相似度搜索。最后，多模态增强的 ESU 带来了显著改进。尽管 MISS 引入了额外的多模态 GSU 阶段，但其在 ESU 阶段并未利用多模态信息，导致结果次优。

![图5](/Users/dazhang/PycharmProject/Papers/3-RecSys/.picture/2025-MUSE-MUSE-A Simple Yet Effective Multimodal Search-Based Framework for Lifelong User Interest Modeling-fig5.png)

**图 5：** GSU 采用不同行为序列长度时的性能。左：GAUC 值。右：在不同序列长度下，使用 MM-Enhanced ESU 的 MUSE 相比 ID-Only ESU 的相对 GAUC 提升。

### 7.3 对行为序列长度的影响
我们评估 MUSE 在不同行为序列长度下的有效性。如图 5 所示，我们观察到：(1) 随着序列长度增加，MUSE 性能显著提升，从 5K 扩展到 100K 条行为带来 GAUC +0.38% 的增益；(2) 在所有长度下，多模态增强的 ESU 都大幅稳定地优于仅 ID 的 ESU。这与我们"ESU 受益于高质量多模态表示"的洞察一致。

### 7.4 在不同用户和 item 上的性能
为评估 MUSE 在不同用户活跃度水平下的表现，我们按行为序列长度将用户划分为九个等大小的组。如图 6 (a) 所示，相对 GAUC 提升从第 1 组（最短）到第 9 组（最长）单调递增，证实 MUSE 从丰富的行为上下文中获益最多。

![图6](/Users/dazhang/PycharmProject/Papers/3-RecSys/.picture/2025-MUSE-MUSE-A Simple Yet Effective Multimodal Search-Based Framework for Lifelong User Interest Modeling-fig6.png)

**图 6：** 按行为长度划分的用户组及按曝光频率划分的 item 组上的相对 GAUC 提升。

类似地，我们将 item 划分为九个等大小的组，从第 1 组（最不流行）到第 9 组（最流行），并计算每组的相对 GAUC 提升。如图 6 (b) 所示，MUSE 在利基（niche）item 上获得更高的增益，展示了多模态信息的强大泛化能力。

### 7.5 在线 A/B 测试结果
**表6：所提出的 MUSE 框架相比生产基线的在线性能提升。**
| 指标 | CTR | RPM | ROI |
|------|------|------|------|
| 提升 | +12.6% | +5.1% | +11.4% |

我们在 2025 年年中在淘宝展示广告系统中实现了 MUSE（序列长度为 100K），并通过长期 A/B 测试评估其在线性能。生产基线基于行为长度为 5K 的 SIM。如表 6 所示，100K 长度版本的 MUSE 部署带来了可观的性能提升，在 CTR（点击率）上实现 +12.6%、在 RPM（千次展示收入，Revenue Per Mille）上实现 +5.1%、在 ROI（投资回报率，Return On Investment）上实现 +11.4% 的增益。

## 8 相关工作
我们的工作与两个活跃研究方向相交：(1) 终身用户兴趣建模，和 (2) 多模态推荐。

### 8.1 终身用户兴趣建模
建模超长用户行为序列对于捕捉不断演化的兴趣至关重要。早期的如 MIMN [18] 等方法使用记忆网络来随时间维护用户兴趣状态。一个重大突破来自 SIM [19] 和 UBR4CTR [20] 引入的两阶段框架，它将检索与建模解耦：通用搜索单元（GSU，General Search Unit）从完整历史中高效检索最相关的 $K$ 条行为，精确搜索单元（ESU，Exact Search Unit）在该子序列上进行细粒度序列建模，用于最终预测。

后续工作进一步完善了 GSU 的设计：ETA [5] 使用带汉明距离的局部敏感哈希进行快速检索；SDIM [2] 按哈希签名对 item 分组；TWIN [3] 将目标注意力[32]扩展到 GSU，选择具有最高注意力分数的行为。然而，由于注意力分数是为行为聚合而非行为选择优化的，因此它们对检索而言是次优的[6]。更根本的是，所有这些方法都仅依赖基于 ID 的特征，继承了众所周知的局限：对长尾 item 泛化能力差、缺乏语义表达能力。这些缺陷促使近期工作——包括我们的工作——将多模态信号整合到终身建模中。

### 8.2 多模态推荐
多模态信息（如图像、文本）在提升推荐准确性方面已展现出巨大潜力[4, 12, 26, 29, 33]。AlignRec [14] 在协同过滤监督下将固定 CLIP 特征与 ID 嵌入对齐。Sheng 等人 [25] 提出语义感知对比学习（SCL，Semantic-aware Contrastive Learning），在预训练期间构建行为接地的正样本对（如搜索查询 ↔ 购买 item），然后使用这些固定表示通过 SimTier 和 MAKE 建模用户兴趣。QARM [15] 在预训练期间将多模态特征与 item–item 关系对齐，并通过量化码[13, 22]实现端到端优化。与我们的工作最相关的是，MISS [10] 通过引入与基于 ID 的多模态 GSU 并行的多模态 GSU，开创了终身行为建模中的多模态整合。然而，MISS 在 ESU 中丢弃了多模态信号，将语义增强限制在 GSU 检索阶段，使建模阶段仍易受 ID 嵌入局限性的影响。

与此相反，我们提出了在 GSU 和 ESU 两个阶段系统性地整合多模态信息。我们的分析揭示了每个阶段各自不同的设计原则——GSU 中简单即足够，而 ESU 受益于显式的多模态序列建模和 ID–语义融合——从而引出一个简单而高效的框架 MUSE。

## 9 结论
在本文中，我们系统性地研究了如何将多模态表示有效地整合到终身用户兴趣建模中。通过大量实验，我们揭示了两阶段架构各自不同的设计原则：通用搜索单元（GSU，General Search Unit）受益于简单——高质量多模态嵌入配合内积相似度即已足够——而精确搜索单元（ESU，Exact Search Unit）需要显式的多模态序列建模以及 ID 与语义信号的有效融合。在这些洞察的指导下，我们提出 MUSE，一个在离线和在线评估中均达到最先进水平的简单而有效的框架。MUSE 自 2025 年年中起已在淘宝展示广告系统中部署，以可忽略不计的延迟开销带来了可观的 top-line 业务指标增益。此外，我们发布了工业部署实践和第一个大规模开源数据集，其中包含与基于 SCL 的多模态嵌入配对的超长行为序列，以支持未来的研究。我们希望我们的分析、框架和资源能激发学术界和工业界对多模态增强终身兴趣建模的更有效和高效的方法。

## 致谢
我们衷心感谢 Jing Huang、Jinjing Li、Jiawen Liao、Zhenyuan Lai、Wenchao Wang、Chaochao Zhao、Yunlong Xu、Zhengxiong Zhou、Huimin Yi、Xingyu Wen、Dun Yang、Yan Zhang、Jinzhe Shan、Gaoming Zhou、Xiang Gao、Rui Du、Xiaorui Zhang、Qifeng Li、Jiamang Wang、Peng Sun 以及为 MUSE 提供宝贵工程支持的其他同事。

## 参考文献
[1] Weijie Bian, Kailun Wu, Lejian Ren, Qi Pi, Yujing Zhang, Can Xiao, Xiang-Rong Sheng, Yong-Nan Zhu, Zhangming Chan, Na Mou, Xinchen Luo, s Shiming Xiang, Guorui Zhou, Xiaoqiang Zhu, and Hongbo Deng. 2022. CAN: Feature Co-Action Network for Click-Through Rate Prediction. In Proceedings of the 15th ACM International Conference on Web Search and Data Mining. 57–65.
[2] Yue Cao, Xiaojiang Zhou, Jiaqi Feng, Peihao Huang, Yao Xiao, Dayao Chen, and Sheng Chen. 2022. Sampling Is All You Need on Modeling Long-Term User Behaviors for CTR Prediction. In Proceedings of the 31st ACM International Conference on Information and Knowledge Management. 2974–2983.
[3] Jianxin Chang, Chenbin Zhang, Zhiyi Fu, Xiaoxue Zang, Lin Guan, Jing Lu, Yiqun Hui, Dewei Leng, Yanan Niu, Yang Song, and Kun Gai. 2023. TWIN: TWo-Stage Interest Network for Lifelong User Behavior Modeling in CTR Prediction at Kuaishou. In Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 3785–3794.
[4] Junxuan Chen, Baigui Sun, Hao Li, Hongtao Lu, and Xian-Sheng Hua. 2016. Proceedings of the 24th ACM international conference on Multimedia. 811–820.
[5] Qiwei Chen, Changhua Pei, Shanshan Lv, Chao Li, Junfeng Ge, and Wenwu Ou. 2021. End-to-end user behavior retrieval in click-through rateprediction model. arXiv preprint arXiv:2108.04468 (2021).
[6] Ningya Feng, Junwei Pan, Jialong Wu, Baixu Chen, Ximei Wang, Qian Li, Xian Hu, Jie Jiang, and Mingsheng Long. 2025. Long-Sequence Recommendation Models Need Decoupled Embeddings. In International Conference on Learning Representations.
[7] Yufei Feng, Fuyu Lv, Weichen Shen, Menghan Wang, Fei Sun, Yu Zhu, and Keping Yang. 2019. Deep session interest network for click-through rate prediction. In Proceedings of the 28th International Joint Conference on Artificial Intelligence. 2301–2307.
[8] Yufei Feng, Fuyu Lv, Weichen Shen, Menghan Wang, Fei Sun, Yu Zhu, and Keping Yang. 2019. Deep session interest network for click-through rate prediction. In Proceedings of the 28th International Joint Conference on Artificial Intelligence. 2301–2307.
[9] Chongming Gao, Shijun Li, Wenqiang Lei, Jiawei Chen, Biao Li, Peng Jiang, Xiangnan He, Jiaxin Mao, and Tat-Seng Chua. 2022. KuaiRec: A Fully-observed Dataset and Insights for Evaluating Recommender Systems. In Proceedings of the 31st ACM International Conference on Information and Knowledge Management. 540–550.
[10] Chengcheng Guo, Junda She, Kuo Cai, Shiyao Wang, Qigen Hu, Qiang Luo, Guorui Zhou, and Kun Gai. 2025. MISS: Multi-Modal Tree Indexing and Searching with Lifelong Sequential Behavior for Retrieval Recommendation. In Proceedings of the 34th ACM International Conference on Information and Knowledge Management. 5683–5690.
[11] Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. 2020. Momentum Contrast for Unsupervised Visual Representation Learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 9729–9738.
[12] Wang-Cheng Kang, Chen Fang, Zhaowen Wang, and Julian McAuley. 2017. Visually-Aware Fashion Recommendation and Design with Generative Image Models. In 2017 IEEE International Conference on Data Mining (ICDM). 207–216.
[13] Doyup Lee, Chiheon Kim, Saehoon Kim, Minsu Cho, and Wook-Shin Han. 2022. Autoregressive Image Generation Using Residual Quantization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 11523–11532.
[14] Yifan Liu, Kangning Zhang, Xiangyuan Ren, Yanhua Huang, Jiarui Jin, Yingjie Qin, Ruilong Su, Ruiwen Xu, Yong Yu, and Weinan Zhang. 2024. AlignRec: Aligning and Training in Multimodal Recommendations. In Proceedings of the 33rd ACM International Conference on Information and Knowledge Management. 1503–1512.
[15] Xinchen Luo, Jiangxia Cao, Tianyu Sun, Jinkai Yu, Rui Huang, Wei Yuan, Hezheng Lin, Yichen Zheng, Shiyao Wang, Qigen Hu, Changqing Qiu, Jiaqi Zhang, Xu Zhang, Zhiheng Yan, Jingming Zhang, Simin Zhang, Mingxing Wen, Zhaojie Liu, Kun Gai, and Guorui Zhou. 2025. QARM: Quantitative Alignment Multi-Modal Recommendation at Kuaishou. 5915–5922 pages.
[16] Yongxin Ni, Yu Cheng, Xiangyan Liu, Junchen Fu, Youhua Li, Xiangnan He, Yongfeng Zhang, and Fajie Yuan. 2023. A Content-Driven Micro-Video Recommendation Dataset at Scale. arXiv:2309.15379 [cs.IR]
[17] Aaron van den Oord, Yazhe Li, and Oriol Vinyals. 2018. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748 (2018).
[18] Qi Pi, Weijie Bian, Guorui Zhou, Xiaoqiang Zhu, and Kun Gai. 2019. Practice on Long Sequential User Behavior Modeling for Click-Through Rate Prediction. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. New York, NY, USA, 2671–2679.
[19] Qi Pi, Guorui Zhou, Yujing Zhang, Zhe Wang, Lejian Ren, Ying Fan, Xiaoqiang Zhu, and Kun Gai. 2020. Search-Based User Interest Modeling with Lifelong Sequential Behavior Data for Click-Through Rate Prediction. In Proceedings of the 29th ACM International Conference on Information and Knowledge Management. 2685–2692.
[20] Jiarui Qin, Weinan Zhang, Xin Wu, Jiarui Jin, Yuchen Fang, and Yong Yu. 2020. User Behavior Retrieval for Click-Through Rate Prediction. In Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval. 2347–2356.
[21] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. 2021. Learning Transferable Visual Models From Natural Language Supervision. In Proceedings of the 38th International Conference on Machine Learning. 8748–8763.
[22] Shashank Rajput, Nikhil Mehta, Anima Singh, Raghunandan Keshavan, Trung Vu, Lukasz Heidt, Lichan Hong, Yi Tay, Vinh Q. Tran, Jonah Samost, Maciej Kula, Ed H. Chi, and Maheswaran Sathiamoorthy. 2023. Recommender systems with generative retrieval. In Proceedings of the 37th International Conference on Neural Information Processing Systems.
[23] Xiang-Rong Sheng, Jingyue Gao, Yueyao Cheng, Siran Yang, Shuguang Han, Hongbo Deng, Yuning Jiang, Jian Xu, and Bo Zheng. 2023. Joint Optimization of Ranking and Calibration with Contextualized Hybrid Model. In Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 4813–4822.
[24] Xiang-Rong Sheng, Liqin Zhao, Guorui Zhou, Xinyao Ding, Binding Dai, Qiang Luo, Siran Yang, Jingshan Lv, Chi Zhang, Hongbo Deng, and Xiaoqiang Zhu. 2021. One Model to Serve All: Star Topology Adaptive Recommender for Multi-Domain CTR Prediction. In Proceedings of The 30th ACM International Conference on Information and Knowledge Management. 4104–4113.
[25] Xiang-Rong Sheng, Feifan Yang, Litong Gong, Biao Wang, Zhangming Chan, Yujing Zhang, Yueyao Cheng, Yong-Nan Zhu, Tiezheng Ge, Han Zhu, Yuning Jiang, Jian Xu, and Bo Zheng. 2024. Enhancing Taobao Display Advertising with Multimodal Representations: Challenges, Approaches and Insights. In Proceedings of the 33rd ACM International Conference on Information and Knowledge Management. 4858–4865.
[26] Yinwei Wei, Xiang Wang, Liqiang Nie, Xiangnan He, Richang Hong, and Tat-Seng Chua. 2019. MMGCN: Multi-modal Graph Convolution Network for Personalized Recommendation of Micro-video. In Proceedings of the 27th ACM International Conference on Multimedia. 1437–1445.
[27] An Yang, Junshu Pan, Junyang Lin, Rui Men, Yichang Zhang, Jingren Zhou, and Chang Zhou. 2022. Chinese CLIP: Contrastive Vision-Language Pretraining in Chinese.
[28] Guanghu Yuan, Fajie Yuan, Yudong Li, Beibei Kong, Shujie Li, Lei Chen, Min Yang, Chenyun Yu, Bo Hu, Zang Li, et al. 2022. Tenrec: A large-scale multipurpose benchmark dataset for recommender systems. Advances in Neural Information Processing Systems, 11480–11493.
[29] Zheng Yuan, Fajie Yuan, Yu Song, Youhua Li, Junchen Fu, Fei Yang, Yunzhu Pan, and Yongxin Ni. 2023. Where to Go Next for Recommender Systems? ID- vs. Modality-based Recommender Models Revisited. In Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval. 2639–2649.
[30] Zhao-Yu Zhang, Xiang-Rong Sheng, Yujing Zhang, Biye Jiang, Shuguang Han, Hongbo Deng, and Bo Zheng. 2022. Towards Understanding the Overfitting Phenomenon of Deep Click-Through Rate Models. In Proceedings of the 31st ACM International Conference on Information & Knowledge Management. 2671–2680.
[31] Guorui Zhou, Na Mou, Ying Fan, Qi Pi, Weijie Bian, Chang Zhou, Xiaoqiang Zhu, and Kun Gai. 2019. Deep interest evolution network for click-through rate prediction. In Proceedings of the AAAI conference on artificial intelligence. 5941–5948.
[32] Guorui Zhou, Xiaoqiang Zhu, Chenru Song, Ying Fan, Han Zhu, Xiao Ma, Yanghui Yan, Junqi Jin, Han Li, and Kun Gai. 2018. Deep Interest Network for Click-Through Rate Prediction. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. 1059–1068.
[33] Xin Zhou, Hongyu Zhou, Yong Liu, Zhiwei Zeng, Chunyan Miao, Pengwei Wang, Yuan You, and Feijun Jiang. 2023. Bootstrap Latent Representations for Multimodal Recommendation. In Proceedings of the ACM Web Conference 2023. 845–854.
[34] Hua Zong, Qingtao Zeng, Zhengxiong Zhou, Zhihua Han, Zhensong Yan, Mingjie Liu, Hechen Sun, Jiawei Liu, Yiwen Hu, Qi Wang, et al. 2025. RecIS: Sparse to Dense, A Unified Training Framework for Recommendation Models. arXiv preprint arXiv:2509.20883 (2025).

## 附录
### A.1 GSU 案例研究
我们在图 7 中展示了三个案例，以说明不同类型 GSU 在不同目标 item（即儿童防晒衣、发夹和移动 Wi-Fi）上的不同检索结果：

![图7](/Users/dazhang/PycharmProject/Papers/3-RecSys/.picture/2025-MUSE-MUSE-A Simple Yet Effective Multimodal Search-Based Framework for Lifelong User Interest Modeling-fig7.png)

**图 7：** 不同 GSU 类型的检索结果可能截然不同的案例。

- **MUSE with SCL**：使用 SCL 嵌入的多模态 GSU 能精确捕捉区分单个 item 的视觉细节，展现出细粒度语义匹配的能力。它能准确检索出在风格、形状、颜色、质感等方面与目标 item 高度相似的历史 item。
- **MUSE with OpenCLIP**：相比之下，基于 OpenCLIP 嵌入的 GSU 以更粗糙的粒度进行检索，关注整体图像内容。它常常基于无关的背景或上下文相似度来检索 item，而非 item 本身。例如，在案例 1 中检索时过度强调了儿童模特，在案例 2 中过度强调了商标，在案例 3 中过度强调了文字"3000G"。
- **SIM-hard**：基于类别的 GSU 能正确识别与目标 item 属于同一类别的 item，但忽略了颜色、风格等细粒度属性——而这些往往是用户最关心的方面。
- **SIM-soft**：基于 ID 的 GSU 优先选择在用户交互历史中与目标 item 频繁共现的 item，导致较高的流行度偏差和较低语义相关性。例如，案例 1 中的儿童玩具、案例 2 中的女性用品和案例 3 中的电子设备，都是主要因共现而非与目标 item 的语义相似而被检索出来的。