# Towards Large Generative Recommendation: A Tokenization Perspective（中文翻译）


本文介绍了 Towards Large Generative Recommendation: A Tokenization Perspective。核心内容：


关键发现：

---


## 摘要

> 侯宇鹏¹、张安²、盛乐恒³、吴建灿²、王翔²、Tat-Seng Chua³、Julian McAuley¹
> ¹UC San Diego、²中国科学技术大学、³新加坡国立大学
> CIKM 2025 Tutorial

---

## 概述

本教程从**分词化（Tokenization）**的视角系统性地介绍了大规模生成式推荐。生成式推荐旨在利用自回归生成模型直接从语料库中生成推荐结果，其核心挑战在于如何将推荐数据（用户、item、交互）表示为适合生成模型的token序列。

---

## 1 背景与介绍

### 1.1 什么是生成模型？

生成模型学习数据的基本分布并能从中生成新样本。与判别模型（学习决策边界）不同，生成模型关注建模数据本身的分布 p(x)。

### 1.2 为什么需要生成式推荐？

传统推荐系统采用"检索-排序"级联架构，每阶段独立优化，存在信息损失和不一致问题。生成式推荐通过统一的序列生成过程，简化系统设计并实现端到端优化。

### 1.3 缩放定律

缩放定律为理解模型规模、数据量和计算量对性能的影响提供了框架。在生成式推荐中，缩放定律表明更大的模型和数据可以带来一致的性能提升。

---

## 2 item分词化

item分词化是生成式推荐的核心问题——如何将item表示为离散token。本教程覆盖以下方法：

### 2.1 传统 ID 映射

- **原子 ID**：为每个item分配唯一整数 ID
- **局限性**：缺乏语义信息，无法泛化到冷启动item

### 2.2 语义 ID

- **层次化聚类**：基于item内容嵌入进行层次化聚类（如 RecForest 中的 K-means 树）
- **残差量化（RQ-VAE）**：通过多级残差量化生成层次化语义编码（如 TIGER）
- **有限标量量化（FSQ）**：通过独立标量量化获得离散token（如 RecGPT）

### 2.3 文本 ID

- **标题/描述作为 ID**：直接使用自然语言文本作为item标识
- **优势**：零样本泛化、与 LLM 天然对齐
- **挑战**：推理效率、生成准确度

### 2.4 比较

| 方法 | 语义性 | 冷启动 | 效率 | 跨域泛化 |
|------|--------|--------|------|---------|
| 原子 ID | 低 | 差 | 高 | 差 |
| 语义 ID | 中 | 中 | 中 | 中 |
| 文本 ID | 高 | 好 | 低 | 好 |

---

## 3 模型架构

### 3.1 编码器-解码器架构

- 编码器处理用户历史序列
- 解码器以自回归方式生成itemtoken
- 代表工作：T5、P5、TIGER

### 3.2 仅解码器架构

- 使用因果注意力统一处理输入和生成
- 更接近 LLM 的主流范式
- 代表工作：RecGPT-7B、OneRec

### 3.3 混合注意力

- 结合双向和因果注意力
- 在item内部使用双向注意力，序列之间使用因果注意力
- 代表工作：RecGPT（HKU）、BERT4Rec

---

## 4 训练与对齐

### 4.1 预训练

- **下一个token预测（NTP）**：标准自回归训练目标
- **掩码建模**：类似 BERT 的预训练方式
- **对比学习**：增强表示判别力

### 4.2 偏好对齐

- **DPO**：直接偏好优化
- **RLHF**：基于人类反馈的强化学习
- **迭代对齐**：通过自生成数据进行迭代改进

### 4.3 多任务学习

统一模型同时处理多个推荐任务：序列推荐、评分预测、解释生成等。通过在共享表示空间中的多任务训练，实现知识的跨任务迁移。

---

## 5 推理与部署

### 5.1 束搜索解码

生成式推荐的标准解码方式。束大小是精度-效率权衡的关键超参数。

### 5.2 目录感知解码

确保生成的token序列映射到目录中存在的item。对于语义 ID 和文本 ID 特别重要。

### 5.3 KV 缓存优化

在自回归解码过程中缓存键值对，避免重复计算，显著降低推理延迟。

### 5.4 工业部署挑战

- **延迟**：生成式推理可能比传统检索方法更慢，需要优化
- **吞吐量**：束搜索可能增加计算开销
- **冷启动**：新item的快速索引

---

## 6 开放问题与未来方向

1. **更高效的分词化**：如何设计更紧凑的语义 ID 方案，在保持语义性的同时提高效率
2. **推理能力**：如何将 LLM 的推理能力（如 CoT）整合到推荐过程中
3. **多模态推荐**：如何利用视觉、音频等多模态信息改进生成式推荐
4. **动态索引**：如何处理item的实时更新和删除
5. **可解释性**：如何利用生成式模型的中间推理过程提供推荐解释
6. **评估**：如何设计更好的评估指标来全面衡量生成式推荐的性能

---

![示意图（第4页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page004.png)
![示意图（第5页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page005.png)
![示意图（第6页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page006.png)
![示意图（第8页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page008.png)
![示意图（第9页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page009.png)
![示意图（第20页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page020.png)
![示意图（第21页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page021.png)
![示意图（第27页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page027.png)
![示意图（第28页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page028.png)
![示意图（第29页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page029.png)
![示意图（第30页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page030.png)
![示意图（第38页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page038.png)
![示意图（第39页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page039.png)
![示意图（第40页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page040.png)
![示意图（第42页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page042.png)
![示意图（第47页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page047.png)
![示意图（第54页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page054.png)
![示意图（第60页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page060.png)
![示意图（第61页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page061.png)
![示意图（第69页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page069.png)
![示意图（第70页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page070.png)
![示意图（第71页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page071.png)
![示意图（第72页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page072.png)
![示意图（第73页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page073.png)
![示意图（第83页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page083.png)
![示意图（第86页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page086.png)
![示意图（第89页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page089.png)
![示意图（第90页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page090.png)
![示意图（第91页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page091.png)
![示意图（第92页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page092.png)
![示意图（第101页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page101.png)
![示意图（第102页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page102.png)
![示意图（第104页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page104.png)
![示意图（第123页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page123.png)
![示意图（第124页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page124.png)
![示意图（第131页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page131.png)
![示意图（第133页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page133.png)
![示意图（第134页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page134.png)
![示意图（第136页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page136.png)
![示意图（第137页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page137.png)
![示意图（第138页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page138.png)
![示意图（第139页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page139.png)
![示意图（第140页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page140.png)
![示意图（第141页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page141.png)
![示意图（第143页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page143.png)
![示意图（第147页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page147.png)
![示意图（第150页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page150.png)
![示意图（第153页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page153.png)
![示意图（第154页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page154.png)
![示意图（第155页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page155.png)
![示意图（第158页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page158.png)
![示意图（第160页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page160.png)
![示意图（第161页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page161.png)
![示意图（第162页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page162.png)
![示意图（第163页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page163.png)
![示意图（第164页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page164.png)
![示意图（第165页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page165.png)
![示意图（第168页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page168.png)
![示意图（第170页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page170.png)
![示意图（第171页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page171.png)
![示意图（第172页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page172.png)
![示意图（第174页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page174.png)
![示意图（第175页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page175.png)
![示意图（第180页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page180.png)
![示意图（第181页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page181.png)
![示意图（第183页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page183.png)
![示意图（第184页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page184.png)
![示意图（第185页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page185.png)
![示意图（第187页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page187.png)
![示意图（第188页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page188.png)
![示意图（第189页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page189.png)
![示意图（第190页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page190.png)
![示意图（第191页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page191.png)
![示意图（第192页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page192.png)
![示意图（第193页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page193.png)
![示意图（第194页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page194.png)
![示意图（第195页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page195.png)
![示意图（第202页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page202.png)
![示意图（第203页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page203.png)
![示意图（第204页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page204.png)
![示意图（第205页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page205.png)
![示意图（第207页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page207.png)
![示意图（第212页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page212.png)
![示意图（第213页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page213.png)
![示意图（第214页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page214.png)
![示意图（第215页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page215.png)
![示意图（第216页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page216.png)
![示意图（第217页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page217.png)
![示意图（第218页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page218.png)
![示意图（第232页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page232.png)
![示意图（第234页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page234.png)
![示意图（第235页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page235.png)
![示意图（第237页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page237.png)
![示意图（第238页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page238.png)
![示意图（第240页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page240.png)
![示意图（第241页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page241.png)
![示意图（第242页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page242.png)
![示意图（第249页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page249.png)
![示意图（第253页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page253.png)
![示意图（第254页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page254.png)
![示意图（第256页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page256.png)
![示意图（第257页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page257.png)
![示意图（第258页）](.picture/2025-overall-large-genrec-tutorial-cikm25-page258.png)
## 参考文献

本教程涵盖了生成式推荐领域的主要工作，包括 TIGER、P5、RecForest、RecGPT、OneRec、TDM、JTM 等论文。详细参考文献列表见原始幻灯片的完整引用部分。
