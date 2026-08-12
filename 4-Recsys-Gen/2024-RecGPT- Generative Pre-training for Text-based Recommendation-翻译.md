# RecGPT: Generative Pre-training for Text-based Recommendation（中文翻译）


本文介绍了 RecGPT: Generative Pre-training for Text-based Recommendation。核心内容：


关键发现：

---


> Hoang Ngo, Dat Quoc Nguyen
> VinAI Research, Vietnam
> {v.hoangnv49, v.datnq9}@vinai.io

---

## 摘要

我们提出了第一个领域自适应且完全训练的大型语言模型 RecGPT-7B，及其指令遵循变体 RecGPT-7B-Instruct，用于基于文本的推荐。在评分预测和序列推荐任务上的实验结果表明，我们的模型 RecGPT-7B-Instruct 优于先前强基线。我们公开发布了我们的 RecGPT 模型及其预训练和微调数据集，以促进基于文本推荐的未来研究和下游应用。我们的 RecGPT 模型和数据集的公开 HuggingFace 链接可在以下网址获取：https://github.com/VinAIResearch/RecGPT。

---

## 1 引言

推荐系统有助于理解用户偏好并向用户提供合适的内容建议（Ansari et al., 2000; Sarwar et al., 2000; Pazzani and Billsus, 2007）。目前，推荐系统已在各个领域得到广泛应用，如电子商务（Schafer et al., 2001; Kang and McAuley, 2018）、新闻（Wang et al., 2018）和电影（Sun et al., 2019）。推荐系统的演进经历了从基础方法到更复杂现代方法的转变。传统方法挖掘交互矩阵以利用用户-item关系（Koren et al., 2009; Konstan et al., 1997; He et al., 2017），随后它们融入深度学习技术如 CNN 和 RNN 来提取item特征和捕获用户偏好（Wang et al., 2018; Hidasi et al., 2016）。然而，这种特定任务的设置存在数据稀疏性、缺乏捕获用户偏好随时间波动的灵活性，以及难以扩展到大量用户和大规模数据集等问题。后来的工作受注意力机制和 Transformer 架构的启发（Vaswani et al., 2017a），将用户历史建模为item序列并将信息编码到稠密向量中（Kang and McAuley, 2018; Sun et al., 2019; Zhou et al., 2020）。

随着大型语言模型（LLM）的进步，最近的工作利用了 LLM 理解用户偏好的能力（Geng et al., 2023; Rajput et al., 2023）。P5 模型（Geng et al., 2022）使用 ID 表示用户和item，试图在基于 T5（Raffel et al., 2020）的统一条件生成模型下聚合推荐任务。此外，Liu et al.（2023）评估了 ChatGPT 在不同推荐任务中的潜在用途。最近，Ji et al.（2024）使用 LoRA（Hu et al., 2022）微调了 LLaMA（Touvron et al., 2023）用于序列推荐。推荐任务经常表现出共享特征，如用户集、item集和交互，因此表明可以训练一个统一模型用于多个任务，而不是为每个任务使用不同的模型。采用单一模型方法，如 P5 所做的那样，不仅鼓励模型泛化，还促进跨任务的协作学习。然而，像 P5 那样用 ID 表示用户和item可能无法完全与 LLM 的文本理解能力对齐。用文本描述表示item、用基于文本的用户与item交互历史表示用户可能会更有效。

在本文中，（I）我们介绍了第一个名为 RecGPT 的领域自适应且完全训练的 LLM 系列，用于基于文本的推荐，其中包括基础预训练模型 RecGPT-7B 及其指令遵循变体 RecGPT-7B-Instruct。在此背景下，我们使用相对较大的推荐特定语料库（205 亿 token）预训练 RecGPT-7B，而 RecGPT-7B-Instruct 是通过在超过 10 万条指令提示及其响应的数据集上进一步微调 RecGPT-7B 得到的模型。（II）我们针对评分预测和序列推荐任务进行了实验，表明我们的 RecGPT-7B-Instruct 优于强基线，包括 P5。（III）我们公开发布了我们的模型以及预训练和微调数据集。我们希望这次发布能促进基于文本推荐领域的未来研究和应用。

---

## 2 我们的模型 RecGPT

本节描述 RecGPT 使用的数据，并概述其架构和优化设置。

### 2.1 预训练和微调数据

我们收集了来自各个领域的丰富而全面的数据集，包括：Amazon Product（McAuley et al., 2015）、Anime[^1]、BookCrossing[^2]、Food（Majumder et al., 2019）、Goodreads（Wan and McAuley, 2018）、HotelRec（Antognini and Faltings, 2020）、MovieLens（Harper and Konstan, 2015）、Netflix（Bennett and Lanning, 2007）、Steam[^3]、WikiRec（AlGhamdi et al., 2021）和 Yelp[^4]。具体而言，我们选择了包含item标题的数据集，这是item表示的关键因素。

[^1]: https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database
[^2]: https://www.kaggle.com/datasets/ruchi798/bookcrossing-dataset
[^3]: https://www.kaggle.com/datasets/tamber/steam-video-games
[^4]: https://www.yelp.com/dataset

每个item关联有元数据，包括属性如标题和品牌，以及用户交互如评分和评论。我们对收集的数据集进行清理预处理，丢弃：（i）没有标题的item，（ii）交互少于 5 的用户，以及（iii）所有背景和人口统计用户信息。最终，我们总共拥有 10,156,309 个用户、10,309,169 个item和 258,100,698 条交互。每个清理后数据集的详细统计信息见附录 A 中的表 4。

然后我们将每个清理后的数据集随机拆分为预训练/微调子集，比例为 99.5/0.5，在"用户"级别进行（即微调子集中的用户不会出现在预训练子集中，反之亦然）。[^5] 关于预训练，用户仅通过其交互历史表示。每个用户的交互历史（称为文本文档）被格式化为按时间顺序排列的基于文本的数据点列表 $i_1$ , $i_2$ , ..., $i_n$ ，其中 $i_k$ 由对应的第 k 个item的元数据和交互表示。例如，在表 1 的预训练样本中， $i_1$ 是 "Title: Rock-a-Stack; Brand: Fisher-Price; Review: My son loves to empty this stacker and play with and teeth on the rings; Rating: 5.0/5.0"。总共，我们创建了一个包含超过 1000 万文档、2050 亿 token 的预训练语料库。

[^5]: 有 4 个数据集我们没有应用 99.5/0.5 比例。更多详情请参考第 3.1 节。

在指令遵循的微调方面，鉴于我们数据集的特性，我们为推荐系统领域的两个流行任务创建了提示-响应对：评分预测和序列推荐。对于每个具有历史 $i_1$ , $i_2$ , ..., $i_n$ 的用户，最后一个item $i_n$ 被认为是序列推荐中基于历史上下文 $i_1$ , $i_2$ , ..., $i_n
$$
_{-}
$$
_1$ 要预测的下一个item。同时，第 (n-1) 个item $i_n
$$
_{-}
$$
_1$ 的评分被用作评分预测的标签，给定剩余的历史上下文 $i_1$ , $i_2$ , ..., $i_n
$$
_{-}
$$
_1$ 而不包含第 (n-1) 个item的评分。根据任务需求，用户历史中每个数据点 $i_k$ 中未使用的特征被丢弃，精简提示及其响应以增强任务相关性和效率。总共，我们创建了一个包含超过 10 万条指令提示和响应对的微调数据集。

预训练文档和提示-响应对的示例见表 1。预训练和微调中使用的数据格式详情见附录 B。

**表 1：预训练和微调数据示例**

| | |
|---|---|
| **预训练样本**（仅展示前 3 个item作为示例） | |
| | 给定一个用户与产品的交互历史如下： |
| | Title: Rock-a-Stack; Brand: Fisher-Price; Review: My son loves to empty this stacker and play with and teeth on the rings; Rating: 5.0/5.0 |
| | Title: Jumbo Puzzle; Brand: Melissa & Doug; Review: My niece text loves this puzzle at my parents house so I had to have it for my son. A classic!; Rating: 5.0/5.0 |
| | Title: So Big Crayons; Brand: Crayola; Review: Good quality as expected from Crayola and easy enough for him to grasp.; Rating: 5.0/5.0 |
| | ... |
| **微调样本** | |
| **评分预测** | |
| *提示* | 预测最后item的评分。给定一个用户与产品的交互历史如下： |
| | Title: Frankenweenie Figure; Brand: Disney; Review: My daughter loves Frankenweenie & I was super excited to find Sparky on here; Rating: 5.0/5.0 |
| | Title: Rubber Ghost Face; Brand: Fun World; Review: The rubber is so flimsy it literally flaps in the wind when you move your hand while holding it.; Rating: 2.0/5.0 |
| | Title: Makeup Signature Set; Brand: L Cosmetics; Review: The rubber is so flimsy it literally flaps in the wind when you move your hand while holding it.; Rating: 4.0/5.0 |
| | Title: Hive Building Sets; Brand: HEXBUG; Review: It is fun & my daughter loves it; Rating: |
| *响应* | 4.0/5.0 |
| **下一个item预测** | |
| *提示* | 预测下一个item。给定一个用户与产品的交互历史如下： |
| | Title: Frankenweenie Figure; Brand: Disney |
| | Title: Makeup Signature Set; Brand: L Cosmetics |
| | Title: Hive Building Sets; Brand: HEXBUG |
| *响应* | Title: Animal Hats; Brand: Zoopur Pets |

### 2.2 RecGPT-7B

RecGPT-7B 是一个基于 Transformer 解码器的模型（Brown et al., 2020; Vaswani et al., 2017b），采用了（Triton）Flash Attention（Dao et al., 2022）和 ALiBi（Press et al., 2022）以实现上下文长度外推。此外，我们使用了 "max_seq_len" 为 2048、"d_model" 为 4096、"n_heads" 为 32、"n_layers" 为 32，以及 GPT-NeoX 的分词器（词汇量 50K token），模型大小约为 70 亿参数。利用 Mosaicml 的 "llm-foundry" 库[^6]，我们用预训练好的 MPT-7B（Team, 2023）的权重初始化 RecGPT-7B 的参数权重，并在我们 2050 亿 token 的预训练语料库上进行持续预训练。

[^6]: https://github.com/mosaicml/llm-foundry：一个支持预训练和微调的强大库。

在优化方面，我们采用 LION 优化器（Chen et al., 2023）和使用 FSDP 的分片数据并行，设置全局 batch 大小为 128（即 128 * 2048 = 每 batch 260K token），在 8 块 A100 GPU（每块 40GB）上运行，并使用峰值学习率 2.5e-5。训练运行 2 个 epoch，使用 bfloat16 混合精度训练，耗时约 18 天。这相当于 20.5B * 2 / 260K = 157K 训练步（其中学习率在前 2K 训练步进行预热）。

预训练使用的总 GPU 时长为 18 * 8 * 24 = 3456 小时。按 GPU 功耗 400W 计算，预训练过程消耗 3456 * 400 = 1,382,400 Wh，相当于约 0.585 $tCO_2$ eq 的碳排放。

### 2.3 RecGPT-7B-Instruct

然后，我们使用第 2.1 节中由超过 10 万条指令提示及其响应对组成的数据集，对基础预训练模型 RecGPT-7B 进行关于评分预测和序列推荐的指令遵循微调。我们采用 LION 优化器，设置全局 batch 大小为 128，在 8 块 A100 GPU（每块 40GB）上运行，使用峰值学习率 1.0e-5，运行 2 个 epoch。得到的微调模型命名为 RecGPT-7B-Instruct。

微调 RecGPT-7B-Instruct 使用一个 8 块 A100 GPU（每块 40GB）的节点耗时 4 小时，总计 32 GPU 小时。这相当于约 0.0054 $tCO_2$ eq 的碳排放。

---

## 3 实验

我们进行实验以比较我们的 RecGPT-7B-Instruct 与强基线在评分预测和序列推荐任务上的表现。

### 3.1 实验设置

**评估数据集：** 我们在来自不同领域的 4 个基准数据集上进行实验，包括 "Amazon Beauty"、"Amazon Sports and Outdoors" 和 "Amazon Toys and Games"（McAuley et al., 2015）以及 Yelp。遵循之前的工作（Geng et al., 2022; Ji et al., 2024），对于这三个 Amazon 数据集，我们采用 5-core 版本的 2014 年数据[^7]；而对于 Yelp，我们考虑 2019 年 1 月 1 日至 2019 年 12 月 31 日期间的交。

[^7]: https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html

**数据泄露问题：** 我们进一步发现了一个以前未被指出的数据泄露问题。由于用于评估的四个实验基准数据集没有预定义的训练-验证-测试划分，之前的工作对每个评估任务应用了不同的拆分策略（Geng et al., 2022）。以用于训练 P5（Geng et al., 2022）的 Amazon Beauty 数据集为例（类似发现也适用于其他数据集）。该数据集包含用户、item以及它们之间的交互。一个交互示例可能是：用户 X 购买item Y，并提供评论和评分 4.0/5.0。原始数据集以交互记录形式呈现，没有预定义的训练-验证-测试拆分。P5 对不同任务采用不同的数据拆分策略。对于评分预测任务，P5 随机将数据按 80-10-10 的比例划分为训练集、验证集和测试集。对于序列推荐任务，P5 按用户聚合数据以构建用户历史，包括他们的交互。然后，P5 采用留一法，其中历史中的最后一项保留用于测试，倒数第二项用于验证，其余项用于训练。因此，评分预测任务的训练集中存在一些交互，这些交互也属于序列推荐任务的测试集，反之亦然（即序列推荐任务的训练集中的一些交互也属于评分预测任务的测试集）。合并两个任务的训练集进行多任务训练（如 P5 中所做的那样），而不过滤重复数据，会导致数据泄露。

为了保持一致的测试集，我们仍然重用他们的拆分，但移除训练集中出现在测试集中的交互。这确保了测试数据不会泄露到训练数据中。请注意，对于这 4 个实验基准，我们在测试集上报告最终得分，而训练集仅用于预训练 RecGPT-7B 以模拟真实场景（即我们不使用训练/验证集进行指令遵循的监督微调）。

**评估指标：** 对于评分预测，我们采用均方根误差（RMSE）和平均绝对误差（MAE）；而对于序列推荐，我们使用 top-k 命中率（HR@k）和 top-k 归一化折损累计增益（NDCG@k）。RMSE 和 MAE 的值越小，HR 和 NDCG 的值越大，表示性能越好。

**推理：** 我们使用 vLLM（Kwon et al., 2023）作为推理引擎。对于评分预测，对于给定的输入提示，我们采用采样解码策略，设置 "temperature" 为 1.0、"top_p" 为 0.9、"top_k" 为 50，然后从生成的响应输出中提取预测值。对于序列推荐，遵循之前的工作（Geng et al., 2022; Ji et al., 2024），对于给定的输入提示，我们使用束搜索解码策略，束大小为 10，生成 10 个响应输出，并使用它们的束搜索得分进行排序。此外，由于 LLM 的幻觉特性，生成的输出可能与真实标签略有不同。因此，我们实现了一个语义相似性匹配方法，分别基于 SentenceTransformers（Reimers and Gurevych, 2019）和 FAISS（Johnson et al., 2021）的文本嵌入模型和匹配模块。该方法利用稠密向量表示上的点积相似度，将每个生成的输出与item集中最相似的item关联起来。

### 3.2 主要结果

**评分预测：** 表 2 列出了我们的 RecGPT-7B-Instruct 和之前强基线在四个实验数据集上的评分预测结果。我们发现，总体而言，基于预训练 LLM 的方法（具体来说是 P5（Geng et al., 2022）、ChatGPT（GPT-3.5-turbo）和 RecGPT-7B-Instruct）优于传统的评分预测方法 MF（Koren et al., 2009）和 MLP（Cheng et al., 2016）。尽管 ChatGPT 并非专门设计用于此任务，但它在 "Beauty" 数据集上展现了有竞争力的性能得分，超过了 P5。我们发现 RecGPT-7B-Instruct 在所有数据集上的两个评估指标 RMSE 和 MAE 均取得了最佳结果，创造了新的最先进性能得分。

**表 2：评分预测结果。** "Sport" 和 "Toys" 分别缩写 "Sports and Outdoors" 和 "Toys and Games"。[*] 表示 Geng et al.（2022）报告的结果。[†] 表示 Liu et al.（2023）实验中不同模型中表现最好的 ChatGPT（GPT-3.5-turbo）的结果。

| 模型 | Beauty | | Sport | | Toys | | Yelp | |
|-------|--------|---|--------|---|-------|---|------|---|
| | RMSE | MAE | RMSE | MAE | RMSE | MAE | RMSE | MAE |
| MF (Koren et al., 2009) [*] | 1.1973 | 0.9461 | 1.0234 | 0.7935 | 1.0123 | 0.7984 | 1.2645 | 1.0426 |
| MLP (Cheng et al., 2016) [*] | 1.3078 | 0.9597 | 1.1277 | 0.7626 | 1.1215 | 0.8097 | 1.2951 | 1.0340 |
| P5 (Geng et al., 2022) [*] | 1.2843 | 0.8534 | 1.0357 | 0.6813 | 1.0544 | 0.7177 | 1.4685 | 1.0054 |
| ChatGPT (few-shot) [†] | 1.0751 | 0.6977 | - | - | - | - | - | - |
| MPT-7B with SFT | 0.5637 | 0.2616 | 0.5446 | 0.2488 | 0.5565 | 0.2668 | 0.5620 | 0.2804 |
| RecGPT-7B-Instruct | **0.5316** | **0.2436** | **0.5208** | **0.2340** | **0.5361** | **0.2535** | **0.5203** | **0.2489** |

**表 3：序列推荐结果。** [⋆] 表示 Rajput et al.（2023）报告的使用标准预处理的 P5 结果，他们对 Yelp 数据集未进行实验。

| 模型 | Beauty | | Sport | | Toys | | Yelp | |
|-------|---------|---------|---------|---------|---------|---------|---------|---------|
| | HR@5 | NDCG@5 | HR@10 | NDCG@10 | | | | |
| | Beauty | | | | | | | |
| P5 [⋆] | 0.0350 | 0.0250 | 0.0480 | 0.0298 | | | | |
| ChatGPT (few-shot) [†] | 0.0135 | 0.0135 | 0.0135 | 0.0135 | | | | |
| OpenP5 (Xu et al.) | 0.0317 | 0.0239 | 0.0437 | 0.0277 | | | | |
| MPT-7B with SFT | 0.0063 | 0.0041 | 0.0088 | 0.0050 | | | | |
| RecGPT-7B-Instruct | **0.0364** | **0.0236** | **0.0527** | **0.0288** | | | | |
| | Toys | | | | | | | |
| P5 [⋆] | 0.0180 | 0.0130 | 0.0235 | 0.0150 | | | | |
| GenRec (Ji et al.) | 0.0190 | 0.0136 | 0.0251 | 0.0157 | | | | |
| MPT-7B with SFT | 0.0088 | 0.0061 | 0.0133 | 0.0075 | | | | |
| RecGPT-7B-Instruct | **0.0430** | **0.0288** | **0.0606** | **0.0343** | | | | |
| | Sport | | | | | | | |
| P5 [⋆] | 0.0107 | 0.0076 | 0.0146 | 0.0088 | | | | |
| MPT-7B with SFT | 0.0021 | 0.0015 | 0.0033 | 0.0018 | | | | |
| RecGPT-7B-Instruct | **0.0173** | **0.0110** | **0.0255** | **0.0136** | | | | |
| | Yelp | | | | | | | |
| MPT-7B with SFT | 0.0390 | 0.0280 | 0.0453 | 0.0298 | | | | |
| RecGPT-7B-Instruct | **0.0479** | **0.0339** | **0.0603** | **0.0377** | | | | |

**序列推荐：** 表 3 展示了不同模型在序列推荐任务上，截止阈值 5 和 10 的 HR 和 NDCG 结果。毫不奇怪的是，在领域内数据方面受限的 ChatGPT 在 "Beauty" 数据集上获得的得分低于其他基线。这凸显了领域内训练数据对于模型理解item集在序列推荐中的关键作用。GenRec（Ji et al., 2024）使用 LoRA（Hu et al., 2022）在整个训练集上微调，与完全微调的模型 RecGPT-7B-Instruct 相比，在 "Toys and Games" 数据集上表现不具有竞争力。此外，我们的 RecGPT-7B-Instruct 在 "Beauty" 数据集上与 P5 和 OpenP5（Xu et al., 2023）取得了有竞争力的结果。而且，RecGPT-7B-Instruct 在 "Sports and Outdoors" 和 "Toys and Games" 数据集上显著优于 P5。

**消融分析：** 为了考察预训练对 RecGPT-7B-Instruct 性能得分提升的贡献，我们还在基础预训练模型 MPT-7B 上进行了指令遵循的有监督微调（SFT）。MPT-7B 的微调过程与第 2.3 节详述的我们的 RecGPT-7B-Instruct 相同。表 2 和表 3 也展示了 MPT-7B with SFT 的结果。我们发现 RecGPT-7B-Instruct 的性能明显优于 MPT-7B with SFT，这突显了持续预训练 RecGPT-7B 在推荐领域适应方面的重大贡献。在表 2 中，评分预测很可能依赖评论文本来预测分数，这可以被视为一个具有更细粒度标签的情感分类任务。因此，考虑到评分预测微调有数万个样本，这个任务（相比于序列推荐任务）不那么困难。此外，基础 LLM 模型 MPT-7B 是在一个 1 万亿 token 的语料库上预训练的，该语料库可能包含来自网络的许多评论。因此，对于评分预测任务，RecGPT-7B-Instruct 相较于基线 "MPT-7B with SFT" 的显著改进不如序列推荐任务那么大。

---

## 4 结论

我们提出了第一个针对基于文本推荐的领域自适应且完全训练的 LLM，包括基础预训练模型 RecGPT-7B 及其指令遵循变体 RecGPT-7B-Instruct。我们证明了 RecGPT 的实用性，展示了 RecGPT-7B-Instruct 在评分预测和序列推荐任务中均优于强基线。通过公开发布 RecGPT 模型以及预训练和有监督微调数据集，我们希望它们能促进基于文本推荐领域的未来研究和应用。

---

## 局限性

LLM 关于任务和item集的知识完全基于训练数据和基础模型的内存。模型可能无法识别训练数据中未覆盖的item。如果发生这种情况，模型可能会生成无关信息并遭受幻觉问题。这个限制也适用于所有基于 LLM 的方法。此外，在这项工作中，我们只评估了两个流行任务；我们将在未来的工作中对其他推荐任务进行实验。

---

## 致谢

我们感谢 Khoa D. Doan（khoa.dd@vinuni.edu.vn）的初步讨论。

---

## 参考文献

Kholoud AlGhamdi, Miaojing Shi, and Elena Simperl. 2021. Learning to Recommend Items to Wikidata Editors. In *The Semantic Web – ISWC 2021: 20th International Semantic Web Conference, ISWC 2021, Virtual Event, October 24–28, 2021, Proceedings*, page 163–181.

Asim Ansari, Skander Essegaier, and Rajeev Kohli. 2000. Internet Recommendation Systems. *Journal of Marketing Research*, 37(3):363–375.

Diego Antognini and Boi Faltings. 2020. HotelRec: a Novel Very Large-Scale Hotel Recommendation Dataset. In *Proceedings of the Twelfth Language Resources and Evaluation Conference*, pages 4917–4923.

James Bennett and Stan Lanning. 2007. The Netflix Prize. In *Proceedings of KDD Cup and Workshop 2007*, page 35.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. 2020. Language Models are Few-Shot Learners. In *Proceedings of NeurIPS*.

Xiangning Chen, Chen Liang, Da Huang, Esteban Real, Kaiyuan Wang, Hieu Pham, Xuanyi Dong, Thang Luong, Cho-Jui Hsieh, Yifeng Lu, and Quoc V Le. 2023. Symbolic discovery of optimization algorithms. In *Thirty-seventh Conference on Neural Information Processing Systems*.

Heng-Tze Cheng, Levent Koc, Jeremiah Harmsen, Tal Shaked, Tushar Chandra, Hrishi Aradhye, Glen Anderson, Greg Corrado, Wei Chai, Mustafa Ispir, Rohan Anil, Zakaria Haque, Lichan Hong, Vihan Jain, Xiaobing Liu, and Hemal Shah. 2016. Wide & Deep Learning for Recommender Systems. In *Proceedings of the 1st Workshop on Deep Learning for Recommender Systems*, page 7–10.

Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. 2022. FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. In *Proceedings of NeurIPS*.

Shijie Geng, Shuchang Liu, Zuohui Fu, Yingqiang Ge, and Yongfeng Zhang. 2022. Recommendation as Language Processing (RLP): A Unified Pretrain, Personalized Prompt & Predict Paradigm (P5). In *Proceedings of the 16th ACM Conference on Recommender Systems*, page 299–315.

Shijie Geng, Juntao Tan, Shuchang Liu, Zuohui Fu, and Yongfeng Zhang. 2023. VIP5: Towards multimodal foundation models for recommendation. In *Findings of the Association for Computational Linguistics: EMNLP 2023*, pages 9606–9620.

F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. *ACM Trans. Interact. Intell. Syst.*, 5(4):1–19.

Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu, and Tat-Seng Chua. 2017. Neural Collaborative Filtering. In *Proceedings of the 26th International Conference on World Wide Web*, page 173–182.

Balázs Hidasi, Alexandros Karatzoglou, Linas Baltrunas, and Domonkos Tikk. 2016. Session-based Recommendations with Recurrent Neural Networks. In *4th International Conference on Learning Representations, ICLR 2016, San Juan, Puerto Rico, May 2-4, 2016, Conference Track Proceedings*.

Edward J Hu, yelong shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022. LoRA: Low-rank adaptation of large language models. In *International Conference on Learning Representations*.

Jianchao Ji, Zelong Li, Shuyuan Xu, Wenyue Hua, Yingqiang Ge, Juntao Tan, and Yongfeng Zhang. 2024. GenRec: Large Language Model for Generative Recommendation. In *Proceedings of the 46th European Conference on Information Retrieval*, page to appear.

Jeff Johnson, Matthijs Douze, and Hervé Jégou. 2021. Billion-Scale Similarity Search with GPUs. *IEEE Transactions on Big Data*, pages 535–547.

Wang-Cheng Kang and Julian McAuley. 2018. Self-Attentive Sequential Recommendation. In *2018 IEEE International Conference on Data Mining (ICDM)*, pages 197–206.

Joseph A. Konstan, Bradley N. Miller, David Maltz, Jonathan L. Herlocker, Lee R. Gordon, and John Riedl. 1997. GroupLens: applying collaborative filtering to Usenet news. *Commun. ACM*, page 77–87.

Yehuda Koren, Robert Bell, and Chris Volinsky. 2009. Matrix Factorization Techniques for Recommender Systems. *Computer*, 42:30–37.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. 2023. Efficient Memory Management for Large Language Model Serving with PagedAttention. In *Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles*.

Junling Liu, Chao Liu, Peilin Zhou, Renjie Lv, Kang Zhou, and Yan Zhang. 2023. Is ChatGPT a Good Recommender? A Preliminary Study. In *Proceedings of the the 1st CIKM Workshop on Recommendation with Generative Models*.

Bodhisattwa Prasad Majumder, Shuyang Li, Jianmo Ni, and Julian McAuley. 2019. Generating Personalized Recipes from Historical User Preferences. In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing*, pages 5976–5982.

Julian McAuley, Christopher Targett, Qinfeng Shi, and Anton van den Hengel. 2015. Image-Based Recommendations on Styles and Substitutes. In *Proceedings of the 38th International ACM SIGIR Conference on Research and Development in Information Retrieval*, page 43–52.

Michael J. Pazzani and Daniel Billsus. 2007. Content-Based Recommendation Systems, pages 325–341.

Ofir Press, Noah Smith, and Mike Lewis. 2022. Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation. In *Proceedings of ICLR*.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. *J. Mach. Learn. Res.*

Shashank Rajput, Nikhil Mehta, Anima Singh, Raghunandan Hulikal Keshavan, Trung Vu, Lukasz Heldt, Lichan Hong, Yi Tay, Vinh Q. Tran, Jonah Samost, Maciej Kula, Ed H. Chi, and Maheswaran Sathiamoorthy. 2023. Recommender Systems with Generative Retrieval. In *Proceedings of the Thirty-seventh Conference on Neural Information Processing Systems*.

Nils Reimers and Iryna Gurevych. 2019. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing*, pages 3982–3992.

Badrul Sarwar, George Karypis, Joseph Konstan, and John Riedl. 2000. Analysis of recommendation algorithms for e-commerce. In *Proceedings of the 2nd ACM Conference on Electronic Commerce*, page 158–167.

J Ben Schafer, Joseph A Konstan, and John Riedl. 2001. E-Commerce Recommendation Applications. *Data Mining and Knowledge Discovery*, 5(1):115–153.

Fei Sun, Jun Liu, Jian Wu, Changhua Pei, Xiao Lin, Wenwu Ou, and Peng Jiang. 2019. BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer. In *Proceedings of the 28th ACM International Conference on Information and Knowledge Management*, page 1441–1450.

MosaicML NLP Team. 2023. Introducing MPT-7B: A New Standard for Open-Source, Commercially Usable LLMs.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. 2023. LLaMA: Open and Efficient Foundation Language Models. *arXiv preprint*, arXiv:2302.13971.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017a. Attention is All you Need. In *Advances in Neural Information Processing Systems*.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017b. Attention is All you Need. In *Proceedings of NIPS*, pages 5998–6008.

Mengting Wan and Julian J. McAuley. 2018. Item recommendation on monotonic behavior chains. In *Proceedings of the 12th ACM Conference on Recommender Systems*, pages 86–94.

Hongwei Wang, Fuzheng Zhang, Xing Xie, and Minyi Guo. 2018. DKN: Deep Knowledge-Aware Network for News Recommendation. In *Proceedings of the 2018 World Wide Web Conference*, page 1835–1844.

Shuyuan Xu, Wenyue Hua, and Yongfeng Zhang. 2023. OpenP5: Benchmarking Foundation Models for Recommendation. *arXiv preprint*, arXiv:2306.11134.

Kun Zhou, Hui Wang, Wayne Xin Zhao, Yutao Zhu, Sirui Wang, Fuzheng Zhang, Zhongyuan Wang, and Ji-Rong Wen. 2020. S3-Rec: Self-Supervised Learning for Sequential Recommendation with Mutual Information Maximization. In *Proceedings of the 29th ACM International Conference on Information & Knowledge Management*, page 1893–1902.

---

## 附录 A：数据集

我们清理后的数据集的统计信息见表 4。请注意，一些数据集有两个与不同发布时间相关联的版本（例如 Amazon 和 Yelp）。为了与之前的工作（Geng et al., 2022; Xu et al., 2023; Liu et al., 2023）保持一致的测试数据，我们保留较旧的版本（Amazon 为 2014 年，Yelp 为 2020 年）用于测试目的，并使用较新的版本（Amazon 为 2018 年，Yelp 为 2021 年）来丰富我们的预训练数据。我们过滤掉较新数据集中重叠的用户及其交互，以防止重复和数据泄露。

注意，如果用户的交互历史很长且包含许多item（即 token 数超过 max_seq_length 2048），我们会将历史预拆分为具有相似item数量的较小块，确保每个块中的 token 数小于 2048。然后每个块被视为一个单独的用户交互历史。

**表 4：用于预训练和微调的数据集统计信息。** 星号（*）表示专门用于预训练和最终评估的数据集。对于这四个标有（*）的数据集，我们采用来自先前工作（Geng et al., 2022; Ji et al., 2024）的训练/验证/测试拆分，但我们移除训练拆分中出现在验证/测试拆分中的用户和交互。这确保了验证/测试数据不会泄露到训练数据中。注意，对于这四个数据集，我们在测试集上报告最终评估分数，而训练集仅用于预训练 RecGPT-7B 以模拟真实场景。换句话说，我们不使用训练/验证集进行指令遵循的监督微调。请注意，一些数据集有两个与不同发布时间相关联的版本（例如 Amazon 和 Yelp）。为了与之前的工作保持一致，我们保留较旧的版本（Amazon 为 2014 年，Yelp 为 2020 年）用于测试目的，并使用较新的版本（Amazon 为 2018 年，Yelp 为 2021 年）来丰富我们的预训练数据。我们过滤掉较新数据集中重叠的用户及其交互，以防止重复和数据泄露。

| 数据集 | #用户 | #item | #交互 |
|--------|-------|-------|-------|
| Amazon All Beauty (2018) | 195 | 85 | 1,026 |
| Amazon AMAZON FASHION | 377 | 31 | 2,985 |
| Amazon Appliances | 20 | 47 | 119 |
| Amazon Arts Crafts and Sewing | 46,651 | 22,855 | 401,244 |
| Amazon Automotive | 181,146 | 79,315 | 1,576,030 |
| Amazon Books | 1,847,930 | 703,927 | 26,751,568 |
| Amazon CDs and Vinyl | 95,287 | 67,599 | 1,193,065 |
| Amazon Cell Phones and Accessories | 155,665 | 48,172 | 1,105,606 |
| Amazon Clothing Shoes and Jewelry | 1,167,022 | 376,853 | 10,628,886 |
| Amazon Digital Music | 34 | 183 | 248 |
| Amazon Electronics | 696,614 | 159,934 | 6,346,560 |
| Amazon Gift Cards | 456 | 148 | 2,961 |
| Amazon Grocery and Gourmet Food | 116,141 | 41,280 | 1,024,096 |
| Amazon Home and Kitchen | 733,886 | 189,038 | 6,406,439 |
| Amazon Industrial and Scientific | 9,391 | 5,327 | 66,091 |
| Amazon Kindle Store | 138,030 | 98,118 | 2,178,518 |
| Amazon Luxury Beauty | 2,779 | 1,577 | 25,386 |
| Amazon Magazine Subscriptions | 309 | 151 | 2,120 |
| Amazon Movies and TV | 282,072 | 60,109 | 3,199,604 |
| Amazon Musical Instruments | 25,402 | 10,611 | 210,646 |
| Amazon Office Products | 88,788 | 27,931 | 689,303 |
| Amazon Patio Lawn and Garden | 91,297 | 32,869 | 694,084 |
| Amazon Pet Supplies | 213,455 | 42,498 | 1,854,600 |
| Amazon Prime Pantry | 13,139 | 4,968 | 127,351 |
| Amazon Software | 1,470 | 802 | 10,571 |
| Amazon Sports and Outdoors (2018) | 302,870 | 104,559 | 2,541,948 |
| Amazon Tools and Home Improvement | 220,804 | 73,548 | 1,865,844 |
| Amazon Toys and Games (2018) | 194,141 | 78,695 | 1,687,243 |
| Amazon Video Games | 50,907 | 17,389 | 452,004 |
| Anime | 60,970 | 11,197 | 6,250,866 |
| BookCrossing | 12,787 | 270,170 | 299,303 |
| Food | 22,018 | 226,590 | 830,889 |
| Goodreads | 260,025 | 2,021,053 | 14,651,363 |
| HotelRec | 2,029,381 | 365,013 | 21,660,081 |
| MovieLens | 162,541 | 59,047 | 24,753,332 |
| Netflix | 472,987 | 17,770 | 99,472,215 |
| Steam | 3,757 | 5,155 | 113,796 |
| WikiRec | 60,648 | 4,871,794 | 13,693,465 |
| Yelp (2021) | 287,113 | 150,346 | 4,350,452 |
| Amazon Beauty (2014)(*) | 22,363 | 12,101 | 198,502 |
| Amazon Sports and Outdoors (2014)(*) | 35,598 | 18,357 | 296,337 |
| Amazon Toys and Games (2014)(*) | 19,412 | 11,924 | 167,597 |
| Yelp (2020)(*) | 30,431 | 20,033 | 316,354 |
| **总计** | **10,156,309** | **10,309,169** | **258,100,698** |

---

## 附录 B：训练和推理中使用的数据格式

我们展示了本工作中使用的提示模板。注意，在预训练和微调阶段，如果用户与许多item有较长的交互历史（即 token 数超过 max_seq_length 2048），我们会将历史预拆分为具有相似item数量的较小块，确保每个块中的 token 数小于 2048。然后每个块被视为一个单独的用户交互历史。

### B.1 预训练阶段使用的数据格式

**Amazon**

```
Given the interaction history of a user with products as follows:
Title: {title}; Brand: {brand}; Review: {review}; Rating: {rating}/5.0
...
Title: {title}; Brand: {brand}; Review: {review}; Rating: {rating}/5.0
```

**Amazon Books**

```
Given the interaction history of a user with books as follows:
Title: {title}; Brand: {brand}; Review: {review}; Rating: {rating}/5.0
...
Title: {title}; Brand: {brand}; Review: {review}; Rating: {rating}/5.0
```

**Anime**

```
Given the interaction history of a user with movies/shows as follows:
Title: {title}; Genres: {genres}; Rating: {rating}/10.0
...
Title: {title}; Genres: {genres}; Rating: {rating}/10.0
```

**BookCrossing**

```
Given the interaction history of a user with books as follows:
Title: {title}; Author: {author}; Rating: {rating}/10.0
...
Title: {title}; Author: {author}; Rating: {rating}/10.0
```

**Food**

```
Given the interaction history of a user with food recipes as follows:
Title: {title}; Review: {review_text}; Rating: {rating}/5.0
...
Title: {title}; Review: {review_text}; Rating: {rating}/5.0
```

**Goodreads**

```
Given the interaction history of a user with books as follows:
Title: {title}; Author: {author}; Genres: {genres}; Review: {review_text}; Rating: {rating}/5.0
...
Title: {title}; Author: {author}; Genres: {genres}; Review: {review_text}; Rating: {rating}/5.0
```

**HotelRec**

```
Given the interaction history of a user with hotels as follows:
Title: {title}; City: {city}; Review: {review_text}; Rating: {rating}/5.0
...
Title: {title}; City: {city}; Review: {review_text}; Rating: {rating}/5.0
```

**MovieLens**

```
Given the interaction history of a user with movies/shows as follows:
Title: {title}; Genres: {genres}; Rating: {rating}/5.0
...
Title: {title}; Genres: {genres}; Rating: {rating}/5.0
```

**Netflix**

```
Given the interaction history of a user with movies/shows as follows:
Title: {title}; Rating: {rating}/5.0
...
Title: {title}; Rating: {rating}/5.0
```

**Steam**

```
Given the interaction history of a user with video games as follows:
Title: {title}
...
...Title: {title}
```

**WikiRec**

```
Given the interaction history of a user with Wikipedia articles as follows:
Title: {title}; Description: {description}
...
Title: {title}; Description: {description}
```

**Yelp**

```
Given the interaction history of a user with businesses as follows:
Title: {title}; City: {city}; Review: {review_text}; Rating: {rating}/5.0
...
Title: {title}; City: {city}; Review: {review_text}; Rating: {rating}/5.0
```

### B.2 微调和推理中使用的数据格式

#### B.2.1 评分预测任务

**Amazon**

```
### Instruction:
Predict rating for the last item.
Given the interaction history of a user with products as follows:
Title: {title}; Brand: {brand}; Review: {review}; Rating: {rating}/5.0
...
Title: {title}; Brand: {brand}; Review: {review}; Rating:
### Response:
{rating}/5.0
```

**Amazon Books**

```
### Instruction:
Predict rating for the last item.
Given the interaction history of a user with books as follows:
Title: {title}; Author: {author}; Review: {review}; Rating: {rating}/5.0
...
Title: {title}; Author: {author}; Review: {review}; Rating:
### Response:
{rating}/5.0
```

**Anime**

```
### Instruction:
Predict rating for the last item.
Given the interaction history of a user with movies/shows as follows:
Title: {title}; Genres: {genres}; Rating: {rating}/10.0
...
Title: {title}; Genres: {genres}; Rating:
### Response:
{rating}/10.0
```

**BookCrossing**

```
### Instruction:
Predict rating for the last item.
Given the interaction history of a user with books as follows:
Title: {title}; Author: {author}; Rating: {rating}/10.0
...
Title: {title}; Author: {author}; Rating:
### Response:
{rating}/10.0
```

**Food**

```
### Instruction:
Predict rating for the last item.
Given the interaction history of a user with food recipes as follows:
Title: {title}; Review: {review_text}; Rating: {rating}/5.0
...
Title: {title}; Review: {review_text}; Rating:
### Response:
{rating}/5.0
```

**Goodreads**

```
### Instruction:
Predict rating for the last item.
Given the interaction history of a user with books as follows:
Title: {title}; Author: {author}; Genres: {genres}; Review: {review_text}; Rating: {rating}/5.0
...
Title: {title}; Author: {author}; Genres: {genres}; Review: {review_text}; Rating:
### Response:
{rating}/5.0
```

**HotelRec**

```
### Instruction:
Predict rating for the last item.
Given the interaction history of a user with hotels as follows:
Title: {title}; City: {city}; Review: {review_text}; Rating: {rating}/5.0
...
Title: {title}; City: {city}; Review: {review_text}; Rating:
### Response:
{rating}/5.0
```

**MovieLens**

```
### Instruction:
Predict rating for the last item.
Given the interaction history of a user with movies/shows as follows:
Title: {title}; Genres: {genres}; Rating: {rating}/5.0
..
Title: {title}; Genres: {genres}; Rating:
### Response:
{rating}/5.0
```

**Netflix**

```
### Instruction:
Predict rating for the last item.
Given the interaction history of a user with movies/shows as follows:
Title: {title}; Rating: {rating}/5.0
...
Title: {title}; Rating:
### Response:
{rating}/5.0
```

**Yelp**

```
### Instruction:
Predict rating for the last item.
Given the interaction history of a user with businesses as follows:
Title: {title}; City: {city}; Review: {review_text}; Rating: {rating}/5.0
...
Title: {title}; City: {city}; Review: {review_text}; Rating:
### Response:
{rating}/5.0
```

#### B.2.2 序列推荐任务

**Amazon**

```
### Instruction:
Predict the next item.
Given the interaction history of a user with products as follows:
Title: {title}; Brand: {brand}
...
Title: {title}; Brand: {brand}
### Response:
Title: {title}; Brand: {brand}
```

**Amazon Books**

```
### Instruction:
Predict the next item.
Given the interaction history of a user with books as follows:
Title: {title}; Author: {brand}
...
Title: {title}; Author: {brand}
### Response:
Title: {title}; Author: {brand}
```

**Anime**

```
### Instruction:
Predict the next item.
Given the interaction history of a user with movies/shows as follows:
Title: {title}; Genres: {genres}
...
Title: {title}; Genres: {genres}
### Response:
Title: {title}; Genres: {genres}
```

**BookCrossing**

```
### Instruction:
Predict the next item.
Given the interaction history of a user with books as follows:
Title: {title}; Author: {author}
...
Title: {title}; Author: {author}
### Response:
Title: {title}; Author: {author}
```

**Food**

```
### Instruction:
Predict the next item.
Given the interaction history of a user with food recipes as follows:
Title: {title}
...
Title: {title}
### Response:
Title: {title}
```

**Goodreads**

```
### Instruction:
Predict the next item.
Given the interaction history of a user with books as follows:
Title: {title}; Author: {author}; Genres: {genres}
...
Title: {title}; Author: {author}; Genres: {genres}
### Response:
Title: {title}; Author: {author}
```

**HotelRec**

```
### Instruction:
Predict the next item.
Given the interaction history of a user with hotels as follows:
Title: {title}; City: {city}
...
Title: {title}; City: {city}
### Response:
Title: {title}; City: {city}
```

**MovieLens**

```
### Instruction:
Predict the next item.
Given the interaction history of a user with movies/shows as follows:
Title: {title}; Genres: {genres}
..
Title: {title}; Genres: {genres}
### Response:
Title: {title}
```

**Netflix**

```
### Instruction:
Predict the next item.
Given the interaction history of a user with movies/shows as follows:
Title: {title}
...
Title: {title}
### Response:
Title: {title}
```

**Steam**

```
### Instruction:
Predict the next item.
Given the interaction history of a user with video games as follows:
Title: {title}
...
Title: {title}
### Response:
Title: {title}
```

**WikiRec**

```
### Instruction:
Predict the next item.
Given the interaction history of a user with Wikipedia articles as follows:
Title: {title}; Description: {description}
...
Title: {title}; Description: {description}
### Response:
Title: {title}; Description: {description}
```

**Yelp**

```
### Instruction:
Predict the next item.
Given the interaction history of a user with businesses as follows:
Title: {title}; City: {city}
...
Title: {title}; City: {city}
### Response:
Title: {title}; City: {city}
```

---

> 翻译自 VinAI Research 论文。
