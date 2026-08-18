# 一双新 GloVe（A New Pair of GloVes）

> Riley Carlson, John Bauer, Christopher D. Manning | 斯坦福大学斯坦福 NLP 组 | arXiv 2025

本文发布了训练于 2024 年新语料（Wikipedia、Gigaword 与 Dolma 子集）的两套全新英文 GloVe 词嵌入，并完整记录数据版本与预处理细节（2014 版从未如此文档化）。核心发现是——**2024 向量纳入了 covid、tiktok、chatgpt 等十年间涌现的文化与技术新词，在类比/相似度任务上与 2014 版持平，并在时间敏感的 NER 数据集上显著提升**。

核心内容：

- 语料构建：2024 Wikipedia 转储 + 5 版 Gigaword（双份）共 11.9B token，以及 Dolma v1.6 的 220B token 子集（Common Crawl 5%、C4 40%、Reddit 100%、Project Gutenberg 100%）
- 词汇选择引入最小频率阈值（MFT，Minimum Frequency Threshold）：MFT=20 时训练向量与 WLS 向量的平均余弦相似度最高，Wiki/Giga 词汇量达 1,291,146
- 训练细节：对称上下文窗口 10、AdaGrad 优化、50/100/200/300 维（Dolma 仅 300 维），学习率 0.05、alpha 0.75、XMax 100
- 三重评估：新词表对比（39 词抽样）、直接评估（Google/MSR 类比 + WordSim353/SimLex999/MEN 相似度）、下游 NER（CoNLL-03/CoNLL-PP/Worldwide/WNUT17）
- 错误分析：混淆矩阵差值、示例句标注对比，揭示 2024 模型对 COVID-19、Bolsonaro 等新实体的识别优势

关键发现：

- 与 2014 版相比新增超 **70 万**（Wiki/Giga）与 50 万（Dolma）个词；新词涵盖新冠（covid）、科技（blockchain、chatgpt）、网络热词（rizz、brainrot）与流行产品（airpods）
- NER 提升显著：Worldwide 数据集 50d 每实体 F1 达 **84.64 vs 82.1**；WNUT17 上 200d 达 37.46 vs 35.68；CoNLL-PP 上 50d 达 83.64 vs 81.58
- 类比任务与 2014 持平：Google 300d 上 0.718 vs 0.717；相似度任务中 2024 擅长近义词/上下位词（cemetery–graveyard），2014 擅长宽松主题关联（blue–red）
- 2024 模型的混淆矩阵差值显示：COVID-19 从"未标注（O）"变为正确标为 MISC，Bolsonaro 从 LOC 变为 PER
- 高维 2024 向量偶尔高估反义词相似度（如 agree–argue），是相似度任务下降的主因

---

## 摘要

本报告记录、描述并评估了新的 2024 英语 GloVe（Global Vectors for Word Representation，全局向量词表示）模型。虽然 2014 年构建的原始 GloVe 模型已被广泛使用并证明有用，但语言和世界在持续演变，我们认为当前的使用可以从更新模型中受益。此外，2014 模型没有仔细记录所使用的确切数据版本和预处理，我们通过记录这些新模型来纠正这一点。我们使用 Wikipedia、Gigaword 和 Dolma 的一个子集训练了两套词嵌入。通过词汇表比较、直接测试和 NER 任务的评估表明，2024 向量纳入了新的文化相关和语言相关词汇，在类比和相似度等结构性任务上表现相当，并在最近的时间敏感 NER 数据集（如非西方新闻专线数据）上展示了改进的性能。

## 1 引言

神经语义词向量空间模型用实值向量表示每个词，称为词嵌入。一组被广泛使用的词嵌入是 Pennington et al. (2014) [11] 引入的 GloVe 词嵌入。该算法利用一个专注于局部上下文构建的全局共现矩阵。GloVe 模型在这个全局矩阵上训练，得到的权重就是词嵌入，其中相似的词在向量空间中聚集得更近。由此产生的词嵌入在稠密向量空间中编码语义关系，使它们对各种 NLP 任务有用。尽管基于 transformer 的模型崛起，像 GloVe 这样的预训练静态嵌入在低资源设置、计算高效模型和注重可解释性的应用中仍然有价值。

我们更新词嵌入以使用更新数据的原因是，自 2014 年原始训练以来，新词已经出现，现有词的语义含义已经转变。例如，'covid' 在 2014 嵌入中没有表示。具有这种更新词表的嵌入在下游任务中具有许多好处，例如减少词汇表外（OOV，Out-of-Vocabulary）问题。为了反映当前英语词汇的使用，需要用近期语言训练的新嵌入。

在这项工作中，我们基于 GloVe-V（Vallebueno et al., 2024）[17] 工作的洞见，在训练更新词嵌入的词汇表选择过程中纳入最小频率阈值（MFT，Minimum Frequency Threshold）。使用 MFT 使我们能够在过滤掉过度稀有和噪声词的同时保留频率较低但上下文重要的词之间取得平衡。GloVe-V 框架通过引入统计不确定性估计扩展了这种方法，该估计解释了因数据稀疏性导致的嵌入位置变异性。这使得训练的词向量不仅稳健且富有表现力，而且更适合于稀有词通常具有关键重要性的下游任务，确保对现代语言使用的适应性（Vallebueno et al., 2024）[17]。

通过本文，我们详细说明了确切的训练过程，并证明 2024 词嵌入拥有反映当今语言使用和文化趋势的更新词表。它们在词类比和相似度任务上与 2014 嵌入表现相当，表明相似的结构和核心语义表现力。此外，2024 嵌入在时间敏感的命名实体识别（NER，Named Entity Recognition）数据集上展示了改进的性能。

在本报告中，我们首先描述用于创建两套不同词嵌入的训练数据，包括所选语料库和预处理步骤。然后，我们概述训练过程以方便复现。最后，我们展示嵌入的评估指标，包括新词汇覆盖、直接评估和下游任务性能。

## 2 数据：2014 vs. 2024

**表 1：2014 和 2024 年用于训练词嵌入的数据源比较，包括语料库大小（以十亿 token 计）。**

| 2014 数据 | 2024 数据 |
|---|---|
| Wikipedia 和 Gigaword（60 亿 token） | Wikipedia（2024）和 Gigaword（第 5 版）（119 亿） |
| Common Crawl（420 亿和 8400 亿） | Dolma 子集（2200 亿） |
| Twitter（270 亿） | |

对于 2024 嵌入，我们使用 3 个语料库训练 2 套嵌入：Wikipedia、Gigaword 和 Dolma。为了与 2014 年 Wikipedia 和 Gigaword 嵌入充分比较，我们采用相同的语料库并使用更新的 Wikipedia 转储。Wikipedia 语料库（Wikipedia 文章的数据集）是词定义的宝贵来源，其环境比字典更自然。与 Wikipedia 一起，还使用了 Gigaword（Parker et al., 2011）[10]。具体到 2024 向量，我们使用了第 5 版 Gigaword。该语料库包含 1994–2010 年间来自 4 到 7 家（视年份而定）不同国际新闻媒体的英语新闻专线。自 2014 年以来，Wikipedia 转储的 token 数量大约翻了一番。为重新平衡这种增长，我们在训练语料库中放入了两份 Gigaword。

除了上述语料库，我们还利用了 Dolma v1.6（Soldaini et al., 2024）[15]。该语料库于 2024 年 1 月发布，包含来自书籍、编程脚本、参考资料、学术文章和在线内容的 3 万亿 token。我们取 Dolma 超过 1TB 的一个子集。表 2 显示了所取的 Dolma 子集和使用的 token 数量。具体来说，我们有来自 Common Crawl 和 C4（Raffel et al., 2020）[13] 的网页、来自 Project Gutenberg 的书籍以及来自 Reddit 的社交媒体。C4 包含截至 2019 年的数据，而其他数据截至 2023 年。

**表 2：Dolma 训练子集**

| 数据集 | 取用百分比 | Token（十亿） |
|---|---|---|
| Common Crawl | 5% | 87.2 |
| C4 | 40% | 60 |
| Reddit | 100% | 68.9 |
| Project Gutenberg | 100% | 2.3 |

## 3 方法

首先，我们将更详细地描述用于训练不同嵌入的三个语料库，以及采取的任何预处理步骤。接下来，我们将描述所有嵌入使用的训练过程。然后，将介绍不同的评估实验。

### 3.1 语料库 #1：Wikipedia 和 Gigaword

2024 向量训练语料库的 Wikipedia 部分从 https://dumps.wikimedia.org/enwiki/20240720/enwiki-20240720-pages-meta-current.xml.bz2 的 Wikipedia 转储下载。然后使用 WikiExtractor（Attardi, 2015）[1] 提取数据。Wikipedia 数据通过移除 `<doc>` 和 `<unk>` 等标记的 token 来清理。

数据使用 Stanford 的 CoreNLP 分词器（版本 4.4.1）[^1] 使用小写字母预处理。然后合并 Wikipedia 和 Gigaword 语料库，Gigaword 包含两次。这个语料库总共约 60GB，其中 Gigaword 约占 74%，Wikipedia 占其余部分。

Wiki/Giga 向量的词汇量大小遵循 Vallebueno et al. (2024) [17] 概述的方法选择。具体来说，词汇量的大小通过为要纳入语料库的词设置 MFT 来确定。通过使用不同 MFT 训练的向量进行实验，观察到 MFT 为 20 时，训练向量与其加权最小二乘（WLS，Weighted Least Squares）向量之间的平均余弦相似度最高。与 WLS 向量的高余弦相似度表明训练嵌入与从共现矩阵推导的统计最优解紧密对齐，反映了稳健且准确的词表示（Vallebueno et al., 2024）[17]。对于该语料库，使用 MFT 为 20 得到 1,291,146 个词的词汇量。

[^1]: https://nlp.stanford.edu/software/tokenizer.html

### 3.2 语料库 #2：Dolma

与其他语料库一样，我们以相同方式使用 Stanford 的 CoreNLP 分词器预处理。预处理后，我们移除了 `<unk>` token。使用最大词汇量 1.2 百万。词汇构建过程在 Dolma 的不同子集上独立进行，并在最后合并为 1.2 百万词汇量。此外，共现矩阵通过在合并后的词汇表上合并各共现矩阵来创建。

### 3.3 训练

训练过程在所有嵌入和语料库中保持一致。对于每个嵌入，首先构建词汇表和共现矩阵。Wikipedia 和 Gigaword 语料库的词汇量设置为超过 1.2 百万，Dolma 语料库为 1.2 百万。使用大小为 10 的对称上下文窗口定义共现。共现矩阵构建完成后，用固定种子打乱：Wiki/Giga 矩阵用 123，Dolma 矩阵用 2024。

为 Wikipedia 和 Gigaword 语料库训练了 50、100、200 和 300 维的嵌入，为 Dolma 语料库训练了 300 维嵌入。嵌入使用 GloVe 的原始优化器 AdaGrad 优化。训练过程使用 GloVe 仓库中提供的 demo.sh 脚本执行，更多文档见 Training README.md 文件。训练中使用的超参数总结在表 3 中。

**表 3：在 Wiki/Giga 上训练的 50d、100d、200d 和 300d 词嵌入以及在 Dolma 上训练的 300d 词嵌入的训练超参数总结**

| 超参数 | 值 |
|---|---|
| 学习率 | 0.05\* |
| Alpha | 0.75 |
| XMax | 100 |
| 种子 | 2024\* |
| 轮次：（50d, 100d） | 50 |
| 轮次：（200d, 300d） | 100 |

\* 0.075 学习率和 123 种子用于 50d Wiki/Giga 向量

### 3.4 评估：更新词表

为评估嵌入质量，我们检查存在于 2024 嵌入但不存在于 2014 嵌入的词，以确定新常用词是否反映在更新嵌入中。比较了 Wikipedia 和 Gigaword 语料库的 2014 和 2024 嵌入词汇表，以及 2024 Dolma 嵌入词汇表与 2014 年 Common Crawl 上训练的 840B 向量词汇表。通过将词汇表表示为集合，我们通过从 2024 集合中减去 2014 集合来计算差集。从这个结果集中，我们为每个训练语料库选择 39 个代表性示例来说明我们的发现。

### 3.5 评估：直接评估

我们对嵌入执行了直接评估任务，与 2014 嵌入进行比较。评估聚焦于两个主要任务：词类比和词相似度。

对于词类比任务，目标是预测类比格式"word 1 : word 2 :: word 3 : ?"中的第四个词，并与金标准标注词比较以计算准确率。我们使用两个基准数据集：

- Google 类比数据集（Mikolov et al., 2013a）[8]，包含 8,869 个语义和 10,675 个句法词对。
- MSR 类比数据集（Mikolov et al., 2013b）[9]，包含 8,000 个句法词对。

对于词相似度任务，嵌入通过给词对分配相似度分数并将这些分数与人工标注基准比较来评估。我们使用三个基准数据集：

- WordSim353（Finkelstein et al., 2001）[4]，包含 353 个词对，分类为高度相似、不太相似但相关或不相关。
- SimLex999（Hill et al., 2015）[5]，包含 999 个标注了语义相似度分数的词对。
- MEN（Bruni et al., 2014）[2]，包含 3,000 个标注了人工判断相关度分数的词对。

为执行这些评估，我们使用了 Jastrzebski et al. (2017) [6] 开发的嵌入评估包，它支持词类比和词相似度数据集的分析。

### 3.6 评估：NER

为进一步评估新嵌入的性能，我们在下游任务命名实体识别（NER）上评估。在此任务中，句子中的 token 被标注为预定义实体类别，如人物、地点、组织和其他专有名词，实现从文本中的结构化信息提取。对于此评估，我们使用 Stanford 的 Stanza NER 模型（Qi et al., 2020）[12] [^2]，修改以将默认词嵌入替换为我们训练的嵌入。我们为每个词嵌入和数据集训练一个模型。

对于 NER 评估，我们使用三个数据集训练和测试模型，分别用于 2014 和 2024 嵌入：

- **CoNLL-03**：发布于 2003 年，该数据集包括人物、地点、组织和杂项类别的实体（其他类别未涵盖的实体）（Tjong Kim Sang and De Meulder, 2003）[16]。
- **CoNLL-PP**：Liu and Ritter (2023) [7] 引入的 CoNLL-03 改进版，具有更新和现代化的数据。我们在 CoNLL-03 上训练模型并在 CoNLL-PP 测试集上评估。
- **English Worldwide Newswire**：该数据集由（Shan et al., 2023）[14] 引入，包含 2023 年发布的 1,000 多篇英语新闻专线文章。这些文章来自 47 个国家，排除了美国新闻媒体，以确保非西方焦点和近期语言使用。该数据集包括对 COVID-19 大流行等重大事件的引用，提供了独特的机会来评估在 2020 年前数据（2014 嵌入）上训练的嵌入如何泛化到近期上下文。
- **新兴和稀有实体识别（WNUT 17）**：Derczynski et al. (2017) [3] 的这个 6 类数据集包括来自 Youtube、Twitter 和 Reddit 等平台用户生成文本的实体。这些实体更稀有且通常未见，即使对人类来说在噪声文本中标注也很具挑战性。该数据旨在改进动态现实世界文本场景中罕见实体的检测和分类。

这些数据集的训练子集用于训练模型，使用默认参数，不进行嵌入微调或字符语言建模，开发集和测试子集用于评估。我们报告每个实体和每个 token 的 F1 分数。

[^2]: https://stanfordnlp.github.io/stanza/ner.html

## 4 结果

我们展示三个评估指标的结果：更新词表、直接评估和下游评估。

### 4.1 更新词表

在这里，我们定性地展示存在于新 2024 嵌入中但不存在于 2014 嵌入中的词。新的 Wikipedia 和 Gigaword 向量提供了大得多的词汇量。在 2024 Wikipedia 和双份 Gigaword 与 2014 Wikipedia 和 Gigaword 嵌入之间，有超过 70 万个新词（不包括数字和含非拉丁字母的词）。在 2024 Dolma 嵌入与 2014 Common Crawl 840B 嵌入之间，有超过 50 万个新词（不包括数字和含非拉丁字母的词）。在表 5 和 6 中，我们报告了作者选择的 39 个文化、政治和技术性质的新词。

**表 5：与 2014 Wiki/Giga 向量相比，包含在 Wiki/Giga 嵌入中的 39 个新词样本。**

afrobeats、antiracism、asmr、binance、bipoc、blockchain、brexit、chatbot、clickbait、covid、fyp、cryptocurrency、deepfake、docuseries、doja、doordash、draftkings、rizz、nonbinary、fintech、fortnite、skibidi\*、idk、jungkook、latinx、lgbtqia、microaggression、lstm、metoo、microplastic、pickleball、retweet、zelenskyy、teladoc、web3、tiktok、transwoman、girlboss、viserys

\* 不在 Dolma 向量中

**表 6：与 2014 840B Common Crawl 向量相比，包含在 Dolma 嵌入中的 39 个新词样本。**

dinkies、profeminist†、theranos†、chatgpt†、adagrad、databricks†、huggingface、tarboosh†、gamestonk、badbunny、yeet†、patreon†、brainrot、xgboost†、bytedance†、fakenews、periodt、duolingo†、mansplains、pytorch†、absofreakinglutely、squidgame、trumpism†、clapback、highkey、bffr、situationship、cybertruck†、boujee†、alphafold†、glowup、openai†、scikit、bingewatch、tensorflow†、kubernetes†、aapi†、airpods†、deeplearning

† 也出现在 Wiki/Giga 向量中

### 4.2 词嵌入评估

我们在表 4 中报告 2014 和 2024 嵌入在词类比和相似度数据集上的结果。类比任务使用 Google 和 MSR 数据集评估，以准确率为度量。词相似度任务在 WordSim353、SimLex999 和 MEN 上使用 Spearman 秩相关系数（$\rho$）评估。

**表 4：2014 和 2024 嵌入在词类比数据集 Google 和 MSR 上的准确率，以及在词相似度数据集 WordSim353、SimLex999 和 MEN 上的 Spearman 秩相关。**

| 嵌入 | Google | MSR | WordSim353 | SimLex999 | MEN |
|---|---|---|---|---|---|
| 2014 50d Wiki/Giga | 0.462 | 0.355 | 0.448 | 0.265 | 0.652 |
| 2024 50d Wiki/Giga | 0.455 | 0.329 | 0.431 | 0.256 | 0.637 |
| 2014 100d Wiki/Giga | 0.631 | 0.550 | 0.477 | 0.298 | 0.681 |
| 2024 100d Wiki/Giga | 0.601 | 0.486 | 0.455 | 0.291 | 0.672 |
| 2014 200d Wiki/Giga | 0.698 | 0.595 | 0.515 | 0.340 | 0.710 |
| 2024 200d Wiki/Giga | 0.696 | 0.574 | 0.480 | 0.326 | 0.688 |
| 2014 300d Wiki/Giga | 0.717 | 0.614 | 0.544 | 0.371 | 0.737 |
| 2024 300d Wiki/Giga | 0.718 | 0.594 | 0.486 | 0.338 | 0.690 |
| 2024 300d Dolma | 0.708 | 0.623 | 0.470 | 0.270 | 0.651 |

对于类比任务，2024 嵌入在 Google 数据集上与 2014 嵌入表现大致相似，但在 MSR 数据集上性能略低。此外，在两个数据集上，2014 和 2024 嵌入的准确率都随着维度大小的增加而持续提高。

对于词相似度任务，2024 嵌入在大多数数据集和维度上与 2014 嵌入具有竞争力。两个 2024 的 300 维嵌入与 2014 嵌入相比都显示秩相关下降，特别是在 SimLex999 上。这种下降在讨论部分中解决。

### 4.3 NER

我们使用四个数据集评估了 2014 和 2024 嵌入在 NER 任务上的性能：CoNLL-03、CoNLL-PP、Worldwide 和 WNUT 17。对于这些数据集，我们报告每个实体和每个 token 基础上的测试 F1 分数。结果总结在表 7、8 和 9 中。

**表 7：2014 和 2024 嵌入在 CoNLL-03（Tjong Kim Sang and De Meulder, 2003）和 CoNLL-PP（Liu and Ritter, 2023）上每个实体和每个 token 的平均测试 F1 分数。我们使用在 CoNLL-03 上训练的、无嵌入微调的 Stanford Stanza NER 模型。**

| 嵌入 | 每实体（2003） | 每实体（PP） | 每 token（2003） | 每 token（PP） |
|---|---|---|---|---|
| 2014 50d Wiki/Giga | 89.52 | 81.58 | 89.30 | 80.77 |
| 2024 50d Wiki/Giga | 89.74 | 83.64 | 89.43 | 82.23 |
| 2014 100d Wiki/Giga | 90.62 | 84.40 | 90.41 | 82.73 |
| 2024 100d Wiki/Giga | 90.34 | 83.53 | 90.16 | 82.04 |
| 2014 200d Wiki/Giga | 90.88 | 84.21 | 90.91 | 82.89 |
| 2024 200d Wiki/Giga | 90.69 | 84.36 | 90.46 | 82.75 |
| 2014 300d Wiki/Giga | 90.60 | 84.25 | 90.43 | 82.70 |
| 2024 300d Wiki/Giga | 90.72 | 84.06 | 90.50 | 82.74 |
| 2024 300d Dolma | 90.05 | 85.14 | 90.12 | 83.69 |

**表 8：2014 和 2024 嵌入在 Worldwide 数据集（Shan et al., 2023）上每个实体和每个 token 的平均测试 F1 分数。我们使用在 Worldwide 数据集上训练的、无嵌入微调的 Stanford Stanza NER 模型。**

| 嵌入 | 每实体 | 每 token |
|---|---|---|
| 2014 50d Wiki/Giga | 82.1 | 81.04 |
| 2024 50d Wiki/Giga | 84.64 | 83.88 |
| 2014 100d Wiki/Giga | 85.29 | 84.58 |
| 2024 100d Wiki/Giga | 85.55 | 84.25 |
| 2014 200d Wiki/Giga | 84.41 | 83.53 |
| 2024 200d Wiki/Giga | 85.68 | 84.92 |
| 2014 300d Wiki/Giga | 84.53 | 83.89 |
| 2024 300d Wiki/Giga | 84.89 | 84.11 |
| 2024 300d Dolma | 86.23 | 85.27 |

**表 9：2014 和 2024 嵌入在 WNUT17 数据集（Derczynski et al., 2017）上每个实体和每个 token 的平均测试 F1 分数。我们使用在 WNUT17 数据集上训练的、无嵌入微调的 Stanford Stanza NER 模型。**

| 嵌入 | 每实体 | 每 token |
|---|---|---|
| 2014 50d Wiki/Giga | 32.95 | 31.05 |
| 2024 50d Wiki/Giga | 35.65 | 33.10 |
| 2014 100d Wiki/Giga | 36.48 | 33.39 |
| 2024 100d Wiki/Giga | 36.33 | 34.23 |
| 2014 200d Wiki/Giga | 35.68 | 33.31 |
| 2024 200d Wiki/Giga | 37.46 | 35.63 |
| 2014 300d Wiki/Giga | 36.64 | 33.73 |
| 2024 300d Wiki/Giga | 37.17 | 33.33 |
| 2024 300d Dolma | 39.44 | 34.22 |

四个 NER 数据集的结果表明，2024 嵌入总体上优于其 2014 对应物，在时间依赖的数据集上尤其显著。

对于表 7 中的 CoNLL 数据集，2024 嵌入在 CoNLL-03 上表现相当，但在现代化的 CoNLL-PP 版本上显示出明显优势，2024 50d Wiki/Giga 嵌入在每实体得分上实现最高的相对提升（83.64 vs. 81.58）。在表 8 的 Worldwide 数据集上，2024 嵌入在每实体和每 token F1 分数上都展现一致的改进，2024 50d Wiki/Giga 嵌入达到 84.64 的每实体 F1 分数，显著优于其 2014 对应物的 82.1 分。表 9 中具有挑战性的 WNUT17 数据集显示，与 CoNLL 和 Worldwide 相比整体性能显著下降（F1 分数在 30-40 之间），但 2024 嵌入始终优于其 2014 对应物，2024 200d Wiki/Giga 嵌入实现最高的每实体和每 token F1 分数，分别为 37.46 和 35.63。

在所有三个数据集上，2024 嵌入在较低维度上展现出最显著的增益，特别是在 50d 上，2024 和 2014 嵌入之间的差异最大。虽然较高维度（200d 和 300d）实现了最好的绝对 F1 分数，但 2024 和 2014 嵌入在这些维度上的相对增益不太明显。这些趋势表明，2024 嵌入在与训练数据时间对齐的数据集上表现与 2014 嵌入一样好或更好，同时在更好反映当代语言使用和文化趋势的现代、语言多样化的数据集上显示出显著改进。

## 5 讨论

2024 词嵌入引入了反映过去十年文化、技术和语言变迁的多样化新词汇，包括与全球重大事件相关的词（'covid' 和 'brexit'）、现代俚语（'brainrot' 和 'periodt'）、新兴技术（'chatgpt' 和 'blockchain'）和流行产品（'airpods'）。虽然许多俚语词不在 Wikipedia 和 Gigaword 训练数据中，因为现代俚语需要时间从社交媒体过渡到 Wikipedia，但多样化的 Dolma 数据集通过捕获非正式和会话语言进行了补偿。缩写（'idk'）、语言混合词（'absofreakinglutely'）和派生词（'retweet'）的纳入揭示了由数字通信和社交媒体驱动的词形成模式的演变。通过捕获这些变迁，2024 嵌入与当前白话一致，并为下游任务提供实际好处，例如减少现代数据集中的词汇表外问题，同时进一步探索这些新词为语言和社会学研究提供了机会。

在表 4 报告的类比任务中，2014 嵌入表现稍好，Google 数据集上的差异很小（准确率在 0.01-0.03 以内），但 MSR 数据集上的差距更大，特别是对于较低维度，100d 和 50d 嵌入的差异分别达到 0.07 和 0.03。约有 1,100 个实例是 2014 50d 预测正确但 2024 50d 预测错误的。这些错误中约一半是基于地理的（如 cairo, egypt : bern, ?）。另一半主要是 2024 嵌入预测了金标准答案的同义词的实例。例如在 simple, simpler: cold, ? 中，2024 嵌入预测了 'cooler' 而不是 'colder'。虽然 2024 50d 嵌入犯了这些错误，但 2014 嵌入也犯了这些常见错误。约有 900 个实例是 2024 50d 预测正确但 2014 50d 预测错误的。这些错误约一半基于地理，另一半使用同义词（在 write, writes : think, ? 中使用 'knows' 而不是 'thinks'）。在基于地理的错误中，2024 嵌入确实"知道"正确答案，但最近邻不是正确答案。例如，2024 嵌入确实知道 Bern 在瑞士，尽管有时最近邻不是瑞士。这可以从 2024 嵌入没有答对 kabul, afghanistan : bern, ?，但答对了 cairo, egypt : bern, ? 看出。同样的现象也出现在 2014 嵌入中。MSR 数据集的错误在句法词类比中看到更多使用同义词的实例。

因此，2024 和 2014 嵌入在类比任务上表现大致相同。这种同等性能是预期的，因为这些任务主要依赖句法结构和常用词，它们在过去的十年中保持相对稳定。

对于词相似度任务，嵌入之间的 Spearman 秩相关系数存在差异。为将这些差异放在适当的角度，我们查看了嵌入预测与人类评估（缩放为 −1 到 1，而非 0–10）之间差异为 0.3 的词对。对于 MEN 数据集，有 70 个词对是 2024 300d 嵌入的预测在阈值内而 2014 300d 嵌入偏离的。相反，有 197 个词对是 2024 嵌入偏离而 2014 嵌入保持在阈值内的。对于 SimLex999 数据集，这些数字分别为 14 和 43，对于 WS353 数据集，分别为 10 和 24。

比较两个嵌入模型在 MEN 上的表现，2024 嵌入擅长捕获近义词和上下位词关系（如 cemetery – graveyard、stair – staircase、ice – snow、sea – water）。在这些实例中，2024 模型与数据集中的高相似度评级紧密对齐，而 2014 嵌入倾向于低估它们的相似度。这表明 2024 嵌入可能在聚类共享核心、近等价意义或清晰部分-整体关系的词方面做得更精确。

然而，也有一些 2014 优于 2024 的情况。这些通常涉及更宽松的主题或分布关联，如颜色词（blue – red、purple – yellow）或数据集因类别或上下文而视为相当相似的常见日常物品（如 daffodil – tulip、puddle – splash、chicken – lamb、potato – tomato）。在这些场景中，2024 模型要么高估要么低估相似度，而 2014 更准确地捕获了这些更广泛的关系。

在 WS353 数据集上，我们观察到与先前比较中出现的类似模式。2024 嵌入倾向于更精确地捕获近义词和明显相关的类别对。例如，gem – jewel 或 coast – shore 等对具有 2024 紧密对齐的高真实相似度得分，而 2014 嵌入低估了这些紧密连接。magician – wizard 或 lobster – food 等其他近义词或上下位词示例也是如此，表明 2024 更可靠地编码强、直接的词汇关系。

相反，2014 在更宽松或更上下文的关联中表现出色，其中概念在功能或主题上相关而非严格同义。例如，plane – car 或 energy – laboratory 在 WS353 中获得中等偏高的相似度评级，反映它们共享一个总体类别或上下文（交通、科学研究）。2024 模型低估了这些关系，暗示它没有同样好地捕获某些宽泛或宽松的概念联系。

在 SimLex-999 上，我们再次看到两个模型在不同领域表现出色。2024 嵌入难以处理一些共享领域或具有中等概念重叠的对，如 disease – infection 或 river – sea。虽然这些对不是同义词，但 SimLex-999 给它们相当高的相似度评级，2024 比 2014 更低估这一点。类似地，2024 有时低估近义动词（如 deserve – earn、remain – retain、replace – restore），暗示即使这些对具有显著程度的语义接近性，它也可能失去对微妙词汇重叠的追踪。

此外，在 2024 Dolma 300 维嵌入与人类标注的前十个偏差中，所有十个例子都是模型在反义词之间给出高相似度（如 agree – argue）。对于 2024 Wiki/Giga 300 维向量，10 个中有 6 个是高度相似的反义词，另外 4 个是没有被给予足够高相似度的同义词（如 creator – maker）。

总体而言，在 MEN、WS353 和 SimLex-999 上出现了一致的模式：2024 嵌入特别擅长捕获强、直接的关系（近义词、上下位词联系和部分-整体连接），同时有时低估更上下文相关的关联。相比之下，2014 嵌入倾向于在更宽松的主题或分布关系上表现更好（如颜色词、共享领域、功能重叠），但偶尔在最优明显的高相似度对上失败，如严格同义词。

与直接评估结果（尚不清楚哪种嵌入表现更好）形成对比的是，NER 评估揭示了 2024 嵌入在更新和领域外数据集上优于 2014 嵌入。在经典的 2003 CoNLL 数据集上，结果几乎相同，暗示 2014 嵌入对于反映其原始训练领域和时代的语料库仍然足够。然而，在 CoNLL-PP 和更新的 Worldwide 和 WNUT-17 数据集上，2024 嵌入展现一致的改进。

检查混淆矩阵有助于阐明为什么 2024 嵌入在更新数据上表现更好。例如，表 12 强调了 2024 模型如何减少将新突出的实体错误分类到 O（无实体）类别或错误类型（如将 'COVID-19' 标记为 O）。相反，更新的嵌入捕获当代术语，因此更可能给它们分配正确的标签，通常是 MISC，而不是让它们未标注或与 LOC、ORG 或 PER 混淆。类似地，在表 13 中，我们看到 2024 模型不太可能将非西方人名与地点混淆，这一转变反映了新语料库中对此类实体的更多接触或更好表示。

**表 10：WNUT-17 测试集上的 2024/2014 300d Wiki/Giga 混淆矩阵差值**

| t\p | O | CORP | CREATIVE-WRK | GROUP | LOC | PER | PROD |
|---|---|---|---|---|---|---|---|
| O | 58 | 8 | −61 | −1 | −18 | 8 | 6 |
| CORP | 0 | 5 | 0 | 2 | −6 | −1 | 0 |
| CREATIVE-WRK | 26 | −3 | −39 | 7 | 8 | −2 | 3 |
| GROUP | 1 | 1 | −2 | 4 | 1 | −3 | −2 |
| LOC | −1 | 1 | −7 | −4 | 16 | −4 | −1 |
| PER | 2 | 5 | −7 | 15 | −5 | −10 | 0 |
| PROD | 186 | 10 | 1 | 7 | 3 | 11 | 35 |

**表 11：NER 数据集的示例句子，显示 2024 模型对加粗词的正确标注与 2014 模型的标注对比**

| 句子 | 2024 | 2014 |
|---|---|---|
| His repeated questioning of the system has prompted the Supreme Court to open an investigation into Bolsonaro.（Worldwide） | PER | LOC |
| Nationwide, COVID-19 infections in United States are at their peak with an average of 193,863 new cases reported each day over the past week . . .（CoNLL-PP） | MISC | O |
| Man Finna bring me a [emoji] up to my job.[^3]（WNUT-17） | O | PER |

**表 12：CoNLL-PP 测试集上的 2024/2014 300d Wiki/Giga 混淆矩阵差值**

| t \ p | O | LOC | MISC | ORG | PER |
|---|---|---|---|---|---|
| O | −17 | −3 | 5 | 18 | −3 |
| LOC | −3 | −19 | 2 | 21 | −1 |
| MISC | 12 | −8 | 8 | 1 | −12 |
| ORG | 13 | −21 | 13 | 2 | −7 |
| PER | 22 | −3 | −4 | −12 | −3 |

**表 13：Worldwide 测试集上的 2024/2014 300d Wiki/Giga 混淆矩阵差值**

| t \ p | O | LOC | MISC | ORG | PER |
|---|---|---|---|---|---|
| O | −19 | 24 | 68 | −52 | −21 |
| LOC | 14 | −29 | 24 | −2 | −7 |
| MISC | 6 | −48 | 138 | −70 | −26 |
| ORG | 32 | −36 | 63 | −30 | −29 |
| PER | 23 | 1 | −6 | −2 | −16 |

这些模式在表 11 的示例句子中显而易见。在来自 Worldwide 的第一个句子中，2024 嵌入能够将 'Bolsonaro' 标注为人，而 2014 将其标注为地点。来自 CoNLL-PP 的第二个示例显示 2024 正确标注 COVID-19，而 2014 嵌入没有标注它。这是近期使用中出现的词应该被正确标注的情况。来自 WNUT-17 的最后一个示例显示了该数据集标注有多么困难。尽管如此，2024 模型能够看出 'finna' 是俚语而不标注它，尽管 2014 模型将其标注为人。这是 2024 嵌入更好地捕获口语白话的示例。

总体而言，我们看到 2024 嵌入比 2014 嵌入更好地表示当前语言使用。在存在时间依赖的情况下（如聊天机器人、NER 标注器等），应该使用 2024 嵌入。

[^3]: 感谢 Twemoji 提供的 emoji！

## 6 结论

我们发布了新的 GloVe 词嵌入，引入了在 Wikipedia、Gigaword 和 Dolma 子集的更新语料库上训练的两套向量。这些嵌入提供了有价值的新词汇，反映了过去十年的文化和技术变迁，并为语言学家提供了分析英语演变的丰富数据，特别是在社交媒体影响力日益增长的背景下。

在词类比上，新嵌入与 2014 嵌入表现相当，展示了同等的结构和核心语义表现力。对于词相似度任务，较高维度的 2024 嵌入偶尔高估反义词之间的语义相似度。尽管如此，2024 嵌入在时间敏感的 NER 数据集（如面向非西方的新闻专线数据）上显示出明显优势。

这些发现强调了更新词嵌入以跟上语言和文化变化的重要性。2024 嵌入代表了现代语言建模的有意义进展，提供了与当代使用更好对齐的工具。它们捕获近期文化、技术和语言变迁的能力使它们对人类中心的 NLP 应用特别有价值，例如改进聊天机器人交互和设计适应多样化且不断演变的用户需求的系统。

## 参考文献

[1] Giusepppe Attardi. 2015. WikiExtractor. https://github.com/attardi/wikiextractor.

[2] Elia Bruni, Nam Khanh Tran, and Marco Baroni. 2014. Multimodal distributional semantics. J. Artif. Int. Res., 49(1):1–47.

[3] Leon Derczynski, Eric Nichols, Marieke van Erp, and Nut Limsopatham. 2017. Results of the WNUT2017 shared task on novel and emerging entity recognition. In Proceedings of the 3rd Workshop on Noisy User-generated Text, pages 140–147, Copenhagen, Denmark. Association for Computational Linguistics.

[4] Lev Finkelstein, Evgeniy Gabrilovich, Yossi Matias, Ehud Rivlin, Zach Solan, Gadi Wolfman, and Eytan Ruppin. 2001. Placing search in context: The concept revisited. ACM Transactions on Information Systems - TOIS, 20:406–414.

[5] Felix Hill, Roi Reichart, and Anna Korhonen. 2015. SimLex-999: Evaluating semantic models with (genuine) similarity estimation. Computational Linguistics, 41(4):665–695.

[6] Stanislaw Jastrzebski, Damian Lesniak, and Wojciech Marian Czarnecki. 2017. How to evaluate word embeddings? On importance of data efficiency and simple supervised tasks. ArXiv preprint arXiv:1702.02170.

[7] Shuheng Liu and Alan Ritter. 2023. Do CoNLL-2003 named entity taggers still work well in 2023? In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 8254–8271, Toronto, Canada. Association for Computational Linguistics.

[8] Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. 2013a. Efficient estimation of word representations in vector space. ArXiv preprint arXiv:1301.3781.

[9] Tomas Mikolov, Wen-tau Yih, and Geoffrey Zweig. 2013b. Linguistic regularities in continuous space word representations. In Proceedings of the 2013 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 746–751, Atlanta, Georgia. Association for Computational Linguistics.

[10] Robert Parker, David Graff, Junbo Kong, Ke Chen, and Kazuaki Maeda. 2011. English Gigaword Fifth Edition. Linguistic Data Consortium, Philadelphia, PA. LDC2011T07.

[11] Jeffrey Pennington, Richard Socher, and Christopher Manning. 2014. GloVe: Global vectors for word representation. In Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 1532–1543, Doha, Qatar. Association for Computational Linguistics.

[12] Peng Qi, Yuhao Zhang, Yuhui Zhang, Jason Bolton, and Christopher D. Manning. 2020. Stanza: A python natural language processing toolkit for many human languages. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics: System Demonstrations, pages 101–108, Online. Association for Computational Linguistics.

[13] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of Machine Learning Research, 21(140):1–67.

[14] Alexander Shan, John Bauer, Riley Carlson, and Christopher Manning. 2023. Do "English" named entity recognizers work well on global Englishes? In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 11778–11791, Singapore. Association for Computational Linguistics.

[15] Luca Soldaini, Rodney Kinney, Akshita Bhagia, Dustin Schwenk, David Atkinson, Russell Authur, Ben Bogin, Khyathi Chandu, Jennifer Dumas, Yanai Elazar, Valentin Hofmann, Ananya Harsh Jha, Sachin Kumar, Li Lucy, Xinxi Lyu, Nathan Lambert, Ian Magnusson, Jacob Morrison, Niklas Muennighoff, Aakanksha Naik, Crystal Nam, Matthew E. Peters, Abhilasha Ravichander, Kyle Richardson, Zejiang Shen, Emma Strubell, Nishant Subramani, Oyvind Tafjord, Pete Walsh, Luke Zettlemoyer, Noah A. Smith, Hannaneh Hajishirzi, Iz Beltagy, Dirk Groeneveld, Jesse Dodge, and Kyle Lo. 2024. Dolma: An Open Corpus of Three Trillion Tokens for Language Model Pretraining Research. arXiv preprint arXiv:2402.00159.

[16] Erik F. Tjong Kim Sang and Fien De Meulder. 2003. Introduction to the CoNLL-2003 shared task: Language-independent named entity recognition. In Proceedings of the Seventh Conference on Natural Language Learning at HLT-NAACL 2003, pages 142–147.

[17] Andrea Vallebueno, Cassandra Handan-Nader, Christopher D Manning, and Daniel E. Ho. 2024. Statistical uncertainty in word embeddings: GloVe-V. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 9032–9047, Miami, Florida, USA. Association for Computational Linguistics.
