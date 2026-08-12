# Semantic IDs for Industrial Recommendation: Mitigating the Curse of Large and Dynamic ID Spaces

> \*哥伦比亚大学, †AI at Meta

本文介绍了 Semantic IDs for Industrial Recommendation: Mitigating the Curse of Large and Dynamic ID Spaces。核心内容：
关键发现：
---
## 面向工业推荐的语义ID：缓解大规模动态ID空间的诅咒
Carolina Zheng\*¹, Minhui Huang†, Dmitrii Pedchenko†, Kaushik Rangadurai†, Siyu Wang†, Gaby Nahum†, Jie Lei†, Yang Yang†,
Tao Liu†, Zutian Luo†, Xiaohan Wei†, Dinesh Ramasamy†, Jiyan Yang†, Yiping Han†, Lin Yang†, Hangjun Xu†, Rong Jin†, Shuang Yang†
¹工作完成于 2024 年在 Meta 实习期间
通信作者：cz2539@columbia.edu, {mhhuang, dmitripedchenko, krangadu, siyuw, gnahum12345, jielei, yzyang, tliu97, zutianluo, ubimeteor, dineshr, chocjy, yipinghan, ylin1, rongjinml, hangjunxu, shuangyang}@meta.com
关键词：推荐系统，内容理解，表示学习，向量量化
## 摘要
在线内容的指数级增长给工业推荐系统中基于 ID 的模型带来了重大挑战，包括极高的基数、动态增长的 ID 空间、高度倾斜的互动分布，以及由于自然 ID 生命周期（例如新 ID 的诞生和旧 ID 的退休）导致的预测不稳定性。为了解决这些问题，许多系统依赖随机哈希来处理 ID 空间并控制相应的模型参数（即嵌入表）。然而，这种方法引入了多个 ID 共享同一嵌入所带来的数据污染，导致模型性能下降和嵌入表示不稳定。
本文审视了这些挑战，并引入了 **Semantic ID prefix ngram**（语义ID前缀 n-gram），一种新颖的词元参数化技术，显著提升了原始语义ID的性能。与随机分配不同，Semantic ID prefix ngram 通过基于item的内容嵌入对item进行层次聚类，从而创建具有语义意义的碰撞。通过大量实验，我们证明了 Semantic ID prefix ngram 不仅解决了嵌入不稳定性问题，还显著改进了尾部 ID 建模、减少了过拟合并缓解了表示偏移。我们进一步强调了 Semantic ID prefix ngram 在基于注意力的模型（其对用户历史进行上下文化）中的优势，展示了显著的性能提升。我们还报告了将 Semantic ID 集成到 Meta 生产广告排序系统中的经验，在在线部署中带来了显著的性能提升和预测稳定性增强。
## 1 引言
item推荐可以涉及许多信号丰富的特征，包括对应于item ID 的分类特征。原始item ID 通常被映射为嵌入，然后由基于深度学习的模型架构（如广泛部署的深度学习推荐模型 DLRM（Covington et al., 2016; Naumov et al., 2019））进一步处理。然而，在工业规模的在线环境中，学习item嵌入表示时出现了一些关键的数据相关挑战。具体来说，包括：**item基数**（item cardinality），即item的总数量巨大；**曝光偏斜**（impression skew），即只有少数item构成了大部分的用户曝光或转化（Milojević, 2010）；以及 **ID 漂移**（ID drifting），即大多数item在短时间内进入和离开系统（Gama et al., 2014）。
一种流行且简单的学习嵌入表示的方法是随机哈希，其中原始item ID 被随机哈希以共享相同的嵌入（Zhang et al., 2020）。由于item基数巨大以及系统对嵌入表大小的限制，哈希被广泛使用。然而，当模型长时间训练时，随机哈希和 ID 漂移共同导致了不可取的嵌入表示不稳定性。这是因为随机哈希碰撞的本质会导致对嵌入权重的矛盾梯度更新。此外，随着系统中item因 ID 漂移而随时间变化，旧item的学习成果会丢失，新item的嵌入权重基本上是随机的。这种方法对于曝光次数少的item（由于曝光偏斜，这些item构成了大多数）效果很差。
为了缓解这些缺陷，需要一种稳定的 ID 空间。稳定的 ID 空间理想地确保随着模型从更多数据中学习，学到的嵌入表示具有稳定的含义。在这项工作中，我们研究了一种最近提出的item表示方法——语义 ID（Semantic ID）（Singh et al., 2023; Rajput et al., 2024），将其作为稳定 ID 空间的候选方案。语义 ID 基于item的文本、图像或视频内容，根据语义相似性学习的层次聚类来推导item ID。然后，某个item的语义 ID 通过参数化方案映射为嵌入表示。重要的是，语义 ID 的 ID 空间是预先固定的，并且具有语义含义——这意味着它可以解决嵌入表示不稳定性问题。然而，在推荐建模中使用语义 ID 的一个挑战是，如何定义从其聚类分配到嵌入表的映射。
本文的主要贡献如下：
*   通过在 Meta 生产广告排序模型简化版本上的实验，我们加深了对语义 ID 如何改善嵌入表示稳定性的实证理解。我们进一步提出了 **Semantic ID prefix-ngram**，一种在语义 ID 之上的新颖词元参数化技术，与原始语义 ID（Singh et al., 2023）相比带来了显著的性能提升。
*   我们根据item数量（item基数）、大多数item曝光次数少（曝光偏斜）以及item在系统中的短生命周期（ID 漂移）来刻画item数据分布，并解释了它们与嵌入表示稳定性的关系。
*   我们描述了将 Semantic ID prefix-ngram 生产化为 Meta 生产系统中稀疏特征和序列特征的过程。我们展示了添加这些特征带来了在线性能提升并改善了在线预测稳定性。
在 Meta 广告排序数据的离线实验中，我们展示了语义 ID 相比随机哈希改进了泛化能力，并且对分布偏移的敏感性更低。证实了我们对曝光偏斜的假设，我们发现语义 ID 的大部分收益来自item分布的长尾。我们展示了通过纳入层次聚类信息，提出的 prefix-ngram 对语义 ID 的有效性至关重要。我们还证明了语义相似性在在线和离线设置下都能转化为预测相似性（第 6.3 节和第 7.4 节）。此外，当将语义 ID 纳入对用户item交互历史进行上下文化的模型时，其带来的收益格外显著。
在在线设置中，我们描述了语义 ID prefix-ngram 特征在 Meta 生产广告推荐系统中的实现。这些特征根据特征重要性成为最重要的稀疏特征，并带来了 0.15% 的在线性能提升。最后，我们发现纳入语义 ID 特征显著降低了模型对同一item的预测方差。这对于确保广告主对 Meta 推荐系统的信任以及改善最终item排序的稳定性至关重要。
剩余部分组织如下：第 2 节解释相关工作。第 3 节提供排序模型的概述。第 4 节介绍语义 ID 和词元参数化。第 5 节解释三个item曝光分布挑战。第 6 节描述离线实验。第 7 节描述语义 ID 在 Meta 的生产化以及在线实验。第 8 节总结。
## 2 相关工作
**推荐中的item表示** 许多现代深度学习推荐模型使用训练好的嵌入来表示分类（"稀疏"）特征（Covington et al., 2016; Naumov et al., 2019; Naumov, 2019）。高item基数的一个简单解决方案是使用随机哈希（Weinberger et al., 2009），但随机的哈希碰撞可能是不理想的。一种选择是修改哈希过程。在这一类别中，无碰撞哈希（Liu et al., 2022）通过动态释放已退休item的嵌入内存，为每个item引入单独的嵌入。双重哈希（Zhang et al., 2020）利用两个独立的哈希函数来减少内存使用，但仍然存在随机碰撞。学习哈希方法（Wang et al., 2017）通过训练基于 MLP 的哈希函数来关注相似性保持。还有一些工作通过对比学习或聚类来解决曝光偏斜问题（Yao et al., 2021; Chang et al., 2024）；我们将这些视为互补的方法。我们采取一种整体方法来设计稳定的 ID 空间，以最小化对哈希的需求，并直接解决嵌入表示偏移问题。
**稳定的嵌入表示** 稳定的 ID 受到 NLP 中词元化方法的启发，这些方法学习一个固定的词汇表来表示文本中的词元（Sennrich, 2015; Kudo, 2018; Devlin, 2018）。在为item推荐设计词元化方案时，Hou et al. (2023) 提出对从item内容理解模型学到的嵌入进行向量量化；Qu et al. (2024) 引入了一种掩码向量量化器，将从协同过滤模型学到的表示转移到生成式推荐器。语义 ID 由 (Singh et al., 2023; Rajput et al., 2024) 同时提出，其基于 (Hou et al., 2023) 并使用 RQ-VAE 进行量化，分别展示了其在泛化性能和序列推荐方面的优势。在这项工作中，我们将语义 ID 作为我们的稳定 ID 方法，并分析其在解决在线item推荐中三个挑战的有效性。
## 3 排序模型概述
推荐问题被建模为一个分类任务，其中数据点是与item曝光或转化相关联的用户侧和item侧特征，以及一个指示用户是否对该item互动或转化的二元标签。我们现在简要概述排序模型架构。
### 3.1 模型
推荐系统采用基于 DLRM（Covington et al., 2016; Naumov et al., 2019）的深度神经网络架构。模型由三个堆叠的部分组成：首先是信息聚合部分（information aggregation section），其中稀疏（即分类）特征、稠密特征和基于用户历史的特征被独立处理。每个模块的输出是一个嵌入向量的列表。其次，这些向量被拼接成一个单一的列表，通过交互层（interaction layer），在该层中，所有向量对之间进行点积（或高阶交互）。第三，交互层的输出通过一个 MLP 转换为 logits 分数，然后通过 sigmoid 函数输出概率。模型使用交叉熵损失进行训练。
在本文的剩余部分，我们专注于模型的信息聚合部分。
**嵌入模块** 令 *I* 为系统中原始 ID 的总数，令 [1..*N*] 表示从 1 到 *N* 的整数。
嵌入表是一个矩阵 **E**
$$
\in
$$
 ℝ^(H
$$
\times
$$
d_m)，其中 *d_m* 是嵌入维度，*H* 是嵌入数量。令 **f** = ( $f_1$ , ..., f_G) : [1..*I*] $\to$ [1..*H*]^G 为一个嵌入查找函数，它将原始 ID 映射到 G 个嵌入表行索引。然后，对于每个原始 ID *x*
$$
\in
$$
 [1..*I*]，稀疏模块查找嵌入行 $e_{f1}$ (x), ..., $e_{f_G}$ (x)，并通过 sum-pooling 生成单一的输出嵌入：**e**_f(x) := $\Sigma$ _{i=1}^{G} $e_{f_i}$ (x)。
**稀疏模块** 一个稀疏特征是一组原始 ID **x** := { $x_1$ , ..., x_n}。例如，这可以是一个item所属的 n 个产品类别 ID 的集合。我们通常通过 sum-pooling 对组成原始 ID 的嵌入 e_f(x_i) 求和，以产生一个单一的嵌入 e_f(x)。
**用户历史模块** 我们将用户的item交互历史建模为稀疏特征的序列 **x^u** := (x^ $u_1$ , ..., x^u_T) 和相应的时间戳。处理这些特征时，由于item数量和序列长度 T 的限制，存在系统约束。我们包含最多三个月的item交互历史，这使模型需要处理的item基数超过 10 亿。用户历史模块在特征被下游进一步处理之前对其进行上下文化处理，这一点很重要。我们在下面描述该架构。
首先，我们使用稀疏模块嵌入每个稀疏特征 x^u_i，并获得一个学习到的时间戳嵌入；两者之和为 e_f(x^u_i)。令 **X** = [e_f(x^ $u_1$ ); ...; e_f(x^u_T)]^T
$$
\in
$$
 ℝ^(T
$$
\times
$$
d_m) 表示结果编码。然后，我们通过一个聚合模块对该嵌入序列进行上下文化处理。我们使用以下三种聚合模块架构之一：**Bypass**、**Transformer** 和 **Pooled Multihead Attention (PMA)**，它们在附录 A 中定义。
### 3.2 指标
**归一化熵 (Normalized Entropy)** 我们通过归一化熵（NE）来衡量模型性能，其定义为模型交叉熵除以从预测数据正标签平均频率中获得的交叉熵。NE 公式如下：
NE = [-(1/N) $\Sigma$ _{i=1}^{N} (y_i log(p_i) + (1-y_i) log(1-p_i))] / [-(p log(p) + (1-p) log(1-p))]   (1)
其中 N 是训练样本数，y_i
$$
\in
$$
 {0,1} 是样本 i 的标签，p_i 是模型对样本 i 的预测，p = ( $\Sigma$ _{i=1}^{N} y_i) / N。值越低越好。
## 4 Semantic ID 与参数化
语义 ID 的主要动机是设计一种高效的聚类方案来表示item，使得具有共享语义的item之间可以进行知识共享。直观地说，如果我们有数百个关于披萨的广告被不同用户点击，我们希望涉及其中一个广告的示例能够从其他广告的表示中获得信息。我们精心设计了语义 ID，以潜在地解决第 5 节中描述的与数据相关的挑战：item基数、曝光偏斜和 ID 漂移。与基于随机聚类的嵌入表示相比，基于语义的表示很可能随着时间的推移更加稳定。基于语义的聚类还将允许尾部item从更多的训练样本中学习。已经离开系统的item的学习成果也可以被利用，新item的嵌入权重不必从头学习。我们在第 6 节中对这些假设进行了实证研究。
首先，我们在第 4.1 节中概述语义 ID。然后，在第 4.2 节中描述词元参数化。这一步对于将语义 ID 纳入推荐模型至关重要。
### 4.1 概述
语义 ID 分两个阶段为item学习得到：首先，对item的文本、图像或视频应用内容理解模型，以生成稠密的内容嵌入。然后，在内容嵌入上训练一个 RQ-VAE（Zeghidour et al., 2021），以获得每个item的向量量化，该量化被表示为一个从粗到细的离散编码序列，称为item的语义 ID。
令 *L* 为层数（即序列长度），*K* 为码本大小（即每层的聚类数）。RQ-VAE 由一个编码器组成，该编码器将内容嵌入 **x**
$$
\in
$$
 ℝ^D 映射到连续的潜在表示 **z**
$$
\in
$$
 ℝ^D'，一个残差量化器将 **z** 量化为一串离散编码 **c** := ( $c_1$ , ..., c_L)
$$
\in
$$
 K^L，以及一个从 **c** 重构 **x** 的解码器。这是通过将每一层 *l* 与一个码本（*K* 个向量的集合 {**v**^l_k}_{k=1}^{K}）相关联来实现的。离散编码序列是层次化的：c_l 对应于码本向量 **v**^ $l_{c_l}$ ，它近似于 r_l，即在对来自层 (l-1) 到 1 的码本向量递归应用之后，从 **z** 中剩余的残差，即：
**r**_l := **z** - $\Sigma$ _{i=1}^{l-1} **v**^ $i_{c_i}$ ,   c_l := arg min_k ||**v**^l_k - **r**_l|| $_2$   (2)
在第 4.2 节中，我们提供了关于 RQ-VAE 层次聚类性质的更多直观理解，以及它如何指导词元参数化的选择。
RQ-VAE 使用两个损失项进行训练：一个重构损失和一个鼓励残差与码本向量接近的损失，
L_RQ-VAE(x) = ||**x** - dec(**c**)||² + $\Sigma$ _{l=1}^{L} $\beta$ ||**r**_l - sg(**v**^ $l_{c_l}$ )||² + ||sg(**r**_l) - **v**^ $l_{c_l}$ ||²
其中 dec(**c**) 是将解码器应用于编码 **c** 的结果，sg(·) 表示停止梯度操作符， $\beta$ 是一个超参数，我们在实验中设置为 0.5。语义 ID 被定义为由编码器和残差量化器产生的离散编码序列 ( $c_1$ , ..., c_L)。
### 4.2 词元参数化
在我们的实验中，我们为每一层使用相同的码本大小，总共得到 K^L 个聚类。RQ-VAE 的一个重要特性是它产生层次化的聚类。为简单起见，假设 L=3，一个原始item ID 被映射为一个序列 ( $c_1$ , $c_2$ , $c_3$ )。向量量化的精度随着从第一个词元 $c_1$ 移动到更深层的词元 $c_2$ ，最后到 $c_3$ 而增加。第一个词元 $c_1$ 表示最粗略的桶：例如，所有与食物相关的广告。第二个词元 $c_2$ 精化这一信息，例如，( $c_1$ , $c_2$ ) 可能表示所有与披萨相关的广告。最后一个词元 $c_3$ 进一步精化信息，例如，( $c_1$ , $c_2$ , $c_3$ ) 可能表示所有与披萨相关且用特定语言（如英语）撰写的广告。
因此，我们可以控制推荐模型从语义 ID 接收的信息量和结构。值得注意的是，提供最精细的信息（所有可能的 ( $c_1$ , $c_2$ , ..., c_L) 元组）通常是不可行的，因为可能的组合基数很高。因此，在词元参数化的基数和模型从语义 ID 接收的信息量之间存在权衡。
**词元参数化技术**
令 s(x) : [1..I] $\to$ K^L 为语义 ID 查找函数，将原始 ID 映射到由 RQ-VAE 学到的语义 ID。考虑到词元的层次性质，我们必须指定一个词元参数化方案，将语义 ID 映射到嵌入表行，p(c; H) : K^L $\to$ [1..H]^G。表 1 定义了几种可能的参数化方案。当语义 ID 基数大于嵌入大小时，应用模哈希函数。当存在多个 ID 时，添加一个偏移因子以避免不同位置之间的碰撞。在所有参数化技术中，只有 Prefix-ngram 包含来自不同粒度的所有可能元组。
**表 1：词元参数化方案 p( $c_1$ , ..., c_L; H)**
| 词元参数化 | 公式 |
|---|---|
| Trigram (三元组) | [K² $c_1$ + K $c_2$ + $c_3$ ] |
| Fourgram (四元组) | [K³ $c_1$ + K² $c_2$ + K $c_3$ + $c_4$ ] |
| All bigrams (所有二元组) | [K²
$$
\times
$$
(i-1) + Kc_i + $c_{i+1}$ , for i in [1..L-1]] |
| Prefix-ngram (前缀 n-gram) | [ $\Sigma$ _{t=1}^{i} K^{i-t}(c_t+1) - 1, for i in [1..n]] |
**表 2：不同词元参数化的 NE 性能**
| RQ-VAE K
$$
\times
$$
L | 词元参数化 | 训练 NE 增益 |
|---|---|---|
| [2048]
$$
\times
$$
 3 | Trigram | −0.028% |
| [2048]
$$
\times
$$
 4 | Fourgram | −0.035% |
| [2048]
$$
\times
$$
 4 | All bigrams | −0.091% |
| [512]
$$
\times
$$
 3 | Prefix-3gram | −0.034% |
| [1024]
$$
\times
$$
 3 | Prefix-3gram | −0.097% |
| [2048]
$$
\times
$$
 3 | Prefix-3gram | −0.141% |
| [2048]
$$
\times
$$
 5 | Prefix-5gram | −0.208% |
| [2048]
$$
\times
$$
 6 | Prefix-6gram | −0.215% |
表 2 总结了不同词元参数化的模型性能。我们得出以下结论：(i) Prefix-ngram 是最好的参数化方法。这表明将聚类的层次性质纳入嵌入表映射中对有效性至关重要，因为它允许比平面映射在更多item之间进行知识共享；(ii) 增加 Prefix-ngram 的深度可以提升 NE 性能；(iii) 增加 RQ-VAE 的基数可以提升 NE 性能。
## 5 item曝光分布问题
在本节中，我们讨论在 Meta 广告排序中给推荐建模带来挑战的数据分布方面的问题，以及我们如何通过使用语义 ID 来解决它们。
**item基数** 对于某些特征，例如目标item，模型考虑的不同item数量 I 可能远大于稀疏模块中可行的嵌入表大小 H。在这种情况下，映射函数 f(x) 会引入碰撞：两个或多个原始 ID 将映射到同一行。映射函数 f(x) 通常被选择为一个简单的哈希函数。由于原始 ID 在item创建时是随机生成的，因此产生的碰撞本质上是随机的。这种随机碰撞会对嵌入的表示质量产生负面影响，并成为跨item有效知识共享的障碍。
**曝光偏斜** 对于目标item特征，训练数据中的item分布高度倾斜。图 2 显示，在我们的系统中，一小部分item主导了item曝光分布：按流行度对item排序时，前 0.1% 的"头部"item拥有 25% 的item曝光量，接下来 5.5% 的"躯干"item拥有 50% 的累积曝光量，而剩余的 94.4% 的"尾部"item则占剩余的 25% 曝光量。
由于尾部item的训练样本很少，学习能够很好泛化的嵌入表示 e(x) 可能具有挑战性。随机哈希不允许头部和躯干item与语义相似的尾部item有效地共享知识，因为多个item到单个嵌入的分配是随机的。
**ID 漂移** 现有的item ID 空间高度动态，大量旧item退出（图 3）和新item进入系统。我们将系统中的这种item分布偏移称为"原始 ID 漂移"。原始 ID 漂移现象源于在线推荐系统的本质，即每天都有新广告被创建，且大多数广告的生命周期相对较短。
作为副产品，基于随机哈希的推荐模型会经历严重的嵌入表示随时间漂移：随着item进入和退出系统，给定的嵌入 e 在不同时间代表了不同的item。
**使用语义 ID 的item表示** 我们假设从原始 ID 切换到语义 ID 可以有效解决上述问题。
当广告主将新广告 x 引入系统并退休之前的广告 y 时，新广告的细粒度内容细节可能与已退休的广告不同，但产品的广泛语义类别通常保持不变。因此，新广告和已退休广告的语义 ID 将匹配（或至少共享一个前缀）。因此，只要广泛的语义类别在时间上保持稳定，语义 ID 空间中的item曝光分布与原始 ID 空间相比表现出更少的漂移。
类似地，如果尾部item x 与头部或躯干item y 具有相似的内容，它们的语义 ID 将匹配（或至少共享一个前缀）。由此产生的语义 ID 空间中的item曝光分布与原始 ID 空间相比表现出更少的偏斜（见附录 B）。
在上述两种情况下，嵌入 e(x) 和 e(y) 将相等（或如果语义 ID 仅共享一个前缀则相似）。这是模型将知识从具有许多训练样本的item y 转移到item x 的一种方式。总结来说，语义概念的时间稳定性导致语义 ID 编码的稳定性，从而缓解了模型的嵌入表示不稳定性问题。
## 6 离线实验
为了研究我们关于语义 ID 相对于基线item表示方法优势的假设，我们进行了一系列离线实验。
我们使用 Meta 生产广告排序模型的简化版本，保留所有稠密特征和用户的item交互历史，但稀疏模块中仅包含目标item（并移除约 100 个其他稀疏特征）。我们在来自四天时间段的生产用户交互数据上进行训练，顺序处理训练数据并训练一个 epoch。我们在下一天数据的前六个小时上评估模型。
### 6.1 基线
在第 5 节中，我们概述了在设计用于基于嵌入的item表示的良好嵌入查找函数 f(x) 时面临的数据相关挑战和机遇。我们描述两种基线方法——**独立嵌入 (IE)** 和**随机哈希 (RH)**——并将其与**语义 ID (SemID)** 进行比较。
**独立嵌入 (IE)** 每个原始 ID 获得自己的嵌入表行，I = H，且 f_IE(x) := x。虽然由于系统限制在生产场景中不现实，但我们出于说明目的考虑此模型。在评估期间，训练期间未见过的 ID 被映射到随机初始化的未训练嵌入。
**随机哈希 (RH)** 在 I $\approx$ a·H (a > 1) 的情况下，我们可以将原始 ID 随机哈希到嵌入表行，f_RH(x) := h(x)，其中 h(x) : [1..I] $\to$ [1..H] 是标准哈希函数之一，如模哈希。这会产生随机碰撞，平均碰撞因子为 a。
**语义 ID (SemID)** item的内容嵌入来自一个多模态图像和文本基础模型。该基础模型在使用图像和文本对齐目标的大规模item训练集上进行了预训练（Radford et al., 2021）。然后，在最近三个月的所有目标item的内容嵌入上训练 RQ-VAE，其中 L=3，K=2048。我们使用第 4.2 节中的 prefix-3gram 参数化，f_SemId = p ∘ s。
第 6.2 节和第 6.3 节的分析聚焦于目标item稀疏特征。我们使用上述三种嵌入查找函数训练三个版本的推荐模型。item嵌入表的大小对于 IE 等于item总数，对于 RH 和 SemID 设置为较小的尺寸，平均碰撞因子为 3。用户历史特征使用随机哈希映射。
在第 6.4 节中，我们将语义 ID 用于用户历史特征，并研究其对聚合模块架构的影响。item交互历史序列长度固定为 O(100)。我们填充或截断用户历史以适应所需的长度。
### 6.2 分段分析
为了理解曝光偏斜对每种方法的影响，我们根据item在训练期间的曝光次数对数据进行分段。我们按曝光次数对所有item进行排序。与之前一样，我们将item分为头部、躯干和尾部item，具体标准为它们在此排序顺序中是否产生了 25%、75% 或 100% 的累积曝光次数。由于曝光偏斜，头部、躯干或尾部item的百分比分别为 0.1%、5.5% 和 94.4%。我们还在仅在评估期间出现且在训练期间未见的新item分段上进行评估。三种item表示方法的性能如表 3a 和 3b 所示。
**表 3：三种item表示方法在不同item分段上的性能**
**(a) 评估 NE（越低越好）。语义 ID 实现了向尾部和新冷启动item的知识迁移。**
| 累积曝光百分比 | item百分位 | RH | IE | SemID | 相对于 RH 的 SemID NE 增益 | 相对于 IE 的 SemID NE 增益 |
|---|---|---|---|---|---|---|
| 25%（头部） | 0.1 | 0.80105 | 0.80101 | 0.80108 | 0.00% | 0.01% |
| 75%（躯干） | 5.6 | 0.83589 | 0.83583 | 0.83580 | −0.01% | −0.00% |
| 100%（尾部） | 100 | 0.83904 | 0.83886 | 0.83872 | −0.04% | −0.02% |
| 训练中见过的item |  | 0.82626 | 0.82612 | 0.82600 | −0.03% | −0.02% |
| 新item |  | 0.83524 | 0.83453 | 0.83180 | −0.41% | −0.33% |
| 所有item |  | 0.82663 | 0.82645 | 0.82621 | −0.05% | −0.03% |
**(b) 对分布偏移的敏感性：NE[ $t_0$ , $t_1$ ] − NE[ $t_0$ +42h, $t_1$ +42h]。越低越好。**
| 累积曝光百分比 | RH | IE | SemID |
|---|---|---|---|
| 25%（头部） | 0.0057 | 0.0065 | 0.0059 |
| 75%（躯干） | 0.0087 | 0.0075 | 0.0076 |
| 100%（尾部） | 0.0128 | 0.0103 | 0.0106 |
| 所有item | 0.0083 | 0.0074 | 0.0073 |
与基线相比，语义 ID 改善了尾部item的泛化能力，对头部item NE 中性，对躯干item略有好处。由于这也是相对于独立嵌入方法的，语义 ID 并不仅仅比随机哈希更好地聚类，而是我们发现目标item特征受益于基于语义的知识共享。
具体来说，知识共享是通过共享的嵌入权重发生的，这些权重会收到语义相关item的更新。知识共享的收益在新item分段上最大，其中 SemID 相对于 RH 和 IE 都取得了较大的增益（分别为 -0.41% 和 -0.33%）。新item使用来自训练期间见过的语义相似item的预训练权重进行预测，而不是使用不相关的权重（RH）或未经训练的权重（IE）。
为了衡量嵌入表示漂移对模型性能的影响，我们在训练数据上但针对两个不同的时间段评估训练好的模型。我们计算训练 epoch 结束前 42-48 小时的 NE 并减去训练最后六小时的 NE。值越小表明由 ID 漂移引起的嵌入表示偏移对模型拟合的影响越小。这是因为我们的模型按时间顺序训练了一个 epoch，因此得到的模型在训练结束时学习拟合最近的训练时间段。结果见表 3b。
独立嵌入方法相较于随机哈希具有更小的性能差距。这突显了随机哈希存在 ID 漂移问题——随着时间的推移，随着权重使用新item样本进行更新，嵌入表示逐渐失去表示旧item的能力。相比之下，语义 ID 与独立嵌入的性能相匹配，表明其学到的表示随时间更稳定。
我们推测这种改进的表示稳定性也使模型在更长的训练时长上能够更好地泛化，此时 ID 漂移变得更加显著。我们将 RH 和 SemID 模型在 20 天的时间内进行训练，并将其与仅在 20 天期间的最后四天训练的相应模型进行比较。表 4 的结果表明，与随机哈希相比，语义 ID 的性能在更长时间段内随着训练数据的增加而更好地扩展。
**表 4：训练 20 天数据相较于仅训练 4 天的 NE 改进**
|  | RH | Semantic ID |
|---|---|---|
| 评估 NE 增益 | −0.18% | −0.23% |
### 6.3 item表示空间
为了更深入地理解item嵌入表示，我们从每个训练好的模型中提取学习到的嵌入权重。可以将随机哈希和语义 ID 视为划分原始item ID 语料库的两种不同方式。我们希望了解由语义 ID 产生的基于语义的划分是否比随机哈希产生的随机划分更适合推荐问题。
当多个item被分配到同一个划分时，它们会被嵌入查找模块映射到相同的嵌入。我们将此嵌入向量视为由独立嵌入模型学习的每item嵌入的摘要。虽然我们在本文中出于说明目的拟合了独立嵌入模型，但 IE 在工业级设置中是不切实际的。具有较低划分内嵌入方差和较高划分间距离的划分可以被视为更有效的独立嵌入摘要。我们为 IE 模型学到的嵌入的 RH 和 SemID 划分计算这些指标。
我们将此实验的碰撞因子设置为 5。结果，RH 和 SemID 划分的结果聚类平均包含 5 个item。然而，由于语义 ID 是由 RQ-VAE 模型学到的潜在编码，因此所得聚类的大小高度可变。我们计算两组语义 ID 聚类的指标：每组包含 4-10 个item的小聚类，以及前 1,000 个聚类，其中每个聚类包含数千个item。表 5 包含了平均方差和平均成对距离，括号内为标准差。指标在嵌入维度上取平均，以产生单一标量进行比较。
**表 5：随机哈希和基于语义 ID 的划分的聚类内和聚类间方差与成对距离**
|  | 方差 | 成对距离 |
|---|---|---|
| 随机哈希 | 1.52
$$
\times
$$
10⁻³ (8.0
$$
\times
$$
10⁻⁴) | 0.22 (0.04) |
| SemID（小聚类） | 1.31
$$
\times
$$
10⁻³ (1.0
$$
\times
$$
10⁻³) | 0.24 (0.09) |
| SemID（前 1,000 个聚类） | 1.23
$$
\times
$$
10⁻³ (5.5
$$
\times
$$
10⁻⁴) | 0.06 (0.02) |
我们观察到，与随机哈希相比，语义 ID 划分产生的聚类具有更低的聚类内方差。然而，产生的成对距离传递了混合的信号。我们推测前 1,000 个聚类之间的低成对距离是因为 RQ-VAE 将多个质心放置在数据密度最高的区域以最小化模型损失。
### 6.4 用户历史建模
在本节中，我们探讨语义 ID 在用户历史建模上的效果。该模块的作用之一是对用户历史进行上下文化处理和总结。
我们发现，使用语义 ID 和基于注意力的上下文化聚合模块（PMA 或 Transformer）相较于不对序列进行上下文化处理的基线（Bypass）带来了超额的收益。这些结果总结在表 6 中。
**表 6：三种聚合模块的性能。基线：各模块使用 RH 的模型。语义 ID 为上下文化模块带来更大的收益。**
|  | 训练 NE 增益 | 评估 NE 增益 |
|---|---|---|
| Bypass | −0.056% | −0.085% |
| Transformer | −0.071% | −0.110% |
| PMA | −0.073% | −0.100% |
为了理解使用语义 ID 如何改变 PMA 和 Transformer 聚合模块中学到的注意力模式，我们在 1,000 个评估样本的随机子集上计算了四个注意力分数指标。
令 **A**
$$
\in
$$
 ℝ^(T
$$
\times
$$
S) 为注意力分数矩阵，其中 T 是目标序列长度，S 是源序列长度。对于 Transformer 和 Bypass，T=S；对于 PMA，T=32。**A** 的每一行 $a_{i,:}$ 表示源词元上的概率分布。我们考虑的指标定义如下：
*   第一个源词元注意力：(1/T) $\Sigma$ _{i=1}^{T} $a_{i,1}$
*   填充词元注意力：(1/T) $\Sigma$ _{i=1}^{T} $\Sigma$ _{j=1}^{S} I{ $a_{i,j}$ = pad} · $a_{i,j}$
*   熵：(1/T) $\Sigma$ _{i=1}^{T} $\Sigma$ _{j=1}^{S} $a_{i,j}$ · lo $g_2$ $a_{i,j}$
*   词元自注意力：(1/T) $\Sigma$ _{i=1}^{T} $a_{i,i}$
**表 7：用户历史item交互特征的随机哈希和基于 SemID 模型的注意力分数评估指标**
|  | 第一个源词元 | 填充词元 | 熵 | 自注意力 |
|---|---|---|---|---|
| Transformer + RH | 0.030 | 0.460 | 2.149 | 0.052 |
| Transformer + SemID | 0.043 | 0.418 | 1.967 | 0.045 |
| PMA + RH | 0.071 | 0.351 | 3.075 | – |
| PMA + SemID | 0.074 | 0.313 | 3.025 | – |
从表 7 的指标读数中，我们看到使用语义 ID 训练的模型具有更低的熵、词元自注意力和填充词元注意力，以及更高的序列第一个源词元注意力分数。这意味着基于语义 ID 的模型将更多权重放在更高信号的词元上（即序列中的第一个和最近的item，而不是早期可能过时的词元或填充词元），注意力分数分布在整个序列上不那么分散（即更低的熵），对于 Transformer，将更高的权重放在其他词元上而不是自注意力上。这些指标是有希望的信号，表明语义 ID item表示在用户历史建模中比基于随机哈希的对应表示更稳定、更有意义。
## 7 生产化
语义 ID 特征已在 Meta 广告推荐系统中生产化超过一年。根据特征重要性研究，它们在现有广告排序模型中是最重要的稀疏特征。在本节中，我们提供在线服务管线的概述和关键实现细节。
**表 8：在 Meta 旗舰广告排序模型中纳入语义 ID 特征的 NE 改进**
|  | 训练 NE 增益 | 评估 NE 增益 |
|---|---|---|
| 基线 + 6 个稀疏特征 | −0.063% | −0.071% |
| 基线 + 1 个序列特征 | −0.110% | −0.123% |
### 7.1 离线 RQ-VAE 训练
RQ-VAE 模型是在 Meta 广告排序的内容理解（CU）模型上训练的。CU 模型在公开的 CC100 数据集（Conneau, 2020）上进行预训练，然后在内部广告数据集上进行微调。我们从过去三个月的数据中对广告 ID 及其对应的内容嵌入进行采样，并离线训练 RQ-VAE 模型。对于生产模型，我们训练 L=6 和 K=2048 的 RQ-VAE，语义 ID 遵循第 4.2 节中的 prefix-5gram 设计，其中 H=O(50M)。训练后，我们使用冻结的 RQ-VAE 检查点进行在线服务。
### 7.2 在线语义 ID 服务系统
图 4 展示了实时语义 ID 特征的在线服务管线。在广告创建时，我们处理广告内容信息并提供给 CU 模型。输出的 CU 嵌入然后通过 RQ-VAE 模型，该模型计算每个原始 ID 的语义 ID 信号。该信号然后存储在实体数据存储（Entity Data Store）中。在特征生成阶段，目标item原始 ID 和用户互动原始 ID 历史从实体数据存储中丰富语义 ID 信号，以产生语义特征。当服务请求到达时，预计算的特征被获取并传递给下游排序模型。
### 7.3 生产性能提升
我们从不同的内容嵌入源（包括文本、图像和视频）创建了六个稀疏特征和一个序列特征，并在表 8 中报告了在 Meta 旗舰广告排序模型上的 NE 增益。在 Meta 广告排序中，大于 0.02% 的离线 NE 增益被认为是有意义的。总体而言，在多个广告排序模型中，纳入语义 ID 特征在我们的首要在线指标上产生了 **0.15% 的性能提升**。由于 Meta 广告推荐系统服务于数十亿用户，并且是公司中优化最充分的模型之一，因此 0.15% 的在线性能增益被认为是有意义的。
### 7.4 语义与预测相似性
直观地说，人们可能会认为，如果两个item在语义上相似，它们的用户互动模式也会相似。然而，用户行为和感知更加微妙，并且不能根据语义进行可预测的连续变化。为了使用语义 ID 实现稳健的投放性能，我们必须确保排序模型行为相对于系统中item之间的语义相似性关系具有一定程度的连续性（或相关性）。
为了衡量这种相关性，我们进行了一项在线 A/B 测试，其中我们选择一个由系统推荐给用户的item集合 S。对于给定用户，以 50% 的概率，我们通过将 S 中的一个item随机替换为来自语义 ID 的具有相同前缀的不同item，将集合 S 变异为 S'。此操作产生：
点击损失率 (Click Loss Rate) := (S' 上的 CTR − S 上的 CTR) / (S 上的 CTR)   (3)
从语义 ID 使用更深层前缀而导致的点击损失率降低总结在图 5 中。
由于语义 ID 基于item语义对item语料库进行划分，我们得出结论：预测相似性与语义相似性相关。这支持了第 6.3 节中的表示空间分析结果。此外，语义 ID 中的编码层次结构有效地捕获了item语义的细粒度细节：更深层的前缀单调地降低了点击损失率。
### 7.5 A/A 方差
基于随机哈希的排序模型的另一个缺点是固有的模型预测方差，导致下游广告投放方差。具体来说，可以创建一个具有不同原始item ID 的item副本。然后，原始item和item副本都进入推荐系统。由于哈希后原始item和副本的嵌入将不同，因此模型预测和投放系统行为也将不同。我们将这种现象称为 **A/A 方差**，其中"A/A"表示我们考虑的是原始item的精确副本。¹ 这种方差是不可取的，因为它降低了下游广告排序顺序的鲁棒性和系统准确 targeting 正确受众的能力。语义 ID 通过消除由随机哈希引起的随机性有助于降低 A/A 方差——完全相同的副本或非常相似的item通常具有相同的 k-prefix 语义 ID。
我们设置了一个在线影子广告实验，其中我们测量给定模型的相对 A/A 预测差异（AAR）。对于 A/A 对 ( $a_1$ , $a_2$ )：
AAR( $a_1$ , $a_2$ ) := 2 |p( $a_1$ ) − p( $a_2$ )| / (p( $a_1$ ) + p( $a_2$ ) + $\epsilon$ )   (4)
其中 p(a_i) 是排序模型对item a_i 的预测。
具有六个语义 ID 稀疏特征的生产模型相较于不具有这六个特征的同一模型，在平均 AAR 上实现了 **43% 的降低**。我们相信 AAR 的降低主要来自尾部item，如第 6.2 节中所研究的。
## 8 结论
我们展示了如何使用语义 ID 为item表示创建稳定的 ID 空间，并提出了 Semantic ID prefix-ngram，它显著提升了语义 ID 在排序模型中的性能。在离线实验中，我们研究了训练好的排序模型，发现与随机哈希和独立嵌入基线相比，在语义 ID 下，嵌入表示不稳定性的有害影响得到了缓解。我们详细描述了语义 ID 特征在 Meta 广告推荐系统中的成功生产化，并展示了在线生产系统既获得了显著的性能提升，又降低了下游广告投放方差。
## 参考文献
Bo Chang, Changping Meng, He Ma, Shuo Chang, Yang Gu, Yajun Peng, Jingchen Feng, Yaping Zhang, Shuchao Bi, Ed H Chi, et al. 2024. Cluster Anchor Regularization to Alleviate Popularity Bias in Recommender Systems. In *Companion Proceedings of the ACM on Web Conference 2024*. 151–160.
Alexis Conneau. 2019. Unsupervised cross-lingual representation learning at scale. *arXiv preprint arXiv:1911.02116* (2019).
Paul Covington, Jay Adams, and Emre Sargin. 2016. Deep neural networks for youtube recommendations. In *Proceedings of the 10th ACM conference on recommender systems*. 191–198.
Jacob Devlin. 2018. BERT: Pre-training of deep bidirectional transformers for language understanding. *arXiv preprint arXiv:1810.04805* (2018).
João Gama, Indrė Žliobaitė, Albert Bifet, Mykola Pechenizkiy, and Abdelhamid Bouchachia. 2014. A survey on concept drift adaptation. *ACM computing surveys (CSUR)* 46, 4 (2014), 1–37.
Yupeng Hou, Zhankui He, Julian McAuley, and Wayne Xin Zhao. 2023. Learning vector-quantized item representation for transferable sequential recommenders. In *Proceedings of the ACM Web Conference 2023*. 1162–1171.
Taku Kudo. 2018. SentencePiece: A simple and language independent subword tokenizer and detokenizer for neural text processing. *arXiv preprint arXiv:1808.06226* (2018).
Juho Lee, Yoonho Lee, Jungtaek Kim, Adam Kosiorek, Seungjin Choi, and Yee Whye Teh. 2019. Set transformer: A framework for attention-based permutation-invariant neural networks. In *International conference on machine learning*. PMLR, 3744–3753.
Zhuoran Liu, Leqi Zou, Xuan Zou, Caihua Wang, Biao Zhang, Da Tang, Bolin Zhu, Yijie Zhu, Peng Wu, Ke Wang, and Youlong Cheng. 2022. Monolith: Real Time Recommendation System With Collisionless Embedding Table. *arXiv:2209.07663* [cs.IR]. https://arxiv.org/abs/2209.07663
Staša Milojević. 2010. Power law distributions in information science: Making the case for logarithmic binning. *Journal of the American Society for Information Science and Technology* 61, 12 (2010), 2417–2425.
Maxim Naumov. 2019. On the dimensionality of embeddings for sparse features and data. *arXiv preprint arXiv:1901.02103* (2019).
Maxim Naumov, Dheevatsa Mudigere, Hao-Jun Michael Shi, Jianyu Huang, Narayanan Sundaraman, Jongsoo Park, Xiaodong Wang, Udit Gupta, Carole-Jean Wu, Alisson G Azzolini, et al. 2019. Deep learning recommendation model for personalization and recommendation systems. *arXiv preprint arXiv:1906.00091* (2019).
Haohao Qu, Wenqi Fan, Zihuai Zhao, and Qing Li. 2024. TokenRec: Learning to Tokenize ID for LLM-based Generative Recommendation. *arXiv preprint arXiv:2406.10450* (2024).
Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. 2021. Learning transferable visual models from natural language supervision. In *International conference on machine learning*. PMLR, 8748–8763.
Shashank Rajput, Nikhil Mehta, Anima Singh, Raghunandan Hulikal Keshavan, Trung Vu, Lukasz Heldt, Lichan Hong, Yi Tay, Vinh Tran, Jonah Samost, et al. 2024. Recommender systems with generative retrieval. *Advances in Neural Information Processing Systems* 36 (2024).
Rico Sennrich. 2015. Neural machine translation of rare words with subword units. *arXiv preprint arXiv:1508.07909* (2015).
Anima Singh, Trung Vu, Nikhil Mehta, Raghunandan Keshavan, Maheswaran Sathiamoorthy, Yilin Zheng, Lichan Hong, Lukasz Heldt, Li Wei, Devansh Tandon, et al. 2023. Better Generalization with Semantic IDs: A Case Study in Ranking for Recommendations. *arXiv preprint arXiv:2306.08121* (2023).
Ashish Vaswani. 2017. Attention is all you need. *Advances in Neural Information Processing Systems* (2017).
Jingdong Wang, Ting Zhang, Nicu Sebe, Heng Tao Shen, et al. 2017. A survey on learning to hash. *IEEE transactions on pattern analysis and machine intelligence* 40, 4 (2017), 769–790.
Kilian Weinberger, Anirban Dasgupta, John Langford, Alex Smola, and Josh Attenberg. 2009. Feature hashing for large scale multitask learning. In *Proceedings of the 26th annual international conference on machine learning*. 1113–1120.
Tiansheng Yao, Xinyang Yi, Derek Zhiyuan Cheng, Felix Yu, Ting Chen, Aditya Menon, Lichan Hong, Ed H Chi, Steve Tjoa, Jieqi Kang, et al. 2021. Self-supervised learning for large-scale item recommendations. In *Proceedings of the 30th ACM international conference on information & knowledge management*. 4321–4330.
Neil Zeghidour, Alejandro Luebs, Ahmed Omran, Jan Skoglund, and Marco Tagliasacchi. 2021. Soundstream: An end-to-end neural audio codec. *IEEE/ACM Transactions on Audio, Speech, and Language Processing* 30 (2021), 495–507.
Caojin Zhang, Yicun Liu, Yuanpu Xie, Sofia Ira Ktena, Alykhan Tejani, Akshay Gupta, Pranay Kumar Myana, Deepak Dilipkumar, Suvadip Paul, Ikuhiro Ihara, et al. 2020. Model size reduction using frequency based double hashing for recommender systems. In *Proceedings of the 14th ACM Conference on Recommender Systems*. 521–526.
## 附录 A：聚合模块架构
**Bypass** 对每个嵌入单独应用线性权重矩阵 **W**
$$
\in
$$
 ℝ^(d_m
$$
\times
$$
 d_m)，
Bypass(**X**) := **XW**   (5)
**Transformer** (Vaswani, 2017) 对嵌入序列应用一个 Transformer 层。注意力子模块定义为：
Attention(**X**) := softmax((**XW**_Q)(**XW**_K)^T / $\sqrt{}$ d_m) (**XW**_V)   (6)
其中 **W**_Q, **W**_K, **W**_V
$$
\in
$$
 ℝ^(d_m
$$
\times
$$
 d_a) 分别是 query、key 和 value 的权重矩阵，d_a 是 query/key/value 向量维度。完整的 Transformer 模块由下式给出：
**X**^(1) = Attention(LayerNorm(**X**)) + **X**   (7)
**X**^(2) = MLP(LayerNorm(**X**^(1))) + **X**^(1)   (8)
其中 LayerNorm 和 MLP 分别表示标准的层归一化和逐位置的 MLP 层。我们在应用 Transformer 或 PMA 模块之前，向编码中添加标准的位置嵌入。
**Pooled Multihead Attention (PMA)** (Lee et al., 2019) 对嵌入序列应用一个 Transformer 层，但将注意力 query 向量替换为 d_s 个可学习的权重向量。PMA 注意力子模块定义为：
PMAttention(**X**) := softmax((**S**(**XW**_K)^T) / $\sqrt{}$ d_m) (**XW**_V)   (9)
其中 **S**
$$
\in
$$
 ℝ^(d_s
$$
\times
$$
 d_a) 由 d_s 个可学习的 query 向量（或种子）组成。在我们的实验中，d_s = 32。
PMA 模块使用与 Transformer 模块相同的公式（公式 7 和 8）构成，只是将 Attention 替换为 PMAttention。
## 附录 B：语义 ID 的点击分布
语义 ID 空间中的点击分布（图 6）明显比原始 ID 空间中的点击分布更不偏斜。注意，图 2 显示的是累积曝光分布，而图 6 显示的是点击的边际分布。