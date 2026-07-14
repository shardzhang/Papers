# MobileRec: A Large-Scale Dataset for Mobile Apps Recommendation（中文翻译）

> 原文标题：MobileRec: A Large-Scale Dataset for Mobile Apps Recommendation
>
> 原文作者：Umar Farooq, M.H. Maqbool, Adib Mosharrof, A.B. Siddique, Hassan Foroosh
>
> 原文出处：SIGIR'23, July 23-27, 2023, Taipei, Taiwan
>
> 原文链接：https://doi.org/10.1145/nnnnnnn.nnnnnnn

## 摘要

推荐系统已在我们数字生活中无处不在，从电子商务网站上的产品推荐到流媒体平台上的电影和音乐推荐。现有的推荐数据集，如 Amazon Product Reviews 和 MovieLens，极大地推动了各自领域推荐系统的研究和开发。尽管移动用户和移动应用的数量在过去十年中呈指数级增长，但移动应用推荐系统的研究却受到显著限制，主要原因是缺乏高质量的基准数据集——这与产品、电影和新闻推荐形成鲜明对比。为了促进应用推荐系统的研究，我们引入了一个大规模数据集，称为 MobileRec。我们根据用户在 Google Play 商店中的活动构建了 MobileRec。MobileRec 包含 1930 万用户交互（即用户对应用的评价），涵盖 48 个类别中的超过 1 万个独立应用。MobileRec 记录了总计 70 万不同用户的序列化活动。每个用户与不少于 5 个不同的应用进行过交互，这与之前仅记录每个用户单个交互的移动应用数据集形成鲜明对比。此外，MobileRec 提供了用户对已安装应用的评分及情感信息，每个应用还包含丰富的元数据，如应用名称、类别、描述和总体评分等。我们通过对几种最先进的推荐方法进行对比研究，证明 MobileRec 可以作为应用推荐的优秀测试平台。定量结果可作为其他研究人员对比其结果的基准。MobileRec 数据集可在 https://huggingface.co/datasets/recmeapp/mobilerec 获取。

## 关键词

序列推荐，GooglePlay 数据集，应用推荐数据集。

## ACM 引用格式

M.H. Maqbool, Umar Farooq, Adib Mosharrof, A.B. Siddique, and Hassan Foroosh. 2023. MobileRec: A Large-Scale Dataset for Mobile Apps Recommendation. In Proceedings of Proceedings of the 46th ACM SIGIR Conference on Research and Development in Information Retrieval, (SIGIR'23). ACM, New York, NY, USA, 10 pages. https://doi.org/10.1145/nnnnnnn.nnnnnnn

---

## 1 引言

移动应用在过去十年中经历了指数级增长，超过 50 亿用户[8]出于各种原因使用它们，包括社交媒体、娱乐、新闻、生产力和拼车等。由于这一蓬勃发展，Google Play[16]和 Apple App Store[5]分别托管了超过 350 万和 220 万个应用[6]。日益拥挤的应用市场给用户有效发现符合其偏好的应用带来了巨大挑战。个性化应用推荐可以减轻用户的认知负担并改善应用安装体验。如图 1 所示，应用推荐系统能够根据用户先前的应用安装和交互向他们推荐新的应用。尽管 Google Play 和 App Store 采用应用推荐技术向其用户推荐应用（可能利用了内部收集的用户数据），但应用推荐的研究几乎不存在。

**图 1：用户活动序列示例。** 基于过去的用户交互（例如应用安装），应用推荐系统推荐新的应用供安装。

---

## 2 相关工作

### 2.1 应用数据集与推荐

现有几个用于移动应用用户评论的数据集，如表 1 所示。在收集用户评论的早期工作中，Iacob 和 Harrison[20]收集了 161 个移动应用的 3,279 条评论，并分析了用户的功能请求。Khalid 等人[25]和[24]专注于 iOS 应用，准备了 20 个应用的 6,390 条用户评论数据集。McIlroy 等人[32]使用了 12,000 个移动应用的 601,221 条用户评论来研究应用商店中的负面评论。Maalej 和 Nabil[29]收集了一个更大的数据集，包含 1,186 个应用的 130 万条评论。这些数据集侧重于用户投诉和用户-开发者对话理解。此外，这些数据集不公开可用。

**表 1：现有移动应用数据集与 MobileRec 的比较。**

| 数据集特征 | Top 20 Apps [3] | RRGen [15] | AARSynth [13] | Srisopha et al. [38]* | PPrior [14]† | MobileRec |
|---|---|---|---|---|---|---|
| 评论数量 | 200K | 309K | 2.1M | 9.3M | 2.1M | 19.3M |
| 应用数量 | 20 | 58 | 103 | 1,600 | 9,869 | 10,173 |
| 应用类别数量 | 9 | 15 | 23 | 32 | 48 | 48 |
| 单个用户的多个评论 | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| 应用元数据 | ✗ | ✗ | ✓ | ✗ | ✗ | ✓ |
| 评论评分 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 评论文本 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 评论时间戳 | ✗ | ✗ | ✗ | ✓ | ✓ | ✓ |

* [38]不公开可用。† [14]仅包含负面用户评论。

Top 20 Apps[3]可在 Kaggle 上获取，包含 20 个应用跨 9 个类别的 20 万条评论。该数据集提供评分和评论的文本。RRGen[15]包含来自 58 个应用的超过 30.9 万条评论。与 Top 20 Apps 类似，RRGen 仅提供评分和评论文本。这两个数据集均不提供应用元数据、评论时间戳和用户唯一标识符。

AARSynth[13]提供超过 200 万条用户评论，涵盖一百多个应用，包括应用元数据。该数据集的评论也缺少与上述数据集类似的关键信息。Srisopha 等人[38]的数据集包含来自 1,600 个应用的超过 900 万条用户评论。该数据集有评论时间戳，有助于在时间周期上下文中理解评论。然而，该数据集不包括用户的唯一标识符和应用元数据。请注意，Srisopha 等人[38]的数据集不公开可用。

最近，PPrior[14]数据集提供了来自 Google Play 的超过 9,000 个应用跨 48 个类别的 200 多万条评论。该数据集提供评分、评论文本和评论时间戳。然而，交互（即评论）上的用户标识符和应用元数据未提供。此外，PPrior 仅提供负面用户评论（即评分仅为 1 和 2 的评论）。因此，PPrior 不适合构建鲁棒的应用推荐系统。在这项工作中，我们提供了一个比上述所有移动应用数据集都更大的数据集，称为 MobileRec，包括每个用户的唯一标识符和交互时间戳。此外，每个用户与应用至少有 5 次交互，所有包含的应用至少有 15 次交互，这使其成为移动推荐系统的理想测试平台。

### 2.2 现有推荐数据集

推荐数据集涵盖了广泛多样的领域，从亚马逊等电子商务平台到以电影为重点的娱乐行业，甚至延伸到在线游戏平台等。这些多样化的数据集为研究人员提供了丰富的机会，以研究和改进推荐系统在广泛用例中的有效性。

Amazon Product Reviews[18, 31, 33]是一个大规模数据集，在 2018 年更新版本中包含 2.331 亿条评论。Amazon Product Reviews 数据集中有 29 个类别。请注意，该数据集于 2014 年发布的早期版本有 1.428 亿次交互和 24 个类别。该数据集提供评论文本、评分和有用性投票，以及描述、类别信息、价格和品牌等产品元数据。Amazon Product Reviews 数据集包含多个类别，包括 All Beauty、Books、CDs and Vinyl 等。每个类别包含数千到数百万条评论。例如，All Beauty 有 37 万条评论和 32K 个独特产品，Books 有 5100 万条评论和 200 万个产品，CDs and Vinyl 包含 100 万条评论和 544K 个产品。5-core 是 Amazon Product Reviews 数据集的小版本，包含至少 5 次用户交互，这使其对构建鲁棒推荐系统很有用。由于 Amazon Product Reviews 是一个非常大的数据集，现有的一些工作[7, 23]仅使用该数据集中的部分类别（如 Books、Beauty、CDs 和 Games）来基准测试其提出的方法。

Yelp[4]数据集从 2018 年到 2022 年有多个版本。最新版本 Yelp2022 有 190 万唯一用户、15 万个不同项目和 690 万条评论。类似地，MovieLens[1]提供 ML-25M[17]，包含 2500 万个评分和 100 万个标签应用于 62K 部电影。该数据集的先前版本包括 ML-100k 和 ML-1M。

除上述大规模数据集外，来自不同领域的各种数据集在推动各自领域的研究中发挥了重要作用。下面我们简要讨论其他推荐数据集。Steam 包含从 Steam 爬取的评论和游戏信息[23]。Book-Crossing[50]和 GoodReads[44, 45]提供用户-项目交互，其中交互类型为评分。DIGINETICA[49]包含 20 万用户、184K 项目和约 100 万次交互，其中交互类型为用户点击。该数据集是从电子商务搜索引擎日志中提取的用户会话编译而成。Twitch[36]提供 1500 万用户在 600 万个项目上的 4.74 亿次交互，交互类型为用户点击。这些数据集使得各自领域的多个推荐系统得以发展[10, 18, 21, 31, 33, 34, 37, 42, 48]。我们期望 MobileRec 发挥类似作用，激发构建鲁棒应用推荐系统的研究。

---

## 3 MOBILEREC 数据集

### 3.1 数据集构建

Google Play[16]是安卓用户安装应用并表达对应用意见的默认应用商店。因此，它通过用户评论托管了海量的用户交互。用户评论在滚动网页时动态加载。因此，传统的网络爬取工具无法支持大规模用户评论数据的下载。为了自动化页面滚动并执行其他点击事件（例如，较长的评论默认不会完全显示），我们使用了 Selenium WebDriver[2]。

为了获取数据，第一步是收集所有可直接从 Google Play 访问的应用的包名（即唯一标识符），方法是浏览所有应用类别页面以及热门排行榜。然后，我们使用这些信息下载应用元数据。应用的元数据包括应用名称、开发者、类别和文本描述等详细信息。通过访问应用的评论部分、递归滚动页面并提取评论文本以及用户的评分、用户信息（稍后匿名化）和时间戳，提取与每个应用关联的用户评论，直到该应用不再有更多评论可用。表 3 展示了数据集的重要特征。我们使用了多种过滤和排序机制（例如，最新、评分）以尽可能下载最多的用户评论。

为确保数据质量，我们设置了多项检查以消除重复条目、错误信息以及在爬取过程中可能发生的其他错误。我们在进一步处理数据以转换为 5-core 之前移除了数据中的重复项。我们还通过引入 16 字符字母数字 uid 对用户进行了匿名化。我们只保留了那些有超过 5 条评论的用户。评论少于 15 条的项目从最终数据集中移除。最终结果是一个大规模 5-core 的用户评论数据集（即带时间戳的交互），用于 Google Play 上的应用，提供了关于用户对各种应用的意见和体验的宝贵见解。表 1 和表 2 提供了关于数据集的重要统计信息。

### 3.2 数据集特征

我们在数据集中引入了各种特征（也在表 3 中展示），这些特征将有助于讨论数据集趋势。uid 是用户的唯一 16 字符字母数字 ID，也用于对用户进行匿名化，因为 MobileRec 出于隐私原因不提供实际用户 ID。review 是用户的实际评论文本。app_package 是应用的唯一安卓包名。rating 表示用户对某个应用（由 app_package 标识）的实际评分。formated_date 用于按时间顺序对用户交互进行排序。更多详情请参考表 3。

**表 3：MobileRec 数据集中各种重要特征的描述。**

| 特征 | 描述 |
|---|---|
| uid | 16 字符字母数字 uid 唯一标识一个用户。同时使用户匿名化。示例：Aj0Sm6myfh6YN3Rn, 3pZUhksFIcLjEXtl, dvx0dqXTKtHUmY3O |
| app_package | 表示应用的安卓包名，唯一标识一个应用。示例：com.google.android.calculator, com.king.crash, org.wikipedia |
| app_name | 应用在 Google Play 上显示的标题。示例：Candy Crush Saga, MONOPOLY - Classic Board Game |
| app_category | 应用的类别。示例：娱乐、金融、生产力 |
| review | 用户对应用给出的文本评论 |
| rating | 用户对应用给出的数字评分。示例：5, 2, 1, 4 |
| votes | 认为此评论有用的用户数量。示例：1, ..., 6, ... |
| date | 特定用户/项目交互的日期，即评论日期。示例：October 21, 2018, November 4, 2021, January 16, 2021 |
| formated_date | 特定用户/项目交互的日期（YYYY-MM-DD 格式的评论日期）。示例：2018-10-21, 2021-11-04, 2021-01-16 |
| unix_timestamp | 转换为 Unix 时间戳的评论日期。示例：1.540094e+09, 1.547788e+09, 1.610773e+09 |

### 3.3 数据集分析

在本节中，我们展示 MobileRec 数据集中的各种趋势，例如用户如何从一类应用迁移到另一类（基于其评论交互），用户行为如何随应用定价演变，用户如何根据应用的内容分级（即 Mature 17+、Teen、Everyone 10+ 和 Everyone）在应用间迁移。

我们在图 2(a)、2(b)、2(c)和 2(d)中捕捉了用户动态行为的交互级快照。我们考虑前两次用户交互来捕捉迁移行为。例如，在图 2(a)中，可以注意到在给定时间步 t 与 Mature17+ 类别交互的大多数用户，最终在时间步 t+1 的交互中转向了 Everyone 和 Teen 内容分级的应用。类似地，图 2(b)揭示了一个预期的趋势，展示了基于价格的序列化用户交互。图 2(b)指出，在时间步 t 与免费应用交互的用户中，极少数会在时间步 t+1 与付费应用交互。相反，许多与付费应用交互的用户迅速迁移到免费应用。此外，从图 2(c)可以看出，与高评分应用交互的用户大多不会在应用质量上妥协（即应用的整体平均评分），这表现为这些用户向低评分应用的迁移率非常低。同样明显的是，喜欢高评分应用的用户可能会迁移到评分在 3.5-4.5 之间的应用。在图 2(d)中可以观察到非常动态的用户迁移模式。用户在 top-5 类别之间非常动态地迁移。这种高度波动的模式表明了在推荐系统中建模用户行为的复杂性，尤其是在冷启动场景中，这种动态行为变得非常具有挑战性。需要注意的是，MobileRec 有 48 个类别，而在图 2(d)中，我们仅展示了按评论数量排名的 top-5 类别。如果考虑所有类别，这种动态波动的迁移模式可能更加复杂。

**图 2：用户序列化交互趋势，关于内容分级、价格层级、评分以及 top-5 类别内。**

图 3 总结了 top-10 类别中的评论数量。可以观察到，Action 类别在 MobileRec 数据集的所有 48 个类别中评论数量最多。同样有趣的是，top-10 类别大约占所有评论的 50%，而其余类别占另外 50%。

**图 3：按评论数量排序的 Top-10 类别。刻度盘周围的数字表示百分比，例如，属于 Action 类别的应用约占 MobileRec 数据集中总评论的 7%。**

图 4 总结了 top-5 类别中的评论长度：Action、Role Playing、Simulation、Strategy 和 Puzzle。我们在图 4 的箱线图中关闭了须线以获得更好的可读性，但我们提供了每个类别中异常值的一些统计信息。例如，在 Action 类别中，共有 1,373,484 条评论，其中 10,357 条评论超过 100 词。因此，异常值约占 Action 类别总评论的 0.754%。类似地，在 Role Playing 类别中，共有 1,321,861 条评论，其中 12,187 条超过 100 词，产生 0.922% 的异常值。Simulation 类别有 1,023,366 条评论，其中 8,056 条超过 100 词，占该类总评论的 0.787%。Strategy 类别有 0.771% 的评论超过 100 词，在总共 963,360 条评论中有 7,430 条属于此类。最后，Puzzle 类别共有 954,909 条评论，其中 4,832 条超过 100 词，仅占该类总评论的 0.506%。从图 4 可以看出，在 top-5 类别中，大多数评论长度少于 100 词。

**图 4：基于评论数量的 Top-5 类别评论长度。**

### 3.4 其他使用场景

除了构建推荐系统之外，这个大规模数据集还可用于多种目的。从应用开发和优化到市场研究和欺诈检测，该数据提供的洞察可以帮助企业和研究人员做出明智决策并制定更有效的策略。在这项工作中，我们专注于为广泛的推荐系统提供基线结果，并将该数据集的其他用途留给研究社区。

与任何数据集一样，该数据集存在潜在的不良使用情况。因此，数据不包含任何关于用户的信息。具体来说，我们对数据进行了去标识化和匿名化处理以防止个人识别。为完整性起见，我们认为强调可能出现的不良用途很重要。例如，公司可能付费购买虚假评论或评分以人为提升其应用的地位，或者他们可能操纵应用元数据，使其应用被某个推荐系统更频繁地推荐。类似地，攻击者可能使用数据来识别更可能被推荐的应用，然后创建包含恶意代码的虚假版本。

总的来说，我们相信这个数据集的益处大于潜在的危害。例如，该数据集有潜力为个人和组织提供许多有用的见解和益处。具体来说，我们期望该数据集能激发应用推荐系统的研究和发展，以及其他良好用途。

---

## 4 基线方法

我们考虑广泛的推荐系统进行基准测试，并为 MobileRec 数据集建立基线结果。

### 4.1 通用推荐

**Pop.** 这是一个简单的基于流行度的模型。在此模型中，记录项目的流行度，并将最流行的项目推荐给用户。例如，在 YouTube 上使用此模型，将向用户推荐观看次数最多的视频。

### 4.2 序列推荐

**SASRec [23].** 马尔可夫链假设用户的下一个动作可以由他们最近的动作（仅一个或几个最近的动作）决定。另一方面，基于 RNN 的方法可以考虑长期用户-项目交互历史来发现用户的隐藏兴趣。SASRec 通过采用注意力机制，将基于马尔可夫链（MC）的方法和基于循环神经网络（RNN）的方法的优点结合在一个统一的设计中。基于 MC 的方法在稀疏数据集的情况下有效，而基于 RNN 的方法适用于涉及更密集数据集的情况。SASRec 提出平衡基于 MC 的方法和基于 RNN 的方法的威力，以获得两全其美的效果。

**ELECRec [7].** 下一个项目预测任务通常被建模为生成式任务。ELECRec 提出在推荐系统中使用判别器进行下一个项目预测任务。判别器负责判断下一个采样项目是否为真实的目标项目。生成器 G 和判别器 D 联合训练。生成器 G 被训练为判别器生成高质量的样本。

**BERT4Rec [39].** 基于序列神经网络是次优且有限的这一观察。这种次优性可归因于对用户行为的限制性单向（从左到右）编码，以及他们假设用户交互序列是有序的。BERT4Rec 是一种序列推荐模型，它采用双向自注意力来编码用户的交互序列。该基线将完形填空目标应用于序列推荐，并通过双向上下文联合条件来预测被掩码的项目。

**HGN [28].** 考虑到近期时间顺序用户-项目交互的重要性，层次门控网络提出了一种通过集成贝叶斯个性化排序（BPR）来捕捉长期和短期用户兴趣的方法。所提出的方法包括特征门控模块、项目门控模块和项目-项目乘积模块。门控模块负责决定哪些特征从特征层和实例层转发到下游层。项目-项目乘积模块捕捉用户过去交互过的项目和将来将交互的项目之间的项目关系。

**SINE [40].** 认识到用户行为序列中多个概念上不同项目的重要性，SINE 提出捕捉用户行为序列的统一嵌入主要受最近用户交互的影响。基于这一观察，SINE 提出使用多个嵌入来捕捉用户交互行为的各个方面。由于概念池可能很大，SINE 能够从大型概念池中推断出稀疏的概念集。拥有多个嵌入后，采用兴趣聚合模块来预测用户当前的意图，这也用于建模下一个项目预测。

### 4.3 基于 Transformer 的推荐

**LightSANs [12].** 自注意力网络（SAN）由于二次复杂度和对过参数化的脆弱性而受限。序列关系中项目间的建模不准确性也是 SAN 的一个限制因素，因为隐式位置编码。为此，LightSAN 提出了一种低秩分解的自注意力网络。低秩分解的自注意力将用户的历史项目投影到几个潜在兴趣上。项目到兴趣的交互用于生成上下文感知的表示。

### 4.4 基于会话的推荐

**GRU4Rec [41].** 该基线使用基于 RNN 的方法进行基于会话的推荐。在所提出的技术中采用了数据增强和处理输入数据分布偏移的方法。

### 4.5 基于 GNN 的推荐

**GCSAN [47].** 该方法提出了一种图上下文化的自注意力模型，同时使用图神经网络和自注意力网络。在 GCSAN 中，利用图神经网络捕捉丰富的局部依赖关系，而使用自注意力网络捕捉长距离依赖关系。每个会话表示为全局偏好和会话当前兴趣的组合。

---

## 5 实验设置

### 5.1 实验配置

对于序列推荐基线，我们保持一致批处理大小为 4096，最大交互长度为 50。我们采用留一法进行验证和测试，并使用完整项目集进行评估。我们使用早停 patience 为 10。我们将序列推荐任务视为多分类任务，并使用交叉熵损失来训练模型。SASRec[23]使用 adam 优化器和 0.001 的学习率训练。层数和注意力头数为 2。dropout 率为 0.5，使用 gelu 作为激活函数。HGN[28]和 SINE[40]使用嵌入大小 64、学习率 0.001 和 adam 优化器训练。LightSANs[12]的潜在兴趣维度为 5，2 个注意力头数和 2 个 Transformer 层。训练使用学习率 0.001 和 adam 优化器。GCSAN[47]有 2 个 Transformer 编码器层、2 个注意力头数、隐藏状态特征大小为 64、前馈层隐藏大小为 256、权重设为 0.6、图神经网络层数为 1，使用 adam 优化，学习率为 0.001。GRU4Rec[41]使用 1 层训练，嵌入大小为 64、隐藏大小为 128、dropout 为 0.3。我们使用 RecBole[49]为 MobileRec 建立基线。

### 5.2 评估指标

我们采用标准指标 Hit@K 和 NDCG@K，其中 K ∈ {1, 5, 10, 15, 20}，作为基准方法的评估指标。Hit@k 考虑预测项目出现在 top-K 列表中的次数，可表示为[27]中的形式：

HR@K = (1/M) * Σ_{u=1}^{M} WR, where WR = Σ_{R=1}^{N} WR · 1{R_u ≤ K} · 1{R_u = R} · (1/M) · Σ_{u=1}^{M} 1X

其中 1X 表示示性随机变量，M 是用户总数，N = |I| 是总项目数。R 是项目在范围 [1, N] 内的整数排名位置。R_u 是项目 i_u 在 I 个项目中对用户 u 的排名。WR 捕捉在位置 R 上拥有项目 i_u 的用户。

NDCG@K 可表示为[22, 49]：

(1/|U|) * Σ_{u∈U} (Σ_{i=1}^{K} δ(i ∈ R(u)) / log2(i+1)) / (Σ_{i=1}^{min(|R(u)|,K)} 1/log2(i+1))

所有项目集均用于预测排名。

**表 4：MobileRec 上各种基线的性能分析。基线属于不同类别，例如通用基线（如 Pop）、序列基线（如 SASRec、ELECRec、BERT4Rec、HGN、SINE）、基于会话的基线（如 GRU4Rec、GCSAN）、基于图神经网络的基线（如 GCSAN）、基于 Transformer 的基线（如 LightSANs）。**

| 方法 ↓ / 指标 → | Hit@1 | Hit@5 | Hit@10 | Hit@15 | Hit@20 | NDCG@5 | NDCG@10 | NDCG@15 | NDCG@20 |
|---|---|---|---|---|---|---|---|---|---|
| Pop | 0.0027 | 0.0086 | 0.0151 | 0.0208 | 0.0256 | 0.0056 | 0.0077 | 0.0092 | 0.0103 |
| SASRec | 0.0026 | 0.0098 | 0.0181 | 0.0242 | 0.0295 | 0.0061 | 0.0088 | 0.0104 | 0.0117 |
| ElecRec | 0.0020 | 0.0094 | 0.0174 | 0.0237 | 0.0293 | 0.0056 | 0.0082 | 0.0098 | 0.0112 |
| Bert4Rec | 0.0024 | 0.0083 | 0.014 | 0.0183 | 0.0221 | 0.0054 | 0.0072 | 0.0083 | 0.0092 |
| HGN | 0.0012 | 0.0054 | 0.0096 | 0.0132 | 0.0165 | 0.0033 | 0.0046 | 0.0056 | 0.0064 |
| SINE | 0.0022 | 0.0087 | 0.0163 | 0.0228 | 0.028 | 0.0054 | 0.0078 | 0.0095 | 0.0107 |
| LightSANs | 0.0024 | 0.0102 | 0.0172 | 0.0227 | 0.028 | 0.0062 | 0.0085 | 0.0099 | 0.0112 |
| GRU4Rec | 0.0021 | 0.0086 | 0.0153 | 0.021 | 0.0261 | 0.0053 | 0.0074 | 0.0089 | 0.0102 |
| GCSAN | 0.0024 | 0.0094 | 0.0161 | 0.0214 | 0.0266 | 0.0059 | 0.0081 | 0.0095 | 0.0107 |

---

## 6 结果与讨论

表 4 展示了各种基线在 MobileRec 上的性能分析。接下来，我们讨论这些结果并提供更多细节。如前所述，Pop 是一个基于流行度的模型，依赖于项目的流行度。Pop 是最朴素的模型，正如预期，其性能几乎逊于所有其他基线。SASRec 之前报告过根据 Hit@10 和 NDCG@10 性能指标，其结果优于 Pop。在 SASRec 中，随机采样了 100 个负样本项目，并针对这 101 个项目（包括 ground truth）计算 Hit@10 和 NDCG@10。使用 MobileRec，我们采用完整项目集进行评估，这是推荐系统更严格的评估标准。我们观察到，在所有性能指标上，SASRec 均优于 Pop，除了 Hit@1。例如，与 Pop 相比，SASRec 的 Hit@10 提升了 19.86%。同样，Hit@15 观察到了 16.34% 的性能提升。类似地，在 MobileRec 上，SASRec 相比 Pop 的 Hit@20 提升了 15.23%。

考虑 NDCG 指标，SASRec 在 Beauty 数据集上评估时，相比 Pop 的 NDCG@10 提升了 41.37%。在 MobileRec 上，我们观察到相比 Pop，NDCG@10 仅提升了 14.28%。我们认为，SASRec 和 Pop 在这两个数据集（Beauty 和 MobileRec）上 NDCG@10 提升的差异可以通过比较 Beauty 和 MobileRec 的数据集统计信息来解释。Beauty 有 52,024 个用户，而 MobileRec 有 700,111 个用户。其次，Beauty 数据集有 57,289 个项目，而 MobileRec 有 10,173 个应用。Beauty 有 40 万次交互，平均每个用户 7.6 次交互，每个项目 6.9 次交互。而 MobileRec 有 1930 万次交互，平均每个用户 27.56 次交互，每个项目（即应用）1,896.88 次交互。例如，MobileRec 每个用户有 27.56 次交互，是 Beauty 每用户交互次数的 200 倍以上。同样，MobileRec 每个项目的平均交互次数也远多于 Beauty。考虑到这些因素，MobileRec 呈现了一个更加动态的推荐场景。MobileRec 中用户和项目交互的高程度可能是 SASRec 相比 Pop 在 NDCG@10 上改进相对较小的原因。其次，由于我们使用完整项目集进行评估，而 SASRec 使用排名策略，这也可能是 SASRec 相比 Pop 性能增益较小的潜在原因。在 Beauty 数据集上，SASRec 报告了比 GRU4Rec 更好的结果。我们在 MobileRec 上训练和评估时，也观察到 SASRec 优于 GRU4Rec 的相同模式。

在 Beauty 数据集上，ELECRec 报告的 Hit@5、Hit@10、NDCG@5 和 NDCG@10 优于 Pop。我们在 MobileRec 上训练和评估时也注意到相同的模式，唯一的例外是 Hit@1，其中 Pop 表现优于 ELECRec。我们推测非常动态的用户-项目交互历史可能是原因。类似地，在 Beauty 数据集上，ELECRec 相比 GRU4Rec 报告的 Hit@5 提升了 329.87%，Hit@10 提升了 242.04%。此外，对于 NDCG@5 和 NDCG@10 指标，ELECRec 相比 GRU4Rec 分别达到了 412.12% 和 331.38% 的提升。着眼于在 MobileRec 上训练和评估时 ELECRec 相对于 GRU4Rec 的性能提升，ELECRec 在 Hit@5 和 Hit@10 上分别实现了 9.30% 和 13.72% 的提升。类似地，在 NDCG@5 和 NDCG@10 上分别观察到 1.88% 和 10.81% 的提升。这一观察也指向了 MobileRec 中高度动态的用户-项目交互。两种模型都难以实现高性能，这导致两种方法之间的性能差距缩小，相比于 ELECRec 中报告的性能差距。MobileRec 中有超过 1900 万次交互，而 Beauty 中只有 40 万次用户-项目交互。用户-项目交互的这种高度动态性将模型推向较低的性能增益。考虑到这一分析，MobileRec 应成为现有推荐数据集的重要补充。

ELECRec 在 Beauty 数据集上也展示了相对于 SASRec 的比较增益。首先，我们将介绍 ELECRec 在 Beauty 数据集上相对于 SASRec 在 Hit@5、Hit@10、NDCG@5 和 NDCG@10 方面的相对性能增益。之后，我们将讨论当训练和评估数据集为 MobileRec 时，ELECRec 相对于 SASRec 实现的性能改进。从 Hit@5 和 Hit@10 开始，ELECRec 相比 SASRec 分别提升了 83.59% 和 59.47%。类似地，ELECRec 相比 SASRec 在 NDCG@5 和 NDCG@10 方面分别报告了 103.61% 和 84.11% 的提升。现在，让我们讨论在 MobileRec 上训练和评估时 ELECRec 和 SASRec 之间的性能比较。可以观察到，SASRec 在 MobileRec 上的 Hit@5、Hit@10、NDCG@5 和 NDCG@10 指标上表现优于 ELECRec。SASRec 相对于 ELECRec 在 Hit@5、Hit@10、NDCG@5 和 NDCG@10 方面的性能改进量化分别为 4.25%、4.02%、8.92% 和 7.31%。我们认为 ELECRec 使用了一个用 NLP 任务训练的生成器，而判别器负责判断项目是否是序列中正确的下一个（真实）项目或虚假的下一个项目。判别器的判别能力取决于生成器生成的样本质量。由于 MobileRec 呈现出高度动态的用户-项目交互序列，且用户兴趣转瞬即逝，生成器可能难以生成高质量的训练样本。由于判别器捕捉真实项目相关性的能力取决于生成器生成的样本质量，低质量样本可能导致判别器部分性能增益下降。

值得回顾图 2，它提供了动态用户-项目交互序列的有趣快照。例如，从图 2(b)可以看出，相当大比例的用户从付费应用迁移到免费应用。类似的动态迁移模式在图 2(d)中也很明显，该图描绘了用户在 top-5 类别中的迁移。我们认为，具有如此动态时间交互模式的 1900 万用户-项目交互可能对生成器学习高质量样本生成构成挑战。此外，ELECRec 报告了其与 BERT4Rec 的比较性能。在原始论文中，ELECRec 在 Beauty 数据集的所有指标上都优于 BERT4Rec[7]。具体而言，ELECRec 在 Hit@5 和 Hit@10 上相比 BERT4Rec 分别报告了 100.85% 和 61.06% 的性能增益，在 NDCG@5 和 NDCG@10 指标上分别报告了 131.50% 和 97% 的增益。我们观察到当使用 MobileRec 进行训练和评估时，ELECRec 在大多数评估指标上优于 BERT4Rec 的相同模式，除了 Hit@1，BERT4Rec 优于 ELECRec。考虑到 Hit@1 是更严格的标准，我们认为 BERT4Rec 采用的通过双向上下文联合条件预测序列中被掩码项目的训练目标，对于像 Hit@1 这样更严格的评估指标可能是更好的训练目标。尤其是在像 MobileRec 这样用户兴趣转瞬即逝的背景下，使用双向上下文联合条件的掩码项目预测目标似乎在编码用户动态行为方面有效。

图上下文化自注意力模型 GCSAN 同时使用图神经网络和自注意力网络进行基于会话的推荐。在原始论文中，报告了 GCSAN 与竞争方法之间的多项比较评估。我们主要关注 GRU4Rec 和 Pop 与 GCSAN 的比较表示学习能力的量化，因为我们的基线中也包括了 Pop 和 GRU4Rec 以及 GCSAN。在原始论文中，使用 Amazon-Books 数据集，GCSAN 报告了在 NDCG@5 和 NDCG@10 上优于 Pop 和 GRU4Rec 的结果。类似地，在不同基准数据集上，Hit@5 和 Hit@10 也报告了优于 Pop 和 GRU4Rec 的结果。我们在 MobileRec 上观察到了 GCSAN 在 NDCG@5、NDCG@10、Hit@5 和 Hit@10 上持续优于 Pop 和 GRU4Rec 的类似模式。我们注意到 GCSAN 在 NDCG@5 和 NDCG@10 上相比 Pop 分别提升了 9.30% 和 6.62%。类似地，GCSAN 相比 GRU4Rec 在 Hit@5、Hit@10、NDCG@5 和 NDCG@10 方面也观察到了改进。

LightSANs 是一种用于下一个项目推荐任务的 Transformer 变体，它采用低秩分解的自注意力将用户的历史兴趣投影到几个潜在兴趣上。LightSANs 在多个竞争基线上报告了 Hit@10 和 NDCG@10 结果。我们将重点放在 Pop、GRU4Rec、BERT4Rec 和 SASRec 上，以比较和分析 LightSANs 在 Amazon-Books 数据集和 MobileRec 上的性能。在进一步深入比较分析 LightSANs 在 Amazon-Beauty 与 MobileRec 上的性能之前，让我们先看 Amazon-Books 数据集。Amazon-Books 数据集有 19K 用户、60K 项目和 170 万次交互。在 Amazon-Books 数据集上，LightSANs 相比 Pop 在 Hit@10 和 NDCG@10 上分别报告了 121.77% 和 172.43% 的提升。类似地，相比 SASRec，在 Hit@10 和 NDCG@10 上分别报告了 3.91% 和 2.65% 的提升。相比 GRU4Rec，在 Hit@10 和 NDCG@10 上分别报告了 8.41% 和 5.72% 的提升。最后，相比 BERT4Rec，报告了 8.76% 和 4.03% 的提升。当数据集为 MobileRec 时，我们观察到相比 Pop，在 Hit@10 和 NDCG@10 上分别有 13.90% 和 10.38% 的提升。LightSANs 相比 GRU4Rec 在同一指标上分别获得了 12.41% 和 14.86% 的提升。

与 ELECRec 类似，LightSANs 相比 BERT4Rec 在 Hit@10 和 NDCG@10 指标上也展示了 22.85% 和 18.05% 的更好结果。我们注意到 LightSANs 在 MobileRec 上保持了优于竞争基线的模式，就像在 Amazon-Books 数据集上一样，但 SASRec 是一个例外，它在 Hit@10 和 NDCG@10 指标上优于 LightSANs。LightSANs 相对于 BERT4Rec、Pop 和 GRU4Rec 表现出的稳定性能可归因于 LightSANs 中更好的设计选择。例如，LightSANs 提出了解耦位置编码来代替隐式位置编码。我们相信这种解耦位置编码有助于 LightSANs 更精确地建模历史用户兴趣和用户-项目交互，从而引导模型在 MobileRec 上获得更好的 Hit@10 和 NDCG@10。尽管如此，LightSANs 被 SASRec 超越。SASRec 的优越性能可能归因于 SASRec 引入的可学习位置嵌入。

SINE 研究了用多个嵌入向量编码用户兴趣的想法，基于其经验发现：用户的行为序列表现出多个不同的兴趣。SINE 使用多个数据集来基准测试其结果。我们考虑 SINE 在 Amazon Product Review 数据集上使用 Hit@10 和 NDCG@10 指标报告的性能。我们将比较分析限制在 SASRec 和 GRU4Rec 上。SINE 报告了在 Hit@50、Hit@100、NDCG@50 和 NDCG@100 上优于 SASRec 的结果。在 Hit@50、Hit@100、NDCG@50 和 NDCG@100 上也报告了优于 GRU4Rec 的性能。在 MobileRec 上训练和评估时，SINE 在 Hit@20 和 NDCG@20 上分别以 7.27% 和 4.90% 优于 GRU4Rec；但在 Hit@20 和 NDCG@20 上难以超越 SASRec。首先，这可能是因为 Hit@20 和 NDCG@20 是比 SINE 选择的 Hit@50、Hit@100、NDCG@50 和 NDCG@100 更严格的指标。其次，考虑到 SINE 致力于从用户的交互历史中捕捉用户的不同兴趣，我们认为在 MobileRec 中收敛到不同的用户兴趣存在挑战，因为用户兴趣波动较大，如图 2 所示。这种从交互序列中学习和嵌入不同用户兴趣的能力不足可能导致 SINE 表现不如 SASRec。

---

## 7 结论

在本文中，我们介绍了 MobileRec，一个大规模序列化用户-应用交互数据集。MobileRec 的独特之处在于它捕捉了每个用户的多个交互，提供了用户行为的更全面视图。MobileRec 总共包含 1930 万次用户-应用交互，涵盖来自 48 个类别的超过 1 万个独立应用，涉及 70 万唯一用户（每个用户至少有 5 次不同的应用交互），为理解用户在不同应用类别中的参与度提供了前所未有的粒度。此外，MobileRec 中的每次用户-应用交互都包含丰富的上下文信息，例如用户评分、评论文本和评论日期。最后但同样重要的是，MobileRec 中的每个应用都包含广泛的元数据，例如应用名称、类别、长文本描述、总体平均评分、开发者信息和内容分级等。我们还通过对各种最先进推荐技术进行对比评估，展示了 MobileRec 数据集作为应用推荐研究实验测试平台的有用性。该评估还建立了将使研究社区受益的基线结果。我们希望我们的数据集能激发进一步的研究，开启新的见解，并为未来的移动应用推荐系统铺平道路。

---

## 附录：表 2——MobileRec 与不同领域知名推荐数据集最新版本的比较

我们使用了 2018 年的 Amazon Reviews 数据集、Yelp 2022 版本和 2019 年 12 月最新发布的 ML-25M 来生成这些统计信息。

| 指标 | Amazon Reviews | Yelp | ML-25M | MobileRec（本文） |
|---|---|---|---|---|
| 总交互数（百万） | 233.1 | 6.99 | 25.0 | 19.3 |
| 至少 5 次交互的用户（百万） | 10.6 | 0.29 | 0.16 | 0.70 |
| 至少 15 次交互的项目（千） | 2072 | 77.58 | 20.59 | 10.17 |
| 单个用户的最大交互数 | 446 | 3048 | 32202 | 256 |
| 单个用户的最小交互数 | 1 | 1 | 20 | 5 |
| 每用户平均交互数 | 5.32 | 3.52 | 153.81 | 27.56 |
| 单个项目的最大交互数 | 13,560 | 7,673 | 81,491 | 14,345 |
| 单个项目的最小交互数 | 1 | 5 | 1 | 20 |
| 每项目平均交互数 | 15.45 | 46.49 | 423.39 | 1896.88 |

## 参考文献

[1] Movielens. https://grouplens.org/datasets/movielens/, 2022. Accessed: 2022-11-06.

[2] Selenium webdrive. https://www.selenium.dev/documentation/webdriver/, 2022. Accessed: 2023-18-02.

[3] Top 20 play store app reviews. https://www.kaggle.com/datasets/odins0n/top-20-play-store-app-reviews-daily-update, 2022. Accessed: 2022-12-09.

[4] Yelp open dataset. https://www.yelp.com/dataset, 2022. Accessed: 2023-18-02.

[5] Apple. Apple app store. https://apps.apple.com/, 2022. Accessed: 2022-11-06.

[6] L. Ceci. Number of apps available in leading app store. https://www.statista.com/statistics/276623/number-of-apps-available-in-leading-app-stores/. Accessed: 2022-11-06.

[7] Yongjun Chen, Jia Li, and Caiming Xiong. Elecrec: Training sequential recommenders as discriminators. arXiv preprint arXiv:2204.02011, 2022.

[8] J. Degenhard. Number of apps available in leading app store. https://www.statista.com/forecasts/1143723/smartphone-users-in-the-world. Accessed: 2022-02-02.

[9] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018.

[10] Qiming Diao, Minghui Qiu, Chao-Yuan Wu, Alexander J Smola, Jing Jiang, and Chong Wang. Jointly modeling aspects, ratings and sentiments for movie recommendation (jmars). In Proceedings of the 20th ACM SIGKDD international conference on Knowledge discovery and data mining, pages 193–202, 2014.

[11] Jeffrey L Elman. Finding structure in time. Cognitive science, 14(2):179–211, 1990.

[12] Xinyan Fan, Zheng Liu, Jianxun Lian, Wayne Xin Zhao, Xing Xie, and Ji-Rong Wen. Lighter and better: low-rank decomposed self-attention networks for next-item recommendation. In Proceedings of the 44th international ACM SIGIR conference on research and development in information retrieval, pages 1733–1737, 2021.

[13] Umar Farooq, AB Siddique, Fuad Jamour, Zhijia Zhao, and Vagelis Hristidis. App-aware response synthesis for user reviews. In 2020 IEEE International Conference on Big Data (Big Data), pages 699–708. IEEE, 2020.

[14] Moghis Fereidouni, Adib Mosharrof, Umar Farooq, and AB Siddique. Proactive prioritization of app issues via contrastive learning. In 2022 IEEE International Conference on Big Data (Big Data), pages 535–544. IEEE, 2022.

[15] Cuiyun Gao, Jichuan Zeng, Xin Xia, David Lo, Michael R Lyu, and Irwin King. Automating app review response generation. In 2019 34th IEEE/ACM International Conference on Automated Software Engineering (ASE), pages 163–175. IEEE, 2019.

[16] Google. Google play store. https://play.google.com/store/apps, 2022. Accessed: 2022-11-06.

[17] F Maxwell Harper and Joseph A Konstan. The movielens datasets: History and context. Acm transactions on interactive intelligent systems (tiis), 5(4):1–19, 2015.

[18] Ruining He and Julian McAuley. Ups and downs: Modeling the visual evolution of fashion trends with one-class collaborative filtering. In proceedings of the 25th international conference on world wide web, pages 507–517, 2016.

[19] Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. Neural computation, 9(8):1735–1780, 1997.

[20] Claudia Iacob and Rachel Harrison. Retrieving and analyzing mobile apps feature requests from online reviews. In 2013 10th working conference on mining software repositories (MSR), pages 41–44. IEEE, 2013.

[21] Tomoharu Iwata, Shinji Watanabe, and Hiroshi Sawada. Fashion coordinates recommender system using photographs from fashion magazines. In Twenty-Second International Joint Conference on Artificial Intelligence, 2011.

[22] Kalervo Järvelin and Jaana Kekäläinen. Cumulated gain-based evaluation of ir techniques. ACM Transactions on Information Systems (TOIS), 20(4):422–446, 2002.

[23] Wang-Cheng Kang and Julian McAuley. Self-attentive sequential recommendation. In 2018 IEEE international conference on data mining (ICDM), pages 197–206. IEEE, 2018.

[24] Hammad Khalid. On identifying user complaints of ios apps. In 2013 35th international conference on software engineering (ICSE), pages 1474–1476. IEEE, 2013.

[25] Hammad Khalid, Emad Shihab, Meiyappan Nagappan, and Ahmed E Hassan. What do mobile app users complain about? IEEE software, 32(3):70–77, 2014.

[26] Yehuda Koren. Factorization meets the neighborhood: a multifaceted collaborative filtering model. In Proceedings of the 14th ACM SIGKDD international conference on Knowledge discovery and data mining, pages 426–434, 2008.

[27] Dong Li, Ruoming Jin, Jing Gao, and Zhi Liu. On sampling top-k recommendation evaluation. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pages 2114–2124, 2020.

[28] Chen Ma, Peng Kang, and Xue Liu. Hierarchical gating networks for sequential recommendation. In Proceedings of the 25th ACM SIGKDD international conference on knowledge discovery & data mining, pages 825–833, 2019.

[29] Walid Maalej and Hadeer Nabil. Bug report, feature request, or simply praise? on automatically classifying app reviews. In 2015 IEEE 23rd international requirements engineering conference (RE), pages 116–125. IEEE, 2015.

[30] Julian McAuley. Amazon product data. http://jmcauley.ucsd.edu/data/amazon/, 2022. Accessed: 2022-11-06.

[31] Julian McAuley, Christopher Targett, Qinfeng Shi, and Anton Van Den Hengel. Image-based recommendations on styles and substitutes. In Proceedings of the 38th international ACM SIGIR conference on research and development in information retrieval, pages 43–52, 2015.

[32] Stuart McIlroy, Nasir Ali, Hammad Khalid, and Ahmed E. Hassan. Analyzing and automatically labelling the types of user issues that are raised in mobile app reviews. Empirical Software Engineering, 21:1067–1106, 2016.

[33] Jianmo Ni, Jiacheng Li, and Julian McAuley. Justifying recommendations using distantly-labeled reviews and fine-grained aspects. In Proceedings of the 2019 conference on empirical methods in natural language processing and the 9th international joint conference on natural language processing (EMNLP-IJCNLP), pages 188–197, 2019.

[34] Jianmo Ni, Larry Muhlstein, and Julian McAuley. Modeling heart rate and activity data for personalized fitness recommendation. In WWW, 2019.

[35] Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners. 2019.

[36] Jérémie Rappaz, Julian McAuley, and Karl Aberer. Recommendation on live-streaming platforms: Dynamic availability and repeat consumption. In Proceedings of the 15th ACM Conference on Recommender Systems, pages 390–399, 2021.

[37] Jérémie Rappaz, Julian McAuley, and Karl Aberer. Recommendation on live-streaming platforms: Dynamic availability and repeat consumption. In RecSys, 2021.

[38] Kamonphop Srisopha, Daniel Link, and Barry Boehm. How should developers respond to app reviews? features predicting the success of developer responses. EASE 2021, page 119–128, New York, NY, USA, 2021. Association for Computing Machinery.

[39] Fei Sun, Jun Liu, Jian Wu, Changhua Pei, Xiao Lin, Wenwu Ou, and Peng Jiang. Bert4rec: Sequential recommendation with bidirectional encoder representations from transformer. In Proceedings of the 28th ACM international conference on information and knowledge management, pages 1441–1450, 2019.

[40] Qiaoyu Tan, Jianwei Zhang, Jiangchao Yao, Ninghao Liu, Jingren Zhou, Hongxia Yang, and Xia Hu. Sparse-interest network for sequential recommendation. In Proceedings of the 14th ACM International Conference on Web Search and Data Mining, pages 598–606, 2021.

[41] Yong Kiam Tan, Xinxing Xu, and Yong Liu. Improved recurrent neural networks for session-based recommendations. In Proceedings of the 1st workshop on deep learning for recommender systems, pages 17–22, 2016.

[42] Mehrab Tanjim, Congzhe Su, Ethan Benjamin, Diane Hu, Liangjie Hong, and Julian McAuley. Attentive sequential models of latent intent for next item recommendation. In WWW, 2020.

[43] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

[44] Mengting Wan and Julian McAuley. Item recommendation on monotonic behavior chains. In Proceedings of the 12th ACM conference on recommender systems, pages 86–94, 2018.

[45] Mengting Wan, Rishabh Misra, Ndapa Nakashole, and Julian McAuley. Fine-grained spoiler detection from large-scale review corpora. arXiv preprint arXiv:1905.13416, 2019.

[46] Fangzhao Wu, Ying Qiao, Jiun-Hung Chen, Chuhan Wu, Tao Qi, Jianxun Lian, Danyang Liu, Xing Xie, Jianfeng Gao, Winnie Wu, et al. Mind: A large-scale dataset for news recommendation. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 3597–3606, 2020.

[47] Chengfeng Xu, Pengpeng Zhao, Yanchi Liu, Victor S Sheng, Jiajie Xu, Fuzhen Zhuang, Junhua Fang, and Xiaofang Zhou. Graph contextualized self-attention network for session-based recommendation. In IJCAI, volume 19, pages 3940–3946, 2019.

[48] An Yan, Chaosheng Dong, Yan Gao, Jinmiao Fu, Tong Zhao, Yi Sun, and Julian McAuley. Personalized complementary product recommendation. In WWW, 2022.

[49] Wayne Xin Zhao, Shanlei Mu, Yupeng Hou, Zihan Lin, Yushuo Chen, Xingyu Pan, Kaiyuan Li, Yujie Lu, Hui Wang, Changxin Tian, Yingqian Min, Zhichao Feng, Xinyan Fan, Xu Chen, Pengfei Wang, Wendi Ji, Yaliang Li, Xiaoling Wang, and Ji-Rong Wen. Recbole: Towards a unified, comprehensive and efficient framework for recommendation algorithms. In CIKM, pages 4653–4664. ACM, 2021.

[50] Cai-Nicolas Ziegler, Sean M McNee, Joseph A Konstan, and Georg Lausen. Improving recommendation lists through topic diversification. In Proceedings of the 14th international conference on World Wide Web, pages 22–32, 2005.
