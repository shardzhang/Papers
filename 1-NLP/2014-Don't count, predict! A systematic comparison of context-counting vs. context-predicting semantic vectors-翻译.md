# Don't count, predict! A systematic comparison of context-counting vs. context-predicting semantic vectors

> 别数了，去预测！上下文计数 与 上下文预测 语义向量 的系统比较
>
> Marco Baroni, Georgiana Dinu, Germán Kruszewski | 特伦托大学心智/大脑科学中心 | ACL 2014



本文对 **基于上下文计数的语义向量** 与 **基于上下文预测的语义向量** 进行了**迄今最系统的对比评估**，覆盖大量 **词法语义任务** 与 众多参数设置。核心发现是——**上下文预测模型在绝大多数任务上以 压倒性优势 战胜计数模型，作者坦言"连我们自己都感到惊讶"**。

核心内容：

- 计数模型（count model）用共现计数初始化向量，经重加权（PPMI、LMI）与降维（SVD、NMF）优化
- 预测模型（predict model）将向量估计直接框定为监督任务，用 word2vec 的 CBOW 架构训练，词向量权重直接优化为预测词所在上下文
- 在 2.8B token 语料（ukWaC + Wikipedia + BNC）上构建 36 个计数模型与 48 个预测模型
- 评估覆盖语义相关性（rg/ws/wss/wsr/men）、同义词检测（TOEFL）、概念分类（ap/esslli/battig）、选择偏好（up/mcrae）、类比（an/ansyn/ansem）14 个任务

关键发现：

- 预测模型在多数任务上大幅超越计数模型，如在 ws 上 75 vs 62、toefl 上 91 vs 76、an 上 68 vs 49
- 预测模型在多个任务上逼近甚至超越当前最优（soa）成绩，而 soa 结果几乎**全部依赖外部知识、手工规则、句法解析或更大语料**
- 预测模型对参数选择稳健：最差配置（平均排名 51）仍远好于计数模型最差配置（平均排名 83）
- **负采样优于层次 softmax**，**子采样频繁词**（类似 PMI 加权直觉）带来显著提升
- 开箱即用的 cw 向量（Collobert & Weston）表现很差（an 上仅 11），说明并非所有预测模型生而平等

---



## 摘要

上下文预测模型（更常被称为 **嵌入或神经语言模型**）是分布式语义领域的新宠。尽管围绕这些模型喧嚣不断，文献仍然缺乏将预测模型与经典的、基于计数向量的分布式语义方法进行系统比较。在本文中，我们对广泛的词法语义任务和众多参数设置进行了如此大规模的评估。结果——出乎我们自己的意料——表明这种喧嚣是完全合理的，因为**上下文预测模型彻底地、响亮地战胜了它们的计数对手**。



## 1 引言

**计算语言学中悠久的传统表明，上下文信息提供了对词义的很好近似，因为语义相似的词往往具有相似的上下文分布** [36]。具体来说，分布式语义模型（DSM，Distributional Semantic Model）使用记录目标词在大语料库中出现上下文（如共现词）的向量**作为意义表示的代理**，并对这些向量应用几何技术来度量相应词的意义相似度 [13, 16, 45]。

几十年来已经很清楚，原始共现计数效果不太好，当对原始向量应用各种变换时，DSM 能获得高得多的性能，例如根据上下文信息量对计数重新加权，并通过降维技术对其进行平滑。这种向量优化过程通常是无监督的，并且基于独立的考虑（例如，上下文重加权通常由信息论考虑证明其合理性，降维优化保留方差的数量等）。偶尔也会使用某种间接监督：尝试几种参数设置，并根据在选定的用于调优的语义任务上的性能选择最佳设置。

过去几年见证了新一代 DSM 的发展，它们将向量估计问题直接框定为监督任务，其中词向量中的权重被设置为最大化该词在语料库中出现的上下文的概率 [6, 14, 15, 25, 32, 44]。传统上下文向量的构造方式被颠倒过来：不是先收集上下文向量再基于各种标准对这些向量重新加权，而是直接设置向量权重以最优地预测相应词倾向于出现的上下文。由于相似的词出现在相似的上下文中，系统自然地学会给相似的词分配相似的向量。

这种训练 DSM 的新方式很有吸引力，因为它用单一的、定义明确的监督学习步骤取代了早期模型中本质上启发式的向量变换堆叠。同时，监督不需要任何手工标注成本，因为用于训练的上下文窗口可以从无标注语料库中自动提取（事实上，它们正是用于构建传统 DSM 的相同数据）。此外，至少部分相关方法可以高效扩展到处理非常大量的输入数据。[^1]

[^1]: 直接基于目标最优函数学习参数向量的想法与潜在狄利克雷分配（LDA，Latent Dirichlet Allocation）模型 [8, 21] 共享，其中参数被设置为优化词和文档的联合概率分布。然而，完全概率化的 LDA 模型在扩展到大数据集方面存在问题。

我们将以传统方式构建的 DSM 称为计数模型（count model）（因为它们用共现计数初始化向量），将其基于训练的替代方案称为预测模型（predict(ive) model）。[^2]

现在，最自然的问题当然是：在经验上哪种方法最好？令人惊讶的是，尽管在标准基准上对替代计数 DSM 进行广泛评估的传统悠久 [1, 3, 10, 11, 41, 37]，现有文献中关于计数与预测 DSM 的直接比较却非常少。这部分是因为上下文预测向量最初是作为语言建模的方法和/或作为基于神经网络的"深度学习"NLP 架构中初始化特征向量的方式而开发的，因此它们作为语义表示的有效性最初被视为不过是一个有趣的副作用。缺乏系统比较也可能部分归因于社会学原因：上下文预测模型是在神经网络社区内开发的，对计算语言学中最近的 DSM 工作很少或没有了解。

无论原因如何，据我们所知只有三项工作报告了直接比较，且都范围有限。[25] 顺带在标准 WordSim353 基准上比较了一个计数模型和几个预测 DSM（他们论文的表 3）。在这个实验中，计数模型实际上优于最佳的预测方法。而在上下文中词相似度任务中（表 5），最佳预测模型优于计数模型，尽管差距不大。[7] 比较了计数和预测表示作为组合函数的输入。在短语相似度任务中，计数向量是更好的输入，而在释义分类实验中两种表示相当。[^3] 最后，[35] 在句法和语义类比任务上比较了他们的预测模型与"潜在语义分析"（LSA）计数向量，发现预测模型高度优越。然而，关于他们使用的 LSA 计数向量，他们提供的细节很少。[^4]

在本文中，我们通过在许多参数设置下以及大量（大多是标准的）词法语义基准上对计数和预测 DSM 进行直接评估，克服了比较稀缺的问题。我们的标题已经透露了我们的发现。

[^2]: 我们要把第一个术语归功于 Hinrich Schütze（私人通信）。预测 DSM 也被称为神经语言模型，因为它们的监督上下文预测训练是用神经网络进行的，或者更晦涩地称为"嵌入"。
[^3]: 我们这里指的是 http://homepages.inf.ed.ac.uk/s1066731/pdf/emnlp2012erratum.pdf 勘误表中报告的更新结果。
[^4]: [12] 提出了扩展的经验评估，但仅限于替代的上下文预测模型，不包括我们这里使用的 word2vec 变体。

## 2 分布式语义模型

计数和预测模型都是从约 28 亿 token 的语料库中提取的，该语料库由 ukWaC、英语 Wikipedia 和英国国家语料库（BNC，British National Corpus）拼接而成。[^5][^6][^7] 对于两种模型类型，我们都将语料库中最频繁的 30 万个词同时作为目标元素和上下文元素。

[^5]: http://wacky.sslmit.unibo.it
[^6]: http://en.wikipedia.org
[^7]: http://www.natcorp.ox.ac.uk

### 2.1 计数模型

我们使用 DISSECT 工具包准备计数模型。[^8] 我们从目标词两侧各两个和五个词组成的对称上下文窗口中提取计数向量。我们考虑了两种加权方案：正逐点互信息（PPMI，Positive Pointwise Mutual Information）和局部互信息（LMI，Local Mutual Information，类似于广泛使用的对数似然比方案）[17]。我们同时使用完整向量和压缩向量。后者通过应用奇异值分解（SVD，Singular Value Decomposition）[20] 或非负矩阵分解（NMF，Non-negative Matrix Factorization）[29] 的 [30] 算法获得，缩减大小从 200 到 500、步长 100。总共评估了 36 个计数模型。

计数模型有着如此悠久而丰富的历史，我们只能探索文献中提出的计数、加权和压缩方法的一小部分子集。然而，值得指出的是，所评估的参数子集包含了在 [10, 11] 进行的参数空间系统探索中被发现最有效的设置（窄上下文窗口、正 PMI、SVD 缩减）。

[^8]: http://clic.cimec.unitn.it/composes/toolkit/

### 2.2 预测模型

我们用 word2vec 工具包训练预测模型。[^9] 该工具包实现了 [32, 34] 的 skip-gram 和 CBOW 方法。我们只用后者做实验，它也是两者中计算效率更高的模型，遵循 [33] 的建议，即 CBOW 更适合更大的数据集。

CBOW 模型学习基于窗口中词向量表示之和来预测对称窗口中间的词。我们考虑了中心元素两侧各 2 个和 5 个词的上下文窗口。我们在 200 到 500 范围内以 100 为步长变化向量维度。word2vec 工具包实现了两种对标准 softmax 分类器计算输出词概率分布的替代方案。层次 softmax 是一种计算高效的方式，使用与 $\log(\text{unigram.perplexity}(W))$ 成正比（而非 $W$，$W$ 为词汇表大小）的输出层来估计整体概率分布。作为替代，负采样通过学习区分输出词与从噪声分布中抽取的样本（负样本数量由参数 $k$ 给出）来估计输出词的概率。我们测试了层次 softmax 和 $k$ 值为 5 和 10 的负采样。

像 the 或 a 这样的非常频繁的词作为上下文特征信息量不大。word2vec 工具包实现了一种降低其影响（同时提高速度性能）的方法。更准确地说，训练数据中的词以与其频率成正比（捕获了与 PMI 等传统计数向量加权度量相同的直觉）的概率被丢弃。这由一个参数 $t$ 控制，频率高于 $t$ 的词被激进地子采样。我们训练了无子采样和子采样 $t = 1e-5$ 的模型（工具包页面根据经验观察建议 $1e-3 - 1e-5$ 为有效范围）。

总共我们评估了 48 个预测模型，与我们考虑的计数模型数量相当。

[^9]: https://code.google.com/p/word2vec/

### 2.3 开箱即用模型

[3] 公开了他们表现最佳的分布式记忆（dm，Distributional Memory）模型的向量。[^10] 该模型基于我们使用的相同输入语料库，代表了一个"语言学丰富"的基于计数的 DSM，它依赖词元（lemma）而非原始词形，其维度编码了连接目标和上下文的句法关系和/或词法-句法模式。Baroni and Lenci 在大规模评估中表明，dm 在多种语义任务中达到接近最先进的性能。

我们还试验了 Ronan Collobert 公开的流行预测向量。[^11] 遵循早期文献，我们称它们为 Collobert and Weston（cw）向量。这些是在 Wikipedia 上训练了两个月的 100 维向量！特别是，这些向量被训练来优化在 11 词上下文窗口中间从随机候选中选择正确词的任务 [15]。

[^10]: http://clic.cimec.unitn.it/dm/
[^11]: http://ronan.collobert.com/senna/



## 3 评估材料

我们在各种基准上测试模型，其中大多数已被广泛用于测试和比较 DSM。下面的基准描述也解释了表 2 中报告的性能度量和最先进结果。

### 语义相关性

第一组语义基准通过让人类受试者在数值量表上对两个词之间的语义相似度或相关程度进行评分来构建。计算模型的性能通过受试者分配给词对的平均得分与模型空间中相应向量之间的余弦值之间的相关性来评估（遵循先前工作，我们对 rg 使用 Pearson 相关，所有其他情况使用 Spearman）。[40] 的经典数据集（rg）包含 65 个名词对。[23] 使用利用 Wikipedia 链接结构和词义消歧技术的技术报告了该集上最先进的性能。[18] 引入了广泛使用的 WordSim353 集（ws），顾名思义，它包含 353 对。当前最先进水平由 [22] 达到，其方法精神上与预测模型一致，但让 WordNet 的同义词信息约束学习过程（通过偏好 WordNet 同义词在语义空间中靠近的解）。[1] 将 ws 集分为相似度（wss）和相关度（wsr）子集。前者包含更紧密的分类学关系，如近义和共下义（king/queen），而后者包含更广泛的、可能主题性或组合性关系（family/planning）。我们报告 Agirre 及其同事从非常大的语料库（比我们的大几个数量级）中提取的不同类型计数向量在两个子集上的最先进性能。最后，我们使用（测试部分）MEN（men），包含 1,000 个词对。这个基准的开发者 [9] 通过在特设训练数据上广泛调优，并使用文本和图像提取的特征表示词义，达到最先进性能。

### 同义词检测

经典的 TOEFL（toefl）集由 [28] 引入。它包含 80 道多项选择题，将一个目标词与 4 个同义词候选配对。例如，对于目标词 levied，必须在 imposed（正确）、believed、requested 和 correlated 之间选择。DSM 计算每个候选向量与目标的余弦值，并选择余弦值最大的候选作为答案。性能以正确答案准确率评估。[11] 通过对计数模型参数空间的非常彻底的探索达到了 100% 的准确率。

### 概念分类

给定一组名词性概念，任务是将它们分组为自然类别（例如，直升机和摩托车应该归入交通工具类，狗和大象归入哺乳动物类）。遵循先前工作，我们将分类视为无监督聚类任务。模型产生的向量使用 CLUTO 工具包 [26] 聚类为 $n$ 组（$n$ 由金标准划分确定），使用全局优化的重复二分方法，其他使用 CLUTO 的默认设置（这些是文献中的标准选择）。性能以纯度评估，纯度衡量每个聚类包含来自单一金标准类别概念的程度。如果金标准划分被完美重现，纯度达到 100%；随着聚类质量恶化接近 0。Almuhareb-Poesio（ap）基准包含组织成 21 个类别的 402 个概念 [2]。[39] 使用基于精心构建的句法链接的计数模型达到了最先进的纯度。ESSLLI 2008 分布式语义研讨会共享任务集（esslli）包含要聚类成 6 个类别的 44 个概念 [4]（我们这里忽略该集附带的三分和两分高层划分）。[27] 使用完整 Web 作为语料库和手工构建的、语言驱动的模式在该集上达到最高性能。最后，[5] 引入的 Battig（battig）测试集包含来自 10 个类别的 83 个概念。当前最先进水平由 [3] 的基于窗口的计数模型达到。

### 选择偏好

我们实验了两个包含动词-名词对的数据集，受试者对这些名词作为动词主语或宾语的原型性进行评分（例如，people 作为 to eat 的主语获得高分，作为同一动词的宾语获得低分）。我们遵循 [3] 提出的程序来解决这个挑战：对于每个动词，我们使用他们提供的基于语料库的元组选择与动词作为主语或宾语关联最强的 20 个名词，并对这些名词的向量求平均，为相关论元槽获得一个"原型"向量。然后我们度量目标名词的向量与相关原型向量的余弦值（例如，people 与吃的主语原型向量的余弦值）。系统通过这些余弦值与平均人类原型性评分的 Spearman 相关来评估。我们的第一个数据集由 [38] 引入，包含 211 对（up）。最高性能由 [24] 的监督计数向量系统达到（监督的意义在于他们直接在金标准数据上训练分类器，与上下文学习方法的零成本监督形成对比）。mcrae 集 [31] 包含 100 个名词-动词对，最高性能由 [3] 的 DepDM 系统达到，这是一个依赖句法信息的计数 DSM。

### 类比

虽然前面所有数据集在 DSM 领域相对标准，用于测试传统计数模型，但我们最后一个基准是 [32] 专门为测试预测模型引入的。数据集包含约 9K 个语义和 10.5K 个句法类比问题。语义问题给出一个示例对（brother-sister）、一个测试词（grandson），并要求找到另一个实例化示例所展示关系（相对于测试词）的词（granddaughter）。句法问题类似，但关系是语法性质的（work-works，speak...speaks）。Mikolov 及其同事通过从第一个示例项向量中减去第二个示例项向量，加上测试项，并寻找所得向量的最近邻来解决这个挑战（$\vec{\text{brother}} - \vec{\text{sister}} + \vec{\text{grandson}}$ 的最近邻是什么？）。系统以整个语义空间中最近邻是正确答案的问题比例来评估（给定的示例和测试向量三元组从最近邻搜索中排除）。[32] 使用与我们类似的 CBOW 预测模型（但在两倍大的语料库上训练）在句法子集（ansyn）上达到最高准确率。整个数据集（an）和语义子集（ansem）上的最高准确率由 [34] 使用 skip-gram 预测模型达到。但请注意，由于任务的框架方式，性能也取决于要搜索的词汇量大小：[32] 在 100 万个词的向量中挑选最近邻，[34] 在 70 万个词中，我们在 30 万个词中。

我们使用的基准的一些特征总结在表 1 中。

**表 1：实验中使用的基准，包括任务类型、性能度量（measure）、原始参考文献（source）和当前最先进系统的参考文献（soa）。**

<img src="/Users/dazhang/PycharmProject/Papers/1-NLP/.picture/image-20260817161036004.png" alt="image-20260817161036004" style="zoom:50%;" />

## 4 结果

表 2 总结了评估结果。表的第一块报告了计数和预测向量（跨所有考虑的参数设置）的每任务最高性能。后者成为明确的赢家，在大多数任务中以很大优势超越计数向量。事实上，预测模型取得了令人印象深刻的总体系数，在几个情况下击败了当前最先进水平，并且在更多情况下接近它。值得强调的是，如第 3 节所述，最先进结果在几乎所有情况下都是使用依赖外部知识、手工规则、句法解析、更大语料库和/或任务特定调优的专门方法获得的。而我们的预测结果只是下载 word2vec 工具包并用工具包开发者推荐的参数范围运行而获得的。

预测模型的成功不能归咎于计数模型的糟糕性能。除了这无法解释预测向量的接近最先进性能之外，计数模型的结果绝对值实际上相当好。事实上，在几种情况下，它们接近甚至优于 dm 获得的结果，dm 是一种语言学上复杂的基于计数的方法，[3] 证明其在各种任务上达到顶级性能。

有趣的是，计数向量只在选择偏好任务上达到与预测向量相当的性能。特别是 up 任务也是预测模型严重落后于最先进水平和 dm 性能的唯一基准。回想第 3 节，我们通过创建代表典型动词论元的平均向量来处理选择偏好。我们推测这种对 dm 向量有效的平均方法可能对预测训练的向量有问题，我们计划在未来的研究中探索构建原型的方法。

我们的结果对参数选择稳健吗，还是归因于非常具体且脆弱的设置？表 2 的接下来几块回答了这个问题。第二块报告了使用在任务间平均性能排名方面最佳的单一计数和预测模型（分别是表 3 和表 4 顶部行的模型）获得的结果。我们看到，对于两种方法，使用单一最佳设置而不是任务特定设置不会严重影响性能，除了最佳预测模型在 esslli 上性能大幅下降（由于该数据集很小？），以及计数模型在 ansem 上更戏剧性的下降。表 2 的第三块报告了一个更有说服力和更有趣的评估，我们看看如果使用任务间性能最差的单一模型会发生什么（回想第 2 节，无论如何，我们探索的是合理参数设置的空间，实验者可能倾向于不调优就选择的那种）。计数模型的性能受到这种不幸选择的严重影响（2 词窗口、局部互信息、NMF、400 维、平均性能排名：83），而预测方法稳健得多：就其最差实例（2 词窗口、层次 softmax、无子采样、200 维、平均排名：51）而言，它的性能仅在 an 和 ansem 任务上比最佳计数模型低 10% 以上，实际上在 3 种情况下更高（注意在 esslli 上最差预测模型的性能比最好的还好，证实了我们对该小数据集脆弱性的怀疑）。第四块报告了在最现实的场景中的性能，即在开发任务上调优参数。具体来说，我们选择在小的 rg 集上表现最好的模型，并报告它们在所有任务上的性能（通过选择其他调优集我们获得了类似结果）。被选中的计数模型是表 3 中报告的整体第三好的同类模型。被选中的预测模型是表 4 中第四好的模型。计数整体性能不受此选择的太大影响。预测模型再次确认了它们的稳健性，因为它们的 rg 调优性能总是接近（在 3 种情况下更好）最佳整体设置所达到的性能。

**表 2：计数（cnt）、预测（pre）、dm 和 cw 模型在所有任务上的性能。性能度量和最先进结果（soa）见第 3 节和表 1。由于 dm 在 an\* 数据集上的覆盖率非常低，我们不在那里报告其性能。**

<img src="/Users/dazhang/PycharmProject/Papers/1-NLP/.picture/image-20260817161100294.png" alt="image-20260817161100294" style="zoom:50%;" />

表 3 和表 4 让我们通过报告两类中最佳模型（基于任务间平均性能排名）的特征，更仔细地审视最重要的计数和预测参数。对于计数模型，PMI 显然是更好的加权方案，SVD 作为降维技术优于 NMF。然而，完全不压缩（使用全部 30 万个原始维度）效果最好。将此与最佳整体预测向量相比，后者只有 400 维，使它们更实用。对于预测模型，我们在表 4 中观察到，负采样（任务是区分目标输出词与从噪声分布中抽取的样本）优于更昂贵的层次 softmax 方法。子采样频繁词（类似于计数模型中 PMI 加权那样降低这些词的重要性）也带来显著改进。

**表 3：在所有任务上基于平均性能排名最佳的顶级计数模型。第一行说明 window-2、PMI、300K 计数模型是最好的计数模型，并且在所有任务中，当所有模型按性能降序排列时，其平均排名为 35。参数解释见第 2.1 节。**

<img src="/Users/dazhang/PycharmProject/Papers/1-NLP/.picture/image-20260817161132915.png" alt="image-20260817161132915" style="zoom:50%;" />

**表 4：在所有任务上基于平均性能排名最佳的顶级预测模型。参数解释见第 2.2 节。**

![image-20260817161151587](/Users/dazhang/PycharmProject/Papers/1-NLP/.picture/image-20260817161151587.png)

最后，我们回到表 2 指出开箱即用的 cw 模型的糟糕性能。我们必须把使我们预测向量比 cw 好得多的参数调查（更多样化的训练语料？窗口大小？使用的目标函数？子采样？……）留给进一步的工作。尽管如此，我们的结果表明，仅仅通过上下文预测训练并不能保证良好的性能。cw 方法非常流行（例如，[25] 和 [7] 在我们第 1 节讨论的研究中都使用了它）。如果我们基于 cw 模型进行计数和预测向量的系统比较，我们就会得出与我们基于 word2vec 训练的向量得出的结论相反的结论！



## 5 结论

本文提出了第一个计数和预测向量的系统性比较评估。作为在开发和计数向量使用方面经验丰富的资深分布式语义学者，我们开展这项研究是因为我们厌烦了经常环绕预测模型的胜利主义口吻，尽管几乎完全没有与计数向量的适当比较。[^12] 我们秘密的愿望是发现这一切都是炒作，计数向量远比它们的预测对手优越。更现实的期望是会出现复杂的图景，预测和计数向量在不同任务上互有胜负。相反，我们发现预测模型如此之好，以至于虽然胜利主义的口吻仍然显得过度，但有非常充分的理由转向新架构。然而，由于篇幅限制，我们这里只关注定量度量：这两种类型的模型在它们所犯的错误上是否互补仍有待观察，如果是这样，组合模型可能是进一步工作的有趣途径。

计数 DSM 的参数空间非常大，完全有可能我们未考虑的一些选项会稍微提高计数向量的性能。尽管如此，鉴于预测向量也优于基于句法的 dm 模型，并且经常接近最先进性能，更有成效的前进方式可能是专注于预测模型的参数和扩展：毕竟，我们只是尝试了 word2vec 默认值的几个变体就获得了已经出色的结果。再加上，除了我们在这里测试的标准词法语义挑战外，预测模型目前正成功应用于前沿领域，如表示短语 [34, 42] 或在共同的语义空间中融合语言和视觉 [19, 43]。

基于这里报告的结果和我们刚才的考虑，我们当然会推荐任何有兴趣将 DSM 用于理论或实际应用的人选择预测模型，并附带一个重要告诫：它们并非生而平等（参见 word2vec 和 cw 模型之间的巨大差异）。同时，鉴于对计数 DSM 已进行了大量工作，我们希望在不久的将来探索某些针对传统 DSM 考虑的问题和方法将如何转移到预测模型。例如，潜在语义分析 [28]、主题模型 [21] 及相关 DSM 的开发者已经表明，这些模型的维度可以被解释为一般的"潜在"（latent）语义域，这赋予相应模型某种先验的认知合理性，同时为有趣的应用铺平道路。DSM 研究的另一个重要方向涉及"上下文工程"：例如，在如何将句法信息编码到上下文特征方面已经做了很多工作 [37]，更近期的研究构建并组合表达主题性与功能性信息的特征空间 [46]。仅举最后一个例子，分布式语义学者已经研究了向量的某些属性是否以预期的方式反映语义关系：例如，上义词的向量是否以某种数学精确的意义"分布式包含"下义词的向量。

预测模型的维度是否也编码潜在语义域？这些模型是否提供计数向量在捕获语言学丰富上下文方面的同样灵活性？预测向量的结构是否模仿有意义的语义关系？所有这些甚至是否重要，还是我们正站在发现彻底处理传统分布式语义中刚才概述的问题的新方法的门槛上？

无论哪种方式，本研究的结果表明这些是计算语义学未来研究的重要方向。

[^12]: 这里有一个例子，其中 word2vec 被称为自然语言处理的皇冠明珠：http://bit.ly/1ipv72M



## 致谢

我们感谢 ERC 2011 独立研究员启动基金 n. 283554（COMPOSES）。



## 参考文献

[1] Eneko Agirre, Enrique Alfonseca, Keith Hall, Jana Kravalova, Marius Pasca, and Aitor Soroa. 2009. A study on similarity and relatedness using distributional and WordNet-based approaches. In Proceedings of HLT-NAACL, pages 19–27, Boulder, CO.

[2] Abdulrahman Almuhareb. 2006. Attributes in Lexical Acquisition. Phd thesis, University of Essex.

[3] Marco Baroni and Alessandro Lenci. 2010. Distributional Memory: A general framework for corpus-based semantics. Computational Linguistics, 36(4):673–721.

[4] Marco Baroni, Stefan Evert, and Alessandro Lenci, editors. 2008. Bridging the Gap between Semantic Theory and Computational Simulations: Proceedings of the ESSLLI Workshop on Distributional Lexical Semantic. FOLLI, Hamburg.

[5] Marco Baroni, Eduard Barbu, Brian Murphy, and Massimo Poesio. 2010. Strudel: A distributional semantic model based on properties and types. Cognitive Science, 34(2):222–254.

[6] Yoshua Bengio, Réjean Ducharme, Pascal Vincent, and Christian Janvin. 2003. A neural probabilistic language model. Journal of Machine Learning Research, 3:1137–1155.

[7] William Blacoe and Mirella Lapata. 2012. A comparison of vector-based representations for semantic composition. In Proceedings of EMNLP, pages 546–556, Jeju Island, Korea.

[8] David Blei, Andrew Ng, and Michael Jordan. 2003. Latent Dirichlet allocation. Journal of Machine Learning Research, 3:993–1022.

[9] Elia Bruni, Nam Khanh Tran, and Marco Baroni. 2013. Multimodal distributional semantics. Journal of Artificial Intelligence Research. In press; http://clic.cimec.unitn.it/marco/publications/mmds-jair.pdf.

[10] John Bullinaria and Joseph Levy. 2007. Extracting semantic representations from word co-occurrence statistics: A computational study. Behavior Research Methods, 39:510–526.

[11] John Bullinaria and Joseph Levy. 2012. Extracting semantic representations from word co-occurrence statistics: Stop-lists, stemming and SVD. Behavior Research Methods, 44:890–907.

[12] Yanqing Chen, Bryan Perozzi, Rami Al-Rfou', and Steven Skiena. 2013. The expressive power of word embeddings. In Proceedings of the ICML Workshop on Deep Learning for Audio, Speech and Language Processing, Atlanta, GA. Published online: https://sites.google.com/site/deeplearningicml2013/accepted_papers.

[13] Stephen Clark. 2013. Vector space models of lexical meaning. In Shalom Lappin and Chris Fox, editors, Handbook of Contemporary Semantics, 2nd ed. Blackwell, Malden, MA. In press; http://www.cl.cam.ac.uk/~sc609/pubs/sem_handbook.pdf.

[14] Ronan Collobert and Jason Weston. 2008. A unified architecture for natural language processing: Deep neural networks with multitask learning. In Proceedings of ICML, pages 160–167, Helsinki, Finland.

[15] Ronan Collobert, Jason Weston, Léon Bottou, Michael Karlen, Koray Kavukcuoglu, and Pavel Kuksa. 2011. Natural language processing (almost) from scratch. Journal of Machine Learning Research, 12:2493–2537.

[16] Katrin Erk. 2012. Vector space models of word meaning and phrase meaning: A survey. Language and Linguistics Compass, 6(10):635–653.

[17] Stefan Evert. 2005. The Statistics of Word Cooccurrences. Ph.D dissertation, Stuttgart University.

[18] Lev Finkelstein, Evgeniy Gabrilovich, Yossi Matias, Ehud Rivlin, Zach Solan, Gadi Wolfman, and Eytan Ruppin. 2002. Placing search in context: The concept revisited. ACM Transactions on Information Systems, 20(1):116–131.

[19] Andrea Frome, Greg Corrado, Jon Shlens, Samy Bengio, Jeff Dean, Marc'Aurelio Ranzato, and Tomas Mikolov. 2013. DeViSE: A deep visual-semantic embedding model. In Proceedings of NIPS, pages 2121–2129, Lake Tahoe, Nevada.

[20] Gene Golub and Charles Van Loan. 1996. Matrix Computations (3rd ed.). JHU Press, Baltimore, MD.

[21] Tom Griffiths, Mark Steyvers, and Josh Tenenbaum. 2007. Topics in semantic representation. Psychological Review, 114:211–244.

[22] Guy Halawi, Gideon Dror, Evgeniy Gabrilovich, and Yehuda Koren. 2012. Large-scale learning of word relatedness with constraints. In Proceedings of KDD, pages 1406–1414.

[23] Samer Hassan and Rada Mihalcea. 2011. Semantic relatedness using salient semantic analysis. In Proceedings of AAAI, pages 884–889, San Francisco, CA.

[24] Amac Herdagdelen and Marco Baroni. 2009. BagPack: A general framework to represent semantic relations. In Proceedings of GEMS, pages 33–40, Athens, Greece.

[25] Eric Huang, Richard Socher, Christopher Manning, and Andrew Ng. 2012. Improving word representations via global context and multiple word prototypes. In Proceedings of ACL, pages 873–882, Jeju Island, Korea.

[26] George Karypis. 2003. CLUTO: A clustering toolkit. Technical Report 02-017, University of Minnesota Department of Computer Science.

[27] Sophia Katrenko and Pieter Adriaans. 2008. Qualia structures and their impact on the concrete noun categorization task. In Proceedings of the ESSLLI Workshop on Distributional Lexical Semantics, pages 17–24, Hamburg, Germany.

[28] Thomas Landauer and Susan Dumais. 1997. A solution to Plato's problem: The latent semantic analysis theory of acquisition, induction, and representation of knowledge. Psychological Review, 104(2):211–240.

[29] Daniel Lee and Sebastian Seung. 2000. Algorithms for Non-negative Matrix Factorization. In Proceedings of NIPS, pages 556–562.

[30] Chih-Jen Lin. 2007. Projected gradient methods for Nonnegative Matrix Factorization. Neural Computation, 19(10):2756–2779.

[31] Ken McRae, Michael Spivey-Knowlton, and Michael Tanenhaus. 1998. Modeling the influence of thematic fit (and other constraints) in on-line sentence comprehension. Journal of Memory and Language, 38:283–312.

[32] Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. 2013a. Efficient estimation of word representations in vector space. http://arxiv.org/abs/1301.3781/.

[33] Tomas Mikolov, Quoc Le, and Ilya Sutskever. 2013b. Exploiting similarities among languages for Machine Translation. http://arxiv.org/abs/1309.4168.

[34] Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, and Jeff Dean. 2013c. Distributed representations of words and phrases and their compositionality. In Proceedings of NIPS, pages 3111–3119, Lake Tahoe, Nevada.

[35] Tomas Mikolov, Wen-tau Yih, and Geoffrey Zweig. 2013d. Linguistic regularities in continuous space word representations. In Proceedings of NAACL, pages 746–751, Atlanta, Georgia.

[36] George Miller and Walter Charles. 1991. Contextual correlates of semantic similarity. Language and Cognitive Processes, 6(1):1–28.

[37] Sebastian Padó and Mirella Lapata. 2007. Dependency-based construction of semantic space models. Computational Linguistics, 33(2):161–199.

[38] Ulrike Padó. 2007. The Integration of Syntax and Semantic Plausibility in a Wide-Coverage Model of Sentence Processing. Dissertation, Saarland University, Saarbrücken.

[39] Klaus Rothenhäusler and Hinrich Schütze. 2009. Unsupervised classification with dependency based word spaces. In Proceedings of GEMS, pages 17–24, Athens, Greece.

[40] Herbert Rubenstein and John Goodenough. 1965. Contextual correlates of synonymy. Communications of the ACM, 8(10):627–633.

[41] Magnus Sahlgren. 2006. The Word-Space Model. Ph.D dissertation, Stockholm University.

[42] Richard Socher, Brody Huval, Christopher Manning, and Andrew Ng. 2012. Semantic compositionality through recursive matrix-vector spaces. In Proceedings of EMNLP, pages 1201–1211, Jeju Island, Korea.

[43] Richard Socher, Milind Ganjoo, Christopher Manning, and Andrew Ng. 2013. Zero-shot learning through cross-modal transfer. In Proceedings of NIPS, pages 935–943, Lake Tahoe, Nevada.

[44] Joseph Turian, Lev-Arie Ratinov, and Yoshua Bengio. 2010. Word representations: A simple and general method for semi-supervised learning. In Proceedings of ACL, pages 384–394, Uppsala, Sweden.

[45] Peter Turney and Patrick Pantel. 2010. From frequency to meaning: Vector space models of semantics. Journal of Artificial Intelligence Research, 37:141–188.

[46] Peter Turney. 2012. Domain and function: A dual-space model of semantic relations and compositions. Journal of Artificial Intelligence Research, 44:533–585.
