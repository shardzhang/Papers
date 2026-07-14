5
2
0
2

l
u
J

5
1

]

G
L
.
s
c
[

1
v
6
4
2
1
1
.
7
0
5
2
:
v
i
X
r
a

生成式点击率预测
在搜索广告中的应用∗

林伟·孔, 王璐, 常平·彭, 占刚·林, Ching Law & 静平·邵
京东
{konglingwei6,wanglu241,pengchangping}@jd.com
{linzhangang,lawching,shaojingping}@jd.com

摘要

点击率（CTR）预测模型是众多工业场景中不可或缺的组成部分，
例如个性化搜索广告。当前的方法通常涉及从用户历史行为序列中提取特征，
结合产品信息，输入到一个判别式模型中，该模型在用户反馈上训练
以估计点击率。随着诸如GPT等模型的成功，
生成式模型在超越判别式模型方面丰富表达能力的潜力
已经显现。鉴于此，我们引入了一种新颖的模型，利用
生成式模型来提高判别式模型中点击率预测的精度。
为了调和两种模型类型的不同数据聚合需求，
我们设计了一个两阶段训练过程：1）生成式预训练，用于给定用户行为序列中
下一个物品类别的下一物品预测；2）在判别式点击率预测框架内微调
训练良好的生成式模型。
我们方法的效果通过在新数据集上的大量实验得到证实，
其显著效用通过在线A/B测试结果得到进一步印证。
目前，该模型已部署在全球最大的电子商务
平台之一上，我们计划在未来发布相关代码和数据集。

1

引言

点击率（CTR）预测在在线广告中至关重要，特别是在按点击付费（CPC）
收入模式下，广告商为其赞助物品的每次点击付费。点击率
预测模型分析历史广告展示和点击日志，以估计用户
点击给定物品的概率。这些估计随后与广告出价一起用于确定
展示哪些广告，影响用户体验和广告商ROI。传统上，点击率预测
模型依赖于判别式方法，使用用户交互和产品细节中的特征
来预测响应（Zhou等人，2018b；Song等人，2021；Zheng等人，2022；Wu等人，2021；
Wang等人，2023；Kong等人，2023）。

我们提出了一种突破性的方法GenCTR（生成式点击率预测），该方法利用
生成式模型的潜力来增强点击率（CTR）
预测系统的表示能力。我们的模型旨在通过有效捕捉用户行为中的复杂模式来超越传统判别式
模型的预测精度。GenCTR采用
一种独特的双阶段训练方法来调和生成式
方法和判别式方法的不同数据处理需求。在初始阶段，生成式预训练通过预测用户交互序列中的
后续物品来建立对用户偏好的稳健
理解。然后，在第二阶段，这个预训练模型在针对
点击率预测优化的判别式框架内进行微调。

为了有效利用GenCTR的固有优势，我们引入了四种核心技术：
生成式预训练阶段的条件自条件解码器和条件负采样，
其次是判别式微调阶段的参数共享和模型集成。

我们通过在世界领先的电子商务平台之一的
大规模在线搜索广告系统中成功部署GenCTR来证明其实用价值，服务于每天数亿用户。
∗本文首次提交于2024年2月9日。

1

为了促进持续的研究进展，我们正在发布一个从真实流量收集的新颖
公共数据集。该数据集包括预训练和微调
数据，为研究社区提供了宝贵的资源。

2 相关工作

2.1 点击率预测模型

对准确点击率（CTR）预测的追求导致了众多
模型的发展，这些模型是在线广告和推荐系统中个性化的基石。
早期的模型如逻辑回归，受限于其线性特性和无法
捕捉复杂的特征交互。Cheng等人（Cheng等人，2016）提出的Wide & Deep模型
代表了重大进展，结合了宽线性模型的记忆能力
和深度神经网络的泛化能力。在此之后，
DeepFM（Guo等人，2017）和xDeepFM（Lian等人，2018）模型通过将因子分解机
集成到深度学习框架中，进一步增强了特征
交互建模。然而，这些模型主要关注静态用户特征，没有考虑用户行为的
时间动态性。为了解决传统点击率模型中用户表示的静态性质，
研究人员开始探索能够融入动态用户行为的模型。Zhou等人（Zhou等人，2018a）提出的深度兴趣
网络（DIN）是这一领域的开创性工作，
引入了注意力机制，在预测不同物品的点击率时对用户的历史行为进行差异化加权。
该模型能够捕捉过去交互相对于当前广告或物品的不同相关性。随后，研究人员探索了两阶段方法，
如SIM（Pi等人，2020）、ETA（Chen等人，2021）和TWIN（Chang等人，2023），这些方法
成功融入了更广泛的用户历史行为，同时确保了可管理的
服务复杂度。这些方法在提高用户兴趣预测的准确性方面取得了有希望的结果。

尽管这些进展显著改善了对用户历史数据的利用，
但在利用丰富的用户正反馈方面仍有未开发的潜力。在我们的工作中，我们
进一步通过在用户行为上预训练生成式模型，利用用户正反馈
作为标签。

2.2 序列推荐

序列推荐系统旨在通过考虑过去交互的顺序来预测用户可能交互的下一个物品。
这些模型的演变始于卷积神经网络（CNN）和循环神经网络（RNN）
在捕捉用户-物品交互序列模式中的应用。基于CNN的模型，如Caser（Tang & Wang，
2018），应用卷积滤波器来学习交互序列的局部特征。基于RNN的
方法，包括GRU4Rec（Hidasi等人，2015），利用门控循环单元动态建模用户的
序列行为。注意力机制的引入标志着序列推荐模型的重大
进展。Kang等人（Kang & McAuley，2018）的自注意力序列推荐
（SASRec）模型利用自注意力来识别哪些
过去的物品对未来交互最具预测性。该模型通过关注用户交互历史中最相关的部分，
能够优于传统的
基于RNN的模型。在基于注意力的模型成功的基础上，Sun等人（Sun等人，2019）的Bert4Rec将
来自变换器的双向编码器表示（BERT）架构融入用户序列建模。
Bert4Rec的双向训练策略提供了对序列中每个物品周围上下文的更全面理解，
从而提高了推荐性能。

虽然上述模型显著推动了序列推荐领域的发展，
但我们的工作不同，专注于在点击率预测的背景下应用序列推荐技术。
我们利用序列推荐模型作为预训练步骤
来丰富点击率预测模型的特征空间。这个预训练步骤使我们的模型
能够捕捉用户行为的时间动态性，这在传统点击率
模型中经常被忽视。

2

图1：GenCTR的整体架构

3 预备知识

我们用X表示输入空间，包含与用户、物品以及
相关上下文相关的所有特征。输出空间Y = {0, 1}，表示是否发生点击。令D表示
X × Y上的潜在分布。点击率（CTR）预测旨在开发一个
模型ŷ：X → (0, 1)。该模型估计对于任何给定输入x ∈ X发生点击的概率（即点击率），
其中ŷ(x)本质上预测Pr_{y~D_{Y|x}}{y = 1 | x}。
每个用户都可以拥有一个独特的用户行为序列，表示他们与物品交互的时间顺序记录
（例如，点击、购买和将物品加入购物车）。该序列可以
进一步通过附加到每个物品的辅助信息进行丰富，提供更深层的上下文。令A为
物品空间，C为与这些物品对应的辅助信息空间。那么，
用户行为的序列空间表示为(A × C)∗。通过将该序列作为
辅助用户特征纳入，点击率预测模型可以利用关于用户偏好
和历史参与度的宝贵见解，最终提高预测精度。

4 所提出的方法

我们介绍GenCTR，一种新颖的两阶段生成式方法，用于增强点击率（CTR）
预测。GenCTR通过预训练利用用户行为序列来增强传统骨干点击率模型。

4.1 预训练

第一步是预训练生成式模型。

生成式模型 给定用户的行为序列，或其中的一个连续片段，生成式
模型旨在通过生成一个捕捉其潜在特征的嵌入来预测序列中的下一个物品。
该嵌入是物品的数值表示，通常
用h ∈ Rd表示，其中d表示嵌入维度。

条件生成式模型
我们不单独依赖物品的历史序列，而是
提出通过增强生成式模型来考虑与下一个物品相关的辅助信息
以改善行为序列的利用。在预测下一个元素时，该辅助信息
与前面的物品一起作为额外输入。本质上，我们的模型
变得条件依赖于这个传入的辅助信息。在数学上，我们可以将
生成式模型表示为函数g：(A × C)∗ × C → Rd。

3

条件自注意力解码器 我们的条件生成式模型实现为
自注意力解码器的一个变体。在传统的自注意力解码器中，查询向量源自
最近的物品，而键和值向量源自它之前的物品。
相比之下，我们的条件自注意力解码器采用了一种不同的方法——其查询向量
源自与下一个物品相关的辅助信息，而键和值向量则来自
当前物品的前面物品（包括最近的物品）。我们将该方法
称为条件自注意力解码器。图1（左）展示了生成式模型的整体
架构。我们的条件自注意力解码器采用仅解码器的多层多头
变换器架构。值得注意的是，位置嵌入是一个16维的嵌入，其
长度为用户行为的长度。

条件负采样 负采样在训练自注意力解码器中发挥重要作用，以提高模型区分能力。
对应于条件自注意力机制，负实例从与下一个物品具有相同辅助信息
的物品中采样。

损失函数 考虑一个用户行为序列。对于每个物品，解码器掩码该物品
（保留用于查询向量的辅助信息）及其后续物品。然后它预测
当前物品的嵌入。真实标签就是物品本身，而负标签
使用条件负采样进行采样。然后对每个预测计算交叉熵
损失。这个过程产生一系列损失，总数量与序列
长度匹配。

4.2

集成

这一步将生成式模型与骨干点击率预测模型无缝融合。采用
两种关键的集成机制：

参数共享 两种模型通常共享物品标识符及其辅助信息等特征。
这种共同基础被利用，通过将生成式模型学习到的这些共享特征的
嵌入参数用于骨干点击率预测模型。这有效
地允许骨干模型受益于生成式模型对这些特定特征捕捉到的
见解。

模型集成 除了参数共享，模型直接连接，形成
集成的点击率预测模型。生成式模型的输出作为额外输入
馈送到骨干点击率预测模型。这用生成式模型识别的潜在上下文和
潜在交互丰富了特征空间。具体来说，给定输入元组(x, s, c)，
其中x ∈ X是骨干点击率模型的原始特征，s ∈ (A × C)∗是用户
行为序列，c是目标物品的辅助信息，集成的点击率
模型的输出是ŷ(x, g(s, c))，其中g(s, c)表示生成式模型的输出。
集成模型在训练数据S = {(x, s, c, y)}上进行端到端训练，解决以下
优化问题：

min
θ

1
|S|

∑
(x,s,c,y)∈S

ℓ(y, ŷ(x, g(s, c; θ)); θ),

(1)

其中ℓ(·, ·)是交叉熵损失，θ包含骨干点击率预测
模型和生成式模型的所有参数。

注意，骨干模型已经利用了来自s和c的信息。生成式模型
通过捕捉序列和上下文数据中的潜在交互和潜在模式来进一步丰富
这种表示。

5 实验

由于没有公开数据集同时包含搜索查询和用户行为信息
用于点击率预测（据我们所知），我们从一个名为GCTR的新数据集进行收集，

4

表1：GCTR上的实验结果。每个实验独立重复5次
（均值±标准差）。↑表示数值越大越好；↓表示数值越小越好。

序号 点击率模型
骨干
1
骨干
2
+解码器
3
+解码器
4
+解码器
5
+解码器
6
+解码器
7
+解码器
8

骨干

集成设置

预训练设置

PS

PS
PS+MI
PS+MI
PS+MI
PS+MI

CS+CD

CS+CD
CS+CD
CS+SD
RS+CD
RS+SD

DNN

DCN V2

DCN V2&TA

AUC↑
0.6072±0.0002
0.6258±0.0003
0.6124±0.0003
0.6310±0.0004
0.6331±0.0007
0.6328±0.0004
0.6259±0.0003
0.6155±0.0006

Logloss↓
0.2381±0.0000
0.2373±0.0001
0.2375±0.0001
0.2371±0.0002
0.2368±0.0002
0.2368±0.0001
0.2363±0.0000
0.2374±0.0001

AUC↑
0.6088±0.0000
0.6161±0.0013
0.6141±0.0002
0.6292±0.0003
0.6329±0.0003
0.6304±0.0004
0.6258±0.0002
0.6123±0.0002

Logloss↓
0.2397±0.0000
0.3083±0.0082
0.2373±0.0000
0.2371±0.0002
0.2368±0.0002
0.2369±0.0001
0.2362±0.0000
0.2376±0.0000

AUC↑
0.6136±0.0001
0.6292±0.0006
0.6150±0.0002
0.6299±0.0008
0.6323±0.0007
0.6318±0.0003
0.6263±0.0016
0.6142±0.0004

Logloss↓
0.2373±0.0000
0.2363±0.0000
0.2372±0.0000
0.2362±0.0001
0.2361±0.0000
0.2361±0.0000
0.2364±0.0002
0.2377±0.0000

该数据集来自全球最大电子商务平台之一的在线搜索广告日志。GCTR
数据集分为四个不同的部分：训练集、测试集、预训练集和
训练采样表。训练集由跨越连续三天的日志采样构建而成，
测试集来自随后一天的日志。
数据集包含详细的物品信息、用户画像以及用户最近200次点击物品及
相应辅助信息。

除了主要的训练集和测试集，我们还整理了一个预训练集和一个
类别物品采样表。预训练集
通过对训练集中的用户条目进行去重生成，突出用户画像和之前的浏览活动。类别物品采
样表则记录了训练集中出现的所有产品类别与其对应
物品的映射。我们后续的离线实验是在此
数据集上进行的，为验证各种点击率模型的有效性提供了坚实基础。

5.1 评估指标

采用两个指标来评估离线性能：AUC和LogLoss。AUC，即接收者操作特征曲线下面积，
是点击率（CTR）预测中最重要的离线指标，衡量整体排序准确性。LogLoss，表示模型的
交叉熵损失，反映其分类效能。此外，对于在线业务影响，
我们测量点击率和RPM（千次展示收入）以验证我们的方法在实际广告生态系统中的
实用效用。

5.2 实验设置

为了评估我们提出方法的有效性，我们使用三种不同的
骨干模型进行点击率预测实验：

1. DNN：一种基本的神经架构，包括一个嵌入层，后跟一个维度为256和128的多层感知机（MLP），
使用ReLU激活函数。

2. DCN V2（Wang等人，2021）：该模型是点击率预测中不融入用户行为的最新最优（SOTA）
模型。我们设置模型具有3个专家，交叉层深度为3，
以及秩为16的低秩交叉层。

3. DCN V2 & TA：在DCN v2的基础上，我们集成目标注意力（TA）来建模用户行为，
利用其在最新最优的精确搜索单元（ESU）中长期用户
行为分析中的已证实效（Pi等人，2020；Chang等人，2023；Chen等人，2021）。这种融合旨在验证
我们的方法对已经融入用户行为的点击率模型的影响。

所有模型在NVIDIA P40 GPU上使用TensorFlow进行训练。我们在
实验设置中保持统一，使用单层解码器，将所有特征的嵌入维度设置为16，
并使用学习率为0.001的Adam优化器。用户行为序列
被截断为最近200次交互。模型进行三个epoch的预训练和一个epoch的点击率
训练。我们的实验结果总结在表1中，配置
表示如下：

1. +解码器：在最终判别器层之前引入预训练模型结构并将其与骨干融合。2. PS：与预训练嵌入进行参数共享。3. MI：继承
预训练模型结构和参数。4. CS：预训练中的类别条件负采样。

5

5. RS：预训练中的随机负采样。6. CD：预训练中的类别条件解码器。7. SD：类似于SASRec的无条件自注意力解码器。

我们的设置旨在严格评估各种模型组件和训练
策略对点击率预测任务的影响。

5.3 实验分析

参数继承的效果 实验1、2、3和4旨在评估共享预训练嵌入参数对点击率模型的影响。
在所有点击率模型中观察到了显著改进，而无需向点击率模型引入任何额外参数。
这证实了通过我们的预训练过程细化的物品表示对点击率估计具有显著的正向效果。

模型继承的效果 对比实验2和5的分析强调了模型继承的独特优势。通过比较实验4和5，我们控制了模型
复杂度并确认了性能提升不仅仅归因于参数数量的增加，
而是源于模型继承的战略优势。

基于嵌入和中间参数整合带来的显著收益，
我们深入研究了预训练模型设计对点击率预测的影响。实验5
代表我们的最终模型配置，作为后续旨在阐明这一方面的消融
研究的基准。

使用负采样方法进行预训练的效果 实验5和7之间以及6和8之间的比较表明，类别条件负采样（CS）在不同预训练上下文中始终
优于随机负采样（RS）。增强的
性能归因于CS对类别内物品差异的细致关注，这对于
细化表示和提高点击率预测精度至关重要。

使用解码器建模进行预训练的有效性 实验5、6和7、8之间的性能对比揭示，我们的条件解码器（CD）在所有骨干模型上都优于自注意力解码器
（SD）。我们认为，CD聚焦于物品特定特征——
而非点击率预测中可获取的类别级特征——使预训练
目标与点击率建模任务更加一致。这种一致性确保预训练
阶段集中于物品的内在属性，从而使预训练模型
对点击率预测更具相关性和有效性。

在线A/B测试 我们于2023年10月在我们的实时搜索广告平台上通过为期两周的在线A/B测试评估了GenCTR。
该测试旨在评估模型在真实流量条件下的性能。
值得注意的是，仅通过参数共享（PS）引入预训练嵌入
就导致点击率（CTR）显著提升0.56%和千次展示收入（RPM）提升1.25%。
在通过模型继承（MI）集成完整预训练模型后，
点击率额外增加了1.32%和RPM增加了1.66%。
这些改进证实了两种继承策略的有效性。

需要强调的是，用于比较的基线模型是此前在真实流量上服务的
模型，该模型已经表现出优越的性能。我们的模型现已
成功集成并正在服务于搜索广告，进一步证明了我们研究的
实用价值。

6 结论

在本文中，我们提出了一种新颖的策略，利用预训练生成式模型来
利用用户行为数据以增强点击率预测模型。我们的方法的特点是
对预训练阶段进行定制化设计，旨在提取和利用用户交互的复杂模式
为点击率预测提供信息。通过大量离线实验对我们方法的经验验证
已确认其有效性，在预测精度上有了显著提高。
此外，其在搜索广告系统中的成功部署展示了

6

该模型的实用适用性。总之，我们的工作为未来研究探索
生成式预训练在点击率建模中的全部潜力开辟了途径。

参考文献

Jianxin Chang, Chenbin Zhang, Zhiyi Fu, Xiaoxue Zang, Lin Guan, Jing Lu, Yiqun Hui, Dewei
Leng, Yanan Niu, Yang Song, and Kun Gai. Twin: Two-stage interest network for lifelong user
behavior modeling in ctr prediction at kuaishou. In Proceedings of the 29th ACM SIGKDD Con-
ference on Knowledge Discovery and Data Mining (KDD), pp. 3785–3794, 2023.

Qiwei Chen, Changhua Pei, Shanshan Lv, Chao Li, Junfeng Ge, and Wenwu Ou. End-to-end user
behavior retrieval in click-through rateprediction model. CoRR, 2021.

Heng-Tze Cheng, Levent Koc, Jeremiah Harmsen, Tal Shaked, Tushar Chandra, Hrishi Aradhye,
Glen Anderson, Greg Corrado, Wei Chai, Mustafa Ispir, et al. Wide & deep learning for recom-
mender systems. In Proceedings of the 1st workshop on deep learning for recommender systems,
pp. 7–10, 2016.

Huifeng Guo, Ruiming Tang, Yunming Ye, Zhenguo Li, and Xiuqiang He. Deepfm: a factorization-
machine based neural network for ctr prediction. CoRR, 2017.

Balázs Hidasi, Alexandros Karatzoglou, Linas Baltrunas, and Domonkos Tikk. Session-based rec-
ommendations with recurrent neural networks. CoRR, 2015.

Wang-Cheng Kang and Julian McAuley. Self-attentive sequential recommendation. In 2018 IEEE
international conference on data mining (ICDM), pp. 197–206, 2018.

Lingwei Kong, Lu Wang, Xiwei Zhao, Junsheng Jin, Zhangang Lin, Jinghe Hu, and Jingping Shao.
In
LOVF: layered organic view fusion for click-through rate prediction in online advertising.
International ACM SIGIR Conference on Research and Development in Information Retrieval
(SIGIR), pp. 2139–2143, 2023.

Jianxun Lian, Xiaohuan Zhou, Fuzheng Zhang, Zhongxia Chen, Xing Xie, and Guangzhong Sun.
In
xdeepfm: Combining explicit and implicit feature interactions for recommender systems.
Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data
mining (KDD), pp. 1754–1763, 2018.

Qi Pi, Guorui Zhou, Yujing Zhang, Zhe Wang, Lejian Ren, Ying Fan, Xiaoqiang Zhu, and Kun
Gai. Search-based user interest modeling with lifelong sequential behavior data for click-through
In Proceedings of the 29th ACM International Conference on Information &
rate prediction.
Knowledge Management (CIKM), pp. 2685–2692, 2020.

Yuhai Song, Lu Wang, Haoming Dang, Weiwei Zhou, Jing Guan, Xiwei Zhao, Changping Peng,
Yongjun Bao, and Jingping Shao. Underestimation refinement: A general enhancement strategy
for exploration in recommendation systems. In The 44th International ACM SIGIR Conference
on Research and Development in Information Retrieval (SIGIR), pp. 1818–1822, 2021.

Fei Sun, Jun Liu, Jian Wu, Changhua Pei, Xiao Lin, Wenwu Ou, and Peng Jiang. Bert4rec: Sequen-
tial recommendation with bidirectional encoder representations from transformer. In Proceedings
of the 28th ACM International Conference on Information and Knowledge Management (CIKM),
pp. 1441–1450, 2019.

Jiaxi Tang and Ke Wang. Personalized top-n sequential recommendation via convolutional sequence
embedding. In Proceedings of the eleventh ACM international conference on web search and data
mining (WSDM), pp. 565–573, 2018.

Lu Wang, Yuhai Song, Zhe Wang, Haoxiang Wang, Yu Li, Weiwei Zhou, Haoming Dang, Mona
Shao, Xiwei Zhao, Zhangang Lin, et al. Pluggable deep thompson sampling with applications to
recommendation. In SIAM International Conference on Data Mining (SDM), pp. 64–72, 2023.

Ruoxi Wang, Rakesh Shivanna, Derek Cheng, Sagar Jain, Dong Lin, Lichan Hong, and Ed Chi. Dcn
v2: Improved deep & cross network and practical lessons for web-scale learning to rank systems.
In Proceedings of the web conference (WWW), pp. 1785–1797, 2021.

7

Le Wu, Xiangnan He, Xiang Wang, Kun Zhang, and Meng Wang. A survey on neural recommen-
dation: From collaborative filtering to content and context enriched recommendation. CoRR,
abs/2104.13030, 2021.

Kaifu Zheng, Lu Wang, Yu Li, Xusong Chen, Hu Liu, Jing Lu, Xiwei Zhao, Changping Peng,
Zhangang Lin, and Jingping Shao. Implicit user awareness modeling via candidate items for ctr
prediction in search ads. In Proceedings of the ACM Web Conference (WWW), pp. 246–255, 2022.

Guorui Zhou, Xiaoqiang Zhu, Chenru Song, Ying Fan, Han Zhu, Xiao Ma, Yanghui Yan, Junqi Jin,
Han Li, and Kun Gai. Deep interest network for click-through rate prediction. In Proceedings of
the 24th ACM SIGKDD international conference on knowledge discovery & data mining (KDD),
pp. 1059–1068, 2018a.

Guorui Zhou, Xiaoqiang Zhu, Chenru Song, Ying Fan, Han Zhu, Xiao Ma, Yanghui Yan, Junqi Jin,
Han Li, and Kun Gai. Deep interest network for click-through rate prediction. In Proceedings
of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining
(KDD), pp. 1059–1068, 2018b.

8
