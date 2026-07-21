# Tenrec：面向推荐系统的大规模多用途基准数据集

> Guanghu Yuan, Fajie Yuan, Yudong Li, Beibei Kong et al. | Westlake University & Chinese Academy of Sciences

本文介绍了 Tenrec：面向推荐系统的大规模多用途基准数据集。核心内容：


关键发现：


¹腾讯
²西湖大学
³中国科学院深圳先进技术研究院
⁵中山大学
⁴中国科学技术大学

gh.yuan0@gmail.com, yuanfajie@westlake.edu.cn, ustclsj@mail.ustc.edu.cn
{lei.chen,min.yang}@siat.ac.cn, yuchy35@mail.sysu.edu.cn
{elsonli,echokong,harryyfhu,gavinzli, henrysxu,tigerqie}@tencent.com


---

## 摘要

现有的推荐系统（RS）基准数据集要么是小规模创建的，要么包含非常有限形式的用户反馈。基于此类数据集评估的RS模型往往缺乏大规模实际应用的价值。在本文中，我们描述了Tenrec*，一个新颖且公开可用的RS数据集合，它记录了来自四个不同推荐场景的各种用户反馈。具体来说，Tenrec具有以下五个特点：（1）大规模，包含约500万用户和1.4亿次交互；（2）不仅包含正向用户反馈，还包含真正的负向反馈（相对于单类推荐）；（3）包含跨四个不同场景的重叠用户和item；（4）包含多种类型的用户正向反馈，形式包括点击、点赞、分享和关注等；（5）包含用户ID和itemID之外的额外特征。我们通过在十个多样化推荐任务上运行每个任务的几个经典基线模型来验证Tenrec。Tenrec有潜力成为大多数流行推荐任务的有用基准数据集。我们的源代码、数据集和排行榜可在https://github.com/yuangh-x/2022-NIPS-Tenrec*获取。

## 1 引言

推荐系统（RS）旨在评估用户对尚未见过的item的偏好。深度学习（DL）的进展催生了广泛的新型复杂神经推荐模型。先前文献中已取得许多改进，然而，其中许多是在非基准数据集上或按现代标准在小规模数据集上进行评估的。这导致了RS社区中严重的可重复性和可信度问题。例如，[37]表明，先前论文中报道的许多"先进"基线在很大程度上是次优的，甚至不如十多年前提出的旧基线——vanilla矩阵分解（MF）[23]。[36]进一步证明，在精心设置下，点积优于学习到的相似度，例如使用多层感知机（MLP）[15]。最近，[9, 24, 44, 20, 8, 7, 1]也从数据集和实验设置角度质疑了RS中一些公认的进展。

大规模高质量的数据集对于加速一个领域的研究具有重要影响，例如计算机视觉（CV）领域的ImageNet [11]和自然语言处理（NLP）领域的GLUE [47]。然而，由于安全或隐私问题，研究人员通常难以访问大规模的真实世界数据集来研究推荐问题。尽管如此，仍然存在几个用于常规推荐任务的流行数据集。例如，MovieLens*（ML）数据集，包括ML-100K、ML-1M和ML-10M等，已成为评分预测[23]任务的稳定基准数据集。其他流行数据集，包括用于电影推荐的Netflix [33]、用于位置推荐的Yelp [6]、用于产品推荐的Amazon [14]、用于新闻推荐的Mind [53]、用于歌曲推荐的Last.fm [60]和Yahoo! Music [13]，也频繁出现在文献中。这些数据集的一个主要缺点是用户反馈数据形式非常有限，例如，大多数只包含一种类型的用户反馈（评分、点击或观看中的一种），或者仅从一个推荐场景收集。这严重限制了真实世界推荐系统的研究范围。

为了促进多样化的推荐研究，我们提出了Tenrec，一个大规模、多用途的真实世界数据集。与现有的公开数据集相比，Tenrec具有几个优点：（1）它包含来自四个不同真实世界推荐场景的重叠用户/item，可用于研究跨域推荐（CDR）和迁移学习（TF）方法；（2）它包含多种类型的正向用户反馈（例如点击、点赞、分享、关注、阅读和收藏），可用于研究多任务学习（MTL）问题；（3）它同时具有正向用户反馈和真正的负向反馈，可用于研究更实际的点击率（CTR）预测场景；（4）它包含身份信息（即用户ID和itemID）之外的额外用户和item特征，可用于上下文/基于内容的推荐。

由于这些优势，Tenrec可用于评估广泛的推荐任务。在本文中，我们通过十个推荐任务检验其特性，包括（1）CTR预测[68, 69]，（2）基于会话的推荐[18]，（3）MTL推荐[30]，（4）CDR推荐[67]，（5）用户画像预测[59]，（6）冷启动推荐[46]，（7）终身用户表示学习[61]，（8）模型压缩[43]，（9）模型训练加速[48]，以及（10）模型推理加速[2]。除了这些任务，我们可以轻松地整合上述部分特性来提出额外或新的任务。据我们所知，Tenrec是迄今为止RS最大的数据集之一，涵盖了大多数推荐场景和任务。我们发布所有数据集和代码以促进可重复性并推动新的推荐研究。

## 2 数据集描述

Tenrec是一个为多种推荐任务开发的数据集套件，收集自腾讯*的两个不同信息流推荐平台，即QQ BOW（QB）和QQ KAN（QK）。* QK/QB中的item可以是新闻文章或视频。请注意，文章和视频推荐模型是使用不同的神经网络和特征分别训练的。因此，我们可以认为Tenrec由来自总共四个场景的用户反馈组成，即QK-video、QK-article、QB-video和QB-article（见图1）。我们收集了2021年9月17日至12月7日期间QK/QB的用户行为日志。过程如下：我们首先从QK-video数据库中随机抽取约502万用户，要求每个用户至少有5次视频点击行为；然后，我们提取他们的反馈（约4.93亿条），包括正向反馈（即视频点击、分享、点赞和关注）和负向反馈（有曝光但无用户操作）；最后，我们获得约1.42亿次点击、1000万次点赞、100万次分享和86万次关注，以及375万个视频。此外，还有用户的年龄和性别特征，以及item的视频类型特征。我们对QK-article、QB-video和QB-article执行类似的数据提取策略。在本文中，我们将QK-video视为主场景，其他三个作为辅助场景，用于各种CDR或TF任务。数据集统计信息如表1所示。

**数据分布。** 图2（a）和（b）显示了QK-videoitem在点击行为方面的流行度。显然，item流行度遵循典型的长尾分布，这在先前的推荐文献[34]中已被广泛报道。（c）显示了会话长度分布，其中长度在[0−20]的会话数量占所有会话的53%。在其他三个数据集上可以观察到类似的分布，因此省略。

**数据重叠。** Tenrec包含跨四个场景的一部分重叠用户和item。关于重叠用户，我们在QK-video与QK-article、QB-video、QB-article之间进行计算，因为QK-video覆盖了最多数量的用户、item和交互。具体来说，QK-video和QK-article之间的重叠用户数为268,207，QK-video和QB-video之间为3,261，QK-video和QB-article之间为58。关于重叠item，QK-video和QB-video之间有78,482个视频重叠。重叠的用户和item可以通过其唯一ID进行关联。这一特性使得Tenrec非常适合研究TF和CDR任务。

**用户反馈。** Tenrec与现有的推荐数据集不同，后者只包含一种类型的用户反馈，无论是隐式反馈还是显式评分。因此，这些数据集中用户偏好的程度无法很好地反映。如表1所示，QK-video和QB-video包含四种类型的正向反馈，其中点击行为数量最多，其次是点赞、分享和关注。这一发现是直观的，因为点赞、分享和关注通常表示比点击更高的偏好。同样，QK-article包含两种额外类型的偏好反馈，即文章阅读和收藏行为。除了各种正向反馈，Tenrec还包括真正的负向反馈——即一个item呈现给用户，但用户没有点击它。这种负向反馈使得Tenrec更适合CTR预测，而大多数现有数据集只涉及正向反馈。

**特征。** QK/QB-video中每个实例的格式为{用户ID, itemID, 点击, 点赞, 分享, 关注, 视频类别, 观看次数, 用户性别, 用户年龄}。请注意，时间戳信息已被腾讯要求删除，但我们按时间顺序呈现所有交互行为。点击、点赞、分享、关注是二进制值，表示用户是否有此类操作。观看次数是用户观看视频的次数。用户ID、itemID、用户性别、用户年龄已因隐私问题而被脱敏。用户年龄已被分箱，每个箱代表一个十年区间。

QK/QB-article中每个实例的格式为{用户ID, itemID, 点击, 点赞, 分享, 关注, 阅读, 收藏, 点击次数, 点赞次数, 评论次数, 曝光次数, 阅读百分比, 二级类别, 一级类别, item评分1, item评分2, item评分3, 阅读时长}。后缀"*_count"表示每篇文章*操作的总数。read_percentage表示用户阅读文章的百分比。category_first和category_second是文章的类别，其中"_first"是粗粒度类别（例如体育、娱乐、军事等），"_second"是细粒度类别（例如NBA、世界杯、科比等）。item_score1、item_score2、item_score3表示通过不同评分系统评估的item质量。read_time是阅读的时长。

## 3 实验评估

在本节中，我们描述Tenrec在十个不同推荐任务中的应用。我们简要介绍每个任务，并报告流行或最先进基线的结果。我们通过参考官方代码、DeepCTR*和Recbole*的代码实现了关键基线。

### 3.1 CTR预测

CTR预测是一个经典的推荐任务，其目标是预测用户是否会点击推荐item。我们在QK-video数据集的采样版本上执行此任务，该版本包含100万随机抽取的用户，称为QK-video-1M。完整QK-video数据集的更多结果见附录表2。

进行采样的原因仅仅是因为在原始数据集上为所有十个任务搜索所有基线的超参数需要太多的计算和训练时间。我们将为原始数据集创建一个公共排行榜。

**数据集。** 我们保留所有正向反馈，并以1:2的正/负采样比例抽取一部分真正的负向反馈。通过这样做，我们总共获得1,948,388个item和86,642,580次交互，稀疏度为96.7%。然后我们将数据按8:1:1分割为训练集、验证集和测试集。*训练示例包含以下特征：用户ID、itemID、性别、年龄、视频类别以及用户过去10个点击的item。我们对类别特征应用嵌入。

**基线和超参数。** 有许多用于CTR预测的深度学习基线。其中，Wide & Deep [3]、DeepFM [26]、NeuralFM [15]、xDeepFM [26]、Attention FM（AFM）[54]、DCN [50]和DCNv2 [51]是文献[68, 51, 42]中一些最知名且最强大的基线。最近的CTR基准[68]表明，许多最近的模型（例如InterHAt [25]、AFN+ [5]和LorentzFM [56]）并不显著优于这些流行的基线。

我们在验证集上通过评估嵌入大小（本文中记为d）在{16, 32, 64, 128}、隐藏单元数（记为f）在{64, 128, 256}和学习率（记为\eta）在{1e−3, 5e−4, 1e−4, 5e−5, 1e−5}中进行超参数搜索。最后，我们将\eta设为5e−5，所有方法的d设为32，将DeepFM、NFM、Wide & Deep和xDeepFM的f设为256，AFM的注意力因子设为8。至于批大小（记为b），使用较大的值通常效果略好。我们将其设为所有模型的4096。我们发现正则化\lambda对结果没有明显影响，可能是因为训练样本数量大得多，因此我们将其设为5e−5。根据验证集的最优结果，我们将AFM的层数h设为1，其他模型设为2。

**结果。** 表2显示了不同方法在QK-video-1M数据集上以曲线下面积（AUC）[35]为指标的结果。我们观察到，这些CTR模型的整体表现非常相似。相比之下，NFM表现最好，而Wide & Deep表现最差，差异约为0.5%。

### 3.2 基于会话的推荐

基于会话的推荐（SBR），也称为序列推荐，旨在根据会话中先前交互item的序列预测下一个item。SBR的一个关键特征是在训练过程中显式建模交互顺序，这通常会产生更好的Top-N结果。

**数据集。** 我们在此报告在QK-video-1M上评估的基线结果。完整QK-video数据集的更多结果见附录表3。遵循常见做法[60]，我们简单过滤掉长度小于10的会话。鉴于平均会话长度为28.34，我们将最大会话长度设为30。长度小于30的会话将被零填充，否则只保留最近的30次交互。预处理后，我们获得928,562个用户、1,189,341个item和37,823,609次点击交互。我们保留会话中的最后一个item用于测试，倒数第二个用于验证，其余用于训练。

**基线和超参数。** 我们使用四个被高度引用的基线来验证Tenrec：基于RNN的GRU4Rec [18, 45]、基于CNN的NextItNet [60]、基于自注意力的SASRec [21]和BERT4Rec [41]。在原始论文中，这些模型采用了不同的损失函数、采样方法和数据增强[45]，在评估网络架构时不可比。因此，为了进行严格的比较，我们为GRU4Rec、NextItNet和SASRec应用标准的自回归[60, 58]训练方式，使用交叉熵损失和softmax函数——即只有它们的网络架构不同。我们使用原始的掩码标记损失训练BERT4Rec，用于与SASRec比较，因为两者都应用了多头自注意力网络架构。

超参数的搜索方式与上述类似。GRU4Rec的\eta设为5e−4，其他三个模型设为1e−4。所有模型的b和\lambda设为32和0。NextItNet和BERT4Rec的d、f和h设为128、128和16，而GRU4Rec和SASRec设为64、64和8。这仅仅是因为SASRec和GRU4Rec在验证集上使用更大的d、f和h时会产生较差的结果。SASRec和BERT4Rec的注意力头数设为4，比1和2的效果稍好。对于BERT4Rec，在{10%, 20%, 30%, 40%, 50%}中搜索后，我们随机掩码每个会话中的30%的item。

**结果。** 我们使用标准的Top-N排序指标评估所有基线，即命中率（HR）[60]和归一化折损累计增益（NDCG）[21]。N设为20。表3显示了四个基线的结果。观察结果如下：（1）单向模型GRU4Rec、NextItNet和SASRec在HR@20和NDCG@20上优于双向的BERT4Rec。这与许多近期工作[66, 28, 10]一致。（2）在相同的训练方式下，三个单向模型表现相似——具有时序CNN架构的NextItNet略优于SASRec和GRU4Rec。我们这里的结果与许多先前出版物[41, 29, 55]不同，在那些工作中，表现最佳的网络架构轻松获得比经典基线超过50%的提升。

### 3.3 多任务学习推荐

多任务学习（MTL）旨在同时学习两个或多个任务，同时最大化其中一个或所有任务的性能。在这里，我们尝试对点击和点赞的用户偏好进行建模，而不仅仅是其中一种。我们使用与CTR预测相同的数据集和分割策略。区别在于，对于MTL，我们有两个输出目标，一个用于点击，另一个用于点赞。鉴于Tenrec包含多种类型的用户反馈，可以利用更多目标构建更具挑战性的MTL任务，例如使用QK-article进行三任务、四任务甚至六任务（即通过组合点击、点赞、分享、关注、阅读、收藏）学习。

**基线和超参数。** 我们在Tenrec上评估了两个强大的MTL基线，即MMOE [30]和ESMM [31]。此外，我们还展示了仅优化点赞或点击目标的单任务学习结果。经过超参数搜索，我们将\eta、d、f、b和h分别设为1e−4、32、128、4096、2。

**结果。** 表4显示了四种方法在QK-video-1M数据集上以曲线下面积（AUC）[35]为指标的结果。如我们所见，ESMM在点击和点赞预测方面都优于MMOE。MMOE并未显著优于单目标优化（SOO）。尽管如此，MMOE可以为两个或多个目标实现良好的权衡，而SOO只关注一个目标。

### 3.4 迁移学习推荐

迁移学习（TF）——通过先预训练然后微调——已成为NLP [12]和CV [19]中的事实标准做法。然而，对于推荐任务[40]，执行TF的最佳方式仍未知。在本节中，我们简单地探索一种基本方式：先在源域预训练一个SBR模型（即NextItNet和SASRec），然后将其隐藏层（即CNN和自注意力）的参数迁移到目标域的相同模型（其他参数随机初始化）。我们将在后续章节中通过考虑数据重叠来研究其他类型的迁移学习。

**数据集、基线和超参数。** 我们使用SBR任务中的相同数据集作为源数据集，使用QB-video点击反馈作为目标数据集。关于基线模型，我们使用NextItNet和SASRec来评估TF效果。除了\eta，其他超参数与SBR任务中描述的完全相同。在QB-video上，SASRec和NextItNet的\eta分别设为1e−4和5e−4。

**结果。** 表5显示了有预训练和无预训练的比较结果。关键观察结果是，NextItNet和SASRec在预训练下都产生了更好的Top-N结果。这表明从大型训练数据集学到的隐藏层参数可以在训练数据不足时为类似推荐任务用作良好的初始化。

### 3.5 用户画像预测

用户画像是个性化RS的重要特征，特别是对于冷/新用户的推荐。最近，[59, 2, 4, 39]证明，通过建模用户在他们有更多行为的平台上收集的点击行为，可以高精度地预测用户画像。

**数据集、基线和超参数。** 我们在QK-video-1M数据集上进行实验。首先，我们移除没有用户画像特征的实例，得到739,737个具有性别特征的实例和741,652个具有年龄特征的实例。我们将每个数据集按8:1:1分割为训练集、验证集和测试集。我们评估了五个基线模型：标准DNN模型、PeterRec [59]和BERT4Rec，各有无预训练两种。PeterRec和BERT4Rec的预训练和微调框架严格遵循[59]。请注意，对于PeterRec，我们使用单向NextItNet作为主干网络，而BERT4Rec是双向的。DNN、PeterRec和BERT4Rec的\eta分别设为1e−4、5e−5、1e−4。其他超参数与第3.2节中设置相同。

**结果。** 表6显示了五个基线模型在标准分类准确率（ACC）方面的结果。首先，PeterRec和BERT4Rec优于DNN，表明CNN和自注意力网络在对用户行为序列建模时更强大。其次，有预训练的PeterRec和BERT4Rec比从头开始训练的自身效果更好。

### 3.6 冷启动推荐

冷启动是推荐任务中一个重要但尚未解决的挑战。Tenrec的一个主要优势是用户重叠和item重叠信息都可用。在这里，我们主要通过应用迁移学习来研究冷用户问题。与第3.4节不同，嵌入层和隐藏层都可以为重叠用户进行迁移。

**数据集、基线和超参数。** 我们将QK-video作为源数据集，QK-article作为目标数据集。在实践中，存在几种不同的冷用户推荐设置。例如，在大多数广告推荐系统中，用户往往只有很少的点击交互，而热用户和冷用户可能共存于其他常规推荐系统中。因此，我们对几种不同设置进行了评估。由于冷用户在我们的数据预处理阶段已被移除，这里我们通过从QK-video和QK-article之间的重叠用户中提取他们最近的5次交互来模拟一个简单的冷用户场景。训练集、验证集和测试集按8:1:1分割。我们在附录表5中提供了其他冷用户设置的更多结果。我们使用PeterRec和BERT4Rec作为基线，因为它们在文献[59, 61, 2, 41]中具有最先进的性能。具体来说，我们首先在QK-video数据集上的所有用户序列行为上执行自监督预训练，然后在QK-video和QK-article数据集之间的这些重叠用户的交互上微调模型，以实现迁移学习。更多细节见[59]。除了微调阶段的\eta，所有超参数与SBR任务中描述的完全相同。我们在微调期间将BERT4Rec和PeterRec的\eta分别设为1e−3和5e−3。

**结果。** 表7显示了冷用户推荐的结果。首先，我们发现PeterRec和BERT4Rec在预训练下都取得了显著的改进。其次，有预训练的BERT4Rec显示出比PeterRec更好的结果。这与NLP领域的研究一致，其中双向编码器比单向编码器能实现更好的迁移学习。

### 3.7 终身用户表示学习

当将神经推荐模型从一个域迁移到另一个域时，为初始任务训练的参数往往会被修改以适应新任务。因此，推荐模型将失去再次服务原始任务的能力，这被称为灾难性遗忘[22]。[61]提出了第一个"一个模型服务所有"的学习范式，旨在仅使用一个主干网络构建通用用户表示（UR）模型。在本节中，我们研究跨四个场景迁移用户偏好的终身学习（LL），即从QK-video到QK-article再到QB-video再到QB-article。

**数据集、基线和超参数。** 由于GPU内存问题，我们从QK-video-1M中随机抽取50%的用户作为任务1的数据集。然后我们使用QK-article、QB-video和QB-article进行后续任务。鉴于TF和LL更有利于数据稀缺场景，我们处理QK-article使其每个用户最多保留三次交互。对于QB-video和QB-article，我们保留其原始数据集，因为用户和点击量要少得多。使用Conure [61]作为基线模型，以NextItNet和SASRec作为主干网络。出于比较目的，我们报告没有过去任务预训练（PT）的Conure在任务2、3、4上的结果。模型无关的超参数与之前类似的方式搜索。任务1、2和3的剪枝率分别设为60%、33%、25%。

**结果。** 表8显示了使用持续学习的用户表示进行推荐的结果。可以清楚地看到，由于过去任务的PT，Conure在任务2、3和4上提供了性能改进。例如，Conure-NextItNet在任务2上将NDCG@20从0.0081提高到0.0095，在任务3上从0.0160提高到0.0167，在任务4上从0.0902提高到0.1074。

### 3.8 模型压缩

模型压缩使得将大型神经模型部署到有限容量的设备（如GPU和TPU（张量处理单元））成为可能。对于RS模型，嵌入层的参数量可以轻松达到数亿到数十亿级别。例如，[27]最近设计了一个超大型推荐模型，拥有高达100万亿个参数。

**数据集、基线和超参数。** 我们对SBR模型执行参数压缩，并使用与第3.2节相同的数据集。尽管具有显著的研究和实用价值，但很少有工作研究推荐任务的参数压缩技术。这里我们报告CpRec [43]框架的结果，这是文献中最先进的基线。我们以NextItNet和SASRec作为主干模型实例化CpRec。超参数与第3.2节完全相同。我们将item集划分为3个聚类，根据[43]按流行度排序，划分比例为25%：50%：25%。

**结果。** 表9显示，CpRec将NextItNet和SASRec压缩到其原始大小的三分之二，准确率下降约2%。

### 3.9 模型训练加速

该任务旨在加速非常深的推荐模型的训练过程。与浅层CTR模型不同，SBR模型可以深得多。最近，[48]揭示，SBR模型如NextItNet和SASRec可以加深到100层以获得最佳结果。*为了加速训练过程，他们提出了StackRec，它首先学习一个浅层模型，然后将这些浅层复制为深层模型的顶层。类似地，我们使用NextItNet和SASRec作为主干来评估StackRec。数据集和所有超参数与第3.2节保持一致。

**结果。** 表10显示了训练加速的结果。可以做出几个观察。（1）StackRec显著减少了NextItNet和SASRec的训练时间；（2）这种训练加速并没有导致推荐准确率下降。事实上，我们甚至发现，具有64层的Stack64-NextItNet的训练速度比具有16层的标准NextItNet快2倍。

### 3.10 模型推理加速

随着网络变深，一个实际问题出现了：推理成本也大幅增加，导致在线服务的高延迟。[2]表明，推荐模型中的用户可以分为困难用户和简单用户，其中为简单用户推荐item不需要通过整个网络。因此，作者提出了SkipRec，它在模型推理阶段自适应地决定哪个用户需要哪一层。我们在QB-video中验证模型推理加速的效果。我们通过分配NextItNet和SASRec作为主干来评估SkipRec。数据预处理和超参数与第3.2节相同。

**结果。** 表11显示了SkipRec在QB-video上的效果。我们看到SkipRec中的跳层策略可以大幅加速SBR模型的推理时间，例如NextItNet约23%，SASRec约32%。特别地，具有32层的SkipRec32-NextItNet仍然比具有16层的原始NextItNet更快。此外，SkipRec的推荐准确率与其原始网络保持相当。

## 4 结论

我们提出了Tenrec，这是最大且最多功能的推荐数据集之一，覆盖了多个具有各种类型用户反馈的真实世界场景。为了展示其广泛的实用性，我们在十个不同的推荐任务上进行了研究，并对文献中最先进的神经模型进行了基准测试。我们公开了代码、数据集和每个任务的排行榜，以促进推荐社区的研究，并希望Tenrec成为一个标准化的基准来评估这些推荐任务的进展。由于篇幅限制，我们尚未探索其全部应用潜力。未来，我们计划（1）研究Tenrec在更多真实世界推荐场景中的应用，例如具有重叠item的跨域推荐[38]、基于反馈的迁移学习（例如基于点击预测点赞和分享）[61]以及具有负采样的item推荐[57, 63]；（2）发布包含item模态信息数据的Tenrec未来版本，例如文章标题、描述和原始视频内容，以促进多模态推荐[49, 62]*。

## 致谢

This work is supported by the Research Center for Industries of the Future (No. WU2022C030) and Shenzhen Basic Research Foundation (No. JCYJ20210324115614039 and No. JCYJ20200109113441941).

## 参考文献

[1] Vito Walter Anelli, Alejandro Bellogín, Tommaso Di Noia, Dietmar Jannach, and Claudio Pomo. Top-n recommendation algorithms: A quest for the state-of-the-art. arXiv preprint arXiv:2203.01155, 2022.

[2] Lei Chen, Fajie Yuan, Jiaxi Yang, Xiang Ao, Chengming Li, and Min Yang. A user-adaptive layer selection framework for very deep sequential recommender models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, pages 3984–3991, 2021.

[3] Heng-Tze Cheng, Levent Koc, Jeremiah Harmsen, Tal Shaked, Tushar Chandra, Hrishi Aradhye, Glen Anderson, Greg Corrado, Wei Chai, Mustafa Ispir, et al. Wide & deep learning for recommender systems. In Proceedings of the 1st workshop on deep learning for recommender systems, pages 7–10, 2016.

[4] Mingyue Cheng, Fajie Yuan, Qi Liu, Xin Xin, and Enhong Chen. Learning transferable user representations with sequential behaviors via contrastive pre-training. In 2021 IEEE International Conference on Data Mining (ICDM), pages 51–60. IEEE, 2021.

[5] Weiyu Cheng, Yanyan Shen, and Linpeng Huang. Adaptive factorization network: Learning adaptive-order feature interactions. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pages 3609–3616, 2020.

[6] Seri Choi et al. An empirical study identifying bias in Yelp dataset. PhD thesis, Massachusetts Institute of Technology, 2021.

[7] Paolo Cremonesi and Dietmar Jannach. Progress in recommender systems research: Crisis? what crisis? AI Magazine, 42(3):43–54, 2021.

[8] Maurizio Ferrari Dacrema, Simone Boglio, Paolo Cremonesi, and Dietmar Jannach. A troubling analysis of reproducibility and progress in recommender systems research. ACM Transactions on Information Systems (TOIS), 39(2):1–49, 2021.

[9] Maurizio Ferrari Dacrema, Paolo Cremonesi, and Dietmar Jannach. Are we really making much progress? A worrying analysis of recent neural recommendation approaches. In Proceedings of the 13th ACM conference on recommender systems, pages 101–109, 2019.

[10] Alexander Dallmann, Daniel Zoller, and Andreas Hotho. A case study on sampling strategies for evaluating neural sequential item recommendation models. In Fifteenth ACM Conference on Recommender Systems, pages 505–514, 2021.

[11] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248–255. Ieee, 2009.

[12] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018.

[13] Gideon Dror, Noam Koenigstein, Yehuda Koren, and Markus Weimer. The yahoo! music dataset and kdd-cup'11. In Proceedings of KDD Cup 2011, pages 3–18. PMLR, 2012.

[14] ROHIT DWIVEDI et al. Product based recommendation system on amazon dataset. 2021.

[15] Xiangnan He and Tat-Seng Chua. Neural factorization machines for sparse predictive analytics. In Proceedings of the 40th International ACM SIGIR conference on Research and Development in Information Retrieval, pages 355–364, 2017.

[16] Xiangnan He, Kuan Deng, Xiang Wang, Yan Li, Yongdong Zhang, and Meng Wang. Lightgcn: Simplifying and powering graph convolution network for recommendation. In Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval, pages 639–648, 2020.

[17] Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu, and Tat-Seng Chua. Neural collaborative filtering. In Proceedings of the 26th international conference on world wide web, pages 173–182, 2017.

[18] Balázs Hidasi, Alexandros Karatzoglou, Linas Baltrunas, and Domonkos Tikk. Session-based recommendations with recurrent neural networks. ICLR, 2016.

[19] Minyoung Huh, Pulkit Agrawal, and Alexei A Efros. What makes imagenet good for transfer learning? arXiv preprint arXiv:1608.08614, 2016.

[20] Yitong Ji, Aixin Sun, Jie Zhang, and Chenliang Li. A critical study on data leakage in recommender system offline evaluation. arXiv preprint arXiv:2010.11060, 2020.

[21] Wang-Cheng Kang and Julian McAuley. Self-attentive sequential recommendation. In 2018 IEEE International Conference on Data Mining (ICDM), pages 197–206. IEEE, 2018.

[22] James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, et al. Overcoming catastrophic forgetting in neural networks. Proceedings of the national academy of sciences, 114(13):3521–3526, 2017.

[23] Yehuda Koren, Robert Bell, and Chris Volinsky. Matrix factorization techniques for recommender systems. Computer, 42(8):30–37, 2009.

[24] Walid Krichene and Steffen Rendle. On sampled metrics for item recommendation. In Proceedings of the 26th ACM SIGKDD international conference on knowledge discovery & data mining, pages 1748–1757, 2020.

[25] Zeyu Li, Wei Cheng, Yang Chen, Haifeng Chen, and Wei Wang. Interpretable click-through rate prediction through hierarchical attention. In Proceedings of the 13th International Conference on Web Search and Data Mining, pages 313–321, 2020.

[26] Jianxun Lian, Xiaohuan Zhou, Fuzheng Zhang, Zhongxia Chen, Xing Xie, and Guangzhong Sun. xdeepfm: Combining explicit and implicit feature interactions for recommender systems. In Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining, pages 1754–1763, 2018.

[27] Xiangru Lian, Binhang Yuan, Xuefeng Zhu, Yulong Wang, Yongjun He, Honghuan Wu, Lei Sun, Haodong Lyu, Chengjun Liu, Xing Dong, et al. Persia: A hybrid system scaling deep learning based recommenders up to 100 trillion parameters. arXiv preprint arXiv:2111.05897, 2021.

[28] Chong Liu, Xiaoyang Liu, Rongqin Zheng, Lixin Zhang, Xiaobo Liang, Juntao Li, Lijun Wu, Min Zhang, and Leyu Lin. C^2-rec: An effective consistency constraint for sequential recommendation. arXiv preprint arXiv:2112.06668, 2021.

[29] Malte Ludewig, Noemi Mauro, Sara Latifi, and Dietmar Jannach. Empirical analysis of session-based recommendation algorithms. User Modeling and User-Adapted Interaction, 31(1):149–181, 2021.

[30] Jiaqi Ma, Zhe Zhao, Xinyang Yi, Jilin Chen, Lichan Hong, and Ed H Chi. Modeling task relationships in multi-task learning with multi-gate mixture-of-experts. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pages 1930–1940, 2018.

[31] Xiao Ma, Liqin Zhao, Guan Huang, Zhi Wang, Zelin Hu, Xiaoqiang Zhu, and Kun Gai. Entire space multi-task model: An effective approach for estimating post-click conversion rate. In The 41st International ACM SIGIR Conference on Research & Development in Information Retrieval, pages 1137–1140, 2018.

[32] Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg S Corrado, and Jeff Dean. Distributed representations of words and phrases and their compositionality. Advances in neural information processing systems, 26, 2013.

[33] Arvind Narayanan and Vitaly Shmatikov. Robust de-anonymization of large sparse datasets. In 2008 IEEE Symposium on Security and Privacy (sp 2008), pages 111–125. IEEE, 2008.

[34] Steffen Rendle and Christoph Freudenthaler. Improving pairwise learning for item recommendation from implicit feedback. In Proceedings of the 7th ACM international conference on Web search and data mining, pages 273–282, 2014.

[35] Steffen Rendle, Christoph Freudenthaler, Zeno Gantner, and Lars Schmidt-Thieme. Bpr: Bayesian personalized ranking from implicit feedback. arXiv preprint arXiv:1205.2618, 2012.

[36] Steffen Rendle, Walid Krichene, Li Zhang, and John Anderson. Neural collaborative filtering vs. matrix factorization revisited. In Fourteenth ACM conference on recommender systems, pages 240–248, 2020.

[37] Steffen Rendle, Li Zhang, and Yehuda Koren. On the difficulty of evaluating baselines: A study on recommender systems. arXiv preprint arXiv:1905.01395, 2019.

[38] Xiang-Rong Sheng, Liqin Zhao, Guorui Zhou, Xinyao Ding, Binding Dai, Qiang Luo, Siran Yang, Jingshan Lv, Chi Zhang, Hongbo Deng, et al. One model to serve all: Star topology adaptive recommender for multi-domain ctr prediction. In Proceedings of the 30th ACM International Conference on Information & Knowledge Management, pages 4104–4113, 2021.

[39] Kyuyong Shin, Hanock Kwak, Kyung-Min Kim, Minkyu Kim, Young-Jin Park, Jisu Jeong, and Seungjae Jung. One4all user representation for recommender systems in e-commerce. arXiv preprint arXiv:2106.00573, 2021.

[40] Kyuyong Shin, Hanock Kwak, Kyung-Min Kim, Su Young Kim, and Max Nihlen Ramstrom. Scaling law for recommendation models: Towards general-purpose user representations. arXiv preprint arXiv:2111.11294, 2021.

[41] Fei Sun, Jun Liu, Jian Wu, Changhua Pei, Xiao Lin, Wenwu Ou, and Peng Jiang. Bert4rec: Sequential recommendation with bidirectional encoder representations from transformer. In Proceedings of the 28th ACM international conference on information and knowledge management, pages 1441–1450, 2019.

[42] Yang Sun, Junwei Pan, Alex Zhang, and Aaron Flores. Fm2: Field-matrixed factorization machines for recommender systems. In Proceedings of the Web Conference 2021, pages 2828–2837, 2021.

[43] Yang Sun, Fajie Yuan, Ming Yang, Guoao Wei, Zhou Zhao, and Duo Liu. A generic network compression framework for sequential recommender systems. In SIGIR, 2020.

[44] Zhu Sun, Di Yu, Hui Fang, Jie Yang, Xinghua Qu, Jie Zhang, and Cong Geng. Are we evaluating rigorously? Benchmarking recommendation for reproducible evaluation and fair comparison. In Fourteenth ACM conference on recommender systems, pages 23–32, 2020.

[45] Yong Kiam Tan, Xinxing Xu, and Yong Liu. Improved recurrent neural networks for session-based recommendations. In Proceedings of the 1st workshop on deep learning for recommender systems, pages 17–22, 2016.

[46] Maksims Volkovs, Guangwei Yu, and Tomi Poutanen. Dropoutnet: Addressing cold start in recommender systems. Advances in neural information processing systems, 30, 2017.

[47] Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R Bowman. Glue: A multi-task benchmark and analysis platform for natural language understanding. arXiv preprint arXiv:1804.07461, 2018.

[48] Jiachun Wang, Fajie Yuan, Jian Chen, Qingyao Wu, Chengmin Li, Min Yang, Yang Sun, and Guoxiao Zhang. Stackrec: Efficient training of very deep sequential recommender models by iterative stacking. SIGIR, 2021.

[49] Jie Wang, Fajie Yuan, Mingyue Cheng, Joemon M Jose, Chenyun Yu, Beibei Kong, Zhijin Wang, Bo Hu, and Zang Li. Transrec: Learning transferable recommendation from mixture-of-modality feedback. arXiv preprint arXiv:2206.06190, 2022.

[50] Ruoxi Wang, Bin Fu, Gang Fu, and Mingliang Wang. Deep & cross network for ad click predictions. In Proceedings of the ADKDD'17, pages 1–7. 2017.

[51] Ruoxi Wang, Rakesh Shivanna, Derek Cheng, Sagar Jain, Dong Lin, Lichan Hong, and Ed Chi. Dcn v2: Improved deep & cross network and practical lessons for web-scale learning to rank systems. In Proceedings of the Web Conference 2021, pages 1785–1797, 2021.

[52] Xiang Wang, Xiangnan He, Meng Wang, Fuli Feng, and Tat-Seng Chua. Neural graph collaborative filtering. In Proceedings of the 42nd international ACM SIGIR conference on Research and development in Information Retrieval, pages 165–174, 2019.

[53] Fangzhao Wu, Ying Qiao, Jiun-Hung Chen, Chuhan Wu, Tao Qi, Jianxun Lian, Danyang Liu, Xing Xie, Jianfeng Gao, Winnie Wu, et al. Mind: A large-scale dataset for news recommendation. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 3597–3606, 2020.

[54] Jun Xiao, Hao Ye, Xiangnan He, Hanwang Zhang, Fei Wu, and Tat-Seng Chua. Attentional factorization machines: Learning the weight of feature interactions via attention networks. arXiv preprint arXiv:1708.04617, 2017.

[55] Xu Xie, Fei Sun, Zhaoyang Liu, Shiwen Wu, Jinyang Gao, Bolin Ding, and Bin Cui. Contrastive learning for sequential recommendation. arXiv preprint arXiv:2010.14395, 2020.

[56] Canran Xu and Ming Wu. Learning feature interactions with lorentzian factorization machine. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pages 6470–6477, 2020.

[57] Fajie Yuan, Guibing Guo, Joemon M Jose, Long Chen, Haitao Yu, and Weinan Zhang. Lambdafm: learning optimal ranking with factorization machines using lambda surrogates. In Proceedings of the 25th ACM international on conference on information and knowledge management, pages 227–236, 2016.

[58] Fajie Yuan, Xiangnan He, Haochuan Jiang, Guibing Guo, Jian Xiong, Zhezhao Xu, and Yilin Xiong. Future data helps training: Modeling future contexts for session-based recommendation. In WWW, pages 303–313, 2020.

[59] Fajie Yuan, Xiangnan He, Alexandros Karatzoglou, and Liguang Zhang. Parameter-efficient transfer from sequential behaviors for user modeling and recommendation. In SIGIR, pages 1469–1478, 2020.

[60] Fajie Yuan, Alexandros Karatzoglou, Ioannis Arapakis, Joemon M Jose, and Xiangnan He. A simple convolutional generative network for next item recommendation. In WSDM, pages 582–590, 2019.

[61] Fajie Yuan, Guoxiao Zhang, Alexandros Karatzoglou, Joemon Jose, Beibei Kong, and Yudong Li. One person, one model, one world: Learning continual user representation without forgetting. SIGIR, 2021.

[62] Zheng Yuan, Fajie Yuan, Yu Song, Youhua Li, Fei Yang, and Yunzhu Pan. Where to go next for recommender systems? ID- vs. modality-based recommender models revisited. https://openreview.net/pdf?id=bz3MAU-RhnW, 2022.

[63] Tong Zhao, Julian McAuley, and Irwin King. Leveraging social connections to improve personalized ranking for collaborative filtering. In Proceedings of the 23rd ACM international conference on conference on information and knowledge management, pages 261–270, 2014.

[64] Guorui Zhou, Na Mou, Ying Fan, Qi Pi, Weijie Bian, Chang Zhou, Xiaoqiang Zhu, and Kun Gai. Deep interest evolution network for click-through rate prediction. In Proceedings of the AAAI conference on artificial intelligence, volume 33, pages 5941–5948, 2019.

[65] Guorui Zhou, Xiaoqiang Zhu, Chenru Song, Ying Fan, Han Zhu, Xiao Ma, Yanghui Yan, Junqi Jin, Han Li, and Kun Gai. Deep interest network for click-through rate prediction. In Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining, pages 1059–1068, 2018.

[66] Kun Zhou, Hui Yu, Wayne Xin Zhao, and Ji-Rong Wen. Filter-enhanced mlp is all you need for sequential recommendation. In Proceedings of the ACM Web Conference 2022, pages 2388–2399, 2022.

[67] Feng Zhu, Yan Wang, Chaochao Chen, Jun Zhou, Longfei Li, and Guanfeng Liu. Cross-domain recommendation: challenges, progress, and prospects. arXiv preprint arXiv:2103.01696, 2021.

[68] Jieming Zhu, Jinyang Liu, Shuai Yang, Qi Zhang, and Xiuqiang He. Open benchmarking for click-through rate prediction. In Proceedings of the 30th ACM International Conference on Information & Knowledge Management, pages 2759–2769, 2021.

[69] Jieming Zhu, Kelong Mao, Quanyu Dai, Liangcai Su, Rong Ma, Jinyang Liu, Guohao Cai, Zhicheng Dou, Xi Xiao, and Rui Zhang. Bars: Towards open benchmarking for recommender systems. arXiv preprint arXiv:2205.09626, 2022.

## 附录

### A 数据集比较

我们在表12中展示了Tenrec与其他流行推荐数据集之间的差异。首先，大多数数据集只包含单一场景。没有重叠的用户和item，很难开发和评估迁移学习推荐方法。此外，Tenrec包含非常丰富的正向用户反馈，可用于评估多任务学习和偏好级迁移学习任务。第三，与大多数推荐数据集相比，Tenrec具有真正的负样本，可用于评估更真实的CTR预测任务。

值得一提的是，Amazon中的多个域与Tenrec的定义不同。在我们的Tenrec中，不同域的item要么来自不同的推荐系统，要么由完全不同的算法推荐。然而，Amazon中的域仅基于其item类别进行划分。不同类别的item是否由相同或不同算法推荐是未知的。如果item由相同的模型推荐且来自同一平台，则不适合用于跨域推荐任务。事实上，我们的Tenrec-QKA也包括许多不同的文章类别。

### B 补充实验

在正文中，我们仅报告了随机采样100万用户的结果，这里我们展示了在完整QK-video数据集上使用500万用户进行CTR预测（表13）和SBR（表14）任务的结果，遵循相同的实验设置。对于每个任务，我们报告了正文中排名靠前的几个基线。

除上述实验外，我们还补充了CTR预测任务中共享历史嵌入（即所有交互item共享相同嵌入）的实验。如表15所示，我们可以做出两个观察：（1）具有共享历史嵌入的CTR模型通常略逊于具有单独历史嵌入（SHE）的模型；（2）具有共享历史嵌入的CTR模型显示出与之前表2中SHE报告相似的准确率排名。

我们还添加了更多具有不同冷启动设置的实验。具体来说，我们注意到在一些实际推荐场景中，冷用户和热用户并存。为了创建这样的场景，我们首先提取QK-video和QK-article之间的重叠用户。然后，我们随机采样n%的用户（例如n=30, 70, 100），然后选择他们最近的k个交互item，其中k是1到5之间的随机整数，确保这些用户是冷用户。其余热用户的行为保持不变。对于训练，我们使用所有这些热用户的行为和冷用户的50%行为。对于评估，我们仅评估这些冷用户的预测准确率，使用25%的交互进行验证，25%进行测试。结果在表18中报告。

这里，我们报告QB-video上标准Top-Nitem推荐任务的基线结果。我们过滤掉会话长度小于10的用户。然后，我们将每个用户的交互按8:1:1分割为训练集、验证集和测试集。我们评估四个流行的基线：MF [23]、NCF [17]、NGCF [52]、LightGCN [16]来验证Tenrec。超参数与之前类似的方式搜索。NGCF的学习率设为5e−4，NCF设为1e−6，MF和LightGCN设为5e−3。所有模型的批大小设为4096。所有模型的嵌入大小设为128。然后NCF、NGCF和LightGCN的层数设为2。我们使用两种类型的负采样器展示结果：随机采样器和word2vec [32]中使用的流行度采样器，幂参数设为0.75。每个用户的负样本数设为4。所有结果报告在表16和表17中。值得一提的是，比随机和流行度采样器更强大的负采样器可以轻松带来更好的推荐准确率，例如LambdaFM [57]中使用的两种动态采样器。换句话说，如果你想比较网络架构，应确保所有其他设置（损失函数、采样比例和分布）在比较时保持一致。

对于其他任务，我们将为完整数据集版本和100万用户版本创建每个任务的排行榜。

### C 数据集通用数据表

#### C.1 动机

**创建该数据集的目的是什么？是否有特定的任务？是否有需要填补的特定空白？请提供描述。**

为了促进多样化的推荐研究，我们提出了Tenrec，一个大规模、多用途的真实世界数据集。与现有的公开数据集相比，Tenrec具有几个优点：（1）它包含来自四个不同真实世界推荐场景的重叠用户/item，可用于研究跨域推荐（CDR）和迁移学习（TF）方法；（2）它包含多种类型的正向用户反馈（例如点击、点赞、分享、关注、阅读和收藏），可用于研究多任务学习（MTL）问题；（3）它同时具有正向用户反馈和真正的负向反馈，可用于研究更实际的点击率（CTR）预测场景；（4）它包含身份信息（即用户ID和itemID）之外的额外用户和item特征，可用于上下文/基于内容的推荐。

**谁创建了数据集（例如哪个团队、研究组）以及代表哪个实体（例如公司、机构、组织）？**

该数据集由袁光虎和孔蓓蓓创建，他们分别是腾讯的实习生和员工。

**谁资助了数据集的创建？如果有相关的资助，请提供资助方名称和资助名称及编号。**

无。

#### C.2 组成

**组成数据集的实例代表什么（例如文档、照片、人员、国家）？是否有多种类型的实例（例如电影、用户和评分；人与人的交互；节点和边）？请提供描述。**

实例是收集自腾讯两个不同信息流推荐平台的用户反馈，包括正向反馈（即视频点击、分享、点赞和关注）和负向反馈（有曝光但无用户操作）。

**总共有多少个实例（如果适用，按类型分）？**

QK（QQ看点）视频数据集中有493,458,970个实例，QB（QQ浏览器）视频数据集中有11,722,249个实例，QK（QQ看点）文章数据集中有46,111,728个实例，QB（QQ浏览器）文章数据集中有348,736个实例样本，每个样本都是用户-item交互。

**数据集包含所有可能的实例还是来自更大集合的样本（不一定是随机的）？如果数据集是样本，那么更大的集合是什么？样本是否代表更大的集合（例如地理覆盖范围）？如果是，请描述如何验证/确认这种代表性。如果样本不代表更大的集合，请说明原因（例如为了涵盖更多样化的实例范围，因为实例被扣留或不可用）。**

该数据集是一个实例样本。我们从腾讯两个不同的信息流推荐平台随机抽取实例，要求每个用户至少有5次视频点击行为。没有进行代表性测试。

**每个实例包含哪些数据？"原始"数据（例如未处理的文本或图像）还是特征？无论哪种情况，请提供描述。**

QK/QB-video中每个实例的格式为{用户ID, itemID, 点击, 点赞, 分享, 关注, 视频类别, 观看次数, 用户性别, 用户年龄, 时间戳}。点击、点赞、分享、关注是二进制值，表示用户是否有此类操作。观看次数是用户观看视频的次数。用户ID、itemID、用户性别、用户年龄和时间戳已因隐私问题而被脱敏。用户年龄已被分箱，每个箱代表一个十年区间。

QK/QB-article中每个实例的格式为{用户ID, itemID, 点击, 点赞, 分享, 关注, 阅读, 收藏, 点击次数, 点赞次数, 评论次数, 曝光次数, 阅读百分比, 二级类别, 一级类别, item评分1, item评分2, item评分3, 阅读时长, 时间戳}。后缀"*_count"表示每篇文章*操作的总数。read_percentage表示用户阅读文章的百分比，值范围为0到100。category_first和category_second是文章的类别，其中"_first"是粗粒度类别（例如体育、娱乐、军事等），"_second"是细粒度类别（例如NBA、世界杯、科比等）。item_score1、item_score2、item_score3表示通过不同评分系统评估的item质量。read_time是阅读的时长。

**每个实例是否有相关的标签或目标？如果是，请提供描述。**

标签是表示用户是否有此类操作或用户画像的二进制值。

**单个实例中是否有缺失的信息？如果是，请提供描述，解释为什么缺少这些信息（例如因为不可用）。这不包括故意移除的信息，但可能包括例如被编辑的文本。**

一小部分实例缺乏视频类别、用户年龄和用户性别。相应信息在真实系统中缺失。

**是否有推荐的数据分割（例如训练集、开发/验证集、测试集）？如果是，请提供这些分割的描述，解释其原理。**

我们遵循一些常见实践将数据按8:1:1分割为训练集、验证集和测试集。

**数据集中是否有任何错误、噪声源或冗余？如果是，请提供描述。**

否。

**数据集是自包含的，还是链接到或以其他方式依赖外部资源（例如网站、推文、其他数据集）？如果链接到或依赖外部资源，a) 是否有保证它们随时间存在且保持不变的保证；b) 是否有完整数据集的官方存档版本（包括创建数据集时存在的外部资源）；c) 与任何外部资源相关的限制（例如许可证、费用）是否适用于数据集消费者？请描述所有外部资源及其相关限制，以及链接或其他访问方式（如适用）。**

该数据集完全自包含。

**数据集中是否包含可能被视为机密的数据（例如受法律特权或医患保密协议保护的数据，包含个人非公开通信内容的数据）？如果是，请提供描述。**

否。

**数据集中是否包含如果直接查看可能具有攻击性、侮辱性、威胁性或可能引起焦虑的数据？如果是，请说明原因。**

否。

**数据集是否识别任何子群体（例如按年龄、性别）？如果是，请描述如何识别这些子群体，并提供其在数据集中的各自分布描述。**

否。

**是否可以从数据集中直接或间接（即结合其他数据）识别个人？如果是，请描述如何识别。**

从数据集信息中不可能识别个人。

**数据集是否包含任何可能被视为敏感的数据（例如揭示种族或民族起源、性取向、宗教信仰、政治观点或工会成员身份或位置的数据；财务或健康数据；生物识别或遗传数据；政府身份证明形式，如社会安全号码；犯罪历史）？如果是，请提供描述。**

否。

#### C.3 收集过程

**每个实例关联的数据是如何获取的？数据是直接可观察的（例如原始文本、电影评分）、由主体报告的（例如调查回复）还是从其他数据中间接推断/衍生的（例如词性标签、基于模型的年龄或语言猜测）？如果数据是由主体报告或从其他数据间接推断/衍生的，数据是否经过验证/确认？如果是，请描述如何验证。**

数据主要来自腾讯信息流推荐平台上的用户反馈，是可观察的。

**使用了哪些机制或程序来收集数据（例如硬件设备或传感器、人工手动整理、软件程序、软件API）？这些机制或程序是如何验证的？**

数据表的作者未知。

**如果数据集是更大集合的样本，采样策略是什么（例如确定性、具有特定采样概率的概率性）？**

我们从数据库中随机抽取用户，要求每个用户至少有5次视频点击行为。

**谁参与了数据收集过程（例如学生、众包工作者、承包商）以及他们是如何获得报酬的（例如众包工作者获得了多少报酬）？**

数据表的作者未知。

**数据是在什么时间范围内收集的？这个时间范围是否与实例关联的数据的创建时间范围匹配（例如最近抓取的旧新闻文章）？如果不匹配，请描述实例关联的数据创建的时间范围。**

我们收集了2021年9月17日至12月7日期间QK/QB的用户行为日志。

**你们是直接从相关个人收集数据，还是通过第三方或其他来源（例如网站）获取？**

数据是从腾讯信息流推荐平台收集的。

#### C.4 预处理/清洗/标注

**是否对数据进行了任何预处理/清洗/标注（例如离散化或分桶、分词、词性标注、SIFT特征提取、移除实例、处理缺失值）？如果是，请提供描述。如果没有，您可以跳过本节剩余问题。**

我们对用户ID和itemID进行匿名化以保护用户隐私。用户画像也被处理成离散或二进制值。

**除了预处理/清洗/标注后的数据，"原始"数据是否也被保存了（例如以支持未预见的未来用途）？如果是，请提供"原始"数据的链接或其他访问方式。**

否。

#### C.5 用途

**数据集是否已经用于任何任务？如果是，请提供描述。**

否。

**是否有链接到使用该数据集的所有或任何论文或系统的仓库？如果是，请提供链接或其他访问方式。**

是。

**数据集还能用于哪些（其他）任务？**

该数据集可用于CTR预测、基于会话的推荐、多任务学习推荐、迁移学习推荐、用户画像预测、终身用户表示学习、冷启动推荐、模型压缩、模型训练加速和模型推理加速。详情见我们的论文。

**数据集的组成或其收集和预处理/清洗/标注方式是否有任何可能影响未来用途的问题？例如，是否有任何数据集消费者需要了解的信息，以避免可能导致不公平对待个人或群体（例如刻板印象、服务质量问题）或其他风险或危害（例如法律风险、财务危害）的用途？如果是，请提供描述。数据集消费者可以做些什么来减轻这些风险或危害？**

在我们将数据集匿名化后，这里几乎没有风险。

#### C.6 分发

**数据集是否会分发给代表其创建实体的第三方（例如公司、机构、组织）以外的方？如果是，请提供描述。**

该数据集在互联网上公开可用。

**数据集将如何分发（例如网站上的压缩包、API、GitHub）？数据集是否有数字对象标识符（DOI）？**

数据集的分发详情见我们的论文。

**数据集何时分发？**

数据集将于2022年6月分发。

**数据集是否将在版权或其他知识产权（IP）许可和/或适用的使用条款下分发？如果是，请描述此许可和/或使用条款，并提供链接或其他访问方式，或以其他方式复制任何相关的许可条款或使用条款，以及这些限制相关的任何费用。**

该数据集根据CC BY-NC 4.0国际许可（https://creativecommons.org/licenses/by-nc/4.0/）许可。如果使用该数据集，要求引用相应的论文。

**是否有任何第三方对实例关联的数据施加了基于IP或其他限制？如果是，请描述这些限制，并提供链接或其他访问方式，或以其他方式复制任何相关的许可条款，以及这些限制相关的任何费用。**

否。

**是否有任何出口管制或其他监管限制适用于数据集或单个实例？如果是，请描述这些限制，并提供链接或其他访问方式，或以其他方式复制任何支持文档。**

数据集的作者未知。

#### C.7 维护

**谁将支持/托管/维护数据集？**

袁光虎和袁法杰正在支持/维护该数据集。

**如何联系数据集的所有者/管理者/维护者（例如电子邮件地址）？**

袁光虎和袁法杰可以分别通过gh.yuan0@gmail.com和yuanfajie@westlake.edu.cn联系。

**是否有勘误表？如果是，请提供链接或其他访问方式。**

尚未发现。

**数据集是否会更新（例如更正标注错误、添加新实例、删除实例）？如果是，请描述更新频率、更新人员以及如何向数据集消费者传达更新（例如邮件列表、GitHub）。**

这将在数据集网页上发布。

**旧版本的数据集是否会继续得到支持/托管/维护？如果是，请描述方式。如果不是，请描述如何向数据集消费者传达其废弃状态。**

我们不维护旧版本的数据集，如果我们更新数据集的版本，我们将在相关GitHub上发布数据集更新的具体细节。

**如果其他人想要扩展/增强/基于此数据集构建/为此数据集做贡献，是否有机制支持他们这样做？如果是，请提供描述。这些贡献是否会经过验证/确认？如果是，请描述方式。如果不是，为什么？是否有向数据集消费者传达/分发这些贡献的流程？如果是，请提供描述。**

如果其他人想要扩展/增强/基于此数据集构建/为此数据集做贡献，请联系原始作者关于整合修复/扩展的事宜。
