# YouTube-8M: A Large-Scale Video Classification Benchmark（中文翻译）

> Google Research | 2016

本文分享了YouTube-8M——一个大规模多标签视频分类基准数据集，包含约800万个YouTube视频、4800个视觉实体类别。核心内容：
- 构建了基于知识图谱实体的视觉标注词汇表，通过人工评估确保实体可视觉识别
- 以每秒1帧解码视频，用预训练Inception网络提取帧级深度特征，经PCA+量化压缩后开放下载
- 提供了帧级模型（Logistic回归+平均池化、Deep Bag of Frames、LSTM）和视频级模型（Logistic回归、Hinge Loss、混合专家MoE）等多种基线方法
- 在Sports-1M和ActivityNet上验证了迁移学习能力，在ActivityNet上将mAP从53.8%提升至77.6%，达到当时最优

关键发现：
- 视频级特征的简单聚合（均值、标准差、顺序统计量）搭配MoE分类器优于复杂帧级模型
- 当底层帧级特征足够强时，复杂视频建模的必要性降低
- 大规模、多样化的视频预训练特征可很好地泛化到其他数据集/任务
- 数据集标注精度78.8%、召回率14.5%，存在标注缺失但相对排序稳定

---

## 摘要

计算机视觉的许多最新进展都归功于大规模数据集。机器学习开源软件包和廉价的通用硬件降低了大规模探索新方法的门槛。在几天内对数百万样本进行模型训练已成为可能。尽管图像理解领域存在大规模数据集（如ImageNet），但尚无同等规模的视频分类数据集。

在本文中，我们介绍了YouTube-8M，这是目前最大的多标签视频分类数据集，包含约800万个视频（50万小时视频），标注了4800个视觉实体。为了获取视频及其标签，我们使用了YouTube视频标注系统，该系统为视频标注其主要主题。虽然标签是机器生成的，但其精度很高，且来源于多种基于人工的信号，包括元数据和查询点击信号。我们使用自动化和人工筛选策略对视频标签（知识图谱实体）进行了过滤，包括请人工评估员判断标签是否可视觉识别。然后，我们以每秒1帧的速度解码每个视频，并使用在ImageNet上预训练的深度CNN提取紧邻分类层之前的隐藏表示。最后，我们对帧特征进行了压缩，并将特征和视频级标签开放下载。

我们在该数据集上训练了各种（适度的）分类模型，使用流行的评估指标进行评估，并将其作为基线报告。尽管数据集规模庞大，但我们的部分模型使用TensorFlow在单台机器上不到一天即可收敛到最优。我们计划发布用于训练TensorFlow模型和计算指标的代码。

## 1. 引言

ImageNet [6] 等大规模数据集是图像理解领域近期进展的关键推动因素 [20, 14, 11]。通过支持具有数百万参数的深度网络的学习过程，这些数据集在图像理解快速进步至接近人类水平精度方面发挥了至关重要的作用 [30]。此外，这些网络的中间层激活已被证明对各种分类之外的任务 [41, 9, 31] 具有强大且可解释的能力。类似地，视频基准的数量和规模也在增长，如面向体育视频的Sports-1M [19] 和面向人类活动的ActivityNet [12]。然而，与包含多样且通用物体/实体集合的ImageNet不同，现有的视频基准仅限于动作和体育类别。

在本文中，我们介绍了YouTube-8M[^1]，一个面向通用多标签视频分类的大规模基准数据集。我们将视频分类的任务定义为：给定视频的帧，生成与该视频相关的标签。因此，与Sports-1M和ActivityNet不同，YouTube-8M不仅限于动作类别。例如，图1展示了"吉他"实体的随机视频示例。

[^1]: http://research.google.com/youtube8m

![图1：YouTube-8M是一个面向通用多标签视频分类的大规模基准。该数据集浏览器的截图展示了数据集中标注了"吉他"实体的一部分视频。数据集浏览器允许浏览和搜索完整的知识图谱实体词汇表（分为24个顶级垂直领域），并查看对应的视频。]

我们首先从知识图谱实体构建视觉标注词汇表，这些实体基于YouTube视频标注系统 [2] 作为YouTube视频的主题标注出现。为确保词汇表由可视觉识别的实体组成，我们使用了包括人工评估在内的多种过滤标准。这些实体涵盖活动（体育、游戏、爱好）、物体（汽车、食物、产品）、场景（旅行）和事件。实体是根据其在YouTube上的流行度和人工评估员对其视觉可识别性的评分来选择的。它们试图用几个简洁的标签来描述视频的核心主题。

然后，我们为每个实体收集一组样本视频，并使用公开可用的最先进的Inception网络 [4] 从中提取特征。具体来说，我们以每秒1帧的速度解码视频，并为每帧提取分类层之前的最后一个隐藏表示。我们压缩了帧级特征，并在我们的网站上开放下载。我们希望预计算的特征能够消除计算障碍，创造公平的竞争环境，使研究人员能够以前所未有的规模探索视频领域的新技术。

为此，我们提供了广泛的实验，比较了多种先进的视频表示学习技术，包括深度网络 [26] 和LSTM（长短期记忆网络）[13] 在该数据集上的表现。此外，我们还展示了在该数据集上学习的视频特征表示迁移到其他基准（如Sports-1M和ActivityNet）上能带来显著的性能提升。

在本文的其余部分，我们首先在第2节回顾现有的图像和视频分类基准。第3节介绍数据集的细节，包括收集过程和对类别和视频的简要分析。第4节回顾了在给定固定帧级特征的情况下进行多标签视频分类的几种方法，并在数据集上对这些方法进行了评估。第5节展示了在我们的大规模数据集上学习的特征和模型在其他基准上泛化得很好。第6节给出总结性评述。

## 2. 相关工作

图像基准在推动计算机视觉算法在图像理解方面的发展中发挥了重要作用。从一批标注良好的小规模数据集如Caltech101/256 [8, 10]、MSRC [32]、PASCAL [7] 开始，图像理解研究已迅速发展为利用更大规模的数据集如ImageNet [6] 和SUN [38] 来推动下一代视觉算法的发展。ImageNet尤其因其类别数（21841）、类别多样性（27个顶级类别）以及数百万标注图像，使得具有数百万参数的深度特征学习技术（如AlexNet [20] 和Inception [14] 架构）得以发展。

视频理解领域也在进行类似的努力，社区已从KTH [22]、Hollywood2 [23]、Weizmann [5] 等仅有几千个视频片段的小型、标注良好的数据集，快速推进到UCF101 [33]、Thumos'14 [16] 和HMDB51 [21] 等中等规模的数据集，包含超过50个动作类别。目前，最大的可用视频基准是Sports-1M [19]（包含487个体育相关活动和100万个视频）、YFCC-100M [34]（包含80万个视频及部分原始元数据：标题、描述、标签）、FCVID [17] 数据集（91223个视频，人工标注了239个类别）以及ActivityNet [12]（约200个人类活动类别和几千个视频）。然而，几乎所有现有的视频基准都局限于识别动作和活动类别，且类别数均小于500。

YouTube-8M在视频基准方面填补了以下空白：

* 一个大规模的视频标注和表示学习基准，反映视频的主要主题。
* 标注类别的数量和多样性显著提升——4800个知识图谱实体，而其他所有数据集均少于500个类别。
* 标注视频数量大幅增加——超过800万个视频，超过50万小时的视频。
* 提供19亿个视频帧的预计算最先进特征。

总的来说，YouTube-8M包含超过800万个视频——超过50万小时的视频——来自4800个类别。图2展示了YouTube-8M与现有图像和视频数据集相比的规模。我们希望该数据集前所未有的规模和多样性能够为开发先进的视频理解和表示学习技术提供有用的资源。

![图2：图像和视频理解任务的数据集发展历程。大规模数据集在两个领域的进步中都发挥了关键作用。]

## 3. YouTube-8M 数据集

YouTube-8M是一个用于视频理解的基准数据集，其主要任务是确定视频的关键主题。我们从YouTube视频入手，因为它们是多种类别知识的一个良好（尽管有噪声）来源，包括各种体育、活动、动物、食物、产品、旅游景点、游戏等等。我们使用YouTube视频标注系统 [2] 来获取视频的主题标注，并根据给定主题检索视频。标注以知识图谱实体 [3]（以前称为Freebase主题 [1]）的形式提供。它们基于视频的元数据、上下文和内容信号 [2] 与每个视频关联。

我们使用知识图谱实体来简洁地描述视频的主要主题。例如，一个在土路和悬崖上骑行的视频，其核心主题是"山地自行车"，而不是"泥土"、"道路"、"人物"、"天空"等。因此，数据集的目标不仅是理解视频每一帧中存在什么，还要识别出最能描述视频内容的少数关键主题。注意，这与典型的事件或场景识别任务不同，在那些任务中每个item属于单一事件或场景 [38, 28]。这也不同于大多数物体识别任务，后者目标是标注图像中所有可见的内容——这会在每个视频上产生数千个标签，却无法回答视频真正是关于什么的。该基准的目标是理解视频中的内容并将其总结为几个关键主题。在以下子节中，我们将描述词汇表和视频选择方案，然后简要总结数据集统计信息。

### 3.1 词汇表构建

在设计数据集词汇表时，我们遵循两个主要原则：1）数据集中的每个标签应仅凭视觉信息即可区分；2）每个标签应有足够数量的视频用于训练模型和在测试集上计算可靠的指标。对于前者，我们结合了人工策划的主题和人工评分来将词汇表筛选为视觉集合。对于后者，我们只考虑数据集中至少有200个视频的实体。

知识图谱包含数百万个主题。每个主题有一个或多个类型，经过高精度策划。例如，有一个详尽的动物列表（类型为animal）和一个详尽的食物列表（类型为food）。为了构建初始词汇表，我们手动选择了一份包含25个我们认为具有视觉性的实体类型的白名单（例如sport、tourist_attraction、inventions），同时加入了一份我们认为非视觉性的类型的黑名单（例如music_artists、music_compositions、album、software）。然后，我们获取了所有至少有一个白名单类型且没有黑名单类型的实体，得到了约5万个实体的初始词汇表。

随后，我们使用人工评估员手动将这个集合修剪为更小的、高置信度被认为具有视觉性且无需非常深入的专业知识即可识别的实体集。评估员获得了说明和示例。每个实体由3名评估员评分，评分取平均。图4a展示了主要的评分问题。该过程最终得到了约1万个被认为视觉上可识别且不过于细粒度（即非领域专家在学习一些示例后也能识别）的实体。这些实体被进一步修剪：我们只保留了拥有超过200个热门视频的实体，如下一节所述。数据集中最终的实体集在描述主题的特定性方面相当平衡，涵盖粗粒度和细粒度实体，如图4b所示。

![图4：评估员评估每个实体在特定性和视觉可识别性方面的指南，采用离散评分（1到5），其中1表示最具视觉性且易于被普通人识别。每个实体由3名评估员评分。我们只保留平均分不超过2.5的实体，并按特定性将其分类为粗粒度、中粒度和细粒度实体，使用等大小的评分区间。]

### 3.2 收集视频

在确定初始目标词汇表后，我们按照以下步骤获取视频：

* 收集了对应于这1万个视觉实体的所有视频，且每个视频至少有1000次观看，使用了YouTube视频标注系统 [2]。我们排除了太短（<120秒）或太长（>500秒）的视频。
* 从中随机采样了1000万个视频。
* 使用YouTube视频标注系统获取这1000万个采样视频的所有实体。至此标注完成。
* 过滤掉少于200个视频的实体，以及没有剩余实体的视频。这使得我们的数据规模缩减至8,264,650个视频。
* 将视频按70%:20%:10%的比例划分为训练集、验证集和测试集。我们发布所有划分的特征，但仅发布训练集和验证集的标签。

### 3.3 特征

原始视频数据集的大小为数百TB，涵盖超过50万小时的视频。这对大多数研究团队来说是不切实际的（使用实时视频处理引擎，处理全部数据需要超过50年）。因此，我们对视频进行预处理，并使用最先进的深度模型——在ImageNet [14] 上训练的公开可用的Inception网络 [4]——提取帧级特征。具体来说，我们以每秒1帧的速度解码每个视频（最多前360秒，即6分钟），将解码后的帧输入Inception网络，并获取分类层之前最后一个隐藏层的ReLU激活（层名pool_3/_reshape）。每秒钟视频的特征向量为2048维。虽然这去除了视频中的运动信息，但最近的研究表明，随着视频数据规模和多样性的增加，运动特征的收益递减 [26, 35]。静态帧级特征提供了极好的基线，而构建紧凑高效的运动特征超出了本文的范围。不过，我们希望未来能用音频和运动特征扩展该数据集。出于存储和计算原因，我们将每个视频的处理限制在前360秒。作为比较，UCF-101中视频的平均长度为10-15秒，Sports-1M为336秒，而本数据集的平均长度为230秒。

之后，我们应用PCA（+白化）将特征维度降至1024，然后进行量化（每系数1字节）。这两种压缩技术将数据大小减少了8倍。用于PCA的均值向量和协方差矩阵是在训练集划分的所有帧上计算的。我们使用最优计算的（非均匀）量化分箱边界将每个32位浮点数量化为256个不同的值（8位）。我们确认了这种大小缩减不会显著影响评估指标。事实上，在全尺寸数据（比我们发布的大8倍）上训练所有基线，所有评估指标的提升不到1%。

需要注意的是，虽然该数据集附带了标准的帧级特征，但它在固定帧级特征之上研究视频表示学习方法方面留下了很大的空间（我们探索的方法见第4节）。

### 3.4 数据集统计信息

YouTube-8M数据集包含4800个类别，总共8,264,650个视频。一个视频可能被标注多个类别，每个视频的平均类别数为1.8。表2显示了我们发布特征的视频数量在三个数据集划分中的分布。

| 数据集 | 训练集 | 验证集 | 测试集 | 总计 |
|---------|--------|--------|--------|------|
| YouTube-8M | 5,786,881 | 1,652,167 | 825,602 | 8,264,650 |

**表2：数据集划分大小。**

我们仅处理了每个视频的前六分钟，每秒1帧。数据集中视频的平均长度为229.6秒，整个数据集约有19亿帧（及相应的特征）。

我们将4800个实体分组为24个顶级类别以衡量统计信息并展示多样性。虽然我们在训练中不使用这些类别，但为了完整性，我们发布了实体到类别的映射。表1展示了每个类别中的顶部实体。需要注意的是，虽然某些类别本身可能看起来不具视觉性，但其中的大多数实体是视觉性的。例如，工作与教育包括大学、教室、讲座等，法律与政府包括警察、应急车辆、军事相关实体，这些都有很好的视觉呈现。

| 顶级类别 | 第1实体 | 第2实体 | 第3实体 | 第4实体 | 第5实体 | 第6实体 | 第7实体 |
|-----------|---------|---------|---------|---------|---------|---------|---------|
| 艺术与娱乐 | 音乐会 | 动画 | 音乐视频 | 舞蹈 | 吉他 | 唱片骑师 | 预告片 |
| 汽车与车辆 | 车辆 | 汽车 | 摩托车 | 自行车 | 飞机 | 卡车 | 船 |
| 美容与健身 | 时尚 | 头发 | 化妆品 | 重量训练 | 发型 | 指甲 | 睫毛膏 |
| 书籍与文学 | 书 | 哈利·波特 | 圣经 | 写作 | 杂志 | 爱丽丝 | 电子书 |
| 商业与工业 | 火车 | 模型飞机 | 鱼 | 水 | 拖拉机牵引 | 广告 | 着陆 |
| 计算机与电子 | 个人电脑 | 视频游戏机 | iPhone | PlayStation3 | 平板电脑 | Xbox360 | Microsoft Windows |
| 金融 | 钱 | 银行 | 外汇 | 欧元 | 美元 | 信用卡 | 现金 |
| 食品与饮料 | 食物 | 烹饪 | 食谱 | 蛋糕 | 巧克力 | 鸡蛋 | 吃 |
| 游戏 | 视频游戏 | 我的世界 | 动作冒险游戏 | 策略视频游戏 | 体育游戏 | 使命召唤 | 侠盗猎车手V |
| 健康 | 医药 | 生食 | 耳朵 | 眼镜 | 伤 | 膳食补充剂 | 牙套 |
| 爱好与休闲 | 钓鱼 | 户外娱乐 | 无线电控制模型 | 婚礼 | 圣诞节 | 狩猎 | 潜水 |
| 家居与园艺 | 园艺 | 家居装修 | 房子 | 厨房 | 花园 | 门 | 游泳池 |
| 互联网与电信 | 手机 | 智能手机 | 电话 | 网站 | Sony Xperia | Google Nexus | 万维网 |
| 工作与教育 | 学校 | 大学 | 高中 | 教师 | 幼儿园 | 校园 | 教室 |
| 法律与政府 | 坦克 | 消防员 | 美国总统 | 士兵 | 总统 | 警察 | 战斗机 |
| 新闻 | 天气 | 雪 | 雨 | 新闻广播 | 报纸 | 美泰 | 冰雹 |
| 人与社会 | 祈祷 | 家庭 | 培乐多 | 人类 | 龙 | 天使 | 塔罗牌 |
| 宠物与动物 | 动物 | 狗 | 马 | 猫 | 鸟 | 水族馆 | 小狗 |
| 房地产 | 房子 | 公寓 | 公寓 | 宿舍 | 大厦 | 摩天大楼 | 阁楼 |
| 参考 | 吸血鬼 | 公交车 | 河流 | 城市 | 美人鱼 | 村庄 | 武士 |
| 科学 | 自然 | 机器人 | 眼睛 | 冰 | 生物学 | 皮肤 | 光 |
| 购物 | 玩具 | 乐高 | 雪橇 | 娃娃 | 鞋 | 小马宝莉 | Nike公司 |
| 体育 | 赛车运动 | 足球 | 冬季运动 | 骑自行车 | 篮球 | 体操 | 摔跤 |
| 旅行 | 游乐园 | 酒店 | 机场 | 海滩 | 过山车 | 湖 | 度假村 |
| 完整词汇表 | 车辆 | 音乐会 | 动画 | 音乐视频 | 视频游戏 | 赛车运动 | 足球 |

**表1：每个顶级类别中最频繁的实体。**

图5展示了实体和视频的对数-对数规模分布。图6a和6b分别展示了按实体数量和视频数量衡量的类别大小。

![图5：视频数量（对数刻度）与实体排名（对数刻度）的关系。实体按视频数量排序。我们注意到这大致遵循自然的Zipf分布。]

![图6：YouTube-8M数据集的顶级类别统计信息。]

### 3.5 人工评估测试集

YouTube视频标注系统的标注可能有噪声且不完整，因为它们是自动从元数据、锚文本、评论和用户参与信号 [2] 生成的。为了量化噪声，我们从测试集划分中均匀采样了8000多个视频，每个视频由3名人工评估员详尽地评估其标签。我们以人工评估员为标准，测量了真实标签的精度和召回率，分别为78.8%和14.5%。需要注意的是，在类似标注任务中，人工评估员之间的一致性通常也在80%左右，因此这些真实标签的精度可能与（非专家）人工提供的标签相当。然而，召回率较低，这使得该数据集成为处理缺失数据的方法的绝佳测试平台。我们主要在（有噪声的）验证集上报告模型的精度，但也在小得多的人工评估集上展示了一些结果，显示某些指标在两个数据集上惊人地相似。

虽然第4节中的基线结果非常令人鼓舞，但我们相信，如果在建模时显式处理不正确 [29]（78.8%精度）或缺失 [40, 25]（14.5%召回率）的训练标签，这些结果可以得到显著改善。我们认为这是一个令人兴奋的研究方向，该数据集将能够在规模上推动这一领域的发展。

## 4. 基线方法

### 4.1 基于帧特征的模型

该数据集的挑战之一是我们只有视频级的真实标签。我们没有关于标签在视频内的定位信息，也没有它们在视频中相对重要性的信息，然而我们需要推断它们对整个视频的重要性。在本节中，我们考虑使用输入帧级特征来训练预测视频主要主题的模型。帧级模型在之前的工作中已经显示出在视频级任务上的竞争性能 [19, 26]。一个视频 `v` 由一序列帧级特征 `x^$v_{1:F_v}$` 给出，其中 `x^v_j` 是视频 `v` 中第 `j` 帧的特征。

#### 4.1.1 帧级模型与平均池化

由于我们没有帧级的真实标签，我们将视频级的真实标签分配给该视频中的每一帧。更复杂的基于多实例学习的公式留给未来工作。我们从每个视频中采样20个随机帧，并将所有帧与视频级真实标签关联。这产生了约1.2亿帧。对于每个实体 `e`，我们得到1.2亿个 `(x_i, y^e_i)` 对的实例，其中 `x_i \in R^1024` 是Inception特征，`y^e_i \in {0,1}` 是与实体 `e` 关联的第 `i` 个样本的真实标签。我们为每个实体 `e` 训练4800个独立的一对多分类器。我们使用在线训练框架，并将每个实体的工作并行化到多个工作节点上。在推理时，我们使用所有类别的模型对测试视频中的每一帧进行评分。由于我们所有的评估都基于视频级真实标签，我们需要将帧级分数（对每个实体）聚合为单个视频级分数。帧级概率使用简单的平均聚合到视频级。我们选择平均而非最大池化，是因为我们希望减少异常检测的影响并捕捉每个实体在整个视频中的突出程度。换句话说，令 `p(e|x)` 为给定特征 `x` 时实体 `e` 存在的概率。我们计算与视频 `v` 关联的实体 `e` 的概率 `p(e|x^$v_{1:F_v}$)` 为：

$$
p(e|x^v_{1:F_v}) = (1/F_v) * \Sigma_{j=1}^{F_v} p(e|x^v_j)    (1)
$$

#### 4.1.2 深度帧袋（DBoF）池化

受各种经典词袋表示方法在视频分类中成功应用的启发 [23, 36]，我们接下来考虑深度帧袋（DBoF）方法。图7展示了我们的DBoF视频分类网络的整体架构。来自视频的k个随机帧的N维输入帧级特征首先被输入到一个具有M个单元和RELU激活函数的全连接层。通常，当M > N时，输入特征被投影到更高维的空间。关键在于，全连接层的参数在k个输入帧之间是共享的。结合RELU激活，这导致输入特征在M维空间中产生稀疏编码。

获得的稀疏编码被输入到一个池化层，该层将k帧的编码聚合成一个单一的固定长度视频表示。我们使用最大池化进行聚合。我们在池化之前使用批归一化层以提高稳定性并加速收敛。获得的固定长度视频描述符现在可以使用Logistic或Softmax层（中间可加额外的全连接层）分类到输出类别中。投影层的M维可以看作M个判别聚类，它们可以通过反向传播在单个网络中进行端到端训练。

整个网络使用随机梯度下降（SGD）训练，Logistic层使用logistic损失，Softmax层使用交叉熵损失。从顶层反向传播的梯度以判别方式训练投影层的权重向量，以提供输入特征袋的强大表示。类似网络在 [26] 中被提出，其中卷积层的输出在视频的所有帧上进行池化以获得固定长度描述符。然而，[26] 中的网络没有使用中间投影层，我们发现这是从输入帧特征学习时的关键区别。需要注意的是，将特征向上投影到稀疏编码与Fisher向量 [27] 和VLAD [15] 方法所做的类似，但这里的投影（即聚类）是以判别方式进行的。我们还尝试了Fisher向量和VLAD，但在可比的码本大小下未能获得竞争性结果。

**超参数：** 我们考虑了投影层单元数取值为 {2048, 4096, 8192}，发现更大的值会带来更好的结果。我们对所有数据集使用8192。在所有实验中，我们在池化层和最终分类层之间使用一个包含1024个单元的单一隐藏层。网络使用SGD with AdaGrad训练，学习率为0.1，权重衰减惩罚为0.0005。

![图7：DBoF方法的网络架构。输入帧特征首先被输入到一个上投影层，该层对所有帧共享参数。接着是一个池化层，将帧级稀疏编码转换为视频级表示。若干隐藏层和一个分类层提供最终的视频级预测。]

#### 4.1.3 长短期记忆网络（LSTM）

我们采用类似 [26] 的方法，利用LSTM进行视频级预测。然而，与那项工作不同，我们无法访问原始视频帧。这意味着我们只能训练LSTM和Softmax层。

我们尝试了不同数量的堆叠LSTM层和隐藏单元数。我们通过实验发现，在验证集上，2层1024个单元的LSTM性能最高。与 [26] 类似，我们也采用了每帧线性递增的权重，从1/N到最后一帧的1。

在训练期间，LSTM展开60次迭代。因此，LSTM的梯度视野为60秒。我们尝试了更大的展开步数，但这会显著减慢训练过程。最终，最好的模型是训练步数最大的模型（而非实时最长的模型）。

为了将学习到的模型迁移到ActivityNet，我们使用了一个全连接模型，该模型以LSTM层在视频最后一帧计算出的输出拼接作为输入。与传统的迁移学习方法不同，我们不微调LSTM层。这种方法比传统方法更鲁棒于过拟合，这对于在ActivityNet上获得竞争性性能至关重要，因为其规模较小。我们在Sports-1M上进行了完整的微调实验，该数据集足够大，可以在预训练后微调整个LSTM模型。

### 4.2 视频级表示

除了直接在帧级特征上训练分类器，我们还探索了从每个视频 `v` 的帧级特征 `x^$v_{1:F_v}$` 中提取任务无关的固定长度视频级特征向量。提取固定长度视频特征有几个好处：

1. **标准分类器可以应用：** 由于表示的维度在各视频间是固定的，我们可以训练标准分类器，如Logistic、SVM、混合专家模型。
2. **紧凑性：** 我们获得了整个视频的紧凑表示，从而将训练数据大小减少了几个数量级。
3. **更适合领域自适应：** 由于视频级表示是无监督的（独立于标签提取），这些表示对当前数据集相关标签的特化程度要低得多，可以更好地泛化到新任务或视频领域。

形式上，视频级特征 `\phi(x^$v_{1:F_v}$)` 是一个固定长度的表示（在视频级）。我们探索了一种简单的聚合技术来获得这些视频级表示。我们还尝试了Fisher向量（FV）[27] 和VLAD [15] 方法来获得任务无关的视频级表示，但在相同维度下未能获得竞争性结果。将开发比下面描述的更简单方法性能更优的紧凑FV或VLAD类型表示留给未来工作。

#### 4.2.1 一阶、二阶和顺序统计量

从帧级特征 `x^$v_{1:F_v}$`（其中 `x^v_j \in R^1024`）中，我们提取均值 `\mu_v \in R^1024` 和标准差 `\sigma_v \in R^1024`。此外，我们还为每个维度提取前5个顺序统计量。形式上，`Top_K(x^$v_{1:F_v}$)` 返回一个K维向量，其中第p维包含特征向量第j维在整个视频中的第p高值。我们记 `Top_K(x^$v_{1:F_v}$)` 为通过连接每个维度的顺序统计量得到的KD维向量。因此，视频的结果特征向量 `\phi(x^$v_{1:F_v}$)` 为：

$$
\phi(x^v_{1:F_v}) = [ \mu(x^v_{1:F_v}); \sigma(x^v_{1:F_v}); Top_K(x^v_{1:F_v}) ]    (2)
$$

#### 4.2.2 特征归一化

特征标准化已被证明有助于在线学习算法 [14, 37]，因为它使基于SGD的算法（如Adagrad）对学习率更鲁棒，并加速收敛。

在基于视频级表示 `\phi(x^$v_{1:F_v}$)`（定义见公式2）训练我们的一对多分类器之前，我们对特征向量应用全局归一化。类似于我们对帧特征的处理方式，我们减去 `\phi(.)` 的均值，然后使用PCA去相关和白化特征。归一化后的视频特征现在近似为具有零均值和单位协方差矩阵的多变量高斯分布。这使得各维度的梯度步长相互独立，学习算法获得每个维度的无偏视图（因为相同的学习率应用于每个维度）。最后，对结果特征进行L2归一化。我们发现这些归一化技术使我们的模型训练更快。

### 4.3 基于视频特征的模型

给定视频级表示后，我们使用所有数据为每个标签训练独立的二分类器。利用各标签之间的结构信息留待未来工作。一个关键挑战是在该数据集的规模下训练这些分类器。即使对于600万训练视频采用紧凑的视频级表示，训练批量优化分类器（如SVM）也是不可行的。相反，我们使用在线学习算法，并使用Adagrad在一小批样本上对权重向量进行模型更新（每个样本关联一个二值真实标签）。

#### 4.3.1 Logistic回归

给定D维视频级特征，Logistic回归分类器的参数 `$\Theta$` 是实体特定的权重向量 `w_e`。在评分时，给定测试样本的视频级特征 `x \in R^{D+1}`，实体 `e` 的概率为 `p(e|x) = \sigma(w_e^T x)`。权重 `w_e` 通过最小化训练数据上的总对数损失获得：

$$
min_{w_e} (\lambda/2) ‖w_e‖²₂ + \Sigma_{i=1}^N L(y_{i,e}, \sigma(w_e^T x_i))    (3)
$$

其中 `\sigma(.)` 是标准logistic函数，`\sigma(z) = 1 / (1 + exp(-z))`。

#### 4.3.2 Hinge Loss

由于在此大规模数据集上训练批量SVM是不可能的，我们采用在线SVM方法。与传统SVM框架一样，我们使用$\pm$1表示负标签和正标签。给定二值真实标签 `y（0或1）` 和预测标签 `ŷ（正值或负值标量）`，hinge损失为：

$$
L(y, ŷ) = max(0, b - (2y - 1)ŷ)    (4)
$$

其中 `b` 是hinge损失参数，可进一步微调或设为1.0。由于存在max函数，一阶导数存在不连续性。这导致更新中使用次梯度，显著减慢了收敛速度。

#### 4.3.3 混合专家（MoE）

混合专家（MoE）由Jacobs和Jordan [18] 首次提出。实体 `e` 的二分类器由一组隐藏状态（或称专家）`H_e` 组成。通常使用softmax来建模选择每个专家的概率。给定一个专家，我们可以使用sigmoid来建模实体的存在性。因此，实体 `e` 存在的最终概率为：

$$
p(e|x) = \Sigma_{h\inH_e} p(h|x) \sigma(u_h^T x)
$$

其中 `p(h|x) = exp(w_h^T x) / (1 + \Sigma_{h'\inH_e} exp($w_{h'}$^T x))` 是softmax。第 `(|H_e| + 1)` 个（最后一个）状态是一个哑状态，始终导致实体不存在。记 `$p_{y|x}$ = p(y=1|x)`，`$p_{h|x}$ = p(h|x)`，`p_h = p(y=1|x,h)`。给定一组训练样本 `(x_i, g_i)_{i=1...N}` 用于二分类器，其中 `x_i` 是特征向量，`g_i \in {0,1}` 是真实标签，令 `L(p, g)` 为预测概率与真实标签之间的对数损失：

$$
L(p, g) = -g log p - (1-g) log(1-p)    (5)
$$

我们可以直接写出 `L($p_{y|x}$, g)` 关于softmax权重 `w_h` 和logistic权重 `u_h` 的导数：

$$
\partialL(p_{y|x}, g) / \partialw_h = x p_{h|x} (p_{y|h,x} - p_{y|x}) (p_{y|x} - g) / (p_{y|x} (1 - p_{y|x}))    (6)

\partialL(p_{y|x}, g) / \partialu_h = x p_{h|x} p_{y|h,x} (1 - p_{y|h,x}) (p_{y|x} - g) / (p_{y|x} (1 - p_{y|x}))    (7)
$$

我们使用学习率为1.0、批量大小为32的Adagrad来学习权重。由于我们为每个标签训练独立的分类器，工作被分布到多台机器上。

对于MoE模型，我们尝试了不同数量的混合（1、2、4），发现从1个混合增加到2个再到4个时，所有指标的性能提升约0.5%-1%，但模型参数数量相应地增加了2倍或4倍。我们选择2个混合作为良好的折中方案，并为所有数据集报告2混合MoE模型的结果。

## 5. 实验

在本节中，我们首先在上述多标签分类方法上提供YouTube-8M数据集的基准基线结果。然后，我们评估在此数据集上学习的视频表示对其他任务（如Sports-1M体育分类和ActivityNet活动分类）的实用性。

### 5.1 评估指标

**均值平均精度（mAP）：** 对每个实体，我们首先将标注分数按10^-4的桶进行四舍五入，并根据模型分数对所有非零标注进行排序。在给定阈值\tau下，精度P(\tau)和召回率R(\tau)为：

$$
P(\tau) = \Sigma_{t\inT} I(y_t \geq \tau) g_t / \Sigma_{t\inT} I(y_t \geq \tau)    (8)
R(\tau) = \Sigma_{t\inT} I(y_t \geq \tau) g_t / \Sigma_{t\inT} g_t    (9)
$$

其中 `I(.)` 是指示函数。平均精度AP（近似精度-召回曲线下的面积）可计算为：

$$
AP = \Sigma_{j=1}^{10000} P(\tau_j) [R(\tau_j) - R(\tau_{j+1})]    (10)
$$

其中 `\tau_j = j/10000`。均值平均精度计算为所有类别平均精度的未加权均值。

**Hit@k：** 这是测试样本中包含至少一个真实标签在前k个预测中的比例。如果 `ran$k_{v,e}$` 是实体e在视频v上的排名（得分最高的实体排名为1），`G_v` 是v的真实标签集合，则Hit@k可写为：

$$
(1/|V|) \Sigma_{v\inV} ∨_{e\inG_v} I(rank_{v,e} \leq k)    (11)
$$

其中 `∨` 是逻辑OR。

**等召回率精度（PERR）：** 我们衡量在检索与真实标签相同数量的实体时，视频级标注的精度。使用与Hit@k相同的表示法，PERR可写为：

$$
(1/|{v: |G_v| > 0}|) \Sigma_{v\inV: |G_v| > 0} (1/|G_v|) \Sigma_{e\inG_v} I(rank_{v,e} \leq |G_v|)    (12)
$$

### 5.2 YouTube-8M上的结果

表3展示了YouTube-8M数据集上所有方法的结果。基于帧级的模型（第1行），在强大的Inception特征上训练Logistic回归，然后对所有帧的预测进行简单平均，在该数据集上表现不佳。这表明视频级预测任务不能简化为简单的帧级分类。

使用帧级特征的简单均值池化将帧级特征聚合到视频级，然后使用hinge损失或Logistic回归模型，在视频级精度上比帧级预测的朴素平均有显著改进。通过使用混合专家模型并添加其他统计量（如标准差和顺序特征），可以观察到进一步改进。需要注意的是，标准偏差和顺序统计量在原始的RELU激活空间中更有意义，因此我们从PCA和量化后的特征中重建RELU特征（通过使用提供的PCA矩阵逆量化及逆PCA，在重建的帧级RELU特征上计算集合统计量，然后重新应用第4.2.2节描述的PCA、白化和L2归一化）。这种简单的任务无关特征池化和归一化策略在该数据集上产生了最具竞争力的结果。

最后，我们还评估了两种在前人基准 [26] 上产生最先进结果的深度网络架构。DBoF架构忽略序列信息，将输入视频视为帧袋，而LSTM则使用状态信息保留视频序列。采用Logistic分类层的DBoF方法在Hit@1和PERR指标上比使用简单均值特征池化和单层Logistic模型提升了2%（绝对），这显示了判别式训练投影层以获得任务特定视频级表示的好处。DBoF的mAP结果略逊于均值池化+Logistic模型，我们将其归因于DBoF在稀有类别上的训练和收敛速度较慢（mAP受稀有类别结果的影响很大，而DBoF的联合类别训练对这些类别不利）。

LSTM网络通常表现最佳，但mAP除外，在该指标上一对多二值MoE分类器表现更好，原因可能同样是稀有类别上的收敛速度较慢。LSTM在Hit@1和PERR指标上确实有提升，这符合预期，因为它能够学习时间域上的长期相关性。此外，在 [26] 中，作者通过从视频中采样多个固定长度的片段并平均结果来进行数据增强，这可能会产生比我们当前结果更好的精度。

我们还考虑了Fisher向量和VLAD，因为它们最近在 [39] 中被成功用于在视频级聚合CNN特征。然而，在与LSTM、DBoF和均值特征等视频级表示相同维度下，它们并未产生竞争性结果。

| 输入特征 | 建模方法 | mAP | Hit@1 | PERR |
|---------|---------|:---:|:-----:|:----:|
| 帧级, {x^v_{1:F_v}} | Logistic回归 + 平均池化 (4.1.1) | 12.0 | 56.9 | 48.7 |
| 帧级, {x^v_{1:F_v}} | 深度帧袋(DBoF) (4.1.2) | 16.0 | 62.5 | 52.1 |
| 帧级, {x^v_{1:F_v}} | LSTM (4.1.3) | 26.6 | 64.5 | 57.3 |
| 视频级, \mu | Hinge损失 (4.3) | 17.0 | 56.3 | 47.9 |
| 视频级, \mu | Logistic回归 (4.3) | 28.1 | 60.5 | 53.0 |
| 视频级, \mu | 2混合专家 (4.3) | 29.6 | 62.3 | 54.9 |
| 视频级, [\mu; \sigma; Top_5] | 2混合专家 (4.3) | 30.0 | 63.3 | 55.8 |

**表3：YouTube-8M数据集上各种基准基线的结果。我们发现，在简单视频级表示上使用二分类器的方法明显优于帧级方法。深度学习方法（如DBoF和LSTM）相比传统密集特征聚合方法并未带来显著提升，因为底层帧级特征已经非常强大。**

#### 5.2.1 人工评估测试集

我们还在表4中报告了三种最佳方法在超过8000个视频的人工评估测试集（见第3.5节）上的结果。由于测试集大小原因，mAP不可靠，我们报告PERR、Hit@1和Hit@5。所有方法的Hit@1数字相较于表3中标注不完整的验证集均普遍更高，而PERR数字则普遍更低。这在很大程度上归因于验证集中的缺失标签（验证集标签的召回率约为15%，而详尽的人工评估约为100%）。然而，各种方法的相对排序在两个数据集之间相当一致，表明验证集结果对于比较不同方法仍然足够可靠。

| 方法 | Hit@1 | PERR | Hit@5 |
|------|:-----:|:----:|:-----:|
| 深度帧袋(DBoF) (4.1.2) | 68.6 | 29.0 | 83.5 |
| LSTM (4.1.3) | 69.1 | 30.5 | 84.7 |
| 2混合专家([\mu;\sigma;Top_5]) (4.3) | 70.1 | 29.1 | 84.8 |

**表4：三种最佳方法在YouTube-8M数据集人工评估测试集上的结果。与验证集结果（表3）的比较表明，各种方法的相对优势在两个数据集上基本保持一致。**

### 5.3 Sports-1M上的结果

接下来，我们研究使用YouTube-8M数据集学习的视频级特征的泛化性，并在Sports-1M数据集上进行迁移学习实验。Sports-1M数据集 [19] 包含487个体育活动类别，共120万个YouTube视频，是体育/活动识别领域最大的基准之一。我们在所有实验中使用以每秒1帧采样的视频前360秒。

为了在该数据集上评估迁移学习，在一个实验中，我们简单地使用基于YouTube-8M数据集学习到的PCA矩阵聚合的视频级描述符，并在其上使用目标域训练数据训练MoE或Logistic模型。

对于LSTM网络，我们有两种方案：1）使用PCA转换后的特征，从头开始学习LSTM模型；2）使用在YouTube-8M任务上预训练的LSTM层，并在Sports-1M数据集上对其进行微调（同时使用新的softmax分类器）。

表5a展示了Sports-1M数据集上各种视频级表示的评估指标。我们学习到的特征在该数据集上具有竞争力，最佳方法除 [26] 的方法外优于所有其他方法。[26] 的方法直接从Sports-1M数据集视频的像素中学习，包括光流，并使用了数据增强策略以及对多个视频片段进行多次推理。我们还表明，即使在这样的数据集（100万个视频）上，在YouTube-8M上预训练仍然有助于提升性能，使LSTM在所有指标上的性能提升约1%（相对于无预训练）。

| 方法 | mAP | Hit@1 | Hit@5 |
|------|:---:|:-----:|:-----:|
| Logistic回归(\mu) (4.3) | 58.0 | 60.1 | 79.6 |
| 2混合专家(\mu) (4.3) | 59.1 | 61.5 | 80.4 |
| 2混合专家([\mu;\sigma;Top_5]) (4.2.1) | 61.3 | 63.2 | 82.6 |
| LSTM (4.1.3) | 66.7 | 64.9 | 85.6 |
| + 在YT-8M上预训练 (4.1.3) | 67.6 | 65.7 | 86.2 |
| 层次化3D卷积 [19] | - | 61.0 | 80.0 |
| 堆叠3D卷积 [35] | - | 61.0 | 85.0 |
| 带光流和像素的LSTM [26] | - | 73.0 | 91.0 |

**表5a：Sports-1M：我们学习到的特征在该数据集上具有竞争力，胜过除 [26] 之外的所有方法。[26] 的方法直接从视频像素学习，[26] 和 [35] 均包含运动特征。**

### 5.4 ActivityNet上的结果

我们最后的实验展示了学习到的特征在ActivityNet未修剪视频分类任务上的泛化能力。与Sports-1M实验类似，我们比较了直接在ActivityNet数据集上训练与在YouTube-8M上预训练后使用聚合方法和LSTM方法的效果。如表5b所示，所有迁移后的特征在所有指标上都远优于仅在ActivityNet上训练。值得注意的是，即使不使用运动信息，我们最好的特征也比 [12] 中使用的HOG、HOF、MBH、FC-6、FC-7特征好最多80%。这一结果表明在YouTube-8M上学习到的特征对其他数据集/任务有很好的泛化能力。我们认为这是因为YouTube-8M中视频的多样性和大规模。

| 方法 | mAP | Hit@1 | Hit@5 |
|------|:---:|:-----:|:-----:|
| 2混合专家(\mu) (4.3) | 69.1 | 68.7 | 85.4 |
| + 在YT-8M上预训练PCA | 74.1 | 72.5 | 89.3 |
| 2混合专家([\mu;\sigma;Top_5]) (4.2.1) | 74.2 | 72.3 | 89.6 |
| + 在YT-8M上预训练PCA | 77.6 | 74.9 | 91.6 |
| LSTM (4.1.3) | 57.9 | 63.4 | 81.0 |
| + 在YT-8M上预训练 (4.1.3) | 75.6 | 74.2 | 92.4 |
| Ma, Bargal 等 [24] | 53.8 | - | - |
| Heilbron 等 [12] | 43.0 | - | - |

**表5b：ActivityNet：由于数据集较小，通过在YouTube-8M上预训练或使用迁移学习的PCA（而非在ActivityNet上从头学习的PCA），我们看到了显著的性能提升。**

## 6. 结论

在本文中，我们介绍了YouTube-8M，一个用于视频分类和表示学习的大规模视频基准。通过YouTube-8M，我们的目标是推动视频理解领域的发展，正如大规模图像数据集为图像理解所做的那样。具体来说，我们解决了大尺度视频理解的两个主要挑战——（1）收集大规模标记视频数据集，具有合理的标签质量；（2）通过预处理数据集并提供最先进的帧级特征来消除计算障碍。我们处理了超过50年时长的视频，为超过800万个视频的近20亿帧提供了特征，这使得使用单台机器上的开源框架在1天内训练一个合理规模的模型成为可能！我们期望该数据集能够为学术界研究人员创造公平的竞争环境，弥合与大尺度标记视频数据集之间的差距，并显著加速视频理解的研究。我们希望该数据集将成为开发新型视频表示学习算法，尤其是有效处理噪声或不完整标签方法的测试平台。

作为副产品，我们还提供了最大且最多样化的公共视觉标注词汇表之一（包含4800个视觉知识图谱实体），该词汇表基于YouTube上的流行度信号和人工筛选构建，并组织成24个顶级类别。

我们提供了广泛的实验，比较了多种强大的视频表示学习基线，包括深度网络和LSTM在该数据集上的表现。我们展示了一类相当未被充分探索的模型（混合专家模型）的有效性，并表明它们可以胜过Logistic回归和SVM等流行分类器。这对于我们的数据集尤为如此，其中许多类别可能是多模态的。我们探索了使用从帧级特征中提取的简单统计量的各种视频级表示，并将给定聚合向量的实体概率建模为MoE。我们展示了这种方法在与更复杂的方法（直接使用帧级信息的方法，如LSTM和DBoF）相比时具有竞争性能。这也表明，如果底层帧级特征足够强，对更复杂的视频级建模技术的需求就会减少。

最后，我们通过在现有视频基准——Sports-1M和ActivityNet——上进行迁移学习实验，说明了该数据集的实用性。我们的实验表明，在此数据集上学习到的特征在这些基准上具有良好的泛化能力，包括在ActivityNet上取得了新的最先进结果。

## 7. 参考文献

[1] Freebase: A community-curated database of well-known people, places, and things. https://www.freebase.com.

[2] Google I/O 2013 - semantic video annotations in the Youtube Topics API: Theory and applications. https://www.youtube.com/watch?v=wf_77z1H-vQ.

[3] Knowledge Graph Search API. https://developers.google.com/knowledge-graph/.

[4] Tensorflow: Image recognition. https://www.tensorflow.org/tutorials/image_recognition.

[5] M. Blank, L. Gorelick, E. Shechtman, M. Irani, and R. Basri. Actions as space-time shapes. In *Proceedings of the International Conference on Computer Vision (ICCV)*, 2005.

[6] J. Deng, W. Dong, R. Socher, L.-jia Li, K. Li, and L. Fei-fei. Imagenet: A large-scale hierarchical image database. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 2009.

[7] M. Everingham, L. V. Gool, C. K. I. Williams, J. Winn, and A. Zisserman. The pascal visual object classes (voc) challenge, 2009.

[8] L. Fei-fei, R. Fergus, and P. Perona. One-shot learning of object categories. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 28, 2006.

[9] R. Girshick. Fast R-CNN. In *Proceedings of the International Conference on Computer Vision (ICCV)*, 2015.

[10] G. Griffin, A. Holub, and P. Perona. Caltech-256 object category dataset. Technical Report 7694, California Institute of Technology, 2007.

[11] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. *CoRR*, abs/1512.03385, 2015.

[12] F. C. Heilbron, V. Escorcia, B. Ghanem, and J. C. Niebles. Activitynet: A large-scale video benchmark for human activity understanding. In *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 961–970, 2015.

[13] S. Hochreiter and J. Schmidhuber. Long short-term memory. *Neural Computing*, 9(8), Nov. 1997.

[14] S. Ioffe and C. Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In *Proceedings of the International Conference on Machine Learning (ICML)*, pages 448–456, 2015.

[15] H. Jegou, F. Perronnin, M. Douze, J. Sanchez, P. Perez, and C. Schmid. Aggregating local image descriptors into compact codes. *IEEE Trans. Pattern Anal. Mach. Intell.*, 34(9), Sept. 2012.

[16] Y. Jiang, J. Liu, A. Roshan Zamir, G. Toderici, I. Laptev, M. Shah, and R. Sukthankar. THUMOS challenge: Action recognition with a large number of classes. http://crcv.ucf.edu/THUMOS14, 2014.

[17] Y.-G. Jiang, Z. Wu, J. Wang, X. Xue, and S.-F. Chang. Exploiting feature and class relationships in video categorization with regularized deep neural networks. *arXiv preprint arXiv:1502.07209*, 2015.

[18] M. I. Jordan. Hierarchical mixtures of experts and the em algorithm. *Neural Computation*, 6, 1994.

[19] A. Karpathy, G. Toderici, S. Shetty, T. Leung, R. Sukthankar, and L. Fei-Fei. Large-scale video classification with convolutional neural networks. In *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 1725–1732, Columbus, Ohio, USA, 2014.

[20] A. Krizhevsky, I. Sutskever, and G. E. Hinton. ImageNet classification with deep convolutional neural networks. In *Advances in Neural Information Processing Systems (NIPS)*, pages 1097–1105, 2012.

[21] H. Kuehne, H. Jhuang, E. Garrote, T. Poggio, and T. Serre. Hmdb: a large video database for human motion recognition. In *Proceedings of the International Conference on Computer Vision (ICCV)*, 2011.

[22] I. Laptev and T. Lindeberg. Space-time interest points. In *Proceedings of the International Conference on Computer Vision (ICCV)*, 2003.

[23] I. Laptev, M. Marszalek, C. Schmid, and B. Rozenfeld. Learning realistic human actions from movies. In *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2008.

[24] S. Ma, S. A. Bargal, J. Zhang, L. Sigal, and S. Sclaroff. Do less and achieve more: Training cnns for action recognition utilizing action images from the web. *CoRR*, abs/1512.07155, 2015.

[25] V. Mnih and G. Hinton. Learning to label aerial images from noisy data. In *Proceedings of the 29th Annual International Conference on Machine Learning (ICML)*, June 2012.

[26] J. Y.-H. Ng, M. J. Hausknecht, S. Vijayanarasimhan, O. Vinyals, R. Monga, and G. Toderici. Beyond short snippets: Deep networks for video classification. In *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 4694–4702, 2015.

[27] F. Perronnin and C. Dance. Fisher kernels on visual vocabularies for image categorization. In *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2007.

[28] A. Quattoni and A. Torralba. Recognizing indoor scenes. In *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2009.

[29] S. Reed, H. Lee, D. Anguelov, C. Szegedy, D. Erhan, and A. Rabinovich. Training deep neural networks on noisy labels with bootstrapping. *arXiv e-prints*, Dec. 2014.

[30] O. Russakovsky, J. Deng, H. Su, J. Krause, S. Satheesh, S. Ma, Z. Huang, A. Karpathy, A. Khosla, M. Bernstein, A. C. Berg, and L. Fei-Fei. ImageNet Large Scale Visual Recognition Challenge. *International Journal of Computer Vision (IJCV)*, 115(3):211–252, 2015.

[31] P. Sermanet, D. Eigen, X. Zhang, M. Mathieu, R. Fergus, and Y. LeCun. Overfeat: Integrated recognition, localization and detection using convolutional networks. In *International Conference on Learning Representations (ICLR)*.

[32] J. Shotton, J. Winn, C. Rother, and A. Criminisi. Textonboost: Joint appearance, shape and context modeling for multi-class object. In *Proceedings of the European Conference on Computer Vision (ECCV)*, 2006.

[33] K. Soomro, A. R. Zamir, and M. Shah. UCF101: A dataset of 101 human actions classes from videos in the wild. In *CRCV-TR-12-01*, 2012.

[34] B. Thomee, D. A. Shamma, G. Friedland, B. Elizalde, K. Ni, D. Poland, D. Borth, and L. Li. The new data and new challenges in multimedia research. *CoRR*, abs/1503.01817, 2015.

[35] D. Tran, L. D. Bourdev, R. Fergus, L. Torresani, and M. Paluri. C3D: generic features for video analysis. *CoRR*, abs/1412.0767, 2014.

[36] H. Wang, M. M. Ullah, A. Kläser, I. Laptev, and C. Schmid. Evaluation of local spatio-temporal features for action recognition. In *Proc. BMVC*, 2009.

[37] S. Wiesler, A. Richard, R. Schlüter, and H. Ney. Mean-normalized stochastic gradient for large-scale deep learning. In *IEEE International Conference on Acoustics, Speech and Signal Processing, ICASSP 2014, Florence, Italy, May 4-9, 2014*, pages 180–184. IEEE, 2014.

[38] J. Xiao, K. A. Ehinger, J. Hays, A. Torralba, A. Oliva, and J. Xiao. Sun database: Exploring a large collection of scene categories, 2013.

[39] Z. Xu, Y. Yang, and A. G. Hauptmann. A discriminative cnn video representation for event detection. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 2015.

[40] H.-F. Yu, P. Jain, P. Kar, and I. Dhillon. Large-scale multi-label learning with missing labels. In *Proceedings of The 31st International Conference on Machine Learning (ICML)*, pages 593–601, 2014.

[41] M. D. Zeiler and R. Fergus. Visualizing and understanding convolutional networks. *CoRR*, abs/1311.2901, 2013.

## 参考文献
