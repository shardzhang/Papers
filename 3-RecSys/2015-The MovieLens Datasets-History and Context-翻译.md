```
本文分享了MovieLens数据集的完整历史与使用指南，涵盖1997年至2015年间的系统演进与数据集发布。核心内容：
- MovieLens系统从v0到v4的五次重大版本迭代及其对数据的影响
- 四个公开数据集（100K/1M/10M/20M）的采样方法、统计特征与适用场景
- 关键功能（浏览、评分、引导、标签、电影编辑、用户互动）的演进历程
- 运行长期在线研究平台的经验教训与最佳实践

关键发现：
- 界面设计变化（如半星评分）会直接改变评分分布，影响数据集特征
- 数据集仅包含评分≥20次的"成功用户"，存在天然选择偏差
- 评分时间戳不代表观看时间，大量回溯性评分降低了时间戳价值
- 20M数据集提供了到外部元数据源的链接，适合结合内容特征的研究
- MovieLens仍是推荐系统领域使用最广泛的基准数据集之一
```

# The MovieLens Datasets: History and Context（中文翻译）

> F. Maxwell Harper, Joseph A. Konstan
> 明尼苏达大学
> ACM TISS 2016

## 摘要

MovieLens 数据集是推荐系统研究中使用最广泛的基准数据集之一。本文介绍了 MovieLens 数据集的历史背景，描述数据收集方法，并给出统计特征。我们讨论了 MovieLens 这一在线推荐系统从1997年至今的关键变化，以及这些变化如何影响用户行为和由此产生的数据集。我们还提供了使用这些数据集进行研究和教育的最佳实践建议。

**关键词**：推荐系统；协同过滤；数据集；MovieLens

## 1. 引言

1997年，明尼苏达大学的研究人员向公众发布了第一个 MovieLens 数据集，其中包含100,000个电影评分。该数据集供学术研究人员和计算机教育工作者使用，提供了一个标准的基准，可用于比较推荐算法。在随后的几年中，GroupLens 研究实验室又发布了三个主要数据集——1M（2003年）、10M（2009年）和20M（2015年）——每个数据集都包含更多的评分，且反映了 MovieLens 在线推荐系统的持续发展和变化。

许多因素促使 MovieLens 数据集取得成功。由于电影评分的数据收集程序相对简单，且电影是许多人感兴趣的主题，该数据集被迅速采用。或许是因为 MovieLens 在推荐系统研究中扮演了关键角色，其成果不仅惠及推荐技术，也惠及更广泛的数据科学技术家族，包括摘要、模式识别和可视化。此外，由于电影偏好高度受个人品味影响，电影领域非常适合测试个性化技术。最后，这种流行可能也反映了人们对电影作为内容领域的可及性认知：电影是共同的兴趣，使得算法输出易于讨论。

本文探讨了 MovieLens 系统的历史，以记录影响最终数据集的各种因素。在此过程中，我们还分享了从长期运行的研究平台中获得的经验教训，并记录了使用这些数据集进行研究的最佳实践。本文包含两个主要部分。第2节分享 MovieLens 系统的历史和经验教训；第3节则介绍数据集的描述和使用指南。

## 2. MovieLens 系统

MovieLens 数据集是用户多年来与 MovieLens 在线推荐系统交互的结果。与大多数长期运行且动态变化的在线系统一样，MovieLens 经历了许多变化——无论是在设计上还是功能上。这些变化必然会影响评分的生成：用户只能对出现在屏幕上的电影进行评分；用户的评分也会受到预测评分本身的影响 [Cosley et al. 2003]。

在本节中，我们介绍 MovieLens 的历史，重点说明那些最可能对数据集内容产生影响的变化。

### 2.1 起源与里程碑

明尼苏达大学的研究人员于1997年夏天开始开发 MovieLens，起因是数字设备公司（DEC）决定关闭他们自己的电影推荐系统 EachMovie。当时，DEC 联系了推荐系统社区，寻找一个组织来开发替代站点，以延续同样的使命，并可能服务于同一批用户（但不使用 DEC 的专有代码）；GroupLens 主动承担了这一任务。法律问题阻止了直接转移用户账户，但 DEC 确实向 GroupLens 转移了一个匿名数据集，GroupLens 使用该数据集训练了第一个版本的 MovieLens 推荐器。

MovieLens 于1997年秋季上线。第一个版本尽可能开发得与它所替代的 EachMovie 界面相似（见图5截图）。EachMovie 使用了专有推荐算法；MovieLens 则采用了 GroupLens Usenet 新闻推荐系统中实现的用户-用户协同过滤（CF）[Konstan et al. 1997]。

1999年末，MovieLens 的使用量显著增加，当时它受到了大众媒体的关注。这些事件——Malcolm Gladwell 在《纽约客》杂志上发表的一篇文章 [Gladwell 1999]、美国广播公司《夜线》节目的一期报道，以及著名影评人 Roger Ebert 的好评——都突出地将 MovieLens 作为在线计算和个性化未来的一个例子。

自那以后，MovieLens 系统的增长一直非常稳定，尤其是考虑到几乎完全没有营销投入。MovieLens 长期以来平均每天有20到30个新用户注册，这主要归功于口碑传播或主动报道。图1展示了 MovieLens 自上线以来的增长情况，图2展示了月度活跃用户数量，图3展示了每个日历年中登录超过 n 次的用户数量，图4展示了评分数据随时间分布的情况。

ACM 交互式智能系统交易，第5卷，第4期，文章19，出版日期：2015年12月。

---

**图1. movielens.org 17年增长视图，标注了事件 A、B 和 C。** 用户注册和评分活动在此期间呈现稳定增长，其中媒体覆盖导致了一次加速（A）。当电影添加过程开放给社区后，添加到 MovieLens 的电影速率增长了（B）。随着"标签表达"功能的发布，标签应用出现了加速（C）。

**图2. 每月登录 MovieLens 的用户数量。** 虚线（1998年7月）之前的数据缺失。

稳定增长并不意味着 MovieLens 自90年代末以来没有变化。事实上，自最初发布以来，MovieLens 经历了五次重大版本（v0–v4），每个版本都代表了服务器端和客户端代码的完整重实现。我们在表 I 中总结了最重要的变化，包括日期和 MovieLens 一些最重要变更的简要总结。此外，为了提供视觉参考，我们在图5至图8中提供了历史截图，并在图9中提供了当前界面的截图。

在本节的剩余部分，我们讨论 MovieLens 功能的关键变化，这些变化最能影响 MovieLens 数据集的形态和研究用途。

ACM 交互式智能系统交易，第5卷，第4期，文章19，出版日期：2015年12月。

---

**图3. 每个完整日历年中登录 MovieLens 至少 n 次的用户比例。** 例如，2014年，访问 MovieLens 的用户中有10%登录了25次或更多。

**图4. 评分分布，按月聚合。** 实线代表每月的平均评分。灰色区域覆盖平均值加减该月评分的标准差。虚线显示每月的评分中位数。两条垂直虚线分别代表 v3（半星评分）和 v4（更简单的星级评分小部件）的发布。

### 2.2 关键功能的变化

#### 2.2.1 浏览功能

搜索、筛选和排序电影的机制会影响用户能够评分的电影，因此对 MovieLens 数据集形态具有根本性影响。MovieLens 的核心用户体验围绕评分与查看推荐之间的反馈循环：用户在推荐列表中查看项目并对其进行评分，这反过来又改变了后续页面视图中显示的项目。

这种评分/推荐循环由协同过滤算法驱动 [Resnick et al. 1994]。在 v4 发布之前，MovieLens 严格根据用户个性化的预测评分值——其底层协同过滤算法的输出——对电影列表进行排序。换句话说，用户搜索结果中最先出现的电影，是算法预测该用户评分最高的电影。v4 则结合了流行度因素与预测评分来排序推荐列表 [Ekstrand et al. 2015; Harper et al. 2015]。

算法本身也随时间发生了变化，这对用户满意度和贡献率产生了大多尚未被研究的影响：

**表 I. MovieLens 里程碑**

| 日期 | 里程碑 |
|------|--------|
| 1997年8月 | v0 界面；EachMovie 种子数据；GroupLens 用户-用户推荐器 |
| 1999年9月 | v1 界面 |
| 1999年11月 | 媒体曝光；NetPerceptions 用户-用户推荐器 |
| 2000年2月 | v2 界面；电影分组；额外外部电影元数据（例如票房、DVD 发行）；评论 |
| 2003年2月 | v3 界面；multilens 项目-项目推荐器；Ajax 评分小部件 |
| 2005年6月 | 讨论论坛 |
| 2005年12月 | 标签功能 |
| 2008年9月 | 基于会员的电影添加 |
| 2012年1月 | LensKit 项目-项目推荐器 |
| 2014年11月 | v4 界面；用户可选择的推荐器；外部电影数据 |

注：MovieLens 自1997年发布以来一直在不断变化；这些是其历史上一些最有意义的事件。

**图5. MovieLens v0 截图，约1998年。**

—1997年：通过 GroupLens Usenet 推荐器实现用户-用户协同过滤 [Konstan et al. 1997]
—1999年：通过 NetPerceptions 实现用户-用户协同过滤³
—2003年：通过 MultiLens 实现项目-项目协同过滤 [Miller 2003]
—2012年：通过 LensKit 实现项目-项目协同过滤 [Ekstrand et al. 2011]
—2014年：用户可选择 [Ekstrand et al. 2015] 非个性化算法、支持新用户的推荐器 [Chang et al. 2015]、项目-项目协同过滤以及通过 LensKit 实现的 FunkSVD

MovieLens 的主页是用户关注的焦点，也是向用户展示顶级推荐电影的最佳位置。早期版本（v0、v1）只是在此页面上显示顶级推荐列表。v2 和 v3 更为复杂，显示多个预筛选的精选列表（近期在影院上映的电影、近期 DVD/VHS 发行的电影），以及指向不同站点功能的链接列表。v4 则简单地显示列表的列表，页面顶部以整体"精选推荐"为首。v2 到 v4 都鼓励用户开始探索电影数据库的特定切片，不过可用的搜索定制化功能随时间不断增长。

MovieLens 界面一直支持搜索和筛选操作，允许用户将视图限制在特定的电影组。所有版本都支持标题搜索；v4 包含自动完成功能。从 v0 开始，界面支持按电影类型和发行日期筛选。v3 进一步扩展了筛选选项，允许用户按演员、导演和其他属性筛选电影。2004年中，MovieLens 添加了一个"高级搜索"页面，提供了一个极具表现力（且复杂）的控制面板来查找电影。v4 发布时搜索筛选器较少，但首页上预定义搜索的数组得到了扩展，并带有预览。

许多推荐界面提供"类似推荐"功能；MovieLens 在2009年底开始提供该功能的一个版本，引入了"电影调谐器" [Vig et al. 2012]。该功能允许用户以不同方式导航信息空间，直接在电影之间跳转，不受主要浏览界面的影响。该功能在 v4 中以不同的界面设计保留下来，是 MovieLens 中唯一使用内容驱动算法（使用标签）而非基于评分技术来计算相似度的功能。

MovieLens 用户在网站上通常只能访问有限的电影元数据集。直到2005年中，电影在列表中出现时仅带有非常基本的信息，如发行日期和演员。2005年中，MovieLens 在网站设计中添加了"电影详情"页面，以支持与讨论论坛的链接（见第2.2.6节）。2005年底电影标签功能的添加（见第2.2.4节）为描述电影提供了客观和主观数据。2009年春季，MovieLens 与 Netflix 集成（Netflix 在2014年底停止支持该 API），将海报艺术和剧情简介纳入电影详情页面。v4 将 MovieLens 与 The Movie Database（TMDb）集成，将剧情摘要、电影艺术作品和预告片直接引入网站。

³Net Perceptions 是一家推荐系统公司，由 GroupLens 的教师和学生于1996年共同创立。

**图6. MovieLens v1 截图，约1999年。**

**图7. MovieLens v2 截图，约2000年。**

**图8. MovieLens v3 截图，约2003年。**

#### 2.2.2 评分功能

自 MovieLens 上线以来，评分一直以"星"值的形式表达，这是用户输入偏好的标准用户界面模式。评分界面最大的变化发生在 v3（2003年2月）发布时，界面从"整星"评分转变为"半星"评分——这是用户调查中用户最需要的功能——将偏好值范围从五个（1–5）翻倍到十个（0.5–5.0）。v3 的发布还升级了评分小部件，使其能够异步提交值，无需用户点击提交按钮。

v0 到 v3 版本在评分小部件上采用了两个独立的视觉元素。视图元素是一张显示一定数量星星的图片，颜色代表预测值或实际评分。输入元素是一个 HTML `<select>` 元素，允许用户选择星值。随着 v4 的发布，用户界面将这两个元素组合成一个接受触摸/点击事件的五星表示形式。

在 v1 到 v3 的整个生命周期中，每个屏幕顶部都有一个描述不同星值的小图例（见图6至图8）。v3 中描述星值的标签发生了变化。考虑到大量文献指出的锚定效应（例如，Lynch et al. [1991]），这些标签可能对评分行为产生了实质性影响，但尚无关于 MovieLens 中该效应的实证研究。在粗略的分析层面上，图4显示这些界面变化并未导致评分分布出现明显的全局性变化。

#### 2.2.3 引导机制

MovieLens 所基于的评分协同过滤算法在用户向系统提供一些初始评分之前效果不佳 [Rashid et al. 2002]。在 v4 发布之前，MovieLens 要求用户跨越一定的评分障碍 [Drenner et al. 2006] 才能进入主系统。

从 v0 开始的早期版本要求用户在进入系统之前评分5部电影。MovieLens 每次向用户显示10部电影，其中9部从数据库中随机选择，第10部从手工挑选的高知名度电影列表中选择 [Rashid et al. 2002]。

在 v3 中，该流程改为要求用户提供15个评分。用户仍然每次看到10部电影，但现在电影是根据流行度选择的，排除了最流行的50到150部电影。其他方法也曾在实验目的下进行过短暂评估 [Rashid et al. 2008]，但未产生持久影响。

v4 放弃了15个评分的要求，转而开发了一个特殊的"新用户推荐器"，基于更快速的群体个性化过程 [Chang et al. 2015]。我们尚不了解放弃这个障碍的长期收益和成本。

**图9. MovieLens v4 截图，2015年。**

#### 2.2.4 标签功能

MovieLens 于2005年12月引入了标签功能；标签出现在后来的 MovieLens 数据集（10M 和 20M）中。该功能允许用户将标签——单词或短语——应用到电影上。MovieLens 在电影旁边显示标签。标签可点击，以显示已应用该标签的电影列表。

早期版本中标签的可见性和排序因用户而异，以提供 A/B 测试数据 [Sen et al. 2006]，但到2006年，界面进行了整合，根据已发表的度量标准按标签为"高价值"的可能性进行排序 [Sen et al. 2009]。2007年1月，MovieLens 推出了标签评分功能，在标签旁边放置了可点击的"赞"和"踩"图标 [Sen et al. 2007]。

2009年春季，标签界面获得了一个名为"标签表达"的新功能 [Vig et al. 2010]，该功能显著影响了标签行为。该界面允许用户更轻松地对电影重新标签，导致标签活动速率增加，以及标签多样性增加。

#### 2.2.5 电影添加与编辑

电影的存在与否塑造了评分和标签行为，相关元数据的准确性和完整性也同样如此。

早期版本的 MovieLens 依赖一小群管理员和一位内容专家来管理电影数据库。在此阶段，MovieLens 代表了一个相对狭窄的电影宇宙：那些在美国广泛影院发行的电影。

从2004年中开始，会员对数据库的控制增加了。v3 发布后不久，MovieLens 添加了一个允许会员建议标题的链接；2004年中，MovieLens 开始允许会员直接编辑电影详情 [Cosley et al. 2005]。在早期实验（约2004年）允许会员添加电影之后，电影添加界面于2008年底成为系统的永久功能。在 v4 中，元数据的控制从会员输入转移到外部来源——http://themoviedb.org。

#### 2.2.6 用户互动

MovieLens 界面一直强调与电影的互动而非与其他用户的互动。然而，在 MovieLens 的历史上，一些功能曾出现过又消失，将用户的注意力引向系统更多的社交用途。

MovieLens 提供过两个版本的专为群体接收联合推荐而设计的功能。v2 包含一个称为"群组"的功能 [O'Connor et al. 2001]，这些是持久性的命名用户集合，可以接收联合优化的推荐。该结构在 v3 中重新设计，此时（改称为"好友"）重新定位为更像社交网络。这一重新设计认识到群组创建太难且群组太不灵活——用户可以建立成对关系，并决定在请求推荐时包含哪些好友。好友功能被大约9,000名用户使用，每位用户与中位数为1个其他 MovieLens 用户建立连接（平均值=1.8）。

MovieLens 提供了三个版本的旨在支持围绕电影对话的功能。第一个版本与 v2 一起发布，是一个简单的线程讨论论坛。该版本与主站点没有深度连接，几乎没有被使用。它在 v3 发布时被停止。

第二个版本于2005年中发布，是一个线程讨论论坛，旨在与电影数据库更深度地集成。该定制论坛软件自动检测文本中的电影标题，并在侧边栏中为每个被引用的电影旁边放置预测/评分小部件。这些小部件链接到每部电影的详情页面，电影详情页面则链接到引用该电影（或类似电影）的最新五篇帖子 [Drenner et al. 2006]。

第三个版本于2009年发布，将论坛界面替换为支持提问的界面。这一重新设计降低了用户对该功能的兴趣，因为提问-回答的格式与社区的期望用途不太自然契合。

v3 讨论论坛承载了中等规模的活动，大部分内容来自一小群用户。在4.5年的时间里，约900名用户每天约发帖5.5次，创建了约10,000篇帖子和11,000个电影界面与讨论论坛之间的链接。以提问为导向的设计使用较少；在5年的时间里，约900名用户每天约发帖1.5次，创建了约2,000篇帖子和1,500个电影界面与提问区域之间的链接。

几项实验——在讨论论坛的生命周期中影响了数千名用户——创造了短期功能，以社交方式吸引用户。第一个实验从2005年初运行到2006年初，在主页上添加了个性化消息，邀请用户参与讨论论坛 [Harper et al. 2007a]。随后，从2006年末到2007年末，进行了一项大规模干预，增加了额外的社交功能，包括个人资料页面、群组资料页面和主页推荐。该设计是一项大规模现场研究的结果，研究在线社区中建立人际友谊或群体认同的不同影响 [Ren et al. 2012; Harper et al. 2007b]。

### 2.3 经验教训

在我们研究实验室运营 MovieLens 超过17年的时间里，我们学到了关于运营一个"真实"系统以支持学术研究的成本和收益的若干经验。

运行实时研究系统的主要好处是显而易见的：我们能够进行在线现场实验（例如，Ekstrand et al. [2015]、Ren et al. [2012]、Harper et al. [2007a]、Sen et al. [2006]、Drenner et al. [2006]、Cosley et al. [2005] 和 Rashid et al. [2002]），从而获得生态有效性和大量实验对象，这在实验室环境中很难实现。由于我们控制系统的运行，我们在设计实验方面拥有完全的灵活性，从用户界面的修改到后台算法。这种灵活性使得收集用户判断和意见变得容易，这通常是判断实验性更改是否成功的关键因素。更棒的是，我们发现 MovieLens 用户对系统核心功能之外的广泛功能实验持开放态度，这使得我们团队能够在标签、在线社区和用户动机方面开展工作。由于这种开放性，我们还可以通过电子邮件邀请 MovieLens 用户参与一次性用户研究 [Shani and Gunawardana 2011]，以测试那些尚未准备好大规模部署或集成到主用户界面中的功能。

运行实时系统的主要成本是研究人员在非研究任务上所需的时间，如软件开发、硬件维护、用户沟通、数据库维护和市场营销。有许多隐藏任务消耗研究人员的时间，例如在电源故障后重启服务器、当用户最喜欢的功能被更改时回复愤怒用户的电子邮件，或维护用户数据的安全备份。然而，其中一些任务为研究人员带来了间接好处：MovieLens 开发人员获得了大规模团队软件开发的体验。此外，与用户社区互动和管理往往能揭示新的研究思路。

在运营 MovieLens 的过程中，我们学到了一些可能对其他希望建立活跃研究平台的研究机构具有普适性的经验教训。

— 技术创业社区的经验同样适用。建立一个受用户欢迎的系统是困难的，因为世界上充满了争夺用户时间的高质量免费服务。我们的研究实验室还推出了其他几个系统，从 Cyclopath（其成功足以运行几个小规模实验，例如 Priedhorsky et al. [2010]）到 GopherAnswers（未能吸引足够的用户来推进我们的研究目标）。因此，我们认识到，科技创业公司世界中的许多相同理念（例如，Ries [2011]）在这个背景下同样适用：快速发布，快速失败——从研究角度来看，花在搭建系统上的大部分时间都不是富有成效的。

— 运行实时网站意味着持续的工作，而非一次性努力。用户期望随着技术规范的发展而快速提升，这给研究网站带来了压力，要求其跟上步伐，否则就会失去用户。例如，搜索引擎公司的创新改变了搜索框的行为规范；作为回应，MovieLens 自上线以来对其搜索实现进行了重大更改。虽然跟上这些技术变化对于保持健康的用户基础可能很重要，但在吸引研究生或其他研究人员方面同样重要，他们可能对使用多年前的代码不感兴趣。

— 通过代码复用和社交编码规范鼓励良好的实验代码。任何给定实验的代码都可能长期损害系统，例如，通过引入错误或降低代码质量。对该问题的自然回应是在代码库中构建一个研究框架来容纳和管理实验。然而，我们发现实验形式多样，很难在任何框架中捕获所有需求。我们通过两种努力取得了更好的成功。首先，我们提供工具使常见的研究任务——例如日志记录和条件分配——更容易、更一致。其次，我们建立了社交编码规范，包括代码审查，以确保新实验遵循既定规范并最小化对核心代码的影响。

— 投资于允许用户社区提供帮助的工具。在许多情况下，可以构建允许用户无需管理员干预即可执行操作的功能。例如，虽然我们实验室的研究人员对维护电影数据库兴趣不大，但我们的用户对此充满热情。

## 3. MovieLens 数据集

目前已有四个 MovieLens 数据集发布，分别称为 100k、1m、10m 和 20m，反映了每个数据集中评分的近似数量。这些数据集于1998年首次发布，此后每5到6年发布一个主要版本，其规模随 MovieLens 系统一起增长。随着 20m 数据集的发布，GroupLens 开始托管额外的非存档数据集（一个完整版用于完整性，一个小型版用于速度），这些数据集定期刷新以包含最新电影：latest 和 latest-small。

MovieLens 数据集的汇总统计信息见表 II。

所有当前发布的 MovieLens 数据集共享几个特征。每个数据集都表示形式为 `<user, item, rating, timestamp>` 的评分元组。评分和其他数据归属于匿名用户 ID（这些 ID 不跨数据集映射）。电影以其在 MovieLens 中的标题以及零个或多个类型列出（电影 ID 跨数据集映射）。仅包含至少20个评分的用户。

这四种数据集的采样方法不同。尽管所有四个数据集都要求用户至少要有20个评分，但 100k 还需要完整的用户人口统计数据。1M 采样了2000年加入系统的用户，但样本收集于2003年初，导致2001年和2002年的评分稀疏。10M 和 20M 在整个系统历史中随机选择用户。这些收集方法创建的数据集反映了 MovieLens 中发生的评分活动爆发。图10展示了四个数据集中评分密度随时间的变化。

几个显著的变化影响了不同版本数据集的结构。100k 和 1M 数据集包含用户的人口统计数据（年龄、性别、职业、邮政编码），而 10M 和 20M 数据集不包含任何人口统计信息（网站已停止收集）。只有 10M 和 20M 数据集包含标签应用（较早的数据集在标签功能存在之前就已发布）。20M 包含一个表格，将 MovieLens 电影 ID 映射到两个外部站点的电影 ID，以允许数据集用户构建更完整的基于内容的项目表示。

如前所述，这些数据集是真实用户与 MovieLens（一个持续演变的系统）交互的结果。一些设计变更，例如改为半星评分，对数据集内容有直接可测量的影响。其他变更，例如在注册过程中增加至少15个评分的要求，可能产生更微妙的影响。诸如此类的变化可能导致不同的用户达到20个评分的最低门槛，或导致他们评分不同类型的电影。这些细微的差异是从一个长期运行且不断变化的系统生成数据集的必然产物。

**表 II. MovieLens 评分数据集定量总结**

| 名称 | 日期范围 | 评分量表 | 用户数 | 电影数 | 评分数量 | 标签应用数 | 密度 |
|------|----------|----------|--------|--------|----------|------------|------|
| ML100K | 1997/9–1998/4 | 1–5, 星 | 943 | 1,682 | 100,000 | 0 | 6.30% |
| ML1M | 2000/4–2003/2 | 1–5, 星 | 6,040 | 3,706 | 1,000,209 | 0 | 4.47% |
| ML10M | 1995/1–2009/1 | 0.5–5, 半星^a | 69,878 | 10,681 | 10,000,054 | 95,580 | 1.34% |
| ML20M | 1995/1–2015/3 | 0.5–5, 半星^a | 138,493 | 27,278 | 20,000,263 | 465,564 | 0.54% |

注：唯一计算的列"密度"表示完整用户-项目矩阵中包含评分值的单元格百分比。
^a MovieLens 于2003年2月18日从1至5星量表改为0.5至5.0半星量表。

**图10. 四个 MovieLens 数据集中评分数量随时间的变化，按月聚合。**

### 3.1 影响

正如研究人员和研究资助者越来越认识到的那样，共享数据集可以对研究、教育和实践产生实质性的积极影响。通过发布覆盖单一演变系统超过17年使用情况的数据集，MovieLens 数据集在推荐系统领域以及相关领域做出了独特的贡献。

#### 3.1.1 对研究的贡献

— 2015年春季，在 Google Books 中搜索"movielens"得到2,750条结果，在 Google Scholar 中得到7,580条结果。
— 我们特别自豪的是，MovieLens 数据集被用于开发和测试推荐系统中的许多核心算法进展，包括项目-项目协同过滤 [Sarwar et al. 2001; Karypis 2001; Deshpande and Karypis 2004]、降维协同过滤 [Sarwar et al. 2000]、基于信任的推荐器 [O'Donovan and Smyth 2005; Massa and Avesani 2007]、快速变化项目集上的推荐器 [Das et al. 2007] 以及冷启动算法 [Schein et al. 2002]。
— 这些数据集至今仍被积极使用（Google Scholar 中1,112条搜索结果日期为2014年），我们希望最近发布的20M评分数据集将帮助许多其他研究人员。

#### 3.1.2 对教育和实践的贡献

— 2014年，这些数据集从 GroupLens 网站（http://grouplens.org/datasets/movielens）被下载超过140,000次。虽然其中一些下载反映了研究用途，但我们认为绝大多数是用于教育目的。我们收到教授们在课程中使用数据集的反馈（有时寻求我们的许可，为学生制作定制格式的版本），也收到学生在作业或自学中使用数据集的反馈，还有教科书作者和其他人在他们的材料中使用这些数据集的反馈。从一开始，我们就使 MovieLens 数据集对任何非商业用途免费（包括商业实体的内部研究和教育）。
— 这些数据集在流行编程和数据科学书籍中作为示例代码的测试数据找到了自己的定位。例如，这些数据集被用于演示在 Python 中构建项目-项目协同过滤算法 [Segaran 2007]，以及演示在高性能分布式计算平台中加载、探索和可视化数据 [Pentreath 2015]。
— 我们通过推荐系统的大规模在线开放课程（MOOC）[Konstan et al. 2014] 以及通过自动化构建工具与 LensKit 推荐系统工具包（http://lenskit.org）[Ekstrand et al. 2011] 的集成，自行传播了 MovieLens 数据集。
— 我们还知道商业实体的相当广泛的使用。有公司联系我们，使用 MovieLens 数据集测试他们自己的推荐算法和系统，进行外部系统的基准测试，以及用于培训和演示目的。

### 3.2 局限性

MovieLens 用户体验的广泛变化不可避免地导致了比保持界面不变时更"不干净"的评分数据。如前所述，搜索工具、推荐算法和可用功能的变化都会改变向用户显示的内容，进而影响最终的评分数据。此外，我们在 MovieLens 平台上进行的广泛实验进一步以多种方式改变了用户的评分行为。然而，MovieLens 在这个问题上并不独特——像 Amazon 和 Netflix 这样的推荐网站也引入了重大的界面变化和通过 A/B 测试进行的扰动。

MovieLens 数据集仅包含至少20个评分的用户数据，因此天生偏向于"成功"用户。也就是说，那些对电影评分不太感兴趣、无法找到足够可评分内容、或者对系统初始体验不满意的用户没有被包含在数据集中。这些用户可能与数据集中的用户有着根本性的不同。

数据集将时间戳与每个评分关联，但这些时间戳不代表消费日期。MovieLens 中的评分可以随时发生，可能是在观看电影多年之后。通常，用户会在单次会话中输入大量评分，为了个人满足感或希望获得更个性化的推荐而回溯填写评分历史。如前所述，这些回溯填写的评分通常由我们显示的推荐所引发，进一步降低了时间戳作为用户何时考虑该电影的估计值。

### 3.3 使用建议

在本节中，我们提供一些基于经验的指南，以便更有效地使用 MovieLens 数据集，推进我们使其对研究人员、学生和实践者普遍有价值的目标。

对于那些进行算法研究的人来说，这些数据集的大部分价值在于能够与先前的研究论文或其他算法比较结果。我们强烈建议在比较已发表结果时，使用与已发表研究相同的数据集。或者，在许多情况下，可以使用诸如 LensKit 之类的工具来复现先前的结果，并将它们——以及你的新算法——应用于新数据集。

对于那些不与历史已发表结果进行比较的人，我们强烈建议使用 20M 评分数据集，因为它有更多的标签和到外部元数据源的链接。基于该数据集发布结果将使其他人更容易将你的结果与那些更多利用元数据和基于内容技术的方法的结果进行比较。

那些对将评分或标签数据与丰富的电影内容数据结合感兴趣的人，应考虑将 20M 数据集与外部资源结合使用。20M 数据集包含一个 MovieLens ID 到 IMDb⁴（"互联网电影数据库"）ID 的映射。虽然 IMDb 不提供 API 或鼓励访问其数据，但其 ID 被许多不同的电影网站所识别，包括几个通过 API 访问有趣元数据源的网站。由于 MovieLens 电影 ID 跨数据集是稳定的，20M 的链接文件可以与任何早期发布的数据集结合使用。

教育工作者应考虑使用 MovieLens latest-small 数据集。其磁盘大小使其下载、解析和处理速度很快。它包含最近的电影，学生可能会喜欢（相比于传统的 100k 数据集，该数据集不包含1998年之后的电影）。该数据集还以带有标题行的 .csv 文件格式提供，这与现代数据集惯例一致。

⁴http://imdb.com。

### 3.4 替代数据集

还有几个其他包含显式评分的数据集已被频繁用于（与 MovieLens 数据集一起使用或替代）研究目的。这些替代方案在规模和形态、领域以及用户交互的背景方面与 MovieLens 不同。当这些数据集更好地匹配研究人员希望模拟的在线系统 [Shani and Gunawardana 2011]，或者当研究人员希望跨多个数据集评估某种方法时，可以考虑使用替代数据集。可供选择的数据集太多，无法一一列举；在本节中，我们讨论几个最常被引用或当前突出的替代方案。概述见表 III，按评分数量排序。

**表 III. 包含显式评分数据的主要替代数据集定量总结**

| 名称 | 日期范围 | 领域 | 评分量表 | 评分数量 | 密度 |
|------|----------|------|----------|----------|------|
| Book-Crossing | 2001–2004 | 书籍 | 0–10, 11个离散值 | 110万 | 0.003% |
| EachMovie^a | 1995–1997 | 电影 | 0–14, 26个离散值^b | 270万 | 2.872% |
| Jester（数据集1） | 1999–2003 | 笑话 | −10–10, 连续值 | 410万 | 57.463% |
| Amazon | 1996–2014 | 多种 | 1–5, 5个离散值 | 8,280万 | <0.001% |
| Netflix Prize^a | 1998–2005 | 电影 | 1–5, 5个离散值 | 1.005亿 | 1.178% |
| Yahoo Music（C15） | 1999–2009 | 音乐^c | 0–100, 101个离散值^d | 2.628亿 | 0.042% |

注：唯一计算的列"密度"表示完整用户-项目矩阵中包含评分值的单元格百分比。
^a 不再可用。
^b 使用多个评分量表并重新缩放为最终数据集的结果。
^c 包含歌曲、专辑、艺人和类型的评分。
^d 98%的评分值是10的倍数。

**Book-Crossing。** Book-Crossing 数据集⁵ [Ziegler et al. 2005] 是一个在线图书评分社区的快照。该数据集包含100,000个用户、340,000本书和110万个评分，收集于2004年。其密度较低（3.2×10⁻⁵）。评分采用11个可能值，范围为0到10。快照于2004年拍摄，但评分不与时间戳关联。然而，该数据集揭示了用户的位置和年龄。

**EachMovie。** EachMovie 数据集在2000年代初被广泛用于研究，之后因 DEC 担心数据集中用户可能被重新识别的法律问题而停止提供。该数据集包含来自59,000个用户对1,500部电影的270万个评分，收集于1995年至1997年。评分采用26个可能值，范围为0到14，反映了使用和后续重新缩放多种评分收集方法的结果。如前所述，该数据集是 MovieLens 推荐器的原始种子数据，但这些评分数据未包含在任何 MovieLens 数据集中。

**Jester。** Jester 数据集⁶ [Goldberg et al. 2001] 包含对笑话的连续显式评分，采用−10到10的量表——用户使用滑块小部件对笑话进行评分。目前有多个版本可用；最大的版本（"数据集1"）包含来自73,421个用户的410万个评分，收集于1999年至2003年。由于只有100个笑话，该数据集相对于其他选项非常密集。该数据集不包含评分时间戳。

**Amazon。** Amazon 评论数据集⁷ [McAuley et al. 2015a, 2015b] 因以下几个原因而值得关注。评分与文本评论相关联，跨越18年，涵盖从即时视频到婴儿服装的广泛产品。完整（"积极去重"）数据集也非常大且稀疏：来自2,100万个用户对近1,000万个项目的8,200万个评分（密度3.9×10⁻⁷）。评分采用5个可能值（1到5），并与时间戳关联。

**Netflix。** Netflix Prize 数据集于2006年作为 Netflix Prize 的一部分发布，旨在提高预测准确性⁸。该数据集因法律原因于2009年下线。训练数据集包含480,000个用户、17,000个项目和1亿个评分；其密度与 MovieLens 10M 数据集相当。评分采用5个可能值（1到5），并与时间戳关联。

**Yahoo Music。** Yahoo! Labs 从其音乐产品中提供了多个音乐数据集⁹。他们的 "C15" 数据集为 KDD Cup 发布 [Dror et al. 2012]，提供了用户对歌曲、专辑、艺人和音乐类型的评分，采用101点（0–100）评分量表（尽管98%的评分是10的倍数）。该数据集包含100万个用户、620,000个音乐项目（前述项目类型混合在一起）和2.62亿个评分。该数据集包含部分时间戳，这些时间戳公开了时间但模糊了绝对日期。

⁵http://www2.informatik.uni-freiburg.de/~cziegler/BX/
⁶http://eigentaste.berkeley.edu/dataset/
⁷http://jmcauley.ucsd.edu/data/amazon/
⁸http://www.netflixprize.com/community/viewtopic.php?id=68
⁹http://webscope.sandbox.yahoo.com/catalog.php?datatype=c 和 http://webscope.sandbox.yahoo.com/catalog.php?datatype=r

## 4. 结论

在本文中，我们提供了 MovieLens 系统和数据集的历史视角，这些系统和数据集对教育、研究和行业产生了实质性影响。这些数据集是 MovieLens 系统的产物，而该系统自首次发布以来的17年中经历了重大变化。我们讨论了系统的关键功能对用户的影响、通过现场研究对研究文献的影响，以及它们对数据集的后续影响。

我们最近发布了新版本的 MovieLens（v4）和新的基准数据集（20M）。我们希望这些版本能够继续鼓励高质量教育材料、软件系统、创业公司和学术研究的发展。

## 致谢

许多人致力于构建和改进 MovieLens 及 MovieLens 数据集。我们特别感谢 John Riedl 的重要贡献和领导。其他关键贡献者包括 Istvan Albert、Al Borchers、Dan Cosley、Brent J. Dahlen、Rich Davies、Michael Ekstrand、Dan Frankowski、Nathaniel Good、Jon Herlocker、Daniel Kluver、Shyong (Tony) Lam、Michael Ludwig、Sean McNee、Chad Salvatore、Shilad Sen 和 Loren Terveen。我们还衷心感谢 MovieLens 会员，是他们使这个项目成为可能。

## 参考文献

Shuo Chang, F. Maxwell Harper, and Loren Terveen. 2015. Using groups of items for preference elicitation in recommender systems. In *Proceedings of the 18th ACM Conference on Computer Supported Cooperative Work & Social Computing (CSCW'15)*. ACM, New York, NY, 1258–1269. DOI:http://dx.doi.org/10.1145/2675133.2675210

Dan Cosley, Dan Frankowski, Sara Kiesler, Loren Terveen, and John Riedl. 2005. How oversight improves member-maintained communities. In *Proceedings of the SIGCHI Conference on Human Factors in Computing Systems (CHI'05)*. ACM, New York, NY, 11–20. DOI:http://dx.doi.org/10.1145/1054972.1054975

Dan Cosley, Shyong K. Lam, Istvan Albert, Joseph A. Konstan, and John Riedl. 2003. Is seeing believing?: How recommender system interfaces affect users' opinions. In *Proceedings of the SIGCHI Conference on Human Factors in Computing Systems (CHI'03)*. ACM, New York, NY, 585–592. DOI:http://dx.doi.org/10.1145/642611.642713

Abhinandan S. Das, Mayur Datar, Ashutosh Garg, and Shyam Rajaram. 2007. Google news personalization: scalable online collaborative filtering. In *Proceedings of the 16th International Conference on World Wide Web (WWW'07)*. ACM, New York, NY, 271–280. DOI:http://dx.doi.org/10.1145/1242572.1242610

Mukund Deshpande and George Karypis. 2004. Item-based top-N recommendation algorithms. *ACM Transactions on Information Systems* 22, 1, 143–177. DOI:http://dx.doi.org/10.1145/963770.963776

Sara Drenner, Max Harper, Dan Frankowski, John Riedl, and Loren Terveen. 2006. Insert movie reference here: A system to bridge conversation and item-oriented web sites. In *Proceedings of the SIGCHI Conference on Human Factors in Computing Systems (CHI'06)*. ACM, New York, NY, 951–954. DOI:http://dx.doi.org/10.1145/1124772.1124914

Gideon Dror, Yahoo Labs, Noam Koenigstein, Yehuda Koren, and Markus Weimer. 2012. The Yahoo! music dataset and KDD Cup 11. In *Journal of Machine Learning Research Workshop and Conference Proceedings: Proceedings of KDD Cup 2011*. 3–18.

Michael D. Ekstrand, Daniel Kluver, F. Maxwell Harper, and Joseph A. Konstan. 2015. Letting users choose recommender algorithms: An experimental study. In *Proceedings of the 9th ACM Conference on Recommender Systems (RecSys'15)*. ACM, New York, NY, 11–18. DOI:http://dx.doi.org/10.1145/2792838.2800195

Michael D. Ekstrand, Michael Ludwig, Joseph A. Konstan, and John T. Riedl. 2011. Rethinking the recommender research ecosystem: Reproducibility, openness, and lenskit. In *Proceedings of the 5th ACM Conference on Recommender Systems (RecSys'11)*. ACM, New York, NY, 133–140. DOI:http://dx.doi.org/10.1145/2043932.2043958

Malcolm Gladwell. 1999. The science of the sleeper. *The New Yorker*. Retrieved November 13, 2015 from http://gladwell.com/the-science-of-the-sleeper/.

Ken Goldberg, Theresa Roeder, Dhruv Gupta, and Chris Perkins. 2001. Eigentaste: A constant time collaborative filtering algorithm. *Information Retrieval* 4, 2, 133–151. DOI:http://dx.doi.org/10.1023/A:1011419012209

F. Maxwell Harper, Dan Frankowski, Sara Drenner, Yuqing Ren, Sara Kiesler, Loren Terveen, Robert Kraut, and John Riedl. 2007a. Talk amongst yourselves: Inviting users to participate in online conversations. In *Proceedings of the 12th International Conference on Intelligent User Interfaces (IUI'07)*. ACM, New York, NY, 62–71. DOI:http://dx.doi.org/10.1145/1216295.1216313

F. Maxwell Harper, Shilad Sen, and Dan Frankowski. 2007b. Supporting social recommendations with activity-balanced clustering. In *Proceedings of the 2007 ACM Conference on Recommender Systems (RecSys'07)*. ACM, New York, NY, 165–168. DOI:http://dx.doi.org/10.1145/1297231.1297262

F. Maxwell Harper, Funing Xu, Harmanpreet Kaur, Kyle Condiff, Shuo Chang, and Loren Terveen. 2015. Putting users in control of their recommendations. In *Proceedings of the 9th ACM Conference on Recommender Systems (RecSys'15)*. ACM, New York, NY, 3–10. DOI:http://dx.doi.org/10.1145/2792838.2800179

George Karypis. 2001. Evaluation of item-based top-N recommendation algorithms. In *Proceedings of the 10th International Conference on Information and Knowledge Management (CIKM'01)*. ACM, New York, NY, 247–254. DOI:http://dx.doi.org/10.1145/502585.502627

Joseph A. Konstan, Bradley N. Miller, David Maltz, Jonathan L. Herlocker, Lee R. Gordon, and John Riedl. 1997. GroupLens: Applying collaborative filtering to Usenet news. *Communications of the ACM* 40, 3, 77–87. DOI:http://dx.doi.org/10.1145/245108.245126

Joseph A. Konstan, J. D. Walker, D. Christopher Brooks, Keith Brown, and Michael D. Ekstrand. 2014. Teaching recommender systems at large scale: Evaluation and lessons learned from a hybrid MOOC. In *Proceedings of the 1st ACM Conference on Learning@Scale Conference (L@S'14)*. ACM, New York, NY, 61–70. DOI:http://dx.doi.org/10.1145/2556325.2566244

John G. Lynch, Jr., Dipankar Chakravarti, and Anusree Mitra. 1991. Contrast effects in consumer judgments: Changes in mental representations or in the anchoring of rating scales? *Journal of Consumer Research* 18, 3, 284–297.

Paolo Massa and Paolo Avesani. 2007. Trust-aware recommender systems. In *Proceedings of the 2007 ACM Conference on Recommender Systems (RecSys'07)*. ACM, New York, NY, 17–24. DOI:http://dx.doi.org/10.1145/1297231.1297235

Julian McAuley, Rahul Pandey, and Jure Leskovec. 2015a. Inferring networks of substitutable and complementary products. In *Proceedings of the 21st ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD'15)*. ACM, New York, NY, 785–794. DOI:http://dx.doi.org/10.1145/2783258.2783381

Julian McAuley, Christopher Targett, Qinfeng Shi, and Anton van den Hengel. 2015b. Image-based recommendations on styles and substitutes. In *Proceedings of the 38th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR'15)*. ACM, New York, NY, 43–52. DOI:http://dx.doi.org/10.1145/2766462.2767755

Bradley Norman Miller. 2003. *Toward a Personal Recommender System*. Ph.D. dissertation. University of Minnesota, Minneapolis, MN.

Mark O'Connor, Dan Cosley, Joseph A. Konstan, and John Riedl. 2001. PolyLens: A recommender system for groups of users. In *Proceedings of the 7th Conference on European Conference on Computer Supported Cooperative Work (ECSCW'01)*. Kluwer Academic Publishers, Norwell, MA, 199–218.

John O'Donovan and Barry Smyth. 2005. Trust in recommender systems. In *Proceedings of the 10th International Conference on Intelligent User Interfaces (IUI'05)*. ACM, New York, NY, 167–174. DOI:http://dx.doi.org/10.1145/1040830.1040870

Nick Pentreath. 2015. *Machine Learning with Spark*. Packt Publishing Ltd, Birmingham, UK.

Reid Priedhorsky, Mikhil Masli, and Loren Terveen. 2010. Eliciting and focusing geographic volunteer work. In *Proceedings of the 2010 ACM Conference on Computer Supported Cooperative Work (CSCW'10)*. ACM, New York, NY, 61–70. DOI:http://dx.doi.org/10.1145/1718918.1718931

Al Mamunur Rashid, Istvan Albert, Dan Cosley, Shyong K. Lam, Sean M. McNee, Joseph A. Konstan, and John Riedl. 2002. Getting to know you: Learning new user preferences in recommender systems. In *Proceedings of the 7th International Conference on Intelligent User Interfaces (IUI'02)*. ACM, New York, NY, 127–134. DOI:http://dx.doi.org/10.1145/502716.502737

Al Mamunur Rashid, George Karypis, and John Riedl. 2008. Learning preferences of new users in recommender systems: An information theoretic approach. *ACM SIGKDD Explorations Newsletter* 10, 2, 90–100. DOI:http://dx.doi.org/10.1145/1540276.1540302

Yuqing Ren, F. Harper, Sara Drenner, Loren Terveen, Sara Kiesler, John Riedl, and Robert Kraut. 2012. Building member attachment in online communities: Applying theories of group identity and interpersonal bonds. *Management Information Systems Quarterly* 36, 3 (Sept. 2012), 841–864.

Paul Resnick, Neophytos Iacovou, Mitesh Suchak, Peter Bergstrom, and John Riedl. 1994. GroupLens: An open architecture for collaborative filtering of Netnews. In *Proceedings of the 1994 ACM Conference on Computer Supported Cooperative Work (CSCW'94)*. ACM, New York, NY, 175–186. DOI:http://dx.doi.org/10.1145/192844.192905

Eric Ries. 2011. *The Lean Startup: How Today's Entrepreneurs Use Continuous Innovation to Create Radically Successful Businesses*. Crown Business, New York, NY.

Badrul Sarwar, George Karypis, Joseph Konstan, and John Riedl. 2000. *Application of Dimensionality Reduction in Recommender System—A Case Study*. Technical Report. DTIC Document.

Badrul Sarwar, George Karypis, Joseph Konstan, and John Riedl. 2001. Item-based collaborative filtering recommendation algorithms. In *Proceedings of the 10th International Conference on World Wide Web (WWW'01)*. ACM, New York, NY, 285–295. DOI:http://dx.doi.org/10.1145/371920.372071

Andrew I. Schein, Alexandrin Popescul, Lyle H. Ungar, and David M. Pennock. 2002. Methods and metrics for cold-start recommendations. In *Proceedings of the 25th Annual International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR'02)*. ACM, New York, NY, 253–260. DOI:http://dx.doi.org/10.1145/564376.564421

Toby Segaran. 2007. *Programming Collective Intelligence: Building Smart Web 2.0 Applications*. O'Reilly Media, Inc., Sebastopol, CA.

Shilad Sen, F. Maxwell Harper, Adam LaPitz, and John Riedl. 2007. The quest for quality tags. In *Proceedings of the 2007 International ACM Conference on Supporting Group Work (GROUP'07)*. ACM, New York, NY, 361–370. DOI:http://dx.doi.org/10.1145/1316624.1316678

Shilad Sen, Shyong K. Lam, Al Mamunur Rashid, Dan Cosley, Dan Frankowski, Jeremy Osterhouse, F. Maxwell Harper, and John Riedl. 2006. Tagging, communities, vocabulary, evolution. In *Proceedings of the 2006 20th Anniversary Conference on Computer Supported Cooperative Work (CSCW'06)*. ACM, New York, NY, 181–190. DOI:http://dx.doi.org/10.1145/1180875.1180904

Shilad Sen, Jesse Vig, and John Riedl. 2009. Learning to recognize valuable tags. In *Proceedings of the 14th International Conference on Intelligent User Interfaces (IUI'09)*. ACM, New York, NY, 87–96. DOI:http://dx.doi.org/10.1145/1502650.1502666

Guy Shani and Asela Gunawardana. 2011. Evaluating recommendation systems. In *Recommender Systems Handbook*, Francesco Ricci, Lior Rokach, Bracha Shapira, and Paul B. Kantor (Eds.). Springer US, New York, NY, 257–297.

Jesse Vig, Shilad Sen, and John Riedl. 2012. The tag genome: Encoding community knowledge to support novel interaction. *ACM Transactions on Interactive Intelligent Systems* 2, 3, 13:1–13:44. DOI:http://dx.doi.org/10.1145/2362394.2362395

Jesse Vig, Matthew Soukup, Shilad Sen, and John Riedl. 2010. Tag expression: Tagging with feeling. In *Proceedings of the 23rd Annual ACM Symposium on User Interface Software and Technology (UIST'10)*. ACM, New York, NY, 323–332. DOI:http://dx.doi.org/10.1145/1866029.1866079

Cai-Nicolas Ziegler, Sean M. McNee, Joseph A. Konstan, and Georg Lausen. 2005. Improving recommendation lists through topic diversification. In *Proceedings of the 14th International Conference on World Wide Web (WWW'05)*. ACM, New York, NY, 22–32. DOI:http://dx.doi.org/10.1145/1060745.1060754

---

收稿日期：2015年7月；修订日期：2015年10月；录用日期：2015年10月

ACM 交互式智能系统交易，第5卷，第4期，文章19，出版日期：2015年12月。
