# EdgeRec: Recommender System on Edge in Mobile Taobao（中文翻译）

> 原文标题：EdgeRec: Recommender System on Edge in Mobile Taobao  
> 作者：Yu Gong\*¹, Ziwen Jiang\*¹, Yufei Feng¹, Binbin Hu², Kaiqi Zhao³, Qingwen Liu¹, Wenwu Ou¹  
> 机构：1阿里巴巴集团, 2蚂蚁金融服务集团, 3奥克兰大学  
> 会议：CIKM '20, October 19–23, 2020, Virtual Event, Ireland  
> DOI: https://doi.org/10.1145/3340531.3412700

---

## 摘要（ABSTRACT）

推荐系统（Recommender System, RS）已成为大多数网络规模应用中的关键模块。近来，大多数推荐系统基于云到端（cloud-to-edge）框架采用瀑布流形式，其中推荐结果通过在云服务器中预先计算后被传输到端侧（例如用户移动端）。尽管有效，但云服务器与端侧之间的网络带宽和延迟可能导致系统反馈和用户感知的延迟。因此，端侧上的实时计算能够帮助更精准地捕获用户偏好，从而做出更令人满意的推荐。据我们所知，我们的工作是首次尝试设计并实现新型的端侧推荐系统（EdgeRec），该系统实现了实时用户感知（Real-time User Perception）和实时系统反馈（Real-time System Feedback）。此外，我们提出了异构用户行为序列建模（Heterogeneous User Behavior Sequence Modeling）和基于行为注意力网络的上下文感知重排序（Context-aware Reranking with Behavior Attention Networks）来捕获用户的多样兴趣并相应地调整推荐结果。在淘宝首页信息流（Taobao home-page feeds）上的离线评估和在线性能实验结果均证明了EdgeRec的有效性。

## 关键词（KEYWORDS）

推荐系统；边缘计算

## ACM引用格式（ACM Reference Format）

Yu Gong, Ziwen Jiang, Yufei Feng, Binbin Hu, Kaiqi Zhao, Qingwen Liu, Wenwu Ou. 2020. EdgeRec: Recommender System on Edge in Mobile Taobao. 见于第29届ACM国际信息与知识管理大会（CIKM '20）论文集，2020年10月19–23日，线上会议，爱尔兰。ACM, 纽约, NY, USA, 8页. https://doi.org/10.1145/3340531.3412700

---

## 1 引言（INTRODUCTION）

网络上可用信息（例如电影、商品、新闻等）的爆炸性增长和多样性常常使用户不堪重负。推荐系统（RS）作为一种应对信息过载问题的有价值手段，从海量候选项中挑选出一系列物品来满足用户的多样化需求。在商业推荐系统的大多数场景中，尤其是在移动端，推荐物品以瀑布流形式展示。

\* 两位作者贡献相同。

如图1所示，大多数瀑布流推荐系统基于云到端框架部署。当用户在瀑布流推荐系统场景中滚动时，移动端客户端首先向云服务器发起分页请求。然后，服务于云服务器的匹配和排序模型响应分页请求，生成一个排序后的物品列表展示给用户。在这种情形下，当前基于云到端的瀑布流推荐系统存在以下局限：

• **系统反馈延迟（Delay for System Feedback）：** 由于云到端框架中的分页机制，云端的推荐系统无法在两个相邻分页请求之间及时调整推荐结果，从而无法满足用户变化的需求。以图1中的一个例子说明，用户点击了当前页面第5个位置的一件连衣裙，这反映了他/她对连衣裙类别的突发偏好。然而，云端的推荐系统无法对此做出响应，除非用户滚动到下一页，这因此无法及时满足他/她的需求并降低了用户体验。

• **用户感知延迟（Delay for User Perception）：** 对于服务于云端的推荐系统模型，由于网络延迟，捕获用户行为存在长达1分钟的延迟，因此在响应端侧时无法建模用户的实时偏好。例如，在图1中，用户对第49个位置物品的行为揭示了他/她当前对收音机的偏好，但云端的推荐系统由于未及时接收到这些行为而无法在下一页推荐类似的收音机。此外，网络带宽进一步限制了当前推荐系统捕获端侧上多样且细致的用户行为。

总结来说，云端推荐系统的局限性在于，推荐结果的延迟调整无法匹配端侧上用户偏好的实时变化，从而严重损害了商业推荐系统的用户体验。

边缘计算（Edge computing）[17–20]非常适合需要高实时性能的应用，并有潜力解决当前基于云到端框架的推荐系统的上述问题。我们的工作首次设计和实现了新型的端侧推荐系统¹（EdgeRec），该系统在不增加对云服务器额外请求的情况下实现了实时用户感知和实时系统反馈。我们的主要贡献总结如下：

• **系统架构（System Architecture）** — 我们设计了EdgeRec的架构以在移动设备上执行重排序，与提供候选物品的云端推荐系统协同工作（见第2.1节）。

• **系统实现（System Implementation）** — EdgeRec支持大规模神经网络模型，通过将模型分布在端侧和云上来考虑移动设备上的高效计算和存储（见第2.2节）。

• **用户行为建模（User Behavior Modeling）** — 我们提出了异构用户行为序列建模来捕获用户变化的行为和动作。我们首先设计了新型的特征系统（见第3.2节），然后同时对用户与物品之间的正向和负向反馈进行建模，同时考虑交互物品及其对应的动作（见第3.3节）。基于EdgeRec，我们的特征系统中多样且细致的用户行为在端侧上被收集、存储和消费，可以实时输入到我们的模型中。

• **上下文感知重排序（Context-aware Reranking）** — 我们提出了基于行为注意力网络的上下文感知重排序用于端侧重排序。具体来说，我们通过提出的行为注意力机制对候选物品与实时用户行为上下文之间的交互进行建模（见第3.4节）。凭借基于EdgeRec在端侧上重排序物品的能力，我们实现了实时响应以满足用户需求。

我们在淘宝首页信息流的真实流量上进行了广泛的离线和在线评估。定量和定性分析都证明了我们提出的EdgeRec系统的合理性和有效性。此外，EdgeRec在在线A/B测试²中贡献了高达1.57%的PV、7.18%的CTR、8.87%的CLICK和10.92%的GMV提升，为当前淘宝推荐系统带来了显著改进。目前EdgeRec已上线并为主要流量提供服务。

¹这里用户的移动设备即为端侧。  
²PV和CLICK定义为用户浏览和点击的物品总数。CTR是点击率，计算公式为CLICK/PV。GMV是用户在推荐物品上花费的总金额（收入）。

---

## 2 系统（SYSTEM）

在本节中，我们介绍EdgeRec系统，旨在及时捕获丰富的用户行为（即实时感知）并及时响应用户需求（即实时反馈），无需对云服务器做任何额外请求。我们首先概述EdgeRec系统，然后详细阐述每个精心设计模块的实现。

### 2.1 系统概述（System Overview）

在图2中，我们展示了EdgeRec系统的概览。注意EdgeRec旨在与云端推荐系统协同工作而非替代它。主要模块和工作流程说明如下：

**客户端原生模块（Client Native, CN）** 首先发起分页请求并缓存来自推荐系统服务器的候选物品及其对应特征。在EdgeRec中，分页大小设置为50，与淘宝原始推荐系统保持一致以保持稳定性。同时，为给设备端重排序提供更多空间，来自推荐系统服务器的返回物品数量设置为100³。然后，CN收集用户在曝光物品上的行为并触发模型服务模块。在从模型服务模块接收到候选（即未曝光）物品的排序结果后，CN调整物品的UI展示。

**模型服务模块（Model Serving, MS）** 是EdgeRec系统的核心模块。当被CN触发时，MS首先对从CN接收到的用户行为和候选物品进行特征工程，然后执行基于神经网络的模型，目的是通过用户行为建模捕获及时的用户行为，并通过上下文感知重排序及时响应用户⁴。最后，MS将日志发送到云端并将候选物品的排序结果返回给CN。

**云端推荐系统（Recommender System, RS）** 可以视为EdgeRec中的召回模块，旨在响应来自CN的分页请求并提供初始排序的候选物品。此外，它可以在响应CN之前从云端的键值存储中查找MS模块中模型所需的物品特征和嵌入向量（例如类别嵌入）。

**离线训练模块（Offline Training, OT）** 首先从MS收集日志并在模型训练前构建样本。接下来，训练好的模型被拆分为三部分：（1）用户行为建模的子模型，（2）上下文感知重排序的子模型，以及（3）嵌入矩阵（例如类别和品牌）。最后，前两个子模型部署在MS模块上，而嵌入矩阵以键值形式保存在云端。

³配置根据具体的推荐系统环境经验性地设置。  
⁴我们使用MNN（https://github.com/alibaba/MNN）作为设备上的在线深度神经网络推理引擎。

### 2.2 系统实现（System Implementation）

在本节中，我们主要介绍EdgeRec系统中两个关键模块的实现细节：客户端原生模块和模型服务模块。

#### 2.2.1 客户端原生模块（Client Native）

客户端原生模块的一个关键部分是收集用户在移动端淘宝推荐系统中的丰富客户端行为，例如浏览和点击记录（更详细的行为见第3.2节）。这些用户行为随后存储在设备上的数据库中。由于EdgeRec模型（即模型服务）的运行由客户端原生模块触发，另一个关键部分是触发策略。这里我们根据用户的在线实时行为设置了几个触发点，总结如下：（1）用户点击一个物品，（2）用户删除一个物品（即长按），以及（3）k个物品已被曝光但未被点击。我们认为这三种类型的用户行为揭示了用户当前对推荐系统的偏好，推荐系统应及时响应他/她（即触发模型服务）。

#### 2.2.2 模型服务模块（Model Serving）

在移动设备上服务深度神经网络模型面临许多相对于传统云端服务的挑战，例如计算和存储的开销。针对EdgeRec模型服务，有两个关键实现分别面向计算和存储效率。其思想是将模型分布在端侧和云上，这使得EdgeRec支持在移动设备上为推荐系统服务大规模神经网络模型。

**计算效率（Computing Efficiency）。** 用户行为建模（见第3.3节）和上下文感知重排序（见第3.4节）共同训练但分开部署并在设备上异步运行。用户行为建模利用循环神经网络（RNN）[13]的序列建模方法，如果总是从头开始推断（即O(n)时间复杂度）则效率非常低下。因此，它利用RNN的循环特性（即O(1)时间复杂度）随用户的在线实时行为独立地进行实时推断，并生成行为编码存储在设备上的数据库中。上下文感知重排序将首先从数据库中检索行为编码，然后基于它们进行模型推断。

**存储效率（Storage Efficiency）。** ID类型的特征在推荐系统模型中常见且重要，我们总是利用嵌入（embedding）[10, 23]技术来转换它们。然而，在移动设备上服务时它们面临存储效率的挑战。例如，我们模型中的物品品牌是一个ID特征，字典大小约为150万，通过嵌入层将转换为40维的隐状态，其中嵌入矩阵大小将为1500000×40（即约230MB）。具有如此大嵌入矩阵的模型在部署到移动设备时会面临存储开销问题。在我们提出的系统中，我们将嵌入矩阵从训练好的模型中提取出来，部署到云端的键值数据库中。这些嵌入矩阵将在推荐系统服务器响应客户端原生模块的分页请求时被对应物品检索，并作为物品特征发送到客户端。部署在设备上的模型其余部分（不含嵌入层，约3MB）将接收嵌入特征作为输入，然后进行模型推断。

此外，我们设计了一个模型版本策略以确保模型更新时的同步，因为模型成功部署到设备可能远滞后于模型（即嵌入矩阵）部署到云端，这取决于用户移动设备的当前状态（例如连接WiFi还是3G）。在EdgeRec系统中，我们将为每个训练好的模型生成一个唯一的版本ID。该版本ID与部署在设备上的模型和存储在云端的嵌入矩阵一起保存。客户端原生模块首先携带设备上的模型版本ID发起分页请求，然后云端推荐系统获取模型版本ID并在响应客户端之前检索对应版本的嵌入矩阵。

---

## 3 算法（ALGORITHM）

在本节中，我们介绍用于用户行为建模和上下文感知重排序的特征系统和方法。

### 3.1 问题定义（Problem Definition）

我们提出的EdgeRec系统旨在针对瀑布流推荐场景在端侧上应用重排序方法。给定由云端现有推荐系统生成的缓存于端侧的初始排序物品列表Sr，对于客户端原生模块触发的模型服务模块中的重排序请求r ∈ R，我们的目标是找到一个评分函数ϕ(xi, s, C)，该函数考虑：（1）物品i的特征xi，（2）来自初始模型的局部排序上下文s，以及（3）当前推荐环境下的实时用户行为上下文C。

考虑局部排序上下文的重排序模型在先前的工作中已被充分研究。局部排序上下文被表示为初始排序候选物品之间的列表式交互，可以通过RNN [1, 26]或Transformer [15]建模。这里我们认为实时用户行为上下文对于重排序问题也很重要，尤其是在瀑布流推荐场景中，而之前很少有工作考虑这一点。第3.3节介绍了我们如何通过异构用户行为序列建模来建模实时用户行为上下文，第3.4节介绍了我们如何通过基于行为注意力网络的上下文感知重排序来建模候选物品与实时用户行为上下文之间的交互。通过结合边缘计算系统和上下文感知重排序模型，我们可以在推荐系统中实现实时感知和实时反馈，以更好地满足用户在线变化的需求。

### 3.2 特征系统（Feature System）

在本节中，我们首先讨论我们的特征系统，然后介绍物品曝光和物品浏览的详细用户动作特征以及对应的物品特征。

#### 3.2.1 洞察（Insight）

在个性化搜索和推荐系统的文献[7, 9, 12, 14, 16, 22–24]中，用户行为通常被建模以表征用户的个性化偏好。然而，这些模型只考虑用户与物品之间的直接"正向反馈"（例如点击或交易），很少关注间接的"负向反馈"（例如跳过或删除）。尽管"正向反馈"相对更清晰且噪声更少，但实时的"负向反馈"也非常重要，尤其是在瀑布流推荐系统中。以一个直观的在线的淘宝推荐系统中的例子说明，当一个物品类别被实时多次曝光后，其再次曝光的对应CTR将显著下降。

另一方面，先前的工作只考虑与用户交互的物品的特征（例如类别和品牌）。然而，用户对物品的"动作"也应被关注。例如，用户点击物品后，在其详情页（称为物品浏览）中的动作（例如收藏和加购）反映了用户对该物品的真实偏好。此外，即使用户没有点击物品，在该物品曝光上的动作（例如滚动速度和曝光时长）也可以表示该物品被视为"负向反馈"的程度。有时如果用户长时间关注一个物品曝光但没有点击，这绝对不能表明他/她不喜欢它。尤其是在当前瀑布流推荐系统中，物品展示越来越信息丰富，例如更大的图片、各种关键词甚至自动播放的视频，点击对于某些用户来说已经成为一种非常"奢侈"的正向反馈。

最后，基于我们提出的EdgeRec系统，所有用户行为特征都在端侧（即用户的移动设备）上收集、提取和消费，这有可能突破当前基于云到端框架的推荐系统的网络延迟和带宽限制。因此，可以以更加实时的方式融入丰富细致的用户行为来推断用户偏好。此外，用户的原始行为在其自己的移动设备上处理和利用，这可以在一定程度上缓解用户数据隐私问题。

总结来说，我们工作中的特征系统是新颖且推进的：（1）从"仅依赖正向反馈交互"到"同时关注正向和负向反馈交互"；（2）从"仅关注交互物品"到"同时考虑交互物品及其对应的动作"；（3）从"准实时方式"到"超实时方式"。

#### 3.2.2 物品曝光用户动作特征（Item Exposure User Action Feature）

物品曝光（Item Exposure, IE）用户动作揭示了用户在推荐系统当前展示页面中对物品曝光的行为方式。图3(a)展示了移动端淘宝瀑布流推荐系统中的物品曝光。其上对应的用户动作特征可分为（详见Tab. 1）：（1）物品曝光统计量（e1∼e2），（2）用户滚动统计量（e3∼e5），（3）用户删除反馈（e6），以及（4）时间衰减（e7）。这里我们将对应物品i的e1∼e7的拼接表示为物品曝光动作特征向量a_IE^i。

#### 3.2.3 物品浏览用户动作特征（Item Page-View User Action Feature）

物品浏览（Item Page-View, IPV）用户动作揭示了用户在点击物品后进入物品详情页中的行为方式。图3(b)展示了移动端淘宝中的物品浏览。其中对应的用户动作特征可分为（详见Tab. 1）：（1）物品浏览统计量（d1），（2）每个区块是否点击（d2∼d11），以及（3）时间衰减（d12）。这里我们将对应物品i的d1∼d12的拼接表示为物品浏览动作特征向量a_IPV^i。

#### 3.2.4 物品特征（Item Feature）

除了用户动作的特征之外，我们还需要对应物品的特征。它们可分为（详见Tab. 1）：（1）通过嵌入学习的离散特征（p1∼p6）和（2）从基础排序模型提供的原始特征（p7）。这里我们将物品i的p1∼p7的拼接表示为物品特征向量pi。

**表1：特征系统详情（e1∼e7：物品曝光用户动作特征；d1∼d12：物品浏览用户动作特征；p1∼p7：物品特征）**

| 变量 | 属性 | 描述 | 类型 |
|------|------|------|------|
| e1 | exposure_duration | 物品曝光总时长 | 分桶 |
| e2 | exposure_count | 物品曝光总次数 | 分桶 |
| e3 | scroll_speed | 物品曝光最大滚动速度 | 分桶 |
| e4 | scroll_duration | 物品曝光最大滚动时长 | 分桶 |
| e5 | scroll_count | 物品曝光总滚动次数 | 分桶 |
| e6 | delete_reason | 删除物品（长按）的原因或否 | 独热 |
| e7 | expose_decay | 从物品曝光到当前的时间衰减 | 分桶 |
| d1 | ipv_duration | 物品浏览总时长 | 分桶 |
| d2 | cart | 加购 | 二值 |
| d3 | buy | 立即购买 | 二值 |
| d4 | favorite | 收藏 | 二值 |
| d5 | comment | 进入评论页 | 二值 |
| d6 | select_SKU | 选择库存量单位（SKU） | 二值 |
| d7 | WDJ | 问大家 | 二值 |
| d8 | wangwang | 点击客服 | 二值 |
| d9 | detail | 进入物品详情页 | 二值 |
| d10 | shop | 进入店铺 | 二值 |
| d11 | recommendation | 进入推荐页 | 二值 |
| d12 | ipv_decay | 从物品浏览到当前的时间衰减 | 分桶 |
| p1 | category | 产品类别 | 嵌入 |
| p2 | brand | 产品品牌 | 嵌入 |
| p3 | gender | 产品适用性别 | 嵌入 |
| p4 | price_level | 产品价格等级 | 嵌入 |
| p5 | age_level | 产品年龄等级 | 嵌入 |
| p6 | bc_type | 产品bc类型 | 嵌入 |
| p7 | scores | 产品特征分数（例如ctr, cvr等） | 原始 |

### 3.3 异构用户行为序列建模（Heterogeneous User Behavior Sequence Modeling）

在本节中，我们将介绍如何对定义为C的实时用户行为上下文进行建模。遵循先前的工作[7, 14, 23]，我们也采用序列建模方法。然而，如第3.2.1节所讨论的，先前的工作只考虑用户正向交互的物品，因此它们无法基于我们提出的特征系统很好地处理用户行为序列建模。挑战在于用户行为数据存在两个方面的异构性。在我们的工作中，我们提出了异构用户行为序列建模（Heterogeneous User Behavior Sequence Modeling, HUBSM），特别针对以下两种异构性。

第一种是"物品曝光行为"和"物品浏览行为"的异构性。由于在推荐系统中物品点击相比物品曝光要稀疏得多，如果它们在一个序列中一起编码，我们认为物品浏览行为将占主导地位。因此我们选择分别对它们进行建模（即物品曝光行为序列建模和物品浏览行为序列建模）。第二种是"用户行为动作"和对应的"用户交互物品"的异构性，它们代表两种特征空间。用户行为动作特征揭示了用户如何对某个物品行为分布的规律，而物品特征则表征了对应物品特征的分布。我们选择先分别对它们进行编码，然后在后续的上下文感知重排序模型中融合以用于行为注意力机制（见第3.4节）。

这里，我们采用常用的门控循环单元（GRU）[6]作为编码器函数，通过更新门和重置门控制网络状态的更新（图4(b)）。我们定义使用多层GRU网络的序列编码器函数如下：

X_hat, s = GRU(X)  (1)

其中X = {xi}_{1≤i≤n}是特征向量的输入序列，X_hat = {x̂_i}_{1≤i≤n}是编码的输出序列，s是RNN的最终状态。这里的融合函数是两个输入特征向量序列X = {xi}_{1≤i≤n}和Y = {yi}_{1≤i≤n}的简单拼接，定义如下：

Z = CONCAT(X, Y)  (2)

其中Z = {zi}_{1≤i≤n}是融合编码的输出序列。当然，更复杂的编码模型（例如Transformer [21]）和融合函数（例如DNN）也可以在这里采用。考虑到设备上的模型大小，我们在实现中分别使用了GRU和拼接。

在以下两个段落中，我们将正式定义我们的两个具体建模：物品曝光行为序列建模和物品浏览行为序列建模（见图4(a)），其中用户行为上下文C由两个对应的元组(P̂_IE, B̂_IE)和(P̂_IPV, B̂_IPV)表示。我们将HUBSM部署在EdgeRec的设备上。基于RNN的循环计算特性，我们如第2.2.2节所讨论的那样同步且实时地对在线传入的用户行为进行建模。

**物品曝光行为序列建模（Item Exposure Behavior Sequence Modeling）。** 我们定义IE行为的动作特征向量输入序列为A_IE = {a_IE^i}_{1≤i≤m}，对应的物品特征向量为P_IE = {p_IE^i}_{1≤i≤m}。这里m是IE行为序列的预定义最大长度，对于更短的序列我们应用零填充。我们得到IE行为的动作编码输出序列Â_IE、物品编码P̂_IE和融合行为编码B̂_IE，分别如下列方程所示：

(Â_IE = {â_IE^i}_{1≤i≤m}, _) = GRU(A_IE)  (3)
(P̂_IE = {p̂_IE^i}_{1≤i≤m}, _) = GRU(P_IE)  (4)
B̂_IE = {b̂_IE^i}_{1≤i≤m} = CONCAT(Â_IE, P̂_IE)  (5)

**物品浏览行为序列建模（Item Page-View Behavior Sequence Modeling）。** 我们定义IPV行为的动作特征向量输入序列为A_IPV = {a_IPV^i}_{1≤i≤n}，对应的物品特征向量为P_IPV = {p_IPV^i}_{1≤i≤n}。这里n是IPV行为序列的预定义最大长度，对于更短的序列我们应用零填充。我们得到IPV行为的动作编码输出序列Â_IPV、物品编码P̂_IPV和融合行为编码B̂_IPV，分别如下列方程所示：

(Â_IPV = {â_IPV^i}_{1≤i≤n}, _) = GRU(A_IPV)  (6)
(P̂_IPV = {p̂_IPV^i}_{1≤i≤n}, _) = GRU(P_IPV)  (7)
B̂_IPV = {b̂_IPV^i}_{1≤i≤n} = CONCAT(Â_IPV, P̂_IPV)  (8)

### 3.4 基于行为注意力网络的上下文感知重排序（Context-aware Reranking with Behavior Attention Networks）

在本节中，我们将深入探讨我们的重排序方法——基于行为注意力网络的上下文感知重排序（见图4(a)），以同时捕获局部排序上下文以及候选物品与实时用户行为上下文之间的交互。遵循[1]，我们使用GRU网络对由初始排序模型排序的候选物品序列进行编码，并将最终状态作为局部排序上下文s。借助注意力技术，我们的重排序模型能够自动（软）搜索与排序目标物品相关的用户行为上下文部分。之前的CTR预测模型（例如DIN [23]和DUPN [14]）仅学习关注用户历史上与目标物品交互过的物品，因此无法基于上述注意力机制建模用户行为动作。相比之下，我们的方法首先从用户行为上下文中关注相关的交互物品（即找到相似的交互物品），然后注意力地组合对应的用户行为动作（这些动作指示用户对这些物品的潜在意图），共同表示为指导目标物品预测的上下文。我们称之为行为注意力（Behavior Attention），它特别利用了物品曝光行为上下文和物品浏览行为上下文。

**候选物品序列编码器（Candidate Item Sequence Encoder）。** 我们定义候选物品序列为P_CND = {p_CND^i}_{1≤i≤k}，由推荐系统服务器中的先前模型生成并排序。这里k是候选物品序列的预定义最大长度，对于更短的序列我们应用零填充。我们应用GRU网络对其进行编码，并将RNN的最终状态表示为局部排序上下文，如下列方程所示：

(P̂_CND = {p̂_CND^i}_{1≤i≤k}, s_CND) = GRU(P_CND)  (9)

其中P̂_CND是候选物品编码的输出序列，s_CND表示局部排序上下文。

**行为注意力（Behavior Attention）。** 针对编码为p̂_CND^t的目标候选物品t，我们首先分别关注用户行为物品序列编码P̂_IE和P̂_IPV，分别对应物品曝光和物品浏览行为。然后我们按照Bahdanau注意力机制[2]指示注意力分布为{att_IE^tj}_{1≤j≤m}和{att_IPV^tj}_{1≤j≤n}。最后，我们通过结合注意力分布与用户行为序列的融合行为编码B̂_IE和B̂_IPV来生成用户行为上下文c_IE^t和c_IPV^t。

具体来说，按照Transformer [21]中三元组（Query, Key, Value）的表示法，我们定义p̂_CND^t为Query，P̂_IE / P̂_IPV为Key，B̂_IE / B̂_IPV为Value。我们在这里认为注意力计算是为了（软）查找相似或相关的物品，因此比较的两个特征空间的表示应该是同质的。这就是为什么我们在第3.3节中选择分别编码"用户行为动作"和对应的"用户交互物品"，并使用用户行为物品序列作为相对于目标物品Query的Key。详见以下方程：

att_IE^tj = softmax(v_1^T tanh(W_1 p̂_CND^t + W_2 p̂_IE^j)), 1 ≤ j ≤ m  (10)
c_IE^t = ∑_{j=1}^m att_IE^tj b̂_IE^j  (11)
att_IPV^tj = softmax(v_2^T tanh(W_3 p̂_CND^t + W_4 p̂_IPV^j)), 1 ≤ j ≤ n  (12)
c_IPV^t = ∑_{j=1}^n att_IPV^tj b̂_IPV^j  (13)

其中权重W_1, W_2, W_3, W_4, v_1和v_2是训练参数。

**模型学习（Model Learning）。** 为了建模ϕ(·)，我们首先简单拼接IPV和IE上的用户行为上下文（即C）、目标候选物品的表示（即p̂_CND^t）和局部排序上下文（即s），然后将它们输入多层感知机（MLP）进行非线性变换。随后，采用交叉熵损失进行模型训练。

---

## 4 实验（EXPERIMENTS）

在本节中，我们通过离线和在线评估在真实的淘宝推荐系统数据集上展示我们模型的有效性。

### 4.1 离线评估（Offline Evaluation）

#### 4.1.1 数据集（Dataset）

我们从移动端淘宝的EdgeRec系统中收集在线日志和对应的物品特征（Tab. 1）。具体来说，我们从两个不同日期（2019-11-14和2019-11-15）的日志中随机采样，并将它们划分为训练集（22,072,671个样本）和测试集（200,000个样本）。此外，我们收集的数据集的IE行为序列和IPV行为序列的平均长度分别为56和26。

#### 4.1.2 对比方法与评估协议（Comparing Methods and Evaluation Protocol）

我们将我们的模型与两个在工业应用中广泛使用的代表性方法进行对比，即DNN-rank [5]和DLCM [1]。为了检验我们提出的异构用户行为序列建模（HUBSM）和基于行为注意力网络的上下文感知重排序（CRBAN）的有效性，除了我们的完整方法CRBAN+HUBSM(IE&IPV)之外，我们还准备了CRBAN的四个变体：（1）CRBAN+HUBSM(IE)，仅考虑物品曝光行为序列建模（IE-BSM）；（2）CRBAN+HUBSM(IPV)，仅考虑物品浏览行为序列建模（IPV-BSM）；（3）CRBAN+HUISM(IE&IPV)，使用DIN [23]而不是HUBSM建模用户行为上下文，尽管IE-BSM和IPV-BSM都被考虑。

我们使用PAI⁵支持的分布式TensorFlow训练模型，训练设置如下：batch size = 512, learning rate = 0.005, GRU层数 = 3, GRU隐层单元数 = 32, 注意力隐层单元数 = 32, MLP隐层大小 = 32, optimizer = "Adam"。注意DNN-rank和DLCM仅利用云端的特征，因为它们无法捕获端侧的特征（Tab. 1）。

GAUC [25]是通过对用户平均AUC [8]来广泛使用的推荐指标。在我们的论文中，我们通过将EdgeRec系统中对客户端原生模块请求r ∈ R（可视为一个重排序会话）的AUC进行平均来扩展GAUC，计算公式如下：

GAUC = (∑_{r∈R} #impression_r × AUC_r) / (∑_{r∈R} #impression_r)  (14)

其中#impression_r和AUC_r分别是对应请求r的物品曝光次数和AUC。

⁵https://data.aliyun.com/product/learn

#### 4.1.3 结果分析（Result Analysis）

表2显示了不同方法的GAUC性能。我们可以看到DLCM优于DNN-rank（第2行 vs. 第1行），这验证了引入局部排序上下文到重排序模型的有效性。此外，我们可以看到所有基于CRBAN的方法都显著优于DLCM。特别是，我们的完整方法CRBAN+HUBSM(IE&IPV)实现了GAUC 2%的相对提升（第6行 vs. 第2行）。这证明了在重排序模型中考虑实时用户行为上下文的优势。因此，如何建模用户行为上下文是我们接下来讨论的重点。

为了验证我们提出的异构用户行为序列建模方法，我们比较了HUBSM(IE&IPV)与HUBSM(IE)（第6行 vs. 第3行）和HUBSM(IPV)（第6行 vs. 第4行）。结果表明，"正向反馈"（即IPV）和"负向反馈"（即IE）的用户行为都对建模用户行为上下文有贡献。我们还发现HUBSM(IPV)优于HUBSM(IE)（第4行 vs. 第3行），这表明IPV用户行为可能比IE用户行为更重要。最后，通过比较HUBSM(IE&IPV)与HUISM(IE&IPV)（第6行 vs. 第5行）的结果表明了通过行为注意力机制同时考虑交互物品及其对应动作所带来的提升。

**表2：总体性能。** *表示与基线（DLCM）相比具有统计显著性提升，通过t检验在p值为0.05水平下测量。

| 方法 | GAUC |
|------|------|
| 1. DNN-rank | 0.62531 |
| 2. DLCM | 0.63552 |
| 3. CRBAN+HUBSM(IE) | 0.63818 |
| 4. CRBAN+HUBSM(IPV) | 0.64039 |
| 5. CRBAN+HUISM(IE&IPV) | 0.64283 |
| 6. CRBAN+HUBSM(IE&IPV) | 0.64825* |

### 4.2 在线性能（Online Performance）

#### 4.2.1 在线A/B测试（Online A/B Testing）

我们在部署于移动端淘宝的EdgeRec系统上进行了在线实验（即A/B测试）⁶。在淘宝瀑布流推荐系统中，在线指标包括PV、CTR、CLICK和GMV，它们评估了用户在推荐系统中浏览（PV）、点击（CTR、CLICK）和购买（GMV）的意愿程度。

EdgeRec已全面部署在移动端淘宝应用中并为数十亿用户提供服务。基线（即A测试）是没有EdgeRec的传统淘宝推荐系统。此处数百万不同的随机用户在同一时间分别参与在线测试A和B。在从2019-10-26到2019-11-08近两周的测试期间，带有完整模型CRBAN+HUBSM(IE&IPV)的EdgeRec平均贡献了高达1.57%的PV、7.18%的CTR、8.87%的CLICK和10.92%的GMV提升。这无疑是一个显著的改进，并证明了我们提出系统的有效性。

此外，我们回顾了淘宝推荐系统中沿展示位置的在线平均物品CTR。图5显示，部署EdgeRec后，当前页面末尾的CTR得到了大幅提升，这表明引入实时感知和实时反馈可以大大增加用户在推荐系统中的点击意愿，因为推荐系统能够及时满足用户的在线需求。

⁶我们使用一种名为"分层分桶（hierarchical bucketing）"的策略，其中云服务器上的不同算法对于端侧的在线实验是透明的。

#### 4.2.2 在线系统性能（Online System Performance）

除了展示生产中业务性能的在线A/B测试外，我们还对移动端淘宝中的EdgeRec效率进行了评估（表3），揭示了部署EdgeRec后在系统效率三个关键方面的显著改进。

**用户行为延迟时间（Delay time for user behaviors）** 影响系统捕获用户对物品个性化偏好的及时性，这可能影响推荐系统中的用户体验。由于网络带宽和延迟的限制，仅基于云到端框架的推荐系统可能导致捕获用户行为长达1分钟的延迟。然而，部署EdgeRec后，用户行为可以在设备上收集和消费，没有任何网络通信开销，这可以将延迟时间控制在300ms以内（例如从设备端数据库读取用户行为的时间）。

**系统响应时间（Response time of system）** 是影响推荐系统用户体验的另一个因素。当客户端原生模块在用户滚动推荐系统场景时向系统发起请求时，系统应及时响应并提供排序后的物品给用户，否则用户将等待，可能使其离开。由于为淘宝数亿用户服务如此复杂的推荐模型的计算开销，仅基于云计算的推荐系统可能导致包括网络传输在内的1秒响应时间。而在EdgeRec中，模型在每个用户的移动设备上服务，解决了集中式计算开销的问题，并使响应时间在100ms以内，且没有任何网络通信。

**用户平均系统反馈次数（Average times of system feedback for users）** 是影响推荐系统用户体验的另一个关键因素。它反映了用户在推荐系统中浏览时，系统能够调整将要展示给用户的物品排序的频率。系统能够越频繁地调整结果，就越能满足用户在推荐系统中的多样化需求。然而，没有EdgeRec的推荐系统无法使系统反馈次数大幅增加，因为这会加重云服务器的计算开销。因此，在当前云到端框架中，没有EdgeRec的淘宝推荐系统中用户平均系统反馈次数为3，分页大小为50（即用户平均发起3次分页请求）。相比之下，EdgeRec中没有明确的分页点，而是由客户端原生模块根据用户行为触发（见第2.2.1节）。在不增加云端额外计算开销的情况下，EdgeRec中的平均系统反馈次数可以达到15（即客户端原生模块平均在一页内触发5次重排序请求，因此总数为15）。

**表3：淘宝推荐系统中有（w/）和无（w/o）EdgeRec的系统性能。** 在流量高峰时段观测并按用户平均计算。

| 指标 | w/ EdgeRec | w/o EdgeRec |
|------|-----------|------------|
| 用户行为延迟时间 | ≤ 300ms | ≤ 1min |
| 系统响应时间 | ≤ 100ms | ≤ 1s |
| 系统反馈次数 | 15 | 3 |

### 4.3 案例研究（Case Study）

我们在移动端淘宝上进行了一个案例研究（图6），以展示HUBSM和CRBAN的有效性。总结来说，我们有以下观察：（1）用户在IPV中的动作揭示了他/她对该物品的正向意图程度偏好，例如加购或咨询客服，而用户在IE中的动作通常推断对该物品的负向意图，例如快速滚动或删除。这意味着HUBSM能够捕获用户对历史交互物品的潜在正向和负向意图。（2）候选衬衫借助IPV中的两件类似衬衫被预测为正向，而候选帽子借助IE中的两个具有较低负向意图程度的相似交互物品被预测为负向。这表明CRBAN能够建模候选物品与用户行为上下文之间的交互，从而更好地指导目标物品的预测。

---

## 5 结论与未来工作（CONCLUSION AND FUTURE WORK）

我们设计并实现了EdgeRec以解决瀑布流推荐系统中用户感知和系统反馈延迟的问题，这是首次尝试将推荐系统与边缘计算相结合。具体来说，我们提出了异构用户行为序列建模和基于行为注意力网络的上下文感知重排序来对用户丰富的行为进行建模。广泛的离线和在线评估验证了EdgeRec在工业推荐系统中的有效性。我们相信EdgeRec将在未来为推荐系统的工业界和研究界带来许多有趣的课题，例如千人千模（即基于设备训练的模型个性化）、联邦学习[3, 11]和交互式推荐[4]。

---

## 参考文献（REFERENCES）

[1] Qingyao Ai, Keping Bi, Jiafeng Guo, and W Bruce Croft. 2018. Learning a deep listwise context model for ranking refinement. 见于 SIGIR.

[2] Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. 2014. Neural machine translation by jointly learning to align and translate. arXiv (2014).

[3] Fei Chen, Zhenhua Dong, Zhenguo Li, and Xiuqiang He. 2018. Federated meta-learning for recommendation. arXiv (2018).

[4] Haokun Chen, Xinyi Dai, Han Cai, Weinan Zhang, Xuejian Wang, Ruiming Tang, Yuzhou Zhang, and Yong Yu. 2019. Large-scale interactive recommendation with tree-structured policy gradient. 见于 AAAI.

[5] Heng-Tze Cheng, Levent Koc, Jeremiah Harmsen, Tal Shaked, Tushar Chandra, Hrishi Aradhye, Glen Anderson, Greg Corrado, Wei Chai, Mustafa Ispir, et al. 2016. Wide & deep learning for recommender systems. 见于 Proceedings of the 1st workshop on deep learning for recommender systems.

[6] Kyunghyun Cho, Bart Van Merriënboer, Dzmitry Bahdanau, and Yoshua Bengio. 2014. On the properties of neural machine translation: Encoder-decoder approaches. arXiv (2014).

[7] Paul Covington, Jay Adams, and Emre Sargin. 2016. Deep neural networks for youtube recommendations. 见于 RecSys.

[8] Tom Fawcett. 2006. An introduction to ROC analysis. Pattern recognition letters 27, 8 (2006), 861–874.

[9] Yufei Feng, Fuyu Lv, Weichen Shen, Menghan Wang, Fei Sun, Yu Zhu, and Keping Yang. 2019. Deep Session Interest Network for Click-Through Rate Prediction. 见于 IJCAI. 2301–2307.

[10] Mihajlo Grbovic and Haibin Cheng. 2018. Real-time personalization using embeddings for search ranking at airbnb. 见于 SIGKDD.

[11] Andrew Hard, Kanishka Rao, Rajiv Mathews, Swaroop Ramaswamy, Françoise Beaufays, Sean Augenstein, Hubert Eichner, Chloé Kiddon, and Daniel Ramage. 2018. Federated learning for mobile keyboard prediction. arXiv (2018).

[12] Chao Li, Zhiyuan Liu, Mengmeng Wu, Yuchi Xu, Pipei Huang, Huan Zhao, Guoliang Kang, Qiwei Chen, Wei Li, and Dik Lun Lee. 2019. Multi-Interest Network with Dynamic Routing for Recommendation at Tmall. arXiv (2019).

[13] Tomáš Mikolov, Martin Karafiát, Lukáš Burget, Jan Černock`y, and Sanjeev Khudanpur. 2010. Recurrent neural network based language model. 见于 ISCA.

[14] Yabo Ni, Dan Ou, Shichen Liu, Xiang Li, Wenwu Ou, Anxiang Zeng, and Luo Si. 2018. Perceive your users in depth: Learning universal user representations from multiple e-commerce tasks. 见于 SIGKDD.

[15] Changhua Pei, Yi Zhang, Yongfeng Zhang, Fei Sun, Xiao Lin, Hanxiao Sun, Jian Wu, Peng Jiang, Junfeng Ge, Wenwu Ou, et al. 2019. Personalized re-ranking for recommendation. 见于 RecSys.

[16] Qi Pi, Weijie Bian, Guorui Zhou, Xiaoqiang Zhu, and Kun Gai. 2019. Practice on Long Sequential User Behavior Modeling for Click-Through Rate Prediction. arXiv (2019).

[17] Weisong Shi, Jie Cao, Quan Zhang, Youhuizi Li, and Lanyu Xu. 2016. Edge computing: Vision and challenges. IEEE Internet of Things Journal 3, 5 (2016), 637–646.

[18] C. Sun, H. Li, X. Li, J. Wen, Q. Xiong, and W. Zhou. 2020. Convergence of Recommender Systems and Edge Computing: A Comprehensive Survey. IEEE Access 8 (2020), 47118–47132.

[19] Tarik Taleb, Sunny Dutta, Adlen Ksentini, Muddesar Iqbal, and Hannu Flinck. 2017. Mobile edge computing potential in making cities smarter. IEEE Communications Magazine 55, 3 (2017).

[20] Carlo Vallati, Antonio Virdis, Enzo Mingozzi, and Giovanni Stea. 2016. Mobile-edge computing come home connecting things in future smart homes using LTE device-to-device communications. IEEE Consumer Electronics Magazine 5, 4 (2016), 77–83.

[21] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. 见于 NIPS.

[22] Guorui Zhou, Na Mou, Ying Fan, Qi Pi, Weijie Bian, Chang Zhou, Xiaoqiang Zhu, and Kun Gai. 2019. Deep interest evolution network for click-through rate prediction. 见于 AAAI.

[23] Guorui Zhou, Xiaoqiang Zhu, Chenru Song, Ying Fan, Han Zhu, Xiao Ma, Yanghui Yan, Junqi Jin, Han Li, and Kun Gai. 2018. Deep interest network for click-through rate prediction. 见于 SIGKDD.

[24] Meizi Zhou, Zhuoye Ding, Jiliang Tang, and Dawei Yin. 2018. Micro behaviors: A new perspective in e-commerce recommender systems. 见于 WSDM.

[25] Han Zhu, Junqi Jin, Chang Tan, Fei Pan, Yifan Zeng, Han Li, and Kun Gai. 2017. Optimized cost per click in taobao display advertising. 见于 SIGKDD. 2191–2200.

[26] Tao Zhuang, Wenwu Ou, and Zhirong Wang. 2018. Globally optimized mutual influence aware ranking in e-commerce search. 见于 AAAI.
