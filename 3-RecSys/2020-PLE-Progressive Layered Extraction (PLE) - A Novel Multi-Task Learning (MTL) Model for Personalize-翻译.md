# 渐进式分层提取（PLE）：一种用于个性化推荐的新型多任务学习（MTL）模型

洪彦唐
腾讯PCG
中国
violatang@tencent.com

赵明
腾讯PCG
中国
marcozhao@tencent.com

刘俊宁
腾讯PCG
中国
korchinliu@tencent.com

宫旭东
腾讯PCG
中国
xudonggong@tencent.com

## 摘要

多任务学习（MTL）已成功应用于许多推荐应用。然而，由于现实推荐系统中复杂且相互竞争的任务相关性，MTL模型常常因负迁移而性能退化。此外，通过对现有最先进MTL模型的广泛实验，我们观察到一个有趣的跷跷板现象：一个任务的性能提升往往以损害其他任务的性能为代价。为了解决这些问题，我们提出了一种渐进式分层提取（PLE）模型，其具有新颖的共享结构设计。PLE明确分离共享组件和任务特定组件，并采用渐进式路由机制逐步提取和分离更深层的语义知识，提高了通用设置下联合表示学习和跨任务信息路由的效率。我们将PLE应用于复杂相关和正常相关的任务，涵盖从两任务到多任务的情况，使用包含10亿样本的真实腾讯视频推荐数据集。结果表明，PLE在不同任务相关性和任务组规模下均显著优于最先进的MTL模型。此外，PLE在腾讯大规模内容推荐平台上的在线评估显示，与最先进的MTL模型相比，观看次数增加了2.23%，观看时长增加了1.84%，这是一个显著的改进，证明了PLE的有效性。最后，在公开基准数据集上的大量离线实验表明，PLE除了推荐之外，还可以应用于多种场景以消除跷跷板现象。PLE现已成功部署到腾讯在线视频推荐系统中。

允许为个人或课堂教学目的制作本作品的全部或部分数字或硬拷贝，无需付费，但前提是不得为盈利或商业目的制作或分发拷贝，且拷贝在首页包含此声明和完整引文。归因于本作品中由他人拥有的组件的版权必须得到尊重。允许带有致谢的摘要。如需以其他方式复制、重新发布、在服务器上发布或分发到列表，需要事先获得特定许可和/或支付费用。请向permissions@acm.org请求许可。

RecSys '20，2020年9月22日至26日，巴西虚拟活动
© 2020国际计算机协会（ACM）
ACM ISBN 978-1-4503-7583-2/20/09...$15.00
https://doi.org/10.1145/3383313.3412236

## CCS概念

• 信息系统 → 推荐系统；检索模型与排序；

## 关键词

多任务学习；推荐系统；跷跷板现象

## ACM引用格式

Hongyan Tang, Junning Liu, Ming Zhao, and Xudong Gong. 2020. Progressive Layered Extraction (PLE): A Novel Multi-Task Learning (MTL) Model for Personalized Recommendations. 收录于第十四届ACM推荐系统大会（RecSys '20），2020年9月22日至26日，巴西虚拟活动。ACM，纽约州纽约市，美国，10页。https://doi.org/10.1145/3383313.3412236

## 1 引言

个性化推荐在在线应用中扮演着至关重要的角色。推荐系统（RS）需要整合各种用户反馈来建模用户兴趣，并最大化用户参与度和满意度。然而，由于问题的高维度，学习算法通常难以直接处理用户满意度。同时，用户满意度和参与度有许多可以直接学习的主要因素，例如点击、看完、分享、收藏和评论的可能性等。因此，将多任务学习（MTL）应用于推荐系统以同时建模用户满意度或参与度的多个方面已成为一种日益增长的趋势。事实上，这已成为主要行业应用中的主流方法[11, 13, 14, 25]。

MTL在单个模型中同时学习多个任务，并通过任务之间的信息共享来提高学习效率[2]。然而，现实推荐系统中的任务往往相关性较弱甚至相互冲突，这可能导致称为负迁移的性能退化[21]。通过在真实大规模视频推荐系统和公开基准数据集上的广泛实验，我们发现当任务相关性复杂且有时依赖于样本时（即多个任务无法同时相对于相应的单任务模型得到改进，本文称之为跷跷板现象），现有的MTL模型通常以牺牲其他任务的性能为代价来改进某些任务。

先前的工作更多地致力于解决负迁移问题，但忽略了跷跷板现象。例如，十字绣网络[16]和水闸网络[18]提出学习静态线性组合来融合不同任务的表示，这无法捕捉样本依赖性。MMOE[13]应用门控网络基于输入组合底层专家以处理任务差异，但忽略了专家之间的区分和交互，在我们的工业实践中被证明存在跷跷板现象。因此，设计一个更强大、更高效的模型来处理复杂相关性并消除具有挑战性的跷跷板现象至关重要。

为实现这一目标，我们提出了一种新颖的MTL模型，称为渐进式分层提取（PLE），该模型在共享网络设计中更好地利用先验知识来捕捉复杂的任务相关性。与MMOE中粗略共享参数的方式相比，PLE明确分离共享专家和任务特定专家，以减轻通用知识和任务特定知识之间的有害参数干扰。此外，PLE引入了多层专家和门控网络，并应用渐进式分离路由，从底层专家中提取更深层知识，并在高层逐步分离任务特定参数。

为评估PLE的性能，我们在真实工业推荐数据集和主要可用的公开数据集（包括census-income[5]、合成数据[13]和Ali-CCP¹）上进行了大量实验。实验结果表明，PLE在所有数据集上均优于最先进的MTL模型，不仅对具有挑战性复杂相关性的任务组表现出一致的改进，而且在不同场景下具有正常相关性的任务组上也表现出一致的改进。此外，在腾讯大规模视频推荐系统上在线指标的显著改进证明了PLE在现实推荐应用中的优势。

本文的主要贡献总结如下：

• 通过在腾讯大规模视频推荐系统和公开基准数据集上的广泛实验，观察到了一个有趣的跷跷板现象：由于复杂的内在相关性，最先进的MTL模型通常以牺牲其他任务的性能为代价来改进某些任务，并且未能在所有任务上超越相应的单任务模型。

• 提出了一种具有新颖共享学习结构的PLE模型，从联合表示学习和信息路由的角度提高共享学习效率，从而进一步解决跷跷板现象和负迁移问题。除推荐应用外，PLE还可以灵活应用于多种场景。

• 进行了大量的离线实验，在工业和公开基准数据集上评估了PLE的有效性。腾讯全球最大内容推荐平台之一的在线A/B测试结果也表明，在现实应用中PLE相对于最先进的MTL模型有显著改进：观看次数增加2.23%，观看时长增加1.84%，产生了显著的商业收益。PLE现已成功部署到推荐系统中，并可潜在地应用于许多其他推荐应用。

¹https://tianchi.aliyun.com/dataset/dataDetail?dataId=408

## 2 相关工作

高效的多任务学习模型以及MTL模型在推荐系统中的应用是我们工作相关的两个研究领域。在本节中，我们简要讨论这两个领域的相关工作。

### 2.1 多任务学习模型

硬参数共享[2]（如图1a所示）是最基本、最常用的MTL结构，但由于参数直接在任务之间共享，可能因任务冲突而遭受负迁移。为处理任务冲突，十字绣网络[16]（图1f）和水闸网络[18]（图1g）均提出学习线性组合的权重，以选择性地融合不同任务的表示。然而，在这些模型中，表示对所有样本使用相同的静态权重进行组合，跷跷板现象未得到解决。在这项工作中，所提出的PLE（渐进式分层提取）模型应用带有门控结构的渐进式路由机制，基于输入融合知识，实现了针对不同输入的自适应组合。

已有一些研究将门控结构和注意力网络应用于信息融合。MOE[8]首先提出在底层共享一些专家，并通过门控网络组合专家。MMOE[13]扩展了MOE，在MTL中为每个任务使用不同的门控以获得不同的融合权重。类似地，MRAN[24]应用多头自注意力在不同特征集上学习不同的表示子空间。在MOE、MMOE（如图1所示）和MRAN中，专家和注意力模块在所有任务之间共享，没有任务特定的概念。相比之下，我们提出的CGC（定制化门控控制）和PLE模型明确分离任务通用和任务特定参数，以避免因复杂任务相关性导致的参数冲突。尽管MMOE从理论上有可能收敛到我们的网络设计，但网络设计中的先验知识很重要，MMOE在实践中很难发现收敛路径。Liu等人[10]应用任务特定的注意力网络选择性地融合共享特征，但在注意力网络融合之前，不同任务仍然共享相同的表示。以往的工作都没有明确解决表示学习和路由的联合优化问题，特别是以不可分割的联合方式，而本工作首次尝试在联合学习和路由的通用框架上提出一种新颖的渐进式分离方式。

也有一些工作利用AutoML方法寻找良好的网络结构。SNR框架[12]通过二元随机变量控制子网络之间的连接，并应用NAS[26]搜索最优结构。类似地，Gumbel矩阵路由框架[15]利用Gumbel-Softmax技巧学习被建模为二元矩阵的MTL模型路由。Rosenbaum等人[17]将路由过程建模为MDP，应用MARL[19]训练路由网络。这些工作中的网络结构基于某些简化的假设设计，不够通用。[17]中的路由网络在每个深度为每个任务选择不超过一个功能块，降低了模型的表达能力。Gumbel矩阵路由网络[15]对表示学习施加了约束，因为每个任务的输入在每一层需要合并为一个表示。此外，这些框架中的融合权重不能针对不同输入进行调整，昂贵的搜索成本是这些方法寻找最优结构的另一挑战。

### 2.2 推荐系统中的多任务学习

为更好地利用各种用户行为，多任务学习已广泛应用于推荐系统并取得了显著改进。一些研究将传统推荐算法（如协同过滤和矩阵分解）与MTL相结合。Lu等人[11]和Wang等人[23]对为推荐任务和解释任务学习的潜在表示施加正则化，以联合优化它们。Wang等人[22]将协同过滤与MTL相结合，以更高效地学习用户-物品相似度。与本文的PLE相比，这些基于分解的模型表达能力较低，无法充分利用任务之间的共性。

作为最基本的MTL结构，硬参数共享已应用于许多基于深度神经网络的推荐系统。ESSM[14]引入了CTR（点击率）和CTCVR两个辅助任务，并在CTR和CVR（转化率）之间共享嵌入参数，以提高CVR预测的性能。Hadash等人[7]提出一个多任务框架，同时学习排序任务和评分任务的参数。[1]中的文本推荐任务通过在底层共享表示得到改进。然而，在任务相关性较弱或复杂的情况下，硬参数共享常常遭受负迁移和跷跷板现象的影响。相比之下，我们提出的模型引入了一种新颖的共享机制，以实现更高效的一般性信息共享。

除硬参数共享外，也有一些推荐系统应用了具有更高效共享学习机制的MTL模型。为更好地利用任务之间的相关性，Chen等人[3]利用层次化多指针协同注意力[20]来提高推荐任务和解释任务的性能。然而，模型中每个任务的塔网络共享相同的表示，仍可能遭受任务冲突。通过应用MMOE[13]通过不同的门控网络为每个任务组合共享专家，[25]中的YouTube视频推荐系统可以更好地捕捉任务差异并高效地优化多个目标。与将所有专家一视同仁而不加区分的MMOE相比，本文的PLE明确分离了任务通用和任务特定专家，并采用新颖的渐进式分离路由，在真实视频推荐系统中实现了相对于MMOE的显著改进。

## 3 推荐中MTL的跷跷板现象

负迁移是MTL中常见的现象，尤其是对于弱相关任务[21]。对于复杂的任务相关性，特别是样本依赖的相关性模式，我们还观察到跷跷板现象：对于当前的MTL模型来说，提高共享学习效率并在所有任务上实现相对于相应单任务模型的显著改进是很困难的。在本节中，我们基于腾讯的大规模视频推荐系统，全面介绍和研究跷跷板现象。

### 3.1 一个用于视频推荐的MTL排序系统

在本小节中，我们简要介绍服务于腾讯新闻的MTL排序系统，腾讯新闻是全球最大的内容平台之一，基于多样化的用户反馈向用户推荐新闻和视频。如图2所示，MTL排序系统中有多个目标来建模不同的用户行为，如点击、分享和评论。在离线训练过程中，我们基于从用户日志中提取的用户行为训练MTL排序模型。每次在线请求后，排序模型为每个任务输出预测值，然后基于加权乘法的排序模块通过方程1所示的组合函数将这些预测分数组合为最终分数，并最终向用户推荐排名最高的视频。

$$\text{score} = p_{\text{VTR}}^{w_{\text{VTR}}} \times p_{\text{VCR}}^{w_{\text{VCR}}} \times p_{\text{SHR}}^{w_{\text{SHR}}} \times p_{\text{CMR}}^{w_{\text{CMR}}} \times \cdots \times f(\text{video\_len}) \tag{1}$$

其中每个w决定每个预测分数的相对重要性，f(video_len)是非线性变换函数，如视频时长上的sigmoid或log函数。wᵥₜᵣ、wᵥ꜀ᵣ、wₛₕᵣ、w꜀ₘᵣ是通过在线实验搜索优化的超参数，以最大化在线指标。

在所有任务中，VCR（观看完成比）和VTR（有效观看率）是两个重要的目标，分别建模观看次数和观看时长的关键在线指标。具体来说，VCR预测是一个回归任务，使用MSE损失训练以预测每次观看的完成比例。VTR预测是一个二分类任务，使用交叉熵损失训练以预测有效观看的概率，有效观看定义为超过某个观看时长阈值的播放行为。VCR和VTR之间的相关性模式很复杂。首先，VTR的标签是播放行为和VCR的耦合因素，因为只有观看时长超过阈值的播放行为才会被视为有效观看。其次，播放行为的分布更加复杂，因为来自WIFI环境下自动播放场景的样本表现出较高的平均播放概率，而来自没有自动播放的显式点击场景的样本则表现出较低的播放概率。由于这种复杂且高度样本依赖的相关性模式，在对VCR和VTR进行联合建模时观察到了跷跷板现象。

### 3.2 MTL中的跷跷板现象

为了更好地理解跷跷板现象，我们在排序系统中对复杂相关的VCR和VTR任务组进行了实验分析，对比了单任务模型和最先进的MTL模型。除了硬参数共享、十字绣[16]、水闸网络[18]和MMOE[13]之外，我们还评估了两种创新性提出的结构：非对称共享和定制化共享：

• **非对称共享**是一种新颖的共享机制，用于捕捉任务之间的非对称关系。如图1b所示，底层在任务之间非对称共享，共享哪个任务的表示取决于任务之间的关系。可以使用常见的融合操作，如拼接、求和池化和平均池化，来组合不同任务底层的输出。

• **定制化共享**如图1c所示，明确分离共享和任务特定参数，以避免固有问题和负迁移。与单任务模型相比，定制化共享增加了一个共享底层来提取共享信息，并将共享底层和任务特定层的拼接结果馈送到相应任务的塔层。

图3展示了实验结果，其中越靠近右上角的点表示性能越好（AUC越高，MSE越低）。值得注意的是，在我们的系统中，AUC或MSE提高0.1%就会对在线指标产生显著改进，[4, 6, 14]中也提到了这一点。可以看出，硬参数共享和十字绣网络遭受显著的负迁移，在VTR上表现最差。通过创新的共享机制捕捉非对称关系，非对称共享在VTR上取得了显著改进，但在VCR上表现出显著退化，与水闸网络类似。由于明确分离了共享层和任务特定层，定制化共享在VTR上仍略有不足，但在VCR上比单任务模型有所改进。MMOE在两个任务上都优于单任务模型，但VCR的改进仅为+0.0001，处于临界状态。尽管这些模型在这两个具有挑战性的任务上表现出不同的学习效率，但我们清楚地观察到跷跷板现象：一个任务的改进往往导致另一个任务的性能退化，因为没有基线MTL模型完全位于第二象限。在公开基准数据集上使用最先进模型的实验也展现了明显的跷跷板现象。详细信息将在第5.2节中提供。

如前所述，VCR和VTR之间的相关性模式复杂且依赖于样本。具体而言，VCR和VTR之间存在部分有序关系，不同样本表现出不同的相关性。因此，使用相同的静态权重为所有样本组合共享表示的十字绣和水闸网络无法捕捉样本依赖性，并遭受跷跷板现象的影响。通过应用门控网络基于输入获得融合权重，MMOE在一定程度上处理了任务差异和样本差异，优于其他基线MTL模型。然而，在MMOE中，所有任务之间共享专家且不加区分，这无法捕捉复杂的任务相关性，并可能给某些任务带来有害噪声。此外，MMOE忽略了不同专家之间的交互，进一步限制了联合优化的性能。除VCR和VTR之外，工业推荐应用中还有许多复杂相关的任务，因为人类行为往往是微妙和复杂的，例如在线广告和电子商务平台中的CTR预测和CVR预测[14]。因此，一个考虑专家之间的区分和交互的强大网络对于消除由复杂任务相关性引起的具有挑战性的跷跷板现象至关重要。

在本文中，我们提出了一种渐进式分层提取（PLE）模型来解决跷跷板现象和负迁移问题。PLE的关键思想如下。首先，它明确分离共享专家和任务特定专家，以避免有害的参数干扰。其次，引入多层专家和门控网络以融合更抽象的表示。最后，采用新颖的渐进式分离路由来建模专家之间的交互，并在复杂相关的任务之间实现更高效的知识迁移。如图3所示，PLE在两个任务上均取得了比MMOE显著的改进。结构设计的细节和实验将分别在第4节和第5节中描述。

## 4 渐进式分层提取

为解决跷跷板现象和负迁移问题，我们在本节中提出了一种具有新颖共享结构设计的渐进式分层提取（PLE）模型。首先，提出了一种定制化门控控制（CGC）模型，明确分离共享专家和任务特定专家。其次，将CGC扩展为具有多层门控网络和渐进式分离路由的通用PLE模型，以实现更高效的信息共享和联合学习。最后，优化损失函数以更好地应对MTL模型联合训练的实际挑战。

### 4.1 定制化门控控制

受定制化共享（通过明确分离共享层和任务特定层实现了与单任务模型相似的性能）的启发，我们首先介绍定制化门控控制（CGC）模型。如图4所示，底部有一些专家模块，顶部有一些任务特定的塔网络。每个专家模块由多个称为专家的子网络组成，每个模块中的专家数量是一个可调的超参数。类似地，塔网络也是多层网络，其宽度和深度作为超参数。具体而言，CGC中的共享专家负责学习共享模式，而特定任务的模式则由任务特定专家提取。每个塔网络从共享专家和自己的任务特定专家中吸收知识，这意味着共享专家的参数受所有任务的影响，而任务特定专家的参数仅受相应特定任务的影响。

在CGC中，共享专家和任务特定专家通过门控网络进行选择性融合。如图4所示，门控网络的结构基于单层前馈网络，以SoftMax作为激活函数，输入作为选择器来计算所选向量的加权和，即专家的输出。更准确地说，任务k的门控网络输出公式为：

$$g_k(x) = w_k(x) S_k(x) \tag{2}$$

其中x是输入表示，w_k(x)是通过线性变换和SoftMax层计算任务k权重向量的权重函数：

$$w_k(x) = \text{Softmax}(W_k^g x) \tag{3}$$

其中W_k^g ∈ ℝ^(m_k + m_s) × d 是参数矩阵，m_s和m_k分别是共享专家和任务k特定专家的数量，d是输入表示的维度。S_k(x)是由所有选定向量（包括共享专家和任务k的特定专家）组成的选择矩阵：

$$S_k(x) = [E_T^{(k,1)}, E_T^{(k,2)}, \ldots, E_T^{(k,m_k)}, E_S^{(s,1)}, E_S^{(s,2)}, \ldots, E_S^{(s,m_s)}]^T \tag{4}$$

最后，任务k的预测为：

$$y_k(x) = t_k(g_k(x)) \tag{5}$$

其中t_k表示任务k的塔网络。

与MMOE相比，CGC移除了一个任务的塔网络与其他任务的任务特定专家之间的连接，使不同类型的专家能够专注于高效学习不同的知识而不受干扰。结合门控网络基于输入动态融合表示的优点，CGC实现了任务之间更灵活的平衡，并更好地处理了任务冲突和样本依赖的相关性。

### 4.2 渐进式分层提取

CGC明确分离了任务特定和共享组件。然而，在深度多任务学习中，学习需要逐步塑造越来越深的语义表示，而通常中间表示应被视为共享还是任务特定并不明确。为解决这个问题，我们通过渐进式分层提取（PLE）对CGC进行泛化。如图5所示，PLE中有多层提取网络来提取更高级别的共享信息。除了任务特定专家的门控外，提取网络还为共享专家采用门控网络，以组合该层中所有专家的知识。因此，PLE中不同任务的参数在早期层中不像CGC那样完全分离，而是在上层中逐步分离。高层提取网络中的门控网络将低层提取网络中门控的融合结果作为选择器，而非原始输入，因为这可能为选择在高层专家中提取的抽象知识提供更好的信息。

PLE中权重函数、选择矩阵和门控网络的计算与CGC相同。具体而言，PLE第j个提取网络中任务k的门控网络公式为：

$$g_{k,j}(x) = w_{k,j}(g_{k,j-1}(x)) S_{k,j}(x) \tag{6}$$

其中w_{k,j}是以g_{k,j-1}为输入的任务k的权重函数，S_{k,j}是第j个提取网络中任务k的选择矩阵。值得注意的是，PLE中共享模块的选择矩阵与任务特定模块略有不同，因为它由该层中的所有共享专家和任务特定专家组成。

计算所有门控网络和专家后，我们最终可以得到PLE中任务k的预测：

$$y_k(x) = t_k(g_{k,N}(x)) \tag{7}$$

通过多层专家和门控网络，PLE为每个任务提取和组合更深层的语义表示以提高泛化能力。如图1所示，MMOE的路由策略是全连接，CGC是早期分离。不同的是，PLE采用渐进式分离路由，从所有低层专家吸收信息，提取更高级别的共享知识，并逐步分离任务特定参数。渐进式分离的过程类似于化学中从化合物中提取所需产品的过程。在PLE的知识提取和转换过程中，低层表示在高层共享专家中被联合提取/聚合和路由，获得共享知识并逐步分发到特定塔层，从而实现更高效和灵活的联合表示学习和共享。尽管MMOE的全连接路由看起来像是CGC和PLE的通用设计，但第5.3节的实践研究表明，尽管存在可能性，MMOE并不能收敛到CGC或PLE的结构。

### 4.3 MTL的联合损失优化

设计了高效的网络结构后，我们现在关注以端到端方式联合训练任务特定层和共享层。在多任务学习中，联合损失的常见公式是每个任务损失的加权和：

$$L(\theta_1, \ldots, \theta_K, \theta_s) = \sum_{k=1}^K \omega_k L_k(\theta_k, \theta_s) \tag{8}$$

其中θ_s表示共享参数，K是任务数量，L_k、ω_k和θ_k分别是任务k的损失函数、损失权重和任务特定参数。

然而，存在几个问题使得实践中MTL模型的联合优化具有挑战性。在本文中，我们优化了联合损失函数以解决在现实推荐系统中遇到的两个关键问题。第一个问题是由于顺序用户行为导致的异质样本空间。例如，用户只能在点击某项目后才能分享或评论，这导致不同任务的样本空间不同，如图6所示。为联合训练这些任务，我们将所有任务的样本空间的并集作为整个训练集，并在计算每个单独任务的损失时忽略其样本空间外的样本：

$$L_k(\theta_k, \theta_s) = \frac{1}{\sum_i \delta_k^i} \sum_i \delta_k^i \text{loss}_k(\hat{y}_k^i(\theta_k, \theta_s), y_k^i) \tag{9}$$

其中loss_k是任务k基于预测ŷₖⁱ和真实值yₖⁱ计算的样本i的损失，δₖⁱ ∈ {0, 1}表示样本i是否在任务k的样本空间中。

第二个问题是MTL模型的性能对训练过程中损失权重的选择很敏感[9]，因为它决定了每个任务对联合损失的相对重要性。在实践中，我们观察到每个任务在不同训练阶段可能具有不同的重要性。因此，我们将每个任务的损失权重视为动态权重而非静态权重。首先，我们为任务k设置初始损失权重ωₖ,₀，然后根据更新比率γₖ在每个步骤后更新其损失权重：

$$\omega_k^{(t)} = \omega_{k,0} \times \gamma_k^t \tag{10}$$

其中t表示训练轮次，ωₖ,₀和γₖ是模型的超参数。

## 5 实验

在本节中，我们在腾讯的大规模推荐系统和公开基准数据集上进行了大量的离线和在线实验，以评估所提出模型的有效性。我们还分析了所有基于门控的MTL模型中的专家利用率，以更好地理解门控网络的工作机制，并进一步验证CGC和PLE的结构价值。

### 5.1 在腾讯视频推荐系统上的评估

在本小节中，我们对腾讯视频推荐系统中具有复杂和正常相关性的任务组以及多任务场景进行离线和在线实验，以评估所提出模型的性能。

#### 5.1.1 数据集

我们通过在连续8天内从服务于腾讯新闻的视频推荐系统中采样用户日志来收集工业数据集。数据集中有4692.6万用户、268.2万视频和9.95亿样本。如前所述，VCR、CTR、VTR、SHR（分享率）和CMR（评论率）是数据集中建模用户偏好的任务。

#### 5.1.2 基线模型

在实验中，我们将CGC和PLE与单任务模型、非对称共享、定制化共享以及最先进的MTL模型（包括十字绣网络、水闸网络和MMOE）进行了比较。由于PLE中共享多层专家，我们将MMOE扩展为ML-MMOE（多层MMOE），如图1h所示，通过添加多层专家进行公平比较。在ML-MMOE中，高层专家通过门控网络组合低层专家的表示，所有门控网络共享相同选择器。

#### 5.1.3 实验设置

在实验中，VCR预测是一个回归任务，使用MSE损失训练和评估；建模其他动作的任务均为二分类任务，使用交叉熵损失训练并以AUC评估。前7天的样本用于训练，其余样本作为测试集。我们为MTL模型和单任务模型中的每个任务采用三层MLP网络，激活函数为RELU，隐藏层大小为[256, 128, 64]。对于MTL模型，我们将专家实现为单层网络，并调整以下模型特定的超参数：共享层数量、硬参数共享和十字绣网络中的十字绣单元、所有基于门控模型的专家数量。为公平比较，我们将所有多级MTL模型实现为两级模型，以保持网络深度相同。

表1: VTR/VCR任务组的性能

| 模型 | AUC (VTR) | MSE (VCR) | MTL增益 (VTR) | MTL增益 (VCR) |
|---|---|---|---|---|
| 单任务 | 0.6787 | 0.1321 | - | - |
| 硬参数共享 | 0.6740 | 0.1320 | -0.0047 | +1.8E-5 |
| 非对称共享 | 0.6823 | 0.1346 | +0.0036 | -0.0025 |
| 十字绣 | 0.6740 | 0.1320 | -0.0047 | +1.6E-5 |
| 水闸网络 | 0.6825 | 0.1329 | +0.0038 | -0.0008 |
| 定制化共享 | 0.6780 | 0.1318 | -0.0007 | +0.0002 |
| MMOE | 0.6803 | 0.1319 | +0.0016 | +0.0001 |
| ML-MMOE | 0.6815 | 0.1329 | +0.0028 | -0.0009 |
| CGC | 0.6832 | 0.1320 | +0.0045 | +3.5E-5 |
| PLE | 0.6831 | 0.1307 | +0.0044 | +0.0013 |

除常见的评估指标（如AUC和MSE）外，我们定义了一个MTL增益指标，以定量评估多任务学习相对于单任务模型对特定任务的收益。如方程11所示，对于给定的任务组和MTL模型q，q在任务A上的MTL增益定义为MTL模型q相对于使用相同网络结构和训练样本的单任务模型在任务A上的性能改进。

$$\text{MTL增益} = 
\begin{cases}
M_{\text{MTL}} - M_{\text{single}}, & M \text{为正向指标} \\
M_{\text{single}} - M_{\text{MTL}}, & M \text{为负向指标}
\end{cases} \tag{11}$$

#### 5.1.4 复杂相关性任务的评估

为更好地捕捉主要的在线参与度指标（如观看次数和观看时长），我们首先在VCR/VTR任务组上进行实验。表1展示了实验结果，我们将最佳分数加粗显示，性能退化（负MTL增益）标灰。结果表明，CGC和PLE在VTR上显著优于所有基线模型。由于VTR和VCR之间的复杂相关性，我们可以通过锯齿状的灰色分布清楚地观察到跷跷板现象：一些模型改进了VCR但损害了VTR，另一些则改进了VTR但损害了VCR。具体来说，MMOE在两项任务上均优于单任务模型，但改进不显著；而ML-MMOE改进了VTR但损害了VCR。与MMOE和ML-MMOE相比，CGC大幅改进了VTR，同时也略微改进了VCR。最终，PLE以相似的收敛速度实现了比上述模型更显著的改进，取得了最佳的VCR MSE和最佳的VTR AUC之一。

表2: CTR/VCR任务组的性能

| 模型 | AUC (CTR) | MSE (VCR) | MTL增益 (CTR) | MTL增益 (VCR) |
|---|---|---|---|---|
| 单任务 | 0.7379 | 0.1179 | - | - |
| 十字绣 | 0.7220 | 0.1158 | -0.0158 | +0.0021 |
| 水闸网络 | 0.7382 | 0.1157 | +0.0004 | +0.0021 |
| MMOE | 0.7382 | 0.1175 | +0.0003 | +0.0004 |
| ML-MMOE | 0.7378 | 0.1169 | -0.0001 | +0.0010 |
| CGC | 0.7398 | 0.1155 | +0.0020 | +0.0023 |
| PLE | 0.7406 | 0.1150 | +0.0027 | +0.0029 |

#### 5.1.5 正常相关性任务的评估

尽管CGC和PLE在具有复杂相关性的任务上表现良好，我们进一步验证了它们在具有正常相关性模式的通用CTR/VCR任务组上的泛化能力。由于CTR和VCR旨在建模不同的用户行为，它们之间的相关性更简单。如表2所示，除十字绣外，所有模型在两个任务上均表现出正MTL增益，这表明CTR和VCR之间的相关性模式并不复杂，没有遭受跷跷板现象的影响。在此场景下，CGC和PLE仍然在两个任务上显著优于所有最先进模型，具有突出的MTL增益，这验证了CGC和PLE的收益是通用的，能够在广泛的任务相关性情况下实现更好的共享学习效率并持续提供递增的性能改进——不仅适用于难以合作的复杂相关性任务，也适用于正常相关的任务。

表3: 在线A/B测试中相对于单任务模型的改进

| 模型 | 总观看次数 | 总观看时长 |
|---|---|---|
| 硬参数共享 | -1.65% | -1.79% |
| 水闸网络 | +0.75% | +1.29% |
| MMOE | +1.94% | +1.73% |
| ML-MMOE | +1.96% | +1.10% |
| CGC | +3.92% | +2.75% |
| PLE | +4.17% | +3.57% |

#### 5.1.6 在线A/B测试

在视频推荐系统中使用VTR和VCR任务组进行了为期4周的精心设计的在线A/B测试。我们在基于C++的深度学习框架中实现了所有MTL模型，将用户随机分配到多个桶中，并将每个模型部署到一个桶中。最终排序分数通过第3节中描述的多个预测分数的组合函数获得。表3显示了MTL模型相对于单任务模型在在线指标（每用户总观看次数和每用户总观看时长，即系统的最终目标）上的改进。结果表明，CGC和PLE在所有在线指标上均比所有基线模型实现了显著增长。此外，PLE在所有在线指标上均显著优于CGC，这表明MTL中AUC或MSE的微小改进可以在在线指标中产生显著的改进。PLE自那时起已部署到腾讯平台。

表4: CGC和PLE在多任务上的MTL增益

| 任务组 | 模型 | MTL增益 (VTR) | MTL增益 (VCR) | MTL增益 (SHR) | MTL增益 (CMR) |
|---|---|---|---|---|---|
| VTR+VCR+SHR | CGC | +0.0131 | +0.0019 | +0.0012 | - |
| VTR+VCR+SHR | PLE | +0.0132 | +0.0036 | +0.0033 | - |
| VTR+VCR+CMR | CGC | +0.0180 | +0.0016 | - | -0.0001 |
| VTR+VCR+CMR | PLE | +0.0197 | +0.0017 | - | +0.0013 |
| VTR+VCR+SHR+CMR | CGC | +0.0097 | +0.0000 | +0.0008 | +0.0012 |
| VTR+VCR+SHR+CMR | PLE | +0.0128 | +0.0001 | +0.0058 | +0.0080 |

#### 5.1.7 多任务评估

最后，我们探索了CGC和PLE在更具挑战性的多任务场景中的可扩展性。除VTR和VCR之外，我们引入SHR（分享率）和CMR（评论率）来建模用户反馈行为。将CGC和PLE扩展到多任务情况很灵活，只需为每个任务添加一个任务特定专家模块、门控网络和塔网络即可。如表4所示，CGC和PLE在几乎所有任务组的所有任务上均取得了比单任务模型显著的改进。这表明CGC和PLE在多于两个任务的通用情况下仍然展现出促进任务合作、防止负迁移和跷跷板现象的优势。PLE在所有情况下均显著优于CGC。因此，PLE在不同大小的任务组上展现出更强的提高共享学习效率的优势。

### 5.2 在公开数据集上的评估

在本小节中，我们在公开基准数据集上进行实验，以进一步评估PLE在推荐之外场景中的有效性。

#### 5.2.1 数据集

• **合成数据**：按照基于[13]的数据合成过程生成，以控制任务相关性。由于[13]中未提供数据合成的超参数，我们按照标准正态分布随机采样α_i和β_i，并设置c=1、m=10、d=512以确保可复现性。为每种相关性生成了140万条具有两个连续标签的样本。

• **Census-income数据集**[5]：包含从1994年人口普查数据库中提取的299,285条样本和40个特征。为与基线模型进行公平比较，我们考虑与[13]相同的任务组。具体而言，任务1旨在预测收入是否超过50K，任务2旨在预测此人的婚姻状况是否为未婚。

• **Ali-CCP数据集**¹：这是一个公开数据集，包含从淘宝推荐系统提取的8400万条样本。CTR和CVR（转化率）是数据集中建模点击和购买行为的两个任务。

#### 5.2.2 实验设置

Census-income数据集的设置与[13]相同。对于合成数据和Ali-CCP数据集，我们为MTL模型和单任务模型中的每个任务采用三层MLP网络，激活函数为RELU，隐藏层大小为[256, 128, 64]。超参数的调整方式与第5.1节中的实验类似。

#### 5.2.3 实验结果

合成数据的实验结果（图7）表明，硬参数共享和MMOE有时会遭受跷跷板现象并在两个任务之间失去平衡。相反，PLE在不同相关性下始终为两个任务取得最佳性能，并且相对于MMOE平均实现87.2%的MTL增益提升。如表5所示的Ali-CCP和census-income数据集的结果，PLE消除了跷跷板现象，并在两个任务上始终优于单任务模型和MMOE。

结合之前在工业数据集和在线A/B测试上的实验，PLE在改善不同任务相关性模式和不同应用的MTL效率和性能方面展现出稳定的通用优势。

表5: Census-income和Ali-CCP数据集的实验结果

| 模型 | Census-income 任务1 AUC | Census-income 任务1 MTL增益 | Census-income 任务2 AUC | Census-income 任务2 MTL增益 | Ali-CCP CTR AUC | Ali-CCP CTR MTL增益 | Ali-CCP CVR AUC | Ali-CCP CVR MTL增益 |
|---|---|---|---|---|---|---|---|---|
| 单任务 | 0.9445 | - | 0.9923 | - | 0.6088 | - | 0.6040 | - |
| MMOE | 0.9393 | +0.0048 | 0.9928 | +0.0005 | 0.6094 | +0.0006 | 0.5738 | -0.0302 |
| PLE | 0.9522 | +0.0078 | 0.9945 | +0.0022 | 0.6112 | +0.0024 | 0.6097 | +0.0057 |

### 5.3 专家利用率分析

为了揭示不同门控如何聚合专家，我们研究了工业数据集VTR/VCR任务组中所有基于门控模型的专家利用率。为简单和公平比较，我们将每个专家视为单层网络，在CGC和PLE的每个专家模块中只保留一个专家，而在MMOE和ML-MMOE的每一层中保留三个专家。图8显示了所有测试数据中每个门控所利用的专家权重分布，其中条形高度表示权重的均值，垂直短线表示标准差。结果表明，在CGC中，VTR和VCR以显著不同的权重组合专家，而在MMOE中权重则相似得多，这表明CGC精心设计的结构有助于实现不同专家之间更好的区分。此外，在MMOE和ML-MMOE中，所有专家都没有零权重，这进一步表明尽管存在理论可能性，MMOE和ML-MMOE在实践中很难在没有先验知识的情况下收敛到CGC和PLE的结构。与CGC相比，PLE中的共享专家对塔网络输入的影响更大，特别是对于VTR任务。PLE性能优于CGC的事实表明共享的更高层深层表示的价值。换句话说，某些深层语义表示需要在任务之间共享，因此渐进式分离路由提供了一种更好的联合路由和学习方案。

## 6 结论

在本文中，我们提出了一种新颖的MTL模型，称为渐进式分层提取（PLE），该模型明确分离任务共享和任务特定参数，并引入了一种创新的渐进式路由方式，以避免负迁移和跷跷板现象，并实现更高效的信息共享和联合表示学习。在工业数据集和公开基准数据集上的离线和在线实验结果显示了PLE相对于最先进的MTL模型显著且一致的改进。探索层次化任务组的相关性将是未来工作的重点。

## 参考文献

[1] Trapit Bansal, David Belanger, and Andrew McCallum. 2016. Ask the gru: Multi-task learning for deep text recommendations. In Proceedings of the 10th ACM Conference on Recommender Systems. 107–114.

[2] Rich Caruana. 1997. Multitask learning. Machine learning 28, 1 (1997), 41–75.

[3] Zhongxia Chen, Xiting Wang, Xing Xie, Tong Wu, Guoqing Bu, Yining Wang, and Enhong Chen. 2019. Co-attentive multi-task learning for explainable recommendation. In Proceedings of the 28th International Joint Conference on Artificial Intelligence. AAAI Press, 2137–2143.

[4] Heng-Tze Cheng, Levent Koc, Jeremiah Harmsen, Tal Shaked, Tushar Chandra, Hrishi Aradhye, Glen Anderson, Greg Corrado, Wei Chai, Mustafa Ispir, et al. 2016. Wide & deep learning for recommender systems. In Proceedings of the 1st workshop on deep learning for recommender systems. 7–10.

[5] Dheeru Dua and Casey Graff. 2017. UCI Machine Learning Repository. http://archive.ics.uci.edu/ml

[6] Huifeng Guo, Ruiming Tang, Yunming Ye, Zhenguo Li, and Xiuqiang He. 2017. DeepFM: a factorization-machine based neural network for CTR prediction. arXiv preprint arXiv:1703.04247 (2017).

[7] Guy Hadash, Oren Sar Shalom, and Rita Osadchy. 2018. Rank and rate: multi-task learning for recommender systems. In Proceedings of the 12th ACM Conference on Recommender Systems. 451–454.

[8] Robert A Jacobs, Michael I Jordan, Steven J Nowlan, and Geoffrey E Hinton. 1991. Adaptive mixtures of local experts. Neural computation 3, 1 (1991), 79–87.

[9] Alex Kendall, Yarin Gal, and Roberto Cipolla. 2018. Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. In Proceedings of the IEEE conference on computer vision and pattern recognition. 7482–7491.

[10] Shikun Liu, Edward Johns, and Andrew J Davison. 2019. End-to-end multi-task learning with attention. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 1871–1880.

[11] Yichao Lu, Ruihai Dong, and Barry Smyth. 2018. Why I like it: multi-task learning for recommendation and explanation. In Proceedings of the 12th ACM Conference on Recommender Systems. 4–12.

[12] Jiaqi Ma, Zhe Zhao, Jilin Chen, Ang Li, Lichan Hong, and Ed H Chi. 2019. SNR: Sub-Network Routing for Flexible Parameter Sharing in Multi-task Learning. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 33. 216–223.

[13] Jiaqi Ma, Zhe Zhao, Xinyang Yi, Jilin Chen, Lichan Hong, and Ed H Chi. 2018. Modeling task relationships in multi-task learning with multi-gate mixture-of-experts. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 1930–1939.

[14] Xiao Ma, Liqin Zhao, Guan Huang, Zhi Wang, Zelin Hu, Xiaoqiang Zhu, and Kun Gai. 2018. Entire space multi-task model: An effective approach for estimating post-click conversion rate. In The 41st International ACM SIGIR Conference on Research & Development in Information Retrieval. 1137–1140.

[15] Krzysztof Maziarz, Efi Kokiopoulou, Andrea Gesmundo, Luciano Sbaiz, Gabor Bartok, and Jesse Berent. 2019. Gumbel-Matrix Routing for Flexible Multi-task Learning. arXiv preprint arXiv:1910.04915 (2019).

[16] Ishan Misra, Abhinav Shrivastava, Abhinav Gupta, and Martial Hebert. 2016. Cross-stitch networks for multi-task learning. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 3994–4003.

[17] Clemens Rosenbaum, Tim Klinger, and Matthew Riemer. 2017. Routing networks: Adaptive selection of non-linear functions for multi-task learning. arXiv preprint arXiv:1711.01239 (2017).

[18] Sebastian Ruder12, Joachim Bingel, Isabelle Augenstein, and Anders Søgaard. 2017. Sluice networks: Learning what to share between loosely related tasks. stat 1050 (2017), 23.

[19] Yoav Shoham, Rob Powers, and Trond Grenager. 2003. Multi-agent reinforcement learning: a critical survey. Web manuscript (2003).

[20] Yi Tay, Anh Tuan Luu, and Siu Cheung Hui. 2018. Multi-pointer co-attention networks for recommendation. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 2309–2318.

[21] Lisa Torrey and Jude Shavlik. 2010. Transfer learning. In Handbook of research on machine learning applications and trends: algorithms, methods, and techniques. IGI Global, 242–264.

[22] Jialei Wang, Steven CH Hoi, Peilin Zhao, and Zhi-Yong Liu. 2013. Online multi-task collaborative filtering for on-the-fly recommender systems. In Proceedings of the 7th ACM conference on Recommender systems. 237–244.

[23] Nan Wang, Hongning Wang, Yiling Jia, and Yue Yin. 2018. Explainable recommendation via multi-task learning in opinionated text data. In The 41st International ACM SIGIR Conference on Research & Development in Information Retrieval. 165–174.

[24] Jiejie Zhao, Bowen Du, Leilei Sun, Fuzhen Zhuang, Weifeng Lv, and Hui Xiong. 2019. Multiple Relational Attention Network for Multi-task Learning. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 1123–1131.

[25] Zhe Zhao, Lichan Hong, Li Wei, Jilin Chen, Aniruddh Nath, Shawn Andrews, Aditee Kumthekar, Maheswaran Sathiamoorthy, Xinyang Yi, and Ed Chi. 2019. Recommending what video to watch next: a multitask ranking system. In Proceedings of the 13th ACM Conference on Recommender Systems. 43–51.

[26] Barret Zoph and Quoc V Le. 2016. Neural architecture search with reinforcement learning. arXiv preprint arXiv:1611.01578 (2016).
