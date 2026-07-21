# 面向电子商务推荐的大规模商品图构建

> Xiaoyong Yang, Yudong Zhang, Yongshu Wang, Yuanquan Yang | Alibaba Group & UC Santa Cruz

本文介绍了 面向电子商务推荐的大规模商品图构建。核心内容：


关键发现：

---


杨晓勇† 朱亚东†∗ 张毅‡ 王小波† 袁泉†
† 阿里巴巴集团，北京，中国
‡ 加州大学圣克鲁兹分校，美国
{xiaoyong.yxy, edgewind.zyd, yongshu.wxb, yuanquan.yq}@alibaba-inc.com
{yiz}@soe.ucsc.edu


---

## 摘要

构建一个服务数十亿日常用户的推荐系统是一个具有挑战性的问题，因为系统需要基于实时用户行为以 O(1) 的时间复杂度每秒做出天文数字级别的预测。这类大规模推荐系统通常严重依赖预构建的商品索引来加速推荐服务，使得在线用户的等待时间几乎不可感知。其中一种重要的索引结构是商品-商品索引，给定一个种子商品，可以检索到一个排序后的商品列表。该索引可以看作是一个带权的商品-商品图。在本文中，我们提出了高效构建此类索引商品图的新技术。特别地，我们提出了 Swing 算法来捕捉商品之间的替代关系，该算法能够利用用户-item点击二部图的子结构。然后我们提出了 Surprise 算法对互补商品关系进行建模，该算法利用商品类别信息并通过聚类技术解决了用户共同购买图的稀疏性问题。基于这两种方法，我们可以为淘宝推荐构建基础商品图。我们通过离线实验和在线实验对这两种方法进行了全面评估，结果证明了该工作的有效性和高效性。

### 分类和主题描述
H.2.8 [数据库管理]：数据库应用–数据挖掘

### 关键词
商品图，电子商务，推荐系统

---
本文介绍了 面向电子商务推荐的大规模商品图构建。核心内容：


关键发现：



## 1. 引言

*通讯作者

© 2020 ACM. ISBN 123-4567-24-567/08/06. . . $15.00$
DOI: 10.475/123 4

随着数十亿用户利用电子商务网站来满足购物需求，同时节省时间并避免拥挤，如何更好地服务在线客户正成为一个日益重要的问题。同时，像淘宝和亚马逊这样的大型电子商务网站通常提供数十亿种商品供消费者选择。推荐系统可以帮助消费者选择合适的商品以满足其品味和特殊需求，在现代电子商务中扮演着关键角色[10]。现有的推荐方法主要分为两类：协同过滤方法（CF）[14, 24]和基于内容的方法（CB）[15, 17, 18]。还存在其他方法，如结合两者的混合方法，或考虑用户当前上下文信息以使推荐更合理的上下文感知方法[2, 1]。

除了具体的推荐方法之外，另一个重要的问题是理解并捕捉商品之间的关系，这是现代电子商务推荐系统的基础[16]。我们可以将商品关系图视为一种重建的商品索引，给定一个种子商品，它可以返回一个排序后的商品列表。该索引可以显著加速推荐服务，使得在线用户的等待时间几乎不可感知。

商品之间存在两种非常重要的关系：替代关系和互补关系。替代商品是指那些可以互换的商品，如图1中展示的衬衫；而互补商品是指那些可能被额外购买的商品，如衬衫和裤子[16]。不同的上下文环境可能对推荐相关性有不同的要求或含义，因此需要不同的关系图来加速推荐。例如，在用户会话的不同阶段，用户的偏好可能有很大差异。如图1所示，当用户专注于衬衫且在会话中尚未购买任何衬衫时，用户可能更希望获得替代商品的推荐以供比较。一旦购买完成，替代商品的相关性就会降低，而一些互补商品（如裤子或夹克）会更具吸引力和相关性。

最近一些研究者认识到了替代品和互补品的重要性，[16]利用商品评论和描述的文本来通过监督方法推断商品之间的关系。然而，现有方法不适用于像淘宝这样的大型推荐系统，因为对于来自数百万卖家的数十亿商品来说，商品评论和描述的文本量巨大且噪声极大，因此基于文本的监督方法不具有成本效益。

另一方面，淘宝拥有来自数十亿客户的海量真实用户行为数据，这些数据相比文本数据是更强大、更可靠的信号来捕捉商品关系。因此，我们专注于直接利用用户行为数据，通过类似于基于相似度的协同过滤的无监督方法来构建商品图。

当我们尝试采用传统方法（如具有局部相似度的item-item协同过滤[23, 14]）时，面临以下挑战：

- **准确性**：传统的局部相似度计算未考虑用户行为图的任何内部结构，而已有研究表明这些内部结构对其他链接预测问题是有用的。因此预测准确性有限。
- **鲁棒性**：用户行为数据（即用户-item点击图）包含许多噪声、随意或偶然的点击，这可能影响预测的可靠性。
- **稀疏性**：尽管淘宝的用户行为数据量巨大，但用户购买的比例相对较小。用户共同购买数据非常稀疏，因此捕捉互补关系非常困难。
- **方向性**：互补关系是非对称的。我们还需要在共同购买图的基础上考虑关系的方向。
- **可扩展性**：传统方法的计算复杂度随客户和商品的数量增长。考虑到淘宝拥有数十亿用户和商品，可扩展性是一个主要挑战。

在本文中，我们提出了一种称为 Swing 的新算法，该算法能够利用用户-item行为图的内部结构——Swing（摆动）来构建替代商品图。Swing 是一种准局部结构，被设计为比传统 CF 方法中使用的单条边更加稳定。它提供了在用户-item二部图上更可靠的计算传播，并有助于减少噪声用户行为数据的影响。然后我们提出了一种称为 Surprise 的新算法来构建互补商品图。Surprise 算法通过利用商品的类别信息和基于 Swing 算法构建的商品聚类来解决用户共同购买图的稀疏性问题。此外，它考虑了共同购买商品的时间敏感性和时间顺序。这两种方法都使用常用的大规模分布式计算框架（如 Map-Reduce 或 Spark）实现了并行运行，因此可扩展性不成问题。基于此，我们可以在淘宝中构建商品替代图和商品互补图，从而为后续的推荐排序模块提供基础索引服务来生成候选商品。

为了评估我们方法的有效性，我们进行了广泛的离线数据实验和在线用户研究。在大型数据集上的离线实验结果表明，所提出的方法在精确率、召回率和 MAP 指标上显著优于现有经过良好调优的基线 CF 方法。在淘宝上使用真实电子商务用户进行的在线实验结果也表明，所提出的方法能够带来显著更高的 CTR（点击率）、CVR（转化率）和 PPM（千次展示收入）。本文还分析和展示了新方法在离线运行时间方面的效率。

本文的主要贡献如下：

1. 一种新的高效且有效的算法（即 Swing），它利用用户行为图的准局部结构信息，提供了更可靠的计算传播并消除了噪声数据的影响。基于 Swing，我们可以构建更可靠的商品间相似关系。
2. 一种新的高效且有效的算法（即 Surprise），它利用商品类别信息和聚类技术，解决了稀疏性问题，使得使用 Surprise 算法时的商品间互补关系更加可靠和合理。
3. 通过并行化对所提方法进行高效的大规模工业实现。
4. 通过全面的离线和在线实验以及详细分析，对所提方法的有效性和效率进行了实证验证。

本文的其余部分组织如下。我们在第 2 节介绍用于替代关系的 Swing 算法。然后我们在第 3 节描述用于互补关系的 Surprise 算法。第 4 节展示实验结果，第 5 节描述相关工作，第 6 节总结全文。

---

## 2. 用于替代关系的 Swing 算法

计算两个item之间的相似度是构建替代图任务的核心。有许多仅需局部信息的相似度度量方法，这些方法通常计算高效，可以应用于超大规模图。推荐中常用的方法如下所示[23, ?]。

**向量余弦相似度**

$$w_{i,j} = \frac{|U_i \cap U_j|}{\sqrt{|U_i| \times |U_j|}}$$

其中 $U_i$ 是点击过item $i$ 的用户集合。

**Jaccard 系数**

$$w_{i,j} = \frac{|U_i \cap U_j|}{|U_i \cup U_j|}$$

**基于相关的相似度**

$$w_{i,j} = \frac{\sum_{u \in U_i \cap U_j} (r_{u,i} - \bar{r_i})(r_{u,j} - \bar{r_j})}{\sqrt{\sum_{u \in U_i \cap U_j} (r_{u,i} - \bar{r_i})^2} \sqrt{\sum_{u \in U_i \cap U_j} (r_{u,j} - \bar{r_j})^2}}$$

其中 $r_{u,i}$ 是用户 $u$ 对item $i$ 的评分，$\bar{r_i}$ 是item $i$ 的平均评分。item $i$ 和item $j$ 之间的相似度通过计算 Pearson 系数来衡量。

大多数传统方法在计算邻域强度时关注item-用户-item路径，并通过基于item的归一化来惩罚热门item。这些方法没有利用用户行为图上的其他内部结构（如用户-item-用户），因此预测准确性有限。

例如，假设有五个用户（Alice、Bob、Chris、David 和 Eric），他们全部都在淘宝上寻找 T 恤。如图 2(a) 所示，每一行代表一个用户及其点击的item。Alice 正在为她的男朋友挑选 T 恤。Bob 想为自己买一件流行的 T 恤。Chris 喜欢一个名为 Mucunsanshe 的特殊品牌。David 是 Michael Jackson 的狂热粉丝，只点击了一件 Michael Jackson 的 T 恤。而 Eric 正在寻找红色的 T 恤。他们都点击了商品 h。点击信息可以总结为图 2(b)，其中每个大写字母代表一个人（对应上述五个用户，我们简写为 A、B、C、D、E），每个小写字母代表一个被点击的商品。为简单且不失一般性，我们只展示同一类别（T 恤）中的商品。用户可能会点击任何item。例如，购买 T 恤后，Alice 可能会浏览并点击裙子给自己，而 David 可能会点击耳机，因为他是个音乐爱好者。

在这种情况下，我们有 $|U_h| = 5$，$|U_p| = 40$，$|U_z| = 60$，$|U_q| = 4$，$|U_t| = 15$，等等。这意味着总共有 5 个用户点击了商品 h，15 个用户点击了商品 t，等等。如果使用余弦相似度对 h 的邻近item进行排序，结果将是 t > z > p > q。然而，我们可以看到 z 与 h 并不是非常相似。除了共同邻居的数量之外，它只取决于分母上每个item节点的度（即有多少用户点击了每个item），这可能受到许多因素的影响，具有很大的不确定性。注意 z 的 CF 得分可能大于 p 和 q，因为它不那么流行。

### 2.1 Swing 算法

由于我们的用户点击数据包含噪声，有许多偶然或随机的点击，我们需要一种节点邻近度度量，能够考虑更鲁棒的内部网络结构信息。稳定的网络结构已在链接预测领域得到研究，例如用户倾向于在社交网络中形成三角形[11, 13]。考虑电子商务中的二部网络，当只有一个用户同时点击了 h 和 i 时，更可能是一种巧合。然而，如果有两个用户都点击了 h 和 i，这种关系就强得多。因此，对于每个种子商品，考虑包含所有点击过该种子商品的用户以及这些用户点击过的所有商品的局部图，我们将局部图上的每个用户-item-用户网络结构称为一个 swing（摆动）。例如，除了种子商品 h 之外，D 和 E 都点击了商品 q，所以 [D, q, E] 是一个 swing。而 [A, z]、[C, y]、[E, o]、[E, x] 只是单条边，不属于任何 swing。

如果用户对 <u, v> 之间存在许多 swing，通常表明有广泛种类的商品可以满足他们的需求。而这些 swing 之间的关系不那么紧密。因此，我们通过每个用户对之间形成的 swing 总数来对每个 swing 进行加权。Swing 分数的定义如下。

$$s(i,j) = \sum_{u \in U_i \cap U_j} \sum_{v \in U_i \cap U_j} \frac{1}{\alpha + |I_u \cap I_v|} \tag{1}$$

其中 $U_i$ 是点击过item $i$ 的用户集合，$I_u$ 是用户 $u$ 点击过的item集合。$\alpha$ 是平滑系数。

考虑图 2 中的例子，并计算商品 h 与其他商品之间的相似度。不失一般性，令 $\alpha = 1$。[A, p, B] 是一个 swing，而在 [A, B] 上有 3 个 swing：[A, t, B]、[A, r, B] 和 [A, p, B]。因此，由 [A, B] 贡献的 p 的 swing 分数为 $\frac{1}{(1+3)}$。[B, p, C] 是一个 swing，且在 [B, C] 上只有一个 swing，所以由 [B, C] 贡献的 p 的 swing 分数为 $\frac{1}{(1+1)}$。类似地，p 也从 [A, C] 获得 $\frac{1}{(1+1)}$，因为 [A, p, C] 是一个 swing。p 的总 swing 分数为：

$$swing(h,p) = \frac{1}{4} + \frac{1}{2} + \frac{1}{2} = 1.25$$

类似地，我们有：

$$swing(h,q) = \frac{1}{2} + \frac{1}{2} + \frac{1}{2} = 1.5$$

$$swing(h,t) = swing(h,r) = \frac{1}{2} = 0.25$$

因此，我们有 q > p > r = t > others。商品 q 是一件 Mucunsanshe 品牌的红色 Michael Jackson T 恤，与商品 h 非常相似。使用 swing，我们能够将商品 q 排到第一位。同时，所有只有单条边连接的item都被严格地排到了底部。

对于每个 swing 结构，我们进一步添加类似于著名的 Adamic/Adar 算法[12]的用户加权因子来惩罚活跃用户。一个用户点击的item越多，获得的权重越小。最终的 swing 分数为：

$$s(i,j) = \sum_{u \in U_i \cap U_j} \sum_{v \in U_i \cap U_j} \frac{w_u \cdot w_v}{\alpha + |I_u \cap I_v|} \tag{2}$$

其中

$$w_u = \frac{1}{\sqrt{|I_u|}}, \quad w_v = \frac{1}{\sqrt{|I_v|}}$$

计算 swing 分数的详细过程在算法 1 中描述。

注意，在本文中，我们主要关注基于用户点击行为的相似度计算，其他因素（如时间衰减）也可以被纳入每种方法中。此外，在淘宝中，大量紧密相关的点击通常发生在同一个用户会话中，时间间隔的影响较小。

假设item总数为 $T$，item的平均节点度为 $N$，用户的平均节点度为 $M$，传统item-item相似度 CF 方法的时间复杂度为 $O(T \cdot N \cdot M)$。Swing 的时间复杂度为 $O(T \cdot N^2 \cdot M)$，由于考虑了用户行为数据上的内部网络结构，复杂度更高。

### 2.2 并行化实现框架

在淘宝推荐系统中，我们在自己的分布式平台——开放数据处理服务（Open Data Processing Service）¹ 上基于 Map-Reduce 编程框架开发了 Swing 的并行实现。详细流程如图 3 所示。

具体来说，对于原始输入的用户点击矩阵，每一行是特定用户点击的item列表。在 Mapper 阶段，对于一行中用户 $i$ 点击的每个item，我们通过 $u_i$ 进行邻域广播，并输出键值对 <$i_{i1}$, $u_i i_{i2} \ldots i_{in}$>。然后在 Reducer 阶段，我们收集点击了item $i$ 的用户集合 $U_i$，以及 $U_i$ 中每个用户点击的item集合。最后按照算法 1 所述计算 Swing，以计算与 $item_i$ 最相似的item。

¹https://www.aliyun.com/product/odps/

---

**算法 1：用于替代关系的 Swing 算法**

**输入：** 用户和item索引表 U, I，平滑系数 \alpha
**输出：** item $item_i$ 的替代item列表

1: for each $u \in U_i$ do
2:     $w_u = 1/\sqrt{|I_u|}$
3:     for each $v \in U_i \backslash \{u\}$ do
4:         $w_v = 1/\sqrt{|I_v|}$
5:         $k = |I_u \cap I_v|$
6:         for each $j \in I_u \cap I_v$ do
7:             $Swing[j] \mathrel{+}= w_u \cdot w_v \cdot \frac{1}{\alpha+k}$
8:         end for
9:     end for
10: end for
11: return $Swing_i = (Swing_i[1], \ldots, Swing_i[n])$

---

## 3. 用于互补关系的 Surprise 算法

在本节中，我们提出了一个全面的框架，通过挖掘用户购买数据来发现互补产品。当用户刚刚购买了一件 T 恤时，再推荐 T 恤就不合适了。相反，短裤和鞋子可能是更好的候选。如果用户刚刚购买了一部手机，展示手机壳和移动电源等配件更为合理，因为用户不再对手机感兴趣。这种重要的商品-商品关系被称为互补关系，它将一个种子商品与用户可能额外或一起购买的相关商品联系起来。

为了发现互补关系，我们借鉴了[24]中提出的思想，使用购买数据来寻找候选商品。此外，我们需要考虑商品的时间敏感性[8, 26, 20, 25]和数据稀疏性问题，尤其是在淘宝场景中。首先，我们引入一个连续的时间衰减因子到行为相关性中，这已被证明能够提高准确性。然后，我们不使用存在于非常高维空间且在不同类别间差异巨大的内容信息，而是引入一种基于用户点击数据的聚类方法来解决数据稀疏性问题。注意，隐式反馈数据（如用户点击）实际上并不那么稀疏，尽管用户购买数据是稀疏的，因为用户通常会浏览和点击大量item，而实际购买的要少得多。我们将这个框架命名为 Surprise，旨在帮助用户发现不同类别中它们可能未曾意识到的紧密相关商品。

### 3.1 类别级相关性

为了找到与用户购买的种子商品最相关的产品，首先我们尝试在产品的分类体系上找到最相关的类别，这通常用于电子商务系统。这消除了来自用户购买行为的大部分无关情况。

通过将每个item映射到其类别，我们得到一个用户-类别矩阵。然后可以使用标准的协同过滤技术来计算类别之间的相关性。$c_j$ 是 $c_i$ 的相关类别的概率定义如下：

$$\theta_{i,j} = p(c_{i,j}|c_j) = \frac{N(c_{i,j})}{N(c_j)} \tag{3}$$

其中 $N(c_j)$ 是类别 $c_j$ 的总购买量，$N(c_{i,j})$ 是在 $c_i$ 之后购买 $c_j$ 的次数。

假设从公式 3 获得的 $c_i$ 的类别列表为 $[c_{j1}, c_{j2}, ..., c_{jm}]$，对应的后验概率列表为 $[\theta_{i,j1}, \theta_{i,j2}, ..., \theta_{i,jm}]$。粗略地取前一定百分比或固定数量的候选作为 $c_i$ 的相关类别是不准确的，因为类别之间差异很大。相反，我们为每个类别 $c_{j,k}$ 计算一个相对下降分数。

$$\eta_k = \frac{(\theta_{i,j_{k+1}} - \theta_{i,j_k})}{\theta_{i,j_k}} \tag{4}$$

排名在最大相对下降点之前的类别被选为 $c_i$ 的顶部相关类别，我们将这些类别记为 $\Gamma(c_i)$。

T 恤（男）和手机的相关类别概率分别如图 4(a) 和图 4(b) 所示。从这两个子图中，我们可以看到相关性分布中存在变化点。通过取最大相对下降点之前的类别，我们得到 T 恤（男）的 8 个顶部相关类别：休闲裤、牛仔裤、夹克、衬衫、低帮鞋、毛衣、卫衣、棉服，以及手机的 3 个紧密相关类别：手机壳、手机膜和移动电源。

### 3.2 商品级相关性

对于每个相关类别，我们计算该类别中商品的相关性分数，可应用标准的基于item的协同过滤技术。在我们的系统中，我们添加了一个约束条件，即候选相关商品 $j$ 应在商品 $i$ 之后被购买。顺序在建模互补关系中非常重要。例如，当用户刚买了一部手机时，推荐移动电源是合理的，但如果用户已经购买了移动电源，再推荐手机就不合适了。相关分数定义如下。

$$s_1(i,j) = \sum_{u \in U_i \cap U_j} \frac{1/(1 + |t_{ui} - t_{uj}|)}{\sqrt{|U_i|} \times \sqrt{|U_j|}} \tag{5}$$

在公式 5 中，其中 $c_j \in \Gamma(c_i)$ 且 $t_{uj} \geq t_{ui}$，我们在分子中添加了一个时间衰减因子。如果商品 $i$ 和商品 $j$ 的购买时间间隔较长，则它们之间存在强相关性的可能性较小。

### 3.3 聚类级相关性

假设 Bob 在 T 恤 $i$ 之后购买了牛仔裤 $j$，我们不能确定 $j$ 是否与 $i$ 很匹配。然而，如果有多个用户在 $i$ 之后购买了 $j$，我们将更有信心 $j$ 很可能与 $i$ 相关。我们在选择候选商品时还引入了 $(i, j)$ 共现次数的阈值。例如，当我们在商品 $i$ 被购买后做推荐时，我们只取 $Co(i, j) > \gamma$ 的候选商品，这可以看作是一种行为置信度。然而，这个约束会进一步带来数据稀疏性问题，因为用户共同购买数据非常稀疏。

考虑另一种情况，Bob 在 $i$ 之后购买了 $j$，David 在 $i$ 之后购买了 $k$，同时商品 $j$ 和商品 $k$ 非常相似，例如它们都是蓝色的 Levi's 牛仔裤。这种共同购买对于产品关系是有信息量的。这启发我们也在聚类级别计算商品的相关性分数，这有助于缓解商品级别共同购买的稀疏性问题。

我们尝试利用上一节中 Swing 算法构建的相似商品图，将相似的商品投影到同一个聚类中。然后我们在聚类级别计算共同购买相关性。

#### 3.3.1 使用标签传播进行聚类

传统的聚类方法（如 k-means 和基于密度的方法）在淘宝场景中不可行，因为数十亿产品存在可扩展性问题。受[19]中提出的社区检测方法的启发，我们在之前创建的替代图上执行类似的标签传播过程。

对于每个商品，由 Swing 计算出的顶部相似商品作为其邻居添加，带有从相似商品指向种子商品的有向链接。边 $e_{j,i}$ 的权重设置为相似度分数 $swing_{ij}$。注意这个权重不一定是对称的。初始时，每个商品节点的标签设置为其节点 ID。$p(\cdot)$ 表示邻居节点所属对应标签的当前概率。对于每个商品节点，在更新标签概率时考虑其所有邻居的边权重，然后基于一定的概率（即 $random() > \beta$）为该商品节点选择具有最大概率值的标签。

我们使用阿里云平台²上的大规模图计算框架实现该算法，约十次迭代后收敛。最终具有相同标签的商品被归入同一个聚类。该算法被证明是一种非常快速且有效的聚类方法，仅需 15 分钟即可聚类数十亿商品，详细算法在算法 2 中给出。

²www.aliyun.com

---

**算法 2：使用标签传播进行聚类**

**输入：** 相似度图 $G(V, E)$，item节点 $x$ 及其邻居 $\Gamma(x)$，阻尼因子 $\beta=0.25$
**输出：** 唯一标签 $L(x)$

1: init $L(x) = x$
2: for $t = 1, ..., n$ do
3:     for each $x \in V$ do
4:         init $p[L[y]] = 0, y \in V$
5:         for each $y \in \Gamma(x)$ do
6:             $p[L[y]] \mathrel{+}= e_{y,x}$
7:         end for
8:         if $random() > \beta$ then
9:             Set $L(x) = k$, where $p[k] = \max(p[1:m])$
10:        end if
11:     end for
12: end for
13: return $L$

---

#### 3.3.2 聚类级相关性

将商品聚类到不同组后，我们计算聚类级别的相关性分数。令 $L(i)$ 为商品 $i$ 所属的聚类。则

$$s_2(i,j) = s_1(L(i), L(j)) \tag{6}$$

其中 $s_1$ 按之前描述的公式 5 计算。也就是说，我们计算商品聚类 $L(j)$ 的购买发生在商品聚类 $L(i)$ 之后的相关性分数。

### 3.4 计算 Surprise 分数

基于两个相关性分数 $s_1(i,j)$ 和 $s_2(i,j)$，我们通过线性组合计算最终的相关分数：

$$s(i,j) = \omega \cdot s_1(i,j) + (1 - \omega) \cdot s_2(i,j) \tag{7}$$

其中 $\omega$ 是组合权重，可以手动设置或从数据中估计。除非另有说明，我们实验中的默认值为 $\omega = 0.8$。

---

## 4. 实验

我们将对提出的新方法进行实证评估。我们首先介绍实验设置，然后报告不同指标下的离线评估结果。该系统还部署在淘宝网上，与在线用户一起验证结果。最后，基于运行时间进行效率分析。

### 4.1 实验设置

我们从淘宝收集了一个用户行为数据集，包含 2015 年 12 月 16 日至 12 月 30 日期间超过 4 亿用户和 5 亿商品。用户点击数据用于捕捉替代关系，用户购买数据用于捕捉互补关系。

我们采用经典的基于item的 CF 与余弦相似度作为主要基线方法，因为它非常适合电子商务场景[14]。其他相似度度量也被评估，其中余弦相似度在现有选择中表现最佳。

为了公平比较，我们还向原始 CF 中引入了一个用户加权因子来惩罚活跃用户，这与 Swing 算法中使用的方式类似。这一改进进一步提升了 CF 的性能，并在本文中作为更强的基线使用。因此，最终的 CF 基线方法定义如下。

其中 $w_u = 1/\sqrt{|I_u|}$，$w_v = 1/\sqrt{|I_v|}$，$|I_u|$ 是用户点击/购买的商品数量。

### 4.2 离线评估

我们提出了一种离线策略来评估不同技术的性能。基本思想是看从历史数据中挖掘出的关系在多大程度上命中（即匹配）未来的真实用户行为序列。例如，如图 5 所示，顶行是一个真实的用户点击序列。我们随机选择序列中的一个中间商品，如时间 T2 处的 A2，作为替代/相似商品推荐的种子商品。底行包含给定商品 A2 后推荐的相似商品。在这种情况下，推荐命中了 A3、A4 和 A5，命中数为 3。命中数越大，推荐质量越高。

给定一组针对用户 $i$ 的某种类型（点击或购买）的商品推荐 $predict_i$，以及一组已知的真实数据 $truth_i$，我们可以使用传统的评估指标进行离线评估，包括精确率（Precision）、召回率（Recall）和 MAP [3]。这些指标的定义如下：

$$precision = \frac{1}{n} \sum_{i=1}^n \frac{|hit_i|}{|predict_i|}$$

$$recall = \frac{1}{n} \sum_{i=1}^n \frac{|hit_i|}{|truth_i|}$$

$$MAP = \frac{1}{n} \sum_{i=1}^n \sum_{k=1}^m precision_i@k$$

其中 $hit_i = predict_i \cap truth_i$，$predict_i$ 表示对第 $i$ 个用户的算法预测序列，$truth_i$ 表示真实行为（点击/购买）序列。$precision_i@k$ 表示在位置 $k$ 处截断的 $predict_i$ 的精确率。我们使用 12 月 16 日至 30 日的数据来挖掘商品关系，然后使用 12 月 31 日生成的真实用户行为序列作为 ground-truth³。Swing 的评估结果显示在表 1 中。结果表明 Swing 算法显著优于经典的基于item的 CF 方法。特别地，Swing 在精确率和召回率上相对于 CF 的绝对提升分别高达 67.6% 和 46.1%。这表明我们的方法能够以更高的精确率找到更多相关的替代商品。此外，我们的方法在 MAP 指标上也以数量级优于 CF，这表明 Swing 将相关商品排序在推荐列表中更靠前的位置。

**表 1：Swing 的离线评估**

| 指标 | CF | Swing（提升） |
|------|-----|----------------|
| Precision | 0.01471 | 0.02466 (+67.6%) |
| Recall | 0.1093 | 0.1597 (+46.1%) |
| MAP | 0.01177 | 0.06109 (+419%) |

**表 2：Surprise 的离线评估**

| 指标 | CF | Surprise（提升） |
|------|-----|------------------|
| Precision | 0.01188 | 0.02519 (+111.9%) |
| Recall | 0.1231 | 0.23875 (+93.9%) |
| MAP | 0.06242 | 0.109558 (+75.5%) |

Surprise 的评估结果显示在表 2 中，我们有类似的发现。

### 4.3 在线评估

我们使用真实电子商务用户进行在线实验，以研究所提方法的效果。我们基于淘宝移动 APP 中的推荐场景进行 A/B 测试。

我们使用的在线指标是点击率（CTR）、点击转化率（CVR）和千次展示收入（PPM），这些指标在电子商务系统中被广泛使用。定义如下：

$$CTR = \frac{\#item\_click}{\#show\_pv}$$

$$CVR = \frac{\#item\_trade}{\#item\_click}$$

$$PPM = \frac{\#Payment}{\#show\_pv} \times 1000$$

#### 4.3.1 Swing 在线实验

我们在购买前场景中使用 Swing 算法和基线方法进行了 A/B 测试，结果如图 6 所示。我们发现 Swing 组在所有三个指标上均显著优于（p-value < 0.05）CF 组。具体来说，Swing 在 CTR 和 CVR 上相对于 CF 的平均相对提升分别高达 9.3% 和 17.6%。这意味着用户使用 Swing 发现了更感兴趣的商品，并且将点击行为转化为实际购买的概率显著提高。Swing 在 PPM 上平均提升了 20.3%。PPM 是电子商务推荐的关键指标，因此这一提升具有巨大的商业价值。

#### 4.3.2 Surprise 在线实验

Surprise 算法部署在购买后场景中，评估结果显示在图 7 中。对于 Surprise 方法，我们还测试了另一个没有聚类级相关性分数的版本，即 $s(i,j) = s_1(i,j)$，表示为 Surprise-NCR。

从图中我们可以发现，Surprise 和 Surprise-NCR 在所有评估指标上都比原始的基于购买的 CF 有显著提升，这表明了顶部相关类别和基于时间的加权机制的重要性。同时，与 Surprise-NCR 相比，Surprise 平均实现了 27.9% 的 CTR 提升，这主要归功于更好的相关性估计及其生成更多商品推荐的能力。在 CVR 指标上，Surprise-NCR 与 Surprise 表现非常相似。我们的假设是，一旦用户点击了某个感兴趣的商品，购买意愿在很大程度上取决于商品本身的属性，而聚类级相关性分数并未捕捉到这一点。在 PPM 指标上，Surprise 分别比 Surprise-NCR 和原始的基于购买的 CF 平均高出 35% 和 183%。总之，Surprise 算法实现了最佳的商业效果。

#### 4.3.3 混合应用

我们还在一个集成场景中对由 Swing（SW）和 Surprise（SP）构建的商品图进行了整体评估。推荐列表通过组合用户历史点击的替代商品和用户当前订单的互补商品生成。详细的评估结果显示在图 8 中。我们提出的方法组合在 CTR、CVR 和 PPM 上分别比原始的 CF + CF 平均高出 33.2%、26.7% 和 62.9%。这些结果进一步证明了我们提出方法的有效性。

### 4.4 效率评估

我们进一步研究了所提方法的效率。如 2.1 节所述，由于考虑了用户点击图上的内部结构，Swing 的时间复杂度高于 CF。虽然两种方法都针对大规模应用开发了实用的并行化实现，但仍然值得比较所有方法的离线计算时间。结果如下（单位：小时）：

$$Click-CF (\sim 2.0h) \prec Swing (\sim 3.5h)$$

$$Purchase-CF (\sim 1.3h) \prec Surprise (\sim 2.5h)$$

Surprise 的计算包括三个部分：顶部相关类别的推断、商品级相关性计算和聚类级相关性计算。利用顶部相关类别，我们可以在 Mapper 阶段预过滤不相关的商品，大大降低计算成本。我们的方法需要更长的离线计算时间，但在可接受范围内，同时在线实验取得了显著的改进。

---

## 5. 相关工作

推荐系统可以帮助消费者选择合适的商品以满足其品味和特殊需求，在现代电子商务系统中扮演着关键角色。现有的大多数推荐方法可以分为两类：基于内容的方法和协同过滤方法。

基于内容的方法基于item特征进行推荐。已经提出了多种方法[17, 18, 15]，例如向量空间模型、贝叶斯分类器和聚类模型。然而，CB 方法在电子商务中的应用是有限的，因为内容特征（如商品标题词、品牌）难以描述商品所有重要的隐藏特征以及用户细粒度的偏好。因此，CB 方法通常作为混合推荐系统的一部分被使用[5, 24]。

协同过滤方法基于用户的历史行为进行推荐[14, 24]。这些方法通常在用户-item矩阵上操作，其中每一行是一个用户向量，每一列是一个item向量。基于用户的方法试图找到与当前用户相似的用户邻居[22, 4]。基于item的方法根据item向量找到与用户点击或购买的item相似的item邻居[23, 7, 6]。在这些方法中，已经提出了各种相似度度量来寻找 top-n 相似邻居，如余弦相似度、Jaccard 相似度、Pearson 相关系数和基于条件概率的相似度。与基于用户的方法相比，基于item的方法由于其有效性和高效性，在大规模电子商务系统中更常用。最近，诸如奇异值分解（SVD）[9]或概率矩阵分解（PMF）[21]等分解模型由于在一些基准数据集上的良好表现而受到广泛关注，这些方法还可以整合额外信息，如隐式用户反馈和时间信息。

本文并非提供单一的推荐算法或解决方案，而是主要关注发现商品之间的关系，这作为快速返回候选商品以供进一步计算密集的推荐算法（如混合过滤或 learning to rank）使用的基础。最近的一项工作[16]也关注同一问题。它利用商品评论和描述的文本，通过监督方法来推断商品之间的关系。与他们的工作相反，我们专注于直接利用用户行为数据（如用户共同点击和共同购买），这些数据对于捕捉商品关系是更强且更可靠的信号。

尽管我们构建的是item-item相似度矩阵，但基于标准局部相似度度量的传统基于item的 CF 方法[23, 14, ?]是"逐点"度量，忽略了用户行为数据的内部结构。另一方面，社交网络和图分析的现有工作已经表明内部结构对其他领域的预测是有用的。这些促使我们将内部结构引入推荐系统。

---

## 6. 结论

本文关注带有替代和互补关系的商品图。在电子商务中构建此类大规模商品图存在几个主要挑战：鲁棒性、数据噪声、数据稀疏性、关系方向性和可扩展性。为了应对这些挑战，我们提出了 Swing 算法，该算法能够利用用户行为数据的内部稳定结构来捕捉商品的替代关系。由于它消除了噪声信息并使预测的关系更加鲁棒，其性能显著更好。然后我们提出了一种先进的 Surprise 算法来建模互补关系。Surprise 利用商品类别信息，并通过标签传播的聚类技术解决了用户共同购买图中的稀疏性问题。此外，它还考虑了传播的时间敏感性和时间顺序，以确保互补关系是合理的。基于这两种方法，我们构建了两个商品图来支持淘宝中的快速推荐。最后，我们通过全面的离线和在线评估验证了我们方法的有效性和效率。

本文提出的商品-商品图构建方法非常通用，可以应用于其他应用，如电子商务中的广告和个性化搜索。这可以在未来进行探索。此外，我们将考虑更有效、更高效地将内容信息与所提出的工作结合起来。本文是将基于准局部用户-用户边的内部结构引入推荐系统的第一步。该方法在实践中简单、高效且极其有效。我们计划考虑不同的内部结构，并以其他方式（如马尔可夫随机场或贝叶斯网络）引入它们，并探索使这些更理论化的解决方案适应我们大规模系统的高效方法。

---

## 7. 参考文献

[1] G. Adomavicius, B. Mobasher, F. Ricci, and A. Tuzhilin. Context-aware recommender systems. AI Magazine, 32(3):67–80, 2011.

[2] G. Adomavicius and A. Tuzhilin. Context-aware recommender systems. In Proceedings of the 2008 ACM Conference on Recommender Systems, RecSys '08, pages 335–336, 2008.

[3] R. A. Baeza-Yates and B. Ribeiro-Neto. Modern Information Retrieval. Addison-Wesley Longman Publishing Co., Inc., Boston, MA, USA, 1999.

[4] J. S. Breese, D. Heckerman, and C. Kadie. Empirical analysis of predictive algorithms for collaborative filtering. UAI'98, pages 43–52, San Francisco, CA, USA, 1998. Morgan Kaufmann Publishers Inc.

[5] R. Burke. Hybrid recommender systems: Survey and experiments. User Modeling and User-Adapted Interaction, 12(4):331–370, Nov. 2002.

[6] M. Deshpande and G. Karypis. Item-based top-n recommendation algorithms. ACM Trans. Inf. Syst., 22(1):143–177, Jan. 2004.

[7] G. Karypis. Evaluation of item-based top-n recommendation algorithms. In Proceedings of the 10th CIKM, pages 247–254, New York, NY, USA, 2001. ACM.

[8] Y. Koren. Collaborative filtering with temporal dynamics. Commun. ACM, 53(4):89–97, Apr. 2010.

[9] Y. Koren, R. Bell, and C. Volinsky. Matrix factorization techniques for recommender systems. Computer, 42(8):30–37, Aug. 2009.

[10] Y. Koren and R. M. Bell. Advances in collaborative filtering. In Recommender Systems Handbook, pages 145–186. Springer, 2011.

[11] G. Kossinets and D. Watts. Empirical analysis of an evolving social network. Science, 311(5757):88–90, 2006.

[12] D. Liben-Nowell and J. Kleinberg. The link prediction problem for social networks. In Proceedings of the 12th CIKM, pages 556–559, New York, NY, USA, 2003. ACM.

[13] R. Lichtenwalter and N. V. Chawla. Vertex collocation profiles: subgraph counting for link analysis and prediction. In A. Mille, F. L. Gandon, J. Misselis, M. Rabinovich, and S. Staab, editors, WWW, pages 1019–1028. ACM, 2012.

[14] G. Linden, B. Smith, and J. York. Amazon.com recommendations: Item-to-item collaborative filtering. IEEE Internet Computing, 7(1):76–80, Jan. 2003.

[15] P. Lops, M. de Gemmis, and G. Semeraro. Content-based recommender systems: State of the art and trends. In F. Ricci, L. Rokach, B. Shapira, and P. B. Kantor, editors, Recommender Systems Handbook, pages 73–105. Springer US, 2011.

[16] J. McAuley, R. Pandey, and J. Leskovec. Inferring networks of substitutable and complementary products. In Proceedings of the 21th ACM SIGKDD, pages 785–794, New York, NY, USA, 2015. ACM.

[17] R. J. Mooney and L. Roy. Content-based book recommending using learning for text categorization. In Proceedings of the Fifth ACM Conference on Digital Libraries, DL '00, pages 195–204, New York, NY, USA, 2000. ACM.

[18] M. Pazzani and D. Billsus. Learning and revising user profiles: The identification of interesting web sites. Machine Learning, 27(3):313–331, 1997.

[19] U. N. Raghavan, R. Albert, and S. Kumara. Near linear time algorithm to detect community structures in large-scale networks. Phys. Rev. E, 76:036106, Sep 2007.

[20] S. Rendle, C. Freudenthaler, and L. Schmidt-Thieme. Factorizing personalized markov chains for next-basket recommendation. In Proceedings of the 19th WWW, pages 811–820, New York, NY, USA, 2010. ACM.

[21] R. Salakhutdinov and A. Mnih. Bayesian probabilistic matrix factorization using markov chain monte carlo. In Proceedings of the 25th ICML, pages 880–887, New York, NY, USA, 2008. ACM.

[22] B. Sarwar, G. Karypis, J. Konstan, and J. Riedl. Analysis of recommendation algorithms for e-commerce. In Proceedings of the 2Nd ACM Conference on Electronic Commerce, EC '00, pages 158–167, New York, NY, USA, 2000. ACM.

[23] B. Sarwar, G. Karypis, J. Konstan, and J. Riedl. Item-based collaborative filtering recommendation algorithms. In Proceedings of the 10th WWW, pages 285–295, New York, NY, USA, 2001. ACM.

[24] J. Wang, B. Sarwar, and N. Sundaresan. Utilizing related products for post-purchase recommendation in e-commerce. In Proceedings of the 5th RecSys, pages 329–332, New York, NY, USA, 2011. ACM.

[25] J. Wang and Y. Zhang. Opportunity model for e-commerce recommendation: Right product; right time. In Proceedings of the 36th International ACM SIGIR, pages 303–312, New York, NY, USA, 2013.

[26] L. Xiang, Q. Yuan, S. Zhao, L. Chen, X. Zhang, Q. Yang, and J. Sun. Temporal recommendation on graphs via long- and short-term preference fusion. In Proceedings of the 16th ACM SIGKDD, pages 723–732, New York, NY, USA, 2010. ACM.

## 参考文献
