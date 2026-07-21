# xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems（中文翻译）


本文介绍了 xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems。核心内容：


关键发现：

---


> Jianxun Lian, Xiaohuan Zhou, Fuzheng Zhang, Zhongxia Chen, Xing Xie, Guangzhong Sun | USTC, Beijing University of Posts and Telecommunications, Microsoft Research
> KDD 2018


---

## 摘要

组合特征对许多商业模型的成功至关重要。由于网络规模系统中原始数据的多样性、海量性和高速性，手动构建这些特征通常成本高昂。基于因子分解的模型通过向量内积衡量交互，能够自动学习组合特征的模式并泛化到未见特征。随着深度神经网络（DNN）在各个领域的巨大成功，最近研究者提出了几种基于DNN的因子分解模型来学习低阶和高阶特征交互。尽管DNN具有从数据中学习任意函数的强大能力，但普通的DNN以隐式方式和位级生成特征交互。在本文中，我们提出了一种新颖的压缩交互网络（CIN），其目标是显式地以向量级方式生成特征交互。我们展示了CIN与卷积神经网络（CNN）和循环神经网络（RNN）共享一些功能。我们进一步将CIN和经典DNN组合成一个统一模型，并将这个新模型命名为极深因子分解机（xDeepFM）。一方面，xDeepFM能够显式地学习特定有界度的特征交互；另一方面，它能够隐式地学习任意的低阶和高阶特征交互。我们在三个真实数据集上进行了全面实验。我们的结果表明xDeepFM优于最先进的模型。我们已在 https://github.com/Leavingseason/xDeepFM 上发布了xDeepFM的源代码。

**CCS概念**：• 信息系统 \rightarrow 个性化；• 计算方法 \rightarrow 神经网络；因子分解方法；

**关键词**：因子分解机，神经网络，推荐系统，深度学习，特征交互

**ACM引用格式**：
Jianxun Lian, Xiaohuan Zhou, Fuzheng Zhang, Zhongxia Chen, Xing Xie, and Guangzhong Sun. 2018. xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems. 见 KDD '18: 第24届ACM SIGKDD国际知识发现与数据挖掘大会, 2018年8月19–23日, 英国伦敦. ACM, 纽约, NY, USA, 10页. https://doi.org/10.1145/3219819.3220023

---
本文介绍了 xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems。核心内容：


关键发现：



## 1 引言

特征在许多预测系统的成功中起着核心作用。由于使用原始特征很难达到最优结果，数据科学家通常花费大量工作对原始特征进行变换，以生成最佳的预测系统[14, 24]或赢得数据挖掘比赛[21, 22, 26]。特征变换的主要类型之一是类别特征的交叉乘积变换[5]。这些特征被称为交叉特征或多路特征，它们衡量多个原始特征的交互。例如，一个三路特征 AND(user_organization=msra, item_category=deeplearning, time=monday) 当用户工作在微软亚洲研究院并在周一看到一篇关于深度学习的技术文章时，其值为1。

传统的交叉特征工程存在三个主要缺点。首先，获取高质量特征的成本很高。因为正确的特征通常是任务特定的，数据科学家需要花费大量时间从产品数据中探索潜在模式，然后才能成为领域专家并提取有意义的交叉特征。其次，在大规模预测系统（如网络规模的推荐系统）中，海量的原始特征使得手动提取所有交叉特征变得不可行。第三，手工制作的交叉特征不能泛化到训练数据中未见过的交互。因此，学习在没有手动工程的情况下进行特征交互是一项有意义的任务。

因子分解机（FM）[32]将每个特征i嵌入到一个潜在因子向量 vi = [vi1, vi2, ..., viD] 中，成对特征交互被建模为潜在向量的内积：f^(2)(i, j) = ⟨vi, vj⟩ xi xj。在本文中，我们使用术语"位"（bit）来表示潜在向量中的一个元素（如 vi1）。经典FM可以扩展到任意高阶特征交互[2]，但一个主要缺点是，[2]提出建模所有特征交互，包括有用和无用的组合。如[43]所示，与无用特征的交互可能会引入噪声并降低性能。近年来，深度神经网络（DNN）凭借其强大的特征表示学习能力，在计算机视觉、语音识别和自然语言处理领域取得了成功。利用DNN学习复杂且具有选择性的特征交互是很有前景的。[46]提出了因子分解机支持的神经网络（FNN）来学习高阶特征交互。它在应用DNN之前使用预训练的因子分解机进行域嵌入。[31]进一步提出了基于积分的神经网络（PNN），它在嵌入层和DNN层之间引入了一个乘积层，并且不依赖预训练的FM。FNN和PNN的主要缺点是它们更关注高阶特征交互，而捕获的低阶交互很少。Wide&Deep [5]和DeepFM [9]模型通过引入混合架构克服了这个问题，该架构包含一个浅层组件和一个深层组件，目的是同时学习记忆和泛化。因此，它们可以共同学习低阶和高阶特征交互。

上述所有模型都利用DNN来学习高阶特征交互。然而，DNN以隐式方式建模高阶特征交互。DNN学到的最终函数可以是任意的，并且没有理论结论表明特征交互的最大度数是多少。此外，DNN在位级上建模特征交互，这与传统的以向量级建模特征交互的FM框架不同。因此，在推荐系统领域，DNN是否是表示高阶特征交互最有效的模型仍然是一个悬而未决的问题。在本文中，我们提出了一种基于神经网络的模型，以显式、向量级的方式学习特征交互。我们的方法基于Deep & Cross Network（DCN）[40]，该网络旨在高效捕获有界度的特征交互。然而，我们将在第2.3节中论证DCN会导致一种特殊形式的交互。因此，我们设计了一种新颖的压缩交互网络（CIN）来替代DCN中的交叉网络。CIN显式地学习特征交互，并且交互的程度随着网络深度而增加。遵循Wide&Deep和DeepFM模型的精神，我们将显式高阶交互模块与隐式交互模块以及传统FM模块结合起来，并将联合模型命名为极深因子分解机（xDeepFM）。新模型不需要手动特征工程，将数据科学家从繁琐的特征搜索工作中解放出来。总结起来，我们做出以下贡献：

* 我们提出了一种新颖的模型，名为极深因子分解机（xDeepFM），它有效地联合学习显式和隐式高阶特征交互，并且不需要手动特征工程。
* 我们在xDeepFM中设计了一个压缩交互网络（CIN），该网络显式地学习高阶特征交互。我们展示了特征交互的程度在每一层增加，并且特征是在向量级而非位级进行交互的。
* 我们在三个真实数据集上进行了大量实验，结果表明我们的xDeepFM显著优于多个最先进的模型。

本文的其余部分组织如下。第2节提供了理解基于深度学习的推荐系统所需的一些预备知识。第3节详细介绍我们提出的CIN和xDeepFM模型。我们将在第4节中展示在多个数据集上的实验探索。相关工作在第5节讨论。第6节总结本文。

---

## 2 预备知识

### 2.1 嵌入层

在计算机视觉或自然语言理解中，输入数据通常是图像或文本信号，这些信号在空间和/或时间上具有相关性，因此DNN可以直接应用于具有密集结构的原始特征。然而，在网络规模的推荐系统中，输入特征是稀疏的、维度巨大，并且没有明显的空间或时间相关性。因此，多域类别形式被相关工作广泛使用[9, 31, 37, 40, 46]。例如，一个输入实例 [user_id=s02, gender=male, organization=msra, interests=comedy&rock] 通常通过域感知独热编码转换为高维稀疏特征：

[0, 1, 0, 0, ..., 0] \leftarrow user_id
[1, 0] \leftarrow gender
[0, 1, 0, 0, ..., 0] \leftarrow organization
[0, 1, 0, 1, ..., 0] \leftarrow interests

在原始特征输入之上应用嵌入层，将其压缩为低维密集实值向量。如果域是单值的，则使用特征嵌入作为域嵌入。以上述实例为例，特征"male"的嵌入被用作域"gender"的嵌入。如果域是多值的，则使用特征嵌入的和作为域嵌入。嵌入层如图1所示。嵌入层的结果是一个宽的拼接向量：

e = [e1, e2, ..., em]

其中m表示域的数量，ei \in RD表示一个域的嵌入。虽然实例的特征长度可能不同，但它们的嵌入具有相同的长度 m $\times$ D，其中D是域嵌入的维度。

**图1: 域嵌入层。本例中嵌入的维度为4。**

### 2.2 隐式高阶交互

FNN [46]、Deep Crossing [37]以及Wide&Deep [5]中的深度部分利用域嵌入向量e上的前馈神经网络来学习高阶特征交互。前向过程为：

x1 = \sigma(W(1)e + b1)
xk = \sigma(W(k)x(k-1) + bk)

其中k是层的深度，\sigma是激活函数，xk是第k层的输出。可视化结构与图2所示的非常相似，除了它们不包括FM或乘积层。这种架构以位级方式建模交互。也就是说，即使在同一域嵌入向量内的元素也会相互影响。

PNN [31]和DeepFM [9]对上述架构进行了略微修改。除了在嵌入向量e上应用DNN之外，它们在架构中增加了一个两路交互层。因此，他们的模型同时包含位级和向量级交互。PNN和DeepFM之间的主要区别在于，PNN将乘积层的输出连接到DNN，而DeepFM将FM层直接连接到输出单元（参见图2）。

**图2: DeepFM（省略线性部分）和PNN的架构。我们重用了[9]中的符号，其中红色边表示权重为1的连接（无参数），灰色边表示正常连接（网络参数）。**

### 2.3 显式高阶交互

[40]提出了交叉网络（CrossNet），其架构如图3所示。它旨在显式地建模高阶特征交互。与经典的全连接前馈网络不同，隐藏层通过以下交叉运算计算：

xk = x0 x_(k-1)^T wk + bk + x_(k-1)

其中 wk、bk、xk \in R^(mD) 分别是第k层的权重、偏置和输出。我们认为CrossNet学习了一种特殊类型的高阶特征交互，其中CrossNet中的每个隐藏层都是 x0 的标量倍数。

**定理2.1**。考虑一个k层交叉网络，其第(i+1)层定义为 xi+1 = x0 xi^T wi+1 + xi。那么，交叉网络的输出 xk 是 x0 的标量倍数。

**证明**。当k=1时，根据矩阵乘法的结合律和分配律，我们有：

x1 = x0(x0^T w1 + 1) = \alpha1 x0

其中标量 \alpha1 = x0^T w1 + 1 实际上是x0的线性回归。因此，x1是x0的标量倍数。假设标量倍数结论对k=i成立。对于k=i+1，我们有：

xi+1 = x0 xi^T wi+1 + xi = x0((\alphai x0)^T wi+1) + \alphai x0 = \alphai+1 x0

其中，\alphai+1 = \alphai (x0^T wi+1 + 1) 是一个标量。因此 xi+1 仍然是 x0 的标量倍数。通过归纳假设，交叉网络的输出 xk 是 x0 的标量倍数。□

注意，标量倍数并不意味着 xk 与 x0 是线性关系。系数 \alphai+1 对 x0 敏感。CrossNet可以非常高效地学习特征交互（其复杂度与DNN模型相比可以忽略不计），但其缺点是：(1) CrossNet的输出被限制在一种特殊形式中，每个隐藏层是 x0 的标量倍数；(2) 交互以位级方式发生。

---

## 3 我们提出的模型

### 3.1 压缩交互网络

我们设计了一种新的交叉网络，名为压缩交互网络（CIN），基于以下考虑：(1) 交互在向量级而非位级应用；(2) 高阶特征交互被显式地衡量；(3) 网络复杂度不会随交互度数指数增长。由于嵌入向量被视为向量级交互的单位，我们将域嵌入的输出形式化为一个矩阵 X0 \in R^(m$\times$D)，其中 X0 的第i行是第i个域的嵌入向量：X0_i,* = ei，D是域嵌入的维度。CIN中第k层的输出也是一个矩阵 Xk \in R^(Hk$\times$D)，其中 Hk 表示第k层中（嵌入）特征向量的数量，我们令 H0 = m。对于每一层，Xk 通过以下方式计算：

X_k^h,* = \Sigma_{i=1}^{H_(k-1)} \Sigma_{j=1}^{m} W_k^h,ij (X_(k-1)^i,* \circ X_0^j,*)

其中 1 \leq h \leq Hk，W_k^h \in R^(H_(k-1)$\times$m) 是第h个特征向量的参数矩阵，\circ 表示Hadamard积，例如 ⟨a1, a2, a3⟩ \circ ⟨b1, b2, b3⟩ = ⟨a1b1, a2b2, a3b3⟩。注意 Xk 是通过 X_(k-1) 和 X0 之间的交互推导得出的，因此特征交互被显式衡量，并且交互的程度随着层深度增加。CIN的结构与循环神经网络（RNN）非常相似，其中下一个隐藏层的输出依赖于上一个隐藏层和一个额外的输入。我们在所有层中保持嵌入向量的结构，因此交互是在向量级应用的。

有趣的是，方程6与计算机视觉中著名的卷积神经网络（CNN）有很强的联系。如图4a所示，我们引入了一个中间张量 Z^(k+1)，它是隐藏层 Xk 和原始特征矩阵 X0 的（沿每个嵌入维度的）外积。然后 Z^(k+1) 可以被视为一种特殊类型的图像，W_k^h 是一个滤波器。我们如图4b所示沿嵌入维度（D）在 Z^(k+1) 上滑动滤波器，得到一个隐藏向量 X_(k+1)^h,*，这在计算机视觉中通常称为特征图。因此，Xk 是 Hk 个不同特征图的集合。CIN名称中的"压缩"表明第k个隐藏层将 H_(k-1) $\times$ m 个向量的潜在空间压缩为 Hk 个向量。

图4c提供了CIN架构的概览。令T表示网络的深度。每个隐藏层 Xk, k \in [1,T] 都与输出单元有连接。我们首先对隐藏层的每个特征图应用求和池化：

p_i^k = \Sigma_{j=1}^{D} X_i,j^k, for i \in [1, Hk]

因此，对于第k个隐藏层，我们有一个池化向量 p^k = [p_1^k, p_2^k, ..., p_Hk^k]，长度为 Hk。来自隐藏层的所有池化向量在连接到输出单元之前被拼接起来：

p^+ = [p^1, p^2, ..., p^T] \in R^(\Sigma_{i=1}^T Hi)

如果我们直接使用CIN进行二分类，输出单元是p^+上的一个sigmoid节点：

ŷ = 1 / (1 + exp(p^+^T wo))

其中 wo 是回归参数。

**图4: 压缩交互网络（CIN）的组件和架构。(a) 沿每个维度外积用于特征交互。张量 Z^(k+1) 是用于进一步学习的中间结果。(b) CIN的第k层。它将中间张量 Z^(k+1) 压缩为 H_(k+1) 个嵌入向量（也称为特征图）。(c) CIN架构的概览。**

### 3.2 CIN分析

我们分析了提出的CIN，以研究模型复杂度和潜在有效性。

#### 3.2.1 空间复杂度

第k层的第h个特征图包含 H_(k-1) $\times$ m 个参数，这正是 W_k^h 的大小。因此，第k层有 Hk $\times$ H_(k-1) $\times$ m 个参数。考虑到输出单元的最后一个回归层，其参数为 \Sigma_{k=1}^T Hk，CIN的总参数数为 \Sigma_{k=1}^T Hk $\times$ (1 + H_(k-1) $\times$ m)。注意CIN与嵌入维度D无关。相比之下，一个普通的T层DNN包含 m $\times$ D $\times$ H1 + HT + \Sigma_{k=2}^T Hk $\times$ H_(k-1) 个参数，并且参数数量会随着嵌入维度D的增加而增加。

通常m和Hk不会很大，所以W_k^h的规模是可接受的。必要时，我们可以利用L阶分解，用两个较小的矩阵 U_k^h \in R^(H_(k-1)$\times$L) 和 V_k^h \in R^(m$\times$L) 替换 W_k^h：

W_k^h = U_k^h (V_k^h)^T

其中 L ≪ H 且 L ≪ m。为简单起见，以下我们假设每个隐藏层具有相同数量（H）的特征图。通过L阶分解，CIN的空间复杂度从 O(mT H^2) 降低到 O(mT HL + T H^2)。相比之下，普通DNN的空间复杂度为 O(mDH + T H^2)，这受域嵌入维度（D）的影响。

#### 3.2.2 时间复杂度

计算张量 Z^(k+1)（如图4a所示）的代价是 O(mHD) 时间。因为我们在一个隐藏层中有H个特征图，计算T层CIN需要 O(T H^2 DT) 时间。相比之下，T层普通DNN需要 O(mHD + H^2 T) 时间。因此，CIN的主要缺点在于时间复杂度。

#### 3.2.3 多项式逼近

接下来我们检验CIN的高阶交互特性。为简单起见，我们假设隐藏层的特征图数量都等于域的数量m。令[m]表示小于或等于m的正整数集合。第一层的第h个特征图，记作 x_h^1 \in R^D，通过下式计算：

x_h^1 = \Sigma_{i\in[m]} \Sigma_{j\in[m]} W_i,j^(1,h) (x_i^0 \circ x_j^0)

因此，第一层的每个特征图用 O(m^2) 个系数对成对交互进行建模。类似地，第二层的第h个特征图为：

x_h^2 = \Sigma_{i\in[m]} \Sigma_{j\in[m]} W_i,j^(2,h) (x_i^1 \circ x_j^0)
     = \Sigma_{i\in[m]} \Sigma_{j\in[m]} W_i,j^(2,h) \Sigma_{l\in[m]} \Sigma_{k\in[m]} W_l,k^(1,i) (x_l^0 \circ x_k^0 \circ x_j^0)

注意，所有与下标l和k相关的计算已经在前一个隐藏层完成。我们展开方程11中的因子只是为了清晰。我们可以观察到，第二层的每个特征图用 O(m^2) 个新参数对三路交互进行建模。

一个经典的k阶多项式有 O(m^k) 个系数。我们证明CIN通过一系列特征图链仅用 O(km^3) 个参数逼近这类多项式。通过归纳假设，我们可以证明第k层的第h个特征图是：

x_h^k = \Sigma_{i\in[m]} \Sigma_{j\in[m]} W_i,j^(k,h) (x_i^(k-1) \circ x_j^0)
     = \Sigma_{i\in[m]} \Sigma_{j\in[m]} W_i,j^(k,h) \Sigma_{l\in[m]} \Sigma_{s\in[m]} ... \Sigma_{r\in[m]} \Sigma_{t\in[m]} W_l,s^(k-1,i) ... W_r,t^(1,\alpha) (x_l^0 \circ ... \circ x_t^0 \circ x_r^0)
       \_____________________________/
             k个向量

为了更好地说明，这里借用[40]中的符号。令 \alpha = [\alpha1, ..., \alpham] \in N^d 表示一个多重索引，且 |\alpha| = \Sigma_{i=1}^m \alpha_i。我们省略 x_i^0 的原始上标，直接使用 xi 表示，因为我们最终展开表达式（参见方程12）中只使用第0层的特征图（即域嵌入）。现在上标用于表示向量操作，例如 x_i^3 = xi \circ xi \circ xi。令 VP_k(X) 表示一个k阶多向量多项式：

VP_k(X) = { \Sigma_\alpha w_\alpha x_1^\alpha1 \circ x_2^\alpha2 \circ ... \circ x_m^\alpham | 2 \leq |\alpha| \leq k }

该类中的每个向量多项式有 O(m^k) 个系数。然后，我们的CIN通过以下方式逼近系数 w_\alpha：

ŵ_\alpha = \Sigma_{i=1}^m \Sigma_{j=1}^m \Sigma_{B\inP_\alpha} \prod_{t=2}^{|\alpha|} W_i,B_t^(t,j)

其中，B = [B1, B2, ..., B_|\alpha|] 是一个多重索引，P_\alpha 是索引 (1..1, ..., m..m) 的所有排列的集合。

### 3.3 与隐式网络的结合

如第2.2节所述，普通DNN学习隐式高阶特征交互。由于CIN和普通DNN可以相互补充，使模型更强的直观方式是将这两种结构结合起来。得到的模型与Wide&Deep或DeepFM模型非常相似。架构如图5所示。我们将新模型命名为极深因子分解机（xDeepFM），考虑到一方面它包含低阶和高阶特征交互；另一方面，它包含隐式特征交互和显式特征交互。其输出单元变为：

ŷ = \sigma(w_linear^T a + w_dnn^T x_dnn^k + w_cin^T p^+ + b)

其中\sigma是sigmoid函数，a是原始特征，x_dnn^k和p^+分别是普通DNN和CIN的输出。w_*和b是可学习参数。对于二分类问题，损失函数是对数损失：

L = - 1/N \Sigma_{i=1}^N (yi log ŷi + (1 - yi) log(1 - ŷi))

其中N是训练实例的总数。优化过程是最小化以下目标函数：

J = L + \lambda*||$\Theta$||

其中\lambda*表示正则化项，$\Theta$表示参数集，包括线性部分、CIN部分和DNN部分中的参数。

**图5: xDeepFM的架构。**

#### 3.3.1 与FM和DeepFM的关系

假设所有域都是单值的。从图5不难观察到，当CIN部分的深度和特征图都设置为1时，xDeepFM通过为FM层学习线性回归权重（注意在DeepFM中，FM层的单元直接连接到输出单元而没有任何系数）成为DeepFM的泛化。当我们进一步移除DNN部分，同时使用一个常数和滤波器（即简单地取输入的和而不进行任何参数学习）用于特征图时，那么xDeepFM降级为传统的FM模型。

---

## 4 实验

在本节中，我们进行大量实验来回答以下问题：

* (Q1) 我们提出的CIN在高阶特征交互学习中的表现如何？
* (Q2) 在推荐系统中，结合显式和隐式高阶特征交互是否必要？
* (Q3) 网络设置如何影响xDeepFM的性能？

在介绍一些基本的实验设置后，我们将回答这些问题。

### 4.1 实验设置

#### 4.1.1 数据集

我们在以下三个数据集上评估我们提出的模型：

1. **Criteo数据集**。这是一个著名的行业基准数据集，用于开发预测广告点击率的模型，可公开访问¹。给定一个用户和他正在访问的页面，目标是预测他点击给定广告的概率。

2. **大众点评数据集**。Dianping.com是中国最大的消费者评论网站。它提供多种功能，如评论、签到和商店元信息（包括地理信息和商店属性）。我们收集了6个月的用户签到活动用于餐厅推荐实验。给定用户的画像、餐厅的属性以及用户最后三个访问过的POI（兴趣点），我们想要预测他将访问该餐厅的概率。对于用户签到实例中的每个餐厅，我们按POI流行度采样了四个3公里范围内的餐厅作为负例。

3. **必应新闻数据集**。Bing News²是微软必应搜索引擎的一部分。为了在真实商业数据集中评估我们模型的性能，我们收集了连续五天的新闻阅读服务曝光日志。我们使用前三天数据进行训练和验证，后两天数据进行测试。

对于Criteo数据集和大众点评数据集，我们按8:1:1的比例随机划分实例用于训练、验证和测试。三个数据集的特征总结在表1中。

**表1: 评估数据集的统计信息。M表示百万，K表示千。**

| 数据集 | #实例 | #域 | #特征（稀疏） |
|--------|-------|-----|--------------|
| Criteo | 45M   | 39  | 2.3M         |
| 大众点评 | 1.2M | 18  | 230K         |
| 必应新闻 | 5M   | 45  | 17K          |

#### 4.1.2 评估指标

我们使用两个指标进行模型评估：AUC（ROC曲线下面积）和Logloss（交叉熵）。这两个指标从两个不同的角度评估性能：AUC衡量正实例被排在随机选择的负实例之前的概率。它只考虑预测实例的顺序，并且对类别不平衡问题不敏感。相比之下，Logloss衡量每个实例的预测分数与真实标签之间的距离。有时我们更依赖Logloss，因为我们需要使用预测概率来估计排序策略的收益（通常调整为 CTR $\times$ bid）。

#### 4.1.3 基线模型

我们将xDeepFM与LR（逻辑回归）、FM、DNN（普通深度神经网络）、PNN（选择iPNN和oPNN中较好的）[31]、Wide & Deep [5]、DCN（Deep & Cross Network）[40]和DeepFM [9]进行比较。如第2节所述，这些模型与我们的xDeepFM高度相关，其中一些是推荐系统的最先进模型。注意本文的重点是自动学习特征交互，因此我们不包含任何手工制作的交叉特征。

#### 4.1.4 可复现性

我们使用Tensorflow³实现我们的方法。每个模型的超参数通过在验证集上进行网格搜索来调整，每个模型的最佳设置将在相应部分中展示。学习率设置为0.001。对于优化方法，我们使用Adam [16]，小批量大小为4096。我们对DNN、DCN、Wide&Deep、DeepFM和xDeepFM使用L2正则化，\lambda = 0.0001，对PNN使用dropout 0.5。每层神经元数量的默认设置是：(1) DNN层为400；(2) Criteo数据集上CIN层为200，大众点评和必应新闻数据集上CIN层为100。由于本文关注神经网络结构，我们使所有模型的域嵌入维度固定为10。我们使用5块Tesla K80 GPU并行进行不同设置的实验。源代码可在 https://github.com/Leavingseason/xDeepFM 获取。

**表2: 各单独模型在Criteo、大众点评和必应新闻数据集上的性能。Depth列表示每个模型的最佳网络深度。**

| 模型 | AUC | Logloss | Depth |
|------|-----|---------|-------|
| **Criteo** | | | |
| FM | 0.7900 | 0.4592 | - |
| DNN | 0.7993 | 0.4491 | 2 |
| CrossNet | 0.7961 | 0.4508 | 3 |
| CIN | 0.8012 | 0.4493 | 3 |
| **大众点评** | | | |
| FM | 0.8165 | 0.3558 | - |
| DNN | 0.8318 | 0.3382 | 3 |
| CrossNet | 0.8283 | 0.3404 | 2 |
| CIN | 0.8576 | 0.3225 | 2 |
| **必应新闻** | | | |
| FM | 0.8223 | 0.2779 | - |
| DNN | 0.8366 | 0.2730 | 2 |
| CrossNet | 0.8304 | 0.2765 | 6 |
| CIN | 0.8377 | 0.2662 | 5 |

**表3: 不同模型在Criteo、大众点评和必应新闻数据集上的总体性能。Depth列以(交叉层, DNN层)的格式呈现网络深度的最佳设置。**

| 模型 | AUC | Logloss | Depth | AUC | Logloss | Depth | AUC | Logloss | Depth |
|------|-----|---------|-------|-----|---------|-------|-----|---------|-------|
| | **Criteo** | | | **大众点评** | | | **必应新闻** | | |
| LR | 0.7577 | 0.4854 | -,- | 0.8018 | 0.3608 | -,- | 0.7988 | 0.2950 | -,- |
| FM | 0.7900 | 0.4592 | -,- | 0.8165 | 0.3558 | -,- | 0.8223 | 0.2779 | -,- |
| DNN | 0.7993 | 0.4491 | -,2 | 0.8318 | 0.3382 | -,3 | 0.8366 | 0.2730 | -,2 |
| DCN | 0.8026 | 0.4467 | 2,2 | 0.8391 | 0.3379 | 4,3 | 0.8379 | 0.2677 | 2,2 |
| Wide&Deep | 0.8000 | 0.4490 | -,3 | 0.8361 | 0.3364 | -,2 | 0.8377 | 0.2668 | -,2 |
| PNN | 0.8038 | 0.4927 | -,2 | 0.8445 | 0.3424 | -,3 | 0.8321 | 0.2775 | -,3 |
| DeepFM | 0.8025 | 0.4468 | -,2 | 0.8481 | 0.3333 | -,2 | 0.8376 | 0.2671 | -,3 |
| **xDeepFM** | **0.8052** | **0.4418** | 3,2 | **0.8639** | **0.3156** | 3,3 | **0.8400** | **0.2649** | 3,2 |

### 4.2 各神经网络组件的性能比较 (Q1)

我们想了解CIN单独的表现如何。注意FM显式地衡量二阶特征交互，DNN隐式地建模高阶特征交互，CrossNet试图用少量参数建模高阶特征交互（在第2.3节中被证明效果不佳），而CIN显式地建模高阶特征交互。没有理论保证一个单独模型优于其他模型，因为这实际上取决于数据集。例如，如果实际数据集不需要高阶特征交互，FM可能是最好的单独模型。因此，我们在这个实验中对哪个模型会表现最好没有任何预期。

表2显示了三个实际数据集中各单独模型的结果。令人惊讶的是，我们的CIN一致优于其他模型。一方面，结果表明对于实际数据集，稀疏特征上的高阶交互是必要的，这可以通过DNN、CrossNet和CIN在所有三个数据集上显著优于FM这一事实得到验证。另一方面，CIN是最好的单独模型，这证明了CIN在显式建模高阶特征交互方面的有效性。注意，k层CIN可以建模k度特征交互。有趣的是，CIN需要5层才能在必应新闻数据集上产生最佳结果。

### 4.3 集成模型的性能 (Q2)

xDeepFM将CIN和DNN集成为一个端到端模型。由于CIN和DNN在学习特征交互方面涵盖了两种截然不同的特性，我们很想知道将它们结合起来进行联合显式和隐式学习是否确实必要和有效。在这里我们比较了几个不限于单独模型的强基线，结果如表3所示。我们观察到LR远差于所有其他模型，这证明了基于因子分解的模型对于衡量稀疏特征至关重要。Wide&Deep、DCN、DeepFM和xDeepFM显著优于DNN，这直接反映出尽管它们很简单，但引入混合组件对于提升预测系统的准确性很重要。我们提出的xDeepFM在所有数据集上取得了最佳性能，这证明了结合显式和隐式高阶特征交互是必要的，并且xDeepFM在学习这类组合方面是有效的。另一个有趣的观察是，所有基于神经网络的模型都不需要非常深的网络结构来获得最佳性能。深度超参数的典型设置是2和3，xDeepFM的最佳深度设置为3，这表明我们学习的交互至多是4阶。

### 4.4 超参数研究 (Q3)

在本节中，我们研究超参数对xDeepFM的影响，包括：(1) 隐藏层数量；(2) 每层神经元数量；(3) 激活函数。我们在保持DNN部分的最佳设置的同时，改变CIN部分的设置进行实验。

**网络深度。** 图6a和7a展示了隐藏层数量的影响。我们可以观察到xDeepFM的性能最初随网络深度增加而提升。然而，当网络深度设置大于3时，模型性能下降。这是由于过拟合引起的，证据是我们注意到当我们添加更多隐藏层时，训练数据的损失仍在持续下降。

**每层神经元数量。** 增加每层神经元数量意味着增加CIN中的特征图数量。如图6b和7b所示，当我们将神经元数量从20增加到200时，必应新闻数据集上的模型性能稳步提升，而在大众点评数据集上，100是每层神经元数量的更合适设置。在这个实验中，我们将网络深度固定为3。

**激活函数。** 注意我们在CIN的神经元上使用恒等函数作为激活函数，如方程6所示。深度学习文献中的常见做法是在隐藏神经元上使用非线性激活函数。因此，我们比较了CIN上不同激活函数的结果（对于DNN中的神经元，我们保持relu激活函数）。如图6c和7c所示，恒等函数确实是CIN中最适合神经元的选择。

**图6: 网络超参数对AUC性能的影响。(a) 层数。(b) 每层神经元数量。(c) 激活函数。**

**图7: 网络超参数对Logloss性能的影响。(a) 层数。(b) 每层神经元数量。(c) 激活函数。**

---

## 5 相关工作

### 5.1 经典推荐系统

#### 5.1.1 非因子分解模型

对于网络规模的推荐系统（RS），输入特征通常是稀疏的、类别-连续混合的、且高维的。线性模型，如带有FTRL的逻辑回归[27]，由于易于管理、维护和部署而被广泛采用。由于线性模型缺乏学习特征交互的能力，数据科学家必须花费大量工作工程化交叉特征以达到更好的性能[22, 35]。考虑到某些隐藏特征难以手动设计，一些研究者利用提升决策树来帮助构建特征变换[14, 25]。

#### 5.1.2 因子分解模型

上述模型的一个主要缺点是它们不能泛化到训练集中未见过的特征交互。因子分解机（FM）[32]通过将每个特征嵌入到一个低维潜在向量中克服了这个问题。矩阵分解（MF）[18]只将ID视为特征，可以被视为FM的一种特例。推荐是通过两个潜在向量的乘积进行的，因此不需要用户和item在训练集中共现。MF是RS文献中最流行的基于模型的协同过滤方法[17, 20, 30, 38]。[4, 28]将MF扩展到利用辅助信息，其中同时包含线性模型和MF模型。另一方面，对于许多推荐系统，只有隐式反馈数据集（如用户的观看历史和浏览活动）可用。因此研究者将因子分解模型扩展到贝叶斯个性化排序（BPR）框架[11, 33, 34, 44]用于隐式反馈。

### 5.2 基于深度学习的推荐系统

深度学习技术在计算机视觉[10, 19]、语音识别[1, 15]和自然语言理解[6, 29]方面取得了巨大成功。因此，越来越多的研究者对在推荐系统中使用DNN感兴趣。

#### 5.2.1 用于高阶交互的深度学习

为了避免手动构建高阶交叉特征，研究者将DNN应用于域嵌入，从而可以自动学习类别特征交互的模式。代表性模型包括FNN [46]、PNN [31]、DeepCross [37]、NFM [12]、DCN [40]、Wide&Deep [5]和DeepFM [9]。这些模型与我们提出的xDeepFM高度相关。由于我们在第1节和第2节中已经回顾了它们，我们不在本节中详细讨论它们。我们已经证明，我们提出的xDeepFM与这些模型相比具有两个特殊性质：(1) xDeepFM以显式和隐式两种方式学习高阶特征交互；(2) xDeepFM在向量级而非位级学习特征交互。

#### 5.2.2 用于精细表示学习的深度学习

我们在本节中包括一些其他基于深度学习的推荐系统，因为它们不太专注于学习特征交互。一些早期的工作主要使用深度学习来建模辅助信息，如视觉数据[11]和音频数据[41]。最近，深度神经网络被用于建模推荐系统中的协同过滤（CF）。[13]提出了神经协同过滤（NCF），通过神经架构可以用任意函数替换MF中的内积。[36, 42]基于自编码器范式建模CF，他们通过实验证明基于自编码器的CF优于几个经典的MF模型。自编码器可以进一步用于联合建模CF和辅助信息，以生成更好的潜在因子[7, 39, 45]。[8, 23]使用神经网络联合训练多个域的潜在因子。[3]提出了注意力协同过滤（ACF）在item级和组件级学习更精细的偏好。[47]展示了传统推荐系统不能有效捕获兴趣多样性和局部激活，因此他们引入深度兴趣网络（DIN）来通过注意力激活机制表示用户的多样化兴趣。

---

## 6 结论

在本文中，我们提出了一种名为压缩交互网络（CIN）的新颖网络，旨在显式地学习高阶特征交互。CIN有两个特殊优点：(1) 它可以有效地学习特定有界度的特征交互；(2) 它在向量级学习特征交互。遵循几个流行模型的精神，我们将CIN和DNN结合在一个端到端框架中，并将得到的模型命名为极深因子分解机（xDeepFM）。因此，xDeepFM能够以显式和隐式两种方式自动学习高阶特征交互，这对于减少手动特征工程工作具有重要意义。我们进行了全面的实验，结果表明我们的xDeepFM在三个真实数据集上一致优于最先进的模型。

未来的工作有两个方向。首先，目前我们只使用求和池化来嵌入多值域。我们可以探索使用DIN机制[47]来根据候选item捕获相关的激活。其次，如第3.2.2节所述，CIN模块的时间复杂度较高。我们有兴趣开发一个分布式版本的xDeepFM，可以在GPU集群上进行高效训练。

---

## 致谢

作者感谢匿名评审人员的深刻评审意见，这些意见对本文的修订非常有帮助。本工作部分得到中国科学院青年创新促进会的支持。

---

## 参考文献

[1] Dario Amodei, Sundaram Ananthanarayanan, Rishita Anubhai, Jingliang Bai, Eric Battenberg, Carl Case, Jared Casper, Bryan Catanzaro, Qiang Cheng, Guoliang Chen, et al. 2016. Deep speech 2: End-to-end speech recognition in english and mandarin. 见 International Conference on Machine Learning. 173–182.

[2] Mathieu Blondel, Akinori Fujino, Naonori Ueda, and Masakazu Ishihata. 2016. Higher-order factorization machines. 见 Advances in Neural Information Processing Systems. 3351–3359.

[3] Jingyuan Chen, Hanwang Zhang, Xiangnan He, Liqiang Nie, Wei Liu, and Tat-Seng Chua. 2017. Attentive collaborative filtering: Multimedia recommendation with item-and component-level attention. 见 Proceedings of the 40th International ACM SIGIR conference on Research and Development in Information Retrieval. ACM, 335–344.

[4] Tianqi Chen, Weinan Zhang, Qiuxia Lu, Kailong Chen, Zhao Zheng, and Yong Yu. 2012. SVDFeature: a toolkit for feature-based collaborative filtering. Journal of Machine Learning Research 13, Dec (2012), 3619–3622.

[5] Heng-Tze Cheng, Levent Koc, Jeremiah Harmsen, Tal Shaked, Tushar Chandra, Hrishi Aradhye, Glen Anderson, Greg Corrado, Wei Chai, Mustafa Ispir, et al. 2016. Wide & deep learning for recommender systems. 见 Proceedings of the 1st Workshop on Deep Learning for Recommender Systems. ACM, 7–10.

[6] Kyunghyun Cho, Bart Van Merriënboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. 2014. Learning phrase representations using RNN encoder-decoder for statistical machine translation. arXiv preprint arXiv:1406.1078 (2014).

[7] Xin Dong, Lei Yu, Zhonghuo Wu, Yuxia Sun, Lingfeng Yuan, and Fangxi Zhang. 2017. A Hybrid Collaborative Filtering Model with Deep Structure for Recommender Systems. 见 AAAI. 1309–1315.

[8] Ali Mamdouh Elkahky, Yang Song, and Xiaodong He. 2015. A multi-view deep learning approach for cross domain user modeling in recommendation systems. 见 Proceedings of the 24th International Conference on World Wide Web. 278–288.

[9] Huifeng Guo, Ruiming Tang, Yunming Ye, Zhenguo Li, and Xiuqiang He. 2017. Deepfm: A factorization-machine based neural network for CTR prediction. arXiv preprint arXiv:1703.04247 (2017).

[10] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. 2016. Deep residual learning for image recognition. 见 Proceedings of the IEEE conference on computer vision and pattern recognition. 770–778.

[11] Ruining He and Julian McAuley. 2016. VBPR: Visual Bayesian Personalized Ranking from Implicit Feedback. 见 AAAI. 144–150.

[12] Xiangnan He and Tat-Seng Chua. 2017. Neural factorization machines for sparse predictive analytics. 见 Proceedings of the 40th International ACM SIGIR conference on Research and Development in Information Retrieval. ACM, 355–364.

[13] Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu, and Tat-Seng Chua. 2017. Neural collaborative filtering. 见 Proceedings of the 26th International Conference on World Wide Web. 173–182.

[14] Xinran He, Junfeng Pan, Ou Jin, Tianbing Xu, Bo Liu, Tao Xu, Yanxin Shi, Antoine Atallah, Ralf Herbrich, Stuart Bowers, et al. 2014. Practical lessons from predicting clicks on ads at facebook. 见 Proceedings of the Eighth International Workshop on Data Mining for Online Advertising. ACM, 1–9.

[15] Geoffrey Hinton, Li Deng, Dong Yu, George E Dahl, Abdel-rahman Mohamed, Navdeep Jaitly, Andrew Senior, Vincent Vanhoucke, Patrick Nguyen, Tara N Sainath, et al. 2012. Deep neural networks for acoustic modeling in speech recognition: The shared views of four research groups. IEEE Signal Processing Magazine 29, 6 (2012), 82–97.

[16] Diederik P Kingma and Jimmy Ba. 2014. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980 (2014).

[17] Yehuda Koren. 2008. Factorization meets the neighborhood: a multifaceted collaborative filtering model. 见 Proceedings of the 14th ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 426–434.

[18] Yehuda Koren, Robert Bell, and Chris Volinsky. 2009. Matrix factorization techniques for recommender systems. Computer 42, 8 (2009).

[19] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. 2012. Imagenet classification with deep convolutional neural networks. 见 Advances in neural information processing systems. 1097–1105.

[20] Joonseok Lee, Seungyeon Kim, Guy Lebanon, and Yoram Singer. 2013. Local low-rank matrix approximation. 见 International Conference on Machine Learning. 82–90.

[21] Jianxun Lian and Xing Xie. 2016. Cross-Device User Matching Based on Massive Browse Logs: The Runner-Up Solution for the 2016 CIKM Cup. arXiv preprint arXiv:1610.03928 (2016).

[22] Jianxun Lian, Fuzheng Zhang, Min Hou, Hongwei Wang, Xing Xie, and Guangzhong Sun. 2017. Practical Lessons for Job Recommendations in the Cold-Start Scenario. 见 Proceedings of the Recommender Systems Challenge 2017 (RecSys Challenge '17). ACM, Article 4, 6 pages. https://doi.org/10.1145/3124791.3124794

[23] Jianxun Lian, Fuzheng Zhang, Xing Xie, and Guangzhong Sun. 2017. CCCFNet: a content-boosted collaborative filtering neural network for cross domain recommender systems. 见 Proceedings of the 26th International Conference on World Wide Web Companion. 817–818.

[24] Jianxun Lian, Fuzheng Zhang, Xing Xie, and Guangzhong Sun. 2017. Restaurant Survival Analysis with Heterogeneous Information. 见 Proceedings of the 26th International Conference on World Wide Web Companion. 993–1002.

[25] Xiaoliang Ling, Weiwei Deng, Chen Gu, Hucheng Zhou, Cui Li, and Feng Sun. 2017. Model Ensemble for Click Prediction in Bing Search Ads. 见 Proceedings of the 26th International Conference on World Wide Web Companion. 689–698.

[26] Guimei Liu, Tam T Nguyen, Gang Zhao, Wei Zha, Jianbo Yang, Jianneng Cao, Min Wu, Peilin Zhao, and Wei Chen. 2016. Repeat buyer prediction for e-commerce. 见 Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. ACM, 155–164.

[27] H Brendan McMahan, Gary Holt, David Sculley, Michael Young, Dietmar Ebner, Julian Grady, Lan Nie, Todd Phillips, Eugene Davydov, Daniel Golovin, et al. 2013. Ad click prediction: a view from the trenches. 见 Proceedings of the 19th ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 1222–1230.

[28] Aditya Krishna Menon and Charles Elkan. 2010. A log-linear model with latent features for dyadic prediction. 见 Data Mining (ICDM), 2010 IEEE 10th International Conference on. IEEE, 364–373.

[29] Tomáš Mikolov, Martin Karafiát, Lukáš Burget, Jan Černocký, and Sanjeev Khudanpur. 2010. Recurrent neural network based language model. 见 Eleventh Annual Conference of the International Speech Communication Association.

[30] Rong Pan, Yunhong Zhou, Bin Cao, Nathan N Liu, Rajan Lukose, Martin Scholz, and Qiang Yang. 2008. One-class collaborative filtering. 见 Data Mining, 2008. ICDM'08. Eighth IEEE International Conference on. IEEE, 502–511.

[31] Yanru Qu, Han Cai, Kan Ren, Weinan Zhang, Yong Yu, Ying Wen, and Jun Wang. 2016. Product-based neural networks for user response prediction. 见 Data Mining (ICDM), 2016 IEEE 16th International Conference on. IEEE, 1149–1154.

[32] Steffen Rendle. 2010. Factorization machines. 见 Data Mining (ICDM), 2010 IEEE 10th International Conference on. IEEE, 995–1000.

[33] Steffen Rendle, Christoph Freudenthaler, Zeno Gantner, and Lars Schmidt-Thieme. 2009. BPR: Bayesian personalized ranking from implicit feedback. 见 Proceedings of the twenty-fifth conference on uncertainty in artificial intelligence. AUAI Press, 452–461.

[34] Steffen Rendle and Lars Schmidt-Thieme. 2010. Pairwise interaction tensor factorization for personalized tag recommendation. 见 Proceedings of the third ACM international conference on Web search and data mining. ACM, 81–90.

[35] Matthew Richardson, Ewa Dominowska, and Robert Ragno. 2007. Predicting clicks: estimating the click-through rate for new ads. 见 Proceedings of the 16th international conference on World Wide Web. ACM, 521–530.

[36] Suvash Sedhain, Aditya Krishna Menon, Scott Sanner, and Lexing Xie. 2015. Autorec: Autoencoders meet collaborative filtering. 见 Proceedings of the 24th International Conference on World Wide Web. ACM, 111–112.

[37] Ying Shan, T Ryan Hoens, Jian Jiao, Haijing Wang, Dong Yu, and JC Mao. 2016. Deep crossing: Web-scale modeling without manually crafted combinatorial features. 见 Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. ACM, 255–262.

[38] Nathan Srebro, Jason Rennie, and Tommi S Jaakkola. 2005. Maximum-margin matrix factorization. 见 Advances in neural information processing systems. 1329–1336.

[39] Hao Wang, Naiyan Wang, and Dit-Yan Yeung. 2015. Collaborative deep learning for recommender systems. 见 Proceedings of the 21th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. ACM, 1235–1244.

[40] Ruoxi Wang, Bin Fu, Gang Fu, and Mingliang Wang. 2017. Deep & Cross Network for Ad Click Predictions. arXiv preprint arXiv:1708.05123 (2017).

[41] Xinxi Wang and Ye Wang. 2014. Improving content-based and hybrid music recommendation using deep learning. 见 Proceedings of the 22nd ACM international conference on Multimedia. ACM, 627–636.

[42] Yao Wu, Christopher DuBois, Alice X Zheng, and Martin Ester. 2016. Collaborative denoising auto-encoders for top-n recommender systems. 见 Proceedings of the Ninth ACM International Conference on Web Search and Data Mining. ACM, 153–162.

[43] Jun Xiao, Hao Ye, Xiangnan He, Hanwang Zhang, Fei Wu, and Tat-Seng Chua. 2017. Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks. 见 Proceedings of the Twenty-Sixth International Joint Conference on Artificial Intelligence, IJCAI 2017, Melbourne, Australia, August 19-25, 2017. 3119–3125. https://doi.org/10.24963/ijcai.2017/435

[44] Fajie Yuan, Guibing Guo, Joemon M Jose, Long Chen, Haitao Yu, and Weinan Zhang. 2016. Lambdafm: learning optimal ranking with factorization machines using lambda surrogates. 见 Proceedings of the 25th ACM International on Conference on Information and Knowledge Management. ACM, 227–236.

[45] Fuzheng Zhang, Nicholas Jing Yuan, Defu Lian, Xing Xie, and Wei-Ying Ma. 2016. Collaborative knowledge base embedding for recommender systems. 见 Proceedings of the 22nd ACM SIGKDD international conference on knowledge discovery and data mining. ACM, 353–362.

[46] Weinan Zhang, Tianming Du, and Jun Wang. 2016. Deep learning over multi-field categorical data. 见 European conference on information retrieval. Springer, 45–57.

[47] Guorui Zhou, Chengru Song, Xiaoqiang Zhu, Xiao Ma, Yanghui Yan, Xingya Dai, Han Zhu, Junqi Jin, Han Li, and Kun Gai. 2017. Deep interest network for click-through rate prediction. arXiv preprint arXiv:1706.06978 (2017).
