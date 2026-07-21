# 所有模型都是错的，但许多是有用的：通过同时研究整个预测模型类来学习变量的重要性

> Aaron Fisher | Takeda Pharmaceuticals, Cambridge, MA 02139, USA
> Cynthia Rudin | Departments of Computer Science and Electrical and Computer Engineering, Duke University, Durham, NC 27708, USA
> Francesca Dominici | Department of Biostatistics, Harvard T.H. Chan School of Public Health, Boston, MA 02115, USA
> （作者按贡献排序，贡献最大者列于首位）

本文提出模型类依赖度（Model Class Reliance, MCR）作为预设模型类中所有表现良好模型对变量重要性（Variable Importance, VI）的取值范围。与仅描述单一模型不同，MCR通过考虑可能拟合数据良好的多个预测模型（可能具有不同参数形式）来提供更全面的重要性描述。在推导MCR的过程中，我们展示了基于排列的VI估计与U-统计量、条件变量重要性、条件因果效应和线性模型系数之间的联系。我们给出了MCR的概率界，并将其应用于Broward County刑事记录公开数据集，以研究累犯预测模型对性别和种族的依赖程度。

核心内容：

- 提出模型类依赖度（MCR）作为跨多个表现良好模型的VI范围度量，解决了"罗生门效应"问题
- 推导了排列重要性估计与U-统计量之间的联系，为理论分析提供基础
- 给出MCR的有限样本概率界，支持经验估计的可靠性
- 建立MR与条件因果效应、加性模型系数之间的联系
- 提出计算MCR的通用优化框架和具体算法实现
- 应用于COMPAS累犯预测模型，分析其对种族和性别的依赖程度

关键发现：

- MCR比单一模型VI度量更稳健，能反映整个模型类中变量重要性的完整范围
- 对于Broward County数据，种族和性别在COMPAS评分中发挥的作用介于"无"到"适度"之间，远小于"可接受"变量（年龄、前科次数等）
- MCR下界可用于识别所有良好模型都必须依赖的变量；上界可用于识别可以被安全丢弃的变量
- 基于排列的MR估计可表达为U-统计量，具有无偏、渐近正态等优良性质

---

## 摘要

变量重要性（Variable Importance, VI）工具描述协变量对预测模型准确度的贡献程度。然而，对于一个表现良好的模型（例如，固定系数向量 $\mathbf{\beta}$ 的线性模型 $f(\mathbf{x}) = \mathbf{x}^T \mathbf{\beta}$）重要的变量，对于另一个模型可能并不重要。在本文中，我们提出模型类依赖度（Model Class Reliance, MCR）作为预设模型类中所有表现良好模型的VI值的取值范围。因此，MCR通过考虑多个（可能具有不同参数形式的）预测模型都可能良好拟合数据这一事实，提供了更全面的重要性描述。在推导MCR的过程中，我们展示了基于排列的VI估计的几个有信息量的结果，这些结果基于随机森林中使用的VI度量。具体地，我们推导了单个预测模型的排列重要性估计与U-统计量、条件变量重要性、条件因果效应和线性模型系数之间的联系。然后，我们使用一种新颖的、可推广的技术给出了MCR的概率界。我们将MCR应用于Broward County刑事记录的公开数据集，以研究累犯预测模型对性别和种族的依赖程度。在该应用中，MCR可用于帮助了解未知的专有模型的VI。

关键词：罗生门效应（Rashomon），排列重要性，条件变量重要性，U-统计量，透明性，可解释模型

## 1. 引言

变量重要性（VI）工具描述预测模型的准确度在多大程度上依赖于每个协变量中的信息。例如，在随机森林中，VI通过置换一个协变量时预测准确度的下降来度量 [7,8,95,1,107,42,26,43]。类似的"扰动"VI度量已被用于神经网络，其中向协变量添加噪声 [79,106,39,85]。这些工具可用于识别需要高精度测量的协变量、提高"黑箱"预测模型的透明性（另见 [82]），或确定可能导致模型失败的情景。

然而，现有的VI度量通常没有考虑到许多预测模型可能以几乎相同的精度拟合数据。在这种情况下，一位分析师使用的模型可能依赖完全不同的协变量信息，而非另一位分析师使用的模型。这种常见情景被称为统计学的"罗生门效应"（Rashomon effect）[8,60,93,100,74,62]。该术语灵感来源于1950年黑泽明同名电影，片中四位证人对同一事件给出了不同的描述和解释。在罗生门效应下，分析师应如何全面描述每个协变量的重要性？一位分析师能否复现另一位分析师的结论？给出最佳预测的模型是否必然给出最准确的解释？

为应对这些挑战，我们分析提供接近最优精度的预测模型集合，我们称之为罗生门集（Rashomon set）。这种方法区别于传统的在预设候选模型类中训练选择单一预测模型的做法。我们的动机是，罗生门集（下文正式定义）概括了一个分析师可能选择的有效预测策略的范围。此外，即使候选模型类不包含真实的数据生成过程，我们仍可期望其中某些模型以与数据生成过程类似的方式运作。特别地，我们可能期望存在某些表现良好的候选模型，它们对感兴趣变量的重要性赋值与底层数据生成过程相同。如果是这样，那么研究表现良好的模型集合将使我们能够推导出关于数据生成过程的信息。

应用这种方法研究变量重要性，我们将模型类依赖度（MCR）定义为给定模型类中任何表现良好的模型在预测准确度上对感兴趣变量的最高和最低依赖程度。粗略地说，MCR捕获了与表现良好模型相关联的解释或机制的范围。由于最终的范围同时总结了多个预测模型（而非单一模型），我们期望该范围较少受到单个分析师在模型拟合过程中所做的选择的影响。MCR的目标不是反映这些选择，而是反映预测问题本身的性质。

在推导MCR的过程中，我们做出了若干具体的技朮贡献。首先，我们回顾了一个核心度量，用于衡量单个预测模型在准确度上对感兴趣协变量的依赖程度，我们称之为模型依赖度（Model Reliance, MR）。该度量基于随机森林的排列重要性度量 [8,7]，可以扩展描述条件重要性（见第8节，以及 [95]）。我们建立了基于排列的重要性估计（MR）与U-统计量之间的联系，这有助于后续的理论结果。此外，我们推导了MR、条件因果效应和加性模型系数之间的联系。在MR基础上扩展，我们提出MCR，将MR的定义推广到模型类。我们推导了MCR的有限样本界，这启发了一种直观的MCR估计量。最后，我们提出了该估计量的计算过程。

我们开发的用于研究罗生门集的工具相当通用，可用于对表现良好模型的任意特征进行有限样本推断。例如，除了描述变量重要性，这些工具还可以描述良好拟合模型分配给特定协变量分布的风险预测范围，或良好拟合模型所做预测的方差。在某些情况下，这些新颖技术可能提供以前不存在的有限样本置信区间（CIs）（见第5节）。

MCR和罗生门效应在刑事累犯预测的背景下变得尤为重要。基于刑事记录数据训练的商业累犯风险模型越来越多地在美国法庭中使用。一个关切点是，这些模型可能依赖通常被视为不可接受的信息（例如，种族、性别或这些变量的代理变量）来估计累犯风险。相关模型通常是专有的，无法直接研究。尽管如此，如果这些模型所做的预测是可公开获取的，就有可能识别出与感兴趣的商业模型足够相似的替代预测模型。

在本文中，我们特别考虑商业模型COMPAS（Correctional Offender Management Profiling for Alternative Sanctions），由Northpointe Inc.公司开发（随后在2017年，Northpointe Inc.、Courtview Justice Solutions Inc.和Constellation Justice Systems Inc.合并成立Equivant）。我们的目标是估计COMPAS在多大程度上依赖种族、性别或这些变量在我们数据集中未测量的代理变量。为此，我们应用一个广泛的灵活基于核的预测模型类来预测COMPAS评分。在这种设置下，MCR区间反映了在我们的模型类中，任何在预测COMPAS评分相对准确的同时在多大程度上可以依赖种族和性别的最高和最低程度。借助MCR，我们可以放宽常见的假设（即能够正确地将未知的感兴趣模型（此处为COMPAS）指定到某个参数形式）。相反，我们不假设COMPAS模型本身包含在我们的模型类中，而是假设我们的模型类至少包含一个表现良好的替代模型，该模型对敏感协变量的依赖程度与COMPAS相同。在此假设下，MCR区间将包含COMPAS的VI值。应用我们的方法，我们发现种族、性别及其潜在的代理变量可能不是COMPAS评分中的主导预测因素（见第10节的分析和讨论）。

本文的其余部分组织如下。在第2节中，我们介绍符号，并给出我们方法的高级总结，附有可视化图示。在第3节和第4节中，我们分别正式介绍MR和MCR，并推导各自的理论性质。我们还回顾了文献中相关的变量重要性实践，例如在移除一个协变量后重新训练模型。在第5节中，我们讨论了我们方法在确定其他问题的有限样本置信区间方面的普遍适用性。在第6节中，我们提出了计算MCR的通用过程。在第7节中，我们给出了该过程在（正则化）线性模型和再生核希尔伯特空间中的线性模型的具体实现。我们还展示了对于加性模型，MR可以用模型的系数表示。在第8节中，我们概述了MR、因果推断和条件变量重要性之间的联系。在第9节中，我们通过模拟的玩具示例说明了MR和MCR，以帮助建立直觉。我们还展示了在错误设定条件下，估计未知底层条件期望函数的MR的模拟研究。我们在第10节分析了前述关于累犯的知名公开数据集。所有证明均在附录中给出。

## 2. 符号与技术总结

"变量重要性"度量的标签已被广泛用于描述推断 [101,31,105] 或预测的方法。虽然这两个目标高度相关，但我们主要关注预测模型在多大程度上依赖协变量来达到准确度。我们使用"模型依赖度"而非"重要性"等术语来明确这一背景。

为了评估预测模型对变量的依赖程度，我们现在介绍随机变量、数据、预测模型类和评估预测的损失函数的符号。令 $Z = (Y, X_1, X_2) \in \mathcal{Z}$ 为随机变量，其中结果 $Y \in \mathcal{Y}$ 和协变量 $X = (X_1, X_2) \in \mathcal{X}$，协变量子集 $X_1 \in \mathcal{X}_1$ 和 $X_2 \in \mathcal{X}_2$ 各自可能是多变量的。我们假设对 $Z$ 的观测是独立同分布的，$n \ge 2$，并且当在本文提到的集合上进行优化时，$\arg\min$ 和 $\arg\max$ 操作的解存在（例如，下文定理4中）。我们的目标是研究不同的预测模型在多大程度上依赖 $X_1$ 来预测 $Y$。

我们将数据集称为 $\mathbf{Z} = \begin{bmatrix} \mathbf{y} & \mathbf{X} \end{bmatrix}$，一个由第一列中的 $n$ 长结果向量 $\mathbf{y}$ 和其余列中的 $n \times p$ 协变量矩阵 $\mathbf{X} = \begin{bmatrix} \mathbf{X}_1 & \mathbf{X}_2 \end{bmatrix}$ 组成的矩阵。一般地，对于给定向量 $\mathbf{v}$，令 $\mathbf{v}[j]$ 表示其第 $j$ 个元素。对于给定矩阵 $\mathbf{A}$，令 $\mathbf{A}'$、$\mathbf{A}[i,\cdot]$、$\mathbf{A}[\cdot,j]$ 和 $\mathbf{A}[i,j]$ 分别表示 $\mathbf{A}$ 的转置、第 $i$ 行、第 $j$ 列以及第 $i$ 行第 $j$ 列的元素。

我们使用术语模型类（model class）指代预设子集 $\mathcal{F} \subset \{f \mid f : \mathcal{X} \to \mathcal{Y}\}$，即从 $\mathcal{X}$ 到 $\mathcal{Y}$ 的可测函数的子集。我们将成员函数 $f \in \mathcal{F}$ 称为预测模型，或简称为模型。给定模型 $f$，我们使用非负损失函数 $L : (\mathcal{F} \times \mathcal{Z}) \to \mathbb{R}_{\ge 0}$ 评估其性能。例如，对于回归问题，$L$ 可以是平方误差损失 $L_{\text{se}}(f, (y, x_1, x_2)) = (y - f(x_1, x_2))^2$；对于分类问题，可以是铰链损失 $L_{\text{h}}(f, (y, x_1, x_2)) = (1 - yf(x_1, x_2))_+$。我们使用术语算法（algorithm）指代任何将数据集作为输入并返回模型 $f \in \mathcal{F}$ 作为输出的过程 $\mathcal{A} : \mathcal{Z}^n \to \mathcal{F}$。

### 2.1 罗生门集与模型类依赖度概述

许多传统的统计估计来自对单个拟合预测模型的描述。相反地，在本节中，我们总结了研究一组接近最优模型的方法。为定义这组模型，我们需要一个预设的"参考"模型，记为 $f_{\text{ref}}$，作为预测性能的基准。例如，$f_{\text{ref}}$ 可以来自用于预测医院急诊室伤害严重程度的流程图，或来自当前在实践中实施的另一量化决策规则。给定参考模型 $f_{\text{ref}}$，我们将总体 $\epsilon$-罗生门集定义为预期损失不超过 $f_{\text{ref}}$ 的预期损失加上 $\epsilon$ 的模型子集。我们将该集合记为 $\mathcal{R}(\epsilon) := \{f \in \mathcal{F} : \mathbb{E}L(f, Z) \le \mathbb{E}L(f_{\text{ref}}, Z) + \epsilon\}$，其中 $\mathbb{E}$ 表示关于总体分布的期望。该集合可以被视为代表那些可能因数据测量、处理、过滤、模型参数化、协变量选择或其他分析选择上的差异而得到的模型（见第4节）。

**图1：罗生门集与模型类依赖度**——面板（A）展示了模型类 $\mathcal{F}$ 内的假想罗生门集 $\mathcal{R}(\epsilon)$。y 轴显示每个模型 $f \in \mathcal{F}$ 的预期损失，x 轴显示每个模型 $f$ 对 $X_1$ 的依赖程度（第3节正式定义）。沿 x 轴，总体水平 MCR 范围以蓝色高亮显示，展示了与表现良好模型对应的 MR 值（见第4节）。面板（B）展示了面板（A）的样本内类比。这里，y 轴表示样本内损失 $\widehat{\mathbb{E}}L(f, \mathbf{Z}) := \frac{1}{n} \sum_{i=1}^n L(f, \mathbf{Z}[i,\cdot])$；x 轴显示每个模型 $f \in \mathcal{F}$ 对 $X_1$ 的经验模型依赖度（见第3节）；x 轴的高亮部分显示经验 MCR（见第4节）。

图1-A展示了假想的总体 $\epsilon$-罗生门集示例。这里，y 轴显示每个模型 $f \in \mathcal{F}$ 的预期损失，x 轴显示每个模型在多大程度上依赖 $X_1$ 来获得其预测精度。更具体地说，给定预测模型 $f$，x 轴显示当向 $X_1$ 添加噪声时，$f$ 的预期损失的百分比增加。我们将该度量称为 $f$ 对 $X_1$ 的模型依赖度（MR），非正式地写为 $\text{MR}(f) := \frac{\text{expected loss of }f\text{ with noise}}{\text{expected loss of }f\text{ without noise}}$（式2.1）。添加的噪声必须满足特定性质，即必须使 $X_1$ 对结果 $Y$ 完全无信息，而不改变 $X_1$ 的边缘分布（详见第3节，以及 [7,8]）。

我们的核心目标是理解模型在保持良好预测能力的同时，对感兴趣协变量（$X_1$）的依赖程度可以有多高或多低。在图1-A中，这个可能的 MR 值范围由 x 轴上的高亮区间显示。我们将这种类型的区间称为总体水平模型类依赖度（MCR）范围（见第4节），正式定义为 $[\text{MCR}^-(\epsilon), \text{MCR}^+(\epsilon)] := \left[\min_{f \in \mathcal{R}(\epsilon)} \text{MR}(f),\ \max_{f \in \mathcal{R}(\epsilon)} \text{MR}(f)\right]$（式2.2）。

为估计这个范围，我们使用总体 $\epsilon$-罗生门集和 MR 的经验类比，基于观测数据（图1-B）。我们将经验 $\epsilon$-罗生门集定义为样本内损失不超过 $f_{\text{ref}}$ 的样本内损失加上 $\epsilon$ 的模型集合，记为 $\widehat{\mathcal{R}}(\epsilon)$。非正式地，我们将模型 $f$ 对 $X_1$ 的经验 MR 定义为 $\widehat{\text{MR}}(f) := \frac{\text{in-sample loss of }f\text{ with noise}}{\text{in-sample loss of }f\text{ without noise}}$（式2.3），即 $f$ 在给定样本中表现出的对 $X_1$ 的依赖程度（详见第3节）。最后，我们将经验模型类依赖度定义为与具有强样本内性能的模型对应的经验 MR 值的范围（见第4节），正式写为 $[\widehat{\text{MCR}}^-(\epsilon), \widehat{\text{MCR}}^+(\epsilon)] := \left[\min_{f \in \widehat{\mathcal{R}}(\epsilon)} \widehat{\text{MR}}(f),\ \max_{f \in \widehat{\mathcal{R}}(\epsilon)} \widehat{\text{MR}}(f)\right]$（式2.4）。在图1-B中，上述范围由 x 轴的高亮部分显示。

我们在开发 MCR 的过程中做出了若干技术贡献。

- MR 和总体水平 MCR 的估计：给定 $f$，我们展示了 $\widehat{\text{MR}}(f)$ 作为 $\text{MR}(f)$ 的估计量的优良性质，使用 U-统计量的结果（第3.1节和定理5）。我们还推导了总体水平 MCR 的有限样本界，其中一些需要对 $\mathcal{F}$ 的复杂度进行限制（以覆盖数的形式）。这些界表明，在相当弱的条件下，经验 MCR 提供了对总体水平 MCR 的合理估计（详见第4节）。

- 经验 MCR 的计算：虽然经验 MCR 在给定样本后是完全确定的，但式2.4中的最小化和最大化需要重要的计算。为解决这个问题，我们概述了 MCR 的通用优化过程（第6节）。我们给出了该过程在模型类 $\mathcal{F}$ 为（正则化）线性回归模型集合或再生核希尔伯特空间中的回归模型集合时的具体实现（第7节）。我们提出的过程的输出是一个包含 $\mathcal{F}$ 的闭式凸包络，可用于近似任何性能水平 $\epsilon$ 的经验 MCR（见图2的示意）。然而，对于标准经验损失最小化仍是开放问题的复杂模型类（例如神经网络），计算经验 MCR 也仍然是一个开放问题。

- MR 用模型系数和因果效应的解释：我们展示了加性模型的 MR 可以写作模型系数的函数（命题15），并且二元协变量 $X_1$ 的 MR 可以写作 $X_1$ 对 $Y$ 的条件因果效应的函数（命题19）。

- 扩展到条件重要性：我们提供了 MR 的扩展，类似于条件重要性的概念 [95]。该扩展描述了模型在多大程度上依赖 $X_1$ 中无法从 $X_2$ 获得的特定信息（第8.2节）。

- 罗生门集的推广：超越变量重要性的概念，我们还推广了 MCR 的有限样结果，以描述总体 $\epsilon$-罗生门集中模型的任意特征。正如我们在并行工作 [20] 中讨论的，这一推广类似于轮廓似然区间，并且可以例如用于界定表现良好的预测模型可能分配给特定协变量集的风险预测范围（第5节）。

我们将在下一节正式回顾模型依赖度。

**图2：经验 MCR 计算过程的输出图示**——我们的计算过程产生一个包含 $\mathcal{F}$ 的闭式凸包络（如上方的紫色实线所示），它为任何 $\epsilon$ 值界定了经验 MCR（见式2.4）。该过程顺序执行，在感兴趣的 $\epsilon$ 值附近尽可能收紧这些界（第6节）。我们数据分析的结果（图8）以与上述紫色包络相同的格式呈现。

## 3. 模型依赖度

为正式描述固定预测模型 $f$ 的预期准确度在多大程度上依赖随机变量 $X_1$，我们使用"交换"损失的概念，即 $X_1$ 被变为无信息。在本节中，我们将 $f$ 视为预先指定的感兴趣预测模型（如 [38]）。令 $Z^{(a)} = (Y^{(a)}, X_1^{(a)}, X_2^{(a)})$ 和 $Z^{(b)} = (Y^{(b)}, X_1^{(b)}, X_2^{(b)})$ 为独立的随机变量，各自遵循与 $Z = (Y, X_1, X_2)$ 相同的分布。我们定义

$$
e_{\text{switch}}(f) := \mathbb{E}L\{f, (Y^{(b)}, X_1^{(a)}, X_2^{(b)})\} \qquad (2.1)
$$

表示模型 $f$ 跨观测对 $(Z^{(a)}, Z^{(b)})$ 的预期损失，其中 $X_1^{(a)}$ 和 $X_1^{(b)}$ 的值已被交换。要理解上述方程的解释，注意我们使用了 $Z^{(b)}$ 中的变量 $(Y^{(b)}, X_2^{(b)})$，但使用了独立副本 $Z^{(b)}$ 中的变量 $X_1^{(b)}$。这就是为什么我们说 $X_1^{(a)}$ 和 $X_1^{(b)}$ 已被交换；$(Y^{(b)}, X_1^{(a)}, X_2^{(b)})$ 的值彼此之间的关系不像它们被一起选择时那样。$e_{\text{switch}}(f)$ 的另一种解释是，当以 $X_1$ 变得完全无信息于 $Y$ 但 $X_1$ 的边缘分布不变的方式向 $X_1$ 添加噪声时，$f$ 的预期损失。

作为参考点，我们将 $e_{\text{switch}}(f)$ 与没有变量被交换时的标准预期损失 $e_{\text{orig}}(f) := \mathbb{E}L(f, (Y, X_1, X_2))$ 进行比较。从这两个量出发，我们正式定义模型依赖度（MR）为比率

$$
\text{MR}(f) := \frac{e_{\text{switch}}(f)}{e_{\text{orig}}(f)}, \qquad (3.1)
$$

如式2.1所述。$\text{MR}(f)$ 值越大，表示 $f$ 对 $X_1$ 的依赖程度越高。例如，$\text{MR}(f)=2$ 表示模型严重依赖 $X_1$，因为当 $X_1$ 被打乱时其损失翻倍。$\text{MR}(f)=1$ 表示对 $X_1$ 没有依赖，因为当 $X_1$ 被打乱时模型损失不变。依赖值严格小于1的模型更难解释，因为它们对感兴趣变量的依赖程度低于随机猜测。有趣的是，存在依赖度小于1的模型是可能的。例如，如果模型 $f'$ 将 $X_1$ 和 $Y$ 视为正相关而实际上它们是负相关的，则可能满足 $\text{MR}(f') < 1$。然而，在许多情况下，存在 $f' \in \mathcal{F}$ 满足 $\text{MR}(f') < 1$ 意味着存在另一个性能更好的模型 $f'' \in \mathcal{F}$ 满足 $\text{MR}(f'') = 1$ 且 $e_{\text{orig}}(f'') \le e_{\text{orig}}(f')$。也就是说，虽然可能存在 MR 值小于1的模型，但它们通常会是次优的（见附录A.2）。

模型依赖度也可以定义为差值而非比率，即 $\text{MR}^{\text{difference}}(f) := e_{\text{switch}}(f) - e_{\text{orig}}(f)$。在附录A.5中，我们讨论了两种定义下许多结果是相似的。

### 3.1 用U-统计量估计模型依赖度及其与排列变量重要性的联系

给定模型 $f$ 和数据集 $\mathbf{Z} = \begin{bmatrix} \mathbf{y} & \mathbf{X} \end{bmatrix}$，我们通过分别估计式3.1的分子和分母来估计 $\text{MR}(f)$。我们用标准经验损失估计 $e_{\text{orig}}(f)$：

$$
\widehat{e}_{\text{orig}}(f) := \frac{1}{n} \sum_{i=1}^n L\{f, (\mathbf{y}[i], \mathbf{X}_1[i,\cdot], \mathbf{X}_2[i,\cdot])\}. \qquad (3.2)
$$

我们通过在所有观测对之间执行"交换"操作来估计 $e_{\text{switch}}(f)$：

$$
\widehat{e}_{\text{switch}}(f) := \frac{1}{n(n-1)} \sum_{i=1}^n \sum_{j \ne i} L\{f, (\mathbf{y}[j], \mathbf{X}_1[i,\cdot], \mathbf{X}_2[j,\cdot])\}. \qquad (3.3)
$$

上面，我们聚合了 $(Y, X_2)$ 和 $X_1$ 的观测值的所有可能组合，排除了原始样本中实际观测到的配对。如果由于样本量原因，对所有可能配对（式3.3）的求和计算成本过高，$e_{\text{switch}}(f)$ 的另一种估计器是

$$
\widehat{e}_{\text{divide}}(f) := \frac{1}{2\lfloor n/2 \rfloor} \sum_{i=1}^{\lfloor n/2 \rfloor} \left\{ L\{f, (\mathbf{y}[i], \mathbf{X}_1[i+\lfloor n/2 \rfloor,\cdot], \mathbf{X}_2[i,\cdot])\} \qquad (3.4) \\
+ L\{f, (\mathbf{y}[i+\lfloor n/2 \rfloor], \mathbf{X}_1[i,\cdot], \mathbf{X}_2[i+\lfloor n/2 \rfloor,\cdot])\} \qquad (3.5) \right\}.
$$

这里，我们不求和所有配对，而是将样本分成两半。然后将前半的 $(Y, X_2)$ 值与后半的 $X_1$ 值配对（第一行），反之亦然（第二行）。所有上述三种估计器（式3.2、3.3和3.5）对于它们各自的估计目标都是无偏的，我们稍后将详细讨论。

最后，我们可以用代入估计器估计 $\text{MR}(f)$：

$$
\widehat{\text{MR}}(f) := \frac{\widehat{e}_{\text{switch}}(f)}{\widehat{e}_{\text{orig}}(f)}, \qquad (3.6)
$$

我们将其定义为 $f$ 对 $X_1$ 的经验模型依赖度。通过这种方式，我们形式化了式2.3中的经验 MR 定义。

再次地，我们对经验 MR 的定义与 Breiman [7] 的基于排列的变量重要性方法非常相似，其中 Breiman 使用单次随机排列，而我们考虑所有可能的配对。为了更精确地比较这两种方法，令 $\{\pi_1, \ldots, \pi_{n!}\}$ 为一组 $n$ 长向量，每个向量包含集合 $\{1, \ldots, n\}$ 的一个不同排列。Breiman [7] 的方法类似于计算损失 $\sum_{i=1}^n L\{f, (\mathbf{y}[i], \mathbf{X}_1[\pi_l[i],\cdot], \mathbf{X}_2[i,\cdot])\}$，其中 $\pi_l \in \{\pi_1, \ldots, \pi_{n!}\}$ 是随机选择的排列向量。类似地，我们式3.3中的计算正比于所有可能的 $(n!)$ 个排列上的损失之和，排除了 $\mathbf{X}_1$ 的行与 $\begin{bmatrix} \mathbf{X}_2 & \mathbf{y} \end{bmatrix}$ 的行在原始样本中出现的 $n$ 个独特组合（见附录A.3）。排除这些观测对于保持 $\widehat{e}_{\text{switch}}(f)$ 的（有限样本）无偏性是必要的。

估计器 $\widehat{e}_{\text{orig}}(f)$、$\widehat{e}_{\text{switch}}(f)$ 和 $\widehat{e}_{\text{divide}}(f)$ 都属于研究充分的 U-统计量类别。因此，在相当温和的条件下，这些估计器是无偏的、渐近正态的，并且具有有限样本概率界 [41,42,84]（另见 [24] 在机器学习中早期使用 U-统计量，以及 [25] 中的注意事项）。据我们所知，基于排列的重要性与 U-统计量之间的联系此前尚未建立。

虽然上述 U-统计量的结果依赖于模型 $f$ 是事先固定的，但我们也可以利用这些结果为足够正则化的类 $\mathcal{F}$ 中的所有模型创建 MR 估计误差的一致界。我们在第4节（定理5）中正式给出这个界，在介绍了模型类复杂度的必要条件之后。这个一致界的存在意味着可以在同一数据上训练模型并估计其对变量的重要性，而不需要使用样本分割。这与随机森林 [7] 的经典 VI 方法不同，后者避免了样本内重要性估计。在那里，集成中的每棵树在数据的随机子集上拟合，并使用保留数据估计该树的 VI。然后聚合特定于树的 VI 估计以获得整个集成的 VI 估计。尽管这种样本分割方法在许多情况下是有帮助的，但 MR 的一致界表明，根据样本量和 $\mathcal{F}$ 的复杂度，它们并非严格必要。

### 3.2 现有变量重要性方法的局限性

变量选择或描述变量间关系的几种常见方法并不一定捕获变量的重要性。零假设检验方法可能识别出一种关系，但并不描述该关系的强度。类似地，检查一个变量是否被稀疏模型拟合算法（如 Lasso [35]）包含，并不能描述该变量被依赖的程度。偏依赖图 [8,35] 在多个变量感兴趣或预测模型包含交互效应时可能难以解释。

另一种常见的 VI 过程是运行模型拟合算法两次：首先在所有数据上，然后在从数据集中移除 $X_1$ 后再运行一次。然后比较两个结果模型的损失，以确定 $X_1$ 的重要性或"必要性"[39]。因为该度量是两个预测模型（而非一个）的函数，它并不衡量任何一个单个模型对 $X_1$ 的依赖程度。我们将这种方法称为测量经验算法依赖度（Algorithm Reliance, AR）对 $X_1$ 的依赖，因为模型拟合算法是两个模型之间的共同属性。相关的过程由 Breiman et al. [8] 和 Breiman [7] 提出，用于测量 $X_1$ 的充分性。

正如我们在第3.1节中讨论的，RF的基于排列的 VI 度量 [7,8] 构成了我们 MR 定义的基础。这个 RF VI 度量一直是实证研究的主题 [2,9,98]，并已提出了该度量的几种变体 [95,96,1,34]。Mentch and Hooker [64] 使用 U-统计量研究在子样本上拟合的集成模型的预测，类似于 RF 中使用的自助聚合。与"平均不纯度减少"相关的过程（RF 衍生的另一种 VI 度量）已被 Louppe et al. [63] 和 Kazemitabar et al. [57] 从理论上研究。所有这些文献都专注于 RF、集成或单棵树的 VI 度量。我们的模型依赖度估计器与传统的 RF VI 度量 [7] 不同之处在于，我们对整个模型的输入进行排列，而不是对每个集成成员的输入进行排列。因此，我们的方法可以普遍使用，不限于树模型或集成模型。

在 RF VI 背景之外，Zhu et al. [107] 提出了类似于我们模型依赖度定义的目标估计量，Gregorutti et al. [42,43] 提出了类似于 $e_{\text{switch}}(f) - e_{\text{orig}}(f)$ 的目标估计量。这些近期工作关注于 $f$ 对 $X_1$ 的模型依赖度，特别是当 $f$ 等于 $Y$ 的条件期望函数（即 $f(x_1, x_2) = \mathbb{E}[Y \mid X_1 = x_1, X_2 = x_2]$）时。相对地，我们考虑任意预测模型 $f$ 的模型依赖度。Datta et al. [26] 研究了当变量子集被排列时（无论排列是否影响损失函数 $L$），模型预测值预期变化程度。这些 VI 方法专门针对单一预测模型，MR 也是如此。在下一节中，我们考虑一个更一般的重要性概念：特定集合中任何模型可能在多大程度上依赖感兴趣的变量。

## 4. 模型类依赖度

与许多统计过程一样，我们的 MR 度量（第3节）产生对单个预测模型的描述。给定具有高预测精度的模型，MR 描述了模型性能在多大程度上依赖于感兴趣的协变量（$X_1$）。然而，通常还有许多其他模型表现同样良好，但对 $X_1$ 的依赖程度不同。基于这一概念，我们现在研究预设模型类 $\mathcal{F}$ 中任何表现良好的模型可能在多大程度上依赖感兴趣的协变量。

回顾第2.1节，为了定义接近最优模型的总体 $\epsilon$-罗生门集，我们必须选择一个"参考"模型 $f_{\text{ref}}$ 作为性能基准。为了讨论这个选择，我们现在引入对总体 $\epsilon$-罗生门集更显式的符号：

$$
\mathcal{R}(\epsilon, f_{\text{ref}}, \mathcal{F}) := \{f \in \mathcal{F} : e_{\text{orig}}(f) \le e_{\text{orig}}(f_{\text{ref}}) + \epsilon\}. \qquad (4.1)
$$

注意我们在 $f_{\text{ref}}$ 和 $\mathcal{F}$ 从上下文清晰时交替使用 $\mathcal{R}(\epsilon, f_{\text{ref}}, \mathcal{F})$ 和 $\mathcal{R}(\epsilon)$。类似地，我们偶尔使用更显式的符号 $\widehat{\mathcal{R}}(\epsilon, f_{\text{ref}}, \mathcal{F}) := \{f \in \mathcal{F} : \widehat{e}_{\text{orig}}(f) \le \widehat{e}_{\text{orig}}(f_{\text{ref}}) + \epsilon\}$ 表示经验 $\epsilon$-罗生门集，但通常缩写为 $\widehat{\mathcal{R}}(\epsilon)$。

虽然 $f_{\text{ref}}$ 可以通过最小化样本内损失来选定，但在 $f_{\text{ref}}$ 被预设的假设下，$\mathcal{R}(\epsilon, f_{\text{ref}}, \mathcal{F})$ 的理论研究得以简化。例如，$f_{\text{ref}}$ 可能来自用于预测医院急诊室伤害严重程度的流程图，或来自当前实践中实施的另一量化决策规则。模型 $f_{\text{ref}}$ 也可以使用样本分割来选定。在某些情况下，可能需要将 $f_{\text{ref}}$ 固定为同类最佳模型 $f^\star := \arg\min_{f \in \mathcal{F}} e_{\text{orig}}(f)$，但这通常不可行，因为 $f^\star$ 是未知的。尽管如此，对于任何 $f_{\text{ref}} \in \mathcal{F}$，使用 $f_{\text{ref}}$ 定义的罗生门集 $\mathcal{R}(\epsilon, f_{\text{ref}}, \mathcal{F})$ 在以下意义上是保守的：它包含使用 $f^\star$ 定义的罗生门集 $\mathcal{R}(\epsilon, f^\star, \mathcal{F})$。

现在我们可以通过简单地将 $\text{MR}(f)$ 和 $\widehat{\text{MR}}(f)$ 的定义（第3节）代入式2.2和2.4来形式化总体水平 MCR 和经验 MCR 的定义。研究总体水平 MCR（式2.2）是本文的主要焦点，因为它提供了比来自单一模型的度量更全面的重要性视图。如果 $\text{MCR}^+(\epsilon)$ 较低，则 $\mathcal{F}$ 中没有表现良好的模型对 $X_1$ 赋予高重要性，且 $X_1$ 可以以低成本被丢弃，无论未来的建模决策如何。如果 $\text{MCR}^-(\epsilon)$ 很大，则 $\mathcal{F}$ 中每个表现良好的模型必须大量依赖 $X_1$，且在建模过程中应给予 $X_1$ 仔细关注。这里，$\mathcal{F}$ 本身可能包含几种参数模型形式（例如，所有线性模型和所有少于6个单分裂节点的决策树模型）。我们强调，范围 $[\text{MCR}^-(\epsilon), \text{MCR}^+(\epsilon)]$ 不依赖于用于选择模型 $f \in \mathcal{F}$ 的拟合算法。该范围对于任何产生 $\mathcal{F}$ 中模型的算法都有效，并适用于任何 $f \in \mathcal{F}$。

在本节的剩余部分，我们推导总体水平 MCR 的有限样本界，由此论证经验 MCR 提供了总体水平 MCR 的合理估计（第4.1节）。在附录B.7中，我们考虑了罗生门集和 MCR 的另一种形式，其中我们将 $\mathcal{R}(\epsilon)$ 定义中的相对损失阈值替换为绝对损失阈值。这种另一种形式在实践中可能类似，但仍然需要指定参考函数 $f_{\text{ref}}$ 以确保 $\mathcal{R}(\epsilon)$ 和 $\widehat{\mathcal{R}}(\epsilon)$ 非空。

### 4.1 通过推导有限样本界来启发 MCR 的经验估计量

在本节中，我们推导 $\text{MCR}^+(\epsilon)$ 和 $\text{MCR}^-(\epsilon)$ 的有限样本概率界。我们的结果意味着，在最小假设下，$\widehat{\text{MCR}}^+(\epsilon)$ 和 $\widehat{\text{MCR}}^-(\epsilon)$ 分别以高概率处于 $\text{MCR}^+(\epsilon)$ 和 $\text{MCR}^-(\epsilon)$ 的邻域内。然而，我们的假设的弱性（对于统计学习理论分析而言是典型的）使得我们得到的置信区间宽度在实践中过大，因此我们仅用这些结果来展示 $\widehat{\text{MCR}}^+(\epsilon)$ 和 $\widehat{\text{MCR}}^-(\epsilon)$ 形成合理点估计的条件。在下面第9.1节和第10节中，我们应用自助法来解释抽样变异性。

为了推导这些结果，我们引入三个有界损失假设，每个都可以在经验上进行评估。令 $B_{\text{orig}}, B_{\text{ind}}, B_{\text{ref}}, B_{\text{switch}} \in \mathbb{R}$ 为已知常数。

**假设1（有界个体损失）** 对于给定模型 $f \in \mathcal{F}$，假设对任意 $(y, x_1, x_2) \in (\mathcal{Y} \times \mathcal{X}_1 \times \mathcal{X}_2)$ 有 $0 \le L(f, (y, x_1, x_2)) \le B_{\text{ind}}$。

**假设2（有界相对损失）** 对于给定模型 $f \in \mathcal{F}$，假设对任意 $(y, x_1, x_2) \in \mathcal{Z}$ 有 $|L(f, (y, x_1, x_2)) - L(f_{\text{ref}}, (y, x_1, x_2))| \le B_{\text{ref}}$。

**假设3（有界聚合损失）** 对于给定模型 $f \in \mathcal{F}$，假设 $\mathbb{P}\{0 < b_{\text{orig}} \le \widehat{e}_{\text{orig}}(f)\} = \mathbb{P}\{\widehat{e}_{\text{switch}}(f) \le B_{\text{switch}}\} = 1$。

每个假设都是特定模型 $f \in \mathcal{F}$ 的性质。符号 $B_{\text{ind}}$ 和 $B_{\text{ref}}$ 指任意单个观测的界，符号 $b_{\text{orig}}$ 和 $B_{\text{switch}}$ 指样本中聚合损失 $L$ 的界。这些有界性假设对我们下面的有限样本保证至关重要。

关键地，通常无界的损失函数 $L$ 只要 $L(f, (y, x_1, x_2))$ 在特定域上有界就可以使用。例如，如果 $Y$ 包含在已知范围内，且预测 $f(x_1, x_2)$ 对于 $(x_1, x_2) \in \mathcal{X}_1 \times \mathcal{X}_2$ 包含在同一范围内，则可以使用平方误差损失。我们在第7.3.2节和7.4.2节给出了确定 $B_{\text{ind}}$ 的示例方法。对于假设3，我们可以通过对数据训练高度灵活的模型，并将 $b_{\text{orig}}$ 设置为结果交叉验证损失的一半（或任何正分数）来近似 $b_{\text{orig}}$。为确定 $B_{\text{switch}}$，我们可以简单设置 $B_{\text{switch}} = B_{\text{ind}}$，尽管这可能保守。例如，在不可分组的二元分类模型情况下（见第9.1节），没有线性分类器可以错误分类所有观测，特别是在协变量被排列之后。因此，必须有 $B_{\text{ind}} > B_{\text{switch}}$。类似地，如果 $f_{\text{ref}}$ 满足假设1，那么 $B_{\text{ref}}$ 可以保守地设置为 $B_{\text{ind}}$。如果模型依赖度被重新定义为差值而非比率，则本节结果的类似形式将在没有假设3的情况下适用（见附录A.5）。

基于这些假设，我们可以创建 $\text{MCR}^+(\epsilon)$ 的有限样本上界和 $\text{MCR}^-(\epsilon)$ 的下界。换句话说，我们创建一个以高概率包含区间 $[\text{MCR}^-(\epsilon), \text{MCR}^+(\epsilon)]$ 的"外部"界。

**定理4（"外部"MCR界）** 给定常数 $\epsilon \ge 0$，令 $f^{+,\epsilon} \in \arg\max_{\mathcal{R}(\epsilon)} \text{MR}(f)$ 和 $f^{-,\epsilon} \in \arg\min_{\mathcal{R}(\epsilon)} \text{MR}(f)$ 为在 $\mathcal{R}(\epsilon)$ 中达到最高和最低模型依赖度的预测模型。如果 $f^{+,\epsilon}$ 和 $f^{-,\epsilon}$ 满足假设1、2和3，则

$$
\mathbb{P}\left(\text{MCR}^+(\epsilon) > \widehat{\text{MCR}}^+(\epsilon_{\text{out}}) + Q_{\text{out}}\right) \le \delta, \quad \text{且} \qquad (4.2)
$$

$$
\mathbb{P}\left(\text{MCR}^-(\epsilon) < \widehat{\text{MCR}}^-(\epsilon_{\text{out}}) - Q_{\text{out}}\right) \le \delta, \qquad (4.3)
$$

其中 $\epsilon_{\text{out}} := \epsilon + 2B_{\text{ref}} \sqrt{\frac{\log(3\delta^{-1})}{2n}}$，且

$$
Q_{\text{out}} := \frac{B_{\text{switch}}}{b_{\text{orig}}} - \frac{B_{\text{switch}} - B_{\text{ind}} \sqrt{\frac{\log(6\delta^{-1})}{n}}}{b_{\text{orig}} + B_{\text{ind}} \sqrt{\frac{\log(6\delta^{-1})}{2n}}}.
$$

式4.2指出，以高概率，$\text{MCR}^+(\epsilon)$ 不高于 $\widehat{\text{MCR}}^+(\epsilon_{\text{out}})$ 加上误差项 $Q_{\text{out}}$。随着 $n$ 增加，$\epsilon_{\text{out}}$ 趋近于 $\epsilon$，$Q_{\text{out}}$ 趋近于零。一个实际的含义是，粗略地说，如果 $\widehat{\text{MCR}}^+(\epsilon) \approx \widehat{\text{MCR}}^+(\epsilon_{\text{out}})$，则经验估计量 $\widehat{\text{MCR}}^+(\epsilon)$ 不太可能显著低估 $\text{MCR}^+(\epsilon)$。通过类似的推理，我们可以从式4.3得出结论，如果 $\widehat{\text{MCR}}^-(\epsilon) \approx \widehat{\text{MCR}}^-(\epsilon_{\text{out}})$，则 $\widehat{\text{MCR}}^-(\epsilon)$ 不太可能显著高估 $\text{MCR}^-(\epsilon)$。通过设置 $\epsilon = 0$，定理4也可用于创建对唯一的（未知的）同类最佳模型在 $X_1$ 上的依赖度的有限样本界（见附录A.4中的推论22），尽管描述单个模型并非本文的主要焦点。

我们在图3中提供了定理4的视觉图示。证明概要如下。首先，我们通过将 $\epsilon$ 增加到 $\epsilon_{\text{out}}$ 来扩大经验 $\epsilon$-罗生门集，使得通过 Hoeffding 不等式，$f^{+,\epsilon} \in \widehat{\mathcal{R}}(\epsilon_{\text{out}})$ 以高概率成立。当 $f^{+,\epsilon} \in \widehat{\mathcal{R}}(\epsilon_{\text{out}})$ 时，我们知道 $\widehat{\text{MR}}(f^{+,\epsilon}) \le \widehat{\text{MCR}}^+(\epsilon_{\text{out}})$，根据 $\widehat{\text{MCR}}^+(\epsilon_{\text{out}})$ 的定义。接下来，$Q_{\text{out}}$ 项利用 U-统计量的有限样结果来考虑使用估计量 $\widehat{\text{MR}}(f^{+,\epsilon})$ 估计 $\text{MR}(f^{+,\epsilon}) = \text{MCR}^+(\epsilon)$ 的误差。因此，我们可以将 $\widehat{\text{MR}}(f^{+,\epsilon})$ 与 $\widehat{\text{MCR}}^+(\epsilon_{\text{out}})$ 和 $\text{MCR}^+(\epsilon)$ 联系起来，以获得式4.2。类似的步骤可应用于获得式4.3。

定理4中的界自然地考虑了潜在的过拟合，而不需要显式限制模型类复杂度（如覆盖数、Rademacher 复杂度或 VC 维度）。相反，这些界依赖于能够完全优化 $\widehat{\mathcal{R}}(\epsilon)$ 形式集合上的 MR。如果我们允许模型类 $\mathcal{F}$ 变得更加灵活，那么 $\widehat{\mathcal{R}}(\epsilon)$ 的大小也会增加。因为定理4中的界来自在 $\widehat{\mathcal{R}}(\epsilon)$ 上的优化，增加 $\widehat{\mathcal{R}}(\epsilon)$ 的大小会导致更宽、更保守的界。通过这种方式，式4.2和4.3隐式地捕获了模型类的复杂度。

到目前为止，定理4让我们限定了与预测良好的模型对应的 MR 值范围，但它没有告诉我们这些界是否实际上被达到。类似地，我们可以从定理4得出结论，$[\text{MCR}^-(\epsilon), \text{MCR}^+(\epsilon)]$ 不太可能超过估计范围 $[\widehat{\text{MCR}}^-(\epsilon), \widehat{\text{MCR}}^+(\epsilon)]$ 太多，但我们无法确定这个估计范围是否不必要地宽。

例如，考虑驱动 $\widehat{\text{MCR}}^+(\epsilon)$ 估计量的模型：那些具有强样本内精度且对 $X_1$ 有高经验依赖的模型。这些模型的样本内性能可能仅仅是过拟合的结果，在这种情况下它们并不直接告诉我们关于 $\mathcal{R}(\epsilon)$ 的信息。或者，即使所有这些模型在期望意义上确实表现良好（即即使它们包含在 $\mathcal{R}(\epsilon)$ 中），对 $X_1$ 具有最高经验依赖的模型可能仅是我们的经验 MR 估计包含最多误差的那个模型。这两种场景都可能导致 $\widehat{\text{MCR}}^+(\epsilon)$ 相对于 $\text{MCR}^+(\epsilon)$ 不必要地高。

幸运的是，这两个有问题的场景都通过要求对 $\mathcal{F}$ 的复杂度进行限制来解决。我们提出以覆盖数形式的复杂度度量，它允许我们控制过拟合或 MR 估计误差的最坏情况。具体地，如果对于任何 $f \in \mathcal{F}$ 和任何分布 $D$，存在 $g \in G_r$ 使得 $\mathbb{E}_{Z\sim D} |L(f, Z) - L(g, Z)| \le r$，则我们将函数集 $G_r$ 定义为 $r$-边际期望覆盖。我们将覆盖数 $\mathcal{N}(\mathcal{F}, r)$ 定义为 $\mathcal{F}$ 的最小 $r$-边际期望覆盖的大小。一般情况下，我们用 $\mathbb{P}_{V\sim D}$ 和 $\mathbb{E}_{V\sim D}$ 表示关于遵循分布 $D$ 的随机变量 $V$ 的概率和期望。当 $V$ 或 $D$ 从上下文中清晰时，我们相应地进行缩写，例如 $\mathbb{P}_D$、$\mathbb{P}_V$ 或简写为 $\mathbb{P}$。除非另有说明，所有期望和概率都是关于（未知的）总体分布。

我们首先展示这个复杂度度量允许我们控制最坏情况的 MR 估计误差，即覆盖数 $\mathcal{N}(\mathcal{F}, r)$ 为所有 $f \in \mathcal{F}$ 提供 $\widehat{\text{MR}}(f)$ 误差的一致界。

**定理5（$\widehat{\text{MR}}$ 的一致界）** 给定 $r > 0$，如果假设1和3对所有 $f \in \mathcal{F}$ 成立，则

$$
\mathbb{P}\left( \sup_{f \in \mathcal{F}} \left| \widehat{\text{MR}}(f) - \text{MR}(f) \right| > q(\delta, r, n) \right) \le \delta, \qquad (4.5)
$$

其中

$$
q(\delta, r, n) := \frac{B_{\text{switch}}}{b_{\text{orig}}} - \frac{B_{\text{switch}} - \left( B_{\text{ind}} \sqrt{\frac{\log(4\delta^{-1}\mathcal{N}(\mathcal{F}, r\sqrt{2}))}{n}} + 2r\sqrt{2} \right)}{b_{\text{orig}} + \left( B_{\text{ind}} \sqrt{\frac{\log(4\delta^{-1}\mathcal{N}(\mathcal{F}, r))}{2n}} + 2r \right)}.
$$

定理5表明，以高概率，$\mathcal{F}$ 中所有模型的 $\text{MR}(f)$ 的最大可能估计误差由 $q(\delta, r, n)$ 界定，后者可以通过增加 $n$ 和减小 $r$ 来变得任意小。正如我们在第3.1节中指出的，这意味着可以在同一数据上训练模型并估计其对变量的依赖度，而无需使用样本分割。

覆盖数 $\mathcal{N}(\mathcal{F}, r)$ 也可用于限制过拟合的程度（见附录B.5.1）。因此，可以设置一个足够低的样本内性能阈值，使其仅被具有强预期性能的模型满足（即真正在 $\mathcal{R}(\epsilon)$ 内的模型）。为了实现这种更严格的性能阈值的思想，我们通过从 $\epsilon$ 中减去一个缓冲项来收缩经验 $\epsilon$-罗生门集。这要求我们将经验 $\epsilon$-罗生门集的定义推广到 $\widehat{\mathcal{R}}(\epsilon, f_{\text{ref}}, \mathcal{F}) := \{f_{\text{ref}}\} \cup \{f \in \mathcal{F} : \widehat{e}_{\text{orig}}(f) \le \widehat{e}_{\text{orig}}(f_{\text{ref}}) + \epsilon\}$，其中 $\epsilon \in \mathbb{R}$，显式包含 $f_{\text{ref}}$ 现在确保了即使对于 $\epsilon < 0$，$\widehat{\mathcal{R}}(\epsilon, f_{\text{ref}}, \mathcal{F})$ 也是非空的。如前所述，我们通常省略符号 $f_{\text{ref}}$ 和 $\mathcal{F}$，简写为 $\widehat{\mathcal{R}}(\epsilon)$。

现在我们已经准备好回答定理4中的界是否实际上被达到，以及估计范围 $[\widehat{\text{MCR}}^-(\epsilon), \widehat{\text{MCR}}^+(\epsilon)]$ 是否不必要地宽这两个问题。我们的回答以 $\text{MCR}^-(\epsilon)$ 的上界和 $\text{MCR}^+(\epsilon)$ 的下界的形式给出。

**定理6（"内部"MCR界）** 给定常数 $\epsilon \ge 0$ 和 $r > 0$，如果假设1、2和3对所有 $f \in \mathcal{F}$ 成立，则

$$
\mathbb{P}\left( \text{MCR}^+(\epsilon) < \widehat{\text{MCR}}^+(\epsilon_{\text{in}}) - Q_{\text{in}} \right) \le \delta, \quad \text{且} \qquad (4.6)
$$

$$
\mathbb{P}\left( \text{MCR}^-(\epsilon) > \widehat{\text{MCR}}^-(\epsilon_{\text{in}}) + Q_{\text{in}} \right) \le \delta, \qquad (4.7)
$$

其中 $\epsilon_{\text{in}} := \epsilon - 2B_{\text{ref}} \sqrt{\frac{\log(4\delta^{-1}\mathcal{N}(\mathcal{F}, r))}{2n}} - 2r$，且 $Q_{\text{in}} = q\left( \frac{\delta}{2}, r, n \right)$，如式4.5所定义。

定理6可以让我们推断出一个以高概率包含在区间 $[\text{MCR}^-(\epsilon), \text{MCR}^+(\epsilon)]$ 内的"内部"界。在图3中，我们展示了定理6的结果，并给出了证明的概要。这个证明遵循与定理4类似的结构，但结合了定理5的 MR 估计误差的一致界（$Q_{\text{in}}$ 项），以及任何模型的样本内损失与其期望损失相差太远的概率的一致界（$\epsilon_{\text{in}}$ 项）。

定理6的一个实际含义是，粗略地说，如果 $\widehat{\text{MCR}}^+(\epsilon_{\text{in}}) \approx \widehat{\text{MCR}}^+(\epsilon)$，则经验估计量 $\widehat{\text{MCR}}^+(\epsilon)$ 不太可能显著低估 $\text{MCR}^+(\epsilon)$。结合定理4，我们可以得出结论，如果 $\widehat{\text{MCR}}^+(\epsilon_{\text{in}}) \approx \widehat{\text{MCR}}^+(\epsilon_{\text{out}})$，则估计量 $\widehat{\text{MCR}}^+(\epsilon)$ 不太可能显著高估或低估 $\text{MCR}^+(\epsilon)$。在大样本中，可以预期条件 $\widehat{\text{MCR}}^+(\epsilon_{\text{in}}) \approx \widehat{\text{MCR}}^+(\epsilon_{\text{out}})$ 成立，因为随着 $n$ 增加，$\epsilon_{\text{in}}$ 和 $\epsilon_{\text{out}}$ 都趋近于 $\epsilon$。同样地，如果 $\widehat{\text{MCR}}^-(\epsilon_{\text{in}}) \approx \widehat{\text{MCR}}^-(\epsilon_{\text{out}})$，我们可以从式4.3和4.7得出结论，经验估计量 $\widehat{\text{MCR}}^-(\epsilon)$ 不太可能显著高估或低估 $\text{MCR}^-(\epsilon)$。出于这个原因，我们认为 $\widehat{\text{MCR}}^-(\epsilon)$ 和 $\widehat{\text{MCR}}^+(\epsilon)$ 形成了总体水平 MCR 的合理估计——每个都以高概率包含在其各自目标估计量的邻域内。图3的辅助 x 轴展示了对这一论证的图示。

**图3：定理4和6中各项的图示**——上图展示了假想模型类 $\mathcal{F}$ 中模型的 $\widehat{\text{MR}}$（x轴）与经验损失（y轴）之间的关系。我们用黑色点标记 $f_{\text{ref}}$。对于每个可能的模型依赖值 $r \ge 0$，弯曲虚线显示了 $f \in \mathcal{F}$ 中满足 $\widehat{\text{MR}}(f) = r$ 的函数的最低可能经验损失。集合 $\widehat{\mathcal{R}}(\epsilon)$ 包含在虚线灰色线内的 $\mathcal{F}$ 中的所有模型。为了创建定理4的界，我们将经验 $\epsilon$-罗生门集扩大，将 $\epsilon$ 增加到 $\epsilon_{\text{out}}$，使得 $f^{+,\epsilon}$（或 $f^{-,\epsilon}$）以高概率包含在 $\widehat{\mathcal{R}}(\epsilon_{\text{out}})$ 中。然后我们加上（或减去）$Q_{\text{out}}$ 以考虑 $\widehat{\text{MR}}(f^{+,\epsilon})$（或 $\widehat{\text{MR}}(f^{-,\epsilon})$）的估计误差。这些步骤上面以蓝色显示，最终界由 x 轴上的蓝色括号符号显示。为了创建定理6中 $\text{MCR}^+(\epsilon)$（和 $\text{MCR}^-(\epsilon)$）的界，我们通过将 $\epsilon$ 减小到 $\epsilon_{\text{in}}$ 来收缩经验 $\epsilon$-罗生门集，使得所有具有高期望损失的模型同时以高概率被排除在 $\widehat{\mathcal{R}}(\epsilon_{\text{in}})$ 之外。然后我们减去（或加上）$Q_{\text{in}}$ 以同时考虑 $\widehat{\mathcal{R}}(\epsilon_{\text{in}})$ 中模型的 MR 估计误差。这些步骤上面以紫色显示，最终界由 x 轴上的紫色括号符号显示。为了强调，在这个图下方我们显示了一个带有选定注释的 x 轴副本，从中可以清楚地看到 $\widehat{\text{MCR}}^-(\epsilon)$ 和 $\widehat{\text{MCR}}^+(\epsilon)$ 始终在定理4和6产生的界之内。以高概率，$\widehat{\text{MCR}}^-(\epsilon)$ 和 $\widehat{\text{MCR}}^+(\epsilon)$ 分别在 $\text{MCR}^-(\epsilon)$ 和 $\text{MCR}^+(\epsilon)$ 的邻域内。

## 5. 罗生门集在变量重要性之外的扩展

在本节中，我们将罗生门集方法推广到 MR 研究之外。在第5.1节中，我们为接近最优或同类最佳模型的其他汇总特征创建了有限样本置信区间。这一推广也有助于说明定理4论证的核心方面：总体中具有接近最优性能的模型往往在随机样本中具有相对良好的性能。

在第5.2节中，我们回顾了关于接近最优模型的现有文献。

### 5.1 来自罗生门集的有限样本置信区间

这里我们不描述模型对 $X_1$ 的依赖程度，而是假设分析师对模型的任意特征感兴趣。我们将这个感兴趣的特征记为 $\phi : \mathcal{F} \to \mathbb{R}$。例如，如果 $f_\beta$ 是线性模型 $f_\beta(\mathbf{x}) = \mathbf{x}'\beta$，那么 $\phi$ 可以定义为关联系数向量的范数（即 $\phi(f_\beta) = \|\beta\|_2^2$）或给定特定协变量分布 $\mathbf{x}_{\text{new}}$ 时 $f_\beta$ 将分配的预测值（即 $\phi(f_\beta) = f_\beta(\mathbf{x}_{\text{new}})$）。

给定描述符 $\phi$，我们现在展示一个通用结果，允许为最佳性能模型 $\mathcal{R}(\epsilon)$ 创建有限样本置信区间。产生的置信区间本身基于经验罗生门集。

**命题7（来自罗生门集的有限样本置信区间）** 令 $\epsilon' := \epsilon + 2B_{\text{ref}} \sqrt{\frac{\log(2\delta^{-1})}{2n}}$，令 $\widehat{\phi}^-(\epsilon') := \min_{f \in \widehat{\mathcal{R}}(\epsilon')} \phi(f)$，令 $\widehat{\phi}^+(\epsilon') := \max_{f \in \widehat{\mathcal{R}}(\epsilon')} \phi(f)$。如果假设2对所有 $f \in \mathcal{R}(\epsilon)$ 成立，则

$$
\mathbb{P}\left[ \{\phi(f) : f \in \mathcal{R}(\epsilon)\} \subseteq \left[ \widehat{\phi}^-(\epsilon'), \widehat{\phi}^+(\epsilon') \right] \right] \ge 1 - \delta.
$$

命题7为表现良好模型对应的 $\phi(f)$ 值范围 $\{\phi(f) : f \in \mathcal{R}(\epsilon)\}$ 生成了一个有限样本置信区间。这个置信区间 $[\widehat{\phi}^-(\epsilon'), \widehat{\phi}^+(\epsilon')]$ 本身可以被解释为经验损失不超过 $f_{\text{ref}}$ 太多的模型 $f$ 对应的 $\phi(f)$ 值范围。因此，该区间既具有严格的覆盖概率，又具有一致的样本内解释。命题7的证明使用 Hoeffding 不等式来证明 $\mathcal{F}$ 中的模型以高概率包含在 $\widehat{\mathcal{R}}(\epsilon')$ 中，即具有良好预期性能的模型往往在随机样本中表现良好。

命题7的一个直接推论是，我们可以通过设置 $\epsilon = 0$ 为所有同类最佳模型 $f^\star \in \arg\min_{f \in \mathcal{F}} \mathbb{E}L(f, Z)$ 生成有限样本置信区间。如果假设单个模型 $f^\star$ 唯一最小化 $\mathbb{E}L(f, Z)$ 对 $f \in \mathcal{F}$，这个推论可以进一步加强（见附录B.6）。

注意，命题7隐式地假设 $\phi(f)$ 可以精确确定对于任何 $f \in \mathcal{F}$，以便区间 $[\widehat{\phi}^-(\epsilon'), \widehat{\phi}^+(\epsilon')]$ 可以被精确确定。例如，如果 $\phi(f) = \text{MR}(f)$，或 $\phi(f) = \text{Var}\{f(X_1, X_2)\}$，这个假设不成立，因为这些量同时依赖于 $f$ 和（未知的）总体分布。在这种情况下，必须结合一个额外的校正因子来考虑 $\phi(f)$ 的估计误差，如定理4中的 $Q_{\text{out}}$ 项。

在并行工作 [20] 中，Coker et al. 展示了轮廓似然区间与命题7中的区间 $[\widehat{\phi}^-(\epsilon'), \widehat{\phi}^+(\epsilon')]$ 具有相同的形式。这意味着轮廓似然区间也可以通过最小化和最大化经验罗生门集来表达。更具体地，考虑损失函数 $L$ 是已知对数似然函数的负数，且 $f_{\text{ref}}$ 是"真实模型"的最大似然估计（在这种情况下为 $f^\star$）的情况。如果满足额外的次要假设（详见附录A.6），则 $\phi(f^\star)$ 的 $(1-\delta)$-水平轮廓似然区间等于 $[\widehat{\phi}^-(\frac{\chi_{1,1-\delta}}{2n}), \widehat{\phi}^+(\frac{\chi_{1,1-\delta}}{2n})]$，其中 $\widehat{\phi}^-$ 和 $\widehat{\phi}^+$ 的定义如命题7，$\chi_{1,1-\delta}$ 是自由度为1的卡方分布的第 $1-\delta$ 百分位数。

相对于轮廓似然方法，命题7的优势在于它不需要渐近理论、不需要似然函数已知到参数形式，并且可以扩展到研究接近最优的预测模型集合 $\mathcal{R}(\epsilon)$，而不是单个可能被错误设定的预测模型 $f^\star$。当不同的接近最优模型准确描述底层数据生成过程的不同方面，但没有任何一个完全捕获它时，这一点尤其有用。命题7的缺点是所需的性能阈值 $\epsilon' = \epsilon + 2B_{\text{ref}} \sqrt{\frac{\log(2\delta^{-1})}{2n}}$ 的下降速度慢于轮廓似然区间所需的性能阈值 $\frac{\chi_{1,1-\delta}}{2n}$。因为第4.1节的结果有类似的缺点，我们主要用这些结果来启发描述罗生门集 $\mathcal{R}(\epsilon)$ 的点估计。

尽管如此，值得强调的是命题7的通用性。通过这个结果，罗生门集允许我们将广泛的有限样本推断问题重新框架为样本内优化问题。隐含的置信区间不一定具有闭式形式，但该方法仍然为推导非渐近结果开辟了令人兴奋的途径。例如，它们意味着现有的轮廓似然区间方法可能能够被重新应用以获得有限样结果。对于轮廓似然难以计算的高度复杂模型类，例如神经网络或随机森林，有时通过近似优化过程来实现近似推断（例如，Chipman et al. [21] 中 Bayesian 加性回归树的马尔可夫链蒙特卡洛）。命题7表明，类似的近似优化方法可以被重新用于为相同的模型类建立近似的有限样本推断。

### 5.2 关于罗生门效应的相关文献

Breiman et al. [8] 引入了统计学的"罗生门效应"作为一个歧义性问题：如果许多模型都良好拟合数据，不清楚我们应该尝试解释哪个模型。Breiman 建议将许多表现良好的模型集成在一起可以解决这种歧义性，因为新的集成模型可能比其任何单个成员性能更好。然而，这种方法可能只是将问题从成员级别推到了集成级别，因为也可能有许多不同的集成模型很好地拟合数据。

罗生门效应在 VI 之外的一些学科领域也被考虑过，包括非统计学术学科 [37,87]。Tulabandhula and Rudin [100] 优化决策规则以在来自任何表现良好模型的预测结果范围内表现良好。Statnikov et al. [93] 提出了一种算法来发现多个马尔可夫边界，即协变量的最小集合，使得以其中任何一个为条件都会在结果和剩余协变量之间诱导独立性。Nevo and Ritov [74] 报告了一组良好拟合的稀疏线性模型对应的解释。Meinshausen and B\"uhlmann [65] 基于结构方面在一组良好拟合模型中的稳定程度来估计底层模型的结构方面（如该模型中包含的变量）。这组良好拟合的模型通过在一系列扰动样本中重复估计过程并使用不同水平的正则化来识别（另见 [3]）。Letham et al. [62] 搜索一对良好拟合的动力系统模型，使它们给出最大程度不同的预测。

## 6. 计算模型类依赖度的经验估计

在本节中，我们提出一个二分搜索过程来界定 $\widehat{\text{MCR}}^-(\epsilon)$ 和 $\widehat{\text{MCR}}^+(\epsilon)$ 的值（见式2.4），它们分别作为 $\text{MCR}^-(\epsilon)$ 和 $\text{MCR}^+(\epsilon)$ 的估计（见第4.1节）。该搜索的每一步都由在 $\mathcal{F}$ 上最小化 $\widehat{e}_{\text{orig}}(f)$ 和 $\widehat{e}_{\text{switch}}(f)$ 的线性组合组成。我们的方法与 Dinkelbach [30] 的分式规划方法相关，但考虑了问题受分母 $\widehat{e}_{\text{orig}}(f)$ 的值约束这一事实。我们还展示了，对于许多模型类，计算 $\widehat{\text{MCR}}^-(\epsilon)$ 只需要最小化 $\widehat{e}_{\text{orig}}(f)$ 和 $\widehat{e}_{\text{switch}}(f)$ 的凸组合，这并不比在扩展和重新加权的样本上最小化平均损失更难（见式6.2和命题11）。然而，计算 $\widehat{\text{MCR}}^+(\epsilon)$ 将要求我们能够最小化 $\widehat{e}_{\text{orig}}(f)$ 和 $\widehat{e}_{\text{switch}}(f)$ 的任意线性组合。在第6.3节中，我们概述了这对于凸模型类（损失函数在模型参数中为凸）如何实现。之后，在第7节中，我们给出当 $\mathcal{F}$ 是线性模型、正则化线性模型或再生核希尔伯特空间中的线性模型的类别时更具体的计算过程。我们在表1中总结了不同模型类的经验 MCR 可计算性。

为简化与参考模型 $f_{\text{ref}}$ 相关的符号，我们根据受绝对尺度上的性能阈值约束的经验 MR 界来呈现计算结果。更具体地，我们呈现满足 $b^-(\epsilon_{\text{abs}}) \le \widehat{\text{MR}}(f) \le b^+(\epsilon_{\text{abs}})$ 同时对所有的 $\{f, \epsilon_{\text{abs}} : \widehat{e}_{\text{orig}}(f) \le \epsilon_{\text{abs}}, f \in \mathcal{F}, \epsilon_{\text{abs}} > 0\}$ 成立的界函数 $b^-$ 和 $b^+$（图2和8展示了这些界的示例）。我们提出的二分搜索过程可用于在感兴趣的特定 $\epsilon_{\text{abs}}$ 值处收紧这些边界。

我们简要指出，作为下面讨论的全局优化过程的替代方案，启发式优化过程如模拟退火在界定经验 MCR 方面也可能是有用的。根据定义，$\widehat{\mathcal{R}}(\epsilon)$ 中任何模型的 $\widehat{\text{MR}}(f)$ 构成 $\widehat{\text{MCR}}^+(\epsilon)$ 的下界和 $\widehat{\text{MCR}}^-(\epsilon)$ 的上界。经验 MR 的启发式最大化和最小化可用于收紧这些边界。

在本节中，我们假设 $0 < \min_{f \in \mathcal{F}} \widehat{e}_{\text{orig}}(f)$，以确保 MR 是有限的。

**表1：不同模型类的经验 MCR 可计算性**——对于每种情况，我们描述了使用我们提出的方法计算 $\widehat{\text{MCR}}^-$ 和 $\widehat{\text{MCR}}^+$ 的可计算性。计算经验 MCR 可以简化为一系列优化问题，其形式在上述表格的括号中注明。

| 模型类和损失函数（$\mathcal{F}$ 和 $L$） | 计算 $\widehat{\text{MCR}}^-$ | 计算 $\widehat{\text{MCR}}^+$ |
|---|---|---|
|（$L_2$ 正则化）线性模型，平方误差损失 | 高度可处理（QP1QC，见第7.2和7.3节） | 高度可处理（QP1QC，见第7.2和7.3节） |
| 再生核希尔伯特空间中的线性模型，平方误差损失 | 中度可处理（QP1QC，见第7.4.1节） | 中度可处理（QP1QC，见第7.4.1节） |
| 不相关协变量不改善预测的情况 | 中度可处理（凸优化问题，见命题11） | 可能不可处理 |
| 最小化经验损失是凸优化问题的情况 | 可能不可处理（DC 规划，见第6.3节） | 可能不可处理（DC 规划，见第6.3节） |

### 6.1 经验 MR 下界的二分搜索

在描述我们的二分搜索过程之前，我们引入本节中使用的新符号。给定常数 $\gamma \in \mathbb{R}$ 和预测模型 $f \in \mathcal{F}$，我们定义线性组合 $\widehat{h}^{-,\gamma}$ 及其最小化器（例如 $\widehat{g}^{-,\gamma,\mathcal{F}}$）为

$$
\widehat{h}^{-,\gamma}(f) := \gamma \widehat{e}_{\text{orig}}(f) + \widehat{e}_{\text{switch}}(f), \qquad \widehat{g}^{-,\gamma,\mathcal{F}} \in \arg\min_{f \in \mathcal{F}} \widehat{h}^{-,\gamma}(f). \qquad (6.1)
$$

我们不要求 $\widehat{h}^{-,\gamma}$ 被唯一最小化，并且在 $\mathcal{F}$ 从上下文中清晰时经常使用缩写符号 $\widehat{g}^{-,\gamma}$。

本节的目标是推导形如 $\{f \in \mathcal{F} : \widehat{e}_{\text{orig}}(f) \le \epsilon_{\text{abs}}\}$ 的 $\mathcal{F}$ 子集上的 $\widehat{\text{MR}}$ 下界。我们通过最小化一系列形如 $\widehat{h}^{-,\gamma}$ 的线性目标函数来实现，使用类似于 Dinkelbach [30] 的方法。通常，最小化线性组合 $\widehat{h}^{-,\gamma}(f)$ 比直接最小化 MR 比率更易处理。

本节和6.2节中的几乎所有结果，如果我们在 $\widehat{\text{MR}}$ 和 $\widehat{h}^{-,\gamma}(f)$ 的定义中整体用 $\widehat{e}_{\text{divide}}$ 替换 $\widehat{e}_{\text{switch}}$（见式3.5），仍然成立。唯一的例外是下面的命题11，如果我们用 $\widehat{e}_{\text{divide}}$ 替换 $\widehat{e}_{\text{switch}}$，仍可以预期其近似成立。

给定观测样本，我们为值对 $\{\gamma, \epsilon_{\text{abs}}\} \in \mathbb{R} \times \mathbb{R}_{>0}$ 和 $\arg\min$ 函数 $\widehat{g}^{-,\gamma}$ 定义以下条件：

**条件8（继续搜索 $\widehat{\text{MR}}$ 下界的标准）** $\widehat{h}^{-,\gamma}(\widehat{g}^{-,\gamma}) \ge 0$ 且 $\widehat{e}_{\text{orig}}(\widehat{g}^{-,\gamma}) \le \epsilon_{\text{abs}}$。

现在我们有条件确定何时可以可处理地创建经验 MR 的下界。

**引理9（$\widehat{\text{MR}}$ 的下界）** 如果 $\gamma \in \mathbb{R}$ 满足 $\widehat{h}^{-,\gamma}(\widehat{g}^{-,\gamma}) \ge 0$，则

$$
\frac{\widehat{h}^{-,\gamma}(\widehat{g}^{-,\gamma})}{\epsilon_{\text{abs}}} - \gamma \le \widehat{\text{MR}}(f) \qquad (6.2)
$$

对所有满足 $\widehat{e}_{\text{orig}}(f) \le \epsilon_{\text{abs}}$ 的 $f \in \mathcal{F}$ 成立。由此也得出 $-\gamma \le \widehat{\text{MR}}(f)$ 对所有 $f \in \mathcal{F}$ 成立。此外，如果 $f = \widehat{g}^{-,\gamma}$ 且条件8中的至少一个不等式取等号，则式6.1取等号。

引理9将对 $\widehat{\text{MR}}(f)$ 的下界问题简化为最小化线性组合 $\widehat{h}^{-,\gamma}(f)$ 的任务。引理9的结果不仅是一个针对特定 $\epsilon_{\text{abs}}$ 值的单一界，而是一个对所有 $\epsilon_{\text{abs}} > 0$ 都成立的界函数，较低的 $\epsilon_{\text{abs}}$ 值导致对 $\widehat{\text{MR}}(f)$ 更严格的下界。

**图4：引理9的几何直观**——在面板（A）中，我们展示了一个假想的模型类 $\mathcal{F}$ 的示例，以封闭区域标记。对于每个模型 $f \in \mathcal{F}$，x轴显示 $\widehat{e}_{\text{orig}}(f)$，y轴显示 $\widehat{e}_{\text{switch}}(f)$。这里，我们可以看到条件 $\min_{f \in \mathcal{F}} \widehat{e}_{\text{orig}}(f) > 0$ 成立。蓝色虚线区域标记具有更高经验损失的模型。我们在 $\mathcal{F}$ 内标记了两个示例模型 $f_1$ 和 $f_2$。连接原点到 $f_1$ 和 $f_2$ 的线的斜率分别等于 $\widehat{\text{MR}}(f_1)$ 和 $\widehat{\text{MR}}(f_2)$。我们的目标是对满足 $\widehat{e}_{\text{orig}}(f) \le \epsilon_{\text{abs}}$ 的任何模型 $f$ 的 $\widehat{\text{MR}}$ 对应的斜率进行下界界定。在面板（B）中，我们考虑 $\gamma = 1$ 时的线性组合 $\widehat{h}^{-,\gamma}(f) = \gamma \widehat{e}_{\text{orig}}(f) + \widehat{e}_{\text{switch}}(f)$。上面，$\widehat{h}^{-,\gamma}$ 的等高线以红色显示。实线红线表示在 $\mathcal{F}$ 上 $\widehat{h}^{-,\gamma}$ 的最小可能值。具体地，其 y 截距等于 $\min_{f \in \mathcal{F}} \widehat{h}^{-,\gamma}(f)$。如果我们能确定这个最小值，我们就能确定 $\mathcal{F}$ 上的一个线性边界约束，即我们知道没有对应模型 $f \in \mathcal{F}$ 的点可能位于上方的阴影区域中。此外，如果 $\min_{f \in \mathcal{F}} \widehat{h}^{-,\gamma}(f) \ge 0$（见引理9），那么我们知道原点要么被这个线性约束排除，要么在边界上。在面板（C）中，我们结合面板（A）和（B）的两个约束，看到满足 $\widehat{e}_{\text{orig}}(f) \le \epsilon_{\text{abs}}$ 的模型 $f \in \mathcal{F}$ 必须对应于上面白色非阴影区域中的点。因此，只要非阴影区域不包含原点，连接原点到满足 $\widehat{e}_{\text{orig}}(f) \le \epsilon_{\text{abs}}$ 的模型 $f$（例如，这里的 $f_1, f_2$）的任何线的斜率必须至少与上面的黑色实线的斜率一样高。可以在代数上证明黑色线的斜率等于式6.1的左侧。因此，式6.1的左侧是所有 $\{f \in \mathcal{F} : \widehat{e}_{\text{orig}}(f) \le \epsilon_{\text{abs}}\}$ 的 $\widehat{\text{MR}}(f)$ 的下界。

除了引理9的正式证明，我们还在图4中提供结果的启发式说明，以帮助建立直觉。

仍然需要确定在式6.1中应该使用哪个 $\gamma$ 值。以下引理表明，给定感兴趣的特定 $\epsilon_{\text{abs}}$ 值，这个值可以通过二分搜索确定。

**引理10（$\widehat{\text{MR}}$ 下界二分搜索的单调性）** 以下单调性结果成立：

1. $\widehat{h}^{-,\gamma}(\widehat{g}^{-,\gamma})$ 在 $\gamma$ 中单调递增。
2. $\widehat{e}_{\text{orig}}(\widehat{g}^{-,\gamma})$ 在 $\gamma$ 中单调递减。
3. 给定 $\epsilon_{\text{abs}}$，式6.1的下界 $\left\{ \frac{\widehat{h}^{-,\gamma}(\widehat{g}^{-,\gamma})}{\epsilon_{\text{abs}}} - \gamma \right\}$ 在 $\widehat{e}_{\text{orig}}(\widehat{g}^{-,\gamma}) \le \epsilon_{\text{abs}}$ 范围内随 $\gamma$ 单调递减，否则递增。

给定感兴趣的性能水平 $\epsilon_{\text{abs}}$，引理10的第3点告诉我们，在仍满足条件8的情况下，使式6.1产生最严格下界的 $\gamma$ 值尽可能低时发生。第1点和第2点表明，如果 $\gamma_0$ 满足条件8，且条件8中的一个等式取等号，则条件8对所有 $\gamma \ge \gamma_0$ 都成立。综合起来，这些结果意味着我们可以使用二分搜索来确定引理9中使用的 $\gamma$ 值，减少该值直到条件8不再满足。

除了引理10的正式证明，我们还在图5中提供结果的图示以帮助建立直觉。

**图5：二分搜索的单调性**——上面我们展示了对两个替代 $\gamma$ 值的图4-C版本。此图旨在为引理10中的单调性结果补充直观理解，超越正式证明。增加 $\gamma$ 等价于减小图4-C中红线的斜率。我们定义两个值 $\gamma_1 < \gamma_2$，其中 $\gamma_1$ 对应于上面的实线红线，$\gamma_2$ 对应于半透明红线。这些线的 y 截距值分别等于 $\widehat{h}^{-,\gamma_1}(\widehat{g}^{-,\gamma_1})$ 和 $\widehat{h}^{-,\gamma_2}(\widehat{g}^{-,\gamma_2})$（见图4-C的说明）。实心和半透明黑色点分别标记 $\widehat{g}^{-,\gamma_1}$ 和 $\widehat{g}^{-,\gamma_2}$。将 $\gamma_1$ 和 $\gamma_2$ 代入式6.1得到 $\widehat{\text{MR}}$ 的两个下界，分别由实心和半透明黑色线的斜率标记（见图4-C的说明）。我们看到（1）$\widehat{h}^{-,\gamma_1}(\widehat{g}^{-,\gamma_1}) \le \widehat{h}^{-,\gamma_2}(\widehat{g}^{-,\gamma_2})$，（2）$\widehat{e}_{\text{orig}}(\widehat{g}^{-,\gamma_1}) \ge \widehat{e}_{\text{orig}}(\widehat{g}^{-,\gamma_2})$，且（3）当 $\widehat{e}_{\text{orig}}(\widehat{g}^{-,\gamma}) \le \epsilon_{\text{abs}}$ 时，式6.1的左侧随 $\gamma$ 递减。这三个结论在上图中用箭头标记，编号与引理10的列举列表匹配。

接下来，我们给出简单的条件，在这些条件下 $\gamma$ 值的二分搜索可以限制在非负实数线上。这个结果大大扩展了我们方法的计算可行性，因为最小化 $\gamma \ge 0$ 的 $\widehat{h}^{-,\gamma}$ 等价于在大小为 $n^2$ 的扩展样本上最小化重新加权的经验损失：

$$
\widehat{h}^{-,\gamma}(f) = \gamma \widehat{e}_{\text{orig}}(f) + \widehat{e}_{\text{switch}}(f) = \sum_{i=1}^n \sum_{j=1}^n w_\gamma(i,j) L\{f, (\mathbf{y}[i], \mathbf{X}_1[j,\cdot], \mathbf{X}_2[i,\cdot])\} \qquad (6.3),
$$

其中 $w_\gamma(i,j) = \frac{\gamma \mathbf{1}(i=j)}{n} + \frac{\mathbf{1}(i \ne j)}{n(n-1)} \ge 0$。

**命题11（$\widehat{\text{MR}}$ 下界二分搜索的非负权重）** 假设 $L$ 和 $\mathcal{F}$ 满足以下条件。

1.（预测对于计算损失是充分的）损失 $L\{f, (Y, X_1, X_2)\}$ 仅通过预测函数 $f$ 依赖于协变量 $(X_1, X_2)$，即每当 $f(x_1^{(a)}, x_2^{(a)}) = f(x_1^{(b)}, x_2^{(b)})$ 时，$L\{f, (y, x_1^{(a)}, x_2^{(a)})\} = L\{f, (y, x_1^{(b)}, x_2^{(b)})\}$。

2.（不相关信息不改善预测）对于任何满足 $X_1 \perp_D (X_2, Y)$ 的分布 $D$，存在一个函数 $f_D$ 满足 $\mathbb{E}_D L\{f_D, (Y, X_1, X_2)\} = \min_{f \in \mathcal{F}} \mathbb{E}_D L\{f, (Y, X_1, X_2)\}$，且对于任何 $x_1^{(a)}, x_1^{(b)} \in \mathcal{X}_1$ 和 $x_2 \in \mathcal{X}_2$，$f_D(x_1^{(a)}, x_2) = f_D(x_1^{(b)}, x_2)$。

令 $\gamma = 0$。在上述假设下，要么（i）存在一个最小化 $\widehat{h}^{-,0}$ 的函数 $\widehat{g}^{-,0}$ 不满足条件8，要么（ii）$\widehat{e}_{\text{orig}}(\widehat{g}^{-,0}) \le \epsilon_{\text{abs}}$，且对于任何最小化 $\widehat{h}^{-,0}$ 的函数 $\widehat{g}^{-,0}$，$\text{MR}(g^{-,0}) \le 1$。

命题11的含义是，当命题11的条件满足时，$\gamma$ 的搜索区域可以限制在非负实数线上，并且最小化 $\widehat{h}^{-,\gamma}$ 将比在扩展样本上最小化重新加权的经验损失更容易（式6.2）。要理解这一点，回忆对于固定的 $\epsilon_{\text{abs}}$，我们可以通过二分搜索满足条件8的最小 $\gamma$ 值来收紧引理9中的界。如果设置 $\gamma$ 等于0不满足条件8，并且 $\gamma$ 的搜索可以限制在非负实数线上，其中最小化 $\widehat{h}^{-,0}$ 更容易处理（见式6.2）。或者，如果 $\widehat{e}_{\text{orig}}(g^{-,0}) \le \epsilon_{\text{abs}}$ 且 $\text{MR}(g^{-,0}) \le 1$，则我们已经识别出一个表现良好的模型 $g^{-,0}$，其经验 MR 不大于1。对于 $\epsilon_{\text{abs}} = \widehat{e}_{\text{orig}}(f_{\text{ref}}) + \epsilon$，这意味着 $\widehat{\text{MCR}}^-(\epsilon) \le 1$，这对于大多数解释目的来说是足够精确的结论（见附录A.2）。

由于 $\widehat{e}_{\text{divide}}$ 中使用的固定配对结构，如果我们在 $\widehat{h}^{-,\gamma}$、$\widehat{\text{MR}}$ 和 $\widehat{\text{MCR}}^-$ 的定义中整体用 $\widehat{e}_{\text{divide}}$ 替换 $\widehat{e}_{\text{switch}}$，命题11将不一定成立（见附录C.3）。然而，由于 $\widehat{e}_{\text{divide}}$ 近似 $\widehat{e}_{\text{switch}}$，我们可以预期命题11近似成立。式6.1中的界在将 $\widehat{e}_{\text{switch}}$ 替换为 $\widehat{e}_{\text{divide}}$ 并将 $\gamma$ 限制为非负实数时仍然有效，尽管在某些情况下可能不够严格。

### 6.2 经验 MR 上界的二分搜索

我们现在简要介绍对 $\widehat{\text{MR}}$ 进行上界界定的二分搜索过程，这镜像了第6.1节的过程。给定常数 $\gamma \in \mathbb{R}$ 和预测模型 $f \in \mathcal{F}$，我们定义线性组合 $\widehat{h}^{+,\gamma}$ 及其最小化器（例如 $\widehat{g}^{+,\gamma,\mathcal{F}}$）为

$$
\widehat{h}^{+,\gamma}(f) := \widehat{e}_{\text{orig}}(f) + \gamma \widehat{e}_{\text{switch}}(f), \qquad \widehat{g}^{+,\gamma,\mathcal{F}} \in \arg\min_{f \in \mathcal{F}} \widehat{h}^{+,\gamma}(f). \qquad (6.5)
$$

与第6.1节一样，$\widehat{h}^{+,\gamma}$ 不一定被唯一最小化，当 $\mathcal{F}$ 从上下文中清晰时我们通常缩写 $\widehat{g}^{+,\gamma,\mathcal{F}}$ 为 $\widehat{g}^{+,\gamma}$。

给定观测样本，我们为值对 $\{\gamma, \epsilon_{\text{abs}}\} \in \mathbb{R}_{\le 0} \times \mathbb{R}_{>0}$ 和 $\arg\min$ 函数 $\widehat{g}^{+,\gamma}$ 定义以下条件：

**条件12（继续搜索 $\widehat{\text{MR}}$ 上界的标准）** $\widehat{h}^{+,\gamma}(\widehat{g}^{+,\gamma}) \ge 0$ 且 $\widehat{e}_{\text{orig}}(\widehat{g}^{+,\gamma}) \le \epsilon_{\text{abs}}$。

现在我们可以开发一个对 $\widehat{\text{MR}}$ 进行上界界定的过程，如下一个引理所示。

**引理13（$\widehat{\text{MR}}$ 的上界）** 如果 $\gamma \in \mathbb{R}$ 满足 $\gamma \le 0$ 且 $\widehat{h}^{+,\gamma}(\widehat{g}^{+,\gamma}) \ge 0$，则

$$
\widehat{\text{MR}}(f) \le \frac{\left( \frac{\widehat{h}^{+,\gamma}(\widehat{g}^{+,\gamma})}{\epsilon_{\text{abs}}} - 1 \right)}{\gamma^{-1}} \qquad (6.4)
$$

对所有满足 $\widehat{e}_{\text{orig}}(f) \le \epsilon_{\text{abs}}$ 的 $f \in \mathcal{F}$ 成立。由此也得出 $\widehat{\text{MR}}(f) \le |\gamma^{-1}|$ 对所有 $f \in \mathcal{F}$ 成立。此外，如果 $f = \widehat{g}^{+,\gamma}$ 且条件12中的至少一个不等式取等号，则式6.4取等号。

与第6.1节一样，给定感兴趣的 $\epsilon_{\text{abs}} \ge \min_{f \in \mathcal{F}} \widehat{e}_{\text{orig}}(f)$，仍然需要确定在引理13中使用的 $\gamma$ 值。下一个引理告诉我们，当 $\gamma$ 在仍满足条件12的情况下尽可能低时，引理13中的边界最严格。

**引理14（$\widehat{\text{MR}}$ 上界二分搜索的单调性）** 以下单调性结果成立：

1. $\widehat{h}^{+,\gamma}(\widehat{g}^{+,\gamma})$ 在 $\gamma$ 中单调递增。
2. 对于 $\gamma \le 0$，$\widehat{e}_{\text{orig}}(\widehat{g}^{+,\gamma})$ 在 $\gamma$ 中单调递减，且条件12对 $\gamma = 0$ 和 $\epsilon_{\text{abs}} \ge \min_{f \in \mathcal{F}} \widehat{e}_{\text{orig}}(f)$ 成立。
3. 给定 $\epsilon_{\text{abs}}$，上界 $\left\{ \frac{\widehat{h}^{+,\gamma}(\widehat{g}^{+,\gamma})}{\epsilon_{\text{abs}}} - 1 \right\} \gamma^{-1}$ 在 $\widehat{e}_{\text{orig}}(\widehat{g}^{+,\gamma}) \le \epsilon_{\text{abs}}$ 且 $\gamma < 0$ 范围内随 $\gamma$ 单调递增，在 $\widehat{e}_{\text{orig}}(\widehat{g}^{+,\gamma}) > \epsilon_{\text{abs}}$ 且 $\gamma < 0$ 范围内递减。

综合起来，引理14的结果意味着我们可以使用 $\gamma \in \mathbb{R}$ 上的二分搜索来收紧引理13中 $\widehat{\text{MR}}$ 的边界。

### 6.3 凸模型

在本节中，我们展示当损失函数在模型参数中是凸的时——即当模型 $f_\theta \in \mathcal{F}$ 由 $d$ 维参数 $\theta \in \Theta \subseteq \mathbb{R}^d$ 索引，且损失函数 $L(f_\theta, (y, x_1, x_2))$ 对所有 $(x_1, x_2, y) \in \mathcal{X}_1 \times (\mathcal{X}_2, \mathcal{Y})$ 在 $\theta$ 中为凸时——经验 MCR 可以被保守地计算。

幸运的是，引理9和引理13都不需要 $\widehat{h}^{-,\gamma}$ 或 $\widehat{h}^{+,\gamma}$ 的精确最小值。对于引理9，$\widehat{h}^{-,\gamma}$ 的任何下界足以确定 $\text{MR}(f)$ 的下界。同样地，对于引理13，$\widehat{h}^{+,\gamma}$ 的任何下界足以确定 $\text{MR}(f)$ 的上界。

为了找到这些下界，我们注意到对于"凸"模型类（如上定义），第6.1和6.2节中的优化问题可以写成凸优化问题或凸差函数（DC）规划。DC 规划是可以写成 $\min_{\{\theta : c_{\text{DC}}(\theta) \le k, \theta \in \Theta\}} g_{\text{DC}}(\theta) - h_{\text{DC}}(\theta)$ 的问题，其中 $c_{\text{DC}}$ 是约束函数，$k \in \mathbb{R}^1$，且 $g_{\text{DC}}$、$h_{\text{DC}}$ 和 $c_{\text{DC}}$ 是凸的。虽然 DC 问题的精确解并不总是可处理的，但可以通过分支定界（B&B）方法获得下界 [44]。一种简单的 B&B 方法是将 $\Theta$ 划分为一组单纯形。在第 $j$ 个单纯形内，可以通过将 $h_{\text{DC}}$ 替换为在 $j$ 个单纯形的每个顶点 $\mathbf{v}$ 上满足 $h_j(\mathbf{v}) = h_{\text{DC}}(\mathbf{v})$ 的超平面函数 $h_j$ 来确定 $g_{\text{DC}}(\theta) - h_{\text{DC}}(\theta)$ 的下界。在这个划分内，$g_{\text{DC}}(\theta) - h_{\text{DC}}(\theta)$ 的下界由 $l_j := \min_\theta g_{\text{DC}}(\theta) - h_j(\theta)$ 界定，这可以作为凸优化问题的解来计算。任何 $l_j$ 被发现过高的划分将被丢弃。一旦每个划分的界 $l_j$ 被计算出来，选择具有最低 $l_j$ 值的划分进一步细分，并为每个新的结果划分重新计算额外的下界。这个过程持续进行，直到达到足够严格的下界（更详细的过程见 [44]）。

这种方法允许我们通过用 B&B 过程的下界替换 $\widehat{h}^{-,\gamma}(\widehat{g}^{-,\gamma})$ 和 $\widehat{h}^{+,\gamma}(\widehat{g}^{+,\gamma})$ 来保守地近似形如式6.1和6.4的 $\widehat{\text{MR}}(f)$ 的界。虽然它总是产生有效的界，但当 $\Theta$ 的维度较高时，该过程可能收敛缓慢，给出高度保守的结果。然而，对于一些特殊情况的模型类，即使高维的 DC 问题也会大大简化。我们在下一节讨论这些情况。

## 7. 线性模型、加性模型和再生核希尔伯特空间中回归模型的 MR 和 MCR

对于线性或加性模型，我们的 MR 和 MCR 方法可以做出许多简化。为简化 MR 的解释，我们在下面展示了线性模型的总体水平 MR 可以用模型的系数表示（第7.1节）。为简化计算，我们展示了计算线性模型的经验 MR 的成本仅随 $n$ 线性增长（第7.1节），尽管经验 MR 定义中的项数随 $n$ 二次增长（见式3.3和3.6）。

从 MR 出发，我们展示了如何为线性模型类（第7.2节）、正则化线性模型（第7.3节）和再生核希尔伯特空间中的回归模型（RKHS，第7.4节）计算经验 MCR。为此，我们基于第6节的方法，给出了在 $\mathcal{F}$ 上最小化 $\widehat{e}_{\text{switch}}(f)$ 和 $\widehat{e}_{\text{orig}}(f)$ 的任意组合的方法。即使相关的目标函数是非凸的，我们也可以为这些模型类可处理地获得全局最小值。我们还讨论了在使用这些模型类时确定任何观测损失的上界 $B_{\text{ind}}$ 的过程（见假设1）。

在本节中，我们假设 $\mathcal{X} \subset \mathbb{R}^p$ 其中 $p \in \mathbb{Z}^+$，$\mathcal{Y} \subset \mathbb{R}^1$，且 $L$ 是平方误差损失函数 $L(f, (y, x_1, x_2)) = (y - f(x_1, x_2))^2$。与第6节一样，我们还假设 $0 < \min_{f \in \mathcal{F}} \widehat{e}_{\text{orig}}(f)$，以确保经验 MR 是有限的。

### 7.1 解释和计算线性或加性模型的 MR

我们首先考虑用平方误差损失评估的线性模型的 MR。对于这种设置，我们可以展示 MR 的可解释定义，以及 $\widehat{e}_{\text{switch}}(f)$ 的计算高效公式。

**命题15（解释 MR 和计算线性模型的经验 MR）** 对于任何预测模型 $f$，令 $e_{\text{orig}}(f)$、$e_{\text{switch}}(f)$、$\widehat{e}_{\text{orig}}(f)$ 和 $\widehat{e}_{\text{switch}}(f)$ 基于平方误差损失 $L(f, (y, x_1, x_2)) := (y - f(x_1, x_2))^2$ 定义，其中 $y \in \mathbb{R}$，$x_1 \in \mathbb{R}^{p_1}$，$x_2 \in \mathbb{R}^{p_2}$，$p_1$ 和 $p_2$ 为正整数。令 $\beta = (\beta_1, \beta_2)$ 且 $f_\beta$ 满足 $\beta_1 \in \mathbb{R}^{p_1}$，$\beta_2 \in \mathbb{R}^{p_2}$，且 $f_\beta(\mathbf{x}) = \mathbf{x}'\beta = \mathbf{x}_1'\beta_1 + \mathbf{x}_2'\beta_2$。则

$$
\text{MR}(f_\beta) = 1 + \frac{2}{e_{\text{orig}}(f_\beta)} \left\{ \text{Cov}(Y, X_1)\beta_1 - \beta_2' \text{Cov}(X_2, X_1)\beta_1 \right\}, \qquad (7.1)
$$

且对于有限样本，

$$
\widehat{e}_{\text{switch}}(f_\beta) = \frac{1}{n} \left( \mathbf{y}'\mathbf{y} - 2 \begin{bmatrix} \mathbf{X}_1' \mathbf{W} \mathbf{y} \\ \mathbf{X}_2' \mathbf{y} \end{bmatrix}' \beta + \beta' \begin{bmatrix} \mathbf{X}_1' \mathbf{X}_1 & \mathbf{X}_1' \mathbf{W} \mathbf{X}_2 \\ \mathbf{X}_2' \mathbf{W} \mathbf{X}_1 & \mathbf{X}_2' \mathbf{X}_2 \end{bmatrix} \beta \right) \qquad (7.2),
$$

其中 $\mathbf{W} := \frac{1}{n-1}(\mathbf{1}_n \mathbf{1}_n' - \mathbf{I}_n)$，$\mathbf{1}_n$ 是 $n$ 长全1向量，$\mathbf{I}_n$ 是 $n \times n$ 单位矩阵。

式7.1表明，线性模型的模型依赖度可以用总体协方差、模型系数和模型精度来解释。Gregorutti et al. [43] 在更强的假设下（$f_\beta$ 等于 $Y$ 的条件期望函数即 $f_\beta(\mathbf{x}) = \mathbb{E}(Y \mid X = \mathbf{x})$，且协变量 $X_1$ 和 $X_2$ 已居中）展示了式7.1的等价形式。

式7.2表明，尽管 $\widehat{e}_{\text{switch}}$ 定义中的项数随 $n$ 二次增长（见式3.3），但对线性模型 $f_\beta$ 的 $\widehat{e}_{\text{switch}}(f_\beta)$ 的计算复杂度仅随 $n$ 线性增长。具体地，式7.2中的项 $\mathbf{X}_1' \mathbf{W} \mathbf{y}$ 和 $\mathbf{X}_1' \mathbf{W} \mathbf{X}_2$ 可以分别计算为 $\frac{1}{n-1}\{(\mathbf{X}_1' \mathbf{1}_n)(\mathbf{1}_n' \mathbf{y}) - (\mathbf{X}_1' \mathbf{y})\}$ 和 $\frac{1}{n-1}\{(\mathbf{X}_1' \mathbf{1}_n)(\mathbf{1}_n' \mathbf{X}_2) - (\mathbf{X}_1' \mathbf{X}_2)\}$，其中每个括号内的项的计算复杂度随 $n$ 线性增长。

与 Gregorutti et al. [43] 一样，命题15的两个结果都可以直接推广到形式为 $f_{g_1,g_2}(X_1, X_2) := g_1(X_1) + g_2(X_2)$ 的加性模型，因为排列 $X_1$ 等价于排列 $g_1(X_1)$。

### 7.2 计算线性模型的经验 MCR

基于上一节的计算结果，我们现在考虑形式为 $\mathcal{F}_{\text{lm}} := \{ f_\beta : f_\beta(\mathbf{x}) = \mathbf{x}'\beta, \beta \in \mathbb{R}^p \}$ 的线性模型类的经验 MCR 计算。为了实现第6.1和6.2节的计算过程，我们必须能够最小化 $\widehat{e}_{\text{orig}}(f_\beta)$ 和 $\widehat{e}_{\text{switch}}(f_\beta)$ 的任意线性组合。幸运的是，对于线性模型，这个最小化简化为一个二次规划，如下面的备注所示。

**备注16（线性模型类经验 MCR 的可处理性）** 对于任何 $f_\beta \in \mathcal{F}_{\text{lm}}$ 和任何固定系数 $\xi_{\text{orig}}, \xi_{\text{switch}} \in \mathbb{R}$，线性组合

$$
\xi_{\text{orig}} \widehat{e}_{\text{orig}}(f_\beta) + \xi_{\text{switch}} \widehat{e}_{\text{switch}}(f_\beta) \qquad (7.3)
$$

在 $\beta$ 中正比于二次函数 $-2\mathbf{q}'\beta + \beta' \mathbf{Q} \beta$，其中

$$
\mathbf{Q} := \xi_{\text{orig}} \mathbf{X}' \mathbf{X} + \xi_{\text{switch}} \begin{bmatrix} \mathbf{X}_1' \mathbf{X}_1 & \mathbf{X}_1' \mathbf{W} \mathbf{X}_2 \\ \mathbf{X}_2' \mathbf{W} \mathbf{X}_1 & \mathbf{X}_2' \mathbf{X}_2 \end{bmatrix},
\quad
\mathbf{q} := \left( \xi_{\text{orig}} \mathbf{y}' \mathbf{X} + \xi_{\text{switch}} \begin{bmatrix} \mathbf{X}_1' \mathbf{W} \mathbf{y} \\ \mathbf{X}_2' \mathbf{y} \end{bmatrix}' \right)',
$$

且 $\mathbf{W} := \frac{1}{n-1}(\mathbf{1}_n \mathbf{1}_n' - \mathbf{I}_n)$。因此，最小化 $\xi_{\text{orig}} \widehat{e}_{\text{orig}}(f_\beta) + \xi_{\text{switch}} \widehat{e}_{\text{switch}}(f_\beta)$ 等价于一个无约束的（可能非凸的）二次规划。

因为第6.1和6.2节的经验 MCR 计算过程由最小化一系列形如式7.3的目标函数组成，备注16表明这个过程对于无约束线性模型类是可处理的。

### 7.3 正则化线性模型

接下来，我们继续基于第7.2节的结果，计算 $\widehat{\text{MR}}$ 对正则化线性模型的边界。我们考虑由 $\mathcal{F}_{\text{lm}}$ 的二次约束子集形成的模型类，定义为

$$
\mathcal{F}_{\text{lm,rlm}} := \{ f_\beta : f_\beta(\mathbf{x}) = \mathbf{x}'\beta, \beta \in \mathbb{R}^p, \beta' \mathbf{M}_{\text{lm}} \beta \le r_{\text{lm}} \} \qquad (7.5),
$$

其中 $\mathbf{M}_{\text{lm}}$ 和 $r_{\text{lm}}$ 是预设的。同样地，该类描述了对系数向量具有二次约束的线性模型。

#### 7.3.1 计算 MCR

如第7.2节，通过引理9和13计算 $\widehat{\text{MR}}$ 的边界要求我们能够对任意 $\xi_{\text{orig}}, \xi_{\text{switch}} \in \mathbb{R}$ 最小化 $\xi_{\text{orig}} \widehat{e}_{\text{orig}}(f_\beta) + \xi_{\text{switch}} \widehat{e}_{\text{switch}}(f_\beta)$ 对 $f_\beta \in \mathcal{F}_{\text{lm,rlm}}$。应用备注16，我们可以再次等价地在式7.4的约束下最小化 $-2\mathbf{q}'\beta + \beta' \mathbf{Q} \beta$：

$$
\text{最小化 } -2\mathbf{q}'\beta + \beta' \mathbf{Q} \beta \quad \text{约束条件 } \beta' \mathbf{M}_{\text{lm}} \beta \le r_{\text{lm}}. \qquad (7.6)
$$

产生的优化问题是一个（可能非凸的）带一个二次约束的二次规划（QP1QC）。这个问题已被充分研究，并与信赖域问题相关 [11,76,75]。因此，第6.1和6.2节中关于 MCR 的界再次对二次约束线性模型类具有计算上的可处理性。

#### 7.3.2 损失的上界

约束系数向量（$\beta' \mathbf{M}_{\text{lm}} \beta \le r_{\text{lm}}$）的一个好处是，它有助于确定损失函数 $L(f_\beta, (y, \mathbf{x})) = (y - \mathbf{x}'\beta)^2$ 的上界 $B_{\text{ind}}$，这自动满足 $\mathcal{F}_{\text{lm,rlm}}$ 中所有 $f$ 的假设1。以下引理给出了确定 $B_{\text{ind}}$ 的充分条件。

**引理17（线性模型的损失上界）** 如果 $\mathbf{M}_{\text{lm}}$ 是正定的，$\mathcal{Y}$ 有界于已知范围内，且存在已知常数 $r_X$ 使得对所有 $x \in (\mathcal{X}_1 \times \mathcal{X}_2)$ 有 $\mathbf{x}' \mathbf{M}_{\text{lm}}^{-1} \mathbf{x} \le r_X$，则假设1对模型类 $\mathcal{F}_{\text{lm,rlm}}$、平方误差损失函数和常数

$$
B_{\text{ind}} = \max \left[ \left( \min_{y \in \mathcal{Y}} (y) - \sqrt{r_X r_{\text{lm}}} \right)^2, \left( \max_{y \in \mathcal{Y}} (y) + \sqrt{r_X r_{\text{lm}}} \right)^2 \right] \qquad (7.7)
$$

成立。

在实践中，常数 $r_X$ 可以通过 $\mathbf{X}$ 和 $\mathbf{Y}$ 的经验分布来近似。引理17中限制 $\mathbf{x}' \mathbf{M}_{\text{lm}}^{-1} \mathbf{x} \le r_X$ 的动机是为 $X$ 和 $\beta$ 创建互补约束。例如，如果 $\mathbf{M}_{\text{lm}}$ 是对角的，则 $\mathbf{M}_{\text{lm}}$ 的最小元素对应 $\beta$ 受 $\beta' \mathbf{M}_{\text{lm}} \beta \le r_{\text{lm}}$ 限制最小的方向（式7.5），同时也对应 $x$ 受 $\mathbf{x}' \mathbf{M}_{\text{lm}}^{-1} \mathbf{x} \le r_X$（引理17）限制最大的方向。

### 7.4 再生核希尔伯特空间（RKHS）中的回归模型

我们现在将模型类的范围扩展到再生核希尔伯特空间中的回归模型，这些模型允许协变量的非线性和非加性特征。我们证明，如第7.3节一样，在此类中的模型上最小化 $\widehat{e}_{\text{orig}}(f)$ 和 $\widehat{e}_{\text{switch}}(f)$ 的线性组合可以表示为 QP1QC，这使我们能够实现第6.1和6.2节的二分搜索过程。

首先，我们引入描述 RKHS 中回归所需的符号。令 $\mathbf{D}$ 为 $(R \times p)$ 矩阵，表示预设的 $R$ 个参考点的字典，使得 $\mathbf{D}$ 的每一行都包含在 $\mathcal{X} = \mathbb{R}^p$ 中。令 $k$ 为预设的正定核函数，令 $\mu$ 为预设的 $\mathbb{E}Y$ 的估计。令 $\mathbf{K}_D$ 为 $R \times R$ 矩阵，其中 $\mathbf{K}_D[i,j] = k(\mathbf{D}[i,\cdot], \mathbf{D}[j,\cdot])$。我们考虑以下形式的预测模型，其中到每个参考点的距离被用作回归特征：

$$
\mathcal{F}_{D,r_k} = \left\{ f_\alpha : f_\alpha(\mathbf{x}) = \mu + \sum_{i=1}^R k(\mathbf{x}, \mathbf{D}[i,\cdot]) \alpha[i], \|f_\alpha\|_k \le r_k, \alpha \in \mathbb{R}^R \right\}. \qquad (7.8)
$$

上面，范数 $\|f_\alpha\|_k$ 定义为

$$
\|f_\alpha\|_k := \sqrt{\sum_{i=1}^R \sum_{j=1}^R \alpha[i] \alpha[j] k(\mathbf{D}[i,\cdot], \mathbf{D}[j,\cdot])} = \sqrt{\alpha' \mathbf{K}_D \alpha}. \qquad (7.9)
$$

在接下来的两个小节中，我们展示经验 MCR 的界再次可以为此类可处理地计算，并且此类中模型的损失可以有上界。

#### 7.4.1 计算 MCR

再次地，从引理9和13计算 $\widehat{\text{MR}}$ 的边界要求我们能够最小化 $\widehat{e}_{\text{orig}}(f_\alpha)$ 和 $\widehat{e}_{\text{switch}}(f_\alpha)$ 的任意线性组合。

给定大小为 $n$ 的测试观测样本 $\mathbf{Z} = \begin{bmatrix} \mathbf{y} & \mathbf{X} \end{bmatrix}$，令 $\mathbf{K}_{\text{orig}}$ 为 $n \times R$ 矩阵，其中元素 $\mathbf{K}_{\text{orig}}[i,j] = k(\mathbf{X}[i,\cdot], \mathbf{D}[j,\cdot])$。令 $\mathbf{Z}_{\text{switch}} = \begin{bmatrix} \mathbf{y}_{\text{switch}} & \mathbf{X}_{\text{switch}} \end{bmatrix}$ 为 $(n(n-1)) \times (1+p)$ 矩阵，其行包含集合 $\{(\mathbf{y}[i], \mathbf{X}_1[j,\cdot], \mathbf{X}_2[i,\cdot]) : i,j \in \{1,\ldots,n\}, i \ne j\}$。最后，令 $\mathbf{K}_{\text{switch}}$ 为 $n(n-1) \times R$ 矩阵，其中 $\mathbf{K}_{\text{switch}}[i,j] = k(\mathbf{X}_{\text{switch}}[i,\cdot], \mathbf{D}[j,\cdot])$。

对于任意两个常数 $\xi_{\text{orig}}, \xi_{\text{switch}} \in \mathbb{R}$，我们可以证明在 $\mathcal{F}_{D,r_k}$ 上最小化线性组合 $\xi_{\text{orig}} \widehat{e}_{\text{orig}}(f_\alpha) + \xi_{\text{switch}} \widehat{e}_{\text{switch}}(f_\alpha)$ 等价于最小化问题

$$
\text{最小化 } \frac{\xi_{\text{orig}}}{n} \|\mathbf{y} - \mu - \mathbf{K}_{\text{orig}} \alpha\|_2^2 + \frac{\xi_{\text{switch}}}{n(n-1)} \|\mathbf{y}_{\text{switch}} - \mu - \mathbf{K}_{\text{switch}} \alpha\|_2^2 \qquad (7.10)
$$

约束条件 $\alpha' \mathbf{K}_D \alpha \le r_k$。

与问题7.5一样，问题7.8-7.9是一个 QP1QC。为展示式7.8-7.9，我们首先写 $\widehat{e}_{\text{orig}}(f_\alpha) = \frac{1}{n} \|\mathbf{y} - \mu - \mathbf{K}_{\text{orig}} \alpha\|_2^2$。遵循类似步骤，我们可以得到 $\widehat{e}_{\text{switch}}(f_\alpha) = \frac{1}{n(n-1)} \|\mathbf{y}_{\text{switch}} - \mu - \mathbf{K}_{\text{switch}} \alpha\|_2^2$。因此，对于任何两个常数 $\xi_{\text{orig}}, \xi_{\text{switch}} \in \mathbb{R}$，我们可以看到 $\xi_{\text{orig}} \widehat{e}_{\text{orig}}(f_\alpha) + \xi_{\text{switch}} \widehat{e}_{\text{switch}}(f_\alpha)$ 在 $\alpha$ 中是二次的。这意味着我们也可以为这个类可处理地计算经验 MCR 的界。

#### 7.4.2 损失的上界

使用与第7.3.2节类似的步骤，以下引理给出了在 RKHS 中回归情况下确定 $B_{\text{ind}}$ 的充分条件。

**引理18（RKHS 中回归的损失上界）** 假设 $\mathcal{Y}$ 有界于已知范围内，且存在已知常数 $r_D$ 使得对所有 $\mathbf{x} \in (\mathcal{X}_1 \times \mathcal{X}_2)$ 有 $\mathbf{v}(\mathbf{x})' \mathbf{K}_D^{-1} \mathbf{v}(\mathbf{x}) \le r_D$，其中 $\mathbf{v} : \mathbb{R}^p \to \mathbb{R}^R$ 是满足 $\mathbf{v}(\mathbf{x})[i] = k(\mathbf{x}, \mathbf{D}[i,\cdot])$ 的函数。在这些条件下，假设1对模型类 $\mathcal{F}_{D,r_k}$、平方误差损失函数和常数

$$
B_{\text{ind}} = \max \left[ \left( \min_{y \in \mathcal{Y}} (y) - (\mu + \sqrt{r_D r_k}) \right)^2, \left( \max_{y \in \mathcal{Y}} (y) + (\mu + \sqrt{r_D r_k}) \right)^2 \right] \qquad (7.11)
$$

成立。

因此，对于 RKHS 中的回归模型，我们可以使假设1对类中所有模型成立。

## 8. MR 与因果性之间的联系

我们的 MR 方法可以根本上描述为研究模型的行为在底层数据干预下如何变化。我们旨在研究这种干预对模型性能的因果效应。这个目标镜像了常规因果推断的目标：研究对变量的干预将如何改变由自然过程产生的结果。

本节进一步探索了与因果推断的这种联系。第8.1节展示了当所考虑的预测模型是自然本身的条件期望函数时，MR 简化为因果文献中通常研究的量。第8.2节提出了 MR 的替代方案，专注于可能在底层数据生成过程中发生的干预或数据扰动。

### 8.1 模型依赖度和因果效应

在本节中，我们展示总体水平模型依赖度与条件平均因果效应之间的联系。为与因果推断文献一致，我们临时将随机变量 $(Y, X_1, X_2)$ 重命名为 $(Y, T, C)$，实现值 $(y, t, c)$。这里，$T := X_1$ 表示二元处理指标，$C := X_2$ 表示一组基线协变量（"C"代表"协变量"），$Y$ 表示感兴趣的结果。在这种符号下，$e_{\text{orig}}(f)$ 表示预测函数 $f$ 的预期损失，$e_{\text{switch}}(f)$ 表示处理已被交换的观测对中的预期损失。

令 $f_0(t, c) := \mathbb{E}(Y \mid C = c, T = t)$ 为 $Y$ 的（未知）条件期望函数，我们对 $f_0$ 的函数形式不做任何限制。

令 $Y_1$ 和 $Y_0$ 分别为处理和对照下的潜在结果，使得 $Y = Y_0(1-T) + Y_1T$。个体的处理效应定义为 $Y_1 - Y_0$，平均处理效应定义为 $\mathbb{E}(Y_1 - Y_0)$。令 $\text{CATE}(c) := \mathbb{E}(Y_1 - Y_0 \mid C = c)$ 为所有 $C=c$ 患者的（未知）条件平均处理效应。因果推断方法通常假设 $(Y_1, Y_0) \perp T \mid C$（条件可忽略性）和 $0 < \mathbb{P}(T=1 \mid C=c) < 1$ 对所有 $c$ 值成立（积极性），以便 $f_0$ 和 CATE 被良好定义和可识别。

下一个命题量化了条件平均处理效应函数（CATE）与 $f_0$ 对 $X_1$ 的模型依赖度之间的关系。

**命题19（MR 的因果解释）** 对于任何预测模型 $f$，令 $e_{\text{orig}}(f)$ 和 $e_{\text{switch}}(f)$ 基于平方误差损失 $L(f, (y, t, c)) := (y - f(t, c))^2$ 定义。如果 $(Y_1, Y_0) \perp T \mid C$（条件可忽略性）且 $0 < \mathbb{P}(T=1 \mid C=c) < 1$ 对所有 $c$ 值成立（积极性），则 $\text{MR}(f_0)$ 等于

$$
1 + \frac{\text{Var}(T)}{\mathbb{E}_{T,C} \text{Var}(Y \mid T, C)} \sum_{t \in \{0,1\}} \left\{ \mathbb{E}(Y_1 - Y_0 \mid T = t)^2 + \text{Var}(\text{CATE}(C) \mid T = t) \right\}, \qquad (8.1)
$$

其中 $\text{Var}(T)$ 是处理分配的边缘方差。

我们看到上面的模型依赖度分解为几个在因果推断中各自重要的项：处理流行率（通过 $\text{Var}(T)$）；$Y$ 中不能被 $C$ 或 $T$ 解释的变异性；平均处理效应的大小（条件于 $T$）；以及条件平均处理效应在子组间的方差。例如，如果所有患者都被处理，那么在随机观测对中打乱处理对损失没有影响。在这种情况下，我们看到 $\text{Var}(T) = 0$ 且 $\text{MR}(f_0) = 1$，表示没有依赖。当 $\text{Var}(T) > 0$ 时，较高的平均处理效应幅度 $(\mathbb{E}(Y_1 - Y_0 \mid T = t)^2)$ 对应于 $f_0$ 更严重依赖 $T$ 来预测 $Y$（其他条件相同）。类似地，如果在子组间存在高度的处理效应异质性（即当 $\text{Var}(\text{CATE}(C) \mid T = t)$ 很大时），模型 $f_0$ 在预测 $Y$ 时将再次更多地使用 $T$。例如，即使平均处理效应为零，只要处理对某些子组的帮助大于其他子组，处理仍可能对预测 $Y$ 很重要。

### 8.2 条件重要性：调整 $X_1$ 和 $X_2$ 之间的依赖关系

多个模型实现低损失的一种常见情况是预测变量集 $X_1$ 和 $X_2$ 高度相关或包含冗余信息。模型可能通过依赖 $X_1$ 或依赖 $X_2$ 来良好预测，因此 MCR 将正确地识别出对 $X_1$ 的潜在依赖的广泛范围。然而，我们可能特别感兴趣的是，模型在多大程度上依赖 $X_1$ 中无法从 $X_2$ 替代获得的信息。

例如，年龄和累积财富可能相关，并且两者都可能预测未来的晋升。我们可能想知道用于预测晋升的模型在多大程度上依赖从财富测量中唯一可用的信息。

为了形式化这个概念，我们定义 $e_{\text{switch}}$ 的一个替代方案，其中以考虑 $X_1$ 和 $X_2$ 之间依赖关系的方式向 $X_1$ 添加噪声。给定一个固定预测模型 $f$，我们问：如果在具有相同 $X_2$ 值的观测之间打乱 $X_1$ 的值，模型 $f$ 的表现会如何？具体地，令 $Z^{(a)} = (Y^{(a)}, X_1^{(a)}, X_2^{(a)})$ 和 $Z^{(b)} = (Y^{(b)}, X_1^{(b)}, X_2^{(b)})$ 表示一对独立的随机向量，遵循与 $Z = (Y, X_1, X_2)$ 相同的分布，如第3节，并令

$$
e_{\text{cond}}(f) := \mathbb{E}_{X_2} \mathbb{E}_{(Y^{(b)}, X_1^{(a)}, X_2^{(b)})} \left[ L\{f, (Y^{(b)}, X_1^{(a)}, X_2^{(b)})\} \mid X_2^{(a)} = X_2^{(b)} = X_2 \right]. \qquad (8.2)
$$

用语言来说，$e_{\text{cond}}(f)$ 是给定模型 $f$ 在观测对 $(Z^{(a)}, Z^{(b)})$ 上的预期损失，其中 $X_1^{(a)}$ 和 $X_1^{(b)}$ 的值已被交换，且这些对在 $X_2$ 上匹配。这个量也可以解释为，如果以使得 $X_1$ 在给定 $X_2$ 的条件下不再提供关于 $Y$ 的信息但保持协变量 $(X_1, X_2)$ 的联合分布不变的方式向 $X_1$ 添加噪声时，$f$ 的预期损失。

然后我们将固定函数 $f$ 的条件模型依赖度，或"核心"模型依赖度（CMR）定义为

$$
\text{CMR}(f) = \frac{e_{\text{cond}}(f)}{e_{\text{orig}}(f)}. \qquad (8.3)
$$

也就是说，CMR 是当 $X_1$ 特有的信息被移除时模型性能下降的因子。如果 $X_1 \perp X_2$，那么 $X_1$ 不包含冗余信息，且 CMR 和 MR 是等价的。否则，在其他条件相同的情况下，CMR 将随着 $X_2$ 对 $X_1$ 的预测性增强而减小。类似于 MCR，我们按照与式2.2相同的方式定义条件 MCR（CMCR），但是将 MR 替换为 CMR。与 MCR 相比，CMCR 通常会产生更接近1（零依赖）的范围。

CMR 的一个优势是，它将"噪声污染"输入限制在域 $\mathcal{X}$ 内，而不是 MR 考虑的扩展域 $\mathcal{X}_1 \times \mathcal{X}_2$。这意味着 CMR 不会受到 $x_1$ 和 $x_2$ 不可能组合的影响，而 MR 可能受其影响。Hooker [38] 讨论了类似的问题，认为对预测模型在不同情况下的行为的评估应根据这些情况发生的可能性进行加权。

CMR 方法面临的一个挑战是，如式8.2中的匹配对可能很少发生，使得非参数估计 CMR 变得困难。我们接下来探讨这个估计问题。

#### 8.2.1 通过加权、匹配或插补估计 CMR

如果协变量空间是离散且低维的，基于加权或匹配的非参数方法可以有效地估计 CMR。具体地，我们可以根据协变量组合 $(X_1[i,\cdot], X_2[j,\cdot])$ 可能发生的概率对每对样本点 $i,j$ 进行加权：

$$
\widehat{e}_{\text{weight}}(f) := \frac{1}{n(n-1)} \sum_{i=1}^n \sum_{j \ne i} w(X_1[i,\cdot], X_2[j,\cdot]) \times L\{f, (\mathbf{y}[j], \mathbf{X}_1[i,\cdot], \mathbf{X}_2[j,\cdot])\},
$$

其中 $w(x_1, x_2) := \frac{\mathbb{P}(X_1 = x_1 \mid X_2 = x_2)}{\mathbb{P}(X_1 = x_1)}$ 是一个重要性权重（另见 [38]）。这里，对应不可能或不太可能的协变量组合的观测对被分别降低权重或丢弃。如果概率 $\mathbb{P}(X_1 = x_1 \mid X_2 = x_2)$ 和 $\mathbb{P}(X_1 = x_1)$ 已知，那么 $\widehat{e}_{\text{weight}}(f)$ 对 $e_{\text{cond}}(f)$ 是无偏的（见附录A.7）。

或者，如果 $X_2$ 是离散且低维的，我们可以将 $e_{\text{cond}}(f)$ 的估计限制为仅考虑 $X_2$ 恒定或"匹配"的样本观测对：

$$
\widehat{e}_{\text{match}}(f) := \frac{1}{n(n-1)} \sum_{i=1}^n \sum_{j \ne i} \frac{\mathbf{1}(\mathbf{X}_2[j,\cdot] = \mathbf{X}_2[i,\cdot])}{\mathbb{P}(X_2 = \mathbf{X}_2[i,\cdot])} \times L\{f, (\mathbf{y}[j], \mathbf{X}_1[i,\cdot], \mathbf{X}_2[j,\cdot])\}.
$$

这种方法允许在不知道条件分布 $\mathbb{P}(X_1 = x_1 \mid X_2 = x_2)$ 的情况下估计 CMR。如果逆概率权重 $\mathbb{P}(X_2 = \mathbf{X}_2[i,\cdot])^{-1}$ 已知，那么 $\widehat{e}_{\text{match}}(f)$ 对 $e_{\text{cond}}(f)$ 是无偏的（见附录A.7）。权重 $\mathbb{P}(X_2 = \mathbf{X}_2[i,\cdot])^{-1}$ 考虑到，对于任何给定值 $x_2$，$X_2$ 取该值的观测比例通常与匹配对 $(X_2^{(a)}, X_2^{(b)})$ 取该值 $x_2$ 的比例不同，因此简单地对所有匹配对求和会导致偏差。

在实践中，比例 $\mathbb{P}(X_2 = \mathbf{X}_2[i,\cdot])$ 可以近似为 $\frac{1}{n-1} \sum_{j' \ne i} \mathbf{1}(\mathbf{X}_2[i,\cdot] = \mathbf{X}_2[j',\cdot])$，并对式8.3进行细微调整以避免除以零。产生的估计类似于因果推断中常用的精确匹配过程，已知当协变量是离散且低维时效果最好，以便精确匹配常见 [97]。

然而，当协变量空间是连续或高维时，我们通常无法非参数估计 CMR。对于这种情况，我们提出在同质残差的假设下估计 CMR。具体地，我们定义 $\mu_1$ 为条件期望函数 $\mu_1(x_2) = \mathbb{E}(X_1 \mid X_2 = x_2)$，并假设随机残差 $X - \mu_1(X_2)$ 与 $X_2$ 独立。在此假设下，可以证明

$$
e_{\text{cond}}(f) = \mathbb{E}L\left[ f, \left( Y^{(b)}, \left\{ X_1^{(a)} - \mu_1(X_2^{(a)}) \right\} + \mu_1(X_2^{(b)}), X_2^{(b)} \right) \right].
$$

也就是说，$e_{\text{cond}}(f)$ 等于 $f$ 在随机观测对 $(Z^{(a)}, Z^{(b)})$ 上的预期损失，其中残差项（在花括号中）的值已被交换。由于独立性假设，不需要匹配或加权。如果 $\mu_1$ 已知，我们可以再次使用 U-统计量产生无偏估计：

$$
\widehat{e}_{\text{impute}}(f) := \frac{1}{n(n-1)} \sum_{i=1}^n \sum_{j \ne i} L\left( f, (\mathbf{y}[j], \left\{ \mathbf{X}_1[i,\cdot] - \mu_1(\mathbf{X}_2[i,\cdot]) \right\} + \mu_1(\mathbf{X}_2[j,\cdot]), \mathbf{X}_2[j,\cdot]) \right).
$$

这个估计器聚合了我们样本中的所有对，交换每对中的残差项（花括号内）的值。在实践中，当 $\mu_1$ 未知时，可以通过回归或相关机器学习技术来估计 $\mu_1$，然后代入上述方程。通过这种方式，$X - \mu_1(X_2) \perp X_2$ 的假设允许我们估计 CMR，而无需显式建模 $X_1$ 和 $X_2$ 的联合分布。

在现有文献中，Strobl et al. [95] 介绍了一种类似的估计条件变量重要性的过程。然而，与 Strobl et al. 的形式比较因作者没有定义特定的目标估计量而变得复杂，且他们的方法局限于基于树的回归模型。其他现有的条件重要性方法包括重新定义 $X_1$ 和 $X_2$ 以在计算类似于 MR 的重要性度量之前诱导近似独立的方法。这可以通过减少使用的协变量总数来实现，从而减少任何一个变量可以被其他变量预测的程度（如 Gregorutti et al. [43]）。或者，$X_2$ 中能预测 $X_1$ 的变量可以直接重组到 $X_1$ 中（如 Tolo¸si and Lengauer [99]；另见 Meinshausen and B\"uhlmann [65] 中 Kirk, Lewin and Stumpf 的讨论）。

总之，CMR 让我们看到模型在多大程度上依赖 $X_1$ 中唯一可用的信息。虽然 CMR 比 MR 更难估计，但当 $X_2$ 是离散的或可以应用同质残差假设时，存在几种可处理的方法。也可以考虑通过仅以 $X_2$ 的子集为条件来扩展 CMR。例如，我们可以考虑仅以被认为对 $X_1$ 有因果效应的 $X_2$ 元素为条件，更改式8.2中的外部期望。为简单起见，本文专注于估计 MR 的基本情形。类似的结果可能也可以推广到 CMR。

## 9. 模拟

在本节中，我们首先提供一个玩具示例来展示 MR、MCR 和 AR 的概念。然后我们展示一个蒙特卡洛模拟，研究 MCR 的自助法置信区间的有效性。

### 9.1 模拟数据的说明性玩具示例

为说明 MR、MCR 和 AR 的概念（见第3.2节），我们考虑一个玩具示例，其中 $X = (X_1, X_2) \in \mathbb{R}^2$，且 $Y \in \{-1, 1\}$ 是一个二元组标签。我们在本节的主要目标是为这三种重要性度量之间的差异建立直观理解，因此我们仅在一个样本中展示它们。我们关注经验版本的重要性度量（$\widehat{\text{MR}}$、$\widehat{\text{MCR}}^-$ 和 $\widehat{\text{MCR}}^+$），并将它们与 AR 进行比较，AR 通常被解释为样本内度量 [7]，或作为估计变量排名的替代重要性度量的中间步骤 [39,71]。

我们模拟 $X \mid Y = -1$ 来自独立的二元正态分布，均值 $\mathbb{E}(X_1 \mid Y=-1) = \mathbb{E}(X_2 \mid Y=-1) = 0$，方差 $\text{Var}(X_1 \mid Y=-1) = \text{Var}(X_2 \mid Y=-1) = 1/9$。我们模拟 $X \mid Y = 1$ 来自相同的二元正态分布，然后加上随机向量 $(C_1, C_2) := (\cos(U), \sin(U))$ 的值，其中 $U$ 是在区间 $[-\pi, \pi]$ 上均匀分布的随机变量。因此，$(C_1, C_2)$ 在单位圆上均匀分布。

给定预测模型 $f : \mathcal{X} \to \mathbb{R}$，我们使用 $f(X_1, X_2)$ 的符号作为对 $Y$ 的预测。对于损失函数，我们使用铰链损失 $L(f, (y, x_1, x_2)) = (1 - yf(x_1, x_2))_+$，其中 $(a)_+ = a$ 如果 $a \ge 0$，否则 $(a)_+ = 0$。铰链损失函数通常用作零一损失 $L(f, (y, x_1, x_2)) = \mathbf{1}[y \ne \text{sign}\{f(x_1, x_2)\}]$ 的凸近似。

我们从上述数据生成过程中模拟两个大小为300的样本，一个用于训练，一个用于测试。然后，对于用于预测 $Y$ 的模型类，我们考虑三次多项式分类器的集合

$$
\mathcal{F}_{d3} = \left\{ f_\theta : \begin{aligned} f_\theta(x_1, x_2) &= \theta[1] + \theta[2]x_1 + \theta[3]x_2 + \theta[4]x_1^2 + \theta[5]x_2^2 + \theta[6]x_1 x_2 \\ &\quad + \theta[7]x_1^3 + \theta[8]x_2^3 + \theta[9]x_1^2 x_2 + \theta[10]x_1 x_2^2; \|\theta[-1]\|_2^2 \le r_{d3} \right\},
$$

其中 $\theta[-1]$ 表示除 $\theta[1]$ 外的所有 $\theta$ 元素，并将 $r_{d3}$ 设置为最小化训练数据中10折交叉验证损失的值。令 $\mathcal{A}_{d3}$ 为在（凸）可行域 $\{f_\theta : \|\theta[-1]\|_2^2 \le r_{d3}\}$ 上最小化铰链损失的算法。我们将 $\mathcal{A}_{d3}$ 应用于训练数据以确定参考模型 $f_{\text{ref}}$。同样使用训练数据，我们将 $\epsilon$ 设置为 $\mathcal{A}_{d3}$ 的交叉验证损失的0.10倍，使得 $\mathcal{R}(\epsilon, f_{\text{ref}}, \mathcal{F}_{d3})$ 包含 $\mathcal{F}_{d3}$ 中损失不超过 $f_{\text{ref}}$ 的损失约10%的所有模型（见式4.1）。然后我们使用测试观测计算经验 AR、MR 和 MCR。

我们首先考虑 $\mathcal{A}_{d3}$ 对 $X_1$ 的 AR。计算 AR 要求我们拟合两个独立的模型：首先使用所有变量在训练数据上拟合模型，然后仅使用 $X_2$ 再次拟合。在这种情况下，第一个模型等价于 $f_{\text{ref}}$。我们将第二个模型记为 $\widehat{f}_2$。为计算 AR，我们在测试观测中评估 $f_{\text{ref}}$ 和 $\widehat{f}_2$。我们在图6-A中说明了这个 AR 计算，用黑色虚线和蓝色虚线分别标记 $f_{\text{ref}}$ 和 $\widehat{f}_2$ 的分类边界，用标记点标记测试观测（"x"表示 $Y=1$，"o"表示 $Y=-1$）。比较这两个模型相关的损失给出了 AR 的一种形式——算法 $\mathcal{A}_{d3}$ 对 $X_1$ 的必要性估计。或者，为估计 $X_1$ 的充分性，我们可以将参考模型 $f_{\text{ref}}$ 与仅使用 $X_1$ 重新训练算法 $\mathcal{A}_{d3}$ 得到的模型进行比较。我们将这第三个模型记为 $\widehat{f}_1$，并在图6-A中用蓝色实线标记其分类边界。

图6-A中的每个分类器也可以对其对 $X_1$ 的依赖度进行评估，如图6-C所示。这里，我们在 $\widehat{\text{MR}}$ 的计算中使用 $\widehat{e}_{\text{divide}}$（见式3.5）。不出意料，不使用 $X_1$ 拟合的分类器（蓝色虚线）的模型依赖度为 $\widehat{\text{MR}}(\widehat{f}_2) = 1$。参考模型 $f_{\text{ref}}$（黑色虚线）的模型依赖度为 $\widehat{\text{MR}}(f_{\text{ref}}) = 3.47$。每个 $\widehat{\text{MR}}$ 值的解释限于单个模型。也就是说，$\widehat{\text{MR}}$ 比较单个模型在不同数据分布下的行为，而不是 AR 方法比较不同模型在来自单个联合分布的边缘分布上的行为。

我们在图6-B中说明了 MCR。与 AR 相比，MCR 始终只是表现良好的预测模型的函数。这里，我们考虑经验 $\epsilon$-罗生门集 $\widehat{\mathcal{R}}(\epsilon, f_{\text{ref}}, \mathcal{F}_{d3})$，即 $\mathcal{F}_{d3}$ 中测试损失不超过 $f_{\text{ref}}$ 的测试损失加上 $\epsilon$ 的模型子集。我们用灰色实线显示了包含在 $\widehat{\mathcal{R}}(\epsilon, f_{\text{ref}}, \mathcal{F}_{d3})$ 中的15个表现良好模型的分类边界。我们还展示了 $\widehat{\mathcal{R}}(\epsilon, f_{\text{ref}}, \mathcal{F}_{d3})$ 中两个分别在 $\widehat{\mathcal{R}}(\epsilon, f_{\text{ref}}, \mathcal{F}_{d3})$ 中近似最大化和最小化对 $X_1$ 的经验依赖的模型。我们将这些模型记为 $\widehat{f}^{+,\epsilon}$ 和 $\widehat{f}^{-,\epsilon}$，并用绿色实线和绿色虚线分别标记。对于图6-B中显示的每个模型，我们还在图6-C中标记了其模型依赖度。然后，从图6-C中我们可以看到，$\widehat{\mathcal{R}}(\epsilon, f_{\text{ref}}, \mathcal{F}_{d3})$ 中每个模型的 $\widehat{\text{MR}}$ 都包含在 $\widehat{\text{MR}}(\widehat{f}^{-,\epsilon})$ 和 $\widehat{\text{MR}}(\widehat{f}^{+,\epsilon})$ 之间，直到小的近似误差。

总之，与 AR 不同，MCR 只是良好拟合数据的模型的函数。

**图6：多项式分类器的 AR、MCR 和 MR 示例**——面板（A）和（B）显示来自模拟数据集的相同300个样本，"x"表示 $Y=1$，"o"表示 $Y=-1$ 的分类。在面板（A）中，对于 AR，我们展示了通过丢弃一个协变量形成的单特征模型。因为这些模型只接受单个输入，我们将它们的分类边界表示为直线。在面板（B）中，对于 MCR，我们展示了几个具有低样本内损失的（双特征）模型的分类边界。在这些模型中，对 $X_1$ 依赖最小的模型用绿色虚线椭圆显示，对 $X_1$ 依赖最大的模型用绿色实线椭圆显示。面板（C）显示了面板（A）和（B）中每个模型对 $X_1$ 的经验模型依赖度。我们在面板（C）中看到，正如预期，没有表现良好的模型（经验地）比 $\widehat{f}^{+,\epsilon}$ 更依赖 $X_1$，或（经验地）比 $\widehat{f}^{-,\epsilon}$ 更少依赖 $X_1$。也就是说，没有表现良好的模型具有大于 $\widehat{\text{MCR}}^+(\epsilon)$ 或小于 $\widehat{\text{MCR}}^-(\epsilon)$ 的经验 MR 值。

### 9.2 自助法置信区间的模拟

在本节中，我们研究 MCR 在模型类错误设定下的表现。我们的目标是估计条件期望函数 $f_0(\mathbf{x}) = \mathbb{E}(Y \mid X = \mathbf{x})$ 在多大程度上依赖协变量的子集。给定参考模型 $f_{\text{ref}}$ 和模型类 $\mathcal{F}$，我们描述 $\text{MR}(f_0)$ 的能力取决于两个条件：

**条件20（接近正确的模型类）** 类 $\mathcal{F}$ 包含一个表现良好的模型 $\widetilde{f} \in \mathcal{R}(\epsilon, f_{\text{ref}}, \mathcal{F})$ 满足 $\text{MR}(\widetilde{f}) = \text{MR}(f_0)$（见式4.1）。

**条件21（自助法覆盖）** 经验 MCR 的自助法置信区间对总体水平 MCR 给出适当的覆盖。

条件20确保区间 $[\text{MCR}^-(\epsilon), \text{MCR}^+(\epsilon)]$ 包含 $\text{MR}(f_0)$，条件21确保这个区间可以在有限样本中估计。条件20也可以解释为 $\text{MR}(f_c)$ 的模型依赖值被类 $\mathcal{F}$ "良好支持"，即使 $\mathcal{F}$ 不包含 $f_0$。我们的主要目标是评估从 MCR 推导的置信区间是否可以对 $\text{MR}(f_0)$ 给出适当的覆盖，这取决于两个条件。作为次要目标，我们也希望能够单独评估条件20和21。

验证上述条件要求我们能够计算总体水平 MCR。为此，我们从一个20000个观测的有限总体中有放回地抽样，在该总体中 MCR 也可以直接计算。为推导基于 MCR 的置信区间，我们将每个模拟样本 $\mathbf{Z}_s$ 划分为训练子集和分析子集。我们使用训练子集拟合参考模型 $f_{\text{ref},s}$，这是总体水平 MCR 定义所必需的。我们通过从分析子集中抽取500个自助法样本计算自助法置信区间，并在每个自助法样本中通过在 $\widehat{\mathcal{R}}(\epsilon, f_{\text{ref},s}, \mathcal{F})$ 上优化来计算 $\widehat{\text{MCR}}^-(\epsilon)$ 和 $\widehat{\text{MCR}}^+(\epsilon)$。然后我们取 $\widehat{\text{MCR}}^-(\epsilon)$ 值的2.5%百分位数和 $\widehat{\text{MCR}}^+(\epsilon)$ 值的97.5%百分位数，分别作为我们置信区间的下限和上限。我们对 $X_1$ 和 $X_2$ 都重复这个过程。

我们根据一个具有不断增加的非线性程度的模型生成数据。对于 $\gamma \in \{0, 0.1, 0.2, 0.3, 0.4, 0.5\}$，我们将连续结果模拟为 $Y = f_0(X) + E$，其中 $f_0$ 是函数 $f_0(\mathbf{x}) = \sum_{j=1}^p j \mathbf{x}[j] - \gamma \mathbf{x}[j]^2$；协变量维度 $p$ 等于2，$X_1$ 和 $X_2$ 定义为 $X$ 的第一和第二个元素；协变量 $X$ 从多元正态分布中抽取，其中 $\mathbb{E}(X_1) = \mathbb{E}(X_2) = 0$，$\text{Var}(X_1) = \text{Var}(X_2) = 1$，$\text{Cov}(X_1, X_2) = 1/4$；$E$ 是正态分布的噪声变量，均值为零，方差 $\sigma_E^2 := \text{Var}(f_0(X))$。我们考虑样本量 $n = 400$ 和 $800$，其中分别有 $n_{\text{tr}} = 200$ 或 $300$ 个观测分配给训练子集。

为实现我们的方法，我们使用模型类 $\mathcal{F}_{\text{lm}} = \{f_\beta : f_\beta(\mathbf{x}) = \beta[1] + \sum_{j=1}^2 \mathbf{x}[j]\beta[j+1], \beta \in \mathbb{R}^3\}$。我们将性能阈值 $\epsilon$ 设置为 $0.1 \times \sigma_E^2$。我们称这种使用 $\mathcal{F}_{\text{lm}}$ 的 MCR 实现为"MCR-Linear"。

作为比较方法，我们考虑一个更简单的自助法方法，称为"Standard-Linear"。这里，我们从模拟数据 $\mathbf{Z}_s$ 中抽取500个自助法样本。在每个标记为 $b$ 的自助法样本中，我们预留 $n_{\text{tr}}$ 个训练点来训练模型 $f_b \in \mathcal{F}_{\text{lm}}$，并从剩余数据点计算 $\widehat{\text{MR}}(f_b)$。然后，我们通过取 $b=1,\ldots,500$ 的 $\widehat{\text{MR}}(f_b)$ 的2.5%和97.5%百分位数，创建 $\text{MR}(f_0)$ 的95%自助法百分位数置信区间。

#### 9.2.1 结果

总的来说，我们发现 MCR 相对于标准的自助法方法，为 $f_0$ 对 $X_1$ 和 $X_2$ 的依赖度提供了更稳健和保守的区间。我们还发现，较高的样本量通常加剧由于错误设定导致的覆盖误差，因为方法对偏差结果变得更加确定。

MCR-Linear 在适度错误设定水平下（$\gamma = 0.3$）给出了适当的覆盖，而 Standard-Linear 在此处开始失效（图7）。对于更大的错误设定水平（$\gamma \ge 0.4$），MCR-Linear 和 Standard-Linear 都未能给出适当的覆盖。

**图7：MR 覆盖**——y 轴显示 $f_0$ 对 $X_1$（左列）或 $X_2$（右列）的依赖度的覆盖概率，其中 $X_2$ 被模拟为比 $X_1$ 更有影响力。x 轴显示不断增加的错误设定水平（$\gamma$）。所有方法旨在为每个场景达到至少95%的覆盖（虚线水平线）。

MCR 增强的稳健性是以更宽的置信区间为代价的。MCR-Linear 的区间通常比 Standard-Linear 的区间大约宽2-4倍。这部分是由于 MCR 的置信区间旨在覆盖值 $[\text{MCR}^-(\epsilon), \text{MCR}^+(\epsilon)]$ 的范围（使用 $f_{\text{ref},s}$ 定义），而不是覆盖一个单一点。

在分别研究条件20和21时，我们发现 MCR-Linear 的覆盖误差主要归因于条件20的违反。条件21对所有研究的场景都保守地成立——在每个场景内，至少95.9%的自助法置信区间包含了总体水平 MCR。

这些模拟结果凸显了 MCR 的一个既是优点也是弱点的方面：MCR 是通用的。MCR 不假设错误设定可能发生的特定方式，并且比做出正确假设的敏感性分析能力弱。尽管如此，MCR 仍然增加了稳健性。对于足够强的信号，仍可能返回有信息量的区间。在下面的应用数据分析中，我们看到情况确实如此。

## 10. 数据分析：刑事累犯预测模型对种族和性别的依赖度

有证据表明，在刑事司法系统中，法官和检察官之间存在偏见 [92,10,72]。为应对这种偏见，训练用于预测累犯的机器学习模型越来越多地被用于告知法官关于审前释放、量刑和假释的决定 [67,73]。理想情况下，预测模型可以避免人类偏见，并为法官提供经验验证的工具。但预测模型也可能反映生成其训练数据的社会的偏见，并大规模地延续同样的偏见。就累犯而言，如果不同人群之间的逮捕率不能代表潜在的犯罪率 [5,81,104]，那么偏差可能产生于（1）结果变量——未来犯罪，通过逮捕或定罪不完美地测量，和（2）协变量——包括被告记录中的前科次数 [17,55]。此外，当预测模型的行为和机制是不透明的黑箱时，模型可以逃避审查，并且无法为被评为"高风险"的个人提供追索或解释。

我们在这里关注透明性问题，这在最近关于商业累犯预测工具 COMPAS 的辩论中扮演了重要角色 [55,17]。虽然 COMPAS 已知不显式依赖种族，但存在担忧它可能通过代理变量——与种族统计相关的变量——隐式地依赖种族（见第11节中的进一步讨论）。

我们的目标是在某些假设下（如下定义），确定 COMPAS 在多大程度上以隐式或显式方式依赖不同的协变量子集。我们分析来自佛罗里达州 Broward County 的被告的公开数据集，其中 COMPAS 评分已被记录 [55]。在这个数据集中，由于其他类别稀疏，我们只包括被测量为非裔美国人或白人的被告（共3,373人）。感兴趣的结果（$Y$）是 COMPAS 暴力累犯评分。在可用的协变量中，我们考虑三个我们称为"可接受"的变量：个人的年龄、前科次数以及当前指控是否为重罪的指标。我们还考虑两个我们称为"不可接受"的变量：个人的种族和性别。我们的"可接受"和"不可接受"标签并非旨在精确法律意义上——事实上，这些类型标签之间的界限并不总是清晰的（见第10.2节）。我们计算每个变量组的经验 MCR 和 AR，以及 MCR 的自助法置信区间（见第9.2节）。

为计算经验 MCR 和 AR，我们考虑一个灵活的 RKHS 线性模型类来预测 COMPAS 评分（下面更详细地描述）。给定这个类，MCR 范围（见式2.2）捕获了该类别中任何模型对每个协变量子集可能依赖的最高和最低程度。我们假设我们的类包含至少一个模型，该模型对"不可接受变量"的依赖程度与 COMPAS 对"不可接受变量"或我们样本中未测量的代理变量的依赖程度相同（类似于条件20）。我们对"可接受变量"做出相同的假设。这些假设可以解释为 COMPAS 的依赖值被我们选择的模型类相对"良好支持"，并允许我们确定 COMPAS 的 MR 值的边界。我们还考虑了更传统但不够稳健的 AR 方法（第3.2节），即如果在 COMPAS 评分上训练的模型拟合算法移除一个变量子集，其精度会受到多大影响。

这些计算要求我们预先定义损失函数、模型类和性能阈值。我们用平方误差损失 $L(f, (y, x_1, x_2)) = \{y - f(x_1, x_2)\}^2$ 定义 MR、MCR 和 AR。我们以式7.6的形式定义模型类 $\mathcal{F}_{D,r_k}$，其中我们基于500个训练观测的子集 $S$ 确定 $D$、$\mu$、$k$ 和 $r_k$。我们将 $D$ 设置为 $S$ 的协变量矩阵；将 $\mu$ 设置为 $S$ 中 $Y$ 的均值；将 $k$ 设置为径向基函数 $k_{\sigma_s}(\mathbf{x}, \widetilde{\mathbf{x}}) = \exp\left( -\frac{\|\mathbf{x} - \widetilde{\mathbf{x}}\|^2}{2\sigma_s} \right)$，其中我们选择 $\sigma_s$ 以最小化拟合到 $S$ 的 Nadaraya-Watson 核回归的交叉验证损失 [35]；并通过 $S$ 上的交叉验证选择参数 $r_k$。我们设置 $\epsilon$ 等于 $S$ 上交叉验证损失的0.1倍。同样使用 $S$，我们训练一个参考模型 $f_{\text{ref}} \in \mathcal{F}_{D,r_k}$。使用保留的2,873个观测，我们然后估计 $\text{MR}(f_{\text{ref}})$ 和 $\mathcal{F}_{D,r_k}$ 的 MCR。为计算 AR，我们使用 $S$ 从 $\mathcal{F}_{D,r_k}$ 训练模型，并在保留观测中评估其性能。

### 10.1 结果

我们的结果表明，种族和性别在确定 COMPAS 评分中发挥的作用介于"无"到"适度"之间，但它们的重要性低于"可接受"因素（图8）。作为比较基准，$f_{\text{ref}}$ 对"不可接受变量"的经验 MR 等于1.09，对"可接受变量"为2.78。"不可接受"和"可接受"变量的 AR 分别等于0.94和1.87，与 MR 大致一致。"不可接受变量"的 MCR 范围等于[1.00, 1.56]，表明对于 $\mathcal{F}_{D,r_k}$ 中经验损失不超过 $f_{\text{ref}}$ 的损失加上 $\epsilon$ 的任何模型，如果种族和性别被排列，模型的损失增加不超过56%。仅基于 AR 或 MR 方法无法做出这样的陈述，因为这些方法不对表现良好模型的依赖值给出上界。"不可接受变量"上 MCR 的95%自助法置信区间为[1.00, 1.73]。因此，在我们的假设下，如果 COMPAS 依赖性别、种族或其未测量的代理变量的因子大于1.73，那么像我们观察到的那样低的区间出现的概率小于0.05。

对于"可接受变量"，MCR 范围等于[1.77, 3.61]，95%自助法置信区间为[1.62, 3.96]。在我们的假设下，这意味着如果 COMPAS 依赖年龄、前科次数、重罪指标或其未测量的代理变量的因子低于1.77，那么像我们观察到的那样高的区间出现的概率小于0.05。这个结果与 Rudin et al. [83] 一致，他们发现年龄对 COMPAS 评分高度预测。

值得注意的是，上限3.61不仅在表现良好的模型中最大化了"可接受变量"上的经验 MR，而且在全局范围内跨类中的所有模型最大化（见图8和式6.5）。换句话说，不可能在 $\mathcal{F}_{D,r_k}$ 中找到在扰动数据上表现任意差但在未扰动数据上仍表现良好的模型，因此 $\widehat{e}_{\text{switch}}(f)$ 与 $\widehat{e}_{\text{orig}}(f)$ 的比率有一个有限的上界。由于 $\mathcal{F}_{D,r_k}$ 的正则化约束排除了高于3.61的 MR 值，经验 MCR 可能低估了 COMPAS 对"可接受变量"的 MR。还要注意，两个 MCR 区间都在1处左截断，因为通常足够精确地得出结论：存在一个对感兴趣变量没有依赖的表现良好模型（即 MR 等于1；见附录A.2）。

**图8：Broward County 刑事记录数据集的经验 MR 和 MCR**——对于任何预测模型 $f$，y 轴显示经验损失（$\widehat{e}_{\text{stnd}}(f)$），x 轴显示对每个协变量子集的经验依赖度（$\widehat{\text{MR}}(f)$）。零依赖（MR 等于1.0）由垂直虚线标记。对不同协变量子集的依赖度按颜色标记（"可接受"=蓝色；"不可接受"=灰色）。例如，$f_{\text{ref}}$ 的模型依赖度值由两个圆形点显示，一个针对"可接受"变量，一个针对"不可接受"变量。不同 $\epsilon$ 值的 MCR 可以表示为这个坐标空间上的边界。为此，对于每个协变量子集，我们计算保守的边界函数（以实线或"碗形"显示），保证包含类中所有模型（见第6节）。具体地，$\mathcal{F}_{D,r_k}$ 中所有模型 $f$ 都保证其经验损失（$\widehat{e}_{\text{stnd}}(f)$）和对"不可接受变量"的经验 MR 值（$\widehat{\text{MR}}(f)$）对应于灰色碗内的一个点。同样地，$\mathcal{F}_{D,r_k}$ 中所有模型都保证其经验损失和对"可接受变量"的经验 MR 值对应于蓝色碗内的一个点。显示为"x"的点表示在我们的计算过程中发现的 $\mathcal{F}_{D,r_k}$ 中的额外模型，因此显示了"碗"边界紧凑的位置。我们的计算过程（见第6节）的目标是在感兴趣的 $\epsilon$ 值附近尽可能收紧边界，如上方的水平虚线所示。这条虚线的 y 截距等于参考模型的损失加上感兴趣的 $\epsilon$ 值。$\text{MCR}^-(\epsilon)$ 和 $\text{MCR}^+(\epsilon)$ 的自助法置信区间用括号标记。

### 10.2 讨论与局限性

询问一个商业模型在调整其他协变量后是否依赖性别和种族，与称为条件统计平价（CSP）的公平性度量相关。如果决策规则在给定一组"合法"协变量 $C$ 的条件下对敏感变量独立，则满足 CSP [18,50]。粗略地说，CSP 反映了具有相似协变量 $C$ 的人群被相似对待的理念 [33]，无论敏感变量（例如种族或性别）如何。然而，如果 $C$ 中包含过多变量，该标准就变得表面化，并应注意避免包含敏感变量的代理变量。还提出了其他几种公平性度量，它们通常形成相互竞争的目标 [56,22,70,18]。在这里，如果 COMPAS 不受种族、性别或与种族或性别相关的变量的影响（以一组"合法"变量为条件），它将满足 CSP。

不幸的是，通常很难区分"合法"（或"可接受"）变量和"非法"变量。一些变量既作为风险合理预测因子的一部分，又作为种族的代理变量。由于不成比例的逮捕率，特别是轻罪和毒品相关犯罪 [104,55]，前轻罪定罪可能充当这样的代理变量 [17,55]。

我们样本中未测量的种族代理变量（定义为与种族统计相关的变量）也不是种族可以预测 COMPAS 评分的唯一原因。COMPAS 算法的其他输入可能仅在以我们归类为"可接受"的变量为条件时才与种族相关。然而，我们在第10.1节中关于种族对 COMPAS 评分的预测效用有限的结果表明，这种条件关系也是有限的。

## 11. 结论

在本文中，我们提出 MCR 作为一组变量对一个类中任何表现良好的模型的重要性的上限和下限。通过这种方式，MCR 提供了比传统单一模型重要性度量更全面和稳健的重要性度量。我们推导了 MCR 的界，这启发了我们对点估计的选择。我们还推导了排列重要性、U-统计量、条件变量重要性和条件因果效应之间的联系。我们在一个刑事累犯数据集中应用 MCR，以帮助了解商业模型 COMPAS 的特征。

有几个令人兴奋的研究领域仍然开放。一个与我们当前工作密切相关的研究方向是为其他模型类和损失函数开发精确或近似的 MCR 计算过程。我们已经证明，对于最小化经验损失是凸优化问题的模型类，MCR 可以通过一系列凸优化问题保守地计算。此外，我们已经证明计算 $\widehat{\text{MCR}}^-$ 通常不比在重新加权样本上最小化经验损失更难。MCR 的通用计算过程仍然是一个开放的研究领域。

另一个方向是考虑将 MCR 用于变量选择。如果某个变量的 MCR+ 很小，则没有表现良好的预测模型可以严重依赖该变量，表明该变量可以被消除。

我们对罗生门集的理论分析依赖于 $\mathcal{F}$ 和 $f_{\text{ref}}$ 是预设的。上面，我们通过将样本分割为大小为 $n_1$ 和 $n_2$ 的子集、使用第一个子集确定 $\mathcal{F}$ 和 $f_{\text{ref}}$、并在第二个子集中估计 MCR 时以 $\mathcal{F}$ 和 $f_{\text{ref}}$ 为条件来实现这一点。因此，我们假设中的有界常数（$B_{\text{ind}}$、$B_{\text{ref}}$、$B_{\text{switch}}$ 和 $b_{\text{orig}}$）依赖于 $\mathcal{F}$，从而依赖于 $n_1$。然而，由于我们的结果是非渐近的，我们没有探讨当 $n_1$ 和 $n_2$ 以不同速率增长时罗生门集的表现。一个令人兴奋的未来扩展是研究随着 $n_1$ 增加而变化的序列 $\{\epsilon_{n_1}, f_{\text{ref},n_1}, \mathcal{F}_{n_1}\}$ 及其对应的罗生门集 $\mathcal{R}(\epsilon_{n_1}, f_{\text{ref},n_1}, \mathcal{F}_{n_1})$，因为这可能更全面地捕获分析师如何确定模型类。

虽然我们开发罗生门集的目标是研究 MR，但罗生门集也可以用于关于同类最佳模型的各种其他属性的有限样本推断（例如，第5节）。罗生门集本身的特征也可能令人感兴趣。例如，在正在进行的工作中，我们正在研究罗生门集的大小及其与模型和模型类泛化的联系 [86]。我们还在开发用于可视化罗生门集的方法 [29]。

致谢

本工作得到了美国国立卫生研究院（资助号 P01CA134294、R01GM111339、R01ES024332、R35CA197449、R01ES026217、P50MD010428、DP2MD012722、R01MD012769 和 R01ES028033）、美国环境保护局（资助号 83615601 和 83587201-0）以及美国健康效应研究所（资助号 4953-RFA14-3/16-4）的支持。

## 附录A. 杂项补充章节

以下附录中所有条目标签以字母开头（例如第A.2节），而正文中条目的引用仅包含数字（例如命题19）。

### A.1 代码

我们在第9.1节的示例和第10节的分析的 R 代码可从 https://github.com/aaronjfisher/mcr-supplement 获取。

### A.2 模型依赖度小于1

虽然违反直觉，但预测模型的预期损失有可能在 $X_1$ 中的信息被移除时下降。粗略地说，一个"病态"模型 $f_{\text{silly}}$ 可能使用 $X_1$ 中的信息来"故意"错误分类 $Y$，使得 $e_{\text{switch}}(f_{\text{silly}}) < e_{\text{orig}}(f_{\text{silly}})$ 且 $\text{MR}(f_{\text{silly}}) < 1$。如果仍有可能从 $X_2$ 中的信息充分好地预测 $Y$，模型 $f_{\text{silly}}$ 甚至可能包含在总体 $\epsilon$-罗生门集中（见第4节）。

然而，在这些情况下，通常存在另一个性能优于 $f_{\text{silly}}$ 且 MR 等于1（即对 $X_1$ 无依赖）的模型。要理解这一点，考虑 $\mathcal{F} = \{f_\theta : \theta \in \mathbb{R}^d\}$ 由参数 $\theta$ 索引的情况。令 $\theta_{\text{silly}}$ 和 $\theta^\star$ 为参数值，使得 $f_{\theta_{\text{silly}}}$ 等价于 $f_{\text{silly}}$，且 $f_{\theta^\star}$ 是同类最佳模型。如果 $f_{\theta^\star}$ 满足 $\text{MR}(f_{\theta^\star}) > 1$ 且模型依赖函数 MR 在 $\theta$ 中连续，则存在 $\theta_{\text{silly}}$ 和 $\theta^\star$ 之间的参数值 $\theta_1$ 使得 $\text{MR}(f_{\theta_1}) = 1$。此外，如果损失函数 $L$ 在 $\theta$ 中为凸，则 $e_{\text{orig}}(f_{\theta^\star}) \le e_{\text{orig}}(f_{\theta_1}) \le e_{\text{orig}}(f_{\text{silly}})$，且任何包含 $f_{\text{silly}}$ 的总体 $\epsilon$-罗生门集也将包含 $f_{\theta_1}$。

### A.3 将 $\widehat{e}_{\text{switch}}(f)$ 与所有可能的样本排列联系起来

沿用第3节的符号，令 $\{\pi_1, \ldots, \pi_{n!}\}$ 为一组 $n$ 长向量，每个向量包含集合 $\{1, \ldots, n\}$ 的一个不同排列。我们在本节展示 $\widehat{e}_{\text{switch}}(f)$ 等于

$$
\sum_{l=1}^{n!} \sum_{i=1}^n L\{f, (\mathbf{y}[i], \mathbf{X}_1[\pi_l[i],\cdot], \mathbf{X}_2[i,\cdot])\} \mathbf{1}(\pi_l[i] \ne i)
$$

与一个仅为 $n$ 的函数的比例常数的乘积。

首先，考虑求和

$$
\sum_{l=1}^{n!} \sum_{i=1}^n L\{f, (\mathbf{y}[i], \mathbf{X}_1[\pi_l[i],\cdot], \mathbf{X}_2[i,\cdot])\},
$$

它省略了式A.1中的指示函数。式A.2中的求和包含 $n(n!)$ 项，每项都是形如 $L\{f, (\mathbf{y}[i], \mathbf{X}_1[j,\cdot], \mathbf{X}_2[i,\cdot])\}$ 的二元组合，其中 $i,j \in \{1,\ldots,n\}$。这种形式只有 $n^2$ 个独特组合，每个必须在式A.2的 $n(n!)$ 项中至少出现 $(n-1)!$ 次。要理解这一点，考虑选择两个整数值 $\widetilde{i}, \widetilde{j} \in \{1,\ldots,n\}$，并枚举项 $L\{f, (\mathbf{y}[\widetilde{i}], \mathbf{X}_1[\widetilde{j},\cdot], \mathbf{X}_2[\widetilde{i},\cdot])\}$ 在式A.2的和中出现的所有次数。在排列向量 $\{\pi_1, \ldots, \pi_{n!}\}$ 中，我们知道其中 $(n-1)!$ 个将 $\widetilde{i}$ 放在第 $\widetilde{j}$ 个位置，即满足 $\pi_l[\widetilde{i}] = \widetilde{j}$。对于每个这样的排列 $\pi_l$，式A.2中对所有可能的 $i$ 值的内层求和必须包含项 $L\{f, (\mathbf{y}[\widetilde{i}], \mathbf{X}_1[\pi_l[\widetilde{i}],\cdot], \mathbf{X}_2[\widetilde{i},\cdot])\} = L\{f, (\mathbf{y}[\widetilde{i}], \mathbf{X}_1[\widetilde{j},\cdot], \mathbf{X}_2[\widetilde{i},\cdot])\}$。因此，式A.2包含项 $L\{f, (\mathbf{y}[\widetilde{i}], \mathbf{X}_1[\widetilde{j},\cdot], \mathbf{X}_2[\widetilde{i},\cdot])\}$ 的至少 $(n-1)!$ 次出现。

到目前为止，我们已经展示了每个独特组合至少出现 $(n-1)!$ 次，但同样地，每个独特组合必须恰好出现 $(n-1)!$ 次。这是因为 $n^2$ 个独特组合中的每个必须至少出现 $(n-1)!$ 次，这总计为 $n^2((n-1)!) = n(n!)$ 项。如上所述，式A.2只有 $n(n)!$ 项，因此不可能有额外的项。然后我们可以简化式A.2为

$$
\sum_{l=1}^{n!} \sum_{i=1}^n L\{f, (\mathbf{y}[i], \mathbf{X}_1[\pi_l[i],\cdot], \mathbf{X}_2[i,\cdot])\} = (n-1)! \sum_{i=1}^n \sum_{j=1}^n L\{f, (\mathbf{y}[i], \mathbf{X}_1[j,\cdot], \mathbf{X}_2[i,\cdot])\}.
$$

通过相同的逻辑，我们可以简化式A.1为

$$
\sum_{l=1}^{n!} \sum_{i=1}^n L\{f, (\mathbf{y}[i], \mathbf{X}_1[\pi_l[i],\cdot], \mathbf{X}_2[i,\cdot])\} \mathbf{1}(\pi_l[i] \ne i) = (n-1)! \sum_{i=1}^n \sum_{j \ne i} L\{f, (\mathbf{y}[i], \mathbf{X}_1[j,\cdot], \mathbf{X}_2[i,\cdot])\},
$$

且式A.3正比于 $\widehat{e}_{\text{switch}}(f)$ 向上到一个仅为 $n$ 的函数的常数。

### A.4 同类最佳预测模型的 MR 界

虽然描述单个模型不是本研究的主要焦点，但定理4的一个推论是，我们可以为（未知的）同类最佳模型 $f^\star$ 对 $X_1$ 的依赖度创建一个概率界。

**推论22（同类最佳 MR 的界）** 令 $f^\star \in \arg\min_{f \in \mathcal{F}} e_{\text{orig}}(f)$ 为达到最小可能预期损失的预测模型，并令 $f^{+,\epsilon}$ 和 $f^{-,\epsilon}$ 如定理4所定义。如果 $f^{+,\epsilon}$ 和 $f^{-,\epsilon}$ 满足假设1、2和3，则

$$
\mathbb{P}\left( \text{MR}(f^\star) \in \left[ \widehat{\text{MCR}}^-(\epsilon_{\text{best}}) - Q_{\text{best}}, \widehat{\text{MCR}}^+(\epsilon_{\text{best}}) + Q_{\text{best}} \right] \right) \ge 1 - \delta,
$$

其中 $\epsilon_{\text{best}} := 2B_{\text{ref}} \sqrt{\frac{\log(6\delta^{-1})}{2n}}$，且

$$
Q_{\text{best}} := \frac{B_{\text{switch}}}{b_{\text{orig}}} - \frac{B_{\text{switch}} - B_{\text{ind}} \sqrt{\frac{\log(12\delta^{-1})}{n}}}{b_{\text{orig}} + B_{\text{ind}} \sqrt{\frac{\log(12\delta^{-1})}{2n}}}.
$$

上述结果不要求 $f^\star$ 是唯一的。如果多个模型达到最小可能预期损失，上述边界同时适用于每个模型。在特殊情况下，当真实条件期望函数 $\mathbb{E}(Y \mid X_1, X_2)$ 等于 $f^\star$ 时，我们就有了函数 $\mathbb{E}(Y \mid X_1, X_2)$ 对 $X_1$ 的依赖度的界。这个依赖度界也可以使用命题19转化为因果陈述。

### A.5 MR 定义中的比率与差值

我们选择基于比率的模型依赖度定义 $\text{MR}(f) = \frac{e_{\text{switch}}(f)}{e_{\text{orig}}(f)}$，以便该度量可以在不同问题之间比较，无论 $Y$ 的尺度如何。然而，一些现有工作以差值定义 VI 度量 [95,26,43]，类似于

$$
\text{MR}^{\text{difference}}(f) := e_{\text{switch}}(f) - e_{\text{orig}}(f).
$$

虽然这种差值度量不太容易解释，但它具有几个计算优势。估计量 $\widehat{\text{MR}}^{\text{difference}}(f) := \widehat{e}_{\text{switch}}(f) - \widehat{e}_{\text{orig}}(f)$ 的均值、方差和渐近分布可以使用 U-统计量的结果轻松确定，而无需使用 delta 方法 [28,61]（另见 [103]）。以 $\widehat{\text{MR}}^{\text{difference}}(f)$ 形式的估计在 $\min_{f \in \mathcal{F}} e_{\text{orig}}(f)$ 较小时也会比基于比率的 MR 定义更稳定。为提高可解释性，我们也可以将 $\text{MR}^{\text{difference}}(f)$ 除以 $Y$ 的方差进行归一化，这可以在不使用模型的情况下轻松估计，如 Williamson et al. [105]。

在基于差值的 MR 定义（式A.4）下，定理4、定理6和推论22的结果将在以下 $Q_{\text{out}}$、$Q_{\text{in}}$ 和 $Q_{\text{best}}$ 的修改定义下仍然成立：

$$
\begin{aligned}
Q_{\text{out,difference}} &:= \left(1 + \frac{1}{\sqrt{2}}\right) B_{\text{ind}} \sqrt{\frac{\log(6\delta^{-1})}{n}}, \\
Q_{\text{in,difference}} &:= B_{\text{ind}} \left\{ \sqrt{\frac{\log(8\delta^{-1}\mathcal{N}(\mathcal{F}, r\sqrt{2}))}{n}} + \sqrt{\frac{\log(8\delta^{-1}\mathcal{N}(\mathcal{F}, r))}{2n}} \right\} + 2r(\sqrt{2} + 1), \\
Q_{\text{best,difference}} &:= \left(1 + \frac{1}{\sqrt{2}}\right) B_{\text{ind}} \sqrt{\frac{\log(12\delta^{-1})}{n}}.
\end{aligned}
$$

分别用 $Q_{\text{out,difference}}$、$Q_{\text{in,difference}}$、$Q_{\text{best,difference}}$、$\text{MR}^{\text{difference}}$ 和 $\widehat{\text{MR}}^{\text{difference}}$ 替换 $Q_{\text{out}}$、$Q_{\text{in}}$、$Q_{\text{best}}$、MR 和 $\widehat{\text{MR}}$ 只需对相应的证明进行微小修改（见附录B.3、B.5和B.4）。结果也将在没有假设3的情况下成立，如 $b_{\text{orig}}$ 和 $B_{\text{switch}}$ 未出现在 $Q_{\text{out,difference}}$、$Q_{\text{in,difference}}$ 或 $Q_{\text{best,difference}}$ 中所示。

我们还在附录B.5.1中证明了关于 $\widehat{\text{MR}}^{\text{difference}}$ 的一致界的定理5的类比版本。

### A.6 罗生门集与轮廓似然区间

我们在第5.1节指出，在特定条件下，命题7返回的置信区间与轮廓似然置信区间 [20] 形式相同。为完整起见，我们简要回顾这一联系。我们在这里假设模型 $f_\theta \in \mathcal{F}$ 由有限维参数向量 $\theta \in \Theta$ 索引，其中 $\theta = (\gamma, \psi)$ 包含一维感兴趣参数 $\gamma \in \mathbb{R}^1$ 和 nuisance 参数 $\psi \in \Psi$。我们还假设 $e_{\text{orig}}(f_\theta)$ 由唯一参数值 $\theta^\star = (\gamma^\star, \psi^\star) \in \Theta$ 最小化，且我们的目标是了解 $\gamma^\star$。

如果 $s_\theta := \int_{\mathcal{Z}} \exp\{-L(f_\theta, z)\} dz$ 对所有 $\theta \in \Theta$ 有限，我们可以将 $L$ 转换为似然函数 $\mathcal{L} : (\mathcal{Z} \times \Theta) \to \mathbb{R}^1$，满足 $\mathcal{L}(z; \theta) = \exp\{-L(f_\theta, z)\}/s_\theta$。作为缩写，令 $\mathcal{L}(\mathbf{Z}; \theta)$ 表示 $\prod_{i=1}^n \mathcal{L}(\mathbf{Z}[i,\cdot]; \theta)$。此外，令 $\widehat{\theta} := \arg\min_{\theta \in \Theta} \widehat{e}_{\text{orig}}(f_\theta)$ 为经验损失最小化器，因此也是 $\theta^\star$ 的最大似然估计。如果 $\mathcal{L}$ 确实是正确的似然函数，那么 $\theta^\star = (\gamma^\star, \psi^\star)$ 对应于真实参数向量。此外，如果 $\phi(f_\theta) = \phi(f_{(\gamma,\psi)}) = \gamma$ 返回感兴趣的参数元素 $(\gamma)$，则 $\phi(f_{\theta^\star}) = \gamma^\star$ 的 $(1-\delta)$-水平轮廓似然区间为

$$
\text{PLI}(\delta) := \left\{ \gamma : \log \mathcal{L}(\mathbf{Z}; \widehat{\theta}) - \log \mathcal{L}(\mathbf{Z}; \widehat{\theta}_\gamma) \le \frac{\chi_{1,1-\delta}}{2}, \text{其中 } \widehat{\theta}_\gamma = \arg\max_{\{\theta \in \Theta : \phi(f_\theta) = \gamma\}} \mathcal{L}(\mathbf{Z}; \theta) \right\}
$$

$$
= \left\{ \gamma : \exists \widehat{\theta}_\gamma \text{ 满足 } \phi(f_{\widehat{\theta}_\gamma}) = \gamma \text{ 且 } \widehat{e}_{\text{orig}}(f_{\widehat{\theta}_\gamma}) \le \widehat{e}_{\text{orig}}(f_{\widehat{\theta}}) + \frac{\chi_{1,1-\delta}}{2n} \right\}
$$

$$
= \left\{ \gamma : \exists f_{\widehat{\theta}_\gamma} \text{ 满足 } \phi(f_{\widehat{\theta}_\gamma}) = \gamma \text{ 且 } f_{\widehat{\theta}_\gamma} \in \widehat{\mathcal{R}}\left(\frac{\chi_{1,1-\delta}}{2n}, f_{\widehat{\theta}}, \mathcal{F}\right) \right\},
$$

其中 $\chi_{1,1-\delta}$ 是自由度为1的卡方分布的第 $1-\delta$ 百分位数。如果 $\text{PLI}(\alpha)$ 确实是一个连续区间，则跨式A.5中经验罗生门集中的模型 $f_\theta$ 最大化或最小化 $\phi(f_\theta)$ 产生相同的区间。

### A.7 CMR 的无偏估计

我们在第8.2节中声称，$\widehat{e}_{\text{match}}(f)$ 和 $\widehat{e}_{\text{weight}}(f)$ 对 $e_{\text{cond}}(f)$ 都是无偏的。为展示 $\widehat{e}_{\text{match}}(f)$ 是无偏的，我们首先注意到 $\widehat{e}_{\text{match}}(f)$ 中的每个求和项具有相同的期望。沿用第3节的符号，令 $Z^{(a)} = (Y^{(a)}, X_1^{(a)}, X_2^{(a)})$ 和 $Z^{(b)} = (Y^{(b)}, X_1^{(b)}, X_2^{(b)})$ 为遵循与 $Z = (Y, X_1, X_2)$ 相同分布的独立随机变量。$\widehat{e}_{\text{match}}(f)$ 的期望为

$$
\mathbb{E} \widehat{e}_{\text{match}}(f) = \mathbb{E} \left[ \frac{\mathbf{1}(X_2^{(a)} = X_2^{(b)})}{p_{x_2}(X_2^{(a)})} \times L\{f, (Y^{(b)}, X_1^{(a)}, X_2^{(b)})\} \right] = e_{\text{cond}}(f).
$$

为展示 $\widehat{e}_{\text{weight}}(f)$ 是无偏的，我们类似地注意到 $\widehat{e}_{\text{weight}}(f)$ 中的每个求和项具有相同的期望。不失一般性，我们展示离散变量 $(Y, X_1, X_2)$ 的结果。$\widehat{e}_{\text{weight}}(f)$ 的期望为

$$
\mathbb{E} \widehat{e}_{\text{weight}}(f) = e_{\text{cond}}(f).
$$

## 附录B. 统计结果的证明

我们在本节展示统计结果的证明，并在附录C中展示计算结果的证明。

### B.1 关于经验罗生门集和总体罗生门集的引理

在剩余的证明中，将总体 $\epsilon$-罗生门集的定义表达为单个损失函数的期望（而不是两个损失函数的比较）将很有用。为此，我们简单地引入"标准化"损失函数 $\widetilde{L}$，定义为

$$
\widetilde{L}(f, z) := L(f, z) - L(f_{\text{ref}}, z).
$$

回顾第2节，$L(f, z)$ 对 $z = (y, x_1, x_2)$ 表示 $L(f, (y, x_1, x_2))$。因为我们假设 $f_{\text{ref}}$ 是预设且固定的，我们在 $\widetilde{L}$ 的定义中省略了 $f_{\text{ref}}$ 的符号。现在我们可以写

$$
\mathcal{R}(\epsilon) = \{f_{\text{ref}}\} \cup \{f \in \mathcal{F} : \mathbb{E}L(f, Z) \le \mathbb{E}L(f_{\text{ref}}, Z) + \epsilon\} = \{f_{\text{ref}}\} \cup \left\{ f \in \mathcal{F} : \mathbb{E}\widetilde{L}(f, Z) \le \epsilon \right\},
$$

并且类似地 $\widehat{\mathcal{R}}(\epsilon) = \{f_{\text{ref}}\} \cup \left\{ f \in \mathcal{F} : \widehat{\mathbb{E}}\widetilde{L}(f, Z) \le \epsilon \right\}$。

使用这个定义，以下引理允许我们限制给定模型 $f_1 \in \mathcal{R}(\epsilon)$ 被排除在经验罗生门集之外的概率。

**引理23** 对于 $\epsilon \in \mathbb{R}$ 和 $\delta \in (0, 1)$，令 $\epsilon'_1 := \epsilon + 2B_{\text{ref}} \sqrt{\frac{\log(\delta^{-1})}{2n}}$，且令 $f_1 \in \mathcal{R}(\epsilon)$ 表示一个特定的（可能未知的）预测模型。如果 $f_1$ 满足假设2，则 $\mathbb{P}\{f_1 \in \widehat{\mathcal{R}}(\epsilon'_1)\} \ge 1 - \delta$。

证明：如果 $f_{\text{ref}}$ 和 $f_1$ 是同一个函数，则结果平凡成立。否则，证明遵循 Hoeffding 不等式 [42]。首先，注意如果 $f_1$ 满足假设2，则 $\widetilde{L}(f_1)$ 有界于长度为 $2B_{\text{ref}}$ 的区间内。在下面的式B.3中应用这一点，我们看到 $\mathbb{P}\{f_1 \notin \widehat{\mathcal{R}}(\epsilon'_1)\} \le \delta$。

### B.2 在界之间转换的引理

以下引理将帮助我们翻译从变量界到这些变量的差值和比率的界。我们将应用此引理将经验损失上的界转换为经验模型依赖度上的界（以比率或差值定义）。

**引理24** 令 $X, Z, \mu_X, \mu_Z, k_X, k_Z \in \mathbb{R}$ 为满足 $|Z - \mu_Z| \le k_Z$ 和 $|X - \mu_X| \le k_X$ 的常数，则

$$
|(Z - X) - (\mu_Z - \mu_X)| \le q_{\text{difference}}(k_Z, k_X),
$$

其中 $q_{\text{difference}}$ 是函数 $q_{\text{difference}}(k_Z, k_X) := k_Z + k_X$。

此外，如果存在常数 $b_{\text{orig}}$ 和 $B_{\text{switch}}$ 使得 $0 < b_{\text{orig}} \le X, \mu_X$ 且 $Z, \mu_Z \le B_{\text{switch}} < \infty$，则

$$
\left| \frac{Z}{X} - \frac{\mu_Z}{\mu_X} \right| \le q_{\text{ratio}}(k_Z, k_X),
$$

其中 $q_{\text{ratio}}$ 是函数

$$
q_{\text{ratio}}(k_Z, k_X) := \frac{B_{\text{switch}}}{b_{\text{orig}}} - \frac{B_{\text{switch}} - k_Z}{b_{\text{orig}} + k_X}.
$$

### B.3 定理4的证明

我们分4步进行证明。

**B.3.1 步骤1：** 证明 $\mathbb{P}\left[ \widehat{\text{MR}}(f^{+,\epsilon}) \le \widehat{\text{MCR}}^+(\epsilon_{\text{out}}) \right] \ge 1 - \frac{\delta}{3}$。

考虑事件 $\widehat{\text{MR}}(f^{+,\epsilon}) \le \widehat{\text{MCR}}^+(\epsilon_{\text{out}})$。如果 $f^{+,\epsilon} \in \widehat{\mathcal{R}}(\epsilon_{\text{out}})$，则式B.10总是成立，因为根据定义 $\widehat{\text{MCR}}^+(\epsilon_{\text{out}})$ 是 $\widehat{\mathcal{R}}(\epsilon_{\text{out}})$ 中模型经验模型依赖度的上界。因此 $\mathbb{P}\left[ \widehat{\text{MR}}(f^{+,\epsilon}) > \widehat{\text{MCR}}^+(\epsilon_{\text{out}}) \right] \le \mathbb{P}\left[ f^{+,\epsilon} \notin \widehat{\mathcal{R}}(\epsilon_{\text{out}}) \right] \le \frac{\delta}{3}$，来自 $\epsilon_{\text{out}}$ 的定义和引理23。

**B.3.2 步骤2：** 以 $\widehat{\text{MR}}(f^{+,\epsilon}) \le \widehat{\text{MCR}}^+(\epsilon_{\text{out}})$ 为条件，用 $\widehat{\text{MCR}}^+(\epsilon_{\text{out}})$ 加上一项误差项来上界 $\text{MR}(f^{+,\epsilon})$。

当式B.10成立时，我们有 $\text{MR}(f^{+,\epsilon}) \le \widehat{\text{MCR}}^+(\epsilon_{\text{out}}) + [\text{MR}(f^{+,\epsilon}) - \widehat{\text{MR}}(f^{+,\epsilon})]$。

**B.3.3 步骤3：** 从概率上界定步骤2中的误差项。

接下来我们证明括号中的项以高概率小于或等于 $Q_{\text{out}}$。令 $q_{\text{difference}}$ 和 $q_{\text{ratio}}$ 如式B.6和B.8定义。令 $q : \mathbb{R} \to \mathbb{R}$ 为函数 $q(k) = q_{\text{ratio}}(k, k/\sqrt{2})$。则 $Q_{\text{out}} = q\left(B_{\text{ind}} \sqrt{\frac{\log(6\delta^{-1})}{n}}\right)$。

应用此关系，我们有

$$
\mathbb{P}\left[ \text{MR}(f^{+,\epsilon}) - \widehat{\text{MR}}(f^{+,\epsilon}) > Q_{\text{out}} \right] \le \frac{2\delta}{3}.
$$

在式B.15中，回顾 $\widehat{e}_{\text{orig}}(f^{+,\epsilon})$ 和 $\widehat{e}_{\text{switch}}(f^{+,\epsilon})$ 都是 U-统计量。注意 $\mathbb{E}[\widehat{e}_{\text{switch}}(f^{+,\epsilon})] = e_{\text{switch}}(f^{+,\epsilon})$，因为 $\widehat{e}_{\text{switch}}(f^{+,\epsilon})$ 是各项的平均值，且每项的期望等于 $e_{\text{switch}}(f^{+,\epsilon})$。出于相同原因，$\mathbb{E}[\widehat{e}_{\text{orig}}(f^{+,\epsilon})] = e_{\text{orig}}(f^{+,\epsilon})$。这允许我们应用 Hoeffding [42] 的式5.7（另见 Serfling [84] 第201页定理A中的式1）来获得式B.15。

或者，如果我们改为将模型依赖度定义为 $\text{MR}^{\text{difference}}(f) = e_{\text{switch}}(f) - e_{\text{orig}}(f)$（见附录A.5），将经验模型依赖度定义为 $\widehat{\text{MR}}^{\text{difference}}(f) := \widehat{e}_{\text{switch}}(f) - \widehat{e}_{\text{orig}}(f)$，并定义 $Q_{\text{out,difference}} := \left(1 + \frac{1}{\sqrt{2}}\right) B_{\text{ind}} \sqrt{\frac{\log(6\delta^{-1})}{n}}$，则相同的证明在没有假设3的情况下成立（将 MR、$\widehat{\text{MR}}$、$Q_{\text{out}}$ 分别替换为 $\text{MR}^{\text{difference}}$、$\widehat{\text{MR}}^{\text{difference}}$、$Q_{\text{out,difference}}$）。

式B.14-B.16也适用于将 $\widehat{e}_{\text{switch}}$ 整体替换为 $\widehat{e}_{\text{divide}}$，包括在假设3中，因为相同的界可用于 $\widehat{e}_{\text{switch}}$ 和 $\widehat{e}_{\text{divide}}$（Hoeffding [42] 的式5.7；另见 Serfling [84] 第201页定理A）。

**B.3.4 步骤4：** 组合结果以证明式4.2。

最后，我们连接上述结果以证明式4.2。我们从式B.12知道式B.10以高概率成立。式B.10意味着式B.13，它将 $\text{MCR}^+(\epsilon) = \text{MR}(f^{+,\epsilon})$ 界住（到括号中的残差项）。我们还从式B.16知道，残差项以高概率小于 $Q_{\text{out}}$。综合这些，我们可以证明式4.2：

$$
\mathbb{P}\left( \text{MCR}^+(\epsilon) > \widehat{\text{MCR}}^+(\epsilon_{\text{out}}) + Q_{\text{out}} \right) \le \delta.
$$

这就完成了式4.2的证明。对于式4.3，我们可以使用相同的方法。类似于式B.12，我们有 $\mathbb{P}\left[ \widehat{\text{MR}}(f^{-,\epsilon}) < \widehat{\text{MCR}}^-(\epsilon_{\text{out}}) \right] \le \frac{\delta}{3}$。类似于式B.13，当 $\widehat{\text{MR}}(f^{-,\epsilon}) \ge \widehat{\text{MR}}(\widehat{f}^{-,\epsilon_{\text{out}}})$ 时，我们有 $\text{MR}(f^{-,\epsilon}) \ge \widehat{\text{MCR}}^-(\epsilon_{\text{out}}) - \left[ \widehat{\text{MR}}(f^{-,\epsilon}) - \text{MR}(f^{-,\epsilon}) \right]$。类似于式B.16，我们有 $\mathbb{P}\left[ \widehat{\text{MR}}(f^{-,\epsilon}) - \text{MR}(f^{-,\epsilon}) > q\left(B_{\text{ind}} \sqrt{\frac{\log(6\delta^{-1})}{n}}\right) \right] \le \frac{2\delta}{3}$。最终，类似于式B.17，我们有 $\mathbb{P}\left( \text{MCR}^-(\epsilon) < \widehat{\text{MCR}}^-(\epsilon_{\text{out}}) - Q_{\text{out}} \right) \le \delta$。

再次地，如果在式B.18和B.19中将 MR、$\widehat{\text{MR}}$、$Q_{\text{out}}$ 分别替换为 $\text{MR}^{\text{difference}}$、$\widehat{\text{MR}}^{\text{difference}}$、$Q_{\text{out,difference}}$，则相同的证明在没有假设3的情况下成立。

### B.4 推论22的证明

根据定义，$\text{MR}(f^{-,\epsilon_{\text{best}}}) \le \text{MR}(f^\star) \le \text{MR}(f^{+,\epsilon_{\text{best}}})$。应用此关系，我们有

$$
\mathbb{P}\left( \text{MR}(f^\star) \in \left[ \widehat{\text{MCR}}^-(\epsilon_{\text{best}}) - Q_{\text{best}}, \widehat{\text{MCR}}^+(\epsilon_{\text{best}}) + Q_{\text{best}} \right] \right) \ge 1 - \delta,
$$

应用定理4，其中 $Q_{\text{best}}$ 和 $\epsilon_{\text{best}}$ 等价于定理4中 $Q_{\text{out}}$ 和 $\epsilon_{\text{out}}$ 的定义，但 $\delta$ 被替换为 $\frac{\delta}{2}$。

或者，如果我们将模型依赖度定义为 $\text{MR}^{\text{difference}}(f) = e_{\text{switch}}(f) - e_{\text{orig}}(f)$，则令 $Q_{\text{best,difference}} := \left(1 + \frac{1}{\sqrt{2}}\right) B_{\text{ind}} \sqrt{\frac{\log(12\delta^{-1})}{n}}$。在这种基于差值的模型依赖度定义下，定理4在没有假设3的情况下成立（将 $Q_{\text{out}}$ 替换为 $Q_{\text{out,difference}}$，见第B.3节），因此我们可以在式B.21中应用这个修改版本的定理4。因此，推论22也在没有假设3的情况下成立（将 MR、$\widehat{\text{MR}}$ 和 $Q_{\text{best}}$ 分别替换为 $\text{MR}^{\text{difference}}$、$\widehat{\text{MR}}^{\text{difference}}$ 和 $Q_{\text{best,difference}}$）。

### B.5 定理5和6的证明

我们首先证明定理5以及相关结果，然后应用这些结果来证明定理6。

#### B.5.1 定理5的证明，以及基于覆盖数的其他估计误差限制

以下定理使用基于 $r$-边际期望覆盖的覆盖数来联合界定任何函数 $f \in \mathcal{F}$ 的经验损失。正文中的定理5直接来自下面的式B.25。

**定理25** 如果假设1、2和3对所有 $f \in \mathcal{F}$ 成立，则对任何 $r > 0$，

$$
\begin{aligned}
&\mathbb{P}_D \left( \sup_{f \in \mathcal{F}} |\widehat{e}_{\text{orig}}(f) - e_{\text{orig}}(f)| > B_{\text{ind}} \sqrt{\frac{\log(2\delta^{-1}\mathcal{N}(\mathcal{F}, r))}{2n}} + 2r \right) \le \delta, \\
&\mathbb{P}_D \left( \sup_{f \in \mathcal{F}} \left| \widehat{\mathbb{E}}\widetilde{L}(f, Z) - \mathbb{E}\widetilde{L}(f, Z) \right| > 2B_{\text{ref}} \sqrt{\frac{\log(2\delta^{-1}\mathcal{N}(\mathcal{F}, r))}{2n}} + 2r \right) \le \delta, \\
&\mathbb{P}_D \left( \sup_{f \in \mathcal{F}} |\widehat{e}_{\text{switch}}(f) - e_{\text{switch}}(f)| > B_{\text{ind}} \sqrt{\frac{\log(2\delta^{-1}\mathcal{N}(\mathcal{F}, r))}{n}} + 2r \right) \le \delta, \\
&\mathbb{P} \left( \sup_{f \in \mathcal{F}} \left| \frac{\widehat{e}_{\text{orig}}(f)}{\widehat{e}_{\text{switch}}(f)} - \frac{e_{\text{orig}}(f)}{e_{\text{switch}}(f)} \right| > Q_4 \right) \le \delta, \\
&\mathbb{P}_D \left( \sup_{f \in \mathcal{F}} |\{\widehat{e}_{\text{switch}}(f) - \widehat{e}_{\text{orig}}(f)\} - \{e_{\text{switch}}(f) - e_{\text{orig}}(f)\}| > Q_{4,\text{difference}} \right) \le \delta,
\end{aligned}
$$

其中

$$
\begin{aligned}
Q_4 &:= q_{\text{ratio}}\left( B_{\text{ind}} \sqrt{\frac{\log(4\delta^{-1}\mathcal{N}(\mathcal{F}, r\sqrt{2}))}{n}} + 2r\sqrt{2},\ B_{\text{ind}} \sqrt{\frac{\log(4\delta^{-1}\mathcal{N}(\mathcal{F}, r))}{2n}} + 2r \right), \\
Q_{4,\text{difference}} &:= q_{\text{difference}}\left( B_{\text{ind}} \sqrt{\frac{\log(4\delta^{-1}\mathcal{N}(\mathcal{F}, r\sqrt{2}))}{n}} + 2r\sqrt{2},\ B_{\text{ind}} \sqrt{\frac{\log(4\delta^{-1}\mathcal{N}(\mathcal{F}, r))}{2n}} + 2r \right),
\end{aligned}
$$

且 $q_{\text{ratio}}$ 和 $q_{\text{difference}}$ 如引理24定义。对于式B.26，结果不受移除假设3的影响。

#### B.5.2 式B.22的证明

令 $G_r$ 为 $\mathcal{F}$ 的大小为 $\mathcal{N}(\mathcal{F}, r)$ 的 $r$-边际期望覆盖。令 $D_p$ 表示总体分布，$D_s$ 表示样本分布，$D^\star$ 为 $D_p$ 和 $D_s$ 的均匀混合。由于 $G_r$ 是 $r$-边际期望覆盖，我们知道对于任何 $f \in \mathcal{F}$，我们可以找到函数 $g \in G_r$ 使得 $\mathbb{E}_{D^\star} |L(g, Z) - L(f, Z)| \le r$，且

$$
\left| \widehat{\mathbb{E}}L(f, Z) - \mathbb{E}L(f, Z) \right| \le \left| \widehat{\mathbb{E}}L(g, Z) - \mathbb{E}L(g, Z) \right| + 2r.
$$

应用上述关系，并使用 Hoeffding 不等式，我们得到 $\mathbb{P}\left( \sup_{f \in \mathcal{F}} \left| \widehat{\mathbb{E}}L(f, Z) - \mathbb{E}L(f, Z) \right| > B_{\text{ind}} \sqrt{\frac{\log(2\delta^{-1}\mathcal{N}(\mathcal{F}, r))}{2n}} + 2r \right) \le \delta$。

#### B.5.3 式B.23的证明

式B.23的证明与式B.22的证明几乎相同。只需在式B.30-B.33中将 $L$ 和 $B_{\text{ind}}$ 分别替换为 $\widetilde{L}$ 和 $(2B_{\text{ref}})$。

#### B.5.4 式B.24的证明

遵循与式B.22证明相同的步骤，但将 $e_{\text{orig}}$、$\widehat{e}_{\text{orig}}$、$D_p$、$D_s$ 和 $D^\star$ 分别替换为 $e_{\text{switch}}$、$\widehat{e}_{\text{switch}}$、$\widetilde{D}_p$、$\widetilde{D}_s$ 和 $\widetilde{D}^\star$，其中 $\widetilde{D}_p$ 是使得 $Y$ 和 $X_2$ 独立于 $X_1$ 的分布，$\widetilde{D}_s$ 是相应的经验分布。同样的界可以推导出 $\mathbb{P}\left( \sup_{f \in \mathcal{F}} |\widehat{e}_{\text{switch}}(f) - e_{\text{switch}}(f)| > B_{\text{ind}} \sqrt{\frac{\log(2\delta^{-1}\mathcal{N}(\mathcal{F}, r))}{n}} + 2r \right) \le \delta$。

#### B.5.5 式B.25的证明

应用引理24和式B.27，我们得到 $\mathbb{P}\left( \sup_{f \in \mathcal{F}} \left| \frac{\widehat{e}_{\text{orig}}(f)}{\widehat{e}_{\text{switch}}(f)} - \frac{e_{\text{orig}}(f)}{e_{\text{switch}}(f)} \right| > Q_4 \right) \le \delta$。

#### B.5.6 式B.26的证明

应用相同的步骤，使用式B.28和引理24，我们得到 $\mathbb{P}\left( \sup_{f \in \mathcal{F}} |\{\widehat{e}_{\text{switch}}(f) - \widehat{e}_{\text{orig}}(f)\} - \{e_{\text{switch}}(f) - e_{\text{orig}}(f)\}| > Q_{4,\text{difference}} \right) \le \delta$。

#### B.5.7 应用定理25来证明定理6

考虑事件 $\exists \widehat{f}^{+,\epsilon_{\text{in}}} \in \arg\max_{f \in \widehat{\mathcal{R}}(\epsilon_{\text{in}})} \widehat{\text{MR}}(f)$ 使得 $\text{MCR}^+(\epsilon) < \text{MR}(\widehat{f}^{+,\epsilon_{\text{in}}})$。我们期望式B.38不太可能，因为 $\epsilon_{\text{in}} < \epsilon$。如果式B.38不成立，则 $\text{MCR}^+(\epsilon) < \widehat{\text{MCR}}^+(\epsilon_{\text{in}}) - Q_{\text{in}}$ 唯一成立的方式是存在 $\widehat{f}^{+,\epsilon_{\text{in}}} \in \arg\max_{f \in \widehat{\mathcal{R}}(\epsilon_{\text{in}})} \widehat{\text{MR}}(f)$，其经验 MR 与其总体水平 MR 相差至少 $Q_{\text{in}}$。

为证明式B.38不太可能，我们应用定理25，得到此事件的概率 $\le \frac{\delta}{2}$。

如果式B.38不成立，则 $\text{MCR}^+(\epsilon) \ge \text{MR}(\widehat{f}^{+,\epsilon_{\text{in}}})$ 对所有 $\widehat{f}^{+,\epsilon_{\text{in}}} \in \arg\max_{f \in \widehat{\mathcal{R}}(\epsilon_{\text{in}})} \widehat{\text{MR}}(f)$，由此可推导出 $\text{MCR}^+(\epsilon) \ge \widehat{\text{MCR}}^+(\epsilon_{\text{in}}) - \sup_{f \in \mathcal{F}} |\widehat{\text{MR}}(f) - \text{MR}(f)|$。

定理25意味着式B.41中的 $\sup$ 项以至少 $1 - \frac{\delta}{2}$ 的概率小于 $Q_{\text{in}}$。现在，检验式4.6的左侧，我们看到 $\mathbb{P}\left( \text{MCR}^+(\epsilon) < \widehat{\text{MCR}}^+(\epsilon_{\text{in}}) - Q_{\text{in}} \right) \le \delta$。

这就完成了式4.6的证明。对于式4.7，我们可以使用相同的方法。

### B.6 命题7的证明和唯一同类最佳模型的推论

我们首先引入一个引理来描述总体 $\epsilon$-罗生门集中任何单个模型的性能。

**引理26** 令 $\epsilon'_1 := 2B_{\text{ref}} \sqrt{\frac{\log(\delta^{-1})}{2n}}$，并令函数 $\widehat{\phi}^-$ 和 $\widehat{\phi}^+$ 如命题7定义。给定函数 $f_1 \in \mathcal{R}(\epsilon)$，如果假设2对 $f_1$ 成立，则 $\mathbb{P}\left\{ \phi(f_1) \in \left[ \widehat{\phi}^-(\epsilon'_1), \widehat{\phi}^+(\epsilon'_1) \right] \right\} \ge 1 - \delta$。

证明：考虑事件 $\phi(f_1) \in [\widehat{\phi}^-(\epsilon'_1), \widehat{\phi}^+(\epsilon'_1)]$。如果 $f_1 \in \widehat{\mathcal{R}}(\epsilon'_1)$，则式B.46总是成立，因为区间 $[\widehat{\phi}^-(\epsilon'_1), \widehat{\phi}^+(\epsilon'_1)]$ 根据定义包含任何 $f \in \widehat{\mathcal{R}}(\epsilon'_1)$ 的 $\phi(f)$。因此 $\mathbb{P}\{ \phi(f_1) \notin [\widehat{\phi}^-(\epsilon'_1), \widehat{\phi}^+(\epsilon'_1)] \} \le \mathbb{P}\{ f_1 \notin \widehat{\mathcal{R}}(\epsilon'_1) \} \le \delta$，来自引理23。

#### B.6.1 命题7的证明

令 $f^{-,\epsilon,\phi} \in \arg\min_{f \in \mathcal{R}(\epsilon)} \phi(f)$ 和 $f^{+,\epsilon,\phi} \in \arg\max_{f \in \mathcal{R}(\epsilon)} \phi(f)$ 分别表示在 $\mathcal{R}(\epsilon)$ 中达到最低和最高 $\phi(f)$ 值的函数。应用 $f^{-,\epsilon,\phi}$ 和 $f^{+,\epsilon,\phi}$ 的定义，我们有 $\mathbb{P}\left( \{\phi(f) : f \in \mathcal{R}(\epsilon)\} \not\subset [\widehat{\phi}^-(\epsilon'), \widehat{\phi}^+(\epsilon')] \right) \le \delta$。

#### B.6.2 唯一同类最佳模型的推论

当同类最佳模型唯一时，可以通过下面的推论来描述。

**推论27** 令 $\widehat{\phi}^-(\epsilon'_0) := \min_{f \in \widehat{\mathcal{R}}(\epsilon'_1)} \phi(f)$ 和 $\widehat{\phi}^+(\epsilon'_1) := \max_{f \in \widehat{\mathcal{R}}(\epsilon'_1)} \phi(f)$，其中 $\epsilon'_0 := 2B_{\text{ref}} \sqrt{\frac{\log(\delta^{-1})}{2n}}$。令 $f^\star \in \arg\min_{f \in \mathcal{F}} e_{\text{orig}}(f)$ 为唯一达到最小可能预期损失的预测模型。如果 $f^\star$ 满足假设2，则 $\mathbb{P}\{ \phi(f^\star) \in [\widehat{\phi}^-(\epsilon'_1), \widehat{\phi}^+(\epsilon'_1)] \} \ge 1 - \delta$。

由于 $f^\star \in \mathcal{R}(0)$，推论27直接来自引理26。

### B.7 罗生门集定义中的绝对损失与相对损失

在本文中，我们主要将罗生门集定义为相对于参考模型 $f_{\text{ref}}$ 表现良好的模型。我们也可以研究罗生门集的另一种形式，将相对损失 $\widetilde{L}$ 替换为非标准化损失 $L$。这产生了罗生门集 $\mathcal{R}(\epsilon_{\text{abs}}, f_{\text{ref}}, \mathcal{F}) = \{f_{\text{ref}}\} \cup \{f \in \mathcal{F} : \mathbb{E}L(f, Z) \le \epsilon_{\text{abs}}\}$ 的新解释，即 $f_{\text{ref}}$ 和绝对损失 $L$ 不高于 $\epsilon_{\text{abs}}$（$\epsilon_{\text{abs}} > 0$）的模型子集的并集。计算经验 MCR 的过程基本上不受使用 $L$ 还是 $\widetilde{L}$ 的影响，因为从一个优化问题转换到另一个是简单的。

我们仍然需要在经验和总体罗生门集中显式包含 $f_{\text{ref}}$ 以确保它们非空。然而，在许多情况下，当解释罗生门集时，这种包含变得冗余（例如，当 $\epsilon \ge 0$ 且 $\mathbb{E}L(f_{\text{ref}}, Z) \le \epsilon_{\text{abs}}$ 时）。在将 $\widetilde{L}$ 替换为 $L$ 的情况下，我们也将假设2替换为假设1（只要这不是冗余的），并在定理4、推论22、定理6、命题7和推论27的 $\epsilon_{\text{out}}$、$\epsilon_{\text{best}}$、$\epsilon_{\text{in}}$、$\epsilon'$ 和 $\epsilon'_1$ 定义中将 $2B_{\text{ref}}$ 替换为 $B_{\text{ind}}$。

### B.8 命题15的证明

为证明式7.1，我们从 $e_{\text{orig}}(f_\beta)$ 开始。对于 $e_{\text{switch}}(f_\beta)$，我们遵循相同的步骤。由于 $(Y^{(b)}, X_1^{(b)}, X_2^{(b)})$ 和 $(Y^{(a)}, X_1^{(a)}, X_2^{(a)})$ 各自与 $(Y, X_1, X_2)$ 同分布，我们可以省略上标符号来展示式7.1。两边除以 $e_{\text{orig}}(f_\beta)$ 得到所需结果。

接下来，我们可以使用类似的方法展示式7.2。

### B.9 命题19的证明

首先我们考虑 $e_{\text{orig}}(f_0)$。在 $(Y_1, Y_0) \perp T \mid C$ 的假设下，我们有 $f_0(t, c) = \mathbb{E}(Y \mid C = c, T = t) = \mathbb{E}(Y_t \mid C = c)$。应用此，我们得到 $e_{\text{orig}}(f_0) = q \mathbb{E}_{C \mid T=0} \text{Var}(Y_0 \mid C) + p \mathbb{E}_{C \mid T=1} \text{Var}(Y_1 \mid C)$，其中 $p := \mathbb{P}(T=1)$ 且 $q := \mathbb{P}(T=0)$。

现在我们考虑 $e_{\text{switch}}(f_0)$。令 $(Y_0^{(a)}, Y_1^{(a)}, T^{(a)}, C^{(a)})$ 和 $(Y_0^{(b)}, Y_1^{(b)}, T^{(b)}, C^{(b)})$ 为一对独立的随机变量向量，各自与 $(Y_0, Y_1, T, C)$ 同分布。然后 $e_{\text{switch}}(f_0) = e_{\text{orig}}(f_0) + \text{Var}(T) \left\{ \mathbb{E}_{C \mid T=0}[ \text{CATE}(C)^2] + \mathbb{E}_{C \mid T=1}[ \text{CATE}(C)^2] \right\}$。两边除以 $e_{\text{orig}}(f_0) = \mathbb{E}_{T,C} \text{Var}(Y \mid T, C)$ 得到所需结果。

## 附录C. 计算结果的证明

本节几乎所有证明在 $\widehat{h}^{-,\gamma}$、$\widehat{h}^{+,\gamma}$、$\widehat{g}^{-,\gamma}$、$\widehat{g}^{+,\gamma}$ 和 $\widehat{\text{MR}}$ 的定义中，将 $\widehat{e}_{\text{switch}}(f)$ 替换为 $\widehat{e}_{\text{divide}}(f)$ 后保持不变。唯一的例外是附录C.3。

在下面的证明中，我们将利用以下事实：对于满足 $a \ge c$ 的常数 $a,b,c,d \in \mathbb{R}$，关系 $a + b \le c + d$ 意味着 $b \le d$。我们还将利用对于任何 $\gamma_1, \gamma_2 \in \mathbb{R}$，$\widehat{g}^{+,\gamma_1}$ 和 $\widehat{g}^{-,\gamma_1}$ 的定义意味着 $\widehat{h}^{+,\gamma_1}(\widehat{g}^{+,\gamma_1}) \le \widehat{h}^{+,\gamma_1}(\widehat{g}^{+,\gamma_2})$ 和 $\widehat{h}^{-,\gamma_1}(g^{-,\gamma_1}) \le \widehat{h}^{-,\gamma_1}(g^{-,\gamma_2})$。此外，$\widehat{h}^{+,\gamma_1}(f) = \widehat{h}^{+,\gamma_2}(f) + (\gamma_1 - \gamma_2)\widehat{e}_{\text{switch}}(f)$，且 $\widehat{h}^{-,\gamma_1}(f) = \widehat{h}^{-,\gamma_2}(f) + (\gamma_1 - \gamma_2)\widehat{e}_{\text{orig}}(f)$。

### C.1 引理9的证明（MR 的下界）

我们分2部分证明引理9。

**C.1.1 第1部分：** 证明式6.1对所有满足 $\widehat{e}_{\text{orig}}(f) \le \epsilon_{\text{abs}}$ 的 $f \in \mathcal{F}$ 成立。

如果 $\widehat{h}^{-,\gamma}(\widehat{g}^{-,\gamma}) \ge 0$，则对于任何满足 $\widehat{e}_{\text{orig}}(f) \le \epsilon_{\text{abs}}$ 的函数 $f \in \mathcal{F}$，我们知道 $\frac{1}{\epsilon_{\text{abs}}} \le \frac{1}{\widehat{e}_{\text{orig}}(f)}$ 且 $\frac{\widehat{h}^{-,\gamma}(\widehat{g}^{-,\gamma})}{\epsilon_{\text{abs}}} \le \frac{\widehat{h}^{-,\gamma}(\widehat{g}^{-,\gamma})}{\widehat{e}_{\text{orig}}(f)}$。

现在，对于任何满足 $\widehat{e}_{\text{orig}}(f) \le \epsilon_{\text{abs}}$ 的 $f \in \mathcal{F}$，$\widehat{g}^{-,\gamma}$ 的定义意味着 $\widehat{h}^{-,\gamma}(f) \ge \widehat{h}^{-,\gamma}(\widehat{g}^{-,\gamma})$，因此 $\widehat{\text{MR}}(f) \ge \frac{\widehat{h}^{-,\gamma}(\widehat{g}^{-,\gamma})}{\epsilon_{\text{abs}}} - \gamma$。

**C.1.2 第2部分：** 证明如果 $f = \widehat{g}^{-,\gamma}$ 且条件8中的至少一个不等式取等号，则式6.1取等号。

我们分别考虑条件8中的两个不等式。如果 $\widehat{h}^{-,\gamma}(\widehat{g}^{-,\gamma}) = 0$，则 $\frac{\widehat{h}^{-,\gamma}(\widehat{g}^{-,\gamma})}{\epsilon_{\text{abs}}} - \gamma = \widehat{\text{MR}}(\widehat{g}^{-,\gamma})$。或者，如果 $\widehat{e}_{\text{orig}}(\widehat{g}^{-,\gamma}) = \epsilon_{\text{abs}}$，则同样 $\frac{\widehat{h}^{-,\gamma}(\widehat{g}^{-,\gamma})}{\epsilon_{\text{abs}}} - \gamma = \widehat{\text{MR}}(\widehat{g}^{-,\gamma})$。

### C.2 引理10的证明（MR 下界二分搜索的单调性）

我们分3部分证明引理10。

**C.2.1 第1部分：** $\widehat{h}^{-,\gamma}(\widehat{g}^{-,\gamma})$ 在 $\gamma$ 中单调递增。

令 $\gamma_1, \gamma_2 \in \mathbb{R}$ 满足 $\gamma_1 < \gamma_2$。我们假设对任何 $f \in \mathcal{F}$ 有 $0 < \widehat{e}_{\text{orig}}(f)$。因此，对于任何 $f \in \mathcal{F}$，$\widehat{h}^{-,\gamma_1}(f) < \widehat{h}^{-,\gamma_2}(f)$。应用此，我们有 $\widehat{h}^{-,\gamma_1}(\widehat{g}^{-,\gamma_1}) \le \widehat{h}^{-,\gamma_1}(\widehat{g}^{-,\gamma_2}) \le \widehat{h}^{-,\gamma_2}(\widehat{g}^{-,\gamma_2})$。这个结果类似于 Dinkelbach [30] 的引理3。

**C.2.2 第2部分：** $\widehat{e}_{\text{orig}}(\widehat{g}^{-,\gamma})$ 在 $\gamma$ 中单调递减。

令 $\gamma_1, \gamma_2 \in \mathbb{R}$ 满足 $\gamma_1 < \gamma_2$。通过推导可得 $\widehat{e}_{\text{orig}}(\widehat{g}^{-,\gamma_1}) \ge \widehat{e}_{\text{orig}}(\widehat{g}^{-,\gamma_2})$。

**C.2.3 第3部分：** $\left\{ \frac{\widehat{h}^{-,\gamma}(\widehat{g}^{-,\gamma})}{\epsilon_{\text{abs}}} - \gamma \right\}$ 在 $\widehat{e}_{\text{orig}}(\widehat{g}^{-,\gamma}) \le \epsilon_{\text{abs}}$ 范围内随 $\gamma$ 单调递减，否则递增。

详细推导见正文附录。

### C.3 命题11的证明（MR 下界二分搜索的非负权重）

令 $\gamma_1 := \frac{1}{n-1}$。首先我们证明存在最小化 $\widehat{h}^{-,\gamma_1}$ 的函数 $\widehat{g}^{-,\gamma_1}$ 使得 $\widehat{\text{MR}}(\widehat{g}^{-,\gamma_1}) = 1$。从 $\gamma_1 = \frac{1}{n-1}$ 和式6.2，我们看到 $\widehat{h}^{-,\gamma_1}(f) \propto \mathbb{E}_{D_m} L\{f, (Y, X_1, X_2)\}$，其中 $D_m$ 是使得 $X_1$ 独立于 $(Y, X_2)$ 的分布。因此，从命题11的条件2，我们知道存在最小化 $\widehat{h}^{-,\gamma_1}$ 的函数 $\widehat{g}^{-,\gamma_1}$ 满足 $\widehat{g}^{-,\gamma_1}(x_1^{(a)}, x_2) = \widehat{g}^{-,\gamma_1}(x_1^{(b)}, x_2)$。命题11的条件1接着意味着 $\widehat{e}_{\text{switch}}(\widehat{g}^{-,\gamma_1}) = \widehat{e}_{\text{orig}}(\widehat{g}^{-,\gamma_1})$，因此 $\widehat{\text{MR}}(\widehat{g}^{-,\gamma_1}) = 1$。

令 $\gamma_2 = 0$。对于任何最小化 $\widehat{h}^{-,\gamma_2}$ 的函数 $\widehat{g}^{-,\gamma_2}$，我们知道 $\widehat{h}^{-,\gamma_2}(\widehat{g}^{-,\gamma_2}) \le \widehat{h}^{-,\gamma_2}(\widehat{g}^{-,\gamma_1})$，因此 $\widehat{e}_{\text{switch}}(\widehat{g}^{-,\gamma_2}) \le \widehat{e}_{\text{switch}}(\widehat{g}^{-,\gamma_1})$。从 $\gamma_2 \le \gamma_1$ 和引理10的第2部分，我们知道 $\widehat{e}_{\text{orig}}(\widehat{g}^{-,\gamma_2}) \ge \widehat{e}_{\text{orig}}(\widehat{g}^{-,\gamma_1})$。结合这些，$\widehat{\text{MR}}(\widehat{g}^{-,\gamma_2}) \le \widehat{\text{MR}}(\widehat{g}^{-,\gamma_1}) = 1$。

由于 $\widehat{h}^{-,\gamma_2}(\widehat{g}^{-,\gamma_2}) = \widehat{e}_{\text{switch}}(\widehat{g}^{-,\gamma_2}) \ge 0$ 根据定义，条件8对 $\gamma_2, \epsilon_{\text{abs}}$ 和 $\widehat{g}^{-,\gamma_2}$ 成立当且仅当 $\widehat{e}_{\text{orig}}(\widehat{g}^{-,\gamma_2}) \le \epsilon_{\text{abs}}$。结合式C.10，这完成了证明。

如果我们替换 $\widehat{e}_{\text{switch}}$ 为 $\widehat{e}_{\text{divide}}$，相同的结果不一定成立。

### C.4 引理13的证明（MR 的上界）

我们分2部分证明引理13。

**C.4.1 第1部分：** 证明式6.4对所有满足 $\widehat{e}_{\text{orig}}(f) \le \epsilon_{\text{abs}}$ 的 $f \in \mathcal{F}$ 成立。

如果 $\widehat{h}^{+,\gamma}(\widehat{g}^{+,\gamma}) \ge 0$，则对于任何满足 $\widehat{e}_{\text{orig}}(f) \le \epsilon_{\text{abs}}$ 的函数 $f \in \mathcal{F}$，我们知道 $\frac{1}{\epsilon_{\text{abs}}} \le \frac{1}{\widehat{e}_{\text{orig}}(f)}$ 且 $\frac{\widehat{h}^{+,\gamma}(\widehat{g}^{+,\gamma})}{\epsilon_{\text{abs}}} \le \frac{\widehat{h}^{+,\gamma}(\widehat{g}^{+,\gamma})}{\widehat{e}_{\text{orig}}(f)}$。

现在，如果 $\gamma \le 0$，则对于任何满足 $\widehat{e}_{\text{orig}}(f) \le \epsilon_{\text{abs}}$ 的 $f \in \mathcal{F}$，$\widehat{g}^{+,\gamma}$ 的定义意味着 $\widehat{h}^{+,\gamma}(f) \ge \widehat{h}^{+,\gamma}(\widehat{g}^{+,\gamma})$，因此 $\widehat{\text{MR}}(f) \le \frac{(\frac{\widehat{h}^{+,\gamma}(\widehat{g}^{+,\gamma})}{\epsilon_{\text{abs}}} - 1)}{\gamma^{-1}}$。

**C.4.2 第2部分：** 证明如果 $f = \widehat{g}^{+,\gamma}$ 且条件12中的至少一个不等式取等号，则式6.4取等号。

分别考虑条件12中的两个不等式，当 $\widehat{h}^{+,\gamma}(\widehat{g}^{+,\gamma}) = 0$ 或 $\widehat{e}_{\text{orig}}(\widehat{g}^{+,\gamma}) = \epsilon_{\text{abs}}$ 时，等号成立。

### C.5 引理14的证明（MR 上界二分搜索的单调性）

分3部分证明，与引理10类似但符号相反。

### C.6 备注16的证明（线性模型类经验 MCR 的可处理性）

应用命题15，可得 $\xi_{\text{orig}} \widehat{e}_{\text{orig}}(f_\beta) + \xi_{\text{switch}} \widehat{e}_{\text{switch}}(f_\beta) \propto -2\mathbf{q}'\beta + \beta' \mathbf{Q} \beta$。

### C.7 引理17的证明（线性模型的损失上界）

在引理17和式7.5的条件下，我们可以通过最大化或最小化 $\mathbf{x}'\beta$ 构造 $L(f_\beta, (y, \mathbf{x})) = (y - \mathbf{x}'\beta)^2$ 的上界。通过变量替换和特征分解，我们得到最大可能值为 $\sqrt{r_X r_{\text{lm}}}$，最小可能值为 $-\sqrt{r_X r_{\text{lm}}}$。因此，损失有界。

### C.8 引理18的证明（RKHS 中回归的损失上界）

证明遵循与第C.7节类似的结构。从引理18的假设，最大可能输出为 $\mu + \sqrt{r_D r_k}$，最小可能输出为 $-(\mu + \sqrt{r_D r_k})$。因此，损失有界。

## 参考文献

[1] André Altmann, Laura Toloși, Oliver Sander, and Thomas Lengauer. Permutation importance: a corrected feature importance measure. *Bioinformatics*, 26(10):1340–1347, 2010.

[2] Kellie J Archer and Ryan V Kimes. Empirical characterization of random forest variable importance measures. *Computational Statistics & Data Analysis*, 52(4):2249–2260, 2008.

[3] Razia Azen, David V Budescu, and Benjamin Reiser. Criticality of predictors in multiple regression. *British Journal of Mathematical and Statistical Psychology*, 54(2):201–225, 2001.

[4] Katherine Beckett, Kris Nyrop, and Lori Pfingst. Race, drugs, and policing: understanding disparities in drug delivery arrests. *Criminology*, 44(1):105–137, 2006.

[5] Irene V Blair, Charles M Judd, and Kristine M Chapleau. The influence of afrocentric facial features in criminal sentencing. *Psychological science*, 15(10):674–679, 2004.

[6] Stephen Boyd and Lieven Vandenberghe. *Convex Optimization*. Cambridge university press, 2004.

[7] Leo Breiman. Random forests. *Machine learning*, 45(1):5–32, 2001.

[8] Leo Breiman et al. Statistical modeling: the two cultures (with comments and a rejoinder by the author). *Statistical science*, 16(3):199–231, 2001.

[9] M Luz Calle and Víctor Urrea. Letter to the editor: stability of random forest importance measures. *Briefings in bioinformatics*, 12(1):86–89, 2010.

[10] Hugh A Chipman, Edward I George, Robert E McCulloch, et al. Bart: Bayesian additive regression trees. *The Annals of Applied Statistics*, 4(1):266–298, 2010.

[11] Alexandra Chouldechova. Fair prediction with disparate impact: a study of bias in recidivism prediction instruments. *Big data*, 5(2):153–163, 2017.

[12] Beau Coker, Cynthia Rudin, and Gary King. A theory of statistical inference for ensuring the robustness of scientific results. *arXiv preprint arXiv:1804.08646*, 2018.

[13] Sam Corbett-Davies, Emma Pierson, Avi Feller, and Sharad Goel. A computer program used for bail and sentencing decisions was labeled biased against blacks. it's actually not that clear. *The Washington Post*, October 2016. URL https://www.washingtonpost.com/news/monkey-cage/wp/2016/10/17/can-an-algorithm-be-racist-our-analysis-is-more-cautious-than-propublicas/.

[14] Sam Corbett-Davies, Emma Pierson, Avi Feller, Sharad Goel, and Aziz Huq. Algorithmic decision making and the cost of fairness. In *Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, pages 797–806. ACM, 2017.

[15] Anupam Datta, Shayak Sen, and Yair Zick. Algorithmic transparency via quantitative input influence: theory and experiments with learning systems. In *Security and Privacy (SP), 2016 IEEE Symposium on*, pages 598–617. IEEE, 2016.

[16] Elizabeth R DeLong, David M DeLong, and Daniel L Clarke-Pearson. Comparing the areas under two or more correlated receiver operating characteristic curves: a nonparametric approach. *Biometrics*, 44(3):837–845, 1988.

[17] Olga V Demler, Michael J Pencina, and Ralph B D'Agostino Sr. Misuse of delong test to compare aucs for nested models. *Statistics in medicine*, 31(23):2577–2587, 2012.

[18] Iván Díaz, Alan Hubbard, Anna Decker, and Mitchell Cohen. Variable importance and prediction methods for longitudinal problems with missing variables. *PloS one*, 10(3):e0120031, 2015.

[19] Werner Dinkelbach. On nonlinear fractional programming. *Management science*, 13(7):492–498, 1967.

[20] Jiayun Dong and Cynthia Rudin. Variable importance clouds: A way to explore variable importance for the set of good models. *arXiv preprint arXiv:1901.03209*, 2019.

[21] R Dorfman. A note on the delta-method for finding variance formulae. *The Biometric Bulletin*, 1(129-137):92, 1938.

[22] Cynthia Dwork, Moritz Hardt, Toniann Pitassi, Omer Reingold, and Richard Zemel. Fairness through awareness. In *Proceedings of the 3rd innovations in theoretical computer science conference*, pages 214–226. ACM, 2012.

[23] Muriel Gevrey, Ioannis Dimopoulos, and Sovan Lek. Review and comparison of methods to study the contribution of variables in artificial neural network models. *Ecological modelling*, 160(3):249–264, 2003.

[24] Baptiste Gregorutti, Bertrand Michel, and Philippe Saint-Pierre. Grouped variable importance with random forests and application to multiple functional data analysis. *Computational Statistics & Data Analysis*, 90:15–35, 2015.

[25] Baptiste Gregorutti, Bertrand Michel, and Philippe Saint-Pierre. Correlation and variable importance in random forests. *Statistics and Computing*, 27(3):659–678, 2017.

[26] Alexander Hapfelmeier, Torsten Hothorn, Kurt Ulm, and Carolin Strobl. A new variable importance measure for random forests with missing data. *Statistics and Computing*, 24(1):21–34, 2014.

[27] T Hastie, R Tibshirani, and J Friedman. *The elements of statistical learning* 2nd edition. New York: Springer, 2009.

[28] Karl G Heider. The Rashomon effect: when ethnographers disagree. *American Anthropologist*, 90(1):73–81, 1988.

[29] Wassily Hoeffding. A class of statistics with asymptotically normal distribution. *The annals of mathematical statistics*, pages 293–325, 1948.

[30] Wassily Hoeffding. Probability inequalities for sums of bounded random variables. *Journal of the American Statistical Association*, 58(301):13–30, 1963. doi: 10.1080/01621459.1963.10500830.

[31] Giles Hooker. Generalized functional anova diagnostics for high-dimensional functions of dependent variables. *Journal of Computational and Graphical Statistics*, 16(3):709–732, 2007.

[32] Reiner Horst and Nguyen V Thoai. Dc programming: overview. *Journal of Optimization Theory and Applications*, 103(1):1–43, 1999.

[33] Faisal Kamiran, Indrė Žliobaitė, and Toon Calders. Quantifying explainable discrimination and removing illegal discrimination in automated decision making. *Knowledge and information systems*, 35(3):613–644, 2013.

[34] Jalil Kazemitabar, Arash Amini, Adam Bloniarz, and Ameet S Talwalkar. Variable importance using decision trees. In *Advances in Neural Information Processing Systems*, pages 425–434, 2017.

[35] Jon Kleinberg, Sendhil Mullainathan, and Manish Raghavan. Inherent trade-offs in the fair determination of risk scores. In *8th Innovations in Theoretical Computer Science Conference (ITCS 2017)*. Schloss Dagstuhl-Leibniz-Zentrum fuer Informatik, 2017.

[36] Jeff Larson, Surya Mattu, Lauren Kirchner, and Julia Angwin. How we analyzed the compas recidivism algorithm. *ProPublica*, May 2016. URL https://www.propublica.org/article/how-we-analyzed-the-compas-recidivism-algorithm.

[37] Guillaume Lecué. *Interplay between concentration, complexity and geometry in learning theory with applications to high dimensional data analysis*. PhD thesis, Université Paris-Est, 2011.

[38] Erich L Lehmann and George Casella. *Theory of point estimation*. Springer Science & Business Media, 2006.

[39] Benjamin Letham, Portia A Letham, Cynthia Rudin, and Edward P Browne. Prediction uncertainty and optimal experimental design for learning dynamical systems. *Chaos: An Interdisciplinary Journal of Nonlinear Science*, 26(6):063110, 2016.

[40] Gilles Louppe, Louis Wehenkel, Antonio Sutera, and Pierre Geurts. Understanding variable importances in forests of randomized trees. In *Advances in neural information processing systems*, pages 431–439, 2013.

[41] Kristian Lum and William Isaac. To predict and serve? *Significance*, 13(5):14–19, 2016.

[42] Nicolai Meinshausen and Peter Bühlmann. Stability selection. *Journal of the Royal Statistical Society: Series B (Statistical Methodology)*, 72(4):417–473, 2010.

[43] Lucas Mentch and Giles Hooker. Quantifying uncertainty in random forests via confidence intervals and hypothesis tests. *The Journal of Machine Learning Research*, 17(1):841–881, 2016.

[44] John Monahan and Jennifer L Skeem. Risk assessment in criminal sentencing. *Annual review of clinical psychology*, 12:489–513, 2016.

[45] Razieh Nabi and Ilya Shpitser. Fair inference on outcomes. In *Proceedings of the... AAAI Conference on Artificial Intelligence. AAAI Conference on Artificial Intelligence*, volume 2018, page 1931. NIH Public Access, 2018.

[46] Daniel Nevo and Ya'acov Ritov. Identifying a minimal class of models for high-dimensional data. *The Journal of Machine Learning Research*, 18(1):797–825, 2017.

[47] Julian D Olden, Michael K Joy, and Russell G Death. An accurate comparison of methods for quantifying variable importance in artificial neural networks using simulated data. *Ecological Modelling*, 178(3):389–397, 2004.

[48] Jaehyun Park and Stephen Boyd. General heuristics for nonconvex quadratically constrained quadratic programming. *arXiv preprint arXiv:1703.07870*, 2017.

[49] Raymond Paternoster and Robert Brame. Reassessing race disparities in maryland capital cases. *Criminology*, 46(4):971–1008, 2008.

[50] Sarah Picard-Fritsche, Michael Rempel, Jennifer A. Tallon, Julian Adler, and Natalie Reyes. Demystifying risk assessment, key principles and controversies. Technical report, 2017. Available at https://www.courtinnovation.org/publications/demystifying-risk-assessment-key-principles-and-controversies.

[51] Imre Pólik and Tamás Terlaky. A survey of the s-lemma. *SIAM review*, 49(3):371–418, 2007.

[52] Rajeev Ramchand, Rosalie Liccardo Pacula, and Martin Y Iguchi. Racial differences in marijuana-users' risk of arrest in the united states. *Drug and alcohol dependence*, 84(3):264–272, 2006.

[53] Friedrich Recknagel, Mark French, Pia Harkonen, and Ken-Ichi Yabunaka. Artificial neural network approach for modelling and prediction of algal blooms. *Ecological Modelling*, 96(1-3):11–28, 1997.

[54] Wendy D Roth and Jal D Mehta. The Rashomon effect: combining positivist and interpretivist approaches in the analysis of contested events. *Sociological Methods & Research*, 31(2):131–173, 2002.

[55] Cynthia Rudin. Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead. *Nature Machine Intelligence*, 1:206–215, May 2019.

[56] Cynthia Rudin, Caroline Wang, and Beau Coker. The age of secrecy and unfairness in recidivism prediction. *Harvard Data Science Review*, 2019. accepted.

[57] Michele Scardi and Lawrence W Harding. Developing an empirical model of phytoplankton primary production: a neural network case study. *Ecological modelling*, 120(2):213–223, 1999.

[58] Lesia Semenova and Cynthia Rudin. A study in rashomon curves and volumes: a new perspective on generalization and model simplicity in machine learning. *arXiv preprint arXiv:1908.01755*, 2019.

[59] Robert J Serfling. *Approximation theorems of mathematical statistics*. John Wiley & Sons, 1980.

[60] Cassia Spohn. Thirty years of sentencing reform: the quest for a racially neutral sentencing process. *Criminal justice*, 3:427–501, 2000.

[61] Alexander Statnikov, Nikita I Lytkin, Jan Lemeire, and Constantin F Aliferis. Algorithms for discovery of multiple markov boundaries. *Journal of Machine Learning Research*, 14(Feb):499–566, 2013.

[62] Carolin Strobl, Anne-Laure Boulesteix, Achim Zeileis, and Torsten Hothorn. Bias in random forest variable importance measures: illustrations, sources and a solution. *BMC bioinformatics*, 8(1):25, 2007.

[63] Carolin Strobl, Anne-Laure Boulesteix, Thomas Kneib, Thomas Augustin, and Achim Zeileis. Conditional variable importance for random forests. *BMC bioinformatics*, 9(1):307, 2008.

[64] Elizabeth A Stuart. Matching methods for causal inference: a review and a look forward. *Statistical science: a review journal of the Institute of Mathematical Statistics*, 25(1):1, 2010.

[65] Laura Toloși and Thomas Lengauer. Classification with correlated features: unreliability of feature ranking and solutions. *Bioinformatics*, 27(14):1986–1994, 2011.

[66] Theja Tulabandhula and Cynthia Rudin. Robust optimization using machine learning for uncertainty sets. *arXiv preprint arXiv:1407.1097*, 2014.

[67] U.S. Department of Justice - Civil Rights Division. Investigation of the Baltimore City Police Department, August 2016. Available at https://www.justice.gov/crt/file/883296/download.

[68] Mark J van der Laan. Statistical inference for variable importance. *The International Journal of Biostatistics*, 2(1), 2006.

[69] Jay M Ver Hoef. Who invented the delta method? *The American Statistician*, 66(2):124–127, 2012.

[70] Huazhen Wang, Fan Yang, and Zhiyuan Luo. An experimental study of the intrinsic stability of random forest variable importance measures. *BMC bioinformatics*, 17(1):60, 2016.

[71] Brian D Williamson, Peter B Gilbert, Noah Simon, and Marco Carone. Nonparametric variable importance assessment using machine learning techniques. *bepress (unpublished preprint)*, 2017.

[72] Jingtao Yao, Nicholas Teng, Hean-Lee Poh, and Chew Lim Tan. Forecasting and analysis of marketing data using neural networks. *J. Inf. Sci. Eng.*, 14(4):843–862, 1998.

[73] Ruoqing Zhu, Donglin Zeng, and Michael R Kosorok. Reinforcement learning trees. *Journal of the American Statistical Association*, 110(512):1770–1784, 2015.
