# 端侧集成重排序与异构行为建模

> Yunjia Xi, Weiwen Liu, Yang Wang, Ruiming Tang, Weinan Zhang, Yue Zhu, Rui Zhang, Yong Yu | Shanghai Jiao Tong University, Huawei Noah's Ark Lab, East China Normal University

本文提出了首个端侧深度集成重排序框架 DIR（on-device Integrated Re-ranking），将来自多个上游来源的候选列表融合为统一排序列表，并通过多序列行为建模捕获用户在不同来源上的异构偏好。核心发现是——**DIR 在两个公开基准数据集和一个工业数据集上均显著优于现有重排序方法，在 NDCG@5 上提升最高达 5.9%**。

核心内容：
- 现有集成重排序方法采用云-边框架，将用户行为历史存储在云端，无法捕获设备端的实时用户偏好，且忽略了不同来源（视频、文章、图片）间的异构行为模式
- 提出 DIR 框架：首个端侧深度集成重排序模型，利用设备端实时行为历史进行个性化重排序
- 设计多序列行为建模模块：通过源内交互和高阶共享机制，从多个异构行为序列中学习用户的源级别偏好
- 引入偏好自适应重排序模块：将用户源级别偏好注入到排序模型的个性化参数中，实现列表级上下文建模；同时设计效用-曝光损失函数保证来源间的曝光公平性

关键发现：
- **DIR 在 KuaiRec、MicroVideo 和工业数据集上相比最强基线，在 NDCG@5 上分别提升 5.9%、3.2% 和 2.8%**
- 多序列行为建模模块的贡献最大，移除后性能下降 3.1%-4.7%；偏好自适应机制相比固定参数方案提升 1.5%-2.3%
- 不同用户在各来源上的偏好差异显著，源级别偏好建模对重排序效果至关重要
- 未来方向：探索更复杂的异构行为交互模式，以及在更多下游任务中验证框架的通用性

---

## 摘要

集成重排序（Integrated Re-ranking）是多阶段推荐系统中的关键环节，它将来自多个上游来源（如不同的推荐通道或排序模型）的候选列表融合为一个统一的排序列表。现有的集成重排序方法通常采用云-边（cloud-to-edge）框架，将用户行为历史存储在云端服务器上进行模型训练和推理。然而，这种框架无法捕获用户在设备端的实时行为偏好，导致重排序结果与用户当前兴趣存在偏差。此外，用户在不同来源上的行为存在显著的异构性——例如，用户可能偏好在视频来源中观看短视频，而在文章来源中阅读深度报道——现有方法未能充分建模这种异构行为模式。

为解决上述问题，本文提出了 DIR（on-device Integrated Re-ranking），首个端侧深度集成重排序框架。DIR 部署在用户设备端，能够直接访问用户的实时行为历史。具体而言，DIR 包含三个核心模块：（1）异构编码模块（Heterogeneous Encoding Module），对来自不同来源的候选 item 进行统一的特征编码；（2）多序列行为建模模块（Multi-sequence Behavior Modeling Module），通过源内交互和高阶共享机制，从多个异构行为序列中学习用户的源级别偏好；（3）偏好自适应重排序模块（Preference-adaptive Re-ranking Module），将用户的源级别偏好注入到排序模型的个性化参数中，实现列表级上下文建模。此外，本文设计了效用-曝光损失函数（Utility and Exposure Loss），在优化排序效用的同时保证不同来源间的曝光公平性。

在两个公开基准数据集（KuaiRec 和 MicroVideo）和一个工业数据集上的大量实验表明，DIR 显著优于现有的重排序方法，在 NDCG@5 上相比最强基线分别提升 5.9%、3.2% 和 2.8%。

**关键词：** integrated re-ranking, on-device recommendation, heterogeneous behavior modeling, preference-adaptive ranking, exposure fairness

## 1 引言

推荐系统通常采用多阶段架构，包括召回、粗排、精排和重排序等阶段 [1, 2]。在重排序阶段，系统需要将来自多个上游来源的候选列表融合为一个统一的排序列表，这一过程被称为集成重排序（Integrated Re-ranking）[3, 4]。集成重排序在实际推荐场景中具有广泛应用，例如：在短视频推荐中，系统需要将来自协同过滤、基于内容的推荐和热门推荐等多个来源的视频列表融合为一个统一的推荐列表 [5]；在信息流推荐中，系统需要将来自不同频道的新闻文章进行统一排序 [6]。

现有的集成重排序方法通常采用云-边框架 [7, 8]。在这种框架下，用户的行为历史被上传到云端服务器，模型在云端进行训练和推理，然后将排序结果下发到用户设备端。然而，这种框架存在两个主要问题：

**问题一：无法捕获设备端的实时用户偏好。** 用户的行为偏好是动态变化的，用户在设备端产生的实时行为（如最近的点击、浏览、分享等）能够更准确地反映其当前兴趣。然而，由于数据同步延迟和隐私保护等原因，这些实时行为往往无法及时上传到云端，导致云端模型使用的用户行为历史是滞后的 [9, 10]。这使得重排序结果与用户的实时兴趣存在偏差，影响用户体验。

**问题二：忽略了不同来源间的异构行为模式。** 用户在不同来源上的行为存在显著差异 [11, 12]。例如，用户可能在视频来源中偏好观看短视频，而在文章来源中偏好阅读深度报道；用户可能在某个来源上表现出高度活跃，而在另一个来源上相对冷淡。这种异构行为模式反映了用户在不同内容类型上的差异化偏好，现有方法未能充分建模这种异构性，导致重排序结果无法准确反映用户的真实偏好。

为解决上述问题，本文提出了 DIR（on-device Integrated Re-ranking），首个端侧深度集成重排序框架。与现有方法不同，DIR 直接部署在用户设备端，能够访问用户的实时行为历史，从而捕获用户的实时偏好。同时，DIR 通过多序列行为建模模块，显式地建模用户在不同来源上的异构行为模式，学习用户的源级别偏好。

具体而言，DIR 的核心设计包括以下三个模块：

1. **异构编码模块（Heterogeneous Encoding Module）：** 对来自不同来源的候选 item 进行统一的特征编码。由于不同来源的 item 具有不同的特征空间，该模块通过共享的嵌入层将不同来源的 item 特征映射到统一的 latent 空间中，为后续的行为建模和重排序提供基础。

2. **多序列行为建模模块（Multi-sequence Behavior Modeling Module）：** 用户在每个来源上都有独立的行为序列，该模块通过源内交互（intra-source interaction）机制，使用 Transformer 编码器 [13] 对每个来源内的行为序列进行建模，捕获用户在该来源上的兴趣演化。同时，通过高阶共享（high-order sharing）机制，在不同来源的行为序列之间进行信息交互，捕获跨来源的兴趣迁移和共享模式。最终，该模块输出用户的源级别偏好表示，反映用户在不同来源上的差异化偏好。

3. **偏好自适应重排序模块（Preference-adaptive Re-ranking Module）：** 将用户的源级别偏好注入到排序模型的个性化参数中，使模型能够根据用户的源级别偏好动态调整排序策略。同时，该模块采用列表级上下文建模（listwise context modeling），考虑候选列表中 item 之间的相互影响，生成最终的排序结果。

此外，本文设计了效用-曝光损失函数（Utility and Exposure Loss），在优化排序效用（即用户对排序结果的满意度）的同时，保证不同来源间的曝光公平性，避免某些来源被过度曝光而其他来源被忽视。

本文的主要贡献总结如下：

- 提出 DIR，首个端侧深度集成重排序框架，能够在用户设备端进行实时的个性化重排序，解决了现有云-边框架无法捕获实时用户偏好的问题。
- 设计多序列行为建模模块，通过源内交互和高阶共享机制，显式地建模用户在不同来源上的异构行为模式，学习用户的源级别偏好。
- 引入偏好自适应重排序机制，将用户的源级别偏好注入到排序模型的个性化参数中，实现列表级上下文建模，提升重排序的个性化程度。
- 在两个公开基准数据集和一个工业数据集上进行大量实验，验证了 DIR 的有效性和各模块的贡献。

## 2 相关工作

### 2.1 重排序

重排序是多阶段推荐系统中的最后一个阶段，其目标是对上游阶段生成的候选列表进行重新排序，以生成更符合用户偏好的推荐列表 [1, 14]。现有的重排序方法可以分为以下几类：

**基于贪心的方法。** 这类方法采用贪心策略，逐个选择最优的 item 添加到排序列表中 [15, 16]。例如，PRM（Personalized Re-ranking Model）[15] 使用 RNN 对已排序的 item 序列进行编码，并预测下一个最优的 item。贪心方法简单高效，但无法保证全局最优。

**基于排列的方法。** 这类方法将重排序建模为排列生成问题，直接生成候选列表的最优排列 [17, 18]。例如，SetRank [17] 使用自注意力机制对候选列表中的所有 item 进行交互建模，生成最终的排序结果。基于排列的方法能够捕获 item 之间的全局交互，但计算复杂度较高。

**基于强化学习的方法。** 这类方法将重排序建模为序列决策问题，使用强化学习算法学习最优的排序策略 [19, 20]。例如，PRM-RL [19] 使用策略梯度方法优化重排序策略，能够在长期收益和即时收益之间取得平衡。

**集成重排序。** 集成重排序是重排序的一个重要分支，其目标是将来自多个上游来源的候选列表融合为一个统一的排序列表 [3, 4]。与普通重排序不同，集成重排序需要处理来自不同来源的异构候选列表，并考虑不同来源之间的曝光公平性问题。现有的集成重排序方法通常采用云-边框架，无法捕获设备端的实时用户偏好。

本文提出的 DIR 是首个端侧深度集成重排序框架，能够在用户设备端进行实时的个性化重排序，解决了现有方法的局限性。

### 2.2 端侧推荐

随着移动设备计算能力的提升和隐私保护需求的增加，端侧推荐（on-device recommendation）成为一个重要的研究方向 [9, 21, 22]。端侧推荐将推荐模型部署在用户设备端，能够直接访问用户的本地数据，具有以下优势：（1）能够捕获用户的实时行为偏好，提升推荐的时效性；（2）无需将用户数据上传到云端，保护用户隐私；（3）减少数据传输延迟，提升用户体验。

现有的端侧推荐方法主要关注以下方面：（1）模型压缩，通过知识蒸馏 [23]、模型剪枝 [24] 等技术减小模型体积，使其能够在资源受限的设备端运行；（2）增量学习，在设备端对模型进行增量更新，使模型能够适应用户的实时行为变化 [25]；（3）联邦学习，在保护用户隐私的前提下，利用多个设备端的数据进行模型训练 [26]。

与现有端侧推荐方法不同，DIR 专注于端侧集成重排序任务，通过多序列行为建模和偏好自适应机制，在设备端实现高质量的个性化重排序。

### 2.3 异构行为建模

用户在不同场景或来源上的行为存在显著差异，这种异构行为模式对推荐系统的设计提出了挑战 [11, 27]。现有的异构行为建模方法主要关注以下方面：

**多行为建模。** 这类方法同时建模用户的多种行为类型（如点击、收藏、分享等），学习不同行为类型之间的关联和差异 [28, 29]。例如，MBRec [28] 使用图神经网络建模用户- item 之间的多种交互行为，学习用户的多维兴趣表示。

**跨域建模。** 这类方法利用用户在不同领域（如电影、音乐、新闻等）的行为数据，学习跨领域的用户偏好 [30, 31]。例如，EMCDR [30] 使用映射函数将用户在源领域中的偏好表示迁移到目标领域中。

**多源建模。** 这类方法同时利用来自多个数据源的行为数据，学习更全面的用户偏好表示 [32, 33]。例如，MVIN [32] 使用多视角注意力机制对来自多个数据源的行为进行建模，学习用户的多视角兴趣表示。

与现有异构行为建模方法不同，DIR 专注于建模用户在不同推荐来源上的异构行为模式，通过多序列行为建模模块，显式地学习用户的源级别偏好，并将其注入到重排序模型的个性化参数中。

## 3 问题定义

本节对端侧集成重排序问题进行形式化定义。

### 3.1 问题定义

设 $\\mathcal{U} = \\{u_1, u_2, \\ldots, u_{|\\mathcal{U}|}\\}$ 为用户集合，$\\mathcal{V} = \\{v_1, v_2, \\ldots, v_{|\\mathcal{V}|}\\}$ 为 item 集合。在集成重排序场景中，系统接收来自 $K$ 个不同上游来源的候选列表。对于用户 $u$，第 $k$ 个来源生成的候选列表记为 $\\mathcal{L}_k = [v_{k,1}, v_{k,2}, \\ldots, v_{k,n_k}]$，其中 $v_{k,i}$ 表示第 $k$ 个来源中的第 $i$ 个候选 item，$n_k$ 为第 $k$ 个来源的候选列表长度。

每个候选 item $v$ 具有特征向量 $\\mathbf{x}_v \\in \\mathbb{R}^d$，其中 $d$ 为特征维度。用户 $u$ 在第 $k$ 个来源上的行为序列为 $\\mathcal{S}_k^u = [v_{k,1}^u, v_{k,2}^u, \\ldots, v_{k,m_k}^u]$，其中 $v_{k,i}^u$ 表示用户 $u$ 在第 $k$ 个来源上交互过的第 $i$ 个 item，$m_k$ 为该行为序列的长度。

集成重排序的目标是学习一个重排序函数 $f$，将来自 $K$ 个来源的候选列表融合为一个统一的排序列表：

$$
\\mathcal{L}^* = f(\\mathcal{L}_1, \\mathcal{L}_2, \\ldots, \\mathcal{L}_K; \\mathcal{S}_1^u, \\mathcal{S}_2^u, \\ldots, \\mathcal{S}_K^u; \\mathbf{x})
$$

其中 $\\mathcal{L}^*$ 为重排序后的统一列表，$\\mathbf{x}$ 为所有候选 item 的特征矩阵。

### 3.2 设计目标

在设计端侧集成重排序模型时，需要考虑以下目标：

1. **效用最大化（Utility Maximization）：** 重排序结果应尽可能满足用户偏好，即用户真正感兴趣的 item 应排在列表的前列。

2. **曝光公平性（Exposure Fairness）：** 不同来源的 item 应获得合理的曝光机会，避免某些来源被过度曝光而其他来源被忽视。曝光公平性对于维护多来源推荐生态系统的健康运行至关重要。

3. **实时性（Real-time）：** 由于模型部署在设备端，需要在有限的计算资源和时间内完成重排序推理，保证用户体验。

4. **个性化（Personalization）：** 重排序结果应充分考虑用户的个性化偏好，包括用户在不同来源上的差异化偏好。

## 4 方法

本节详细介绍 DIR 框架的设计。如图 1 所示，DIR 由三个核心模块组成：异构编码模块（4.1 节）、多序列行为建模模块（4.2 节）和偏好自适应重排序模块（4.3 节）。此外，本节还介绍了效用-曝光损失函数的设计（4.4 节）。



![图1](/Users/dazhang/PycharmProject/Papers/9-app/.picture/2023-DIR-On-device Integrated Re-ranking with Heterogeneous Behavior Modeling-fig1.png)

图 1：端侧集成重排序（Edge Integrated Re-ranking）示意图。来自多个上游来源的候选列表在设备端进行融合重排序，设备端保存着用户的实时行为历史，能够捕获用户的实时偏好。](.picture/image-2023DIR-fig1.png)




图 1：端侧集成重排序示意图。来自多个上游来源的候选列表在设备端进行融合重排序，设备端保存着用户的实时行为历史，能够捕获用户的实时偏好。

### 4.1 异构编码模块

异构编码模块的目标是对来自不同来源的候选 item 进行统一的特征编码。由于不同来源的 item 具有不同的特征空间，该模块需要将不同来源的 item 特征映射到统一的 latent 空间中。

对于第 $k$ 个来源中的候选 item $v_{k,i}$，其原始特征向量为 $\\mathbf{x}_{k,i} \\in \\mathbb{R}^{d_k}$，其中 $d_k$ 为第 $k$ 个来源的特征维度。异构编码模块通过来源特定的嵌入层将原始特征映射到统一的 latent 空间：

$$
\\mathbf{e}_{k,i} = \\text{Embed}_k(\\mathbf{x}_{k,i}) \\in \\mathbb{R}^d
$$

其中 $\\text{Embed}_k$ 为第 $k$ 个来源的嵌入函数，$d$ 为统一的 latent 空间维度。

类似地，用户行为序列中的 item 也通过对应的嵌入层进行编码。对于用户 $u$ 在第 $k$ 个来源上的行为序列 $\\mathcal{S}_k^u$，编码后的行为序列表示为：

$$
\\mathbf{E}_k^u = [\\mathbf{e}_{k,1}^u, \\mathbf{e}_{k,2}^u, \\ldots, \\mathbf{e}_{k,m_k}^u] \\in \\mathbb{R}^{m_k \\times d}
$$

### 4.2 多序列行为建模模块

多序列行为建模模块是 DIR 的核心模块，其目标是从用户的多个异构行为序列中学习用户的源级别偏好。该模块由两个子模块组成：源内交互子模块和高阶共享子模块。

#### 4.2.1 源内交互

源内交互子模块使用 Transformer 编码器 [13] 对每个来源内的行为序列进行建模，捕获用户在该来源上的兴趣演化。

对于用户 $u$ 在第 $k$ 个来源上的行为序列 $\\mathbf{E}_k^u$，源内交互子模块通过 Transformer 编码器对其进行编码：

$$
\\mathbf{H}_k^u = \\text{Transformer}_k(\\mathbf{E}_k^u) \\in \\mathbb{R}^{m_k \\times d}
$$

其中 $\\mathbf{H}_k^u$ 为编码后的行为序列表示，$\\text{Transformer}_k$ 为第 $k$ 个来源的 Transformer 编码器。

为了得到用户在第 $k$ 个来源上的整体兴趣表示，对编码后的行为序列进行平均池化：

$$
\\mathbf{p}_k^u = \\frac{1}{m_k} \\sum_{i=1}^{m_k} \\mathbf{h}_{k,i}^u \\in \\mathbb{R}^d
$$

其中 $\\mathbf{h}_{k,i}^u$ 为 $\\mathbf{H}_k^u$ 中第 $i$ 个位置的表示，$\\mathbf{p}_k^u$ 为用户 $u$ 在第 $k$ 个来源上的兴趣表示。

#### 4.2.2 高阶共享

高阶共享子模块在不同来源的行为序列之间进行信息交互，捕获跨来源的兴趣迁移和共享模式。具体而言，该模块使用多头注意力机制 [13] 对所有来源的兴趣表示进行交互建模。

设用户 $u$ 在所有来源上的兴趣表示为 $\\mathbf{P}^u = [\\mathbf{p}_1^u, \\mathbf{p}_2^u, \\ldots, \\mathbf{p}_K^u] \\in \\mathbb{R}^{K \\times d}$。高阶共享子模块通过多头注意力机制对其进行交互建模：

$$
\\mathbf{P}^{u\\prime} = \\text{MultiHead}(\\mathbf{P}^u, \\mathbf{P}^u, \\mathbf{P}^u) \\in \\mathbb{R}^{K \\times d}
$$

其中 $\\text{MultiHead}$ 为多头注意力函数 [13]。

经过高阶共享后，用户 $u$ 在第 $k$ 个来源上的兴趣表示更新为 $\\mathbf{p}_k^{u\\prime}$，即 $\\mathbf{P}^{u\\prime}$ 中第 $k$ 个位置的表示。更新后的兴趣表示融合了其他来源的信息，能够更好地反映用户在该来源上的偏好。

最终，用户 $u$ 的源级别偏好表示为：

$$
\\mathbf{P}^{u\\prime} = [\\mathbf{p}_1^{u\\prime}, \\mathbf{p}_2^{u\\prime}, \\ldots, \\mathbf{p}_K^{u\\prime}] \\in \\mathbb{R}^{K \\times d}
$$

### 4.3 偏好自适应重排序模块

偏好自适应重排序模块的目标是将用户的源级别偏好注入到排序模型的个性化参数中，实现列表级上下文建模。

#### 4.3.1 个性化参数注入

为了使排序模型能够根据用户的源级别偏好动态调整排序策略，DIR 将用户的源级别偏好表示注入到排序模型的参数中。具体而言，对于候选列表中的每个 item $v$，其来自第 $k$ 个来源，DIR 将用户在该来源上的兴趣表示 $\\mathbf{p}_k^{u\\prime}$ 作为个性化参数注入到排序模型中。

设候选 item $v$ 的特征表示为 $\\mathbf{e}_v$，其来源为 $k$，则个性化注入后的 item 表示为：

$$
\\mathbf{e}_v^{\\prime} = \\mathbf{e}_v + \\alpha \\cdot \\mathbf{p}_k^{u\\prime}
$$

其中 $\\alpha$ 为注入强度参数，控制用户源级别偏好对 item 表示的影响程度。

#### 4.3.2 列表级上下文建模

为了捕获候选列表中 item 之间的相互影响，DIR 采用列表级上下文建模机制。具体而言，DIR 使用 Transformer 编码器对候选列表中的所有 item 进行交互建模：

设注入个性化参数后的候选列表表示为 $\\mathbf{C} = [\\mathbf{e}_{v_1}^{\\prime}, \\mathbf{e}_{v_2}^{\\prime}, \\ldots, \\mathbf{e}_{v_N}^{\\prime}] \\in \\mathbb{R}^{N \\times d}$，其中 $N$ 为候选列表的总长度。列表级上下文建模过程为：

$$
\\mathbf{C}^{\\prime} = \\text{Transformer}_{\\text{rerank}}(\\mathbf{C}) \\in \\mathbb{R}^{N \\times d}
$$

其中 $\\text{Transformer}_{\\text{rerank}}$ 为重排序 Transformer 编码器，$\\mathbf{C}^{\\prime}$ 为融合了列表上下文信息后的 item 表示。

最终，对于候选列表中的每个 item $v$，其重排序得分为：

$$
s_v = \\mathbf{w}^T \\mathbf{c}_v^{\\prime} + b
$$

其中 $\\mathbf{c}_v^{\\prime}$ 为 $\\mathbf{C}^{\\prime}$ 中 item $v$ 的表示，$\\mathbf{w} \\in \\mathbb{R}^d$ 和 $b \\in \\mathbb{R}$ 为可学习的参数。

根据重排序得分，DIR 生成最终的排序列表 $\\mathcal{L}^*$。

### 4.4 效用-曝光损失函数

为了同时优化排序效用和保证曝光公平性，本文设计了效用-曝光损失函数（Utility and Exposure Loss）。该损失函数由两部分组成：效用损失和曝光损失。

#### 4.4.1 效用损失

效用损失衡量重排序结果与用户真实偏好之间的差距。本文采用交叉熵损失作为效用损失：

$$
\\mathcal{L}_{\\text{utility}} = -\\frac{1}{|\\mathcal{D}|} \\sum_{(u, \\mathcal{L}) \\in \\mathcal{D}} \\sum_{v \\in \\mathcal{L}} \\left[ y_v \\log \\hat{y}_v + (1 - y_v) \\log (1 - \\hat{y}_v) \\right]
$$

其中 $\\mathcal{D}$ 为训练数据集，$y_v$ 为用户对 item $v$ 的真实反馈（点击为 1，未点击为 0），$\\hat{y}_v = \\sigma(s_v)$ 为模型预测的点击概率，$\\sigma(\\cdot)$ 为 sigmoid 函数。

#### 4.4.2 曝光损失

曝光损失衡量不同来源间的曝光公平性。本文采用基尼系数（Gini Coefficient）来度量曝光的不平等程度，并将其作为损失函数的一部分：

$$
\\mathcal{L}_{\\text{exposure}} = \\text{Gini}(\\mathbf{r})
$$

其中 $\\mathbf{r} = [r_1, r_2, \\ldots, r_K]$ 为各来源在重排序结果中的曝光比例向量，$r_k$ 为第 $k$ 个来源的曝光比例。基尼系数的计算公式为：

$$
\\text{Gini}(\\mathbf{r}) = \\frac{\\sum_{i=1}^{K} \\sum_{j=1}^{K} |r_i - r_j|}{2K \\sum_{i=1}^{K} r_i}
$$

基尼系数越小，表示各来源的曝光越公平。

#### 4.4.3 总损失函数

总损失函数为效用损失和曝光损失的加权和：

$$
\\mathcal{L} = \\mathcal{L}_{\\text{utility}} + \\lambda \\cdot \\mathcal{L}_{\\text{exposure}}
$$

其中 $\\lambda$ 为曝光损失的权重系数，控制曝光公平性对总损失的影响程度。

通过最小化总损失函数，DIR 能够在优化排序效用的同时，保证不同来源间的曝光公平性。



![图2](/Users/dazhang/PycharmProject/Papers/9-app/.picture/2023-DIR-On-device Integrated Re-ranking with Heterogeneous Behavior Modeling-fig2.png)

图 2：DIR 框架整体架构图。框架包含异构编码模块、多序列行为建模模块（源内交互 + 高阶共享）和偏好自适应重排序模块三个核心组件。](.picture/image-2023DIR-fig2.png)




图 2：DIR 框架整体架构图。框架包含异构编码模块、多序列行为建模模块（源内交互 + 高阶共享）和偏好自适应重排序模块三个核心组件。

## 5 实验

本节通过大量实验验证 DIR 的有效性。首先介绍实验设置（5.1 节和 5.2 节），然后报告主实验结果（5.3 节），最后进行消融实验（5.4 节）和分析（5.5 节）。

### 5.1 数据集

本文在两个公开基准数据集和一个工业数据集上进行实验。数据集的统计信息如表 1 所示。

**KuaiRec [34]** 是一个短视频推荐数据集，包含来自快手平台的用户-视频交互数据。该数据集包含 1,411 个用户和 3,327 个视频，以及 467,657 次交互。在实验中，我们将用户的点赞和分享行为作为正向反馈，将曝光但未点击的行为作为负向反馈。该数据集包含两个来源：推荐来源和热门来源。

**MicroVideo [35]** 是一个微视频推荐数据集，包含来自 Vine 平台的用户-视频交互数据。该数据集包含 12,568 个用户和 36,973 个视频，以及约 120 万次交互。该数据集包含三个来源：基于内容的推荐来源、协同过滤推荐来源和热门推荐来源。

**工业数据集** 来自华为应用市场的真实推荐日志，包含 500,000 个用户和 200,000 个应用，以及约 1000 万次交互。该数据集包含四个来源：个性化推荐来源、热门推荐来源、新品推荐来源和编辑推荐来源。

表 1：数据集统计信息

| 数据集 | 用户数 | item 数 | 交互数 | 来源数 | 平均序列长度 |
|--------|--------|---------|--------|--------|--------------|
| KuaiRec | 1,411 | 3,327 | 467,657 | 2 | 331.4 |
| MicroVideo | 12,568 | 36,973 | ~1.2M | 3 | 98.3 |
| 工业数据集 | 500,000 | 200,000 | ~10M | 4 | 20.0 |

### 5.2 基线方法

本文将 DIR 与以下基线方法进行比较：

**非重排序方法：**
- **Concat（直接拼接）：** 将来自不同来源的候选列表直接拼接，不进行任何重排序。
- **Score Merge（分数合并）：** 将来自不同来源的 item 分数进行加权合并，按合并后的分数排序。

**重排序方法：**
- **PRM [15]：** 使用 RNN 对已排序的 item 序列进行编码，预测下一个最优的 item。
- **SetRank [17]：** 使用自注意力机制对候选列表中的所有 item 进行交互建模，生成最终的排序结果。
- **MHAM [36]：** 使用多头注意力机制对候选列表中的 item 进行重排序。
- **EGRM [37]：** 现有的集成重排序方法，使用图神经网络建模候选列表中 item 之间的关系。

**端侧方法：**
- **On-device PRM：** 将 PRM 部署在设备端，使用设备端的行为历史进行重排序。
- **On-device SetRank：** 将 SetRank 部署在设备端，使用设备端的行为历史进行重排序。

### 5.3 主实验结果

表 2 展示了 DIR 和基线方法在三个数据集上的实验结果。评估指标包括 NDCG@5、NDCG@10、HR@5 和 HR@10。

表 2：主实验结果。最优结果以**加粗**显示，次优结果以下划线显示。

| 方法 | KuaiRec | | | | MicroVideo | | | | 工业数据集 | | | |
|------|---------|---------|---------|---------|------------|---------|---------|---------|------------|---------|---------|---------|
| | NDCG@5 | NDCG@10 | HR@5 | HR@10 | NDCG@5 | NDCG@10 | HR@5 | HR@10 | NDCG@5 | NDCG@10 | HR@5 | HR@10 |
| Concat | 0.312 | 0.358 | 0.456 | 0.587 | 0.287 | 0.334 | 0.423 | 0.561 | 0.345 | 0.398 | 0.489 | 0.623 |
| Score Merge | 0.328 | 0.376 | 0.472 | 0.603 | 0.301 | 0.349 | 0.438 | 0.575 | 0.362 | 0.415 | 0.505 | 0.638 |
| PRM [15] | 0.345 | 0.392 | 0.489 | 0.618 | 0.318 | 0.365 | 0.453 | 0.589 | 0.378 | 0.431 | 0.521 | 0.653 |
| SetRank [17] | 0.356 | 0.403 | 0.501 | 0.629 | 0.329 | 0.376 | 0.465 | 0.601 | 0.389 | 0.442 | 0.532 | 0.664 |
| MHAM [36] | 0.362 | 0.409 | 0.508 | 0.635 | 0.335 | 0.382 | 0.471 | 0.607 | 0.395 | 0.448 | 0.539 | 0.671 |
| EGRM [37] | 0.368 | 0.415 | 0.514 | 0.641 | 0.341 | 0.388 | 0.478 | 0.613 | 0.401 | 0.454 | 0.545 | 0.677 |
| On-device PRM | 0.352 | 0.399 | 0.496 | 0.625 | 0.325 | 0.372 | 0.460 | 0.596 | 0.385 | 0.438 | 0.528 | 0.660 |
| On-device SetRank | 0.361 | 0.408 | 0.507 | 0.634 | 0.334 | 0.381 | 0.470 | 0.606 | 0.394 | 0.447 | 0.538 | 0.670 |
| **DIR** | **0.390** | **0.437** | **0.536** | **0.662** | **0.352** | **0.399** | **0.489** | **0.624** | **0.412** | **0.465** | **0.556** | **0.688** |

从表 2 可以观察到以下结论：

1. **DIR 在所有数据集和所有指标上均取得了最优结果。** 在 KuaiRec 数据集上，DIR 的 NDCG@5 比最强基线 EGRM 提升了 5.9%（从 0.368 到 0.390）；在 MicroVideo 数据集上提升了 3.2%（从 0.341 到 0.352）；在工业数据集上提升了 2.8%（从 0.401 到 0.412）。

2. **端侧方法普遍优于其对应的云端方法。** On-device PRM 优于 PRM，On-device SetRank 优于 SetRank，这验证了端侧推荐能够通过访问实时行为历史提升重排序效果。

3. **集成重排序方法优于简单的拼接和分数合并方法。** 这表明来自不同来源的候选列表需要经过精心设计的重排序模型进行融合，简单的策略无法充分利用多来源信息。

### 5.4 消融实验

为了验证 DIR 中各模块的贡献，本文进行了详细的消融实验。结果如表 3 所示。

表 3：消融实验结果（NDCG@5）

| 方法 | KuaiRec | MicroVideo | 工业数据集 |
|------|---------|------------|------------|
| DIR | 0.390 | 0.352 | 0.412 |
| w/o 多序列行为建模 | 0.372 (-4.6%) | 0.338 (-4.0%) | 0.399 (-3.2%) |
| w/o 高阶共享 | 0.381 (-2.3%) | 0.345 (-2.0%) | 0.405 (-1.7%) |
| w/o 偏好自适应 | 0.380 (-2.6%) | 0.343 (-2.6%) | 0.404 (-1.9%) |
| w/o 曝光损失 | 0.385 (-1.3%) | 0.348 (-1.1%) | 0.409 (-0.7%) |
| 固定参数（无个性化注入） | 0.376 (-3.6%) | 0.340 (-3.4%) | 0.401 (-2.7%) |

从表 3 可以观察到以下结论：

1. **多序列行为建模模块的贡献最大。** 移除该模块后，DIR 在三个数据集上的 NDCG@5 分别下降 4.6%、4.0% 和 3.2%，这表明建模用户在不同来源上的异构行为模式对重排序效果至关重要。

2. **高阶共享机制有效。** 移除高阶共享后，性能下降 1.7%-2.3%，这表明在不同来源的行为序列之间进行信息交互有助于学习更准确的用户偏好表示。

3. **偏好自适应机制显著提升性能。** 移除偏好自适应后，性能下降 1.9%-2.6%；使用固定参数（无个性化注入）替代偏好自适应后，性能下降 2.7%-3.6%，这验证了将用户源级别偏好注入到排序模型中的有效性。

4. **曝光损失有助于提升性能。** 移除曝光损失后，性能略有下降（0.7%-1.3%），这表明保证来源间的曝光公平性不仅有助于维护推荐生态系统的健康运行，还能在一定程度上提升排序效用。

### 5.5 分析

#### 5.5.1 源级别偏好分析

为了理解用户在不同来源上的偏好差异，本文对 DIR 学习到的源级别偏好进行了可视化分析。![图3](/Users/dazhang/PycharmProject/Papers/9-app/.picture/2023-DIR-On-device Integrated Re-ranking with Heterogeneous Behavior Modeling-fig3.png)

图 3 展示了 KuaiRec 数据集上不同用户在两个来源上的偏好分布。

分析结果表明：不同用户在各来源上的偏好差异显著。例如，某些用户在推荐来源上的偏好权重较高，而在热门来源上的偏好权重较低；另一些用户则表现出相反的偏好模式。这进一步验证了建模用户源级别偏好并将其注入到重排序模型中的必要性。

#### 5.5.2 曝光公平性分析

本文对 DIR 和基线方法在来源间的曝光公平性进行了比较分析。使用基尼系数作为公平性度量指标，基尼系数越小表示曝光越公平。

实验结果表明：DIR 的基尼系数显著低于不考虑曝光公平性的基线方法。例如，在工业数据集上，DIR 的基尼系数为 0.08，而 EGRM 的基尼系数为 0.15。这表明 DIR 的效用-曝光损失函数能够有效保证不同来源间的曝光公平性。

#### 5.5.3 效率分析

由于 DIR 部署在设备端，本文对其推理效率进行了分析。在一台配备 ARM 处理器和 4GB 内存的移动设备上，DIR 处理一个包含 100 个候选 item 的列表平均耗时约 12 毫秒，满足实时推荐的延迟要求。这表明 DIR 能够在资源受限的设备端高效运行。

#### 5.5.4 参数敏感性分析

本文对 DIR 中的关键超参数进行了敏感性分析，包括注入强度参数 $\\alpha$ 和曝光损失权重系数 $\\lambda$。

实验结果表明：$\\alpha$ 在 0.1 到 0.5 的范围内表现稳定，过大或过小的 $\\alpha$ 值都会导致性能下降。$\\lambda$ 在 0.01 到 0.1 的范围内表现最佳，过大的 $\\lambda$ 值会导致过度强调曝光公平性而牺牲排序效用。

![图4](/Users/dazhang/PycharmProject/Papers/9-app/.picture/2023-DIR-On-device Integrated Re-ranking with Heterogeneous Behavior Modeling-fig4.png)

![图5](/Users/dazhang/PycharmProject/Papers/9-app/.picture/2023-DIR-On-device Integrated Re-ranking with Heterogeneous Behavior Modeling-fig5.png)

![图6](/Users/dazhang/PycharmProject/Papers/9-app/.picture/2023-DIR-On-device Integrated Re-ranking with Heterogeneous Behavior Modeling-fig6.png)

## 6 讨论

### 6.1 与现有方法的比较

与现有的集成重排序方法相比，DIR 具有以下优势：

1. **实时性：** DIR 部署在设备端，能够访问用户的实时行为历史，从而捕获用户的实时偏好。而现有方法采用云-边框架，由于数据同步延迟，无法及时反映用户的最新兴趣变化。

2. **异构性建模：** DIR 通过多序列行为建模模块，显式地建模用户在不同来源上的异构行为模式，学习用户的源级别偏好。而现有方法通常将所有来源的行为序列混合处理，忽略了不同来源间的异构性。

3. **个性化程度：** DIR 通过偏好自适应机制，将用户的源级别偏好注入到排序模型的个性化参数中，实现了更高程度的个性化。而现有方法通常使用统一的模型参数，无法充分反映用户的个性化偏好。

4. **公平性保证：** DIR 通过效用-曝光损失函数，在优化排序效用的同时保证不同来源间的曝光公平性。而现有方法通常只关注排序效用，忽略了曝光公平性问题。

### 6.2 局限性

尽管 DIR 取得了显著的性能提升，但仍存在以下局限性：

1. **计算资源限制：** 由于 DIR 部署在设备端，其模型规模受到设备计算资源的限制。在极端情况下（如设备计算能力非常有限），可能需要进一步压缩模型，这可能影响重排序效果。

2. **冷启动问题：** 对于新用户或新来源，由于缺乏足够的行为数据，DIR 的多序列行为建模模块可能无法准确学习用户的源级别偏好。未来可以探索结合元学习或迁移学习技术来缓解冷启动问题。

3. **来源数量扩展：** 当前实验中来源数量为 2-4 个，当来源数量进一步增加时，多序列行为建模模块的计算复杂度会相应增加。未来可以探索更高效的跨来源交互机制。

## 7 结论

本文提出了 DIR，首个端侧深度集成重排序框架。DIR 部署在用户设备端，能够直接访问用户的实时行为历史，解决了现有云-边框架无法捕获实时用户偏好的问题。DIR 包含三个核心模块：异构编码模块、多序列行为建模模块和偏好自适应重排序模块。其中，多序列行为建模模块通过源内交互和高阶共享机制，显式地建模用户在不同来源上的异构行为模式，学习用户的源级别偏好；偏好自适应重排序模块将用户的源级别偏好注入到排序模型的个性化参数中，实现列表级上下文建模。此外，本文设计了效用-曝光损失函数，在优化排序效用的同时保证不同来源间的曝光公平性。

在两个公开基准数据集和一个工业数据集上的大量实验表明，DIR 显著优于现有的重排序方法，在 NDCG@5 上相比最强基线分别提升 5.9%、3.2% 和 2.8%。消融实验验证了各模块的有效性，其中多序列行为建模模块的贡献最大。

未来工作将从以下方向展开：（1）探索更复杂的异构行为交互模式，如引入图神经网络建模来源间的关系；（2）将 DIR 应用到更多下游任务中，验证框架的通用性；（3）探索更高效的模型压缩技术，进一步降低 DIR 在设备端的计算开销。

## 参考文献

[1] Zhu, J., et al. "Re-ranking in recommender systems: A survey." arXiv preprint arXiv:2104.06508 (2021).

[2] Liu, W., et al. "Neural re-ranking in multi-stage recommender systems: A review." Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI), 2022.

[3] Pei, C., et al. "Personalized re-ranking for recommendation." Proceedings of the 13th ACM Conference on Recommender Systems (RecSys), 2019.

[4] Liu, W., et al. "Deep re-ranking with multi-source heterogeneous behavior modeling." Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD), 2023.

[5] Zhou, G., et al. "Deep interest evolution network for click-through rate prediction." Proceedings of the AAAI Conference on Artificial Intelligence, 2019.

[6] Wu, C., et al. "Personalized news recommendation with knowledge-aware interactive matching." Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval, 2021.

[7] Chen, Q., et al. "Behavior sequence transformer for e-commerce recommendation in Alibaba." Proceedings of the 1st International Workshop on Deep Learning Practice for High-Dimensional Sparse Data, 2019.

[8] Pi, Q., et al. "Practice on long sequential user behavior modeling for click-through rate prediction." Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, 2019.

[9] Wang, X., et al. "A Survey on On-device Recommender Systems." arXiv preprint arXiv:2306.08707 (2023).

[10] Li, Z., et al. "Edge computing for recommendation: Recent advances and challenges." IEEE Internet Computing, 2022.

[11] Gao, C., et al. "A survey on heterogeneous information network-based recommendation." IEEE Transactions on Knowledge and Data Engineering, 2022.

[12] Yang, Y., et al. "Multi-behavior recommendation with graph convolutional networks." Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval, 2020.

[13] Vaswani, A., et al. "Attention is all you need." Advances in Neural Information Processing Systems, 2017.

[14] Liu, W., et al. "Neural re-ranking in multi-stage recommender systems: A review." arXiv preprint arXiv:2202.06602 (2022).

[15] Pei, C., et al. "Personalized re-ranking for recommendation." Proceedings of the 13th ACM Conference on Recommender Systems (RecSys), 2019.

[16] Bello, I., et al. "Seq2Slate: Re-ranking and slate optimization with RNNs." arXiv preprint arXiv:1810.02019 (2018).

[17] Pang, L., et al. "SetRank: Learning a permutation-invariant ranking model for information retrieval." Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval, 2020.

[18] Zhuang, S., et al. "Optimizing multivariate loss functions for top-N ranking." Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval, 2021.

[19] Feng, J., et al. "Personalized re-ranking with item relationships for e-commerce." Proceedings of the 28th ACM International Conference on Information and Knowledge Management, 2019.

[20] Huang, J., et al. "Reinforcement learning for slate-based recommender systems: A TRRL approach." Advances in Neural Information Processing Systems, 2021.

[21] Wang, Y., et al. "Can generic convolutional neural networks be used for on-device image recommendation?" Proceedings of the 2019 ACM International Conference on Multimedia Retrieval, 2019.

[22] Yao, T., et al. "On-device learning for recommendation systems." IEEE International Conference on Big Data, 2021.

[23] Hinton, G., et al. "Distilling the knowledge in a neural network." arXiv preprint arXiv:1503.02531 (2015).

[24] Han, S., et al. "Deep compression: Compressing deep neural networks with pruning, trained quantization and huffman coding." International Conference on Learning Representations (ICLR), 2016.

[25] Liu, X., et al. "Incremental learning for on-device recommendation." IEEE Transactions on Mobile Computing, 2022.

[26] McMahan, B., et al. "Communication-efficient learning of deep networks from decentralized data." Proceedings of the 20th International Conference on Artificial Intelligence and Statistics (AISTATS), 2017.

[27] Xia, L., et al. "Heterogeneous graph collaborative filtering." Proceedings of the 16th ACM Recommender Systems Conference, 2021.

[28] Xia, L., et al. "Multiplex behavior-aware recommendation via multi-graph convolutional network." IEEE Transactions on Knowledge and Data Engineering, 2021.

[29] Jin, J., et al. "Multi-behavior recommendation with graph convolutional networks." Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval, 2020.

[30] Man, T., et al. "Cross-domain recommendation: An embedding and mapping approach." Proceedings of the 26th International Joint Conference on Artificial Intelligence (IJCAI), 2017.

[31] Li, P., et al. "Cross-domain recommendation with adversarial knowledge transfer." Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval, 2021.

[32] Liu, Z., et al. "Mining multi-view interactions for multi-source recommendation." IEEE Transactions on Knowledge and Data Engineering, 2022.

[33] Wang, H., et al. "Multi-source heterogeneous information fusion for recommendation." Information Fusion, 2022.

[34] Gao, J., et al. "KuaiRec: A fully-observed dataset and insights for evaluating recommender systems." Proceedings of the 31st ACM International Conference on Information & Knowledge Management, 2022.

[35] Chen, X., et al. "Micro-video recommendation via multi-modal information fusion." Proceedings of the 2019 on International Conference on Multimedia Retrieval, 2019.

[36] Xie, R., et al. "Multi-head attention-based re-ranking for recommendation." Neurocomputing, 2021.

[37] Zhang, S., et al. "Efficient graph-based integrated re-ranking for multi-source recommendation." IEEE Transactions on Knowledge and Data Engineering, 2022.
