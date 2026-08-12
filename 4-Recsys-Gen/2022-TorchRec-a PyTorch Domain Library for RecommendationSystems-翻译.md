# TorchRec：一个面向推荐系统的PyTorch领域库


本文介绍了 TorchRec：一个面向推荐系统的PyTorch领域库。核心内容：


关键发现：

---


德米特罗·伊夫琴科
> divchenko@fb.com
Meta AI
美国加利福尼亚州门洛帕克

刘星
xingl@fb.com
Meta AI
美国加利福尼亚州门洛帕克

丹尼斯·范德斯塔伊
dstaay@fb.com
Meta AI
美国加利福尼亚州门洛帕克

冯伟
willfeng@fb.com
Meta AI
美国加利福尼亚州门洛帕克

科林·泰勒
colin2328@fb.com
Meta AI
美国加利福尼亚州门洛帕克

拉胡尔·金迪
rahulkindi@fb.com
Meta AI
美国纽约州纽约

阿尼鲁德·苏达尚
anirudhs@fb.com
Meta AI
美国加利福尼亚州门洛帕克

沙欣·塞法蒂
shahinsefati@fb.com
Meta AI
美国纽约州纽约

## 摘要

推荐系统（RecSys）构成了当今生产部署AI的重要部分。
基于神经网络的推荐系统与其他领域的深度学习模型不同，
它使用高基数的类别型稀疏特征，
需要训练大型嵌入表。
在本次报告中，我们介绍TorchRec，
一个面向推荐系统的PyTorch领域库。
这个新库提供了通用的稀疏性和并行性原语，
使研究人员能够构建最先进的个性化模型并将其部署到生产中。
在本次报告中，我们涵盖了TorchRec库的构建模块，
包括建模原语（如嵌入袋和锯齿张量）、
由FBGEMM驱动的优化推荐系统内核、
一个灵活的分片器，支持多种策略用于分区嵌入表、
一个自动生成优化且高性能的分片计划的规划器、
对GPU推理的支持以及用于构建推荐系统模型的通用建模模块。
TorchRec库目前用于在Meta训练大规模推荐模型。
我们将介绍TorchRec如何帮助Meta的推荐系统平台
从CPU异步训练过渡到基于加速器的全同步训练。

## CCS概念

• 信息系统 $\to$ 推荐系统；开源软件；
• 计算方法论 $\to$ 大规模并行算法。

## 关键词

推荐系统，信息检索

---

允许为个人或课堂使用制作本作品的部分或全部数字或硬拷贝，无需付费，
前提是复制的副本不以营利或商业优势为目的，
并且副本在第一页包含此声明和完整引用。
必须尊重本作品中第三方组件的版权。
对于所有其他用途，请联系所有者/作者。
RecSys '22，2022年9月18–23日，美国华盛顿州西雅图
© 2022 版权归所有者/作者所有。
ACM ISBN 978-1-4503-9278-5/22/09。
https://doi.org/10.1145/3523227.3547387

## ACM引用格式

德米特罗·伊夫琴科、丹尼斯·范德斯塔伊、科林·泰勒、刘星、冯伟、
拉胡尔·金迪、阿尼鲁德·苏达尚和沙欣·塞法蒂。
2022。TorchRec：一个面向推荐系统的PyTorch领域库。
在第十六届ACM推荐系统会议（RecSys '22），
2022年9月18–23日，美国华盛顿州西雅图。
ACM，美国纽约州纽约，2页。
https://doi.org/10.1145/3523227.3547387

---

## 1 TORCHREC

到2020年中，PyTorch团队收到了大量反馈，
指出开源PyTorch生态系统中缺乏大规模生产质量的推荐系统包。
从Meta的技术栈出发，我们开始模块化和设计一个完全可扩展的代码库，
能够适应多样化的推荐用例。
我们的目标是提取Meta软件栈的部分内容，
以同时支持创造性探索和规模扩展。
我们希望这个包能够在RecSys社区中开启对话和协作，
Meta是第一个重要的贡献者。

### 1.1 Meta的性能扩展

PyTorch引入了对加速器（即GPU）的支持，
以及通过提供基于eager模式的Python API改善了开发者体验和生产力。
PyTorch生态系统仍然缺乏模型并行性
以完全推动全同步训练的极限。
对大规模生产级推荐系统的支持
推动了TorchRec库的开发。

TorchRec现在用于在Meta训练大型推荐模型，
模型参数超过3
$$
\times
$$
10¹²。
TorchRec帮助Meta从CPU异步训练过渡到基于加速器的全同步训练，
大幅增加了FLOPs和模型规模，
带来了显著的模型质量提升。
因为TorchRec基于PyTorch，
它提供了更好的建模灵活性，
机器学习工程师现在可以使用自由形式的Python代码
编写训练程序。

除了训练之外，TorchRec还支持推理模型准备和优化，
例如量化和剪枝，
因为这些操作可以应用于模型的分片版本。
TorchRec还

482  RecSys '22，2022年9月18–23日，美国华盛顿州西雅图

泰勒等人

支持GPU推理。
它提供低延迟的通信原语和优化的量化计算内核。
TorchRec运行在torch::deploy
（https://pytorch.org/docs/stable/deploy.html）上，
以确保低延迟和高吞吐量，
同时保留Python模型编写的灵活性。

在整个生产生命周期
（即训练、推理模型准备和推理服务）中使用Python的能力，
使得在Meta能够快速进行建模探索，
并简化从研究探索到生产工作流的流程。

### 1.2 TorchRec库构建模块

TorchRec包含可扩展的低级建模基础
以及丰富的即用型模块。
我们最初针对"双塔"（[1], [2]）架构，
该架构有独立的子模块来学习
候选项和查询或上下文的表示。
输入信号可以是浮点"稠密"特征
或高基数的类别型"稀疏"特征的混合，
需要训练大型嵌入表。
高效训练此类架构需要结合
数据并行性（复制计算的"稠密"部分）
和模型并行性（将大型嵌入表分区到多个节点上）。
TorchRec库包括：

• 建模原语，如嵌入袋和锯齿张量，
使得使用混合数据并行和模型并行
轻松编写大型、高性能的多设备/多节点模型。

• 由FBGEMM驱动的优化RecSys内核，
包括对稀疏和量化操作的支持、
融合稀疏优化器、
GPU内存缓存和其他高级性能优化。

• 一个分片器，可以使用多种不同策略
对嵌入表进行分区，
包括数据并行、按表、按行、按表按行和按列分片，
利用节点内和节点间的硬件拓扑。

• 一个规划器，可以自动生成优化的模型分片计划，
评估多种计划并根据模型和硬件特征
选择性能最佳的计划。

• 流水线，用于重叠数据加载设备传输（拷贝到GPU）、
设备间通信（输入分发）和计算（前向、反向），
以提升性能。

• GPU推理支持，
包括低延迟高性能的量化内核和通信。

• RecSys通用模块，
例如模型和公共数据集（Criteo和MovieLens）。

代码可在 https://github.com/pytorch/torchrec 获取。

## 致谢

我们要感谢为开发TorchRec库做出贡献的合作者：
Renfei Chen、Shabab Ayub、Joshua Deng、Zheng Yan、
Ning Wang、Ying Liu、Leon Gao、Liang Luo、
Hongbo Zhang、Xing Wang、Donny Greenberg、
Alex Morgan Bell、Bernard Nguyen以及更多人士。

483

## 参考文献

[1] Maxim Naumov, Dheevatsa Mudigere, Hao-Jun Michael Shi, Jianyu Huang,
Narayanan Sundaraman, Jongsoo Park, Xiaodong Wang, Udit Gupta, Carole-Jean
Wu, Alisson G. Azzolini, Dmytro Dzhulgakov, Andrey Mallevich, Ilia Cherniavskii,
Yinghai Lu, Raghuraman Krishnamoorthi, Ansha Yu, Volodymyr Kondratenko,
Stephanie Pereira, Xianjie Chen, Wenlin Chen, Vijay Rao, Bill Jia, Liang Xiong,
and Misha Smelyanskiy. 2020. DLRM: An advanced, open source deep learning
recommendation model.

[2] Xinyang Yi, Ji Yang, Lichan Hong, Derek Zhiyuan Cheng, Lukasz Heldt, Aditee Ajit
Kumthekar, Zhe Zhao, Li Wei, and Ed Chi (Eds.). 2019. Sampling-Bias-Corrected
Neural Modeling for Large Corpus Item Recommendations.

## 演讲者简介

德米特罗·伊夫琴科是Meta的软件工程师。
德米特罗在6年多前加入Meta。
他是PyTorch团队的一部分，专注于推荐系统。
在加入Meta之前，德米特罗曾在LinkedIn工作，
在那里他编写了用于搜索人员、职位和其他站点内容的个性化搜索引擎。

丹尼斯·范德斯塔伊是Meta的软件工程师。
丹尼斯在2年前加入Meta。
他是PyTorch团队的一部分，专注于推荐系统。
在加入Meta之前，丹尼斯在Branch International从事机器学习工程工作，
这是一家由安德森·霍洛维茨基金支持的初创公司。
