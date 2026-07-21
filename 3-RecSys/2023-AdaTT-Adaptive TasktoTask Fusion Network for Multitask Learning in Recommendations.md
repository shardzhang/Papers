# AdaTT Adaptive TasktoTask Fusion Network for Multitask Learning in Recommendations

## 论文信息
- **标题**: AdaTT Adaptive TasktoTask Fusion Network for Multitask Learning in Recommendations
- **文件路径**: /home/runner/work/Papers/Papers/RecSys/AdaTT Adaptive TasktoTask Fusion Network for Multitask Learning in Recommendations.pdf

## 主要内容

### 摘要

AdaTT是一种用于推荐系统多任务学习的自适应任务到任务融合网络，通过设计残差结构和专家融合模块来提升多任务学习的效果。

### 主要贡献

- 提出了自适应任务到任务融合网络架构
- 设计了包含单独模块的残差结构来融合不同专家
- 在多任务推荐场景下验证了方法的有效性

### 方法/技术

- **残差设计**: AdaTT的残差设计包含单独的模块来融合不同的专家
- **专家融合**: 通过NativeExpertLF模块验证残差设计的重要性
- **多级融合**: 在不同级别进行任务关系建模

### 实验结果

我们消除了 𝑁𝑎𝑡𝑖𝑣𝑒𝐸𝑥𝑝𝑒𝑟𝑡𝐿𝐹 模块来验证 AdaTT 残差设计的重要性，该设计包含单独的模块来融合不同的专家。

首先，在较低的融合级别（级别 0），我们的模型能够**辨别任务之间的关系**。

消费任务组和参与任务组之间有明显的区别。此外，两个消费任务之间存在不对称的共享模式：分类消费任务主要使用专家2，分类回归任务大致平等地使用专家1和2。

### 结论

AdaTT通过自适应的任务到任务融合机制，能够有效捕获不同任务之间的关系，提升多任务学习在推荐系统中的表现。

### 个人笔记

- 多任务学习在推荐系统中的应用越来越重要
- 任务之间的关系建模是关键技术难点
- 残差设计和专家融合是有效的技术方案


