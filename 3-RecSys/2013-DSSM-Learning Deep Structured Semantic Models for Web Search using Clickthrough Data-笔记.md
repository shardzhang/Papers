# Learning Deep Structured Semantic Models for Web Search using Clickthrough Data





重要意义：双塔模型开山之作。

业务场景：搜索场景

业务指标：搜索相关性

离线指标：NDCG@1、10

数据集：商业搜索引擎一年查询日志文件中采样1亿条<query，doc>查询文档对

模型类型：LTR中listwise建模，大规模多分类问题。

创新点：

- 词哈希：降低原始词向量维度，解决计算效率限制
- 有监督学习：仅适用有点击样本（正样本），将问题转换为大规模多分类问题
- 最后有些示例展示了模型确实能够学习到单词的语义结构

个人收获：

- 文章附录部分对SoftmaxCrossEntropy loss求导过程做了介绍，很有价值。这里暂时跳过。

