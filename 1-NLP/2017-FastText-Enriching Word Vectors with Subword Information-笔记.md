# Enriching Word Vectors with Subword Information

> Piotr Bojanowski, Edouard Grave, Armand Joulin, Tomas Mikolov | Facebook AI Research



词相似度任务





## FQA



#### Q：受分布式假设 [18] 启发，词表示被训练来预测在其上下文中出现的词。

这句话描述的是 **Word2Vec 类模型**（如 CBOW 和 Skip-gram）的核心训练思想。

分布式假设 (Distributional Hypothesis)

> **"一个词的意义由它周围的词决定"** —— Harris, 1954

即：**语义相似的词倾向于出现在相似的上下文中**。例如：

```
"我 喜欢 吃 [苹果]"
"我 喜欢 吃 [香蕉]"
```

"苹果"和"香蕉"出现在相似上下文中 → 它们语义相近。



训练方式：预测上下文中的词

模型学习一个词向量 $v_w$，使得给定中心词时能预测周围的词：



**Skip-gram（预测上下文）：**

$$P(w_{context} | w_{center}) = \frac{\exp(v'_{w_{context}} \cdot v_{w_{center}})}{\sum \exp(v'_w \cdot v_{w_{center}})}$$



**CBOW（用上下文预测中心词）：**

$$P(w_{center} | w_{context}) = \frac{\exp(v'_{w_{center}} \cdot \bar{v}_{context})}{\sum \exp(v'_w \cdot \bar{v}_{context})}$$



通俗理解

| 概念               | 含义                            |
| ------------------ | ------------------------------- |
| 分布式假设         | 词的意义由其上下文决定          |
| 词表示被训练来预测 | 优化目标是让模型能准确预测      |
| 上下文中出现的词   | 周围窗口内的词（如前后各5个词） |

**本质**：通过"猜上下文"这个简单任务，迫使模型**把语义相似的词映射到相近的向量位置**，从而自动学出有意义的词表示。
