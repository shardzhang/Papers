# MobileLLM-Optimizing Sub-billion Parameter Language Models for On-Device Use Cases

https://github.com/facebookresearch/MobileLLM





任务：问答和阅读理解任务







## FQA

#### Q：这句话合理么? 帮我计算推理下

我们的实验在 **32块A100 GPU**上进行，每块GPU的批大小为32。我们使用 120k次 迭代在 0.25T token上进行了探索性实验。

翻译准确。数学验证：
- 全局 batch = 32 GPU × 32 sample/GPU = 1024 samples/iter
- 每次迭代 tokens = 0.25T / 120k ≈ 2,083,333 tokens/iter
- 序列长度 = 2,083,333 / 1024 ≈ 2034 tokens/sample（标准序列长度 2048）

结论：句子合理，数字自洽。 翻译与原文一致，数学推导成立（序列长度约 2048，与常见 LLM 训练设置吻合）。
