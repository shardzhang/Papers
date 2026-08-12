# Sequence to Sequence Learning with Neural Networks

> **Ilya Sutskever**, Google | Oriol Vinyals, Google | Quoc V. Le, Google





## FQA

#### Q：SMT

SMT 是 Statistical Machine Translation（统计机器翻译）的缩写。

在神经机器翻译（NMT）出现之前，SMT 是主流的机器翻译方法，主要包括：

- 基于短语的 SMT（Phrase-based SMT）：将句子切分为短语进行翻译
- 基于规则的 SMT：使用语言学规则
- 代表系统：Google Translate（早期）、Moses 等

Seq2Seq 论文中提到"LSTM 的 BLEU 分数达到 34.8，超过了基于短语的 SMT 系统（33.3）"，这标志着神经翻译首次超越统计翻译。



#### Q：Seq2Seq和Bahdanau这两个工作为什么没有直接比较? 两者之间到底哪个更好?

两篇论文都是 2014 年机器翻译领域的里程碑工作，关系如下：

Seq2Seq (Sutskever et al., 2014) 是基础架构：

- 用 Encoder LSTM 将整个输入句子压缩成固定长度向量
- 用 Decoder LSTM 从该向量解码输出句子
- 问题：长句子信息丢失（瓶颈问题）

Bahdanau et al. (2014) 是对 Seq2Seq 的改进：

- 引入注意力机制（Attention），让 Decoder 在生成每个词时能动态关注输入句子的不同部分
- 解决了 Seq2Seq 把整个句子压成固定向量的信息瓶颈问题
- 在长句子翻译上效果显著优于原始 Seq2Seq

从翻译中可以看到：

- Seq2Seq 论文明确引用了 Bahdanau 的工作 2
- 原文提到 Bahdanau 使用注意力机制"来克服 Cho 等人在长句上遇到的性能不佳问题"
- Seq2Seq 的解决方案是"反转源句子词序"（引入短期依赖），而 Bahdanau 的方案是注意力机制

总结： Seq2Seq 是先驱，Bahdanau Attention 是对其关键缺陷的改进，两者共同奠定了现代 NMT 的基础。



#### Q：Cho vs Seq2Seq vs Bahdanau

Cho et al. (2014) "Learning phrase representations using RNN encoder-decoder for statistical machine translation" 是：
- Seq2Seq 的前身：提出了 RNN Encoder-Decoder 框架（用编码器将句子压缩成向量，再用解码器解码）
- Bahdanau 的基础：Bahdanau 的注意力机制正是在这个 RNN Encoder-Decoder 框架上添加的
- 关键贡献：学习短语表示，用于统计机器翻译（SMT）

三篇论文的关系链：

Cho et al. (2014) [5] — 提出 RNN Encoder-Decoder 框架
  ↓
Sutskever et al. (2014) — 深化为 Seq2Seq，用 LSTM，证明可超越 SMT
  ↓
Bahdanau et al. (2014) [2] — 在 Cho 的框架上加注意力机制，解决长句问题

Cho 是第一作者，Bengio 是共同作者（Cho 的导师）。所以 Bahdanau 引用的 5 其实是自己团队的早期工作，然后在此基础上加了注意力机制。



关于团队关系：不完全是同一个团队，但有密切联系：

| 论文	|作者|	所属机构|
| ---- | ---- | ---- |
|Cho et al. [5]	|Cho, Merrienboer, Gulcehre, Bougares, Schwenk, Bengio	|蒙特利尔大学|
|Bahdanau et al. [2]	|Bahdanau, Cho, Bengio	|蒙特利尔大学|
|Sutskever et al. Seq2Seq	|Sutskever, Vinyals, Le	|Google|

关键点：

- Cho 和 Bahdanau 是 Bengio 的学生/博后。Bengio 是深度学习三巨头之一（Hinton、LeCun、Bengio）
- Sutskever 是 Hinton 的学生，后加入 Google（OpenAI 联合创始人）
- 蒙特利尔团队（Bengio 组）和 Google 团队是竞争对手，几乎同时做出类似工作

所以是两个独立团队的竞争性同时发现（simultaneous discovery），不是同一团队。





