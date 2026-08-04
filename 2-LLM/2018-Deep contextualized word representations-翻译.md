# Deep contextualized word representations（深度上下文词表示）

> Matthew E. Peters†, Mark Neumann†, Mohit Iyyer†, Matt Gardner†, Christopher Clark∗, Kenton Lee∗, Luke Zettlemoyer†∗
> †Allen Institute for Artificial Intelligence
> ∗Paul G. Allen School of Computer Science & Engineering, University of Washington

我们引入了一种新型的深度上下文词表示，它同时建模了（1）词使用的复杂特征（例如，语法和语义），以及（2）这些使用在不同语言上下文中的变化方式（即，对多义性建模）。我们的词向量是从深度双向语言模型（biLM）内部状态学习到的函数，该模型在大规模文本语料上预训练。我们展示了这些表示可以轻松添加到现有模型中，并在六个具有挑战性的自然语言处理问题上显著提升了最先进水平，包括问答、文本蕴含和情感分析。我们还进行了分析，表明暴露预训练网络的深层内部结构至关重要，这使得下游模型能够混合不同类型的半监督信号。

---

## 摘要

我们引入了一种新型的深度上下文化词表示，它同时建模了（1）词使用的复杂特征（例如，语法和语义），以及（2）这些使用在不同语言上下文中的变化方式（即，对多义性建模）。我们的词向量是从深度双向语言模型（biLM）内部状态学习到的函数，该模型在大规模文本语料上预训练。我们展示了这些表示可以轻松添加到现有模型中，并在六个具有挑战性的自然语言处理问题上显著提升了最先进水平，包括问答、文本蕴含和情感分析。我们还进行了分析，表明暴露预训练网络的深层内部结构至关重要，这使得下游模型能够混合不同类型的半监督信号。

## 1 引言

预训练词表示（Mikolov et al., 2013; Pennington et al., 2014）[1][2]是许多神经语言理解模型的关键组成部分。然而，学习高质量的表示可能是具有挑战性的。它们理想情况下应同时建模（1）词使用的复杂特征（例如，语法和语义），以及（2）这些使用在不同语言上下文中的变化方式（即，对多义性建模）。在本文中，我们引入了一种新型的深度上下文化词表示，它直接解决了这两个挑战，可以轻松集成到现有模型中，并且在多个具有挑战性的语言理解问题中显著提升了每个被考虑任务的最先进水平。

我们的表示与传统的词类型（type）嵌入不同，每个token（标记）被分配一个表示，该表示是整个输入句子的函数。我们使用从双向LSTM中导出的向量，该LSTM通过耦合的语言模型（LM）目标在大规模文本语料上进行训练。出于这个原因，我们称它们为ELMo（来自语言模型的嵌入）表示。与之前学习上下文化词向量的方法（Peters et al., 2017; McCann et al., 2017）[3][4]不同，ELMo表示是深度的，从某种意义上说，它们是biLM所有内部层的函数。更具体地说，我们为每个最终任务学习每个输入词之上堆叠向量的线性组合，这比仅使用顶层LSTM层显著提高了性能。

以这种方式组合内部状态允许非常丰富的词表示。通过内在评估，我们展示了较高层的LSTM状态捕捉词义的上下文相关方面（例如，它们可以直接用于在监督词义消歧任务上表现良好），而较低层的状态建模语法方面（例如，它们可以用于进行词性标注）。同时暴露所有这些信号是非常有益的，允许学习到的模型为每个最终任务选择最有用的半监督类型。

大量实验表明ELMo表示在实践中效果极好。我们首先展示它们可以轻松添加到六个多样且具有挑战性的语言理解问题的现有模型中，包括文本蕴含、问答和情感分析。仅添加ELMo表示就在每种情况下显著提升了最先进水平，包括高达20%的相对错误率降低。对于可以直接比较的任务，ELMo优于CoVe（McCann et al., 2017）[4]，后者使用神经机器翻译编码器计算上下文化表示。最后，对ELMo和CoVe的分析表明，深度表示优于仅从LSTM顶层导出的表示。我们的训练好的模型和代码是公开可用的，我们期望ELMo将为许多其他NLP问题提供类似的提升。

## 2 相关工作

由于预训练词向量（Turian et al., 2010; Mikolov et al., 2013; Pennington et al., 2014）[5][1][2]能够从大规模无标注文本中捕捉词的语法和语义信息，它们已成为大多数最先进NLP架构的标准组件，包括问答（Liu et al., 2017）[6]、文本蕴含（Chen et al., 2017）[7]和语义角色标注（He et al., 2017）[8]。然而，这些学习词向量的方法仅允许每个词有一个与上下文无关的表示。

先前提出的方法通过用子词信息丰富传统词向量（例如，Wieting et al., 2016; Bojanowski et al., 2017）[9][10]或为每个词义学习单独的向量（例如，Neelakantan et al., 2014）[11]来克服其一些缺点。我们的方法也通过使用字符卷积受益于子词单元，并且我们无缝地将多义信息融入下游任务中，而无需显式训练来预测预定义的义项类别。

其他近期工作也专注于学习上下文相关的表示。context2vec（Melamud et al., 2016）[12]使用双向长短期记忆网络（LSTM; Hochreiter and Schmidhuber, 1997）[13]来编码中心词周围的上下文。其他学习上下文嵌入的方法将中心词本身包含在表示中，并使用有监督的神经机器翻译（MT）系统（CoVe; McCann et al., 2017）[4]或无监督的语言模型（Peters et al., 2017）[3]的编码器进行计算。这两种方法都受益于大规模数据集，尽管MT方法受限于平行语料库的规模。在本文中，我们充分利用丰富的单语数据，在一个包含约3000万句子的语料库（Chelba et al., 2014）[14]上训练我们的biLM。我们还将这些方法泛化到深度上下文表示，并展示了这些表示在一系列多样的NLP任务上表现良好。

先前的工作还表明，深度双向RNN的不同层编码了不同类型的信息。例如，在深度LSTM的较低层引入多任务语法监督（如词性标签）可以提高高层任务（如依存句法分析（Hashimoto et al., 2017）[15]或CCG超级标注（Søgaard and Goldberg, 2016）[16]）的整体性能。在基于RNN的编码器-解码器机器翻译系统中，Belinkov等人（2017）[17]表明，在2层LSTM编码器的第一层学到的表示在预测词性标签方面比第二层更好。最后，用于编码词上下文的LSTM顶层（Melamud et al., 2016）[12]已被证明可以学习词义的表示。我们表明，类似的信号也由我们ELMo表示的改进语言模型目标所诱导，并且为下游任务学习混合这些不同类型的半监督的模型可能是非常有益的。

Dai和Le（2015）[18]以及Ramachandran等人（2017）[19]使用语言模型和序列自编码器预训练编码器-解码器对，然后通过任务特定的监督进行微调。相比之下，在使用无标注数据预训练biLM之后，我们固定权重并增加额外的任务特定模型容量，使我们在下游训练数据规模决定较小监督模型的情况下，能够利用大规模、丰富且通用的biLM表示。

## 3 ELMo：来自语言模型的嵌入

与最广泛使用的词嵌入（Pennington et al., 2014）[2]不同，ELMo词表示是整个输入句子的函数，如本节所述。它们是在带有字符卷积的双层biLM（第3.1节）之上计算的，作为内部网络状态的线性函数（第3.2节）。这种设置使我们能够进行半监督学习，其中biLM在大规模上进行预训练（第3.4节），并可以轻松地集成到广泛的现有神经NLP架构中（第3.3节）。

### 3.1 双向语言模型

给定一个包含N个token的序列 $(t_1, t_2, \ldots, t_N)$ ，前向语言模型通过对token $t_k$ 在给定历史 $(t_1, \ldots, t_{k-1})$ 条件下的概率建模来计算序列的概率：

$$
p(t_1, t_2, \ldots, t_N) = \prod_{k=1}^{N} p(t_k \mid t_1, t_2, \ldots, t_{k-1})
$$

最近最先进的神经语言模型（Józefowicz et al., 2016; Melis et al., 2017; Merity et al., 2017）[20][21][22]计算与上下文无关的token表示 $x_k^{LM}$ （通过token嵌入或字符上的CNN），然后将其通过 $L$ 层前向LSTM。在每个位置 $k$ ，每个LSTM层输出一个上下文相关的表示 $\overrightarrow{h}_{k,j}^{LM}$ ，其中 $j = 1, \ldots, L$ 。顶层LSTM的输出 $\overrightarrow{h}_{k,L}^{LM}$ 用于通过Softmax层预测下一个token $t_{k+1}$ 。

后向LM与前向LM类似，只是它反向运行序列，在给定未来上下文的条件下预测前一个token：

$$
p(t_1, t_2, \ldots, t_N) = \prod_{k=1}^{N} p(t_k \mid t_{k+1}, t_{k+2}, \ldots, t_N)
$$

它可以以与前向LM类似的方式实现，深度为 $L$ 层的模型中的每个后向LSTM层 $j$ 生成给定 $(t_{k+1}, \ldots, t_N)$ 条件下 $t_k$ 的表示 $\overleftarrow{h}_{k,j}^{LM}$ 。

biLM结合了前向和后向LM。我们的公式联合最大化前向和后向方向的对数似然：

$$
\sum_{k=1}^{N} \big( \log p(t_k \mid t_1, \ldots, t_{k-1}; \Theta_x, \overrightarrow{\Theta}_{\text{LSTM}}, \Theta_s) + \log p(t_k \mid t_{k+1}, \ldots, t_N; \Theta_x, \overleftarrow{\Theta}_{\text{LSTM}}, \Theta_s) \big)
$$

我们将前向和后向方向的token表示（ $\Theta_x$ ）和Softmax层（ $\Theta_s$ ）的参数绑定在一起，同时为每个方向上的LSTM保留独立的参数。总体而言，这个公式与Peters等人（2017）[3]的方法类似，不同之处在于我们在方向之间共享一些权重，而不是使用完全独立的参数。在下一节中，我们与之前的工作不同，引入了一种新的学习词表示的方法，该表示是biLM层的线性组合。

### 3.2 ELMo

ELMo是biLM中间层表示的任务特定组合。对于每个token $t_k$ ，一个 $L$ 层的biLM计算一组 $2L + 1$ 个表示：

$$
R_k = \{ x_k^{LM}, \overrightarrow{h}_{k,j}^{LM}, \overleftarrow{h}_{k,j}^{LM} \mid j = 1, \ldots, L \} = \{ h_{k,j}^{LM} \mid j = 0, \ldots, L \}
$$

其中 $h_{k,0}^{LM}$ 是token层， $h_{k,j}^{LM} = [\overrightarrow{h}_{k,j}^{LM}; \overleftarrow{h}_{k,j}^{LM}]$ ，对应每个biLSTM层。

为了纳入下游模型，ELMo将 $R$ 中的所有层折叠为一个向量 $\text{ELMo}_k = E(R_k; \Theta_e)$ 。在最简单的情况下，ELMo仅选择顶层 $E(R_k) = h_{k,L}^{LM}$ ，如TagLM（Peters et al., 2017）[3]和CoVe（McCann et al., 2017）[4]中。更一般地，我们计算所有biLM层的任务特定加权：

$$
\text{ELMo}_k^{\text{task}} = E(R_k; \Theta^{\text{task}}) = $\gamma$^{\text{task}} \sum_{j=0}^{L} s_j^{\text{task}} h_{k,j}^{LM} \qquad (1)
$$

在式(1)中， $s^{\text{task}}$ 是softmax归一化的权重，标量参数 $\gamma^{\text{task}}$ 允许任务模型缩放整个ELMo向量。 $\gamma$ 对于辅助优化过程具有实际重要性（详见补充材料）。考虑到每个biLM层的激活具有不同的分布，在某些情况下，在加权之前对每个biLM层应用层归一化（Ba et al., 2016）[23]也是有帮助的。

### 3.3 使用biLM进行监督NLP任务

给定一个预训练的biLM和一个目标NLP任务的监督架构，使用biLM来改进任务模型是一个简单的过程。我们只需运行biLM并记录每个词的所有层表示。然后，让最终任务模型学习这些表示的线性组合，如下所述。

首先考虑没有biLM的监督模型的最低层。大多数监督NLP模型在最低层共享一个通用架构，这使我们能够以一致、统一的方式添加ELMo。给定一个token序列 $(t_1, \ldots, t_N)$ ，标准做法是使用预训练的词嵌入和可选的基于字符的表示为每个token位置形成一个与上下文无关的token表示 $x_k$ 。然后，模型形成一个上下文敏感的表示 $h_k$ ，通常使用双向RNN、CNN或前馈网络。

为了将ELMo添加到监督模型中，我们首先冻结biLM的权重，然后将ELMo向量 $\text{ELMo}_k^{\text{task}}$ 与 $x_k$ 拼接起来，并将ELMo增强的表示 $[x_k; \text{ELMo}_k^{\text{task}}]$ 传入任务RNN。对于某些任务（例如，SNLI、SQuAD），我们观察到通过引入另一组特定于输出的线性权重并将 $h_k$ 替换为 $[h_k; \text{ELMo}_k^{\text{task}}]$ ，在任务RNN的输出处也包含ELMo能带来进一步改进。由于监督模型的其余部分保持不变，这些添加可以在更复杂的神经模型中进行。例如，参见第4节中SNLI实验（其中biLSTM之后是双向注意力层），或共指消解实验（其中在biLSTM之上分层聚类模型）。

最后，我们发现向ELMo添加适量的dropout（Srivastava et al., 2014）[24]并在某些情况下通过在损失中添加 $\lambda \| w \|_2^2$ 来正则化ELMo权重是有益的。这施加了一个归纳偏置，使ELMo权重保持在所有biLM层的平均附近。

### 3.4 预训练双向语言模型架构

本文中的预训练biLM与Józefowicz等人（2016）[20]和Kim等人（2015）[25]中的架构类似，但经过修改以支持两个方向的联合训练以及在LSTM层之间添加残差连接。我们在本工作中专注于大规模biLM，因为Peters等人（2017）[3]强调了使用biLM而非仅前向LM以及大规模训练的重要性。

为了在保持纯字符输入表示的同时，平衡整体语言模型困惑度与模型大小及下游任务的计算需求，我们将Józefowicz等人（2016）[20]中最佳单模型CNN-BIG-LSTM的所有嵌入和隐藏维度减半。最终模型使用 $L = 2$ 个biLSTM层，每层4096个单元和512维投影，并在第一层到第二层之间加入残差连接。上下文不敏感的类型（type）表示使用2048个字符n-gram卷积滤波器，随后是两个高速网络层（Srivastava et al., 2015）[26]和一个线性投影降至512维表示。因此，biLM为每个输入token提供三层表示，包括因纯字符输入而在训练集之外的token。相比之下，传统词嵌入方法仅能为固定词汇表中的token提供一层表示。

在1B Word Benchmark（Chelba et al., 2014）[14]上训练10个epoch后，前向和后向的平均困惑度为39.7，而前向CNN-BIG-LSTM为30.0。总体而言，我们发现前向和后向困惑度大致相等，后向值略低。

一旦预训练完成，biLM可以为任何任务计算表示。在某些情况下，在领域特定数据上微调biLM会导致困惑度显著下降和下游任务性能提升。这可以视为biLM的一种领域迁移。因此，在大多数情况下我们在下游任务中使用微调后的biLM。详见补充材料。

## 4 评估

表1显示了ELMo在六个多样化的基准NLP任务中的性能。在每个被考虑的任务中，仅添加ELMo就建立了新的最先进结果，相对错误率在强基线上的降低范围从6%到20%。这是一个跨越多样化模型架构和语言理解任务的非常通用的结果。在本节的剩余部分，我们提供各个任务结果的高级概述；完整的实验细节见补充材料。

**问答** Stanford问答数据集（SQuAD）（Rajpurkar et al., 2016）[27]包含超过10万个众包问答对，答案位于给定维基百科段落的一个片段中。我们的基线模型（Clark and Gardner, 2017）[28]是Seo等人（BiDAF; 2017）[29]中Bidirectional Attention Flow模型的改进版本。它添加了一个自注意力层在双向注意力组件之后，简化了一些池化操作，并将LSTM替换为门控循环单元（GRU; Cho et al., 2014）[30]。在基线模型上添加ELMo后，测试集F1从81.1%提升了4.7%至85.8%，相对基线的错误率降低了24.9%，并将整体单模型最先进水平提高了1.4%。一个11成员的集成将F1推至87.4，在提交排行榜时是整体最先进水平。ELMo带来的4.7%的提升也显著大于向基线模型添加CoVe带来的1.8%的提升（McCann et al., 2017）[4]。

表1：ELMo增强的神经模型与六个基准NLP任务上最先进单模型基线的测试集比较。性能指标因任务而异——SNLI和SST-5使用准确率；SQuAD、SRL和NER使用F1；Coref使用平均F1。由于NER和SST-5的测试集规模较小，我们报告了五次不同随机种子运行的平均值和标准差。"提升"列列出了相对于我们基线的绝对和相对改进。

**文本蕴含** 文本蕴含是在给定"前提"的条件下判断"假设"是否为真的任务。Stanford自然语言推理（SNLI）语料库（Bowman et al., 2015）[31]提供了约55万对假设/前提对。我们的基线模型是来自Chen等人（2017）[7]的ESIM序列模型，它使用biLSTM编码前提和假设，后接矩阵注意力层、局部推理层、另一个biLSTM推理组合层，最后在输出层之前进行池化操作。总体而言，在ESIM模型中添加ELMo在五个随机种子上平均提高了0.7%的准确率。一个五成员集成将整体准确率推至89.3%，超过了之前88.9%的集成最优结果（Gong et al., 2018）[32]。

**语义角色标注** 语义角色标注（SRL）系统建模句子的谓词-论元结构，通常被描述为回答"谁对谁做了什么"。He等人（2017）[8]将SRL建模为一个BIO标注问题，并使用了一个8层深度biLSTM，前向和后向方向交错排列，遵循Zhou和Xu（2015）[33]的方法。如表1所示，当将ELMo添加到He等人（2017）[8]的复现实现中时，单模型测试集F1从81.4%跃升3.2%至84.6%——在OntoNotes基准（Pradhan et al., 2013）[34]上创造了新的最先进水平，甚至比以前的最佳集成结果提高了1.2%。

**共指消解** 共指消解是将文本中指向同一真实世界实体的提及聚类在一起的任务。我们的基线模型是Lee等人（2017）[35]的端到端基于片段的神经模型。它使用biLSTM和注意力机制首先计算片段表示，然后应用softmax提及排序模型来寻找共指链。在我们使用CoNLL 2012共享任务（Pradhan et al., 2012）[36]的OntoNotes共指标注进行的实验中，添加ELMo使平均F1提高了3.2%，从67.2提升到70.4，建立了新的最先进水平，再次比以前的最佳集成结果提高了1.6%的F1。

**命名实体识别** CoNLL 2003 NER任务（Sang and Meulder, 2003）[37]由Reuters RCV1语料库的新闻专线组成，标注了四种不同实体类型（PER、LOC、ORG、MISC）。遵循最近的先进系统（Lample et al., 2016; Peters et al., 2017）[38][3]，基线模型使用预训练词嵌入、基于字符的CNN表示、两个biLSTM层和条件随机场（CRF）损失（Lafferty et al., 2001）[39]，类似于Collobert等人（2011）[40]。如表1所示，我们的ELMo增强的biLSTM-CRF在五次运行中的平均F1达到92.22%。我们的系统与Peters等人（2017）[3]之前最先进水平的关键区别在于，我们允许任务模型学习所有biLM层的加权平均值，而Peters等人（2017）[3]仅使用顶层biLM层。如第5.1节所示，使用所有层而非仅最后一层可以提升多个任务的性能。

**情感分析** Stanford情感树库（SST-5; Socher et al., 2013）[41]中的细粒度情感分类任务涉及从五个标签（从非常负面到非常正面）中选择一个来描述来自电影评论的句子。这些句子包含多样的语言现象，如习语和复杂的句法结构（例如否定），这些对模型来说难以学习。我们的基线模型是来自McCann等人（2017）[4]的双向注意力分类网络（BCN），该网络在使用CoVe嵌入增强时也保持了之前的最先进结果。在BCN模型中将CoVe替换为ELMo，相比最先进水平提高了1.0%的绝对准确率。

## 5 分析

本节提供了消融分析以验证我们的主要主张，并阐明ELMo表示的一些有趣方面。第5.1节表明，在下游任务中使用深度上下文表示可以提升仅使用顶层的先前工作的性能，无论这些表示是来自biLM还是MT编码器，并且ELMo表示提供了最佳整体性能。第5.3节探讨了biLM中捕捉到的不同类型上下文信息，并使用两个内在评估表明，语法信息在较低层得到更好的表示，而语义信息在较高层被捕捉，这与MT编码器一致。它还表明我们的biLM始终提供比CoVe更丰富的表示。此外，我们分析了ELMo在任务模型中的位置敏感性（第5.2节）、训练集大小（第5.4节），并可视化了跨任务的ELMo学习权重（第5.5节）。

### 5.1 替代层加权方案

对于组合biLM层，公式(1)有许多替代方案。先前关于上下文表示的工作仅使用最后一层，无论是来自biLM（Peters et al., 2017）[3]还是MT编码器（CoVe; McCann et al., 2017）[4]。正则化参数 $\lambda$ 的选择也很重要，因为较大的值（如 $\lambda = 1$ ）有效地将加权函数简化为各层的简单平均，而较小的值（例如 $\lambda = 0.001$ ）允许层权重变化。

表2比较了这些替代方案在SQuAD、SNLI和SRL上的表现。包含所有层的表示比仅使用最后一层提高了整体性能，而包含来自最后一层的上下文表示比基线提高了性能。例如，在SQuAD的情况下，仅使用最后一层biLM层比基线提高了开发集F1的3.9%。使用所有biLM层的平均值而非仅使用最后一层使F1又提高了0.3%（比较"Last Only"和 $\lambda=1$ 列），而允许任务模型学习各层权重使F1再提高0.2%（ $\lambda=1$ 对比 $\lambda=0.001$ ）。在大多数情况下，ELMo偏好较小的 $\lambda$ ，尽管对于NER（一个训练集较小的任务），结果对 $\lambda$ 不敏感（未显示）。

CoVe的总体趋势类似，但相对于基线的提升更小。对于SNLI，使用 $\lambda=1$ 对所有层取平均相比仅使用最后一层将开发准确率从88.2%提高到88.7%。SRL的F1在 $\lambda=1$ 情况下相比于仅使用最后一层仅增加了边际的0.1%，达到82.2。

### 5.2 ELMo应在何处包含？

本文中所有任务架构都仅将词嵌入作为最低层双向RNN的输入。然而，我们发现对于某些任务，在任务特定架构中的双向RNN输出处包含ELMo能改善整体结果。如表3所示，对于SNLI和SQuAD，在输入层和输出层都包含ELMo比仅在输入层包含有改进，但对于SRL（和共指消解，未显示），仅在输入层包含时性能最高。对此结果的一个可能解释是，SNLI和SQuAD架构都在双向RNN后使用注意力层，因此在这一层引入ELMo允许模型直接关注biLM的内部表示。在SRL情况下，任务特定的上下文表示可能比biLM的那些更重要。

### 5.3 biLM的表示捕捉了什么信息？

由于添加ELMo比单独使用词向量提高了任务性能，biLM的上下文表示必须编码了对NLP任务通常有用的、但词向量未能捕捉的信息。直观地说，biLM必须使用上下文来消解词的含义。考虑"play"这个高度多义词。表4顶部列出了使用GloVe向量的"play"的最近邻。它们分布在几个词性中（例如，"played"、"playing"作为动词，"player"、"game"作为名词），但集中在"play"的体育相关义项上。相比之下，底部两行展示了使用biLM在源句子中对"play"的上下文表示，从SemCor数据集（见下文）中得到的最近邻句子。在这些情况下，biLM能够消解源句子中的词性和词义。

这些观察可以通过类似于Belinkov等人（2017）[17]的上下文表示内在评估来量化。为了隔离biLM编码的信息，这些表示被直接用于对细粒度词义消歧（WSD）任务和词性标注任务进行预测。使用这种方法，还可以与CoVe以及各个单独层进行比较。

**词义消歧** 给定一个句子，我们可以使用biLM表示通过简单的1-最近邻方法来预测目标词的义项，类似于Melamud等人（2016）[12]。为此，我们首先使用biLM计算SemCor 3.0（我们的训练语料库）（Miller et al., 1994）[42]中所有词的表示，然后取每个义项的平均表示。在测试时，我们再次使用biLM计算给定目标词的表示，并从训练集中取最近邻的义项，对于训练中未观察到的词元则回退到WordNet的第一义项。

表5比较了使用Raganato等人（2017b）[43]的评估框架在Raganato等人（2017a）[44]中四个测试集上的WSD结果。总体而言，biLM顶层表示的F1为69.0，在WSD上优于第一层。这与使用手工特征的最先进WSD监督模型（Iacobacci et al., 2016）[45]和同样使用辅助粗粒度语义标签和词性标签训练的任务特定biLSTM（Raganato et al., 2017a）[44]相竞争。CoVe biLSTM层与biLM层的模式类似（第二层的整体性能高于第一层）；然而，我们的biLM优于CoVe biLSTM，后者落后于WordNet第一义项基线。

**词性标注** 为了检查biLM是否捕捉基本语法，我们使用上下文表示作为线性分类器的输入，该分类器使用Penn Treebank的Wall Street Journal部分（Marcus et al., 1993）[46]预测词性标签。由于线性分类器仅增加少量模型容量，这是对biLM表示的直接测试。与WSD类似，biLM表示与精心调整的任务特定biLSTM（Ling et al., 2015; Ma and Hovy, 2016）[47][48]相竞争。然而，与WSD不同，使用biLM第一层的准确率高于顶层，这与深度biLSTM在多任务训练（Søgaard and Goldberg, 2016; Hashimoto et al., 2017）[16][15]和MT（Belinkov et al., 2017）[17]中的结果一致。CoVe词性标注准确率与biLM具有相同模式，并且就像WSD一样，biLM比CoVe编码器获得了更高的准确率。

**对监督任务的启示** 综合来看，这些实验证实了biLM中不同层表示不同类型的信息，并解释了为什么包含所有biLM层对于下游任务获得最高性能是重要的。此外，biLM的表示比CoVe中的表示更易于迁移到WSD和词性标注任务，这有助于说明为什么ELMo在下游任务中优于CoVe。

### 5.4 样本效率

向模型添加ELMo显著提高了样本效率，无论是在达到最先进性能所需的参数更新次数方面，还是在整体训练集大小方面。例如，SRL模型在不使用ELMo的情况下，经过486个epoch的训练达到最大开发F1。添加ELMo后，模型在第10个epoch就超过了基线的最大值，达到相同性能水平所需的更新次数相对减少了98%。

此外，ELMo增强的模型比没有ELMo的模型更有效地使用较小的训练集。图1比较了有和没有ELMo的基线模型在完整训练集百分比从0.1%变化到100%时的性能。ELMo带来的改进在训练集较小时最大，并显著减少了达到给定性能水平所需的训练数据量。在SRL情况下，使用1%训练集的ELMo模型与使用10%训练集的基线模型具有大致相同的F1。

### 5.5 学习权重的可视化

图2可视化了softmax归一化的学习层权重。在输入层，任务模型偏好第一个biLSTM层。对于共指消解和SQuAD，这种偏好很强，但对于其他任务，分布较小尖峰。输出层权重相对平衡，略微偏好较低层。

## 6 结论

我们引入了一种从biLM学习高质量深度上下文相关表示的通用方法，并展示了将ELMo应用于广泛的NLP任务时带来的巨大改进。通过消融和其他受控实验，我们还证实了biLM层有效地编码了关于上下文中的词的不同类型的语法和语义信息，并且使用所有层可以改善整体任务性能。

## 参考文献

[1] Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg S Corrado, and Jeff Dean. 2013. Distributed representations of words and phrases and their compositionality. In *NIPS*.

[2] Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. Glove: Global vectors for word representation. In *EMNLP*.

[3] Matthew E. Peters, Waleed Ammar, Chandra Bhagavatula, and Russell Power. 2017. Semi-supervised sequence tagging with bidirectional language models. In *ACL*.

[4] Bryan McCann, James Bradbury, Caiming Xiong, and Richard Socher. 2017. Learned in translation: Contextualized word vectors. In *NIPS 2017*.

[5] Joseph P. Turian, Lev-Arie Ratinov, and Yoshua Bengio. 2010. Word representations: A simple and general method for semi-supervised learning. In *ACL*.

[6] Xiaodong Liu, Yelong Shen, Kevin Duh, and Jianfeng Gao. 2017. Stochastic answer networks for machine reading comprehension. *arXiv preprint arXiv:1712.03556*.

[7] Qian Chen, Xiao-Dan Zhu, Zhen-Hua Ling, Si Wei, Hui Jiang, and Diana Inkpen. 2017. Enhanced lstm for natural language inference. In *ACL*.

[8] Luheng He, Kenton Lee, Mike Lewis, and Luke S. Zettlemoyer. 2017. Deep semantic role labeling: What works and what's next. In *ACL*.

[9] John Wieting, Mohit Bansal, Kevin Gimpel, and Karen Livescu. 2016. Charagram: Embedding words and sentences via character n-grams. In *EMNLP*.

[10] Piotr Bojanowski, Edouard Grave, Armand Joulin, and Tomas Mikolov. 2017. Enriching word vectors with subword information. *TACL* 5:135–146.

[11] Arvind Neelakantan, Jeevan Shankar, Alexandre Passos, and Andrew McCallum. 2014. Efficient nonparametric estimation of multiple embeddings per word in vector space. In *EMNLP*.

[12] Oren Melamud, Jacob Goldberger, and Ido Dagan. 2016. context2vec: Learning generic context embedding with bidirectional lstm. In *CoNLL*.

[13] Sepp Hochreiter and Jürgen Schmidhuber. 1997. Long short-term memory. *Neural Computation* 9.

[14] Ciprian Chelba, Tomas Mikolov, Mike Schuster, Qi Ge, Thorsten Brants, Phillipp Koehn, and Tony Robinson. 2014. One billion word benchmark for measuring progress in statistical language modeling. In *INTERSPEECH*.

[15] Kazuma Hashimoto, Caiming Xiong, Yoshimasa Tsuruoka, and Richard Socher. 2017. A joint many-task model: Growing a neural network for multiple nlp tasks. In *EMNLP 2017*.

[16] Anders Søgaard and Yoav Goldberg. 2016. Deep multi-task learning with low level tasks supervised at lower layers. In *ACL 2016*.

[17] Yonatan Belinkov, Nadir Durrani, Fahim Dalvi, Hassan Sajjad, and James R. Glass. 2017. What do neural machine translation models learn about morphology? In *ACL*.

[18] Andrew M. Dai and Quoc V. Le. 2015. Semi-supervised sequence learning. In *NIPS*.

[19] Prajit Ramachandran, Peter Liu, and Quoc Le. 2017. Improving sequence to sequence learning with unlabeled data. In *EMNLP*.

[20] Rafal Józefowicz, Oriol Vinyals, Mike Schuster, Noam Shazeer, and Yonghui Wu. 2016. Exploring the limits of language modeling. *CoRR* abs/1602.02410.

[21] Gábor Melis, Chris Dyer, and Phil Blunsom. 2017. On the state of the art of evaluation in neural language models. *CoRR* abs/1707.05589.

[22] Stephen Merity, Nitish Shirish Keskar, and Richard Socher. 2017. Regularizing and optimizing lstm language models. *CoRR* abs/1708.02182.

[23] Jimmy Ba, Ryan Kiros, and Geoffrey E. Hinton. 2016. Layer normalization. *CoRR* abs/1607.06450.

[24] Nitish Srivastava, Geoffrey E. Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov. 2014. Dropout: a simple way to prevent neural networks from overfitting. *Journal of Machine Learning Research* 15:1929–1958.

[25] Yoon Kim, Yacine Jernite, David Sontag, and Alexander M Rush. 2015. Character-aware neural language models. In *AAAI 2016*.

[26] Rupesh Kumar Srivastava, Klaus Greff, and Jürgen Schmidhuber. 2015. Training very deep networks. In *NIPS*.

[27] Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. 2016. Squad: 100,000+ questions for machine comprehension of text. In *EMNLP*.

[28] Christopher Clark and Matthew Gardner. 2017. Simple and effective multi-paragraph reading comprehension. *CoRR* abs/1710.10723.

[29] Min Joon Seo, Aniruddha Kembhavi, Ali Farhadi, and Hannaneh Hajishirzi. 2017. Bidirectional attention flow for machine comprehension. In *ICLR*.

[30] Kyunghyun Cho, Bart van Merrienboer, Dzmitry Bahdanau, and Yoshua Bengio. 2014. On the properties of neural machine translation: Encoder-decoder approaches. In *SSST@EMNLP*.

[31] Samuel R. Bowman, Gabor Angeli, Christopher Potts, and Christopher D. Manning. 2015. A large annotated corpus for learning natural language inference. In *Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing (EMNLP)*.

[32] Yichen Gong, Heng Luo, and Jian Zhang. 2018. Natural language inference over interaction space. In *ICLR*.

[33] Jie Zhou and Wei Xu. 2015. End-to-end learning of semantic role labeling using recurrent neural networks. In *ACL*.

[34] Sameer Pradhan, Alessandro Moschitti, Nianwen Xue, Hwee Tou Ng, Anders Björkelund, Olga Uryupina, Yuchen Zhang, and Zhi Zhong. 2013. Towards robust linguistic analysis using ontonotes. In *CoNLL*.

[35] Kenton Lee, Luheng He, Mike Lewis, and Luke S. Zettlemoyer. 2017. End-to-end neural coreference resolution. In *EMNLP*.

[36] Sameer Pradhan, Alessandro Moschitti, Nianwen Xue, Olga Uryupina, and Yuchen Zhang. 2012. Conll-2012 shared task: Modeling multilingual unrestricted coreference in ontonotes. In *EMNLP-CoNLL Shared Task*.

[37] Erik F. Tjong Kim Sang and Fien De Meulder. 2003. Introduction to the CoNLL-2003 shared task: Language-independent named entity recognition. In *CoNLL*.

[38] Guillaume Lample, Miguel Ballesteros, Sandeep Subramanian, Kazuya Kawakami, and Chris Dyer. 2016. Neural architectures for named entity recognition. In *NAACL-HLT*.

[39] John D. Lafferty, Andrew McCallum, and Fernando Pereira. 2001. Conditional random fields: Probabilistic models for segmenting and labeling sequence data. In *ICML*.

[40] Ronan Collobert, Jason Weston, Léon Bottou, Michael Karlen, Koray Kavukcuoglu, and Pavel P. Kuksa. 2011. Natural language processing (almost) from scratch. In *JMLR*.

[41] Richard Socher, Alex Perelygin, Jean Y Wu, Jason Chuang, Christopher D Manning, Andrew Y Ng, and Christopher Potts. 2013. Recursive deep models for semantic compositionality over a sentiment treebank. In *EMNLP*.

[42] George A. Miller, Martin Chodorow, Shari Landes, Claudia Leacock, and Robert G. Thomas. 1994. Using a semantic concordance for sense identification. In *HLT*.

[43] Alessandro Raganato, Jose Camacho-Collados, and Roberto Navigli. 2017b. Word sense disambiguation: A unified evaluation framework and empirical comparison. In *EACL*.

[44] Alessandro Raganato, Claudio Delli Bovi, and Roberto Navigli. 2017a. Neural sequence learning models for word sense disambiguation. In *EMNLP*.

[45] Ignacio Iacobacci, Mohammad Taher Pilehvar, and Roberto Navigli. 2016. Embeddings for word sense disambiguation: An evaluation study. In *ACL*.

[46] Mitchell P. Marcus, Beatrice Santorini, and Mary Ann Marcinkiewicz. 1993. Building a large annotated corpus of english: The penn treebank. *Computational Linguistics* 19:313–330.

[47] Wang Ling, Chris Dyer, Alan W. Black, Isabel Trancoso, Ramon Fermandez, Silvio Amir, Luís Marujo, and Tiago Luís. 2015. Finding function in form: Compositional character models for open vocabulary word representation. In *EMNLP*.

[48] Xuezhe Ma and Eduard H. Hovy. 2016. End-to-end sequence labeling via bi-directional LSTM-CNNs-CRF. In *ACL*.

---

# 附录A：补充材料——伴随《深度上下文化词表示》

本补充材料包含第4节中最先进模型的架构、训练流程和超参数选择的详细信息。

所有单独的模型在最低层共享一个通用架构：在若干层堆叠的RNN（所有情况下均为LSTM，除了使用GRU的SQuAD模型）之下是一个与上下文无关的token表示。

## A.1 微调biLM

如第3.4节所述，在任务特定数据上微调biLM通常会导致困惑度显著降低。为了在给定任务上进行微调，监督标签被暂时忽略，biLM在训练集分割上微调一个epoch并在开发集分割上评估。微调后，在任务训练期间固定biLM权重。

表7列出了所考虑任务的开发集困惑度。除了CoNLL 2012外，在所有情况下微调都带来了困惑度的大幅改善，例如SNLI从72.1降至16.8。

微调对监督性能的影响取决于任务。以SNLI为例，微调biLM使单个最佳模型的开发准确率从88.9%提高了0.6%至89.5%。然而，对于情感分类，无论是否使用微调的biLM，开发集准确率大致相同。

## A.2 式(1)中 $\gamma$ 的重要性

式(1)中的 $\gamma$ 参数对于辅助优化具有实际重要性，因为biLM内部表示和任务特定表示之间存在不同的分布。在第5.1节的仅用最后一层的情况下尤为重要。没有这个参数，仅用最后一层的情况在SNLI上表现不佳（远低于基线），而在SRL上训练完全失败。

## A.3 文本蕴含

我们的SNLI基线模型是来自Chen等人（2017）[7]的ESIM序列模型。遵循原始实现，我们对所有LSTM和前馈层使用300维，并使用预训练的300维GloVe嵌入，在训练期间固定。对于正则化，我们对每个LSTM层的输入添加了50%变分dropout（Gal and Ghahramani, 2016）[49]，并在最后两个全连接层的输入处添加了50% dropout（Srivastava et al., 2014）[24]。所有前馈层使用ReLU激活。参数使用Adam（Kingma and Ba, 2015）[50]优化，梯度范数裁剪在5.0，初始学习率0.0004，每当开发集准确率在后续epoch中没有增加时减半。批大小为32。

最佳ELMo配置将ELMo向量添加到最低层LSTM的输入和输出，使用公式(1)并带层归一化和 $\lambda = 0.001$ 。由于ELMo模型参数增加，我们向所有循环和前馈权重矩阵添加了正则化系数为0.0001的 $\ell_2$ 正则化，并在注意力层后添加了50%的dropout。

表8比较了我们系统与先前已发表系统的测试集准确率。总体而言，将ELMo添加到ESIM模型提高了0.7%的准确率，建立了88.7%的新单模型最先进水平，五成员集成将整体准确率推至89.3%。

## A.4 问答

我们的问答模型是Clark和Gardner（2017）[28]模型的简化版本。它通过将每个token的大小写敏感的300维GloVe词向量（Pennington et al., 2014）[2]与使用卷积神经网络然后在学到的字符嵌入上进行最大池化产生的字符嵌入拼接来嵌入token。token嵌入通过一个共享的双向GRU，然后通过BiDAF（Seo et al., 2017）[29]的双向注意力机制。增强后的上下文向量随后通过一个带ReLU激活的线性层、一个使用GRU后接相同注意力机制（应用于上下文对上下文）的残差自注意力层，以及另一个带ReLU的线性层。最后，结果通过线性层来预测答案的开始和结束token。

在GRU和线性层的输入之前使用0.2比率的变分dropout。GRU使用90维，线性层使用180维。我们使用Adadelta优化模型，批大小为45。测试时，我们使用权重的指数移动平均，并将输出跨度限制为最多17。训练期间我们不更新词向量。

当ELMo不带层归一化添加到上下文GRU层的输入和输出，且ELMo权重不进行正则化（ $\lambda=0$ ）时，性能最高。

表9比较了当我们在2017年11月17日提交系统时SQuAD排行榜的测试集结果。总体而言，我们的提交获得了最高的单模型和集成结果，将先前单模型结果（SAN）提高了1.4% F1，将我们的基线提高了4.2%。11成员集成将F1推至87.4%，比之前最佳集成提高了1.0%。

## A.5 语义角色标注

我们的SRL基线模型是He等人（2017）[8]的精确复现。词使用100维向量表示的拼接来表示，初始化使用GloVe（Pennington et al., 2014）[2]和每个词的二元谓语特征（使用100维嵌入表示）。这个200维的token表示然后通过一个8层"交错"biLSTM，隐藏层大小为300维，其中LSTM层的方向逐层交替。这个深层LSTM使用层间高速连接（Srivastava et al., 2015）[26]和变分循环dropout（Gal and Ghahramani, 2016）[49]。这个深度表示然后通过一个最终稠密层后接softmax激活进行投影，形成对所有可能标签的分布。标签包括来自PropBank（Palmer et al., 2005）[51]的语义角色，并使用BIO标注方案表示论元跨度。在训练期间，我们使用Adadelta（学习率1.0， $\rho = 0.95$ ）（Zeiler, 2012）[52]最小化标签序列的负对数似然。在测试时，我们执行Viterbi解码以使用BIO约束强制有效跨度。对所有LSTM隐藏层添加10%的变分dropout。梯度值超过1.0时进行裁剪。模型训练最多500个epoch，或直到验证F1在200个epoch内不改善（以先到者为准）。预训练的GloVe向量在训练期间微调。最终稠密层和所有LSTM的所有单元初始化为正交。所有LSTM的遗忘门偏置初始化为1，所有其他门初始化为0，按照Józefowicz等人（2015）[53]的方法。

表10比较了我们使用ELMo增强的He等人（2017）[8]实现的测试集F1分数与先前的结果。我们单模型的84.6 F1分数代表了CONLL 2012语义角色标注任务的新最先进结果，超过了先前单模型结果2.9 F1和5模型集成1.2 F1。

## A.6 共指消解

我们的共指消解基线模型是Lee等人（2017）[35]的端到端神经模型，所有超参数严格遵循原始实现。

最佳配置将ELMo添加到最低层biLSTM的输入，并使用公式(1)对biLM层进行加权，不进行任何正则化（ $\lambda=0$ ）或层归一化。对ELMo表示添加了50%的dropout。

表11比较了我们的结果与先前已发表的结果。总体而言，我们将单模型最先进水平提高了3.2%的平均F1，并且我们的单模型结果比之前的最佳集成提高了1.6%的F1。除了biLSTM输入外在biLSTM输出也添加ELMo使F1降低了约0.7%（未显示）。

## A.7 命名实体识别

我们的NER基线模型将50维预训练Senna向量（Collobert et al., 2011）[40]与基于CNN的字符表示拼接起来。字符表示使用16维字符嵌入和128个宽度为三个字符的卷积滤波器、ReLU激活和最大池化。token表示通过两个biLSTM层，第一层有200个隐藏单元，第二层有100个隐藏单元，然后是一个最终稠密层和softmax层。在训练期间，我们使用CRF损失，在测试时使用Viterbi算法执行解码，同时确保输出标签序列有效。

对两个biLSTM层的输入都添加了变分dropout。训练期间，如果梯度的 $\ell_2$ 范数超过5.0则重新缩放，使用Adam（恒定学习率0.001）更新参数。预训练的Senna嵌入在训练期间微调。我们在开发集上采用早停，并报告五次不同随机种子运行的平均测试集分数。

ELMo被添加到最低层任务biLSTM的输入。由于CoNLL 2003 NER数据集相对较小，我们通过设置公式(1)中 $\lambda = 0.1$ 来约束可训练的层权重使其有效恒定，从而获得了最佳性能。

表12比较了我们使用ELMo增强的biLSTM-CRF标注器与先前结果的测试集F1分数。总体而言，我们系统的92.22% F1建立了新的最先进水平。与Peters等人（2017）[3]相比，使用biLM所有层的表示提供了适度的改进。

## A.8 情感分类

我们使用了与McCann等人（2017）[4]中描述的几乎相同的双向注意力分类网络架构，不同之处在于我们将最终的maxout网络替换为一个更简单的前馈网络，由两个ReLU层和dropout组成。在我们的实验中，带有批归一化maxout网络的BCN模型达到了显著较低的验证准确率，尽管我们的实现与McCann等人（2017）[4]之间可能存在差异。为了匹配CoVe的训练设置，我们仅在包含四个或更多token的短语上进行训练。我们对biLSTM使用300维隐藏状态，并使用Adam（Kingma and Ba, 2015）[50]优化模型参数，学习率为0.0001。可训练的biLM层权重通过 $\lambda = 0.001$ 进行正则化，我们将ELMo添加到biLSTM的输入和输出；输出的ELMo向量通过第二个biLSTM计算并拼接到输入。

## 补充参考文献

[49] Yarin Gal and Zoubin Ghahramani. 2016. A theoretically grounded application of dropout in recurrent neural networks. In *NIPS*.

[50] Diederik P. Kingma and Jimmy Ba. 2015. Adam: A method for stochastic optimization. In *ICLR*.

[51] Martha Palmer, Paul Kingsbury, and Daniel Gildea. 2005. The proposition bank: An annotated corpus of semantic roles. *Computational Linguistics* 31:71–106.

[52] Matthew D. Zeiler. 2012. Adadelta: An adaptive learning rate method. *CoRR* abs/1212.5701.

[53] Rafal Józefowicz, Wojciech Zaremba, and Ilya Sutskever. 2015. An empirical exploration of recurrent network architectures. In *ICML*.
