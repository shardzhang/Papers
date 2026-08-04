# MapReduce：简化大规模集群上的数据处理

> Jeffrey Dean, Sanjay Ghemawat | Google


本文介绍了 MapReduce：简化大规模集群上的数据处理。核心内容：

- 提出 MapReduce 编程模型，一种用于大规模数据集（大于 1 TB）并行计算的编程模型和实现，用户只需编写 Map 和 Reduce 函数
- MapReduce 运行时自动处理数据分区、任务调度、容错和机器间通信，隐藏并行化、容错性、局部性优化和负载均衡的细节
- 在 Google 大规模集群上的实验表明：排序程序可在 891 秒内完成 1 TB 数据的排序，容错机制能在 200 个 worker 失效时仅增加 5% 的执行时间

关键发现：

- 限制性编程模型使得并行化和分布式计算变得简单，天然支持容错
- 网络带宽是稀缺资源，局部性优化和中间数据本地写入可显著减少网络传输
- 冗余执行（备份任务）可大幅减少慢机器的影响，禁用后排序时间增加 44%

---


## 摘要

MapReduce 是一个用于处理和生成大规模数据集的编程模型及相关实现。用户通过指定一个处理键/值对以生成一组中间键/值对的 **Map** 函数，和一个合并所有与同一中间键关联的中间值的 **Reduce** 函数，来描述计算任务。如本文所示，许多现实世界的任务都可以在这个模型内表达。以该函数式风格编写的程序会自动被并行化并在大规模商用机器集群上执行。运行时系统负责对数据进行分片、在一组机器上调度执行、处理机器故障和管理必需的机器间通信等细节。这使得没有并行和分布式系统经验的程序员也能轻松利用大规模分布式资源。我们的 MapReduce 实现运行在由商用机器组成的大规模集群上，并且具有高度可扩展性：一个典型的 MapReduce 计算可以在数千台机器上处理数 TB 的数据。程序员发现该系统易于使用：已有数千个 MapReduce 程序被实现，每天有超过一千个 MapReduce 作业在 Google 的集群上执行。


## 1 引言

过去五年，本文作者及 Google 的许多人实现了数百个用于处理海量原始数据的专用计算，如爬取文档、Web 请求日志等，以计算各种衍生数据，如倒排索引、Web 文档的图结构表示、每台主机抓取的页面摘要、每日最热门查询的集合等。其中大部分计算在概念上都很直接，但输入数据通常很大（可达数 TB），且计算必须分布到成百上千台机器上以合理时间内完成。

如何并行化计算、分发数据和处理故障等问题，使原本简单的计算因需要处理大量复杂代码而变得晦涩难懂。

针对此问题，我们设计了一种新的抽象，使我们能将上述简单计算中涉及的复杂逻辑表达出来，同时隐藏并行化、容错性、数据分布和负载均衡等混乱细节。这一抽象借鉴了 Lisp 和许多其他函数式语言中的 `map` 和 `reduce` 原语。我们意识到我们的大多数计算都涉及对输入中的每个逻辑"记录"进行 **map** 操作以计算一组中间键/值对，然后对所有具有相同键的值进行 **reduce** 操作以适当合并导出数据。结合用户指定的 map 和 reduce 操作的函数式模型，我们实现了一个自动将计算并行化并在大规模商用 PC 集群上执行高可靠性的 MapReduce 运行库。

本文的主要贡献在于：一个简单而强大的接口，支持自动并行化和大规模分布式计算，且其实现可在大规模商用 PC 集群上实现高性能。


## 2 编程模型

MapReduce 编程模型以输入键/值对集合出发，生成输出键/值对集合。MapReduce 库的用户将计算表示为两个函数：**Map** 和 **Reduce**。

用户编写的 **Map** 函数接收一个输入对并生成一组中间键/值对。MapReduce 库将所有与同一中间键 $k$ 相关联的中间值分组，然后传递给 **Reduce** 函数。

**Reduce** 函数接受一个中间键 $k$ 和该键的一组值，将它们合并成一个更小的值集合。通常每次 Reduce 调用只产生零个或一个输出值。中间值通过迭代器提供给用户的 Reduce 函数，这使我们能处理因值列表太大而无法放入内存的数据集。


### 2.1 示例

考虑统计大量文档中每个单词出现次数的问题。用户将编写类似以下伪代码：

```
map(String key, String value):
  // key: document name
  // value: document contents
  for each word w in value:
    EmitIntermediate(w, "1");

reduce(String key, Iterator values):
  // key: a word
  // values: a list of counts
  int result = 0;
  for each v in values:
    result += ParseInt(v);
  Emit(AsString(result));
```

map 函数对每个单词发出一条附带出现次数（本例中为 '1'）的记录。reduce 函数将某个特定单词的所有计数加和。

此外，用户编写代码填充一个 mapreduce 规约对象，指定输入和输出文件名及可选的调优参数。然后用户调用 MapReduce 函数，传入规约对象。用户的代码与 MapReduce 库（用 C++ 实现）链接在一起。附录 A 包含此示例的完整程序文本。

### 2.2 类型

尽管上述伪代码使用字符串输入和输出，但从概念上讲，用户提供的 map 和 reduce 函数有相关的类型：

$$
\text{map } (k_1, v_1) \to \text{list}(k_2, v_2)
$$
$$
\text{reduce } (k_2, \text{list}(v_2)) \to \text{list}(v_2)
$$

即输入键和值来自与输出键和值不同的域。此外，中间键和值与输出键和值来自同一个域。

我们的 C++ 实现将字符串传递给用户定义的函数，并让用户代码负责在字符串与适当类型之间进行转换。

### 2.3 更多示例

以下是一些可轻松表示为 MapReduce 计算的有趣程序的简单示例：

**分布式 Grep**：如果输入行匹配给定模式，map 函数发出该行。reduce 函数是一个恒等函数，仅将中间数据复制到输出。

**URL 访问频率计数**：map 函数处理网页请求日志并输出 $\langle \text{URL}, 1 \rangle$ 对。reduce 函数将同一 URL 的所有值加和，输出 $\langle \text{URL}, \text{total count} \rangle$ 对。

**反向 Web 链接图**：map 函数对在名为 source 的页面中发现的指向某个目标 URL 的每个链接输出 $\langle \text{target}, \text{source} \rangle$ 对。reduce 函数将与给定目标 URL 关联的所有源 URL 列表连接，输出 $\langle \text{target}, \text{list}(\text{source}) \rangle$ 对。

**每主机词向量**：词向量将文档或文档集中出现的最重要单词总结为 $\langle \text{word}, \text{frequency} \rangle$ 对列表。map 函数对每个输入文档输出 $\langle \text{hostname}, \text{term vector} \rangle$ 对。reduce 函数接收一个给定主机的所有文档词向量，将它们相加，丢弃低频词，然后输出最终的 $\langle \text{hostname}, \text{term vector} \rangle$ 对。

**倒排索引**：map 函数解析每个文档，发出 $\langle \text{word}, \text{document ID} \rangle$ 对序列。reduce 函数接收给定单词的所有对，对相应的文档 ID 排序，输出 $\langle \text{word}, \text{list}(\text{document ID}) \rangle$ 对。所有输出对的集合构成一个简单的倒排索引。扩展此计算以跟踪单词位置也很容易。

**分布式排序**：map 函数从每条记录中提取键，发出 $\langle \text{key}, \text{record} \rangle$ 对。reduce 函数原样输出所有对。此计算依赖于第 4.1 节所述的分区工具和第 4.2 节所述的排序属性。


## 3 实现

MapReduce 接口可以有多种不同的实现。正确选择取决于环境。例如，一种实现可能适用于小型共享内存机器，另一种适用于大型 NUMA 多处理器，还有一种适用于更大规模的联网机器集合。

本节描述针对 Google 广泛使用的计算环境的一种实现：由交换以太网 [4] 连接的商用 PC 大规模集群。在我们的环境中：

1. 机器通常是运行 Linux 的双处理器 x86 处理器，每台机器 2-4 GB 内存。
2. 使用商用网络硬件——通常在机器层面为 100 兆比特/秒或 1 吉比特/秒，但总对分带宽平均要低得多。
3. 集群由成百上千台机器组成，因此机器故障很常见。
4. 存储由直接连接到各台机器的廉价 IDE 磁盘提供。使用内部开发的分布式文件系统 [8] 来管理存储在这些磁盘上的数据。该文件系统利用复制在不可靠硬件上提供可用性和可靠性。
5. 用户将作业提交到调度系统。每个作业由一组任务组成，由调度器映射到集群中的一组可用机器。

### 3.1 执行概述

Map 调用通过自动将输入数据划分为 $M$ 个分片分布到多台机器上。这些输入分片可以由不同机器并行处理。Reduce 调用通过使用分区函数（如 $\text{hash}(\text{key}) \bmod R$ ）将中间键空间划分为 $R$ 个分片来分布。分区数 $R$ 和分区函数由用户指定。

图 1 展示了我们的实现中 MapReduce 操作的总体流程。当用户程序调用 MapReduce 函数时，发生以下操作序列（图 1 中的编号标签对应于下面列表中的编号）：

1. 用户程序中的 MapReduce 库首先将输入文件分割为 $M$ 个分片，每个分片通常为 16 MB 到 64 MB（可通过可选参数由用户控制）。然后它在集群的一台台机器上启动程序的许多副本。
2. 程序的副本中有一个是特殊的——**master**。其余是由 master 分配工作的 **worker**。有 $M$ 个 map 任务和 $R$ 个 reduce 任务需要分配。master 挑选空闲 worker 并给每个 worker 分配一个 map 任务或 reduce 任务。
3. 被分配 map 任务的 worker 读取相应输入分片的内容。它从输入数据中解析出键/值对，并将每个对传递给用户定义的 Map 函数。Map 函数产生的中间键/值对缓存在内存中。
4. 缓冲的对定期写入本地磁盘，由分区函数划分为 $R$ 个区域。这些缓冲对在本地磁盘上的位置被传递回 master，后者负责将这些位置转发给 reduce worker。
5. 当 reduce worker 被 master 通知这些位置时，它使用远程过程调用从 map worker 的本地磁盘读取缓冲数据。当 reduce worker 读取了所有中间数据后，它按中间键排序，以便将所有相同键的出现分组在一起。排序是必要的，因为通常许多不同键映射到同一个 reduce 任务。如果中间数据量太大无法放入内存，则使用外部排序。
6. reduce worker 遍历排序后的中间数据，对于遇到的每个唯一中间键，将该键和相应的一组中间值传递给用户的 Reduce 函数。Reduce 函数的输出追加到该 reduce 分区的最终输出文件中。
7. 当所有 map 任务和 reduce 任务完成时，master 唤醒用户程序。此时，用户程序中的 MapReduce 调用返回到用户代码。

成功完成后，mapreduce 执行的输出可在 $R$ 个输出文件中获取（每个 reduce 任务一个文件，文件名由用户指定）。通常，用户不需要将这些 $R$ 个输出文件合并为一个文件——他们通常将这些文件作为输入传递给另一个 MapReduce 调用，或从另一个能够处理分片为多个文件的输入的分布式应用中使用。

### 3.2 Master 数据结构

Master 维护若干数据结构。对于每个 map 任务和 reduce 任务，它存储状态（空闲、进行中或已完成）和 worker 机器的标识（对于非空闲任务）。

Master 是将中间文件区域的位置信息从 map 任务传播到 reduce 任务的通道。因此，对于每个已完成的 map 任务，master 存储该 map 任务产生的 $R$ 个中间文件区域的位置和大小。随着 map 任务完成，不断接收位置和大小信息的更新。这些信息被增量推送给拥有进行中 reduce 任务的 worker。

### 3.3 容错

由于 MapReduce 库旨在帮助使用成百上千台机器处理非常大的数据量，因此它必须优雅地容忍机器故障。

**Worker 故障**。Master 定期 ping 每个 worker。如果在特定期限内未收到 worker 的响应，master 将该 worker 标记为失败。该 worker 已完成的任何 map 任务都被重置回初始空闲状态，从而可在其他 worker 上调度。类似地，失败 worker 上任何进行中的 map 任务或 reduce 任务也被重置为空闲并重新调度。

已完成的 map 任务在故障时需要重新执行，因为它们的输出存储在失败机器的本地磁盘上，因此无法访问。已完成的 reduce 任务不需要重新执行，因为它们的输出存储在全局文件系统中。

当一个 map 任务首先由 worker A 执行，然后由 worker B 执行（因为 A 失败）时，所有执行 reduce 任务的 worker 都会被通知重新执行。任何尚未从 worker A 读取数据的 reduce 任务将从 worker B 读取数据。

MapReduce 能容忍大规模 worker 故障。例如，在一次 MapReduce 操作期间，正在运行集群的网络维护导致一组 80 台机器同时变得无法访问数分钟。MapReduce master 简单地重新执行了不可达 worker 机器完成的工作，并持续向前推进，最终完成了 MapReduce 操作。

**Master 故障**。让 master 定期写入上述 master 数据结构的检查点很容易。如果 master 任务死亡，可以从最后一个检查点状态启动一个新副本。然而，由于只有一个 master，其故障的可能性不大；因此我们当前的实现在 master 失败时中止 MapReduce 计算。客户端可以检查此条件并按需重试 MapReduce 操作。

**存在故障时的语义**。当用户提供的 map 和 reduce 操作符是其输入值的确定性函数时，我们的分布式实现产生与整个程序的无故障顺序执行相同的输出。

我们依赖 map 和 reduce 任务输出的原子提交来实现这一特性。每个进行中的任务将其输出写入私有临时文件。reduce 任务产生一个这样的文件，而 map 任务产生 $R$ 个这样的文件（每个 reduce 任务一个）。当 map 任务完成时，worker 向 master 发送一条消息，并在消息中包含 $R$ 个临时文件的名称。如果 master 收到一个已完成的 map 任务的完成消息，它忽略该消息。否则，它将 $R$ 个文件的名称记录在 master 数据结构中。

当 reduce 任务完成时，reduce worker 原子性地将临时输出文件重命名为最终输出文件。如果同一个 reduce 任务在多台机器上执行，将对同一个最终输出文件执行多次重命名调用。我们依赖底层文件系统提供的原子重命名操作来保证最终文件系统状态只包含一次 reduce 任务执行所产生的数据。

绝大多数 map 和 reduce 操作符是确定性的，在这种情况下我们的语义等价于顺序执行，这使得程序员很容易推理其程序的行为。当 map 和/或 reduce 操作符是非确定性时，我们提供的语义较弱但仍然合理。在存在非确定性操作符的情况下，特定 reduce 任务 $R_1$ 的输出等价于非确定性程序的顺序执行为 $R_1$ 产生的输出。然而，不同 reduce 任务 $R_2$ 的输出可能对应于非确定性程序的另一次顺序执行为 $R_2$ 产生的输出。

考虑 map 任务 $M$ 和 reduce 任务 $R_1$ 、 $R_2$ 。令 $e(R_i)$ 为 $R_i$ 的已提交执行。较弱语义出现是因为 $e(R_1)$ 可能读取了一次 $M$ 执行产生的输出，而 $e(R_2)$ 读取了另一次 $M$ 执行产生的输出。

### 3.4 局部性

在我们的计算环境中，网络带宽是相对稀缺的资源。我们利用输入数据（由 GFS [8] 管理）存储在构成集群的机器本地磁盘上这一事实来节省网络带宽。GFS 将每个文件分为 64 MB 块，并在不同机器上存储每个块的多个副本（通常为 3 个）。MapReduce master 考虑输入文件的位置信息，尝试在包含相应输入数据副本的机器上调度 map 任务。如果这不可行，它尝试在靠近该任务输入数据副本的位置调度 map 任务（例如，在与包含数据的机器位于同一网络交换机上的 worker 机器上）。当在集群中相当一部分 worker 上运行大型 MapReduce 操作时，大多数输入数据在本地读取，不消耗网络带宽。

### 3.5 任务粒度

如前所述，我们将 map 阶段细分为 $M$ 个分片，将 reduce 阶段细分为 $R$ 个分片。理想情况下， $M$ 和 $R$ 应远大于 worker 机器的数量。让每个 worker 执行许多不同任务可改善动态负载均衡，并在 worker 失败时加快恢复速度：其已完成的大量 map 任务可以分布到所有其他 worker 机器上。

在我们的实现中， $M$ 和 $R$ 的大小存在实际限制，因为 master 必须做出 $O(M + R)$ 个调度决策，并在内存中维护 $O(M \times R)$ 状态。不过内存使用的常数因子很小： $O(M \times R)$ 部分的状态大约为每个 map 任务/reduce 任务对应一个字节的数据。

此外， $R$ 通常受用户限制，因为每个 reduce 任务的输出最终位于一个单独的输出文件中。实践中，我们倾向于选择 $M$ 使得每个单独任务大约处理 16 MB 到 64 MB 的输入数据（以便上述局部性优化最有效），并使 $R$ 为我们期望使用的 worker 机器数量的小倍数。我们经常执行 $M = 200,000$ 和 $R = 5,000$ 的 MapReduce 计算，使用 2,000 个 worker 机器。

### 3.6 备份任务

延长 MapReduce 操作总耗时的常见原因之一是"落后者"：一台在计算中完成最后几个 map 或 reduce 任务之一时花费异常长时间的机器。落后者可能由多种原因引起。例如，磁盘有问题的机器可能频繁出现可纠正的错误，使其读取性能从 30 MB/s 降至 1 MB/s。集群调度系统可能在该机器上调度了其他任务，由于争夺 CPU、内存、本地磁盘或网络带宽而导致 MapReduce 代码执行变慢。我们最近遇到的一个问题是机器初始化代码中的 bug 导致处理器缓存被禁用：受影响机器上的计算速度降低了 100 倍以上。

我们有一个通用机制来缓解落后者问题。当 MapReduce 操作接近完成时，master 调度剩余进行中任务的备份执行。无论是主执行还是备份执行完成，任务即标记为已完成。我们对这一机制进行了调优，使其通常仅将操作使用的计算资源增加几个百分点。我们发现这显著缩短了大型 MapReduce 操作的完成时间。例如，第 5.3 节中描述的排序程序在禁用备份任务机制时需要多 44% 的时间才能完成。


## 4 改进

虽然编写 Map 和 Reduce 函数提供的基本功能对大多数需求已足够，但我们发现了一些有用的扩展。本节描述这些扩展。

### 4.1 分区函数

MapReduce 的用户指定所需的 reduce 任务/输出文件数量 $R$ 。数据通过中间键上的分区函数分布到这些任务。提供了一个默认分区函数，使用哈希（如 " $\text{hash}(\text{key}) \bmod R$ "）。这通常会产生相当均衡的分区。然而，在某些情况下，根据键的其他函数对数据进行分区是有用的。例如，有时输出键是 URL，我们希望单个主机的所有条目最终位于同一输出文件中。为了支持此类情况，MapReduce 库的用户可以提供特殊的分区函数。例如，使用 " $\text{hash}(\text{Hostname}(\text{urlkey})) \bmod R$ " 作为分区函数可使来自同一主机的所有 URL 最终位于同一输出文件中。

### 4.2 排序保证

我们保证在给定分区内，中间键/值对按键递增顺序处理。这一排序保证使得为每个分区生成排序后的输出文件变得容易，当输出文件格式需要支持高效的按键随机访问查找时，或输出文件的用户发现数据排序后更方便时，这很有用。

### 4.3 Combiner 函数

在某些情况下，每个 map 任务产生的中间键存在显著重复，且用户指定的 Reduce 函数是可交换和可结合的。第 2.1 节的单词统计示例就是一个很好的例子。由于词频趋向于遵循 Zipf 分布，每个 map 任务将产生数百或数千条格式为 `the, 1` 的记录。所有这些计数将通过网络发送到单个 reduce 任务，然后由 Reduce 函数加和为一个数字。我们允许用户指定可选的 Combiner 函数，在数据通过网络发送之前进行部分合并。

Combiner 函数在每台执行 map 任务的机器上执行。通常同一段代码用于实现 combiner 和 reduce 函数。reduce 函数和 combiner 函数之间的唯一区别是 MapReduce 库如何处理函数的输出。reduce 函数的输出写入最终输出文件。combiner 函数的输出写入将发送到 reduce 任务的中间文件。

部分合并显著加速了某些类别的 MapReduce 操作。附录 A 包含一个使用 combiner 的示例。

### 4.4 输入和输出类型

MapReduce 库支持以多种不同格式读取输入数据。例如，"text" 模式输入将每一行视为一个键/值对：键是文件中的偏移量，值是行的内容。另一种常见的支持格式存储按键排序的键/值对序列。每种输入类型实现知道如何将自身分割成有意义的数据范围，作为单独的 map 任务进行处理。用户可以通过提供简单读取器接口的实现来添加对新输入类型的支持，不过大多数用户只使用少数预定义输入类型之一。

读取器不一定要提供从文件读取的数据。例如，很容易定义一个从数据库或内存中映射的数据结构中读取记录的读取器。

类似地，我们支持一组输出类型，用于以不同格式生成数据，用户代码很容易添加对新输出类型的支持。

### 4.5 副作用

在某些情况下，MapReduce 用户发现从 map 和/或 reduce 操作符的额外输出中生成辅助文件很方便。我们依赖应用程序编写者使这样的副作用具有原子性和幂等性。通常应用程序写入临时文件，并在完全生成后原子性地重命名此文件。

我们不提供对单个任务产生的多个输出文件进行原子两阶段提交的支持。因此，产生具有跨文件一致性要求的多个输出文件的任务应该是确定性的。这一限制在实践中从未成为问题。

### 4.6 跳过坏记录

有时用户代码中存在 bug，导致 Map 或 Reduce 函数在某些记录上确定性崩溃。此类 bug 阻止 MapReduce 操作完成。通常的做法是修复 bug，但有时这不可行；也许 bug 位于无法获得源代码的第三方库中。此外，有时忽略少量记录是可接受的，例如在对大型数据集进行统计分析时。我们提供一种可选的执行模式，MapReduce 库在其中检测导致确定性崩溃的记录并跳过这些记录以推进工作。

每个 worker 进程安装一个信号处理程序，捕获段错误和总线错误。在调用用户 Map 或 Reduce 操作之前，MapReduce 库将参数的序列号存储在全局变量中。如果用户代码产生信号，信号处理程序将一个包含序列号的"最后一搏"UDP 包发送给 MapReduce master。当 master 在特定记录上看到超过一次失败时，它在发出相应 Map 或 Reduce 任务的下一次重新执行时指示应跳过该记录。

### 4.7 本地执行

Map 或 Reduce 函数中的调试问题可能很棘手，因为实际计算发生在分布式系统中，通常涉及数千台机器，工作分配决策由 master 动态做出。为了便于调试、性能分析和中小规模测试，我们开发了 MapReduce 库的替代实现，在本地机器上顺序执行 MapReduce 操作的所有工作。提供控制给用户，以便计算可限制在特定的 map 任务。用户使用特殊标志调用其程序，然后可以轻松使用任何他们认为有用的调试或测试工具（如 gdb）。

### 4.8 状态信息

Master 运行一个内部 HTTP 服务器，导出一组状态页面供人工使用。状态页面显示计算的进度，例如已完成的任务数、进行中的任务数、输入字节数、中间数据字节数、输出字节数、处理速率等。页面还包含每个任务生成的标准错误和标准输出文件的链接。用户可以使用这些数据预测计算将花费多长时间，以及是否应该增加更多资源。这些页面也可用于找出计算速度远慢于预期的原因。

此外，顶级状态页面显示哪些 worker 失败，以及它们在失败时正在处理哪些 map 和 reduce 任务。该信息在尝试诊断用户代码中的 bug 时很有用。

### 4.9 计数器

MapReduce 库提供一个计数器工具来计数各种事件的发生次数。例如，用户代码可能需要统计处理的单词总数或索引的德语文档数量等。

要使用此工具，用户代码创建一个命名计数器对象，然后在 Map 和/或 Reduce 函数中适当递增计数器。例如：

```
Counter* uppercase;
uppercase = GetCounter("uppercase");
map(String name, String contents):
  for each word w in contents:
    if (IsCapitalized(w)):
      uppercase->Increment();
    EmitIntermediate(w, "1");
```

来自各个 worker 机器的计数器值定期传播给 master（随 ping 响应捎带）。master 聚合来自成功 map 和 reduce 任务的计数器值，并在 MapReduce 操作完成时将其返回给用户代码。当前计数器值也显示在 master 状态页面上，以便人工观察实时计算的进度。在聚合计数器值时，master 消除同一 map 或 reduce 任务的重复执行的影响，以避免重复计数。（重复执行可能由备份任务的使用或由于故障导致的任务重新执行而产生。）

一些计数器值由 MapReduce 库自动维护，例如处理的输入键/值对数和产生的输出键/值对数。

用户发现计数器工具对 MapReduce 操作的行为完整性检查很有用。例如，在某些 MapReduce 操作中，用户代码可能需要确保产生的输出对数恰好等于处理的输入对数，或确保处理的德语文档比例在总处理文档数量的可接受范围内。


## 5 性能

本节我们在大规模机器集群上对两个计算衡量 MapReduce 的性能。一个计算搜索约 1 TB 的数据以查找特定模式。另一个计算对约 1 TB 的数据进行排序。

这两个程序代表了 MapReduce 用户编写的实际程序的很大一个子集——一类程序将数据从一种表示转换为另一种表示，另一类程序从大数据集中提取少量有趣数据。

### 5.1 集群配置

所有程序都在一个由约 1800 台机器组成的集群上执行。每台机器配备两个启用超线程的 2 GHz Intel Xeon 处理器、4 GB 内存、两个 160 GB IDE 磁盘和一个千兆以太网链路。机器排列为两层树形交换网络，根部可用总带宽约 100-200 Gbps。所有机器位于同一托管设施，因此任意两台机器之间的往返时间小于一毫秒。

在 4 GB 内存中，约 1-1.5 GB 被集群上运行的其他任务保留。程序在周末下午执行，此时 CPU、磁盘和网络大部分空闲。

### 5.2 Grep

Grep 程序扫描 $10^{10}$ 条 100 字节记录，搜索一个相对罕见的三字符模式（该模式出现在 92,337 条记录中）。输入分为约 64 MB 的分片（ $M = 15000$ ），整个输出放入一个文件（ $R = 1$ ）。

图 2 显示计算随时间推移的进度。Y 轴显示输入数据的扫描速率。随着更多机器被分配给此 MapReduce 计算，速率逐渐上升，当分配了 1764 个 worker 时达到峰值超过 30 GB/s。随着 map 任务完成，速率开始下降，在计算开始约 80 秒时降至零。整个计算从头到尾约需 150 秒。其中包括约一分钟的启动开销。开销源于将程序传播到所有 worker 机器，以及与 GFS 交互以打开 1000 个输入文件集和获取局部性优化所需信息的延迟。

### 5.3 Sort

Sort 程序对 $10^{10}$ 条 100 字节记录（约 1 TB 数据）进行排序。此程序以 TeraSort 基准测试 [10] 为模型。

排序程序由不足 50 行用户代码组成。一个三行的 Map 函数从文本行中提取 10 字节的排序键，并发出键和原始文本行作为中间键/值对。我们使用内置的 Identity 函数作为 Reduce 操作符。此函数将中间键/值对原样作为输出键/值对传递。最终的排序输出写入一组 2 路复制的 GFS 文件（即程序输出写入 2 TB）。

与前类似，输入数据分为 64 MB 分片（ $M = 15000$ ）。我们将排序输出分为 4000 个文件（ $R = 4000$ ）。分区函数使用键的初始字节将其分到 $R$ 个分片之一。此基准测试的分区函数内置了键分布的知识。在通用排序程序中，我们会增加一个预处理的 MapReduce 操作来收集键的样本，并使用采样键的分布计算最终排序阶段的切分点。

图 3(a) 显示排序程序正常执行的进度。左上图显示输入读取速率。速率峰值约 13 GB/s，并在 200 秒内相当快速地消失，因为所有 map 任务在 200 秒前完成。注意输入速率低于 grep。这是因为排序 map 任务花费大约一半的时间和 I/O 带宽将中间输出写入本地磁盘。而 grep 的相应中间输出大小可忽略。

中左图显示数据从 map 任务通过网络发送到 reduce 任务的速率。这种 shuffle 在第一个 map 任务完成时立即开始。图中的第一个驼峰对应第一批约 1700 个 reduce 任务（整个 MapReduce 分配了约 1700 台机器，每台机器一次最多执行一个 reduce 任务）。计算开始约 300 秒时，这批第一批 reduce 任务中的一些完成，我们开始为剩余的 reduce 任务 shuffle 数据。所有 shuffle 在计算开始约 600 秒时完成。

左下角图显示 reduce 任务将排序数据写入最终输出文件的速率。第一次 shuffle 期结束和写入期开始之间有一个延迟，因为机器正在忙于排序中间数据。写入以约 2-4 GB/s 的速率持续一段时间。所有写入在计算开始约 850 秒时完成。包括启动开销，整个计算耗时 891 秒。这与 TeraSort 基准测试 [18] 当前报告的最佳结果 1057 秒相近。

几点说明：输入速率高于 shuffle 速率和输出速率，因为我们的局部性优化使大多数数据从本地磁盘读取，绕过了我们相对带宽受限的网络。Shuffle 速率高于输出速率，因为输出阶段写入排序数据的两份副本（出于可靠性和可用性原因，我们产生输出的两个副本）。我们写入两个副本是因为底层文件系统提供的可靠性和可用性机制。如果底层文件系统使用纠删码 [14] 而非复制，写入数据的网络带宽需求将会降低。

### 5.4 备份任务的效果

图 3(b) 显示禁用备份任务的排序程序执行。执行流程与图 3(a) 类似，只是存在一个几乎没有写入活动的极长尾部。960 秒后，除 5 个 reduce 任务外全部完成。然而这最后几个落后者直到又过了 300 秒才完成。整个计算耗时 1283 秒，执行时间增加了 44%。

### 5.5 机器故障

图 3(c) 显示我们故意在计算开始几分钟后杀死 200 个 worker 进程（共 1746 个）时排序程序的执行。底层集群调度器立即在这些机器上重新启动了新的 worker 进程（因为只有进程被杀死，机器仍然正常运行）。

Worker 死亡显示为负输入速率，因为一些之前完成的 map 工作消失（由于相应的 map worker 被杀死）并需要重做。这些 map 工作的重新执行相对较快。整个计算包括启动开销在 933 秒内完成（仅比正常执行时间增加 5%）。


## 6 经验

我们于 2003 年 2 月编写了 MapReduce 库的第一个版本，并于 2003 年 8 月进行了显著增强，包括局部性优化、跨 worker 机器任务执行的动态负载均衡等。自那时起，我们惊喜地发现 MapReduce 库对我们所处理的各种问题的广泛适用性。它在 Google 内部的广泛领域中得到使用，包括：

- 大规模机器学习问题
- Google News 和 Froogle 产品的聚类问题
- 用于生成热门查询报告的数据提取（如 Google Zeitgeist）
- 用于新实验和产品的网页属性提取（如从大规模网页语料中提取地理位置以进行本地化搜索）
- 大规模图计算

图 4 显示了随时间推移检入我们主源代码管理系统的独立 MapReduce 程序数量的显著增长，从 2003 年初的 0 个到 2004 年 9 月下旬的近 900 个独立实例。MapReduce 之所以如此成功，是因为它使编写简单程序并在半小时内高效地在千台机器上运行成为可能，极大地加快了开发和原型设计周期。此外，它允许没有分布式和/或并行系统经验的程序员轻松利用大量资源。

在每个作业结束时，MapReduce 库记录作业使用的计算资源的统计信息。在表 1 中，我们展示了 2004 年 8 月在 Google 运行的一部分 MapReduce 作业的统计数据。

| 指标 | 数值 |
|------|------|
| 作业数量 | 29,423 |
| 平均作业完成时间 | 634 秒 |
| 机器使用天数 | 79,186 天 |
| 输入数据读取量 | 3,288 TB |
| 中间数据产生量 | 758 TB |
| 输出数据写入量 | 193 TB |
| 每作业平均 worker 数量 | 157 |
| 每作业平均 worker 死亡数 | 1.2 |
| 每作业平均 map 任务数 | 3,351 |
| 每作业平均 reduce 任务数 | 55 |
| 唯一 map 实现数 | 395 |
| 唯一 reduce 实现数 | 269 |
| 唯一 map/reduce 组合数 | 426 |

### 6.1 大规模索引

迄今为止，MapReduce 最重要的用途之一是完全重写了用于生成 Google Web 搜索服务所用数据结构的产生索引系统。索引系统输入大量由我们的爬虫系统检索的文档，存储为一组 GFS 文件。这些文档的原始内容超过 20 TB 数据。索引过程作为五到十个 MapReduce 操作的序列运行。使用 MapReduce（而非先前版本索引系统中的 ad-hoc 分布式遍历）带来了若干好处：

- 索引代码更简单、更短、更易理解，因为处理容错、分布和并行化的代码隐藏在 MapReduce 库中。例如，计算的一个阶段在表达为 MapReduce 时，从约 3800 行 C++ 代码减少到约 700 行。
- MapReduce 库的性能足够好，使得我们可以将概念上无关的计算分开，而非混合在一起以避免额外的数据遍历。这使得更改索引过程变得容易。例如，在旧索引系统中需要数月才能完成的更改，在新系统中仅需数天即可实现。
- 索引过程变得更易操作，因为大多数由机器故障、慢机和网络问题引起的问题都由 MapReduce 库自动处理，无需操作员干预。此外，通过向索引集群添加新机器来改善索引过程的性能也很容易。


## 7 相关工作

许多系统提供了受限的编程模型，并利用这些限制自动并行化计算。例如，关联函数可以在 $N$ 个处理器上以 $\log N$ 时间通过并行前缀计算 [6, 9, 13] 计算 $N$ 元素数组的所有前缀。MapReduce 可以被视为基于我们在大型现实世界计算中的经验对这些模型的一些简化和提炼。更重要的是，我们提供了一个可扩展到数千个处理器的容错实现。相比之下，大多数并行处理系统仅在较小规模上实现，并将处理机器故障的细节留给程序员。

Bulk Synchronous Programming [17] 和一些 MPI 原语 [11] 提供更高级的抽象，使程序员更容易编写并行程序。这些系统与 MapReduce 之间的一个关键区别在于，MapReduce 利用受限的编程模型自动并行化用户程序并提供透明的容错性。

我们的局部性优化借鉴了 active disks [12, 15] 等技术，其中计算被推到靠近本地磁盘的处理单元，以减少跨 I/O 子系统或网络发送的数据量。我们在直接连接少量磁盘的商用处理器上运行，而非直接在磁盘控制器处理器上运行，但总体方法相似。

我们的备份任务机制类似于 Charlotte 系统 [3] 中使用的 eager scheduling 机制。简单 eager scheduling 的一个缺点是，如果给定任务导致重复失败，整个计算将无法完成。我们通过跳过坏记录的机制修复了此问题的一些实例。

MapReduce 实现依赖于内部集群管理系统，该系统负责在大量共享机器上分发和运行用户任务。虽然不是本文的重点，但该集群管理系统在精神上类似于 Condor [16] 等其他系统。

作为 MapReduce 库一部分的排序设施在操作上类似于 NOW-Sort [1]。源机器（map worker）将要排序的数据分区并发送给 $R$ 个 reduce worker 之一。每个 reduce worker 在本地对其数据排序（可能时在内存中）。当然 NOW-Sort 没有使用户库广泛适用的用户可定义 Map 和 Reduce 函数。

River [2] 提供一种编程模型，其中进程通过分布式队列发送数据来相互通信。与 MapReduce 类似，River 系统试图在存在由异构硬件或系统扰动导致的非均匀性时提供良好的平均情况性能。River 通过精细调度磁盘和网络传输以实现均衡的完成时间来实现这一点。MapReduce 采用不同的方法。通过限制编程模型，MapReduce 框架能够将问题划分为大量细粒度任务。这些任务在可用 worker 上动态调度，以便更快的 worker 处理更多任务。受限的编程模型还允许我们在作业接近结束时调度任务的冗余执行，这在存在非均匀性（如慢或卡住的 worker）时大大减少了完成时间。

BAD-FS [5] 具有与 MapReduce 非常不同的编程模型，且与 MapReduce 不同，其目标是在广域网上执行作业。然而，两者有两个根本性的相似之处：（1）两个系统都使用冗余执行来从故障导致的数据丢失中恢复。（2）两者都使用局部性感知调度来减少跨拥塞网络链路发送的数据量。

TACC [7] 是一个旨在简化高可用网络服务构建的系统。与 MapReduce 类似，它依赖重新执行作为实现容错的机制。


## 8 结论

MapReduce 编程模型已在 Google 成功用于许多不同目的。我们将这一成功归因于几个原因。首先，该模型易于使用，即使对于没有并行和分布式系统经验的程序员也是如此，因为它隐藏了并行化、容错性、局部性优化和负载均衡的细节。其次，大量各类问题易于表达为 MapReduce 计算。例如，MapReduce 用于生成 Google 产品 Web 搜索服务的数据、排序、数据挖掘、机器学习和许多其他系统。第三，我们开发了一个可扩展到包含数千台机器的大规模集群的 MapReduce 实现。该实现有效地利用了这些机器资源，因此适用于 Google 遇到的许多大型计算问题。

我们从这项工作中学到了几件事。首先，限制编程模型使得并行化和分布计算以及使此类计算具备容错性变得容易。其次，网络带宽是稀缺资源。因此，我们系统中的许多优化旨在减少通过网络发送的数据量：局部性优化允许我们从本地磁盘读取数据，将中间数据的一份副本写入本地磁盘节省了网络带宽。第三，冗余执行可用于减少慢机器的影响，并处理机器故障和数据丢失。


## 致谢

Josh Levenberg 基于他使用 MapReduce 的经验和其他人的增强建议，在修改和扩展用户级 MapReduce API 的新功能方面发挥了重要作用。MapReduce 从 Google 文件系统 [8] 读取输入并写入输出。感谢 Mohit Aron、Howard Gobioff、Markus Gutschke、David Kramer、Shun-Tak Leung 和 Josh Redstone 在开发 GFS 方面的工作。还要感谢 Percy Liang 和 Olcan Sercinoglu 在开发 MapReduce 使用的集群管理系统方面的工作。Mike Burrows、Wilson Hsieh、Josh Levenberg、Sharon Perl、Rob Pike 和 Debby Wallach 对本文初稿提供了有益意见。匿名 OSDI 审稿人和我们的指导人 Eric Brewer 提供了许多关于论文可改进领域的宝贵建议。最后，感谢 Google 工程组织中所有 MapReduce 用户提供了有益的反馈、建议和 bug 报告。


## 参考文献

[1] Andrea C. Arpaci-Dusseau, Remzi H. Arpaci-Dusseau, David E. Culler, Joseph M. Hellerstein, and David A. Patterson. High-performance sorting on networks of workstations. In *Proceedings of the 1997 ACM SIGMOD International Conference on Management of Data*, Tucson, Arizona, May 1997.

[2] Remzi H. Arpaci-Dusseau, Eric Anderson, Noah Treuhaft, David E. Culler, Joseph M. Hellerstein, David Patterson, and Kathy Yelick. Cluster I/O with River: Making the fast case common. In *Proceedings of the Sixth Workshop on Input/Output in Parallel and Distributed Systems (IOPADS '99)*, pages 10–22, Atlanta, Georgia, May 1999.

[3] Arash Baratloo, Mehmet Karaul, Zvi Kedem, and Peter Wyckoff. Charlotte: Metacomputing on the web. In *Proceedings of the 9th International Conference on Parallel and Distributed Computing Systems*, 1996.

[4] Luiz A. Barroso, Jeffrey Dean, and Urs Hölzle. Web search for a planet: The Google cluster architecture. *IEEE Micro*, 23(2):22–28, April 2003.

[5] John Bent, Douglas Thain, Andrea C. Arpaci-Dusseau, Remzi H. Arpaci-Dusseau, and Miron Livny. Explicit control in a batch-aware distributed file system. In *Proceedings of the 1st USENIX Symposium on Networked Systems Design and Implementation NSDI*, March 2004.

[6] Guy E. Blelloch. Scans as primitive parallel operations. *IEEE Transactions on Computers*, C-38(11), November 1989.

[7] Armando Fox, Steven D. Gribble, Yatin Chawathe, Eric A. Brewer, and Paul Gauthier. Cluster-based scalable network services. In *Proceedings of the 16th ACM Symposium on Operating System Principles*, pages 78–91, Saint-Malo, France, 1997.

[8] Sanjay Ghemawat, Howard Gobioff, and Shun-Tak Leung. The Google file system. In *19th Symposium on Operating Systems Principles*, pages 29–43, Lake George, New York, 2003.

[9] S. Gorlatch. Systematic efficient parallelization of scan and other list homomorphisms. In L. Bouge, P. Fraigniaud, A. Mignotte, and Y. Robert, editors, *Euro-Par'96. Parallel Processing*, Lecture Notes in Computer Science 1124, pages 401–408. Springer-Verlag, 1996.

[10] Jim Gray. Sort benchmark home page. http://research.microsoft.com/barc/SortBenchmark/.

[11] William Gropp, Ewing Lusk, and Anthony Skjellum. *Using MPI: Portable Parallel Programming with the Message-Passing Interface*. MIT Press, Cambridge, MA, 1999.

[12] L. Huston, R. Sukthankar, R. Wickremesinghe, M. Satyanarayanan, G. R. Ganger, E. Riedel, and A. Ailamaki. Diamond: A storage architecture for early discard in interactive search. In *Proceedings of the 2004 USENIX File and Storage Technologies FAST Conference*, April 2004.

[13] Richard E. Ladner and Michael J. Fischer. Parallel prefix computation. *Journal of the ACM*, 27(4):831–838, 1980.

[14] Michael O. Rabin. Efficient dispersal of information for security, load balancing and fault tolerance. *Journal of the ACM*, 36(2):335–348, 1989.

[15] Erik Riedel, Christos Faloutsos, Garth A. Gibson, and David Nagle. Active disks for large-scale data processing. *IEEE Computer*, pages 68–74, June 2001.

[16] Douglas Thain, Todd Tannenbaum, and Miron Livny. Distributed computing in practice: The Condor experience. *Concurrency and Computation: Practice and Experience*, 2004.

[17] L. G. Valiant. A bridging model for parallel computation. *Communications of the ACM*, 33(8):103–111, 1997.

[18] Jim Wyllie. Spsort: How to sort a terabyte quickly. http://alme1.almaden.ibm.com/cs/spsort.pdf.


## 附录 A：词频统计

本节包含一个程序，统计命令行指定的输入文件集合中每个唯一单词的出现次数。

```cpp
#include "mapreduce/mapreduce.h"

// User's map function
class WordCounter : public Mapper {
 public:
  virtual void Map(const MapInput& input) {
    const string& text = input.value();
    const int n = text.size();
    for (int i = 0; i < n; ) {
      // Skip past leading whitespace
      while ((i < n) && isspace(text[i]))
        i++;
      // Find word end
      int start = i;
      while ((i < n) && !isspace(text[i]))
        i++;
      if (start < i)
        Emit(text.substr(start, i - start), "1");
    }
  }
};
REGISTER_MAPPER(WordCounter);

// User's reduce function
class Adder : public Reducer {
  virtual void Reduce(ReduceInput* input) {
    // Iterate over all entries with the
    // same key and add the values
    int64 value = 0;
    while (!input->done()) {
      value += StringToInt(input->value());
      input->NextValue();
    }
    // Emit sum for input->key()
    Emit(IntToString(value));
  }
};
REGISTER_REDUCER(Adder);

int main(int argc, char** argv) {
  ParseCommandLineFlags(argc, argv);

  MapReduceSpecification spec;

  // Store list of input files into "spec"
  for (int i = 1; i < argc; i++) {
    MapReduceInput* input = spec.add_input();
    input->set_format("text");
    input->set_filepattern(argv[i]);
    input->set_mapper_class("WordCounter");
  }

  // Specify the output files:
  //   /gfs/test/freq-00000-of-00100
  //   /gfs/test/freq-00001-of-00100
  //   ...
  MapReduceOutput* out = spec.output();
  out->set_filebase("/gfs/test/freq");
  out->set_num_tasks(100);
  out->set_format("text");
  out->set_reducer_class("Adder");

  // Optional: do partial sums within map
  // tasks to save network bandwidth
  out->set_combiner_class("Adder");

  // Tuning parameters: use at most 2000
  // machines and 100 MB of memory per task
  spec.set_machines(2000);
  spec.set_map_megabytes(100);
  spec.set_reduce_megabytes(100);

  // Now run it
  MapReduceResult result;
  if (!MapReduce(spec, &result)) abort();

  // Done: 'result' structure contains info
  // about counters, time taken, number of
  // machines used, etc.
  return 0;
}
```
