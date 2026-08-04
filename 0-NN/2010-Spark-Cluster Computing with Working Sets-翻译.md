# Spark：面向工作集的集群计算

> Matei Zaharia, Mosharaf Chowdhury, Michael J. Franklin, Scott Shenker, Ion Stoica | University of California, Berkeley

本文介绍了 Spark：面向工作集的集群计算。核心内容：

- 提出 Spark 框架，支持跨多个并行操作重用工作集数据的应用，包括迭代式机器学习算法和交互式数据分析工具
- 引入弹性分布式数据集（RDD）抽象——一种跨机器分区的只读对象集合，可在分区丢失时重建，实现容错和内存级性能
- 通过 RDD 的血统（lineage）概念实现容错：每个 RDD 包含足够的信息来从其派生来源重新计算丢失的分区

关键发现：

- Spark 在迭代式机器学习作业中比 Hadoop 快 10 倍
- 可在 15 台机器上交互式查询 39 GB 数据集，亚秒级响应时间
- 广播变量使 ALS 协同过滤性能提升 2.8 倍

---

## 摘要

MapReduce 及其变体在商用集群上实现大规模数据密集型应用方面取得了巨大成功。然而，这些系统大多建立在有向无环数据流模型之上，不适用于其他流行的应用。本文聚焦于其中一类应用：那些跨多个并行操作重用工作集数据的应用。这包括许多迭代式机器学习算法以及交互式数据分析工具。我们提出了一个名为 Spark 的新框架，它在保持 MapReduce 的可扩展性和容错性的同时支持这些应用。为实现这些目标，Spark 引入了一种称为弹性分布式数据集（RDD）的抽象。RDD 是一种跨机器分区的只读对象集合，可在分区丢失时重建。Spark 在迭代式机器学习作业中可比 Hadoop 快 10 倍，并可用于以亚秒级响应时间交互式查询 39 GB 的数据集。

## 1 引言

一种新的集群计算模型已变得广泛流行：数据并行计算在不可靠机器集群上执行，由自动提供局部性感知调度、容错和负载均衡的系统管理。MapReduce [11] 开创了这一模型，而 Dryad [17] 和 Map-Reduce-Merge [24] 等系统推广了所支持的数据流类型。

这些系统通过提供一个编程模型来实现可扩展性和容错性，即用户创建有向无环数据流图，将输入数据通过一组操作符传递。这使得底层系统能够管理调度并对故障做出反应，无需用户干预。

虽然这种数据流编程模型对大量应用有用，但有些应用无法高效地表达为有向无环数据流。本文聚焦于其中一类应用：那些跨多个并行操作重用工作集数据的应用。这包括两种我们看到 Hadoop 用户报告 MapReduce 存在不足的用例：

- **迭代式作业**：许多常见的机器学习算法对同一数据集重复应用函数以优化参数（如通过梯度下降）。虽然每次迭代可以表达为一个 MapReduce/Dryad 作业，但每个作业必须从磁盘重新加载数据，带来显著的性能损失。
- **交互式分析**：Hadoop 常通过 Pig [21] 和 Hive [1] 等 SQL 接口对大型数据集运行 ad-hoc 探索性查询。理想情况下，用户应该能够将感兴趣的数据集加载到多台机器的内存中并重复查询。然而，使用 Hadoop 时，每次查询会产生显著延迟（数十秒），因为它作为单独的 MapReduce 作业运行并从磁盘读取数据。

本文提出了一个新的集群计算框架 Spark，支持带有工作集的应用，同时提供与 MapReduce 类似的可扩展性和容错性。

Spark 中的主要抽象是弹性分布式数据集（RDD），它表示一组跨机器分区的只读对象集合，可在分区丢失时重建。用户可以显式地将 RDD 缓存在机器内存中，并在多个类似 MapReduce 的并行操作中重用它。RDD 通过一种称为血统（lineage）的概念实现容错：如果 RDD 的一个分区丢失，RDD 拥有足够的信息来了解它如何从其他 RDD 派生，从而仅重建该分区。虽然 RDD 不是通用共享内存抽象，但它在表达性（一方面）和可扩展性与可靠性（另一方面）之间代表了一个最佳平衡点，我们已发现它适用于多种应用。

Spark 用 Scala [5] 实现，这是一种用于 Java VM 的静态类型高级编程语言，并暴露了类似于 DryadLINQ [25] 的函数式编程接口。此外，Spark 可从 Scala 解释器的修改版本中交互式使用，允许用户定义 RDD、函数、变量和类，并在集群上的并行操作中使用它们。我们相信 Spark 是第一个允许将高效、通用编程语言交互式用于在集群上处理大数据集的系统。

虽然我们的 Spark 实现仍是一个原型，但早期经验令人鼓舞。我们展示 Spark 在迭代式机器学习工作负载上可比 Hadoop 快 10 倍，并可用于交互式扫描 39 GB 数据集，亚秒级延迟。

## 2 编程模型

要使用 Spark，开发者编写一个实现应用高层控制流并启动各种并行操作的驱动程序。Spark 为并行编程提供两种主要抽象：弹性分布式数据集以及在这些数据集上的并行操作（通过传递函数应用于数据集来调用）。此外，Spark 支持两种受限类型的共享变量，可在集群上运行的函数中使用。

### 2.1 弹性分布式数据集（RDD）

弹性分布式数据集（RDD）是一组跨机器分区的只读对象集合，可在分区丢失时重建。RDD 的元素不需要存在于物理存储中；相反，RDD 的句柄包含足够的信息来从可靠存储中的数据计算 RDD。这意味着如果节点失败，RDD 总是可以重建。

在 Spark 中，每个 RDD 由一个 Scala 对象表示。Spark 允许程序员通过四种方式构造 RDD：

- 来自共享文件系统中的文件，如 Hadoop 分布式文件系统（HDFS）。
- 通过在驱动程序中"并行化"一个 Scala 集合（如数组），即将其分为若干分片，发送到多个节点。
- 通过转换现有 RDD。类型为 $A$ 的元素的数据集可以使用名为 `flatMap` 的操作转换为类型为 $B$ 的数据集，该操作将每个元素传递给用户提供的类型为 $A \Rightarrow \text{List}[B]$ 的函数。其他转换可以用 `flatMap` 表达，包括 `map`（将元素通过类型为 $A \Rightarrow B$ 的函数）和 `filter`（选取匹配谓词的元素）。
- 通过更改现有 RDD 的持久性。默认情况下，RDD 是惰性和临时的。也就是说，数据集的分区在并行操作中使用时按需物化，使用后即从内存中丢弃。然而，用户可以通过两个操作改变 RDD 的持久性：
  - `cache` 操作保持数据集的惰性，但提示在首次计算后应将其保留在内存中，因为它将被重用。
  - `save` 操作计算数据集并将其写入分布式文件系统（如 HDFS）。在后续操作中使用保存的版本。

我们注意到 `cache` 操作只是一个提示：如果集群中没有足够的内存来缓存数据集的所有分区，Spark 会在它们被使用时重新计算。我们选择这种设计是为了 Spark 程序在节点失败或数据集过大时仍能工作（性能有所降低）。这个想法大致类似于虚拟内存。我们还计划扩展 Spark 以支持其他级别的持久性（如跨多个节点的内存内复制）。我们的目标是让用户在存储 RDD 的成本、访问速度、丢失部分 RDD 的概率和重新计算成本之间进行权衡。

### 2.2 并行操作

可以对 RDD 执行以下几种并行操作：

- `reduce`：使用关联函数组合数据集元素，在驱动程序中产生结果。
- `collect`：将数据集的所有元素发送到驱动程序。例如，并行更新数组的一种简单方法是并行化、map 和 collect 该数组。
- `foreach`：将每个元素传递给用户提供的函数。仅用于函数的副作用（可能将数据复制到另一个系统或更新共享变量）。

我们注意到 Spark 目前不支持像 MapReduce 中的分组 reduce 操作；reduce 结果只在一个进程（驱动程序）中收集。不过，即使使用单个 reducer 也足以表达多种有用的算法。例如，一篇关于多核系统上机器学习的 MapReduce 论文 [10] 在不支持并行归约的情况下实现了十种学习算法。

### 2.3 共享变量

程序员通过向 Spark 传递闭包（函数）来调用 `map`、`filter` 和 `reduce` 等操作。如同函数式编程中的典型做法，这些闭包可以引用它们创建时所在作用域中的变量。通常，当 Spark 在 worker 节点上运行闭包时，这些变量会被复制到 worker。然而，Spark 也允许程序员创建两种受限类型的共享变量，以支持两种简单但常见的使用模式：

- **广播变量**：如果一块大的只读数据（如查找表）在多个并行操作中使用，最好只将其分发到 worker 一次，而不是随每个闭包打包发送。Spark 允许程序员创建一个包装该值的"广播变量"对象，确保它只被复制到每个 worker 一次。
- **累加器**：这些是 worker 只能使用关联操作"添加"、且只有驱动程序能读取的变量。它们可用于实现 MapReduce 中的计数器，并为并行求和提供更命令式的语法。累加器可以为任何具有"添加"操作和"零"值的类型定义。由于其"只增"语义，它们易于实现容错。

## 3 示例

我们现在展示一些 Spark 程序示例。注意我们省略了变量类型，因为 Scala 支持类型推断。

### 3.1 文本搜索

假设我们希望统计存储在 HDFS 中的大型日志文件中包含错误的行数。可以通过如下方式以文件数据集对象实现：

```scala
val file = spark.textFile("hdfs://...")
val errs = file.filter(_.contains("ERROR"))
val ones = errs.map(_ => 1)
val count = ones.reduce(_+_)
```

我们首先创建一个名为 `file` 的分布式数据集，将 HDFS 文件表示为行的集合。我们转换此数据集以创建包含"ERROR"的行集合（`errs`），然后将每行映射为 1，并使用 `reduce` 将这些 1 相加。`filter`、`map` 和 `reduce` 的参数是 Scala 中函数字面量的语法。

注意 `errs` 和 `ones` 是惰性 RDD，永远不会被物化。相反，当调用 `reduce` 时，每个 worker 节点以流式方式扫描输入块以计算 `ones`，相加执行本地 reduce，并将其本地计数发送给驱动程序。以这种方式与惰性数据集结合使用时，Spark 紧密模拟了 MapReduce。

Spark 与其他框架的不同之处在于，它可以使某些中间数据集跨操作持久化。例如，如果想重用 `errs` 数据集，我们可以从中创建一个缓存的 RDD，如下所示：

```scala
val cachedErrs = errs.cache()
```

我们现在可以像往常一样在 `cachedErrs` 或其派生的数据集上调用并行操作，但节点在首次计算后将 `cachedErrs` 的分区缓存在内存中，大大加快了后续操作的速度。

### 3.2 逻辑回归

以下程序实现了逻辑回归 [3]，一种迭代式分类算法，试图找到最佳分离两组点的超平面 $w$ 。该算法执行梯度下降：将 $w$ 初始化为随机值，每次迭代对数据求和 $w$ 的函数，以向改进方向移动 $w$ 。因此，它在跨迭代的内存中缓存数据方面获益巨大。我们不详细解释逻辑回归，但用它来展示一些新的 Spark 特性。

```scala
// Read points from a text file and cache them
val points = spark.textFile(...)
  .map(parsePoint).cache()

// Initialize w to random D-dimensional vector
var w = Vector.random(D)

// Run multiple iterations to update w
for (i <- 1 to ITERATIONS) {
  val grad = spark.accumulator(new Vector(D))
  for (p <- points) { // Runs in parallel
    val s = (1/(1+exp(-p.y*(w dot p.x)))-1)*p.y
    grad += s * p.x
  }
  w -= grad.value
}
```

首先，虽然我们创建了一个名为 `points` 的 RDD，但我们通过在其上运行 `for` 循环来处理它。Scala 中的 `for` 关键字是调用集合的 `foreach` 方法（将循环体作为闭包）的语法糖。即，代码 `for(p <- points){body}` 等价于 `points.foreach(p => {body})`。因此，我们正在调用 Spark 的并行 `foreach` 操作。

其次，为求和梯度，我们使用了一个名为 `gradient` 的累加器变量（类型为 `Vector`）。注意循环使用重载的 `+=` 运算符向 `gradient` 添加。累加器和 `for` 语法的结合使得 Spark 程序看起来很像命令式串行程序。事实上，此示例与逻辑回归的串行版本仅有三行不同。

### 3.3 交替最小二乘法

最后一个示例是称为交替最小二乘法（ALS）的算法。ALS 用于协同过滤问题，如根据用户的电影评分历史预测其对未看过的电影的评分（如 Netflix 挑战赛）。与之前的示例不同，ALS 是计算密集型而非数据密集型。

我们简要介绍 ALS，详情请读者参考 [27]。假设我们要预测 $u$ 个用户对 $m$ 部电影的评分，并且我们有一个部分填充的矩阵 $R$ ，包含某些用户-电影对的已知评分。ALS 将 $R$ 建模为两个矩阵 $M$ （维度 $m \times k$ ）和 $U$ （维度 $k \times u$ ）的乘积；即每个用户和每部电影有一个 $k$ 维的"特征向量"描述其特征，用户对电影的评分是其特征向量与电影特征向量的点积。ALS 使用已知评分求解 $M$ 和 $U$ ，然后计算 $M \times U$ 来预测未知评分。通过以下迭代过程完成：

1. 将 $M$ 初始化为随机值。
2. 在给定 $M$ 的情况下优化 $U$ 以最小化 $R$ 上的误差。
3. 在给定 $U$ 的情况下优化 $M$ 以最小化 $R$ 上的误差。
4. 重复步骤 2 和 3 直到收敛。

ALS 可以通过在步骤 2 和 3 中在不同节点上更新不同用户/电影来并行化。然而，由于所有步骤都使用 $R$ ，将 $R$ 设为广播变量是有帮助的，这样它在每一步不会被重新发送到每个节点。一个 Spark 的 ALS 实现如下所示。注意我们并行化了集合 `0 until u`（一个 Scala 范围对象）并收集它以更新每个数组：

```scala
val Rb = spark.broadcast(R)
for (i <- 1 to ITERATIONS) {
  U = spark.parallelize(0 until u)
    .map(j => updateUser(j, Rb, M))
    .collect()
  M = spark.parallelize(0 until m)
    .map(j => updateUser(j, Rb, U))
    .collect()
}
```

## 4 实现

Spark 构建在 Mesos [16, 15] 之上，后者是一个"集群操作系统"，允许以细粒度方式在集群上共享多个并行应用，并提供 API 供应用在集群上启动任务。这使得 Spark 可以与现有集群计算框架（如 Hadoop 和 MPI 的 Mesos 端口）并行运行，并与它们共享数据。此外，构建在 Mesos 之上大大减少了 Spark 所需的编程工作。

Spark 的核心是弹性分布式数据集的实现。例如，假设我们定义一个名为 `cachedErrs` 的缓存数据集，表示日志文件中的错误消息，并使用 `map` 和 `reduce` 计算其元素数量，如第 3.1 节所示：

```scala
val file = spark.textFile("hdfs://...")
val errs = file.filter(_.contains("ERROR"))
val cachedErrs = errs.cache()
val ones = cachedErrs.map(_ => 1)
val count = ones.reduce(_+_)
```

这些数据集将存储为一个对象链，捕获每个 RDD 的谱系，如图 1 所示。每个数据集对象包含一个指向其父对象的指针和关于父对象如何被转换的信息。

在内部，每个 RDD 对象实现了相同的简单接口，包含三个操作：

- `getPartitions`，返回分区 ID 列表。
- `getIterator(partition)`，迭代分区。
- `getPreferredLocations(partition)`，用于任务调度以实现数据局部性。

当在数据集上调用并行操作时，Spark 创建任务来处理数据集的每个分区，并将这些任务发送到 worker 节点。我们尝试使用延迟调度 [26] 将任务发送到其首选位置之一。一旦在 worker 上启动，每个任务调用 `getIterator` 开始读取其分区。

不同类型的 RDD 仅在如何实现 RDD 接口方面有所不同。例如，对于 `HdfsTextFile`，分区是 HDFS 中的块 ID，其首选位置是块位置，`getIterator` 打开一个流来读取块。在 `MappedDataset` 中，分区和首选位置与父对象相同，但迭代器对父对象的元素应用 map 函数。最后，在 `CachedDataset` 中，`getIterator` 方法查找分区的本地缓存副本，每个分区的首选位置最初等于父对象的首选位置，但在分区被缓存在某个节点后更新为优先重用该节点。

这种设计使故障易于处理：如果节点失败，其分区从父数据集重新读取，并最终缓存在其他节点上。

最后，将任务发送到 worker 需要将闭包发送给它们——既包括用于定义分布式数据集的闭包，也包括传递给 `reduce` 等操作的闭包。为实现这一点，我们利用 Scala 闭包是 Java 对象且可以使用 Java 序列化进行序列化的事实；这是 Scala 的一个特性，使得将计算发送到另一台机器相对简单。然而，Scala 内置的闭包实现并不理想，因为我们发现有些情况下闭包对象引用了闭包外部作用域中在闭包体内未实际使用的变量。我们已经报告了这个问题，但与此同时，我们通过对闭包类的字节码执行静态分析来检测这些未使用的变量，并将闭包对象中的相应字段设为 `null`，从而解决了该问题。由于篇幅限制，我们省略了此分析的细节。

**共享变量**：Spark 中的两种类型共享变量——广播变量和累加器——使用具有自定义序列化格式的类实现。当创建一个携带值 $v$ 的广播变量 $b$ 时， $v$ 被保存到共享文件系统中的一个文件。 $b$ 的序列化形式是指向该文件的路径。当在 worker 节点上查询 $b$ 的值时，Spark 首先检查 $v$ 是否在本地缓存中，如果没有则从文件系统读取。我们最初使用 HDFS 来广播变量，但我们正在开发一个更高效的流式广播系统。

累加器使用不同的"序列化技巧"实现。每个累加器在创建时被赋予一个唯一的 ID。当累加器被保存时，其序列化形式包含其 ID 和该类型的"零"值。在 worker 上，每个运行任务的线程使用线程局部变量为累加器创建一个单独的副本，并在任务开始时重置为零。每个任务运行后，worker 向驱动程序发送一条消息，包含它对各种累加器所做的更新。驱动程序对每次操作的每个分区仅应用一次更新，以防止在任务因故障重新执行时重复计数。

**解释器集成**：由于篇幅限制，我们仅简要介绍如何将 Spark 集成到 Scala 解释器中。Scala 解释器通常通过为每行用户输入编译一个类来操作。该类包含一个单例对象，包含该行上的变量或函数，并在其构造函数中运行该行的代码。例如，如果用户输入 `var x = 5` 后跟 `println(x)`，解释器定义一个包含 $x$ 的类（如 `Line1`），并导致第二行编译为 `println(Line1.getInstance().x)`。这些类被加载到 JVM 中以运行每行。为使解释器与 Spark 一起工作，我们做了两项更改：

1. 我们让解释器将其定义的类输出到共享文件系统，worker 可以通过自定义 Java 类加载器从那里加载它们。
2. 我们更改了生成的代码，使得每行的单例对象直接引用前几行的单例对象，而不是通过静态的 `getInstance` 方法。这允许闭包在序列化发送到 worker 时捕获它们引用的单例的当前状态。如果不这样做，对单例对象的更新（例如，在上例中设置 `x = 7` 的行）将不会传播到 worker。

## 5 结果

虽然我们的 Spark 实现仍处于早期阶段，但我们报告了三个实验的结果，展示了它作为集群计算框架的前景。

**逻辑回归**：我们比较了第 3.2 节中逻辑回归作业与 Hadoop 逻辑回归实现的性能，使用 20 个具有 4 核的"m1.xlarge"EC2 节点上的 29 GB 数据集。结果如图 2 所示。使用 Hadoop 时，每次迭代耗时 127 秒，因为它作为独立的 MapReduce 作业运行。使用 Spark 时，第一次迭代耗时 174 秒（可能是由于使用 Scala 而非 Java），但后续每次迭代仅需 6 秒，因为它们重用了缓存数据。这使得作业运行速度提升高达 10 倍。

我们还尝试在作业运行时崩溃一个节点。在 10 次迭代的情况下，这使作业平均减慢 50 秒（21%）。丢失节点上的数据分区被重新计算并并行缓存在其他节点上，但当前实验中的恢复时间相当高，因为我们使用了较大的 HDFS 块大小（128 MB），因此每个节点只有 12 个块，恢复过程无法利用集群中的所有核心。较小的块大小将产生更快的恢复时间。

**交替最小二乘法**：我们实现了第 3.3 节中的交替最小二乘法作业，以衡量广播变量对将共享数据集复制到多个节点的迭代式作业的好处。我们发现不使用广播变量时，每次迭代重新发送评分矩阵 $R$ 的时间主导了作业的运行时间。此外，使用朴素的广播实现（使用 HDFS 或 NFS）时，广播时间随节点数量线性增长，限制了作业的可扩展性。我们实现了一个应用层多播系统来缓解这一问题。然而，即使使用快速广播，每次迭代重新发送 $R$ 也是昂贵的。在包含 5000 部电影和 15000 个用户的 30 节点 EC2 集群上的实验中，使用广播变量将 $R$ 缓存在 worker 内存中将性能提升了 2.8 倍。

**交互式 Spark**：我们使用 Spark 解释器将 39 GB 的 Wikipedia 转储加载到 15 个"m1.xlarge"EC2 机器的内存中，并进行交互式查询。首次查询数据集时，大约需要 35 秒，与运行 Hadoop 作业相当。然而，后续查询仅需 0.5 到 1 秒，即使它们扫描所有数据。这提供了质的不同的体验，类似于处理本地数据。

## 6 相关工作

**分布式共享内存**：Spark 的弹性分布式数据集可被视为分布式共享内存（DSM）的一种抽象，后者已被广泛研究 [20]。RDD 与 DSM 接口在两个方ä¸不同。首先，RDD 提供了更受限的编程模型，但使数据集能在集群节点失败时高效重建。虽然某些 DSM 系统通过检查点实现容错 [18]，Spark 使用 RDD 对象中捕获的血统信息重建丢失的 RDD 分区。这意味着只需重新计算丢失的分区，并且可以在不同节点上并行重新计算，无需程序回滚到检查点。此外，如果没有节点失败，则没有开销。其次，RDD 将计算推送到数据所在位置（如 MapReduce [11]），而不是让任意节点访问全局地址空间。

其他系统也限制了 DSM 编程模型以提高性能、可靠性和可编程性。Munin [8] 允许程序员用访问模式注释变量，以便为它们选择最佳一致性协议。Linda [13] 提供了一个元组空间编程模型，可以以容错方式实现。Thor [19] 提供了持久化共享对象的接口。

**集群计算框架**：Spark 的并行操作符合 MapReduce 模型 [11]。然而，它们操作于可跨操作持久化的 RDD。扩展 MapReduce 以支持迭代式作业的需求也被 Twister [6, 12] 所认识，这是一个允许长时间运行的 map 任务在作业间保留静态数据在内存中的 MapReduce 框架。然而，Twister 目前不实现容错。Spark 的弹性分布式数据集抽象既具备容错性，又比迭代式 MapReduce 更通用。Spark 程序可以定义多个 RDD 并在它们之间交替运行操作，而 Twister 程序只有一个 map 函数和一个 reduce 函数。这也使得 Spark 对交互式数据分析很有用，用户可以定义多个数据集然后查询它们。Spark 的广播变量提供类似于 Hadoop 的分布式缓存 [2] 的功能，后者可以将文件分发到运行特定作业的所有节点。然而，广播变量可以跨并行操作重用。

**语言集成**：Spark 的语言集成类似于 DryadLINQ [25]，后者使用 .NET 对语言集成查询的支持来捕获定义查询的表达式树并在集群上运行。与 DryadLINQ 不同，Spark 允许 RDD 跨并行操作持久化在内存中。此外，Spark 通过支持共享变量（广播变量和累加器）丰富了语言集成模型，这些变量使用具有自定义序列化形式的类实现。

我们受 SMR [14]（一个使用闭包定义 map 和 reduce 任务的 Hadoop 的 Scala 接口）的启发而使用 Scala 进行语言集成。我们在 SMR 上的贡献是共享变量和更稳健的闭包序列化实现。

最后，IPython [22] 是为科学家设计的 Python 解释器，允许用户使用容错任务队列接口或低级消息传递接口在集群上启动计算。Spark 提供了类似的交互式接口，但专注于数据密集型计算。

**血统**：捕获数据集的谱系或来源信息长期以来一直是科学计算和数据库领域的研究课题，用于解释结果、允许他人重现结果以及在发现工作流步骤中的 bug 或数据集丢失时重新计算数据等应用。我们请读者参考 [7]、[23] 和 [9] 来了解这些工作的综述。Spark 提供了一种受限的并行编程模型，其中细粒度的谱系捕获成本低廉，因此该信息可用于重新计算丢失的数据集元素。

## 7 讨论和未来工作

Spark 提供了三种简单的数据抽象用于编程集群：弹性分布式数据集（RDD）和两种受限类型的共享变量——广播变量和累加器。虽然这些抽象是有限的，但我们发现它们足以表达对现有集群计算框架构成挑战的多种应用，包括迭代式和交互式计算。此外，我们相信 RDD 背后的核心理念——数据集句柄包含足够的信息从可靠存储中的数据（重）建数据集——可能在开发其他编程集群的抽象方面被证明是有用的。

在未来的工作中，我们计划聚焦于四个方面：

1. 形式化刻画 RDD 和 Spark 其他抽象的特性，以及它们对各种应用和工作负载的适用性。
2. 增强 RDD 抽象，允许程序员在存储成本和重建成本之间进行权衡。
3. 定义新的 RDD 转换操作，包括按给定键重新分区 RDD 的"shuffle"操作。这样的操作将允许我们实现 group-by 和连接操作。
4. 在 Spark 解释器之上提供更高级的交互式接口，如 SQL 和 R [4] shell。

## 致谢

感谢 Ali Ghodsi 对本文的反馈。本研究得到了 California MICRO、California Discovery、加拿大自然科学与工程研究理事会以及以下 Berkeley RAD Lab 赞助商的支持：Sun Microsystems、Google、Microsoft、Amazon、Cisco、Cloudera、eBay、Facebook、Fujitsu、HP、Intel、NetApp、SAP、VMware 和 Yahoo!。

## 参考文献

[1] Apache Hive. http://hadoop.apache.org/hive.

[2] Hadoop Map/Reduce tutorial. http://hadoop.apache.org/common/docs/r0.20.0/mapred_tutorial.html.

[3] Logistic regression – Wikipedia. http://en.wikipedia.org/wiki/Logistic_regression.

[4] The R project for statistical computing. http://www.r-project.org.

[5] Scala programming language. http://www.scala-lang.org.

[6] Twister: Iterative MapReduce. http://iterativemapreduce.org.

[7] R. Bose and J. Frew. Lineage retrieval for scientific data processing: a survey. *ACM Computing Surveys*, 37:1–28, 2005.

[8] J. B. Carter, J. K. Bennett, and W. Zwaenepoel. Implementation and performance of Munin. In *SOSP '91*. ACM, 1991.

[9] J. Cheney, L. Chiticariu, and W.-C. Tan. Provenance in databases: Why, how, and where. *Foundations and Trends in Databases*, 1(4):379–474, 2009.

[10] C. T. Chu, S. K. Kim, Y. A. Lin, Y. Yu, G. R. Bradski, A. Y. Ng, and K. Olukotun. Map-reduce for machine learning on multicore. In *NIPS '06*, pages 281–288. MIT Press, 2006.

[11] J. Dean and S. Ghemawat. MapReduce: Simplified data processing on large clusters. *Commun. ACM*, 51(1):107–113, 2008.

[12] J. Ekanayake, S. Pallickara, and G. Fox. MapReduce for data intensive scientific analyses. In *ESCIENCE '08*, pages 277–284, Washington, DC, USA, 2008. IEEE Computer Society.

[13] D. Gelernter. Generative communication in linda. *ACM Trans. Program. Lang. Syst.*, 7(1):80–112, 1985.

[14] D. Hall. A scalable language, and a scalable framework. http://www.scala-blogs.org/2008/09/scalable-language-and-scalable.html.

[15] B. Hindman, A. Konwinski, M. Zaharia, A. Ghodsi, A. D. Joseph, R. H. Katz, S. Shenker, and I. Stoica. Mesos: A platform for fine-grained resource sharing in the data center. Technical Report UCB/EECS-2010-87, EECS Department, University of California, Berkeley, May 2010.

[16] B. Hindman, A. Konwinski, M. Zaharia, and I. Stoica. A common substrate for cluster computing. In *Workshop on Hot Topics in Cloud Computing (HotCloud) 2009*, 2009.

[17] M. Isard, M. Budiu, Y. Yu, A. Birrell, and D. Fetterly. Dryad: Distributed data-parallel programs from sequential building blocks. In *EuroSys 2007*, pages 59–72, 2007.

[18] A.-M. Kermarrec, G. Cabillic, A. Gefflaut, C. Morin, and I. Puaut. A recoverable distributed shared memory integrating coherence and recoverability. In *FTCS '95*. IEEE Computer Society, 1995.

[19] B. Liskov, A. Adya, M. Castro, S. Ghemawat, R. Gruber, U. Maheshwari, A. C. Myers, M. Day, and L. Shrira. Safe and efficient sharing of persistent objects in thor. In *SIGMOD '96*, pages 318–329. ACM, 1996.

[20] B. Nitzberg and V. Lo. Distributed shared memory: a survey of issues and algorithms. *Computer*, 24(8):52–60, aug 1991.

[21] C. Olston, B. Reed, U. Srivastava, R. Kumar, and A. Tomkins. Pig latin: a not-so-foreign language for data processing. In *SIGMOD '08*. ACM, 2008.

[22] F. Pérez and B. E. Granger. IPython: a system for interactive scientific computing. *Comput. Sci. Eng.*, 9(3):21–29, May 2007.

[23] Y. L. Simmhan, B. Plale, and D. Gannon. A survey of data provenance in e-science. *SIGMOD Rec.*, 34(3):31–36, 2005.

[24] H.-c. Yang, A. Dasdan, R.-L. Hsiao, and D. S. Parker. Map-reduce-merge: simplified relational data processing on large clusters. In *SIGMOD '07*, pages 1029–1040. ACM, 2007.

[25] Y. Yu, M. Isard, D. Fetterly, M. Budiu, Ú. Erlingsson, P. K. Gunda, and J. Currey. DryadLINQ: A system for general-purpose distributed data-parallel computing using a high-level language. In *OSDI '08*, San Diego, CA, 2008.

[26] M. Zaharia, D. Borthakur, J. Sen Sarma, K. Elmeleegy, S. Shenker, and I. Stoica. Delay scheduling: A simple technique for achieving locality and fairness in cluster scheduling. In *EuroSys 2010*, April 2010.

[27] Y. Zhou, D. Wilkinson, R. Schreiber, and R. Pan. Large-scale parallel collaborative filtering for the Netflix prize. In *AAIM '08*, pages 337–348, Berlin, Heidelberg, 2008. Springer-Verlag.
