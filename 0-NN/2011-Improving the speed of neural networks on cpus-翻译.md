# Improving the speed of neural networks on CPUs

> Vincent Vanhoucke | Google, Inc.
> Andrew Senior | Google, Inc.
> Mark Z. Mao | Google, Inc.

在 CPU 上提升神经网络的速度

深度学习的近期进展使得使用具有数千万参数的大型深度神经网络成为可能，适用于许多需要实时处理的应用。这些网络的庞大规模即使在现代 CPU 上也可能构成严峻的计算负担。因此，GPU 被常规地用于训练和运行此类网络。本文是一篇面向学生和研究人员的教程，介绍一些能够在现代 x86 CPU 上显著降低这种计算成本的技术。我们强调数据布局、计算批处理、SSE2 指令的使用，并特别利用 SSSE3 和 SSE4 定点指令，相比优化的浮点基线实现了 3 倍的提升。我们以语音识别为例，展示了一个实时混合隐马尔可夫模型/神经网络（HMM/NN）大词汇量系统，相比未优化的基线实现了 10 倍加速，相比激进优化的浮点基线实现了 4 倍加速，且精度无损。所述技术可直接推广到神经网络训练，并提供了专用硬件的有效替代方案。

- 利用定点算术和 SSSE3/SSE4 指令，相比优化 BLAS 包实现 3 倍加速
- 通过数据布局优化、循环展开、并行累加器和 SIMD 指令提升性能
- 8 位量化和定点实现使网络内存占用减少 3-4 倍
- 批处理和惰性评估进一步加速语音识别任务
- 端到端语音识别系统在精度无损前提下实现 3 倍加速

---

## 摘要

Recent advances in deep learning have made the use of large, deep neural networks with tens of millions of parameters suitable for a number of applications that require real-time processing. The sheer size of these networks can represent a challenging computational burden, even for modern CPUs. For this reason, GPUs are routinely used instead to train and run such networks. This paper is a tutorial for students and researchers on some of the techniques that can be used to reduce this computational cost considerably on modern x86 CPUs. We emphasize data layout, batching of the computation, the use of SSE2 instructions, and particularly leverage SSSE3 and SSE4 fixed-point instructions which provide a 3\\times improvement over an optimized floating-point baseline. We use speech recognition as an example task, and show that a real-time hybrid hidden Markov model / neural network (HMM/NN) large vocabulary system can be built with a 10\\times speedup over an unoptimized baseline and a 4\\times speedup over an aggressively optimized floating-point baseline at no cost in accuracy. The techniques described extend readily to neural network training and provide an effective alternative to the use of specialized hardware.

---

## 1 引言

神经网络近期重新引起兴趣，在一定程度上归功于廉价且强大的 GPU 的普及，这些 GPU 常将大型矩阵计算等常见操作加速 5 到 50 倍 [1-3]。这使得研究人员能够利用神经网络、自编码器或深度置信网络 [4-6] 解决更大、更困难的机器学习任务。然而，由于成本、组件可靠性和编程复杂性等多种因素，GPU 在计算集群中仍然是例外而非常态。问题因此变为：是投资 GPU 资源，还是让传统 CPU 运行得足够快，使得通过分布式计算，它们能够产生相似或更优的可扩展性和性能。本文的目的并非解决这一争论，而是向神经网络研究人员介绍一些工具，这些工具能够以易于访问的形式显著提升 Intel 和 AMD CPU 上神经网络的性能。其中一些技术对于精通高性能计算的研究人员来说可能并不新颖，但它们为超越现有优化 BLAS 包所能获得的改进奠定了基础。我们将特别展示如何利用定点算术和 SSSE3/SSE4 指令，以 3 倍于优化 BLAS 包的速度运行。

为了说明这一论点，我们将使用一个在移动语音输入的语音识别任务中达到最优性能的神经网络。基准测试的基本设置在第 2 节中详细描述，完整的语音识别评估系统在第 6 节中描述。第 3 节将涵盖数据布局和浮点 SIMD 操作的基础知识。第 4 节描述某些矩阵运算的定点实现。第 5 节介绍更具体适用于语音识别任务的进一步增强。我们将在第 6 节中给出端到端系统的性能结果并得出结论。

## 2 基准测试设置

在以下内容中，我们将使用一个具有 5 层的神经网络作为基准。输入层有 440 个输入，由 11 个连续、重叠的 25 ms 语音帧组成，每 10 ms 采样一次。每帧包含从梅尔频率尺度上的滤波器组提取的 40 个对数能量。除最后一层外，每层使用 sigmoid 作为非线性激活函数。每个中间层由 2000 个节点组成，最终的 softmax 层有 7969 个输出，对应于 HMM/NN 语音识别器中上下文相关声学状态的对数后验概率。解码期间 softmax 输出不进行归一化。HMM/NN 系统的细节可在 [7] 和第 6 节中找到。

评估设置包括在 Intel Xeon DP Quad Core E5640 机器上（运行 Ubuntu 操作系统），在单个 CPU 上通过网络运行 100 帧数据。CPU 缩放已禁用，每次运行至少执行 5 次并取平均值。结果摘要见表 1。对于与惰性评估（†）相关的结果，表 1 报告了假设实际只需要计算 30% 神经网络输出时的性能（详见第 5 节）。

表 1：结果摘要

| 优化方法 | 节号 | 处理 1 秒语音的时间 | 增量加速比 |
|---|---|---|---|
| 浮点基线 | 2 | 3.89 s | |
| 手工调优浮点 | 3.2 | 3.09 s | 26% |
| 浮点 SSE2 | 3.5 | 1.36 s | 127% |
| 浮点 GPU | 3.7 | 0.02 至 0.49 s | |
| 8 位量化 | 4.1 | 1.52 s | -12% |
| 整数 SSSE3 | 4.2 | 0.51 s | 198% |
| 整数 SSE4 | 4.3 | 0.47 s | 9% |
| 批处理 | 5.1 | 0.36 s | 30% |
| 惰性评估 † | 5.2 | ~0.26 s | 27% |
| 批处理惰性评估 † | 5.3 | ~0.21 s | 19% |

## 3 浮点实现

### 3.1 内存局部性

高性能计算最基本的原则是，一旦你访问某个内存地址，该地址附近的连续内存就会被加载到处理器芯片上的各级缓存中。这使得附近的数据对 CPU 的访问速度远快于从内存中获取。最直接的后果是，应当力求任何数值计算的最内层循环遍历连续内存。例如，考虑矩阵 $A = [a_{i,j}]$ 和 $B = [b_{k,l}]$ 的乘法。矩阵 $C = AB = [c_{i,l}]$ 的每个条目为：

$$
c_{i,l} = \sum_m a_{i,m} b_{m,l} \qquad (1)
$$

由于循环变量 $m$ 遍历 $A$ 的列和 $B$ 的行，$A$ 最好按行主序存储，而 $B$ 最好按列主序存储。在考虑单指令多数据（SIMD）指令时，这一点更为重要。

### 3.2 循环展开与并行累加器

有几种众所周知的方法可以改进朴素实现的矩阵乘法。

矩阵乘法的最内层循环是乘加操作：`c += a[i]*b[i]`。为减少检查循环终止的开销，可以部分展开计算，一次累加多个元素：

```c
c += a[i]*b[i] + a[i+1]*b[i+1] + a[i+2]*b[i+2] + a[i+3]*b[i+3]
```

第二种技术是并行使用多个累加器，这给编译器更多自由来流水线操作并在浮点单元间分配操作：

```c
c0 += a[i]*b[i];
c1 += a[i+1]*b[i+1];
c2 += a[i+2]*b[i+2];
c3 += a[i+3]*b[i+3];
c  = c0 + c1 + c2 + c3;
```

在我们的示例基准测试中（表 1），以 8 为块展开循环并使用 4 个并行累加器带来了 26% 的速度提升。

### 3.3 SIMD

SIMD 指令是 CPU 上低级并行化的基本构建块。这些指令对连续数据并行执行多个操作，使得数据局部性问题更加关键。在 x86 家族的 Intel 和 AMD CPU 上，它们通常一次处理 16 字节的数据：2 个 double、4 个 float、8 个 short 或 16 个 char。定义了多种数据类型来表示它们：`__m128i`、`__m128` 和 `__m128d`：

```
__m128i    ...128 位 / 16 个 char ...
__m128i    short  short  short  short  short  short  short  short
__m128i    int    int    int    int
__m128     float  float  float  float
__m128d    double double
```

这些数据类型的基本操作使用汇编指令，但通过称为内建函数（intrinsics）[8] 的轻量封装函数更容易地集成到 C 和 C++ 中，这些函数可用于最流行的 C 和 C++ 编译器。例如，如果有两个存储了 4 个 float 的向量 `a = [a1, a2, a3, a4]` 和 `b = [b1, b2, b3, b4]`，类型为 `__m128`，可以通过以下 C 代码得到 `c = [a1+b1, a2+b2, a3+b3, a4+b4]`：

```c
#include <mmintrin.h>
__m128 c = _mm_add_ps(a, b);
```

### 3.4 面向 SIMD 的数据布局优化

在最优地布局数据以利用 SIMD 指令方面，有两个主要困难。

首先，SIMD 指令通常对内存中 16 字节对齐的 16 字节块操作得更快。"16 字节对齐"意味着第一个字节的内存地址是 16 的倍数。因此，如果待处理的数据数组未对齐到 16 字节，性能将严重下降。在 C 中，可以通过将 `malloc()` 调用替换为 `posix_memalign()`，或在使用标准模板库时使用自定义分配器来强制 16 字节对齐。

其次，由于每条指令操作 16 字节的块，如果数据向量的大小不是 16 的倍数，就必须处理边缘效应。最简单的解决方案是零填充：将每个大小为 $N$ 的向量视为大小为 $((N + 15) / 16) \times 16$（整数运算）的向量，末尾补零。对于大多数线性操作，如标量积、求和、矩阵乘法，嵌入到这个更大的向量空间中是不变的，不会影响结果。

在下文中，我们将假设每个向量以及行主序矩阵中的行（或列主序矩阵中的列）都是对齐且经过零填充的。

### 3.5 浮点 SIMD 与 Intel SSE2

支持 SSE2 的 Intel 和 AMD 处理器提供了基本指令来使用浮点 SIMD 算术执行乘加步骤：`_mm_add_ps()` 和 `_mm_mul_ps()`。以下是一个简化示例，将 `__m128 *a` 和 `__m128 *b` 的标量积累加到 `__m128 sum`：

```c
// c[0] = a[0]*b[0], ..., c[3] = a[3]*b[3]
__m128 c = _mm_mul_ps(*a, *b);
// sum[0] += c[0], ..., sum[3] += c[3]
sum = _mm_add_ps(c, sum);
```

`sum` 现在包含 4 个部分和，必须通过水平相加得到最终结果，可使用 SSE2 提供的 shuffle 指令或 SSE3 提供的水平求和指令（`_mm_hadd_ps()`）。大多数现代编译器能够在一定程度上自动利用 SIMD 指令。然而，根据我们的经验，自动向量化远不及通过显式利用 SIMD 编写代码所能达到的性能提升。

### 3.6 与优化 BLAS 包的比较

为了说明这些简单技术与市售快速矩阵库相比的表现，我们将基准测试中的矩阵乘法与 Eigen [9] 进行了比较。Eigen 是一个速度非常快的库，特别关注缓存优化。

表 2：与 Eigen 的比较

| 矩阵 A | 矩阵 B | $A \\times B$ (Eigen 2.0) | $A \\times B$ (Eigen 3.0) | $A \\times B$ (自定义) |
|---|---|---|---|---|
| $2000 \\times 2000$ | 2000 \\times 1 | 6.0 ms | 5.6 ms | 1.2 ms |
| $2000 \\times 2000$ | 2000 \\times 2 | 5.8 ms | 6.5 ms | 2.4 ms |
| $2000 \\times 2000$ | 2000 \\times 4 | 5.1 ms | 6.1 ms | 4.6 ms |
| $2000 \\times 2000$ | 2000 \\times 8 | 5.0 ms | 7.7 ms | 9.4 ms |
| $2000 \\times 2000$ | 2000 \\times 16 | 8.7 ms | 10.7 ms | 19.0 ms |

表 2 显示，在此特定场景下，我们的实现与 Eigen 相当。然而，Eigen 是一个通用得多的库，因此这些数字不应被理解为关于 Eigen 性能的一般性陈述。在此特定上下文中，我们的实现在处理较薄的 B 矩阵时显得更快，但对于较大的矩阵扩展性不佳。Eigen 的扩展性随数据大小增加而非线性，表明该库使用了块启发式策略来优化其缓存行为。

### 3.7 与优化 GPU 实现的比较

我们将第 2 节中描述的端到端神经网络实现与基于 CUDAMat [3] 的实现进行了比较。GPU 实验在与用于其他基准测试的同一台机器上安装的 NVIDIA Tesla C2070 GPU 板上运行。

由于其大规模并行性，GPU 非常适合批处理数据。表 3 显示，从无批处理时的 2.8 倍性能增益开始，对于小批量，GPU 吞吐量几乎随批量大小线性扩展。正如我们将在第 5.1 节中看到的，在 CPU 上，批处理仅使可比基准测试提升一小部分。

表 3：GPU 实现

| 批量大小 | 处理 1 秒语音的时间 |
|---|---|
| CPU 1 | 1360 ms |
| GPU 1 | 490 ms |
| GPU 2 | 250 ms |
| GPU 4 | 125 ms |
| GPU 8 | 66 ms |
| GPU 128 | 20 ms |

## 4 定点实现

神经网络有几个特性使其成为定点实现的主要候选对象。首先，激活值是 $[0, 1]$ 区间内的概率，这意味着它们可以表示为无符号整数而无需过多考虑缩放。其次，所有中间层的输入都是激活值，其输出通过 sigmoid 函数压缩。这使得权重的动态范围保持有界，使其成为有符号整数表示的良好候选对象。第三，由于操作的线性性质和 sigmoid 的动态范围压缩，量化误差往往次线性传播，不会引起数值不稳定。

### 4.1 线性量化

出于即将阐明的原因，我们使用 8 位量化将激活值转换为 `unsigned char`，将中间层权重转换为 `signed char`，但偏置编码为 32 位 `int` 除外。一个例外是输入层，它保持浮点，以更好地适应可能具有更大动态范围的非概率输入。在我们的特定用例中，输入层（$440\\times2000$）远小于任何后续层，对整体速度影响不大。将网络的大部分量化到 8 位的一个重要好处是网络的总体内存占用因此减少了 3 到 4 倍。

权重通过取每层的最大幅度归一化到 $[-128, 127]$ 范围内进行缩放。偏置按相同比例缩放并线性量化为 32 位。每层的矩阵乘法产生 32 位整数，然后通过快速近似 sigmoid 实现映射为 8 位概率。

注意，这个定点网络的合理调优实现虽然比等效的浮点实现快 2 倍，但仍然慢于 SSE2 优化的系统（表 1）。在许多其他场景中已经观察到，定点实现在现代 CPU 上不一定能与浮点等效实现竞争。下面我们将看到如何重新获得显著优势。

### 4.2 Intel SSSE3

Intel SSSE3 指令集 [8] 引入了 `pmaddubsw` 指令（对应的内建函数为 `_mm_maddubs_epi16()`），它与我们量化的神经网络计算完美匹配。该指令接受 16 个无符号 8 位整数（激活值）和 16 个有符号 8 位整数（权重），对它们执行并行乘加操作，产生 8 个 16 位整数。注意，由于 16 位可能不足以容纳 2 个有符号 8 位数与 2 个无符号 8 位数的和积，结果会被饱和到 16 位，这意味着该操作可能是真实乘加的近似。在实践中，这对神经网络不是问题，因为大幅值输出往往通过 sigmoid 被压缩。

在支持该指令的 CPU 上，将 `__m128i *u` 和 `__m128i *s` 的定点和积累加到 `__m128i sum` 可编写为：

```c
// c[0] = saturate(u[0]*s[0] + u[1]*s[1]) ...
// c[7] = saturate(u[14]*s[14] + u[15]*s[15])
__m128i c = _mm_maddubs_epi16(u, s);
// 将 4 个最低 16 位整数解包为 32 位
__m128i lo = _mm_srai_epi32(_mm_unpacklo_epi16(c, c), 16);
// 将 4 个最高 16 位整数解包为 32 位
__m128i hi = _mm_srai_epi32(_mm_unpackhi_epi16(c, c), 16);
// 将它们加到 4 个 32 位整数累加器中
sum = _mm_add_epi32(_mm_add_epi32(lo, hi), sum);
```

这带来了 3 倍的计算加速，使我们的基准测试轻松达到实时。

### 4.3 Intel SSE4

SSE4.1 指令集 [8] 引入了一个小的优化，使用单条指令进行 16 位到 32 位的转换：

```c
// c[0] = saturate(u[0]*s[0] + u[1]*s[1]) ...
// c[7] = saturate(u[14]*s[14] + u[15]*s[15])
__m128i c = _mm_maddubs_epi16(u, s);
// 将 4 个最低 16 位整数解包为 32 位
__m128i lo = _mm_cvtepi16_epi32(c);
// 将 4 个最高 16 位整数解包为 32 位
__m128i hi = _mm_cvtepi16_epi32(_mm_shuffle_epi32(c, 0x4e));
// 将它们加到 4 个 32 位整数累加器中
sum = _mm_add_epi32(_mm_add_epi32(lo, hi), sum);
```

在我们的基准测试中，这相比 SSSE3 带来了 9% 的相对速度提升。

## 5 进一步的任务特定优化

表 4 中的所有性能数据均适用于第 2 节中描述的基准测试：计算 100 帧（1 秒）语音。

表 4：批处理和惰性评估的效果

| 评估方式 | 节号 | 活跃状态 | 批量大小 | 最佳加速比 |
|---|---|---|---|---|
| | | | 1 | 2 | 4 | 8 |
| 急迫评估 | 5.1 | ⋆ | 472 ms | 409 ms | 374 ms | 356 ms | 25% |
| 惰性评估 | 5.2 | 1% | 257 ms | 203 ms | 180 ms | 169 ms | 64% |
| | | 10% | 298 ms | 243 ms | 214 ms | 199 ms | 58% |
| | | 30% | 371 ms | 308 ms | 273 ms | 255 ms | 46% |
| | | 50% | 436 ms | 366 ms | 326 ms | 307 ms | 35% |
| | | 100% † | 579 ms | 499 ms | 458 ms | 439 ms | 7% |
| 批处理惰性评估 | 5.3 | 1% | 261 ms | 201 ms | 177 ms | 166 ms | 65% |
| | | 10% | 299 ms | 231 ms | 197 ms | 181 ms | 62% |
| | | 30% | 380 ms | 286 ms | 233 ms | 212 ms | 55% |
| | | 50% | 454 ms | 337 ms | 268 ms | 241 ms | 49% |
| | | 100% | 617 ms | 450 ms | 349 ms | 311 ms | 34% |

### 5.1 批处理

值得注意的是，使用了第 4.3 节优化的 CPU 实现，在无批处理的情况下略优于第 3.7 节中描述的 GPU 实现（见表 1 和表 3）。虽然对于离线应用，GPU 相比 CPU 具有非常大的优势，但当批处理不可选时，这种优势似乎变得可以忽略不计。然而，批处理可以进一步改善内存局部性，在 CPU 上也同样有益。这种好处需要与实时应用中可能增加的延迟进行权衡。在流式语音识别中，通常在话语开始时加入几百毫秒的前瞻，以帮助改善语音和噪声统计量的运行时估计，这使得可以在几十毫秒内以小批量处理帧。为充分利用批处理，批次必须批量地通过网络的所有层传播，这样每个线性计算都成为矩阵-矩阵乘法，可以充分利用 CPU 对权重和激活值的缓存。

### 5.2 惰性评估

传统的 GMM/HMM 语音识别模型相比混合神经网络方法具有计算优势。在解码期间，每帧只需要计算一小部分状态分数。由于每个状态有自己的小型高斯集合，每次只需要访问总参数空间的一小部分。在一个大词汇量任务的小样本上，我们发现例如在任何时间点，平均约 25% 到 30% 的状态是活跃的。假设 GMM 系统需要的参数总数与混合系统大致相同，这显著减少了算术运算和内存访问次数。此外，有几种众所周知的 Gaussian selection 技术可以进一步缩小需要评估的高斯池 [10,11]。在密集神经网络的情况下，原则上每个参数在每帧都必须被访问。一个显著的例外是最后一层，它只需要在解码需要某个状态的后验概率时才进行计算。这开启了惰性评估的可能性，即仅在解码器需要时才计算状态后验概率。例如，在我们的基准测试中，很大一部分计算花费在评估最后一层上，该层包含了全部参数的整整 55%。如表 4 所示（比较行 ⋆ 和行 †），以惰性方式评估最后一层给矩阵计算增加了低效性，因此引入了约 22% 的固定额外开销。然而，在这个例子中，即使需要计算多达 50% 的输出分数，它仍然是有益的，相比仅批处理带来了 14% 的相对提升。

### 5.3 批处理惰性评估

采用惰性评估后，我们不能再简单地跨多帧计算所有输出的批处理分数，但我们可以继续批处理除最后一层外的所有层的计算。此外，我们可以利用语音信号的片段平稳特性，即如果解码器在 $t$ 帧需要某个状态，它在 $t+1$ 帧很可能也需要。在权重仍在缓存中时，同时计算连续帧的这些后验概率批次因此提供了进一步的效率，代价是有时会计算某一帧在后续不需要的状态后验概率。表 4 展示了以批次计算输出分数（批次大小如上）与以随机顺序计算相同数量的分数相比的效果。

## 6 语音识别评估

表 1 显示，对于神经网络计算，速度提升相对于朴素基线约为 10 倍，相对于我们最佳的浮点系统约为 4 倍。我们现在在真实语音识别任务的背景下评估我们最佳的浮点和定点系统，包括解码器的开销以及真实任务内存访问模式带来的成本。

端到端系统包括一个隐马尔可夫模型、上下文相关的语音识别器。第 2 节中描述的神经网络为每个状态提供后验概率，然后除以状态先验概率以得到观测似然。评估任务包括 27400 条移动语音输入话语，如搜索查询和短消息，并使用针对该任务组合调优的大型 N-gram 语言模型。搜索使用加权有限状态转换器解码器 [12] 执行。评估在复制真实生产环境的机器集群上进行，并调优到典型生产参数。

表 5 显示，量化神经网络参数不会造成性能损失，但端到端系统运行速度提升 3 倍，且满足实时约束。

表 5：大词汇量语音识别的速度/精度结果

| | 词错误率 | 插入 | 删除 | 替换 | 第 90 百分位实时因子 |
|---|---|---|---|---|---|
| 浮点 | 14.9% | 3.0% | 2.3% | 9.6% | 2.91 |
| 定点 | 14.8% | 3.1% | 2.1% | 9.6% | 0.90 |

所引用的实时因子是所有话语中解码话语所花费时间与话语持续时间之比的第 90 百分位数。

## 7 结论

在 CPU 上评估神经网络所需的大型矩阵计算优化是一个复杂的课题，涉及不断演进的架构和性能权衡的生态系统。本文表明，简单技术可以显著提升基于神经网络的系统的性能。特别值得关注的是 x86 处理器中最近引入的定点 SIMD 操作，它们再次将性能天平倾斜向定点算术。我们展示了利用这些更快的指令，可以在精度无损的情况下使用非常大的混合网络构建实时语音识别器。

## 参考文献

[1] Victor W. Lee, Changkyu Kim, Jatin Chhugani, Michael Deisher, Daehyun Kim, Anthony D. Nguyen, Nadathur Satish, Mikhail Smelyanskiy, Srinivas Chennupaty, Per Hammarlun, Ronak Singhal, and Pradeep Dubey (2010) *Debunking the 100\\times GPU vs. CPU myth: an evaluation of throughput computing on CPU and GPU*, Proceedings of the 37th annual international symposium on Computer architecture, ISCA'10, ACM.

[2] Noriyuki Fujimoto (2008) *Faster Matrix-Vector Multiplication on GeForce 8800GTX*, Proceedings of the 22nd IEEE International Parallel and Distributed Processing Symposium (IPDPS), LSPP-402, pp. 1–8.

[3] Volodymyr Mnih (2009) *CUDAMat: a CUDA-based matrix class for Python*, Technical Report UTML TR 2009-004, Department of Computer Science, University of Toronto.

[4] Rajat Raina, Anand Madhavan, and Andrew Y. Ng (2009) *Large-scale deep unsupervised learning using graphics processors*, Proceedings of the 26th Annual International Conference on Machine Learning, ICML'09, ACM.

[5] Kyoung-Su Oh and Keechul Jung (2004) *GPU implementation of neural networks*, Pattern Recognition, 37(6):1311-1314.

[6] Honghoon Jang, Anjin Park, and Keechul Jung (2008) *Neural Network Implementation using CUDA and OpenMP*, Proceedings of the 2008 Digital Image Computing: Techniques and Applications, pp 155–161.

[7] Navdeep Jaitly, Patrick Nguyen, and Vincent Vanhoucke (2012) *Application of Pretrained Deep Neural Networks to Large Vocabulary Speech Recognition*, Submitted to ICASSP'12.

[8] Intel C++ Intrinsics Reference, http://cache-www.intel.com/cd/00/00/34/76/347603 347603.pdf

[9] Eigen, a C++ template library for linear algebra, http://eigen.tuxfamily.org/

[10] Jürgen Fritsch and Ivica Rogina (1996) *The bucket box intersection (BBI) algorithm for fast approximative evaluation of diagonal mixture Gaussians*, Proceedings of ICASSP'96.

[11] Kate M. Knill, Mark J.F. Gales, and Steve J. Young (1996) *Use of Gaussian selection in large vocabulary continuous speech recognition using HMMs*, Proceedings of ICSLP'96.

[12] OpenFst Library, http://www.openfst.org
