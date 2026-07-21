## 算法部分

#### Algorithm 1.1: $\text{Downpour SGD Client}(\alpha, n_{fetch}, n_{push})$

Algorithm 1.1 **Downpour SGD Client** 是模型副本（客户端）的伪代码。

**三个参数**：
- `α`：学习率
- `n_fetch`：每多少步从参数服务器拉取一次参数（控制拉取频率）
- `n_push`：每多少步向参数服务器推送一次累积梯度（控制推送频率）



核心是用**三个 弱同步异步线程** 解耦参数拉取、梯度推送和训练计算。算法开始时声明全局变量 `parameters`（本地参数副本）和 `accruedgradients`（累积梯度缓冲区），然后进入无限循环。

1. **Fetch 线程**（异步启动）：当 `(step mod nfetch) == 0` 时触发，从参数服务器拉取最新 全局参数 **覆盖本地** `parameters`。
2. **Push 线程**（异步启动）：当 `(step mod npush) == 0` 时触发，将 `accruedgradients` 推送给参数服务器，然后立即置零 `accruedgradients ← 0`。
3. **主线程**（训练循环，每步执行）：
   - 从训练数据中获取一个 minibatch
   - 本地计算梯度 `gradient = ComputeGradient(parameters, data)`
   - `accruedgradients += gradient`（累积到本地缓冲区，留待推送）
   - `parameters -= α · gradient`（本地参数每步用当前单步梯度立即更新，不等推送完成，不受推送频率影响）
   - step + 1

**`accruedgradients` 的作用**：本地累积多个 batch 的梯度，凑够 `npush` 步再一次性推送给参数服务器，降低通信频率。`accruedgradients` 推送给参数服务器后，参数服务器将其加到全**局累积梯度**上。当积累到足够的梯度后，参数服务器会用这些聚合梯度去更新其持有的**全局参数**。之后模型副本再从参数服务器拉取更新后的全局参数（通过 `FETCHINGPARAMETERS`），从而**让所有副本间接地共享学习进度**。简单说：

- **客户端**：`accruedgradients` 是本地多个 batch 的梯度累积，推给参数服务器。
- **参数服务器**：收到后合并到全局梯度，用于更新全局参数。
- **全局参数**：被其他副本下次 `FETCH` 拉取，实现分布式协同。



#### Algorithm 1.2: $\text{Sandblaster L-BFGS}()$

**Algorithm 1.2 Sandblaster L-BFGS**

与 Downpour SGD 的异步更新不同，Sandblaster 是**分布式批优化框架**，核心思想是将 L-BFGS 的运算分解到参数服务器和模型副本上，减少通信。

**两个核心过程**：

1. **`REPLICA.PROCESSPORTION(portion)`**（模型副本）：
   - 如果本地参数未更新，先从参数服务器拉取。
   - 取自己负责的那部分数据（`portion`），计算梯度。
   - 将梯度累积到 `localAccruedGradients`。

2. **`PARAMETERSERVER.PERFORMOPERATION(operation)`**（参数服务器）：
   - 执行 L-BFGS 所需的向量运算（如标量-向量乘 $\alpha \times x$、向量内积 $x^T y$），这些运算天然可分解到各个分片。

**主循环**：
- 遍历所有模型副本（并行异步循环），每个可用副本执行 `PROCESSPORTION`。
- 当一个副本工作完成且到发送时机时，将其梯度发送给参数服务器。
- 参数服务器累加所有副本的梯度到 `PS.accruedGradients`。
- 收集足够梯度后，**一次性执行 L-BFGS 方向计算**（`COMPUTELBFGSDIRECTION`，利用历史梯度-方向对信息），再做**线搜索**（`LINESEARCH`）找到最优步长，最后更新全局参数。



#### 关键区别 Downpour SGD VS Sandblaster L-BFGS

| 特性     | Downpour SGD                   | Sandblaster L-BFGS                        |
| -------- | ------------------------------ | ----------------------------------------- |
| 更新方式 | 异步，每个副本独立更新本地参数 | 同步批处理，收集所有副本梯度后再做 L-BFGS |
| 优化器   | SGD（一阶）                    | L-BFGS（拟牛顿，二阶近似）                |
| 通信模式 | 每 n 步推送/拉取               | 分片数据 + 分布式向量运算                 |
| 适用场景 | 大规模、容忍噪声               | 需要更精确方向的批量优化                  |



## 激活函数

语音识别任务是将短音频片段中的中心区域（或帧）分类为数千个声学状态之一。我们使用了一个五层深度网络：**四个隐藏层采用 sigmoid 激活**，每层 2560 个节点，以及一个具有 8192 个节点的 softmax 输出层。输入表示为 11 个连续重叠的 25 ms 语音帧，每帧由 40 个对数能量值表示。网络是**层间全连接的**，总共有约 4200 万个模型参数。我们使用 11 亿个弱标签样本的数据集进行训练，并在保留测试集上评估。有关类似的深度网络配置和训练过程，请参见 [27]。

2012 年时 sigmoid 仍是语音任务的主流选择：

1. **Sigmoid 是当时的默认激活** — ReLU（Nair & Hinton 2010）刚提出不久，尚未在语音领域普及。2012 年的标准做法是用 sigmoid/tanh。
2. **继承基线配置 [27]** — 论文直接沿用了 [27] 的网络结构，目的是验证分布式系统（Downpour/Sandblaster）的可扩展性，而非探索激活函数。
3. **异步更新的稳定性** — sigmoid 输出范围 (0,1)，梯度有界，在异步并行大规模 SGD 中比 ReLU 更容易保持稳定（ReLU 的 unbounded 激活在异步环境下可能放大噪声）。
4. **深度较浅（5 层）** — sigmoid 的梯度消失问题在 5 层网络中尚不严重，未成为瓶颈。

今天看 sigmoid 在深层网络中已基本被 ReLU 族替代，但在当时这是一个合理且主流的选择。





## 附录

#### 1.1 伪代码

```python
DOWNPOURSGDCLIENT(α, nfetch, npush)
    procedure STARTASYNCHRONOUSLYFETCHINGPARAMETERS(parameters)
        parameters ← GETPARAMETERSFROMPARAMSERVER()

    procedure STARTASYNCHRONOUSLYPUSHINGGRADIENTS(accruedgradients)
        SENDGRADIENTSTOPARAMSERVER(accruedgradients)
        accruedgradients ← 0

    main
        global parameters, accruedgradients
        step ← 0
        accruedgradients ← 0
        while true do
            if (step mod nfetch) == 0 then
                STARTASYNCHRONOUSLYFETCHINGPARAMETERS(parameters)
            data ← GETNEXTMINIBATCH()
            gradient ← COMPUTEGRADIENT(parameters, data)
            accruedgradients ← accruedgradients + gradient
            parameters ← parameters − α ∗ gradient
            if (step mod npush) == 0 then
                STARTASYNCHRONOUSLYPUSHINGGRADIENTS(accruedgradients)
            step ← step + 1
```

#### 1.2 伪代码


```python
SANDBLASTERLBFGS()
    procedure REPLICA.PROCESSPORTION(portion)
        if (!hasParametersForStep) then
            parameters ← GETPARAMETERSFROMPARAMSERVER()
        data ← GETDATAPORTION(portion)
        gradient ← COMPUTEGRADIENT(parameters, data)
        localAccruedGradients ← localAccruedGradients + gradient

    procedure PARAMETERSERVER.PERFORMOPERATION(operation)
        PerformOperation

    main
        step ← 0
        while true do
            comment: PS: ParameterServer
            PS.accruedgradients ← 0
            while (batchProcessed < batchSize) do
                for all (modelReplicas) comment: 并行异步循环
                    if (modelReplicaAvailable) then
                        REPLICA.PROCESSPORTION(modelReplica)
                        batchProcessed ← batchProcessed + portion
                        if (modelReplicaWorkDone and timeToSendGradients) then
                            SENDGRADIENTS(modelReplica)
                PS.accruedGradients ← PS.accruedGradients + gradient
            COMPUTELBFGSDIRECTION(PS.Gradients, PS.History, PS.Direction)
            LINESEARCH(PS.Parameters, PS.Direction)
            PS.UPDATEPARAMETERS(PS.parameters, PS.accruedGradients)
            step ← step + 1
```
