# 大规模分布式深度网络：附录

> Jeffrey Dean, Greg S. Corrado, Rajat Monga, Kai Chen, Matthieu Devin, Quoc V. Le, Mark Z. Mao, Marc'Aurelio Ranzato, Andrew Senior, Paul Tucker, Ke Yang, Andrew Y. Ng | Google Inc., Mountain View, CA

本文提供了Downpour SGD和Sandblaster L-BFGS的模型副本端伪代码补充。核心内容：

- Downpour SGD客户端算法的详细伪代码（算法1.1）
- Sandblaster L-BFGS协调器与参数服务器的协作伪代码（算法1.2）

关键发现：

- Downpour SGD通过异步获取参数和推送梯度实现高效的分布式SGD训练
- Sandblaster通过将L-BFGS分解为分布式标量-向量和向量-向量运算，在参数服务器分片上本地执行

---

## 摘要

## 附录

为完整起见，此处提供了Downpour SGD模型副本（客户端）端（算法1.1）和Sandblaster L-BFGS（算法1.2）的伪代码。

**算法1.1：Downpour SGD客户端**

```
procedure DOWNPOURSGDCLIENT($\alpha$, nfetch, npush)
    procedure STARTASYNCHRONOUSLYFETCHINGPARAMETERS(parameters)
        parameters $\leftarrow$ GETPARAMETERSFROMPARAMSERVER()

    procedure STARTASYNCHRONOUSLYPUSHINGGRADIENTS(accruedgradients)
        SENDGRADIENTSTOPARAMSERVER(accruedgradients)
        accruedgradients $\leftarrow$ 0

    main
        global parameters, accruedgradients
        step $\leftarrow$ 0
        accruedgradients $\leftarrow$ 0
        while true do
            if (step mod nfetch) == 0 then
                STARTASYNCHRONOUSLYFETCHINGPARAMETERS(parameters)
            data $\leftarrow$ GETNEXTMINIBATCH()
            gradient $\leftarrow$ COMPUTEGRADIENT(parameters, data)
            accruedgradients $\leftarrow$ accruedgradients + gradient
            parameters $\leftarrow$ parameters − $\alpha$ ∗ gradient
            if (step mod npush) == 0 then
                STARTASYNCHRONOUSLYPUSHINGGRADIENTS(accruedgradients)
            step $\leftarrow$ step + 1
```

Sandblaster是一个用于分布式批处理优化过程的框架。Sandblaster中的一个核心概念是将操作分解为在DistBelief参数服务器上的本地计算。举例来说，假设我们有10亿个参数和10个参数服务器分片，那么每个分片拥有 $1/10$ 的参数。可以将L-BFGS分解为一系列标量-向量乘积（ $\alpha \times \mathbf{x}$ ）和向量-向量内积（ $\mathbf{x}^T \mathbf{y}$ ）的序列，其中每个向量都是10亿维的。如果让第一个分片始终负责L-BFGS内部使用的每个向量的前 $1/10$ ，第二个分片始终负责每个向量的第二个 $1/10$ ，依此类推直到最后一个分片始终负责每个向量的最后 $1/10$ ，那么可以证明，这些标量-向量运算和向量-向量运算都可以通过极少的通信以分布式方式完成。这样一来，任何中间向量值结果都会自动以同样的分布式方式存储，而任何中间标量值结果则会广播到所有分片。

**算法1.2：Sandblaster L-BFGS**

```
procedure REPLICA.PROCESSPORTION(portion)
    if (!hasParametersForStep) then
        parameters $\leftarrow$ GETPARAMETERSFROMPARAMSERVER()
    data $\leftarrow$ GETDATAPORTION(portion)
    gradient $\leftarrow$ COMPUTEGRADIENT(parameters, data)
    localAccruedGradients $\leftarrow$ localAccruedGradients + gradient

procedure PARAMETERSERVER.PERFORMOPERATION(operation)
    PerformOperation

main
    step $\leftarrow$ 0
    while true do
        comment: PS: ParameterServer
        PS.accruedgradients $\leftarrow$ 0
        while (batchProcessed < batchSize) do
            for all (modelReplicas) do comment: Loop is parallel and asynchronous
                if (modelReplicaAvailable) then
                    REPLICA.PROCESSPORTION(modelReplica)
                    batchProcessed $\leftarrow$ batchProcessed + portion
                if (modelReplicaWorkDone and timeToSendGradients) then
                    SENDGRADIENTS(modelReplica)
                    PS.accruedGradients $\leftarrow$ PS.accruedGradients + gradient
        COMPUTELBFGSDIRECTION(PS.Gradients, PS.History, PS.Direction)
        LINESEARCH(PS.Parameters, PS.Direction)
        PS.UPDATEPARAMETERS(PS.parameters, PS.accruedGradients)
        step $\leftarrow$ step + 1
```
