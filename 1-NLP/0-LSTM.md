# 0-LSTM



## RNN

Sequence to Sequence Learning with Neural Networks

![image-20260812104632532](.picture/image-20260812104632532.png)

$\boldsymbol{h}_{t} = \tanh(\boldsymbol{h}_{t-1}\boldsymbol{W}_h + \boldsymbol{x}_t\boldsymbol{W}_x + \boldsymbol{b})$

如图 6-10 所示，这里将 $\tanh(\boldsymbol{h}_{t-1}\boldsymbol{W}_h + \boldsymbol{x}_t\boldsymbol{W}_x + \boldsymbol{b})$ 这个计算表示为一个长方形节点 $\tanh$（$\boldsymbol{h}_{t-1}$ 和 $\boldsymbol{x}_t$ 是行向量），这个长方形节点中包含了矩阵乘积、偏置的和以及基于 $\tanh$ 函数的变换。



## LSTM

![image-20260812123220564](/Users/dazhang/PycharmProject/Papers/1-NLP/.picture/image-20260812123220564.png)