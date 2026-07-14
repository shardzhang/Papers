面向匹配的深度学习 在
搜索与推荐

建议引用: 徐军, 何向南 和 李航 (2020), “深度学习 用于
匹配 在 搜索与推荐”, : Vol. XX, 无. XX, pp 1–193. DOI: XXX.

徐军

何向南

李航

该 文章 可以 是 使用了 仅 用于  目的 的 研究, t每个ing,
和/或 private 研究. 商业 使用 或 系统性的 downloading
(通过 robots 或 其他 自动 processes) 是 prohibited 无 显式 Publisher approval.

Boston — Delft

Contents

1 介绍

1.1 搜索与推荐 . . . . . . . . . . . . . . . .
1.2 从匹配视角统一搜索与推荐 . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
1.3 搜索中的失配挑战 . . . . . . . . . . . . . .
1.4 推荐中的失配挑战 . . . . . . . .
1.5 最新进展 . . . . . . . . . . . . . . . . . . . . . . .
1.6 关于本文综述 . . . . . . . . . . . . . . . . . . . . . .

2 传统匹配模型

2.1 学习匹配 . . . . . . . . . . . . . . . . . . . . . .
2.2 搜索与推荐中的匹配模型 . . . . .
2.3 搜索中的潜在空间模型 . . . . . . . . . . . . . . .
2.4 推荐中的潜在空间模型 . . . . . . . . .
2.5 扩展阅读 . . . . . . . . . . . . . . . . . . . . . . .

3 面向匹配的深度学习

3.1 深度学习概述 . . . . . . . . . . . . . . . . .
3.2 面向匹配的深度学习概述 . . . . . . . . . .

4 搜索中的深度匹配模型

4.1 基于表示学习的匹配 . . . . . . . .

3
3

5
7
8
9
10

13
13
19
22
25
28

30
30
45

52
53

4.2 基于匹配函数学习的匹配 . . . . . .
4.3 讨论与扩展阅读 . . . . . . . . . . . . . .

71
88

5 推荐中的深度匹配模型

98
5.1 基于表示学习的匹配 . . . . . . . .
98
5.2 基于匹配函数学习的匹配 . . . . . . 129
5.3 扩展阅读 . . . . . . . . . . . . . . . . . . . . . . . 140

6 结论与未来方向

144
6.1 综述总结 . . . . . . . . . . . . . . . . . . . 144
6.2 其他任务中的匹配
. . . . . . . . . . . . . . . . . . 145
6.3 开放问题与未来方向 . . . . . . . . . . . 146

Acknowledgements

References

149

150

面向匹配的深度学习 在
搜索与推荐
徐军1, 何向南2 和 李航3

1Gaoling School 的 Artiﬁcial 智能, Renmin University 的 China
2School 的 信息 科学 和 技术, University 的 科学
和 技术 的 China
3字节跳动人工智能实验室

抽象

匹配 是  关键 问题 在 两者 搜索与推荐, 该 是  度量  相关性 的  文档  
查询 或  兴趣 的  用户 在  物品. 机器学习
具有 是 exploited  解决  问题, 该 学习
 匹配函数 基于 在 输入 表示 和
从 标注数据, 也 称为  作为 “学习匹配”.
在 最近 年, 努力 具有 是 使  开发 深度
学习 技术 用于 匹配 任务 在 搜索与推荐. 用  可用性 的  大 量 的 数据,
强大 计算 resources, 和 高级 深度学习 技术, 面向匹配的深度学习 现在 成为
 最先进的 技术 用于 搜索与推荐.  关键   成功 的  深度学习方法
是 其 强 能力 在 学习 的 表示 和 泛化 的 匹配 patterns 从 数据 (e.g., 查询,
文档, 用户, 物品, 和 contexts, 特别是 在 其
raw 形式).

该 综述 给出  系统性的 和 全面 介绍   深度匹配 模型 用于 搜索与推荐 开发了 最近. 它 首先 给出  uniﬁed 视角 的

徐军, 何向南 和 李航 (2020), “面向匹配的深度学习 在 搜索
和 推荐”, : Vol. XX, 无. XX, pp 1–193. DOI: XXX.

2

匹配 在 搜索与推荐. 在 该 方式, 
解决方案 从  二 ﬁelds 可以 是 比较了 之下 一
框架. 则,  综述 分类  当前 深度
学习 解决方案 到 二 类型: 方法 的 表示
学习 和 方法 的 匹配函数学习. 
基本 问题, 以及  最先进的 解决方案 的 查询-文档匹配 在 搜索 和 用户-物品
匹配 在 推荐, 是 描述.  综述
旨在  帮助 研究ers 从 两者 搜索与推荐 communities  get 在-深度 理解 和 洞察
到  spaces, 激励 更多 ideas 和 讨论, 和
促进 developments 的 新 technologies.

匹配 是 不 有限的  搜索与推荐.
相似 问题 可以 是 发现 在 复述, 问题
answering, 图像标注, 和 许多 其他 应用.
在 通用,  technologies 介绍 在  综述 可以 是
广义 到  更多 通用 任务 的 匹配 之间
对象 从 二 spaces.

1

介绍

1.1 搜索与推荐

用  快速 增长 的  internet, 一 的  基本 问题
在 信息 科学 成为 甚至 更多 关键 today, 该 是, 如何
 识别  信息 满足  用户’s 需求 从  通常 巨大
池 的 信息.  目标 是  呈现  用户 仅  信息
该 是 的 兴趣 和 相关性, 在  正确 时间, 位置, 和 上下文.
如今, 二 类型 的 信息 accessing 范式, 搜索 和
推荐, 是 广泛使用 在  great 多样性 的 场景.

在 搜索, 文档 (e.g., web 文档, Twitter posts, 或 Ecommerce products) 是 首先 pre-processed 和 索引 在  搜索
engine. 之后 该,  搜索 engine 取  查询 ( 数量 的 关键词) 从  用户.  查询 描述  用户’s 信息需求.
相关 文档 是 retrieved 从  索引, 匹配了 用 
查询, 和 ranked 根据 其 相关性   查询. 用于 例子,
如果  用户 是 感兴趣的 在 news 关于 quantum 计算中,  查询
“quantum 计算中” mat 是 提交   搜索 engine 和 get
news articles 关于  主题 将 是 returned.

Diﬀerent 从 搜索,  推荐 系统 典型地 进行 不
取  查询. 而是, 它 分析  用户’s proﬁle (e.g., 人口统计

3

4

介绍

表 1.1: 信息-提供中 mechanisms 的 搜索与推荐.

搜索 推荐

查询 可用 Yes
交付 模型 Pull
用户
Beneﬁciary
无
意外发现

无
Push
用户 和 provider
Yes

和 contexts) 和 历史 interactions 在 物品, 和 则 使
推荐 在 物品   用户.  用户 特征 和 物品
特征 是 索引 和 stored 在  系统 在 advance.  物品 是
ranked 根据  似然 该  用户 是 感兴趣的 在 它们.
用于 例子, 在  news website, 当  用户 browses 和 clicks  新
文章, 若干 news articles 用 相似 主题 或 news articles 该
其他 用户 具有 clicked 一起 用  当前 一 可以 是 显示.

表 1.1 总结  diﬀerences 之间 搜索与推荐.  基本 机制 的 搜索 是 “pull”, 因为
用户 首先 使 speciﬁc requests (i.e., 提交 查询) 和 则 receive 信息.  基本 mechanisms 的 推荐
是 “push”, 因为 用户 是 提供 信息 该 它们 进行 不
speciﬁcally 请求 (e.g., 提交 查询). 这里 “beneﬁciary” means 
人们 其 interests 是  是 met 在  任务. 在  搜索 engine,
 results 是 典型地 创建了 仅基于  用户‘s needs, 和
因此  beneﬁciary 是  用户. 在  推荐 engine,  results 通常 需求  满足 两者  用户 和 providers, 和 因此 
beneﬁciary 是 所有 的 它们. 然而,  区别 是 成为中 模糊
最近. 用于 例子, 一些 搜索 engines mix 搜索 results 用 paid
advertisements, 该 beneﬁts 两者  用户 和  providers. 作为
用于 “意外发现”, 它 means 该 传统 搜索 聚焦 更多 在
信息 该 是 clearly 相关. 传统 推荐, 在
 其他 hand, 是 允许了  oﬀer 意外的 但 有用 信息.

1.2. 从匹配视角统一搜索与推荐 5

1.2 从匹配视角统一搜索与推荐

Garcia-Molina 等人 (2011) 指出 该  基本 问题
在 搜索与推荐 是  识别 信息对象 满足 用户’ 信息需求. 它 是 也 指出 该 搜索 (信息
检索) 和 推荐 (信息 过滤ing) 是  二 sides
的  相同 coin, 具有 强 connections 和 相似之处 (Belkin 和
Croft, 1992). 图 1.1 说明  uniﬁed 匹配视角 的 搜索
和 推荐.  目标 在 共同 是  呈现   用户
 信息 它们 需求.

搜索 是  检索 任务, 该 旨在  检索  文档 该
是 相关   查询. 在 对比, 推荐 是  过滤ing 任务,
该 旨在  过滤 out  物品 该 是 的 兴趣   用户 (Adomavicius 和 Tuzhilin, 2005). 作为 这样的, 搜索 可以 是 考虑了 作为
进行 匹配 之间 查询 和 文档, 和 推荐 可以 是 考虑了 作为 进行 匹配 之间 用户 和
物品. 更多 形式化地, 两者  匹配 在 搜索与推荐
可以 是 考虑了 作为 构建  匹配模型 f : X × Y 7→ R
该 计算  匹配程度 之间 二 输入 对象 x 和
y, 其中 X 和 Y 表示 二 对象 spaces. X 和 Y 是  spaces 的
查询 和 文档 在 搜索, 或  spaces 的 用户 和 物品 在
推荐.

之下  uniﬁed 匹配视角 在 图 1.1, 我们 使用  术语 信息对象  表示  文档/物品  检索/recommend,
和 使用 信息需求  表示  查询/用户 在  各自
任务. 通过 统一  二 任务 之下  相同 视角 的 匹配 和
可比地 回顾 现有 技术, 我们 可以 提供 deeper 洞察 和 更多 强大 解决方案   问题. 此外, 统一
 二 任务 也 具有 实际 和 理论的 启示.

搜索与推荐 具有 已经 是 结合 在 一些
实际 应用. 用于 例子, 在 一些 E-commerce sites, 当
 用户 提交  查询,  排序 列表 的 products 是 呈现了
基于 在 不 仅 相关性 (查询-乘积 匹配) 但 也 用户
兴趣 (用户-乘积 匹配). 在 一些 生活方式 apps, 当 
用户 searches 用于 餐厅,  results 是 returned 基于 在 两者

6

介绍

图 1.1: Uniﬁed 视角 的 匹配 在 搜索与推荐.

相关性 (查询-restaurant 匹配) 和 用户 兴趣 (用户-restaurant
匹配). 那里 是  清晰 趋势 该 搜索与推荐 将
是 集成 到  单一 系统 在 一定 场景  meet 用户’
needs 更好, 其中 匹配 plays  本质 角色.

搜索与推荐 已经 具有 许多 共享 technologies 因为 其 相似之处 在 匹配. 一些 搜索 问题
可以 是 解决 通过 使用中 推荐 技术 (Zamani 等人,
2016), 和 反之亦然 (Costa 和 Roda, 2011), 在  基础 的 匹配.
用  使用 的 深度学习 technologies,  匹配 模型 用于
搜索与推荐 bear 甚至 更多 相似性 在 架构 和 方法论, 作为 reﬂected 在  技术: 嵌入 
输入 (查询, 用户, 文档, 和 物品) 作为 分布式 表示, 组合中 神经网络 组件  表示  匹配
函数, 和 训练  模型 参数 在  端--端 方式.
此外, 搜索与推荐 可以 是 联合地 建模 和 优化 如果 它们 share  相同 集合 的 信息对象 (作为 在  以上
例子 的 E-commerce sites 和 生活方式 apps) (Zamani 和 Croft,
2018; Schedl 等人, 2018; Zamani 和 Croft, 2020). 因此, 在 顺序
 开发 更多 高级 ones, 它 是 必要 和 advantageous 
取  uniﬁed 匹配视角  分析 和 比较 现有 搜索
和 推荐 technologies.

 匹配 任务 在 搜索 和 在 推荐 面临 diﬀerent
挑战 在 实践.  底层 问题 是 本质上  相同,
然而, 该 是,  失配 挑战. 下一个, 我们 介绍  关键
挑战 的  二 任务, 分别地.

查询/用户&ContextRelevantdocuments/itemsInformationneedsInformationobjectsSearchengine/RecommendersystemIndexofdocuments/itemsmatching1.3. 搜索中的失配挑战

7

1.3 搜索中的失配挑战

在 搜索, 查询 和 文档 (通常 其 titles) 是 取 作为
文本.  相关性 的  文档   查询 是 主要地 表示了
通过  匹配程度 之间  二.  文档 是 考虑了
相关   查询 如果  匹配程度 是 高. 自然 语言
理解 通过 计算机 是 仍然 有挑战, 和 因此  计算
的 匹配程度 是 仍然 有限的   文本层面 但 不 在  语义
水平.  高 匹配 程度 在  文本层面 进行 不 necessarily 均值
高 相关性 在  语义层面, 和 反之亦然. 此外, 查询
是 issued 通过 用户, 同时 文档 是 编写 通过 编辑. Due
  歧义 的 自然 语言, 用户 和 编辑 是 可能 
使用 diﬀerent 语言 风格 和 表达 用于 呈现中  相同
concepts 或 主题. 作为  结果,  搜索 系统 可以 suﬀer 从  socalled 查询-文档 失配 问题. Speciﬁcally, 当  用户
的  搜索 engine 和  编辑 的  文档 使用 diﬀerent 文本
 描述  相同 概念 (e.g., “ny 次” v.s. “新 york 次”),
查询-文档 失配 可以 发生. 该 是 仍然 一 的  主要
挑战 用于 搜索. 转向   交叉-模态 IR (e.g., 使用中 文本
查询  检索 图像 文档),  查询-文档 失配
问题 成为 甚至 更多 严重, 因为 diﬀerent 模态 具有
diﬀerent 类型 的 表示. 在 交叉-模态 检索, 一 主要
挑战 是 如何  构建  匹配函数 该 可以 桥接 
“异质性鸿沟” amongst  模态.

 解决  查询-文档 失配 挑战, 方法 具有
是 提出  执行 匹配 在  语义层面, 称为
 作为 语义匹配.  关键 想法 在  解决方案 是 或 
执行 更多 查询 和 文档理解  更好 表示
 meanings 的  查询 和 文档, 或  构建 更多 强大
匹配 函数 该 可以 桥接  语义鸿沟 之间  查询
和 文档. 两者 传统机器学习 方法 (Li 和
Xu, 2014) 和 深度学习 方法 (Guo 等人, 2019b; Mitra 和
Craswell, 2018; Onal 等人, 2018) 具有 是 开发了 用于 语义
匹配.

8

介绍

1.4 推荐中的失配挑战

 mismatching 问题 是 甚至 更多 严重 在 推荐. 在
搜索, 查询 和 文档 由组成 的 术语 在  相同 语言1,
使 它 在 至少 有意义的  进行 直接 匹配 在 其 术语.
在 推荐, 然而, 用户 和 物品 是 通常 表示了
通过 diﬀerent 类型 的 特征, 用于 例子,  特征 的 用户 可以 是
 用户 ID, age, income 水平, 和 最近 behaviors, 同时  特征
用于 物品 可以 是  物品 ID, 类别, price, 和 brand 命名. 由于 
特征 的 用户 和 物品 是 从  spaces 的 diﬀerent 语义,
 naive 方法 基于 在  匹配 的 superﬁcial 特征 进行
不 工作 用于 推荐. 更多 challengingly,  物品 可以 是
描述 通过 多-模态 特征, e.g., images 的 clothing products
和 覆盖 images 的 movies, 该 可以 play  pivotal 角色 在 ﬀecting
 决策-使 的 用户. 在 这样的 visually-aware 场景, 我们 需求
 考虑  交叉-模态 匹配 之间 用户 和 多-模态
content.

 解决  推荐中的失配挑战,  协同 过滤ing 原则 具有 是 提出 (Shi 等人, 2014). 协同过滤 (CF), 该 工作 作为  基本 基础 的
几乎 所有 personalized 推荐 系统, 假设 该  用户
可以 如 (consume)  物品 该 是 liked (消费) 通过  相似
用户, 用于 该  相似度 是 判断 从  历史 interactions (Sarwar 等人, 2001). 然而, 直接地 评估中  相似度
之间 用户 (物品) suﬀers 从  稀疏性 问题, 由于  用户 仅
消费  少数 物品 在  整体 物品空间.  典型 假设
 解决  稀疏性 问题 是 该  用户-物品交互 矩阵
是 低-排序, 该 因此 可以 是 估计 从 低-维 用户
(和 物品) 潜在 特征 矩阵. 则  用户 (物品) 相似度 可以 是
更多 可靠地 reﬂected 在  潜在 特征 矩阵. 该 导致  
eﬀectiveness 的 矩阵分解 用于 协同 过滤ing (Koren
等人, 2009; Rendle 等人, 2009), 该 成为  强 CF 方法
和  本质 设计 用于 许多 推荐 模型. 除之外 矩阵
factorization, 许多 其他 类型 的 CF 方法 具有 是 开发了

1这里 我们 进行 不 考虑 交叉-语言 信息检索.

1.5. 最新进展

9

如 神经网络-基于 方法 (他 等人, 2017c; Liang 等人, 2018)
和 图-基于 方法 (Wang 等人, 2019b; Ying 等人, 2018).

 利用  各种 辅助信息 超出  交互
矩阵, 这样的 作为 用户 proﬁles, 物品 attributes, 和  当前 contexts,
许多 generic 推荐 模型 该 follow  标准 监督
学习 范式 具有 是 提出. 这些 模型 可以 是 使用了 在
 (re-)排序 阶段 的  推荐 engine, e.g., 通过 预测中
 点击率 (CTR) 的  物品.  代表性 模型 是
分解机 (FM) (Rendle, 2010), 该 扩展  低-排序
假设 的 矩阵分解  模型 特征 interactions. 由于
 表达能力 的 FM 是 受限于 其 线性 和 秒-顺序
交互 建模, 许多 后来 努力 补充 它 用 神经
网络 用于 非线性 和 更高-顺序 交互 建模 (他 和
Chua, 2017; Lian 等人, 2018; Zhou 等人, 2018). 这些 神经网络
模型 具有 现在 是 密集地 使用了 在 工业 应用. Batmaz
等人 (2019) 和 Zhang 等人 (2017) 综述了 深度学习 方法
用于 推荐 系统.

Please 注意 该 尽管 查询-文档匹配 和 用户-物品
匹配 是 关键 用于 搜索 engines 和 推荐 系统,
这些 系统 也 包括 其他 重要 组件. 除之外 匹配, 网页搜索 engines 也 包括 爬取, 索引, 文档理解, 查询理解, 和 排序, etc. 推荐
系统 也 包括 组件 这样的 作为 用户建模 (proﬁling),
索引, 缓存, 多样性 控制, 和 在线 探索, etc.

1.5 最新进展

尽管 传统机器学习 是 成功 用于 匹配 在
搜索与推荐, 最新进展 在 深度学习 具有
brought 甚至 更多 signiﬁcant 进展   领域 用  大 数量
的 深度匹配 模型 提出.  幂 的 深度学习 模型
lies 在  能力  学习 分布式 表示 从  raw
数据 (e.g., 文本) 用于  匹配 问题,  避免 许多 limitations
的 hand-crafted 特征, 和  学习  表示 和 匹配
网络 在  端--端 fashion. 此外, 深度 神经 网络 具有
suﬃcient 容量  模型 复杂 匹配 任务. 它们 具有

10

介绍

 ﬂexibility 的 扩展中  交叉-模态 匹配 自然地, 其中
 共同 语义 空间 是 学会了  表示 数据 的 diﬀerent
模态 统一地. 所有 这些 特征 是 有帮助 在 处理
 复杂度 的 搜索与推荐.

在 搜索,  失配 之间 查询 和 文档 是 更多 eﬀectively 解决了 通过 深度 神经 网络, 包括中  馈-前向
神经 网络 (FFNs), 卷积 神经 网络 (CNNs), 和
Recurrent 神经 网络 (CNNs), 因为 它们 具有 stronger capabilities 在 表示学习 和 匹配函数学习.
大多数 值得注意的是, 来自Transformer的双向编码器表示
(BERT) 具有 signiﬁcantly 增强  准确率 的 匹配 在 搜索
和 脱颖而出 作为  最先进的 技术 现在.

在 推荐, 最近 焦点 具有 转移 从 行为-centric
协同 过滤ing  信息-rich 用户-物品匹配 作为 在
序列, 上下文-aware, 和 知识图谱 增强 recommendations, 该 是 所有 实际 场景-驱动. 从角度 技术,
图 神经 网络 (GNNs) 成为  新兴 工具 用于 表示学习 (Wang 等人, 2019b; Wang 等人, 2019), 因为
推荐 数据 可以 是 自然地 组织 在  异质
图 和 GNNs 具有  能力  利用 这样的 数据.  处理 用户 行为 序列 数据, 自身-注意力 和 BERT 是 也
采用, 该 证明 有前景 results 在 序列推荐 (Sun 等人, 2019; Yuan 等人, 2020).

1.6 关于本文综述

该 综述 聚焦 在  基本 问题 的 匹配 在 搜索
和 推荐. 最先进的 匹配 解决方案 使用中 深度
学习 是 描述.  uniﬁed 视角 的 搜索与推荐
从 匹配 是 提供.  ideas 和 解决方案 解释了 可以 激励 工业 从业者  轮次  研究 results 到 products.
 方法 和  讨论 可以 帮助 学术 研究ers 
开发 新 方法.  uniﬁed 视角 可以 bring 研究ers 在 
搜索 和  推荐 communities 一起 和 启发 它们
 探索 新 directions.

 综述 是 组织 作为 follows: 章 2 描述  tradi-

1.6. 关于本文综述

11

tional 机器学习 方法  匹配 用于 搜索与推荐; 章 3 给出  通用 形式化 的 深度匹配
方法; 章 4 和 章 5 描述  细节 的  深度学习 方法  搜索与推荐 分别地. 每个 章
包括  表示学习-基于 方法 和 匹配
函数 学习-基于 方法; 章 6 总结  综述
和 讨论 open 问题. Chapters 2, 3, 4, 和 5 是 自身-包含了,
和  读者 可以 选择  read 在  基础 的 其 兴趣 和
需求.

注意 该 深度学习 用于 搜索与推荐 是  very hot
主题 的 研究. 作为 这样的, 该 综述 进行 不 try  覆盖 所有 相关
工作 在  ﬁelds 的 信息检索 和 推荐 系统.
而是, 我们 讨论  大多数 代表性 方法 的  二 ﬁelds
从  视角 的 匹配, 旨在  总结 其 关键 ideas
该 是 通用 和 本质. 在 particular, 该 综述 覆盖 
代表性 工作 在之前 2019.

若干 之前 FnTIR 问题 具有 给定 详细 introductions 
相关 主题. 一 问题 (Li 和 Xu, 2014) 介绍  传统
机器学习 方法   语义匹配 问题, 特别是 在 网页搜索. 我们的 综述 在 该 问题 是 very diﬀerent 从
它 在  sense 该 1) 它 聚焦 在  newly 开发了 深度学习
方法, 和 2) 它 考虑 两者 搜索与推荐. Mitra
和 Craswell (2018) 进行了  全面 综述 在 深度 神经
网络 用于 信息检索, 称为  作为 神经 IR. Bast 等人
(2016) 进行  综述 在  技术 和 系统 的 语义
搜索, 该 means 搜索 用 关键词 查询, structured 查询,
和 自然 语言 查询,  文档, 知识 基于, 和 其
combinations.

若干 综述 和 教程 具有 是 使 在 深度学习 用于
信息检索 和 推荐. 用于 例子, Onal 等人
(2018b) 具有 解释了 神经 模型 用于 临时检索, 查询理解, 问答, 赞助商搜索, 和 相似 物品
检索. Zhang 等人 (2019) 综述 深度学习-基于 推荐 方法 根据  分类 的 深度学习 技术,
e.g., MLP, CNN, RNN, 自编码器-基于, 和 因此 在. 其他 相关
综述 和 教程 包括 Kenter 等人 (2017), Li 和 Lu (2016),

12

介绍

Guo 等人 (2019b), Batmaz 等人 (2019), 和 Zhang 等人 (2017).
它们 所有 quite diﬀer 从 该 综述, 该 总结 现有 工作
从  视角 的 匹配 (e.g., 输入 表示 和 
方式 用于 匹配).

该 综述 聚焦 在 最先进的 匹配 技术 使用中
深度学习. 我们 期望 该  读者 具有  一定 知识
的 搜索与推荐. 那些 谁 是 不 熟悉 用 
areas 可以 参考 现有 材料 (e.g., Croft 等人 (2009), Li 和 Xu
(2014), Liu (2009), Ricci 等人 (2015), 和 Adomavicius 和 Tuzhilin
(2005)). 我们 也 假设 该  读者 具有 suﬃcient 知识 的
机器学习, 特别是 深度学习.

2

传统匹配模型

方法 用于 进行 查询-文档匹配 在 搜索 和 useritem 匹配 在 推荐 使用中 传统机器学习
技术 具有 是 提出.  方法 可以 是 formalized 之内
 更多 通用 框架, 称为 通过 我们 “学习匹配”. 除之外
搜索与推荐, 它 是 也 applicable  其他 应用
这样的 作为 复述, 问答, 和 自然语言对话. 该 节 首先 给出  形式 deﬁnition 的 学习匹配.
则, 它 介绍 传统 学习匹配 方法 开发了
用于 搜索与推荐. 最后, 它 提供 扩展阅读 在
该 方向.

2.1 学习匹配

2.1.1 匹配函数

 学习匹配 问题 可以 是 deﬁned 作为 follows. 假设
该 那里 是 二 spaces X 和 Y.  类 的 匹配 函数
F = {f (x, y)} 是 deﬁned 在 二 对象 从  二 spaces x ∈ X 和
y ∈ Y, 其中 每个 函数 f : X × Y 7→ R 表示  匹配
程度 之间  二 对象 x 和 y.  二 对象 x 和 y, 和

13

14

传统匹配模型

其 关系 可以 是 描述 用  集合 的 特征 Φ(x, y).

 匹配函数 f (x, y) 可以 是  线性 组合 的 特征:

f (x, y) = hw, Φ(x, y)i,

其中 w 是  参数 向量. 它 可以 也 是  广义 线性
模型,  树 模型, 或  神经网络.

2.1.2 学习 的 匹配 函数

监督学习 可以 是 采用了  学习  参数 的 
匹配函数 f , 作为 显示 在 图 2.1. 监督学习 用于
匹配 典型地 由组成 的 二 phases: oﬄine 学习 和 在线 匹配. 在 oﬄine 学习,  集合 的 训练实例 D =
{(x1, y1, r1), · · · , (xN , yN , rN )} 是 给定, 其中 ri 是  布尔 值 或
真实 数量 表明中  匹配程度 之间 对象 xi 和 yi,
和 N 是  大小 的 训练数据. 学习 是 进行了  选择 
匹配函数 f ∈ F 该 可以 执行  最好 在 匹配. 在
在线 匹配, 给定  测试 实例 ( 对 的 对象) (x, y) ∈ X × Y,
 学会了 匹配函数 f 是 利用了  预测  匹配
程度 之间  对象 对 表示为 作为 f (x, y).

相似  其他 监督学习 问题, 我们 可以 deﬁne 
目标 的 学习匹配 作为 minimizing  损失函数, 该 表示
如何 许多 准确率  匹配函数 可以 实现 在  训练
数据 以及  测试数据. 更多 speciﬁcally, 给定  训练数据
D,  学习 amounts  solving  以下 问题:

arg min
f ∈F

L(D, f ) + Ω(f ),

 目标 由组成 的 二 parts:  经验的 损失 L(D, f ) 测量
 整体 损失 incurred 通过  匹配函数 f 在 训练数据,
和  regularizer Ω(f ) prevents 超过ﬁtting   训练数据. Ω(f )
是 典型地 chosen  impose  惩罚 在  复杂度 的 f . 流行
regularizers 包括 ‘1, ‘2, 和  混合 的 它们.

Diﬀerent deﬁnitions 的  经验的 损失函数 L(D, f ) lead 
diﬀerent 类型 的 学习匹配 算法. 三 类型 的 损失
函数, 分别地 称为  作为 逐点 损失函数, 成对
损失函数, 和 列表级 损失函数, 具有 是 popularly 使用了 在

2.1. 学习匹配

15

图 2.1: 监督学习 用于 匹配.

 文献 (他 等人, 2017c; Rendle 等人, 2009; Nallapati, 2004;
Joachims, 2002; Cao 等人, 2006). 下一个, 我们 brieﬂy 描述  三
类型 的 损失 函数.

逐点 损失函数

 逐点 损失函数 是 deﬁned 仅 在 一 实例, i.e.,  来源
对象 和  目标 对象. 假设 该 那里 是  对 的 对象 (x, y)
用  真 匹配程度 的 r. 进一步, 假设  预测了
匹配程度 的 (x, y) 给定 通过  匹配模型 是 f (x, y). 
逐点 损失函数 是 deﬁned 作为  度量 表示中  不一致 之间  匹配 degrees, 表示为 作为 ‘指出(r, f (x, y)). 
closer f (x, y) 是  r,  更少 值  损失函数 具有.

在 学习, 给定  训练 数据集 D = {(x1, y1, r1), · · · , (xN , yN , rN )},

我们 是  minimize  总计 损失 在  训练数据, 或  sum 的
 losses 的 对象 pairs:

Lpoint(D, f ) =

N
X

i=1

‘指出(f (xi, yi), ri),

(2.1)

其中 ri 是  ground-真值 匹配程度 的 训练实例 (xi, yi).

!"!#!$%$%"%#&"&#&$'(学习 系统!%?匹配模型 *匹配 systemtraining datatest 数据̂&=*!,%…'(16

传统匹配模型

作为  例子 的  逐点 损失, 均方误差 (MSE) 是
 广泛使用 损失函数. 给定  labeled 实例 (x, y, r) 和 
匹配模型 f ,  MSE 是 deﬁned 作为:

‘MSE = (f (x, y) − r)2.

另一个 例子 是  交叉-熵 损失函数. 交叉-熵 损失
函数 假设 该 r ∈ {0, 1} 其中 1 表明 相关 和 0
否则. 它 进一步 假设 该 f (x, y) ∈ [0, 1] 是  预测了
概率 该 x 和 y 是 相关. 则,  交叉-熵 损失 是
deﬁned 作为:

‘交叉−熵 = −r log f (x, y) − (1 − r) log(1 − f (x, y)).

成对 损失函数

假设 该 那里 是 二 pairs 的 对象 (x, y+) 和 (x, y−), 用
一 的  对象 x 是 共享. 我们 称为 x 来源 对象 (e.g., 查询 或
用户) 和 y+ 和 y− 目标 对象 (e.g., 文档 或 物品). 进一步
假设 该 那里 exists  顺序 之间  对象 y+ 和 y− 给定
 对象 x, 表示为 作为 r+ (cid:31) r−. 这里 r+ 和 r− 表示  匹配
degrees 的 (x, y+) 和 (x, y−) 分别地.  顺序 relations 之间
对象 可以 是 显式地 或 隐式地 获得了.

我们 使用 f (x, y+) 和 f (x, y−)  表示  匹配 degrees 的
(x, y+) 和 (x, y−) 给定 通过  匹配模型 f , 分别地. 
成对 损失函数 是 deﬁned 作为  度量 表示中  不一致 之间  匹配 degrees 和  顺序 关系, 表示为 作为
‘对(f (x, y+), f (x, y−)).  larger f (x, y+) 是 比 f (x, y−),  更少
值  损失函数 具有.

在 学习, 给定  训练 数据集 D,  集合 的 ordered 对象

pairs P 是 derived 作为 follows:

P = {(x, y+, y−)|(x, y+, r+) ∈ D ∧ (x, y−, r−) ∈ D ∧ r+ (cid:31) r−},

 总计 经验的 损失 在  训练数据 是  sum 的  losses
超过  ordered 对象 pairs:

Lpair(P, f ) = X

‘对(f (x, y+), f (x, y−)).

(2.2)

(x,y+,y−)∈P

2.1. 学习匹配

17

我们 可以 see 该  成对 损失函数 是 deﬁned 在 ordered pairs 的
对象.

作为  例子,  成对 合页损失 是 通常地 采用. 给定
 偏好 对 (x, y+, y−) 和  匹配模型 f ,  成对
合页损失 是 deﬁned 作为

‘成对-hinge = max{0, 1 − f (x, y+) + f (x, y−)}.

另一个 共同 选择 的 成对 损失 在 推荐 是 
贝叶斯个性化排序 (BPR) 损失 (Rendle 等人, 2009), 该
旨在  maximize  间隔 之间  预测 的  正
实例 和 该 的 负 实例:

‘成对-BPR = − ln σ(f (x, y+) − f (x, y−)),

其中 σ(·) 是  Sigmoid函数.

列表级 损失函数

在 搜索与推荐,  来源 对象 (e.g.,  查询 或  用户)
是 通常 相关  多个 目标 对象 (e.g., 多个 文档 或
物品).  评估 测量 用于 搜索与推荐 通常
treat  列表 的 目标 对象 作为  整体. 它 是 reasonable, 因此, 
deﬁne  损失函数 超过  列表 的 目标 对象, 称为 列表级 损失
函数. 假设 该  来源 对象 x 是 相关  多个 目标 对象 y = {y1, y2, · · · , yN }, 和  相应 真 匹配 degrees
是 r = {r1, r2, · · · , yN }, 分别地.  预测了 匹配 degrees
通过 f 之间 x 和 y1, y2, · · · , yN 是 ˆr = {f (x, y1), · · · , f (x, yN )}.
 列表级 损失函数 是 deﬁned 作为  度量  表示  不一致 之间  真 匹配 degrees 和 预测了 匹配
degrees, 表示为 作为 ‘列表(ˆr, r).  更多  预测了 匹配 degrees
在 ˆr agree 用  真 匹配 degrees 在 r,  lower 值  损失
函数 具有. 在 学习, 给定  训练数据 D = {(xi, yi, ri)}M
i=1,
 经验的 损失函数 是 deﬁned 作为  sum 的  列表级 losses
超过  训练实例:

Llist(D, f ) = X

‘列表(ˆr, r).

(2.3)

(x,y,r)∈D

18

传统匹配模型

作为  例子 的 列表级 损失函数, 一些 方法 deﬁne 它 作为
 负 概率 的  相关 对象 给定  其他 不相关
对象. Speciﬁcally, let 我们 假设 该 那里 exists 仅 一 相关
文档 在 y 表示为 作为 y+. 则,  列表 的 labeled 对象 可以
是 written 作为 (x, y = {y+, y−
M 是  M
不相关 对象.  列表-wise 损失函数 可以 是 deﬁned 作为 
负 概率 该 y+ 是 相关 给定 x:

M }), 其中 y−

1 , · · · , y−

1 , · · · , y−

‘prob = −P (y+|x) = −

exp(λf (x, y+))
y∈y exp(λf (x, y))

P

,

其中 λ > 0 是  参数.

关系 用 学习  排序

我们 视角 学习匹配 和 学习  排序 作为 二 diﬀerent 机器
学习 问题, 虽然 它们 是 强烈 相关. 学习 
排序 (Liu, 2009; Li, 2011) 是  学习  函数 表示了 作为 g(x, y)
其中 x 和 y 可以 是 查询 和 文档 在 搜索 和 用户 和 物品
在 推荐 分别地. 在 搜索, 用于 例子,  排序
函数 g(x, y) 可以 包含 特征 关于  relations 之间 x 和
y, 以及 特征 在 x 和 特征 在 y. 在 对比,  匹配
函数 f (x, y) 仅 包含 特征 关于  relations 之间 x
和 y.

通常  匹配函数 f (x, y) 是 训练了 首先 和 则 
排序 函数 g(x, y) 是 训练了 用 f (x, y) 是  特征. 用于
排序, 确定 的  顺序 的 多个 对象 是  关键, 同时
用于 匹配, 确定 的  关系 之间 二 对象 是 
关键. 当  排序 函数 g(x, y) 仅 由组成 的  匹配
函数 f (x, y), 一 仅 needs  采用 学习匹配.

在 搜索,  特征 在 x 可以 是 语义 categories 的 查询
x 和  特征 在 y 可以 是 PageRank 分数 和 URL 长度 的
文档 y.  特征 deﬁned 通过  匹配函数 f (x, y)
可以 是 BM25 在 传统 IR 或  函数 学会了 通过 传统
机器学习 或 深度学习.  排序 函数 g(x, y) 可以 是
implemented 通过 LambdaMART (Burges, 2010) 该 是  算法
的 传统机器学习. 表 2.1 lists 一些 关键 diﬀerences
之间 学习  匹配 和 学习  排序.

2.2. 搜索与推荐中的匹配模型

19

表 2.1: 学习匹配 v.s. 学习  排序.

学习匹配

学习  排序

预测 匹配程度 之间 排序 列表 的 文档

查询 和 文档
f (x, y)

模型
挑战 失配

g(x, y1), · · · , g(x, yN )
正确 排序 在  top

最近, 研究ers 发现 该  univariate scoring 范式 在
传统 IR 是 sub-最优 因为 它 fails  捕获 inter-文档
relationships 和 局部 上下文 信息. 排序 模型 该 直接地 排序  列表 的 文档 一起 用 multivariate scoring
函数 具有 是 开发了 (Ai 等人, 2018; Bello 等人, 2018; Jiang
等人, 2019b; Pang 等人, 2020). 相似 努力 具有 是 使 在
推荐 (Pei 等人, 2019). 因此,  问题 的 匹配
和 排序 可以 是 甚至 更多 distinctively separated 在 该 sense.

2.2 搜索与推荐中的匹配模型

下一个, 我们 给出  概述 的 搜索与推荐中的匹配模型, 和 介绍  方法 的 匹配 在  潜在空间.

2.2.1 匹配 模型 在 搜索

当 应用  搜索, 学习匹配 可以 是 描述 作为 follows.
 集合 的 查询-文档 pairs D = {(q1, d1, r1), (q2, d2, r2), · · · ,
(qN , dN , rN )} 是 给定 作为 训练数据, 其中 qi, di, 和 ri (i =
1, · · · , N ) 表示  查询,  文档, 和  查询-文档匹配 程度 (相关性), 分别地. 每个 元组 (q, d, r) ∈ D 是 生成了
在  以下 方式: 查询 q 是 生成了 根据 概率
分布 P (q), 文档 d 是 生成了 根据 conditional
概率 分布 P (d|q), 和 相关性 r 是 生成了 according
 conditional 概率 分布 P (r|q, d). 该 corresponds 
 fact: 查询 是 提交   搜索 系统 独立地,
文档 associated 用  查询 是 retrieved 用  查询 词,
和  相关性 的  文档 用 respect   查询 是 determined

20

传统匹配模型

通过  contents 的  查询 和 文档. 人类 标注数据 或
点击数据 可以 是 使用了 作为 训练数据.

 目标 的 学习匹配 用于 搜索 是  自动地 学习
 匹配模型 表示了 作为  scoring 函数 f (q, d) (或 作为 
conditional 概率 分布 P (r|q, d)).  学习 问题
可以 是 formalized 作为 最小化 的  逐点 损失函数 在
方程 (2.1),  成对 损失函数 在 方程 (2.2), 或 
列表级 损失函数 在 方程 (2.3).  学会了 模型 必须 具有
 泛化 能力  进行 匹配 在 unseen 测试数据.

2.2.2 匹配 模型 在 推荐

当 应用  推荐, 学习匹配 可以 是 描述
作为 follows.  集合 的 M 用户 U = {u1, · · · , uM } 和  集合 的 N 物品
V = {i1, · · · , 在 }, 以及  评分矩阵 R ∈ RM ×N 是 给定,
其中 每个 条目 rij 表示  rating (交互) 的 用户 ui 在
物品 ij 和 rij 是 集合  零 如果  rating (交互) 是 未知.
我们 假设 该 每个 元组 (ui, ij, rij) 是 生成了 在  以下
方式: 用户 ui 是 生成了 根据 概率 分布 P (ui),
物品 ij 是 生成了 根据 概率 分布 P (ij), 和
rating rij 是 生成了 根据 conditional 概率 分布
P (rij|ui, ij). 该 corresponds   fact: 用户 和 物品 是 呈现了
在  推荐 系统, 和  兴趣 的  用户 在  物品 是
determined 通过  已知 兴趣 的 用户 在 物品 在  系统.

 目标 的 学习匹配 用于 推荐 是  学习 
底层 匹配模型 f (ui, ij) 该 可以 使 predictions 在 
ratings (interactions) 的  零 entries 在 矩阵 R:

ˆrij = f (ui, ij),

其中 ˆrij 表示  估计 ﬃnity 分数 之间 用户 ui 和 物品
ij. 在 该 方式, 给定  用户,  子集 的 物品 用  highest 分数
用 respect   用户 可以 是 recommended.  学习 问题
可以 是 formalized 作为 minimizing  regularized 经验的 损失函数.
仍然,  损失函数 可以 是 或 逐点 损失, 成对 损失, 或
列表级 损失 作为 在 方程 (2.1), 方程 (2.2), 或 方程 (2.3). 如果
 损失函数 是 逐点 损失 如 square 损失 或 交叉-熵, 

2.2. 搜索与推荐中的匹配模型

21

模型 学习 成为  回归 或 classiﬁcation 问题, 其中 
预测 值 表明  强度 的 兴趣. 如果  损失函数
是 成对 损失 或 列表级 损失, 它 成为  genuine 排序 问题,
其中  预测 值 表明  relative strengths 的 兴趣
在 物品 用于  用户.

2.2.3 匹配 在 潜在空间

作为 解释了 在 章 1,  基本 挑战  匹配
在 搜索与推荐 是  失配 之间 对象 从
二 diﬀerent spaces (查询 和 文档, 以及 用户 和 物品).
一 eﬀective 方法  dealing 用  挑战 是  表示
 二 对象 在 匹配 在  共同空间, 和  执行 
任务 的 匹配 在  共同空间. 作为  空间 可以 不 具有 
显式 deﬁnition, 它 是 经常 称为  作为 “潜在空间”. 该 是 
基本 想法 在之后  方法 的 匹配 在  潜在空间,
用于 两者 搜索 (Wu 等人, 2013b) 和 推荐 (Koren 等人,
2009b).

无 损失 的 generality, let 我们 取 搜索 作为  例子. 图 2.2 说明 查询-文档匹配 在  潜在空间. 那里
是 三 spaces: 查询空间, 文档空间, 和 潜在空间, 和
那里 exist 语义 gaps 之间  查询空间 和 文档空间.
查询 和 文档 是 首先 映射   潜在空间, 和 则
匹配 是 进行了 在  潜在空间. 二 映射 函数
specify  mappings 从  查询空间 和 文档空间 到 
潜在空间.  使用 的 diﬀerent 类型 的 映射 函数 (e.g., 线性
和 non-线性) 和 相似度 测量 在  潜在空间 (e.g., inner
乘积 和 欧氏距离) lead  diﬀerent 类型 的 匹配
模型.

形式化地, let Q 表示  查询空间 (查询 q ∈ Q) 和 D 表示
 文档空间 (文档 d ∈ D), 分别地, 和 H 表示
 潜在空间.  映射函数 从 Q  H 是 表示了 作为
φ : Q 7→ H, 其中 φ(q) stands 用于  映射 向量 的 q 在 H. 类似地,
 映射函数 从 D  H 是 表示了 作为 φ0 : D 7→ H, 其中
φ0(d) stands 用于  映射 向量 的 d 在 H.  匹配分数
之间 q 和 d 是 deﬁned 作为  相似度 之间  映射 vectors

22

传统匹配模型

图 2.2: 查询-文档匹配 在 潜在空间.

(表示) 的 q 和 d 在  潜在空间, i.e., φ(q) 和 φ0(d).

在之前  prevalence 的 深度学习, 大多数 方法 是 “浅层”,
在  sense 该 线性 函数 和 内积 是 采用 作为 
映射函数 和 相似度, 分别地,

s(q, d) = hφ(q), φ0(d)i,

(2.4)

其中 φ 和 φ0 表示 线性 函数 和 h·i 表示 内积.
在 学习 的  模型, 训练实例 表明中  匹配
relations 之间 查询 和 文档 是 给定. 用于 例子, clickthrough 数据 可以 是 自然地 使用了.  训练数据 是 表示了 作为
(q1, d1, c1), (q2, d2, c2), · · · , (qN , dN , cN ), 其中 每个 实例 是  三元组
的 查询, 文档, 和 点击-数量 (或 对数 的 点击-数量).

2.3 搜索中的潜在空间模型

下一个, 我们 介绍  匹配 模型 基于 在 潜在 spaces 作为
例子.  完整 介绍  语义匹配 在 搜索
可以 是 发现 在 (Li 和 Xu, 2014). Speciﬁcally, 我们 brieﬂy 介绍
代表性 方法 用于 搜索 该 执行 匹配 在  潜在
空间, 包括中 偏最小二乘 (PLS) (Rosipal 和 Krämer, 2006),
正则化潜在空间匹配 (RMLS) (Wu 等人, 2013b), 和
监督语义索引 (SSI) (Bai 等人, 2009; Bai 等人, 2010).

2.3.1 偏最小二乘

偏最小二乘 (PLS) 是  技术 初始 提出 用于 回归 在 统计 (Rosipal 和 Krämer, 2006). 它 是 显示 该 PLS 可以

MappingqdQuery spaceDocument spaceLatent 空间!!"!#!"$2.3. 搜索中的潜在空间模型

23

是 采用了 在 学习 的 潜在空间 模型 用于 搜索 (Wu 等人,
2013).

Let 我们 考虑 使用中  匹配函数 f (q, d) 在 方程 (2.4).
Let 我们 也 假设 该  映射 函数 是 deﬁned 作为 φ(q) = Lqq
和 φ0(d) = Ldd, 其中 q 和 d 是 特征 vectors 表示中 查询
q 和 文档 d, 和 Lq 和 Ld 是 标准正交 矩阵. 因此, 
匹配函数 成为

f (q, d) = hLqq, Lddi.

(2.5)

其中 Lq 和 Ld 是  是 学会了.

给定  训练数据,  学习 的 Lq 和 Ld amounts  optimizing  目标 函数 (基于 在 逐点 损失) 用 constraints:

arg max
Lq,Ld

s.t. LqLT

cif (qi, di),

= X
(qi,di)
q = I, LdLT

d = I,

(2.6)

其中 (qi, di) 是  对 的 查询 和 文档, ci 是  点击 数量 的
 对, 和 I 是  identity 矩阵. 该 是  非凸优化
问题, 然而,  全局 optimum exists 和 可以 是 实现了 通过
采用中 SVD (Singular 值 分解) (Wu 等人, 2013; Wu
等人, 2013b).

2.3.2 Regularized 映射  潜在空间

PLS 假设 该  映射 函数 是 标准正交 矩阵.
当  训练数据 大小 是 大, 学习 成为 hard 因为
它 needs  solve SVD, 该 是 的 高 时间复杂度.  解决
 问题, Wu 等人 (2013b) 提出  新 方法 称为 正则化潜在空间匹配 (RMLS), 在 该  orthonormality
constraints 在 PLS 是 replaced 用 ‘1 和 ‘2 regularizations, 之下
 假设 该  解决方案 是 稀疏. 在 该 方式, 那里 是 无
需求  solve SVD, 和  优化 可以 是 carried out eﬃciently.
Speciﬁcally,  优化 问题 成为 该 的 minimizing 

24

传统匹配模型

目标 函数 (基于 在 逐点 损失) 用 ‘1 和 ‘2 constraints:

arg max
Lq,Ld

cif (qi, di),

= X
(qi,di)
|lj
d| ≤ θd, klj

s.t. ∀j :

|lj
q| ≤ θq,

qk ≤ τq, klj

dk ≤ τd,

(2.7)

其中 (qi, di) 是  对 的 查询 和 文档, ci 是  点击 数量 的
 对, Lq 和 Ld 是 线性 映射 矩阵, lj
d 是  j-th
row vectors 的 Lq 和 Ld, 和 θq, θd, τq, 和 τd 是 thresholds. | · | 和
k · k 表示 ‘1 和 ‘2 norms, 分别地. 注意 该  regularizations
是 deﬁned 在  row vectors, 不 column vectors.  使用 的 ‘2 范数
是  避免  平凡 解决方案.

q 和 lj

 学习 在 RMLS 是 也  非凸优化 问题.
那里 是 无 guarantee 该  globally 最优 解决方案 可以 是 发现.
一 方式  cope 用  问题 是  采用 替代 优化,
该 是,  首先 ﬁx Lq 和 optimize Ld, 和 则 ﬁx Ld 和 optimize Lq,
和 repeat 直到 收敛. 一 可以 easily see 该  优化
可以 是 decomposed 和 执行了 row 通过 row 和 column 通过 column
的  矩阵. 该 means 该  学习 在 RMLS 可以 是 easily
parallelized 和 缩放 向上.

 匹配函数 在 方程 (2.5) 可以 是 rewritten 作为 

双线性 函数

f (q, d) =(Lqq)T (Ldd)
=qT (LT
q Ld)d
=qT Wd,

(2.8)

其中 W = LT
q Ld. 因此, 两者 PLS 和 RMLS 可以 是 viewed 作为 
方法 的 学习  binear 函数 用 矩阵 W 该 可以 是
factorized 到 二 低-排序 矩阵 Lq 和 Ld.

2.3.3 监督语义索引

 special 假设 可以 是 使 在 PLS 和 RMLS; 该 是,  查询
空间 和  文档空间 具有  相同 dimensions. 用于 例子,
当 两者 查询 和 文档 是 表示了 作为 bag-的-词, 它们
具有  相同 dimensions 在  查询 和 文档 spaces. 作为 
结果, W 在 方程 (2.8) 成为  square 矩阵.  方法 的

2.4. 推荐中的潜在空间模型

25

监督语义索引 (SSI) 提出 通过 (Bai 等人, 2009; Bai
等人, 2010) 确切地 使  假设. 它 进一步 表示 W 作为
 低 排序 和 diagonal preserving 矩阵:

W = LT

q Ld + I,

其中 I 表示  identity 矩阵. 因此,  匹配函数
成为

f (q, d) = qT (LT

q Ld + I)d.

 添加 的  identity 矩阵 means 该 SSI 使  tradeoﬀ
之间  使用 的  低-维 潜在空间 和  使用 的 
经典 向量空间模型 (VSM)1.  diagonal 的 矩阵 W 给出
 分数  每个 术语 该 occurs 在 两者 查询 和 文档.

给定 点击数据, ordered 查询-文档 pairs 是 首先
1 , d−
derived, 表示为 作为 P = {(q1, d+
M )} 其中 d+ 是
ranked 更高 比 d− 和 M 是  数量 的 pairs.  目标 的
学习 是  选择 Lq 和 Ld 这样的 该 f (q, d+) > f (q, d−) holds 用于
所有 pairs.  成对 损失函数 是 利用了.  优化 问题
成为

1 ), · · · , (qM , d+

M , d−

arg min
Lq,Ld

= arg min
Lq,Ld

X

(q,d+,d−)∈P
X

(q,d+,d−)∈P

max(0, 1 − (f (q, d+) − f (q, d−))),

max(0, 1 − qT (LT

q Ld + I)(d+ − d−)).

(2.9)

 学习 的 SSI 是 也  非凸优化 问题 和 那里
是 无 guarantee  发现  全局 最优 解决方案.  优化
可以 是 conduced 在  方式 相似  该 的 RMLS.

2.4 推荐中的潜在空间模型

下一个, 我们 brieﬂy 介绍 代表性 方法 用于 推荐 该 执行 匹配 在  潜在空间, 包括中 偏置矩阵分解 (BMF) (Koren 等人, 2009b), 因子化物品相似度模型 (FISM) (Kabbur 等人, 2013), 和 分解机 (FM) (Rendle, 2010).

1如果 W = I, 则  模型 degenerates  VSM. 如果 W = LT

q Ld, 则  模型

是 等价   模型 的 PLS 和 RMLS.

26

传统匹配模型

2.4.1 偏置矩阵分解

偏置矩阵分解 (BMF) 是  模型 提出 用于 预测中
 ratings 的 用户 (Koren 等人, 2009b), i.e., formalizing 推荐 作为  回归 任务. 它 是 开发了 期间  时期 的 Netﬂix
挑战 和 快速 成为 流行 due  其 简洁性 和 eﬀectiveness.  匹配模型 可以 是 formulated 作为:

f (u, i) = b0 + bu + bi + pT

u qi,

(2.10)

其中 b0, bu, 和 bi 是 scalars 表示中  整体 偏置, 用户 偏置, 和
物品 偏置 在 rating 分数, 和 pu 和 qi 是 潜在 vectors 表示中
 用户 和  物品. 该 可以 是 interpreted 作为 仅 使用中  IDs 的
用户 和 物品 作为 特征 的 它们, 和 projecting  IDs 到 
潜在空间 用 二 线性 函数. Let u 是  一-hot ID 向量
的 用户 u 和 i 是  一-hot ID 向量 的 物品 i, 和 P 是  用户
投影 矩阵 和 Q 是  物品 投影 矩阵. 则 我们 可以
express  模型 之下  映射 框架 的 方程 (2.4):

f (u, i) = hφ(u), φ0(i)i = h[b0, bu, 1, P · u], [1, 1, bi, Q · i]i,

(2.11)

其中 [·, ·] 表示 向量 concatenation.

给定  训练数据,  学习 的 模型 参数 (Θ =
{b0, bu, bi, P, Q} ) 成为 optimizing  逐点 回归 误差
用 正则化:

arg min

Θ

X

(u,i)∈D

(Rui − f (u, i))2 + λ||Θ||2,

(2.12)

其中 D 表示 所有 观察到 ratings, Rui 表示  rating 用于 (u, i),
和 λ 是  L2 正则化 coeﬃcient. 作为 它 是  非凸优化 问题, alternating 至少 squares (他 等人, 2016b) 或 随机
梯度 decent (Koren 等人, 2009b) 是 典型地 采用了, 该
cannot guarantee  发现  全局 optimum.

2.4.2 因子化物品相似度模型

因子化物品相似度模型 (FISM) (Kabbur 等人, 2013) adopts 
假设 的 物品-基于 协同 过滤ing, i.e., 用户 将 prefer
物品 该 是 相似  什么 它们 具有 chosen 因此 far.  进行 因此, FISM

2.4. 推荐中的潜在空间模型

27

使用  物品 该  用户 具有 chosen  表示  用户 和 projects
 结合 物品 到  潜在空间.  模型 形式化 的
FISM 是:

f (u, i) = bu + bi + d−α

pj)T qi,

(2.13)

u ( X
j∈D+
u

其中 D+
u 表示  物品 该 用户 u 具有 chosen, du 表示 
数量 的 这样的 物品, 和 d−α
表示 归一化 跨越 用户.
qi 是  潜在向量 的 目标 物品 i, 和 pj 是  潜在向量 的
历史 物品 j chosen 通过 用户 u. FISM treats pT
j qj 作为  相似度
之间 物品 i 和 j, 和 aggregates  相似之处 的  目标 物品
i 和  历史 物品 的 用户 u.

u

FISM 采用  成对 损失 和 学习  模型 从 二进制 隐式

反馈. Let U 是 所有 用户,  总计 成对 损失 是 给定 通过

X

X

X

u∈U

i∈D+
u

j /∈D+
u

(f (u, i) − f (u, j) − 1)2 + λ||Θ||2,

(2.14)

该 forces  分数 的  正 (观察到) 实例  是 larger 比
该 的  负 (unobserved) 实例 用  间隔 的 一. 另一个
成对 损失,  贝叶斯个性化排序 (BPR) (Rendle 等人,
2009) 损失, 是 也 广泛使用,

X

X

X

u∈U

i∈D+
u

j /∈D+
u

− ln σ(f (u, i) − f (u, j)) + λ||Θ||2,

(2.15)

其中 σ(·) 表示  Sigmoid函数 该 converts  diﬀerence
的 分数   概率 值 之间 零 和 一, 和 因此 
损失 具有  概率 解释.  主要 diﬀerence 之间 
二 losses 是 该 BPR enforces  diﬀerences 之间 正 和
负 实例 作为 大 作为 可能, 无 显式地 deﬁning 
间隔. 两者 成对 losses 可以 是 seen 作为  surrogate 的  AUC
度量, 该 测量 如何 许多 pairs 的 物品 是 correctly ranked
通过  模型.

2.4.3 分解机

分解机 (FM) (Rendle, 2010) 是 开发了 作为  通用
模型 用于 推荐. 除之外  交互 信息

28

传统匹配模型

之间 用户 和 物品, FM 也 incorporates 辅助信息 的 用户
和 物品, 这样的 作为 用户 proﬁles (e.g., age, gender, etc.), 物品 attributes
(e.g., 类别, 标签, etc.) 和 contexts (e.g., 时间, 位置, etc.). 
输入  FM 是  特征 向量 x = [x1, x2, ..., xn] 该 可以 包含
任何 特征 用于 表示中  匹配函数, 作为 描述 以上.
因此, FM casts  匹配 问题 作为  监督学习
问题. 它 projects  特征 到  潜在空间, 建模 其
interactions 用  内积:

f (x) = b0 +

n
X

i=1

bixi +

n
X

n
X

i=1

j=i+1

vT

i vjxixj,

(2.16)

其中 b0 是  偏置, bi 是  权重 的 特征 xi, 和 vi 是  潜在
向量 的 特征 xi. 给定  fact 该  输入 向量 x 可以 是 大
但 稀疏, e.g., 多-hot 编码 的 categorical 特征, FM 仅
捕获 interactions 之间 non-零 特征 (用 术语 xixj).

FM 是  very 通用 模型 在  sense 该 feeding diﬀerent 输入
特征 到  模型 将 lead  diﬀerent formulations 的  模型.
用于 例子, 当 x 仅 retains  用户 ID 和 目标 物品 ID, FM
成为  BMF 模型; 和 当 x 仅 keeps  IDs 的 用户’s
历史上 chosen 物品 和 目标 物品 ID, FM 成为  FISM
模型. 其他 prevalent 潜在空间 模型 这样的 作为 SVD++ (Koren,
2008) 和 Factorized Personalized Markov Chain (FPMC) (Rendle et
al., 2010) 可以 也 是 subsumed 通过 FM 用 适当 特征工程.

2.5 扩展阅读

查询 重新形式化 是 另一个 方式  解决  查询-文档
失配 在 搜索, 该 是,  变换  查询  另一个 查询
该 可以 进行 更好 匹配. 查询 变换 包括 spelling
误差 修正 的  查询. 用于 例子, Brill 和 Moore (2000)
提出  来源 channel 模型, 和 Wang 等人 (2011) 提出 
判别 方法 用于  任务. 查询 变换 也 包括
查询 分割 (Bergsma 和 Wang, 2007; Bendersky 等人, 2011;
Guo 等人, 2008). Inspired 通过 统计 机器 翻译 (SMT),
研究ers 也 考虑 leveraging 翻译 technologies  deal 用
查询 文档 失配, 假设 该 查询 是 在 一 语言 和

2.5. 扩展阅读

29

文档 是 在 另一个. Berger 和 Laﬀerty (1999) 利用  词-基于
翻译 模型  执行  任务. Gao 等人 (2004) 提出 使用中
 短语-基于 翻译 模型  捕获  dependencies 之间
词 在  查询 和 文档 title. 主题 模型 可以 也 是 利用了
 解决  失配 问题.  简单 和 eﬀective 方法 是 
使用  线性 组合 的 术语 匹配分数 和 主题 匹配
分数 (Hofmann, 1999). 概率 主题 模型 是 也 采用了 
平滑 文档 语言 模型 (或 查询 语言 模型) (Wei
和 Croft, 2006; Yi 和 Allan, 2009). Li 和 Xu (2014) 提供 
全面 综述 在  传统机器学习 方法
 语义匹配 在 搜索.

在 推荐, 除之外  经典 潜在因子 模型 介绍, 其他 类型 的 方法 具有 是 开发了. 用于 例子,
匹配 可以 是 进行了 在  原始 交互 空间 用 predeﬁned heuristics, 如 物品-基于 CF (Sarwar 等人, 2001) 和 uniﬁed
用户-基于 和 物品-基于 CF (Wang 等人, 2006). 用户-物品 interactions 可以 是 组织 作为  bipartite 图, 在 该 随机 walk 是
执行了  estimate  相关性 之间 任何 二 nodes ( 用户
和  物品, 二 用户, 或 二 物品) (他 等人, 2017b; Eksombatchai
等人, 2018). 一 可以 也 模型  生成 过程 的 用户-物品
interactions 使用中 概率 graphical 模型 (Salakhutdinov 和
Mnih, 2007).  incorporate 各种 辅助信息 这样的 作为  用户
proﬁles 和 contexts, 除之外  FM 模型 介绍, 张量 factorization (Karatzoglou 等人, 2010) 和 集体 矩阵分解 (他
等人, 2014) 是 也 exploited. 我们 指代  读者  二 综述 papers
在  传统 匹配 方法 用于 推荐 (Adomavicius
和 Tuzhilin, 2005; Shi 等人, 2014).

3

面向匹配的深度学习

最近 年 具有 观察到 巨大 进展 在 应用 的
深度学习 到 匹配 在 搜索与推荐 (Guo 等人,
2019b; Naumov 等人, 2019).  主要 原因 用于  成功 是 due 
深度学习’s 强 能力 在 学习 的 表示 用于 输入
(i.e., 查询, 文档, 用户, 和 物品) 和 学习 的 非线性
函数 用于 匹配. 在 该 章, 我们 首先 给出  概述 的 深度
学习 (DL) 技术 和 则 描述  通用 框架, 典型
架构, 和 designing principles 的 面向匹配的深度学习 在
搜索与推荐.

3.1 深度学习概述

3.1.1 深度 神经 网络

深度 神经 网络 是 复杂 非线性 函数 从 输入 
输出. 在 该 节, 我们 描述 若干 神经网络 架构
该 是 广泛使用. Please 指代  (Goodfellow 等人, 2016) 用于 
更多 详细 介绍.

30

3.1. 深度学习概述

31

馈-前向 神经 网络

 馈-前向 神经 网络 (FFN), 也 称为 Multilayer Perceptron (MLP), 是 神经 网络 consisting 的 多个 层 的
单元, 该 是 connected 层 通过 层 无  循环. 它 是 称为
馈-前向 因为  信号 仅 moves 在 一 方向 在  网络, 从  输入 层, 通过  隐藏 层, 和 ﬁnally 
 输出 层.  馈-前向 神经 网络 可以 是 利用了 
近似 任何 函数, 用于 例子,  regressor y = f (x) 该 maps
 向量 输入 x   标量 输出 y.

图 3.1 显示  feedforward 神经网络 用 一 隐藏层.
用于  输入 向量 x,  神经网络 returns  输出 向量 y.
 模型 是 deﬁned 作为  以下 non-线性 函数

y = σ (W2 · σ (W1 · x + b1) + b2) ,

其中 σ 是  元素-wise Sigmoid函数, W1, W2, b1 和 b2 是
模型 参数  是 determined 在 学习.  构建  deeper
神经网络, 一 仅 needs  stack 更多 层 在  top 的 
网络. 除之外 Sigmoid函数, 其他 函数 这样的 作为 tanh 和
Rectiﬁed 线性 单元 (ReLU) 是 也 利用了.

在 学习, 训练数据 的 输入-输出 pairs 是 fed 到 
网络 作为 ground-真值.  损失 是 计算了 用于 每个 实例 通过
contrasting  真实值 和  预测 通过  网络, 和
 训练 是 执行了 通过 adjusting  参数 因此 该 
总计 损失 是 minimized.  好-已知 back-propagation 算法 是
采用了  进行  最小化.

卷积 神经 网络

卷积 神经 网络 是 神经 网络 该 使 使用 的
卷积 operations 在 在 至少 一 的  层. 它们 是 专门
神经 网络 用于 processing 数据 该 具有  grid-如 结构, e.g.,
时间 serious 数据 (1-D grid 的 时间 间隔) 和 图像 数据 (2-D grid
的 pixels).

作为 显示 在 图 3.2,  典型 卷积 网络 由组成 的
多个 stacked 层: 卷积 层, detector 层, 和 池化
层. 在  卷积 层, 卷积 函数 是 应用 在

32

面向匹配的深度学习

图 3.1:  简单 前馈神经网络.

并行  产生  集合 的 线性 activations. 在  detector 层, 
集合 的 线性 activations 是 run 通过  非线性 激活函数.
在  池化 层, 池化 函数 是 使用了  进一步 modify 
集合 的 输出.

图 3.3 显示  比较 之间  完全地 connected 层 和
 卷积 层.  完全地 connected 层 使用  权重 矩阵
 模型  全局 特征 的 所有 输入 单元, 和 因此 它 具有
稠密 connections 之间 层.  卷积 层, 在  其他
hand, 使用 卷积 核 vectors (或 矩阵)  模型  局部
特征 的 每个 位置 (单元), 其中  weights 的 kernels 是 共享
跨越 positions (单元). 因此 它 具有 许多 稀疏 connections 之间
层. Speciﬁcally, 在 一-维度, 给定 卷积 核 w 和
输入 向量 x,  输出 的 位置 (单元) i 是 decided 作为:
yi = (x ∗ w)(i) = X

xi−awa,



其中 ‘*’ 表示  卷积 operator.

在  二-维度 案例, 给定  卷积 核 K 和 

输入 矩阵 X,  输出 的 位置 (单元) (i, j) 是 decided 作为:

Yi,j = (X ∗ W)(i, j) = X

X

Xi−,j−bWa,b.



b

inputhiddenoutputb1b2W1W2xy3.1. 深度学习概述

33

图 3.2: 典型 卷积 层.

在  秒 detector 层, 非线性 activation 函数 这样的 作为

sigmoid, tanh, 和 ReLU 是 通常 利用了.

在  第三 池化 层, 池化 函数 这样的 作为 max-池化,
average-池化, 和 min-池化 是 exploited. 用于 例子, 在 maxpooling,  输出 在 每个 位置 是 determined 作为  最大值 的
输出 的  核 在  neighborhood.

Recurrent 神经 网络

Recurrent 神经 网络 是 神经 网络 用于 processing 序列
数据 x(1), · · · , x(T ). Unlike FFNs 该 可以 仅 处理 一 实例
在  时间, RNN 可以 处理  长 序列 的 实例 用  变量
长度.

作为 显示 在 图 3.4,  RNN shares  相同 参数 在
diﬀerent positions. 该 是, 在 每个 位置,  输出 是 从 
相同 函数 的 输入 在  当前 位置 以及 输出 在 
之前 位置.  输出 是 determined 用  相同 规则 跨越
positions. Speciﬁcally, 在 每个 位置 t = 1, · · · , T ,  输出 向量

输入  layersconvolutional layerdetector layere.g.,rectified linearpooling layeroutput  下一个 层34

面向匹配的深度学习

图 3.3: 完全地 connected 层 和 卷积 层.

o(t) 和  陈述 的  隐藏 单元 h(t) 是 计算了 作为

h(t) = tanh(Wh(t−1) + Ux(t) + b1),
o(t) =softmax(Vh(t) + b2),

其中 W, U, V, b1, 和 b2 是 模型 参数.

Unfolding  RNN, 我们 获得  深度神经网络 用 许多
stages (正确 部分 的 图 3.4) 该 share 参数 超过 所有 stages.
 数量 的 stages 是 determined 通过  长度 的  输入 序列.
该 结构 使 它 有挑战  学习  模型 的 RNN, 因为
 gradients propagated 超过  positions 可以 或 vanish 或 explode.
 解决  问题, variations 的 RNN 这样的 作为 长 短 术语
记忆 (LSTM) 和 Gated Recurrent 单元 (GRU) 是 提出.

注意力-基于 神经 网络

注意力 是  有用 工具 在 深度学习. 它 是 最初 提出
 dynamically 和 selectively collect 信息 从  来源
句子 在  编码器-解码器 模型 在 神经 机器 翻译
(NMT) (Bahdanau 等人, 2015).
注意力 基于 模型: 图 3.5 显示  编码器-解码器 模型
用  additive 注意力机制. 假设 该 那里 是 

fullyconnected layerdense 交互, 用 不同 weightsconvolutional layersparse 交互, 用 sharedweightsmove forwardinput xfeature map ykernel3.1. 深度学习概述

35

图 3.4:  RNN 和 其 unfolded 形式.

输入 序列 (w1, w2, · · · , wM ) 的 长度 M 和  输出 序列
(y1, y2, · · · , yN ) 的 长度 N .  编码器 (e.g.,  RNN) 创建 
隐藏 陈述 hi 在 每个 输入 位置 wi(i = 1, · · · , M ).  解码器
构建  隐藏 陈述 st = f (st−1, yt−1, ct) 在 输出 位置 t
(t = 1, · · · , N ), 其中 f 是  函数 的  解码器, st−1 和 yt−1
是  陈述 和 输出 的  之前 位置, 和 ct 是  上下文
向量 在  位置.  上下文 向量 是 deﬁned 作为  sum 的
隐藏 陈述 在 所有 输入 positions, weighted 通过 注意力 分数:

ct =

M
X

i=1

αt,ihi,

和  注意力 分数 αt,i 是 deﬁned 作为:

αt,i =

exp(g(st, hi))
j=1 exp(g(st, hj))

PM

.

 函数 g 是 determined 通过  隐藏 陈述 的  之前 输出
位置 和  上下文 向量 的  当前 输出 位置. 它 可以
是 deﬁned 作为, 用于 例子,  馈-前向 网络 用  单一 隐藏
层:

g(st, hi)) = vT
其中 va 和 Wa 是 参数.

 tanh(Wa[st, hi]),

我们 可以 see 该  上下文 向量 ct selectively 和 dynamically
组合  信息 的  整个 输入 序列 用  注意力

输入 x(t)隐藏 h(t)unfoldhidden…inputx(1)x(2)x(3)x(T)…Uoutput o(t)o(1)o(2)o(3)o(T)WVWWWWUUUUVVVV36

面向匹配的深度学习

图 3.5:  编码器-解码器 模型 用 additive 注意力机制.

机制. 比较了   传统 编码器-解码器 模型 在
该 仅  单一 向量 是 使用了, 多个 vectors 是 使用了  捕获
 信息 的  编码器 无论 的  distance.
Transformer: Transformer (Vaswani 等人, 2017) 是 另一个 attentionbased 神经网络 之下  编码器 和 解码器 框架. Diﬀerent 从  前述 模型 该 顺序地 reads  输入
序列 (左--正确 或 正确--左), Transformer reads  整个
输入 序列 在 一旦.  特性 使能 它  学习  模型
通过 考虑中 两者  左 和  正确 上下文 的  词.

作为 显示 在 图 3.6, Transformer 由组成 的  编码器 用于 transforming  输入 序列 的 词 到  序列 的 vectors (internal
表示) 和  解码器 用于 生成中  输出 序列 的
词 一 通过 一 给定  internal 表示.  编码器 是 
stack 的 编码器 组件 用  完全相同 结构, 和  解码器 是 也  stack 的 解码器 组件 用  完全相同 结构,
其中  编码器 和 解码器 具有  相同 数量 组件.

每个 编码器 组件 或 层 由组成 的  自身-注意力 sublayer 和  馈-前向 网络 sub-层. 它 receives  序列 的
vectors (packed 到  矩阵) 作为 输入, processes  vectors 用
 自身-注意力 sub-层, 和 则 passes 它们 通过  feedencoderattentionlayerdecoderw"⋯⋯s1st+1st⋯𝛼%,"𝐡"𝐡(𝐡)𝛼%,(𝛼%,)⋯y"y%y%+"⋯⋯w(w)sourcetargetcontextvector3.1. 深度学习概述

37

图 3.6:  例子 Transformer 用 二-层 的 编码器 和 解码器. 
编码器 是 responsible 用于 创建中 internal 表示 的 输入 词. 
解码器 是 应用 多个 次  生成  输出 词 一 通过 一. 注意
该  residual connections 大约  sub-层 是 不 显示 在  ﬁgure.

前向 网络 sub-层. 最后, 它 sends  vectors 作为 输出 
 下一个 编码器 组件. Speciﬁcally,  输入 是  序列 的
词 (w1, w2, · · · , wM ) 用 长度 M . 每个 词 wi 是 表示了
通过  向量 xi 作为 sum 的  词嵌入 和 positional 编码
的 它.  vectors 是 packed 到  矩阵 X(0) = [x1, x2, · · · , xM ]T .
 自身-注意力 sub-层 converts X(0) 到 Z = [z1, z2, · · · , zM ]T
通过 自身-注意力 deﬁned 作为

Z = 注意力(Q, K, V) = softmax

!

V,

QKT
√
dk

其中 K, V, 和 Q 是 矩阵 的 关键 vectors, 值 vectors, 和 查询
vectors 分别地; dk 是  维度 的 关键 向量; K 是 
resulting 矩阵 consisting 的 M vectors.  矩阵 K, V, 和 Q

DecoderEncodercomponent#2positionalencodingwordembedding⋯⋯⋯+++⋯𝐗($)自身-注意力===𝐱’𝐱(𝐱)层 normalizationlayer normalizationFFNFFNFFN𝐳’𝐳(𝐳)𝐗(’)Encodercomponent#1⋯𝐗(()w1w2wMEncoder⋯自身-attentionlayer normalizationlayer normalizationFFNFFNFFNpositionalencodingencoder-decoderattentionlayer 归一化++Decodercomponent #2linearandsoftmaxyt(t=1,2,…,N)y1yt-1wordembedding[开始]+⋯Decodercomponent #1𝐊=𝐕=𝐗(() 
38

面向匹配的深度学习

是 计算了 作为

Q = X(0)WQ,
K = X(0)WK,
V = X(0)WV ,

其中 WQ, WK 和 WV 是 嵌入 矩阵. 之后 该,  vectors zi 在 Z 是 独立地 processed 通过  馈-前向 网络
sub-层. 在 每个 sub-层,  残差连接 是 采用了, followed
通过 层-归一化.  输出 的  编码器 组件 是 表示了 作为 X(1) = [x(1)
M ]T . X(1) 是 则 fed 到  下一个
编码器 组件.  编码器 ﬁnally 输出  vectors (表示) 相应  所有 输入 词, 表示为 作为 Xenc. 那里 是
多个 heads 在 每个 注意力 sub-层 和 我们 omit  描述
在 它 这里.

2 , · · · , x(1)

1 , x(1)

每个 解码器 组件 或 层 在  解码器 由组成 的  selfattention sub-层,  编码器-解码器 注意力 sub-层, 和  feedforward 网络 sub-层.  sub-层 具有  相同 架构 作为
该 的  编码器 组件. 之后 编码,  输出 的  编码器
是 使用了  表示  关键 和 值 vectors: K = V = Xenc, 该
是 则 使用了 用于 “编码器-解码器 注意力” 在 每个 解码器 组件.  解码器 顺序地 生成 词 用于 所有 输出 positions
1, 2, · · · , N . 在 每个 位置 1 ≤ t ≤ N ,  bottom 解码器 组件
receives  之前 outputted 词 “[开始], y1, · · · , yt−1”, masks
 未来 positions, 和 输出 internal 表示 用于  下一个
解码器 组件. 最后,  词 在 位置 t, 表示为 作为 vt, 是
selected 根据  概率 分布 生成了 通过  softmax 层 在  top 解码器 组件.  过程 是 repeated 直到
 special 符号 (e.g., “[端]”) 是 生成了, 或  maximal 长度 是
r每个ed.

AutoEncoders

AutoEncoders 是 神经 网络 该 aim  学习  隐藏 信息 的  输入, 通过 compressing  输入 到  潜在-空间
表示 和 则 reconstructing  输出 从  表示. 在  模型, 高-维 数据 是 首先 converted 到 

3.1. 深度学习概述

39

图 3.7: 架构 的  AutoEncoders

低-维 潜在 表示 通过  multilayer 编码器 神经
网络. 则,  数据 是 重建 从  潜在 表示
通过  multilayer 解码器 神经网络. 因此, 它 由组成 的 二
parts:  编码器 y = f (x); 和  解码器 ˆx = g(y).  自编码器
作为  整体 可以 是 描述 通过  函数 g(f (x)) = ˆx 其中 它 是
expected 该 ˆx 是 作为 close   原始 输入 x 作为 可能.
Vanilla 自编码器 (Hinton 和 Salakhutdinov, 2006): 图 3.7
显示  架构 的  vanilla 自编码器 模型.  编码器
和 解码器 可以 是 神经 网络 用 一 隐藏层 或 深度
神经 网络 用 多个 隐藏 层.  目标 在 学习 是
 构建  编码器 和 解码器 因此 作为  使  输出 ˆx 作为
close   输入 x 作为 可能, i.e., g(f (x)) ≈ x. 假设 该 我们 是
给定  训练 数据集 D = {x1, · · · , xN },  编码器 和 解码器
(f 和 g) 可以 是 学会了 通过 最小化 的  squared-误差 损失
函数:

N
X

min
f,g

‘(xi, ˆxi) =

kxi − g(f (xi))k2.

N
X

i=1
 替代 是 最小化 的  重建 交叉-熵, 其中
它 是 假设 该 x 和 ˆx 是 或 bit vectors 或 概率 vectors,

i=1

min
f,g

N
X

i=1

‘H (xi, ˆxi) = −

N
X

D
X

[xk

i log ˆxk

i + (1 − xk

i ) log(1 − ˆxk

i )],

i=1

k=1

其中 D 是  维度 的 输入, 和 xk
i 是  k-th
维度 的 xi 和 ˆxi, 分别地.  优化 在 学习 的
自编码器 是 典型地 carried out 通过 随机梯度下降.

i 和 ˆxk

OriginalinputReconstructedoutputEncoderneuralnetwork!LatentrepresentationDecoderneuralnetwork"#$%$ℓ$,#$40

面向匹配的深度学习

图 3.8: 架构 的  De-noising 自编码器.

通过 limiting  维度 的 潜在 表示 y, 自编码器 是 forced  学习  表示 在 该  大多数 salient
特征 的 数据 是 uncovered 在  “压缩” 低 维
表示.

Denoising Autoencoders (DAE):  deal 用 corruption 在 
数据, DAE 是 也 提出 (Vincent 等人, 2008) 作为  扩展 的
vanilla 自编码器. 它 是 假设 该 DAE receives  corrupted 数据
样本 作为 输入 和 预测  原始 uncorrupted 数据 样本
作为 输出. 第一,  corrupted 输入 ˜x 是 创建了 从  原始
uncorrupted 输入 x 通过  随机 过程 ˜x ∼ qD(˜x|x), 其中 qD
是  conditional 分布 超过 corrupted 数据 samples conditioned 在
 uncorrupted 数据 样本. 下一个,  corrupted 输入 ˜x 是 映射 到
 潜在 表示 y = f (˜x) 和 则  潜在 表示 是
映射 到  重建 输出 ˆx = g(f (˜x)), 作为 显示 在 图 3.8.
 参数 的  编码器 和  解码器 是 学会了  minimize
 average 重建 误差 (e.g., 交叉 熵 误差) 之间 
输入 x 和 重建 输出 ˆx.

其他 类型 的 AutoEncoders 包括 稀疏 自编码器 (Ranzato
等人, 2007), Variational 自编码器 (Kingma 和 Welling, 2014), 和
卷积 自编码器 (Masci 等人, 2011).

OriginalinputEncoderneuralnetwork!LatentrepresentationDecoderneuralnetwork"#$Reconstructedoutput$%$&ℓ$,#$Stochasticmapping%$~*+%$|$3.1. 深度学习概述

41

图 3.9: 架构 的 CBOW.

3.1.2 表示学习

 强 能力 在 表示学习 是  主要 原因 用于 
大 成功 的 深度学习. 在 该 节, 我们 介绍 若干 方法
的 表示学习 该 是 successfully 应用 在 匹配,
包括中 方法 的 学习 词 embeddings 和 contextualized
词 表示.

词 Embeddings

词嵌入 是  基本 方式 的 表示中 词 在 自然语言处理 (NLP) 和 信息检索 (IR). Embeddings
的 词 是 通常 创建了 基于 在  假设 该  含义
的  词 可以 是 determined 通过 其 contexts 在 文档.
词2Vec: Mikolov 等人 (2013) 提出  词2Vec 工具 和
使 词嵌入 流行. 词2Vec 学习 embeddings 的 词
从  大 语料库 使用中 浅层 神经 网络 在  无监督
方式. 那里 是 二 speciﬁc 方法 在 词2Vec: 连续 Bag
的 词 (CBOW) 和 跳字模型.

作为 显示 在 图 3.9, CBOW 取  上下文 的  词 作为 输入

!|#|×%!|#|×%!|#|×%&%×|#|'-dim|(|-dimone-hotvectorsofthecontextwords(cid:1)HiddenlayerInputlayer)*+,)*+,-.)*-,/0)*12*|2*+,,⋯,2*-,|(|-dimone-hotvectorofthecenterword5*42

面向匹配的深度学习

和 预测  词 从  上下文. 它 旨在  学习 二 矩阵,
U ∈ RD×|V | 和 W ∈ R|V |×D, 其中 D 是  大小 的 嵌入 空间,
V 是  词汇, 和 |V | 是  大小 的  词汇. U 是 
输入 词 矩阵 这样的 该  i-th column 的 U, 表示为 作为 ui, 是
 D-维 嵌入 向量 的 输入 词 wi. 类似地, W 是
 输出 词 矩阵 这样的 该  j-th row 的 W, 表示为 作为 wj, 是
 D-维 嵌入 向量 的 输出 词 wj. 注意 该 每个
词 wi 具有 二 嵌入 vectors, i.e.,  输入 词 向量 ui 和
 输出 向量 wi. 给定  语料库,  学习 的  U 和 W
amounts   以下 计算:

1. Selecting  词 序列 的 大小 2m+1: (wc−m, · · · , wc−1, wc, wc+1,
· · · , wc+m) 和 生成中  一-hot 词 向量 用于 wc, 表示为
作为 xc, 和 生成中 一-hot 词 vectors 用于  上下文 的 wc,
表示为 作为 (xc−m, · · · , xc−1, xc+1, · · · , xc+m);

2. 映射   嵌入 词 vectors 的  上下文: (uc−m =
Uxc−m, · · · , uc−1 = Uxc−1, uc+1 = Uxc+1, uc+m = Uxc+m);

3. Getting  average 上下文: ˆu = 1

2m (uc−m+· · ·+uc−1+uc+1 · · · uc+m);

4. 映射   嵌入 vectors 的  中心 词: wc = Wxc;

5. 假设 该  中心 词 是 “生成了” 通过  average

上下文 ˆu.

CBOW adjusts  参数 在 U 和 W 这样的 该

arg min
U,W

‘ = − Y
c
= − Y
c

log P (wc|wc−m, · · · , wc−1, wc+1, · · · , wc+m)

log

exp{wT
c ˆu}
k=1 exp{wT

P|V |

k ˆu}

.

 设置 的 跳字模型 是 相似  该 的 CBOW, 同时 输入
和 输出 是 swapped. 作为 显示 在 图 3.10,  输入 的 Skip
Gram 是  一-hot 向量 的  中心 词, 和  输出 是
 vectors 的  词 在  上下文. 跳字模型 也 旨在  学习
二 矩阵 U ∈ RD×|V | 和 W ∈ R|V |×D. 给定  文本 语料库, 
学习 amounts   以下 计算:

3.1. 深度学习概述

43

图 3.10: 架构 的 跳字模型.

1. Selecting  词 序列 的 大小 2m+1: (wc−m, · · · , wc−1, wc, wc+1,
· · · , wc+m) 和 生成中  一-hot 词 向量 用于 wc, 表示为
作为 xc, 和 生成中 一-hot 词 vectors 用于  上下文 的 wc,
表示为 作为 (xc−m, · · · , xc−1, xc+1, · · · , xc+m);

2. 映射   嵌入 词 向量 的  中心 词: uc =

Uxc;

3. 映射   嵌入 词 vectors 的  上下文: wc−m =

Wxc−m, · · · , wc−1 = Wxc−1, wc+1 = Wxc+1, · · · , wc+m = Wxc+m;

4. 假设 该  嵌入 vectors 的  上下文 wc−m, · · · , wc−1,

wc+1, · · · , wc+m 是 “生成了” 通过  中心 词 uc.

跳字模型 adjusts  参数 在 U 和 W 这样的 该

arg min
U,W

‘ = − log Y
c

= − log Y
c

2m
Y

j=0;j6=m

2m
Y

j=0;j6=m

P (wc−m+j|wc)

exp{wT
P|V |
k=1 exp{wT

c−m+juc}
k uc}

.

!"×$%"×|$|'-dim|(|-dimone-hotvectorofthecenterword)*+,)*+,-.)*-,/*)*|(|-dimone-hotvectorofthecontextwords(cid:1)%"×|$|%"×|$|01*+,|1*2*+,2*+,-.2*-,01*+,-.|1*01*-,|1*(cid:1)44

面向匹配的深度学习

除之外 词2Vec,  数量 的 词嵌入 (和 文档
嵌入) 模型 具有 是 开发了 在 最近 年, 包括中
 GloVe (全局 Vectors) (Pennington 等人, 2014), fastText 1, 和
doc2Vec (Le 和 Mikolov, 2014).

Contextualised 词 表示

 经典 词嵌入 模型 (e.g., 词2Vec 和 GloVe) 具有
 基本 shortcoming: 它们 生成 和 利用  相同 embeddings 的  相同 词 在 diﬀerent contexts. 因此, 它们 cannot
eﬀectively deal 用  上下文-dependent 本质 的 词. Contextualized 词 embeddings aim 在 捕获中 lexical 语义 在 diﬀerent
contexts.  数量 的 模型 具有 是 开发了, 包括中 ULMFiT
(通用 语言 模型 精细-调整), ELMo (Peters 等人, 2018),
GPT (Radford 等人, 2018), GPT-2 (Radford 等人, 2019), Bidirectional
编码器 表示 从 Transformers (BERT) (Devlin 等人,
2019), 和 XLNet (Yang 等人, 2019c).

之中  模型, BERT 是  大多数 广泛使用. BERT 是  mask
语言 模型 ( denoising auto-编码器) 该 旨在  reconstruct
 原始 句子 从  corrupted ones. 该 是, 在  pretrain 阶段,  输入 句子 是 corrupted 通过 replacing 一些 原始
词 用 ‘[MASK]’.  学习 目标, 因此, 是  预测
 masked 词  get  原始 句子.

作为 说明了 在 图 3.11, BERT 采用  Transformer 
学习  contextual relations 之间 词 在  文本. Speciﬁcally,
 模型 receives  对 的 句子 separated 通过 词元 ‘[SEP]’ 作为
输入.  二 句子 是 称为  作为  左 上下文 和 
正确 上下文, 分别地. BERT manages  捕获  语言
表示 的 两者  左 和  正确 contexts 通过 使用中  Transformer 编码器. 形式化地, 给定  词 序列 W = {w1, w2, · · · , wN }
的 该 由组成 的  词 在  左 上下文,  词元 ‘[SEP]’,
和  词 在  正确 上下文, BERT 首先 构建  输入
表示 en 用于 每个 词 wn 通过 summing  相应
词嵌入 vT
n (i.e.,
表明 是否 它 是 从  左 上下文 和 或  正确 上下文),

n (i.e., 通过 词2Vec), 片段 嵌入 对比

1https://fasttext.cc/

3.2. 面向匹配的深度学习概述

45

和 位置 嵌入 vP
n (i.e., 表明  位置 的  词 在 
序列).  输入 表示 是 则 fed 到  layered 块
的 Transformer 编码器  获得 contextualized 表示 用于 
词 的  对 的 句子. 每个 Transformer 块 是 composed 的
 多头注意力 followed 通过  馈-前向 层.

 学习 的 BERT 由组成 的 二 stages: pre-训练 和 ﬁnetuning. 在 pre-训练, 句子 pairs collected 从  大 语料库
是 使用了 作为 训练数据.  模型 参数 是 determined 使用中
二 训练 strategies: mask 语言 建模 和 下一个 句子
预测. (1) 在 mask 语言 建模, 15% 的  随机地 chosen
词 在  二 句子 是 replaced 用 词元 ‘[MASK]’ 在之前
feeding 它们 到  模型.  训练 目标, 则, 是  预测
 原始 masked 词, 基于 在  contexts 提供 通过 
non-masked 词 在  句子. (2) 在 下一句预测,
 模型 receives pairs 的 句子 作为 输入.  训练 目标 是 
预测 如果  秒 句子 在  对 是  后续 句子 在
 原始 文档. 关于 50% 的  输入 是 正 例子,
同时  其他 50% 的  输入 是 负 例子. 在 pre-训练,
mask 语言 建模 和 下一句预测 是 进行了
联合地, 通过 minimizing  结合 损失 函数 的  二 strategies.
精细-调整 的  pre-训练了 BERT 模型 是 进行了 在  监督学习 方式, 用于 生成中 词 表示 tailored
用于  speciﬁc 任务. Let 我们 使用 文本 classiﬁcation 作为  例子. 
classiﬁcation 层 是 added 在 top 的  BERT 模型  构建
 classiﬁer. 假设 该 每个 实例 在  训练数据 由组成
的  词 序列 x1, · · · , xM , 和  标签 y.  模型 feeds 
序列 通过  pre-训练了 BERT 模型, 生成  表示 用于  ‘[CLS]’ 词元, 和 预测  标签 ˆy.  ﬁne-调整
目标, 因此, 可以 是 deﬁned   ground-真值 标签 y 和
 预测了 标签 ˆy.

3.2 面向匹配的深度学习概述

面向匹配的深度学习, 称为  作为 深度匹配, 具有 成为
 最先进的 technologies 在 搜索与推荐 (Mitra
和 Craswell, 2019). 比较了 用  传统 机器 学习-

46

面向匹配的深度学习

图 3.11: BERT 训练 过程: ()  pre-训练 阶段 和 Transformer
架构; (b)  ﬁne-调整 阶段 modiﬁes  pre-训练了 参数 通过 taskspeciﬁc 训练.

ing 方法,  深度学习 方法 改进  匹配
准确率 在 三 ways: 1) 使用中 深度 神经 网络  构建
richer 表示 用于 匹配 的 对象 (i.e., 查询, 文档,
用户, 和 物品), 2) 使用中 深度学习 算法  构建 更多
强大 函数 用于 匹配, 和 3) 学习  表示
和 匹配 函数 联合地 在  端--端 fashion. 另一个 优势 的 深度匹配 方法 是 其 ﬂexibility 的 扩展中
 多-模态 匹配 其中  共同 语义 空间 可以 是
学会了  统一地 表示 数据 的 diﬀerent 模态.

各种 神经网络 架构 具有 是 开发了. 这里, 我们
给出  通用 框架, 典型 架构, 和 designing principles
的 深度匹配.

3.2. 面向匹配的深度学习概述

47

图 3.12:  通用 匹配 框架.

3.2.1 通用 框架 用于 深度匹配

图 3.12 显示  通用 框架 的 匹配.  匹配
框架 取 二 匹配 对象 作为 其 输入 和 输出 
数值 值  表示  匹配程度.  框架 具有
输入 和 输出 层 在  bottom 和  top. 之间  输入
和 输出 层, 那里 是 三 consecutive 层. 每个 层 可以 是
implemented 作为  神经网络 或  部分 的  神经网络:

 输入 层 receives  二 匹配 对象 该 可以 是 词

embeddings, ID vectors, 或 特征 vectors.

 表示 层 converts  输入 vectors 到  分布式 表示. 神经 网络 这样的 作为 MLP, CNN,
和 RNN 可以 是 使用了 这里, depending 在  类型 和 本质 的
 输入.

 交互 层 比较  匹配 对象 (i.e., 二 分布式 表示) 和 输出  数量 的 (局部 或 全局)
匹配 signals. 矩阵 和 张量 可以 是 使用了 用于 storing 
signals 和 其 locations.

输出: MLPAggregation: 池化, ConcatenationInteraction: 矩阵, TensorRepresentation: MLP, CNN, LSTMInput: ID Vectors, 特征 Vectors48

面向匹配的深度学习

图 3.13:  共同 神经网络架构 一 用于 深度匹配.

 聚合 层 aggregates  个体 匹配 signals
到  高-水平 匹配 向量. Operations 在 深度 神经
网络 这样的 作为 池化 和 concatenation 是 通常 采用
在 该 层.

 输出 层 取  高-水平 匹配 向量 和 输出
 匹配分数. 线性 模型, MLP, 神经 张量 网络
(NTN), 或 其他 神经 网络 可以 是 利用了.

神经网络 架构 开发了 因此 far 用于 查询-文档
匹配 在 搜索 和 用户-物品匹配 在 推荐 可以 是
总结了 在  框架.  框架

3.2.2 典型 架构 用于 深度匹配

图 3.13 显示  典型 架构 用于 深度匹配 在 搜索 (Huang
等人, 2013) 和 推荐 (他 等人, 2017c). 在  架构,
 输入 X 和 Y 是 二 文本 在 搜索 或 二 特征 vectors 在
推荐.  二 输入 是 首先 processed 用 二 神经

AggregationRepresentationYXRepresentationInteractionMatch3.2. 面向匹配的深度学习概述

49

图 3.14:  神经网络架构 二 用于 查询-文档匹配.

网络 独立地, 用于 创建中 其 表示. 则, 
架构 计算  interactions 之间  二 表示 和 输出 匹配 signals. 最后,  匹配 signals 是
aggregated  形式  ﬁnal 匹配分数.  special 案例 的  架构 是  let  二 表示 神经 网络 完全相同 和
其 参数 共享 (Huang 等人, 2013; Shen 等人, 2014). 
simpliﬁcation 使  网络 架构 easier  训练 和 更多
鲁棒, 该 是 可能 当 两者 X 和 Y 是 文本.

图 3.14 显示  架构 广泛使用 用于 深度匹配 在
搜索 (Hu 等人, 2014; Pang 等人, 2016b). 它 取 二 文本 X 和
Y 作为 输入, 和 每个 词 在  文本 是 表示了 通过 其 嵌入.
 架构 首先 计算 lexical interactions 之间  二
文本.  results 的  lexical interactions 是 stored 在  矩阵 或
 张量, 用于 keeping  results 以及 其 locations. 则, 
interactions 在  lexical 水平 是 aggregated 到  ﬁnal 匹配
分数. 比较了   首先 架构, 该 架构 具有 二
显著的 properties: (1) 进行  interactions 在  lexical
水平 而 比 在  语义层面, 该 是 通常 必要 用于
搜索; (2) storing 和 利用中  位置 的 每个 交互 在 
下一个 步骤.

YXInteractionAggregationMatch50

面向匹配的深度学习

图 3.15:  神经网络架构 用于 用户-物品匹配.

图 3.15 显示  架构 广泛使用 用于 深度匹配
在 推荐 (他 和 Chua, 2017; Xin 等人, 2019).  输入
匹配 对象 是  用户 (查询) 和  物品 (文档) 用 其
特征 vectors.  输入 vectors 可以 是 结合 (concatenated).
则,  结合 输入 vectors 是 processed 用  神经网络
用于 创建中  分布式表示 (embeddings) 的 它们. 下一个,
 架构 计算  interactions 之间  用户 和 物品,
e.g., 首先-顺序 interactions 使用中 线性 回归, 秒-顺序 顺序
interactions 使用中 分解机, 和 更高-顺序 interactions
使用中 MLP 或 CNN. 最后, 这些 interactions 是 aggregated  获得
 ﬁnal 匹配分数. Please 注意 该 尽管 它 是 最初 designed 在 推荐 系统 文献,  混合 结构 是 也
流行 在 搜索. 用于 例子,  搜索 模型 的 Duet Mitra 等人
(2017) adopts 相似 架构 用于 匹配 查询 和 文档.

AggregationMatchRepresentationYXInteraction (2ndand/orhigherorder)EmbeddingsInteraction (1storder)3.2. 面向匹配的深度学习概述

51

3.2.3 Designing Principles 的 深度匹配

我们 提出 二 designing principles 用于  开发 的 深度 搜索与推荐中的匹配模型:  modular 原则 和
 混合 原则.

 modular 原则 postulates 该  匹配模型 通常
由组成 的 多个 模块 (函数), 和 因此  开发
的 这样的  模型 应当 也 取  modular 方法. 用于 例子,
 表示 模块 可以 是 implemented 用 CNN, RNN, 或
MLP,  交互 模块 可以 是  矩阵 或  张量, 和 
聚合 模块 可以 是  池化 或 concatenating operator. Diﬀerent
combinations 的  模块 结果 在 diﬀerent 匹配 模型.

 混合 原则 断言 该  组合 的 dichotomic 技术 是 有帮助 在  开发 的 匹配 模型. 用于 例子,
在 用户-物品匹配 在 推荐,  首先-顺序, 秒-顺序,
和 更高-顺序 interactions 所有 contribute   确定 的 
ﬁnal 匹配程度. 在 查询-文档匹配 在 搜索,  查询
和 文档 可以 是 表示了 用 两者 bag-的-词 和 sequences
的 词 embeddings. 此外, 在 两者 搜索与推荐,
 表示 和 交互 之间 对象 可以 是 结合
使用中  组合 的 深度 和 广泛 神经 网络, 或 非线性
和 线性 模型.

 下一个 二 chapters 将 介绍  神经网络 架构 designed 用于 搜索与推荐 用 更多 细节.

4

搜索中的深度匹配模型

 深度学习 方法  查询-文档匹配 在 搜索
主要地 fall 到 二 categories: 表示学习 和 匹配
函数 学习, 其 架构 是 描述了 在 图 3.13 和
图 3.14, 分别地. 在  二 categories, 神经 网络 是
利用了 用于 表示中 查询 和 文档, 用于 进行 
interactions 之间 查询 和 文档, 和 用于 aggregating 
匹配 signals. Diﬀerent combinations 的 技术 结果 在 diﬀerent 深度匹配 模型. 表 4.1 总结  深度匹配
模型 在 搜索.  首先 column 分类  模型 作为 匹配
基于 在 表示学习, 匹配 基于 在 匹配函数
学习, 和  模型 该 组合  二 方法.  秒
column 进一步 分类  模型 根据 如何 词 顺序
信息 是 使用了 在  初始 表示. 用于 例子, “bag 的
letter tri-grams” 和 “bag 的 词 means” means 该  顺序 的
词 在  查询 和/或 文档 是 不 考虑了. “序列
的 词” means 该  模型 利用  词 ordering 信息.  第三 column 描述  类型 的  神经 网络 (或
变换 函数) 采用了 在  模型. 注意 该 我们 使用
 术语 “表示学习” 和 “匹配函数学习”

52

4.1. 基于表示学习的匹配

53

用于 ease 的 解释.  匹配函数学习 方法 也
学习 和 利用 词 (和 句子) 表示.

在  remaining 的  章, 我们 介绍  代表性
模型 基于 在 表示学习 在 节 4.1 和  模型
基于 在 匹配函数学习 在 节 4.2. 实验的 results
是 也 显示 在 每个 节.

4.1 基于表示学习的匹配

4.1.1 通用 框架

 表示学习 方法 假设 该 查询 和 文档
可以 是 表示了 通过 低-维 和 稠密 vectors. 那里 是 二
关键 问题: 1) 什么 种类 的 神经 网络  使用 用于 创建中 
表示 的 查询 和 文档, 和 2) 什么 种类 的 函数 
使用 用于 计算中  ﬁnal 匹配分数 基于 在  表示.
Let 我们 假设 该 那里 是 二 spaces: 查询空间 和 文档
空间.  查询空间 包含 所有  查询, 和  文档空间
包含 所有  文档.  查询空间 和 文档空间 可以
是 异质, 和  相似之处 之间 查询 和 文档
跨越  spaces 可以 是 hard  计算. 进一步, 假设 该 那里
是  新 空间 在 该 两者 查询 和 文档 可以 是 映射
到, 和  相似度 函数 是 也 deﬁned 在  新 空间. 在 搜索,
 相似度 函数 可以 是 利用了  表示  匹配 degrees
之间 查询 和 文档. 在 其他 词, 匹配 之间
查询 和 文档 是 进行了 在  新 空间 之后 映射.
图 4.1 显示  框架 的 查询-文档匹配 基于 在
表示学习.

形式化地, 给定 查询 q 在  查询空间 q ∈ Q 和 文档
d 在  文档空间 d ∈ D, 函数 φq : Q 7→ H 和 φd :
D 7→ H 表示 映射 从  查询空间 和 映射 从
 文档空间   新 空间 H, 分别地.  匹配
函数 之间 q 和 d 是 deﬁned 作为

f (q, d) = F (φq(q), φd(d)),

其中 F 是  相似度 函数 deﬁned 在  新 空间 H.

54

搜索中的深度匹配模型

,
)
c
8
1
0
2

,
)
6
1
0
2

,
.
l


,
.
l


t
e

t
e

n
i
Y
(
N
N
C
B


,
)

9
1
0
2

,
.
l


t
e

g
n

Y
(

2
E
R

,
)
8
1
0
2

,
.
l


t
e

g
n
o
G
(
N

I
I

D

,
)
8
1
0
2

y

T
(
N
R
C
H

,
)
d
8
1
0
2

,
.
l


,
.
l


t
e

t
e

y

T
(
N

C
M

n

T
(
N

w
M

n
o
i
t
n
e
t
t


,
)
5
1
0
2

,
k
y
z
c
j

l
o
k
i
M
d
n


n

Y

;
3
1
0
2

,
.
l


t
e
w
e
r
d
n

(

C
C
p
e
e
D

N
N
C

)
b
8
1
0
2

,
.
l


t
e

y

T
(
N

R
S
C

,
)

9
1
0
2

,
.
l


t
e

g
n

i
J
(
N
N
R
H
S

M

n
o
i
t
n
e
t
t

+
N
N
R

)
5
1
0
2

,
.
l


t
e


M

(

s
N
N
C
m

-

,
)

7
1
0
2

,
.
l


t
e

g
n

W

(
R
M
C


l

d
o
m

-
s
s
o
r
c

,
)
8
1
0
2

,
.
l


t
e

e
i
N
(

M
C

M

,
)
5
1
0
2

,
g
n

u
H
d
n


u
i
Q
(
N
T
N
C

,
)
4
1
0
2

,
.
l


t
e

u
H
(

I
-
C
R


,
)
4
1
0
2

,
.
l


t
e

n
e
h
S
(

M
S
L
C

)
5
1
0
2

,
e
z
t
ü
h
c
S

d
n


n
i
Y
(
N
N
C
n

r
G

-
i
t
l
u
M

,
)
c
8
1
0
2

,
.
l


t
e

i
n

m

Z
(

F
-
M
R
N

,
)
6
1
0
2

,
.
l


t
e

i
g
n

l

P
(
N
N
R
M
T
S
L

-

)

6
1
0
2

,
.
l


t
e

n

W

(

M
T
S
L
-
V
M

)
b
8
1
0
2

,
.
l


t
e

i
n

m

Z
(

M
R
N
S

)
3
1
0
2

,
.
l


t
e

g
n

u
H
(

M
S
S
D

)
8
1
0
2

,
.
l


t
e

l
e
s
y
G
(

M
S
V
N

P
L
M

r

e
n
i
l

P
L
M

N
N
C

N
N
R

m

r
g
-
i
r
t

r
e
t
t
e
l

f
o

g

b

s
d
r
o
w

f
o

g

b

s
d
r
o
w

f
o

e
c
n
e
u
q
e
s

s
l
e
d
o
M
g
n
i
h
c
t

M

e
r
u
t
c
e
t
i
h
c
r

k
r
o
w
t
e
N

n
o
i
t

t
n
e
s
e
r
p
e
R
t
u
p
n
I

.
h
c
r

e
s

n
i

g
n
i
h
c
t

m

t
n
e
m
u
c
o
d
-
y
r
e
u
q

o
t

s
e
h
c

o
r
p
p


g
n
i
n
r

e
l

p
e
e
D

:
1
.
4

e
l
b

T

,
)
b
6
1
0
2

,
.
l


t
e

g
n

P
(

d
i
m

r
y
P
h
c
t

M

,
)
4
1
0
2

,
.
l


t
e

u
H
(

I
I
-
C
R


,
)
6
1
0
2

,
.
l


t
e

h
k
i
r

P
(

l
e
d
o
M
n
o
i
t
n
e
t
t

e
l
b
l

s
o
p
m
o
c
e
D

)
6
1
0
2

,
.
l


t
e

g
n

Y
(

M
M
N


,
)
7
1
0
2

,
.
l


t
e

i
u
H
(
R
R
C

P

,
)

7
1
0
2

,
.
l


)
8
1
0
2

,
.
l


t
e

t
e

g
n

P
(

k
n

R
p
e
e
D

i
u
H
(
R
R
C

P
-
o
C

)
c
7
1
0
2

,
.
l


t
e

g
n

W

(

M
P
M
B

i

,
)
b
7
1
0
2

,
.
l


t
e

n
e
h
C
(

M
I
S
E

)
5
1
0
2

,
i

L

d
n


y
h
t

p
r

K
(
N
N
R
B

)
7
1
0
2

,
.
l


t
e

g
n
o
i
X
(

M
R
N
K

-

)
d
9
1
0
2

,
.
l


t
e

g
n

W

(

M
C
R

)
6
1
0
2

,
.
l


t
e

o
u
G
(

M
M
R
D

N
N
C
+
N
N
R

n
o
i
t
n
e
t
t


P
L
M

l
e
n
r
e
K
-
F
B
R

n
o
i
t
n
e
t
t


N
N
C

N
N
R

s
d
r
o
w

f
o

g

b

)
8
1
0
2

,
.
l


t
e

n

F
(
T
N
H

i

,
)
b
6
1
0
2

,
.
l


t
e

n

W

(
N
N
R
S
-
h
c
t

M

N
N
R

l

i
t

p
s

,
)
b
8
1
0
2

,
.
l


t
e

n
e
h
C
(
X
M

I

,
)
9
1
0
2

,
o
h
C
d
n



r
i
e
u
g
o
N
(

h
c
t

M
4
T
R
E
B

s
d
r
o
w

f
o

e
c
n
e
u
q
e
s

g
n
i
h
c
t

M

n
o

d
e
s

b

n
o
i
t

t
n
e
s
e
r
p
e
r

g
n
i
n
r

e
l

g
n
i
h
c
t

M

n
o

d
e
s

b

g
n
i
h
c
t

m

n
o
i
t
c
n
u
f

g
n
i
n
r

e
l

)
8
1
0
2

,
.
l


t
e

i

D
(

M
R
N
K
-
v
n
o
C

)
9
1
0
2

,
.
l


t
e

u
h
Z
(
R

H

)
7
1
0
2

,
.
l


t
e


r
t
i

M

(

t
e
u
D

l
e
n
r
e
K
-
F
B
R

P
L
M
+
N
N
C

s
d
r
o
w

f
o

e
c
n
e
u
q
e
s

d
e
n
i
b
m
o
C

4.1. 基于表示学习的匹配

55

图 4.1: 查询-文档 基于表示学习的匹配.

Diﬀerent 神经 网络 可以 是 利用了 用于 表示中  查询
和 文档, 以及 计算中  匹配 分数 给定 
表示, resulting 在 diﬀerent 匹配 模型. 大多数 的 
匹配 模型 (e.g., DSSM) 使用 完全相同 网络 structures 用于
查询 和 文档 (i.e., φq = φd). 它们 可以 是 广义  具有
diﬀerent 网络 structures 用于 查询 和 文档, 分别地.

4.1.2 表示中 用 Feedforward 神经 网络

Feedforward 神经 网络 是  首先 网络 架构 使用了 
创建 语义 表示 的 查询 和 文档. 用于 例子,
Huang 等人 (2013) 提出 表示中 查询 和 文档 用
深度 神经 网络, 使用中  模型 称为  作为 深度 Structured
语义 模型 (DSSM). 图 4.2 显示  架构 的 DSSM.
DSSM 首先 表示 查询 q 和 其 associated 文档 d’s
(d1, d2, · · · , dn) 作为 vectors 的 术语 和 取  vectors 作为 输入. 
overcome  diﬃculties resulting 从  very 大 词汇 大小 在
网页搜索, DSSM maps  术语 vectors  letter n-gram vectors. 用于
例子, 词 “good” 是 映射 到 letter trigrams: (“#go”, “goo”,
“ood”, “od#”), 其中 “#” 表示 starting 和 ending marks. 在 该
方式,  dimensions 的 输入 vectors 可以 是 减少了 从 500k  30k,
因为  数量 的 letter n-grams 在 English 是 有限的. 它 则
maps  letter n-gram vectors 到 输出 vectors 的 lower dimensions
通过 深度 神经 网络:

yq = DNN(q)
yd = DNN(d),

querydocumentneural networkneural networkmatchingscorequery representationdocumentrepresentation56

搜索中的深度匹配模型

图 4.2: 深度结构化语义模型

其中 DNN(·) 是  深度神经网络 使用了 在 DSSM, yq 和 yd
是  输出 vectors 该 表示  隐藏 主题 在 查询 q 和
文档 d, 分别地.

下一个, DSSM 取  余弦相似度 之间  输出 向量
的 查询 (表示为 作为 yq) 和  输出 向量 的 文档 (表示为
作为 yd) 作为 匹配分数:

f (q, d) = cos(yq, yd).

DSSM 学习  模型 参数 通过 最大似然 估计 (MLE) 在  基础 的 查询, associated 文档, 和 clicks.
Speciﬁcally, 给定 查询 q 和  列表 的 文档 D = {d+, d−
1 , · · · , d−
k },
其中 d+ 是  clicked 文档 和 d−
k 是 unclicked (显示 但
skipped) 文档.  目标 的 学习 amounts  maximizing
 conditional probabilities 的 文档 d+ 给定 查询 q:

1 , · · · , d−

P (d+|q) =

exp(λf (q, d+))
d0∈D exp λf (q, d0)

P

,

其中 λ > 0 是  参数.

500kqd1dn500k500k30k30k30k300300300300300300128128128wordvectorwordhashinghiddenlayersEmbeddingvectorsmatchingscoresconditionalprobabilities!",$%!",$&'$%|"'$&|"………4.1. 基于表示学习的匹配

57

4.1.3 表示中 用 卷积 神经 网络

虽然 成功 在 网页搜索, 研究ers 发现 该 DSSM 具有 二
shortcomings. 第一, 深度 神经 网络 包含 也 许多 参数, 该 使 它 diﬃcult  训练  模型. 秒, DSSM views 
查询 (或  文档) 作为  bag 的 词 但 不  序列 的 词.
作为  结果, DSSM 是 不 eﬀective 在 处理 局部 上下文 信息
之间 词. 这些 二 drawbacks 可以 是 解决了 好 用 CNN.
第一, CNN 具有  smaller 数量 的 参数 比 DNN, 因为 其
参数 是 共享 在 diﬀerent 输入 positions (偏移 不变性), 作为
显示 在 图 3.3. 秒,  基本 operations 的 卷积 和
max-池化 在 CNN 可以 keep  局部 上下文 信息. 因此, CNN 是  very eﬀective 架构 用于 表示中 查询 和
文档 在 搜索.

卷积潜在语义模型 (CLSM)

Shen 等人 (2014) 提出  捕获 局部 上下文 信息 用于 潜在
语义 建模 使用中  卷积神经网络 称为  作为
CLSM. 作为 显示 在 图 4.3, CLSM 使  follow modiﬁcations
 DSSM 用于 表示中 查询 和 文档:

•  输入 句子 (查询 或 文档) 是 表示了 作为 lettertrigram vectors 基于 在 词 n-grams, 该 是  concatenation
的  letter-trigram vectors 的 每个 词 在  词 n-gram.

• 卷积 operations 是 使用了  模型 上下文 特征 的
词 n-grams.  上下文 特征 的 词 n-grams 是 投影
 vectors 该 是 close  每个 其他 如果 它们 是 semantically
相似.

• Max-池化 operations 是 使用了  捕获  句子-水平

语义 特征.

CLSM 取  余弦相似度 之间  表示 vectors 的
查询 和 文档 作为  ﬁnal 匹配分数.

相似  DSSM,  模型 参数 的 CLSM 是 学会了 
maximize  似然 的 clicked 文档 给定 查询 在 

58

搜索中的深度匹配模型

图 4.3: 卷积潜在语义模型

训练数据.  方法 的 计算中  conditional 概率 的
文档 d+ 给定 查询 q 是  相同 作为 该 的 DSSM.

卷积 匹配模型 (ARC-I)

Hu 等人 (2014) 提出  使用 卷积 架构 用于 匹配
二 句子.  模型, 称为 ARC-I, 首先 发现s  表示
的 每个 句子 用 卷积 神经 网络 和 则 比较
 表示 的  二 句子 用  多层感知机.
 ARC-I 模型 取  序列 的 embeddings 的 词 (i.e.,
词 embeddings 训练了 beforehand 用 词2vec (Mikolov 等人,
2013)) 作为 输入.  输入 是 总结了 通过  层 的 卷积 和 池化   ﬁxed-长度 表示 在  ﬁnal 层.
 解决  问题 该 diﬀerent 句子 具有 diﬀerent lengths,
ARC-I puts 零   元素 之后  最后 词 的  句子
直到  最大值 长度.

形式化地, 给定 查询 q 和 文档 d, ARC-I 表示 每个
的 它们 作为  序列 的 embeddings 的 词. 在 该 方式,  词
顺序 信息 是 kept. 它 则 maps  序列 的 embeddings
到 输出 vectors 的 lower dimensions 用  1-D 卷积 神经

Convolutionletter-trigrammax 池化90kthe90kcat90ksat……90kmat300300300…………300maxmax……max128fullyconnected<s><s><s>thecatthecatsatquery/documentwordn-gramthemat<s>……4.1. 基于表示学习的匹配

59

网络:

yq = CNN(q)
yd = CNN(d),
其中 CNN(·) 是  1-D 卷积神经网络, yq 和 yd 是
 输出 vectors 的 q 和 d, 分别地.

 计算  匹配分数, ARC-I 利用  多个 层

perceptron:

f (q, d) = W2 · σ

W1

#

"

yq
yd

!

+ b1

+ b2,

其中 W1, b1, W2, 和 b2 是 参数, 和 σ(·) 是  sigmoid
函数.

图 4.4 说明 ARC-I 用  例子 的 二-层 卷积 神经 网络. 给定  输入 句子, 每个 词 是 首先
表示了 用 词嵌入. 则,  卷积 层 生成
上下文 表示, 该 oﬀer  多样性 的 compositions 的 词
之内  三-词 window 和 用 diﬀerent conﬁdences (gray color
表明 低 conﬁdence).  池化 层 则 chooses 之间 二
adjacent 上下文 表示 用于 每个 组合 类型.  输出
的  神经网络 (表示 的 句子) 是  concatenation 的
 池化 results.

判别 训练 用  大 间隔 标准 是 使用了 在
学习 的  模型 参数 在 ARC-I. 给定 查询 q,  相关
查询-文档 对 (q, d) 在  训练数据 应当 receive  更高
分数 比  查询-文档 对 在 该  相关 文档
是 replaced 用  随机 一, i.e., (q, d0). 因此,  ARC-I 模型
minimizes  以下 目标:
L = X
(q,d)∈C

(cid:2)1 − f (q, d) + f (q, d0)(cid:3)

(q,d0)∈C0

+ ,

X

其中 C 和 C0 是 collections 的 相关 查询-文档 pairs 和
不相关 查询-文档 pairs, 分别地.

卷积 神经 张量 网络 (CNTN)

神经 张量 网络 (NTN) 是 最初 提出  显式地 模型
多个 interactions 的 relational 数据 (Socher 等人, 2013). NTN 具有


60

搜索中的深度匹配模型

图 4.4: 卷积 匹配模型 (Arc-I).

强大 表示 能力 和 可以 表示 多个 相似度
函数, 包括中 余弦相似度, 点积, 和 双线性 乘积,
etc.  模型  复杂 interactions 之间 查询 和 文档,
Qiu 和 Huang (2015) 提出  计算  相似度 之间 查询
和 文档 用  张量 层 在 NTN.

相似  ARC-I, 给定 查询 q 和 文档 d, CNTN 首先
表示 每个 的 它们 作为  序列 的 词 embeddings. 则 每个
的  sequences 是 processed 用  1-D 卷积神经网络,
获得中  低-维 表示:

yq = CNN(q)
yd = CNN(d).

作为 显示 在 图 4.5, 在 CNTN,  表示 的  查询

和 文档 是 fed 到 NTN 用于 计算中  匹配分数:

f (q, d) = uT σ

q M[1:r]yd + V
yT

!

+ b

,

#

"

yq
yd

其中 σ 是  元素-wise Sigmoid函数, M[1:r] 是  张量 用
r 片, 和 V, u, 和 b 是 参数.  双线性 张量 乘积
q M[1:r]yd returns  r-维 向量. 一 优势 的 CNTN
yT

thecatonthesatthecatcat satthe cat satcat satsat 在 cat sat onsat onon thesat 在 theon thethe maton  matmatthecatsat onthe cat satsat onthe maton  matConvolutionPooling. . . Concatenationword 嵌入 
4.1. 基于表示学习的匹配

61

图 4.5: 卷积 神经 张量 网络.

是 该 它 可以 联合地 模型  表示 和 interactions. 
表示 的 句子 是 建模 用  卷积 层,
和  interactions 之间 句子 是 建模 用  张量
层.

相似  ARC-I,  学习 的 模型 参数 在 CNTN 也
relies 在 判别 训练 用  大 间隔 标准. 给定
 相关 pairs C 和 不相关 pairs C0,  学习 amounts 
minimizing:

L = X
(q,d)∈C

X

(q,d0)∈C0

(cid:2)γ − f (q, d) + f (q, d0)(cid:3)

+ + λkΘk2,

其中 Θ 包括  参数 在 词嵌入, CNN, 和 NTN;
γ > 0 和 λ > 0 是  间隔 和 正则化 hyper-参数,
分别地.

4.1.4 表示中 用 Recurrent 神经 网络

给定  fact 该 两者 查询 和 文档 是 文本, 它 是 自然
 应用 RNN  表示  查询 和 文档 (Palangi 等人,
2016).  主要 想法 是  发现  稠密 和 低 维 语义
表示 的 查询 (或 文档) 通过 顺序地 processing 每个
词 的  文本. 作为 显示 在 图 4.6, RNN 顺序地 processes
每个 词 在  输入 文本, 和  语义表示 的  最后
词 成为  语义表示 的  整体 文本.

!(cid:1)(cid:1)uMVbq(cid:8)(cid:11)(cid:4)(cid:9)(cid:12)yq(cid:3)(cid:7)(cid:2)(cid:11)(cid:5)(cid:4)(cid:6)(cid:10)ydd1-DCNN1-DCNN62

搜索中的深度匹配模型

 解决  diﬃculty 的 学习 长 术语 dependency 之内
 序列 due   vanishing 梯度 问题, LSTM-RNN 使
使用 的 LSTM 而非  原始 RNN. 期间  scanning 的 
输入 文本,  gates 在 LSTM store  长 术语 dependencies 到
 cells. Speciﬁcally,  前向 pass 用于 LSTM-RNN 是 deﬁned 作为

u(t) = tanh(W4l(t) + Wrec4y(t − 1) + b4),
i(t) =σ(W3l(t) + Wrec3y(t − 1) + Wp3c(t − 1) + b3),
f (t) =σ(W2l(t) + Wrec2y(t − 1) + Wp2c(t − 1) + b2),
c(t) =f (t) (cid:12) c(t − 1) + i(t) (cid:12) u(t),

o(t) =σ(W1l(t) + Wrec1y(t − 1) + Wp1c(t) + b1),
y(t) =o(t) (cid:12) tanh(c(t)),

其中 i(t), f (t), o(t), c(t) 是  输入 门, forget 门, 输出 门,
和 cell 陈述, 分别地; “(cid:12)” 表示  Hadamard (元素-wise)
乘积. 矩阵 W 和 向量 b 是 模型 参数. 向量 y(t) 是
 表示 直到  t-th 词.  表示 的  最后
词 y(m) 是 使用了 作为  表示 的  整个 文本.

给定 查询 q 和 文档 d, LSTM-RNN 首先 创建 其
表示 vectors yq(|q|) 和 yd(|d|), 其中 | · | 表示  长度
的 输入.  匹配分数 是 deﬁned 作为  余弦相似度 之间
 二 vectors:

f (q, d) = cos(yq(|q|), yd(|d|)).

相似  DSSM 和 CLSM, LSTM-RNN 也 学习  模型
参数 通过 MLE 在  基础 的 查询, associated 文档, 和
clicked (正) 文档. 给定 查询 q 和 associated 文档
D = {d+, d−
1 , · · · , d−
k
是  unclicked 文档.  conditional 概率 的 文档
d+ 给定 查询 q 是

k }, 其中 d+ 是  clicked 文档 和 d−

1 , · · · , d−

P (d+|q) =

exp(γf (q, d+))

exp(γf (q, d+)) + P

d−∈D\{d+} exp γf (q, d−)

,

其中 γ > 0 是  参数.

4.1. 基于表示学习的匹配

63

图 4.6: RNN 用于 查询/文档 表示.

4.1.5 表示学习 用 Un-监督/弱监督

无监督学习 和 弱 监督学习 方法 是
采用了  学习 表示 的 查询 和 文档.

神经 向量空间模型 (NVSM)

Traditionally,  低-维 表示 的 词 和 文档 可以 是 学会了 用 主题 模型 和 词/文档嵌入
方法. Gysel 等人 (2018) 呈现 NVSM, 该 学习  lowdimensional 表示 的 词 和 文档 在  语料库 使用中
投影.

 模型 架构 是 显示 在 图 4.7. 给定  大 语料库
D 用 |D| 文档 和  词汇 V 包含中 所有 词 在
 文档,  目标 是  学习  表示 的 文档
RD ∈ <|D|×kd 和  表示 的 词 RV ∈ <|V |×kv . 注意
该  文档 表示 具有 kd dimensions 同时  词
表示 具有 kv dimensions. NVSM 首先 samples  n-gram 用
n contiguous 词 B = (w1, · · · , wn) 从 文档 d 作为  短语.
则, 它 projects  n-gram 短语 到  文档空间 作为:

~h(B) = ~h(w1, · · · , wn) = (f ◦ 范数 ◦ g)(w1, · · · , wn),

letter tri-gram vectorl(i)隐藏 vectory(i)……查询/documentthecatsatmat……1-hotvectorx(i)embeddingvectorWhWWrec64

搜索中的深度匹配模型

图 4.7: 神经 向量空间模型.

~Rwi
其中 g(w1, · · · , wn) = 1
n
tations 在  短语, “范数” 是  ‘-2 归一化 因素, 和

V 是  average 的  词 represeni=1

Pn

f (~x) = W~x,

其中 W 是  变换 矩阵.  目标 的 学习 是 
maximize  相似度 之间  投影 短语 表示
和  文档 表示:

max
RD,RV ,W

Y

Y

d∈D

B:B∼d

σ(h ~Rd

D, ~h(B)i),

其中 ‘B ∼ d’ means 该 短语 B 是 sampled 从 文档 d.

在 在线 匹配, 给定 查询 q 和 文档 d, NVSM projects
 查询   文档空间, 相似  该 用于  n-gram 短语.
 匹配分数 是 计算了 作为  余弦相似度 之间 
文档 表示 和 投影 查询 表示:

f (q, d) = cos(~h(q), ~Rd

D).

Standalone 神经 排序模型 (SNRM)

最近, 研究ers 在  IR 社区 提出  训练 神经
匹配 和 排序 模型 使用中 弱监督 (Dehghani 等人,

(cid:1)(cid:1)wordembeddings!"documentembeddings!#queryqdocumentdcorpush%embeddingofq&#'embeddingofd:cosh%,&#'matchingscore4.1. 基于表示学习的匹配

65

图 4.8: Standalone 神经 排序模型.

2017), 其中  标签 是 获得了 自动地 无 标注
通过 人类 或 利用 的 additional resources (e.g., 点击 数据).
Zamani 等人 (2018b) 呈现 SNRM 通过 介绍中 稀疏性 到 
学会了 潜在 presentations 的 查询 和 文档, 和 构建
 inverted 索引 用于  整体 集合 基于 在  表示.
作为 显示 在 图 4.8,  SNRM 网络 考虑 两者 查询 和
文档 作为  序列 的 词. 之后 going 通过  嵌入
层,  词 序列 成为  序列 的 词 vectors. 则,
 序列 是 decomposed 作为  集合 的 n-grams 和 processed 通过
完全地-connected 层 用 稀疏性 constraints, 生成中  集合 的 highdimensional 稀疏 n-gram 表示. 最后,  average 池化
层 是 使用了  aggregate  n-gram 表示 和 生成 
ﬁnal 序列 表示.

更多 speciﬁcally,  表示 的  文档 d 是 deﬁned 作为

yd =

1
|d| − n + 1

|d|−n+1
X

i=1

φngram(wi, wi+1, · · · , wi+n−1),

其中 w1, w2, · · · , w|d| 表示  词 序列 在 d 和 φngram 表示  高-维 和 稀疏 表示 的  n-gram
wi, wi+1, · · · , wi+n−1. 该 是, φ 首先 converts  n-gram 的 词 到
 n-gram 的 词 vectors, 和 则 使用 多个 馈-前向 层 
生成  表示 的  n-gram. 类似地,  表示

querydocumentembeddinglayer……sequenceofwordvectorsfully-connectedlayersn-gramofwordvectorsn-gramsparserepresentations……averagepoolingmatchingscoredotproductlatentsparserepresentationsequenceofwords……66

搜索中的深度匹配模型

的  查询 q 是 deﬁned 作为

yq =

1
|q| − n + 1

|q|−n+1
X

i=1

φngram(qi, qi+1, · · · , qi+n−1),

其中 qi, qi+1, · · · , q|q| 表示  词 序列 在 q 和 φngram 表示  高-维 和 稀疏 表示 的  n-gram
qi, qi+1, · · · , qi+n−1.  ﬁnal 匹配分数 是 deﬁned 作为  点积 的  二 表示:

f (q, d) = hyq, ydi.

 模型 参数 在 SNRM 是 训练了 用 弱监督
使用中 传统 IR 模型. 给定  查询 q 和  对 的 文档
d1 和 d2,  偏好 标签 z ∈ {−1, 1} 表明 该 文档
是 更多 相关   查询. 在 弱监督, z 是 deﬁned 通过 
传统 IR 模型 的 查询 似然:

z = sign(pQL(q, d1) − pQL(q, d2)),

其中 pQL 表示  查询 概率 用  Dirichlet 先验, 和
‘sign’ extracts  sign 的  真实 数量. 因此, 给定  训练
实例 (q, d1, d2, z), SNRM 训练 其 参数 通过 minimizing 
以下 损失函数

min L(q, d1, d2, z) + λL1([yq, yd1, yd2]),

其中 L(q, d1, d2, z) = max{0, (cid:15) − z[f (q, d1) − f (q, d2)]} 是  成对
合页损失 用 间隔, L1 是  ‘1 正则化 超过  concatenation
的  表示 yq, yd1 和 yd2, 和  hyper-参数 λ > 0
controls  稀疏性 的  学会了 表示.

4.1.6 表示中 多-模态 查询 和 文档

在 交叉-模态 搜索 用户 进行 搜索 跨越 多个 模态
(e.g.,  查询 是 文本 和  文档 是 images). 如果  查询
和 文档 是 表示了 在 diﬀerent 模态, 则 那里 exists
 signiﬁcant 鸿沟 之间 它们. 因此,  关键 是  创建 共同
(模态 agnostic) 表示 用于 查询 和 文档. 深度
学习 可以 确实 fulﬁll  需求, 和 模型 是 提出 用于 
目的.

4.1. 基于表示学习的匹配

67

深度 CCA

一 流行 方法  多-模态 匹配 是  学习  潜在
嵌入 空间 其中 multimedia 对象 (e.g., images 和 文本) 是
均匀地 表示了. 典型相关分析 (CCA) (Hardoon
等人, 2004) 是 这样的  方法 该 发现s 线性 projections 该 maximize
 相关 之间  投影 vectors 的 对象 从  二
原始 spaces.

 增强  表示 能力, Andrew 等人 (2013) 和 Yan
和 Mikolajczyk (2015) 提出  扩展 CCA 到  深度学习
框架. 深度 CCA 直接地 学习 非线性 mappings 用于  任务 的
图像-文本 匹配. Speciﬁcally, 它 计算 表示 的 对象
在  二 spaces (e.g.,  文本 查询 和  图像 文档) 通过 passing 它们 通过 多个 stacked 层 的 非线性 transformations,
作为 显示 在 图 4.9.

深度 CCA 表示  文本 (e.g., 查询 q) 作为  向量 的 术语.
每个 元素 的  向量 是  tf-idf 值 的  相应 术语.
 向量 是 输入 到  文本 网络 该 由组成 的 n stacked
triplets 的 完全地 connected (FC) 层, ReLU 层, 和 随机失活 层.
深度 CCA 表示  图像 (e.g., 图像 文档 d) 作为  raw
图像 向量.  向量 是 输入 到  图像 网络 该 由组成
的 m stacked doubles 的 卷积 层 和 ReLU 层, 和  最后
完全地 connected 层.

 目标 的 学习 是  联合地 estimate  参数 在  文本
网络 和  图像 网络 因此 该  深度 非线性 mappings
的  二 类型 的 数据 是 maximally correlated. 假设 该 (Q, D)
表示  vectors 的  文本 查询 和  相关 图像 文档,
分别地. 进一步, 假设 该 Θ1 和 Θ2 是  参数 的
 文本 网络 和 图像 网络, 分别地. 因此, 深度 CCA
amounts  maximize  以下 目标 函数:

max
Θ1,Θ2

corr(TextNN(Q; Θ1), ImageNN(D; Θ2)),

其中 “corr” 是  相关 的 二 vectors, “TextNN” 和 “ImageNN”
是  文本 网络 和 图像 网络, 分别地.

68

搜索中的深度匹配模型

图 4.9: 深度 CCA 架构. 它 由组成 的 二 深度 网络 用于 文本 和
图像.

对抗 交叉 模态 检索

对抗学习 可以 是 采用了  构建  共同空间
在 该 物品 在 diﬀerent 模态 是 表示了 和 比较了,
作为 显示 在 (Wang 等人, 2017).  方法, 称为 对抗
交叉 模态 检索 (ACMR), 进行  极小极大博弈 涉及中
二 players:  特征投影器 和  模态 classiﬁer.  特征
projector 是 responsible 用于 生成中 模态-invariant 表示
用于 物品 从 diﬀerent 模态 在  共同空间. 它 具有 
目标 的 confusing  模态 classiﬁer 作为  adversary. 
模态 classiﬁer 是 responsible 用于 distinguishing 物品 从 其
模态. 通过 bringing 在  模态 classiﬁer, 它 是 expected 该
 学习 的 特征投影器 可以 是 执行了 更多 eﬀectively, 在
 sense 该 模态 不变性 是 获得了.

Speciﬁcally,  文本 分支 的 ACMR 取 bag-的-词 特征 作为
输入.  深度神经网络, 表示为 作为 fT (·; θT ), 是 使用了  进行
文本 特征 投影.  图像 分支 的 ACMR 取 CNN 图像
特征 作为 输入.  深度神经网络, 表示为 作为 fV (·; θV ), 是 使用了
 进行 图像 特征 投影. θV 和 θT 是  参数 在
 二 网络.

给定  集合 的 N 训练 triples D = {(vi, ti, yi)}N

i=1, 其中 vi 是 
图像 特征 向量, ti 是  文本 特征 向量, 和 yi 是  类别
的 vi 和 ti, ACMR deﬁnes 其 模态 classiﬁer 和 特征投影器
作为 follows.

 模态 classiﬁer D 是  馈 前向 神经网络 用 参数 θD 该 预测  概率 的 模态 给定  实例
(图像 或 文本).  投影 特征 的  图像 是 assigned 一-hot

textqueryimagedocFC1’ReLU1’随机失活1’(cid:1)(cid:1)Conv1ReLU1ReLU7FC8(cid:1)(cid:1)Dropoutn’CCAobjective:tracenorm tf-idfrawimagerep.4.1. 基于表示学习的匹配

69

图 4.10: Flowchart 的 ACMR.  模态 classiﬁer tries  区分 
物品 从 其 模态.  特征 projector manages  confuse  模态
classiﬁer 通过 生成中 模态-invariant 和 判别 表示.

向量 [0, 1], 同时  投影 特征 的  文本 是 assigned 一-hot
向量 [1, 0].  模态 classiﬁer acts 作为  adversary. 它 manages 
minimize  对抗损失:

Ladv(ΘD) = −

1
N

N
X

i=1

(mi · (log D(fV (vi); θD) + log(1 − D(fT (ti); θD)))) ,

其中 mi 是  模态 标签 的  i-th 实例, expressed 作为 一-hot
向量.

 特征投影器 进行 模态-invariant 嵌入 的
文本 和 images 到  共同空间, consisting 的 二 parts: 标签
预测 和 结构保持.  标签预测 minimizes 
损失函数 Limd  ensure 该  特征 表示 belonging
  相同 categories 是 suﬃciently close.  结构保持
minimizes  损失函数 Limi  ensure 该  特征 表示
belonging   相同 categories 是 suﬃciently close 跨越 模态,
和  特征 表示 belonging  diﬀerent categories 是
suﬃciently far apart 之内  模态.  整体 生成损失,
表示为 作为 Lemb, 是  组合 的 标签预测 损失 Limd, 结构
保持 损失 Limi, 和  正则化 术语 Lreg:

Lemb(θV , θT , θimd) = α · Limd + β · Limi + Lreg,

其中 α > 0 和 β > 0 是 trading-oﬀ coeﬃcients.

modalityclassifiertextqueryimagedocCNNimagefeatureBagofwordsfeatureTextfeatureprojection𝑓𝑇;𝜃%Imagefeatureprojection𝑓𝑇;𝜃%图像/TextmodalityclassificationInter-modaladversarialloss𝐿’()labelpredictionIntra-modaldiscriminationloss𝐿*+(Tripletconstraintstructurepreservation𝑣*,𝑡/0,𝑡12𝑡*,𝑣/0,𝑣12Inter-modalinvarianceloss𝐿*+*featureprojectorc0c170

搜索中的深度匹配模型

表 4.2: Performances 的 表示学习 方法 在 MSRP 数据集.

.

TF-IDF (基线)
DSSM
CLSM
ARC-I

准确率

0.7031
0.7009
0.6980
0.6960

F1
0.7762
0.8096
0.8042
0.8027

最后,  学习 的  ACMR 模型 是 进行了 通过 联合地
minimizing  对抗 和  生成 losses, 作为  minimax
game:

(ˆθV , ˆθT , ˆθimd) = arg min

θV ,θT ,θimd

(Lemb(θV , θT , θimd) − Ladv(ˆθD)),

ˆθD = arg max
θD

(Lemb(ˆθV , ˆθT , ˆθimd) − Ladv(θD)).

4.1.7 实验的 Results

我们 呈现  实验的 results 的 搜索 相关性 通过  方法
的 表示学习, 报告了 在 Yin 和 Schütze (2015) 和 Pang
等人 (2017b). 在  实验,  基准 数据 的 MSRP1 是
利用了, 和 准确率 和 F1 是 采用 作为 评估 测量. 
results 在 表 4.2 表明 该  方法 基于 在 表示
学习 可以 outperform  基线 的 TF-IDF 从角度 F1.

我们 也 呈现  实验的 results 的  多-模态 搜索
在 表 4.3.  实验 是 基于 在  Wikipedia 数据集 和
报告了 在 Wang 等人 (2017). 在  实验, 均值 Average
精确率 (MAP) 是 使用了 作为  评估 度量.  results 表明
该  多-模态 匹配 方法 的 ACMR 可以 signiﬁcantly
outperform  基线, 特别 当 深度 特征 是 使用了.

1https://www.microsoft.com/en-我们/download/细节.aspx?id=52398

4.2. 基于匹配函数学习的匹配

71

表 4.3: Performances 的 多-模态 匹配 方法 在 Wikipedia 数据集 在
术语 的 MAP

.
图像  文本

文本  图像

average

CCA (浅层 特征)
CCA (深度 特征)
ACMR (浅层 特征)
ACMR (深度 特征)

0.255
0.267
0.366
0.619

0.185
0.222
0.277
0.489

0.220
0.245
0.322
0.546

4.2 基于匹配函数学习的匹配

4.2.1 通用 框架

 匹配程度 之间 查询 和 文档 可以 是 determined
通过 aggregating  局部 和 全局 匹配 signals 之间  查询
和 文档.  匹配 signals, 以及 其 locations, 可以 是
捕获了 从  输入 查询 和 文档 表示.

Researchers 提出  使用 深度 神经 网络  自动地
学习  匹配 patterns 之间 查询 和 文档, 称为 
这里 作为 匹配函数学习. 那里 是 二 关键 问题 在
该 方法: 1) 如何  表示 和 计算  匹配 signals,
和 2) 如何  aggregate  匹配 signals  计算  ﬁnal
匹配分数.

图 4.11 显示  通用 框架. 在  框架, 
查询 和 文档 是 比较了 用 每个 其他  创建 匹配
signals, 和  匹配 signals 是 aggregated  输出  匹配
分数, 所有 在  单一 神经网络.

一 方法 是 首先  let  查询 和 文档 interact 基于
在 其 raw 表示, yielding  数量 的 局部 匹配 signals,
和 则  aggregate  局部 匹配 signals  输出  ﬁnal
匹配分数. 另一个 方法 是  创建  表示 的
查询 和 文档 以及 其 interactions 在 两者 局部 和
全局 levels 用  单一 神经网络 通常 使用中 注意力.

72

搜索中的深度匹配模型

图 4.11: 查询-文档 基于匹配函数学习的匹配.

4.2.2 学习 匹配函数 用 匹配矩阵

匹配矩阵 是 使用了 用于 storing 词-水平 匹配 signals 和 其
positions.  columns 和 rows 的  匹配矩阵 correspond 
 词 在  查询 和 文档, 分别地. 每个 条目 表示
 位置 的 匹配, 和  值 的 每个 条目 表示 
程度 的 匹配.  匹配矩阵 是 输入   神经网络
作为  整体.

Advantages 的 学习 用  匹配矩阵 是 作为 follows: (1)
 匹配矩阵 是 精确 在  sense 该  局部 匹配
信息 (程度 和 位置) 是 accurately 表示了 在 它. (2)
 匹配矩阵 是 直观 在  sense 该  局部 匹配
信息 可以 是 easily visualized 和 interpreted.

卷积 匹配模型 (ARC-II)

 卷积 匹配模型 (Arc-II) (Hu 等人, 2014) 是 直接地
构建了 在  interactions 之间 查询 和 文档.  想法 是 首先
 let  查询 和 文档 interact 用 其 raw 表示
和 则 捕获  匹配 signals 从  interactions.

作为 显示 在 图 4.12, 在  首先 层, ARC-II 取  sliding
window 在  查询 和 文档, 和 模型  interactions 之内
 window 在 每个 位置 使用中 一-维 卷积. 用于
片段 i 的 查询 q 和 片段 j 的 文档 d, ARC-II 构建
 交互 表示:

i,j = [qT
z0

i:i+k1−1, dT

j:j+k1−1]T ,

其中 k1 是  宽度 的 sliding window, 和 qT

i:i+k1−1 = [qT

i , qT

i+1, · · · , qT

i+k1−1]T

QueryDocumentMatching signalsAggregationmatchingscore4.2. 基于匹配函数学习的匹配

73

j:j+k1−1 = [dT

(和 dT
j+k1−1]T ) 是  concatenation 的 嵌入 vectors 的 k1 词 在  查询 片段 (和 在  文档
片段). 因此,  相应 值 在  特征 map f 是

j+1, · · · , dT

j , dT

z(1,f )
i,j = g(z0

i,j) · σ(w(1,f )z0

i,j + b(1,f )),

其中 σ 是  激活函数, w(1,f ) 和 b(1,f ) 是  卷积
参数, 和 g(·) 是  门控 函数 这样的 该 g(·) = 0 如果 所有 
元素 在  输入 vectors 相等 0, 否则 g(·) = 1. 这里, g(·)
工作 作为  零 padding 函数. 用于 所有 可能 查询 词 和
文档 词,  一-维 卷积 层 输出 
二-维 匹配矩阵.

 下一个 层 进行 二-维 max-池化 在 每个 nonoverlapping 2×2-window.  (i, j)-th 条目 在  输出 矩阵 是

z(2,f )
i,j = max

(cid:16)

2i−1,2j−1, z(1,f )
z(1,f )

2i−1,2j, z(1,f )

2i,2j−1, z(1,f )

2i,2j

(cid:17)

.

 max-池化 层 快速 shrinks  大小 的 匹配矩阵 通过
过滤ing out weak (possibly noisy) 匹配 signals.

则, 二-维 卷积 是 应用   输出 矩阵
的  max-池化 层. 该 是, 交互 表示 是 创建了
之内  sliding window 的 大小 k3 × k3 在 每个 位置 在  矩阵
使用中 二-维 卷积.  (i, j)-th 值 在  特征
map f 是

i,j = g(Z(2)
z(3,f )

i,j ) · σ

(cid:16)

W(3,f )Z(2)

i,j + b(3,f )(cid:17)

,

其中 W(3,f ) 和 b(3,f ) 是  卷积 参数 和 Z(2)
i,j 是
 输入 矩阵. 更多 层 的 二-维 max-池化 和
卷积 层 可以 是 added afterward.

在  最后 层,  MLP 是 利用了  总结  匹配

signals 和 输出  匹配分数

f (q, d) = W2σ

(cid:16)

W1Z(3) + b1

(cid:17)

+ b2,

其中 Z(3) 是  最后 层 特征 map.

 训练   模型 参数, ARC-II 使 使用 的  相同
判别 策略 作为 在 ARC-I. 该 是, 给定 查询 q, 相关

74

搜索中的深度匹配模型

图 4.12: 卷积 匹配模型 (ARC-II)

查询-文档 对 (q, d) 和 不相关 查询-文档 对 (q, d0).
ARC-II minimizes  目标:

L = X
(q,d)∈C

X

(q,d0)∈C0

(cid:2)1 − f (q, d) + f (q, d0)(cid:3)

+ ,

其中 C 和 C0 包含  相关 和 不相关 查询-文档
pairs, 分别地.

MatchPyramid

 卷积 匹配模型 ARC-II 使 early interactions
之间 查询 和 文档. 然而,  meanings 的  interactions (i.e.,  一-维 卷积) 是 不 清晰. Pang 等人
(2016b) 指出 out 该  匹配矩阵 可以 是 构建了 更多
straightforwardly.  提出 模型, 称为 MatchPyramid, rede-
ﬁnes  匹配矩阵 作为  词-水平 相似度 矩阵. 则, 
二-维 卷积神经网络 是 exploited  提取
查询-文档匹配 patterns, 总结  匹配 signals,
和 计算  ﬁnal 匹配分数.  主要 想法 的 MatchPyramid
是  视角 文本 匹配 作为 图像 识别, 通过 取中  匹配
矩阵 作为  图像, 作为 显示 在 图 4.13.  输入   卷积神经网络 是 匹配矩阵 M, 其中 元素 Mij
表示  基本 交互 的  i-th 查询 词 qi 和  j-th
文档 词 dj. 在 通用, Mij stands 用于  相似度 之间
qi 和 dj, 该 可以 具有 diﬀerent deﬁnitions.  表明 函数
Mij = 1qi=dj 可以 是 使用了  产生 或 1 或 0  表明 是否
 二 词 是 完全相同.  embeddings 的 查询 词 和 文档 词 可以 也 是 使用了  表示 语义 相似之处 之间

documentqueryMLPmatchingscore…2D 池化2D convolution更多 2D 卷积 和 池化1D 卷积4.2. 基于匹配函数学习的匹配

75

图 4.13: 架构 的 MatchPyramid

 二 词. 用于 例子, 余弦相似度: Mij = cos(qi, dj) 其中
qi 和 dj 是  embeddings 的 qi 和 dj 分别地, 和 点积:
Mij = qT

i dj.

MatchPyramid 是  二-维 卷积神经网络
用  匹配矩阵 M 作为 输入. Let z(0) = M.  k-th 核
w(1,k) scans  匹配矩阵 z(0) 和 生成  特征 map z(1,k)
其 值 是

z(1,k)
ij = σ





rk−1
X

rk−1
X

s,t z(0)
w(1,k)

i+s,j+t + b(1,k)


 ,

s=0

t=0

其中 rk 是  大小 的  k-th 核. 动态 池化 是 则 利用了
 deal 用  变异性 在 文本 长度.  ﬁxed-大小 特征 maps
outputted 通过 动态 池化 是:

z(2,k)
ij = max
0≤s≤dk

max
0≤t≤d0
k

z(1,k)
i·dk+s,j·d0

k+t,

其中 dk 和 d0
k 是  宽度 和 长度 的  池化 核. 用
动态 池化  输出 特征 map 成为 ﬁxed-sized. 更多
层 的 卷积 和 动态 池化 可以 是 stacked.

在  最后 层, MatchPyramid 利用  MLP  产生 

ﬁnal 匹配分数:

[s0, s1]T = f (q, d) = W2σ (W1z + b1) + b2,

其中 z 是  输入 特征 map 和 W1, W2, b1 和 b2 是 
参数.

 学习  模型 参数, MatchPyramid 利用  softmax
函数 和 交叉 熵 损失. 给定  集合 的 N 训练 triples D =

querydocumentq1q2q5...querydocumentw1w2w5……MLPmatchingscorematchingmatrixsimilarityoperator2-Dconvolution2-Dpooling76

搜索中的深度匹配模型

图 4.14:  eﬀects 的  2-维 卷积 核. 图 从 (Pang
等人, 2016b).

{(qn, dn, rn)}N
1 用于 相关 和 0 用于 不相关. 交叉 熵 损失 是 deﬁned 作为:

n=1 其中 rn ∈ {0, 1} 是  真实值 相关性 标签,

L = − X

(q,d,r)∈D

[r log(p(rel|q, d)) + (1 − r) log(1 − p(rel|q, d))] ,

其中 p(rel|q, d) = es1
 查询 q, 给定 通过  softmax 函数.

es0 +es1 是  概率 该 文档 d 是 相关

一 attractive 特性 的  二-维 卷积 是
该 它 是 capable  自动地 提取 高 水平 (soft) 匹配
patterns 和 store 它们 在  kernels, 该 是 相似  visual 模式
提取 在 图像 识别. 图 4.14 说明  例子, 用
 handcrafted 匹配矩阵 基于 在  指标 函数. 给定
二 kernels, 它 是 清晰 该  首先 卷积 层 可以 捕获
两者 n-gram 匹配 signals 这样的 作为 “向下  ages” 和 n-术语
匹配 signals 这样的 作为 “(noodles 和 dumplings) v.s. (dumplings 和
noodles)”, 作为 显示 在  特征 maps 的  首先 层. 则, 
秒 卷积 层 使 compositions 和 形式 更高 水平
匹配 patterns, 作为 显示 在  特征 map 的  秒 层.

匹配-SRNN

二-维 Recurrent 神经 网络 (RNN) 可以 也 是 使用了 
discover 匹配 patterns 在  匹配矩阵. Wan 等人 (2016b)

4.2. 基于匹配函数学习的匹配

77

提出  方法  划分  匹配 的 二 文本 到  序列
的 sub-问题 的 匹配, 和 solve  sub-问题 recursively.
 提出 模型, 称为 MatchSRNN, 应用  二-维
RNN (Graves 等人, 2007)  scan  匹配矩阵 从  top-左
  bottom-正确.  陈述 在  最后 (bottom-正确) 位置 是
考虑了 作为  整体 表示 的 匹配.

作为 显示 在 图 4.15, 匹配-SRNN 由组成 的 三 组件:
 NTN 用于 discovering 词-水平 匹配 signals,  spatial RNN 用于
summarizing  句子-水平 匹配 表示, 和  线性
层 用于 计算中  ﬁnal 匹配分数.

第一, 给定 查询 q 和 文档 d,  NTN 是 利用了  计算
 相似度 之间 i-th 查询 词 qi 和  j-th 文档 词
dj:

s(qi, dj) = uT σ

i M[1:r]dj + V
qT

"

#

!

+ b

,

qi
dj

其中 qi 和 dj 是  embeddings 的  i-th 查询 词 和 
j-th 文档 词.

下一个,  spatial RNN (二-维 RNN) 是 采用了  scan
 outputted 匹配矩阵 recursively. Speciﬁcally,  计算 
匹配 表示 之间  查询 preﬁx q[1:i] 和 文档
preﬁx q[1:j],  表示 的 其 preﬁxes 是 首先 计算了:

hi−1,j =SpatialRNN(q[1:i−1], d[1:j]),
hi−1,j−1 =SpatialRNN(q[1:i−1], d[1:j−1]),
hi,j−1 =SpatialRNN(q[1:i], d[1:j−1]),

其中 SpatialRNN(·, ·) 是  二-维 RNN 应用  
preﬁxes. 则,  匹配 表示 是 计算了 作为

hi,j = SpatialRNN(q[1:i], d[1:j]) = f (hi−1,j, hi,j−1, hi−1,j−1, sqi,di),

其中 f 表示  模型 的 二-维 RNN. 而非 twodimensional RNN 更多 强大 模型 这样的 作为 二-维 GRU
和 LSTM 可以 也 是 exploited.

 最后 表示 在  正确 bottom corner, h|q|,|d|, reﬂects
 全局 匹配 表示 之间  查询 和 文档.


78

搜索中的深度匹配模型

图 4.15: 匹配-SRNN 模型.

最后,  线性 函数 是 使用了  计算  ﬁnal 匹配分数:

f (q, d) = wh|q|,|d| + b,

其中 w 和 b 是 参数.

 学习  模型 参数, 匹配-SRNN 利用  成对
合页损失. 给定 查询 q,  相关 查询-文档 对 (q, d+)
在  训练数据 应当 receive  更高 分数 比  不相关
查询-文档 对 (q, d−), deﬁned 作为:

‘(q, d+, d−) = max(0, 1 − f (q, d+) + f (q, d−)).

给定  训练数据, 所有  参数 在  匹配-SRNN 模型
是 训练了 通过 反向传播.

4.2.3 学习 匹配函数 用 注意力

 最近 趋势 是  利用 注意力, 该 是 inspired 通过  注意力
机制 在 人类 cognition. 注意力 是 successfully 应用 
任务 在 NLP 和 IR, 包括中 查询-文档匹配.

Decomposable 注意力 模型

Parikh 等人 (2016) 指出 out 该 匹配 signals 可以 是 捕获了
和 表示了 用  decomposable 注意力机制. 作为 显示
在 图 4.16,  模型 由组成 的 三 steps: attend, 比较, 和
aggregate. 给定  查询 和  文档, 其中 每个 词 在 
查询 和  文档 是 表示了 通过  嵌入 向量, 
模型 首先 创建  soft 对齐 矩阵 使用中 注意力; 则 它 使用

 cat sat 在  … dog playsball 在 …thecatsatonthethe   dog   playsball   onthe    dog   playsball   onthecatsatonthematchingmatrixsimilarityoperator:NTN2-DRNNlinearlayermatchingscore4.2. 基于匹配函数学习的匹配

79

 (soft) 对齐  decompose  任务 到 subproblems; ﬁnally, 它
merges  results 的  subproblems  产生  ﬁnal 匹配
分数.

Speciﬁcally, 给定  查询-文档 对 (q, d) 其中 每个 词
在 q 是 表示了 作为  嵌入 向量 q = (q1, · · · , q|q|) 和 |q|
是  数量 的 词 在 q, 和 每个 词 在 d 是 表示了 作为 
嵌入 向量 d = (d1, · · · , d|d|) 和 |d| 是  数量 的 词 在
d. 在  attend 步骤  注意力 矩阵 之间 每个 查询 词 和
文档 词 是 构建了.  unnormalized 注意力 权重 eij
是 计算了 用  decomposable 函数:

eij = F 0(qi, dj) = F (qi)T F (dj),

其中 F 是  前馈神经网络. 用  注意力 weights,
 整体 文档 aligned   i-th 查询 词 是

βi =

|d|
X

j=1

exp(eij)
k=1 exp(eik)

P|d|

dj.

类似地,  整体 查询 aligned   j-th 文档 词 是

αj =

|q|
X

i=1

exp(eij)
k=1 exp(ekj)

P|q|

qi.

在  比较 步骤, 每个 查询 词 和 其 aligned 版本
i=1 是 比较了 分别地 用  馈-前向 网络 G:

{(qi, βi)}|q|

v1,i = G([qT

i , βT

i ]T ),

∀i = 1, · · · , |q|,

其中 [·, ·] concatenates 二 vectors. 每个 文档 词 和 其
aligned 版本 {(dj, αj)}|d|
j=1 是 比较了 分别地 用  相同
馈-前向 网络 G:

v2,j = G([dT

j , αT

j ]T ),

∀j = 1, · · · , |d|.

最后 在  aggregate 步骤,  二 集合 的 比较 signals

{v1,i} 和 {v2,j} 是 summed 分别地:

v1 =

|q|
X

i=1

v1,i,

v2 =

|d|
X

j=1

v1,j.

80

搜索中的深度匹配模型

图 4.16: Decomposable 注意力 模型 用于 匹配.

 二 aggregated vectors 是 则 输入   馈 前向 网络
followed 通过  线性 层 H, 给出中 多个-类 分数:

ˆy = H([vT

1 , vT

2 ]T ).

 预测了 类 (e.g., 相关 或 不相关) 是 decided 通过 ˆy =
arg maxi yi.

在 训练 的  模型, 交叉-熵 损失 是 利用了:

L =

1
N

N
X

C
X

n=1

c=1

y(n)
c

log

c

exp(ˆy(n)
)
c0=1 exp(ˆy(n)
c0 )

PC

,

其中 C 是  数量 的 类2, N 是  数量 的 实例 在 
训练数据.

匹配 用 BERT

最近, BERT ( 来自Transformer的双向编码器表示) 成为  最先进的 模型 用于 语言 理解
任务 用 其 更好 performances (Devlin 等人, 2019). 在 pre-训练

2那里 是 二 类 用于  查询-文档匹配 任务: 相关 和

不相关.

 cat sat 在  mat… dog plays ball 在 …      dog     plays   ball     onthecatsatonthematdecomposable 注意力 权重𝐹‘mat,ball=𝐹mat𝐹(ball),匹配 scoreH,查询 词-aligned doc subphrase 匹配 signals, 生成了 通过 aneuralnetworkdocument 词-aligned 查询 subphrase 匹配 signals, 生成了 通过 aneuralnetwork  attendcompareaggregate4.2. 基于匹配函数学习的匹配

81

图 4.17: 精细-调整 BERT 用于 查询-文档匹配.

的 BERT  表示 的 二 文本 是 学会了 从  大
量 的 未标注数据 通过 mask 语言 建模 和 下一个
句子 预测. 在 ﬁne-调整  表示 是 进一步 re-
ﬁned 用于  downstream 任务 用  输出 层 added 在 top 的 
模型 和  小 量 的 任务-speciﬁc 标注数据.

当 应用  搜索, BERT 可以 是 利用了  计算 
匹配程度 之间 查询 和 文档, 作为 长 作为 训练数据
是 提供 (Nogueira 和 Cho, 2019). 该 是,  pre-训练了 BERT
模型 可以 是 适应  查询-文档匹配 用 ﬁne-调整.

图 4.17 显示  流行 方法 的 使用中 BERT 用于 querydocument 匹配. 给定  查询-文档 对 (q, d),  输入
  BERT 模型 包括 查询 词, 文档 词: “[CLS],
q1, · · · , qN , [SEP], d1, · · · , dM , [SEP]”, 其中 “[CLS]” 是  词元 
表明 是否  查询-文档 对 是 相关 或 不, “[SEP]” 是
 词元  表明  分离 的 查询 和 文档, 和 qi 和
dj 是  i-th 查询 词 和  j-th 文档 词, 分别地.

BERTCT[SEP]T1TN…T’1…T’ME[CLS]E[SEP]E1EN…E’1…E’M[CLS]q1[SEP]qN…d1dM…wordembeddingssegmentembeddingsEqEqEqEq…EdEd…positionembeddingsE0E1EN+1EN…EN+2EM+N+1…queryqdocumentdsoftmaxE[SEP]T[SEP][SEP]EdEM+N+2!rel%,')82

搜索中的深度匹配模型

 查询 (和 文档) 是 padded 或 truncated  具有 N (和
M ) 词. 每个 词 是 表示了 用 其 嵌入.  输入
嵌入 的  词 是  sum 的  相应 词嵌入,
 片段 嵌入, 和  位置 嵌入.

 模型 的 BERT 是  编码器 的 Transformer (Vaswani et
al., 2017), 该 输出  序列 的 高 水平 语义 表示 用于  special 输入 tokens 和 查询 和 文档 词:
“C, T1, · · · , TN , T[SEP], T 0
M , T 0
[SEP]”, 其中 C 是  representa-
1, · · · , T 0
tion 的  词元 [CLS], T1, · · · , TN 的  查询 词, T 0
M
的  文档 词, T[SEP] 和 T 0
[SEP] 的  二 separators. 
表示 的  [CLS] 词元 C 是 fed 到  输出 层 (e.g.,
单一 层 神经网络)  获得 p(rel|q, d), 该 表示 
概率 的 文档’s 是 相关  查询.

1, · · · , T 0

 BERTLARGE 模型 released 通过 Google 是 广泛使用 作为 pretrained 模型3. 它 是 则 采用了 在  ﬁne-调整 用于 查询 文档
匹配. 假设 该 我们 是 给定  集合 的 训练 triples D =
{(qn, dn, rn)}N
n=1 其中 rn ∈ {0, 1} 是  真实值 标签.  crossentropy 损失 是 计算了:

L = X

(q,d,r)∈D

−r log(p(rel|q, d)) − (1 − r) log(1 − p(rel|q, d)).

比较了   现有 模型, BERT oﬀers 若干 advantages
用于 查询-文档匹配. 第一, 在 BERT,  查询 和 文档
是 联合地 输入   模型, 使 它 可能  同时地
表示  intra-查询, intra-文档, 和 inter 查询-文档
interactions. 秒, 在 BERT,  表示 的 查询 和 文档 以及 查询-文档 交互 是 变换 多个
次 在  层次 架构. 作为  结果, 复杂 局部
和 全局 匹配 patterns 可以 是 表示了. 第三, BERT 使用 
pre-训练/ﬁne-调整 框架, 其中  pre-训练了 BERT 模型
可以 利用  信息 在  大 量 的 未标注数据.
其他 匹配 模型, 然而, 具有 更少 强大 表示
abilities 和 因此 cannot 实现 相似 performances. 研究 显示
该 pre-训练 的 BERT 可以 使  模型 favor 文本 pairs 该 是
semantically 相似 和 因此 可以 使  模型 执行 very 好 在

3https://github.com/google-研究/bert

4.2. 基于匹配函数学习的匹配

83

匹配 (Nogueira 和 Cho, 2019; Nogueira 等人, 2019; Qiao 等人,
2019).

4.2.4 学习 匹配 函数 在 搜索

那里 exist diﬀerences 之间  匹配 任务 在 搜索 和 那些
在 NLP.  匹配 任务 在 搜索 是 主要地 关于 主题 相关性,
同时  匹配 任务 在 NLP 是 主要地 concerned 用 语义.
用于 例子, 匹配 模型 在 搜索 应当 是 able  处理 精确
匹配 signals very 好, 这样的 作为 查询 术语 重要性 和 多样性
的 匹配 (Guo 等人, 2016). 若干 匹配 模型 tailored 用于
搜索 是 开发了 和 proved  是 eﬀective.

深度 相关性 匹配模型 (DRMM)

在 (Guo 等人, 2016),  相关性 匹配模型 称为 DRMM 是
提出. 图 4.18 显示  模型 架构.  查询 q 和
 文档 d 是 表示了 作为 二 集合 的 词 vectors 分别地:
q = {q1, q2, · · · , q|q|} 和 d = {w1, w2, · · · , w|d|}, 其中 qi 和 wj 表示
 查询 词 向量 和  文档 词 向量 两者 生成了
通过 词2Vec. 用于 每个 查询 词 qi,  匹配直方图 z(0)
是
构建了  刻画  交互 之间 qi 和  整体
文档:

i

z(0)
i = h(qi

O d),

用于 i = 1, · · · , |q|, 其中 N 表示 余弦相似度 计算 之间
qi 和 所有 词 在 d, outputting  集合 的 余弦相似度 分数 在 
间隔 的 [−1, 1]. 则, 函数 h discretizes  间隔 到  集合 的
ordered bins, counts  数量 的 相似度 分数 在 每个 bin, 和
计算  对数 的  counts, 生成中  histogram 向量
z(0)
i

i

是 则 passed 通过 L-层 馈-前向 层
 生成  匹配分数 用于 每个 查询 词 qi, 表示为 作为 z(L)
.
给定  匹配 分数 的 个体 查询 词,  ﬁnal 匹配
分数 之间 q 和 d, f (q, d), 是 计算了 作为  weighted sum 的 

i

.
 向量 z(0)

84

搜索中的深度匹配模型

匹配 分数:

f (q, d) =

|q|
X

i=1

giz(L)
i

,

其中  权重 的 查询 词 qi, gi, 是 生成了 通过  术语 门控
网络:

gi =

exp(wgxi)
j=1 exp(wgxj)

,

P|q|

其中 wg 是  参数 向量 在  词项门控网络 和 xj
(j = 1, · · · , |q|) 是  特征 向量  刻画  重要性 的
查询 词 qi.  特征 可以 是 术语 向量 或 inverse 文档
频率 (IDF).

 参数 在  馈-前向 网络 和 词项门控网络
是 联合地 学会了. 给定  训练 例子 (q, d+, d−), 其中 d+
和 d− 分别地 表示  相关 文档 和  non-相关
文档,  学习 的 DRMM amounts  minimizing  成对
合页损失 用 间隔:

L = max(0, 1 − f (q, d+) + f (q, d−)).

 随机梯度下降 方法 的 Adagrad 用 mini-batches 是
应用  进行  最小化. 早停 策略 是 采用
用于 正则化.

核 基于 神经 排序模型 (K-NRM)

DRMM 是 eﬀective 在 建模  interactions 之间 查询 词
和 文档 词. 然而,  histogram 池化 部分 (i.e., counting
的 相似度 值 在 每个 bin) 是 不  diﬀerentiable 函数, 该
hinders 端--端 学习 的  匹配模型.  cope 用 
问题, DRMM 使 使用 的 pre-训练了 词 vectors  表示
 词 在  查询 和 文档. Xiong 等人 (2017) 提出
 相关性 匹配模型 称为 K-NRM. 在  模型, 而非
counting  相似度 分数, 核池化 是 采用了  刻画
 匹配程度 之间 每个 查询 词 和  文档. 作为 
结果,  模型 可以 是 训练了 端--端.

作为 显示 在 图 4.19, 给定  查询 q 和  文档 d,
K-NRM 首先 使用  嵌入 层  map 每个 词 (在 q 和 d)

4.2. 基于匹配函数学习的匹配

85

图 4.18: 深度 相关性 匹配模型.

图 4.19: 核 基于 神经 排序模型.

到  嵌入 向量. 则, 它 构建  翻译 矩阵 (i.e.,
匹配矩阵) M 其中  (i, j)-th 元素 在 Mij 是  嵌入
相似度 (余弦相似度) 之间  i-th 查询 词 和  j-th
文档 词. 则, 它 应用  kennel 池化 operator  每个 row
的 M (相应  每个 查询 词), 生成中  K-维
向量 ~K. Speciﬁcally,  k-th 维度 的  池化 向量 的 
i-th 查询 词 是 deﬁned 作为  RBF 核 函数:

Kk(Mi) = X

exp

−

j

(Mij − µk)2
2σ2
k

!

,

其中 Mi 是  i-th row 的 M, Mij 是  j-th 元素 在 Mi, 和 µk

 cat sat 在  mat… dog playsball 在 …      dog   playsball     onthecatsatonthematMatchingmatrixbasedonwordembeddingsFeed-forwardMatchingNetworkFeed-前向 MatchingNetworkFeed-前向 MatchingNetworkweightedsummatching scorethe cat sat 在  mat…TermGatingNetworkg1g2…Matchinghistogrammappingg|Q| cat sat 在  mat… dog playsball 在 …      dog   playsball     onthecatsatonthemattranslation(匹配)matrixMkernel 池化...核池化!Soft-TFfeatures"matchingscoretanh'"+)核池化!*exp−/0*−1232523embeddinglayer……wordembeddingscosquerydocument 
86

搜索中的深度匹配模型

和 σk 是  均值 和 方差 的  RBF 核, 分别地.

给定  池化 vectors 的 所有 查询 词,  池化 vectors

是 summed  创建  soft-TF 特征:

φ(M ) =

|q|
X

i=1

log ~K(Mi),

其中 ~K(Mi) = [K1(Mi), · · · , KK(Mi)] 和 log 是 应用  每个
维度 的 ~K(Mi). 最后,  soft-TF 特征 是 结合 一起,
yielding  ﬁnal 匹配分数

f (q, d) = tanh(hw, φ(M )i + b),

其中 w 和 b 是 weights 和 偏置 分别地.

一 优势 的 K-NRM 是 该 学习 可以 是 进行了 在 
端--端 方式. 给定  集合 的 训练 例子 D = {(qi, d+
i )}N
其中 d+
i 分别地 表示  相关 文档 和  nonrelevant 文档 w.r.t. qi,  学习 的 K-NRM mounts  minimizing  成对 hinge 损失函数:

i 和 d−

i , d−

i=1,

L(w, b, V) =

N
X

i=1

max(0, 1 − f (qi, d+

i ) + f (qi, d−

i )).

Back propagation 可以 是 采用了 在 学习 的  kernels, 该
使 它 可能  学习 两者  参数 w, b 和 词 embeddings
V 期间 训练.

Duet

基于表示学习的匹配 relies 在  分布式 表示 的 查询 和 文档. 在 对比, 匹配 基于 在
匹配函数学习 relies 在  局部 匹配 表示
的 查询 和 文档. 在  匹配模型 称为 Duet (Mitra
等人, 2017),  混合 方法 是 取 和  advantages 的  二
方法 是 两者 leveraged.

作为 显示 在 图 4.20, Duet 由组成 的 二 分离 深度 神经
网络, 一 该 匹配  查询 和  文档 使用中 局部
表示, 和  其他 该 匹配  查询 和  文档

4.2. 基于匹配函数学习的匹配

87

图 4.20: 模型 架构 的 Duet.

使用中 分布式 表示. 给定  查询 q 和  文档
d,  ﬁnal Duet 匹配分数 f (q, d) 是 deﬁned 作为

f (q, d) = fl(q, d) + fd(q, d),

其中 fl(q, d) 和 fd(q, d) 分别地 表示  局部 和 分布式
匹配 分数.

在  模型 的 fl(q, d), 每个 查询 词 (和 每个 文档 词)
是 表示了 通过  一-hot 向量.  模型 则 创建  二进制
匹配矩阵 X ∈ {0, 1}|d|×|q| 其中  (i, j)-th 条目 表示 
精确 匹配 关系 的  i-th 文档 词 和  j-th 查询
词.  匹配矩阵 X 是 首先 passed 通过  卷积
层.  输出 是 则 passed 通过 二 完全地-connected 层, 
随机失活 层, 和  完全地-connected 层  产生  单一 匹配
分数.

在  模型 的 fd(q, d), (相似  DSSM (Huang 等人, 2013)),
 词 在 q 和 d 是 分别地 表示了 作为 频率 vectors
的 letter n-grams. 则,  向量 的 查询 q 是 passed 通过 
卷积 层,  max-池化 层, 和  完全地-connected 层,
yielding  查询 表示 的 向量 Q. 类似地,  向量 的
文档 d 是 passed 通过  卷积 层,  max-池化 层,
和  完全地-connected 层, yielding  文档 表示 的
矩阵 D. 下一个, 元素-wise 乘积 是 计算了 之间 D 和 
扩展了 Q.  resulting 矩阵 是 passed 通过 完全地 connected

matchingscoresumbinarymatrixforcapturingexactmatchingsignalsdocumentmatching signalsconvolution+完全地-connectedlocalmatchingscoreconvolution+完全地-connectedconvolution+完全地-connecteddistributedmatchingscorefully-connectedqueryelement-wiseproduct 88

搜索中的深度匹配模型

层, 和  随机失活 层  产生  单一 匹配分数.

 二 网络 在 Duet 是 联合地 训练了 作为  单一 神经
网络. 给定  训练 例子 该 由组成 的  查询 q, 
相关 文档 d+, 和  集合 的 non-相关 文档 D =
{d−
k },  目标 的 学习 是 deﬁned 作为  conditional
概率 的 文档 d+ 给定 查询 q:

1 , · · · , d−

P (d+|q) =

exp(f (q, d+))
d0∈D exp f (q, d0)

P

.

随机梯度下降 是 利用了  maximize  log 似然
log P (d+|q).

4.2.5 实验的 Results

我们 呈现  实验的 results 通过 匹配函数学习 方法, 报告了 在 Wan 等人 (2016b). 在  实验,  基准
的 Yahoo! 答案 是 利用了, 和 P@1 和 平均倒数排名
(MRR) 是 采用 作为 评估 测量.  results 在 表 4.4
表明 该 两者  方法 的 表示学习 和 
方法 的 匹配函数学习 可以 outperform  基线 的
BM25.  匹配函数学习 方法, 在 通用, 执行
更好 比  表示学习 方法. 表 4.5 呈现
一些 实验的 results 的  匹配 方法 在 ad hoc 检索.
 results 是 基于 在  实验 进行了 在 (Dai 等人,
2018; Mitra 等人, 2017). 我们 也 表示  实验的 results 的
BERT 报告了 在 Nogueira 和 Cho (2019), 用于  任务 的 段落
排序. 在  实验,  基准 的 MS MARCO 是 利用了,
和 MRR@10 是 采用 作为  评估 度量.  results 在 表 4.6 表明 该 ﬁne-调整 BERTLARGE signiﬁcantly outperforms
 最先进的 段落 排序 模型.

4.3 讨论与扩展阅读

在 该 节, 我们 讨论  特征 的  二 匹配
方法 和 给出 更多 references 用于 扩展阅读.

4.3. 讨论与扩展阅读

89

表 4.4: Performances 的 一些 表示学习 方法 和 匹配
函数 学习 方法 在 Yahoo! 答案.

表示
学习

.

BM25 (基线)
ARC-I
CNTN
LSTM-RNN

匹配函数 ARC-II
学习

MatchPyramid
匹配-SRNN

P@1 MRR

0.579
0.581
0.626
0.690
0.591
0.764
0.790

0.726
0.756
0.781
0.822
0.765
0.867
0.882

表 4.5: Performances 的 匹配函数学习 方法 在 ad hoc 检索,
基于 在 Bing 搜索 Log 和 Sogou Log.
.

Bing 搜索 Log

Sogou Log

NDCG@1 NDCG@10 NDCG@1 NDCG@10

DSSM
Duet
DRMM
MatchPyramid
K-NRM

0.258
0.322
0.243
-
-

0.482
0.530
0.452
-
-

-
-
0.137
0.218
0.264

-
-
0.315
0.379
0.428

表 4.6: Performances 的  ﬁne-调整 BERTLARGE 和 其他 方法 在 MS
MARCO.

.
MRR@10 (Dev) MRR@10 (Eval)

BM25
K-NRM (Xiong 等人, 2017)
Conv-KNRM (Dai 等人, 2018)
BERTLARGE

0.167
0.218
0.290
0.365

0.165
0.198
0.271
0.358

90

搜索中的深度匹配模型

4.3.1 讨论

两者  方法 的 表示学习 和  方法 的
匹配函数学习 具有 是 密集地 研究了.  二
方法 具有 两者 advantages 和 limitations 和 具有 强
connections  传统 匹配 和 排序 模型 在 IR.

表示学习 给出  ﬁnal 匹配分数 基于 在 
语义 表示 的 查询 和 文档, 该 是 分别地
学会了 从  raw 表示 的 查询 和 文档. 
语义 表示 的 查询 和 文档 是 嵌入 vectors
(真实-valued vectors), 该 means 该 我们 表示  查询 和
 文档 在  共同 语义 空间. 该 方法 是 自然
和 具有  强 连接 用  传统 潜在空间 模型.
 方法 可以 eﬀectively 解决  术语 失配 问题 如果 
语义 的 查询 和 文档 是 表示了 very 好. 然而,
那里 也 exist limitations 在  方法.  查询 和 文档
是 表示了 独立地 在之前  ﬁnal 步骤 的 匹配分数
计算.  底层 假设 是 该 那里 exist 通用
表示 用于 查询 和 文档, 和  表示
可以 是 比较了 用于 确定 的 相关性. 然而, 查询 和
文档 通常 具有 语义 在 多个 levels (e.g., 局部 和
全局 levels). 它 是 更好, 因此, 如果 查询 和  文档 可以
是 比较了 在 diﬀerent levels. 在 其他 词,  表示 的
查询 和 文档 和  interactions 的 查询 和 文档
在 多个 levels 是 更好  是 建模.

传统 潜在空间 匹配 模型 用于 搜索 (e.g., PLS,
RMLS) 和 匹配 方法 使用中 主题 模型 (e.g., LSI, PLSA,
LDA) (Li 和 Xu, 2014) 也 学习  语义 表示 的
查询 和 文档. 从 该 视角,  方法 的 表示学习 具有 相似之处 用  传统 方法. 然而,
 深度学习 模型 具有 advantages, 因为 (1) 它们 采用 深度
神经 网络  map  查询 和 文档 到  语义
空间 和 因此 可以 获得 richer 表示; 和 (2)  映射
函数 和 embeddings 的 词 在  查询 和 文档 可以
是 联合地 学会了 在  端--端 fashion.

匹配函数学习, 在  其他 hand, 生成  ﬁnal

4.3. 讨论与扩展阅读

91

匹配分数 在  基础 的 两者 表示 和 interactions
的 查询 和 文档. 由于  基本 匹配 signals 是 建模
用 两者 高-水平 表示 (e.g., 语义 表示) 和
低-水平 表示 (e.g., 术语-水平 表示),  方法
具有  能力  进行 更多 准确 匹配.

传统 IR 模型 (e.g., VSM, BM25, 和 LM4IR) 也 比较
查询 词 和 文档 词 和 aggregate  匹配 signals.
从 该 视角,  方法 的 匹配函数学习 具有
相似之处 用  传统 IR 方法.  深度学习 模型
是 优于   传统 IR 模型, 然而, 因为 (1) 它们 可以
捕获 匹配 signals 不 仅 在  局部 水平 (e.g., 术语 水平) 但
也 在  全局 水平 (e.g., 语义层面); (2) 它们 可以 自然地 keep
和 取 到 consideration  positions 的 匹配; (3) 它 是 可能
 进行 端--端 学习 和 实现 更好 性能; 和 (4)
它们 可以 更多 easily 利用 weak-监督 数据 (e.g., clickthrough
logs).

表示学习 和 匹配函数学习 是 不
mutually exclusive. 匹配 模型 具有 也 是 开发了 该 可以
取  advantages 的 两者 方法. 一些 方法 直接地 组合
 分数 从  表示学习 和  匹配函数
学习, 作为 在 (Mitra 等人, 2017). 其他 方法 利用  注意力
机制  或者 构建  表示 的 查询 和
文档 和 使 interactions 之间  表示 (Yang
等人, 2019).

4.3.2 扩展阅读

语义匹配 在 搜索 是  very 主动 研究 主题. 这里 我们
列表 其他 相关 工作 在 文本 匹配 和 交叉-模态 匹配 和
 基准 datasets 和 open-来源 software packages.

Papers

 大 数量 的 模型 是 提出 用于 进行 匹配 在
搜索. 一 研究 方向 是  学习 更多 sophisticated 表示. 用于  表示学习 方法, Yin 和 Schütze
(2015) 提出  CNN-基于 多-GranCNN 模型, 该 学习

92

搜索中的深度匹配模型

查询 表示 和 文档 表示 在 多个 levels, 包括中 词, 短语, 和  整个 文本. Wan 等人 (2016)
呈现 MV-LSTM,  LSTM-基于 模型  实现 多个 positional 句子 表示, 用于 捕获中  局部 信息 作为
好 作为  全局 信息 在  句子. Nie 等人 (2018) 指出
out 该 diﬀerent levels 的 匹配, 从 低-水平 术语 匹配 
高-水平 语义匹配, 是 需要了 due   本质 的 自然
语言.  多-水平 抽象 卷积 模型 (MACM) 是
提出 用于 生成中 多-levels 的 表示 和 aggregating
 多-levels 的 匹配 分数. Huang 等人 (2017) 也 呈现 
方法 该 组合 CNN 和 LSTM  利用 从 字符-水平
 句子-水平 特征 用于 执行中 匹配. Jiang 等人 (2019)
呈现 MASH RNN 用于 匹配 的 长 文档. 在  方法,
bidirectional GRU 和  注意力机制 是 采用 作为 
encoders 用于 构建  文档 表示 在  句子
水平, 段落 水平, 和 文档 水平. Liu 等人 (2019) 提出
 encode news articles 用  概念 交互 图 和 进行
匹配 基于 在  句子 该 enclose  相同 概念 vertices.
用于  匹配函数学习 方法,  注意力机制 是 密集地 使用了. 用于 例子, 注意力 基于 卷积
神经网络 (ABCNN) (Yin 等人, 2016) 集成  注意力
机制 到 CNNs 用于 通用 句子 对 建模. 在  模型,
 表示 的 每个 句子 可以 取 其 counterpart 到 consideration.  Bilateral 多-视角 匹配 (BiMPM) (Wang
等人, 2017c) 模型 匹配  二 encoded 句子 在 二 directions.
在 每个 匹配 方向, 每个 位置 的 一 句子 是 匹配了
针对 (attended ) 所有 positions 的  其他 句子 从 多个 perspectives. 多-Cast 注意力 网络 (MCAN) (Tay 等人,
2018d) 执行  序列 的 soft 注意力 operations 用于 问题-答案
匹配. 一 advantages 的 MCAN 是 该 它 允许  任意 数量 的 注意力 mechanisms  是 casted, 和 允许 多个 注意力
类型 (e.g., co-注意力, intra-注意力) 和 注意力 variants (e.g.,
对齐-池化, max-池化, 均值-池化)  是 executed 同时地. 也, Tay 等人 (2018c) 论证 该  co-注意力 模型
在 asymmetrical 匹配 任务 需要 diﬀerent treatments  
attentions 用于 symmetrical 任务. 它们 提出 Hermitian Co-注意力

4.3. 讨论与扩展阅读

93

Recurrent 网络 (HCRN) 在 该  注意力机制 是 基于
在  复杂-valued 内积 (Hermitian products). Tan 等人
(2018) 提出  Multiway 注意力 网络 (MwAN) 在 该
多个 注意力 函数 是 采用了  匹配 句子 pairs 之下
 匹配-聚合 框架. Chen 等人 (2018b) 提出 
多-channel 信息 Crossing (MIX) 模型  比较  查询
和 文档 在 各种 granularities, 形成中  序列 的 匹配
矩阵. 注意力 层 是 则 imposed 用于 捕获中  interactions 和 产生  ﬁnal 匹配分数. 注意力-基于 神经
匹配模型 (aNMM) (Yang 等人, 2016) 是 另一个 注意力-基于
神经 匹配模型. 给定  匹配矩阵, 用于 每个 查询
词,  值-共享 weighting 方案 而非  位置-共享
weighting 方案 是 使用了 用于 计算中  匹配 signals 用于 
词.  signals 的 diﬀerent 查询 词 是 aggregated 用  注意力 网络. Nogueira 等人 (2019) 提出  三-阶段 排序
架构 用于 搜索.  首先 阶段 是 implemented 用 BM25,
和  秒 阶段 和  第三 阶段 是 分别地 implemented
用 逐点 和 成对 BERT 基于 在 学习  排序. Yang
等人 (2019b) 和 Qiao 等人 (2019) 也 应用  BERT 模型 
临时检索 和 段落检索. Reimers 和 Gurevych (2019)
提出 句子-BERT 用于 减少中  计算 overhead 用于
文本 匹配. 用于  任务 的 自然 语言 推理, Chen 等人
(2017b) 提出  序列 推理 模型 基于 在 chain LSTMs,
称为 增强 序列 推理 模型 (ESIM). ESIM 显式地
考虑 递归 架构 在 两者 局部 推理 建模 和
推理 组合. Gong 等人 (2018) 提出  交互 推理 网络 (IIN) 该 hierarchically extracts 语义 特征
从 交互 空间 和 执行 高-水平 理解 的 
句子 对. 它 是 显示 该  交互 张量 (注意力 权重)
可以 捕获  语义 信息  solve  自然 语言
推理 任务.

用于 无监督 匹配 模型, Van Gysel 等人 (2017) 提出
 adapt NVSM   实体 空间, 该 可以 是 使用了 用于 乘积
搜索 (Van Gysel 等人, 2018; Van Gysel 等人, 2016) 和 expert
搜索 (Van Gysel 等人, 2016b). Zamani 和 Croft (2017) 也 呈现
 方法  学习  词 和 查询 表示 在 

94

搜索中的深度匹配模型

无监督 方式. 弱 监督 模型 是 也 提出
在  IR 社区 用于  任务 的 文本 表示, 匹配,
和 排序. 作为 用于 训练 模型 使用中 弱监督, Zamani
和 Croft (2016) 提出  框架 在 该  查询 可以 是
表示了 基于 在  个体 词嵌入 vectors. 
参数 是 估计 使用中 pseudo-相关 文档 作为 训练
signals. Zamani 和 Croft (2017) 提出  训练  神经 排序
模型 使用中 弱监督. 仍然,  标签 是 获得了 自动地
从 pseudo-相关性 反馈 和 无  需求 的 使用中 人类
标注 或 external 资源. Dehghani 等人 (2017) 提出 
使用  输出 的 BM25 作为  弱监督 信号  训练 神经
排序 模型. Haddad 和 Ghosh (2019) 建议  利用 多个
无监督 rankers  生成 soft 训练 标签 和 则 学习
神经 排序 模型 基于 在  生成了 数据. Zamani 等人
(2018) 提出  训练  查询 性能 预测 模型 用
多个 弱监督 signals. Zamani 和 Croft (2018b) 提供
 理论的 justiﬁcation 用于 弱监督 用于 信息检索.
通常,  匹配 模型 假设 该 查询 和 文档 是
homogeneous (e.g., 短 文本4), 和 symmetric 匹配 函数
是 利用了. Pang 等人 (2016) 研究 匹配 之间 短 查询
和 长 文档, 在  基础 的  前述 MatchPyramid
模型 (Pang 等人, 2016b).  results 显示 该 查询 和 文档 在 网页搜索 是 异质 在 本质: 查询 是 短 同时
文档 是 长, 和 因此 学习  asymmetric 匹配模型
是  更好 选择. 卷积 核-基于 神经 排序模型
(Conv-KNRM) (Dai 等人, 2018) 扩展  K-NRM 模型, 使
使用 的 CNNs  表示 n-grams 的 各种 lengths, 和 执行 soft
匹配 的  n-grams 在  uniﬁed 嵌入 空间. Researchers
也 观察 该  长 文档 由组成 的 多个 段落 和
匹配 用  查询 可以 是 determined 通过 一些 的 它们. 因此,
 文档 可以 是 拆分 到 多个 段落 和 匹配了 individually 用于 捕获中 ﬁne-grained 匹配 signals. 在 DeepRank (Pang
等人, 2017),  文档 是 拆分 到 术语-centric contexts, 每个
相应   查询 术语.  局部 相关性 之间 每个 (查询,

4用于 文档, 仅  titles, anchors, 或 clicked 查询 是 使用了.

4.3. 讨论与扩展阅读

95

术语-centric 上下文) 对 是 计算了 用  MatchPyramid 模型.
 局部 相关性 分数 是 则 aggregated 作为  查询-文档
匹配分数.

类似地, 位置-Aware 卷积-Recurrent 相关性 匹配 (PACRR) (Hui 等人, 2017) splits  文档 用  sliding
window.  聚焦了 区域 可以 是  首先-k 词 在  文档
或  大多数 相似 上下文 positions 在  文档 (k-window). 
上下文-aware PACRR (Co-PACRR) (Hui 等人, 2018) 扩展 PACRR
通过 incorporating  组件 该 可以 模型  上下文 信息 的  匹配 signals. Fan 等人 (2018) 提出  层次
神经 匹配模型 (HiNT) 模型 在 该  文档 是 也
拆分 到 段落.  局部 相关性 signals 是 计算了 之间
 查询 和  段落 的  文档.  局部 signals 是
accumulated 到 diﬀerent granularities 和 ﬁnally 结合 到 
ﬁnal 匹配分数. 商业 网页搜索 engines 需求  考虑
更多 比 仅 一 文档 ﬁeld. Incorporating diﬀerent sources 的
文档 描述 (e.g., title, URL, body, anchor, etc.) 是 有用 
确定  相关性 的  文档   查询 (Robertson 等人,
2004).  解决  问题 的 leveraging 多个 ﬁelds, Zamani 等人
(2018c) 提出 NRM-F 该 介绍  masking 方法  处理
missing 信息 从 一 ﬁeld, 和  ﬁeld-水平 随机失活 方法 
避免 relying 也 许多 在 一 ﬁeld. 在 层次 注意力 检索
模型 (HAR) (Zhu 等人, 2019), 词-水平 交叉-注意力 是 进行了
 识别  词 该 大多数 相关 用于  查询, 和 层次
注意力 是 进行了 在  句子 和 文档 levels.

作为 用于 交叉-模态 查询-文档匹配, CCA (Hardoon 等人,
2004) 和 语义 相关 匹配 (Rasiwasia 等人, 2010) 是 传统 模型. 两者 的  模型 aim  学习 线性 transformations
 项目 二 对象 在 diﬀerent 模态 到  共同空间 这样的
该 其 相关 是 maximized.  扩展  线性 transformations
到 non-线性 transformations, 核 典型相关分析
(KCCA) (Hardoon 和 Shawe-Taylor, 2003) 是 提出, 该 发现s
maximally correlated projections 在  reproducing 核 Hilbert 空间
使用中  核 函数.  方法 是 介绍 在 (Karpathy 等人,
2014)  embed fragments 的 images (对象 在  图像) 和 fragments
的 句子 到  共同空间, 和 计算 其 相似之处 作为

96

搜索中的深度匹配模型

dot products.  ﬁnal 图像-句子 匹配分数 是 deﬁned 作为 
average thresholded 分数 的 其 成对 fragment 匹配 分数.
Multimodal 卷积 神经 网络 (m-CNNs) (Ma 等人, 2015)
adopts CNN  计算  多-模态 匹配 分数 在  wordlevel, 短语-水平, 和 句子-水平.  更好 均匀地 表示
多-媒体 对象 用 embeddings,  多样性 的 多-模态 深度
神经 网络 是 开发了, 包括中  模型 提出 在 Wang
等人 (2016), Eisenschtat 和 Wolf (2017), Liu 等人 (2017), Wang 等人
(2018b), Huang 等人 (2018), Balaneshin-kordan 和 Kotov (2018), 和
Guo 等人 (2018).

基准 Datasets

 数量 的 publicly 可用 基准 datasets 是 使用了 用于 训练 和 测试 语义匹配 模型. 用于 例子,  传统 信息检索 datasets 这样的 作为  TREC collections5
(e.g., 鲁棒, ClueWeb, 和 Gov2 etc.),  NTCIR collections6, 和
Sogou-QCL (Zheng 等人, 2018b) 是 适合 用于 实验 在
查询-文档匹配. 问答 (QA) (和 communitybased QA) collections 这样的 作为 TREC QA7, WikiQA (Yang 等人,
2015), WikiPassageQA (Cohen 等人, 2018), Quora’s 2017 问题
数据集8, Yahoo! 答案 集合 (Surdeanu 等人, 2011), 和 MS
MARCO (Nguyen 等人, 2016) 是 也 使用了 用于 研究 在 深度匹配 模型 作为 好. 其他 自然语言处理 datasets 这样的 作为
MSRP (Dolan 和 Brockett, 2005) 和 SNLI (Bowman 等人, 2015)
是 也 exploited.

Open 来源 Packages

 数量 的 open-来源 packages 用于 匹配 是 可用 在 
web. MatchZoo9 是  open-来源 项目 dedicated  研究 在 深度

5https://trec.nist.gov/数据.html
6http://研究.nii.ac.jp/ntcir/数据/数据-en.html
7https://trec.nist.gov/数据/qamain.html
8https://数据.quora.com/第一-Quora-数据集-Release-问题-Pairs
9https://github.com/NTMC-社区/MatchZoo

4.3. 讨论与扩展阅读

97

文本 匹配 (Guo 等人, 2019). TensorFlow 排序10 是  subproject
的 TensorFlow 该 旨在 在 solving 大-规模 搜索 排序 问题
在  深度学习 框架 (Pasumarthi 等人, 2019). Anserini11
是  open-来源 信息检索 toolkit 构建了 在 Lucene 该
旨在 在 bridging  鸿沟 之间 学术 研究 和 真实-世界
应用 (Yang 等人, 2018).

10https://github.com/张量ﬂow/排序
11https://github.com/castorini/Anserini

5

推荐中的深度匹配模型

在 该 章, 我们 介绍 代表性 深度匹配 方法
在 推荐. 作为 在 章 4, 我们 分类  方法 到
二 组: 1) 方法 的 表示学习, 和 2) 方法 的
匹配函数学习. 在  首先 类别, 神经 网络 是
采用了  创建 表示 的 用户 和 物品  使  比较 之间 它们 和 生成  ﬁnal 匹配分数. 在  秒
类别, 神经 网络 是 利用了  进行 interactions 之间
用户 和 物品 (和 possibly contexts)  生成 匹配 signals 和
aggregate 它们   ﬁnal 匹配分数. 表 5.1 显示  分类
的  代表性 推荐中的深度匹配模型.

5.1 基于表示学习的匹配

匹配 模型 基于 在 表示学习 adopt  通用
框架 的 潜在空间 模型 作为 描述 在 图 4.1. 在 短,
给定 用户 u 在  用户空间 u ∈ U 和 物品 i 在  物品空间 i ∈ I,
函数 φu : U 7→ H 和 φi : I 7→ H stand 用于 mappings 从  用户
空间 U 和 从  物品空间 I   新 空间 H, 分别地. 

98

5.1. 基于表示学习的匹配

99

n
o
i
t

d
n
e
m
m
o
c
e
R
n
i

s
l
e
d
o
M
g
n
i
h
c
t

M
p
e
e
D

:
1
.
5

e
l
b

T

)
8
1
0
2

,
.
l


t
e

g
n
i
Y
(

e
g

S
n
i
P

,
)
9
1
0
2

,
.
l


t
e

i
e

W

(
N
C
G
M
M

,
)

7
1
0
2

,
.
l


t
e

n
e
h
C
(

F
C


,
)
6
1
0
2

,
.
l


t
e

i
e
L
(

L
D
C

,
)

6
1
0
2

,
y
e
l
u

c
M
d
n


e
H
(
R
P
B
V

)
9
1
0
2

,
.
l


t
e

i

L
(
P
R

C

,
)

8
1
0
2

,
.
l


t
e

n
e
h
C
(
E
R
R

N

,
)
7
1
0
2

,
.
l


t
e

g
n
e
h
Z
(
N
N
o
C
p
e
e
D

,
)

8
1
0
2

,
.
l


t
e

g
n

W

(

t
e
N
e
l
p
p
i
R

,
)

9
1
0
2

,
.
l


)
c
9
1
0
2

,
.
l


t
e

t
e

g
n

W

(
T

G
K

g
n

W

(
N
R
P
K

,
)
8
1
0
2

,
.
l


t
e

g
n
i
Y
(

e
g

S
n
i
P

,
)
b
9
1
0
2

,
.
l


t
e

g
n

W

(

F
C
G
N

)
0
2
0
2

,
.
l


t
e

e
H
(
N
C
G
t
h
g
i
L

s
w
e
i
v
e
R
r
e
s
U

t
n
e
t
n
o
C

i
d
e
m

i
t
l
u
M

h
p

r
G
m
e
t
i
-
r
e
s
U

h
p

r
G
e
g
d
e
l
w
o
n
K

)
9
1
0
2

,
.
l


t
e

n

F
(

c
e
R
h
p

r
G

,
)
b
9
1
0
2

,
.
l


t
e

u
W

(

t
e
N
ﬀ
D

i

k
r
o
w
t
e
N

l

i
c
o
S

,
)
b
8
1
0
2

,
.
l


t
e

e
H
(

F
C
N
v
n
o
C

,
)
c
7
1
0
2

,
.
l


t
e

e
H
(

F
C
N

)
7
1
0
2

,
.
l


t
e

i

B
(

F
C
N
N

,
)
9
1
0
2

,
.
l


t
e

e
u
X
(

F
C
I
p
e
e
D

i

g
n
n
r

e
L

y
t
i
r

l
i

m
S

i

)

8
1
0
2

,
.
l


t
e

y

T
(

L
M
R
L

,
)

7
1
0
2

,
.
l


t
e

e
H
(

c
e
R
s
n

r
T

,
)
7
1
0
2

,
.
l


t
e

h
e
i
s
H
(

L
M
C

i

g
n
n
r

e
L

c
i
r
t
e

M

,
)
9
1
0
2

,
.
l


t
e

o

T
(

M
F

o
H

,
)
7
1
0
2

,
.
l


t
e

o

i
X
(

M
F


,
)
7
1
0
2

,

u
h
C
d
n


e
H
(

M
F
N

)
8
1
0
2

,
y
e
l
u

c
M
d
n



h
c
i
r
s

P
(

M
F
s
n

r
T

,
)
8
1
0
2

,
.
l


t
e

n

i
L
(
N
I
C

,
)
6
1
0
2

,
.
l


t
e

g
n
e
h
C
(

p
e
e
D
&
e
d
i
W

,
)
6
1
0
2

,
.
l


t
e

n
o
t
g
n
i
v
o
C
(
N
N
D
e
b
u
T
u
o
Y

)
8
1
0
2

,
.
l


t
e

u
o
h
Z
(
N
D

I

,
)
6
1
0
2

,
.
l


t
e

n

h
S
(

g
n
i
s
s
o
r
C
p
e
e
D

g
n

i
l
e
d
o
M
n
o
i
t
c

r
e
t
n
I

t
i
c
i
l

p
m

I

g
n

i
l
e
d
o
M
n
o
i
t

r
e
t
n
I

t
i
c
i
l

p
x
E

,
)
7
1
0
2

,
.
l


t
e

o
u
G

(

M
F
p
e
e
D

,
)
6
1
0
2

,
.
l


t
e

g
n
e
h
C
(

p
e
e
D
&
e
d
i
W

d
n


t
i
c
i
l

p
x
E

f
o

n
o
i
t

n
b
m
o
C

i

)
8
1
0
2

,
.
l


t
e

n

i
L
(

M
F
p
e
e
D
x

g
n

i
l
e
d
o
M
n
o
i
t
c

r
e
t
n
I

t
i
c
i
l

p
m

I

,
)
9
1
0
2

,
.
l


t
e

n

u
Y
(

t
e
N
t
I
t
x
e
N

,
)
8
1
0
2

,
g
n

W
d
n


g
n

T
(

r
e
s

C

)
8
1
0
2

,
.
l


t
e

l
e
t
u
e
B
(

s
s
o
r
C

)
0
2
0
2

,
.
l


t
e

n

u
Y
(

c
e
R
G

d
e
s

b
-
N
N
C

,
)
6
1
0
2

,
.
l


t
e

i
s

d
i
H
(

c
e
R
4
U
R
G

t
n
e
t

L

,
)
7
1
0
2

,
.
l


t
e

u
W

(
N
R
R

,
)
7
1
0
2

,
.
l


t
e

i

L
(

M
R

N

d
e
s

b
-
N
N
R

)
9
1
0
2

,
.
l


t
e

n
u
S
(

c
e
R
4
t
r
e
B

,
)
8
1
0
2

,
y
e
l
u

c
M
d
n


g
n

K
(

c
e
R
S

S

d
e
s

b
-
n
o
i
t
n
e
t
t


,
)
5
1
0
2

,
.
l


t
e

i

L
(

F
C
p
e
e
D

,
)
b
7
1
0
2

,
.
l


t
e

g
n

W

(
R
C
S
N

s
e
t
u
b

i
r
t
t


l

c
i
r
o
g
e
t

C

l

i
t
n
e
u
q
e
S

s
n
o
i
t
c

r
e
t
n
I

l

d
o
m

-
i
t
l
u
M

t
n
e
t
n
o
C


t

D
h
p

r
G

y

w
-
o
w
T

i

g
n
h
c
t

M

y

w

-
i
t
l
u
M

i

g
n
h
c
t

M

)
8
1
0
2

,
.
l


t
e

u
o
h
Z
(
N
D

I

,
)

7
1
0
2

,
.
l


t
e

n
e
h
C
(

F
C


,
)

8
1
0
2

,
.
l


t
e

e
H
(

S
I

N

d
e
s

b
-
n
o
i
t
n
e
t
t


,
)
6
1
0
2

,
.
l


t
e

n
o
t
g
n
i
v
o
C
(
N
N
D
e
b
u
T
u
o
Y

,
)
7
1
0
2

,
.
l


t
e

e
u
X
(

F
M
p
e
e
D

,
)
b
6
1
0
2

,
.
l


t
e

u
W

(
E

D
C

,
)
5
1
0
2

,
.
l


t
e

n
i

h
d
e
S
(

c
e
R
o
t
u


)
5
1
0
2

,
.
l


t
e

y
k
h

k
l
E
(
N
N
D
V
M

-

)
8
1
0
2

,
.
l


t
e

g
n

i
L
(
E

V
-
t
l
u
M

d
e
s

b
-
P
L
M

d
e
s

b
-
r
e
d
o
c
n
e
o
t
u


d
e
r
e
d
r
o
n
U

s
n
o
i
t
c

r
e
t
n
I

s
l
e
d
o
M

n
o
i
t

z
i
r
o
g
e
t

C
e
u
q
i
n
h
c
e
T

n
o
i
t

t
n
e
s
e
r
p
e
R
t
u
p
n
I

n
o
i
t

t
n
e
s
e
r
p
e
R
n
o

d
e
s

b

s
d
o
h
t
e

M

i

g
n
n
r

e
L

d
e
s

b

s
d
o
h
t
e

M

i

g
n
h
c
t

M
n
o

i

g
n
n
r

e
L

n
o
i
t
c
n
u
F

100

推荐中的深度匹配模型

匹配模型 之间 u 和 i 是 deﬁned 作为

f (u, i) = F (φu(u), φi(i)),

(5.1)

其中 F 是  相似度 函数 超过 H, 这样的 作为 内积 或
余弦相似度. Diﬀerent 神经 网络 可以 是 使用了  realize 
表示 函数 φu 和 φi, depending 在  形式 的 输入
数据 和  数据 properties 的 兴趣. 我们 进一步 分类 
方法 到 四 类型 基于 在  形式 的 输入 数据: 1) unordered
interactions, 2) 序列 interactions, 3) 多-模态 content, 和 4)
linked 图.

 remainder 的 该 节 是 组织  呈现 每个 类型
的  方法 在 一 小节. 在 小节 5.1.1, 我们 描述
方法 该 表示  用户 用 他的/她的 unordered interactions 用
 系统, 这样的 作为 深度 矩阵分解 和 auto-编码器 基于
方法. 在 小节 5.1.2, 我们 解释 方法 该 表示  用户
用  序列 的 他的/她的 interactions (ordered interactions), 这样的
作为 RNN-基于 和 CNN-基于 序列推荐 方法.
在 小节 5.1.3, 我们 呈现 方法 该 incorporate 多-模态
content 到  学习 的 表示, 这样的 作为 用户/物品 attributes,
文本, 和 images. 在 小节 5.1.4, 我们 介绍 最近 开发了
方法 该 执行  学习 的 表示 在 图 数据,
这样的 作为 用户-物品 图 和 知识图谱.

5.1.1 表示学习 从 Unordered Interactions

 传统 矩阵分解 模型 利用  一-hot ID 向量
 表示  用户 ( 物品), 和 执行 一-层 线性 投影
 获得  用户 (物品) 表示. 作为  一-hot 向量 仅 包含
 ID 信息, 它 是 不 有意义的  执行 多个 层 的
non-线性 变换 在  向量. 给定 该  abundance 的
用户-物品交互 数据 是 可用 在  推荐 系统,
 自然 想法 是  表示  用户 用 他的/她的 交互 历史,
该 encodes richer 信息. 如果 我们 ignore  顺序 的 用户-物品
interactions,  交互 历史 可以 是 考虑了 作为  unordered
集合 的 interactions. 每个 交互 可以 是 表示了 作为  多-hot
向量 表示中  interacted 物品 通过  用户, 其中 每个 维度

5.1. 基于表示学习的匹配

101

图 5.1: 模型 架构 的 DeepMF.

corresponds   物品. 我们 下一个 综述 三 类型 的 方法 该
学习  用户 表示 从 unordered interactions: MLP-基于,
Auto-编码器-基于, 和 注意力-基于 方法.

MLP-基于 方法

深度 矩阵分解 (DeepMF) (Xue 等人, 2017) adopts 
架构 的 DSSM (Huang 等人, 2013). 它 具有  二-tower 结构,
其中 一 tower 是 用于 学习 用户 表示, 和  其他 是 用于
学习 物品 表示. 在 每个 tower,  MLP 是 采用了 
学习  表示 从  多-hot 向量.  期望 是 该
 多-层 非线性 transformations 在  交互 历史 可以
学习 更好 表示  桥接  语义鸿沟 之间 
用户 和 物品. 图 5.1 说明  架构 的  DeepMF
模型.

Let  用户-物品交互 矩阵 是 Y ∈ RM ×N 其中 M 和 N
表示  数量 的 用户 和 物品, 分别地; 用于 显式反馈,
每个 条目 yui 是  rating 分数, 和  分数 的 0 means 该 用户 u
具有 不 rated 在 物品 i 在之前; 用于 隐式反馈, 每个 条目 是 
二进制 值, 和  值 的 1 和 0 表示 是否 或 不 用户 u
具有 interacted 用 物品 i 在之前. Let yu∗ ∈ RN 表示  u-th row
的 Y, i.e.,  多-hot 历史 向量 的 用户 u, 和 y∗i ∈ RM 表示

pu…qiInteractionmatrixYy*iyu*多-layernonlinearprojection…LatentrepresentationmatchingscoreLayer1层1LayerNLayerN102

推荐中的深度匹配模型

 i-th column 的 Y, i.e.,  多-hot 历史 向量 的 物品 i. 则,
我们 可以 express  匹配函数 的 DeepMF 作为:

pu = MLP1(yu∗), qi = MLP2(y∗i),

f (u, i) = cosine(pu, qi) =

pT
u qi
||pu||2||qi||2 .

(5.2)

作为  spaces 的 用户 和 物品 是 diﬀerent, DeepMF 使用 二 MLPs
用 diﬀerent 参数  表示 用户 和 物品. 注意 该 due 
 稀疏 本质 的 yu∗ 和 y∗i,  整体 复杂度 的  模型 是
acceptable 如果 我们 omit  零 entries 在 Y 在 实现. 它 是 也
worth mentioning 该 它 是 不 compulsory  使用 二 towers — 一
可以 使用 MLP1 仅  获得 pu 和 使用 简单 嵌入 lookup 用于
qi. 这样的  simpliﬁcation 是 本质上 等价   auto-编码器
基于 架构, 该 是 介绍 下一个.

Auto-编码器 基于 方法

Auto-编码器 是 另一个 选择  构建  推荐 模型 从
 交互 历史. Auto-编码器 transforms  输入 数据 到 
隐藏 表示, 这样的 该 从  good 隐藏 表示, 一
是 几乎 able  recover  输入 数据. 在 物品-基于 AutoRec (Sedhain
等人, 2015),  输入 是  用户 历史 向量 yu∗ ∈ RN , 和 
重建 的  输入 是:

ˆyu∗ = σ2(W · σ1(Vyu∗ + b1) + b2),
其中 σ2 和 σ1 是 activation 函数, V ∈ Rd×N 和 W ∈ RN ×d
是 权重 矩阵, b1 ∈ Rd 和 b1 ∈ RN 是 偏置 vectors. 
重建 向量 ˆyu∗ 是  N -维度 向量 该 stores 
预测了 匹配 分数 的 所有 物品 用于 用户 u.  学习 参数
θ = {V, W, b1, b2}, AutoRec minimizes  总计 损失 超过 所有 输入
(用户) 用 L2 正则化:

(5.3)

L =

M
X

u=1

||yu∗ − ˆyu∗||2 + λ||θ||2.

给定 该 推荐 是 本质上  匹配 plus 排序 任务,
其他 损失 函数 如 交叉-熵, 合页损失 和 成对 损失 可以
也 是 采用了 这里, 作为 证明了 通过 Wu 等人 (2016b).

5.1. 基于表示学习的匹配

103

在 fact, 我们 可以 视角  AutoRec 模型 作为 一 使用中 MLP 在
交互 历史  学习  用户 表示, 和 使用中 嵌入
lookup  获得  物品 表示.  是 更多 speciﬁc, 我们 可以
reformulate 方程 (5.3)  get  元素-wise 匹配函数:

f (u, i) = ˆyu∗,i = σ2(wi∗
|{z}
qi

· σ1(Vyu∗ + b1)
}
{z
|
=⇒ pu=MLP(yu∗)

+b2),

(5.4)

其中 wi∗ 表示  i-th row 的 W, 该 可以 是 seen 作为  ID
嵌入 的 物品 i, 和 用户 表示 pu 是 等价  
输出 的  一-层 MLP 用 yu∗ 作为 输入.  匹配分数 是
本质上  内积 的 用户 表示 pu 和 物品 ID
嵌入 qi, 该 falls 到  潜在空间 框架 deﬁned 在
方程 (5.1). 如果 多个 隐藏 层 是 使用了  构建  “深度”
auto-编码器, 我们 可以 interpret 它 作为 replacing 一-层 MLP 用 multilayer MLP  获得  用户 表示. 作为 这样的,  auto-编码器
架构 可以 是 seen 作为  simpliﬁed 变体 的 DeepMF.

一些 后来 variants 的 AutoRec 包括 协同 Denoising
Auto-编码器 (CDAE) (Wu 等人, 2016b), 该 扩展 AutoRec
通过 corrupting  输入 yu∗ 用 随机 noises  prevent  模型
从 学习  简单 identity 函数 和  discover  更多 鲁棒 表示. Liang 等人 (2018) 提出 扩展中 variational
auto-编码器 用于 推荐, solving  表示学习
问题 从  视角 的 生成 概率 建模.

注意力-基于 方法

一 观察 在  学习 的 用户 表示 是 该 历史 物品 可以 不 equally contribute   建模 的  用户’s
偏好. 用于 例子,  用户 可以 选择  trendy 物品 基于 在 其
高 popularity 而 比 他的/她的 own 兴趣. 虽然, 在 原则,
 MLP 学会了 从 交互 历史 可以 是 able  捕获 
复杂 relationships (c.f.  通用 approximation 定理 的
神经 网络 (Hornik, 1991)),  过程 是 也 隐式 和 那里 是
无 guarantee 用于 该.  solve  问题,  神经 Attentive 物品
相似度 (NAIS) 模型 (他 等人, 2018) 采用  神经 注意力
网络  显式地 学习  权重 的 每个 历史 物品. 图 5.2

104

推荐中的深度匹配模型

显示  架构 的  模型.

图 5.2: 模型 架构 的 NAIS.

在 短, NAIS 是  扩展 的 FISM 通过 使用中  learnable 权重
在 每个 interacted 物品 的  用户. Let Yu 是  集合 的 interacted 物品
的 用户 u, 和 每个 物品 i 是 associated 用 二 ID 嵌入 vectors
pi 和 qi  表示 其 角色 作为  目标 物品 和  历史 物品,
分别地.  匹配函数 在 NAIS 是 formulated 作为

f (u, i) = ( X

aijqj)T pi,

j∈Yu\{i}

aij =

exp(g(pi, qj))
j∈Yu\{i} exp(g(pi, qj))]β ,

[P

(5.5)

其中 aij 是  注意力 权重 该 controls  权重 的  历史
物品 j 在 估计 的  用户 u’s 匹配分数 在  目标 物品
i.  注意力 网络 g 是 典型地 implemented 作为  一-层
MLP 该 输出  标量 值 (e.g.,  MLP 取 concatenation
或 元素-wise 乘积 的 pi 和 qj 作为 输入).  输出 的 g 是
进一步 processed 通过  smoothed softmax 函数 其中 β 是 在 (0, 1)
 平滑  weighted sum 的 主动 用户 ( default 值 的 β 是
0.5). 通过 显式地 学习  权重 的 每个 interacted 物品 用 

attentionoutputattentioninputinnerproduct010101…000100…q2q4q6p3s32s34s36attentionnetworka323436∑matchingscore5.1. 基于表示学习的匹配

105

注意力 网络,  可解释性 的 表示 学会了 从
交互 历史 可以 也 是 改进了. 一 可以 进一步 增强
 non-线性 的 表示学习 通过 stacking  MLP 以上
 sum 池化, 这样的 作为 在  深度 神经网络架构 用于
YouTube 推荐 (Covington 等人, 2016).

它 是 worth highlighting 该  NAIS 注意力 是 aware 的  目标
物品 i 当 estimating  权重 的  历史 物品 j. 该 purposeful
设计 是  解决  局限性 的 静态 用户 表示 当
interacting 用 diﬀerent 物品. 用于 例子, 当  用户 考虑
是否  purchase  clothing 物品,  历史 behaviors 在 
fashion 类别 是 更多 reﬂective 的 他的/她的 aesthetic 偏好 比
 历史 behaviors 在  electronic 类别.  深度 兴趣
网络 (DIN) 模型 (Zhou 等人, 2018), 该 是 独立地
提出 通过  Alibaba 团队 在  相同 时间, adopts  相同 方式
的 动态 (目标 物品-aware) 用户 表示. 它 是 显示  是
有用  distill 有用 signals 从 用户 行为 历史 在 大-规模
e-commerce CTR 预测.

5.1.2 表示学习 从 序列 Interactions

用户-物品 interactions 是 自然地 associated 用 timestamps, 该
记录 当  交互 happens. 如果  顺序 的 用户-物品 interactions
是 考虑了,  交互 历史 成为  序列 的 物品 IDs.
建模 这样的  序列 可以 是 有用 用于 预测 的 用户 行为 在
 未来, 用于 例子, purchase 转移 patterns 从 一 物品 (e.g.,
phone)  另一个 物品 (e.g., phone 案例) exist 和 最近 purchases
是 更多 predictive 的 下一个 purchases. 下一个, 我们 呈现 二 类型
的 序列 顺序 基于 推荐 方法: RNN-基于 和
CNN-基于 方法.

RNN-基于 方法

作为 一 的  pioneering 工作 在 RNN 用于 基于会话的推荐, Hidasi 等人 (2016) 提出  GRU-基于 RNN 用于 summarizing
 序列 interactions (e.g.,  序列 的  clicked 物品) 在
 会话 和 使 推荐, 称为 GRU4Rec.  输入
  GRU4Rec 模型 是  序列 的 r clicked 物品 在  会话

106

推荐中的深度匹配模型

x = (x1, · · · , xr−1, xr), 其中 每个 物品 是 表示了 作为  一-hot
N -维 向量 和 N 是  数量 的 物品.  输出 是
 下一个 事件 (clicked 物品) 在  会话. 更多 speciﬁcally, 在 每个
位置 i 在  序列 x,  输入 是  陈述 的  会话, 该
可以 是  一-hot 表示 的  当前 物品, 或  weighted sum
的 表示 的 物品 因此 far, 作为 显示 在 图 5.3. 作为  core 的
 网络,  多-层 GRU 是 使用了  receive  embeddings 的 
输入 表示, 和  输出 的 每个 GRU 层 是  输入
  下一个 层. 最后, feedforward 层 是 added 之间 
最后 GRU 层 和  输出 层.  输出 是  N -维
向量, 每个 表示中  概率 的  相应 物品 是
clicked 在  下一个 事件 的  会话.

期间 训练, 成对 排序 损失 是 利用了  学习 
模型 参数. 二 类型 的 损失 函数 是 使用了.  BPR 损失
比较  分数 的  正 (preferred) 物品 用 那些 的 若干
sampled 负 物品. 因此,  BPR 损失 在  位置 是 deﬁned 作为:

Ls = −

1
Ns

·

NsX

j=1

log(σ(ˆrs,i − ˆrs,j)),

其中 Ns 是  数量 的 sampled 负 物品, ˆrs,i (或 ˆrs,j) 是 
预测了 分数 的 物品 i (或 j), i 是  正 物品, 和 j 是 
负 物品.  其他 类型 的 损失 称为 TOP1 是 也 devised, 该
是  比率 的 correctly ranked pairs 用 正则化.  TOP1
损失 在  位置 是 deﬁned 作为:

Ls =

1
Ns

·

NsX

j=1

σ(ˆrs,j − ˆrs,i) + σ(ˆr2

s,j).

 解决  问题 该  lengths 的 sessions vary, GRU4Rec
采用 会话-并行 mini-batches 用于  优化. 在 训练,
它 使用 popularity-基于 负 sampling, 该 假设 该 
更多 流行  物品 是,  更多 可能  用户 knows 关于 它, 用于
生成中  负 物品.

一 问题 用  RNN-基于 模型 (包括中  以上
介绍 GRU4Rec) 是 该 它们 仅 考虑  用户’s 序列
behaviors (短-术语 兴趣) 在  当前 会话, 和 进行 不 put

5.1. 基于表示学习的匹配

107

图 5.3: 模型 架构 的 GRURec. 它 processes 一 物品 的  物品
序列 在 一旦.

enough 重点 在  用户’s 通用 兴趣.  解决  问题,
Li 等人 (2017) 提出  组合  注意力机制 用
RNN, 称为 神经 Attentive 推荐 机器 (NARM). 作为
显示 在 图 5.4, NARM 采用  编码器-解码器 框架 用于
会话-基于 序列推荐. 给定  用户’s 点击 序列
x = (x1, x2, · · · , xt) consisting 的 t clicked 物品,  全局 编码器
在 NARM scans  输入 序列 用  GRU, 和 使用  ﬁnal
隐藏 陈述 cg
t = ht 作为  表示 的  用户’s 序列
行为.  局部 编码器 在 NARM 也 scans  输入 序列
用 另一个 GRU, 和 取  weighted sum 的  隐藏 陈述 作为
 表示 的  用户’s 主要 intent:

cl
t =

t
X

j=1

αtjhj,

其中 αtj 是  注意力 之间  positions j 和 t.  uniﬁed

输入: 1-的-N 编码 的 itemEmbedding layerGRU layerGRU layerGRU 层…Feedforward layeroutput: 分数 在 N 物品108

推荐中的深度匹配模型

序列 表示 是 形成了 作为  组合 的 cg

t 和 cl
t:

ct =

#

.

"

cg
t
cl
t

 uniﬁed 序列 表示 在 位置 t, 以及 嵌入 的 候选 物品, 是 fed 到  解码器.  相似度 之间
 序列 表示 在 位置 t 和 嵌入 的 候选
物品 i 是 计算了 作为  双线性 函数:

si = eT

i Bct,

其中 B 是  矩阵  是 学会了.  softmax 层 是 进一步 imposed
在  m 物品 分数  生成  分布 (的 点击) 超过 所有
候选 物品, 其中 m 是  数量 的 candidates.

 学习  模型 参数, 交叉-熵 损失 是 使用了. Speciﬁcally,
给定  训练 序列, 在  位置 t NARM 首先 预测 
概率 分布 超过  m 物品 qt. 从  log 我们 know
该  ground-真值 概率 分布 在 t 是 pt. 因此, 
交叉-熵 损失 是 deﬁned 作为

L =

m
X

i=1

t log qi
pi
t,

t 和 qi

其中 pi
t 是  预测了 概率 和 ground-真值 概率 用于 物品 i, 分别地.  损失函数 可以 是 优化 用
 标准 mini-批处理 SGD.

CNN-基于 方法

 代表性 CNN-基于 序列推荐 方法 是
Caser (卷积 序列 嵌入 推荐 模型)
(Tang 和 Wang, 2018).  基本 想法 是  treat  interacted 物品
在  嵌入 空间 作为  “图像”, 和 则 执行 2D 卷积
在  图像. 图 5.5 显示  结构 的  Caser 模型.

Let E ∈ <t×k 是  嵌入 矩阵 的 interacted 物品, 其中
t 是  数量 的 interacted 物品 (长度) 和 k 是  维度 的
embeddings (宽度). 每个 row 的  矩阵 是  嵌入 向量 的
 物品. Unlike  真实 图像 在 计算机 视觉, 那里 是 二 diﬃculties

5.1. 基于表示学习的匹配

109

图 5.4: 模型 架构 的 NARM.

在 应用中 卷积 operations 到 E 用于 序列推荐.
第一,  “图像” 长度 t 可以 是 diﬀerent 用于 diﬀerent 用户. 秒,
E 可以 不 具有 spatial relations 如 真实 images 从角度  宽度
的 嵌入 空间. 因此, 它 是 不 适合  采用  标准
2D CNN 过滤s, 这样的 作为 3 × 3 或 5 × 5.

 solve  二 问题, Caser 介绍 ‘完整-宽度’ CNN 过滤s
和 max-池化 operations. Speciﬁcally,  卷积 operations
在 Caser 覆盖  完整 columns 的  序列 “图像”. 该 是, 
宽度 的  过滤 具有  相同 大小 作为  嵌入 维度, 和
 高度 的  过滤 varies (see  diﬀerent colors 的 图 5.5 ()).
作为  结果, 过滤s 的 diﬀerent sizes 产生 特征 maps 用 diﬀerent
lengths.  ensure 所有 特征 maps 具有  相同 大小,  max-池化
操作 是 则 执行了 在 每个 特征 map 通过 extracting 仅
 largest 值. 作为 显示 在 图 5.5 (b),  数量 的 1 × 1 特征
maps 是 产生 之后 max 池化. 以下  concatenation
操作 (图 5.5 (c)) 和 softmax 层 (5.5 (d)), Caser 输出
 probabilities 的 下一个 物品. 注意 该 除之外  horizontal
过滤, Caser 也 利用  vertical 过滤 用  大小 的 t, 该 是
omitted 在 图 5.5.  特征 maps 1 × k 是 concatenated 一起
用 其他 特征 maps.

在 fact, due   max-池化 operations, Caser 是 不 好-suited
 模型 长-范围 sequences 或 repeated sequences.  alleviate 

!"!#!$!%…ℎ""ℎ#"ℎ$"ℎ%'!"!#!$!%…ℎ"(ℎ#(ℎ$(ℎ%(attentionsignal)=+(ℎ%(,ℎ%')sessionfeaturegenerator/%=[/%(,/%']/%2341234223432344……234msimilaritylayer:;=234;<=/%softmaxlayer>?23">?23#>?23$>?23@……>?23……localencoderglobalencoderrankingscoresitemembeddingsEncoderDecoder110

推荐中的深度匹配模型

图 5.5: 架构 的 Caser.

问题, Caser 采用  数据增强 方法 通过 sliding  window 超过  原始 序列  创建  集合 的 subsequences. 用于
例子, 假设 该  原始 序列 是 {x1, ..., x10} 和 
sliding window 大小 是 5, 和 则  subsequences 是 生成了 作为
{x1, ..., x5}, {x2, ..., x6}, ..., {x6, ..., x10}, 该 是 fed 到 模型 训练 一起 用  原始 序列.

之后 Caser 是 提出, 若干 方法 是 开发了  改进
 CNN 框架 用于 长-范围 序列推荐.  代表性 方法 是 NextItNet (Yuan 等人, 2019), 该 diﬀers 从
Caser 在 二 ways: (1) NextItNet 模型  用户 序列 在  autoregressive 方式, i.e., 序列--序列 (seq2seq); (2) NextItNet
exploits  stacked dilated CNN 层  增加  模型 receptive
ﬁeld, 和 因此 omits  使用 的 max-池化. Let p(x) 是  联合
分布 的 物品 序列 {x0, ..., xt}. 根据  链式法则,
p(x) 可以 是 建模 作为:

p(x) =

t
Y

i=1

p(xi|x0, · · · , xi−1, θ)p(x0),

(5.6)

其中 θ 表示  模型 参数, 和 Qt
i=1 p(xi|x0, ..., xi−1, θ)
表示  概率 的  i-th 物品 xi conditioned 在 所有 preceding 物品 {x0, ..., xi−1}. 用于 clarity, 我们 使  比较 之间

EmbeddingLook-upConvolutionLayersMax-poolingFeedForwardLayert()(b)(c)(d)5.1. 基于表示学习的匹配

111

NextItNet 和 Caser 在  生成 过程:

Caser : {x0, x1, ..., xi−1}
}

|

{z
输入

⇒ xi
|{z}
输出

N extItN et : {x0, x1, ..., xi−1}
|
}

{z
输入

⇒ {x1, x2, ..., xi}
}

|

{z
输出

(5.7)

其中 ⇒ 表示 ‘预测’. 在 fact,  ﬁnal 目标 函数 的
NextItNet 是  组合 的 所有 losses 的 tokens 在  整个 输出
序列. 因此, NextItNet 是 通常 不 sensitive   批处理 大小.

()

(b)

图 5.6: Dilated residual 块 (), (b).

此外, NextItNet 介绍 二 类型 的 dilated residual 块,
作为 说明了 在 图 5.6.  dilation factors 是 doubled 用于 每个
卷积 层 和 则 repeated, e.g., {1, 2, 4, 8, 16, ..., 1, 2, 4, 8, 16}.
 设计 允许  exceptional 增加 的 receptive ﬁelds. 因此,
NextItNet 是 好-suited  模型 长-范围 用户 sequences 和 捕获

NormalizationMasked1×3输入𝐸NormalizationNormalization+1×11×12𝑘𝑘ReLUReLUReLU2𝑘𝐸𝐹𝐸+𝐸𝐹𝐸Masked1×3输入𝐸NormalizationNormalization+Masked1×3𝐹𝐸+𝐸𝐸ReLUReLU2𝑘112

推荐中的深度匹配模型

图 5.7:  结构 的 SASRec

长-distance 物品 dependencies. 在 添加, unlike  RNN 模型,
 CNN 模型 基于 在  seq2seq 框架 面临  数据 leakage
问题 因为  未来 数据 可以 是 观察到 通过  更高-层
的  网络.  overcome 该 问题, NextItNet 介绍 
masking 技术, 通过 该  预测了 物品 itself 和 未来 物品
是 隐藏   更高-层. Masking 可以 是 simply implemented 通过
padding  输入 序列.

注意力-基于 方法

注意力 是 也 使用了 用于 学习 表示 从 序列 interactions.  好-已知 方法 是  自身-注意力 基于 序列
推荐 (SASRec) (Kang 和 McAuley, 2018) 模型. 它 是 inspired 通过  Transformer (Vaswani 等人, 2017), 取中  core 设计
的 自身-注意力  assign  权重   物品 在  序列 adaptively.
图 5.7 显示  结构 的  SASRec 模型.

Let E = V + P ∈ Rt×k 是  嵌入 矩阵 的  输入
序列, 其中 每个 row 表示  interacted 物品.  二 constituent 矩阵, V 表示  embeddings 的  物品 和 P 表示  embeddings 的  positions 的  相应 物品 在
 序列.  原因 用于 injecting P 是  augment  注意力
机制 用  序列 顺序 的  物品, 由于  注意力

5.1. 基于表示学习的匹配

113

机制 通过 本质 是 不 aware 的  序列 顺序. 则, E 是
fed 到  stack 的 自身-注意力 块, 其中 每个 块 具有 二 parts:
 自身-注意力 (SA) 层 和  指出-wise 馈-前向 网络
(FFN):

S(l) = SA(F(l−1)),

F(l) = F F N (S(l)),

(5.8)

其中 F(0) = E.  SA 层 是 deﬁned 作为:

SA(F) = 注意力(FWQ, FWK, FWV )

注意力(Q, K, V) = sof tmax(

,

)V

QKT
√
d

(5.9)

其中 WQ, WK 和 WV 是  权重 矩阵 的 查询, keys 和
值, 分别地.  SA 层 使用  相同 对象 F 作为 quires,
keys, 和 值, 该 是 投影 通过 diﬀerent 权重 矩阵 
改进 模型 ﬂexibility. Intuitively,  注意力 计算  weighted
sum 的 所有 值 vectors, 其中  权重 之间 查询 i 和 值 j
relates   交互 之间 查询 i 和 关键 j, i.e.,  结果 的
sof tmax(·).  denominator
d 是  避免 overly 大 值 的 
inner products 该 可以 cause 梯度 问题.

√

指出-wise 馈-前向 网络 具有  以下 形式:

F F N (S) = Relu(SW1 + b1)W2 + b2

(5.10)

其中 W1, W2 和 b1, b2 是 weights 和 偏置, 和 Relu 是 activation
函数. FFN 是 使用了  使能 非线性 和 考虑  interactions 之间 diﬀerent 潜在 dimensions.

另一个 问题  是 注意到 是 该 当 recommending  下一个 物品
的 一 序列, 仅  之前 物品 是 已知 (see 图 5.7). 该
是 实现了 通过 forbidding  链接 之间 Qi ( i-th 查询) 和 Kj
( j-th 关键) 用于 j > i, i.e., 设置  相应 注意力 weights
 0. 当  网络 goes deeper,  模型 成为 更多 diﬃcult 
训练.  solve  问题, SASRec 采用 层归一化 (Ba et
al., 2016), 随机失活 (Srivastava 等人, 2014), 和 残差连接 (他
等人, 2016) 在 每个 SA 层 和 FFN 层:

S(l) = F(l−1) + 随机失活(SA(LayerN orm(F(l−1)))),
F(l) = S(l) + 随机失活(F F N (LayerN orm(S(l)))).

(5.11)

114

推荐中的深度匹配模型

在 最后,  输出 的  最后 自身-注意力 块 是 使用了 用于 预测. 给定  历史 物品 序列 {v1, v2, . . . , vt},  下一个 物品
需求 是 预测了 基于 在 F(L)
, 其中 L 是  数量 的 块. 
预测了 分数 的  目标 物品 i 是:

t

ˆri = NT

j F(L)

i

,

(5.12)

N ∈ R(|I| × d) 是  嵌入 矩阵 的 目标 物品, 该 可以
或 是 训练了 端--端, 或  相同 作为  物品 embeddings 在
 输入 层.  authors 显示 该 sharing  物品 embeddings
可以 是 beneﬁcial.  目标 函数 是 逐点 交叉-熵
损失, 该 是 相似  Caser: 预测中 v2 基于 在  sub-序列
{v1}, 预测中 v3 基于 在  sub-序列 {v1, v2}, 和 因此 在.

除之外 SASRec, 另一个 代表性 注意力-基于
方法 用于 学习  表示 的 序列 interactions 是
BERT4Rec (Sun 等人, 2019).  主要 diﬀerence 是 该 它 取 
bi-directional 自身-注意力 模型  过程  序列, 该 可以
利用 两者 左 (之前) 和 正确 (未来) interactions.  未来
interactions 是 arguably 有用 用于 预测 (Yuan 等人, 2020) 因为 它们 也 reﬂect  偏好 的  用户, 和  rigid 顺序
的 interactions 可以 不 是 因此 重要 ( 顺序 是 derived 从 
交互 timestamp).  该 端, 它们 revise SASRec 在 二 ways:
(1) revise 自身-注意力  remove  零 constraints 在  注意力
weights 的 Qi 和 Kj 用于 j > i, 和 (2) 随机地 mask 一些 物品 在
 序列 和 预测  masked 物品 基于 在  左 和 正确
interactions,  避免 信息 leakage.

5.1.3 表示学习 从 多-模态 Content

除之外 用户-物品 interactions, 用户 和 物品 是 经常 associated
用 描述性 特征 这样的 作为 categorical attributes (e.g., age, gender,
乘积 类别) 和 文本 (e.g., 乘积 描述, 用户 综述).
除之外, 在  推荐 系统 用于 多-模态 物品 如 images,
videos, 和 musics, 其 多-模态 描述性 特征 是 readily
可用. Leveraging 这样的 辅助信息 是 beneﬁcial 用于 学习
更好 表示, 特别 用于 稀疏 用户 和 物品 该 具有
少数 interactions. 在 该 小节, 我们 综述 神经 推荐

5.1. 基于表示学习的匹配

115

模型 该 集成 多-模态 辅助信息 用于 表示
学习.  表示学习 组件 可以 是 abstracted 作为:

φu(u) = 组合(pu, f (Fu)),
φi(i) = 组合(qi, g(Gi)),

(5.13)

其中 pu 表示  嵌入 的 用户 u 该 是 学会了 从 历史
interactions (e.g.,  ID 嵌入 和  embeddings 从 
之前 subsections 可以 是 使用了), Fu 表示  侧 特征 的 用户
u 该 可以 是  矩阵 或  向量, 和 f (·) 是  表示
学习 函数 用于 侧 特征; 相似 notations 应用  qi, Gi,
和 g(·) 用于  物品 侧. 组合(·, ·) 是  函数 该 组合
 嵌入 从 历史 interactions 和  侧 特征. 
函数 f (·), g(·), 和 组合(·, ·) 可以 所有 是 realized 作为  深度
神经网络. 在  下一个, 我们 介绍 speciﬁc 方法 和 划分
它们 到 三 类型: 学习 从 categorical attributes, 用户 综述,
和 multimedia content 这样的 作为 图像 和 视频.

学习 从 Categorical Attributes

Wang 等人 (2017b) 提出  属性-aware 深度 CF 模型, 该
是 说明了 在 图 5.8. 它 projects 每个 categorical 特征 到 
嵌入 向量 和 则 执行 bi-交互 池化 (他 和
Chua, 2017) 用  用户 (物品) ID 嵌入. 最后,  pooled
用户 向量 和 物品 向量 是 结合 到  MLP  获得 
预测 分数:

φu(u) = BI-交互(pu, {fu

t }Vu

t=1) =

φi(i) = BI-交互(qi, {gi

t}Vi

t=1) =

VuX

t=1
ViX

t=1

pu (cid:12) fu

t +

qu (cid:12) gi

t +

VuX

VuX

t=1
ViX

t0=t+1
ViX

t=1

t0=t+1

t (cid:12) fu
fu
t0,

t (cid:12) gi
gi
t0,

ˆyui = MLP(φu(u) (cid:12) φi(i)),

t 和 gi

(5.14)
其中 fu
t 分别地 表示  embeddings 的 用户 属性
和 物品 属性, Vu 和 Vi 表示  数量 的 attributes 用于
用户 u 和 物品 i, 分别地.  bi-交互 池化 操作

116

推荐中的深度匹配模型

图 5.8: 模型 架构 的 属性-aware 深度 CF 模型.

考虑 所有 成对 interactions 之中 用户 ID 嵌入 和 属性 embeddings.  结合 用户 表示 φu(u) 和 物品
表示 φi(i) 是 interacted 通过 元素-wise 乘积, followed
通过  MLP 用于  ﬁnal 预测.  MLP 是  learnable 匹配
函数 (更多 细节 将 是 介绍 在 节 5.2), 该 可以
也 是 replaced 用 简单 内积.  优势 的 该 架构 是 该  interactions 之间 用户 (物品) attributes 和
 交叉-interactions 之间 用户 attributes 和 物品 attributes 是
好 捕获了.

Li 等人 (2015) 提出  正则化-基于 方法  incorporate attributes 到 推荐.  想法 是  首先 学习 表示 从 用户 特征 和 物品 特征 用 二 auto-encoders,
分别地, 和 则 联合地 训练  表示 之内  推荐 任务.  auto-编码器 损失 可以 是 treated 作为  正则化
术语 用于 推荐. 图 5.9 显示  模型 架构. 

10…001…001…010…1UserNodesItemNodesAttributeNodesAttributeNodes!"#"$"%&'())*+,-(!,01)'())*+,-(&,0+)MLPmatchingscorePoolingLayer31(!)3+(&)5.1. 基于表示学习的匹配

117

图 5.9: 模型 架构 的 属性-aware 深度 CF 模型.

左 auto-编码器 是 构建了 从 用户 特征 X 用  隐藏层
U 作为 用户 表示 和 L(X, U ) 作为 损失函数;  正确 自编码器 是 构建了 从 物品 特征 Y 用  隐藏层 V 作为 物品
表示 和 L(Y, V ) 作为  损失函数. 则 U 和 V 是
使用了  reconstruct  用户-物品 评分矩阵 R 用于 推荐 用 l(R, U, V ) 作为 损失函数.  整体 模型 是 训练了 通过
联合 优化 的  三 损失 函数 L(X, U ), L(Y, V ), 和
l(R, U, V ).

学习 从 用户 综述

 综述 给定 通过 其他 用户 经常 signiﬁcantly 在ﬂuence  用户’
在线 purchasing decisions 在  推荐 系统. 最近 研究
发现 该 leveraging  信息 在  综述 可以 帮助  系统
 不 仅 改进  准确率 但 也 增强  可解释性 在
推荐.

作为 一 的  代表性 工作, Zheng 等人 (2017) 提出  深度
学习 模型  联合地 学习 物品 properties 和 用户 opinions 从
 综述, 称为 深度 Cooperative 神经 网络 (DeepCoNN).
作为 显示 在 图 5.10, DeepCoNN 由组成 的 二 并行 神经

InputLayerHiddenLayer×UserFeatureXUserFeatureY#V≈OutputLayerUser-ItemRatingMatrixRr'(,*'+,,'-,*,,×××118

推荐中的深度匹配模型

图 5.10: 模型 架构 的 DeepCoNN.

网络. 一 聚焦 在 学习 的 用户 opinions 从  综述
(称为 N etu), 和  其他 学习 的 物品 properties 从  综述
(称为 N eti).  二 网络 是 coupled 一起 在 其 最后 层
和 联合地 学会了. 给定 所有  综述 written 通过  用户 u, N etu
首先 merges  综述 到  单一 文档 du
1:n 用 n 词. 则,
 文档 是 表示了 作为  矩阵 的 词 vectors V u
1:n:

1:n = [φ(du
V u

1 ), φ(du

2 ), · · · , φ(du

n)],

k 表示  k-th 词 在  文档 du

其中 du
1:n,  look-向上
k) returns  嵌入 的  输入 词 du
函数 φ(du
k, 和 c 是 
维度 的 embeddings. 则,  一-维 CNN 是 采用了
 总结  综述 到  表示 向量 xu:

xu = N etu(du

1:n) = CN N (V u

1:n).

类似地, 给定 所有  综述 用于  物品 i, N eti 也 merges 
综述 到  单一 文档 di
1:m 的 m 词, 创建  矩阵 的
词 vectors V i
1:m, 和 采用  一-维 CNN  总结
 综述 到  表示 向量:

xi = N eti(di

1:m) = CN N (V i

1:m).

综述 written 通过  用户⋯convolutionallayer𝑑#$𝑑%$𝑑&$𝑉&:#$𝐱$综述 written 通过  用户⋯convolutionallayer𝑑#*𝑑%*𝑑&*𝑉&:+*𝐱*factorizationmatching𝑦$*5.1. 基于表示学习的匹配

119

 ﬁnal 匹配分数 的 用户 u 和 物品 i 是 计算了 在  基础 的
 二 表示 vectors. Speciﬁcally, xu 和 xi 是 concatenated
到 一 向量 z = [xT
i ]T 和  分解机 (FM) 是 使用了
 计算  分数:

u , xT

yui = w0 +

|z|
X

wkzk +

|z|
X

|z|
X

wklzkzl,

k=1
其中 w0, wk, wkl’s 是 参数 的 FM.

k=1

l=k+1

Chen 等人 (2018) 指出 out 该 简单 concatenation 的 综述 作为
在 DeepCoNN means 相等 treatments 的 informative 综述 和 noninformative 综述.  解决  问题, 它们 提出  神经
注意力 回归 用 综述-水平 解释 (NARRE) 在 该
综述 是 assigned weights 和 informative 综述 是 强调了.
 模型 架构 的 NARRE 是 显示 在 图 5.11. Speciﬁcally,
给定 所有  m 综述 written 用于  物品 i,  综述 是 首先
变换 到 矩阵 Vi,1, Vi,2, · · · , Vi,m.  矩阵 是 则 sent
  convolutions 层 获得中  特征 vectors Oi,1, Oi,2, · · · , Oi,m.
之后 该,  注意力-基于 池化 层 是 exploited  aggregate
informative 综述  刻画  物品 i.  注意力 权重 用于
物品 i’s l-th 综述 是 deﬁned 作为

ai,l =

exp(∗
il)
k=1 exp(∗

ik)

Pm

,

其中 ∗

il 是  注意力 权重

il = hT ReLU(WOOi,l + Wuuil + b1) + b2,
∗
其中 uil 是  嵌入 的  用户 谁 writes  l-th 综述;
WO, Wu, h, b1, 和 b2 是 模型 参数.  ﬁnal 表示
的 物品 i 是 written 作为

xi = W0

m
X

l=1

ai,lOi,l + b0,

其中 W0 和 b0 是 模型 参数. 给定 所有  m 综述 written
通过  用户 u, 其 表示, 表示为 作为 xu, 是 计算了 类似地.
在 NARRE,  扩展了 潜在因子 模型 是 使用了 作为  预测

层 用于 计算中  ﬁnal 用户-物品匹配 分数:

yui = wT

1 ((qu + xu) (cid:12) (pi + xi)) + bu + bi + µ,

120

推荐中的深度匹配模型

图 5.11: 模型 架构 的 NARRE.

其中 (cid:12) 表示 元素-wise 乘积, qu 和 pi 分别地 表示
 用户 preferences 和 物品 特征, w1 是  权重 向量, bu, bi
和 µ 是  用户 偏置, 物品 偏置, 和 全局 偏置, 分别地.

学习 从 Multimedia Content

CNN 是 已知  是  eﬀective 特征 extractor 从 multimedia
content 这样的 作为 图像 和 视频 和 是 广泛使用 在 multimedia 推荐.  early 工作 是 Visual 贝叶斯个性化排序
(VBPR) (他 和 McAuley, 2016), 该 使用  深度 CNN  提取
 4096-维度 特征 向量 gi 从 每个 乘积 i’s 图像. 作为
 维度 的 gi 是 更高 比  维度 的 embeddings 在
协同 过滤ing, 该 是 典型地 在 顺序 的 数百, VBPR
projects gi 到  嵌入 空间 用  特征 变换

综述 written 通过  用户⋯CNNreview1𝑉#,%𝐱#CNNCNNreview2reviewn𝑉#,’𝑉#,(𝑂#%𝑂#’𝑂#(𝑖#%𝑖#’𝑖#(用户'综述'注意力⋯𝑎#%𝑎#’𝑎#(∑𝐪#(useridembedding)+综述 written 用于  物品⋯CNNreview1𝑉#,%𝐱.CNNCNNreview2reviewm𝑉#,’𝑉#,/𝑂#%𝑂#’𝑂#/𝑢.%𝑢.’𝑢./物品'综述'注意力⋯𝑎.%𝑎.’𝑎./∑𝐪.(itemidembedding)+元素-wiseproductpredictionlayer𝑦#.5.1. 基于表示学习的匹配

121

矩阵 E, 该 是, θi = Egi. 它 则 concatenates θi 用  物品 ID
嵌入 qi  形式  ﬁnal 物品 表示. 最后, 它 interacts
 物品 表示 用  用户 表示 用 内积
 获得  预测 分数, 该 是, ˆyui = φu(u)T [qi, Egi]. 注意 该
 偏置 术语 是 omitted 用于 clarity.  模型 是 学会了 用 
成对 BPR 损失.

它 是 worth noting 该 在 VBPR,  深度 CNN 是 pre-训练了 作为 
特征 extractor, 该 是 不 updated 期间 推荐 训练.
由于 深度 CNN 是 典型地 训练了 从  通用 图像 语料库 如
ImageNet, 它 可以 不 是 适合 用于  推荐 任务 如
clothing 推荐.  解决  问题, 三 解决方案 是
提出:

- Lei 等人 (2016) 提出  Comparative 深度学习 (CDL)
方法 用于 content-基于 图像 推荐. 而非 ﬁxing
 参数 的 深度 CNN, 它 也 updates 它们 在 训练. 
目标 函数 是 tailored 用于 推荐, 更多 speciﬁcally, 
变体 的  成对 BPR 损失 基于 在 用户 interactions. 作为  整体
模型 是 训练了 在  端--端 fashion,  特征 extracted 通过
深度 CNN 是 更多 适合 用于  推荐 任务.  最近 工作
采用 对抗 训练 和 学习 两者 深度 CNN 参数
和 推荐 参数 在  相似 方式 (Tang 等人, 2020).
然而, 作为  数量 的 用户-物品 interactions 是 典型地 许多
larger 比  数量 的 labeled 实例 在  图像 语料库, 该
解决方案 可以 suﬀer 从 长 训练 时间.

- Ying 等人 (2018) 提出 PinSage 用于 图像 推荐 和
Wei 等人 (2019) 提出 MMGCN 用于 micro-视频 推荐,
该 share  相同 想法 — reﬁning  extracted 图像 表示 在  用户-物品交互 图 用  图 卷积
网络.  extracted 图像 表示 是 treated 作为  初始 特征 的 物品 nodes, 该 是 propagated 在  交互
图 用  图 卷积 操作. 由于  交互 图
结构 包含 用户 偏好 在 物品 特别  协同
过滤ing signals, 该 方法 可以 使  reﬁned visual 特征 更多
适合 用于 personalized 推荐. 在  下一个 小节 (表示学习 从 图 数据), 我们 将 介绍 细节 的
如何 它 工作.

122

推荐中的深度匹配模型

- Diﬀerent 从  以上 二 解决方案 该 学习  整体 图像
表示 用  深度 CNN,  Attentive 协同过滤
(ACF) 方法 (Chen 等人, 2017) cuts  图像 到 49 (7 × 7)
regions. 它 采用  pre-训练了 深度 CNN  提取 特征 从
每个 区域 和  注意力 网络  学习  权重 的 它, 其中
 底层 假设 是 该 diﬀerent 用户 可以 是 感兴趣的
在 diﬀerent regions 的  图像.  49 regions 是 ﬁnally pooled 
获得  图像 表示. 作为  注意力 网络 是 训练了
基于 在 用户-物品 interactions,  图像 表示 是 适应
用于  推荐 任务.  框架 的 ACF 是 也 应用 
视频 推荐 通过  authors, 其中  区域 是 replaced 用
 frame sampled 从  视频.

5.1.4 表示学习 从 图 数据

 以上-提及了 表示学习 方法 具有  缺点 — 学习 从  信息 的  用户 ( 物品) 分别地,
同时  relations 之中 用户 和 物品 是 ignored.  用户-物品
交互 图 提供 rich 信息 在 用户 和 物品 relations,
和  物品 知识图谱 提供 rich 信息 在 物品 relations, 和 因此 学习 表示 从 这样的 graphs 可以 overcome
 缺点 和 具有  potential  改进  准确率 的 推荐. 若干 最近 工作 try  利用 该 信息
和 开发了 图 表示学习 基于 推荐 系统 (Wang 等人, 2019b; Wang 等人, 2019; Ying 等人, 2018; Wang
等人, 2019c). 用户-物品 interactions 是 组织 作为  bipartite 图,
social relations 之中 用户 是 呈现了 在 social 网络, 和 物品
知识 (e.g. 物品 attributes 和 relations) 是 表示了 在  形式
的 知识图谱 (aka. 异质信息网络). 这样的
 图 结构 connects 用户 和 物品, opening 向上 possibilities
 利用 高-顺序 relationships 之中 它们, 捕获 有意义的
patterns 在 它们 (e.g. 协同 过滤ing, social 在ﬂuence eﬀect, 和
知识-基于 推理), 和 改进  表示学习
的 它们.

我们 可以 分类 现有 工作 到 二 组 — (1) 二-阶段
学习 方法 (Wang 等人, 2019c; Gao 等人, 2018), 该 首先

5.1. 基于表示学习的匹配

123

extracts relations 作为 triples 或 paths, 和 则 学习 节点 表示 使用中  relations, 和 (2) 端--端 学习 方法 (Wang
等人, 2019b; Wang 等人, 2019), 该 直接地 学习 表示
的 nodes 其中 propagation 的 信息 是 carried out 之中 nodes.

端--端 建模: 神经 图 协同过滤 (NGCF)

由于 用户-物品 interactions 可以 是 表示了 在  bipartite 图,
神经 图 协同过滤 (NGCF) (Wang 等人, 2019b)
revisits 协同 过滤ing (CF) 通过 deﬁning CF signals 作为 高-顺序
connectivities 在  图. Intuitively, 直接 connections 显式地
刻画 用户 和 物品 —  用户’s interacted 物品 给出 支持中
证据 在  用户’s 偏好, 同时  物品’s associated 用户 可以
是 viewed 作为 特征 的  物品. 此外, 高-顺序 connectivities
reﬂect 更多 复杂 patterns —  路径 u1 ← i1 ← u2 表明 
behavioral 相似度 之间 用户 u1 和 u2, 作为 两者 具有 interacted
用 物品 i1;  longer 路径 u1 ← i1 ← u2 ← i2 建议 u1’s 偏好
在 i2, 由于 他的/她的 相似 用户 u2 具有 采用 i2. 图 5.12 显示 
例子 的 这样的 高-顺序 connectivities, 该 reﬂect  用户-用户
和 物品-物品 dependencies. NGCF 旨在  inject 这样的 signals 到 
表示 的 用户 和 物品.

图 5.12:  例子 的 高阶连通性 revealed 在 用户-物品交互
图.  ﬁgure 是 取 从 (Wang 等人, 2019b)

Inspired 通过  最近 成功 的 图 神经 网络 (GNNs) (Ying
等人, 2018), 该 是 构建了  信息 propagation (或 messaging passing) 在 graphs, NGCF 执行 嵌入 propagation 在 
bipartite 用户-物品交互 图. 图 5.13 显示 其 框架.
形式化地,  图 卷积 层 的 GNN 是 composed 的 二

124

推荐中的深度匹配模型

图 5.13: 模型 架构 的 NGCF.

组件 — (1) 消息 构建, 该 deﬁnes  消息
是 propagated 从  neighbor 节点   当前 节点 和 (2)
消息 聚合, 该 aggregates  messages propagated 从
 neighbor nodes  更新  表示 的  当前 节点.
一 广泛-使用了 实现 在  l-th 层 是 作为 follows:

u = ρ(m(l)
p(l)

u←u + X

j∈Nu

m(l)

u←j), m(l)

u←j = αujW(l)q(l−1)

j

,

(5.15)

其中 p(l)
u 表示  表示 的 用户 u 之后 l-层 propagation,
ρ(·) 是  非线性 激活函数, 和 Nu 是  neighbor 集合 的
u; m(l)
u←j 是  消息 是 propagated, αuj 是  decay 因素 用于
propagation 在 边 (u, j) 该 是 heuristically 集合 作为 1/
|Nu||Nj|,
和 W(l) 是  learnable 变换 矩阵 在  l-th 层. 作为 这样的,
 L-顺序 connectivity 是 encoded 到  updated 表示.
此后, NGCF concatenates 表示 从 diﬀerent 层
该 reﬂect 变化 contributions  用户 preferences 和 执行
预测 作为:

q

f (u, i) = p∗>

u q∗

i , p∗

u = p(0)|| · · · ||p(L), q∗

i = q(0)|| · · · ||q(L), (5.16)

其中 || 表示  concatenation 操作.

!"#$%#!$#$%#!$&$%#!"#'!()'*#*'*'+++=1+=2+=3+=1+=2+=3concatenateconcatenate!"##!"#&!"#0!(##!(#&!(#0×!"∗!(∗2345678#,:)EmbeddingsEmbeddingsPropagationPredication!()$%#!"&$%#!"0$%#*#5.1. 基于表示学习的匹配

125

它 是 worth mentioning 该 MF 和 SVD++ 可以 是 viewed 作为
special 案例 的 NGCF 用 无 和 一-顺序 propagation 层, 分别地. 此外, 一 可以 implement  图 卷积 层
在 diﬀerent ways. 用于 例子, SpectralCF (Zheng 等人, 2018) 使用
 谱卷积 操作  执行 信息 propagation;
GC-MC (Berg 等人, 2017) 组合 MLP 用 方程 (5.15) 
捕获 非线性 和 复杂 patterns.

同时 NGCF 具有 证明了  strengths 的 使用中  交互
图 结构 用于 表示学习,  最近 研究 (他 等人,
2020) (LightGCN) 显示 该 许多 designs 在 NGCF 是 redundant,
特别  非线性 特征 transformations.  主要 论证 是
该 在  用户-物品交互 图, 每个 节点 (用户 或 物品) 是 仅
描述 通过  一-hot ID, 该 具有 无 语义 除之外 是 
identiﬁer. 在 这样的  案例, 执行中 多个 层 的 非线性 特征
变换, 该 是  标准 操作 的 神经 网络, 将
bring 无 beneﬁt.  validate 该 论证, 它们 提出  简单 模型
名为 LightGCN, 该 retains 仅  邻域聚合 在
图 卷积:

u = X
p(l)

i∈Nu

i = X
q(l)

u∈Ni

1
p|Nu|p|Ni|
1
p|Ni|p|Nu|

q(l−1)
i

,

p(l−1)
u

,

(5.17)

i

u 和 q(0)

其中 p(0)
是  模型 参数 的 ID embeddings. 我们 可以
see 该 在  轻 图 卷积, 非线性 特征 transformations
和 自身-connections 是 removed. 之后 L 这样的 层  aggregate highorder neighborhood, LightGCN sums 向上  表示 的 所有 层
作为  ﬁnal 表示 用于  用户/物品:

p∗

u =

L
X

l=0

αlp(l)
u ;

q∗

i =

L
X

l=0

αlq(l)
i

,

(5.18)

其中 αl 表示  重要性 的  表示 的  l-th
层, 该 是 pre-deﬁned.  authors prove 在 理论 该  sum
aggregator subsumes  自身-connections 在 图 卷积. 因此,
 自身-connections 可以 是 safely removed 从 图 卷积. 用

126

推荐中的深度匹配模型

 相同 数据 和 评估 方法 的 NGCF, LightGCN 获得
关于 15% 相对地 improvements, 该 是 very signiﬁcant.

端--端 建模: 知识图谱注意力网络 (KGAT)

除之外 用户-物品 interactions, 更多 最近 工作 也 取
到 consideration  relations 之中 物品 在  知识图谱.
知识图谱 (KG) 是  强大 资源 该 提供 rich 侧
信息 在 物品 (i.e. 物品 attributes 和 物品 relations), 其中
nodes 是 entities 和 edges 表示  relations 之间 它们.
通常 KG 组织 facts 或 beliefs 在  异质 directed 图
G = {(h, r, t)|h, t ∈ E, r ∈ R}, 其中  triplet (h, r, t) 表明 该
那里 是  关系 r 从 head 实体 h  tail 实体 t. 用于 例子,
(Hugh Jackman, ActorOf, Logan) 陈述  fact 该 Hugh Jackman 是
 actor 的  movie Logan.

 使用 的  知识图谱 可以 增强  学习 的 物品
表示 和 建模 的 用户-物品 relationships. 在 particular,
直接 connections 的  实体 — 更多 speciﬁcally 其 associated triples
— proﬁle 其 特征. 用于 例子,  movie 可以 是 刻画了 通过 其
director, actors, 和 genres. 此外,  connections 之间 entities,
特别 多-hop paths, stand 用于 复杂 relationships, 和 捕获
复杂 协会 patterns. 在 movie 推荐, 用于 例子,
用户 是 connected  Logan 因为 它们 如  Greatest Showman
acted 通过  相同 actor Hugh Jackman. Obviously, 这样的 connections
可以 帮助  原因 关于 unseen 用户-物品 interactions (i.e.  potential
推荐).

朝向  端, 知识图谱注意力网络 (KGAT)
(Wang 等人, 2019) 扩展 NGCF 通过 adaptively extracting 信息
从  neighborhood 的 高阶连通性. Diﬀerent 从 NGCF
其中  decay 因素 αht 在 propagation 的 信息 在 边 (h, t)
是 ﬁxed, KGAT 采用  relational 注意力机制 取中 到
consideration  关系 r 的 边 (h, r, t). 图 5.14 显示 
框架.  attentive 嵌入 propagation 层 是 formulated

5.1. 基于表示学习的匹配

127

图 5.14: 模型 架构 的 KGAT.  左 subﬁgure 说明  整体
模型 框架, 和  正确 subﬁgure 说明  图 卷积 操作
在 KGAT.

作为:

, {m(l)

h = f1(p(l−1)
p(l)
(h,r,t) = f2(q(l−1)
m(l)

h

t

(h,r,t)|(h, r, t) ∈ Nh}),

(5.19)

, α(h,r,t)), α(h,r,t) =

exp g(ph, er, qt)
(h,r0,t0) exp g(ph, er0, qt0)

,

P

其中 f1(·) 表示  消息 聚合 函数, 该 updates
 表示 的  head 实体 h, f2(·) 是  注意力 消息
构建 函数, yielding messages 从 tail 实体 t  head
实体 h, α(h,r,t) 是  attentive decay 因素 derived 从  注意力 网络 g(·), 表明中 如何 许多 信息 是 propagated 和
identifying 重要性 的 neighbors 用 regard  关系 r. 之后
establishing  表示, KGAT 使用  相同 预测 模型
作为 方程 (5.16)  estimate 如何 可能  用户 将 adopt  物品.

二-阶段 建模: 知识 路径 Recurrent 网络 (KPRN)

除之外 端--端 建模  增强 表示学习 用
高阶连通性, 一些 工作 (Gao 等人, 2018; Wang 等人,
2019c) 介绍 meta-paths 或 paths  直接地 reﬁne  相似之处
之间 用户 和 物品. 在 particular,  模型 首先 或 deﬁne
meta-路径 patterns (Gao 等人, 2018) 或 提取 qualiﬁed paths (Wang
等人, 2019c), 和 则 馈 它们 到  监督学习 模型 
预测  分数. 这样的  方法 可以 是 formulated 作为 follows. 给定

!"#$!%&$'=1'=2'=3concatenateconcatenate!"##!"#,!"#&!%##!%#,!%#&×!"∗!%∗./0#,2&CKGEmbeddingLayerPredicationLayer0#0,0&032#2,2&23!#!,!&'=1'=2'=3AttentiveEmbeddingsPropagationAttentiveEmbeddingsPropagationAttentiveEmbeddingsPropagation⊗⊗⊗⊕LeakyReLU𝑒$%&’(𝑒)*&’(𝑒+,&’(π𝑖/,𝑟/,𝑒(π𝑖/,−𝑟(,𝑢4𝑊(&⊗⊗⊗⊕LeakyReLU𝑒$%&’(𝑒)*&’(𝑒+,&’(π𝑖/,𝑟/,𝑒(π𝑖/,−𝑟(,𝑢4𝑊4&⊗⊕𝑒)*(&)128

推荐中的深度匹配模型

 用户 u,  目标 物品 i, 和  集合 的 paths P(ui) = {p1, · · · , pK}
connecting u 和 i, 其 匹配分数 是 计算了 作为 f (u, i|P(u, i)).

图 5.15: 模型 架构 的 KPRN.

之中 它们, 知识 路径 Recurrent 网络 (KPRN) (Wang
等人, 2019c) 是  代表性 模型, 该 是 显示 在 图 5.15.
给定  路径 之中 entities, KPRN 采用 recurrent 网络 如
LSTM  encode  元素 在  路径  捕获  compositional
语义 的 entities 和 relations. 此后, KPRN exploits  池化
层  组合 多个 路径 表示 到  单一 向量, 和
则 feeds 它 到  MLP  获得  ﬁnal 分数 用于  用户-物品
对. 形式化地,  预测 模型 是 deﬁned 作为:

xk = LSTM([ph1||er1, · · · , phL||erL]),
f (u, i) = MLP( X

xk),

k∈P(u,i)

(5.20)

其中 pk = [h1, r1, · · · , hL, rL] 是  k-th 路径, (hl, rl, hl+1) 是  l-th
triplet 在 pk, 和 L 表示  triplet 数量. 作为 这样的, KPRN 可以
采用  LSTM 模型  利用  序列 信息 在 
知识图谱 和  增强  推荐 模型’s 解释
能力, revealing 为何  推荐 是 使.

ShapeofYouEdSheeran÷IseeFireItemPersonAlbumItemSungByProduceContainSong<端>CCCCC⊕⊕⊕⊕⊕⊕⊕⊕⊕⊕𝑟#𝑟$𝑟%𝑟&𝑟’())𝑒$𝑒%𝑒&𝑒+𝑒$,𝑒%,𝑒&,𝑒+,𝑥#𝑥$𝑥%𝑥&𝑥+ℎ#ℎ$ℎ%ℎ&AliceUser𝑒#𝑒#,Interact𝑝01#𝑝0𝑝02#(Alice,Internet,IseeFire)𝑠𝜏𝑃PoolingLayerEmbeddingLayerLSTMLayer5.2. 基于匹配函数学习的匹配

129

5.2 基于匹配函数学习的匹配

 匹配函数 输出  匹配分数 之间  用户
和  物品, 用 用户-物品交互 信息 作为 输入, 沿着
用 可能 辅助信息 包括中 用户 attributes, 物品 attributes,
contexts, 和 others. 我们 分类  方法 到 二 类型 基于
在  输入   匹配函数 — 二-方式 匹配 (仅
用户 信息 和 物品 信息 是 提供) 和 多-方式
匹配 (其他 辅助信息 是 也 提供).

5.2.1 二-方式 匹配

传统 潜在空间 模型 计算 内积 或 余弦相似度 之间 用户 和 物品  获得  匹配分数. 然而,
这样的  简单 方式 的 匹配 具有 limitations 在 模型 表达能力.
用于 例子, (他 等人, 2017c) 显示 该 它 可以 incur  大 排序
损失 due  其 inability 的 maintaining  triangle inequality (Hsieh
等人, 2017). 因此, 它 是 必要  开发 更多 复杂 和
expressive 匹配 函数. 我们 分类 现有 工作 沿着 该
行 到 二 类型: 相似度 学习 方法 和 度量 学习
方法.

相似度 学习 方法

神经协同过滤 (NCF) (他 等人, 2017c) exploits 
通用 神经网络 框架 用于 协同 过滤ing.  想法
是  位置  多-层 神经网络 以上 用户 嵌入 和 物品
嵌入  学习 其 交互 分数:

f (u, i) = F (φu(u), φi(i)),

其中 F 是  交互 神经网络  是 speciﬁed, φu(u) 和
φi(i) 表示  embeddings 的 用户 u 和 物品 i, 分别地. 若干
实例 是 提出 之下  NCF 框架:

- 多-层 感知 (MLP).  straightforward 方式 是  stack
 MLP 以上  concatenation 的 用户 嵌入 和 物品 嵌入, leveraging  non-线性 建模 能力 的 MLP  学习 
交互 函数: F (φu(u), φi(i)) = M LP ([φu(u), φi(i)]). 虽然

130

推荐中的深度匹配模型

theoretically sound (由于 MLP 可以 近似 任何 连续 函数 在 理论), 该 方法 进行 不 执行 好 在 实践 和
underperforms  简单 MF 模型 大多数 的  时间 (用于 证据 see
(他 等人, 2017c)). 作为 revealed 在 (Beutel 等人, 2018),  关键 原因
是 该 它 是 practically diﬃcult 用于 MLP  学习  multiplication
操作, 该 是, 然而, very 重要 用于 建模 的 交互 在 CF (相应   低-排序 假设 的 用户-物品
交互 矩阵). 它 是 重要, 因此,  显式地 express 
multiplication 或 相似 eﬀect 在  匹配 网络.

- 广义 矩阵分解 (GMF).  泛化 MF 之下  NCF 框架,  authors 的 NCF 首先 计算 elementwise 乘积 在 用户 嵌入 和 物品 嵌入, 则 输出
 预测 分数 用  完全地 connected 层: F (φu(u), φi(i)) =
σ(wT ([φu(u) (cid:12) φi(i)])). w 是  trainable 权重 向量 的  层,
该 assigns diﬀerent weights  interactions 的 diﬀerent dimensions.
Fixing w 作为  所有-一 向量 1 可以 完全地 recover  MF 模型. 因此,
在 原则, GMF 可以 实现 更好 性能 比 MF (注意 该
 choices 的 损失函数 可以 ﬀect  results). 它 是 也 reasonable
 进一步 stack  MLP 以上  元素-wise 乘积 在 GMF, 该
是  自然 方式  解决  inability 的 multiplication 学习 通过
MLP. 该 方法 appears 在 (Zhang 等人, 2017b) 和 证明
good 性能.

- 神经 矩阵分解 (NeuMF).  使用 的 MLP 可以 endow
 交互 函数 用 non-线性.  补充 GMF 用
MLP 和 组合 其 strengths,  authors 的 NCF 提出  集成 模型, 作为 说明了 在 图 5.16. 它 使用 separated 嵌入
集合 用于 GMF 和 MLP, concatenating  最后 隐藏 层 的 
二 模型 在之前 projecting   ﬁnal 匹配分数. 该 模型
具有 更高 表示 能力. 然而, 它 是 也 hard  训练
如果 训练 是 进行了 从  scratch. Empirically, initializing 
参数 用 pre-训练了 GMF 和 MLP 导致  更好 性能, 该 是 高度 encouraged 在 实践. 此外, sharing 
嵌入 层 是 也 reasonable  减少  数量 的 参数,
该 是 subjected  设计 (Guo 等人, 2017).

- 卷积 NCF (ConvNCF) (他 等人, 2018b).  显式地
模型  correlations (interactions) 之间 嵌入 dimensions,

5.2. 基于匹配函数学习的匹配

131

图 5.16: 模型 架构 的 NeuMF.

他 等人 (2018b) 提出  使用 outer 乘积 在 用户 嵌入
和 物品 嵌入, followed 通过  CNN  aggregate  interactions
hierarchically. 图 5.17 说明  模型.  输出 的  outer
乘积 是  2D 矩阵, 其中  (k, t)-th 条目 是 (pu ⊗ qi)kt = puk · qit,
捕获中  交互 之间  k-th 维度 和  t-th
维度 (pu 和 qi 表示 用户 嵌入 和 物品 嵌入).
作为  2D 矩阵 encodes 成对 interactions 之间 嵌入
dimensions, stacking  CNN 以上 它 可以 捕获 高-顺序 interactions
之中 嵌入 dimensions, 因为 每个 更高 层 具有  larger
receptive ﬁeld 在  矩阵. 此外, CNN 具有 少数er 参数
比 MLP, 该 可以 是 diﬃcult  训练 和 是 不 encouraged.

度量 学习 方法

度量 学习 方法 aim  学习 和 利用 distance metrics 
quantitatively 度量  relationships 之中 数据 实例. Mathematically,  distance 度量 needs  满足 若干 conditions, 和
之中 它们  triangle inequality 是  重要 一 用于 泛化 (Kulis 等人, 2013).  early 和 代表性 工作 该

000100…000010…MFUserVectorMLPUserVectorMFItemVectorMLPItemVectorGMFLayerMLPLayer1MLPLayerNNeuMFLayer…concatenationelement-wiseproductReLUReLUconcatenation!"#$"#$trainingloglossuser%物品&scoretarget132

推荐中的深度匹配模型

图 5.17: 模型 架构 的 ConvNCF.

介绍 度量 学习 到 推荐 是 协同 度量
学习 (CML) (Hsieh 等人, 2017), 该 指出 out 若干 limitations 的 使用中  内积 用于 协同 过滤ing 因为
的  dissatisfaction 的 triangle inequality. 作为  结果, 它 是 不 able
 捕获 ﬁner-grained 用户 preferences 和 用户-用户 和 物品-物品
relations (因为  相似度 关系 cannot 是 properly propagated
用  内积). 它们 则 formulate  基本 度量 学习
框架 用于 协同 过滤ing, 该 是 扩展了 通过 一些 后来
工作 如 (他 等人, 2017; Tay 等人, 2018). 下一个, 我们 brieﬂy 介绍
 方法.

- 协同 度量 学习 (CML) (Hsieh 等人, 2017). 
用户-物品 度量 在 CML 是 deﬁned 作为  欧氏距离 之间
用户 嵌入 和 物品 嵌入:

d(u, i) = ||pu − qi||,

(5.21)

其中 pu 是  用户 嵌入 向量, qi 是  物品 嵌入 向量,
和 d(u, i) 是  distance 之间 用户 u 和 物品 i,  smaller, 
更多 相似.  优势 的 学习 和 利用中  度量 是 该
 相似度 之中 实例 可以 是 propagated. 用于 例子, 如果 它 是
已知 该 “pu 是 相似  两者 qi 和 qj”, 则  学会了 度量

用户!物品"UserEmbeddingItemEmbeddingInteractionMap…InteractionFeature#$⊗&'#(×*&+×*,-$'BPRTrainingHiddenLayers5.2. 基于匹配函数学习的匹配

133

将 不 仅 使 pu get closer  qi 和 qj, 但 也 使 qi 和
qj themselves closer. 该 属性 是 fairly 有用  捕获 用户-用户
和 物品-物品 relationships 从 用户-物品 interactions.  直觉
是 该  物品 该  用户 likes 是 close   用户 比  其他
物品 该  用户 进行 不 如. 因此,  间隔-基于 成对 损失 是
deﬁned 作为:

L = X

X

(u,i)∈D+

(u,j)∈D−

wui[δ + d(u, i)2 − d(u, j)2]+,

(5.22)

其中 i 表示  物品 该 u likes, j 表示  物品 该 u 进行 不
如, δ > 0 是  predeﬁned 间隔 大小, [z]+ = max(z, 0) 表示 
hinge 损失函数, 和 wui 是  权重 的 训练实例 该
是 predeﬁned.  authors 提出 若干 additional constraints 
改进  质量 的  学会了 metrics, 包括中  bound 的 用户
嵌入 和 物品 嵌入 之内  单元 sphere (i.e., ||p∗|| ≤ 1
和 ||q∗|| ≤ 1), 和  regularizer  de-correlate  dimensions 的 
学会了 度量. 我们 指代  读者   原始 论文 用于 更多
细节 (Hsieh 等人, 2017).

- 翻译-基于 推荐 (TransRec) (他 等人, 2017).
TransRec 可以 是 seen 作为  扩展 的 CML 用于 下一物品推荐, 该 accounts 用于 用户 序列 行为 通过 建模 
第三-顺序 交互 之中  用户,  之前 visited 物品, 和
 下一个 物品  visit (Rendle 等人, 2010).  想法 是 该,  用户
是 表示了 作为  “翻译 向量”, 该 translates  之前
物品   下一个 物品, i.e., qj + pu ≈ qi.  distance 度量  realize
 翻译 是:

d(qj + pu, qi) = ||qj + pu − qi||,

(5.23)

其中 所有 嵌入 vectors 是 re-缩放 在  单元 长度.  authors
则 estimate  似然 该  用户 使  转移 从 物品
j  物品 i 作为:

prob(i|u, j) = βi − d(qj + pu, qi),

(5.24)

其中 βi 是  偏置 术语  捕获  物品 popularity.  TransRec
模型 是 学会了 用  成对 BPR 损失. 比较了 用 CML,
TransRec 考虑  之前 物品 和  转移 关系 之间

134

推荐中的深度匹配模型

它 和  下一个 物品. 最近, Wu 等人 (2019) 扩展 TransRec 通过
建模 多个 之前 物品 和 高-顺序 interactions 用 它们.
- 潜在 Relational 度量 学习 (LRML) (Tay 等人, 2018).
LRML 进展 TransRec 通过 进一步 学习  关系 之间
 用户 和  下一个 物品.  优势 是 该  度量 是 更多
geometrically ﬂexible.  度量 的 LRML 是:

d(u, i) = ||pu + r − qi||,

(5.25)

其中 r 是  潜在 关系 向量  是 学会了. 而非 学习 
uniform r 用于 所有 用户-物品 pairs, LRML parameterizes 它 作为  attentive sum 超过 external 记忆 vectors. 图 5.18 显示  模型
架构. Let  external 记忆 vectors 是 {mt}T
t=1, 和  keys
的  记忆 vectors 是 {kt}T
t=1, 和 两者 的 该 是 模型 free
参数  是 学会了.  关系 向量 r 是 parameterized 作为:

r =

M
X

t=1

atmt,

在 = sof tmax((pu (cid:12) qi)T kt),

(5.26)

其中 在 是  attentive 权重 的 记忆 mt 生成了 通过  注意力
网络 该 取  交互 之间 用户 嵌入 和 物品
嵌入 作为 输入. 在 该 方式,  关系 向量 是 用户-物品交互 aware, 该 增加  geometrical ﬂexibility 的  度量.
 模型 是 学会了 通过 optimizing  成对 合页损失 该 是 
相同 作为 该 在 CML.

5.2.2 多-方式 匹配

方法 的 多-方式 匹配 是 generic 特征-基于 方法, 仅
如  FM 模型 该 取 特征 作为 输入 和 incorporates 特征
interactions 在  匹配函数.  方法 允许  utilization
的 任何 种类 的 辅助信息 和 上下文. 然而, 它们 可以 具有
更高 复杂度 比较了  二-方式 匹配 和 表示
学习-基于 方法. 作为 这样的, 它们 是 更多 经常 使用了 在 
排序 阶段 而 比 在  候选检索 阶段, 这样的 作为
点击率 (CTR) 预测.

5.2. 基于匹配函数学习的匹配

135

图 5.18: 模型 架构 的 LRML.

特征 交互 建模 旨在  捕获 交叉-特征 eﬀects,
i.e., signals 从 多个 特征. 用于 例子, 用户 的 age 20-25
(特征 1) 和 gender female (特征 2) 是 更多 可能  purchase
iPhones 的 pink color (特征 3).  naive 解决方案 用于 捕获中 这样的
eﬀects 是  手动地 构建 交叉 特征, feeding 它们 到 
线性 模型 该 可以 学习 和 memorize  重要性 的  交叉
特征 (Cheng 等人, 2016).  问题 是 该 它 可以 仅 memorize 
seen 交叉 特征 (在 训练数据), 和 cannot 泛化  unseen
交叉 特征. 此外,  数量 的 交叉 特征 增加 polynomially 用  顺序 的 crossing. 因此 它 需要 领域 知识 
选择 有用 交叉 特征 而非 使用中 所有 交叉 特征. 因此,
我们 需求 更多 eﬀective 和 eﬃcient 技术 用于 特征 交互
建模.

我们 分类 现有 工作 到 三 类型 基于 在 如何 特征
interactions 是 建模: 隐式 交互 建模, 显式 交互 建模, 和  组合 的 隐式 和 显式 交互
建模.

UserEmbeddingLayerItemEmbeddingLayerUserVector!′HadamardProductUserVector!PairwiseHingeLossK1K2K3K4K5K6M1M2M3M4M5M6RelationLayer!+$−&TranslationLayer!′+$−&′ItemVector&′ItemVectorqJointEmbeddingSoftmaxNegativeSamplingUser-ItemKeysUser-ItemKeysLatentRelationVector$MemorySlicesRelationModelingLayer136

推荐中的深度匹配模型

隐式 交互 建模

在 Recsys 2016,  YouTube 团队 呈现了  深度神经网络
模型 用于 YouTube 推荐 (Covington 等人, 2016). 它 projects
每个 categorical 特征 作为  嵌入 向量 (用于 序列 特征
如 watched 物品, 它 执行 average 池化  获得  序列
嵌入 向量). 它 则 concatenates 所有 embeddings, feeding  concatenated 向量 到  三-层 MLP  获得  ﬁnal 预测.
 MLP 是 expected  学习  interactions 之中 特征 embeddings, 因为 其 强 表示 能力 在 approximating
任何 连续 函数. 然而,  特征 交互 建模
是 而  隐式 过程 由于  interactions 是 encoded 在 
隐藏 单元 的 MLP, 和 那里 是 无 方式  识别 该 interactions
是 重要 用于  预测 之后  模型 是 训练了. 此外, 它 是
practically diﬃcult 用于 MLP  学习 multiplication eﬀect (Beutel 等人,
2018), 该 是 重要  捕获 交叉 特征.

该 简单 架构 成为  pioneer 工作 的 利用中 深度
神经 网络 用于 推荐, 和 许多 后来 工作 使 extensions 在 它. 用于 例子, 广泛&深度 (Cheng 等人, 2016) ensembles 
深度模型 (i.e.,  深度 部分) 用  线性 回归 模型 (i.e., 
广泛 部分), 该 包含 sophisticated 特征 包括中 手动地
构建了 交叉 特征. 深度 Crossing (Shan 等人, 2016) deepens
 MLP  十 层 用 residual connections 之间 层. 作为 将
是 介绍 下一个, 许多 集成 模型 如 DeepFM (Guo 等人,
2017) 和 xDeepFM (Lian 等人, 2018) 集成  深度 架构
到  浅层 架构  augment 隐式 交互 建模
用 显式 交互 建模.

显式 交互 建模

FM 是  传统 模型 该 执行 秒-顺序 交互 建模 (我们 具有 介绍  模型 在 章 2.4.3). Speciﬁcally,
它 projects 每个 non-零 特征 xi 到  嵌入 vi, 执行
内积 在 每个 对 的 non-零 特征 embeddings, 和 sums
超过 所有 inner products ( 首先-顺序 线性 回归 部分 是 omitted
用于 clarity). Due  其 eﬀectiveness, FM 是 扩展了 之下  神经
网络 框架 用于 显式 交互 建模.

5.2. 基于匹配函数学习的匹配

137

图 5.19 显示  神经 分解机 (NFM) 模型 (他
和 Chua, 2017).  想法 是  replace  内积 用 
元素-wise 乘积, 该 输出  向量 而 比  标量, 和
则 stacks  MLP 以上  sum 的 元素-wise products.  core
操作 在 NFM 是 称为 Bi-交互 池化, deﬁned 作为:

fBI (Vx) =

n
X

n
X

i=1

j=i+1

xivi (cid:12) xjvj,

(5.27)

其中 xi 表示  值 的 特征 i, Vx 表示  集合 的 embeddings
的 non-零 特征, 和 n 表示  数量 的 non-零 特征.
 向量 获得了 通过 Bi-交互 池化 encodes 秒-顺序
interactions. 通过 stacking  MLP 以上 它,  模型 具有  能力 
学习 高-顺序 特征 interactions.

图 5.19: 模型 架构 的 NFM.

一 问题 用 FM 和 NFM 是 该 所有 秒-顺序 interactions
是 考虑了 equally 重要, 和 它们 contribute evenly  
预测.  解决 该 问题, Attentional 分解机
(AFM) (Xiao 等人, 2017) 是 提出  diﬀerentiate  重要性
的 interactions 用  注意力 网络. 图 5.20 显示  架构 的 AFM.  输入 和 嵌入 层 是  相同 作为
那些 在  标准 FM.  对-wise 交互 层 执行 
元素-wise 乘积 在 每个 对 的 特征 embeddings  获得 
交互 vectors; 该 步骤 是 等价  那些 的 FM 和 NFM.
 注意力 网络 取 每个 交互 向量 vi (cid:12) vj 作为 输入
和 输出  重要性 权重 aij 用  二-层 MLP. 则,

010100.20……InputFeatureVectorV"V#V$……EmbeddingLayerBi-InteractionPooling…%&B-InteractionLayerHiddenLayersPredictionScore138

推荐中的深度匹配模型

 模型 使用  重要性 权重  re-权重 每个 交互
向量 和 sum 向上 所有 交互 vectors  获得  ﬁnal 分数. 
方程 的 AFM 是:

ˆyAF M (x) = pT

n
X

n
X

aij(vi (cid:12) vj)xixj,

j=i+1
其中 aij = sof tmax(hT M LP (vi (cid:12) vj)).

i=1

(5.28)

 注意力 权重 aij 可以 是 使用了  interpret  重要性 的
每个 秒-顺序 交互 用于  预测. 它 是 straightforward
 进一步 利用  strengths 的 NFM 在 高-顺序 交互
建模 和 AFM 在 秒-顺序 交互 建模, 通过 appending
 MLP 到  注意力-基于 池化 层. 该 自然地 导致 
深度 AFM, 该 具有 更好 表示 能力 和 可以 yield 更好
性能.  最近 工作 (Tao 等人, 2019) 提出  高-顺序
attentive FM (HoAFM), 该 具有 线性 复杂度 在 顺序 大小.

正交   工作 的 FM, Lian 等人 (2018) 提出 压缩 交互 网络 (CIN), 该 显式地 模型 高-顺序
特征 interactions 在  递归 方式. Let  embeddings 的 non-零
输入 特征 是  矩阵 V0 ∈ Rn×D, 其中 n 是  数量 的
非零 特征 和 D 是  嵌入 大小;  i-th row 在 V0 是 
嵌入 向量 的  i-th 非零 特征: V0
i,∗ = vi. Let  输出
的  k-th 层 在 CIN 是  矩阵 Xk ∈ RHk×D, 其中 Hk 表示
 数量 的 嵌入 vectors 在  k-th 层 和 是  架构
选择  specify (注意 该 H0 = n).  递归 deﬁnition 的 CIN
该 可以 捕获 高-顺序 特征 交互 是 deﬁned 作为:

Vk

h,∗ =

Hk−1
X

n
X

i=1

j=1

Wk,h

ij (Vk−1

i,∗ (cid:12) V0

j,∗),

(5.29)

其中 1 ≤ h ≤ Hk, Wk,h ∈ RHk−1×n 是  参数 矩阵 用于 
h-th 特征 向量. 作为 Vk 是 derived 通过  交互 之间 Vk−1
和 V0,  顺序 的 特征 interactions 增加 用  层 深度.
假设 该  模型 stacks K 这样的 层,  ﬁnal 预测
是 基于 在  输出 矩阵 Xk 的 所有 K 层, 该 uniﬁes 
特征 interactions 向上  K orders. 注意 该  时间复杂度 的
高-顺序 交互 建模 增加 linearly 用  数量 的

5.2. 基于匹配函数学习的匹配

139

orders, 该 是 smaller 比 该 的 高-顺序 FM. 然而, CIN
介绍 更多 参数  训练 — 用于 层 k, 它 具有 Hk trainable
权重 矩阵 的 大小 Hk−1 × n, 该 amounts   参数 张量
的 大小 Hk × Hk−1 × n.

图 5.20: 模型 架构 的 AFM.

组合 的 显式 和 隐式 交互 建模

作为 隐式 交互 建模 和 显式 交互 建模 工作
在 diﬀerent ways, 集成中 它们 到  uniﬁed 框架 具有 
potential  改进  性能. 在  最近 文献, 
报告了 最好 performances 是 获得了 通过 混合 模型 该 组合
多个 交互 模型 (Guo 等人, 2017; Lian 等人, 2018). 我们
brieﬂy 综述  集成 方法 在 该 小节.

广泛&深度 (Cheng 等人, 2016) ensembles  线性 回归 模型
该 利用 手动地 构建了 交叉 特征 ( 广泛 部分) 和
 MLP 模型 该 隐式地 使用  interactions 的 特征 (
深度 部分).  广泛 部分 是  memorize  seen 交叉 特征, 和
 深度 部分 是  泛化  unseen 交叉 特征. Let x 是 
原始 输入 特征 和 φ(x) 是  构建了 交叉 特征. 则
 模型 用于 预测中  点击率 是:

p(点击|x) = σ(wT

广泛[x, φ(x)] + wT

deepa(L) + b),

(5.30)

其中 σ(·) 是  Sigmoid函数  输出  概率 值, wwide

010100.20……!"·$"!%·$%!&·$&EmbeddingLayerSparseInput$'$"$($%$)$&$*!"⊙!%$"$%!"⊙!&$"$&!%⊙!&$%$&PairwiseInteractionLayer,"%,"&,%&∑./0!/⊙!0$/$0AttentionBasedPoolingPredictionScoreAttentionNet12140

推荐中的深度匹配模型

表示  weights 的  广泛 部分, wdeep 表示  weights 的 
深度 部分, (L) 表示  最后 隐藏层 的  MLP 模型, 和 b
是  偏置 术语.

 工作 的 广泛&深度 inspired 许多 后来 工作  采用 相似
ensembles, 但 在 diﬀerent 模型. 用于 例子, DeepFM (Guo 等人,
2017) ensembles FM 和 MLP 通过: ˆyDeepF M = σ(ˆyF M + ˆyM LP ), 其中
ˆyF M 和 ˆyM LP 表示  predictions 的 FM 和 MLP, 分别地. 
FM 模型 学习 秒-顺序 特征 interactions 显式地, 和 
MLP 模型 学习 高-顺序 特征 interactions 隐式地. 在 添加,
DeepFM shares  嵌入 层 的 FM 和 MLP  减少 模型
参数. xDeepFM (Lian 等人, 2018) 进一步 ensembles DeepFM
用 CIN, 该 显式地 模型 高-顺序 特征 interactions. 那里
是 其他 集成 模型. 我们 进行 不 描述 它们 这里 due 
空间 limitations.  通用 观察 是 该 组合中 模型
该 account 用于 diﬀerent 类型 的 interactions 通常 yields 更好
performances.

5.3 扩展阅读

推荐 remains  是  重要 和 hot 主题 在 信息检索 和 数据 挖掘. 新 技术 是 持续地 开发了,
和 新 方法 是 evolving 快. 这里 我们 给出 更多 references 用于
扩展阅读.

5.3.1 Papers

用 regard  学习 表示 从 用户 序列 interactions,
一些 最近 工作 (Sun 等人, 2019; Yuan 等人, 2020) 论证 该 用户
behaviors 可以 不 是 严格地 formalized 作为 sequences 的 interactions.
该 是,  序列 的 interactions 进行 不 necessarily encode 强
语义 作为  句子 的 词, 和  recorded 序列 仅 reﬂects
 用户’s 一 选择 和 其他 choices 可以 也 是 可能. 作为 
结果,  “左--正确” 训练 范式, 这样的 作为 RNN, 可以 不 是
最优 因为  未来 interactions 是 ignored 在  预测 的
interactions. 这样的 信息 是 也 indicative 的 用户’s 偏好 和
应当 是 leveraged.  关键 问题 在 utilization 的 未来 信息

5.3. 扩展阅读

141

是 如何  避免 信息 leakage.  解决 该 挑战, (Sun
等人, 2019; Yuan 等人, 2020) 采用  ﬁll-在--blank 训练
范式 inspired 通过 BERT, 该 随机地 masks 一些 interactions
在  编码器 用  aim 的 预测中  masked interactions 或
通过  编码器 itself (Sun 等人, 2019) 或 通过  additional 解码器 (Yuan
等人, 2020).

用于 学习 表示 从 多-模态 content, 一些 最近
工作 exploits  交互 图  propagate multimedia 特征
在  图 用 图 卷积 网络 (Ying 等人, 2018; Wei
等人, 2019). 在 该 方式,  multimedia 特征 是 smoothed 和
成为 更多 有用 用于 推荐. 用于 例子,  多-模态
GCN (MMGCN) (Wei 等人, 2019) 构建  模态-aware GCN
在  用户-物品 图 用 特征 的 每个 模态 (visual, textual,
和 acoustic), 和 fuses  输出 的 每个 模态  get  ﬁnal
表示 的  micro-视频. 用 regard  表示学习
在 图 数据, 除之外  用户-物品 图 和 知识图谱 作为 我们
具有 描述, social 网络 (Wu 等人, 2019b) 和 会话 图 (Wu
等人, 2019c; Qiu 等人, 2019) 是 也 使用了 用于 学习 的 GCN. 作为
 图 提供  形式 方式 的 描述中 diﬀerent 类型 的 entities
和 其 relations, 匹配 基于 在  异质 图 是 
有前景 解决方案 用于 推荐 系统 在 diﬀerent 应用.
作为 用于 学习 动态 表示 的 用户 用 respect  diﬀerent
物品, 注意力 网络 是 designed  学习 用户’s speciﬁc preferences
 diﬀerent aspects 的 物品 通过 leveraging 综述 和 images (Cheng
等人, 2018; Liu 等人, 2019b).

Due   多样性 的 推荐 场景 在 实践, 研究ers 开发 神经 推荐 模型 从 diﬀerent perspectives. 用于 例子, (Gao 等人, 2019b) 模型 多个 cascading 用户
behaviors 如 点击, add--cart, 和 purchase, 在  序列 多-任务
框架. Li 等人 (2019) 开发  capsule 网络  识别 sentiment 从 用户 综述 用于 推荐. Xin 等人 (2019b) 取
到 account relations 的 多个 物品 (e.g., 相同 类别, 共享
属性, etc.) 用于 物品-基于 CF. Gao 等人 (2019) 开发  privacypreserving 方法 用于 交叉-领域 推荐 通过 transferring
用户 embeddings 而 比 raw 行为 数据. Pan 等人 (2019) tailor
 学习 的 嵌入 参数 用于 物品 cold-开始 recommenda-

142

推荐中的深度匹配模型

tion 通过 meta-学习.

所有  以上-讨论 神经网络 模型 是 oﬄine 推荐 方法, 该 利用  oﬄine 历史 数据  estimate 用户
偏好. 另一个 thriving 领域 是 在线 推荐, 用于 该
 bandit-基于 方法 是 prevalent (Li 等人, 2010; Wu 等人, 2016;
Zhang 等人, 2020). 它们 aim  pursue  利用-探索
trade-oﬀ 当 interacting 用 用户 在 在线 recommendations. 二
共同 类型 的 bandit 方法 是 Upper Conﬁdence Bound (UCB)-
基于 和 Thompson Sampling (TS)-基于, 和 两者 方法 具有 pros
和 cons. 除之外 interacting 用 用户 用 物品 recommendations,
最近 工作 (Zhang 等人, 2020) 考虑 asking 属性 偏好,
该 可以 发现  大多数 相关 物品 更多 eﬀectively. 神经 网络
可以 serve 作为  利用 组件 用于  bandit-基于 方法,
和 更多 investigations remain  是 完成 朝向 组合中 oﬄine
深度 模型 用 在线 探索 strategies.

5.3.2 基准 Datasets

那里 是  数量 的 基准 datasets 可用 用于 训练 和
测试 推荐 模型 在 diﬀerent 场景. 用于 实例, 
MovieLens 集合1, Amazon 乘积 集合2 (他 和 McAuley,
2016b), 和 Gowalla3 (Liang 等人, 2016) 是 基准 datasets
该 由组成 的 用户-物品交互 数据. Yelp4, Ciao5, 和 Epinions6 (Richardson 等人, 2003) 是 datasets 该 此外 包括
social relations 之中 用户 该 是 有用 用于 social 推荐. Yoochoose7 和 Diginetica8 包含 streams 的 用户 clicks 在
e-commerce, 和 因此 是 适合 用于 会话-基于 (序列) rec-

1https://grouplens.org/datasets/movielens
2http://jmcauley.ucsd.edu/数据/amazon/
3https://snap.stanford.edu/数据/loc-gowalla.html
4https://www.yelp.com/数据集/挑战
5http://www.ciao.co.uk
6https://snap.stanford.edu/数据/soc-Epinions1.html
7http://2015.recsyschallenge.com/challege.html
8http://cikm2016.cs.iupui.edu/cikm-cup

5.3. 扩展阅读

143

ommendation. Criteo9, Avazu10, 和 Frappe11 (Baltrunas 等人, 2015)
是 comprised 的 上下文 信息 的 interactions, 和 是 广泛
使用了 在 CTR 预测 和 特征-基于 推荐. 此外,
作为 Amazon, Yelp, 和 TripAdvisor12 (Wang 等人, 2018c) 提供 rich
综述 和 comments 在 物品, 它们 是 广泛 利用了 在 综述-基于
推荐 模型. 除之外, 那里 是 若干 datasets 呈现中 
知识图谱 用于 推荐, 这样的 作为 KB4Rec13 (Zhao 等人,
2019) 和 KGAT14 (Wang 等人, 2019).

5.3.3 Open 来源 Packages

若干 open-来源 packages 或 libraries 用于 推荐 是 publicly 可用, 用  aim 的 facilitating 相关 研究. Microsoft
Recommenders15 oﬀers tens 的 例子 模型 用于 构建中 推荐 系统. NeuRec16 是  open-来源 library 该 包括 
大 数量 的 最先进的 推荐 模型, ranging 从
协同 过滤ing 和 social 推荐  序列推荐. 它 是 worthwhile highlighting 该 NeuRec 是  modular
框架 在 该  模型 可以 是 构建了  reusable 模块 用
标准 interfaces. 因此 它 允许 用户  构建 其 own 模型 easily.
Simiarly, OpenRec17 是  open-来源 项目 该 包含 若干
推荐 方法.

9http://labs.criteo.com/2014/02/kaggle-display-advertising-挑战-数据集
10https://www.kaggle.com/c/avazu-ctr-预测/数据
11http://baltrunas.info/研究-menu/frappe
12https://github.com/xiangwang1223/树_增强_嵌入_模型
13https://github.com/RUCDM/KB4Rec
14https://github.com/xiangwang1223/知识_图_注意力_网络
15https://github.com/microsoft/recommenders
16https://github.com/NExTplusplus/NeuRec
17https://github.com/ylongqi/openrec

6

结论与未来方向

6.1 综述总结

如何  桥接  语义鸿沟 之间 二 匹配 entities 是 
大多数 基本 和 有挑战 问题 在 搜索与推荐.
在 搜索,  searchers 和  authors 的 文档 可以 使用 diﬀerent 表达  表示  相同 meanings, resulting 在  大多数
不期望 outcomes 在 该 相关 文档 exist 但 cannot 是
发现. 在 推荐,  用户 和  物品 belong  diﬀerent
类型 的 entities 和 是 表示了 通过 diﬀerent superﬁcial 特征,
使 它 diﬃcult  进行 匹配 之间  特征 和 因此
oﬀer satisfactory recommendations 在 物品  用户.  桥接 
语义鸿沟, 研究ers 在 两者 搜索与推荐 具有 提出  构建 和 利用 匹配 模型 用 机器学习
技术.

在 最近 年, 深度学习 具有 是 应用  搜索与推荐, 和 great 成功 具有 是 实现了. 在 该 综述, 我们
具有 首先 介绍  uniﬁed 视角 在 匹配 在 搜索与推荐. 之下 该 视角, 我们 具有 则 分类  学习
解决方案  查询-文档匹配 在 搜索 和 用户-物品匹配
在 推荐 到 二 类型: 方法 的 表示学习

144

6.2. 其他任务中的匹配

145

和 方法 的 匹配函数学习. 之后 该, 代表性
传统 匹配 方法, 以及 深度匹配 方法, 具有
是 解释了 用 细节. 实验的 results, benchmarks, 和
software packages 具有 也 是 介绍.

 uniﬁed 视角 的 匹配 提供  新 means  比较 和
分析  机器学习 方法, 特别是 深度学习
方法, 开发了 用于 搜索与推荐. 虽然 现有 匹配 模型 用于 搜索 和 用于 推荐 是 开发了
用于 diﬀerent purposes 之内 diﬀerent communities (e.g., SIGIR 和
RecSys), 它们 bear 相似 设计 principles 和 模型 properties. 该
综述 可以 是 beneﬁcial 用于 人们 在 communities 用 其 uniﬁed 视角.
在 fact,  边界 之间 搜索与推荐 成为
blurry, 和 那里 emerges  趋势  unify  二 范式 (Zhang
等人, 2018; Schedl 等人, 2018).  uniﬁed 视角 提供  新 angle
 devise 新颖 模型 用于 搜索与推荐.

一 可以 see 该 面向匹配的深度学习 具有 使 和 是
使 signiﬁcant 进展 在 搜索与推荐. 一 可以
也 foresee 该 它 具有  potential  使 影响 在 相似 问题
在 其他 ﬁelds, 包括中 在线 advertising, 问答, 图像
标注, 和 drug 设计.

6.2 其他任务中的匹配

语义匹配 是  基本 问题 在 其他 任务 超出 搜索
和 推荐. 由于 匹配 是 进行了 之间 二 集合
的 对象, 它 可以 是 分类 作为 文本 匹配 和 实体 匹配.
在 文本 匹配, 那里 exists  顺序 之间  元素 之内
每个 对象 (e.g., 词 在  句子). 查询-文档匹配 是 
典型 例子 的 文本 匹配. 在 实体 匹配, 那里 是 无 顺序
exits 之间  对象. 用户-物品匹配 在 推荐 是
 例子 的 实体 匹配. 其他 匹配 任务 具有 也 是
研究了. 我们 列表 一些 的 它们 这里.

Paraphrase 检测 Determining 是否 二 句子 是 用
 相同 含义 是  重要 主题 的 语义匹配 在
自然语言处理.  匹配 是 进行了 在 

146

结论与未来方向

语义层面, 和  学会了 匹配函数 是 symmetric.

社区 QA 给定  问题,  目标 是  发现 问题 用
 相同 含义 从  知识-基于 在 社区 QA.
 任务 是 相似  paraphrase 检测, 同时  二 句子
是 问题.  匹配 之间 二 问题 是 进行了
在  语义层面.

文本 entailment 文本 entailment 指代   问题 的 determining
implication 或 none-implication 关系 之间 二 statements.
尽管 相似, entailment 是 diﬀerent 从 paraphrase 检测
在 该 它 聚焦 在 determining  logical 关系 之间 二
文本.  匹配 应当 也 是 进行了 在  语义
水平, 和  匹配函数 是 不 symmetric.

检索-基于 对话 一 关键 问题 在 检索-基于 对话
是  发现  大多数 适合 响应 给定 话语 在 
上下文 的  会话.  响应 是 通常  句子
同时  话语 可以 是, 用于 例子, 一 单一 话语 或
所有 话语 在  上下文 (在 多-轮次 dialog). 它 是 obvious
该  匹配 是 进行了 之间 文本 在  语义
水平.

在线 advertising 在 搜索 ads, 如何  匹配  用户’s 搜索 查询
 advertisers’ 关键词 大大 ﬀects  概率 该 
用户 将 see 和 点击  ads. 在 contextual ads, 匹配 是
进行了 之间  关键词 和  contents 的 webpages.
在 两者 案例, 语义匹配 是 有帮助 在 choosing 正确 ads
和 构建  正确 顺序 通过 该  ads 是 displayed.

6.3 开放问题与未来方向

那里 是 许多 open 问题 用 regard  深度匹配 用于 搜索
和 推荐. 这里, 我们 仅 列表 一些 的 它们.

1. Lack 的 训练数据 (i.e., 监督学习 数据) 是 仍然 一 的
 关键 挑战. 在 对比, 深度匹配 模型 需求  大

6.3. 开放问题与未来方向

147

量 的 数据  训练. 如何  利用 无监督学习,
弱-监督学习, 半-监督学习, 和 distant
监督学习 技术  deal 用  问题 是 
重要 问题.

2.  大 fraction 的 深度匹配 模型 是 训练了 用 点击
数据. 现有 研究 显示 该 直接地 使用中 点击 数据 作为 训练 signals 经常 yields 次优 results. 在 学习  排序, 
counterfactual 推理 框架 是 提出  推导 unbiased
学习  排序 模型 (Joachims 等人, 2017). 如何  overcome
 偏置 问题 在 深度匹配 是  exciting 未来 方向.

3.  学习 的 现有 深度匹配 模型 是 purely datadriven. 有时, rich 先验 知识 进行 exist (e.g., 领域
知识, 知识-基于, 匹配 rules), 和  使用 的 它
应当 是 有帮助 在 改进中  performances 的 匹配. 如何
 集成 先验 知识 到 匹配 模型 是  重要
方向  探索.

4. 匹配 模型 是 通常 学会了 用 一 单一 目标,
i.e.,“相似度”. 那里 可以 需求  利用 多个 objectives 在
学习 (e.g., 归纳 能力, fairness) 根据 应用.
如何  add 其他 criteria 到  学习 的 匹配 模型 是
另一个 重要 问题  研究.

5.   大 extent, 当前 深度匹配 模型 是 black boxes.
在 真实 搜索 和 推荐 系统, 然而, 它 是 经常
需要了 该  匹配 模型 不 仅 实现 高 准确率,
但 也 给出 直观 explanations 的  results. 这样的 可解释性 是 有帮助  改进  透明性, persuasiveness, 和
trustworthiness 的  系统. 如何  创建  解释
能力 在 深度匹配 模型 是 仍然  open 问题.

6. 大多数 深度匹配 模型 仅 学习 correlations 从  数据.
然而, 相关 是 不 因果性, 和 它 falls 短 在 revealing
 reasons 在之后  数据 (e.g.,  reasons 该  用户 prefers
 物品 超过 另一个 一).  增强  匹配模型 用
因果 推理 能力, 我们 需求  介绍  mechanisms 的

148

结论与未来方向

干预 和 counterfactual 推理 到  模型 (Pearl,
2019). 此外,  collected 数据 是 通常 biased 通过 许多
factors, 如  位置 偏置, exposure 偏置, 和 因此 在. 它 是 
新兴 方向  开发 因果 方法 用于 匹配, 该
是 鲁棒   各种 数据 偏置 和 able  reveal  reasons
在之后  数据. 章-结论 2020-05-28 19.21.04

7. 在 搜索 和 推荐 系统,  processes 的 匹配
和 排序 是 通常 separated: 首先 匹配 和 则 排序.
因此,  results 的 匹配 是 自然地 使用了 作为 特征
的 排序. 然而,  分离 的 排序 和 匹配 可以
不 是 必要 有时. 一 自然 问题 是 是否 它
是 可能  构建  端--端 系统 在 该  匹配
和 排序 模型 是 联合地 学会了.

8. 搜索 和 推荐 系统 是 成为中 更多 和 更多
交互, 该 可以 帮助 用户  发现 相关 或 interesting
信息 在  探索性 方式. 用于 例子, 一些 搜索
engines let  用户  reﬁne  查询 之后 checking  初始 results. 类似地, 一些 推荐 系统 recommend
物品 基于 在 什么  用户 具有 chosen, 或 通过 asking 用户
什么 种类 的 物品 attributes 它们 prefer (Lei 等人, 2020). 因此, 如何  结构  用户-系统 interactions 和 进行
查询-文档 (或 用户-物品) 匹配 在  交互 和
conversational 场景 是  重要 和 interesting 研究
主题.

Acknowledgements

我们 thank  编辑 和  三 anonymous reviewers 用于 其
valuable comments  改进  manuscript. 我们 thank Dr. Wang
Xiang 和 Dr. Yuan Fajie 用于 提供中 材料 用于  writing 的 
书.  工作 是 支持了 通过  National 自然 科学 基础
的 China (61872338, 61972372, U19207, 61832017), Beijing Academy
的 Artiﬁcial 智能 (BAAI2019ZD0305), 和 Beijing 杰出的
Young Scientist 程序 (BJJWZYJH012019100020098).

149

References

Adomavicius, G. 和 . Tuzhilin. 2005. “朝向  下一个 生成
的 推荐 系统:  综述 的  陈述-的--Art 和
可能 Extensions”. IEEE Transactions 在 知识 和 数据
工程. 17(6): 734–749. issn: 1041-4347. doi: 10.1109/TKDE.
2005.99. url: https://doi.org/10.1109/TKDE.2005.99.

Ai, Q., K. Bi, J. Guo, 和 W. B. Croft. 2018. “学习  深度 列表级
上下文 模型 用于 排序 Reﬁnement”. 在:  41st 国际
ACM SIGIR 会议 在 研究 & 开发 在 信息检索. SIGIR ’18. Ann Arbor, MI, USA: 协会 用于
计算中 Machinery. 135–144. isbn: 9781450356572. doi: 10.1145/
3209978.3209985. url: https://doi.org/10.1145/3209978.3209985.
Andrew, G., R. Arora, J. Bilmes, 和 K. Livescu. 2013. “深度 Canonical
相关 分析”. 在: 会议论文集 的  30th 国际
会议 在 国际 会议 在 机器学习 -
卷 28. ICML’13. Atlanta, GA, USA: JMLR.org. III-1247–III-
1255. url: http://dl.acm.org/引用.cfm?id=3042817.3043076.
Ba, J. L., J. R. Kiros, 和 G. E. Hinton. 2016. “层归一化”.
CoRR. abs/1607.06450. arXiv: 1607.06450. url: http://arxiv.org/
abs/1607.06450.

150

References

151

Bahdanau, D., K. Cho, 和 Y. Bengio. 2015. “神经 机器 翻译 通过 联合地 学习  Align 和 Translate”. 在: 3rd 国际 会议 在 学习 表示. url: http :
//arxiv.org/abs/1409.0473.

Bai, B., J. Weston, D. Grangier, R. Collobert, K. Sadamasa, Y. Qi, O.
Chapelle, 和 K. Weinberger. 2009. “监督语义索引”.
在: 会议论文集 的  18th ACM 会议 在 信息 和
知识 管理. CIKM ’09. Hong Kong, China: ACM. 187–
196. isbn: 978-1-60558-512-3. doi: 10.1145/1645953.1645979. url:
http://doi.acm.org/10.1145/1645953.1645979.

Bai, B., J. Weston, D. Grangier, R. Collobert, K. Sadamasa, Y. Qi, O.
Chapelle, 和 K. Weinberger. 2010. “学习  排序 用 ( Lot
的) 词 特征”. Inf. Retr. 13(3): 291–314. issn: 1386-4564. doi:
10.1007/s10791-009-9117-9. url: http://dx.doi.org/10.1007/s10791-
009-9117-9.

Bai, T., J.-R. Wen, J. Zhang, 和 W. X. Zhao. 2017. “ 神经协同过滤 模型 用 交互-基于 Neighborhood”. 在:
会议论文集 的  2017 ACM 在 会议 在 信息 和
知识 管理. CIKM ’17. Singapore, Singapore: ACM.
1979–1982. isbn: 978-1-4503-4918-5. doi: 10.1145/3132847.3133083.
url: http://doi.acm.org/10.1145/3132847.3133083.

Balaneshin-kordan, S. 和 . Kotov. 2018. “深度 神经 架构
用于 多-模态 检索 基于 在 联合 嵌入 空间 用于 文本
和 Images”. 在: 会议论文集 的  Eleventh ACM 国际
会议 在 网页搜索 和 数据 挖掘. WSDM ’18. Marina
Del Rey, CA, USA: ACM. 28–36. isbn: 978-1-4503-5581-0. doi:
10 . 1145 / 3159652 . 3159735. url: http : / / doi . acm . org / 10 . 1145 /
3159652.3159735.

Baltrunas, L., K. Church, . Karatzoglou, 和 N. Oliver. 2015. “Frappe:
理解  用法 和 感知 的 Mobile App Recommendations 在--Wild”. CoRR. abs/1505.03014. arXiv: 1505.03014.
url: http://arxiv.org/abs/1505.03014.

Bast, H., B. Björn, 和 E. Haussmann. 2016. “语义 搜索 在 文本
和 知识 基于”. 发现. Trends Inf. Retr. 10(2-3): 119–271.
issn: 1554-0669. doi: 10.1561/1500000032. url: https://doi.org/10.
1561/1500000032.

152

References

Batmaz, Z., . Yurekli, . Bilge, 和 C. Kaleli. 2019. “ 综述 在
深度学习 用于 推荐 系统: 挑战 和 remedies”.
Artiﬁcial 智能 综述. 52(1): 1–37. doi: 10.1007/s10462-018-
9654-y. url: https://doi.org/10.1007/s10462-018-9654-y.

Belkin, N. J. 和 W. B. Croft. 1992. “信息 过滤 和 信息检索: 二 Sides 的  相同 Coin?” Commun. ACM.
35(12): 29–38. issn: 0001-0782. doi: 10.1145/138859.138861. url:
http://doi.acm.org/10.1145/138859.138861.

Bello, I., S. Kulkarni, S. Jain, C. Boutilier, E. H. Chi, E. Eban, X.
Luo, . Mackey, 和 O. Meshi. 2018. “Seq2Slate: Re-排序 和
Slate 优化 用 RNNs”. CoRR. abs/1810.02019. arXiv:
1810.02019. url: http://arxiv.org/abs/1810.02019.

Bendersky, M., W. B. Croft, 和 D. . Smith. 2011. “联合 标注
的 搜索 查询”. 在: 会议论文集 的  49th Annual Meeting 的
 协会 用于 计算 语言学: 人类 语言
Technologies - 卷 1. HLT ’11. Portland, Oregon: 协会
用于 计算 语言学. 102–111. isbn: 978-1-932432-87-9.
url: http://dl.acm.org/引用.cfm?id=2002472.2002486.

Berg, R. van den, T. N. Kipf, 和 M. Welling. 2017. “图 卷积 矩阵 完成”. CoRR. abs/1706.02263. arXiv: 1706.
02263. url: http://arxiv.org/abs/1706.02263.

Berger, . 和 J. Laﬀerty. 1999. “信息检索 作为 统计
翻译”. 在: 会议论文集 的  22Nd Annual 国际 ACM
SIGIR 会议 在 研究 和 开发 在 信息
检索. SIGIR ’99. Berkeley, California, USA: ACM. 222–229.
isbn: 1-58113-096-1. doi: 10.1145/312624.312681. url: http://doi.
acm.org/10.1145/312624.312681.

Bergsma, S. 和 Q. I. Wang. 2007. “学习 Noun 短语 查询
分割”. 在: 会议论文集 的  2007 联合 会议 在
经验的 方法 在 自然语言处理 和 计算 自然 语言 学习 (EMNLP-CoNLL). Prague, Czech
Republic: 协会 用于 计算 语言学. 819–826. url:
https://www.aclweb.org/anthology/D07-1086.

References

153

Beutel, ., P. Covington, S. Jain, C. Xu, J. Li, V. Gatto, 和 E. H.
Chi. 2018. “潜在 交叉: 使 使用 的 上下文 在 Recurrent
推荐 系统”. 在: 会议论文集 的  Eleventh ACM 国际 会议 在 网页搜索 和 数据 挖掘. WSDM ’18.
Marina Del Rey, CA, USA: ACM. 46–54. isbn: 978-1-4503-5581-0.
doi: 10.1145/3159652.3159727. url: http://doi.acm.org/10.1145/
3159652.3159727.

Bowman, S. R., G. Angeli, C. Potts, 和 C. D. Manning. 2015. “
大 annotated 语料库 用于 学习 自然 语言 推理”.
在: 会议论文集 的  2015 会议 在 经验的 方法 在
自然语言处理. Lisbon, Portugal: 协会 用于 计算 语言学. 632–642. doi: 10.18653/v1/D15-1075. url:
https://www.aclweb.org/anthology/D15-1075.

Brill, E. 和 R. C. Moore. 2000. “ 改进了 误差 模型 用于 Noisy
Channel Spelling 修正”. 在: 会议论文集 的  38th Annual
Meeting 在 协会 用于 计算 语言学. ACL ’00.
Hong Kong: 协会 用于 计算 语言学. 286–293.
doi: 10.3115/1075218.1075255. url: https://doi.org/10.3115/
1075218.1075255.

Burges, C. J. 2010. “从 RankNet  LambdaRank  LambdaMART:
 概述”. Tech. rep. 无. MSR-TR-2010-82. url: https : / /
www.microsoft.com/en-我们/研究/出版物/从-ranknet-tolambdarank--lambdamart--概述/.

Cao, Y., J. Xu, T.-Y. Liu, H. Li, Y. Huang, 和 H.-W. Hon. 2006.
“Adapting 排序 SVM  文档检索”. 在: 会议论文集
的  29th Annual 国际 ACM SIGIR 会议 在 研究 和 开发 在 信息检索. SIGIR ’06. Seattle, Washington, USA: ACM. 186–193. isbn: 1-59593-369-7. doi:
10 . 1145 / 1148170 . 1148205. url: http : / / doi . acm . org / 10 . 1145 /
1148170.1148205.

Chen, C., M. Zhang, Y. Liu, 和 S. Ma. 2018. “神经 Attentional
Rating 回归 用 综述-水平 Explanations”. 在: 会议论文集
的  2018 世界 广泛 Web 会议. WWW ’18. Lyon, France.
1583–1592. isbn: 978-1-4503-5639-8. doi: 10.1145/3178876.3186070.
url: https://doi.org/10.1145/3178876.3186070.

154

References

Chen, H., F. X. Han, D. Niu, D. Liu, K. Lai, C. Wu, 和 Y. Xu. 2018b.
“MIX: 多-Channel 信息 Crossing 用于 文本 匹配”. 在:
会议论文集 的  24th ACM SIGKDD 国际 会议 在
知识 发现 & 数据 挖掘. KDD ’18. London, United
Kingdom: ACM. 110–119. isbn: 978-1-4503-5552-0. doi: 10.1145/
3219819 . 3219928. url: http : / / doi . acm . org / 10 . 1145 / 3219819 .
3219928.

Chen, J., H. Zhang, X. 他, L. Nie, W. Liu, 和 T.-S. Chua. 2017.
“Attentive 协同过滤: Multimedia 推荐
用 物品- 和 组件-水平 注意力”. 在: 会议论文集 的
 40th 国际 ACM SIGIR 会议 在 研究 和
开发 在 信息检索. SIGIR ’17. Shinjuku, Tokyo,
Japan: ACM. 335–344. isbn: 978-1-4503-5022-8. doi: 10 . 1145 /
3077136 . 3080797. url: http : / / doi . acm . org / 10 . 1145 / 3077136 .
3080797.

Chen, Q., X. Zhu, Z.-H. Ling, S. Wei, H. Jiang, 和 D. Inkpen. 2017b.
“增强 LSTM 用于 自然 语言 推理”. 在: 会议论文集
的  55th Annual Meeting 的  协会 用于 计算 语言学 (卷 1: 长 Papers). Vancouver, Canada: 协会
用于 计算 语言学. 1657–1668. doi: 10.18653/v1/P17-
1152. url: https://www.aclweb.org/anthology/P17-1152.

Cheng, H.-T., L. Koc, J. Harmsen, T. Shaked, T. Chandra, H. Aradhye,
G. Anderson, G. Corrado, W. Chai, M. Ispir, R. Anil, Z. Haque, L.
Hong, V. Jain, X. Liu, 和 H. Shah. 2016. “广泛 & 深度学习
用于 推荐 系统”. 在: 会议论文集 的  1st 研讨会 在
深度学习 用于 推荐 系统. DLRS 2016. Boston, MA,
USA: ACM. 7–10. isbn: 978-1-4503-4795-2. doi: 10.1145/2988450.
2988454. url: http://doi.acm.org/10.1145/2988450.2988454.

Cheng, Z., Y. Ding, X. 他, L. Zhu, X. Song, 和 M. S. Kankanhalli.
2018. “3NCF:  自适应 方面 注意力 模型 用于 Rating
预测”. 在: 会议论文集 的  二十-第七 国际
联合 会议 在 Artiﬁcial 智能. 3748–3754. doi: 10.24963/
ijcai.2018/521. url: https://doi.org/10.24963/ijcai.2018/521.

References

155

Cohen, D., L. Yang, 和 W. B. Croft. 2018. “WikiPassageQA:  基准 集合 用于 研究 在 Non-factoid 答案 段落检索”. 在:  41st 国际 ACM SIGIR 会议 在
研究 & 开发 在 信息检索. SIGIR ’18. Ann
Arbor, MI, USA: ACM. 1165–1168. isbn: 978-1-4503-5657-2. doi:
10 . 1145 / 3209978 . 3210118. url: http : / / doi . acm . org / 10 . 1145 /
3209978.3210118.

Costa, . 和 F. Roda. 2011. “推荐 系统 通过 Means 的
信息检索”. 在: 会议论文集 的  国际 会议 在 Web 智能, 挖掘 和 语义. WIMS ’11.
Sogndal, Norway: ACM. 57:1–57:5. isbn: 978-1-4503-0148-0. doi:
10 . 1145 / 1988688 . 1988755. url: http : / / doi . acm . org / 10 . 1145 /
1988688.1988755.

Covington, P., J. Adams, 和 E. Sargin. 2016. “深度 神经 网络
用于 youtube recommendations”. 在: 会议论文集 的  10th ACM
会议 在 推荐 系统. 191–198.

Croft, W. B., D. Metzler, 和 T. Strohman. 2009. 搜索 Engines: 信息检索 在 实践. 1st. USA: Addison-Wesley Publishing
公司. I–XXV, 1–524. isbn: 0136072240, 9780136072249.
Dai, Z., C. Xiong, J. Callan, 和 Z. Liu. 2018. “卷积 神经
网络 用于 Soft-匹配 N-Grams 在 Ad-hoc 搜索”. 在: 会议论文集 的  Eleventh ACM 国际 会议 在 网页搜索
和 数据 挖掘. WSDM ’18. Marina Del Rey, CA, USA: ACM.
126–134. isbn: 978-1-4503-5581-0. doi: 10.1145/3159652.3159659.
url: http://doi.acm.org/10.1145/3159652.3159659.

Dehghani, M., H. Zamani, . S每个n, J. Kamps, 和 W. B. Croft.
2017. “神经 排序 模型 用 弱监督”. 在: 会议论文集 的  40th 国际 ACM SIGIR 会议 在 研究 和 开发 在 信息检索. SIGIR ’17. Shinjuku, Tokyo, Japan: 协会 用于 计算中 Machinery. 65–74.
isbn: 9781450350228. doi: 10.1145/3077136.3080832. url: https:
//doi.org/10.1145/3077136.3080832.

156

References

Devlin, J., M.-W. Chang, K. Lee, 和 K. Toutanova. 2019. “BERT:
Pre-训练 的 深度 Bidirectional Transformers 用于 语言 理解”. 在: 会议论文集 的  2019 会议 的  North
American 章 的  协会 用于 计算 语言学:
人类 语言 Technologies, 卷 1 (长 和 短 Papers).
Minneapolis, Minnesota: 协会 用于 计算 语言学.
4171–4186. url: https://www.aclweb.org/anthology/N19-1423.
Dolan, B. 和 C. Brockett. 2005. “自动地 构建  语料库
的 Sentential Paraphrases”. 在: 第三 国际 研讨会 在
复述 (IWP2005). Asia Federation 的 自然 语言
Processing. url: https : / / www . microsoft . com / en - 我们 / 研究 /
出版物 / 自动地 - 构建 -  - 语料库 - 的 - sentential -
paraphrases/.

Eisenschtat, . 和 L. Wolf. 2017. “Linking 图像 和 文本 用 2-方式
Nets”. 在: 2017 IEEE 会议 在 计算机 视觉 和 模式
识别 (CVPR). 1855–1865. doi: 10.1109/CVPR.2017.201.
Eksombatchai, C., P. Jindal, J. Z. Liu, Y. Liu, R. Sharma, C. Sugnet, M.
Ulrich, 和 J. Leskovec. 2018. “Pixie:  系统 用于 Recommending
3+ 十亿 物品  200+ 百万 用户 在 真实-时间”. 在: 会议论文集 的  2018 世界 广泛 Web 会议 在 世界 广泛
Web, WWW 2018, Lyon, France, April 23-27, 2018. 1775–1784. doi:
10.1145/3178876.3186183. url: https://doi.org/10.1145/3178876.
3186183.

Elkahky, . M., Y. Song, 和 X. 他. 2015. “ 多-视角 深度学习
方法 用于 交叉 领域 用户建模 在 推荐
系统”. 在: 会议论文集 的  24th 国际 会议 在
世界 广泛 Web. Republic 和 Canton 的 Geneva, CHE. 278–288.
isbn: 9781450334693. doi: 10.1145/2736277.2741667. url: https:
//doi.org/10.1145/2736277.2741667.

Fan, W., Y. Ma, Q. Li, Y. 他, E. Zhao, J. Tang, 和 D. Yin. 2019.
“图 神经 网络 用于 Social 推荐”. 在:  世界
广泛 Web 会议. WWW ’19. San Francisco, CA, USA: 协会 用于 计算中 Machinery. 417–426. isbn: 9781450366748. doi:
10.1145/3308558.3313488. url: https://doi.org/10.1145/3308558.
3313488.

References

157

Fan, Y., J. Guo, Y. Lan, J. Xu, C. Zhai, 和 X. Cheng. 2018. “建模 多样 相关性 Patterns 在 临时检索”. 在:  41st
国际 ACM SIGIR 会议 在 研究 & 开发
在 信息检索. SIGIR ’18. Ann Arbor, MI, USA: ACM.
375–384. isbn: 978-1-4503-5657-2. doi: 10.1145/3209978.3209980.
url: http://doi.acm.org/10.1145/3209978.3209980.

Gao, C., X. Chen, F. Feng, K. Zhao, X. 他, Y. Li, 和 D. Jin. 2019.
“交叉-领域 推荐 无 Sharing 用户-相关
数据”. 在: 会议论文集 的  2019 世界 广泛 Web 会议 在
世界 广泛 Web. 491–502. doi: 10.1145/3308558.3313538. url:
https://doi.org/10.1145/3308558.3313538.

Gao, C., X. 他, D. Gan, X. Chen, F. Feng, Y. Li, 和 T.-S. Chua. 2019b.
“神经 多-任务 推荐 从 多-行为 数据”.
在: ICDE.

Gao, J., J.-Y. Nie, G. Wu, 和 G. Cao. 2004. “依赖 语言
模型 用于 信息检索”. 在: 会议论文集 的  27th Annual
国际 ACM SIGIR 会议 在 研究 和 开发
在 信息检索. SIGIR ’04. 她ﬃeld, United Kingdom:
ACM. 170–177. isbn: 1-58113-881-4. doi: 10.1145/1008992.1009024.
url: http://doi.acm.org/10.1145/1008992.1009024.

Gao, L., H. Yang, J. Wu, C. Zhou, W. Lu, 和 Y. Hu. 2018. “推荐 用 多-来源 异质 信息”. 在:
会议论文集 的  二十-第七 国际 联合 会议
在 Artiﬁcial 智能, IJCAI-18. 国际 联合 Conferences
在 Artiﬁcial 智能 组织. 3378–3384. doi: 10.24963/
ijcai.2018/469. url: https://doi.org/10.24963/ijcai.2018/469.
Garcia-Molina, H., G. Koutrika, 和 . Parameswaran. 2011. “信息 Seeking: 收敛 的 搜索, Recommendations, 和
Advertising”. Commun. ACM. 54(11): 121–130. issn: 0001-0782.
doi: 10.1145/2018396.2018423. url: http://doi.acm.org/10.1145/
2018396.2018423.

Gong, Y., H. Luo, 和 J. Zhang. 2018. “自然 语言 推理 超过
交互 空间”. 在: 6th 国际 会议 在 学习
表示, ICLR 2018.

Goodfellow, I., Y. Bengio, 和 . Courville. 2016. 深度学习. http:

//www.deeplearningbook.org. MIT Press.

158

References

Graves, ., S. Fernández, 和 J. Schmidhuber. 2007. “多-维
Recurrent 神经 网络”. 在: Artiﬁcial 神经 网络 – ICANN
2007. Berlin, Heidelberg: Springer Berlin Heidelberg. 549–558. isbn:
978-3-540-74690-4.

Guo, H., R. Tang, Y. Ye, Z. Li, 和 X. 他. 2017. “DeepFM:  Factorizationmachine 基于 神经网络 用于 CTR 预测”. 在: 会议论文集
的  26th 国际 联合 会议 在 Artiﬁcial 智能.
IJCAI’17. Melbourne, Australia: AAAI Press. 1725–1731. isbn: 978-
0-9992411-0-3. url: http://dl.acm.org/引用.cfm?id=3172077.
3172127.

Guo, J., Y. Fan, Q. Ai, 和 W. B. Croft. 2016. “ 深度 相关性
匹配模型 用于 临时检索”. 在: 会议论文集 的  25th
ACM 国际 在 会议 在 信息 和 知识
管理. CIKM ’16. Indianapolis, Indiana, USA: ACM. 55–
64. isbn: 978-1-4503-4073-1. doi: 10.1145/2983323.2983769. url:
http://doi.acm.org/10.1145/2983323.2983769.

Guo, J., Y. Fan, X. Ji, 和 X. Cheng. 2019. “MatchZoo:  学习, Practicing, 和 开发中 系统 用于 神经 文本 匹配”.
在: 会议论文集 的  42Nd 国际 ACM SIGIR 会议 在 研究 和 开发 在 信息检索. SIGIR’19. Paris, France: ACM. 1297–1300. isbn: 978-1-4503-6172-9.
doi: 10.1145/3331184.3331403. url: http://doi.acm.org/10.1145/
3331184.3331403.

Guo, J., Y. Fan, L. Pang, L. Yang, Q. Ai, H. Zamani, C. Wu, W. B.
Croft, 和 X. Cheng. 2019b. “ 深度 Look 到 神经 排序
模型 用于 信息检索”. CoRR. abs/1903.06902. arXiv:
1903.06902. url: http://arxiv.org/abs/1903.06902.

Guo, J., G. Xu, H. Li, 和 X. Cheng. 2008. “ Uniﬁed 和 判别
模型 用于 查询 Reﬁnement”. 在: 会议论文集 的  31st Annual
国际 ACM SIGIR 会议 在 研究 和 开发
在 信息检索. SIGIR ’08. Singapore, Singapore: ACM.
379–386. isbn: 978-1-60558-164-4. doi: 10.1145/1390334.1390400.
url: http://doi.acm.org/10.1145/1390334.1390400.

References

159

Guo, Y., Z. Cheng, L. Nie, X. Xu, 和 M. S. Kankanhalli. 2018. “Multimodal 偏好 建模 用于 产品搜索”. 在: 会议论文集 的
 26th ACM 国际 会议 在 Multimedia. 1865–1873.
doi: 10.1145/3240508.3240541. url: https://doi.org/10.1145/
3240508.3240541.

Gysel, C. V., M. de Rijke, 和 E. Kanoulas. 2018. “神经 向量
Spaces 用于 无监督 信息检索”. ACM Trans. Inf.
Syst. 36(4): 38:1–38:25. issn: 1046-8188. doi: 10.1145/3196826. url:
http://doi.acm.org/10.1145/3196826.

Haddad, D. 和 J. Ghosh. 2019. “学习 更多 从 更少: 朝向
Strengthening 弱监督 用于 临时检索”. 在: 会议论文集 的  42nd 国际 ACM SIGIR 会议 在
研究 和 开发 在 信息检索. SIGIR’19.
Paris, France: 协会 用于 计算中 Machinery. 857–860. isbn:
9781450361729. doi: 10.1145/3331184.3331272. url: https://doi.
org/10.1145/3331184.3331272.

Hardoon, D. R. 和 J. Shawe-Taylor. 2003. “KCCA 用于 diﬀerent 水平
精确率 在 content-基于 图像 检索”. 事件 Dates: 22 - 24
September 2004. url: https://eprints.soton.ac.uk/259596/.

Hardoon, D. R., S. R. Szedmak, 和 J. R. Shawe-taylor. 2004. “Canonical
相关 分析:  概述 用 应用  学习
方法”. 神经 Comput. 16(12): 2639–2664. issn: 0899-7667. doi:
10 . 1162 / 0899766042321814. url: http : / / dx . doi . org / 10 . 1162 /
0899766042321814.

他, K., X. Zhang, S. Ren, 和 J. Sun. 2016. “深度 Residual 学习
用于 图像 识别”. 在: 2016 IEEE 会议 在 计算机
视觉 和 模式 识别 (CVPR). 770–778. doi: 10.1109/
CVPR.2016.90.

他, R., W.-C. Kang, 和 J. McAuley. 2017. “翻译-基于 推荐”. 在: 会议论文集 的  Eleventh ACM 会议
在 推荐 系统. RecSys ’17. Como, Italy: ACM. 161–
169. isbn: 978-1-4503-4652-8. doi: 10.1145/3109859.3109882. url:
http://doi.acm.org/10.1145/3109859.3109882.

160

References

他, R. 和 J. McAuley. 2016. “VBPR: Visual 贝叶斯 Personalized
排序 从 隐式反馈”. 在: 会议论文集 的  Thirtieth
AAAI 会议 在 Artiﬁcial 智能. AAAI’16. Phoenix, Arizona: AAAI Press. 144–150. url: http://dl.acm.org/引用.cfm?
id=3015812.3015834.

他, R. 和 J. J. McAuley. 2016b. “Ups 和 Downs: 建模 
Visual 演化 的 Fashion Trends 用 一-类 协同
过滤”. 在: 会议论文集 的  25th 国际 会议
在 世界 广泛 Web, WWW 2016, Montreal, Canada, April 11
- 15, 2016. 507–517. doi: 10.1145/2872427.2883037. url: https:
//doi.org/10.1145/2872427.2883037.

他, X., Z. 他, J. Song, Z. Liu, Y. Jiang, 和 T. Chua. 2018. “NAIS:
神经 Attentive 物品 相似度 模型 用于 推荐”. IEEE
Transactions 在 知识 和 数据 工程. 30(12): 2354–
2366. issn: 1558-2191. doi: 10.1109/TKDE.2018.2831682.

他, X. 和 T.-S. Chua. 2017. “神经 Factorization Machines 用于 稀疏
Predictive Analytics”. 在: 会议论文集 的  40th 国际 ACM
SIGIR 会议 在 研究 和 开发 在 信息
检索. SIGIR ’17. Shinjuku, Tokyo, Japan: ACM. 355–364. isbn:
978-1-4503-5022-8. doi: 10.1145/3077136.3080777. url: http://doi.
acm.org/10.1145/3077136.3080777.

他, X., K. Deng, X. Wang, Y. Li, Y. Zhang, 和 M. Wang. 2020.
“LightGCN: Simplifying 和 Powering 图 卷积 网络
用于 推荐”. 在:  43rd 国际 ACM SIGIR
会议 在 研究 & 开发 在 信息检索.
SIGIR ’20. 新 York, NY, USA.

他, X., X. Du, X. Wang, F. Tian, J. Tang, 和 T.-S. Chua. 2018b. “Outer
乘积-基于 神经协同过滤”. 在: 会议论文集 的
 二十-第七 国际 联合 会议 在 Artiﬁcial
智能, IJCAI-18. 国际 联合 Conferences 在 Artiﬁcial
智能 组织. 2227–2233. doi: 10.24963/ijcai.2018/308.
url: https://doi.org/10.24963/ijcai.2018/308.

他, X., M. Gao, M.-Y. Kan, 和 D. Wang. 2017b. “BiRank: 朝向
排序 在 Bipartite Graphs”. IEEE Trans. 在 Knowl. 和 数据
Eng. 29(1): 57–71. issn: 1041-4347. doi: 10 . 1109 / TKDE . 2016 .
2611584. url: https://doi.org/10.1109/TKDE.2016.2611584.

References

161

他, X., M.-Y. Kan, P. Xie, 和 X. Chen. 2014. “评论-基于 Multiview 聚类 的 Web 2.0 物品”. 在: 会议论文集 的  23rd 国际 会议 在 世界 广泛 Web. WWW ’14. Seoul, Korea:
ACM. 771–782. isbn: 978-1-4503-2744-2. doi: 10.1145/2566486.
2567975. url: http://doi.acm.org/10.1145/2566486.2567975.
他, X., L. Liao, H. Zhang, L. Nie, X. Hu, 和 T.-S. Chua. 2017c. “神经
协同过滤”. 在: 会议论文集 的  26th 国际
会议 在 世界 广泛 Web. WWW ’17. Perth, Australia. 173–
182. isbn: 978-1-4503-4913-0. doi: 10.1145/3038912.3052569. url:
https://doi.org/10.1145/3038912.3052569.

他, X., H. Zhang, M.-Y. Kan, 和 T.-S. Chua. 2016b. “快 矩阵
Factorization 用于 在线 推荐 用 隐式反馈”.
在: 会议论文集 的  39th 国际 ACM SIGIR 会议 在
研究 和 开发 在 信息检索. SIGIR ’16. Pisa,
Italy: ACM. 549–558. isbn: 978-1-4503-4069-4. doi: 10.1145/2911451.
2911489. url: http://doi.acm.org/10.1145/2911451.2911489.

Hidasi, B., . Karatzoglou, L. Baltrunas, 和 D. Tikk. 2016. “Sessionbased Recommendations 用 Recurrent 神经 网络”. 在: 4th
国际 会议 在 学习 表示, ICLR 2016,
San Juan, Puerto Rico, 可以 2-4, 2016, 会议 轨迹 会议论文集. url: http://arxiv.org/abs/1511.06939.

Hinton, G. E. 和 R. R. Salakhutdinov. 2006. “减少中  维度 的 数据 用 神经 网络”. 科学. 313(5786): 504–
507. issn: 0036-8075. doi: 10.1126/科学.1127647. url: https:
//科学.sciencemag.org/content/313/5786/504.

Hofmann, T. 1999. “概率 潜在 语义 索引”. 在: 会议论文集 的  22nd annual 国际 ACM SIGIR 会议
在 研究 和 开发 在 信息检索. SIGIR ’99.
Berkeley, California, United 陈述: ACM. 50–57. isbn: 1-58113-096-
1. doi: 10.1145/312624.312649. url: http://doi.acm.org/10.1145/
312624.312649.

Hornik, K. 1991. “Approximation capabilities 的 multilayer feedforward
网络”. 神经 网络. 4(2): 251–257. issn: 0893-6080. doi:
https : / / doi . org / 10 . 1016 / 0893 - 6080(91 ) 90009 - T. url: http :
//www.sciencedirect.com/科学/文章/pii/089360809190009T.

162

References

Hsieh, C.-K., L. Yang, Y. Cui, T.-Y. Lin, S. Belongie, 和 D. Estrin.
2017. “协同 度量 学习”. 在: 会议论文集 的  26th
国际 会议 在 世界 广泛 Web. WWW ’17. Perth,
Australia. 193–201. isbn: 978-1-4503-4913-0. doi: 10.1145/3038912.
3052639. url: https://doi.org/10.1145/3038912.3052639.

Hu, B., Z. Lu, H. Li, 和 Q. Chen. 2014. “卷积神经网络 架构 用于 匹配 自然 语言 句子”. 在:
进展 在 神经 信息 Processing 系统 27. Curran
Associates, Inc. 2042–2050. url: http://papers.nips.cc/论文/5550-
卷积-神经-网络-架构-用于-匹配-naturallanguage-句子.pdf.

Huang, J., S. Yao, C. Lyu, 和 D. Ji. 2017. “多-粒度 神经
句子 模型 用于 测量中 短 文本 相似度”. 在: 数据库
系统 用于 高级 应用. Cham: Springer 国际
Publishing. 439–455. isbn: 978-3-319-55753-3.

Huang, P.-S., X. 他, J. Gao, L. Deng, . Acero, 和 L. Heck. 2013.
“学习 深度 Structured 语义 模型 用于 网页搜索 使用中
Clickthrough 数据”. 在: 会议论文集 的  22Nd ACM 国际
会议 在 信息 & 知识 管理. CIKM ’13.
San Francisco, California, USA: ACM. 2333–2338. isbn: 978-1-4503-
2263-8. doi: 10.1145/2505515.2505665. url: http://doi.acm.org/10.
1145/2505515.2505665.

Huang, Y., Q. Wu, W. Wang, 和 L. Wang. 2018. “图像 和 句子
匹配 通过 语义 Concepts 和 顺序 学习”. IEEE Transactions 在 模式 分析 和 机器 智能: 1–1. issn:
0162-8828. doi: 10.1109/TPAMI.2018.2883466.

Hui, K., . Yates, K. Berberich, 和 G. de Melo. 2017. “PACRR: 
位置-Aware 神经 IR 模型 用于 相关性 匹配”. 在: 会议论文集 的  2017 会议 在 经验的 方法 在 自然
语言 Processing. Copenhagen, Denmark: 协会 用于 计算 语言学. 1049–1058. doi: 10.18653/v1/D17-1110. url:
http://aclweb.org/anthology/D17-1110.

References

163

Hui, K., . Yates, K. Berberich, 和 G. de Melo. 2018. “Co-PACRR: 
上下文-Aware 神经 IR 模型 用于 临时检索”. 在: 会议论文集 的  Eleventh ACM 国际 会议 在 网页搜索
和 数据 挖掘. WSDM ’18. Marina Del Rey, CA, USA: ACM.
279–287. isbn: 978-1-4503-5581-0. doi: 10.1145/3159652.3159689.
url: http://doi.acm.org/10.1145/3159652.3159689.

Jiang, J.-Y., M. Zhang, C. Li, M. Bendersky, N. Golbandi, 和 M.
Najork. 2019. “语义 文本 匹配 用于 长-形式 文档”.
在:  世界 广泛 Web 会议. WWW ’19. San Francisco,
CA, USA: 协会 用于 计算中 Machinery. 795–806. isbn:
9781450366748. doi: 10.1145/3308558.3313707. url: https://doi.
org/10.1145/3308558.3313707.

Jiang, R., S. Gowal, Y. Qian, T. . Mann, 和 D. J. Rezende. 2019b.
“超出 Greedy 排序: Slate 优化 通过 列表-CVAE”. 在:
7th 国际 会议 在 学习 表示, ICLR
2019, 新 Orleans, LA, USA, 可以 6-9, 2019. OpenReview.net. url:
https://openreview.net/forum?id=r1xX42R5Fm.

Joachims, T. 2002. “Optimizing 搜索 Engines 使用中 Clickthrough
数据”. 在: 会议论文集 的  第八 ACM SIGKDD 国际
会议 在 知识 发现 和 数据 挖掘. KDD ’02.
Edmonton, Alberta, Canada: ACM. 133–142. isbn: 1-58113-567-X.
doi: 10.1145/775047.775067. url: http://doi.acm.org/10.1145/
775047.775067.

Joachims, T., . Swaminathan, 和 T. Schnabel. 2017. “Unbiased
学习--排序 用 Biased 反馈”. 在: 会议论文集 的 
第十 ACM 国际 会议 在 网页搜索 和 数据
挖掘. WSDM ’17. Cambridge, United Kingdom. 781–789. doi:
10.1145/3018661.3018699. url: https://doi.org/10.1145/3018661.
3018699.

Kabbur, S., X. Ning, 和 G. Karypis. 2013. “FISM: Factored 物品
相似度 模型 用于 top-N 推荐 系统”. 在: 会议论文集
的  19th ACM SIGKDD 国际 会议 在 知识
发现 和 数据 挖掘. KDD ’13. Chicago, Illinois, USA: ACM.
659–667. isbn: 978-1-4503-2174-7. doi: 10.1145/2487575.2487589.
url: http://doi.acm.org/10.1145/2487575.2487589.

164

References

Kang, W. 和 J. J. McAuley. 2018. “自身-Attentive 序列推荐”. 在: IEEE 国际 会议 在 数据 挖掘.
197–206.

Karatzoglou, ., X. Amatriain, L. Baltrunas, 和 N. Oliver. 2010.
“Multiverse 推荐: N-维 张量 Factorization
用于 上下文-aware 协同过滤”. 在: 会议论文集 的 
第四 ACM 会议 在 推荐 系统. RecSys ’10.
Barcelona, Spain: ACM. 79–86. isbn: 978-1-60558-906-0. doi: 10.
1145/1864708.1864727. url: http://doi.acm.org/10.1145/1864708.
1864727.

Karpathy, ., . Joulin, 和 L. Fei-Fei. 2014. “深度 Fragment Embeddings 用于 Bidirectional 图像 句子 映射”. 在: 会议论文集 的
 27th 国际 会议 在 神经 信息 Processing
系统 - 卷 2. NIPS’14. Montreal, Canada: MIT Press. 1889–
1897. url: http://dl.acm.org/引用.cfm?id=2969033.2969038.
Karpathy, . 和 F. Li. 2015. “深度 visual-语义 alignments 用于
生成中 图像 描述”. 在: IEEE 会议 在 计算机
视觉 和 模式 识别, CVPR 2015, Boston, MA, USA,
June 7-12, 2015. IEEE 计算机 社会. 3128–3137. doi: 10.1109/
CVPR.2015.7298932. url: https://doi.org/10.1109/CVPR.2015.
7298932.

Kenter, T., . Borisov, C. Van Gysel, M. Dehghani, M. de Rijke, 和
B. Mitra. 2017. “神经 网络 用于 信息检索”. 在:
会议论文集 的  40th 国际 ACM SIGIR 会议 在
研究 和 开发 在 信息检索. SIGIR ’17.
Shinjuku, Tokyo, Japan: ACM. 1403–1406. isbn: 978-1-4503-5022-8.
doi: 10.1145/3077136.3082062. url: http://doi.acm.org/10.1145/
3077136.3082062.

Kingma, D. P. 和 M. Welling. 2014. “Auto-编码 Variational
Bayes”. 在: 2nd 国际 会议 在 学习 表示, ICLR 2014, Banﬀ, AB, Canada, April 14-16, 2014, 会议
轨迹 会议论文集. url: http://arxiv.org/abs/1312.6114.

References

165

Koren, Y. 2008. “Factorization Meets  Neighborhood:  Multifaceted
协同过滤 模型”. 在: 会议论文集 的  14th ACM
SIGKDD 国际 会议 在 知识 发现 和
数据 挖掘. KDD ’08. Las Vegas, Nevada, USA: ACM. 426–434.
isbn: 978-1-60558-193-4. doi: 10.1145/1401890.1401944. url: http:
//doi.acm.org/10.1145/1401890.1401944.

Koren, Y., R. Bell, 和 C. Volinsky. 2009. “矩阵分解
技术 用于 推荐 系统”. 计算机. 42(8): 30–37.
issn: 0018-9162. doi: 10.1109/MC.2009.263. url: http://dx.doi.
org/10.1109/MC.2009.263.

Koren, Y., R. Bell, 和 C. Volinsky. 2009b. “矩阵分解
技术 用于 推荐 系统”. 计算机. 42(8): 30–37.
issn: 0018-9162. doi: 10.1109/MC.2009.263. url: http://dx.doi.
org/10.1109/MC.2009.263.

Kulis, B. 等人 2013. “度量 学习:  综述”. Foundations 和

Trends R(cid:13) 在 机器学习. 5(4): 287–364.

Le, Q. 和 T. Mikolov. 2014. “分布式 表示 的 句子
和 文档”. 在: 会议论文集 的  31st 国际 会议 在 国际 会议 在 机器学习 - 卷
32. ICML’14. Beijing, China: JMLR.org. II–1188–II–1196.

Lei, C., D. Liu, W. Li, Z. Zha, 和 H. Li. 2016. “Comparative 深度
学习 的 混合 表示 用于 图像 Recommendations”.
在: 2016 IEEE 会议 在 计算机 视觉 和 模式 识别, CVPR 2016, Las Vegas, NV, USA, June 27-30, 2016. 2545–
2553. doi: 10.1109/CVPR.2016.279. url: https://doi.org/10.1109/
CVPR.2016.279.

Lei, W., X. 他, Y. Miao, Q. Wu, R. Hong, M.-Y. Kan, 和 T.-S. Chua.
2020. “估计-动作-Reﬂection: 朝向 深度 交互 之间 Conversational 和 推荐 系统”. 在: 会议论文集
的  13th ACM 国际 会议 在 网页搜索 和 数据
挖掘. WSDM ’20. 新 York, NY, USA: ACM.

166

References

Li, C., C. Quan, L. Peng, Y. Qi, Y. Deng, 和 L. Wu. 2019. “ Capsule
网络 用于 推荐 和 解释中 什么 你 如 和
Dislike”. 在: 会议论文集 的  42nd 国际 ACM SIGIR
会议 在 研究 和 开发 在 信息检索,
SIGIR 2019, Paris, France, July 21-25, 2019. 275–284. doi: 10.1145/
3331184.3331216. url: https://doi.org/10.1145/3331184.3331216.
Li, H. 2011. “学习  排序 用于 信息检索 和 自然
语言 processing”. 综合 Lectures 在 人类 语言 Technologies. 4(1): 1–113.

Li, H. 和 Z. Lu. 2016. “深度学习 用于 信息检索”.
在: 会议论文集 的  39th 国际 ACM SIGIR 会议
在 研究 和 开发 在 信息检索. SIGIR
’16. Pisa, Italy: ACM. 1203–1206. isbn: 978-1-4503-4069-4. doi:
10 . 1145 / 2911451 . 2914800. url: http : / / doi . acm . org / 10 . 1145 /
2911451.2914800.

Li, H. 和 J. Xu. 2014. “语义匹配 在 搜索”. Foundations 和
Trends 在 信息检索. 7(5): 343–469. issn: 1554-0669. doi:
10.1561/1500000035. url: http://dx.doi.org/10.1561/1500000035.
Li, J., P. Ren, Z. Chen, Z. Ren, T. Lian, 和 J. Ma. 2017. “神经
Attentive 基于会话的推荐”. 在: 会议论文集 的 
2017 ACM 在 会议 在 信息 和 知识 管理. CIKM ’17. Singapore, Singapore: ACM. 1419–1428. isbn:
978-1-4503-4918-5. doi: 10 . 1145 / 3132847 . 3132926. url: http :
//doi.acm.org/10.1145/3132847.3132926.

Li, L., W. Chu, J. Langford, 和 R. E. Schapire. 2010. “ ContextualBandit 方法  Personalized News 文章 推荐”.
在: 会议论文集 的  19th 国际 会议 在 世界 广泛
Web. WWW ’10. Raleigh, North Carolina, USA: 协会 用于
计算中 Machinery. 661–670. isbn: 9781605587998. doi: 10.1145/
1772690.1772758. url: https://doi.org/10.1145/1772690.1772758.
Li, S., J. Kawale, 和 Y. Fu. 2015. “深度 协同过滤 通过
Marginalized Denoising Auto-编码器”. 在: 会议论文集 的  24th
ACM 国际 在 会议 在 信息 和 知识
管理. CIKM ’15. Melbourne, Australia: ACM. 811–820.
isbn: 978-1-4503-3794-6. doi: 10.1145/2806416.2806527. url: http:
//doi.acm.org/10.1145/2806416.2806527.

References

167

Lian, J., X. Zhou, F. Zhang, Z. Chen, X. Xie, 和 G. Sun. 2018.
“xDeepFM: 组合中 显式 和 隐式 特征 Interactions 用于
推荐 系统”. 在: 会议论文集 的  24th ACM SIGKDD
国际 会议 在 知识 发现 & 数据 挖掘.
KDD ’18. London, United Kingdom: ACM. 1754–1763. isbn: 978-1-
4503-5552-0. doi: 10.1145/3219819.3220023. url: http://doi.acm.
org/10.1145/3219819.3220023.

Liang, D., L. Charlin, J. McInerney, 和 D. M. Blei. 2016. “建模
用户 Exposure 在 推荐”. 在: 会议论文集 的  25th 国际 会议 在 世界 广泛 Web, WWW 2016, Montreal,
Canada, April 11 - 15, 2016. 951–961. doi: 10.1145/2872427.2883090.
url: https://doi.org/10.1145/2872427.2883090.

Liang, D., R. G. Krishnan, M. D. Hoﬀman, 和 T. Jebara. 2018. “Variational Autoencoders 用于 协同过滤”. 在: 会议论文集
的  2018 世界 广泛 Web 会议. WWW ’18. Lyon, France:
国际 世界 广泛 Web Conferences Steering Committee.
689–698. isbn: 9781450356398. doi: 10.1145/3178876.3186150. url:
https://doi.org/10.1145/3178876.3186150.

Liu, B., D. Niu, H. Wei, J. Lin, Y. 他, K. Lai, 和 Y. Xu. 2019. “匹配 文章 Pairs 用 Graphical 分解 和 Convolutions”.
在: 会议论文集 的  57th Annual Meeting 的  协会 用于
计算 语言学. Florence, Italy: 协会 用于 计算 语言学. 6284–6294. doi: 10.18653/v1/P19-1632. url:
https://www.aclweb.org/anthology/P19-1632.

Liu, F., Z. Cheng, C. Sun, Y. Wang, L. Nie, 和 M. S. Kankanhalli.
2019b. “用户 多样 偏好 建模 通过 Multimodal Attentive 度量 学习”. 在: 会议论文集 的  27th ACM 国际 会议 在 Multimedia. 1526–1534. doi: 10.1145/3343031.
3350953. url: https://doi.org/10.1145/3343031.3350953.

Liu, T.-Y. 2009. “学习  排序 用于 信息检索”. 发现.
Trends Inf. Retr. 3(3): 225–331. issn: 1554-0669. doi: 10 . 1561 /
1500000016. url: http://dx.doi.org/10.1561/1500000016.

Liu, Y., Y. Guo, E. M. Bakker, 和 M. S. Lew. 2017. “学习 
Recurrent Residual 融合 网络 用于 Multimodal 匹配”. 在:
2017 IEEE 国际 会议 在 计算机 视觉 (ICCV).
4127–4136. doi: 10.1109/ICCV.2017.442.

168

References

Ma, L., Z. Lu, L. Shang, 和 H. Li. 2015. “Multimodal 卷积
神经 网络 用于 匹配 图像 和 句子”. 在: 会议论文集
的  2015 IEEE 国际 会议 在 计算机 视觉
(ICCV). ICCV ’15. Washington, DC, USA: IEEE 计算机 社会.
2623–2631. isbn: 978-1-4673-8391-2. doi: 10.1109/ICCV.2015.301.
url: http://dx.doi.org/10.1109/ICCV.2015.301.

Masci, J., U. Meier, D. Cireş, 和 J. Schmidhuber. 2011. “Stacked 卷积 Auto-encoders 用于 层次 特征提取”. 在:
会议论文集 的  21th 国际 会议 在 Artiﬁcial 神经
网络 - 卷 部分 I. ICANN’11. Espoo, Finland: SpringerVerlag. 52–59. isbn: 978-3-642-21734-0. url: http://dl.acm.org/
引用.cfm?id=2029556.2029563.

Mikolov, T., I. Sutskever, K. Chen, G. Corrado, 和 J. Dean. 2013.
“分布式 表示 的 词 和 短语 和 其 Compositionality”. 在: 会议论文集 的  26th 国际 会议
在 神经 信息 Processing 系统 - 卷 2. NIPS’13.
Lake Tahoe, Nevada: Curran Associates Inc. 3111–3119. url: http:
//dl.acm.org/引用.cfm?id=2999792.2999959.

Mitra, B. 和 N. Craswell. 2018. “ 介绍  神经 信息
检索”. Foundations 和 Trends R(cid:13) 在 信息检索.
13(1): 1–126. url: https://www.microsoft.com/en- 我们/研究/
出版物/介绍-神经-信息-检索/.

Mitra, B. 和 N. Craswell. 2019. “DUET 在 TREC 2019 深度学习 轨迹”. 在: 会议论文集 的  二十-第八 文本 检索
会议, TREC 2019, Gaithersburg, Maryland, USA, November 13-15, 2019. url: https://trec.nist.gov/pubs/trec28/papers/
Microsoft.DL.pdf.

Mitra, B., F. Diaz, 和 N. Craswell. 2017. “学习匹配 使用中
局部 和 分布式 表示 的 文本 用于 网页搜索”. 在:
会议论文集 的  26th 国际 会议 在 世界 广泛
Web. WWW ’17. Perth, Australia. 1291–1299. isbn: 978-1-4503-4913-
0. doi: 10.1145/3038912.3052579. url: https://doi.org/10.1145/
3038912.3052579.

References

169

Nallapati, R. 2004. “判别 模型 用于 信息检索”.
在: 会议论文集 的  27th Annual 国际 ACM SIGIR 会议 在 研究 和 开发 在 信息检索.
SIGIR ’04. 她ﬃeld, United Kingdom: ACM. 64–71. isbn: 1-58113-
881-4. doi: 10.1145/1008992.1009006. url: http://doi.acm.org/10.
1145/1008992.1009006.

Naumov, M., D. Mudigere, H. M. Shi, J. Huang, N. Sundaraman, J.
Park, X. Wang, U. Gupta, C. Wu, . G. Azzolini, D. Dzhulgakov,
. Mallevich, I. Cherniavskii, Y. Lu, R. Krishnamoorthi, . Yu,
V. Kondratenko, S. Pereira, X. Chen, W. Chen, V. Rao, B. Jia, L.
Xiong, 和 M. Smelyanskiy. 2019. “深度学习 推荐
模型 用于 个性化 和 推荐 系统”. CoRR.
abs/1906.00091. arXiv: 1906.00091. url: http://arxiv.org/abs/1906.
00091.

Nguyen, T., M. Rosenberg, X. Song, J. Gao, S. Tiwary, R. Majumder,
和 L. Deng. 2016. “MS MARCO:  人类 生成了 机器
阅读 理解 数据集”. CoRR. abs/1611.09268. arXiv:
1611.09268. url: http://arxiv.org/abs/1611.09268.

Nie, Y., . Sordoni, 和 J.-Y. Nie. 2018. “多-水平 抽象 卷积 模型 用 弱监督 用于 信息检索”.
在:  41st 国际 ACM SIGIR 会议 在 研究 &
开发 在 信息检索. SIGIR ’18. Ann Arbor, MI,
USA: ACM. 985–988. isbn: 978-1-4503-5657-2. doi: 10.1145/3209978.
3210123. url: http://doi.acm.org/10.1145/3209978.3210123.

Nogueira, R. 和 K. Cho. 2019. “段落 Re-排序 用 BERT”.
CoRR. abs/1901.04085. arXiv: 1901.04085. url: http://arxiv.org/
abs/1901.04085.

Nogueira, R., W. Yang, K. Cho, 和 J. Lin. 2019. “多-阶段 文档 排序 用 BERT”. arXiv: 1910.14424. url: https://arxiv.
org/abs/1910.14424.

170

References

Onal, K. D., Y. Zhang, I. S. Altingovde, M. M. Rahman, P. Karagoz, .
Braylan, B. Dang, H.-L. Chang, H. Kim, Q. McNamara, . Angert,
E. Banner, V. Khetan, T. McDonnell, . T. Nguyen, D. Xu, B. C.
Wallace, M. de Rijke, 和 M. Lease. 2018. “神经 信息
检索: 在  端 的  early 年”. 信息检索
Journal. 21(2): 111–182. issn: 1573-7659. doi: 10.1007/s10791-017-
9321-y. url: https://doi.org/10.1007/s10791-017-9321-y.

Onal, K. D., Y. Zhang, I. S. Altingovde, M. M. Rahman, P. Karagoz,
. Braylan, B. Dang, H.-L. Chang, H. Kim, Q. Mcnamara, .
Angert, E. Banner, V. Khetan, T. Mcdonnell, . T. Nguyen, D. Xu,
B. C. Wallace, M. Rijke, 和 M. Lease. 2018b. “神经 信息
检索: 在  端 的  Early 年”. Inf. Retr. 21(2-3): 111–
182. issn: 1386-4564. doi: 10.1007/s10791-017-9321-y. url: https:
//doi.org/10.1007/s10791-017-9321-y.

Palangi, H., L. Deng, Y. Shen, J. Gao, X. 他, J. Chen, X. Song, 和
R. Ward. 2016. “深度 句子 嵌入 使用中 长 Shortterm 记忆 网络: 分析 和 应用  信息
检索”. IEEE/ACM Trans. 音频, 语音 和 Lang. Proc. 24(4):
694–707. issn: 2329-9290. doi: 10.1109/TASLP.2016.2520371. url:
https://doi.org/10.1109/TASLP.2016.2520371.

Pan, F., S. Li, X. Ao, P. Tang, 和 Q. 他. 2019. “Warm 向上 Cold-开始
Advertisements: 改进中 CTR Predictions 通过 学习  学习
ID Embeddings”. 在: 会议论文集 的  42nd 国际 ACM
SIGIR 会议 在 研究 和 开发 在 信息
检索, SIGIR 2019, Paris, France, July 21-25, 2019. 695–704.
doi: 10.1145/3331184.3331268. url: https://doi.org/10.1145/
3331184.3331268.

Pang, L., Y. Lan, J. Guo, J. Xu, 和 X. Cheng. 2016. “ 研究 的
MatchPyramid 模型 在 临时检索”. CoRR. abs/1606.04648.
arXiv: 1606.04648. url: http://arxiv.org/abs/1606.04648.

Pang, L., Y. Lan, J. Guo, J. Xu, S. Wan, 和 X. Cheng. 2016b. “文本
匹配 作为 图像 识别”. 在: 会议论文集 的  Thirtieth
AAAI 会议 在 Artiﬁcial 智能. AAAI’16. Phoenix, Arizona: AAAI Press. 2793–2799. url: http://dl.acm.org/引用.cfm?
id=3016100.3016292.

References

171

Pang, L., Y. Lan, J. Guo, J. Xu, J. Xu, 和 X. Cheng. 2017. “DeepRank:
 新 深度 架构 用于 相关性 排序 在 信息
检索”. 在: 会议论文集 的  2017 ACM 在 会议 在
信息 和 知识 管理. CIKM ’17. Singapore,
Singapore: ACM. 257–266. isbn: 978-1-4503-4918-5. doi: 10.1145/
3132847 . 3132914. url: http : / / doi . acm . org / 10 . 1145 / 3132847 .
3132914.

Pang, L., Y. Lan, J. Xu, J. Guo, S.-X. Wan, 和 X. Cheng. 2017b. “ 综述 在 深度 文本 匹配”. Chinese Journal 的 Computers. 40(4):
985–1003. issn: 0254-4164. doi: 10.11897/SP.J.1016.2017.00985.
url: http://cjc.ict.ac.cn/在线/onlinepaper/pl-201745181647.pdf.
Pang, L., J. Xu, Q. Ai, Y. Lan, X. Cheng, 和 J.-R. Wen. 2020. “SetRank:
学习  Permutation-Invariant 排序模型 用于 信息
检索”. 在:  43rd 国际 ACM SIGIR 会议
在 研究 & 开发 在 信息检索. SIGIR ’20.
协会 用于 计算中 Machinery.

Parikh, ., O. Täckström, D. Das, 和 J. Uszkoreit. 2016. “ Decomposable 注意力 模型 用于 自然 语言 推理”. 在:
会议论文集 的  2016 会议 在 经验的 方法 在 自然
语言 Processing. Austin, Texas: 协会 用于 计算
语言学. 2249–2255. doi: 10.18653/v1/D16- 1244. url: http:
//www.aclweb.org/anthology/D16-1244.

Pasricha, R. 和 J. McAuley. 2018. “翻译-基于 Factorization
Machines 用于 序列推荐”. 在: 会议论文集 的 
12th ACM 会议 在 推荐 系统. RecSys ’18. Vancouver, British Columbia, Canada: 协会 用于 计算中 Machinery. 63–71. isbn: 9781450359016. doi: 10.1145/3240323.3240356.
url: https://doi.org/10.1145/3240323.3240356.

Pasumarthi, R. K., S. Bruch, X. Wang, C. Li, M. Bendersky, M. Najork,
J. Pfeifer, N. Golbandi, R. Anil, 和 S. Wolf. 2019. “TF-排序:
可扩展 TensorFlow Library 用于 学习--排序”. 在: 会议论文集
的  25th ACM SIGKDD 国际 会议 在 知识
发现 & 数据 挖掘. KDD ’19. Anchorage, AK, USA: ACM.
2970–2978. isbn: 978-1-4503-6201-6. doi: 10.1145/3292500.3330677.
url: http://doi.acm.org/10.1145/3292500.3330677.

172

References

Pearl, J. 2019. “ 七 Tools 的 因果 推理, 用 Reﬂections 在
机器学习”. Commun. ACM. 62(3): 54–60. issn: 0001-0782.
doi: 10.1145/3241036. url: https://doi.org/10.1145/3241036.
Pei, C., Y. Zhang, Y. Zhang, F. Sun, X. Lin, H. Sun, J. Wu, P. Jiang,
J. Ge, W. Ou, 和 D. Pei. 2019. “Personalized Re-排序 用于
推荐”. 在: 会议论文集 的  13th ACM 会议 在
推荐 系统. RecSys ’19. Copenhagen, Denmark: 协会 用于 计算中 Machinery. 3–11. isbn: 9781450362436. doi:
10.1145/3298689.3347000. url: https://doi.org/10.1145/3298689.
3347000.

Pennington, J., R. Socher, 和 C. Manning. 2014. “Glove: 全局 Vectors
用于 词 表示”. 在: 会议论文集 的  2014 会议 在
经验的 方法 在 自然语言处理 (EMNLP). Doha,
Qatar: 协会 用于 计算 语言学. 1532–1543. doi:
10.3115/v1/D14-1162. url: https://www.aclweb.org/anthology/
D14-1162.

Peters, M., M. Neumann, M. Iyyer, M. Gardner, C. Clark, K. Lee, 和 L.
Zettlemoyer. 2018. “深度 Contextualized 词 表示”. 在:
会议论文集 的  2018 会议 的  North American 章
的  协会 用于 计算 语言学: 人类 语言
Technologies, 卷 1 (长 Papers). 新 Orleans, Louisiana: 协会 用于 计算 语言学. 2227–2237. doi: 10.18653/
v1/N18-1202. url: https://www.aclweb.org/anthology/N18-1202.
Qiao, Y., C. Xiong, Z. Liu, 和 Z. Liu. 2019. “理解 
Behaviors 的 BERT 在 排序”. CoRR. abs/1904.07531. arXiv:
1904.07531. url: http://arxiv.org/abs/1904.07531.

Qiu, R., J. Li, Z. Huang, 和 H. Yin. 2019. “Rethinking  物品 顺序
在 基于会话的推荐 用 图 神经 网络”.
在: 会议论文集 的  28th ACM 国际 会议 在 信息 和 知识 管理, CIKM 2019, Beijing, China,
November 3-7, 2019. 579–588. doi: 10.1145/3357384.3358010. url:
https://doi.org/10.1145/3357384.3358010.

References

173

Qiu, X. 和 X. Huang. 2015. “卷积 神经 张量 网络
架构 用于 社区-基于 问答”. 在: 会议论文集 的  24th 国际 会议 在 Artiﬁcial 智能. IJCAI’15. Buenos Aires, Argentina: AAAI Press. 1305–1311.
isbn: 978-1-57735-738-4. url: http://dl.acm.org/引用.cfm?id=
2832415.2832431.

Radford, ., K. Narasimhan, T. Salimans, 和 I. Sutskever. 2018.
“改进中 语言 理解 通过 生成 pre-训练”.
Radford, ., J. Wu, R. Child, D. Luan, D. Amodei, 和 I. Sutskever.
2019. “语言 模型 是 无监督 Multitask Learners”.
Ranzato, M. ., Y.-L. Boureau, 和 Y. LeCun. 2007. “稀疏 特征
学习 用于 深度 信念 网络”. 在: 会议论文集 的  20th
国际 会议 在 神经 信息 Processing 系统.
NIPS’07. Vancouver, British Columbia, Canada: Curran Associates
Inc. 1185–1192. isbn: 978-1-60560-352-0. url: http://dl.acm.org/
引用.cfm?id=2981562.2981711.

Rasiwasia, N., J. Costa Pereira, E. Coviello, G. Doyle, G. R. Lanckriet,
R. Levy, 和 N. Vasconcelos. 2010. “ 新 方法  Crossmodal Multimedia 检索”. 在: 会议论文集 的  18th ACM
国际 会议 在 Multimedia. MM ’10. Firenze, Italy:
ACM. 251–260. isbn: 978-1-60558-933-6. doi: 10.1145/1873951.
1873987. url: http://doi.acm.org/10.1145/1873951.1873987.

Reimers, N. 和 I. Gurevych. 2019. “句子-BERT: 句子 Embeddings 使用中 Siamese BERT-网络”. 在: 会议论文集 的  2019
会议 在 经验的 方法 在 自然语言处理.
协会 用于 计算 语言学. url: http://arxiv.org/
abs/1908.10084.

Rendle, S. 2010. “Factorization Machines”. 在: 会议论文集 的  2010
IEEE 国际 会议 在 数据 挖掘. ICDM ’10. Washington, DC, USA: IEEE 计算机 社会. 995–1000. isbn: 978-0-
7695-4256-0. doi: 10.1109/ICDM.2010.127. url: http://dx.doi.org/
10.1109/ICDM.2010.127.

174

References

Rendle, S., C. Freudenthaler, Z. Gantner, 和 L. Schmidt-Thieme.
2009. “BPR: 贝叶斯个性化排序 从 隐式反馈”.
在: 会议论文集 的  二十-第五 会议 在 不确定性 在
Artiﬁcial 智能. UAI ’09. Montreal, Quebec, Canada: AUAI
Press. 452–461. isbn: 978-0-9749039-5-8. url: http://dl.acm.org/
引用.cfm?id=1795114.1795167.

Rendle, S., C. Freudenthaler, 和 L. Schmidt-Thieme. 2010. “Factorizing
Personalized Markov Chains 用于 下一个-basket 推荐”. 在:
会议论文集 的  19th 国际 会议 在 世界 广泛
Web. WWW ’10. Raleigh, North Carolina, USA: ACM. 811–820.
isbn: 978-1-60558-799-8. doi: 10.1145/1772690.1772773. url: http:
//doi.acm.org/10.1145/1772690.1772773.

Ricci, F., L. Rokach, 和 B. Shapira. 2015. 推荐 系统
Handbook. 2nd. Springer Publishing 公司, Incorporated. isbn:
1489976361, 9781489976369.

Richardson, M., R. Agrawal, 和 P. M. Domingos. 2003. “信任 管理 用于  语义 Web”. 在:  语义 Web - ISWC
2003, 秒 国际 语义 Web 会议, Sanibel Island, FL, USA, October 20-23, 2003, 会议论文集. 351–368. doi:
10.1007/978-3-540-39718-2\_23. url: https://doi.org/10.1007/978-
3-540-39718-2%5C_23.

Robertson, S., H. Zaragoza, 和 M. Taylor. 2004. “简单 BM25 扩展  多个 Weighted Fields”. 在: 会议论文集 的  Thirteenth ACM 国际 会议 在 信息 和 知识 管理. CIKM ’04. Washington, D.C., USA: ACM. 42–49.
isbn: 1-58113-874-1. doi: 10 . 1145 / 1031171 . 1031181. url: http :
//doi.acm.org/10.1145/1031171.1031181.

Rosipal, R. 和 N. Krämer. 2006. “概述 和 最新进展 在
部分 至少 squares”. 在: 会议论文集 的  2005 国际
会议 在 子空间, 潜在 结构 和 特征选择.
SLSFS’05. Bohinj, Slovenia: Springer-Verlag. 34–51. isbn: 3-540-
34137-4, 978-3-540-34137-6. doi: 10.1007/11752790_2. url: http:
//dx.doi.org/10.1007/11752790_2.

References

175

Salakhutdinov, R. 和 . Mnih. 2007. “概率 矩阵分解”. 在: 会议论文集 的  20th 国际 会议 在
神经 信息 Processing 系统. NIPS’07. Vancouver, British
Columbia, Canada: Curran Associates Inc. 1257–1264. isbn: 978-
1-60560-352-0. url: http://dl.acm.org/引用.cfm?id=2981562.
2981720.

Sarwar, B., G. Karypis, J. Konstan, 和 J. Riedl. 2001. “物品-基于 协同过滤 推荐 算法”. 在: 会议论文集
的  10th 国际 会议 在 世界 广泛 Web. WWW
’01. Hong Kong, Hong Kong: ACM. 285–295. isbn: 1-58113-348-0.
doi: 10.1145/371920.372071. url: http://doi.acm.org/10.1145/
371920.372071.

Schedl, M., H. Zamani, C.-W. Chen, Y. Deldjoo, 和 M. Elahi. 2018.
“当前 挑战 和 visions 在 music 推荐 系统 研究”. 国际 Journal 的 Multimedia 信息检索.
7(2): 95–116. doi: 10.1007/s13735-018-0154-2. url: https://doi.org/
10.1007/s13735-018-0154-2.

Sedhain, S., . K. Menon, S. Sanner, 和 L. Xie. 2015. “AutoRec:
Autoencoders Meet 协同过滤”. 在: 会议论文集 的 
24th 国际 会议 在 世界 广泛 Web. WWW ’15
Companion. Florence, Italy: ACM. 111–112. isbn: 978-1-4503-3473-0.
doi: 10.1145/2740908.2742726. url: http://doi.acm.org/10.1145/
2740908.2742726.

Shan, Y., T. R. Hoens, J. Jiao, H. Wang, D. Yu, 和 J. Mao. 2016.
“深度 Crossing: Web-规模 建模 无 手动地 Crafted
Combinatorial 特征”. 在: 会议论文集 的  22Nd ACM SIGKDD
国际 会议 在 知识 发现 和 数据 挖掘.
KDD ’16. San Francisco, California, USA: ACM. 255–262. isbn:
978-1-4503-4232-2. doi: 10.1145/2939672.2939704. url: http://doi.
acm.org/10.1145/2939672.2939704.

176

References

Shen, Y., X. 他, J. Gao, L. Deng, 和 G. Mesnil. 2014. “ 潜在 语义 模型 用 卷积-池化 结构 用于 信息
检索”. 在: 会议论文集 的  23rd ACM 国际 会议 在 会议 在 信息 和 知识 管理.
CIKM ’14. Shanghai, China: ACM. 101–110. isbn: 978-1-4503-2598-
1. doi: 10.1145/2661829.2661935. url: http://doi.acm.org/10.1145/
2661829.2661935.

Shi, Y., M. Larson, 和 . Hanjalic. 2014. “协同过滤
超出  用户-物品 矩阵:  综述 的  陈述 的  Art
和 未来 挑战”. ACM Comput. Surv. 47(1): 3:1–3:45. issn:
0360-0300. doi: 10.1145/2556270. url: http://doi.acm.org/10.1145/
2556270.

Socher, R., D. Chen, C. D. Manning, 和 . Y. Ng. 2013. “推理
用 神经 张量 网络 用于 知识库 完成”.
在: 会议论文集 的  26th 国际 会议 在 神经
信息 Processing 系统 - 卷 1. NIPS’13. Lake Tahoe,
Nevada: Curran Associates Inc. 926–934. url: http://dl.acm.org/
引用.cfm?id=2999611.2999715.

Srivastava, N., G. Hinton, . Krizhevsky, I. Sutskever, 和 R. Salakhutdinov. 2014. “随机失活:  简单 方式  Prevent 神经 网络
从 超过ﬁtting”. J. Mach. 学习. Res. 15(1): 1929–1958. issn:
1532-4435. url: http://jmlr.org/papers/v15/srivastava14.html.
Sun, F., J. Liu, J. Wu, C. Pei, X. Lin, W. Ou, 和 P. Jiang. 2019.
“BERT4Rec: 序列推荐 用 Bidirectional 编码器 表示 从 Transformer”. 在: 会议论文集 的 
28th ACM 国际 会议 在 信息 和 知识 管理. CIKM ’19. Beijing, China: ACM. 1441–1450.
isbn: 978-1-4503-6976-3. doi: 10.1145/3357384.3357895. url: http:
//doi.acm.org/10.1145/3357384.3357895.

Surdeanu, M., M. Ciaramita, 和 H. Zaragoza. 2011. “学习 
排序 答案  Non-Factoid 问题 从 Web Collections”.
计算 语言学. 37(2): 351–383. doi: 10.1162/COLI\_\
_00051. eprint: https://doi.org/10.1162/COLI__00051. url:
https://doi.org/10.1162/COLI_a_00051.

References

177

Tan, C., F. Wei, W. Wang, W. Lv, 和 M. Zhou. 2018. “Multiway 注意力 网络 用于 建模 句子 Pairs”. 在: 会议论文集
的  27th 国际 联合 会议 在 Artiﬁcial 智能. IJCAI’18. Stockholm, Sweden: AAAI Press. 4411–4417. isbn:
9780999241127.

Tang, J., X. Du, X. 他, F. Yuan, Q. Tian, 和 T. Chua. 2020. “对抗 训练 朝向 鲁棒 Multimedia 推荐 系统”.
IEEE Transactions 在 知识 和 数据 工程. 32(5): 855–
867. issn: 1558-2191. doi: 10.1109/TKDE.2019.2893638.

Tang, J. 和 K. Wang. 2018. “Personalized Top-N 序列推荐 通过 卷积 序列 嵌入”. 在: 会议论文集
的  Eleventh ACM 国际 会议 在 网页搜索 和
数据 挖掘. WSDM ’18. Marina Del Rey, CA, USA: 协会 用于
计算中 Machinery. 565–573. isbn: 9781450355810. doi: 10.1145/
3159652.3159656. url: https://doi.org/10.1145/3159652.3159656.
Tao, Z., X. Wang, X. 他, X. Huang, 和 T.-S. Chua. 2019. “HoAFM:
 高-顺序 Attentive 分解机 用于 CTR 预测”.
信息 Processing &#38; 管理: 102076. issn: 0306-
4573. doi: https://doi.org/10.1016/j.ipm.2019.102076. url: http:
//www.sciencedirect.com/科学/文章/pii/S0306457319302389.
Tay, Y., L. Anh Tuan, 和 S. C. Hui. 2018. “潜在 Relational 度量
学习 通过 记忆-基于 注意力 用于 协同 排序”.
在: 会议论文集 的  2018 世界 广泛 Web 会议. WWW
’18. Lyon, France. 729–739. isbn: 978-1-4503-5639-8. doi: 10.1145/
3178876.3186154. url: https://doi.org/10.1145/3178876.3186154.
Tay, Y., . T. Luu, 和 S. C. Hui. 2018b. “Co-Stack Residual ﬃnity
网络 用 多-水平 注意力 Reﬁnement 用于 匹配 文本
Sequences”. 在: 会议论文集 的  2018 会议 在 经验的
方法 在 自然语言处理. Brussels, Belgium: 协会 用于 计算 语言学. 4492–4502. doi: 10.18653/v1/
D18-1479. url: https://www.aclweb.org/anthology/D18-1479.

178

References

Tay, Y., . T. Luu, 和 S. C. Hui. 2018c. “Hermitian Co-注意力 网络 用于 文本 匹配 在 Asymmetrical Domains”. 在: 会议论文集
的  二十-第七 国际 联合 会议 在 Artiﬁcial
智能, IJCAI-18. 国际 联合 Conferences 在 Artiﬁcial
智能 组织. 4425–4431. doi: 10.24963/ijcai.2018/615.
url: https://doi.org/10.24963/ijcai.2018/615.

Tay, Y., L. . Tuan, 和 S. C. Hui. 2018d. “多-Cast 注意力
网络”. 在: 会议论文集 的  24th ACM SIGKDD 国际
会议 在 知识 发现 & 数据 挖掘. KDD ’18. 新
York, NY, USA: 协会 用于 计算中 Machinery. 2299–2308.
isbn: 9781450355520. doi: 10.1145/3219819.3220048. url: https:
//doi.org/10.1145/3219819.3220048.

Van Gysel, C., M. de Rijke, 和 E. Kanoulas. 2016. “学习 潜在
向量 Spaces 用于 产品搜索”. 在: 会议论文集 的  25th
ACM 国际 在 会议 在 信息 和 知识
管理. CIKM ’16. Indianapolis, Indiana, USA: ACM. 165–
174. isbn: 978-1-4503-4073-1. doi: 10.1145/2983323.2983702. url:
http://doi.acm.org/10.1145/2983323.2983702.

Van Gysel, C., M. de Rijke, 和 E. Kanoulas. 2017. “Structural Regularities 在 文本-基于 实体 向量 Spaces”. 在: 会议论文集 的
 ACM SIGIR 国际 会议 在 理论 的 信息检索. ICTIR ’17. Amsterdam,  Netherlands: ACM.
3–10. isbn: 978-1-4503-4490-6. doi: 10.1145/3121050.3121066. url:
http://doi.acm.org/10.1145/3121050.3121066.

Van Gysel, C., M. de Rijke, 和 E. Kanoulas. 2018. “Mix ’N 匹配:
集成中 文本 匹配 和 乘积 Substitutability 之内
产品搜索”. 在: 会议论文集 的  27th ACM 国际
会议 在 信息 和 知识 管理. CIKM
’18. Torino, Italy: ACM. 1373–1382. isbn: 978-1-4503-6014-2. doi:
10 . 1145 / 3269206 . 3271668. url: http : / / doi . acm . org / 10 . 1145 /
3269206.3271668.

References

179

Van Gysel, C., M. de Rijke, 和 M. Worring. 2016b. “无监督,
Eﬃcient 和 语义 专长 检索”. 在: 会议论文集 的 
25th 国际 会议 在 世界 广泛 Web. WWW ’16.
Montr&#233;al, Qu&#233;bec, Canada. 1069–1079. isbn: 978-1-
4503-4143-1. doi: 10.1145/2872427.2882974. url: https://doi.org/
10.1145/2872427.2882974.

Vaswani, ., N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, . N. Gomez,
L. Kaiser, 和 I. Polosukhin. 2017. “注意力 是 所有 你 需求”.
NIPS’17: 6000–6010.

Vincent, P., H. Larochelle, Y. Bengio, 和 P.-. Manzagol. 2008. “Extracting 和 Composing 鲁棒 特征 用 Denoising Autoencoders”. 在: 会议论文集 的  25th 国际 会议 在
机器学习. ICML ’08. Helsinki, Finland: ACM. 1096–1103.
isbn: 978-1-60558-205-4. doi: 10.1145/1390156.1390294. url: http:
//doi.acm.org/10.1145/1390156.1390294.

Wan, S., Y. Lan, J. Guo, J. Xu, L. Pang, 和 X. Cheng. 2016. “
深度 架构 用于 语义匹配 用 多个 Positional
句子 表示”. 在: 会议论文集 的  Thirtieth AAAI
会议 在 Artiﬁcial 智能. AAAI’16. Phoenix, Arizona:
AAAI Press. 2835–2841. url: http://dl.acm.org/引用.cfm?id=
3016100.3016298.

Wan, S., Y. Lan, J. Xu, J. Guo, L. Pang, 和 X. Cheng. 2016b. “MatchSRNN: 建模  递归 匹配 结构 用 Spatial
RNN”. 在: 会议论文集 的  二十-第五 国际 联合 会议 在 Artiﬁcial 智能. IJCAI’16. 新 York, 新 York,
USA: AAAI Press. 2922–2928. isbn: 978-1-57735-770-4. url: http:
//dl.acm.org/引用.cfm?id=3060832.3061030.

Wang, B., Y. Yang, X. Xu, . Hanjalic, 和 H. T. Shen. 2017. “对抗 交叉-模态 检索”. 在: 会议论文集 的  25th ACM
国际 会议 在 Multimedia. MM ’17. Mountain 视角,
California, USA: ACM. 154–162. isbn: 978-1-4503-4906-2. doi: 10.
1145/3123266.3123326. url: http://doi.acm.org/10.1145/3123266.
3123326.

180

References

Wang, H., F. Zhang, J. Wang, M. Zhao, W. Li, X. Xie, 和 M. Guo.
2018. “RippleNet: Propagating 用户 Preferences 在  知识
图 用于 推荐 系统”. 在: 会议论文集 的  27th ACM
国际 会议 在 信息 和 知识 管理. 新 York, NY, USA: 协会 用于 计算中 Machinery.
417–426. isbn: 9781450360142. doi: 10.1145/3269206.3271739. url:
https://doi.org/10.1145/3269206.3271739.

Wang, J., . P. de Vries, 和 M. J. T. Reinders. 2006. “统一
用户-基于 和 物品-基于 协同过滤 方法 通过
相似度 融合”. 在: 会议论文集 的  29th Annual 国际 ACM SIGIR 会议 在 研究 和 开发 在
信息检索. SIGIR ’06. Seattle, Washington, USA: ACM.
501–508. isbn: 1-59593-369-7. doi: 10.1145/1148170.1148257. url:
http://doi.acm.org/10.1145/1148170.1148257.

Wang, L., Y. Li, J. Huang, 和 S. Lazebnik. 2018b. “学习 TwoBranch 神经 网络 用于 图像-文本 匹配 任务”. IEEE
Transactions 在 模式 分析 和 机器 智能: 1–1.
issn: 0162-8828. doi: 10.1109/TPAMI.2018.2797921.

Wang, L., Y. Li, 和 S. Lazebnik. 2016. “学习 深度 StructurePreserving 图像-文本 Embeddings”. 在: 2016 IEEE 会议 在
计算机 视觉 和 模式 识别 (CVPR). Vol. 00. 5005–
5013. doi: 10.1109/CVPR.2016.541. url: doi.ieeecomputersociety.
org/10.1109/CVPR.2016.541.

Wang, X., X. 他, Y. Cao, M. Liu, 和 T. Chua. 2019. “KGAT: 知识图谱注意力网络 用于 推荐”. 在: 会议论文集 的  25th ACM SIGKDD 国际 会议 在
知识 发现 & 数据 挖掘, KDD 2019, Anchorage, AK,
USA, August 4-8, 2019. 950–958. doi: 10.1145/3292500.3330989.
url: https://doi.org/10.1145/3292500.3330989.

Wang, X., X. 他, F. Feng, L. Nie, 和 T. Chua. 2018c. “TEM: Treeenhanced 嵌入 模型 用于 可解释 推荐”. 在:
会议论文集 的  2018 世界 广泛 Web 会议 在 世界
广泛 Web. 1543–1552. doi: 10.1145/3178876.3186066. url: https:
//doi.org/10.1145/3178876.3186066.

References

181

Wang, X., X. 他, L. Nie, 和 T.-S. Chua. 2017b. “物品 Silk Road:
Recommending 物品 从 信息 Domains  Social 用户”.
在: 会议论文集 的  40th 国际 ACM SIGIR 会议
在 研究 和 开发 在 信息检索. SIGIR ’17.
Shinjuku, Tokyo, Japan: ACM. 185–194. isbn: 978-1-4503-5022-8.
doi: 10.1145/3077136.3080771. url: http://doi.acm.org/10.1145/
3077136.3080771.

Wang, X., X. 他, M. Wang, F. Feng, 和 T.-S. Chua. 2019b. “神经
图 协同过滤”. 在: 会议论文集 的  42nd 国际 ACM SIGIR 会议 在 研究 和 开发
在 信息检索. SIGIR’19. Paris, France: 协会 用于
计算中 Machinery. 165–174. isbn: 9781450361729. doi: 10.1145/
3331184.3331267. url: https://doi.org/10.1145/3331184.3331267.
Wang, X., D. Wang, C. Xu, X. 他, Y. Cao, 和 T. Chua. 2019c. “可解释 推理 超过 知识 Graphs 用于 推荐”.
在:  三十-第三 AAAI 会议 在 Artiﬁcial 智能,
AAAI 2019. 5329–5336. url: https ://aaai.org/ ojs/索引. php/
AAAI/文章/视角/4470.

Wang, X., Q. Huang, . Celikyilmaz, J. Gao, D. Shen, Y.-F. Wang,
W. Y. Wang, 和 L. Zhang. 2019d. “Reinforced 交叉-模态 匹配 和 自身-监督 Imitation 学习 用于 视觉-语言
Navigation”. 在:  IEEE 会议 在 计算机 视觉 和
模式 识别 (CVPR).

Wang, Z., W. Hamza, 和 R. Florian. 2017c. “Bilateral 多-视角
匹配 用于 自然 语言 句子”. 在: 会议论文集 的 
二十-第六 国际 联合 会议 在 Artiﬁcial 智能, IJCAI-17. 4144–4150. doi: 10.24963/ijcai.2017/579. url:
https://doi.org/10.24963/ijcai.2017/579.

Wang, Z., G. Xu, H. Li, 和 M. Zhang. 2011. “ 快 和 准确
方法 用于 近似 字符串 搜索”. 在: 会议论文集 的  49th
Annual Meeting 的  协会 用于 计算 语言学:
人类 语言 Technologies - 卷 1. HLT ’11. Portland,
Oregon: 协会 用于 计算 语言学. 52–61. isbn:
978-1-932432-87-9. url: http : / / dl . acm . org / 引用 . cfm ? id =
2002472.2002480.

182

References

Wei, X. 和 W. B. Croft. 2006. “LDA-基于 文档 模型 用于 adhoc 检索”. 在: 会议论文集 的  29th annual 国际 ACM
SIGIR 会议 在 研究 和 开发 在 信息
检索. SIGIR ’06. Seattle, Washington, USA: ACM. 178–185.
isbn: 1-59593-369-7. doi: 10 . 1145 / 1148170 . 1148204. url: http :
//doi.acm.org/10.1145/1148170.1148204.

Wei, Y., X. Wang, L. Nie, X. 他, R. Hong, 和 T.-S. Chua. 2019.
“MMGCN: 多-模态 图 卷积 网络 用于 Personalized 推荐 的 Micro-视频”. 在: 会议论文集 的  27th
ACM 国际 会议 在 Multimedia. MM ’19. Nice, France:
ACM. 1437–1445. isbn: 978-1-4503-6889-6. doi: 10.1145/3343031.
3351034. url: http://doi.acm.org/10.1145/3343031.3351034.

Wu, B., X. 他, Z. Sun, L. Chen, 和 Y. Ye. 2019. “ATM:  Attentive 翻译 模型 用于 下一物品推荐”. IEEE
Transactions 在 工业 Informatics: 1–1. issn: 1941-0050. doi:
10.1109/TII.2019.2947174.

Wu, C.-Y., . Ahmed, . Beutel, . J. Smola, 和 H. Jing. 2017.
“Recurrent 推荐 网络”. 在: 会议论文集 的  第十
ACM 国际 会议 在 网页搜索 和 数据 挖掘.
新 York, NY, USA. 495–503. doi: 10.1145/3018661.3018689. url:
https://doi.org/10.1145/3018661.3018689.

Wu, L., P. Sun, Y. Fu, R. Hong, X. Wang, 和 M. Wang. 2019b. “
神经 在ﬂuence Diﬀusion 模型 用于 Social 推荐”. 在:
会议论文集 的  42nd 国际 ACM SIGIR 会议 在
研究 和 开发 在 信息检索, SIGIR 2019,
Paris, France, July 21-25, 2019. 235–244. doi: 10.1145/3331184.
3331214. url: https://doi.org/10.1145/3331184.3331214.

Wu, Q., H. Wang, Q. Gu, 和 H. Wang. 2016. “Contextual Bandits
在  协同 环境”. 在: 会议论文集 的  39th 国际 ACM SIGIR 会议 在 研究 和 开发 在
信息检索. SIGIR ’16. Pisa, Italy. 529–538. doi: 10.1145/
2911451.2911528. url: https://doi.org/10.1145/2911451.2911528.

References

183

Wu, S., Y. Tang, Y. Zhu, L. Wang, X. Xie, 和 T. Tan. 2019c. “SessionBased 推荐 用 图 神经 网络”. 在: 
三十-第三 AAAI 会议 在 Artiﬁcial 智能, AAAI
2019, Honolulu, Hawaii, USA, 2019. 346–353. doi: 10.1609/aaai.
v33i01.3301346. url: https://doi.org/10.1609/aaai.v33i01.3301346.
Wu, W., H. Li, 和 J. Xu. 2013. “学习 查询 和 文档
相似之处 从 点击-通过 Bipartite 图 用 Metadata”.
在: 会议论文集 的  第六 ACM 国际 会议 在 Web
搜索 和 数据 挖掘. WSDM ’13. Rome, Italy: ACM. 687–
696. isbn: 978-1-4503-1869-3. doi: 10.1145/2433396.2433481. url:
http://doi.acm.org/10.1145/2433396.2433481.

Wu, W., Z. Lu, 和 H. Li. 2013b. “学习 双线性 模型 用于 匹配
查询 和 文档”. J. Mach. 学习. Res. 14(1): 2519–2548.
issn: 1532-4435. url: http://dl.acm.org/引用.cfm?id=2567709.
2567742.

Wu, Y., C. DuBois, . X. Zheng, 和 M. Ester. 2016b. “协同
Denoising Auto-Encoders 用于 Top-N 推荐 系统”. 在:
会议论文集 的  第九 ACM 国际 会议 在 Web
搜索 和 数据 挖掘. WSDM ’16. San Francisco, California, USA:
ACM. 153–162. isbn: 978-1-4503-3716-8. doi: 10.1145/2835776.
2835837. url: http://doi.acm.org/10.1145/2835776.2835837.

Xiao, J., H. Ye, X. 他, H. Zhang, F. Wu, 和 T.-S. Chua. 2017. “Attentional Factorization Machines: 学习  权重 的 特征
Interactions 通过 注意力 网络”. 在: 会议论文集 的  26th
国际 联合 会议 在 Artiﬁcial 智能. IJCAI’17.
Melbourne, Australia: AAAI Press. 3119–3125. isbn: 978-0-9992411-
0-3. url: http://dl.acm.org/引用.cfm?id=3172077.3172324.
Xin, X., B. Chen, X. 他, D. Wang, Y. Ding, 和 J. Jose. 2019. “CFM:
卷积 Factorization Machines 用于 上下文-Aware 推荐”. 在: 会议论文集 的  28th 国际 联合 会议
在 Artiﬁcial 智能. IJCAI’19.

184

References

Xin, X., X. 他, Y. Zhang, Y. Zhang, 和 J. M. Jose. 2019b. “Relational 协同过滤: 建模 多个 物品 Relations 用于
推荐”. 在: 会议论文集 的  42nd 国际 ACM
SIGIR 会议 在 研究 和 开发 在 信息
检索, SIGIR 2019, Paris, France, July 21-25, 2019. 125–134.
doi: 10.1145/3331184.3331188. url: https://doi.org/10.1145/
3331184.3331188.

Xiong, C., Z. Dai, J. Callan, Z. Liu, 和 R. 幂. 2017. “端--端
神经 Ad-hoc 排序 用 核池化”. 在: 会议论文集 的
 40th 国际 ACM SIGIR 会议 在 研究 和
开发 在 信息检索. SIGIR ’17. Shinjuku, Tokyo,
Japan: ACM. 55–64. isbn: 978-1-4503-5022-8. doi: 10.1145/3077136.
3080809. url: http://doi.acm.org/10.1145/3077136.3080809.
Xue, F., X. 他, X. Wang, J. Xu, K. Liu, 和 R. Hong. 2019. “深度
物品-基于 协同过滤 用于 Top-N 推荐”.
ACM Trans. Inf. Syst. 37(3). issn: 1046-8188. doi: 10.1145/3314578.
url: https://doi.org/10.1145/3314578.

Xue, H.-J., X. Dai, J. Zhang, S. Huang, 和 J. Chen. 2017. “深度 矩阵
Factorization 模型 用于 推荐 系统”. 在: 会议论文集
的  二十-第六 国际 联合 会议 在 Artiﬁcial
智能, IJCAI-17. 3203–3209. doi: 10.24963/ijcai.2017/447.
url: https://doi.org/10.24963/ijcai.2017/447.

Yan, F. 和 K. Mikolajczyk. 2015. “深度 相关 用于 匹配
images 和 文本”. 在: 2015 IEEE 会议 在 计算机 视觉
和 模式 识别 (CVPR). 3441–3450. doi: 10.1109/CVPR.
2015.7298966.

Yang, L., Q. Ai, J. Guo, 和 W. B. Croft. 2016. “aNMM: 排序
短 答案 文本 用 注意力-基于 神经 匹配模型”.
在: 会议论文集 的  25th ACM 国际 在 会议 在
信息 和 知识 管理. CIKM ’16. Indianapolis,
Indiana, USA: ACM. 287–296. isbn: 978-1-4503-4073-1. doi: 10.
1145/2983323.2983818. url: http://doi.acm.org/10.1145/2983323.
2983818.

References

185

Yang, P., H. Fang, 和 J. Lin. 2018. “Anserini: Reproducible 排序
Baselines 使用中 Lucene”. J. 数据 和 信息 质量. 10(4):
16:1–16:20. issn: 1936-1955. doi: 10 . 1145 / 3239571. url: http :
//doi.acm.org/10.1145/3239571.

Yang, R., J. Zhang, X. Gao, F. Ji, 和 H. Chen. 2019. “简单 和
Eﬀective 文本 匹配 用 Richer 对齐 特征”. 在: 会议论文集 的  57th Annual Meeting 的  协会 用于 计算 语言学. Florence, Italy: 协会 用于 计算
语言学. 4699–4709. doi: 10.18653/v1/P19- 1465. url: https:
//www.aclweb.org/anthology/P19-1465.

Yang, W., H. Zhang, 和 J. Lin. 2019b. “简单 应用 的 BERT
用于 Ad Hoc 文档检索”. CoRR. abs/1903.10972. arXiv:
1903.10972. url: http://arxiv.org/abs/1903.10972.

Yang, Y., S. W.-t. Yih, 和 C. Meek. 2015. “WikiQA:  挑战
数据集 用于 Open-领域 问答”. 在: 会议论文集 的
 2015 会议 在 经验的 方法 在 自然 语言
Processing. ACL - 协会 用于 计算 语言学. url:
https://www.microsoft.com/en-us/研究/publication/wikiqa-achallenge-dataset-for-open-domain-question-answering/.

Yang, Z., Z. Dai, Y. Yang, J. Carbonell, R. R. Salakhutdinov, 和
Q. V. Le. 2019c. “XLNet: 广义 Autoregressive Pretraining
用于 语言 理解”. 在: 进展 在 神经 信息
Processing 系统 32. Curran Associates, Inc. 5753–5763. url: http:
//papers.nips.cc/论文/8812- xlnet- 广义- autoregressivepretraining-用于-语言-理解.pdf.

Yi, X. 和 J. Allan. 2009. “ Comparative 研究 的 利用中 主题
模型 用于 信息检索”. 在: 会议论文集 的  31th European 会议 在 IR 研究 在 进展 在 信息
检索. ECIR ’09. Toulouse, France: Springer-Verlag. 29–41. isbn:
978-3-642-00957-0. doi: 10 . 1007 / 978 - 3 - 642 - 00958 - 7 _ 6. url:
http://dx.doi.org/10.1007/978-3-642-00958-7_6.

186

References

Yin, W. 和 H. Schütze. 2015. “MultiGranCNN:  架构 用于
通用 匹配 的 文本 块 在 多个 Levels 的 粒度”. 在: 会议论文集 的  53rd Annual Meeting 的  协会
用于 计算 语言学 和  7th 国际 联合 会议 在 自然语言处理 (卷 1: 长 Papers).
Beijing, China: 协会 用于 计算 语言学. 63–73.
doi: 10 . 3115 / v1 / P15 - 1007. url: https : / / www . aclweb . org /
anthology/P15-1007.

Yin, W., H. Schütze, B. Xiang, 和 B. Zhou. 2016. “ABCNN: AttentionBased 卷积神经网络 用于 建模 句子 Pairs”.
Transactions 的  协会 用于 计算 语言学. 4:
259–272. doi: 10.1162/tacl\_\_00097. url: https://doi.org/10.
1162/tacl__00097.

Ying, R., R. 他, K. Chen, P. Eksombatchai, W. L. Hamilton, 和 J.
Leskovec. 2018. “图 卷积 神经 网络 用于 WebScale 推荐 系统”. 在: 会议论文集 的  24th ACM
SIGKDD 国际 会议 在 知识 发现 & 数据
挖掘. KDD ’18. London, United Kingdom: ACM. 974–983. isbn:
978-1-4503-5552-0. doi: 10.1145/3219819.3219890. url: http://doi.
acm.org/10.1145/3219819.3219890.

Yuan, F., X. 他, H. Jiang, G. Guo, J. Xiong, Z. Xu, 和 Y. Xiong. 2020.
“未来 数据 Helps 训练: 建模 未来 Contexts 用于 SessionBased 推荐”. 在: 会议论文集 的  Web 会议
2020. WWW ’20. Taipei, Taiwan: 协会 用于 计算中 Machinery. 303–313. isbn: 9781450370233. doi: 10.1145/3366423.3380116.
url: https://doi.org/10.1145/3366423.3380116.

Yuan, F., . Karatzoglou, I. Arapakis, J. M. Jose, 和 X. 他. 2019.
“ 简单 卷积 生成 网络 用于 下一个 物品 推荐”. 在: 会议论文集 的  Twelfth ACM 国际
会议 在 网页搜索 和 数据 挖掘. WSDM ’19. Melbourne VIC, Australia: 协会 用于 计算中 Machinery. 582–
590. isbn: 9781450359405. doi: 10.1145/3289600.3290975. url:
https://doi.org/10.1145/3289600.3290975.

References

187

Zamani, H. 和 W. B. Croft. 2016. “Estimating 嵌入 Vectors 用于
查询”. 在: 会议论文集 的  2016 ACM 国际 会议 在  理论 的 信息检索. ICTIR ’16. Newark,
Delaware, USA: 协会 用于 计算中 Machinery. 123–132.
isbn: 9781450344975. doi: 10.1145/2970398.2970403. url: https:
//doi.org/10.1145/2970398.2970403.

Zamani, H. 和 W. B. Croft. 2017. “相关性-基于 词嵌入”.
在: 会议论文集 的  40th 国际 ACM SIGIR 会议
在 研究 和 开发 在 信息检索. SIGIR ’17.
Shinjuku, Tokyo, Japan: 协会 用于 计算中 Machinery.
505–514. isbn: 9781450350228. doi: 10.1145/3077136.3080831. url:
https://doi.org/10.1145/3077136.3080831.

Zamani, H. 和 W. B. Croft. 2018. “联合 建模 和 优化
的 搜索与推荐”. 在: 会议论文集 的  第一 Biennial 会议 在 设计 的 实验的 搜索 & 信息
检索 系统. DESIRES ’18. Bertinoro, Italy: CEUR-WS. 36–
41. url: http://ceur-ws.org/Vol-2167/论文2.pdf.

Zamani, H. 和 W. B. Croft. 2018b. “在  理论 的 弱监督
用于 信息检索”. 在: 会议论文集 的  2018 ACM SIGIR
国际 会议 在 理论 的 信息检索. ICTIR
’18. Tianjin, China: 协会 用于 计算中 Machinery. 147–154.
isbn: 9781450356565. doi: 10.1145/3234944.3234968. url: https:
//doi.org/10.1145/3234944.3234968.

Zamani, H. 和 W. B. Croft. 2020. “学习  联合 搜索与推荐 模型 从 用户-物品 Interactions”. 在: 会议论文集 的
 13th 国际 会议 在 网页搜索 和 数据 挖掘.
WSDM ’20. Houston, TX, USA: 协会 用于 计算中 Machinery. 717–725. isbn: 9781450368223. doi: 10.1145/3336191.3371818.
url: https://doi.org/10.1145/3336191.3371818.

Zamani, H., W. B. Croft, 和 J. S. Culpepper. 2018. “神经 查询
性能 预测 使用中 弱监督 从 多个
Signals”. 在:  41st 国际 ACM SIGIR 会议 在
研究 和 开发 在 信息检索. SIGIR ’18.
Ann Arbor, MI, USA: 协会 用于 计算中 Machinery. 105–
114. isbn: 9781450356572. doi: 10 . 1145 / 3209978 . 3210041. url:
https://doi.org/10.1145/3209978.3210041.

188

References

Zamani, H., J. Dadashkarimi, . Shakery, 和 W. B. Croft. 2016.
“Pseudo-相关性 反馈 基于 在 矩阵分解”. 在:
会议论文集 的  25th ACM 国际 在 会议 在 信息 和 知识 管理. CIKM ’16. Indianapolis,
Indiana, USA: ACM. 1483–1492. isbn: 978-1-4503-4073-1. doi: 10.
1145/2983323.2983844. url: http://doi.acm.org/10.1145/2983323.
2983844.

Zamani, H., M. Dehghani, W. B. Croft, E. 学会了-Miller, 和 J. Kamps.
2018b. “从 神经 Re-排序  神经 排序: 学习 
稀疏 表示 用于 Inverted 索引”. 在: 会议论文集 的 
27th ACM 国际 会议 在 信息 和 知识
管理. CIKM ’18. Torino, Italy: 协会 用于 计算中
Machinery. 497–506. isbn: 9781450360142. doi: 10.1145/3269206.
3271800. url: https://doi.org/10.1145/3269206.3271800.

Zamani, H., B. Mitra, X. Song, N. Craswell, 和 S. Tiwary. 2018c. “神经 排序 模型 用 多个 文档 Fields”. 在: 会议论文集
的  Eleventh ACM 国际 会议 在 网页搜索 和
数据 挖掘. WSDM ’18. Marina Del Rey, CA, USA: ACM. 700–
708. isbn: 978-1-4503-5581-0. doi: 10.1145/3159652.3159730. url:
http://doi.acm.org/10.1145/3159652.3159730.

Zhang, S., L. Yao, 和 . Sun. 2017. “深度学习 基于 推荐 系统:  综述 和 新 Perspectives”. CoRR. abs/1707.07435.
arXiv: 1707.07435. url: http://arxiv.org/abs/1707.07435.

Zhang, S., L. Yao, . Sun, 和 Y. Tay. 2019. “深度学习 基于
推荐 系统:  综述 和 新 Perspectives”. ACM
Comput. Surv. 52(1). issn: 0360-0300. doi: 10.1145/3285029. url:
https://doi.org/10.1145/3285029.

Zhang, X., H. Xie, H. Li, 和 J. C.S. Lui. 2020. “Conversational Contextual Bandit: 算法 和 应用”. 在: 会议论文集 的 
Web 会议 2020. WWW ’20. Taipei, Taiwan: 协会 用于
计算中 Machinery. 662–672. isbn: 9781450370233. doi: 10.1145/
3366423.3380148. url: https://doi.org/10.1145/3366423.3380148.

References

189

Zhang, Y., Q. Ai, X. Chen, 和 W. B. Croft. 2017b. “联合 表示学习 用于 Top-N 推荐 用 异质
信息 Sources”. 在: 会议论文集 的  2017 ACM 在 会议 在 信息 和 知识 管理. CIKM ’17.
Singapore, Singapore: ACM. 1449–1458. isbn: 978-1-4503-4918-5.
doi: 10.1145/3132847.3132892. url: http://doi.acm.org/10.1145/
3132847.3132892.

Zhang, Y., X. Chen, Q. Ai, L. Yang, 和 W. B. Croft. 2018. “朝向
Conversational 搜索与推荐: 系统 Ask, 用户
Respond”. 在: 会议论文集 的  27th ACM 国际 会议
在 信息 和 知识 管理. CIKM ’18. Torino,
Italy: ACM. 177–186. isbn: 978-1-4503-6014-2. doi: 10.1145/3269206.
3271776. url: http://doi.acm.org/10.1145/3269206.3271776.

Zhao, W. X., G. 他, K. Yang, H. Dou, J. Huang, S. Ouyang, 和 J.
Wen. 2019. “KB4Rec:  数据 集合 用于 Linking 知识 基于
用 推荐 系统”. 数据 智能. 1(2): 121–136. doi:
10.1162/dint\_\_00008. url: https://doi.org/10.1162/dint__
00008.

Zheng, L., C. Lu, F. Jiang, J. Zhang, 和 P. S. Yu. 2018. “Spectral
协同 过滤ing”. 在: 会议论文集 的  12th ACM 会议
在 推荐 系统, RecSys 2018, Vancouver, BC, Canada,
October 2-7, 2018. 311–319. doi: 10.1145/3240323.3240343. url:
https://doi.org/10.1145/3240323.3240343.

Zheng, L., V. Noroozi, 和 P. S. Yu. 2017. “联合 深度 建模 的 用户
和 物品 使用中 综述 用于 推荐”. 在: 会议论文集
的  第十 ACM 国际 会议 在 网页搜索 和
数据 挖掘. WSDM ’17. Cambridge, United Kingdom: ACM. 425–
434. isbn: 978-1-4503-4675-7. doi: 10.1145/3018661.3018665. url:
http://doi.acm.org/10.1145/3018661.3018665.

Zheng, Y., Z. Fan, Y. Liu, C. Luo, M. Zhang, 和 S. Ma. 2018b. “SogouQCL:  新 数据集 用 点击 相关性 标签”. 在:  41st
国际 ACM SIGIR 会议 在 研究 & 开发
在 信息检索. SIGIR ’18. Ann Arbor, MI, USA: ACM.
1117–1120. isbn: 978-1-4503-5657-2. doi: 10.1145/3209978.3210092.
url: http://doi.acm.org/10.1145/3209978.3210092.

190

References

Zhou, G., X. Zhu, C. Song, Y. Fan, H. Zhu, X. Ma, Y. Yan, J. Jin, H. Li,
和 K. Gai. 2018. “深度 兴趣 网络 用于 点击率
预测”. 在: 会议论文集 的  24th ACM SIGKDD 国际
会议 在 知识 发现 & 数据 挖掘. KDD ’18.
London, United Kingdom: 协会 用于 计算中 Machinery.
1059–1068. isbn: 9781450355520. doi: 10.1145/3219819.3219823.
url: https://doi.org/10.1145/3219819.3219823.

Zhu, M., . Ahuja, W. Wei, 和 C. K. Reddy. 2019. “ 层次
注意力 检索 模型 用于 Healthcare 问答”.
在:  世界 广泛 Web 会议. WWW ’19. San Francisco,
CA, USA: 协会 用于 计算中 Machinery. 2472–2482. isbn:
9781450366748. doi: 10.1145/3308558.3313699. url: https://doi.
org/10.1145/3308558.3313699.

References

列表 的 Acronyms

PLS

偏最小二乘

RMLS 正则化潜在空间匹配

191

SSI

监督语义索引

BMF 偏置矩阵分解

FISM 因子化物品相似度模型

FM 分解机

FFN Feedforward 神经网络

MLP Multilayer Perceptron

CNN 卷积 神经 网络

RNN Recurrent 神经 网络

GAN 生成 对抗 网络

AE

Autoencoders

DAE Denoising Autoencoders

CBOW 连续词袋模型

SG

跳字模型

BERT 来自Transformer的双向编码器表示

DSSM 深度 Structured 语义 模型

CLSM 卷积潜在语义模型

CNTN 卷积 神经 张量 网络

LSTM-RNN Recurrent 神经 网络 用 长 短-术语

记忆 cells

NVSM 神经 向量空间模型

192

References

SNRM Standalone 神经 排序模型

ACMR 对抗 交叉 模态 检索

ARC-II 卷积 匹配模型 II

DRMM 深度 相关性 匹配模型

K-NRM 核 基于 神经 排序模型

DeepMF 深度 矩阵分解

CDAE 协同 Denoising Auto-编码器

NAIS 神经 Attentive 物品 相似度

NARM 神经 Attentive 推荐 机器

DeepCoNN 深度 Cooperative 神经 网络

NARRE 神经 注意力 回归 用 综述-水平 解释

VBPR Visual 贝叶斯个性化排序

CDL Comparative 深度学习

ACF Attentive 协同过滤

NGCF 神经 图 协同过滤

KGAT 知识图谱注意力网络

KPRN 知识 路径 Recurrent 网络

NCF 神经协同过滤

ConvNCF 卷积 神经协同过滤

GMF 广义 矩阵分解

NeuMF 神经 矩阵分解

CML 协同 度量 学习

References

193

TransRec 翻译-基于 推荐

LRML 潜在 Relational 度量 学习

NFM 神经 分解机

AFM Attentional 分解机

