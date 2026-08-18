# CS224d: Deep Learning for Natural Language Processing

> 斯坦福大学（Stanford University）| 2016 春季学期 | 讲师：Richard Socher

> [!NOTE]
>
> https://cs224d.stanford.edu/
>
> 本课程 2017 年冬季（Winter 2017）版本请参见 [cs224n.stanford.edu](http://cs224n.stanford.edu)。



## 课程描述

自然语言处理（Natural Language Processing，NLP）是信息时代最重要的技术之一。理解复杂的语言表达也是人工智能的关键组成部分。NLP 的应用无处不在，因为人们几乎用语言交流一切：网页搜索、广告、电子邮件、客户服务、语言翻译、放射学报告等等。支撑 NLP 应用的有大量底层任务和机器学习模型。近年来，深度学习方法在许多不同的 NLP 任务上都取得了非常高的性能。这些模型通常可以用一个单一的端到端模型进行训练，不需要传统的、针对特定任务的特征工程。在本春季学期课程中，学生将学习实现、训练、调试、可视化和发明自己的神经网络模型。本课程将深入探讨应用于 NLP 的深度学习前沿研究。期末项目将涉及训练一个复杂的循环神经网络（Recurrent Neural Network，RNN），并将其应用于一个大规模的 NLP 问题。在模型方面，我们将涵盖词向量表示、基于窗口的神经网络、循环神经网络、长短期记忆（Long Short-Term Memory，LSTM）模型、递归神经网络（Recursive Neural Network）、卷积神经网络（Convolutional Neural Network，CNN），以及一些涉及记忆组件的全新模型。通过讲座和编程作业，学生将学习使神经网络在实践问题上工作的必要工程技巧。

### 往届项目报告

- [2015 报告](reports_2015.html)
- [2016 报告](reports_2016.html)

## 课程讲师

[Richard Socher](http://socher.org)

## 助教

- [James Hong](https://www.linkedin.com/in/jameshong1993)
- [Sameep Bagadia](https://www.linkedin.com/in/sameep-bagadia-47a40699)
- [David Dindi](https://www.linkedin.com/in/david-dindi-2a532a48)
- [B. Ramsundar](http://web.stanford.edu/~rbharath/)
- [N. Arivazhagan](https://www.linkedin.com/in/naveenariva)
- [Qiaojing Yan](https://www.linkedin.com/in/qiaojing-yan-4534b9112)

- [详细教学大纲（含材料）](syllabus.html)
- [Piazza 论坛](https://piazza.com/class/ilx0v32x8ce7dh)

## 上课时间与地点

2016 春季学期（3 月 - 6 月）
讲座：周二、周四 3:00-4:20
地点：[Gates B1](http://www-cs.stanford.edu/about/gates-computer-science-building)

## 答疑时间（Office Hours）

**Richard**：周二 4:30-6:30pm，Huang Basement
（用于研究和项目讨论）

助教：
- **David**：周一 6:00-8:00pm，Huang 138
- **Bharath**：周二 1:00-3:00pm，Huang Basement
- **James**：周三 5:30-7:30pm，Gates B26
- **Sameep**：周四 12:45-2:45pm，Gates B21
- **Naveen**：周五 1:00-3:00pm，Huang Basement
- **Qiaojing**：周日 4:00-6:00pm，Gates B24

## 评分政策

- 作业 #1：15%
- 作业 #2：15%
- 作业 #3：15%
- 期中考试：15%
- 期末项目：40%

## 课程讨论

- 斯坦福学生：[Piazza](https://piazza.com/class/ilx0v32x8ce7dh)（面向斯坦福学生）
- 在线讨论：[Reddit 群组](http://www.reddit.com/r/CS224d)（面向非斯坦福学生）
- 我们的 Twitter 账号：[@CS224d](https://twitter.com/cs224d)

## 作业详情

关于如何提交作业的更多细节，请参见[作业页面](assignments.html)。

## 课程项目详情

关于课程项目的更多细节，请参见[项目页面](project.html)。

## 先修要求

- **熟练掌握 Python**
  所有课程作业将使用 Python（以及 numpy）。这里有一个[教程](http://cs231n.github.io/python-numpy-tutorial/)供不太熟悉 Python 的同学使用。如果你有很多编程经验但使用的是其他语言（如 C/C++/Matlab/Javascript），你大概也没问题。

- **大学微积分、线性代数（如 MATH 19 或 41、MATH 51）**
  你应该能熟练地求导，并理解矩阵向量运算及其记号。

- **基础概率与统计（如 CS 109 或其他统计课程）**
  你应该了解概率、高斯分布、均值、标准差等基础知识。

- **与 CS229（机器学习）相当的知识**
  我们将构建代价函数、求导并使用梯度下降进行优化。

## 推荐（非必需）

- **自然语言处理知识（CS224N 或 CS224U）**
  我们将讨论很多不同的任务，如果你知道在这些任务上已经做了多少工作、相关模型是如何解决它们的，你会更加欣赏深度学习技术的力量。

- **凸优化**
  有了这个背景，你可能会觉得一些优化技巧更直观。

- **卷积神经网络知识（CS231n）**
  第一套问题集对你来说可能会更容易。我们不能假设你上过这门课，所以会有大约 3 节讲座内容重叠。你可以利用那段时间更深入地探索一些方面。

## 常见问题（FAQ）

**这是这门课第一次开设吗？**

这是本课程的第二次开设。本课程旨在向学生介绍自然语言处理中的深度学习。我们将特别强调神经网络，这是最近在许多不同的 NLP 任务上都取得改进的一类深度学习模型。

**我可以从外部跟进学习吗？**

如果你加入我们，我们会很高兴！我们计划让课程材料广泛可用：**作业、课程笔记和幻灯片将在网上提供。** 我们可能会提供视频。但我们无法给你课程学分。

**我可以按通过/不通过（credit/no credit）方式选这门课吗？**

可以。凡本应获得 C- 或以上成绩的学生都将获得学分。

**我可以旁听吗？**

一般来说，如果你是斯坦福社区的成员（注册学生、工作人员和/或教员），我们非常欢迎旁听者。出于礼貌，我们希望你首先给我们发邮件，或者在你参加的第一节课后与讲师交谈。

**期末项目可以小组合作吗？**

可以，最多两人一组。

**我对课程有疑问，联系课程组的最佳方式是什么？**

斯坦福学生请使用 Piazza 上的内部课程论坛，这样其他学生也能从你的问题和我们的回答中受益。如果你有私人事务，请发送邮件到班级邮件列表**将很快添加**。

**我可以把期末项目与其他课程合并吗？**

可以。与 CS224d 同期开设的几门课程都是自然的选择，例如 CS224u（自然语言理解，由 Chris Potts 教授和 Bill MacCartney 教授授课）。如果你在上相关的课程，请与讲师沟通，获得合并期末项目作业的许可。

**作为 SCPD 学生，我如何弥补海报展示部分？**

对于期末海报展示，你可以通过 YouTube 提交关于你项目的视频。

**作为 SCPD 学生，我如何参加期中考试？**

对于期中考试，我们可以使用标准的 SCPD 流程，由你的经理或你公司的人员在考试期间监督你。

**SCPD 学生会有虚拟答疑时间吗？**

所有答疑时间都可以通过 Google Hangouts 访问。Hangout 的链接在 Piazza 上。

---

> 网页设计：Andrej Karpathy
