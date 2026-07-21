# ViT-An Image is Worth 16x16 Words-Transformers for Image Recognition at Scale

原文：https://arxiv.org/abs/2010.11929

[Submitted on 22 Oct 2020 ([v1](https://arxiv.org/abs/2010.11929v1)), last revised 3 Jun 2021 (this version, v2)]

代码：https://github.com/google-research/vision_transformer



While the Transformer architecture has become the de-facto standard for natural language processing tasks, its applications to computer vision remain limited. In vision, attention is either applied in conjunction with convolutional networks, or used to replace certain components of convolutional networks while keeping their overall structure in place. 

We show that this reliance on CNNs is not necessary and **a pure transformer applied directly to sequences of image patches** can perform very well on image classification tasks. When **pre-trained on large amounts of data** and transferred to multiple mid-sized or small image recognition benchmarks (ImageNet, CIFAR-100, VTAB, etc.), Vision Transformer (ViT) attains excellent results compared to state-of-the-art convolutional networks while requiring substantially fewer computational resources to train.

尽管 Transformer 架构已成为自然语言处理任务的事实标准，但它在计算机视觉中的应用仍然有限。在视觉领域，注意力机制要么与卷积网络结合使用，要么在保留卷积网络整体结构的同时替换其中的某些组件。我们表明对卷积神经网络（CNN）的这种依赖并非必要：**将纯 Transformer 直接应用于图像补丁序列**，在图像分类任务上也能取得很好的表现。在大量数据上进行预训练并迁移到多个中小型图像识别基准（ImageNet、CIFAR-100、VTAB 等）后，视觉 Transformer（ViT）与最先进的卷积网络相比取得了优异的结果，同时训练时所需的计算资源显著更少。
