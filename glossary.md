# 术语表

## Alignment

安全对齐。让模型行为更符合人类期望，例如拒绝危险请求、减少有害输出。

## SFT

Supervised Fine-Tuning，监督微调。给模型很多“输入 -> 理想输出”的样本，让模型学习生成目标答案。

## Harmful Fine-tuning

有害微调。用户用包含危险回答的数据继续训练模型，导致模型安全能力下降。

## LoRA

Low-Rank Adaptation，一种参数高效微调方法。它不训练全部大模型参数，而是在部分线性层旁边加小矩阵，只训练这些小矩阵。

## PEFT

Parameter-Efficient Fine-Tuning，参数高效微调。LoRA 是 PEFT 的一种。

## Tokenizer

分词器。把文本切成 token，并转换成数字 id。大模型不能直接处理字符串。

## Embedding

嵌入向量。把 token id 转成连续向量，模型后续层处理的是这些向量。

## Hidden Representation

隐藏表示。模型中间层输出的向量。Vaccine 关注的就是这类中间表示在有害微调后是否漂移。

## Attention

Transformer 的核心模块。它让每个 token 根据上下文关注其他 token。

## Gradient

梯度。告诉我们如果想让 loss 下降，参数应该往哪个方向更新。

## Backward Hook

反向传播 hook。可以在梯度流过某个模块时执行自定义代码。Vaccine 用它记录 hidden output 的梯度。

## Forward Hook

前向传播 hook。可以在模块输出时执行自定义代码。Vaccine 用它给 hidden output 加扰动。

## Perturbation

扰动。对向量加上的小变化。Vaccine 里扰动不是随机乱加，而是根据第一次 backward 得到的梯度方向构造。

## Rho

Vaccine 里的扰动强度超参数。越大，训练时模拟的 hidden drift 越强，但过大可能伤害正常能力。

## Baseline

对照方法。比如普通 SFT 是 Vaccine 的重要 baseline。

## Ablation

消融实验。去掉或改变方法中的某个组件，看结果如何变化，用来判断该组件是否真的有用。

## Poison Ratio

有害数据比例。比如 `0.1` 表示微调数据里 10% 是有害数据。

## Checkpoint

训练保存的模型或 LoRA 权重。这个项目里很多结果保存在 `ckpt/`。

