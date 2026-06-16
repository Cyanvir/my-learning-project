# 需要补的知识清单

你不用一次学完。按照“刚好能读懂这个项目”的顺序补。

## 第 0 层：工具基础

- PowerShell 基本命令：`Get-ChildItem`、`Get-Content`、`cd`
- Python 基本语法：函数、类、列表、字典、文件读写、命令行参数
- Git/GitHub 基本概念：仓库、commit、branch、README、issue
- Conda/pip：环境、依赖、版本冲突

最低目标：能看懂项目文件结构，能运行一个 Python 脚本，能知道报错大概来自哪个库。

## 第 1 层：机器学习基础

- 训练集、验证集、测试集
- 分类任务和准确率
- loss / objective function
- gradient / backpropagation
- optimizer / learning rate
- overfitting
- baseline / ablation

最低目标：知道训练不是“让模型背答案”，而是不断用 loss 的梯度更新参数。

## 第 2 层：深度学习和 PyTorch

- tensor、shape、batch
- `nn.Module`
- forward / backward
- `requires_grad`
- hook：在 forward 或 backward 中插入自定义逻辑
- GPU、CUDA、显存

最低目标：读 `trainer.py` 时知道为什么可以注册 hook、为什么要 `zero_grad()`。

## 第 3 层：NLP 和大语言模型

- tokenizer：文字如何变成 token id
- embedding：token id 如何变成向量
- Transformer：attention、MLP、layer
- causal language modeling：根据前文预测下一个 token
- prompt / instruction / response
- generation：模型如何生成文本

最低目标：理解 `train.py` 里为什么要把 `instruction + input + response` 拼起来训练。

## 第 4 层：LLM 微调

- SFT：supervised fine-tuning，监督微调
- LoRA：只训练低秩适配器，降低显存和训练成本
- PEFT：parameter-efficient fine-tuning
- Hugging Face `Trainer`
- `TrainingArguments`
- checkpoint

最低目标：知道这个项目大部分实验不是全量训练 Llama，而是训练 LoRA 权重。

## 第 5 层：安全对齐和有害微调

- alignment：让模型行为符合安全/有用/诚实等目标
- harmful prompt：诱导模型输出危险内容的提示词
- harmful fine-tuning：用有害数据继续训练，使模型安全性下降
- safety evaluation：评估模型输出是否危险
- threat model：攻击者能做什么，不能做什么

最低目标：能用自己的话解释“为什么用户微调是一个新的攻击面”。

## 第 6 层：Vaccine 方法

要理解的关键词：

- hidden embedding / hidden representation
- harmful embedding drift
- perturbation-aware alignment
- `rho`：扰动强度
- two-step training
- gradient-based perturbation

最低目标：能解释 `VaccineTrainer.training_step` 为什么要先 backward 一次，再 zero grad，再 backward 第二次。

## 推荐资料入口

- PyTorch Tutorials: https://docs.pytorch.org/tutorials/
- Hugging Face Transformers: https://huggingface.co/docs/transformers/index
- Hugging Face Datasets: https://huggingface.co/docs/datasets/index
- Hugging Face PEFT: https://huggingface.co/docs/peft/index
- Vaccine paper: https://arxiv.org/abs/2402.01109
- Vaccine GitHub: https://github.com/git-disl/Vaccine

