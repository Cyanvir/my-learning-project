# 实验复现计划和注意事项

这个项目不能急着完整复现。README 里说明原实验使用 H100，至少需要 A100 40G 级别显卡。普通笔记本或消费级显卡很可能无法直接跑 Llama2-7B 的完整脚本。

## 复现实验的层级

## Level 0：不跑大模型，只理解算法

目标：

- 跑 `learning/toy_vaccine_manual.py`
- 解释普通训练和 Vaccine-style 训练差异
- 画出 `trainer.py` 的训练流程

适合现在做。

## Level 1：只做代码静态阅读

目标：

- 不下载大模型。
- 不跑 Hugging Face 数据集。
- 读懂 `train.py` 数据流和 `trainer.py` 方法流。
- 做一张“从脚本参数到 Trainer 的路径图”。

适合现在做。

## Level 2：小模型小数据验证

目标：

- 用比 Llama2-7B 小很多的 causal LM。
- 用几十条样本验证脚本流程。
- 不追求论文结果，只追求确认代码能跑通。

注意：

- 当前 `train.py` 的 LoRA `target_modules` 主要面向 Llama 类结构，小模型可能需要改 target module。
- CPU 或小显卡上跑 Transformers 训练会很慢。
- 这一步最好等你理解 `train.py` 后再做。

## Level 3：接近论文复现

目标：

- 申请 Llama2 权限。
- 配置 Hugging Face token。
- 准备 A100/H100 或学校集群。
- 按 `script/alignment/` 和 `script/*_finetune/` 跑完整实验。

这一步不是入门第一周的任务。

## 关键实验变量

| 变量 | 含义 | 在哪里出现 |
|---|---|---|
| `rho` | Vaccine 扰动强度 | `Vaccine.sh`、`trainer.py` |
| `poison_ratio` | 有害数据比例 | finetune 脚本、`SupervisedDataset` |
| `sample_num` | 微调样本总数 | finetune 脚本、`SupervisedDataset` |
| `benign_dataset` | 混入的正常任务数据 | finetune 脚本 |
| `lora_folder` | alignment 阶段保存的 LoRA 权重 | finetune 和 eval 脚本 |

## 关键指标

- 安全性：有害请求下模型是否输出危险内容。
- 正常任务能力：SST2、GSM8K、AG News 等任务分数。
- 鲁棒性：不同 `poison_ratio` 下安全性下降是否更慢。
- 代价：训练时间、显存、正常能力损失。

## 初学者不要踩的坑

- 不要把跑不动大模型理解为“我学不会”。资源不够和能力无关。
- 不要只看最终分数，要看对照组是否公平。
- 不要泄露 `huggingface_token.txt`。
- 不要在公开环境随意发布 harmful prompt 和模型危险输出。
- 不要直接改很多代码。先复制脚本或记录改动，保证能回到原始逻辑。

## 建议你和学长汇报的内容

第一次汇报不用说“我复现了论文”。可以说：

- 我理解了 harmful fine-tuning 的威胁模型。
- 我知道 Vaccine 的核心是在 hidden representation 上做梯度扰动训练。
- 我能指出核心代码在 `trainer.py` 的 `VaccineTrainer`。
- 我知道完整实验需要高端 GPU，所以我先做了 toy demo 和静态代码阅读。
- 我下一步想读懂 `train.py` 的数据构造和 LoRA 加载流程。

