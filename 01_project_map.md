# Vaccine 项目地图

这个仓库是论文代码，不是普通课程项目。它的结构围绕“准备数据 -> 安全对齐 -> 有害微调攻击 -> 评估安全性和任务性能”展开。

## 根目录核心文件

| 文件 | 作用 | 你要学什么 |
|---|---|---|
| `README.md` | 作者给出的复现说明 | 先读，知道论文目标和运行方式 |
| `train.py` | 训练入口 | 参数解析、数据加载、模型加载、LoRA、Trainer 选择 |
| `trainer.py` | 自定义训练逻辑 | Vaccine 方法核心，重点文件 |
| `utils.py` | JSON 读写工具 | 小型工程工具函数 |
| `vaccine.yml` | conda 环境 | 依赖和 CUDA/PyTorch 环境 |
| `vaccine_pip.txt` | pip 依赖 | Transformers、Datasets、PEFT、DeepSpeed 等 |

## 数据和任务目录

| 目录 | 作用 |
|---|---|
| `sst2/` | 情感分类数据构造和评估，测试正常任务能力 |
| `gsm8k/` | 数学推理数据构造和评估，测试推理能力 |
| `ag_news/` | 新闻分类数据构造和评估，测试分类能力 |
| `alpaca/` | 指令跟随相关评估 |
| `poison/evaluation/` | 有害请求生成结果和安全性评估 |

## 脚本目录

| 目录 | 作用 |
|---|---|
| `script/alignment/` | 初始安全对齐：SFT、Vaccine、random Vaccine |
| `script/vaccine_finetune/` | 在 Vaccine 对齐模型上做后续微调攻击/任务微调 |
| `script/sft_finetune/` | 在普通 SFT 对齐模型上做同样微调，用作对照 |
| `script/no_alignment_finetune/` | 不做安全对齐时的微调 |
| `script/EWC_finetune/` | EWC 正则方法相关脚本 |
| `script/KL_finetune/` | KL 正则方法相关脚本 |

## 一条完整实验链路

以 README 里的 SST2 例子为例：

1. `script/alignment/Vaccine.sh 2`：用安全数据做 Vaccine alignment，`rho=2`。
2. 保存 LoRA 权重到 `ckpt/..._vaccine_2`。
3. `script/vaccine_finetune/sst2.sh 2 0.1 1000`：加载刚才的 Vaccine 权重，用混合数据继续微调，其中 10% 是 harmful data，总样本数 1000。
4. `poison/evaluation/pred.py`：对有害请求生成回答。
5. `poison/evaluation/eval_sentiment.py`：评估回答是否有害。
6. `sst2/pred_eval.py`：评估 SST2 正常任务性能。

普通 SFT 对照组走 `script/alignment/SFT.sh` 和 `script/sft_finetune/sst2.sh`，其他设置尽量保持一致。

## 最重要的代码路径

`train.py` 负责“搭舞台”：

- 解析命令行参数。
- 加载 Llama/OPT 等 causal language model。
- 加载 tokenizer。
- 准备 supervised fine-tuning 数据。
- 初始化或加载 LoRA。
- 根据 `--optimizer` 选择普通 `Trainer`、`VaccineTrainer`、`FITrainer` 或 `KLTrainer`。

`trainer.py` 负责“真正改训练算法”：

- `VaccineTrainer`：本项目核心方法。
- `RandomVaccineTrainer`：随机扰动对照。
- `FITrainer`：类似 EWC，用 Fisher 信息约束参数变化。
- `KLTrainer`：用 KL 正则让学生模型接近教师模型。

## 阅读优先级

先读：

1. `README.md`
2. `trainer.py` 里的 `VaccineTrainer`
3. `train.py` 里的 `train()`
4. `sst2/build_dataset.py`
5. `script/alignment/Vaccine.sh`
6. `script/vaccine_finetune/sst2.sh`

后读：

- `poison/evaluation/`
- `FITrainer`、`KLTrainer`
- 其他任务脚本

