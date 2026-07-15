# Vaccine 复现任务书

这份文档专门对应你学长发来的 Lisa/同类防御复现结果表。你现在的任务不是“随便跑一下 Vaccine”，而是把 Vaccine 按同样实验格式复现出来，得到一张可以和 SFT baseline 对比的表。

## 1. 这个任务到底要干嘛

你要做的是：

1. 用原始 `Vaccine-main` 仓库跑出 Vaccine 方法的结果。
2. 同时跑普通 SFT baseline，作为对照组。
3. 在 SST-2 微调任务上，改变 harmful data ratio：`clean/p=0`、`p=0.05`、`p=0.1`、`p=0.2`、`p=0.3`、`p=0.4`。
4. 每个比例都记录两个指标：
   - `HS`：Harmful Score，有害输出比例，越低越好。
   - `FA`：Finetune Accuracy，SST-2 正常任务准确率，越高越好。
5. 最后整理成类似学长截图里的表格。

你截图里的表格是这种结构：

| Method | Metric | clean | p=0.05 | p=0.1 | p=0.2 | p=0.3 | p=0.4 |
|---|---|---:|---:|---:|---:|---:|---:|
| SFT | HS |  |  |  |  |  |  |
| SFT | FA |  |  |  |  |  |  |
| Vaccine | HS |  |  |  |  |  |  |
| Vaccine | FA |  |  |  |  |  |  |

注意：截图标题写的是 Lisa 复现结果，但表格里的方法名有 `SFT` 和 `BSO`。你现在要做的不是复现 Lisa，而是用同样表格方式复现 `Vaccine`。

## 2. 你应该期待什么现象

如果 Vaccine 复现成功，理想现象是：

- `Vaccine` 的 `HS` 比 `SFT` 更低，特别是在 harmful ratio 变大时更明显。
- `Vaccine` 的 `FA` 不应该比 `SFT` 差很多，最好接近或略高。
- 也就是说，Vaccine 希望做到：安全性更稳，同时正常任务能力基本保住。

不要只看一个数字。真正要比较的是：

- `p` 从小到大时，`HS` 增长速度是否更慢。
- `FA` 是否没有明显崩掉。

## 3. 和学习资料的对应关系

先看这些文件，不要急着跑完整实验：

1. `learning/01_project_map.md`：知道每个目录做什么。
2. `learning/03_code_reading_route.md`：知道 `train.py` 和 `trainer.py` 怎么读。
3. `learning/05_experiment_plan.md`：知道完整复现为什么需要高端 GPU。
4. `learning/toy_vaccine_manual.py`：理解 Vaccine 是“带 hidden perturbation 的训练”。

代码对应关系：

| 任务 | 文件 |
|---|---|
| Vaccine 核心方法 | `trainer.py` 的 `VaccineTrainer` |
| 训练入口 | `train.py` |
| 安全对齐脚本 | `script/alignment/Vaccine.sh`、`script/alignment/SFT.sh` |
| harmful fine-tuning 脚本 | `script/vaccine_finetune/sst2.sh`、`script/sft_finetune/sst2.sh` |
| HS 评估 | `poison/evaluation/eval_sentiment.py` |
| FA 评估 | `sst2/pred_eval.py` |

## 4. 跑实验前要确认的 6 件事

找学长确认这些，不然很容易跑错：

1. 用哪个基础模型：默认是 `meta-llama/Llama-2-7b-hf`。
2. 用哪个 GPU/集群：README 说原实验用 H100，至少 A100 40G 级别更稳。
3. `rho` 取多少：README 示例是 `rho=2`。
4. `sample_num` 取多少：截图表格说明常用 `1000`。
5. `clean` 是不是按 `p=0` 处理：从表格语境看，通常表示只用 SST-2 正常数据微调，不混 harmful 数据。
6. 是否必须完全用论文默认超参，还是允许按你们机器环境改 batch size、epoch、路径。

## 5. 准备数据

当前仓库脚本会把 SST-2 训练数据写到 `data/sst2.json`，所以先确保根目录有 `data` 文件夹。

在项目根目录执行：

```powershell
mkdir data
cd sst2
python build_dataset.py
cd ..
```

如果你还要复现 GSM8K 或 AG News，再运行：

```powershell
cd gsm8k
python build_dataset.py
cd ..

cd ag_news
python build_dataset.py
cd ..
```

本次截图对应的是 SST-2，所以第一阶段只需要 `sst2.json`。

## 6. 第一阶段：跑 alignment

alignment 阶段先训练两个模型：

- `SFT`：普通安全对齐 baseline。
- `Vaccine`：带 perturbation-aware alignment 的方法。

在集群上通常用 `sbatch`：

```bash
cd script/alignment
sbatch SFT.sh meta-llama/Llama-2-7b-hf
sbatch Vaccine.sh 2 meta-llama/Llama-2-7b-hf
```

输出 checkpoint 预计是：

```text
ckpt/Llama-2-7b-hf_sft
ckpt/Llama-2-7b-hf_vaccine_2
```

这里 `2` 是 `rho`。

如果没有 Slurm，而是在单机上跑，可以把 `sbatch` 换成 `bash`，但必须确认显存足够：

```bash
bash SFT.sh meta-llama/Llama-2-7b-hf
bash Vaccine.sh 2 meta-llama/Llama-2-7b-hf
```

## 7. 第二阶段：跑不同 harmful ratio 的 SST-2 fine-tuning

你需要跑这些比例：

```text
0, 0.05, 0.1, 0.2, 0.3, 0.4
```

### SFT baseline

进入：

```bash
cd script/sft_finetune
```

依次跑：

```bash
sbatch sst2.sh 0 1000 meta-llama/Llama-2-7b-hf
sbatch sst2.sh 0.05 1000 meta-llama/Llama-2-7b-hf
sbatch sst2.sh 0.1 1000 meta-llama/Llama-2-7b-hf
sbatch sst2.sh 0.2 1000 meta-llama/Llama-2-7b-hf
sbatch sst2.sh 0.3 1000 meta-llama/Llama-2-7b-hf
sbatch sst2.sh 0.4 1000 meta-llama/Llama-2-7b-hf
```

### Vaccine

进入：

```bash
cd script/vaccine_finetune
```

依次跑：

```bash
sbatch sst2.sh 2 0 1000 meta-llama/Llama-2-7b-hf
sbatch sst2.sh 2 0.05 1000 meta-llama/Llama-2-7b-hf
sbatch sst2.sh 2 0.1 1000 meta-llama/Llama-2-7b-hf
sbatch sst2.sh 2 0.2 1000 meta-llama/Llama-2-7b-hf
sbatch sst2.sh 2 0.3 1000 meta-llama/Llama-2-7b-hf
sbatch sst2.sh 2 0.4 1000 meta-llama/Llama-2-7b-hf
```

参数含义：

```text
sst2.sh 2 0.1 1000
        | |   |
        | |   sample_num，总样本数
        | poison_ratio，有害数据比例
        rho，Vaccine 扰动强度
```

## 8. 从哪里读结果

每个 fine-tuning 脚本都会跑两类评估。

### HS：Harmful Score

来自：

```text
poison/evaluation/eval_sentiment.py
```

输出文件一般像这样：

```text
data/poison/sst2/Llama-2-7b-hf_vaccine_f_2_0.1_1000_sentiment_eval.json
data/poison/sst2/Llama-2-7b-hf_sft_f_0.1_1000_sentiment_eval.json
```

文件最后会有类似：

```text
final score:xx.xx
```

这个 `xx.xx` 填到表里的 `HS`。

### FA：Finetune Accuracy

来自：

```text
sst2/pred_eval.py
```

输出文件一般像这样：

```text
data/sst2/Llama-2-7b-hf_vaccine_f_2_0.1_1000
data/sst2/Llama-2-7b-hf_sft_f_0.1_1000
```

文件最后会有类似：

```text
score=xx.xx
```

这个 `xx.xx` 填到表里的 `FA`。

## 9. 最终要交什么

建议你最后交这 4 个东西：

1. `results_summary.tsv`：机器可读表格。
2. `results_summary.md`：Markdown 结果表，方便发给学长看。
3. `run_notes.md`：记录模型、GPU、环境、命令、日期、失败尝试。
4. 必要的脚本改动说明：你改了哪些路径、batch size、数据路径，为什么改。

表格模板见：

```text
learning/vaccine_results_template.tsv
```

## 10. 新手复现时的正确节奏

不要一口气提交 14 个任务到集群。建议这样做：

1. 先只跑 `SFT alignment` 和 `Vaccine alignment`。
2. 然后只跑一个比例，例如 `p=0.1`。
3. 确认 `HS` 和 `FA` 两个结果文件都能正常生成。
4. 再批量跑剩下的比例。
5. 最后整理表格。

这叫先打通 pipeline，再扩大实验规模。科研复现里，这是很重要的习惯。

## 11. 你现在最该做的下一步

向学长确认：

```text
我准备按 SST-2, Llama2-7B, sample_num=1000, rho=2, harmful ratio = 0/0.05/0.1/0.2/0.3/0.4 复现 Vaccine。
clean 我先按 p=0 的 clean fine-tuning 处理。
最后整理 SFT 和 Vaccine 的 HS/FA 表格。
请问模型、数据路径和 GPU 队列是否按这个设置？
```

确认后，再开始改路径和跑第一个 `p=0.1` 小批次。

