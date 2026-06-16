# Vaccine 项目入门学习包

这个目录是给“大一信息安全 + 科研零基础”准备的学习入口。你不用一开始就跑完整实验，这个项目原始实验需要 A100/H100 级别显卡；真正该先做的是：读懂问题、读懂论文主张、读懂代码如何验证主张，然后再考虑复现和改进。

## 这个项目一句话

Vaccine 研究的是大语言模型安全对齐：模型本来经过安全对齐后会拒绝有害请求，但用户再拿少量有害数据微调，可能把安全性破坏掉。Vaccine 的思路是在对齐训练阶段给 hidden embedding 加“有意设计的扰动”，让模型学到对这种微调扰动更稳定的内部表示。

## 建议学习顺序

1. 读 `00_research_onboarding.md`：先知道科研入门不是“直接看懂论文”，而是按问题、假设、方法、实验、结论来拆。
2. 读 `01_project_map.md`：知道仓库每个文件夹负责什么。
3. 读 `02_prerequisites.md`：补齐必须知识，不要一次学太散。
4. 运行 `toy_vaccine_manual.py`：用一个小模型理解 Vaccine 的二次训练步骤。
5. 读 `03_code_reading_route.md`：按指定顺序看 `train.py` 和 `trainer.py`。
6. 用 `04_paper_reading_template.md` 做论文笔记。
7. 用 `05_experiment_plan.md` 理解真正复现实验需要哪些资源、指标和对照组。
8. 用 `99_learning_log.md` 记录每次学习输出。

## 你最终应该从这个文件夹学到什么

- 怎么把一个 GitHub 论文代码仓库拆成：论文问题、数据、训练、方法实现、评估、脚本、结果。
- 怎么读一个 LLM 安全方向项目，而不是陷在每一行代码里。
- 怎么理解 SFT、LoRA、Trainer、hidden embedding、gradient、perturbation、harmful fine-tuning 这些关键词。
- 怎么从“会跑代码”逐步走到“能提出一个小改进或小实验”。

## 第一次任务

在项目根目录运行：

```powershell
python learning\toy_vaccine_manual.py
```

然后打开 `99_learning_log.md`，回答里面的第 1 组问题。你不需要一次懂完，先把“不懂的词”记下来，下一轮我们逐个拆。

## 安全提醒

本仓库根目录有 `huggingface_token.txt`。这是访问模型的私密凭证，不要发给别人，不要截图，不要提交到公开 GitHub。做安全研究时，也要把有害提示词、模型输出和实验环境控制在学习/研究范围内。

