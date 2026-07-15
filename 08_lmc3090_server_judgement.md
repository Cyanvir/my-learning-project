# lmc-3090 服务器判断结果

这是根据你运行 `check_remote_server.sh` 得到的日志整理的判断。

## 1. 服务器配置结论

这台服务器：

- CPU：Intel Xeon Gold 6348，112 线程，足够。
- 内存：约 251 GB，足够。
- 硬盘：`/data` 约 15 TB，剩余 2.7 TB，足够。
- GPU：8 张 RTX 3090，每张 24 GB 显存。
- CUDA driver：正常，`nvidia-smi` 可用。
- GPU 当前空闲：日志里 8 张卡显存占用都是 0 MiB。
- Slurm：没有，`sbatch not found`。
- Python 环境：当前 shell 里 `python` 命令不可用，需要先修环境。

## 2. 能不能直接跑原始 Vaccine？

结论：**不建议直接用原脚本跑完整 Llama2-7B Vaccine 实验。**

原因：

1. 原脚本默认只用单卡：

   ```bash
   CUDA_VISIBLE_DEVICES=0 python train.py ...
   ```

   也就是只用一张 RTX 3090。

2. 每张 RTX 3090 只有 24 GB 显存。Vaccine README 里说原实验使用 H100，至少 A100 40G 更稳。

3. `train.py` 默认：

   ```python
   load_in_8bit=False
   torch_dtype=torch.float16
   ```

   这不是 8bit 量化训练。Llama2-7B + LoRA + batch size 5 + Vaccine 两次 backward，很容易 OOM。

4. 服务器没有 Slurm，所以 README 里的 `sbatch` 命令不能直接用，要改成 `bash` 或自己写后台命令。

所以这台服务器适合：

- 跑小模型 pipeline 测试。
- 尝试把 batch size 改小后跑 Llama2-7B 的局部实验。
- 多卡并行或模型切分实验，但需要改脚本/训练配置。

不适合：

- 不改参数，直接照 README 跑完整复现。

## 3. 当前最先要解决的问题：Python 环境

日志里出现：

```text
python: command not found
```

先在服务器执行：

```bash
which python
which python3
python3 --version
conda env list
```

再试：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate myenv
which python
python --version
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

如果 `myenv` 没有相关包，再考虑新建环境。

## 4. 进入项目根目录再检查

你刚才是在：

```bash
/data/home/zhaozihan
```

不是 Vaccine 项目根目录，所以脚本提示：

```text
Current directory may not be the Vaccine project root.
```

先找项目目录：

```bash
find /data/home/zhaozihan -maxdepth 3 -name train.py -o -name README.md
```

或者如果你知道目录名：

```bash
cd ~/Vaccine-main
ls
```

正确项目根目录应该能看到：

```text
README.md
train.py
trainer.py
script/
sst2/
poison/
```

然后再运行：

```bash
bash learning/check_remote_server.sh | tee server_check_in_project.log
```

## 5. 这台机器上的推荐复现策略

不要一开始跑完整表格。建议分三步：

### Step 1：只打通环境

目标：

- `python` 可用。
- `torch.cuda.is_available()` 是 `True`。
- `transformers/datasets/peft/accelerate` 可导入。
- 能进入 Vaccine 项目根目录。

### Step 2：跑 SST-2 数据准备

```bash
mkdir -p data
cd sst2
python build_dataset.py
cd ..
ls data
```

这一步不需要大模型，主要确认数据下载和 Python 包可用。

### Step 3：先做小模型 pipeline 测试

不要先用 Llama2-7B。可以先问学长是否允许用一个小模型测试 pipeline，例如：

```text
facebook/opt-125m
```

如果小模型能跑通 `alignment -> finetune -> HS/FA evaluation`，说明代码流程是通的。之后再想办法上 Llama2-7B。

## 6. 如果一定要在 3090 上尝试 Llama2-7B

需要和学长确认是否允许改参数。可能要改：

- `per_device_train_batch_size`: 从 `5` 改成 `1`
- `gradient_accumulation_steps`: 适当增大，比如 `5`
- `model_max_length`: 适当减小
- 尝试 `load_in_8bit=True` 或 4bit/QLoRA，但这会偏离原论文脚本
- 尝试多卡 `device_map="auto"`，但原脚本里 `CUDA_VISIBLE_DEVICES=0` 会限制只用一张卡，需要去掉或改成多卡

这些都属于“工程适配”，会影响复现一致性。改之前先问学长。

## 7. 建议发给学长的结论

可以直接发：

```text
我检查了 lmc-3090：机器有 8 张 RTX 3090，每张 24GB，CPU/内存/硬盘充足，GPU 当前空闲；但没有 Slurm，当前 python 环境也还没配好。Vaccine 原脚本默认单卡跑 Llama2-7B，README 推荐 H100 或至少 A100 40G，所以这台机器不适合不改参数直接跑完整原实验。

我建议先配好 conda 环境，跑 SST-2 数据准备和小模型 pipeline；如果要在 3090 上跑 Llama2-7B，需要确认是否允许改 batch size、model_max_length、量化或多卡设置。
```

