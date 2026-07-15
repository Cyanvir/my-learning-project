# 远程服务器配置检查

你要先确认服务器能不能跑 Vaccine，再决定是否直接提交实验。Vaccine 默认用 Llama2-7B，README 里说原实验使用 H100，至少建议 A100 40G 级别显卡。普通小显存 GPU 很可能会 OOM。

## 1. 登录服务器

在你本机终端执行：

```bash
ssh 用户名@服务器地址
```

如果学校服务器需要跳板机或 VPN，按学长给的方式登录。

## 2. 一键检查脚本

把这个文件传到服务器：

```text
learning/check_remote_server.sh
```

然后在服务器上执行：

```bash
bash check_remote_server.sh | tee server_check.log
```

把 `server_check.log` 发给我或学长看。

## 3. 你需要重点看什么

### GPU

重点看：

- GPU 型号：最好是 A100/H100，A40 也可能能跑部分脚本。
- 单卡显存：Llama2-7B 训练建议至少 40GB。
- GPU 是否空闲：`nvidia-smi` 里显存占用不能太高。

### CUDA 和 PyTorch

重点看：

- `nvidia-smi` 是否正常。
- Python 里 `torch.cuda.is_available()` 是否是 `True`。
- PyTorch CUDA 版本和驱动是否匹配。

### Python 环境

重点看是否已安装：

- `torch`
- `transformers`
- `datasets`
- `peft`
- `accelerate`
- `deepspeed`

如果没有，才需要用 `vaccine.yml` 或 `vaccine_pip.txt` 建环境。

### Slurm

README 和脚本默认用 `sbatch`，所以要看服务器是否有 Slurm：

- 有 Slurm：用 `sbatch script/.../*.sh`
- 没有 Slurm：可能要改成 `bash xxx.sh`，并手动指定 GPU

### 硬盘

至少要有几十 GB 空间，最好 100GB 以上。模型、缓存、checkpoint、生成结果都会占空间。

## 4. 最低判断标准

可以直接尝试跑完整 Vaccine 的条件：

- 有 A100 40G / A100 80G / H100，或学长确认可用的同等级 GPU。
- `nvidia-smi` 正常。
- `torch.cuda.is_available()` 是 `True`。
- 能访问或已经缓存 `meta-llama/Llama-2-7b-hf`。
- 有 Hugging Face token，并且已经获得 Llama2 权限。
- 有足够硬盘空间。

如果只有 24GB 显存，例如 RTX 3090/4090，不建议直接跑原脚本。可以先改小模型、小 batch 或只做 pipeline 测试。

## 5. 发给学长确认的话

可以这样问：

```text
我准备先检查服务器是否能跑 Vaccine。请问这台服务器是否有 A100/H100，是否用 Slurm 提交任务？我会先运行 check_remote_server.sh，确认 GPU、CUDA、PyTorch、磁盘、Hugging Face 访问和 Llama2 权限，再跑 p=0.1 的小测试。
```

