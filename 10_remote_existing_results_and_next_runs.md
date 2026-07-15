# Remote Existing Results and Next Runs

根据远程服务器 `/data/home/zhaozihan/Defence/Vaccine` 的输出，当前状态如下。

## 1. 环境状态

远程 `vaccine` conda 环境已经基本可用：

```text
torch: 2.1.0+cu121
torch.cuda.is_available(): True
torch.cuda.device_count(): 8
basic packages: torch, transformers, datasets, peft, accelerate OK
```

数据也已经存在，不需要重新跑 `build_dataset.py`：

```text
data/sst2.json
data/ag_news.json
data/gsm8k.json
data/alpaca.json
```

## 2. 已有主要结果

远程已有 `results_summary.md` 和 `results_summary.tsv`。虽然终端里中文显示乱码，但关键数字能读。

### Base Alignment

| Method | HS |
|---|---:|
| SFT | 51.00 |
| Vaccine rho=2 | 39.40 |

### SST2, N=1000, Poison Ratio Sweep

已有 Vaccine：

| Method | p | HS | FA |
|---|---:|---:|---:|
| Vaccine | 0.00 | 40.00 | 94.60 |
| Vaccine | 0.01 | 39.80 | 94.60 |
| Vaccine | 0.05 | 43.60 | 94.60 |
| Vaccine | 0.10 | 49.80 | 94.80 |
| Vaccine | 0.20 | 65.20 | 94.80 |

已有 SFT：

| Method | p | HS | FA |
|---|---:|---:|---:|
| SFT | 0.10 | 60.60 | 94.40 |

### SST2, Vaccine p=0.05, Sample Size Sweep

| N | HS | FA |
|---:|---:|---:|
| 500 | 41.80 | 91.20 |
| 1000 | 43.60 | 94.60 |
| 1500 | 49.80 | 94.80 |
| 2000 | 55.60 | 94.80 |
| 2500 | 62.40 | 94.80 |

## 3. 对照学长截图还缺什么

如果目标是做类似下面的 SST2 表：

```text
Method    Metric    clean/p=0    p=0.05    p=0.1    p=0.2    p=0.3    p=0.4
SFT       HS
SFT       FA
Vaccine   HS
Vaccine   FA
```

当前缺口大概是：

- `Vaccine`: 缺 `p=0.3`、`p=0.4`
- `SFT`: 只找到 `p=0.1`，缺 `p=0`、`p=0.05`、`p=0.2`、`p=0.3`、`p=0.4`

注意：`base alignment SFT HS=51.00` 不一定等于 SST2 表里的 `clean/p=0`。最好向学长确认 clean 是用 base alignment，还是用 `poison_ratio=0` 的 SST2 fine-tuning。

## 4. 先检查结果文件是否真的缺失

在远程服务器运行：

```bash
cd /data/home/zhaozihan/Defence/Vaccine
for p in 0 0.05 0.1 0.2 0.3 0.4; do
  echo "=== p=$p ==="
  ls data/poison/sst2/*vaccine*f_2_${p}_1000* 2>/dev/null || true
  ls data/sst2/*vaccine*f_2_${p}_1000* 2>/dev/null || true
  ls data/poison/sst2/*sft*f_${p}_1000* 2>/dev/null || true
  ls data/sst2/*sft*f_${p}_1000* 2>/dev/null || true
done
```

## 5. 无 Slurm 时如何跑缺失任务

服务器没有 `sbatch`，所以用 `bash` 或 `nohup bash`。

建议先只跑一个缺失任务，例如 Vaccine `p=0.3`：

```bash
cd /data/home/zhaozihan/Defence/Vaccine
mkdir -p logs
cd script/vaccine_finetune
nohup bash sst2.sh 2 0.3 1000 meta-llama/Llama-2-7b-hf > ../../logs/vaccine_sst2_p03.log 2>&1 &
```

查看日志：

```bash
tail -f /data/home/zhaozihan/Defence/Vaccine/logs/vaccine_sst2_p03.log
```

查看 GPU：

```bash
nvidia-smi -l 5
```

如果这个任务能成功，再跑其他缺失比例。

## 6. 重要提醒

脚本内部写了：

```bash
CUDA_VISIBLE_DEVICES=0 python train.py ...
```

所以它默认会用 GPU 0。不要同时开多个任务，否则它们会抢同一张卡。想并行跑需要先改脚本里的 GPU 编号。

