# Remote Environment Pip Conflict Fix

你在服务器执行：

```bash
pip install -r vaccine_pip.txt
```

遇到的冲突是：

```text
mkl-fft 1.3.1 depends on numpy<1.23.0
vaccine_pip.txt requests numpy==1.24.1
```

这不是你的操作错误。`mkl-fft` 和 `mkl-service` 更像 conda/MKL 环境里的包，本项目代码并不直接依赖它们。服务器上已经单独装好了 GPU 版 PyTorch，所以推荐跳过这两个冲突包。

## Recommended Fix

在远程服务器的 Vaccine 项目根目录执行：

```bash
cd /data/home/zhaozihan/Defence/Vaccine
cp vaccine_pip.txt vaccine_pip_original.txt
grep -v -E '^(mkl-fft|mkl-service)==' vaccine_pip.txt > vaccine_pip_3090.txt
pip install -r vaccine_pip_3090.txt
```

装完后检查：

```bash
python -c "import torch, transformers, datasets, peft, accelerate; print('basic ok')"
python -c "import deepspeed, bitsandbytes, wandb; print('optional ok')"
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

如果 `deepspeed` 或 `bitsandbytes` 检查失败，先不要慌。数据准备和小模型 pipeline 可能暂时不需要它们，但正式训练前要再处理。

## Next Step

如果关键包检查通过，准备 SST-2 数据：

```bash
mkdir -p data
cd sst2
python build_dataset.py
cd ..
ls -lh data
```

