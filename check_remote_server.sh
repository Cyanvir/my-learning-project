#!/usr/bin/env bash
set -u

echo "========== BASIC =========="
date
hostname
whoami
pwd
uname -a

echo
echo "========== CPU / MEMORY =========="
if command -v lscpu >/dev/null 2>&1; then
  lscpu | sed -n '1,25p'
else
  echo "lscpu not found"
fi
free -h || true

echo
echo "========== DISK =========="
df -h .
df -h | sed -n '1,20p'

echo
echo "========== GPU =========="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
  echo
  nvidia-smi -L
  echo
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,driver_version,compute_cap --format=csv
else
  echo "nvidia-smi not found: NVIDIA GPU/driver may be unavailable."
fi

echo
echo "========== CUDA =========="
if command -v nvcc >/dev/null 2>&1; then
  nvcc --version
else
  echo "nvcc not found. This is OK if PyTorch ships its own CUDA runtime, but note it."
fi

echo
echo "========== SLURM =========="
if command -v sbatch >/dev/null 2>&1; then
  sbatch --version
  echo
  sinfo || true
else
  echo "sbatch not found: this server may not use Slurm."
fi

echo
echo "========== PYTHON =========="
which python || true
python --version || true
which pip || true
pip --version || true
if command -v conda >/dev/null 2>&1; then
  conda --version
  conda env list || true
else
  echo "conda not found"
fi

echo
echo "========== PYTORCH / ML PACKAGES =========="
python - <<'PY'
import importlib

packages = ["torch", "transformers", "datasets", "peft", "accelerate", "deepspeed", "bitsandbytes"]
for name in packages:
    try:
        mod = importlib.import_module(name)
        ver = getattr(mod, "__version__", "unknown")
        print(f"{name}: {ver}")
    except Exception as exc:
        print(f"{name}: NOT AVAILABLE ({type(exc).__name__}: {exc})")

try:
    import torch
    print("torch.cuda.is_available:", torch.cuda.is_available())
    print("torch.version.cuda:", torch.version.cuda)
    print("torch.backends.cuda.matmul.allow_tf32:", torch.backends.cuda.matmul.allow_tf32)
    if torch.cuda.is_available():
        print("torch.cuda.device_count:", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"gpu[{i}].name:", props.name)
            print(f"gpu[{i}].total_memory_GB:", round(props.total_memory / 1024**3, 2))
except Exception as exc:
    print("torch cuda check failed:", repr(exc))
PY

echo
echo "========== HUGGING FACE CONNECTIVITY =========="
if command -v curl >/dev/null 2>&1; then
  curl -I --max-time 10 https://huggingface.co 2>/dev/null | sed -n '1,5p' || echo "curl huggingface failed"
else
  echo "curl not found"
fi

echo
echo "========== VACCINE PROJECT FILES =========="
if [ -f README.md ] && [ -f train.py ] && [ -f trainer.py ]; then
  echo "Looks like current directory is the Vaccine project root."
else
  echo "Current directory may not be the Vaccine project root."
  echo "Expected README.md, train.py, trainer.py"
fi

if [ -f huggingface_token.txt ]; then
  echo "huggingface_token.txt exists. Do not print or share its content."
else
  echo "huggingface_token.txt not found."
fi

echo
echo "========== SUMMARY HINT =========="
echo "For original Vaccine reproduction with Llama2-7B, prefer A100/H100 and >=40GB GPU memory."
echo "If GPU memory is 24GB or lower, first run a small pipeline test or ask advisor for reduced settings."

