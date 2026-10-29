#!/usr/bin/env bash
# One-time: create a lean eval-only env (no verl / LLaMA-Factory needed). Run on a login node.
set -euo pipefail
source "$(dirname "$0")/config.sh"
module load anaconda3 2>/dev/null || true
if ! conda env list | grep -q "^$CONDA_ENV "; then
  conda create -y -n "$CONDA_ENV" python=3.12
fi
conda run -n "$CONDA_ENV" python -m pip install --upgrade pip
# Versions per the repo README: torch 2.7.1+cu128, vLLM 0.10.1.1 (vllm pulls a matching torch).
conda run -n "$CONDA_ENV" python -m pip install "vllm==0.10.1.1" transformers "huggingface_hub[cli]>=0.34" tqdm numpy matplotlib
conda run -n "$CONDA_ENV" python -c "import vllm, torch, transformers; print('vllm', vllm.__version__, 'torch', torch.__version__, 'transformers', transformers.__version__)"
