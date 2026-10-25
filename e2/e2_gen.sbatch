#!/bin/bash
#SBATCH --job-name=e2-gen
#SBATCH --partition=ice-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -C "L40S|A100-80GB|A100-40GB|H100|H200"
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ~/scratch/conda-envs/interplay
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
cd ~/scratch/interplay/Interplay-LM-Reasoning
nvidia-smi --query-gpu=name,memory.total --format=csv
M=~/scratch/interplay/models/context_pretrain_2
RL=rl/contextzoo_0.5zoo_0.5teacher_process_strict
for pair in "seed0.1:0.999zoo_op2-20+0.001teacher_op2" "seed1:0.99zoo_op2-20+0.01teacher_op2" "seed10:0.9zoo_op2-20+0.1teacher_op2"; do
  label=${pair%%:*}; mix=${pair#*:}
  for kind in base rl; do
    ck=$M/$mix/base; [ $kind = rl ] && ck=$M/$mix/$RL
    out=results/e2/${label}_${kind}
    ls $out/*_metrics.json >/dev/null 2>&1 && { echo "skip $out"; continue; }
    echo "=== $label $kind  $(date)"
    python -u scripts/eval_checkpoints.py --checkpoints-root $ck --checkpoints-pattern none \
      --data-root eval/e2 --id-op-threshold 10 --gen-backend vllm --skip-loss \
      --sample-k 64 --max-new-tokens 1024 --output-dir $out
  done
done
echo "=== all done $(date)"
