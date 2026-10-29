# E3 — mid-training vs RL under a fixed compute budget. Source this file.
# Edit the four SLURM/PATH values below for your PACE allocation.

export PACE_ACCOUNT="${PACE_ACCOUNT:-gts-CHANGEME}"     # e.g. gts-<pi-username>
export GPU_PARTITION="${GPU_PARTITION:-gpu-l40s}"       # or gpu-a100
export GPU_GRES="${GPU_GRES:-gpu:l40s:1}"               # or gpu:a100:1
export QOS="${QOS:-inferno}"                            # inferno (paid) or embers (free, preemptible)

export SCRATCH="${SCRATCH:-$HOME/scratch}"
export IROOT="$SCRATCH/interplay"
export REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"   # repo root (this folder lives at <repo>/e3)
export CONDA_ENV="${CONDA_ENV:-interplay-eval}"

# Fixed E3 settings (from the plan)
export OPS="11 12 13 14 15 16 18 20"
export ID_OP_THRESHOLD=14        # edge (11-14) = ID, hard (15-20) = OOD
export SAMPLE_K=64
export MAX_NEW_TOKENS=1024
export TEMPERATURE=0.7
export TOP_P=1.0
# vLLM "auto" would run fp32 checkpoints (base, mid-training) in fp16 but RL ones in bf16.
# Pin one dtype so precision is identical across all six checkpoints.
export VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"

export EVAL_DIR="$REPO/eval/e3"
export RESULTS="$REPO/results/e3"
export SCORED="$REPO/results/e3_graph"
export GOLD="$REPO/results/e3_gold"
export MANIFEST="$REPO/e3/checkpoints.resolved.tsv"

P=id2-10_0.2easy_0.3medium_0.5hard
# label | HF repo | subfolder to download
export CHECKPOINTS="
base|extrapolation_rl|$P/base
full_midtrain|extrapolation_midtrain|$P/midtrain/cpt0.2-uniform_0.8-11-14_plus/checkpoint-7740
light_rl_0.2|extrapolation_midtrain|$P/rl/cpt0.8-rl-op11-14_uniform-200step-0.2RL
medium_rl_0.5|extrapolation_midtrain|$P/rl/cpt0.5-rl-op11-14_uniform-200step-0.5RL
heavy_rl_0.8|extrapolation_midtrain|$P/rl/cpt0.2-rl-op11-14_uniform-200step-0.8RL
full_rl|extrapolation_rl|$P/rl/op11-14_uniform
"
