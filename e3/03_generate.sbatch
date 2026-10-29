#!/usr/bin/env bash
#SBATCH -J e3-gen
#SBATCH -N1 --ntasks-per-node=1 --cpus-per-task=8 --mem=48G
#SBATCH -t 01:30:00
#SBATCH --array=0-5
#SBATCH -o logs/e3-gen-%A_%a.out
# Submit from <repo>/e3:
#   sbatch -A $PACE_ACCOUNT -p $GPU_PARTITION --gres=$GPU_GRES -q $QOS 03_generate.sbatch
# One array task per checkpoint (~102k samples each). SMOKE=1 -> base only, op11+op15, k=4.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
source ./config.sh
module load anaconda3 2>/dev/null || true
eval "$(conda shell.bash hook)"; conda activate "$CONDA_ENV"
cd "$REPO"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

task=${SLURM_ARRAY_TASK_ID:-0}
line=$(sed -n "$((task + 1))p" "$MANIFEST")
label=$(cut -f1 <<<"$line"); ckdir=$(cut -f2 <<<"$line")
[ -n "$label" ] || { echo "no manifest row $task"; exit 1; }

data_root="$EVAL_DIR"; k=$SAMPLE_K; out="$RESULTS/$label"
if [ "${SMOKE:-0}" = 1 ]; then
  [ "$task" = 0 ] || { echo "SMOKE runs task 0 only"; exit 0; }
  data_root="$REPO/eval/e3_smoke"; mkdir -p "$data_root"
  ln -sfn "$EVAL_DIR/op11-200.jsonl" "$data_root/"; ln -sfn "$EVAL_DIR/op15-200.jsonl" "$data_root/"
  k=4; out="$REPO/results/e3_smoke/$label"
fi
mkdir -p "$out"
echo "== $label  ckpt=$ckdir  k=$k  data=$data_root"

start=$(date +%s)
python scripts/eval_checkpoints.py \
  --checkpoints-root "$ckdir" --checkpoints-pattern none \
  --data-root "$data_root" --id-op-threshold "$ID_OP_THRESHOLD" \
  --gen-backend vllm --skip-loss --vllm-data-parallel-size 1 --vllm-dtype "$VLLM_DTYPE" \
  --sample-k "$k" --max-new-tokens "$MAX_NEW_TOKENS" \
  --temperature "$TEMPERATURE" --top-p "$TOP_P" \
  --output-dir "$out" --resume
secs=$(( $(date +%s) - start ))

n=$(cat "$out"/*_generations.jsonl | wc -l)
python - "$n" "$secs" "$k" <<'PY'
import sys
n, s, k = int(sys.argv[1]), max(int(sys.argv[2]), 1), int(sys.argv[3])
print(f"samples={n} wall={s}s rate={n/s:.1f} samples/s")
full = 8 * 200 * 64          # one checkpoint, full E3
est = full / (n / s) / 3600
print(f"projected full-E3 time per checkpoint: {est:.2f} h  (array limit 1.5 h; whole E3 ~{6*est:.1f} GPU-h)")
if est > 1.4: print("WARNING: raise -t, or drop an op file (keep k=64) per the plan.")
PY
