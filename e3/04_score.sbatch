#!/usr/bin/env bash
#SBATCH -J e3-score
#SBATCH -N1 --ntasks-per-node=1 --cpus-per-task=24 --mem=64G
#SBATCH -t 03:00:00
#SBATCH -o logs/e3-score-%j.out
# CPU job. Submit from <repo>/e3 after all 6 generation tasks finish:
#   sbatch -A $PACE_ACCOUNT -p cpu-small -q $QOS 04_score.sbatch     (SMOKE=1 for the smoke run)
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
source ./config.sh
module load anaconda3 2>/dev/null || true
eval "$(conda shell.bash hook)"; conda activate "$CONDA_ENV"
cd "$REPO"
W=${SLURM_CPUS_PER_TASK:-8}
data=$EVAL_DIR; gen=$RESULTS; scored=$SCORED; gold=$GOLD
if [ "${SMOKE:-0}" = 1 ]; then
  data=$REPO/eval/e3_smoke; gen=$REPO/results/e3_smoke; scored=$REPO/results/e3_smoke_graph; gold=$REPO/results/e3_smoke_gold
fi

python scripts/build_gold_dependency_graphs.py "$data" "$gold"
python scripts/score_dependency_graphs.py "$gold" "$gen" --results-dir "$scored" --workers "$W"
python scripts/aggregate_graph_pass_results.py "$scored" --workers "$W" --overwrite \
  --pass-at 1 --pass-at 2 --pass-at 4 --pass-at 8 --pass-at 16 --pass-at 32 --pass-at 64
report=$REPO/results/e3_report; extra=""
[ "${SMOKE:-0}" = 1 ] && { report=$REPO/results/e3_smoke_report; extra="--smoke"; }
python e3/05_analyze.py --gen "$gen" --scored "$scored" --out "$report" --k "$([ "${SMOKE:-0}" = 1 ] && echo 4 || echo $SAMPLE_K)" --max-new-tokens "$MAX_NEW_TOKENS" $extra
