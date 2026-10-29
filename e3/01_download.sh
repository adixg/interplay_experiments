#!/usr/bin/env bash
# Run on a login node. Downloads only E3's 6 checkpoints + 8 val files to scratch.
set -euo pipefail
source "$(dirname "$0")/config.sh"
module load anaconda3 2>/dev/null || true
eval "$(conda shell.bash hook)"; conda activate "$CONDA_ENV"
mkdir -p "$IROOT/models" "$IROOT/data/composition"

echo "$CHECKPOINTS" | while IFS='|' read -r label repo sub; do
  [ -z "$label" ] && continue
  echo "== $label  ($repo/$sub)"
  hf download "Interplay-LM-Reasoning/$repo" --include "$sub/*" --local-dir "$IROOT/models/$repo"
done

includes=(); for op in $OPS; do includes+=("val/op$op-200.jsonl"); done
hf download Interplay-LM-Reasoning/composition --repo-type dataset --include "${includes[@]}" \
  --local-dir "$IROOT/data/composition"
du -sh "$IROOT/models"/* "$IROOT/data/composition"
