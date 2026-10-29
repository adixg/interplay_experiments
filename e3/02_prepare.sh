#!/usr/bin/env bash
# Builds eval/e3 (symlinks), resolves each checkpoint to the folder holding config.json,
# and sanity-checks data so strict scoring will pair generations with gold problems.
set -euo pipefail
source "$(dirname "$0")/config.sh"
module load anaconda3 2>/dev/null || true
eval "$(conda shell.bash hook)"; conda activate "$CONDA_ENV"

mkdir -p "$EVAL_DIR"
for op in $OPS; do
  src="$IROOT/data/composition/val/op$op-200.jsonl"
  [ -f "$src" ] || { echo "MISSING $src (run 01_download.sh)"; exit 1; }
  ln -sfn "$src" "$EVAL_DIR/"
done

# Data checks: 200 rows per file and an 'op' field matching the filename.
# The scorer matches generations by (op, __idx % rows_in_op_file), so both must hold.
python - "$EVAL_DIR" <<'PY'
import json, sys, re, pathlib
bad = False
for f in sorted(pathlib.Path(sys.argv[1]).glob("*.jsonl")):
    rows = [json.loads(l) for l in open(f) if l.strip()]
    op = int(re.search(r"op(\d+)", f.name).group(1))
    ops = {r.get("op") for r in rows}
    ok = len(rows) == 200 and ops <= {op, str(op)} and all(isinstance(r.get("solution"), str) for r in rows)
    bad |= not ok
    print(f"{'ok ' if ok else 'BAD'} {f.name}: rows={len(rows)} op_values={sorted(map(str, ops))} keys={sorted(rows[0])[:8]}")
sys.exit(1 if bad else 0)
PY

# Resolve each checkpoint. Released RL runs nest weights as rl/<run>/global_step_N/actor/huggingface.
: > "$MANIFEST"
echo "$CHECKPOINTS" | while IFS='|' read -r label repo sub; do
  [ -z "$label" ] && continue
  root="$IROOT/models/$repo/$sub"
  mapfile -t cands < <(find "$root" -name config.json -printf '%h\n' 2>/dev/null | sort -V)
  [ ${#cands[@]} -ge 1 ] || { echo "NO config.json under $root"; exit 1; }
  dir="${cands[-1]}"   # highest global_step if several
  [ ${#cands[@]} -gt 1 ] && echo "note: $label has ${#cands[@]} model dirs; using $dir"
  tok=$(ls "$dir" | grep -E '^tokenizer(\.json|_config\.json)$' | head -1 || true)
  [ -n "$tok" ] || echo "WARN: $label has no tokenizer files in $dir"
  dtype=$(python -c "import json;print(json.load(open('$dir/config.json')).get('torch_dtype'))")
  printf '%s\t%s\n' "$label" "$dir" >> "$MANIFEST"
  echo "$label -> $dir  (torch_dtype=$dtype, tokenizer=${tok:-none})"
done
echo "Wrote $MANIFEST"
