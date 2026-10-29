# E3 on PACE — mid-training vs RL at a fixed compute budget

Copy this folder to `<Interplay-LM-Reasoning repo root>/e3/` on PACE. Edit the top four values in `config.sh` (account, partition, GRES, QOS). Then run the steps below.

```bash
cd <repo>/e3 && source config.sh && mkdir -p logs

bash 00_setup_env.sh                 # login node, once: eval-only conda env (vLLM 0.10.1.1)
bash 01_download.sh                  # login node: 6 checkpoints + 8 val files -> $SCRATCH/interplay
bash 02_prepare.sh                   # eval/e3 symlinks, resolves model dirs, checks data

# Smoke test, about 15 min: base checkpoint, op11 + op15, k=4. Prints samples/s and a projected runtime.
SMOKE=1 sbatch -A $PACE_ACCOUNT -p $GPU_PARTITION --gres=$GPU_GRES -q $QOS --array=0 -t 00:30:00 03_generate.sbatch
SMOKE=1 sbatch -A $PACE_ACCOUNT -p cpu-small -q $QOS -t 00:30:00 04_score.sbatch
cat ../results/e3_smoke_report/report.md     # 0 FATAL lines, low "not in graph results" share

# Full run: 6 GPU array tasks (one per checkpoint), then 1 CPU scoring job that waits for them
jid=$(sbatch --parsable -A $PACE_ACCOUNT -p $GPU_PARTITION --gres=$GPU_GRES -q $QOS 03_generate.sbatch)
sbatch -A $PACE_ACCOUNT -p cpu-small -q $QOS --dependency=afterok:$jid 04_score.sbatch
```

Outputs are written to `results/e3_report/`:
- `report.md`: band table with 95% CIs, per-op table and a check against each of the paper's predictions.
- `strict_pass1_by_op.png`, `strict_pass64_by_op.png` and `strict_pass_at_k_bands.png`.
- `per_op.csv`, `bands.csv` and `pass_at_k.csv`. These include answer-only scores, mean response length and the truncated share.

## How this differs from the plan, and why
- **One array task per checkpoint (1.5 h each), not one 3-hour job.** The full run is 614k samples. A 100M model on one GPU may not finish them all in 3 hours. Splitting the run also means a failed task only loses one checkpoint. `--resume` skips checkpoints that are already done.
- **Checkpoint paths are resolved, not passed directly.** The released RL runs store their weights at `rl/<run>/global_step_N/actor/huggingface`. Passing `rl/<run>` would find no model. `02_prepare.sh` locates the folder that holds `config.json` and writes it to `checkpoints.resolved.tsv`.
- **`--vllm-dtype bfloat16` for all six checkpoints.** With "auto", vLLM would run the fp32 checkpoints (base and mid-training) in fp16 but the RL checkpoints in bf16. Override this with `VLLM_DTYPE=auto` in `config.sh` if you want the repo's behaviour.
- **pass@k uses the unbiased estimator over all 64 samples, with bootstrap CIs over problems.** The repo's aggregator only counts the first k samples, so its pass@1 rests on one sample per problem. Its `results.json` is still produced so you can compare.

## Scorer pairing check (answers the plan's `__idx` question)
Generations carry `__idx` but no problem `id`. The scorer matches a generation to its gold problem by `(op, __idx mod rows in that op file)`. This is correct only when every op file has exactly 200 rows and each row has an `op` field. `02_prepare.sh` checks both. The analysis stops with an error if no samples were matched.

## Open questions from the plan that still need checking on PACE
- Was `rl/op11-14_uniform` trained for 200 steps, and with which reward? Check its `config.json`, or any training metadata in its folder.
- Is `checkpoint-7740` really full mid-training at the 200-step budget? The plan infers this from the README table (1,548 × 5 = 7,740).
