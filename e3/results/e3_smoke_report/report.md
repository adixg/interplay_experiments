# E3 — Mid-training vs RL under a fixed compute budget

**SMOKE RUN** (k is small; numbers are not meaningful).

## Data checks
- base: 1600 samples, 400 problems {11: 200, 15: 200}; problems with n!=k: 0; samples not in graph results (parse fail / no gold match): 0 (0.0%)

## Band averages — strict pass@k (95% bootstrap CI over problems; 2000 reps)

| band | k | checkpoint | strict | 95% CI | answer-only | Δ vs base | Δ vs full mid-training |
|---|---|---|---|---|---|---|---|
| edge | 1 | base | 0.490 | [0.429, 0.549] | 0.849 | — | — |
| edge | 4 | base | 0.655 | [0.590, 0.720] | 0.940 | — | — |
| hard | 1 | base | 0.000 | [0.000, 0.000] | 0.166 | — | — |
| hard | 4 | base | 0.000 | [0.000, 0.000] | 0.255 | — | — |

## Per-op strict pass@1 / pass@4

| checkpoint | op11 | op15 |
|---|---|---|
| base | 0.49 / 0.66 | 0.00 / 0.00 |

Notes: pass@k uses the unbiased estimator over all n samples (the repo's aggregator instead uses the first k samples). Strict = every step and the answer correct; unscored samples (parse fail) count as wrong. Light RL starts from checkpoint-6579, not 0.8 × 7,740, so the budget split is approximate.
