# E3 — Mid-training vs RL under a fixed compute budget

## Data checks
- base: 102400 samples, 1600 problems {11: 200, 12: 200, 13: 200, 14: 200, 15: 200, 16: 200, 18: 200, 20: 200}; problems with n!=k: 0; samples not in graph results (parse fail / no gold match): 0 (0.0%)
- full_midtrain: 102400 samples, 1600 problems {11: 200, 12: 200, 13: 200, 14: 200, 15: 200, 16: 200, 18: 200, 20: 200}; problems with n!=k: 0; samples not in graph results (parse fail / no gold match): 0 (0.0%)
- light_rl_0.2: 102400 samples, 1600 problems {11: 200, 12: 200, 13: 200, 14: 200, 15: 200, 16: 200, 18: 200, 20: 200}; problems with n!=k: 0; samples not in graph results (parse fail / no gold match): 0 (0.0%)
- medium_rl_0.5: 102400 samples, 1600 problems {11: 200, 12: 200, 13: 200, 14: 200, 15: 200, 16: 200, 18: 200, 20: 200}; problems with n!=k: 0; samples not in graph results (parse fail / no gold match): 0 (0.0%)
- heavy_rl_0.8: 102400 samples, 1600 problems {11: 200, 12: 200, 13: 200, 14: 200, 15: 200, 16: 200, 18: 200, 20: 200}; problems with n!=k: 0; samples not in graph results (parse fail / no gold match): 0 (0.0%)
- full_rl: 102400 samples, 1600 problems {11: 200, 12: 200, 13: 200, 14: 200, 15: 200, 16: 200, 18: 200, 20: 200}; problems with n!=k: 0; samples not in graph results (parse fail / no gold match): 0 (0.0%)

## Band averages — strict pass@k (95% bootstrap CI over problems; 2000 reps)

| band | k | checkpoint | strict | 95% CI | answer-only | Δ vs base | Δ vs full mid-training |
|---|---|---|---|---|---|---|---|
| edge | 1 | base | 0.183 | [0.165, 0.202] | 0.465 | — | -0.376 [-0.401, -0.351] |
| edge | 1 | full mid-training | 0.559 | [0.530, 0.588] | 0.913 | +0.376 [+0.351, +0.402] | — |
| edge | 1 | light RL (0.2) | 0.565 | [0.536, 0.594] | 0.913 | +0.382 [+0.355, +0.408] | +0.006 [-0.001, +0.013] |
| edge | 1 | medium RL (0.5) | 0.567 | [0.535, 0.597] | 0.916 | +0.384 [+0.358, +0.411] | +0.008 [+0.000, +0.016] |
| edge | 1 | heavy RL (0.8) | 0.550 | [0.521, 0.580] | 0.896 | +0.367 [+0.341, +0.394] | -0.009 [-0.020, +0.002] |
| edge | 1 | full RL | 0.506 | [0.476, 0.535] | 0.865 | +0.323 [+0.298, +0.349] | -0.053 [-0.066, -0.041] |
| edge | 64 | base | 0.406 | [0.385, 0.427] | 0.696 | — | -0.516 [-0.540, -0.491] |
| edge | 64 | full mid-training | 0.922 | [0.904, 0.940] | 0.995 | +0.516 [+0.491, +0.540] | — |
| edge | 64 | light RL (0.2) | 0.914 | [0.894, 0.931] | 0.993 | +0.507 [+0.482, +0.534] | -0.009 [-0.026, +0.009] |
| edge | 64 | medium RL (0.5) | 0.894 | [0.873, 0.914] | 0.984 | +0.487 [+0.460, +0.514] | -0.029 [-0.048, -0.010] |
| edge | 64 | heavy RL (0.8) | 0.874 | [0.851, 0.895] | 0.984 | +0.468 [+0.440, +0.495] | -0.049 [-0.070, -0.030] |
| edge | 64 | full RL | 0.829 | [0.801, 0.854] | 0.964 | +0.422 [+0.394, +0.451] | -0.094 [-0.117, -0.070] |
| hard | 1 | base | 0.000 | [0.000, 0.000] | 0.160 | — | -0.086 [-0.102, -0.073] |
| hard | 1 | full mid-training | 0.086 | [0.073, 0.100] | 0.377 | +0.086 [+0.072, +0.101] | — |
| hard | 1 | light RL (0.2) | 0.097 | [0.082, 0.113] | 0.382 | +0.097 [+0.082, +0.114] | +0.011 [+0.003, +0.018] |
| hard | 1 | medium RL (0.5) | 0.095 | [0.080, 0.111] | 0.381 | +0.095 [+0.080, +0.112] | +0.009 [+0.001, +0.017] |
| hard | 1 | heavy RL (0.8) | 0.106 | [0.090, 0.123] | 0.388 | +0.106 [+0.090, +0.122] | +0.019 [+0.008, +0.030] |
| hard | 1 | full RL | 0.095 | [0.080, 0.111] | 0.371 | +0.095 [+0.079, +0.111] | +0.009 [-0.005, +0.022] |
| hard | 64 | base | 0.000 | [0.000, 0.000] | 0.380 | — | -0.220 [-0.241, -0.200] |
| hard | 64 | full mid-training | 0.220 | [0.199, 0.241] | 0.636 | +0.220 [+0.199, +0.241] | — |
| hard | 64 | light RL (0.2) | 0.241 | [0.219, 0.263] | 0.640 | +0.241 [+0.219, +0.263] | +0.021 [+0.003, +0.040] |
| hard | 64 | medium RL (0.5) | 0.245 | [0.223, 0.265] | 0.624 | +0.245 [+0.223, +0.268] | +0.025 [+0.003, +0.048] |
| hard | 64 | heavy RL (0.8) | 0.258 | [0.234, 0.281] | 0.654 | +0.258 [+0.235, +0.281] | +0.037 [+0.015, +0.060] |
| hard | 64 | full RL | 0.268 | [0.245, 0.291] | 0.662 | +0.268 [+0.242, +0.292] | +0.048 [+0.022, +0.072] |

## Per-op strict pass@1 / pass@64

| checkpoint | op11 | op12 | op13 | op14 | op15 | op16 | op18 | op20 |
|---|---|---|---|---|---|---|---|---|
| base | 0.49 / 0.90 | 0.24 / 0.66 | 0.00 / 0.06 | 0.00 / 0.01 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| full mid-training | 0.57 / 0.95 | 0.62 / 0.95 | 0.55 / 0.90 | 0.49 / 0.89 | 0.30 / 0.69 | 0.05 / 0.18 | 0.00 / 0.00 | 0.00 / 0.00 |
| light RL (0.2) | 0.59 / 0.94 | 0.63 / 0.96 | 0.55 / 0.89 | 0.49 / 0.86 | 0.31 / 0.70 | 0.08 / 0.26 | 0.00 / 0.00 | 0.00 / 0.00 |
| medium RL (0.5) | 0.59 / 0.93 | 0.63 / 0.93 | 0.56 / 0.86 | 0.49 / 0.85 | 0.31 / 0.71 | 0.07 / 0.26 | 0.00 / 0.01 | 0.00 / 0.00 |
| heavy RL (0.8) | 0.59 / 0.91 | 0.61 / 0.92 | 0.54 / 0.86 | 0.46 / 0.81 | 0.31 / 0.70 | 0.11 / 0.31 | 0.00 / 0.01 | 0.00 / 0.00 |
| full RL | 0.56 / 0.92 | 0.59 / 0.91 | 0.47 / 0.78 | 0.40 / 0.72 | 0.27 / 0.68 | 0.10 / 0.34 | 0.01 / 0.05 | 0.00 / 0.00 |

## Against the paper's predictions

| prediction | holds? | evidence |
|---|---|---|
| Edge: light RL has the best pass@1 | no | best = medium RL (0.5) (0.567); light RL 0.565 |
| Edge: full mid-training is close to light RL (pass@1) | yes | light RL − mid-training = +0.006 [-0.001, +0.013] |
| Hard: heavy RL has the best pass@1 | yes | best = heavy RL (0.8) (0.106); heavy RL 0.106 |
| Hard: heavy RL has the best pass@64 | no | best = full RL (0.268); heavy RL 0.258 |
| Hard: pass@1 rises with RL share (0 → 0.2 → 0.5 → 0.8) | no | 0.086 → 0.097 → 0.095 → 0.106 |

Paper caveat: at larger budgets full RL wins hard pass@128, so mid-training is not expected to win every metric.

Notes: pass@k uses the unbiased estimator over all n samples (the repo's aggregator instead uses the first k samples). Strict = every step and the answer correct; unscored samples (parse fail) count as wrong. Light RL starts from checkpoint-6579, not 0.8 × 7,740, so the budget split is approximate.
