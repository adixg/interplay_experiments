#!/usr/bin/env python3
"""E3 analysis: unbiased strict / answer-only pass@k per op and per band, bootstrap CIs,
paired differences, plots, and a check against the paper's predictions.

Reads   <gen>/<label>/*_generations.jsonl            (per-sample exact_match, resp_tokens)
        <scored>/<label>/*.graph_results.jsonl       (per-sample 'perfect' = strict)
Writes  <out>/report.md, per_op.csv, bands.csv, pass_at_k.csv, *.png
"""
from __future__ import annotations
import argparse, csv, json, math
from collections import defaultdict
from pathlib import Path
import numpy as np

LABELS = ["base", "full_midtrain", "light_rl_0.2", "medium_rl_0.5", "heavy_rl_0.8", "full_rl"]
NICE = {"base": "base", "full_midtrain": "full mid-training", "light_rl_0.2": "light RL (0.2)",
        "medium_rl_0.5": "medium RL (0.5)", "heavy_rl_0.8": "heavy RL (0.8)", "full_rl": "full RL"}
EDGE, HARD = range(11, 15), range(15, 21)


def pass_at_k(n: np.ndarray, c: np.ndarray, k: int) -> np.ndarray:
    """Unbiased estimator 1 - C(n-c,k)/C(n,k), vectorised over problems."""
    out = np.zeros(len(n), dtype=float)
    for i, (ni, ci) in enumerate(zip(n, c)):
        if ni < k:
            out[i] = np.nan
        elif ni - ci < k:
            out[i] = 1.0
        else:
            out[i] = 1.0 - np.prod(1.0 - k / np.arange(ni - ci + 1, ni + 1))
    return out


def jl(path: Path):
    with path.open() as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def load_label(gen_dir: Path, scored_dir: Path, rows_per_op: int, max_new: int):
    n = defaultdict(int); em = defaultdict(int); strict = defaultdict(int)
    trunc = defaultdict(int); toks = defaultdict(list); graph_rows = 0
    for f in sorted(gen_dir.glob("*_generations.jsonl")):
        for r in jl(f):
            key = (int(r["op"]), int(r["__idx"]) % rows_per_op)
            n[key] += 1; em[key] += int(r.get("exact_match", 0))
            t = int(r.get("resp_tokens", 0)); toks[key[0]].append(t); trunc[key[0]] += t >= max_new
    for f in sorted(scored_dir.rglob("*.graph_results.jsonl")):
        for r in jl(f):
            graph_rows += 1
            idx = r.get("index") if r.get("index") is not None else r.get("prompt_key_value")
            strict[(int(r["op"]), int(idx) % rows_per_op)] += bool(r.get("perfect"))
    probs = sorted(n)
    return {
        "keys": probs,
        "op": np.array([k[0] for k in probs]),
        "n": np.array([n[k] for k in probs]),
        "c_em": np.array([em[k] for k in probs]),
        "c_strict": np.array([strict.get(k, 0) for k in probs]),
        "graph_rows": graph_rows,
        "trunc": dict(trunc), "toks": {o: float(np.mean(v)) for o, v in toks.items()},
    }


def band_mask(ops, band):
    return np.isin(ops, list(band))


def boot_ci(per_op_vals: dict[int, np.ndarray], ops: list[int], reps: int, rng, other=None):
    """Stratified bootstrap over problems within each op; mean over the band's problems.
    If `other` is given (same problems, paired), returns CI of the difference."""
    est_parts, boots = [], np.zeros(reps)
    total = sum(len(per_op_vals[o]) for o in ops)
    for o in ops:
        v = per_op_vals[o] - (other[o] if other is not None else 0)
        est_parts.append(v.sum())
        idx = rng.integers(0, len(v), size=(reps, len(v)))
        boots += v[idx].sum(axis=1)
    est = sum(est_parts) / total; boots /= total
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return est, lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", type=Path, required=True)
    ap.add_argument("--scored", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--rows-per-op", type=int, default=200)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    ks = [k for k in (1, 2, 4, 8, 16, 32, 64) if k <= a.k]
    K = ks[-1]

    labels = [l for l in LABELS if (a.gen / l).is_dir()]
    data, problems, notes = {}, [], []
    for l in labels:
        d = load_label(a.gen / l, a.scored / l, a.rows_per_op, a.max_new_tokens)
        data[l] = d
        tot = int(d["n"].sum())
        per_op_counts = {int(o): int((d["op"] == o).sum()) for o in np.unique(d["op"])}
        bad_n = int((d["n"] != a.k).sum())
        lost = tot - d["graph_rows"]
        msg = (f"{l}: {tot} samples, {len(d['keys'])} problems {per_op_counts}; "
               f"problems with n!=k: {bad_n}; samples not in graph results (parse fail / no gold match): {lost} ({lost/max(tot,1):.1%})")
        notes.append(msg); print(msg)
        if d["graph_rows"] == 0:
            raise SystemExit(f"FATAL {l}: no graph results — gold matching failed; check op field / __idx pairing.")
        if lost / max(tot, 1) > 0.25:
            notes.append(f"WARNING {l}: >25% of samples unscored; strict scores are lower bounds. Inspect raw generations.")

    # Per-problem pass@k for every label/metric/k
    P = {l: {m: {k: pass_at_k(d["n"], d[f"c_{m}"], k) for k in ks} for m in ("strict", "em")} for l, d in data.items()}
    ops_all = sorted({int(o) for d in data.values() for o in np.unique(d["op"])})

    def per_op(l, m, k):
        return {o: P[l][m][k][data[l]["op"] == o] for o in ops_all}

    # per-op table
    with open(a.out / "per_op.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["checkpoint", "op", "metric", *[f"pass@{k}" for k in ks], "mean_resp_tokens", "truncated_share"])
        for l in labels:
            for o in ops_all:
                m_ = data[l]["op"] == o
                nsamp = data[l]["n"][m_].sum()
                for m in ("strict", "em"):
                    w.writerow([l, o, "strict" if m == "strict" else "answer_only",
                                *[f"{np.nanmean(P[l][m][k][m_]):.4f}" for k in ks],
                                f"{data[l]['toks'].get(o, 0):.0f}", f"{data[l]['trunc'].get(o, 0)/max(nsamp,1):.4f}"])

    # pass@k curves per band
    with open(a.out / "pass_at_k.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["checkpoint", "band", "metric", *[f"pass@{k}" for k in ks]])
        for l in labels:
            for bname, band in (("edge", EDGE), ("hard", HARD)):
                bm = band_mask(data[l]["op"], band)
                for m in ("strict", "em"):
                    w.writerow([l, bname, m, *[f"{np.nanmean(P[l][m][k][bm]):.4f}" for k in ks]])

    # band table with CIs + paired diffs
    rows = []
    for bname, band in (("edge", EDGE), ("hard", HARD)):
        bops = [o for o in ops_all if o in band]
        if not bops:
            continue
        for k in (1, K):
            for l in labels:
                est, lo, hi = boot_ci(per_op(l, "strict", k), bops, a.bootstrap, rng)
                row = {"band": bname, "k": k, "checkpoint": l, "strict": est, "lo": lo, "hi": hi}
                em_est = np.nanmean(np.concatenate([per_op(l, "em", k)[o] for o in bops]))
                row["answer_only"] = em_est
                for ref in ("base", "full_midtrain"):
                    if ref in data and ref != l:
                        d, dlo, dhi = boot_ci(per_op(l, "strict", k), bops, a.bootstrap, rng, other=per_op(ref, "strict", k))
                        row[f"vs_{ref}"] = (d, dlo, dhi)
                rows.append(row)
    with open(a.out / "bands.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["band", "k", "checkpoint", "strict", "ci_lo", "ci_hi", "answer_only",
                                        "diff_vs_base", "diff_vs_base_lo", "diff_vs_base_hi",
                                        "diff_vs_midtrain", "diff_vs_midtrain_lo", "diff_vs_midtrain_hi"])
        for r in rows:
            vb = r.get("vs_base", ("", "", "")); vm = r.get("vs_full_midtrain", ("", "", ""))
            w.writerow([r["band"], r["k"], r["checkpoint"], *(f"{x:.4f}" for x in (r["strict"], r["lo"], r["hi"], r["answer_only"])),
                        *(f"{x:.4f}" if x != "" else "" for x in (*vb, *vm))])

    # plots
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        for k in (1, K):
            fig, ax = plt.subplots(figsize=(7, 4.2))
            for l in labels:
                ys = [np.nanmean(P[l]["strict"][k][data[l]["op"] == o]) for o in ops_all]
                ax.plot(ops_all, ys, marker="o", label=NICE[l], linestyle="--" if l in ("base", "full_midtrain") else "-")
            ax.axvspan(10.5, 14.5, color="0.92", zorder=0); ax.text(12.5, 1.02, "edge", ha="center", fontsize=9)
            ax.text(17.5, 1.02, "hard", ha="center", fontsize=9)
            ax.set(xlabel="op count", ylabel=f"strict pass@{k}", ylim=(0, 1.08), xticks=ops_all,
                   title=f"E3 — strict pass@{k} by op (same compute budget)")
            ax.legend(fontsize=8, frameon=False); ax.grid(alpha=.3); fig.tight_layout()
            fig.savefig(a.out / f"strict_pass{k}_by_op.png", dpi=150); plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        for ax, (bname, band) in zip(axes, (("edge op 11–14", EDGE), ("hard op 15–20", HARD))):
            for l in labels:
                bm = band_mask(data[l]["op"], band)
                ax.plot(ks, [np.nanmean(P[l]["strict"][k][bm]) for k in ks], marker="o", label=NICE[l])
            ax.set(xscale="log", xticks=ks, xticklabels=ks, xlabel="k", title=bname); ax.grid(alpha=.3)
        axes[0].set_ylabel("strict pass@k"); axes[1].legend(fontsize=8, frameon=False); fig.tight_layout()
        fig.savefig(a.out / "strict_pass_at_k_bands.png", dpi=150); plt.close(fig)
    except ImportError:
        notes.append("matplotlib missing; plots skipped")

    # prediction checks
    def get(band, k, l):
        return next((r for r in rows if r["band"] == band and r["k"] == k and r["checkpoint"] == l), None)

    def best(band, k):
        c = [r for r in rows if r["band"] == band and r["k"] == k]
        return max(c, key=lambda r: r["strict"]) if c else None

    checks = []
    if not a.smoke and all(l in data for l in LABELS):
        b = best("edge", 1); lr = get("edge", 1, "light_rl_0.2")
        checks.append(("Edge: light RL has the best pass@1", lr["strict"] >= b["strict"] - 1e-9,
                       f"best = {NICE[b['checkpoint']]} ({b['strict']:.3f}); light RL {lr['strict']:.3f}"))
        d = lr["vs_full_midtrain"]
        checks.append(("Edge: full mid-training is close to light RL (pass@1)", d[1] <= 0 <= d[2] or abs(d[0]) < 0.03,
                       f"light RL − mid-training = {d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]"))
        for k in (1, K):
            b = best("hard", k)
            checks.append((f"Hard: heavy RL has the best pass@{k}", get("hard", k, "heavy_rl_0.8")["strict"] >= b["strict"] - 1e-9,
                           f"best = {NICE[b['checkpoint']]} ({b['strict']:.3f}); heavy RL {get('hard', k, 'heavy_rl_0.8')['strict']:.3f}"))
        hp = [get("hard", 1, l)["strict"] for l in ("full_midtrain", "light_rl_0.2", "medium_rl_0.5", "heavy_rl_0.8")]
        checks.append(("Hard: pass@1 rises with RL share (0 → 0.2 → 0.5 → 0.8)", all(x <= y for x, y in zip(hp, hp[1:])),
                       " → ".join(f"{x:.3f}" for x in hp)))

    # report
    L = ["# E3 — Mid-training vs RL under a fixed compute budget", ""]
    if a.smoke:
        L += ["**SMOKE RUN** (k is small; numbers are not meaningful).", ""]
    L += ["## Data checks", *[f"- {s}" for s in notes], ""]
    L += [f"## Band averages — strict pass@k (95% bootstrap CI over problems; {a.bootstrap} reps)", "",
          "| band | k | checkpoint | strict | 95% CI | answer-only | Δ vs base | Δ vs full mid-training |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        f = lambda t: f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]" if t else "—"
        L.append(f"| {r['band']} | {r['k']} | {NICE[r['checkpoint']]} | {r['strict']:.3f} | [{r['lo']:.3f}, {r['hi']:.3f}] | "
                 f"{r['answer_only']:.3f} | {f(r.get('vs_base'))} | {f(r.get('vs_full_midtrain'))} |")
    L += ["", f"## Per-op strict pass@1 / pass@{K}", "", "| checkpoint | " + " | ".join(f"op{o}" for o in ops_all) + " |",
          "|---|" + "---|" * len(ops_all)]
    for l in labels:
        cells = [f"{np.nanmean(P[l]['strict'][1][data[l]['op']==o]):.2f} / {np.nanmean(P[l]['strict'][K][data[l]['op']==o]):.2f}" for o in ops_all]
        L.append(f"| {NICE[l]} | " + " | ".join(cells) + " |")
    if checks:
        L += ["", "## Against the paper's predictions", "", "| prediction | holds? | evidence |", "|---|---|---|"]
        L += [f"| {c} | {'yes' if ok else 'no'} | {ev} |" for c, ok, ev in checks]
        L += ["", "Paper caveat: at larger budgets full RL wins hard pass@128, so mid-training is not expected to win every metric."]
    L += ["", "Notes: pass@k uses the unbiased estimator over all n samples (the repo's aggregator instead uses the first k samples). "
          "Strict = every step and the answer correct; unscored samples (parse fail) count as wrong. "
          "Light RL starts from checkpoint-6579, not 0.8 × 7,740, so the budget split is approximate."]
    (a.out / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
