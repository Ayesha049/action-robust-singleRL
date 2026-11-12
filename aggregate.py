"""Aggregate eval.csv files into summary tables, verdicts and figures.

Outputs (in --out):
  curves.csv    mean return per (env, mode, seed, sweep, alpha)
  metrics.csv   per-seed clean return, AUC, relative AUC, breakdown alpha
  verdicts.csv  paired (by seed) robust-minus-DDPG differences, 95% bootstrap CI
  curves_<sweep>.png, curves_<sweep>_normalised.png
"""

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

COLORS = {"none": "#4c4c4c", "pr": "#1f77b4", "nr": "#d95f02"}
LABELS = {"none": "DDPG", "pr": "PR-DDPG", "nr": "NR-DDPG"}


def bootstrap_ci(x, n=10_000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    means = rng.choice(x, (n, len(x)), replace=True).mean(1)
    return np.percentile(means, [2.5, 97.5])


def breakdown_alpha(alphas, returns, clean):
    for a, r in zip(alphas, returns):
        if r < 0.5 * clean:
            return a
    return np.inf


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs")
    p.add_argument("--out", default="results")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    files = sorted(glob.glob(os.path.join(a.runs, "*", "eval.csv")))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    shas = set()
    for f in files:
        with open(os.path.join(os.path.dirname(f), "config.json")) as fh:
            shas.add(json.load(fh).get("git_sha", "unknown"))

    curves = (df.groupby(["env_id", "mode", "seed", "sweep", "alpha"])["return"]
                .mean().reset_index())
    curves.to_csv(os.path.join(a.out, "curves.csv"), index=False)

    rows = []
    for (env, mode, seed, sweep), g in curves.groupby(["env_id", "mode", "seed", "sweep"]):
        g = g.sort_values("alpha")
        clean = g.loc[g.alpha == 0, "return"].iloc[0]
        auc = g["return"].mean()
        rows.append(dict(env_id=env, mode=mode, seed=seed, sweep=sweep, clean=clean, auc=auc,
                         rel_auc=auc / clean if clean > 0 else np.nan,
                         breakdown=breakdown_alpha(g.alpha.values, g["return"].values, clean)))
    metrics = pd.DataFrame(rows)
    metrics.to_csv(os.path.join(a.out, "metrics.csv"), index=False)

    verdicts = []
    for (env, sweep), g in metrics.groupby(["env_id", "sweep"]):
        base = g[g["mode"] == "none"].set_index("seed")
        for mode in ["pr", "nr"]:
            rob = g[g["mode"] == mode].set_index("seed")
            seeds = base.index.intersection(rob.index)
            if len(seeds) < 2:
                continue
            for metric in ["clean", "auc", "rel_auc"]:
                d = (rob.loc[seeds, metric] - base.loc[seeds, metric]).values
                lo, hi = bootstrap_ci(d)
                mean = d.mean()
                verdict = ("supported" if lo > 0 else "not supported" if mean < 0
                           else "weakly supported")
                verdicts.append(dict(env_id=env, sweep=sweep, method=mode, metric=metric,
                                     n_seeds=len(seeds), mean_diff=mean, ci_low=lo, ci_high=hi,
                                     verdict=verdict))
    pd.DataFrame(verdicts).to_csv(os.path.join(a.out, "verdicts.csv"), index=False)

    envs = sorted(curves.env_id.unique())
    for sweep in sorted(curves.sweep.unique()):
        for norm in [False, True]:
            fig, axes = plt.subplots(1, len(envs), figsize=(5 * len(envs), 3.8), squeeze=False)
            for ax, env in zip(axes[0], envs):
                for mode in ["none", "pr", "nr"]:
                    g = curves[(curves.env_id == env) & (curves.sweep == sweep)
                               & (curves["mode"] == mode)]
                    if g.empty:
                        continue
                    piv = g.pivot(index="seed", columns="alpha", values="return")
                    if norm:
                        piv = piv.div(piv[0.0], axis=0)
                    m = piv.mean(axis=0)
                    lo, hi = zip(*[bootstrap_ci(piv[c].values) for c in piv.columns])
                    ax.plot(piv.columns, m, color=COLORS[mode], label=LABELS[mode], lw=2)
                    ax.fill_between(piv.columns, lo, hi, color=COLORS[mode], alpha=0.2)
                ax.set_title(f"{env} - {sweep.upper()}-random")
                ax.set_xlabel("test alpha")
                ax.set_ylabel("return / clean return" if norm else "episodic return")
                ax.grid(alpha=0.3)
            axes[0][0].legend()
            fig.tight_layout()
            name = f"curves_{sweep}{'_normalised' if norm else ''}.png"
            fig.savefig(os.path.join(a.out, name), dpi=150)
            plt.close(fig)

    with open(os.path.join(a.out, "provenance.json"), "w") as f:
        json.dump({"git_shas": sorted(shas), "n_eval_files": len(files)}, f, indent=2)
    print(f"aggregated {len(files)} runs -> {a.out}")


if __name__ == "__main__":
    main()
