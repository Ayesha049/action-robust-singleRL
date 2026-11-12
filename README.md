# action-robust-singleRL

Action-robust DDPG (PR-MDP and NR-MDP) on Gymnasium MuJoCo. It's a 2026 adaptation of
Tessler, Efroni & Mannor (2019), *Action Robust Reinforcement Learning and
Applications in Continuous Control* (ICML, arXiv:1901.09184).

This repo holds code and run scripts only. The plan and analysis live in a separate
research repo.

## Files

| File | Purpose |
|---|---|
| `train.py` | CleanRL-style DDPG with an adversary: `--mode none` (DDPG), `pr` (PR-MDP), `nr` (NR-MDP) |
| `evaluate.py` | Random PR and NR action-perturbation sweeps over test alpha (0 to 0.5) for a trained actor |
| `aggregate.py` | Per-seed metrics (clean return, AUC, relative AUC, breakdown alpha), paired bootstrap verdicts, figures |
| `scripts/launch_train.sh` | Runs the training grid in parallel; skips finished runs |
| `scripts/launch_eval.sh` | Evaluates all finished runs, then aggregates |

## Method

| | DDPG | PR-DDPG | NR-DDPG |
|---|---|---|---|
| Executed action in training | agent | adversary with prob. α, else agent | (1−α)·agent + α·adversary |
| Training α | — | 0.1 | 0.1 |
| Actor : adversary updates | — | 10 : 1 | 1 : 1 |

The critic learns Q of the joint policy. The PR target uses the expectation
(1−α)Q(s′, μ′) + αQ(s′, ν′). Everything else (networks 256-256, lr 3e-4, batch 256,
τ 0.005, Gaussian exploration 0.1, 25k warm-up steps, actor update every 2 steps)
follows CleanRL's DDPG defaults.

## Deviations from the paper

- Gymnasium v5 tasks instead of gym v2.
- 1M steps instead of 2M.
- Gaussian exploration for all methods, instead of parameter-space noise (PR) and OU
  noise (NR).
- CleanRL network sizes and learning rates.
- Evaluation covers random PR and NR sweeps only (no mass sweep).

## Environment

Python 3.11; see `requirements.txt` for exact versions. Install the CPU build of PyTorch:
these networks are too small to benefit from a GPU.

## Outputs

Each run writes `runs/<env>__<mode>__s<seed>/`, containing:
- `config.json`, including the git SHA;
- `train_episodes.csv`, `train_stats.csv`;
- `actor.pt`, `adversary.pt`, `checkpoint.pt`;
- `DONE`;
- `eval.csv` once evaluated.

`aggregate.py` writes `curves.csv`, `metrics.csv`, `verdicts.csv`, figures and
`provenance.json`.
