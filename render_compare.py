"""Side-by-side GIF: clean baseline | perturbed baseline | perturbed robust policy.

Each panel uses the same environment reset seed and the same perturbation
random stream as evaluate.py (common random numbers), so the only difference
between panels 2 and 3 is the policy. Policies are taken from the median seed
by clean return (from aggregate.py's metrics.csv), so they're representative
rather than the best seed.

Example:
  MUJOCO_GL=egl python render_compare.py --env-id Walker2d-v5 --sweep pr --alpha 0.2 \
      --metrics results/runs/metrics.csv --out walker2d_pr0.2.gif
"""

import argparse
import os

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont

from evaluate import SWEEPS, load_actor

LABELS = {"none": "DDPG", "pr": "PR-DDPG", "nr": "NR-DDPG"}


def median_seed(metrics, env_id, mode):
    m = metrics[(metrics.env_id == env_id) & (metrics["mode"] == mode)].drop_duplicates("seed")
    m = m.sort_values("clean").reset_index(drop=True)
    return int(m.loc[len(m) // 2, "seed"]), float(m.loc[len(m) // 2, "clean"])


def representative_episode(conds):
    """Episode closest to the median for ALL panels at once.

    conds: list of (run_dir, sweep, alpha) per panel. Uses evaluate.py's eval.csv.
    Score = max over panels of |percentile rank - 0.5|; the smallest score wins.
    """
    cols = {}
    for k, (run_dir, sweep, alpha) in enumerate(conds):
        ev = pd.read_csv(os.path.join(run_dir, "eval.csv"))
        cols[k] = ev[(ev.sweep == sweep) & (ev.alpha == alpha)].set_index("episode")["return"]
    df = pd.DataFrame(cols)
    score = (df.rank(pct=True) - 0.5).abs().max(axis=1)
    e = int(score.idxmin())
    return e, float(score[e]), df.rank(pct=True).loc[e].round(2).tolist()


def rollout(env_id, run_dir, sweep, alpha, alpha_idx, episode, size, every):
    # v5's default camera is free (doesn't follow the robot); track the torso, side view
    import mujoco
    cam = {"type": mujoco.mjtCamera.mjCAMERA_TRACKING, "trackbodyid": 1, "distance": 4.0,
           "elevation": -15.0, "azimuth": 90.0}
    env = gym.make(env_id, render_mode="rgb_array", width=size, height=size,
                   default_camera_config=cam)
    actor = load_actor(run_dir, "actor.pt", env)
    low, high = env.action_space.low, env.action_space.high
    obs, _ = env.reset(seed=100_000 + episode)
    rng = np.random.default_rng([SWEEPS[sweep], alpha_idx, episode])
    frames, rets, total, done, t, fell = [], [], 0.0, False, 0, False
    while not done:
        if t % every == 0:
            frames.append(env.render())
            rets.append(total)
        with torch.no_grad():
            a = actor(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))[0].numpy()
        coin = rng.random()
        rand_a = rng.uniform(low, high)
        if sweep == "pr":
            if coin < alpha:
                a = rand_a
        else:
            a = (1 - alpha) * a + alpha * rand_a
        obs, r, term, trunc, _ = env.step(np.clip(a, low, high).astype(np.float32))
        total += r
        t += 1
        done = term or trunc
        fell = bool(term)
    frames.append(env.render())
    rets.append(total)
    env.close()
    return frames, rets, total, t, fell


def annotate(frame, title, sub, ret, status, size):
    img = Image.fromarray(frame).convert("RGB")
    bar = Image.new("RGB", (size, 46), (255, 255, 255))
    canvas = Image.new("RGB", (size, size + 46), (255, 255, 255))
    canvas.paste(bar, (0, 0))
    canvas.paste(img, (0, 46))
    d = ImageDraw.Draw(canvas)
    try:
        f1 = ImageFont.truetype("DejaVuSans-Bold.ttf", 14)
        f2 = ImageFont.truetype("DejaVuSans.ttf", 12)
    except OSError:
        f1 = f2 = ImageFont.load_default()
    d.text((6, 4), title, fill=(20, 20, 20), font=f1)
    d.text((6, 24), f"{sub}   return {ret:7.0f}", fill=(60, 60, 60), font=f2)
    if status:
        d.text((size - 70, 4), status, fill=(200, 30, 30), font=f1)
    return canvas


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", default="Walker2d-v5")
    p.add_argument("--sweep", choices=["pr", "nr"], default="pr")
    p.add_argument("--alpha", type=float, default=0.2)
    p.add_argument("--alphas", default="0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5",
                   help="evaluate.py's grid, to reuse its random stream for this alpha")
    p.add_argument("--robust", choices=["pr", "nr"], default="pr")
    p.add_argument("--episode", default="auto",
                   help="evaluation episode index, or 'auto' = most representative for all panels")
    p.add_argument("--metrics", default="results/runs/metrics.csv")
    p.add_argument("--runs", default="runs")
    p.add_argument("--size", type=int, default=240)
    p.add_argument("--every", type=int, default=4, help="render every k-th control step")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    grid = [float(x) for x in a.alphas.split(",")]
    ai = grid.index(a.alpha)
    metrics = pd.read_csv(a.metrics)
    spec = [("none", 0.0, "clean"),
            ("none", a.alpha, f"{a.sweep.upper()} a={a.alpha}"),
            (a.robust, a.alpha, f"{a.sweep.upper()} a={a.alpha}")]
    dirs = [os.path.join(a.runs, f"{a.env_id}__{m}__s{median_seed(metrics, a.env_id, m)[0]}")
            for m, _, _ in spec]
    if a.episode == "auto":
        episode, score, pct = representative_episode(
            [(d, a.sweep, al) for d, (_, al, _) in zip(dirs, spec)])
        print(f"episode {episode}: percentile ranks {pct} (max |pct-0.5| = {score:.2f})")
    else:
        episode = int(a.episode)
    panels = []
    for run_dir, (mode, sweep_alpha, cond) in zip(dirs, spec):
        seed = int(run_dir.rsplit("__s", 1)[1])
        frames, rets, total, steps, fell = rollout(
            a.env_id, run_dir, a.sweep, sweep_alpha, grid.index(sweep_alpha), episode,
            a.size, a.every)
        panels.append(dict(frames=frames, rets=rets, title=f"{LABELS[mode]} | {cond}",
                           sub=f"seed {seed}", total=total, steps=steps, fell=fell))
        print(f"{LABELS[mode]:8s} {cond:12s} seed {seed} return {total:8.1f} "
              f"steps {steps} {'FELL' if fell else ''}")

    n = max(len(pn["frames"]) for pn in panels)
    out = []
    for i in range(n):
        tiles = []
        for pn in panels:
            j = min(i, len(pn["frames"]) - 1)
            status = "FELL" if pn["fell"] and i >= len(pn["frames"]) - 1 else ""
            tiles.append(annotate(pn["frames"][j], pn["title"], pn["sub"], pn["rets"][j], status,
                                  a.size))
        row = Image.new("RGB", (a.size * 3 + 8, a.size + 46), (255, 255, 255))
        for k, tile in enumerate(tiles):
            row.paste(tile, (k * (a.size + 4), 0))
        out.append(row.quantize(colors=128, method=Image.Quantize.MEDIANCUT))
    out[0].save(a.out, save_all=True, append_images=out[1:], duration=int(1000 / a.fps),
                loop=0, optimize=True)
    print(f"wrote {a.out}: {n} frames, {os.path.getsize(a.out) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
