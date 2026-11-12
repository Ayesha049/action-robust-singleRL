"""Random PR / NR action-perturbation sweeps for a trained actor.

PR-random : each step, with prob. alpha, the action is replaced by one drawn
            uniformly from the action box.
NR-random : each step, executed action = (1-alpha)*actor + alpha*uniform.

The actor acts deterministically; no adversary is used at test time.
Common random numbers: environment reset seeds and perturbation streams depend
only on (sweep, alpha, episode), so every method sees identical randomness.

Writes <run_dir>/eval.csv with one row per episode.
"""

import argparse
import csv
import json
import os

import gymnasium as gym
import numpy as np
import torch

from train import Actor

SWEEPS = {"pr": 0, "nr": 1}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--alphas", default="0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5")
    p.add_argument("--sweeps", default="pr,nr")
    p.add_argument("--weights", default="actor.pt", help="actor.pt or checkpoint.pt")
    p.add_argument("--threads", type=int, default=1)
    return p.parse_args()


def load_actor(run_dir, weights, env):
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    actor = Actor(obs_dim, act_dim, env.action_space.low, env.action_space.high)
    state = torch.load(os.path.join(run_dir, weights), map_location="cpu")
    if "actor" in state:
        state = state["actor"]
    actor.load_state_dict(state)
    actor.eval()
    return actor


def run_episode(env, actor, sweep, alpha, reset_seed, pert_rng):
    low, high = env.action_space.low, env.action_space.high
    obs, _ = env.reset(seed=reset_seed)
    total, length, done = 0.0, 0, False
    while not done:
        with torch.no_grad():
            a = actor(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))[0].numpy()
        # draw both numbers every step so the stream is identical across policies
        coin = pert_rng.random()
        rand_a = pert_rng.uniform(low, high)
        if sweep == "pr":
            if coin < alpha:
                a = rand_a
        else:
            a = (1 - alpha) * a + alpha * rand_a
        obs, r, term, trunc, _ = env.step(np.clip(a, low, high).astype(np.float32))
        total += r
        length += 1
        done = term or trunc
    return total, length


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    with open(os.path.join(args.run_dir, "config.json")) as f:
        cfg = json.load(f)
    env = gym.make(cfg["env_id"])
    actor = load_actor(args.run_dir, args.weights, env)
    alphas = [float(x) for x in args.alphas.split(",")]

    out = os.path.join(args.run_dir, "eval.csv")
    with open(out + ".tmp", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["env_id", "mode", "seed", "sweep", "alpha", "episode", "return", "length"])
        for sweep in args.sweeps.split(","):
            for ai, alpha in enumerate(alphas):
                for ep in range(args.episodes):
                    pert_rng = np.random.default_rng([SWEEPS[sweep], ai, ep])
                    ret, length = run_episode(env, actor, sweep, alpha, 100_000 + ep, pert_rng)
                    w.writerow([cfg["env_id"], cfg["mode"], cfg["seed"], sweep, alpha, ep,
                                round(ret, 3), length])
    os.replace(out + ".tmp", out)
    env.close()


if __name__ == "__main__":
    main()
