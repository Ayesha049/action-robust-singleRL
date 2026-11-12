"""Action-robust DDPG (PR-MDP / NR-MDP) on Gymnasium MuJoCo.

Single-file, CleanRL-style DDPG extended with an adversary, after
Tessler, Efroni & Mannor (2019), "Action Robust Reinforcement Learning and
Applications in Continuous Control" (ICML), Algorithm 5.

Modes
-----
none : plain DDPG (adversary disabled; identical code path otherwise)
pr   : PR-MDP  - with prob. alpha the adversary's action is executed
nr   : NR-MDP  - executed action = (1-alpha)*agent + alpha*adversary

The critic estimates Q of the *joint* (agent, adversary) policy. The agent
maximises it, the adversary minimises it; the adversary is updated once every
`adv_ratio` actor updates (paper: 10 for PR, 1 for NR).
"""

import argparse
import csv
import json
import os
import random
import subprocess
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", default="Hopper-v5")
    p.add_argument("--mode", choices=["none", "pr", "nr"], default="none")
    p.add_argument("--alpha", type=float, default=0.1, help="training uncertainty level")
    p.add_argument("--adv-ratio", type=int, default=None,
                   help="actor updates per adversary update (default: 10 for pr, 1 for nr)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--total-timesteps", type=int, default=1_000_000)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--buffer-size", type=int, default=1_000_000)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--exploration-noise", type=float, default=0.1)
    p.add_argument("--learning-starts", type=int, default=25_000)
    p.add_argument("--policy-frequency", type=int, default=2)
    p.add_argument("--checkpoint-every", type=int, default=100_000)
    p.add_argument("--out-dir", default="runs")
    p.add_argument("--run-name", default=None)
    p.add_argument("--threads", type=int, default=1)
    a = p.parse_args()
    if a.adv_ratio is None:
        a.adv_ratio = {"pr": 10, "nr": 1}.get(a.mode, 1)
    if a.mode == "none":
        a.alpha = 0.0
    return a


def git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


class QNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim + act_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))


class Actor(nn.Module):
    """Deterministic policy; used for both the agent and the adversary."""

    def __init__(self, obs_dim, act_dim, low, high):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, act_dim)
        self.register_buffer("action_scale", torch.tensor((high - low) / 2.0, dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor((high + low) / 2.0, dtype=torch.float32))

    def forward(self, x):
        x = F.relu(self.fc2(F.relu(self.fc1(x))))
        return torch.tanh(self.fc_mu(x)) * self.action_scale + self.action_bias


class ReplayBuffer:
    def __init__(self, size, obs_dim, act_dim):
        self.obs = np.zeros((size, obs_dim), np.float32)
        self.next_obs = np.zeros((size, obs_dim), np.float32)
        self.act = np.zeros((size, act_dim), np.float32)
        self.rew = np.zeros((size, 1), np.float32)
        self.term = np.zeros((size, 1), np.float32)
        self.size, self.ptr, self.full = size, 0, False

    def add(self, o, no, a, r, t):
        i = self.ptr
        self.obs[i], self.next_obs[i], self.act[i], self.rew[i], self.term[i] = o, no, a, r, t
        self.ptr = (i + 1) % self.size
        self.full = self.full or self.ptr == 0

    def sample(self, n):
        idx = np.random.randint(0, self.size if self.full else self.ptr, n)
        return (torch.from_numpy(self.obs[idx]), torch.from_numpy(self.act[idx]),
                torch.from_numpy(self.next_obs[idx]), torch.from_numpy(self.rew[idx]),
                torch.from_numpy(self.term[idx]))


def joint_value(qf, obs, mu, nu, mode, alpha):
    """Q of the joint policy at obs, given agent action mu and adversary action nu."""
    if mode == "pr":
        return (1 - alpha) * qf(obs, mu) + alpha * qf(obs, nu)
    if mode == "nr":
        return qf(obs, (1 - alpha) * mu + alpha * nu)
    return qf(obs, mu)


def soft_update(target, source, tau):
    for tp, p in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tau * p.data + (1 - tau) * tp.data)


def main():
    args = parse_args()
    run_name = args.run_name or f"{args.env_id}__{args.mode}__s{args.seed}"
    run_dir = os.path.join(args.out_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    env = gym.wrappers.RecordEpisodeStatistics(gym.make(args.env_id))
    env.action_space.seed(args.seed)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    low, high = env.action_space.low, env.action_space.high
    scale = (high - low) / 2.0

    config = dict(vars(args), run_name=run_name, git_sha=git_sha(),
                  torch=torch.__version__, gymnasium=gym.__version__)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    actor = Actor(obs_dim, act_dim, low, high)
    actor_t = Actor(obs_dim, act_dim, low, high)
    actor_t.load_state_dict(actor.state_dict())
    qf = QNetwork(obs_dim, act_dim)
    qf_t = QNetwork(obs_dim, act_dim)
    qf_t.load_state_dict(qf.state_dict())
    q_opt = torch.optim.Adam(qf.parameters(), lr=args.learning_rate)
    a_opt = torch.optim.Adam(actor.parameters(), lr=args.learning_rate)

    use_adv = args.mode != "none"
    if use_adv:
        adv = Actor(obs_dim, act_dim, low, high)
        adv_t = Actor(obs_dim, act_dim, low, high)
        adv_t.load_state_dict(adv.state_dict())
        adv_opt = torch.optim.Adam(adv.parameters(), lr=args.learning_rate)

    rb = ReplayBuffer(args.buffer_size, obs_dim, act_dim)

    ep_file = open(os.path.join(run_dir, "train_episodes.csv"), "w", newline="")
    ep_log = csv.writer(ep_file)
    ep_log.writerow(["global_step", "episodic_return", "episodic_length", "adv_executed_frac"])
    stat_file = open(os.path.join(run_dir, "train_stats.csv"), "w", newline="")
    stat_log = csv.writer(stat_file)
    stat_log.writerow(["global_step", "qf_loss", "actor_loss", "adv_loss", "q_mean", "sps"])

    obs, _ = env.reset(seed=args.seed)
    ep_adv_steps = ep_steps = 0
    actor_updates = 0
    last = dict(qf_loss=np.nan, actor_loss=np.nan, adv_loss=np.nan, q_mean=np.nan)
    start = time.time()

    for step in range(args.total_timesteps):
        # ---- act -------------------------------------------------------------
        adv_used = False
        if step < args.learning_starts:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                o = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                mu = actor(o)[0].numpy()
                mu = np.clip(mu + rng.normal(0, scale * args.exploration_noise), low, high)
                if use_adv:
                    nu = adv(o)[0].numpy()
                    nu = np.clip(nu + rng.normal(0, scale * args.exploration_noise), low, high)
            if args.mode == "pr":
                adv_used = rng.random() < args.alpha
                action = nu if adv_used else mu
            elif args.mode == "nr":
                action = (1 - args.alpha) * mu + args.alpha * nu
            else:
                action = mu
            action = np.clip(action, low, high).astype(np.float32)

        next_obs, reward, term, trunc, info = env.step(action)
        rb.add(obs, next_obs, action, reward, float(term))
        ep_steps += 1
        ep_adv_steps += int(adv_used)
        obs = next_obs

        if term or trunc:
            ep = info["episode"]
            ep_log.writerow([step + 1, float(ep["r"]), int(ep["l"]),
                             ep_adv_steps / max(ep_steps, 1)])
            ep_adv_steps = ep_steps = 0
            obs, _ = env.reset()

        # ---- learn -----------------------------------------------------------
        if step >= args.learning_starts:
            b_obs, b_act, b_next, b_rew, b_term = rb.sample(args.batch_size)
            with torch.no_grad():
                nu_next = adv_t(b_next) if use_adv else None
                q_next = joint_value(qf_t, b_next, actor_t(b_next), nu_next, args.mode, args.alpha)
                target = b_rew + args.gamma * (1 - b_term) * q_next
            q = qf(b_obs, b_act)
            qf_loss = F.mse_loss(q, target)
            q_opt.zero_grad()
            qf_loss.backward()
            q_opt.step()
            last["qf_loss"], last["q_mean"] = qf_loss.item(), q.mean().item()

            if step % args.policy_frequency == 0:
                # agent maximises the joint value; adversary held fixed
                mu_b = actor(b_obs)
                if args.mode == "pr":
                    actor_loss = -((1 - args.alpha) * qf(b_obs, mu_b)).mean()
                elif args.mode == "nr":
                    with torch.no_grad():
                        nu_b = adv(b_obs)
                    actor_loss = -qf(b_obs, (1 - args.alpha) * mu_b + args.alpha * nu_b).mean()
                else:
                    actor_loss = -qf(b_obs, mu_b).mean()
                a_opt.zero_grad()
                actor_loss.backward()
                a_opt.step()
                actor_updates += 1
                last["actor_loss"] = actor_loss.item()

                # adversary minimises the joint value; agent held fixed
                if use_adv and actor_updates % args.adv_ratio == 0:
                    nu_b = adv(b_obs)
                    if args.mode == "pr":
                        adv_loss = (args.alpha * qf(b_obs, nu_b)).mean()
                    else:
                        with torch.no_grad():
                            mu_d = actor(b_obs)
                        adv_loss = qf(b_obs, (1 - args.alpha) * mu_d + args.alpha * nu_b).mean()
                    adv_opt.zero_grad()
                    adv_loss.backward()
                    adv_opt.step()
                    last["adv_loss"] = adv_loss.item()

                soft_update(actor_t, actor, args.tau)
                soft_update(qf_t, qf, args.tau)
                if use_adv:
                    soft_update(adv_t, adv, args.tau)

        if (step + 1) % 5000 == 0:
            sps = (step + 1) / (time.time() - start)
            stat_log.writerow([step + 1, last["qf_loss"], last["actor_loss"], last["adv_loss"],
                               last["q_mean"], round(sps, 1)])
            stat_file.flush()
            ep_file.flush()

        if (step + 1) % args.checkpoint_every == 0 or step + 1 == args.total_timesteps:
            ckpt = {"actor": actor.state_dict(), "step": step + 1, "config": config}
            if use_adv:
                ckpt["adversary"] = adv.state_dict()
            torch.save(ckpt, os.path.join(run_dir, "checkpoint.pt"))

    torch.save(actor.state_dict(), os.path.join(run_dir, "actor.pt"))
    if use_adv:
        torch.save(adv.state_dict(), os.path.join(run_dir, "adversary.pt"))
    with open(os.path.join(run_dir, "DONE"), "w") as f:
        f.write(f"{time.time() - start:.0f} seconds\n")
    ep_file.close()
    stat_file.close()
    env.close()


if __name__ == "__main__":
    main()
