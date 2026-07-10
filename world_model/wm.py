"""CoinRun next-frame diffusion world model: data, diffusion, UNet, EMA."""
import glob
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CONTEXT = 4
NUM_ACTIONS = 15
T_TRAIN = 1000


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

def load_episodes(root="dataset"):
    """Load all episodes into RAM. Split by session: last 2 (sorted) = val."""
    sessions = sorted(glob.glob(os.path.join(root, "session_*")))
    assert len(sessions) >= 3, f"need >=3 sessions in {root}, found {len(sessions)}"
    val_sessions = set(sessions[-2:])
    train_eps, val_eps = [], []
    for s in sessions:
        for f in sorted(glob.glob(os.path.join(s, "episode_*.npz"))):
            with np.load(f) as d:
                ep = {"obs": d["observations"], "actions": d["actions"]}
            (val_eps if s in val_sessions else train_eps).append(ep)
    return train_eps, val_eps


def norm_frame(f):
    """(H,W,3) uint8 -> (3,H,W) float32 in [-1,1]."""
    return torch.from_numpy(f.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1)


class TransitionDataset(torch.utils.data.Dataset):
    """(4 context frames, action, next frame). obs[t] is the frame seen
    before actions[t]; target is obs[t+1], so valid t is 0..T-2."""

    def __init__(self, episodes):
        self.episodes = episodes
        self.index = [
            (i, t)
            for i, ep in enumerate(episodes)
            for t in range(len(ep["actions"]) - 1)
        ]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        i, t = self.index[idx]
        ep = self.episodes[i]
        ctx_idx = [max(0, t - k) for k in range(CONTEXT - 1, -1, -1)]  # oldest..newest
        ctx = torch.cat([norm_frame(ep["obs"][j]) for j in ctx_idx], dim=0)
        target = norm_frame(ep["obs"][t + 1])
        return ctx, torch.tensor(int(ep["actions"][t])), target
