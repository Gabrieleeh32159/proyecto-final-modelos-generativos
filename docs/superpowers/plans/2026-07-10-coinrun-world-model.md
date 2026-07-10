# CoinRun Next-Frame Diffusion World Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a conditional DDPM that predicts the next CoinRun frame (64×64×3) from 4 context frames + the action, using the human gameplay dataset in `dataset/`, on a MacBook Air M3 16GB.

**Architecture:** A ~10M-param UNet (channels 64/128/256 at resolutions 64/32/16, 2 res-blocks per level, self-attention at 16×16 and bottleneck) denoises the target frame concatenated with 4 context frames, conditioned on timestep + action embeddings. Cosine schedule, v-prediction, DDIM 20-step sampling. Evaluation is one-step PSNR/SSIM on held-out sessions vs. a copy-last-frame baseline, plus a sample grid and optional rollout GIF.

**Tech Stack:** Python (conda `procgen` env), PyTorch (MPS backend), numpy, imageio, pytest.

**Spec:** `docs/superpowers/specs/2026-07-10-coinrun-world-model-design.md`

## Global Constraints

- Hardware: MacBook Air M3, 16 GB RAM. Device = `"mps"` if available else `"cpu"`. fp32 only — no autocast, no custom kernels (MPS op support).
- New dependencies allowed: `torch`, `imageio` only (`numpy`, `pytest` already present in the procgen env).
- Constants (define once in `world_model/wm.py`): `CONTEXT = 4`, `NUM_ACTIONS = 15`, `T_TRAIN = 1000`. Frames normalized to `[-1, 1]` via `x / 127.5 - 1`.
- Hyperparameters: lr `1e-4` (AdamW), batch 32, EMA decay 0.999, checkpoint every 2000 steps, DDIM 20 sampling steps, target 100k train steps.
- Dataset split: sessions sorted by name; **last 2 sessions are validation**, rest train. Never mix.
- Do NOT modify anything under `procgen/` (upstream engine code). All new code lives in `world_model/`.
- `world_model/` has NO `__init__.py` — scripts run as `python world_model/train.py`, tests as `pytest world_model/test_wm.py` (both put `world_model/` on `sys.path`, so plain `from wm import ...` works).
- Commit messages: plain, imperative, no Co-Authored-By lines.
- All commands below run from the repo root: `/Users/gabrielespinoza/UTEC/proyecto_generative_models/procgen`.

## File Structure

- `world_model/wm.py` — dataset loading (`load_episodes`, `TransitionDataset`), diffusion process (`Diffusion`), model (`UNet` + blocks), `EMA`. One module: these pieces change together and total ~350 lines.
- `world_model/train.py` — CLI training loop: checkpoint/resume, loss CSV.
- `world_model/eval.py` — metrics (`psnr`, `ssim`), sample grid, rollout GIF, CLI.
- `world_model/test_wm.py` — all tests.

---

### Task 1: Data pipeline

**Files:**
- Create: `world_model/wm.py`
- Test: `world_model/test_wm.py`

**Interfaces:**
- Produces: `load_episodes(root="dataset") -> (train_eps, val_eps)` — lists of dicts `{"obs": (T,64,64,3) uint8 ndarray, "actions": (T,) uint8 ndarray}`; `TransitionDataset(episodes)` — `torch.utils.data.Dataset` yielding `(ctx (12,64,64) float32 in [-1,1], action () int64, target (3,64,64) float32 in [-1,1])`; constants `CONTEXT=4`, `NUM_ACTIONS=15`, `T_TRAIN=1000`.

- [ ] **Step 1: Verify dependencies**

Run: `python -c "import torch, imageio, numpy; print(torch.__version__, torch.backends.mps.is_available())"`
Expected: prints a torch version and `True`. If it fails: `pip install torch imageio`, then re-run.

- [ ] **Step 2: Write the failing tests**

Create `world_model/test_wm.py`:

```python
import numpy as np
import torch

from wm import CONTEXT, NUM_ACTIONS, T_TRAIN, TransitionDataset, load_episodes


def fake_episode(T=6, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "obs": rng.integers(0, 256, (T, 64, 64, 3), dtype=np.uint8),
        "actions": rng.integers(0, NUM_ACTIONS, (T,), dtype=np.uint8),
    }


def norm_frame(f):
    return torch.from_numpy(f.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1)


def test_dataset_shapes_and_edge_padding():
    ep = fake_episode()
    ds = TransitionDataset([ep])
    assert len(ds) == 5  # T-1: last action has no recorded next frame
    ctx, action, target = ds[0]
    assert ctx.shape == (12, 64, 64) and ctx.dtype == torch.float32
    assert target.shape == (3, 64, 64)
    assert action.dtype == torch.int64
    assert ctx.min() >= -1.0 and ctx.max() <= 1.0
    # t=0: all 4 context frames are edge-padded copies of frame 0
    for k in range(1, CONTEXT):
        assert torch.equal(ctx[3 * k:3 * k + 3], ctx[0:3])
    assert torch.allclose(target, norm_frame(ep["obs"][1]))


def test_dataset_context_order_and_action():
    ep = fake_episode()
    ds = TransitionDataset([ep])
    ctx, action, target = ds[4]  # t=4: context frames 1,2,3,4 oldest->newest
    assert torch.allclose(ctx[9:12], norm_frame(ep["obs"][4]))   # newest last
    assert torch.allclose(ctx[0:3], norm_frame(ep["obs"][1]))    # oldest first
    assert int(action) == int(ep["actions"][4])
    assert torch.allclose(target, norm_frame(ep["obs"][5]))


def test_load_episodes_session_split():
    train_eps, val_eps = load_episodes("dataset")
    assert len(train_eps) > 0 and len(val_eps) > 0
    ep = train_eps[0]
    assert ep["obs"].dtype == np.uint8 and ep["obs"].shape[1:] == (64, 64, 3)
    assert len(ep["obs"]) == len(ep["actions"])
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest world_model/test_wm.py -v`
Expected: FAIL/ERROR with `ModuleNotFoundError: No module named 'wm'`

- [ ] **Step 4: Implement the data pipeline**

Create `world_model/wm.py`:

```python
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest world_model/test_wm.py -v`
Expected: 3 passed (the `load_episodes` test takes ~30-60s — it decompresses the whole dataset).

- [ ] **Step 6: Commit**

```bash
git add world_model/wm.py world_model/test_wm.py
git commit -m "Add world model data pipeline: session-split episode loading and transition dataset"
```

---

### Task 2: Diffusion process

**Files:**
- Modify: `world_model/wm.py` (append)
- Test: `world_model/test_wm.py` (append)

**Interfaces:**
- Produces: `Diffusion(timesteps=T_TRAIN)` with attribute `alpha_bar (timesteps,) float32` and methods `add_noise(x0, t, noise) -> x_t`, `v_target(x0, t, noise) -> v`, `to_x0_eps(x_t, t, v) -> (x0, eps)`, `ddim_sample(model, ctx, action, steps=20) -> (B,3,64,64) in [-1,1]`. `t` is a `(B,)` LongTensor of values in `[0, timesteps)`. The `model` callable signature is `model(x_noisy, ctx, t, action) -> v` (matches Task 3's `UNet.forward`).

- [ ] **Step 1: Write the failing tests**

Append to `world_model/test_wm.py`:

```python
from wm import Diffusion


def test_alpha_bar_schedule():
    d = Diffusion()
    ab = d.alpha_bar
    assert ab.shape == (T_TRAIN,)
    assert ab[0] > 0.99 and ab[-1] < 0.01          # ~clean start, ~pure noise end
    assert bool((ab[1:] <= ab[:-1] + 1e-8).all())  # monotone decreasing


def test_v_parameterization_roundtrip():
    d = Diffusion()
    torch.manual_seed(0)
    x0 = torch.randn(4, 3, 64, 64).clamp(-1, 1)
    noise = torch.randn_like(x0)
    t = torch.tensor([0, 250, 500, 999])
    x_t = d.add_noise(x0, t, noise)
    v = d.v_target(x0, t, noise)
    x0_rec, eps_rec = d.to_x0_eps(x_t, t, v)
    assert torch.allclose(x0_rec, x0, atol=1e-4)
    assert torch.allclose(eps_rec, noise, atol=1e-4)


def test_ddim_sample_shape_and_range():
    d = Diffusion()
    dummy = lambda x, ctx, t, a: torch.zeros_like(x)  # pretend v=0 everywhere
    out = d.ddim_sample(dummy, torch.zeros(2, 12, 64, 64), torch.zeros(2, dtype=torch.long))
    assert out.shape == (2, 3, 64, 64)
    assert torch.isfinite(out).all()
    assert out.min() >= -1.0 and out.max() <= 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest world_model/test_wm.py -v -k "alpha or roundtrip or ddim"`
Expected: FAIL with `ImportError: cannot import name 'Diffusion'`

- [ ] **Step 3: Implement Diffusion**

Append to `world_model/wm.py`:

```python
# --------------------------------------------------------------------------
# diffusion (cosine schedule, v-prediction, DDIM sampling)
# --------------------------------------------------------------------------

class Diffusion:
    def __init__(self, timesteps=T_TRAIN):
        self.timesteps = timesteps
        s = 0.008
        t = torch.arange(timesteps + 1, dtype=torch.float64) / timesteps
        f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
        self.alpha_bar = (f[1:] / f[0]).clamp(1e-4, 0.9999).float()

    def _ab(self, t):
        return self.alpha_bar.to(t.device)[t].view(-1, 1, 1, 1)

    def add_noise(self, x0, t, noise):
        ab = self._ab(t)
        return ab.sqrt() * x0 + (1 - ab).sqrt() * noise

    def v_target(self, x0, t, noise):
        ab = self._ab(t)
        return ab.sqrt() * noise - (1 - ab).sqrt() * x0

    def to_x0_eps(self, x_t, t, v):
        ab = self._ab(t)
        x0 = ab.sqrt() * x_t - (1 - ab).sqrt() * v
        eps = (1 - ab).sqrt() * x_t + ab.sqrt() * v
        return x0, eps

    @torch.no_grad()
    def ddim_sample(self, model, ctx, action, steps=20):
        B, device = ctx.shape[0], ctx.device
        x = torch.randn(B, 3, 64, 64, device=device)
        ts = torch.linspace(self.timesteps - 1, 0, steps).long().to(device)
        for i, t in enumerate(ts):
            tb = torch.full((B,), int(t), device=device, dtype=torch.long)
            v = model(x, ctx, tb, action)
            x0, eps = self.to_x0_eps(x, tb, v)
            x0 = x0.clamp(-1, 1)
            if i < steps - 1:
                ab_prev = self.alpha_bar.to(device)[ts[i + 1]]
                x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps
            else:
                x = x0
        return x
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest world_model/test_wm.py -v -k "alpha or roundtrip or ddim"`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add world_model/wm.py world_model/test_wm.py
git commit -m "Add cosine-schedule diffusion with v-prediction and DDIM sampling"
```

---

### Task 3: UNet and EMA

**Files:**
- Modify: `world_model/wm.py` (append)
- Test: `world_model/test_wm.py` (append)

**Interfaces:**
- Consumes: constants from Task 1.
- Produces: `UNet(base=64, emb_dim=256, ctx_ch=12, num_actions=NUM_ACTIONS)` with `forward(x (B,3,64,64), ctx (B,12,64,64), t (B,) long, action (B,) long) -> (B,3,64,64)` (the v-prediction); `EMA(model, decay=0.999)` with attribute `shadow` (a state_dict-shaped dict of tensors, loadable via `model.load_state_dict(ema.shadow)`) and method `update(model)`.

- [ ] **Step 1: Write the failing tests**

Append to `world_model/test_wm.py`:

```python
import torch.nn as nn

from wm import EMA, UNet


def test_unet_output_shape_and_size():
    model = UNet()
    x = torch.randn(2, 3, 64, 64)
    ctx = torch.randn(2, 12, 64, 64)
    t = torch.randint(0, T_TRAIN, (2,))
    a = torch.randint(0, NUM_ACTIONS, (2,))
    out = model(x, ctx, t, a)
    assert out.shape == (2, 3, 64, 64)
    n_params = sum(p.numel() for p in model.parameters())
    assert 5e6 < n_params < 30e6


def test_unet_action_changes_output():
    torch.manual_seed(0)
    model = UNet().eval()
    x = torch.randn(1, 3, 64, 64)
    ctx = torch.randn(1, 12, 64, 64)
    t = torch.tensor([500])
    with torch.no_grad():
        out_a = model(x, ctx, t, torch.tensor([1]))
        out_b = model(x, ctx, t, torch.tensor([7]))
    assert not torch.allclose(out_a, out_b)


def test_ema_update():
    m = nn.Linear(2, 2)
    ema = EMA(m, decay=0.5)
    before = {k: v.clone() for k, v in ema.shadow.items()}
    with torch.no_grad():
        for p in m.parameters():
            p.add_(1.0)
    ema.update(m)
    for k in before:
        assert torch.allclose(ema.shadow[k], before[k] + 0.5)
    m.load_state_dict(ema.shadow)  # shadow is a loadable state dict
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest world_model/test_wm.py -v -k "unet or ema"`
Expected: FAIL with `ImportError: cannot import name 'EMA'`

- [ ] **Step 3: Implement UNet and EMA**

Append to `world_model/wm.py`:

```python
# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------

def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([args.cos(), args.sin()], dim=-1)


class ResBlock(nn.Module):
    """GroupNorm/SiLU/conv x2 with FiLM (scale-shift) conditioning."""

    def __init__(self, in_ch, out_ch, emb_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.emb = nn.Linear(emb_dim, out_ch * 2)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.emb(F.silu(emb))[:, :, None, None].chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift
        h = self.conv2(F.silu(h))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    def __init__(self, ch, heads=4):
        super().__init__()
        self.heads = heads
        self.norm = nn.GroupNorm(8, ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = (
            self.qkv(self.norm(x))
            .reshape(B, 3, self.heads, C // self.heads, H * W)
            .unbind(1)
        )
        out = F.scaled_dot_product_attention(
            q.transpose(-1, -2), k.transpose(-1, -2), v.transpose(-1, -2)
        )
        out = out.transpose(-1, -2).reshape(B, C, H, W)
        return x + self.proj(out)


class UNet(nn.Module):
    """64->32->16 encoder, attention at 16x16 and bottleneck, mirrored decoder.
    Input: noisy target frame (3ch) + 4 context frames (12ch)."""

    def __init__(self, base=64, emb_dim=256, ctx_ch=3 * (CONTEXT - 1) + 3,
                 num_actions=NUM_ACTIONS):
        super().__init__()
        c1, c2, c3 = base, base * 2, base * 4
        self.emb_dim = emb_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim)
        )
        self.action_emb = nn.Embedding(num_actions, emb_dim)

        self.conv_in = nn.Conv2d(3 + ctx_ch, c1, 3, padding=1)
        self.enc1 = nn.ModuleList([ResBlock(c1, c1, emb_dim), ResBlock(c1, c1, emb_dim)])
        self.down1 = nn.Conv2d(c1, c1, 3, stride=2, padding=1)   # 64 -> 32
        self.enc2 = nn.ModuleList([ResBlock(c1, c2, emb_dim), ResBlock(c2, c2, emb_dim)])
        self.down2 = nn.Conv2d(c2, c2, 3, stride=2, padding=1)   # 32 -> 16
        self.enc3 = nn.ModuleList([ResBlock(c2, c3, emb_dim), ResBlock(c3, c3, emb_dim)])
        self.attn3 = SelfAttention(c3)

        self.mid1 = ResBlock(c3, c3, emb_dim)
        self.mid_attn = SelfAttention(c3)
        self.mid2 = ResBlock(c3, c3, emb_dim)

        self.dec3 = nn.ModuleList([ResBlock(c3 * 2, c3, emb_dim), ResBlock(c3, c3, emb_dim)])
        self.attn_d3 = SelfAttention(c3)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(c3, c2, 3, padding=1)
        )
        self.dec2 = nn.ModuleList([ResBlock(c2 * 2, c2, emb_dim), ResBlock(c2, c2, emb_dim)])
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(c2, c1, 3, padding=1)
        )
        self.dec1 = nn.ModuleList([ResBlock(c1 * 2, c1, emb_dim), ResBlock(c1, c1, emb_dim)])
        self.out = nn.Sequential(
            nn.GroupNorm(8, c1), nn.SiLU(), nn.Conv2d(c1, 3, 3, padding=1)
        )

    def forward(self, x, ctx, t, action):
        emb = self.time_mlp(timestep_embedding(t, self.emb_dim)) + self.action_emb(action)
        h = self.conv_in(torch.cat([x, ctx], dim=1))
        for b in self.enc1:
            h = b(h, emb)
        s1 = h
        h = self.down1(h)
        for b in self.enc2:
            h = b(h, emb)
        s2 = h
        h = self.down2(h)
        for b in self.enc3:
            h = b(h, emb)
        h = self.attn3(h)
        s3 = h
        h = self.mid2(self.mid_attn(self.mid1(h, emb)), emb)
        h = torch.cat([h, s3], dim=1)
        for b in self.dec3:
            h = b(h, emb)
        h = self.attn_d3(h)
        h = self.up2(h)
        h = torch.cat([h, s2], dim=1)
        for b in self.dec2:
            h = b(h, emb)
        h = self.up1(h)
        h = torch.cat([h, s1], dim=1)
        for b in self.dec1:
            h = b(h, emb)
        return self.out(h)


class EMA:
    """Exponential moving average of a model's state dict."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v, alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest world_model/test_wm.py -v -k "unet or ema"`
Expected: 3 passed

- [ ] **Step 5: Run the full suite and commit**

Run: `pytest world_model/test_wm.py -v` — Expected: 9 passed

```bash
git add world_model/wm.py world_model/test_wm.py
git commit -m "Add conditional UNet (timestep+action FiLM, attention at 16x16) and EMA"
```

---

### Task 4: Training script

**Files:**
- Create: `world_model/train.py`
- Modify: `.gitignore` (append)

**Interfaces:**
- Consumes: `load_episodes`, `TransitionDataset`, `Diffusion`, `UNet`, `EMA` from `wm.py`.
- Produces: checkpoint file `world_model/checkpoints/latest.pt` — a dict `{"model": state_dict, "ema": state_dict, "opt": state_dict, "step": int}` (Task 6's eval loads the `"ema"` entry); `world_model/checkpoints/loss.csv` with rows `step,loss`.

No unit test for the loop itself — the deliverable check is a real smoke run + resume run against the actual dataset (steps 3-4), which exercises every code path.

- [ ] **Step 1: Ignore training artifacts**

Append to `.gitignore`:

```
world_model/checkpoints/
world_model/results/
```

- [ ] **Step 2: Write train.py**

Create `world_model/train.py`:

```python
"""Train the CoinRun next-frame diffusion model.

Usage:
    python world_model/train.py                 # full run (100k steps)
    python world_model/train.py --resume        # continue from latest.pt
"""
import argparse
import csv
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from wm import EMA, Diffusion, TransitionDataset, UNet, load_episodes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="dataset")
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--out", default="world_model/checkpoints")
    p.add_argument("--ckpt-every", type=int, default=2000)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    train_eps, val_eps = load_episodes(args.data)
    ds = TransitionDataset(train_eps)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, drop_last=True)

    model = UNet().to(device)
    diff = Diffusion()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ema = EMA(model)
    step = 0

    os.makedirs(args.out, exist_ok=True)
    ckpt_path = os.path.join(args.out, "latest.pt")
    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"])
        ema.shadow = ck["ema"]
        opt.load_state_dict(ck["opt"])
        step = ck["step"]
        print(f"resumed at step {step}")

    n_params = sum(q.numel() for q in model.parameters())
    print(f"params: {n_params / 1e6:.1f}M  device: {device}  "
          f"train examples: {len(ds)}  val episodes: {len(val_eps)}")

    def save():
        torch.save(
            {"model": model.state_dict(), "ema": ema.shadow,
             "opt": opt.state_dict(), "step": step},
            ckpt_path,
        )
        print(f"saved checkpoint at step {step}")

    loss_f = open(os.path.join(args.out, "loss.csv"), "a", newline="")
    loss_w = csv.writer(loss_f)

    model.train()
    t0, step0 = time.time(), step
    while step < args.steps:
        for ctx, action, target in dl:
            if step >= args.steps:
                break
            ctx = ctx.to(device)
            action = action.to(device)
            target = target.to(device)
            t = torch.randint(0, diff.timesteps, (target.shape[0],), device=device)
            noise = torch.randn_like(target)
            x_t = diff.add_noise(target, t, noise)
            loss = F.mse_loss(model(x_t, ctx, t, action), diff.v_target(target, t, noise))
            opt.zero_grad()
            loss.backward()
            opt.step()
            ema.update(model)
            step += 1
            if step % args.log_every == 0:
                rate = (step - step0) / (time.time() - t0)
                print(f"step {step}/{args.steps}  loss {loss.item():.4f}  {rate:.2f} it/s")
                loss_w.writerow([step, f"{loss.item():.5f}"])
                loss_f.flush()
            if step % args.ckpt_every == 0:
                save()
    save()
    loss_f.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Smoke run**

Run: `python world_model/train.py --steps 5 --batch 8 --ckpt-every 5 --log-every 1`
Expected: prints `params: ~10M  device: mps ...`, 5 step lines with finite loss around 0.5-1.5, `saved checkpoint at step 5` (twice: periodic + final). Verify artifacts: `ls world_model/checkpoints/` shows `latest.pt` and `loss.csv`.

- [ ] **Step 4: Verify resume**

Run: `python world_model/train.py --steps 8 --batch 8 --ckpt-every 100 --log-every 1 --resume`
Expected: first line `resumed at step 5`, then steps 6-8, final `saved checkpoint at step 8`.

- [ ] **Step 5: Commit**

```bash
git add world_model/train.py .gitignore
git commit -m "Add diffusion training loop with EMA, checkpointing, and resume"
```

---

### Task 5: Metrics (PSNR + SSIM)

**Files:**
- Create: `world_model/eval.py` (metrics only; CLI comes in Task 6)
- Test: `world_model/test_wm.py` (append)

**Interfaces:**
- Produces: `psnr(a, b) -> (B,) tensor` and `ssim(a, b) -> (B,) tensor`, both taking `(B,3,H,W)` floats in `[0,1]`, per-image scores.

- [ ] **Step 1: Write the failing tests**

Append to `world_model/test_wm.py`:

```python
from eval import psnr, ssim


def test_metrics_identical_images():
    a = torch.rand(2, 3, 64, 64)
    assert psnr(a, a).min() > 50
    assert ssim(a, a).min() > 0.999


def test_metrics_degrade_with_noise():
    torch.manual_seed(0)
    a = torch.rand(2, 3, 64, 64)
    b = (a + 0.3 * torch.randn_like(a)).clamp(0, 1)
    assert psnr(a, b).max() < 25
    assert ssim(a, b).max() < 0.9
    assert psnr(a, a).min() > psnr(a, b).max()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest world_model/test_wm.py -v -k metrics`
Expected: FAIL with `ModuleNotFoundError: No module named 'eval'`

- [ ] **Step 3: Implement metrics**

Create `world_model/eval.py`:

```python
"""Evaluate the world model: PSNR/SSIM vs copy-last-frame, sample grid, rollout GIF."""
import math

import torch
import torch.nn.functional as F


def psnr(a, b):
    """Per-image PSNR. a, b: (B,3,H,W) in [0,1]. Returns (B,)."""
    mse = ((a - b) ** 2).flatten(1).mean(1).clamp_min(1e-10)
    return 10 * torch.log10(1.0 / mse)


def _gaussian_kernel(size=11, sigma=1.5):
    g = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-g ** 2 / (2 * sigma ** 2))
    k = g[:, None] * g[None, :]
    return k / k.sum()


def ssim(a, b):
    """Per-image SSIM (gaussian 11x11, per-channel then averaged).
    a, b: (B,3,H,W) in [0,1]. Returns (B,)."""
    k = _gaussian_kernel().to(a.device).repeat(3, 1, 1, 1)  # (3,1,11,11)
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mu_a = F.conv2d(a, k, groups=3)
    mu_b = F.conv2d(b, k, groups=3)
    var_a = F.conv2d(a * a, k, groups=3) - mu_a ** 2
    var_b = F.conv2d(b * b, k, groups=3) - mu_b ** 2
    cov = F.conv2d(a * b, k, groups=3) - mu_a * mu_b
    s = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / (
        (mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2)
    )
    return s.flatten(1).mean(1)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest world_model/test_wm.py -v -k metrics`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add world_model/eval.py world_model/test_wm.py
git commit -m "Add per-image PSNR and gaussian SSIM metrics"
```

---

### Task 6: Evaluation CLI (metrics report, sample grid, rollout GIF)

**Files:**
- Modify: `world_model/eval.py` (append)

**Interfaces:**
- Consumes: `psnr`, `ssim` (Task 5); `load_episodes`, `TransitionDataset`, `Diffusion`, `UNet` from `wm.py`; checkpoint format from Task 4.
- Produces: CLI writing `world_model/results/samples.png`, optional `world_model/results/rollout.gif`, and printing a metrics table.

Correctness of the pieces is covered by Tasks 1-5's unit tests; this task's deliverable check is the smoke run in step 2 against the smoke checkpoint.

- [ ] **Step 1: Implement the CLI**

Append to `world_model/eval.py`:

```python
import argparse
import os

import imageio
import numpy as np

from wm import CONTEXT, Diffusion, TransitionDataset, UNet, load_episodes, norm_frame


def to_uint8(x):
    """(3,H,W) in [-1,1] -> (H,W,3) uint8."""
    return ((x.clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).numpy()


def _batch(ds, idxs, device):
    ctx = torch.stack([ds[i][0] for i in idxs]).to(device)
    act = torch.stack([ds[i][1] for i in idxs]).to(device)
    gt = torch.stack([ds[i][2] for i in idxs]).to(device)
    return ctx, act, gt


@torch.no_grad()
def compute_metrics(model, diff, ds, device, num_eval, batch, ddim_steps):
    idxs = np.linspace(0, len(ds) - 1, num_eval).astype(int)
    scores = {"psnr": [], "ssim": [], "psnr_copy": [], "ssim_copy": []}
    for i in range(0, len(idxs), batch):
        ctx, act, gt = _batch(ds, idxs[i:i + batch], device)
        pred = diff.ddim_sample(model, ctx, act, ddim_steps)
        last = ctx[:, -3:]  # newest context frame = copy baseline
        gt01, pred01, last01 = [(z.clamp(-1, 1) + 1) / 2 for z in (gt, pred, last)]
        scores["psnr"].append(psnr(pred01, gt01).cpu())
        scores["ssim"].append(ssim(pred01, gt01).cpu())
        scores["psnr_copy"].append(psnr(last01, gt01).cpu())
        scores["ssim_copy"].append(ssim(last01, gt01).cpu())
    return {k: torch.cat(v).mean().item() for k, v in scores.items()}


@torch.no_grad()
def sample_grid(model, diff, ds, out_path, device, rows, ddim_steps):
    """Rows of: 4 context frames | ground truth | 3 independent samples."""
    idxs = np.linspace(0, len(ds) - 1, rows).astype(int)
    ctx, act, gt = _batch(ds, idxs, device)
    samples = [diff.ddim_sample(model, ctx, act, ddim_steps).cpu() for _ in range(3)]
    row_imgs = []
    for r in range(rows):
        cells = [ctx[r, 3 * k:3 * k + 3].cpu() for k in range(CONTEXT)]
        cells += [gt[r].cpu()] + [s[r] for s in samples]
        row_imgs.append(np.concatenate([to_uint8(c) for c in cells], axis=1))
    imageio.imwrite(out_path, np.concatenate(row_imgs, axis=0))
    print(f"wrote {out_path} (cols: {CONTEXT} context | ground truth | 3 samples)")


@torch.no_grad()
def rollout_gif(model, diff, ep, out_path, device, n, ddim_steps):
    """Autoregressive dream replaying recorded actions; GIF shows real|dream."""
    n = min(n, len(ep["actions"]) - 1)
    ctx_frames = [norm_frame(ep["obs"][0])] * CONTEXT
    frames = []
    for t in range(n):
        ctx = torch.cat(ctx_frames, dim=0)[None].to(device)
        act = torch.tensor([int(ep["actions"][t])], device=device)
        pred = diff.ddim_sample(model, ctx, act, ddim_steps)[0].cpu()
        ctx_frames = ctx_frames[1:] + [pred]
        frames.append(np.concatenate([ep["obs"][t + 1], to_uint8(pred)], axis=1))
    imageio.mimsave(out_path, frames, fps=15, loop=0)
    print(f"wrote {out_path} ({n} steps, left=real right=dream)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="world_model/checkpoints/latest.pt")
    p.add_argument("--data", default="dataset")
    p.add_argument("--out-dir", default="world_model/results")
    p.add_argument("--num-eval", type=int, default=256)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--ddim-steps", type=int, default=20)
    p.add_argument("--grid-rows", type=int, default=6)
    p.add_argument("--rollout", type=int, default=0, help="rollout GIF length (0 = off)")
    args = p.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    _, val_eps = load_episodes(args.data)
    ds = TransitionDataset(val_eps)
    model = UNet().to(device)
    ck = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ck["ema"])
    model.eval()
    diff = Diffusion()
    print(f"checkpoint step {ck['step']}  val transitions: {len(ds)}  device: {device}")

    m = compute_metrics(model, diff, ds, device, args.num_eval, args.batch, args.ddim_steps)
    print(f"model      PSNR {m['psnr']:6.2f}  SSIM {m['ssim']:.4f}")
    print(f"copy-last  PSNR {m['psnr_copy']:6.2f}  SSIM {m['ssim_copy']:.4f}")

    os.makedirs(args.out_dir, exist_ok=True)
    sample_grid(model, diff, ds, os.path.join(args.out_dir, "samples.png"),
                device, args.grid_rows, args.ddim_steps)
    if args.rollout:
        rollout_gif(model, diff, val_eps[0], os.path.join(args.out_dir, "rollout.gif"),
                    device, args.rollout, args.ddim_steps)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke run against the Task 4 smoke checkpoint**

Run: `python world_model/eval.py --num-eval 16 --rollout 10 --ddim-steps 5`
Expected: prints checkpoint step, a metrics table where **copy-last beats the model** (only 8 train steps — noise output is correct here), and writes `world_model/results/samples.png` + `world_model/results/rollout.gif`. Open `samples.png` and confirm the layout: 6 rows × 8 tiles (4 context, 1 ground truth, 3 noisy samples).

- [ ] **Step 3: Run the full test suite**

Run: `pytest world_model/test_wm.py -v`
Expected: 11 passed

- [ ] **Step 4: Commit**

```bash
git add world_model/eval.py
git commit -m "Add evaluation CLI: metrics vs copy baseline, sample grid, rollout GIF"
```

---

### Task 7: Real training run (user-driven)

Not an agent task — hand off to the user:

- [ ] Start training: `python world_model/train.py` (100k steps, overnight; Ctrl-C anytime — `--resume` continues).
- [ ] After ~20k steps, sanity-check: `python world_model/eval.py --num-eval 64` — model PSNR should now beat copy-last, samples should look like CoinRun.
- [ ] Final deliverables: `python world_model/eval.py --rollout 100` → metrics table, `samples.png`, `rollout.gif`, and the loss curve in `world_model/checkpoints/loss.csv`.

**Success criteria (from spec):** val PSNR meaningfully above copy-last baseline; sharp, action-consistent samples; full pipeline runs end-to-end on the M3 Air.
