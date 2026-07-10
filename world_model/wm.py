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
