"""Evaluate the world model: PSNR/SSIM vs copy-last-frame, sample grid, rollout GIF."""
import argparse
import os

import imageio
import numpy as np
import torch
import torch.nn.functional as F

from wm import CONTEXT, Diffusion, TransitionDataset, UNet, load_episodes, norm_frame


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


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


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

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
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
