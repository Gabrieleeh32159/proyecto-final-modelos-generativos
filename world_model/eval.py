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
