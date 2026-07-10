# CoinRun Next-Frame Diffusion World Model — Design

**Date:** 2026-07-10
**Status:** Approved

## Goal

Train a generative model of CoinRun dynamics from the human gameplay dataset in
`dataset/`: given the last 4 frames and the action taken, generate the next
frame. Deliverable is next-frame prediction with quantitative metrics
(PSNR/SSIM), plus qualitative sample grids and an optional autoregressive
rollout GIF. Must train on a MacBook Air M3, 16 GB RAM (PyTorch MPS).

## Dataset (already recorded)

- `dataset/session_*/episode_*.npz` — 21 sessions, 1,904 episodes, ~165k steps.
- Per episode: `observations (T,64,64,3) uint8`, `actions (T,) uint8`,
  `rewards (T,) float32`, plus metadata. `observations[t]` is the frame seen
  *before* `actions[t]`.
- Only actions {1, 4, 7, 8} occur (left, noop, right, right+jump), but the
  action embedding covers all 15 — no remapping.
- Uncompressed frames ≈ 2 GB uint8 → load everything into RAM once.

## Architecture

Conditional DDPM (DIAMOND-style, scaled down):

- **UNet ~15M params.** Base channels 64, multipliers (1, 2, 4) →
  64/128/256 at resolutions 64/32/16, 2 residual blocks per level,
  self-attention at the 16×16 level and in the bottleneck.
- **Input:** 15 channels — noisy target frame (3) concatenated with 4 context
  frames (12), all normalized to [-1, 1].
- **Conditioning:** sinusoidal timestep embedding + learned action embedding
  (15 entries), summed and injected into every residual block (FiLM/shift-scale,
  standard DDPM conditioning).
- **Objective:** v-prediction. Cosine noise schedule, 1000 train timesteps.
- **Sampling:** DDIM, 20 steps.

## Training

- PyTorch on MPS, fp32. AdamW, lr 1e-4, batch 32 (fallback 16 if memory
  pressure), EMA of weights with decay 0.999.
- A training example is an index (episode, t): context = frames t-3..t
  (edge-padded by repeating the first frame when t < 3), action = actions[t],
  target = frame t+1. ~157k usable examples.
- **Split by session:** 19 sessions train / 2 sessions val (fixed, listed in
  code) so no level leaks between splits.
- Checkpoint (model + EMA + optimizer + step) every 2k steps to
  `world_model/checkpoints/`; `--resume` continues from the latest. Loss logged
  to a CSV. Target ~100k steps; usable samples expected by ~20k.

## Evaluation (`world_model/eval.py`)

- **One-step PSNR + SSIM** on val sessions: predict frame t+1 from real
  context via DDIM, compare to ground truth. SSIM implemented inline
  (~15 lines, gaussian-window version) — no scikit-image dependency.
- **Sample grid PNG:** rows of (context | ground truth | 3 independent
  samples) to show sharpness and diversity.
- **Rollout GIF (optional flag):** N-step autoregressive dream — feed
  predictions back as context, replay the recorded human actions.

## Files

All new; nothing upstream is touched.

- `world_model/wm.py` — dataset loading, UNet, diffusion process (one module)
- `world_model/train.py` — training loop, checkpointing, resume, loss CSV
- `world_model/eval.py` — PSNR/SSIM, sample grid, rollout GIF

Dependencies: `torch`, `numpy`, `imageio` (GIF only).

## Risks / mitigations

- **Thermal throttling** on the fanless Air → checkpoints are useful early;
  training can be stopped and resumed at any time.
- **Memory:** dataset ~2 GB + model + batch fits comfortably in 16 GB; batch
  drops to 16 if needed.
- **MPS op gaps:** stick to plain conv/attention/GroupNorm ops that MPS
  supports; no flash-attention or custom kernels.

## Success criteria

- Val PSNR meaningfully above the copy-last-frame baseline (report both).
- Sample grid shows sharp, plausible next frames with correct
  action-dependent motion (e.g., jump vs. walk).
- Full pipeline (train → eval → figures) runs end-to-end on the M3 Air.
