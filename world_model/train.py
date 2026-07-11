"""Train the CoinRun next-frame diffusion model.

Usage:
    python world_model/train.py                 # full run (100k steps)
    python world_model/train.py --resume        # continue from latest.pt
"""
import argparse
import csv
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

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
        tmp = ckpt_path + ".tmp"
        torch.save(
            {"model": model.state_dict(), "ema": ema.shadow,
             "opt": opt.state_dict(), "step": step},
            tmp,
        )
        os.replace(tmp, ckpt_path)
        tqdm.write(f"saved checkpoint at step {step}")

    with open(os.path.join(args.out, "loss.csv"), "a", newline="") as loss_f:
        loss_w = csv.writer(loss_f)

        model.train()
        pbar = tqdm(total=args.steps, initial=step, unit="step", dynamic_ncols=True)
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
                pbar.update(1)
                if step % args.log_every == 0:
                    pbar.set_postfix(loss=f"{loss.item():.4f}")
                    loss_w.writerow([step, f"{loss.item():.5f}"])
                    loss_f.flush()
                if step % args.ckpt_every == 0:
                    save()
        pbar.close()
        save()


if __name__ == "__main__":
    main()
