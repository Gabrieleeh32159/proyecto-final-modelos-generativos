"""Play CoinRun inside the world model: keys drive the dream.

Usage:
    python world_model/play.py                  # arranca desde un episodio de validacion
    python world_model/play.py --ddim-steps 5   # mas rapido, algo menos nitido

Teclas: <- izquierda | -> derecha | ^ salto (derecha+salto) | otra = quieto
        r = reiniciar desde otro episodio | q = salir
"""
import argparse

import matplotlib.pyplot as plt
import torch

from wm import CONTEXT, Diffusion, UNet, load_episodes, norm_frame

# combos de env.py: 1=LEFT, 4=noop, 7=RIGHT, 8=RIGHT+UP (las 4 acciones del dataset)
KEY_TO_ACTION = {"left": 1, "right": 7, "up": 8}
NOOP = 4


@torch.no_grad()
def dream_step(model, diff, ctx_frames, action, device, ddim_steps):
    """ctx_frames: list of CONTEXT (3,64,64) cpu tensors, oldest..newest."""
    ctx = torch.cat(ctx_frames, dim=0)[None].to(device)
    act = torch.tensor([action], device=device)
    pred = diff.ddim_sample(model, ctx, act, ddim_steps)[0].cpu()
    return ctx_frames[1:] + [pred], pred


def to_img(x):
    return ((x.clamp(-1, 1) + 1) / 2).permute(1, 2, 0).numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="world_model/checkpoints/latest.pt")
    p.add_argument("--data", default="dataset")
    p.add_argument("--ddim-steps", type=int, default=10)
    args = p.parse_args()

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    _, val_eps = load_episodes(args.data)
    model = UNet().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device)["ema"])
    model.eval()
    diff = Diffusion()

    state = {"ep": 0, "ctx": None, "steps": 0}

    def reset():
        ep = val_eps[state["ep"] % len(val_eps)]
        state["ep"] += 1
        state["ctx"] = [norm_frame(ep["obs"][0])] * CONTEXT
        state["steps"] = 0
        im.set_data(to_img(state["ctx"][-1]))
        ax.set_title("suenna: paso 0 — usa las flechas")
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == "q":
            plt.close(fig)
            return
        if event.key == "r":
            reset()
            return
        action = KEY_TO_ACTION.get(event.key, NOOP)
        ax.set_title("generando...")
        fig.canvas.draw()
        fig.canvas.flush_events()
        state["ctx"], pred = dream_step(
            model, diff, state["ctx"], action, device, args.ddim_steps)
        state["steps"] += 1
        im.set_data(to_img(pred))
        ax.set_title(f"suenna: paso {state['steps']}  (accion {action})")
        fig.canvas.draw_idle()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axis("off")
    im = ax.imshow(torch.zeros(64, 64, 3).numpy())
    fig.canvas.mpl_connect("key_press_event", on_key)
    reset()
    print("ventana abierta: flechas para jugar, r reinicia, q sale "
          f"(~1 frame/s en {device} con ddim {args.ddim_steps})")
    plt.show()


if __name__ == "__main__":
    main()
