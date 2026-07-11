"""Play CoinRun inside the world model: keys drive the dream.

Usage:
    python world_model/play.py                  # arranca desde un episodio de validacion
    python world_model/play.py --ddim-steps 5   # mas rapido, algo menos nitido

El suenno avanza continuamente (accion "quieto" por defecto); manten presionada
una tecla para actuar:
    <- izquierda | -> derecha | ^ salto (derecha+salto)
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

    state = {"ep": 0, "ctx": None, "steps": 0, "action": NOOP}

    def reset():
        ep = val_eps[state["ep"] % len(val_eps)]
        state["ep"] += 1
        state["ctx"] = [norm_frame(ep["obs"][0])] * CONTEXT
        state["steps"] = 0
        state["action"] = NOOP
        im.set_data(to_img(state["ctx"][-1]))
        fig.canvas.draw_idle()

    def on_press(event):
        if event.key == "q":
            plt.close(fig)
        elif event.key == "r":
            reset()
        elif event.key in KEY_TO_ACTION:
            state["action"] = KEY_TO_ACTION[event.key]

    def on_release(event):
        if KEY_TO_ACTION.get(event.key) == state["action"]:
            state["action"] = NOOP

    def tick():
        state["ctx"], pred = dream_step(
            model, diff, state["ctx"], state["action"], device, args.ddim_steps)
        state["steps"] += 1
        im.set_data(to_img(pred))
        ax.set_title(f"suenna: paso {state['steps']}  (accion {state['action']})")
        fig.canvas.draw_idle()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axis("off")
    im = ax.imshow(torch.zeros(64, 64, 3).numpy())
    fig.canvas.mpl_connect("key_press_event", on_press)
    fig.canvas.mpl_connect("key_release_event", on_release)
    reset()
    timer = fig.canvas.new_timer(interval=66)  # corre al ritmo que de la GPU
    timer.add_callback(tick)
    timer.start()
    print("ventana abierta: manten las flechas para actuar, r reinicia, q sale "
          f"(en {device} con ddim {args.ddim_steps})")
    plt.show()


if __name__ == "__main__":
    main()
