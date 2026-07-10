import numpy as np
import torch

from wm import CONTEXT, NUM_ACTIONS, T_TRAIN, TransitionDataset, load_episodes, Diffusion


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
