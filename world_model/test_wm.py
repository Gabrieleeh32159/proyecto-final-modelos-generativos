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
