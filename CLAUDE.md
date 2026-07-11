# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A fork of [openai/procgen](https://github.com/openai/procgen) used for a UTEC generative-models course project. Upstream procgen is 16 procedurally-generated RL environments (C++ game logic + Qt rendering) exposed to Python via the gym3 `libenv` C interface. This fork adds `record.py`, which records human CoinRun gameplay into `.npz` episode datasets (under `dataset/` and `coinrun_dataset/`) for training a World Model.

Local deviations from upstream:
- `procgen/CMakeLists.txt`: `-march=native` replaced with `-mcpu=apple-m1` (Apple Silicon).
- `chaser.cpp` / `starpilot.cpp`: `[[maybe_unused]]` annotations to silence warnings that break the build with newer clang.
- Upstream is in maintenance mode and deliberately never fixes gameplay bugs (reproducibility of published results) — don't "fix" known environment issues listed in README.md.

## Commands

Requires a conda env with Qt5 (see `environment.yml`): `conda env update --name procgen --file environment.yml && conda activate procgen`, then `pip install -e .`.

There is no separate build step: importing `procgen` triggers a cmake build of the C++ code into `procgen/.build/` (see `procgen/builder.py`). After editing C++, just re-run any Python entry point and it rebuilds. Prints `building procgen...done` on success. `debug=True` env kwarg (or editing while iterating) uses a debug build. If cmake can't find Qt, set `PROCGEN_CMAKE_PREFIX_PATH`.

```bash
# play an environment interactively
python -m procgen.interactive --env-name coinrun

# record a human CoinRun dataset session (writes coinrun_dataset/session_<timestamp>/)
python record.py --difficulty easy

# smoke test
python -c "from procgen import ProcgenGym3Env; ProcgenGym3Env(num=1, env_name='coinrun')"

# tests (pytest, live inside the package)
pytest procgen/env_test.py
pytest procgen/env_test.py::test_seeding          # single test
pytest procgen/state_test.py                      # get_state/set_state determinism (slow)
```

## Architecture

Python layer (`procgen/`):
- `builder.py` — compiles the C++ into a shared library at import time (process- and file-locked).
- `env.py` — `ProcgenGym3Env`: loads the shared lib through `gym3.libenv`, defines `ENV_NAMES`, the 15-action button-combo table, and all env options (`num_levels`, `start_level`, `distribution_mode`, etc.). Options are serialized to C++ via `vecoptions.cpp`.
- `gym_registration.py` — wraps gym3 env for the classic gym API (`procgen:procgen-<name>-v0`).
- `interactive.py` — Qt-based human play; `record.py` subclasses its `ProcgenInteractive` to persist episodes.

C++ layer (`procgen/src/`):
- `vecgame.cpp` implements the `libenv` C interface: owns the vector of games, threading, and the `info` dict tensor definitions (add a `libenv_tensortype` in its constructor to expose new info keys to Python).
- `game.cpp` / `basic-abstract-game.cpp` — base classes. Every game in `src/games/*.cpp` subclasses `BasicAbstractGame`, which provides the 2D grid world, entity/physics/collision helpers, and asset handling. Games self-register via `game-registry.cpp` (a static registrar object in each game file), so adding a game = copy `games/bigfish.cpp`, rename, add to `CMakeLists.txt`.
- Determinism matters: all randomness must go through `rand_gen` (`randgen.cpp`), and full game state must round-trip through `serialize`/`deserialize` (verified by `state_test.py`).
- Rendering uses Qt (`QPainter`); observations are 64x64x3 RGB, with an optional hi-res `info["rgb"]` when `render_mode="rgb_array"`.

## record.py dataset format

Each session dir holds `episode_NNNNN.npz` (`observations (T,64,64,3) uint8`, `actions (T,) uint8`, `rewards (T,) float32`, plus `level_seed`, `completed`, `truncated`, `episode_length`, `total_reward`) and a `session.json` manifest. Episodes shorter than `--min-steps` (default 10) are discarded. The recorder wraps the gym3 env: it renders 512x512 for the human via `info["rgb"]` while storing the true 64x64 agent observation.
