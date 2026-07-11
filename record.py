"""
Grabar partidas de CoinRun (procgen)

Renderiza simultaneamente a 64x64 (observacion del agente) y 512x512 (display humano). 
Uso:
    python record.py
    python record.py --difficulty  easy

Dificutades disponibles: 
    easy, hard
    
Formato por episodio (episode_NNNNN.npz):
    observations : (T, 64, 64, 3) uint8   estado visto ANTES de cada accion
    actions      : (T,)           uint8   accion tomada en cada estado (0..14)
    rewards      : (T,)           float32 reward recibido por cada accion
    level_seed   : ()             int32
    completed    : ()             bool    termino por completar nivel
    truncated    : ()             bool    el script se cerro a mitad
    episode_length : ()           int32
    total_reward   : ()           float32
"""

import argparse
import datetime as dt
import json
import platform
import time
from pathlib import Path

import numpy as np

from gym3 import Wrapper
from procgen import ProcgenGym3Env
from procgen.interactive import ProcgenInteractive


DATASET_SCHEMA_VERSION = 1
class CoinRunDatasetRecorder(Wrapper):
    def __init__(self, env, session_dir, min_steps_to_keep=10):
        super().__init__(env)
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.min_steps_to_keep = min_steps_to_keep

        self._obs_buf = []
        self._action_buf = []
        self._reward_buf = []
        self._level_seed = None

        self._pending_action = None
        self._last_obs = None
        self._initialized = False

        self.episodes_saved = 0
        self.episode_records = []
        self.total_steps = 0
        self.start_time = time.time()

    def act(self, ac):
        self._pending_action = int(np.asarray(ac).flatten()[0])
        return self.env.act(ac)

    def observe(self):
        rew, obs, first = self.env.observe()

        # observe() es idempotente entre acts; solo registramos cuando hay
        # una accion pendiente de cerrar.
        if self._pending_action is not None:
            self._handle_transition(rew, obs, first)
            self._pending_action = None
        elif not self._initialized:
            self._begin_new_episode_with(obs)
            self._initialized = True

        self._last_obs = obs["rgb"][0].copy()
        return rew, obs, first

    def _begin_new_episode_with(self, obs):
        self._obs_buf = [obs["rgb"][0].copy()]
        self._action_buf = []
        self._reward_buf = []
        try:
            info = self.env.get_info()[0]
            self._level_seed = int(info.get("level_seed", -1))
        except Exception:
            self._level_seed = -1

    def _handle_transition(self, rew, obs, first):
        action = self._pending_action
        reward = float(np.asarray(rew).flatten()[0])
        is_first = bool(np.asarray(first).flatten()[0])
        next_obs = obs["rgb"][0]
        info = self.env.get_info()[0]

        self._action_buf.append(action)
        self._reward_buf.append(reward)

        if is_first:
            # Episodio terminado: next_obs ya es del siguiente.
            completed = bool(info.get("prev_level_complete", 0))
            prev_seed = int(info.get("prev_level_seed", self._level_seed))
            self._save_current_episode(
                level_seed=prev_seed, completed=completed, truncated=False,
            )
            self._obs_buf = [next_obs.copy()]
            self._action_buf = []
            self._reward_buf = []
            self._level_seed = int(info.get("level_seed", -1))
        else:
            self._obs_buf.append(next_obs.copy())

    def _save_current_episode(self, level_seed, completed, truncated):
        T = len(self._action_buf)
        if T < self.min_steps_to_keep:
            return

        obs_arr = np.stack(self._obs_buf[:T], axis=0).astype(np.uint8)
        action_arr = np.array(self._action_buf, dtype=np.uint8)
        reward_arr = np.array(self._reward_buf, dtype=np.float32)

        ep_idx = self.episodes_saved + 1
        filename = f"episode_{ep_idx:05d}.npz"
        filepath = self.session_dir / filename

        np.savez_compressed(
            filepath,
            observations=obs_arr,
            actions=action_arr,
            rewards=reward_arr,
            level_seed=np.int32(level_seed),
            completed=np.bool_(completed),
            truncated=np.bool_(truncated),
            episode_length=np.int32(T),
            total_reward=np.float32(reward_arr.sum()),
        )

        self.episodes_saved += 1
        self.total_steps += T
        self.episode_records.append({
            "file": filename,
            "length": int(T),
            "total_reward": float(reward_arr.sum()),
            "completed": bool(completed),
            "truncated": bool(truncated),
            "level_seed": int(level_seed),
        })

        outcome = "WIN " if completed else "loss"
        print(f"ep {ep_idx:05d}  T={T:>4d}  reward={reward_arr.sum():+.2f}  "
              f"{outcome}  seed={level_seed}", flush=True)

    def flush_on_close(self):
        if len(self._action_buf) >= self.min_steps_to_keep:
            self._save_current_episode(
                level_seed=self._level_seed if self._level_seed is not None else -1,
                completed=False,
                truncated=True,
            )

    def summary(self):
        elapsed = time.time() - self.start_time
        wins = sum(1 for e in self.episode_records if e["completed"])
        avg_len = (np.mean([e["length"] for e in self.episode_records])
                   if self.episode_records else 0.0)
        avg_rew = (np.mean([e["total_reward"] for e in self.episode_records])
                   if self.episode_records else 0.0)
        return {
            "episodes": self.episodes_saved,
            "total_steps": self.total_steps,
            "wins": int(wins),
            "win_rate": wins / max(1, self.episodes_saved),
            "avg_episode_length": float(avg_len),
            "avg_episode_reward": float(avg_rew),
            "elapsed_seconds": float(elapsed),
        }


def write_session_manifest(session_dir, header, recorder):
    manifest = dict(header)
    manifest["episodes"] = recorder.episode_records
    manifest["summary"] = recorder.summary()
    with open(Path(session_dir) / "session.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


class RecordingInteractive(ProcgenInteractive):
    """ProcgenInteractive que persiste el manifest tras cada episodio guardado."""

    def __init__(self, *args, recorder, session_dir, manifest_header, **kwargs):
        super().__init__(*args, **kwargs)
        self._recorder = recorder
        self._session_dir = session_dir
        self._manifest_header = manifest_header
        self._last_persisted_count = 0

    def _update(self, dt_, keys_clicked, keys_pressed):
        super()._update(dt_, keys_clicked, keys_pressed)
        if self._recorder.episodes_saved != self._last_persisted_count:
            write_session_manifest(
                self._session_dir, self._manifest_header, self._recorder
            )
            self._last_persisted_count = self._recorder.episodes_saved


def build_session_dir(output_root):
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    session_id = f"session_{timestamp}"
    return Path(output_root) / session_id, session_id


def get_procgen_version():
    try:
        from importlib.metadata import version
        return version("procgen")
    except Exception:
        return "unknown"


def main():
    parser = argparse.ArgumentParser(
        description="Graba partidas humanas de CoinRun para entrenar un "
                    "World Model. Visualizacion en hi-res, dataset en 64x64."
    )
    parser.add_argument("--output-dir", default="./coinrun_dataset")
    parser.add_argument("--difficulty", default="easy",
                        choices=["easy", "hard", "extreme", "memory", "exploration"])
    parser.add_argument("--num-levels", type=int, default=0,
                        help="0 = infinitos niveles")
    parser.add_argument("--start-level", type=int, default=0)
    parser.add_argument("--rand-seed", type=int, default=None)
    parser.add_argument("--min-steps", type=int, default=10,
                        help="Descartar episodios mas cortos que N steps")
    parser.add_argument("--display-scale", type=int, default=12,
                        help="Multiplicador de 64 para el tamano de ventana")
    args = parser.parse_args()

    session_dir, session_id = build_session_dir(args.output_dir)
    session_dir.mkdir(parents=True, exist_ok=False)
    print(f"session: {session_dir}", flush=True)

    env_kwargs = dict(
        env_name="coinrun",
        distribution_mode=args.difficulty,
        num_levels=args.num_levels,
        start_level=args.start_level,
        render_mode="rgb_array",  # activa info["rgb"] hi-res sin tocar la obs 64x64
        center_agent=True,
        use_backgrounds=True,
        restrict_themes=False,
        use_monochrome_assets=False,
        paint_vel_info=False,
        use_generated_assets=False,
    )
    if args.rand_seed is not None:
        env_kwargs["rand_seed"] = args.rand_seed

    env = ProcgenGym3Env(num=1, **env_kwargs)

    recorder = CoinRunDatasetRecorder(
        env,
        session_dir=session_dir,
        min_steps_to_keep=args.min_steps,
    )

    manifest_header = {
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "session_id": session_id,
        "start_time_iso": dt.datetime.now().isoformat(),
        "env_name": "coinrun",
        "observation_shape": [64, 64, 3],
        "action_space": {
            "type": "Discrete",
            "n": 15,
            "combos": [
                ["LEFT", "DOWN"], ["LEFT"], ["LEFT", "UP"],
                ["DOWN"], [], ["UP"],
                ["RIGHT", "DOWN"], ["RIGHT"], ["RIGHT", "UP"],
                ["D"], ["A"], ["W"], ["S"], ["Q"], ["E"],
            ],
        },
        "env_kwargs": {k: v for k, v in env_kwargs.items() if k != "render_mode"},
        "procgen_version": get_procgen_version(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }
    write_session_manifest(session_dir, manifest_header, recorder)

    h, w, _ = recorder.ob_space["rgb"].shape
    scale = args.display_scale

    ia = RecordingInteractive(
        recorder,
        ob_key=None,
        info_key="rgb",
        width=w * scale,
        height=h * scale,
        recorder=recorder,
        session_dir=session_dir,
        manifest_header=manifest_header,
    )

    try:
        ia.run()
    except KeyboardInterrupt:
        pass
    finally:
        recorder.flush_on_close()
        manifest_header["end_time_iso"] = dt.datetime.now().isoformat()
        write_session_manifest(session_dir, manifest_header, recorder)

        s = recorder.summary()
        print(f"\ndone: {s['episodes']} episodios, {s['total_steps']} steps, "
              f"win_rate={100*s['win_rate']:.1f}%, "
              f"avg_reward={s['avg_episode_reward']:+.2f}")


if __name__ == "__main__":
    main()