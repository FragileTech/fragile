"""Debug script: compare serial vs parallel AtariFractalGas on MsPacman."""
import os
import time
import traceback

os.environ.pop("DISPLAY", None)
os.environ["PYGLET_HEADLESS"] = "1"
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["MPLBACKEND"] = "Agg"

import numpy as np
import torch

import plangym
from fragile.fractalai.videogames.atari_gas import AtariFractalGas


def run_gas(env, label, n_iters=80, N=20, seed=42, record_frames=True):
    """Run AtariFractalGas and print per-iteration diagnostics."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  env type: {type(env).__name__}, N={N}, iters={n_iters}")
    print(f"  record_frames={record_frames}")
    print(f"{'='*60}")

    gas = AtariFractalGas(
        env=env, N=N, seed=seed, record_frames=record_frames, dt_range=(3, 8),
    )

    t0 = time.time()
    try:
        state = gas.reset()
    except Exception:
        print("ERROR during reset:")
        traceback.print_exc()
        return

    t_reset = time.time() - t0
    print(f"  reset took {t_reset:.2f}s, {state.N} walkers")

    for i in range(n_iters):
        t_step = time.time()
        try:
            state, info = gas.step(state)
        except Exception:
            print(f"  ERROR at iteration {i}:")
            traceback.print_exc()
            return

        elapsed = time.time() - t_step
        alive = info["alive_count"]
        max_r = info["max_reward"]
        mean_r = info["mean_reward"]
        cloned = info["num_cloned"]
        frame = info.get("best_frame")
        frame_str = f"frame={frame.shape}" if frame is not None else "frame=None"

        # Print every iteration for full visibility
        print(
            f"  iter {i:3d} | {elapsed:5.2f}s | alive={alive:3d} | "
            f"cloned={cloned:2d} | max_r={max_r:6.1f} | mean_r={mean_r:6.1f} | {frame_str}"
        )

        # Sanity checks
        if alive == 0:
            print("  WARNING: all walkers dead!")
        if elapsed < 0.001:
            print("  WARNING: step took <1ms — suspiciously fast, env may not be stepping")

    total = time.time() - t0
    print(f"  DONE in {total:.1f}s  ({n_iters/total:.1f} iter/s)")
    print(f"  Final max_reward={info['max_reward']:.1f}, alive={info['alive_count']}")


if __name__ == "__main__":
    game = "ALE/MsPacman-v5"
    N = 20
    n_iters = 40

    # Common kwargs — frameskip=5 is critical: VectorizedEnv defaults to
    # frameskip=1, overriding AtariEnv's default of 5.
    common = dict(
        obs_type="ram", render_mode=None,
        autoreset=False, remove_time_limit=True, episodic_life=True,
        frameskip=5,
    )

    # --- SERIAL ---
    env_serial = plangym.make(game, **common)
    run_gas(env_serial, f"SERIAL (n_workers=1) — {game}", n_iters=n_iters, N=N)
    env_serial.close()

    # --- PARALLEL n_workers=2 ---
    env_par2 = plangym.make(game, n_workers=2, **common)
    run_gas(env_par2, f"PARALLEL (n_workers=2) — {game}", n_iters=n_iters, N=N)
    env_par2.close()

    # --- PARALLEL n_workers=4 ---
    env_par4 = plangym.make(game, n_workers=4, **common)
    run_gas(env_par4, f"PARALLEL (n_workers=4) — {game}", n_iters=n_iters, N=N)
    env_par4.close()

    # Check for orphan processes
    import subprocess
    result = subprocess.run(
        ["ps", "aux"], capture_output=True, text=True,
    )
    ale_procs = [l for l in result.stdout.splitlines() if "ale" in l.lower() or "plangym" in l.lower()]
    if ale_procs:
        print(f"\nWARNING: {len(ale_procs)} orphan ALE/plangym processes found:")
        for p in ale_procs:
            print(f"  {p}")
    else:
        print("\nNo orphan processes found — cleanup OK")
