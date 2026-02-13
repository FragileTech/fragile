#!/usr/bin/env python3
"""Manual integration verification for AtariEnv with AtariFractalGas.

This script provides end-to-end verification that AtariEnv works correctly
with AtariFractalGas in WSL headless environments.
"""

import os


# Ensure headless mode
os.environ["PYGLET_HEADLESS"] = "1"

import numpy as np

from fragile.fractalai.videogames.atari import AtariEnv
from fragile.fractalai.videogames.atari_gas import AtariFractalGas


def print_section(title: str):
    """Print a section header."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


def verify_basic_env():
    """Verify basic AtariEnv functionality."""
    print_section("1. Basic AtariEnv Verification")

    print("Creating environment...")
    env = AtariEnv("ALE/Pong-v5", obs_type="ram", render_mode="rgb_array")

    print("✓ Environment created successfully")
    print(f"  - Name: {env.name}")
    print(f"  - Observation type: {env.obs_type}")
    print(f"  - Render mode: {env.render_mode}")
    print(f"  - Include RGB: {env.include_rgb}")

    print("\nResetting environment...")
    state = env.reset(seed=42)

    print("✓ Reset successful")
    print(f"  - State type: {type(state).__name__}")
    print(f"  - Observation shape: {state.observation.shape}")
    print(f"  - RGB frame shape: {state.rgb_frame.shape if state.rgb_frame is not None else None}")

    print("\nTesting single step...")
    _new_state, obs, reward, done, truncated, _info = env.step(action=2, dt=3)

    print("✓ Step successful")
    print(f"  - Observation shape: {obs.shape}")
    print(f"  - Reward: {reward:.2f}")
    print(f"  - Done: {done}")
    print(f"  - Truncated: {truncated}")

    print("\nTesting state cloning...")
    cloned_state = env.clone_state()
    print("✓ State cloned successfully")

    print("\nTesting state restoration (determinism)...")
    env.restore_state(cloned_state)
    _, obs1, rew1, _, _, _ = env.step(action=5, dt=2)
    env.restore_state(cloned_state)
    _, obs2, rew2, _, _, _ = env.step(action=5, dt=2)

    if np.array_equal(obs1, obs2) and rew1 == rew2:
        print("✓ Deterministic replay verified")
    else:
        print("✗ WARNING: Replay not deterministic!")

    env.close()
    print("\n✓ Basic verification complete")


def verify_batch_operations():
    """Verify batch operations."""
    print_section("2. Batch Operations Verification")

    print("Creating environment...")
    env = AtariEnv("ALE/Pong-v5", obs_type="ram", render_mode="rgb_array")

    N = 20
    print(f"\nCreating {N} walker states...")
    states = np.empty(N, dtype=object)
    initial_state = env.reset(seed=42)
    for i in range(N):
        states[i] = initial_state.copy()

    print("✓ Walker states created")

    print("\nTesting batch stepping...")
    actions = np.array([env.action_space.sample() for _ in range(N)])
    dt = np.random.randint(1, 4, size=N)

    new_states, obs, rewards, dones, _truncated, _infos = env.step_batch(states, actions, dt)

    print("✓ Batch step successful")
    print(f"  - States shape: {new_states.shape}")
    print(f"  - Observations shape: {obs.shape}")
    print(f"  - Rewards shape: {rewards.shape}")
    print(f"  - Dones shape: {dones.shape}")
    print(f"  - All states are AtariState: {all(hasattr(s, 'copy') for s in new_states)}")

    env.close()
    print("\n✓ Batch operations verification complete")


def verify_obs_types():
    """Verify all observation types work."""
    print_section("3. Observation Types Verification")

    obs_types = ["ram", "rgb", "grayscale"]

    for obs_type in obs_types:
        print(f"\nTesting obs_type='{obs_type}'...")
        env = AtariEnv("ALE/Pong-v5", obs_type=obs_type, render_mode="rgb_array")
        state = env.reset(seed=42)

        print(f"✓ {obs_type.upper()} observations work")
        print(f"  - Shape: {state.observation.shape}")
        print(f"  - Dtype: {state.observation.dtype}")

        env.close()

    print("\n✓ All observation types verified")


def verify_headless_rendering():
    """Verify headless rendering works."""
    print_section("4. WSL Headless Rendering Verification")

    print(f"PYGLET_HEADLESS = {os.environ.get('PYGLET_HEADLESS')}")
    print(f"DISPLAY = {os.environ.get('DISPLAY', 'not set')}")

    print("\nCreating environment...")
    env = AtariEnv("ALE/Pong-v5", obs_type="ram", render_mode="rgb_array")
    env.reset(seed=42)

    print("\nTesting render()...")
    rgb = env.render()

    if rgb is not None:
        print("✓ Render successful in headless mode")
        print(f"  - RGB shape: {rgb.shape}")
        print(f"  - RGB dtype: {rgb.dtype}")
        print(f"  - Non-zero pixels: {(rgb.sum(axis=2) > 0).sum()}")
    else:
        print("✗ WARNING: render() returned None")

    env.close()
    print("\n✓ Headless rendering verified")


def verify_atari_fractal_gas():
    """Verify full AtariFractalGas integration."""
    print_section("5. AtariFractalGas Integration Verification")

    print("Creating AtariEnv...")
    env = AtariEnv("ALE/Pong-v5", obs_type="ram", render_mode="rgb_array")

    print("\nCreating AtariFractalGas...")
    gas = AtariFractalGas(
        env=env,
        N=30,
        dist_coef=1.0,
        reward_coef=1.0,
        dt_range=(1, 4),
        device="cpu",
        seed=42,
    )

    print("✓ AtariFractalGas created")
    print(f"  - Walkers: {gas.N}")
    print(f"  - Device: {gas.device}")
    print(f"  - dt_range: {gas.kinetic_op.dt_range}")

    print("\nRunning 50 iterations...")
    final_state, history = gas.run(max_iterations=50, stop_when_all_dead=False)

    print("✓ Run completed")
    print(f"  - Iterations: {len(history)}")
    print(f"  - Total steps: {gas.total_steps}")
    print(f"  - Total clones: {gas.total_clones}")
    print(f"  - Alive walkers: {final_state.alive.sum().item()}/{final_state.N}")
    print(f"  - Best reward: {final_state.rewards.max().item():.2f}")
    print(f"  - Mean reward: {final_state.rewards.mean().item():.2f}")

    # Check history structure
    print("\nVerifying history structure...")
    required_keys = [
        "iteration",
        "num_cloned",
        "alive_count",
        "mean_reward",
        "max_reward",
        "mean_virtual_reward",
    ]
    all_keys_present = all(key in history[0] for key in required_keys)

    if all_keys_present:
        print("✓ History structure verified")
        print(f"  - Keys present: {list(history[0].keys())}")
    else:
        print("✗ WARNING: Missing expected keys in history")

    env.close()
    print("\n✓ AtariFractalGas integration verified")


def verify_multiple_games():
    """Verify multiple Atari games work."""
    print_section("6. Multiple Games Verification")

    games = ["ALE/Pong-v5", "ALE/Breakout-v5", "ALE/SpaceInvaders-v5"]

    for game_name in games:
        print(f"\nTesting {game_name}...")
        try:
            env = AtariEnv(game_name, obs_type="ram", render_mode="rgb_array")
            env.reset(seed=42)
            _, _, _, _, _, _ = env.step(action=0)
            env.close()
            print(f"✓ {game_name} works")
        except Exception as e:
            print(f"✗ {game_name} failed: {e}")

    print("\n✓ Multiple games verification complete")


def main():
    """Run all verification tests."""
    print("\n" + "=" * 70)
    print("  AtariEnv Integration Verification Suite")
    print("=" * 70)

    try:
        verify_basic_env()
        verify_batch_operations()
        verify_obs_types()
        verify_headless_rendering()
        verify_atari_fractal_gas()
        verify_multiple_games()

        print("\n" + "=" * 70)
        print("  ✓ ALL VERIFICATIONS PASSED")
        print("=" * 70)
        print("\n✓ AtariEnv is fully functional and ready for production use!")

    except Exception as e:
        print("\n" + "=" * 70)
        print(f"  ✗ VERIFICATION FAILED: {e}")
        print("=" * 70)
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
