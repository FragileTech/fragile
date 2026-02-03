#!/usr/bin/env python3
"""
Command-line interface for running Atari Fractal Gas simulations.

This script bypasses the dashboard GUI to avoid XCB threading issues on WSL.
It runs simulations directly from the command line and saves results to disk.

Usage:
    python scripts/run_atari_gas_cli.py --game Pong --N 10 --iterations 100
"""

import argparse
import json
import time
from pathlib import Path
import sys
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def create_environment(game_name, obs_type="rgb", use_gymnasium=True):
    """Create Atari environment (main thread safe).

    Args:
        game_name: Game name (e.g., "Pong", "Breakout")
        obs_type: Observation type ("rgb" or "ram")
        use_gymnasium: Try gymnasium first, fall back to plangym

    Returns:
        Environment instance
    """
    print(f"Creating {game_name} environment...")

    if use_gymnasium:
        try:
            import gymnasium as gym

            # Try multiple name formats for gymnasium
            # Different versions of gymnasium use different naming conventions
            name_formats = []

            if not game_name.endswith("-v5") and not game_name.endswith("-v4"):
                # Try different formats
                name_formats = [
                    f"{game_name}NoFrameskip-v4",  # Classic format
                    f"ALE/{game_name}-v5",          # New ALE namespace
                    f"{game_name}-v4",              # Short format
                    f"{game_name}-ramNoFrameskip-v4" if obs_type == "ram" else f"{game_name}NoFrameskip-v4",
                ]
            else:
                name_formats = [game_name]

            last_error = None
            for env_name in name_formats:
                try:
                    print(f"  Trying gymnasium: {env_name}")
                    render_mode = "rgb_array" if obs_type == "rgb" else None
                    base_env = gym.make(env_name, render_mode=render_mode)
                    print(f"  ✓ Successfully created: {env_name}")
                    break
                except Exception as e:
                    last_error = e
                    continue
            else:
                # None of the formats worked
                raise last_error if last_error else Exception("No valid environment name found")

            # Wrap to match plangym interface
            class GymEnvWrapper:
                def __init__(self, env):
                    self.env = env
                    self.obs_type = "rgb"
                    self.action_space = env.action_space

                def reset(self):
                    obs, info = self.env.reset()
                    return obs

                def step(self, action):
                    obs, reward, terminated, truncated, info = self.env.step(action)
                    return obs, reward, terminated or truncated, info

                def render(self):
                    return self.env.render()

                def close(self):
                    self.env.close()

            env = GymEnvWrapper(base_env)
            print(f"  ✓ Using gymnasium")
            return env

        except Exception as e:
            print(f"  Gymnasium failed: {e}")
            print(f"  Falling back to plangym...")

    # Fall back to plangym
    try:
        from plangym import AtariEnvironment

        # Map game names to plangym format
        if game_name.endswith("-v5") or game_name.endswith("-v4"):
            game_name = game_name.split("-")[0].split("/")[-1]

        env = AtariEnvironment(
            name=game_name,
            obs_type=obs_type,
        )
        print(f"  ✓ Using plangym")
        return env

    except ImportError as e:
        print(f"  ✗ Failed to create environment: {e}")
        print("\nInstall dependencies:")
        print("  pip install gymnasium[atari] gymnasium[accept-rom-license]")
        print("  OR")
        print("  pip install plangym")
        sys.exit(1)


def run_simulation(
    game_name,
    N=10,
    max_iterations=100,
    dist_coef=1.0,
    reward_coef=1.0,
    use_cumulative_reward=True,
    dt_range=(0.5, 5.0),
    device="cpu",
    seed=None,
    record_frames=False,
    obs_type="rgb",
    output_dir=None,
):
    """Run Atari Fractal Gas simulation.

    Args:
        game_name: Atari game name (e.g., "Pong")
        N: Number of walkers
        max_iterations: Maximum number of iterations
        dist_coef: Distance coefficient for virtual reward
        reward_coef: Game reward coefficient
        use_cumulative_reward: Use cumulative rewards
        dt_range: Time step range (min, max)
        device: PyTorch device ("cpu" or "cuda")
        seed: Random seed
        record_frames: Whether to record frames
        obs_type: Observation type ("rgb" or "ram")
        output_dir: Directory to save results

    Returns:
        dict: Simulation results
    """
    from fragile.fractalai.videogames.atari_gas import AtariFractalGas

    print("\n" + "="*70)
    print("ATARI FRACTAL GAS - COMMAND LINE")
    print("="*70)
    print(f"\nGame: {game_name}")
    print(f"Walkers (N): {N}")
    print(f"Max iterations: {max_iterations}")
    print(f"Distance coef: {dist_coef}")
    print(f"Reward coef: {reward_coef}")
    print(f"Cumulative reward: {use_cumulative_reward}")
    print(f"dt range: {dt_range}")
    print(f"Device: {device}")
    print(f"Seed: {seed}")
    print(f"Record frames: {record_frames}")
    print(f"Observation type: {obs_type}")
    print()

    # Create environment in main thread (safe for X11/OpenGL)
    env = create_environment(game_name, obs_type=obs_type)

    # Create gas algorithm
    print("Initializing Fractal Gas algorithm...")
    gas = AtariFractalGas(
        env=env,
        N=N,
        dist_coef=dist_coef,
        reward_coef=reward_coef,
        use_cumulative_reward=use_cumulative_reward,
        dt_range=dt_range,
        device=device,
        seed=seed,
        record_frames=record_frames,
    )

    # Run simulation with progress updates
    print("\nRunning simulation...")
    print("-"*70)

    start_time = time.time()
    results = {
        "game": game_name,
        "N": N,
        "max_iterations": max_iterations,
        "iterations_completed": 0,
        "best_reward": float("-inf"),
        "total_reward": 0.0,
        "episode_rewards": [],
        "episode_lengths": [],
        "timestamps": [],
    }

    try:
        for i in range(max_iterations):
            # Step the gas algorithm
            step_result = gas.step()

            # Update results
            results["iterations_completed"] = i + 1

            if step_result.get("done", False):
                episode_reward = step_result.get("reward", 0.0)
                results["episode_rewards"].append(float(episode_reward))
                results["episode_lengths"].append(i + 1)
                results["total_reward"] += episode_reward

                if episode_reward > results["best_reward"]:
                    results["best_reward"] = float(episode_reward)

            # Progress update every 10 iterations
            if (i + 1) % 10 == 0 or i == 0:
                elapsed = time.time() - start_time
                iter_per_sec = (i + 1) / elapsed if elapsed > 0 else 0

                print(f"Iteration {i+1:4d}/{max_iterations} | "
                      f"Episodes: {len(results['episode_rewards']):3d} | "
                      f"Best reward: {results['best_reward']:7.1f} | "
                      f"Avg reward: {np.mean(results['episode_rewards']) if results['episode_rewards'] else 0:7.1f} | "
                      f"Speed: {iter_per_sec:.1f} it/s")

            results["timestamps"].append(time.time() - start_time)

    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user")

    finally:
        # Clean up
        print("-"*70)
        elapsed = time.time() - start_time

        print("\nSimulation complete!")
        print(f"Time elapsed: {elapsed:.1f}s")
        print(f"Iterations: {results['iterations_completed']}/{max_iterations}")
        print(f"Episodes completed: {len(results['episode_rewards'])}")

        if results["episode_rewards"]:
            print(f"Best reward: {results['best_reward']:.1f}")
            print(f"Average reward: {np.mean(results['episode_rewards']):.1f}")
            print(f"Std reward: {np.std(results['episode_rewards']):.1f}")

        # Save results
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Save results JSON
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            results_file = output_path / f"{game_name}_{timestamp}_results.json"

            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)

            print(f"\nResults saved to: {results_file}")

            # Save frames if recorded
            if record_frames and hasattr(gas, "recorded_frames") and gas.recorded_frames:
                frames_file = output_path / f"{game_name}_{timestamp}_frames.npy"
                np.save(frames_file, np.array(gas.recorded_frames))
                print(f"Frames saved to: {frames_file}")

        # Close environment
        env.close()

    print("="*70)
    return results


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Run Atari Fractal Gas simulations from command line",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run Pong with 10 walkers for 100 iterations
  python scripts/run_atari_gas_cli.py --game Pong --N 10 --iterations 100

  # Run Breakout with custom parameters
  python scripts/run_atari_gas_cli.py --game Breakout --N 20 --iterations 500 \\
      --dist-coef 1.5 --reward-coef 2.0 --seed 42

  # Record frames and save to output directory
  python scripts/run_atari_gas_cli.py --game Pong --N 10 --iterations 100 \\
      --record-frames --output-dir results/pong

  # Use RAM observations instead of RGB
  python scripts/run_atari_gas_cli.py --game Pong --obs-type ram --N 10

For WSL, make sure to run with xvfb:
  xvfb-run -a python scripts/run_atari_gas_cli.py --game Pong --N 10
        """
    )

    # Game configuration
    parser.add_argument("--game", type=str, default="Pong",
                        help="Atari game name (default: Pong)")
    parser.add_argument("--obs-type", type=str, default="rgb", choices=["rgb", "ram"],
                        help="Observation type (default: rgb)")

    # Algorithm parameters
    parser.add_argument("--N", type=int, default=10,
                        help="Number of walkers (default: 10)")
    parser.add_argument("--iterations", type=int, default=100,
                        help="Maximum iterations (default: 100)")
    parser.add_argument("--dist-coef", type=float, default=1.0,
                        help="Distance coefficient (default: 1.0)")
    parser.add_argument("--reward-coef", type=float, default=1.0,
                        help="Reward coefficient (default: 1.0)")
    parser.add_argument("--no-cumulative-reward", action="store_true",
                        help="Disable cumulative reward (default: enabled)")
    parser.add_argument("--dt-min", type=float, default=0.5,
                        help="Minimum time step (default: 0.5)")
    parser.add_argument("--dt-max", type=float, default=5.0,
                        help="Maximum time step (default: 5.0)")

    # System configuration
    parser.add_argument("--device", type=str, default="cpu",
                        help="PyTorch device (default: cpu)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (default: None)")

    # Output configuration
    parser.add_argument("--record-frames", action="store_true",
                        help="Record frames during simulation")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save results (default: None)")

    args = parser.parse_args()

    # Run simulation
    try:
        results = run_simulation(
            game_name=args.game,
            N=args.N,
            max_iterations=args.iterations,
            dist_coef=args.dist_coef,
            reward_coef=args.reward_coef,
            use_cumulative_reward=not args.no_cumulative_reward,
            dt_range=(args.dt_min, args.dt_max),
            device=args.device,
            seed=args.seed,
            record_frames=args.record_frames,
            obs_type=args.obs_type,
            output_dir=args.output_dir,
        )

        return 0

    except Exception as e:
        print(f"\n✗ Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
