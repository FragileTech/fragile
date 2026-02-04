#!/usr/bin/env python
"""Quick verification script for dashboard AtariEnv integration."""

import os
import sys

# Enable headless mode
os.environ["PYGLET_HEADLESS"] = "1"

def test_atari_env_creation():
    """Test that AtariEnv can be created with dashboard parameters."""
    print("Testing AtariEnv creation with dashboard parameters...")

    from fragile.fractalai.videogames.atari import AtariEnv

    # Test with different observation types
    for obs_type in ["ram", "rgb", "grayscale"]:
        print(f"\n  Testing obs_type='{obs_type}'...")
        try:
            env = AtariEnv(
                name="ALE/Pong-v5",
                obs_type=obs_type,
                render_mode="rgb_array",
                include_rgb=True,
            )

            # Verify required methods exist
            required_methods = [
                'reset', 'step', 'step_batch', 'get_state',
                'clone_state', 'restore_state', 'render', 'close', 'action_space'
            ]
            missing = [m for m in required_methods if not hasattr(env, m)]

            if missing:
                print(f"    ✗ Missing methods: {missing}")
                env.close()
                return False

            # Test basic functionality
            state = env.reset(seed=42)
            if not hasattr(state, 'copy'):
                print(f"    ✗ State doesn't have copy() method")
                env.close()
                return False

            if state.rgb_frame is None:
                print(f"    ✗ RGB frame is None (include_rgb=True should provide it)")
                env.close()
                return False

            env.close()
            print(f"    ✓ obs_type='{obs_type}' works correctly")

        except Exception as e:
            print(f"    ✗ Failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    return True


def test_atari_gas_integration():
    """Test that AtariEnv works with AtariFractalGas."""
    print("\nTesting AtariEnv integration with AtariFractalGas...")

    from fragile.fractalai.videogames.atari import AtariEnv
    from fragile.fractalai.videogames.atari_gas import AtariFractalGas

    try:
        # Create environment (as dashboard does)
        env = AtariEnv(
            name="ALE/Pong-v5",
            obs_type="ram",
            render_mode="rgb_array",
            include_rgb=True,
        )
        print("  ✓ Environment created")

        # Test with AtariFractalGas (as dashboard does)
        gas = AtariFractalGas(
            env=env,
            N=10,
            dist_coef=1.0,
            reward_coef=1.0,
            dt_range=(1, 4),
            device="cpu",
            seed=42,
            record_frames=True,  # Dashboard default
        )
        print("  ✓ AtariFractalGas initialized")

        # Run a few iterations
        state = gas.reset()
        print("  ✓ Gas reset successful")

        for i in range(5):
            state, info = gas.step(state)
            print(f"  ✓ Iteration {i}: reward={info['max_reward']:.2f}, clones={info['num_cloned']}")

            if "best_frame" in info and info["best_frame"] is not None:
                print(f"    Frame captured: {info['best_frame'].shape}")
            else:
                print(f"    ⚠ No frame captured (record_frames=True)")

        env.close()
        print("  ✓ Integration test completed successfully!")
        return True

    except Exception as e:
        print(f"  ✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_game_names():
    """Test that all dashboard game names work."""
    print("\nTesting all dashboard game names...")

    from fragile.fractalai.videogames.atari import AtariEnv

    games = [
        "ALE/Pong-v5",
        "ALE/Breakout-v5",
        "ALE/MsPacman-v5",
        "ALE/SpaceInvaders-v5",
    ]

    for game in games:
        try:
            env = AtariEnv(
                name=game,
                obs_type="ram",
                render_mode="rgb_array",
                include_rgb=False,  # Fast test
            )
            env.reset(seed=42)
            env.close()
            print(f"  ✓ {game}")
        except Exception as e:
            print(f"  ✗ {game}: {e}")
            return False

    return True


def main():
    """Run all verification tests."""
    print("=" * 70)
    print("Dashboard AtariEnv Integration Verification")
    print("=" * 70)

    results = []

    # Test 1: Environment creation
    results.append(("AtariEnv Creation", test_atari_env_creation()))

    # Test 2: Gas integration
    results.append(("AtariFractalGas Integration", test_atari_gas_integration()))

    # Test 3: Game names
    results.append(("Game Names", test_game_names()))

    # Summary
    print("\n" + "=" * 70)
    print("Test Summary:")
    print("=" * 70)

    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False

    print("=" * 70)

    if all_passed:
        print("\n✓ All tests passed! Dashboard integration is ready.")
        return 0
    else:
        print("\n✗ Some tests failed. Check output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
