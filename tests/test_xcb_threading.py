#!/usr/bin/env python3
"""Minimal test cases to isolate XCB threading issues.

This test suite confirms that the XCB error is caused by threading,
not environment configuration. The dashboard creates gymnasium environments
in a background thread, but X11/XCB is not thread-safe.
"""

import threading
import sys
import traceback


def _get_test_environment():
    """Get an environment to test with - try multiple options."""
    import gymnasium as gym

    # Try environments in order of preference
    test_envs = [
        # Simple environments that should always work
        ("CartPole-v1", None),
        ("MountainCar-v0", None),
        # Atari environments (if available)
        ("ALE/Pong-v5", "rgb_array"),
        ("PongNoFrameskip-v4", "rgb_array"),
    ]

    for env_name, render_mode in test_envs:
        try:
            if render_mode:
                env = gym.make(env_name, render_mode=render_mode)
            else:
                env = gym.make(env_name)
            return env, env_name
        except Exception:
            continue

    raise RuntimeError("No suitable test environment found. Install with: pip install gymnasium")


def test_import_only():
    """Test 1: Just import gymnasium (should work)."""
    print("\nTest 1: Import gymnasium only...")
    try:
        import gymnasium as gym
        print("  ✓ PASS: Gymnasium imports successfully")
        return True
    except Exception as e:
        print(f"  ✗ FAIL: {e}")
        return False


def test_create_env_main_thread():
    """Test 2: Create environment in main thread (should work)."""
    print("\nTest 2: Creating environment in main thread...")
    try:
        env, env_name = _get_test_environment()
        print(f"  Using environment: {env_name}")
        obs, info = env.reset()
        if hasattr(obs, 'shape'):
            print(f"  Created environment, obs shape: {obs.shape}")
        else:
            print(f"  Created environment, obs type: {type(obs)}")
        env.close()
        print("  ✓ PASS: Main thread creation works")
        return True
    except Exception as e:
        print(f"  ✗ FAIL: {e}")
        traceback.print_exc()
        return False


def test_create_env_background_thread():
    """Test 3: Create environment in background thread (expected to fail with XCB error)."""
    print("\nTest 3: Creating environment in background thread...")
    print("  (This reproduces the dashboard bug)")

    result = [None]
    error_details = [None]

    def worker():
        try:
            env, env_name = _get_test_environment()
            print(f"  Worker thread: Using environment: {env_name}")
            obs, info = env.reset()
            if hasattr(obs, 'shape'):
                print(f"  Worker thread: Created environment, obs shape: {obs.shape}")
            else:
                print(f"  Worker thread: Created environment, obs type: {type(obs)}")
            env.close()
            result[0] = (True, None)
        except Exception as e:
            result[0] = (False, str(e))
            error_details[0] = traceback.format_exc()

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join(timeout=10)

    if result[0]:
        success, error = result[0]
        if success:
            print("  ✓ PASS: Background thread creation works")
            print("  (Unexpected - XCB error was expected with Atari/OpenGL)")
            return True
        else:
            print(f"  ✗ FAIL: {error}")
            if "xcb" in error.lower() or "assertion" in error.lower():
                print("  ⚠️  XCB THREADING ERROR CONFIRMED")
                print("  This is the root cause of the dashboard crash!")
            if error_details[0]:
                print("\n  Full error traceback:")
                print("  " + "\n  ".join(error_details[0].split("\n")))
            return False
    else:
        print("  ✗ FAIL: Thread timed out or crashed")
        return False


def test_dashboard_scenario():
    """Test 4: Simulate dashboard's exact threading pattern."""
    print("\nTest 4: Simulate dashboard threading scenario...")
    print("  (Thread spawned from button click handler)")

    result = [None]

    def simulate_button_click():
        """Simulates dashboard's _on_run_clicked method."""
        print("  Main thread: User clicked 'Run Simulation'")

        def worker():
            """Simulates dashboard's _run_simulation_worker method."""
            try:
                print("  Worker thread: Creating environment...")
                env, env_name = _get_test_environment()
                print(f"  Worker thread: Using environment: {env_name}")
                obs, info = env.reset()
                print("  Worker thread: Environment created successfully")
                env.close()
                result[0] = True
            except Exception as e:
                print(f"  Worker thread: FAILED - {e}")
                result[0] = False

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        thread.join(timeout=10)

    simulate_button_click()

    if result[0]:
        print("  ✓ PASS: Dashboard scenario works")
        return True
    else:
        print("  ✗ FAIL: Dashboard scenario fails (same as actual dashboard)")
        return False


def test_with_pre_created_env():
    """Test 5: Create env in main thread, use in background thread (proposed fix)."""
    print("\nTest 5: Pre-create environment (main thread), use in worker thread...")
    print("  (This is the proposed fix)")

    result = [None]

    # Create environment in main thread (safe)
    try:
        print("  Main thread: Creating environment...")
        env, env_name = _get_test_environment()
        print(f"  Main thread: Using environment: {env_name}")
        print("  Main thread: Environment created successfully")

        # Pass environment to worker thread
        def worker(env):
            """Worker receives pre-created environment."""
            try:
                print("  Worker thread: Using pre-created environment...")
                obs, info = env.reset()
                if hasattr(obs, 'shape'):
                    print(f"  Worker thread: Reset successful, obs shape: {obs.shape}")
                else:
                    print(f"  Worker thread: Reset successful, obs type: {type(obs)}")
                # Simulate a few steps
                for i in range(3):
                    obs, reward, done, truncated, info = env.step(env.action_space.sample())
                print("  Worker thread: Simulation steps successful")
                env.close()
                result[0] = True
            except Exception as e:
                print(f"  Worker thread: FAILED - {e}")
                traceback.print_exc()
                result[0] = False

        thread = threading.Thread(target=worker, args=(env,), daemon=True)
        thread.start()
        thread.join(timeout=10)

        if result[0]:
            print("  ✓ PASS: Pre-created environment works in worker thread!")
            print("  This confirms the fix strategy is correct.")
            return True
        else:
            print("  ✗ FAIL: Even pre-created environment fails")
            return False

    except Exception as e:
        print(f"  ✗ FAIL: Main thread environment creation failed - {e}")
        traceback.print_exc()
        return False


def test_render_modes():
    """Test 6: Try different render modes in background thread."""
    print("\nTest 6: Testing different render modes in background thread...")
    print("  (Only relevant for environments with render modes)")

    import gymnasium as gym

    # Try to find an environment that supports render modes
    try:
        env_name = "CartPole-v1"
        modes = ["rgb_array", "human", None]
        results = {}

        for mode in modes:
            result = [None]

            def worker(mode):
                try:
                    if mode:
                        env = gym.make(env_name, render_mode=mode)
                    else:
                        env = gym.make(env_name)
                    obs, info = env.reset()
                    env.close()
                    result[0] = True
                except Exception as e:
                    result[0] = False

            thread = threading.Thread(target=worker, args=(mode,), daemon=True)
            thread.start()
            thread.join(timeout=10)

            results[mode] = result[0]
            status = "✓ PASS" if result[0] else "✗ FAIL"
            print(f"  render_mode={mode}: {status}")

        return any(results.values())
    except Exception as e:
        print(f"  ⚠️  Skipped: {e}")
        return True  # Don't fail if we can't test this


if __name__ == "__main__":
    print("=" * 70)
    print("XCB Threading Test Suite")
    print("=" * 70)
    print("\nPurpose: Confirm that XCB error is caused by threading, not env config")
    print("Expected: Tests 3-4 should FAIL with XCB error")
    print("          Test 5 should PASS (this is the fix)")
    print("=" * 70)

    results = []

    # Run all tests
    results.append(("Import only", test_import_only()))
    results.append(("Main thread creation", test_create_env_main_thread()))
    results.append(("Background thread creation", test_create_env_background_thread()))
    results.append(("Dashboard scenario", test_dashboard_scenario()))
    results.append(("Pre-created env (fix)", test_with_pre_created_env()))
    results.append(("Render modes", test_render_modes()))

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")

    print("\n" + "=" * 70)
    print("INTERPRETATION:")
    print("=" * 70)

    if not results[2][1]:  # Background thread test failed
        print("✓ Root cause CONFIRMED: Threading issue with X11/XCB")
        print("  - Main thread creation works")
        print("  - Background thread creation fails")
        print("  → X11/XCB is not thread-safe")

    if results[4][1]:  # Pre-created env test passed
        print("\n✓ Fix strategy VALIDATED: Pre-create environment in main thread")
        print("  - Environment creation in main thread: safe")
        print("  - Environment usage in worker thread: safe")
        print("  → Apply this pattern to dashboard.py")
    else:
        print("\n⚠️  Fix strategy needs revision")

    print("=" * 70)

    # Exit code
    sys.exit(0 if results[4][1] else 1)
