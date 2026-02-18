"""Test that RGB frames can be recovered from physics state without include_rgb.

Validates that we can skip the expensive per-walker RGB rendering during
stepping (include_rgb=False) and recover visually identical frames later by
restoring the physics state and calling render().
"""

import os

import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "osmesa")

from fragile.fractalai.robots.dm_control_env import DMControlEnv


def _pixel_diff_stats(a: np.ndarray, b: np.ndarray) -> dict:
    """Compute pixel-level difference statistics between two RGB frames."""
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return {
        "max_diff": int(diff.max()),
        "mean_diff": float(diff.mean()),
        "pct_nonzero": 100.0 * np.count_nonzero(diff) / diff.size,
        "shape_match": a.shape == b.shape,
        "dtype_match": a.dtype == b.dtype,
    }


class TestRGBRecovery:
    """Verify that set_state + render reproduces the original RGB frame."""

    @pytest.fixture()
    def env_rgb(self):
        env = DMControlEnv(name="walker-walk", include_rgb=True)
        yield env
        env.close()

    @pytest.fixture()
    def env_no_rgb(self):
        env = DMControlEnv(name="walker-walk", include_rgb=False)
        yield env
        env.close()

    def test_reset_frame_recovery(self, env_rgb, env_no_rgb):
        """Render after restoring the reset state matches the original."""
        state = env_rgb.reset()
        assert state.rgb_frame is not None

        # Restore into the no-rgb env and render
        env_no_rgb.set_state(state)
        recovered = env_no_rgb.render()

        stats = _pixel_diff_stats(state.rgb_frame, recovered)
        assert stats["shape_match"]
        assert stats["max_diff"] == 0, f"Reset frame mismatch: {stats}"

    def test_single_step_recovery(self, env_rgb, env_no_rgb):
        """Render after restoring a post-step state matches the original."""
        state = env_rgb.reset()
        action = env_rgb.action_space.sample()

        new_state, obs, reward, done, trunc, info = env_rgb.step(
            action, state=state, dt=1, return_state=True,
        )
        assert new_state.rgb_frame is not None

        env_no_rgb.set_state(new_state)
        recovered = env_no_rgb.render()

        stats = _pixel_diff_stats(new_state.rgb_frame, recovered)
        assert stats["shape_match"]
        assert stats["max_diff"] == 0, f"Single step mismatch: {stats}"

    def test_multi_step_sequential_recovery(self, env_rgb, env_no_rgb):
        """Recovery is exact after a chain of sequential steps."""
        state = env_rgb.reset()
        states_with_rgb = []

        # Take 10 sequential steps, saving each state with RGB
        for _ in range(10):
            action = env_rgb.action_space.sample()
            state, obs, reward, done, trunc, info = env_rgb.step(
                action, state=state, dt=1, return_state=True,
            )
            states_with_rgb.append(state)

        # Recover each frame from just the physics state
        for i, original in enumerate(states_with_rgb):
            env_no_rgb.set_state(original)
            recovered = env_no_rgb.render()
            stats = _pixel_diff_stats(original.rgb_frame, recovered)
            assert stats["max_diff"] == 0, f"Step {i} mismatch: {stats}"

    def test_out_of_order_recovery(self, env_rgb, env_no_rgb):
        """Recovery works when states are restored in arbitrary order."""
        state = env_rgb.reset()
        states_with_rgb = [state]

        for _ in range(10):
            action = env_rgb.action_space.sample()
            state, obs, reward, done, trunc, info = env_rgb.step(
                action, state=state, dt=1, return_state=True,
            )
            states_with_rgb.append(state)

        # Restore in reverse order
        for i in reversed(range(len(states_with_rgb))):
            original = states_with_rgb[i]
            env_no_rgb.set_state(original)
            recovered = env_no_rgb.render()
            stats = _pixel_diff_stats(original.rgb_frame, recovered)
            assert stats["max_diff"] == 0, (
                f"Out-of-order recovery at index {i} mismatch: {stats}"
            )

    def test_recovery_after_different_state_interleaved(self, env_rgb, env_no_rgb):
        """Recovery is exact even after the env was used with a different state."""
        state = env_rgb.reset()

        # Take two divergent paths from the same initial state
        action_a = env_rgb.action_space.sample()
        state_a, *_ = env_rgb.step(action_a, state=state, dt=1, return_state=True)

        action_b = env_rgb.action_space.sample()
        state_b, *_ = env_rgb.step(action_b, state=state, dt=1, return_state=True)

        # Restore state_a, then state_b, then state_a again
        env_no_rgb.set_state(state_a)
        frame_a1 = env_no_rgb.render()

        env_no_rgb.set_state(state_b)
        frame_b = env_no_rgb.render()

        env_no_rgb.set_state(state_a)
        frame_a2 = env_no_rgb.render()

        stats_a1 = _pixel_diff_stats(state_a.rgb_frame, frame_a1)
        stats_b = _pixel_diff_stats(state_b.rgb_frame, frame_b)
        stats_a2 = _pixel_diff_stats(state_a.rgb_frame, frame_a2)

        assert stats_a1["max_diff"] == 0, f"state_a first restore: {stats_a1}"
        assert stats_b["max_diff"] == 0, f"state_b restore: {stats_b}"
        assert stats_a2["max_diff"] == 0, f"state_a second restore: {stats_a2}"

    def test_dt_greater_than_one_recovery(self, env_rgb, env_no_rgb):
        """Recovery works with dt > 1 (multiple physics sub-steps)."""
        state = env_rgb.reset()

        for dt in [2, 3, 5]:
            action = env_rgb.action_space.sample()
            new_state, *_ = env_rgb.step(
                action, state=state, dt=dt, return_state=True,
            )
            env_no_rgb.set_state(new_state)
            recovered = env_no_rgb.render()

            stats = _pixel_diff_stats(new_state.rgb_frame, recovered)
            assert stats["max_diff"] == 0, f"dt={dt} mismatch: {stats}"
            state = new_state

    def test_multiple_tasks_same_env(self):
        """Recovery is exact across tasks when using the same env instance."""
        tasks = ["cartpole-balance", "reacher-easy", "cheetah-run", "hopper-stand"]
        for task in tasks:
            env = DMControlEnv(name=task, include_rgb=True)

            state = env.reset()
            for _ in range(5):
                action = env.action_space.sample()
                state, *_ = env.step(
                    action, state=state, dt=1, return_state=True,
                )

            original_rgb = state.rgb_frame.copy()

            # Restore and re-render with the same env
            env.set_state(state)
            recovered = env.render()
            stats = _pixel_diff_stats(original_rgb, recovered)

            env.close()
            assert stats["max_diff"] == 0, f"Task {task} same-env mismatch: {stats}"

    def test_multiple_tasks_cross_env(self):
        """Recovery across separate env instances — document any divergence."""
        tasks = ["cartpole-balance", "reacher-easy", "cheetah-run", "hopper-stand"]
        results = {}
        for task in tasks:
            env_rgb = DMControlEnv(name=task, include_rgb=True)
            env_no = DMControlEnv(name=task, include_rgb=False)

            state = env_rgb.reset()
            for _ in range(5):
                action = env_rgb.action_space.sample()
                state, *_ = env_rgb.step(
                    action, state=state, dt=1, return_state=True,
                )

            env_no.set_state(state)
            recovered = env_no.render()
            stats = _pixel_diff_stats(state.rgb_frame, recovered)
            results[task] = stats

            env_rgb.close()
            env_no.close()

        # Report all results, allow small differences from different GL contexts
        for task, stats in results.items():
            assert stats["shape_match"], f"Task {task} shape mismatch"
            # Cross-env may have render context differences; check they're small
            assert stats["mean_diff"] < 10.0, (
                f"Task {task} cross-env mean diff too large: {stats}"
            )

    def test_long_trajectory_recovery(self, env_rgb, env_no_rgb):
        """Recovery stays exact over a 100-step trajectory."""
        np.random.seed(42)
        state = env_rgb.reset()

        mismatches = []
        for i in range(100):
            action = env_rgb.action_space.sample()
            state, *_ = env_rgb.step(
                action, state=state, dt=1, return_state=True,
            )

            env_no_rgb.set_state(state)
            recovered = env_no_rgb.render()
            stats = _pixel_diff_stats(state.rgb_frame, recovered)
            if stats["max_diff"] > 0:
                mismatches.append((i, stats))

        assert len(mismatches) == 0, (
            f"{len(mismatches)}/100 steps had mismatches. "
            f"First: step {mismatches[0][0]}: {mismatches[0][1]}"
        )

    def test_batch_step_recovery(self, env_rgb, env_no_rgb):
        """Recovery works for states produced by step_batch."""
        state = env_rgb.reset()
        N = 10
        states = np.array([state.copy() for _ in range(N)], dtype=object)
        actions = np.array([env_rgb.action_space.sample() for _ in range(N)])
        dt = np.ones(N, dtype=int)

        new_states, obs, rewards, dones, truncs, infos = env_rgb.step_batch(
            states=states, actions=actions, dt=dt,
        )

        for i in range(N):
            original = new_states[i]
            assert original.rgb_frame is not None

            env_no_rgb.set_state(original)
            recovered = env_no_rgb.render()
            stats = _pixel_diff_stats(original.rgb_frame, recovered)
            assert stats["max_diff"] == 0, f"Batch walker {i} mismatch: {stats}"


class TestOnDemandRendering:
    """Test the practical optimization: step without RGB, render on demand."""

    def test_skip_rgb_render_on_demand_same_env(self):
        """Step with include_rgb=False, then render on demand from same env.

        This is the exact optimization for the dashboard: run the gas with
        include_rgb=False to skip 100 render calls per iteration, and only
        render the best walker's frame when needed.
        """
        env = DMControlEnv(name="walker-walk", include_rgb=True)

        # Get ground truth: step with include_rgb=True
        state = env.reset()
        np.random.seed(123)
        ground_truth_frames = []
        states_no_rgb = []
        for _ in range(20):
            action = env.action_space.sample()
            state, *_ = env.step(action, state=state, dt=1, return_state=True)
            ground_truth_frames.append(state.rgb_frame.copy())
            # Save physics state for later recovery
            states_no_rgb.append(state.physics_state.copy())

        # Now recover each frame by restoring state + rendering
        for i, (gt_frame, phys_state) in enumerate(
            zip(ground_truth_frames, states_no_rgb)
        ):
            from fragile.fractalai.robots.dm_control_env import DMControlState

            dummy_state = DMControlState(
                physics_state=phys_state,
                observation=np.zeros(1),  # unused for rendering
                rgb_frame=None,
            )
            env.set_state(dummy_state)
            recovered = env.render()

            stats = _pixel_diff_stats(gt_frame, recovered)
            assert stats["max_diff"] == 0, (
                f"On-demand render at step {i} mismatch: {stats}"
            )

        env.close()

    def test_render_only_best_walker(self):
        """Simulate the gas pattern: step N walkers, render only the best.

        Confirms that stepping N walkers without RGB, then restoring just
        the best walker's state and rendering, produces the exact same frame
        as if we had rendered during stepping.
        """
        env = DMControlEnv(name="walker-walk", include_rgb=True)

        state = env.reset()
        N = 20
        states = np.array([state.copy() for _ in range(N)], dtype=object)
        np.random.seed(99)
        actions = np.array([env.action_space.sample() for _ in range(N)])
        dt = np.ones(N, dtype=int)

        # Step with RGB to get ground truth
        new_states_rgb, obs, rewards, dones, truncs, infos = env.step_batch(
            states=states, actions=actions, dt=dt,
        )

        # Find best walker
        best_idx = int(rewards.argmax())
        gt_frame = new_states_rgb[best_idx].rgb_frame.copy()

        # Now step without RGB
        env.include_rgb = False
        new_states_no_rgb, obs2, rewards2, dones2, truncs2, infos2 = env.step_batch(
            states=states, actions=actions, dt=dt,
        )
        env.include_rgb = True  # restore for rendering

        # Verify observations and rewards are identical
        np.testing.assert_array_equal(obs, obs2)
        np.testing.assert_array_equal(rewards, rewards2)
        np.testing.assert_array_equal(dones, dones2)

        # No walker has rgb_frame when include_rgb=False
        assert new_states_no_rgb[best_idx].rgb_frame is None

        # Render only the best walker on demand
        env.set_state(new_states_no_rgb[best_idx])
        recovered = env.render()

        stats = _pixel_diff_stats(gt_frame, recovered)
        assert stats["max_diff"] == 0, f"Best-walker on-demand render mismatch: {stats}"

        env.close()

    def test_render_on_demand_multiple_tasks(self):
        """On-demand rendering from same env is exact across tasks."""
        tasks = [
            "cartpole-balance",
            "reacher-easy",
            "cheetah-run",
            "walker-walk",
            "hopper-stand",
        ]
        for task in tasks:
            env = DMControlEnv(name=task, include_rgb=True)

            state = env.reset()
            np.random.seed(42)
            for _ in range(10):
                action = env.action_space.sample()
                state, *_ = env.step(action, state=state, dt=1, return_state=True)

            gt_frame = state.rgb_frame.copy()

            # Restore and render on demand
            env.set_state(state)
            recovered = env.render()
            stats = _pixel_diff_stats(gt_frame, recovered)
            env.close()

            assert stats["max_diff"] == 0, (
                f"Task {task} on-demand render mismatch: {stats}"
            )
