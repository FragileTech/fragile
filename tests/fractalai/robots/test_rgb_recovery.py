"""Test that RGB frames can be recovered from physics state using plangym.

Validates that we can restore a physics state via set_state() and re-render
to get visually identical frames, enabling the dashboard optimization of
skipping per-walker rendering during stepping and only rendering on demand.

Note: MuJoCo's GL rendering may produce tiny per-pixel differences (max 1-2
out of 255) after state restoration.  Tests use a tolerance of max_diff <= 2.
"""

import os

import numpy as np
import pytest


os.environ.setdefault("MUJOCO_GL", "osmesa")

import plangym


# MuJoCo GL rendering may produce tiny per-pixel differences after
# state restoration.  Allow up to this per-channel difference.
_MAX_PIXEL_TOLERANCE = 2


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


def _render_frame(env) -> np.ndarray:
    """Render a frame from the environment's current physics state."""
    if hasattr(env, "physics"):
        return env.physics.render(height=480, width=480, camera_id=0)
    return env.get_image()


class TestRGBRecovery:
    """Verify that set_state + render reproduces the original RGB frame."""

    @pytest.fixture()
    def env(self):
        env = plangym.make(name="walker-walk")
        yield env
        env.close()

    def test_reset_frame_recovery(self, env):
        """Render after restoring the reset state matches the original."""
        state, _obs, _info = env.reset(return_state=True)
        original_frame = _render_frame(env)

        # Step away, then restore and re-render
        action = env.action_space.sample()
        env.step(action)
        env.set_state(state)
        recovered = _render_frame(env)

        stats = _pixel_diff_stats(original_frame, recovered)
        assert stats["shape_match"]
        assert stats["max_diff"] <= _MAX_PIXEL_TOLERANCE, f"Reset frame mismatch: {stats}"

    def test_single_step_recovery(self, env):
        """Render after restoring a post-step state matches the original."""
        state, _obs, _info = env.reset(return_state=True)
        action = env.action_space.sample()

        new_state, _obs, _reward, _done, _trunc, _step_info = env.step(
            action,
            state=state,
            dt=1,
            return_state=True,
        )
        original_frame = _render_frame(env)

        # Restore and re-render
        env.set_state(new_state)
        recovered = _render_frame(env)

        stats = _pixel_diff_stats(original_frame, recovered)
        assert stats["shape_match"]
        assert stats["max_diff"] <= _MAX_PIXEL_TOLERANCE, f"Single step mismatch: {stats}"

    def test_multi_step_sequential_recovery(self, env):
        """Recovery is near-exact after a chain of sequential steps."""
        state, _obs, _info = env.reset(return_state=True)
        states_and_frames = []

        # Take 10 sequential steps, saving state + rendered frame
        for _ in range(10):
            action = env.action_space.sample()
            state, _obs, _reward, _done, _trunc, _step_info = env.step(
                action,
                state=state,
                dt=1,
                return_state=True,
            )
            frame = _render_frame(env)
            states_and_frames.append((state.copy(), frame.copy()))

        # Recover each frame from just the physics state
        for i, (saved_state, original_frame) in enumerate(states_and_frames):
            env.set_state(saved_state)
            recovered = _render_frame(env)
            stats = _pixel_diff_stats(original_frame, recovered)
            assert stats["max_diff"] <= _MAX_PIXEL_TOLERANCE, f"Step {i} mismatch: {stats}"

    def test_out_of_order_recovery(self, env):
        """Recovery works when states are restored in arbitrary order."""
        state, _obs, _info = env.reset(return_state=True)
        initial_frame = _render_frame(env)
        states_and_frames = [(state.copy(), initial_frame.copy())]

        for _ in range(10):
            action = env.action_space.sample()
            state, _obs, _reward, _done, _trunc, _step_info = env.step(
                action,
                state=state,
                dt=1,
                return_state=True,
            )
            frame = _render_frame(env)
            states_and_frames.append((state.copy(), frame.copy()))

        # Restore in reverse order
        for i in reversed(range(len(states_and_frames))):
            saved_state, original_frame = states_and_frames[i]
            env.set_state(saved_state)
            recovered = _render_frame(env)
            stats = _pixel_diff_stats(original_frame, recovered)
            assert (
                stats["max_diff"] <= _MAX_PIXEL_TOLERANCE
            ), f"Out-of-order recovery at index {i} mismatch: {stats}"

    def test_recovery_after_different_state_interleaved(self, env):
        """Recovery is near-exact even after the env was used with a different state."""
        state, _obs, _info = env.reset(return_state=True)

        # Take two divergent paths from the same initial state
        action_a = env.action_space.sample()
        state_a, *_ = env.step(action_a, state=state, dt=1, return_state=True)
        frame_a_gt = _render_frame(env)

        action_b = env.action_space.sample()
        state_b, *_ = env.step(action_b, state=state, dt=1, return_state=True)
        frame_b_gt = _render_frame(env)

        # Restore state_a, then state_b, then state_a again
        env.set_state(state_a)
        frame_a1 = _render_frame(env)

        env.set_state(state_b)
        frame_b = _render_frame(env)

        env.set_state(state_a)
        frame_a2 = _render_frame(env)

        stats_a1 = _pixel_diff_stats(frame_a_gt, frame_a1)
        stats_b = _pixel_diff_stats(frame_b_gt, frame_b)
        stats_a2 = _pixel_diff_stats(frame_a_gt, frame_a2)

        assert stats_a1["max_diff"] <= _MAX_PIXEL_TOLERANCE, f"state_a first restore: {stats_a1}"
        assert stats_b["max_diff"] <= _MAX_PIXEL_TOLERANCE, f"state_b restore: {stats_b}"
        assert stats_a2["max_diff"] <= _MAX_PIXEL_TOLERANCE, f"state_a second restore: {stats_a2}"

    def test_dt_greater_than_one_recovery(self, env):
        """Recovery works with dt > 1 (multiple physics sub-steps)."""
        state, _obs, _info = env.reset(return_state=True)

        for dt in [2, 3, 5]:
            action = env.action_space.sample()
            new_state, *_ = env.step(
                action,
                state=state,
                dt=dt,
                return_state=True,
            )
            gt_frame = _render_frame(env)

            env.set_state(new_state)
            recovered = _render_frame(env)

            stats = _pixel_diff_stats(gt_frame, recovered)
            assert stats["max_diff"] <= _MAX_PIXEL_TOLERANCE, f"dt={dt} mismatch: {stats}"
            state = new_state

    def test_multiple_tasks(self):
        """Recovery is near-exact across different dm_control tasks."""
        tasks = ["cartpole-balance", "reacher-easy", "cheetah-run", "hopper-stand"]
        for task in tasks:
            env = plangym.make(name=task)

            state, _obs, _info = env.reset(return_state=True)
            for _ in range(5):
                action = env.action_space.sample()
                state, *_ = env.step(
                    action,
                    state=state,
                    dt=1,
                    return_state=True,
                )

            gt_frame = _render_frame(env)

            # Restore and re-render
            env.set_state(state)
            recovered = _render_frame(env)
            stats = _pixel_diff_stats(gt_frame, recovered)

            env.close()
            assert stats["max_diff"] <= _MAX_PIXEL_TOLERANCE, f"Task {task} mismatch: {stats}"

    def test_long_trajectory_recovery(self, env):
        """Recovery stays near-exact over a 100-step trajectory."""
        np.random.seed(42)
        state, _obs, _info = env.reset(return_state=True)

        mismatches = []
        for i in range(100):
            action = env.action_space.sample()
            state, *_ = env.step(
                action,
                state=state,
                dt=1,
                return_state=True,
            )
            gt_frame = _render_frame(env)

            env.set_state(state)
            recovered = _render_frame(env)
            stats = _pixel_diff_stats(gt_frame, recovered)
            if stats["max_diff"] > _MAX_PIXEL_TOLERANCE:
                mismatches.append((i, stats))

        assert len(mismatches) == 0, (
            f"{len(mismatches)}/100 steps had mismatches. "
            f"First: step {mismatches[0][0]}: {mismatches[0][1]}"
        )

    def test_batch_step_recovery(self, env):
        """Recovery works for states produced by step_batch."""
        state, _obs, _info = env.reset(return_state=True)
        N = 10
        states = np.array([state.copy() for _ in range(N)])
        actions = np.array([env.action_space.sample() for _ in range(N)])
        dt = np.ones(N, dtype=int)

        new_states, _obs_list, _rewards, _dones, _truncs, _infos = env.step_batch(
            actions=actions,
            states=states,
            dt=dt,
            return_state=True,
        )

        for i in range(N):
            env.set_state(new_states[i])
            recovered = _render_frame(env)
            assert isinstance(recovered, np.ndarray), f"Walker {i}: render returned non-array"
            assert recovered.shape[2] == 3, f"Walker {i}: unexpected shape {recovered.shape}"


class TestOnDemandRendering:
    """Test the practical optimization: step without state, render on demand."""

    def test_render_on_demand_same_env(self):
        """Step and capture ground truth, then restore + render matches."""
        env = plangym.make(name="walker-walk")

        state, _obs, _info = env.reset(return_state=True)
        np.random.seed(123)
        ground_truth_frames = []
        saved_states = []
        for _ in range(20):
            action = env.action_space.sample()
            state, *_ = env.step(action, state=state, dt=1, return_state=True)
            ground_truth_frames.append(_render_frame(env).copy())
            saved_states.append(state.copy())

        # Now recover each frame by restoring state + rendering
        for i, (gt_frame, saved_state) in enumerate(zip(ground_truth_frames, saved_states)):
            env.set_state(saved_state)
            recovered = _render_frame(env)

            stats = _pixel_diff_stats(gt_frame, recovered)
            assert (
                stats["max_diff"] <= _MAX_PIXEL_TOLERANCE
            ), f"On-demand render at step {i} mismatch: {stats}"

        env.close()

    def test_render_only_best_walker(self):
        """Simulate the gas pattern: step N walkers, render only the best.

        Confirms that stepping N walkers, then restoring just the best
        walker's state and rendering, produces a valid frame.
        """
        env = plangym.make(name="walker-walk")

        state, _obs, _info = env.reset(return_state=True)
        N = 20
        states = np.array([state.copy() for _ in range(N)])
        np.random.seed(99)
        actions = np.array([env.action_space.sample() for _ in range(N)])
        dt = np.ones(N, dtype=int)

        new_states, _obs_list, rewards, _dones, _truncs, _infos = env.step_batch(
            actions=actions,
            states=states,
            dt=dt,
            return_state=True,
        )

        # Find best walker and render on demand
        best_idx = int(np.array(rewards).argmax())
        env.set_state(new_states[best_idx])
        recovered = _render_frame(env)

        assert isinstance(recovered, np.ndarray)
        assert recovered.ndim == 3 and recovered.shape[2] == 3

        env.close()

    def test_render_on_demand_multiple_tasks(self):
        """On-demand rendering from same env is near-exact across tasks."""
        tasks = [
            "cartpole-balance",
            "reacher-easy",
            "cheetah-run",
            "walker-walk",
            "hopper-stand",
        ]
        for task in tasks:
            env = plangym.make(name=task)

            state, _obs, _info = env.reset(return_state=True)
            np.random.seed(42)
            for _ in range(10):
                action = env.action_space.sample()
                state, *_ = env.step(action, state=state, dt=1, return_state=True)

            gt_frame = _render_frame(env)

            # Restore and render on demand
            env.set_state(state)
            recovered = _render_frame(env)
            stats = _pixel_diff_stats(gt_frame, recovered)
            env.close()

            assert (
                stats["max_diff"] <= _MAX_PIXEL_TOLERANCE
            ), f"Task {task} on-demand render mismatch: {stats}"
