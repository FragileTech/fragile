"""Tests for PlanningHistory dataclass."""

import numpy as np
import pytest

from fragile.fractalai.planning_gas import PlanningHistory, PlanningTrajectory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trajectory(n_steps: int = 5, with_frames: bool = False) -> PlanningTrajectory:
    """Build a minimal PlanningTrajectory for testing."""
    traj = PlanningTrajectory(
        frames=[] if with_frames else None,
    )
    traj.states.append("initial_state")
    traj.observations.append(np.zeros(4))

    cum = 0.0
    for i in range(n_steps):
        reward = float(i + 1)
        cum += reward
        traj.actions.append(i % 4)
        traj.rewards.append(reward)
        traj.cumulative_rewards.append(cum)
        traj.dones.append(i == n_steps - 1)
        traj.planning_infos.append({
            "alive_count": 20 - i,
            "inner_mean_reward": 0.5 + i * 0.1,
            "inner_max_reward": 1.0 + i * 0.2,
            "inner_iterations": 5,
        })
        traj.states.append(f"state_{i}")
        traj.observations.append(np.ones(4) * i)
        if with_frames:
            traj.frames.append(np.random.randint(0, 255, (210, 160, 3), dtype=np.uint8))

    return traj


def _make_planning_history(
    n_steps: int = 5, with_frames: bool = False, env_name: str = "TestEnv"
) -> PlanningHistory:
    """Build a PlanningHistory via from_trajectory."""
    traj = _make_trajectory(n_steps, with_frames=with_frames)
    return PlanningHistory.from_trajectory(traj, N=30, env_name=env_name)


# ---------------------------------------------------------------------------
# TestPlanningHistory
# ---------------------------------------------------------------------------


class TestPlanningHistory:
    def test_from_trajectory(self):
        traj = _make_trajectory(n_steps=5)
        ph = PlanningHistory.from_trajectory(traj, N=30, env_name="TestEnv")
        assert isinstance(ph, PlanningHistory)
        assert len(ph.iterations) == 5
        assert ph.N == 30
        assert ph.env_name == "TestEnv"
        assert ph.max_steps == 5

    def test_from_trajectory_empty(self):
        traj = PlanningTrajectory()
        ph = PlanningHistory.from_trajectory(traj, N=10, env_name="Empty")
        assert len(ph.iterations) == 0
        assert len(ph.step_rewards) == 0
        assert len(ph.cumulative_rewards) == 0
        assert len(ph.actions) == 0
        assert len(ph.dones) == 0
        assert ph.max_steps == 0

    def test_to_atari_history(self):
        ph = _make_planning_history(n_steps=5, env_name="ALE/Pong-v5")
        ah = ph.to_atari_history()
        assert len(ah.iterations) == 5
        assert ah.N == 30
        assert ah.game_name == "ALE/Pong-v5"
        assert ah.max_iterations == 5
        # rewards_mean and rewards_max should be cumulative
        assert ah.rewards_mean == ph.cumulative_rewards
        assert ah.rewards_max == ph.cumulative_rewards
        # rewards_min should be step rewards
        assert ah.rewards_min == ph.step_rewards
        # num_cloned should be 0 for planning
        assert ah.num_cloned == [0] * 5

    def test_to_robotic_history(self):
        ph = _make_planning_history(n_steps=5, env_name="cartpole-balance")
        rh = ph.to_robotic_history()
        assert len(rh.iterations) == 5
        assert rh.N == 30
        assert rh.task_name == "cartpole-balance"
        assert rh.max_iterations == 5
        assert rh.rewards_mean == ph.cumulative_rewards
        assert rh.rewards_max == ph.cumulative_rewards
        assert rh.rewards_min == ph.step_rewards
        assert rh.num_cloned == [0] * 5

    def test_step_rewards_preserved(self):
        ph = _make_planning_history(n_steps=5)
        # step_rewards should be [1.0, 2.0, 3.0, 4.0, 5.0]
        assert ph.step_rewards == [1.0, 2.0, 3.0, 4.0, 5.0]
        # cumulative should be different
        assert ph.cumulative_rewards == [1.0, 3.0, 6.0, 10.0, 15.0]
        assert ph.step_rewards != ph.cumulative_rewards

    def test_cumulative_rewards_monotonic(self):
        ph = _make_planning_history(n_steps=10)
        for i in range(1, len(ph.cumulative_rewards)):
            assert ph.cumulative_rewards[i] >= ph.cumulative_rewards[i - 1]

    def test_dones_preserved(self):
        ph = _make_planning_history(n_steps=5)
        assert isinstance(ph.dones, list)
        assert all(isinstance(d, bool) for d in ph.dones)
        # Last step should be done (from our helper)
        assert ph.dones[-1] is True
        # Others should be False
        assert all(d is False for d in ph.dones[:-1])

    def test_actions_preserved(self):
        ph = _make_planning_history(n_steps=5)
        assert len(ph.actions) == 5
        assert ph.actions == [0, 1, 2, 3, 0]

    def test_inner_planning_metrics(self):
        ph = _make_planning_history(n_steps=5)
        assert len(ph.inner_alive_counts) == 5
        assert len(ph.inner_mean_rewards) == 5
        assert len(ph.inner_max_rewards) == 5
        assert len(ph.inner_iterations) == 5
        # Check specific values
        assert ph.inner_alive_counts[0] == 20
        assert ph.inner_alive_counts[4] == 16
        assert ph.inner_iterations == [5] * 5

    def test_has_frames_with_frames(self):
        ph = _make_planning_history(n_steps=3, with_frames=True)
        assert ph.has_frames is True

    def test_has_frames_without_frames(self):
        ph = _make_planning_history(n_steps=3, with_frames=False)
        assert ph.has_frames is False

    def test_has_frames_empty(self):
        traj = PlanningTrajectory()
        ph = PlanningHistory.from_trajectory(traj, N=10, env_name="Empty")
        assert ph.has_frames is False

    def test_max_steps_matches(self):
        for n in [1, 5, 20]:
            ph = _make_planning_history(n_steps=n)
            assert ph.max_steps == n
            assert len(ph.iterations) == n

    def test_to_atari_history_virtual_rewards(self):
        ph = _make_planning_history(n_steps=3)
        ah = ph.to_atari_history()
        assert ah.virtual_rewards_mean == ph.inner_mean_rewards
        assert ah.virtual_rewards_max == ph.inner_max_rewards

    def test_to_robotic_history_virtual_rewards(self):
        ph = _make_planning_history(n_steps=3)
        rh = ph.to_robotic_history()
        assert rh.virtual_rewards_mean == ph.inner_mean_rewards
        assert rh.virtual_rewards_max == ph.inner_max_rewards

    def test_to_atari_history_alive_counts(self):
        ph = _make_planning_history(n_steps=3)
        ah = ph.to_atari_history()
        assert ah.alive_counts == ph.inner_alive_counts

    def test_to_atari_history_with_frames(self):
        ph = _make_planning_history(n_steps=3, with_frames=True)
        ah = ph.to_atari_history()
        assert ah.has_frames
        assert len(ah.best_frames) == 3

    def test_to_robotic_history_with_frames(self):
        ph = _make_planning_history(n_steps=3, with_frames=True)
        rh = ph.to_robotic_history()
        assert rh.has_frames
        assert len(rh.best_frames) == 3


# ---------------------------------------------------------------------------
# TestPlanningTrajectoryConvenience
# ---------------------------------------------------------------------------


class TestPlanningTrajectoryConvenience:
    def test_to_planning_history(self):
        traj = _make_trajectory(n_steps=5)
        ph = traj.to_planning_history(N=20, env_name="ConvenienceTest")
        assert isinstance(ph, PlanningHistory)
        assert ph.N == 20
        assert ph.env_name == "ConvenienceTest"
        assert ph.max_steps == 5
        assert ph.step_rewards == list(traj.rewards)

    def test_to_planning_history_empty(self):
        traj = PlanningTrajectory()
        ph = traj.to_planning_history(N=10, env_name="Empty")
        assert ph.max_steps == 0
        assert len(ph.actions) == 0
