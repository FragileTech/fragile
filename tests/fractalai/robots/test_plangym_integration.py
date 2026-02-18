"""Integration tests for plangym step_batch → kinetic.py → torch.tensor pipeline.

Validates that plangym's list-based step_batch returns are properly converted
to numpy arrays with correct dtypes before reaching torch.tensor() calls in
fractal_gas.py.
"""

import os

import numpy as np
import pytest
import torch

os.environ.setdefault("MUJOCO_GL", "osmesa")

import plangym

from fragile.fractalai.videogames.kinetic import RandomActionOperator

ENV_NAME = "CartPole-v1"


@pytest.fixture
def plangym_env():
    """Create a plangym environment for testing."""
    env = plangym.make(ENV_NAME)
    yield env
    if hasattr(env, "close"):
        env.close()


@pytest.fixture
def kinetic_op(plangym_env):
    """Create a RandomActionOperator wrapping a plangym env."""
    return RandomActionOperator(env=plangym_env, dt_range=(1, 2), seed=42)


def _get_initial_states(env, n: int):
    """Reset env and return n copies of the initial state."""
    obs, info = env.reset(return_state=False)
    state = env.get_state()
    return np.array([state for _ in range(n)], dtype=object)


class TestStepBatchDtypes:
    """Verify that kinetic_op.apply returns arrays castable to torch tensors."""

    def test_step_batch_returns_convertible_to_tensors(self, kinetic_op, plangym_env):
        """Observations, rewards, and dones from apply() must convert to tensors."""
        N = 4
        states = _get_initial_states(plangym_env, N)

        new_states, observations, rewards, dones, truncated, infos = kinetic_op.apply(states)

        # Verify types are numpy arrays, not raw lists
        assert isinstance(observations, np.ndarray), f"observations is {type(observations)}"
        assert isinstance(rewards, np.ndarray), f"rewards is {type(rewards)}"
        assert isinstance(dones, np.ndarray), f"dones is {type(dones)}"
        assert isinstance(truncated, np.ndarray), f"truncated is {type(truncated)}"

        # Verify dtypes are numeric, not object
        assert observations.dtype != np.dtype("O"), f"observations dtype is {observations.dtype}"
        assert rewards.dtype != np.dtype("O"), f"rewards dtype is {rewards.dtype}"

        # The critical test: these must not raise "Cannot cast dtype('O') to dtype('float64')"
        obs_tensor = torch.tensor(observations, dtype=torch.float32)
        reward_tensor = torch.tensor(rewards, dtype=torch.float32)
        done_tensor = torch.tensor(dones, dtype=torch.bool)
        trunc_tensor = torch.tensor(truncated, dtype=torch.bool)

        assert obs_tensor.shape[0] == N
        assert reward_tensor.shape == (N,)
        assert done_tensor.shape == (N,)
        assert trunc_tensor.shape == (N,)

    def test_states_roundtrip_through_object_array(self, kinetic_op, plangym_env):
        """States from step_batch survive np.array(states, dtype=object) and can be reused."""
        N = 3
        states = _get_initial_states(plangym_env, N)

        new_states, _, _, _, _, _ = kinetic_op.apply(states)

        # Convert to object array (as fractal_gas.py:411 does)
        new_states_arr = np.array(new_states, dtype=object)
        assert new_states_arr.shape == (N,)

        # Use those states for a second step_batch call
        new_states2, obs2, rewards2, dones2, truncated2, infos2 = kinetic_op.apply(new_states_arr)

        assert isinstance(obs2, np.ndarray)
        assert obs2.dtype != np.dtype("O")
        assert len(new_states2) == N

    def test_full_pipeline_reset_and_step(self, plangym_env):
        """End-to-end: reset → step_batch → torch conversion, mimicking fractal_gas.py."""
        N = 5
        env = plangym_env
        kinetic = RandomActionOperator(env=env, dt_range=(1, 3), seed=0)

        # Reset
        obs, info = env.reset(return_state=False)
        state = env.get_state()
        states = np.array([state for _ in range(N)], dtype=object)

        # Step (like fractal_gas.py step())
        new_states_list, obs_np, step_rewards_np, dones_np, truncated_np, infos = (
            kinetic.apply(states)
        )

        # Convert to object array for states (fractal_gas.py:411)
        new_states = np.array(new_states_list, dtype=object)

        # Convert to tensors (fractal_gas.py:414-417)
        device = torch.device("cpu")
        dtype = torch.float32
        observations = torch.tensor(obs_np, device=device, dtype=dtype)
        step_rewards = torch.tensor(step_rewards_np, device=device, dtype=dtype)
        dones = torch.tensor(dones_np, device=device, dtype=torch.bool)
        truncated = torch.tensor(truncated_np, device=device, dtype=torch.bool)

        # Cumulative reward update (fractal_gas.py:420)
        prev_rewards = torch.zeros(N, device=device, dtype=dtype)
        cumulative = prev_rewards + step_rewards

        assert observations.shape[0] == N
        assert cumulative.shape == (N,)
        assert dones.shape == (N,)
