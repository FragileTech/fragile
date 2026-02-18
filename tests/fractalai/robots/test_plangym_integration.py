"""Integration tests for plangym step_batch → kinetic.py → torch.tensor pipeline.

Validates that plangym's list-based step_batch returns are properly converted
to numpy arrays with correct dtypes before reaching torch.tensor() calls in
fractal_gas.py, and that states stored as 1D object arrays preserve their
original dtype when passed back to physics.set_state().
"""

import os

import numpy as np
import pytest
import torch


os.environ.setdefault("MUJOCO_GL", "osmesa")

import plangym

from fragile.fractalai.fractal_gas import _make_object_array
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
    """Reset env and return n copies of the initial state as a 1D object array."""
    _obs, _info = env.reset(return_state=False)
    state = env.get_state()
    return _make_object_array([state.copy() for _ in range(n)])


class TestMakeObjectArray:
    """Verify _make_object_array preserves element dtypes."""

    def test_same_shape_arrays_stay_1d(self):
        """Same-shape sub-arrays must produce a 1D object array, not 2D."""
        items = [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])]
        arr = _make_object_array(items)
        assert arr.shape == (2,), f"Expected (2,), got {arr.shape}"
        assert arr.dtype == object
        # Each element must be the original float64 array
        assert arr[0].dtype == np.float64
        assert arr[1].dtype == np.float64
        np.testing.assert_array_equal(arr[0], items[0])

    def test_contrast_with_np_array(self):
        """np.array(..., dtype=object) collapses same-shape arrays — our helper must not."""
        items = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
        # Standard numpy creates a 2D object array
        bad = np.array(items, dtype=object)
        assert bad.shape == (2, 2), "Sanity check: numpy collapses same-shape arrays"
        assert bad[0].dtype == object, "Elements lose their float64 dtype"

        # Our helper preserves 1D
        good = _make_object_array(items)
        assert good.shape == (2,)
        assert good[0].dtype == np.float64


class TestStepBatchDtypes:
    """Verify that kinetic_op.apply returns arrays castable to torch tensors."""

    def test_step_batch_returns_convertible_to_tensors(self, kinetic_op, plangym_env):
        """Observations, rewards, and dones from apply() must convert to tensors."""
        N = 4
        states = _get_initial_states(plangym_env, N)

        _new_states, observations, rewards, dones, truncated, _infos = kinetic_op.apply(states)

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
        """States from step_batch survive _make_object_array and can be reused."""
        N = 3
        states = _get_initial_states(plangym_env, N)

        new_states, _, _, _, _, _ = kinetic_op.apply(states)

        # Convert to 1D object array (as fractal_gas.py does)
        new_states_arr = _make_object_array(new_states)
        assert new_states_arr.shape == (N,)
        # Each element must preserve its original dtype
        assert new_states_arr[0].dtype != object

        # Use those states for a second step_batch call
        new_states2, obs2, _rewards2, _dones2, _truncated2, _infos2 = kinetic_op.apply(
            new_states_arr
        )

        assert isinstance(obs2, np.ndarray)
        assert obs2.dtype != np.dtype("O")
        assert len(new_states2) == N

    def test_full_pipeline_reset_and_step(self, plangym_env):
        """End-to-end: reset → step_batch → torch conversion, mimicking fractal_gas.py."""
        N = 5
        env = plangym_env
        kinetic = RandomActionOperator(env=env, dt_range=(1, 3), seed=0)

        # Reset
        _obs, _info = env.reset(return_state=False)
        state = env.get_state()
        states = _make_object_array([state.copy() for _ in range(N)])

        # Step (like fractal_gas.py step())
        new_states_list, obs_np, step_rewards_np, dones_np, truncated_np, _infos = kinetic.apply(
            states
        )

        # Convert to 1D object array for states
        _make_object_array(new_states_list)

        # Convert to tensors (fractal_gas.py:414-417)
        device = torch.device("cpu")
        dtype = torch.float32
        observations = torch.tensor(obs_np, device=device, dtype=dtype)
        step_rewards = torch.tensor(step_rewards_np, device=device, dtype=dtype)
        dones = torch.tensor(dones_np, device=device, dtype=torch.bool)
        torch.tensor(truncated_np, device=device, dtype=torch.bool)

        # Cumulative reward update (fractal_gas.py:420)
        prev_rewards = torch.zeros(N, device=device, dtype=dtype)
        cumulative = prev_rewards + step_rewards

        assert observations.shape[0] == N
        assert cumulative.shape == (N,)
        assert dones.shape == (N,)

    def test_robotic_gas_reset_and_multi_step(self):
        """RoboticFractalGas.reset() + multiple step() calls with plangym env."""
        from fragile.fractalai.robots.robotic_gas import RoboticFractalGas

        env = plangym.make("cartpole-balance")
        try:
            gas = RoboticFractalGas(
                env=env,
                N=4,
                dist_coef=1.0,
                reward_coef=1.0,
                use_cumulative_reward=True,
                dt_range=(1, 1),
                seed=42,
            )
            state = gas.reset()
            assert state.states.shape == (4,)
            assert state.states.dtype == object
            assert state.states[0].dtype == np.float64

            # Run 3 iterations — exercises clone + step_batch + tensor conversion
            for _ in range(3):
                state, _info = gas.step(state)
                assert state.observations.dtype == torch.float32
                assert state.states.shape == (4,)
                assert state.states[0].dtype == np.float64
        finally:
            env.close()
