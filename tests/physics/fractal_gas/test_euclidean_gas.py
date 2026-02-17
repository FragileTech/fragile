"""Comprehensive tests for fragile.physics.fractal_gas.euclidean_gas module.

Tests cover:
- random_pairing_fisher_yates: pairing properties, edge cases
- SwarmState: construction, properties, clone, copy_from
- EuclideanGas: initialization, step, run
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from fragile.physics.fractal_gas.euclidean_gas import (
    EuclideanGas,
    random_pairing_fisher_yates,
    SwarmState,
)


# =========================================================================
# TestRandomPairingFisherYates
# =========================================================================


class TestRandomPairingFisherYates:
    """Tests for the random_pairing_fisher_yates function."""

    @pytest.mark.parametrize("n", [2, 4, 10, 20, 50, 100])
    def test_output_shape(self, n: int):
        """Output tensor has shape [N]."""
        c = random_pairing_fisher_yates(n)
        assert c.shape == (n,)
        assert c.dtype == torch.long

    @pytest.mark.parametrize("n", [2, 4, 10, 20, 50])
    def test_mutual_pairing(self, n: int):
        """Companion map is involutory: c(c(i)) == i for all i."""
        torch.manual_seed(123)
        c = random_pairing_fisher_yates(n)
        assert torch.all(c[c] == torch.arange(n)), "c(c(i)) must equal i for all i"

    @pytest.mark.parametrize("n", [2, 6, 10, 50])
    def test_valid_range(self, n: int):
        """All indices must be in [0, N)."""
        c = random_pairing_fisher_yates(n)
        assert (c >= 0).all()
        assert (c < n).all()

    @pytest.mark.parametrize("n", [3, 5, 7, 11])
    def test_odd_n_last_unpaired(self, n: int):
        """For odd N, exactly one walker maps to itself (the leftover)."""
        torch.manual_seed(99)
        c = random_pairing_fisher_yates(n)
        self_maps = (c == torch.arange(n)).sum().item()
        assert self_maps == 1, f"Odd N={n} should have exactly 1 self-map, got {self_maps}"

    @pytest.mark.parametrize("n", [2, 4, 6, 10, 20])
    def test_even_n_no_self_maps(self, n: int):
        """For even N, no walker should map to itself (almost surely)."""
        # Run multiple seeds to reduce flakiness; if ANY seed produces
        # no self-maps, we accept (probability of all-self-map is vanishing).
        torch.manual_seed(42)
        c = random_pairing_fisher_yates(n)
        self_maps = (c == torch.arange(n)).sum().item()
        assert self_maps == 0, f"Even N={n} should have 0 self-maps, got {self_maps}"

    def test_n_equals_1(self):
        """N=1: single walker maps to itself."""
        c = random_pairing_fisher_yates(1)
        assert c.shape == (1,)
        assert c[0].item() == 0

    def test_n_equals_0(self):
        """N=0: returns empty tensor."""
        c = random_pairing_fisher_yates(0)
        assert c.shape == (0,)
        assert c.dtype == torch.long


# =========================================================================
# TestSwarmState
# =========================================================================


class TestSwarmState:
    """Tests for the SwarmState class."""

    def test_init_stores_tensors(self, positions: Tensor, velocities: Tensor):
        """Constructor stores x and v correctly."""
        state = SwarmState(positions, velocities)
        assert torch.equal(state.x, positions)
        assert torch.equal(state.v, velocities)

    def test_property_N(self, N: int, d: int):
        """Property N returns number of walkers."""
        x = torch.randn(N, d)
        v = torch.zeros(N, d)
        state = SwarmState(x, v)
        assert state.N == N

    def test_property_d(self, N: int, d: int):
        """Property d returns spatial dimension."""
        x = torch.randn(N, d)
        v = torch.zeros(N, d)
        state = SwarmState(x, v)
        assert state.d == d

    def test_property_device(self):
        """Property device returns correct device."""
        x = torch.randn(5, 3)
        v = torch.zeros(5, 3)
        state = SwarmState(x, v)
        assert state.device == torch.device("cpu")

    def test_property_dtype(self):
        """Property dtype returns correct dtype."""
        x = torch.randn(5, 3, dtype=torch.float64)
        v = torch.zeros(5, 3, dtype=torch.float64)
        state = SwarmState(x, v)
        assert state.dtype == torch.float64

    def test_clone_independence(self, swarm_state):
        """clone() creates an independent copy; modifying clone does not affect original."""
        original_x = swarm_state.x.clone()
        original_v = swarm_state.v.clone()

        cloned = swarm_state.clone()

        # Modify clone
        cloned.x += 100.0
        cloned.v -= 100.0

        # Original unchanged
        assert torch.equal(swarm_state.x, original_x)
        assert torch.equal(swarm_state.v, original_v)

    def test_copy_from_with_mask(self, N: int, d: int):
        """copy_from updates only masked walkers."""
        x1 = torch.zeros(N, d)
        v1 = torch.zeros(N, d)
        state1 = SwarmState(x1, v1)

        x2 = torch.ones(N, d)
        v2 = torch.ones(N, d) * 2.0
        state2 = SwarmState(x2, v2)

        mask = torch.zeros(N, dtype=torch.bool)
        mask[0] = True
        mask[N - 1] = True

        state1.copy_from(state2, mask)

        # Masked walkers should be updated
        assert torch.equal(state1.x[0], x2[0])
        assert torch.equal(state1.x[N - 1], x2[N - 1])
        assert torch.equal(state1.v[0], v2[0])
        assert torch.equal(state1.v[N - 1], v2[N - 1])

        # Unmasked walkers should remain unchanged
        if N > 2:
            assert torch.equal(state1.x[1], torch.zeros(d))
            assert torch.equal(state1.v[1], torch.zeros(d))

    def test_copy_from_type_error(self, swarm_state):
        """copy_from raises TypeError for non-SwarmState source."""
        mask = torch.ones(swarm_state.N, dtype=torch.bool)
        with pytest.raises(TypeError, match="Expected SwarmState"):
            swarm_state.copy_from("not a state", mask)

    def test_copy_from_mask_dtype_error(self, swarm_state):
        """copy_from raises ValueError for non-bool mask."""
        other = swarm_state.clone()
        mask = torch.ones(swarm_state.N, dtype=torch.float32)
        with pytest.raises(ValueError, match="Mask must be boolean"):
            swarm_state.copy_from(other, mask)

    def test_copy_from_mask_size_mismatch(self, swarm_state):
        """copy_from raises ValueError for mask size mismatch."""
        other = swarm_state.clone()
        mask = torch.ones(swarm_state.N + 5, dtype=torch.bool)
        with pytest.raises(ValueError, match="Mask size mismatch"):
            swarm_state.copy_from(other, mask)

    def test_shape_mismatch_raises(self):
        """Mismatched x and v shapes raise AssertionError."""
        x = torch.randn(10, 3)
        v = torch.randn(10, 4)
        with pytest.raises(AssertionError, match="same shape"):
            SwarmState(x, v)


# =========================================================================
# Helper: build a minimal EuclideanGas for testing
# =========================================================================


@pytest.fixture
def gas_kinetic_op():
    """KineticOperator properly configured for EuclideanGas integration tests.

    Uses riemannian_kernel_volume weighting to match the gas default
    neighbor_weight_modes, ensuring edge weights are available for viscous
    coupling computation.
    """
    from fragile.physics.fractal_gas.kinetic_operator import KineticOperator

    return KineticOperator(
        gamma=1.0,
        beta=1.0,
        delta_t=0.01,
        temperature=0.5,
        nu=0.1,
        use_viscous_coupling=True,
        viscous_neighbor_weighting="riemannian_kernel_volume",
        beta_curl=0.0,
    )


@pytest.fixture
def gas(gas_kinetic_op, clone_op, fitness_op) -> EuclideanGas:
    """Build a minimal EuclideanGas with N=20, d=3 for testing."""
    return EuclideanGas(
        N=20,
        d=3,
        kinetic_op=gas_kinetic_op,
        cloning=clone_op,
        fitness_op=fitness_op,
        neighbor_weight_modes=["riemannian_kernel_volume"],
    )


@pytest.fixture
def gas_2d(gas_kinetic_op, clone_op, fitness_op) -> EuclideanGas:
    """Build a minimal EuclideanGas with N=20, d=2 for testing."""
    return EuclideanGas(
        N=20,
        d=2,
        kinetic_op=gas_kinetic_op,
        cloning=clone_op,
        fitness_op=fitness_op,
        neighbor_weight_modes=["riemannian_kernel_volume"],
    )


# =========================================================================
# TestEuclideanGasInit
# =========================================================================


class TestEuclideanGasInit:
    """Tests for EuclideanGas initialization and initialize_state."""

    def test_default_parameters(self, kinetic_op, clone_op, fitness_op):
        """Default N, d, dtype are set correctly."""
        g = EuclideanGas(
            N=30,
            d=4,
            kinetic_op=kinetic_op,
            cloning=clone_op,
            fitness_op=fitness_op,
        )
        assert g.N == 30
        assert g.d == 4
        assert g.dtype == "float32"

    def test_torch_dtype_property(self, gas):
        """torch_dtype returns correct PyTorch dtype."""
        assert gas.torch_dtype == torch.float32

    def test_torch_dtype_float64(self, kinetic_op, clone_op, fitness_op):
        """torch_dtype returns float64 when configured."""
        g = EuclideanGas(
            N=10,
            d=2,
            dtype="float64",
            kinetic_op=kinetic_op,
            cloning=clone_op,
            fitness_op=fitness_op,
        )
        assert g.torch_dtype == torch.float64

    def test_initialize_state_defaults(self, gas):
        """initialize_state with defaults: random x, zero v."""
        torch.manual_seed(0)
        state = gas.initialize_state()
        assert state.N == gas.N
        assert state.d == gas.d
        assert state.dtype == gas.torch_dtype
        # v should be zeros by default
        assert torch.all(state.v == 0)
        # x should not be all zeros (random)
        assert not torch.all(state.x == 0)

    def test_initialize_state_custom(self, gas):
        """initialize_state with custom x_init and v_init."""
        x_custom = torch.ones(gas.N, gas.d)
        v_custom = torch.full((gas.N, gas.d), 2.0)
        state = gas.initialize_state(x_init=x_custom, v_init=v_custom)
        assert torch.allclose(state.x, x_custom)
        assert torch.allclose(state.v, v_custom)

    def test_device_dtype_propagation(self, kinetic_op, clone_op, fitness_op):
        """Device and dtype propagate to initialized state tensors."""
        g = EuclideanGas(
            N=10,
            d=2,
            dtype="float64",
            device=torch.device("cpu"),
            kinetic_op=kinetic_op,
            cloning=clone_op,
            fitness_op=fitness_op,
        )
        state = g.initialize_state()
        assert state.dtype == torch.float64
        assert state.device == torch.device("cpu")


# =========================================================================
# TestEuclideanGasStep
# =========================================================================


class TestEuclideanGasStep:
    """Tests for EuclideanGas.step()."""

    def test_step_returns_two_states(self, gas):
        """step() with return_info=False returns (state_cloned, state_final)."""
        torch.manual_seed(42)
        state = gas.initialize_state()
        result = gas.step(state, return_info=False)
        assert isinstance(result, tuple)
        assert len(result) == 2
        state_cloned, state_final = result
        assert isinstance(state_cloned, SwarmState)
        assert isinstance(state_final, SwarmState)

    def test_step_returns_info(self, gas):
        """step() with return_info=True returns (state_cloned, state_final, info)."""
        torch.manual_seed(42)
        state = gas.initialize_state()
        result = gas.step(state, return_info=True)
        assert isinstance(result, tuple)
        assert len(result) == 3
        state_cloned, state_final, info = result
        assert isinstance(state_cloned, SwarmState)
        assert isinstance(state_final, SwarmState)
        assert isinstance(info, dict)

    def test_step_info_keys(self, gas):
        """return_info=True provides info dict with expected keys."""
        torch.manual_seed(42)
        state = gas.initialize_state()
        _, _, info = gas.step(state, return_info=True)
        expected_keys = {"fitness", "rewards", "companions_distance", "companions_clone"}
        assert expected_keys.issubset(info.keys()), f"Missing keys: {expected_keys - info.keys()}"

    def test_step_preserves_shapes(self, gas):
        """State shapes are preserved after step."""
        torch.manual_seed(42)
        state = gas.initialize_state()
        state_cloned, state_final = gas.step(state, return_info=False)
        assert state_cloned.N == gas.N
        assert state_cloned.d == gas.d
        assert state_final.N == gas.N
        assert state_final.d == gas.d

    def test_step_modifies_state(self, gas):
        """Step should modify positions and/or velocities (not identical to input)."""
        torch.manual_seed(42)
        state = gas.initialize_state()
        x_before = state.x.clone()
        v_before = state.v.clone()

        _, state_final = gas.step(state, return_info=False)

        # At minimum, the kinetic update should change positions
        # (unless delta_t=0 which is not the case here)
        changed = not torch.equal(state_final.x, x_before) or not torch.equal(
            state_final.v, v_before
        )
        assert changed, "Step should modify positions or velocities"


# =========================================================================
# TestEuclideanGasRun
# =========================================================================


class TestEuclideanGasRun:
    """Tests for EuclideanGas.run()."""

    def test_run_returns_run_history(self, gas):
        """run() returns a RunHistory object."""
        from fragile.physics.fractal_gas.history import RunHistory

        torch.manual_seed(42)
        history = gas.run(n_steps=3, seed=42)
        assert isinstance(history, RunHistory)

    def test_run_history_metadata(self, gas):
        """RunHistory has correct N, d, n_steps, n_recorded."""
        history = gas.run(n_steps=5, seed=42)
        assert history.N == gas.N
        assert history.d == gas.d
        assert history.n_steps == 5

    def test_record_every(self, gas):
        """record_every controls recording frequency."""
        history = gas.run(n_steps=10, record_every=3, seed=42)
        # Steps: 0, 3, 6, 9, 10 (final always recorded)
        expected_steps = [0, 3, 6, 9, 10]
        assert history.recorded_steps == expected_steps
        assert history.n_recorded == len(expected_steps)

    def test_seed_determinism(self, gas):
        """Same seed produces deterministic results."""
        h1 = gas.run(n_steps=5, seed=123)
        h2 = gas.run(n_steps=5, seed=123)
        assert torch.allclose(h1.x_final, h2.x_final), "Same seed should yield same x_final"
        assert torch.allclose(h1.v_final, h2.v_final), "Same seed should yield same v_final"

    def test_recorded_steps_list(self, gas):
        """recorded_steps list matches expectations for record_every=1."""
        history = gas.run(n_steps=4, record_every=1, seed=42)
        assert history.recorded_steps == [0, 1, 2, 3, 4]

    def test_summary_returns_string(self, gas):
        """summary() returns a non-empty string."""
        history = gas.run(n_steps=3, seed=42)
        s = history.summary()
        assert isinstance(s, str)
        assert len(s) > 0
        assert "RunHistory" in s
