"""Comprehensive tests for KineticOperator (Langevin dynamics with BAOAB/Boris-BAOAB)."""

from __future__ import annotations

import warnings

import pytest
import torch
from torch import Tensor

from fragile.physics.fractal_gas.euclidean_gas import SwarmState
from fragile.physics.fractal_gas.kinetic_operator import KineticOperator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_op(**kwargs) -> KineticOperator:
    """Shorthand for building a KineticOperator with sensible defaults."""
    defaults = {"gamma": 1.0, "beta": 1.0, "delta_t": 0.01, "temperature": 0.5, "nu": 0.1}
    defaults.update(kwargs)
    return KineticOperator(**defaults)


# ===================================================================
# TestKineticOperatorInit
# ===================================================================


class TestKineticOperatorInit:
    """Tests for __init__ parameter storage and validation."""

    def test_parameters_stored_correctly(self):
        op = KineticOperator(
            gamma=2.0,
            beta=0.5,
            delta_t=0.02,
            temperature=1.5,
            nu=0.3,
            use_viscous_coupling=True,
            beta_curl=0.7,
        )
        assert float(op.gamma) == 2.0
        assert float(op.beta) == 0.5
        assert float(op.delta_t) == 0.02
        assert float(op.temperature) == 1.5
        assert float(op.nu) == 0.3
        assert op.use_viscous_coupling is True
        assert float(op.beta_curl) == 0.7

    def test_device_and_dtype_defaults(self):
        op = _make_op()
        assert op.device == torch.device("cpu")
        assert op.dtype == torch.float32

    def test_device_and_dtype_explicit(self):
        op = _make_op(device=torch.device("cpu"), dtype=torch.float64)
        assert op.device == torch.device("cpu")
        assert op.dtype == torch.float64

    def test_callable_curl_field_accepted(self):
        def curl_fn(x):
            return torch.zeros(x.shape[0], x.shape[1], x.shape[1])

        op = _make_op(curl_field=curl_fn, beta_curl=1.0)
        assert op.curl_field is curl_fn

    def test_non_callable_curl_field_raises(self):
        with pytest.raises(TypeError, match="curl_field must be callable"):
            _make_op(curl_field="not_callable", beta_curl=1.0)

    def test_ou_coefficients_computed(self):
        op = _make_op()
        assert hasattr(op, "c1")
        assert hasattr(op, "c2")
        assert isinstance(op.c1, Tensor)
        assert isinstance(op.c2, Tensor)


# ===================================================================
# TestEffectiveBeta
# ===================================================================


class TestEffectiveBeta:
    """Tests for effective_beta, effective_temperature, noise_std."""

    def test_manual_mode_returns_beta(self):
        op = _make_op(beta=3.0)
        op.auto_thermostat = False
        assert op.effective_beta() == pytest.approx(3.0)

    def test_auto_thermostat_fdt(self):
        """beta_eff = 1 / temperature."""
        op = _make_op(gamma=2.0, temperature=0.25)
        op.auto_thermostat = True
        expected = 1.0 / 0.25  # 4.0
        assert op.effective_beta() == pytest.approx(expected)

    def test_effective_temperature_inverse(self):
        op = _make_op(beta=4.0)
        op.auto_thermostat = False
        assert op.effective_temperature() == pytest.approx(1.0 / 4.0)

    def test_noise_std_returns_float(self):
        op = _make_op()
        ns = op.noise_std()
        assert isinstance(ns, float)
        assert ns > 0.0


# ===================================================================
# TestComputeViscousForce
# ===================================================================


class TestComputeViscousForce:
    """Tests for _compute_viscous_force."""

    def test_output_shape(self, positions, velocities, simple_neighbor_edges, simple_edge_weights):
        op = _make_op(nu=0.5)
        force = op._compute_viscous_force(
            positions, velocities, simple_neighbor_edges, simple_edge_weights
        )
        assert force.shape == velocities.shape

    def test_zero_nu_gives_zero_force(
        self, positions, velocities, simple_neighbor_edges, simple_edge_weights
    ):
        op = _make_op(nu=0.0)
        force = op._compute_viscous_force(
            positions, velocities, simple_neighbor_edges, simple_edge_weights
        )
        assert torch.allclose(force, torch.zeros_like(force))

    def test_none_edges_returns_zero_with_warning(self, positions, velocities):
        op = _make_op(nu=1.0)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            force = op._compute_viscous_force(positions, velocities, None, None)
            assert any("neighbor_edges empty" in str(warn.message) for warn in w)
        assert torch.allclose(force, torch.zeros_like(force))

    def test_empty_edges_returns_zero_with_warning(self, positions, velocities):
        op = _make_op(nu=1.0)
        empty_edges = torch.zeros(0, 2, dtype=torch.long)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            force = op._compute_viscous_force(positions, velocities, empty_edges, None)
            assert any("neighbor_edges empty" in str(warn.message) for warn in w)
        assert torch.allclose(force, torch.zeros_like(force))

    def test_uniform_velocity_gives_zero_force(self):
        """When all walkers have the same velocity, v_j - v_i = 0."""
        N, d = 10, 3
        x = torch.randn(N, d)
        v = torch.ones(N, d) * 2.0  # identical velocities
        edges = torch.tensor([[0, 1], [1, 0], [1, 2], [2, 1]], dtype=torch.long)
        weights = torch.ones(edges.shape[0])
        op = _make_op(nu=1.0)
        force = op._compute_viscous_force(x, v, edges, weights)
        assert torch.allclose(force, torch.zeros_like(force), atol=1e-7)

    def test_known_two_walkers(self):
        """Two walkers: v0=[1,0], v1=[0,1], edges 0->1 and 1->0, unit weights."""
        x = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        v = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        edges = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        weights = torch.ones(2)
        nu = 0.5
        op = _make_op(nu=nu)

        force = op._compute_viscous_force(x, v, edges, weights)
        # walker 0: nu * w * (v1 - v0) = 0.5 * 1 * ([-1, 1]) = [-0.5, 0.5]
        expected_0 = torch.tensor([-0.5, 0.5])
        # walker 1: nu * w * (v0 - v1) = 0.5 * 1 * ([1, -1]) = [0.5, -0.5]
        expected_1 = torch.tensor([0.5, -0.5])
        assert torch.allclose(force[0], expected_0, atol=1e-6)
        assert torch.allclose(force[1], expected_1, atol=1e-6)

    def test_self_loops_filtered_out(self):
        """Edges containing self-loops (i==j) should be ignored."""
        x = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        v = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        # Include self-loop (0->0) plus real edge
        edges = torch.tensor([[0, 0], [0, 1], [1, 0]], dtype=torch.long)
        weights = torch.ones(3)
        op = _make_op(nu=1.0)
        force = op._compute_viscous_force(x, v, edges, weights)
        # Self-loop contributes zero (v0 - v0 = 0, but it's also filtered).
        # Force on walker 0 from edge (0,1): nu * 1 * (v1-v0)  = 1.0 * [-1, 1]
        expected_0 = torch.tensor([-1.0, 1.0])
        assert torch.allclose(force[0], expected_0, atol=1e-6)


# ===================================================================
# TestComputeViscousCurl
# ===================================================================


class TestComputeViscousCurl:
    """Tests for _compute_viscous_curl."""

    def test_output_shape(self, positions, simple_neighbor_edges, simple_edge_weights):
        N, d = positions.shape
        viscous_force = torch.randn(N, d)
        op = _make_op()
        curl = op._compute_viscous_curl(
            positions, viscous_force, simple_neighbor_edges, simple_edge_weights
        )
        assert curl.shape == (N, d, d)

    def test_skew_symmetric(self, positions, simple_neighbor_edges, simple_edge_weights):
        N, d = positions.shape
        torch.manual_seed(99)
        viscous_force = torch.randn(N, d)
        op = _make_op()
        curl = op._compute_viscous_curl(
            positions, viscous_force, simple_neighbor_edges, simple_edge_weights
        )
        # curl + curl^T should be ~0
        sym_part = curl + curl.transpose(-1, -2)
        assert torch.allclose(sym_part, torch.zeros_like(sym_part), atol=1e-5)

    def test_uniform_viscous_force_gives_zero_curl(self):
        """If viscous force is identical everywhere, Jacobian is zero => curl is zero."""
        N, d = 10, 3
        x = torch.randn(N, d)
        viscous_force = torch.ones(N, d) * 3.0  # uniform
        edges = []
        for i in range(N - 1):
            edges.append([i, i + 1])
            edges.append([i + 1, i])
        edges = torch.tensor(edges, dtype=torch.long)
        weights = torch.ones(edges.shape[0])
        op = _make_op()
        curl = op._compute_viscous_curl(x, viscous_force, edges, weights)
        assert torch.allclose(curl, torch.zeros_like(curl), atol=1e-5)


# ===================================================================
# TestBorisRotate
# ===================================================================


class TestBorisRotate:
    """Tests for _boris_rotate."""

    def test_3d_vector_curl_preserves_magnitude(self):
        """Boris rotation with 3D vector curl should preserve |v|."""
        N = 10
        torch.manual_seed(7)
        v = torch.randn(N, 3)
        curl = 0.1 * torch.randn(N, 3)
        op = _make_op(beta_curl=1.0)
        v_rot = op._boris_rotate(v, curl)
        # Magnitudes should be preserved
        mag_before = v.norm(dim=-1)
        mag_after = v_rot.norm(dim=-1)
        assert torch.allclose(mag_before, mag_after, atol=1e-5)

    def test_matrix_curl_skew_symmetric_accepted(self):
        """Skew-symmetric [N, d, d] curl should be accepted."""
        N, d = 8, 4
        torch.manual_seed(8)
        A = torch.randn(N, d, d)
        skew = (A - A.transpose(-1, -2)) / 2  # make skew-symmetric
        v = torch.randn(N, d)
        op = _make_op(beta_curl=1.0)
        v_rot = op._boris_rotate(v, skew)
        assert v_rot.shape == v.shape

    def test_non_skew_symmetric_raises(self):
        """A non-skew-symmetric matrix curl should raise ValueError."""
        N, d = 5, 3
        v = torch.randn(N, d)
        # Symmetric matrix -- not skew-symmetric
        sym = torch.eye(d).unsqueeze(0).expand(N, d, d).clone()
        op = _make_op(beta_curl=1.0)
        with pytest.raises(ValueError, match="skew-symmetric"):
            op._boris_rotate(v, sym)

    def test_shape_mismatch_raises(self):
        """Mismatched curl vs v shapes should raise ValueError."""
        N = 5
        v = torch.randn(N, 3)
        curl_bad = torch.randn(N + 1, 3)  # wrong batch size
        op = _make_op(beta_curl=1.0)
        with pytest.raises(ValueError):
            op._boris_rotate(v, curl_bad)


# ===================================================================
# TestApply
# ===================================================================


class TestApply:
    """Tests for the full BAOAB apply step."""

    def test_returns_swarm_state_with_correct_shapes(self, swarm_state, kinetic_op):
        new_state = kinetic_op.apply(swarm_state)
        assert isinstance(new_state, SwarmState)
        assert new_state.x.shape == swarm_state.x.shape
        assert new_state.v.shape == swarm_state.v.shape

    def test_return_info_provides_expected_keys(self, swarm_state, kinetic_op):
        _new_state, info = kinetic_op.apply(swarm_state, return_info=True)
        expected_keys = {
            "force_stable",
            "force_adapt",
            "force_viscous",
            "force_friction",
            "force_total",
            "noise",
        }
        assert expected_keys.issubset(info.keys()), f"Missing keys: {expected_keys - info.keys()}"

    def test_position_changes_after_apply(self, swarm_state, kinetic_op):
        x_orig = swarm_state.x.clone()
        new_state = kinetic_op.apply(swarm_state)
        # With nonzero delta_t and velocity, position should change
        assert not torch.allclose(new_state.x, x_orig, atol=1e-12)

    def test_velocity_changes_due_to_noise(self, swarm_state, kinetic_op):
        v_orig = swarm_state.v.clone()
        new_state = kinetic_op.apply(swarm_state)
        # Noise in O-step makes velocity change very likely
        assert not torch.allclose(new_state.v, v_orig, atol=1e-12)

    def test_no_viscous_coupling_no_edges_still_works(self, swarm_state):
        """use_viscous_coupling=True but no edges provided -- should still run."""
        op = _make_op(use_viscous_coupling=True, nu=0.5)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            new_state = op.apply(swarm_state)
        assert isinstance(new_state, SwarmState)

    def test_with_neighbor_edges_viscous_force_nonzero(
        self, swarm_state, simple_neighbor_edges, simple_edge_weights
    ):
        op = _make_op(nu=1.0, use_viscous_coupling=True)
        _new_state, info = op.apply(
            swarm_state,
            neighbor_edges=simple_neighbor_edges,
            edge_weights=simple_edge_weights,
            return_info=True,
        )
        # With non-uniform velocities and neighbor edges, viscous force should be nonzero
        assert info["force_viscous"].abs().sum() > 0

    def test_boris_integrator_with_curl_field(self, swarm_state):
        """With a curl_field callable, Boris rotation should be applied."""
        d = swarm_state.d

        def curl_fn(x: Tensor) -> Tensor:
            N = x.shape[0]
            A = torch.randn(N, d, d)
            return (A - A.transpose(-1, -2)) / 2  # skew-symmetric

        op = _make_op(beta_curl=1.0, curl_field=curl_fn)
        new_state, info = op.apply(swarm_state, return_info=True)
        assert isinstance(new_state, SwarmState)
        # curl_field info should be recorded
        assert "curl_field" in info

    def test_multiple_sequential_applications(self, swarm_state, kinetic_op):
        state = swarm_state
        for _ in range(5):
            state = kinetic_op.apply(state)
        assert isinstance(state, SwarmState)
        assert state.x.shape == swarm_state.x.shape
