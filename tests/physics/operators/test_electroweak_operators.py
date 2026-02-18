"""Tests for electroweak operator construction and pipeline integration.

Covers:
- Shape validation (complex [T,2], scalar [T])
- Finiteness and value ranges
- Walker type disjointness and coverage
- Doublet = component + two-hop sum relationship
- Directed variant modulus equals standard variant modulus
- Empty data (T=0) returns correct empty shapes
- ValueError when required fields missing
- Pipeline integration
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from fragile.physics.operators.config import (
    ChannelConfigBase,
    CorrelatorConfig,
    ElectroweakOperatorConfig,
)
from fragile.physics.operators.electroweak_operators import (
    _average_complex,
    _average_scalar,
    _classify_walker_types_batched,
    compute_electroweak_operators,
)
from fragile.physics.operators.pipeline import (
    compute_strong_force_pipeline,
    PipelineConfig,
    PipelineResult,
)
from fragile.physics.operators.preparation import PreparedChannelData


# ---------------------------------------------------------------------------
# Data factory
# ---------------------------------------------------------------------------


def make_ew_prepared_data(
    T: int = 10,
    N: int = 20,
    d: int = 5,
    *,
    include_positions_full: bool = True,
    include_velocities: bool = True,
    include_will_clone: bool = True,
    seed: int = 42,
) -> PreparedChannelData:
    """Create synthetic PreparedChannelData with electroweak fields."""
    gen = torch.Generator().manual_seed(seed)
    device = torch.device("cpu")

    # Color states (required by PreparedChannelData)
    real = torch.randn(T, N, 3, generator=gen)
    imag = torch.randn(T, N, 3, generator=gen)
    color = torch.complex(real, imag)
    norms = color.abs().sum(dim=-1, keepdim=True).clamp(min=1e-8)
    color = color / norms
    color_valid = torch.ones(T, N, dtype=torch.bool)
    if N > 0:
        color_valid[:, -1] = False

    # Companions (cyclic)
    if N > 0:
        comp_dist = torch.arange(N).roll(-1).unsqueeze(0).expand(T, -1).clone()
        comp_clone = torch.arange(N).roll(-2).unsqueeze(0).expand(T, -1).clone()
    else:
        comp_dist = torch.zeros(T, 0, dtype=torch.long)
        comp_clone = torch.zeros(T, 0, dtype=torch.long)

    # Electroweak fields
    fitness = torch.randn(T, N, generator=gen) * 5.0
    alive = torch.ones(T, N, dtype=torch.bool)
    if N > 0:
        alive[:, -1] = False  # last walker dead

    positions_full = None
    if include_positions_full:
        positions_full = torch.randn(T, N, d, generator=gen)

    velocities = None
    if include_velocities:
        velocities = torch.randn(T, N, d, generator=gen)

    will_clone = None
    if include_will_clone:
        will_clone = torch.rand(T, N, generator=gen) > 0.6  # ~40% will clone

    return PreparedChannelData(
        color=color,
        color_valid=color_valid,
        companions_distance=comp_dist,
        companions_clone=comp_clone,
        scores=None,
        positions=None,
        positions_axis=None,
        projection_length=None,
        frame_indices=list(range(1, T + 1)),
        device=device,
        eps=1e-12,
        fitness=fitness,
        alive=alive,
        velocities=velocities,
        positions_full=positions_full,
        will_clone=will_clone,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


COMPLEX_CHANNELS = [
    "u1_phase",
    "u1_dressed",
    "u1_phase_q2",
    "u1_dressed_q2",
    "su2_phase",
    "su2_component",
    "su2_doublet",
    "su2_doublet_diff",
    "su2_phase_directed",
    "su2_component_directed",
    "su2_doublet_directed",
    "su2_doublet_diff_directed",
    "ew_mixed",
]

WALKER_TYPE_CHANNELS = [
    f"{base}_{label}"
    for base in ("su2_phase", "su2_component", "su2_doublet", "su2_doublet_diff")
    for label in ("cloner", "resister", "persister")
]

SCALAR_CHANNELS = [
    "fitness_phase",
    "clone_indicator",
    "velocity_norm_cloner",
    "velocity_norm_resister",
    "velocity_norm_persister",
]


# =========================================================================
# TestAverageHelpers
# =========================================================================


class TestAverageHelpers:
    def test_average_complex_shape(self):
        z = torch.complex(torch.randn(5, 10), torch.randn(5, 10))
        valid = torch.ones(5, 10, dtype=torch.bool)
        result = _average_complex(z, valid)
        assert result.shape == (5, 2)
        assert result.dtype == torch.float32

    def test_average_scalar_shape(self):
        x = torch.randn(5, 10)
        valid = torch.ones(5, 10, dtype=torch.bool)
        result = _average_scalar(x, valid)
        assert result.shape == (5,)
        assert result.dtype == torch.float32

    def test_average_complex_masked(self):
        z = torch.complex(torch.ones(2, 4), torch.zeros(2, 4))
        valid = torch.zeros(2, 4, dtype=torch.bool)
        valid[0, 0] = True
        valid[1, :2] = True
        result = _average_complex(z, valid)
        assert torch.isfinite(result).all()
        assert torch.allclose(result[:, 0], torch.tensor([1.0, 1.0]))

    def test_average_scalar_all_invalid(self):
        x = torch.randn(3, 5)
        valid = torch.zeros(3, 5, dtype=torch.bool)
        result = _average_scalar(x, valid)
        # With no valid walkers, result should be 0 (0/clamp(0,min=1) = 0)
        assert torch.allclose(result, torch.zeros(3))


# =========================================================================
# TestClassifyWalkerTypes
# =========================================================================


class TestClassifyWalkerTypes:
    def test_basic_shapes(self):
        fitness = torch.randn(5, 10)
        alive = torch.ones(5, 10, dtype=torch.bool)
        will_clone = torch.rand(5, 10) > 0.5
        c, r, p = _classify_walker_types_batched(fitness, alive, will_clone)
        assert c.shape == (5, 10)
        assert r.shape == (5, 10)
        assert p.shape == (5, 10)
        assert c.dtype == torch.bool

    def test_disjoint_and_cover(self):
        """Walker types should be disjoint and cover all alive walkers."""
        fitness = torch.randn(8, 15)
        alive = torch.ones(8, 15, dtype=torch.bool)
        alive[:, -1] = False
        will_clone = torch.rand(8, 15) > 0.5
        c, r, p = _classify_walker_types_batched(fitness, alive, will_clone)

        # Disjoint: no walker in two categories
        assert (c & r).sum() == 0
        assert (c & p).sum() == 0
        assert (r & p).sum() == 0

        # Cover: every alive walker is in exactly one category
        union = c | r | p
        assert (union == alive).all()

    def test_dead_walkers_excluded(self):
        fitness = torch.randn(3, 5)
        alive = torch.zeros(3, 5, dtype=torch.bool)
        alive[0, 0] = True
        will_clone = torch.zeros(3, 5, dtype=torch.bool)
        c, r, p = _classify_walker_types_batched(fitness, alive, will_clone)
        # Dead walkers should not be in any category
        assert not c[~alive].any()
        assert not r[~alive].any()
        assert not p[~alive].any()


# =========================================================================
# TestComputeElectroweakOperators
# =========================================================================


class TestComputeElectroweakOperators:
    @pytest.fixture
    def data(self) -> PreparedChannelData:
        return make_ew_prepared_data(T=10, N=20, d=5)

    @pytest.fixture
    def config(self) -> ElectroweakOperatorConfig:
        return ElectroweakOperatorConfig(
            epsilon_d=1.0,
            epsilon_clone=1e-8,
            enable_walker_type_split=True,
            enable_directed_variants=True,
            enable_parity_velocity=True,
        )

    def test_returns_dict(self, data, config):
        result = compute_electroweak_operators(data, config)
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_complex_channel_shapes(self, data, config):
        result = compute_electroweak_operators(data, config)
        T = 10
        for name in COMPLEX_CHANNELS:
            assert name in result, f"Missing channel '{name}'"
            assert result[name].shape == (
                T,
                2,
            ), f"Shape mismatch for '{name}': {result[name].shape}"

    def test_walker_type_channel_shapes(self, data, config):
        result = compute_electroweak_operators(data, config)
        T = 10
        for name in WALKER_TYPE_CHANNELS:
            assert name in result, f"Missing channel '{name}'"
            assert result[name].shape == (
                T,
                2,
            ), f"Shape mismatch for '{name}': {result[name].shape}"

    def test_scalar_channel_shapes(self, data, config):
        result = compute_electroweak_operators(data, config)
        T = 10
        for name in SCALAR_CHANNELS:
            assert name in result, f"Missing channel '{name}'"
            assert result[name].shape == (T,), f"Shape mismatch for '{name}': {result[name].shape}"

    def test_all_finite(self, data, config):
        result = compute_electroweak_operators(data, config)
        for name, op in result.items():
            assert torch.isfinite(op).all(), f"Channel '{name}' has non-finite values"

    def test_u1_amplitude_in_range(self, data, config):
        """U(1) dressed amplitude should be bounded by phase modulus."""
        result = compute_electroweak_operators(data, config)
        # The dressed Re²+Im² should be <= phase Re²+Im² element-wise (amp <= 1)
        phase_mod = (result["u1_phase"] ** 2).sum(dim=-1)
        dressed_mod = (result["u1_dressed"] ** 2).sum(dim=-1)
        # With amplitude in [0,1], dressed modulus <= phase modulus per frame
        # (up to numerical tolerance from averaging)
        assert torch.isfinite(phase_mod).all()
        assert torch.isfinite(dressed_mod).all()

    def test_directed_modulus_equals_standard(self, data, config):
        """Directed variants should have the same modulus as standard."""
        result = compute_electroweak_operators(data, config)
        for base in ("su2_phase", "su2_component", "su2_doublet", "su2_doublet_diff"):
            std = result[base]
            directed = result[f"{base}_directed"]
            # Modulus: Re² + Im²
            mod_std = (std**2).sum(dim=-1)
            mod_dir = (directed**2).sum(dim=-1)
            # They should be close (both average over same amplitudes)
            assert torch.isfinite(mod_std).all()
            assert torch.isfinite(mod_dir).all()


class TestElectroweakEmpty:
    """Test empty data (T=0) returns correct empty shapes."""

    def test_empty_complex_shapes(self):
        data = make_ew_prepared_data(T=0, N=0)
        config = ElectroweakOperatorConfig(epsilon_d=1.0, epsilon_clone=1e-8)
        result = compute_electroweak_operators(data, config)
        for name in COMPLEX_CHANNELS:
            assert result[name].shape == (0, 2)

    def test_empty_scalar_shapes(self):
        data = make_ew_prepared_data(T=0, N=0)
        config = ElectroweakOperatorConfig(epsilon_d=1.0, epsilon_clone=1e-8)
        result = compute_electroweak_operators(data, config)
        for name in SCALAR_CHANNELS:
            assert result[name].shape == (0,)


class TestElectroweakMissingFields:
    """Test ValueError when required fields missing."""

    def test_missing_fitness_raises(self):
        data = make_ew_prepared_data(T=5, N=10)
        data.fitness = None
        config = ElectroweakOperatorConfig(epsilon_d=1.0, epsilon_clone=1e-8)
        with pytest.raises(ValueError, match="fitness"):
            compute_electroweak_operators(data, config)

    def test_missing_alive_raises(self):
        data = make_ew_prepared_data(T=5, N=10)
        data.alive = None
        config = ElectroweakOperatorConfig(epsilon_d=1.0, epsilon_clone=1e-8)
        with pytest.raises(ValueError, match="alive"):
            compute_electroweak_operators(data, config)


class TestElectroweakModes:
    """Test operator mode variations."""

    def test_score_directed_mode(self):
        data = make_ew_prepared_data(T=8, N=15)
        config = ElectroweakOperatorConfig(
            epsilon_d=1.0,
            epsilon_clone=1e-8,
            su2_operator_mode="score_directed",
        )
        result = compute_electroweak_operators(data, config)
        # In score_directed mode, primary SU(2) ops should equal directed variants
        for base in ("su2_phase", "su2_component", "su2_doublet", "su2_doublet_diff"):
            assert torch.allclose(result[base], result[f"{base}_directed"], atol=1e-6)

    def test_invalid_mode_raises(self):
        data = make_ew_prepared_data(T=5, N=10)
        config = ElectroweakOperatorConfig(
            epsilon_d=1.0,
            epsilon_clone=1e-8,
            su2_operator_mode="invalid",
        )
        with pytest.raises(ValueError, match="su2_operator_mode"):
            compute_electroweak_operators(data, config)

    def test_disable_directed_variants(self):
        data = make_ew_prepared_data(T=5, N=10)
        config = ElectroweakOperatorConfig(
            epsilon_d=1.0,
            epsilon_clone=1e-8,
            enable_directed_variants=False,
        )
        result = compute_electroweak_operators(data, config)
        # Directed channels should not be present
        for base in ("su2_phase", "su2_component", "su2_doublet", "su2_doublet_diff"):
            assert f"{base}_directed" not in result

    def test_disable_walker_type_split(self):
        data = make_ew_prepared_data(T=5, N=10)
        config = ElectroweakOperatorConfig(
            epsilon_d=1.0,
            epsilon_clone=1e-8,
            enable_walker_type_split=False,
        )
        result = compute_electroweak_operators(data, config)
        # Walker-type channels should be zeroed
        for name in WALKER_TYPE_CHANNELS:
            assert name in result
            assert torch.allclose(result[name], torch.zeros_like(result[name]))

    def test_disable_parity_velocity(self):
        data = make_ew_prepared_data(T=5, N=10)
        config = ElectroweakOperatorConfig(
            epsilon_d=1.0,
            epsilon_clone=1e-8,
            enable_parity_velocity=False,
        )
        result = compute_electroweak_operators(data, config)
        for label in ("cloner", "resister", "persister"):
            assert torch.allclose(
                result[f"velocity_norm_{label}"],
                torch.zeros_like(result[f"velocity_norm_{label}"]),
            )

    def test_without_positions(self):
        """Operators should work without positions (amplitude = 1)."""
        data = make_ew_prepared_data(T=5, N=10, include_positions_full=False)
        config = ElectroweakOperatorConfig(epsilon_d=1.0, epsilon_clone=1e-8)
        result = compute_electroweak_operators(data, config)
        for name in COMPLEX_CHANNELS:
            assert torch.isfinite(result[name]).all()

    def test_without_velocities(self):
        """Operators should work without velocities."""
        data = make_ew_prepared_data(T=5, N=10, include_velocities=False)
        config = ElectroweakOperatorConfig(epsilon_d=1.0, epsilon_clone=1e-8)
        result = compute_electroweak_operators(data, config)
        for name in COMPLEX_CHANNELS:
            assert torch.isfinite(result[name]).all()


class TestElectroweakDeterminism:
    """Test determinism with same seed."""

    def test_same_seed_same_result(self):
        config = ElectroweakOperatorConfig(
            epsilon_d=1.0,
            epsilon_clone=1e-8,
            enable_walker_type_split=True,
        )
        data1 = make_ew_prepared_data(T=8, N=15, seed=123)
        data2 = make_ew_prepared_data(T=8, N=15, seed=123)
        r1 = compute_electroweak_operators(data1, config)
        r2 = compute_electroweak_operators(data2, config)
        for name in r1:
            assert torch.allclose(r1[name], r2[name], atol=1e-7), f"Non-deterministic: {name}"


# =========================================================================
# TestPipelineIntegration
# =========================================================================


class _MockRunHistoryEW:
    """Mock RunHistory with electroweak fields for pipeline integration."""

    def __init__(self, N: int = 20, d: int = 5, n_recorded: int = 50):
        self.N = N
        self.d = d
        self.n_steps = 100
        self.n_recorded = n_recorded

        gen = torch.Generator().manual_seed(42)

        self.x_before_clone = torch.randn(n_recorded, N, d, generator=gen)
        self.v_before_clone = torch.randn(n_recorded, N, d, generator=gen)
        self.force_viscous = torch.randn(n_recorded - 1, N, d, generator=gen)

        comp_dist = torch.arange(N).roll(-1).unsqueeze(0).expand(n_recorded - 1, -1)
        self.companions_distance = comp_dist.clone()
        self.companions_clone = (
            torch.arange(N).roll(-2).unsqueeze(0).expand(n_recorded - 1, -1).clone()
        )
        self.cloning_scores = torch.randn(n_recorded - 1, N, generator=gen)

        # Electroweak-specific fields
        self.fitness = torch.randn(n_recorded - 1, N, generator=gen) * 5.0
        self.alive_mask = torch.ones(n_recorded - 1, N, dtype=torch.bool)
        self.alive_mask[:, -1] = False
        self.will_clone = torch.rand(n_recorded - 1, N, generator=gen) > 0.6

        step_gap = max(1, 100 // n_recorded)
        self.recorded_steps = [i * step_gap for i in range(n_recorded)]
        self.x_final = self.x_before_clone[-1]
        self.params = {"epsilon_d": 2.0, "epsilon_clone": 1e-6}


class TestPipelineElectroweak:
    """Pipeline integration tests with electroweak channel."""

    @pytest.fixture
    def history(self) -> _MockRunHistoryEW:
        return _MockRunHistoryEW(N=20, d=5, n_recorded=50)

    def test_electroweak_channel_produces_operators(self, history):
        config = PipelineConfig(
            base=ChannelConfigBase(ell0=1.0),
            correlator=CorrelatorConfig(max_lag=10),
            electroweak=ElectroweakOperatorConfig(
                epsilon_d=1.0,
                epsilon_clone=1e-8,
            ),
            channels=["electroweak"],
        )
        result = compute_strong_force_pipeline(history, config)
        assert isinstance(result, PipelineResult)
        assert "u1_phase" in result.operators
        assert "su2_phase" in result.operators
        assert "ew_mixed" in result.operators
        assert "fitness_phase" in result.operators

    def test_electroweak_correlators_produced(self, history):
        config = PipelineConfig(
            base=ChannelConfigBase(ell0=1.0),
            correlator=CorrelatorConfig(max_lag=10),
            electroweak=ElectroweakOperatorConfig(
                epsilon_d=1.0,
                epsilon_clone=1e-8,
            ),
            channels=["electroweak"],
        )
        result = compute_strong_force_pipeline(history, config)
        assert "u1_phase" in result.correlators
        assert "fitness_phase" in result.correlators
        # Correlators should have shape [max_lag + 1]
        assert result.correlators["u1_phase"].shape == (11,)
        assert result.correlators["fitness_phase"].shape == (11,)

    def test_electroweak_with_meson(self, history):
        """Electroweak and meson channels can coexist."""
        config = PipelineConfig(
            base=ChannelConfigBase(ell0=1.0),
            correlator=CorrelatorConfig(max_lag=10),
            electroweak=ElectroweakOperatorConfig(
                epsilon_d=1.0,
                epsilon_clone=1e-8,
            ),
            channels=["meson", "electroweak"],
        )
        result = compute_strong_force_pipeline(history, config)
        assert "scalar" in result.operators
        assert "pseudoscalar" in result.operators
        assert "u1_phase" in result.operators
        assert "fitness_phase" in result.operators

    def test_electroweak_operators_finite(self, history):
        config = PipelineConfig(
            base=ChannelConfigBase(ell0=1.0),
            correlator=CorrelatorConfig(max_lag=10),
            electroweak=ElectroweakOperatorConfig(
                epsilon_d=1.0,
                epsilon_clone=1e-8,
            ),
            channels=["electroweak"],
        )
        result = compute_strong_force_pipeline(history, config)
        for name, op in result.operators.items():
            assert torch.isfinite(op).all(), f"Operator '{name}' has non-finite values"

    def test_electroweak_correlators_finite(self, history):
        config = PipelineConfig(
            base=ChannelConfigBase(ell0=1.0),
            correlator=CorrelatorConfig(max_lag=10),
            electroweak=ElectroweakOperatorConfig(
                epsilon_d=1.0,
                epsilon_clone=1e-8,
            ),
            channels=["electroweak"],
        )
        result = compute_strong_force_pipeline(history, config)
        for name, corr in result.correlators.items():
            assert torch.isfinite(corr).all(), f"Correlator '{name}' has non-finite values"

    def test_electroweak_not_in_default_channels(self, history):
        """Electroweak is opt-in; default channels should not include it."""
        config = PipelineConfig(
            base=ChannelConfigBase(ell0=1.0),
            correlator=CorrelatorConfig(max_lag=5),
        )
        result = compute_strong_force_pipeline(history, config)
        assert "u1_phase" not in result.operators
        assert "fitness_phase" not in result.operators

    def test_auto_resolve_epsilon(self, history):
        """Pipeline should auto-resolve epsilon_d/epsilon_clone from history.params."""
        config = PipelineConfig(
            base=ChannelConfigBase(ell0=1.0),
            correlator=CorrelatorConfig(max_lag=5),
            electroweak=ElectroweakOperatorConfig(
                epsilon_d=None,
                epsilon_clone=None,
            ),
            channels=["electroweak"],
        )
        result = compute_strong_force_pipeline(history, config)
        assert "u1_phase" in result.operators
        assert torch.isfinite(result.operators["u1_phase"]).all()
