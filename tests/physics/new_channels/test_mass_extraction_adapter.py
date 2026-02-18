"""Tests for the new_channels → mass_extraction adapter."""

from __future__ import annotations

import pytest
import torch

from fragile.physics.new_channels.baryon_triplet_channels import (
    BaryonTripletCorrelatorOutput,
)
from fragile.physics.new_channels.correlator_channels import (
    ChannelCorrelatorResult,
)
from fragile.physics.new_channels.glueball_color_channels import (
    GlueballColorCorrelatorOutput,
)
from fragile.physics.new_channels.mass_extraction_adapter import (
    collect_correlators,
    extract_baryon,
    extract_glueball,
    extract_meson_phase,
    extract_multiscale,
    extract_tensor_momentum,
    extract_vector_meson,
)
from fragile.physics.new_channels.meson_phase_channels import (
    MesonPhaseCorrelatorOutput,
)
from fragile.physics.new_channels.multiscale_strong_force import (
    MultiscaleStrongForceOutput,
)
from fragile.physics.new_channels.tensor_momentum_channels import (
    TensorMomentumCorrelatorOutput,
)
from fragile.physics.new_channels.vector_meson_channels import (
    VectorMesonCorrelatorOutput,
)
from fragile.physics.operators.pipeline import PipelineResult


# ---------------------------------------------------------------------------
# Fixtures — lightweight mock outputs
# ---------------------------------------------------------------------------

LAG = 10  # max_lag + 1
T = 50  # time-series length
S = 3  # number of scales
N_MODES = 4  # momentum modes


@pytest.fixture
def meson_output() -> MesonPhaseCorrelatorOutput:
    return MesonPhaseCorrelatorOutput(
        pseudoscalar=torch.randn(LAG),
        pseudoscalar_raw=torch.randn(LAG),
        pseudoscalar_connected=torch.randn(LAG),
        scalar=torch.randn(LAG),
        scalar_raw=torch.randn(LAG),
        scalar_connected=torch.randn(LAG),
        counts=torch.ones(LAG, dtype=torch.int64),
        frame_indices=list(range(T)),
        pair_counts_per_frame=torch.ones(T, dtype=torch.int64),
        pair_selection="distance",
        mean_pseudoscalar=0.1,
        mean_scalar=0.2,
        disconnected_pseudoscalar=0.01,
        disconnected_scalar=0.02,
        n_valid_source_pairs=100,
        operator_pseudoscalar_series=torch.randn(T),
        operator_scalar_series=torch.randn(T),
    )


@pytest.fixture
def baryon_output() -> BaryonTripletCorrelatorOutput:
    return BaryonTripletCorrelatorOutput(
        correlator=torch.randn(LAG),
        correlator_raw=torch.randn(LAG),
        correlator_connected=torch.randn(LAG),
        counts=torch.ones(LAG, dtype=torch.int64),
        frame_indices=list(range(T)),
        triplet_counts_per_frame=torch.ones(T, dtype=torch.int64),
        disconnected_contribution=0.05,
        mean_baryon_real=0.3,
        mean_baryon_imag=0.0,
        n_valid_source_triplets=50,
        operator_baryon_series=torch.randn(T),
    )


@pytest.fixture
def vector_output() -> VectorMesonCorrelatorOutput:
    return VectorMesonCorrelatorOutput(
        vector=torch.randn(LAG),
        vector_raw=torch.randn(LAG),
        vector_connected=torch.randn(LAG),
        axial_vector=torch.randn(LAG),
        axial_vector_raw=torch.randn(LAG),
        axial_vector_connected=torch.randn(LAG),
        counts=torch.ones(LAG, dtype=torch.int64),
        frame_indices=list(range(T)),
        pair_counts_per_frame=torch.ones(T, dtype=torch.int64),
        pair_selection="distance",
        use_unit_displacement=True,
        mean_vector=torch.randn(3),
        mean_axial_vector=torch.randn(3),
        disconnected_vector=0.01,
        disconnected_axial_vector=0.02,
        n_valid_source_pairs=100,
        operator_vector_series=torch.randn(T, 3),
        operator_axial_vector_series=torch.randn(T, 3),
    )


@pytest.fixture
def glueball_output() -> GlueballColorCorrelatorOutput:
    return GlueballColorCorrelatorOutput(
        correlator=torch.randn(LAG),
        correlator_raw=torch.randn(LAG),
        correlator_connected=torch.randn(LAG),
        counts=torch.ones(LAG, dtype=torch.int64),
        frame_indices=list(range(T)),
        triplet_counts_per_frame=torch.ones(T, dtype=torch.int64),
        disconnected_contribution=0.03,
        mean_glueball=0.4,
        n_valid_source_triplets=30,
        operator_glueball_series=torch.randn(T),
    )


@pytest.fixture
def tensor_output() -> TensorMomentumCorrelatorOutput:
    return TensorMomentumCorrelatorOutput(
        component_labels=(
            "q_xy",
            "q_xz",
            "q_yz",
            "q_xx_minus_yy",
            "q_2zz_minus_xx_minus_yy",
        ),
        frame_indices=list(range(T)),
        pair_selection="distance",
        component_series=torch.randn(T, 5),
        component_counts_per_frame=torch.ones(T, dtype=torch.int64),
        n_valid_source_pairs=100,
        momentum_modes=torch.arange(N_MODES, dtype=torch.float32),
        momentum_correlator=torch.randn(N_MODES, 5, LAG),
        momentum_correlator_raw=torch.randn(N_MODES, 5, LAG),
        momentum_correlator_connected=torch.randn(N_MODES, 5, LAG),
        momentum_correlator_err=None,
        momentum_contracted_correlator=torch.randn(N_MODES, LAG),
        momentum_contracted_correlator_raw=torch.randn(N_MODES, LAG),
        momentum_contracted_correlator_connected=torch.randn(N_MODES, LAG),
        momentum_contracted_correlator_err=None,
        momentum_operator_cos_series=torch.randn(N_MODES, 5, T),
        momentum_operator_sin_series=torch.randn(N_MODES, 5, T),
        momentum_axis=0,
        momentum_length_scale=1.0,
        momentum_valid_frames=T,
    )


@pytest.fixture
def multiscale_output() -> MultiscaleStrongForceOutput:
    per_scale: dict[str, list[ChannelCorrelatorResult]] = {}
    for ch in ("scalar", "pseudoscalar"):
        per_scale[ch] = [
            ChannelCorrelatorResult(
                channel_name=ch,
                correlator=torch.randn(LAG),
                correlator_err=None,
                effective_mass=torch.randn(LAG - 1),
                mass_fit={},
                series=torch.randn(T),
                n_samples=T,
                dt=1.0,
            )
            for _ in range(S)
        ]
    return MultiscaleStrongForceOutput(
        scales=torch.linspace(0.5, 2.0, S),
        frame_indices=list(range(T)),
        per_scale_results=per_scale,
        best_results={ch: per_scale[ch][0] for ch in per_scale},
        best_scale_index=dict.fromkeys(per_scale, 0),
        series_by_channel={ch: torch.randn(S, T) for ch in per_scale},
        bootstrap_mode_applied="none",
        notes=[],
    )


# ---------------------------------------------------------------------------
# Per-type extractor tests
# ---------------------------------------------------------------------------


class TestExtractMesonPhase:
    def test_connected(self, meson_output):
        corrs, ops = extract_meson_phase(meson_output, use_connected=True)
        assert set(corrs.keys()) == {"pseudoscalar", "scalar"}
        assert set(ops.keys()) == {"pseudoscalar", "scalar"}
        assert torch.equal(corrs["pseudoscalar"], meson_output.pseudoscalar_connected)
        assert torch.equal(corrs["scalar"], meson_output.scalar_connected)
        assert corrs["pseudoscalar"].shape == (LAG,)
        assert ops["pseudoscalar"].shape == (T,)

    def test_raw(self, meson_output):
        corrs, _ = extract_meson_phase(meson_output, use_connected=False)
        assert torch.equal(corrs["pseudoscalar"], meson_output.pseudoscalar_raw)
        assert torch.equal(corrs["scalar"], meson_output.scalar_raw)

    def test_prefix(self, meson_output):
        corrs, ops = extract_meson_phase(meson_output, prefix="meson_")
        assert set(corrs.keys()) == {"meson_pseudoscalar", "meson_scalar"}
        assert set(ops.keys()) == {"meson_pseudoscalar", "meson_scalar"}


class TestExtractBaryon:
    def test_connected(self, baryon_output):
        corrs, ops = extract_baryon(baryon_output, use_connected=True)
        assert set(corrs.keys()) == {"nucleon"}
        assert torch.equal(corrs["nucleon"], baryon_output.correlator_connected)
        assert torch.equal(ops["nucleon"], baryon_output.operator_baryon_series)

    def test_raw(self, baryon_output):
        corrs, _ = extract_baryon(baryon_output, use_connected=False)
        assert torch.equal(corrs["nucleon"], baryon_output.correlator_raw)


class TestExtractVectorMeson:
    def test_connected(self, vector_output):
        corrs, ops = extract_vector_meson(vector_output, use_connected=True)
        assert set(corrs.keys()) == {"vector", "axial_vector"}
        assert torch.equal(corrs["vector"], vector_output.vector_connected)
        assert ops["vector"].shape == (T, 3)

    def test_raw(self, vector_output):
        corrs, _ = extract_vector_meson(vector_output, use_connected=False)
        assert torch.equal(corrs["vector"], vector_output.vector_raw)


class TestExtractGlueball:
    def test_connected(self, glueball_output):
        corrs, ops = extract_glueball(glueball_output, use_connected=True)
        assert set(corrs.keys()) == {"glueball"}
        assert torch.equal(corrs["glueball"], glueball_output.correlator_connected)
        assert ops["glueball"].shape == (T,)


class TestExtractTensorMomentum:
    def test_mode_zero(self, tensor_output):
        corrs, ops = extract_tensor_momentum(
            tensor_output,
            use_connected=True,
            momentum_mode=0,
        )
        assert set(corrs.keys()) == {"tensor"}
        assert corrs["tensor"].shape == (LAG,)
        assert ops["tensor"].shape == (T,)
        # Verify the correlator is the contracted one at mode 0
        expected = tensor_output.momentum_contracted_correlator_connected[0]
        assert torch.equal(corrs["tensor"], expected)

    def test_different_mode(self, tensor_output):
        corrs, _ = extract_tensor_momentum(
            tensor_output,
            momentum_mode=2,
        )
        expected = tensor_output.momentum_contracted_correlator_connected[2]
        assert torch.equal(corrs["tensor"], expected)

    def test_operator_is_positive(self, tensor_output):
        _, ops = extract_tensor_momentum(tensor_output)
        assert (ops["tensor"] >= 0).all()


class TestExtractMultiscale:
    def test_shapes(self, multiscale_output):
        corrs, ops = extract_multiscale(multiscale_output)
        assert set(corrs.keys()) == {"scalar", "pseudoscalar"}
        for key in corrs:
            assert corrs[key].shape == (S, LAG)
            assert ops[key].shape == (S, T)

    def test_prefix(self, multiscale_output):
        corrs, _ = extract_multiscale(multiscale_output, prefix="ms_")
        assert set(corrs.keys()) == {"ms_scalar", "ms_pseudoscalar"}


# ---------------------------------------------------------------------------
# Combiner tests
# ---------------------------------------------------------------------------


class TestCollectCorrelators:
    def test_single_output(self, meson_output):
        result = collect_correlators(meson_output)
        assert isinstance(result, PipelineResult)
        assert set(result.correlators.keys()) == {"pseudoscalar", "scalar"}
        assert set(result.operators.keys()) == {"pseudoscalar", "scalar"}

    def test_multiple_outputs(self, meson_output, baryon_output):
        result = collect_correlators(meson_output, baryon_output)
        expected_keys = {"pseudoscalar", "scalar", "nucleon"}
        assert set(result.correlators.keys()) == expected_keys
        assert set(result.operators.keys()) == expected_keys

    def test_all_types(
        self,
        meson_output,
        baryon_output,
        vector_output,
        glueball_output,
        tensor_output,
        multiscale_output,
    ):
        # Use prefix for multiscale to avoid "scalar" collision with meson
        result = collect_correlators(
            meson_output,
            baryon_output,
            vector_output,
            glueball_output,
            tensor_output,
        )
        assert "pseudoscalar" in result.correlators
        assert "nucleon" in result.correlators
        assert "vector" in result.correlators
        assert "glueball" in result.correlators
        assert "tensor" in result.correlators
        assert "scalar" in result.correlators
        # Multiscale separately with prefix to avoid key collision
        ms_corrs, _ms_ops = extract_multiscale(multiscale_output, prefix="ms_")
        assert "ms_scalar" in ms_corrs

    def test_duplicate_key_raises(self, meson_output):
        with pytest.raises(ValueError, match="Duplicate correlator key"):
            collect_correlators(meson_output, meson_output)

    def test_prefix_avoids_collision(self, meson_output):
        # Two meson outputs with different prefixes should work
        corrs1, _ops1 = extract_meson_phase(meson_output, prefix="a_")
        corrs2, _ops2 = extract_meson_phase(meson_output, prefix="b_")
        assert set(corrs1.keys()).isdisjoint(corrs2.keys())

    def test_unsupported_type(self):
        with pytest.raises(TypeError, match="Unsupported output type"):
            collect_correlators("not_an_output")

    def test_use_connected_false(self, meson_output):
        result = collect_correlators(meson_output, use_connected=False)
        assert torch.equal(
            result.correlators["pseudoscalar"],
            meson_output.pseudoscalar_raw,
        )

    def test_prefix_kwarg(self, meson_output):
        result = collect_correlators(meson_output, prefix="test_")
        assert "test_pseudoscalar" in result.correlators


# ---------------------------------------------------------------------------
# Convenience entry point test
# ---------------------------------------------------------------------------


class TestExtractMassesFromChannels:
    def test_import(self):
        """Verify the convenience function is importable."""
        from fragile.physics.new_channels.mass_extraction_adapter import (
            extract_masses_from_channels,
        )

        assert callable(extract_masses_from_channels)

    def test_importable_from_package(self):
        """Verify re-export from __init__.py."""
        from fragile.physics.new_channels import extract_masses_from_channels

        assert callable(extract_masses_from_channels)
