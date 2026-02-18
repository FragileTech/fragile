"""Tests for source-pair propagation correlators."""

from __future__ import annotations

import pytest
import torch

from fragile.physics.operators.config import (
    BaryonOperatorConfig,
    GlueballOperatorConfig,
    MesonOperatorConfig,
    VectorOperatorConfig,
)
from fragile.physics.operators.propagator_correlators import (
    compute_baryon_propagator,
    compute_glueball_propagator,
    compute_meson_propagator,
    compute_propagator_pipeline,
    compute_vector_propagator,
    PropagatorChannelResult,
    PropagatorResult,
)

from .conftest import make_prepared_data


MAX_LAG = 5


# ---------------------------------------------------------------------------
# Meson propagator tests
# ---------------------------------------------------------------------------


class TestMesonPropagator:
    def test_shape(self):
        data = make_prepared_data(T=10, N=20)
        result = compute_meson_propagator(data, MesonOperatorConfig(), max_lag=MAX_LAG)
        assert "meson_scalar" in result
        assert "meson_pseudoscalar" in result
        assert result["meson_scalar"].raw.shape == (MAX_LAG + 1,)
        assert result["meson_pseudoscalar"].raw.shape == (MAX_LAG + 1,)
        assert result["meson_scalar"].counts.shape == (MAX_LAG + 1,)

    def test_lag0_positive(self):
        data = make_prepared_data(T=10, N=20)
        result = compute_meson_propagator(data, MesonOperatorConfig(), max_lag=MAX_LAG)
        # Raw auto-correlation at zero lag should be positive (sum of squares)
        assert result["meson_scalar"].raw[0].item() > 0

    def test_connected_vs_raw(self):
        data = make_prepared_data(T=10, N=20)
        result = compute_meson_propagator(
            data,
            MesonOperatorConfig(),
            max_lag=MAX_LAG,
            use_connected=True,
        )
        r = result["meson_scalar"]
        # Connected subtracts mean^2, so connected <= raw at lag 0
        assert r.connected[0].item() <= r.raw[0].item() + 1e-6

    def test_score_directed(self):
        data = make_prepared_data(T=10, N=20, include_scores=True)
        cfg = MesonOperatorConfig(operator_mode="score_directed")
        result = compute_meson_propagator(data, cfg, max_lag=MAX_LAG)
        assert result["meson_scalar"].raw.shape == (MAX_LAG + 1,)

    def test_empty_data(self):
        data = make_prepared_data(T=0, N=5)
        result = compute_meson_propagator(data, MesonOperatorConfig(), max_lag=MAX_LAG)
        assert result["meson_scalar"].raw.shape == (MAX_LAG + 1,)
        assert result["meson_scalar"].raw.abs().sum().item() == 0.0


# ---------------------------------------------------------------------------
# Vector propagator tests
# ---------------------------------------------------------------------------


class TestVectorPropagator:
    def test_shape(self):
        data = make_prepared_data(T=10, N=20, include_positions=True)
        result = compute_vector_propagator(data, VectorOperatorConfig(), max_lag=MAX_LAG)
        assert "vector_full" in result
        assert "axial_full" in result
        assert result["vector_full"].raw.shape == (MAX_LAG + 1,)
        assert result["axial_full"].raw.shape == (MAX_LAG + 1,)

    def test_dot_product_lag0_positive(self):
        data = make_prepared_data(T=10, N=20, include_positions=True)
        result = compute_vector_propagator(data, VectorOperatorConfig(), max_lag=MAX_LAG)
        # Dot product of vector with itself at lag=0 should be non-negative
        assert result["vector_full"].raw[0].item() >= -1e-6

    def test_empty_data(self):
        data = make_prepared_data(T=0, N=5, include_positions=True)
        result = compute_vector_propagator(data, VectorOperatorConfig(), max_lag=MAX_LAG)
        assert result["vector_full"].raw.shape == (MAX_LAG + 1,)


# ---------------------------------------------------------------------------
# Baryon propagator tests
# ---------------------------------------------------------------------------


class TestBaryonPropagator:
    def test_shape(self):
        data = make_prepared_data(T=10, N=20)
        result = compute_baryon_propagator(data, BaryonOperatorConfig(), max_lag=MAX_LAG)
        assert "baryon_nucleon" in result
        assert result["baryon_nucleon"].raw.shape == (MAX_LAG + 1,)

    def test_det_abs_nonnegative(self):
        data = make_prepared_data(T=10, N=20)
        cfg = BaryonOperatorConfig(operator_mode="det_abs")
        result = compute_baryon_propagator(data, cfg, max_lag=MAX_LAG)
        # |det| * |det| is non-negative, so raw C(0) >= 0
        assert result["baryon_nucleon"].raw[0].item() >= -1e-6

    def test_flux_modes(self):
        data = make_prepared_data(T=10, N=20)
        for mode in ("flux_action", "flux_sin2", "flux_exp"):
            cfg = BaryonOperatorConfig(operator_mode=mode)
            result = compute_baryon_propagator(data, cfg, max_lag=MAX_LAG)
            assert result["baryon_nucleon"].raw.shape == (MAX_LAG + 1,)

    def test_empty_data(self):
        data = make_prepared_data(T=0, N=5)
        result = compute_baryon_propagator(data, BaryonOperatorConfig(), max_lag=MAX_LAG)
        assert result["baryon_nucleon"].raw.shape == (MAX_LAG + 1,)


# ---------------------------------------------------------------------------
# Glueball propagator tests
# ---------------------------------------------------------------------------


class TestGlueballPropagator:
    def test_shape(self):
        data = make_prepared_data(T=10, N=20)
        result = compute_glueball_propagator(data, GlueballOperatorConfig(), max_lag=MAX_LAG)
        assert "glueball_plaquette" in result
        assert result["glueball_plaquette"].raw.shape == (MAX_LAG + 1,)

    def test_modes(self):
        data = make_prepared_data(T=10, N=20)
        for mode in ("re_plaquette", "action_re_plaquette", "phase_action", "phase_sin2"):
            cfg = GlueballOperatorConfig(operator_mode=mode)
            result = compute_glueball_propagator(data, cfg, max_lag=MAX_LAG)
            assert result["glueball_plaquette"].raw.shape == (MAX_LAG + 1,)

    def test_empty_data(self):
        data = make_prepared_data(T=0, N=5)
        result = compute_glueball_propagator(data, GlueballOperatorConfig(), max_lag=MAX_LAG)
        assert result["glueball_plaquette"].raw.shape == (MAX_LAG + 1,)


# ---------------------------------------------------------------------------
# Pipeline tests
# ---------------------------------------------------------------------------


class TestPropagatorPipeline:
    def test_all_channels(self):
        data = make_prepared_data(T=10, N=20, include_positions=True, include_scores=True)
        result = compute_propagator_pipeline(data, max_lag=MAX_LAG)
        assert isinstance(result, PropagatorResult)
        assert "meson_scalar" in result.channels
        assert "meson_pseudoscalar" in result.channels
        assert "vector_full" in result.channels
        assert "axial_full" in result.channels
        assert "baryon_nucleon" in result.channels
        assert "glueball_plaquette" in result.channels
        assert result.prepared_data is data

    def test_single_channel(self):
        data = make_prepared_data(T=10, N=20)
        result = compute_propagator_pipeline(data, channels=["meson"], max_lag=MAX_LAG)
        assert "meson_scalar" in result.channels
        assert "meson_pseudoscalar" in result.channels
        assert "baryon_nucleon" not in result.channels

    def test_counts_decrease(self):
        data = make_prepared_data(T=10, N=20)
        result = compute_propagator_pipeline(data, channels=["meson"], max_lag=MAX_LAG)
        counts = result.channels["meson_scalar"].counts
        # Counts should be non-increasing with lag
        for i in range(len(counts) - 1):
            assert counts[i].item() >= counts[i + 1].item()

    def test_empty_data(self):
        data = make_prepared_data(T=0, N=5)
        result = compute_propagator_pipeline(data, channels=["meson"], max_lag=MAX_LAG)
        assert result.channels["meson_scalar"].raw.abs().sum().item() == 0.0
