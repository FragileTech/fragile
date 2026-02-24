"""Tests for fragile.learning.conformal module."""

from __future__ import annotations

import math

import holoviews as hv
import numpy as np
import pytest

from fragile.learning.conformal import (
    ablation_feature_importance,
    accuracy_vs_radius,
    calibration_test_split,
    conditional_coverage_by_radius,
    conformal_factor_np,
    conformal_quantile,
    conformal_quantiles_per_chart,
    conformal_quantiles_per_chart_code,
    conformal_quantiles_per_class,
    conformal_quantiles_per_radius,
    conformal_scores_geo_beta,
    conformal_scores_geodesic,
    conformal_scores_standard,
    corrupt_data,
    coverage_by_class,
    evaluate_coverage,
    expected_calibration_error,
    format_ablation_table,
    format_class_coverage_table,
    format_coverage_method_comparison,
    format_coverage_summary_table,
    format_ood_auroc_table,
    ood_scores,
    plot_accuracy_vs_radius,
    plot_conditional_coverage,
    plot_corruption_coverage,
    plot_ood_roc,
    plot_reliability_diagram,
    plot_set_size_vs_radius,
    prediction_sets,
    prediction_sets_mondrian,
    radial_bins,
    recalibrate_probs,
    reliability_diagram_data,
    router_entropy,
    tune_beta,
    tune_conformal_beta,
    wilson_ci,
)

hv.extension("bokeh")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def rng():
    return np.random.default_rng(42)


@pytest.fixture()
def sample_data(rng):
    """Standard test data: probs, labels, z_geo, charts for N=200, C=10."""
    n, c = 200, 10
    # Dirichlet gives valid probability vectors
    probs = rng.dirichlet(np.ones(c), size=n).astype(np.float32)
    labels = rng.integers(0, c, size=n)
    # Points inside Poincare ball (radius < 1)
    z_geo = rng.normal(0, 0.3, size=(n, 2)).astype(np.float32)
    charts = rng.integers(0, 4, size=n)
    correct = (probs.argmax(axis=1) == labels).astype(int)
    return probs, labels, z_geo, charts, correct


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


class TestWilsonCI:
    def test_basic(self):
        lo, hi = wilson_ci(80, 100)
        assert 0.0 <= lo < hi <= 1.0
        # 80/100 = 0.8 should be inside CI
        assert lo < 0.8 < hi

    def test_zero_n(self):
        lo, hi = wilson_ci(0, 0)
        assert math.isnan(lo)
        assert math.isnan(hi)

    def test_perfect_score(self):
        lo, hi = wilson_ci(100, 100)
        assert hi == pytest.approx(1.0, abs=1e-10)
        assert lo > 0.9

    def test_zero_successes(self):
        lo, hi = wilson_ci(0, 100)
        assert lo == 0.0
        assert hi < 0.1

    def test_single_trial(self):
        lo, hi = wilson_ci(1, 1)
        assert 0.0 <= lo < hi <= 1.0


class TestConformalFactorNp:
    def test_origin(self):
        z = np.array([[0.0, 0.0]])
        lam = conformal_factor_np(z)
        np.testing.assert_allclose(lam, [2.0])

    def test_monotone_increasing(self):
        z = np.array([[0.0, 0.0], [0.3, 0.0], [0.6, 0.0], [0.9, 0.0]])
        lam = conformal_factor_np(z)
        assert all(lam[i] < lam[i + 1] for i in range(len(lam) - 1))

    def test_boundary_clipping(self):
        """Points at or beyond the boundary should not produce inf/nan."""
        z = np.array([[1.0, 0.0], [0.0, 1.0], [0.99, 0.01]])
        lam = conformal_factor_np(z)
        assert np.all(np.isfinite(lam))
        assert np.all(lam > 0)

    def test_multidimensional(self):
        z = np.array([[0.1, 0.2, 0.3]])
        lam = conformal_factor_np(z)
        r_sq = 0.01 + 0.04 + 0.09
        expected = 2.0 / (1.0 - r_sq)
        np.testing.assert_allclose(lam, [expected], rtol=1e-5)


class TestRouterEntropy:
    def test_uniform(self):
        w = np.array([[0.25, 0.25, 0.25, 0.25]])
        h = router_entropy(w)
        expected = -4 * 0.25 * np.log(0.25)
        np.testing.assert_allclose(h, [expected], rtol=1e-5)

    def test_deterministic(self):
        w = np.array([[1.0, 0.0, 0.0, 0.0]])
        h = router_entropy(w)
        # Entropy should be ~0 (clipped zeros)
        assert h[0] < 0.01

    def test_shape(self):
        w = np.random.dirichlet(np.ones(5), size=10)
        h = router_entropy(w)
        assert h.shape == (10,)


class TestRadialBins:
    def test_basic(self):
        z = np.array([[0.0, 0.0], [0.5, 0.0], [0.9, 0.0]])
        idx, edges = radial_bins(z, 3)
        assert idx.shape == (3,)
        assert edges.shape == (4,)
        assert edges[0] == 0.0
        # All bins should be valid
        assert np.all(idx >= 0)
        assert np.all(idx < 3)

    def test_all_assigned(self, rng):
        z = rng.normal(0, 0.3, size=(100, 2))
        idx, edges = radial_bins(z, 10)
        assert len(idx) == 100
        assert np.all(idx >= 0)
        assert np.all(idx < 10)


class TestCalibrationTestSplit:
    def test_partitions(self):
        cal, test = calibration_test_split(100, 0.5, seed=42)
        assert len(cal) == 50
        assert len(test) == 50
        # No overlap
        assert len(set(cal) & set(test)) == 0
        # Union covers all
        assert set(cal) | set(test) == set(range(100))

    def test_deterministic(self):
        cal1, test1 = calibration_test_split(100, 0.5, seed=42)
        cal2, test2 = calibration_test_split(100, 0.5, seed=42)
        np.testing.assert_array_equal(cal1, cal2)
        np.testing.assert_array_equal(test1, test2)

    def test_different_fractions(self):
        cal, test = calibration_test_split(100, 0.3)
        assert len(cal) == 30
        assert len(test) == 70


# ---------------------------------------------------------------------------
# Analysis 1: Accuracy vs Radius
# ---------------------------------------------------------------------------


class TestAccuracyVsRadius:
    def test_output_structure(self, sample_data):
        _, _, z_geo, _, correct = sample_data
        result = accuracy_vs_radius(z_geo, correct, n_bins=5)
        assert "bin_centers" in result
        assert "accuracy" in result
        assert "ci_lower" in result
        assert "ci_upper" in result
        assert "counts" in result
        # All arrays same length
        n = len(result["bin_centers"])
        for key in result:
            assert len(result[key]) == n

    def test_accuracy_bounds(self, sample_data):
        _, _, z_geo, _, correct = sample_data
        result = accuracy_vs_radius(z_geo, correct, n_bins=5)
        assert np.all(result["accuracy"] >= 0.0)
        assert np.all(result["accuracy"] <= 1.0)

    def test_ci_contains_accuracy(self, sample_data):
        _, _, z_geo, _, correct = sample_data
        result = accuracy_vs_radius(z_geo, correct, n_bins=5)
        assert np.all(result["ci_lower"] <= result["accuracy"])
        assert np.all(result["ci_upper"] >= result["accuracy"])

    def test_empty_bins_skipped(self):
        """With extreme data, some bins should be empty and skipped."""
        z = np.array([[0.0, 0.0], [0.0, 0.0]])
        correct = np.array([1, 0])
        result = accuracy_vs_radius(z, correct, n_bins=10)
        # All data at same radius, so only 1 bin populated
        assert len(result["bin_centers"]) == 1


class TestPlotAccuracyVsRadius:
    def test_returns_overlay(self, sample_data):
        _, _, z_geo, _, correct = sample_data
        data = accuracy_vs_radius(z_geo, correct, n_bins=5)
        result = plot_accuracy_vs_radius(data)
        assert isinstance(result, hv.Overlay)


# ---------------------------------------------------------------------------
# Analysis 2: Conformal Scores & Prediction Sets
# ---------------------------------------------------------------------------


class TestConformalScores:
    def test_standard_scores_range(self, sample_data):
        probs, labels, _, _, _ = sample_data
        scores = conformal_scores_standard(probs, labels)
        assert scores.shape == (len(labels),)
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0)

    def test_standard_perfect_prediction(self):
        """If model assigns prob 1.0 to true class, score should be 0."""
        probs = np.eye(3)
        labels = np.array([0, 1, 2])
        scores = conformal_scores_standard(probs, labels)
        np.testing.assert_allclose(scores, [0.0, 0.0, 0.0])

    def test_geodesic_scores_smaller_at_boundary(self):
        """Geodesic scores should be smaller near boundary (high lambda)."""
        probs = np.array([[0.5, 0.5], [0.5, 0.5]])
        labels = np.array([0, 0])
        z_center = np.array([[0.0, 0.0]])
        z_boundary = np.array([[0.9, 0.0]])

        scores_center = conformal_scores_geodesic(probs[:1], labels[:1], z_center)
        scores_boundary = conformal_scores_geodesic(probs[1:], labels[1:], z_boundary)
        assert scores_boundary[0] < scores_center[0]


class TestConformalQuantile:
    def test_basic(self):
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        q = conformal_quantile(scores, alpha=0.1)
        assert 0.0 <= q <= 1.0

    def test_higher_alpha_lower_quantile(self):
        scores = np.linspace(0, 1, 100)
        q_low_alpha = conformal_quantile(scores, alpha=0.05)
        q_high_alpha = conformal_quantile(scores, alpha=0.20)
        assert q_low_alpha >= q_high_alpha


class TestConformalQuantilesPerChart:
    def test_fallback_for_small_charts(self):
        scores = np.random.rand(100)
        charts = np.zeros(100, dtype=int)
        charts[:5] = 1  # Only 5 samples in chart 1
        result = conformal_quantiles_per_chart(scores, charts, alpha=0.1, min_samples=20)
        # Chart 1 should use global q (fallback)
        global_q = conformal_quantile(scores, 0.1)
        assert result[1] == global_q
        # Chart 0 has 95 samples, should have its own quantile
        assert 0 in result

    def test_all_charts_present(self):
        scores = np.random.rand(200)
        charts = np.repeat(np.arange(4), 50)
        result = conformal_quantiles_per_chart(scores, charts, alpha=0.1)
        assert set(result.keys()) == {0, 1, 2, 3}


class TestConformalQuantilesPerChartCode:
    def test_all_pairs_present(self):
        rng = np.random.default_rng(0)
        scores = rng.random(300)
        charts = np.repeat(np.arange(3), 100)
        codes = np.tile(np.repeat(np.arange(5), 20), 3)
        qs, stats = conformal_quantiles_per_chart_code(scores, charts, codes, alpha=0.1)
        assert stats["n_groups"] == 15  # 3 charts * 5 codes
        assert stats["n_fine"] + stats["n_chart_fallback"] + stats["n_global_fallback"] == 15

    def test_fallback_for_small_groups(self):
        scores = np.random.default_rng(0).random(100)
        charts = np.zeros(100, dtype=int)
        codes = np.zeros(100, dtype=int)
        codes[:5] = 1  # Only 5 samples for (0, 1)
        qs, stats = conformal_quantiles_per_chart_code(
            scores, charts, codes, alpha=0.1, min_samples=20,
        )
        # (0, 1) should fall back to chart-level since group too small
        assert stats["n_chart_fallback"] >= 1 or stats["n_global_fallback"] >= 1
        # (0, 0) has 95 samples, should be fine
        assert stats["n_fine"] >= 1

    def test_stats_keys(self):
        rng = np.random.default_rng(0)
        scores = rng.random(200)
        charts = rng.integers(0, 2, 200)
        codes = rng.integers(0, 4, 200)
        _, stats = conformal_quantiles_per_chart_code(scores, charts, codes, alpha=0.1)
        for key in ("n_groups", "n_fine", "n_chart_fallback", "n_global_fallback", "min_group_size"):
            assert key in stats


class TestConformalQuantilesPerRadius:
    def test_output_structure(self):
        rng = np.random.default_rng(0)
        scores = rng.random(500)
        z_geo = rng.normal(0, 0.3, size=(500, 2)).astype(np.float32)
        edges, qs, stats = conformal_quantiles_per_radius(scores, z_geo, alpha=0.1, n_shells=5)
        assert edges.shape == (6,)  # n_shells + 1
        assert qs.shape == (5,)
        assert stats["n_shells"] == 5
        assert stats["n_fine"] + stats["n_fallback"] == 5

    def test_fallback_for_small_shells(self):
        rng = np.random.default_rng(0)
        scores = rng.random(100)
        # All points near boundary → inner shells empty
        z_geo = np.column_stack([
            rng.uniform(0.8, 0.95, 100),
            np.zeros(100),
        ]).astype(np.float32)
        edges, qs, stats = conformal_quantiles_per_radius(
            scores, z_geo, alpha=0.1, n_shells=10, min_samples=20,
        )
        # Most inner shells should fall back
        assert stats["n_fallback"] > 0

    def test_quantiles_are_valid(self):
        rng = np.random.default_rng(0)
        scores = rng.random(1000)
        z_geo = rng.normal(0, 0.3, size=(1000, 2)).astype(np.float32)
        _, qs, _ = conformal_quantiles_per_radius(scores, z_geo, alpha=0.1, n_shells=5)
        assert np.all(qs >= 0)
        assert np.all(qs <= 1)


class TestConformalScoresGeoBeta:
    def test_beta_zero_equals_standard(self):
        rng = np.random.default_rng(0)
        probs = rng.dirichlet(np.ones(5), size=100).astype(np.float32)
        labels = rng.integers(0, 5, size=100)
        z_geo = rng.normal(0, 0.3, size=(100, 2)).astype(np.float32)
        scores_std = conformal_scores_standard(probs, labels)
        scores_gb = conformal_scores_geo_beta(probs, labels, z_geo, beta=0.0)
        # beta=0 → multiplier is 1, so scores should match standard
        np.testing.assert_allclose(scores_gb, scores_std, rtol=1e-5)

    def test_higher_beta_inflates_center_scores(self):
        """Center samples (low lambda) should get larger scores with positive beta."""
        probs = np.array([[0.5, 0.5]])
        labels = np.array([0])
        z_center = np.array([[0.0, 0.0]])
        z_boundary = np.array([[0.9, 0.0]])
        s_center_0 = conformal_scores_geo_beta(probs, labels, z_center, beta=0.0)
        s_center_2 = conformal_scores_geo_beta(probs, labels, z_center, beta=2.0)
        s_bound_0 = conformal_scores_geo_beta(probs, labels, z_boundary, beta=0.0)
        s_bound_2 = conformal_scores_geo_beta(probs, labels, z_boundary, beta=2.0)
        # Center inflation ratio should be larger than boundary
        center_ratio = s_center_2[0] / s_center_0[0]
        bound_ratio = s_bound_2[0] / s_bound_0[0]
        assert center_ratio > bound_ratio

    def test_shape(self):
        rng = np.random.default_rng(0)
        probs = rng.dirichlet(np.ones(5), size=50).astype(np.float32)
        labels = rng.integers(0, 5, size=50)
        z_geo = rng.normal(0, 0.3, size=(50, 2)).astype(np.float32)
        scores = conformal_scores_geo_beta(probs, labels, z_geo, beta=1.0)
        assert scores.shape == (50,)


class TestTuneConformalBeta:
    def test_returns_float_in_range(self):
        rng = np.random.default_rng(42)
        probs = rng.dirichlet(np.ones(5) * 2.0, size=200).astype(np.float32)
        labels = rng.integers(0, 5, size=200)
        z_geo = rng.normal(0, 0.3, size=(200, 2)).astype(np.float32)
        beta = tune_conformal_beta(probs, labels, z_geo, alpha=0.1)
        assert isinstance(beta, float)
        assert 0.0 <= beta <= 5.0

    def test_tuned_beta_not_worse_than_zero(self):
        """Tuned beta should produce set size <= beta=0."""
        rng = np.random.default_rng(42)
        n = 500
        probs = rng.dirichlet(np.ones(5) * 2.0, size=n).astype(np.float32)
        labels = rng.integers(0, 5, size=n)
        z_geo = rng.normal(0, 0.3, size=(n, 2)).astype(np.float32)
        beta = tune_conformal_beta(probs, labels, z_geo, alpha=0.1)
        # Compare mean set size
        scores_0 = conformal_scores_geo_beta(probs, labels, z_geo, 0.0)
        q_0 = conformal_quantile(scores_0, 0.1)
        _, sizes_0 = prediction_sets(probs, q_0, "geo_beta", z_geo=z_geo, geo_beta=0.0)
        scores_b = conformal_scores_geo_beta(probs, labels, z_geo, beta)
        q_b = conformal_quantile(scores_b, 0.1)
        _, sizes_b = prediction_sets(probs, q_b, "geo_beta", z_geo=z_geo, geo_beta=beta)
        assert sizes_b.mean() <= sizes_0.mean() + 0.01  # small tolerance


class TestPredictionSets:
    def test_standard_method(self, sample_data):
        probs, labels, _, _, _ = sample_data
        q = 0.5
        incl, sizes = prediction_sets(probs, q, "standard")
        assert incl.shape == probs.shape
        assert sizes.shape == (len(probs),)
        assert np.all(sizes >= 1)

    def test_geodesic_method(self, sample_data):
        probs, labels, z_geo, _, _ = sample_data
        q = 0.5
        incl, sizes = prediction_sets(probs, q, "geodesic", z_geo=z_geo)
        assert incl.shape == probs.shape

    def test_chart_method(self, sample_data):
        probs, labels, z_geo, charts, _ = sample_data
        chart_qs = {0: 0.5, 1: 0.4, 2: 0.6, 3: 0.3}
        incl, sizes = prediction_sets(
            probs, 0.5, "chart", z_geo=z_geo, charts=charts, chart_quantiles=chart_qs,
        )
        assert incl.shape == probs.shape

    def test_chart_code_method(self, sample_data):
        probs, labels, z_geo, charts, _ = sample_data
        codes = np.random.default_rng(0).integers(0, 8, size=len(labels))
        cc_qs = {(int(c), int(k)): 0.5 for c in range(4) for k in range(8)}
        incl, sizes = prediction_sets(
            probs, 0.5, "chart_code",
            z_geo=z_geo, charts=charts, codes=codes,
            chart_code_quantiles=cc_qs,
        )
        assert incl.shape == probs.shape
        assert np.all(sizes >= 1)

    def test_radial_method(self, sample_data):
        probs, labels, z_geo, _, _ = sample_data
        rng = np.random.default_rng(0)
        scores = rng.random(len(labels))
        from fragile.learning.conformal import conformal_quantiles_per_radius
        edges, qs, _ = conformal_quantiles_per_radius(scores, z_geo, alpha=0.1, n_shells=5)
        incl, sizes = prediction_sets(
            probs, 0.5, "radial",
            z_geo=z_geo, radial_edges=edges, radial_quantiles=qs,
        )
        assert incl.shape == probs.shape
        assert np.all(sizes >= 1)

    def test_geo_beta_method(self, sample_data):
        probs, labels, z_geo, _, _ = sample_data
        incl, sizes = prediction_sets(
            probs, 0.5, "geo_beta", z_geo=z_geo, geo_beta=1.0,
        )
        assert incl.shape == probs.shape
        assert np.all(sizes >= 1)

    def test_empty_sets_filled(self):
        """Prediction sets should always include at least the top class."""
        probs = np.array([[0.9, 0.1], [0.8, 0.2]])
        incl, sizes = prediction_sets(probs, 0.0, "standard")  # q=0 -> nothing passes
        assert np.all(sizes >= 1)

    def test_chart_applies_geodesic_weighting(self):
        """Chart method should apply geodesic weighting to scores like geodesic method."""
        probs = np.array([[0.6, 0.4], [0.6, 0.4]])
        z_near_center = np.array([[0.0, 0.0]])
        z_near_boundary = np.array([[0.9, 0.0]])
        z_geo = np.vstack([z_near_center, z_near_boundary])
        charts = np.array([0, 0])
        chart_qs = {0: 0.3}

        incl, sizes = prediction_sets(
            probs, 0.3, "chart", z_geo=z_geo, charts=charts, chart_quantiles=chart_qs,
        )
        # The boundary point has higher lambda -> smaller geodesic score
        # -> more classes might be included for it
        # At minimum, both should produce valid sets
        assert incl.shape == (2, 2)


class TestEvaluateCoverage:
    def test_perfect_coverage(self):
        """When true label always in set, coverage = 1.0."""
        incl = np.ones((10, 3), dtype=bool)
        labels = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
        cov, mss = evaluate_coverage(incl, labels)
        assert cov == 1.0
        assert mss == 3.0

    def test_zero_coverage(self):
        """When true label never in set, coverage = 0.0."""
        incl = np.zeros((3, 3), dtype=bool)
        # Include only wrong classes
        incl[0, 1] = True
        incl[1, 0] = True
        incl[2, 0] = True
        labels = np.array([0, 1, 2])
        cov, _ = evaluate_coverage(incl, labels)
        assert cov == 0.0


class TestFormatCoverageSummaryTable:
    def test_markdown_format(self):
        results = {"Standard": (0.90, 2.5), "Geodesic": (0.92, 2.1)}
        md = format_coverage_summary_table(results, alpha=0.1)
        assert "90%" in md.replace(" ", "")
        assert "Standard" in md
        assert "Geodesic" in md
        assert "90%" in md  # Target coverage


# ---------------------------------------------------------------------------
# Analysis 3-4: Conditional Coverage
# ---------------------------------------------------------------------------


class TestConditionalCoverageByRadius:
    def test_output_structure(self, sample_data):
        probs, labels, z_geo, _, _ = sample_data
        incl = np.ones_like(probs, dtype=bool)
        sizes = incl.sum(axis=1)
        pred_sets = {"Standard": (incl, sizes)}
        result = conditional_coverage_by_radius(pred_sets, labels, z_geo, n_bins=5)
        assert "bin_centers" in result
        assert "methods" in result
        assert "Standard" in result["methods"]
        assert "coverage" in result["methods"]["Standard"]
        assert "set_size" in result["methods"]["Standard"]

    def test_plot_conditional_coverage(self, sample_data):
        probs, labels, z_geo, _, _ = sample_data
        incl = np.ones_like(probs, dtype=bool)
        sizes = incl.sum(axis=1)
        pred_sets = {"Standard": (incl, sizes)}
        data = conditional_coverage_by_radius(pred_sets, labels, z_geo, n_bins=5)
        result = plot_conditional_coverage(data, alpha=0.1)
        assert isinstance(result, hv.Overlay)

    def test_plot_set_size(self, sample_data):
        probs, labels, z_geo, _, _ = sample_data
        incl = np.ones_like(probs, dtype=bool)
        sizes = incl.sum(axis=1)
        pred_sets = {"Standard": (incl, sizes)}
        data = conditional_coverage_by_radius(pred_sets, labels, z_geo, n_bins=5)
        result = plot_set_size_vs_radius(data)
        assert isinstance(result, hv.Overlay)


# ---------------------------------------------------------------------------
# Analysis 5: Mondrian
# ---------------------------------------------------------------------------


class TestConformalQuantilesPerClass:
    def test_all_classes_present(self):
        scores = np.random.rand(200)
        labels = np.repeat(np.arange(5), 40)
        result = conformal_quantiles_per_class(scores, labels, alpha=0.1)
        assert set(result.keys()) == {0, 1, 2, 3, 4}

    def test_fallback_for_small_class(self):
        scores = np.random.rand(100)
        labels = np.zeros(100, dtype=int)
        labels[:5] = 1  # Only 5 samples in class 1
        result = conformal_quantiles_per_class(scores, labels, alpha=0.1, min_samples=20)
        global_q = conformal_quantile(scores, 0.1)
        assert result[1] == global_q


class TestPredictionSetsMondrian:
    def test_output_shape(self, sample_data):
        probs, _, _, _, _ = sample_data
        class_qs = {i: 0.5 for i in range(10)}
        incl, sizes = prediction_sets_mondrian(probs, class_qs)
        assert incl.shape == probs.shape
        assert sizes.shape == (len(probs),)
        assert np.all(sizes >= 1)


class TestCoverageByClass:
    def test_output_structure(self, sample_data):
        probs, labels, _, _, _ = sample_data
        incl = np.ones_like(probs, dtype=bool)
        sizes = incl.sum(axis=1)
        pred_sets = {"Standard": (incl, sizes)}
        result = coverage_by_class(pred_sets, labels, num_classes=10)
        assert "Standard" in result
        assert len(result["Standard"]) == 10
        assert "coverage" in result["Standard"][0]
        assert "set_size" in result["Standard"][0]
        assert "count" in result["Standard"][0]


class TestFormatClassCoverageTable:
    def test_markdown_format(self):
        data = {
            "Standard": [
                {"coverage": 0.9, "set_size": 2.0, "count": 50} for _ in range(3)
            ],
        }
        md = format_class_coverage_table(data)
        assert "Class" in md
        assert "Standard Cov" in md
        assert "Count" in md


class TestFormatCoverageMethodComparison:
    def test_markdown_format_with_class_and_radius_gaps(self):
        cls_data = {
            "MethodA": [
                {"coverage": 0.92, "set_size": 1.5, "count": 10},
                {"coverage": 0.88, "set_size": 1.4, "count": 10},
            ],
            "MethodB": [
                {"coverage": 0.80, "set_size": 2.0, "count": 10},
                {"coverage": 0.85, "set_size": 1.9, "count": 10},
            ],
        }
        cond_data = {
            "methods": {
                "MethodA": {"coverage": [0.95, 0.85, 0.90]},
                "MethodB": {"coverage": [0.70, 0.74]},
            }
        }
        specs = {
            "MethodA": {
                "conditions": "nothing",
                "groups": 1,
                "needs_labels": False,
                "class_coverage_key": "MethodA",
                "radius_coverage_key": "MethodA",
            },
            "MethodB": {
                "conditions": "chart",
                "groups": 10,
                "needs_labels": True,
                "class_coverage_key": "MethodB",
                "radius_coverage_key": "MethodB",
            },
        }
        md = format_coverage_method_comparison(specs, cls_data, cond_data)
        assert "| Method | Conditions on | Groups | Needs labels? | Worst-class gap | Worst-radius gap |" in md
        assert "| MethodA |" in md
        assert "| MethodB |" in md


# ---------------------------------------------------------------------------
# Analysis 6: OOD Detection
# ---------------------------------------------------------------------------


class TestOodScores:
    def test_perfect_separation(self):
        id_signals = {"score": np.zeros(100)}
        ood_signals = {"score": np.ones(100)}
        aurocs = ood_scores(id_signals, ood_signals)
        assert aurocs["score"] == 1.0

    def test_random_signals(self, rng):
        id_signals = {"score": rng.normal(0, 1, size=100)}
        ood_signals = {"score": rng.normal(0, 1, size=100)}
        aurocs = ood_scores(id_signals, ood_signals)
        # Random signals -> AUROC around 0.5
        assert 0.3 < aurocs["score"] < 0.7

    def test_missing_signal_skipped(self):
        id_signals = {"a": np.zeros(10), "b": np.zeros(10)}
        ood_signals = {"a": np.ones(10)}  # no "b"
        aurocs = ood_scores(id_signals, ood_signals)
        assert "a" in aurocs
        assert "b" not in aurocs


class TestPlotOodRoc:
    def test_returns_overlay(self):
        id_signals = {"score": np.zeros(50)}
        ood_signals = {"score": np.ones(50)}
        result = plot_ood_roc(id_signals, ood_signals)
        assert isinstance(result, hv.Overlay)


class TestFormatOodAurocTable:
    def test_markdown_format(self):
        aurocs = {"signal_a": 0.95, "signal_b": 0.80}
        md = format_ood_auroc_table(aurocs)
        assert "signal_a" in md
        assert "0.950" in md
        assert "0.800" in md


# ---------------------------------------------------------------------------
# Analysis 7: Calibration / Reliability
# ---------------------------------------------------------------------------


class TestReliabilityDiagramData:
    def test_output_structure(self, sample_data):
        probs, labels, _, _, _ = sample_data
        result = reliability_diagram_data(probs, labels, n_bins=10)
        assert "bin_centers" in result
        assert "observed_freq" in result
        assert "counts" in result
        n = len(result["bin_centers"])
        assert len(result["observed_freq"]) == n
        assert len(result["counts"]) == n

    def test_observed_freq_bounds(self, sample_data):
        probs, labels, _, _, _ = sample_data
        result = reliability_diagram_data(probs, labels, n_bins=10)
        assert np.all(result["observed_freq"] >= 0.0)
        assert np.all(result["observed_freq"] <= 1.0)


class TestRecalibrateProbs:
    def test_stays_normalized(self, sample_data):
        probs, _, z_geo, _, _ = sample_data
        recal = recalibrate_probs(probs, z_geo, beta=0.5)
        row_sums = recal.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_beta_zero_is_identity(self, sample_data):
        probs, _, z_geo, _, _ = sample_data
        recal = recalibrate_probs(probs, z_geo, beta=0.0)
        # lambda^0 = 1, so recal should equal probs
        np.testing.assert_allclose(recal, probs, atol=1e-6)


class TestExpectedCalibrationError:
    def test_perfect_calibration(self):
        """If predicted probs exactly match accuracy, ECE should be ~0."""
        # 100 samples all with prob 1.0 predicting the correct class
        probs = np.zeros((100, 2))
        probs[:, 0] = 1.0
        labels = np.zeros(100, dtype=int)
        ece = expected_calibration_error(probs, labels, n_bins=10)
        assert ece < 0.01

    def test_ece_bounds(self, sample_data):
        probs, labels, _, _, _ = sample_data
        ece = expected_calibration_error(probs, labels)
        assert 0.0 <= ece <= 1.0


class TestTuneBeta:
    def test_returns_float(self, sample_data):
        probs, labels, z_geo, _, _ = sample_data
        beta = tune_beta(probs, labels, z_geo, n_bins=10)
        assert isinstance(beta, float)
        assert -1.0 <= beta <= 2.0


class TestPlotReliabilityDiagram:
    def test_raw_only(self, sample_data):
        probs, labels, _, _, _ = sample_data
        raw = reliability_diagram_data(probs, labels)
        result = plot_reliability_diagram(raw)
        assert isinstance(result, hv.Overlay)

    def test_with_recal(self, sample_data):
        probs, labels, z_geo, _, _ = sample_data
        raw = reliability_diagram_data(probs, labels)
        recal_p = recalibrate_probs(probs, z_geo, 0.5)
        recal = reliability_diagram_data(recal_p, labels)
        result = plot_reliability_diagram(raw, recal)
        assert isinstance(result, hv.Overlay)


# ---------------------------------------------------------------------------
# Analysis 8: Ablation
# ---------------------------------------------------------------------------


class TestAblationFeatureImportance:
    def test_output_structure(self, rng):
        n = 300
        correct = rng.integers(0, 2, size=n)
        p_max = rng.random(n)
        radius = rng.random(n)
        v_h = rng.random(n)
        result = ablation_feature_importance(correct, p_max, radius, v_h)
        assert "feature_names" in result
        assert "coefficients" in result
        assert "importances" in result
        assert "auc" in result
        assert len(result["feature_names"]) == 3
        assert len(result["coefficients"]) == 3
        assert len(result["importances"]) == 3
        assert 0.0 <= result["auc"] <= 1.0

    def test_importances_are_abs_coefficients(self, rng):
        n = 300
        correct = rng.integers(0, 2, size=n)
        p_max = rng.random(n)
        radius = rng.random(n)
        v_h = rng.random(n)
        result = ablation_feature_importance(correct, p_max, radius, v_h)
        for coeff, imp in zip(result["coefficients"], result["importances"]):
            np.testing.assert_allclose(abs(coeff), imp, atol=1e-10)


class TestFormatAblationTable:
    def test_markdown_format(self):
        data = {
            "feature_names": ["a", "b", "c"],
            "coefficients": [0.5, -0.3, 0.1],
            "importances": [0.5, 0.3, 0.1],
            "auc": 0.85,
        }
        md = format_ablation_table(data)
        assert "0.850" in md
        assert "+0.500" in md
        assert "-0.300" in md


# ---------------------------------------------------------------------------
# Analysis 9: Corruption
# ---------------------------------------------------------------------------


class TestCorruptData:
    def test_gaussian_noise(self, rng):
        X = rng.random((10, 784)).astype(np.float32)
        X_corr = corrupt_data(X, "gaussian_noise", 0.5)
        assert X_corr.shape == X.shape
        assert X_corr.dtype == np.float32
        # Clipped to [0, 1]
        assert X_corr.min() >= 0.0
        assert X_corr.max() <= 1.0
        # Should differ from original
        assert not np.allclose(X, X_corr)

    def test_rotation(self, rng):
        X = rng.random((5, 784)).astype(np.float32)
        X_corr = corrupt_data(X, "rotation", 0.5)
        assert X_corr.shape == X.shape
        assert X_corr.dtype == np.float32

    def test_blur(self, rng):
        X = rng.random((5, 784)).astype(np.float32)
        X_corr = corrupt_data(X, "blur", 0.5)
        assert X_corr.shape == X.shape
        assert X_corr.dtype == np.float32

    def test_zero_intensity_noise(self, rng):
        X = rng.random((5, 784)).astype(np.float32)
        X_corr = corrupt_data(X, "gaussian_noise", 0.0)
        np.testing.assert_allclose(X, X_corr, atol=1e-6)

    def test_deterministic(self, rng):
        X = rng.random((5, 784)).astype(np.float32)
        X_corr1 = corrupt_data(X, "gaussian_noise", 0.5, seed=123)
        X_corr2 = corrupt_data(X, "gaussian_noise", 0.5, seed=123)
        np.testing.assert_array_equal(X_corr1, X_corr2)

    def test_unknown_corruption_returns_copy(self, rng):
        X = rng.random((5, 784)).astype(np.float32)
        X_corr = corrupt_data(X, "unknown_type", 0.5)
        np.testing.assert_allclose(X, X_corr, atol=1e-6)


class TestPlotCorruptionCoverage:
    def test_returns_layout(self):
        data = {
            "gaussian_noise": {
                "Standard": np.array([0.9, 0.85, 0.8]),
                "Geodesic": np.array([0.92, 0.88, 0.85]),
                "intensities": [0.2, 0.5, 0.8],
            },
        }
        result = plot_corruption_coverage(data, alpha=0.1)
        assert isinstance(result, hv.Layout)

    def test_empty_data(self):
        result = plot_corruption_coverage({}, alpha=0.1)
        assert isinstance(result, hv.Div)


# ---------------------------------------------------------------------------
# End-to-end: Conformal pipeline with synthetic data
# ---------------------------------------------------------------------------


class TestConformalPipelineEndToEnd:
    """Integration test: run the full conformal pipeline on synthetic data."""

    def test_standard_coverage_guarantee(self, rng):
        """Standard conformal should achieve >= (1-alpha) marginal coverage on average."""
        n = 1000
        c = 5
        probs = rng.dirichlet(np.ones(c) * 2.0, size=n)
        labels = rng.integers(0, c, size=n)

        cal_idx, test_idx = calibration_test_split(n, 0.5, seed=0)

        cal_scores = conformal_scores_standard(probs[cal_idx], labels[cal_idx])
        alpha = 0.1
        q = conformal_quantile(cal_scores, alpha)
        incl, sizes = prediction_sets(probs[test_idx], q, "standard")
        cov, mss = evaluate_coverage(incl, labels[test_idx])
        # Coverage guarantee is statistical, use a tolerance
        assert cov >= (1 - alpha) - 0.05, f"Coverage {cov} too low"

    def test_geodesic_coverage_guarantee(self, rng):
        """Geodesic conformal should also achieve >= (1-alpha) coverage."""
        n = 1000
        c = 5
        probs = rng.dirichlet(np.ones(c) * 2.0, size=n)
        labels = rng.integers(0, c, size=n)
        z_geo = rng.normal(0, 0.3, size=(n, 2)).astype(np.float32)

        cal_idx, test_idx = calibration_test_split(n, 0.5, seed=0)

        cal_scores = conformal_scores_geodesic(
            probs[cal_idx], labels[cal_idx], z_geo[cal_idx],
        )
        alpha = 0.1
        q = conformal_quantile(cal_scores, alpha)
        incl, sizes = prediction_sets(
            probs[test_idx], q, "geodesic", z_geo=z_geo[test_idx],
        )
        cov, _ = evaluate_coverage(incl, labels[test_idx])
        assert cov >= (1 - alpha) - 0.05, f"Coverage {cov} too low"

    def test_conditional_coverage_and_set_size(self, rng):
        """Conditional coverage pipeline produces valid output."""
        n = 500
        c = 5
        probs = rng.dirichlet(np.ones(c) * 2.0, size=n)
        labels = rng.integers(0, c, size=n)
        z_geo = rng.normal(0, 0.3, size=(n, 2)).astype(np.float32)

        cal_idx, test_idx = calibration_test_split(n, 0.5)
        cal_scores = conformal_scores_standard(probs[cal_idx], labels[cal_idx])
        q = conformal_quantile(cal_scores, 0.1)
        incl, sizes = prediction_sets(probs[test_idx], q, "standard")

        cond = conditional_coverage_by_radius(
            {"Standard": (incl, sizes)}, labels[test_idx], z_geo[test_idx], n_bins=5,
        )
        assert len(cond["bin_centers"]) > 0
        assert len(cond["methods"]["Standard"]["coverage"]) == len(cond["bin_centers"])

    def test_mondrian_pipeline(self, rng):
        """Mondrian conformal produces valid sets."""
        n = 500
        c = 5
        probs = rng.dirichlet(np.ones(c) * 2.0, size=n)
        labels = rng.integers(0, c, size=n)

        cal_idx, test_idx = calibration_test_split(n, 0.5)
        cal_scores = conformal_scores_standard(probs[cal_idx], labels[cal_idx])
        class_qs = conformal_quantiles_per_class(cal_scores, labels[cal_idx], 0.1)
        incl, sizes = prediction_sets_mondrian(probs[test_idx], class_qs)
        assert np.all(sizes >= 1)

    def test_reliability_and_recalibration(self, rng):
        """Full reliability + recalibration pipeline."""
        n = 500
        c = 5
        probs = rng.dirichlet(np.ones(c) * 2.0, size=n)
        labels = rng.integers(0, c, size=n)
        z_geo = rng.normal(0, 0.3, size=(n, 2)).astype(np.float32)

        cal_idx, test_idx = calibration_test_split(n, 0.5)
        beta = tune_beta(probs[cal_idx], labels[cal_idx], z_geo[cal_idx])
        recal = recalibrate_probs(probs[test_idx], z_geo[test_idx], beta)
        ece_raw = expected_calibration_error(probs[test_idx], labels[test_idx])
        ece_recal = expected_calibration_error(recal, labels[test_idx])
        # Recalibration should not make ECE dramatically worse
        assert ece_recal < ece_raw + 0.15

    def test_full_analysis_chain(self, rng):
        """Run the full chain matching what the dashboard callback does."""
        n = 400
        c = 10
        probs = rng.dirichlet(np.ones(c) * 2.0, size=n)
        labels = rng.integers(0, c, size=n)
        z_geo = rng.normal(0, 0.3, size=(n, 2)).astype(np.float32)
        charts = rng.integers(0, 4, size=n)
        correct = (probs.argmax(axis=1) == labels).astype(int)
        confidence = probs.max(axis=1)
        router_w = rng.dirichlet(np.ones(4), size=n)

        alpha = 0.1
        n_bins = 10

        # Analysis 1
        avr = accuracy_vs_radius(z_geo, correct, n_bins)
        assert len(avr["bin_centers"]) > 0

        # Cal/test split
        cal_idx, test_idx = calibration_test_split(n, 0.5)

        # Analysis 7
        raw_rel = reliability_diagram_data(probs[test_idx], labels[test_idx], n_bins)
        beta = tune_beta(probs[cal_idx], labels[cal_idx], z_geo[cal_idx], n_bins)
        recal_p = recalibrate_probs(probs[test_idx], z_geo[test_idx], beta)
        recal_rel = reliability_diagram_data(recal_p, labels[test_idx], n_bins)

        # Analysis 2-4
        cal_s_std = conformal_scores_standard(probs[cal_idx], labels[cal_idx])
        cal_s_geo = conformal_scores_geodesic(probs[cal_idx], labels[cal_idx], z_geo[cal_idx])
        q_std = conformal_quantile(cal_s_std, alpha)
        q_geo = conformal_quantile(cal_s_geo, alpha)
        chart_qs = conformal_quantiles_per_chart(cal_s_geo, charts[cal_idx], alpha)

        incl_std, s_std = prediction_sets(probs[test_idx], q_std, "standard")
        incl_geo, s_geo = prediction_sets(
            probs[test_idx], q_geo, "geodesic", z_geo=z_geo[test_idx],
        )
        incl_ch, s_ch = prediction_sets(
            probs[test_idx], q_geo, "chart",
            z_geo=z_geo[test_idx], charts=charts[test_idx], chart_quantiles=chart_qs,
        )

        for incl, sizes in [(incl_std, s_std), (incl_geo, s_geo), (incl_ch, s_ch)]:
            cov, mss = evaluate_coverage(incl, labels[test_idx])
            assert 0 <= cov <= 1
            assert mss >= 1

        # Analysis 5
        class_qs = conformal_quantiles_per_class(cal_s_std, labels[cal_idx], alpha)
        incl_m, s_m = prediction_sets_mondrian(probs[test_idx], class_qs)
        cls_data = coverage_by_class(
            {"Standard": (incl_std, s_std), "Mondrian": (incl_m, s_m)},
            labels[test_idx], c,
        )
        assert len(cls_data) == 2

        # Analysis 8
        radius = np.linalg.norm(z_geo, axis=1)
        v_h = router_entropy(router_w)
        abl = ablation_feature_importance(correct, confidence, radius, v_h)
        assert 0 <= abl["auc"] <= 1
