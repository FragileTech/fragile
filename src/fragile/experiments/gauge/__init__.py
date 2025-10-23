"""Gauge symmetry testing package for the Fractal Set framework.

This package provides comprehensive testing infrastructure for validating gauge theory
interpretations of the Fragile Gas algorithm's symmetry structure.

**Purpose:**
Test whether the proposed symmetry redefinitions (using collective fields d'_i, r'_i)
enable local gauge theory interpretation or operate in mean-field regime.

**Framework References:**
- old_docs/source/13_fractal_set_new/00_SUMMARY_SYMMETRY_ANALYSIS.md - Overall summary
- old_docs/source/13_fractal_set_new/04b_executive_summary.md - Exec summary
- old_docs/source/13_fractal_set_new/04c_test_cases.md - Detailed test procedures
- old_docs/source/13_fractal_set_new/03_yang_mills_noether.md - Current gauge structure
- old_docs/source/13_fractal_set_new/04_symmetry_redefinition_viability_analysis.md - Proposed structure

**Module Structure:**

- `observables`: Common measurement utilities (collective fields, statistics, plotting)
- `u1_symmetry`: U(1)_fitness symmetry tests (current vs proposed)
- `su2_symmetry`: SU(2)_weak symmetry tests (current vs proposed)
- `locality_tests`: Tests 1A-1C, 1E (correlation, gradients, perturbation, waves)
- `gauge_covariance`: Test 1D - CRITICAL gauge covariance test
- `regime_comparison`: Mean-field vs local regime comparison, crossover scan

**Quick Start:**

    >>> import torch
    >>> from fragile.experiments.gauge import observables, locality_tests
    >>>
    >>> # Setup test configuration
    >>> positions = torch.randn(1000, 2)
    >>> velocities = torch.randn(1000, 2) * 0.1
    >>> rewards = torch.randn(1000)
    >>> alive = torch.ones(1000, dtype=torch.bool)
    >>> companions = torch.randint(0, 1000, (1000,))
    >>>
    >>> # Compute collective fields in local regime
    >>> fields = observables.compute_collective_fields(
    ...     positions, velocities, rewards, alive, companions, rho=0.05
    ... )
    >>>
    >>> # Run locality tests
    >>> results = locality_tests.run_all_locality_tests(
    ...     positions, velocities, rewards, alive, companions, rho=0.05
    ... )

**Key Features:**

1. **Dual Framework Support**: Every module implements BOTH current and proposed symmetries
2. **ρ-Parameter Integration**: All tests use ρ-localized statistics for local regime
3. **Modular Design**: Each test can be run independently
4. **Comprehensive Documentation**: References to analysis documents throughout
5. **HoloViz Visualization**: All plots use HoloViews/Bokeh

**Critical Test:**

The gauge covariance test (Test 1D in gauge_covariance module) is the definitive
experiment that determines whether the proposed structure supports local gauge theory:

- **If gauge covariant** (Δd' ~ O(α)): Local gauge theory interpretation valid
- **If gauge invariant** (Δd' ≈ 0): Mean-field interpretation applies

**Version:** 1.0.0
**Date:** 2025-10-23
"""

from fragile.experiments.gauge import observables

__all__ = [
    "observables",
]

__version__ = "1.0.0"
