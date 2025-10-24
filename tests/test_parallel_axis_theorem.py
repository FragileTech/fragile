"""
SymPy validation for the Parallel Axis Theorem (variance decomposition).

Source: docs/source/1_euclidean_gas/05_kinetic_contraction.md, lines 1801-1811
Category: Variance Decomposition

Algebraic claims:
1) (1/N) Σ ||v_i||^2 = (1/N) Σ ||v_i - μ_v||^2 + ||μ_v||^2
2) Var(v) := (1/N) Σ ||v_i - μ_v||^2 = (1/N) Σ ||v_i||^2 - ||μ_v||^2
"""

from __future__ import annotations

import numpy as np
import sympy as sp

from fragile.validation.variance import parallel_axis_theorem_symbolic


def test_parallel_axis_theorem_symbolic_equivalence():
    """Symbolically verify both identities with N as a symbol.

    Uses a 1D symbolic sequence {v_i} and the mean μ_v = (1/N) Σ v_i.
    The vector result follows component-wise from Euclidean norm additivity.
    """

    assert parallel_axis_theorem_symbolic() is True

    # Additionally, directly check Identity 2 from first principles
    N = sp.Symbol("N", integer=True, positive=True)
    i = sp.Symbol("i", integer=True)
    v = sp.IndexedBase("v")

    sum_v = sp.Sum(v[i], (i, 1, N))
    mu_v = sum_v / N
    sum_v_sq = sp.Sum(v[i] ** 2, (i, 1, N))

    var_expr = (1 / N) * sp.Sum((v[i] - mu_v) ** 2, (i, 1, N))
    var_expanded = (
        (1 / N) * sum_v_sq
        - (2 * mu_v / N) * sum_v
        + (1 / N) * sp.summation(mu_v**2, (i, 1, N))
    )
    var_simplified = sp.simplify(var_expanded.subs(sum_v, N * mu_v))
    rhs2 = (1 / N) * sum_v_sq - mu_v**2

    # Var(v) = E[||v||^2] - ||E[v]||^2 in 1D; vector case is component-wise
    assert sp.simplify(var_simplified - rhs2) == 0


def test_parallel_axis_theorem_numeric_R3_N5():
    """Concrete numerical test in R^3 with N=5 random vectors.

    Verifies both identities match within numerical precision.
    """

    rng = np.random.default_rng(12345)
    N = 5
    d = 3
    V = rng.standard_normal((N, d))

    # Framework symbols (ascii): mu_v for μ_v
    mu_v = V.mean(axis=0)

    # Identity 1: E[||v||^2] = E[||v - mu||^2] + ||mu||^2
    lhs1 = np.mean(np.sum(V * V, axis=1))
    rhs1 = np.mean(np.sum((V - mu_v) ** 2, axis=1)) + float(np.dot(mu_v, mu_v))
    np.testing.assert_allclose(lhs1, rhs1, rtol=1e-12, atol=1e-12)

    # Identity 2: Var(v) = E[||v||^2] - ||mu||^2
    var_v = np.mean(np.sum((V - mu_v) ** 2, axis=1))
    rhs2 = np.mean(np.sum(V * V, axis=1)) - float(np.dot(mu_v, mu_v))
    np.testing.assert_allclose(var_v, rhs2, rtol=1e-12, atol=1e-12)

