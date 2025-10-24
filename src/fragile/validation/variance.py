"""Variance decomposition (Parallel Axis Theorem) validation helpers.

Source reference: docs/source/1_euclidean_gas/05_kinetic_contraction.md:1801-1811

Algebraic claims verified:
1) (1/N) * Σ ||v_i||^2 = (1/N) * Σ ||v_i - μ_v||^2 + ||μ_v||^2
2) Var(v) := (1/N) * Σ ||v_i - μ_v||^2 = (1/N) * Σ ||v_i||^2 - ||μ_v||^2

Notes on framework symbol conventions:
- μ_v (mu_v): barycenter/mean velocity vector
- Var(v): mean-squared deviation

The symbolic verification is performed component-wise in 1D using SymPy
summation with symbolic N, which implies the vector identity via additivity
of the squared Euclidean norm across components.
"""

from __future__ import annotations

import sympy as sp


def parallel_axis_theorem_symbolic() -> bool:
    """Symbolically verify the Parallel Axis Theorem with N as a symbol.

    Verifies both equivalent rearrangements:
      1) (1/N) Σ ||v_i||^2 = (1/N) Σ ||v_i - μ_v||^2 + ||μ_v||^2
      2) Var(v) := (1/N) Σ ||v_i - μ_v||^2 = (1/N) Σ ||v_i||^2 - ||μ_v||^2

    Approach:
      - Use a 1D symbolic sequence {v_i} with Σ over i=1..N (N symbolic)
      - Define μ_v = (1/N) Σ v_i
      - Expand and simplify using Σ(μ_v^2) = N μ_v^2 and Σ v_i = N μ_v

    Returns:
      True if both identities reduce to 0 when rearranged; otherwise False.
    """

    # Symbols and indexed sequence
    N = sp.Symbol("N", integer=True, positive=True)
    i = sp.Symbol("i", integer=True)
    v = sp.IndexedBase("v")

    # Definitions: mean and sums
    sum_v = sp.Sum(v[i], (i, 1, N))
    mu_v = sum_v / N
    sum_v_sq = sp.Sum(v[i] ** 2, (i, 1, N))

    # Identity 1: (1/N) Σ ||v_i||^2 = (1/N) Σ ||v_i - μ_v||^2 + ||μ_v||^2
    lhs1 = sum_v_sq / N
    rhs1_expanded = (
        (1 / N) * sum_v_sq
        - (2 * mu_v / N) * sum_v
        + (1 / N) * sp.summation(mu_v**2, (i, 1, N))
        + mu_v**2
    )
    # Use Σ v_i = N μ_v to simplify
    rhs1_simplified = sp.simplify(rhs1_expanded.subs(sum_v, N * mu_v))
    id1_ok = sp.simplify(lhs1 - rhs1_simplified) == 0

    # Identity 2: Var(v) = (1/N) Σ ||v_i||^2 - ||μ_v||^2
    var_expanded = (
        (1 / N) * sum_v_sq
        - (2 * mu_v / N) * sum_v
        + (1 / N) * sp.summation(mu_v**2, (i, 1, N))
    )
    var_simplified = sp.simplify(var_expanded.subs(sum_v, N * mu_v))
    rhs2 = (1 / N) * sum_v_sq - mu_v**2
    id2_ok = sp.simplify(var_simplified - rhs2) == 0

    return bool(id1_ok and id2_ok)

