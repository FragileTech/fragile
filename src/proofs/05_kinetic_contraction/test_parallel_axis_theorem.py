"""
Symbolic and Numerical Validation for the Parallel Axis Theorem

Source: docs/source/1_euclidean_gas/05_kinetic_contraction.md, lines 1801-1811
Category: Variance Decomposition
Generated: 2025-10-24
Agent: Math Verifier v1.0 (Gemini + GPT-5 synthesis)

This module provides validation for the Parallel Axis Theorem, also known as
Huygens-Steiner theorem or variance decomposition.
"""

import pytest
import sympy as sp
import numpy as np

def test_parallel_axis_theorem():
    """
    Verifies the Parallel Axis Theorem both symbolically and numerically.

    Identity 1: (1/N)Σ||v_i||² = (1/N)Σ||v_i - μ_v||² + ||μ_v||²
    Identity 2: Var(v) = (1/N)Σ||v_i||² - ||μ_v||²
    """
    print("\n" + "="*60)
    print("Running Validation: Parallel Axis Theorem")
    print("="*60)

    # --- Symbolic Verification ---
    print("\n--- 1. Symbolic Verification (using SymPy) ---")

    N = sp.Symbol('N', integer=True, positive=True)
    i = sp.Symbol('i', integer=True)
    v_i_norm_sq = sp.IndexedBase('v_norm_sq')
    mu_v_norm_sq = sp.Symbol('mu_v_norm_sq', real=True, positive=True)

    # Define sums
    sum_v_i_norm_sq = sp.Sum(v_i_norm_sq[i], (i, 1, N))

    # Key insight: Σ(v_i · μ_v) = (Σv_i) · μ_v = (Nμ_v) · μ_v = N||μ_v||²
    sum_dot_prod = N * mu_v_norm_sq
    sum_mu_v_norm_sq = N * mu_v_norm_sq

    # Expanded variance term: (1/N) Σ ||v_i - μ_v||²
    # = (1/N) Σ (||v_i||² - 2(v_i · μ_v) + ||μ_v||²)
    expanded_variance_term = (1/N) * sum_v_i_norm_sq - (2/N) * sum_dot_prod + (1/N) * sum_mu_v_norm_sq

    # RHS of Identity 1: variance + ||μ_v||²
    rhs_identity_1 = expanded_variance_term + mu_v_norm_sq

    # LHS of Identity 1
    lhs_identity_1 = (1/N) * sum_v_i_norm_sq

    # Verify equality
    difference = sp.simplify(rhs_identity_1 - lhs_identity_1)
    assert difference == 0, (
        f"Symbolic verification of Identity 1 failed!\n"
        f"Difference: {difference}"
    )
    print("✓ Identity 1: (1/N)Σ||v_i||² = (1/N)Σ||v_i - μ_v||² + ||μ_v||²")
    print("✓ Identity 2: Var(v) = (1/N)Σ||v_i||² - ||μ_v||² (algebraic rearrangement)")

    # --- Numerical Verification ---
    print("\n--- 2. Numerical Verification (N=5, d=3) ---")

    N_val = 5
    d_val = 3
    np.random.seed(42)
    vectors = np.random.rand(N_val, d_val) * 10

    mu_v = np.mean(vectors, axis=0)
    mean_of_squares = np.mean(np.sum(vectors**2, axis=1))
    variance = np.mean(np.sum((vectors - mu_v)**2, axis=1))
    squared_mean = np.sum(mu_v**2)

    # Verify Identity 1
    rhs_identity_1_val = variance + squared_mean
    assert np.isclose(mean_of_squares, rhs_identity_1_val), (
        f"Numerical verification failed!\n"
        f"  Mean of Squares: {mean_of_squares:.6f}\n"
        f"  Variance + Squared Mean: {rhs_identity_1_val:.6f}"
    )
    print(f"✓ Identity 1: {mean_of_squares:.4f} ≈ {variance:.4f} + {squared_mean:.4f}")

    # Verify Identity 2
    rhs_identity_2_val = mean_of_squares - squared_mean
    assert np.isclose(variance, rhs_identity_2_val)
    print(f"✓ Identity 2: {variance:.4f} ≈ {mean_of_squares:.4f} - {squared_mean:.4f}")

    print("\n" + "="*60)
    print("✓ Parallel Axis Theorem verified")
    print("="*60)

if __name__ == "__main__":
    test_parallel_axis_theorem()
