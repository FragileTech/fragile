"""
Demonstration of Symbolic Geometry Algebra.

This script shows how to use the fitness_algebra module to:
1. Create symbolic expressions for geometric quantities
2. Verify formulas from Chapter 9
3. Export to LaTeX
4. Generate numerical functions

Note: Full 3D computation with multiple walkers is computationally expensive.
This demo uses simplified cases for clarity.

Usage:
    python examples/symbolic_geometry_demo.py
"""

import sympy as sp
from sympy import symbols, exp, sqrt, simplify, expand, Matrix, pprint, init_printing

from fragile.fitness_algebra import (
    FitnessPotential,
    EmergentMetric,
    RescaleFunction,
    MeasurementFunction,
    create_algorithmic_parameters,
    export_to_latex,
)

# Enable pretty printing
init_printing(use_unicode=True)


def demo_1_rescale_function():
    """Demo 1: Verify sigmoid rescale function (Chapter 9.7.1)."""
    print("=" * 70)
    print("DEMO 1: Sigmoid Rescale Function")
    print("=" * 70)

    z = symbols('z', real=True)
    A = symbols('A', positive=True, real=True)

    # Sigmoid
    g_A = RescaleFunction.sigmoid(z, A)
    print("\nSigmoid rescale: g_A(z) = A / (1 + exp(-z))")
    print(f"\nSymbolic form:")
    pprint(g_A)

    # First derivative
    g_A_prime = RescaleFunction.sigmoid_derivative(z, A, order=1)
    print(f"\nFirst derivative g'_A(z):")
    pprint(simplify(g_A_prime))

    # Second derivative
    g_A_double_prime = RescaleFunction.sigmoid_derivative(z, A, order=2)
    print(f"\nSecond derivative g''_A(z):")
    pprint(simplify(g_A_double_prime))

    # Verify bounds
    print(f"\nBounds:")
    print(f"  g_A(0) = {g_A.subs(z, 0)} = A/2")
    print(f"  lim_(z→∞) g_A(z) = {g_A.limit(z, sp.oo)} = A")
    print(f"  lim_(z→-∞) g_A(z) = {g_A.limit(z, -sp.oo)} = 0")

    # Maximum of first derivative
    print(f"\nMaximum of g'_A(z):")
    print(f"  At z=0: g'_A(0) = {simplify(g_A_prime.subs(z, 0))} = A/4")

    return {'g_A': g_A, 'g_A_prime': g_A_prime, 'g_A_double_prime': g_A_double_prime}


def demo_2_localization_weights():
    """Demo 2: Localization weights (Chapter 9.2)."""
    print("\n" + "=" * 70)
    print("DEMO 2: Localization Weights")
    print("=" * 70)

    # Simple 2D case with 2 walkers
    print("\n2D state space with 2 walkers")

    fitness = FitnessPotential(dim=2, num_walkers=2)

    print("\nComputing localization weights w_j(ρ)...")
    weights = fitness.localization_weights()

    print(f"\nNumber of weights: {len(weights)}")
    print("\nWeight w_1 (truncated):")
    print(str(weights[0])[:200] + "...")

    # Verify normalization
    print("\nVerifying normalization: Σ_j w_j = 1")
    weight_sum = simplify(sum(weights))
    print(f"  Σ_j w_j = {weight_sum}")
    assert weight_sum == 1, "Weights should sum to 1!"
    print("  ✓ Normalized correctly")

    return {'fitness': fitness, 'weights': weights}


def demo_3_metric_tensor_2d():
    """Demo 3: Metric tensor in 2D (Chapter 9.4-9.5)."""
    print("\n" + "=" * 70)
    print("DEMO 3: Metric Tensor (2D Simplified)")
    print("=" * 70)

    # Create a symbolic Hessian
    h11, h12, h22 = symbols('h11 h12 h22', real=True)
    H = Matrix([[h11, h12],
                [h12, h22]])

    print("\nTest Hessian H:")
    pprint(H)

    # Create metric
    fitness = FitnessPotential(dim=2, num_walkers=2)
    metric_obj = EmergentMetric(fitness)

    epsilon_Sigma = fitness.params['epsilon_Sigma']

    g = metric_obj.metric_tensor(H)
    print(f"\nMetric tensor g = H + ε_Σ I:")
    pprint(g)

    # Determinant
    det_g = g.det()
    print(f"\nDeterminant det(g):")
    det_expanded = simplify(expand(det_g))
    pprint(det_expanded)

    # Volume element
    vol = sqrt(det_g)
    print(f"\nVolume element √det(g):")
    vol_simplified = simplify(vol)
    pprint(vol_simplified)

    # Inverse metric
    g_inv = g.inv()
    print(f"\nInverse metric g⁻¹ (diffusion tensor):")
    pprint(simplify(g_inv))

    return {'H': H, 'g': g, 'det_g': det_expanded, 'vol': vol_simplified}


def demo_4_volume_element_3d():
    """Demo 4: Volume element with explicit 3D formula (Chapter 9.7.6)."""
    print("\n" + "=" * 70)
    print("DEMO 4: Volume Element (3D Explicit Formula)")
    print("=" * 70)

    # Create symbolic Hessian components
    h11, h12, h13 = symbols('h11 h12 h13', real=True)
    h22, h23, h33 = symbols('h22 h23 h33', real=True)

    H = Matrix([[h11, h12, h13],
                [h12, h22, h23],
                [h13, h23, h33]])

    print("\n3×3 Hessian H:")
    pprint(H)

    fitness = FitnessPotential(dim=3, num_walkers=2)
    metric_obj = EmergentMetric(fitness)
    epsilon_Sigma = fitness.params['epsilon_Sigma']

    # Use explicit formula
    print("\nUsing explicit formula from Chapter 9.7.6:")
    print("det(g) = det(H) + ε_Σ·tr(adj(H)) + ε_Σ²·tr(H) + ε_Σ³")

    vol = metric_obj.volume_element_3d_explicit(H)

    print("\nVolume element √det(g):")
    print("(Showing structure, not fully expanded)")

    # Show determinant components
    det_H = H.det()
    tr_H = H.trace()

    print(f"\ndet(H) has {len(str(det_H))} characters")
    print(f"tr(H) = {tr_H}")

    # Specific case: diagonal Hessian
    print("\nSpecial case: Diagonal Hessian (h_ij = 0 for i≠j)")
    H_diag = Matrix([[h11, 0, 0],
                     [0, h22, 0],
                     [0, 0, h33]])

    vol_diag = metric_obj.volume_element_3d_explicit(H_diag)
    vol_diag_simplified = simplify(vol_diag)

    print("Volume element for diagonal H:")
    pprint(vol_diag_simplified)

    # This should be sqrt((h11+ε)(h22+ε)(h33+ε))
    expected = sqrt((h11 + epsilon_Sigma) * (h22 + epsilon_Sigma) * (h33 + epsilon_Sigma))
    print("\nExpected (diagonal case):")
    pprint(expected)

    print("\nVerifying they match:")
    diff = simplify(vol_diag_simplified - expected)
    print(f"  Difference: {diff}")
    if diff == 0:
        print("  ✓ Verified!")

    return {'H': H, 'vol': vol, 'H_diag': H_diag, 'vol_diag': vol_diag_simplified}


def demo_5_latex_export():
    """Demo 5: Export formulas to LaTeX."""
    print("\n" + "=" * 70)
    print("DEMO 5: LaTeX Export")
    print("=" * 70)

    # Simple metric tensor
    h11, h12, h22 = symbols('h_{11} h_{12} h_{22}', real=True)
    epsilon_Sigma = symbols('epsilon_Sigma', positive=True)

    H = Matrix([[h11, h12],
                [h12, h22]])

    g = H + epsilon_Sigma * sp.eye(2)

    print("\nMetric tensor g = H + ε_Σ I:")
    latex_g = export_to_latex(g, name='g')
    print(latex_g)

    # Determinant
    det_g = g.det()
    print("\nDeterminant det(g):")
    latex_det = export_to_latex(det_g, name='\\det(g)')
    print(latex_det)

    # Volume element
    vol = sqrt(det_g)
    print("\nVolume element:")
    latex_vol = export_to_latex(vol, name='\\sqrt{\\det(g)}')
    print(latex_vol)

    return {'latex_g': latex_g, 'latex_det': latex_det, 'latex_vol': latex_vol}


def demo_6_hessian_structure():
    """Demo 6: Hessian decomposition (Chapter 9.3)."""
    print("\n" + "=" * 70)
    print("DEMO 6: Hessian Structure (Rank-1 + Full)")
    print("=" * 70)

    print("\nHessian has two terms:")
    print("  H = g''_A(Z) ∇Z ⊗ ∇Z  +  g'_A(Z) ∇²Z")
    print("      └── Rank-1 term ──┘    └── Full term ──┘")

    # Create symbolic components
    z = symbols('z', real=True)
    A = symbols('A', positive=True)

    # Gradient of Z-score (2D example)
    dZ_dx1, dZ_dx2 = symbols('partial_Z_x1 partial_Z_x2', real=True)
    nabla_Z = Matrix([dZ_dx1, dZ_dx2])

    # Hessian of Z-score
    d2Z_11, d2Z_12, d2Z_22 = symbols('partial2_Z_11 partial2_Z_12 partial2_Z_22', real=True)
    nabla2_Z = Matrix([[d2Z_11, d2Z_12],
                       [d2Z_12, d2Z_22]])

    # Rescale function derivatives
    g_prime = RescaleFunction.sigmoid_derivative(z, A, order=1)
    g_double_prime = RescaleFunction.sigmoid_derivative(z, A, order=2)

    print("\nRank-1 term (outer product):")
    rank1_term = g_double_prime * nabla_Z * nabla_Z.T
    print("  g''_A(Z) ∇Z ⊗ ∇Z =")
    pprint(rank1_term)

    print("\nFull Hessian term:")
    full_term = g_prime * nabla2_Z
    print("  g'_A(Z) ∇²Z =")
    pprint(full_term)

    print("\nTotal Hessian H:")
    H_total = rank1_term + full_term
    print("  H =")
    pprint(H_total)

    print("\nKey insight:")
    print("  - Rank-1 term is aligned with gradient direction")
    print("  - Creates anisotropy along fitness level sets")
    print("  - Full term captures intrinsic curvature of measurement")

    return {'rank1': rank1_term, 'full': full_term, 'H_total': H_total}


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 70)
    print("SYMBOLIC GEOMETRY ALGEBRA DEMONSTRATION")
    print("Implementation of Chapter 9 (docs/source/08_emergent_geometry.md)")
    print("=" * 70)

    results = {}

    # Run all demos
    results['demo1'] = demo_1_rescale_function()
    results['demo2'] = demo_2_localization_weights()
    results['demo3'] = demo_3_metric_tensor_2d()
    results['demo4'] = demo_4_volume_element_3d()
    results['demo5'] = demo_5_latex_export()
    results['demo6'] = demo_6_hessian_structure()

    print("\n" + "=" * 70)
    print("ALL DEMONSTRATIONS COMPLETE")
    print("=" * 70)

    print("\nKey Takeaways:")
    print("1. All formulas from Chapter 9 are symbolically verifiable")
    print("2. SymPy can manipulate and simplify geometric quantities")
    print("3. Expressions can be exported to LaTeX for documentation")
    print("4. Simplified cases (2D, diagonal) provide intuition")
    print("5. Full 3D computation is possible but computationally expensive")

    print("\nNext Steps:")
    print("- Use symbolic expressions to verify specific formulas")
    print("- Generate numerical functions with sp.lambdify()")
    print("- Compute Christoffel symbols for specific cases")
    print("- Explore parameter dependencies symbolically")

    return results


if __name__ == '__main__':
    results = main()
