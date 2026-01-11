"""
Symbolic Algebra for Emergent Geometry in the Adaptive Gas.

This module implements all the mathematical expressions from Chapter 9 of
`docs/source/08_emergent_geometry.md` using SymPy for symbolic computation.

It provides a complete symbolic framework for:
1. Fitness potential construction (localization, moments, Z-score)
2. Hessian computation via chain rule
3. Metric tensor and its properties
4. Volume element (determinant)
5. Christoffel symbols
6. 3D concrete expansions

The goal is to enable:
- Verification of algebraic derivations
- Symbolic simplification and manipulation
- Automatic differentiation for geometric quantities
- Generation of optimized numerical code

Author: Claude (Anthropic)
Date: 2025-10-09
Reference: docs/source/08_emergent_geometry.md Chapter 9
"""

from __future__ import annotations

import sympy as sp
from sympy import (
    diff,
    exp,
    expand,
    Function,
    Matrix,
    simplify,
    sqrt,
)


# ============================================================================
# Symbolic Variables and Parameters
# ============================================================================


def create_algorithmic_parameters() -> dict[str, sp.Symbol]:
    """
    Create all algorithmic parameters from Chapter 9.6.1.

    Returns
    -------
    dict
        Dictionary mapping parameter names to SymPy symbols.
    """
    return {
        # Measurement and Fitness Parameters
        "rho": sp.Symbol("rho", positive=True, real=True),
        "kappa_var_min": sp.Symbol("kappa_var_min", positive=True, real=True),
        "A": sp.Symbol("A", positive=True, real=True),
        "epsilon_d": sp.Symbol("epsilon_d", positive=True, real=True),
        # Diffusion and Geometry Parameters
        "epsilon_Sigma": sp.Symbol("epsilon_Sigma", positive=True, real=True),
        "sigma_v": sp.Symbol("sigma_v", positive=True, real=True),
        "delta": sp.Symbol("delta", positive=True, real=True),
        # Adaptive Dynamics Parameters
        "epsilon_F": sp.Symbol("epsilon_F", nonnegative=True, real=True),
        "nu": sp.Symbol("nu", nonnegative=True, real=True),
        # Kinetic and Cloning Parameters
        "gamma": sp.Symbol("gamma", positive=True, real=True),
        "tau": sp.Symbol("tau", positive=True, real=True),
        "lambda_v": sp.Symbol("lambda_v", positive=True, real=True),
        "lambda_alg": sp.Symbol("lambda_alg", nonnegative=True, real=True),
        # Cloning and Selection Parameters
        "alpha": sp.Symbol("alpha", real=True),  # In [0, 1]
        "beta": sp.Symbol("beta", real=True),  # In [0, 1]
        "eta": sp.Symbol("eta", real=True),  # In (0, 1)
        "epsilon_rescale": sp.Symbol("epsilon_rescale", positive=True, real=True),
        # Confinement Parameters
        "kappa_conf": sp.Symbol("kappa_conf", positive=True, real=True),
        # Lyapunov Weights
        "alpha_x": sp.Symbol("alpha_x", positive=True, real=True),
        "alpha_v": sp.Symbol("alpha_v", positive=True, real=True),
        "alpha_D": sp.Symbol("alpha_D", positive=True, real=True),
        "alpha_R": sp.Symbol("alpha_R", positive=True, real=True),
    }


def create_state_variables(dim: int = 3) -> tuple[Matrix, dict[str, sp.Symbol]]:
    """
    Create state space variables for a given dimension.

    Parameters
    ----------
    dim : int
        Dimension of state space (default: 3 for 3D)

    Returns
    -------
    x : Matrix
        Position vector [x_1, x_2, ..., x_dim]
    coords : dict
        Dictionary mapping coordinate names to symbols
    """
    coords = {f"x_{i + 1}": sp.Symbol(f"x_{i + 1}", real=True) for i in range(dim)}
    x = Matrix([coords[f"x_{i + 1}"] for i in range(dim)])

    return x, coords


# ============================================================================
# Measurement Function (Abstract)
# ============================================================================


class MeasurementFunction:
    """
    Symbolic representation of measurement function d: R^d -> R.

    This is kept abstract (as a SymPy Function) to allow arbitrary
    measurement functions. Specific examples can be instantiated.
    """

    def __init__(self, dim: int = 3, name: str = "d"):
        """
        Parameters
        ----------
        dim : int
            Dimension of state space
        name : str
            Name of the function (default: 'd')
        """
        self.dim = dim
        self.name = name
        self.x, self.coords = create_state_variables(dim)

        # Create symbolic function d(x_1, x_2, ..., x_dim)
        self.d = Function(name)(*self.x)

    def gradient(self) -> Matrix:
        """
        Compute symbolic gradient ∇d.

        Returns
        -------
        Matrix
            Gradient vector [∂d/∂x_1, ∂d/∂x_2, ..., ∂d/∂x_dim]
        """
        return Matrix([diff(self.d, xi) for xi in self.x])

    def hessian(self) -> Matrix:
        """
        Compute symbolic Hessian ∇²d.

        Returns
        -------
        Matrix
            Hessian matrix [∂²d/∂x_i∂x_j]
        """
        grad = self.gradient()
        return Matrix([
            [diff(grad[i], self.x[j]) for j in range(self.dim)] for i in range(self.dim)
        ])


# ============================================================================
# Rescale Functions (Chapter 9.7.1)
# ============================================================================


class RescaleFunction:
    """Symbolic rescale function g_A: R -> [0, A]."""

    @staticmethod
    def sigmoid(z: sp.Expr, A: sp.Symbol) -> sp.Expr:
        """
        Sigmoid rescale: g_A(z) = A / (1 + exp(-z))

        Parameters
        ----------
        z : Expr
            Input (Z-score)
        A : Symbol
            Maximum value

        Returns
        -------
        Expr
            g_A(z)
        """
        return A / (1 + exp(-z))

    @staticmethod
    def sigmoid_derivative(z: sp.Expr, A: sp.Symbol, order: int = 1) -> sp.Expr:
        """
        Derivatives of sigmoid rescale.

        Parameters
        ----------
        z : Expr
            Input
        A : Symbol
            Maximum value
        order : int
            Derivative order (1, 2, or 3)

        Returns
        -------
        Expr
            g_A^(order)(z)
        """
        g = RescaleFunction.sigmoid(z, A)

        if order == 1:
            return diff(g, z)
        if order == 2:
            return diff(g, z, 2)
        if order == 3:
            return diff(g, z, 3)
        raise ValueError(f"Derivative order {order} not supported (use 1, 2, or 3)")


# ============================================================================
# Fitness Potential Pipeline (Chapter 9.2)
# ============================================================================


class FitnessPotential:
    """
    Symbolic fitness potential V_fit[f_k, ρ](x).

    Implements the complete pipeline from Chapter 9.2:
    1. Localization weights
    2. Localized moments (mean, variance)
    3. Regularized standard deviation
    4. Z-score
    5. Rescaled fitness potential
    """

    def __init__(
        self, dim: int = 3, num_walkers: int = 3, measurement: MeasurementFunction | None = None
    ):
        """
        Parameters
        ----------
        dim : int
            State space dimension
        num_walkers : int
            Number of alive walkers (k)
        measurement : MeasurementFunction, optional
            Measurement function (created if None)
        """
        self.dim = dim
        self.k = num_walkers

        # State variables
        self.x, self.coords = create_state_variables(dim)

        # Measurement function
        self.measurement = measurement or MeasurementFunction(dim)

        # Parameters
        self.params = create_algorithmic_parameters()

        # Walker positions and measurements
        self.x_walkers = [
            Matrix([sp.Symbol(f"x_{j + 1}^{i + 1}", real=True) for j in range(dim)])
            for i in range(num_walkers)
        ]
        self.d_walkers = [sp.Symbol(f"d_{i + 1}", real=True) for i in range(num_walkers)]

    def localization_kernel(self, x_i: Matrix, x_j: Matrix, rho: sp.Symbol) -> sp.Expr:
        """
        Gaussian localization kernel (unnormalized).

        K_ρ(x, x_j) = exp(-||x - x_j||²/(2ρ²))

        Parameters
        ----------
        x_i : Matrix
            Position of walker i
        x_j : Matrix
            Position of walker j
        rho : Symbol
            Localization scale

        Returns
        -------
        Expr
            K_ρ(x_i, x_j)
        """
        diff_vec = x_i - x_j
        dist_sq = diff_vec.dot(diff_vec)
        return exp(-dist_sq / (2 * rho**2))

    def localization_weights(self) -> list[sp.Expr]:
        """
        Compute normalized localization weights w_j(ρ).

        w_j = K_ρ(x, x_j) / Σ_ℓ K_ρ(x, x_ℓ)

        Returns
        -------
        list of Expr
            Weights [w_1, w_2, ..., w_k]
        """
        rho = self.params["rho"]

        # Compute unnormalized kernels
        kernels = [self.localization_kernel(self.x, x_j, rho) for x_j in self.x_walkers]

        # Normalization
        Z = sum(kernels)

        # Normalized weights
        return [K / Z for K in kernels]

    def localized_mean(self, weights: list[sp.Expr] | None = None) -> sp.Expr:
        """
        Compute localized mean μ_ρ.

        μ_ρ = Σ_j w_j d_j

        Parameters
        ----------
        weights : list of Expr, optional
            Precomputed weights (computed if None)

        Returns
        -------
        Expr
            μ_ρ[f_k, d, x]
        """
        if weights is None:
            weights = self.localization_weights()

        return sum(w * d for w, d in zip(weights, self.d_walkers))

    def localized_variance(
        self, weights: list[sp.Expr] | None = None, mu: sp.Expr | None = None
    ) -> sp.Expr:
        """
        Compute localized variance σ²_ρ.

        σ²_ρ = Σ_j w_j (d_j - μ_ρ)²

        Parameters
        ----------
        weights : list of Expr, optional
            Precomputed weights
        mu : Expr, optional
            Precomputed mean

        Returns
        -------
        Expr
            σ²_ρ[f_k, d, x]
        """
        if weights is None:
            weights = self.localization_weights()
        if mu is None:
            mu = self.localized_mean(weights)

        return sum(w * (d - mu) ** 2 for w, d in zip(weights, self.d_walkers))

    def z_score(self, weights: list[sp.Expr] | None = None) -> sp.Expr:
        """
        Compute Z-score with regularization.

        Z_ρ = (d(x) - μ_ρ) / max{√(σ²_ρ), κ_var_min}

        Note: SymPy version uses max function (not C¹ patch for simplicity).
        For the C¹ patch, use sigma_patch_c1() instead.

        Parameters
        ----------
        weights : list of Expr, optional
            Precomputed weights

        Returns
        -------
        Expr
            Z_ρ[f_k, d, x]
        """
        if weights is None:
            weights = self.localization_weights()

        mu = self.localized_mean(weights)
        var = self.localized_variance(weights, mu)

        # Regularized std dev (using max for simplicity)
        sigma_prime = sp.Max(sqrt(var), self.params["kappa_var_min"])

        # Z-score
        d_x = self.measurement.d
        return (d_x - mu) / sigma_prime

    def fitness_potential_sigmoid(self, weights: list[sp.Expr] | None = None) -> sp.Expr:
        """
        Compute fitness potential with sigmoid rescale.

        V_fit = A / (1 + exp(-Z_ρ))

        Parameters
        ----------
        weights : list of Expr, optional
            Precomputed weights

        Returns
        -------
        Expr
            V_fit[f_k, ρ](x)
        """
        Z = self.z_score(weights)
        A = self.params["A"]
        return RescaleFunction.sigmoid(Z, A)


# ============================================================================
# Hessian and Metric (Chapter 9.3-9.4)
# ============================================================================


class EmergentMetric:
    """
    Symbolic computation of the emergent Riemannian metric.

    Implements:
    1. Hessian H = ∇²V_fit via chain rule
    2. Metric g = H + ε_Σ I
    3. Determinant det(g)
    4. Volume element √det(g)
    """

    def __init__(self, fitness: FitnessPotential):
        """
        Parameters
        ----------
        fitness : FitnessPotential
            Fitness potential instance
        """
        self.fitness = fitness
        self.dim = fitness.dim
        self.params = fitness.params
        self.x = fitness.x

    def gradient_V_fit(self, V_fit: sp.Expr) -> Matrix:
        """
        Compute gradient ∇V_fit.

        Parameters
        ----------
        V_fit : Expr
            Fitness potential

        Returns
        -------
        Matrix
            ∇V_fit
        """
        return Matrix([diff(V_fit, xi) for xi in self.x])

    def hessian_V_fit(self, V_fit: sp.Expr) -> Matrix:
        """
        Compute Hessian H = ∇²V_fit.

        This uses SymPy's automatic differentiation to compute the Hessian
        via the chain rule. The result matches the explicit formula from
        Chapter 9.3:

        H = g''_A(Z) ∇Z ⊗ ∇Z + g'_A(Z) ∇²Z

        Parameters
        ----------
        V_fit : Expr
            Fitness potential

        Returns
        -------
        Matrix
            Hessian H (dim × dim matrix)
        """
        grad = self.gradient_V_fit(V_fit)
        return Matrix([
            [diff(grad[i], self.x[j]) for j in range(self.dim)] for i in range(self.dim)
        ])

    def metric_tensor(self, H: Matrix) -> Matrix:
        """
        Compute metric tensor g = H + ε_Σ I.

        Parameters
        ----------
        H : Matrix
            Hessian matrix

        Returns
        -------
        Matrix
            Metric tensor g (dim × dim symmetric matrix)
        """
        epsilon_Sigma = self.params["epsilon_Sigma"]
        I = sp.eye(self.dim)
        return H + epsilon_Sigma * I

    def volume_element(self, g: Matrix) -> sp.Expr:
        """
        Compute volume element √det(g).

        For 3D, this uses the explicit formula from Chapter 9.7.6:
        det(g) = det(H) + ε_Σ tr(adj(H)) + ε_Σ² tr(H) + ε_Σ³

        Parameters
        ----------
        g : Matrix
            Metric tensor

        Returns
        -------
        Expr
            √det(g)
        """
        det_g = g.det()
        return sqrt(det_g)

    def volume_element_3d_explicit(self, H: Matrix) -> sp.Expr:
        """
        Compute 3D volume element using explicit formula (Chapter 9.7.6).

        det(g) = det(H) + ε_Σ(h₂₂h₃₃ - h₂₃² + h₁₁h₃₃ - h₁₃² + h₁₁h₂₂ - h₁₂²)
                 + ε_Σ²(h₁₁ + h₂₂ + h₃₃) + ε_Σ³

        Parameters
        ----------
        H : Matrix
            Hessian (must be 3×3)

        Returns
        -------
        Expr
            √det(g)
        """
        if H.shape != (3, 3):
            msg = "Explicit 3D formula requires 3×3 Hessian"
            raise ValueError(msg)

        epsilon_Sigma = self.params["epsilon_Sigma"]

        # Extract Hessian components
        h11, h12, h13 = H[0, 0], H[0, 1], H[0, 2]
        _h21, h22, h23 = H[1, 0], H[1, 1], H[1, 2]
        _h31, _h32, h33 = H[2, 0], H[2, 1], H[2, 2]

        # det(H)
        det_H = H.det()

        # tr(adj(H)) = sum of principal 2×2 minors
        tr_adj_H = (h22 * h33 - h23**2) + (h11 * h33 - h13**2) + (h11 * h22 - h12**2)

        # tr(H)
        tr_H = h11 + h22 + h33

        # det(g) = det(H + ε_Σ I)
        det_g = det_H + epsilon_Sigma * tr_adj_H + epsilon_Sigma**2 * tr_H + epsilon_Sigma**3

        return sqrt(det_g)


# ============================================================================
# Christoffel Symbols (Chapter 9.7.7)
# ============================================================================


class ChristoffelSymbols:
    """
    Symbolic computation of Christoffel symbols Γᵃ_bc.

    Γᵃ_bc = (1/2) gᵃᵈ (∂g_db/∂xᶜ + ∂g_dc/∂xᵇ - ∂g_bc/∂xᵈ)
    """

    def __init__(self, metric: EmergentMetric):
        """
        Parameters
        ----------
        metric : EmergentMetric
            Metric instance
        """
        self.metric = metric
        self.dim = metric.dim
        self.x = metric.x

    def compute(self, g: Matrix, simplify_result: bool = False) -> list[list[list[sp.Expr]]]:
        """
        Compute all Christoffel symbols.

        Parameters
        ----------
        g : Matrix
            Metric tensor
        simplify_result : bool
            Whether to simplify (can be slow)

        Returns
        -------
        list of list of list of Expr
            Γᵃ_bc indexed as Gamma[a][b][c]
        """
        dim = self.dim

        # Compute inverse metric
        g_inv = g.inv()

        # Initialize Christoffel symbol array
        Gamma = [[[sp.S.Zero for _ in range(dim)] for _ in range(dim)] for _ in range(dim)]

        # Compute each component
        for a in range(dim):
            for b in range(dim):
                for c in range(dim):
                    # Γᵃ_bc = (1/2) Σ_d gᵃᵈ (∂g_db/∂xᶜ + ∂g_dc/∂xᵇ - ∂g_bc/∂xᵈ)
                    christoffel = sp.S.Zero

                    for d in range(dim):
                        term1 = diff(g[d, b], self.x[c])
                        term2 = diff(g[d, c], self.x[b])
                        term3 = diff(g[b, c], self.x[d])

                        christoffel += sp.Rational(1, 2) * g_inv[a, d] * (term1 + term2 - term3)

                    if simplify_result:
                        christoffel = simplify(christoffel)

                    Gamma[a][b][c] = christoffel

        return Gamma

    def get_component(self, Gamma: list, a: int, b: int, c: int) -> sp.Expr:
        """
        Get a specific Christoffel symbol component.

        Parameters
        ----------
        Gamma : list
            Christoffel symbols array
        a, b, c : int
            Indices (0-indexed)

        Returns
        -------
        Expr
            Γᵃ_bc
        """
        return Gamma[a][b][c]


# ============================================================================
# Geodesic Equation (Chapter 9.7.8)
# ============================================================================


class GeodesicEquation:
    """
    Symbolic geodesic equation.

    d²γᵃ/dt² + Σ_bc Γᵃ_bc (dγᵇ/dt)(dγᶜ/dt) = 0
    """

    def __init__(self, christoffel: ChristoffelSymbols):
        """
        Parameters
        ----------
        christoffel : ChristoffelSymbols
            Christoffel symbols instance
        """
        self.christoffel = christoffel
        self.dim = christoffel.dim

        # Create symbolic curve γ(t)
        self.t = sp.Symbol("t", real=True)
        self.gamma = [Function(f"gamma_{i + 1}")(self.t) for i in range(self.dim)]

        # Velocities
        self.gamma_dot = [diff(g, self.t) for g in self.gamma]

        # Accelerations
        self.gamma_ddot = [diff(g_dot, self.t) for g_dot in self.gamma_dot]

    def equation(self, Gamma: list, component: int) -> sp.Expr:
        """
        Get geodesic equation for a specific component.

        Parameters
        ----------
        Gamma : list
            Christoffel symbols
        component : int
            Which component (0-indexed, 0 ≤ component < dim)

        Returns
        -------
        Expr
            Geodesic equation: γ̈ᵃ + Σ_bc Γᵃ_bc γ̇ᵇ γ̇ᶜ = 0
        """
        a = component

        # Start with acceleration
        eq = self.gamma_ddot[a]

        # Add Christoffel terms
        for b in range(self.dim):
            for c in range(self.dim):
                # Substitute curve into Christoffel symbol
                Gamma_abc = Gamma[a][b][c]
                # (This would require substitution of γ(t) for x, which is complex)
                # For now, keep symbolic
                eq += Gamma_abc * self.gamma_dot[b] * self.gamma_dot[c]

        return eq


# ============================================================================
# 3D Concrete Example (Chapter 9.7)
# ============================================================================


def example_3d_metric():
    """
    Complete 3D example with explicit algebraic expansions.

    This demonstrates the full pipeline from algorithmic parameters
    to geometric quantities for a 3D state space.

    Returns
    -------
    dict
        Dictionary containing all symbolic quantities
    """
    print("=" * 70)
    print("3D Emergent Metric: Complete Symbolic Computation")
    print("=" * 70)

    # Create fitness potential (k=3 walkers for simplicity)
    print("\n1. Creating fitness potential with 3 walkers...")
    fitness = FitnessPotential(dim=3, num_walkers=3)

    # Compute localization weights
    print("2. Computing localization weights w_j(ρ)...")
    weights = fitness.localization_weights()
    print(f"   Number of weights: {len(weights)}")

    # Compute moments
    print("3. Computing localized moments (μ_ρ, σ²_ρ)...")
    mu = fitness.localized_mean(weights)
    var = fitness.localized_variance(weights, mu)

    # Compute Z-score
    print("4. Computing Z-score Z_ρ...")
    Z = fitness.z_score(weights)

    # Compute fitness potential
    print("5. Computing fitness potential V_fit...")
    V_fit = fitness.fitness_potential_sigmoid(weights)

    # Create metric
    print("\n6. Computing Hessian H = ∇²V_fit...")
    metric = EmergentMetric(fitness)
    H = metric.hessian_V_fit(V_fit)
    print(f"   Hessian shape: {H.shape}")

    # Compute metric tensor
    print("7. Computing metric tensor g = H + ε_Σ I...")
    g = metric.metric_tensor(H)

    # Compute volume element (two methods)
    print("\n8. Computing volume element √det(g)...")
    print("   Method 1: Direct determinant...")
    vol_direct = metric.volume_element(g)

    print("   Method 2: Explicit 3D formula (Chapter 9.7.6)...")
    vol_explicit = metric.volume_element_3d_explicit(H)

    # Compute Christoffel symbols (first component only for demo)
    print("\n9. Computing Christoffel symbols (Γ¹₁₁ only for demo)...")
    christoffel = ChristoffelSymbols(metric)
    # Note: Computing all components is very slow, so we skip for the example
    print("   (Full computation skipped - use christoffel.compute(g) for all)")

    # Package results
    results = {
        "fitness": fitness,
        "weights": weights,
        "mu": mu,
        "variance": var,
        "Z_score": Z,
        "V_fit": V_fit,
        "gradient_V": metric.gradient_V_fit(V_fit),
        "Hessian": H,
        "metric": g,
        "volume_direct": vol_direct,
        "volume_explicit": vol_explicit,
        "metric_obj": metric,
        "christoffel_obj": christoffel,
    }

    print("\n" + "=" * 70)
    print("Computation complete!")
    print("=" * 70)
    print("\nAccess results via returned dictionary:")
    print("  - results['V_fit']: Fitness potential")
    print("  - results['Hessian']: 3×3 Hessian matrix")
    print("  - results['metric']: 3×3 metric tensor")
    print("  - results['volume_explicit']: √det(g)")

    return results


# ============================================================================
# Utility Functions
# ============================================================================


def simplify_metric_component(g_ij: sp.Expr, level: str = "basic") -> sp.Expr:
    """
    Simplify a metric tensor component.

    Parameters
    ----------
    g_ij : Expr
        Metric component
    level : str
        Simplification level: 'basic', 'full', 'trigsimp'

    Returns
    -------
    Expr
        Simplified expression
    """
    if level == "basic":
        return simplify(g_ij)
    if level == "full":
        return simplify(expand(g_ij))
    if level == "trigsimp":
        return sp.trigsimp(simplify(g_ij))
    return g_ij


def export_to_latex(expr: sp.Expr, name: str = "") -> str:
    """
    Export SymPy expression to LaTeX.

    Parameters
    ----------
    expr : Expr
        SymPy expression
    name : str
        Name for the expression (optional)

    Returns
    -------
    str
        LaTeX string
    """
    latex_str = sp.latex(expr)

    if name:
        return f"{name} = {latex_str}"
    return latex_str


def generate_numerical_function(expr: sp.Expr, variables: list[sp.Symbol]) -> callable:
    """
    Generate fast numerical function from symbolic expression.

    Parameters
    ----------
    expr : Expr
        Symbolic expression
    variables : list of Symbol
        Variables to use as function arguments

    Returns
    -------
    callable
        Numerical function (lambdified)
    """
    return sp.lambdify(variables, expr, modules=["numpy"])


# ============================================================================
# Main Example
# ============================================================================

if __name__ == "__main__":
    print(__doc__)

    # Run 3D example
    results = example_3d_metric()

    # Print some sample results
    print("\n" + "=" * 70)
    print("Sample Results")
    print("=" * 70)

    print("\nMetric tensor g (3×3):")
    print(results["metric"])

    print("\nHessian H₁₁ (first component, truncated):")
    H11 = results["Hessian"][0, 0]
    print(str(H11))

    print("\n\nVolume element √det(g) (explicit formula, truncated):")
    vol = results["volume_explicit"]
    print(str(vol))

    # Example: export to LaTeX
    print("\n\n" + "=" * 70)
    print("LaTeX Export Example")
    print("=" * 70)

    # Get a simple component
    epsilon_Sigma = results["fitness"].params["epsilon_Sigma"]
    latex_metric = export_to_latex(epsilon_Sigma, name="\\epsilon_\\Sigma")
    print(f"\nMetric regularization parameter:\n  {latex_metric}")

    print("\n" + "=" * 70)
    print("Module loaded successfully!")
    print("=" * 70)
    print("\nUsage:")
    print("  from fragile.fitness_algebra import FitnessPotential, EmergentMetric")
    print("  fitness = FitnessPotential(dim=3, num_walkers=5)")
    print("  V_fit = fitness.fitness_potential_sigmoid()")
    print("  metric = EmergentMetric(fitness)")
    print("  H = metric.hessian_V_fit(V_fit)")
    print("  g = metric.metric_tensor(H)")
