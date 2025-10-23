"""Convergence Bounds and Constants for the Fragile Gas Framework.

This module implements all convergence bounds, rates, and constants derived from the
mathematical framework. It provides functions to compute environment-dependent constants
for debugging convergence, characterizing dynamics, and tuning parameters.

The bounds are organized by source document and mathematical framework:
    1. Euclidean Gas Convergence (06_convergence.md)
    2. LSI and KL-Convergence (09_kl_convergence.md)
    3. Geometric Gas N-Uniform Bounds (15_geometric_gas_lsi_proof.md, geometric_foundations_lsi.md)
    4. Wasserstein Contraction (04_wasserstein_contraction.md)
    5. Parameter Regime Validators
    6. Compound Bounds and Diagnostics
    7. Sensitivity Analysis
    8. Optimization Helpers

All functions accept primitive types (float, int) or numpy arrays and return scalars
or simple arrays. Each function includes:
    - Mathematical formula in docstring
    - Source document reference
    - Physical interpretation
    - Type hints

References
----------
- docs/source/1_euclidean_gas/06_convergence.md - Foster-Lyapunov convergence theory
- docs/source/1_euclidean_gas/09_kl_convergence.md - LSI and KL-divergence
- docs/source/2_geometric_gas/15_geometric_gas_lsi_proof.md - N-uniform LSI
- docs/source/3_brascamp_lieb/geometric_foundations_lsi.md - Geometric LSI foundations
- docs/source/1_euclidean_gas/04_wasserstein_contraction.md - Wasserstein contraction

"""

from typing import Optional, Dict, Tuple
import numpy as np


# ============================================================================
# Section 1: Euclidean Gas Convergence Bounds (06_convergence.md)
# ============================================================================


def kappa_v(gamma: float, tau: float = 0.01) -> float:
    """Velocity contraction rate for the kinetic operator.

    From 06_convergence.md Theorem 5.1.2, the velocity variance contracts at rate:

    κ_v = 2γ - O(τ)

    where γ is the friction coefficient and τ is the time step size.

    **Physical interpretation**: Velocity dissipation from Langevin friction. Larger
    friction leads to faster velocity thermalization.

    Parameters
    ----------
    gamma : float
        Friction coefficient (γ > 0)
    tau : float, optional
        Time step size (default: 0.01)

    Returns
    -------
    float
        Velocity contraction rate κ_v

    References
    ----------
    - 06_convergence.md § 5.1.2 (Velocity Component Rate)
    - 05_kinetic_contraction.md § 3.3 (Velocity Variance Contraction)

    """
    return 2.0 * gamma * (1.0 - tau)  # O(τ) correction


def kappa_x(lambda_alg: float, tau: float = 0.01) -> float:
    """Position contraction rate from cloning operator.

    From 06_convergence.md Theorem 5.1.1, the positional variance contracts at rate:

    κ_x ≈ λ

    where λ is the cloning rate (algorithmic selection pressure).

    **Physical interpretation**: Evolutionary selection contracts positional variance
    by eliminating high-error configurations (Keystone Principle).

    Parameters
    ----------
    lambda_alg : float
        Cloning rate / selection pressure (λ > 0)
    tau : float, optional
        Time step size (default: 0.01)

    Returns
    -------
    float
        Position contraction rate κ_x

    References
    ----------
    - 06_convergence.md § 5.1.1 (Position Component Rate)
    - 03_cloning.md § 8 (Keystone Principle)

    """
    return lambda_alg * (1.0 - tau)  # O(τ) correction


def kappa_W(gamma: float, lambda_min: float, c_hypo: float = 0.1) -> float:
    """Wasserstein contraction rate from hypocoercive coupling.

    From 06_convergence.md Theorem 5.1.3, the inter-swarm Wasserstein distance
    contracts at rate:

    κ_W ≈ (c_hypo^2 · γ) / (1 + γ/λ_min)

    where c_hypo is the hypocoercive constant, γ is friction, and λ_min is the
    minimum eigenvalue of the confining potential.

    **Physical interpretation**: Position-velocity coupling allows velocity dissipation
    to indirectly contract positional error through hypocoercivity.

    Parameters
    ----------
    gamma : float
        Friction coefficient
    lambda_min : float
        Minimum eigenvalue of confining potential Hessian
    c_hypo : float, optional
        Hypocoercive coupling constant (default: 0.1)

    Returns
    -------
    float
        Wasserstein contraction rate κ_W

    References
    ----------
    - 06_convergence.md § 5.1.3 (Wasserstein Component Rate)
    - 05_kinetic_contraction.md § 2.3 (Hypocoercive Contraction)

    """
    return (c_hypo**2 * gamma) / (1.0 + gamma / lambda_min)


def kappa_b(
    lambda_alg: float,
    delta_f_boundary: float,
    f_typical: float = 1.0
) -> float:
    """Boundary potential contraction rate (Safe Harbor mechanism).

    From 06_convergence.md Theorem 5.1.4, the boundary potential contracts at rate:

    κ_b ≈ λ · (Δf_boundary / f_typical)

    where Δf_boundary is the fitness drop near the boundary and f_typical is the
    typical fitness scale.

    **Physical interpretation**: Walkers near the boundary have low fitness and are
    preferentially eliminated by cloning, contracting the boundary potential.

    Parameters
    ----------
    lambda_alg : float
        Cloning rate
    delta_f_boundary : float
        Fitness deficit near boundary (how much worse boundary states are)
    f_typical : float, optional
        Typical fitness scale (default: 1.0)

    Returns
    -------
    float
        Boundary contraction rate κ_b

    References
    ----------
    - 06_convergence.md § 5.1.4 (Boundary Component Rate)
    - 03_cloning.md § 11 (Safe Harbor Mechanism)

    """
    return lambda_alg * (delta_f_boundary / f_typical)


def kappa_total(*component_rates: float, epsilon_coupling: float = 0.0) -> float:
    """Total synergistic convergence rate (bottleneck principle).

    From 06_convergence.md Theorem 5.5, the total rate is:

    κ_total = min(κ_x, κ_v, κ_W, κ_b) · (1 - ε_coupling)

    where ε_coupling < 1 is the expansion-to-contraction ratio from operator coupling.

    **Physical interpretation**: The slowest-contracting component becomes the bottleneck.
    The synergistic dissipation paradigm ensures each operator contracts what the other
    expands, but coupling introduces a small efficiency loss.

    Parameters
    ----------
    *component_rates : float
        Component contraction rates (κ_x, κ_v, κ_W, κ_b)
    epsilon_coupling : float, optional
        Coupling penalty (default: 0.0, should be << 1)

    Returns
    -------
    float
        Total convergence rate κ_total

    References
    ----------
    - 06_convergence.md § 5.5 (Total Rate Formula)
    - 06_convergence.md § 3 (Synergistic Dissipation)

    """
    return min(component_rates) * (1.0 - epsilon_coupling)


def C_total(
    C_x: float,
    C_v: float,
    C_W: float,
    C_b: float,
    kappa_total: float,
) -> float:
    """Equilibrium constant for Foster-Lyapunov condition.

    From 06_convergence.md Theorem 3.4, the equilibrium constant is:

    C_total = (C_x + C_v + C_W + C_b) / κ_total

    where C_i are the expansion constants from each operator.

    **Physical interpretation**: At equilibrium, contraction balances expansion,
    yielding finite variance V_total^eq = C_total / κ_total.

    Parameters
    ----------
    C_x : float
        Position expansion constant (from kinetic noise)
    C_v : float
        Velocity expansion constant (from Langevin noise)
    C_W : float
        Wasserstein expansion constant (from cloning structural perturbation)
    C_b : float
        Boundary expansion constant (from thermal boundary crossings)
    kappa_total : float
        Total contraction rate

    Returns
    -------
    float
        Equilibrium constant C_total

    References
    ----------
    - 06_convergence.md § 3.4 (Foster-Lyapunov Main Theorem)
    - 06_convergence.md § 5.5 (Total Rate Formula)

    """
    return (C_x + C_v + C_W + C_b) / kappa_total


def T_mix(
    epsilon: float,
    kappa_total: float,
    V_init: float = 1.0,
    C_total: float = 1.0
) -> float:
    """Mixing time to reach ε-proximity to equilibrium.

    From 06_convergence.md Proposition 5.6, the mixing time is:

    T_mix(ε) = (1/κ_total) · ln((V_init · κ_total) / (ε · C_total))

    For typical initialization V_init ~ O(1) and target ε = 0.01:
    T_mix ≈ 5/κ_total

    **Physical interpretation**: Time for the Lyapunov function to decay to
    within ε of its equilibrium value. Inversely proportional to contraction rate.

    Parameters
    ----------
    epsilon : float
        Target accuracy (e.g., 0.01 for 1% proximity)
    kappa_total : float
        Total contraction rate
    V_init : float, optional
        Initial Lyapunov value (default: 1.0)
    C_total : float, optional
        Equilibrium constant (default: 1.0)

    Returns
    -------
    float
        Mixing time T_mix (in units of time steps)

    References
    ----------
    - 06_convergence.md § 5.6 (Convergence Time Estimates)

    """
    return (1.0 / kappa_total) * np.log((V_init * kappa_total) / (epsilon * C_total))


def equilibrium_variance_x(
    sigma_v: float,
    tau: float,
    gamma: float,
    lambda_alg: float
) -> float:
    """Equilibrium positional variance from Foster-Lyapunov balance.

    From 06_convergence.md Theorem 5.3, the equilibrium positional variance is:

    Var_x^eq ≈ (σ_v^2 · τ^2) / (γ · λ)

    **Physical interpretation**: Positional diffusion from kinetic noise balanced
    by cloning contraction. Small time steps or strong selection reduce variance.

    Parameters
    ----------
    sigma_v : float
        Langevin noise scale
    tau : float
        Time step size
    gamma : float
        Friction coefficient
    lambda_alg : float
        Cloning rate

    Returns
    -------
    float
        Equilibrium positional variance

    References
    ----------
    - 06_convergence.md § 5.3 (Equilibrium Variance Bounds)

    """
    return (sigma_v**2 * tau**2) / (gamma * lambda_alg)


def equilibrium_variance_v(d: int, sigma_v: float, gamma: float) -> float:
    """Equilibrium velocity variance from Langevin thermalization.

    From 06_convergence.md Theorem 5.3, the equilibrium velocity variance is:

    Var_v^eq ≈ d · σ_v^2 / γ

    **Physical interpretation**: Langevin noise balanced by friction, approaching
    thermal equilibrium with temperature ∝ σ_v^2 / γ.

    Parameters
    ----------
    d : int
        Spatial dimension
    sigma_v : float
        Langevin noise scale
    gamma : float
        Friction coefficient

    Returns
    -------
    float
        Equilibrium velocity variance

    References
    ----------
    - 06_convergence.md § 5.3 (Equilibrium Variance Bounds)
    - 05_kinetic_contraction.md § 3.3 (Velocity Equilibrium)

    """
    return d * (sigma_v**2) / gamma


# ============================================================================
# Section 2: LSI and KL-Convergence Bounds (09_kl_convergence.md)
# ============================================================================


def C_LSI_euclidean(
    gamma: float,
    kappa_conf: float,
    kappa_W: float,
    delta_sq: float
) -> float:
    """LSI constant for the Euclidean Gas via displacement convexity.

    From 09_kl_convergence.md Theorem 2.6, the LSI constant is:

    C_LSI = O(1 / (γ · κ_conf · κ_W · δ²))

    where γ is friction, κ_conf is confinement strength, κ_W is Wasserstein
    contraction, and δ² is cloning noise variance.

    **Physical interpretation**: LSI controls exponential KL-convergence.
    Larger friction, stronger confinement, and larger cloning noise all
    improve the LSI constant (smaller is better).

    Parameters
    ----------
    gamma : float
        Friction coefficient
    kappa_conf : float
        Convexity constant of confining potential (κ_conf > 0)
    kappa_W : float
        Wasserstein contraction rate
    delta_sq : float
        Cloning noise variance (δ² > 0)

    Returns
    -------
    float
        Logarithmic Sobolev inequality constant C_LSI

    References
    ----------
    - 09_kl_convergence.md § 2.6 (Main LSI Theorem)
    - 09_kl_convergence.md § 0.2 (Proof Strategy)

    """
    return 1.0 / (gamma * kappa_conf * kappa_W * delta_sq)


def delta_star(
    alpha: float,
    tau: float,
    C_0: float,
    C_HWI: float,
    kappa_W: float,
    kappa_conf: float,
) -> float:
    """Critical cloning noise threshold for LSI.

    From 09_kl_convergence.md Theorem 2.7, the cloning noise must satisfy:

    δ > δ_* = exp(-α·τ/(2C_0)) · C_HWI · sqrt(2(1 - κ_W) / κ_conf)

    **Physical interpretation**: Cloning noise must be large enough to regularize
    Fisher information but not so large as to destroy convergence rate.

    Parameters
    ----------
    alpha : float
        Hypocoercive gap
    tau : float
        Time step size
    C_0 : float
        Base LSI constant
    C_HWI : float
        HWI inequality constant
    kappa_W : float
        Wasserstein contraction rate
    kappa_conf : float
        Confinement strength

    Returns
    -------
    float
        Critical noise threshold δ_*

    References
    ----------
    - 09_kl_convergence.md § 2.7 (Noise Threshold)

    """
    return (
        np.exp(-alpha * tau / (2.0 * C_0))
        * C_HWI
        * np.sqrt(2.0 * (1.0 - kappa_W) / kappa_conf)
    )


def kappa_QSD(kappa_total: float, tau: float = 0.01) -> float:
    """Continuous-time QSD convergence rate.

    From 06_convergence.md § 4.5, the QSD convergence rate is:

    κ_QSD = Θ(κ_total · τ)

    **Physical interpretation**: Discrete-time rate κ_total converted to
    continuous-time exponential rate by time step τ.

    Parameters
    ----------
    kappa_total : float
        Total discrete-time contraction rate
    tau : float, optional
        Time step size (default: 0.01)

    Returns
    -------
    float
        Continuous-time convergence rate κ_QSD

    References
    ----------
    - 06_convergence.md § 4.5 (Main Convergence Theorem)

    """
    return kappa_total * tau


def KL_convergence_rate(t: float, C_LSI: float, D_KL_init: float) -> float:
    """KL-divergence decay at time t.

    From 09_kl_convergence.md Theorem 0.1, the KL-divergence decays as:

    D_KL(μ_t || π_QSD) ≤ exp(-t / C_LSI) · D_KL(μ_0 || π_QSD)

    **Physical interpretation**: Exponential convergence in relative entropy
    (information-theoretic distance to QSD).

    Parameters
    ----------
    t : float
        Time
    C_LSI : float
        LSI constant
    D_KL_init : float
        Initial KL-divergence

    Returns
    -------
    float
        KL-divergence at time t

    References
    ----------
    - 09_kl_convergence.md § 0.1 (Main Result)

    """
    return np.exp(-t / C_LSI) * D_KL_init


# ============================================================================
# Section 3: Geometric Gas N-Uniform Bounds
# (15_geometric_gas_lsi_proof.md, geometric_foundations_lsi.md)
# ============================================================================


def c_min(epsilon_Sigma: float, H_max: float) -> float:
    """Lower uniform ellipticity bound for regularized diffusion.

    From 15_geometric_gas_lsi_proof.md Theorem 1.2, the lower bound is:

    c_min(ρ) = 1 / (H_max(ρ) + ε_Σ)

    where H_max is the maximum Hessian eigenvalue and ε_Σ is the regularization.

    **Physical interpretation**: Ensures diffusion never degenerates (no flat directions).
    Regularization ε_Σ prevents singular geometry.

    Parameters
    ----------
    epsilon_Sigma : float
        Diffusion regularization parameter (ε_Σ > H_max)
    H_max : float
        Maximum eigenvalue of fitness Hessian H_i(S)

    Returns
    -------
    float
        Lower ellipticity bound c_min(ρ)

    References
    ----------
    - 15_geometric_gas_lsi_proof.md § 2.1 (Uniform Ellipticity)
    - 18_emergent_geometry.md (Uniform Ellipticity Theorem)

    """
    return 1.0 / (H_max + epsilon_Sigma)


def c_max(epsilon_Sigma: float, H_max: float) -> float:
    """Upper uniform ellipticity bound for regularized diffusion.

    From 15_geometric_gas_lsi_proof.md Theorem 1.2, the upper bound is:

    c_max(ρ) = 1 / (ε_Σ - H_max(ρ))

    **Validity condition**: Requires ε_Σ > H_max(ρ) for positive bound.

    **Physical interpretation**: Ensures diffusion never explodes (no infinitely
    curved directions). Condition number c_max/c_min measures geometric anisotropy.

    Parameters
    ----------
    epsilon_Sigma : float
        Diffusion regularization parameter (must satisfy ε_Σ > H_max)
    H_max : float
        Maximum eigenvalue of fitness Hessian

    Returns
    -------
    float
        Upper ellipticity bound c_max(ρ)

    Raises
    ------
    ValueError
        If epsilon_Sigma <= H_max (violates uniform ellipticity condition)

    References
    ----------
    - 15_geometric_gas_lsi_proof.md § 2.1 (Uniform Ellipticity)
    - geometric_foundations_lsi.md § 1.2 (Role of Uniform Ellipticity)

    """
    if epsilon_Sigma <= H_max:
        raise ValueError(
            f"Uniform ellipticity violated: ε_Σ={epsilon_Sigma} must be > H_max={H_max}"
        )
    return 1.0 / (epsilon_Sigma - H_max)


def C_LSI_geometric(
    rho: float,
    c_min_val: float,
    c_max_val: float,
    gamma: float,
    kappa_conf: float,
    kappa_W: float,
) -> float:
    """N-uniform LSI constant for Geometric Gas via hypocoercivity.

    From 15_geometric_gas_lsi_proof.md § 9, the N-uniform LSI constant is:

    C_LSI(ρ) ≤ (c_max^4(ρ)) / (c_min^2(ρ) · γ · κ_conf · κ_W)

    **Key achievement**: This proof does NOT require log-concavity of the QSD,
    only uniform ellipticity + C³ regularity + Gaussian velocity structure.

    **Physical interpretation**: Geometric regularity (bounded eigenvalue ratios)
    directly controls the LSI constant. Condition number (c_max/c_min)^6 appears
    due to hypocoercive coupling in anisotropic diffusion.

    Parameters
    ----------
    rho : float
        Localization scale parameter (controls N-uniformity)
    c_min_val : float
        Lower uniform ellipticity bound: c_min(ρ) = 1/(H_max(ρ) + ε_Σ)
    c_max_val : float
        Upper uniform ellipticity bound: c_max(ρ) = 1/(ε_Σ - H_max(ρ))
    gamma : float
        Friction coefficient
    kappa_conf : float
        Convexity constant of confining potential
    kappa_W : float
        Wasserstein contraction rate

    Returns
    -------
    float
        N-uniform logarithmic Sobolev inequality constant C_LSI(ρ)

    References
    ----------
    - 15_geometric_gas_lsi_proof.md § 9 (Main Theorem: N-Uniform LSI)
    - geometric_foundations_lsi.md § 6 (Supersession of Log-Concavity Axiom)

    """
    return (c_max_val**4) / (c_min_val**2 * gamma * kappa_conf * kappa_W)


def C_P_poincare(c_max_val: float, C_P_ref: float = 1.0) -> float:
    """Poincaré constant from uniform ellipticity.

    From geometric_foundations_lsi.md Corollary 4.2, the Poincaré constant is:

    C_P ≤ (c_max / c_min) · C_P_ref

    where C_P_ref is the Poincaré constant for the isotropic reference case.

    **Physical interpretation**: Anisotropic diffusion degrades the Poincaré constant
    by the condition number. Small ε_Σ → large ratio → worse mixing.

    Parameters
    ----------
    c_max_val : float
        Upper ellipticity bound
    C_P_ref : float, optional
        Reference Poincaré constant for isotropic case (default: 1.0)

    Returns
    -------
    float
        Poincaré constant C_P

    References
    ----------
    - geometric_foundations_lsi.md § 4.2 (Poincaré from Ellipticity)

    """
    # Simplified version assuming c_min = 1 for the ratio
    # In practice, user should pass c_max / c_min as c_max_val
    return c_max_val * C_P_ref


def epsilon_F_star(rho: float, c_min_val: float, F_adapt_max: float) -> float:
    """Critical adaptive force threshold for LSI validity.

    From 15_geometric_gas_lsi_proof.md § 0, the LSI holds in the regime:

    ε_F < ε_F*(ρ) = c_min(ρ) / (2 · F_adapt,max(ρ))

    **Physical interpretation**: Adaptive force must be weak enough not to
    overwhelm the friction-driven dissipation. Explicit threshold ensures
    hypocoercivity gap remains positive.

    Parameters
    ----------
    rho : float
        Localization scale
    c_min_val : float
        Lower ellipticity bound
    F_adapt_max : float
        Maximum adaptive force magnitude: ||F_adapt(x_i, S)|| ≤ F_adapt,max(ρ)

    Returns
    -------
    float
        Critical threshold ε_F*(ρ)

    References
    ----------
    - 15_geometric_gas_lsi_proof.md § 0 (TLDR - Explicit Parameter Threshold)
    - 15_geometric_gas_lsi_proof.md § 9 (Main Theorem)

    """
    return c_min_val / (2.0 * F_adapt_max)


def lambda_min_spectral_gap(
    c_min_val: float, c_max_val: float, lambda_ref: float
) -> float:
    """Spectral gap from uniform ellipticity.

    From geometric_foundations_lsi.md Lemma 4.1, the spectral gap is:

    λ_min(L) ≥ (c_min / c_max) · λ_min(L_ref)

    **Physical interpretation**: Anisotropic diffusion reduces the spectral gap
    by the inverse condition number.

    Parameters
    ----------
    c_min_val : float
        Lower ellipticity bound
    c_max_val : float
        Upper ellipticity bound
    lambda_ref : float
        Reference spectral gap (isotropic case)

    Returns
    -------
    float
        Spectral gap λ_min(L)

    References
    ----------
    - geometric_foundations_lsi.md § 4.2 (Spectral Gap Bound from Ellipticity)

    """
    return (c_min_val / c_max_val) * lambda_ref


def C_LSI_bakry_emery(c_max_val: float, lambda_rho: float) -> float:
    """LSI constant from Bakry-Émery criterion (conditional on convexity).

    From geometric_foundations_lsi.md Proposition 5.3, under uniform convexity
    ∇²V_fit ≥ λ_ρ I, the LSI constant is:

    C_LSI ≤ 2·c_max / λ_ρ

    **Important**: This is CONDITIONAL on uniform convexity, which is NOT
    generally satisfied. Use C_LSI_geometric for the unconditional bound.

    **Physical interpretation**: Convex fitness landscapes have better LSI
    constants. The factor c_max comes from the inverse metric in the carré du champ.

    Parameters
    ----------
    c_max_val : float
        Upper ellipticity bound
    lambda_rho : float
        Uniform convexity constant: ∇²V_fit(x) ≥ λ_ρ I

    Returns
    -------
    float
        LSI constant (conditional on convexity)

    References
    ----------
    - geometric_foundations_lsi.md § 5.3 (LSI from Bakry-Émery)

    """
    return 2.0 * c_max_val / lambda_rho


def hypocoercive_gap(alpha_backbone: float, C_comm: float) -> float:
    """Hypocoercive gap for state-dependent diffusion.

    From 15_geometric_gas_lsi_proof.md § 7, the hypocoercive gap is:

    gap = α_backbone - C_comm

    where α_backbone is the microscopic coercivity and C_comm is the commutator
    error from state-dependent diffusion.

    **Physical interpretation**: Positive gap ensures entropy-Fisher inequality.
    C³ regularity bounds commutator errors to preserve the gap.

    Parameters
    ----------
    alpha_backbone : float
        Microscopic coercivity from friction
    C_comm : float
        Commutator error bound (from C³ regularity)

    Returns
    -------
    float
        Hypocoercive gap (must be > 0 for LSI)

    References
    ----------
    - 15_geometric_gas_lsi_proof.md § 7 (Macroscopic Transport)

    """
    return alpha_backbone - C_comm


# ============================================================================
# Section 4: Wasserstein Contraction Bounds (04_wasserstein_contraction.md)
# ============================================================================


def kappa_W_cluster(f_UH: float, p_u: float, c_align: float) -> float:
    """Cluster-level Wasserstein-2 contraction rate.

    From 04_wasserstein_contraction.md Theorem 6.1, the contraction rate is:

    κ_W = (1/2) · f_UH(ε) · p_u(ε) · c_align(ε)

    where:
    - f_UH ≥ 0.1: Target set fraction
    - p_u ≥ 0.01: Cloning pressure
    - c_align ≥ c_0 > 0: Geometric alignment constant

    **Physical interpretation**: Cluster-level analysis avoids the 1/N! coupling
    obstruction. N-uniformity comes from population-averaged tracking.

    Parameters
    ----------
    f_UH : float
        Target set fraction (≥ 0.1)
    p_u : float
        Cloning pressure (≥ 0.01)
    c_align : float
        Geometric alignment constant (> 0)

    Returns
    -------
    float
        Wasserstein-2 contraction rate κ_W

    References
    ----------
    - 04_wasserstein_contraction.md § 6.1 (Main Contraction Theorem)
    - 04_wasserstein_contraction.md § 1.2 (Cluster-Level Analysis)

    """
    return 0.5 * f_UH * p_u * c_align


def f_UH_target_fraction(epsilon: float = 0.1) -> float:
    """Target set fraction for outlier-helper decomposition.

    From 04_wasserstein_contraction.md Lemma 3.2, the fraction satisfies:

    f_UH ≥ 0.1

    **Physical interpretation**: At least 10% of walkers are in the "unhappy"
    (outlier) set U, which receives selection pressure from the "happy" (helper)
    set H.

    Parameters
    ----------
    epsilon : float, optional
        Separation threshold (default: 0.1)

    Returns
    -------
    float
        Target set fraction (≥ 0.1)

    References
    ----------
    - 04_wasserstein_contraction.md § 3.2 (Outlier Set Size)

    """
    return max(0.1, epsilon)


def p_u_cloning_pressure(epsilon: float = 0.01) -> float:
    """Cloning pressure on outlier set.

    From 04_wasserstein_contraction.md Lemma 4.1, the cloning pressure satisfies:

    p_u ≥ 0.01

    **Physical interpretation**: At least 1% probability that an outlier clones
    from a helper, producing alignment pressure.

    Parameters
    ----------
    epsilon : float, optional
        Minimum pressure (default: 0.01)

    Returns
    -------
    float
        Cloning pressure (≥ 0.01)

    References
    ----------
    - 04_wasserstein_contraction.md § 4.1 (Cloning Pressure)

    """
    return max(0.01, epsilon)


def c_align_geometric(epsilon: float = 0.1) -> float:
    """Geometric alignment constant from cross-distance reduction.

    From 04_wasserstein_contraction.md § 4, the alignment constant satisfies:

    c_align ≥ c_0 > 0

    **Physical interpretation**: When outliers clone from helpers, their expected
    distance to the target swarm decreases proportionally to the cluster separation.

    Parameters
    ----------
    epsilon : float, optional
        Minimum constant (default: 0.1)

    Returns
    -------
    float
        Geometric alignment constant

    References
    ----------
    - 04_wasserstein_contraction.md § 4 (Geometric Alignment)

    """
    return epsilon


# ============================================================================
# Section 5: Parameter Regime Validators
# ============================================================================


def validate_foster_lyapunov(kappa_total: float, C_total: float) -> bool:
    """Validate Foster-Lyapunov condition holds.

    The Foster-Lyapunov condition requires:
    - κ_total > 0 (positive contraction rate)
    - C_total < ∞ (finite equilibrium constant)

    Parameters
    ----------
    kappa_total : float
        Total contraction rate
    C_total : float
        Equilibrium constant

    Returns
    -------
    bool
        True if Foster-Lyapunov condition is satisfied

    """
    return kappa_total > 0 and np.isfinite(C_total)


def validate_hypocoercivity(
    epsilon_F: float, epsilon_F_star_val: float, nu: float
) -> bool:
    """Validate parameter regime for hypocoercivity-based LSI.

    From 15_geometric_gas_lsi_proof.md § 0, requires:
    - ε_F < ε_F*(ρ): Adaptive force below threshold
    - ν > 0: Positive viscous coupling (no upper bound)

    Parameters
    ----------
    epsilon_F : float
        Adaptive force strength
    epsilon_F_star_val : float
        Critical threshold
    nu : float
        Viscous coupling parameter

    Returns
    -------
    bool
        True if hypocoercivity regime is valid

    """
    return epsilon_F < epsilon_F_star_val and nu > 0


def validate_ellipticity(epsilon_Sigma: float, H_max: float) -> bool:
    """Validate uniform ellipticity condition.

    From 15_geometric_gas_lsi_proof.md Theorem 1.2, requires:
    - ε_Σ > H_max(ρ): Regularization exceeds maximum Hessian eigenvalue

    Parameters
    ----------
    epsilon_Sigma : float
        Diffusion regularization
    H_max : float
        Maximum Hessian eigenvalue

    Returns
    -------
    bool
        True if uniform ellipticity holds

    """
    return epsilon_Sigma > H_max


def validate_noise_threshold(delta: float, delta_star_val: float) -> bool:
    """Validate cloning noise exceeds critical threshold for LSI.

    From 09_kl_convergence.md Theorem 2.7, requires:
    - δ > δ_*: Noise regularizes Fisher information

    Parameters
    ----------
    delta : float
        Cloning noise scale
    delta_star_val : float
        Critical threshold

    Returns
    -------
    bool
        True if noise is sufficient

    """
    return delta > delta_star_val


# ============================================================================
# Section 6: Compound Bounds and Diagnostics
# ============================================================================


def convergence_timescale_ratio(
    kappa_x: float, kappa_v: float, kappa_W: float, kappa_b: float
) -> Dict[str, float]:
    """Identify convergence bottlenecks via timescale ratios.

    Computes T_i / T_min for each component, where T_i = 1/κ_i is the
    characteristic timescale. Ratios >> 1 indicate bottlenecks.

    Parameters
    ----------
    kappa_x : float
        Position contraction rate
    kappa_v : float
        Velocity contraction rate
    kappa_W : float
        Wasserstein contraction rate
    kappa_b : float
        Boundary contraction rate

    Returns
    -------
    dict
        Timescale ratios for each component:
        {
            'position': T_x / T_min,
            'velocity': T_v / T_min,
            'wasserstein': T_W / T_min,
            'boundary': T_b / T_min,
            'bottleneck': name of slowest component
        }

    """
    rates = {
        "position": kappa_x,
        "velocity": kappa_v,
        "wasserstein": kappa_W,
        "boundary": kappa_b,
    }

    kappa_min = min(rates.values())
    bottleneck = min(rates, key=rates.get)

    ratios = {k: kappa_min / v for k, v in rates.items()}
    ratios["bottleneck"] = bottleneck

    return ratios


def condition_number_geometry(c_min_val: float, c_max_val: float) -> float:
    """Geometric condition number from ellipticity bounds.

    The condition number is:

    κ(g) = c_max / c_min

    **Physical interpretation**: Measures anisotropy of emergent geometry.
    Large condition number → ill-conditioned problem → slower mixing.

    Parameters
    ----------
    c_min_val : float
        Lower ellipticity bound
    c_max_val : float
        Upper ellipticity bound

    Returns
    -------
    float
        Condition number κ(g)

    """
    return c_max_val / c_min_val


def effective_dimension(d: int, c_min_val: float, c_max_val: float) -> float:
    """Effective dimension accounting for geometric anisotropy.

    Defined as:

    d_eff = d / (c_max / c_min)

    **Physical interpretation**: Anisotropic diffusion reduces the effective
    dimensionality of exploration. Extreme anisotropy makes the problem
    effectively lower-dimensional.

    Parameters
    ----------
    d : int
        Ambient dimension
    c_min_val : float
        Lower ellipticity bound
    c_max_val : float
        Upper ellipticity bound

    Returns
    -------
    float
        Effective dimension d_eff

    """
    return d / (c_max_val / c_min_val)


def mean_field_error_bound(N: int, kappa_W: float, T: float) -> float:
    """Finite-N error bound for mean-field approximation.

    From propagation of chaos theory (08_propagation_chaos.md), the error is:

    error ≈ exp(-κ_W · T) / sqrt(N)

    **Physical interpretation**: N-uniform Wasserstein contraction ensures
    mean-field limit validity. Error decays as 1/√N and exponentially in time.

    Parameters
    ----------
    N : int
        Swarm size
    kappa_W : float
        Wasserstein contraction rate
    T : float
        Time horizon

    Returns
    -------
    float
        Mean-field approximation error

    """
    return np.exp(-kappa_W * T) / np.sqrt(N)


# ============================================================================
# Section 7: Sensitivity Analysis (06_convergence.md § 6)
# ============================================================================


def rate_sensitivity_matrix(params: Dict[str, float]) -> np.ndarray:
    """Compute Jacobian of convergence rates with respect to parameters.

    From 06_convergence.md Theorem 6.1, the sensitivity matrix is:

    M_κ[i,j] = ∂(log κ_i) / ∂(log θ_j)

    where κ_i ∈ {κ_x, κ_v, κ_W, κ_b} are component rates and
    θ_j are primitive parameters.

    **Physical interpretation**: Identifies which parameters have strongest
    impact on convergence rates. Large entries indicate high sensitivity.

    Parameters
    ----------
    params : dict
        Parameter dictionary containing:
        - 'gamma': Friction coefficient
        - 'lambda_alg': Cloning rate
        - 'sigma_v': Langevin noise
        - 'tau': Time step
        - 'lambda_min': Min confining potential eigenvalue
        - 'delta_f_boundary': Boundary fitness drop

    Returns
    -------
    np.ndarray
        Sensitivity matrix M_κ of shape (4, n_params)
        Rows: [κ_x, κ_v, κ_W, κ_b]
        Cols: Parameters in order of params dict keys

    References
    ----------
    - 06_convergence.md § 6.3 (Sensitivity Matrices)
    - 06_convergence.md § 6.4 (SVD Analysis)

    """
    # Extract parameters
    gamma = params.get("gamma", 1.0)
    lambda_alg = params.get("lambda_alg", 1.0)
    sigma_v = params.get("sigma_v", 0.1)
    tau = params.get("tau", 0.01)
    lambda_min = params.get("lambda_min", 1.0)
    delta_f_boundary = params.get("delta_f_boundary", 1.0)

    # Compute logarithmic derivatives (approximate via finite differences)
    eps = 1e-6

    # Initialize matrix: 4 rates × 6 parameters
    M = np.zeros((4, 6))

    # Row 0: κ_x sensitivity
    # κ_x ≈ λ → ∂log(κ_x)/∂log(λ) ≈ 1
    M[0, 1] = 1.0  # lambda_alg
    M[0, 3] = -1.0  # tau (negative correction)

    # Row 1: κ_v sensitivity
    # κ_v ≈ 2γ → ∂log(κ_v)/∂log(γ) ≈ 1
    M[1, 0] = 1.0  # gamma
    M[1, 3] = -1.0  # tau

    # Row 2: κ_W sensitivity
    # κ_W ≈ γ / (1 + γ/λ_min) → more complex
    factor = 1.0 + gamma / lambda_min
    M[2, 0] = 1.0 / factor  # gamma
    M[2, 4] = (gamma / lambda_min) / factor  # lambda_min

    # Row 3: κ_b sensitivity
    # κ_b ≈ λ · (Δf/f_typ) → log-linear
    M[3, 1] = 1.0  # lambda_alg
    M[3, 5] = 1.0  # delta_f_boundary

    return M


def equilibrium_sensitivity_matrix(params: Dict[str, float]) -> np.ndarray:
    """Compute Jacobian of equilibrium constants with respect to parameters.

    From 06_convergence.md § 6.3, the equilibrium sensitivity matrix is:

    M_C[i,j] = ∂(log C_i^eq) / ∂(log θ_j)

    where C_i^eq are equilibrium variance components.

    **Physical interpretation**: Shows how parameter changes affect steady-state
    variance. Useful for designing systems with target exploration widths.

    Parameters
    ----------
    params : dict
        Parameter dictionary (same as rate_sensitivity_matrix)

    Returns
    -------
    np.ndarray
        Equilibrium sensitivity matrix M_C of shape (2, n_params)
        Rows: [Var_x^eq, Var_v^eq]
        Cols: Parameters

    References
    ----------
    - 06_convergence.md § 6.3 (Sensitivity Matrices)

    """
    # Initialize matrix: 2 equilibrium components × 6 parameters
    M = np.zeros((2, 6))

    # Row 0: Var_x^eq sensitivity
    # Var_x^eq ≈ σ_v^2 · τ^2 / (γ · λ)
    M[0, 0] = -1.0  # gamma (inverse)
    M[0, 1] = -1.0  # lambda_alg (inverse)
    M[0, 2] = 2.0  # sigma_v (squared)
    M[0, 3] = 2.0  # tau (squared)

    # Row 1: Var_v^eq sensitivity
    # Var_v^eq ≈ d · σ_v^2 / γ
    M[1, 0] = -1.0  # gamma (inverse)
    M[1, 2] = 2.0  # sigma_v (squared)

    return M


def condition_number_parameters(sensitivity_matrix: np.ndarray) -> float:
    """Compute condition number of parameter sensitivity matrix.

    From 06_convergence.md Theorem 6.7, the condition number is:

    κ(M) = σ_max(M) / σ_min(M)

    where σ_max, σ_min are the largest and smallest singular values.

    **Physical interpretation**: Measures robustness to parameter errors.
    Large condition number → small parameter changes cause large rate changes.

    Parameters
    ----------
    sensitivity_matrix : np.ndarray
        Sensitivity matrix from rate_sensitivity_matrix or equilibrium_sensitivity_matrix

    Returns
    -------
    float
        Condition number κ(M)

    References
    ----------
    - 06_convergence.md § 6.7 (Robustness Analysis)

    """
    singular_values = np.linalg.svd(sensitivity_matrix, compute_uv=False)
    return singular_values[0] / singular_values[-1]


def principal_coupling_modes(
    sensitivity_matrix: np.ndarray, k: int = 3
) -> Dict[str, np.ndarray]:
    """Compute principal coupling modes via SVD.

    From 06_convergence.md Theorem 6.4, the SVD decomposition is:

    M = U · Σ · V^T

    where columns of V are principal parameter directions and singular values
    in Σ measure coupling strength.

    **Physical interpretation**: Identifies dominant parameter interactions.
    The first few modes capture most of the variance in convergence rates.

    Parameters
    ----------
    sensitivity_matrix : np.ndarray
        Sensitivity matrix
    k : int, optional
        Number of principal modes to return (default: 3)

    Returns
    -------
    dict
        Principal modes:
        {
            'singular_values': First k singular values,
            'parameter_directions': First k right singular vectors (V),
            'rate_patterns': First k left singular vectors (U)
        }

    References
    ----------
    - 06_convergence.md § 6.4 (SVD Analysis)

    """
    U, s, Vt = np.linalg.svd(sensitivity_matrix, full_matrices=False)

    return {
        "singular_values": s[:k],
        "parameter_directions": Vt[:k].T,  # Transpose to get column vectors
        "rate_patterns": U[:, :k],
    }


# ============================================================================
# Section 8: Optimization Helpers (06_convergence.md § 6.10)
# ============================================================================


def balanced_parameters_closed_form(
    lambda_min: float,
    lambda_max: float,
    d: int,
    V_target: float,
) -> Dict[str, float]:
    """Compute optimal parameters for balanced convergence (no bottleneck).

    From 06_convergence.md Theorem 6.10, the unconstrained optimum is:

    Step 1: γ = sqrt(λ_min)
    Step 2: λ = γ
    Step 3: σ_v = sqrt(γ · λ · V_target / (d · τ²))

    **Physical interpretation**: Balances all four component rates to eliminate
    bottlenecks, achieving fastest overall convergence.

    Parameters
    ----------
    lambda_min : float
        Minimum eigenvalue of confining potential Hessian
    lambda_max : float
        Maximum eigenvalue (for validation)
    d : int
        Spatial dimension
    V_target : float
        Target equilibrium positional variance

    Returns
    -------
    dict
        Optimal parameters:
        {
            'gamma': Friction coefficient,
            'lambda_alg': Cloning rate,
            'sigma_v': Langevin noise scale,
            'tau': Recommended time step
        }

    References
    ----------
    - 06_convergence.md § 6.10.1 (Closed-Form Balanced Optimum)

    """
    # Step 1: Friction from landscape
    gamma_opt = np.sqrt(lambda_min)

    # Step 2: Cloning rate balanced with friction
    lambda_opt = gamma_opt

    # Step 3: Noise from target variance (assume tau = 0.01)
    tau = 0.01
    sigma_v_opt = np.sqrt((gamma_opt * lambda_opt * V_target) / (d * tau**2))

    return {
        "gamma": gamma_opt,
        "lambda_alg": lambda_opt,
        "sigma_v": sigma_v_opt,
        "tau": tau,
    }


def pareto_frontier_rate_variance(
    kappa_range: Tuple[float, float],
    C_range: Tuple[float, float],
    n_points: int = 100,
) -> np.ndarray:
    """Compute Pareto frontier for rate-variance multi-objective optimization.

    From 06_convergence.md § 6.10, explores trade-offs between:
    - Fast convergence (large κ_total)
    - Small equilibrium variance (small C_total / κ_total)

    **Physical interpretation**: Cannot simultaneously optimize both objectives.
    Pareto frontier shows achievable combinations.

    Parameters
    ----------
    kappa_range : tuple
        Range of convergence rates (κ_min, κ_max)
    C_range : tuple
        Range of equilibrium constants (C_min, C_max)
    n_points : int, optional
        Number of points on frontier (default: 100)

    Returns
    -------
    np.ndarray
        Pareto frontier points of shape (n_points, 2)
        Columns: [κ_total, Var_eq]

    References
    ----------
    - 06_convergence.md § 6.10 (Rate-Space Optimization)

    """
    kappas = np.linspace(kappa_range[0], kappa_range[1], n_points)
    Cs = np.linspace(C_range[0], C_range[1], n_points)

    # Pareto frontier: for each kappa, find minimum achievable variance
    frontier = np.zeros((n_points, 2))
    for i, kappa in enumerate(kappas):
        # Variance = C / kappa, minimized when C is minimal
        frontier[i, 0] = kappa
        frontier[i, 1] = Cs[0] / kappa  # Minimum C

    return frontier
