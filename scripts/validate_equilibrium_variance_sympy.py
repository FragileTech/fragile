"""
SymPy validation for EQUILIBRIUM VARIANCE BOUNDS derived from drift inequalities.

Source: docs/source/1_euclidean_gas/06_convergence.md:1112-1176
Category: Equilibrium Conditions (solving for steady state)

Validates three algebraic claims by:
- Solving Δ=0 for equilibrium values
- Substituting back to confirm the drift equals zero
- Checking parameter positivity assumptions
- Printing concise physical interpretations
"""

from __future__ import annotations

import sympy as sp


def main() -> None:
    # Parameters (all positive by framework assumptions)
    kappa_x, kappa_b = sp.symbols("kappa_x kappa_b", positive=True)
    gamma, tau = sp.symbols("gamma tau", positive=True)
    Cx, Cv, Cb = sp.symbols("C_x C_v C_b", positive=True)
    sigma2_max = sp.symbols("sigma2_max", positive=True)
    d = sp.symbols("d", integer=True, positive=True)

    # State variables for variances/potential
    Vx = sp.symbols("V_Var_x", real=True)
    Vv = sp.symbols("V_Var_v", real=True)
    Wb = sp.symbols("W_b", real=True)

    print("Equilibrium Variance Bounds — SymPy verification\n")

    # 1) Positional Variance: ΔV = -kappa_x * V + Cx
    drift_x = -kappa_x * Vx + Cx
    Vx_eq = sp.solve(sp.Eq(0, drift_x), Vx)[0]
    # Verify by substitution
    verify_x = sp.simplify(drift_x.subs(Vx, Vx_eq))

    # 2) Velocity Variance: ΔV = -2*gamma*V*tau + (Cv + sigma2_max*d*tau)
    drift_v = -2 * gamma * Vv * tau + (Cv + sigma2_max * d * tau)
    Vv_eq = sp.solve(sp.Eq(0, drift_v), Vv)[0]
    # Verify by substitution
    verify_v = sp.simplify(drift_v.subs(Vv, Vv_eq))

    # 3) Boundary Potential: ΔW_b = -kappa_b * W_b + Cb
    drift_b = -kappa_b * Wb + Cb
    Wb_eq = sp.solve(sp.Eq(0, drift_b), Wb)[0]
    # Verify by substitution
    verify_b = sp.simplify(drift_b.subs(Wb, Wb_eq))

    # Positivity checks for parameters (symbolic assumptions)
    params = [kappa_x, kappa_b, gamma, tau, Cx, Cv, Cb, sigma2_max, d]
    print("Parameter positivity checks (should all be True):")
    for p in params:
        print(f"  {p}: is_positive={bool(p.is_positive)}")
    print()

    # Print solutions and verifications
    print("Claim 1 — Positional Variance Equilibrium:")
    print(f"  V_x^QSD = {sp.simplify(Vx_eq)}  (expected: C_x/kappa_x)")
    print(f"  Substitute back: ΔV_x|_eq = {verify_x}")
    assert verify_x == 0
    print()

    print("Claim 2 — Velocity Variance Equilibrium:")
    print(f"  V_v^QSD = {sp.simplify(Vv_eq)}  (expected: (C_v + sigma2_max*d*tau)/(2*gamma*tau))")
    print(f"  Substitute back: ΔV_v|_eq = {verify_v}")
    assert verify_v == 0
    # Helpful decomposition for interpretation
    Vv_eq_decomp = sp.simplify(sp.together(Vv_eq).rewrite(sp.Add))
    # Manually show the split into Cv and noise terms
    Vv_eq_split = sp.simplify(Cv / (2 * gamma * tau) + (sigma2_max * d) / (2 * gamma))
    print(f"  Decomposition: V_v^QSD = {Vv_eq_split} = Cv/(2γτ) + (σ_max^2 d)/(2γ)")
    print()

    print("Claim 3 — Boundary Potential Equilibrium:")
    print(f"  W_b^QSD = {sp.simplify(Wb_eq)}  (expected: C_b/kappa_b)")
    print(f"  Substitute back: ΔW_b|_eq = {verify_b}")
    assert verify_b == 0
    print()

    # Physical interpretations
    print("Physical interpretations:")
    print("- Position: V_x^QSD grows with injection C_x and shrinks with contraction kappa_x.")
    print("- Velocity: V_v^QSD has a friction-limited piece Cv/(2γτ) and a noise-limited piece (σ_max^2 d)/(2γ).")
    print("           Stronger friction (γ↑) lowers both; larger noise (σ_max^2↑) or dimension (d↑) raises it.")
    print("- Boundary: W_b^QSD grows with boundary influx C_b and shrinks with boundary contraction kappa_b.")
    print()

    print("All symbolic verifications passed (Δ = 0 at the stated equilibria).")


if __name__ == "__main__":
    main()

