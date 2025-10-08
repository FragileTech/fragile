# Theorems Reference Table

## Volume: Hydrodynamics and Fluid Dynamics

| ID | Theorem | TLDR | Source | Proof | Notes |
|----|---------|------|--------|-------|-------|
| <a id="t-chapman-enskog-corrected">T-9.1</a> | Chapman-Enskog Closure (Corrected) | Rigorous derivation of Navier-Stokes from kinetic theory with proper momentum-conserving collision operator | [Chapter 9](../09_hydrodynamics.md#chapman-enskog) | [Proof](../theorems/proofs/9_01_chapman_enskog_closure_proof_CORRECTED.md) | Corrects previous error; uses friction toward local mean velocity |

## Key Results

### Theorem 9.1: Chapman-Enskog Closure (Corrected)

**Statement**: For the scaled kinetic equation with the corrected collision operator $\mathcal{L}[f] = \nabla_v \cdot [(v - \mathbf{u}[f](x)) f] + T \Delta_v f$, where $\mathbf{u}[f]$ is the self-consistent local mean velocity, the solution converges to a local Maxwellian:

$$\|f^\epsilon(t,x,v) - \rho(t,x) M_T(v - \mathbf{u}(t,x))\|_{L^1} \leq C \epsilon$$

where $(\rho, \mathbf{u})$ satisfy the incompressible Navier-Stokes equations with viscosity $\mu = \rho T/\gamma$.

**Significance**: This theorem establishes the rigorous connection between microscopic particle dynamics and macroscopic fluid behavior, with the critical correction that friction acts toward the local mean velocity (not zero), ensuring proper momentum conservation and Galilean invariance.

**Key Innovation**: The proof handles the nonlinearity introduced by the self-consistent velocity field $\mathbf{u}[f]$ using:
- Fixed-point theory for existence and uniqueness
- Hypocoercivity for spectral gap analysis
- Relative entropy methods for convergence estimates

**Physical Importance**: Only this corrected operator:
1. Conserves momentum locally
2. Yields proper Navier-Stokes (not Stokes) equations
3. Respects Galilean invariance
4. Matches the BGK model structure from kinetic theory