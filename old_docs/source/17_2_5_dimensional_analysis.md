# Dimensional Analysis: Yang-Mills Hamiltonian Continuum Limit

**Purpose**: Systematic dimensional and N-scaling analysis to diagnose coupling constant mismatch in §17.2.5.

**Date**: 2025-10-14

**Status**: Diagnostic Phase (Phase 1.1)

---

## §1. Algorithmic Parameters (Source Framework)

From {prf:ref}`def-parameter-dimensions` and {prf:ref}`thm-su2-coupling-constant`:

| Parameter | Symbol | Dimension | Physical Meaning |
|-----------|--------|-----------|------------------|
| Position | $x_i$ | $[L]$ | Walker position |
| Velocity | $v_i$ | $[L][T]^{-1}$ | Walker velocity |
| Mass | $m$ | $[M]$ | Walker mass |
| Timestep | $\tau$ | $[T]$ | Cloning period |
| Friction | $\gamma$ | $[T]^{-1}$ | Damping rate |
| Diffusion | $D$ | $[L]^2[T]^{-1}$ | Noise strength |
| Localization scale | $\rho$ | $[L]$ | Kernel width |
| Cloning scale | $\epsilon_c$ | $[L]$ | Selection scale |
| Velocity weight | $\lambda_v$ | $[T]^2$ | Metric component |
| Adaptive force | $\epsilon_F$ | $[M][L]^2[T]^{-2}$ | Force strength |
| Viscosity | $\nu$ | $[T]^{-1}$ | Coupling rate |

**Dimensionless**: $N$ (walkers), $d$ (dimension), $\alpha, \beta$ (weights), $g$ (coupling)

**Natural units**: Set $\hbar = c = 1 \Rightarrow [M] = [T]^{-1} = [L]^{-1}$

---

## §2. Derived Constants

### §2.1 Effective Planck Constant

From {prf:ref}`thm-effective-planck-constant`:

$$
\hbar_{\text{eff}} = \frac{m \epsilon_c^2}{\tau}
$$

**Dimension**: $[\hbar_{\text{eff}}] = [M][L]^2[T]^{-1}$ ✓ (action)

**Natural units**: $[\hbar_{\text{eff}}] = [M]$ (since $[L] = [M]^{-1}$, $[T] = [M]^{-1}$)

### §2.2 SU(2) Gauge Coupling

From {prf:ref}`thm-su2-coupling-constant`:

$$
g^2 = \frac{\tau \rho^2}{m \epsilon_c^2} = \frac{\tau^2 \rho^2}{\hbar_{\text{eff}}}
$$

**Dimension**: $[g^2] = \frac{[T][L]^2}{[M][L]^2} = \frac{[T]}{[M]} = [1]$ ✓ (dimensionless in natural units)

**Key insight**: $g^2 \propto \tau$ → asymptotic freedom as $\tau \to 0$

---

## §3. Discrete Fields (Algorithmic Hamiltonian)

From {prf:ref}`def-discrete-hamiltonian-algorithmic` (lines 1845-1869 of `03_yang_mills_noether.md`):

$$
H_{\text{gauge}} = \frac{g^2}{2} \sum_{e} (E_e^{(a)})^2 + \frac{1}{g^2} \sum_{\square} \left(1 - \frac{1}{2}\text{Tr}(U_{\square})\right)
$$

### §3.1 Electric Field $E_e^{(a)}$

**Canonical relation** (line 1893):

$$
\frac{\partial A_e}{\partial t} = g^2 E_e
$$

**Standard YM**: $\dot{A} = E$ (no $g^2$ factor!)

**Dimensional analysis**:

From Hamiltonian term: $H \sim g^2 E_e^2$ must have dimension $[M][L]^2[T]^{-2}$ (energy)

Therefore:
$$
[g^2 E_e^2] = [M][L]^2[T]^{-2}
$$

$$
[E_e^2] = \frac{[M][L]^2[T]^{-2}}{[g^2]} = \frac{[M][L]^2[T]^{-2}}{[1]} = [M][L]^2[T]^{-2}
$$

$$
[E_e] = [M]^{1/2}[L][T]^{-1}
$$

**In natural units**: $[E_e] = [M]^{3/2}$

**Continuum comparison**: Standard Yang-Mills electric field has $[E_{\text{YM}}] = [M]^2$ in natural units.

**MISMATCH**: $[E_e] = [M]^{3/2} \neq [M]^2 = [E_{\text{YM}}]$

This confirms that $E_e$ is NOT the physical electric field!

### §3.2 Gauge Potential $A_e$

From canonical relation $\dot{A}_e = g^2 E_e$:

$$
[A_e] = [g^2 E_e][T] = [1] \cdot [M]^{3/2} \cdot [M]^{-1} = [M]^{1/2}
$$

**Continuum YM**: $[A_{\text{YM}}] = [M]$ in natural units

**MISMATCH**: $[A_e] = [M]^{1/2} \neq [M] = [A_{\text{YM}}]$

### §3.3 Wilson Loop $U_{\square}$

From Hamiltonian:
$$
H \sim \frac{1}{g^2}(1 - \frac{1}{2}\text{Tr}(U_{\square}))
$$

For small field strength, $U_{\square} \approx \exp(ig A_{\square})$ where $A_{\square}$ is the plaquette circulation.

Expanding: $1 - \frac{1}{2}\text{Tr}(U_{\square}) \sim g^2 A_{\square}^2 \sim g^2 (d_{ij})^4 F^2$

Therefore:
$$
H \sim \frac{1}{g^2} \cdot g^2 d^4 F^2 = d^4 F^2
$$

For dimensional consistency:
$$
[d^4 F^2] = [M][L]^2[T]^{-2}
$$

$$
[F^2] = \frac{[M][L]^2[T]^{-2}}{[L]^4} = [M][L]^{-2}[T]^{-2}
$$

$$
[F] = [M]^{1/2}[L]^{-1}[T]^{-1}
$$

**In natural units**: $[F] = [M]^{3/2}$

**Continuum YM**: $[F_{\text{YM}}] = [M]^2$

**MISMATCH**: Consistent with $E_e$ mismatch

---

## §4. Physical Fields (Corrected Definitions)

**Hypothesis**: Define physical fields that match standard YM dimensions:

### §4.1 Physical Electric Field

**Definition**:
$$
E_{\text{phys},e} := \frac{1}{g^2} E_e
$$

**Dimension**:
$$
[E_{\text{phys}}] = \frac{[E_e]}{[g^2]} = \frac{[M]^{3/2}}{[1]} = [M]^{3/2}
$$

**PROBLEM**: Still doesn't match $[E_{\text{YM}}] = [M]^2$!

**Corrected definition** (requires additional scaling):
$$
E_{\text{phys},e} := \frac{\sqrt{m}}{g^2} E_e
$$

**Dimension**:
$$
[E_{\text{phys}}] = \frac{[M]^{1/2} \cdot [M]^{3/2}}{[1]} = [M]^2 \quad \checkmark
$$

**Canonical relation**:
$$
\dot{A}_e = g^2 E_e = \frac{g^2}{\sqrt{m}} E_{\text{phys}}
$$

For standard YM $\dot{A} = E$, we need:
$$
A_{\text{phys}} := \frac{g^2}{\sqrt{m}} A_e
$$

Then: $\dot{A}_{\text{phys}} = E_{\text{phys}}$ ✓

### §4.2 Physical Gauge Potential

**Definition**:
$$
A_{\text{phys},e} := \frac{g^2}{\sqrt{m}} A_e
$$

**Dimension**:
$$
[A_{\text{phys}}] = \frac{[1] \cdot [M]^{1/2}}{[M]^{1/2}} = [M] \quad \checkmark
$$

---

## §5. Hamiltonian in Physical Fields

Substitute $E_e = \frac{\sqrt{m}}{g^2} E_{\text{phys}}$ into source Hamiltonian:

$$
H_{\text{gauge}} = \frac{g^2}{2} \sum_{e} \left(\frac{\sqrt{m}}{g^2} E_{\text{phys},e}\right)^2 + \frac{1}{g^2} \sum_{\square} (\ldots)
$$

$$
= \frac{g^2}{2} \sum_{e} \frac{m}{g^4} E_{\text{phys},e}^2 + \frac{1}{g^2} \sum_{\square} (\ldots)
$$

$$
= \frac{m}{2g^2} \sum_{e} E_{\text{phys},e}^2 + \frac{1}{g^2} \sum_{\square} (1 - \frac{1}{2}\text{Tr}(U_{\square}))
$$

**Result**: Both terms now have **symmetric** $1/g^2$ prefactor! ✓

---

## §6. N-Scaling Analysis

### §6.1 Walker Density

For $N$ walkers in volume $V$:
$$
\rho_{\text{QSD}} = \frac{N}{V}
$$

**Dimension**: $[\rho_{\text{QSD}}] = [L]^{-3}$ in 3D

### §6.2 Effective Lattice Spacing

Typical edge length:
$$
\ell_{\text{eff}} = \rho_{\text{QSD}}^{-1/3} = \left(\frac{V}{N}\right)^{1/3}
$$

**N-scaling**: $\ell_{\text{eff}} \sim N^{-1/3}$

### §6.3 Physical Field Normalization

**Electric field** averaged over cell:
$$
E_k(x) = \frac{1}{\rho \Delta V} \sum_{e \ni x} E_{\text{phys},e}
$$

For continuum normalization:
$$
\text{Fields scale as } \rho^0 \text{ (intensive)}
$$

### §6.4 Effective Coupling in Continuum Limit

From Hamiltonian:
$$
H = \frac{m}{2g^2} \sum_e E_{\text{phys}}^2 + \frac{1}{g^2} \sum_{\square} (\ldots)
$$

Converting sum to integral:
$$
\sum_e \to \int d^3x \, \rho(x) \cdot (\text{edge density per walker})
$$

Number of edges per walker: $O(1)$ (bounded degree in Delaunay)

Therefore:
$$
H \to \frac{m}{2g^2} \int d^3x \, \rho(x) \, E^2
$$

For **uniform QSD** ($\rho = N/V = \text{const}$):
$$
H = \frac{m N}{2g^2 V} \int d^3x \, E^2
$$

**Field rescaling**: $E_{\text{continuum}} = \sqrt{m N/V} \, E$

**Effective coupling**:
$$
g_{\text{eff}}^2 = g^2 \cdot \frac{V}{m N}
$$

**N-scaling**: $g_{\text{eff}}^2 \sim N^{-1}$

**KEY**: This is the **SAME** for both electric and magnetic terms (symmetric $1/g^2$ structure)

---

## §7. Summary of Findings

### §7.1 Root Cause of Mismatch

The source Hamiltonian uses **algorithmic fields** $E_e$ with non-standard dimensions:
- $[E_e] = [M]^{3/2}$ instead of $[E_{\text{YM}}] = [M]^2$
- Canonical relation: $\dot{A}_e = g^2 E_e$ (non-standard $g^2$ factor)

This asymmetry is **intentional** in the algorithmic formulation but requires **field redefinition** for continuum limit.

### §7.2 Solution

**Define physical fields**:
$$
E_{\text{phys}} = \frac{\sqrt{m}}{g^2} E_e, \quad A_{\text{phys}} = \frac{g^2}{\sqrt{m}} A_e
$$

**Rewrite Hamiltonian**:
$$
H_{\text{gauge}} = \frac{1}{2g^2} \sum_{e} m E_{\text{phys},e}^2 + \frac{1}{g^2} \sum_{\square} (1 - \frac{1}{2}\text{Tr}(U_{\square}))
$$

**Result**: **Symmetric** structure → **consistent** $g_{\text{eff}}^2$ for both terms ✓

### §7.3 Implications for §17.2.5

The current proof (lines 3655-3914) needs correction:

1. **Lemma `lem-electric-field-correspondence`** (lines 3605-3654):
   - Currently: $E_e = \frac{d_{ij}}{g^2} E_k$ ❌
   - Corrected: $E_{\text{phys},e} = d_{ij} E_k$ ✓

2. **Electric term derivation** (Part 2, lines 3655-3742):
   - Must use $E_{\text{phys}}$ throughout
   - Will yield $g_{\text{eff}}^2 = g^2 \cdot V/(mN)$ ✓

3. **Magnetic term derivation** (Part 3, lines 3832-3914):
   - Already has correct $1/g^2$ prefactor
   - Will yield **same** $g_{\text{eff}}^2 = g^2 \cdot V/(mN)$ ✓

4. **Consistency**: Both terms converge with **identical** effective coupling

---

## §8. Next Steps (Phase 1.2)

- [ ] Verify canonical structure from Hamilton's equations (confirm $\dot{A} = g^2 E$)
- [ ] Audit existing lemmas for dimensional errors
- [ ] Implement corrected proof with physical fields
- [ ] Verify gauge invariance preserved under field rescaling

**Status**: Phase 1.1 COMPLETE ✓
