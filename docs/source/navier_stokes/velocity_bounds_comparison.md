# Velocity Bounds: Hard Clamping vs Smooth Squashing

**Technical Appendix:** This document explains the relationship between two mechanisms for enforcing velocity bounds in the Fragile Gas framework and their effects on the Navier-Stokes equations.

---

## Executive Summary

The Fragile Gas framework and Navier-Stokes Millennium Problem proof use two different mechanisms to bound particle velocities:

1. **Hard Velocity Clamping** ([hydrodynamics.md](hydrodynamics.md)): Radial projection $\Pi_V(v) := v \cdot \min(1, V/\|v\|)$
2. **Smooth Velocity Squashing** ([NS_millennium.md](NS_millennium.md), [02_euclidean_gas.md](02_euclidean_gas.md)): $\psi_v(v) := V \frac{v}{V + \|v\|}$

**Key Result:** Both mechanisms are **mathematically equivalent** for the purposes of global regularity. The choice affects analytical convenience but not the quantitative bounds. The smooth squashing is preferred for continuum analysis due to its $C^\infty$ regularity.

---

## 1. The Two Mechanisms

### 1.1. Hard Velocity Clamping (Radial Projection)

:::{prf:definition} Hard Velocity Clamp
:label: def-hard-velocity-clamp

The **hard radial projection** enforces a sharp boundary at radius $V_{\text{alg}}$:

$$
\Pi_V(v) := \begin{cases}
v & \text{if } \|v\| \leq V \\
V \frac{v}{\|v\|} & \text{if } \|v\| > V
\end{cases}
$$

Equivalently: $\Pi_V(v) = v \cdot \min\left(1, \frac{V}{\|v\|}\right)$

**Properties:**
- Enforces $\|\Pi_V(v)\| \leq V$ exactly (closed ball)
- $\Pi_V$ is 1-Lipschitz in the interior $\{v : \|v\| < V\}$
- Discontinuous derivative at the boundary $\{v : \|v\| = V\}$
- Identity map inside the ball: $\Pi_V(v) = v$ for $\|v\| \leq V$
- Radial projection at boundary: $\Pi_V(\alpha v) = V \frac{v}{\|v\|}$ for $\alpha > V/\|v\|$
:::

**Physical Interpretation:** When a particle exceeds the velocity threshold, it is **instantaneously reflected** back to the boundary sphere. This creates a **hard wall** at $\|v\| = V$.

**Mathematical Description:** This is the **Skorokhod reflection problem** on the velocity domain. The dynamics live on the state space $\mathcal{X} \times \mathbb{B}_V$ where $\mathbb{B}_V = \{v \in \mathbb{R}^d : \|v\| \leq V\}$ is the closed ball.

### 1.2. Smooth Velocity Squashing

:::{prf:definition} Smooth Squashing Map
:label: def-smooth-squashing-map

The **smooth squashing map** is a $C^\infty$ diffeomorphism onto the open ball:

$$
\psi_v(v) := V_{\text{alg}} \frac{v}{V_{\text{alg}} + \|v\|}
$$

**Properties:**
- Enforces $\|\psi_v(v)\| < V_{\text{alg}}$ strictly (open ball)
- $C^\infty$ smooth everywhere (including at infinity)
- 1-Lipschitz globally: $\|\psi_v(v_1) - \psi_v(v_2)\| \leq \|v_1 - v_2\|$
- Identity near origin: $\psi_v(v) \approx v$ for $\|v\| \ll V_{\text{alg}}$
- Asymptotic limit: $\psi_v(v) \to V_{\text{alg}} \frac{v}{\|v\|}$ as $\|v\| \to \infty$
- Invertible: $\psi_v^{-1}(w) = V_{\text{alg}} \frac{w}{V_{\text{alg}} - \|w\|}$ for $\|w\| < V_{\text{alg}}$
:::

**Physical Interpretation:** As particles approach high velocities, they experience increasingly strong **smooth resistance** that prevents them from ever reaching $V_{\text{alg}}$. There is no sharp boundary - just a gradual "squashing" of phase space.

**Mathematical Description:** The dynamics live on the state space $\mathcal{X} \times \mathbb{R}^d$, but the map $\psi_v$ ensures velocities remain in the open ball $\{v : \|v\| < V_{\text{alg}}\}$.

---

## 2. Comparison of Properties

| Property | Hard Clamp $\Pi_V$ | Smooth Squashing $\psi_v$ |
|----------|-------------------|--------------------------|
| **Domain** | $\mathbb{R}^d \to \mathbb{B}_V$ (closed ball) | $\mathbb{R}^d \to \mathbb{B}_V^\circ$ (open ball) |
| **Smoothness** | Discontinuous derivative at $\partial \mathbb{B}_V$ | $C^\infty$ everywhere |
| **Lipschitz** | 1-Lipschitz in interior, not globally | 1-Lipschitz globally |
| **Identity region** | $\{v : \|v\| \leq V\}$ exactly | $\{v : \|v\| \ll V\}$ approximately |
| **Boundary behavior** | Hard reflection | Asymptotic approach |
| **Implementation** | `v_clamp = v * min(1, V/||v||)` | `v_squash = V * v / (V + ||v||)` |

**Derivative Analysis:**

For the hard clamp at $\|v\| = V$:
$$
\frac{\partial \Pi_V}{\partial v_i} = \begin{cases}
I & \text{if } \|v\| < V \\
\text{undefined/distributional} & \text{if } \|v\| = V \\
V \left(\frac{I}{\|v\|} - \frac{v v^T}{\|v\|^3}\right) & \text{if } \|v\| > V \text{ (before projection)}
\end{cases}
$$

For the smooth squashing everywhere:
$$
\frac{\partial \psi_v}{\partial v_i} = \frac{V_{\text{alg}}}{(V_{\text{alg}} + \|v\|)^2} \left[(V_{\text{alg}} + \|v\|) I - \frac{v v^T}{\|v\|}\right]
$$

The smooth map has bounded derivatives globally, while the hard clamp has a singularity at the boundary.

---

## 3. Effect on the Navier-Stokes Equations

### 3.1. Particle-Level Dynamics

**With Hard Clamping:**

The velocity update in the Langevin step is:
$$
v_{i}^{(t+1)} = \Pi_V\Big(v_i^{(t)} + \tau F(x_i) - \gamma v_i^{(t)} \tau + \sqrt{2\epsilon \tau} \xi_i\Big)
$$

When $\|v_i^{(t+1)}\|$ exceeds $V_{\text{alg}}$ **before** projection, the particle is reflected:
- Reflection introduces **boundary measure** concentrated on $\partial \mathbb{B}_V$
- Stochastic dynamics require Skorokhod problem formulation
- Local time at boundary $L_t = \int_0^t \mathbf{1}_{\{\|v_s\| = V\}} d\ell_s$ accumulates

**With Smooth Squashing:**

The velocity update is:
$$
v_{i}^{(t+1)} = \psi_v\Big(v_i^{(t)} + \tau F(x_i) - \gamma v_i^{(t)} \tau + \sqrt{2\epsilon \tau} \xi_i\Big)
$$

Or equivalently, apply $\psi_v$ after each time step. The map is smooth, so:
- No boundary singularities
- No Skorokhod problem
- Standard SDE theory applies

### 3.2. Mean-Field Continuum Limit

Both mechanisms lead to the same limiting SPDE as $N \to \infty$:

$$
d\mathbf{u} = \left[\nu \nabla^2 \mathbf{u} - (\mathbf{u} \cdot \nabla)\mathbf{u} - \nabla p + \mathbf{F}\right] dt + \sqrt{2\epsilon} \, dW
$$

with different boundary conditions on the velocity space:

**Hard Clamp:**
- Reflecting boundary conditions at $\|\mathbf{u}(t,x)\| = V_{\text{alg}}$
- Probability measure $\mu_t$ lives on $\mathcal{X} \times \mathbb{B}_V$
- Requires analysis of boundary layer near $\|\mathbf{u}\| = V_{\text{alg}}$

**Smooth Squashing:**
- No hard boundary - velocities live in open ball $\|\mathbf{u}(t,x)\| < V_{\text{alg}}$
- Probability measure $\mu_t$ lives on $\mathcal{X} \times \mathbb{B}_V^\circ$
- The map $\psi_v$ is absorbed into the velocity dynamics

**Key Point:** In both cases, the effective dynamics at the QSD are identical because velocities remain **far from the boundary** with overwhelming probability.

---

## 4. Why Both Mechanisms Are Equivalent

The critical observation is that **both mechanisms are inactive** in the regime relevant for global regularity:

### 4.1. Velocity Concentration at QSD

From the QSD energy balance ([NS_millennium.md](NS_millennium.md) §5.3.2):

$$
\mathbb{E}_{\mu_\epsilon}[\|\mathbf{u}\|_{L^2}^2] = O\left(\frac{\epsilon L^3}{\nu}\right)
$$

With $V_{\text{alg}} = 1/\epsilon$, typical velocities satisfy:

$$
\|\mathbf{u}\|_{L^\infty} \sim O\left(\sqrt{\frac{\epsilon L^3}{\nu}}\right) \ll \frac{1}{\epsilon} = V_{\text{alg}}
$$

for small $\epsilon$.

### 4.2. Large Deviation Bound

The LSI concentration theorem ([NS_millennium.md](NS_millennium.md) Theorem 5.3.2.1) gives:

$$
\mathbb{P}_{\mu_\epsilon}\left(\|\mathbf{u}\|_{L^\infty} > \frac{1}{\epsilon}\right) \leq \exp\left(-\frac{\nu}{2C_0 C_{\text{Sob}}^2 L^3 \epsilon^2}\right) = o(\epsilon^c)
$$

for **any** $c > 0$.

**Consequence:** The velocity bound is violated with **super-exponentially small probability**. Therefore:

1. **Hard clamp:** The projection $\Pi_V$ is activated with probability $o(\epsilon^c)$, so the system evolves as if unconstrained almost surely.

2. **Smooth squashing:** The map $\psi_v$ acts nearly as the identity since $\|v\| \ll V_{\text{alg}}$:
   $$
   \psi_v(v) = V_{\text{alg}} \frac{v}{V_{\text{alg}} + \|v\|} \approx v \left(1 - \frac{\|v\|}{V_{\text{alg}}}\right) \approx v
   $$
   with error $O(\epsilon \|v\|) \ll v$.

### 4.3. Mathematical Equivalence

Both approaches give:
- Same QSD measure $\mu_\epsilon$ up to exponentially small corrections
- Same uniform bounds on $H^3$ regularity
- Same limiting classical NS as $\epsilon \to 0$

**Difference:** Only in how we **justify** that the velocity bound doesn't affect the dynamics:
- **Hard clamp:** Probabilistic argument (bound rarely hit)
- **Smooth squashing:** Analytical argument (map acts as identity)

---

## 5. Analytical Advantages of Smooth Squashing

### 5.1. Regularity Theory

**Problem with hard clamp:**
- Sobolev embedding $H^k \subset C^{k'}$ requires smoothness
- Hard projection introduces discontinuity in velocity derivatives
- Requires careful boundary layer analysis

**Advantage of smooth squashing:**
- $C^\infty$ map preserves regularity: if $v \in H^k$, then $\psi_v(v) \in H^k$
- No boundary singularities to analyze
- Standard PDE theory applies directly

### 5.2. Lipschitz Continuity

**Problem with hard clamp:**
- $\Pi_V$ is not Lipschitz at the boundary
- SDE well-posedness requires Lipschitz coefficients
- Skorokhod reflection needs specialized theory

**Advantage of smooth squashing:**
- $\psi_v$ is globally 1-Lipschitz
- Standard SDE existence/uniqueness theorems apply
- No need for reflected SDE machinery

### 5.3. Continuum Limit

**Problem with hard clamp:**
- Particle systems with reflection require careful $N \to \infty$ limit
- Boundary measure needs to be tracked
- Coupling to PDE is technical

**Advantage of smooth squashing:**
- Standard mean-field limit theory applies
- No singular measures to handle
- Clean SPDE formulation

---

## 6. When Does the Choice Matter?

The choice between hard clamping and smooth squashing **only matters** in the following scenarios:

### 6.1. Near-Boundary Dynamics ❌ (Not Relevant)

**When velocities approach $V_{\text{alg}}$:**
- Hard clamp: Particles reflect, creating boundary layer
- Smooth squashing: Particles slow down smoothly

**Why this doesn't matter:** At the QSD with $\epsilon$ small, the probability of reaching $V_{\text{alg}} = 1/\epsilon$ is $o(\epsilon^c)$, so near-boundary dynamics never occur with probability approaching 1.

### 6.2. Short-Time Dynamics ❌ (Not Relevant)

**For transient initial conditions with $\|\mathbf{u}_0\|_{L^\infty} \approx 1/\epsilon$:**
- Hard clamp: Immediate reflection prevents blow-up
- Smooth squashing: Gradual slowing prevents blow-up

**Why this doesn't matter:** The initial condition is assumed to be $H^3$ regular with $\|\mathbf{u}_0\|_{H^3} < \infty$. For reasonable initial data, $\|\mathbf{u}_0\| \ll 1/\epsilon$ for small $\epsilon$.

### 6.3. Numerical Implementation ✓ (Matters Practically)

**Computational efficiency:**
- Hard clamp: `v_new = v * min(1.0, V / ||v||)` - simple, fast
- Smooth squashing: `v_new = V * v / (V + ||v||)` - slightly more expensive

**Numerical stability:**
- Hard clamp: Can create artificial boundary layer in discretization
- Smooth squashing: Smoother numerical behavior

**Choice:** For **particle simulations** (hydrodynamics.md), hard clamp is simpler. For **continuum analysis** (NS_millennium.md), smooth squashing is cleaner.

---

## 7. Usage in the Codebase

### 7.1. Current Usage

| File | Mechanism | Context | Justification |
|------|-----------|---------|---------------|
| **hydrodynamics.md** | Hard clamp $\Pi_V$ | Particle-level Fragile NS | Simpler conceptually, easier to implement |
| **NS_millennium.md** | Smooth squashing $\psi_v$ | Continuum SPDE proof | $C^\infty$ regularity, avoids Skorokhod problem |
| **02_euclidean_gas.md** | Smooth squashing $\psi_v$ | Euclidean Gas implementation | Consistent with framework, smooth dynamics |

### 7.2. Consistency Note

The **apparent inconsistency** (hard clamp in hydrodynamics.md vs smooth squashing in NS_millennium.md) is not an error:

- **hydrodynamics.md** describes the **particle-level** dynamics, where hard clamping is natural and computationally efficient
- **NS_millennium.md** analyzes the **mean-field continuum limit**, where smooth squashing provides analytical advantages
- **02_euclidean_gas.md** implements the smooth squashing in code, which is used in actual simulations

### 7.3. Recommendation

For **pedagogical clarity**, one could:

1. **Option A (No change):** Acknowledge that both formulations are valid and lead to the same results. Current usage reflects particle vs continuum perspectives.

2. **Option B (Harmonize):** Update hydrodynamics.md to use smooth squashing $\psi_v$ throughout for consistency with the implementation and NS proof.

3. **Option C (Document both):** Add this appendix to hydrodynamics.md and NS_millennium.md as a cross-reference explaining the relationship.

**Current recommendation:** **Option C** - this document serves as the technical appendix explaining why both are valid.

---

## 8. Mathematical Formalism

### 8.1. Hard Clamp as Skorokhod Problem

The hard-clamped dynamics satisfy:

$$
\begin{cases}
dv_t = F(x_t, v_t) \, dt + \sigma \, dW_t + dL_t \\
\|v_t\| \leq V_{\text{alg}} \quad \forall t \\
L_t \text{ is a process with } dL_t \perp T_{v_t}(\partial \mathbb{B}_V) \text{ (inward normal)}
\end{cases}
$$

where $L_t$ is the **local time at the boundary** that enforces the constraint.

**Properties:**
- $L_t$ increases only when $\|v_t\| = V_{\text{alg}}$
- $dL_t$ points radially inward
- Well-posed under Lipschitz drift and diffusion

### 8.2. Smooth Squashing as Change of Variables

The smooth-squashed dynamics can be written as:

$$
dv_t = F_{\psi}(x_t, v_t) \, dt + \sigma_{\psi}(v_t) \, dW_t
$$

where:
$$
F_{\psi}(x, v) := D\psi_v(v) \cdot F(x, \psi_v(v))
$$
$$
\sigma_{\psi}(v) := D\psi_v(v) \cdot \sigma
$$

with $D\psi_v$ being the Jacobian of $\psi_v$.

**Properties:**
- No boundary measure
- $F_{\psi}$ and $\sigma_{\psi}$ are globally Lipschitz
- Standard SDE theory applies

### 8.3. Limiting Equivalence

As $\epsilon \to 0$ with $V_{\text{alg}} = 1/\epsilon \to \infty$:

**Hard clamp:**
$$
\Pi_V(v) = v \quad \text{with probability } 1 - o(\epsilon^c)
$$

**Smooth squashing:**
$$
\psi_v(v) = v \left(1 - O\left(\frac{\epsilon \|v\|}{1}\right)\right) \to v \quad \text{as } \epsilon \to 0
$$

Both converge to the **classical NS** without velocity constraints:

$$
\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla)\mathbf{u} = -\nabla p + \nu \nabla^2 \mathbf{u}
$$

---

## 9. Conclusion

**Summary of Key Points:**

1. **Two Mechanisms:** Hard radial projection $\Pi_V$ vs smooth squashing $\psi_v$

2. **Mathematical Equivalence:** Both give the same quantitative regularity bounds because velocities remain far from the boundary with overwhelming probability

3. **Analytical Preference:** Smooth squashing is preferred for continuum analysis due to:
   - $C^\infty$ regularity (no boundary singularities)
   - Global 1-Lipschitz property
   - Avoids Skorokhod problem technicalities

4. **Usage Context:**
   - **Particle simulations:** Hard clamp (simpler)
   - **Continuum proofs:** Smooth squashing (cleaner)
   - **Current codebase:** Mixed usage is intentional and valid

5. **No Contradiction:** The different choices in hydrodynamics.md vs NS_millennium.md reflect different levels of description (particle vs continuum), not an inconsistency in the mathematics

**Bottom Line:** The Navier-Stokes Millennium Problem proof is **robust** to the choice of velocity bounding mechanism. Either formulation leads to global regularity. The smooth squashing is simply more convenient for rigorous analysis.

---

## References

1. [hydrodynamics.md](hydrodynamics.md) - Fragile Navier-Stokes with particle-level hard clamping
2. [NS_millennium.md](NS_millennium.md) - Millennium Problem proof using smooth squashing
3. [02_euclidean_gas.md](02_euclidean_gas.md) - Euclidean Gas implementation with smooth squashing
4. Tanaka, H. (1979). "Stochastic differential equations with reflecting boundary condition in convex regions." *Hiroshima Math. J.* 9(1), 163-177. (Skorokhod problem theory)
5. Lions, P.L., Sznitman, A.S. (1984). "Stochastic differential equations with reflecting boundary conditions." *Comm. Pure Appl. Math.* 37(4), 511-537.
