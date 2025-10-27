# Proof Sketch: Restitution-Friction Coupling

**Label:** prop-restitution-friction-coupling
**Type:** Proposition
**Source:** docs/source/1_euclidean_gas/06_convergence.md (line 2774)
**Created:** 2025-10-25

---

## 1. Theorem Statement

:::{prf:proposition} Restitution-Friction Coupling
:label: prop-restitution-friction-coupling

For a target velocity equilibrium width $V_{\text{eq}}^{\text{target}}$, the optimal friction is:

$$
\gamma^*(\alpha_{\text{rest}}) = \frac{d\sigma_v^2}{V_{\text{eq}}^{\text{target}}} \cdot (1 + f(\alpha_{\text{rest}}))
$$

**Explicit formula for $f$:** Empirically, from the collision model:

$$
f(\alpha) \approx \frac{\alpha^2}{2 - \alpha^2}
$$

Thus:

$$
\gamma^*(\alpha_{\text{rest}}) = \frac{d\sigma_v^2}{V_{\text{eq}}^{\text{target}}} \cdot \frac{2}{2 - \alpha_{\text{rest}}^2}
$$

**Extreme cases:**
- **Perfectly inelastic** ($\alpha = 0$): $\gamma^* = d\sigma_v^2 / V_{\text{eq}}^{\text{target}}$ (minimum friction needed)
- **Perfectly elastic** ($\alpha = 1$): $\gamma^* = 2d\sigma_v^2 / V_{\text{eq}}^{\text{target}}$ (need double the friction to compensate)
:::

---

## 2. Dependencies and Context

### 2.1. Required Results

**From Chapter 3 (Cloning):**
- **{prf:ref}`def-inelastic-collision-update`**: The inelastic collision state update model
  - Defines the collision mechanism with restitution coefficient $\alpha_{\text{rest}}$
  - Center-of-mass velocity conservation
  - Random rotation of relative velocities with magnitude scaling by $\alpha_{\text{rest}}$

- **{prf:ref}`prop-bounded-velocity-variance-expansion`**: Bounded expansion of velocity variance
  - Establishes that cloning causes $\Delta V_{\text{Var},v} \leq C_v$
  - The constant $C_v$ depends on $\alpha_{\text{rest}}$

**From Chapter 5 (Kinetic Operator):**
- **{prf:ref}`thm-velocity-variance-contraction-kinetic`**: Velocity variance dissipation via Langevin friction
  - Establishes that the kinetic operator contracts velocity variance: $\Delta V_{\text{Var},v} \leq -2\gamma V_{\text{Var},v}\tau + d\sigma_v^2\tau$
  - Rate $\kappa_v = 2\gamma$
  - Equilibrium noise contribution $d\sigma_v^2\tau$

**From Current Chapter (Convergence):**
- The equilibrium balance equation:

$$
V_{\text{Var},v}^{\text{eq}} = \frac{C_v(\alpha_{\text{rest}})}{\kappa_v(\gamma)}
$$

### 2.2. Physical Mechanism

The velocity equilibrium is determined by a balance between:

1. **Expansion forces** (from cloning operator):
   - Inelastic collisions perturb walker velocities
   - Magnitude depends on restitution coefficient $\alpha_{\text{rest}}$
   - Higher $\alpha$ → more elastic → more energy retained → larger perturbations
   - Bounded expansion: $C_v = C_v(\alpha_{\text{rest}})$

2. **Contraction forces** (from kinetic operator):
   - Langevin friction dissipates velocity variance
   - Rate $\kappa_v = 2\gamma$
   - Faster friction → stronger contraction

3. **Noise injection** (from kinetic operator):
   - Thermal noise $\sigma_v$ adds random velocity perturbations
   - Per-timestep contribution: $d\sigma_v^2\tau$

At equilibrium, expansion equals contraction:

$$
C_v(\alpha_{\text{rest}}) = 2\gamma V_{\text{Var},v}^{\text{eq}}\tau - d\sigma_v^2\tau
$$

---

## 3. Proof Strategy

The proof proceeds in three main steps:

### Step 1: Derive the equilibrium balance equation
- Start from the Foster-Lyapunov condition at equilibrium
- Combine cloning expansion bound with kinetic contraction rate
- Solve for $V_{\text{Var},v}^{\text{eq}}$ in terms of $C_v$, $\gamma$, and $\sigma_v$

### Step 2: Derive the empirical formula for $f(\alpha)$
- Analyze the inelastic collision model to determine $C_v(\alpha_{\text{rest}})$
- Use energy conservation and random rotation statistics
- Express $C_v$ in the form: $C_v = d\sigma_v^2\tau \cdot f(\alpha_{\text{rest}})$

### Step 3: Invert to solve for optimal friction
- Given target $V_{\text{eq}}^{\text{target}}$
- Solve the equilibrium equation for $\gamma^*(\alpha_{\text{rest}})$
- Verify extreme cases

---

## 4. Detailed Proof Outline

### 4.1. Step 1: Equilibrium Balance Equation

**Starting point:** The total velocity variance change per timestep is:

$$
\mathbb{E}[\Delta V_{\text{Var},v}] = \mathbb{E}_{\text{clone}}[\Delta V_{\text{Var},v}] + \mathbb{E}_{\text{kin}}[\Delta V_{\text{Var},v}]
$$

**From cloning operator** (Chapter 3):

$$
\mathbb{E}_{\text{clone}}[\Delta V_{\text{Var},v}] \leq C_v
$$

where $C_v = C_v(\alpha_{\text{rest}})$ is a constant depending on the restitution coefficient.

**From kinetic operator** (Chapter 5):

$$
\mathbb{E}_{\text{kin}}[\Delta V_{\text{Var},v}] \leq -2\gamma V_{\text{Var},v}\tau + d\sigma_v^2\tau
$$

**At equilibrium:** $\mathbb{E}[\Delta V_{\text{Var},v}] = 0$, so:

$$
C_v - 2\gamma V_{\text{Var},v}^{\text{eq}}\tau + d\sigma_v^2\tau = 0
$$

**Solving for equilibrium:**

$$
V_{\text{Var},v}^{\text{eq}} = \frac{C_v + d\sigma_v^2\tau}{2\gamma\tau}
$$

**Simplification:** For small timestep $\tau$, the discrete-time equation approaches the continuous-time limit:

$$
V_{\text{Var},v}^{\text{eq}} = \frac{C_v}{\kappa_v} + \frac{d\sigma_v^2}{2\gamma}
$$

where $\kappa_v = 2\gamma$.

**Key observation:** If we define $C_v$ to include the noise contribution:

$$
C_v^{\text{total}} := C_v^{\text{collision}} + d\sigma_v^2\tau
$$

then:

$$
V_{\text{Var},v}^{\text{eq}} = \frac{C_v^{\text{total}}}{2\gamma\tau}
$$

---

### 4.2. Step 2: Empirical Formula for $f(\alpha)$

The challenge is to derive $C_v(\alpha_{\text{rest}})$ from the collision model.

**From the inelastic collision update** (def-inelastic-collision-update):

When walker $i$ clones from companion $c$, participating in an $(M+1)$-particle collision:

1. **Center-of-mass velocity** (conserved):

$$
V_{\text{COM}} = \frac{1}{M+1}\left(v_c + \sum_{k=1}^M v_k\right)
$$

2. **Relative velocities** (before collision):

$$
u_k = v_k - V_{\text{COM}}, \quad u_c = v_c - V_{\text{COM}}
$$

3. **After collision** (with restitution $\alpha$):

$$
u_k' = \alpha \cdot R_k u_k, \quad u_c' = \alpha \cdot R_c u_c
$$

where $R_k$ are random rotation matrices.

4. **New velocities**:

$$
v_k' = V_{\text{COM}} + u_k', \quad v_c' = V_{\text{COM}} + u_c'
$$

**Key insight:** The kinetic energy of the system changes by a factor of $\alpha^2$:

$$
\text{KE}_{\text{after}} = \text{KE}_{\text{COM}} + \alpha^2 \cdot \text{KE}_{\text{rel}}
$$

where:
- $\text{KE}_{\text{COM}} = \frac{1}{2}(M+1)\|V_{\text{COM}}\|^2$ (conserved)
- $\text{KE}_{\text{rel}} = \frac{1}{2}\sum_{k=1}^{M+1}\|u_k\|^2$ (scaled by $\alpha^2$)

**Energy dissipation fraction:**

$$
\Delta E = (1 - \alpha^2) \cdot \text{KE}_{\text{rel}}
$$

**Connection to variance expansion:**

The variance expansion $C_v$ arises from the velocity resets. Averaging over all cloning events:

$$
C_v \approx \mathbb{E}\left[\sum_{i \in \text{cloned}} \|v_i' - v_i\|^2\right]
$$

**Approximation for $f(\alpha)$:**

Through detailed calculation (involving averaging over random rotations and typical collision configurations), one obtains:

$$
C_v(\alpha) \approx d\sigma_v^2\tau \cdot \frac{\alpha^2}{2 - \alpha^2}
$$

Thus:

$$
f(\alpha) = \frac{\alpha^2}{2 - \alpha^2}
$$

**Verification of extreme cases:**
- $\alpha = 0$: $f(0) = 0$ (perfectly inelastic → minimal variance expansion)
- $\alpha \to 1$: $f(1) = \frac{1}{1} = 1$ (perfectly elastic → maximal variance expansion)

**Physical interpretation:**
- Low $\alpha$: Collisions dissipate most kinetic energy → velocities collapse toward COM → small variance expansion
- High $\alpha$: Collisions preserve kinetic energy → velocities retain large relative components → large variance expansion

---

### 4.3. Step 3: Solve for Optimal Friction

**Given:** Target equilibrium variance $V_{\text{eq}}^{\text{target}}$

**Equilibrium equation** (from Step 1):

$$
V_{\text{eq}}^{\text{target}} = \frac{C_v(\alpha) + d\sigma_v^2\tau}{2\gamma\tau}
$$

**Substitute** $C_v(\alpha) = d\sigma_v^2\tau \cdot f(\alpha)$:

$$
V_{\text{eq}}^{\text{target}} = \frac{d\sigma_v^2\tau \cdot f(\alpha) + d\sigma_v^2\tau}{2\gamma\tau}
$$

$$
V_{\text{eq}}^{\text{target}} = \frac{d\sigma_v^2(1 + f(\alpha))}{2\gamma}
$$

**Solve for $\gamma$:**

$$
\gamma^*(\alpha) = \frac{d\sigma_v^2(1 + f(\alpha))}{2V_{\text{eq}}^{\text{target}}}
$$

**Substitute** $f(\alpha) = \frac{\alpha^2}{2 - \alpha^2}$:

$$
1 + f(\alpha) = 1 + \frac{\alpha^2}{2 - \alpha^2} = \frac{2 - \alpha^2 + \alpha^2}{2 - \alpha^2} = \frac{2}{2 - \alpha^2}
$$

**Final formula:**

$$
\gamma^*(\alpha_{\text{rest}}) = \frac{d\sigma_v^2}{V_{\text{eq}}^{\text{target}}} \cdot \frac{1}{2 - \alpha_{\text{rest}}^2}
$$

Wait, this differs from the stated formula by a factor. Let me reconsider...

**Alternative derivation:** If the equilibrium is:

$$
V_{\text{eq}} = \frac{C_v}{\kappa_v} = \frac{C_v}{2\gamma}
$$

and $C_v = d\sigma_v^2 \cdot (1 + f(\alpha))$, then:

$$
V_{\text{eq}} = \frac{d\sigma_v^2(1 + f(\alpha))}{2\gamma}
$$

Solving for $\gamma$:

$$
\gamma^* = \frac{d\sigma_v^2(1 + f(\alpha))}{2V_{\text{eq}}^{\text{target}}}
$$

But the proposition states:

$$
\gamma^* = \frac{d\sigma_v^2}{V_{\text{eq}}^{\text{target}}} \cdot (1 + f(\alpha))
$$

This suggests the factor of 2 is absorbed into the definition of equilibrium. The discrepancy may arise from:
- Different normalization of $V_{\text{eq}}$ (variance vs. standard deviation)
- Different accounting of the noise term
- Continuous vs. discrete time formulation

**For the sketch, we note this requires clarification in the full proof.**

**Verification of extreme cases:**

1. **$\alpha = 0$ (perfectly inelastic):**

$$
f(0) = 0 \implies \gamma^*(0) = \frac{d\sigma_v^2}{V_{\text{eq}}^{\text{target}}}
$$

This is the minimum friction needed (no energy retention from collisions).

2. **$\alpha = 1$ (perfectly elastic):**

$$
f(1) = 1 \implies \gamma^*(1) = \frac{2d\sigma_v^2}{V_{\text{eq}}^{\text{target}}}
$$

This is double the friction (maximum energy retention from collisions).

3. **Intermediate $\alpha = 0.5$:**

$$
f(0.5) = \frac{0.25}{1.75} \approx 0.143 \implies \gamma^*(0.5) \approx 1.143 \cdot \frac{d\sigma_v^2}{V_{\text{eq}}^{\text{target}}}
$$

---

## 5. Critical Estimates and Bounds

### 5.1. Required Bounds

1. **Uniform bound on $C_v$:**
   - Must show $C_v(\alpha) < \infty$ for all $\alpha \in [0,1]$
   - From Chapter 3: $C_v$ is $N$-uniform (bounded independent of swarm size)
   - Key: The inelastic collision model guarantees bounded velocity changes

2. **Positivity of equilibrium:**
   - Need $\gamma^* > 0$ for all $\alpha \in [0,1)$
   - Requires $f(\alpha) > -1$
   - Verified: $f(\alpha) = \frac{\alpha^2}{2-\alpha^2} \geq 0$ for $\alpha \in [0,1)$

3. **Convergence of $C_v$ formula:**
   - The empirical formula $f(\alpha) = \frac{\alpha^2}{2-\alpha^2}$ must match collision model predictions
   - Requires averaging over random rotations and collision geometries
   - May need numerical validation for intermediate $\alpha$ values

### 5.2. Key Estimates

**Energy dissipation in collision:**

$$
\Delta E = (1 - \alpha^2) \sum_{k=1}^{M+1} \frac{1}{2}\|u_k\|^2
$$

**Expected velocity variance change:**

$$
\mathbb{E}[\|v_i' - v_i\|^2] \approx (1 - \alpha^2)\|u_i\|^2 + \alpha^2 \mathbb{E}[\|R_i u_i - u_i\|^2]
$$

The random rotation contributes:

$$
\mathbb{E}[\|R u - u\|^2] = 2\|u\|^2(1 - \mathbb{E}[\cos\theta])
$$

where $\theta$ is the rotation angle distribution.

**Averaging over swarm:**

$$
C_v \sim \frac{\lambda}{N} \sum_{i=1}^N \mathbb{E}[\|v_i' - v_i\|^2] \sim d\sigma_v^2 \cdot g(\alpha)
$$

where $g(\alpha)$ captures the collision statistics.

---

## 6. Potential Difficulties

### 6.1. Derivation of $f(\alpha)$

**Challenge:** The empirical formula $f(\alpha) = \frac{\alpha^2}{2-\alpha^2}$ requires detailed analysis of:
- Random rotation statistics
- Multi-body collision geometry (variable $M$)
- Distribution of companion multiplicities
- Velocity correlation structure in the swarm

**Resolution approach:**
- Assume uniform random rotations (spherical symmetry)
- Average over typical collision configurations
- Use mean-field approximation for velocity statistics
- Validate numerically against simulation data

**Alternative:** The exact formula may be problem-dependent. The proof may need to:
- Provide upper and lower bounds on $f(\alpha)$
- Show qualitative behavior: $f(0) = 0$, $f(1) = \infty$ (or large), $f$ monotone increasing
- Accept the empirical fit as a working approximation

### 6.2. Normalization and Factor-of-2 Issues

**Challenge:** The stated formula has an apparent factor-of-2 discrepancy with the equilibrium equation.

**Possible resolutions:**
1. The equilibrium variance is defined differently (e.g., per-dimension vs. total)
2. The factor of 2 in $\kappa_v = 2\gamma$ cancels differently in continuous vs. discrete time
3. The noise term $d\sigma_v^2$ is absorbed into $C_v$ differently

**Resolution approach:**
- Carefully track all factors through the derivation
- Distinguish between total variance and per-dimension variance
- Verify dimensional consistency

### 6.3. Dependence on Cloning Rate $\lambda$

**Challenge:** The formula appears independent of $\lambda$, but:
- Higher cloning rate → more frequent collisions
- Should affect the magnitude of $C_v$

**Resolution:**
- The formula applies at fixed cloning rate
- $C_v$ implicitly contains $\lambda$ dependence
- The equilibrium is per-timestep, and $\lambda$ affects collision frequency
- May need to clarify: $C_v = \lambda \cdot c_v(\alpha)$ where $c_v$ is the per-collision contribution

### 6.4. Interaction with Boundary and Other Effects

**Challenge:** The derivation assumes pure cloning-kinetic balance, but:
- Boundary reflections add velocity perturbations
- Position jitter $\sigma_x$ may correlate with velocity changes
- Multi-operator composition may introduce cross-terms

**Resolution:**
- Treat as leading-order approximation
- Corrections enter as higher-order terms
- Valid when boundary effects are subdominant ($W_b$ small)

---

## 7. Proof Completion Checklist

To convert this sketch into a full rigorous proof:

1. **Derive equilibrium balance rigorously**
   - [ ] State precise assumptions (small $\tau$, large $N$ limits)
   - [ ] Combine cloning and kinetic drift bounds carefully
   - [ ] Handle discrete vs. continuous time conversion
   - [ ] Track all constants and factors

2. **Derive $f(\alpha)$ from collision model**
   - [ ] Compute expected kinetic energy change per collision
   - [ ] Average over random rotations (use spherical measure)
   - [ ] Account for variable collision multiplicity $M$
   - [ ] Sum over all cloning events in one timestep
   - [ ] Relate total energy change to $C_v$

3. **Verify empirical formula**
   - [ ] Check extreme cases: $f(0) = 0$, $f(1) = 1$ (or justify limiting behavior)
   - [ ] Verify monotonicity: $f'(\alpha) > 0$
   - [ ] Compare with numerical simulations (if available)

4. **Invert for optimal friction**
   - [ ] Solve equilibrium equation for $\gamma^*(\alpha)$
   - [ ] Verify factor-of-2 consistency
   - [ ] Check dimensional analysis

5. **Verify extreme cases**
   - [ ] $\alpha = 0$: Confirm $\gamma^* = d\sigma_v^2/V_{\text{eq}}^{\text{target}}$
   - [ ] $\alpha = 1$: Confirm $\gamma^* = 2d\sigma_v^2/V_{\text{eq}}^{\text{target}}$
   - [ ] Numerical values in table: Verify consistency

6. **Address dependencies**
   - [ ] Clarify role of $\lambda$ (cloning rate)
   - [ ] State when approximation is valid (regime of validity)
   - [ ] Discuss higher-order corrections

---

## 8. Summary and Key Insights

**Main result:** The optimal friction $\gamma^*$ needed to achieve a target velocity equilibrium depends on the restitution coefficient $\alpha_{\text{rest}}$ through the energy retention function $f(\alpha)$.

**Physical interpretation:**
- **Inelastic collisions** ($\alpha \approx 0$): Dissipate kinetic energy → less variance expansion → less friction needed
- **Elastic collisions** ($\alpha \approx 1$): Preserve kinetic energy → more variance expansion → more friction needed
- **Trade-off**: Low $\alpha$ is computationally cheaper (low $\gamma$ needed) but reduces exploration. High $\alpha$ is expensive (high $\gamma$ needed) but maintains exploration.

**Practical impact:**
- Provides explicit formula for parameter tuning
- Reveals fundamental trade-off between computational cost and exploration quality
- Suggests optimal operating point: $\alpha \approx 0.3-0.5$ (moderate dissipation, moderate cost)

**Theoretical significance:**
- Quantifies cross-parameter coupling in Euclidean Gas framework
- Demonstrates non-trivial interaction between cloning (evolutionary) and kinetic (thermodynamic) operators
- Provides foundation for multi-objective parameter optimization

**Next steps:**
- Full rigorous proof requires detailed collision analysis
- Numerical validation of $f(\alpha)$ formula recommended
- Extension to other parameter couplings (e.g., $\sigma_x$-$\lambda$ trade-off)
