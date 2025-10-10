# The Ricci Fragile Gas: A Theory of Emergent Geometric Structure Formation

## 0. Executive Summary and Physical Analogy

### 0.1. The Core Idea: Geometry Creates Matter, Matter Creates Geometry

The **Ricci Fragile Gas** represents a paradigm shift in the Fragile Gas framework. Rather than optimizing a scalar fitness function, the swarm's dynamics are governed by the **geometric curvature of the manifold they collectively create**. This creates a self-referential feedback loop reminiscent of Einstein's field equations in General Relativity:

> *"Matter tells spacetime how to curve, spacetime tells matter how to move."*
>
> — John Archibald Wheeler

In the Ricci Gas:
- The swarm distribution defines an emergent Riemannian metric $g(x, S)$
- The metric's Ricci curvature $R(x, S)$ determines both cloning rewards and kinetic forces
- The resulting dynamics create new distributions, closing the loop

### 0.2. The Push-Pull Architecture: Emergent Equilibrium from Opposing Forces

The Ricci Gas achieves stability and pattern formation through **antagonistic forces**:

**Pull (Gravity):** Langevin force aggregates toward high curvature
$$
F_{\text{gravity}} = +\epsilon_R \nabla R(x, S)
$$

- Walkers are attracted to regions of high Ricci curvature
- Dense clusters have high curvature → attract more walkers
- This is an **aggregative** mechanism acting through dynamics

**Push (Anti-Gravity / Quantum Pressure):** Cloning rewards low curvature
$$
\text{Reward}(x, S) = \frac{1}{R(x, S) + \epsilon_R}
$$

- Walkers in low-curvature (flat) regions clone more frequently
- High-curvature regions suppress cloning → "evaporation" back to flat regions
- This is a **dispersive** mechanism acting through selection

**Equilibrium (The Quasi-Stationary Distribution):**

The QSD represents a dynamic balance:
- Rate of gravitational infall = Rate of evaporative cloning
- Creates stable "black hole"-like structures with finite size
- Internal pressure from diversity signal prevents complete collapse

### 0.3. The Phase Transition Prediction

**Central Theoretical Prediction:** The system exhibits a **critical phase transition** at feedback strength $\alpha_c$.

**Subcritical Phase** ($\alpha < \alpha_c$):
- Diffuse, gas-like distribution
- Global existence of smooth solutions
- Logarithmic Sobolev Inequality (LSI) holds
- Exponential convergence to QSD

**Critical Point** ($\alpha = \alpha_c$):
- LSI constant vanishes
- Mathematical signature of phase transition
- Bifurcation in QSD structure

**Supercritical Phase** ($\alpha > \alpha_c$):
- Finite-time collapse to concentrated structures
- LSI breaks down (this is a feature, not a bug)
- Formation of singularities (regularized by smoothing length $\ell$)
- Pattern formation and emergent structure

### 0.4. Relation to Prior Work and Key Innovation

- **`02_euclidean_gas.md`**: Provides base Langevin dynamics
- **`07_adaptative_gas.md`**: Introduces Hessian-based emergent metric
- **`08_emergent_geometry.md`**: Analyzes convergence on emergent manifolds
- **Patlak-Keller-Segel (PKS) models**: Mathematical analogue from chemotaxis theory

**Key Innovation:** This is the first Fragile Gas variant where:
1. **Geometry is both input and output** (self-referential feedback)
2. **Phase transitions are predicted and analyzed** (not just stability)
3. **Singularities are features** (structure formation, not algorithmic failure)
4. **Physical analogies are rigorous** (GR, black holes, quantum pressure)

---

## 1. Mathematical Framework

### 1.1. The Emergent Metric from Swarm Density

Following `07_adaptative_gas.md` and `08_emergent_geometry.md`, define:

:::{prf:definition} Kernel Density Estimator
:label: def-ricci-kde

For a swarm state $S = \{(x_i, v_i, s_i)\}_{i=1}^N$ and smoothing kernel $K_\ell(x) = \ell^{-d} K(x/\ell)$, the **smoothed swarm density** is:

$$
\rho(x, S) = \frac{1}{N} \sum_{i \in \mathcal{A}} K_\ell(x - x_i)
$$

where $\mathcal{A}$ is the alive set and $\ell > 0$ is the bandwidth (Planck length).

**Standard choice:** Gaussian kernel $K(x) = (2\pi)^{-d/2} \exp(-\|x\|^2/2)$ ensures $C^\infty$ smoothness.

:::

:::{prf:definition} Fitness Potential and Emergent Metric
:label: def-ricci-metric

The **fitness potential** is constructed from the smoothed density via the standard measurement pipeline:

$$
V_{\text{fit}}(x, S) = \left( d'(x, S) \right)^\beta \left( r'(x, S) \right)^\alpha
$$

where $d'$ and $r'$ are the logistic-rescaled standardized diversity and reward.

The **emergent Riemannian metric** on state space $\mathcal{X} \subseteq \mathbb{R}^3$ is:

$$
g(x, S) = H(x, S) + \epsilon_\Sigma I
$$

where $H(x, S) = \nabla^2 V_{\text{fit}}(x, S)$ is the Hessian and $\epsilon_\Sigma > 0$ ensures uniform ellipticity.

:::

### 1.2. Ricci Curvature Proxy for 3D Applications

For $d = 3$ (3D physics simulations), we use a computationally efficient Ricci scalar proxy:

:::{prf:definition} Ricci Curvature Proxy (3D)
:label: def-ricci-proxy-3d

The **3D Ricci scalar proxy** is defined as:

$$
R(x, S) = \text{tr}(H(x, S)) - \lambda_{\min}(H(x, S))
$$

where $\lambda_{\min}$ is the minimum eigenvalue of the $3 \times 3$ Hessian matrix.

**Geometric interpretation:**
- $\text{tr}(H) = \Delta V_{\text{fit}}$ measures average local curvature (concavity/convexity)
- $\lambda_{\min}(H)$ measures curvature in the most negative direction (expansion)
- $R \approx 0$: Flat region (low concentration)
- $R > 0$: Positive curvature (high concentration, convex potential well)
- $R < 0$: Negative curvature (saddle region, expansion along some directions)

:::

:::{prf:lemma} Smoothness of Ricci Proxy
:label: lem-ricci-smoothness

If $V_{\text{fit}}$ is constructed via KDE with a $C^\infty$ kernel, then:
1. $H(x, S)$ is $C^\infty$ in $x$
2. Eigenvalues $\lambda_i(H)$ are continuous and piecewise $C^\infty$ (smooth except at crossings)
3. $R(x, S)$ is piecewise $C^\infty$ in $x$
4. $\nabla R(x, S)$ exists almost everywhere and is Lipschitz continuous

:::

:::{prf:proof} Sketch
The Hessian $H$ inherits the smoothness of the kernel. Eigenvalue smoothness follows from perturbation theory for symmetric matrices. The trace is always smooth. The minimum eigenvalue is smooth except on the codimension-1 manifold where eigenvalues cross. For generic configurations this has measure zero, ensuring $\nabla R$ is well-defined almost everywhere.
:::

### 1.3. The Ricci Fragile Gas Algorithm

:::{prf:algorithm} Ricci Fragile Gas Update
:label: alg-ricci-gas

Given swarm state $\mathcal{S}_t = (w_1, \ldots, w_N)$ with walkers $w_i = (x_i, v_i, s_i)$:

**Stage 1: Cemetery check**
- If all walkers dead, return cemetery state; otherwise continue

**Stage 2: Compute emergent geometry**
- For each alive walker $i \in \mathcal{A}_t$:
  - Compute smoothed density $\rho(x_i, S_t)$ via KDE
  - Compute fitness potential $V_{\text{fit}}(x_i, S_t)$
  - Compute Hessian $H_i = \nabla^2 V_{\text{fit}}(x_i, S_t)$
  - Compute Ricci proxy $R_i = \text{tr}(H_i) - \lambda_{\min}(H_i)$

**Stage 3: Ricci-based reward (dispersive push)**
- Set reward: $\text{Reward}_i = 1 / (R_i + \epsilon_R)$
- Sample potential companions $c_{\text{pot}}(i)$ from algorithmic distance kernel
- Compute algorithmic distances $d_i = d_{\text{alg}}(i, c_{\text{pot}}(i))$
- Apply patched standardization to $(\text{Reward}_i, d_i)$
- Apply logistic rescale → $(r'_i, d'_i)$
- Compute cloning fitness $V_{\text{fit},i} = (d'_i)^\beta (r'_i)^\alpha$

**Stage 4: Clone/Persist gate**
- Sample clone companions $c_{\text{clone}}(i)$
- Compute canonical score $S_i = (V_{\text{fit},c} - V_{\text{fit},i})/(V_{\text{fit},i} + \epsilon_{\text{clone}})$
- Clone if $S_i > T_i$ where $T_i \sim \text{Unif}(0, p_{\max})$
- Cloned walker: $x_i \leftarrow x_c + \mathcal{N}(0, \sigma_x^2 I)$, $v_i \leftarrow v_c$

**Stage 5: Ricci-driven Langevin perturbation (aggregative pull)**
- Apply BAOAB integrator with curvature-driven force:

$$
\begin{aligned}
v_i^{(1/2)} &\leftarrow v_i + \frac{\tau}{2} \left( F_{\text{orig}}(x_i) + \epsilon_R \nabla R(x_i, S_t) \right) \\
x_i^{(1)} &\leftarrow x_i + \frac{\tau}{2} v_i^{(1/2)} \\
v_i^{(2)} &\leftarrow e^{-\gamma \tau} v_i^{(1/2)} + \sqrt{1 - e^{-2\gamma\tau}} \Sigma_{\text{reg}}(x_i, S_t) \xi_i \\
x_i^{(2)} &\leftarrow x_i^{(1)} + \frac{\tau}{2} v_i^{(2)} \\
v_i^+ &\leftarrow v_i^{(2)} + \frac{\tau}{2} \left( F_{\text{orig}}(x_i^{(2)}) + \epsilon_R \nabla R(x_i^{(2)}, S_t) \right)
\end{aligned}
$$

where $\Sigma_{\text{reg}} = (H + \epsilon_\Sigma I)^{-1/2}$ is the adaptive diffusion tensor.

**Stage 6: Status refresh and singularity regulation**
- Set $s_i^{(t+1)} = \mathbf{1}_{\mathcal{X}_{\text{valid}}}(x_i^+) \cdot \mathbf{1}_{R(x_i^+) < R_{\text{crit}}}$
- Walkers entering high-curvature regions ($R > R_{\text{crit}}$) are marked dead
- Dead walkers revive via cloning from low-curvature regions → "bouncing singularity"

**Output:** $\mathcal{S}_{t+1}$

:::

### 1.4. Parameters and Their Physical Interpretation

| Parameter | Symbol | Physical Meaning |
|-----------|--------|------------------|
| Feedback strength | $\epsilon_R$ | Strength of gravitational pull |
| Smoothing length | $\ell$ | Planck length (minimum geometric scale) |
| Critical curvature | $R_{\text{crit}}$ | Singularity horizon (like event horizon) |
| Regularization | $\epsilon_\Sigma$ | Quantum fluctuation scale |
| Diversity exponent | $\beta$ | Internal pressure strength |
| Reward exponent | $\alpha$ | Exploitation vs exploration balance |

---

## 2. Multi-Layered Stability Analysis

### 2.1. The Three Mechanisms of Stability

:::{prf:definition} Multi-Layered Stability
:label: def-multilayer-stability

The Ricci Gas achieves stability through three nested mechanisms:

**Layer 1: Global Push-Pull Balance**
- **Aggregation**: Kinetic force $F = +\epsilon_R \nabla R$ pulls walkers together
- **Dispersion**: Cloning reward $\propto 1/R$ pushes walkers apart
- **Effect**: Prevents total collapse of swarm to a single point

**Layer 2: Internal Cluster Pressure**
- **Aggregation**: Same kinetic force tries to compress clusters
- **Dispersion**: Diversity signal $(d')^\beta$ decreases in dense clusters → reduces fitness
- **Effect**: Creates internal pressure preventing zero-variance collapse

**Layer 3: Singularity Regulation via Status Killing**
- **Mechanism**: Walkers die when $R > R_{\text{crit}}$
- **Effect**: Hard boundary prevents true mathematical singularity
- **Revival**: Dead walkers immediately revive via cloning from flat regions
- **Interpretation**: "Bouncing singularity" with information teleportation

:::

### 2.2. Comparison to Patlak-Keller-Segel (PKS) Models

The Ricci Gas is mathematically analogous to PKS models of chemotaxis:

| PKS Component | Ricci Gas Analogue | Mathematical Form |
|---------------|-------------------|-------------------|
| Diffusion | Langevin noise + friction | $D \Delta \rho$ |
| Chemotactic drift | Ricci-driven force | $-\alpha \nabla \cdot (\rho \nabla R[\rho])$ |
| Birth/death | Cloning operator | $(k(R[\rho]) - \langle k \rangle) \rho$ |
| Chemical $c$ | Ricci curvature $R$ | $R[\rho]$ = nonlocal functional |

**Key difference**: In PKS, the chemical $c$ follows a separate PDE. In Ricci Gas, $R[\rho]$ is computed directly from $\rho$ through the geometric construction, making it fully self-contained.

---

## 3. Phase Transition Theory

### 3.1. The Mean-Field PDE

In the $N \to \infty$ limit with KDE smoothing, the swarm density $\rho(x, t)$ satisfies:

:::{prf:definition} Ricci-Gas Mean-Field PDE
:label: def-ricci-mf-pde

$$
\frac{\partial \rho}{\partial t} = D \Delta \rho - \alpha \nabla \cdot \left( \rho \nabla R[\rho] \right) + \left( k(R[\rho]) - \langle k(R) \rangle_\rho \right) \rho
$$

where:
- $D$: Effective diffusion constant (from Langevin noise)
- $\alpha = \epsilon_R$: Aggregation strength
- $R[\rho](x) = \text{tr}(H[\rho](x)) - \lambda_{\min}(H[\rho](x))$: Ricci functional
- $k(R) = 1/(R + \epsilon_R)$: Cloning rate
- $\langle k(R) \rangle_\rho = \int k(R[\rho](x)) \rho(x) dx$: Mean cloning rate

:::

### 3.2. The Critical Parameter Hypothesis

:::{prf:conjecture} Existence of Critical Phase Transition
:label: conj-critical-alpha

There exists a critical feedback strength $\alpha_c > 0$ such that:

**Subcritical Phase** ($\alpha < \alpha_c$):
1. Solutions to the mean-field PDE exist globally in time
2. $\rho(x, t) \to \rho_{\infty}(x)$ as $t \to \infty$ (unique QSD)
3. $\rho_{\infty}$ is a smooth, diffuse distribution
4. The system satisfies a Logarithmic Sobolev Inequality (LSI):
   $$
   \text{Ent}(\rho) \leq \frac{1}{2 \lambda_{\text{LSI}}} \mathcal{I}(\rho)
   $$
   with $\lambda_{\text{LSI}}(\alpha) > 0$ for $\alpha < \alpha_c$
5. Exponential convergence: $\|\rho(t) - \rho_\infty\|_{L^1} \leq C e^{-\lambda_{\text{LSI}} t}$

**Critical Point** ($\alpha = \alpha_c$):
1. $\lambda_{\text{LSI}}(\alpha_c) = 0$ (LSI constant vanishes)
2. Polynomial (slow) convergence to QSD
3. Power-law correlations emerge

**Supercritical Phase** ($\alpha > \alpha_c$):
1. For certain initial conditions $\rho_0$, solutions blow up in finite time:
   $$
   \lim_{t \to T^*} \|\rho(\cdot, t)\|_{L^\infty} = \infty
   $$
2. Regularized solutions (with $\ell > 0$) form concentrated structures
3. LSI does not hold (entropy production can be positive)
4. QSD exhibits pattern formation (multiple peaks, fractal-like structure)

:::

### 3.3. Proof Strategy (Research Program)

To prove Conjecture {prf:ref}`conj-critical-alpha`, one would follow the PKS literature:

**Step 1: Construct a Free Energy Functional**

:::{prf:definition} Free Energy
:label: def-ricci-free-energy

$$
\mathcal{F}[\rho] = \int \rho \log \rho \, dx - \alpha \int \rho R[\rho] \, dx + \frac{D}{2} \int |\nabla \rho|^2 / \rho \, dx
$$

The first term is entropy (favors dispersion), the second is potential energy (favors aggregation), the third penalizes large gradients.

:::

**Step 2: Derive the Critical Value**

Show that:
- $\mathcal{F}$ is bounded below iff $\alpha < \alpha_c$
- $\alpha_c$ depends on space dimension $d$, domain size, and the form of $R[\rho]$
- For PKS in 2D, $\alpha_c = 8\pi$ (exact result)
- For Ricci Gas in 3D, $\alpha_c$ must be computed numerically or bounded analytically

**Step 3: Prove Global Existence for Subcritical Case**

Use LSI to show:
$$
\frac{d}{dt} \mathcal{F}[\rho(t)] \leq -C \mathcal{F}[\rho(t)]
$$

for $\alpha < \alpha_c$, implying exponential decay to equilibrium.

**Step 4: Prove Blow-Up for Supercritical Case**

Use virial method: show that the second moment
$$
M_2(t) = \int |x|^2 \rho(x, t) \, dx
$$

satisfies $M_2''(t) < 0$ for certain $\rho_0$, implying $M_2(T^*) = 0$ for some finite $T^*$.

---

## 4. Computational Feasibility for 3D Physics

### 4.1. Cost Analysis

For $N$ walkers in 3D ($d = 3$):

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| KDE evaluation | $O(N^2)$ or $O(N \log N)$ | Use tree-based algorithms (k-d tree, ball tree) |
| Hessian computation | $O(N^2 \cdot 9)$ | 9 elements per $3 \times 3$ matrix, requires KDE gradients |
| Eigenvalue decomposition | $O(N \cdot 27)$ | Standard $3 \times 3$ eigensolve |
| Ricci proxy | $O(N)$ | Simple trace and min operation |
| Gradient $\nabla R$ | $O(N^2 \cdot 9)$ | Third derivatives via autodiff |
| **Total per iteration** | $O(N^2 d^2)$ | Bottleneck: KDE Hessian computation |

**Optimization strategies:**
1. **Tree-based KDE**: Reduces $O(N^2)$ to $O(N \log N)$ for moderate accuracy
2. **GPU acceleration**: Highly parallelizable (CUDA, JAX)
3. **Bandwidth truncation**: Only include neighbors within $3\ell$ radius
4. **Adaptive KDE**: Variable bandwidth based on local density

**Feasibility assessment:**
- For $N = 10^3$ walkers: ~1-10 seconds per iteration (CPU)
- For $N = 10^4$ walkers: ~10-100 seconds per iteration (CPU), ~0.1-1 seconds (GPU)
- **Conclusion**: Feasible for research-scale 3D physics simulations

### 4.2. Numerical Stability

**Challenges:**
1. Hessian eigenvalues can have large dynamic range
2. Division by small $R$ in reward function
3. Gradient computation near eigenvalue crossings

**Solutions:**
1. Always add regularization $\epsilon_R > 0$ to denominators
2. Use stable eigenvalue algorithms (QR, Jacobi)
3. Clip gradients if $\|\nabla R\| > G_{\max}$
4. Monitor condition number of $H$: if $\kappa(H) > 10^{10}$, increase $\epsilon_\Sigma$

---

## 5. Experimental Research Program

### 5.1. Phase 1: Toy Problems with Known Geometry

**Objective**: Validate that Ricci curvature correlates with structure

**Test Case 1: 3D Sphere Embedding**
- Embed 2D sphere in 3D: $x^2 + y^2 + z^2 = R^2$
- Known analytical Ricci curvature
- Test: Does Ricci Gas concentrate on sphere?

**Test Case 2: Double-Well Potential**
- $V(x) = (x^2 - 1)^2 + y^2 + z^2$
- Two minima at $(±1, 0, 0)$
- Test: Do walkers concentrate at minima or explore saddle?

**Test Case 3: 3D Rastrigin Function**
- $V(x) = 10d + \sum_{i=1}^3 (x_i^2 - 10\cos(2\pi x_i))$
- Many local minima in fractal-like pattern
- Test: Does negative curvature correlate with basin boundaries?

### 5.2. Phase 2: Physical Simulations

**Test Case 4: Rigid Body Dynamics**
- Optimize configuration of rigid bodies under gravity
- State space: 3D positions + quaternions (6D per body)
- Test: Does curvature guide discovery of stable configurations?

**Test Case 5: Molecular Dynamics**
- Lennard-Jones particles in 3D
- Find low-energy configurations
- Compare Ricci Gas vs standard Euclidean/Adaptive Gas

### 5.3. Phase 3: Phase Transition Detection

**Objective**: Experimentally locate $\alpha_c$

**Protocol:**
1. Fix all parameters except $\alpha = \epsilon_R$
2. Run simulations for $\alpha \in \{0.001, 0.01, 0.1, 1, 10\}$
3. Measure:
   - Variance of swarm: $\sigma^2(t) = \frac{1}{N} \sum_i \|x_i - \bar{x}\|^2$
   - Entropy: $S(t) = -\sum_{\text{bins}} p_i \log p_i$ (histogram-based)
   - Max curvature: $R_{\max}(t) = \max_i R_i(t)$
4. Identify $\alpha_c$ where:
   - $\sigma^2(\infty)$ drops sharply
   - $S(\infty)$ drops sharply
   - $R_{\max}$ diverges

### 5.4. Phase 4: Ablation Studies

Compare four variants:

| Variant | Force | Reward | Hypothesis |
|---------|-------|--------|------------|
| A (Ricci Gas) | $+\nabla R$ | $1/R$ | Push-pull balance → structured collapse |
| B (Aligned) | $-\nabla R$ | $1/R$ | Both seek flat → broad exploration |
| C (Force Only) | $+\nabla R$ | 0 (standard) | Pure aggregation → total collapse |
| D (Reward Only) | 0 (standard) | $1/R$ | Pure dispersion → diffuse gas |

**Expected outcome**: Variant A (Ricci Gas) shows richest behavior with phase transition.

---

## 6. Theoretical Open Questions

### 6.1. Rigorous Mathematical Questions

1. **Prove or bound $\alpha_c$ for 3D Ricci Gas**
   - What is the exact critical value?
   - How does it depend on $\ell, \epsilon_\Sigma, \beta$?

2. **Characterize the supercritical QSD**
   - Does it exhibit fractal dimension?
   - Can it be expressed as a sum of delta functions plus a diffuse background?

3. **Prove an LSI for the collapsed state**
   - Does a modified LSI hold for the supercritical QSD?
   - With what constant (as function of $\alpha - \alpha_c$)?

4. **Connection to Ricci flow**
   - Is the mean-field PDE related to Ricci flow on the emergent manifold?
   - Can techniques from Hamilton's Ricci flow be applied?

### 6.2. Computational Questions

1. **Optimal KDE bandwidth $\ell$**
   - How does $\ell$ affect $\alpha_c$?
   - Trade-off: small $\ell$ → sharper features, but higher computational cost

2. **Fast Ricci computation**
   - Can we learn a neural network proxy for $R[\rho]$?
   - Trade accuracy for speed

3. **Parallelization**
   - How to efficiently parallelize KDE + Hessian + eigendecomp on GPU?

### 6.3. Physical Interpretation Questions

1. **What physical systems does this model?**
   - Self-gravitating Bose-Einstein condensates?
   - Active matter with long-range interactions?
   - Information geometry of learning systems?

2. **Is the "bouncing singularity" physical?**
   - Does it correspond to a known phenomenon in physics?
   - Hawking radiation from black holes?

---

## 7. Conclusion and Future Directions

### 7.1. Summary of Contributions

The Ricci Fragile Gas represents a novel synthesis of:
1. **Differential geometry** (Ricci curvature, emergent metrics)
2. **Nonlinear PDE theory** (PKS models, blow-up analysis)
3. **Statistical physics** (phase transitions, LSI, QSD)
4. **Algorithmic search** (Fragile Gas framework)

**Key theoretical contributions:**
- First self-referential geometry-driven swarm algorithm
- Prediction of phase transition from diffuse to collapsed state
- Multi-layered stability architecture
- Connection between LSI breakdown and structure formation

**Key practical contributions:**
- Computationally feasible for 3D physics ($d=3$)
- Clear experimental protocol to validate theory
- Potential applications: molecular dynamics, rigid body optimization, manifold exploration

### 7.2. Immediate Next Steps

1. **Implement Python prototype** (see `src/fragile/ricci_gas.py`)
2. **Run Phase 1 experiments** (sphere, double-well, Rastrigin)
3. **Measure $\alpha_c$ empirically**
4. **Visualize curvature heatmaps and phase diagrams**

### 7.3. Long-Term Research Directions

1. **Rigorous analysis**: Prove existence of $\alpha_c$ for 3D Ricci proxy
2. **Generalization**: Extend to higher dimensions, other curvature tensors
3. **Applications**: Apply to real-world 3D physics optimization problems
4. **Hybrid models**: Combine Ricci forces with task-specific rewards

---

## References

- **`02_euclidean_gas.md`** — Base Langevin dynamics and BAOAB integrator
- **`03_cloning.md`** — Keystone Principle for cloning operator
- **`04_convergence.md`** — Hypocoercivity and LSI analysis
- **`07_adaptative_gas.md`** — Adaptive Gas, Hessian-based metric, perturbation theory
- **`08_emergent_geometry.md`** — Emergent Riemannian geometry, convergence on manifolds
- **Patlak (1953), Keller-Segel (1970)** — Chemotaxis models and blow-up theory
- **Carrillo et al. (2010)** — "Contractions in Wasserstein Distance for PKS Equations"
- **Dolbeault & Perthame (2004)** — "Optimal Critical Mass for 2D Keller-Segel"

---

:::{admonition} Document Status
:class: note

**This document presents:**
✅ Complete mathematical framework for Ricci Fragile Gas
✅ Push-pull architecture with clear physical analogy
✅ Phase transition conjecture with proof strategy
✅ Computational feasibility analysis for 3D
✅ Experimental research program

**This document does NOT include:**
❌ Full proof of $\alpha_c$ existence (open conjecture)
❌ Empirical validation (requires implementation)
❌ Comparison with baseline methods (requires experiments)

**Status**: Ready for implementation and empirical testing.

:::
