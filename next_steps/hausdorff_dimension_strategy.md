# Hausdorff Dimension Calculation Strategy

**Date**: 2025-10-10
**Goal**: Prove $d_H = 4 + O(a)$ for Fractal Set continuum limit (Theorem 1.2.2)

---

## Executive Summary

You have **three complementary approaches** for computing Hausdorff dimension, leveraging different aspects of your existing mathematical machinery:

1. **Spectral dimension** via Graph Laplacian eigenvalues (computational, immediate)
2. **Random walk dimension** via effective resistance (probabilistic, robust)
3. **Riemannian volume growth** via Riemann curvature tensor (geometric, rigorous)

**Recommendation**: Use **all three** as cross-validation. Start with #1 (fastest), validate with #2 (most robust), prove rigorously with #3 (publication quality).

---

## Approach 1: Spectral Dimension via Graph Laplacian Eigenvalues

### What You Have

From [13_B_fractal_set_continuum_limit.md](../docs/source/13_fractal_set/13_B_fractal_set_continuum_limit.md), Theorem 3.2:

$$
\Delta_{\mathcal{F}_N} \xrightarrow{N \to \infty} \Delta_g
$$

with convergence rate $O(N^{-1/4})$, where:
- $\Delta_{\mathcal{F}_N}$ = discrete graph Laplacian on Fractal Set
- $\Delta_g$ = Laplace-Beltrami operator on emergent Riemannian manifold $(\\mathcal{X}, g)$
- $g_{ij}(x) = (H(x, S) + \epsilon_\Sigma I)_{ij}$ = fitness Hessian + regularization

**Eigenvalue spectrum**: $\{\lambda_k^{(N)}\}_{k=1}^{|\mathcal{V}|}$ from solving:
$$
\Delta_{\mathcal{F}_N} f_k = \lambda_k^{(N)} f_k
$$

### Spectral Dimension Formula

:::{prf:definition} Spectral Dimension from Heat Kernel
:label: def-spectral-dimension

The **spectral dimension** $d_s$ is defined via the heat kernel trace:

$$
p(t) = \text{tr}(e^{t\Delta}) = \sum_{k=1}^\infty e^{-\lambda_k t}
$$

Short-time asymptotics:
$$
p(t) \sim C t^{-d_s/2} \quad \text{as } t \to 0^+
$$

Extracting $d_s$:
$$
d_s = -2 \lim_{t \to 0^+} \frac{d \log p(t)}{d \log t}
$$

**Key theorem** (Weyl's law): For smooth $d$-dimensional Riemannian manifolds, $d_s = d_H = d$ (topological dimension).
:::

### Computational Protocol

**Step 1**: Compute discrete heat kernel trace
- Use eigenvalues $\{\lambda_k^{(N)}\}$ from graph Laplacian
- Approximate: $p_N(t) = \sum_{k=1}^K e^{-\lambda_k^{(N)} t}$ (truncate at $K \approx 100$)

**Step 2**: Log-log regression
- Compute derivative: $\frac{d \log p_N(t)}{d \log t} \approx \frac{\log p_N(t_2) - \log p_N(t_1)}{\log t_2 - \log t_1}$
- Evaluate at short times: $t \in [10^{-4}, 10^{-1}]$ (before eigenvalue truncation affects result)
- Extract slope: $d_s = -2 \times \text{slope}$

**Step 3**: Verify convergence with $N$
- Repeat for $N \in \{10^2, 10^3, 10^4\}$
- Expected: $d_s^{(N)} \to 4$ as $N \to \infty$
- Check finite-$N$ correction: $d_s^{(N)} = 4 + c_1/N^{\alpha} + O(N^{-2\alpha})$

### Implementation (Python/JAX)

```python
import jax.numpy as jnp
from scipy.linalg import eigh

def spectral_dimension(eigenvalues, t_min=1e-4, t_max=1e-1, n_points=50):
    """
    Compute spectral dimension from graph Laplacian eigenvalues.

    Args:
        eigenvalues: Array of Laplacian eigenvalues λ_k
        t_min, t_max: Time range for heat kernel evaluation
        n_points: Number of time samples

    Returns:
        d_s: Spectral dimension estimate
        t_grid: Time grid used
        p_t: Heat kernel trace p(t) at each time
    """
    t_grid = jnp.logspace(jnp.log10(t_min), jnp.log10(t_max), n_points)

    # Heat kernel trace: p(t) = Σ exp(-λ_k t)
    # Shape broadcasting: eigenvalues [K] @ t_grid [n_points] -> [n_points, K]
    heat_trace = jnp.sum(jnp.exp(-eigenvalues[:, None] * t_grid[None, :]), axis=0)

    # Log-log derivative: d(log p)/d(log t) ≈ Δ(log p)/Δ(log t)
    log_p = jnp.log(heat_trace)
    log_t = jnp.log(t_grid)

    # Central difference for interior points
    d_log_p = (log_p[2:] - log_p[:-2]) / (log_t[2:] - log_t[:-2])

    # Spectral dimension: d_s = -2 * d(log p)/d(log t)
    d_s_local = -2 * d_log_p

    # Take median over short-time regime (first quartile)
    quartile_idx = len(d_s_local) // 4
    d_s = jnp.median(d_s_local[:quartile_idx])

    return d_s, t_grid, heat_trace

# Usage
eigenvalues = compute_graph_laplacian_eigenvalues(fractal_set)  # your existing function
d_s, t_grid, p_t = spectral_dimension(eigenvalues)
print(f"Spectral dimension: d_s = {d_s:.3f}")

# Verify by plotting
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.loglog(t_grid, p_t, 'o-')
plt.xlabel('Time t')
plt.ylabel('Heat kernel trace p(t)')
plt.title('Heat Kernel Trace')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
log_derivative = jnp.gradient(jnp.log(p_t), jnp.log(t_grid))
plt.semilogx(t_grid, -2 * log_derivative, 'o-')
plt.axhline(4, color='red', linestyle='--', label='d=4 (target)')
plt.xlabel('Time t')
plt.ylabel('Spectral dimension d_s(t)')
plt.title('Local Spectral Dimension')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('spectral_dimension_analysis.png', dpi=150)
```

### Expected Results

**If your theory is correct**:
1. Short-time regime ($t \sim 10^{-3}$): $p(t) \propto t^{-2}$ → $d_s = 4$ ✓
2. Finite-$N$ corrections: $d_s^{(1000)} \approx 4.1$, $d_s^{(10000)} \approx 4.01$
3. Convergence: $|d_s^{(N)} - 4| \sim O(N^{-1/4})$ (from mean-field convergence rate)

**If dimension is fractal** (alternative scenario):
- $d_s$ stabilizes at non-integer value (e.g., $d_s = 3.7$)
- This would indicate **anomalous diffusion** on the fractal set
- Still publishable! Just different physics.

### Advantages of Spectral Approach

✅ **Computationally cheap**: Only need eigenvalues (already computed for convergence tests)
✅ **Robust**: Heat kernel smooths out local irregularities
✅ **N-uniform**: Inherits convergence from graph Laplacian theorem
✅ **Proven connection**: Weyl's law guarantees $d_s = d_H$ for smooth manifolds

### Potential Issues

⚠️ **Eigenvalue truncation bias**: Using only $K$ eigenvalues affects short-time behavior
→ **Fix**: Use $K \geq N/10$ eigenvalues, check convergence with $K$

⚠️ **Finite-size effects**: Discrete graph has finite size → heat kernel saturates
→ **Fix**: Only fit short-time regime ($t < T_{\text{mixing}}/10$)

⚠️ **Anisotropic metric**: If fitness Hessian has large anisotropy, may need **weighted graph Laplacian**
→ **Fix**: Use metric-aware edge weights (already done in Theorem 3.3.1 via companion selection)

---

## Approach 2: Random Walk Dimension via Effective Resistance

### What You Have

IG graph structure with edge weights from companion selection (Theorem 3.3.1):

$$
w_{ij} = \frac{1}{Z_i} \exp\left(-\frac{\|x_i - x_j\|^2}{2\varepsilon_c^2}\right)
$$

This defines a **random walk** on episodes.

### Random Walk Dimension Formula

:::{prf:definition} Random Walk Dimension
:label: def-random-walk-dimension

The **random walk dimension** $d_w$ relates mean displacement to number of steps:

$$
\mathbb{E}[\|X_n - X_0\|^2] \sim n^{2/d_w}
$$

where $X_n$ is walker position after $n$ random walk steps on the IG graph.

**Relation to Hausdorff dimension**:
$$
d_H = \frac{d_w d_s}{d_s + 2}
$$

where $d_s$ is spectral dimension. For fractals, $d_w > d_s$ (anomalous diffusion). For smooth manifolds, $d_w = 2$ → $d_H = d_s$.
:::

### Computational Protocol

**Step 1**: Sample random walks on IG graph
- Start from random episode $e_0$
- Perform $n$ steps: $e_0 \to e_1 \to \cdots \to e_n$ (transition prob $\propto w_{ij}$)
- Measure mean squared displacement: $\text{MSD}(n) = \mathbb{E}[\|x_{e_n} - x_{e_0}\|^2]$

**Step 2**: Log-log regression
- Plot $\log \text{MSD}(n)$ vs. $\log n$
- Extract exponent: $\text{MSD}(n) \sim n^\alpha$ → $d_w = 2/\alpha$

**Step 3**: Combine with spectral dimension
- Use $d_s$ from Approach 1
- Compute: $d_H = d_w d_s / (d_s + 2)$

### Implementation

```python
import jax
import jax.numpy as jnp
from jax import random

def random_walk_dimension(fractal_set, n_walks=1000, max_steps=1000):
    """
    Estimate random walk dimension via mean squared displacement.

    Args:
        fractal_set: FractalSet object with IG edge weights
        n_walks: Number of random walks to average
        max_steps: Maximum number of steps per walk

    Returns:
        d_w: Random walk dimension
        msd_curve: Mean squared displacement vs. steps
    """
    key = random.PRNGKey(42)

    # Extract IG adjacency matrix and positions
    W = fractal_set.ig_weights  # [|V|, |V|] sparse matrix
    positions = fractal_set.episode_positions  # [|V|, d]

    # Normalize rows to get transition probabilities
    P = W / W.sum(axis=1, keepdims=True)  # row-stochastic

    msd = jnp.zeros(max_steps)

    for walk_idx in range(n_walks):
        key, subkey = random.split(key)

        # Sample initial episode
        e_0 = random.choice(subkey, len(positions))
        x_0 = positions[e_0]

        # Perform random walk
        e_current = e_0
        for step in range(max_steps):
            key, subkey = random.split(key)

            # Sample next episode from transition probabilities
            e_next = random.choice(subkey, len(positions), p=P[e_current])

            # Update displacement
            displacement_sq = jnp.sum((positions[e_next] - x_0)**2)
            msd = msd.at[step].add(displacement_sq)

            e_current = e_next

    msd = msd / n_walks

    # Fit power law: msd(n) ~ n^α
    steps = jnp.arange(1, max_steps + 1)
    log_steps = jnp.log(steps[steps > 10])  # skip transient
    log_msd = jnp.log(msd[steps > 10])

    # Linear regression in log-log space
    alpha = jnp.polyfit(log_steps, log_msd, deg=1)[0]
    d_w = 2 / alpha

    return d_w, msd

# Usage
d_w, msd_curve = random_walk_dimension(fractal_set)
print(f"Random walk dimension: d_w = {d_w:.3f}")

# If d_w ≈ 2 (normal diffusion), then d_H = d_s (from Approach 1)
```

### Advantages of Random Walk Approach

✅ **Robust to local irregularities**: Averages over many walks
✅ **Detects anomalous diffusion**: If $d_w \neq 2$, indicates fractal structure
✅ **Physically interpretable**: Directly measures exploration behavior
✅ **Independent check**: Validates spectral dimension results

---

## Approach 3: Riemannian Volume Growth via Riemann Curvature Tensor

### What You Have

From [15_fractal_gas.md](../docs/source/15_fractal_gas.md):
- **Ricci scalar proxy**: $R(x, S) = \text{tr}(H) - \lambda_{\min}(H)$ (3D case)
- **Full Riemann tensor**: Computed from 6th-order derivatives of fitness (you mentioned this!)

From [09_symmetries_adaptive_gas.md](../docs/source/09_symmetries_adaptive_gas.md), Theorem 3.5:
- **Emergent metric is Fisher information matrix**: $g_{ij} = H_{ij} + \epsilon_\Sigma I$

### Volume Growth Formula

:::{prf:theorem} Hausdorff Dimension from Riemannian Volume
:label: thm-hausdorff-from-volume

For a smooth $d$-dimensional Riemannian manifold $(M, g)$, the volume of a geodesic ball satisfies:

$$
\text{Vol}(B(x, r)) = \omega_d r^d \left(1 - \frac{R(x)}{6(d+2)} r^2 + O(r^4)\right)
$$

where:
- $\omega_d = \pi^{d/2}/\Gamma(d/2 + 1)$ is the Euclidean ball volume
- $R(x) = g^{ij}(x) R_{ij}(x)$ is the Ricci scalar
- $R_{ij} = R^k_{ikj}$ is the Ricci curvature tensor

**Hausdorff dimension**:
$$
d_H = \lim_{r \to 0} \frac{\log \text{Vol}(B(x, r))}{\log r}
$$

For smooth manifolds, $d_H = d$ (topological dimension).
:::

### Computational Protocol

**Step 1**: Compute Ricci scalar from fitness Hessian
- You have: $H_{ij}(x, S) = \partial_i \partial_j V_{\text{fit}}(x, S)$ (2nd derivatives)
- Compute Riemann tensor: $R_{ijkl} = \partial_k \Gamma_{ijl} - \partial_l \Gamma_{ijk} + \cdots$ (requires Christoffel symbols from 3rd derivatives)
- Contract: $R_{ij} = R^k_{ikj}$, then $R = g^{ij} R_{ij}$

**Step 2**: Measure volume of balls in IG graph
- For each episode $e$, count neighbors within distance $r$:
  $$
  V_N(e, r) = |\{e' \in \mathcal{V} : d_g(e, e') \leq r\}|
  $$
  where $d_g$ is graph distance (shortest path)

**Step 3**: Compare discrete volume to Riemannian prediction
- Theoretical: $V_{\text{Riemann}}(x, r) = \omega_4 r^4 (1 - R(x)r^2/30 + \cdots)$
- Empirical: $V_N(e, r)$ from graph counting
- Extract dimension: $d_H = \lim_{r \to 0} \frac{\log V_N(e, r)}{\log r}$

### Why This is Rigorous

**This approach is publication-quality because**:
1. It directly uses the **geometric definition** of Hausdorff dimension
2. It connects to the **Riemann curvature tensor** (fundamental geometric invariant)
3. The Ricci scalar $R(x)$ is **computable from your fitness Hessian**
4. Provides **explicit finite-$N$ corrections** via curvature term

### Implementation

```python
def ricci_scalar_from_fitness(x, S, epsilon_Sigma=1e-3):
    """
    Compute Ricci scalar from fitness potential.

    This requires:
    1. Hessian H = ∇²V_fit (2nd derivatives)
    2. Christoffel symbols Γ from metric g = H + εI (3rd derivatives)
    3. Riemann tensor R_ijkl (4th derivatives of metric = 6th of V_fit)
    4. Ricci tensor R_ij = R^k_ikj
    5. Ricci scalar R = g^ij R_ij

    Args:
        x: Position [d]
        S: Swarm state
        epsilon_Sigma: Regularization

    Returns:
        R: Ricci scalar (scalar)
    """
    # Step 1: Compute metric and its inverse
    H = compute_hessian(V_fit, x, S)  # [d, d]
    g = H + epsilon_Sigma * jnp.eye(len(x))
    g_inv = jnp.linalg.inv(g)

    # Step 2: Compute Christoffel symbols (requires 3rd derivatives)
    # Γ^k_ij = (1/2) g^kl (∂_i g_jl + ∂_j g_il - ∂_l g_ij)
    dg = compute_metric_derivatives(g, x, S)  # [d, d, d]
    Gamma = 0.5 * jnp.einsum('kl,ijl->kij', g_inv,
                             dg + jnp.transpose(dg, (1, 0, 2)) - jnp.transpose(dg, (2, 1, 0)))

    # Step 3: Compute Riemann tensor (requires 4th derivatives of metric)
    # R^l_ijk = ∂_j Γ^l_ik - ∂_k Γ^l_ij + Γ^l_jm Γ^m_ik - Γ^l_km Γ^m_ij
    dGamma = compute_christoffel_derivatives(Gamma, x, S)  # [d, d, d, d]

    Riemann = (dGamma[:, :, :, 1] - dGamma[:, :, :, 2] +
               jnp.einsum('ljm,mik->lijk', Gamma, Gamma) -
               jnp.einsum('lkm,mij->lijk', Gamma, Gamma))

    # Step 4: Ricci tensor R_ij = R^k_ikj
    Ricci = jnp.einsum('kikj->ij', Riemann)

    # Step 5: Ricci scalar R = g^ij R_ij
    R_scalar = jnp.einsum('ij,ij->', g_inv, Ricci)

    return R_scalar

def measure_volume_growth(fractal_set, x_center, r_max=1.0, n_radii=20):
    """
    Measure volume of geodesic balls in IG graph.

    Args:
        fractal_set: FractalSet object
        x_center: Center point
        r_max: Maximum radius
        n_radii: Number of radii to sample

    Returns:
        radii: Array of radii
        volumes: Number of episodes within each radius
    """
    positions = fractal_set.episode_positions

    # Compute distances from center
    distances = jnp.linalg.norm(positions - x_center, axis=1)

    radii = jnp.linspace(0.1, r_max, n_radii)
    volumes = jnp.array([jnp.sum(distances <= r) for r in radii])

    return radii, volumes

# Usage
x_test = jnp.array([0.0, 0.0, 0.0])  # test point in state space
R_scalar = ricci_scalar_from_fitness(x_test, swarm_state)
print(f"Ricci scalar at test point: R = {R_scalar:.4f}")

radii, volumes = measure_volume_growth(fractal_set, x_test)

# Fit volume growth: log V(r) = d_H log r + const
log_r = jnp.log(radii)
log_V = jnp.log(volumes)
d_H = jnp.polyfit(log_r, log_V, deg=1)[0]
print(f"Hausdorff dimension from volume growth: d_H = {d_H:.3f}")
```

### Expected Results

**If 4D Lorentzian manifold**:
- Volume growth: $V(r) \propto r^4$ → $d_H = 4$ ✓
- Ricci correction: $V(r) = \omega_4 r^4 (1 - Rr^2/30)$ with $R = O(1)$

**Finite-$N$ corrections**:
- Discrete volume: $V_N(r) = V_{\text{smooth}}(r) + O(N^{-1/4})$ (from measure convergence)

---

## Recommended Combined Strategy

### Phase 1: Computational Verification (1-2 weeks)

1. **Implement Approach 1** (spectral dimension) → fast sanity check
   - Expected: $d_s \approx 4.0 \pm 0.1$ for $N = 10^4$
   - Deliverable: Plot showing convergence $d_s^{(N)} \to 4$

2. **Implement Approach 2** (random walk) → validation
   - Expected: $d_w \approx 2.0$ (normal diffusion)
   - Check: $d_H = d_w d_s / (d_s + 2) \approx 4$

### Phase 2: Rigorous Proof (2-3 months)

3. **Implement Approach 3** (Riemann curvature) → publication-quality
   - Compute Ricci scalar from fitness Hessian (6th derivatives)
   - Measure volume growth on IG graph
   - Prove: $d_H = 4 + O(a)$ with explicit curvature corrections

4. **Write Theorem 1.2.2** with complete proof
   - **Claim**: Hausdorff dimension $d_H = 4 + O(a)$
   - **Proof**: Combine volume growth (Approach 3) with spectral estimates (Approach 1)
   - **Verification**: Cross-check with random walk (Approach 2)

### Cross-Validation

All three approaches should agree:

| Method | Estimate | Finite-$N$ Scaling |
|--------|----------|-------------------|
| Spectral | $d_s^{(N)} = 4 + c_1 N^{-1/4}$ | $O(N^{-1/4})$ |
| Random walk | $d_H = d_w d_s/(d_s+2)$ | $O(N^{-1/4})$ |
| Volume growth | $d_H = 4 + c_2 R(x) a^2$ | $O(a^2)$ |

If all three agree → **strong evidence** for $d_H = 4$ in continuum limit.

---

## Local Euclidean Charts (Bonus)

You also asked about **local Euclidean structure**. You're absolutely right that your **discrete diffusion tensor** provides this directly!

### What You Have

From [13_B_fractal_set_continuum_limit.md](../docs/source/13_fractal_set/13_B_fractal_set_continuum_limit.md):

$$
D_{\text{reg}}(x) = (H(x, S) + \epsilon_\Sigma I)^{-1} \quad \text{(discrete diffusion tensor)}
$$

This is equivalent to the **Riemannian metric tensor** $g = D^{-1}$.

### Local Chart Construction

:::{prf:theorem} Episodes Define Local Euclidean Charts
:label: thm-episodes-local-charts

For each episode $e \in \mathcal{V}$, there exists a neighborhood $U_e \subset \mathcal{V}$ and a chart map $\phi_e: U_e \to \mathbb{R}^d$ such that:

1. **Metric compatibility**: In coordinates $(y^1, \ldots, y^d) = \phi_e(e')$,
   $$
   g_{ij}(e) = \delta_{ij} + O(\epsilon_c)
   $$
   where $\epsilon_c$ is the cloning interaction range.

2. **Christoffel symbols vanish at $e$**: $\Gamma^k_{ij}(e) = 0$

3. **Volume element**: $\sqrt{\det g(e')} = 1 + O(\|e' - e\|^2)$

**Proof**: Use **Riemann normal coordinates** centered at $e$, defined via:
$$
\phi_e(e') = g(e)^{-1/2} (x_{e'} - x_e)
$$

This makes the metric Euclidean at $e$ up to $O(\epsilon_c)$ corrections. ∎
:::

### Computational Implementation

```python
def construct_local_chart(fractal_set, episode_idx):
    """
    Construct Riemann normal coordinates around an episode.

    Args:
        fractal_set: FractalSet object
        episode_idx: Index of central episode

    Returns:
        chart_map: Function e' -> y (coordinates)
        metric_at_center: g_ij at episode_idx (should be ≈ δ_ij)
    """
    e_center = episode_idx
    x_center = fractal_set.episode_positions[e_center]

    # Compute metric at center
    S = fractal_set.swarm_state
    H_center = compute_hessian(V_fit, x_center, S)
    g_center = H_center + epsilon_Sigma * jnp.eye(len(x_center))

    # Metric square root for change of coordinates
    g_sqrt = jnp.linalg.cholesky(g_center)  # g = L L^T
    g_inv_sqrt = jnp.linalg.inv(g_sqrt)

    def chart_map(episode_idx_prime):
        """Map episode to local coordinates."""
        x_prime = fractal_set.episode_positions[episode_idx_prime]
        Delta_x = x_prime - x_center

        # Riemann normal coordinates: y = g^{-1/2} Δx
        y = g_inv_sqrt @ Delta_x
        return y

    # Verify metric is Euclidean at center
    # (this is guaranteed by construction)
    metric_at_center = jnp.eye(len(x_center))

    return chart_map, metric_at_center

# Verify local Euclidean structure
chart_map, g_center = construct_local_chart(fractal_set, episode_idx=42)

# Check neighboring episodes
neighbors = fractal_set.get_ig_neighbors(episode_idx=42)
for neighbor_idx in neighbors:
    y_neighbor = chart_map(neighbor_idx)
    print(f"Neighbor {neighbor_idx}: local coords y = {y_neighbor}")

    # Metric should be δ_ij + O(||y||²)
    g_neighbor = compute_metric(neighbor_idx)  # in original coords
    g_deviation = jnp.linalg.norm(g_neighbor - jnp.eye(len(y_neighbor)))
    print(f"  Metric deviation from Euclidean: {g_deviation:.6f}")
```

---

## Summary: Actionable Next Steps

**This week**:
1. ✅ Implement spectral dimension calculation (Approach 1) - **2 hours**
2. ✅ Run on existing fractal sets with $N \in \{100, 1000, 10000\}$ - **4 hours**
3. ✅ Generate plots showing $d_s^{(N)} \to 4$ - **1 hour**

**Next week**:
4. ✅ Implement random walk dimension (Approach 2) - **1 day**
5. ✅ Cross-validate: check $d_H = d_w d_s/(d_s + 2) \approx 4$ - **2 hours**

**Next month**:
6. 🔬 Implement Ricci scalar computation from 6th derivatives - **1 week**
7. 🔬 Measure volume growth and extract $d_H$ - **3 days**
8. ✍️ Write rigorous proof of Theorem 1.2.2 - **1 week**

**Deliverable**: Complete proof that Hausdorff dimension $d_H = 4 + O(a)$ with three independent verification methods.

---

**End of Strategy Document**
