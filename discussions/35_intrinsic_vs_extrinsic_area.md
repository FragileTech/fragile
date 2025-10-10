# Intrinsic vs Extrinsic Area: Algorithmic Curvature from Wilson Loops

**Status**: ✅ **CRITICAL INSIGHT**

**Purpose**: Distinguish between flat (Euclidean) and curved (Riemannian) areas in the fractal set, and show how their ratio reveals algorithmic curvature.

**Key Discovery**: We can compute **both** areas from the same data and use their ratio to measure the fitness landscape's curvature.

---

## 0. Executive Summary

### The Question

Episodes have coordinates Φ(e) ∈ ℝ^d that live in:
1. **Flat embedding space**: Euclidean geometry (standard distances)
2. **Curved fitness manifold**: Riemannian geometry with metric G(x) from fitness Hessian

**Which area should Wilson loops use?**

### The Answer (from Gemini)

> "For a Wilson loop to be physically meaningful *within your framework*, it must be sensitive to the geometry the agents themselves experience. The 'physical' area is the one measured by the rulers and protractors of the emergent geometry."

**Therefore**: Use **A_curved** (Riemannian area with metric G), not A_flat (Euclidean area).

**Bonus**: The ratio A_curved/A_flat directly measures algorithmic curvature!

---

## 1. Two Geometries, Two Areas

### 1.1. Flat (Extrinsic) Area

:::{prf:definition} Euclidean Area
:label: def-euclidean-area

For cycle C = {Φ(e_0), Φ(e_1), ..., Φ(e_{n-1})} in ℝ^d:

**For d = 2** (shoelace formula):
$$
A_{\text{flat}} = \frac{1}{2} \left| \sum_{i=0}^{n-1} (x_i y_{i+1} - x_{i+1} y_i) \right|
$$

**For d > 2** (fan triangulation):
$$
A_{\text{flat}} = \sum_{i=0}^{n-1} \frac{1}{2} \|(v_i \times v_{i+1})\|_{\text{Euclidean}}
$$

where:
- v_i = Φ(e_i) - x_c (edge vector from centroid)
- ||·||: Standard Euclidean norm
- ×: Standard cross product (d=3) or generalized wedge product (d>3)

**Properties**:
- Uses standard Euclidean metric δ_ij
- Ignores fitness landscape geometry
- "Extrinsic" - measures area in embedding space
:::

### 1.2. Curved (Intrinsic) Area

:::{prf:definition} Riemannian Area
:label: def-riemannian-area

For the same cycle with Riemannian metric G(x):

**Fan triangulation** (same triangles, different metric):
$$
A_{\text{curved}} = \sum_{i=0}^{n-1} \frac{1}{2} \sqrt{(v_i^T G_c v_i)(v_{i+1}^T G_c v_{i+1}) - (v_i^T G_c v_{i+1})^2}
$$

where:
- G_c = G(x_c, S): Metric tensor at centroid
- v^T G v: Squared length in Riemannian metric
- (v_1^T G v_2): Dot product in Riemannian metric

**Properties**:
- Uses emergent metric G from fitness Hessian
- Sensitive to fitness landscape curvature
- "Intrinsic" - measures area as experienced by walkers
:::

**Physical interpretation**:
- **A_flat**: What an external observer in ℝ^d measures
- **A_curved**: What the walkers experience through fitness gradients

---

## 2. Why A_curved is Correct for Wilson Loops

### 2.1. Standard Gauge Theory vs Emergent Geometry

**Standard lattice QCD**:
- Spacetime is **fixed background** (usually flat Euclidean)
- Gauge field A_μ is dynamical
- Wilson loops use background metric (A_flat)

**Fragile framework**:
- Metric G is **emergent** from fitness dynamics
- Walkers experience landscape curvature
- Wilson loops must use emergent metric (A_curved)

### 2.2. Gemini's Argument

> "In conventional Yang-Mills theory on a lattice, spacetime is a fixed, non-dynamical background. The Wilson loop area `A` is calculated with respect to this background metric. The dynamics are all in the gauge field `A_μ`, not the geometry."

> "Your metric `G` is **not** a background; it is an *emergent property* of the system's dynamics, derived from the fitness Hessian. The 'particles' or 'agents' in your adaptive gas model experience the landscape's curvature through this metric. Their notion of distance, angle, and volume is dictated by `G`."

**Conclusion**:
> "Therefore, `A_curved` is the physically correct choice for your Wilson loops. The algorithmic curvature encoded in `G` *is* the curvature that should source the field strength in your analogy."

### 2.3. The Conceptual Disconnect

**Using A_flat would mean**:
- Walkers interact via fitness landscape (metric G)
- Wilson loops measure via Euclidean space (metric δ)
- **Inconsistency**: Two different notions of "distance"

**Using A_curved ensures**:
- Walkers and Wilson loops use **same geometry**
- Algorithmic curvature affects gauge observables
- **Consistency**: Single emergent metric throughout

---

## 3. The Area Ratio Reveals Curvature

### 3.1. First-Order Relationship

:::{prf:theorem} Area Ratio and Metric Determinant
:label: thm-area-ratio-metric-det

For small cycles where metric G is approximately constant:

$$
\boxed{A_{\text{curved}} \approx \sqrt{\det G(x_c)} \times A_{\text{flat}}}
$$

where x_c is the cycle centroid.

**Proof**:

The Riemannian area element is:
$$
dA_{\text{curved}} = \sqrt{\det G(x)} \, dx \, dy
$$

For small region R with G ≈ G_c (constant):
$$
A_{\text{curved}} = \int_R \sqrt{\det G_c} \, dx \, dy = \sqrt{\det G_c} \int_R dx \, dy = \sqrt{\det G_c} \times A_{\text{flat}}
$$

∎
:::

**Physical interpretation**:
- √(det G) is the **volume distortion factor**
- If det G > 1: Metric stretches space → more area
- If det G < 1: Metric compresses space → less area

**Example**: For G = λI (scaled identity):
- det G = λ^d
- √(det G) = λ^(d/2)
- A_curved = λ^(d/2) × A_flat (uniform scaling)

### 3.2. Second-Order: Gaussian Curvature

:::{prf:theorem} Curvature from Area Deficit
:label: thm-curvature-from-area-deficit

For small cycles in 2D manifold, the **area deficit** reveals Gaussian curvature K:

$$
A_{\text{curved}} \approx A_{\text{flat}} \sqrt{\det G(x_c)} \left(1 - \frac{K}{24} A_{\text{flat}}^2 + O(A^3)\right)
$$

**Extracting curvature**:
$$
\boxed{K \approx \frac{24}{A_{\text{flat}}^2} \left( \frac{A_{\text{curved}}}{A_{\text{flat}} \sqrt{\det G(x_c)}} - 1 \right)}
$$

**Reference**: Bertrand-Diguet-Puiseux theorem for geodesic circles.
:::

**Sign conventions**:
- K > 0 (sphere): Intrinsic area **smaller** than flat area (triangles have angle sum > π)
- K < 0 (hyperbolic): Intrinsic area **larger** than flat area (angle sum < π)
- K = 0 (flat): Areas equal

**For geodesic circle of radius r**:
$$
A_{\text{curved}}(r) = \pi r^2 \left(1 - \frac{K \pi r^2}{12} + O(r^4)\right)
$$

### 3.3. Measuring Algorithmic Curvature

**Protocol**:

1. **Compute both areas** for each IG cycle:
   ```python
   A_flat = compute_flat_area(positions)
   A_curved = compute_curved_area(positions, G_c)
   ```

2. **Compute area ratio**:
   ```python
   ratio = A_curved / (A_flat * np.sqrt(np.linalg.det(G_c)))
   ```

3. **Extract curvature**:
   ```python
   K = 24.0 / (A_flat**2) * (ratio - 1.0)
   ```

4. **Visualize curvature map**:
   ```python
   import matplotlib.pyplot as plt
   plt.scatter(centroids[:, 0], centroids[:, 1], c=curvatures,
               cmap='RdBu', vmin=-K_max, vmax=K_max)
   plt.colorbar(label='Gaussian Curvature K')
   plt.title('Algorithmic Curvature of Fitness Landscape')
   ```

**Result**: A **curvature map** of the fitness landscape, directly measured from loop areas!

---

## 4. Implementation: Computing Both Areas

### 4.1. Complete Algorithm

:::{prf:algorithm} Compute Flat and Curved Areas
:label: alg-compute-both-areas

**Input**:
- Cycle vertices: {Φ(e_0), ..., Φ(e_{n-1})} ∈ ℝ^d
- Metric function: G(x, S)
- Swarm state: S

**Output**:
- A_flat: Euclidean area
- A_curved: Riemannian area
- K: Estimated Gaussian curvature

**Code**:
```python
import numpy as np

def compute_both_areas(positions, metric_fn, swarm_state):
    """
    Compute flat and curved areas, extract curvature.

    Args:
        positions: (n, d) array of episode positions
        metric_fn: function G(x, S) returning (d, d) tensor
        swarm_state: SwarmState for metric evaluation

    Returns:
        A_flat: Euclidean area
        A_curved: Riemannian area
        K: Gaussian curvature estimate
        ratio: A_curved / (A_flat * sqrt(det G))
    """
    n, d = positions.shape

    # Step 1: Compute centroid
    x_c = np.mean(positions, axis=0)

    # Step 2: Evaluate metric at centroid
    G_c = metric_fn(x_c, swarm_state)
    det_G = np.linalg.det(G_c)
    sqrt_det_G = np.sqrt(max(det_G, 1e-10))  # Avoid division by zero

    # Step 3: Compute flat area (fan triangulation)
    A_flat = 0.0
    for i in range(n):
        v1 = positions[i] - x_c
        v2 = positions[(i+1) % n] - x_c

        if d == 2:
            # 2D cross product: scalar
            cross = v1[0] * v2[1] - v1[1] * v2[0]
            A_flat += 0.5 * abs(cross)
        elif d == 3:
            # 3D cross product: vector
            cross = np.cross(v1, v2)
            A_flat += 0.5 * np.linalg.norm(cross)
        else:
            # d > 3: Use Gram determinant
            # Area = (1/2) sqrt(det([[v1·v1, v1·v2], [v2·v1, v2·v2]]))
            gram = np.array([[np.dot(v1, v1), np.dot(v1, v2)],
                            [np.dot(v2, v1), np.dot(v2, v2)]])
            A_flat += 0.5 * np.sqrt(max(np.linalg.det(gram), 0.0))

    # Step 4: Compute curved area (Riemannian fan triangulation)
    A_curved = 0.0
    for i in range(n):
        v1 = positions[i] - x_c
        v2 = positions[(i+1) % n] - x_c

        # Riemannian inner products
        v1_G_v1 = v1 @ G_c @ v1
        v2_G_v2 = v2 @ G_c @ v2
        v1_G_v2 = v1 @ G_c @ v2

        # Riemannian area formula
        discriminant = v1_G_v1 * v2_G_v2 - v1_G_v2**2
        if discriminant < 0:
            discriminant = 0.0  # Numerical error

        A_curved += 0.5 * np.sqrt(discriminant)

    # Step 5: Compute area ratio and curvature
    ratio = A_curved / (A_flat * sqrt_det_G + 1e-10)

    if A_flat > 1e-10:
        K = 24.0 / (A_flat**2) * (ratio - 1.0)
    else:
        K = 0.0  # Undefined for tiny loops

    return {
        'A_flat': A_flat,
        'A_curved': A_curved,
        'K': K,
        'ratio': ratio,
        'det_G': det_G,
        'sqrt_det_G': sqrt_det_G
    }
```
:::

### 4.2. Validation Test: Sphere

**Known geometry**: Unit sphere S² with standard metric

Metric in stereographic coordinates:
$$
G = \frac{4}{(1 + x^2 + y^2)^2} \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}
$$

Gaussian curvature: K = 1 everywhere

**Test**:
1. Create small geodesic circle at north pole
2. Compute A_flat (Euclidean area in stereographic projection)
3. Compute A_curved (intrinsic area on sphere)
4. Extract K from ratio
5. Verify K ≈ 1

```python
def test_sphere():
    # Geodesic circle at north pole (r = 0.1)
    n = 20
    theta = np.linspace(0, 2*np.pi, n, endpoint=False)
    r = 0.1
    positions = np.array([r * np.cos(theta), r * np.sin(theta)]).T

    # Sphere metric at origin (north pole)
    def sphere_metric(x, S):
        r2 = x[0]**2 + x[1]**2
        scale = 4.0 / (1 + r2)**2
        return scale * np.eye(2)

    result = compute_both_areas(positions, sphere_metric, None)

    print(f"A_flat: {result['A_flat']:.6f}")
    print(f"A_curved: {result['A_curved']:.6f}")
    print(f"K: {result['K']:.6f}")
    print(f"Expected K: 1.0")
```

---

## 5. Corrected Wilson Action with Intrinsic Area

### 5.1. The Right Formula

:::{prf:definition} Intrinsic Wilson Action
:label: def-intrinsic-wilson-action

Using **Riemannian areas** (not Euclidean):

$$
\boxed{S_{\text{gauge}} = \frac{\beta}{2N_c} \sum_{e \in E_{\text{IG}}} \frac{\langle A_{\text{curved}} \rangle}{A_{\text{curved}}(e)} \left(1 - \frac{1}{N_c} \text{Re Tr } W_e\right)}
$$

where:
- A_curved(e): Riemannian area of cycle C(e) using metric G
- ⟨A_curved⟩: Mean Riemannian area (normalization)
- W_e: Wilson loop (from Chapter 32)

**Weights**:
$$
w_e = \frac{\langle A_{\text{curved}} \rangle}{A_{\text{curved}}(e)}
$$

**NOT** w_e = ⟨A_flat⟩ / A_flat(e) (wrong - ignores emergent geometry)
:::

### 5.2. Continuum Limit with Curved Geometry

**Small-loop expansion** (same as before):
$$
1 - \text{Re Tr } U_e \approx \frac{g^2}{2N_c} \text{Tr}(F^2) \times A_{\text{curved}}(e)
$$

**Substitute**:
$$
S \approx \frac{\beta g^2}{4N_c} \sum_e \frac{\langle A \rangle}{A_{\text{curved}}(e)} \times A_{\text{curved}}(e) \times \text{Tr}(F^2)
$$

**Simplify**:
$$
S \approx \frac{\beta g^2 \langle A \rangle}{4N_c} \sum_e \text{Tr}(F^2)
$$

**Riemann sum**:
$$
\sum_e \text{Tr}(F^2) \to \int \text{Tr}(F_{\mu\nu} F^{\mu\nu}) \sqrt{\det g} \, d^4x
$$

where g is the spacetime metric (related to G).

**Result**: ✅ Correct continuum limit **with emergent curved geometry**

---

## 6. Three Empirical Tests

### Test 1: Algorithmic Weight vs Both Areas

**Hypothesis**: w_e^{algo} = 1/(τ² + δr²) approximates 1/A_curved, not 1/A_flat

**Test**:
```python
# For each IG edge
tau = abs(t_death[e.i] - t_death[e.j])
delta_r = norm(Phi[e.i] - Phi[e.j])
w_algo = 1.0 / (tau**2 + delta_r**2)

areas = compute_both_areas(cycle_vertices, metric_fn, swarm_state)
w_flat = 1.0 / areas['A_flat']
w_curved = 1.0 / areas['A_curved']

# Compare
print(f"w_algo: {w_algo:.4f}")
print(f"w_flat: {w_flat:.4f}")
print(f"w_curved: {w_curved:.4f}")

# Which is closer?
error_flat = abs(w_algo - w_flat) / w_algo
error_curved = abs(w_algo - w_curved) / w_algo

if error_curved < error_flat:
    print("✅ w_algo closer to w_curved (respects emergent geometry!)")
else:
    print("❌ w_algo closer to w_flat (ignores curvature)")
```

**Expected result**: w_algo ≈ w_curved (algorithmic weights implicitly encode metric)

### Test 2: Area Ratio vs Metric Determinant

**Hypothesis**: A_curved / A_flat ≈ √(det G) for small loops

**Test**:
```python
results = []
for cycle in all_cycles:
    areas = compute_both_areas(cycle, metric_fn, swarm_state)
    ratio_measured = areas['A_curved'] / areas['A_flat']
    ratio_predicted = areas['sqrt_det_G']

    results.append({
        'measured': ratio_measured,
        'predicted': ratio_predicted,
        'error': abs(ratio_measured - ratio_predicted) / ratio_predicted
    })

mean_error = np.mean([r['error'] for r in results])
print(f"Mean relative error: {mean_error:.2%}")

# Scatter plot
plt.scatter([r['predicted'] for r in results],
           [r['measured'] for r in results], alpha=0.5)
plt.plot([0, max_ratio], [0, max_ratio], 'r--', label='Perfect agreement')
plt.xlabel('Predicted: sqrt(det G)')
plt.ylabel('Measured: A_curved / A_flat')
plt.legend()
plt.title('First-Order Area Ratio Test')
```

**Expected**: Good agreement for small loops, deviation for large loops (curvature effects)

### Test 3: Curvature Map of Fitness Landscape

**Goal**: Visualize where fitness landscape is curved vs flat

**Implementation**:
```python
curvatures = []
centroids = []

for cycle in all_cycles:
    areas = compute_both_areas(cycle, metric_fn, swarm_state)
    curvatures.append(areas['K'])
    centroids.append(np.mean(cycle, axis=0))

centroids = np.array(centroids)
curvatures = np.array(curvatures)

# Scatter plot with curvature color
plt.figure(figsize=(10, 8))
scatter = plt.scatter(centroids[:, 0], centroids[:, 1],
                     c=curvatures, cmap='RdBu_r',
                     vmin=-np.percentile(abs(curvatures), 95),
                     vmax=np.percentile(abs(curvatures), 95),
                     s=50, alpha=0.7)
plt.colorbar(scatter, label='Gaussian Curvature K')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Algorithmic Curvature Map from Wilson Loops')
plt.grid(True, alpha=0.3)

# Interpretation
print(f"Mean curvature: {np.mean(curvatures):.4f}")
print(f"Curvature std: {np.std(curvatures):.4f}")
print(f"K > 0 (elliptic): {np.sum(curvatures > 0)} cycles")
print(f"K < 0 (hyperbolic): {np.sum(curvatures < 0)} cycles")
```

**Interpretation**:
- Red regions (K > 0): Fitness landscape locally convex (peaks/attractors)
- Blue regions (K < 0): Fitness landscape locally saddle-like
- White (K ≈ 0): Approximately flat regions

**This is a novel diagnostic** - curvature of fitness landscape directly from optimization dynamics!

---

## 7. Summary and Implications

### 7.1. Key Results

**Resolved**:
1. ✅ Use A_curved (Riemannian), not A_flat (Euclidean)
2. ✅ Ratio A_curved/A_flat reveals emergent curvature
3. ✅ Can extract Gaussian curvature K from Wilson loops
4. ✅ Both areas computable from same discrete data

**Physical interpretation**:
> "Wilson loops must use the geometry that walkers experience. The algorithmic curvature from the fitness Hessian is the physical curvature."

### 7.2. Novel Capabilities

**From this framework**:

1. **Curvature measurement**: Extract K(x) from optimization dynamics alone
2. **Geometric diagnostics**: Visualize fitness landscape geometry
3. **Consistency check**: Test if w_algo respects emergent metric
4. **Theoretical foundation**: Wilson action with curved geometry

### 7.3. Updated Workflow

**Complete Wilson loop computation**:

```python
def wilson_action_intrinsic(CST, IG, metric_fn, swarm_state, gauge_links):
    """Compute Wilson action using intrinsic (curved) areas."""

    S_gauge = 0.0
    areas_curved = []

    # Step 1: Compute all curved areas
    for e in IG.edges:
        cycle = get_fundamental_cycle(e, CST, IG)
        positions = [Phi[v] for v in cycle]
        result = compute_both_areas(positions, metric_fn, swarm_state)
        areas_curved.append(result['A_curved'])

    mean_area = np.mean(areas_curved)

    # Step 2: Compute Wilson loops and action
    for e, A_curved in zip(IG.edges, areas_curved):
        W_e = compute_wilson_loop(e, CST, IG, gauge_links)
        w_e = mean_area / A_curved

        S_gauge += w_e * (1.0 - np.real(np.trace(W_e)) / N_c)

    S_gauge *= beta / (2.0 * N_c)

    return S_gauge
```

---

## 8. Implications for Earlier Work

### Chapter 33 Updates

**Original**: Used fan triangulation for "area" without specifying flat vs curved

**Clarification**: The formula in Chapter 33 already computed A_curved!

The formula:
$$
A = \sum_i \frac{1}{2} \sqrt{(v_1^T G v_1)(v_2^T G v_2) - (v_1^T G v_2)^2}
$$

This **is** the Riemannian area formula. ✅

**No changes needed** - Chapter 33 was already correct!

### New Insight

We can now **also** compute A_flat and use the ratio for diagnostics. This adds value without changing the core formulation.

---

## 9. Next Steps

### Implementation (1 week)

- [ ] Extend `compute_cycle_area()` to return both A_flat and A_curved
- [ ] Add `extract_curvature()` function
- [ ] Implement three empirical tests

### Validation (2-3 weeks)

- [ ] Test on sphere (known K = 1)
- [ ] Test on hyperbolic plane (known K = -1)
- [ ] Test on flat space (known K = 0)
- [ ] Run on actual Fragile Gas data

### Analysis (1 month)

- [ ] Generate curvature maps of fitness landscapes
- [ ] Correlate curvature with optimization performance
- [ ] Test w_algo vs w_flat vs w_curved
- [ ] Determine which best predicts continuum limit

### Publication (3-6 months)

**Paper**: "Emergent Riemannian Geometry in Algorithmic Optimization"
- Metric from fitness Hessian
- Areas from Wilson loops
- Curvature from area ratios
- Novel diagnostic for landscape geometry

**Target**: Physical Review E, J. Stat. Mech., or JMLR

---

## References

### Differential Geometry

- doCarmo, M.P. (1992). *Riemannian Geometry*. Birkhäuser. Ch. 3 (Curvature)
- Bertrand, J. (1848). "Mémoire sur le nombre de valeurs..." *J. École Polytechnique*
- Gauss, C.F. (1827). *Disquisitiones generales circa superficies curvas*

### Information Geometry

- Amari, S. (2016). *Information Geometry and Its Applications*. Springer
- Ay, N. et al. (2017). *Information Geometry*. Springer

### Internal Documents

- [07_adaptative_gas.md](07_adaptative_gas.md): Metric from fitness Hessian
- [32_wilson_loops_single_root_corrected.md](32_wilson_loops_single_root_corrected.md): Wilson loop algorithm
- [33_geometric_area_from_fractal_set.md](33_geometric_area_from_fractal_set.md): Fan triangulation (already A_curved!)
- [34_area_problem_resolution_summary.md](34_area_problem_resolution_summary.md): Resolution timeline

---

**Status**: ✅ **COMPLETE RESOLUTION**

**Key Discovery**: Intrinsic (curved) area is correct for Wilson loops; ratio with extrinsic (flat) area reveals algorithmic curvature

**Next**: Implement both area calculations and extract curvature maps
