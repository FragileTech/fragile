# Geometric Algorithms and Computational Methods

**Status:** ✅ Complete
**Purpose:** Practical algorithms for computing geometric quantities on the Fractal Set
**Scope:** Implementation-focused companion to theoretical framework

---

## Introduction

The Fractal Set $\mathcal{F} = (\mathcal{E}, E_{\text{CST}} \cup E_{\text{IG}})$ is not an abstract combinatorial graph—it is a **discrete approximation of a Riemannian manifold** $(\mathcal{X}, G)$. Each episode $e \in \mathcal{E}$ has:

- **Position**: $\Phi(e) \in \mathcal{X} \subseteq \mathbb{R}^d$ (death position in state space)
- **Metric**: $G(x)$ (fitness-induced Riemannian metric from Chapter 7)
- **Trajectory**: $\gamma_e: [t^b_e, t^d_e) \to \mathcal{X}$ (path during episode lifetime)

This geometric structure enables **computable** geometric quantities that are impossible on generic graphs:
- Riemannian areas (via fan triangulation)
- Path lengths (via metric tensor)
- Curvature (from area ratios)
- Parallel transport (along CST+IG paths)
- Wilson loops (gauge-invariant observables)

This document provides **practical algorithms** for computing these quantities, including pseudocode, complexity analysis, and implementation notes.

**Key Sources:**
- Fan triangulation: [13_D_fractal_set_emergent_qft_comprehensive.md § 8.2](../13_fractal_set_old/13_D_fractal_set_emergent_qft_comprehensive.md)
- Metric tensor: [fitness_algebra.py](../../src/fragile/fitness_algebra.py)
- Continuum limit: [13_B_fractal_set_continuum_limit.md](../13_fractal_set_old/13_B_fractal_set_continuum_limit.md)

---

## 1. Fan Triangulation for Riemannian Area

### 1.1. Algorithm Statement

:::{prf:algorithm} Fan Triangulation for Cycle Area
:label: alg-fan-triangulation

**Input:**
- Cycle $C = \{e_0, e_1, \ldots, e_{n-1}\}$ with positions $\Phi(e_i) \in \mathbb{R}^d$
- Metric function $G: \mathcal{X} \times \mathcal{S} \to \mathbb{R}^{d \times d}$ (positive definite)
- Current swarm state $S$ (for metric evaluation)

**Output:** Riemannian area $A(C)$

**Procedure:**

1. **Compute centroid:**

$$
x_c = \frac{1}{n} \sum_{i=0}^{n-1} \Phi(e_i)
$$

2. **Evaluate metric at centroid:**

$$
G_c = G(x_c, S)
$$

3. **For each triangle** $T_i = (x_c, \Phi(e_i), \Phi(e_{i+1}))$:

   a. Compute edge vectors:

$$
v_1 = \Phi(e_i) - x_c, \quad v_2 = \Phi(e_{i+1}) - x_c
$$

   b. Compute Riemannian inner products:

$$
\begin{aligned}
\langle v_1, v_1 \rangle_G &= v_1^T G_c v_1 \\
\langle v_2, v_2 \rangle_G &= v_2^T G_c v_2 \\
\langle v_1, v_2 \rangle_G &= v_1^T G_c v_2
\end{aligned}
$$

   c. Compute triangle area (Riemannian Gram determinant):

$$
A_i = \frac{1}{2} \sqrt{\langle v_1, v_1 \rangle_G \cdot \langle v_2, v_2 \rangle_G - \langle v_1, v_2 \rangle_G^2}
$$

4. **Sum all triangles:**

$$
A(C) = \sum_{i=0}^{n-1} A_i
$$

**Complexity:** $O(n \cdot d^2)$ for $n$ vertices in dimension $d$
:::

**From:** [13_D § 8.2 - Theorem: Riemannian Area via Fan Triangulation](../13_fractal_set_old/13_D_fractal_set_emergent_qft_comprehensive.md#82-fan-triangulation-formula)

### 1.2. Proof of Base Independence

:::{prf:theorem} Fan Triangulation is Base-Independent
:label: thm-fan-base-independence

The choice of base vertex (centroid vs. arbitrary vertex) does **not** affect the total area, up to $O(d^2)$ curvature corrections.

**Statement:** For cycle $C$ embedded in Riemannian manifold $(\mathcal{X}, G)$:

$$
A_{\text{centroid}}(C) \approx A_{\text{vertex}}(C) \quad \text{(error } O(R \cdot \text{diam}(C)^3) \text{)}
$$

where $R = \|\text{Riemann}(x)\|$ is the curvature scale.

**Proof Sketch:**

1. **Flat space:** Exact equality (shoelace formula independent of base)
2. **Curved space:** Metric varies as $G(x + \delta x) = G(x) + O(\|\nabla G\| \cdot \|\delta x\|)$
3. **Centroid minimizes** $\sum_i \|x - \Phi(e_i)\|^2$ → smallest metric variation
4. **Error bound:** Using Taylor expansion of $G$ around base point:

$$
|A_{\text{base}_1}(C) - A_{\text{base}_2}(C)| \leq C_d \cdot R \cdot \text{diam}(C)^3
$$

**Conclusion:** For small cycles (typical in IG), choice of base vertex negligible.
:::

**Practical Implication:** Use centroid for symmetry, but any interior point works.

### 1.3. Implementation

**Python Implementation:**

```python
import numpy as np
from typing import Callable

def compute_cycle_area_riemannian(
    positions: np.ndarray,
    metric_fn: Callable[[np.ndarray, object], np.ndarray],
    swarm_state: object
) -> float:
    """
    Compute Riemannian area of cycle using fan triangulation.

    Parameters
    ----------
    positions : np.ndarray, shape (n, d)
        Episode positions Φ(e_i) for cycle vertices
    metric_fn : callable
        Function G(x, S) → (d, d) positive definite metric tensor
    swarm_state : object
        Current swarm state for metric evaluation

    Returns
    -------
    area : float
        Total Riemannian area of cycle

    Notes
    -----
    - Uses centroid as base vertex for triangulation
    - Handles numerical stability via max(discriminant, 0)
    - Complexity: O(n * d^2)
    """
    n, d = positions.shape

    # Step 1: Compute centroid
    x_c = np.mean(positions, axis=0)

    # Step 2: Evaluate metric at centroid
    G_c = metric_fn(x_c, swarm_state)

    # Step 3-4: Sum triangle areas
    area = 0.0
    for i in range(n):
        # Edge vectors from centroid
        v1 = positions[i] - x_c
        v2 = positions[(i + 1) % n] - x_c

        # Riemannian inner products
        v1_G_v1 = v1 @ G_c @ v1
        v2_G_v2 = v2 @ G_c @ v2
        v1_G_v2 = v1 @ G_c @ v2

        # Gram determinant (handle numerical error)
        discriminant = v1_G_v1 * v2_G_v2 - v1_G_v2**2
        area += 0.5 * np.sqrt(max(discriminant, 0.0))

    return area


def compute_cycle_area_flat(positions: np.ndarray) -> float:
    """
    Compute flat (Euclidean) area for comparison.

    Parameters
    ----------
    positions : np.ndarray, shape (n, d)
        Cycle vertices

    Returns
    -------
    area : float
        Euclidean area (shoelace formula for d=2, fan triangulation for d>2)
    """
    n, d = positions.shape

    if d == 2:
        # Shoelace formula (exact)
        x, y = positions[:, 0], positions[:, 1]
        area = 0.5 * np.abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
        return area

    else:
        # Fan triangulation with Euclidean norm
        x_c = np.mean(positions, axis=0)
        area = 0.0
        for i in range(n):
            v1 = positions[i] - x_c
            v2 = positions[(i + 1) % n] - x_c
            # Cross product magnitude (only valid for d=3)
            if d == 3:
                cross = np.cross(v1, v2)
                area += 0.5 * np.linalg.norm(cross)
            else:
                # General d: use Gram determinant with identity metric
                area += 0.5 * np.sqrt(max(np.dot(v1, v1) * np.dot(v2, v2) - np.dot(v1, v2)**2, 0.0))
        return area
```

**Usage Example:**

```python
# From fragile.adaptive_gas import AdaptiveGas
# from fragile.gas_parameters import AdaptiveGasParams

# Setup
gas = AdaptiveGas(params)
swarm_state = gas.step()  # Run optimization

# Extract cycle from IG edge
ig_edge = (e_i, e_j)  # IG edge between episodes
cycle_vertices = find_fundamental_cycle(ig_edge, CST, IG)  # See § 2.2
positions = np.array([Phi(e) for e in cycle_vertices])

# Compute areas
A_curved = compute_cycle_area_riemannian(positions, gas.get_metric, swarm_state)
A_flat = compute_cycle_area_flat(positions)

# Area ratio reveals curvature (see § 3.3)
ratio = A_curved / A_flat
det_G = np.linalg.det(gas.get_metric(np.mean(positions, axis=0), swarm_state))
print(f"Area ratio: {ratio:.4f}, √det(G): {np.sqrt(det_G):.4f}")
```

---

## 2. IG Fundamental Cycles

### 2.1. Cycle Basis Construction

:::{prf:theorem} IG Edges Close Fundamental Cycles
:label: thm-ig-cycles

**Assumption:** CST is a rooted spanning tree (single common ancestor)

**Claim:** For IG graph with $k$ edges $E_{\text{IG}} = \{e_1, \ldots, e_k\}$:

1. Each IG edge $e_i = (e_a \sim e_b)$ closes exactly one fundamental cycle $C(e_i)$
2. The cycles $\{C(e_1), \ldots, C(e_k)\}$ form a complete basis for the cycle space

**Construction:** For IG edge $e_i = (e_a \sim e_b)$:

$$
C(e_i) := e_i \cup P_{\text{CST}}(e_a, e_b)
$$

where $P_{\text{CST}}(e_a, e_b)$ is the unique path from $e_a$ to $e_b$ in the CST tree.

**Proof:**

*Part 1 (Unique path):* CST is tree → unique path between any two vertices

*Part 2 (Closed cycle):* $e_i$ connects $e_a \to e_b$ (IG), $P_{\text{CST}}$ connects $e_b \to e_a$ (CST) → closed loop

*Part 3 (Linear independence):* Each $C(e_i)$ contains IG edge $e_i$, no other cycle contains $e_i$ → independent

*Part 4 (Completeness):* Cycle space dimension = $|E_{\text{total}}| - |V| + 1 = (|E_{\text{CST}}| + k) - |V| + 1$. Since $|E_{\text{CST}}| = |V| - 1$ (tree property), dimension = $k$. We have $k$ independent cycles → complete basis. ∎
:::

**From:** [13_D § 7.1 - Theorem: IG Edges Close Fundamental Cycles](../13_fractal_set_old/13_D_fractal_set_emergent_qft_comprehensive.md#71-fundamental-cycles-from-ig-edges)

### 2.2. Algorithm for Finding Fundamental Cycles

:::{prf:algorithm} Find Fundamental Cycle from IG Edge
:label: alg-fundamental-cycle

**Input:**
- IG edge $e = (e_a \sim e_b)$
- CST as adjacency list `cst[episode] = parent_episode`
- Root episode $e_{\text{root}}$

**Output:** Ordered cycle vertices $C(e) = [e_a, \ldots, e_{\text{LCA}}, \ldots, e_b]$

**Procedure:**

1. **Find Lowest Common Ancestor (LCA):**

   ```python
   def find_lca(e_a, e_b, cst):
       # Trace paths to root
       path_a = trace_to_root(e_a, cst)
       path_b = trace_to_root(e_b, cst)

       # Find first common vertex
       ancestors_a = set(path_a)
       for vertex in path_b:
           if vertex in ancestors_a:
               return vertex
   ```

2. **Build upward path** $P_{\text{up}}(e_a, e_{\text{LCA}})$:

   ```python
   def path_to_ancestor(start, ancestor, cst):
       path = [start]
       current = start
       while current != ancestor:
           current = cst[current]  # Parent
           path.append(current)
       return path
   ```

3. **Build downward path** $P_{\text{down}}(e_{\text{LCA}}, e_b)$:

   Same as step 2, then reverse

4. **Concatenate:**

   ```python
   cycle = path_to_ancestor(e_a, lca, cst)[:-1]  # Exclude LCA
   cycle.append(lca)
   cycle.extend(reversed(path_to_ancestor(e_b, lca, cst)[:-1]))
   return cycle
   ```

**Complexity:**
- Without preprocessing: $O(h)$ where $h$ is tree height
- With LCA preprocessing (sparse table): $O(\log N)$ per query after $O(N \log N)$ setup
:::

**Complete Implementation:**

```python
from typing import Dict, List, Tuple, Set
from collections import defaultdict

class FractalSetCycles:
    """Compute fundamental cycles on Fractal Set (CST + IG)."""

    def __init__(self, cst_edges: List[Tuple], ig_edges: List[Tuple]):
        """
        Parameters
        ----------
        cst_edges : list of (parent, child) tuples
            Directed edges in Causal Spacetime Tree
        ig_edges : list of (e_i, e_j) tuples
            Undirected edges in Information Graph
        """
        self.cst = self._build_cst_adjacency(cst_edges)
        self.ig_edges = ig_edges
        self.root = self._find_root()

        # Preprocessing for fast LCA queries
        self._preprocess_lca()

    def _build_cst_adjacency(self, edges: List[Tuple]) -> Dict:
        """Build parent pointers for CST."""
        cst = {}
        for parent, child in edges:
            cst[child] = parent
        return cst

    def _find_root(self) -> int:
        """Find root episode (no parent)."""
        all_episodes = set(self.cst.keys()) | set(self.cst.values())
        children = set(self.cst.keys())
        roots = all_episodes - children
        assert len(roots) == 1, "CST must have single root"
        return roots.pop()

    def _trace_to_root(self, episode: int) -> List[int]:
        """Trace path from episode to root."""
        path = [episode]
        while episode in self.cst:
            episode = self.cst[episode]
            path.append(episode)
        return path

    def _preprocess_lca(self):
        """Precompute LCA data structures (simplified version)."""
        # Store depth and parent for each node
        self.depth = {}
        self.ancestors = defaultdict(set)

        # BFS from root
        queue = [(self.root, 0)]
        self.depth[self.root] = 0

        while queue:
            node, d = queue.pop(0)
            # Add all ancestors
            current = node
            while current in self.cst:
                current = self.cst[current]
                self.ancestors[node].add(current)

            # Add children
            for child, parent in self.cst.items():
                if parent == node and child not in self.depth:
                    self.depth[child] = d + 1
                    queue.append((child, d + 1))

    def find_lca(self, e_a: int, e_b: int) -> int:
        """
        Find Lowest Common Ancestor of two episodes.

        Parameters
        ----------
        e_a, e_b : int
            Episode identifiers

        Returns
        -------
        lca : int
            Lowest common ancestor episode
        """
        # Trace both paths to root
        path_a = self._trace_to_root(e_a)
        path_b = self._trace_to_root(e_b)

        # Find first common ancestor
        ancestors_a = set(path_a)
        for vertex in path_b:
            if vertex in ancestors_a:
                return vertex

        # Should never reach here if CST is valid tree
        raise ValueError("No common ancestor found (invalid CST)")

    def path_to_ancestor(self, start: int, ancestor: int) -> List[int]:
        """
        Find path from start to ancestor in CST.

        Parameters
        ----------
        start : int
            Starting episode
        ancestor : int
            Target ancestor episode

        Returns
        -------
        path : list of int
            Ordered path [start, ..., ancestor]
        """
        path = [start]
        current = start

        while current != ancestor:
            if current not in self.cst:
                raise ValueError(f"Episode {ancestor} is not ancestor of {start}")
            current = self.cst[current]
            path.append(current)

        return path

    def fundamental_cycle(self, ig_edge: Tuple[int, int]) -> List[int]:
        """
        Find fundamental cycle closed by IG edge.

        Parameters
        ----------
        ig_edge : (e_a, e_b)
            IG edge between episodes

        Returns
        -------
        cycle : list of int
            Ordered cycle vertices [e_a, ..., lca, ..., e_b]
        """
        e_a, e_b = ig_edge

        # Find LCA
        lca = self.find_lca(e_a, e_b)

        # Build paths
        path_up = self.path_to_ancestor(e_a, lca)
        path_down = self.path_to_ancestor(e_b, lca)

        # Concatenate: e_a → lca → e_b (IG edge completes cycle)
        # Exclude duplicate LCA from path_down
        cycle = path_up + list(reversed(path_down[:-1]))

        return cycle

    def all_fundamental_cycles(self) -> Dict[Tuple, List[int]]:
        """
        Compute all fundamental cycles (one per IG edge).

        Returns
        -------
        cycles : dict
            Mapping {ig_edge: cycle_vertices}
        """
        cycles = {}
        for ig_edge in self.ig_edges:
            cycles[ig_edge] = self.fundamental_cycle(ig_edge)
        return cycles
```

**Usage:**

```python
# Build Fractal Set
cst_edges = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5)]  # Parent → child
ig_edges = [(3, 5), (4, 5)]  # Companions in cloning

# Find cycles
fractal = FractalSetCycles(cst_edges, ig_edges)
cycles = fractal.all_fundamental_cycles()

for ig_edge, cycle in cycles.items():
    print(f"IG edge {ig_edge} closes cycle: {cycle}")
    # Output: IG edge (3, 5) closes cycle: [3, 1, 0, 2, 5]
```

---

## 3. Metric Tensor Estimation

### 3.1. Local Covariance Matrix Computation

:::{prf:definition} Empirical Metric from IG Edges
:label: def-empirical-metric

For episode $e_i$ at position $x_i = \Phi(e_i)$, the **local covariance matrix** from IG neighbors is:

$$
\Sigma_i = \frac{1}{|\mathcal{N}_{\text{IG}}(e_i)|} \sum_{e_j \in \mathcal{N}_{\text{IG}}(e_i)} w_{ij} \cdot \Delta x_{ij} \Delta x_{ij}^T
$$

where:
- $\mathcal{N}_{\text{IG}}(e_i) = \{e_j : (e_i \sim e_j) \in E_{\text{IG}}\}$ (IG neighbors)
- $\Delta x_{ij} = \Phi(e_j) - \Phi(e_i)$ (displacement vector)
- $w_{ij} = 1 / ((\Delta t_{ij})^2 + \|\Delta x_{ij}\|^2)$ (algorithmic weight)

**Relationship to metric:**

$$
\Sigma_i^{-1} \approx G(x_i, S) + O(1/N)
$$

The **inverse covariance** approximates the metric tensor in the continuum limit.
:::

**From:** [13_B § 5.2 - Local Covariance Matrix](../13_fractal_set_old/13_B_fractal_set_continuum_limit.md) and [fitness_algebra.py](../../src/fragile/fitness_algebra.py)

**Algorithm:**

```python
def compute_local_covariance(
    episode: int,
    ig_neighbors: List[int],
    positions: Dict[int, np.ndarray],
    death_times: Dict[int, float]
) -> np.ndarray:
    """
    Compute local covariance matrix from IG neighbors.

    Parameters
    ----------
    episode : int
        Episode identifier
    ig_neighbors : list of int
        IG neighbor episodes
    positions : dict
        Mapping {episode: position_vector}
    death_times : dict
        Mapping {episode: death_time}

    Returns
    -------
    Sigma : np.ndarray, shape (d, d)
        Local covariance matrix
    """
    x_i = positions[episode]
    t_i = death_times[episode]
    d = len(x_i)

    # Initialize
    Sigma = np.zeros((d, d))
    total_weight = 0.0

    for e_j in ig_neighbors:
        x_j = positions[e_j]
        t_j = death_times[e_j]

        # Displacement
        delta_x = x_j - x_i
        delta_t = t_j - t_i

        # Algorithmic weight
        w_ij = 1.0 / (delta_t**2 + np.dot(delta_x, delta_x))

        # Outer product
        Sigma += w_ij * np.outer(delta_x, delta_x)
        total_weight += w_ij

    # Normalize
    if total_weight > 0:
        Sigma /= total_weight

    return Sigma


def estimate_metric_from_covariance(Sigma: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """
    Estimate metric tensor as inverse covariance (regularized).

    Parameters
    ----------
    Sigma : np.ndarray, shape (d, d)
        Local covariance matrix
    epsilon : float
        Regularization for numerical stability

    Returns
    -------
    G : np.ndarray, shape (d, d)
        Estimated metric tensor
    """
    d = Sigma.shape[0]
    Sigma_reg = Sigma + epsilon * np.eye(d)
    G = np.linalg.inv(Sigma_reg)
    return G
```

### 3.2. Regularized Hessian Approximation

The **fitness-induced metric** from Chapter 7 is:

$$
G(x, S) = H_\Phi(x, S) + \epsilon_\Sigma I
$$

where $H_\Phi = \nabla^2_x V_{\text{fit}}$ is the Hessian of the fitness potential.

**Symbolic Computation** (using `fitness_algebra.py`):

```python
from fragile.fitness_algebra import FitnessPotential, EmergentMetric

# Create fitness potential
fitness = FitnessPotential(dim=3, num_walkers=5)
V_fit = fitness.fitness_potential_sigmoid()

# Compute Hessian and metric
metric = EmergentMetric(fitness)
H = metric.hessian_V_fit(V_fit)  # Symbolic Hessian
g = metric.metric_tensor(H)       # g = H + ε_Σ I

# Export to numerical function
import sympy as sp
x_vars = fitness.x  # Position variables [x_1, x_2, x_3]
params = fitness.params  # Algorithmic parameters

# Lambdify for fast evaluation
g_numerical = sp.lambdify(
    [x_vars, *params.values()],
    g,
    modules='numpy'
)

# Evaluate at specific point
x_eval = np.array([1.0, 2.0, 3.0])
param_values = {
    'epsilon_Sigma': 0.01,
    'A': 10.0,
    'rho': 1.0,
    # ... other parameters
}
G_eval = g_numerical(x_eval, *param_values.values())
```

**Finite Difference Approximation** (for empirical data):

```python
def estimate_hessian_finite_difference(
    fitness_fn: Callable[[np.ndarray], float],
    x: np.ndarray,
    h: float = 1e-5
) -> np.ndarray:
    """
    Estimate Hessian using central finite differences.

    H_ij ≈ [f(x + h e_i + h e_j) - f(x + h e_i) - f(x + h e_j) + f(x)] / h²

    Parameters
    ----------
    fitness_fn : callable
        Fitness function V_fit(x) → scalar
    x : np.ndarray, shape (d,)
        Evaluation point
    h : float
        Step size

    Returns
    -------
    H : np.ndarray, shape (d, d)
        Approximate Hessian
    """
    d = len(x)
    H = np.zeros((d, d))
    f_x = fitness_fn(x)

    for i in range(d):
        for j in range(i, d):  # Symmetric, compute upper triangle
            e_i = np.zeros(d)
            e_i[i] = h
            e_j = np.zeros(d)
            e_j[j] = h

            if i == j:
                # Diagonal: second derivative
                f_plus = fitness_fn(x + e_i)
                f_minus = fitness_fn(x - e_i)
                H[i, i] = (f_plus - 2*f_x + f_minus) / (h**2)
            else:
                # Off-diagonal: mixed derivative
                f_pp = fitness_fn(x + e_i + e_j)
                f_pm = fitness_fn(x + e_i)
                f_mp = fitness_fn(x + e_j)
                H[i, j] = (f_pp - f_pm - f_mp + f_x) / (h**2)
                H[j, i] = H[i, j]  # Symmetry

    return H
```

---

## 4. Parallel Transport and Holonomy Computation

### 4.1. Path Tracing on CST+IG

:::{prf:definition} Parallel Transport on Fractal Set
:label: def-parallel-transport-fractal

For gauge field $U_e \in SU(N_c)$ on each edge $e$:

**CST edge** $e_p \to e_c$ (parent → child):
- Forward: $U(e_p, e_c)$
- Backward: $U(e_c, e_p) = U(e_p, e_c)^\dagger$

**IG edge** $e_i \sim e_j$ (undirected):
- Either direction: $U(e_i, e_j)$ or $U(e_j, e_i) = U(e_i, e_j)^\dagger$

**Path-ordered product** along path $P = \{e_0 \to e_1 \to \cdots \to e_n\}$:

$$
U(P) = U(e_{n-1}, e_n) \times U(e_{n-2}, e_{n-1}) \times \cdots \times U(e_0, e_1)
$$

(Rightmost operator acts first, consistent with quantum mechanics)
:::

**Algorithm:**

```python
import numpy as np
from typing import List, Dict, Tuple

def path_ordered_product(
    path: List[int],
    gauge_links: Dict[Tuple[int, int], np.ndarray],
    edge_directions: Dict[Tuple[int, int], str]
) -> np.ndarray:
    """
    Compute path-ordered product U(P) along path.

    Parameters
    ----------
    path : list of int
        Ordered episode indices [e_0, e_1, ..., e_n]
    gauge_links : dict
        Mapping {(e_i, e_j): U_matrix} for directed edges
    edge_directions : dict
        Mapping {(e_i, e_j): 'forward'|'backward'} indicating traversal direction

    Returns
    -------
    U_path : np.ndarray, shape (N_c, N_c)
        Path-ordered parallel transport operator
    """
    N_c = list(gauge_links.values())[0].shape[0]
    U_path = np.eye(N_c, dtype=complex)

    for i in range(len(path) - 1):
        e_start, e_end = path[i], path[i+1]

        # Determine edge direction
        if (e_start, e_end) in gauge_links:
            U_edge = gauge_links[(e_start, e_end)]
            direction = edge_directions.get((e_start, e_end), 'forward')
        elif (e_end, e_start) in gauge_links:
            U_edge = gauge_links[(e_end, e_start)]
            direction = 'backward'  # Reversed edge
        else:
            raise ValueError(f"No gauge link between {e_start} and {e_end}")

        # Apply dagger if traversing backward
        if direction == 'backward':
            U_edge = U_edge.conj().T

        # Right-multiply (path ordering)
        U_path = U_path @ U_edge

    return U_path
```

### 4.2. Wilson Loop Evaluation

:::{prf:algorithm} Wilson Loop from IG Edge
:label: alg-wilson-loop

**Input:**
- IG edge $e = (e_i \sim e_j)$
- Fundamental cycle $C(e)$ (from Algorithm {prf:ref}`alg-fundamental-cycle`)
- Gauge links $\{U_{ab}\}$ on all CST and IG edges

**Output:** Wilson loop observable $W_e = \text{Tr}(U_C) \in \mathbb{R}$

**Procedure:**

1. **Split cycle into segments:**
   - IG segment: $e_i \to e_j$ (via IG edge)
   - CST segment: $e_j \to e_i$ (via CST path)

2. **Compute IG transport:**

$$
U_{\text{IG}} = U(e_i, e_j)
$$

3. **Compute CST transport:**

$$
U_{\text{CST}} = U(P_{\text{CST}}(e_j, e_i))
$$

   Using path-ordered product from § 4.1

4. **Multiply to close loop:**

$$
U_C = U_{\text{IG}} \times U_{\text{CST}}
$$

5. **Take trace (gauge-invariant observable):**

$$
W_e = \text{Re}\left[\frac{1}{N_c} \text{Tr}(U_C)\right]
$$

**Complexity:** $O(|C| \cdot N_c^3)$ where $|C|$ is cycle length, $N_c$ is gauge group dimension
:::

**Implementation:**

```python
def compute_wilson_loop(
    ig_edge: Tuple[int, int],
    cycle: List[int],
    gauge_links: Dict[Tuple[int, int], np.ndarray],
    edge_directions: Dict[Tuple[int, int], str]
) -> float:
    """
    Compute Wilson loop for IG edge.

    Parameters
    ----------
    ig_edge : (e_i, e_j)
        IG edge closing the cycle
    cycle : list of int
        Fundamental cycle vertices from Algorithm 2.2
    gauge_links : dict
        Gauge links U on all edges
    edge_directions : dict
        Edge direction indicators

    Returns
    -------
    W : float
        Real part of normalized Wilson loop Tr(U_C) / N_c
    """
    e_i, e_j = ig_edge
    N_c = list(gauge_links.values())[0].shape[0]

    # Compute path-ordered product around cycle
    # Cycle should already include IG edge
    U_cycle = path_ordered_product(cycle, gauge_links, edge_directions)

    # Take trace and normalize
    trace = np.trace(U_cycle)
    W = np.real(trace) / N_c

    return W


def wilson_action(
    cycles: Dict[Tuple, List[int]],
    areas: Dict[Tuple, float],
    gauge_links: Dict[Tuple[int, int], np.ndarray],
    edge_directions: Dict[Tuple[int, int], str],
    beta: float = 1.0
) -> float:
    """
    Compute total Wilson action for all IG cycles.

    S_gauge = (β / 2N_c) Σ_e (w_e) (1 - Re Tr[W_e])

    where w_e = <A> / A(e) (inverse area weighting)

    Parameters
    ----------
    cycles : dict
        Mapping {ig_edge: cycle_vertices}
    areas : dict
        Mapping {ig_edge: area}
    gauge_links : dict
        Gauge links on all edges
    edge_directions : dict
        Edge directions
    beta : float
        Coupling parameter β = 2N_c / g²

    Returns
    -------
    S : float
        Total Wilson action
    """
    N_c = list(gauge_links.values())[0].shape[0]

    # Compute mean area for normalization
    mean_area = np.mean(list(areas.values()))

    action = 0.0
    for ig_edge, cycle in cycles.items():
        # Wilson loop
        W = compute_wilson_loop(ig_edge, cycle, gauge_links, edge_directions)

        # Weight (inverse area)
        w = mean_area / areas[ig_edge]

        # Add contribution
        action += w * (1.0 - W)

    action *= beta / (2 * N_c)

    return action
```

---

## 5. Implementation Notes

### 5.1. Numerical Stability

**Issue 1: Metric Tensor Conditioning**

The metric $G(x, S) = H_\Phi(x, S) + \epsilon_\Sigma I$ can become ill-conditioned if:
- $\epsilon_\Sigma$ too small → $G$ nearly singular
- $H_\Phi$ has large eigenvalue spread → anisotropic diffusion

**Solutions:**
1. **Adaptive regularization:**

```python
def regularize_metric(H: np.ndarray, epsilon_min: float = 1e-6) -> np.ndarray:
    """Add adaptive regularization based on condition number."""
    eigvals = np.linalg.eigvalsh(H)
    cond = eigvals.max() / (eigvals.min() + 1e-12)

    if cond > 1e6:  # Ill-conditioned
        epsilon = epsilon_min * np.sqrt(cond)
    else:
        epsilon = epsilon_min

    G = H + epsilon * np.eye(len(H))
    return G
```

2. **Eigenvalue clipping:**

```python
def clip_metric_eigenvalues(G: np.ndarray, lambda_min: float = 1e-3, lambda_max: float = 1e3) -> np.ndarray:
    """Clip eigenvalues to safe range."""
    eigvals, eigvecs = np.linalg.eigh(G)
    eigvals_clipped = np.clip(eigvals, lambda_min, lambda_max)
    G_clipped = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
    return G_clipped
```

**Issue 2: Area Computation Discriminant**

The Gram determinant can be slightly negative due to floating point error:

$$
D = \langle v_1, v_1 \rangle_G \cdot \langle v_2, v_2 \rangle_G - \langle v_1, v_2 \rangle_G^2
$$

**Solution:** Use `max(D, 0)` before taking square root, as shown in implementation.

**Issue 3: Wilson Loop Phase**

For $SU(N_c)$ gauge group, $\text{Tr}(U_C)$ is complex but should have $|\text{Tr}(U_C)| \leq N_c$.

**Sanity check:**

```python
def validate_wilson_loop(U_cycle: np.ndarray) -> None:
    """Check Wilson loop satisfies SU(N_c) constraints."""
    N_c = U_cycle.shape[0]

    # Check unitarity
    assert np.allclose(U_cycle @ U_cycle.conj().T, np.eye(N_c)), "U not unitary"

    # Check determinant = 1
    det = np.linalg.det(U_cycle)
    assert np.abs(det - 1.0) < 1e-6, f"det(U) = {det}, not 1"

    # Check trace magnitude
    trace = np.trace(U_cycle)
    assert np.abs(trace) <= N_c + 1e-6, f"|Tr(U)| = {np.abs(trace)} > {N_c}"
```

### 5.2. Complexity Analysis

**Summary Table:**

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Fan triangulation | $O(n d^2)$ | $n$ = cycle vertices, $d$ = dimension |
| LCA query (no preprocessing) | $O(h)$ | $h$ = tree height |
| LCA query (with preprocessing) | $O(\log N)$ | Amortized after $O(N \log N)$ setup |
| Fundamental cycle extraction | $O(h)$ or $O(\log N)$ | Depends on LCA method |
| Local covariance estimation | $O(k d^2)$ | $k$ = IG degree |
| Hessian finite differences | $O(d^2 \cdot T_f)$ | $T_f$ = fitness evaluation time |
| Path-ordered product | $O(\ell N_c^3)$ | $\ell$ = path length, $N_c$ = gauge group |
| Wilson loop | $O(\ell N_c^3)$ | Dominated by parallel transport |
| Total Wilson action | $O(K \bar{\ell} N_c^3)$ | $K$ = # IG edges, $\bar{\ell}$ = avg cycle length |

**Typical Parameters:**
- $d = 2$ or $3$ (low-dimensional problems)
- $n \sim 5$-$20$ (small cycles)
- $N = 100$-$1000$ (walkers)
- $K = O(N)$ (sparse IG)
- $\bar{\ell} \sim 10$-$50$ (moderate tree depth)
- $N_c = 2$ or $3$ (SU(2) or SU(3))

**Bottlenecks:**
1. **Hessian computation** (if using symbolic differentiation)
2. **Wilson action** (if $N_c$ large or many IG edges)

**Optimizations:**
- Cache metric tensors at frequently queried positions
- Parallelize over IG edges (independent Wilson loops)
- Use sparse matrix storage for CST (parent pointers only)

### 5.3. Example Code

**Complete Workflow:**

```python
import numpy as np
from fragile.adaptive_gas import AdaptiveGas
from fragile.gas_parameters import AdaptiveGasParams

# ============================================================================
# Step 1: Run Adaptive Gas and Extract Fractal Set
# ============================================================================

params = AdaptiveGasParams(
    N=200,
    d=3,
    epsilon_Sigma=0.01,
    # ... other parameters
)

gas = AdaptiveGas(params)

# Run for several steps
for _ in range(100):
    swarm_state = gas.step()

# Extract episode data
episodes = gas.get_episodes()  # List of episode objects
cst_edges = gas.get_cst_edges()  # Parent-child pairs
ig_edges = gas.get_ig_edges()  # Companion pairs

# Position mapping
positions = {e.id: e.death_position for e in episodes}
death_times = {e.id: e.death_time for e in episodes}

# ============================================================================
# Step 2: Find Fundamental Cycles
# ============================================================================

fractal = FractalSetCycles(cst_edges, ig_edges)
cycles = fractal.all_fundamental_cycles()

print(f"Found {len(cycles)} fundamental cycles")

# ============================================================================
# Step 3: Compute Geometric Areas
# ============================================================================

def get_metric(x, S):
    """Wrapper for metric function."""
    return gas.compute_metric_tensor(x, S)

areas_curved = {}
areas_flat = {}

for ig_edge, cycle in cycles.items():
    # Get positions
    cycle_positions = np.array([positions[e] for e in cycle])

    # Compute areas
    A_curved = compute_cycle_area_riemannian(cycle_positions, get_metric, swarm_state)
    A_flat = compute_cycle_area_flat(cycle_positions)

    areas_curved[ig_edge] = A_curved
    areas_flat[ig_edge] = A_flat

# ============================================================================
# Step 4: Analyze Curvature
# ============================================================================

for ig_edge in ig_edges:
    A_c = areas_curved[ig_edge]
    A_f = areas_flat[ig_edge]

    # Centroid
    cycle = cycles[ig_edge]
    centroid = np.mean([positions[e] for e in cycle], axis=0)

    # Metric determinant
    G = get_metric(centroid, swarm_state)
    det_G = np.linalg.det(G)

    # Area ratio
    ratio = A_c / (A_f * np.sqrt(det_G))

    # Extract curvature (Chapter 35 formula)
    K = 24.0 / (A_f**2) * (ratio - 1.0)

    print(f"IG edge {ig_edge}: A_curved={A_c:.4f}, A_flat={A_f:.4f}, K={K:.4f}")

# ============================================================================
# Step 5: Compute Wilson Loops (if gauge links available)
# ============================================================================

# Initialize random gauge links (for demonstration)
# In real implementation, these would evolve with dynamics
gauge_links = {}
edge_directions = {}
N_c = 2  # SU(2)

for parent, child in cst_edges:
    # Random SU(2) matrix
    U = random_su2()
    gauge_links[(parent, child)] = U
    edge_directions[(parent, child)] = 'forward'

for e_i, e_j in ig_edges:
    U = random_su2()
    gauge_links[(e_i, e_j)] = U
    edge_directions[(e_i, e_j)] = 'forward'

# Compute Wilson action
beta = 6.0  # Coupling
S_wilson = wilson_action(cycles, areas_curved, gauge_links, edge_directions, beta)

print(f"\nWilson action: S = {S_wilson:.6f}")

# ============================================================================
# Step 6: Visualization
# ============================================================================

import holoviews as hv
hv.extension('bokeh')

# Scatter plot: centroid position colored by curvature
centroids = []
curvatures = []

for ig_edge, cycle in cycles.items():
    centroid = np.mean([positions[e] for e in cycle], axis=0)
    centroids.append(centroid)

    A_c = areas_curved[ig_edge]
    A_f = areas_flat[ig_edge]
    G = get_metric(centroid, swarm_state)
    det_G = np.linalg.det(G)
    ratio = A_c / (A_f * np.sqrt(det_G))
    K = 24.0 / (A_f**2) * (ratio - 1.0)
    curvatures.append(K)

centroids = np.array(centroids)
curvatures = np.array(curvatures)

# 2D projection (if d > 2)
if params.d == 2:
    plot = hv.Scatter((centroids[:, 0], centroids[:, 1], curvatures),
                      vdims='Curvature')
else:  # d = 3
    plot = hv.Scatter((centroids[:, 0], centroids[:, 1], curvatures),
                      vdims='Curvature').opts(color='Curvature', cmap='RdBu', size=8)

plot.opts(title='Gaussian Curvature from Wilson Loops',
          xlabel='x_1', ylabel='x_2', width=600, height=400)

hv.save(plot, 'curvature_map.html')
print("\nVisualization saved to curvature_map.html")


def random_su2():
    """Generate random SU(2) matrix using Haar measure."""
    # Parameterization: U = exp(i θ σ·n̂)
    theta = np.random.uniform(0, 2*np.pi)
    n = np.random.randn(3)
    n /= np.linalg.norm(n)

    # Pauli matrices
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

    sigma_n = n[0]*sigma_x + n[1]*sigma_y + n[2]*sigma_z
    U = np.cos(theta)*np.eye(2) + 1j*np.sin(theta)*sigma_n

    return U
```

---

## 6. Validation and Testing

### 6.1. Unit Tests for Geometric Algorithms

**Test Suite Structure:**

```python
import pytest
import numpy as np
from numpy.testing import assert_allclose

class TestFanTriangulation:
    """Tests for Riemannian area computation."""

    def test_flat_space_matches_shoelace(self):
        """Fan triangulation should match shoelace formula in 2D flat space."""
        # Square in 2D
        positions = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)

        # Identity metric (flat space)
        def metric_flat(x, S):
            return np.eye(2)

        A_fan = compute_cycle_area_riemannian(positions, metric_flat, None)
        A_shoelace = compute_cycle_area_flat(positions)

        assert_allclose(A_fan, A_shoelace, rtol=1e-10)
        assert_allclose(A_fan, 1.0, rtol=1e-10)  # Unit square

    def test_scaled_metric_scales_area(self):
        """Metric scaling should scale area by √det(G)."""
        positions = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)

        # Scaled metric
        scale = 4.0
        def metric_scaled(x, S):
            return scale * np.eye(2)

        A_scaled = compute_cycle_area_riemannian(positions, metric_scaled, None)
        A_flat = compute_cycle_area_flat(positions)

        expected = A_flat * np.sqrt(scale)
        assert_allclose(A_scaled, expected, rtol=1e-10)

    def test_anisotropic_metric(self):
        """Test with anisotropic (diagonal) metric."""
        positions = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)

        # Diagonal metric (stretch in x direction)
        def metric_aniso(x, S):
            return np.diag([4.0, 1.0])

        A_aniso = compute_cycle_area_riemannian(positions, metric_aniso, None)
        A_flat = compute_cycle_area_flat(positions)

        expected = A_flat * np.sqrt(4.0 * 1.0)  # √(det(G)) = √4 = 2
        assert_allclose(A_aniso, expected, rtol=1e-10)


class TestFundamentalCycles:
    """Tests for cycle finding algorithms."""

    def test_simple_tree_cycle(self):
        """Test cycle finding on simple tree + 1 IG edge."""
        # Tree: 0 → 1 → 2, 0 → 3
        cst_edges = [(0, 1), (1, 2), (0, 3)]

        # IG edge: 2 ~ 3
        ig_edges = [(2, 3)]

        fractal = FractalSetCycles(cst_edges, ig_edges)
        cycle = fractal.fundamental_cycle((2, 3))

        # Expected cycle: 2 → 1 → 0 → 3 (or reversed)
        expected = [2, 1, 0, 3]
        assert cycle == expected or cycle == list(reversed(expected))

    def test_lca_symmetric(self):
        """LCA should be symmetric."""
        cst_edges = [(0, 1), (0, 2), (1, 3), (1, 4)]
        ig_edges = [(3, 4)]

        fractal = FractalSetCycles(cst_edges, ig_edges)

        lca_34 = fractal.find_lca(3, 4)
        lca_43 = fractal.find_lca(4, 3)

        assert lca_34 == lca_43 == 1


class TestWilsonLoops:
    """Tests for Wilson loop computation."""

    def test_trivial_loop_trace_unity(self):
        """Trivial loop (identity elements) should give Tr(I) = N_c."""
        N_c = 2
        path = [0, 1, 2, 0]  # Closed loop

        # Identity gauge links
        gauge_links = {
            (0, 1): np.eye(N_c, dtype=complex),
            (1, 2): np.eye(N_c, dtype=complex),
            (2, 0): np.eye(N_c, dtype=complex),
        }
        edge_directions = {k: 'forward' for k in gauge_links.keys()}

        U_path = path_ordered_product(path, gauge_links, edge_directions)

        assert_allclose(U_path, np.eye(N_c), atol=1e-12)
        assert_allclose(np.trace(U_path), N_c, atol=1e-12)

    def test_wilson_loop_gauge_invariance(self):
        """Wilson loop should be gauge invariant under local gauge transformation."""
        N_c = 2
        cycle = [0, 1, 2, 0]

        # Random gauge links
        np.random.seed(42)
        U_01 = random_su2()
        U_12 = random_su2()
        U_20 = random_su2()

        gauge_links = {(0, 1): U_01, (1, 2): U_12, (2, 0): U_20}
        edge_directions = {k: 'forward' for k in gauge_links.keys()}

        # Compute Wilson loop
        W_original = compute_wilson_loop((2, 0), cycle, gauge_links, edge_directions)

        # Apply gauge transformation g(i) at each vertex
        g = {0: random_su2(), 1: random_su2(), 2: random_su2()}

        gauge_links_transformed = {
            (0, 1): g[0] @ U_01 @ g[1].conj().T,
            (1, 2): g[1] @ U_12 @ g[2].conj().T,
            (2, 0): g[2] @ U_20 @ g[0].conj().T,
        }

        W_transformed = compute_wilson_loop((2, 0), cycle, gauge_links_transformed, edge_directions)

        # Should be equal (gauge invariant)
        assert_allclose(W_original, W_transformed, rtol=1e-10)


# Run tests
if __name__ == '__main__':
    pytest.main([__file__, '-v'])
```

### 6.2. Integration Test: End-to-End Workflow

```python
def test_full_geometric_pipeline():
    """Integration test: Adaptive Gas → Fractal Set → Geometric observables."""

    # Step 1: Run Adaptive Gas
    from fragile.adaptive_gas import AdaptiveGas
    from fragile.gas_parameters import AdaptiveGasParams

    params = AdaptiveGasParams(N=50, d=2, tau=0.1, gamma=1.0, epsilon_Sigma=0.01)
    gas = AdaptiveGas(params)

    for _ in range(50):
        gas.step()

    # Step 2: Extract Fractal Set
    episodes = gas.get_episodes()
    cst_edges = gas.get_cst_edges()
    ig_edges = gas.get_ig_edges()

    assert len(episodes) > 0, "No episodes generated"
    assert len(cst_edges) > 0, "CST is empty"
    assert len(ig_edges) > 0, "IG is empty"

    # Step 3: Find cycles
    fractal = FractalSetCycles(cst_edges, ig_edges)
    cycles = fractal.all_fundamental_cycles()

    assert len(cycles) == len(ig_edges), "Mismatch: cycles vs IG edges"

    # Step 4: Compute areas
    positions = {e.id: e.death_position for e in episodes}
    swarm_state = gas.get_current_swarm_state()

    for ig_edge, cycle in cycles.items():
        cycle_positions = np.array([positions[e] for e in cycle])

        A_curved = compute_cycle_area_riemannian(
            cycle_positions,
            gas.compute_metric_tensor,
            swarm_state
        )
        A_flat = compute_cycle_area_flat(cycle_positions)

        assert A_curved > 0, f"Non-positive curved area: {A_curved}"
        assert A_flat > 0, f"Non-positive flat area: {A_flat}"

        # Sanity check: curved area should be same order of magnitude
        assert 0.1 < A_curved / A_flat < 10.0, \
            f"Unreasonable area ratio: {A_curved / A_flat}"

    print("✅ Full geometric pipeline test passed")
```

---

## References

### Theoretical Foundations

1. **Fan Triangulation and Riemannian Areas:**
   - [13_D_fractal_set_emergent_qft_comprehensive.md § 8](../13_fractal_set_old/13_D_fractal_set_emergent_qft_comprehensive.md)
   - **Theorem 8.2:** Riemannian Area via Fan Triangulation
   - **Section 9:** Intrinsic vs Extrinsic Geometry

2. **Fundamental Cycles:**
   - [13_D_fractal_set_emergent_qft_comprehensive.md § 7](../13_fractal_set_old/13_D_fractal_set_emergent_qft_comprehensive.md)
   - **Theorem 7.1:** IG Edges Close Fundamental Cycles
   - **Algorithm:** Wilson Loop Construction

3. **Metric Tensor from Fitness Hessian:**
   - [fitness_algebra.py](../../src/fragile/fitness_algebra.py)
   - [08_emergent_geometry.md](../08_emergent_geometry.md)
   - **Definition:** $G(x, S) = H_\Phi(x, S) + \epsilon_\Sigma I$

4. **Continuum Limit and Local Covariance:**
   - [13_B_fractal_set_continuum_limit.md § 5](../13_fractal_set_old/13_B_fractal_set_continuum_limit.md)
   - **Proposition 5.2:** Inverse Covariance Approximates Metric

5. **Wilson Loops and Gauge Theory:**
   - [13_D_fractal_set_emergent_qft_comprehensive.md § 10](../13_fractal_set_old/13_D_fractal_set_emergent_qft_comprehensive.md)
   - **Definition 10.1:** Wilson Action with Riemannian Areas
   - [12_gauge_theory_adaptive_gas.md](../12_gauge_theory_adaptive_gas.md)

### Computational References

6. **Discrete Differential Geometry:**
   - Desbrun, M., Meyer, M., Schröder, P., & Barr, A. H. (2002). "Discrete Differential-Geometry Operators for Triangulated 2-Manifolds." *VisMath*.
   - Crane, K. (2013). "Discrete Differential Geometry: An Applied Introduction." CMU Lecture Notes.

7. **LCA Algorithms:**
   - Bender, M. A., & Farach-Colton, M. (2000). "The LCA Problem Revisited." *LATIN 2000: Theoretical Informatics*, 88-94.
   - Sparse table method: $O(N \log N)$ preprocessing, $O(\log N)$ queries

8. **Lattice Gauge Theory:**
   - Wilson, K. G. (1974). "Confinement of Quarks." *Physical Review D*, 10(8), 2445.
   - Creutz, M. (1983). *Quarks, Gluons and Lattices*. Cambridge University Press.

---

## Appendix: Mathematical Notation Summary

| Symbol | Meaning | Definition |
|--------|---------|------------|
| $\mathcal{F}$ | Fractal Set | $(\mathcal{E}, E_{\text{CST}} \cup E_{\text{IG}})$ |
| $\mathcal{E}$ | Episode set | Vertices of Fractal Set |
| $E_{\text{CST}}$ | CST edges | Directed parent→child genealogy |
| $E_{\text{IG}}$ | IG edges | Undirected companion interactions |
| $\Phi(e)$ | Episode embedding | Death position $x^d_e \in \mathcal{X}$ |
| $G(x, S)$ | Metric tensor | $H_\Phi(x, S) + \epsilon_\Sigma I$ |
| $H_\Phi$ | Fitness Hessian | $\nabla^2_x V_{\text{fit}}$ |
| $C(e)$ | Fundamental cycle | Closed by IG edge $e$ |
| $A(C)$ | Riemannian area | $\sum_i A_i$ via fan triangulation |
| $P_{\text{CST}}(a, b)$ | CST path | Unique tree path from $a$ to $b$ |
| $U_e$ | Gauge link | $SU(N_c)$ matrix on edge $e$ |
| $W_e$ | Wilson loop | $\text{Re Tr}(U_C) / N_c$ |
| $S_{\text{gauge}}$ | Wilson action | $\frac{\beta}{2N_c} \sum_e w_e (1 - W_e)$ |

---

**Document Complete**

This comprehensive guide provides:
- ✅ Practical algorithms with pseudocode
- ✅ Full Python implementations
- ✅ Complexity analysis
- ✅ Numerical stability considerations
- ✅ Validation tests
- ✅ End-to-end example workflow
- ✅ Cross-references to theoretical foundations

Ready for implementation in the Fragile codebase!
