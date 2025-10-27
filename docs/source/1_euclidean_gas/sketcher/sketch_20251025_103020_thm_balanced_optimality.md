# Proof Sketch: Balanced Optimality Condition

**Date:** 2025-10-25
**Theorem:** thm-balanced-optimality (Necessity of Balanced Rates at Optimum)
**Source:** docs/source/1_euclidean_gas/06_convergence.md § 6.5.3
**Type:** Theorem (Optimization Theory)

---

## 1. Theorem Statement

:::{prf:theorem} Necessity of Balanced Rates at Optimum
:label: sketch-thm-balanced-optimality

If $\mathbf{P}^*$ is a **local maximum** of $\kappa_{\text{total}}(\mathbf{P})$ in the interior of the feasible region, then at least two rates must be equal:

$$
\exists i \neq j : \kappa_i(\mathbf{P}^*) = \kappa_j(\mathbf{P}^*) = \kappa_{\text{total}}(\mathbf{P}^*)
$$

where $\kappa_{\text{total}}(\mathbf{P}) = \min(\kappa_1(\mathbf{P}), \kappa_2(\mathbf{P}), \kappa_3(\mathbf{P}), \kappa_4(\mathbf{P}))$ and $\kappa_i$ are the four convergence rates: $\kappa_x$ (position), $\kappa_v$ (velocity), $\kappa_W$ (Wasserstein), $\kappa_b$ (boundary).
:::

**Physical interpretation:** The optimal parameter configuration must balance at least two competing mechanisms. A single bottleneck rate can always be improved without harming other rates, contradicting optimality.

---

## 2. Dependencies and Prerequisites

This theorem relies on the following established results and definitions:

### From 06_convergence.md

1. **Theorem (Subgradient of min() Function)** {prf:ref}`thm-subgradient-min` (§ 6.5.2)
   - The subgradient set of $\kappa_{\text{total}} = \min(\kappa_i)$ is the convex hull of gradients of active constraints
   - Key property: At unique minimum, $\partial \kappa_{\text{total}} = \{\nabla \kappa_i\}$ where $i = \arg\min \kappa_j$

2. **Definition (Log-Sensitivity Matrix)** {prf:ref}`def-rate-sensitivity-matrix` (§ 6.3.1)
   - Relates parameter perturbations to rate changes: $(M_\kappa)_{ij} = \frac{\partial \log \kappa_i}{\partial \log P_j}$

3. **Explicit Rate Formulas** (§ 5, Chapter 5)
   - $\kappa_x = \lambda \cdot c_{\text{fit}} \cdot (1 - O(\tau))$
   - $\kappa_v = 2\gamma(1 - O(\tau))$
   - $\kappa_W = \frac{c_{\text{hypo}}^2 \gamma}{1 + \gamma/\lambda_{\min}}$
   - $\kappa_b = \min(\lambda, \kappa_{\text{wall}} + \gamma)$

### From Convex Analysis (Standard Results)

4. **First-Order Optimality Condition**
   - At interior local maximum: $0 \in \partial f(\mathbf{P}^*)$
   - For differentiable $f$: $\nabla f(\mathbf{P}^*) = 0$

5. **Concavity of min() Function**
   - $\min(f_1, \ldots, f_n)$ is concave if each $f_i$ is concave
   - Subgradient calculus applies

### Key Assumptions

- $\mathbf{P}^*$ is in the **interior** of the feasible region (not on boundary constraints)
- Each $\kappa_i(\mathbf{P})$ is **differentiable** in the interior
- Gradients $\nabla \kappa_i$ are **non-zero** (rates can be improved by parameter adjustment)

---

## 3. Proof Strategy

The proof proceeds by **contradiction** using subgradient calculus and first-order optimality conditions.

### High-Level Approach

1. **Assume the negation:** All four rates are strictly distinct at $\mathbf{P}^*$
2. **Identify the unique bottleneck:** $\kappa_{\text{total}} = \kappa_1 < \kappa_2, \kappa_3, \kappa_4$ (WLOG)
3. **Apply subgradient calculus:** At unique minimum, subgradient reduces to single gradient
4. **Invoke optimality condition:** For local maximum, must have $\nabla \kappa_1 = 0$
5. **Derive contradiction:** Show $\nabla \kappa_1 \neq 0$ using explicit rate formulas
6. **Construct improving direction:** Exhibit parameter perturbation that increases $\kappa_{\text{total}}$
7. **Conclude:** At least two rates must be equal at any local maximum

### Why This Works

The key insight is that the $\min()$ operator creates a **piecewise smooth** objective with non-smooth points exactly where rates coincide. At a **unique minimum** (all rates distinct), the subgradient simplifies to a single gradient, forcing a differentiability condition. But the explicit rate formulas show each rate has a non-zero gradient with respect to some parameter, allowing local improvement.

---

## 4. Detailed Proof Steps

### Step 1: Setup and Assumption

**Assume (for contradiction):** At $\mathbf{P}^*$, all four rates are strictly distinct:

$$
\kappa_1(\mathbf{P}^*) < \kappa_2(\mathbf{P}^*) < \kappa_3(\mathbf{P}^*) < \kappa_4(\mathbf{P}^*)
$$

**Without loss of generality:** Relabel so that $\kappa_1$ is the minimum.

**Consequence:**

$$
\kappa_{\text{total}}(\mathbf{P}^*) = \min_i \kappa_i(\mathbf{P}^*) = \kappa_1(\mathbf{P}^*)
$$

**Key observation:** Since the minimum is **unique** (no ties), the $\min()$ function is **differentiable** at $\mathbf{P}^*$ in a neighborhood where the ordering persists.

---

### Step 2: Subgradient Simplification

**Apply Theorem {prf:ref}`thm-subgradient-min`:**

Since only $\kappa_1$ achieves the minimum at $\mathbf{P}^*$, the subgradient set is:

$$
\partial \kappa_{\text{total}}(\mathbf{P}^*) = \text{conv}\{\nabla \kappa_i : \kappa_i(\mathbf{P}^*) = \kappa_{\text{total}}(\mathbf{P}^*)\} = \text{conv}\{\nabla \kappa_1(\mathbf{P}^*)\}
$$

**Simplification:** Convex hull of a single point is that point:

$$
\partial \kappa_{\text{total}}(\mathbf{P}^*) = \{\nabla \kappa_1(\mathbf{P}^*)\}
$$

**Conclusion:** At $\mathbf{P}^*$, the subgradient is **unique** and equals $\nabla \kappa_1(\mathbf{P}^*)$.

**Critical implication:** The non-smooth $\min()$ function behaves like a smooth function at this point, with gradient $\nabla \kappa_1$.

---

### Step 3: First-Order Optimality Condition

**Invoke standard optimality theory:**

If $\mathbf{P}^*$ is a **local maximum** of $\kappa_{\text{total}}$ in the interior of the feasible region, then:

$$
0 \in \partial \kappa_{\text{total}}(\mathbf{P}^*)
$$

**From Step 2:** Since $\partial \kappa_{\text{total}}(\mathbf{P}^*) = \{\nabla \kappa_1(\mathbf{P}^*)\}$, we have:

$$
\nabla \kappa_1(\mathbf{P}^*) = 0
$$

**Meaning:** The gradient of the bottleneck rate $\kappa_1$ must vanish at the optimal point.

**Physical interpretation:** No first-order parameter adjustment can improve the bottleneck rate.

---

### Step 4: Explicit Rate Analysis (Deriving Contradiction)

**Examine the explicit formula for $\kappa_1$:**

From the rate formulas in Chapter 5, each rate has the general structure:

$$
\kappa_i = f_i(\lambda, \gamma, \tau, \ldots)
$$

**Case analysis:** Depending on which rate is the bottleneck.

#### Case 4a: $\kappa_1 = \kappa_x$ (Position Rate is Bottleneck)

From § 5.2:

$$
\kappa_x = \lambda \cdot c_{\text{fit}} \cdot (1 - O(\tau))
$$

where $c_{\text{fit}} = c_{\text{fit}}(\lambda_{\text{alg}}, \epsilon_c) > 0$ is the fitness-variance correlation (depends on landscape, but is positive).

**Compute partial derivative:**

$$
\frac{\partial \kappa_x}{\partial \lambda} = c_{\text{fit}} \cdot (1 - O(\tau)) > 0
$$

**Conclusion:** $\nabla \kappa_x \neq 0$ because we can increase $\kappa_x$ by increasing $\lambda$.

#### Case 4b: $\kappa_1 = \kappa_v$ (Velocity Rate is Bottleneck)

From § 5.1:

$$
\kappa_v = 2\gamma(1 - O(\tau))
$$

**Compute partial derivative:**

$$
\frac{\partial \kappa_v}{\partial \gamma} = 2(1 - O(\tau)) > 0
$$

**Conclusion:** $\nabla \kappa_v \neq 0$ because we can increase $\kappa_v$ by increasing $\gamma$.

#### Case 4c: $\kappa_1 = \kappa_W$ (Wasserstein Rate is Bottleneck)

From § 5.3:

$$
\kappa_W = \frac{c_{\text{hypo}}^2 \gamma}{1 + \gamma/\lambda_{\min}}
$$

**Compute partial derivative:**

$$
\frac{\partial \kappa_W}{\partial \gamma} = c_{\text{hypo}}^2 \cdot \frac{\lambda_{\min}}{(1 + \gamma/\lambda_{\min})^2} > 0
$$

**Conclusion:** $\nabla \kappa_W \neq 0$ because we can increase $\kappa_W$ by increasing $\gamma$.

#### Case 4d: $\kappa_1 = \kappa_b$ (Boundary Rate is Bottleneck)

From § 5.4:

$$
\kappa_b = \min(\lambda, \kappa_{\text{wall}} + \gamma)
$$

**Subcase (i):** If $\kappa_b = \lambda$, then $\frac{\partial \kappa_b}{\partial \lambda} = 1 > 0$.

**Subcase (ii):** If $\kappa_b = \kappa_{\text{wall}} + \gamma$, then $\frac{\partial \kappa_b}{\partial \kappa_{\text{wall}}} = 1 > 0$ or $\frac{\partial \kappa_b}{\partial \gamma} = 1 > 0$.

**Conclusion:** $\nabla \kappa_b \neq 0$ in all cases.

---

### Step 5: Constructing an Improving Direction

**From Step 4:** In every case, $\nabla \kappa_1(\mathbf{P}^*) \neq 0$.

**Contradiction with Step 3:** We required $\nabla \kappa_1(\mathbf{P}^*) = 0$ for optimality, but the explicit formulas show this cannot hold.

**Explicit improvement:** Let $P_j$ be a parameter such that $\frac{\partial \kappa_1}{\partial P_j} > 0$. Consider the perturbation:

$$
\mathbf{P}_\epsilon = \mathbf{P}^* + \epsilon \mathbf{e}_j, \quad \epsilon > 0 \text{ small}
$$

where $\mathbf{e}_j$ is the unit vector in the $j$-th parameter direction.

**Effect on rates:**

1. **Bottleneck rate increases:** $\kappa_1(\mathbf{P}_\epsilon) = \kappa_1(\mathbf{P}^*) + \epsilon \frac{\partial \kappa_1}{\partial P_j} + O(\epsilon^2) > \kappa_1(\mathbf{P}^*)$

2. **Non-bottleneck rates unchanged (to first order):** Since $\kappa_i(\mathbf{P}^*) > \kappa_1(\mathbf{P}^*)$ for $i \neq 1$, and we assume $\mathbf{P}^*$ is interior, for sufficiently small $\epsilon$:

$$
\kappa_i(\mathbf{P}_\epsilon) \geq \kappa_i(\mathbf{P}^*) - O(\epsilon) > \kappa_1(\mathbf{P}^*) + \epsilon \frac{\partial \kappa_1}{\partial P_j} = \kappa_1(\mathbf{P}_\epsilon)
$$

for $\epsilon$ small enough.

**Conclusion:**

$$
\kappa_{\text{total}}(\mathbf{P}_\epsilon) = \min_i \kappa_i(\mathbf{P}_\epsilon) = \kappa_1(\mathbf{P}_\epsilon) > \kappa_1(\mathbf{P}^*) = \kappa_{\text{total}}(\mathbf{P}^*)
$$

**This contradicts the assumption that $\mathbf{P}^*$ is a local maximum.**

---

### Step 6: Conclusion

**Contradiction:** Our assumption that all four rates are strictly distinct leads to the conclusion that $\mathbf{P}^*$ is **not** a local maximum, contradicting the hypothesis.

**Resolution:** The assumption must be false. Therefore, at any local maximum $\mathbf{P}^*$, at least two rates must be equal:

$$
\exists i \neq j : \kappa_i(\mathbf{P}^*) = \kappa_j(\mathbf{P}^*) = \kappa_{\text{total}}(\mathbf{P}^*)
$$

**Q.E.D.**

---

## 5. Critical Estimates and Technical Requirements

### Regularity Conditions

For the proof to work, we need:

1. **Differentiability:** Each $\kappa_i(\mathbf{P})$ must be continuously differentiable in the interior of the feasible region.
   - **Verification:** The explicit formulas in § 5 are smooth functions of parameters (except at boundaries).

2. **Interior hypothesis:** $\mathbf{P}^*$ must be in the interior, not on the boundary.
   - **Necessity:** At boundary points, constraints can prevent parameter adjustments even if $\nabla \kappa_i \neq 0$.

3. **Non-degenerate gradients:** Each rate must have at least one parameter with non-zero partial derivative.
   - **Verification:** Step 4 shows this explicitly for all four rates.

### Quantitative Bounds (Not Required for Existence, But Useful)

The proof is **qualitative** (existence of improvement), not quantitative (how much improvement). For practical parameter optimization, one would additionally need:

- **Lipschitz constants** for $\nabla \kappa_i$ to bound the $O(\epsilon)$ terms in Step 5
- **Lower bounds** on $\frac{\partial \kappa_i}{\partial P_j}$ to ensure finite step sizes
- **Compactness** of feasible region to guarantee global optimum exists

These are **not needed** for the theorem as stated (which only concerns local maxima), but would be needed for algorithmic convergence guarantees.

---

## 6. Potential Difficulties and Resolutions

### Difficulty 1: Non-Smoothness of $\min()$ Operator

**Issue:** The objective $\kappa_{\text{total}} = \min(\kappa_i)$ is **non-differentiable** when multiple rates are equal.

**Resolution:**
- Use **subgradient calculus** (Theorem {prf:ref}`thm-subgradient-min`)
- The proof only requires analyzing points where all rates are **distinct** (the contradiction case)
- At such points, the $\min()$ function **is** differentiable with gradient $\nabla \kappa_i$ where $i = \arg\min$

### Difficulty 2: Boundary Effects

**Issue:** Parameters have **physical bounds** (e.g., $\lambda > 0$, $\gamma > 0$). Optimal points might lie on boundaries.

**Resolution:**
- Theorem explicitly assumes $\mathbf{P}^*$ is **interior**
- For boundary optima, use **KKT conditions** (Karush-Kuhn-Tucker) which generalize first-order conditions
- At boundary, balanced rates are **not necessary** (constraint can prevent improvement)

### Difficulty 3: Higher-Order Terms in Step 5

**Issue:** The perturbation analysis uses first-order Taylor expansion. What if higher-order terms dominate?

**Resolution:**
- The strict inequality $\frac{\partial \kappa_1}{\partial P_j} > 0$ ensures first-order term dominates for **sufficiently small** $\epsilon$
- The assumption that $\kappa_i(\mathbf{P}^*) > \kappa_1(\mathbf{P}^*)$ for $i \neq 1$ provides a **gap** that persists under small perturbations
- Formal statement: For each $\mathbf{P}^*$ with distinct rates, $\exists \delta > 0$ such that $\forall \epsilon \in (0, \delta)$, the improvement holds

### Difficulty 4: Multiple Local Maxima

**Issue:** The theorem applies to **each** local maximum individually. There may be multiple balanced configurations.

**Resolution:**
- The theorem is a **necessary condition**, not sufficient
- Multiple local maxima are expected (corresponding to different balancing strategies: $\kappa_x = \kappa_v$, or $\kappa_x = \kappa_W$, etc.)
- To find the **global** maximum, must compare all locally balanced configurations
- Typical approach: Enumerate all $\binom{4}{2} = 6$ two-way balances, solve each, compare

---

## 7. Geometric Interpretation

### Rate Surfaces in Parameter Space

Each rate $\kappa_i(\mathbf{P})$ defines a **hypersurface** in the 12-dimensional parameter space. The level sets:

$$
\{\mathbf{P} : \kappa_i(\mathbf{P}) = c\}
$$

are $(12-1) = 11$-dimensional surfaces.

### The Feasible Region

Constraints define a **polyhedral** or **convex** feasible region $\mathcal{F} \subseteq \mathbb{R}^{12}$.

### The Optimal Point Lies on an Intersection

**Theorem implication:** $\mathbf{P}^*$ must lie on the **intersection** of at least two rate surfaces:

$$
\mathbf{P}^* \in \{\mathbf{P} : \kappa_i(\mathbf{P}) = \kappa_j(\mathbf{P})\} \cap \mathcal{F}
$$

**Dimension reduction:** This intersection has dimension $\leq 11 - 1 = 10$ (codimension 2), further reducing the search space.

### Typical Optimal Configurations

From numerical experiments (§ 6.8 of 06_convergence.md):

1. **Friction-cloning balance:** $\kappa_x = \kappa_v < \kappa_W, \kappa_b$
   - Optimal for **overdamped** regimes ($\gamma$ large)

2. **Kinetic balance:** $\kappa_x = \kappa_v = \kappa_W < \kappa_b$
   - Optimal for **intermediate** regimes

3. **Full balance:** $\kappa_x = \kappa_v = \kappa_W = \kappa_b$
   - Rare, requires fine-tuning all parameters

---

## 8. Connections to Other Results

### Relation to Pareto Optimality

The balanced optimality condition is analogous to **Pareto efficiency** in multi-objective optimization:

- Each rate $\kappa_i$ is an "objective" to maximize
- The $\min()$ aggregation enforces a "worst-case" criterion
- Optimal points lie on the **Pareto frontier** where improving one rate requires sacrificing another

### Relation to Game Theory

The four rates can be viewed as "players" in a cooperative game:

- Each player wants to maximize its own rate
- The "team utility" is $\min(\kappa_i)$ (weakest link)
- Nash equilibrium requires balanced contributions (no player is strictly bottlenecking)

### Algorithmic Implications

For parameter optimization algorithms (§ 6.9 of 06_convergence.md):

- **Gradient ascent** will naturally converge to balanced points (where subgradient includes zero)
- **Greedy improvement** (always increase the bottleneck rate) leads to balanced configurations
- **Bisection methods** can exploit the structure: binary search for parameters that equalize two rates

---

## 9. Extensions and Open Questions

### Extension 1: Three-Way or Four-Way Balance

**Question:** When is a three-way or four-way balance optimal (not just two-way)?

**Conjecture:** This occurs when the **null space** of the sensitivity matrix $M_\kappa$ intersects the balance manifold:

$$
\mathcal{N}(M_\kappa) \cap \{\mathbf{P} : \kappa_1 = \kappa_2 = \kappa_3 = \kappa_4\} \neq \emptyset
$$

**Significance:** Such configurations are **highly degenerate** and may have enhanced robustness.

### Extension 2: Global Optimality

**Question:** Which balanced configuration is the **global** maximum?

**Approach:**
- Enumerate all $2^4 - 1 = 15$ possible subsets of active constraints
- For each, solve the balance equations: $\kappa_i = \kappa_j = \ldots = c$
- Compare the resulting $c$ values; largest is global optimum

**Challenge:** Balance equations may have **no solution** for some subsets (incompatible constraints).

### Extension 3: Boundary Optima

**Question:** Can the global optimum lie on the boundary of the feasible region?

**Answer:** Yes, if constraints bind before balance is achieved.

**Example:** If $\lambda \leq \lambda_{\max}$ constraint is active and $\kappa_x = \lambda \cdot c_{\text{fit}}$ is still the bottleneck, then $\mathbf{P}^* = (\lambda_{\max}, \ldots)$ is optimal despite $\kappa_x < \kappa_v$.

**KKT Conditions:** At boundary, first-order condition becomes:

$$
\nabla \kappa_{\text{total}} = \sum_i \mu_i \nabla g_i \quad (\mu_i \geq 0 \text{ for active constraints } g_i \geq 0)
$$

### Extension 4: Time-Varying Parameters

**Question:** For annealing schedules, how should parameters evolve?

**Heuristic:** Start with high exploration ($\sigma_v$ large, $\lambda$ moderate), gradually increase exploitation ($\sigma_v \to 0$, $\lambda$ increases), maintaining balance $\kappa_x \approx \kappa_v$ throughout.

**Open problem:** Optimal annealing schedule that maximizes **cumulative reward** (not just convergence rate).

---

## 10. Summary

### Key Takeaways

1. **Necessity of Balance:** At any **interior** local maximum, at least two convergence rates must be equal.

2. **Proof Technique:** Contradiction via subgradient calculus + first-order optimality conditions.

3. **Critical Insight:** Explicit rate formulas show all gradients are non-zero, preventing unique bottlenecks at optima.

4. **Geometric Picture:** Optimal points lie on intersections of rate surfaces (codimension ≥ 2).

5. **Algorithmic Consequence:** Optimization algorithms should target **balanced configurations** for efficient search.

### Proof Sketch Validation Checklist

- [x] Theorem statement clearly reproduced
- [x] Dependencies identified and referenced
- [x] Proof strategy outlined (contradiction)
- [x] Key steps numbered and detailed
- [x] Critical estimates examined (gradient non-vanishing)
- [x] Potential difficulties addressed (non-smoothness, boundary effects, higher-order terms)
- [x] Geometric interpretation provided
- [x] Connections to other results noted
- [x] Extensions and open questions discussed

### Next Steps for Full Proof

1. **Formalize regularity conditions:** State precise differentiability assumptions on $\kappa_i(\mathbf{P})$.

2. **Generalize to $n$ rates:** The proof extends immediately to $\kappa_{\text{total}} = \min(\kappa_1, \ldots, \kappa_n)$ for arbitrary $n$.

3. **Quantitative bounds:** Derive explicit constants for the improvement step size $\delta$ in Step 5.

4. **Boundary analysis:** Extend theorem to boundary cases using KKT conditions.

5. **Global optimality:** Develop algorithms to enumerate and compare all locally balanced configurations.

6. **Numerical validation:** Verify the theorem on concrete parameter optimization examples from § 6.8.

---

**End of Proof Sketch**
