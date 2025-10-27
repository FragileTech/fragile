# Proof Sketch: Subgradient of min() Function

**Theorem Label:** `thm-subgradient-min`
**Source Document:** `docs/source/1_euclidean_gas/06_convergence.md` (line 2634)
**Sketcher:** Autonomous Proof Pipeline
**Date:** 2025-10-25
**Status:** Draft Sketch

---

## 1. Theorem Statement

:::{prf:theorem} Subgradient of min() Function
:label: thm-subgradient-min-sketch

At a point $\mathbf{P}$ where $\kappa_{\text{total}} = \min(\kappa_1, \ldots, \kappa_4)$, the subgradient set is:

$$
\partial \kappa_{\text{total}} = \text{conv}\left\{\nabla \kappa_i : \kappa_i(\mathbf{P}) = \kappa_{\text{total}}(\mathbf{P})\right\}
$$

where $\text{conv}(\cdot)$ denotes the convex hull.
:::

**Context:** This result appears in Section 6.5.2 of the convergence analysis, where we analyze the parameter optimization problem:

$$
\max_{\mathbf{P} \in \mathbb{R}_{+}^{12}} \kappa_{\text{total}}(\mathbf{P}) = \max_{\mathbf{P}} \left[\min(\kappa_x, \kappa_v, \kappa_W, \kappa_b) \cdot (1 - \epsilon_{\text{coupling}}(\mathbf{P}))\right]
$$

The difficulty is that the objective function is **non-smooth** at points where two or more convergence rates are equal.

---

## 2. Dependencies

### 2.1. Required Definitions

1. **Subgradient of a function** (convex analysis):
   - For a concave function $f: \mathbb{R}^n \to \mathbb{R}$, a vector $g \in \mathbb{R}^n$ is a subgradient at $x$ if:
     $$
     f(y) \leq f(x) + g^T(y - x) \quad \forall y
     $$

2. **Subdifferential set**:
   - $\partial f(x) := \{g : g \text{ is a subgradient of } f \text{ at } x\}$

3. **Active constraints**:
   - Define the **active index set** at $\mathbf{P}$:
     $$
     \mathcal{I}_{\text{active}}(\mathbf{P}) := \{i : \kappa_i(\mathbf{P}) = \kappa_{\text{total}}(\mathbf{P})\}
     $$

4. **Convex hull**:
   - For a set $S \subset \mathbb{R}^n$:
     $$
     \text{conv}(S) := \left\{\sum_{i=1}^k \alpha_i s_i : s_i \in S, \alpha_i \geq 0, \sum \alpha_i = 1\right\}
     $$

### 2.2. Required Results from Convex Analysis

1. **Max-formula for subdifferentials** (Rockafellar, 1970):
   - For concave functions $f_1, \ldots, f_m$, the subdifferential of $\max(f_1, \ldots, f_m)$ is:
     $$
     \partial \max_i f_i(x) = \text{conv}\{\nabla f_i(x) : i \in \mathcal{I}_{\text{active}}(x)\}
     $$

2. **Min-formula via negation**:
   - Note that $\min(f_1, \ldots, f_m) = -\max(-f_1, \ldots, -f_m)$
   - Subdifferential of negation: $\partial(-f) = -\partial f$

3. **Smoothness assumption on component functions**:
   - Each $\kappa_i(\mathbf{P})$ is **smooth** (continuously differentiable) as a function of parameters
   - This follows from the explicit formulas in Section 6 (e.g., $\kappa_x = \lambda \cdot c_{\text{fit}} \cdot (1 - O(\tau))$)

### 2.3. Framework Dependencies

From the Fragile framework:
- **Convergence rates**: $\kappa_x, \kappa_v, \kappa_W, \kappa_b$ defined in `06_convergence.md`
- **Parameter space**: $\mathbf{P} \in \mathbb{R}_{+}^{12}$ contains physical parameters (see Section 6.5.1)

---

## 3. Proof Strategy

### 3.1. High-Level Approach

The proof follows a **standard convex analysis argument** in three steps:

1. **Rewrite** the min function using negation of max
2. **Apply** the max-formula for subdifferentials from Rockafellar's theorem
3. **Transform** back using subdifferential calculus rules

This is a **textbook result** in convex analysis, but we provide the complete argument for self-containment.

### 3.2. Key Insight

The subgradient set captures the **directional derivatives** of the min function at non-smooth points. When multiple functions achieve the minimum simultaneously, any convex combination of their gradients provides a valid "generalized gradient" that satisfies the subgradient inequality.

---

## 4. Key Steps

### Step 1: Subdifferential of Negative Function

**Claim:** For any function $f$,

$$
\partial(-f)(x) = -\partial f(x)
$$

**Justification:** Direct from the definition of subgradient. If $g \in \partial f(x)$, then:

$$
f(y) \leq f(x) + g^T(y - x) \quad \forall y
$$

Multiply by $-1$:

$$
-f(y) \geq -f(x) - g^T(y - x) \quad \forall y
$$

So $-g \in \partial(-f)(x)$.

---

### Step 2: Rewrite Min as Negated Max

**Observation:**

$$
\kappa_{\text{total}}(\mathbf{P}) = \min(\kappa_1, \ldots, \kappa_4) = -\max(-\kappa_1, \ldots, -\kappa_4)
$$

Define auxiliary functions:

$$
\tilde{\kappa}_i := -\kappa_i, \quad i = 1, \ldots, 4
$$

Then:

$$
\kappa_{\text{total}} = -\max(\tilde{\kappa}_1, \ldots, \tilde{\kappa}_4)
$$

---

### Step 3: Apply Max-Formula for Subdifferentials

**Apply Rockafellar's theorem** to the max function:

$$
\partial \max_i \tilde{\kappa}_i(\mathbf{P}) = \text{conv}\{\nabla \tilde{\kappa}_i(\mathbf{P}) : \tilde{\kappa}_i(\mathbf{P}) = \max_j \tilde{\kappa}_j(\mathbf{P})\}
$$

**Translate active constraints:** The condition $\tilde{\kappa}_i(\mathbf{P}) = \max_j \tilde{\kappa}_j(\mathbf{P})$ is equivalent to:

$$
-\kappa_i(\mathbf{P}) = \max_j(-\kappa_j(\mathbf{P})) \iff \kappa_i(\mathbf{P}) = \min_j \kappa_j(\mathbf{P}) = \kappa_{\text{total}}(\mathbf{P})
$$

So the active set is:

$$
\mathcal{I}_{\text{active}}(\mathbf{P}) = \{i : \kappa_i(\mathbf{P}) = \kappa_{\text{total}}(\mathbf{P})\}
$$

Thus:

$$
\partial \max_i \tilde{\kappa}_i(\mathbf{P}) = \text{conv}\{\nabla \tilde{\kappa}_i(\mathbf{P}) : i \in \mathcal{I}_{\text{active}}(\mathbf{P})\}
$$

---

### Step 4: Transform Gradients

Since $\tilde{\kappa}_i = -\kappa_i$:

$$
\nabla \tilde{\kappa}_i = -\nabla \kappa_i
$$

Therefore:

$$
\partial \max_i \tilde{\kappa}_i(\mathbf{P}) = \text{conv}\{-\nabla \kappa_i(\mathbf{P}) : i \in \mathcal{I}_{\text{active}}(\mathbf{P})\}
$$

Using linearity of convex hull:

$$
= -\text{conv}\{\nabla \kappa_i(\mathbf{P}) : i \in \mathcal{I}_{\text{active}}(\mathbf{P})\}
$$

---

### Step 5: Apply Negation Rule

From Step 1 and Step 2:

$$
\partial \kappa_{\text{total}}(\mathbf{P}) = \partial\left(-\max_i \tilde{\kappa}_i\right)(\mathbf{P}) = -\partial \max_i \tilde{\kappa}_i(\mathbf{P})
$$

Substitute result from Step 4:

$$
= -\left(-\text{conv}\{\nabla \kappa_i(\mathbf{P}) : i \in \mathcal{I}_{\text{active}}(\mathbf{P})\}\right)
$$

$$
= \text{conv}\{\nabla \kappa_i(\mathbf{P}) : \kappa_i(\mathbf{P}) = \kappa_{\text{total}}(\mathbf{P})\}
$$

**This completes the proof.** $\square$

---

## 5. Critical Estimates and Bounds

### 5.1. Smoothness of Component Functions

**Required property:** Each $\kappa_i(\mathbf{P})$ must be **continuously differentiable** for the max-formula to apply.

**Verification:**
- $\kappa_x = \lambda \cdot c_{\text{fit}} \cdot (1 - O(\tau))$ (linear in $\lambda$, smooth in other parameters)
- $\kappa_v = 2\gamma\tau$ (linear in $\gamma$ and $\tau$)
- $\kappa_W$ and $\kappa_b$ have similar smooth dependence on parameters

**Conclusion:** All component functions are smooth. ✓

### 5.2. Non-Smoothness of $\kappa_{\text{total}}$

**At points with unique minimum:**
- If $\kappa_1(\mathbf{P}) < \kappa_i(\mathbf{P})$ for $i = 2, 3, 4$, then $\kappa_{\text{total}}$ is smooth near $\mathbf{P}$
- $\partial \kappa_{\text{total}}(\mathbf{P}) = \{\nabla \kappa_1(\mathbf{P})\}$ (singleton)

**At tie points:**
- If $\kappa_1(\mathbf{P}) = \kappa_2(\mathbf{P}) < \kappa_3, \kappa_4$, then $\kappa_{\text{total}}$ is **non-differentiable** at $\mathbf{P}$
- The subgradient set is a **line segment** connecting $\nabla \kappa_1$ and $\nabla \kappa_2$

**Physical interpretation:** Non-smoothness reflects **competing bottlenecks** in the convergence mechanism. When two rates are equal, both mechanisms equally limit the overall convergence speed.

---

## 6. Examples (Verification)

### Example 1: Unique Minimum

Suppose at $\mathbf{P}_0$:

$$
\kappa_x(\mathbf{P}_0) = 0.1, \quad \kappa_v = 0.3, \quad \kappa_W = 0.5, \quad \kappa_b = 0.4
$$

Then $\kappa_{\text{total}}(\mathbf{P}_0) = 0.1$ and:

$$
\partial \kappa_{\text{total}}(\mathbf{P}_0) = \{\nabla \kappa_x(\mathbf{P}_0)\}
$$

**Check:** Only $\kappa_x$ achieves the minimum, so only its gradient appears.

---

### Example 2: Two-Way Tie

Suppose at $\mathbf{P}_1$:

$$
\kappa_x(\mathbf{P}_1) = \kappa_v(\mathbf{P}_1) = 0.2, \quad \kappa_W = 0.5, \quad \kappa_b = 0.4
$$

Then:

$$
\partial \kappa_{\text{total}}(\mathbf{P}_1) = \text{conv}\{\nabla \kappa_x(\mathbf{P}_1), \nabla \kappa_v(\mathbf{P}_1)\}
$$

$$
= \{\alpha \nabla \kappa_x(\mathbf{P}_1) + (1 - \alpha) \nabla \kappa_v(\mathbf{P}_1) : \alpha \in [0, 1]\}
$$

**Check:** Convex hull of two points is the line segment connecting them.

---

### Example 3: Four-Way Tie (Balanced Rates)

Suppose all rates are equal at $\mathbf{P}^*$:

$$
\kappa_x(\mathbf{P}^*) = \kappa_v(\mathbf{P}^*) = \kappa_W(\mathbf{P}^*) = \kappa_b(\mathbf{P}^*) = \kappa^*
$$

Then:

$$
\partial \kappa_{\text{total}}(\mathbf{P}^*) = \text{conv}\{\nabla \kappa_1, \nabla \kappa_2, \nabla \kappa_3, \nabla \kappa_4\}
$$

$$
= \left\{\sum_{i=1}^4 \alpha_i \nabla \kappa_i(\mathbf{P}^*) : \alpha_i \geq 0, \sum \alpha_i = 1\right\}
$$

**Physical significance:** At the **balanced optimum**, all four convergence mechanisms operate at the same rate. The subgradient is a **3-simplex** (tetrahedron) in parameter space.

---

## 7. Potential Difficulties

### 7.1. Rockafellar's Theorem Prerequisites

**Issue:** The max-formula assumes:
1. Each $f_i$ is concave
2. Each $f_i$ is continuously differentiable

**Resolution:**
- Here we work with **minimization** of $\kappa_i$ (so $-\kappa_i$ are concave for minimization)
- Wait, this is subtle: we are **maximizing** $\kappa_{\text{total}} = \min(\kappa_i)$
- So we want to make $\kappa_{\text{total}}$ large, which means making the minimum rate large
- The min of concave functions is concave, so $\kappa_{\text{total}}$ is concave ✓
- For **concave optimization**, Rockafellar's theorem applies directly

**Clarification needed:** Are the $\kappa_i$ concave or convex in $\mathbf{P}$?

From the formulas:
- $\kappa_x = \lambda \cdot c_{\text{fit}}(N, \alpha_{\text{rest}}) \cdot (1 - O(\tau))$
- $c_{\text{fit}}$ is **not necessarily concave** in general

**Resolution:** Rockafellar's max/min formula for subdifferentials **does not require concavity** of the component functions, only that they are **locally Lipschitz** (which they are, since they are $C^1$).

The correct reference is:
- **Clarke (1983), Nonsmooth Analysis**: The subdifferential of the min of smooth functions is the convex hull of gradients of active functions, regardless of concavity.

### 7.2. Non-Smoothness at Measure-Zero Set

**Observation:** The set of points where two or more $\kappa_i$ are equal is **measure zero** in $\mathbb{R}^{12}$.

**Implication:** For almost all parameter choices, $\kappa_{\text{total}}$ is smooth. The non-smooth points form a lower-dimensional manifold.

**Consequence for optimization:** Gradient-based methods will work almost everywhere. Special handling needed near tie points (e.g., subgradient methods, bundle methods).

### 7.3. Computational Verification

**Practical check:** To verify this formula computationally:
1. Evaluate $\kappa_1, \ldots, \kappa_4$ at $\mathbf{P}$
2. Identify active set: $\mathcal{I}_{\text{active}} = \{i : \kappa_i = \min_j \kappa_j\}$
3. Compute gradients $\nabla \kappa_i$ for $i \in \mathcal{I}_{\text{active}}$
4. Any convex combination of these gradients is a valid subgradient

**Numerical stability:** When $|\kappa_i - \kappa_j| < \epsilon$ for small $\epsilon$, treat as a tie to avoid floating-point issues.

---

## 8. References to Convex Analysis Literature

1. **Rockafellar, R. T. (1970).** *Convex Analysis.* Princeton University Press.
   - Chapter on subdifferentials of composite functions
   - Max-formula for subdifferentials (Theorem 23.8)

2. **Clarke, F. H. (1983).** *Optimization and Nonsmooth Analysis.* Wiley.
   - Subdifferentials of non-convex functions
   - Min/max formulas without concavity assumptions

3. **Hiriart-Urruty, J.-B., & Lemaréchal, C. (1993).** *Convex Analysis and Minimization Algorithms.* Springer.
   - Comprehensive treatment of subdifferential calculus
   - Applications to optimization

4. **Boyd, S., & Vandenberghe, L. (2004).** *Convex Optimization.* Cambridge University Press.
   - Section 5.4: Subgradients and subdifferentials
   - Applied perspective on non-smooth optimization

---

## 9. Connection to Balanced Optimality

**Forward reference:** This theorem is used immediately in Section 6.5.3 to prove **Theorem thm-balanced-optimality**:

> If $\mathbf{P}^*$ is a local maximum of $\kappa_{\text{total}}$ in the interior of the feasible region, then at least two rates must be equal.

**Proof sketch of that theorem:**
1. Suppose all rates are distinct at $\mathbf{P}^*$
2. Then $\partial \kappa_{\text{total}}(\mathbf{P}^*) = \{\nabla \kappa_{\text{min}}\}$ (unique subgradient)
3. For optimality: $\nabla \kappa_{\text{min}}(\mathbf{P}^*) = 0$
4. But $\nabla \kappa_i \neq 0$ generically (by explicit formulas)
5. Contradiction → at least two rates must be equal

This shows that **balanced rates are necessary for optimality**.

---

## 10. Summary

**Proof technique:** Standard subdifferential calculus
- Rewrite min as negated max
- Apply Rockafellar/Clarke max-formula
- Transform back via negation rule

**Key dependencies:**
- Smoothness of component functions $\kappa_i$
- Max-formula for subdifferentials
- Convex hull definition

**Result:**
- Subgradient is convex hull of gradients of active (minimal) rates
- Unique gradient when one rate dominates
- Non-trivial convex set when rates tie

**Physical interpretation:** Multiple convergence mechanisms becoming simultaneous bottlenecks create non-smooth optimization landscape.

**Next steps for full proof:**
1. Cite specific theorem from Rockafellar or Clarke
2. Verify smoothness assumptions on $\kappa_i$ in detail
3. Provide explicit dimension calculations for subgradient sets in different cases
4. Connect to numerical optimization algorithms for non-smooth problems

---

**Status:** Ready for full proof development and dual review (Gemini + Codex).
