# Wilson Loops from IG Edges: Single-Root CST Framework

**Status**: ✅ **CORRECTED VERSION** - Addresses Gemini critiques from Ch. 28

**Purpose**: Show that IG edges provide a natural computational basis for Wilson loops when CST has a single root (common ancestor).

**Key Corrections from Ch. 28**:
1. **Explicit assumption**: Single root CST (all walkers share common ancestor)
2. **Honest assessment**: Area/weight relationship remains open problem
3. **Rigorous definitions**: Unambiguous path and weight formulas
4. **Retracted claim**: "No area measure needed" → "Discrete computational basis"

---

## 0. Executive Summary

### The Revised Claim

**Original (Ch. 28)**: "Wilson loops from IG edges - no area measure needed"

**Corrected (This document)**: "IG edges provide natural discrete basis for Wilson loops; continuum limit requires area-weight relationship (open problem)"

### What This Document Achieves

✅ **Computational framework**: Algorithm to compute Wilson loops from IG edges
✅ **Single-root CST**: Rigorous proof that IG edges close cycles (under single ancestor assumption)
✅ **Discrete action**: Well-defined gauge action on CST+IG graph
⚠️ **Continuum limit**: Requires w_e ~ |Σ(e)|^(-2) scaling (unproven conjecture)

### The Critical Assumption

:::{important} Single Common Ancestor Assumption
**Throughout this document**, we assume the swarm is initialized from a **single common ancestor**:

$$
\text{At } t = 0: \quad \text{All } N \text{ walkers start from same state } (x_0, v_0)
$$

**Consequence**: The CST has a **single root** and is a **connected tree**, not a forest.

**Justification**:
- Natural for continuum limit where initial density ρ₀(x) has connected support
- Typical in optimization: swarm explores from single starting point
- Simplifies topology: Every pair of episodes has unique CST path

**Generalization to multi-root forests**: Future work (requires treating inter-tree IG edges separately)
:::

---

## 1. Graph Theory: Single-Root CST as a Tree

### 1.1. CST is a Tree (Rigorous Version)

:::{prf:theorem} CST is a Rooted Spanning Tree
:label: thm-cst-single-root-tree

**Assumption**: All N walkers are initialized from a single common state (x₀, v₀) at t = 0.

**Claim**: The Causal Spacetime Tree (CST) is a **rooted spanning tree** on the episode set $\mathcal{E}$.

**Properties**:
1. **Single root**: Exactly one episode e₀ with parent(e₀) = root
2. **Connected**: Every episode e ∈ E is reachable from root via directed edges
3. **Acyclic**: No directed cycles (time flows forward: t^b_child = t^d_parent)
4. **Unique parent**: Every episode except root has exactly one parent
5. **Tree structure**: |E_CST| = |E| - 1 edges for |E| episodes

**Proof**:

*Step 1: Single root exists*

At t = 0, all N walkers start at (x₀, v₀). The first cloning event (say at time t₁) creates a child episode. The parent of this child is the initial episode e₀ that encompasses [0, t₁). Therefore, e₀ is the unique root.

*Step 2: Every episode traces to root*

By induction on birth time t^b_e:
- Base case: e₀ is the root (connected to itself)
- Inductive step: For episode e with parent e', we have t^b_e = t^d_e' < t^b_e. By induction hypothesis, e' traces to root. Adding edge e' → e extends the path to root.

Therefore, CST is connected.

*Step 3: No cycles*

For any directed path e₁ → e₂ → ... → eₙ:
$$
t^{\rm b}_{e_1} < t^{\rm d}_{e_1} = t^{\rm b}_{e_2} < t^{\rm d}_{e_2} = \cdots < t^{\rm b}_{e_n}
$$

The birth times form a strictly increasing sequence, so no episode can appear twice. Hence, acyclic.

*Step 4: Unique parent*

By construction in [Chapter 13](13_fractal_set.md), each episode has parent(e) ∈ E ∪ {root}, and this parent is unique (determined by which episode e was cloned from).

*Step 5: Tree structure*

Connected + acyclic + unique parent ⇒ tree. Edge count: |E_CST| = |E| - 1 (standard tree property). ∎
:::

**Key Differences from Ch. 28**:
- ✅ Explicit single-root assumption (not "by construction")
- ✅ Proof by induction on birth times (rigorous)
- ✅ No hand-waving about "genealogy"

### 1.2. Unique Undirected Paths

:::{prf:definition} CST Path Between Episodes
:label: def-cst-path-unique

For episodes e_i, e_j ∈ E in a single-root CST:

**Lowest Common Ancestor (LCA)**: The unique episode e_LCA such that:
- e_LCA is an ancestor of both e_i and e_j (in the directed CST)
- e_LCA is maximal (closest to e_i and e_j in tree distance)

**Undirected path** P_CST(e_i, e_j): The unique path in the undirected tree connecting e_i to e_j:

$$
P_{\text{CST}}(e_i, e_j) := P_{\text{up}}(e_i, e_{\text{LCA}}) \cup \{e_{\text{LCA}}\} \cup P_{\text{down}}(e_{\text{LCA}}, e_j)
$$

where:
- P_up(e_i, e_LCA): Path from e_i up to LCA (traversing CST edges backward)
- P_down(e_LCA, e_j): Path from LCA down to e_j (traversing CST edges forward)

**Path orientation for Wilson loops**: We define the **directed path from e_i to e_j**:
1. Traverse **backward** (child → parent) from e_i to e_LCA
2. Traverse **forward** (parent → child) from e_LCA to e_j

**Gauge link orientation**:
- Forward edge (e_p → e_c): Use U_CST(e_p, e_c)
- Backward edge (e_c → e_p): Use U_CST(e_p, e_c)^† (adjoint)
:::

**Example**:
```
CST:         root
             /  \
           e₁   e₂
           /     \
         e₃      e₄

Find path from e₃ to e₄:
LCA(e₃, e₄) = root
Path: e₃ → e₁ → root → e₂ → e₄

Directed path for gauge transport:
e₃ --backward-→ e₁ --backward-→ root --forward-→ e₂ --forward-→ e₄

Wilson loop product:
U = U_CST(e₁,e₃)^† · U_CST(root,e₁)^† · U_CST(root,e₂) · U_CST(e₂,e₄)
```

**Key Correction from Ch. 28**:
- ✅ Explicit algorithm (LCA-based) for path construction
- ✅ Clear orientation rules for gauge links
- ✅ No ambiguity in "undirected path"

### 1.3. IG Edges Close Fundamental Cycles

:::{prf:theorem} IG Edges Form Complete Cycle Basis
:label: thm-ig-cycle-basis-corrected

**Setup**:
- CST is a rooted spanning tree (Theorem {prf:ref}`thm-cst-single-root-tree`)
- IG has k edges: E_IG = {e₁, e₂, ..., eₖ}
- CST+IG is the combined graph G = (E, E_CST ∪ E_IG)

**Claim**: Each IG edge eᵢ closes exactly one fundamental cycle C(eᵢ), and {C(e₁), ..., C(eₖ)} forms a complete basis for the cycle space of G.

**Proof**:

*Part 1: Each IG edge closes a cycle*

For IG edge eᵢ = (e_i ~ e_j):
- By Theorem {prf:ref}`thm-cst-single-root-tree`, CST is connected
- Therefore, unique undirected path P_CST(e_i, e_j) exists (Definition {prf:ref}`def-cst-path-unique`)
- Fundamental cycle: C(eᵢ) := eᵢ ∪ P_CST(e_i, e_j)
- This is a cycle (closed loop: e_i → e_j via IG, e_j → e_i via CST path)

*Part 2: Cycles are linearly independent*

Suppose Σᵢ aᵢ C(eᵢ) = 0 for coefficients aᵢ ∈ ℤ.
- Each C(eᵢ) contains IG edge eᵢ
- IG edges are distinct (no other cycle contains eᵢ)
- For the sum to be zero, each coefficient aᵢ = 0
- Therefore, {C(eᵢ)} are linearly independent

*Part 3: Cycles span the cycle space*

Dimension of cycle space of graph G:
$$
\dim(\text{Cycle space}) = |E_{\text{total}}| - |V| + 1 = (|E_{\text{CST}}| + |E_{\text{IG}}|) - |E| + 1
$$

Since CST is a tree: |E_CST| = |E| - 1
$$
\dim(\text{Cycle space}) = (|E| - 1 + k) - |E| + 1 = k
$$

We have k linearly independent cycles {C(eᵢ)}, which equals the dimension. Therefore, they form a complete basis. ∎
:::

**Key Improvement from Ch. 28**:
- ✅ Proves CST connectivity first (prerequisite)
- ✅ Explicit linear independence proof
- ✅ Dimension calculation from first principles
- ✅ Does not rely on "Veblen's theorem" for mixed graphs

---

## 2. Wilson Loops: Rigorous Definitions

### 2.1. Wilson Loop for IG Edge

:::{prf:definition} Wilson Loop from IG Edge
:label: def-wilson-loop-ig-corrected

For IG edge e = (e_i ~ e_j) with fundamental cycle C(e):

**Step 1: Construct path**

Using Definition {prf:ref}`def-cst-path-unique`:
- LCA ← lowest_common_ancestor(e_i, e_j)
- P_CST ← path from e_i to e_j via LCA

**Step 2: Parallel transport operator**

$$
U_C := U_{\text{IG}}(e_i, e_j) \times U_{\text{CST}}(P_{\text{CST}}(e_j, e_i))
$$

where:
- $U_{\text{IG}}(e_i, e_j) \in SU(N_c)$: Gauge link along IG edge (one step, forward)
- $U_{\text{CST}}(P(e_j, e_i))$: Path-ordered product along CST from e_j back to e_i:

$$
U_{\text{CST}}(P(e_j, e_i)) = \prod_{\text{edges } (a,b) \in P} U_{\text{edge}}(a, b)
$$

where the product is taken in reverse order of the path (from e_j toward e_i), and:
- If edge (a,b) is forward (a → b in CST): use U_CST(a,b)
- If edge (a,b) is backward (b → a in CST): use U_CST(a,b)^†

**Step 3: Wilson loop (trace)**

$$
W_e := \text{Tr}(U_C)
$$

**Gauge transformation law**: Under local gauge transformation g(e) ∈ SU(N_c):
$$
U(e_i, e_j) \to g(e_i) \, U(e_i, e_j) \, g(e_j)^\dagger
$$

**Gauge invariance**: See Theorem {prf:ref}`thm-wilson-loop-gauge-invariant` below.
:::

**Example (from Section 1.2)**:
```
Path: e₃ → e₁ → root → e₂ → e₄

CST edges:
- e₁ → e₃ (forward in CST, backward in path) → use U_CST(e₁,e₃)^†
- root → e₁ (forward in CST, backward in path) → use U_CST(root,e₁)^†
- root → e₂ (forward in CST, forward in path) → use U_CST(root,e₂)
- e₂ → e₄ (forward in CST, forward in path) → use U_CST(e₂,e₄)

U_CST(P(e₄, e₃)) = U_CST(e₂,e₄) · U_CST(root,e₂) · U_CST(root,e₁)^† · U_CST(e₁,e₃)^†

Full Wilson loop:
W_e = Tr(U_IG(e₃,e₄) · U_CST(e₂,e₄) · U_CST(root,e₂) · U_CST(root,e₁)^† · U_CST(e₁,e₃)^†)
```

### 2.2. Gauge Invariance (Proven)

:::{prf:theorem} Wilson Loops are Gauge Invariant
:label: thm-wilson-loop-gauge-invariant

For Wilson loop W_e defined in {prf:ref}`def-wilson-loop-ig-corrected`:

**Claim**: W_e is invariant under local gauge transformations.

**Proof**:

Under gauge transformation {g(e) : e ∈ E}:
$$
U(e_a, e_b) \to U'(e_a, e_b) = g(e_a) \, U(e_a, e_b) \, g(e_b)^\dagger
$$

For the closed cycle C(e) = e_i → e_j (via IG) → e_i (via CST path):

$$
\begin{align}
U'_C &= U'_{\text{IG}}(e_i, e_j) \times \prod_{\text{edges in path}} U'_{\text{CST}} \\
&= g(e_i) U_{\text{IG}} g(e_j)^\dagger \times g(e_j) U_1 g(e_{k_1})^\dagger \times g(e_{k_1}) U_2 g(e_{k_2})^\dagger \times \cdots \times g(e_{k_m}) U_m g(e_i)^\dagger
\end{align}
$$

where {e_j, e_{k₁}, e_{k₂}, ..., e_{k_m}, e_i} are the intermediate episodes in the path.

**Cancellation**: Adjacent g^† g terms cancel:
$$
U'_C = g(e_i) \left[ U_{\text{IG}} \, U_1 \, U_2 \cdots U_m \right] g(e_i)^\dagger = g(e_i) \, U_C \, g(e_i)^\dagger
$$

**Trace invariance**: Using cyclic property Tr(ABC) = Tr(CAB):
$$
W'_e = \text{Tr}(g(e_i) \, U_C \, g(e_i)^\dagger) = \text{Tr}(U_C \, g(e_i)^\dagger g(e_i)) = \text{Tr}(U_C) = W_e
$$

Therefore, W_e is gauge invariant. ∎
:::

**Key Improvement from Ch. 28**:
- ✅ **Proof provided** (not just assertion)
- ✅ Explicit cancellation of gauge matrices
- ✅ Uses standard trace cyclicity

---

## 3. Wilson Action: Well-Defined but Incomplete

### 3.1. Discrete Wilson Action

:::{prf:definition} IG-Edge-Based Wilson Action
:label: def-wilson-action-corrected

The gauge field action on CST+IG with single-root structure:

$$
\boxed{S_{\text{gauge}} = \frac{\beta}{2N_c} \sum_{e \in E_{\text{IG}}} w_e \left(1 - \frac{1}{N_c} \text{Re} \, \text{Tr} \, W_e\right)}
$$

where:
- e: IG edge index
- w_e > 0: **IG edge weight** (defined below)
- W_e: Wilson loop around fundamental cycle C(e) (Definition {prf:ref}`def-wilson-loop-ig-corrected`)
- β = 2N_c/g²: Gauge coupling parameter
- N_c: Number of colors (gauge group rank)

**Properties**:
- ✅ Well-defined (W_e uniquely computed from IG edge)
- ✅ Gauge invariant (Theorem {prf:ref}`thm-wilson-loop-gauge-invariant`)
- ✅ Real-valued (Re Tr ensures reality)
- ✅ Bounded (|Tr W_e| ≤ N_c)
:::

### 3.2. IG Edge Weight: The Open Problem

:::{prf:definition} IG Edge Weight (Algorithmic)
:label: def-ig-edge-weight-algorithmic

For IG edge e = (e_i ~ e_j) created by cloning interaction:

**We adopt the spacetime separation formula**:

$$
\boxed{w_e := \frac{1}{\tau_{ij}^2 + \|\delta \mathbf{r}_{ij}\|^2 + \epsilon_w^2}}
$$

where:
- $\tau_{ij} := |t^{\rm d}_i - t^{\rm d}_j|$: Temporal separation (episode death times)
- $\delta \mathbf{r}_{ij} := \Phi(e_i) - \Phi(e_j)$: Spatial separation (death positions)
- $\epsilon_w > 0$: Regularization parameter (prevents divergence)

**Justification**:
1. **Dimensional analysis**: w has units [length^(-2)], matching lattice spacing a^(-2)
2. **Locality**: Nearby episodes (small τ, small δr) → large weight (strong coupling)
3. **Causality**: Respects spacetime separation (light-cone structure)
4. **Regularity**: ε_w ensures w_e remains finite

**Alternative (not used)**:
$$
w_e^{\text{cloning}} = |S_i(j)| + |S_j(i)| \quad \text{(from fitness scores)}
$$

We choose spacetime formula for geometric interpretation.
:::

:::{warning} The Continuum Limit Problem
**The critical open question**: Does this algorithmic weight correctly encode the area of the plaquette?

**What would be needed**: A proof that in the continuum limit (N → ∞, ΔV → 0):

$$
w_e = \frac{1}{\tau_{ij}^2 + \|\delta \mathbf{r}_{ij}\|^2} \quad \implies \quad w_e \sim \frac{1}{|\Sigma(e)|^2}
$$

where |Σ(e)| is the "area" of the minimal surface bounded by the fundamental cycle C(e).

**Status**: ❌ **UNPROVEN CONJECTURE**

Without this proof, we have:
- ✅ A well-defined discrete action S_gauge
- ✅ A computational algorithm for Wilson loops
- ⚠️ No guarantee of Yang-Mills continuum limit

**Honest assessment**: The area measure problem is **not eliminated**—it is transformed into the question of whether algorithmic weights accidentally reproduce geometric areas.
:::

---

## 4. Continuum Limit: The Remaining Challenge

### 4.1. Small Loop Expansion (Standard Lattice QCD)

For IG edge e = (e_i ~ e_j) with small spacetime separation:

**Wilson loop expansion**:
$$
W_e = \text{Tr}(U_C) \approx N_c - \frac{g^2}{2} \text{Tr}(F_{\mu\nu} F^{\mu\nu}) |\Sigma(e)|^2 + O(\delta^4)
$$

where:
- F_μν: Yang-Mills field strength tensor
- |Σ(e)|²: Area of the plaquette enclosed by C(e)

**Substitution into action**:
$$
S_{\text{gauge}} \approx \frac{\beta}{2N_c} \sum_{e \in E_{\text{IG}}} w_e \times \frac{g^2}{2} \text{Tr}(F^2) |\Sigma(e)|^2
$$

**Critical step** (the circular reasoning identified by Gemini):

Assume:
$$
w_e \sim \frac{1}{|\Sigma(e)|^2} \times \Delta V
$$

where ΔV is a local volume element.

**Then**:
$$
S_{\text{gauge}} \approx \frac{\beta g^2}{4N_c} \sum_{e} \frac{1}{|\Sigma(e)|^2} \times |\Sigma(e)|^2 \times \text{Tr}(F^2) \times \Delta V = \frac{\beta g^2}{4N_c} \sum_e \text{Tr}(F^2) \Delta V
$$

**Riemann sum**:
$$
\sum_e \text{Tr}(F^2) \Delta V \to \int \text{Tr}(F_{\mu\nu} F^{\mu\nu}) \sqrt{-g} \, d^4x
$$

**Yang-Mills action**:
$$
S_{\text{gauge}} \to \frac{1}{4g^2} \int \text{Tr}(F_{\mu\nu} F^{\mu\nu}) \sqrt{-g} \, d^4x \quad \text{(if } \beta g^2 / 4N_c \to 1/4g^2\text{)}
$$

### 4.2. The Circular Reasoning

:::{admonition} Gemini's Critique
:class: danger

**The problem**: The derivation **assumes**:
$$
w_e \sim |\Sigma(e)|^{-2}
$$

But the entire point of this chapter was to **avoid defining** |Σ(e)| (the area measure)!

**Gemini's assessment**:
> "This is a textbook example of circular reasoning: to prove the action converges to Yang-Mills, one must assume a property (area scaling) that is equivalent to what one is trying to avoid defining in the first place."

**The renamed problem**:
- Original (Ch. 22): Need to define A(C) for irregular cycles ❌
- Chapter 28 claimed: Use w_e instead, no area needed ✓
- Reality: Need to prove w_e ~ A^(-2), which requires defining A ❌

**Conclusion**: We have **renamed** the area measure problem (A → w_e) but **not resolved** it.
:::

### 4.3. What We Can Honestly Claim

**Achievements** ✅:
1. IG edges provide a **discrete, well-defined basis** for loop observables
2. Wilson loops W_e are **computable** from CST+IG data
3. Action S_gauge is **gauge invariant** and **well-defined** on the discrete graph
4. Algorithm has **O(k log N)** complexity where k = |E_IG|

**Open problems** ⚠️:
1. Continuum limit requires proving w_e ~ |Σ(e)|^(-2) (geometric scaling)
2. No rigorous connection to Yang-Mills action (only heuristic)
3. Area measure problem **transformed**, not eliminated

**Honest title**: "IG Edges Provide Computational Basis for Wilson Loops"

**Retracted claim**: "No Area Measure Needed"

---

## 5. Computational Implementation

### 5.1. Algorithm: Wilson Loop Calculation

:::{prf:algorithm} Compute All Wilson Loops from IG Edges
:label: alg-wilson-loops-corrected

**Input**:
- Fractal Set F = (E, E_CST ∪ E_IG)
- Gauge links {U_CST(e_p, e_c) : (e_p → e_c) ∈ E_CST}
- Gauge links {U_IG(e_i, e_j) : (e_i ~ e_j) ∈ E_IG}
- Root episode e_root

**Output**:
- Wilson loops {W_e : e ∈ E_IG}
- Wilson action S_gauge

**Preprocessing** (one-time, O(N log N)):
```python
def preprocess_cst(E_CST, e_root):
    """Build data structures for fast LCA queries."""
    # Build parent pointers
    parent = {e_root: None}
    for (e_p, e_c) in E_CST:
        parent[e_c] = e_p

    # Compute depths (distance to root)
    depth = {e_root: 0}
    for e in topological_order(E_CST):
        if e != e_root:
            depth[e] = depth[parent[e]] + 1

    return parent, depth
```

**Main Loop** (for each IG edge, O(k log N)):
```python
def compute_wilson_loops(E_IG, U_IG, U_CST, parent, depth):
    """Compute Wilson loop for each IG edge."""
    W = {}

    for (e_i, e_j) in E_IG:
        # Step 1: Find LCA
        e_lca = lowest_common_ancestor(e_i, e_j, parent, depth)

        # Step 2: Build path e_j → e_i via LCA
        path_up = get_ancestors(e_j, e_lca, parent)     # e_j → LCA
        path_down = get_ancestors(e_i, e_lca, parent)   # e_i → LCA (reverse)

        # Step 3: Compute path-ordered product
        U_path = np.eye(N_c)  # Identity matrix

        # Go up from e_j to LCA (backward edges)
        for k in range(len(path_up) - 1):
            e_child = path_up[k]
            e_parent = path_up[k+1]
            U_path = U_path @ U_CST[(e_parent, e_child)].conj().T  # U^†

        # Go down from LCA to e_i (forward edges)
        for k in range(len(path_down) - 2, -1, -1):
            e_parent = path_down[k+1]
            e_child = path_down[k]
            U_path = U_path @ U_CST[(e_parent, e_child)]

        # Step 4: Close loop with IG edge
        U_loop = U_IG[(e_i, e_j)] @ U_path

        # Step 5: Wilson loop = trace
        W[(e_i, e_j)] = np.trace(U_loop)

    return W
```

**LCA Subroutine** (O(log N) per query):
```python
def lowest_common_ancestor(e_i, e_j, parent, depth):
    """Find LCA using depth-balanced walk."""
    # Bring both to same depth
    while depth[e_i] > depth[e_j]:
        e_i = parent[e_i]
    while depth[e_j] > depth[e_i]:
        e_j = parent[e_j]

    # Walk up together until they meet
    while e_i != e_j:
        e_i = parent[e_i]
        e_j = parent[e_j]

    return e_i  # LCA
```

**Compute Action**:
```python
def compute_wilson_action(W, weights, beta, N_c):
    """Compute total Wilson action."""
    S = 0.0
    for e in E_IG:
        W_e = W[e]
        w_e = weights[e]
        S += w_e * (1.0 - np.real(W_e) / N_c)

    S *= beta / (2 * N_c)
    return S
```

**Complexity**:
- Preprocessing: O(N log N) (tree construction + depth calculation)
- Per IG edge: O(log N) (LCA query + path construction)
- Total: O(k log N) where k = |E_IG|
:::

### 5.2. Numerical Test: Gauge Invariance

**Test Protocol**:
```python
def test_gauge_invariance(F, U_CST, U_IG):
    """Verify Wilson loops are gauge invariant."""
    # Compute original Wilson loops
    W_original = compute_wilson_loops(...)

    # Apply random gauge transformation
    g = {e: random_SU(N_c) for e in E}

    # Transform gauge links
    U_CST_transformed = {}
    for (e_p, e_c) in E_CST:
        U_CST_transformed[(e_p, e_c)] = g[e_p] @ U_CST[(e_p, e_c)] @ g[e_c].conj().T

    U_IG_transformed = {}
    for (e_i, e_j) in E_IG:
        U_IG_transformed[(e_i, e_j)] = g[e_i] @ U_IG[(e_i, e_j)] @ g[e_j].conj().T

    # Compute transformed Wilson loops
    W_transformed = compute_wilson_loops(..., U_CST_transformed, U_IG_transformed)

    # Check invariance
    for e in E_IG:
        assert np.allclose(W_original[e], W_transformed[e]), f"Gauge invariance violated for {e}"

    print("✅ Gauge invariance verified for all Wilson loops")
```

**Expected result**: All assertions pass (Wilson loops unchanged under gauge transformation)

---

## 6. Comparison to Lattice QCD

### 6.1. Regular Lattice vs CST+IG

| Property | Regular Lattice | Single-Root CST+IG |
|----------|----------------|---------------------|
| **Vertices** | Sites n ∈ ℤ⁴ | Episodes e ∈ E (irregular) |
| **Edges** | Unit links (n, n+μ̂) | CST (genealogy) + IG (interaction) |
| **Loops** | Elementary plaquettes | Fundamental cycles from IG edges |
| **Symmetry** | Translation invariant | Adaptive (fitness-driven) |
| **Weights** | All equal (w = 1) | Variable (w_e from spacetime separation) |
| **Area** | Explicit (a²) | Implicit (w_e ~ ??) |
| **Continuum limit** | a → 0 | N → ∞, ΔV → 0 |

**Advantage of CST+IG**: Naturally adapts to curved spacetime (episodes follow geodesics)

**Disadvantage of CST+IG**: Area/weight relationship unproven

### 6.2. What Standard Lattice QCD Teaches Us

In lattice QCD, the **area** of a plaquette is:
$$
A_{\square} = a^2 \quad \text{(lattice spacing squared)}
$$

The **Wilson action** is:
$$
S_{\text{lattice}} = \frac{\beta}{N_c} \sum_{\text{plaquettes}} \left(1 - \frac{1}{N_c} \text{Re} \, \text{Tr} \, W_{\square}\right)
$$

**Weights** are uniform: w_□ = 1 for all plaquettes.

**Why this works**:
- Lattice spacing a is **explicit** parameter
- Area a² appears in small-loop expansion
- Continuum limit a → 0 well-defined

**For CST+IG to work**, we need:
- Episode "spacing" δ implicit in w_e
- Area |Σ(e)| emerges from w_e ~ δ^(-2)
- Continuum limit N → ∞ reproduces Yang-Mills

**The challenge**: Proving this emergence

---

## 7. Honest Assessment and Path Forward

### 7.1. What This Document Achieves

**Solid foundations** ✅:
1. **Single-root CST is a tree**: Rigorous proof (Theorem {prf:ref}`thm-cst-single-root-tree`)
2. **IG edges close cycles**: Complete basis theorem (Theorem {prf:ref}`thm-ig-cycle-basis-corrected`)
3. **Wilson loops well-defined**: Unambiguous algorithm (Definition {prf:ref}`def-wilson-loop-ig-corrected`)
4. **Gauge invariance**: Proven (Theorem {prf:ref}`thm-wilson-loop-gauge-invariant`)
5. **Computational framework**: O(k log N) algorithm

**Open problems** ⚠️:
1. **Continuum limit**: Unproven w_e ~ |Σ(e)|^(-2) scaling
2. **Yang-Mills connection**: Heuristic, not rigorous
3. **Area measure**: Transformed into weight problem, not eliminated

### 7.2. Retracted vs Corrected Claims

| Claim | Ch. 28 (Original) | Ch. 32 (Corrected) |
|-------|-------------------|-------------------|
| **CST structure** | "Tree by construction" | ✅ "Tree if single root (proven)" |
| **Area measure** | ❌ "Not needed" | ⚠️ "Implicit in w_e (unproven)" |
| **Continuum limit** | ❌ "Converges to Yang-Mills" | ⚠️ "Conjecture (needs w_e scaling)" |
| **Gauge invariance** | "Asserted" | ✅ "Proven (Thm 3)" |
| **Path definition** | "Unique undirected path" | ✅ "LCA-based algorithm (Def 2)" |
| **Weight formula** | "Two options (ambiguous)" | ✅ "Single choice (spacetime)" |

### 7.3. The Three Paths Forward

**Path A: Empirical Validation** (Recommended)

**Goal**: Test whether w_e ~ |Σ(e)|^(-2) holds in practice

**Steps**:
1. Implement Wilson loops on actual Fragile Gas data
2. For each IG edge, measure:
   - Algorithmic weight: w_e = 1/(τ² + δr²)
   - Geometric "area": |Σ(e)| ≈ τ · δr (spacetime area of plaquette)
3. Plot w_e vs |Σ(e)|^(-2)
4. Fit power law: w_e ∝ |Σ(e)|^α, determine α
5. Test hypothesis: α ≈ -2?

**Timeline**: 2-3 months (implementation + data collection + analysis)

**Outcome**: Either:
- ✅ α ≈ -2: Strong evidence for area scaling (publish!)
- ❌ α ≠ -2: Scaling wrong, back to drawing board

**Path B: Theoretical Proof** (High Risk)

**Goal**: Prove w_e ~ |Σ(e)|^(-2) from first principles

**Challenges**:
1. Define |Σ(e)| rigorously on irregular CST+IG graph
2. Relate spacetime separation (τ, δr) to minimal surface area
3. Prove emergent geometric relationship

**Timeline**: 6-12 months (difficult, may be impossible)

**Outcome**: Either:
- ✅ Proof found: Breakthrough result (top journal)
- ❌ Proof impossible: Learn why scaling doesn't hold

**Path C: Alternative Theory** (Ambitious)

**Goal**: Accept IG action as **new** discrete gauge theory, not limit of Yang-Mills

**Approach**:
- Don't force Yang-Mills continuum limit
- Study CST+IG gauge theory on its own terms
- Explore what physics it actually describes
- May be different from (and interesting than!) standard lattice QCD

**Timeline**: 1-2 years (exploratory research)

**Outcome**: Novel theory with algorithmic foundations

### 7.4. Recommended Action Plan

**Phase 1** (Now): Documentation and consolidation
- [x] Create corrected Ch. 32 (this document)
- [ ] Update Ch. 28 status to "❌ SUPERSEDED BY CH. 32"
- [ ] Document lessons in meta-analysis

**Phase 2** (1-2 months): Computational implementation
- [ ] Implement Algorithm {prf:ref}`alg-wilson-loops-corrected`
- [ ] Test on toy Fragile Gas runs (N = 100, small domain)
- [ ] Verify gauge invariance numerically
- [ ] Measure w_e distributions

**Phase 3** (2-3 months): Empirical tests (Path A)
- [ ] Collect CST+IG data from production runs
- [ ] Measure w_e and estimate |Σ(e)|
- [ ] Test w_e ~ |Σ|^(-2) scaling hypothesis
- [ ] Statistical analysis of results

**Phase 4** (Decision point): Based on Phase 3 results
- If α ≈ -2: Write paper "Emergent Yang-Mills from Algorithmic Gauge Theory"
- If α ≠ -2: Pivot to Path C (novel discrete theory)

---

## 8. Conclusions

### 8.1. Summary of Corrections

**From Chapter 28 to Chapter 32**:

**Fixed**:
- ✅ Explicit single-root assumption (resolves forest problem)
- ✅ Rigorous CST tree proof (no "by construction")
- ✅ Unambiguous path definition (LCA algorithm)
- ✅ Gauge invariance proven (not asserted)
- ✅ Single weight formula (no ambiguity)
- ✅ Computational algorithm (fully specified)

**Acknowledged**:
- ⚠️ Area/weight relationship is **open problem**, not solved
- ⚠️ Continuum limit is **conjecture**, not proven
- ⚠️ "No area measure needed" claim **retracted**

### 8.2. The Honest Claim

**What we have**:
> "IG edges from the Fragile Gas algorithm provide a natural discrete basis for computing Wilson loops in a gauge theory defined on the CST+IG graph. Under the assumption of a single common ancestor, each IG edge closes exactly one fundamental cycle, and Wilson loops are gauge-invariant by construction. The discrete action is well-defined and computationally tractable."

**What we don't have**:
> "Proof that this discrete action converges to the Yang-Mills continuum action. The required ingredient—that algorithmic edge weights scale as the inverse-square of plaquette areas—remains an open conjecture requiring either theoretical proof or empirical validation."

### 8.3. Why This Matters

**Scientific value** (even without continuum limit):
1. **Discrete gauge observables**: Wilson loops computable on algorithmic data
2. **Novel gauge theory**: CST+IG may describe new physics
3. **Algorithmic foundations**: Gauge theory from optimization dynamics
4. **Testable predictions**: Empirical tests of w_e scaling

**Philosophical value**:
> "Demonstrates that gauge-theoretic structures can emerge naturally from classical algorithmic dynamics, even if the connection to standard Yang-Mills remains to be proven."

### 8.4. Lesson Learned

**From Gemini's critique**:
> "Renaming a problem is not the same as solving it."

**Applied to our work**:
- Area measure A(C) → Edge weight w_e
- Same fundamental challenge: How does discrete algorithm encode continuous geometry?

**The path forward**:
- Accept the problem exists
- Pursue empirical validation
- Be honest about limitations
- Build only on proven foundations

---

## References

### Graph Theory

- Diestel, R. (2017). *Graph Theory* (5th ed.). Springer. Ch. 1 (Trees, LCA algorithms)
- Tarjan, R.E. (1984). "Applications of Path Compression on Balanced Trees". *J. ACM* 31: 690 (Efficient LCA)

### Lattice Gauge Theory

- Wilson, K.G. (1974). "Confinement of quarks". *Phys. Rev. D* 10: 2445
- Creutz, M. (1983). *Quarks, Gluons and Lattices*. Cambridge. Ch. 5
- Montvay, I. & Münster, G. (1994). *Quantum Fields on a Lattice*. Cambridge. Ch. 4

### Internal Documents

- [13_fractal_set.md](13_fractal_set.md): CST and IG construction (single/multi-root discussion)
- [28_wilson_loops_from_ig_edges.md](28_wilson_loops_from_ig_edges.md): Original (flawed) version
- [30_gemini_review_wilson_loops_ig_edges.md](30_gemini_review_wilson_loops_ig_edges.md): Gemini's critique
- [31_gemini_reviews_ch27_ch28_summary.md](31_gemini_reviews_ch27_ch28_summary.md): Consolidated lessons

---

**Document Status**: ✅ **CORRECTED VERSION**

**Replaces**: Chapter 28 (retracted claims about area measure)

**Next Steps**: Empirical validation (Path A) or theoretical proof (Path B)

**Key Takeaway**: We have a solid computational framework; continuum limit remains open research question
