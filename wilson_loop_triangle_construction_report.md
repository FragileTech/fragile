# Wilson Loop Triangle Construction in the Fractal Set: IA vs IG Edges

**Report Date:** 2026-01-16
**Document Version:** 1.0
**Primary Sources:**
- `docs/source/3_fractal_gas/2_fractal_set/01_fractal_set.md` (Fractal Set definition)
- `docs/source/3_fractal_gas/2_fractal_set/03_lattice_qft.md` (Lattice QFT structure)
- SVG diagrams in `docs/svg_images/` and `docs/source/3_fractal_gas/2_fractal_set/figures/`

---

## 1. Executive Summary

**Confirmation:** The diagrams are correct—**IA (Influence Attribution) edges close the Wilson loop**, not IG (Information Geometry) edges.

**Triangle Structure:** Each interaction triangle $\triangle_{ij,t}$ consists of three edge types:
- **IG edge** (spacelike): connects contemporaneous walkers $(n_{i,t}, n_{j,t})$ — "walker $j$ influences walker $i$"
- **CST edge** (timelike): connects the same walker across time $(n_{i,t}, n_{i,t+1})$ — "walker $i$ evolves"
- **IA edge** (diagonal/retrocausal): connects effect to cause $(n_{i,t+1}, n_{j,t})$ — "attribute $i$'s evolution to $j$"

**Key Insight:** IA edges are **retrocausal** in the sense of causal *attribution*, not physical time travel. They point from the effect (walker $i$ at time $t+1$) backward to the cause (walker $j$ at time $t$), closing the causal loop that represents one complete interaction event.

**Difference from Genealogy Intuition:** The initial expectation that triangles might connect two walkers sharing a common ancestor (via CST edges) with an IG edge closing the loop conflates *genealogical relationships* (who cloned from whom) with *interaction triangles* (who influenced whose evolution). These are distinct concepts:
- **Genealogy** is tracked in node attributes (clone sources)
- **Triangles** track dynamic interactions and influence

**Mathematical Necessity:** CST and IG edges alone cannot form closed loops because:
- IG edges connect *different walkers* at the *same time* (spacelike)
- CST edges connect the *same walker* at *different times* (timelike)
- No combination of these can close a 3-cycle without a diagonal edge

The IA edge provides the necessary diagonal connection, making triangles the minimal closed loops in the Fractal Set's causal structure.

---

## 2. What the Documentation Says

### 2.1 The Three Edge Types

From `01_fractal_set.md` (lines 49-58):

| Edge Type | Connects | Encodes | Directionality |
|-----------|----------|---------|----------------|
| **CST** (Causal Spacetime Tree) | $(n_{i,t}, n_{i,t+1})$ | Temporal evolution of single walker | Directed (time's arrow) |
| **IG** (Information Graph) | $(n_{i,t}, n_{j,t})$ | Spatial coupling between contemporaneous walkers | Directed (selection asymmetry) |
| **IA** (Influence Attribution) | $(n_{i,t+1}, n_{j,t})$ | Causal attribution from effect to cause | Directed (retrocausal) |

### 2.2 Triangle Definition

From `01_fractal_set.md:969-997` ({prf:ref}`def-fractal-set-triangle`):

An **interaction triangle** $\triangle_{ij,t}$ is the 2-simplex with:

**Vertices** (0-faces):
$$V(\triangle_{ij,t}) = \{n_{j,t}, n_{i,t}, n_{i,t+1}\}$$

**Edges** (1-faces), forming the **boundary** $\partial\triangle_{ij,t}$:
- $e_{\mathrm{IG}} = (n_{i,t}, n_{j,t}) \in E_{\mathrm{IG}}$: "walker $j$ influences walker $i$"
- $e_{\mathrm{CST}} = (n_{i,t}, n_{i,t+1}) \in E_{\mathrm{CST}}$: "walker $i$ evolves"
- $e_{\mathrm{IA}} = (n_{i,t+1}, n_{j,t}) \in E_{\mathrm{IA}}$: "attribute $i$'s update to $j$"

Note the IA edge direction: $(n_{i,t+1}, n_{j,t})$ — from the future (effect) to the past (cause).

### 2.3 Wilson Loop Formula

From `01_fractal_set.md:1125-1132` ({prf:ref}`def-fractal-set-wilson-loop`):

$$W(\triangle_{ij,t}) := U_{\mathrm{IG}}(e_{\mathrm{IG}}) \cdot U_{\mathrm{CST}}(e_{\mathrm{CST}}) \cdot U_{\mathrm{IA}}(e_{\mathrm{IA}})^*$$

$$= \exp\left(i(\theta_{ij} + \phi_{\mathrm{CST}} - \phi_{\mathrm{IA}})\right)$$

where:
- $\theta_{ij}$ = fitness phase difference on IG edge
- $\phi_{\mathrm{CST}}$ = phase accumulated during evolution on CST edge
- $\phi_{\mathrm{IA}}$ = attribution phase on IA edge
- The conjugate $U_{\mathrm{IA}}^*$ appears because the IA edge is traversed backward when forming the closed loop

### 2.4 The "Atom of Interaction"

From `01_fractal_set.md:999-1007`:

> "The triangle is the atom of interaction. You cannot break it into smaller pieces without losing causality.
>
> Consider what happens when walker $j$ influences walker $i$: at time $t$, both walkers exist with their own positions and velocities. Walker $i$ "sees" walker $j$ through the IG edge—this is the influence channel. Then $i$ evolves from $t$ to $t+1$ along its CST edge—this is the evolution channel. Finally, the IA edge closes the loop by recording "this evolution was partly due to $j$"—this is the attribution channel.
>
> Without the IG edge, we wouldn't know who influenced whom. Without the CST edge, we wouldn't know what changed. Without the IA edge, we couldn't close the causal loop.
>
> The triangle is irreducible. It's the smallest closed loop in the structure. And it corresponds exactly to the physical process: one walker influencing another's evolution."

---

## 3. Diagram Analysis

### 3.1 Basic Triangle Structure (`fractal_set_triangle.svg`)

This diagram shows the simplest representation:

```
        n_j@t ───────IG─────→ n_i@t
                                │
                               CST
                                │
                                ↓
        n_j@t ←────IA─────── n_i@t+1
```

**Visual elements:**
- **Horizontal IG edge**: connects $(n_{i,t} \to n_{j,t})$ at the same time slice
- **Vertical CST edge**: connects $(n_{i,t} \to n_{i,t+1})$ moving forward in time
- **Diagonal IA edge**: connects $(n_{i,t+1} \to n_{j,t})$ **backward** in time (retrocausal)

The triangle's interior is shaded, indicating it is a 2-simplex (a filled triangle), not just a 1-skeleton (edges only).

### 3.2 Wilson Loop with Parallel Transport (`interaction-triangle-transport.svg`)

This diagram adds the gauge-theoretic interpretation:

```
        n_j,t ──────U_IG────→ n_i,t
          ↑                      │
          │                     │
        U_IA*                 U_CST
          │                     │
          │                     ↓
        n_j,t ←──────────── n_i,t+1
```

**Wilson loop traversal:**
1. Start at $n_{i,t}$
2. Transport via $U_{\mathrm{IG}}$ to $n_{j,t}$ (IG edge)
3. Wait at $n_{j,t}$ (no edge; this is where the diagram shows we're at the "corner")
4. Actually, we need to go from $n_{j,t}$ up to $n_{i,t+1}$ via the CST edge of walker $i$

**Correction:** Let me reread the actual path. The Wilson loop follows the boundary:
1. Start at $n_{j,t}$
2. IG edge: $(n_{j,t} \to n_{i,t})$ with $U_{\mathrm{IG}}$
3. CST edge: $(n_{i,t} \to n_{i,t+1})$ with $U_{\mathrm{CST}}$
4. IA edge *backward*: $(n_{i,t+1} \to n_{j,t})$ with $U_{\mathrm{IA}}^*$ (conjugated because traversed backward)

The label in the diagram shows: `W(triangle) = U_IG U_CST U_IA*`

### 3.3 Plaquette Formation (`triangles-to-plaquette.svg`)

This diagram shows two triangles sharing an IG edge:

```
         n_i,t+1 ──────IA──────→ n_j,t ──────IA──────→ n_i,t+1
             ↑                     ↑  ↓                     ↑
            CST                   CST CST                  CST
             │                     │  │                     │
             │                     │  │                     │
         n_i,t ←─────IG──────→ n_j,t  n_i,t ←─────IG──────→ n_j,t
```

Actually, let me describe this more carefully. The hourglass plaquette has:
- Two time slices: $t$ (bottom) and $t+1$ (top)
- Two walkers: $i$ and $j$
- Four vertices: $n_{i,t}, n_{j,t}, n_{i,t+1}, n_{j,t+1}$

**Triangle 1** ($\triangle_{ij,t}$): vertices $\{n_{j,t}, n_{i,t}, n_{i,t+1}\}$
- IG edge: $(n_{j,t} \to n_{i,t})$
- CST edge: $(n_{i,t} \to n_{i,t+1})$
- IA edge: $(n_{i,t+1} \to n_{j,t})$

**Triangle 2** ($\triangle_{ji,t}$): vertices $\{n_{i,t}, n_{j,t}, n_{j,t+1}\}$
- IG edge: $(n_{i,t} \to n_{j,t})$ ← **shared with Triangle 1 (opposite direction)**
- CST edge: $(n_{j,t} \to n_{j,t+1})$
- IA edge: $(n_{j,t+1} \to n_{i,t})$

The shared IG edge appears with opposite orientations in the two triangles, so it **cancels** when computing the boundary $\partial P_{ij,t}$. The result is a 4-cycle ("hourglass") consisting of:
- CST for walker $i$: $(n_{i,t} \to n_{i,t+1})$
- IA diagonal: $(n_{i,t+1} \to n_{j,t})$
- CST for walker $j$: $(n_{j,t} \to n_{j,t+1})$
- IA diagonal: $(n_{j,t+1} \to n_{i,t})$

The diagram confirms: **IA edges close both the atomic 3-cycles (triangles) and appear in the plaquette boundaries**.

---

## 4. Why IA Edges Must Close the Loop

### 4.1 Topological Necessity: Why CST+IG Cannot Form Closed Loops

Let's attempt to construct a closed loop using only CST and IG edges and see where it fails.

**Attempt 1: Start with an IG edge**

1. Start at vertex $n_{i,t}$ (walker $i$ at time $t$)
2. Take an IG edge to $n_{j,t}$ (different walker, same time): $(n_{i,t} \to n_{j,t})$
3. Take a CST edge to $n_{j,t+1}$ (same walker, next time): $(n_{j,t} \to n_{j,t+1})$
4. Now we're at $n_{j,t+1}$. How do we get back to $n_{i,t}$?

**Problem:** We need to:
- Get from walker $j$ to walker $i$: requires an IG edge, but we're at time $t+1$
- Get from time $t+1$ to time $t$: requires going backward in time

**Option A:** Take an IG edge at time $t+1$
- This gives $(n_{j,t+1} \to n_{i,t+1})$
- Now we're at $n_{i,t+1}$, but we need to get to $n_{i,t}$

**Option B:** Go backward in time via CST
- This would be $(n_{j,t+1} \to n_{j,t})$, but CST edges are **directed forward** in time
- Backward time travel violates causality

**Conclusion:** We're stuck. There's no way to close the loop using only CST and IG edges.

**Attempt 2: Try a different path**

1. Start at $n_{i,t}$
2. CST to $n_{i,t+1}$: $(n_{i,t} \to n_{i,t+1})$
3. IG to $n_{j,t+1}$: $(n_{i,t+1} \to n_{j,t+1})$
4. CST backward to $n_{j,t}$? **Impossible** (CST is forward-only)
5. IG to ... where? We're at time $t+1$, but our starting point $n_{i,t}$ is at time $t$

**Problem:** IG edges connect walkers at the **same time**. Once we've moved forward in time (via CST), we cannot use IG to connect back to vertices at the previous time.

**The fundamental issue:**
- **IG edges** connect different walkers at the **same** time (spacelike, horizontal)
- **CST edges** connect the same walker at **different** times (timelike, vertical)
- To form a closed loop, we need to connect **different walkers at different times**
- This requires a **diagonal edge**: precisely what IA provides

**The IA edge solution:**
- IA edge: $(n_{i,t+1} \to n_{j,t})$ connects walker $i$ at time $t+1$ to walker $j$ at time $t$
- This is the **diagonal** that closes the triangle
- It's the only edge type that connects different walkers at different times

### 4.2 Causal Attribution Structure

The IA edge is not an arbitrary diagonal connection—it has deep causal meaning.

**The complete interaction story:**
1. **Influence (IG edge):** At time $t$, walker $j$'s state influences walker $i$ through viscous coupling, cloning potential, or companion selection
2. **Evolution (CST edge):** Walker $i$ evolves from time $t$ to $t+1$, integrating forces and updating its state
3. **Attribution (IA edge):** Looking backward from $i$'s state at $t+1$, we attribute part of this evolution to $j$'s influence at time $t$

The IA edge represents **retrocausal attribution**:
- Not physical time travel or backward causation
- But the logical inference: "Given that $i$ is now at state $s_{i,t+1}$, which past influences were responsible?"
- This is standard in causal inference, counterfactual reasoning, and influence analysis

**From `01_fractal_set.md:1144-1161`:**

> "What does the Wilson loop around a triangle measure? It measures the **quantum phase accumulated during one complete interaction**.
>
> The Wilson loop combines three phases from the three edges of the triangle boundary:
> - **IG phase** $\theta_{ij}$: the fitness phase difference between walkers $i$ and $j$
> - **CST phase** $\phi_{\mathrm{CST}}$: the phase accumulated during $i$'s evolution
> - **IA phase** $-\phi_{\mathrm{IA}}$: the attribution phase (conjugated due to edge orientation)
>
> The total holonomy $\Phi(\triangle) = \theta_{ij} + \phi_{\mathrm{CST}} - \phi_{\mathrm{IA}}$ is the **interaction phase**. [...]
>
> A **flat interaction** ($\Phi(\triangle) = 0$) means the three phases balance perfectly: the "cost" of influence ($\theta_{ij}$) plus the "cost" of evolution ($\phi_{\mathrm{CST}}$) exactly equals the "cost" of attribution ($\phi_{\mathrm{IA}}$). The interaction was *self-consistent*—what $j$ put in, $i$ got out, with no residue.
>
> A **curved interaction** ($\Phi(\triangle) \neq 0$) means something was left over. [...] The nonzero holonomy is a signature that *something happened* in this interaction beyond the simple story of '$j$ influenced $i$'."

The IA edge completes the causal loop: **influence → evolution → attribution** forms an irreducible unit.

### 4.3 Gauge Theory Perspective

In lattice gauge theory, Wilson loops are the fundamental gauge-invariant observables that measure the holonomy (parallel transport) around closed paths.

**Standard lattice QFT (hypercubic lattice):**
- Lattice edges aligned with spacetime coordinates
- Minimal closed loops are **4-cycles** (plaquettes): $\square_{x\mu\nu}$ in the $\mu$-$\nu$ plane
- These measure the field strength $F_{\mu\nu}$

**Fractal Set (three edge types):**
- Edge types have distinct causal roles (spacelike, timelike, diagonal)
- Minimal closed loops are **3-cycles** (triangles): one edge of each type
- These measure the interaction phase $\Phi(\triangle)$

**Key insight:** The minimal Wilson loop must include **all three edge types** to be gauge-invariant and represent a complete interaction.

From `03_lattice_qft.md:1164-1171`:

> "**Why triangles instead of plaquettes?** In standard lattice gauge theory on a hypercubic lattice, the minimal closed loops are 4-cycles (plaquettes) because the lattice has only coordinate-aligned edges. The Fractal Set has a richer structure: three edge types with different causal roles. The minimal closed loop that involves all three edge types is a 3-cycle (triangle), not a 4-cycle.
>
> This is not a choice—it is forced by the causal structure. An IG edge (spacelike) connects two walkers at the *same* time. A CST edge (timelike) connects the *same* walker at different times. An IA edge (diagonal) connects a walker's future to another walker's past. The smallest loop that uses one of each is a triangle."

**Plaquettes as composite structures:**
- Plaquettes in the Fractal Set are **pairs of triangles** sharing an IG edge
- Wilson loop on plaquette = product of Wilson loops on the two triangles
- From `01_fractal_set.md:1134-1142`: $W(P_{ij,t}) = W(\triangle_{ij,t}) \cdot W(\triangle_{ji,t})^*$

The triangular structure is **primary**; plaquettes are derived.

---

## 5. Alternative Construction: Why Not CST+IG Only?

### 5.1 Impossibility of CST+IG Closure (Detailed Walk-Through)

We've seen that CST+IG edges cannot form closed loops. Let's be more systematic.

**Given:**
- CST edges: $(n_{i,t}, n_{i,t+1})$ for fixed $i$, varying $t$
- IG edges: $(n_{i,t}, n_{j,t})$ for $i \neq j$, fixed $t$

**Question:** Can any sequence of these edges form a closed loop (a cycle that returns to its starting vertex)?

**Path construction:**

A path alternating between CST and IG edges might look like:
```
n_{i_0,t_0} --IG--> n_{i_1,t_0} --CST--> n_{i_1,t_1} --IG--> n_{i_2,t_1} --CST--> n_{i_2,t_2} --> ...
```

Each CST edge increases time: $t_0 < t_1 < t_2 < \ldots$

**Closure requirement:** To close the loop, we need to return to $n_{i_0,t_0}$.

**Problem 1:** Time is strictly increasing.
- After $k$ CST edges, we're at time $t_0 + k$
- IG edges don't change time
- We can never get back to $t_0$ (unless we allow backward CST, which violates causality)

**Problem 2:** Even if we tried to close the loop at a later time slice by returning to walker $i_0$:
- After CST and IG steps, suppose we return to walker $i_0$ at time $t_k > t_0$
- We'd be at $n_{i_0,t_k}$, not $n_{i_0,t_0}$
- This is a different vertex—the loop is not closed

**Conclusion:** No finite sequence of CST and IG edges can form a closed loop. The fundamental reason is that **CST changes time monotonically** and **IG preserves time**, so any path involving CST cannot return to its starting point in spacetime.

### 5.2 Genealogy vs. Interaction: Distinct Concepts

**Initial intuition (genealogy-based):**
- Two walkers that cloned have a common ancestor
- Their CST edges trace back to this ancestor
- An IG edge between them would "close the triangle"

**Why this intuition is natural:**
- Cloning creates a genealogical tree structure
- Trees have branches and a root
- It seems like you could form triangles by connecting branches

**Why this doesn't match the Fractal Set structure:**

1. **CST edges are not genealogical links**
   - CST edges connect a walker to its temporal successor: $(n_{i,t}, n_{i,t+1})$
   - They form a **forest** (one tree per walker ID)
   - Cloning does NOT create CST edges between walker IDs

2. **Genealogy is stored in node attributes**
   - From `01_fractal_set.md:367`: `c(n)` = clone source (walker ID that walker $i$ cloned from)
   - This is metadata on the node, not an edge
   - The genealogical tree can be reconstructed from clone source pointers

3. **IG edges are not genealogy-specific**
   - IG edges connect **all** pairs of alive walkers at each timestep
   - From `01_fractal_set.md:75-76`: "Directed edges $(n_{i,t}, n_{j,t})$ connecting all ordered pairs of distinct alive walkers at the same timestep"
   - There are $k_t(k_t-1)$ IG edges at timestep $t$ (complete directed graph)
   - IG edges represent *potential influence* (companion selection, viscous coupling), not genealogy

**The fundamental distinction:**
- **Cloning genealogy:** "Walker $i$ was created by copying walker $j$ at time $t$"
  - Stored in: Node attribute `c(n_{i,t}) = j`
  - Purpose: Track ancestry, understand population dynamics

- **Interaction triangles:** "Walker $j$ influenced walker $i$'s evolution from $t$ to $t+1$"
  - Stored in: Triangle $\triangle_{ij,t}$ with IG, CST, IA edges
  - Purpose: Track dynamic interactions, compute Wilson loops, measure interaction phases

These are **complementary** structures serving different purposes.

### 5.3 What About IG Edges Between Clones?

**Question:** If walker $i$ cloned from walker $j$ at time $t$, is there an IG edge between them?

**Answer:** Yes, but not because of the cloning relationship.

At time $t$, there are IG edges between **all** pairs of alive walkers, including:
- $(n_{i,t}, n_{j,t})$ — walker $i$ to walker $j$
- $(n_{j,t}, n_{i,t})$ — walker $j$ to walker $i$

These IG edges exist regardless of whether $i$ cloned from $j$. They represent the spatial coupling / companion selection mechanism.

**The cloning event:**
- Recorded in the node attribute: `c(n_{i,t+1}) = j` (if $i$ clones from $j$ at step $t \to t+1$)
- **Does not create a special edge**
- The influence of cloning appears in the **IA edge weight**: the IA edge $(n_{i,t+1}, n_{j,t})$ has $\chi_{\mathrm{clone}}(e) = 1$ and $w_{\mathrm{IA}}(e) = 1$ if cloning occurred

**Summary:** IG edges form a dense interaction graph (all pairs at each time). Cloning is a special event that increases the weight of specific IA edges, but it doesn't change the fundamental triangle structure.

---

## 6. Physical/Algorithmic Interpretation

### 6.1 What Each Edge Represents in the Algorithm

**IG edge:** $(n_{i,t}, n_{j,t})$ — "Walker $i$ considers walker $j$ as a companion"
- **Algorithmically:** Walker $i$ samples walker $j$ from the companion distribution (soft phase-space distance)
- **Physically:** Spatial coupling in the information geometry
- **Data stored:**
  - Spinor-encoded positions $\psi_{x_i}$, $\psi_{x_j}$
  - Phase potential $\theta_{ij} = -(\Phi_j - \Phi_i)/\hbar_{\text{eff}}$ (fitness phase difference)
  - Viscous coupling weight, distance metrics

**CST edge:** $(n_{i,t}, n_{i,t+1})$ — "Walker $i$ integrates forces and evolves"
- **Algorithmically:** Boris-BAOAB integration: kick, drift, thermostat, drift
- **Physically:** Temporal evolution along the walker's worldline
- **Data stored:**
  - Spinor-encoded velocity $\psi_v$, displacement $\psi_{\Delta x}$
  - Force spinors: $\psi_{\mathbf{F}_{\mathrm{stable}}}$, $\psi_{\mathbf{F}_{\mathrm{adapt}}}$, $\psi_{\mathbf{F}_{\mathrm{viscous}}}$
  - Phase $\phi_{\mathrm{CST}}$ accumulated during evolution
  - Diffusion tensor $\Sigma_{\mathrm{reg}}$, noise realization $\psi_{\mathrm{noise}}$

**IA edge:** $(n_{i,t+1}, n_{j,t})$ — "Attribute walker $i$'s update to walker $j$'s influence"
- **Algorithmically:** Post-hoc influence attribution based on companion selection and interaction strength
- **Physically:** Retrocausal attribution connecting effect to cause
- **Data stored:**
  - Influence weight $w_{\mathrm{IA}}(e) \in [0,1]$ (fraction of $i$'s update due to $j$)
  - Clone indicator $\chi_{\mathrm{clone}}(e) \in \{0,1\}$ (1 if $i$ cloned from $j$)
  - Phase contribution $\phi_{\mathrm{IA}}$

### 6.2 Wilson Loop Phase Meaning

The total holonomy around a triangle:

$$\Phi(\triangle_{ij,t}) = \theta_{ij} + \phi_{\mathrm{CST}} - \phi_{\mathrm{IA}}$$

**Interpretation of each term:**

1. **$\theta_{ij} = -(\Phi_j - \Phi_i)/\hbar_{\text{eff}}$** (IG phase)
   - Fitness phase difference between walkers
   - Positive if $j$ is fitter than $i$ (uphill influence)
   - Negative if $i$ is fitter than $j$ (downhill influence)

2. **$\phi_{\mathrm{CST}}$** (CST phase)
   - Phase accumulated during walker $i$'s evolution
   - Related to the action integral: $\phi_{\mathrm{CST}} = \int_{t}^{t+1} L \, dt$ (with appropriate normalization)
   - Includes contributions from kinetic energy, potential energy, and fitness potential

3. **$-\phi_{\mathrm{IA}}$** (IA phase, conjugated)
   - Attribution phase (negative because IA edge traversed backward)
   - Measures the "cost" of attributing $i$'s change to $j$'s influence

**Flat vs. Curved Interactions:**

**Flat interaction:** $\Phi(\triangle) = 0$
- The three phases exactly balance: $\theta_{ij} + \phi_{\mathrm{CST}} = \phi_{\mathrm{IA}}$
- **Meaning:** The influence from $j$ ($\theta_{ij}$) plus $i$'s evolution ($\phi_{\mathrm{CST}}$) perfectly matches what we attribute to $j$ ($\phi_{\mathrm{IA}}$)
- The interaction is **self-consistent** — no residual phase
- This is the "expected" or "trivial" case in a smooth fitness landscape

**Curved interaction:** $\Phi(\triangle) \neq 0$
- Nonzero residual phase
- **Meaning:** Something happened beyond the simple additive story
  - Perhaps $i$ was influenced by other walkers besides $j$
  - Or the fitness landscape has curvature that created extra phase
  - Or stochastic noise added/subtracted phase
- The interaction is **non-trivial** — there's "interaction strength"
- Analogous to magnetic flux through a plaquette in lattice QED

**Physical significance:**

In regions of the Fractal Set where many triangles have large $|\Phi(\triangle)|$:
- High interaction complexity
- Strong coupling between walkers
- Non-additive influence (nonlinear effects)
- Possible emergent collective behavior

In regions with $\Phi(\triangle) \approx 0$:
- Weak or linear interactions
- Walkers evolving relatively independently
- Additive influence model holds

### 6.3 Directed IG Edges and Cloning Antisymmetry

**Question:** The IG edges are pairwise — there's an edge from $i$ to $j$ AND from $j$ to $i$. The cloning score has antisymmetric structure: one walker has positive cloning probability, the other negative. Does the edge directionality account for this antisymmetry?

**Answer:** Yes, the directed nature of IG edges DOES account for the cloning antisymmetry, through a combination of structural and data-level mechanisms.

#### The Antisymmetric Cloning Score

From `03_lattice_qft.md:357-382` ({prf:ref}`thm-cloning-antisymmetry-lqft`):

$$S_i(j) := \frac{V_j - V_i}{V_i + \varepsilon_{\mathrm{clone}}}$$

This gives the antisymmetry relation:

$$S_i(j) = -S_j(i) \cdot \frac{V_j + \varepsilon_{\mathrm{clone}}}{V_i + \varepsilon_{\mathrm{clone}}}$$

When $V_i \approx V_j$: $S_i(j) \approx -S_j(i)$ (exact antisymmetry)

**At any given time, for walkers $i$ and $j$:**
- If $V_i < V_j$: $S_i(j) > 0$ (i CAN clone from j), $S_j(i) < 0$ (j CANNOT clone from i)
- If $V_i > V_j$: $S_i(j) < 0$ (i CANNOT clone from j), $S_j(i) > 0$ (j CAN clone from i)
- If $V_i = V_j$: $S_i(j) = 0$ (neither clones), $S_j(i) = 0$

This is the **Algorithmic Exclusion Principle** ({prf:ref}`thm-exclusion-principle`): at most one walker per pair can clone in any given direction.

#### How Directed Edges Encode Antisymmetry

**Two separate directed edges for each pair:**

```
Walker i ----IG(i→j)----> Walker j     [i evaluates j]
          θ_ij = -(Φ_j - Φ_i)/ℏ_eff

Walker j ----IG(j→i)----> Walker i     [j evaluates i]
          θ_ji = -(Φ_i - Φ_j)/ℏ_eff = -θ_ij
```

**The phase antisymmetry:**

$$\theta_{ji} = -\theta_{ij}$$

This phase antisymmetry on the edges directly translates to cloning score antisymmetry because:
- Fitness difference: $(V_j - V_i)$ appears in $S_i(j)$
- Phase difference: $(\Phi_j - \Phi_i)$ appears in $\theta_{ij}$ (with opposite sign convention)
- The fitness and cumulative fitness phase are related in the selection framework

#### What Edge Direction Represents

**Edge $(i \to j)$:**
- **Algorithmic interpretation:** "Walker $i$ is evaluating walker $j$ as a potential influence source"
- **Data on this edge:** The phase $\theta_{ij}$ that walker $i$ uses when computing its cloning score toward $j$
- **Cloning potential:** $S_i(j)$ can be computed from data on this edge
- **Perspective:** Walker $i$'s view of walker $j$

**Edge $(j \to i)$:**
- **Algorithmic interpretation:** "Walker $j$ is evaluating walker $i$ as a potential influence source"
- **Data on this edge:** The phase $\theta_{ji} = -\theta_{ij}$ (opposite sign)
- **Cloning potential:** $S_j(i) = -S_i(j) \cdot \text{(scaling factor)}$
- **Perspective:** Walker $j$'s view of walker $i$

#### Important Clarifications

**1. All pairs have edges, not just cloning pairs:**
- IG edges exist for ALL $k_t(k_t-1)$ ordered pairs at timestep $t$
- Not limited to pairs that actually clone
- They represent potential influence channels (companion selection, viscous coupling)

**2. Cloning involves companion sampling:**
- Each walker samples a "cloning companion" from the soft companion distribution ({prf:ref}`def-fractal-set-companion-kernel`)
- Only the sampled companion gets considered for actual cloning
- But IG edges exist even for non-sampled pairs (complete directed graph)

**3. The outcome is recorded on IA edges:**
- IA edge $(n_{i,t+1}, n_{j,t})$ has attribute $\chi_{\mathrm{clone}}(e) = 1$ if $i$ actually cloned from $j$ ({prf:ref}`def-fractal-set-ia-attributes`)
- This is the RESULT, not the potential
- The cloning potential is computed from IG edge data

#### Summary: How Directionality Accounts for Antisymmetry

**Yes**, the directionality accounts for cloning asymmetry through:

1. **Structural accounting:** Two directed edges $(i \to j)$ and $(j \to i)$ allow each walker to "evaluate" the other independently

2. **Data antisymmetry:** The phases $\theta_{ij}$ and $\theta_{ji} = -\theta_{ij}$ carry opposite-sign information

3. **Algorithmic accounting:** When walker $i$ uses edge $(i \to j)$ to compute $S_i(j)$, and walker $j$ uses edge $(j \to i)$ to compute $S_j(i)$, the antisymmetry $S_i(j) \approx -S_j(i)$ emerges from the opposite-sign phases

4. **Physical meaning:** The direction encodes "who is the evaluator" — the asymmetry isn't just in the data, but in the *perspective* each walker has when looking at the pair

**Crucially:** The edges themselves don't "know" about cloning probabilities a priori. The antisymmetric structure ensures that when you COMPUTE cloning scores from the edge data, you get antisymmetric results. The directionality is **necessary but not sufficient** — you also need the antisymmetric data ($\theta_{ji} = -\theta_{ij}$) stored on the edges.

**Analogy to gauge theory:** This is similar to how gauge fields encode local symmetries. The directed edges provide the structure, the phase data provides the connection, and the antisymmetry emerges from the combination — just as the electromagnetic field tensor $F_{\mu\nu} = \partial_\mu A_\nu - \partial_\nu A_\mu$ is antisymmetric by construction from the gauge potential.

---

## 7. Conclusion

### 7.1 Summary of Findings

1. **Confirmation:** The documentation and diagrams are correct. **IA (Influence Attribution) edges close the Wilson loop**, not IG edges.

2. **Triangle structure:** Each interaction triangle $\triangle_{ij,t}$ consists of:
   - IG edge (spacelike): contemporary influence
   - CST edge (timelike): temporal evolution
   - IA edge (diagonal/retrocausal): causal attribution

3. **Topological necessity:** CST and IG edges alone **cannot** form closed loops because:
   - IG connects different walkers at the same time
   - CST connects the same walker at different times
   - No combination can close a cycle without a diagonal edge
   - IA provides the necessary diagonal: $(n_{i,t+1} \to n_{j,t})$

4. **Causal meaning:** IA edges represent **retrocausal attribution**—not physical time travel, but the logical inference "which past influences caused this current state?"

5. **Gauge theory structure:** Triangles are the minimal closed loops in the Fractal Set, analogous to plaquettes in standard lattice QFT but adapted to the three-edge-type causal structure.

### 7.2 Why This Makes Sense

**Three complementary reasons:**

1. **Topological:** The mathematics forces IA edges. You cannot build closed loops without diagonal connections.

2. **Causal:** The semantics justify IA edges. A complete interaction requires influence → evolution → attribution.

3. **Physical:** The gauge theory validates IA edges. Wilson loops on triangles measure interaction phases, giving a proper lattice QFT framework.

### 7.3 Reconciling with Genealogy Intuition

The initial expectation that triangles might involve cloning genealogy (common ancestors + IG edges) is understandable but conflates two distinct concepts:

| Concept | What it tracks | How it's stored | Purpose |
|---------|---------------|-----------------|---------|
| **Cloning genealogy** | Who cloned from whom | Node attributes (`c(n)`) | Understand ancestry, population dynamics |
| **Interaction triangles** | Who influenced whose evolution | 2-simplices (IG+CST+IA) | Compute Wilson loops, measure interaction strength |

Both are important, but they serve different purposes:
- **Genealogy** answers: "How did the population structure arise?"
- **Triangles** answer: "How did walkers' states change through dynamic interactions?"

**The key insight:** IA edges are not genealogical edges—they are **attribution edges**. They don't say "you are my ancestor"; they say "your influence contributed to my evolution."

### 7.4 Final Answer

**Yes, it makes sense to use IA edges to close the Wilson loop.**

In fact, it is **mathematically necessary** (CST+IG alone cannot form cycles) and **semantically meaningful** (IA completes the causal attribution loop).

The Fractal Set's three-edge-type structure gives it a richer causal topology than standard lattice QFT, with triangles (3-cycles) as the atomic Wilson loops instead of plaquettes (4-cycles). This is not a deficiency—it's a feature that reflects the algorithmic structure of the Fractal Gas dynamics.

---

## References

1. **Fractal Set definition:** `docs/source/3_fractal_gas/2_fractal_set/01_fractal_set.md`
   - Section 5.3: IA edges (lines 931-964)
   - Section 5.4: Interaction triangles (lines 966-1007)
   - Section 5.5: Plaquettes (lines 1009-1062)
   - Section 5.7: Wilson loops (lines 1111-1171)

2. **Lattice QFT structure:** `docs/source/3_fractal_gas/2_fractal_set/03_lattice_qft.md`
   - Section 2.2: Parallel transport (lines 95-157)
   - Section 2.3: Field strength tensor (lines 172-212)
   - Section 3: Wilson loops and holonomy (lines 247-331)

3. **Diagrams:**
   - `docs/svg_images/fractal_set_triangle.svg` — Basic triangle structure
   - `docs/source/3_fractal_gas/2_fractal_set/figures/interaction-triangle-transport.svg` — With parallel transport operators
   - `docs/source/3_fractal_gas/2_fractal_set/figures/triangles-to-plaquette.svg` — Plaquette formation

---

**END OF REPORT**
