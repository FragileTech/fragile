# Part III: The Axiom System

The Hypostructure $\mathbb{H}$ is valid if it satisfies the following structural constraints (Axioms). In the operational Sieve, these are verified as **Interface Permits** at the corresponding nodes.

(sec-conservation-constraints)=
## Conservation Constraints

:::{prf:axiom} Axiom D (Dissipation)
:label: ax-dissipation

The energy-dissipation inequality holds:
$$\Phi(S_t x) + \int_0^t \mathfrak{D}(S_s x) \, ds \leq \Phi(x)$$

**Enforced by:** {prf:ref}`def-node-energy` — Certificate $K_{D_E}^+$
:::

:::{prf:axiom} Axiom Rec (Recovery)
:label: ax-recovery

Discrete events are finite: $N(J) < \infty$ for any bounded interval $J$.

**Enforced by:** {prf:ref}`def-node-zeno` — Certificate $K_{\text{Rec}_N}^+$
:::

(sec-duality-constraints)=
## Duality Constraints

:::{prf:axiom} Axiom C (Compactness)
:label: ax-compactness

Bounded energy sequences admit convergent subsequences modulo the symmetry group $G$:
$$\sup_n \Phi(u_n) < \infty \implies \exists (n_k), g_k \in G: g_k \cdot u_{n_k} \to u_\infty$$

**Enforced by:** {prf:ref}`def-node-compact` — Certificate $K_{C_\mu}^+$ (or dispersion via $K_{C_\mu}^-$)
:::

:::{prf:axiom} Axiom SC (Scaling)
:label: ax-scaling

Dissipation scales faster than time: $\alpha > \beta$, where $\alpha$ is the energy scaling dimension and $\beta$ is the dissipation scaling dimension.

**Enforced by:** {prf:ref}`def-node-scale` — Certificate $K_{SC_\lambda}^+$
:::

(sec-symmetry-constraints)=
## Symmetry Constraints

:::{prf:axiom} Axiom LS (Stiffness)
:label: ax-stiffness

The Łojasiewicz-Simon inequality holds near equilibria, ensuring a mass gap:
$$\inf \sigma(L) > 0$$
where $L$ is the linearized operator at equilibrium.

**Enforced by:** {prf:ref}`def-node-stiffness` — Certificate $K_{LS_\sigma}^+$
:::

:::{prf:axiom} Axiom GC (Gradient Consistency)
:label: ax-gradient-consistency

Gauge invariance and metric compatibility: the control $T(u)$ matches the disturbance $d$.

**Enforced by:** {prf:ref}`def-node-align` — Certificate $K_{GC_T}^+$
:::

(sec-topology-constraints)=
## Topology Constraints

:::{prf:axiom} Axiom TB (Topological Background)
:label: ax-topology

Topological sectors are separated by an action gap:
$$[\pi] \in \pi_0(\mathcal{C})_{\text{acc}} \implies E < S_{\min} + \Delta$$

**Enforced by:** {prf:ref}`def-node-topo` — Certificate $K_{TB_\pi}^+$
:::

:::{prf:axiom} Axiom Cap (Capacity)
:label: ax-capacity

Capacity density bounds prevent concentration on thin sets:
$$\text{codim}(S) \geq 2 \implies \text{Cap}_H(S) = 0$$

**Enforced by:** {prf:ref}`def-node-geom` — Certificate $K_{\text{Cap}_H}^+$
:::

:::{prf:axiom} Axiom Geom (Geometric Structure License — Tits Alternative)
:label: ax-geom-tits

The Thin Kernel's simplicial complex $K$ must satisfy the **Discrete Tits Alternative**: it admits either polynomial growth (Euclidean/Nilpotent), hyperbolic structure (Logic/Free Groups), or is a CAT(0) space (Higher-Rank Lattices).

**Predicate**:
$$P_{\text{Geom}}(K) := (\text{Growth}(K) \leq \text{Poly}(d)) \lor (\delta_{\text{hyp}}(K) < \infty) \lor (\text{Cone}(K) \in \text{Buildings})$$

**Operational Check** (Node 7c):
1. **Polynomial Growth Test**: If ball growth satisfies $|B_r(x)| \sim r^d$ for some $d < \infty$, emit $K_{\text{Geom}}^{+}(\text{Poly})$. *(Euclidean/Nilpotent structures)*
2. **Hyperbolic Test**: Compute Gromov $\delta$-hyperbolicity constant. If $\delta < \epsilon \cdot \text{diam}(K)$ for small $\epsilon$, emit $K_{\text{Geom}}^{+}(\text{Hyp})$. *(Logic trees/Free groups)*
3. **CAT(0) Test**: Check triangle comparison inequality $d^2(m,x) \leq \frac{1}{2}d^2(y,x) + \frac{1}{2}d^2(z,x) - \frac{1}{4}d^2(y,z)$ for all triangles. If satisfied, emit $K_{\text{Geom}}^{+}(\text{CAT0})$. *(Higher-rank lattices/Yang-Mills)*

**Rejection Mode**:
If all three tests fail (exponential growth AND fat triangles AND no building structure), the object is an **Expander Graph** (thermalized, no coherent structure). Emit $K_{\text{Geom}}^{-}$ and route to **Mode D.D (Dispersion)** unless rescued by Spectral Resonance ({prf:ref}`ax-spectral-resonance`).

**Physical Interpretation**:
The Tits Alternative is the **universal dichotomy** for discrete geometric structures:
- **Polynomial/CAT(0)**: Structured (Crystal phase) → Admits finite description
- **Hyperbolic**: Critical (Liquid phase) → Admits logical encoding (infinite but compressible)
- **Expander**: Thermal (Gas phase) → No compressible structure (unless arithmetically constrained)

**Certificate**:
$$K_{\text{Geom}}^{+} = (\text{GrowthType} \in \{\text{Poly}, \text{Hyp}, \text{CAT0}\}, \text{evidence}, \text{parameters})$$

**Literature:** {cite}`Tits72` (Tits Alternative); {cite}`Gromov87` (Hyperbolic groups); {cite}`BridsonHaefliger99` (CAT(0) geometry); {cite}`Lubotzky94` (Expander graphs)

**Enforced by:** Node 7c (Geometric Structure Check) — Certificate $K_{\text{Geom}}^{\pm}$
:::

:::{prf:axiom} Axiom Spec (Spectral Resonance — Arithmetic Rescue)
:label: ax-spectral-resonance

An object **rejected** by {prf:ref}`ax-geom-tits` as an Expander (thermal chaos) is **re-admitted** if it exhibits **Spectral Rigidity** — non-decaying Bragg peaks indicating hidden arithmetic structure.

**Predicate**:
Let $\rho(\lambda)$ be the spectral density of states for the combinatorial Laplacian $\Delta_K$. Define the **Structure Factor**:
$$S(t) := \left|\int e^{i\lambda t} \rho(\lambda)\, d\lambda\right|^2$$

The object passes the **Spectral Resonance Test** if:
$$\exists \{p_i\}_{i=1}^N : \lim_{T \to \infty} \frac{1}{T} \int_0^T S(t)\, dt > \eta_{\text{noise}}$$

where $\{p_i\}$ are **quasi-periods** (resonances) and $\eta_{\text{noise}}$ is the random matrix theory baseline.

**Operational Check** (Node 7d):
1. **Eigenvalue Computation**: Compute spectrum $\text{spec}(\Delta_K) = \{\lambda_i\}$
2. **Level Spacing Statistics**: Compute nearest-neighbor spacing distribution $P(s)$
   - **Poisson** $P(s) \sim e^{-s}$: Random (Gas phase) → Fail
   - **GUE/GOE** $P(s) \sim s^\beta e^{-cs^2}$: Quantum chaos (Critical) → Pass if Trace Formula detected
3. **Trace Formula Detection**: Check for periodic orbit formula:
   $$\rho(\lambda) = \rho_{\text{Weyl}}(\lambda) + \sum_{\gamma \text{ periodic}} A_\gamma \cos(\lambda \ell_\gamma)$$
   If present, emit $K_{\text{Spec}}^{+}(\text{ArithmeticChaos})$

**Physical Interpretation**:
This distinguishes:
- **Arithmetic Chaos** (e.g., Riemann zeros, Quantum graphs, Maass forms): Expander-like growth BUT spectral correlations follow number-theoretic laws → **Admits arithmetic encoding**
- **Thermal Chaos** (Random matrices, generic expanders): No long-range spectral correlations → **Truly random (Gas phase)**

**Connection to Number Theory**:
The **Selberg Trace Formula** and **Explicit Formula** for the Riemann zeta function are instances of spectral resonance:
$$\psi(x) = x - \sum_\rho \frac{x^\rho}{\rho} - \log(2\pi)$$
where $\rho$ are the non-trivial zeros. The zeros exhibit GUE statistics (quantum chaos) but are **arithmetically structured**.

**Certificate**:
$$K_{\text{Spec}}^{+} = (\text{LevelStatistics} = \text{GUE/GOE}, \text{TraceFormula}, \{p_i\}, \eta_{\text{signal}}/\eta_{\text{noise}})$$

**Literature:** {cite}`Selberg56` (Trace formula); {cite}`MontgomeryOdlyzko73` (Pair correlation conjecture); {cite}`Sarnak95` (Quantum chaos); {cite}`KatzSarnak99` (Random matrix theory)

**Enforced by:** Node 7d (Spectral Resonance Check) — Certificate $K_{\text{Spec}}^{\pm}$
:::

:::{prf:remark} Universal Coverage via Tits + Spectral
:label: rem-universal-coverage

The combination of {prf:ref}`ax-geom-tits` and {prf:ref}`ax-spectral-resonance` achieves **universal coverage** of discrete structures:

| Structure Class | Tits Test | Spectral Test | Verdict | Example |
|----------------|-----------|---------------|---------|---------|
| Euclidean/Nilpotent | ✓ Poly | N/A | **REGULAR** (Crystal) | $\mathbb{Z}^d$, Heisenberg group |
| Hyperbolic/Free | ✓ Hyp | N/A | **PARTIAL** (Liquid) | Free groups, Logic trees |
| Higher-Rank/CAT(0) | ✓ CAT0 | N/A | **REGULAR** (Structured) | $SL(n,\mathbb{Z})$ ($n \geq 3$), Buildings |
| Arithmetic Chaos | ✗ Expander | ✓ Resonance | **PARTIAL** (Arithmetic) | Riemann zeros, Quantum graphs |
| Thermal Chaos | ✗ Expander | ✗ Random | **HORIZON** (Gas) | Random matrices, Generic expanders |

This resolves the **completeness gap**: every discrete structure routes to exactly one verdict.
:::

(sec-boundary-constraints)=
## Boundary Constraints

The Boundary Constraints enforce coupling between bulk dynamics and environmental interface via the Thin Interface $\partial^{\text{thin}} = (\mathcal{B}, \text{Tr}, \mathcal{J}, \mathcal{R})$.

:::{prf:axiom} Axiom Bound (Input/Output Coupling)
:label: ax-boundary

The system's boundary morphisms satisfy:
- $\mathbf{Bound}_\partial$: $\text{Tr}: \mathcal{X} \to \mathcal{B}$ is not an equivalence (open system) — {prf:ref}`def-node-boundary`
- $\mathbf{Bound}_B$: $\mathcal{J}$ factors through a bounded subobject $\mathcal{J}: \mathcal{B} \to \underline{[-M, M]}$ — {prf:ref}`def-node-overload`
- $\mathbf{Bound}_{\Sigma}$: The integral $\int_0^T \mathcal{J}_{\text{in}}$ exists as a morphism in $\text{Hom}(\mathbf{1}, \underline{\mathbb{R}}_{\geq r_{\min}})$ — {prf:ref}`def-node-starve`
- $\mathbf{Bound}_{\mathcal{R}}$: The **reinjection diagram** commutes:
  $$\mathcal{J}_{\text{out}} \simeq \mathcal{J}_{\text{in}} \circ \mathcal{R} \quad \text{in } \text{Hom}_{\mathcal{E}}(\mathcal{B}, \underline{\mathbb{R}})$$

**Enforced by:** {prf:ref}`def-node-boundary`, {prf:ref}`def-node-overload`, {prf:ref}`def-node-starve`, {prf:ref}`def-node-align`
:::

:::{prf:remark} Reinjection Boundaries (Fleming-Viot)
:label: rem-reinjection

When $\mathcal{R} \not\simeq 0$, the boundary acts as a **non-local transport morphism** rather than an absorbing terminal object. This captures:
- **Fleming-Viot processes:** The reinjection factors through the **probability monad** $\mathcal{P}: \mathcal{E} \to \mathcal{E}$
- **McKean-Vlasov dynamics:** $\mathcal{R}$ depends on global sections $\Gamma(\mathcal{X}, \mathcal{O}_\mu)$
- **Piecewise Deterministic Markov Processes:** $\mathcal{R}$ is a morphism in the Kleisli category of the probability monad

The Sieve verifies regularity by checking **Axiom Rec** at the boundary:
1. **{prf:ref}`def-node-boundary`:** Detects that $\mathcal{J} \neq 0$ (non-trivial exit flux)
2. **{prf:ref}`def-node-starve`:** Verifies $\mathcal{R}$ preserves the **total mass section** ($K_{\text{Mass}}^+$)

Categorically, this defines a **non-local boundary condition** as a span:
$$\mathcal{X} \xleftarrow{\text{Tr}} \mathcal{B} \xrightarrow{\mathcal{R}} \mathcal{P}(\mathcal{X})$$
The resulting integro-differential structure is tamed by **Axiom C** applied to the Wasserstein $\infty$-stack $\mathcal{P}_2(\mathcal{X})$.
:::
