(sec-lock)=
# Part XI: The Lock (Conjecture Prover Backend)

:::{div} feynman-prose
Now we come to the final gate. All the diagnostics we have seen so far ask specific questions: Is the energy bounded? Does the system mix properly? Are there dangerous symmetry-breaking cascades? Each of these is a necessary condition for regularity. But here is the real question: can a pathological structure actually *embed* into our system?

Think of it this way. The bad pattern, whatever it is, needs to find a way to live inside your system. It needs a morphism from the universal bad object into your hypostructure. If no such morphism exists, the bad thing cannot happen. Period.

This is what the Lock does. It is not checking individual symptoms; it is asking the fundamental categorical question: is the Hom-set empty? And it has thirteen different ways to prove emptiness, each attacking the problem from a different angle.
:::

(sec-lock-contract)=
## Lock Contract

:::{div} feynman-prose
Let me explain what makes this node special. Every other node in the Sieve answers a local question. The Lock answers a global one: can the universal bad pattern find any foothold at all?

The notation looks intimidating, but the idea is simple. We have a category of hypostructures, and we have identified the "worst case" pattern that would represent failure. We are asking: is there any structure-preserving map from that bad pattern into our system? If not, we win.
:::

:::{prf:definition} Lock contract
:label: def-lock-contract

The **Categorical Lock** is the final barrier with special structure:

**Trigger**: All prior checks passed or blocked (convergent paths)

**Pre-certificates**: Full context $\Gamma$ from prior nodes

**Question**: Is $\mathrm{Hom}_{\mathbf{Hypo}}(\mathbb{H}_{\mathrm{bad}}, \mathcal{H}) = \varnothing$?

Where:
- $\mathbf{Hypo}$ is the category of hypostructures
- $\mathbb{H}_{\mathrm{bad}}$ is the universal bad pattern (initial object of R-breaking subcategory)
- $\mathcal{H}$ is the system under analysis

**Outcomes**:
- **Blocked** ($K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$ or $K_{\mathrm{Cat}_{\mathrm{Hom}}}^+$): Hom-set empty; implies GLOBAL REGULARITY
- **MorphismExists** ($K_{\mathrm{Cat}_{\mathrm{Hom}}}^-$): Explicit morphism $\phi: \mathbb{H}_{\mathrm{bad}} \to \mathcal{H}$; implies FATAL ERROR

**Goal Certificate:** For the Lock, the designated goal certificate for the proof completion criterion ({prf:ref}`def-proof-complete`) is $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$. This certificate suffices for proof completion—no additional promotion to $K_{\mathrm{Cat}_{\mathrm{Hom}}}^+$ is required. The blocked outcome at the Lock establishes morphism exclusion directly.

:::

---

(sec-lock-exclusion-tactics)=
## E1--E13 Exclusion Tactics

:::{div} feynman-prose
Here is the beautiful part. There are many different reasons why a morphism might fail to exist, and each one gives us a proof. If dimensions do not match, there is no morphism. If invariants do not match, there is no morphism. If one structure has positive energy and the other requires negative, there is no morphism.

The thirteen tactics below are like different keys for the same lock. We try each one. If any key turns, we have proven Hom-emptiness. The system is safe.

What makes this powerful is that these tactics are *independent*. If the dimension tactic fails because both objects have the same dimension, maybe the invariant tactic succeeds because their topological invariants differ. The tactics attack different aspects of the structure.
:::

The Lock attempts thirteen proof-producing tactics to establish Hom-emptiness:

:::{prf:definition} E1: Dimension obstruction
:label: def-e1

**Sieve Signature:**
- **Required Permits:** $\mathrm{Rep}_K$ (representability), $\mathrm{Cat}_{\mathrm{Hom}}$
- **Weakest Precondition:** $\{K_{\mathrm{Rep}_K}^+\}$ (finite representability confirmed)
- **Produces:** $K_{\mathrm{E1}}^+ \Rightarrow K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$ (Hom-emptiness via dimension)
- **Blocks:** All failure modes (Global Regularity)
- **Breached By:** Dimensions compatible or dimension not computable

**Method**: Linear algebra / dimension counting

**Mechanism**: If $\dim(\mathbb{H}_{\mathrm{bad}}) \neq \dim(\mathcal{H})$ in a way incompatible with morphisms, Hom is empty.

**Certificate Logic:**
$$K_{\mathrm{Rep}_K}^+ \wedge (d_{\mathrm{bad}} \neq d_{\mathcal{H}}) \Rightarrow K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$$

**Certificate Payload**: $(d_{\text{bad}}, d_{\mathcal{H}}, \text{dimension mismatch proof})$

**Automation**: Fully automatable via linear algebra

**Literature:** Brouwer invariance of domain {cite}`Brouwer11`; dimension theory {cite}`HurewiczWallman41`.

:::

:::{div} feynman-prose
E1 is the simplest obstruction: dimension counting. You cannot fit a three-dimensional thing into a two-dimensional space. Linear algebra tells you immediately. This is the first thing to check because it is cheap and often works.
:::

:::{prf:definition} E2: Invariant mismatch
:label: def-e2

**Sieve Signature:**
- **Required Permits:** $\mathrm{Rep}_K$, $\mathrm{TB}_\pi$ (topological background), $\mathrm{Cat}_{\mathrm{Hom}}$
- **Weakest Precondition:** $\{K_{\mathrm{Rep}_K}^+, K_{\mathrm{TB}_\pi}^+\}$
- **Produces:** $K_{\mathrm{E2}}^+ \Rightarrow K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$
- **Blocks:** All failure modes (Global Regularity)
- **Breached By:** Invariants match or invariant not extractable

**Method**: Invariant extraction + comparison

**Mechanism**: If morphisms must preserve invariant $I$ but $I(\mathbb{H}_{\mathrm{bad}}) \neq I(\mathcal{H})$, Hom is empty.

**Certificate Logic:**
$$K_{\mathrm{TB}_\pi}^+ \wedge (I_{\mathrm{bad}} \neq I_{\mathcal{H}}) \Rightarrow K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$$

**Certificate Payload**: $(I, I_{\text{bad}}, I_{\mathcal{H}}, I_{\text{bad}} \neq I_{\mathcal{H}} \text{ proof})$

**Automation**: Automatable for extractable invariants (Euler char, homology, etc.)

**Literature:** Topological invariants {cite}`EilenbergSteenrod52`; K-theory {cite}`Quillen73`.

:::

:::{div} feynman-prose
E2 uses topological invariants. Even if two spaces have the same dimension, they might have different numbers of holes, different Euler characteristics, different homology groups. A torus is not a sphere, even though both are two-dimensional surfaces. Morphisms preserve these invariants, so a mismatch blocks the Hom-set.
:::

:::{prf:definition} E3: Positivity obstruction
:label: def-e3

**Sieve Signature:**
- **Required Permits:** $D_E$ (energy), $\mathrm{LS}_\sigma$ (local stiffness), $\mathrm{Cat}_{\mathrm{Hom}}$
- **Weakest Precondition:** $\{K_{D_E}^+, K_{\mathrm{LS}_\sigma}^+\}$
- **Produces:** $K_{\mathrm{E3}}^+ \Rightarrow K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$
- **Blocks:** All failure modes (Global Regularity)
- **Breached By:** Positivity compatible or cone structure absent

**Method**: Cone / positivity constraints

**Mechanism**: If morphisms must preserve positivity but $\mathbb{H}_{\mathrm{bad}}$ violates positivity required by $\mathcal{H}$, Hom is empty.

**Certificate Logic:**
$$K_{\mathrm{LS}_\sigma}^+ \wedge (\Phi_{\mathrm{bad}} \notin \mathcal{C}_+) \Rightarrow K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$$

**Certificate Payload**: $(P, \text{positivity constraint}, \text{violation witness})$

**Automation**: Via semidefinite programming / cone analysis

**Literature:** Positive energy theorems {cite}`SchoenYau79`; {cite}`Witten81`; convex cones {cite}`Rockafellar70`.

:::

:::{div} feynman-prose
E3 is about positivity constraints. Physical systems often require energy to be positive, or certain operators to live in a cone. If the bad pattern violates positivity and your system enforces it, there can be no morphism. This is the physics speaking: you cannot continuously deform a positive-energy configuration into a negative-energy one.
:::

:::{prf:definition} E4: Integrality obstruction
:label: def-e4

**Sieve Signature:**
- **Required Permits:** $\mathrm{Rep}_K$, $\mathrm{Cat}_{\mathrm{Hom}}$
- **Weakest Precondition:** $\{K_{\mathrm{Rep}_K}^+\}$ (arithmetic structure available)
- **Produces:** $K_{\mathrm{E4}}^+ \Rightarrow K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$
- **Blocks:** All failure modes (Global Regularity)
- **Breached By:** Arithmetic structures compatible or not decidable

**Method**: Discrete / arithmetic constraints

**Mechanism**: If morphisms require integral/rational structure but bad pattern has incompatible arithmetic, Hom is empty.

**Certificate Logic:**
$$K_{\mathrm{Rep}_K}^+ \wedge (\Lambda_{\mathrm{bad}} \not\hookrightarrow \Lambda_{\mathcal{H}}) \Rightarrow K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$$

**Certificate Payload**: $(\text{arithmetic structure}, \text{incompatibility proof})$

**Automation**: Via number theory / SMT with integer arithmetic

**Literature:** Arithmetic obstructions {cite}`Serre73`; lattice theory {cite}`CasselsSwinnerton70`.

:::

:::{prf:definition} E5: Functional equation obstruction
:label: def-e5

**Sieve Signature:**
- **Required Permits:** $\mathrm{Rep}_K$, $\mathrm{GC}_\nabla$ (gauge covariance), $\mathrm{Cat}_{\mathrm{Hom}}$
- **Weakest Precondition:** $\{K_{\mathrm{Rep}_K}^+\}$
- **Produces:** $K_{\mathrm{E5}}^+ \Rightarrow K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$
- **Blocks:** All failure modes (Global Regularity)
- **Breached By:** Functional equations solvable or undecidable

**Method**: Rewriting / functional constraints

**Mechanism**: If morphisms must satisfy functional equations that have no solution, Hom is empty.

**Certificate Logic:**
$$K_{\mathrm{Rep}_K}^+ \wedge (\text{FuncEq}(\phi) = \bot) \Rightarrow K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$$

**Certificate Payload**: $(\text{functional eq.}, \text{unsolvability proof})$

**Automation**: Via term rewriting / constraint solving

**Literature:** Functional equations {cite}`AczélDhombres89`; rewriting systems {cite}`BaaderNipkow98`.

:::

:::{div} feynman-prose
E4 and E5 deal with discrete arithmetic and functional constraints. Sometimes the structure requires integer or rational values that simply do not fit together. Sometimes there are functional equations that have no solution. These are more subtle than dimension counting but equally fatal when they apply.
:::

:::{prf:definition} E6: Causal obstruction (Well-Foundedness)
:label: def-e6

**Sieve Signature:**
- **Required Permits:** $\mathrm{TB}_\pi$ (topological/causal structure), $D_E$ (dissipation), $\mathrm{Cat}_{\mathrm{Hom}}$
- **Weakest Precondition:** $\{K_{\mathrm{TB}_\pi}^+, K_{D_E}^+\}$ (causal structure and energy bound available)
- **Produces:** $K_{\mathrm{E6}}^+ \Rightarrow K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$
- **Blocks:** All failure modes (Global Regularity); specifically excludes CTCs
- **Breached By:** Causal structure compatible or well-foundedness undecidable

**Method**: Order theory / Causal set analysis

**Mechanism**: If morphisms must preserve the causal partial order $\prec$ but $\mathbb{H}_{\mathrm{bad}}$ contains infinite descending chains $v_0 \succ v_1 \succ \cdots$ (violating well-foundedness/Artinian condition), Hom is empty. The axiom of foundation connects to chronology protection: infinite causal descent requires unbounded negative energy, violating $D_E$.

**Certificate Logic:**
$$K_{\mathrm{TB}_\pi}^+ \wedge K_{D_E}^+ \wedge (\exists \text{ infinite descending chain in } \mathbb{H}_{\mathrm{bad}}) \Rightarrow K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$$

**Certificate Payload**: $(\prec_{\mathrm{bad}}, \text{descending chain witness}, \text{Artinian violation proof})$

**Automation**: Via order-theoretic analysis / transfinite induction / causal set algorithms

**Literature:** Causal set theory {cite}`Bombelli87`; {cite}`Sorkin05`; set-theoretic foundations {cite}`Jech03`.

:::

:::{div} feynman-prose
E6 is deep. It says: if the bad pattern contains closed timelike curves, infinite causal descent, it cannot embed into a well-founded causal structure. This is the axiom of foundation in set theory meeting physics. You cannot have an infinite regress of causes; the energy required would be unbounded.
:::

:::{prf:definition} E7: Thermodynamic obstruction (Entropy)
:label: def-e7

**Sieve Signature:**
- **Required Permits:** $D_E$ (dissipation/energy), $\mathrm{SC}_\lambda$ (scaling/entropy), $\mathrm{Cat}_{\mathrm{Hom}}$
- **Weakest Precondition:** $\{K_{D_E}^+, K_{\mathrm{SC}_\lambda}^+\}$ (energy dissipation and scaling available)
- **Produces:** $K_{\mathrm{E7}}^+ \Rightarrow K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$
- **Blocks:** All failure modes (Global Regularity); specifically Mode C.E (energy blow-up)
- **Breached By:** Entropy production compatible or Lyapunov function absent

**Method**: Lyapunov analysis / Entropy production bounds

**Mechanism**: If morphisms must respect the Second Law ($\Delta S \geq 0$) but $\mathbb{H}_{\mathrm{bad}}$ requires entropy decrease incompatible with $\mathcal{H}$, Hom is empty. Lyapunov functions satisfying $\frac{d\mathcal{L}}{dt} \leq -\lambda \mathcal{L} + b$ (Foster-Lyapunov condition) enforce monotonic approach to equilibrium.

**Certificate Logic:**
$$K_{D_E}^+ \wedge K_{\mathrm{SC}_\lambda}^+ \wedge (\Delta S_{\mathrm{bad}} < 0) \Rightarrow K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$$

**Certificate Payload**: $(S_{\mathrm{bad}}, S_{\mathcal{H}}, \Delta S < 0 \text{ witness}, \text{Second Law violation proof})$

**Automation**: Via Lyapunov analysis / entropy production estimation / drift-diffusion bounds

**Literature:** Optimal transport {cite}`Villani09`; fluctuation theorems {cite}`Jarzynski97`; Foster-Lyapunov {cite}`MeynTweedie93`.

:::

:::{div} feynman-prose
E7 is the Second Law as a morphism obstruction. If the bad pattern requires entropy to decrease but your system enforces thermodynamic consistency, there can be no embedding. This is not a technicality; it is fundamental. The arrow of time is an obstruction to certain morphisms.
:::

:::{prf:definition} E8: Data Processing Interaction (DPI)
:label: def-e8

**Sieve Signature:**
- **Required Permits:** $\mathrm{Cap}_H$ (capacity), $\mathrm{TB}_\pi$ (topological boundary), $\mathrm{Cat}_{\mathrm{Hom}}$
- **Weakest Precondition:** $\{K_{\mathrm{Cap}_H}^+, K_{\mathrm{TB}_\pi}^+\}$ (capacity bound and topology available)
- **Produces:** $K_{\mathrm{E8}}^+ \Rightarrow K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$
- **Blocks:** All failure modes (Global Regularity); specifically Mode C.D (geometric collapse)
- **Breached By:** Information density exceeding channel capacity

**Method**: Data Processing Inequality / Channel Capacity Analysis

**Mechanism**: The boundary $\partial \mathcal{X}$ acts as a communication channel $Y$ between the bulk system $X$ and the external observer $Z$. By the **Data Processing Inequality (DPI)**, processing cannot increase information: $I(X; Z) \leq I(X; Y)$. If the bulk requires transmitting more information than the boundary channel capacity $C(Y)$ allows ($I_{\text{bulk}} > C_{\text{boundary}}$), the interaction is impossible. The singularity is "hidden" because it cannot be faithfully observed.

**Certificate Logic:**
$$K_{\mathrm{Cap}_H}^+ \wedge K_{\mathrm{TB}_\pi}^+ \wedge (I_{\mathrm{bad}} > C_{\max}(\partial \mathcal{H})) \Rightarrow K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$$

**Certificate Payload**: $(I_{\mathrm{bad}}, C_{\max}, \text{DPI violation proof})$

**Automation**: Via mutual information estimation / channel capacity computation

**Literature**: Data Processing Inequality {cite}`CoverThomas06`; Channel Capacity {cite}`Shannon48`.

:::

:::{div} feynman-prose
E8 uses information theory. The boundary of your system acts like a communication channel with finite capacity. If the bad pattern requires transmitting more information through the boundary than the channel can handle, the interaction is impossible. The singularity is "hidden" in the same sense that information cannot exceed channel capacity.
:::

:::{prf:definition} E9: Ergodic obstruction (Mixing)
:label: def-e9

**Sieve Signature:**
- **Required Permits:** $\mathrm{TB}_\rho$ (mixing/ergodic structure), $C_\mu$ (compactness), $\mathrm{Cat}_{\mathrm{Hom}}$
- **Weakest Precondition:** $\{K_{\mathrm{TB}_\rho}^+, K_{C_\mu}^+\}$ (mixing rate and concentration available)
- **Produces:** $K_{\mathrm{E9}}^+ \Rightarrow K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$
- **Blocks:** All failure modes (Global Regularity); specifically Mode T.D (glassy freeze)
- **Breached By:** Mixing properties compatible or spectral gap not computable

**Method**: Spectral gap analysis / Mixing time bounds

**Mechanism**: If morphisms must preserve mixing properties but $\mathbb{H}_{\mathrm{bad}}$ has incompatible spectral gap, Hom is empty. Mixing systems satisfy $\mu(A \cap S_t^{-1}B) \to \mu(A)\mu(B)$, with spectral gap $\gamma > 0$ implying exponential correlation decay $|C(t)| \leq e^{-\gamma t}$. Glassy dynamics (localization) cannot map into rapidly mixing systems.

**Certificate Logic:**
$$K_{\mathrm{TB}_\rho}^+ \wedge K_{C_\mu}^+ \wedge (\gamma_{\mathrm{bad}} = 0 \wedge \gamma_{\mathcal{H}} > 0) \Rightarrow K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$$

**Certificate Payload**: $(\tau_{\text{mix, bad}}, \tau_{\text{mix}, \mathcal{H}}, \text{spectral gap mismatch proof})$

**Automation**: Via spectral gap estimation / Markov chain analysis / correlation function computation

**Literature:** Ergodic theorem {cite}`Birkhoff31`; mixing times {cite}`LevinPeresWilmer09`; recurrence {cite}`Furstenberg81`.

:::

:::{div} feynman-prose
E9 is about dynamics. A rapidly mixing system forgets its initial conditions exponentially fast. A glassy system gets stuck. These are fundamentally different behaviors, and you cannot map one into the other while preserving the dynamical structure.
:::

:::{prf:definition} E10: Definability obstruction (Tameness)
:label: def-e10

**Sieve Signature:**
- **Required Permits:** $\mathrm{TB}_O$ (o-minimal/tame structure), $\mathrm{Rep}_K$ (representability), $\mathrm{Cat}_{\mathrm{Hom}}$
- **Weakest Precondition:** $\{K_{\mathrm{TB}_O}^+, K_{\mathrm{Rep}_K}^+\}$ (tameness and finite representation available)
- **Produces:** $K_{\mathrm{E10}}^+ \Rightarrow K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$
- **Blocks:** All failure modes (Global Regularity); specifically Mode T.C (labyrinthine/wild)
- **Breached By:** Both structures tame or definability undecidable

**Method**: Model theory / O-minimal structure analysis

**Mechanism**: If morphisms must preserve o-minimal (tame) structure but $\mathbb{H}_{\mathrm{bad}}$ involves wild topology, Hom is empty. O-minimality ensures definable subsets of $\mathbb{R}$ are finite unions of points and intervals. The cell decomposition theorem gives finite stratification with bounded Betti numbers $\sum_k b_k(A) \leq C$. Wild embeddings (Alexander horned sphere, Cantor boundaries) cannot exist in tame structures.

**Certificate Logic:**
$$K_{\mathrm{TB}_O}^+ \wedge K_{\mathrm{Rep}_K}^+ \wedge (\mathbb{H}_{\mathrm{bad}} \notin \mathcal{O}\text{-min}) \Rightarrow K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$$

**Certificate Payload**: $(\text{definability class}, \text{wild topology witness}, \text{cell decomposition failure})$

**Automation**: Via model-theoretic analysis / stratification algorithms / Betti number computation

**Literature:** Tame topology {cite}`vandenDries98`; quantifier elimination {cite}`Tarski51`; model completeness {cite}`Wilkie96`.

:::

:::{div} feynman-prose
E10 asks: is the bad pattern "tame" or "wild"? Tame topology, in the sense of o-minimal structures, has finite complexity. Wild topology, like an Alexander horned sphere, has infinite complexity. Tame structures cannot contain wild ones. This is not just a technicality; it is a fundamental barrier from model theory.
:::

:::{prf:definition} E11: Galois-Monodromy Lock
:label: def-e11

**Sieve Signature:**
- **Required Permits:** $\mathrm{Rep}_K$ (representation/algebraic structure), $\mathrm{TB}_\pi$ (topology/monodromy), $\mathrm{Cat}_{\mathrm{Hom}}$
- **Weakest Precondition:** $\{K_{\mathrm{Rep}_K}^+, K_{\mathrm{TB}_\pi}^+\}$ (Galois group and monodromy available)
- **Produces:** $K_{\mathrm{E11}}^+ \Rightarrow K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$
- **Blocks:** S.E (Supercritical Cascade); S.C (Computational Overflow)
- **Breached By:** Galois group solvable or monodromy finite

:::{prf:definition} Galois Group
:label: def-galois-group-permit

For a polynomial $f(x) \in \mathbb{Q}[x]$, the **Galois group** $\mathrm{Gal}(f)$ is the group of automorphisms of the splitting field $K$ that fix $\mathbb{Q}$.
:::

:::{prf:definition} Monodromy Group
:label: def-monodromy-group-permit

For a differential equation with singularities, the **monodromy group** $\mathrm{Mon}(f)$ describes how solutions transform when analytically continued around singularities.
:::

**Method**: Galois theory / Monodromy representation analysis

**Mechanism**: If morphisms must preserve algebraic structure but $\mathbb{H}_{\mathrm{bad}}$ has non-solvable Galois group, no closed-form solution exists. The key constraints are:

1. **Orbit Finiteness:** If $\mathrm{Gal}(f)$ is finite, the orbit of any root under field automorphisms is finite:
   $$|\{\sigma(\alpha) : \sigma \in \mathrm{Gal}(f)\}| = |\mathrm{Gal}(f)| < \infty$$

2. **Solvability Obstruction:** If $\mathrm{Gal}(f)$ is not solvable (e.g., $S_n$ for $n \geq 5$), then $f$ has no solution in radicals. The system cannot be simplified beyond a certain complexity threshold.

3. **Monodromy Constraint:** For a differential equation, if the monodromy group is infinite, solutions have infinitely many branches (cannot be single-valued on any open set).

4. **Computational Barrier:** Determining $\mathrm{Gal}(f)$ is generally hard (no polynomial-time algorithm known). This prevents algorithmic shortcuts in solving algebraic systems.

**Certificate Logic:**
$$K_{\mathrm{Rep}_K}^+ \wedge K_{\mathrm{TB}_\pi}^+ \wedge (\mathrm{Gal}(f) \text{ non-solvable} \vee |\mathrm{Mon}(f)| = \infty) \Rightarrow K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$$

**Certificate Payload**: $(\mathrm{Gal}(f), \text{solvability status}, \mathrm{Mon}(f), \text{Abel-Ruffini witness})$

**Automation**: Via factorization over primes / Chebotarev density analysis / monodromy computation

**Literature:** Abel-Ruffini theorem {cite}`Abel1826`; Galois theory {cite}`DummitFoote04`; differential Galois theory {cite}`vanderPutSinger03`; Schlesinger's theorem {cite}`Schlesinger12`.

:::

:::{prf:proof} Proof Sketch (Abel-Ruffini)
:label: proof-e11-abel-ruffini

*Step 1 (Galois correspondence).* For $f(x) \in \mathbb{Q}[x]$ with splitting field $K$, the Galois group $\mathrm{Gal}(K/\mathbb{Q})$ embeds into $S_n$ via root permutations. The Fundamental Theorem establishes bijection: subgroups $H \subseteq \mathrm{Gal}(K/\mathbb{Q}) \leftrightarrow$ intermediate fields $\mathbb{Q} \subseteq F \subseteq K$.

*Step 2 (Solvability criterion).* $f$ is solvable by radicals iff $\mathrm{Gal}(f)$ is a solvable group (admits subnormal series with abelian quotients). Each radical extension $F_{i+1} = F_i(\sqrt[n_i]{a_i})$ corresponds to cyclic Galois quotient.

*Step 3 (Non-solvability of $S_n$).* For $n \geq 5$, $A_n$ is simple (non-trivial normal subgroups contain 3-cycles, which generate $A_n$). The derived series $S_n \triangleright A_n \triangleright \{e\}$ fails to terminate abelianly; $S_n$ is not solvable.

*Step 4 (Generic quintic).* For generic quintic $f(x) = x^5 + \cdots$, $\mathrm{Gal}(f) \cong S_5$. No radical expression exists for roots.

*Step 5 (Monodromy-Galois correspondence).* For Fuchsian ODEs, the monodromy group $\mathrm{Mon}(f)$ is Zariski-dense in the differential Galois group $G_{\mathrm{diff}}$. Infinite monodromy implies infinitely many solution branches.

:::

:::{div} feynman-prose
E11 reaches deep into algebra. Remember Abel's proof that the general quintic has no solution in radicals? The Galois group is not solvable. This is not a matter of cleverness; it is a structural impossibility. If the bad pattern requires solving an unsolvable equation, there can be no morphism that produces it.

The monodromy obstruction is the same idea for differential equations. When you analytically continue solutions around singularities, they transform. If the transformations form an infinite group, the solutions have infinitely many branches. This infinite complexity cannot embed into finite structures.
:::

:::{prf:definition} E12: Algebraic Compressibility (Permit Schema with Alternative Backends)
:label: def-e12

**Sieve Signature:**
- **Required Permits (Alternative Backends):**
  - **Backend A:** $K_{\mathrm{Rep}_K}^+$ (hypersurface) + $K_{\mathrm{SC}_\lambda}^{\text{deg}}$ → $K_{\mathrm{E12}}^{\text{hypersurf}}$
  - **Backend B:** $K_{\mathrm{Rep}_K}^+$ (complete intersection) + $K_{\mathrm{SC}_\lambda}^{\text{Bez}}$ → $K_{\mathrm{E12}}^{\text{c.i.}}$
  - **Backend C:** $K_{\mathrm{Rep}_K}^+$ (morphism) + $K_{\mathrm{DegImage}_m}^+$ + $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\text{deg}}$ → $K_{\mathrm{E12}}^{\text{morph}}$
- **Weakest Precondition:** $\{K_{\mathrm{Rep}_K}^+\}$ (algebraic variety structure available)
- **Produces:** $K_{\mathrm{E12}}^+ \Rightarrow K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$
- **Blocks:** S.E (Supercritical Cascade); S.C (Computational Overflow)
- **Breached By:** Degree compatibility, linear structure, or compatible morphism exists

**Context:** Algebraic compressibility obstructions arise when attempting to approximate or represent a high-degree variety using lower-degree data. The degree of an algebraic variety is an intrinsic geometric invariant that resists compression.

**Critical Remark:** The naive claim "degree $\delta$ cannot be represented by polynomials of degree $< \delta$" is **imprecise** for general varieties (e.g., a parametric representation can use lower-degree maps). The following backends make the obstruction precise by specifying what "representation" means.

:::{prf:definition} Algebraic Variety
:label: def-algebraic-variety-permit

An **algebraic variety** $V \subset \mathbb{P}^n$ (or $\mathbb{C}^n$) is the zero locus of polynomial equations:
$$V = \{x \in \mathbb{P}^n : f_1(x) = \cdots = f_k(x) = 0\}$$
:::

:::{prf:definition} Degree of a Variety
:label: def-variety-degree-permit

The **degree** $\deg(V)$ of an irreducible variety $V \subset \mathbb{P}^n$ of dimension $d$ is the number of intersection points with a generic linear subspace $L$ of complementary dimension $(n-d)$:
$$\deg(V) = \#(V \cap L)$$
counted with multiplicity. Equivalently, $\deg(V) = \int_V c_1(\mathcal{O}(1))^d$.
:::

**Certificate Logic:**
$$K_{\mathrm{Rep}_K}^+ \wedge K_{\mathrm{SC}_\lambda}^+ \wedge \left(K_{\mathrm{E12}}^{\text{hypersurf}} \vee K_{\mathrm{E12}}^{\text{c.i.}} \vee K_{\mathrm{E12}}^{\text{morph}}\right) \Rightarrow K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$$

---

#### Backend A: Hypersurface Form

**Hypotheses:**
1. $V = Z(f) \subset \mathbb{P}^n$ is an **irreducible hypersurface**
2. $f \in \mathbb{C}[x_0, \ldots, x_n]$ is irreducible with $\deg(f) = \delta$
3. "Representation" means: a single polynomial whose zero locus is $V$

**Certificate:** $K_{\mathrm{E12}}^{\text{hypersurf}} = (\delta, f, \text{irreducibility witness})$

**Literature:** Irreducibility and defining equations {cite}`Hartshorne77`; Nullstellensatz {cite}`CoxLittleOShea15`

---

#### Backend B: Complete Intersection Form

**Hypotheses:**
1. $V \subset \mathbb{P}^n$ is a **complete intersection** of codimension $k$
2. $V = Z(f_1, \ldots, f_k)$ where $\deg(f_i) = d_i$ and $\dim V = n - k$ (expected dimension)
3. "Representation" means: $k$ equations cutting out $V$ scheme-theoretically

**Certificate:** $K_{\mathrm{E12}}^{\text{c.i.}} = (\deg(V), k, (d_1, \ldots, d_k))$

**Literature:** Bézout's theorem {cite}`Fulton84`; complete intersections {cite}`EisenbudHarris16`

---

#### Backend C: Morphism / Compression Form

**Hypotheses:**
1. $V \subset \mathbb{P}^n$ is an irreducible variety of dimension $d$ and degree $\delta$
2. A "compression of complexity $m$" is a generically finite morphism $\phi: W \to V$ of degree $\leq m$ from a variety $W$ of degree $< \delta$
3. Equivalently: $V$ is the image of a low-degree variety under a low-degree map

**Certificate:** $K_{\mathrm{E12}}^{\text{morph}} = (\delta, d, m_{\min}, \text{Bézout witness})$

**Literature:** Degrees of morphisms {cite}`Lazarsfeld04`; projection formulas {cite}`Fulton84`

---

**Backend Selection Logic:**

| Backend | Hypothesis | Best For |
|:-------:|:----------:|:--------:|
| A | $V$ is irreducible hypersurface | Single-equation varieties, cryptographic hardness |
| B | $V$ is complete intersection | Multi-equation varieties, computational algebra |
| C | Morphism/parametric representation | Parametrization complexity, circuit lower bounds |

**Automation:** Via degree computation / resultant analysis / intersection multiplicity bounds / Gröbner bases

**Use:** Blocks attempts to approximate high-complexity algebraic patterns using low-degree/low-complexity tools. Essential for: cryptographic hardness, complexity lower bounds, and geometric obstructions.

**Literature:** Bézout's theorem {cite}`Fulton84`; intersection theory {cite}`EisenbudHarris16`; algebraic geometry {cite}`Hartshorne77`; elimination theory {cite}`CoxLittleOShea15`; positivity {cite}`Lazarsfeld04`.

:::

:::{prf:proof} E12 Backend A (Hypersurface Form)
:label: proof-e12-backend-a

*Step 1 (Hypersurface Setup).* Let $V = Z(f)$ where $f$ is an irreducible homogeneous polynomial of degree $\delta$. The degree of $V$ as a variety equals $\delta$ (a generic line intersects $V$ in $\delta$ points by Bézout).

*Step 2 (Defining Equation Characterization).* A polynomial $g$ defines the same hypersurface ($Z(g) = V$) if and only if $g$ and $f$ have the same irreducible factors (up to units). Since $f$ is irreducible, $Z(g) = V$ implies $\sqrt{(g)}^{\mathrm{sat}} = (f)$ in the homogeneous coordinate ring, where $(-)^{\mathrm{sat}}$ denotes saturation by the irrelevant ideal $(x_0, \ldots, x_n)$. (In the affine case, saturation is automatic.)

*Step 3 (Degree Lower Bound via Irreducibility).* Since $Z(g) = Z(f) = V$, the radical ideals coincide: $\sqrt{(g)} = \sqrt{(f)}$. Because $f$ is irreducible, $(f)$ is a prime ideal, so $\sqrt{(f)} = (f)$. Hence $g \in \sqrt{(g)} = (f)$, which means $f | g$ (i.e., $g = f \cdot h$ for some polynomial $h$). Therefore:
$$\deg(g) = \deg(f) + \deg(h) \geq \deg(f) = \delta$$

*Step 4 (Sharpness).* The bound is achieved by $g = f$ itself. No polynomial of degree $< \delta$ can define $V$.

*Step 5 (Certificate Construction).* The obstruction: if $\mathbb{H}_{\mathrm{bad}}$ requires representing $V$ with $\deg < \delta$, this is impossible.

:::

:::{prf:proof} E12 Backend B (Complete Intersection Form)
:label: proof-e12-backend-b

*Step 1 (Complete Intersection Definition).* $V$ is a complete intersection if it is cut out by exactly $\text{codim}(V)$ equations and has the expected dimension. The ideal $I_V = (f_1, \ldots, f_k)$ is generated by a regular sequence.

*Step 2 (Degree via Bézout / Intersection Theory).* For a complete intersection:
$$\deg(V) = d_1 \cdot d_2 \cdots d_k$$
This follows from iterative application of Bézout's theorem {cite}`Fulton84` (Example 8.4.6).

*Step 3 (Representation Bounds).* Suppose $V = Z(g_1, \ldots, g_k)$ is another complete intersection representation with $\deg(g_i) = e_i$, **where $(g_1, \ldots, g_k)$ is also a regular sequence cutting out $V$ scheme-theoretically in expected codimension**. Then:
$$\deg(V) = e_1 \cdots e_k = d_1 \cdots d_k$$
The product of degrees is an invariant of the scheme structure.

*Step 4 (AM-GM Minimum Degree Constraint).* Among all complete intersection representations, the maximum single-equation degree satisfies:
$$\max_i(e_i) \geq \deg(V)^{1/k}$$
by AM-GM. If $d_1 \geq d_2 \geq \cdots \geq d_k$, then $d_1 \geq \deg(V)^{1/k}$.

*Step 5 (Certificate Construction).* The obstruction: if all $e_i < \deg(V)^{1/k}$, then $e_1 \cdots e_k < \deg(V)$, contradiction. Cannot uniformly lower defining degrees.

:::

:::{prf:proof} E12 Backend C (Morphism / Compression Form)
:label: proof-e12-backend-c

*Step 1 (Morphism Degree Definition).* For a generically finite morphism $\phi: W \to V$, the **degree** $d_\phi$ is the generic fiber cardinality: $d_\phi = |\phi^{-1}(p)|$ for generic $p \in V$.

*Step 2 (Projection Formula).* For a finite morphism $\phi: W \to V$ of degree $d_\phi$:
$$\deg(V) \cdot d_\phi = \deg(\phi^* H^{\dim V})$$
More directly: $\deg(V) \leq d_\phi \cdot \deg(W)$ with equality for finite morphisms.

*Step 3 (Degree Bound for Images).* By permit $K_{\mathrm{DegImage}_m}^+$ (degree-of-image bound, Definition {prf:ref}`def-permit-degimage`), after resolving indeterminacy (or using the graph), if $\phi$ is induced by a base-point-free linear system of degree $\leq m$, then:
$$\deg(\overline{\phi(W)}) \leq m^{\dim W} \cdot \deg(W)$$
The permit payload specifies whether $\phi$ is treated as a morphism $W \to \mathbb{P}^N$ or a rational map with resolved base locus.

*Step 4 (Compression Obstruction).* Suppose we want to represent $V$ (degree $\delta$) as $\phi(W)$ where $\deg(W) = w < \delta$ and $\phi$ has degree $\leq m$. Then:
$$\delta = \deg(V) \leq m^d \cdot w < m^d \cdot \delta$$
This is only possible if $m^d \geq \delta/w > 1$, hence $m \geq (\delta/w)^{1/d}$.

*Step 5 (Certificate Construction).* The morphism complexity lower bound:
$$m_{\min} = \left(\frac{\delta}{\deg(W)}\right)^{1/\dim V}$$
Any compression must have complexity at least $m_{\min}$.

:::

:::{div} feynman-prose
E12 uses algebraic degree as an obstruction. A high-degree algebraic variety cannot be faithfully represented by low-degree polynomials. This is not about approximation; it is about exact representation. Bezout's theorem says the product of degrees is an invariant. You cannot cheat it.

The three backends (A, B, C) handle different cases: hypersurfaces where a single polynomial defines everything, complete intersections where multiple equations cut out the variety, and morphisms where you try to compress via a map from a simpler space. Each backend has its own version of the degree obstruction.

This brings us to E13, which is not a specific tactic but the exhaustive check: have we tried everything? If all thirteen tactics fail to prove Hom-emptiness but also fail to construct an explicit morphism, we route to reconstruction rather than declaring victory or defeat.
:::

**Summary: The Thirteen Exclusion Tactics**

| Tactic | Name | Required Permits | Produces | Blocks |
|--------|------|------------------|----------|--------|
| E1 | Dimension | $\mathrm{Rep}_K$, $\mathrm{Cat}_{\mathrm{Hom}}$ | $K_{\mathrm{E1}}^+ \Rightarrow K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$ | All (dim mismatch) |
| E2 | Invariant | $\mathrm{Rep}_K$, $\mathrm{TB}_\pi$, $\mathrm{Cat}_{\mathrm{Hom}}$ | $K_{\mathrm{E2}}^+ \Rightarrow K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$ | All (invariant) |
| E3 | Positivity | $D_E$, $\mathrm{LS}_\sigma$, $\mathrm{Cat}_{\mathrm{Hom}}$ | $K_{\mathrm{E3}}^+ \Rightarrow K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$ | All (cone) |
| E4 | Integrality | $\mathrm{Rep}_K$, $\mathrm{Cat}_{\mathrm{Hom}}$ | $K_{\mathrm{E4}}^+ \Rightarrow K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$ | All (lattice) |
| E5 | Functional Eq. | $\mathrm{Rep}_K$, $\mathrm{GC}_\nabla$, $\mathrm{Cat}_{\mathrm{Hom}}$ | $K_{\mathrm{E5}}^+ \Rightarrow K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$ | All (unsolvable) |
| E6 | Causal | $\mathrm{TB}_\pi$, $D_E$, $\mathrm{Cat}_{\mathrm{Hom}}$ | $K_{\mathrm{E6}}^+ \Rightarrow K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$ | All (CTC) |
| E7 | Thermodynamic | $D_E$, $\mathrm{SC}_\lambda$, $\mathrm{Cat}_{\mathrm{Hom}}$ | $K_{\mathrm{E7}}^+ \Rightarrow K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$ | C.E (2nd Law) |
| E8 | DPI | $\mathrm{Cap}_H$, $\mathrm{TB}_\pi$, $\mathrm{Cat}_{\mathrm{Hom}}$ | $K_{\mathrm{E8}}^+ \Rightarrow K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$ | C.D (Capacity) |
| E9 | Ergodic | $\mathrm{TB}_\rho$, $C_\mu$, $\mathrm{Cat}_{\mathrm{Hom}}$ | $K_{\mathrm{E9}}^+ \Rightarrow K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$ | T.D (mixing) |
| E10 | Definability | $\mathrm{TB}_O$, $\mathrm{Rep}_K$, $\mathrm{Cat}_{\mathrm{Hom}}$ | $K_{\mathrm{E10}}^+ \Rightarrow K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$ | T.C (tameness) |
| E11 | Galois-Monodromy | $\mathrm{Rep}_K$, $\mathrm{TB}_\pi$, $\mathrm{Cat}_{\mathrm{Hom}}$ | $K_{\mathrm{E11}}^+ \Rightarrow K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$ | S.E (solvability) |
| E12 | Algebraic Compressibility | $\mathrm{Rep}_K$, $\mathrm{SC}_\lambda$, $\mathrm{Cat}_{\mathrm{Hom}}$ | $K_{\mathrm{E12}}^+ \Rightarrow K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$ | S.E (degree) |
| E13 | Algorithmic Completeness | $\mathrm{Rep}_K$, $\mathrm{Cat}_{\mathrm{Hom}}$ | $K_{\mathrm{E13}}^+ \Rightarrow K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$ | All (modality exhaust) |

:::{prf:definition} Breached-Inconclusive Certificate (Lock Tactic Exhaustion)
:label: def-lock-breached-inc

If all thirteen tactics fail to prove Hom-emptiness but also fail to construct an explicit morphism:

$$K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{br\text{-}inc}} = (\mathsf{tactics\_exhausted}: \{E1,\ldots,E13\}, \mathsf{partial\_progress}, \mathsf{trace})$$

This is a NO verdict (Breached) with inconclusive subtype—routing to {prf:ref}`mt-lock-reconstruction` (Structural Reconstruction) rather than fatal error. The certificate records which tactics were attempted and any partial progress (e.g., dimension bounds that narrowed but did not close, spectral gaps that are positive but not sufficient).

:::

:::{div} feynman-prose
And that is the Lock. Thirteen tactics, each based on a different mathematical obstruction, each producing a proof when it succeeds. If any one of them succeeds, the Hom-set is empty and the bad pattern cannot exist in your system. If all of them fail but none constructs an explicit embedding, we are in uncertain territory and need to try reconstruction.

The beauty of this design is that it leverages decades of mathematics: topology, algebra, analysis, information theory, thermodynamics, model theory. Each field contributes its characteristic obstruction. Together, they form a powerful filter.

This is not a heuristic. When the Lock produces a blocked certificate, you have a proof. The bad thing cannot happen because the mathematics forbids it.
:::

---
