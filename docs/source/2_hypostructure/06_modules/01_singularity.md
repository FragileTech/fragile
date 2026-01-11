# Part IX: Universal Singularity Modules

:::{div} feynman-prose feynman-added

Now we come to what I think is the most beautiful part of this whole framework. You have been wading through definitions and theorems, building up machinery. Here is where it pays off.

The question is: What happens when things go wrong? When your dynamical system develops a singularity---when energy concentrates, when gradients blow up, when the solution stops making sense---what do you *do*?

The standard answer in mathematics is: you work very hard. You classify all possible ways the system can fail, you prove theorems about each type of failure, you develop surgery procedures to cut out the bad parts and glue in good ones. This is what Perelman did for Ricci flow, what Huisken did for mean curvature flow, what Kenig and Merle did for dispersive equations. Brilliant work, but each case requires its own heroic effort.

Here is the key insight of this chapter: *we can automate all of that*. Not by being clever about each individual case, but by recognizing the pattern that underlies all of them. The singularity resolution problem has a universal structure, and once you see it, you can write theorems that work for *any* system that fits the pattern.

The user provides the physics---what quantities are conserved, how symmetries act, what the energy looks like. The Framework handles the singularity theory. That is the division of labor we are establishing here.

:::

**Design Philosophy:** These metatheorems act as **Factory Functions** that automatically implement the `ProfileExtractor` and `SurgeryOperator` interfaces. Users do not invent surgery procedures or profile classifiers---the Framework constructs them from the thin kernel objects.

:::{prf:remark} Factory Function Pattern
:label: rem-factory-pattern

The Universal Singularity Modules implement a **dependency injection** pattern:

| Interface | Factory Metatheorem | Input | Output |
|-----------|---------------------|-------|--------|
| `ProfileExtractor` | {prf:ref}`mt-resolve-profile` | $G^{\text{thin}}, \Phi^{\text{thin}}$ | Canonical library $\mathcal{L}_T$ |
| `SurgeryAdmissibility` | {prf:ref}`mt-resolve-admissibility` | $\mu, \mathfrak{D}^{\text{thin}}$ | Admissibility predicate |
| `SurgeryOperator` | {prf:ref}`mt-act-surgery` | Full $\mathcal{H}$ | Pushout surgery $\mathcal{O}_S$ |

**Key Insight:** Given thin objects satisfying the consistency conditions of {prf:ref}`mt-resolve-expansion`, these factories produce valid implementations for all required interfaces. The user's task reduces to specifying the physics (energy, dissipation, symmetry); the Framework handles the singularity theory.
:::

:::{prf:definition} Automation Guarantee
:label: def-automation-guarantee

A Hypostructure $\mathcal{H}$ satisfies the **Automation Guarantee** if:

1. **Profile extraction is automatic:** Given any singularity point $(t^*, x^*)$, the Framework computes the profile $V$ without user intervention via scaling limit:

   $$
   V = \lim_{\lambda \to 0} \lambda^{-\alpha} \cdot x(t^* + \lambda^2 t, x^* + \lambda y)
   $$

2. **Surgery construction is automatic:** Given admissibility certificate $K_{\text{adm}}$, the Framework constructs the surgery operator $\mathcal{O}_S$ as a categorical pushout.

3. **Termination is guaranteed:** The surgery sequence either:
   - Terminates (global regularity achieved), or
   - Reaches a horizon (irreducible singularity), or
   - Has bounded count (finite surgeries per unit time)

**Type Coverage:**
- For types $T \in \{T_{\text{parabolic}}, T_{\text{dispersive}}, T_{\text{hyperbolic}}\}$: The Automation Guarantee holds whenever the thin objects are well-defined.
- For $T_{\text{algorithmic}}$: The guarantee holds when the complexity measure $\mathcal{C}$ is well-founded (decreases with each step). In this case:
  - "Profiles" are fixed points or limit cycles of the discrete dynamics
  - "Surgery" is state reset or backtracking
  - "Termination" follows from well-foundedness of $\mathcal{C}$
- For $T_{\text{Markov}}$: The guarantee holds when the spectral gap is positive. Profiles are stationary distributions; surgery is measure truncation.

**Non-PDE Convergence Criteria:** The Łojasiewicz-Simon condition used in PDE applications can be replaced by:
- **Algorithmic:** Discrete Lyapunov functions with $\mathcal{C}(x') < \mathcal{C}(x)$
- **Markov:** Spectral gap $\lambda_1 > 0$ implies exponential mixing
- **Dynamical systems:** Contraction mappings with Lipschitz constant $L < 1$
:::

---

(sec-profile-classification-trichotomy)=
## Profile Classification Trichotomy

:::{div} feynman-prose feynman-added

Here is the first key question: when something goes wrong, *what kind* of thing is going wrong?

Think about it this way. You are watching a solution evolve, and it starts to develop a singularity---some quantity is blowing up at a point. You zoom in on that point, rescaling space and time to keep the interesting behavior at order one. What do you see?

The remarkable fact---and this took decades of hard analysis to establish---is that you do not see just *anything*. The zoomed-in picture, the "profile," belongs to a surprisingly small catalog of possibilities. For Ricci flow, you get cylinders, spheres, or Bryant solitons. For mean curvature flow, you get shrinking spheres and cylinders. For the nonlinear Schrodinger equation, you get ground states and their relatives.

Why should this be? Because the rescaled dynamics have to be *self-similar*---they have to look the same at every scale. That is a very restrictive condition. It forces the profile to be a special object, typically a stationary solution or a self-similar blowup of the underlying equation.

The trichotomy in this theorem captures the three possibilities:
1. Your profile is in the "canonical library"---a finite, pre-computed list of known profiles.
2. Your profile is in a "tame family"---an infinite but well-behaved parametric family.
3. Your profile is genuinely wild---chaotic, undecidable, beyond classification.

Cases 1 and 2 are where we can make progress. Case 3 is where we honestly say "this is too hard" and route to a different mode.

:::

:::{prf:theorem} [RESOLVE-Profile] Profile Classification Trichotomy
:label: mt-resolve-profile

**Rigor Class:** L (Literature-Anchored) — see {prf:ref}`def-rigor-classification`

**Bridge Verification:**
1. *Hypothesis Translation:* Certificates $K_{D_E}^+ \wedge K_{C_\mu}^+$ imply "bounded sequence in $\dot{H}^{s_c}(\mathbb{R}^n)$ with concentration at scale $\lambda_n \to 0$"
2. *Domain Embedding:* $\iota: \mathbf{Hypo}_T \to L^p(\mathbb{R}^n)$ via Sobolev embedding with critical exponent $p = 2n/(n-2s_c)$
3. *Conclusion Import:* Lions' profile decomposition {cite}`Lions84` $\Rightarrow K_{\text{lib}}^+$ (finite library) or $K_{\text{strat}}^+$ (tame family)

At the Profile node (after CompactCheck YES), the framework produces exactly one of three certificates:

**Case 1: Finite library membership**

$$
K_{\text{lib}} = (V, \text{canonical list } \mathcal{L}, V \in \mathcal{L})
$$

The limiting profile $V$ belongs to a finite, pre-classified library $\mathcal{L}$ of canonical profiles. Each library member has known properties enabling subsequent checks.

**Case 2: Tame stratification**

$$
K_{\text{strat}} = (V, \text{definable family } \mathcal{F}, V \in \mathcal{F}, \text{stratification data})
$$

Profiles are parameterized in a definable (o-minimal) family $\mathcal{F}$ with finite stratification. Classification is tractable though not finite.

**Case 3: Classification Failure (NO-inconclusive or NO-wild)**

$$
K_{\mathrm{prof}}^- := K_{\mathrm{prof}}^{\mathrm{wild}} \sqcup K_{\mathrm{prof}}^{\mathrm{inc}}
$$

- **NO-wild** ($K_{\mathrm{prof}}^{\mathrm{wild}}$): Profile exhibits wildness witness (chaotic attractor, turbulent cascade, undecidable structure)
- **NO-inconclusive** ($K_{\mathrm{prof}}^{\mathrm{inc}}$): Classification methods exhausted without refutation (Rep/definability constraints insufficient)

Routes to T.C/D.C-family modes for reconstruction or explicit wildness acknowledgment.

**Literature:** Concentration-compactness profile decomposition {cite}`Lions84`; {cite}`Lions85`; blow-up profile classification {cite}`MerleZaag98`; o-minimal stratification {cite}`vandenDries98`.

:::

:::{prf:proof}
Proof Sketch

*Step 1 (Limit Extraction).* Given singularity sequence $(t_n, x_n) \to (T_*, x_*)$ with $t_n \nearrow T_*$, apply compactness modulo symmetry (from $C_\mu$): there exist $g_n \in G$ such that $g_n \cdot u(t_n)$ has a convergent subsequence $\to V$.

*Step 2 (Classification Attempt).* Query the profile library $\mathcal{L}_T$:
- If $V \in \mathcal{L}_T$: Case 1 (finite library).
- Else, query the definable family $\mathcal{F}_T$ using o-minimal cell decomposition {cite}`vandenDries98`:
- If $V \in \mathcal{F}_T$: Case 2 (tame stratification).

*Step 3 (Failure Modes).* If neither holds:
- Check for wildness witnesses (positive Lyapunov exponent, undecidability): Case 3a (NO-wild).
- If no witness but methods exhausted: Case 3b (NO-inconclusive).

The trichotomy is exhaustive since $V$ either belongs to a classifiable family or demonstrates classifiability failure.
:::

:::{prf:remark} Library examples by type
:label: rem-library-examples

- $T_{\text{parabolic}}$: Cylinders, spheres, Bryant solitons (Ricci); spheres, cylinders (MCF)
- $T_{\text{dispersive}}$: Ground states, traveling waves, multi-solitons
- $T_{\text{algorithmic}}$: Fixed points, limit cycles, strange attractors
:::

:::{prf:remark} Oscillating and Quasi-Periodic Profiles
:label: rem-oscillating-profiles

**Edge Case:** The scaling limit $V = \lim_{n \to \infty} V_n$ may fail to converge in systems with oscillating or multi-scale behavior. Such systems are handled as follows:

**Case 2a (Periodic oscillations):** If the sequence $\{V_n\}$ has periodic or quasi-periodic structure:

$$
V_{n+p} \approx V_n \quad \text{for some period } p
$$

then the profile $V$ is defined as the **orbit** $\{V_n\}_{n \mod p}$, which falls into Case 2 (Tame Family) with a finite-dimensional parameter space $\mathbb{Z}/p\mathbb{Z}$ or $\mathbb{T}^k$ (torus for quasi-periodic).

**Case 3a (Wild oscillations):** If oscillations are unbounded or aperiodic without definable structure, the system produces a NO-wild certificate ($K_{\mathrm{prof}}^{\mathrm{wild}}$, Case 3). This is common in:
- Turbulent cascades (energy spreads across all scales)
- Chaotic attractors with positive Lyapunov exponent
- Undecidable algorithmic dynamics

**Practical consequence:** For well-posed physical systems, periodic/quasi-periodic profiles are typically tame. Wild oscillations indicate genuine physical complexity (turbulence) or computational irreducibility.
:::

:::{prf:definition} Moduli Space of Profiles
:label: def-moduli-profiles

The **Moduli Space of Profiles** for type $T$ is:

$$
\mathcal{M}_{\text{prof}}(T) := \{V : V \text{ is a scaling-invariant limit of type } T \text{ flow}\} / \sim
$$

where $V_1 \sim V_2$ if related by symmetry action: $V_2 = g \cdot V_1$ for $g \in G$.

**Structure:**
- $\mathcal{M}_{\text{prof}}$ is a (possibly infinite-dimensional) moduli stack
- The Canonical Library $\mathcal{L}_T \subset \mathcal{M}_{\text{prof}}(T)$ consists of **isolated points** with trivial automorphism
- The Tame Family $\mathcal{F}_T$ consists of **definable strata** parameterized by finite-dimensional spaces

**Computation:** Given $G^{\text{thin}} = (\text{Grp}, \rho, \mathcal{S})$ and $\Phi^{\text{thin}} = (F, \nabla, \alpha)$:

$$
\mathcal{M}_{\text{prof}}(T) = \{V : \mathcal{S} \cdot V = V, \nabla F(V) = 0\} / \text{Grp}
$$

:::

### Implementation in Sieve

:::{prf:remark} Profile Extraction Algorithm
:label: rem-profile-extraction

The Framework implements `ProfileExtractor` as follows:

**Input:** Singularity point $(t^*, x^*)$, thin objects $(\mathcal{X}^{\text{thin}}, \Phi^{\text{thin}}, G^{\text{thin}})$

**Algorithm:**
1. **Rescaling:** For sequence $\lambda_n \to 0$, compute:

   $$
   V_n := \lambda_n^{-\alpha} \cdot x(t^* + \lambda_n^2 t, x^* + \lambda_n y)
   $$

2. **Compactification:** Apply CompactCheck ($\mathrm{Cap}_H$) to verify subsequence converges

3. **Limit Extraction:** Extract $V = \lim_{n \to \infty} V_n$ in appropriate topology

4. **Library Lookup:**
   - If $V \in \mathcal{L}_T$: Return Case 1 certificate $K_{\text{lib}}$
   - If $V \in \mathcal{F}_T \setminus \mathcal{L}_T$: Return Case 2 certificate $K_{\text{strat}}$
   - If classification fails: Return Case 3 certificate $K_{\text{hor}}$

**Output:** Profile $V$ with classification certificate
:::

:::{prf:theorem} [RESOLVE-AutoProfile] Automatic Profile Classification (Multi-Mechanism OR-Schema)
:label: mt-resolve-auto-profile

**Sieve Target:** ProfileExtractor / Profile Classification Trichotomy

**Goal Certificate:** $K_{\mathrm{prof}}^+ \in \{K_{\text{lib}}, K_{\text{strat}}, K_{\text{hor}}\}$

For any Hypostructure $\mathcal{H} = (\mathcal{X}, \Phi, \mathfrak{D}, G)$ satisfying the Automation Guarantee (Definition {prf:ref}`def-automation-guarantee`), the Profile Classification Trichotomy (MT {prf:ref}`mt-resolve-profile`) is **automatically computed** by the Sieve without user-provided classification code.

### Unified Output Certificate

**Profile Classification Certificate:**

$$
K_{\mathrm{prof}}^+ := (V, \mathcal{L}_T \text{ or } \mathcal{F}_T, \mathsf{route\_tag}, \mathsf{classification\_data})
$$

where $\mathsf{route\_tag} \in \{\text{CC-Rig}, \text{Attr-Morse}, \text{Tame-LS}, \text{Lock-Excl}\}$ indicates which mechanism produced the certificate.

**Downstream Independence:** All subsequent theorems (Lock promotion, surgery admissibility, etc.) depend only on $K_{\mathrm{prof}}^+$, never on which mechanism produced it.

---

### Public Signature (Soft Interfaces Only)

**User-Provided (Soft Core):**

$$
K_{D_E}^+ \wedge K_{C_\mu}^+ \wedge K_{\mathrm{SC}_\lambda}^+ \wedge K_{\mathrm{LS}_\sigma}^+
$$

**Mechanism-Specific Soft Extensions:**
| Mechanism | Additional Soft Interfaces |
|-----------|---------------------------|
| A: CC+Rigidity | $K_{\mathrm{Mon}_\phi}^+ \wedge K_{\mathrm{Rep}_K}^+$ |
| B: Attractor+Morse | $K_{\mathrm{TB}_\pi}^+$ |
| C: Tame+LS | $K_{\mathrm{TB}_O}^+$ (o-minimal definability) |
| D: Lock/Hom-Exclusion | $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$ (Lock blocked) |

**Certificate Logic (Multi-Mechanism Disjunction):**

$$
\underbrace{K_{D_E}^+ \wedge K_{C_\mu}^+ \wedge K_{\mathrm{SC}_\lambda}^+ \wedge K_{\mathrm{LS}_\sigma}^+}_{\text{SoftCore}} \wedge \big(\text{MechA} \lor \text{MechB} \lor \text{MechC} \lor \text{MechD}\big) \Rightarrow K_{\mathrm{prof}}^+
$$

**Unified Proof (5 Steps):**

*Step 1 (OR-Schema Soundness).* The multi-mechanism disjunction is sound: if ANY mechanism $M \in \{A, B, C, D\}$ produces $K_{\mathrm{prof}}^+$, then the profile classification is valid. Each mechanism's soundness is proven independently (see mechanism-specific proofs below), and the disjunction inherits soundness from its disjuncts.

*Step 2 (Mechanism Independence).* Each mechanism operates on different soft interface extensions. No mechanism depends on another mechanism's output—they are **parallel alternatives**, not sequential stages. This ensures no circular dependencies.

*Step 3 (Dispatcher Completeness).* For any hypostructure $\mathcal{H}$ satisfying the Automation Guarantee, at least one mechanism applies:
- If $\mathcal{H}$ has monotonicity ($K_{\mathrm{Mon}_\phi}^+$): Mechanism A applies
- If $\mathcal{H}$ has finite topology ($K_{\mathrm{TB}_\pi}^+$): Mechanism B applies
- If $\mathcal{H}$ is o-minimal definable ($K_{\mathrm{TB}_O}^+$): Mechanism C applies
- If $\mathcal{H}$ has Lock obstruction ($K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$): Mechanism D applies
The Automation Guarantee ensures at least one of these conditions holds for "good" types.

*Step 4 (Downstream Independence).* All subsequent theorems (surgery, Lock promotion) consume only the certificate $K_{\mathrm{prof}}^+$, never the mechanism tag. This is enforced by the certificate interface: downstream theorems pattern-match on the certificate type, not its provenance.

*Step 5 (Termination).* The dispatcher tries mechanisms in fixed order until success or exhaustion. Since each mechanism's evaluation terminates (by their respective proofs), and there are finitely many mechanisms (4), the dispatcher terminates.

**Key Architectural Point:** Backend permits ($K_{\mathrm{WP}}$, $K_{\mathrm{ProfDec}}$, $K_{\mathrm{KM}}$, $K_{\mathrm{Rigidity}}$, $K_{\mathrm{Attr}}$, $K_{\mathrm{MorseDecomp}}$) are **derived internally** via the Soft-to-Backend Compilation layer (Section {ref}`sec-soft-backend-compilation`), not required from users.

- **Produces:** $K_{\text{prof}}^+ \in \{K_{\text{lib}}, K_{\text{strat}}, K_{\text{hor}}\}$
- **Blocks:** Mode C.D (Geometric Collapse), Mode T.C (Labyrinthine), Mode D.C (Semantic Horizon)
- **Breached By:** Wild/undecidable dynamics, non-good types
:::

:::{prf:proof}

### Dispatcher Logic

The Sieve tries mechanisms in order until one succeeds:

```
try MechA(SoftCore); if YES → emit K_prof^+ (tag: CC-Rig)
else try MechB(SoftCore); if YES → emit K_prof^+ (tag: Attr-Morse)
else try MechC(SoftCore); if YES → emit K_prof^+ (tag: Tame-LS)
else try MechD(SoftCore); if YES → emit K_prof^+ (tag: Lock-Excl)
else emit NO with K_prof^inc (mechanism_failures: [A,B,C,D])
```

---

### Mechanism A: Concentration-Compactness + Rigidity

**Best For:** NLS, NLW, critical dispersive PDEs

**Sufficient Soft Condition:**

$$
K_{D_E}^+ \wedge K_{C_\mu}^+ \wedge K_{\mathrm{SC}_\lambda}^+ \wedge K_{\mathrm{LS}_\sigma}^+ \wedge K_{\mathrm{Mon}_\phi}^+ \wedge K_{\mathrm{Rep}_K}^+
$$

(proof-mt-resolve-auto-profile-mech-a)=
**Proof (5 Steps via Compilation):**

*Step A1 (Well-Posedness).* By MT-SOFT→WP (MT {prf:ref}`mt-fact-soft-wp`), derive $K_{\mathrm{WP}_{s_c}}^+$ from template matching. The evaluator recognizes the equation structure and applies the appropriate critical LWP theorem.

*Step A2 (Profile Decomposition).* By MT-SOFT→ProfDec (MT {prf:ref}`mt-fact-soft-profdec`), derive $K_{\mathrm{ProfDec}_{s_c,G}}^+$ from $K_{C_\mu}^+ \wedge K_{\mathrm{SC}_\lambda}^+$. Any bounded sequence $\{u_n\}$ in $\dot{H}^{s_c}$ admits:

$$
u_n = \sum_{j=1}^J g_n^{(j)} \cdot V^{(j)} + w_n^{(J)}
$$

with orthogonal symmetry parameters and vanishing remainder.

*Step A3 (Kenig-Merle Machine).* By MT-SOFT→KM (MT {prf:ref}`mt-fact-soft-km`), derive $K_{\mathrm{KM}_{\mathrm{CC+stab}}}^+$ from composition of $K_{\mathrm{WP}}^+ \wedge K_{\mathrm{ProfDec}}^+ \wedge K_{D_E}^+$. This extracts the minimal counterexample $u^*$ with:
- $\Phi(u^*) = E_c$ (critical energy threshold),
- Trajectory is **almost periodic modulo $G$**.

*Step A4 (Hybrid Rigidity).* By MT-SOFT→Rigidity (MT {prf:ref}`mt-fact-soft-rigidity`), derive $K_{\mathrm{Rigidity}_T}^+$ via the hybrid mechanism:
1. **Monotonicity:** $K_{\mathrm{Mon}_\phi}^+$ provides virial/Morawetz identity forcing dispersion or concentration.
2. **Łojasiewicz Closure:** $K_{\mathrm{LS}_\sigma}^+$ prevents oscillation near critical points.
3. **Lock Exclusion:** Any "bad" $u^*$ would embed a forbidden pattern; Lock blocks this.

Conclusion: almost-periodic solutions are either **stationary** (soliton/ground state) or **self-similar**.

*Step A5 (Emit Certificate).* Classify $u^*$ into $\mathcal{L}_T$:
- **Case 1 (Library):** $V \in \mathcal{L}_T$ isolated. Emit YES with $K_{\text{lib}} = (V, \mathcal{L}_T, \text{Aut}(V), \text{CC-Rig})$
- **Case 2 (Tame Stratification):** $V \in \mathcal{F}_T$ definable. Emit YES with $K_{\text{strat}} = (V, \mathcal{F}_T, \dim, \text{CC-Rig})$
- **Case 3 (Classification Failure):** Emit NO with $K_{\mathrm{prof}}^{\mathrm{wild}}$ (if wildness witness found) or $K_{\mathrm{prof}}^{\mathrm{inc}}$ (if method insufficient)

**Literature:** Concentration-compactness {cite}`Lions84`; profile decomposition {cite}`BahouriGerard99`; Kenig-Merle {cite}`KenigMerle06`; rigidity {cite}`DuyckaertsKenigMerle11`.

---

### Mechanism B: Attractor + Morse Decomposition

**Best For:** Reaction-diffusion, Navier-Stokes (bounded domain), MCF

**Sufficient Soft Condition:**

$$
K_{D_E}^+ \wedge K_{C_\mu}^+ \wedge K_{\mathrm{SC}_\lambda}^+ \wedge K_{\mathrm{LS}_\sigma}^+ \wedge K_{\mathrm{TB}_\pi}^+
$$

(proof-mt-resolve-auto-profile-mech-b)=
**Proof (4 Steps via Compilation):**

*Step B1 (Global Attractor).* By MT-SOFT→Attr (MT {prf:ref}`mt-fact-soft-attr`), derive $K_{\mathrm{Attr}}^+$ from $K_{D_E}^+ \wedge K_{C_\mu}^+ \wedge K_{\mathrm{TB}_\pi}^+$. The attractor $\mathcal{A}$ exists, is compact, invariant, and attracts bounded sets:

$$
\mathcal{A} := \bigcap_{t \geq 0} \overline{\bigcup_{s \geq t} S_s(\mathcal{X})}
$$

*Step B2 (Morse Decomposition).* By MT-SOFT→MorseDecomp (MT {prf:ref}`mt-fact-soft-morse`), derive $K_{\mathrm{MorseDecomp}}^+$ from $K_{\mathrm{Attr}}^+ \wedge K_{D_E}^+ \wedge K_{\mathrm{LS}_\sigma}^+$. For gradient-like systems, the attractor decomposes as:

$$
\mathcal{A} = \mathcal{E} \cup \bigcup_{\xi \in \mathcal{E}} W^u(\xi)
$$

where $\mathcal{E}$ is the equilibrium set. No periodic orbits exist (Lyapunov monotonicity).

*Step B3 (Profile Identification).* The profile space is:

$$
\mathcal{M}_{\text{prof}} = \mathcal{A} / G
$$

By compactness of $\mathcal{A}$, this is a compact moduli space. The canonical library is:

$$
\mathcal{L}_T := \{\xi \in \mathcal{E} / G : \xi \text{ isolated}, |\text{Stab}(\xi)| < \infty\}
$$

*Step B4 (Emit Certificate).* Classify rescaling limits into $\mathcal{A}/G$:
- **Case 1 (Library):** Isolated equilibrium. Emit YES with $K_{\text{lib}} = (V, \mathcal{L}_T, \text{Morse index}, \text{Attr-Morse})$
- **Case 2 (Tame Stratification):** Connecting orbit. Emit YES with $K_{\text{strat}} = (V, W^u(\xi)/G, \dim, \text{Attr-Morse})$
- **Case 3 (Classification Failure):** Strange attractor detected. Emit NO with $K_{\mathrm{prof}}^{\mathrm{wild}} = (\text{strange\_attractor}, h_{\text{top}}(\mathcal{A}))$

**Literature:** Global attractor theory {cite}`Temam97`; gradient-like structure {cite}`HaleBook88`; Morse decomposition {cite}`Conley78`.

---

### Mechanism C: Tame + Łojasiewicz (O-Minimal Types)

**Best For:** Algebraic/analytic systems, polynomial nonlinearities

**Sufficient Soft Condition:**

$$
K_{D_E}^+ \wedge K_{C_\mu}^+ \wedge K_{\mathrm{LS}_\sigma}^+ \wedge K_{\mathrm{TB}_O}^+
$$

(proof-mt-resolve-auto-profile-mech-c)=
**Proof (3 Steps):**

*Step C1 (Definability).* By $K_{\mathrm{TB}_O}^+$, the profile space $\mathcal{M}_{\text{prof}}$ is **o-minimal definable** in the structure $\mathbb{R}_{\text{an}}$ (or $\mathbb{R}_{\text{alg}}$ for polynomial systems). This captures all algebraic, semialgebraic, and globally subanalytic families.

*Step C2 (Cell Decomposition).* By the o-minimal cell decomposition theorem, the profile space admits a **finite stratification**:

$$
\mathcal{M}_{\text{prof}} = \bigsqcup_{i=1}^N C_i
$$

where each $C_i$ is a definable cell (diffeomorphic to $(0,1)^{d_i}$). The stratification is canonical and computable from the defining formulas.

*Step C3 (Łojasiewicz Convergence + Emit).* By $K_{\mathrm{LS}_\sigma}^+$, trajectories converge to strata (no oscillation across cells). Emit:
- **Case 1 (Library):** Limit in 0-dimensional stratum. Emit YES with $K_{\text{lib}} = (V, \mathcal{L}_T, \text{cell ID}, \text{Tame-LS})$
- **Case 2 (Tame Stratification):** Limit in positive-dimensional stratum. Emit YES with $K_{\text{strat}} = (V, C_i, \dim C_i, \text{Tame-LS})$
- **Case 3 (Classification Failure):** Non-definable family (escape from o-minimal). Emit NO with $K_{\mathrm{prof}}^{\mathrm{wild}} = (\text{non\_definable}, \mathsf{escape\_witness})$

**Key Advantage:** No PDE-specific machinery required—works purely from definability + gradient structure.

**Literature:** O-minimal structures {cite}`vandenDries98`; tame geometry {cite}`Shiota97`; Łojasiewicz inequality {cite}`Lojasiewicz84`.

---

### Mechanism D: Lock / Hom-Exclusion (Categorical Types)

**Best For:** Systems where categorical obstruction is stronger than analytic classification

**Sufficient Soft Condition:**

$$
K_{D_E}^+ \wedge K_{C_\mu}^+ \wedge K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}
$$

(proof-mt-resolve-auto-profile-mech-d)=
**Proof (2 Steps):**

*Step D1 (Lock Obstruction).* By $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$, the Lock mechanism certifies:

$$
\mathrm{Hom}(\mathbb{H}_{\mathrm{bad}}, \mathcal{H}) = \emptyset
$$

for all "bad patterns" $\mathbb{H}_{\mathrm{bad}}$ (singularity templates, wild dynamics markers). This is a **categorical statement**: no morphism from any forbidden object can land in the hypostructure.

*Step D2 (Emit Trivial Classification).* Since no singularity can form (Lock blocks all singular behavior), profile classification is **vacuous or trivial**:
- All solutions remain regular
- The "library" is just the space of smooth solutions
- Emit $K_{\text{lib}} = (\text{smooth}, \mathcal{L}_T := \emptyset, \text{vacuous}, \text{Lock-Excl})$

Alternatively, if Lock blocks specific patterns but allows others, classify the allowed profiles as in other mechanisms.

**Key Advantage:** No hard estimates needed—regularity follows from **categorical obstruction** rather than analytic a priori bounds.

**Literature:** Lock mechanism (Section {ref}`sec-lock`); categorical obstructions in PDE {cite}`Fargues21`.

---

### Mechanism Comparison

| Mechanism | Additional Soft | Best For | Hard Estimates? | Route Tag |
|-----------|-----------------|----------|-----------------|-----------|
| **A: CC+Rig** | $K_{\mathrm{Mon}_\phi}^+$, $K_{\mathrm{Rep}_K}^+$ | NLS, NLW, dispersive | No (compiled) | CC-Rig |
| **B: Attr+Morse** | $K_{\mathrm{TB}_\pi}^+$ | Reaction-diffusion, MCF | No (gradient-like) | Attr-Morse |
| **C: Tame+LS** | $K_{\mathrm{TB}_O}^+$ | Algebraic, polynomial | No (definability) | Tame-LS |
| **D: Lock** | $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$ | Categorical systems | No (obstruction) | Lock-Excl |

**Mechanism Selection:** The Sieve automatically selects the first applicable mechanism based on which soft interfaces are available. Users may also specify a preferred mechanism via the `route_hint` parameter.

:::

---

(sec-surgery-admissibility-trichotomy)=
## Surgery Admissibility Trichotomy

:::{div} feynman-prose feynman-added

Now, here is the second key question: once you know what kind of singularity you have, can you actually *fix* it?

Not every singularity can be surgically repaired. Think about cutting out a tumor---you can only do it if the tumor is localized enough that removing it does not kill the patient. Same principle here.

The admissibility check is asking three questions:

First, is the profile in our catalog? If we know what kind of singularity this is---if it matches one of our canonical profiles---then we know exactly how to cap it off. If the profile is something wild and unclassifiable, we are stuck.

Second, is the singular set small enough? The technical condition is "codimension at least 2," which means the singularity is thin---like a curve in 3D space rather than a surface. Thin singularities can be excised without global damage. Fat singularities cannot.

Third, is the energy cost bounded? Surgery releases energy. If the surgery would cost an unbounded amount of energy, it violates conservation and we cannot do it. The capacity bound $\text{Cap}(\Sigma) \leq \varepsilon_{\text{adm}}$ ensures the energy cost is controlled.

When all three conditions are met, we get an "admissible" certificate and can proceed. When they are not, we either find an equivalent admissible surgery (Case 2) or acknowledge that this singularity is beyond our ability to repair (Case 3).

:::

:::{prf:theorem} [RESOLVE-Admissibility] Surgery Admissibility Trichotomy
:label: mt-resolve-admissibility

Before invoking any surgery $S$ with mode $M$ and data $D_S$, the framework produces exactly one of three certificates:

**Case 1: Admissible**

$$
K_{\text{adm}} = (M, D_S, \text{admissibility proof}, K_{\epsilon}^+)
$$

The surgery satisfies:
1. **Canonicity**: Profile at surgery point is in canonical library
2. **Codimension**: Singular set has codimension $\geq 2$
3. **Capacity**: $\mathrm{Cap}(\text{excision}) \leq \varepsilon_{\text{adm}}$
4. **Progress Density**: Energy drop satisfies $\Delta\Phi_{\text{surg}} \geq \epsilon_T$ where $\epsilon_T > 0$ is the problem-specific discrete progress constant. The certificate $K_{\epsilon}^+$ witnesses this bound.

**Case 2: Admissible up to equivalence (YES$^\sim$)**

$$
K_{\text{adm}}^{\sim} = (K_{\text{equiv}}, K_{\text{transport}}, K_{\text{adm}}[\tilde{x}])
$$

After an admissible equivalence move, the surgery becomes admissible.

**Case 3: Not admissible**

$$
K_{\text{inadm}} = (\text{failure reason}, \text{witness})
$$

Explicit reason certificate:
- Capacity too large: $\mathrm{Cap}(\text{excision}) > \varepsilon_{\text{adm}}$
- Codimension too small: $\mathrm{codim} < 2$
- Horizon: Profile not classifiable (Case 3 of Profile Trichotomy)

**Literature:** Surgery admissibility in Ricci flow {cite}`Perelman03`; capacity and removable singularities {cite}`Federer69`; {cite}`EvansGariepy15`.

:::

:::{prf:proof}
Proof Sketch

*Step 1 (Canonicity Verification).* Given surgery data $(\Sigma, V)$, query profile library: is $V \in \mathcal{L}_T$? If yes, proceed. If $V \in \mathcal{F}_T \setminus \mathcal{L}_T$, check for equivalence move (YES$^\sim$). If $V \notin \mathcal{F}_T$, return Case 3 (Horizon).

*Step 2 (Codimension Bound).* Compute Hausdorff dimension: $\dim_H(\Sigma) \leq n - 2$ where $n = \dim(\mathcal{X})$. This follows from capacity bound: $\text{Cap}(\Sigma) < \infty$ implies $\dim_H(\Sigma) \leq n - 2$ by {cite}`Federer69`.

*Step 3 (Capacity Bound).* Verify $\text{Cap}(\Sigma) \leq \varepsilon_{\text{adm}}(T)$ using Sobolev capacity. Small capacity implies singularity is "removable" in the sense of Federer: surgery can excise and cap without global energy loss.

The trichotomy is exhaustive: all checks pass (Case 1), equivalence move available (Case 2), or some check fails (Case 3).
:::

:::{prf:definition} Canonical Library
:label: def-canonical-library

The **Canonical Library** for type $T$ is:

$$
\mathcal{L}_T := \{V \in \mathcal{M}_{\text{prof}}(T) : \text{Aut}(V) \text{ is finite}, V \text{ is isolated in } \mathcal{M}_{\text{prof}}\}
$$

**Properties:**
- $\mathcal{L}_T$ is finite for good types (parabolic, dispersive)
- Each $V \in \mathcal{L}_T$ has a **surgery recipe** $\mathcal{O}_V$ attached
- Library membership is decidable via gradient flow to critical points

**Examples by Type:**

| Type | Library $\mathcal{L}_T$ | Size |
|------|------------------------|------|
| $T_{\text{Ricci}}$ | $\{\text{Sphere}, \text{Cylinder}, \text{Bryant}\}$ | 3 |
| $T_{\text{MCF}}$ | $\{\text{Sphere}^n, \text{Cylinder}^k\}_{k \leq n}$ | $n+1$ |
| $T_{\text{NLS}}$ | $\{Q, Q_{\text{excited}}\}$ | 2 |
| $T_{\text{wave}}$ | $\{\text{Ground state}\}$ | 1 |
:::

:::{prf:remark} Good Types
:label: rem-good-types

A type $T$ is **good** if:
1. **Compactness:** Scaling limits exist in a suitable topology (e.g., weak convergence in $L^2$, Gromov-Hausdorff)
2. **Finite stratification:** $\mathcal{M}_{\text{prof}}(T)$ admits finite stratification into isolated points and tame families
3. **Constructible caps:** Asymptotic matching for surgery caps is well-defined (unique cap per profile)

**Good types:** $T_{\text{Ricci}}$, $T_{\text{MCF}}$, $T_{\text{NLS}}$, $T_{\text{wave}}$, $T_{\text{parabolic}}$, $T_{\text{dispersive}}$.

**Non-good types:** Wild/undecidable systems that reach Horizon modes. For such systems, the Canonical Library may be empty or infinite, and the Automation Guarantee (Definition {prf:ref}`def-automation-guarantee`) does not apply.

**Algorithmic types:** $T_{\text{algorithmic}}$ is good when the complexity measure $\mathcal{C}$ is well-founded (terminates in finite steps). In this case, "profiles" are limit cycles or fixed points of the discrete dynamics.
:::

### Implementation in Sieve

:::{prf:remark} Admissibility Check Algorithm
:label: rem-admissibility-algorithm

The Framework implements `SurgeryAdmissibility` as follows:

**Input:** Singularity data $(\Sigma, V, t^*)$, thin objects $(\mathcal{X}^{\text{thin}}, \mathfrak{D}^{\text{thin}})$

**Algorithm:**
1. **Canonicity Check:**
   - Query: Is $V \in \mathcal{L}_T$?
   - If YES: Continue. If NO (but $V \in \mathcal{F}_T$): Try equivalence move. If Horizon: Return Case 3.

2. **Codimension Check:**
   - Compute $\text{codim}(\Sigma)$ using dimension of $\mathcal{X}$
   - Require: $\text{codim}(\Sigma) \geq 2$

3. **Capacity Check:**
   - Compute $\text{Cap}(\Sigma)$ using measure $\mu$ from $\mathcal{X}^{\text{thin}}$
   - Require: $\text{Cap}(\Sigma) \leq \varepsilon_{\text{adm}}(T)$

**Decision:**
- All checks pass → Case 1: $K_{\text{adm}}$
- Canonicity fails but equivalence available → Case 2: $K_{\text{adm}}^\sim$
- Any check fails without recovery → Case 3: $K_{\text{inadm}}$

**Output:** Admissibility certificate
:::

:::{prf:theorem} [RESOLVE-AutoAdmit] Automatic Admissibility
:label: mt-resolve-auto-admit

For any Hypostructure satisfying the Automation Guarantee, the Surgery Admissibility Trichotomy is **automatically computed** from thin objects without user-provided admissibility code.

**Key Computation:** The capacity bound is computed as:

$$
\text{Cap}(\Sigma) = \inf\left\{\int |\nabla \phi|^2 \, d\mu : \phi|_\Sigma = 1, \phi \in H^1(\mathcal{X})\right\}
$$

using the measure $\mu$ from $\mathcal{X}^{\text{thin}}$ and the metric $d$.

**Literature:** Sobolev capacity {cite}`AdamsHedberg96`; Hausdorff dimension bounds {cite}`Federer69`.
:::

:::{prf:proof}
Proof Sketch

*Step 1 (Thin Object Extraction).* From $\mathcal{X}^{\text{thin}} = (\mathcal{X}, d, \mu)$, extract the metric structure for capacity computation. From $\mathfrak{D}^{\text{thin}} = (R, \beta)$, identify the singular locus $\Sigma = \{x : R(x) \to \infty\}$.

*Step 2 (Capacity Computation).* Apply the Sobolev capacity formula. For $\Sigma$ of finite capacity, {cite}`AdamsHedberg96` provides the estimate $\text{Cap}(\Sigma) \sim \text{Vol}(\Sigma)^{(n-2)/n}$ in dimension $n$.

*Step 3 (Trichotomy Application).* Feed $(\Sigma, V, \text{Cap}(\Sigma))$ into MT {prf:ref}`mt-resolve-admissibility`. The output determines admissibility without user-provided verification code.
:::

---

(sec-structural-surgery-principle)=
## Structural Surgery Principle

:::{div} feynman-prose feynman-added

Now we come to the surgery itself. This is where we actually cut and paste.

The key idea is categorical: surgery is a *pushout*. You have your original space with a bad region. You excise the bad region, leaving a boundary. You take a pre-fabricated "cap" that matches the boundary asymptotically. You glue them together. The universal property of pushouts guarantees the gluing is well-defined and unique.

What makes this work? Three things:

First, the cap is determined by the profile. Once you know you have a cylindrical singularity, there is exactly one cap that matches---a spherical cap that smoothly interpolates to the cylinder at infinity. The canonical library stores these caps alongside the profiles.

Second, the gluing respects energy. The surgery cannot create energy from nothing or lose track of energy. In fact, surgery *releases* energy---the singular region was holding excess energy, and removing it decreases the total. This is crucial for termination: if each surgery releases at least $\epsilon_T$ energy, and total energy is finite, there can only be finitely many surgeries.

Third, the surgered solution is smoother than what we started with. The singularity was precisely the locus of bad behavior; removing it and capping with a smooth cap leaves us with a better-behaved solution. We can then restart the flow and continue until the next singularity, or until we reach global regularity.

The beautiful thing is that none of this requires user input. The Framework computes the cap from the profile, constructs the pushout automatically, and verifies the energy and regularity bounds. The user specified the physics; the Framework handles the topology.

:::

:::{prf:theorem} [ACT-Surgery] Structural Surgery Principle (Certificate Form)
:label: mt-act-surgery

**Rigor Class:** L (Literature-Anchored) — see {prf:ref}`def-rigor-classification`

**Bridge Verification:**
1. *Hypothesis Translation:* Certificates $K^{\mathrm{br}} \wedge K_{\text{adm}}^+$ imply Perelman's surgery hypotheses: curvature pinching $R \geq \epsilon^{-1}$, canonical neighborhood structure, and $\delta$-neck existence
2. *Domain Embedding:* $\iota: \mathbf{Hypo}_T \to \mathbf{RicciFlow}$ mapping hypostructure state to 3-manifold with Ricci flow metric
3. *Conclusion Import:* Perelman's surgery theorem {cite}`Perelman03` $\Rightarrow K^{\mathrm{re}}$ (re-entry) with energy decrease $\Phi(x') \leq \Phi(x^-) - c \cdot \text{Vol}(\Sigma)^{2/n}$

Let $M$ be a failure mode with breach certificate $K^{\mathrm{br}}$, and let $S$ be the associated surgery with admissibility certificate $K_{\text{adm}}$ (or $K_{\text{adm}}^{\sim}$).

**Inputs**:
- $K^{\mathrm{br}}$: Breach certificate from barrier
- $K_{\text{adm}}$ or $K_{\text{adm}}^{\sim}$: From Surgery Admissibility Trichotomy
- $D_S$: Surgery data

**Guarantees**:
1. **Flow continuation**: Evolution continues past surgery with well-defined state $x'$
2. **Jump control**: $\Phi(x') \leq \Phi(x^-) + \delta_S$ for controlled jump $\delta_S$
3. **Certificate production**: Re-entry certificate $K^{\mathrm{re}}$ satisfying $K^{\mathrm{re}} \Rightarrow \mathrm{Pre}(\text{target})$
4. **Progress**: Either bounded surgery count or decreasing complexity

**Failure case**: If $K_{\text{inadm}}$ is produced, no surgery is performed; the run terminates at the mode as a genuine singularity (or routes to reconstruction via {prf:ref}`mt-lock-reconstruction`).

**Literature:** Hamilton's surgery program {cite}`Hamilton97`; Perelman's surgery algorithm {cite}`Perelman03`; {cite}`KleinerLott08`.

:::

:::{prf:proof}
Proof Sketch

*Step 1 (Excision).* Given admissible singularity $(\Sigma, V)$ with $\text{Cap}(\Sigma) \leq \varepsilon_{\text{adm}}$, remove neighborhood $\mathcal{X}_\Sigma = B_\epsilon(\Sigma)$. The removed region has controlled measure: $\mu(\mathcal{X}_\Sigma) \lesssim \epsilon^2 \cdot \text{Cap}(\Sigma)$.

*Step 2 (Capping).* Profile $V \in \mathcal{L}_T$ determines unique cap $\mathcal{X}_{\text{cap}}$ via asymptotic matching. The gluing is performed via pushout in the ambient category, preserving continuity/smoothness.

*Step 3 (Energy Control).* By {cite}`Perelman03`, the surgery releases energy: $\Phi(x') \leq \Phi(x^-) - c \cdot \text{Vol}(\Sigma)^{2/n}$. Combined with finite surgery count (bounded by energy budget), the flow terminates.

*Step 4 (Re-entry).* The surgered state $x' \in \mathcal{X}'$ satisfies the preconditions for continuing the Sieve: emit re-entry certificate $K^{\mathrm{re}}$ with updated energy bound.
:::

:::{prf:definition} Surgery Morphism
:label: def-surgery-morphism

A **Surgery Morphism** for singularity $(\Sigma, V)$ is a categorical pushout:

$$
\begin{CD}
\mathcal{X}_{\Sigma} @>{\iota}>> \mathcal{X} \\
@V{\text{excise}}VV @VV{\mathcal{O}_S}V \\
\mathcal{X}_{\text{cap}} @>{\text{glue}}>> \mathcal{X}'
\end{CD}
$$

where:
- $\mathcal{X}_\Sigma = \{x \in \mathcal{X} : d(x, \Sigma) < \epsilon\}$ is the singular neighborhood
- $\iota$ is the inclusion
- $\mathcal{X}_{\text{cap}}$ is a **capping object** determined by profile $V$
- $\mathcal{X}' = (\mathcal{X} \setminus \mathcal{X}_\Sigma) \sqcup_{\partial} \mathcal{X}_{\text{cap}}$ is the surgered space

**Universal Property:** For any morphism $f: \mathcal{X} \to \mathcal{Y}$ that annihilates $\Sigma$ (i.e., $f|_\Sigma$ factors through a point), there exists unique $\tilde{f}: \mathcal{X}' \to \mathcal{Y}$ with $\tilde{f} \circ \mathcal{O}_S = f$.

**Categorical Context:** The pushout is computed in the appropriate category determined by the ambient topos $\mathcal{E}$:
- **Top** (topological spaces): For continuous structure and homotopy type
- **Meas** (measure spaces): For measure $\mu$ and capacity computations
- **Diff** (smooth manifolds): For PDE applications with regularity
- **FinSet** (finite sets): For algorithmic/combinatorial applications

The transfer of structures ($\Phi', \mathfrak{D}'$) to $\mathcal{X}'$ uses the universal property: any structure on $\mathcal{X}$ that is constant on $\Sigma$ induces a unique structure on $\mathcal{X}'$.
:::

:::{div} feynman-prose feynman-added

Here is the theorem that makes the whole program work. You might worry: what if surgery goes on forever? What if we keep finding singularities, keep cutting and pasting, in an infinite regress?

The Conservation of Flow theorem says: *that cannot happen*. And here is why.

Every admissible surgery releases at least $\epsilon_T$ worth of energy. Not "some positive amount"---a *fixed* positive amount that depends only on the type of system. This is the discrete progress constant. It comes from a lower bound on the volume of admissible singularities: you cannot have singularities that are arbitrarily small, because tiny singularities would have infinite capacity-to-volume ratio.

Now count: you started with energy $\Phi(x_0)$. After $N$ surgeries, you have released at least $N \cdot \epsilon_T$ energy. But energy cannot go below $\Phi_{\min}$ (the ground state energy). So $N \leq (\Phi(x_0) - \Phi_{\min})/\epsilon_T$.

That is a finite number. A number you can compute. Not "eventually finite" or "well-founded in some abstract sense"---an actual integer that you can write down before you even start.

This is what distinguishes the framework from handwaving. We do not just say "the process terminates." We say exactly how many surgeries can possibly occur, and we prove it from the physics of the problem.

:::

:::{prf:theorem} [RESOLVE-Conservation] Conservation of Flow
:label: mt-resolve-conservation

For any admissible surgery $\mathcal{O}_S: \mathcal{X} \dashrightarrow \mathcal{X}'$, the following are conserved:

1. **Energy Drop (with Discrete Progress):**

   $$
   \Phi(x') \leq \Phi(x^-) - \Delta\Phi_{\text{surg}}
   $$

   where $\Delta\Phi_{\text{surg}} \geq \epsilon_T > 0$ is the **problem-specific discrete progress constant**. This bound follows from:
   - **Volume Lower Bound:** Admissible surgeries have $\text{Vol}(\Sigma) \geq v_{\min}(T)$ (excludes infinitesimal singularities)
   - **Isoperimetric Scaling:** $\Delta\Phi_{\text{surg}} \geq c \cdot \text{Vol}(\Sigma)^{(n-2)/n} \geq c \cdot v_{\min}^{(n-2)/n} =: \epsilon_T$
   The discrete progress constraint prevents Zeno surgery sequences.

2. **Regularization:**

   $$
   \sup_{\mathcal{X}'} |\nabla^k \Phi| < \infty \quad \text{for all } k \leq k_{\max}(V)
   $$

   The surgered solution has bounded derivatives (smoother than pre-surgery).

3. **Countability (Discrete Bound):**

   $$
   N_{\text{surgeries}} \leq \frac{\Phi(x_0) - \Phi_{\min}}{\epsilon_T}
   $$

   Since each surgery drops energy by at least $\epsilon_T > 0$, the surgery count is explicitly bounded. This is a finite natural number, not merely an abstract well-foundedness argument.
:::

:::{prf:proof}
Proof Sketch

*Step 1 (Energy Drop).* The excised region $\mathcal{X}_\Sigma$ contains concentrated curvature/energy. By the isoperimetric inequality and capacity bounds:

$$
\Delta\Phi_{\text{surg}} \geq c_n \cdot \text{Vol}(\Sigma)^{(n-2)/n} \cdot \sup_{\mathcal{X}_\Sigma} |\nabla^2 \Phi|
$$

Excision removes this energy permanently.

*Step 2 (Regularization).* The cap $\mathcal{X}_{\text{cap}}$ is chosen from the canonical library with bounded geometry. By asymptotic matching, the glued solution $\Phi'$ inherits the cap's regularity: $|\nabla^k \Phi'|_{\mathcal{X}_{\text{cap}}} \leq C_k(V)$ for all $k$.

*Step 3 (Countability).* Each surgery drops energy by at least $\delta_{\min} > 0$. Total surgery count satisfies:

$$
N \cdot \delta_{\min} \leq \Phi(x_0) - \Phi_{\min}
$$

Hence $N \leq (\Phi(x_0) - \Phi_{\min})/\delta_{\min} < \infty$.
:::

### Implementation in Sieve

:::{prf:remark} Surgery Operator Construction
:label: rem-surgery-construction

The Framework implements `SurgeryOperator` as follows:

**Input:** Admissibility certificate $K_{\text{adm}}$, profile $V \in \mathcal{L}_T$

**Algorithm:**
1. **Neighborhood Selection:**
   - Compute singular neighborhood $\mathcal{X}_\Sigma = \{d(x, \Sigma) < \epsilon(V)\}$
   - Verify $\text{Cap}(\mathcal{X}_\Sigma) \leq \varepsilon_{\text{adm}}$

2. **Cap Selection:**
   - Look up cap $\mathcal{X}_{\text{cap}}(V)$ from library $\mathcal{L}_T$
   - Each profile $V$ has a unique asymptotically-matching cap

3. **Pushout Construction:**
   - Form pushout $\mathcal{X}' = \mathcal{X} \sqcup_{\partial \mathcal{X}_\Sigma} \mathcal{X}_{\text{cap}}$
   - Transfer height $\Phi'$ and dissipation $\mathfrak{D}'$ to $\mathcal{X}'$

4. **Certificate Production:**
   - Produce re-entry certificate $K^{\text{re}}$ with:
     - New state $x' \in \mathcal{X}'$
     - Energy bound $\Phi(x') \leq \Phi(x^-) + \delta_S$
     - Regularity guarantee for post-surgery solution

**Output:** Surgered state $x' \in \mathcal{X}'$ with re-entry certificate
:::

:::{prf:theorem} [RESOLVE-AutoSurgery] Automatic Surgery
:label: mt-resolve-auto-surgery

For any Hypostructure satisfying the Automation Guarantee, the Structural Surgery Principle is **automatically executed** by the Sieve using the pushout construction from $\mathcal{L}_T$.

**Key Insight:** The cap $\mathcal{X}_{\text{cap}}(V)$ is uniquely determined by the profile $V$ via asymptotic matching. Users provide the symmetry group $G$ and scaling $\alpha$; the Framework constructs the surgery operator as a categorical pushout.

**Literature:** Pushouts in category theory {cite}`MacLane71`; surgery caps in geometric flows {cite}`Hamilton97`; {cite}`KleinerLott08`.
:::

:::{prf:proof}
Proof Sketch

*Step 1 (Cap Existence).* Given profile $V \in \mathcal{L}_T$ with finite automorphism group, asymptotic analysis determines a unique cap geometry matching $V$'s asymptotic expansion. For Ricci flow, this is the Bryant soliton; for MCF, this is the standard cylinder cap.

*Step 2 (Pushout Construction).* The surgery operator $\mathcal{O}_S$ is the categorical pushout along the boundary inclusion $\partial \mathcal{X}_\Sigma \hookrightarrow \mathcal{X}_{\text{cap}}$. By {cite}`MacLane71`, pushouts exist in the ambient category $\mathcal{E}$ whenever $\mathcal{E}$ has finite colimits.

*Step 3 (Automation).* Given thin objects $(\mathcal{X}^{\text{thin}}, G^{\text{thin}})$, the Framework:
- Computes $\mathcal{L}_T$ as fixed points of the scaling action
- Attaches cap $\mathcal{X}_{\text{cap}}(V)$ to each $V \in \mathcal{L}_T$
- Implements $\mathcal{O}_S$ as the pushout functor
:::

### Automated Workflow Summary

:::{prf:remark} Complete Automation Pipeline
:label: rem-automation-pipeline

The Universal Singularity Modules provide an **end-to-end automated pipeline**:

| Stage | Sieve Node | Input | Module | Output |
|-------|------------|-------|--------|--------|
| 1. Detect | {prf:ref}`def-node-compact` | Flow $x(t)$ | — | Singular point $(t^*, x^*)$ |
| 2. Profile | {prf:ref}`def-node-scale` | $(t^*, x^*)$ | {prf:ref}`mt-resolve-profile` | Profile $V$ with certificate |
| 3. Barrier | Mode Barrier | $V$ | Metatheorem FACT-Barrier | Breach certificate $K^{\text{br}}$ |
| 4. Admissibility | Pre-Surgery | $(\Sigma, V)$ | {prf:ref}`mt-resolve-admissibility` | Admissibility certificate |
| 5. Surgery | Surgery | $K_{\text{adm}}$ | {prf:ref}`mt-act-surgery` | Surgered state $x'$ |
| 6. Re-entry | Post-Surgery | $x'$ | {prf:ref}`mt-act-surgery` | Re-entry certificate $K^{\text{re}}$ |

**User Input:** Thin objects $(\mathcal{X}^{\text{thin}}, \Phi^{\text{thin}}, \mathfrak{D}^{\text{thin}}, G^{\text{thin}})$

**Framework Output:** Either:
- GlobalRegularity (no singularities)
- Classified Mode $M_i$ with certificates
- Horizon (irreducible singularity)

**Zero User Code for Singularity Handling:** The user never writes profile classification, admissibility checking, or surgery construction code.
:::

:::{prf:corollary} Minimal User Burden for Singularity Resolution
:label: cor-minimal-user-burden

Given thin objects satisfying the consistency conditions:
1. $(\mathcal{X}, d)$ is a complete metric space
2. $F: \mathcal{X} \to \mathbb{R} \cup \{\infty\}$ is lower semicontinuous
3. $R \geq 0$ and $\frac{d}{dt}F \leq -R$
4. $\rho: G \times \mathcal{X} \to \mathcal{X}$ is continuous

The Sieve automatically:
- Detects all singularities
- Classifies all profiles
- Determines all surgery admissibilities
- Constructs all surgery operators
- Bounds all surgery counts

**Consequence:** The "singularity problem" becomes a **typing problem**: specify the correct thin objects, and the Framework handles singularity resolution.
:::

:::{div} feynman-prose feynman-added

And there it is.

What Perelman did for Ricci flow in three dimensions, what Huisken did for mean curvature flow, what Kenig and Merle did for nonlinear dispersive equations---all of these monumental achievements in singularity resolution---the Framework does automatically, for any system that fits the pattern.

The user provides four things: a metric space where the dynamics live, an energy functional that decreases along trajectories, a dissipation mechanism that measures how fast energy is lost, and a symmetry group under which the dynamics are equivariant. From these four ingredients, the Framework derives the entire singularity resolution pipeline.

This is not a claim that singularity theory is easy. The theorems we invoke---Lions' concentration-compactness, Perelman's surgery algorithm, Kenig-Merle's rigidity---represent some of the deepest mathematics of the last half century. What we claim is that this deep mathematics has a *universal structure*, and that structure can be captured in abstract form and applied mechanically to new problems.

The singularity problem becomes a typing problem. Get the types right, and the theorems follow.

:::
