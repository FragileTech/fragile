# Part I: Categorical Foundations

:::{div} feynman-prose
Now, here is something that trips people up when they first encounter this material. They see "category theory" and "infinity-toposes" and think we are doing abstract nonsense for its own sake. But that is exactly backwards. The reason we use these tools is *precision*, not abstraction.

Think about it this way. When you have a physical system with symmetries, gauge redundancies, and configuration spaces that wrap around on themselves in interesting ways, you need a language that can talk about these structures without losing information. Set theory is like trying to describe a doughnut by listing all its points---you lose the essential fact that it has a hole. Category theory is the mathematics that remembers the hole.

The key insight of this chapter is simple: *singularities are morphisms*. When something blows up in a dynamical system, it is not random---it is following a pattern. And patterns can be classified. The categorical machinery lets us build a universal catalog of "all possible ways things can go wrong" and then check whether our specific system can access any of them.

If that catalog comes up empty for your system, you have proven global regularity. That is the punchline. Everything else is working out the details.
:::

(sec-ambient-substrate)=
## The Ambient Substrate

:::{div} feynman-prose
Before we can talk about singularities, we need to say precisely what kind of mathematical universe we are working in. It is like setting up a laboratory before doing an experiment---you need the right equipment.

The equipment here is a *cohesive infinity-topos*. Do not let the jargon scare you. A topos is just a mathematical universe where you can do logic and geometry at the same time. "Cohesive" means it has enough structure to talk about continuous paths and homotopy. "Infinity" means it tracks not just whether two things are equal, but *how* they are equal, and how those equalities relate to each other, all the way up.

Why do we need this? Because gauge symmetries are not just symmetries---they are equivalences with structure. A rotation by 360 degrees is equivalent to doing nothing, but it matters that you went around the full circle. The cohesive topos remembers this.
:::

To ensure robustness against deformation and gauge redundancies, we work within **Higher Topos Theory** and **Homotopy Type Theory (HoTT)**. This framework is strictly more expressive than ZFC set theory and naturally encodes the homotopical structure of configuration spaces.

:::{prf:definition} Ambient $\infty$-Topos
:label: def-ambient-topos

Let $\mathcal{E}$ be a **cohesive $(\infty, 1)$-topos** equipped with the shape/flat/sharp modality adjunction:
$$\Pi \dashv \flat \dashv \sharp : \mathcal{E} \to \infty\text{-Grpd}$$

The cohesion structure provides:
- **Shape** $\Pi$: Extracts the underlying homotopy type (fundamental $\infty$-groupoid)
- **Flat** $\flat$: Includes discrete $\infty$-groupoids as "constant" objects
- **Sharp** $\sharp$: Includes codiscrete objects (contractible path spaces)

Standard examples include the topos of smooth $\infty$-stacks $\mathbf{Sh}_\infty(\mathbf{CartSp})$ and differential cohesive types.
:::

:::{div} feynman-prose
What does this actually mean in practice? The three modalities---Shape, Flat, and Sharp---let you move between different levels of resolution. Shape extracts the topology (just the connectivity, forgetting the geometry). Flat lets you bring in "constants"---things that do not vary. Sharp lets you pretend everything is contractible (paths exist between any two points).

These are the zoom controls on your mathematical microscope. Sometimes you want to see the fine structure (the full topos), sometimes you just want the topology (apply Shape), and sometimes you need to compare with a baseline (Flat). The adjunctions ensure these operations play nicely together.
:::

(sec-hypostructure-object)=
## The Hypostructure Object

:::{div} feynman-prose
Now we come to the central character of this theory: the Hypostructure. This is the mathematical object that captures everything we need to know about a dynamical system to determine whether it can develop singularities.

Think of it as a blueprint that includes not just the state space, but also the rules of motion, the energy landscape, the constraints that must be satisfied, and the interface with the outside world. Each piece plays a specific role in the singularity analysis.

The key insight is that singularities do not happen in isolation. They involve the interplay of all these structures. A blow-up occurs when the dynamics push the system toward a configuration that violates the constraints. By packaging everything together, we can ask the categorical question: "Does this system contain the pattern that leads to blow-up?"
:::

A Hypostructure is not merely a set of equations, but a geometric object equipped with a connection and a filtration.

:::{prf:definition} Categorical Hypostructure
:label: def-categorical-hypostructure

A **Hypostructure** is a tuple $\mathbb{H} = (\mathcal{X}, \nabla, \Phi_\bullet, \tau, \partial_\bullet)$ where:

1. **State Stack** $\mathcal{X} \in \text{Obj}(\mathcal{E})$: The **configuration stack** representing all possible states. This is an $\infty$-sheaf encoding both the state space and its symmetries. The homotopy groups $\pi_n(\mathcal{X})$ capture:
   - $\pi_0$: Connected components (topological sectors)
   - $\pi_1$: Gauge symmetries and monodromy
   - $\pi_n$ ($n \geq 2$): Higher coherences and anomalies

2. **Flat Connection** $\nabla: \mathcal{X} \to T\mathcal{X}$: A section of the tangent $\infty$-bundle, encoding the dynamics as **parallel transport**. The semiflow $S_t$ is recovered as the exponential map:
   $$S_t = \exp(t \cdot \nabla): \mathcal{X} \to \mathcal{X}$$
   The flatness condition $[\nabla, \nabla] = 0$ ensures consistency of time evolution.

3. **Cohomological Height** $\Phi_\bullet: \mathcal{X} \to \mathbb{R}_\infty$: A **cohomological field theory** assigning to each state its energy/complexity. The notation $\Phi_\bullet$ indicates this is a **derived functor**—it comes equipped with higher coherences $\Phi_n$ for all $n$.

4. **Truncation Structure** $\tau = (\tau_C, \tau_D, \tau_{SC}, \tau_{LS})$: The axioms are realized as **truncation functors** on the homotopy groups of $\mathcal{X}$:
   - **Axiom C**: $\tau_C$ truncates unbounded orbits
   - **Axiom D**: $\tau_D$ bounds the energy filtration
   - **Axiom SC**: $\tau_{SC}$ constrains weight gradings
   - **Axiom LS**: $\tau_{LS}$ truncates unstable modes

5. **Boundary Morphism** $\partial_\bullet: \mathcal{X} \to \mathcal{X}_\partial$: A **restriction functor** to the boundary $\infty$-stack, representing the **Holographic Screen**—the interface between bulk dynamics and the external environment. Formally, $\partial_\bullet$ is the pullback along the inclusion $\iota: \partial\mathcal{X} \hookrightarrow \mathcal{X}$:
   $$\partial_\bullet := \iota^*: \mathbf{Sh}_\infty(\mathcal{X}) \to \mathbf{Sh}_\infty(\partial\mathcal{X})$$

   This structure satisfies:

   - **Stokes' Constraint (Differential Cohomology):** Let $\hat{\Phi} \in \hat{H}^n(\mathcal{X}; \mathbb{R})$ be the differential refinement of the energy class. The **integration pairing** satisfies:
     $$\langle d\hat{\Phi}, [\mathcal{X}] \rangle = \langle \hat{\Phi}, [\partial\mathcal{X}] \rangle$$
     where $d: \hat{H}^n \to \Omega^{n+1}_{\text{cl}}$ is the curvature map. This rigidly links internal dissipation to boundary flux via the **de Rham-Cheeger-Simons sequence**.

   - **Cobordism Interface:** For Surgery operations, $\partial_\bullet$ defines the gluing interface in the symmetric monoidal $(\infty,1)$-category $\mathbf{Bord}_n^{\text{or}}$. Given a cobordism $W: M_0 \rightsquigarrow M_1$, the boundary functor satisfies:
     $$\partial W \simeq M_0 \sqcup \overline{M_1} \quad \text{in } \mathbf{Bord}_n$$
     Surgery is a morphism in this category; gluing is composition.

   - **Holographic Bound (Two-Level Structure):** The framework employs two complementary information bounds:

     1. **Cohomological Bound:** The **singularity complexity** $S_{\text{coh}}(\mathcal{X}) := \log|\pi_0(\mathcal{X}_{\text{sing}})|$ (counting connected components of the singular locus) is bounded by:
        $$S_{\text{coh}}(\mathcal{X}) \leq C \cdot \chi(\partial\mathcal{X})$$
        where $\chi$ denotes the Euler characteristic. This topological bound constrains how singularities can distribute across the boundary topology.

     2. **Information Bound (Data Processing Inequality):** The mutual information between bulk and observer is bounded by the capacity of the boundary channel: $I(X; Z) \leq I(X; Y)$. This information-theoretic constraint is enforced by Tactic E8 ({prf:ref}`def-e8`) using the Capacity interface $\mathrm{Cap}_H$.

     The cohomological bound detects *topological* obstructions (too many singularity components for the boundary topology), while the metric bound detects *geometric* obstructions (too much information density for the boundary size). Both are necessary: the Sieve uses $\chi$ for categorical exclusion and Area for physical exclusion.
:::

:::{prf:remark} Classical Recovery
:label: rem-classical-recovery

When $\mathcal{E} = \mathbf{Set}$ (the trivial topos), the categorical definition reduces to classical structural flow data: $\mathcal{X}$ becomes a Polish space $X$, the connection $\nabla$ becomes a vector field generating a semiflow, the truncation functors become decidable propositions, and the boundary morphism $\partial_\bullet$ becomes the Sobolev trace operator $u \mapsto u|_{\partial\Omega}$ with flux $\mathcal{J} = \nabla u \cdot \nu$ (normal derivative).
:::

:::{div} feynman-prose
Let me make sure you understand what this "Classical Recovery" remark is really telling you. It says that when you strip away all the fancy categorical language and work in the ordinary world of sets and functions, you get back the classical PDE setup that you know and love.

This is important because it means the categorical framework is not changing the mathematics---it is *organizing* it. The Polish space, the vector field, the boundary trace operator: these are all sitting inside the categorical picture as special cases. We have not thrown out classical analysis; we have given it a home where it can talk to gauge theory and topology.

The payoff comes when you have systems with gauge symmetry or non-trivial topology. Then the categorical structure starts doing real work, keeping track of equivalences that would otherwise slip through the cracks.
:::

(sec-fixed-point-principle)=
## The Fixed-Point Principle

:::{div} feynman-prose
Here is the beautiful thing about all those axioms in the Hypostructure definition. You might think they were chosen arbitrarily, that someone sat down and listed properties that seemed useful. But they were not. They all come from a single requirement: *the system must be self-consistent*.

What do I mean by self-consistent? I mean that if you let the system evolve, it should not contradict itself. The states that persist indefinitely are exactly the fixed points---the places where the evolution brings you right back to where you started.

This is not a metaphysical statement. It is a mathematical fact that follows from dissipation. If energy always decreases (except at equilibria), then the system must settle down. It cannot oscillate forever or drift to infinity with bounded energy. The only thing it can do is find a fixed point.

And here is the key: singularities are places where this self-consistency breaks down. The evolution tries to produce a state that cannot exist. That is why detecting singularities is equivalent to checking the fixed-point structure.
:::

The hypostructure axioms are not independent postulates chosen for technical convenience. They are manifestations of a single organizing principle: **self-consistency under evolution**.

:::{prf:definition} Self-Consistency
:label: def-self-consistency

A trajectory $u: [0, T) \to X$ is **self-consistent** if:
1. **Temporal coherence:** The evolution $F_t: x \mapsto S_t x$ preserves the structural constraints defining $X$.
2. **Asymptotic stability:** Either $T = \infty$, or the trajectory approaches a well-defined limit as $t \nearrow T$.
:::

:::{prf:theorem} [KRNL-Consistency] The Fixed-Point Principle
:label: mt-krnl-consistency

Let $\mathcal{S}$ be a structural flow datum with **strict dissipation** (i.e., $\Phi(S_t x) < \Phi(x)$ unless $x$ is an equilibrium). The following are equivalent:
1. The system $\mathcal{S}$ satisfies the hypostructure axioms on all finite-energy trajectories.
2. Every finite-energy trajectory is asymptotically self-consistent.
3. The only persistent states are fixed points of the evolution operator $F_t = S_t$ satisfying $F_t(x) = x$.

**Extension to Non-Gradient Systems:** For systems with non-strict dissipation (Backend B: Morse-Smale, Backend C: Conley-Morse in {prf:ref}`def-permit-morsedecomp`), statement (3) generalizes to: *persistent states are contained in the maximal invariant set (global attractor) $\mathcal{A}$, which may include periodic orbits or more complex recurrence*. The LaSalle Invariance Principle {cite}`LaSalle76` ensures convergence to this invariant set rather than to fixed points specifically.

**Interpretation:** The equation $F(x) = x$ encapsulates the principle: *structures that persist under their own evolution are precisely those that satisfy the hypostructure axioms*. Singularities represent states where $F(x) \neq x$ in the limit—the evolution attempts to produce a state incompatible with its own definition.

**Literature:** {cite}`Banach22`; {cite}`LaSalle76`; {cite}`Lyapunov92`
:::

:::{prf:proof} Proof Sketch (for strict dissipation)
:label: sketch-mt-krnl-consistency

*Step 1 (1 ⇒ 2).* If $\mathcal{S}$ satisfies the axioms, then by the Dissipation axiom ($D_E$), energy is non-increasing: $\Phi(S_t x) \leq \Phi(x)$. Combined with the Compactness axiom, bounded orbits have convergent subsequences. Any limit point $x^*$ satisfies $S_t x^* = x^*$ by continuity.

*Step 2 (2 ⇒ 3).* Self-consistency means $\lim_{t \to \infty} S_t x = x^*$ exists. Taking $t \to \infty$ in $S_{t+s} x = S_s(S_t x)$ yields $S_s x^* = x^*$ for all $s$.

*Step 3 (3 ⇒ 1).* If non-fixed-point states cannot persist, then any trajectory either: (a) reaches a fixed point (satisfying axioms), or (b) exits in finite time. Case (b) is excluded by the finite-energy hypothesis and the energy-dissipation inequality.
:::

(sec-foundation-theorems)=
### Foundation Theorems

:::{div} feynman-prose
Now we come to the theorems that make the whole machinery work. I want to be clear about what these theorems actually do.

The first one, KRNL-Exclusion, says: "If no bad pattern can map into your system, then your system is regular." This sounds almost tautological until you realize what it enables. It means we can prove regularity by proving the *absence* of something---that no morphism exists from the universal bad pattern to your system.

The second one, KRNL-Trichotomy, classifies every finite-time breakdown into exactly three categories: the energy disperses and everything is fine, the energy concentrates but all constraints are satisfied, or there is a genuine singularity. There is no fourth option. This is the concentration-compactness dichotomy that underlies all modern PDE regularity theory, now stated in categorical language.

The third piece, the Analytic-to-Categorical Bridge, connects the PDE analysis to the category theory. It says that when you do blow-up analysis and extract a limiting profile, that profile corresponds to a morphism in the categorical picture. This is the handshake between the two worlds.

These theorems are not about any specific equation. They are metatheorems---theorems about the framework itself. Once you have them, you can apply the machinery to any system that fits the hypostructure mold.
:::

The following metatheorems establish the logical soundness of the Sieve before examining any specific node. They prove that the framework's categorical approach to singularity resolution is mathematically valid.

::::{prf:theorem} [KRNL-Exclusion] Principle of Structural Exclusion
:label: mt-krnl-exclusion

**Rigor Class:** F (Framework-Original) — see {prf:ref}`def-rigor-classification`

**Sieve Target:** Node 17 (Lock) — proves the Lock mechanism is valid

**Statement:** Let $T$ be a problem type with category of admissible T-hypostructures $\mathbf{Hypo}_T$. Let $\mathbb{H}_{\mathrm{bad}}^{(T)}$ be the universal Rep-breaking pattern. For any concrete object $Z$ with admissible hypostructure $\mathbb{H}(Z)$, if:

$$\mathrm{Hom}_{\mathbf{Hypo}_T}(\mathbb{H}_{\mathrm{bad}}^{(T)}, \mathbb{H}(Z)) = \emptyset$$

then Interface Permit $\mathrm{Rep}_K(T, Z)$ holds, and hence the conjecture for $Z$ holds.

**Hypotheses (N1–N11):**
1. **(N1)** Category $\mathbf{Hypo}_T$ of admissible T-hypostructures satisfying core interface permits $C_\mu$, $D_E$, $\mathrm{SC}_\lambda$, $\mathrm{LS}_\sigma$, $\mathrm{Cap}_H$, $\mathrm{TB}_\pi$, $\mathrm{GC}_\nabla$
2. **(N2)** Hypostructure assignment $Z \mapsto \mathbb{H}(Z)$
3. **(N3)** Conjecture equivalence: $\mathrm{Conj}(T,Z) \Leftrightarrow \text{Interface Permit } \mathrm{Rep}_K(T,Z)$
4. **(N8)** Representational completeness of parametrization $\Theta$
5. **(N9)** Existence of universal Rep-breaking pattern with initiality property (see Initiality Lemma below)
6. **(N10)** Admissibility of $\mathbb{H}(Z)$
7. **(N11)** Obstruction condition: $\mathrm{Hom}(\mathbb{H}_{\mathrm{bad}}^{(T)}, \mathbb{H}(Z)) = \emptyset$

:::{prf:theorem} Categorical Completeness of the Singularity Spectrum
:label: thm-categorical-completeness

**Statement:** For any problem type $T$, the category of singularity patterns admits a universal object $\mathbb{H}_{\mathrm{bad}}^{(T)}$ that is **categorically exhaustive**: every singularity in any $T$-system factors through $\mathbb{H}_{\mathrm{bad}}^{(T)}$.

**Key Mechanism:**
1. **Node 3 (Compactness)** converts analytic blow-up $\to$ categorical germ via concentration-compactness ({prf:ref}`mt-krnl-trichotomy`)
2. **Small Object Argument** proves the germ set $\mathcal{G}_T$ is small (a set, not a proper class)
3. **Cofinality** proves every pattern factors through $\mathcal{G}_T$
4. **Node 17 (Lock)** checks if the universal bad pattern embeds into $\mathbb{H}(Z)$

**Consequence:** The Bad Pattern Library is logically exhaustive—no singularity can "escape" the categorical check. This addresses the "Completeness Gap" critique: the proof that physical singularities map to categorical germs is provided by concentration-compactness (Node 3), while the proof that germs are exhaustive is provided by the Small Object Argument (Initiality Lemma below).

:::

:::{prf:proof}
:label: proof-thm-categorical-completeness

See the Initiality Lemma (N9) and Cofinality argument below.
:::

:::{prf:proof} Initiality Lemma (Proof of N9)
:label: proof-initiality-lemma

The universal Rep-breaking pattern $\mathbb{H}_{\mathrm{bad}}^{(T)}$ exists and is initial in the category of singularity patterns.

*Germ Set Construction:* Define the **set of singularity germs** $\mathcal{G}_T$ as the set of isomorphism classes $[P, \pi]$ where:
- $P$ is a local singularity profile satisfying subcriticality: $\dim_H(P) \leq d - 2s_c$ for critical exponent $s_c$
- $\pi: P \to \mathbb{R}^n$ is a blow-up parametrization with $\|\pi\|_{\dot{H}^{s_c}} \leq \Lambda_T$ (energy bound)
- Two pairs $(P, \pi) \sim (P', \pi')$ if they are equivalent under local diffeomorphism respecting the blow-up structure
- **Functor** $\mathcal{D}([P, \pi]) := \mathbb{H}_{[P,\pi]}$: the minimal hypostructure containing the germ

*Smallness via Cardinality Boundedness Argument:* The set $\mathcal{G}_T$ is **small** (a set, not a proper class) by the following argument:

1. **Compactness:** The energy bound $\|\pi\|_{\dot{H}^{s_c}} \leq \Lambda_T$ implies precompactness in the weak topology by Banach-Alaoglu
2. **Finite-dimensional moduli:** Quotienting by the symmetry group $G$ (translations, rotations, scaling), the space of germs has finite-dimensional moduli $\dim(\mathcal{G}_T/G) < \infty$
3. **Countable representative system:** By separability of the ambient Sobolev space, there exists a countable dense subset $\mathcal{G}_T^0 \subset \mathcal{G}_T$. Every germ in $\mathcal{G}_T$ is equivalent (under $G$-action and weak limits) to a representative in $\mathcal{G}_T^0$.

**Terminological note:** This is NOT Quillen's Small Object Argument ({cite}`Quillen67` §II.3), which concerns generating cofibrations in model categories. Rather, we use finite-dimensionality + energy bounds to establish cardinality bounds directly.

Define the **small index category** $\mathbf{I}_{\text{small}}$:
- Objects: Elements of $\mathcal{G}_T$
- Morphisms: Profile embeddings respecting blow-up structure

*Existence of Colimit:* The category $\mathbf{Hypo}_T$ is locally presentable ({cite}`Lurie09` §5.5). Since $\mathbf{I}_{\text{small}}$ is a **small category**, the colimit exists by standard results. Define:
$$\mathbb{H}_{\mathrm{bad}}^{(T)} := \mathrm{colim}_{\mathbf{I}_{\text{small}}} \mathcal{D}$$

*Cofinality:* For any singularity pattern $\mathbb{H}_P$ with $(P, \pi)$ in the "large" category of all patterns, there exists a representative germ $[P', \pi'] \in \mathcal{G}_T$ such that $\mathbb{H}_P \to \mathbb{H}_{[P',\pi']}$ factors through the germ. Thus $\mathbf{I}_{\text{small}}$ is **cofinal** in the hypothetical large category, and:
$$\mathrm{colim}_{\mathbf{I}_{\text{small}}} \mathcal{D} \cong \mathrm{colim}_{\mathbf{I}} \mathcal{D}$$
by cofinality ({cite}`MacLane71` §IX.3).

*Initiality Verification:* By the universal property of colimits, for any germ $[P, \pi] \in \mathcal{G}_T$:
$$\exists! \; \iota_{[P,\pi]}: \mathbb{H}_{[P,\pi]} \to \mathbb{H}_{\mathrm{bad}}^{(T)}$$
(the coprojection). Conversely, for any $\mathbb{H} \in \mathbf{Hypo}_T$ receiving all germ patterns:
$$(\forall [P,\pi] \in \mathcal{G}_T.\, \mathbb{H}_{[P,\pi]} \to \mathbb{H}) \Rightarrow (\mathbb{H}_{\mathrm{bad}}^{(T)} \to \mathbb{H})$$

*Explicit Construction by Type:*
- **Algebraic ($T_{\mathrm{alg}}$):** $\mathbb{H}_{\mathrm{bad}}$ is the universal Hodge structure failing the Hodge conjecture: the direct sum of all non-algebraic $(p,p)$-classes. The germ set $\mathcal{G}_{T_{\mathrm{alg}}}$ consists of minimal non-algebraic $(p,p)$-classes up to Hodge isomorphism. Initiality: any non-algebraic cycle factors through this universal failure.
- **Parabolic ($T_{\mathrm{para}}$):** $\mathbb{H}_{\mathrm{bad}}$ is the Type I blow-up profile with minimal energy. The germ set $\mathcal{G}_{T_{\mathrm{para}}}$ consists of blow-up profiles below energy threshold $\Lambda_T$, modulo scaling and translation. By {cite}`MerleZaag98`, this set is finite-dimensional. Initiality: concentration-compactness forces convergence to profiles in $\mathcal{G}_{T_{\mathrm{para}}}$.
- **Quantum ($T_{\mathrm{quant}}$):** $\mathbb{H}_{\mathrm{bad}}$ is the zero-mass instanton (or 't Hooft operator). The germ set $\mathcal{G}_{T_{\mathrm{quant}}}$ consists of instantons with action $\leq \Lambda_T$, modulo gauge equivalence. By {cite}`Uhlenbeck82`, this moduli space is finite-dimensional. Initiality: bubbling produces instantons at concentration points.

*Certificate:* The initiality proof produces $K_{\mathrm{init}}^+ := (\mathbb{H}_{\mathrm{bad}}^{(T)}, \mathcal{G}_T, \mathbf{I}_{\text{small}}, \mathrm{colim}, \{\iota_{[P,\pi]}\}_{[P,\pi] \in \mathcal{G}_T})$

**Conclusion:** Global regularity via categorical obstruction: singularities cannot embed into admissible structures.
:::

:::{prf:proof} Proof of KRNL-Exclusion (Categorical Proof Template)
:label: proof-mt-krnl-exclusion

*Step 1 (Ambient Setup).* Let $\mathcal{E}$ be the cohesive $(\infty,1)$-topos containing $\mathbf{Hypo}_T$ as a full subcategory. By {cite}`Lurie09` §6.1, $\mathcal{E}$ admits an **internal logic** given by its subobject classifier $\Omega$. Propositions in $\mathcal{E}$ correspond to morphisms $p: 1 \to \Omega$ where $1$ is the terminal object.

*Step 2 (Construction: Singularity Sheaf).* Define the **Singularity Sheaf** $\mathcal{S}_{\mathrm{bad}}: \mathbf{Hypo}_T^{\mathrm{op}} \to \mathbf{Set}$ by:
$$\mathcal{S}_{\mathrm{bad}}(\mathbb{H}) := \mathrm{Hom}_{\mathbf{Hypo}_T}(\mathbb{H}_{\mathrm{bad}}^{(T)}, \mathbb{H})$$
This is a presheaf assigning to each hypostructure its set of "singular embeddings."

*Step 3 (Internal Logic Translation).* In the internal logic of $\mathcal{E}$, the statement "$\mathbb{H}$ has no singularities" translates to:
$$\llbracket \mathcal{S}_{\mathrm{bad}}(\mathbb{H}) = \emptyset \rrbracket = \top$$
where $\llbracket - \rrbracket$ denotes the truth value in the Heyting algebra $\Omega(\mathcal{E})$. The internal negation is:
$$\neg\exists \phi.\, \phi: \mathbb{H}_{\mathrm{bad}} \to \mathbb{H}$$

*Step 4 (Well-definedness via Yoneda).* The Singularity Sheaf is representable by $\mathbb{H}_{\mathrm{bad}}^{(T)}$ via Yoneda:
$$\mathcal{S}_{\mathrm{bad}} \cong y(\mathbb{H}_{\mathrm{bad}}^{(T)}) = \mathrm{Hom}(\mathbb{H}_{\mathrm{bad}}^{(T)}, -)$$
The initiality property (N9) ensures $\mathbb{H}_{\mathrm{bad}}^{(T)}$ is the **colimit** of all singularity patterns, making it the universal testing object.

*Step 5 (Universal Property Verification).* For any $\mathbb{H}(Z) \in \mathbf{Hypo}_T$:
- If $\mathcal{S}_{\mathrm{bad}}(\mathbb{H}(Z)) \neq \emptyset$: there exists $\phi: \mathbb{H}_{\mathrm{bad}} \to \mathbb{H}(Z)$, witnessing a singularity in $Z$
- If $\mathcal{S}_{\mathrm{bad}}(\mathbb{H}(Z)) = \emptyset$: no morphism exists, so by the **internal logic of $\mathcal{E}$**, the proposition "$Z$ is singular" is **internally false**

*Step 6 (Contrapositive in Internal Logic).* The logical structure is:
$$(\exists \phi.\, \phi: \mathbb{H}_{\mathrm{bad}} \to \mathbb{H}(Z)) \Leftrightarrow \neg\mathrm{Rep}_K(T,Z)$$
Taking the contrapositive in the Heyting algebra:
$$\neg(\exists \phi.\, \phi: \mathbb{H}_{\mathrm{bad}} \to \mathbb{H}(Z)) \Rightarrow \mathrm{Rep}_K(T,Z)$$
The empty Hom-set (N11) verifies the antecedent, yielding the consequent.

*Step 7 (Certificate Production).* The proof is constructive in the sense that:
- The certificate $K_{\text{Lock}}^{\mathrm{blk}}$ witnesses $\mathrm{Hom} = \emptyset$
- The verification is decidable: enumerate tactics E1–E12 and confirm each produces an obstruction
- The payload contains the explicit obstruction witnesses from each tactic

**Certificate Produced:** $K_{\text{Lock}}^{\mathrm{blk}}$ with payload $(\mathrm{Hom} = \emptyset, Z, T, \text{obstruction witnesses})$

**Literature:** {cite}`Grothendieck67` SGA 1 Exposé V (representability); {cite}`MacLane71` §III.3 (limits and colimits); {cite}`Lurie09` §5.5–6.1 (presentable $\infty$-categories, internal logic); {cite}`Johnstone02` (Sketches of an Elephant, topos internal logic)
:::
::::

:::{div} feynman-prose
Now let me tell you about one of the most powerful ideas in PDE analysis: concentration-compactness. The question is simple. You have a sequence of solutions with bounded energy. What can happen as you take a limit?

There are really only two possibilities. Either the energy spreads out and disperses to infinity, becoming negligible everywhere---like smoke diffusing in a room---or it concentrates at one or more points. The first case is good: dispersed energy cannot cause blow-up. The second case is where the action is.

When energy concentrates, you can zoom in on the concentration points and extract a "profile"---a limiting object that captures the essential structure of the concentration. This is the profile decomposition technique. It says that any bounded sequence can be written as a finite number of these profiles plus a remainder that disperses.

The trichotomy theorem puts this into the categorical framework. Every breakdown is either dispersion (good), concentration that satisfies all constraints (good), or concentration that violates a constraint (bad---genuine singularity). The beautiful thing is that "violating a constraint" translates directly to "admitting a morphism from the bad pattern." This is how PDE analysis plugs into the categorical machinery.
:::

:::{prf:theorem} [KRNL-Trichotomy] Structural Resolution
:label: mt-krnl-trichotomy

**Sieve Target:** Node 3 (CompactCheck) --- justifies the Concentration/Dispersion dichotomy

**Statement:** Let $\mathcal{S}$ be a structural flow datum satisfying minimal regularity (Reg) and dissipation ($D_E$) interface permits. Every trajectory $u(t) = S_t x$ with finite breakdown time $T_*(x) < \infty$ classifies into exactly one of three outcomes:

| **Outcome** | **Modes** | **Mechanism** |
|-------------|-----------|---------------|
| Global Existence | Mode D.D | Energy disperses, no concentration, solution scatters |
| Global Regularity | Modes S.E, C.D, T.E, S.D | Concentration but all permits satisfied, preventing singularity |
| Genuine Singularity | Mode C.E | Energy escapes or structured blow-up with permits violated |

**Hypotheses:**
1. **(Reg)** Minimal regularity: semiflow $S_t$ well-defined on $X$
2. **(D)** Dissipation: energy-dissipation inequality holds
3. **(C)** Compactness: bounded energy implies profile convergence modulo $G$

**Certificate Produced:** Trichotomy classification $\{K_{\text{D.D}}, K_{\text{Reg}}, K_{\text{C.E}}\}$

**Literature:** {cite}`Lions84` Lemma I.1 (concentration-compactness); {cite}`BahouriGerard99` Theorem 1 (profile decomposition); {cite}`KenigMerle06` Theorem 1.1 (rigidity); {cite}`Struwe90` §3 (singularity analysis)
:::

:::{prf:proof} Proof Sketch
:label: sketch-mt-krnl-trichotomy

*Step 1 (Energy Dichotomy).* By (D), $\Phi(u(t))$ is non-increasing. Either $\Phi(u(t)) \to 0$ (dispersion, Mode D.D), or $\Phi(u(t)) \to \Phi_* > 0$ (concentration). This is the **concentration-compactness dichotomy** of {cite}`Lions84` Lemma I.1: for bounded sequences in Sobolev spaces, either mass disperses to infinity or concentrates at finitely many points.

*Step 2 (Profile Extraction).* In the concentration case, by (C), there exists a sequence $t_n \to T_*$ and symmetry elements $g_n \in G$ such that $g_n \cdot u(t_n) \to v^*$ (profile). This is the **profile decomposition** of {cite}`BahouriGerard99` Theorem 1: any bounded sequence in $\dot{H}^{s_c}$ admits decomposition into orthogonal profiles with asymptotically vanishing remainder.

*Step 3 (Profile Classification).* The limiting profile $v^*$ either: (a) satisfies all interface permits → Global Regularity via {cite}`KenigMerle06` Theorem 1.1 (rigidity implies scattering or soliton), or (b) violates at least one permit → Genuine Singularity (Mode C.E) with {cite}`Struwe90` providing the singularity structure.
:::

:::{prf:lemma} Analytic-to-Categorical Bridge
:label: lem-bridge

**Statement:** Every **profile-extractable** analytic blow-up in a $T$-system induces a morphism from a singularity germ to the system's hypostructure.

**Hypothesis (Profile Extractability):** The blow-up must satisfy the concentration-compactness criterion of {prf:ref}`mt-krnl-trichotomy`, yielding a finite-energy limiting profile. This excludes wild oscillations, turbulent cascades, and non-compact symmetry orbits.

**Mechanism:**
1. **Analytic Input:** Trajectory $u(t)$ with breakdown time $T_* < \infty$
2. **Profile Extraction:** By {prf:ref}`mt-krnl-trichotomy`, concentration-compactness yields profile $v^*$ with finite energy $\|v^*\|_{\dot{H}^{s_c}} \leq \Lambda_T$
3. **Germ Construction:** Profile $v^*$ determines germ $[P, \pi] \in \mathcal{G}_T$ via the blow-up parametrization (scaling, centering, symmetry quotient)
4. **Morphism Induction:** The singularity locus inclusion $\iota: \Sigma \hookrightarrow Z$ induces $\phi: \mathbb{H}_{[P,\pi]} \to \mathbb{H}(Z)$ by functoriality of the hypostructure assignment

**This lemma is the "handshake" between PDE analysis (Steps 1-2) and category theory (Steps 3-4).** Node 3 ({prf:ref}`def-node-compact`) performs Step 2; the Initiality Lemma in {prf:ref}`mt-krnl-exclusion` handles Steps 3-4.

**Remark (Non-Extractable Blow-ups):** Blow-ups failing the Profile Extractability hypothesis do not satisfy the Bridge conditions. Such cases—including wild oscillations with unbounded variation, turbulent cascades lacking definable structure, and profiles with non-compact automorphism groups—route to the **Horizon mechanism** ({prf:ref}`mt-resolve-admissibility`, Case 3) with certificate $K_{\text{inadm}}$ indicating "profile unclassifiable." The Lock (Node 17) remains sound because:
1. Non-extractable blow-ups are explicitly excluded from this Bridge lemma
2. Such blow-ups trigger the Horizon exit, halting the Sieve with an honest "unclassifiable" verdict
3. No false positive (claiming regularity for a genuine singularity) can occur

**Consequence:** The Sieve's categorical check (Node 17) is connected to physical reality by this bridge: every genuine *classifiable* singularity produces a morphism, and the Lock's Hom-emptiness test detects all such morphisms. Unclassifiable singularities are handled separately by the Horizon mechanism, ensuring the framework's soundness without claiming universal profile extractability.
:::

:::{prf:theorem} [KRNL-Equivariance] Equivariance Principle
:label: mt-krnl-equivariance

**Sieve Target:** Meta-Learning — guarantees learned parameters preserve symmetry group $G$

**Statement:** Let $G$ be a compact Lie group acting on the system distribution $\mathcal{S}$ and parameter space $\Theta$. Under compatibility assumptions:

1. **(Group-Covariant Distribution)** $S \sim \mathcal{S} \Rightarrow g \cdot S \sim \mathcal{S}$ for all $g \in G$
2. **(Equivariant Parametrization)** $g \cdot \mathcal{H}_\Theta(S) \simeq \mathcal{H}_{g \cdot \Theta}(g \cdot S)$
3. **(Defect-Level Equivariance)** $K_{A,g \cdot S}^{(g \cdot \Theta)}(g \cdot u) = K_{A,S}^{(\Theta)}(u)$

Then:
- Every risk minimizer $\widehat{\Theta}$ lies in the $G$-orbit: $\widehat{\Theta} \in G \cdot \Theta^*$
- Gradient flow preserves equivariance: if $\Theta_0$ is $G$-equivariant, so is $\Theta_t$
- Learned hypostructures inherit all symmetries of the system distribution

**Certificate Produced:** $K_{\text{SV08}}^+$ (Symmetry Preservation)

**Literature:** {cite}`Noether18`; {cite}`CohenWelling16`; {cite}`Kondor18`; {cite}`Weyl46`
:::

:::{prf:proof} Proof Sketch
:label: sketch-mt-krnl-equivariance

*Step 1 (Risk Invariance).* By hypothesis (1), if $S \sim \mathcal{S}$ then $g \cdot S \sim \mathcal{S}$. The risk functional $R(\Theta) = \mathbb{E}_{S \sim \mathcal{S}}[\mathcal{L}(\Theta, S)]$ satisfies $R(g \cdot \Theta) = R(\Theta)$ by change of variables.

*Step 2 (Gradient Equivariance).* Since $R$ is $G$-invariant, $\nabla R$ is $G$-equivariant: $\nabla R(g \cdot \Theta) = g \cdot \nabla R(\Theta)$. The gradient flow $\dot{\Theta} = -\nabla R(\Theta)$ preserves $G$-orbits.

*Step 3 (Defect Transfer).* By hypothesis (3), defects transform covariantly: $K_{A,g \cdot S}^{(g \cdot \Theta)}(g \cdot u) = K_{A,S}^{(\Theta)}(u)$. Symmetries of the input distribution transfer to learned parameters and their certificates.
:::

:::{prf:theorem} Halting/AIT Sieve Thermodynamics (Phase Transition Witness)
:label: thm-halting-ait-sieve-thermo

In the algorithmic-thermodynamic translation, let $\mathcal{K} = \{e : \varphi_e(e)\downarrow\}$ be the halting set and let Kolmogorov complexity ({prf:ref}`def-kolmogorov-complexity`) act as energy. Then there is a sharp boundary between:

- **Crystal Phase (Decidable):** Families with $K(L \cap [0,n]) = O(\log n)$, where the Sieve certifies **REGULAR** via Axiom R
- **Liquid Phase (C.E.):** Families with $K(L \cap [0,n]) = O(\log n)$ but Axiom R fails (e.g., Halting Set) → **HORIZON**
- **Gas Phase (Random):** Families with $K(L \cap [0,n]) \geq n - O(1)$, where the Sieve routes to **HORIZON**

This theorem formalizes the phase transition detected by {prf:ref}`mt-krnl-horizon-limit`.
:::

:::{prf:proof}
:label: proof-thm-halting-ait-sieve-thermo

We establish the sharp phase boundary in four steps.

**Step 1 (Crystal Regime).** Let $L \subseteq \mathbb{N}$ be decidable. Then there exists a Turing machine $M$ with finite description computing $\chi_L$. For the initial segment $L_n := L \cap [0,n]$:
$$K(L_n) \leq |M| + O(\log n) = O(\log n)$$
The $O(\log n)$ term encodes $n$. Since $L$ is decidable, Axiom R holds (the decider serves as recovery operator). Sieve verdict: **REGULAR** with $K_{\text{Crystal}}^+$.

**Step 2 (Gas Regime).** Let $L \subseteq \mathbb{N}$ be Martin-Löf random. By the Levin-Schnorr Theorem {cite}`Levin73b,Schnorr73`:
$$K(L_n) \geq n - O(1)$$
No computable predictor can anticipate the membership of $L$. Axiom R fails absolutely—no recovery operator exists. Sieve verdict: **HORIZON** with $K_{\text{Gas}}^{\text{blk}}$.

**Step 3 (Phase Transition).** The Halting Set $\mathcal{K}$ exhibits **liquid** behavior:
- *Description complexity:* $K(\mathcal{K}_n) = O(\log n)$ since $\mathcal{K}$ is c.e. (the enumeration program has finite length).
- *Axiom R failure:* Despite low complexity, $\mathcal{K}$ is undecidable—no total computable recovery operator exists (Turing 1936 {cite}`Turing36`).
- *Certificate:* $K_{\text{Liquid}}^{\text{blk}} = (\text{c.e. index}, \text{diagonal construction})$

This is the critical phase transition: low complexity does not imply decidability when Axiom R fails.

**Step 4 (Sharp Boundary).** The boundary is sharp in the following sense: for any $\epsilon > 0$, there exist sets $L^+, L^-$ with:
$$K(L^+_n) = (1-\epsilon)n, \quad K(L^-_n) = O(\log n)$$
such that $L^+$ is undecidable (gas) and $L^-$ is decidable (crystal). The Halting Set $\mathcal{K}$ lies exactly at the boundary, demonstrating that the phase transition occurs at the computability threshold, not the complexity threshold.

**Thermodynamic Interpretation:** Under the correspondence of {prf:ref}`thm-sieve-thermo-correspondence`, this is a first-order phase transition in the decidability order parameter $\rho_R$ (Axiom R satisfaction).
:::

:::{div} feynman-prose
Now I want to be completely honest with you about what this framework can and cannot do. This is important, because there is a tendency in theoretical work to oversell. We are not going to do that.

The Horizon Limit theorem says: there are problems the Sieve cannot solve, and it knows it cannot solve them. When you hit such a problem, you get a HORIZON verdict---not "REGULAR" or "SINGULAR" but "I cannot tell." This is honesty, not failure.

Think about why this must be true. The Sieve is a finite computational procedure. It has finite memory and runs for finite time. But some problems require infinite information to describe---these are the incompressible or algorithmically random problems. You cannot fit an infinite description into a finite memory. It is not a limitation of our particular approach; it is mathematics.

The Halting Problem sits at a fascinating boundary. It has low descriptive complexity (you can write down the definition easily), but it is undecidable because the *decision procedure* would require infinite resources. The Sieve recognizes this: low complexity but Axiom R fails. That is the "liquid phase"---between the crystalline order of decidable problems and the gaseous chaos of random ones.

What matters is that the Sieve never lies. It never says "REGULAR" for something that could be singular. When it cannot tell, it says so. That is the honest epistemics this framework is built on.
:::

:::{prf:theorem} [KRNL-HorizonLimit] The Horizon Limit (Framework Boundaries)
:label: mt-krnl-horizon-limit

**Rigor Class:** F (Framework-Original) --- Explicit statement of framework limitations

**Sieve Target:** Honest Epistemics — establishes what the Sieve **cannot** do

**Statement:** For any computational problem $\mathcal{I}$ whose **Kolmogorov complexity** exceeds the Sieve's finite memory buffer $M_{\text{sieve}}$, the verdict is provably **HORIZON**. The Sieve does not solve undecidable problems; it classifies them as "thermodynamically irreducible."

**Formal Statement**:
Let $\mathcal{S}$ be the Structural Sieve with finite memory $M_{\text{sieve}} \in \mathbb{N}$ (in bits). For any problem $\mathcal{I} \subseteq \mathbb{N}$:

$$K(\mathcal{I}) > M_{\text{sieve}} \Rightarrow \text{Verdict}(\mathcal{S}, \mathcal{I}) = \texttt{HORIZON}$$

where $K(\mathcal{I}) := \min\{|p| : U(p) = \text{characteristic function of } \mathcal{I}\}$ is the Kolmogorov complexity ({cite}`LiVitanyi19`).

**What the Sieve CAN do**:
- ✓ Classify decidable problems as **REGULAR** (Axiom R holds)
- ✓ Detect phase transitions between decidable/undecidable via AIT ({prf:ref}`thm-halting-ait-sieve-thermo`)
- ✓ Provide certificates for Axiom R failure (diagonal construction, Rice's Theorem)
- ✓ Classify profile-extractable blow-ups via concentration-compactness

**What the Sieve CANNOT do (Horizon Limits)**:
- ✗ Solve the Halting Problem (Turing 1936 {cite}`Turing36`)
- ✗ Store/classify problems with $K(\mathcal{I}) > M_{\text{sieve}}$ (information-theoretic impossibility)
- ✗ Provide infinite computational resources (finite memory/time constraint)
- ✗ Classify non-extractable blow-ups (wild oscillations, turbulent cascades) → routes to **HORIZON** with $K_{\text{inadm}}$

**Honest Verdict Protocol**:
When $K(\mathcal{I}) > M_{\text{sieve}}$, emit:
$$K_{\text{Horizon}}^{\text{blk}} = (\text{"Thermodynamically irreducible"}, K(\mathcal{I}) > M_{\text{sieve}}, \text{Axiom R failure proof})$$

**Connection to Algorithmic Information Theory**:
The Horizon Limit instantiates the Sieve-Thermodynamic Correspondence ({prf:ref}`thm-sieve-thermo-correspondence`):

| AIT Component | Formal Reference | Sieve Role |
|---------------|------------------|------------|
| Energy $E(x) = K(x)$ | {prf:ref}`def-kolmogorov-complexity` | Memory bound |
| Partition Function $Z = \Omega$ | {prf:ref}`def-chaitin-omega` | Normalization |
| Computational Depth $d_s(x)$ | {prf:ref}`def-computational-depth` | Thermodynamic cost |
| Phase Classification | {prf:ref}`def-algorithmic-phases` | Verdict assignment |

For the Halting Set $\mathcal{K} = \{e : \varphi_e(e)\downarrow\}$:
- **Crystal:** Decidable $L$ with $K(L_n) = O(\log n)$ → **REGULAR**
- **Liquid:** C.e. sets like $\mathcal{K}$ with $K(\mathcal{K}_n) = O(\log n)$ but Axiom R fails → **HORIZON**
- **Gas:** Random $L$ with $K(L_n) \geq n - O(1)$ → **HORIZON**

See {prf:ref}`thm-halting-ait-sieve-thermo` for the phase transition witness theorem.

**Certificate Produced:** $K_{\text{Horizon}}^{\text{blk}}$ with payload documenting complexity bound

**Literature:** {cite}`Turing36` (Halting Problem); {cite}`Chaitin75` (Algorithmic Randomness); {cite}`LiVitanyi19` (Kolmogorov Complexity); {cite}`Zurek89` (Thermodynamic Cost of Computation)
:::

:::{prf:proof} Proof
:label: proof-mt-krnl-horizon-limit

**Step 1 (Information-Theoretic Lower Bound)**:
To decide membership in $\mathcal{I}$, the sieve must store a representation of $\mathcal{I}$ requiring at least $K(\mathcal{I})$ bits (by definition of Kolmogorov complexity). Any shorter representation would contradict the minimality of $K(\mathcal{I})$.

**Step 2 (Memory Constraint)**:
If $K(\mathcal{I}) > M_{\text{sieve}}$, no representation of $\mathcal{I}$ fits in the sieve's finite memory buffer.

**Step 3 (Horizon Verdict)**:
Unable to store $\mathcal{I}$, the sieve **cannot** execute the decision procedure. By the Honest Epistemics Protocol, it outputs **HORIZON** with certificate:
$$K_{\text{Horizon}}^{\text{blk}} = (\text{Complexity } K(\mathcal{I}) = [value] > M_{\text{sieve}}, \text{proof of memory insufficiency})$$

**Step 4 (No False Negatives)**:
The HORIZON verdict does **not** claim the problem is unsolvable in principle—only that it exceeds the sieve's finite capacity. This maintains soundness: the sieve never claims regularity for a problem it cannot classify.

**Corollary (Halting Problem)**:
For the halting set $K$, individual programs $e$ with $K(e) > M_{\text{sieve}}$ receive **HORIZON** verdict. The set $K$ itself has $K(K \cap [0,n]) = O(\log n)$ (c.e.), but Axiom R fails by diagonal argument → **HORIZON** via Axiom R obstruction, not memory limit.

$\square$
:::

:::{prf:remark} Framework Boundaries and Honest Epistemics
:label: rem-honest-epistemics

This theorem makes the framework's limitations **explicit and mathematically rigorous**:

1. **Computational Realism**: The Sieve has finite memory $M_{\text{sieve}}$ and finite time, instantiating the resource constraints of any computable procedure.

2. **No Oracle Claims**: The Sieve does not claim to solve undecidable problems. The HORIZON verdict is honest acknowledgment: "This exceeds my computable resources."

3. **AIT Interpretation**: Problems with $K(\mathcal{I}) > M_{\text{sieve}}$ are **algorithmically irreducible** relative to the Sieve's capacity. By the Invariance Theorem ({prf:ref}`def-kolmogorov-complexity`), this characterization is machine-independent up to $O(1)$ constant.

**Phase Classification** ({prf:ref}`def-algorithmic-phases`):

| Input Type | Complexity | Axiom R | Sieve Verdict |
|------------|------------|---------|---------------|
| Crystal (Decidable) | $K = O(\log n)$ | Holds | REGULAR |
| Liquid (C.E.) | $K = O(\log n)$, R fails | Fails | HORIZON |
| Gas (Random) | $K \geq n - O(1)$ | Fails | HORIZON |

The Sieve's HORIZON verdict encompasses both **information-theoretic limits** (Gas phase: incompressible problems) and **logical limits** (Liquid phase: c.e. problems where Axiom R fails despite low complexity). This classification is grounded in rigorous AIT, not physical analogy.
:::
