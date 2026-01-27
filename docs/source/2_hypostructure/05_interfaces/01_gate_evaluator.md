---
title: "Universal Gate Evaluator Interface"
---

(sec-gate-evaluator)=
# Universal Gate Evaluator Interface

(sec-gate-evaluator-interface)=
## The Universal Gate Evaluator Interface

:::{div} feynman-prose

Now here is where we get to the heart of the matter. We want to build a machine that can look at any dynamical system - a fluid, a neural network, a proof search algorithm, whatever - and decide whether it is going to behave nicely or blow up in our faces.

The trouble is, these systems look completely different on the surface. A fluid has velocities and pressures, a graph has nodes and edges, a type theory has terms and judgments. How can we possibly have one framework that handles all of them?

The answer is to ask: what do we actually need to check? Forget the specifics. Every system we care about has some notion of "state" (where it is now), some notion of "cost" or "energy" (how far from equilibrium it is), and some notion of "evolution" (how it changes over time). If we can measure these things, we can check whether the system is behaving.

So what follows is a kind of universal checklist - a sequence of gates that any system must pass through. Each gate asks a specific yes-or-no question: Is the energy bounded? Do discrete events accumulate? Is the system converging somewhere sensible? And so on.

The magic is that once you translate your particular system into this common language, the same verification machinery applies to all of them. That is what we mean by "universal."

:::

A Hypostructure $\mathbb{H}$ is an object in a cohesive $(\infty, 1)$-topos $\mathcal{E}$ equipped with a **Gate Evaluator Interface**. This interface maps abstract structural data to decidable types (Propositions) that the Sieve machine can verify.

To support everything from **Navier-Stokes** to **Graph Theory** to **Homotopy Type Theory**, we define the interfaces using the language of **Higher Topos Theory** (specifically, internal logic of an $(\infty,1)$-topos). This allows "Space" to be a manifold, a graph, or a type; "Energy" to be a functional, a complexity measure, or a truth value.

---

### Ambient Structure

:::{div} feynman-prose

Before we can ask questions about a system, we need to say what kind of mathematical universe it lives in. This is like asking "what is the stage on which our drama plays out?"

The answer is a topos - specifically, a cohesive $(\infty,1)$-topos. Now that sounds terrifying, but the idea is simple: a topos is a mathematical universe where you can do logic. It has objects (the things), morphisms (the relationships), and a notion of truth. The "cohesive" part means it knows about topology - it can distinguish between "connected" and "disconnected," between "nearby" and "far away."

Why do we need such generality? Because a fluid lives in the world of smooth manifolds, a graph lives in the world of discrete sets, and a type lives in the world of homotopy types. By working in a topos, we get a single language that speaks to all of them.

:::


:::{prf:definition} Ambient Topos (Formal)
:label: def-ambient-topos-formal

An **Ambient Topos** for hypostructure analysis is a cohesive $(\infty,1)$-topos $\mathcal{E}$ equipped with:
1. A terminal object $1 \in \mathcal{E}$
2. Shape/flat/sharp modalities $(\int, \flat, \sharp)$ satisfying cohesion axioms
3. An internal type-theoretic logic with judgments $\Gamma \vdash t : A$
4. A subobject classifier $\Omega$ (truth values)
:::

**Notation:**
- $\mathcal{X}$: The object in $\mathcal{E}$ representing the system state (Space/Type)
- $\mathcal{H}$: The height object representing values/costs
- $\Omega$: The subobject classifier (truth values)
- $\vdash$: The judgment symbol ("System implements...")

:::{prf:definition} Height Object
:label: def-height-object

A **Height Object** $\mathcal{H}$ in $\mathcal{E}$ is an object equipped with:
1. A partial order $\leq: \mathcal{H} \times \mathcal{H} \to \Omega$
2. A bottom element $0: 1 \to \mathcal{H}$
3. An addition operation $+: \mathcal{H} \times \mathcal{H} \to \mathcal{H}$ (for accumulation)

| Domain | Height Object $\mathcal{H}$ | Interpretation |
|--------|---------------------------|----------------|
| PDEs | $\mathbb{R}_{\geq 0}$ (Dedekind reals) | Energy |
| Graphs | $\mathbb{N}$ | Discrete count |
| HoTT | $\text{Level}$ | Universe level |
| Tropical | $\mathbb{T}_\infty = ([0,\infty], \min, +)$ | Min-plus algebra |
:::

:::{div} feynman-prose

Now here is the key abstraction. An "interface" is just a contract between the system and the verifier. It says: "If you can tell me these things about your system (the required structure), then I can compute a definite answer to this question (the evaluator), and I will give you a proof of my answer (the certificate)."

Think of it like a type signature in programming. The interface does not care how you implement the internals - it just needs you to expose certain data in a certain form. Then it can do its job.

The three possible answers are crucial:
- **YES** means "I checked, and the property holds. Here is the proof."
- **NO** with a witness means "I found a counterexample. Here is the offending object."
- **INC** (inconclusive) means "I ran out of time or my method does not apply. I am not saying the property fails - I just cannot decide."

This three-valued logic is honest about the limits of computation. We do not pretend to solve undecidable problems. We just route around them.

:::

:::{prf:definition} Interface Permit
:label: def-interface-permit

An **Interface Permit** $I$ is a tuple $(\mathcal{D}, \mathcal{P}, \mathcal{K})$ consisting of:
1. **Required Structure** $\mathcal{D}$: Objects and morphisms in $\mathcal{E}$ the system must define.
2. **Evaluator** $\mathcal{P}$: A procedure $\mathcal{P}: \mathcal{D} \to \{\texttt{YES}, \texttt{NO}, \texttt{INC}\} \times \mathcal{K}$:
   - **YES:** Predicate holds with constructive witness ($K^+$)
   - **NO:** Predicate refuted with counterexample ($K^{\mathrm{wit}}$)
   - **INC:** Evaluation exceeds computational bounds or method insufficient ($K^{\mathrm{inc}}$)—not a semantic refutation, but a routing signal for fallback pathways

   The outcome is deterministic given the computation budget: INC indicates resource exhaustion, not non-determinism.
3. **Certificate Type** $\mathcal{K}$: The witness structure produced by the evaluation, always a sum type $K^+ \sqcup K^{\mathrm{wit}} \sqcup K^{\mathrm{inc}}$.

A system **implements** Interface $I$ if it provides interpretations for all objects in $\mathcal{D}$ and a computable evaluator for $\mathcal{P}$.

**Evaluation Model (Free-Evaluator Semantics):**
Interfaces may be evaluated in any order permitted by their structural dependencies. The diagram represents a *conventional evaluation flow*, but any interface evaluator $\mathcal{P}_i$ may be called at any time if its required structure ($\mathcal{D}_i$) is available. Certificate accumulation is monotonic but not strictly sequential.

This enables:
- Early evaluation of downstream gates when prerequisites are met
- Parallel evaluation of independent interfaces
- Caching and reuse of certificates across Sieve traversals
:::

---

### $\mathcal{H}_0$ (Substrate Interface)
*The Substrate Definition.*

:::{div} feynman-prose

This is Gate Zero - the most basic question we can ask: does the system even exist in a well-defined sense? Can you tell me what states it can be in, and how it evolves from one state to another?

You might think this is trivial, but it is not. Some PDEs do not have solutions for all initial data. Some algorithms get stuck in infinite loops. Some type-theoretic constructions are not well-founded. This gate catches those problems at the door.

The key insight is the "refinement filter" - a way of taking limits. This matters because many interesting systems are defined as limits of approximations (think of the Navier-Stokes equations as limits of regularized equations). The substrate interface says: "You must tell me how limits work in your system."

:::

:::{prf:definition} Interface $\mathcal{H}_0$
:label: def-interface-h0

**Purpose:** Ensures the system is a valid object in the topos with a notion of "existence."

**Required Structure ($\mathcal{D}$):**
- **State Object:** An object $\mathcal{X} \in \text{Obj}(\mathcal{E})$.
- **Evolution Morphism:** A family of endomorphisms $S_t: \mathcal{X} \to \mathcal{X}$ (the flow/algorithm).
- **Refinement Filter:** A topology or filtration $\mathcal{F}$ on $\mathcal{X}$ allowing limits (e.g., metric completion, domain theory limits).

**Evaluator ($\mathcal{P}_{\mathcal{H}_0}$):**
Is the morphism $S_t$ well-defined on the domain of interest?

$$\vdash S_t \in \text{Hom}_{\mathcal{E}}(\mathcal{X}, \mathcal{X})$$

**Certificates ($\mathcal{K}_{\mathcal{H}_0}$):**
- $K_{\mathcal{H}_0}^+$: A witness term $w : S_t$.
- $K_{\mathcal{H}_0}^-$: A witness that $\text{dom}(S_t) = \emptyset$ (Vacuous system).

**Does Not Promise:** Global existence. The refinement filter may exhaust at finite time.
:::

---

### $D_E$ (Energy Interface)
*The Cost Interface. Enables Node 1: EnergyCheck*

:::{div} feynman-prose

If the substrate interface asks "does this thing exist?", the energy interface asks "is it going somewhere sensible?"

Here is the intuition. Almost every well-behaved dynamical system has some quantity that decreases (or at least does not increase without bound) as time goes on. For fluids, it is kinetic energy. For optimization algorithms, it is the loss function. For theorem provers, it might be the complexity of the remaining goals.

This interface asks: what is your "height function"? How do you measure progress? And crucially: can you guarantee that the system does not climb forever?

The key inequality is $\Phi(S_t x) \leq \Phi(x) + \int \mathfrak{D}$. Read this as: "the energy at time $t$ is at most the initial energy plus whatever dissipation happened along the way." If dissipation is negative (the system is losing energy), this is a bound. If dissipation can be positive without limit, you have a potential blow-up.

:::

:::{prf:definition} Interface $D_E$
:label: def-interface-de

**Purpose:** Defines a mapping from States to Values, establishing an ordering on configurations.

**Required Structure ($\mathcal{D}$):**
- **Height Morphism:** $\Phi: \mathcal{X} \to \mathcal{H}$ (Energy / Entropy / Complexity).
- **Dissipation Morphism:** $\mathfrak{D}: \mathcal{X} \to \mathcal{H}$ (Rate of change).
- **Comparison Operator:** A relation $\leq$ on $\mathcal{H}$.

**Evaluator ($\mathcal{P}_1$ - EnergyCheck):**
Does the evolution map states to lower (or bounded) height values?

$$\Phi(S_t x) \leq \Phi(x) + \int \mathfrak{D}$$

**Certificates ($\mathcal{K}_{D_E}$):**
- $K_{D_E}^+$: A bound $B \in \mathcal{H}$.
- $K_{D_E}^-$: A path $\gamma: [0,1] \to \mathcal{X}$ where $\Phi(\gamma(t)) \to \infty$ (Blow-up).

**Does Not Promise:** That energy is actually bounded.
:::

---

### $\mathrm{Rec}_N$ (Recovery Interface)
*The Discrete Event Interface. Enables Node 2: ZenoCheck*

:::{prf:definition} Interface $\mathrm{Rec}_N$
:label: def-interface-recn

**Purpose:** Handles discrete transitions, surgeries, or logical steps.

**Required Structure ($\mathcal{D}$):**
- **Bad Subobject:** $\mathcal{B} \hookrightarrow \mathcal{X}$ (The singular locus or error states).
- **Recovery Map:** $\mathcal{R}: \mathcal{B} \to \mathcal{X} \setminus \mathcal{B}$ (The reset/surgery operator).
- **Counting Measure:** $\#: \text{Path}(\mathcal{X}) \to \mathbb{N}$ (Counting entrances to $\mathcal{B}$).

**Evaluator ($\mathcal{P}_2$ - ZenoCheck):**
Is the count of recovery events finite on finite intervals?

$$\#\{t \mid S_t(x) \in \mathcal{B}\} < \infty$$

**Certificates ($\mathcal{K}_{\mathrm{Rec}_N}$):**
- $K_{\mathrm{Rec}_N}^+$: An integer $N$.
- $K_{\mathrm{Rec}_N}^-$: An accumulation point $t_*$ (Zeno paradox).

**Does Not Promise:** That Zeno behavior is impossible.
:::

---

### $C_\mu$ (Compactness Interface)
*The Limit Interface. Enables Node 3: CompactCheck*

:::{prf:definition} Interface $C_\mu$
:label: def-interface-cmu

**Purpose:** Defines convergence and structure extraction.

**Required Structure ($\mathcal{D}$):**
- **Symmetry Group Object:** $G \in \text{Grp}(\mathcal{E})$ acting on $\mathcal{X}$.
- **Quotient Object:** $\mathcal{X} // G$ (The stack/moduli space).
- **Limit Operator:** $\lim: \text{Seq}(\mathcal{X} // G) \to \mathcal{X} // G$.

**Evaluator ($\mathcal{P}_3$ - CompactCheck):**
Does a bounded sequence have a limit object (Profile) in the quotient?

$$\exists V \in \mathcal{X} // G : x_n \to V$$

**Certificates ($\mathcal{K}_{C_\mu}$):**
- $K_{C_\mu}^+$ (Concentration): The profile object $V$ and the gauge sequence $\{g_n\}$.
- $K_{C_\mu}^-$ (Dispersion): A witness that the measure of the state vanishes (e.g., $L^\infty \to 0$).

**Does Not Promise:** Compactness. Dispersion ($K_{C_\mu}^-$) is a valid success state.
:::

---

### $\mathrm{SC}_\lambda$ (Scaling Interface)
*The Homogeneity Interface. Enables Node 4: ScaleCheck*

:::{prf:definition} Interface $\mathrm{SC}_\lambda$
:label: def-interface-sclambda

**Purpose:** Defines behavior under renormalization/rescaling.

**Required Structure ($\mathcal{D}$):**
- **Scaling Action:** An action of the multiplicative group $\mathbb{G}_m$ (or $\mathbb{R}^+$) on $\mathcal{X}$.
- **Weights:** Morphisms $\alpha, \beta: \mathcal{X} \to \mathbb{Q}$ defining how $\Phi$ and $\mathfrak{D}$ transform under scaling.
- **Critical Threshold:** A scalar $\lambda_c$ defining the subcritical window (typically $0$ in the homogeneous case).

**Evaluator ($\mathcal{P}_4$ - ScaleCheck):**
Are the exponents ordered correctly for stability?

$$\beta(V) - \alpha(V) < \lambda_c$$

*(Does cost grow faster than time compression?)*

**Certificates ($\mathcal{K}_{\mathrm{SC}_\lambda}$):**
- $K_{\mathrm{SC}_\lambda}^+$: The values $\alpha, \beta, \lambda_c$.
- $K_{\mathrm{SC}_\lambda}^-$: A witness of criticality ($\beta - \alpha = \lambda_c$) or supercriticality ($\beta - \alpha > \lambda_c$).

**Does Not Promise:** Subcriticality.
:::

---

### $\mathrm{SC}_{\partial c}$ (Parameter Interface)
*Enables Node 5: ParamCheck*

:::{prf:definition} Interface $\mathrm{SC}_{\partial c}$
:label: def-interface-scdc

**Purpose:** Defines stability of modulation parameters and coupling constants.

**Required Structure ($\mathcal{D}$):**
- **Parameter Object:** $\Theta \in \text{Obj}(\mathcal{E})$.
- **Parameter Morphism:** $\theta: \mathcal{X} \to \Theta$.
- **Reference Point:** $\theta_0: 1 \to \Theta$ (global section).
- **Distance Morphism:** $d: \Theta \times \Theta \to \mathcal{H}$.

**Evaluator ($\mathcal{P}_5$ - ParamCheck):**
Are structural constants stable along the trajectory?

$$\forall t.\, d(\theta(S_t x), \theta_0) \leq C$$

**Certificates ($\mathcal{K}_{\mathrm{SC}_{\partial c}}$):**
- $K_{\mathrm{SC}_{\partial c}}^+$: $(\theta_0, C, \text{stability proof})$.
- $K_{\mathrm{SC}_{\partial c}}^-$: $(\text{parameter drift witness}, t_{\text{drift}})$.

**Does Not Promise:** Parameter stability.
:::

---

### $\mathrm{Cap}_H$ (Capacity Interface)
*The Measure/Dimension Interface. Enables Node 6: GeomCheck*

:::{prf:definition} Interface $\mathrm{Cap}_H$
:label: def-interface-caph

**Purpose:** Quantifies the "size" of subobjects.

**Required Structure ($\mathcal{D}$):**
- **Capacity Functional:** $\text{Cap}: \text{Sub}(\mathcal{X}) \to \mathcal{H}$ (e.g., Hausdorff dim, Kolmogorov complexity, Channel capacity).
- **Threshold:** A critical value $C_{\text{crit}}: 1 \to \mathcal{H}$.
- **Singular Subobject:** $\Sigma \hookrightarrow \mathcal{X}$.

**Evaluator ($\mathcal{P}_6$ - GeomCheck):**
Is the capacity of the singular set below the threshold?

$$\text{Cap}(\Sigma) < C_{\text{crit}}$$

**Certificates ($\mathcal{K}_{\mathrm{Cap}_H}$):**
- $K_{\mathrm{Cap}_H}^+$: The value $\text{Cap}(\Sigma)$.
- $K_{\mathrm{Cap}_H}^-$: A measure-preserving map from a large object into $\Sigma$.

**Does Not Promise:** That singularities are small.
:::

---

### $\mathrm{LS}_\sigma$ (Stiffness Interface)
*The Local Convexity Interface. Enables Node 7: StiffnessCheck*

:::{prf:definition} Interface $\mathrm{LS}_\sigma$
:label: def-interface-lssigma

**Purpose:** Defines the local geometry of the potential landscape.

**Required Structure ($\mathcal{D}$):**
- **Gradient Operator:** $\nabla: \text{Hom}(\mathcal{X}, \mathcal{H}) \to T\mathcal{X}$ (Tangent bundle section).
- **Comparison:** An inequality relating gradient norm to height value.

**Evaluator ($\mathcal{P}_7$ - StiffnessCheck):**
Does the Łojasiewicz-Simon inequality hold?

$$\|\nabla \Phi(x)\| \geq C |\Phi(x) - \Phi(V)|^{1-\theta}$$

**Certificates ($\mathcal{K}_{\mathrm{LS}_\sigma}$):**
- $K_{\mathrm{LS}_\sigma}^+$: The exponent $\theta \in (0, 1]$.
- $K_{\mathrm{LS}_\sigma}^-$: A witness of flatness (e.g., a non-trivial kernel of the Hessian).

**Does Not Promise:** Convexity. Flat landscapes ($K_{\mathrm{LS}_\sigma}^-$) trigger the Spectral Barrier.
:::

---

### $\mathrm{Mon}_\phi$ (Monotonicity Interface)
*The Virial/Morawetz Interface. Enables Soft→Rigidity Compilation*

:::{prf:definition} Interface $\mathrm{Mon}_\phi$ (Monotonicity / Virial-Morawetz)
:label: def-interface-mon

**Purpose:** Defines monotonicity identities that force dispersion or concentration for almost-periodic solutions.

**Required Structure ($\mathcal{D}$):**
- **Monotonicity Functional:** $M: \mathcal{X} \times \mathbb{R} \to \mathbb{R}$ (Morawetz/virial action).
- **Weight Function:** $\phi: \mathcal{X} \to \mathbb{R}$ (typically radial or localized convex weight).
- **Sign Certificate:** $\sigma \in \{+, -, 0\}$ (convexity type determining inequality direction).

**Evaluator ($\mathcal{P}_{\mathrm{Mon}}$ - MonotonicityCheck):**
Does the monotonicity identity hold with definite sign for the declared functional?

$$\frac{d^2}{dt^2} M_\phi(t) \geq c \cdot \|\nabla u\|^2 - C \cdot \|u\|^2$$

(or $\leq$ depending on $\sigma$), where $M_\phi(t) = \int \phi(x) |u(t,x)|^2 dx$ or appropriate variant.

**Certificates ($\mathcal{K}_{\mathrm{Mon}_\phi}$):**
- $K_{\mathrm{Mon}_\phi}^+ := (\phi, M, \sigma, \mathsf{identity\_proof})$ asserting:
  1. The identity is algebraically verifiable from the equation structure
  2. For almost-periodic solutions mod $G$, integration forces dispersion or concentration
  3. The sign $\sigma$ is definite (not degenerate)
- $K_{\mathrm{Mon}_\phi}^- := \text{witness that no monotonicity identity holds with useful sign}$

**Evaluator (Computable for Good Types):**
- Check if equation has standard form (semilinear wave/Schrödinger/heat with power nonlinearity)
- Verify convexity of $\phi$ and compute second derivative identity algebraically
- Return YES with $K_{\mathrm{Mon}_\phi}^+$ if sign is definite; else NO with $K_{\mathrm{Mon}_\phi}^{\mathrm{inc}}$ (if verification method insufficient) or $K_{\mathrm{Mon}_\phi}^{\mathrm{wit}}$ (if sign is provably indefinite)

**Does Not Promise:** Rigidity directly. Combined with $K_{\mathrm{LS}_\sigma}^+$ and Lock obstruction, enables hybrid rigidity derivation.

**Used by:** MT-SOFT→Rigidity compilation metatheorem.

**Literature:** Morawetz estimates {cite}`Morawetz68`; virial identities {cite}`GlasseyScattering77`; interaction Morawetz {cite}`CollianderKeelStaffilaniTakaokaTao08`.
:::

---

### $\mathrm{TB}_\pi$ (Topology Interface)
*The Invariant Interface. Enables Node 8: TopoCheck*

:::{prf:definition} Interface $\mathrm{TB}_\pi$
:label: def-interface-tbpi

**Purpose:** Defines discrete sectors that cannot be continuously deformed into one another.

**Required Structure ($\mathcal{D}$):**
- **Sector Set:** A discrete set $\pi_0(\mathcal{X})$ (Connected components, homotopy classes).
- **Invariant Map:** $\tau: \mathcal{X} \to \pi_0(\mathcal{X})$.

**Evaluator ($\mathcal{P}_8$ - TopoCheck):**
Is the trajectory confined to a single sector?

$$\tau(S_t x) = \tau(x)$$

**Certificates ($\mathcal{K}_{\mathrm{TB}_\pi}$):**
- $K_{\mathrm{TB}_\pi}^+$: The value $\tau$.
- $K_{\mathrm{TB}_\pi}^-$: A path connecting two distinct sectors (Tunneling/Topology change).

**Does Not Promise:** Topological stability.
:::

---

### $\mathrm{TB}_O$ (Tameness Interface)
*Enables Node 9: TameCheck*

:::{prf:definition} Interface $\mathrm{TB}_O$
:label: def-interface-tbo

**Purpose:** Defines the "tameness" of the singular locus via definability.

**Required Structure ($\mathcal{D}$):**
- **Definability Modality:** $\text{Def}: \text{Sub}(\mathcal{X}) \to \Omega$.
- **Tame Structure:** $\mathcal{O} \hookrightarrow \text{Sub}(\mathcal{E})$ (sub-Boolean algebra of definable subobjects).

**Evaluator ($\mathcal{P}_9$ - TameCheck):**
Is the singular locus $\mathcal{O}$-definable?

$$\Sigma \in \mathcal{O}\text{-definable}$$

**Certificates ($\mathcal{K}_{\mathrm{TB}_O}$):**
- $K_{\mathrm{TB}_O}^+$: $(\text{tame structure}, \text{definability proof})$.
- $K_{\mathrm{TB}_O}^-$: $(\text{wildness witness})$.

**Does Not Promise:** Tameness. Wild topology ($K_{\mathrm{TB}_O}^-$) routes to the O-Minimal Barrier.
:::

---

### $\mathrm{TB}_\rho$ (Mixing Interface)
*Enables Node 10: ErgoCheck*

:::{prf:definition} Interface $\mathrm{TB}_\rho$
:label: def-interface-tbrho

**Purpose:** Defines ergodic/mixing properties of the dynamics.

**Required Structure ($\mathcal{D}$):**
- **Measure Object:** $\mathcal{M}(\mathcal{X})$ (probability measures internal to $\mathcal{E}$).
- **Invariant Subobject:** $\text{Inv}_S \hookrightarrow \mathcal{M}(\mathcal{X})$.
- **Mixing Time Morphism:** $\tau_{\text{mix}}: \mathcal{X} \to \mathcal{H}$.

**Evaluator ($\mathcal{P}_{10}$ - ErgoCheck):**
Does the system mix with finite mixing time?

$$\tau_{\text{mix}}(x) < \infty$$

**Certificates ($\mathcal{K}_{\mathrm{TB}_\rho}$):**
- $K_{\mathrm{TB}_\rho}^+$: $(\tau_{\text{mix}}, \text{mixing proof})$.
- $K_{\mathrm{TB}_\rho}^-$: $(\text{trap certificate}, \text{invariant subset})$.

**Does Not Promise:** Mixing.
:::

---

### $\mathrm{Rep}_K$ (Dictionary Interface)
*The Equivalence Interface. Enables Node 11: ComplexCheck*

:::{prf:definition} Interface $\mathrm{Rep}_K$
:label: def-interface-repk

**Purpose:** Defines the mapping between the "Territory" (System) and the "Map" (Representation).

**Required Structure ($\mathcal{D}$):**
- **Language Object:** $\mathcal{L} \in \text{Obj}(\mathcal{E})$ (formal language or category).
- **Dictionary Morphism:** $D: \mathcal{X} \to \mathcal{L}$.
- **Faithfulness:** An inverse map $D^{-1}$ or equivalence witness.
- **Complexity:** $K: \mathcal{L} \to \mathbb{N}_\infty$.

**Evaluator ($\mathcal{P}_{11}$ - ComplexCheck):**
Is the state representable with finite complexity?

$$K(D(x)) < \infty$$

**Stochastic Extension:** For stochastic systems (e.g., post-S12), complexity refers to the Kolmogorov complexity of the probability law $K(\mu)$, defined as the shortest program that samples from the distribution. Formally: $K(\mu) := \min\{|p| : U(p, r) \sim \mu \text{ for random } r\}$. This ensures that SDEs with finite-description coefficients $(b, \sigma)$ satisfy the complexity check even though individual sample paths are algorithmically random.

**Computability Warning:** $K(\mu)$ is uncomputable in general (Rice's Theorem for distributions). Consequently, $\mathrm{Rep}_K$ for stochastic systems typically returns $K^{\mathrm{inc}}$ unless an explicit program witness $p$ with $U(p, r) \sim \mu$ is provided by the user. The framework remains sound—$K^{\mathrm{inc}}$ routes Lock to geometry-only tactics (E1--E3).

**Certificates ($\mathcal{K}_{\mathrm{Rep}_K}$):**
- $K_{\mathrm{Rep}_K}^+$: The code/description $p$.
- $K_{\mathrm{Rep}_K}^-$: A proof of uncomputability or undecidability.

**Does Not Promise:** Computability.

**Epistemic Role:** $\mathrm{Rep}_K$ is the boundary between "analysis engine" and "conjecture prover engine." When $\mathrm{Rep}_K$ produces a NO-inconclusive certificate ($K_{\mathrm{Rep}_K}^{\mathrm{inc}}$), the Lock uses only geometric tactics (E1--E3).
:::

---

### $\mathrm{GC}_\nabla$ (Gradient Interface)
*The Geometry Interface. Enables Node 12: OscillateCheck*

:::{prf:definition} Interface $\mathrm{GC}_\nabla$
:label: def-interface-gcnabla

**Purpose:** Defines the "Natural" geometry of the space.

**Required Structure ($\mathcal{D}$):**
- **Metric Tensor:** $g: T\mathcal{X} \otimes T\mathcal{X} \to \mathcal{H}$ (Inner product).
- **Compatibility:** A relation between the flow vector field $v$ and the potential $\Phi$:

$$v \stackrel{?}{=} -\nabla_g \Phi$$

**Evaluator ($\mathcal{P}_{12}$ - OscillateCheck):**
Does the system follow the gradient?

$$\mathfrak{D}(x) = \|\nabla_g \Phi(x)\|^2$$

**Certificates ($\mathcal{K}_{\mathrm{GC}_\nabla}$):**
- $K_{\mathrm{GC}_\nabla}^+$ (Oscillation Present): Witness of oscillatory behavior (symplectic structure, curl, or non-gradient dynamics).
- $K_{\mathrm{GC}_\nabla}^-$ (Gradient Flow): Witness that flow is monotonic (no oscillation, pure gradient descent).

**Does Not Promise:** Absence of oscillation.

**Optionality:** $\mathrm{GC}_\nabla$ is not required for basic singularity exclusion. It only unlocks "explicit Lyapunov/action reconstruction" upgrades.
:::

---

### Open System Interfaces
*Enables Nodes 13-16: BoundaryCheck, OverloadCheck, StarveCheck, AlignCheck*

The open system checks are split into four distinct interfaces, each handling a specific aspect of boundary coupling.

#### $\mathrm{Bound}_\partial$ (Boundary Interface)
*Enables Node 13: BoundaryCheck*

:::{prf:definition} Interface $\mathrm{Bound}_\partial$
:label: def-interface-bound-partial

**Purpose:** Determines whether the system is open (has external boundary).

**Required Structure ($\mathcal{D}$):**
- **State Space:** $\mathcal{X}$ with topological boundary $\partial\mathcal{X}$.

**Evaluator ($\mathcal{P}_{13}$ - BoundaryCheck):**
Is the system open? Does it have a non-trivial boundary?

$$\partial\mathcal{X} \neq \emptyset$$

**Certificates ($\mathcal{K}_{\mathrm{Bound}_\partial}$):**
- $K_{\mathrm{Bound}_\partial}^+$ (Open System): Witness that boundary exists and is non-trivial.
- $K_{\mathrm{Bound}_\partial}^-$ (Closed System): Witness that system is closed; skip to Node 17.
:::

#### $\mathrm{Bound}_B$ (Input Bound Interface)
*Enables Node 14: OverloadCheck*

:::{prf:definition} Interface $\mathrm{Bound}_B$
:label: def-interface-bound-b

**Purpose:** Verifies that external inputs are bounded.

**Required Structure ($\mathcal{D}$):**
- **Input Object:** $\mathcal{U} \in \text{Obj}(\mathcal{E})$.
- **Input Morphism:** $\iota: \mathcal{U} \to \mathcal{X}$ (or $\mathcal{U} \times \mathcal{T} \to \mathcal{X}$).

**Evaluator ($\mathcal{P}_{14}$ - OverloadCheck):**
Is the input bounded in authority?

$$\|Bu\|_{L^\infty} \leq M \quad \land \quad \int_0^T \|u(t)\|^2 dt < \infty$$

**Certificates ($\mathcal{K}_{\mathrm{Bound}_B}$):**
- $K_{\mathrm{Bound}_B}^+$ (Bounded Input): $(\text{bound } M, \text{authority margin})$.
- $K_{\mathrm{Bound}_B}^-$ (Overload): $(\text{overload witness}, t^*)$ — triggers BarrierBode.
:::

#### $\mathrm{Bound}_{\Sigma}$ (Resource Interface)
*Enables Node 15: StarveCheck*

:::{prf:definition} Interface $\mathrm{Bound}_{\Sigma}$
:label: def-interface-bound-int

**Purpose:** Verifies that resource/energy supply is sufficient.

**Required Structure ($\mathcal{D}$):**
- **Resource Function:** $r: \mathcal{T} \to \mathbb{R}_{\geq 0}$.
- **Minimum Threshold:** $r_{\min} > 0$.

**Evaluator ($\mathcal{P}_{15}$ - StarveCheck):**
Is the integrated resource supply sufficient?

$$\int_0^T r(t) \, dt \geq r_{\min}$$

**Certificates ($\mathcal{K}_{\mathrm{Bound}_{\Sigma}}$):**
- $K_{\mathrm{Bound}_{\Sigma}}^+$ (Sufficient Supply): $(r_{\min}, \text{sufficiency proof})$.
- $K_{\mathrm{Bound}_{\Sigma}}^-$ (Starvation): $(\text{deficit time})$ — triggers BarrierInput.
:::

#### $\mathrm{GC}_T$ (Control Alignment Interface)
*Enables Node 16: AlignCheck*

:::{prf:definition} Interface $\mathrm{GC}_T$
:label: def-interface-gc-t

**Purpose:** Verifies that control inputs align with safe descent directions.

**Required Structure ($\mathcal{D}$):**
- **Control Law:** $T: \mathcal{U} \to \mathcal{X}$ (the realized control).
- **Desired Behavior:** $d \in \mathcal{Y}$ (the reference or goal).
- **Alignment Metric:** Distance function $\Delta: \mathcal{Y} \times \mathcal{Y} \to \mathbb{R}_{\geq 0}$.

**Evaluator ($\mathcal{P}_{16}$ - AlignCheck):**
Is the control matched to the desired behavior?

$$\Delta(T(u), d) \leq \varepsilon_{\text{align}}$$

**Certificates ($\mathcal{K}_{\mathrm{GC}_T}$):**
- $K_{\mathrm{GC}_T}^+$ (Aligned Control): $(\text{alignment certificate}, \Delta_{\text{achieved}})$.
- $K_{\mathrm{GC}_T}^-$ (Misaligned): $(\text{misalignment mode})$ — triggers BarrierVariety.
:::

---

### $\mathrm{Cat}_{\mathrm{Hom}}$ (Categorical Interface)
*Enables Node 17: The Lock (BarrierExclusion)*

:::{div} feynman-prose

And here we arrive at the Lock - the final gate, the moment of truth.

All the previous interfaces were about checking specific properties: energy bounds, compactness, scaling, topology. But they do not directly answer the question we really care about: will the system blow up or not?

The Lock approaches this differently. It asks: "I have a library of known bad patterns - the canonical ways that singularities can form in systems of your type. Can any of these bad patterns embed into your system?"

This is a categorical question. A "morphism" from a bad pattern $B$ to your hypostructure $\mathcal{H}$ would mean that $B$ can be realized inside $\mathcal{H}$ - that your system contains the seeds of singularity. If no such morphism exists (the Hom-set is empty), then your system is structurally forbidden from developing that kind of singularity.

The beautiful thing is that this reduces an analytic question (will something blow up?) to an algebraic question (do certain morphisms exist?). And algebraic questions are often much more tractable.

:::

:::{prf:definition} Interface $\mathrm{Cat}_{\mathrm{Hom}}$
:label: def-interface-cathom

**Purpose:** Final structural consistency verification. Certifies that no bad pattern from the library embeds into the candidate hypostructure, establishing global regularity.

**Required Structure ($\mathcal{D}$):**
- **Hypostructure Category:** $\mathbf{Hypo}_T$ — the category of admissible hypostructures for type $T$.
- **Bad Pattern Library:** $\mathcal{B} = \{B_i\}_{i \in I}$ — a finite set of *minimal bad patterns* committed to for problem type $T$. Each $B_i \in \text{Obj}(\mathbf{Hypo}_T)$ is a canonical singularity-forming structure.
- **Morphism Spaces:** $\text{Hom}_{\mathbf{Hypo}_T}(B_i, \mathcal{H})$ for each $B_i \in \mathcal{B}$.

**Completeness Axiom (Problem-Type Dependent):**
For each problem type $T$, we assume: *every singularity of type $T$ factors through some $B_i \in \mathcal{B}$.* This is a **problem-specific axiom** that must be verified for each instantiation (e.g., for Navier-Stokes, the library consists of known blow-up profiles; for Riemann Hypothesis, the library consists of zero-causing structures).

**Evaluator ($\mathcal{P}_{17}$ - BarrierExclusion):**

$$\forall i \in I: \text{Hom}_{\mathbf{Hypo}_T}(B_i, \mathcal{H}) = \emptyset$$

The Lock evaluator checks whether any morphism exists from any bad pattern to the candidate hypostructure. If all Hom-sets are empty, no singularity-forming pattern can embed, and global regularity follows.

**Certificates ($\mathcal{K}_{\mathrm{Cat}_{\mathrm{Hom}}}$):**
- $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$ (Blocked/VICTORY): Proof that $\forall i: \text{Hom}(B_i, \mathcal{H}) = \emptyset$. Techniques include:
  - **E1 (Dimension):** $\dim(B_i) > \dim(\mathcal{H})$
  - **E2 (Invariant Mismatch):** $I(B_i) \neq I(\mathcal{H})$ for preserved invariant $I$
  - **E3 (Positivity/Integrality):** Obstruction from positivity or integrality constraints
  - **E4 (Functional Equation):** No solution to induced functional equations
  - **E5 (Modular):** Obstruction from modular/arithmetic properties
- $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{morph}}$ (Breached/FATAL): Explicit morphism $f: B_i \to \mathcal{H}$ for some $i$, witnessing that singularity formation is possible.

**Does Not Promise:** That the Lock is decidable. Tactics E1-E13 may exhaust without resolution, yielding a Breached-inconclusive certificate ($K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{br\text{-}inc}}$).

**Remark (Library vs. Universal Object):**
The universal bad object $\mathbb{H}_{\mathrm{bad}}^{(T)}$ is well-defined as the colimit over the **small** germ set $\mathcal{G}_T$ (see {prf:ref}`mt-krnl-exclusion`, Initiality Lemma). The Small Object Argument ({cite}`Quillen67` §II.3) ensures $\mathcal{G}_T$ is a genuine set by exploiting energy bounds and symmetry quotients. The library formulation $\mathcal{B} = \{B_i\}_{i \in I}$ is the **constructive implementation**: it provides a finite list of computable representatives. The density theorem {prf:ref}`mt-fact-germ-density` proves that checking $\mathcal{B}$ suffices to verify the full categorical obstruction.
:::

---

### What base level Does NOT Promise

At the soft layer, the framework does **not** assume:

- **Global regularity**: Solutions may blow up; the sieve determines whether they must
- **Finite canonical profile library**: Profiles may be classified, stratified, or horizon
- **Surgery admissibility**: Each surgery must be checked via the Admissibility Trichotomy
- **Decidable Lock outcomes**: E1--E13 tactics may exhaust without resolution
- **Unique gradient structure**: $\mathrm{GC}_\nabla$ is optional; many systems lack gradient form
- **Closed system**: $\mathrm{Bound}$ explicitly handles open systems with inputs

These are obtained by **upgrades**:
- Profile Classification Trichotomy (finite library, tame stratification, or horizon)
- Surgery Admissibility Trichotomy (admissible, admissible$^\sim$, or inadmissible)
- Promotion rules (blocked $\to$ YES via certificate accumulation)
- Lock tactics (E1--E13 for Hom-emptiness proofs)

This separation makes the framework **honest about its assumptions** and enables systematic identification of what additional structure is needed when an inconclusive NO certificate ($K^{\mathrm{inc}}$) is produced.

---

### Backend-Specific Permits (Specialized Certificates)

The following permits capture **backend-specific hypotheses** that are required by particular metatheorems. Unlike the universal interfaces (8.1–8.15), these permits encode deep theorems from specific mathematical domains (e.g., dispersive PDE, dynamical systems, algebraic geometry) that cannot be derived from the generic interface structure alone.

:::{prf:definition} Critical Index and Critical Phase Space
:label: def-critical-index

A type $T$ with Scaling Interface data ({prf:ref}`def-interface-sclambda`) specifies:
1. A scaling action $\rho_\lambda$ on thin states $\mathcal{X}$, and
2. A scale family of control seminorms $\{\|\cdot\|_s\}_{s \in \mathcal{S}}$ declared by the type
   (for PDE this is typically Sobolev $s$, but for non-PDE types it may be a spectral,
   complexity, or filtration index).

The **critical index** $s_c$ is any $s$ such that the control norm is scale-invariant:

$$
\|\rho_\lambda u\|_{s_c} = \|u\|_{s_c} \quad \text{for all admissible } \lambda.
$$

The **critical phase space** $X_c$ is the completion of thin states under $\|\cdot\|_{s_c}$.

**Certification note:** If the scale family $\{\|\cdot\|_s\}$ is not declared or the
invariance check cannot be certified, then $s_c$ is **undefined** and any statement
depending on $s_c$ is conditional on the scaling-data certificate.
:::

:::{prf:definition} Permit $\mathrm{WP}_{s_c}$ (Critical Well-Posedness + Continuation)
:label: def-permit-wp-sc

**Name:** CriticalWP

**Question:** Does the evolution problem $T$ admit local well-posedness in the critical phase space $X_c$ (typically $X_c = \dot{H}^{s_c}$), with a continuation criterion?

**YES certificate**

$$K_{\mathrm{WP}_{s_c}}^+ := \big(\mathsf{LWP},\ \mathsf{uniq},\ \mathsf{cont},\ \mathsf{crit\_blowup}\big)$$

where the payload asserts all of:
1. (**Local existence**) For every $u_0 \in X_c$ there exists $T(u_0) > 0$ and a solution $u \in C([0,T]; X_c)$.
2. (**Uniqueness**) The solution is unique in the specified solution class.
3. (**Continuous dependence**) The data-to-solution map is continuous (or Lipschitz) on bounded sets in $X_c$.
4. (**Continuation criterion**) If $T_{\max} < \infty$ then a specified *critical control norm* blows up:

   $$\|u\|_{S([0, T_{\max}))} = \infty \quad (\text{for a declared control norm } S).$$

**NO certificate** (sum type $K_{\mathrm{WP}_{s_c}}^- := K_{\mathrm{WP}_{s_c}}^{\mathrm{wit}} \sqcup K_{\mathrm{WP}_{s_c}}^{\mathrm{inc}}$)

*NO-with-witness:*

$$K_{\mathrm{WP}_{s_c}}^{\mathrm{wit}} := (\mathsf{counterexample}, \mathsf{mode})$$

where $\mathsf{mode} \in \{\texttt{NORM\_INFLATION}, \texttt{NON\_UNIQUE}, \texttt{ILL\_POSED}, \texttt{NO\_CONTINUATION}\}$ identifies which of (1)–(4) fails, with an explicit counterexample (e.g., a sequence demonstrating norm inflation, or a pair of distinct solutions from identical data).

*NO-inconclusive:*

$$K_{\mathrm{WP}_{s_c}}^{\mathrm{inc}} := (\mathsf{obligation}, \mathsf{missing}, \mathsf{failure\_code}, \mathsf{trace})$$

Typical $\mathsf{missing}$: "no matching WP template (parabolic/dispersive/hyperbolic)", "state space $X_c$ not recognized", "operator conditions not provided by soft layer".

**Used by:** `mt-auto-profile` Mechanism A (CC+Rig), and any node that invokes "critical LWP + continuation".
:::

:::{prf:definition} Permit $\mathrm{ProfDec}_{s_c,G}$ (Profile Decomposition modulo Symmetries)
:label: def-permit-profdec-scg

**Name:** ProfileDecomp

**Question:** Does every bounded sequence in $X_c$ admit a Bahouri–Gérard/Lions type profile decomposition modulo the symmetry group $G$?

**YES certificate**

$$K_{\mathrm{ProfDec}_{s_c,G}}^+ := \big(\{\phi^j\}_{j \geq 1},\ \{g_n^j\}_{n,j},\ \{r_n^J\}_{n,J},\ \mathsf{orth},\ \mathsf{rem}\big)$$

meaning: for every bounded $(u_n) \subset X_c$ there exist profiles $\phi^j \in X_c$ and symmetry parameters $g_n^j \in G$ such that for every $J$,

$$u_n = \sum_{j=1}^J g_n^j \phi^j + r_n^J,$$

with:
1. (**Asymptotic orthogonality**) The parameters $(g_n^j)$ are pairwise orthogonal in the standard sense for $G$.
2. (**Decoupling**) Conserved quantities/energies decouple across profiles up to $o_n(1)$ errors.
3. (**Remainder smallness**) The remainder $r_n^J$ is small in the critical control norm:

   $$\lim_{J \to \infty}\ \limsup_{n \to \infty}\ \|r_n^J\|_S = 0.$$

**NO certificate** (sum type $K_{\mathrm{ProfDec}}^- := K_{\mathrm{ProfDec}}^{\mathrm{wit}} \sqcup K_{\mathrm{ProfDec}}^{\mathrm{inc}}$)

*NO-with-witness:*

$$K_{\mathrm{ProfDec}}^{\mathrm{wit}} := (\mathsf{bounded\_seq}, \mathsf{failed\_property})$$

where $\mathsf{failed\_property} \in \{\texttt{NO\_ORTH}, \texttt{NO\_DECOUPLE}, \texttt{NO\_REMAINDER\_SMALL}\}$ identifies which of (1)–(3) fails, with a concrete bounded sequence $(u_n)$ demonstrating the failure.

*NO-inconclusive:*

$$K_{\mathrm{ProfDec}}^{\mathrm{inc}} := (\mathsf{obligation}, \mathsf{missing}, \mathsf{failure\_code}, \mathsf{trace})$$

Typical $\mathsf{missing}$: "symmetry group $G$ not recognized as standard decomposition group", "control norm $S$ not provided or checkable", "space not in supported class (Hilbert/Banach with required compactness structure)".

**Used by:** `mt-auto-profile` Mechanism A (CC+Rig).

**Literature:** {cite}`BahouriGerard99`; {cite}`Lions84`; {cite}`Lions85`.
:::

:::{prf:definition} Permit $\mathrm{KM}_{\mathrm{CC+stab}}$ (Concentration–Compactness + Stability Machine)
:label: def-permit-km-ccstab

**Name:** KM-Machine

**Question:** Can failure of the target property (regularity/scattering/etc.) be reduced to a *minimal counterexample* that is almost periodic modulo symmetries, using concentration–compactness plus a perturbation/stability lemma?

**YES certificate**

$$K_{\mathrm{KM}_{\mathrm{CC+stab}}}^+ := \big(\mathsf{min\_obj},\ \mathsf{ap\_modG},\ \mathsf{stab},\ \mathsf{nl\_profiles}\big)$$

where the payload asserts:
1. (**Minimal counterexample extraction**) If the target property fails, there exists a solution $u^*$ minimal with respect to a declared size functional (energy/mass/critical norm threshold).
2. (**Almost periodicity**) The orbit $\{u^*(t)\}$ is precompact in $X_c$ modulo $G$ ("almost periodic mod $G$").
3. (**Long-time perturbation**) A stability lemma: any approximate solution close in the control norm remains close to an exact solution globally on the interval.
4. (**Nonlinear profile control**) The nonlinear evolution decouples across profiles to the extent needed for the minimal-element argument.

**NO certificate** (sum type $K_{\mathrm{KM}}^- := K_{\mathrm{KM}}^{\mathrm{wit}} \sqcup K_{\mathrm{KM}}^{\mathrm{inc}}$)

*NO-with-witness:*

$$K_{\mathrm{KM}}^{\mathrm{wit}} := (\mathsf{failure\_obj}, \mathsf{step\_failed})$$

where $\mathsf{step\_failed} \in \{\texttt{NO\_MIN\_EXTRACT}, \texttt{NO\_ALMOST\_PERIODIC}, \texttt{NO\_STABILITY}, \texttt{NO\_PROFILE\_CONTROL}\}$ identifies which of (1)–(4) fails, with a concrete object demonstrating the failure.

*NO-inconclusive:*

$$K_{\mathrm{KM}}^{\mathrm{inc}} := (\mathsf{obligation}, \mathsf{missing}, \mathsf{failure\_code}, \mathsf{trace})$$

Typical $\mathsf{missing}$: "composition requires $K_{\mathrm{WP}}^+$ which was not derived", "profile decomposition not available", "stability lemma not computable for this equation class".

**Used by:** `mt-auto-profile` Mechanism A (CC+Rig).

**Literature:** {cite}`KenigMerle06`; {cite}`KillipVisan10`; {cite}`DuyckaertsKenigMerle11`.
:::

:::{prf:definition} Permit $\mathrm{Attr}^+$ (Global Attractor Existence)
:label: def-permit-attractor

**Name:** GlobalAttractor

**Question:** Does the semiflow $(S_t)_{t \geq 0}$ on a phase space $X$ admit a compact global attractor?

**YES certificate**

$$K_{\mathrm{Attr}}^+ := (\mathsf{semiflow},\ \mathsf{absorbing},\ \mathsf{asymp\_compact},\ \mathsf{attractor})$$

asserting:
1. (**Semiflow structure**) $S_{t+s} = S_t \circ S_s$, $S_0 = \mathrm{id}$, and $S_t$ is continuous on bounded sets.
2. (**Dissipativity**) There exists a bounded absorbing set $B \subset X$.
3. (**Asymptotic compactness**) For any bounded $B_0 \subset X$ and any $t_n \to \infty$, the set $S_{t_n}(B_0)$ has precompact closure.
4. (**Attractor**) There exists a compact invariant set $\mathcal{A}$ attracting bounded sets:

   $$\mathrm{dist}(S_t(B_0), \mathcal{A}) \to 0 \quad (t \to \infty).$$

**NO certificate** (sum type $K_{\mathrm{Attr}}^- := K_{\mathrm{Attr}}^{\mathrm{wit}} \sqcup K_{\mathrm{Attr}}^{\mathrm{inc}}$)

*NO-with-witness:*

$$K_{\mathrm{Attr}}^{\mathrm{wit}} := (\mathsf{obstruction}, \mathsf{type})$$

where $\mathsf{type} \in \{\texttt{NO\_SEMIFLOW}, \texttt{NO\_ABSORBING\_SET}, \texttt{NO\_ASYMP\_COMPACT}, \texttt{NO\_ATTRACTOR}\}$ identifies which of (1)–(4) fails, with a concrete obstruction object.

*NO-inconclusive:*

$$K_{\mathrm{Attr}}^{\mathrm{inc}} := (\mathsf{obligation}, \mathsf{missing}, \mathsf{failure\_code}, \mathsf{trace})$$

Typical $\mathsf{missing}$: "cannot verify asymptotic compactness from current soft interfaces", "Temam-Raugel template requires compactness lemma not provided", "insufficient bounds to certify absorbing set".

**Used by:** `mt-auto-profile` Mechanism B (Attr+Morse) and any node invoking global attractor machinery.

**Literature:** {cite}`Temam97`; {cite}`Raugel02`; {cite}`HaleBook88`.
:::

:::{prf:definition} Permit $\mathrm{DegImage}_m$ (Degree-of-Image Bound for Degree-$m$ Maps)
:label: def-permit-degimage

**Name:** DegImageBound

**Question:** For the chosen "compression map" $\phi$ of (algebraic) degree $\leq m$, does the standard degree inequality for images apply in your setting?

**YES certificate**

$$K_{\mathrm{DegImage}_m}^+ := (\phi,\ \mathsf{model},\ \mathsf{basepointfree},\ \mathsf{deg\_ineq})$$

asserting:
1. (**Model choice fixed**) You specify whether $\phi$ is a morphism $W \to \mathbb{P}^N$, or a rational map represented via its graph / resolution of indeterminacy.
2. (**Base-point-free representation**) After the chosen resolution/graph step, $\phi$ is induced by a base-point-free linear system of degree $\leq m$.
3. (**Degree inequality**) For projective closures, the inequality holds:

   $$\deg(\overline{\phi(W)}) \leq m^{\dim W} \cdot \deg(W)$$

   (or your preferred standard variant with the same monotone dependence on $m$).

**NO certificate** (sum type $K_{\mathrm{DegImage}_m}^- := K_{\mathrm{DegImage}_m}^{\mathrm{wit}} \sqcup K_{\mathrm{DegImage}_m}^{\mathrm{inc}}$)

*NO-with-witness:*

$$K_{\mathrm{DegImage}_m}^{\mathrm{wit}} := (\mathsf{map\_model}, \mathsf{violation})$$

where $\mathsf{violation} \in \{\texttt{NOT\_BPF}, \texttt{DEGREE\_EXCEEDS}, \texttt{INDETERMINACY\_UNRESOLVABLE}\}$ specifies which hypothesis fails with a concrete witness (e.g., a base locus, or a degree computation exceeding the bound).

*NO-inconclusive:*

$$K_{\mathrm{DegImage}_m}^{\mathrm{inc}} := (\mathsf{obligation}, \mathsf{missing}, \mathsf{failure\_code}, \mathsf{trace})$$

Typical $\mathsf{missing}$: "resolution of indeterminacy not computable", "degree of image not algorithmically determinable for this variety class", "base-point-free verification requires Bertini-type theorem not available".

**Used by:** `def-e12` Backend C (morphism/compression).

**Literature:** {cite}`Lazarsfeld04`; {cite}`Fulton84`.
:::

:::{prf:definition} Permit $\mathrm{CouplingSmall}^+$ (Coupling Control in Product Regularity)
:label: def-permit-couplingsmall

**Name:** CouplingSmall

**Question:** Is the interaction term $\Phi_{\mathrm{int}}$ controlled strongly enough (in the norms used by $K_{\mathrm{Cat}_{\mathrm{Hom}}}^A, K_{\mathrm{Cat}_{\mathrm{Hom}}}^B$) to prevent the coupling from destroying the component bounds?

**YES certificate**

$$K_{\mathrm{CouplingSmall}}^+ := (\varepsilon,\ C_\varepsilon,\ \mathsf{bound\_form},\ \mathsf{closure})$$

asserting the existence of an inequality of one of the following standard "closure" types (declare which one you use):
- (**Energy absorbability**) For a product energy $E = E_A + E_B$,

  $$\left|\frac{d}{dt} E_{\mathrm{int}}(t)\right| \leq \varepsilon \, E(t) + C_\varepsilon,$$

  with $\varepsilon$ small enough to be absorbed by dissipation/Grönwall.
- (**Relative boundedness**) $\Phi_{\mathrm{int}}$ is bounded or relatively bounded w.r.t. the product generator (for semigroup closure).
- (**Local Lipschitz + small parameter**) $\|\Phi_{\mathrm{int}}(u_A, u_B)\| \leq \varepsilon \, F(\|u_A\|, \|u_B\|) + C$ with $\varepsilon$ in the regime required by the bootstrap.

**NO certificate** (sum type $K_{\mathrm{CouplingSmall}}^- := K_{\mathrm{CouplingSmall}}^{\mathrm{wit}} \sqcup K_{\mathrm{CouplingSmall}}^{\mathrm{inc}}$)

*NO-with-witness:*

$$K_{\mathrm{CouplingSmall}}^{\mathrm{wit}} := (\mathsf{interaction}, \mathsf{unbounded\_mode})$$

where $\mathsf{unbounded\_mode} \in \{\texttt{ENERGY\_SUPERLINEAR}, \texttt{NOT\_REL\_BOUNDED}, \texttt{LIPSCHITZ\_FAILS}\}$ specifies which closure-usable bound fails, with a concrete sequence/trajectory demonstrating growth.

*NO-inconclusive:*

$$K_{\mathrm{CouplingSmall}}^{\mathrm{inc}} := (\mathsf{obligation}, \mathsf{missing}, \mathsf{failure\_code}, \mathsf{trace})$$

Typical $\mathsf{missing}$: "absorbability constant $\varepsilon$ not computable from current interfaces", "relative boundedness requires spectral information not provided", "Lipschitz constant estimation exceeds available bounds".

**Used by:** `mt-product` Backend A (when "subcritical scaling" is intended to imply analytic absorbability), and as a general interface to justify persistence of Lock bounds under coupling.
:::

:::{prf:definition} Permit $\mathrm{ACP}^+$ (Abstract Cauchy Problem Formulation)
:label: def-permit-acp

**Name:** AbstractCauchyProblem

**Question:** Can the dynamics be represented (equivalently, in the sense you require) as an abstract Cauchy problem on a Banach/Hilbert space?

**YES certificate**

$$K_{\mathrm{ACP}}^+ := (X,\ A,\ D(A),\ \mathsf{mild},\ \mathsf{equiv})$$

asserting:
1. (**State space**) A Banach/Hilbert space $X$ is fixed for the evolution state.
2. (**Generator**) A (possibly nonlinear) operator $A$ with declared domain $D(A)$ is specified such that the evolution is

   $$u'(t) = A(u(t)) \quad (\text{or } u'(t) = Au(t) + F(u(t)) \text{ in the semilinear case}).$$

3. (**Mild/strong solutions**) A mild formulation exists (e.g., Duhamel/variation of constants) in the class used by the Sieve.
4. (**Equivalence**) Solutions in the analytic/PDE sense correspond to (mild/strong) solutions of the ACP in the time intervals under consideration.

**NO certificate** (sum type $K_{\mathrm{ACP}}^- := K_{\mathrm{ACP}}^{\mathrm{wit}} \sqcup K_{\mathrm{ACP}}^{\mathrm{inc}}$)

*NO-with-witness:*

$$K_{\mathrm{ACP}}^{\mathrm{wit}} := (\mathsf{space\_candidate}, \mathsf{obstruction})$$

where $\mathsf{obstruction} \in \{\texttt{NO\_GENERATOR}, \texttt{DOMAIN\_MISMATCH}, \texttt{MILD\_FAILS}, \texttt{EQUIV\_BREAKS}\}$ specifies which of (1)–(4) fails, with a concrete witness (e.g., a solution in the PDE sense not representable in the ACP framework).

*NO-inconclusive:*

$$K_{\mathrm{ACP}}^{\mathrm{inc}} := (\mathsf{obligation}, \mathsf{missing}, \mathsf{failure\_code}, \mathsf{trace})$$

Typical $\mathsf{missing}$: "generator domain $D(A)$ not characterizable from soft interfaces", "mild solution formula requires semigroup estimates not provided", "equivalence of solution notions requires regularity theory beyond current scope".

**Used by:** `mt-product` Backend B (semigroup/perturbation route), and anywhere you invoke generator/semigroup theorems.

**Literature:** {cite}`EngelNagel00`; {cite}`Pazy83`.
:::

---

:::{prf:definition} Permit $\mathrm{Rigidity}_T^+$ (Rigidity / No-Minimal-Counterexample Theorem)
:label: def-permit-rigidity

**Name:** Rigidity

**Question:** Given an almost-periodic (mod symmetries) minimal obstruction $u^\ast$ produced by the CC+stability machine, can it be ruled out (or classified into an explicit finite library) by a rigidity argument for this specific type $T$?

**Input prerequisites (expected):**
- A critical well-posedness + continuation certificate $K_{\mathrm{WP}_{s_c}}^+$.
- A profile decomposition certificate $K_{\mathrm{ProfDec}_{s_c,G}}^+$.
- A CC+stability machine certificate $K_{\mathrm{KM}_{\mathrm{CC+stab}}}^+$ producing a minimal almost-periodic $u^\ast$ (mod $G$).
- A declared target property $\mathcal P$ (e.g. scattering, global regularity) and a declared minimality functional (energy/mass/etc.).

**YES certificate**

$$K_{\mathrm{Rigidity}_T}^+ := \big(\mathsf{rigid\_statement},\ \mathsf{hypotheses},\ \mathsf{conclusion},\ \mathsf{proof\_ref}\big)$$

where the payload contains:
1. (**Rigidity statement**) A precise proposition of the form:
   > If $u$ is a maximal-lifespan solution of type $T$ which is almost periodic modulo $G$ and minimal among counterexamples to $\mathcal P$, then $u$ is impossible (contradiction), **or** $u$ lies in an explicitly listed finite family $\mathcal L_T$ (soliton, self-similar, traveling wave, etc.).
2. (**Hypotheses**) The exact analytic assumptions required (e.g. Morawetz/virial identity validity, monotonicity formula, coercivity, channel of energy, interaction Morawetz, frequency-localized estimates, etc.).
3. (**Conclusion**) One of:
   - (**Elimination**) no such $u$ exists (hence $\mathcal P$ holds globally), or
   - (**Classification**) every such $u$ belongs to the declared library $\mathcal L_T$.
4. (**Proof reference**) Either (a) a full internal proof in the current manuscript, or (b) an external theorem citation with the exact matching hypotheses.

**NO certificate** (sum type $K_{\mathrm{Rigidity}_T}^- := K_{\mathrm{Rigidity}_T}^{\mathrm{wit}} \sqcup K_{\mathrm{Rigidity}_T}^{\mathrm{inc}}$)

*NO-with-witness:*

$$K_{\mathrm{Rigidity}_T}^{\mathrm{wit}} := (u^*, \mathsf{failure\_mode})$$

where $u^*$ is an almost-periodic minimal counterexample that exists and is not eliminated/classified, and $\mathsf{failure\_mode} \in \{\texttt{NOT\_ELIMINATED}, \texttt{NOT\_IN\_LIBRARY}, \texttt{MONOTONICITY\_FAILS}, \texttt{LS\_CLOSURE\_FAILS}\}$ records which rigidity argument fails.

*NO-inconclusive:*

$$K_{\mathrm{Rigidity}_T}^{\mathrm{inc}} := (\mathsf{obligation}, \mathsf{missing}, \mathsf{failure\_code}, \mathsf{trace})$$

Typical $\mathsf{missing}$: "$K_{\mathrm{Mon}_\phi}^+$ certificate insufficient to validate monotonicity inequality", "$K_{\mathrm{LS}_\sigma}^+$ constants/exponent missing", "no rigidity template (Morawetz/virial/channel-of-energy) matches type $T$".

**Used by:** `mt-auto-profile` Mechanism A (CC+Rig), Step "Hybrid Rigidity".

**Literature:** {cite}`DuyckaertsKenigMerle11`; {cite}`KenigMerle06`.
:::

---

:::{prf:definition} Permit $\mathrm{MorseDecomp}^+$ (Attractor Structure via Morse/Conley or Gradient-like Dynamics)
:label: def-permit-morsedecomp

**Name:** MorseDecomp

**Question:** Does the semiflow $(S_t)_{t\ge0}$ admit a *structural decomposition* of the global attractor sufficient to classify all bounded complete trajectories into equilibria and connecting orbits (or other explicitly described recurrent pieces)?

**Input prerequisites (expected):**
- A global attractor existence certificate $K_{\mathrm{Attr}}^+$ (compact attractor $\mathcal A$ exists).

**YES certificate**

$$K_{\mathrm{MorseDecomp}}^+ := \big(\mathsf{structure\_type},\ \{\mathcal M_i\}_{i=1}^N,\ \mathsf{order},\ \mathsf{chain\_rec},\ \mathsf{classification}\big)$$

where the payload asserts one of the following **declared structure types** (choose one and commit to it in the theorem that uses this permit):

**(A) Gradient-like / Lyapunov structure backend:**
- There exists a continuous strict Lyapunov function $L:X\to\mathbb R$ such that:
  1. $t\mapsto L(S_t x)$ is strictly decreasing unless $x$ is an equilibrium;
  2. the set of equilibria $\mathcal E$ is compact (often finite/mod-$G$);
  3. every bounded complete trajectory has $\alpha$- and $\omega$-limits contained in $\mathcal E$.
- **Classification payload:** every bounded complete trajectory is an equilibrium or a heteroclinic connection between equilibria; no periodic orbits occur.

**(B) Morse–Smale backend (stronger, if you want it):**
- The flow on $\mathcal A$ is Morse–Smale (hyperbolic equilibria/periodic orbits, transverse invariant manifolds, no complicated recurrence).
- **Classification payload:** $\mathcal A$ is a finite union of invariant sets (equilibria and possibly finitely many periodic orbits) plus their stable/unstable manifolds; every trajectory converges to one of the basic pieces.

**(C) Conley–Morse decomposition backend (most general/topological):**
- There exists a finite Morse decomposition $\{\mathcal M_i\}_{i=1}^N$ of $\mathcal A$ with a partial order $\preceq$ such that:
  1. each $\mathcal M_i$ is isolated invariant;
  2. every full trajectory in $\mathcal A$ either lies in some $\mathcal M_i$ or connects from $\mathcal M_i$ to $\mathcal M_j$ with $i\succ j$;
  3. the chain recurrent set is contained in $\bigcup_i \mathcal M_i$.
- **Classification payload:** bounded dynamics reduce to membership in one of the Morse sets plus connecting orbits; recurrent behavior is completely captured by the declared Morse sets.

**NO certificate** (sum type $K_{\mathrm{MorseDecomp}}^- := K_{\mathrm{MorseDecomp}}^{\mathrm{wit}} \sqcup K_{\mathrm{MorseDecomp}}^{\mathrm{inc}}$)

*NO-with-witness:*

$$K_{\mathrm{MorseDecomp}}^{\mathrm{wit}} := (\mathsf{recurrence\_obj}, \mathsf{failure\_type})$$

where $\mathsf{failure\_type} \in \{\texttt{STRANGE\_ATTRACTOR}, \texttt{UNCAPTURED\_CYCLE}, \texttt{INFINITE\_CHAIN\_REC}\}$ identifies recurrence in $\mathcal{A}$ not captured by any declared decomposition type, with a concrete witness object.

*NO-inconclusive:*

$$K_{\mathrm{MorseDecomp}}^{\mathrm{inc}} := (\mathsf{obligation}, \mathsf{missing}, \mathsf{failure\_code}, \mathsf{trace})$$

Typical $\mathsf{missing}$: "Lyapunov function not verified to be strict", "$K_{D_E}^+$ provides weak inequality only", "Conley index computation not supported for this system class".

**Used by:** `mt-auto-profile` Mechanism B (Attr+Morse), anywhere you claim "all bounded trajectories are equilibria/heteroclinic/periodic" or a finite Morse decomposition of $\mathcal A$.

**Literature:** {cite}`Conley78`; {cite}`Hale88`; {cite}`SellYou02`.
:::

---

### Summary Tables

#### Interface Summary Table

| Interface | Object Type | Predicate Logic | Certificate Data |
| :--- | :--- | :--- | :--- |
| **$\mathcal{H}_0$** | State Object | Well-posedness | Existence proof |
| **$D_E$** | Height Morphism | Bound check | $B \in \mathcal{H}$ |
| **$\mathrm{Rec}_N$** | Bad Subobject | Count check | Integer $N$ |
| **$C_\mu$** | Group Action | Concentration | Profile $V$ |
| **$\mathrm{SC}_\lambda$** | Scaling Action | Inequality $\beta - \alpha < \lambda_c$ | Exponents |
| **$\mathrm{SC}_{\partial c}$** | Parameter Object | Stability check | Reference $\theta_0$ |
| **$\mathrm{Cap}_H$** | Capacity Functional | Threshold check | Capacity value |
| **$\mathrm{LS}_\sigma$** | Gradient Operator | Gradient domination | Exponent $\theta$ |
| **$\mathrm{TB}_\pi$** | Sector Set | Invariance | Sector ID |
| **$\mathrm{TB}_O$** | Definability Modality | Definability | Tame structure |
| **$\mathrm{TB}_\rho$** | Measure Object | Finite mixing | Mixing time |
| **$\mathrm{Rep}_K$** | Language Object | Finite description | Program/Code |
| **$\mathrm{GC}_\nabla$** | Metric Tensor | Metric compatibility | Flow type |
| **$\mathrm{Bound}$** | Input/Output Objects | Boundary conditions | Bounds |
| **$\mathrm{Cat}_{\mathrm{Hom}}$** | Hypostructure Category | Hom-emptiness | E1-E13 obstruction |

This table constitutes the **Type Signature** of a Hypostructure.

#### Object-Interface Map

| Interface | **PDE** | **Graph** | **HoTT** | **Neural Net** |
| :--- | :--- | :--- | :--- | :--- |
| $\mathcal{X}$ | $H^s(\mathbb{R}^d)$ | $(V, E)$ | Type $A$ | $\mathbb{R}^n$ |
| $\Phi$ | Energy | Edge count | Level | Loss |
| $G$ | ISO$(d)$ | Aut$(G)$ | Aut$(A)$ | $S_n$ |
| $\mathcal{B}$ | Singular locus | Disconnected | $\bot$ | Saddle points |

#### Domain Instantiation Checklist

| Domain | $\mathcal{E}$ | Reg | D | C | SC | Cap | LS | TB | Rep |
|--------|--------------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Navier-Stokes** | Sh(Diff) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — | ✓ |
| **Graph Coloring** | Set | ✓ | ✓ | ✓ | — | ✓ | — | ✓ | ✓ |
| **HoTT** | $\infty$-Grpd | ✓ | ✓ | ✓ | ✓ | — | — | ✓ | ✓ |
| **Neural Networks** | Smooth | ✓ | ✓ | ✓ | ✓ | — | ✓ | — | ✓ |
| **Crypto Protocols** | Sh(FinSet) | ✓ | ✓ | — | — | — | — | ✓ | ✓ |

---

### Master Node Map

The following table provides the complete mapping from Sieve nodes to interfaces:

| Node | Name | Interface | Predicate $\mathcal{P}$ | YES → | NO → |
|:---:|:---|:---|:---|:---|:---|
| 1 | EnergyCheck | $D_E$ | $\Phi(S_t x) \leq B$ | Node 2 | BarrierSat |
| 2 | ZenoCheck | $\mathrm{Rec}_N$ | $\#\mathcal{B} < \infty$ | Node 3 | BarrierCausal |
| 3 | CompactCheck | $C_\mu$ | Concentration controlled | Node 4 | **BarrierScat** |
| 4 | ScaleCheck | $\mathrm{SC}_\lambda$ | $\beta - \alpha < \lambda_c$ | Node 5 | BarrierTypeII |
| 5 | ParamCheck | $\mathrm{SC}_{\partial c}$ | $\dot{\theta} \approx 0$ | Node 6 | BarrierVac |
| 6 | GeomCheck | $\mathrm{Cap}_H$ | $\text{Cap}(\Sigma) \leq C$ | Node 7 | BarrierCap |
| 7 | StiffnessCheck | $\mathrm{LS}_\sigma$ | $\|\nabla\Phi\| \geq c\Phi^\theta$ | Node 8 | BarrierGap |
| 8 | TopoCheck | $\mathrm{TB}_\pi$ | Sector preserved | Node 9 | BarrierAction |
| 9 | TameCheck | $\mathrm{TB}_O$ | $\mathcal{O}$-definable | Node 10 | BarrierOmin |
| 10 | ErgoCheck | $\mathrm{TB}_\rho$ | Mixing finite | Node 11 | BarrierMix |
| 11 | ComplexCheck | $\mathrm{Rep}_K$ | $K(u) < \infty$ | Node 12 | BarrierEpi |
| 12 | OscillateCheck | $\mathrm{GC}_\nabla$ | Oscillation present | BarrierFreq | Node 13 |
| 13 | BoundaryCheck | $\mathrm{Bound}_\partial$ | Open system? | Node 14 | Node 17 |
| 14 | OverloadCheck | $\mathrm{Bound}_B$ | Input bounded | Node 15 | BarrierBode |
| 15 | StarveCheck | $\mathrm{Bound}_{\Sigma}$ | Supply sufficient | Node 16 | BarrierInput |
| 16 | AlignCheck | $\mathrm{GC}_T$ | Control aligned | Node 17 | BarrierVariety |
| 17 | **The Lock** | $\mathrm{Cat}_{\mathrm{Hom}}$ | $\text{Hom}=\emptyset$ | **VICTORY** | **NO** (typed) |

*Note: Node 17 has binary output with typed NO certificates: Blocked → VICTORY, Breached-with-witness → FATAL, Breached-inconclusive → Reconstruction (see {ref}`The Lock <sec-lock>`).*

#### Restoration Subtree (Mode D Recovery)

When CompactCheck (Node 3) returns NO with a concentration profile, the system enters **Mode D.D** (Definite Deviation). The Restoration Subtree attempts recovery:

| Sub-Node | Name | Check | Success → | Failure → |
|:---:|:---|:---|:---|:---|
| 3a | ProfileID | Profile in library? | Node 3b | Mode D.H (Horizon) |
| 3b | SurgeryAdmit | Surgery admissible? | Node 3c | Mode D.I (Inadmissible) |
| 3c | SurgeryExec | Execute surgery | Node 3d | BarrierSurgery |
| 3d | ReEntry | Re-enter at Node 4 | Node 4 | BarrierReEntry |

---

(sec-kernel-objects)=
### 19.A. The Kernel Objects: Interface Implementations

:::{div} feynman-prose

So far we have listed all the gates - all the questions the Sieve can ask. But where does the data come from to answer these questions?

That is what the kernel objects provide. A Hypostructure has exactly four pieces of core data:

1. **The State Object** $\mathcal{X}$ - Where can the system be? This is the arena.
2. **The Height Object** $\Phi$ - How do we measure progress or cost? This is the potential.
3. **The Dissipation Object** $\mathfrak{D}$ - How fast is energy being lost? This is the friction.
4. **The Symmetry Object** $G$ - What transformations leave the physics unchanged? This is the invariance.

Everything else is derived from these four. The interfaces are like projections - each one reaches into the kernel objects, extracts the data it needs, and computes its verdict.

Think of it like a compiler: you write source code (the four kernel objects), and the compiler (the Sieve) generates machine code (the interface evaluators) automatically. You do not have to write the evaluators yourself.

:::

A Hypostructure $\mathcal{H} = (\mathcal{X}, \Phi, \mathfrak{D}, G)$ consists of four **kernel objects**. Each kernel object implements a subset of the interfaces defined in Sections 19.1-19.15.

This section specifies each kernel object using a standardized template:
1. **Component Table**: Maps each internal component to its mathematical type, the interface it satisfies, and its role in the Sieve
2. **Formal Definition**: Tuple specification with implementation constraints and domain examples

This serves as a "header file" for instantiation - users can read the table and know exactly what data structures to provide for their domain.

---

#### 19.A.1. $\mathcal{X}$ — The State Object
**Implements Interfaces:** **$\mathcal{H}_0$**, **$\mathrm{Cap}_H$**, **$\mathrm{TB}_\pi$**, **$\mathrm{TB}_O$**, **$\mathrm{Rep}_K$**, **$\mathrm{Bound}$**, **$\mathrm{Cat}_{\mathrm{Hom}}$**

| Component | Mathematical Type | Interface | Role / Description |
| :--- | :--- | :--- | :--- |
| **1. $\mathcal{X}$** | Object in $\mathcal{E}$ | **$\mathcal{H}_0$** | The configuration space (all possible states). |
| **2. $\mathcal{T}$** | Monoid Object in $\mathcal{E}$ | **$\mathcal{H}_0$** | The **Time Object** ($\mathbb{R}_{\geq 0}$, $\mathbb{N}$, or ordinal). |
| **3. $S$** | Morphism $\mathcal{T} \times \mathcal{X} \to \mathcal{X}$ | **$\mathcal{H}_0$** | The evolution action (semigroup/flow). Powers **Node 1**. |
| **4. $\mathcal{X}_0$** | Subobject $\hookrightarrow \mathcal{X}$ | **$\mathcal{H}_0$** | Initial data space. Defines well-posed initial conditions. |
| **5. $\mathcal{B}$** | Subobject $\hookrightarrow \mathcal{X}$ | **$\mathrm{Rec}_N$**, **$\mathrm{Cap}_H$** | The **Bad Locus** (singular/blow-up set). Powers **Nodes 2, 6**. |
| **6. $\{S_\tau\}_{\tau \in \mathcal{I}}$** | Coproduct $\mathcal{X} \simeq \coprod_\tau S_\tau$ | **$\mathrm{TB}_\pi$** | **Topological Sectors** indexed by invariant $\tau$. Powers **Node 8**. |
| **7. Cap** | Morphism $\text{Sub}(\mathcal{X}) \to \mathcal{H}$ | **$\mathrm{Cap}_H$** | **Capacity measure** on subobjects. Powers **Node 6**. |
| **8. $\mathcal{O}$** | O-minimal Structure on $\flat(\mathcal{X})$ | **$\mathrm{TB}_O$** | **Definability structure** for tame geometry. Powers **Node 9**. |
| **9. desc** | Morphism $\mathcal{X} \to \mathcal{L}$ | **$\mathrm{Rep}_K$** | **Description map** into language object $\mathcal{L}$. Powers **Node 11**. |
| **10. $(\iota, \pi)$** | $\iota: \mathcal{U} \to \mathcal{X}$, $\pi: \mathcal{X} \to \mathcal{Y}$ | **$\mathrm{Bound}$** | **Open system coupling** (input/output). Powers **Nodes 13-16**. |
| **11. $\mathcal{B}$** | Set of objects in $\mathbf{Hypo}_T$ | **$\mathrm{Cat}_{\mathrm{Hom}}$** | **Bad Pattern Library** $\{B_i\}_{i \in I}$ for morphism obstruction. Powers **Node 17**. |

---

#### 19.A.2. $\Phi$ — The Height Object
**Implements Interfaces:** **$D_E$**, **$\mathrm{SC}_\lambda$**, **$\mathrm{LS}_\sigma$**

| Component | Mathematical Type | Interface | Role / Description |
| :--- | :--- | :--- | :--- |
| **1. $\Phi$** | Morphism $\mathcal{X} \to \mathcal{H}$ | **$D_E$** | The **Height Functional** (Energy/Cost/Complexity). Powers **Node 1**. |
| **2. $\mathcal{H}$** | Ordered Monoid in $\mathcal{E}$ | **$D_E$** | The **Height Codomain** with addition and comparison. |
| **3. $(\leq, +, 0, \infty)$** | Ordered monoid structure | **$D_E$** | **Algebraic structure** enabling bounds and accumulation. |
| **4. $\alpha$** | Element of $\mathbb{R}$ (or $\mathcal{H}$-exponent) | **$\mathrm{SC}_\lambda$** | **Height scaling exponent**: $\Phi(\lambda \cdot x) = \lambda^\alpha \Phi(x)$. Powers **Node 4**. |
| **5. $\nabla\Phi$** | Morphism $\mathcal{X} \to \mathcal{X}^*$ | **$\mathrm{LS}_\sigma$** | The **Gradient/Slope Operator** (generalized derivative). Powers **Node 7**. |
| **6. $\|\cdot\|$** | Norm $\mathcal{X}^* \to \mathcal{H}$ | **$\mathrm{LS}_\sigma$** | **Gradient magnitude** for Łojasiewicz inequality. |
| **7. $\theta$** | Element of $(0, 1]$ | **$\mathrm{LS}_\sigma$** | **Łojasiewicz exponent**: $\|\nabla\Phi\| \geq c|\Phi - \Phi_\infty|^\theta$. |
| **8. $\Phi_\infty$** | Element of $\mathcal{H}$ | **$\mathrm{LS}_\sigma$** | **Limit height** (infimum/equilibrium value). |

---

#### 19.A.3. $\mathfrak{D}$ — The Dissipation Object
**Implements Interfaces:** **$D_E$**, **$\mathrm{Rec}_N$**, **$\mathrm{SC}_\lambda$**, **$\mathrm{GC}_\nabla$**, **$\mathrm{TB}_\rho$**

| Component | Mathematical Type | Interface | Role / Description |
| :--- | :--- | :--- | :--- |
| **1. $\mathfrak{D}$** | Morphism $\mathcal{X} \to \mathcal{H}$ | **$D_E$** | The **Dissipation Rate** (instantaneous height decrease). Powers **Node 1**. |
| **2. $\beta$** | Element of $\mathbb{R}$ | **$\mathrm{SC}_\lambda$** | **Dissipation scaling exponent**: $\mathfrak{D}(\lambda \cdot x) = \lambda^\beta \mathfrak{D}(x)$. Powers **Node 4**. |
| **3. $\mathfrak{D}^{-1}(0)$** | Subobject of $\mathcal{X}$ | **$\mathrm{Rec}_N$** | The **Zero-Dissipation Locus** (equilibria/critical points). Powers **Node 2**. |
| **4. $g$** | Metric structure on $\mathcal{X}$ | **$\mathrm{GC}_\nabla$** | **Riemannian/metric tensor** enabling $\mathfrak{D} = \|\nabla\Phi\|_g^2$. Powers **Node 12**. |
| **5. $\tau_{\text{mix}}$** | Morphism $\mathcal{X} \to \mathcal{T}$ | **$\mathrm{TB}_\rho$** | **Mixing time** to equilibrium. Powers **Node 10**. |

---

#### 19.A.4. $G$ — The Symmetry Object
**Implements Interfaces:** **$\mathcal{H}_0$**, **$C_\mu$**, **$\mathrm{SC}_\lambda$**, **$\mathrm{LS}_\sigma$**, **$\mathrm{SC}_{\partial c}$**

| Component | Mathematical Type | Interface | Role / Description |
| :--- | :--- | :--- | :--- |
| **1. $\mathcal{G}$** | Group Object in $\mathcal{E}$ | **$\mathcal{H}_0$** | The abstract symmetry group (Lie, Discrete, Quantum, Higher). |
| **2. $\rho$** | Action $\mathcal{G} \times \mathcal{X} \to \mathcal{X}$ | **$C_\mu$** | Group action defining orbits $[x] = G \cdot x$. Enables quotient $\mathcal{X} // G$. |
| **3. $\mathcal{S}$** | Monoid homomorphism $(\mathbb{R}^+, \times) \to \mathcal{G}$ | **$\mathrm{SC}_\lambda$** | **Scaling subgroup** for dimensional analysis. Powers **Node 4**. |
| **4. $\mathfrak{P}$** | Partial map $\mathcal{X}^{\mathbb{N}} \rightharpoonup \mathcal{X} // G$ | **$C_\mu$** | **Profile Extractor** (concentration-compactness). Powers **Node 3**. |
| **5. Stab** | Map $\mathcal{X} \to \text{Sub}(\mathcal{G})$ | **$\mathrm{LS}_\sigma$** | **Stabilizer/Isotropy** at each point. Powers **Node 7** (symmetry breaking). |
| **6. $\Theta$** | Parameter space object | **$\mathrm{SC}_{\partial c}$** | **Moduli of symmetry-breaking parameters**. Powers **Node 5**. |

---

### 19.B. The Instantiation Metatheorem

:::{prf:theorem} [FACT-ValidInst] Valid Instantiation
:label: mt-fact-valid-inst

**Statement:** To instantiate a Hypostructure for a system $S$ of type $T$ is to provide:
1. An ambient $(\infty,1)$-topos $\mathcal{E}$ (or a 1-topos/category with sufficient structure)
2. Concrete implementations $(\mathcal{X}, \Phi, \mathfrak{D}, G)$ satisfying the specifications of {ref}`sec-kernel-objects`
3. For each relevant interface $I \in \{\text{Reg}^0, \text{D}^0, \ldots, \text{Lock}^0\}$:
   - The required structure $\mathcal{D}_I$ from the interface definition
   - A computable predicate $\mathcal{P}_I$ evaluating to $\{\text{YES}, \text{NO}, \text{Blocked}\}$ with typed NO certificates ($K^{\mathrm{wit}}$ or $K^{\mathrm{inc}}$)
   - Certificate schemas $\mathcal{K}_I^+$, $\mathcal{K}_I^{\mathrm{wit}}$, and $\mathcal{K}_I^{\mathrm{inc}}$

**Consequence:** Upon valid instantiation, the Sieve Algorithm becomes a well-defined computable function:

$$\text{Sieve}: \text{Instance}(\mathcal{H}) \to \text{Result}$$

where $\text{Result} \in \{\text{GlobalRegularity}, \text{Mode}_{1..15}, \text{FatalError}\}$. NO-inconclusive certificates route to reconstruction rather than terminating as a separate outcome.

**Verification Checklist:**
- [ ] Each kernel object is defined in $\mathcal{E}$
- [ ] Each interface's required structure is provided
- [ ] Predicates are computable (or semi-decidable with timeout)
- [ ] Certificate schemas are well-formed
- [ ] Type $T$ is specified from the catalog ({prf:ref}`def-problem-type`)

**Literature:** Higher topos theory {cite}`Lurie09`; internal logic of toposes {cite}`Johnstone77`; type-theoretic semantics {cite}`HoTTBook`.
:::

:::{prf:theorem} [FACT-MinInst] Minimal Instantiation
:label: mt-fact-min-inst

**Statement:** To instantiate a Hypostructure for system $S$ using the **thin object** formalism ({ref}`sec-thin-kernel-objects`), the user provides only:

1. **The Space** $\mathcal{X}$ and its geometry (metric $d$, measure $\mu$)
2. **The Energy** $\Phi$ and its scaling $\alpha$
3. **The Dissipation** $\mathfrak{D}$ and its scaling $\beta$
4. **The Symmetry Group** $G$ with action $\rho$ and scaling subgroup $\mathcal{S}$

**The Framework (Sieve) automatically derives:**
1. **Profiles:** Via Universal Profile Trichotomy ({prf:ref}`mt-resolve-profile`)
2. **Admissibility:** Via Surgery Admissibility Predicate ({prf:ref}`mt-resolve-admissibility`)
3. **Regularization:** Via Structural Surgery Operator ({prf:ref}`mt-act-surgery`)
4. **Topology:** Via persistent homology on measure $\mu$
5. **Bad Sets:** Via concentration locus of $\mathfrak{D}$

**User vs Framework Responsibility Matrix:**

| Task | User Provides | Framework Derives |
|------|---------------|-------------------|
| Singularity Detection | Energy scaling $\alpha$ | Profile $V$ via scaling group |
| Stability Analysis | Gradient $\nabla$ | Stiffness $\theta$ via Łojasiewicz |
| Surgery Construction | Measure $\mu$ | SurgeryOperator if Cap$(\Sigma)$ small |
| Topology | Space $\mathcal{X}$ | Sectors via $\pi_0$ |
| Bad Set | Dissipation $R$ | $\Sigma = \{x: R(x) \to \infty\}$ |
| Profile Library | Symmetry $G$ | Canonical library via moduli |

**Consequence:** The full instantiation of MT {prf:ref}`mt-fact-valid-inst` is achieved by the **Thin-to-Full Expansion** (MT {prf:ref}`mt-resolve-expansion`), reducing user burden from ~30 components to 10 primitive inputs.

**Literature:** Scaling analysis in PDE {cite}`Tao06`; moduli spaces {cite}`MumfordFogartyKirwan94`; persistent homology {cite}`EdelsbrunnerHarer10`.
:::

:::{prf:remark} Instantiation Examples
:label: rem-instantiation-examples

**Navier-Stokes ($T = T_{\text{parabolic}}$):**
- $\mathcal{E} = \text{Sh}(\text{Diff})$ (sheaves on smooth manifolds)
- $\mathcal{X} = L^2_\sigma(\mathbb{R}^3)$ (divergence-free vector fields)
- $\Phi = \frac{1}{2}\int |u|^2$ (kinetic energy)
- $\mathfrak{D} = \nu \int |\nabla u|^2$ (enstrophy dissipation)
- $G = \text{ISO}(3) \ltimes \mathbb{R}^3$ (rotations, translations, scaling)

**Graph Coloring ($T = T_{\text{algorithmic}}$):**
- $\mathcal{E} = \text{Set}$
- $\mathcal{X} = \text{Map}(V, [k])$ (vertex colorings)
- $\Phi = \#\{\text{monochromatic edges}\}$ (conflict count)
- $\mathfrak{D} = \Delta\Phi$ (per-step improvement)
- $G = \text{Aut}(G) \times S_k$ (graph automorphisms, color permutations)
:::

---

(sec-thin-kernel-objects)=
### 19.C Thin Kernel Objects

:::{div} feynman-prose

Now here is the punchline of this whole section.

You might look at all those kernel objects with their dozens of components and think: "This is going to be a nightmare to instantiate. I have to provide all that structure myself?"

No. You provide the essentials - just 10 pieces of data - and the framework derives everything else automatically.

This is the thin-to-full expansion. You tell us:
- What space your system lives in (and how to measure distance in it)
- What quantity you are trying to minimize (and how it scales)
- How fast energy dissipates (and how that scales)
- What symmetries your system has

From these primitives, the Sieve constructs the full apparatus: the bad sets, the profiles, the surgery operators, the topological sectors. You do not have to be an expert in concentration-compactness or o-minimal geometry. You just describe your system, and the machinery figures out how to analyze it.

This is what makes the framework actually usable in practice.

:::

**Design Principle:** The full Kernel Objects of {ref}`sec-kernel-objects` contain both *structural data* (user-provided) and *algorithmic machinery* (Framework-derived). This section extracts the **minimal user burden** - the "thin" objects that users must specify. Everything else is automatically constructed by the Sieve via the Universal Singularity Modules ({prf:ref}`mt-resolve-profile`, {prf:ref}`mt-resolve-admissibility`, {prf:ref}`mt-act-surgery`).

:::{prf:definition} User vs Framework Responsibility
:label: def-user-framework-split

| Aspect | User Provides | Framework Derives |
|--------|---------------|-------------------|
| **Topology** | Space $\mathcal{X}$, metric $d$ | Sectors via $\pi_0(\mathcal{X})$, dictionary via dimension |
| **Dynamics** | Energy $\Phi$, gradient $\nabla$ | Drift detection, stability via Łojasiewicz |
| **Singularity** | Scaling dimension $\alpha$ | Profile $V$ via scaling group extraction |
| **Dissipation** | Rate $R$, scaling $\beta$ | Bad set as $\{x: R(x) \to \infty\}$ |
| **Surgery** | Measure $\mu$ | Surgery operator if Cap$(\Sigma)$ small |
| **Symmetry** | Group $G$, action $\rho$ | ProfileExtractor, VacuumStabilizer |
:::

---

#### 19.C.1 $\mathcal{X}^{\text{thin}}$ — Thin State Object

**Motto:** *"The Arena"*

:::{prf:definition} Thin State Object
:label: def-thin-state

The **Thin State Object** is a tuple:

$$\mathcal{X}^{\text{thin}} = (\mathcal{X}, d, \mu)$$

| Component | Type | What User Provides |
|-----------|------|-------------------|
| $\mathcal{X}$ | Object in $\mathcal{E}$ | The state space (Polish space, scheme, $\infty$-groupoid) |
| $d$ | $\mathcal{X} \times \mathcal{X} \to [0,\infty]$ | Metric or distance structure |
| $\mu$ | Measure on $\mathcal{X}$ | Reference measure for capacity computation |

**Automatically Derived by Framework:**

| Derived Component | Construction | Used By |
|-------------------|--------------|---------|
| $\text{SectorMap}$ | $\pi_0(\mathcal{X})$ (connected components) | $\mathrm{TB}_\pi$, $C_\mu$ |
| $\text{Dictionary}$ | $\dim(\mathcal{X})$ + type signature | All interfaces |
| $\mathcal{X}_{\text{bad}}$ | $\{x : R(x) \to \infty\}$ | $\mathrm{Cat}_{\mathrm{Hom}}$ |
| $\mathcal{O}$ | O-minimal structure from $d$ | $\mathrm{TB}_O$ |
:::

---

#### 19.C.2 $\Phi^{\text{thin}}$ — Thin Height Object

**Motto:** *"The Potential"*

:::{prf:definition} Thin Height Object
:label: def-thin-height

The **Thin Height Object** is a tuple:

$$\Phi^{\text{thin}} = (F, \nabla, \alpha)$$

| Component | Type | What User Provides |
|-----------|------|-------------------|
| $F$ | $\mathcal{X} \to \mathbb{R} \cup \{\infty\}$ | Energy/height functional |
| $\nabla$ | Gradient or slope operator | Local descent direction |
| $\alpha$ | $\mathbb{Q}_{>0}$ | Scaling dimension |

**Automatically Derived by Framework:**

| Derived Component | Construction | Used By |
|-------------------|--------------|---------|
| $\Phi_\infty$ | $\limsup_{x \to \Sigma} F(x)$ | $\mathrm{LS}_\sigma$ |
| Parameter drift | $\sup_t |\partial_t \theta|$ via $\nabla$ flow | $\mathrm{SC}_{\partial c}$ |
| Critical set | $\text{Crit}(F) = \{x : \nabla F = 0\}$ | $\mathrm{LS}_\sigma$ |
| Stiffness | $\theta$ from $\|F - F_\infty\| \leq C \|\nabla F\|^\theta$ | $\mathrm{LS}_\sigma$ |
:::

---

#### 19.C.3 $\mathfrak{D}^{\text{thin}}$ — Thin Dissipation Object

**Motto:** *"The Cost"*

:::{prf:definition} Thin Dissipation Object
:label: def-thin-dissipation

The **Thin Dissipation Object** is a tuple:

$$\mathfrak{D}^{\text{thin}} = (R, \beta)$$

| Component | Type | What User Provides |
|-----------|------|-------------------|
| $R$ | Rate morphism | Dissipation rate satisfying $\frac{d}{dt}F \leq -R$ |
| $\beta$ | $\mathbb{Q}_{>0}$ | Scaling dimension of dissipation |

**Automatically Derived by Framework:**

| Derived Component | Construction | Used By |
|-------------------|--------------|---------|
| Bad set $\Sigma$ | $\{x : R(x) \to \infty\}$ | $D_E$, $\mathrm{Cap}_H$ |
| Mixing time | $\tau_{\text{mix}} = \inf\{t : \|P_t - \pi\|_{\text{TV}} < 1/e\}$ | $\mathrm{TB}_\rho$ |
| Concentration locus | $\{x : \mu(\epsilon\text{-ball}) \to 0\}$ | $C_\mu$ |
:::

---

#### 19.C.4 $G^{\text{thin}}$ — Thin Symmetry Object

**Motto:** *"The Invariance"*

:::{prf:definition} Thin Symmetry Object
:label: def-thin-symmetry

The **Thin Symmetry Object** is a tuple:

$$G^{\text{thin}} = (\text{Grp}, \rho, \mathcal{S})$$

| Component | Type | What User Provides |
|-----------|------|-------------------|
| $\text{Grp}$ | Group object in $\mathcal{E}$ | The symmetry group |
| $\rho$ | $\text{Grp} \times \mathcal{X} \to \mathcal{X}$ | Group action on state space |
| $\mathcal{S}$ | Subgroup of $\text{Grp}$ | Scaling subgroup |

**Automatically Derived by Framework (via Universal Singularity Modules):**

| Derived Component | Construction | Used By |
|-------------------|--------------|---------|
| ProfileExtractor | {prf:ref}`mt-resolve-profile` (Profile Classification) | Modes 2-3 |
| VacuumStabilizer | Isotropy group of vacuum | $\mathrm{Rep}_K$ |
| SurgeryOperator | {prf:ref}`mt-act-surgery` (Structural Surgery) | Modes 4+barrier |
| Parameter Moduli | $\Theta = \mathcal{X}/G$ | $\mathrm{SC}_{\partial c}$ |
:::

---

#### 19.C.5 Summary: The Four Thin Objects

:::{prf:remark} Minimal Instantiation Burden
:label: rem-minimal-burden

To instantiate a Hypostructure, the user provides exactly **10 primitive components**:

| Object | Components | Physical Meaning |
|--------|------------|------------------|
| $\mathcal{X}^{\text{thin}}$ | $\mathcal{X}, d, \mu$ | "Where does the system live?" |
| $\Phi^{\text{thin}}$ | $F, \nabla, \alpha$ | "What is being minimized?" |
| $\mathfrak{D}^{\text{thin}}$ | $R, \beta$ | "How fast does energy dissipate?" |
| $G^{\text{thin}}$ | Grp, $\rho, \mathcal{S}$ | "What symmetries does the system have?" |

The **full Kernel Objects** of {ref}`sec-kernel-objects` are then constructed automatically:

$$\mathcal{H}^{\text{full}} = \text{Expand}(\mathcal{X}^{\text{thin}}, \Phi^{\text{thin}}, \mathfrak{D}^{\text{thin}}, G^{\text{thin}})$$

This expansion is performed by the **Universal Singularity Modules** ({prf:ref}`mt-resolve-profile`, {prf:ref}`mt-resolve-admissibility`, {prf:ref}`mt-act-surgery`), which implement the `ProfileExtractor` and `SurgeryOperator` interfaces as metatheorems rather than user-provided code.
:::

:::{prf:theorem} [RESOLVE-Expansion] Thin-to-Full Expansion
:label: mt-resolve-expansion

Given thin objects $(\mathcal{X}^{\text{thin}}, \Phi^{\text{thin}}, \mathfrak{D}^{\text{thin}}, G^{\text{thin}})$, the Framework automatically constructs:

1. **Topological Structure:**
   - SectorMap $\leftarrow \pi_0(\mathcal{X})$
   - Dictionary $\leftarrow \dim(\mathcal{X}) + $ type signature

2. **Singularity Detection:**
   - $\mathcal{X}_{\text{bad}} \leftarrow \{x : R(x) \to \infty\}$
   - $\Sigma \leftarrow \text{support of singular measure}$

3. **Profile Classification** ({prf:ref}`mt-resolve-profile`):
   - ProfileExtractor $\leftarrow$ scaling group orbit analysis
   - Canonical library $\leftarrow$ moduli space computation

4. **Surgery Construction** ({prf:ref}`mt-act-surgery`):
   - SurgeryOperator $\leftarrow$ pushout along excision
   - Admissibility $\leftarrow$ capacity bounds from $\mu$

**Guarantee:** If the thin objects satisfy basic consistency (metric is complete, $F$ is lower semicontinuous, $R \geq 0$, $\rho$ is continuous), then the expansion produces valid full Kernel Objects.

**Literature:** Concentration-compactness profile extraction {cite}`Lions84`; moduli space theory {cite}`MumfordFogartyKirwan94`; excision in surgery {cite}`Perelman03`.
:::

---

(sec-soft-backend-compilation)=
### 19.19. Soft-to-Backend Compilation

:::{div} feynman-prose

Here is the final piece of the automation puzzle.

Some of the permits we need are genuinely hard to verify. Well-posedness theory for PDEs. Profile decomposition theorems. Rigidity results for minimal counterexamples. These are deep theorems that took decades of research to establish.

But here is the thing: for "good types" (systems that match known patterns), these theorems have already been proved. Semilinear parabolic equations have well-posedness in energy spaces. Critical dispersive equations have profile decomposition. And so on.

So instead of asking the user to re-prove these theorems for their specific system, we build a compiler. The compiler looks at the soft interfaces the user provided, matches them against a database of known theorem templates, and automatically emits the backend permits.

You give us energy bounds and scaling exponents. We give you well-posedness certificates. You give us symmetry groups and compactness properties. We give you profile decomposition and rigidity theorems.

This is how the framework achieves practical automation: by standing on the shoulders of existing mathematical knowledge.

:::

This section defines the **compilation layer** that automatically derives backend permits from soft interfaces for good types. Users implement only soft interfaces; the framework derives WP, ProfDec, KM, Rigidity, etc.

#### 19.19.1 Architecture

```
USER PROVIDES (Soft Layer)
────────────────────────────────────────
D_E, C_μ, SC_λ, LS_σ, Mon_φ, Rep_K, TB_π, TB_O
────────────────────────────────────────
         ↓ Compilation Metatheorems
────────────────────────────────────────
FRAMEWORK DERIVES (Backend Layer)
────────────────────────────────────────
WP_{s_c}, ProfDec, KM, Rigidity, MorseDecomp, Attr
────────────────────────────────────────
         ↓ Existing Metatheorems
────────────────────────────────────────
FINAL RESULTS
────────────────────────────────────────
Lock^blk, K_prof^+, Global Regularity
```

For **good types** (satisfying the Automation Guarantee), soft interface verification **automatically discharges** backend permits via compilation metatheorems.

---

:::{prf:theorem} [FACT-SoftWP] Soft-to-WP Compilation
:label: mt-fact-soft-wp

**Statement:** For good types $T$ satisfying the Automation Guarantee, critical well-posedness is derived from soft interfaces.

**Soft Hypotheses:**

$$K_{\mathcal{H}_0}^+ \wedge K_{D_E}^+ \wedge K_{\mathrm{Bound}}^+ \wedge K_{\mathrm{SC}_\lambda}^+ \wedge K_{\mathrm{Rep}_K}^+$$

**Produces:**

$$K_{\mathrm{WP}_{s_c}}^+$$

**Mechanism (Template Matching):**
The evaluator `Eval_WP(T)` checks whether $T$ matches a known well-posedness template:

| Template | Soft Signature | WP Theorem Applied |
|----------|----------------|---------------------|
| Semilinear parabolic | $D_E^+$ (coercive) + $\mathrm{Bound}^+$ (Dirichlet/Neumann) | Energy-space LWP |
| Semilinear wave | $\mathrm{SC}_\lambda^+$ (finite speed) + $\mathrm{Bound}^+$ | Strichartz estimates |
| Semilinear Schrödinger | $\mathrm{SC}_\lambda^+$ + $D_E^+$ (conservation) | Dispersive estimates |
| Symmetric hyperbolic | $\mathrm{Rep}_K^+$ (finite description) | Friedrichs method |

**Certificate Emitted:**
$K_{\mathrm{WP}_{s_c}}^+ = (\mathsf{template\_ID}, \mathsf{theorem\_citation}, s_c, \mathsf{continuation\_criterion})$

**NO-Inconclusive Case:** If $T$ matches no template, emit $K_{\mathrm{WP}}^{\mathrm{inc}}$ with $\mathsf{failure\_code} = \texttt{TEMPLATE\_MISS}$. The user may supply a WP proof manually or extend the template database.

**Literature:** {cite}`CazenaveSemilinear03`; {cite}`Tao06`.
:::

---

:::{prf:theorem} Soft-to-Backend Completeness
:label: thm-soft-backend-complete

**Statement:** For good types $T$ satisfying the Automation Guarantee, all backend permits are derived from soft interfaces.

$$\underbrace{K_{D_E}^+ \wedge K_{C_\mu}^+ \wedge K_{\mathrm{SC}_\lambda}^+ \wedge K_{\mathrm{LS}_\sigma}^+ \wedge K_{\mathrm{Rep}_K}^+ \wedge K_{\mathrm{Mon}_\phi}^+}_{\text{Soft Layer (User Provides)}}$$

$$\Downarrow \text{Compilation}$$

$$\underbrace{K_{\mathrm{WP}}^+ \wedge K_{\mathrm{ProfDec}}^+ \wedge K_{\mathrm{KM}}^+ \wedge K_{\mathrm{Rigidity}}^+}_{\text{Backend Layer (Framework Derives)}}$$

**Consequence:** The public signature of `mt-auto-profile` requires only soft interfaces. Backend permits appear only in the **internal compilation proof**, not in the user-facing hypotheses.
:::

---

#### 19.19.2 Evaluators for Derived Permits

The Sieve implements proof-producing evaluators for each derived permit. Every evaluator returns a binary YES/NO verdict with typed certificates:

| Evaluator | Input | YES Output | NO Output | Template Database |
|-----------|-------|------------|-----------|-------------------|
| `Eval_WP(T)` | Type $T$, soft certs | $K_{\mathrm{WP}}^+$ | $K_{\mathrm{WP}}^{\mathrm{wit}}$ or $K_{\mathrm{WP}}^{\mathrm{inc}}$ | Semilinear parabolic, wave, Schrödinger, hyperbolic |
| `Eval_ProfDec(T)` | Type $T$, $C_\mu^+$, $\mathrm{SC}_\lambda^+$ | $K_{\mathrm{ProfDec}}^+$ | $K_{\mathrm{ProfDec}}^{\mathrm{wit}}$ or $K_{\mathrm{ProfDec}}^{\mathrm{inc}}$ | Hilbert space + standard symmetry |
| `Eval_KM(T)` | $\mathrm{WP}^+$, $\mathrm{ProfDec}^+$, $D_E^+$ | $K_{\mathrm{KM}}^+$ | $K_{\mathrm{KM}}^{\mathrm{wit}}$ or $K_{\mathrm{KM}}^{\mathrm{inc}}$ | Composition (no template needed) |
| `Eval_Rigidity(T)` | $\mathrm{Mon}^+$, $\mathrm{KM}^+$, $\mathrm{LS}^+$, Lock | $K_{\mathrm{Rigidity}}^+$ | $K_{\mathrm{Rigidity}}^{\mathrm{wit}}$ or $K_{\mathrm{Rigidity}}^{\mathrm{inc}}$ | Hybrid mechanism |
| `Eval_Attr(T)` | $D_E^+$, $C_\mu^+$, $\mathrm{TB}_\pi^+$ | $K_{\mathrm{Attr}}^+$ | $K_{\mathrm{Attr}}^{\mathrm{wit}}$ or $K_{\mathrm{Attr}}^{\mathrm{inc}}$ | Temam-Raugel theorem |
| `Eval_MorseDecomp(T)` | $\mathrm{Attr}^+$, $D_E^+$, $\mathrm{LS}^+$ | $K_{\mathrm{MorseDecomp}}^+$ | $K_{\mathrm{MorseDecomp}}^{\mathrm{wit}}$ or $K_{\mathrm{MorseDecomp}}^{\mathrm{inc}}$ | Gradient-like / Morse-Smale |

**NO-Inconclusive Policy:** If no template matches, the evaluator returns NO with $K_P^{\mathrm{inc}}$ (not a semantic refutation). The user may then:
1. Supply the backend permit manually (escape hatch)
2. Add a new template to the database
3. Accept that the type is "non-good" and requires custom proof

**Routing on NO Certificates:** The Sieve branches on certificate type:
- **NO with $K^{\mathrm{wit}}$**: Fatal route—a genuine counterexample exists
- **NO with $K^{\mathrm{inc}}$**: Reconstruction route—try adding interfaces, refining library, or extending templates

---

:::{prf:theorem} [FACT-GermDensity] Germ Set Density
:label: mt-fact-germ-density

**Rigor Class:** F (Framework-Original)

**Sieve Target:** Node 17 (Lock) — ensures the finite library suffices

**Statement:** Let $\mathcal{G}_T$ be the small germ set for problem type $T$, and let $\mathcal{B} = \{B_i\}_{i \in I}$ be the finite Bad Pattern Library from Interface $\mathrm{Cat}_{\mathrm{Hom}}$. Then $\mathcal{B}$ is **dense** in $\mathbb{H}_{\mathrm{bad}}^{(T)}$ in the following sense:

For any germ $[P, \pi] \in \mathcal{G}_T$, there exists $B_i \in \mathcal{B}$ and a factorization:

$$\mathbb{H}_{[P,\pi]} \to B_i \to \mathbb{H}_{\mathrm{bad}}^{(T)}$$

**Consequence:** If $\mathrm{Hom}(B_i, \mathbb{H}(Z)) = \emptyset$ for all $B_i \in \mathcal{B}$, then $\mathrm{Hom}(\mathbb{H}_{\mathrm{bad}}^{(T)}, \mathbb{H}(Z)) = \emptyset$.

**Certificate Produced:** $K_{\mathrm{density}}^+ := (\mathcal{B}, \mathcal{G}_T, \text{factorization witnesses}, \text{type-specific completeness})$

**Literature:** {cite}`Quillen67` (Small Object Argument); {cite}`Hovey99` §2.1 (cellular structures); {cite}`Lurie09` §A.1.5 (presentability and generation)
:::
