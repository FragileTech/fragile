---
title: "Instantiation Protocol"
---

# Instantiation

:::{div} feynman-prose
Now we come to the part where theory meets practice. You have all these beautiful metatheorems from Part XII saying "there exist factories that build verifiers" and "there exist barriers with correct properties." Wonderful. But the working mathematician or engineer asks: "Fine, but how do I actually use this thing?"

That is what instantiation is about. We are going to show you the complete recipe for taking a specific PDE, or a Markov chain, or an optimization algorithm, and plugging it into the Sieve framework so that all those theoretical guarantees actually apply to your problem.

The key insight is this: the user supplies the physics (the energy functional, the symmetries, the natural scales), and the framework supplies the logic (the verification machinery, the barrier implementations, the surgery recipes). The separation is clean. You do not need to reinvent the regularity theory; you just need to tell the system what your problem looks like.
:::

(sec-certificate-generator-library)=
## Certificate Generator Library

:::{div} feynman-prose
Here is where the rubber meets the road. Every gate in the Sieve produces a certificate when it passes or fails. But where do these certificates come from? Not from thin air. They come from actual mathematical theorems that have been proven over decades of work by analysts, geometers, and probabilists.

This table is the Rosetta Stone. On the left: the Sieve nodes with their abstract names. On the right: the concrete mathematical tools that power them. When EnergyCheck says "YES," it is because someone (maybe Gronwall, maybe the user) proved an energy inequality. When BarrierSat blocks, it is Foster-Lyapunov theory doing the work under the hood.

The beauty is that you do not have to know all this literature to use the framework. But if you want to understand why the certificates are trustworthy, here is exactly where to look.
:::

The **Certificate Generator Library** maps standard literature lemmas to permits:

| **Node/Barrier** | **Literature Tool** | **Certificate** |
|---|---|---|
| EnergyCheck | Energy inequality, Gronwall | $K_{D_E}^+$ |
| BarrierSat | Foster-Lyapunov, drift control | $K_{\text{sat}}^{\mathrm{blk}}$ |
| ZenoCheck | Dwell-time lemma, event bounds | $K_{\mathrm{Rec}_N}^+$ |
| CompactCheck (YES) | Concentration-compactness | $K_{C_\mu}^+$, $K_{\text{prof}}$ |
| CompactCheck (NO) | Dispersion estimates | $K_{C_\mu}^-$, leads to D.D |
| ScaleCheck | Scaling analysis, critical exponents | $K_{\mathrm{SC}_\lambda}^+$ |
| BarrierTypeII | Monotonicity formulas, Kenig-Merle | $K_{\text{II}}^{\mathrm{blk}}$ |
| ParamCheck | Modulation theory, orbital stability | $K_{\mathrm{SC}_{\partial c}}^+$ |
| GeomCheck | Hausdorff dimension, capacity | $K_{\mathrm{Cap}_H}^+$ |
| BarrierCap | Epsilon-regularity, partial regularity | $K_{\text{cap}}^{\mathrm{blk}}$ |
| StiffnessCheck | Lojasiewicz-Simon, spectral gap | $K_{\mathrm{LS}_\sigma}^+$ |
| BarrierGap | Poincare inequality, mass gap | $K_{\text{gap}}^{\mathrm{blk}}$ |
| TopoCheck | Sector classification, homotopy | $K_{\mathrm{TB}_\pi}^+$ |
| TameCheck | O-minimal theory, definability | $K_{\mathrm{TB}_O}^+$ |
| ErgoCheck | Mixing times, ergodic theory | $K_{\mathrm{TB}_\rho}^+$ |
| ComplexCheck | Kolmogorov complexity, MDL | $K_{\mathrm{Rep}_K}^+$ |
| OscillateCheck | Monotonicity, De Giorgi-Nash-Moser | $K_{\mathrm{GC}_\nabla}^-$ |
| Lock | Cohomology, invariant theory | $K_{\mathrm{Cat}_{\mathrm{Hom}}}^+$ |

---

(sec-minimal-instantiation-checklist)=
## Minimal Instantiation Checklist

:::{div} feynman-prose
Here is the question that matters: what is the absolute minimum you need to provide to get the Sieve running on your problem?

The answer is surprisingly short. Eight pieces of information. That is it. You tell the system what your state space is, what energy means, what dissipation looks like, and a few optional extras depending on your problem type. Everything else—the gate evaluators, the barriers, the surgery protocols, the Lock tactics—gets compiled automatically from the factory metatheorems.

This is the power of abstraction done right. We have factored out the common structure of regularity theory across dozens of problem classes, and what remains for the user is just the irreducible physics of their specific problem.

Let me walk you through what happens when you provide these eight items. The framework takes your definitions and runs them through five factories in sequence. Each factory produces implementations that depend only on your input and on previous factory outputs—never on future outputs. This acyclic structure is what guarantees the whole thing terminates and makes sense.
:::

:::{prf:theorem} [FACT-Instantiation] Instantiation Metatheorem
:label: mt-fact-instantiation

For any system of type $T$ with user-supplied functionals, there exists a canonical sieve implementation satisfying all contracts:

**User provides (definitions only)**:
1. State space $X$ and symmetry group $G$
2. Height functional $\Phi: X \to [0, \infty]$
3. Dissipation functional $\mathfrak{D}: X \to [0, \infty]$
4. Recovery functional $\mathcal{R}: X \to [0, \infty)$ (if $\mathrm{Rec}_N$)
5. Capacity gauge $\mathrm{Cap}$ (if $\mathrm{Cap}_H$)
6. Sector label $\tau: X \to \mathcal{T}$ (if $\mathrm{TB}_\pi$)
7. Dictionary map $D: X \to \mathcal{T}$ (if $\mathrm{Rep}_K$, optional)
8. Type selection $T \in \{T_{\text{parabolic}}, T_{\text{dispersive}}, T_{\text{metricGF}}, T_{\text{Markov}}, T_{\text{algorithmic}}\}$

**Framework provides (compiled from factories)**:
1. Gate evaluators (TM-1)
2. Barrier implementations (TM-2)
3. Surgery schemas (TM-3)
4. Equivalence + Transport (TM-4)
5. Lock backend (TM-5)

**Output**: Sound sieve run yielding either:
- Regularity certificate (VICTORY)
- Mode certificate with admissible repair (surgery path)
- NO-inconclusive certificate ($K^{\mathrm{inc}}$) (explicit obstruction to classification/repair)

**Literature:** Type-theoretic instantiation {cite}`HoTTBook`; certified regularity proofs {cite}`Leroy09`; singularity resolution via surgery {cite}`Perelman03`; well-founded induction {cite}`Aczel77`.

:::

:::{prf:proof}

*Step 1 (Factory Composition).* Given type $T$ and user-supplied functionals $(\Phi, \mathfrak{D}, G, \ldots)$, apply factories TM-1 through TM-5 in sequence:
- TM-1 instantiates gate evaluators $\{V_i^T\}_{i=1}^{17}$ from $(\Phi, \mathfrak{D})$
- TM-2 instantiates barrier implementations $\{\mathcal{B}_j^T\}$ from TM-1 outputs
- TM-3 instantiates surgery schemas from profile library (if type admits surgery)
- TM-4 instantiates equivalence/transport from $G$ and scaling data
- TM-5 instantiates Lock backend from $\mathrm{Rep}_K$ (if available)

*Step 2 (Non-Circularity).* Each factory's outputs depend only on:
- User-supplied data: $(\Phi, \mathfrak{D}, G, \mathrm{Cap}, \tau, D)$
- Type templates: Pre-defined for each $T \in \{T_{\text{parabolic}}, \ldots\}$
- Prior factory outputs: TM-$k$ may use TM-$(k-1)$ outputs but never TM-$(k+1)$

The dependency graph is acyclic by construction: factories are numbered in topological order. No gate evaluator depends on its own output, and no barrier depends on the barrier it implements.

*Step 3 (Soundness Inheritance).* Each factory produces sound implementations by {prf:ref}`mt-fact-gate` through {prf:ref}`mt-fact-lock`. Composition preserves soundness by transitivity:
- If $V_i^T(x, \Gamma) = (\text{YES}, K_i^+)$, then predicate $P_i^T(x)$ holds (TM-1 soundness)
- If barrier $\mathcal{B}_j^T$ blocks, then the obstruction is genuine (TM-2 soundness)
- Combined: if the Sieve reaches VICTORY, all 17 gates passed with valid certificates

*Step 4 (Contract Satisfaction).* The composed implementation satisfies all contracts from the {ref}`Gate Catalog <sec-gate-node-specs>`:
- Each gate contract specifies: Pre-certificates required, Post-certificates produced, Routing rules
- Factory composition ensures: $\text{Post}(V_i^T) \subseteq \text{Pre}(V_{i+1}^T)$ for the graph edges
- Transport lemmas (TM-4) ensure certificate validity is preserved across equivalence moves

*Step 5 (Termination).* The Sieve execution terminates in finite time:
- **Gate termination:** Each $V_i^T$ terminates by TM-1 (finite computation on bounded data)
- **Surgery termination:** Each surgery decreases a well-founded measure (energy + surgery count) by {prf:ref}`mt-resolve-admissibility`
- **Global termination:** The Sieve DAG has finitely many nodes (17 + barriers + surgery). Surgery count is bounded by $N_{\max} = \lfloor \Phi(x_0)/\delta_{\text{surgery}} \rfloor$. Total steps $\leq 17 \cdot N_{\max} \cdot (\text{max barrier iterations})$.

*Step 6 (Output Trichotomy).* The Sieve execution terminates with exactly one of:
- **VICTORY:** All gates pass, Lock blocked → emit $K_{\text{Lock}}^{\mathrm{blk}}$ (Global Regularity)
- **Surgery path:** Barrier breach + admissibility → surgery iteration, returns to Step 1 with surgered state
- **$K^{\mathrm{inc}}$:** Tactic exhaustion at some node → emit $K_P^{\mathrm{inc}}$ with $\mathsf{missing}$ set, route to {prf:ref}`mt-lock-reconstruction`

The trichotomy is exhaustive: at each node, the verifier returns YES, NO-with-witness (barrier), or NO-inconclusive. These three outcomes cover all possibilities by the decidability of each gate predicate.

$\square$
:::

:::{div} feynman-prose
Let me make sure you understand what just happened in that proof. The six steps tell a complete story:

**Factory composition** (Step 1): Your definitions flow through the five factories like an assembly line. Each factory adds more structure. By the end, you have a complete, working Sieve implementation.

**Non-circularity** (Step 2): This is crucial. Nothing depends on itself. The factories are numbered in dependency order. This is not an accident; it is the whole architecture.

**Soundness inheritance** (Step 3): If each factory does its job correctly, the composition does its job correctly. This is transitivity of correctness, and it is why we can trust the output.

**Contract satisfaction** (Step 4): Every gate knows what it needs (preconditions) and what it produces (postconditions). The factory composition ensures these match up at every edge in the Sieve graph.

**Termination** (Step 5): The Sieve always finishes. This is not obvious! With surgery loops and barrier re-evaluations, you might worry about infinite cycling. But the well-founded progress measure kills that worry dead.

**Output trichotomy** (Step 6): At the end, you get exactly one of three things: global regularity (VICTORY), a surgery path to try again, or an honest admission of incompleteness. No fourth option. No silent failures.
:::

---

(sec-metatheorem-unlock-table)=
## Metatheorem Unlock Table

:::{div} feynman-prose
Now here is something beautiful. The metatheorems from earlier in this document are not just abstract guarantees—they are unlockable achievements. When the Sieve runs and produces certain certificates, those certificates unlock the ability to apply specific metatheorems.

Think of it like a video game. You cannot use the "Type II Exclusion" power until you have collected both the subcriticality badge (from ScaleCheck) and the energy badge (from EnergyCheck). The table below tells you exactly which badges unlock which powers.

This is the semantic content of the certificates. They are not just "passed" or "failed" labels; they are mathematical claims that enable downstream reasoning.
:::

The following table specifies which metatheorems are unlocked by which certificates:

| **Metatheorem** | **Required Certificates** | **Producing Nodes** |
|---|---|---|
| Structural Resolution | $K_{C_\mu}^+$ (profile) | CompactCheck YES |
| Type II Exclusion | $K_{\mathrm{SC}_\lambda}^+$ (subcritical) + $K_{D_E}^+$ (energy) | ScaleCheck YES + EnergyCheck YES |
| Capacity Barrier | $K_{\mathrm{Cap}_H}^+$ or $K_{\text{cap}}^{\mathrm{blk}}$ | GeomCheck YES/Blocked |
| Topological Suppression | $K_{\mathrm{TB}_\pi}^+$ + $K_{\mathrm{Cat}_{\mathrm{Hom}}}^+$ (Lock) | TopoCheck YES + Lock Blocked |
| Canonical Lyapunov | $K_{\mathrm{LS}_\sigma}^+$ (stiffness) + $K_{\mathrm{GC}_\nabla}^-$ (no oscillation) | StiffnessCheck YES + OscillateCheck NO |
| Functional Reconstruction | $K_{\mathrm{LS}_\sigma}^+$ + $K_{\mathrm{Rep}_K}^+$ (Rep) + $K_{\mathrm{GC}_\nabla}^-$ | LS + Rep + GC |
| Profile Classification | $K_{C_\mu}^+$ | CompactCheck YES |
| Surgery Admissibility | $K_{\text{lib}}$ or $K_{\text{strat}}$ | Profile Trichotomy Case 1/2 |
| Global Regularity | $K_{\mathrm{Cat}_{\mathrm{Hom}}}^+$ (Lock Blocked) | BarrierExclusion Blocked |

---

(sec-diagram-specification-cross-reference)=
## Diagram ↔ Specification Cross-Reference

:::{div} feynman-prose
Finally, we come to the bookkeeping. If you have been reading the Sieve diagrams elsewhere in this book and wondering "Where is the formal definition of this node?"—here is your answer.

This cross-reference table is not exciting, but it is essential. When debugging a Sieve run, or when trying to understand why a particular gate behaved the way it did, you need to find the formal specification quickly. Each row gives you the node number, its name in diagrams, and the exact label of its formal definition.

Notice that some nodes have associated barriers and some do not. The nodes without barriers are purely diagnostic—they route information but do not block execution. The nodes with barriers are the workhorses that actually enforce constraints on the system's behavior.
:::

The following table provides a complete cross-reference between diagram node names and their formal definitions:

| **Node** | **Diagram Label** | **Predicate Def.** | **Barrier/Surgery** |
|---|---|---|---|
| 1 | EnergyCheck | {prf:ref}`def-node-energy` | BarrierSat ({prf:ref}`def-barrier-sat`) |
| 2 | ZenoCheck | {prf:ref}`def-node-zeno` | BarrierCausal ({prf:ref}`def-barrier-causal`) |
| 3 | CompactCheck | {prf:ref}`def-node-compact` | BarrierScat ({prf:ref}`def-barrier-scat`) |
| 4 | ScaleCheck | {prf:ref}`def-node-scale` | BarrierTypeII ({prf:ref}`def-barrier-type2`) |
| 5 | ParamCheck | {prf:ref}`def-node-param` | BarrierVac ({prf:ref}`def-barrier-vac`) |
| 6 | GeomCheck | {prf:ref}`def-node-geom` | BarrierCap ({prf:ref}`def-barrier-cap`) |
| 7 | StiffnessCheck | {prf:ref}`def-node-stiffness` | BarrierGap ({prf:ref}`def-barrier-gap`) |
| 7a | BifurcateCheck | {prf:ref}`def-node-bifurcate` | Mode S.D |
| 7b | SymCheck | {prf:ref}`def-node-sym` | --- |
| 7c | CheckSSB | {prf:ref}`def-node-checkssb` | ActionSSB ({prf:ref}`def-action-ssb`) |
| 7d | CheckTB | {prf:ref}`def-node-checktb` | ActionTunnel ({prf:ref}`def-action-tunnel`) |
| 8 | TopoCheck | {prf:ref}`def-node-topo` | BarrierAction ({prf:ref}`def-barrier-action`) |
| 9 | TameCheck | {prf:ref}`def-node-tame` | BarrierOmin ({prf:ref}`def-barrier-omin`) |
| 10 | ErgoCheck | {prf:ref}`def-node-ergo` | BarrierMix ({prf:ref}`def-barrier-mix`) |
| 11 | ComplexCheck | {prf:ref}`def-node-complex` | BarrierEpi ({prf:ref}`def-barrier-epi`) |
| 12 | OscillateCheck | {prf:ref}`def-node-oscillate` | BarrierFreq ({prf:ref}`def-barrier-freq`) |
| 13 | BoundaryCheck | {prf:ref}`def-node-boundary` | --- |
| 14 | OverloadCheck | {prf:ref}`def-node-overload` | BarrierBode ({prf:ref}`def-barrier-bode`) |
| 15 | StarveCheck | {prf:ref}`def-node-starve` | BarrierInput ({prf:ref}`def-barrier-input`) |
| 16 | AlignCheck | {prf:ref}`def-node-align` | BarrierVariety ({prf:ref}`def-barrier-variety`) |
| 17 | BarrierExclusion | {prf:ref}`def-node-lock` | Lock ({ref}`sec-lock`) |
