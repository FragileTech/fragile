---
title: "Instantiation Protocol"
---

# Part XIII: Instantiation

(sec-certificate-generator-library)=
## Certificate Generator Library

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

:::{prf:theorem} [FACT-Instantiation] Instantiation Metatheorem
:label: mt-fact-instantiation
:class: metatheorem

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
:label: proof-mt-fact-instantiation

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

*Step 4 (Contract Satisfaction).* The composed implementation satisfies all contracts from the {ref}`Gate Catalog <sec-node-specs>`:
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

---

(sec-metatheorem-unlock-table)=
## Metatheorem Unlock Table

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

The following table provides a complete cross-reference between diagram node names and their formal definitions:

| **Node** | **Diagram Label** | **Predicate Def.** | **Barrier/Surgery** |
|---|---|---|---|
| 1 | EnergyCheck | {ref}`def-node-energy` | BarrierSat ({ref}`def-barrier-sat`) |
| 2 | ZenoCheck | {ref}`def-node-zeno` | BarrierCausal ({ref}`def-barrier-causal`) |
| 3 | CompactCheck | {ref}`def-node-compact` | BarrierScat ({ref}`def-barrier-scat`) |
| 4 | ScaleCheck | {ref}`def-node-scale` | BarrierTypeII ({ref}`def-barrier-type2`) |
| 5 | ParamCheck | {ref}`def-node-param` | BarrierVac ({ref}`def-barrier-vac`) |
| 6 | GeomCheck | {ref}`def-node-geom` | BarrierCap ({ref}`def-barrier-cap`) |
| 7 | StiffnessCheck | {ref}`def-node-stiffness` | BarrierGap ({ref}`def-barrier-gap`) |
| 7a | BifurcateCheck | {ref}`def-node-bifurcate` | Mode S.D |
| 7b | SymCheck | {ref}`def-node-sym` | --- |
| 7c | CheckSC | {ref}`def-node-checksc` | ActionSSB ({ref}`def-action-ssb`) |
| 7d | CheckTB | {ref}`def-node-checktb` | ActionTunnel ({ref}`def-action-tunnel`) |
| 8 | TopoCheck | {ref}`def-node-topo` | BarrierAction ({ref}`def-barrier-action`) |
| 9 | TameCheck | {ref}`def-node-tame` | BarrierOmin ({ref}`def-barrier-omin`) |
| 10 | ErgoCheck | {ref}`def-node-ergo` | BarrierMix ({ref}`def-barrier-mix`) |
| 11 | ComplexCheck | {ref}`def-node-complex` | BarrierEpi ({ref}`def-barrier-epi`) |
| 12 | OscillateCheck | {ref}`def-node-oscillate` | BarrierFreq ({ref}`def-barrier-freq`) |
| 13 | BoundaryCheck | {ref}`def-node-boundary` | --- |
| 14 | OverloadCheck | {ref}`def-node-overload` | BarrierBode ({ref}`def-barrier-bode`) |
| 15 | StarveCheck | {ref}`def-node-starve` | BarrierInput ({ref}`def-barrier-input`) |
| 16 | AlignCheck | {ref}`def-node-align` | BarrierVariety ({ref}`def-barrier-variety`) |
| 17 | BarrierExclusion | {ref}`def-node-lock` | Lock ({ref}`sec-lock`) |
