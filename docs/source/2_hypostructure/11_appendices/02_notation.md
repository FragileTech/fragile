---
title: "Notation Index"
---

(sec-notation-index)=
# Notation Index

The following notation is used consistently throughout this document. Symbols are organized by their role in the Hypostructure formalism.

(sec-notation-core-objects)=
## Core Objects

| Symbol | Name | Definition | Reference |
|--------|------|------------|-----------|
| $\mathcal{X}$ | State Space | Configuration $\infty$-stack representing system states | {prf:ref}`def-categorical-hypostructure` |
| $\mathcal{B}$ | Boundary Space | Environmental interface / boundary data | {ref}`Boundary Constraints <sec-boundary-constraints>` |
| $\Phi$ | Height / Energy | Cohomological height functional | {prf:ref}`def-categorical-hypostructure` |
| $\mathfrak{D}$ | Dissipation | Rate of energy loss / entropy production | {prf:ref}`def-categorical-hypostructure` |
| $G$ | Symmetry Group | Invariance group acting on $\mathcal{X}$ | {prf:ref}`def-categorical-hypostructure` |
| $\mathbb{H}$ | Hypostructure | Full 5-tuple $(\mathcal{X}, \nabla, \Phi_\bullet, \tau, \partial_\bullet)$ | {prf:ref}`def-categorical-hypostructure` |
| $\mathcal{T}$ | Thin Object | Minimal 5-tuple of physical data | {ref}`Thin Kernel <sec-thin-kernel>` |

(sec-notation-energy-scaling)=
## Energy and Scaling

| Symbol | Name | Context |
|--------|------|---------|
| $E$ | Specific Energy | Instance of height $\Phi$; $E[\Phi] = \sup_t \Phi(u(t))$ |
| $\alpha$ | Energy Scaling | Exponent: $\Phi(\mathcal{S}_\lambda x) = \lambda^\alpha \Phi(x)$ |
| $\beta$ | Dissipation Scaling | Exponent: $\mathfrak{D}(\mathcal{S}_\lambda x) = \lambda^\beta \mathfrak{D}(x)$ |
| $E_{\text{sat}}$ | Saturation Ceiling | Upper bound on drift (BarrierSat) |
| $\mathcal{S}_\lambda$ | Scaling Operator | One-parameter family of dilations |

(sec-notation-boundary-reinjection)=
## Boundary and Reinjection

| Symbol | Name | Definition |
|--------|------|------------|
| $\partial_\bullet$ | Boundary Morphism | Restriction functor $\iota^*: \mathbf{Sh}_\infty(\mathcal{X}) \to \mathbf{Sh}_\infty(\partial\mathcal{X})$ |
| $\text{Tr}$ | Trace Morphism | $\text{Tr}: \mathcal{X} \to \mathcal{B}$ (restriction to boundary) |
| $\mathcal{J}$ | Flux Morphism | $\mathcal{J}: \mathcal{B} \to \underline{\mathbb{R}}$ (energy flow across boundary) |
| $\mathcal{R}$ | Reinjection Kernel | $\mathcal{R}: \mathcal{B} \to \mathcal{P}(\mathcal{X})$ (Markov kernel with Feller property) |

(sec-notation-certificates)=
## Certificate Notation

| Symbol | Meaning |
|--------|---------|
| $K^+$ | Positive certificate (predicate holds) |
| $K^-$ | Negative certificate (sum type: $K^{\mathrm{wit}} \sqcup K^{\mathrm{inc}}$) |
| $K^{\mathrm{wit}}$ | NO-with-witness certificate (actual refutation / counterexample found) |
| $K^{\mathrm{inc}}$ | NO-inconclusive certificate (method insufficient, not a semantic refutation) |
| $K^{\text{blk}}$ | Blocked certificate (barrier holds, obstruction present) |
| $K^{\text{br}}$ | Breached certificate (barrier fails: $K^{\mathrm{br\text{-}wit}}$ or $K^{\mathrm{br\text{-}inc}}$) |
| $K^{\text{re}}$ | Re-entry certificate (surgery completed successfully) |
| $\Gamma$ | Certificate accumulator (full chain of certificates) |

(sec-notation-categorical)=
## Categorical Notation

| Symbol | Name | Definition |
|--------|------|------------|
| $\mathcal{E}$ | Ambient Topos | Cohesive $(\infty,1)$-topos |
| $\mathbf{Hypo}_T$ | Hypostructure Category | Category of type-$T$ hypostructures |
| $\mathbf{Thin}_T$ | Thin Category | Category of thin kernel objects |
| $\mathbb{H}_{\text{bad}}$ | Bad Pattern | Universal singularity object |
| $\text{Hom}(\cdot, \cdot)$ | Hom Functor | Morphism space (Node 17 Lock) |
| $F_{\text{Sieve}}$ | Sieve Functor | Left adjoint $F_{\text{Sieve}} \dashv U$ |

(sec-notation-interface-identifiers)=
## Interface Identifiers

| ID | Name | Node |
|----|------|------|
| $D_E$ | Energy Interface | Node 1 |
| $\mathrm{Rec}_N$ | Recovery Interface | Node 2 |
| $C_\mu$ | Compactness Interface | Node 3 |
| $\mathrm{SC}_\lambda$ | Scaling Interface | Node 4 |
| $\mathrm{Cap}_H$ | Capacity Interface | Node 6 |
| $\mathrm{LS}_\sigma$ | Stiffness Interface | Node 7 |
| $\mathrm{TB}_\pi$ | Topology Interface | Node 8 |
| $\mathrm{TB}_\rho$ | Mixing Interface | Node 10 |
| $\mathrm{Rep}_K$ | Complexity Interface | Node 11 |
| $\mathrm{GC}_\nabla$ | Gradient Interface | Node 12 |
| $\mathrm{Cat}_{\mathrm{Hom}}$ | Categorical Interface | Node 17 |

(sec-notation-rigor-classification)=
## Rigor Classification

| Symbol | Name | Definition |
|--------|------|------------|
| Rigor Class L | Literature-Anchored | Proof rigor offloaded to external literature via Bridge Verification (Def. {prf:ref}`def-rigor-classification`) |
| Rigor Class F | Framework-Original | First-principles categorical proof using cohesive topos theory (Def. {prf:ref}`def-rigor-classification`) |
| Rigor Class B | Bridge | Cross-foundation translation between categorical framework and classical foundations (Def. {prf:ref}`def-rigor-classification`) |
| $\mathcal{H}_{\text{tr}}$ | Hypothesis Translation | Bridge Verification Step 1: $\Gamma_{\text{Sieve}} \vdash \mathcal{H}_{\mathcal{L}}$ |
| $\iota$ | Domain Embedding | Bridge Verification Step 2: $\iota: \mathbf{Hypo}_T \to \mathbf{Dom}_{\mathcal{L}}$ |
| $\mathcal{C}_{\text{imp}}$ | Conclusion Import | Bridge Verification Step 3: $\mathcal{C}_{\mathcal{L}}(\iota(\mathbb{H})) \Rightarrow K^+$ |
| $\Pi \dashv \flat \dashv \sharp$ | Cohesion Adjunction | Shape/Flat/Sharp modality adjunction in cohesive $(\infty,1)$-topos |
| $\mathcal{F} \dashv U$ | Expansion Adjunction | Thin-to-Hypo expansion as left adjoint (Thm. {prf:ref}`thm-expansion-adjunction`) |

(sec-notation-progress-measures)=
## Progress Measures (Type A/B)

| Symbol | Name | Definition |
|--------|------|------------|
| Type A | Bounded Count | Surgery count bounded by $N(T, \Phi(x_0))$; finite budget termination |
| Type B | Well-Founded | Complexity measure $\mathcal{C}: X \to \mathbb{N}$ strictly decreases per surgery |

**Note:** Type A/B classification refers to *progress measures* for termination analysis (Definition {prf:ref}`def-progress-measures`), distinct from Rigor Class L/F/B which refers to *proof provenance*.

(sec-notation-zfc-translation)=
## ZFC Translation

| Symbol | Name | Definition | Reference |
|--------|------|------------|-----------|
| $\mathcal{U}$ | Grothendieck Universe | Transitive set closed under ZFC operations; anchor for size consistency | {ref}`Universe Anchoring <sec-zfc-universe-anchoring>` |
| $V_\mathcal{U}$ | Universe Hierarchy | Cumulative hierarchy $\bigcup_{\alpha < \kappa} V_\alpha$ up to $\mathcal{U}$ | {ref}`Universe Anchoring <sec-zfc-universe-anchoring>` |
| $\mathbf{K}$ | Certificate Chain | $(K_1, \ldots, K_{17})$; full Sieve output (avoids conflict with $\Gamma$) | {ref}`Universe Anchoring <sec-zfc-universe-anchoring>` |
| $\tau_0$ | 0-Truncation Functor | Left adjoint $\tau_0 \dashv \Delta$; extracts $\pi_0$ (connected components) | {ref}`Truncation Functor <sec-zfc-truncation>` |
| $\Delta$ | Discrete Embedding | $\Delta: \mathbf{Set} \hookrightarrow \mathcal{E}$; embeds sets as discrete objects | {ref}`Truncation Functor <sec-zfc-truncation>` |
| $\mathcal{B}_{\text{ZFC}}$ | Bridge Certificate | Payload $(\mathcal{U}, \varphi, \text{axioms\_used}, \text{AC\_status}, \text{trace})$ | {ref}`Cross-Foundation Audit <sec-zfc-cross-foundation-audit>` |
| $\Omega$ | Subobject Classifier | Truth-value object in $\mathcal{E}$ | {ref}`Axiomatic Dictionary <sec-zfc-axiomatic-dictionary>` |
| $\mathbb{N}_\mathcal{E}$ | Natural Number Object | NNO in topos $\mathcal{E}$; realizes Axiom of Infinity | {ref}`Axiomatic Dictionary <sec-zfc-axiomatic-dictionary>` |
| $\mathcal{H}$ | Heyting Algebra | Internal logic of $\mathcal{E}$ (intuitionistic) | {ref}`Classicality Operator <sec-zfc-classicality>` |
| $\mathcal{B}$ | Boolean Subalgebra | Decidable propositions; logic of $\flat(\mathbf{Set})$ | {ref}`Classicality Operator <sec-zfc-classicality>` |
| $\delta$ | Decidability Operator | $\delta: \text{Sub}(X) \to \Omega$; classifies decidable subobjects | {ref}`Classicality Operator <sec-zfc-classicality>` |
| IAC | Internal Axiom of Choice | Epimorphism splitting inside $\mathcal{E}$ (fails in non-trivial topoi) | {ref}`Internal vs External Choice <sec-zfc-internal-external-choice>` |
| EAC | External Axiom of Choice | Meta-theoretic AC in ambient set theory | {ref}`Internal vs External Choice <sec-zfc-internal-external-choice>` |
| $\mathcal{R}$ | Translation Residual | $\bigoplus_{n \geq 1} \pi_n(\mathcal{X})$; higher homotopy discarded by $\tau_0$ | {ref}`Translation Residual <sec-zfc-translation-residual>` |
| $\Psi(u)$ | Singular Exclusion | Set-theoretic translation of "no bad morphism lands on orbit $u$" | {ref}`Fundamental Theorem <sec-zfc-fundamental-theorem>` |

(sec-notation-key-bridge-theorems)=
## Key Bridge Theorems

| Label | Name | Statement Summary | Reference |
|-------|------|-------------------|-----------|
| {prf:ref}`thm-bridge-zfc-fundamental` | Fundamental Theorem of Set-Theoretic Reflection | $\mathcal{E} \models (\mathrm{Hom}(\mathbb{H}_{\mathrm{bad}}, \mathbb{H}) \simeq \emptyset) \implies V_\mathcal{U} \vDash \forall u \in \tau_0(\mathcal{X}), \Psi(u)$ | {ref}`Fundamental Theorem <sec-zfc-fundamental-theorem>` |
| {prf:ref}`cor-singular-contradiction` | Singular Point Contradiction | Set-theoretic non-existence of singular points satisfying $\mathbb{H}_{\mathrm{bad}}$ in $V_\mathcal{U}$ | {ref}`Fundamental Theorem <sec-zfc-fundamental-theorem>` |

(sec-notation-zfc-axiom-abbreviations)=
## ZFC Axiom Abbreviations

| Abbrev. | Full Name | Sieve Node Usage |
|---------|-----------|------------------|
| ZFC-Sep | Axiom of Separation | Nodes 1, 2, 5, 8, 12 |
| ZFC-Rep | Axiom of Replacement | Nodes 1, 7, 17 |
| ZFC-Pow | Axiom of Power Set | Node 3 |
| ZFC-Inf | Axiom of Infinity | Nodes 3, 9, 10 |
| ZFC-AC | Axiom of Choice | Nodes 3, 6, 10 (AC-dependent) |
| ZFC-Found | Axiom of Foundation | Nodes 4, 17 |
| ZFC-Ext | Axiom of Extensionality | Node 11 |

(sec-notation-ait)=
## Algorithmic Information Theory

| Symbol | Name | Definition | Reference |
|--------|------|------------|-----------|
| $K(x)$ | Kolmogorov Complexity | $\min\{|p| : U(p) = x\}$; shortest program length | {prf:ref}`def-kolmogorov-complexity` |
| $\Omega_U$ | Chaitin's Halting Probability | $\sum_{p : U(p)\downarrow} 2^{-|p|}$; partition function | {prf:ref}`def-chaitin-omega` |
| $d_s(x)$ | Computational Depth | Time of fastest program within $s$ bits of optimal | {prf:ref}`def-computational-depth` |
| $Kt(x)$ | Levin Complexity | $K(x) + \log t(x)$; resource-bounded measure | {prf:ref}`def-thermodynamic-horizon` |
| $m(x)$ | Algorithmic Probability | $\asymp 2^{-K(x)}$; Boltzmann weight analog | {prf:ref}`thm-sieve-thermo-correspondence` |

(sec-notation-ait-phase-classification)=
## AIT Phase Classification

| Phase | Complexity | Axiom R | Decidability | Sieve Verdict |
|-------|------------|---------|--------------|---------------|
| Crystal | $K = O(\log n)$ | Holds | Decidable | REGULAR |
| Liquid | $K = O(\log n)$ | Fails | C.E. not decidable | HORIZON |
| Gas | $K \geq n - O(1)$ | Fails | Random/Undecidable | HORIZON |

See {prf:ref}`def-algorithmic-phases` for formal definitions and {prf:ref}`thm-sieve-thermo-correspondence` for the Sieve-Thermodynamic Correspondence.
