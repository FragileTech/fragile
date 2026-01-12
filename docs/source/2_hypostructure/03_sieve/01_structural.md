# The Structural Sieve

(sec-homotopical-resolution)=
## The Homotopical Resolution of the Singularity Spectrum

*A Postnikov decomposition of the Regularity Functor $\mathcal{R}$ in a cohesive $(\infty,1)$-topos.*

The following diagram is the **authoritative specification** of the obstruction-theoretic resolution. All subsequent definitions and theorems must align with this categorical atlas.

:::{div} feynman-prose

Now, here is the thing you have to understand about singularities. When a dynamical system fails---when your robot arm crashes, when your neural network diverges, when your market simulation blows up---it is not just "failing." It is failing in one of a finite number of *specific ways*, each with its own signature, its own mathematics, its own potential repair.

The Structural Sieve is our systematic method for asking: "What went wrong, and can we fix it?" Think of it like a diagnostic flowchart for a car mechanic, except instead of checking the battery and the spark plugs, we are checking whether energy stayed bounded, whether topology remained tame, whether the flow mixed properly.

The beautiful thing---and I want you to really appreciate this---is that these failure modes are not arbitrary. They arise from the fundamental structure of dynamics itself: conservation laws, topological constraints, symmetry principles, boundary conditions. The Sieve simply asks, in order: "Did we violate conservation? Did we violate topology? Did we violate symmetry?" And so on.

:::

### The Taxonomy of Failure Modes

The singularity spectrum admits a natural classification by two orthogonal axes: the **constraint class** that is violated (Conservation, Topology, Duality, Symmetry, Boundary) and the **mechanism** of violation (Excess, Deficiency, Complexity). This yields the following periodic table of obstructions.

**Table 1: The Taxonomy of Failure Modes**
*The 15 fundamental ways a dynamical system can lose coherence.*

| Constraint       | Excess (Unbounded Growth)    | Deficiency (Collapse)             | Complexity (Entanglement)            |
|:-----------------|:-----------------------------|:----------------------------------|:-------------------------------------|
| **Conservation** | **Mode C.E**: Energy Blow-up | **Mode C.D**: Geometric Collapse  | **Mode C.C**: Event Accumulation     |
| **Topology**     | **Mode T.E**: Metastasis     | **Mode T.D**: Glassy Freeze       | **Mode T.C**: Labyrinthine           |
| **Duality**      | **Mode D.E**: Oscillatory    | **Mode D.D**: Dispersion          | **Mode D.C**: Semantic Horizon       |
| **Symmetry**     | **Mode S.E**: Supercritical  | **Mode S.D**: Stiffness Breakdown | **Mode S.C**: Parametric Instability |
| **Boundary**     | **Mode B.E**: Injection      | **Mode B.D**: Starvation          | **Mode B.C**: Misalignment           |

:::{div} feynman-prose

Let me walk you through this table, because it encodes a lot of physics in a small space.

The rows tell you *what kind of law* is being violated. Conservation means something that should stay constant---energy, probability mass, particle number---is leaking or exploding. Topology means the shape of your state space is doing something pathological. Duality relates to the relationship between forward and backward dynamics. Symmetry means your system had an invariance that is breaking in a bad way. Boundary means your system is not handling its inputs and outputs properly.

The columns tell you *how* the violation happens. Excess means something is growing without bound. Deficiency means something is collapsing or disappearing. Complexity means things are getting tangled in ways that prevent analysis.

Here is an example to make it concrete. Mode C.E (Conservation-Excess) is "Energy Blow-up"---your Hamiltonian is not conserving energy, and it is going to infinity. Mode T.D (Topology-Deficiency) is "Glassy Freeze"---your system got stuck in a region of configuration space from which it cannot escape, like a ball trapped in a local minimum with infinite barriers.

Each mode has its own surgery---its own repair mechanism---if such a repair is possible.

:::

### Computational Boundaries and Undecidability

:::{prf:remark} Acknowledgment of Fundamental Limits
:label: rem-undecidability

The Structural Sieve operates within the computational limits imposed by fundamental results in mathematical logic:

1. **Gödel's Incompleteness (1931):** No sufficiently powerful formal system can prove all true statements about arithmetic within itself {cite}`Godel31`.
2. **Halting Problem (Turing, 1936):** There is no general algorithm to determine whether an arbitrary program will halt {cite}`Turing36`.
3. **Rice's Theorem (1953):** All non-trivial semantic properties of programs are undecidable {cite}`Rice53`.

**Implication for the Sieve:** For sufficiently complex systems (e.g., those encoding universal computation), certain interface predicates $\mathcal{P}_i$ may be **undecidable**—no algorithm can determine their truth value in finite time for all inputs.

The framework addresses this through **Binary Certificate Logic** with typed NO certificates. Every predicate evaluation returns exactly YES or NO---never a third truth value. The NO certificate carries type information distinguishing *refutation* from *inconclusiveness*.
:::

:::{div} feynman-prose

Now here is a subtle point that trips people up, so let me explain it carefully.

When the Sieve asks a question like "Is energy finite?", it has to give an answer: YES or NO. But wait---what if we genuinely do not know? What if the problem is undecidable, or we ran out of time, or we lacked the right theorem to prove it either way?

The answer is: we still say NO. But---and this is crucial---we attach *metadata* to that NO explaining *why* we said it. A NO-with-witness means "I found a counterexample; this property definitely fails." A NO-inconclusive means "I could not prove it either way, so I am being conservative."

Why does this matter? Because the Sieve routes these two kinds of NO differently. A genuine refutation (NO-with-witness) routes to a failure mode---the system really is broken. An inconclusive answer routes to the reconstruction loop, where we try to acquire the missing information or extend our proof library.

This is how we remain honest about what we know while still making definite progress. The Sieve never hangs in an "I don't know" state. It always moves forward, but it remembers what kind of evidence led to its decisions.

:::

:::{prf:definition} Typed NO Certificates (Binary Certificate Logic)
:label: def-typed-no-certificates

For any predicate $P$ with YES certificate $K_P^+$, the NO certificate is a **coproduct** (sum type) in the category of certificate objects:

$$
K_P^- := K_P^{\mathrm{wit}} + K_P^{\mathrm{inc}}

$$

**Component 1: NO-with-witness** ($K_P^{\mathrm{wit}}$)

A constructive refutation consisting of a counterexample or breach object that demonstrates $\neg P$. Formally:

$$
K_P^{\mathrm{wit}} := (\mathsf{witness}: W_P, \mathsf{verification}: W_P \vdash \neg P)

$$

where $W_P$ is the type of refutation witnesses for $P$.

**Component 2: NO-inconclusive** ($K_P^{\mathrm{inc}}$)

A record of evaluator failure that does *not* constitute a semantic refutation. Formally:

$$
K_P^{\mathrm{inc}} := (\mathsf{obligation}: P, \mathsf{missing}: \mathcal{M}, \mathsf{code}: \mathcal{C}, \mathsf{trace}: \mathcal{T})

$$

where:
- $\mathsf{obligation} \in \mathrm{Pred}(\mathcal{H})$: The exact predicate instance attempted
- $\mathsf{missing} \in \mathcal{P}(\mathrm{Template} \cup \mathrm{Precond})$: Prerequisites or capabilities absent
- $\mathsf{code} \in \{\texttt{TEMPLATE\_MISS}, \texttt{PRECOND\_MISS}, \texttt{NOT\_IMPLEMENTED}, \texttt{RESOURCE\_LIMIT}, \texttt{UNDECIDABLE}\}$
- $\mathsf{trace} \in \mathrm{Log}$: Reproducible evaluation trace (template DB hash, attempted tactics, bounds)

**Injection Maps:** The coproduct structure provides canonical injections:

$$
\iota_{\mathrm{wit}}: K_P^{\mathrm{wit}} \to K_P^-, \quad \iota_{\mathrm{inc}}: K_P^{\mathrm{inc}} \to K_P^-

$$

**Case Analysis:** Any function $f: K_P^- \to X$ factors uniquely through case analysis:

$$
f = [f_{\mathrm{wit}}, f_{\mathrm{inc}}] \circ \mathrm{case}

$$

where $f_{\mathrm{wit}}: K_P^{\mathrm{wit}} \to X$ and $f_{\mathrm{inc}}: K_P^{\mathrm{inc}} \to X$.
:::

:::{prf:remark} Routing Semantics
:label: rem-routing-semantics

The Sieve branches on certificate kind via case analysis:
- **NO with $K^{\mathrm{wit}}$** $\mapsto$ Fatal route (structural inconsistency confirmed; no reconstruction possible)
- **NO with $K^{\mathrm{inc}}$** $\mapsto$ Reconstruction route (invoke {prf:ref}`mt-lock-reconstruction`; add interface/refine library/extend templates)

This design maintains **proof-theoretic honesty**:
- The verdict is always in $\{$YES, NO$\}$—classical two-valued logic
- The certificate carries the epistemic distinction between "refuted" and "not yet proven"
- Reconstruction is triggered by $K^{\mathrm{inc}}$, never by $K^{\mathrm{wit}}$

**Literature:** {cite}`Godel31`; {cite}`Turing36`; {cite}`Rice53`. For sum types in type theory: {cite}`MartinLof84`; {cite}`HoTTBook`.
:::

:::{admonition} Categorical Interpretation
:class: note

This Directed Acyclic Graph represents the **spectral sequence** of obstructions to global regularity.

- **Nodes (Objects):** Each node represents a **Classifying Stack** $\mathcal{M}_i$ for a specific obstruction class (Energy, Topology, Stiffness).
- **Solid Edges (Morphisms):** Represent **Truncation Functors** $\tau_{\leq k}$. A traversal $A \to B$ indicates that the obstruction at $A$ vanishes (is trivial in cohomology), allowing the lift to the next covering space $B$.
- **Dotted Edges (Surgery):** Represent **Cobordism Morphisms** in the category of manifolds. They denote a change of topology (Pushout) required to bypass a non-trivial cohomological obstruction.
- **Terminals (Limits):** The "Victory" node represents the **Contractible Space** (Global Regularity), where all homotopy groups of the singularity vanish.
:::

:::{prf:remark} The Sieve as Spectral Sequence
:label: rem-spectral-sequence

The Structural Sieve admits a natural interpretation as a **spectral sequence** $\{E_r^{p,q}, d_r\}_{r \geq 0}$ converging to the regularity classification:

- **$E_0^{p,q}$**: Initial Thin Kernel data, filtered by obstruction type ($p \in \{\text{Conservation}, \text{Duality}, \text{Symmetry}, \text{Topology}, \text{Boundary}\}$) and filtration level ($q$)
- **Differentials** $d_r: E_r^{p,q} \to E_r^{p+r, q-r+1}$: Obstruction maps at each sieve node
  - $d_1 \sim$ EnergyCheck: Tests finite energy ($\ker d_1 =$ bounded energy states)
  - $d_2 \sim$ CompactCheck: Tests concentration vs. dispersion
  - $d_3 \sim$ ScaleCheck: Tests subcriticality
- **Gate Pass** ($K^+$): Class survives to next page ($d_r(\alpha) = 0$)
- **Gate Fail** ($K^-$): Non-zero differential ($d_r(\alpha) \neq 0$), triggers barrier/surgery
- **Global Regularity**: Collapse at $E_\infty$ with $E_\infty^{p,q} = 0$ for all $(p,q)$ corresponding to singular behavior

This interpretation connects the Sieve to classical obstruction theory in algebraic topology {cite}`McCleary01`.
:::

:::{div} feynman-prose

Before you look at this diagram, let me tell you what you are going to see.

The diagram is the complete Structural Sieve---every check, every branch, every surgery, every failure mode. It is big, and that is okay. Real systems have many ways to fail, and each one needs to be addressed.

The blue diamonds are the main gates---questions like "Is energy finite?" or "Does the flow mix?" Each gate produces a certificate: YES if the property holds, NO if it does not. The orange diamonds are barriers---secondary defenses that activate when a primary gate fails. The purple boxes are surgeries---repair mechanisms that try to fix the problem and re-enter the Sieve at an earlier point. The red boxes are failure modes---terminal classifications of what went wrong.

Follow the flow from START at the top. If everything passes (YES at every gate), you reach GLOBAL REGULARITY at the bottom---victory. If something fails (NO at a gate), you branch to a barrier. If the barrier holds, you continue. If the barrier is breached, you check whether surgery is possible. If surgery works, you re-enter (dotted lines). If nothing works, you hit a failure mode.

The key insight: this is not just a flowchart. It is a *proof*. Every path through the diagram that reaches VICTORY is a formal argument that no singularity can form. The certificates accumulated along the way *are* the proof.

:::

:::{dropdown} 💡 Interactive Viewing Options
:open:

This diagram is large. For better viewing:
- **Zoom**: Use your browser's zoom (Ctrl/Cmd + scroll)
- **Full-screen editor**: [Open in Mermaid Live Editor](https://mermaid.live) and paste the code from the diagram below
- **Download**: In Mermaid Live Editor, use the export button to download as SVG or PNG
:::

```mermaid
graph TD
    Start(["<b>START DIAGNOSTIC</b>"]) --> EnergyCheck{"<b>1. D_E:</b> Is Energy Finite?<br>E[Φ] < ∞"}

    %% --- LEVEL 1: 0-TRUNCATION (Energy Bounds) ---
    EnergyCheck -- "No: K-_DE" --> BarrierSat{"<b>B1. D_E:</b> Is Drift Bounded?<br>E[Φ] ≤ E_sat"}
    BarrierSat -- "Yes: Kblk_DE" --> ZenoCheck
    BarrierSat -- "No: Kbr_DE" --> SurgAdmCE{"<b>A1. SurgCE:</b> Admissible?<br>conformal ∧ ∂∞X def."}
    SurgAdmCE -- "Yes: K+_Conf" --> SurgCE["<b>S1. SurgCE:</b><br>Ghost/Cap Extension"]
    SurgAdmCE -- "No: K-_Conf" --> ModeCE["<b>Mode C.E</b>: Energy Blow-Up"]
    SurgCE -. "Kre_SurgCE" .-> ZenoCheck

    EnergyCheck -- "Yes: K+_DE" --> ZenoCheck{"<b>2. Rec_N:</b> Are Discrete Events Finite?<br>N(J) < ∞"}
    ZenoCheck -- "No: K-_RecN" --> BarrierCausal{"<b>B2. Rec_N:</b> Infinite Depth?<br>D#40;T*#41; = ∞"}
    BarrierCausal -- "No: Kbr_RecN" --> SurgAdmCC{"<b>A2. SurgCC:</b> Admissible?<br>∃N_max: events ≤ N_max"}
    SurgAdmCC -- "Yes: K+_Disc" --> SurgCC["<b>S2. SurgCC:</b><br>Discrete Saturation"]
    SurgAdmCC -- "No: K-_Disc" --> ModeCC["<b>Mode C.C</b>: Event Accumulation"]
    SurgCC -. "Kre_SurgCC" .-> CompactCheck
    BarrierCausal -- "Yes: Kblk_RecN" --> CompactCheck

    ZenoCheck -- "Yes: K+_RecN" --> CompactCheck{"<b>3. C_μ:</b> Does Energy Concentrate?<br>μ(V) > 0"}

    %% --- LEVEL 2: COMPACTNESS LOCUS (Profile Moduli) ---
    CompactCheck -- "No: K-_Cmu" --> BarrierScat{"<b>B3. C_μ:</b> Is Interaction Finite?<br>M[Φ] < ∞"}
    BarrierScat -- "Yes: Kben_Cmu" --> ModeDD["<b>Mode D.D</b>: Dispersion<br><i>#40;Global Existence#41;</i>"]
    BarrierScat -- "No: Kpath_Cmu" --> SurgAdmCD_Alt{"<b>A3. SurgCD_Alt:</b> Admissible?<br>V ∈ L_soliton ∧ ‖V‖_H¹ < ∞"}
    SurgAdmCD_Alt -- "Yes: K+_Prof" --> SurgCD_Alt["<b>S3. SurgCD_Alt:</b><br>Concentration-Compactness"]
    SurgAdmCD_Alt -- "No: K-_Prof" --> ModeCD_Alt["<b>Mode C.D</b>: Geometric Collapse<br><i>#40;Via Escape#41;</i>"]
    SurgCD_Alt -. "Kre_SurgCD_Alt" .-> Profile

    CompactCheck -- "Yes: K+_Cmu" --> Profile["<b>Canonical Profile V Emerges</b>"]

    %% --- LEVEL 3: EQUIVARIANT DESCENT ---
    Profile --> ScaleCheck{"<b>4. SC_λ:</b> Is Profile Subcritical?<br>λ(V) < λ_c"}

    ScaleCheck -- "No: K-_SClam" --> BarrierTypeII{"<b>B4. SC_λ:</b> Is Renorm Cost ∞?<br>∫D̃ dt = ∞"}
    BarrierTypeII -- "No: Kbr_SClam" --> SurgAdmSE{"<b>A4. SurgSE:</b> Admissible?<br>α-β < ε_crit ∧ V smooth"}
    SurgAdmSE -- "Yes: K+_Lift" --> SurgSE["<b>S4. SurgSE:</b><br>Regularity Lift"]
    SurgAdmSE -- "No: K-_Lift" --> ModeSE["<b>Mode S.E</b>: Supercritical Cascade"]
    SurgSE -. "Kre_SurgSE" .-> ParamCheck
    BarrierTypeII -- "Yes: Kblk_SClam" --> ParamCheck

    ScaleCheck -- "Yes: K+_SClam" --> ParamCheck{"<b>5. SC_∂c:</b> Are Constants Stable?<br>‖∂c‖ < ε"}
    ParamCheck -- "No: K-_SCdc" --> BarrierVac{"<b>B5. SC_∂c:</b> Is Phase Stable?<br>ΔV > k_B T"}
    BarrierVac -- "No: Kbr_SCdc" --> SurgAdmSC{"<b>A5. SurgSC:</b> Admissible?<br>‖∂θ‖ < C_adm ∧ θ stable"}
    SurgAdmSC -- "Yes: K+_Stab" --> SurgSC["<b>S5. SurgSC:</b><br>Convex Integration"]
    SurgAdmSC -- "No: K-_Stab" --> ModeSC["<b>Mode S.C</b>: Parameter Instability"]
    SurgSC -. "Kre_SurgSC" .-> GeomCheck
    BarrierVac -- "Yes: Kblk_SCdc" --> GeomCheck

    ParamCheck -- "Yes: K+_SCdc" --> GeomCheck{"<b>6. Cap_H:</b> Is Codim ≥ Threshold?<br>codim(S) ≥ 2"}

    %% --- LEVEL 4: DIMENSION FILTRATION ---
    GeomCheck -- "No: K-_CapH" --> BarrierCap{"<b>B6. Cap_H:</b> Is Measure Zero?<br>Cap_H#40;S#41; = 0"}
    BarrierCap -- "No: Kbr_CapH" --> SurgAdmCD{"<b>A6. SurgCD:</b> Admissible?<br>Cap#40;Σ#41; ≤ ε ∧ V ∈ L_neck"}
    SurgAdmCD -- "Yes: K+_Neck" --> SurgCD["<b>S6. SurgCD:</b><br>Auxiliary/Structural"]
    SurgAdmCD -- "No: K-_Neck" --> ModeCD["<b>Mode C.D</b>: Geometric Collapse"]
    SurgCD -. "Kre_SurgCD" .-> StiffnessCheck
    BarrierCap -- "Yes: Kblk_CapH" --> StiffnessCheck

    GeomCheck -- "Yes: K+_CapH" --> StiffnessCheck{"<b>7. LS_σ:</b> Is Gap Certified?<br>inf σ(L) > 0"}

    %% --- LEVEL 5: SPECTRAL OBSTRUCTION ---
    StiffnessCheck -- "No: K-_LSsig" --> BarrierGap{"<b>B7. LS_σ:</b> Is Kernel Finite?<br>dim ker#40;L#41; < ∞ ∧ σ_ess > 0"}
    BarrierGap -- "Yes: Kblk_LSsig" --> TopoCheck
    BarrierGap -- "No: Kstag_LSsig" --> BifurcateCheck{"<b>7a. LS_∂²V:</b> Is State Unstable?<br>∂²V(x*) ⊁ 0"}

    %% --- LEVEL 5b: SPECTRAL RESTORATION (Bifurcation Resolution) ---
    BifurcateCheck -- "No: K-_LSd2V" --> SurgAdmSD{"<b>A7. SurgSD:</b> Admissible?<br>dim ker#40;H#41; < ∞ ∧ V iso."}
    SurgAdmSD -- "Yes: K+_Iso" --> SurgSD["<b>S7. SurgSD:</b><br>Ghost Extension"]
    SurgAdmSD -- "No: K-_Iso" --> ModeSD["<b>Mode S.D</b>: Stiffness Breakdown"]
    SurgSD -. "Kre_SurgSD" .-> TopoCheck
    BifurcateCheck -- "Yes: K+_LSd2V" --> SymCheck{"<b>7b. G_act:</b> Is G-orbit Degenerate?<br>⎸G·v₀⎸ = 1"}

    %% Path A: Symmetry Breaking (Governed by SC_∂c)
    SymCheck -- "Yes: K+_Gact" --> CheckSC{"<b>7c. SC_∂c:</b> Are Constants Stable?<br>‖∂c‖ < ε"}
    CheckSC -- "Yes: K+_SCdc" --> ActionSSB["<b>ACTION: SYM. BREAKING</b><br>Generates Mass Gap"]
    ActionSSB -- "Kgap" --> TopoCheck
    CheckSC -- "No: K-_SCdc" --> SurgAdmSC_Rest{"<b>A8. SurgSC_Rest:</b> Admissible?<br>ΔV > k_B T ∧ Γ < Γ_crit"}
    SurgAdmSC_Rest -- "Yes: K+_Vac" --> SurgSC_Rest["<b>S8. SurgSC_Rest:</b><br>Auxiliary Extension"]
    SurgAdmSC_Rest -- "No: K-_Vac" --> ModeSC_Rest["<b>Mode S.C</b>: Parameter Instability<br><i>#40;Vacuum Decay#41;</i>"]
    SurgSC_Rest -. "Kre_SurgSC_Rest" .-> TopoCheck

    %% Path B: Tunneling (Governed by TB_S)
    SymCheck -- "No: K-_Gact" --> CheckTB{"<b>7d. TB_S:</b> Is Tunneling Finite?<br>S[γ] < ∞"}
    CheckTB -- "Yes: K+_TBS" --> ActionTunnel["<b>ACTION: TUNNELING</b><br>Instanton Decay"]
    ActionTunnel -- "Ktunnel" --> TameCheck
    CheckTB -- "No: K-_TBS" --> SurgAdmTE_Rest{"<b>A9. SurgTE_Rest:</b> Admissible?<br>V ≅ S^n×I ∧ S_R[γ] < ∞"}
    SurgAdmTE_Rest -- "Yes: K+_Inst" --> SurgTE_Rest["<b>S9. SurgTE_Rest:</b><br>Structural"]
    SurgAdmTE_Rest -- "No: K-_Inst" --> ModeTE_Rest["<b>Mode T.E</b>: Topological Twist<br><i>#40;Metastasis#41;</i>"]
    SurgTE_Rest -. "Kre_SurgTE_Rest" .-> TameCheck

    StiffnessCheck -- "Yes: K+_LSsig" --> TopoCheck{"<b>8. TB_π:</b> Is Sector Reachable?<br>[π] ∈ π₀(C)_acc"}

    %% --- LEVEL 6: HOMOTOPICAL OBSTRUCTIONS ---
    TopoCheck -- "No: K-_TBpi" --> BarrierAction{"<b>B8. TB_π:</b> Energy < Gap?<br>E < S_min + Δ"}
    BarrierAction -- "No: Kbr_TBpi" --> SurgAdmTE{"<b>A10. SurgTE:</b> Admissible?<br>V ≅ S^n×R #40;Neck#41;"}
    SurgAdmTE -- "Yes: K+_Topo" --> SurgTE["<b>S10. SurgTE:</b><br>Tunnel"]
    SurgAdmTE -- "No: K-_Topo" --> ModeTE["<b>Mode T.E</b>: Topological Twist"]
    SurgTE -. "Kre_SurgTE" .-> TameCheck
    BarrierAction -- "Yes: Kblk_TBpi" --> TameCheck

    TopoCheck -- "Yes: K+_TBpi" --> TameCheck{"<b>9. TB_O:</b> Is Topology Tame?<br>Σ ∈ O-min"}

    TameCheck -- "No: K-_TBO" --> BarrierOmin{"<b>B9. TB_O:</b> Is It Definable?<br>S ∈ O-min"}
    BarrierOmin -- "No: Kbr_TBO" --> SurgAdmTC{"<b>A11. SurgTC:</b> Admissible?<br>Σ ∈ O-ext def. ∧ dim < n"}
    SurgAdmTC -- "Yes: K+_Omin" --> SurgTC["<b>S11. SurgTC:</b><br>O-minimal Regularization"]
    SurgAdmTC -- "No: K-_Omin" --> ModeTC["<b>Mode T.C</b>: Labyrinthine"]
    SurgTC -. "Kre_SurgTC" .-> ErgoCheck
    BarrierOmin -- "Yes: Kblk_TBO" --> ErgoCheck

    TameCheck -- "Yes: K+_TBO" --> ErgoCheck{"<b>10. TB_ρ:</b> Does Flow Mix?<br>τ_mix < ∞"}

    ErgoCheck -- "No: K-_TBrho" --> BarrierMix{"<b>B10. TB_ρ:</b> Mixing Finite?<br>τ_mix < ∞"}
    BarrierMix -- "No: Kbr_TBrho" --> SurgAdmTD{"<b>A12. SurgTD:</b> Admissible?<br>Trap iso. ∧ ∂T > 0"}
    SurgAdmTD -- "Yes: K+_Mix" --> SurgTD["<b>S12. SurgTD:</b><br>Mixing Enhancement"]
    SurgAdmTD -- "No: K-_Mix" --> ModeTD["<b>Mode T.D</b>: Glassy Freeze"]
    SurgTD -. "Kre_SurgTD" .-> ComplexCheck
    BarrierMix -- "Yes: Kblk_TBrho" --> ComplexCheck

    ErgoCheck -- "Yes: K+_TBrho" --> ComplexCheck{"<b>11. Rep_K:</b> Is K(x) Computable?<br>K(x) ∈ ℕ"}

    %% --- LEVEL 7: KOLMOGOROV FILTRATION ---
    ComplexCheck -- "No: K-_RepK" --> BarrierEpi{"<b>B11. Rep_K:</b> Approx. Bounded?<br>sup K_ε#40;x#41; ≤ S_BH"}
    BarrierEpi -- "No: Kbr_RepK" --> SurgAdmDC{"<b>A13. SurgDC:</b> Admissible?<br>K ≤ S_BH+ε ∧ Lipschitz"}
    SurgAdmDC -- "Yes: K+_Lip" --> SurgDC["<b>S13. SurgDC:</b><br>Viscosity Solution"]
    SurgAdmDC -- "No: K-_Lip" --> ModeDC["<b>Mode D.C</b>: Semantic Horizon"]
    SurgDC -. "Kre_SurgDC" .-> OscillateCheck
    BarrierEpi -- "Yes: Kblk_RepK" --> OscillateCheck

    ComplexCheck -- "Yes: K+_RepK" --> OscillateCheck{"<b>12. GC_∇:</b> Does Flow Oscillate?<br>ẋ ≠ -∇V"}

    OscillateCheck -- "Yes: K+_GCnabla" --> BarrierFreq{"<b>B12. GC_∇:</b> Oscillation Finite?<br>∫ω²S dω < ∞"}
    BarrierFreq -- "No: Kbr_GCnabla" --> SurgAdmDE{"<b>A14. SurgDE:</b> Admissible?<br>∃Λ: trunc. moment < ∞ ∧ elliptic"}
    SurgAdmDE -- "Yes: K+_Ell" --> SurgDE["<b>S14. SurgDE:</b><br>De Giorgi-Nash-Moser"]
    SurgAdmDE -- "No: K-_Ell" --> ModeDE["<b>Mode D.E</b>: Oscillatory"]
    SurgDE -. "Kre_SurgDE" .-> BoundaryCheck
    BarrierFreq -- "Yes: Kblk_GCnabla" --> BoundaryCheck

    OscillateCheck -- "No: K-_GCnabla" --> BoundaryCheck{"<b>13. Bound_∂:</b> Is System Open?<br>∂Ω ≠ ∅"}

    %% --- LEVEL 8: BOUNDARY COBORDISM ---
    BoundaryCheck -- "Yes: K+_Bound" --> OverloadCheck{"<b>14. Bound_B:</b> Is Input Bounded?<br>‖Bu‖ ≤ M"}

    OverloadCheck -- "No: K-_BoundB" --> BarrierBode{"<b>B14. Bound_B:</b> Waterbed Bounded?<br>∫ln‖S‖dω > -∞"}
    BarrierBode -- "No: Kbr_BoundB" --> SurgAdmBE{"<b>A15. SurgBE:</b> Admissible?<br>‖S‖_∞ < M ∧ φ_margin > 0"}
    SurgAdmBE -- "Yes: K+_Marg" --> SurgBE["<b>S15. SurgBE:</b><br>Saturation"]
    SurgAdmBE -- "No: K-_Marg" --> ModeBE["<b>Mode B.E</b>: Injection"]
    SurgBE -. "Kre_SurgBE" .-> StarveCheck
    BarrierBode -- "Yes: Kblk_BoundB" --> StarveCheck

    OverloadCheck -- "Yes: K+_BoundB" --> StarveCheck{"<b>15. Bound_∫:</b> Is Input Sufficient?<br>∫r dt ≥ r_min"}

    StarveCheck -- "No: K-_BoundInt" --> BarrierInput{"<b>B15. Bound_∫:</b> Reserve Positive?<br>r_reserve > 0"}
    BarrierInput -- "No: Kbr_BoundInt" --> SurgAdmBD{"<b>A16. SurgBD:</b> Admissible?<br>r_res > 0 ∧ recharge > drain"}
    SurgAdmBD -- "Yes: K+_Res" --> SurgBD["<b>S16. SurgBD:</b><br>Reservoir"]
    SurgAdmBD -- "No: K-_Res" --> ModeBD["<b>Mode B.D</b>: Starvation"]
    SurgBD -. "Kre_SurgBD" .-> AlignCheck
    BarrierInput -- "Yes: Kblk_BoundInt" --> AlignCheck

    StarveCheck -- "Yes: K+_BoundInt" --> AlignCheck{"<b>16. GC_T:</b> Is Control Matched?<br>T(u) ~ d"}
    AlignCheck -- "No: K-_GCT" --> BarrierVariety{"<b>B16. GC_T:</b> Variety Sufficient?<br>H#40;u#41; ≥ H#40;d#41;"}
    BarrierVariety -- "No: Kbr_GCT" --> SurgAdmBC{"<b>A17. SurgBC:</b> Admissible?<br>H#40;u#41; < H#40;d#41; ∧ bridgeable"}
    SurgAdmBC -- "Yes: K+_Ent" --> SurgBC["<b>S17. SurgBC:</b><br>Controller Augmentation"]
    SurgAdmBC -- "No: K-_Ent" --> ModeBC["<b>Mode B.C</b>: Misalignment"]
    SurgBC -. "Kre_SurgBC" .-> BarrierExclusion

    %% --- LEVEL 9: THE COHOMOLOGICAL BARRIER ---
    %% All successful paths funnel here
    BoundaryCheck -- "No: K-_Bound" --> BarrierExclusion
    BarrierVariety -- "Yes: Kblk_GCT" --> BarrierExclusion
    AlignCheck -- "Yes: K+_GCT" --> BarrierExclusion

    BarrierExclusion{"<b>17. Cat_Hom:</b> Is Hom#40;Bad, S#41; = ∅?<br>Hom#40;B, S#41; = ∅"}

    BarrierExclusion -- "Yes: Kblk_CatHom" --> VICTORY(["<b>GLOBAL REGULARITY</b><br><i>#40;Structural Exclusion Confirmed#41;</i>"])
    BarrierExclusion -- "No: Kmorph_CatHom" --> ModeCat["<b>FATAL ERROR</b><br>Structural Inconsistency"]
    BarrierExclusion -- "NO(inc): Kbr-inc_CatHom" --> ReconstructionLoop["<b>LOCK-Reconstruction:</b><br>Structural Reconstruction"]
    ReconstructionLoop -- "Verdict: Kblk" --> VICTORY
    ReconstructionLoop -- "Verdict: Kmorph" --> ModeCat

    %% ====== STYLES ======
    %% Success states - Green
    style VICTORY fill:#22c55e,stroke:#16a34a,color:#000000,stroke-width:4px
    style ModeDD fill:#22c55e,stroke:#16a34a,color:#000000

    %% Failure modes - Red
    style ModeCE fill:#ef4444,stroke:#dc2626,color:#ffffff
    style ModeCC fill:#ef4444,stroke:#dc2626,color:#ffffff
    style ModeSE fill:#ef4444,stroke:#dc2626,color:#ffffff
    style ModeSC fill:#ef4444,stroke:#dc2626,color:#ffffff
    style ModeCD fill:#ef4444,stroke:#dc2626,color:#ffffff
    style ModeCD_Alt fill:#ef4444,stroke:#dc2626,color:#ffffff
    style ModeSD fill:#ef4444,stroke:#dc2626,color:#ffffff
    style ModeTE fill:#ef4444,stroke:#dc2626,color:#ffffff
    style ModeTC fill:#ef4444,stroke:#dc2626,color:#ffffff
    style ModeTD fill:#ef4444,stroke:#dc2626,color:#ffffff
    style ModeDC fill:#ef4444,stroke:#dc2626,color:#ffffff
    style ModeDE fill:#ef4444,stroke:#dc2626,color:#ffffff
    style ModeBE fill:#ef4444,stroke:#dc2626,color:#ffffff
    style ModeBD fill:#ef4444,stroke:#dc2626,color:#ffffff
    style ModeBC fill:#ef4444,stroke:#dc2626,color:#ffffff
    style ModeCat fill:#ef4444,stroke:#dc2626,color:#ffffff

    %% Barriers - Orange/Amber
    style BarrierSat fill:#f59e0b,stroke:#d97706,color:#000000
    style BarrierCausal fill:#f59e0b,stroke:#d97706,color:#000000
    style BarrierScat fill:#f59e0b,stroke:#d97706,color:#000000
    style BarrierTypeII fill:#f59e0b,stroke:#d97706,color:#000000
    style BarrierVac fill:#f59e0b,stroke:#d97706,color:#000000
    style BarrierCap fill:#f59e0b,stroke:#d97706,color:#000000
    style BarrierGap fill:#f59e0b,stroke:#d97706,color:#000000
    style BarrierAction fill:#f59e0b,stroke:#d97706,color:#000000
    style BarrierOmin fill:#f59e0b,stroke:#d97706,color:#000000
    style BarrierMix fill:#f59e0b,stroke:#d97706,color:#000000
    style BarrierEpi fill:#f59e0b,stroke:#d97706,color:#000000
    style BarrierFreq fill:#f59e0b,stroke:#d97706,color:#000000
    style BarrierBode fill:#f59e0b,stroke:#d97706,color:#000000
    style BarrierInput fill:#f59e0b,stroke:#d97706,color:#000000
    style BarrierVariety fill:#f59e0b,stroke:#d97706,color:#000000

    %% Reconstruction Loop - Yellow/Gold
    style ReconstructionLoop fill:#fbbf24,stroke:#f59e0b,color:#000000,stroke-width:2px

    %% The Final Gate - Purple with thick border
    style BarrierExclusion fill:#8b5cf6,stroke:#7c3aed,color:#ffffff,stroke-width:4px

    %% Interface Checks - Blue
    style EnergyCheck fill:#3b82f6,stroke:#2563eb,color:#ffffff
    style ZenoCheck fill:#3b82f6,stroke:#2563eb,color:#ffffff
    style CompactCheck fill:#3b82f6,stroke:#2563eb,color:#ffffff
    style ScaleCheck fill:#3b82f6,stroke:#2563eb,color:#ffffff
    style ParamCheck fill:#3b82f6,stroke:#2563eb,color:#ffffff
    style GeomCheck fill:#3b82f6,stroke:#2563eb,color:#ffffff
    style StiffnessCheck fill:#3b82f6,stroke:#2563eb,color:#ffffff
    style TopoCheck fill:#3b82f6,stroke:#2563eb,color:#ffffff
    style TameCheck fill:#3b82f6,stroke:#2563eb,color:#ffffff
    style ErgoCheck fill:#3b82f6,stroke:#2563eb,color:#ffffff
    style ComplexCheck fill:#3b82f6,stroke:#2563eb,color:#ffffff
    style OscillateCheck fill:#3b82f6,stroke:#2563eb,color:#ffffff
    style BoundaryCheck fill:#3b82f6,stroke:#2563eb,color:#ffffff
    style OverloadCheck fill:#3b82f6,stroke:#2563eb,color:#ffffff
    style StarveCheck fill:#3b82f6,stroke:#2563eb,color:#ffffff
    style AlignCheck fill:#3b82f6,stroke:#2563eb,color:#ffffff

    %% Intermediate nodes - Purple
    style Start fill:#8b5cf6,stroke:#7c3aed,color:#ffffff
    style Profile fill:#8b5cf6,stroke:#7c3aed,color:#ffffff

    %% Restoration checks - Blue (interface permit checks)
    style BifurcateCheck fill:#3b82f6,stroke:#2563eb,color:#ffffff
    style SymCheck fill:#3b82f6,stroke:#2563eb,color:#ffffff
    style CheckSC fill:#3b82f6,stroke:#2563eb,color:#ffffff
    style CheckTB fill:#3b82f6,stroke:#2563eb,color:#ffffff

    %% Restoration mechanisms - Purple (escape mechanisms)
    style ActionSSB fill:#8b5cf6,stroke:#7c3aed,color:#ffffff
    style ActionTunnel fill:#8b5cf6,stroke:#7c3aed,color:#ffffff

    %% Restoration failure modes - Red
    style ModeSC_Rest fill:#ef4444,stroke:#dc2626,color:#ffffff
    style ModeTE_Rest fill:#ef4444,stroke:#dc2626,color:#ffffff

    %% Surgery recovery nodes - Purple
    style SurgCE fill:#8b5cf6,stroke:#7c3aed,color:#ffffff
    style SurgCC fill:#8b5cf6,stroke:#7c3aed,color:#ffffff
    style SurgCD_Alt fill:#8b5cf6,stroke:#7c3aed,color:#ffffff
    style SurgSE fill:#8b5cf6,stroke:#7c3aed,color:#ffffff
    style SurgSC fill:#8b5cf6,stroke:#7c3aed,color:#ffffff
    style SurgCD fill:#8b5cf6,stroke:#7c3aed,color:#ffffff
    style SurgSD fill:#8b5cf6,stroke:#7c3aed,color:#ffffff
    style SurgSC_Rest fill:#8b5cf6,stroke:#7c3aed,color:#ffffff
    style SurgTE_Rest fill:#8b5cf6,stroke:#7c3aed,color:#ffffff
    style SurgTE fill:#8b5cf6,stroke:#7c3aed,color:#ffffff
    style SurgTC fill:#8b5cf6,stroke:#7c3aed,color:#ffffff
    style SurgTD fill:#8b5cf6,stroke:#7c3aed,color:#ffffff
    style SurgDC fill:#8b5cf6,stroke:#7c3aed,color:#ffffff
    style SurgDE fill:#8b5cf6,stroke:#7c3aed,color:#ffffff
    style SurgBE fill:#8b5cf6,stroke:#7c3aed,color:#ffffff
    style SurgBD fill:#8b5cf6,stroke:#7c3aed,color:#ffffff
    style SurgBC fill:#8b5cf6,stroke:#7c3aed,color:#ffffff

    %% Surgery Admissibility checks - Light Purple with border
    style SurgAdmCE fill:#e9d5ff,stroke:#9333ea,color:#000000,stroke-width:2px
    style SurgAdmCC fill:#e9d5ff,stroke:#9333ea,color:#000000,stroke-width:2px
    style SurgAdmCD_Alt fill:#e9d5ff,stroke:#9333ea,color:#000000,stroke-width:2px
    style SurgAdmSE fill:#e9d5ff,stroke:#9333ea,color:#000000,stroke-width:2px
    style SurgAdmSC fill:#e9d5ff,stroke:#9333ea,color:#000000,stroke-width:2px
    style SurgAdmCD fill:#e9d5ff,stroke:#9333ea,color:#000000,stroke-width:2px
    style SurgAdmSD fill:#e9d5ff,stroke:#9333ea,color:#000000,stroke-width:2px
    style SurgAdmSC_Rest fill:#e9d5ff,stroke:#9333ea,color:#000000,stroke-width:2px
    style SurgAdmTE_Rest fill:#e9d5ff,stroke:#9333ea,color:#000000,stroke-width:2px
    style SurgAdmTE fill:#e9d5ff,stroke:#9333ea,color:#000000,stroke-width:2px
    style SurgAdmTC fill:#e9d5ff,stroke:#9333ea,color:#000000,stroke-width:2px
    style SurgAdmTD fill:#e9d5ff,stroke:#9333ea,color:#000000,stroke-width:2px
    style SurgAdmDC fill:#e9d5ff,stroke:#9333ea,color:#000000,stroke-width:2px
    style SurgAdmDE fill:#e9d5ff,stroke:#9333ea,color:#000000,stroke-width:2px
    style SurgAdmBE fill:#e9d5ff,stroke:#9333ea,color:#000000,stroke-width:2px
    style SurgAdmBD fill:#e9d5ff,stroke:#9333ea,color:#000000,stroke-width:2px
    style SurgAdmBC fill:#e9d5ff,stroke:#9333ea,color:#000000,stroke-width:2px

```

:::{prf:remark} Operational Semantics of the Diagram
:label: rem-operational-semantics

We interpret this diagram as the computation of the **Limit** of a diagram of shapes in the $(\infty,1)$-topos of Hypostructures.

The flow proceeds by **Iterative Obstruction Theory**:

1. **Filtration:** The hierarchy (Levels 1–9) establishes a filtration of the moduli space of singularities by obstruction complexity.

2. **Lifting:** A "Yes" branch represents the successful lifting of the solution across an obstruction class—e.g., from $L^2$ energy bounds to $H^1$ regularity. The functor projects the system onto the relevant cohomology; if the class is trivial, the system lifts to the next level.

3. **Surgery as Cobordism:** The dotted "Surgery" loops represent the active cancellation of a non-trivial cohomology class (the singularity) via geometric modification. These are **Pushouts** in the category of manifolds—changing topology to bypass obstructions.

4. **Convergence to the Limit:** The **Cohomological Obstruction** ({prf:ref}`def-node-lock`) verifies that the **Inverse Limit** of this tower is the empty set—i.e., all obstruction classes vanish—thereby proving $\mathrm{Sing}(\Phi) = \emptyset$.

5. **The Structure Sheaf:** The accumulation of certificates $\Gamma$ forms a **Structure Sheaf** $\mathcal{O}_{\mathrm{Reg}}$ over the trajectory space. A "Victory" is a proof that the **Global Sections** of the singularity sheaf vanish.
:::

---

### Interface Registry: The Obstruction Atlas

:::{div} feynman-prose

Now we need to define precisely what each node in that big diagram *does*. This is where the engineering meets the mathematics.

Each node asks a specific question about the system. "Is energy finite?" is a question you can answer by computing an integral. "Is the spectral gap positive?" is a question you can answer by analyzing an operator. The table below lists every interface---every question the Sieve asks---along with the mathematical predicate that defines it.

Think of this as the specification sheet for building an actual Sieve implementation. If you wanted to build this for your favorite dynamical system, you would go through this table and implement each predicate for your particular setup.

:::

The following table defines the **Obstruction Atlas**---the collection of classifying stacks and their associated projection functors. Each interface evaluates whether a specific cohomology class vanishes.

| Interpretation | Engineering Term | Categorical Term |
|----------------|------------------|------------------|
| Node | Check | Classifying Stack $\mathcal{M}_i$ |
| Edge (Yes) | Pass | Truncation Functor $\tau_{\leq k}$ |
| Edge (No) | Fail | Non-trivial Obstruction Class |
| Certificate | Token | Section of Structure Sheaf |

To instantiate the sieve for a specific system, one must implement each projection functor for the relevant hypostructure component.

| Node | ID                            | Name             | Certificates (Output)                                                                                 | Symbol         | Object                   | Hypostructure                   | Description                        | Question                                        | Predicate                                     |
|------|-------------------------------|------------------|-------------------------------------------------------------------------------------------------------|----------------|--------------------------|---------------------------------|------------------------------------|-------------------------------------------------|-----------------------------------------------|
| 1    | $D_E$                         | EnergyCheck      | $K_{D_E}^+$ / $K_{D_E}^-$                                                                             | $E$            | Flow $\Phi$              | $\mathfrak{D}$ on $\Phi$        | Energy functional                  | Is Energy Finite?                               | $E[\Phi] < \infty$                            |
| 2    | $\mathrm{Rec}_N$              | ZenoCheck        | $K_{\mathrm{Rec}_N}^+$ / $K_{\mathrm{Rec}_N}^-$                                                       | $N$            | Jump sequence $J$        | $\mathfrak{D}$ on $\Phi$        | Event counter                      | Are Discrete Events Finite?                     | $N(J) < \infty$                               |
| 3    | $C_\mu$                       | CompactCheck     | $K_{C_\mu}^+$ / $K_{C_\mu}^-$                                                                         | $\mu$          | Profile $V$              | $\mathfrak{D}$ on $\mathcal{X}$ | Concentration measure              | Does Energy Concentrate?                        | $\mu(V) > 0$                                  |
| 4    | $\mathrm{SC}_\lambda$         | ScaleCheck       | $K_{\mathrm{SC}_\lambda}^+$ / $K_{\mathrm{SC}_\lambda}^-$                                             | $\lambda$      | Profile $V$              | $\mathfrak{D}$ on $\mathcal{X}$ | Scaling dimension                  | Is Profile Subcritical?                         | $\lambda(V) < \lambda_c$                      |
| 5    | $\mathrm{SC}_{\partial c}$    | ParamCheck       | $K_{\mathrm{SC}_{\partial c}}^+$ / $K_{\mathrm{SC}_{\partial c}}^-$                                   | $\partial c$   | Constants $c$            | $\mathfrak{D}$ on $\mathcal{X}$ | Parameter derivative               | Are Constants Stable?                           | $\|\partial_c\| < \epsilon$           |
| 6    | $\mathrm{Cap}_H$              | GeomCheck        | $K_{\mathrm{Cap}_H}^+$ / $K_{\mathrm{Cap}_H}^-$                                                       | $\dim_H$       | Singular set $S$         | $\mathfrak{D}$ on $\mathcal{X}$ | Hausdorff dimension                | Is Codim $\geq$ Threshold?                      | $\mathrm{codim}(S) \geq 2$                    |
| 7    | $\mathrm{LS}_\sigma$          | StiffnessCheck   | $K_{\mathrm{LS}_\sigma}^+$ / $K_{\mathrm{LS}_\sigma}^-$                                               | $\sigma$       | Linearization $L$        | $\mathfrak{D}$ on $\Phi$        | Spectrum                           | Is Gap Certified?                               | $\inf \sigma(L) > 0$                          |
| 7a   | $\mathrm{LS}_{\partial^2 V}$  | BifurcateCheck   | $K_{\mathrm{LS}_{\partial^2 V}}^+$ / $K_{\mathrm{LS}_{\partial^2 V}}^-$                               | $\partial^2 V$ | Equilibrium $x^*$        | $\mathfrak{D}$ on $\mathcal{X}$ | Hessian                            | Is State Unstable?                              | $\partial^2 V(x^*) \not\succ 0$               |
| 7b   | $G_{\mathrm{act}}$            | SymCheck         | $K_{G_{\mathrm{act}}}^+$ / $K_{G_{\mathrm{act}}}^-$                                                   | $G$            | Vacuum $v_0$             | $G$                             | Group action                       | Is $G$-orbit Degenerate?                        | $\lvert G \cdot v_0 \rvert = 1$               |
| 7c   | $\mathrm{SC}_{\partial c}$    | CheckSC          | $K_{\mathrm{SC}_{\partial c}}^+$ / $K_{\mathrm{SC}_{\partial c}}^-$                                   | $\partial c$   | Constants $c$            | $\mathfrak{D}$ on $\mathcal{X}$ | Parameter derivative (restoration) | Are Constants Stable?                           | $\|\partial_c\| < \epsilon$           |
| 7d   | $\mathrm{TB}_S$               | CheckTB          | $K_{\mathrm{TB}_S}^+$ / $K_{\mathrm{TB}_S}^-$                                                         | $S$            | Instanton path $\gamma$  | $\mathfrak{D}$ on $\mathcal{X}$ | Action functional                  | Is Tunneling Finite?                            | $S[\gamma] < \infty$                          |
| 8    | $\mathrm{TB}_\pi$             | TopoCheck        | $K_{\mathrm{TB}_\pi}^+$ / $K_{\mathrm{TB}_\pi}^-$                                                     | $\pi$          | Configuration $C$        | $\mathfrak{D}$ on $\mathcal{X}$ | Homotopy class                     | Is Sector Reachable?                            | $[\pi] \in \pi_0(\mathcal{C})_{\mathrm{acc}}$ |
| 9    | $\mathrm{TB}_O$               | TameCheck        | $K_{\mathrm{TB}_O}^+$ / $K_{\mathrm{TB}_O}^-$                                                         | $O$            | Stratification $\Sigma$  | $\mathfrak{D}$ on $\mathcal{X}$ | O-minimal structure                | Is Topology Tame?                               | $\Sigma \in \mathcal{O}\text{-min}$           |
| 10   | $\mathrm{TB}_\rho$            | ErgoCheck        | $K_{\mathrm{TB}_\rho}^+$ / $K_{\mathrm{TB}_\rho}^-$                                                   | $\rho$         | Invariant measure $\mu$  | $\mathfrak{D}$ on $\Phi$        | Mixing rate                        | Does Flow Mix?                                  | $\rho(\mu) > 0$                               |
| 11   | $\mathrm{Rep}_K$              | ComplexCheck     | $K_{\mathrm{Rep}_K}^+$ / $K_{\mathrm{Rep}_K}^-$                                                       | $K$            | State $x$                | $\mathfrak{D}$ on $\mathcal{X}$ | Kolmogorov complexity              | Is K(x) Computable?                             | $K(x) \in \mathbb{N}$                         |
| 12   | $\mathrm{GC}_\nabla$          | OscillateCheck   | $K_{\mathrm{GC}_\nabla}^+$ / $K_{\mathrm{GC}_\nabla}^-$                                               | $\nabla$       | Potential $V$            | $\mathfrak{D}$ on $\mathcal{X}$ | Gradient operator                  | Does Flow Oscillate?                            | $\dot{x} \neq -\nabla V$                      |
| 13   | $\mathrm{Bound}_\partial$     | BoundaryCheck    | $K_{\mathrm{Bound}_\partial}^+$ / $K_{\mathrm{Bound}_\partial}^-$                                     | $\partial$     | Domain $\Omega$          | $\mathfrak{D}$ on $\mathcal{X}$ | Boundary operator                  | Is System Open?                                 | $\partial\Omega \neq \emptyset$               |
| 14   | $\mathrm{Bound}_B$            | OverloadCheck    | $K_{\mathrm{Bound}_B}^+$ / $K_{\mathrm{Bound}_B}^-$                                                   | $B$            | Control signal $u$       | $\mathfrak{D}$ on $\Phi$        | Input operator                     | Is Input Bounded?                               | $\|Bu\| \leq M$                     |
| 15   | $\mathrm{Bound}_{\Sigma}$         | StarveCheck      | $K_{\mathrm{Bound}_{\Sigma}}^+$ / $K_{\mathrm{Bound}_{\Sigma}}^-$                                             | $\int$         | Resource $r$             | $\mathfrak{D}$ on $\Phi$        | Supply integral                    | Is Input Sufficient?                            | $\int_0^T r \, dt \geq r_{\min}$              |
| 16   | $\mathrm{GC}_T$               | AlignCheck       | $K_{\mathrm{GC}_T}^+$ / $K_{\mathrm{GC}_T}^-$                                                         | $T$            | Pair $(u,d)$             | $\mathfrak{D}$ on $\Phi$        | Gauge transform                    | Is Control Matched?                             | $T(u) \sim d$                                 |
| 17   | $\mathrm{Cat}_{\mathrm{Hom}}$ | BarrierExclusion | $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$ / $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{morph}}$ | $\mathrm{Hom}$ | Morphisms $\mathrm{Mor}$ | $\mathfrak{D}$ categorical      | Hom functor                        | Is $\mathrm{Hom}(\mathrm{Bad}, S) = \emptyset$? | $\mathrm{Hom}(\mathcal{B}, S) = \emptyset$    |

:::{prf:remark} Interface Composition
:label: rem-interface-composition

Barrier checks compose multiple interfaces. For example, the **Saturation Barrier** at {prf:ref}`def-node-energy` combines the energy interface $D_E$ with a drift control predicate. Surgery admissibility checks (the light purple diamonds) query the same interfaces as their parent gates but with different predicates.
:::

### Barrier Registry: Secondary Obstruction Classes

:::{div} feynman-prose

Here is where the defense-in-depth strategy becomes concrete.

When a primary gate fails---say, energy is not finite---that does not immediately mean game over. There may be a *weaker* condition that still excludes pathology. Maybe the energy is infinite but the *drift* is bounded, so it cannot blow up too fast. Maybe the topology is not tame in general but it is still *definable* in a simple logical system.

Barriers are these weaker conditions. They give the system a second chance. If the barrier holds ("Blocked"), we can continue to the next gate---we have found a workaround. If the barrier is breached, then we must consider surgery or accept a failure mode.

Think of barriers as fallback defenses. The primary gate is the first line of defense; the barrier is the second. Many systems fail the primary check but pass the barrier, and that is fine. The Sieve only terminates in failure when both lines of defense fall.

:::

The following table defines the **Secondary Obstruction Classes**---cohomological barriers that activate when the primary obstruction is non-trivial. Each barrier represents a weaker cohomology condition that may still force triviality of the singularity class.

| Node | Barrier ID       | Interfaces                                       | Permits ($\Gamma$)                         | Certificates (Output)                                                                                 | Blocked Predicate                                              | Question                                                     | Metatheorem                |
|------|------------------|--------------------------------------------------|--------------------------------------------|-------------------------------------------------------------------------------------------------------|----------------------------------------------------------------|--------------------------------------------------------------|----------------------------|
| 1    | BarrierSat       | $D_E$, $\mathrm{SC}_\lambda$                     | $\emptyset$ (Entry)                        | $K_{D_E}^{\mathrm{blk}}$ / $K_{D_E}^{\mathrm{br}}$                                                    | $E[\Phi] \leq E_{\text{sat}} \lor \operatorname{Drift} \leq C$         | Is the energy drift bounded by a saturation ceiling?         | Saturation Principle       |
| 2    | BarrierCausal    | $\mathrm{Rec}_N$, $\mathrm{TB}_\pi$              | $K_{D_E}^\pm$                              | $K_{\mathrm{Rec}_N}^{\mathrm{blk}}$ / $K_{\mathrm{Rec}_N}^{\mathrm{br}}$                              | $D(T_*) = \int_0^{T_*} \frac{c}{\lambda(t)} dt = \infty$       | Does the singularity require infinite computational depth?   | Algorithmic Causal Barrier |
| 3    | BarrierScat      | $C_\mu$, $D_E$                                   | $K_{D_E}^\pm, K_{\mathrm{Rec}_N}^\pm$      | $K_{C_\mu}^{\mathrm{ben}}$ / $K_{C_\mu}^{\mathrm{path}}$                                              | $\mathcal{M}[\Phi] < \infty$                                   | Is the interaction functional finite (implying dispersion)?  | Scattering-Compactness     |
| 4    | BarrierTypeII    | $\mathrm{SC}_\lambda$, $D_E$                     | $K_{C_\mu}^+$                              | $K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}$ / $K_{\mathrm{SC}_\lambda}^{\mathrm{br}}$                    | $\int \tilde{\mathfrak{D}}(S_t V) dt = \infty$                 | Is the renormalization cost of the profile infinite?         | Type II Exclusion          |
| 5    | BarrierVac       | $\mathrm{SC}_{\partial c}$, $\mathrm{LS}_\sigma$ | $K_{C_\mu}^+, K_{\mathrm{SC}_\lambda}^\pm$ | $K_{\mathrm{SC}_{\partial c}}^{\mathrm{blk}}$ / $K_{\mathrm{SC}_{\partial c}}^{\mathrm{br}}$          | $\Delta V > k_B T$                                             | Is the phase stable against thermal/parameter drift?         | Mass Gap Principle         |
| 6    | BarrierCap       | $\mathrm{Cap}_H$                                 | $K_{\mathrm{SC}_{\partial c}}^\pm$         | $K_{\mathrm{Cap}_H}^{\mathrm{blk}}$ / $K_{\mathrm{Cap}_H}^{\mathrm{br}}$                              | $\operatorname{Cap}_H(S) = 0$                                        | Is the singular set of measure zero?                         | Capacity Barrier           |
| 7    | BarrierGap       | $\mathrm{LS}_\sigma$, $\mathrm{GC}_\nabla$       | $K_{\mathrm{Cap}_H}^\pm$                   | $K_{\mathrm{LS}_\sigma}^{\mathrm{blk}}$ / $K_{\mathrm{LS}_\sigma}^{\mathrm{stag}}$                    | $\dim(\ker L) < \infty \land \sigma_{\mathrm{ess}}(L) > 0$     | Is the kernel finite-dimensional with essential spectral gap? | Spectral Generator         |
| 8    | BarrierAction    | $\mathrm{TB}_\pi$                                | $K_{\mathrm{LS}_\sigma}^\pm$               | $K_{\mathrm{TB}_\pi}^{\mathrm{blk}}$ / $K_{\mathrm{TB}_\pi}^{\mathrm{br}}$                            | $E[\Phi] < S_{\min} + \Delta$                                  | Is the energy insufficient to cross the topological gap?     | Topological Suppression    |
| 9    | BarrierOmin      | $\mathrm{TB}_O$, $\mathrm{Rep}_K$                | $K_{\mathrm{TB}_\pi}^\pm$                  | $K_{\mathrm{TB}_O}^{\mathrm{blk}}$ / $K_{\mathrm{TB}_O}^{\mathrm{br}}$                                | $S \in \mathcal{O}\text{-min}$                                 | Is the topology definable in an o-minimal structure?         | O-Minimal Taming           |
| 10   | BarrierMix       | $\mathrm{TB}_\rho$, $D_E$                        | $K_{\mathrm{TB}_O}^\pm$                    | $K_{\mathrm{TB}_\rho}^{\mathrm{blk}}$ / $K_{\mathrm{TB}_\rho}^{\mathrm{br}}$                          | $\tau_{\text{mix}} < \infty$                                   | Does the system mix fast enough to escape traps?             | Ergodic Mixing             |
| 11   | BarrierEpi       | $\mathrm{Rep}_K$, $\mathrm{Cap}_H$               | $K_{\mathrm{TB}_\rho}^\pm$                 | $K_{\mathrm{Rep}_K}^{\mathrm{blk}}$ / $K_{\mathrm{Rep}_K}^{\mathrm{br}}$                              | $\sup_\epsilon K_\epsilon(x) \leq S_{\text{BH}}$               | Is approximable complexity within holographic bounds?        | Epistemic Horizon          |
| 12   | BarrierFreq      | $\mathrm{GC}_\nabla$, $\mathrm{SC}_\lambda$      | $K_{\mathrm{Rep}_K}^\pm$                   | $K_{\mathrm{GC}_\nabla}^{\mathrm{blk}}$ / $K_{\mathrm{GC}_\nabla}^{\mathrm{br}}$                      | $\int \omega^2 S(\omega) \,d\omega < \infty$                     | Is the total oscillation energy finite?                      | Frequency Barrier          |
| 14   | BarrierBode      | $\mathrm{Bound}_B$, $\mathrm{LS}_\sigma$         | $K_{\mathrm{Bound}_\partial}^+$            | $K_{\mathrm{Bound}_B}^{\mathrm{blk}}$ / $K_{\mathrm{Bound}_B}^{\mathrm{br}}$                          | $\int_0^\infty \ln \|S(i\omega)\| \,d\omega > -\infty$ | Is the sensitivity integral conserved (waterbed effect)?     | Bode Sensitivity           |
| 15   | BarrierInput     | $\mathrm{Bound}_{\Sigma}$, $C_\mu$                   | $K_{\mathrm{Bound}_B}^\pm$                 | $K_{\mathrm{Bound}_{\Sigma}}^{\mathrm{blk}}$ / $K_{\mathrm{Bound}_{\Sigma}}^{\mathrm{br}}$                    | $r_{\text{reserve}} > 0$                                       | Is there a reservoir to prevent starvation?                  | Input Stability            |
| 16   | BarrierVariety   | $\mathrm{GC}_T$, $\mathrm{Cap}_H$                | $K_{\mathrm{Bound}_{\Sigma}}^\pm$              | $K_{\mathrm{GC}_T}^{\mathrm{blk}}$ / $K_{\mathrm{GC}_T}^{\mathrm{br}}$                                | $H(u) \geq H(d)$                                               | Does control entropy match disturbance entropy?              | Requisite Variety          |
| 17   | BarrierExclusion | $\mathrm{Cat}_{\mathrm{Hom}}$                    | Full $\Gamma$                              | $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$ / $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{morph}}$ / $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{hor}}$ | $\mathrm{Hom}(\mathcal{B}, S) = \emptyset$                     | Is there a categorical obstruction to the bad pattern?       | Morphism Exclusion / Reconstruction |

### Surgery Registry: Cobordism Morphisms

:::{div} feynman-prose

Now we arrive at the active repair mechanisms---the surgeries.

When both the primary gate and the barrier fail, we have a genuine obstruction. But sometimes we can *fix* the obstruction by modifying the system itself. This is surgery: a controlled modification that removes the pathological feature and allows us to re-enter the Sieve at an earlier point.

The mathematics behind this is *cobordism theory*. Think of it this way: if your state space has a problematic region, surgery cuts it out and glues in something better. If your singularity is approaching a point, surgery might "cap" that point with a smooth patch. If your flow is accumulating too many discrete events, surgery might coarsen the event structure to a manageable level.

Each surgery has an *admissibility condition*---you cannot just arbitrarily modify the system. The modification must be compatible with the structure we are trying to preserve. For instance, the "Ghost/Cap Extension" surgery (S1) requires that the blow-up be *conformal* and that the boundary at infinity be *definable*. Without these conditions, the surgery would not be mathematically valid.

The re-entry arrows (dotted lines in the diagram) show where the system resumes after surgery. This is crucial: a successful surgery does not just fix the immediate problem, it puts the system back into a state where the rest of the Sieve can run.

:::

The following table defines the **Cobordism Morphisms**---categorical pushouts that modify the topology of the state space to cancel non-trivial obstruction classes. Each surgery constructs a new manifold where the obstruction vanishes, enabling re-entry into the resolution tower.

| #   | Surgery ID   | Interfaces                                       | Input Certificate                            | Output Certificate                        | Admissibility Predicate                                                                      | Action                    | Metatheorem             |
|-----|--------------|--------------------------------------------------|----------------------------------------------|-------------------------------------------|----------------------------------------------------------------------------------------------|---------------------------|-------------------------|
| S1  | SurgCE       | $D_E$, $\mathrm{Cap}_H$                          | $K_{D_E}^{\mathrm{br}}$                      | $K_{\mathrm{SurgCE}}^{\mathrm{re}}$       | Growth conformal $\land$ $\partial_\infty X$ definable                                       | Ghost/Cap Extension       | Compactification        |
| S2  | SurgCC       | $\mathrm{Rec}_N$, $\mathrm{TB}_\pi$              | $K_{\mathrm{Rec}_N}^{\mathrm{br}}$           | $K_{\mathrm{SurgCC}}^{\mathrm{re}}$       | $\exists N_{\max}$: events $\leq N_{\max}$                                                   | Discrete Saturation       | Event Coarsening        |
| S3  | SurgCD\_Alt  | $C_\mu$, $D_E$                                   | $K_{C_\mu}^{\mathrm{path}}$                  | $K_{\mathrm{SurgCD\_Alt}}^{\mathrm{re}}$  | $V \in \mathcal{L}_{\text{soliton}} \land \|V\|_{H^1} < \infty$                              | Concentration-Compactness | Profile Extraction      |
| S4  | SurgSE       | $\mathrm{SC}_\lambda$, $D_E$                     | $K_{\mathrm{SC}_\lambda}^{\mathrm{br}}$      | $K_{\mathrm{SurgSE}}^{\mathrm{re}}$       | $\alpha - \beta < \varepsilon_{\text{crit}} \land V$ smooth                                  | Regularity Lift           | Perturbative Upgrade    |
| S5  | SurgSC       | $\mathrm{SC}_{\partial c}$, $\mathrm{LS}_\sigma$ | $K_{\mathrm{SC}_{\partial c}}^{\mathrm{br}}$ | $K_{\mathrm{SurgSC}}^{\mathrm{re}}$       | $\|\partial_t \theta\| < C_{\text{adm}} \land \theta \in \Theta_{\text{stable}}$             | Convex Integration        | Parameter Freeze        |
| S6  | SurgCD       | $\mathrm{Cap}_H$, $\mathrm{LS}_\sigma$           | $K_{\mathrm{Cap}_H}^{\mathrm{br}}$           | $K_{\mathrm{SurgCD}}^{\mathrm{re}}$       | $\operatorname{Cap}_H(\Sigma) \leq \varepsilon_{\text{adm}} \land V \in \mathcal{L}_{\text{neck}}$ | Auxiliary/Structural      | Excision-Capping        |
| S7  | SurgSD       | $\mathrm{LS}_{\partial^2 V}$, $\mathrm{GC}_\nabla$ | $K_{\mathrm{LS}_{\partial^2 V}}^{-}$        | $K_{\mathrm{SurgSD}}^{\mathrm{re}}$       | $\dim(\ker(H_V)) < \infty \land V$ isolated                                                  | Ghost Extension           | Spectral Lift           |
| S8  | SurgSC\_Rest | $\mathrm{SC}_{\partial c}$, $\mathrm{LS}_\sigma$ | $K_{\mathrm{SC}_{\partial c}}^{-}$          | $K_{\mathrm{SurgSC\_Rest}}^{\mathrm{re}}$ | $\Delta V > k_B T \land \Gamma < \Gamma_{\text{crit}}$                                       | Auxiliary Extension       | Vacuum Shift            |
| S9  | SurgTE\_Rest | $\mathrm{TB}_S$, $C_\mu$                         | $K_{\mathrm{TB}_S}^{-}$                     | $K_{\mathrm{SurgTE\_Rest}}^{\mathrm{re}}$ | $V \cong S^{n-1} \times I \land S_R[\gamma] < \infty$ (renormalized)                         | Structural                | Instanton Reconnection  |
| S10 | SurgTE       | $\mathrm{TB}_\pi$, $C_\mu$                       | $K_{\mathrm{TB}_\pi}^{\mathrm{br}}$          | $K_{\mathrm{SurgTE}}^{\mathrm{re}}$       | $V \cong S^{n-1} \times \mathbb{R}$ (Neck)                                                   | Tunnel                    | Topological Surgery     |
| S11 | SurgTC       | $\mathrm{TB}_O$, $\mathrm{Rep}_K$                | $K_{\mathrm{TB}_O}^{\mathrm{br}}$            | $K_{\mathrm{SurgTC}}^{\mathrm{re}}$       | $\Sigma \in \mathcal{O}_{\text{ext}}$-definable $\land \dim(\Sigma) < n$                     | O-minimal Regularization  | Structure Extension     |
| S12 | SurgTD       | $\mathrm{TB}_\rho$, $D_E$                        | $K_{\mathrm{TB}_\rho}^{\mathrm{br}}$         | $K_{\mathrm{SurgTD}}^{\mathrm{re}}$       | Trap isolated $\land \partial T$ has positive measure                                        | Mixing Enhancement        | Stochastic Perturbation |
| S13 | SurgDC       | $\mathrm{Rep}_K$, $\mathrm{Cap}_H$               | $K_{\mathrm{Rep}_K}^{\mathrm{br}}$           | $K_{\mathrm{SurgDC}}^{\mathrm{re}}$       | $K(x) \leq S_{\text{BH}} + \varepsilon \land x \in W^{1,\infty}$                             | Viscosity Solution        | Mollification           |
| S14 | SurgDE       | $\mathrm{GC}_\nabla$, $\mathrm{SC}_\lambda$      | $K_{\mathrm{GC}_\nabla}^{\mathrm{br}}$       | $K_{\mathrm{SurgDE}}^{\mathrm{re}}$       | $\exists\Lambda: \int_{\lvert\omega\rvert\leq\Lambda} \omega^2 S \,d\omega < \infty \land$ uniform ellipticity | De Giorgi-Nash-Moser      | Holder Regularization   |
| S15 | SurgBE       | $\mathrm{Bound}_B$, $\mathrm{LS}_\sigma$         | $K_{\mathrm{Bound}_B}^{\mathrm{br}}$         | $K_{\mathrm{SurgBE}}^{\mathrm{re}}$       | $\|S(i\omega)\|_\infty < M \land$ phase margin $> 0$                                         | Saturation                | Gain Limiting           |
| S16 | SurgBD       | $\mathrm{Bound}_{\Sigma}$, $C_\mu$                   | $K_{\mathrm{Bound}_{\Sigma}}^{\mathrm{br}}$      | $K_{\mathrm{SurgBD}}^{\mathrm{re}}$       | $r_{\text{reserve}} > 0 \land$ recharge $>$ drain                                            | Reservoir                 | Buffer Addition         |
| S17 | SurgBC       | $\mathrm{GC}_T$, $\mathrm{Cap}_H$                | $K_{\mathrm{GC}_T}^{\mathrm{br}}$            | $K_{\mathrm{SurgBC}}^{\mathrm{re}}$       | $H(u) < H(d) - \varepsilon \land \exists u': H(u') \geq H(d)$                                | Controller Augmentation   | Entropy Matching        |

:::{note} Restoration vs. Barrier Surgeries
Surgeries S7–S9 (SurgSD, SurgSC\_Rest, SurgTE\_Rest) are **restoration surgeries** triggered by gate failures ($K^-$) within the Stiffness Restoration Subtree (Nodes 7a–7d), *not* by barrier breaches ($K^{\mathrm{br}}$). This distinction is critical: barrier surgeries repair systems that have *breached* a constraint, while restoration surgeries repair systems that have *failed* a sub-gate within a recovery path.
:::

(sec-surgery-admissibility-registry)=
## Surgery Admissibility Registry

:::{div} feynman-prose

Now here is the final piece of the puzzle: how do we know when surgery is *possible*?

Not every broken system can be fixed. Some singularities are too wild, some topologies too tangled, some complexities too high. Before we attempt surgery, we need to check whether the repair is even mathematically admissible.

The admissibility checks (the light purple diamonds in the diagram, labeled A1 through A17) are the gatekeepers for surgery. They examine the specific nature of the obstruction and determine whether it falls into a class we know how to handle.

For example, consider the "Neck?" check (A6). This asks whether the singularity has the shape of a cylinder $S^{n-1} \times \mathbb{R}$---a "neck" pinching off. If yes, we can perform neck-pinch surgery, cutting through the neck and capping both ends. If no, the singularity is "fat" and we cannot perform this surgery; we terminate at Mode C.D (Geometric Collapse).

Each admissibility check produces a certificate that either authorizes surgery (YES) or forbids it (NO). The NO certificate must carry a *witness*---concrete data proving that the singularity is too pathological for repair. This ensures we never claim repair is impossible without evidence.

:::

The following table defines all **admissibility checks** in the Structural Sieve. Each admissibility check node (A1--A17) evaluates whether a breached barrier admits surgical repair or requires termination at a failure mode.

### Admissibility Node Logic

Each admissibility node $A_i$ implements the following evaluation:

$$
\mathrm{eval}_{A_i}(K^{\mathrm{br}}_{\mathrm{Barrier}}, \Sigma, V) \rightarrow \{K^+_{A_i}, K^-_{A_i}\}

$$

**Inputs:**
- $K^{\mathrm{br}}_{\mathrm{Barrier}}$: Breach certificate from the upstream barrier
- $\Sigma$: Singular set or defect locus
- $V$: Blow-up profile or local structure

**Outputs:**
- **YES** ($K^+_{A_i}$): Admissibility Certificate — required token to proceed to Surgery
- **NO** ($K^-_{A_i}$): Inadmissibility Certificate — required token to terminate at Failure Mode

### Admissibility Token Schema

**YES Certificate** ($K^+_{\mathrm{Adm}}$) contains:
- $V_{\mathrm{can}}$: Identification of singularity profile in the Canonical Library $\mathcal{L}$
- $\pi_{\mathrm{match}}$: Isomorphism/diffeomorphism witnessing that current state matches library cap
- $\mathrm{CapBound}$: Proof that $\operatorname{Cap}(\Sigma) \leq \varepsilon_{\mathrm{crit}}$

**NO Certificate** ($K^-_{\mathrm{Adm}}$) contains:
- $\mathrm{ObstructionType} \in \{\texttt{WildProfile}, \texttt{FatSingularity}, \texttt{Horizon}\}$
- $\mathrm{Witness}$: Data proving the obstruction (e.g., accumulation point, unbounded kernel, non-definable set)

### Admissibility Registry Table

| Node | ID | Question | YES Certificate ($K^+$) | NO Certificate ($K^-$) | YES Target | NO Target |
|------|-----|----------|------------------------|------------------------|------------|-----------|
| A1 | $\mathrm{Adm}_{\mathrm{CE}}$ | Conformal? | $K^+_{\mathrm{Conf}}$: conformal factor $\Omega(x)$, definable $\partial_\infty X$ | $K^-_{\mathrm{Conf}}$: anisotropic blow-up witness | S1 | Mode C.E |
| A2 | $\mathrm{Adm}_{\mathrm{CC}}$ | Discrete? | $K^+_{\mathrm{Disc}}$: bound $N_{\max}$ on event density | $K^-_{\mathrm{Disc}}$: accumulation point $t^*$ | S2 | Mode C.C |
| A3 | $\mathrm{Adm}_{\mathrm{CDA}}$ | Soliton? | $K^+_{\mathrm{Prof}}$: $V \in \mathcal{L}_{\text{soliton}}$, finite $H^1$ norm | $K^-_{\mathrm{Prof}}$: diffusive/undefined profile | S3 | Mode C.D |
| A4 | $\mathrm{Adm}_{\mathrm{SE}}$ | Smooth? | $K^+_{\mathrm{Lift}}$: regularity gap $\alpha - \beta < \varepsilon$, $V$ smooth | $K^-_{\mathrm{Lift}}$: gap too large, profile rough | S4 | Mode S.E |
| A5 | $\mathrm{Adm}_{\mathrm{SC}}$ | Stable $\theta$? | $K^+_{\mathrm{Stab}}$: $\|\dot{\theta}\| < \delta$, $\theta$ in stable basin | $K^-_{\mathrm{Stab}}$: velocity unbounded, chaotic | S5 | Mode S.C |
| A6 | $\mathrm{Adm}_{\mathrm{CD}}$ | Neck? | $K^+_{\mathrm{Neck}}$: $V \cong S^{n-1} \times \mathbb{R}$, $\operatorname{Cap}(\Sigma) \leq \varepsilon$ | $K^-_{\mathrm{Neck}}$: fat singularity, non-cylindrical | S6 | Mode C.D |
| A7 | $\mathrm{Adm}_{\mathrm{SD}}$ | Isolated? | $K^+_{\mathrm{Iso}}$: $\dim(\ker H) < \infty$, isolated critical pt | $K^-_{\mathrm{Iso}}$: infinite kernel, continuum of vacua | S7 | Mode S.D |
| A8 | $\mathrm{Adm}_{\mathrm{SCR}}$ | Slow Tunnel? | $K^+_{\mathrm{Vac}}$: gap $\Delta V > k_B T$, $\Gamma < \Gamma_{\mathrm{crit}}$ | $K^-_{\mathrm{Vac}}$: barrier collapse, thermal runaway | S8 | Mode S.C |
| A9 | $\mathrm{Adm}_{\mathrm{TER}}$ | Renormalizable? | $K^+_{\mathrm{Inst}}$: $S_R[\gamma] < \infty$ after cutoff regularization | $K^-_{\mathrm{Inst}}$: non-renormalizable divergence | S9 | Mode T.E |
| A10 | $\mathrm{Adm}_{\mathrm{TE}}$ | Neck Pinch? | $K^+_{\mathrm{Topo}}$: $V \cong \text{Neck}$, $\pi_1$ compatible | $K^-_{\mathrm{Topo}}$: exotic topology, knotting | S10 | Mode T.E |
| A11 | $\mathrm{Adm}_{\mathrm{TC}}$ | Definable? | $K^+_{\mathrm{Omin}}$: $\Sigma$ in $\mathcal{O}_{\text{ext}}$-definable | $K^-_{\mathrm{Omin}}$: wild set (Cantor) | S11 | Mode T.C |
| A12 | $\mathrm{Adm}_{\mathrm{TD}}$ | Escapable? | $K^+_{\mathrm{Mix}}$: $\partial T$ has positive measure | $K^-_{\mathrm{Mix}}$: hermetic seal, infinite depth | S12 | Mode T.D |
| A13 | $\mathrm{Adm}_{\mathrm{DC}}$ | Lipschitz? | $K^+_{\mathrm{Lip}}$: $x \in W^{1,\infty}$, $K \approx S_{\mathrm{BH}}$ | $K^-_{\mathrm{Lip}}$: $K \gg S_{\mathrm{BH}}$, fractal state | S13 | Mode D.C |
| A14 | $\mathrm{Adm}_{\mathrm{DE}}$ | Elliptic? | $K^+_{\mathrm{Ell}}$: marginal divergence, elliptic | $K^-_{\mathrm{Ell}}$: hyperbolic/chaotic oscillation | S14 | Mode D.E |
| A15 | $\mathrm{Adm}_{\mathrm{BE}}$ | Phase Margin? | $K^+_{\mathrm{Marg}}$: phase margin $> 0$ | $K^-_{\mathrm{Marg}}$: zero phase margin | S15 | Mode B.E |
| A16 | $\mathrm{Adm}_{\mathrm{BD}}$ | Recharge? | $K^+_{\mathrm{Res}}$: recharge $>$ drain, $r > 0$ | $K^-_{\mathrm{Res}}$: systemic deficit | S16 | Mode B.D |
| A17 | $\mathrm{Adm}_{\mathrm{BC}}$ | Bridgeable? | $K^+_{\mathrm{Ent}}$: $\exists u^*$ matching entropy | $K^-_{\mathrm{Ent}}$: fundamental variety gap | S17 | Mode B.C |

:::{prf:remark} Proof Chain Completion
:label: rem-adm-chain

The admissibility registry completes the certificate chain for surgical repair:

1. **Barrier** issues breach certificate $K^{\mathrm{br}}$
2. **Admissibility Check** consumes $K^{\mathrm{br}}$ and issues either $K^+_{\mathrm{Adm}}$ or $K^-_{\mathrm{Adm}}$
3. **Surgery** accepts only $K^+_{\mathrm{Adm}}$ as input token, produces re-entry certificate $K^{\mathrm{re}}$
4. **Failure Mode** accepts only $K^-_{\mathrm{Adm}}$ as input token, terminates run with classification

This ensures that no surgery executes without verified admissibility, and no failure mode activates without witnessed obstruction.

:::
