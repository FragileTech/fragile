(sec-surgery-nodes)=
# Surgery Nodes

(sec-surgery-node-specs)=
## Surgery Node Specifications (Purple Nodes)

:::{div} feynman-prose
Now we come to what I think is the most interesting part of the Sieve architecture. The gate nodes ask questions, the barrier nodes stand guard, but the surgery nodes---they actually *fix things*. When something goes wrong, when a barrier is breached and singularity threatens, these purple nodes perform the delicate operations that put the system back on track.

Think of it this way. A pilot flying through a storm hits turbulence that exceeds the autopilot's design limits. The autopilot (gate nodes) detects the problem. The warning system (barrier nodes) confirms this is serious. But then what? The surgery nodes are the emergency procedures---the controlled maneuvers that bring the aircraft back into a regime where normal flight can resume.

The key insight here is that each surgery is not just "fixing" something arbitrary. Each one corresponds to a *specific* type of mathematical pathology, and each one has a *proven* re-entry point where normal operation resumes. This is engineering at its most rigorous: failure modes are anticipated, and recovery procedures are guaranteed to work.
:::

Each surgery is specified by:
- **Inputs**: Breach certificate + surgery data
- **Action**: Abstract operation performed
- **Postcondition**: Re-entry certificate + target node
- **Progress measure**: Ensures termination

:::{prf:theorem} Non-circularity rule
:label: thm-non-circularity

A barrier invoked because predicate $P_i$ failed **cannot** assume $P_i$ as a prerequisite. Formally:

$$
\operatorname{Trigger}(B) = \operatorname{Gate}_i\,\text{NO} \Rightarrow P_i \notin \mathrm{Pre}(B)

$$

**Scope of Non-Circularity:** This syntactic check ($K_i^- \notin \Gamma$) prevents direct circular dependencies. Semantic circularity (proof implicitly using an equivalent of the target conclusion) is addressed by the derivation-dependency constraint: certificate proofs must cite only lemmas of lower rank in the proof DAG. The ranking is induced by the topological sort of the Sieve, ensuring well-foundedness ({cite}`VanGelder91`).

**Literature:** Well-founded semantics {cite}`VanGelder91`; stratification in logic programming {cite}`AptBolPedreschi94`.

:::

:::{div} feynman-prose
Why is non-circularity so important? Here is the thing to keep in your mind: imagine you are trying to prove that a bridge will not collapse. You cannot assume in your proof that the bridge does not collapse---that would be circular. Similarly, if a surgery is triggered because some property $P_i$ failed, that surgery cannot rely on $P_i$ being true. It must work *despite* the failure, using only properties that are still known to hold.

This is not just logical hygiene. It is what guarantees termination. Without non-circularity, a surgery could trigger another surgery that triggers the first one again, creating an infinite loop of "repairs" that never actually fix anything. The ranking induced by the proof DAG ensures this cannot happen.
:::

---

### Surgery Contracts Table

| **Surgery**  | **Input Mode**              | **Action**                | **Target**       |
|--------------|-----------------------------|---------------------------|------------------|
| SurgCE       | C.E (Energy Blow-up)        | Ghost/Cap extension       | ZenoCheck        |
| SurgCC       | C.C (Event Accumulation)    | Discrete saturation       | CompactCheck     |
| SurgCD\_Alt  | C.D (via Escape)            | Concentration-compactness | Profile          |
| SurgSE       | S.E (Supercritical)         | Regularity lift           | ParamCheck       |
| SurgSC       | S.C (Parameter Instability) | Convex integration        | GeomCheck        |
| SurgCD       | C.D (Geometric Collapse)    | Auxiliary/Structural      | StiffnessCheck   |
| SurgSD       | S.D (Stiffness Breakdown)   | Ghost extension           | TopoCheck        |
| SurgSC\_Rest | S.C (Vacuum Decay)          | Auxiliary extension       | TopoCheck        |
| SurgTE\_Rest | T.E (Metastasis)            | Structural                | TameCheck        |
| SurgTE       | T.E (Topological Twist)     | Tunnel                    | TameCheck        |
| SurgTC       | T.C (Labyrinthine)          | O-minimal regularization  | ErgoCheck        |
| SurgTD       | T.D (Glassy Freeze)         | Mixing enhancement        | ComplexCheck     |
| SurgDC       | D.C (Semantic Horizon)      | Viscosity solution        | OscillateCheck   |
| SurgDE       | D.E (Oscillatory)           | De Giorgi-Nash-Moser      | BoundaryCheck    |
| SurgBE       | B.E (Injection)             | Saturation                | StarveCheck      |
| SurgBD       | B.D (Starvation)            | Reservoir                 | AlignCheck       |
| SurgBC       | B.C (Misalignment)          | Controller Augmentation   | BarrierExclusion |

:::{div} feynman-prose
Look at this table carefully. Each row tells you a complete story: something went wrong (the Input Mode), here is what we do about it (the Action), and here is where we rejoin the normal flow (the Target). Notice that the targets are always *later* nodes in the Sieve---never earlier ones. This is the progress guarantee in action.

The actions themselves read like a catalog of mathematical repair techniques, each one carefully matched to its failure mode. Energy blowing up? Compactify the space. Events accumulating infinitely fast? Saturate the discrete structure. Topology getting too complicated? Use o-minimal regularization. Each is a proven technique from analysis, geometry, or control theory, repurposed here as a systematic repair mechanism.
:::

---

### Surgery Contract Template

:::{div} feynman-prose
Before we dive into the individual surgeries, let me show you the template they all follow. This is important because it reveals the *structure* that makes the whole system work. Every surgery has the same skeleton: what it takes in, what conditions must hold for it to be safe, what transformation it performs, and what it guarantees afterward.

The key innovation here is the *Admissibility Predicate*---what I like to call "the Diamond" because it represents the narrow conditions under which surgery is possible. If you are outside the diamond, the surgery cannot help you. But if you are inside, the surgery is guaranteed to work. This is not just a safety check; it is a *design constraint* that ensures each surgery is used only where it is mathematically valid.
:::

:::{prf:definition} Surgery Specification Schema
:label: def-surgery-schema

A **Surgery Specification** is a transformation of the Hypostructure $\mathcal{H} \to \mathcal{H}'$. Each surgery defines:

**Surgery ID:** `[SurgeryID]` (e.g., SurgCE)
**Target Mode:** `[ModeID]` (e.g., Mode C.E)

**Interface Dependencies:**
- **Primary:** `[InterfaceID_1]` (provides the singular object/profile $V$ and locus $\Sigma$)
- **Secondary:** `[InterfaceID_2]` (provides the canonical library $\mathcal{L}_T$ or capacity bounds)

**Admissibility Signature:**
- **Input Certificate:** $K_{[\text{ModeID}]}^{\mathrm{br}}$ (The breach witnessing the singularity)
- **Admissibility Predicate (The Diamond):**
  $V \in \mathcal{L}_T \land \operatorname{Cap}(\Sigma) \le \varepsilon_{\text{adm}}$
  *(Conditions required to perform surgery safely, corresponding to Case 1 of the Trichotomy.)*

**Transformation Law ($\mathcal{O}_S$):**
- **State Space:** $X' = (X \setminus \Sigma_\varepsilon) \cup_{\partial} X_{\text{cap}}$
- **Height Jump:** $\Phi(x') \le \Phi(x) - \delta_S$
- **Topology:** $\tau(x') = [\text{New Sector}]$

**Postcondition:**
- **Re-entry Certificate:** $K_{[\text{SurgeryID}]}^{\mathrm{re}}$
- **Re-entry Target:** `[TargetNodeName]`
- **Progress Guarantee:** `[Type A (Count) or Type B (Complexity)]`

**Required Progress Certificate ($K_{\mathrm{prog}}$):**
Every surgery must produce a progress certificate witnessing either:
- **Type A (Bounded Resource):** $\Delta R \leq C$ per surgery invocation (bounded consumption)
- **Type B (Well-Founded Decrease):** $\mu(x') < \mu(x)$ for some ordinal-valued measure $\mu$

The non-circularity checker must verify that the progress measure is compatible with the surgery's re-entry target, ensuring termination of the repair loop.
:::

:::{div} feynman-prose
The progress certificate is crucial. Notice that there are two types: Type A says "we can only do this a bounded number of times" (like counting surgeries), while Type B says "some measure strictly decreases each time" (like energy or complexity). Either way, the surgeries cannot go on forever. This is not an afterthought---it is baked into the very definition of what a valid surgery is.
:::

---

### Surgery Specifications

:::{div} feynman-prose
Now let us walk through the individual surgeries. I will not bore you with every detail, but I want you to see the pattern: each surgery is designed for a *specific* mathematical pathology, uses a *specific* classical technique, and produces a *specific* certificate that lets normal operation resume. This is the payoff of all that abstract machinery---concrete, verifiable repair procedures.
:::

:::{prf:definition} Surgery Specification: Lyapunov Cap
:label: def-surgery-ce

**Surgery ID:** `SurgCE`
**Target Mode:** `Mode C.E` (Energy Blow-up)

**Interface Dependencies:**
- **Primary:** $D_E$ (Energy Interface: provides the unbounded potential $\Phi$)
- **Secondary:** $\mathrm{Cap}_H$ (Capacity Interface: provides the compactification metric)

**Admissibility Signature:**
- **Input Certificate:** $K_{D_E}^{\mathrm{br}}$ (Energy unbounded)
- **Admissibility Predicate:**
  $\operatorname{Growth}(\Phi) \text{ is conformal} \land \partial_\infty X \text{ is definable}$
  *(The blow-up must allow conformal compactification.)*

**Transformation Law ($\mathcal{O}_S$):**
- **State Space:** $\hat{X} = X \cup \partial_\infty X$ (One-point or boundary compactification)
- **Height Rescaling:** $\hat{\Phi} = \tanh(\Phi)$ (Maps $[0, \infty) \to [0, 1)$)
- **Boundary Condition:** $\hat{S}_t |_{\partial_\infty X} = \text{Absorbing}$

**Postcondition:**
- **Re-entry Certificate:** $K_{\mathrm{SurgCE}}^{\mathrm{re}}$ (Witnesses $\hat{\Phi}$ is bounded)
- **Re-entry Target:** `ZenoCheck` ({prf:ref}`def-node-zeno`)
- **Progress Guarantee:** **Type A**. The system enters a bounded domain; blow-up is geometrically impossible in $\hat{X}$.

**Literature:** Compactification and boundary conditions {cite}`Dafermos16`; energy methods {cite}`Leray34`.

:::

:::{div} feynman-prose
The Lyapunov Cap surgery (SurgCE) is beautiful in its simplicity. When energy wants to blow up to infinity, what do you do? You change the rules of the game. Instead of working in an unbounded space, you compactify it---you add a "point at infinity" and rescale everything so that what was infinite becomes finite. The $\tanh$ function is perfect for this: it squashes the entire interval $[0, \infty)$ into $[0, 1)$. Now blow-up is geometrically impossible. The system might try to run to infinity, but infinity is no longer infinitely far away.
:::

:::{prf:definition} Surgery Specification: Discrete Saturation
:label: def-surgery-cc

**Surgery ID:** `SurgCC`
**Target Mode:** `Mode C.C` (Event Accumulation)

**Interface Dependencies:**
- **Primary:** $\mathrm{Rec}_N$ (Recovery Interface: provides event count $N$)
- **Secondary:** $\mathrm{TB}_\pi$ (Topology Interface: provides sector structure)

**Admissibility Signature:**
- **Input Certificate:** $K_{\mathrm{Rec}_N}^{\mathrm{br}}$ (Zeno accumulation detected)
- **Admissibility Predicate:**
  $\exists N_{\max} : \#\{\text{events in } [t, t+\epsilon]\} \leq N_{\max} \text{ for small } \epsilon$
  *(Events must be locally finite, not truly Zeno.)*

**Transformation Law ($\mathcal{O}_S$):**
- **State Space:** $X' = X$ (no topological change)
- **Time Reparametrization:** $t' = \int_0^t \frac{ds}{1 + \#\text{events}(s)}$
- **Event Coarsening:** Merge events within $\epsilon$-windows

**Postcondition:**
- **Re-entry Certificate:** $K_{\mathrm{SurgCC}}^{\mathrm{re}}$ (Witnesses finite event rate)
- **Re-entry Target:** `CompactCheck` ({prf:ref}`def-node-compact`)
- **Progress Guarantee:** **Type A**. Event count bounded by $N(T, \Phi_0)$.

**Literature:** Surgery bounds in Ricci flow {cite}`Perelman03`; {cite}`KleinerLott08`.

:::

:::{div} feynman-prose
The Discrete Saturation surgery (SurgCC) handles Zeno-like behavior---when events accumulate infinitely fast. The trick here is time reparametrization. You slow down the clock as events pile up, so that in the new time, the events are spread out. It is like switching from "real time" to "event time." The integral formula shows exactly how: the more events per unit time, the slower the new clock ticks. And by merging events within small windows, you prevent artificial inflation of the event count.
:::

:::{prf:definition} Surgery Specification: Concentration-Compactness
:label: def-surgery-cd-alt

**Surgery ID:** `SurgCD_Alt`
**Target Mode:** `Mode C.D` (via Escape/Soliton)

**Interface Dependencies:**
- **Primary:** $C_\mu$ (Compactness Interface: provides escaping profile $V$)
- **Secondary:** $D_E$ (Energy Interface: provides energy tracking)

**Admissibility Signature:**
- **Input Certificate:** $K_{C_\mu}^{\mathrm{path}}$ (Soliton-like escape detected)
- **Admissibility Predicate:**
  $V \in \mathcal{L}_{\text{soliton}} \land \|V\|_{H^1} < \infty$
  *(Profile must be a recognizable traveling wave.)*

**Transformation Law ($\mathcal{O}_S$):**
- **State Space:** $X' = X / \sim_V$ (quotient by soliton orbit)
- **Energy Subtraction:** $\Phi(x') = \Phi(x) - E(V)$
- **Remainder:** Track $x - V$ in lower energy class

**Postcondition:**
- **Re-entry Certificate:** $K_{\mathrm{SurgCD\_Alt}}^{\mathrm{re}}$ (Witnesses profile extracted)
- **Re-entry Target:** `Profile` (Re-check for further concentration)
- **Progress Guarantee:** **Type B**. Energy strictly decreases: $\Phi(x') < \Phi(x)$.

**Literature:** Concentration-compactness principle {cite}`Lions84`; profile decomposition {cite}`KenigMerle06`.

:::

:::{div} feynman-prose
Concentration-Compactness (SurgCD_Alt) is one of the most elegant ideas in modern analysis. When a soliton or traveling wave tries to escape to infinity, you do not chase it---you *factor it out*. You recognize the escaping profile (it must be from a known library of soliton shapes), subtract its energy from the total, and track what remains. Each time you do this, energy strictly decreases. Eventually there is nothing left to extract, and you are done. This is Pierre-Louis Lions' deep insight, adapted here for systematic recovery.
:::

:::{prf:definition} Surgery Specification: Regularity Lift
:label: def-surgery-se

**Surgery ID:** `SurgSE`
**Target Mode:** `Mode S.E` (Supercritical Cascade)

**Interface Dependencies:**
- **Primary:** $\mathrm{SC}_\lambda$ (Scaling Interface: provides critical exponent)
- **Secondary:** $D_E$ (Energy Interface: provides energy bounds)

**Admissibility Signature:**
- **Input Certificate:** $K_{\mathrm{SC}_\lambda}^{\mathrm{br}}$ (Supercritical scaling detected)
- **Admissibility Predicate:**
  $0 < \beta - \alpha < \epsilon_{\text{crit}} \land \text{Profile } V \text{ is smooth}$
  *(Slightly supercritical with smooth profile allows perturbative lift.)*

**Transformation Law ($\mathcal{O}_S$):**
- **State Space:** $X' = X$ (same space, better regularity)
- **Regularity Upgrade:** Promote $x \in H^s$ to $x' \in H^{s+\delta}$
- **Height Adjustment:** $\Phi' = \Phi + \text{regularization penalty}$

**Postcondition:**
- **Re-entry Certificate:** $K_{\mathrm{SurgSE}}^{\mathrm{re}}$ (Witnesses improved regularity)
- **Re-entry Target:** `ParamCheck` ({prf:ref}`def-node-param`)
- **Progress Guarantee:** **Type B**. Regularity index strictly increases.

**Literature:** Regularity lift in critical problems {cite}`CaffarelliKohnNirenberg82`; bootstrap arguments {cite}`DeGiorgi57`.

:::

:::{div} feynman-prose
The Regularity Lift surgery (SurgSE) is the mathematical equivalent of "if you are almost there, you can get all the way there." When a system is near-critical on the supercritical side (the exponents $\alpha$ and $\beta$ are close, with $\beta$ only slightly larger), you can bootstrap to better regularity. You promote from $H^s$ to $H^{s+\delta}$---a small step in smoothness, but enough to escape the critical regime. This is the engine behind many regularity theorems in PDE.
:::

:::{prf:definition} Surgery Specification: Convex Integration
:label: def-surgery-sc

**Surgery ID:** `SurgSC`
**Target Mode:** `Mode S.C` (Parameter Instability)

**Interface Dependencies:**
- **Primary:** $\mathrm{SC}_{\partial c}$ (Parameter Interface: provides drifting constants)
- **Secondary:** $\mathrm{LS}_\sigma$ (Stiffness Interface: provides spectral data)

**Admissibility Signature:**
- **Input Certificate:** $K_{\mathrm{SC}_{\partial c}}^{\mathrm{br}}$ (Parameter drift detected)
- **Admissibility Predicate:**
  $\|\partial_t \theta\| < C_{\text{adm}} \land \theta \in \Theta_{\text{stable}}$
  *(Drift is slow and within stable region.)*

**Transformation Law ($\mathcal{O}_S$):**
- **State Space:** $X' = X \times \Theta'$ (extended parameter space)
- **Parameter Freeze:** $\theta' = \theta_{\text{avg}}$ (time-averaged parameter)
- **Convex Correction:** Add corrector field to absorb drift

**Postcondition:**
- **Re-entry Certificate:** $K_{\mathrm{SurgSC}}^{\mathrm{re}}$ (Witnesses stable parameters)
- **Re-entry Target:** `GeomCheck` ({prf:ref}`def-node-geom`)
- **Progress Guarantee:** **Type B**. Parameter variance strictly decreases.

**Literature:** Convex integration method {cite}`DeLellisSzekelyhidi09`; {cite}`Isett18`.

:::

:::{div} feynman-prose
Convex Integration (SurgSC) handles the subtle problem of drifting parameters. If your system's constants are slowly changing, you cannot pretend they are fixed. But you can *absorb* the drift by extending the parameter space and adding corrector fields. Think of it like a thermostat: instead of demanding constant temperature, you add a heating/cooling system that compensates for fluctuations. The parameter variance strictly decreases each time, so you converge to stability.
:::

:::{prf:definition} Surgery Specification: Auxiliary/Structural
:label: def-surgery-cd

**Surgery ID:** `SurgCD`
**Target Mode:** `Mode C.D` (Geometric Collapse)

**Interface Dependencies:**
- **Primary:** $\mathrm{Cap}_H$ (Capacity Interface: provides singular set measure)
- **Secondary:** $\mathrm{LS}_\sigma$ (Stiffness Interface: provides local geometry)

**Admissibility Signature:**
- **Input Certificate:** $K_{\mathrm{Cap}_H}^{\mathrm{br}}$ (Positive capacity singularity)
- **Admissibility Predicate:**
  $\operatorname{Cap}_H(\Sigma) \leq \varepsilon_{\text{adm}} \land V \in \mathcal{L}_{\text{neck}}$
  *(Small singular set with recognizable neck structure.)*

**Transformation Law ($\mathcal{O}_S$):**
- **Excision:** $X' = X \setminus B_\epsilon(\Sigma)$
- **Capping:** Glue auxiliary space $X_{\text{aux}}$ matching boundary
- **Height Drop:** $\Phi(x') \leq \Phi(x) - c \cdot \operatorname{Vol}(\Sigma)^{2/n}$

**Postcondition:**
- **Re-entry Certificate:** $K_{\mathrm{SurgCD}}^{\mathrm{re}}$ (Witnesses smooth excision)
- **Re-entry Target:** `StiffnessCheck` ({prf:ref}`def-node-stiffness`)
- **Progress Guarantee:** **Type B**. Singular set measure strictly decreases.

**Literature:** Ricci flow surgery {cite}`Hamilton97`; {cite}`Perelman03`; geometric measure theory {cite}`Federer69`.

:::

:::{div} feynman-prose
The Auxiliary/Structural surgery (SurgCD) is directly inspired by Perelman's work on Ricci flow. When geometry collapses at a singular set, you do not try to resolve the singularity in place---you cut it out and cap the wound. The excision removes a small neighborhood of the bad set, and you glue in an auxiliary space that matches smoothly at the boundary. The height drops by an amount proportional to the volume of what you removed, guaranteeing progress.
:::

:::{prf:definition} Surgery Specification: Ghost Extension
:label: def-surgery-sd

**Surgery ID:** `SurgSD`
**Target Mode:** `Mode S.D` (Stiffness Breakdown)

**Interface Dependencies:**
- **Primary:** $\mathrm{LS}_\sigma$ (Stiffness Interface: provides spectral gap data)
- **Secondary:** $\mathrm{GC}_\nabla$ (Gradient Interface: provides flow structure)

**Admissibility Signature:**
- **Input Certificate:** $K_{\mathrm{LS}_\sigma}^{\mathrm{br}}$ (Zero spectral gap at equilibrium)
- **Admissibility Predicate:**
  $\dim(\ker(H_V)) < \infty \land V \text{ is isolated}$
  *(Finite-dimensional kernel, isolated critical point.)*

**Transformation Law ($\mathcal{O}_S$):**
- **State Space:** $\hat{X} = X \times \mathbb{R}^k$ (ghost variables for null directions)
- **Extended Potential:** $\hat{\Phi}(x, \xi) = \Phi(x) + \frac{1}{2}|\xi|^2$
- **Artificial Gap:** New system has spectral gap $\lambda_1 > 0$

**Postcondition:**
- **Re-entry Certificate:** $K_{\mathrm{SurgSD}}^{\mathrm{re}}$ (Witnesses positive gap in extended system)
- **Re-entry Target:** `TopoCheck` ({prf:ref}`def-node-topo`)
- **Progress Guarantee:** **Type A**. Bounded surgeries per unit time.

**Literature:** Ghost variable methods {cite}`Simon83`; spectral theory {cite}`FeehanMaridakis19`.

:::

:::{div} feynman-prose
Ghost Extension (SurgSD) is a clever trick for handling degenerate critical points---places where the Hessian has zero eigenvalues. The problem is that zero eigenvalues mean no restoring force, so the system can drift forever without converging. The solution? Add "ghost" variables that artificially break the degeneracy. You extend the state space with extra dimensions and add a quadratic term that provides the missing stiffness. Now the extended system has a spectral gap, even though the original one did not.
:::

:::{prf:definition} Surgery Specification: Vacuum Auxiliary
:label: def-surgery-sc-rest

**Surgery ID:** `SurgSC_Rest`
**Target Mode:** `Mode S.C` (Vacuum Decay in Restoration)

**Interface Dependencies:**
- **Primary:** $\mathrm{SC}_{\partial c}$ (Parameter Interface: provides vacuum instability)
- **Secondary:** $\mathrm{LS}_\sigma$ (Stiffness Interface: provides mass gap)

**Admissibility Signature:**
- **Input Certificate:** $K_{\mathrm{SC}_{\partial c}}^{\mathrm{br}}$ (Vacuum decay detected)
- **Admissibility Predicate:**
  $\Delta V > k_B T \land \text{tunneling rate } \Gamma < \Gamma_{\text{crit}}$
  *(Mass gap exists and tunneling is slow.)*

**Transformation Law ($\mathcal{O}_S$):**
- **State Space:** $X' = X$ (same space)
- **Vacuum Shift:** $v_0 \to v_0'$ (new stable vacuum)
- **Energy Recentering:** $\Phi' = \Phi - \Phi(v_0') + \Phi(v_0)$

**Postcondition:**
- **Re-entry Certificate:** $K_{\mathrm{SurgSC\_Rest}}^{\mathrm{re}}$ (Witnesses new stable vacuum)
- **Re-entry Target:** `TopoCheck` ({prf:ref}`def-node-topo`)
- **Progress Guarantee:** **Type B**. Vacuum energy strictly decreases.

**Literature:** Vacuum stability {cite}`Coleman75`; symmetry breaking {cite}`Goldstone61`; {cite}`Higgs64`.

:::

:::{div} feynman-prose
Vacuum Auxiliary (SurgSC_Rest) handles the exotic case of vacuum decay---when the ground state of a system is not truly stable but can tunnel to a lower energy state. This is physics straight from Coleman's instanton calculus. If the mass gap is larger than the thermal energy and tunneling is slow enough, you can shift to the new stable vacuum and recenter the energy. Each shift strictly decreases vacuum energy, so the process terminates.
:::

:::{prf:definition} Surgery Specification: Structural (Metastasis)
:label: def-surgery-te-rest

**Surgery ID:** `SurgTE_Rest`
**Target Mode:** `Mode T.E` (Topological Metastasis in Restoration)

**Interface Dependencies:**
- **Primary:** $\mathrm{TB}_\pi$ (Topology Interface: provides sector invariants)
- **Secondary:** $C_\mu$ (Compactness Interface: provides profile structure)

**Admissibility Signature:**
- **Input Certificate:** $K_{\mathrm{TB}_\pi}^{\mathrm{br}}$ (Sector transition via decay)
- **Admissibility Predicate:**
  $V \cong S^{n-1} \times I \land \text{instanton action } S[\gamma] < \infty$
  *(Domain wall with finite tunneling action.)*

**Transformation Law ($\mathcal{O}_S$):**
- **Excision:** $X' = X \setminus (\text{domain wall})$
- **Reconnection:** Connect sectors via instanton path
- **Sector Update:** $\tau(x') = \tau_{\text{new}}$

**Postcondition:**
- **Re-entry Certificate:** $K_{\mathrm{SurgTE\_Rest}}^{\mathrm{re}}$ (Witnesses sector transition complete)
- **Re-entry Target:** `TameCheck` ({prf:ref}`def-node-tame`)
- **Progress Guarantee:** **Type B**. Topological complexity (Betti sum) strictly decreases.

**Literature:** Instanton tunneling {cite}`Coleman75`; topological field theory {cite}`Floer89`.

:::

:::{div} feynman-prose
Structural Metastasis (SurgTE_Rest) deals with domain walls---those thin regions where the system transitions between topological sectors. If the domain wall has finite instanton action, you can excise it and reconnect the sectors directly. Think of it like surgery on a soap bubble: you cut through the membrane separating two regions and let them merge. The Betti sum (a measure of topological complexity) strictly decreases each time.
:::

:::{prf:definition} Surgery Specification: Topological Tunneling
:label: def-surgery-te

**Surgery ID:** `SurgTE`
**Target Mode:** `Mode T.E` (Topological Twist)

**Interface Dependencies:**
- **Primary:** $\mathrm{TB}_\pi$ (Topology Interface: provides sector $\tau$ and invariants)
- **Secondary:** $C_\mu$ (Compactness Interface: provides the neck profile $V$)

**Admissibility Signature:**
- **Input Certificate:** $K_{\mathrm{TB}_\pi}^{\mathrm{br}}$ (Sector transition attempted)
- **Admissibility Predicate:**
  $V \cong S^{n-1} \times \mathbb{R}$ *(Canonical Neck)*
  *(The singularity must be a recognizable neck pinch or domain wall.)*

**Transformation Law ($\mathcal{O}_S$):**
- **Excision:** $X' = X \setminus (S^{n-1} \times (-\varepsilon, \varepsilon))$
- **Capping:** Glue two discs $D^n$ to the exposed boundaries.
- **Sector Change:** $\tau(x') = \tau(x) \pm 1$ (Change in Euler characteristic/Betti number).

**Postcondition:**
- **Re-entry Certificate:** $K_{\mathrm{SurgTE}}^{\mathrm{re}}$ (Witnesses new topology is manifold)
- **Re-entry Target:** `TameCheck` ({prf:ref}`def-node-tame`)
- **Progress Guarantee:** **Type B**. Topological complexity (e.g., volume or Betti sum) strictly decreases: $\mathcal{C}(X') < \mathcal{C}(X)$.

**Literature:** Topological surgery {cite}`Smale67`; {cite}`Conley78`; Ricci flow surgery {cite}`Perelman03`.

:::

:::{div} feynman-prose
Topological Tunneling (SurgTE) is the classic neck-pinch surgery from differential topology. When a manifold forms a neck (think of an hourglass about to pinch in the middle), you do not wait for it to pinch---you cut through the neck and cap both ends with disks. The manifold might split into two pieces, or its topology might change (Euler characteristic going up or down), but either way the complexity strictly decreases. This is exactly what Perelman did for Ricci flow, and it works here too.
:::

:::{prf:definition} Surgery Specification: O-Minimal Regularization
:label: def-surgery-tc

**Surgery ID:** `SurgTC`
**Target Mode:** `Mode T.C` (Labyrinthine/Wild Topology)

**Interface Dependencies:**
- **Primary:** $\mathrm{TB}_O$ (Tameness Interface: provides definability structure)
- **Secondary:** $\mathrm{Rep}_K$ (Dictionary Interface: provides complexity bounds)

**Admissibility Signature:**
- **Input Certificate:** $K_{\mathrm{TB}_O}^{\mathrm{br}}$ (Non-definable topology detected)
- **Admissibility Predicate:**
  $\Sigma \in \mathcal{O}_{\text{ext}}\text{-definable} \land \dim(\Sigma) < n$
  *(Wild set is definable in extended o-minimal structure.)*

**Transformation Law ($\mathcal{O}_S$):**
- **Structure Extension:** $\mathcal{O}' = \mathcal{O}[\exp]$ or $\mathcal{O}[\text{Pfaffian}]$
- **Stratification:** Replace $\Sigma$ with definable stratification
- **Tameness Certificate:** Produce o-minimal cell decomposition

**Postcondition:**
- **Re-entry Certificate:** $K_{\mathrm{SurgTC}}^{\mathrm{re}}$ (Witnesses tame stratification)
- **Re-entry Target:** `ErgoCheck` ({prf:ref}`def-node-ergo`)
- **Progress Guarantee:** **Type B**. Definability complexity strictly decreases.

**Literature:** O-minimal structures {cite}`vandenDries98`; {cite}`Wilkie96`; stratification theory {cite}`Lojasiewicz65`.

:::

:::{div} feynman-prose
O-Minimal Regularization (SurgTC) tames wild topology. Some sets are so pathological (fractals, cantor sets, worse) that no analysis is possible. But there is a clever escape: extend to a richer o-minimal structure (add exponentials, Pfaffian functions) where the wild set becomes definable. Once it is definable, it has a nice stratification into smooth pieces. The definability complexity strictly decreases because you have replaced something intractable with something structured.
:::

:::{prf:definition} Surgery Specification: Mixing Enhancement
:label: def-surgery-td

**Surgery ID:** `SurgTD`
**Target Mode:** `Mode T.D` (Glassy Freeze/Trapping)

**Interface Dependencies:**
- **Primary:** $\mathrm{TB}_\rho$ (Mixing Interface: provides mixing time)
- **Secondary:** $D_E$ (Energy Interface: provides energy landscape)

**Admissibility Signature:**
- **Input Certificate:** $K_{\mathrm{TB}_\rho}^{\mathrm{br}}$ (Infinite mixing time detected)
- **Admissibility Predicate:**
  $\text{Trap } T \text{ is isolated} \land \partial T \text{ has positive measure}$
  *(Trap has accessible boundary.)*

**Transformation Law ($\mathcal{O}_S$):**
- **State Space:** $X' = X$ (same space)
- **Dynamics Modification:** Add noise term $\sigma dW_t$ to escape trap
- **Mixing Acceleration:** $\tau'_{\text{mix}} = \tau_{\text{mix}} / (1 + \sigma^2)$

**Postcondition:**
- **Re-entry Certificate:** $K_{\mathrm{SurgTD}}^{\mathrm{re}}$ (Witnesses finite mixing time)
- **Re-entry Target:** `ComplexCheck` ({prf:ref}`def-node-complex`)
- **Progress Guarantee:** **Type A**. Bounded mixing enhancement per unit time.

**Complexity Type on Re-entry:** The re-entry evaluates $K(\mu_t)$ where $\mu_t = \operatorname{Law}(x_t)$ is the probability measure on trajectories, not $K(x_t(\omega))$ for individual sample paths. The SDE has finite description length (drift $b$, diffusion $\sigma$, initial law $\mu_0$) even though individual realizations are algorithmically incompressible (white noise is random). This ensures S12 does not cause immediate failure at Node 11.

**Literature:** Stochastic perturbation and mixing {cite}`MeynTweedie93`; {cite}`HairerMattingly11`.

:::

:::{div} feynman-prose
Mixing Enhancement (SurgTD) is the physical solution to glassy dynamics. When a system gets trapped and cannot escape (infinite mixing time), you add noise. The added stochastic term shakes the system out of local minima and speeds up exploration. The mixing time becomes finite because now the system can hop over barriers instead of waiting forever. The crucial point in the "Complexity Type" clarification is subtle but important: after this surgery, we track the probability distribution over trajectories, not individual trajectories. The noise is random, but the distribution evolves deterministically.
:::

:::{prf:definition} Surgery Specification: Viscosity Solution
:label: def-surgery-dc

**Surgery ID:** `SurgDC`
**Target Mode:** `Mode D.C` (Semantic Horizon/Complexity Explosion)

**Interface Dependencies:**
- **Primary:** $\mathrm{Rep}_K$ (Dictionary Interface: provides complexity measure)
- **Secondary:** $\mathrm{Cap}_H$ (Capacity Interface: provides dimension bounds)

**Admissibility Signature:**
- **Input Certificate:** $K_{\mathrm{Rep}_K}^{\mathrm{br}}$ (Complexity exceeds bound)
- **Admissibility Predicate:**
  $K_\epsilon(T_{\mathrm{thin}}) \leq S_{\text{BH}} + \epsilon \land T_{\mathrm{thin}} \in W^{1,\infty}$
  *(Near holographic bound with Lipschitz regularity.)*

**Transformation Law ($\mathcal{O}_S$):**
- **State Space:** $X' = X$ (same space, coarsened description)
- **Viscosity Regularization:** $x' = x * \phi_\epsilon$ (convolution smoothing)
- **Complexity Reduction:** $K_\epsilon(T_{\mathrm{thin}}') \leq K_\epsilon(T_{\mathrm{thin}}) - c \cdot \epsilon$

**Postcondition:**
- **Re-entry Certificate:** $K_{\mathrm{SurgDC}}^{\mathrm{re}}$ (Witnesses reduced complexity)
- **Re-entry Target:** `OscillateCheck` ({prf:ref}`def-node-oscillate`)
- **Progress Guarantee:** **Type B**. Kolmogorov complexity strictly decreases.

**Literature:** Viscosity solutions {cite}`CrandallLions83`; regularization and mollification {cite}`EvansGariepy15`.

:::

:::{div} feynman-prose
Viscosity Solution (SurgDC) handles complexity explosions---when the description length of a state exceeds what is physically meaningful. The fix is convolution smoothing: blur everything at scale $\epsilon$. This is like looking at a photograph from farther away; fine details vanish, but the important structure remains. Kolmogorov complexity strictly decreases because the smoothed version has fewer degrees of freedom. This is precisely what viscosity solutions do in PDE theory: they select the "physical" solution by adding a small amount of regularization.
:::

:::{prf:definition} Surgery Specification: De Giorgi-Nash-Moser
:label: def-surgery-de

**Surgery ID:** `SurgDE`
**Target Mode:** `Mode D.E` (Oscillatory Divergence)

**Interface Dependencies:**
- **Primary:** $\mathrm{GC}_\nabla$ (Gradient Interface: provides oscillation structure)
- **Secondary:** $\mathrm{SC}_\lambda$ (Scaling Interface: provides frequency bounds)

**Admissibility Signature:**
- **Input Certificate:** $K_{\mathrm{GC}_\nabla}^{\mathrm{br}}$ (Infinite oscillation energy)
- **Admissibility Predicate:**
  There exists a cutoff scale $\Lambda$ such that the truncated second moment is finite:

$$
\exists \Lambda < \infty:\, \sup_{\Lambda' \leq \Lambda} \int_{|\omega| \leq \Lambda'} \omega^2\, S(\omega)\, d\omega < \infty \quad \land \quad \text{uniform ellipticity}

$$

  *(Divergence is "elliptic-regularizable" — De Giorgi-Nash-Moser applies to truncated spectrum.)*

**Transformation Law ($\mathcal{O}_S$):**
- **State Space:** $X' = X$ (same space, improved regularity)
- **Hölder Regularization:** Apply De Giorgi-Nash-Moser iteration
- **Oscillation Damping:** $\operatorname{osc}_{B_r}(x') \leq C r^\alpha \operatorname{osc}_{B_1}(x)$

**Postcondition:**
- **Re-entry Certificate:** $K_{\mathrm{SurgDE}}^{\mathrm{re}}$ (Witnesses Hölder continuity)
- **Re-entry Target:** `BoundaryCheck` ({prf:ref}`def-node-boundary`)
- **Progress Guarantee:** **Type A**. Bounded regularity improvements per unit time.

**Literature:** De Giorgi's original regularity theorem {cite}`DeGiorgi57`; Nash's parabolic regularity {cite}`Nash58`; Moser's Harnack inequality and iteration {cite}`Moser61`; unified treatment in {cite}`GilbargTrudinger01`.

:::

:::{div} feynman-prose
De Giorgi-Nash-Moser (SurgDE) is one of the crown jewels of 20th century analysis. When oscillations blow up (infinite energy in high frequencies), this machinery tames them. The admissibility condition is precisely calibrated: if you can cut off the spectrum at some finite frequency and still have finite energy, then the De Giorgi iteration kicks in and produces Holder continuity. The oscillation in smaller and smaller balls decays like a power of the radius. This is *the* technique for handling elliptic and parabolic regularity.
:::

:::{prf:definition} Surgery Specification: Saturation
:label: def-surgery-be

**Surgery ID:** `SurgBE`
**Target Mode:** `Mode B.E` (Sensitivity Injection)

**Interface Dependencies:**
- **Primary:** $\mathrm{Bound}_B$ (Input Bound Interface: provides sensitivity integral)
- **Secondary:** $\mathrm{LS}_\sigma$ (Stiffness Interface: provides gain bounds)

**Admissibility Signature:**
- **Input Certificate:** $K_{\mathrm{Bound}_B}^{\mathrm{br}}$ (Bode sensitivity violated)
- **Admissibility Predicate:**
  $\|S(i\omega)\|_\infty < M \land \text{phase margin } > 0$
  *(Bounded gain with positive phase margin.)*

**Transformation Law ($\mathcal{O}_S$):**
- **Controller Modification:** Add saturation element $\operatorname{sat}(u) = \operatorname{sign}(u) \min(|u|, u_{\max})$
- **Gain Limiting:** $\|S'\|_\infty \leq \|S\|_\infty / (1 + \epsilon)$
- **Waterbed Conservation:** Redistribute sensitivity to safe frequencies

**Postcondition:**
- **Re-entry Certificate:** $K_{\mathrm{SurgBE}}^{\mathrm{re}}$ (Witnesses bounded sensitivity)
- **Re-entry Target:** `StarveCheck` ({prf:ref}`def-node-starve`)
- **Progress Guarantee:** **Type A**. Bounded saturation adjustments.

**Literature:** Bode sensitivity integrals and waterbed effect {cite}`Bode45`; $\mathcal{H}_\infty$ robust control {cite}`ZhouDoyleGlover96`; anti-windup for saturating systems {cite}`SeronGoodwinDeCarlo00`.

:::

:::{div} feynman-prose
Saturation (SurgBE) comes from control theory's deep understanding of the "waterbed effect." You cannot make a controller infinitely responsive everywhere---if you push sensitivity down in one frequency range, it pops up somewhere else (Bode's integral constraint). The surgery adds a saturation element that clips the control signal, preventing it from going to infinity. The key insight is that you can *redistribute* sensitivity to safe frequencies where it will not cause trouble.
:::

:::{prf:definition} Surgery Specification: Reservoir
:label: def-surgery-bd

**Surgery ID:** `SurgBD`
**Target Mode:** `Mode B.D` (Resource Starvation)

**Interface Dependencies:**
- **Primary:** $\mathrm{Bound}_{\Sigma}$ (Supply Interface: provides resource integral)
- **Secondary:** $C_\mu$ (Compactness Interface: provides state bounds)

**Admissibility Signature:**
- **Input Certificate:** $K_{\mathrm{Bound}_{\Sigma}}^{\mathrm{br}}$ (Resource depletion detected)
- **Admissibility Predicate:**
  $r_{\text{reserve}} > 0 \land \text{recharge rate } > \text{drain rate}$
  *(Positive reserve with sustainable recharge.)*

**Transformation Law ($\mathcal{O}_S$):**
- **State Space:** $X' = X \times [0, R_{\max}]$ (add reservoir variable)
- **Resource Dynamics:** $\dot{r} = \text{recharge} - \text{consumption}$
- **Buffer Zone:** Maintain $r \geq r_{\min}$ always

**Postcondition:**
- **Re-entry Certificate:** $K_{\mathrm{SurgBD}}^{\mathrm{re}}$ (Witnesses positive reservoir)
- **Re-entry Target:** `AlignCheck` ({prf:ref}`def-node-align`)
- **Progress Guarantee:** **Type A**. Bounded reservoir adjustments.

**Literature:** Reservoir computing and echo state networks {cite}`Jaeger04`; resource-bounded computation {cite}`Bellman57`; stochastic inventory theory {cite}`Arrow58`.

:::

:::{div} feynman-prose
Reservoir (SurgBD) handles the practical problem of resource starvation. When input supply runs dry, you need a buffer. The surgery extends the state space with a reservoir variable that tracks resource levels, and the dynamics ensure you never drop below a minimum threshold. Think of it like a battery backup: when the power flickers, the battery kicks in. The resource dynamics are explicitly modeled so you can reason about sustainability.
:::

:::{prf:definition} Surgery Specification: Controller Augmentation via Adjoint Selection
:label: def-surgery-bc

**Surgery ID:** `SurgBC`
**Target Mode:** `Mode B.C` (Control Misalignment / Variety Deficit)

**Interface Dependencies:**
- **Primary:** $\mathrm{GC}_T$ (Gauge Transform Interface: provides alignment data)
- **Secondary:** $\mathrm{Cap}_H$ (Capacity Interface: provides entropy bounds)

**Admissibility Signature:**
- **Input Certificate:** $K_{\mathrm{GC}_T}^{\mathrm{br}}$ (Variety deficit detected: $H(u) < H(d)$)
- **Admissibility Predicate:**
  $H(u) < H(d) - \epsilon \land \exists u' : H(u') \geq H(d)$
  *(Entropy gap exists but is bridgeable—there exists a control with sufficient variety.)*

**Transformation Law ($\mathcal{O}_S$):**
- **Controller Augmentation:** Lift control from $u \in \mathcal{U}$ to $u^* \in \mathcal{U}^* \supseteq \mathcal{U}$ where $\mathcal{U}^*$ has sufficient degrees of freedom (satisfying Ashby's Law of Requisite Variety)
- **Adjoint Selection:** Select $u^*$ from the admissible set $\{u : H(u) \geq H(d)\}$ via adjoint criterion: $u^* = \arg\max_{u \in \mathcal{U}^*} \langle u, \nabla\Phi \rangle$
- **Entropy Matching:** $H(u^*) \geq H(d)$ (guaranteed by augmentation)
- **Alignment Guarantee:** $\langle u^*, d \rangle \geq 0$ (non-adversarial, from adjoint selection)

**Semantic Clarification:** This surgery addresses Ashby's Law violation by **adding degrees of freedom** (controller augmentation), not merely aligning existing controls. The adjoint criterion selects the optimal control from the augmented space. Pure directional alignment without augmentation cannot satisfy $H(u) \geq H(d)$ if the original control space has insufficient entropy.

**Postcondition:**
- **Re-entry Certificate:** $K_{\mathrm{SurgBC}}^{\mathrm{re}}$ (Witnesses entropy-sufficient control with alignment)
- **Re-entry Target:** `BarrierExclusion` ({prf:ref}`def-node-lock`)
- **Progress Guarantee:** **Type B**. Entropy gap strictly decreases to zero.

**Literature:** Ashby's Law of Requisite Variety {cite}`Ashby56`; Pontryagin maximum principle {cite}`Pontryagin62`; adjoint methods in optimal control {cite}`Lions71`; entropy and control {cite}`ConantAshby70`.

:::

:::{div} feynman-prose
Controller Augmentation (SurgBC) addresses Ashby's Law of Requisite Variety---one of the deepest insights in cybernetics. If your controller does not have enough "variety" (entropy, degrees of freedom) to match the disturbances it faces, it *cannot possibly* maintain control. The solution is not to try harder with what you have; it is to *add degrees of freedom*. You augment the control space until $H(u) \geq H(d)$, then use the adjoint criterion to select the best control from this larger space. The entropy gap strictly decreases to zero, ensuring you eventually have enough variety.
:::

---

### Action Nodes (Dynamic Restoration)

:::{div} feynman-prose
The Action Nodes are the final piece of the surgery machinery. Unlike the other surgeries that modify state space or dynamics, these nodes represent *dynamic* transitions---symmetry breaking, tunneling between sectors. They are triggered by the stiffness restoration subtree when the system is stuck at a degenerate critical point and needs to *do something* rather than just *modify something*.
:::

:::{prf:definition} ActionSSB (Symmetry Breaking)
:label: def-action-ssb

**Trigger**: CheckSC YES in restoration subtree

**Action**: Spontaneous symmetry breaking of group $G$

**Output**: Mass gap certificate $K_{\text{gap}}$ guaranteeing stiffness

**Target**: TopoCheck (mass gap implies LS holds)

**Literature:** Goldstone theorem on massless modes {cite}`Goldstone61`; Higgs mechanism for mass generation {cite}`Higgs64`; Anderson's gauge-invariant treatment {cite}`Anderson63`.

:::

:::{div} feynman-prose
ActionSSB (Symmetry Breaking) is the dynamical counterpart of what the Higgs mechanism does for particle physics. When a symmetry group $G$ acts on a degenerate vacuum, the system can "fall" into a particular vacuum state, breaking the symmetry. This generates a mass gap (the Higgs field giving mass to gauge bosons is the famous example). The mass gap certificate then guarantees stiffness---you have escaped the flat direction by picking a direction and rolling down.
:::

:::{prf:definition} ActionTunnel (Instanton Decay)
:label: def-action-tunnel

**Trigger**: CheckTB YES in restoration subtree

**Action**: Quantum/thermal tunneling to new sector

**Output**: Sector transition certificate

**Target**: TameCheck (new sector reached)

**Literature:** Instanton calculus in quantum field theory {cite}`Coleman79`; 't Hooft's instanton solutions {cite}`tHooft76`; semiclassical tunneling {cite}`Vainshtein82`.

:::

:::{div} feynman-prose
ActionTunnel (Instanton Decay) is the other escape route from a degenerate vacuum---instead of breaking symmetry, you tunnel to a different topological sector entirely. This is quantum tunneling made mathematically precise: the instanton path is a classical solution in imaginary time that connects two sectors. Once you have transitioned, you are in a new sector where (hopefully) things are better behaved. This completes the stiffness restoration subtree: either break symmetry and generate mass, or tunnel to somewhere new.
:::
