(sec-surgery-node-specs)=
## Surgery Node Specifications (Purple Nodes)

Each surgery is specified by:
- **Inputs**: Breach certificate + surgery data
- **Action**: Abstract operation performed
- **Postcondition**: Re-entry certificate + target node
- **Progress measure**: Ensures termination

:::{prf:theorem} Non-circularity rule
:label: thm-non-circularity

A barrier invoked because predicate $P_i$ failed **cannot** assume $P_i$ as a prerequisite. Formally:
$$\text{Trigger}(B) = \text{Gate}_i \text{ NO} \Rightarrow P_i \notin \mathrm{Pre}(B)$$

**Scope of Non-Circularity:** This syntactic check ($K_i^- \notin \Gamma$) prevents direct circular dependencies. Semantic circularity (proof implicitly using an equivalent of the target conclusion) is addressed by the derivation-dependency constraint: certificate proofs must cite only lemmas of lower rank in the proof DAG. The ranking is induced by the topological sort of the Sieve, ensuring well-foundedness ({cite}`VanGelder91`).

**Literature:** Well-founded semantics {cite}`VanGelder91`; stratification in logic programming {cite}`AptBolPedreschi94`.

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

---

### Surgery Contract Template

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
  $V \in \mathcal{L}_T \land \text{Cap}(\Sigma) \le \varepsilon_{\text{adm}}$
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

---

### Surgery Specifications

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
  $\text{Growth}(\Phi) \text{ is conformal} \land \partial_\infty X \text{ is definable}$
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
  $\alpha - \beta < \epsilon_{\text{crit}} \land \text{Profile } V \text{ is smooth}$
  *(Near-critical with smooth profile allows perturbative lift.)*

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
  $\text{Cap}_H(\Sigma) \leq \varepsilon_{\text{adm}} \land V \in \mathcal{L}_{\text{neck}}$
  *(Small singular set with recognizable neck structure.)*

**Transformation Law ($\mathcal{O}_S$):**
- **Excision:** $X' = X \setminus B_\epsilon(\Sigma)$
- **Capping:** Glue auxiliary space $X_{\text{aux}}$ matching boundary
- **Height Drop:** $\Phi(x') \leq \Phi(x) - c \cdot \text{Vol}(\Sigma)^{2/n}$

**Postcondition:**
- **Re-entry Certificate:** $K_{\mathrm{SurgCD}}^{\mathrm{re}}$ (Witnesses smooth excision)
- **Re-entry Target:** `StiffnessCheck` ({prf:ref}`def-node-stiffness`)
- **Progress Guarantee:** **Type B**. Singular set measure strictly decreases.

**Literature:** Ricci flow surgery {cite}`Hamilton97`; {cite}`Perelman03`; geometric measure theory {cite}`Federer69`.

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

**Complexity Type on Re-entry:** The re-entry evaluates $K(\mu_t)$ where $\mu_t = \text{Law}(x_t)$ is the probability measure on trajectories, not $K(x_t(\omega))$ for individual sample paths. The SDE has finite description length (drift $b$, diffusion $\sigma$, initial law $\mu_0$) even though individual realizations are algorithmically incompressible (white noise is random). This ensures S12 does not cause immediate failure at Node 11.

**Literature:** Stochastic perturbation and mixing {cite}`MeynTweedie93`; {cite}`HairerMattingly11`.

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
  $K(x) \leq S_{\text{BH}} + \epsilon \land x \in W^{1,\infty}$
  *(Near holographic bound with Lipschitz regularity.)*

**Transformation Law ($\mathcal{O}_S$):**
- **State Space:** $X' = X$ (same space, coarsened description)
- **Viscosity Regularization:** $x' = x * \phi_\epsilon$ (convolution smoothing)
- **Complexity Reduction:** $K(x') \leq K(x) - c \cdot \epsilon$

**Postcondition:**
- **Re-entry Certificate:** $K_{\mathrm{SurgDC}}^{\mathrm{re}}$ (Witnesses reduced complexity)
- **Re-entry Target:** `OscillateCheck` ({prf:ref}`def-node-oscillate`)
- **Progress Guarantee:** **Type B**. Kolmogorov complexity strictly decreases.

**Literature:** Viscosity solutions {cite}`CrandallLions83`; regularization and mollification {cite}`EvansGariepy15`.

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
  $$\exists \Lambda < \infty: \sup_{\Lambda' \leq \Lambda} \int_{|\omega| \leq \Lambda'} \omega^2 S(\omega) d\omega < \infty \quad \land \quad \text{uniform ellipticity}$$
  *(Divergence is "elliptic-regularizable" — De Giorgi-Nash-Moser applies to truncated spectrum.)*

**Transformation Law ($\mathcal{O}_S$):**
- **State Space:** $X' = X$ (same space, improved regularity)
- **Hölder Regularization:** Apply De Giorgi-Nash-Moser iteration
- **Oscillation Damping:** $\text{osc}_{B_r}(x') \leq C r^\alpha \text{osc}_{B_1}(x)$

**Postcondition:**
- **Re-entry Certificate:** $K_{\mathrm{SurgDE}}^{\mathrm{re}}$ (Witnesses Hölder continuity)
- **Re-entry Target:** `BoundaryCheck` ({prf:ref}`def-node-boundary`)
- **Progress Guarantee:** **Type A**. Bounded regularity improvements per unit time.

**Literature:** De Giorgi's original regularity theorem {cite}`DeGiorgi57`; Nash's parabolic regularity {cite}`Nash58`; Moser's Harnack inequality and iteration {cite}`Moser61`; unified treatment in {cite}`GilbargTrudinger01`.

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
- **Controller Modification:** Add saturation element $\text{sat}(u) = \text{sign}(u) \min(|u|, u_{\max})$
- **Gain Limiting:** $\|S'\|_\infty \leq \|S\|_\infty / (1 + \epsilon)$
- **Waterbed Conservation:** Redistribute sensitivity to safe frequencies

**Postcondition:**
- **Re-entry Certificate:** $K_{\mathrm{SurgBE}}^{\mathrm{re}}$ (Witnesses bounded sensitivity)
- **Re-entry Target:** `StarveCheck` ({prf:ref}`def-node-starve`)
- **Progress Guarantee:** **Type A**. Bounded saturation adjustments.

**Literature:** Bode sensitivity integrals and waterbed effect {cite}`Bode45`; $\mathcal{H}_\infty$ robust control {cite}`ZhouDoyleGlover96`; anti-windup for saturating systems {cite}`SeronGoodwinDeCarlo00`.

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

---

### Action Nodes (Dynamic Restoration)

:::{prf:definition} ActionSSB (Symmetry Breaking)
:label: def-action-ssb

**Trigger**: CheckSC YES in restoration subtree

**Action**: Spontaneous symmetry breaking of group $G$

**Output**: Mass gap certificate $K_{\text{gap}}$ guaranteeing stiffness

**Target**: TopoCheck (mass gap implies LS holds)

**Literature:** Goldstone theorem on massless modes {cite}`Goldstone61`; Higgs mechanism for mass generation {cite}`Higgs64`; Anderson's gauge-invariant treatment {cite}`Anderson63`.

:::

:::{prf:definition} ActionTunnel (Instanton Decay)
:label: def-action-tunnel

**Trigger**: CheckTB YES in restoration subtree

**Action**: Quantum/thermal tunneling to new sector

**Output**: Sector transition certificate

**Target**: TameCheck (new sector reached)

**Literature:** Instanton calculus in quantum field theory {cite}`Coleman79`; 't Hooft's instanton solutions {cite}`tHooft76`; semiclassical tunneling {cite}`Vainshtein82`.

:::
