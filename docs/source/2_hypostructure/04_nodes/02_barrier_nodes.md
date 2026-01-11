(sec-barrier-node-specs)=
## Barrier Node Specifications (Orange Nodes)

Each barrier is specified by:
- **Trigger**: Which gate's NO invokes it
- **Pre-certificates**: Required context (non-circular)
- **Outcome alphabet**: Blocked/Breached (or special)
- **Blocked certificate**: Must imply Pre(next node)
- **Breached certificate**: Must imply mode activation + surgery admissibility
- **Next nodes**: Routing for each outcome

---

### BarrierSat (Saturation Barrier)

:::{prf:definition} Barrier Specification: Saturation
:label: def-barrier-sat

**Barrier ID:** `BarrierSat`

**Interface Dependencies:**
- **Primary:** $D_E$ (provides energy functional $E[\Phi]$ and its drift rate)
- **Secondary:** $\mathrm{SC}_\lambda$ (provides saturation ceiling $E_{\text{sat}}$ and drift bound $C$)

**Sieve Signature:**
- **Weakest Precondition:** $\emptyset$ (entry barrier, no prior certificates required)
- **Barrier Predicate (Blocked Condition):**
  $$E[\Phi] \leq E_{\text{sat}} \lor \text{Drift} \leq C$$

**Natural Language Logic:**
"Is the energy drift bounded by a saturation ceiling?"
*(Even if energy is not globally bounded, the drift rate may be controlled by a saturation mechanism that prevents blow-up.)*

**Outcomes:**
- **Blocked** ($K_{D_E}^{\mathrm{blk}}$): Drift is controlled by saturation ceiling. Singularity excluded via energy saturation principle.
- **Breached** ($K_{D_E}^{\mathrm{br}}$): Uncontrolled drift detected. Activates **Mode C.E** (Energy Blow-up).

**Routing:**
- **On Block:** Proceed to `ZenoCheck`.
- **On Breach:** Trigger **Mode C.E** → Enable Surgery `SurgCE` → Re-enter at `ZenoCheck`.

**Literature:** Saturation and drift bounds via Foster-Lyapunov conditions {cite}`MeynTweedie93`; energy dissipation in physical systems {cite}`Dafermos16`.

:::

---

### BarrierCausal (Causal Censor)

:::{prf:definition} Barrier Specification: Causal Censor
:label: def-barrier-causal

**Barrier ID:** `BarrierCausal`

**Interface Dependencies:**
- **Primary:** $\mathrm{Rec}_N$ (provides computational depth $D(T_*)$ of event tree)
- **Secondary:** $\mathrm{TB}_\pi$ (provides time scale $\lambda(t)$ and horizon $T_*$)

**Sieve Signature:**
- **Weakest Precondition:** $\{K_{D_E}^{\pm}\}$
- **Barrier Predicate (Blocked Condition):**
  $$D(T_*) = \int_0^{T_*} \frac{c}{\lambda(t)} dt = \infty$$

**Natural Language Logic:**
"Does the singularity require infinite computational depth?"
*(If the integral diverges, the singularity would require unbounded computational resources to describe, making it causally inaccessible—a censorship mechanism.)*

**Outcomes:**
- **Blocked** ($K_{\mathrm{Rec}_N}^{\mathrm{blk}}$): Depth diverges; singularity causally censored. Implies Pre(CompactCheck).
- **Breached** ($K_{\mathrm{Rec}_N}^{\mathrm{br}}$): Finite depth; singularity computationally accessible. Activates **Mode C.C** (Event Accumulation).

**Routing:**
- **On Block:** Proceed to `CompactCheck`.
- **On Breach:** Trigger **Mode C.C** → Enable Surgery `SurgCC` → Re-enter at `CompactCheck`.

**Literature:** Causal structure and cosmic censorship {cite}`Penrose69`; {cite}`HawkingPenrose70`; computational depth bounds {cite}`Kolmogorov65`.

:::

---

### BarrierScat (Scattering Barrier) --- Special Alphabet

:::{prf:definition} Barrier Specification: Scattering
:label: def-barrier-scat

**Barrier ID:** `BarrierScat`

**Interface Dependencies:**
- **Primary:** $C_\mu$ (provides concentration measure and interaction functional $\mathcal{M}[\Phi]$)
- **Secondary:** $D_E$ (provides dispersive energy structure)

**Sieve Signature:**
- **Weakest Precondition:** $\{K_{D_E}^{\pm}, K_{\mathrm{Rec}_N}^{\pm}\}$
- **Barrier Predicate (Benign Condition):**
  $$\mathcal{M}[\Phi] < \infty$$

**Natural Language Logic:**
"Is the interaction functional finite (implying dispersion)?"
*(Finite Morawetz interaction implies scattering to free solutions; the energy disperses rather than concentrating.)*

**Outcome Alphabet:** $\{\texttt{Benign}, \texttt{Pathological}\}$ (special)

**Outcomes:**
- **Benign** ($K_{C_\mu}^{\mathrm{ben}}$): Interaction finite; dispersion confirmed. **Success exit** via **Mode D.D** (Global Existence).
- **Pathological** ($K_{C_\mu}^{\mathrm{path}}$): Infinite interaction; soliton-like escape. Activates **Mode C.D** (Concentration-Escape).

**Routing:**
- **On Benign:** Exit to **Mode D.D** (Success: dispersion implies global existence).
- **On Pathological:** Trigger **Mode C.D** → Enable Surgery `SurgCD_Alt` → Re-enter at `Profile`.

**Literature:** Morawetz estimates and scattering {cite}`Morawetz68`; concentration-compactness rigidity {cite}`KenigMerle06`; {cite}`KillipVisan10`.

:::

---

### BarrierTypeII (Type II Barrier)

:::{prf:definition} Barrier Specification: Type II Exclusion
:label: def-barrier-type2

**Barrier ID:** `BarrierTypeII`

**Interface Dependencies:**
- **Primary:** $\mathrm{SC}_\lambda$ (provides scale parameter $\lambda(t)$ and renormalization action)
- **Secondary:** $D_E$ (provides energy functional and blow-up profile $V$)

**Sieve Signature:**
- **Weakest Precondition:** $\{K_{C_\mu}^+\}$ (concentration confirmed, profile exists)
- **Barrier Predicate (Blocked Condition):**
  $$\int \tilde{\mathfrak{D}}(S_t V) dt = \infty$$

**Natural Language Logic:**
"Is the renormalization cost of the profile infinite?"
*(If the integrated defect of the rescaled profile diverges, Type II (self-similar) blow-up is excluded by infinite renormalization cost.)*

**Outcomes:**
- **Blocked** ($K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}$): Renormalization cost infinite; self-similar blow-up excluded. Implies Pre(ParamCheck).
- **Breached** ($K_{\mathrm{SC}_\lambda}^{\mathrm{br}}$): Finite renormalization cost; Type II blow-up possible. Activates **Mode S.E** (Supercritical).

**Routing:**
- **On Block:** Proceed to `ParamCheck`.
- **On Breach:** Trigger **Mode S.E** → Enable Surgery `SurgSE` → Re-enter at `ParamCheck`.

**Non-circularity note:** This barrier is triggered by ScaleCheck NO (supercritical: $\alpha \leq \beta$). Subcriticality ($\alpha > \beta$) may be used as an optional *sufficient* condition for Blocked (via Type I exclusion), but is not a *prerequisite* for barrier evaluation.

**Literature:** Type II blow-up and renormalization {cite}`MerleZaag98`; {cite}`RaphaelSzeftel11`; {cite}`CollotMerleRaphael17`.

:::

---

### BarrierVac (Vacuum Barrier)

:::{prf:definition} Barrier Specification: Vacuum Stability
:label: def-barrier-vac

**Barrier ID:** `BarrierVac`

**Interface Dependencies:**
- **Primary:** $\mathrm{SC}_{\partial c}$ (provides vacuum potential $V$ and thermal scale $k_B T$)
- **Secondary:** $\mathrm{LS}_\sigma$ (provides stability landscape and barrier heights $\Delta V$)

**Sieve Signature:**
- **Weakest Precondition:** $\{K_{C_\mu}^+, K_{\mathrm{SC}_\lambda}^{\pm}\}$
- **Barrier Predicate (Blocked Condition):**
  $$\Delta V > k_B T$$

**Natural Language Logic:**
"Is the phase stable against thermal/parameter drift?"
*(If the potential barrier exceeds the thermal energy scale, the vacuum is stable against fluctuation-induced decay—the mass gap principle.)*

**Outcomes:**
- **Blocked** ($K_{\mathrm{SC}_{\partial c}}^{\mathrm{blk}}$): Phase stable; barrier exceeds thermal scale. Implies Pre(GeomCheck).
- **Breached** ($K_{\mathrm{SC}_{\partial c}}^{\mathrm{br}}$): Phase unstable; vacuum decay possible. Activates **Mode S.C** (Parameter Instability).

**Routing:**
- **On Block:** Proceed to `GeomCheck`.
- **On Breach:** Trigger **Mode S.C** → Enable Surgery `SurgSC` → Re-enter at `GeomCheck`.

**Literature:** Vacuum stability and phase transitions {cite}`Goldstone61`; {cite}`Higgs64`; {cite}`Coleman75`.

:::

---

### BarrierCap (Capacity Barrier)

:::{prf:definition} Barrier Specification: Capacity
:label: def-barrier-cap

**Barrier ID:** `BarrierCap`

**Interface Dependencies:**
- **Primary:** $\mathrm{Cap}_H$ (provides Hausdorff capacity $\mathrm{Cap}_H(S)$ of singular set $S$)
- **Secondary:** None (pure geometric criterion)

**Sieve Signature:**
- **Weakest Precondition:** $\{K_{\mathrm{SC}_{\partial c}}^{\pm}\}$
- **Barrier Predicate (Blocked Condition):**
  $$\mathrm{Cap}_H(S) = 0$$

**Natural Language Logic:**
"Is the singular set of measure zero?"
*(Zero capacity implies the singular set is negligible—it cannot carry enough mass to affect the dynamics. This is the capacity barrier principle.)*

**Outcomes:**
- **Blocked** ($K_{\mathrm{Cap}_H}^{\mathrm{blk}}$): Singular set has zero capacity; negligible. Implies Pre(StiffnessCheck).
- **Breached** ($K_{\mathrm{Cap}_H}^{\mathrm{br}}$): Positive capacity; singular set non-negligible. Activates **Mode C.D** (Geometric Collapse).

**Routing:**
- **On Block:** Proceed to `StiffnessCheck`.
- **On Breach:** Trigger **Mode C.D** → Enable Surgery `SurgCD` → Re-enter at `StiffnessCheck`.

**Literature:** Capacity and removable singularities {cite}`Federer69`; {cite}`EvansGariepy15`; {cite}`AdamsHedberg96`.

:::

---

### BarrierGap (Spectral Barrier) --- Special Alphabet

:::{prf:definition} Barrier Specification: Spectral Gap
:label: def-barrier-gap

**Barrier ID:** `BarrierGap`

**Interface Dependencies:**
- **Primary:** $\mathrm{LS}_\sigma$ (provides spectrum $\sigma(L)$ of linearized operator $L$)
- **Secondary:** $\mathrm{GC}_\nabla$ (provides gradient structure and Hessian at critical points)

**Sieve Signature:**
- **Weakest Precondition:** $\{K_{\mathrm{Cap}_H}^{\pm}\}$
- **Barrier Predicate (Blocked Condition):**
  $$\inf \sigma(L) > 0$$

**Natural Language Logic:**
"Is there a spectral gap (positive curvature) at the minimum?"
*(A positive spectral gap implies exponential decay toward the critical point via Łojasiewicz-Simon inequality—the spectral generator principle.)*

**Outcome Alphabet:** $\{\texttt{Blocked}, \texttt{Stagnation}\}$ (special)

**Outcomes:**
- **Blocked** ($K_{\mathrm{LS}_\sigma}^{\mathrm{blk}}$): Spectral gap exists; exponential convergence guaranteed. Implies Pre(TopoCheck).
- **Stagnation** ($K_{\mathrm{LS}_\sigma}^{\mathrm{stag}}$): No spectral gap; system may stagnate at degenerate critical point. Routes to restoration subtree.

**Routing:**
- **On Block:** Proceed to `TopoCheck`.
- **On Stagnation:** Enter restoration subtree via `BifurcateCheck` (Node 7a).

**Literature:** Spectral gap and gradient flows {cite}`Simon83`; {cite}`FeehanMaridakis19`; {cite}`Huang06`.

:::

:::{prf:lemma} Gap implies Łojasiewicz-Simon
:label: lem-gap-to-ls

Under the Gradient Condition ($\mathrm{GC}_\nabla$) plus analyticity of $\Phi$ near critical points:
$$\text{Spectral gap } \lambda_1 > 0 \Rightarrow \text{LS}(\theta = \tfrac{1}{2}, C_{\text{LS}} = \sqrt{\lambda_1})$$
This is the **canonical promotion** from gap certificate to stiffness certificate, bridging the diagram's ``Hessian positive?'' intuition with the formal LS inequality predicate.

:::

---

### BarrierAction (Action Barrier)

:::{prf:definition} Barrier Specification: Action Gap
:label: def-barrier-action

**Barrier ID:** `BarrierAction`

**Interface Dependencies:**
- **Primary:** $\mathrm{TB}_\pi$ (provides topological action gap $S_{\min}$ and threshold $\Delta$)
- **Secondary:** $D_E$ (provides current energy $E[\Phi]$)

**Sieve Signature:**
- **Weakest Precondition:** $\{K_{\mathrm{LS}_\sigma}^{\pm}\}$
- **Barrier Predicate (Blocked Condition):**
  $$E[\Phi] < S_{\min} + \Delta$$

**Natural Language Logic:**
"Is the energy insufficient to cross the topological gap?"
*(If current energy is below the action threshold, topological transitions (tunneling, kink formation) are energetically forbidden—the topological suppression principle.)*

**Outcomes:**
- **Blocked** ($K_{\mathrm{TB}_\pi}^{\mathrm{blk}}$): Energy below action gap; tunneling suppressed. Implies Pre(TameCheck).
- **Breached** ($K_{\mathrm{TB}_\pi}^{\mathrm{br}}$): Energy sufficient for topological transition. Activates **Mode T.E** (Topological Transition).

**Routing:**
- **On Block:** Proceed to `TameCheck`.
- **On Breach:** Trigger **Mode T.E** → Enable Surgery `SurgTE` → Re-enter at `TameCheck`.

**Literature:** Topological obstructions and action principles {cite}`Smale67`; {cite}`Conley78`; {cite}`Floer89`.

:::

---

### BarrierOmin (O-Minimal Barrier)

:::{prf:definition} Barrier Specification: O-Minimal Taming
:label: def-barrier-omin

**Barrier ID:** `BarrierOmin`

**Interface Dependencies:**
- **Primary:** $\mathrm{TB}_O$ (provides o-minimal structure $\mathcal{O}$ and definability criteria)
- **Secondary:** $\mathrm{Rep}_K$ (provides representation-theoretic bounds on complexity)

**Sieve Signature:**
- **Weakest Precondition:** $\{K_{\mathrm{TB}_\pi}^{\pm}\}$
- **Barrier Predicate (Blocked Condition):**
  $$S \in \mathcal{O}\text{-min}$$

**Natural Language Logic:**
"Is the topology definable in an o-minimal structure?"
*(O-minimal definability implies tameness: no pathological fractals, finite stratification, controlled asymptotic behavior—the o-minimal taming principle.)*

**Outcomes:**
- **Blocked** ($K_{\mathrm{TB}_O}^{\mathrm{blk}}$): Topology is o-minimally definable; wild behavior tamed. Implies Pre(ErgoCheck).
- **Breached** ($K_{\mathrm{TB}_O}^{\mathrm{br}}$): Topology not definable; genuinely wild structure. Activates **Mode T.C** (Topological Complexity).

**Routing:**
- **On Block:** Proceed to `ErgoCheck`.
- **On Breach:** Trigger **Mode T.C** → Enable Surgery `SurgTC` → Re-enter at `ErgoCheck`.

**Literature:** O-minimal structures and tame topology {cite}`vandenDries98`; {cite}`Kurdyka98`; {cite}`Wilkie96`.

:::

---

### BarrierMix (Mixing Barrier)

:::{prf:definition} Barrier Specification: Ergodic Mixing
:label: def-barrier-mix

**Barrier ID:** `BarrierMix`

**Interface Dependencies:**
- **Primary:** $\mathrm{TB}_\rho$ (provides mixing time $\tau_{\text{mix}}$ and escape probability)
- **Secondary:** $D_E$ (provides energy landscape for trap depth estimation)

**Sieve Signature:**
- **Weakest Precondition:** $\{K_{\mathrm{TB}_O}^{\pm}\}$
- **Barrier Predicate (Blocked Condition):**
  $$\tau_{\text{mix}} < \infty$$

**Natural Language Logic:**
"Does the system mix fast enough to escape traps?"
*(Finite mixing time implies ergodicity: the system explores all accessible states and cannot be permanently trapped—the ergodic mixing principle.)*

**Outcomes:**
- **Blocked** ($K_{\mathrm{TB}_\rho}^{\mathrm{blk}}$): Mixing time finite; trap escapable. Implies Pre(ComplexCheck).
- **Breached** ($K_{\mathrm{TB}_\rho}^{\mathrm{br}}$): Infinite mixing time; permanent trapping possible. Activates **Mode T.D** (Trapping).

**Routing:**
- **On Block:** Proceed to `ComplexCheck`.
- **On Breach:** Trigger **Mode T.D** → Enable Surgery `SurgTD` → Re-enter at `ComplexCheck`.

**Literature:** Ergodic theory and mixing {cite}`Birkhoff31`; {cite}`Furstenberg81`; {cite}`MeynTweedie93`.

:::

---

### BarrierEpi (Epistemic Barrier)

:::{prf:definition} Barrier Specification: Epistemic Horizon
:label: def-barrier-epi

**Barrier ID:** `BarrierEpi`

**Interface Dependencies:**
- **Primary:** $\mathrm{Rep}_K$ (provides Kolmogorov complexity $K(x)$ of state description)
- **Secondary:** $\mathrm{Cap}_H$ (provides DPI information bound $I_{\max}$)

**Sieve Signature:**
- **Weakest Precondition:** $\{K_{\mathrm{TB}_\rho}^{\pm}\}$
- **Barrier Predicate (Blocked Condition):**
  $$\sup_{\epsilon > 0} K_\epsilon(x) \leq S_{\text{BH}}$$
  where $K_\epsilon(x) := \min\{|p| : d(U(p), x) < \epsilon\}$ is the $\epsilon$-approximable complexity.

**Semantic Clarification:**
This barrier is triggered when Node 11 determines that exact complexity is uncomputable. The predicate now asks: "Even though we cannot compute $K(x)$ exactly, can we bound all computable approximations within the holographic limit?" This makes the "Blocked" outcome logically reachable:
- If approximations converge to a finite limit $\leq S_{\text{BH}}$ → Blocked
- If approximations diverge or exceed $S_{\text{BH}}$ → Breached

**Natural Language Logic:**
"Is the approximable description length within physical bounds?"
*(Even when exact complexity is uncomputable, if all computable approximations stay within the holographic bound, the system cannot encode more information than spacetime permits—the epistemic horizon principle.)*

**Outcomes:**
- **Blocked** ($K_{\mathrm{Rep}_K}^{\mathrm{blk}}$): Approximable complexity bounded; within holographic limit. Implies Pre(OscillateCheck).
- **Breached** ($K_{\mathrm{Rep}_K}^{\mathrm{br}}$): Approximations diverge or exceed holographic bound; epistemic horizon violated. Activates **Mode D.C** (Complexity Explosion).

**Routing:**
- **On Block:** Proceed to `OscillateCheck`.
- **On Breach:** Trigger **Mode D.C** → Enable Surgery `SurgDC` → Re-enter at `OscillateCheck`.

**Literature:** Kolmogorov complexity {cite}`Kolmogorov65`; holographic bounds {cite}`tHooft93`; {cite}`Susskind95`; {cite}`Bousso02`; resource-bounded complexity {cite}`LiVitanyi08`.

:::

---

### BarrierFreq (Frequency Barrier)

:::{prf:definition} Barrier Specification: Frequency
:label: def-barrier-freq

**Barrier ID:** `BarrierFreq`

**Interface Dependencies:**
- **Primary:** $\mathrm{GC}_\nabla$ (provides spectral density $S(\omega)$ and oscillation structure)
- **Secondary:** $\mathrm{SC}_\lambda$ (provides frequency cutoff and scaling)

**Sieve Signature:**
- **Weakest Precondition:** $\{K_{\mathrm{Rep}_K}^{\pm}\}$
- **Barrier Predicate (Blocked Condition):**
  $$\int \omega^2 S(\omega) d\omega < \infty$$

**Natural Language Logic:**
"Is the total oscillation energy finite?"
*(Finite second moment of the spectral density implies bounded oscillation energy—the frequency barrier principle prevents infinite frequency cascades.)*

**Outcomes:**
- **Blocked** ($K_{\mathrm{GC}_\nabla}^{\mathrm{blk}}$): Oscillation integral finite; no frequency blow-up. Implies Pre(BoundaryCheck).
- **Breached** ($K_{\mathrm{GC}_\nabla}^{\mathrm{br}}$): Infinite oscillation energy; frequency cascade detected. Activates **Mode D.E** (Oscillation Divergence).

**Routing:**
- **On Block:** Proceed to `BoundaryCheck`.
- **On Breach:** Trigger **Mode D.E** → Enable Surgery `SurgDE` → Re-enter at `BoundaryCheck`.

**Literature:** De Giorgi-Nash-Moser regularity theory {cite}`DeGiorgi57`; {cite}`Nash58`; {cite}`Moser60`.

:::

---

### Boundary Barriers (BarrierBode, BarrierInput, BarrierVariety)

:::{prf:definition} Barrier Specification: Bode Sensitivity
:label: def-barrier-bode

**Barrier ID:** `BarrierBode`

**Interface Dependencies:**
- **Primary:** $\mathrm{Bound}_B$ (provides sensitivity function $S(s)$ and Bode integral $B_{\text{Bode}}$)
- **Secondary:** $\mathrm{LS}_\sigma$ (provides stability landscape for waterbed constraints)

**Sieve Signature:**
- **Weakest Precondition:** $\{K_{\mathrm{Bound}_\partial}^+\}$ (open system confirmed)
- **Barrier Predicate (Blocked Condition):**
  $$\int_0^\infty \ln \lVert S(i\omega) \rVert d\omega > -\infty$$

**Natural Language Logic:**
"Is the sensitivity integral conserved (waterbed effect)?"
*(The Bode integral constraint implies sensitivity cannot be reduced everywhere—reduction in one frequency band must be compensated elsewhere. Finite integral means the waterbed is bounded.)*

**Outcomes:**
- **Blocked** ($K_{\mathrm{Bound}_B}^{\mathrm{blk}}$): Bode integral finite; sensitivity bounded. Implies Pre(StarveCheck).
- **Breached** ($K_{\mathrm{Bound}_B}^{\mathrm{br}}$): Unbounded sensitivity; waterbed constraint violated. Activates **Mode B.E** (Sensitivity Explosion).

**Routing:**
- **On Block:** Proceed to `StarveCheck`.
- **On Breach:** Trigger **Mode B.E** → Enable Surgery `SurgBE` → Re-enter at `StarveCheck`.

**Literature:** Bode integral constraints and robust control {cite}`DoyleFrancisTannenbaum92`; {cite}`Sontag98`.

:::

:::{prf:definition} Barrier Specification: Input Stability
:label: def-barrier-input

**Barrier ID:** `BarrierInput`

**Interface Dependencies:**
- **Primary:** $\mathrm{Bound}_{\Sigma}$ (provides input reserve $r_{\text{reserve}}$ and flow integrals)
- **Secondary:** $C_\mu$ (provides concentration structure for resource distribution)

**Sieve Signature:**
- **Weakest Precondition:** $\{K_{\mathrm{Bound}_B}^{\pm}\}$
- **Barrier Predicate (Blocked Condition):**
  $$r_{\text{reserve}} > 0$$

**Natural Language Logic:**
"Is there a reservoir to prevent starvation?"
*(Positive reserve ensures the system can buffer transient input deficits—the input stability principle prevents resource starvation.)*

**Outcomes:**
- **Blocked** ($K_{\mathrm{Bound}_{\Sigma}}^{\mathrm{blk}}$): Reserve positive; buffer exists against starvation. Implies Pre(AlignCheck).
- **Breached** ($K_{\mathrm{Bound}_{\Sigma}}^{\mathrm{br}}$): Reserve depleted; system vulnerable to input starvation. Activates **Mode B.D** (Resource Depletion).

**Routing:**
- **On Block:** Proceed to `AlignCheck`.
- **On Breach:** Trigger **Mode B.D** → Enable Surgery `SurgBD` → Re-enter at `AlignCheck`.

**Literature:** Input-to-state stability {cite}`Khalil02`; {cite}`Sontag98`.

:::

:::{prf:definition} Barrier Specification: Requisite Variety
:label: def-barrier-variety

**Barrier ID:** `BarrierVariety`

**Interface Dependencies:**
- **Primary:** $\mathrm{GC}_T$ (provides control entropy $H(u)$ and tangent cone structure)
- **Secondary:** $\mathrm{Cap}_H$ (provides disturbance entropy $H(d)$ and capacity bounds)

**Sieve Signature:**
- **Weakest Precondition:** $\{K_{\mathrm{Bound}_{\Sigma}}^{\pm}\}$
- **Barrier Predicate (Blocked Condition):**
  $$H(u) \geq H(d)$$

**Natural Language Logic:**
"Does control entropy match disturbance entropy?"
*(Ashby's Law of Requisite Variety: a controller can only regulate what it can match in variety. Control must have at least as much entropy as the disturbance it counters.)*

**Outcomes:**
- **Blocked** ($K_{\mathrm{GC}_T}^{\mathrm{blk}}$): Control variety sufficient; can counter all disturbances. Implies Pre(BarrierExclusion).
- **Breached** ($K_{\mathrm{GC}_T}^{\mathrm{br}}$): Variety deficit; control cannot match disturbance complexity. Activates **Mode B.C** (Control Deficit).

**Routing:**
- **On Block:** Proceed to `BarrierExclusion`.
- **On Breach:** Trigger **Mode B.C** → Enable Surgery `SurgBC` → Re-enter at `BarrierExclusion`.

**Literature:** Requisite variety and cybernetics {cite}`Ashby56`; {cite}`ConantAshby70`.

:::

---
