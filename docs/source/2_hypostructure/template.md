---
title: "Hypostructure Proof Object Template"
---

# Structural Sieve Proof: [PROBLEM NAME]

## Metadata

| Field | Value |
|-------|-------|
| **Problem** | [Full problem statement, e.g., "Global regularity for 3D Navier-Stokes"] |
| **System Type** | $T_{\text{[type]}}$ (e.g., $T_{\text{parabolic}}$, $T_{\text{dispersive}}$, $T_{\text{arithmetic}}$, $T_{\text{algorithmic}}$) |
| **Target Claim** | [e.g., Global Regularity, Existence, Uniqueness, Finite-Time Blowup] |
| **Framework Version** | Hypostructure v1.0 |
| **Date** | [YYYY-MM-DD] |

### Label Naming Conventions

When filling out this template, replace `[problem-slug]` with a lowercase, hyphenated identifier for your problem (e.g., `navier-stokes`, `riemann`, `yang-mills`). Use these label prefixes:

| Type | Pattern | Example |
|------|---------|---------|
| Definitions | `def-[problem-slug]-*` | `def-navier-stokes-arena` |
| Theorems | `thm-[problem-slug]-*` | `thm-navier-stokes-main` |
| Lemmas | `lem-[problem-slug]-*` | `lem-navier-stokes-energy` |
| Remarks | `rem-[problem-slug]-*` | `rem-navier-stokes-scaling` |
| Proofs | `proof-[problem-slug]-*` | `proof-thm-navier-stokes-main` |
| Proof Sketches | `sketch-[problem-slug]-*` | `sketch-thm-navier-stokes-main` |

---

## Automation Witness (Framework Offloading Justification)

We certify that this instance is eligible for the Universal Singularity Modules.

- **Type witness:** $T_{\text{[type]}}$ is a **good type** (finite stratification + constructible caps).
- **Automation witness:** The Hypostructure satisfies the **Automation Guarantee** (Definition {prf:ref}`def-automation-guarantee`), hence profile extraction, admissibility, and surgery are computed automatically by the framework factories.

**Certificate:**
$$K_{\mathrm{Auto}}^+ = (T_{\text{[type]}}\ \text{good},\ \text{AutomationGuarantee holds},\ \text{factories enabled: RESOLVE-AutoProfile, RESOLVE-AutoAdmit, RESOLVE-AutoSurgery})$$

---

## Abstract

This document presents a **machine-checkable proof object** for **[PROBLEM NAME]** using the Hypostructure framework.

**Approach:** [Brief description of the approach, e.g., "We instantiate the parabolic hypostructure with energy $\Phi$, dissipation $\mathfrak{D}$, and verify all 17 sieve nodes to establish global regularity."]

**Result:** [Brief statement of the result, e.g., "The Lock is blocked via Tactic E7 (thermodynamic), establishing global smoothness for all $H^1$ initial data."]

---

## Theorem Statement

::::{prf:theorem} [Problem Name]
:label: thm-problem-slug-main

**Given:**
- State space: $\mathcal{X} = $ [Define]
- Dynamics: [Equation or evolution rule]
- Initial data: [Constraints]

**Claim:** [Precise mathematical statement to prove]

**Notation:**
| Symbol | Definition |
|--------|------------|
| $\mathcal{X}$ | State space |
| $\Phi$ | Height functional |
| $\mathfrak{D}$ | Dissipation rate |
| $S_t$ | Flow/semigroup |
| $\Sigma$ | Singular set |

::::

---

:::{dropdown} **LLM Execution Protocol** (Click to expand)

## Note for LLM Agents: Complete Algorithmic Protocol

> **IMPORTANT:** When using this template, you are NOT doing traditional mathematical analysis. You are executing a **deterministic diagnostic algorithm** that produces a **machine-checkable proof object**.

---

### **A.1 Mindset Shift**

**Abandon the traditional mathematician mindset.** Do not try to "understand" or "prove" anything creatively. Instead:

1. **Implement interfaces mechanically.** Each interface permit has a checklist. Fill in the blanks with the problem's data.
2. **Run the sieve node-by-node.** At each node, check a single predicate. Record the certificate.
3. **Follow the flowchart.** If YES → next node. If NO → check barrier. If INC → record obligation, continue.
4. **Trust the framework.** The mathematical heavy lifting is already encoded in the metatheorems.

**Think of yourself as a compiler, not a mathematician.** You parse the problem into the interface format, then execute the sieve algorithm. The theorems fall out automatically from the certificate chain.

**Core principle: Unknown is recoverable.** If you cannot verify a condition, emit `K^{inc}` with a structured payload. Do NOT guess, do NOT skip, do NOT pretend success.

---

### **A.2 Certificate Outcome Types**

Every node produces exactly one certificate. Here is the decision tree:

| Outcome | Symbol | Meaning | Action |
|---------|--------|---------|--------|
| **YES** | $K_X^+$ | Condition X verified | Record witnesses, continue |
| **NO (witness)** | $K_X^{\mathrm{wit}}$ | Condition X refuted | Check barrier |
| **NO (inconclusive)** | $K_X^{\mathrm{inc}}$ | Verification incomplete | Record obligation payload, continue |
| **BLOCKED** | $K_X^{\mathrm{blk}}$ | Barrier prevented bad behavior | Continue (bad case excluded) |
| **BREACHED** | $K_X^{\mathrm{br}}$ | Barrier failed | Trigger surgery protocol |

*Note:* By Definition {prf:ref}`def-typed-no-certificates`, **NO** is a sum type:
$K_X^- := K_X^{\mathrm{wit}} \sqcup K_X^{\mathrm{inc}}$. Use $K_X^-$ only as shorthand when the branch is irrelevant.

**Decision flowchart at each node:**
```
Can you verify the predicate?
├─ YES → Emit K^+, go to next node
├─ NO (witness) → Emit K^{wit}, check barrier
│   ├─ Barrier blocks → Emit K^{blk}, go to next node
│   └─ Barrier breached → Emit K^{br}, trigger surgery
└─ NO (inconclusive) → Emit K^{inc} with payload, go to next node
```

---

### **A.3 Inc Permit Protocol**

When you cannot close an estimate or verify a property:

1. **DO NOT** return $K^+$ (that would be unsound)
2. **DO NOT** return $K^{\mathrm{wit}}$ (you haven't refuted it)
3. **EMIT** $K^{\mathrm{inc}}$ with structured payload:

```
K_X^{inc} = {
  obligation: "What needs to be established",
  missing: ["List of certificates that would close the gap"],
  failure_code: "STANDARDIZED_CODE",
  trace: "Where in your reasoning the gap occurred"
}
```

**Standard failure codes:**
| Code | Meaning | Typical Resolution |
|------|---------|-------------------|
| `MISSING_EMBEDDING` | Need Sobolev/interpolation inequality | Add embedding certificate |
| `MISSING_SPECTRAL_GAP` | Need spectral gap/Poincaré inequality | Add spectral gap certificate |
| `MISSING_CAPACITY_BOUND` | Need Hausdorff dimension estimate | Add capacity certificate |
| `MISSING_VORTICITY_LINK` | Need elliptic regularity / Biot-Savart | Switch to vorticity formulation |
| `MISSING_STIFFNESS` | Need Łojasiewicz-Simon inequality | Add stiffness certificate |
| `MISSING_COMPACTNESS` | Need profile decomposition | Add concentration-compactness |
| `NEEDS_UPGRADE` | Premises exist but upgrade not yet applied | Apply upgrade rule |

---

### **A.4 Upgrade Rule Execution**

After the sieve pass, execute upgrade rules to discharge inc certificates:

**Step 1: Scan** Γ for all $K^{\mathrm{inc}}$ certificates.

**Step 2: For each** $K^{\mathrm{inc}}$, check if any upgrade rule $U$ applies:
- **Premises:** Does Γ contain all certificates listed in `missing`?
- **Non-circularity:** The target $K^+$ must NOT be used to derive the premises.

**Step 3: If upgrade applies:**
- Add $K^+$ to Γ
- Mark obligation as discharged in ledger
- Keep $K^{\mathrm{inc}}$ as audit trail

**Upgrade rule template:**
```
U_{X→+}: K_X^{inc} ∧ K_A^+ ∧ K_B^+ ⟹ K_X^+

Premises: {K_A^+, K_B^+} ⊆ Γ
Target: K_X^{inc} (with missing = {A, B})
Non-circularity: K_X^+ not used to derive K_A^+ or K_B^+
```

**Two upgrade types:**
1. **Instantaneous** (same pass): Premises available before the inc certificate was emitted.
2. **A-posteriori** (after surgery/later nodes): Premises obtained from subsequent nodes or surgery.

---

### **A.5 Breach Detection and Surgery**

When $K^{\mathrm{wit}}$ is emitted at a barrier node:

**Step 1: Check if barrier blocks.**
- Does existing Γ contradict the bad scenario?
- Are there certificates that exclude the failure mode?

**Step 2: If blocked:**
- Emit $K^{\mathrm{blk}}$ with reason
- Continue to next node

**Step 3: If breached:**
- Emit $K^{\mathrm{br}}$ with breach obligations
- Trigger surgery protocol:

**Surgery Protocol:**
1. **Select surgery map:** Choose a semantics-preserving transformation (e.g., velocity → vorticity, Fourier → physical space).
2. **Emit** $K_{\mathrm{Surg}}^+(\text{map\_id})$ certifying the transformation preserves the theorem.
3. **Run post-surgery nodes:** Execute new verification nodes in the transformed representation.
4. **Re-enter:** Use new certificates to discharge the breach obligations.
5. **Apply a-posteriori upgrades:** Check if new certificates enable upgrades of earlier inc permits.

---

### **A.6 Obligation Tracking**

**Maintain an obligation ledger throughout the run:**

| Event | Action |
|-------|--------|
| $K^{\mathrm{inc}}$ emitted | Add obligation to ledger with ID |
| Upgrade succeeds | Mark obligation as DISCHARGED |
| $K^{\mathrm{br}}$ emitted | Add breach obligations to ledger |
| Re-entry succeeds | Mark breach obligations as DISCHARGED |

**The ledger must be EMPTY for a valid proof object.**

---

### **A.7 Completion Criteria**

A proof object is **VALID** if and only if:

- [ ] **All nodes executed** (no skips)
- [ ] **Lock passed:** $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$
- [ ] **Obligation ledger is EMPTY**
- [ ] **No unresolved** $K^{\mathrm{inc}}$ in final Γ (all upgraded or discharged)

If any of these fail, the run produces a **conditional proof object** that documents exactly what remains to be established.

---

### **A.8 Step-by-Step Implementation Guide for New Problems**

**Phase 1: Instantiation (Part I)**
1. Identify the system type $T$ from the taxonomy.
2. Fill in the four Thin Objects: Arena, Potential, Cost, Invariance.
3. For each, implement the required interface templates from Part 0.

**Phase 2: Sieve Execution (Part II)**
1. Execute nodes 1-12 (core) sequentially.
2. At each node, follow the decision flowchart (A.2).
3. Record every certificate in Γ (accumulating context).
4. Record every inc obligation in the ledger.

**Phase 3: Boundary Nodes (if open system)**
1. Execute nodes 13-16.
2. If system is closed, skip to Lock.

**Phase 4: Lock (Node 17)**
1. Construct universal bad pattern $\mathcal{H}_{\text{bad}}$.
2. Apply exclusion tactics E1-E10 until one succeeds.
3. Emit Lock verdict.

**Phase 5: Upgrade Pass (Part II-B)**
1. Scan Γ for all $K^{\mathrm{inc}}$.
2. Apply all applicable upgrade rules.
3. Update obligation ledger.

**Phase 6: Surgery (if needed, Part II-C)**
1. If any barriers breached, execute surgery protocol.
2. Re-enter and discharge obligations.

**Phase 7: Closure (Part III)**
1. If Lyapunov reconstruction conditions met, construct $\mathcal{L}$.
2. Extract results via metatheorems.
3. Apply retroactive upgrades.

**Phase 8: Finalization (Part IV)**
1. Verify completion criteria.
2. Assemble final certificate chain.
3. Document proof summary.

**Phase 0: Dashboard Generation (Post-Run)**
*Once Phase 8 is complete, return to the top and fill the Executive Summary.*
1. Fill the **System Instantiation** table with the Arena, Potential, Cost, and Invariance definitions.
2. Fill the **Execution Trace** table using data from your completed run.
   - **Crucial:** The table must match the `trace.json` logic exactly.
   - If you performed surgery, show the branching rows clearly (use `--` for Surgery/Re-entry rows).
3. Fill the **Lock Mechanism** table showing which tactics E1-E10 were attempted and their outcomes.
4. Fill the **Final Verdict** with Status, Obligation Ledger state, and Singularity Set description.

:::

---

## **Part 0: Interface Permit Implementation Checklist**
*Complete this section before running the Sieve. Each permit requires specific mathematical structures to be defined.*

### **0.1 Core Interface Permits (Nodes 1-12)**

| #  | Permit ID                  | Node           | Question                 | Required Implementation                                                   | Certificate                          |
|----|----------------------------|----------------|--------------------------|---------------------------------------------------------------------------|--------------------------------------|
| 1  | $D_E$                      | EnergyCheck    | Is Energy Finite?        | Height $\Phi$, Dissipation $\mathfrak{D}$, Bound $B$                      | $K_{D_E}^{\pm}$                      |
| 2  | $\mathrm{Rec}_N$           | ZenoCheck      | Are Events Finite?       | Bad set $\mathcal{B}$, Recovery map $\mathcal{R}$, Counter $\#$           | $K_{\mathrm{Rec}_N}^{\pm}$           |
| 3  | $C_\mu$                    | CompactCheck   | Does Energy Concentrate? | Symmetry $G$, Quotient $\mathcal{X}//G$, Limit $\lim$                     | $K_{C_\mu}^{\pm}$                    |
| 4  | $\mathrm{SC}_\lambda$      | ScaleCheck     | Is Profile Subcritical?  | Scaling action $\mathbb{G}_m$, Exponents $\alpha,\beta$                   | $K_{\mathrm{SC}_\lambda}^{\pm}$      |
| 5  | $\mathrm{SC}_{\partial c}$ | ParamCheck     | Are Constants Stable?    | Parameters $\Theta$, Reference $\theta_0$, Distance $d$                   | $K_{\mathrm{SC}_{\partial c}}^{\pm}$ |
| 6  | $\mathrm{Cap}_H$           | GeomCheck      | Is Codim $\geq 2$?       | Capacity $\text{Cap}$, Threshold $C_{\text{crit}}$, Singular set $\Sigma$ | $K_{\mathrm{Cap}_H}^{\pm}$           |
| 7  | $\mathrm{LS}_\sigma$       | StiffnessCheck | Is Gap Certified?        | Gradient $\nabla$, Łoj-Simon exponent $\theta$                            | $K_{\mathrm{LS}_\sigma}^{\pm}$       |
| 8  | $\mathrm{TB}_\pi$          | TopoCheck      | Is Sector Preserved?     | Components $\pi_0(\mathcal{X})$, Sector map $\tau$                        | $K_{\mathrm{TB}_\pi}^{\pm}$          |
| 9  | $\mathrm{TB}_O$            | TameCheck      | Is Topology Tame?        | O-minimal structure $\mathcal{O}$, Definability $\text{Def}$              | $K_{\mathrm{TB}_O}^{\pm}$            |
| 10 | $\mathrm{TB}_\rho$         | ErgoCheck      | Does Flow Mix?           | Measure $\mathcal{M}$, Mixing time $\tau_{\text{mix}}$                    | $K_{\mathrm{TB}_\rho}^{\pm}$         |
| 11 | $\mathrm{Rep}_K$           | ComplexCheck   | Is Description Finite?   | Language $\mathcal{L}$, Dictionary $D$, Complexity $K$                    | $K_{\mathrm{Rep}_K}^{\pm}$           |
| 12 | $\mathrm{GC}_\nabla$       | OscillateCheck | Is Flow Gradient?        | Metric $g$, Compatibility $v \sim -\nabla\Phi$                            | $K_{\mathrm{GC}_\nabla}^{\pm}$       |

### **0.2 Boundary Interface Permits (Nodes 13-16)**
*For open systems with inputs/outputs. Skip if system is closed.*

| # | Permit ID | Node | Question | Required Implementation | Certificate |
|---|-----------|------|----------|------------------------|-------------|
| 13 | $\mathrm{Bound}_\partial$ | BoundaryCheck | Is System Open? | Input $\mathcal{U}$, Output $\mathcal{Y}$, Maps $\iota, \pi$ | $K_{\mathrm{Bound}_\partial}^{\pm}$ |
| 14 | $\mathrm{Bound}_B$ | OverloadCheck | Is Input Bounded? | Authority bound $\mathcal{C}$ | $K_{\mathrm{Bound}_B}^{\pm}$ |
| 15 | $\mathrm{Bound}_{\Sigma}$ | StarveCheck | Is Input Sufficient? | Minimum $u_{\min}$ | $K_{\mathrm{Bound}_{\Sigma}}^{\pm}$ |
| 16 | $\mathrm{GC}_T$ | AlignCheck | Is Control Matched? | Alignment $\langle \iota(u), \nabla\Phi \rangle$ | $K_{\mathrm{GC}_T}^{\pm}$ |

### **0.2b Derived Witness Certificates (Optional)**
*These are **not** gate permits. They are payload certificates derived from the gate results or supplied explicitly.*

| Certificate | Derived From | Payload | Notes |
|---|---|---|---|
| $K_{D_{\max}}^+$ | $C_\mu^+$ + boundary/overload/starve permits | $(D_{\max},\ \text{support/diameter proof})$ | Bounded algorithmic diameter on the alive core |
| $K_{\rho_{\max}}^+$ | $K_{\mathrm{TB}_\rho}^+ + K_{\mathrm{Cap}_H}^+ + K_{D_E}^+$ | $(\rho_{\max},\ \text{density bound proof})$ | Uniform invariant/QSD density bound on the alive core |

**Record any witness certificates** in the Execution Trace payloads; if they are used to
justify analytic bridge admissibility, cite them explicitly in the Lock Mechanism section.

### **0.3 The Lock (Node 17)**

| Permit ID | Node | Question | Required Implementation | Certificate |
|-----------|------|----------|------------------------|-------------|
| $\mathrm{Cat}_{\mathrm{Hom}}$ | Lock | Is $\text{Hom}(\mathcal{H}_{\text{bad}}, \mathcal{H}) = \emptyset$? | Category $\mathbf{Hypo}_T$, Universal bad $\mathcal{H}_{\text{bad}}$, Tactics E1-E10 | $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk/morph}}$ |

### **0.4 Implementation Templates**

#### **Template: $D_E$ (Energy Interface)**
- [ ] **Height Functional $\Phi$:** [Define $\Phi: \mathcal{X} \to \mathcal{H}$]
- [ ] **Dissipation Rate $\mathfrak{D}$:** [Define $\mathfrak{D}: \mathcal{X} \to \mathcal{H}$]
- [ ] **Energy Inequality:** $\Phi(S_t x) \leq \Phi(x) + \int_0^t \mathfrak{D}(S_s x) ds$
- [ ] **Bound Witness:** $B = $ [Insert explicit bound or $\infty$]

#### **Template: Derived Witness Certificates (Optional)**
- [ ] **$K_{D_{\max}}^+$ (diameter witness):** Provide $D_{\max}$ and a support/diameter proof
      derived from $C_\mu^+$ and boundary/overload/starve permits.
- [ ] **$K_{\rho_{\max}}^+$ (density witness):** Provide $\rho_{\max}$ and a density-bound proof
      derived from $K_{\mathrm{TB}_\rho}^+$, $K_{\mathrm{Cap}_H}^+$, and $K_{D_E}^+$.

#### **Template: $\mathrm{Rec}_N$ (Recovery Interface)**
- [ ] **Bad Set $\mathcal{B}$:** [Define $\mathcal{B} \hookrightarrow \mathcal{X}$]
- [ ] **Recovery Map $\mathcal{R}$:** [Define $\mathcal{R}: \mathcal{B} \to \mathcal{X} \setminus \mathcal{B}$]
- [ ] **Event Counter $\#$:** [Define counting measure]
- [ ] **Finiteness:** $\#\{t : S_t(x) \in \mathcal{B}\} < \infty$?

#### **Template: $C_\mu$ (Compactness Interface)**
- [ ] **Symmetry Group $G$:** [Define group structure]
- [ ] **Group Action $\rho$:** [Define $\rho: G \times \mathcal{X} \to \mathcal{X}$]
- [ ] **Quotient Space:** $\mathcal{X} // G = $ [Define moduli space]
- [ ] **Concentration Measure:** [Define how energy concentrates]

#### **Template: $\mathrm{SC}_\lambda$ (Scaling Interface)**
- [ ] **Scaling Action:** $\mathcal{S}_\lambda: \mathcal{X} \to \mathcal{X}$
- [ ] **Height Exponent $\alpha$:** $\Phi(\mathcal{S}_\lambda x) = \lambda^\alpha \Phi(x)$, $\alpha = $ [Value]
- [ ] **Dissipation Exponent $\beta$:** $\mathfrak{D}(\mathcal{S}_\lambda x) = \lambda^\beta \mathfrak{D}(x)$, $\beta = $ [Value]
- [ ] **Criticality:** $\beta - \alpha = $ [Value] (< $\lambda_c$ subcritical, = $\lambda_c$ critical, > $\lambda_c$ supercritical; $\lambda_c = 0$ in the homogeneous case)

#### **Template: $\mathrm{SC}_{\partial c}$ (Parameter Interface)**
- [ ] **Parameter Space $\Theta$:** [Define parameter object]
- [ ] **Parameter Map $\theta$:** [Define $\theta: \mathcal{X} \to \Theta$]
- [ ] **Reference Point $\theta_0$:** [Define reference parameters]
- [ ] **Stability Bound:** $d(\theta(S_t x), \theta_0) \leq C$ for all $t$?

#### **Template: $\mathrm{Cap}_H$ (Capacity Interface)**
- [ ] **Capacity Functional:** $\text{Cap}: \text{Sub}(\mathcal{X}) \to [0, \infty]$
- [ ] **Singular Set $\Sigma$:** [Define where singularities can occur]
- [ ] **Codimension:** $\text{codim}(\Sigma) = $ [Value] (need $\geq 2$ for removability)
- [ ] **Capacity Bound:** $\text{Cap}(\Sigma) \leq $ [Value]

#### **Template: $\mathrm{LS}_\sigma$ (Stiffness Interface)**
- [ ] **Gradient Operator $\nabla$:** [Define gradient structure]
- [ ] **Critical Set $M$:** [Define where $\nabla\Phi = 0$]
- [ ] **Łojasiewicz Exponent $\theta$:** [Value in $(0, 1]$]
- [ ] **Łojasiewicz-Simon Inequality:** $\|\nabla\Phi(x)\| \geq C|\Phi(x) - \Phi(V)|^{1-\theta}$

#### **Template: $\mathrm{TB}_\pi$ (Topology Interface)**
- [ ] **Topological Invariant $\tau$:** [Define $\tau: \mathcal{X} \to \pi_0(\mathcal{X})$]
- [ ] **Sector Classification:** [List all sectors]
- [ ] **Sector Preservation:** $\tau(S_t x) = \tau(x)$ for all $t$?
- [ ] **Tunneling Events:** [Define how sectors can change]

#### **Template: $\mathrm{TB}_O$ (Tameness Interface)**
- [ ] **O-minimal Structure $\mathcal{O}$:** [Define tame structure]
- [ ] **Definability $\text{Def}$:** [Define $\text{Def}: \text{Sub}(\mathcal{X}) \to \Omega$]
- [ ] **Singular Set Tameness:** $\Sigma \in \mathcal{O}\text{-definable}$?
- [ ] **Cell Decomposition:** [Verify finite stratification]

#### **Template: $\mathrm{TB}_\rho$ (Mixing Interface)**
- [ ] **Measure $\mathcal{M}$:** [Define measure object $\mathcal{M}(\mathcal{X})$]
- [ ] **Invariant Measure $\mu$:** [Define $\mu \in \text{Inv}_S$]
- [ ] **Mixing Time $\tau_{\text{mix}}$:** [Define $\tau_{\text{mix}}: \mathcal{X} \to \mathcal{H}$]
- [ ] **Mixing Property:** $\tau_{\text{mix}}(x) < \infty$?

#### **Template: $\mathrm{Rep}_K$ (Dictionary Interface)**
- [ ] **Language $\mathcal{L}$:** [Formal language for descriptions]
- [ ] **Dictionary $D$:** [Define $D: \mathcal{X} \to \mathcal{L}$]
- [ ] **Complexity Measure $K$:** [Define $K: \mathcal{L} \to \mathbb{N}_\infty$]
- [ ] **Faithfulness:** $D$ is injective on relevant states?

#### **Template: $\mathrm{GC}_\nabla$ (Gradient Interface)**
- [ ] **Metric Tensor $g$:** [Define $g: T\mathcal{X} \otimes T\mathcal{X} \to \mathcal{H}$]
- [ ] **Vector Field $v$:** [Define flow vector field]
- [ ] **Gradient Compatibility:** $v = -\nabla_g \Phi$?
- [ ] **Monotonicity:** $\mathfrak{D}(x) = \|\nabla_g \Phi(x)\|^2$?

#### **Template: $\mathrm{Cat}_{\mathrm{Hom}}$ (Lock Interface)**
- [ ] **Category $\mathbf{Hypo}_T$:** [Define morphisms in hypostructure category]
- [ ] **Universal Bad Pattern $\mathcal{H}_{\text{bad}}$:** [Construct worst-case singularity]
- [ ] **Primary Tactic Selected:** [e.g., E2, E7, etc.]
- [ ] **Tactic Logic:**
    * $I(\mathcal{H}) = $ [Value for the actual hypostructure]
    * $I(\mathcal{H}_{\text{bad}}) = $ [Value for the bad pattern]
    * Conclusion: Mismatch $\implies$ $\mathrm{Hom} = \emptyset$.
- [ ] **Exclusion Tactics Available:**
  - [ ] E1 (Dimension): $\dim(\mathcal{H}_{\text{bad}}) \neq \dim(\mathcal{H})$?
  - [ ] E2 (Invariant): $I(\mathcal{H}_{\text{bad}}) \neq I(\mathcal{H})$?
  - [ ] E3 (Positivity): Cone violation?
  - [ ] E4 (Integrality): Arithmetic obstruction?
  - [ ] E5 (Functional): Unsolvable equations?
  - [ ] E6 (Causal): Well-foundedness violation?
  - [ ] E7 (Thermodynamic): Entropy violation?
  - [ ] E8 (Holographic): Bekenstein bound violation?
  - [ ] E9 (Ergodic): Mixing incompatibility?
  - [ ] E10 (Definability): Tameness violation?

---

:::{dropdown} **Part 0.5: Certificate Schemas and Upgrade Protocol** (Reference - Click to expand)

*Reference: For formal definitions, see `hypopermits_jb.md` Definitions `def-typed-no-certificates`, `def-inc-upgrades`, `def-promotion-permits`, `def-closure`.*

### **0.5.1 Certificate Schemas**

Every node emits one of these certificate types:

#### **Positive Certificate ($K_X^+$)**
```
K_X^+ = (witness_1, witness_2, ..., witness_n)
```
**Contents:** Explicit witnesses that verify the predicate. These are mathematical objects (bounds, functions, exponents) that can be checked independently.

**Example:** $K_{D_E}^+ = (\Phi, \mathfrak{D}, B)$ where $B$ is the explicit energy bound.

#### **NO-with-Witness Certificate ($K_X^{\mathrm{wit}}$)**
```
K_X^{wit} = (counterexample, reason)
```
**Contents:** Evidence that the predicate fails. Triggers barrier check.

**Example:** $K_{\mathrm{SC}_\lambda}^{\mathrm{wit}} = (\beta - \alpha = 1, \text{"supercritical"})$

#### **NO-Inconclusive Certificate ($K_X^{\mathrm{inc}}$)**
```
K_X^{inc} = {
  obligation: "What must be established to resolve",
  missing: ["K_A", "K_B", ...],  // certificates needed
  failure_code: "STANDARDIZED_CODE",
  trace: "Step N of NodeCheck where gap occurred"
}
```
**Contents:** Structured payload documenting exactly what is missing. This certificate is **recoverable** via upgrade rules.

**Example:**
```
K_{NL}^{inc} = {
  obligation: "Close H¹ differential inequality",
  missing: ["K_Emb^+", "K_SG^+"],
  failure_code: "MISSING_EMBEDDING",
  trace: "Step 4 of NonlinearityControl"
}
```

*Note:* The overall NO certificate is the sum type
$K_X^- := K_X^{\mathrm{wit}} \sqcup K_X^{\mathrm{inc}}$ (Definition {prf:ref}`def-typed-no-certificates`).

#### **Blocked Certificate ($K_X^{\mathrm{blk}}$)**
```
K_X^{blk} = (barrier_id, blocking_reason, blocking_certificates)
```
**Contents:** Evidence that the barrier prevents the bad scenario. The predicate failed ($K^{\mathrm{wit}}$) but the failure mode is excluded.

**Example:** $K_{\text{Sat}}^{\mathrm{blk}} = (\text{BarrierSat}, \text{"drift controlled by saturation"}, \{K_{D_E}^+\})$

#### **Breached Certificate ($K_X^{\mathrm{br}}$)**
```
K_X^{br} = {
  barrier_id: "Which barrier failed",
  reason: "Why it couldn't block",
  obligations: ["O1", "O2", ...]  // what surgery must provide
}
```
**Contents:** Documents barrier failure and what must be recovered via surgery.

---

### **0.5.2 Upgrade Rule Schema**

Upgrade rules convert $K^{\mathrm{inc}}$ to $K^+$ when prerequisites are satisfied.

#### **Rule Template**
```
U_{X→+}(premises, target, guard):
  IF   K_X^{inc} ∈ Γ
  AND  ∀ m ∈ missing(K_X^{inc}): K_m^+ ∈ Γ
  AND  K_X^+ ∉ depends(premises)   // non-circularity
  THEN Γ := Γ ∪ {K_X^+}
       discharge(obligation(K_X^{inc}))
```

#### **Non-Circularity Guard**
The upgrade is **invalid** if any premise certificate depends on the target $K_X^+$. This prevents circular reasoning.

**Check:** For each premise $K_m^+$, trace its derivation. If $K_X^+$ appears anywhere in that derivation, the upgrade is blocked.

#### **Upgrade Types**

| Type | When Applied | Premises From |
|------|--------------|---------------|
| **Instantaneous** | Same sieve pass | Earlier nodes |
| **A-posteriori** | After surgery or later nodes | Later nodes or surgery |

---

### **0.5.2b Promotion Permits (Blocked → YES$^\sim$)**

Blocked certificates can be **promoted** when the barrier result plus supporting certificates logically implies the original predicate (Definition {prf:ref}`def-promotion-permits`). Use the YES$^\sim$ form when the promotion is valid only up to an admissible equivalence (Definition {prf:ref}`def-yes-tilde`). Keep the original $K^{\mathrm{blk}}$ as an audit trail and record the promotion witness alongside it.

### **0.5.3 Surgery Certificate Schema**

When a barrier is breached, surgery changes the representation.

```
K_Surg^+ = {
  map_id: "Transformation name (e.g., Curl2D)",
  source: "Original representation",
  target: "New representation",
  preservation: "Proof that theorem is unchanged",
  recovery: "How to translate back (if needed)"
}
```

**Example (2D NS):**
```
K_Surg^+(Curl2D) = {
  map_id: "Curl2D",
  source: "velocity (u, p)",
  target: "vorticity ω = curl(u)",
  preservation: "Biot-Savart recovers u from ω",
  recovery: "u = ∇⊥(-Δ)^{-1}ω"
}
```

---

### **0.5.4 Re-entry Certificate Schema**

After surgery, re-entry certificates discharge breach obligations.

```
K_re^+(item) = {
  discharged: "Which obligation/missing item",
  via: "How it was established post-surgery",
  certificates: ["K_A^+", "K_B^+", ...]  // supporting certs
}
```

**Example:**
```
K_re^+(GradBound) = {
  discharged: "Control |∇u|_2 on [0,T]",
  via: "Enstrophy bound + Biot-Savart",
  certificates: ["K_Ens^+", "K_BS^+"]
}
```

---

### **0.5.5 Context Accumulation**

The **context** Γ accumulates certificates throughout the run:

$$\Gamma_0 = \{K_{\text{Init}}^+\}$$
$$\Gamma_{n+1} = \Gamma_n \cup \{\text{certificate from Node } n+1\}$$

The **promotion closure** $\mathrm{Cl}(\Gamma)$ applies all upgrade rules until fixed point:
$$\mathrm{Cl}(\Gamma) = \bigcup_{k=0}^{\infty} \Gamma_k$$
where $\Gamma_{k+1}$ applies all valid promotions and upgrades to $\Gamma_k$.

:::

---

## **Part I: The Instantiation (Thin Object Definitions)**
*User Input: Define the four "Thin Objects" (Section 8.C). The Factory Metatheorems (TM-1 to TM-4) automatically expand these into the full Kernel Objects.*

### **1. The Arena ($\mathcal{X}^{\text{thin}}$)**
*Implements: $\mathcal{H}_0$, $\mathrm{Cap}_H$, $\mathrm{TB}_\pi$, $\mathrm{TB}_O$, $\mathrm{Rep}_K$ — see Section 0.4 for templates*
* **State Space ($\mathcal{X}$):** [The set of all possible states, e.g., $H^1$ functions, Graphs, Inputs]
* **Metric ($d$):** [Distance function, e.g., $L^2$ norm, Edit distance]
* **Measure ($\mu$):** [Reference measure, e.g., Lebesgue, Counting]
    * *Framework Derivation:* Capacity Functional $\text{Cap}(\Sigma)$ via RESOLVE-AutoAdmit.

### **2. The Potential ($\Phi^{\text{thin}}$)**
*Implements: $D_E$, $\mathrm{SC}_\lambda$, $\mathrm{LS}_\sigma$ — see Section 0.4 for templates*
* **Height Functional ($F$):** [Quantity to optimize/bound, e.g., Energy, Entropy, Loss]
* **Gradient/Slope ($\nabla$):** [Local descent direction, e.g., $-\Delta u$, Gradient Descent]
* **Scaling Exponent ($\alpha$):** [How $F$ scales: $F(\lambda x) = \lambda^\alpha F(x)$]
    * *Framework Derivation:* EnergyCheck, ScaleCheck, StiffnessCheck verifiers.

### **3. The Cost ($\mathfrak{D}^{\text{thin}}$)**
*Implements: $\mathrm{Rec}_N$, $\mathrm{GC}_\nabla$, $\mathrm{TB}_\rho$ — see Section 0.4 for templates*
* **Dissipation Rate ($R$):** [Rate of progress/loss, e.g., $\int |\nabla u|^2$, Information Gain]
* **Scaling Exponent ($\beta$):** [How $R$ scales: $R(\lambda x) = \lambda^\beta R(x)$]
    * *Framework Derivation:* Singular Locus $\Sigma = \{x : R(x) \to \infty\}$.

### **4. The Invariance ($G^{\text{thin}}$)**
*Implements: $C_\mu$, $\mathrm{SC}_{\partial c}$ — see Section 0.4 for templates*
* **Symmetry Group ($\text{Grp}$):** [Inherent symmetries, e.g., Rotation, Permutation, Gauge]
* **Action ($\rho$):** [How the group transforms the state]
* **Scaling Subgroup ($\mathcal{S}$):** [Dilations / Renormalization action]
    * *Framework Derivation:* Profile Library $\mathcal{L}_T$ via RESOLVE-AutoProfile (Profile Trichotomy).

---

## **Part II: Sieve Execution (Verification Run)**
*Execute the Canonical Sieve Algorithm node-by-node. At each node, perform the specified check and record the certificate.*

### **EXECUTION PROTOCOL**

For each node:
1. **Read** the interface permit question
2. **Check** the predicate using textbook definitions
3. **Record** the certificate: $K^+$ (yes), $K^{\mathrm{wit}}$ (no with witness), $K^{\mathrm{inc}}$ (no-inconclusive), or $K^{\mathrm{blk}}$ (barrier blocked)
4. **Follow** the flowchart to the next node

---

### **Level 1: Conservation**

#### **Node 1: EnergyCheck ($D_E$)**

**Question:** Is the height functional $\Phi$ bounded along trajectories?

**Step-by-step execution:**
1. [ ] Write down the energy inequality: $\Phi(S_t x) \leq \Phi(x) - \int_0^t \mathfrak{D}(S_s x)\, ds + C \cdot t$
2. [ ] Check: Is $C = 0$ (strict dissipation) or $C > 0$ (drift)?
3. [ ] If $C = 0$: Is $\Phi$ bounded below? Check $\Phi_{\min} > -\infty$.
4. [ ] If $C > 0$: Go to BarrierSat (check if drift is controlled by saturation).

**Certificate:**
* [ ] $K_{D_E}^+ = (\Phi, \mathfrak{D}, B)$ where $B$ is the explicit bound → **Go to Node 2**
* [ ] $K_{D_E}^{\mathrm{wit}}$ → Check BarrierSat
  * [ ] $K_{\text{sat}}^{\mathrm{blk}}$: Drift controlled → **Go to Node 2**
  * [ ] Breached: Enable Surgery `SurgCE`
* [ ] $K_{D_E}^{\mathrm{inc}}$ → **Record obligation, Go to Node 2**
  ```
  { obligation: "Establish energy bound B",
    missing: ["explicit_bound", "dissipation_rate"],
    failure_code: "MISSING_ENERGY_BOUND",
    trace: "Step 1-3 of EnergyCheck" }
  ```

---

#### **Node 2: ZenoCheck ($\mathrm{Rec}_N$)**

**Question:** Does the trajectory visit the bad set $\mathcal{B}$ only finitely many times?

**Step-by-step execution:**
1. [ ] Define the bad set: $\mathcal{B} = \{x \in \mathcal{X} : \text{[singular condition]}\}$
2. [ ] Define the recovery map: $\mathcal{R}: \mathcal{B} \to \mathcal{X} \setminus \mathcal{B}$
3. [ ] Count bad events: $N(T) = \#\{t \in [0,T] : S_t x \in \mathcal{B}\}$
4. [ ] Check: Is $\sup_T N(T) < \infty$?

**Certificate:**
* [ ] $K_{\mathrm{Rec}_N}^+ = (\mathcal{B}, \mathcal{R}, N_{\max})$ → **Go to Node 3**
* [ ] $K_{\mathrm{Rec}_N}^-$ → Check BarrierCausal
  * [ ] $K^{\mathrm{blk}}$: Depth censored → **Go to Node 3**
  * [ ] Breached: Enable Surgery `SurgCC`
* [ ] $K_{\mathrm{Rec}_N}^{\mathrm{inc}}$ → **Record obligation, Go to Node 3**
  ```
  { obligation: "Prove finite bad-event count",
    missing: ["recovery_map", "event_bound"],
    failure_code: "MISSING_RECOVERY",
    trace: "Step 2-4 of ZenoCheck" }
  ```

---

#### **Node 3: CompactCheck ($C_\mu$)**

**Question:** Do sublevel sets of $\Phi$ have compact closure modulo symmetry?

**Step-by-step execution:**
1. [ ] Define sublevel set: $\{\Phi \leq E\} = \{x \in \mathcal{X} : \Phi(x) \leq E\}$
2. [ ] Identify symmetry group $G$ acting on $\mathcal{X}$
3. [ ] Form quotient: $\{\Phi \leq E\} / G$
4. [ ] Check: Is quotient precompact? (Bounded sequences have convergent subsequences)
5. [ ] If YES: Profile decomposition exists (energy concentrates)
6. [ ] If NO: Check if scattering (energy disperses to infinity)

**Certificate:**
* [ ] $K_{C_\mu}^+ = (G, \mathcal{X}//G, \lim)$ → **Profile Emerges. Go to Node 4**
* [ ] $K_{C_\mu}^-$ → Check BarrierScat
  * [ ] Benign scattering: **VICTORY (Mode D.D) - Global Existence**
  * [ ] Pathological: Enable Surgery `SurgCD_Alt`
* [ ] $K_{C_\mu}^{\mathrm{inc}}$ → **Record obligation, Go to Node 4**
  ```
  { obligation: "Establish compactness modulo symmetry",
    missing: ["symmetry_group", "profile_decomposition"],
    failure_code: "MISSING_COMPACTNESS",
    trace: "Step 3-4 of CompactCheck" }
  ```

---

### **Level 2: Duality & Symmetry**

#### **Node 4: ScaleCheck ($\mathrm{SC}_\lambda$)**

**Question:** Is the scaling exponent subcritical ($\beta - \alpha < \lambda_c$)?

**Step-by-step execution:**
1. [ ] Compute height scaling: $\Phi(\mathcal{S}_\lambda x) = \lambda^\alpha \Phi(x)$. Record $\alpha = $ ____
2. [ ] Compute dissipation scaling: $\mathfrak{D}(\mathcal{S}_\lambda x) = \lambda^\beta \mathfrak{D}(x)$. Record $\beta = $ ____
3. [ ] Compute criticality: $\beta - \alpha = $ ____
4. [ ] Classify:
   - $\beta - \alpha < \lambda_c$: Subcritical (singularities cost infinite energy)
   - $\beta - \alpha = \lambda_c$: Critical (borderline)
   - $\beta - \alpha > \lambda_c$: Supercritical (singularities can form)

**Certificate:**
* [ ] $K_{\mathrm{SC}_\lambda}^+ = (\alpha, \beta, \lambda_c, \beta - \alpha < \lambda_c)$ → **Go to Node 5**
* [ ] $K_{\mathrm{SC}_\lambda}^-$ → Check BarrierTypeII
  * [ ] $K^{\mathrm{blk}}$: Renorm cost infinite → **Go to Node 5**
  * [ ] Breached: Enable Surgery `SurgSE`
* [ ] $K_{\mathrm{SC}_\lambda}^{\mathrm{inc}}$ → **Record obligation, Go to Node 5**
  ```
  { obligation: "Determine scaling exponents α, β",
    missing: ["scaling_action", "exponent_computation"],
    failure_code: "MISSING_SCALING",
    trace: "Step 1-3 of ScaleCheck" }
  ```

---

#### **Node 5: ParamCheck ($\mathrm{SC}_{\partial c}$)**

**Question:** Are physical constants stable under the flow?

**Step-by-step execution:**
1. [ ] Identify parameters: $\Theta = \{$constants appearing in the equation$\}$
2. [ ] Define parameter map: $\theta: \mathcal{X} \to \Theta$
3. [ ] Pick reference: $\theta_0 \in \Theta$ (e.g., initial/asymptotic values)
4. [ ] Check stability: $d(\theta(S_t x), \theta_0) \leq C$ for all $t$?

**Certificate:**
* [ ] $K_{\mathrm{SC}_{\partial c}}^+ = (\Theta, \theta_0, C)$ → **Go to Node 6**
* [ ] $K_{\mathrm{SC}_{\partial c}}^-$ → Check BarrierVac
  * [ ] $K^{\mathrm{blk}}$: Phase stable → **Go to Node 6**
  * [ ] Breached: Enable Surgery `SurgSC`
* [ ] $K_{\mathrm{SC}_{\partial c}}^{\mathrm{inc}}$ → **Record obligation, Go to Node 6**
  ```
  { obligation: "Establish parameter stability",
    missing: ["parameter_space", "stability_bound"],
    failure_code: "MISSING_PARAM_STABILITY",
    trace: "Step 3-4 of ParamCheck" }
  ```

---

### **Level 3: Geometry & Stiffness**

#### **Node 6: GeomCheck ($\mathrm{Cap}_H$)**

**Question:** Is the singular set small (codimension $\geq 2$)?

**Step-by-step execution:**
1. [ ] Define singular set: $\Sigma = \{x : \mathfrak{D}(x) = \infty$ or $\Phi$ undefined$\}$
2. [ ] Compute Hausdorff dimension: $\dim_{\mathcal{H}}(\Sigma) = $ ____
3. [ ] Compute ambient dimension: $\dim(\mathcal{X}) = $ ____
4. [ ] Check codimension: $\text{codim}(\Sigma) = \dim(\mathcal{X}) - \dim_{\mathcal{H}}(\Sigma) \geq 2$?
5. [ ] Alternative: Check capacity $\text{Cap}(\Sigma) = 0$?

**Certificate:**
* [ ] $K_{\mathrm{Cap}_H}^+ = (\Sigma, \dim_{\mathcal{H}}(\Sigma), \text{codim} \geq 2)$ → **Go to Node 7**
* [ ] $K_{\mathrm{Cap}_H}^-$ → Check BarrierCap
  * [ ] $K^{\mathrm{blk}}$: Capacity zero → **Go to Node 7**
  * [ ] Breached: Enable Surgery `SurgCD`
* [ ] $K_{\mathrm{Cap}_H}^{\mathrm{inc}}$ → **Record obligation, Go to Node 7**
  ```
  { obligation: "Establish Hausdorff dimension of singular set",
    missing: ["capacity_estimate", "removability_theorem"],
    failure_code: "MISSING_CAPACITY_BOUND",
    trace: "Step 2-4 of GeomCheck" }
  ```

---

#### **Node 7: StiffnessCheck ($\mathrm{LS}_\sigma$)**

**Question:** Does the Łojasiewicz-Simon inequality hold near critical points?

**Step-by-step execution:**
1. [ ] Identify critical set: $M = \{x : \nabla\Phi(x) = 0\}$
2. [ ] Near $M$, check Łojasiewicz inequality: $\|\nabla\Phi(x)\| \geq c|\Phi(x) - \Phi_{\min}|^{1-\theta}$
3. [ ] Find exponent: $\theta \in (0, 1]$. Record $\theta = $ ____
4. [ ] This implies convergence rate: $\mathrm{dist}(S_t x, M) \lesssim t^{-\theta/(1-\theta)}$

**Certificate:**
* [ ] $K_{\mathrm{LS}_\sigma}^+ = (M, \theta, c)$ → **Go to Node 8**
* [ ] $K_{\mathrm{LS}_\sigma}^-$ → Check BarrierGap
  * [ ] $K^{\mathrm{blk}}$: Gap exists → **Go to Node 8**
  * [ ] Stagnation: Enter Restoration Subtree (SymCheck → SSB)
* [ ] $K_{\mathrm{LS}_\sigma}^{\mathrm{inc}}$ → **Record obligation, Go to Node 8**
  ```
  { obligation: "Establish Łojasiewicz-Simon inequality",
    missing: ["critical_set", "lojasiewicz_exponent"],
    failure_code: "MISSING_STIFFNESS",
    trace: "Step 2-3 of StiffnessCheck" }
  ```

---

### **Level 4: Topology**

#### **Node 8: TopoCheck ($\mathrm{TB}_\pi$)**

**Question:** Is the topological sector accessible (trajectory stays in reachable sector)?

**Step-by-step execution:**
1. [ ] Identify topological invariant: $\tau: \mathcal{X} \to \pi_0(\mathcal{X})$ (e.g., winding number, homotopy class, degree)
2. [ ] List all sectors: $\pi_0(\mathcal{X}) = \{$sector labels$\}$
3. [ ] Determine initial sector: $\tau(x_0) = $ ____
4. [ ] Check sector preservation: $\tau(S_t x) = \tau(x)$ for all $t$?
5. [ ] If NO: Identify obstructing barrier (action gap $\Delta E$ required to tunnel)

**Certificate:**
* [ ] $K_{\mathrm{TB}_\pi}^+ = (\tau, \pi_0(\mathcal{X}), \text{sector preservation proof})$ → **Go to Node 9**
* [ ] $K_{\mathrm{TB}_\pi}^-$ → Check BarrierAction
  * [ ] $K^{\mathrm{blk}}$: Energy below action gap → **Go to Node 9**
  * [ ] Breached: Enable Surgery `SurgTE` (Tunnel)
* [ ] $K_{\mathrm{TB}_\pi}^{\mathrm{inc}}$ → **Record obligation, Go to Node 9**
  ```
  { obligation: "Determine sector preservation",
    missing: ["topological_invariant", "sector_map"],
    failure_code: "MISSING_TOPOLOGY",
    trace: "Step 1-4 of TopoCheck" }
  ```

---

#### **Node 9: TameCheck ($\mathrm{TB}_O$)**

**Question:** Is the singular locus tame (definable in an o-minimal structure)?

**Step-by-step execution:**
1. [ ] Identify singular set: $\Sigma = \{x : \text{singularity condition}\}$
2. [ ] Choose o-minimal structure: $\mathcal{O} = $ ____ (e.g., $\mathbb{R}_{\text{an}}$, $\mathbb{R}_{\text{exp}}$, semialgebraic)
3. [ ] Check definability: Is $\Sigma$ definable in $\mathcal{O}$?
   - Semialgebraic: Defined by polynomial inequalities?
   - Subanalytic: Locally finite Boolean combination of analytic sets?
4. [ ] If definable: Whitney stratification exists with finitely many strata
5. [ ] If NOT definable: Wild topology (Cantor-like, fractal structure)

**Certificate:**
* [ ] $K_{\mathrm{TB}_O}^+ = (\mathcal{O}, \Sigma \in \mathcal{O}\text{-def}, \text{stratification})$ → **Go to Node 10**
* [ ] $K_{\mathrm{TB}_O}^-$ → Check BarrierOmin
  * [ ] $K^{\mathrm{blk}}$: Definability recoverable → **Go to Node 10**
  * [ ] Breached: Enable Surgery `SurgTC` (O-minimal Regularization)
* [ ] $K_{\mathrm{TB}_O}^{\mathrm{inc}}$ → **Record obligation, Go to Node 10**
  ```
  { obligation: "Establish o-minimal definability of singular set",
    missing: ["o_minimal_structure", "definability_proof"],
    failure_code: "MISSING_TAMENESS",
    trace: "Step 2-4 of TameCheck" }
  ```

---

### **Level 5: Mixing**

#### **Node 10: ErgoCheck ($\mathrm{TB}_\rho$)**

**Question:** Does the flow mix (ergodic with finite mixing time)?

**Step-by-step execution:**
1. [ ] Identify invariant measure: $\mu \in \text{Inv}_S$ (e.g., Liouville, Gibbs, uniform)
2. [ ] Compute correlation decay: $C_f(t) = \int f(S_t x) f(x)\, d\mu - (\int f\, d\mu)^2$
3. [ ] Check mixing: $C_f(t) \to 0$ as $t \to \infty$?
4. [ ] Estimate mixing time: $\tau_{\text{mix}} = \inf\{t : |C_f(t)| < \epsilon\}$. Record $\tau_{\text{mix}} = $ ____
5. [ ] If $\tau_{\text{mix}} = \infty$: System trapped in metastable states (glassy dynamics)

**Certificate:**
* [ ] $K_{\mathrm{TB}_\rho}^+ = (\mu, \tau_{\text{mix}}, \text{mixing proof})$ → **Go to Node 11**
* [ ] $K_{\mathrm{TB}_\rho}^-$ → Check BarrierMix
  * [ ] $K^{\mathrm{blk}}$: Trap escapable → **Go to Node 11**
  * [ ] Breached: Enable Surgery `SurgTD` (Mixing Enhancement)
* [ ] $K_{\mathrm{TB}_\rho}^{\mathrm{inc}}$ → **Record obligation, Go to Node 11**
  ```
  { obligation: "Establish mixing/ergodicity",
    missing: ["invariant_measure", "mixing_time"],
    failure_code: "MISSING_MIXING",
    trace: "Step 1-4 of ErgoCheck" }
  ```

---

### **Level 6: Complexity**

#### **Node 11: ComplexCheck ($\mathrm{Rep}_K$)**

**Question:** Does the system admit a finite description (bounded Kolmogorov complexity)?

**Step-by-step execution:**
1. [ ] Choose description language: $\mathcal{L} = $ ____ (e.g., finite elements, spectral coefficients, neural net weights)
2. [ ] Define dictionary map: $D: \mathcal{X} \to \mathcal{L}$
3. [ ] Compute complexity: $K(x) = |D(x)|$ (description length)
4. [ ] Check finiteness: $K(x) < \infty$?
5. [ ] If infinite: Singularity contains unbounded information (semantic horizon)

**Certificate:**
* [ ] $K_{\mathrm{Rep}_K}^+ = (\mathcal{L}, D, K(x) < C)$ → **Go to Node 12**
* [ ] $K_{\mathrm{Rep}_K}^-$ → Check BarrierEpi
  * [ ] $K^{\mathrm{blk}}$: Description within holographic bound → **Go to Node 12**
  * [ ] Breached: Enable Surgery `SurgDC` (Viscosity Solution)
* [ ] $K_{\mathrm{Rep}_K}^{\mathrm{inc}}$ → **Record obligation, Go to Node 12**
  ```
  { obligation: "Establish finite description complexity",
    missing: ["description_language", "complexity_bound"],
    failure_code: "MISSING_FINITE_DESCRIPTION",
    trace: "Step 2-4 of ComplexCheck" }
  ```

---

#### **Node 12: OscillateCheck ($\mathrm{GC}_\nabla$)**

**Question:** Does the flow oscillate (NOT a gradient flow)?

*Note: This is a dichotomy classifier. NO (gradient flow) is a benign outcome, not a failure.*

**Step-by-step execution:**
1. [ ] Check gradient structure: Does $\dot{x} = -\nabla_g \Phi(x)$ for some metric $g$?
2. [ ] Test monotonicity: Is $\frac{d}{dt}\Phi(S_t x) \leq 0$ for all $t$?
3. [ ] If YES (monotonic): Flow is gradient-like → **No oscillation**
4. [ ] If NO (non-monotonic): Identify oscillation mechanism
   - Periodic orbits: $S_T x = x$ for some $T > 0$?
   - Quasi-periodic: Torus dynamics?
   - Chaotic: Positive Lyapunov exponent?

**Certificate:**
* [ ] $K_{\mathrm{GC}_\nabla}^-$ (NO oscillation, gradient flow) → **Go to Node 13 (BoundaryCheck)**
* [ ] $K_{\mathrm{GC}_\nabla}^+$ (YES oscillation) → Check BarrierFreq
  * [ ] $K^{\mathrm{blk}}$: Oscillation integral finite → **Go to Node 13**
  * [ ] Breached: Enable Surgery `SurgDE` (De Giorgi-Nash-Moser)
* [ ] $K_{\mathrm{GC}_\nabla}^{\mathrm{inc}}$ → **Record obligation, Go to Node 13**
  ```
  { obligation: "Determine if flow is gradient-like",
    missing: ["metric_tensor", "monotonicity_proof"],
    failure_code: "MISSING_GRADIENT_STRUCTURE",
    trace: "Step 1-2 of OscillateCheck" }
  ```

---

### **Level 7: Boundary (Open Systems)**
*These nodes apply only if the system has external inputs/outputs. Skip to Node 17 if system is closed.*

#### **Node 13: BoundaryCheck ($\mathrm{Bound}_\partial$)**

**Question:** Is the system open (has boundary interactions)?

**Step-by-step execution:**
1. [ ] Identify domain boundary: $\partial\Omega = $ ____
2. [ ] Check for inputs: Does external signal $u(t)$ enter the system?
3. [ ] Check for outputs: Is observable $y(t)$ extracted from the system?
4. [ ] If open system: Define input/output maps
   - Input: $\iota: \mathcal{U} \to \mathcal{X}$
   - Output: $\pi: \mathcal{X} \to \mathcal{Y}$

**Certificate:**
* [ ] $K_{\mathrm{Bound}_\partial}^+$ (System is OPEN) → **Go to Node 14 (OverloadCheck)**
* [ ] $K_{\mathrm{Bound}_\partial}^-$ (System is CLOSED: $\partial\Omega = \emptyset$) → **Go to Node 17 (Lock)**

*Note: This is a dichotomy classifier. Both outcomes are valid; they determine which path to follow.*

---

#### **Node 14: OverloadCheck ($\mathrm{Bound}_B$)**

**Question:** Is the input bounded (no injection overload)?

**Step-by-step execution:**
1. [ ] Identify input bound: $M = \sup_t \|u(t)\|$
2. [ ] Check boundedness: $\|\iota(u)\|_{\mathcal{X}} \leq M < \infty$?
3. [ ] If bounded: System cannot be overloaded by external injection
4. [ ] If unbounded: Input can drive system to infinity (saturation needed)

**Certificate:**
* [ ] $K_{\mathrm{Bound}_B}^+ = (M, \text{input bound proof})$ → **Go to Node 15**
* [ ] $K_{\mathrm{Bound}_B}^-$ → Check BarrierBode
  * [ ] $K^{\mathrm{blk}}$: Sensitivity bounded (waterbed constraint satisfied) → **Go to Node 15**
  * [ ] Breached: Enable Surgery `SurgBE` (Saturation)
* [ ] $K_{\mathrm{Bound}_B}^{\mathrm{inc}}$ → **Record obligation, Go to Node 15**
  ```
  { obligation: "Establish input boundedness",
    missing: ["input_bound", "sensitivity_estimate"],
    failure_code: "MISSING_INPUT_BOUND",
    trace: "Step 1-2 of OverloadCheck" }
  ```

---

#### **Node 15: StarveCheck ($\mathrm{Bound}_{\Sigma}$)**

**Question:** Is the input sufficient (no resource starvation)?

**Step-by-step execution:**
1. [ ] Identify minimum required input: $r_{\min} = $ ____
2. [ ] Compute cumulative supply: $\int_0^T r(t)\, dt$
3. [ ] Check sufficiency: $\int_0^T r(t)\, dt \geq r_{\min}$?
4. [ ] If sufficient: System has adequate resources
5. [ ] If insufficient: Resource depletion will cause failure

**Certificate:**
* [ ] $K_{\mathrm{Bound}_{\Sigma}}^+ = (r_{\min}, \int r\, dt, \text{sufficiency proof})$ → **Go to Node 16**
* [ ] $K_{\mathrm{Bound}_{\Sigma}}^-$ → Check BarrierInput
  * [ ] $K^{\mathrm{blk}}$: Reserve sufficient (buffer exists) → **Go to Node 16**
  * [ ] Breached: Enable Surgery `SurgBD` (Reservoir)
* [ ] $K_{\mathrm{Bound}_{\Sigma}}^{\mathrm{inc}}$ → **Record obligation, Go to Node 16**
  ```
  { obligation: "Establish input sufficiency",
    missing: ["minimum_requirement", "supply_estimate"],
    failure_code: "MISSING_INPUT_SUFFICIENCY",
    trace: "Step 1-3 of StarveCheck" }
  ```

---

#### **Node 16: AlignCheck ($\mathrm{GC}_T$)**

**Question:** Is control matched to disturbance (requisite variety)?

**Step-by-step execution:**
1. [ ] Identify control signal: $u \in \mathcal{U}$
2. [ ] Identify disturbance: $d \in \mathcal{D}$
3. [ ] Check variety: $|\mathcal{U}| \geq |\mathcal{D}|$ (Ashby's Law)?
4. [ ] Check alignment: $\langle \iota(u), \nabla\Phi \rangle \leq 0$ (control assists descent)?
5. [ ] If aligned: Control counteracts disturbances effectively
6. [ ] If misaligned: Control fights against natural dynamics

**Certificate:**
* [ ] $K_{\mathrm{GC}_T}^+ = (u, d, \text{alignment proof})$ → **Go to Node 17 (Lock)**
* [ ] $K_{\mathrm{GC}_T}^-$ → Check BarrierVariety
  * [ ] $K^{\mathrm{blk}}$: Variety sufficient → **Go to Node 17**
  * [ ] Breached: Enable Surgery `SurgBC` (Adjoint)
* [ ] $K_{\mathrm{GC}_T}^{\mathrm{inc}}$ → **Record obligation, Go to Node 17**
  ```
  { obligation: "Establish control-disturbance alignment",
    missing: ["control_variety", "alignment_proof"],
    failure_code: "MISSING_ALIGNMENT",
    trace: "Step 3-4 of AlignCheck" }
  ```

---

### **Level 8: The Lock**

#### **Node 17: BarrierExclusion ($\mathrm{Cat}_{\mathrm{Hom}}$)**

**Question:** Is $\text{Hom}(\mathcal{H}_{\text{bad}}, \mathcal{H}) = \emptyset$?

**Step-by-step execution:**
1. [ ] Construct Universal Bad Pattern $\mathcal{H}_{\text{bad}}$ (worst-case singularity for type $T$)
2. [ ] Try each exclusion tactic E1-E10 until one succeeds:

**Tactic Checklist:**
* [ ] **E1 (Dimension):** $\dim(\mathcal{H}_{\text{bad}}) \neq \dim(\mathcal{H})$?
* [ ] **E2 (Invariant):** $I(\mathcal{H}_{\text{bad}}) \neq I(\mathcal{H})$ for some conserved quantity $I$?
* [ ] **E3 (Positivity):** Does $\mathcal{H}_{\text{bad}}$ violate a cone condition?
* [ ] **E4 (Integrality):** Arithmetic obstruction (e.g., index mismatch)?
* [ ] **E5 (Functional):** Required equations unsolvable?
* [ ] **E6 (Causal):** Well-foundedness violated?
* [ ] **E7 (Thermodynamic):** Second law / entropy production violated?
* [ ] **E8 (Holographic):** Bekenstein bound / capacity mismatch?
* [ ] **E9 (Ergodic):** Mixing rates incompatible?
* [ ] **E10 (Definability):** O-minimal tameness violated?

**Lock Verdict:**
* [ ] **BLOCKED** ($K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$) via Tactic E__ → **GLOBAL REGULARITY ESTABLISHED**
* [ ] **MORPHISM EXISTS** ($K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{morph}}$) → **SINGULARITY CONFIRMED**
* [ ] **NO-INCONCLUSIVE** ($K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{br\text{-}inc}}$) → **Record obligation**
  ```
  { obligation: "Resolve Lock verdict",
    missing: ["successful_tactic", "morphism_construction"],
    failure_code: "LOCK_UNDECIDED",
    trace: "All tactics E1-E10 exhausted without resolution" }
  ```
  *Note: An inc Lock means the proof object is conditional. Document exactly which tactics were attempted and what additional structure might resolve the verdict.*

---

## **Part II-B: Upgrade Pass**
*After completing the sieve, apply upgrade rules to discharge inc certificates.*

### **Upgrade Pass Protocol**

**Step 1: Collect all inc certificates**
List all $K_X^{\mathrm{inc}}$ in Γ:
| ID | Node | Obligation | Missing |
|----|------|------------|---------|
| | | | |

**Step 2: For each inc certificate, check upgrade applicability**

For each $K_X^{\mathrm{inc}}$:
1. [ ] List certificates in `missing`
2. [ ] Check if all missing certificates are now in Γ
3. [ ] Verify non-circularity: target $K_X^+$ not used to derive premises
4. [ ] If applicable: Apply upgrade, emit $K_X^+$, discharge obligation

**Step 3: Iterate until no more upgrades apply**

The upgrade pass terminates when no $K^{\mathrm{inc}}$ can be upgraded.

---

## **Part II-C: Breach/Surgery/Re-entry Protocol**
*If any barrier was breached, execute this protocol.*

### **Breach Detection**

Collect all $K_X^{\mathrm{br}}$ certificates:
| Barrier | Reason | Obligations |
|---------|--------|-------------|
| | | |

### **Surgery Selection**

For each breach, select an appropriate surgery:

| Breach Type | Recommended Surgery | Map ID |
|-------------|---------------------|--------|
| Energy/Dissipation | Change functional | SurgCE |
| Zeno/Recovery | Regularization | SurgCC |
| Compactness | Profile decomposition | SurgCD_Alt |
| Scaling | Renormalization | SurgSE |
| Capacity | Removable singularity | SurgCD |
| Stiffness | Symmetry breaking | SSB |
| Topology | Tunnel/Sector | SurgTE |
| Tameness | O-minimal extension | SurgTC |
| Mixing | Ergodic enhancement | SurgTD |
| Complexity | Viscosity | SurgDC |
| Oscillation | Regularization | SurgDE |
| Boundary | Control redesign | SurgB* |

### **Surgery Execution**

For each surgery:
1. [ ] Emit $K_{\mathrm{Surg}}^+(\text{map\_id})$
2. [ ] Document semantics preservation
3. [ ] Execute post-surgery nodes in new representation
4. [ ] Collect new certificates

### **Re-entry Protocol**

For each breach obligation:
1. [ ] Identify which new certificates address the obligation
2. [ ] Emit $K_{\mathrm{re}}^+(\text{item})$
3. [ ] Apply a-posteriori upgrades to earlier inc certificates
4. [ ] Update obligation ledger

---

## **Part III-A: Lyapunov Reconstruction**
*If Nodes 1, 3, 7 passed ($K_{D_E}^+$, $K_{C_\mu}^+$, $K_{\mathrm{LS}_\sigma}^+$), construct the canonical Lyapunov functional.*

### **Lyapunov Existence Check**

**Precondition:** All three certificates present?
* [ ] $K_{D_E}^+$ (dissipation with $C=0$)
* [ ] $K_{C_\mu}^+$ (compactness)
* [ ] $K_{\mathrm{LS}_\sigma}^+$ (stiffness)

If YES → Proceed with construction. If NO → Lyapunov may not exist canonically.

---

### **Step 1: Value Function Construction (KRNL-Lyapunov)**

**Compute:**
$$\mathcal{L}(x) := \inf\left\{\Phi(y) + \mathcal{C}(x \to y) : y \in M\right\}$$

where the cost-to-go is:
$$\mathcal{C}(x \to y) := \inf\left\{\int_0^T \mathfrak{D}(S_s x)\, ds : S_T x = y, T < \infty\right\}$$

**Fill in:**
* Safe manifold $M = $ [Insert: $\{x : \nabla\Phi(x) = 0\}$]
* Minimum energy $\Phi_{\min} = $ [Insert value]
* Cost functional $\mathcal{C}(x \to M) = $ [Insert formula or bound]
* **Lyapunov functional:** $\mathcal{L}(x) = $ [Insert explicit formula]

**Certificate:** $K_{\mathcal{L}}^+ = (\mathcal{L}, M, \Phi_{\min}, \mathcal{C})$

---

### **Step 2: Jacobi Metric Reconstruction (KRNL-Jacobi)**

**Additional requirement:** $K_{\mathrm{GC}_\nabla}^+$ (gradient consistency)

If flow is gradient ($\dot{u} = -\nabla_g \Phi$), the Lyapunov equals geodesic distance in Jacobi metric:

**Compute:**
1. Base metric: $g = $ [Insert metric on $\mathcal{X}$]
2. Jacobi metric: $g_{\mathfrak{D}} := \mathfrak{D} \cdot g$
3. **Explicit Lyapunov:**
$$\mathcal{L}(x) = \Phi_{\min} + \mathrm{dist}_{g_{\mathfrak{D}}}(x, M)$$

**Alternative integral form:**
$$\mathcal{L}(x) = \Phi_{\min} + \inf_{\gamma: x \to M} \int_0^1 \sqrt{\mathfrak{D}(\gamma(s))} \cdot \|\dot{\gamma}(s)\|_g \, ds$$

**Certificate:** $K_{\text{Jacobi}}^+ = (g_{\mathfrak{D}}, \mathrm{dist}_{g_{\mathfrak{D}}}, M)$

---

### **Step 3: Hamilton-Jacobi PDE (KRNL-HamiltonJacobi)**

**The Lyapunov satisfies the static Hamilton-Jacobi equation:**

$$\|\nabla_g \mathcal{L}(x)\|_g^2 = \mathfrak{D}(x)$$

with boundary condition: $\mathcal{L}|_M = \Phi_{\min}$

**To solve:**
1. [ ] Set up PDE: $|\nabla \mathcal{L}|^2 = \mathfrak{D}$ on $\mathcal{X} \setminus M$
2. [ ] Boundary: $\mathcal{L} = \Phi_{\min}$ on $M$
3. [ ] Find viscosity solution (e.g., via characteristics or numerics)

**Certificate:** $K_{\text{HJ}}^+ = (\mathcal{L}, \nabla_g \mathcal{L}, \mathfrak{D})$

---

### **Step 4: Verify Lyapunov Properties**

Check the reconstructed $\mathcal{L}$ satisfies:

* [ ] **Monotonicity:** $\frac{d}{dt}\mathcal{L}(S_t x) = -\mathfrak{D}(S_t x) \leq 0$
* [ ] **Strict decay:** $\frac{d}{dt}\mathcal{L}(S_t x) < 0$ when $x \notin M$
* [ ] **Minimum on $M$:** $\mathcal{L}(x) = \Phi_{\min}$ iff $x \in M$
* [ ] **Coercivity:** $\mathcal{L}(x) \to \infty$ as $\|x\| \to \infty$ (or appropriate boundary)

**Final Lyapunov Certificate:** $K_{\mathcal{L}}^{\text{verified}}$

---

## **Part III-B: Result Extraction (Mining the Run)**
*Use the Extraction Metatheorems to pull rigorous math objects from the certificates.*

### **3.1 Global Theorems**
* [ ] **Global Regularity Theorem:** (From Node 17 Blocked + KRNL-Consistency).
    * *Statement:* "The system defined by $(\mathcal{X}, \Phi, \mathfrak{D})$ admits global regular solutions."
* [ ] **Singularity Classification:** (From Node 3 + RESOLVE-AutoProfile).
    * *Statement:* "All singularities are isomorphic to the set: [List Profiles]."

### **3.2 Quantitative Bounds**
* [ ] **Energy/Density Bound:** (From Node 1 / BarrierSat + UP-Saturation).
    * *Formula:* $\Phi(t) \le$ [Insert Bound]
* [ ] **Dimension Bound:** (From Node 6 / BarrierCap + UP-Capacity).
    * *Formula:* $\text{dim}_{\mathcal{H}}(\Sigma) \le$ [Insert Value]
* [ ] **Convergence Rate:** (From Node 7 / Stiffness + KRNL-Lyapunov).
    * *Formula:* Rate $\sim$ [Exp/Poly] (based on $\theta$).

### **3.3 Functional Objects**
* [ ] **Strict Lyapunov Function ($\mathcal{L}$):** (From Part III-A / KRNL-Lyapunov through KRNL-MetricAction).
    * *Definition:* $\mathcal{L}(x) = \Phi_{\min} + \mathrm{dist}_{g_{\mathfrak{D}}}(x, M)$ (Jacobi metric form, if gradient flow).
    * *Alternative:* $\mathcal{L}(x) = \inf\{\Phi(y) + \mathcal{C}(x \to y) : y \in M\}$ (Value function form, general case).
    * *Value:* Proves strict monotonicity, convergence to equilibrium, and defines stability basins.
* [ ] **Surgery Operator ($\mathcal{O}_S$):** (From RESOLVE-AutoSurgery).
    * *Definition:* Canonical pushout for [Profile Name].
    * *Value:* Defines the rigorous mechanism for extending flow past singularities.
* [ ] **Spectral Constraint ($H$):** (From Node 17 / Tactic E4 + LOCK-SpectralGen).
    * *Definition:* Operator satisfying Lock constraints.
    * *Value:* Defines the "Missing Physics" (e.g., Berry-Keating Hamiltonian).

### **3.4 Retroactive Upgrades**
*Check if late-stage proofs upgrade earlier weak permits.*
* [ ] **Lock-Back (UP-LockBack):** Did Node 17 pass? $\implies$ All Barrier Blocks are **Regular**.
* [ ] **Symmetry-Gap (UP-SymmetryBridge):** Did SymCheck pass? $\implies$ Stiffness Stagnation is **Mass Gap**.
* [ ] **Tame-Topology (UP-TameSmoothing):** Did TameCheck pass? $\implies$ Zero Capacity sets are **Removable**.

### **3.5 ZFC Proof Export (Chapter 56 Bridge)**
*Use the ZFC Translation Layer (Chapter 56 of `hypopermits_jb.md`) to compile the categorical run into a classical, set-theoretic audit trail.*

**Precondition:** You have a Lock verdict (typically $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$) and a complete certificate chain $\mathbf{K} = (K_1,\ldots,K_{17})$.

**ZFC Bridge Checklist**
* [ ] Fix a Grothendieck universe $\mathcal{U}$ (Chapter 56.1).
* [ ] Record `axioms_used` via the Sieve-to-ZFC table (Definition {prf:ref}`def-sieve-zfc-correspondence`).
* [ ] Record `AC_status` via the Choice audit (Definition {prf:ref}`def-ac-dependency` and {prf:ref}`def-choice-sensitive-stratum`).
* [ ] Record `translation_trace := (\tau_0(K_1),\ldots,\tau_0(K_{17}))` (Definition {prf:ref}`def-truncation-functor-tau0`).

**Step 1: Choose the set-theoretic target formula $\varphi$**
Fill in a first-order statement $\varphi \in \mathcal{L}_{\mathrm{ZFC}}$ (or an equivalent encoding) such that:
$$V_\mathcal{U} \vDash \varphi \quad\text{and}\quad (\varphi \Rightarrow \mathrm{Reg}(Z)).$$

Canonical options from Chapter 56:
- Hom-emptiness form (Metatheorem {prf:ref}`mt-krnl-zfc-bridge`): $\varphi := ``\tau_0(\mathrm{Hom}(\mathbb{H}_{\mathrm{bad}},\mathbb{H}(Z))) = \emptyset''$.
- Orbit exclusion form (Theorem {prf:ref}`thm-bridge-zfc-fundamental`): $\varphi := \forall u \in \tau_0(\mathcal{X}),\,\Psi(u)$.

**Step 2: Emit the Bridge Certificate payload**
$$\mathcal{B}_{\text{ZFC}} := (\mathcal{U}, \varphi, \text{axioms\_used}, \text{AC\_status}, \text{translation\_trace}).$$

**Step 3: Write the ZFC-audit proof sketch**
* [ ] Use Theorem {prf:ref}`thm-zfc-grounding` to justify that each $\tau_0(K_i)$ is set-level data in $V_\mathcal{U}$.
* [ ] For each node, cite the corresponding axiom(s) from Definition {prf:ref}`def-sieve-zfc-correspondence` and state the translated lemma $\psi_i$.
* [ ] Conclude $V_\mathcal{U} \vDash \varphi$ (and hence the classical regularity claim) via Metatheorem {prf:ref}`mt-krnl-zfc-bridge`.

---

## **Part III-C: Obligation Ledger**
*Track all obligations introduced during the run and their discharge status.*

### **Introduced Obligations**

| ID | Node | Certificate | Obligation | Missing | Status |
|----|------|-------------|------------|---------|--------|
| O1 | [#] | $K_X^{\mathrm{inc}}$ | [Description] | [List] | [ ] Pending / [ ] Discharged |
| O2 | [#] | $K_Y^{\mathrm{br}}$ | [Description] | [List] | [ ] Pending / [ ] Discharged |
| ... | | | | | |

### **Discharge Events**

| Obligation ID | Discharged At | Mechanism | Using Certificates |
|---------------|---------------|-----------|-------------------|
| O1 | Node [#] / Upgrade Pass | Upgrade $U_{X\to+}$ | $K_A^+, K_B^+$ |
| O2 | Re-entry | $K_{\mathrm{re}}^+$ | $K_C^+, K_D^+$ |
| ... | | | |

### **Remaining Obligations**

**Count:** ___

| ID | Obligation | Why Unresolved |
|----|------------|----------------|
| | | |

### **Ledger Validation**

* [ ] **All inc certificates either upgraded or documented as conditional**
* [ ] **All breach obligations either discharged or documented**
* [ ] **Remaining obligations count = 0** (for unconditional proof)

**Ledger Status:** [ ] EMPTY (valid unconditional proof) / [ ] NON-EMPTY (conditional proof)

---

## **Part IV: Final Certificate Chain**

### **4.1 Validity Checklist**

Before declaring the proof object complete, verify:

- [ ] **All 12 core nodes executed** (Nodes 1-12)
- [ ] **Boundary nodes executed** (Nodes 13-16, if system is open)
- [ ] **Lock executed** (Node 17)
- [ ] **Lock verdict obtained:** $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$ or $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$ or $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{morph}}$
- [ ] **Upgrade pass completed** (Part II-B)
- [ ] **Surgery/Re-entry completed** (Part II-C, if any breaches)
- [ ] **Obligation ledger is EMPTY** (Part III-C)
- [ ] **No unresolved $K^{\mathrm{inc}}$** in final Γ

**Validity Status:** [ ] UNCONDITIONAL PROOF / [ ] CONDITIONAL PROOF

### **4.2 Certificate Accumulation Trace**

```
Node 1:  K_{D_E}^? (energy-dissipation)
Node 2:  K_{Rec_N}^? (recovery/surgeries)
Node 3:  K_{C_μ}^? (compactness)
Node 4:  K_{SC_λ}^? (scaling)
Node 5:  K_{SC_∂c}^? (parameters)
Node 6:  K_{Cap_H}^? (capacity/geometry)
Node 7:  K_{LS_σ}^? (stiffness)
Node 8:  K_{TB_π}^? (topology)
Node 9:  K_{TB_O}^? (tameness)
Node 10: K_{TB_ρ}^? (mixing)
Node 11: K_{Rep_K}^? (complexity)
Node 12: K_{GC_∇}^? (gradient)
Node 13: K_{Bound_∂}^? (boundary)
---
[Surgery: K_{Surg}^? if applicable]
[Re-Entry: K^{re}_{...} if applicable]
---
Node 17: K_{Cat_Hom}^? (Lock)
```

*Replace ? with actual certificate type (+, -, blk, inc, br, etc.)*

### **4.3 Final Certificate Set**

$$\Gamma_{\mathrm{final}} = \{K_{D_E}^{?}, K_{\mathrm{Rec}_N}^{?}, K_{C_\mu}^{?}, K_{\mathrm{SC}_\lambda}^{?}, K_{\mathrm{SC}_{\partial c}}^{?}, K_{\mathrm{Cap}_H}^{?}, K_{\mathrm{LS}_\sigma}^{?}, K_{\mathrm{TB}_\pi}^{?}, K_{\mathrm{TB}_O}^{?}, K_{\mathrm{TB}_\rho}^{?}, K_{\mathrm{Rep}_K}^{?}, K_{\mathrm{GC}_\nabla}^{?}, K_{\mathrm{Bound}_\partial}^{?}, K_{\mathrm{Cat}_{\mathrm{Hom}}}^{?}\}$$

### **4.4 Conclusion**

**Conclusion:** The Conjecture is [TRUE / FALSE / UNDECIDABLE / CONDITIONAL].

**Proof Summary ($\Gamma$):**
"The system is [Regular/Singular] because:
1.  **Conservation:** Established by [Certificate ID] (e.g., $K_{D_E}^+$, $K_{\text{sat}}^{\mathrm{blk}}$).
2.  **Structure:** Established by [Certificate ID] (e.g., $K_{C_\mu}^+$, $K_{\mathrm{Cap}_H}^+$).
3.  **Stiffness:** Established by [Certificate ID] (e.g., $K_{\mathrm{LS}_\sigma}^+$, $K_{\text{gap}}^{\mathrm{blk}}$).
4.  **Lyapunov:** Constructed via Part III-A (e.g., $K_{\mathcal{L}}^{\text{verified}}$, $K_{\text{Jacobi}}^+$, $K_{\text{HJ}}^+$).
5.  **Exclusion:** Established by [Certificate ID] (e.g., $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$ via Tactic E__)."

**Full Certificate Chain:**
$$\Gamma = \{K_{D_E}^+, K_{\mathrm{Rec}_N}^+, K_{C_\mu}^+, K_{\mathrm{SC}_\lambda}^+, K_{\mathrm{SC}_{\partial c}}^+, K_{\mathrm{Cap}_H}^+, K_{\mathrm{LS}_\sigma}^+, K_{\mathcal{L}}^{\text{verified}}, K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}\}$$

---

## **Formal Proof**

::::{prf:proof} Proof of Theorem {prf:ref}`thm-problem-slug-main`

The proof proceeds by structural sieve analysis in seven phases:

**Phase 1 (Instantiation):** We defined the hypostructure $({\mathcal{X}, \Phi, \mathfrak{D}, G})$ in Part I, implementing the required interface permits.

**Phase 2 (Conservation):** Nodes 1-3 established energy control ($K_{D_E}$), finite bad events ($K_{\mathrm{Rec}_N}$), and compactness modulo symmetry ($K_{C_\mu}$).

**Phase 3 (Scaling):** Nodes 4-5 verified subcriticality ($K_{\mathrm{SC}_\lambda}$) and parameter stability ($K_{\mathrm{SC}_{\partial c}}$).

**Phase 4 (Geometry):** Nodes 6-7 established small singular set ($K_{\mathrm{Cap}_H}$) and stiffness ($K_{\mathrm{LS}_\sigma}$).

**Phase 5 (Topology):** Nodes 8-12 verified sector preservation ($K_{\mathrm{TB}_\pi}$), tameness ($K_{\mathrm{TB}_O}$), mixing ($K_{\mathrm{TB}_\rho}$), finite complexity ($K_{\mathrm{Rep}_K}$), and gradient structure ($K_{\mathrm{GC}_\nabla}$).

**Phase 6 (Boundary):** [If applicable] Nodes 13-16 verified boundary conditions.

**Phase 7 (Lock):** Node 17 blocked the universal bad pattern $\mathcal{H}_{\text{bad}}$ via Tactic E[X], establishing $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$.

**Conclusion:** By the Lock Metatheorem (KRNL-Consistency), the blocked Lock certificate implies the target claim.

$$\therefore \text{[CLAIM]} \quad \square$$

::::

---

## **Verification Summary**

| Component | Status | Certificate |
|-----------|--------|-------------|
| Nodes 1-12 (Core) | [PASS/FAIL/INC] | [List] |
| Nodes 13-16 (Boundary) | [N/A/PASS/FAIL] | [List] |
| Node 17 (Lock) | [BLOCKED/MORPHISM/INC] | $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{?}$ |
| Obligation Ledger | [EMPTY/NON-EMPTY] | — |
| Upgrade Pass | [COMPLETE] | [List upgrades] |

**Final Verdict:** [ ] UNCONDITIONAL PROOF / [ ] CONDITIONAL PROOF / [ ] SINGULARITY CONFIRMED

---

## **References**

1. Hypostructure Framework v1.0 (`hypopermits_jb.md`)
2. [Add relevant mathematical references]
3. [Add relevant prior work]

---

## Appendix: Replay Bundle (Machine-Checkability)

This proof object is replayed by providing:
1. `trace.json`: ordered node outcomes
2. `certs/`: serialized certificates with payload hashes
3. `inputs.json`: thin objects and initial-state hash
4. `closure.cfg`: promotion/closure settings

**Replay acceptance criterion:** The checker recomputes the same $\Gamma_{\mathrm{final}}$ and emits `FINAL`.

---

## Executive Summary: The Proof Dashboard
*Fill this section after completing the sieve run (Phase 0: Dashboard Generation).*

### 1. System Instantiation (The Physics)
*Mapping the physical problem to the Hypostructure categories.*

| Object | Definition | Role |
| :--- | :--- | :--- |
| **Arena ($\mathcal{X}$)** | [e.g., $H^1(\mathbb{R}^3)$] | State Space |
| **Potential ($\Phi$)** | [e.g., Energy $E(u)$] | Lyapunov Functional |
| **Cost ($\mathfrak{D}$)** | [e.g., Enstrophy $\|\nabla u\|^2$] | Dissipation |
| **Invariance ($G$)** | [e.g., $SO(3) \times \mathbb{R}^3$] | Symmetry Group |

### 2. Execution Trace (The Logic)
*The chronological flow of the Sieve. If Surgery occurs, list the re-entry steps.*

| Node | Check | Outcome | Certificate Payload | Ledger State |
| :--- | :--- | :---: | :--- | :--- |
| **1** | Energy Bound | [YES/NO/INC] | [e.g., $E(t) \le E_0$] | `[]` |
| **2** | Zeno Check | [YES/NO/INC] | [Payload] | `[]` |
| **3** | Compact Check | [YES/NO/INC] | [Payload] | `[]` |
| **4** | Scale Check | [YES/NO/INC] | [Payload] | `[]` |
| **5** | Param Check | [YES/NO/INC] | [Payload] | `[]` |
| **6** | Geom Check | [YES/NO/INC] | [Payload] | `[]` |
| **7** | Stiffness Check | [YES/NO/INC] | [Payload] | `[]` |
| **8** | Topo Check | [YES/NO/INC] | [Payload] | `[]` |
| **9** | Tame Check | [YES/NO/INC] | [Payload] | `[]` |
| **10** | Ergo Check | [YES/NO/INC] | [Payload] | `[]` |
| **11** | Complex Check | [YES/NO/INC] | [Payload] | `[]` |
| **12** | Oscillate Check | [YES/NO/INC] | [Payload] | `[]` |
| **13** | Boundary Check | [OPEN/CLOSED] | [Payload] | `[]` |
| **14** | Overload Check | [YES/NO/INC/N/A] | [Payload] | `[]` |
| **15** | Starve Check | [YES/NO/INC/N/A] | [Payload] | `[]` |
| **16** | Align Check | [YES/NO/INC/N/A] | [Payload] | `[]` |
| **--** | **SURGERY** | **[EXEC/N/A]** | [Surgery Name] | `[OBL-?]` |
| **--** | **RE-ENTRY** | **[OK/N/A]** | [Recovery proof] | `[]` |
| **17** | **LOCK** | **[BLOCK/MORPH/INC]** | [Tactic ID] | `[]` |

### 3. Lock Mechanism (The Exclusion)
*How the singularity is structurally forbidden at Node 17.*

| Tactic | Description | Status | Reason / Mechanism |
| :--- | :--- | :---: | :--- |
| **E1** | Dimension | [PASS/FAIL/N/A] | [e.g., $d < d_c$] |
| **E2** | Invariant | [PASS/FAIL/N/A] | [e.g., Conservation vs Blowup] |
| **E3** | Positivity | [PASS/FAIL/N/A] | [e.g., Cone condition] |
| **E4** | Integrality | [PASS/FAIL/N/A] | [e.g., Index mismatch] |
| **E5** | Functional | [PASS/FAIL/N/A] | [e.g., Unsolvable PDE] |
| **E6** | Causal | [PASS/FAIL/N/A] | [e.g., Well-foundedness] |
| **E7** | Thermodynamic | [PASS/FAIL/N/A] | [e.g., Entropy production] |
| **E8** | Holographic | [PASS/FAIL/N/A] | [e.g., Bekenstein bound] |
| **E9** | Ergodic | [PASS/FAIL/N/A] | [e.g., Mixing rates] |
| **E10** | Definability | [PASS/FAIL/N/A] | [e.g., O-minimal structure] |

**Auxiliary Witnesses (if analytic bridge admissibility is invoked):**

| Witness Certificate | Status | Note |
| :--- | :---: | :--- |
| $K_{D_{\max}}^+$ | [PRESENT/ABSENT] | [e.g., bounded diameter used for Gevrey admissibility] |
| $K_{\rho_{\max}}^+$ | [PRESENT/ABSENT] | [e.g., density bound used for Gevrey admissibility] |

### 4. Final Verdict

* **Status:** [UNCONDITIONAL / CONDITIONAL]
* **Obligation Ledger:** [EMPTY / NON-EMPTY (list remaining)]
* **Singularity Set:** $\Sigma = \emptyset$ (or description of allowable set)
* **Primary Blocking Tactic:** [E? - Description]

---

## Document Information

| Field | Value |
|-------|-------|
| **Document Type** | Proof Object |
| **Framework** | Hypostructure v1.0 |
| **Problem Class** | [e.g., Classical PDE, Open Problem] |
| **System Type** | $T_{\text{[type]}}$ |
| **Verification Level** | Machine-checkable |
| **Inc Certificates** | [N] introduced, [M] discharged |
| **Final Status** | [ ] Draft / [ ] Final |
| **Generated** | [YYYY-MM-DD] |

---

*This document constitutes a machine-checkable proof object under the Hypostructure framework.*
*Each certificate can be independently verified against the definitions in `hypopermits_jb.md`.*

**QED**
