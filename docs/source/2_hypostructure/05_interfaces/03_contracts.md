# Part VIII: Barrier and Surgery Contracts

(sec-barrier-atlas)=
## Certificate-Driven Barrier Atlas

:::{prf:definition} Barrier contract format
:label: def-barrier-format

Each barrier entry in the atlas specifies:

1. **Trigger**: Which gate NO invokes this barrier
2. **Pre**: Required certificates (from $\Gamma$), subject to non-circularity
3. **Blocked certificate**: $K^{\mathrm{blk}}$ satisfying $K^{\mathrm{blk}} \Rightarrow \mathrm{Pre}(\text{next gate})$
4. **Breached certificate**: $K^{\mathrm{br}}$ satisfying:
   - $K^{\mathrm{br}} \Rightarrow \text{Mode } m \text{ active}$
   - $K^{\mathrm{br}} \Rightarrow \mathrm{SurgeryAdmissible}(m)$
5. **Scope**: Which types $T$ this barrier applies to

:::

:::{prf:theorem} Non-circularity
:label: thm-barrier-noncircular

For any barrier $B$ triggered by gate $i$ with predicate $P_i$:
$$P_i \notin \mathrm{Pre}(B)$$
A barrier invoked because $P_i$ failed cannot assume $P_i$ as a prerequisite.

**Literature:** Stratification and well-foundedness {cite}`VanGelder91`; non-circular definitions {cite}`AptBolPedreschi94`.

:::

---

(sec-surgery-contracts)=
## Surgery Contracts

:::{prf:definition} Surgery contract format
:label: def-surgery-format

Each surgery entry follows the **Surgery Specification Schema** (Definition {prf:ref}`def-surgery-schema`):

1. **Surgery ID** and **Target Mode**: Unique identifier and triggering failure mode
2. **Interface Dependencies**:
   - **Primary:** Interface providing the singular object/profile $V$ and locus $\Sigma$
   - **Secondary:** Interface providing canonical library $\mathcal{L}_T$ or capacity bounds
3. **Admissibility Signature**:
   - **Input Certificate:** $K^{\mathrm{br}}$ from triggering barrier
   - **Admissibility Predicate:** Conditions for safe surgery (Case 1 of Trichotomy)
4. **Transformation Law** ($\mathcal{O}_S$):
   - **State Space:** How $X \to X'$
   - **Height Jump:** Energy/height change guarantee
   - **Topology:** Sector changes if any
5. **Postcondition**:
   - **Re-entry Certificate:** $K^{\mathrm{re}}$ with $K^{\mathrm{re}} \Rightarrow \mathrm{Pre}(\text{target node})$
   - **Re-entry Target:** Node to resume sieve execution
   - **Progress Guarantee:** Type A (bounded count) or Type B (well-founded complexity)

See {prf:ref}`def-surgery-schema` for the complete Surgery Specification Schema.
:::

:::{prf:definition} Progress measures
:label: def-progress-measures

Valid progress measures for surgery termination:

**Type A (Bounded count)**:
$$\#\{S\text{-surgeries on } [0, T)\} \leq N(T, \Phi(x_0))$$
for explicit bound $N$ depending on time and initial energy.

**Type B (Well-founded)**:
A complexity measure $\mathcal{C}: X \to \mathbb{N}$ (or ordinal $\alpha$) with:
$$\mathcal{O}_S(x) = x' \Rightarrow \mathcal{C}(x') < \mathcal{C}(x)$$

**Discrete Progress Constraint (Required for Type A):**
When using energy $\Phi: X \to \mathbb{R}_{\geq 0}$ as progress measure, termination requires a **uniform minimum drop**:
$$\exists \epsilon_T > 0: \quad \mathcal{O}_S(x) = x' \Rightarrow \Phi(x) - \Phi(x') \geq \epsilon_T$$
This converts the continuous codomain $\mathbb{R}_{\geq 0}$ into a well-founded order by discretizing into levels $\{0, \epsilon_T, 2\epsilon_T, \ldots\}$. The surgery count is then bounded:
$$N \leq \frac{\Phi(x_0)}{\epsilon_T}$$

**Remark (Zeno Prevention):** Without the discrete progress constraint, a sequence of surgeries could have $\Delta\Phi_n \to 0$ (e.g., $\Delta\Phi_n = 2^{-n}$), summing to finite total but comprising infinitely many steps. The constraint $\Delta\Phi \geq \epsilon_T$ excludes such Zeno sequences.

:::

---
