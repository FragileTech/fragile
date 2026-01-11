# Part X: Equivalence and Transport

:::{prf:remark} Naming convention

This part defines **equivalence moves** ({prf:ref}`def-equiv-symmetry`--{prf:ref}`def-equiv-bridge`) and **transport lemmas** ({prf:ref}`def-transport-t1`--{prf:ref}`def-transport-t6`). These are distinct from the **Lock tactics** ({prf:ref}`def-e1`--{prf:ref}`def-e10`) defined in {ref}`the Lock Exclusion Tactics section <sec-lock-exclusion-tactics>`. The `Eq` prefix distinguishes equivalence moves from Lock tactics.

:::

(sec-equivalence-library)=
## Equivalence Library

:::{prf:definition} Admissible equivalence move
:label: def-equiv-move

An **admissible equivalence move** for type $T$ is a transformation $(x, \Phi, \mathfrak{D}) \mapsto (\tilde{x}, \tilde{\Phi}, \tilde{\mathfrak{D}})$ with:
1. **Comparability bounds**: Constants $C_1, C_2 > 0$ with
   $$\begin{aligned}
   C_1 \Phi(x) &\leq \tilde{\Phi}(\tilde{x}) \leq C_2 \Phi(x) \\
   C_1 \mathfrak{D}(x) &\leq \tilde{\mathfrak{D}}(\tilde{x}) \leq C_2 \mathfrak{D}(x)
   \end{aligned}$$
2. **Structural preservation**: Interface permits preserved
3. **Certificate production**: Equivalence certificate $K_{\text{equiv}}$

:::

---

### Standard Equivalence Moves

:::{prf:definition} Eq1: Symmetry quotient
:label: def-equiv-symmetry

For symmetry group $G$ acting on $X$:
$$\tilde{x} = [x]_G \in X/G$$
Comparability: $\Phi([x]_G) = \inf_{g \in G} \Phi(g \cdot x)$ (coercivity modulo $G$)

:::

:::{prf:definition} Eq2: Metric deformation (Hypocoercivity)
:label: def-equiv-metric

Replace metric $d$ with equivalent metric $\tilde{d}$:
$$C_1 d(x, y) \leq \tilde{d}(x, y) \leq C_2 d(x, y)$$
Used when direct LS fails but deformed LS holds.

:::

:::{prf:definition} Eq3: Conjugacy
:label: def-equiv-conjugacy

For invertible $h: X \to Y$:
$$\tilde{S}_t = h \circ S_t \circ h^{-1}$$
Comparability: $\Phi_Y(h(x)) \sim \Phi_X(x)$

:::

:::{prf:definition} Eq4: Surgery identification
:label: def-equiv-surgery-id

Outside excision region $E$:
$$x|_{X \setminus E} = x'|_{X \setminus E}$$
Transport across surgery boundary.

:::

:::{prf:definition} Eq5: Analytic-hypostructure bridge
:label: def-equiv-bridge

Between classical solution $u$ and hypostructure state $x$:
$$x = \mathcal{H}(u), \quad u = \mathcal{A}(x)$$
with inverse bounds.

:::

---

(sec-yes-tilde-permits)=
## YES$^\sim$ Permits

:::{prf:definition} YES$^\sim$ certificate
:label: def-yes-tilde-cert

A **YES$^\sim$ certificate** for predicate $P_i$ is a triple:
$$K_i^{\sim} = (K_{\text{equiv}}, K_{\text{transport}}, K_i^+[\tilde{x}])$$
where:
- $K_{\text{equiv}}$: Certifies $x \sim_{\mathrm{Eq}} \tilde{x}$ for some equivalence move {prf:ref}`def-equiv-symmetry`--{prf:ref}`def-equiv-bridge`
- $K_{\text{transport}}$: Transport lemma certificate (from {prf:ref}`def-transport-t1`--{prf:ref}`def-transport-t6`)
- $K_i^+[\tilde{x}]$: YES certificate for $P_i$ on the equivalent object $\tilde{x}$

:::

:::{prf:definition} YES$^\sim$ acceptance
:label: def-yes-tilde-accept

A metatheorem $\mathcal{M}$ **accepts YES$^\sim$** if:
$$\mathcal{M}(K_{I_1}, \ldots, K_{I_i}^{\sim}, \ldots, K_{I_n}) = \mathcal{M}(K_{I_1}, \ldots, K_{I_i}^+, \ldots, K_{I_n})$$
That is, YES$^\sim$ certificates may substitute for YES certificates in the metatheorem's preconditions.

:::

---

(sec-transport-toolkit)=
## Transport Toolkit

:::{prf:definition} T1: Inequality transport
:label: def-transport-t1

Under comparability $C_1 \Phi \leq \tilde{\Phi} \leq C_2 \Phi$:
$$\tilde{\Phi}(\tilde{x}) \leq E \Rightarrow \Phi(x) \leq E/C_1$$

:::

:::{prf:definition} T2: Integral transport
:label: def-transport-t2

Under dissipation comparability:
$$\int \tilde{\mathfrak{D}} \leq C_2 \int \mathfrak{D}$$

:::

:::{prf:definition} T3: Quotient transport
:label: def-transport-t3

For $G$-quotient with coercivity:
$$P_i(x) \Leftarrow P_i([x]_G) \wedge \text{(orbit bound)}$$

:::

:::{prf:definition} T4: Metric equivalence transport
:label: def-transport-t4

LS inequality transports under equivalent metrics:
$$\text{LS}_{\tilde{d}}(\theta, C) \Rightarrow \text{LS}_d(\theta, C/C_2)$$

:::

:::{prf:definition} T5: Conjugacy transport
:label: def-transport-t5

Invariants transport under conjugacy:
$$\tau(x) = \tilde{\tau}(h(x))$$

:::

:::{prf:definition} T6: Surgery identification transport
:label: def-transport-t6

Outside excision, all certificates transfer:
$$K[x|_{X \setminus E}] = K[x'|_{X \setminus E}]$$

:::

---

(sec-promotion-system)=
## Promotion System

:::{prf:definition} Immediate promotion
:label: def-promotion-immediate

Rules using only past/current certificates:

**Barrier-to-YES**: If blocked certificate plus earlier certificates imply the predicate:
$$K_i^{\mathrm{blk}} \wedge \bigwedge_{j < i} K_j^+ \Rightarrow K_i^+$$

Example: $K_{\text{Cap}}^{\mathrm{blk}}$ (singular set measure zero) plus $K_{\text{SC}}^+$ (subcritical) may together imply $K_{\text{Geom}}^+$.

:::

:::{prf:definition} A-posteriori promotion
:label: def-promotion-aposteriori

Rules using later certificates:

$$K_i^{\mathrm{blk}} \wedge \bigwedge_{j > i} K_j^+ \Rightarrow K_i^+$$

Example: Full Lock passage ($K_{\mathrm{Cat}_{\mathrm{Hom}}}^+$) may retroactively promote earlier blocked certificates to full YES.

:::

:::{prf:definition} Promotion closure
:label: def-promotion-closure

The **promotion closure** $\mathrm{Cl}(\Gamma)$ is the least fixed point:
$$\Gamma_0 = \Gamma, \quad \Gamma_{n+1} = \Gamma_n \cup \{K : \text{promoted or inc-upgraded from } \Gamma_n\}$$
$$\mathrm{Cl}(\Gamma) = \bigcup_n \Gamma_n$$

This includes both blocked-certificate promotions (Definition {prf:ref}`def-promotion-permits`) and inconclusive-certificate upgrades (Definition {prf:ref}`def-inc-upgrades`).

:::

:::{prf:definition} Replay semantics
:label: def-replay

Given final context $\Gamma_{\text{final}}$, the **replay** is a re-execution of the sieve under $\mathrm{Cl}(\Gamma_{\text{final}})$, potentially yielding a different (stronger) fingerprint.

:::
