# Part XII: Factory Metatheorems

:::{div} feynman-prose
Now we come to what I consider the most elegant part of the whole framework. You have seen all these gates, barriers, and surgery operations. But where do they actually come from? How do you know they are correct?

Here is the beautiful thing: we do not construct them by hand. We have a *factory*---a systematic machine that takes a mathematical specification and produces correct code automatically. This is not some magical oracle. It is more like a well-designed compiler: you tell it what you want (the type specification), and it generates verifiers that are *guaranteed* to satisfy their contracts.

Think of it this way. Suppose you want to check whether a solution has finite energy. The factory takes your energy functional $\Phi$ and produces a verifier that: (1) returns YES with a certificate proving the bound holds, or (2) returns NO with evidence of violation, or (3) admits it cannot decide and routes to fallback. The factory metatheorems prove this process is *sound*---if the factory says YES, then the property genuinely holds.

Why does this matter? Because it separates the *what* from the *how*. The mathematician specifies what properties matter. The factory handles how to check them. And the metatheorems guarantee the connection is tight.
:::

(sec-tm1-gate-evaluator)=
## TM-1: Gate Evaluator Factory

:::{div} feynman-prose
Let me explain what the Gate Evaluator Factory actually does. You have 17 gates in the Sieve, each checking some property of your solution. The question is: given a new type $T$ (say, a new PDE), how do you produce verifiers for all these gates?

The answer is that you do not do it manually. You supply the structural data---the energy functional $\Phi$, the dissipation functional $\mathfrak{D}$, the symmetry group $G$---and the factory produces verifiers that are correct by construction. It is like defining an interface in programming: the factory guarantees that anything it produces satisfies the interface contract.

But here is the subtlety that trips people up: this is not claiming we can decide everything. Some predicates are genuinely undecidable. What the factory guarantees is that the verifier *always terminates* and *never lies*. It may say "I cannot decide," but it will never say YES when the answer is NO.
:::

::::{prf:theorem} [FACT-Gate] Gate Evaluator Factory
:label: mt-fact-gate

**Rigor Class:** F (Framework-Original) — see {prf:ref}`def-rigor-classification`

This metatheorem establishes that the factory-generated code is **Correct-by-Construction**. The factory is a natural transformation between the "Type Specification" functor and the "Logic Evaluator" functor, ensuring that code generation preserves semantics.

For any system of type $T$ with user-defined objects $(\Phi, \mathfrak{D}, G, \mathcal{R}, \mathrm{Cap}, \tau, D)$, there exist canonical verifiers for all gate nodes:

**Input**: Type $T$ structural data + user definitions

**Output**: For each gate $i \in \{1, \ldots, 17\}$:
- Predicate instantiation $P_i^T$
- Verifier $V_i^T: X \times \Gamma \to \{`YES`, `NO`\} \times \mathcal{K}_i$

**Soundness**: $V_i^T(x, \Gamma) = (`YES`, K_i^+) \Rightarrow P_i^T(x)$

:::{prf:remark} Interface Specification, Not Oracle
:class: feynman-added

This metatheorem specifies the **interface contract** for verifiers, not an existence claim for a universal decision procedure. The framework assumes verifiers satisfying this contract are either:
1. **Provided by the user** (for domain-specific predicates), or
2. **Derived from type definitions** (via the factory composition $\mathcal{F} = \mathcal{V} \circ \mathcal{T}$)

Soundness follows from the contract; the user's responsibility is to supply correct verifiers for their specific domain. The factory metatheorem guarantees that *if* verifiers satisfy the interface, *then* the Sieve produces sound certificates. This is analogous to type class constraints in programming: we specify what operations must exist, not how to implement them for all cases.

For undecidable predicates (e.g., Gate 17), the framework uses the tactic library E1-E12 with $K^{\mathrm{inc}}$ (inconclusive) fallback—the verifier always terminates, but may return "inconclusive" rather than a definite YES/NO.
:::

:::{prf:proof}
:label: proof-mt-fact-gate

*Proof (Following Categorical Proof Template — Natural Transformation Soundness).*

*Step 0 (Ambient Setup: Functor Categories).* Define the relevant functor categories:
- **Type Specification Functor** $\mathcal{T}: \mathbf{Type} \to \mathbf{Pred}$ mapping types $T$ to their predicate systems $\{P_i^T\}_{i=1}^{17}$
- **Logic Evaluator Functor** $\mathcal{V}: \mathbf{Pred} \to \mathbf{Verifier}$ mapping predicates to certified verifiers
- The factory is the composition $\mathcal{F} = \mathcal{V} \circ \mathcal{T}: \mathbf{Type} \to \mathbf{Verifier}$

The naturality square commutes: for any type morphism $f: T \to T'$, we have $\mathcal{F}(f) \circ V_i^T = V_i^{T'} \circ f^*$ where $f^*$ is the induced map on inputs.

**Predicate Decidability Analysis:**

Each gate predicate $P_i^T$ belongs to one of three decidability classes:

| Gate | Predicate | Decidability Class | Witness Type | Undecidability Source |
|------|-----------|-------------------|--------------|----------------------|
| 1 (Energy) | $\Phi(x) < M$ | $\Sigma_1^0$ (semi-decidable) | $(x, \Phi(x), M)$ | Infinite sup over time |
| 3 (Compact) | $\exists V: \mu(B_\varepsilon(V)) > 0$ | $\Sigma_1^0$ | $(V, \varepsilon, \mu_{\mathrm{witness}})$ | Profile enumeration |
| 4 (Scale) | $\alpha < \beta + \lambda_c$ | Decidable | $(\alpha, \beta, \lambda_c)$ | None (arithmetic) |
| 7 (Stiff) | $\|\nabla\Phi\| \geq C|\Delta\Phi|^\theta$ | $\Pi_2^0$ | $(C, \theta, \mathrm{gradient\_bound})$ | Infimum over manifold |
| 17 (Lock) | $\operatorname{Hom}(\mathbb{H}_{\mathrm{bad}}, -) = \emptyset$ | Undecidable in general | Obstruction cocycle | Rice's Theorem |

*Decidability Mechanisms:*
- **Semi-decidable ($\Sigma_1^0$):** Predicate can be verified by finite search if true, but may loop if false. Resolution: introduce timeout with $K^{\mathrm{inc}}$ fallback.
- **Decidable:** Both truth and falsity can be determined in finite time. Resolution: direct evaluation.
- **Undecidable ($\Pi_2^0$ or higher):** No general algorithm exists. Resolution: tactic library (E1-E12) with $K^{\mathrm{inc}}$ exhaustion.

**Decidability Contingencies:** The complexity classifications above assume:
- **Rep-Constructive:** Computable representation of system states (e.g., constructive reals with effective moduli of continuity)
- **Cert-Finite($T$):** Finite certificate alphabet for type $T$
- **Explicit backends:** Effective computation of functionals $\Phi$, $\mathfrak{D}$, moduli bounds

Without these assumptions, semi-decidable gates may return $K^{\mathrm{inc}}$ on all inputs. The framework is sound regardless—$K^{\mathrm{inc}}$ routes to fallback—but decidability guarantees are contingent on the effective layer.

*Decidability-Preserving Approximation:* For predicates in $\Pi_2^0$ or higher:
1. Replace universal quantifier with finite approximation: $\forall x \in X$ becomes $\forall x \in X_N$ for truncation $X_N$
2. Add precision certificate: $K^{\mathrm{approx}} := (N, \epsilon, \|P - P_N\|)$
3. Propagate approximation error through the Sieve via error composition rules

**Formal Witness Structure:**

Each certificate $K_i^+$ has a formally specified **witness type** $W_i^T$:

```
Witness[EnergyCheck] := {
  state: X,
  energy_value: ℝ₊,
  bound: ℝ₊,
  proof: energy_value < bound
}

Witness[CompactCheck] := {
  profile: 𝒫,           -- from profile library
  scale: ℝ₊,             -- concentration scale ε
  mass: ℝ₊,              -- μ(Bε(V))
  proof: mass > 0
}

Witness[ScaleCheck] := {
  alpha: ℝ,              -- energy scaling exponent
  beta: ℝ,               -- dissipation scaling exponent
  lambda_c: ℝ,           -- critical threshold
  proof: alpha < beta + lambda_c
}

Witness[StiffnessCheck] := {
  equilibrium: X,        -- point where LS holds
  constant_C: ℝ₊,        -- Łojasiewicz constant
  exponent_theta: (0,1), -- Łojasiewicz exponent
  gradient_bound: ℝ₊,    -- ‖∇Φ(x)‖ lower bound
  proof: gradient_bound ≥ C·|Φ(x)-Φ_min|^θ
}

Witness[LockCheck] := {
  obstruction_class: H^*(ℋ_bad; Ω), -- cohomological obstruction
  tactic_trace: List[TacticResult],  -- E1-E12 outcomes
  hom_emptiness: Hom = ∅ ∨ Witness[morph],
  proof: tactic_trace ⊢ hom_emptiness
}
```

*Witness Validity Invariant:* For all certificates $K_i^+$:

$$
\operatorname{Valid}(K_i^+) \Leftrightarrow \exists w \in W_i^T.\, \operatorname{Verify}(w) = \mathrm{true} \wedge \operatorname{Extract}(K_i^+) = w
$$

*Proof (5 Steps).*

*Step 1 (Predicate Extraction).* From type $T$'s structural data, extract the semantic content of each gate predicate $P_i^T$:
- EnergyCheck: $P_1^T(x) \equiv \Phi(x) < \infty$ (finite energy) — **Decidability:** $\Sigma_1^0$ via numerical evaluation
- CompactCheck: $P_3^T(x) \equiv \exists V \in \mathcal{L}_T: \mu(B_\varepsilon(V)) > 0$ (concentration) — **Decidability:** $\Sigma_1^0$ via profile search
- ScaleCheck: $P_4^T(x) \equiv \alpha(x) < \beta(x) + \lambda_c$ (subcriticality) — **Decidability:** Decidable (arithmetic)
- StiffnessCheck: $P_7^T(x) \equiv \|\nabla\Phi(x)\| \geq C|\Phi(x) - \Phi_{\min}|^\theta$ (Łojasiewicz) — **Decidability:** $\Pi_2^0$ via variational methods

The predicates are derived from the user-supplied $(\Phi, \mathfrak{D}, G)$ using type-specific templates from the {ref}`Gate Catalog <sec-node-specs>`.

*Step 2 (Verifier Construction).* For each gate $i$, construct verifier $V_i^T: X \times \Gamma \to \{\mathrm{YES}, \mathrm{NO}\} \times \mathcal{K}_i$:
1. **Input parsing:** Extract relevant state $x$ and context certificates $\Gamma$
2. **Predicate evaluation:** Compute $P_i^T(x)$ using functional evaluation of $\Phi, \mathfrak{D}$
3. **Certificate generation:** If $P_i^T(x)$ holds, produce $K_i^+ = (x, \text{witness})$; otherwise produce $K_i^- = (x, \text{failure\_data})$

*Step 3 (Soundness).* The verifier is sound: $V_i^T(x, \Gamma) = (\mathrm{YES}, K_i^+) \Rightarrow P_i^T(x)$.

*Proof.* By construction, $K_i^+$ is only produced when the verifier confirms $P_i^T(x)$. The certificate carries a witness: for EnergyCheck, this is $(\Phi(x), \mathrm{bound})$; for CompactCheck, this is $(V, \varepsilon, \mu(B_\varepsilon(V)))$. The witness data certifies the predicate by inspection. This is the Curry-Howard correspondence {cite}`HoTTBook`: the certificate $K_i^+$ is a proof term for proposition $P_i^T(x)$.

*Step 4 (Completeness).* For each gate, the verifier covers all cases:
- If $P_i^T(x)$ holds: returns $(\mathrm{YES}, K_i^+)$ with witness
- If $\neg P_i^T(x)$ is finitely refutable: returns $(\mathrm{NO}, K_i^{\mathrm{wit}})$ with counterexample
- If undecidable or negation not finitely witnessable: returns $(\mathrm{INC}, K_i^{\mathrm{inc}})$ with obligation ledger

The three outcomes partition all inputs. No verifier returns $\bot$ (undefined).

**Note:** $K^{\mathrm{wit}}$ (counterexample) is available only for predicates with finitely refutable negations—e.g., finite-dimensional checks, SMT-reducible constraints, explicit blow-up constructions. For predicates requiring infinite witness data (e.g., "no blow-up ever occurs"), negation produces $K^{\mathrm{inc}}$ routing to barriers/surgery.

*Step 5 (Canonicity).* The verifier is **canonical** in the sense that it depends only on:
- Type template $T$ (fixed at framework design time)
- User-supplied functionals $(\Phi, \mathfrak{D}, G)$
- Prior certificates in context $\Gamma$

Two instantiations with the same inputs produce identical verifiers. This ensures reproducibility across Sieve runs.

**Literature:** Type-theoretic verification {cite}`HoTTBook`; certified programming {cite}`Leroy09`; predicate abstraction {cite}`GrafSaidi97`.

$\square$
:::

::::

---

(sec-tm2-barrier-implementation)=
## TM-2: Barrier Implementation Factory

:::{div} feynman-prose
When a gate says NO---when your solution fails some check---what happens next? This is where barriers come in. A barrier is a mathematical argument that says: "Even though this check failed, the obstruction cannot persist. Here is why."

The Barrier Factory takes a type $T$ and its literature of theorems, and produces barrier implementations for every gate failure mode. The key property is *non-circularity*: the barrier cannot assume the very thing it is trying to prove. This sounds obvious, but it is precisely the kind of subtle bug that destroys verification systems.

Think of barriers as the framework's immune system. The gate detects something wrong; the barrier determines whether it is a transient fluctuation (blocked) or a genuine problem requiring surgery (breached). The factory guarantees every gate NO path has at least one barrier watching it.
:::

:::{prf:theorem} [FACT-Barrier] Barrier Implementation Factory
:label: mt-fact-barrier

For any system of type $T$, there exist default barrier implementations with correct outcomes and non-circular preconditions:

**Input**: Type $T$ + available literature lemmas

**Output**: For each barrier $B$:
- Default implementation $\mathcal{B}^T$
- Blocked/Breached certificate generators
- Scope specification

**Properties**:
1. Non-circularity: Trigger predicate not in Pre
2. Certificate validity: Outputs satisfy contract
3. Completeness: At least one barrier per node NO path

**Literature:** Epsilon-regularity theorems {cite}`CaffarelliKohnNirenberg82`; Foster-Lyapunov barriers {cite}`MeynTweedie93`; singularity barriers {cite}`Hamilton82`; barrier certificates {cite}`Prajna04`.
:::

:::{prf:proof}
:label: proof-mt-fact-barrier

*Step 1 (Barrier Catalog).* For each gate NO outcome, identify the corresponding barrier type from literature:
- EnergyCheck NO $\to$ Foster-Lyapunov barrier (drift $\mathcal{L}V \leq -\gamma V + C\mathbf{1}_K$)
- CompactCheck NO $\to$ Scattering barrier (dispersion estimate $\|u\|_{L^p L^q} \leq C$)
- ScaleCheck NO $\to$ Type II barrier (monotonicity formula violation)
- GeomCheck NO $\to$ Capacity barrier ($\epsilon$-regularity: $\mathcal{H}^{n-2}(S) = 0$)

Each barrier is instantiated from the corresponding literature theorem by substituting the type-specific functionals $(\Phi, \mathfrak{D})$.

*Step 2 (Non-Circularity Verification).* For each barrier $\mathcal{B}_j$, verify the dependency constraint:

$$
\mathrm{Trig}(\mathcal{B}_j) \cap \mathrm{Pre}(V_i) = \emptyset
$$

where $V_i$ is the gate that triggers $\mathcal{B}_j$. This is checked syntactically: the trigger predicate $\mathrm{Trig}(\mathcal{B}_j)$ uses quantities from $K_i^-$ (the gate's NO output), while $\mathrm{Pre}(V_i)$ uses quantities from $\Gamma$ (prior context). Since $K_i^- \not\in \Gamma$ at evaluation time, circularity is impossible.

*Step 3 (Barrier Soundness).* Each barrier implementation is sound in two directions:
- **Blocked soundness:** If $\mathcal{B}_j$ returns $K_j^{\mathrm{blk}}$, then the obstruction genuinely cannot persist. For Foster-Lyapunov barriers, this follows from {cite}`MeynTweedie93` Theorem 15.0.1: the drift condition implies geometric ergodicity, so unbounded energy is transient.
- **Breached soundness:** If $\mathcal{B}_j$ returns $K_j^{\mathrm{br}}$, the barrier method is insufficient to exclude the obstruction. For capacity barriers: $\mathrm{Cap}(\Sigma) > \varepsilon_{\mathrm{reg}}$ means epsilon-regularity ({cite}`CaffarelliKohnNirenberg82`) cannot be applied; singularity is *not excluded* but also *not proven*. Breached is a routing signal for surgery/fallback pathways, not a semantic guarantee that singularity exists.

*Step 4 (Certificate Production).* Given trigger activation, the barrier implementation produces certificates with full payload:
- **Blocked:** $K_B^{\mathrm{blk}} = (\mathrm{barrier\_type}, \mathrm{obstruction}, \mathrm{bound}, \mathrm{literature\_ref})$
- **Breached:** $K_B^{\mathrm{br}} = (\mathrm{mode}, \mathrm{profile}, \mathrm{surgery\_data}, \mathrm{capacity})$

The payload structure ensures downstream consumers (surgery, Lock) have all necessary information without re-querying.

*Step 5 (Completeness).* For each gate NO path, at least one barrier is defined and reachable:
- The barrier catalog covers all 17 gates with NO outcomes
- Each barrier terminates (finite computation on bounded data)
- The union $\bigcup_j \mathrm{Trig}(\mathcal{B}_j) \supseteq \bigcup_i \{K_i^-\}$ ensures no NO certificate is orphaned

$\square$
:::

---

(sec-tm3-surgery-schema)=
## TM-3: Surgery Schema Factory

:::{div} feynman-prose
When barriers are breached---when the framework cannot prove the obstruction is transient---you need surgery. This is controlled demolition: you excise the bad region, cap it off with something well-behaved, and continue.

The Surgery Factory is perhaps the most delicate part of the whole system. It takes a type $T$ and its profile library, and produces surgery operators that: (1) are well-defined (the gluing actually works), (2) preserve the essential structure, and (3) make progress (you cannot loop forever doing surgeries).

The key insight is the *progress measure*. Every surgery strictly decreases a well-founded quantity. This is the mathematical equivalent of proving your algorithm terminates. Without it, you could have solutions that bounce forever between surgery and re-entry, never reaching completion.

Notice the fallback: if a type does not admit surgery at all, the factory honestly says so and routes to reconstruction. The framework never pretends to do more than it can.
:::

:::{prf:theorem} [FACT-Surgery] Surgery Schema Factory
:label: mt-fact-surgery

For any type $T$ admitting surgery, there exist default surgery operators matching diagram re-entry targets:

**Input**: Type $T$ + canonical profile library + admissibility interface

**Output**: For each surgery $S$:
- Surgery operator $\mathcal{O}_S^T$
- Admissibility checker
- Re-entry certificate generator
- Progress measure

**Fallback**: If type $T$ does not admit surgery, output "surgery unavailable" certificate ($K_{\mathrm{Surg}}^{\mathrm{inc}}$) routing to reconstruction ({prf:ref}`mt-lock-reconstruction`).

**Literature:** Hamilton-Perelman surgery {cite}`Hamilton97`; {cite}`Perelman03`; surgery in mean curvature flow {cite}`HuiskenSinestrari09`; well-founded orderings {cite}`Aczel77`.

:::

:::{prf:proof}
:label: proof-mt-fact-surgery

*Step 1 (Profile-Surgery Correspondence).* For each canonical profile $\mathcal{P}_i \in \mathcal{L}_T$, identify the corresponding surgery operator from literature:
- Concentration profile $\to$ bubble extraction (blow-up analysis + rescaling)
- Traveling wave profile $\to$ wave removal (modulation + subtraction)
- Soliton profile $\to$ soliton surgery (spectral decomposition)
- Neck singularity $\to$ neck-pinch surgery (Hamilton-Perelman construction)

The correspondence is type-specific and encoded in the profile library: each $\mathcal{P}_i$ has an attached surgery recipe $\mathcal{O}_i$ derived from the regularity theory for type $T$.

*Step 2 (Surgery Well-Definedness).* For each surgery operator $\mathcal{O}_S^T$, verify well-definedness:
- **Domain:** The surgery is defined on the set $\{x : \exists (\Sigma, V) \text{ admissible at } x\}$ where $V \in \mathcal{L}_T$
- **Pushout existence:** The categorical pushout $\mathcal{X}' = (\mathcal{X} \setminus B_\varepsilon(\Sigma)) \sqcup_\partial \mathcal{X}_{\mathrm{cap}}$ exists by completeness of the ambient category
- **Gluing smoothness:** The capping region $\mathcal{X}_{\mathrm{cap}}$ matches the excised boundary $\partial B_\varepsilon(\Sigma)$ by asymptotic analysis of the profile $V$

*Step 3 (Admissibility Verification).* The admissibility checker tests the surgery preconditions:
- **Scale separation:** $\lambda_{\mathrm{sing}} \ll \lambda_{\mathrm{bulk}}$ ensures the singularity is localized
- **Isolation:** Singularity regions $\Sigma_1, \ldots, \Sigma_k$ are pairwise disjoint
- **Energy bound:** $\Phi(\mathrm{extracted}) \leq \delta \cdot \Phi(\mathrm{total})$ for small $\delta$ ensures bounded energy loss
- **Capacity bound:** $\mathrm{Cap}(\Sigma) \leq \varepsilon_{\mathrm{adm}}$ by {cite}`Federer69` Theorem 2.10.19

If any condition fails, return $K_{\mathrm{inadm}}$ routing to reconstruction ({prf:ref}`mt-lock-reconstruction`).

*Step 4 (Progress Measure).* Define the well-founded progress measure:

$$
\mathcal{P}(x, N_S) = (N_{\max} - N_S, \Phi_{\mathrm{residual}}(x)) \in \omega \times [0, \infty)
$$

ordered lexicographically. Each surgery strictly decreases $\mathcal{P}$:
- $N_S \mapsto N_S + 1$ strictly decreases the first component
- $\Phi_{\mathrm{residual}}$ decreases by at least $\delta_{\mathrm{surgery}} > 0$ per surgery by {cite}`Perelman03` Lemma 4.3

Since $\mathcal{P}$ takes values in a well-founded order, termination follows.

*Step 5 (Re-entry Certificate).* Upon successful surgery, generate re-entry certificate:

$$
K^{\mathrm{re}} = (\mathcal{O}_S, (\Sigma, V), x', \Phi(x') < \Phi(x^-), N_S + 1)
$$

The certificate attests:
- The surgery $\mathcal{O}_S$ was applied to singularity $(\Sigma, V)$
- The surgered state $x' = \mathcal{O}_S(x^-)$ satisfies gate preconditions for re-entry
- Energy decreased: $\Phi(x') < \Phi(x^-)$ (strict by excision)
- Surgery count incremented

$\square$
:::

---

(sec-tm4-equivalence-transport)=
## TM-4: Equivalence + Transport Factory

:::{div} feynman-prose
Here is something subtle but tremendously powerful. Many solutions that look different are actually *equivalent*---related by scaling, rotation, gauge transformation, or some other symmetry. If you have proved something about solution $u$, you should be able to transport that proof to any equivalent solution $u'$.

The Transport Factory makes this automatic. It takes the type's symmetry structure and produces rules for: (1) recognizing when two solutions are equivalent, (2) transporting certificates from one to the other, and (3) upgrading "equivalent-YES" to "genuine-YES" when the equivalence is small enough.

This is the univalence principle in action: equivalent things are equal for all practical purposes. The factory ensures that the Sieve respects this principle---you do not re-verify properties that you have already verified up to equivalence.

The promotion rules are the clever part. If you know something holds up to a scaling factor $\lambda$, and later you learn $|\lambda - 1| < \epsilon$, you can upgrade your certificate. The framework accumulates evidence and applies it retroactively.
:::

:::{prf:theorem} [FACT-Transport] Equivalence + Transport Factory
:label: mt-fact-transport

For any type $T$, there exists a library of admissible equivalence moves and transport lemmas:

**Input**: Type $T$ structural assumptions

**Output**:
- Equivalence moves $\mathrm{Eq}_1^T, \ldots, \mathrm{Eq}_k^T$ with comparability bounds (instantiations of {prf:ref}`def-equiv-symmetry`--{prf:ref}`def-equiv-bridge`)
- Transport lemmas $T_1^T, \ldots, T_6^T$ instantiated for $T$
- YES$^\sim$ production rules
- Promotion rules (immediate and a-posteriori)

**Literature:** Transport of structure in category theory {cite}`MacLane71`; univalent transport {cite}`HoTTBook`; symmetry classification {cite}`Olver93`.

:::

:::{prf:proof}
:label: proof-mt-fact-transport

*Step 1 (Equivalence Instantiation).* For each abstract equivalence $\mathrm{Eq}_i$, instantiate using the type's structural assumptions:
- {prf:ref}`def-equiv-symmetry` (Scaling): $u \sim_\lambda \lambda^{\alpha} u(\lambda^\beta \cdot)$ with exponents from $T$'s critical scaling
- {prf:ref}`def-equiv-metric` (Symmetry): $u \sim_g g \cdot u$ for $g \in G$ the type's symmetry group
- {prf:ref}`def-equiv-conjugacy` (Gauge): $u \sim_\phi e^{i\phi} u$ for gauge-invariant types
- {prf:ref}`def-equiv-surgery-id` (Modulation): $u \sim_{c,x} u(\cdot - c t) e^{i(c\cdot + x)}$ for dispersive types

The comparability bounds follow from the type's energy functional: $|\Phi(u) - \Phi(u')| \leq C \cdot d_{\mathrm{Eq}}(u, u')$.

*Step 2 (Transport Soundness).* For each transport lemma $T_i^T$, verify soundness: if $P(u)$ holds (witnessed by $K_P^+(u)$) and $u \sim u'$ (witnessed by $K_{\mathrm{Eq}}(u, u')$), then $P(u')$ holds.

(proof-mt-fact-transport-soundness)=
*Proof of soundness.* The transported certificate $T_P(K_P^+(u), K_{\mathrm{Eq}}) = K_P^{\sim}(u')$ carries:
- The original witness transformed by the equivalence
- The equivalence parameter (scaling factor, group element, etc.)
- A bound on the transport error: $\|P(u') - P(u)\| \leq C \cdot d_{\mathrm{Eq}}(u, u')$

This follows the univalence principle {cite}`HoTTBook`: equivalent types have equivalent properties.

*Step 3 (YES$^\sim$ Production).* The production rules for YES$^\sim$ (equivalent-YES) are:

$$
\frac{V_i^T(u, \Gamma) = (\mathrm{YES}, K_i^+) \quad u \sim u' \quad K_{\mathrm{Eq}}(u, u')}{V_i^T(u', \Gamma') = (\mathrm{YES}^\sim, T_i(K_i^+, K_{\mathrm{Eq}}))}
$$

The rule is applied automatically by the Sieve when an equivalence certificate is in context.

*Step 4 (Promotion Rules).* YES$^\sim$ promotes to YES$^+$ under bounded equivalence parameters:
- **Immediate promotion:** If $|\lambda - 1| < \epsilon_{\mathrm{prom}}$ or $d(g, e) < \delta_{\mathrm{prom}}$, the equivalence is "small" and YES$^\sim$ becomes YES$^+$
- **A-posteriori promotion:** If later gates provide stronger bounds that retroactively satisfy the promotion condition, apply promotion during closure ({prf:ref}`mt-up-inc-aposteriori`)

Promotion thresholds $\epsilon_{\mathrm{prom}}, \delta_{\mathrm{prom}}$ are type-specific and derived from the comparability bounds.

*Step 5 (Completeness of Equivalence Library).* The equivalence library is complete for type $T$ if:

$$
\forall u, u' \in X: u \sim_T u' \Rightarrow \exists i: u \sim_{\mathrm{Eq}_i} u'
$$

where $\sim_T$ is the type's intrinsic equivalence relation. For well-studied types (NLS, Navier-Stokes, Ricci flow), this follows from the classification of symmetries in {cite}`Olver93`.

$\square$
:::

---

(sec-tm5-lock-backend)=
## TM-5: Lock Backend Factory

:::{div} feynman-prose
The Lock is the framework's last line of defense. When all else fails---gates, barriers, surgery---the Lock asks: is there any way for a bad pattern to persist? If the Lock can prove the answer is no, you are safe. If not, the framework honestly admits it cannot decide.

The Lock Backend Factory produces *tactics*---systematic methods for proving obstructions cannot exist. These range from simple geometric arguments (the singularity set is too small to matter) to sophisticated cohomological machinery (the pattern would require a non-trivial homomorphism that provably does not exist).

What makes this subtle is that the question "does a bad pattern persist?" is often *undecidable* in the formal sense. No algorithm can answer it in all cases. The factory handles this by: (1) trying a sequence of increasingly powerful tactics, (2) using timeouts for semi-decidable checks, and (3) admitting failure honestly when tactics are exhausted.

This is *honest incompleteness*. The system never claims more certainty than it has. When the Lock emits $K^{\mathrm{inc}}$, it means: "I tried everything I know, and I could not decide. Here is what I learned along the way." That partial progress may be useful for human mathematicians to take over.
:::

:::{prf:theorem} [FACT-Lock] Lock Backend Factory
:label: mt-fact-lock

For any type $T$ with $\mathrm{Rep}_K$ available, there exist E1--E10 tactics for the Lock:

**Input**: Type $T$ + representation substrate

**Output**:
- Tactic implementations $E_1^T, \ldots, E_5^T$
- Automation level indicators
- Horizon fallback procedure

**Rep unavailable**: If $\mathrm{Rep}_K$ is not available, Lock uses only E1--E3 (geometry-based tactics) with limited automation.

**Literature:** Automated theorem proving {cite}`BaaderNipkow98`; invariant theory {cite}`MumfordFogartyKirwan94`; obstruction theory {cite}`EilenbergSteenrod52`.

:::

:::{prf:proof}
:label: proof-mt-fact-lock

*Step 1 (Tactic Classification).* The Lock backend tactics E1--E5 are instantiated from the type's representation substrate $\mathrm{Rep}_K$:
- E1 (Geometric): Direct geometric obstruction (Hausdorff dimension, capacity bounds)
- E2 (Topological): Homotopy/homology invariants preventing continuation
- E3 (Variational): Critical point theory, mountain pass arguments
- E4 (Cohomological): Čech/sheaf cohomology obstructions from $\mathrm{Rep}_K$
- E5 (Representation-theoretic): Invariant theory using $G$-equivariance

Each tactic $E_i$ has a **decidability class**: E1--E3 are decidable **under the effective layer** (Rep-Constructive + Cert-Finite($T$) + explicit invariant computation backends); E4--E5 require $\mathrm{Rep}_K$ and may be semi-decidable. Without the effective layer, E3 (mountain pass) involves global optimization and may return $K^{\mathrm{inc}}$.

*Step 2 (Tactic Soundness).* For each tactic $E_i^T$, prove soundness: if $E_i^T$ returns BLOCKED, then genuinely $\operatorname{Hom}(\mathbb{H}_{\mathrm{bad}}, \mathcal{H}) = \emptyset$.

**Per-tactic soundness:**
- **E1 (Geometric):** If $\dim(\Sigma) < n - 2$, then singularity set has zero capacity and cannot support a genuine obstruction by {cite}`Federer69`
- **E2 (Topological):** If $\pi_k(\mathcal{X} \setminus \Sigma) \neq \pi_k(\mathcal{X})$ for some $k$, the topological type changed, blocking any pattern-preserving morphism
- **E3 (Variational):** If $\Phi$ has no critical points in the bad region by mountain pass {cite}`AmbrosettiRabinowitz73`, no stationary singularity exists
- **E4 (Cohomological):** If the Čech cohomology obstruction class $[\omega] \in \check{H}^k$ is non-trivial, the pattern cannot extend by {cite}`Grothendieck67`
- **E5 (Representation):** If $\operatorname{Hom}_G(\rho_{\mathrm{bad}}, \rho_{\mathcal{H}}) = 0$ by Schur orthogonality, no equivariant morphism exists

*Step 3 (Tactic Exhaustiveness).* The tactics are ordered by strength and applied sequentially:

$$
E_1^T \to E_2^T \to E_3^T \to E_4^T \to E_5^T \to \mathrm{Horizon}
$$

Each tactic is **complete for its class**: E1 catches all geometric obstructions, E2 catches all topological obstructions, etc. The union covers all known obstruction mechanisms for type $T$.

*Step 4 (Horizon Fallback).* If all tactics fail, the Lock enters horizon mode:
- Emit $K_{\mathrm{Lock}}^{\mathrm{inc}} = (\mathrm{tactics\_exhausted}, \{E_1, \ldots, E_5\}, \mathrm{partial\_progress})$
- The certificate records which tactics were tried and any partial progress (near-obstructions, dimension bounds)
- Route to {prf:ref}`mt-lock-reconstruction` for explicit construction attempt

This ensures **honest incompleteness**: the system admits when the problem exceeds current theory rather than producing false positives.

*Step 5 (Termination).* Lock evaluation terminates:
- Each tactic $E_i^T$ terminates in finite time (decidable or semi-decidable with timeout)
- The tactic sequence has fixed length (5 tactics)
- Total Lock evaluation time is bounded: $T_{\mathrm{Lock}} \leq \sum_{i=1}^5 T_{E_i} + T_{\mathrm{horizon}} < \infty$

For semi-decidable tactics (E4, E5), a timeout mechanism ensures termination: if $E_i$ exceeds $T_{\max}$, it returns "inconclusive for this tactic" and passes to $E_{i+1}$.

$\square$
:::
