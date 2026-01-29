# The Kernel

:::{div} feynman-prose
Now we come to the heart of the matter. In Part IV we saw the sieve diagram---all those nodes and edges and colored boxes. But a diagram is just a picture. What makes it a *proof* rather than a pretty flowchart?

Here is the key insight: every time the sieve moves from one node to another, it produces a *certificate*---a formal witness that says "I checked this property, and here is the evidence." These certificates accumulate. By the time we reach VICTORY, we do not just *claim* global regularity; we have built up an auditable trail of evidence that can be machine-verified.

This is what computer scientists call a "proof-carrying program." The execution *is* the proof. You cannot separate the two. If the sieve says a system is regular, you can ask to see the receipts---and they will be there.
:::

(sec-sieve-proof-carrying)=
## The Sieve as a Proof-Carrying Program

:::{prf:definition} Sieve epoch
:label: def-sieve-epoch

An **epoch** is a single execution of the sieve from the START node to either:
1. A terminal node (VICTORY, Mode D.D, or FATAL ERROR), or
2. A surgery re-entry point (dotted arrow target).
Each epoch visits finitely many nodes ({prf:ref}`thm-epoch-termination`). A complete run consists of finitely many epochs ({prf:ref}`thm-finite-runs`).

:::

:::{prf:definition} Node numbering
:label: def-node-numbering

The sieve contains the following node classes:
- **Gates (Blue):** Nodes 1--17 performing interface permit checks
- **Barriers (Orange):** Secondary defense nodes triggered by gate failures
- **Modes (Red):** Failure mode classifications
- **Surgeries (Purple):** Repair mechanisms with re-entry targets
- **Actions (Purple):** Dynamic restoration mechanisms (SSB, Tunneling)
- **Restoration subnodes (7a--7d):** The stiffness restoration subtree

:::

:::{div} feynman-prose
Think of an epoch like a single pass through airport security. You start at the entrance, you go through various checkpoints---metal detector, bag scanner, maybe a secondary inspection---and you end up either at your gate (VICTORY), turned away (failure mode), or sent back to try again with modified equipment (surgery re-entry).

The crucial property is that each pass is *finite*. You cannot get stuck in an infinite loop within a single epoch. The DAG structure guarantees this.
:::

---

(sec-operational-semantics)=
## Operational Semantics

:::{div} feynman-prose
Now let us be precise about what happens at each node. This is the operational semantics---the rules of the game. We need to specify:
- What is a "state" of the system?
- What does it mean to "evaluate" a node?
- How do certificates flow from node to node?

The beauty of this formalism is that once we fix these definitions, the sieve becomes a deterministic machine. Given the same input, it will always produce the same output. There is no ambiguity, no room for hand-waving.
:::

:::{prf:definition} State space
:label: def-state-space

Let $X$ be a Polish space (complete separable metric space) representing the configuration space of the system under analysis. A **state** $x \in X$ is a point in this space representing the current system configuration at a given time or stage of analysis.

:::

:::{prf:definition} Certificate
:label: def-certificate

A **certificate** $K$ is a formal witness object that records the outcome of a verification step. Certificates are typed: each certificate $K$ belongs to a certificate type $\mathcal{K}$ specifying what property it witnesses.

:::

:::{prf:definition} Context
:label: def-context

The **context** $\Gamma$ is a finite **set** of certificates accumulated during a sieve run:

$$
\Gamma = \{K_{D_E}, K_{\mathrm{Rec}_N}, K_{C_\mu}, \ldots, K_{\mathrm{Cat}_{\mathrm{Hom}}}\}

$$

The context grows monotonically during an epoch: certificates are added but never removed (except at surgery re-entry, where context may be partially reset). Duplicate certificates are idempotent under set union.

:::

:::{prf:definition} Node evaluation function
:label: def-node-evaluation

Each node $N$ in the sieve defines an **evaluation function**:

$$
\mathrm{eval}_N : X \times \Gamma \to \mathcal{O}_N \times \mathcal{K}_N \times X \times \Gamma

$$

where:
- $\mathcal{O}_N$ is the **outcome alphabet** for node $N$
- $\mathcal{K}_N$ is the **certificate type** produced by node $N$
- The function maps $(x, \Gamma) \mapsto (o, K_o, x', \Gamma')$ where:
   - $o \in \mathcal{O}_N$ is the outcome
   - $K_o \in \mathcal{K}_N$ is the certificate witnessing outcome $o$
   - $x' \in X$ is the (possibly modified) state
   - $\Gamma' = \Gamma \cup \{K_o\}$ is the extended context

:::

:::{prf:definition} Edge validity
:label: def-edge-validity

An edge $N_1 \xrightarrow{o} N_2$ in the sieve diagram is **valid** if and only if:

$$
K_o \Rightarrow \mathrm{Pre}(N_2)

$$

That is, the certificate produced by node $N_1$ with outcome $o$ logically implies the precondition required for node $N_2$ to be evaluable.

:::

:::{prf:definition} Determinism policy
:label: def-determinism

For **soft checks** (where the predicate cannot be definitively verified), the sieve adopts the following policy:
- If verification succeeds: output YES with positive certificate $K^+$
- If verification fails: output NO with witness certificate $K^{\mathrm{wit}}$
- If verification is inconclusive (UNKNOWN): output NO with inconclusive certificate $K^{\mathrm{inc}}$
This ensures the sieve is deterministic: UNKNOWN is conservatively treated as NO, routing to the barrier defense layer.
By Binary Certificate Logic ({prf:ref}`def-typed-no-certificates`), the NO certificate is the coproduct

$$
K^- := K^{\mathrm{wit}} + K^{\mathrm{inc}}
$$
so routing is always binary even when epistemic status differs.

:::

:::{div} feynman-prose
Here is a point that trips people up: Why do we treat UNKNOWN as NO? Is that not overly conservative?

Think about it this way. The sieve is trying to *prove* regularity. If we cannot prove a property holds, we cannot claim victory on that basis. So we route to the barrier layer, which gives us a second chance to establish a weaker condition. If the barrier also fails, we try surgery.

The key insight is that the NO-inconclusive certificate ($K^{\mathrm{inc}}$) is *different* from a NO-with-witness certificate ($K^{\mathrm{wit}}$). The former says "I could not prove it"; the latter says "I found a counterexample." Only the latter is fatal. The former triggers reconstruction---maybe we need a better proof technique, or more refined templates, or additional assumptions.

This is honest bookkeeping. We do not pretend to know things we do not know.
:::

---

(sec-permit-vocabulary)=
## Permit Vocabulary and Certificate Types

:::{div} feynman-prose
Now we catalog the different kinds of certificates the sieve can produce. Think of this as the vocabulary of the proof language.

There are several categories:
- **Gate permits**: The basic YES/NO certificates from the blue diagnostic nodes
- **Barrier permits**: Blocked/Breached certificates from the orange fallback checks
- **Surgery permits**: Re-entry certificates that authorize resuming the sieve after a repair
- **Promotion permits**: Certificates that upgrade weaker guarantees to stronger ones
- **Derived witness certificates**: Auxiliary bound witnesses used for analytic bridge admissibility
  (see {prf:ref}`def-witness-certificates-bounds`)

The beautiful thing is that all these certificates fit together like puzzle pieces. The precondition of one node is satisfied by the postcondition of another. The whole system is type-safe in a precise sense.
:::

:::{prf:definition} Gate permits
:label: def-gate-permits

For each gate (blue node) $i$, the outcome alphabet is:

$$
\mathcal{O}_i = \{`YES`, `NO`\}

$$

with certificate types:
- $K_i^+$ (`YES` certificate): Witnesses that predicate $P_i$ holds on the current state/window
- $K_i^{\mathrm{wit}}$ (`NO` with witness): Counterexample to $P_i$
- $K_i^{\mathrm{inc}}$ (`NO` inconclusive): Method/budget insufficient to certify $P_i$

We package the NO outcomes as a single type via the coproduct

$$
K_i^- := K_i^{\mathrm{wit}} + K_i^{\mathrm{inc}}
$$
consistent with {prf:ref}`def-typed-no-certificates`.

:::

:::{prf:remark} Dichotomy classifiers
:label: rem-dichotomy

Some gates are **dichotomy classifiers** where NO is a benign branch rather than an error:
- **{prf:ref}`def-node-compact`**: NO = scattering $\to$ global existence (Mode D.D)
- **{prf:ref}`def-node-oscillate`**: NO = no oscillation $\to$ proceed to boundary checks
For these gates, the **witness** branch $K^{\mathrm{wit}}$ encodes the benign classification outcome.
The **inconclusive** branch $K^{\mathrm{inc}}$ still routes along the NO edge but signals reconstruction rather
than a semantic classification.

:::

:::{prf:definition} Barrier permits
:label: def-barrier-permits

For each barrier (orange node), the outcome alphabet is one of:

**Standard barriers** (most barriers):

$$
\mathcal{O}_{\text{barrier}} = \{`Blocked`, `Breached`\}

$$

**Special barriers with extended alphabets:**
- **BarrierScat** (Scattering): $\mathcal{O} = \{`Benign`, `Pathological`\}$
- **BarrierGap** (Spectral): $\mathcal{O} = \{`Blocked`, `Stagnation`\}$
- **BarrierExclusion** (Lock): $\mathcal{O} = \{`Blocked`, `MorphismExists`\}$

Certificate semantics:
- $K^{\mathrm{blk}}$ (`Blocked`): Barrier holds; certificate enables passage to next gate
- $K^{\mathrm{br}}$ (`Breached`): Barrier fails; certificate activates failure mode and enables surgery

:::

:::{prf:definition} Surgery permits
:label: def-surgery-permits

For each surgery (purple node), the output is a **re-entry certificate**:

$$
K^{\mathrm{re}} = (D_S, x', \pi)

$$

where $D_S$ is the surgery data, $x'$ is the post-surgery state, and $\pi$ is a proof that $\mathrm{Pre}(\text{TargetNode})$ holds for $x'$.

The re-entry certificate witnesses that after surgery with data $D_S$, the precondition of the dotted-arrow target node is satisfied:

$$
K^{\mathrm{re}} \Rightarrow \mathrm{Pre}(\text{TargetNode})(x')

$$

:::

:::{prf:definition} YES-tilde permits
:label: def-yes-tilde

A **YES$^\sim$ permit** (YES up to equivalence) is a certificate of the form:

$$
K_i^{\sim} = (K_{\mathrm{equiv}}, K_{\mathrm{transport}}, K_i^+[\tilde{x}])

$$

where:
- $K_{\mathrm{equiv}}$ certifies that $\tilde{x}$ is equivalent to $x$ under an admissible equivalence move
- $K_{\mathrm{transport}}$ is a transport lemma certificate
- $K_i^+[\tilde{x}]$ is a YES certificate for predicate $P_i$ on the equivalent object $\tilde{x}$

YES$^\sim$ permits are accepted by metatheorems that tolerate equivalence.

:::

:::{div} feynman-prose
The YES-tilde permit captures a subtle but important idea. Sometimes we cannot prove a property holds for an object $x$ directly, but we *can* prove it holds for an equivalent object $\tilde{x}$.

For example, suppose we have a function that is messy in one coordinate system but simple in another. We prove the property in the simple coordinates, then transport the result back. The YES$^\sim$ certificate is the formal record of this maneuver.

This is not cheating---the equivalence itself must be certified. We record exactly what equivalence move was made and prove it was legitimate. The whole chain is auditable.
:::

:::{prf:definition} Promotion permits
:label: def-promotion-permits

**Promotion permits** upgrade blocked certificates to full YES certificates:

**Immediate promotion** (past-only): A blocked certificate at node $i$ may be promoted if all prior nodes passed:

$$
K_i^{\mathrm{blk}} \wedge \bigwedge_{j < i} K_j^+ \Rightarrow K_i^+

$$

(Here $K_j^+$ denotes a YES certificate at node $j$.)

**A-posteriori promotion** (future-enabled): A blocked certificate may be promoted after later nodes pass:

$$
K_i^{\mathrm{blk}} \wedge \bigwedge_{j > i} K_j^+ \Rightarrow K_i^+

$$

**Combined promotion**: Blocked certificates may also promote if the barrier's ``Blocked'' outcome combined with other certificates logically implies the original predicate $P_i$ holds.

Promotion rules are applied during context closure ({prf:ref}`def-closure`).

:::

:::{prf:definition} Inconclusive upgrade permits
:label: def-inc-upgrades

**Inconclusive upgrade permits** discharge NO-inconclusive certificates by supplying certificates that satisfy their $\mathsf{missing}$ prerequisites ({prf:ref}`def-typed-no-certificates`).

**Immediate inc-upgrade** (past/current): An inconclusive certificate may be upgraded if certificates satisfying its missing prerequisites are present:

$$
K_P^{\mathrm{inc}} \wedge \bigwedge_{j \in J} K_j^+ \Rightarrow K_P^+

$$

where $J$ indexes the certificate types listed in $\mathsf{missing}(K_P^{\mathrm{inc}})$.

**A-posteriori inc-upgrade** (future-enabled): An inconclusive certificate may be upgraded after later nodes provide the missing prerequisites:

$$
K_P^{\mathrm{inc}} \wedge \bigwedge_{j \in J'} K_j^+ \Rightarrow K_P^+

$$

where $J'$ indexes certificates produced by nodes evaluated after the node that produced $K_P^{\mathrm{inc}}$.

**To YES$^\sim$** (equivalence-tolerant): An inconclusive certificate may upgrade to YES$^\sim$ when the discharge is valid only up to an admissible equivalence move ({prf:ref}`def-yes-tilde`):

$$
K_P^{\mathrm{inc}} \wedge \bigwedge_{j \in J} K_j^+ \Rightarrow K_P^{\sim}

$$

**Discharge condition (obligation matching):** An inc-upgrade rule is admissible only if its premises imply the concrete obligation instance recorded in the payload:

$$
\bigwedge_{j \in J} K_j^+ \Rightarrow \mathsf{obligation}(K_P^{\mathrm{inc}})

$$

This makes inc-upgrades structurally symmetric with blocked-certificate promotions ({prf:ref}`def-promotion-permits`).

:::

:::{div} feynman-prose
Let me make sure you understand what promotion and inc-upgrades are really doing. They are the mechanism by which the sieve learns from its own execution.

Consider this scenario: You fail a check early in the sieve, but later checks provide additional information. With that new information, your original "failure" might now be dischargeable. The promotion rules let us retroactively upgrade a blocked certificate to a full YES.

This is not changing history---it is recognizing that the context has changed. The certificate system tracks dependencies precisely, so we know exactly when such upgrades are valid.

The inc-upgrade rules work similarly for inconclusive certificates. If you could not prove something because you were missing prerequisite facts, and those facts are established later, your inconclusive verdict can be upgraded to a definite YES.

This is how the sieve achieves robustness: it does not give up just because facts arrive out of order.
:::

---

(sec-kernel-theorems)=
## Kernel Theorems

:::{div} feynman-prose
Now we prove the fundamental guarantees of the sieve. These are the theorems that make everything work:

1. **DAG structure**: The sieve has no cycles, so execution cannot loop forever
2. **Epoch termination**: Each pass through the sieve finishes in finite time
3. **Finite runs**: The entire execution (including surgeries) terminates
4. **Soundness**: Every transition is certificate-justified---no cheating allowed

These are not merely desirable properties; they are *necessary* for the sieve to function as a proof system. Without termination, we could not claim to have a decision procedure. Without soundness, our certificates would be meaningless.

Let us prove each one.
:::

:::{prf:theorem} DAG structure
:label: thm-dag

The sieve diagram is a directed acyclic graph (DAG). All edges, including dotted surgery re-entry edges, point forward in the topological ordering. Consequently:
1. No backward edges exist
2. Each epoch visits at most $|V|$ nodes where $|V|$ is the number of nodes
3. The sieve terminates

**Literature:** Topological sorting of DAGs {cite}`Kahn62`; termination via well-founded orders {cite}`Floyd67`.

:::

:::{prf:proof}

By inspection of the diagram: all solid edges flow downward (increasing node number or to barriers/modes), and all dotted surgery edges target nodes strictly later in the flow than their source mode. The restoration subtree (7a--7d) only exits forward to TopoCheck or TameCheck.

:::

:::{div} feynman-prose
You might worry: what about the surgery re-entry arrows? Those dotted lines go back up the diagram---are they not backward edges that could create cycles?

The key is in the word "strictly later." When surgery S7 (say) re-enters at TopoCheck, it enters at a node that is *topologically later* than the failure mode that triggered it. The node numbering respects the DAG structure. So even though the dotted arrow points upward visually, it points forward in the partial order.

This is a crucial design constraint on the sieve diagram. It is not an accident.
:::

:::{prf:theorem} Epoch termination
:label: thm-epoch-termination

Each epoch terminates in finite time, visiting finitely many nodes.

**Literature:** Termination proofs via ranking functions {cite}`Floyd67`; {cite}`Turing49`.

:::

:::{prf:proof}

Immediate from {prf:ref}`thm-dag`: the DAG structure ensures no cycles, hence any path through the sieve has bounded length.

:::

:::{prf:theorem} Finite complete runs
:label: thm-finite-runs

A complete sieve run consists of finitely many epochs.

**Literature:** Surgery bounds for Ricci flow {cite}`Perelman03`; well-founded induction {cite}`Floyd67`.

:::

:::{prf:proof}

Each surgery has an associated progress measure ({prf:ref}`def-progress-measures`):

**Type A (Bounded count)**: The surgery count is bounded by $N(T, \Phi(x_0))$, a function of the time horizon $T$ and initial energy $\Phi(x_0)$. For parabolic PDE, this bound is typically imported from classical surgery theory (e.g., Perelman's surgery bound for Ricci flow: $N \leq C(\Phi_0) T^{d/2}$). For algorithmic/iterative systems, it may be a budget constraint.

**Type B (Well-founded)**:  The complexity measure $\mathcal{C}: X \to \mathbb{N}$ (or ordinal $\alpha$) strictly decreases at each surgery:

$$
\mathcal{O}_S(x) = x' \Rightarrow \mathcal{C}(x') < \mathcal{C}(x)

$$

Since well-founded orders have no infinite descending chains, the surgery sequence terminates.

**Discrete Energy Progress (Type A Strengthening):** When using continuous energy $\Phi: X \to \mathbb{R}_{\geq 0}$, termination requires the **Discrete Progress Constraint** ({prf:ref}`def-progress-measures`): each surgery must drop energy by at least $\epsilon_T > 0$. The Surgery Admissibility Trichotomy ({prf:ref}`mt-resolve-admissibility`) enforces this via the Progress Density condition. Hence:

$$
N_{\text{surgeries}} \leq \frac{\Phi(x_0)}{\epsilon_T} < \infty

$$

The total number of distinct surgery types is finite (at most 17, one per failure mode). Hence the total number of surgeries---and thus epochs---is finite.

:::

:::{div} feynman-prose
This is the deep theorem. A single epoch terminates by the DAG property---but what about the surgeries? Each surgery re-enters the sieve, potentially triggering more surgeries. Could this go on forever?

The answer is no, and the reason is physics (or its mathematical abstraction). Each surgery must make *progress*---either by bounded count (there can only be so many surgeries of type X) or by decreasing some well-founded measure (like energy). You cannot keep cutting away at something forever without eventually finishing.

This is why the progress measures ({prf:ref}`def-progress-measures`) are so important. They are not just bookkeeping; they are the formal guarantees that the surgical repair process terminates.
:::

:::{prf:theorem} Soundness
:label: thm-soundness

Every transition in a sieve run is certificate-justified. Formally, if the sieve transitions from node $N_1$ to node $N_2$ with outcome $o$, then:
1. A certificate $K_o$ was produced by $N_1$
2. $K_o$ implies the precondition $\mathrm{Pre}(N_2)$
3. $K_o$ is added to the context $\Gamma$

**Literature:** Proof-carrying code {cite}`Necula97`; certified compilation {cite}`Leroy09`.

:::

:::{prf:proof}

By construction: {prf:ref}`def-node-evaluation` requires each node evaluation to produce a certificate, and {prf:ref}`def-edge-validity` requires edge validity.

:::

:::{div} feynman-prose
Soundness is a "by construction" property. We designed the sieve so that you *cannot* take a transition without producing a certificate that justifies it. The definitions force this.

This might seem trivial---of course if you define everything to produce certificates, it will produce certificates. But the non-trivial content is in the *coherence* of the certificates. The certificate from node A must actually imply the precondition of node B for the edge A-to-B to be valid. This is checked at design time (when we build the sieve diagram) and at runtime (when we verify the certificates).

The result is a complete audit trail. Every claim can be traced back to its justification.
:::

:::{prf:definition} Fingerprint
:label: def-fingerprint

The **fingerprint** of a sieve run is the tuple:

$$
\mathcal{F} = (\mathrm{tr}, \vec{v}, \Gamma_{\mathrm{final}})

$$

where:
- $\mathrm{tr}$ is the **trace**: ordered sequence of (node, outcome) pairs visited
- $\vec{v}$ is the **node vector**: for each gate $i$, the outcome $v_i \in \{`YES`, `NO`, `---`\}$ (--- if not visited)
- $\Gamma_{\mathrm{final}}$ is the final certificate context

:::

:::{prf:definition} Certificate finiteness condition
:label: def-cert-finite

For type $T$, the certificate language $\mathcal{K}(T)$ satisfies the **finiteness condition** if either:
1. **Bounded description length**: Certificates have bounded description complexity (finite precision, bounded parameters), or
2. **Depth budget**: Closure is computed to a specified depth/complexity budget $D_{\max}$
Non-termination under infinite certificate language is treated as a NO-inconclusive certificate ({prf:ref}`rem-inconclusive-general`).

**Volume 2 convention (finite regime):** Throughout this volume we assume **(1)** bounded description length, so the
certificate language is finite and full closure terminates. The **depth-budget** option is an engineering fallback for
potentially infinite certificate languages; formal guarantees for that regime require additional theorems and are not
claimed here.

:::

:::{prf:definition} Promotion closure
:label: def-closure

The **promotion closure** $\mathrm{Cl}(\Gamma)$ is the least fixed point of the context under all promotion and upgrade rules:

$$
\mathrm{Cl}(\Gamma) = \bigcup_{n=0}^{\infty} \Gamma_n

$$

where $\Gamma_0 = \Gamma$ and $\Gamma_{n+1}$ applies all applicable immediate and a-posteriori promotions ({prf:ref}`def-promotion-permits`) **and all applicable inc-upgrades** ({prf:ref}`def-inc-upgrades`) to $\Gamma_n$.

:::

:::{prf:theorem} Closure termination
:label: thm-closure-termination
:class: rigor-class-f

**Rigor Class:** F (Framework-Original) --- see {prf:ref}`def-rigor-classification`

Under the **bounded-description** regime of the certificate finiteness condition ({prf:ref}`def-cert-finite`, case 1),
the promotion closure $\mathrm{Cl}(\Gamma)$ is computable in finite time. Moreover, the closure is independent of the
order in which upgrade rules are applied.

**Literature:** Knaster-Tarski fixed-point theorem {cite}`Tarski55`; Kleene iteration {cite}`Kleene52`; lattice theory {cite}`DaveyPriestley02`
:::

:::{div} feynman-prose
Here is a theorem that sounds technical but has a beautiful meaning. When you have a bag of certificates and you want to compute all the consequences---all the promotions and upgrades that follow---you keep applying rules until nothing new happens. This process is called taking the "closure."

The question is: does this terminate? And does it matter what order you apply the rules?

The answers are: yes (under finiteness conditions), and no (the order does not matter). This is the Knaster-Tarski fixed point theorem at work. The closure is unique---there is exactly one "complete" context that contains all consequences of your initial certificates.

This is crucial for reproducibility. Two implementations of the sieve that start with the same certificates will compute the same closure, even if they apply rules in different orders.
:::

:::{prf:proof}

*Step 1 (Ambient Setup: Certificate Lattice).* Define the **certificate lattice** $(\mathcal{L}, \sqsubseteq)$ where:
- $\mathcal{L} := \mathcal{P}(\mathcal{K}(T))$ is the power set of all certificates of type $T$
- $\Gamma_1 \sqsubseteq \Gamma_2 :\Leftrightarrow \Gamma_1 \subseteq \Gamma_2$ (set inclusion)
- Meet: $\Gamma_1 \sqcap \Gamma_2 = \Gamma_1 \cap \Gamma_2$
- Join: $\Gamma_1 \sqcup \Gamma_2 = \Gamma_1 \cup \Gamma_2$
- Bottom: $\bot = \emptyset$
- Top: $\top = \mathcal{K}(T)$

The lattice $(\mathcal{L}, \sqsubseteq)$ is **complete** since every subset of $\mathcal{L}$ has a supremum (union) and infimum (intersection).

*Step 2 (Construction: Promotion Operator).* Define the **promotion operator** $F: \mathcal{L} \to \mathcal{L}$ by:

$$
F(\Gamma) := \Gamma \cup \{K' : \exists \text{ rule } R,\, R(\Gamma) \vdash K'\}

$$

where rules $R$ include:
- Immediate and a-posteriori promotions ({prf:ref}`def-promotion-permits`)
- Inc-upgrades ({prf:ref}`def-inc-upgrades`)

*Step 3 (Monotonicity Verification).* We verify $F$ is **monotonic** (order-preserving):

*Claim:* If $\Gamma_1 \sqsubseteq \Gamma_2$, then $F(\Gamma_1) \sqsubseteq F(\Gamma_2)$.

*Proof of Claim:* Let $\Gamma_1 \subseteq \Gamma_2$. Then:
- $\Gamma_1 \subseteq \Gamma_2 \subseteq F(\Gamma_2)$ (by definition of $F$)
- If rule $R$ derives $K'$ from $\Gamma_1$, then $R$ also derives $K'$ from $\Gamma_2$ (rules only require presence of certificates, never absence)
- Thus $F(\Gamma_1) \subseteq F(\Gamma_2)$ $\checkmark$

*Step 4 (Knaster-Tarski Application).* By the **Knaster-Tarski Fixed Point Theorem** {cite}`Tarski55`:

> In a complete lattice $(L, \leq)$, every monotonic function $f: L \to L$ has a **least fixed point** given by:
>
> $$
> \mathrm{lfp}(f) = \bigwedge \{x \in L : f(x) \leq x\}
> $$

Applying this to $(F, \mathcal{L})$:

$$
\mathrm{Cl}(\Gamma) = \mathrm{lfp}_{\Gamma}(F) = \bigcap \{\Gamma' : F(\Gamma') \subseteq \Gamma' \text{ and } \Gamma \subseteq \Gamma'\}

$$

*Step 5 (Finiteness and Termination).* Under bounded description length ({prf:ref}`def-cert-finite`, case 1):
- $|\mathcal{K}(T)| < \infty$, so $|\mathcal{L}| = 2^{|\mathcal{K}(T)|} < \infty$
- The Kleene iteration $\Gamma_0, \Gamma_1, \Gamma_2, \ldots$ where $\Gamma_{n+1} = F(\Gamma_n)$ forms a strictly increasing chain
- By finiteness of $\mathcal{L}$, the chain stabilizes in at most $|\mathcal{K}(T)|$ steps

**Certificate Grammar Assumption:** The finiteness of $\mathcal{K}(T)$ holds because certificates are drawn from a **finite alphabet** $\Sigma_T$ with bounded-length representations:
- Rational parameters have bounded numerator/denominator: $|p/q| \leq M$ with $\gcd(p,q) = 1$ and $|p|, |q| \leq B_T$
- Symbolic identifiers (node IDs, permit types) form a finite enumeration
- Witness objects have bounded description length in the chosen encoding

This ensures $|\mathcal{K}(T)| < \infty$ and hence termination in $\leq |\mathcal{K}(T)|$ steps. The bound $B_T$ is type-dependent but finite for each fixed $T$.

*Step 6 (Order Independence via Confluence).* The fixed point is **order-independent** (confluence):

*Claim:* For any two orderings $\sigma, \tau$ of rule applications, $\mathrm{Cl}_\sigma(\Gamma) = \mathrm{Cl}_\tau(\Gamma)$.

*Proof of Claim:* Both orderings compute the same least fixed point by Knaster-Tarski. The least fixed point is unique and characterized universally, independent of the iteration strategy used to reach it. $\checkmark$

*Step 7 (Certificate Production).* Under bounded description length, the algorithm terminates in $\leq |\mathcal{K}(T)|$ steps.
:::

:::{prf:remark} Budgeted Closure (extension)
:label: rem-closure-budgeted

If one uses the **depth-budget** regime of {prf:ref}`def-cert-finite` (case 2), the closure computation is truncated
after $D_{\max}$ iterations and yields a partial-closure certificate

$$
K_{\mathrm{Promo}}^{\mathrm{inc}} := (\text{``promotion depth exceeded''}, D_{\max}, \Gamma_{D_{\max}}, \text{trace}).
$$
This budgeted behavior is an engineering fallback for potentially infinite certificate languages. Formal guarantees for
its completeness/optimality require additional theorems and are outside Volume 2; hence the main termination claim above
is stated only for the bounded-description regime.

:::

:::{prf:remark} NO-Inconclusive Certificates ($K^{\mathrm{inc}}$)
:label: rem-inconclusive-general

The framework produces explicit **NO-inconclusive certificates** ($K^{\mathrm{inc}}$) when classification or verification is infeasible with current methods—these are NO verdicts that do *not* constitute semantic refutations:

- **Profile Trichotomy Case 3**: $K_{\mathrm{prof}}^{\mathrm{inc}}$ with classification obstruction witness
- **Surgery Admissibility Case 3**: $K_{\mathrm{Surg}}^{\mathrm{inc}}$ with inadmissibility reason
- **Promotion Closure**: $K_{\mathrm{Promo}}^{\mathrm{inc}}$ recording non-termination under budget
- **Lock (E1--E13 fail)**: $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{br\text{-}inc}}$ with tactic exhaustion trace

The certificate structure ({prf:ref}`def-typed-no-certificates`) ensures these are first-class outputs rather than silent failures. When $K^{\mathrm{inc}}$ is produced, the Sieve routes to reconstruction ({prf:ref}`mt-lock-reconstruction`) rather than fatal error, since inconclusiveness does not imply existence of a counterexample.

:::

:::{div} feynman-prose
The NO-inconclusive certificate is one of the most important innovations in this framework. It represents epistemic humility formalized.

When you cannot prove something, there are two very different situations:
1. You found a counterexample (you *know* it is false)
2. You ran out of techniques (you *do not know* either way)

Traditional systems often conflate these. The sieve keeps them separate. Case 1 is a genuine failure---the system has a defect that cannot be repaired. Case 2 is an opportunity for improvement---maybe we need better templates, or a different proof strategy, or additional assumptions.

This distinction is not academic. It determines whether the sieve terminates at FATAL ERROR (structural inconsistency confirmed, no hope) or routes to reconstruction (try harder, expand capabilities, ask for help).
:::

:::{prf:definition} Obligation ledger
:label: def-obligation-ledger

Given a certificate context $\Gamma$, define the **obligation ledger**:

$$
\mathsf{Obl}(\Gamma) := \{ (\mathsf{id}, \mathsf{obligation}, \mathsf{missing}, \mathsf{code}) : K_P^{\mathrm{inc}} \in \Gamma \}

$$

Each entry corresponds to a NO-inconclusive certificate ({prf:ref}`def-typed-no-certificates`) with payload $K_P^{\mathrm{inc}} = (\mathsf{obligation}, \mathsf{missing}, \mathsf{code}, \mathsf{trace})$.

Each entry records an undecided predicate—one where the verifier could not produce either $K_P^+$ or $K_P^{\mathrm{wit}}$.

:::

:::{prf:definition} Goal dependency cone
:label: def-goal-cone

Fix a goal certificate type $K_{\mathrm{Goal}}$ (e.g., $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$ for the Lock).
The **goal dependency cone** $\Downarrow(K_{\mathrm{Goal}})$ is the set of certificate types that may be referenced by the verifier or promotion rules that produce $K_{\mathrm{Goal}}$.

Formally, $\Downarrow(K_{\mathrm{Goal}})$ is the least set closed under:
1. $K_{\mathrm{Goal}} \in \Downarrow(K_{\mathrm{Goal}})$
2. If a verifier or upgrade rule has premise certificate types $\{K_1, \ldots, K_n\}$ and conclusion type in $\Downarrow(K_{\mathrm{Goal}})$, then all premise types are in $\Downarrow(K_{\mathrm{Goal}})$
3. If a certificate type is required by a transport lemma used by a verifier in $\Downarrow(K_{\mathrm{Goal}})$, it is also in $\Downarrow(K_{\mathrm{Goal}})$

**Purpose:** The goal cone determines which `inc` certificates are relevant to a given proof goal. Obligations outside the cone do not affect proof completion for that goal.

:::

:::{prf:definition} Proof completion criterion
:label: def-proof-complete

A sieve run with final context $\Gamma_{\mathrm{final}}$ **proves the goal** $K_{\mathrm{Goal}}$ if:
1. $\Gamma_{\mathrm{final}}$ contains the designated goal certificate (e.g., $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$), and
2. $\mathsf{Obl}(\mathrm{Cl}(\Gamma_{\mathrm{final}}))$ contains **no entries whose certificate type lies in the goal dependency cone** $\Downarrow(K_{\mathrm{Goal}})$

Equivalently, all NO-inconclusive obligations relevant to the goal have been discharged.

**Consequence:** An `inc` certificate whose type lies outside $\Downarrow(K_{\mathrm{Goal}})$ does not affect proof completion for that goal.

:::

:::{div} feynman-prose
The proof completion criterion is subtle but crucial. It is not enough to reach a "VICTORY" node---we need to make sure there are no lingering obligations that could undermine the proof.

The goal dependency cone tells us which obligations matter. If you are trying to prove global regularity, you do not care about an inconclusive certificate for some auxiliary calculation that is not in the dependency chain. But if a certificate in the cone is inconclusive, your proof is incomplete.

Think of it like a legal chain of custody. Every link in the chain must be solid. A broken link outside the chain does not matter; a broken link inside the chain invalidates everything downstream.
:::
