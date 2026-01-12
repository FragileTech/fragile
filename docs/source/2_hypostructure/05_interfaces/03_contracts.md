# Barrier and Surgery Contracts

(sec-barrier-atlas)=
## Certificate-Driven Barrier Atlas

:::{div} feynman-prose

Now we come to the part where the rubber meets the road. We have been talking about barriers and surgeries as if they were abstract mathematical objects, but here is the practical question: how do you actually *write down* what a barrier or surgery does?

The answer is contracts. Think of a contract like a legal document between two parties. The barrier says: "If you give me these certificates (proof that certain conditions hold), then I promise to either block you (give you a 'blocked' certificate so you can try something else) or let you through (give you a 'breached' certificate that authorizes a surgery)."

Why do we need such formality? Because without it, you get spaghetti. Barrier A calls surgery B which triggers barrier C which... and soon nobody knows what is going on. The contract format forces everything to be explicit: what you need, what you get, and what happens next.

The key insight is that barriers are *certificate transformers*. They take proofs in, and they produce proofs out. That is what makes the whole system compositional - you can chain barriers together and still know exactly what is being guaranteed at each step.

:::

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

:::{div} feynman-prose

Let me make sure you understand what each of these pieces does.

The **Trigger** is just "which gate said NO?" If gate number 7 rejects your system, that is the trigger for barrier 7.

The **Pre** (prerequisites) is where it gets interesting. These are the certificates you already have - the proofs you collected from earlier gates. The barrier can use these to decide what to do. But here is a subtle point that trips people up: you cannot assume the very thing that just failed! If gate 7 failed, you cannot list "gate 7 passed" as a prerequisite. That would be circular nonsense.

The **Blocked certificate** is what you get if the barrier says "no surgery for you." It is a proof that you should try something else - maybe go back to an earlier node, or take a different branch.

The **Breached certificate** is the authorization to perform surgery. It says two things: (1) which failure mode we are in, and (2) that surgery is actually admissible for this mode. Think of it as a signed permission slip.

The **Scope** tells you which types of objects this barrier applies to. Not every barrier applies to every system.

:::

:::{prf:theorem} Non-circularity
:label: thm-barrier-noncircular

For any barrier $B$ triggered by gate $i$ with predicate $P_i$:

$$P_i \notin \mathrm{Pre}(B)$$

A barrier invoked because $P_i$ failed cannot assume $P_i$ as a prerequisite.

:::

:::{prf:proof}

Suppose toward contradiction that $P_i \in \mathrm{Pre}(B)$.

**Step 1.** Barrier $B$ is invoked precisely when gate $i$ returns NO, meaning $\neg P_i$ holds.

**Step 2.** If $P_i \in \mathrm{Pre}(B)$, then evaluating $B$ requires a certificate witnessing $P_i$.

**Step 3.** But we have $\neg P_i$, so no such certificate exists.

**Step 4.** Hence the barrier cannot be evaluated, contradicting its role as the handler for gate $i$ failure.

Therefore $P_i \notin \mathrm{Pre}(B)$.

$\square$

:::

:::{div} feynman-prose

This non-circularity condition is not just a technicality - it is essential for the whole architecture. If a barrier could assume the thing that just failed, you would have an infinite regress: "I need X to handle the failure of X, but X failed, so I need X to..."

The literature references point to work on stratified logic programs and well-founded semantics. The core idea there is the same: you cannot define something in terms of itself in a circular way. You need a hierarchy, a stratification, where each level only depends on things defined at lower levels.

:::

:::{note}
:class: feynman-added

**Literature:** Stratification and well-foundedness {cite}`VanGelder91`; non-circular definitions {cite}`AptBolPedreschi94`.
:::

---

(sec-surgery-contracts)=
## Surgery Contracts

:::{div} feynman-prose

Now let us talk about surgeries. A surgery is a drastic operation - you are going to modify the state space itself, not just evolve within it. Think of it like this: if normal dynamics is driving on a road, surgery is rebuilding the road.

Why would you ever need such a thing? Because sometimes the system gets into a state that the normal dynamics cannot handle. A fluid develops a singularity. A proof search hits an infinite loop. A control system saturates. In these cases, you need to do something radical to get back on track.

But here is the danger: if you allow arbitrary surgeries, you lose all guarantees. Someone could "fix" a blow-up by just setting everything to zero, destroying all the information the system had accumulated. We need surgeries that are *controlled*, that preserve essential structure while fixing the pathology.

That is what the surgery contract does. It says exactly: what state are you in, what are you going to do, and what state will you be in afterward. No hand-waving, no "trust me, this works." Every transformation must be specified and justified.

:::

:::{prf:definition} Surgery contract format
:label: def-surgery-format

Each surgery entry follows the **Surgery Specification Schema** ({prf:ref}`def-surgery-schema`):

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

:::{div} feynman-prose

Let me walk you through each piece of this contract, because every single one matters.

The **Surgery ID** and **Target Mode** tell you *when* this surgery applies. You do not just perform surgeries randomly - each surgery is the specific treatment for a specific disease. Mode C.E (concentration explosion) gets a different surgery than Mode V.D (vortex degeneration).

The **Interface Dependencies** are what the surgery needs to see. The "primary" interface gives you the actual pathological object - the singular point, the diverging field, the stuck proof term. The "secondary" interface gives you the tools to fix it - a library of canonical replacements, bounds on what is allowed.

The **Admissibility Signature** is the gatekeeper. Just because a surgery *exists* does not mean you can *use* it. You need the right input certificate (proof that you are in the right failure mode) and the admissibility conditions must hold (the surgery is actually safe to perform here).

The **Transformation Law** is the actual surgery. How does the state change? What happens to the energy? Do you end up in a different topological sector? This must be completely explicit - no magic, no hidden side effects.

Finally, the **Postcondition** tells you where you end up. After surgery, you need a *re-entry certificate* that lets you get back into the normal flow of the Sieve. And crucially, you need a *progress guarantee* - proof that you are actually making progress, not just thrashing around.

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

:::{div} feynman-prose

Here is the central problem with termination: how do you know you will not perform infinitely many surgeries?

There are two strategies, and you must pick one.

**Type A** says: count the surgeries. If you can prove there is some maximum number $N$ that depends on the initial conditions, you are done. This is the brute-force approach, but it works. The catch is the *discrete progress constraint*. If your energy can drop by arbitrarily small amounts, you could have infinitely many tiny surgeries that sum to a finite total energy loss. The constraint $\Delta\Phi \geq \epsilon_T$ prevents this by forcing each surgery to make a "real" step.

**Type B** is more elegant: use a well-founded ordering. Instead of counting, you assign a complexity measure to each state, and you prove that every surgery strictly decreases this measure. Since a well-founded ordering has no infinite descending chains (that is what "well-founded" means), you automatically get termination.

The Zeno remark is worth understanding deeply. Zeno's paradox asks: how can Achilles catch the tortoise if he must first reach where the tortoise was, then where it moved to, and so on forever? The resolution is that infinitely many steps can happen in finite time if the steps get small enough fast enough. But for surgeries, we do not want that! Each surgery is a discrete computational step, and we cannot perform infinitely many of them. The $\epsilon_T$ constraint is our anti-Zeno shield.

:::
