(sec-barrier-node-specs)=
## Barrier Node Specifications (Orange Nodes)

:::{div} feynman-prose
Now we come to the heart of what makes the Sieve actually *work*. You see, the gates we discussed earlier—they're like bouncers at a nightclub, checking if you've got the right credentials. But barriers? Barriers are different. They're more like... well, imagine you're trying to push a ball up a hill. The barrier is the hill itself. Either you don't have enough energy to get over it (blocked), or you do and something interesting happens (breached).

Here's the key insight: every barrier is a *physical* principle dressed up in mathematical clothing. The saturation barrier asks "is your energy drift bounded?" The causal barrier asks "would reaching this singularity require infinite computational depth?" These aren't arbitrary checks—they're fundamental laws of nature manifesting as routing decisions.

What makes the barrier specifications below so powerful is their *interface dependencies*. Each barrier draws from specific mathematical structures—energy functionals, recursion depths, concentration measures—and transforms them into binary (or sometimes ternary) decisions. The blocked outcome always implies something useful: either you proceed to the next check, or you've proven something impossible can't happen. The breached outcome activates a *surgery mode*—a systematic repair procedure that lets you continue after fixing the violation.

Think of it this way: if gates are customs officers checking your passport, barriers are the laws of physics checking whether your journey is even possible.
:::

Each barrier is specified by:
- **Trigger**: Which gate's NO invokes it
- **Pre-certificates**: Required context (non-circular)
- **Outcome alphabet**: Blocked/Breached (or special)
- **Blocked certificate**: Must imply Pre(next node)
- **Breached certificate**: Must imply mode activation + surgery admissibility
- **Next nodes**: Routing for each outcome

---

(sec-barrier-sat)=
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

$$
E[\Phi] \leq E_{\text{sat}} \lor \operatorname{Drift} \leq C
$$

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

:::{div} feynman-prose
Let me tell you what this barrier is *really* about. Imagine you're watching a pot of water on the stove. The energy keeps going in—heat from the burner—but does the temperature blow up to infinity? Of course not! There's a *saturation ceiling*: the water boils, and all that extra energy goes into phase change rather than temperature increase.

The BarrierSat does exactly this check for dynamical systems. It asks: "Even if energy is flowing in, is there some mechanism that prevents blow-up?" The answer could be physical dissipation (friction eating up energy), a saturation nonlinearity (like our boiling water), or a Foster-Lyapunov drift condition (a mathematical guarantee that things don't escape to infinity).

The beautiful thing is that this is the *first* barrier you hit—the entry point. No prior certificates required. Either energy is under control and you proceed to check for Zeno behavior, or energy is blowing up and you need surgery to fix it before continuing.
:::

---

(sec-barrier-causal)=
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

$$
D(T_*) = \int_0^{T_*} \frac{c}{\lambda(t)} \,dt = \infty
$$

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

:::{div} feynman-prose
Here's something that should make you sit up. This barrier is essentially asking: "Would it take *infinite* computation to describe this singularity?"

Think about what that means. Penrose's cosmic censorship conjecture says that naked singularities—singularities visible to distant observers—shouldn't form in nature. But why? One answer: because describing them would require infinite information. The integral $\int_0^{T_*} c/\lambda(t)\,dt$ measures computational depth—how many "layers" of calculation you'd need to specify what happens at time $T_*$.

If this integral diverges, the singularity is *causally censored*. Not by some arbitrary rule, but by the fundamental limits of computation. You literally cannot describe what happens there with any finite program. That's not a bug—that's the universe protecting itself from inconsistency.

When the integral is finite, though, watch out. The singularity is computationally accessible, which means events are accumulating faster than you can process them. Time for surgery.
:::

---

(sec-barrier-scat)=
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

$$
\mathcal{M}[\Phi] < \infty
$$

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

:::{div} feynman-prose
Now this is where things get physically interesting. The scattering barrier has a *special alphabet*—not just Blocked/Breached, but Benign/Pathological. Why?

Here's the picture: imagine dropping a pebble in a pond. The ripples spread out, getting weaker and weaker as they disperse. That's scattering—the energy disperses to infinity rather than concentrating at a point. The Morawetz interaction functional $\mathcal{M}[\Phi]$ measures exactly this: if it's finite, your solution scatters like those pond ripples.

But what if the energy *doesn't* disperse? What if it concentrates, forming something like a soliton—a self-reinforcing wave packet that holds its shape? That's the pathological case. Not necessarily catastrophic (solitons can be perfectly well-behaved), but definitely requiring special attention.

This barrier is one of the *success exits* from the Sieve. If you get Benign, congratulations—you've proven global existence via dispersion. Your system won't blow up because the energy spreads out to infinity. That's a real theorem, not just a diagnostic.
:::

---

(sec-barrier-type2)=
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

$$
\int \tilde{\mathfrak{D}}(S_t V) \,dt = \infty
$$

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

:::{div} feynman-prose
Type II blow-up is one of the most subtle phenomena in nonlinear dynamics, and understanding why this barrier exists requires a bit of history.

When a solution blows up—goes to infinity in finite time—there are basically two ways it can happen. Type I blow-up is "self-similar": the solution looks the same at every scale, just faster and smaller. Think of a whirlpool tightening uniformly. Type II blow-up is stranger—the blow-up rate doesn't match the scaling symmetry of the equation. It's like a whirlpool that speeds up faster than geometry alone would predict.

What this barrier checks is the *renormalization cost*. If you try to zoom in on a Type II blow-up profile and rescale it, you accumulate some "defect" at each scale. The integral $\int \tilde{\mathfrak{D}}(S_t V)\,dt$ measures the total cost of this rescaling process. If it diverges, Type II blow-up is energetically impossible—you simply cannot pay the infinite renormalization bill.

The non-circularity note is important: we don't *assume* subcriticality to run this check. Subcriticality is a *bonus*—if you already know the scaling exponent is favorable, Type I is excluded for free. But this barrier works even without that assumption.
:::

---

(sec-barrier-vac)=
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

$$
\Delta V > k_B T
$$

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

:::{div} feynman-prose
And here's where particle physics shows up in our dynamical systems framework!

The vacuum barrier is asking the same question that keeps particle physicists up at night: "Is our vacuum stable?" In quantum field theory, the vacuum isn't nothing—it's the lowest energy state of all the fields. But what if there's an even lower state somewhere else in configuration space, separated by an energy barrier?

The condition $\Delta V > k_B T$ is the mass gap principle. If the potential barrier between your current vacuum and any other state exceeds the thermal energy scale, you're safe—random fluctuations can't kick you over the hill. This is exactly like asking whether a ball in a valley can spontaneously jump into the next valley due to thermal jiggling.

What's beautiful is that this connects field theory to mundane thermodynamics. Whether you're worried about the Higgs vacuum decaying or a bistable chemical reaction switching states, the mathematics is the same: compare barrier height to temperature. If the barrier wins, your phase is stable.
:::

---

(sec-barrier-cap)=
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

$$
\mathrm{Cap}_H(S) = 0
$$

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

:::{div} feynman-prose
This barrier is pure geometric measure theory, and it's one of my favorites because the intuition is so clean.

Ask yourself: how "big" is the set where things go wrong? If singularities form, how much space do they occupy? The Hausdorff capacity $\mathrm{Cap}_H(S)$ gives a precise answer, and zero capacity means the singular set is negligible—it's so small that it can't carry any "weight" in the dynamics.

Here's a good mental picture: think of capacity as "how much current can flow through this set?" Zero capacity means zero conductivity—the singular set is electrically invisible. Or think of it probabilistically: a Brownian motion has zero probability of ever hitting a zero-capacity set. The singularities might technically exist, but they're so thin that any reasonable trajectory never encounters them.

This is the mathematical machinery behind "removable singularities"—singularities that look scary but don't actually affect the solution. If $\mathrm{Cap}_H(S) = 0$, you can ignore the singular set and extend your solution smoothly across it.
:::

---

(sec-barrier-gap)=
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

$$
\inf \sigma(L) > 0
$$

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

:::{div} feynman-prose
The spectral gap barrier is asking one of the most fundamental questions in dynamics: "Is there a definite 'downhill' direction from this critical point?"

Think of a ball on a landscape. At a local minimum, every direction curves upward—there's positive curvature in all directions. The smallest eigenvalue of the Hessian (the matrix of second derivatives) tells you the curvature of the gentlest direction. If this smallest eigenvalue is positive, you're at a genuine minimum. If it's zero or negative, you're at a saddle point or worse.

The spectral gap $\inf \sigma(L) > 0$ is exactly this: the linearized operator $L$ has all positive eigenvalues. What does this buy you? *Exponential convergence*. The system doesn't just drift toward equilibrium—it rushes there, with deviations decaying like $e^{-\lambda_1 t}$.

This barrier has a special "Stagnation" outcome instead of "Breached." If there's no spectral gap, you don't get surgery—you get rerouted to a bifurcation check. Why? Because zero eigenvalues often signal that the critical point is about to split into multiple equilibria. That's not pathological; it's just delicate and requires different analysis.
:::

:::{prf:lemma} Gap implies Lojasiewicz-Simon
:label: lem-gap-to-ls

Under the Gradient Condition ($\mathrm{GC}_\nabla$) plus analyticity of $\Phi$ near critical points:

$$
\lambda_1 > 0 \Rightarrow \operatorname{LS}(\theta = \tfrac{1}{2}, C_{\text{LS}} = \sqrt{\lambda_1})
$$

where $\lambda_1$ is the spectral gap. This is the **canonical promotion** from gap certificate to stiffness certificate, bridging the diagram's "Hessian positive?" intuition with the formal LS inequality predicate.

:::

:::{div} feynman-prose
This lemma is the mathematical bridge between linear and nonlinear analysis. The Lojasiewicz-Simon inequality says that near a critical point, the gradient of the energy functional is controlled by a power of the energy itself. With a spectral gap, you get the optimal power $\theta = 1/2$, and the constant is explicitly $\sqrt{\lambda_1}$.

Why does this matter? Because the LS inequality is what lets you prove *finite-time convergence* and *uniqueness of limits* for gradient flows. Without it, solutions could spiral around forever, never quite reaching equilibrium. With it, they must converge—and the rate is completely determined by that spectral gap.
:::

---

(sec-barrier-action)=
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

$$
E[\Phi] < S_{\min} + \Delta
$$

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

:::{div} feynman-prose
Now we come to topology—the mathematics of what can and cannot be continuously deformed into something else.

The action barrier asks: "Does this system have enough energy to change its topological type?" Think of a rubber band: stretching and squeezing it is easy, but tearing it or fusing two loops together costs real energy. In field theory, this shows up as *action gaps*—minimum energies required to create kinks, monopoles, instantons, or other topological defects.

The beautiful thing about $E[\Phi] < S_{\min} + \Delta$ is that it's an *energetic lock* on topology. If your current state doesn't have enough energy to climb over the topological barrier, you're stuck in your current topological sector. No tunneling, no kink formation, no surprises.

This connects to some deep physics: the reason certain quantum numbers are conserved (like baryon number) is that changing them would require going over an action barrier that's energetically forbidden. The topology is *protected* by energy.
:::

---

(sec-barrier-omin)=
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

$$
S \in \mathcal{O}\text{-min}
$$

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

:::{div} feynman-prose
O-minimal structures are one of the great gifts that model theory has given to analysis, and this barrier is where that gift pays off.

Here's the problem: not all sets are "tame." Some are genuinely pathological—fractals, Cantor sets, space-filling curves. These sets can have arbitrarily complicated boundary behavior, infinitely many connected components, and all sorts of nastiness that makes analysis impossible.

O-minimal structures are a precise definition of "tameness." A set is o-minimally definable if it belongs to a special collection of sets that are guaranteed to be well-behaved: finite stratification, no wild oscillations, controlled asymptotic behavior. If your singular set $S$ is o-minimally definable, you can trust that it won't surprise you with pathological fractal structure.

The condition $S \in \mathcal{O}\text{-min}$ is asking: "Is your topology *boring* in a good way?" Boring topology is predictable topology. Wild topology requires surgery.
:::

---

(sec-barrier-mix)=
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

$$
\tau_{\text{mix}} < \infty
$$

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

:::{div} feynman-prose
Ergodic theory asks one of the most important questions in statistical physics: "If I wait long enough, will my system explore all the states it's allowed to visit?"

The mixing time $\tau_{\text{mix}}$ quantifies this. It's the time it takes for your system to "forget" where it started and settle into its equilibrium distribution. Finite mixing time means the system is *ergodic*—time averages equal ensemble averages, and you can trust statistical mechanics.

But what if the mixing time is *infinite*? Then you have trapping. The system can get stuck in a local region of phase space and never escape. This violates the fundamental assumption of statistical mechanics—that all accessible states are eventually visited.

Here's the physical picture: imagine a ball rolling in a landscape with deep wells. If the wells are too deep relative to the thermal energy, the ball can get trapped forever in one well, never making it over the hills to explore the rest of the landscape. The mixing barrier checks whether this pathological trapping is happening.
:::

---

(sec-barrier-epi)=
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

$$
\sup_{\epsilon > 0} K_\epsilon(x) \leq S_{\text{BH}}
$$

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

:::{div} feynman-prose
And now we arrive at something genuinely mind-bending: the epistemic horizon, where information theory meets the physics of spacetime.

Here's the question: how much information can you encode in a given region of space? The holographic bound says the answer is *finite*—proportional to the surface area, not the volume. This is the Bekenstein-Hawking entropy, and it puts a hard limit on how complex any physical state can be.

The barrier predicate $\sup_{\epsilon > 0} K_\epsilon(x) \leq S_{\text{BH}}$ is asking: "Even though we can't compute the exact Kolmogorov complexity, do all *approximations* to the complexity stay within the holographic bound?"

This is subtle. Exact Kolmogorov complexity is famously uncomputable—you can never know for certain the length of the shortest program that generates a given string. But we can compute *approximations* that converge from above. If these approximations stay bounded, we're safe. If they blow up, we've hit an epistemic horizon—the state contains more information than spacetime can hold, which is physically impossible.
:::

---

(sec-barrier-freq)=
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

$$
\int \omega^2 S(\omega) \,d\omega < \infty
$$

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

:::{div} feynman-prose
The frequency barrier is about a very specific kind of pathology: oscillation blow-up.

Think about a guitar string. The fundamental frequency carries most of the energy, but there are harmonics—higher frequencies that add richness to the sound. The integral $\int \omega^2 S(\omega)\,d\omega$ is the total oscillation energy, weighted by frequency squared. If this diverges, energy is cascading to higher and higher frequencies without bound.

Why is this bad? Because high frequencies mean small scales, and small scales mean you need finer and finer resolution to track what's happening. Infinite oscillation energy means the solution is developing structure at arbitrarily small scales—a frequency cascade that no finite computer can track.

The De Giorgi-Nash-Moser theory tells us that elliptic and parabolic PDEs have built-in frequency barriers. Solutions can't oscillate too wildly—they inherit regularity from the equation itself. This barrier checks whether your system has similar protection.
:::

---

(sec-barrier-boundary)=
### Boundary Barriers (BarrierBode, BarrierInput, BarrierVariety)

:::{div} feynman-prose
We now turn to the boundary barriers—checks that handle systems with inputs and outputs, where information and resources flow across the system boundary. These are the barriers that matter for control theory, cybernetics, and any system that isn't closed.

The three barriers here form a logical sequence: first check sensitivity (Bode), then resources (Input), then control capacity (Variety). Each one captures a different way an open system can fail.
:::

:::{prf:definition} Barrier Specification: Bode Sensitivity
:label: def-barrier-bode

**Barrier ID:** `BarrierBode`

**Interface Dependencies:**
- **Primary:** $\mathrm{Bound}_B$ (provides sensitivity function $S(s)$ and Bode integral $B_{\text{Bode}}$)
- **Secondary:** $\mathrm{LS}_\sigma$ (provides stability landscape for waterbed constraints)

**Sieve Signature:**
- **Weakest Precondition:** $\{K_{\mathrm{Bound}_\partial}^+\}$ (open system confirmed)
- **Barrier Predicate (Blocked Condition):**

$$
\int_0^\infty \ln \lVert S(i\omega) \rVert \,d\omega > -\infty
$$

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

:::{div} feynman-prose
The Bode barrier encodes one of the deepest truths of feedback control: you cannot win everywhere.

Here's the waterbed effect: imagine a waterbed. If you push down in one place, water has to go somewhere—it bulges up elsewhere. The Bode integral constraint says the same thing about sensitivity: the integral of log-sensitivity over all frequencies is *conserved*. If you reduce sensitivity at one frequency, you must increase it somewhere else.

This is not a design limitation you can engineer around—it's a *theorem*. Any linear feedback system must obey it. The barrier checks whether your system respects this fundamental tradeoff or is trying to violate it (which means your model is wrong or your controller is about to do something unstable).

The condition $\int_0^\infty \ln \lVert S(i\omega) \rVert\,d\omega > -\infty$ ensures the waterbed is bounded—you haven't tried to push sensitivity to zero everywhere, which is impossible.
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

$$
r_{\text{reserve}} > 0
$$

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

:::{div} feynman-prose
The input barrier is about resources—the fuel that keeps an open system running.

Every living system, every robot, every economy needs inputs: energy, materials, information. The reserve $r_{\text{reserve}}$ is the buffer—how much can you withstand a temporary shortage before things start to fail?

Positive reserve means resilience. You can absorb a supply chain disruption, a temporary power outage, a delay in information. Zero or negative reserve means you're living hand-to-mouth, and any fluctuation in input can cascade into system failure.

This is Input-to-State Stability (ISS) from control theory: the idea that bounded inputs should lead to bounded states. If your reserve is positive, you have a buffer against input disturbances. If not, you need surgery to either increase reserves or reduce consumption.
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

$$
H(u) \geq H(d)
$$

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

:::{div} feynman-prose
And here is Ashby's Law—one of the most profound insights of cybernetics, and a fitting end to our tour of barriers.

The Law of Requisite Variety says: "Only variety can destroy variety." A thermostat can only regulate temperature because it has (at least) two states: heat on, heat off. A chess player can only counter their opponent's moves because they have at least as many strategic options available.

The condition $H(u) \geq H(d)$ is entropy comparison: the control entropy must match or exceed the disturbance entropy. If you face a disturbance with 10 bits of uncertainty, you need at least 10 bits of control authority to counter it. Anything less, and some disturbances will get through uncontrolled.

This barrier is checking whether your controller is *fundamentally adequate* for the task. Not whether it's optimal, not whether it's well-tuned, but whether it has enough variety to even play the game. If not, no amount of clever algorithm design can save you—you need more control authority.
:::

---
