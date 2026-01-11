---
title: "Gate, Barrier, and Surgery Theorems"
---

(sec-gate-evaluator-theorems)=
## Gate Evaluator Theorems

*These theorems define exact mathematical predicates for YES/NO checks at blue nodes.*

:::{div} feynman-prose
Now, what are these gate evaluator theorems really about? Think of the Sieve as a security checkpoint at an airport. Each blue node is a guard asking one very specific question: "Does this trajectory have property X?" The answer is YES or NO, nothing in between. These theorems tell you exactly what question each guard asks and why that question matters.

The beautiful thing is that these are not arbitrary checks. Each one corresponds to a known mathematical mechanism by which systems can fail catastrophically. If you can prove the predicate is satisfied, you have mathematically certified that this particular failure mode cannot occur. That is the whole game: turning vague safety concerns into precise mathematical certificates.
:::

---

### Type II Exclusion (ScaleCheck Predicate)

:::{div} feynman-prose
Here is the first guard at our checkpoint. This one asks: "Are your scaling exponents in the right regime?" Let me tell you what this means physically.

When a system develops a singularity, it often does so through self-similar blow-up. The solution starts concentrating energy at smaller and smaller scales, like water going down a drain that gets infinitely narrow. Now, there are two numbers that control whether this can actually happen: the exponent $\alpha$ that tells you how dissipation scales, and the exponent $\beta$ that tells you how time scales when you zoom in.

Here is the key insight: if $\alpha > \beta$, dissipation wins. As you zoom into smaller scales, the system loses energy faster than it can concentrate it. The singularity literally cannot afford the energy bill. That is what "subcritical" means, it means the accounting does not work out for the singularity.

The theorem proves this rigorously: any self-similar blow-up sequence that tried to form would require infinite dissipation, which contradicts our finite energy budget. So the singularity is excluded not by fiat, but by thermodynamics.
:::

:::{prf:theorem} [LOCK-Tactic-Scale] Type II Exclusion
:label: mt-lock-tactic-scale

**Sieve Target:** Node 4 (ScaleCheck) — predicate $\alpha > \beta$ excludes supercritical blow-up

**Statement:** Let $\mathcal{S}$ be a hypostructure satisfying interface permits $D_E$ and $\mathrm{SC}_\lambda$ with scaling exponents $(\alpha, \beta)$ satisfying $\alpha > \beta$ (strict subcriticality). Let $x \in X$ with $\Phi(x) < \infty$ and $\mathcal{C}_*(x) < \infty$ (finite total cost). Then **no supercritical self-similar blow-up** can occur at $T_*(x)$.

More precisely: if a supercritical sequence produces a nontrivial ancient trajectory $v_\infty$, then:

$$\int_{-\infty}^0 \mathfrak{D}(v_\infty(s)) \, ds = \infty$$

**Certificate Produced:** $K_4^+$ with payload $(\alpha, \beta, \alpha > \beta)$ or $K_{\text{TypeII}}^{\text{blk}}$

**Literature:** {cite}`MerleZaag98`; {cite}`KenigMerle06`; {cite}`Struwe88`; {cite}`Tao06`
:::

:::{prf:proof}
:label: proof-mt-lock-tactic-scale

*Step 1 (Change of Variables).* For rescaled time $s = \lambda_n^\beta(t - t_n)$ and rescaled state $v_n(s) = \mathcal{S}_{\lambda_n} \cdot u(t)$:

$$\int_{t_n}^{T_*(x)} \mathfrak{D}(u(t)) \, dt$$

*Step 2 (Dissipation Scaling).* By interface permit $\mathrm{SC}_\lambda$ with exponent $\alpha$:

$$\mathfrak{D}(u(t)) = \mathfrak{D}(\mathcal{S}_{\lambda_n}^{-1} \cdot v_n(s)) \sim \lambda_n^{-\alpha} \mathfrak{D}(v_n(s))$$

*Step 3 (Cost Transformation).* Substituting:

$$\int_{t_n}^{T_*(x)} \mathfrak{D}(u(t)) \, dt = \lambda_n^{-(\alpha + \beta)} \int_0^{S_n} \mathfrak{D}(v_n(s)) \, ds$$

*Step 4 (Supercritical Regime).* For nontrivial $v_\infty$ with $S_n \to \infty$:

$$\int_0^{S_n} \mathfrak{D}(v_n(s)) \, ds \gtrsim C_0 \lambda_n^\beta(T_*(x) - t_n)$$

*Step 5 (Contradiction).* If $\alpha > \beta$, summing over dyadic scales requires $\int_{-\infty}^0 \mathfrak{D}(v_\infty) ds = \infty$ for consistency with $\mathcal{C}_*(x) < \infty$.
:::

---

### Spectral Generator (StiffnessCheck Predicate)

:::{div} feynman-prose
This guard asks about the shape of the energy landscape near your destination. Imagine you are rolling a marble into a bowl. If the bowl has a nice curved bottom, the marble settles smoothly into the center. But if the bottom is flat, or worse, has ridges and valleys, the marble can wander around forever without really settling.

The spectral generator theorem says that if your energy function has positive curvature at the minimum, meaning the Hessian has all positive eigenvalues, then you get a very powerful inequality. The gradient cannot be small unless you are actually close to the minimum. This is the Lojasiewicz-Simon inequality, and it is what guarantees finite-time convergence rather than asymptotic approach.

The key number is the exponent $\theta$. For analytic functions, $\theta = 1/2$, which gives you exponential convergence. For merely smooth functions, $\theta$ might be closer to 1, which gives slower but still guaranteed convergence. The theorem tells you that having a spectral gap in the Hessian is the mathematical certificate you need to ensure the system actually reaches equilibrium.
:::

:::{prf:theorem} [LOCK-SpectralGen] Spectral Generator
:label: mt-lock-spectral-gen

**Sieve Target:** Node 7 (StiffnessCheck) — spectral gap $\Rightarrow$ Łojasiewicz-Simon inequality

**Statement:** Let $\mathcal{S}$ be a hypostructure satisfying interface permits $D_E$, $\mathrm{LS}_\sigma$, and $\mathrm{GC}_\nabla$. The local behavior near the safe manifold $M$ determines the sharp functional inequality governing convergence:

$$\nabla^2 \Phi|_M \succ 0 \quad \Longrightarrow \quad \|\nabla \Phi(x)\| \geq c \cdot |\Phi(x) - \Phi_{\min}|^\theta$$

for some $\theta \in [1/2, 1)$ and $c > 0$.

**Required Interface Permits:** $D_E$ (Dissipation), $\mathrm{LS}_\sigma$ (Stiffness), $\mathrm{GC}_\nabla$ (Gradient Consistency)

**Prevented Failure Modes:** S.D (Stiffness Breakdown), C.E (Energy Escape)

**Certificate Produced:** $K_7^+$ with payload $(\sigma_{\min}, \theta, c)$

**Literature:** {cite}`Lojasiewicz63`; {cite}`Simon83`; {cite}`HuangTang06`; {cite}`ColdingMinicozzi15`
:::

:::{prf:proof}
:label: proof-mt-lock-spectral-gen

*Step 1 (Hessian structure).* Near a critical point $x_* \in M$, the height functional $\Phi$ admits Taylor expansion:

$$\Phi(x) = \Phi(x_*) + \frac{1}{2}\langle \nabla^2 \Phi|_{x_*} (x - x_*), (x - x_*) \rangle + O(|x - x_*|^3)$$

*Step 2 (Spectral gap from positivity).* If $\nabla^2 \Phi|_{x_*} \succ 0$ with smallest eigenvalue $\sigma_{\min} > 0$, then:

$$\Phi(x) - \Phi(x_*) \geq \frac{\sigma_{\min}}{2}|x - x_*|^2$$

*Step 3 (Gradient bound).* The gradient satisfies $\|\nabla \Phi(x)\| \geq \sigma_{\min}|x - x_*|$. Combined with Step 2:

$$\|\nabla \Phi(x)\| \geq \sigma_{\min} \sqrt{\frac{2}{\sigma_{\min}}(\Phi(x) - \Phi_{\min})} = \sqrt{2\sigma_{\min}} |\Phi(x) - \Phi_{\min}|^{1/2}$$

This gives the Łojasiewicz exponent $\theta = 1/2$ (optimal for analytic functions).

*Step 4 (Simon's extension).* For infinite-dimensional systems, Simon (1983) extended the Łojasiewicz inequality to Banach spaces with analytic structure, showing $\theta \in [1/2, 1)$ suffices for convergence.
:::

---

### Ergodic Mixing Barrier (ErgoCheck Predicate)

:::{div} feynman-prose
This is one of my favorites because it connects dynamics to probability in a deep way. The guard here is asking: "Is your system mixing?"

What does mixing mean? Think of stirring cream into coffee. At first, the cream is localized in one spot. As you stir, it spreads out. A mixing system is one where everything eventually spreads out, where no region of phase space can remain isolated forever. Correlations decay. The system forgets where it started.

Why does this prevent singularities? Because a singularity is localization. It is energy concentrating at a point. But a mixing system fights localization. Any concentration gets dispersed by the dynamics. Birkhoff's ergodic theorem tells you that time averages equal space averages, the system visits everywhere proportionally to the measure.

The theorem makes this precise: if your system is mixing and you have finite energy, then localized singular structures cannot persist. They would require a permanent concentration of measure, but mixing dilutes any such concentration exponentially fast. The mixing time $\tau_{\text{mix}}$ tells you how fast this happens.
:::

:::{prf:theorem} [LOCK-ErgodicMixing] Ergodic Mixing Barrier
:label: mt-lock-ergodic-mixing

**Sieve Target:** Node 10 (ErgoCheck) — mixing prevents localization

**Statement:** Let $(X, S_t, \mu)$ be a measure-preserving dynamical system satisfying interface permits $C_\mu$ and $D_E$. If the system is **mixing**, then:

1. Correlation functions decay: $C_f(t) := \int f(S_t x) f(x) d\mu - (\int f d\mu)^2 \to 0$ as $t \to \infty$
2. No localized invariant structures can persist
3. Mode T.D (Glassy Freeze) is prevented

**Required Interface Permits:** $C_\mu$ (Compactness), $D_E$ (Dissipation)

**Prevented Failure Modes:** T.D (Glassy Freeze), C.E (Escape)

**Certificate Produced:** $K_{10}^+$ (mixing) with payload $(\tau_{\text{mix}}, C_f(t))$

**Literature:** {cite}`Birkhoff31`; {cite}`Furstenberg81`; {cite}`Sinai70`; {cite}`Bowen75`
:::

:::{prf:proof}
:label: proof-mt-lock-ergodic-mixing

*Step 1 (Mixing definition).* The system is mixing if for all $f, g \in L^2(\mu)$:

$$\lim_{t \to \infty} \int f(S_t x) g(x) d\mu = \int f d\mu \int g d\mu$$

*Step 2 (Birkhoff ergodic theorem).* By Birkhoff (1931), for ergodic systems:

$$\frac{1}{T} \int_0^T f(S_t x) dt \to \int f d\mu \quad \text{a.e.}$$

*Step 3 (Localization obstruction).* A localized singular structure would require $\mu(B_\varepsilon(x_*)) > 0$ to persist under the flow for all time. But mixing implies:

$$\mu(S_t^{-1}(B_\varepsilon(x_*)) \cap B_\varepsilon(x_*)) \to \mu(B_\varepsilon(x_*))^2$$
For small $\varepsilon$, the measure of return diminishes, preventing persistent localization.

*Step 4 (Escape guarantee).* Combined with interface permit $D_E$, the trajectory cannot remain trapped in any finite region indefinitely. Mixing spreads mass throughout the accessible phase space.
:::

---

### Spectral Distance Isomorphism (OscillateCheck Predicate)

:::{div} feynman-prose
Now we get into some genuinely beautiful mathematics from Alain Connes. This guard is checking for oscillatory breakdown, and the tool is the Connes distance formula from noncommutative geometry.

Here is the intuition. In ordinary geometry, the distance between two points is the length of the shortest path. But Connes realized you can turn this around: instead of measuring distances with rulers, measure them with functions. How different can a Lipschitz function be at two points? The answer turns out to be exactly the geodesic distance.

The formula $d_D(x,y) = \sup\{|f(x) - f(y)| : \|[D,f]\| \leq 1\}$ looks abstract, but here is what it says: the distance between $x$ and $y$ is the maximum amount any "well-behaved" function can change between them. The condition $\|[D,f]\| \leq 1$ is the Lipschitz condition in disguise, the commutator $[D,f]$ is the noncommutative gradient.

Why does this detect oscillatory breakdown? Because when derivatives blow up, the commutator norm explodes. Oscillations at higher and higher frequencies have larger and larger gradients. This is the mathematical signature of oscillatory singularities, and the theorem shows that the Connes distance formula catches them.
:::

:::{prf:theorem} [LOCK-SpectralDist] Spectral Distance Isomorphism
:label: mt-lock-spectral-dist

**Sieve Target:** Node 12 (OscillateCheck) — commutator $\|[D,a]\|$ detects oscillatory breakdown

**Statement:** In the framework of noncommutative geometry, the Connes distance formula provides a spectral characterization of metric structure:

$$d_D(x, y) = \sup\{|f(x) - f(y)| : \|[D, f]\| \leq 1\}$$

The interface permit $\mathrm{GC}_\nabla$ (Gradient Consistency) is equivalent to the spectral distance formula when the geometry admits a Dirac-type operator $D$.

**Bridge Type:** NCG $\leftrightarrow$ Metric Spaces

**Dictionary:**
- Commutator $[D, a]$ $\leftrightarrow$ Gradient $\nabla f$
- Spectral distance $d_D$ $\leftrightarrow$ Geodesic distance
- $\|[D, a]\| \leq 1$ $\leftrightarrow$ $\|\nabla f\| \leq 1$ (Lipschitz condition)

**Certificate Produced:** $K_{12}^+$ with payload $(D, \|[D, \cdot]\|, d_D)$

**Literature:** {cite}`Connes94`; {cite}`Connes96`; {cite}`GraciaBondia01`; {cite}`Landi97`
:::

:::{prf:proof}
:label: proof-mt-lock-spectral-dist

*Step 1 (Spectral triple).* A spectral triple $(\mathcal{A}, \mathcal{H}, D)$ consists of: algebra $\mathcal{A}$ acting on Hilbert space $\mathcal{H}$, self-adjoint Dirac operator $D$ with compact resolvent.

*Step 2 (Commutator as gradient).* For $a \in \mathcal{A}$, the commutator $[D, a]$ acts on spinors. In the classical limit, $[D, f] \to \gamma(\nabla f)$ where $\gamma$ is Clifford multiplication.

*Step 3 (Distance duality).* The supremum over Lipschitz functions with $\|[D, f]\| \leq 1$ recovers the geodesic distance: $d_D(x, y) = d_g(x, y)$ for the Riemannian metric $g$.

*Step 4 (Oscillation detection).* Oscillatory breakdown corresponds to $\|[D, a]\| \to \infty$ for bounded $a$—the derivative blows up. This violates interface permit $\mathrm{GC}_\nabla$.
:::

---

### Antichain-Surface Correspondence (BoundaryCheck Predicate)

:::{div} feynman-prose
This theorem connects two seemingly unrelated ideas: antichains in causal sets and minimal surfaces in geometry. Let me explain why this matters for boundary checks in the Sieve.

First, what is an antichain? In a partially ordered set, an antichain is a collection of elements where none is "before" or "after" any other. In a causal set representing spacetime, an antichain is a spacelike surface, a moment of "now" where all points are simultaneous.

Menger's theorem from graph theory tells us that the max-flow through a network equals the min-cut. This is the famous max-flow min-cut duality. In the causal set context, the "flow" is information traveling through time, and the "cut" is a spacelike surface that information must cross.

Here is the beautiful part: as you take finer and finer causal sets approximating a smooth spacetime, the minimal antichains converge to minimal surfaces. The discrete combinatorial notion becomes the continuous geometric notion. This is Gamma-convergence at work.

For the Sieve, this means boundary interactions can be measured by cut sizes. The boundary check asks: how much information flow crosses this interface? The antichain-surface correspondence gives us a rigorous way to compute this from discrete data.
:::

:::{prf:theorem} [LOCK-Antichain] Antichain-Surface Correspondence
:label: mt-lock-antichain

**Sieve Target:** Node 13 (BoundaryCheck) — boundary interaction measure via min-cut/max-flow

**Statement:** In a causal set $(C, \prec)$ with interface permit $\mathrm{Cap}_H$, discrete antichains converge to minimal surfaces in the continuum limit. The correspondence:

- **Antichain** (maximal set of pairwise incomparable elements) $\leftrightarrow$ **Spacelike hypersurface**
- **Cut size** $|A|$ in causal graph $\leftrightarrow$ **Area** of minimal surface

**Bridge Type:** Causal Sets $\leftrightarrow$ Riemannian Geometry

**Dictionary:**
- Antichain $A$ $\to$ Surface $\Sigma$
- Causal order $\prec$ $\to$ Metric structure
- Min-cut in causal graph $\to$ Minimal surface (area-minimizing)

**Certificate Produced:** $K_{13}^+$ with payload $(|A|, \text{Area}(Σ), \text{min-cut})$

**Literature:** {cite}`Menger27`; {cite}`DeGiorgi75`; {cite}`Sorkin91`; {cite}`BombelliLeeEtAl87`
:::

:::{prf:proof}
:label: proof-mt-lock-antichain

*Step 1 (Menger's theorem).* In a finite graph, the maximum flow from source to sink equals the minimum cut capacity. For causal graphs, this relates the "information flow" through time to the minimal separating surface.

*Step 2 (Discrete approximation).* Let $C_n$ be a sequence of causal sets approximating a Lorentzian manifold $(M, g)$. The number of elements in a causal diamond scales as the spacetime volume: $|J^+(p) \cap J^-(q)| \sim V_g(D(p,q))$.

*Step 3 (Γ-convergence).* The discrete cut functional:

$$F_n(A) = \frac{|A|}{n^{(d-1)/d}}$$

Γ-converges to the area functional:

$$F(Σ) = \operatorname{Area}_g(Σ)$$

for hypersurfaces $Σ$ in the continuum limit.

*Step 4 (Minimal surface emergence).* Minimizers of $F_n$ (minimal antichains) converge to minimizers of $F$ (minimal surfaces). This is the boundary measure in the Sieve.
:::

---

(sec-barrier-defense-theorems)=
## Barrier Defense Theorems

*These theorems prove that barriers actually stop singularities.*

:::{div} feynman-prose
We have just seen how the gates work, how each predicate asks a yes/no question about the trajectory. But what happens when the system fails a check? This is where barriers come in.

A barrier is not just a wall, it is a mathematical proof that certain trajectories cannot exist. The theorems in this section prove that if a singularity tried to form, it would violate a conservation law, an energy bound, or an information-theoretic limit. The singularity is not blocked by force but by logic. It simply cannot satisfy all the constraints simultaneously.

Think of it like a bank vault. You do not stop robbers by posting more guards. You design the vault so that opening it requires solving an impossible equation. The robber can try all day, but the math does not work out. That is what these barrier theorems do: they show that the math does not work out for singularities.
:::

---

### Saturation Principle (BarrierSat)

:::{div} feynman-prose
The saturation principle is about energy budgets. Here is the key insight: pathological behavior requires energy, and if you can bound the energy budget, you bound the pathology.

The Foster-Lyapunov condition is the workhorse of stochastic stability. It says: find a function $\mathcal{V}$ that acts like "potential energy" and show that the dynamics push it downward on average, except possibly in some compact region $C$ where it can grow by at most $b$. Then the system must eventually settle into a regime where the average energy is at most $E^* = b/\lambda$.

What makes this a barrier? The singularity is energy-hungry. It wants to concentrate infinite energy at a point. But the drift condition puts a ceiling on the available energy. The system can try to concentrate, but it keeps getting pushed back. Asymptotically, it saturates at $E^*$ and cannot blow up.

This is why Lyapunov functions are so powerful: they convert dynamical questions (will this system blow up?) into static questions (is this function bounded?).
:::

:::{prf:theorem} [UP-Saturation] Saturation Principle
:label: mt-up-saturation-principle

**Sieve Target:** BarrierSat — drift control prevents blow-up

**Statement:** Let $\mathcal{S}$ be a hypostructure where interface permit $D_E$ depends on an analytic inequality of the form $\Phi(u) + \alpha \mathfrak{D}(u) \leq \text{Drift}(u)$. If there exists a Lyapunov function $\mathcal{V}: X \to [0, \infty)$ satisfying the **Foster-Lyapunov drift condition**:

$$\mathcal{L}\mathcal{V}(x) \leq -\lambda \mathcal{V}(x) + b \cdot \mathbf{1}_C(x)$$

for generator $\mathcal{L}$, constant $\lambda > 0$, bound $b < \infty$, and compact set $C$, then:

1. The process is positive recurrent
2. Energy blow-up (Mode C.E) is prevented
3. A threshold energy $E^* = b/\lambda$ bounds the asymptotic energy

**Required Interface Permits:** $D_E$ (Dissipation), $\mathrm{SC}_\lambda$ (Scaling)

**Prevented Failure Modes:** C.E (Energy Blow-up), S.E (Supercritical Cascade)

**Certificate Produced:** $K_{\text{Sat}}^{\text{blk}}$ with payload $(E^*, \lambda, b, C)$

**Literature:** {cite}`MeynTweedie93`; {cite}`HairerMattingly06`; {cite}`Khasminskii12`; {cite}`Lyapunov1892`
:::

:::{prf:proof}
:label: proof-mt-up-saturation-principle

*Step 1 (Generator bound).* Apply Itô's lemma to $\mathcal{V}(X_t)$:

$$d\mathcal{V}(X_t) = \mathcal{L}\mathcal{V}(X_t) dt + \text{martingale}$$

*Step 2 (Drift control).* The drift condition ensures:

$$\mathbb{E}[\mathcal{V}(X_t)] \leq e^{-\lambda t} \mathcal{V}(x_0) + \frac{b}{\lambda}(1 - e^{-\lambda t})$$

*Step 3 (Asymptotic bound).* As $t \to \infty$:

$$\limsup_{t \to \infty} \mathbb{E}[\mathcal{V}(X_t)] \leq \frac{b}{\lambda} = E^*$$

*Step 4 (Pathological saturation).* Pathologies saturate the inequality: the threshold energy $E^*$ is determined by the ground state of the singular profile. Energy cannot exceed $E^*$ asymptotically.
:::

---

### Physical Computational Depth Limit (BarrierCausal)

:::{div} feynman-prose
This is one of the most surprising barriers because it comes from quantum mechanics and computation theory. The guard here is asking: "Can you actually compute this singularity?"

Margolus and Levitin proved something remarkable in 1998: there is a fundamental speed limit on computation. The maximum rate at which a quantum system can transition between orthogonal states is $4E/\pi\hbar$, where $E$ is the energy above the ground state. This is not an engineering limit, it is physics.

Now think about what a Zeno singularity requires. It needs infinitely many distinguishable events to happen in finite time. But each distinguishable event is a state transition, and each transition takes at least $\pi\hbar/4E$ of time. With finite energy, you can only have finitely many transitions.

This is a profound result. It says that certain mathematical singularities are not just unlikely, they are physically impossible. They would require infinite computational depth in finite time, which violates a fundamental bound from quantum mechanics. The universe simply cannot process that much information.
:::

:::{prf:theorem} [UP-CausalBarrier] Physical Computational Depth Limit
:label: mt-up-causal-barrier

**Source:** Margolus-Levitin Theorem (1998)

**Sieve Target:** BarrierCausal — infinite event sequences require infinite energy-time (Zeno exclusion)

**Input Certificates:**
1. $K_{D_E}^+$: System has finite average energy $E$ relative to ground state
2. $K_{C_\mu}^+$: Singular region confined to finite volume

**Statement (Margolus-Levitin Theorem):**
The maximum rate of orthogonal state evolution is bounded by energy:

$$\nu_{\max} \leq \frac{4E}{\pi\hbar}$$

Therefore, the maximum number of distinguishable events in time interval $[0,T]$ is:

$$N(T) \leq \frac{4}{\pi\hbar} \int_0^T (E(t) - E_0) \, dt$$

**Required Interface Permits:** $D_E$ (Finite Energy), $C_\mu$ (Confinement)

**Prevented Failure Modes:** C.C (Event Accumulation / Zeno)

**Blocking Logic:**
If a singularity requires an infinite event sequence (Zeno accumulation) but the energy integral is finite (Node 1 passes), then Mode C.C is physically impossible:

$$K_{D_E}^+ \wedge (N_{\text{req}} = \infty) \Rightarrow K_{\mathrm{Rec}_N}^{\mathrm{blk}}$$

**Certificate Produced:** $K_{\mathrm{Rec}_N}^{\text{blk}}$ with payload $(E_{\max}, N_{\max}, T_{\text{horizon}})$ where $N_{\max} = \frac{4 E_{\max} T_{\text{horizon}}}{\pi\hbar}$

**Literature:** {cite}`MargolisLevitin98`; {cite}`Lloyd00`; {cite}`CoverThomas06`
:::

:::{prf:proof}
:label: proof-mt-up-causal-barrier

*Step 1 (Energy bound).* By interface permit $D_E$, the system has finite average energy $E = \int_0^T (E(t) - E_0) dt < \infty$.

*Step 2 (Margolus-Levitin).* By quantum mechanics, the minimum time to transition between orthogonal states is $\Delta t \geq \pi\hbar / 4E$. This is a fundamental limit independent of the physical implementation.

*Step 3 (Event counting).* If $N$ distinguishable events (state changes) occur in time $T$, then:

$$N \leq \frac{4}{\pi\hbar} \int_0^T E(t) \, dt$$

*Step 4 (Zeno exclusion).* A Zeno sequence (infinitely many events in finite time) would require $N = \infty$ with $T < \infty$. By the bound above, this requires $\int_0^T E(t) dt = \infty$, contradicting the energy certificate $K_{D_E}^+$.
:::

---

### Capacity Barrier (BarrierCap)

:::{div} feynman-prose
The capacity barrier uses geometric measure theory to show that singularities cannot hide in thin sets. Let me explain what "capacity" means here.

Ordinary measure tells you how much volume a set has. Capacity is different: it tells you how much energy a set can "hold" or "support." A set can have zero volume but positive capacity (like a curve in the plane), or zero capacity and zero volume (like a point).

The key insight is that certain sets are too thin to carry any energy. If a blow-up tried to concentrate on such a set, the energy would leak out. The capacity barrier makes this precise: if the total capacity of your thin sets is finite, then the trajectory cannot spend infinite time in them.

Why does this matter? Some singularities try to form on lower-dimensional sets, concentrating energy along curves or surfaces rather than at points. The capacity barrier catches these: if the dimension is below a critical threshold, the set cannot support the energy needed for blow-up. A zero-energy profile cannot mediate a singularity.
:::

:::{prf:theorem} [LOCK-Tactic-Capacity] Capacity Barrier
:label: mt-lock-tactic-capacity

**Sieve Target:** BarrierCap — zero-capacity sets cannot sustain energy

**Statement:** Let $\mathcal{S}$ be a hypostructure with geometric background (BG) satisfying interface permit $\mathrm{Cap}_H$. Let $(B_k)$ be a sequence of subsets with increasing "thinness" (e.g., tubular neighborhoods of codimension-$\kappa$ sets with radius $r_k \to 0$) such that:

$$\sum_k \operatorname{Cap}(B_k) < \infty$$

Then **occupation time bounds** hold: the trajectory cannot spend infinite time in thin sets.

**Required Interface Permits:** $\mathrm{Cap}_H$ (Capacity), $\mathrm{TB}_\pi$ (Background Geometry)

**Prevented Failure Modes:** C.D (Geometric Collapse)

**Certificate Produced:** $K_{\text{Cap}}^{\text{blk}}$ with payload $(\text{Cap}(B), d_c, \mu_T)$

**Literature:** {cite}`Federer69`; {cite}`EvansGariepy92`; {cite}`AdamsHedberg96`; {cite}`Maz'ya85`
:::

:::{prf:proof}
:label: proof-mt-lock-tactic-capacity

*Step 1 (Capacity-codimension bound).* By the background geometry interface permit (BG4):

$$\operatorname{Cap}(B) \leq C \cdot r^{d-\kappa}$$

for sets of codimension $\kappa$ and radius $r$.

*Step 2 (Occupation measure).* The occupation measure $\mu_T(B) = \frac{1}{T}\int_0^T \mathbf{1}_B(u(t)) dt$ satisfies:

$$\mu_T(B_k) \leq \frac{C_{\text{cap}}(\Phi(x) + T)}{\operatorname{Cap}(B_k)}$$

*Step 3 (Summability).* For $\sum_k \operatorname{Cap}(B_k) < \infty$:

$$\sum_k \mu_T(B_k) < \infty$$

The trajectory can spend at most finite total time in all thin sets combined.

*Step 4 (Blocking mechanism).* If a blow-up required concentrating on sets with $\dim(\Sigma) < d_c$ (critical codimension), the capacity is too small to support the energy:

$$\int_\Sigma |V|^2 d\mathcal{H}^{\dim(\Sigma)} < \infty \implies E(V) = 0$$
A zero-energy profile cannot mediate blow-up.
:::

---

### Topological Sector Suppression (BarrierAction)

:::{div} feynman-prose
This barrier uses concentration of measure, one of the most powerful tools in modern probability theory. The idea is simple: in high-dimensional spaces, almost everything is typical.

Think of a sphere in high dimensions. Almost all the volume is concentrated near the equator. A random point is almost certainly close to the average. This is concentration of measure: deviations from the typical are exponentially suppressed.

The topological sector suppression theorem applies this to action gaps. If you want to reach a nontrivial topological sector (like a configuration with nonzero winding number), you need to climb over an energy barrier of height $\Delta$. The log-Sobolev inequality tells you that the probability of being that far from typical decays like $\exp(-c\Delta^2)$.

This is a probabilistic barrier. The singularity might be mathematically possible, but it requires reaching a state that is exponentially rare. As the action gap increases, or as the concentration constant improves, the probability of encountering such states becomes negligible. The dangerous topological sectors are effectively invisible.
:::

:::{prf:theorem} [UP-Shadow] Topological Sector Suppression
:label: mt-up-shadow

**Sieve Target:** BarrierAction — exponential suppression by action gap

**Statement:** Assume the topological background (TB) with action gap $\Delta > 0$ and an invariant probability measure $\mu$ satisfying a log-Sobolev inequality with constant $\lambda_{\text{LS}} > 0$. Assume the action functional $\mathcal{A}$ is Lipschitz with constant $L > 0$. Then:

$$\mu(\{x : \tau(x) \neq 0\}) \leq C \exp\left(-c \lambda_{\text{LS}} \frac{\Delta^2}{L^2}\right)$$

for universal constants $C = 1$, $c = 1/8$.

**Certificate Produced:** $K_{\text{Action}}^{\text{blk}}$ with payload $(\Delta, \lambda_{\text{LS}}, L)$

**Literature:** {cite}`Herbst75`; {cite}`Lojasiewicz63`; {cite}`Ledoux01`; {cite}`BobkovGotze99`
:::

:::{prf:proof}
:label: proof-mt-up-shadow

*Step 1 (Herbst argument).* The log-Sobolev inequality (LSI) with constant $\lambda_{\text{LS}}$ implies concentration of measure. For any 1-Lipschitz function $f$:

$$\mu(\{f \geq \mathbb{E}_\mu[f] + t\}) \leq \exp\left(-\frac{\lambda_{\text{LS}} t^2}{2}\right)$$

*Step 2 (Action gap setup).* By interface permit $\mathrm{TB}_\pi$ (action gap), states in nontrivial topological sectors have:

$$\tau(x) \neq 0 \implies \mathcal{A}(x) \geq \mathcal{A}_{\min} + \Delta$$

*Step 3 (Lipschitz rescaling).* The action $\mathcal{A}$ has Lipschitz constant $L$. Define $f = \mathcal{A}/L$ (1-Lipschitz). Then:

$$\{x : \tau(x) \neq 0\} \subseteq \{f \geq f_{\min} + \Delta/L\}$$

*Step 4 (Measure bound).* By the Herbst estimate:

$$\mu(\{x : \tau(x) \neq 0\}) \leq \exp\left(-\frac{\lambda_{\text{LS}} (\Delta/L)^2}{2}\right) = \exp\left(-\frac{\lambda_{\text{LS}} \Delta^2}{2L^2}\right)$$

*Step 5 (Exponential suppression).* The probability of residing in a nontrivial topological sector decays exponentially with the action gap squared. Large $\Delta$ or strong LSI exponentially suppresses topological obstructions.
:::

---

### Bode Sensitivity Integral (BarrierBode)

:::{div} feynman-prose
The Bode integral is a conservation law from control theory, and it tells you that you cannot have your cake and eat it too when it comes to disturbance rejection.

Imagine you are designing a feedback controller. You want to reject disturbances, which means making the sensitivity $|S(j\omega)|$ small. But Bode showed that the integral of $\log|S|$ over all frequencies is fixed by the unstable poles of your system. If you push sensitivity down at some frequencies, it must pop up somewhere else. This is the waterbed effect: push down one end, the other end rises.

Why is this a barrier? Because it prevents infinite stiffness. A system that could perfectly reject all disturbances at all frequencies would need $|S| < 1$ everywhere. But the Bode integral says the area under $\log|S|$ is constant (and positive if you have unstable poles). You cannot make $|S|$ arbitrarily small everywhere.

This connects to the Sieve because certain failure modes involve controllers that are too aggressive, that try to reject disturbances with infinite gain. The Bode integral says no: there is a fundamental limit on how much disturbance rejection you can achieve, and exceeding it creates instability elsewhere.
:::

:::{prf:theorem} Bode Sensitivity Integral
:label: thm-bode

**Sieve Target:** BarrierBode — waterbed effect conservation law

**Statement:** Let $\mathcal{S}$ be a feedback control system with loop transfer function $L(s)$, sensitivity $S(s) = (1 + L(s))^{-1}$, and $n_p$ unstable poles $\{p_i\}$ in the right half-plane. Then:

**Waterbed Effect:**

$$\int_0^\infty \log |S(j\omega)| \, d\omega = \pi \sum_{i=1}^{n_p} p_i$$

**Consequence:** If $|S(j\omega)| < 1$ (good rejection) on some frequency band $[\omega_1, \omega_2]$, then there must exist frequencies where $|S(j\omega)| > 1$ (amplification). Sensitivity cannot be uniformly suppressed.

**Required Interface Permits:** $\mathrm{LS}_\sigma$ (Stiffness/Stability)

**Prevented Failure Modes:** S.D (Infinite Stiffness), C.E (Instability)

**Certificate Produced:** $K_{\text{Bode}}^{\text{blk}}$ with payload $(\int \log|S| d\omega, \{p_i\})$

**Literature:** {cite}`Bode45`; {cite}`DoyleFrancisTannenbaum92`; {cite}`SkogestadPostlethwaite05`; {cite}`Freudenberg85`
:::

:::{prf:proof}
:label: proof-thm-bode

*Step 1 (Cauchy integral setup).* Consider the contour integral of $\log S(s)$ around the right half-plane: a semicircle from $-jR$ to $jR$ closed by the imaginary axis.

*Step 2 (Residue calculation).* The only singularities of $\log S(s)$ inside the contour are at the zeros of $1 + L(s)$ (closed-loop poles). For stable systems, there are none. The contribution from unstable poles of $L(s)$ comes from the integral representation.

*Step 3 (Arc contribution).* As $R \to \infty$, the semicircular arc contributes zero if $L(s) \to 0$ as $|s| \to \infty$ (strictly proper $L$).

*Step 4 (Imaginary axis integral).* The integral along the imaginary axis is:

$$\int_{-j\infty}^{j\infty} \log S(s) \, ds = 2j \int_0^\infty \log|S(j\omega)| d\omega$$

(using $\log S(-j\omega) = \overline{\log S(j\omega)}$ for real systems).

*Step 5 (Poisson-Jensen formula).* By the Poisson-Jensen formula for functions analytic in the right half-plane:

$$\int_0^\infty \log|S(j\omega)| d\omega = \pi \sum_{p_i \in \operatorname{RHP}} \operatorname{Re}(p_i)$$

where the sum is over unstable poles of $L(s)$.

*Step 6 (Waterbed interpretation).* The integral is fixed by unstable poles. Pushing down $|S|$ at some frequencies forces it up elsewhere—this is the "waterbed effect."
:::

---

### Epistemic Horizon Principle (BarrierEpi)

:::{div} feynman-prose
The epistemic horizon is perhaps the most subtle barrier because it operates at the level of information itself. The question is not whether the singularity has enough energy, but whether it has enough information.

Consider what happens as a system evolves chaotically. Information about the initial condition gets smeared out. Lyapunov exponents measure how fast this happens: positive exponents mean nearby trajectories diverge exponentially. Pesin's formula connects this to entropy production. The system is literally forgetting its past.

Now, a singularity requires very specific information to coordinate at the blow-up point. But the data processing inequality says information can only decrease through processing, never increase. If entropy production exceeds the channel capacity between initial conditions and the would-be singularity, the necessary information simply cannot be transmitted.

This is Landauer's principle applied to dynamics: computation requires energy, and losing information requires dissipation. The epistemic horizon says there is a fundamental limit on how much information about the past can survive to coordinate a future singularity. The horizon is not spatial but informational.
:::

:::{prf:theorem} [ACT-Horizon] Epistemic Horizon Principle
:label: mt-act-horizon

**Sieve Target:** BarrierEpi — one-way barrier via data processing inequality

**Statement:** Information acquisition is bounded by thermodynamic dissipation. The **Landauer bound** and **data processing inequality** establish fundamental limits:

1. **Landauer's principle:** Erasing one bit of information requires at least $k_B T \ln 2$ of energy dissipation
2. **Data processing inequality:** For any Markov chain $X \to Y \to Z$:

$$I(X; Z) \leq I(X; Y)$$

Information cannot increase through processing.

**Required Interface Permits:** $\mathrm{Cap}_H$ (Capacity), $D_E$ (Dissipation)

**Prevented Failure Modes:** D.E (Observation), D.C (Measurement)

**Certificate Produced:** $K_{\text{Epi}}^{\text{blk}}$ with payload $(I_{\max}, h_\mu, k_B T \ln 2)$

**Literature:** {cite}`CoverThomas06`; {cite}`Landauer61`; {cite}`Bennett82`; {cite}`Pesin77`
:::

:::{prf:proof}
:label: proof-mt-act-horizon

*Step 1 (Entropy production).* For a system with positive Lyapunov exponents $\lambda_i > 0$, Pesin's formula gives the KS entropy:

$$h_\mu = \sum_{\lambda_i > 0} \lambda_i > 0$$

*Step 2 (Total entropy).* The total entropy production up to time $T_*$ is:

$$\Sigma(T_*) = \int_0^{T_*} h_\mu(S_\tau) d\tau > 0$$

*Step 3 (Data processing).* By the data processing inequality, for $u_0 \to u(t) \to V_\lambda$:

$$I(u_0; V_\lambda) \leq I(u(t); V_\lambda) \leq I(u_0; u(t))$$

*Step 4 (Mutual information decay).* Entropy production causes information loss:

$$I(u_0; u(T_*)) \leq H(u_0) - \Sigma(T_*)$$

*Step 5 (Channel capacity bound).* The singularity requires information about the initial condition to be preserved to the blow-up time. The channel capacity is bounded:

$$I(u_0; V_\lambda) \leq \min\{C_\Phi(\lambda), H(u_0) - \Sigma(T_*)\}$$

If entropy production exceeds channel capacity, the singularity cannot form.
:::

---

(sec-surgery-construction-theorems)=
## Surgery Construction Theorems

*These theorems provide constructive methods for purple surgery nodes.*

:::{div} feynman-prose
Now we come to the most constructive part of the Sieve: the surgery nodes. Gates say yes or no. Barriers prove impossibility. But surgery nodes actually fix problems. They are the mechanics who repair the system when something goes wrong.

The philosophy here is different from blocking. Instead of proving a singularity cannot happen, we provide a recipe for modifying the system so that it can continue past where it would have failed. These are not hacks or patches. They are principled mathematical techniques that preserve the essential structure while eliminating the pathology.

Each surgery theorem tells you: when you encounter this type of failure, here is how to lift, cut, extend, or transform the system to get past it. Hairer's regularity structures, Perelman's Ricci flow surgery, BRST ghost fields, these are all surgery techniques that mathematicians developed to handle specific pathologies. The Sieve organizes them systematically.
:::

---

### Regularity Lift Principle (SurgSE)

:::{div} feynman-prose
Martin Hairer won the Fields Medal for this, and rightly so. The regularity lift principle solves a problem that seemed impossible: how do you make sense of equations that involve multiplying distributions?

Here is the issue. Distributions, like the Dirac delta function, are not ordinary functions. You cannot square them. If your equation involves noise that is too rough, like space-time white noise, the nonlinear terms become meaningless. $\xi^2$ does not exist as a distribution.

Hairer's insight was to lift the equation to a richer space where these products do make sense. You build an algebraic structure that includes all the singular terms you need, with explicit rules for how they multiply. Then you solve the equation in this lifted space and project back down to get your actual solution.

This is surgery for singular SPDEs. When the original equation is ill-posed, you do not give up. You embed it in a larger structure where it becomes well-posed, solve it there, and then reconstruct the solution in the original space. The regularity structure is the scaffolding that makes this possible.
:::

:::{prf:theorem} [ACT-Lift] Regularity Lift Principle
:label: mt-act-lift

**Rigor Class:** L (Literature-Anchored) — see {prf:ref}`def-rigor-classification`

**Bridge Verification:**
1. *Hypothesis Translation:* Singular SPDE $\partial_t u = \mathcal{L}u + F(u,\xi)$ with distributional noise $\xi$ satisfies subcriticality condition $\gamma_c := \min_\tau(|\tau|) > 0$
2. *Domain Embedding:* $\iota: \mathbf{Hypo}_T \to \mathbf{RegStruct}(\mathscr{T})$ lifting state to modelled distribution $\hat{u} \in \mathcal{D}^\gamma$
3. *Conclusion Import:* Hairer's reconstruction theorem {cite}`Hairer14` $\Rightarrow K_{\text{SurgSE}}$ (solution $u = \mathcal{R}\hat{u}$ exists and is unique)

**Sieve Target:** SurgSE (Regularity Extension) — rough path $\to$ regularity structure lift

**Repair Class:** Symmetry (Algebraic Lifting)

**Statement:** Consider a singular SPDE:

$$\partial_t u = \mathcal{L}u + F(u, \xi)$$

where $\xi$ is distributional noise (e.g., space-time white noise) and $F$ involves products ill-defined in classical distribution theory. There exists:

1. A **regularity structure** $\mathscr{T} = (T, A, G)$ with model space $T$, grading $A$, and structure group $G$
2. A **lift** $\hat{u} \in \mathcal{D}^\gamma$ (modelled distributions of regularity $\gamma$)
3. A **reconstruction operator** $\mathcal{R}: \mathcal{D}^\gamma \to \mathcal{D}'$ such that $u = \mathcal{R}\hat{u}$ solves the SPDE

**Certificate Produced:** $K_{\text{SurgSE}}$ with payload $(\mathscr{T}, \hat{u}, \mathcal{R})$

**Literature:** {cite}`Hairer14`; {cite}`GubinelliImkellerPerkowski15`; {cite}`BrunedHairerZambotti19`; {cite}`FrizHairer14`
:::

:::{prf:proof}
:label: proof-mt-act-lift

*Step 1 (Regularity structure).* Build an abstract polynomial-like structure that encodes:
- Basis elements representing canonical noise terms
- Multiplication rules encoding renormalized products
- Group action implementing Taylor reexpansion under translation

*Step 2 (Admissible model).* Construct a concrete realization $\Pi_x: T \to \mathcal{D}'$ that:
- Maps basis elements to actual distributions
- Satisfies coherence: $\Pi_y = \Pi_x \circ \Gamma_{xy}$ for structure group elements

*Step 3 (Modelled distributions).* Define $\hat{u} \in \mathcal{D}^\gamma$ by local Taylor-like expansion:

$$\hat{u}(x) = \sum_{\tau \in T, |\tau| < \gamma} u_\tau(x) \cdot \tau$$

with regularity controlled by $|\hat{u}(y) - \Gamma_{xy}\hat{u}(x)| \lesssim |x-y|^\gamma$

*Step 4 (Abstract fixed point).* Solve the lifted equation:

$$\hat{u} = P * \hat{F}(\hat{u}, \hat{\xi})$$

in the space of modelled distributions. The fixed point exists by Banach contraction.

*Step 5 (Reconstruction).* Apply $\mathcal{R}$ to obtain $u = \mathcal{R}\hat{u} \in \mathcal{D}'$, the actual solution.
:::

---

### Structural Surgery Principle (SurgTE)

:::{div} feynman-prose
Perelman's surgery for Ricci flow is perhaps the most famous example of mathematical surgery. It is how he proved the Poincare conjecture, and it is exactly the kind of technique we need in the Sieve.

The Ricci flow evolves a metric on a manifold, trying to smooth it out. But sometimes it develops singularities: necks that pinch off, regions that collapse. Hamilton studied this for years but could not get past the singularities. Perelman showed how to cut them out and keep going.

Here is the picture. When a neck starts to pinch, you detect it early, before the curvature actually blows up. You cut the manifold along the neck, throw away the pinching piece, and glue in a smooth cap. The surgery removes the singularity while preserving the topology that matters.

The key insight is that you can do this finitely many times. Each surgery costs entropy, and entropy is bounded. So the singularities cannot recur forever. Eventually the flow becomes smooth and you can analyze it completely. This is surgery as a systematic procedure, not a one-time fix.
:::

:::{prf:theorem} [ACT-Surgery] Structural Surgery Principle
:label: mt-act-surgery-2

**Sieve Target:** SurgTE (Topological Extension) — Perelman cut-and-paste surgery

**Repair Class:** Topology (Structural Excision)

**Statement:** Let $(M, g(t))$ be a Ricci flow developing a singularity at time $T$. There exists a **surgery procedure**:

1. **Detect**: Identify neck regions where curvature exceeds threshold $|Rm| > \rho^{-2}$
2. **Excise**: Cut the manifold along approximate round spheres in neck regions
3. **Cap**: Glue in standard caps (round hemispheres with controlled geometry)
4. **Continue**: Restart the flow from the surgered manifold

The procedure maintains:
- Uniform local geometry control
- Monotonicity of Perelman's $\mathcal{W}$-entropy
- Finite number of surgeries in finite time

**Certificate Produced:** $K_{\text{SurgTE}}$ with payload $(M_{\text{new}}, n_{\text{surg}}, \mathcal{W})$

**Literature:** {cite}`Perelman02`; {cite}`Perelman03a`; {cite}`Perelman03b`; {cite}`KleinerLott08`; {cite}`Hamilton97`
:::

:::{prf:proof}
:label: proof-mt-act-surgery-2

*Step 1 (Canonical neighborhood theorem).* Near high-curvature points, the geometry is modeled by one of:
- Shrinking round spheres $S^n$
- Shrinking cylinders $S^{n-1} \times \mathbb{R}$
- Quotients of the above
This provides surgery location candidates.

*Step 2 (Neck detection).* A neck is a region diffeomorphic to $S^{n-1} \times [-L, L]$ with:

$$\left|g - g_{cyl}\right| < \varepsilon$$

for the standard cylinder metric $g_{cyl}$.

*Step 3 (Surgery procedure).* Cut along $S^{n-1} \times \{0\}$, discard the high-curvature component, glue a standard cap:

$$M_{\text{new}} = M_{\text{low}} \cup_\partial \operatorname{Cap}$$

where Cap has uniformly bounded geometry.

*Step 4 (Entropy control).* Perelman's $\mathcal{W}$-entropy satisfies:

$$\mathcal{W}(g_{\text{new}}) \geq \mathcal{W}(g_{\text{old}}) - C\varepsilon$$
Surgeries only decrease entropy by controlled amounts.

*Step 5 (Finite surgery).* The entropy is bounded below; each surgery costs at least $\delta > 0$ entropy. Total surgeries $\leq (\mathcal{W}_{\max} - \mathcal{W}_{\min})/\delta < \infty$.
:::

---

### Projective Extension (SurgCD)

:::{div} feynman-prose
Sometimes the problem is not a singularity in the dynamics but a singularity in the constraint set. The feasible region collapses to measure zero, and there is nowhere left to go. Projective extension is surgery for this.

The technique comes from optimization theory. When constraints are too tight, you relax them. You introduce slack variables that let you violate the constraints by a small amount $\varepsilon$. The relaxed problem has positive capacity, meaning there is actually room to move around.

As you reduce $\varepsilon$ toward zero, the relaxed solutions converge to the original constrained solution, if one exists. The slack variables are scaffolding: they give you room to work while you approach the tight constraints from the interior.

Interior point methods use exactly this idea. The logarithmic barrier keeps you away from constraint boundaries while the central path guides you toward the optimum. You never actually reach the boundary, but you get arbitrarily close. This is surgery by regularization: soften the hard constraint, solve the soft problem, take the limit.
:::

:::{prf:theorem} [ACT-Projective] Projective Extension
:label: mt-act-projective

**Sieve Target:** SurgCD (Constraint Relaxation) — slack variable method for geometric collapse

**Repair Class:** Geometry (Constraint Relaxation)

**Statement:** Let $K = \{x : g_i(x) \leq 0, h_j(x) = 0\}$ be a constraint set that has collapsed to measure zero ($\operatorname{Cap}(K) = 0$). Introduce **slack variables** $s_i \geq 0$ to obtain the relaxed problem:

$$K_\varepsilon = \{(x, s) : g_i(x) \leq s_i, h_j(x) = 0, \|s\| \leq \varepsilon\}$$

The relaxation satisfies:
1. $\operatorname{Cap}(K_\varepsilon) > 0$ for $\varepsilon > 0$
2. $K_\varepsilon \to K$ as $\varepsilon \to 0$ in Hausdorff distance
3. Solutions of the relaxed problem converge to solutions of the original (if they exist)

**Certificate Produced:** $K_{\text{SurgCD}}$ with payload $(\varepsilon, s^*, x^*)$

**Literature:** {cite}`BoydVandenberghe04`; {cite}`NesterovNemirovskii94`; {cite}`Rockafellar70`; {cite}`BenTalNemirovski01`
:::

:::{prf:proof}
:label: proof-mt-act-projective

*Step 1 (Slack introduction).* Replace hard constraint $g_i(x) \leq 0$ with soft constraint $g_i(x) - s_i \leq 0$ and $s_i \geq 0$. The feasible region expands.

*Step 2 (Capacity restoration).* For $\varepsilon > 0$:

$$\operatorname{Vol}(K_\varepsilon) \geq c_n \varepsilon^{n_s} \cdot \operatorname{Vol}(U)$$

where $n_s$ is the number of slack variables and $U$ is a neighborhood. Positive volume implies positive capacity.

*Step 3 (Barrier function).* Use logarithmic barrier:

$$f_\mu(x, s) = f(x) - \mu \sum_i \log s_i$$

The central path follows $\nabla f_\mu = 0$ as $\mu \to 0$.

*Step 4 (Convergence).* As $\varepsilon \to 0$ (equivalently $\mu \to 0$), the relaxed solutions converge to the original constrained optimum by standard interior point convergence theory.
:::

---

### Derived Extension / BRST (SurgSD)

:::{div} feynman-prose
The BRST construction is surgery for gauge theories, and it is genuinely clever. The problem is that gauge symmetry creates infinities: the path integral wants to sum over all configurations, but configurations related by gauge transformations are physically identical. You are counting the same thing infinitely many times.

The naive fix is gauge fixing: pick one representative from each gauge orbit. But this introduces determinants that depend on your gauge choice, and they often diverge. Faddeev and Popov showed that these determinants can be written as integrals over new fields, the ghost fields.

Here is the magic: ghost fields are Grassmann-valued, meaning they anticommute. When you integrate over them, determinants appear in the numerator instead of the denominator. Fermions give $\det(M)$ while bosons give $1/\det(M)$. The ghosts exactly cancel the divergent gauge orbit volume.

The BRST symmetry packages all of this elegantly. There is a nilpotent operator $s$ with $s^2 = 0$, and physical states are those in the kernel of $s$ modulo the image, the BRST cohomology. This is surgery at its most algebraic: extend the space with auxiliary fields, introduce a symmetry, and quotient by its cohomology to get the physical Hilbert space.
:::

:::{prf:theorem} [ACT-Ghost] Derived Extension / BRST
:label: mt-act-ghost

**Sieve Target:** SurgSD (Symmetry Deformation) — ghost fields cancel divergent determinants

**Repair Class:** Symmetry (Graded Extension)

**Statement:** Let $\mathcal{A}$ be a space of connections with gauge group $\mathcal{G}$. The naive path integral $\int_\mathcal{A} e^{-S} \mathcal{D}A$ diverges due to infinite gauge orbit volume. Introduce **ghost fields** $(c, \bar{c})$ of opposite statistics to obtain:

$$Z = \int e^{-S_{\text{tot}}} \mathcal{D}A \mathcal{D}c \mathcal{D}\bar{c}$$

where $S_{\text{tot}} = S + S_{\text{gf}} + S_{\text{ghost}}$.

The BRST construction provides:
1. **Stiffness Restoration**: $\nabla^2 \Phi_{\text{tot}}$ becomes non-degenerate
2. **Capacity Cancellation**: Ghost fields provide negative capacity exactly canceling gauge orbit volume
3. **Physical State Isomorphism**: $\mathcal{H}_{\text{phys}} \cong H^0_s(X_{\text{BRST}})$ (BRST cohomology)

**Certificate Produced:** $K_{\text{SurgSD}}$ with payload $(s, H^*_s, c, \bar{c})$

**Literature:** {cite}`BecchiRouetStora76`; {cite}`Tyutin75`; {cite}`FaddeevPopov67`; {cite}`Weinberg96`
:::

:::{prf:proof}
:label: proof-mt-act-ghost

*Step 1 (Gauge fixing).* Choose gauge-fixing function $F(A) = 0$. Insert:

$$1 = \int_\mathcal{G} \mathcal{D}g \, \delta(F(A^g)) \det\left(\frac{\delta F(A^g)}{\delta g}\right)$$

*Step 2 (Faddeev-Popov determinant).* The determinant $\det(\delta F/\delta g) = \det(M_{FP})$ is the Faddeev-Popov determinant. Represent it using Grassmann (ghost) fields:

$$\det(M_{FP}) = \int \mathcal{D}c \mathcal{D}\bar{c} \, e^{-\bar{c} M_{FP} c}$$

*Step 3 (BRST symmetry).* The total action $S_{\text{tot}}$ is invariant under the nilpotent BRST transformation:

$$s: A \mapsto Dc, \quad c \mapsto -\frac{1}{2}[c, c], \quad \bar{c} \mapsto B, \quad s^2 = 0$$

*Step 4 (Cohomological quotient).* Physical observables are BRST-closed: $sO = 0$. Physical states form the cohomology:

$$\mathcal{H}_{\text{phys}} = \frac{\ker(s)}{\operatorname{Im}(s)} = H^0_s(X_{\text{BRST}})$$

*Step 5 (Capacity cancellation).* Fermionic integration contributes $(\det M)^{-1}$ for bosons vs. $\det M$ for fermions. Ghost fields (Grassmann) contribute:

$$\int \mathcal{D}c\mathcal{D}\bar{c} \, e^{-\bar{c}Mc} = \det(M)$$

This exactly cancels the divergent gauge orbit volume, yielding finite $Z$.
:::

---

### Adjoint Surgery (SurgBC)

:::{div} feynman-prose
Adjoint surgery handles boundary misalignment, when what happens at the boundary does not match what the bulk dynamics want to do. The tool is the Lagrange multiplier, and the insight is that constraints are just forces in disguise.

Here is the picture. You want to minimize $f(x)$ subject to $g(x) = 0$. The constraint defines a surface, and you need to stay on it. The Lagrangian $\mathcal{L}(x, \lambda) = f(x) + \lambda^T g(x)$ turns the constrained problem into a saddle point problem. The primal variable $x$ wants to minimize, the dual variable $\lambda$ wants to maximize.

At the saddle point, the gradients align: $\nabla f = -\lambda^T \nabla g$. The cost gradient is perpendicular to the constraint surface, which is exactly the condition for constrained optimality. The multiplier $\lambda$ tells you how much the constraint is "costing" you, the shadow price.

In reinforcement learning, this becomes the actor-critic framework. The actor (policy) is the primal variable trying to minimize cost. The critic (value function) is the dual variable enforcing the Bellman constraint. Convergence requires them to stay aligned. When they drift apart, you get instability, the boundary condition (what the critic says) does not match the bulk dynamics (what the actor does). Adjoint surgery fixes this by explicitly coupling the two through the Lagrangian.
:::

:::{prf:theorem} [ACT-Align] Adjoint Surgery
:label: mt-act-align

**Sieve Target:** SurgBC (Boundary Correction) — Lagrange multiplier / Actor-Critic mechanism

**Repair Class:** Boundary (Alignment Enforcement)

**Statement:** When boundary conditions become misaligned with bulk dynamics (Mode B.C), introduce **adjoint variables** $\lambda$ to enforce alignment:

$$\mathcal{L}(x, \lambda) = f(x) + \lambda^T g(x)$$

The saddle-point problem:

$$\min_x \max_\lambda \mathcal{L}(x, \lambda)$$

ensures:
1. Primal variables $x$ minimize objective
2. Dual variables $\lambda$ enforce constraints $g(x) = 0$
3. Gradient alignment: $\nabla_x f \parallel \nabla_x g$ at optimum

**Certificate Produced:** $K_{\text{SurgBC}}$ with payload $(\lambda^*, x^*, \nabla_x f \parallel \nabla_x g)$

**Literature:** {cite}`Pontryagin62`; {cite}`Lions71`; {cite}`KondaMitsalis03`; {cite}`Bertsekas19`
:::

:::{prf:proof}
:label: proof-mt-act-align

*Step 1 (KKT conditions).* At the saddle point $(x^*, \lambda^*)$:

$$\nabla_x f(x^*) + \lambda^{*T} \nabla_x g(x^*) = 0$$

$$g(x^*) = 0$$

*Step 2 (Gradient alignment).* The first condition states:

$$\nabla_x f = -\lambda^T \nabla_x g$$

The cost gradient lies in the span of constraint gradients—they are aligned.

*Step 3 (Pontryagin interpretation).* In optimal control, $\lambda(t)$ is the costate satisfying:

$$\dot{\lambda} = -\nabla_x H(x, u, \lambda)$$

The Hamiltonian $H = f + \lambda^T \dot{x}$ couples state and costate dynamics.

*Step 4 (Actor-Critic mechanism).* In reinforcement learning:
- Actor (primal): updates policy to minimize expected cost
- Critic (dual): estimates value function (Lagrange multiplier)
- Convergence requires actor-critic alignment, preventing boundary misalignment.
:::

---

### Lyapunov Compactification (SurgCE)

:::{div} feynman-prose
The final surgery handles escape to infinity. Some systems do not blow up, they simply run away. The state goes to infinity in finite time, or asymptotically forever. Conformal compactification brings infinity into the picture by shrinking it to a finite boundary.

The idea goes back to Penrose's conformal diagrams for spacetime. Multiply the metric by a conformal factor $\Omega$ that goes to zero at infinity. Distances get squished: what was infinitely far away becomes finitely distant in the new metric. Infinity becomes a boundary $\partial_\Omega M$ that you can actually draw on paper.

Why is this surgery? Because it tames escape trajectories. A trajectory that would run off to infinity in the original metric reaches the conformal boundary in finite conformal time. You can now analyze what happens "at infinity" by studying the boundary behavior.

This is especially powerful in general relativity, where Penrose diagrams let you see the causal structure of entire spacetimes at a glance. Null infinity, spatial infinity, timelike infinity, they all become concrete locations on the boundary. For the Sieve, it means escape modes can be handled: compactify the space, bring infinity into view, and analyze the boundary conditions there.
:::

:::{prf:theorem} [ACT-Compactify] Lyapunov Compactification
:label: mt-act-compactify

**Sieve Target:** SurgCE (Conformal Extension) — conformal rescaling bounds infinite domains

**Repair Class:** Geometry (Conformal Compactification)

**Statement:** Let $(M, g)$ be a non-compact Riemannian manifold with possibly infinite diameter. There exists a **conformal factor** $\Omega: M \to (0, 1]$ such that:

1. $\tilde{g} = \Omega^2 g$ has finite diameter
2. The conformal boundary $\partial_\Omega M = \{\Omega = 0\}$ compactifies $M$
3. Trajectories approaching infinity in $(M, g)$ approach $\partial_\Omega M$ in finite $\tilde{g}$-distance

**Certificate Produced:** $K_{\text{SurgCE}}$ with payload $(\Omega, \tilde{g}, \partial_\Omega M)$

**Literature:** {cite}`Penrose63`; {cite}`HawkingEllis73`; {cite}`ChoquetBruhat09`; {cite}`Wald84`
:::

:::{prf:proof}
:label: proof-mt-act-compactify

*Step 1 (Conformal factor construction).* Choose $\Omega$ vanishing at infinity:

$$\Omega(x) = \frac{1}{1 + d_g(x, x_0)^2}$$

or for asymptotically flat/hyperbolic spaces, use geometric constructions.

*Step 2 (Diameter bound).* The conformal metric $\tilde{g} = \Omega^2 g$ has geodesics satisfying:

$$\tilde{d}(x, y) = \int_\gamma \Omega \, ds_g$$

Since $\int_0^\infty \Omega(r) dr < \infty$ for suitable $\Omega$, the diameter is finite.

*Step 3 (Boundary addition).* The conformal boundary $\partial_\Omega M$ represents "points at infinity." In the compactified manifold $\bar{M} = M \cup \partial_\Omega M$:
- Null infinity $\mathscr{I}^{\pm}$ for Minkowski space
- Conformal boundary for hyperbolic space
- Point at infinity for Euclidean space

*Step 4 (Trajectory control).* A trajectory $\gamma(t) \to \infty$ in $(M, g)$ satisfies:

$$\tilde{d}(\gamma(0), \gamma(t)) \leq \int_0^t \Omega(\gamma(s)) |\dot{\gamma}(s)|_g \, ds < \infty$$

The trajectory reaches $\partial_\Omega M$ in finite $\tilde{g}$-time, preventing "escape to infinity."
:::
