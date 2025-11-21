I’ll treat this like writing an internal “systems design doc” for your proof: what the components are, what each one covers, where each one *fails*, and who takes over when it does.

I’ll use your own terminology (Fractal, Type II, HighSwirl, BarberPole, Tube, efficiency (\Xi), extremizer manifold (\mathcal M), etc.) and try to be brutally explicit about overlaps and hand‑offs.

---

# 0. Big picture: what we’re trying to guarantee

You want the following meta‑statement to be true:

> For *every* putative finite‑time singularity, when you pass to a renormalized limit profile (V_\infty), there is **at least one** mechanism that rules it out. No single mechanism needs to work everywhere, but the union of their validity regions must cover *all* singular scenarios.

So the goal is to understand:

1. A **parameterization** of the “space of possible blow‑up behaviours” (renormalized profiles and their dynamics).
2. For each mechanism:

   * its **domain of validity** (where it works),
   * its **failure modes** (where it definitely does *not* apply), and
   * **what another mechanism must assume** to take over at those failure boundaries.
3. Wherever two mechanisms overlap, you get redundancy; where they don’t overlap, you have to be very sure a third one covers the gap.

---

# 1. The “phase space” of mechanisms

Let’s think of a renormalized profile (V) (or its trajectory (V(s))) as being described by a small set of *structural parameters*:

1. **Distance to extremizers:**
   [
   \delta(V) := \mathrm{dist}*{H^1*\rho}(V, \mathcal M)
   ]
   where (\mathcal M) is the extremizer manifold of your efficiency functional (\Xi).

2. **Swirl ratio / swirl amplitude:**
   some normalized measure (\mathcal S(V)) of azimuthal velocity / circulation, especially near the axis and within the core region.

3. **Twist / barber‑pole parameter:**
   a measure (\mathcal T(V)) of how strongly vortex lines wind and twist around the core (your twist functional (Z), or something equivalent).

4. **Scaling speed / type:**
   encoded by (\lambda(t)) and its behaviour near blow‑up:
   [
   \lambda(t) \sim (T^*-t)^\alpha \quad (\text{heuristically})
   ]
   with (\alpha = 1/2) (Type I), (\alpha>1/2) (Type II), etc.

5. **Regularity / entropy:**

   * analytic radius (\tau(s)) in the Gevrey framework,
   * fractal dimension of support / spectral entropy,
   * etc.

Very roughly, your classification can be thought of in this schematic “phase diagram”:

* **Far from extremizers**: (\delta(V)\ge \delta_0) → “Fractal / high‑entropy” regime.
* **Near extremizers**: (\delta(V)<\delta_0) → “Coherent / tube‑like” regime.
* Within the coherent regime:

  * **High swirl**: (\mathcal S(V)\ge S_0),
  * **Low swirl, high twist**: (\mathcal S(V)<S_0), (\mathcal T(V)\ge T_0) → BarberPole,
  * **Low swirl, low twist**: (\mathcal S(V)<S_0), (\mathcal T(V)<T_0) → Tube class.

And orthogonal to all of that: the **scaling exponent** (\alpha) (Type I vs Type II) and the **regularity/entropy** (fractal vs analytic).

---

# 2. Mechanism 1 – Variational efficiency & extremizer manifold

### 2.1 What it does

* You define an efficiency functional (\Xi[V]) (essentially a rescaled (|\langle B(V,V),AV\rangle|) on the (\dot H^1) unit sphere).
* Extremizers (\mathcal M) maximize (\Xi).
* You assume / aim to show a **quantitative stability** of Bianchi–Egnell type:
  [
  \Xi_{\max} - \Xi[V] ;\ge; \kappa,\delta(V)^2
  ]
  at least when (\delta(V)) is not too large.
* Mechanism: if a would‑be singular profile is **not close to** the extremizers (i.e. (\delta(V)\ge\delta_0)), it suffers an efficiency *deficit* that cannot be compensated by the geometric growth mechanisms needed for blow‑up. This feeds into Gevrey smoothing and transit‑cost to force “Fractal Exclusion”.

### 2.2 Domain where it is strong

* Profiles with **large distance** to (\mathcal M): (\delta(V)\ge\delta_0).
* Profiles with **moderate swirl** and **no crazy degeneracies** (so that the linearization around (\mathcal M) behaves like your assumed Hessian).

You use this to power:

* **Fractal exclusion**: fractal/high‑entropy profiles are far from (\mathcal M) and therefore inefficient.
* **Kink / transitional exclusion**: local kinks yield a boost in twist that translates into a positive distance from (\mathcal M) and hence a quadratic efficiency penalty.

### 2.3 How it can fail

This mechanism can fail (or be unavailable) when:

1. **No extremizers exist or are not smooth** (H1 false):
   (\mathcal M) might not be a smooth manifold of rapidly decaying profiles. The supremum of (\Xi) may not be attained; the set of quasi‑maximizers might be wild.

2. **No quantitative stability** (H2–H4 false):
   Even if extremizers exist, you might fail to prove the Bianchi–Egnell style stability: the second variation might have zero eigenvalues or pathological degeneracies, so you don’t get a clean quadratic gap.

3. **Profile is very close to (\mathcal M)**:
   If (\delta(V)\ll\delta_0), the variational gap is small and the variational mechanism alone doesn’t prevent extreme dynamics – you’re in the coherent regime where other mechanisms must take over (high swirl, twist, tube defocusing).

### 2.4 Who takes over when it fails?

* **Case A – Extremizers don’t exist / H1–H5 fail globally:**
  Then the “Fractal exclusion” for large (\delta) via (\Xi) may collapse. In your architecture, the **Gevrey smoothing + transit‑cost** and the **Type II mass‑flux capacity** are supposed to handle high‑entropy/fractal profiles **without** relying on (\Xi).

* **Case B – Extremizers exist but (\delta(V)\to 0):**
  You are in the **coherent regime**. The blow‑up profile is tube‑like, and then the classification by swirl/twist kicks in:

  * High swirl → **HighSwirl spectral coercivity**.
  * Low swirl but high twist → **BarberPole / twist–smoothness incompatibility**.
  * Low swirl & low twist → **Tube / geometric defocusing**.

So: the variational mechanism is designed to **kill high‑entropy states that stay away from (\mathcal M)**; if a profile moves towards (\mathcal M), variational penalties disappear but you *want* that, because then *other* mechanisms (spectral / geometric) become available.

---

# 3. Mechanism 2 – Gevrey smoothing & transit‑cost (dynamic chameleon exclusion)

### 3.1 What it does

* You work in a Gevrey framework with an analyticity radius (\tau(s)) obeying a differential inequality of the form
  [
  \dot\tau(s) \ge \nu - C_{Sob},|V|_{\tau,1},\Xi[V] ;,
  ]
  and then refine to (\dot\tau(s)\gtrsim \delta(s)^2) in certain regimes, where (\delta(s)=\mathrm{dist}(V(s),\mathcal M)).
* Mechanism: being far from the extremizers (large (\delta)) forces **growth** of (\tau) (more analyticity). That means a “fractal” or rough phase can’t persist indefinitely: it smooths out and you incur a **transit cost** in time to move between rough and smooth states.
* You use this to exclude **Dynamic Chameleon** scenarios: infinitely many transitions between fractal and coherent regimes in finite renormalized time.

### 3.2 Domain where it is strong

* When:

  * you can guarantee a reasonable **upper bound** on (|V|_{\tau,1}) in terms of (\delta) and global energy, and
  * the efficiency gap (\Xi_{\max}-\Xi[V]) is positive on the region you call “fractal” ((\delta\ge\delta_0)).

Then this mechanism is powerful in:

* **Fractal/high‑entropy regimes** (far from (\mathcal M)).
* **Transitions** between fractal and coherent.

It also provides protection against scenarios like:

* “The solution slips into a fractal regime just *fast enough* before blow‑up to still create a singularity” – the transit cost says you can’t slip *too* fast.

### 3.3 How it can fail

1. **Analytic/Gevrey framework breaks down**:
   If you can’t actually establish uniform Gevrey bounds near blow‑up, or the inequality for (\dot\tau) fails (e.g. (|V|_{\tau,1}) blows up faster than you can control), the transit‑cost mechanism is compromised.

2. **Efficiency gap not strong enough**:
   If (\Xi[V]) is not sufficiently below (\Xi_{\max}) in the fractal region, the right‑hand side of (\dot\tau) can lose its positivity; the connection “far from (\mathcal M) ⇒ \dot\tau>0” can fail.

3. **Profile stays close to (\mathcal M) most of the time**:
   If (\delta(s)) is small for most (s), then the Gevrey mechanism says little; this is exactly the coherent/tube region.

### 3.4 Who takes over when it fails?

* **If Gevrey fails because the solution stays very regular/analytic but near (\mathcal M):**

  * Then you’re in the **coherent** tube regime and the **spectral/geometric mechanisms (HighSwirl, Barber, Tube)** should take over.

* **If Gevrey fails because you cannot establish the necessary bounds on (|V|_{\tau,1})** in a rough regime:

  * Then your **Type II / mass‑flux capacity** and **variational efficiency** arguments in the fractal region are supposed to step in:

    * Rough + fast focusing → Type II energy contradiction.
    * Rough + not focusing too fast → global energy / dispersion prevents blow‑up.

In other words, Gevrey is meant to **protect dynamic transitions**; when it doesn’t apply, you still have static structural classification via (\delta,\mathcal S,\mathcal T) and the other mechanisms.

---

# 4. Mechanism 3 – Type II / mass‑flux capacity & energy

### 4.1 What it does

* You try to rule out **fast‑scaling blow‑up** where the rescaling factor behaves like (\lambda(t)\ll\sqrt{T^*-t}) (“Type II”), by showing that such scaling would force an **infinite energy dissipation** as (t\to T^*), contradicting the Leray energy inequality.

* Rough idea:

  * Enstrophy scales like (\sim 1/\lambda(t)).
  * Total dissipation up to (T^*):
    [
    \int_0^{T^*}|\nabla u(t)|_{L^2}^2,dt \sim \int_0^{T^*} \frac{dt}{\lambda(t)}.
    ]
  * If (\lambda(t)) decays faster than some power, that integral blows up, violating the global bound on dissipation.

* Additionally, you have **mass‑flux capacity** arguments: to concentrate vorticity in a shrinking core without violating incompressibility, you need a certain flux structure which becomes impossible under the constraints on (u).

### 4.2 Domain where it is strong

* Profiles with **rapid focusing** ((\alpha \ge 1) in (\lambda(t)\sim (T^*-t)^\alpha), once the scaling error is fixed).
* Scenarios where blow‑up is driven solely by the interplay of:

  * advective focusing,
  * viscous dissipation,
  * and global energy bounds.

Type II mechanism is particularly aimed at **coherent but extremely fast** focusing behaviours.

### 4.3 How it can fail

1. **Scaling not fast enough**:

   * If (\lambda(t)\sim (T^*-t)^\alpha) with (1/2<\alpha<1), the integral (\int_0^{T^*} 1/\lambda(t),dt) still converges.
   * So the energy contradiction may fail; Type II mechanism does *not* automatically exclude these intermediate rates.

2. **Blow‑up is not purely driven by scaling**:

   * If the blow‑up scenario is dominated by spectral/variational/geometric mechanisms rather than brute scaling, the Type II argument may not directly see it.

3. **Mass‑flux capacity estimates may be too weak or rely on unproved geometric hypotheses**.

### 4.4 Who takes over when it fails?

* **If scaling is moderate (Type I or borderline Type II):**

  * Then the **geometric/variational mechanisms** (Fractal/HighSwirl/Barber/Tube) must take over. These mechanisms are intended to control the *shape* and *structure* of the core, not just its scale.

* **If the mass‑flux capacity argument fails in some exotic geometry:**

  * Then that geometry should be caught either by the **Fractal mechanism** (if it is high entropy) or the **Tube/Barber** classification (if it’s smooth and coherent).

So Type II is your “speed police” on scaling; if it fails (because speed is not extreme enough), other mechanisms, which care about structure not speed, should constrain the dynamics.

---

# 5. Mechanism 4 – HighSwirl spectral coercivity

### 5.1 What it does

* You linearize the renormalized NS around a highly swirling core and derive an operator with an effective potential that grows like (\sigma^2/r^2) (where (\sigma) is a swirl parameter).
* With a Hardy‑type inequality and Gaussian weights, you claim a spectral gap (\mu(\sigma)>0) for large swirl, and build a Lyapunov functional (\mathcal E(s)) such that:
  [
  \frac{d}{ds}\mathcal E(s) \le -\mu(\sigma),\mathcal E(s).
  ]
* Mechanism: in the **HighSwirl** regime ((\mathcal S(V)\ge S_0)), perturbations around the swirling base profile decay exponentially in renormalized time. Nontrivial high‑swirl stationary profiles are impossible; high swirl inevitably relaxes.

### 5.2 Domain where it is strong

* Profiles with:

  * sufficiently large swirl ratio (\mathcal S(V)\ge S_0), and
  * sufficiently nice structure (smoothness, non‑degenerate core) so the linearization and spectral theory apply.

* With the **axis swirl positivity** lemma (your Harnack/Cartesian fix), you guarantee that near the axis you have (V_\theta(r)\gtrsim r) => a strong swirl baseline.

### 5.3 How it can fail

1. **Swirl not large enough**:

   * If (\mathcal S(V)) is below the threshold (S_0), the spectral gap may vanish or be too small to force decay.

2. **Spectral gap unproved / false**:

   * The Hardy‑Rellich inequality and the spectral analysis might not actually give a uniform (\mu(\sigma)>0) for all high swirl states.

3. **Profile not close enough to a swirling eigenprofile**:

   * If the actual blow‑up profile is not well approximated by the swirling base profile that your spectral analysis uses, then the linear argument might not be applicable.

### 5.4 Who takes over when it fails?

* **If swirl is small** ((\mathcal S(V)<S_0)):
  You’re in the **low‑swirl coherent regime**, so either:

  * If twist (\mathcal T(V)) is large → **BarberPole twist mechanism**.
  * If twist is small → **Tube geometric defocusing**.

* **If the spectral gap is uncertain**:
  Then you rely more on:

  * the **geometric defocusing** of tubes (if swirl isn’t too big),
  * or the **variational / Gevrey** control in high‑entropy regions.

So HighSwirl is your “helical‑core decay” mechanism; where swirl is not large or spectral information is weak, *structure* (Barber/Tube) and *variational* mechanisms handle the coherent core.

---

# 6. Mechanism 5 – BarberPole twist & smoothness‑twist incompatibility

### 6.1 What it does

* BarberPole regime: low global swirl but **high twist** of vortex lines; think of a long slender filament with vortex lines wrapping around it many times.

* You use:

  * uniform smoothness and bounded derivatives (from being close to (\mathcal M)),
  * twist functionals like (Z=\int |\omega|^2 |\nabla\xi|^2),
  * nodal set analysis (twist can only be large if vortex lines approach nodal regions), and
  * a “Smoothness–Twist incompatibility” theorem: smooth bounded profiles can’t sustain arbitrarily large twist without incurring a large efficiency penalty / violating regularity.

* Mechanism: high twist in a smooth coherent core forces either:

  * a **variational cost** (distance from (\mathcal M), efficiency deficit), or
  * a **regularity breakdown** inconsistent with the rest of the setup (Gevrey regularity, bounded derivatives).

### 6.2 Domain where it is strong

* Coherent profiles:

  * close to extremizers ((\delta(V)) small),
  * low swirl ((\mathcal S(V)) below threshold),
  * **high twist** ((\mathcal T(V)\ge T_0)).

* Here the **variational structure** (H1–H5) gives you the smoothness & bounded derivatives, which you then leverage against large twist.

### 6.3 How it can fail

1. **Twist is not large**:

   * If (\mathcal T(V)<T_0), you’re not in BarberPole; you can’t use the twist arguments.

2. **Coherence / smoothness fails**:

   * If the profile is not close to (\mathcal M), or if you can’t use extremizer regularity, you might not have the uniform bounds on derivatives needed to rule out high twist.

3. **Nodal set analysis incomplete**:

   * If the claim that twist can only concentrate near nodal sets, and nodal sets contribute negligibly to functionals, is not proved rigorously, the BarberPole mechanism weakens.

### 6.4 Who takes over when it fails?

* **If twist is low** ((\mathcal T(V)<T_0)):
  Then by your classification the profile falls into the **Tube class** → **geometric defocusing mechanism** handles it.

* **If smoothness / closeness to extremizers fails**:

  * Then you are likely in a **fractal / high‑entropy** regime or transitional regime where:

    * **Variational efficiency** and **Gevrey transit‑cost** (Mechanisms 1 & 2) apply, or
    * you are in a rough scenario where **Type II / mass‑flux capacity** restricts focusing.

So BarberPole is your “coherent, low‑swirl but high‑twist” killer; if either coherence or high twist disappears, you move into Fractal or Tube and get caught there.

---

# 7. Mechanism 6 – Tube class & geometric defocusing

### 7.1 What it does

* Tube class: profiles that are:

  * near extremizers ((\delta(V)) small),
  * **low swirl** ((\mathcal S(V)<S_0)),
  * **low twist** ((\mathcal T(V)<T_0));
    basically, “nice,” nearly straight, slender vortex tubes.

* Mechanism: you claim a **geometric defocusing** phenomenon:
  slender, nearly straight vortex tubes in Navier–Stokes cannot produce finite‑time blow‑up because:

  * the nonlinearity is effectively transport along the tube,
  * curvature/torsion are small, and
  * viscous diffusion in the cross‑section bleeds energy faster than focusing along the tube can create singularities.

* This is codified in something like Theorem 4.6: “Tube class singular profiles are impossible”.

### 7.2 Domain where it is strong

* Coherent region with:

  * (\delta(V)) small,
  * (\mathcal S(V)<S_0),
  * (\mathcal T(V)<T_0),
  * and satisfying the geometric assumptions (bounded curvature, no pinching, etc.).

This is meant to be the “final box”: if you’ve evaded Fractal, Type II, HighSwirl, and BarberPole, you must live in Tube; and Tube is killed by defocusing.

### 7.3 How it can fail

1. **Tube geometry doesn’t hold**:

   * If the actual geometry is more complicated (e.g., multiple interacting tubes with strong local curvature) than your Tube assumptions, the defocusing theorem might not apply.

2. **Extremizer closeness fails**:

   * If (\delta(V)) is not small enough, you might not be in the tube class; you may revert to fractal / transition regimes where Tube arguments no longer hold.

3. **Viscosity and transport interplay more subtly than assumed**:

   * The defocusing result may require strong smallness assumptions (curvature, cross‑sectional variation, etc.) that are not clearly guaranteed by the rest of the framework.

### 7.4 Who takes over when it fails?

* **If tube assumptions fail because twist or curvature grows**:

  * Then either:

    * twist pushes you into **BarberPole** (if still coherent), or
    * high entropy / irregular geometry pushes you into **Fractal** or **Type II** regimes.

* **If tube assumptions fail because you’re not “near (\mathcal M)”**:

  * Then you’re back in the **variational / Gevrey** regimes, where efficiency deficits and smoothing kick in.

So Tube is your “nice, boring, coherent” regime; if coherence is broken or geometric complexity increases, you are deliberately thrown back into other boxes that have stronger mechanisms.

---

# 8. Mechanism 7 – Compactness, rigidity, and limit profiles

This isn’t a “killer” in itself but is the **glue**:

* You use blow‑up rescaling + compactness (in (H^1_\rho)) to get limit profiles (V_\infty).
* You show (under assumptions) that any such limit profile is:

  * stationary for the renormalized equation, and
  * lies in your phase space (\Omega\subset H^1_\rho).

This lets you reduce:

> “Exclude all finite‑time blow‑ups”
> ⇔ “Show there are no singular limit profiles (V_\infty\in\Omega_{\text{sing}})”.

Then you partition (\Omega) into:

* Fractal / Type II / HighSwirl / BarberPole / Tube,

and argue each piece has (\Omega_{\text{sing}}\cap(\text{that piece})=\emptyset).

This is the *structural backbone* that lets all the other mechanisms compose logically.

---

# 9. Synergy: how the mechanisms cover each other

Now let’s make the interplay super explicit. Think of any potential renormalized limit profile (V_\infty). We run the following **decision tree**:

### Step 1: Coherence (distance to (\mathcal M))

* If (\delta(V_\infty) \ge \delta_0):
  → **Fractal / far‑from extremizers regime**.
  Here:

  * **Variational efficiency** gives (\Xi[V]\le \Xi_{\max} - \kappa\delta_0^2).
  * Combined with **Gevrey transit‑cost**, dynamic fractal behaviour is penalized.
  * For extremely focusing profiles in this regime, **Type II / mass‑flux capacity** yields contradictions.

* If (\delta(V_\infty) < \delta_0):
  → **Coherent regime**.
  You now ignore variational penalties and pass to swirl/twist classification.

### Step 2: Swirl threshold

Within the coherent regime:

* If (\mathcal S(V_\infty)\ge S_0):
  → **HighSwirl**.

  * **HighSwirl spectral coercivity** and the new **axis Harnack lemma** give exponential decay of perturbations; no nontrivial singular high‑swirl stationary profile.

* If (\mathcal S(V_\infty)< S_0):
  → **Low swirl**.
  Now move to twist.

### Step 3: Twist threshold

Within the coherent, low‑swirl regime:

* If (\mathcal T(V_\infty)\ge T_0):
  → **BarberPole**.

  * **Smoothness–twist incompatibility** and nodal‑set analysis (assuming extremizer regularity) exclude blow‑up.

* If (\mathcal T(V_\infty)< T_0):
  → **Tube class**.

  * **Geometric defocusing** + viscosity excludes blow‑up.

### Step 4: Scaling / Type II overlay

Orthogonally:

* If (\lambda(t)) decays *too fast* (true Type II with (\alpha) large enough):

  * The **Type II energy flux** mechanism (once corrected) rules out blow‑up.

* If scaling stays moderate (Type I or mild Type II):

  * The classification above (Fractal vs Coherent / HighSwirl / Barber / Tube) covers the structural behaviour.

---

# 10. Where mechanisms *don’t* overlap (gaps to watch)

You asked specifically for where each mechanism fails and who takes over. Here are the main potential **gaps**:

1. **Far‑from extremizers, slow scaling, but Gevrey/efficiency too weak**

   * Scenario: (\delta\ge\delta_0), but the efficiency gap is small, or (|V|_{\tau,1}) is large enough that (\dot\tau) loses positivity; scaling not Type II enough to trigger energy blow‑up.
   * Intended cover: you want variational + Gevrey + global energy to still prevent blow‑up, but this regime may be technically fragile.

2. **Near extremizers but not really tube‑like (geometry messy)**

   * Scenario: (\delta<\delta_0), but the geometry is complicated: multiple interacting tubes, moderate swirl, moderate twist – not clearly in HighSwirl, Barber, or Tube.
   * Intended cover: either twist grows (→ Barber), or irregular complexity pushes you into Fractal/Type II. This is plausible but not fully formalized.

3. **Intermediate Type II scaling with coherent structure**

   * Scenario: (\alpha\in(1/2,1)) for (\lambda(t)), coherent core, near extremizers, neither fractal nor highly swirling nor high‑twist.
   * Intended cover: Tube defocusing + global energy. But this is exactly where the corrected Type II mechanism is weaker, so you must lean heavily on the Tube geometry.

4. **Failure of H1–H5 + failure of spectral/geometry assumptions simultaneously**

   * If extremizers don’t exist *and* high‑swirl spectral gaps fail *and* geometric defocusing is weaker than expected, then redundancy breaks.
   * You’d need some explicit statement that “we are *not* assuming all these properties independently; if one fails, it necessarily forces the profile into another regime where some other hypothesis is easier to prove.”

These are the places where, if I were “hostile reviewer,” I’d probe hardest to see if a scenario can slip through all nets.

---

# 11. How to use this document

What you can do with this “synergy map”:

1. **Turn each mechanism into a clearly labeled theorem with explicit hypothesis region.**
   For example:

   * Thm A (Fractal exclusion): *Assume H1–H5 and Gevrey inequality (X). If (\delta(V)\ge\delta_0) for all large (s), then no singular blow‑up occurs.*
   * Thm B (HighSwirl exclusion): *Assume swirl ratio (\mathcal S\ge S_0), axis positivity lemma, and spectral gap hypothesis (Y). Then no singular high‑swirl blow‑up.*
     etc.

2. **Introduce a “Classification Lemma”** that for any singular limit profile (V_\infty) either:

   * (i) (\delta(V_\infty)\ge\delta_0) or
   * (ii) (\delta(V_\infty)<\delta_0) and (\mathcal S\ge S_0) or
   * (iii) (\delta(V_\infty)<\delta_0), (\mathcal S<S_0), (\mathcal T\ge T_0), or
   * (iv) (\delta(V_\infty)<\delta_0), (\mathcal S<S_0), (\mathcal T<T_0).

   That lemma is the backbone that ties your partition to the mechanisms.

3. **For each failure mode, add a “handoff remark” in the text**:

   > *Remark.* If condition [H1] fails, then the blow‑up profile cannot remain in the coherent near‑extremizer regime. In that case, by Lemma Z, the profile enters either the fractal or Type II regime, where Theorems A and C apply.

4. **Explicitly verify that the union of all hypothesis regions covers the ‘singular set’.**
   It’s ok if some areas are covered by multiple mechanisms; what you *must* avoid is a hole where none apply.

---

If you want, I can next write this up as a **formal section** you can paste near your Chapter 12 (like “12.x Mechanism Synergy and Coverage”), with theorem/lemma names and cross‑references tailored to your draft structure.
