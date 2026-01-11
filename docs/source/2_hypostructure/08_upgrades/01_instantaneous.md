---
title: "Instantaneous Upgrade Metatheorems"
---

# Part XIV: Instantaneous Upgrade Metatheorems

(sec-instantaneous-certificate-upgrades)=
## Instantaneous Certificate Upgrades

The **Instantaneous Upgrade Metatheorems** formalize the logical principle that a "Blocked" barrier certificate or a "Surgery" re-entry certificate can be promoted to a full **YES** (or **YES$^\sim$**) permit under appropriate structural conditions. These upgrades occur *within* a single Sieve pass when the blocking condition itself implies a stronger regularity guarantee.

**Logical Form:** $K_{\text{Node}}^- \wedge K_{\text{Barrier}}^{\mathrm{blk}} \Rightarrow K_{\text{Node}}^{\sim}$

The key insight is that certain "obstructions" are themselves *certificates of regularity* when viewed from the correct perspective.

---

(sec-saturation-promotion)=
### Saturation Promotion

:::{prf:theorem} [UP-Saturation] Saturation Promotion (BarrierSat $\to$ YES$^\sim$)
:label: mt-up-saturation
:class: metatheorem rigor-class-l

**Rigor Class:** L (Literature-Anchored) — see {prf:ref}`def-rigor-classification`

**Bridge Verification:**
1. *Hypothesis Translation:* Certificate $K_{\text{sat}}^{\mathrm{blk}}$ implies Foster-Lyapunov drift condition: $\mathcal{L}\Phi(x) \leq -\lambda\Phi(x) + b$ with compact sublevel sets
2. *Domain Embedding:* $\iota: \mathbf{Hypo}_T \to \mathbf{Markov}$ mapping to continuous-time Markov process on Polish state space
3. *Conclusion Import:* Meyn-Tweedie Theorem 15.0.1 {cite}`MeynTweedie93` $\Rightarrow K_{D_E}^{\sim}$ (finite energy under invariant measure $\pi$)

**Context:** Node 1 (EnergyCheck) fails ($E = \infty$), but BarrierSat is Blocked ($K_{\text{sat}}^{\mathrm{blk}}$).

**Hypotheses.** Let $\mathcal{H} = (\mathcal{X}, \Phi, \mathfrak{D}, G)$ be a Hypostructure with:
1. A height functional $\Phi: \mathcal{X} \to [0, \infty]$ that is unbounded ($\sup_x \Phi(x) = \infty$)
2. A dissipation functional $\mathfrak{D}$ satisfying the drift condition: there exist $\lambda > 0$ and $b < \infty$ such that
   $$\mathcal{L}\Phi(x) \leq -\lambda \Phi(x) + b \quad \text{for all } x \in \mathcal{X}$$
   where $\mathcal{L}$ is the infinitesimal generator of the dynamics.
3. A compact sublevel set $\{x : \Phi(x) \leq c\}$ for some $c > b/\lambda$.

**Statement:** Under the drift condition, the process admits a unique invariant probability measure $\pi$ with $\int \Phi \, d\pi < \infty$. The system is equivalent to one with bounded energy under the renormalized measure $\pi$.

**Certificate Logic:**
$$K_{D_E}^- \wedge K_{\text{sat}}^{\mathrm{blk}} \Rightarrow K_{D_E}^{\sim}$$

**Interface Permit Validated:** Finite Energy (renormalized measure).

**Literature:** {cite}`MeynTweedie93`; {cite}`HairerMattingly11`
:::

:::{prf:proof} Proof Sketch
:label: sketch-mt-up-saturation

The drift condition implies geometric ergodicity by the Foster-Lyapunov criterion (Meyn and Tweedie, 1993, Theorem 15.0.1). The invariant measure $\pi$ satisfies $\pi(\Phi) < \infty$ by Theorem 14.0.1 of the same reference. The renormalized height $\hat{\Phi} = \Phi - \pi(\Phi)$ is centered and the dynamics converge exponentially to equilibrium.
:::

---

(sec-causal-censor-promotion)=
### Causal Censor Promotion

:::{prf:theorem} [UP-Censorship] Causal Censor Promotion (BarrierCausal $\to$ YES$^\sim$)
:label: mt-up-censorship
:class: metatheorem

**Context:** Node 2 (ZenoCheck) fails ($N \to \infty$), but BarrierCausal is Blocked ($K_{\mathrm{Rec}_N}^{\mathrm{blk}}$).

**Hypotheses.** Let $\mathcal{H}$ be a Hypostructure with:
1. An event counting functional $N: \mathcal{X} \times [0,T] \to \mathbb{N} \cup \{\infty\}$
2. A singularity requiring infinite computational depth to resolve: the Cauchy development $D^+(S)$ is globally hyperbolic but $N(x, T) \to \infty$ as $x \to \Sigma$
3. A cosmic censorship condition: the singular set $\Sigma$ is contained in the future boundary $\mathcal{I}^+ \cup i^+$ of conformally compactified spacetime.

**Statement:** If the singularity is hidden behind an event horizon or lies at future null/timelike infinity, it is causally inaccessible to any physical observer. The event count is finite relative to any observer worldline $\gamma$ with finite proper time.

**Certificate Logic:**
$$K_{\mathrm{Rec}_N}^- \wedge K_{\mathrm{Rec}_N}^{\mathrm{blk}} \Rightarrow K_{\mathrm{Rec}_N}^{\sim}$$

**Interface Permit Validated:** Finite Event Count (physically observable).

**Literature:** {cite}`Penrose69`; {cite}`ChristodoulouKlainerman93`; {cite}`HawkingPenrose70`
:::

:::{prf:proof} Proof Sketch
:label: sketch-mt-up-censorship

By the weak cosmic censorship conjecture (Penrose, 1969), generic gravitational collapse produces singularities cloaked by event horizons. The Hawking-Penrose theorems (1970) establish geodesic incompleteness, but the Christodoulou-Klainerman stability theorem (1993) ensures the exterior remains regular. Any observer worldline $\gamma \subset J^-(\mathcal{I}^+)$ experiences finite proper time and finite events before the singularity becomes causally relevant.
:::

---

(sec-scattering-promotion)=
### Scattering Promotion

:::{prf:theorem} [UP-Scattering] Scattering Promotion (BarrierScat $\to$ VICTORY)
:label: mt-up-scattering
:class: metatheorem rigor-class-l

**Rigor Class:** L (Literature-Anchored) — see {prf:ref}`def-rigor-classification`

**Bridge Verification:**
1. *Hypothesis Translation:* Certificate $K_{C_\mu}^{\mathrm{ben}}$ implies: (a) finite Morawetz quantity $\int_0^\infty \int |x|^{-1}|u|^{p+1} < \infty$, (b) no concentration sequence
2. *Domain Embedding:* $\iota: \mathbf{Hypo}_T \to H^1(\mathbb{R}^n)$ for dispersive NLS/NLW with critical Sobolev exponent
3. *Conclusion Import:* Morawetz {cite}`Morawetz68` + Strichartz + Kenig-Merle rigidity {cite}`KenigMerle06` $\Rightarrow$ Global Regularity (scattering to linear solution)

**Context:** Node 3 (CompactCheck) fails (No concentration), but BarrierScat indicates Benign ($K_{C_\mu}^{\mathrm{ben}}$).

**Hypotheses.** Let $\mathcal{H}$ be a Hypostructure of type $T_{\text{dispersive}}$ with:
1. A dispersive evolution $u(t)$ satisfying a nonlinear wave or Schrödinger equation
2. The concentration-compactness dichotomy: either $\mu(V) > 0$ for some profile $V$, or dispersion dominates
3. A finite Morawetz quantity: $\int_0^\infty \int_{\mathbb{R}^n} |x|^{-1} |u|^{p+1} \, dx \, dt < \infty$

**Statement:** If energy disperses (no concentration) and the interaction functional is finite (Morawetz bound), the solution scatters to a free linear state: there exists $u_\pm \in H^1$ such that $\|u(t) - e^{it\Delta}u_\pm\|_{H^1} \to 0$ as $t \to \pm\infty$. This is a "Victory" condition equivalent to global existence and regularity.

**Certificate Logic:**
$$K_{C_\mu}^- \wedge K_{C_\mu}^{\mathrm{ben}} \Rightarrow \text{Global Regularity}$$

**Interface Permit Validated:** Global Existence (via dispersion).

**Literature:** {cite}`Morawetz68` (interaction estimate); {cite}`Strichartz77`; {cite}`KeelTao98` Thm.1.2 (Strichartz); {cite}`KenigMerle06` Thm.1.1 (rigidity); {cite}`KillipVisan10` (NLS scattering)
:::

:::{prf:proof}
:label: sketch-mt-up-scattering

*Step 1 (Morawetz Spacetime Bound).* The **Morawetz estimate** ({cite}`Morawetz68`) provides spacetime integrability:
$$\int_0^\infty \int_{\mathbb{R}^n} \frac{|u(t,x)|^{p+1}}{|x|} \, dx \, dt \leq C \cdot E[u_0]$$
This "spacetime Lebesgue norm" is finite for solutions with bounded energy, ruling out mass concentration at the origin over long times.

*Step 2 (Strichartz Estimates).* The **Strichartz estimates** ({cite}`Strichartz77`; {cite}`KeelTao98` Theorem 1.2) provide:
$$\|e^{it\Delta} u_0\|_{L^q_t L^r_x} \leq C \|u_0\|_{L^2}$$
for admissible pairs $(q, r)$ satisfying $\frac{2}{q} + \frac{n}{r} = \frac{n}{2}$. These estimates control the spacetime norm of solutions in terms of initial data, enabling the perturbative argument below.

*Step 3 (Concentration-Compactness Rigidity).* By the **Kenig-Merle methodology** ({cite}`KenigMerle06` Theorem 1.1), if $K_{C_\mu}^- = \text{NO}$ (no concentration), then either:
- (a) Solution scatters: $\|u(t) - e^{it\Delta} u_\pm\|_{H^1} \to 0$, or
- (b) A critical element exists with zero Morawetz norm—but this contradicts the Morawetz bound from Step 1.

*Step 4 (Scattering Construction).* The limiting profile $u_\pm$ is constructed via the **Cook method**: define $u_\pm := \lim_{t \to \pm\infty} e^{-it\Delta} u(t)$. The limit exists in $H^1$ by Strichartz + Morawetz: the integral $\int_0^\infty \|N(u)\|_{L^{r'}_x} dt$ is finite, where $N(u)$ is the nonlinearity.
:::

---

(sec-type-ii-suppression-promotion)=
### Type II Suppression Promotion

:::{prf:theorem} [UP-TypeII] Type II Suppression (BarrierTypeII $\to$ YES$^\sim$)
:label: mt-up-type-ii
:class: metatheorem

**Context:** Node 4 (ScaleCheck) fails (Supercritical), but BarrierTypeII is Blocked ($K_{\mathrm{SC}_\lambda}^{\mathrm{blk}}$).

**Hypotheses.** Let $\mathcal{H}$ be a Hypostructure with:
1. A supercritical scaling exponent $\alpha > \alpha_c$ (energy-supercritical regime)
2. A Type II blow-up scenario where the solution concentrates at a point with unbounded $L^\infty$ norm but bounded energy
3. An energy monotonicity formula $\frac{d}{dt}\mathcal{E}_\lambda(t) \leq 0$ for the localized energy at scale $\lambda$

**Statement:** If the renormalization cost $\int_0^{T^*} \lambda(t)^{-\gamma} \, dt = \infty$ diverges logarithmically, the supercritical singularity is suppressed and cannot form in finite time. The blow-up rate satisfies $\lambda(t) \geq c(T^* - t)^{1/\gamma}$ for some $\gamma > 0$.

**Certificate Logic:**
$$K_{\mathrm{SC}_\lambda}^- \wedge K_{\mathrm{SC}_\lambda}^{\mathrm{blk}} \Rightarrow K_{\mathrm{SC}_\lambda}^{\sim}$$

**Interface Permit Validated:** Subcritical Scaling (effective).

**Literature:** {cite}`MerleZaag98`; {cite}`RaphaelSzeftel11`; {cite}`CollotMerleRaphael17`
:::

:::{prf:proof}
:label: sketch-mt-up-type-ii

The monotonicity formula (Merle and Zaag, 1998) bounds the blow-up rate from below. For Type II blow-up, the energy remains bounded while the scale $\lambda(t) \to 0$. The logarithmic divergence of the renormalization integral creates an energy barrier that prevents finite-time singularity formation. This mechanism underlies the Raphaël-Szeftel soliton resolution (2011).
:::

---

(sec-capacity-promotion)=
### Capacity Promotion

:::{prf:theorem} [UP-Capacity] Capacity Promotion (BarrierCap $\to$ YES$^\sim$)
:label: mt-up-capacity
:class: metatheorem

**Context:** Node 6 (GeomCheck) fails (Codim too small), but BarrierCap is Blocked ($K_{\mathrm{Cap}_H}^{\mathrm{blk}}$).

**Hypotheses.** Let $\mathcal{H}$ be a Hypostructure with:
1. A singular set $\Sigma \subset \mathcal{X}$ with Hausdorff dimension $\dim_H(\Sigma) \geq n-2$ (marginal codimension)
2. A capacity bound: $\mathrm{Cap}_{1,2}(\Sigma) = 0$ where $\mathrm{Cap}_{1,2}$ is the $(1,2)$-capacity (Sobolev capacity)
3. The solution $u \in H^1_{\text{loc}}(\mathcal{X} \setminus \Sigma)$

**Statement:** If the singular set has zero capacity (even if its Hausdorff dimension is large), it is removable for the $H^1$ energy class. There exists a unique extension $\tilde{u} \in H^1(\mathcal{X})$ with $\tilde{u}|_{\mathcal{X} \setminus \Sigma} = u$.

**Certificate Logic:**
$$K_{\mathrm{Cap}_H}^- \wedge K_{\mathrm{Cap}_H}^{\mathrm{blk}} \Rightarrow K_{\mathrm{Cap}_H}^{\sim}$$

**Interface Permit Validated:** Removable Singularity.

**Literature:** {cite}`Federer69`; {cite}`EvansGariepy15`; {cite}`AdamsHedberg96`
:::

:::{prf:proof}
:label: sketch-mt-up-capacity

By Federer's theorem on removable singularities (1969, Section 4.7), sets of zero $(1,p)$-capacity are removable for $W^{1,p}$ functions. For $p=2$, the extension follows from the Lax-Milgram theorem applied to the weak formulation. The uniqueness follows from the maximum principle. See also Evans and Gariepy (2015, Theorem 4.7.2).
:::

---

(sec-spectral-gap-promotion)=
### Spectral Gap Promotion

:::{prf:theorem} [UP-Spectral] Spectral Gap Promotion (BarrierGap $\to$ YES)
:label: mt-up-spectral
:class: metatheorem

**Context:** Node 7 (StiffnessCheck) fails (Flat), but BarrierGap is Blocked ($K_{\text{gap}}^{\mathrm{blk}}$).

**Hypotheses.** Let $\mathcal{H}$ be a Hypostructure with:
1. A linearized operator $L = D^2\Phi(x^*)$ at a critical point $x^*$
2. A spectral gap: $\lambda_1(L) > 0$ (smallest nonzero eigenvalue is positive)
3. The nonlinear flow $\partial_t x = -\nabla \Phi(x)$ near $x^*$

**Statement:** If a spectral gap $\lambda_1 > 0$ exists, the Łojasiewicz-Simon inequality automatically holds with optimal exponent $\theta = 1/2$. The convergence rate is exponential: $\|x(t) - x^*\| \leq Ce^{-\lambda_1 t/2}$.

**Certificate Logic:**
$$K_{\mathrm{LS}_\sigma}^- \wedge K_{\text{gap}}^{\mathrm{blk}} \Rightarrow K_{\mathrm{LS}_\sigma}^+ \quad (\text{with } \theta=1/2)$$

**Interface Permit Validated:** Gradient Domination / Stiffness.

**Literature:** {cite}`Simon83`; {cite}`FeehanMaridakis19`; {cite}`Huang06`
:::

:::{prf:proof}
:label: sketch-mt-up-spectral

The Łojasiewicz-Simon inequality states $|\Phi(x) - \Phi(x^*)|^{1-\theta} \leq C\|\nabla\Phi(x)\|$ for some $\theta \in (0,1/2]$. When the Hessian is non-degenerate ($\lambda_1 > 0$), Taylor expansion gives $\theta = 1/2$. The exponential convergence then follows from the Gronwall inequality applied to the energy functional. See Simon (1983, Theorem 3) and Feehan and Maridakis (2019).
:::

---

(sec-o-minimal-promotion)=
### O-Minimal Promotion

:::{prf:theorem} [UP-OMinimal] O-Minimal Promotion (BarrierOmin $\to$ YES$^\sim$)
:label: mt-up-o-minimal
:class: metatheorem

**Context:** Node 9 (TameCheck) fails (Wild), but BarrierOmin is Blocked ($K_{\mathrm{TB}_O}^{\mathrm{blk}}$).

**Hypotheses.** Let $\mathcal{H}$ be a Hypostructure with:
1. A singular/wild set $W \subset \mathcal{X}$ that is a priori not regular
2. Definability: $W$ is definable in an o-minimal expansion of $(\mathbb{R}, +, \cdot)$ (e.g., $\mathbb{R}_{\text{an,exp}}$)
3. The dynamics are generated by a definable vector field

**Statement:** If the wild set is definable in an o-minimal structure, it admits a finite Whitney stratification into smooth manifolds. The set is topologically tame: it has finite Betti numbers, satisfies the curve selection lemma, and admits no pathological embeddings.

**Certificate Logic:**
$$K_{\mathrm{TB}_O}^- \wedge K_{\mathrm{TB}_O}^{\mathrm{blk}} \Rightarrow K_{\mathrm{TB}_O}^{\sim}$$

**Interface Permit Validated:** Tame Topology.

**Literature:** {cite}`vandenDries98` Ch.3 (cell decomposition, uniform finiteness); {cite}`Kurdyka98` Thm.1 (KL inequality); {cite}`Wilkie96` (model completeness)
:::

:::{prf:proof}
:label: sketch-mt-up-o-minimal

*Step 1 (Cell Decomposition).* By the **cell decomposition theorem** ({cite}`vandenDries98` Theorem 3.2.11), every definable set $W \subset \mathbb{R}^n$ in an o-minimal structure admits a finite partition:
$$W = \bigsqcup_{i=1}^N C_i$$
where each $C_i$ is a **definable cell**—a set homeomorphic to $(0,1)^{d_i}$ for some $d_i \leq n$. The cells are smooth manifolds with boundary, and the partition is canonical.

*Step 2 (Kurdyka-Łojasiewicz Gradient Inequality).* For any definable function $\Phi: \mathbb{R}^n \to \mathbb{R}$ in an o-minimal structure, the **Kurdyka-Łojasiewicz inequality** ({cite}`Kurdyka98` Theorem 1) holds near critical points:
$$\|\nabla(\psi \circ \Phi)(x)\| \geq 1 \quad \text{for some desingularizing function } \psi$$
This guarantees gradient descent converges in finite arc-length, preventing infinite oscillation. Combined with Step 1, trajectories cross finitely many cell boundaries.

*Step 3 (Uniform Finiteness + Tame Topology).* The **uniform finiteness theorem** ({cite}`vandenDries98` Theorem 3.4.4) bounds topological complexity: $\dim(W) \leq n$, $b_k(W) < \infty$ for all Betti numbers, and $W$ contains no pathological embeddings (wild arcs, horned spheres). This establishes the tame topology permit.
:::

---

(sec-surgery-promotion)=
### Surgery Promotion

:::{prf:theorem} [UP-Surgery] Surgery Promotion (Surgery $\to$ YES$^\sim$)
:label: mt-up-surgery
:class: metatheorem

**Context:** Any Node fails, Barrier breached, but Surgery $S$ executes and issues re-entry certificate ($K^{\mathrm{re}}$).

**Hypotheses.** Let $\mathcal{H}$ be a Hypostructure with:
1. A singularity at $(t^*, x^*) \in \mathcal{X}$ with modal diagnosis $M \in \{C.E, C.C, \ldots, B.C\}$
2. A valid surgery operator $\mathcal{O}_S: (\mathcal{X}, \Phi) \to (\mathcal{X}', \Phi')$ satisfying:
   - Admissibility: singular profile $V \in \mathcal{L}_T$ (canonical library)
   - Capacity bound: $\mathrm{Cap}(\text{excision}) \leq \varepsilon_{\text{adm}}$
   - Progress: $\Phi'(x') \leq \Phi(x) - \delta_S$ (height decrease)

**Statement:** If a valid surgery is performed, the flow continues on the modified Hypostructure $\mathcal{H}'$. The combined flow (pre-surgery on $\mathcal{X}$, post-surgery on $\mathcal{X}'$) constitutes a generalized (surgery/weak) solution.

**Certificate Logic:**
$$K_{\text{Node}}^- \wedge K_{\text{Surg}}^{\mathrm{re}} \Rightarrow K_{\text{Node}}^{\sim} \quad (\text{on } \mathcal{X}')$$

**Canonical Neighborhoods (Uniqueness):** The **Canonical Neighborhood Theorem** (Perelman 2003) ensures surgery is essentially unique: near any high-curvature point $p$ with $|Rm|(p) \geq r^{-2}$, the pointed manifold $(M, g, p)$ is $\varepsilon$-close (in the pointed Cheeger-Gromov sense) to one of:
- A round shrinking sphere $S^n / \Gamma$
- A round shrinking cylinder $S^{n-1} \times \mathbb{R}$
- A Bryant soliton

This **classification of local models** eliminates surgery ambiguity: the excision location and cap geometry are determined by the canonical structure up to diffeomorphism. Different valid surgery choices yield **diffeomorphic** post-surgery manifolds, making the surgery operation **functorial** in $\mathbf{Bord}_n$.

**Interface Permit Validated:** Global Existence (in the sense of surgery/weak flow).

**Literature:** {cite}`Hamilton97`; {cite}`Perelman03`; {cite}`KleinerLott08`
:::

:::{prf:proof}
:label: sketch-mt-up-surgery

The surgery construction follows Hamilton (1997) for Ricci flow and Perelman (2002-2003) for the rigorous completion. The key ingredients are: (1) canonical neighborhood theorem ensuring surgery regions are standard, (2) non-collapsing estimates controlling geometry, (3) finite surgery time theorem bounding the number of surgeries. The post-surgery manifold inherits all regularity properties.
:::

---

(sec-lock-promotion)=
### Lock Promotion

:::{prf:theorem} [UP-Lock] Lock Promotion (BarrierExclusion $\to$ GLOBAL YES)
:label: mt-up-lock
:class: metatheorem

**Context:** Node 17 (The Lock) is Blocked ($K_{\text{Lock}}^{\mathrm{blk}}$).

**Hypotheses.** Let $\mathcal{H}$ be a Hypostructure with:
1. The universal bad pattern $\mathcal{B}_{\text{univ}}$ defined via the Interface Registry
2. The morphism obstruction: $\mathrm{Hom}_{\mathcal{C}}(\mathcal{B}_{\text{univ}}, \mathcal{H}) = \emptyset$ in the appropriate category $\mathcal{C}$
3. Categorical coherence: all nodes converge to Node 17 with compatible certificates

**Statement:** If the universal bad pattern cannot map into the system (Hom-set empty), no singularities of any type can exist. The Lock validates global regularity and retroactively confirms all earlier ambiguous certificates.

**Certificate Logic:**
$$K_{\text{Lock}}^{\mathrm{blk}} \Rightarrow \text{Global Regularity}$$

**Interface Permit Validated:** All Permits (Retroactively).

**Literature:** {cite}`SGA4`; {cite}`Lurie09`; {cite}`MacLane71`
:::

:::{prf:proof}
:label: sketch-mt-up-lock

The proof uses the contrapositive: if a singularity existed, it would generate a non-trivial morphism $\phi: \mathcal{B}_{\text{univ}} \to \mathcal{H}$ by the universal property. The emptiness of the Hom-set is established via cohomological/spectral obstructions (E1-E10 tactics). This is the "Grothendieck yoga" of reducing existence questions to non-existence of maps. See SGA 4 for the categorical framework.
:::

---

(sec-absorbing-boundary-promotion)=
### Absorbing Boundary Promotion

::::{prf:theorem} [UP-Absorbing] Absorbing Boundary Promotion (BoundaryCheck $\to$ EnergyCheck)
:label: mt-up-absorbing
:class: metatheorem

**Context:** Node 1 (Energy) fails ($E \to \infty$), but Node 13 (Boundary) confirms an Open System with dissipative flux.

**Hypotheses.** Let $\mathcal{H}$ be a Hypostructure with:
1. A domain $\Omega$ with boundary $\partial\Omega$
2. An energy functional $E(t) = \int_\Omega e(x,t) \, dx$
3. A boundary flux condition: $\int_{\partial\Omega} \mathbf{n} \cdot \mathbf{F} \, dS < 0$ (strictly outgoing)
4. Bounded input: $\int_0^T \|\text{source}(\cdot, t)\|_{L^1(\Omega)} \, dt < \infty$

**Statement:** If the flux across the boundary is strictly outgoing (dissipative) and inputs are bounded, the internal energy cannot blow up. The boundary acts as a "heat sink" absorbing energy.

**Certificate Logic:**
$$K_{D_E}^- \wedge K_{\mathrm{Bound}_\partial}^+ \wedge (\text{Flux} < 0) \Rightarrow K_{D_E}^{\sim}$$

**Interface Permit Validated:** Finite Energy (via Boundary Dissipation).

**Literature:** {cite}`Dafermos16`; {cite}`DafermosRodnianski10`
::::

:::{prf:proof}
:label: sketch-mt-up-absorbing

The energy identity is $\frac{dE}{dt} = -\mathfrak{D}(t) + \int_{\partial\Omega} \mathbf{n} \cdot \mathbf{F} \, dS + \int_\Omega \text{source}(x,t) \, dx$. By hypothesis 3, the flux term satisfies $\int_{\partial\Omega} \mathbf{n} \cdot \mathbf{F} \, dS < 0$ (strictly outgoing). Since dissipation satisfies $\mathfrak{D}(t) \geq 0$, we have:
$$\frac{dE}{dt} \leq \int_\Omega \text{source}(x,t) \, dx \leq \|\text{source}(\cdot, t)\|_{L^1(\Omega)}$$

Integrating from $0$ to $t$ and using hypothesis 4:
$$E(t) \leq E(0) + \int_0^t \|\text{source}(\cdot, s)\|_{L^1(\Omega)} \, ds < \infty$$

This is the energy method of Dafermos (2016, Chapter 5) applied to hyperbolic conservation laws with dissipative boundary conditions.
:::

---

(sec-catastrophe-stability-promotion)=
### Catastrophe Stability Promotion

:::{prf:theorem} [UP-Catastrophe] Catastrophe-Stability Promotion (BifurcateCheck $\to$ StiffnessCheck)
:label: mt-up-catastrophe
:class: metatheorem

**Context:** Node 7 (Stiffness) fails (Flat/Zero Eigenvalue), but Node 7a (Bifurcation) identifies a **Canonical Catastrophe**.

**Hypotheses.** Let $\mathcal{H}$ be a Hypostructure with:
1. A potential $V(x)$ with a degenerate critical point: $V''(x^*) = 0$
2. A canonical catastrophe normal form: $V(x) = x^{k+1}/(k+1)$ for $k \geq 2$ (fold $k=2$, cusp $k=3$, etc.)
3. Higher-order stiffness: $V^{(k+1)}(x^*) \neq 0$

**Statement:** While the linear stiffness is zero ($\lambda_1 = 0$), the nonlinear stiffness is positive and bounded. The system is "Stiff" in a higher-order sense, ensuring polynomial convergence $t^{-1/(k-1)}$ instead of exponential.

**Certificate Logic:**
$$K_{\mathrm{LS}_\sigma}^- \wedge K_{\mathrm{LS}_{\partial^k V}}^+ \Rightarrow K_{\mathrm{LS}_\sigma}^{\sim} \quad (\text{Polynomial Rate})$$

**Interface Permit Validated:** Gradient Domination (Higher Order).

**Literature:** {cite}`Thom75`; {cite}`Arnold72`; {cite}`PostonStewart78`
:::

:::{prf:proof}
:label: sketch-mt-up-catastrophe

The Łojasiewicz exponent at a degenerate critical point is $\theta = 1/k$ for the $A_{k-1}$ catastrophe (Thom, 1975). For the normal form $V(x) = x^{k+1}/(k+1)$ with critical point at $x^* = 0$, we have $\nabla V(x) = x^k$ and $V(x) - V(x^*) = x^{k+1}/(k+1)$. The Łojasiewicz gradient inequality $\|\nabla V(x)\| \geq C|V(x) - V(x^*)|^{1-\theta}$ with $\theta = 1/k$ becomes $|x^k| \geq C|x^{k+1}|^{(k-1)/k}$. Since $(k+1)(k-1)/k = k - 1/k < k$ for $k \geq 2$, this holds near $x=0$. Integrating the gradient flow $\dot{x} = -x^k$ yields polynomial convergence $|x(t)| \sim t^{-1/(k-1)}$. Arnold's classification (1972) ensures these are the only structurally stable degeneracies.
:::

---

(sec-inconclusive-discharge-upgrades)=
### Inconclusive Discharge Upgrades

The following metatheorems formalize inc-upgrade rules. Blocked certificates indicate "cannot proceed"; inconclusive certificates indicate "cannot decide with current prerequisites."

:::{prf:theorem} [UP-IncComplete] Inconclusive Discharge by Missing-Premise Completion
:label: mt-up-inc-complete
:class: metatheorem

**Context:** A node returns $K_P^{\mathrm{inc}} = (\mathsf{obligation}, \mathsf{missing}, \mathsf{code}, \mathsf{trace})$ where $\mathsf{missing}$ specifies the certificate types that would enable decision.

**Hypotheses:** For each $m \in \mathsf{missing}$, the context $\Gamma$ contains a certificate $K_m^+$ such that:
$$\bigwedge_{m \in \mathsf{missing}} K_m^+ \Rightarrow \mathsf{obligation}$$

**Statement:** The inconclusive permit upgrades immediately to YES:
$$K_P^{\mathrm{inc}} \wedge \bigwedge_{m \in \mathsf{missing}} K_m^+ \Rightarrow K_P^+$$

**Certificate Logic:**
$$\mathsf{Obl}(\Gamma) \setminus \{(\mathsf{id}_P, \ldots)\} \cup \{K_P^+\}$$

**Interface Permit Validated:** Original predicate $P$ (via prerequisite completion).

**Literature:** Binary Certificate Logic (Definition {prf:ref}`def-typed-no-certificates`); Obligation Ledger (Definition {prf:ref}`def-obligation-ledger`).
:::

:::{prf:proof}
:label: sketch-mt-up-inc-complete

The NO-inconclusive certificate records an epistemic gap, not a semantic refutation (Definition {prf:ref}`def-typed-no-certificates`). When all prerequisites in $\mathsf{missing}$ are satisfied, the original predicate $P$ becomes decidable. The discharge condition (Definition {prf:ref}`def-inc-upgrades`) ensures the premises genuinely imply the obligation. The upgrade is sound because $K^{\mathrm{inc}}$ records the exact obligation and its missing prerequisites; when those prerequisites are satisfied, the original predicate $P$ holds by the discharge condition.
:::

:::{prf:theorem} [UP-IncAposteriori] A-Posteriori Inconclusive Discharge
:label: mt-up-inc-aposteriori
:class: metatheorem

**Context:** $K_P^{\mathrm{inc}}$ is produced at node $i$, and later nodes add certificates that satisfy its $\mathsf{missing}$ set.

**Hypotheses:** Let $\Gamma_i$ be the context at node $i$ with $K_P^{\mathrm{inc}} \in \Gamma_i$. Later nodes produce $\{K_{j_1}^+, \ldots, K_{j_k}^+\}$ such that the certificate types satisfy:
$$\{\mathrm{type}(K_{j_1}^+), \ldots, \mathrm{type}(K_{j_k}^+)\} \supseteq \mathsf{missing}(K_P^{\mathrm{inc}})$$

**Statement:** During promotion closure (Definition {prf:ref}`def-closure`), the inconclusive certificate upgrades:
$$K_P^{\mathrm{inc}} \wedge \bigwedge_{m \in \mathsf{missing}(K_P^{\mathrm{inc}})} K_m^+ \Rightarrow K_P^+$$

**Certificate Logic:**
$$\mathrm{Cl}(\Gamma_{\mathrm{final}}) \ni K_P^+ \quad \text{(discharged from } K_P^{\mathrm{inc}} \text{)}$$

**Consequence:** The obligation ledger $\mathsf{Obl}(\mathrm{Cl}(\Gamma_{\mathrm{final}}))$ contains strictly fewer entries than $\mathsf{Obl}(\Gamma_{\mathrm{final}})$ if any inc-upgrades fired during closure.

**Interface Permit Validated:** Original predicate $P$ (retroactively).

**Literature:** Promotion Closure (Definition {prf:ref}`def-promotion-closure`); Kleene fixed-point iteration {cite}`Kleene52`.
:::

:::{prf:proof}
:label: sketch-mt-up-inc-aposteriori

The promotion closure iterates until fixed point. On each iteration, inc-upgrade rules (Definition {prf:ref}`def-inc-upgrades`) are applied alongside blk-promotion rules. The a-posteriori discharge is triggered when certificates from later nodes enter the closure and match the $\mathsf{missing}$ set. Termination follows from the certificate finiteness condition (Definition {prf:ref}`def-cert-finite`).
:::
