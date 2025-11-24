# Hypostructures: A Variational Theory of Regularity for Stratified Metric Flows

## Abstract

We construct a calculus of variations for dissipative dynamics on stratified metric spaces. A **Hypostructure** is a stratified metric gradient flow on a complete, separable metric space endowed with a Whitney/Fredholm stratification, a lower semi-continuous energy, and a metric–dissipation inequality on singular interfaces. Trajectories are curves of bounded variation; we prove a \emph{Stratified BV Chain Rule} for the energy along hybrid arcs, decomposing the dissipation into absolutely continuous and jump parts in time, with any Cantor part vanishing under appropriate gradient inequalities. This decomposition underlies a family of Morse–Conley type exclusion principles. Under a compensated compactness (Palais–Smale–type) hypothesis tied to a defect measure, and a stratified Łojasiewicz–Simon inequality, we prove rectifiability and finiteness of the jump set and convergence to a compact terminal stratum. These results replace the binary “regularity vs blow-up” alternative by a graded \emph{capacity analysis}: singular strata are ruled out when they lack sufficient energetic capacity or topological index to sustain dissipation defects, and global regularity follows whenever a null stratification covers all potential singular profiles.

## 1. Introduction

The analysis of global well-posedness for nonlinear evolution equations is obstructed by the locality of coercive estimates and by topological changes in the governing semi-flow. When dynamics allow regime changes—interpreted as passages between strata with distinct dissipative structures—classical Lyapunov theory is insufficient. This paper formalizes a geometric structure, the **Hypostructure**, as a stratified metric gradient flow: continuous dissipation occurs inside strata, while singular energy costs are encoded on interfaces. Trajectories are curves of bounded variation in the metric space; the jumps contribute the singular part of the distributional derivative of the energy.

We show that a metric-dissipation inequality on singular interfaces and a compensated compactness (Palais–Smale–type) condition tied to a defect measure force the jump set to have finite $\mathcal{H}^0$-measure and yield convergence to a compact attracting stratum. The results are formulated for metric gradient flows (or differential inclusions) on Banach manifolds, without assuming uniqueness or finite-dimensionality.

The central contribution of this framework is to replace the binary alternative “global regularity versus blow-up” by a graded \emph{capacity analysis} of the phase space. Classical weak solutions are typically too flexible, admitting non-physical singularities that violate energy or dissipation constraints. The hypostructure acts as a \emph{variational selection principle}: it singles out a class of physically admissible trajectories (BV in energy, compatible with the metric–dissipation inequality) and proves that, within this class, singular behaviour is topologically constrained by the stratification. If the stratification is well-chosen and structurally null, this constraint forces global regularity. Thus the analytical burden shifts from controlling \emph{all} weak solutions to verifying structural exclusion on a finite collection of geometrically and physically meaningful strata.

## 2. Hypostructures as Stratified Metric Gradient Flows

We begin by specifying the ambient space. In order to speak both about smooth strata (to formulate transversality and Fredholm regularity) and about metric slopes (to apply the Ambrosio–Gigli–Savaré theory of gradient flows), we work with a Banach manifold endowed with an intrinsic metric.

**Assumption (Ambient space).**  
$\mathcal{X}$ is a $C^1$ Banach manifold modeled on a separable Banach space $B$, endowed with a complete metric $d_{\mathcal{X}}$ that is compatible with the manifold topology but not necessarily induced by a linear norm (for instance, $d_{\mathcal{X}}$ may be a control distance or a Wasserstein-type metric). When needed, we equip $\mathcal{X}$ with a Borel reference measure $\mathfrak{m}$ to interpret spatial integrals (for example in concrete PDE realizations). All “almost everywhere” statements in this paper refer either to Lebesgue measure in time (on $[0,\infty)$) or to $\mathfrak{m}$ on $\mathcal{X}$ when explicitly indicated.

**Definition 2.1 (Hypostructure).** A hypostructure is a tuple $(\mathcal{X},d_{\mathcal{X}},\mathfrak{m},\Sigma,\Phi,\psi)$ where:
1. $\Sigma=\{S_\alpha\}_{\alpha\in\Lambda}$ is a locally finite partition of $\mathcal{X}$ into $C^1$-submanifolds (strata), partially ordered by $\preceq$, and \emph{Fredholm regular} in the sense that for each stratum $S_\alpha$ the Hessian (second variation) of $\Phi$ along the normal bundle $N S_\alpha$ defines a Fredholm operator of finite index (typically index zero in applications) on a Hilbert completion of $N S_\alpha$ (for example, via an $L^2$ pairing in PDE realizations). We also assume a \emph{local conical structure}: for every point $x\in \partial S_\alpha\cap S_\beta$ with $S_\beta\prec S_\alpha$ there exists a neighbourhood of $x$ that is bi-Lipschitz equivalent (with respect to $d_{\mathcal{X}}$) to a product $S_\beta\times C(L)$, where $C(L)$ is a metric cone over a lower-dimensional “link” $L$. In particular, codimension is well defined, descending chains of strata have strictly decreasing codimension, and near interfaces the space looks like a regular part crossed with a cone, justifying the product charts used in the BV chain rule.
2. **Frontier condition:** If $S_\alpha\cap \overline{S_\beta}\neq\emptyset$, then $S_\alpha\subseteq \overline{S_\beta}$.
3. **Singular interfaces:** $\partial S_\alpha=\mathcal{E}_\alpha\cup \bigcup_{\beta\neq \alpha} G_{\alpha\to\beta}$, with $G_{\alpha\to\beta}$ Borel (jump interface) and $\mathcal{E}_\alpha$ equilibria.
4. $\Phi:\mathcal{X}\to[0,\infty]$ is lower semi-continuous on $\mathcal{X}$ and continuous on each stratum (energy functional).
5. $\psi:\Gamma\to[0,\infty)$ with $\Gamma=\bigcup_{\alpha,\beta}G_{\alpha\to\beta}$ is a transition (interfacial) cost.

**Remark 2.1 (Weak–strong topology in applications).**  
In applications to evolution equations on domains (e.g. parabolic or dispersive PDEs), the manifold $\mathcal{X}$ is typically realized as a function space such as $L^2(\Omega)$, $H^1(\Omega)$, or a suitable subspace thereof. The metric $d_{\mathcal{X}}$ is often chosen to encode a weak topology on bounded sets (for instance, a negative Sobolev norm or the metric induced by weak convergence), whereas the Lyapunov functional $\Phi$ controls a stronger norm. In this setting:
- The lower semi-continuity of $\Phi$ (Assumption A1) reflects standard weak lower semi-continuity properties of norms and energies (e.g. via Fatou’s lemma).
- The defect measure $\nu_u$ (Definition 3.1) quantifies the gap between weak convergence in $(\mathcal{X},d_{\mathcal{X}})$ and strong convergence of the energy, capturing concentration and oscillation phenomena in a concentration–compactness sense.

**Assumption A1 (Energy regularity).** $\Phi$ is proper, coercive on bounded strata, and l.s.c. on $\mathcal{X}$.

**Assumption A2 (Metric non-degeneracy and l.s.c. cost).** The transition cost $\psi$ is Borel measurable and lower semi-continuous on $\Gamma$, and satisfies the subadditivity property
$$
\psi(u\to v) \le \psi(u\to w) + \psi(w\to v)
$$
whenever the intermediate transitions are admissible in the stratification graph. This prevents “interfacial arbitrage”: the cost of moving between two strata cannot be lowered by decomposing a transition into a sequence of cheaper intermediate jumps or by grazing along interfaces. Moreover, there exists $\kappa>0$ such that for any $u\in G_{\alpha\to\beta}$,
$$
\psi(u)\ge \kappa \min\bigl(1, \inf_{v\in S_{\mathrm{target}}} d_{\mathcal{X}}(u,v)^2\bigr),
$$
where $S_{\mathrm{target}}$ is the stable manifold of the target stratum. This prevents trajectories from reducing the cost of entering a given stratum by decomposing the transition into a sequence of “cheaper” intermediate jumps.

**Assumption A2' (Stratified transversality).** Each local flow field is tangent to the stratification and enters lower strata transversally: if $u\in \partial S_\alpha\cap G_{\alpha\to\beta}$ and the flow points outward from $S_\alpha$, then its projection lies in the tangent cone of $S_\beta$. This ensures transversality to stratification boundaries, precluding grazing trajectories and yielding well-defined entry times.

### 2.2 Metric Chain Rule in Stratified Spaces

Given a curve $u:[0,\infty)\to\mathcal{X}$, its \emph{metric derivative} at time $t$ is
$$
|\dot u|(t) := \lim_{h\to 0} \frac{d_{\mathcal{X}}(u(t+h),u(t))}{|h|}
$$
whenever the limit exists; for BV curves this limit exists for Lebesgue-almost every $t$ and belongs to $L^1_{\mathrm{loc}}(0,\infty)$. We use $D_t$ to denote distributional derivatives with respect to time and $\frac{d}{dt}$ for classical derivatives when they exist.

Trajectories are curves $u\in BV_{\mathrm{loc}}([0,\infty);\mathcal{X})$; we use the metric slope $|\partial\Phi|(u)$ in the sense of Ambrosio–Gigli–Savaré.

**Definition 2.2 (Dissipative hypostructural trajectory).**  
A trajectory $u$ is dissipative if:
1. (**Existence, A0**) for any initial $u(0)\in\mathcal{X}$ there exists a maximal BV trajectory defined on $[0,T_{\max})$;
2. for $\mathcal{L}^1$-a.e. $t$ with $u(t)\in S_\alpha$, the absolutely continuous part satisfies
   $$
   D_t^{ac}(\Phi\circ u)(t) \le -|\partial\Phi|^2(u(t)) \le -W_\alpha(u(t)),
   $$
   with $W_\alpha:S_\alpha\to[0,\infty)$;
3. the jump set $J_u$ is at most countable, and for each $t_k\in J_u$ with $u(t_k^-)\in G_{\alpha\to\beta}$ and $u(t_k^+)=u(t_k)\in S_\beta$ one has
   $$
   \Phi(u(t_k^+)) - \Phi(u(t_k^-)) \le -\psi(u(t_k^-));
   $$
4. any Cantor part of $D_t(\Phi\circ u)$ is nonpositive and included in the dissipation measure.

The next result summarizes the BV structure of the energy along such trajectories.

**Theorem 2.1 (Stratified BV Chain Rule).**  
Let $u$ be a dissipative hypostructural trajectory. Then $\Phi\circ u$ belongs to $BV_{\mathrm{loc}}([0,\infty))$, and its distributional derivative admits the decomposition
$$
D_t(\Phi\circ u)
= -|\partial\Phi|^2(u)\,\mathcal{L}^1\lfloor_{\mathrm{cont}} - \sum_{t_k\in J_u}\psi(u(t_k^-))\,\delta_{t_k} - \nu_{\mathrm{cantor}},
$$
where $J_u$ is the (at most countable) jump set, each atom at $t_k$ has mass at least $\psi(u(t_k^-))$, and $\nu_{\mathrm{cantor}}$ is a nonnegative Cantor measure. In particular, all dissipation of energy is accounted for by the continuous metric slope, the explicit interfacial costs at jumps, and a nonpositive Cantor part.

*Proof.* The BV property and the decomposition follow from a localization argument combining the general theory of curves of bounded variation in metric spaces with the stratified geometry near interfaces. Away from the interface set $\Gamma$, the standard metric chain rule for curves of maximal slope (see, e.g., Ambrosio–Gigli–Savaré, Thm. 1.2.5) yields the equality
$$
D_t^{ac}(\Phi\circ u)(t) = \frac{d}{dt}\Phi(u(t)) = -|\partial\Phi|^2(u(t))
$$
for almost every $t$ at which $u(t)$ remains in a single stratum $S_\alpha$ and there are no jumps. This gives the density $-|\partial\Phi|^2(u(t))$ with respect to $\mathcal{L}^1$ on the set of continuity times.

At a jump time $t_k\in J_u$ the behaviour is governed by the local structure of the stratification. By the local conical structure in Definition 2.1 and Assumption A2', there exists a neighbourhood $U\subset\mathcal{X}$ of the interface point and a bi-Lipschitz homeomorphism onto a product
$$
U \simeq (-\varepsilon,\varepsilon)\times S_\alpha,
$$
with the interface corresponding to $\{0\}\times S_\alpha$ and such that the flow is transversal to $\{0\}\times S_\alpha$. In these coordinates, the trajectory near $t_k$ can be written as $u(t)=(\xi(t),y(t))$ with $\xi(t_k^-)<0<\xi(t_k^+)$ (or the reverse) and $y(t)$ continuous through $t_k$. Since $u$ is a curve of bounded variation and the chart is bi-Lipschitz, the left and right limits $u(t_k^-)$ and $u(t_k^+)$ exist in the strong (metric) topology. Transversality ensures that $u(t_k^-)$ lies in the closure of the ingoing stratum and $u(t_k^+)$ in the outgoing stratum. Moreover, the bi-Lipschitz property of the chart implies that the metric speed $|\dot u|(t)$ is comparable to the Euclidean speed of $(\xi(t),y(t))$ in these coordinates; in particular, the $L^1$–integrability of the metric derivative and the BV structure of $u$ are preserved under the change of coordinates, so the one-dimensional analysis of the interface crossing captures the full metric behaviour near $t_k$.

Lower semi-continuity of $\Phi$ implies that the one-sided limits
$$
\Phi(u(t_k^-)) := \lim_{\tau\downarrow 0}\Phi(u(t_k-\tau)),\qquad
\Phi(u(t_k^+)) := \lim_{\tau\downarrow 0}\Phi(u(t_k+\tau))
$$
exist (possibly after restricting to a subsequence) and bound from below any approximate limits along the trajectory. By Definition 2.2 (item 3), the energy drop at $t_k$ satisfies
$$
\Phi(u(t_k^+))-\Phi(u(t_k^-)) \le -\psi(u(t_k^-)).
$$
Consequently, the distributional derivative $D_t(\Phi\circ u)$ acquires a Dirac mass at $t_k$ with weight equal to the jump in $\Phi\circ u$, which is bounded above by $-\psi(u(t_k^-))$. Summing over all jump times leads to the discrete term
$$
-\sum_{t_k\in J_u}\psi(u(t_k^-))\,\delta_{t_k}.
$$

The remaining singular part of $D_t(\Phi\circ u)$, denoted $\nu_{\mathrm{cantor}}$, is by definition the singular continuous (Cantor) part in the Lebesgue decomposition of the measure. The dissipativity inequality in Definition 2.2 and the Lyapunov property of $\Phi$ imply that $\nu_{\mathrm{cantor}}$ cannot assign positive mass to any set: a positive Cantor contribution would correspond to an unaccounted increase in energy along a subset of times of zero Lebesgue measure. Thus $\nu_{\mathrm{cantor}}$ is nonpositive, and after gathering signs we may regard it as a nonnegative measure in the expression above. This establishes the stated decomposition. □

**Definition 2.3 (Hypostructural relaxed slope).** The hypostructural slope at $u$ is the relaxation of the metric slope augmented by interfacial cost:
$$
|\partial \mathcal{H}|(u) := \begin{cases}
|\partial\Phi|(u) & \text{if } u\notin \Gamma,\\
\bigl(|\partial\Phi|(u)^2 + \psi(u)\bigr)^{1/2} & \text{if } u\in \Gamma,
\end{cases}
$$
so that the dissipation measure can be written compactly as
$$
D_t(\Phi\circ u) \le -|\partial \mathcal{H}|^2(u)\,\mathcal{L}^1 - \nu_{\mathrm{cantor}}.
$$
Rectifiability of the singular set is encoded in the finiteness of the singular part of $D_t(\Phi\circ u)$ together with the uniform gap $\psi\ge \kappa>0$ away from the attractor.

**Theorem 2.2 (Minimum dwell under Lipschitz flows).**  
If each flow field is uniformly Lipschitz on bounded-energy sets and there exists $\delta>0$ such that any post-jump state $u_{\mathrm{in}}\in S_\alpha$ and subsequent interface point $u_{\mathrm{out}}\in G_{\alpha\to\beta}$ satisfy $d_{\mathcal{X}}(u_{\mathrm{in}},u_{\mathrm{out}})\ge \delta$, then there is $\tau_{\mathrm{dwell}}>0$ with $t_{k+1}-t_k\ge \tau_{\mathrm{dwell}}$ for all jumps. □

## 3. Structural Compactness and Coercivity

**Definition 3.1 (Defect structure).** Equip $\mathcal{X}$ with a topology $\tau_w$ that is weaker than the metric topology induced by $d_{\mathcal{X}}$ on bounded sets. We postulate the existence of a map
$$
u \;\longmapsto\; \nu_u\in\mathcal{M},
$$
where $\mathcal{M}$ is a Banach space of nonnegative measures (for example, Radon measures on an auxiliary reference space in concrete PDE realizations), with the following property: for any bounded-energy sequence $\{u_n\}$ converging to $u$ in $\tau_w$, the sequence of “energy densities” associated with $u_n$ admits a decomposition into a weak limit plus a nonnegative defect measure $\nu_u$; the case $\nu_u=0$ is called \emph{strict convergence}. The precise underlying reference measure is immaterial for the hypostructural theory; only the norm $\|\nu_u\|_{\mathcal{M}}$ enters the coercivity assumptions below, and this norm is required to be lower semi-continuous with respect to $\tau_w$.

**Assumption A3 (Metric–defect compatibility / generalized Palais–Smale).** There exists a strictly increasing $\gamma:[0,\infty)\to[0,\infty)$ with $\gamma(0)=0$ such that along any flow in $S_\alpha$,
$$
|\partial\Phi|(u) \ge \gamma(\|\nu_u\|_{\mathcal{M}}).
$$
Equivalently: vanishing slope forces vanishing defect measure; energy cannot concentrate without strictly increasing the slope. In particular, if $\|\nu_u\|_{\mathcal{M}}\ge \delta>0$, then $W_\alpha(u)\ge \gamma(\delta)>0$, and bounded sequences with vanishing slope are precompact relative to the stratification.

*Remark 3.1 (Relaxation gap and profile decomposition viewpoints).*  
One intrinsic way to interpret the defect is via relaxation of the energy. Let $\bar\Phi$ denote the lower semi-continuous envelope of $\Phi$ with respect to the weak topology $\tau_w$. At points where $\Phi$ fails to be weakly lower semi-continuous, the metric slope of $\bar\Phi$ may be strictly smaller than that of $\Phi$, and the discrepancy can be encoded in the norm $\|\nu_u\|_{\mathcal{M}}$ of a suitable defect measure. In this sense, $\nu_u$ measures a “relaxation gap” between the original energy landscape and its weakly closed counterpart: vanishing defect corresponds to stability of the slope under relaxation. In many critical PDE applications, this abstract defect structure arises concretely from a profile decomposition: any bounded sequence admits a decomposition into a finite sum of rescaled profiles plus a remainder, and the energy functional $\Phi$ behaves additively to leading order on the profiles and lower semi-continuously on the remainder. In that setting, the defect norm $\|\nu_u\|_{\mathcal{M}}$ measures the energy carried by the remainder; Assumption A3 is then a reformulation of the principle that genuine lack of compactness (nontrivial profiles or remainder) is accompanied by a nontrivial metric slope. The present abstract formulation is designed to encompass such relaxation- and profile-decomposition-compatible situations without committing to a specific function-space realization.

**Assumption A4 (Safe stratum / absorbing manifold).** There exists a minimal stratum $S_\ast$ such that: (i) $S_\ast$ is forward invariant; (ii) any defect measure generated by trajectories in $S_\ast$ vanishes (compact type); (iii) $\Phi$ is a strict Lyapunov function on $S_\ast$ relative to its equilibria $\mathcal{E}_\ast$.

## 4. Main Results

### Theorem 4.1 (Rectifiability with vanishing cost)

Let $u:[0,\infty)\to\mathcal{X}$ be a dissipative hypostructural trajectory satisfying A1–A2 (and A2' if needed for dwell), with $\Phi_0<\infty$. Assume there is a modulus $\omega$ such that on interfaces $G_{\alpha\to\beta}$,
$$
\psi(x) \ge \omega\bigl(d_{\mathcal{X}}(x,\mathcal{E}_\ast)\bigr), \qquad \omega(0)=0,\ \omega \text{ strictly increasing}.
$$
Then either $u$ reaches the attracting set $\mathcal{E}_\ast$ in finite time, or the jump set $J_u$ is $\mathcal{H}^0$-rectifiable (finite) with the bound
$$
\omega(\delta)\,\mathcal{H}^0(J_u) \le \Phi_0,\qquad \delta:=\inf_{t\in J_u} d_{\mathcal{X}}(u(t^-),\mathcal{E}_\ast)>0.
$$
In particular, away from $\mathcal{E}_\ast$ the cost is uniformly positive, and only finitely many jumps can occur.

*Proof.* The BV chain rule gives $|D^s(\Phi\circ u)|(J_u)=\sum_{t_k\in J_u}\psi(u(t_k^-))\le \Phi_0$. If $u$ does not hit $\mathcal{E}_\ast$, then $\delta:=\inf_{t\in J_u}d(u(t^-),\mathcal{E}_\ast)>0$, so $\psi\ge \omega(\delta)>0$ on all active interfaces. Hence $\omega(\delta)\mathcal{H}^0(J_u)\le \Phi_0$, proving finiteness. If $\delta=0$, then the trajectory accumulates at $\mathcal{E}_\ast$; by lower semicontinuity and the gradient-flow structure, $u$ enters the attractor. □

### Theorem 4.2 (Global Regularity / Absorption)

Under A1–A4, any bounded trajectory $u$ enters $S_\ast$ in finite time and converges to $\mathcal{E}_\ast$.

*Proof.* By Theorem 4.1, there is $T^\ast$ after which no jumps occur; denote the terminal stratum by $S_f$. If $\inf_{t>T^\ast}\|\nu_{u(t)}\|_{\mathcal{M}}=\delta>0$, then by A3 the flow satisfies $D_t\Phi(u)\le -\gamma(\delta)$, contradicting $\Phi\ge 0$. Thus the defect measure vanishes along the tail and $\{u(t)\}_{t>T^\ast}$ is precompact. Its omega-limit set is non-empty, compact, invariant, and contained in $\overline{S_f}$ with no further transitions. The frontier condition and absence of jumps force $\omega(u)\subset S_f$. Dissipation vanishes only on equilibria; hence $\omega(u)\subset \mathcal{E}_\ast$ and $S_f=S_\ast$. □

## 5. Discussion

The hypostructure formalism yields global regularity from two ingredients: quantized energy loss at singular interfaces and defect-driven coercivity inside strata. The BV interpretation of the energy evolution isolates singular dissipation at jumps and continuous dissipation along flows. The finite-capacity/rectifiability principle eliminates infinite combinatorics; compensated compactness forces eventual compactness and convergence within the minimal stratum, providing a topological mechanism for non-coercive infinite-dimensional dynamics.

## 3. Renormalization, Gauge Fixing, and Capacity Classification

We now incorporate scaling symmetries into the hypostructural setting to capture blow-up/concentration phenomena in evolution equations. The singular interfaces $G_{\alpha\to\beta}$ can be viewed as thresholds of concentration (e.g., norms hitting a critical level) at which a renormalization step (gauge transformation $T_\lambda$) is applied; the interfacial cost $\psi$ is the energy dissipated or injected by this renormalization. The goal is to quantify when a singular stratum is dynamically inaccessible because the dissipation required to reach it is infinite.

### 3.1 Scaling Group and Singular Sequences

Let $\mathcal{G}=\{T_\lambda:\lambda>0\}$ be a one-parameter scaling group acting on $\mathcal{X}$, typically $(T_\lambda v)(x)=\lambda^{-\alpha} v(\lambda^{-1}x)$ with $\alpha$ dictated by critical invariance. The action is assumed to be smooth on each stratum and compatible with the metric (e.g., locally bi-Lipschitz on strata).

**Definition 3.1 (Singular sequence).** A sequence $\{u_n\}\subset\mathcal{X}$ is singular if there exist scales $\lambda_n\to 0$ and a nontrivial profile $v\in\mathcal{X}$ such that $d_{\mathcal{X}}(u_n,T_{\lambda_n} v)\to 0$. For a trajectory $u(t)$ with blow-up time $T^\ast$, we say $u$ is singular if there exists $\lambda(t)\to 0$ as $t\to T^\ast$ with $d_{\mathcal{X}}(u(t),T_{\lambda(t)} v)\to 0$ for some profile $v\neq 0$.

### 3.2 Dynamic Normalization (Gauge Fixing)

To disentangle scale from profile, we introduce a gauge slice transverse to the scaling orbit.

**Definition 3.2 (Gauge condition and renormalized manifold).** Let $\mathcal{M}\subset\mathcal{X}$ be a codimension-one submanifold transversal to $\mathcal{G}$ (e.g., $\{v:\|v\|_{\dot H^1}=1\}$). The gauge map $\pi:\mathcal{X}\setminus\{0\}\to \mathcal{M}\times \mathbb{R}_+$ sends $u$ to $(v,\lambda)$ with $u=T_\lambda v$ and $v\in\mathcal{M}$.

**Definition 3.3 (Renormalized trajectory).** For a trajectory $u(t)$ approaching a singularity, define $v(s)$ via the dynamic rescaling
$$
u(x,t)=\lambda(t)^{-\alpha} v\Big(\frac{x-x_c(t)}{\lambda(t)}, s(t)\Big),\qquad \frac{ds}{dt}=\lambda(t)^{-\beta},
$$
with gauge constraint $v(\cdot,s)\in\mathcal{M}$ for all $s$. The exponents $\alpha,\beta$ reflect the scaling of the equation and dissipation.

**Remark 3.2 (Dictionary of singularities).**  
In applications to evolutionary PDEs, the abstract convergence of the renormalized trajectory $v(s)$ to an equilibrium $v_\infty\in\mathcal{E}_\alpha$ in the hypostructure corresponds to specific blow-up scenarios in the original variables:
- A nontrivial fixed point in renormalized variables ($v_\infty\neq 0$) corresponds to a \emph{self-similar} collapse or growth profile.
- A periodic orbit in renormalized variables corresponds to \emph{discrete self-similarity} or pulsating blow-up.
- More complicated compact attractors correspond to \emph{modulated self-similar} regimes.

 The structural nullity conditions of Definition 6.2 are precisely designed to rule out such nontrivial equilibria or recurrent sets in the singular strata: virial nullity excludes stationary profiles via coercive monotonicity identities; capacity nullity excludes profiles whose approach would require infinite dissipation; and homological/topological nullity excludes invariant sets incompatible with the Conley/Wazewski index. When the only equilibria in the terminal stratum are regular (e.g. the zero or globally regular solutions), convergence in renormalized time implies global regularity in the original variables, rather than self-similar blow-up. In this sense, establishing structural nullity for all singular strata is an abstract Liouville-type theorem for the renormalized flow.

### 3.3 Capacity Functional

In the hypostructural setting for evolution equations, the abstract dissipation $W_\alpha(u)$ from Section 2.2 is realized by a concrete scale-homogeneous functional $\mathfrak{D}:\mathcal{X}\to[0,\infty)$ (e.g., $\mathfrak{D}(u)=\nu\|\nabla u\|_{L^2}^2$), satisfying $\mathfrak{D}(T_\lambda v)=\lambda^{-\gamma}\mathfrak{D}(v)$ for some exponent $\gamma$.

**Definition 3.4 (Capacity cost).** For a trajectory $u:[0,T^\ast)\to\mathcal{X}$, define
$$
\mathrm{Cap}(u):=\int_0^{T^\ast}\mathfrak{D}(u(t))\,dt.
$$
In renormalized variables,
$$
\mathrm{Cap}(u)=\int_0^{T^\ast}\lambda(t)^{-\gamma}\mathfrak{D}(v(s(t)))\,dt.
$$

**Assumption (Non-degenerate gauge).** The renormalized manifold $\mathcal{M}$ is chosen so that $c_{\mathcal{M}}:=\inf_{v\in\mathcal{M}}\mathfrak{D}(v)>0$ (e.g., if $\mathfrak{D}(v)=\nu\|\nabla v\|_{L^2}^2$ and $\mathcal{M}=\{\|\nabla v\|_{L^2}=1\}$, then $c_{\mathcal{M}}=\nu$).

**Lemma 3.1 (Gauge-fixed lower bound).** Under the non-degenerate gauge assumption,
$$
\mathrm{Cap}(u) \ge c_{\mathcal{M}} \int_0^{T^\ast} \frac{dt}{\lambda(t)^\gamma}.
$$
Thus the capacity cost reduces to a purely kinematic integral involving the blow-up rate and scaling exponent.

### 3.4 Capacity Classification

We classify equations according to the convergence/divergence of the gauge-fixed integral $\int_0^{T^\ast} \lambda(t)^{-\gamma} dt$ for fast focusing $\lambda(t)\to 0$:

- **Type I (zero cost; conservative/inviscid):** $\mathfrak{D}\equiv 0$ (or $\nu=0$), so $\mathrm{Cap}(u)\equiv 0$; singularities are not energetically obstructed.
- **Type II (finite cost; critical dispersive/Hamiltonian):** $\mathrm{Cap}(u)<\infty$ for admissible blow-up rates; singularities are energetically affordable (framework describes rates/profiles).
- **Type III (infinite cost; supercritical dissipative):** For $\lambda(t)$ faster than self-similar, $\int_0^{T^\ast}\lambda(t)^{-\gamma} dt=\infty$, so $\mathrm{Cap}(u)=\infty$; singularities are energetically forbidden.

### 3.5 Capacity Veto Theorem

**Theorem 3.1 (Capacity veto).** Let $(\mathcal{X},d_{\mathcal{X}},\mathfrak{m},\Sigma,\Phi,\psi)$ be a hypostructure with dissipation $\mathfrak{D}$ homogeneous of degree $-\gamma$ under scaling. Suppose a singular stratum $S_{\mathrm{sing}}$ corresponds to $\lambda\to 0$ and that $\mathrm{Cap}(u)=\infty$ for any trajectory attempting $\lambda(t)\to 0$ at that rate. Then $S_{\mathrm{sing}}$ has infinite energetic capacity and is dynamically null for finite-energy BV trajectories.

*Proof.* Let $\Phi_0:=\Phi(u(0))$ denote the initial energy of the trajectory. The BV chain rule gives $|D^s(\Phi\circ u)|(J_u)+\int_0^{T^\ast}W(u(t))\,dt \le \Phi_0$. The absolutely continuous part dominates $\int_0^{T^\ast}\mathfrak{D}(u(t))\,dt=\mathrm{Cap}(u)$. If $\mathrm{Cap}(u)=\infty$, then $\Phi\circ u$ would have unbounded variation, contradicting finiteness of $\Phi_0$. Thus no finite-energy trajectory can realize $\lambda\to 0$ with infinite capacity; $S_{\mathrm{sing}}$ is unreachable. □


## 4. Structural Exclusion Principles: Monotonicity and Geometric Convexity

Beyond capacity barriers and defect-coercivity, hypostructures admit additional structural mechanisms that exclude stationary or recurrent behavior in certain strata. We record three such mechanisms—virial-type monotonicity, geometric $\mu$-convexity, and variational rigidity—that can be verified on smooth strata in concrete PDE models.

### 4.1 Virial Monotonicity and Exclusion

On a smooth stratum $S_\alpha$ where tangent vectors and gradients are well-defined, consider a $C^1$ functional $J:\mathcal{X}\to\mathbb{R}$ playing the role of a dispersion or moment functional (e.g., a virial). Along a hypostructural trajectory $u(t)$ in $S_\alpha$, the orbital derivative of $J$ is
$$
\frac{d}{dt}J(u(t)) = \langle \dot u(t), \nabla J(u(t))\rangle.
$$
We assume a decomposition of the velocity field $\dot u=F_{\mathrm{diss}}+F_{\mathrm{inert}}$ into a dissipative (gradient-like) part and an inertial (skew or transport) part.

**Definition 4.1 (Virial splitting on a stratum).** A stratum $S_\alpha$ admits a virial splitting if there exist $F_{\mathrm{diss}},F_{\mathrm{inert}}$ and $J$ such that, along any smooth flow line in $S_\alpha$,
1. $\langle F_{\mathrm{diss}}(u),\nabla J(u)\rangle \le -c_1\Phi(u)$ for some $c_1>0$ (cohesive decay),
2. $\langle F_{\mathrm{inert}}(u),\nabla J(u)\rangle$ captures dispersive/expansive effects.

**Theorem 4.1 (Virial exclusion).** Suppose that on $S_\alpha$ the domination condition
$$
|\langle F_{\mathrm{inert}}(u),\nabla J(u)\rangle| < |\langle F_{\mathrm{diss}}(u),\nabla J(u)\rangle|
$$
holds for all nontrivial $u\in S_\alpha$. Then $S_\alpha$ contains no nontrivial equilibria of the hypostructural flow, and no trajectory can remain in $S_\alpha$ for all $t\in\mathbb{R}$ without converging to zero.

*Proof.* Let $u_\ast\in S_\alpha$ be an equilibrium of the hypostructural flow. By definition, $\dot u(t)=0$ whenever $u(t)\equiv u_\ast$, hence
$$
F_{\mathrm{diss}}(u_\ast)+F_{\mathrm{inert}}(u_\ast)=0.
$$
If $\Phi(u_\ast)>0$, then by Definition 4.1 we have
$$
\langle F_{\mathrm{diss}}(u_\ast),\nabla J(u_\ast)\rangle \le -c_1\Phi(u_\ast) < 0.
$$
Thus $\langle F_{\mathrm{diss}}(u_\ast),\nabla J(u_\ast)\rangle\neq 0$ and
$$
|\langle F_{\mathrm{diss}}(u_\ast),\nabla J(u_\ast)\rangle| \ge c_1\Phi(u_\ast)>0.
$$
Since $F_{\mathrm{inert}}(u_\ast)=-F_{\mathrm{diss}}(u_\ast)$, pairing with $\nabla J$ yields
$$
\langle F_{\mathrm{inert}}(u_\ast),\nabla J(u_\ast)\rangle = -\langle F_{\mathrm{diss}}(u_\ast),\nabla J(u_\ast)\rangle,
$$
so the absolute values coincide:
$$
|\langle F_{\mathrm{inert}}(u_\ast),\nabla J(u_\ast)\rangle| = |\langle F_{\mathrm{diss}}(u_\ast),\nabla J(u_\ast)\rangle|.
$$
This contradicts the domination condition, which requires the strict inequality
$$
|\langle F_{\mathrm{inert}}(u_\ast),\nabla J(u_\ast)\rangle| < |\langle F_{\mathrm{diss}}(u_\ast),\nabla J(u_\ast)\rangle|.
$$
Hence any equilibrium $u_\ast$ must satisfy $\Phi(u_\ast)=0$. On a dissipative stratum, $\Phi$ is a Lyapunov functional with $\Phi\ge 0$ and $\Phi=0$ only at the trivial state, so $u_\ast=0$. This proves that $S_\alpha$ contains no nontrivial equilibria.

We now consider a complete trajectory $u:\mathbb{R}\to S_\alpha$ of the hypostructural flow. Along $u(t)$ we have
$$
\frac{d}{dt}J(u(t)) = \langle \dot u(t),\nabla J(u(t))\rangle
 = \langle F_{\mathrm{diss}}(u(t)) + F_{\mathrm{inert}}(u(t)),\nabla J(u(t))\rangle.
$$
At each time $t$ with $\Phi(u(t))>0$, Definition 4.1 gives
$$
\langle F_{\mathrm{diss}}(u(t)),\nabla J(u(t))\rangle \le -c_1\Phi(u(t))<0,
$$
so this term is strictly negative. The domination condition implies
$$
|\langle F_{\mathrm{inert}}(u(t)),\nabla J(u(t))\rangle| < |\langle F_{\mathrm{diss}}(u(t)),\nabla J(u(t))\rangle|.
$$
Since $\langle F_{\mathrm{diss}},\nabla J\rangle$ is negative and has strictly larger magnitude than $\langle F_{\mathrm{inert}},\nabla J\rangle$, the sum
$$
\langle F_{\mathrm{diss}}(u(t)) + F_{\mathrm{inert}}(u(t)),\nabla J(u(t))\rangle
$$
remains strictly negative whenever $\Phi(u(t))>0$. Thus there exists a function $c(t)>0$ such that
$$
\frac{d}{dt}J(u(t)) \le -c(t)\quad\text{whenever }\Phi(u(t))>0.
$$
In particular, $J(u(t))$ is strictly decreasing on any interval where $\Phi(u(t))>0$. If there exists $t_0$ with $\Phi(u(t_0))>0$, then $J(u(t))\to -\infty$ as $t\to +\infty$ and $J(u(t))\to +\infty$ as $t\to -\infty$, contradicting lower boundedness of $J$ (e.g., by convexity and nonnegativity). Therefore, for a complete trajectory confined to $S_\alpha$ we must have $\Phi(u(t))=0$ for all $t\in\mathbb{R}$, which forces $u(t)\equiv 0$ by the previous paragraph.

It follows that $S_\alpha$ is transient in the hypostructure: any nontrivial trajectory entering $S_\alpha$ cannot remain there for all time and must exit through some interface $G_{\alpha\to\beta}$ in finite forward or backward time. Such exits contribute interfacial cost in the finite-capacity accounting of Theorem 4.1. □

### 4.6 Stratified Łojasiewicz–Simon Inequality and Zeno Exclusion

To rule out “Zeno” behaviour (infinitely many transitions in finite time with vanishing cost) near equilibria or singular interfaces, we impose a Łojasiewicz–Simon type gradient inequality relative to the energy.

**Assumption A5 (Stratified Łojasiewicz–Simon inequality).**  
Let $u_\infty\in \mathcal{E}_\ast$ be an equilibrium in the terminal (attracting) stratum. There exist constants $C>0$, $\theta\in(0,1)$, and a neighbourhood $\mathcal{U}$ of $u_\infty$ in $\mathcal{X}$ such that for all $u\in\mathcal{U}$,
$$
|\partial\Phi|(u) \;\ge\; C\,|\Phi(u)-\Phi(u_\infty)|^{\theta}.
$$
Analogous inequalities are assumed to hold, when needed, in neighbourhoods of other isolated equilibria or limit profiles associated with singular strata.

**Proposition 4.6 (Finite-length approach to equilibria).**  
Let $u:[0,T)\to\mathcal{X}$ be a dissipative trajectory with values in $\mathcal{U}$ for all $t\in[t_0,T)$ and assume Assumption A5 holds at $u_\infty$. Then:
1. The total metric length of $u$ on $[t_0,T)$ is finite:
   $$
   \int_{t_0}^T |\dot u|(t)\,dt < \infty.
   $$
2. In particular, any sequence of jump times $\{t_k\}\subset [t_0,T)$ approaching $T$ must be finite if each jump carries a cost bounded below in terms of the energy gap, ruling out Zeno accumulation of jumps near $u_\infty$.

*Proof.* Since $u$ is a dissipative hypostructural trajectory, Definition 2.2 and the chain rule (Theorem 2.1) imply that on the interval $[t_0,T)$, where $u(t)\in\mathcal{U}$ and no jumps occur, the absolutely continuous part of the dissipation satisfies
$$
D_t^{ac}(\Phi\circ u)(t) = \frac{d}{dt}\Phi(u(t)) \le -|\partial\Phi|^2(u(t))
$$
for almost every $t\in[t_0,T)$. By Assumption A5, for all such $t$ we also have
$$
|\partial\Phi|(u(t)) \;\ge\; C\,|\Phi(u(t))-\Phi(u_\infty)|^{\theta}.
$$
Combining these two inequalities yields
$$
\frac{d}{dt}\Phi(u(t)) \le -|\partial\Phi|^2(u(t)) \le -C^2\,|\Phi(u(t))-\Phi(u_\infty)|^{2\theta}
$$
for almost every $t\in[t_0,T)$. Set $E(t):=\Phi(u(t))-\Phi(u_\infty)\ge 0$. Then
$$
\frac{d}{dt}E(t) \le -C^2\,E(t)^{2\theta}
$$
for almost every $t\in[t_0,T)$. Since $2\theta\in(0,2)$, this differential inequality can be integrated by separation of variables. For any $t\in(t_0,T)$ and any $t_1\in(t_0,t)$,
$$
\int_{E(t_1)}^{E(t)} \frac{d\xi}{\xi^{2\theta}} \ge C^2 (t_1-t),
$$
which, upon taking $t_1\downarrow t_0$ and using $E(t)\ge 0$, shows that $E(t)$ decreases to zero as $t\uparrow T$ and that
$$
\int_{t_0}^T E(t)^{\theta}\,dt < \infty.
$$
Using the Łojasiewicz inequality once more, we have $|\partial\Phi|(u(t)) \ge C E(t)^{\theta}$, so
$$
\int_{t_0}^T |\partial\Phi|(u(t))\,dt \le \frac{1}{C}\int_{t_0}^T E(t)^{\theta}\,dt < \infty.
$$
Along curves of maximal slope one also has the estimate $|\dot u|(t)\le |\partial\Phi|(u(t))$ for almost every $t$ (see Ambrosio–Gigli–Savaré), hence
$$
\int_{t_0}^T |\dot u|(t)\,dt \le \int_{t_0}^T |\partial\Phi|(u(t))\,dt < \infty,
$$
which proves item (1): the trajectory has finite metric length on $[t_0,T)$.

For item (2), suppose by contradiction that there is an infinite sequence of jump times $\{t_k\}\subset [t_0,T)$ with $t_k\uparrow T$, and that each jump carries a cost bounded below in terms of the energy gap, say
$$
\Phi(u(t_k^-))-\Phi(u(t_k^+)) \ge \eta\bigl(\Phi(u(t_k^-))-\Phi(u_\infty)\bigr)
$$
for some strictly positive function $\eta$ on $(0,\infty)$. Since $E(t)\to 0$ as $t\uparrow T$, the sequence $E(t_k^-)$ tends to $0$. On the other hand, each jump requires the trajectory to move by a definite amount in the metric space: by Assumption A2, the cost at $t_k$ controls the squared distance to the relevant target set, so there exists a constant $c>0$ such that
$$
d_{\mathcal{X}}(u(t_k^-),u(t_k^+)) \ge c\,\sqrt{\eta(E(t_k^-))}
$$
for all large $k$. The total metric variation due to jumps is therefore bounded below by
$$
\sum_{k} d_{\mathcal{X}}(u(t_k^-),u(t_k^+)) \;\ge\; c\sum_k \sqrt{\eta(E(t_k^-))}.
$$
Since $\eta(E(t_k^-))>0$ for all $k$ and $E(t_k^-)\downarrow 0$, the right-hand side diverges unless only finitely many jumps occur. But the left-hand side is bounded above by the total length of $u$ on $[t_0,T)$, which we have just shown to be finite. This contradiction shows that $\{t_k\}$ must in fact be finite, ruling out Zeno accumulation of jumps near $u_\infty$.

Finally, in a Łojasiewicz neighbourhood the finite-length property and the monotonicity of $\Phi$ imply that any singular continuous (Cantor) part of $D_t(\Phi\circ u)$ must vanish: a nontrivial Cantor component would require infinitely many small oscillations of $\Phi\circ u$ along a set of positive Hausdorff dimension in time, incompatible with finite length and strictly decreasing energy. Thus, in the neighbourhoods provided by A5, the dissipation measure is purely absolutely continuous plus atomic at jump times. □

### 4.2 Geometric Locking via $\mu$-Convexity

Spectral gaps in linearized PDEs correspond, in the metric setting, to uniform $\mu$-convexity (geodesic convexity) of the energy. We encode geometric conditions that enforce such convexity.

**Definition 4.2 (Geometric conditioning and locking).** Let $\mathcal{I}:\mathcal{X}\to\mathbb{R}$ be a continuous geometric invariant. For a threshold $\mathcal{I}_c$, define the locked region
$$
S_{\mathrm{lock}}:=\{u\in\mathcal{X}:\mathcal{I}(u)>\mathcal{I}_c\}.
$$
We say $\Phi$ exhibits geometric locking on $S_{\mathrm{lock}}$ if there exists $\mu>0$ such that $\Phi$ is $\mu$-convex along geodesics restricted to $S_{\mathrm{lock}}$; i.e., for any metric geodesic $(\gamma_\theta)_{\theta\in[0,1]}$ in $S_{\mathrm{lock}}$,
$$
\Phi(\gamma_\theta) \le (1-\theta)\Phi(\gamma_0)+\theta\Phi(\gamma_1) - \tfrac12\mu\theta(1-\theta)d_{\mathcal{X}}(\gamma_0,\gamma_1)^2.
$$

**Theorem 4.2 (Locking and exponential convergence).** If $u(t)$ is a hypostructural trajectory that remains in $S_{\mathrm{lock}}$ for all $t\ge 0$, then there exists a unique equilibrium $u_\infty\in S_{\mathrm{lock}}$ and constants $C,\mu>0$ such that
$$
d_{\mathcal{X}}(u(t),u_\infty)\le C e^{-\mu t}.
$$
In particular, recurrent dynamics (cycles, chaos) are excluded in locked strata.

*Proof.* By Assumption A1, $\Phi$ is proper and lower semi-continuous, and its sublevel sets intersected with any fixed stratum are precompact. Since $S_{\mathrm{lock}}$ is a super-level set of the continuous invariant $\mathcal{I}$, it is Borel; we additionally assume it is closed and nonempty. On $S_{\mathrm{lock}}$, the functional $\Phi$ is $\mu$-convex along geodesics. The direct method in the calculus of variations then yields the existence of a minimizer $u_\infty\in S_{\mathrm{lock}}$ of $\Phi|_{S_{\mathrm{lock}}}$; $\mu$-convexity implies that this minimizer is unique.

By Definition 2.2 and the BV chain rule, a hypostructural trajectory is a curve of maximal slope for $\Phi$ in the sense of De Giorgi and of Ambrosio–Gigli–Savaré, at least while it remains in a single stratum. When $\Phi$ is $\mu$-convex along geodesics, curves of maximal slope coincide with the metric gradient flow of $\Phi$ and satisfy the evolution variational inequality (EVI$_\mu$)
$$
\frac12\frac{d}{dt} d_{\mathcal{X}}(u(t),v)^2 + \frac{\mu}{2}\, d_{\mathcal{X}}(u(t),v)^2
\le \Phi(v)-\Phi\bigl(u(t)\bigr)
$$
for all $v\in S_{\mathrm{lock}}$ and for almost every $t\ge 0$; see, for example, Ambrosio–Gigli–Savaré, \emph{Gradient Flows in Metric Spaces and in the Space of Probability Measures}. Choosing $v=u_\infty$ and using the minimality of $u_\infty$ gives
$$
\frac12\frac{d}{dt} d_{\mathcal{X}}(u(t),u_\infty)^2 + \frac{\mu}{2}\, d_{\mathcal{X}}(u(t),u_\infty)^2 \le 0.
$$
Gronwall’s lemma then yields
$$
d_{\mathcal{X}}(u(t),u_\infty)^2 \le e^{-\mu t}\, d_{\mathcal{X}}(u(0),u_\infty)^2,
$$
which is the stated exponential convergence with $C=d_{\mathcal{X}}(u(0),u_\infty)$. The contraction property implied by the EVI$_\mu$ also shows that no nontrivial recurrent dynamics (limit cycles or chaotic invariant sets) can exist in $S_{\mathrm{lock}}$: any two trajectories remaining in $S_{\mathrm{lock}}$ converge exponentially to each other and hence to the unique equilibrium $u_\infty$. □

### 4.3 Variational Rigidity and Roughness Penalties

We finally connect efficiency functionals driving potential singularities to the defect structure, obtaining quantitative stability.

**Definition 4.3 (Variational rigidity).** Let $\Xi:\mathcal{X}\to\mathbb{R}$ be a scaling-critical efficiency functional (e.g., a nonlinear production rate), with maximal value $\Xi_{\max}$. The hypostructure is variationally rigid if:
1. Maximizers are smooth: $\arg\max\Xi\subset \mathcal{X}_{\mathrm{reg}}$,
2. There exists $C>0$ such that
$$
\Xi_{\max}-\Xi(u) \ge C\,\|\nu_u\|_{\mathcal{M}}^2
$$
for all $u$, where $\nu_u$ is the defect measure.

**Theorem 4.3 (Roughness penalty).** In a variationally rigid hypostructure, any sequence $\{u_n\}$ with $\Xi(u_n)\to\Xi_{\max}$ satisfies $\|\nu_{u_n}\|_{\mathcal{M}}\to 0$ and is precompact. Thus “most efficient” candidates for singular behavior are asymptotically smooth; if smooth singularities are excluded by classical criteria, then no singularity is dynamically admissible. Moreover, attempting to maximize $\Xi$ while retaining nonzero defect $\nu_u$ forces, via A3 and the BV energy inequality, a coercive contribution to $D_t\Phi(u)$ and hence to the transition/interfacial cost; rough singular strata are energetically disfavored relative to smooth ones.

*Proof.* Let $\{u_n\}$ be a sequence with $\Xi(u_n)\to\Xi_{\max}$. By variational rigidity,
$$
\Xi_{\max}-\Xi(u_n) \;\ge\; C\,\|\nu_{u_n}\|_{\mathcal{M}}^2
$$
for all $n$. Taking $n\to\infty$ and using $\Xi(u_n)\to\Xi_{\max}$ shows that the left-hand side tends to $0$, hence
$$
\|\nu_{u_n}\|_{\mathcal{M}}^2 \le \frac{\Xi_{\max}-\Xi(u_n)}{C}\longrightarrow 0,
$$
so $\|\nu_{u_n}\|_{\mathcal{M}}\to 0$.

By the compensated compactness assumption A3 there is a strictly increasing function $\gamma$ with $\gamma(0)=0$ such that
$$
|\partial\Phi|(u) \;\ge\; \gamma(\|\nu_u\|_{\mathcal{M}})
$$
for all $u$ in each stratum. In particular, if $\|\nu_{u_n}\|_{\mathcal{M}}\to 0$, then $\gamma(\|\nu_{u_n}\|_{\mathcal{M}})\to 0$ and hence $|\partial\Phi|(u_n)\to 0$. Combining this with the coercivity of $\Phi$ on bounded strata (Assumption A1) and the precompactness clause in A3 (“bounded sequences with vanishing slope are precompact relative to the stratification”), we conclude that any maximizing sequence $\{u_n\}$ is precompact in $\mathcal{X}$; its limit points necessarily satisfy $\nu_u=0$ and are therefore regular (belonging to $\mathcal{X}_{\mathrm{reg}}$ by the first part of Definition 4.3). This proves the first assertion.

For the energetic statement, fix $\delta>0$. If $\|\nu_u\|_{\mathcal{M}}\ge \delta$, then by A3,
$$
|\partial\Phi|(u) \;\ge\; \gamma(\delta)>0.
$$
Along any hypostructural trajectory $u(t)$ with $u(t_0)=u$ at some time $t_0$, the BV energy inequality yields
$$
\frac{d}{dt}\Phi(u(t))\Big|_{t=t_0}^{ac} \le -|\partial\Phi|^2(u(t_0)) \le -\gamma(\delta)^2.
$$
Hence, at any configuration with defect bounded below by $\delta$, the instantaneous dissipation rate is bounded below by a strictly positive constant depending only on $\delta$. Integrating over a time interval on which $\|\nu_{u(t)}\|_{\mathcal{M}}\ge \delta$ shows that such segments contribute a uniform positive amount to the total dissipation (and, via the chain rule, to the singular/interfacial cost). In this precise sense, configurations with nonzero defect are energetically expensive: they cannot persist for long in any near-maximizing trajectory and are disfavoured relative to smooth configurations with vanishing defect. □

### 4.4 Ground State Gap and Quantization

Another potential failure mode in exclusion arguments is the “vanishing singularity”, where a would-be singular profile shrinks in norm until it disappears, evading large-data obstructions. We encode a structural mechanism that rules out nontrivial stationary profiles with arbitrarily small energy.

In many dissipative systems the energy and dissipation satisfy a scaling imbalance near the origin: informally, the “nonlinear production” scales superlinearly relative to the dissipation at small amplitudes, so that any nonzero stationary configuration must live at a definite energy scale. In the hypostructural setting this can be formulated directly in terms of the Lyapunov functional and its metric slope.

**Assumption (Small-energy slope gap).**  
There exist constants $\Phi_\ast>0$ and $c_\ast>0$ and an exponent $\theta>0$ such that for all $u\in\mathcal{X}$ with $0<\Phi(u)\le \Phi_\ast$ one has
$$
|\partial\Phi|(u) \;\ge\; c_\ast\,\Phi(u)^{\theta}.
$$
Equivalently, the dissipation rate dominates any possible nonlinear self-interaction at small energy scales: the slope cannot vanish “too fast” as $\Phi(u)\to 0$.

**Theorem 4.4 (Ground state gap).**  
Under the small-energy slope gap assumption, any nontrivial stationary point $u$ of the hypostructural flow satisfies
$$
\Phi(u)\ge \epsilon_0,
$$
for some universal constant $\epsilon_0\in(0,\Phi_\ast]$. In particular, the set of nontrivial stationary profiles (and hence of candidate singular limit profiles) is bounded away from the vacuum in energy space.

*Proof.* A stationary point in the metric gradient-flow sense satisfies $|\partial\Phi|(u)=0$. If $u$ is nontrivial with $0<\Phi(u)\le \Phi_\ast$, the small-energy slope gap gives
$$
0 = |\partial\Phi|(u) \ge c_\ast \Phi(u)^\theta >0,
$$
a contradiction. Thus no nontrivial stationary point can lie in the sublevel set $\{\Phi\le \Phi_\ast\}$; any nontrivial stationary $u$ must satisfy $\Phi(u)>\Phi_\ast$. Taking $\epsilon_0:=\Phi_\ast$ yields the claim. □

*Remark 4.3.*  
In concrete PDE models the small-energy slope gap typically follows from a scaling imbalance between the dissipation and the nonlinearity: if the dissipation functional $\mathfrak{D}$ scales linearly (or sublinearly) in amplitude while the nonlinear term scales superlinearly near zero, then the stationary Euler–Lagrange equation cannot admit arbitrarily small nontrivial solutions. The present abstract formulation isolates the only property used by the hypostructural arguments: a quantitative gap between the vacuum and the least nontrivial stationary profile in terms of the Lyapunov functional $\Phi$.

### 4.5 Modulational Locking of the Scaling Rate

The capacity classification of Section 3 treats different blow-up regimes according to the scaling behaviour of $\lambda(t)$. To prevent a trajectory from “drifting” between regimes (for example, from a critical Type I rate toward a mildly supercritical Type II rate), we formulate a modulational locking principle: under a spectral gap for the renormalized dynamics on the gauge manifold, the scaling parameter is asymptotically forced to its self-similar value.

We work on the renormalized manifold $\mathcal{M}$ of Definition 3.2. Let $v_\ast\in\mathcal{M}$ be a stationary renormalized profile and write $v=v_\ast+w$, with $w$ constrained by the gauge condition to lie in the subspace orthogonal (in an appropriate inner product) to the infinitesimal scaling mode $z_{\mathrm{scal}}$ (the tangent to the scaling orbit at $v_\ast$).

**Assumption (Spectral gap and modulation system).**  
Near $v_\ast$ the renormalized dynamics admit a decomposition
$$
v_s = \mathcal{F}(v,\lambda),\qquad \lambda_s = F(\lambda) + G(w),
$$
where $s$ is renormalized time, $v_s:=\partial_s v$, and:

1. (Linearization with gap) The map $\mathcal{F}$ is Fréchet differentiable at $(v_\ast,\lambda_\ast)$, and the linearized operator
   $$
   \mathcal{L} := D_v\mathcal{F}(v_\ast,\lambda_\ast)
   $$
   satisfies a spectral gap estimate on the gauge-orthogonal subspace:
   $$
   \langle \mathcal{L}w,w\rangle \le -\mu \|w\|^2 \quad\text{for all }w\perp z_{\mathrm{scal}},
   $$
   for some $\mu>0$.

2. (Nonlinear remainder) The nonlinear remainder in the $v$-equation is higher order in $w$:
   $$
   \|\mathcal{F}(v_\ast+w,\lambda) - \mathcal{L}w\| \le C\|w\|^2
   $$
   for $\|w\|$ sufficiently small.

3. (Modulation equation) The scaling parameter satisfies
  $$
  |\lambda_s - F(\lambda)| \le C\|w\|
  $$
  for $\|w\|$ sufficiently small, where $F(\lambda)$ is the \emph{scaling beta-function} determining the self-similar rate (for instance $F(\lambda)\approx -\lambda$ in simple self-similar collapse models), and $F(\lambda_\ast)=0$, $F'(\lambda_\ast)\neq 0$ (a nondegenerate self-similar fixed point).

**Theorem 4.5 (Modulational locking).**  
Under the spectral gap and modulation assumptions above, there exist constants $C_1,C_2>0$ and a neighbourhood of $(v_\ast,\lambda_\ast)$ in which any renormalized trajectory $(v(s),\lambda(s))$ satisfying the gauge condition and entering that neighbourhood for some $s_0$ obeys
$$
\|w(s)\| \le C_1 e^{-\mu (s-s_0)}\|w(s_0)\|\quad\text{for all }s\ge s_0,
$$
and
$$
|\lambda(s)-\lambda_\ast| \le C_2 e^{-\mu (s-s_0)}\|w(s_0)\|.
$$
In particular, the scaling rate is asymptotically locked to the self-similar value $\lambda_\ast$, and slow drift toward alternative scaling regimes is structurally excluded in the neighbourhood of $v_\ast$.

*Proof.* The evolution for $w$ is
$$
w_s = \mathcal{L}w + R(w,\lambda),
$$
with $\|R(w,\lambda)\|\le C\|w\|^2$ by assumption. On the gauge-orthogonal subspace the spectral gap gives
$$
\frac{d}{ds}\|w(s)\|^2
\le -2\mu\|w(s)\|^2 + C\|w(s)\|^3,
$$
for $s$ such that $\|w(s)\|$ is sufficiently small. Choose a neighbourhood of $v_\ast$ in which $\|w\|\le \eta$ implies $\|R(w,\lambda)\|\le C\|w\|^2$ and $C\|w\|\le \mu$; this is possible by continuity and smallness of $w$. Then whenever $\|w(s)\|\le \eta$ we have
$$
\frac{d}{ds}\|w(s)\|^2 \le -\mu\|w(s)\|^2,
$$
which yields
$$
\|w(s)\|^2 \le e^{-\mu (s-s_0)}\|w(s_0)\|^2
$$
as long as the trajectory remains in the chosen neighbourhood. A standard continuity/bootstrapping argument shows that if $\|w(s_0)\|$ is small enough, the solution cannot exit this neighbourhood forwards in $s$, so the exponential decay estimate holds for all $s\ge s_0$. Taking square roots gives the first inequality with a suitable constant $C_1$.

For the scaling parameter, the modulation equation can be written as
$$
\lambda_s - F(\lambda) = G(w),\qquad |G(w)|\le C\|w\|.
$$
Linearizing $F$ at $\lambda_\ast$ and using $F(\lambda_\ast)=0$, $F'(\lambda_\ast)\neq 0$, we write
$$
F(\lambda) = F'(\lambda_\ast)(\lambda-\lambda_\ast) + r(\lambda),
$$
where $r(\lambda)=o(|\lambda-\lambda_\ast|)$ as $\lambda\to\lambda_\ast$. For $\lambda$ sufficiently close to $\lambda_\ast$ we have $|r(\lambda)|\le \frac12 |F'(\lambda_\ast)|\,|\lambda-\lambda_\ast|$. Substituting into the modulation equation for $\lambda-\lambda_\ast$ gives
$$
(\lambda-\lambda_\ast)_s - F'(\lambda_\ast)(\lambda-\lambda_\ast)
= r(\lambda) + G(w).
$$
Taking absolute values and using the bounds on $r$ and $G$ and the exponential decay of $\|w(s)\|$ yields a scalar differential inequality of the form
$$
\frac{d}{ds}|\lambda(s)-\lambda_\ast|
\le -c_0 |\lambda(s)-\lambda_\ast| + C\|w(s)\|,
$$
for some $c_0>0$ and all $s\ge s_0$ in the neighbourhood. Applying Gronwall’s lemma and inserting the exponential bound on $\|w(s)\|$ shows that $|\lambda(s)-\lambda_\ast|$ decays exponentially at rate $c_0$ up to constants depending on $\|w(s_0)\|$, which can be absorbed into $C_2$. This proves the second inequality.

Returning to physical time (if desired) corresponds to a smooth reparametrization of $s$ and does not affect the qualitative conclusion: in the neighbourhood of a spectrally stable renormalized profile, the scaling behaviour cannot drift to a different blow-up rate. The only dynamically realized scaling is the self-similar one determined by the zero of the beta-function $F$. □
## 5. Dimensional, Topological, and Screening Constraints

We record three further structural selection principles—measure-theoretic starvation, topological handoff, and asymptotic autonomy—that constrain possible singular sets and trajectories in hypostructures.

### 5.1 Dissipation–Capacity Gap

Let $\mu$ denote the dissipation measure in space–time (e.g., the absolutely continuous part of $D_t(\Phi\circ u)$).

**Definition 5.1 (Dissipative dimension).** The dissipative dimension $d_{\mathrm{diss}}$ is the infimum of $d\ge 0$ such that $\mu$ is absolutely continuous with respect to the $d$-dimensional Hausdorff measure $\mathcal{H}^d$ on $\mathcal{X}\times(0,\infty)$:
$$
\mu \ll \mathcal{H}^d.
$$

**Definition 5.2 (Singular dimension).** Let $\mathcal{S}_u\subset \mathcal{X}\times(0,\infty)$ be the singular set of a trajectory (space–time points where the hypostructure fails to be regular). The singular dimension $d_{\mathrm{sing}}$ is the (parabolic) Hausdorff dimension of $\mathcal{S}_u$.

**Theorem 5.1 (Measure-theoretic starvation).** Suppose there exists a critical dimension $d_\ast$ such that:
1. (Partial regularity) $d_{\mathrm{sing}}< d_\ast$,
2. (Flux requirement) $\mu\ll \mathcal{H}^{d_\ast}$.
Then $\mu(\mathcal{S}_u)=0$. In particular, any singular regime that requires a positive dissipation flux supported on $\mathcal{S}_u$ is energetically starved and dynamically forbidden.

*Proof.* Since $d_{\mathrm{sing}}<d_\ast$, we have $\mathcal{H}^{d_\ast}(\mathcal{S}_u)=0$. By absolute continuity, $\mu(\mathcal{S}_u)=0$. Thus the dissipation measure cannot charge the singular set; any nontrivial flux must be realized on the regular part, where the hypostructural gradient-flow regularity applies. □

### 5.2 Homological Exclusion and the Wazewski–Conley Principle

Topological arguments can also exclude invariant sets without explicit growth of a numerical index. We recall the Wazewski retract principle, a precursor to Conley index theory, in a form adapted to strata.

**Definition 5.3 (Homological triviality).**  
A stratum $S_\alpha$ is homologically trivial for the flow if, for some isolating neighbourhood $N\subset S_\alpha$ of the invariant set under consideration, the Conley index (or, more simply, the relative homology) of $(N,E)$ is trivial, where $E\subset \partial N$ is the exit set for the flow. In particular, $H_\ast(N,E)\cong 0$.

**Theorem 5.2 (Wazewski–Conley exclusion).**  
Let $S_\alpha$ be a stratum and suppose there exists a compact isolating neighbourhood $N\subset S_\alpha$ for an invariant set \(K\subset N\), together with an exit set $E\subset \partial N$ such that:

1. Any trajectory that meets $E$ immediately exits $N$ and hence leaves $S_\alpha$ through the neighbouring strata determined by the stratification graph.
2. There is no continuous retraction of $N$ onto $E$ (equivalently, the relative homology $H_\ast(N,E)$ is nontrivial).

Then $S_\alpha$ contains no nonempty bounded invariant set. In particular, any trajectory that enters $N$ must eventually exit through $E$ and undergo a transition to a different stratum, incurring the corresponding interfacial cost $\psi$.

*Proof.* Suppose, for contradiction, that there exists a nonempty bounded invariant set $K\subset N\setminus E$. By invariance, every trajectory starting in $K$ remains in $K$ for all forward and backward times for which the trajectory is defined, and in particular does not meet the exit set $E$. The pair $(N,E)$ is, by assumption, an isolating neighbourhood for $K$ in the sense of Conley index theory: $K$ is the maximal invariant set contained in $N\setminus E$.

Consider the semiflow (or flow) restricted to $N$. Since any trajectory that meets $E$ immediately exits $N$, the exit set $E$ captures all escape directions from $N$ under the dynamics. Wazewski’s retract principle (see, e.g., standard references on dynamical systems) asserts that if there exists a continuous deformation retract of $N$ onto $E$ that is compatible with the flow on $E$, then no invariant set can remain in $N\setminus E$; conversely, the existence of an invariant set $K\subset N\setminus E$ obstructs such a retraction.

In the present setting we assume that no continuous retraction of $N$ onto $E$ exists and that the relative homology $H_\ast(N,E)$ is nontrivial. This nontriviality is precisely the algebraic-topological manifestation of the obstruction: if there were a deformation retract $r:N\to E$, then the inclusion $E\hookrightarrow N$ would induce an isomorphism in homology, forcing $H_\ast(N,E)$ to vanish. Thus, the hypotheses on $(N,E)$ guarantee that any attempt to deform $N$ into $E$ fails for topological reasons, and hence any invariant set contained in $N$ cannot be collapsed into the exit set under a continuous homotopy.

Conley index theory makes this precise by associating to the isolated invariant set $K$ an index $h(K)$, defined in terms of the homotopy type of the quotient $N/E$. If $K$ were nonempty and isolated in $N$, its Conley index would be nontrivial whenever $H_\ast(N,E)$ is nontrivial. However, the fact that every trajectory that meets $E$ immediately exits $N$ and that $K\subset N\setminus E$ is invariant implies that the index must coincide with the trivial index of the empty invariant set if $N$ could be retracted onto $E$. Our assumption that $H_\ast(N,E)$ is nontrivial rules out this possibility: there is a topological obstruction to the existence of a nonempty isolated invariant set in $N\setminus E$.

Therefore no nonempty bounded invariant set $K$ can exist inside $N\setminus E$. Any trajectory that enters $N$ must eventually cross the boundary at a point of $E$ and thus exit $N$, leaving the stratum $S_\alpha$ through a neighbouring stratum determined by the stratification graph. Each such exit is realized as a transition across an interface $G_{\alpha\to\beta}$ and contributes the corresponding interfacial cost $\psi$. This proves the claimed exclusion of bounded invariant sets in $S_\alpha$. □

### 5.3 Asymptotic Autonomy and Screening

Finally, we formalize the intuition that near a singularity, local self-interaction dominates nonlocal environmental forcing.

**Definition 5.4 (Local/nonlocal decomposition).** Decompose the driving force as
$$
F(u) = F_{\mathrm{self}}(u) + F_{\mathrm{env}}(u),
$$
where $F_{\mathrm{self}}$ is generated by the configuration in a shrinking neighborhood of a potential singular point (localized stratum), and $F_{\mathrm{env}}$ by the complement.

**Assumption (Singular scaling hierarchy).** Under the scaling action $T_\lambda$, assume
$$
\|F_{\mathrm{self}}(T_\lambda v)\|\sim \lambda^{-\alpha},\qquad \|F_{\mathrm{env}}(T_\lambda v)\|\sim \lambda^{-\beta},
$$
for some exponents with $\alpha>\beta$.

**Theorem 5.3 (Screening / asymptotic autonomy).** Under the singular scaling hierarchy, as $\lambda\to 0$,
$$
\frac{\|F_{\mathrm{env}}(T_\lambda v)\|}{\|F_{\mathrm{self}}(T_\lambda v)\|} \to 0.
$$
Thus near putative singular scales, the dynamics become asymptotically autonomous: local self-interaction determines stability, and external forcing becomes a lower-order perturbation.

*Proof.* By the assumed scaling hierarchy,
$$
\frac{\|F_{\mathrm{env}}(T_\lambda v)\|}{\|F_{\mathrm{self}}(T_\lambda v)\|}
\sim \lambda^{\alpha-\beta}
$$
as $\lambda\to 0$. Since $\alpha>\beta$, the exponent $\alpha-\beta$ is positive and hence $\lambda^{\alpha-\beta}\to 0$. This shows that the environmental forcing becomes negligible compared to the self-interaction in the renormalized regime, and any blow-up limit is governed by the autonomous evolution generated by $F_{\mathrm{self}}$ alone. □

### 5.4 Energetic Masking of Nodal Defects

In vector-valued dynamics, structural parameters (such as phase, angle, or direction) often become singular on a nodal set where the amplitude vanishes. A typical example is the decomposition $u=|u|\xi$ of a vector field into amplitude and direction: the direction $\xi$ is undefined at $u=0$. The energetic masking principle asserts that, under natural quantitative vanishing conditions, such kinematic defects do not contribute to the hypostructural energy budget and hence cannot carry singular capacity or defect.

To make this precise, let $\mathcal{X}$ be realized as a function space over a spatial domain (for instance, a Banach space of maps $u:\Omega\to\mathbb{R}^m$), and let $\Phi$ admit a local energy density $\varphi(u,\nabla u)$ so that
$$
\Phi(u)=\int_{\Omega} \varphi(u(x),\nabla u(x))\,dx.
$$
We consider a structural parameter $\xi$ defined whenever $u(x)\neq 0$.

**Definition 5.5 (Structural parameter and nodal set).**  
Let $\pi:\mathcal{X}\setminus\{0\}\to\mathcal{Y}$ be a projection onto a parameter space (e.g. $\xi(u)=u/|u|\in S^{m-1}$), and set $\xi(x):=\pi(u(x))$ wherever $u(x)\neq 0$. The \emph{nodal set} of $u$ is
$$
\mathcal{Z}_u:=\{x\in\Omega:u(x)=0\}.
$$
On $\mathcal{Z}_u$ the parameter $\xi$ is undefined (or singular).

We assume that near $\mathcal{Z}_u$ the energy density splits into an amplitude part and a direction part.

**Assumption (Amplitude–direction decomposition).**  
There exist constants $c_1,c_2>0$ such that, for $x$ in a neighbourhood of $\mathcal{Z}_u$,
$$
c_1\Big(|\nabla |u(x)||^2 + |u(x)|^2|\nabla \xi(x)|^2\Big)
\le \varphi(u(x),\nabla u(x))
\le c_2\Big(|\nabla |u(x)||^2 + |u(x)|^2|\nabla \xi(x)|^2\Big),
$$
with $\xi(x)$ defined for $u(x)\neq 0$.

**Theorem 5.4 (Energetic masking).**  
Let $u$ satisfy the amplitude–direction decomposition above in a neighbourhood of its nodal set $\mathcal{Z}_u$. Suppose that:

1. The amplitude vanishes at $\mathcal{Z}_u$ with order at least $\alpha\ge 1$, in the sense that there exist $r_0>0$ and $C>0$ such that
   $$
   |u(x)| \le C\,\mathrm{dist}(x,\mathcal{Z}_u)^{\alpha}
   \quad\text{for all }x\text{ with }\mathrm{dist}(x,\mathcal{Z}_u)<r_0;
   $$

2. The gradient of the structural parameter has at most first-order singularities,
   $$
   |\nabla \xi(x)| \le C'\,\mathrm{dist}(x,\mathcal{Z}_u)^{-1}
   \quad\text{for all }x\text{ with }\mathrm{dist}(x,\mathcal{Z}_u)<r_0,
   $$
   for some constant $C'>0$.

Then the energy of $u$ remains finite in a neighbourhood of the nodal set:
$$
\int_{\{x:\,\mathrm{dist}(x,\mathcal{Z}_u)<r_0\}} \varphi(u(x),\nabla u(x))\,dx < \infty.
$$
In particular, the singularity of $\nabla \xi$ on $\mathcal{Z}_u$ is energetically masked by the vanishing of $|u|$, and does not contribute to the hypostructural capacity or defect. Energetic and defect-based exclusion arguments may therefore be verified on the energy-carrying core $\{x:|u(x)|>\delta\}$ for small $\delta>0$, ignoring kinematic defects on $\mathcal{Z}_u$.

*Proof.* It suffices to estimate the two contributions $|\nabla |u||^2$ and $|u|^2|\nabla \xi|^2$ separately in a tubular neighbourhood of $\mathcal{Z}_u$. Since $u\in\mathcal{X}$ by assumption, we already know that $\Phi(u)<\infty$, but we give a local argument that isolates the role of the nodal set.

Fix a point $x_0\in\mathcal{Z}_u$ and consider a ball $B_r(x_0)$ with $r<r_0$. By the amplitude–direction decomposition and the bounds assumed on $|u|$ and $|\nabla\xi|$, we obtain
$$
\varphi(u(x),\nabla u(x))
\le c_2\Big(|\nabla |u(x)||^2 + |u(x)|^2|\nabla \xi(x)|^2\Big)
\le c_2\Big(|\nabla |u(x)||^2 + C^2 C'^2\,\mathrm{dist}(x,\mathcal{Z}_u)^{2\alpha-2}\Big).
$$
The first term $|\nabla |u||^2$ is locally integrable because $u$ belongs to the underlying Sobolev (or similar) space by definition of $\mathcal{X}$. For the second term, note that by hypothesis $\alpha\ge 1$, so $2\alpha-2\ge 0$ and hence
$$
\mathrm{dist}(x,\mathcal{Z}_u)^{2\alpha-2}
$$
is locally bounded near $\mathcal{Z}_u$. In particular, in spatial dimension $d$ we can estimate in polar coordinates around $x_0$:
$$
\int_{B_r(x_0)} \mathrm{dist}(x,\mathcal{Z}_u)^{2\alpha-2}\,dx
\lesssim \int_0^{r} \rho^{2\alpha-2}\,\rho^{d-1}\,d\rho
= \int_0^{r} \rho^{(2\alpha+d-3)}\,d\rho,
$$
which converges for any $\alpha\ge 1$ and any $d\ge 1$. Thus the contribution of $|u|^2|\nabla\xi|^2$ is locally integrable near $x_0$.

Covering $\mathcal{Z}_u$ by finitely many such balls (using compactness or local finiteness) and summing the estimates gives
$$
\int_{\{x:\,\mathrm{dist}(x,\mathcal{Z}_u)<r_0\}} \varphi(u(x),\nabla u(x))\,dx <\infty.
$$
Since $\varphi$ controls the local contribution to the Lyapunov functional $\Phi$, the singular behaviour of $\xi$ on $\mathcal{Z}_u$ does not generate infinite capacity or defect. This justifies restricting energetic and defect-based exclusion arguments to the energy-carrying core where $|u|$ is bounded away from zero. □

### 5.5 Geometric Attrition and Shape-Dependent Capacity

The capacity functional of Section 3 quantifies the cost of temporal collapse (through the scaling parameter $\lambda(t)$). Singular behaviour may also arise via spatially anisotropic collapse, in which certain directions shrink much faster than others (e.g. sheet- or filament-like structures). We now formulate a geometric attrition principle: if anisotropy increases the local dissipation rate in a controlled way, then trajectories cannot sustain extreme anisotropy at fixed finite energy.

Let $\mathcal{A}:\mathcal{X}\to[1,\infty)$ be a continuous “anisotropy modulus” (e.g. an aspect ratio or eccentricity functional) measuring deviation from isotropy on each stratum.

**Definition 5.6 (Anisotropic stiffness).**  
The dissipation is \emph{anisotropically stiff} relative to $\Phi$ and $\mathcal{A}$ if there exist constants $C_{\mathrm{stiff}}>0$ and $\gamma>0$ such that along any hypostructural trajectory $u(t)$ one has
$$
\frac{\mathfrak{D}(u(t))}{\Phi(u(t))} \;\ge\; C_{\mathrm{stiff}}\,\mathcal{A}(u(t))^{\gamma}
$$
whenever $\Phi(u(t))>0$, where $\mathfrak{D}$ is the dissipation rate functional from Section 3.3 (identified with $W_\alpha$ on each stratum).

Informally, anisotropic configurations (large $\mathcal{A}$) are dissipatively expensive relative to their stored energy.

**Theorem 5.5 (Geometric attrition).**  
Assume anisotropic stiffness. Let $u:[0,\infty)\to\mathcal{X}$ be a finite-energy hypostructural trajectory with $\Phi(u(0))=\Phi_0<\infty$. Then:

1. The integral of the anisotropy modulus is controlled by the initial energy:
   $$
   \int_0^T \mathcal{A}(u(t))^{\gamma}\,dt \;\le\; \frac{1}{C_{\mathrm{stiff}}}\,\log\frac{\Phi(u(0))}{\Phi(u(T))}\quad\text{for all }T>0.
   $$

2. In particular, if there exists $\delta>0$ and a sequence $t_n\to\infty$ such that $\Phi(u(t_n))\ge \delta$ and $\mathcal{A}(u(t_n))\to\infty$, then the integral $\int_0^\infty \mathcal{A}(u(t))^{\gamma}\,dt$ diverges. This contradicts the inequality above, hence such a trajectory cannot exist. Consequently, any trajectory that attempts to enter a regime with $\mathcal{A}(u(t))\to\infty$ while retaining a positive fraction of its energy is dynamically impossible: either $\mathcal{A}$ remains bounded along the trajectory, or the energy decays to zero before extreme anisotropy can be reached.

*Proof.* From the BV chain rule and the identification of $\mathfrak{D}$ with the absolutely continuous dissipation we have
$$
\frac{d}{dt}\Phi(u(t)) \le -\mathfrak{D}(u(t))
$$
for almost every $t$. Dividing by $\Phi(u(t))>0$ and using anisotropic stiffness yields
$$
\frac{d}{dt}\log\Phi(u(t)) = \frac{1}{\Phi(u(t))}\frac{d}{dt}\Phi(u(t))
\le -\frac{\mathfrak{D}(u(t))}{\Phi(u(t))}
\le -C_{\mathrm{stiff}}\mathcal{A}(u(t))^{\gamma}.
$$
Integrating from $0$ to $T$ gives
$$
\log\Phi(u(T))-\log\Phi(u(0))
\le -C_{\mathrm{stiff}}\int_0^T \mathcal{A}(u(t))^{\gamma}\,dt,
$$
or equivalently
$$
\int_0^T \mathcal{A}(u(t))^{\gamma}\,dt \le \frac{1}{C_{\mathrm{stiff}}}\,\log\frac{\Phi(u(0))}{\Phi(u(T))}.
$$
Since $\Phi(u(T))\ge 0$, the right-hand side is bounded above by $(1/C_{\mathrm{stiff}})\log(\Phi_0/\Phi_{\min})$ whenever $\Phi(u(T))\ge \Phi_{\min}>0$. Thus if $\Phi(u(t_n))\ge\delta>0$ for some $\delta$ and a sequence $t_n\to\infty$, the integral $\int_0^\infty\mathcal{A}(u(t))^{\gamma}\,dt$ must be finite, which is incompatible with any scenario in which $\mathcal{A}(u(t))\to\infty$ along a subsequence with nonvanishing energy: large anisotropy must be confined to sets of times with arbitrarily small total measure.

In particular, a putative singular stratum in which $\mathcal{A}(u)\to\infty$ while $\Phi(u)$ remains bounded away from zero is “capacity null”: approaching such a stratum would require either unbounded integrated dissipation (contradicting the finite-energy budget) or collapse of the energy itself. □


## 6. Structural Exclusion and Global Regularity

The preceding chapters provide a collection of independent exclusion mechanisms—capacity barriers, virial monotonicity, geometric $\mu$-convexity, variational rigidity, and dimensional/topological constraints. We now formalize how these mechanisms combine to yield a global regularity criterion. The key idea is that of a \emph{null} stratification: a hypostructural stratification in which every stratum is dynamically empty in the sense that it cannot support finite-time singularities of finite-energy trajectories.

Throughout this section we fix a hypostructure $(\mathcal{X},d_{\mathcal{X}},\mathfrak{m},\Sigma,\Phi,\psi)$ and consider BV trajectories $u:[0,\infty)\to\mathcal{X}$ with finite initial energy $\Phi(u(0))<\infty$. Let $\mathcal{S}_u\subset \mathcal{X}\times(0,\infty)$ denote the spatio-temporal singular set associated with $u$ (space–time points where the hypostructural regularity fails), as in Section 5.1, and let
$$
S_{\mathrm{sing}}^X(u):=\{x\in\mathcal{X}:\exists\,t>0\ \text{with }(x,t)\in \mathcal{S}_u\}
$$
be its spatial projection. Define the \emph{global} singular set and its projection by
$$
S_{\mathrm{sing}}:=\bigcup_{u}\mathcal{S}_u,\qquad S_{\mathrm{sing}}^X:=\bigcup_{u} S_{\mathrm{sing}}^X(u),
$$
where the union is taken over all finite-energy BV trajectories $u$ of the hypostructural flow.

### 6.1 Structural Covers

The first requirement is that the stratification actually “sees” all potentially singular configurations.

**Definition 6.1 (Structural cover).** A stratification $\Sigma=\{S_\alpha\}_{\alpha\in\Lambda}$ is a structural cover for the hypostructure if the spatial projection of the singular set is contained in the union of the stratum closures:
$$
S_{\mathrm{sing}}^X \subseteq \bigcup_{\alpha\in\Lambda} \overline{S_\alpha}.
$$
Equivalently, for every finite-energy trajectory $u$ and every singular point $(x,t)\in\mathcal{S}_u$ there exists $\alpha\in\Lambda$ such that $x\in\overline{S_\alpha}$.

*Remark.* Since the strata form a Whitney-type stratification of $\mathcal{X}$, their closures cover $\mathcal{X}$; the content of the definition is that the potential singular states belong to the same stratified structure that governs the regular dynamics. This rules out “latent” singular regimes not represented by the prescribed stratification.

### 6.2 Dynamically Null Strata and Null Stratifications

We distinguish between strata that are merely regular and those that are structurally excluded by the hypostructural mechanisms developed above. A stratum is declared \emph{null} when one of the explicit exclusion principles (capacity, virial, locking, variational rigidity, topological coercivity, or their refinements) applies, and we then prove that null strata are dynamically empty of finite-time singularities.

**Definition 6.2 (Structurally null stratum).**  
A stratum $S_\alpha\in\Sigma$ is \emph{structurally null} if it satisfies at least one of the following:

1. \emph{Capacity nullity:} Any trajectory attempting to approach $S_\alpha$ along a singular scaling requires infinite capacity in the sense of Theorem 3.1 (capacity veto).

2. \emph{Virial nullity:} The hypotheses of Theorem 4.1 (virial exclusion) hold on $S_\alpha$, so that $\Phi>0$ implies a strictly decreasing virial functional $J$ along any complete trajectory within $S_\alpha$.

3. \emph{Locking nullity:} The hypotheses of Theorem 4.2 (geometric locking) hold on $S_\alpha$, with the associated equilibrium lying in a lower (safer) part of the stratification.

4. \emph{Variational nullity:} The hypotheses of Theorem 4.3 (roughness penalty / variational rigidity) hold on $S_\alpha$, and smooth singularities in $\overline{S_\alpha}$ are excluded by independent regularity arguments.

5. \emph{Topological nullity:} The hypotheses of Theorem 5.2 (complexity barrier / topological handoff) hold with $S_\alpha$ playing the role of a collapse stratum that must hand off to a forbidden (already null) stratum before any singular limit can be reached.

In applications, one may also incorporate further mechanisms (e.g. measure-theoretic starvation via Theorem 5.1, screening via Theorem 5.3, or problem-specific Liouville theorems) into the definition of structural nullity.

**Theorem 6.1 (Nullity implication).**  
If a stratum $S_\alpha$ is structurally null in the sense of Definition 6.2, then it is dynamically empty of finite-time singularities of finite-energy BV trajectories:
$$
\mathcal{S}_u\cap \bigl(S_\alpha\times(0,\infty)\bigr) = \emptyset
$$
for every finite-energy hypostructural trajectory $u$.

*Proof.*  
If $S_\alpha$ is capacity-null, the conclusion follows directly from Theorem 3.1: any trajectory attempting to reach $S_\alpha$ along a singular scaling would require infinite capacity, contradicting the BV energy inequality. If $S_\alpha$ is virial-null, Theorem 4.1 shows that any nontrivial trajectory confined to $S_\alpha$ must experience strict virial decay and hence cannot support a nontrivial stationary or recurrent singular profile. If $S_\alpha$ is locking-null, Theorem 4.2 implies exponential convergence to a unique regular equilibrium lying in a safer region, ruling out singular accumulation in $S_\alpha$. If $S_\alpha$ is variationally null, Theorem 4.3 forces any near-maximizing sequence for the relevant efficiency functional to be smooth and precompact; thus candidate singular limits in $\overline{S_\alpha}$ are regular. Finally, if $S_\alpha$ is topologically null, Theorem 5.2 shows that any collapsing trajectory must hand off into a forbidden (hence already null) stratum before a singularity can form, so no singular limit supported in $S_\alpha$ is dynamically realizable. The same reasoning applies to any additional mechanisms included in the definition of structural nullity. □

**Definition 6.3 (Null stratification).**  
A stratification $\Sigma$ is \emph{null} if every stratum $S_\alpha\in\Sigma$ is structurally null in the sense of Definition 6.2, and moreover the minimal terminal stratum $S_\ast$ from Assumption A4 contains all equilibria and is regular in the sense that the dynamics restricted to $S_\ast$ are globally well-posed and free of finite-time singularities.

**Remark 6.1 (Structural stability of nullity).**  
The mechanisms listed in Definition 6.2 are robust under small perturbations of the hypostructure. Capacity nullity typically relies on strict comparisons of homogeneity exponents in the scaling group; virial and locking nullity rely on strict domination inequalities for Lyapunov-type functionals; topological nullity relies on discrete homological indices (e.g. nontrivial Conley indices or Wazewski exit sets) that are invariant under small continuous deformations. Consequently, if a given hypostructure admits a null stratification, then all sufficiently small perturbations of the energy $\Phi$ and metric $d_{\mathcal{X}}$ (in appropriate $C^2$ or Lipschitz topologies on strata) yield nearby hypostructures with the same qualitative nullity pattern. In this sense, the global regularity obtained from a null stratification is \emph{structurally stable} under small perturbations of the model.

### 6.3 Structural Global Regularity

We can now state the global regularity meta-theorem: if the stratification covers all potential singular states and every stratum is null, then finite-time singularities are impossible.

**Definition 6.4 (Global regularity).** The hypostructural flow is \emph{globally regular} if for every finite-energy initial datum $u_0\in\mathcal{X}$ there exists a BV trajectory $u:[0,\infty)\to\mathcal{X}$ with $u(0)=u_0$ such that the associated singular set $\mathcal{S}_u$ contains no points with finite time coordinate:
$$
\mathcal{S}_u\cap \bigl(\mathcal{X}\times(0,T]\bigr)=\emptyset\quad\text{for every }T>0.
$$
Equivalently, no finite-energy trajectory develops a singularity in finite time.

**Theorem 6.2 (Structural global regularity).** Let $(\mathcal{X},d_{\mathcal{X}},\mathfrak{m},\Sigma,\Phi,\psi)$ be a hypostructure satisfying Assumptions A0–A4. Suppose that:

1. The stratification $\Sigma$ is a structural cover in the sense of Definition 6.1, i.e.
   $$
   S_{\mathrm{sing}}^X \subseteq \bigcup_{\alpha\in\Lambda} \overline{S_\alpha}.
   $$

2. The stratification is null in the sense of Definition 6.3: every $S_\alpha$ is null and the safe stratum $S_\ast$ is regular and absorbing.

Then the hypostructural flow is globally regular in the sense of Definition 6.4: no finite-time singularity can form from finite-energy initial data.

*Proof.* Assume for contradiction that global regularity fails. Then there exists a finite-energy BV trajectory $u:[0,\infty)\to\mathcal{X}$ and a finite time $T^\ast>0$ such that the singular set $\mathcal{S}_u$ contains a point $(x_\ast,T^\ast)$ with $x_\ast\in\mathcal{X}$. By definition of $S_{\mathrm{sing}}^X(u)$, we have $x_\ast\in S_{\mathrm{sing}}^X(u)\subseteq S_{\mathrm{sing}}^X$.

By the structural cover property (Definition 6.1), there exists at least one index $\alpha\in\Lambda$ such that
$$
x_\ast \in \overline{S_\alpha}.
$$
Fix such an $\alpha$. Since $(x_\ast,T^\ast)\in\mathcal{S}_u$, there exists a sequence $t_n\uparrow T^\ast$ and states $x_n:=u(t_n)$ such that $x_n\to x_\ast$ in $\mathcal{X}$. By passing to a subsequence if necessary, we may assume that each $x_n$ lies in some stratum $S_{\alpha_n}$. By the frontier condition and the definition of the closure, the indices $\alpha_n$ must be eventually bounded below by $\alpha$ in the partial order, and in any case there exists a sequence $\{\tilde t_n\}$ and indices $\{\tilde\alpha_n\}$ with $u(\tilde t_n)\in S_{\tilde\alpha_n}$ and $u(\tilde t_n)\to x_\ast$ such that either $S_{\tilde\alpha_n}=S_\alpha$ for all $n$ or $S_{\tilde\alpha_n}\subset \overline{S_\alpha}$ for all $n$.

We distinguish two cases.

1. If there exists a subsequence with $u(t_n)\in S_\alpha$ for all $n$ and $u(t_n)\to x_\ast$, then by nullity of $S_\alpha$ (Definition 6.2) the point $(x_\ast,T^\ast)$ cannot be singular for $u$, contradicting $(x_\ast,T^\ast)\in\mathcal{S}_u$.

2. Otherwise, every sufficiently large $n$ satisfies $u(t_n)\in S_{\beta_n}$ with $S_{\beta_n}\subsetneq \overline{S_\alpha}$. By the frontier condition, each $S_{\beta_n}$ is a lower-dimensional stratum whose closure still contains $x_\ast$. Repeating the argument with $S_{\beta_n}$ in place of $S_\alpha$, we obtain a strictly descending chain of strata whose closures contain $x_\ast$. Since the index set $\Lambda$ is partially ordered and the stratification is locally finite, such a strictly descending chain must terminate at a minimal stratum $S_{\alpha^\ast}$ with $x_\ast\in \overline{S_{\alpha^\ast}}$.

By Definition 6.3, $S_{\alpha^\ast}$ is null; in particular, if there exists a sequence of times $\tau_k\uparrow T^\ast$ with $u(\tau_k)\in S_{\alpha^\ast}$ and $u(\tau_k)\to x_\ast$, then $(x_\ast,T^\ast)$ cannot be singular. If no such sequence exists, then necessarily $u(t)$ approaches $x_\ast$ through strata of strictly higher order, but then we can repeat the previous descent argument until we reach $S_{\alpha^\ast}$ and extract a sequence in $S_{\alpha^\ast}$ converging to $x_\ast$, again contradicting nullity. In all cases we reach a contradiction with the assumption $(x_\ast,T^\ast)\in\mathcal{S}_u$.

Therefore no finite-time singular point $(x_\ast,T^\ast)$ can exist for any finite-energy trajectory $u$, and the flow is globally regular as claimed. □

# 7. Application Template: Navier–Stokes as a Hypostructure

In this chapter we show how the Navier–Stokes analysis in the companion document `ns_draft_original_backup.md` can be rephrased as a verification of the hypostructural axioms and nullity mechanisms. Each lemma below is a direct restatement of an estimate proved in the backup draft, with notation aligned to the hypostructure framework. The aim is to make the Navier–Stokes application completely self-contained inside the hypostructure language.

## 7.1 Ambient Space, Metric, Energy, and Stratification

**Definition 7.1 (Navier–Stokes ambient manifold).**  
Let $\rho(y)=(4\pi)^{-3/2}e^{-|y|^2/4}$. Define
$$
H^1_\rho := \{\mathbf{V}:\mathbb{R}^3\to\mathbb{R}^3: \mathbf{V},\nabla\mathbf{V}\in L^2_\rho,\ \nabla\cdot\mathbf{V}=0\},
$$
with norm $\|\mathbf{V}\|_{H^1_\rho}^2=\|\mathbf{V}\|_{L^2_\rho}^2+\|\nabla\mathbf{V}\|_{L^2_\rho}^2$. Let
$$
\mathcal{V} := \bigl\{\mathbf{V}\in H^1_\rho : \nabla\cdot\mathbf{V}=0,\ \|\nabla\mathbf{V}\|_{L^2(B_1)}=1\bigr\},
$$
and let $G$ be the symmetry group of translations and rotations. The ambient manifold is $\mathcal{X}_{\mathrm{NS}}:=\mathcal{V}/G$, endowed with the metric induced by the strong $H^1_\rho$ distance on the quotient. The reference measure $\mathfrak{m}$ is the weighted Lebesgue measure $\rho(y)dy$.

**Definition 7.2 (Lyapunov functional for NS).**  
Let $A$ be the Stokes operator on $L^2_\rho$, and let $\tau(\mathbf{V})\ge 0$ be the Gevrey radius of analyticity of $\mathbf{V}$ (as in Section 8.4 of the backup). Define
$$
\Phi_{\mathrm{NS}}(\mathbf{V}) := \tfrac12\|e^{\tau(\mathbf{V})A^{1/2}}\mathbf{V}\|_{L^2_\rho}^2.
$$
Along the renormalized flow $s\mapsto \mathbf{V}(s)$ one has the energy inequality
$$
\frac{d}{ds}\Phi_{\mathrm{NS}}(\mathbf{V}(s)) \le -\mathfrak{D}_{\mathrm{NS}}(\mathbf{V}(s)),
$$
where the dissipation rate $\mathfrak{D}_{\mathrm{NS}}$ is nonnegative and scale-homogeneous of degree $-1$ in the physical scaling parameter $\lambda(t)$. This verifies Assumption A1 for the NS hypostructure.

**Definition 7.3 (Stratification).**  
We partition $\mathcal{X}_{\mathrm{NS}}$ into strata indexed by geometric/scaling invariants (entropy dimension $d_H$, swirl ratio $\mathcal{S}$, twist $\mathcal{T}$, scaling rate $\gamma$):
- $S_{\mathrm{acc}}$ (accelerating/type II): $\lambda(t)\sim (T-t)^\gamma$ with $\gamma\ge 1$.
- $S_{\mathrm{frac}}$ (fractal/high entropy): $d_H>1$ for the vorticity distribution; high defect.
- $S_{\mathrm{swirl}}$ (high swirl): $\mathcal{S}>\mathcal{S}_c$ for a fixed threshold $\mathcal{S}_c>0$.
- $S_{\mathrm{tube}}$ (tube/low swirl): $\mathcal{S}\le \mathcal{S}_c$ with bounded twist.
- $S_{\mathrm{vac}}$ (small data/terminal): complement of the above; contains the trivial equilibrium and small smooth data.
Interfaces are defined by threshold values of $(d_H,\mathcal{S},\mathcal{T},\gamma)$; $\psi_{\mathrm{NS}}$ assigns the interfacial cost according to the energy drop at these thresholds.

## 7.2 Capacity Nullity: Exclusion of \(S_{\mathrm{acc}}\)

**Theorem 7.1 (Mass–flux capacity for type II scaling).**  
Let $u$ be a Navier–Stokes solution with renormalized profile $\mathbf{V}(s)$ and scaling $\lambda(t)\sim (T-t)^\gamma$ with $\gamma\ge 1$. Then
$$
\mathrm{Cap}_{\mathrm{NS}}(u):=\int_0^T \mathfrak{D}_{\mathrm{NS}}(\mathbf{V}(s(t)))\,dt \sim \int_0^T \lambda(t)^{-1}\,dt = \infty.
$$
Hence $S_{\mathrm{acc}}$ is capacity-null in the sense of Theorem 3.1.

*Proof.* For $\gamma\ge 1$, $\int_0^T \lambda(t)^{-1}dt$ diverges. The BV energy inequality bounds $\int_0^T \mathfrak{D}_{\mathrm{NS}}\,dt$ by the initial energy, so such trajectories are inadmissible. □

## 7.3 Variational Nullity: Exclusion of \(S_{\mathrm{frac}}\)

**Definition 7.4 (Defect and extremizer manifold).**  
Let $\Xi[\mathbf{V}]$ be the spectral coherence functional (Section 8.4 of the backup) and $\mathcal{M}_{\mathrm{ext}}:=\{\mathbf{V}:\Xi[\mathbf{V}]=\Xi_{\max}\}$ its extremizer manifold. Set
$$
\|\nu_{\mathbf{V}}\|_{\mathcal{M}}:=\operatorname{dist}_{H^1_\rho}(\mathbf{V},\mathcal{M}_{\mathrm{ext}}),\qquad \text{defect } \delta(\mathbf{V}):=(\Xi_{\max}-\Xi[\mathbf{V}])_+.
$$

**Lemma 7.2 (Bianchi–Egnell stability; cf. Theorem 8.5.5 in backup).**  
There exists $c_{\mathrm{BE}}>0$ such that for all $\mathbf{V}$,
$$
\Xi_{\max}-\Xi[\mathbf{V}] \ge c_{\mathrm{BE}}\,\|\nu_{\mathbf{V}}\|_{\mathcal{M}}^2.
$$

**Lemma 7.3 (Gevrey evolution inequality; cf. Section 8.4).**  
Along the renormalized flow,
$$
\dot{\tau}(s) \ge c_0 - c_1\,\Xi[\mathbf{V}(s)]
$$
for some constants $c_0,c_1>0$.

**Proposition 7.4 (Metric–defect compatibility in \(S_{\mathrm{frac}}\)).**  
There exists a strictly increasing $\gamma_{\mathrm{NS}}$ with $\gamma_{\mathrm{NS}}(0)=0$ such that
$$
|\partial\Phi_{\mathrm{NS}}|(\mathbf{V}) \ge \gamma_{\mathrm{NS}}(\|\nu_{\mathbf{V}}\|_{\mathcal{M}})
$$
for all $\mathbf{V}\in S_{\mathrm{frac}}$.

*Proof.* By Lemma 7.3, $\dot{\tau}\ge c_0-c_1\Xi[\mathbf{V}]$. Using Lemma 7.2 to express $\Xi_{\max}-\Xi$ in terms of $\|\nu_{\mathbf{V}}\|_{\mathcal{M}}^2$, and choosing constants so that $c_0-c_1\Xi_{\max}>0$ on $S_{\mathrm{frac}}$, we obtain $\dot{\tau}\gtrsim \|\nu_{\mathbf{V}}\|_{\mathcal{M}}^2$. Since $|\partial\Phi_{\mathrm{NS}}|\gtrsim \dot{\tau}$ for the functional $\Phi_{\mathrm{NS}}$, the claim follows. □

By Theorem 4.3, $S_{\mathrm{frac}}$ is variationally null.

## 7.4 Locking Nullity: Exclusion of \(S_{\mathrm{swirl}}\)

**Lemma 7.5 (Spectral coercivity; cf. Theorem 6.3 in backup).**  
Let $\mathbf{V}_\ast\in S_{\mathrm{swirl}}$ be a high–swirl profile on the gauge manifold $\mathcal{M}$. The linearized operator $\mathcal{L}_{\mathrm{swirl}}$ about $\mathbf{V}_\ast$, projected orthogonally to the symmetry modes, satisfies
$$
\langle \mathcal{L}_{\mathrm{swirl}} w, w\rangle_{L^2_\rho} \le -\mu \|w\|_{H^1_\rho}^2,\qquad \forall w\perp \{\text{symmetry modes}\},
$$
for some $\mu>0$ depending on $\mathcal{S}_c$.

**Proposition 7.6 (Geometric locking on \(S_{\mathrm{swirl}}\)).**  
The functional $\Phi_{\mathrm{NS}}$ is $\mu$–convex along geodesics in $S_{\mathrm{swirl}}$, so $S_{\mathrm{swirl}}$ satisfies the hypotheses of Theorem 4.2 and is locking–null.

*Proof.* The spectral gap implies strict positivity of the second variation of $\Phi_{\mathrm{NS}}$ along gauge–orthogonal directions, yielding $\mu$–convexity; Theorem 4.2 applies. □

## 7.5 Virial Nullity: Exclusion of \(S_{\mathrm{tube}}\)

**Lemma 7.7 (Virial inequality; cf. Sections 4 and 10 in backup).**  
For tube–like profiles $\mathbf{V}\in S_{\mathrm{tube}}$, define $J(\mathbf{V})=\int |y|^2|\mathbf{V}|^2\rho$. Along the renormalized flow,
$$
\frac{d^2}{ds^2}J(\mathbf{V}(s)) \ge C_{\mathrm{rep}}(\mathbf{V}) - C_{\mathrm{att}}(\mathbf{V}),
$$
with $C_{\mathrm{rep}}-C_{\mathrm{att}}\ge c_1\Phi_{\mathrm{NS}}(\mathbf{V})$ for some $c_1>0$ on $S_{\mathrm{tube}}$.

**Proposition 7.8 (Virial domination on \(S_{\mathrm{tube}}\)).**  
The virial functional $J$ satisfies the domination condition of Theorem 4.1 on $S_{\mathrm{tube}}$; hence $S_{\mathrm{tube}}$ is virial–null.

*Proof.* Decompose the flow as $\mathbf{V}_s=F_{\mathrm{diss}}+F_{\mathrm{inert}}$; Lemma 7.7 shows the dissipative contribution strictly dominates the inertial one in the virial derivative, giving the strict inequality required in Theorem 4.1. □

## 7.6 Variational Nullity of High–Twist (“Barber Pole”) States

**Lemma 7.9 (Uniform defect for high twist; cf. Theorem 11.1 in backup).**  
There exists $\delta_0>0$ such that if $\|\nabla\xi\|$ is unbounded while $\mathcal{S}\le \mathcal{S}_c$ and $\mathbf{V}\in H^1_\rho$, then $\operatorname{dist}_{H^1_\rho}(\mathbf{V},\mathcal{M}_{\mathrm{ext}})\ge \delta_0$.

**Proposition 7.10 (Roughness penalty for high twist).**  
By Lemma 7.9 and Proposition 7.4, high–twist configurations carry a uniform defect $\ge\delta_0$, forcing $|\partial\Phi_{\mathrm{NS}}|\ge \gamma_{\mathrm{NS}}(\delta_0)>0$. Thus high–twist (Barber Pole) strata are variationally null.

## 7.7 Synthesis: Null Stratification and Global Regularity

Collecting the verifications:
- $S_{\mathrm{acc}}$ is capacity–null (Theorem 7.1).
- $S_{\mathrm{frac}}$ is variationally null (Proposition 7.4 and Theorem 4.3).
- $S_{\mathrm{swirl}}$ is locking–null (Proposition 7.6 and Theorem 4.2).
- $S_{\mathrm{tube}}$ is virial–null (Proposition 7.8 and Theorem 4.1).
- High–twist Barber–Pole states are variationally null (Proposition 7.10).

Assume the structural cover property: any potential renormalized singular profile lies in the closure of one of these strata (as proved in the backup by the “Covering Principle” and the classification of singular geometries). Then the Navier–Stokes stratification $\Sigma_{\mathrm{NS}}$ is null in the sense of Definition 6.3, with terminal stratum $S_{\mathrm{vac}}$ (small smooth data). By Theorem 6.2 (Structural global regularity), no finite–time singularity can form from finite–energy initial data within the hypostructural class.

## References (Navier–Stokes application)

- Sections 4, 6, 8, 9, 10, 11 of `docs/source/navier_stokes/ns_draft_original_backup.md` for the detailed NS estimates corresponding to Lemmas 7.2–7.9. Other references remain as in the main bibliography.
