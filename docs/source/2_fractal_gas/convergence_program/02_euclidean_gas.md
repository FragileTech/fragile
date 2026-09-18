# Euclidean Gas: Canonical Transition and Operator Estimates

(sec-eg-tldr)=
## 0. TLDR

:::{div} feynman-prose
The Euclidean Gas advances a complete marked population through one fixed sequence: sampled measurements, frozen acceptance decisions, connected-component collisions, BAOAB, position diffusion, smooth velocity compression, and terminal boundary classification. One shared orthogonal matrix rotates all relative velocities in a component. This gives exact component momentum conservation and multiplies its relative kinetic energy by the square of the restitution coefficient.

The comparison coordinates are bounded; the physical coordinates are retained. Dead slots use those retained coordinates when choosing revival donors and entering collisions. The primary population description is the discrete map $\mu_{n+1}=\mathcal F_h(\mu_n)$ derived in {doc}`08_mean_field`.
:::

(sec-eg-introduction)=
## 1. Introduction

:::{div} feynman-prose
Imagine stopping a simulation just before an update and writing down every random choice it is about to make. That list is the algorithm's transition law. A theory of this gas must average that same list: which companion was measured, which donor was selected, which edge was accepted, and which other walkers joined the collision.

Two details make a large difference. First, acceptance acts on sampled fitness. An average measurement substituted before the nonlinear gate changes the law. Second, an accepted recipient can itself be a donor. The connected components of all accepted edges specify the collision groups without overlapping writes. A shared rotation preserves the sum of relative velocities; independent rotations do not.

We define the complete transition, prove its component identities, and derive kinetic bounds directly from its BAOAB stages. The final position has an exact Gaussian conditional law, which makes terminal survival calculations particularly transparent. The complete finite-population kernel is permutation equivariant and Feller. The population and chaos proofs are in {doc}`08_mean_field` and {doc}`09_propagation_chaos`.
:::

```{mermaid}
flowchart LR
    S["Frozen marked swarm"] --> M["Sample measurements and retain fitness"]
    M --> G["Draw donors and acceptance gates"]
    G --> C["Accepted-edge components"]
    C --> J["Frozen position copies and jitter"]
    C --> R["One shared orthogonal matrix per component"]
    J --> K["BAOAB"]
    R --> K
    K --> X["Position diffusion"]
    X --> V["Smooth velocity cap"]
    V --> A["Terminal alive/dead marks"]
```

(sec-eg-framework-alignment)=
## 2. Physical states and comparison geometry

:::{div} feynman-prose
A bounded comparison feature is a way of measuring similarity, not a wall in physical space. Two distant positions may look similar after squashing, yet the force and the boundary still use their actual positions. We therefore keep both notions visible: the physical position–velocity metric for kinetic estimates, and the squashed metric for companion weights and bounded measurements.

The framework's deterministic aggregation estimates can be used when their stated input bounds hold. Its independent-output arguments cannot be applied to the shared component collision. Our kernel proof below conditions on the whole accepted graph, and the population proof controls the graph seen by a tagged walker.
:::

(sec-eg-definition)=
## 3. Definition: the Euclidean Gas

A **Euclidean Gas** is the Fragile Swarm $\mathcal F_{\text{EG}}$ given by the tuple of environmental structures, parameters, operators, and noise measures below. It induces a Markov chain on the swarm state space via the **Fragile Gas Algorithm** $\mathcal{S}_{t+1}\!\sim\!\Psi_{\mathcal F_{\text{EG}}}(\mathcal S_t,\cdot)$ (Def. *Fragile Gas Algorithm* ({prf:ref}`def-fragile-gas-algorithm`)). The canonical measurement pipeline uses global population variance with a positive quadratic regularization floor, followed by positive logistic rescaling. The detailed formulas below distinguish this choice from the explicitly scoped auxiliary standardization estimates.

### 3.1 Euclidean Gas algorithm (canonical pipeline)

:::{div} feynman-prose
Freeze a photograph of the swarm at the beginning of the step. Every measurement, donor choice, and acceptance decision refers to this photograph. Draw an edge from each accepted recipient to its donor. If two collision groups overlap, they are one connected group: a walker has exactly one velocity destination. Rotate the whole group's relative velocities with the same matrix. The vectors then still add to zero, which is the reason momentum is conserved.

After cloning, advance all slots through the kinetic stages and check the boundary at the end. A revived slot joins the collision using its retained input velocity. Its position comes from its selected donor, but its momentum has not been replaced by a second copy of the donor's momentum.
:::

:::{prf:algorithm} Euclidean Gas Update
:label: alg-euclidean-gas

Given the complete marked swarm $S=((x_i,v_i,a_i))_{i=1}^N$, retain all coordinates, including those of dead slots.

1. If $M=\sum_i a_i=0$, return $S$ unchanged and stop the run.
2. Each alive row draws one measurement companion from the normalized Gaussian law on squashed phase-space distance, excluding itself when another alive row exists. Measure reward and the sampled separation, including the positive diversity floor, as specified in {ref}`sec-eg-stage2`.
3. Compute alive-population means and population variances. Standardize each channel using $\sqrt{\operatorname{Var}+\sigma_{\min}^2}$, apply its positive logistic map, and form the fitness product. Retain every realized sampled fitness throughout the cloning stage.
4. Each row draws one current cloning donor from the same distance-weighted law. For alive rows, accept with probability $\min\{1,[(V_{\mathrm{fit},j}-V_{\mathrm{fit},i})/(p_{\max}(V_{\mathrm{fit},i}+\varepsilon_{\mathrm{clone}}))]_+\}$. Accept every dead row whenever $M>0$. An alive singleton has no distinct donor and does not clone.
5. Form the undirected graph of accepted recipient–donor edges and compute its connected components. Copy accepted recipient positions from their frozen donors and add independent Gaussian jitter $\sigma_{\mathrm{clone}}\zeta_i$. For each nontrivial component $C$, independently draw $R_C$ from normalized Haar measure on $O(d)$ and set
   $$
   \bar v_C=\frac1{|C|}\sum_{i\in C}v_i,\qquad
   \widetilde v_i=\bar v_C+\alpha_{\mathrm{restitution}}R_C(v_i-\bar v_C).
   $$
   Uninvolved rows keep their input position and velocity. A donor's position changes only if that donor is itself an accepted recipient. Every slot is alive after this stage.
6. Apply the BAOAB map of {prf:ref}`def-eg-baoab-canonical` to each intermediate row, then independent Gaussian position diffusion and the smooth radial velocity cap. Use fresh independent kinetic noises across rows.
7. Apply terminal boundary classification once. Retain the positions and velocities of every terminally dead row in the output.

All coordinates used in Steps 2–5 are frozen input coordinates. In particular, velocity updates include donors and revived recipients, and no updated donor coordinate feeds another recipient during the same step.
:::

### 3.2 Compiled configuration and experiment contract

:::{prf:definition} Canonical Rust configuration
:label: def-eg-canonical-rust

`GasConfig::euclidean(d, h)` in the Rust `algorithmic-gas` library selects current-frame independent companions, Gaussian width $\epsilon_D=\epsilon_C=2$, squashing radii $R_x=R_v=2$, and phase-space weight $\lambda_v=1$. Both standardization floors are $0.1$, both logistic maps are $g(z)=2/(1+e^{-z})+0.1$, and both fitness exponents are $1$. The separation floor is $10^{-3}$; clone saturation is $p_{\max}=1$ and acceptance denominator floor is $10^{-6}$.

The collision has $\alpha_{\mathrm{restitution}}=0.5$ and $\sigma_{\mathrm{clone}}=0.1$. BAOAB uses unit mass, friction $1$, velocity diffusion factor $1$, final position diffusion $\sigma_x=0.1$, and velocity radius $V_{\mathrm{alg}}=2$. The absorbing box is $[-2,2]^d$, with terminal-only boundary classification. The caller supplies the objective and its potential gradient; the benchmark experiments use an explicit quadratic objective. Rust supports $1\le d\le256$ for the shared orthogonal sampler.

The mathematical formulas describe the real-arithmetic transition. Each interactive experiment executes the Rust engine and records its configuration, seed, accepted edges, component rotations, stage outputs, and terminal status. They do not run a separate Python trajectory implementation.

Uniform companions, uncapped kinetics, substep absorption, direct-copy cloning, historical donors, and additional force terms are separately configured library extensions. Statements for the canonical configuration apply to an extension only after its changed transition and proof hypotheses have been checked.
:::


### 3.3 Position–velocity foundations and projection (Sasaki metric)

- **Physical position space** is $\mathbb R^d$. The canonical absorbing domain $D=\mathcal X_{\mathrm{valid}}$ is a closed box with nonempty interior. Its boundary has Lebesgue measure zero; smooth boundary is unnecessary here. Dead coordinates and intermediate Gaussian perturbations can lie outside $D$. The unbounded configuration takes $D=\mathbb R^d$ and requires physical moment and tail estimates separately.
- **Velocity radius** $V_{\mathrm{alg}}\in(0,\infty)$ and **velocity cap** $\mathcal V_{\mathrm{alg}}:=\{v\in\mathbb R^d:\|v\|\le V_{\mathrm{alg}}\}$.
- **Positional radius** $R_x\in(0,\infty)$, which sets the characteristic scale of the bounded algorithmic position space.
- **Walker state** $w_i=(x_i,v_i,s_i)\in\mathcal X\times\mathbb R^d\times\{0,1\}$ collects position, velocity, and status.
- **Algorithmic space and Sasaki metric** $(\mathcal Y,d_{\mathcal Y}^{\mathrm{Sasaki}})$ where the algorithmic space is the ({prf:ref}`def-algorithmic-space-generic`) closure of the projection image,

  $$
  \mathcal Y\;:=\;\overline{B(0,R_x)}\times\overline{B(0,V_{\mathrm{alg}})}\subset\mathbb R^d\times\mathbb R^d,

  $$

  endowed with the Sasaki metric

  $$
  d_{\mathcal Y}^{\mathrm{Sasaki}}\bigl((y_x,y_v),(y'_x,y'_v)\bigr)^2:=\|y_x-y'_x\|^2+\lambda_v\|y_v-y'_v\|^2

  $$
  for some fixed weight $\lambda_v>0$. For physical states we write $y=\varphi(x,v)$ and $y'=\varphi(x',v')$; the metric therefore measures differences between squashed coordinates. The compactness of $\mathcal Y$ ensures a finite algorithmic diameter.
- **Projection** $\varphi:\mathbb R^d\times\mathbb R^d\to B(0,R_x)\times B(0,V_{\mathrm{alg}})$ given by $\varphi(x,v)=(\psi_x(x),\psi_v(v))$ with the smooth squashing maps

  $$
  \psi_x(x)\ :=\ R_x\,\frac{x}{R_x+\|x\|},\qquad
  \psi_v(v)\ :=\ V_{\mathrm{alg}}\,\frac{v}{V_{\mathrm{alg}}+\|v\|}.

  $$

  ::: {admonition} Design Note
  :class: tip
  Each squashing map is $C^1$ globally and $C^\infty$ away from the origin. Its image is the open ball, so the final smooth cap does not create an atom on the velocity sphere. Minorization and any further continuum identification must use this particular map.
  :::

  The projection $\varphi$ maps the physical state space $\mathbb R^d\times\mathbb R^d$ into the bounded product $B(0,R_x)\times B(0,V_{\mathrm{alg}})$. Its image has compact closure $\mathcal Y$, so the **Axiom of Bounded Algorithmic Diameter** ({prf:ref}`axiom-bounded-algorithmic-diameter`) holds by construction. Lemma {prf:ref}`lem-squashing-properties-generic` shows that each squashing map is $1$-Lipschitz, and Lemma {prf:ref}`lem-projection-lipschitz` extends this to $\varphi$ under the Sasaki metric.

- **Algorithmic distance for companion selection ({prf:ref}`def-alg-distance`).** For intra-swarm measurements (companion selection for diversity and cloning), the algorithm uses the **algorithmic distance** between two walkers $i$ and $j$:

  $$
  d_{\text{alg}}(i,j)^2 := \|\psi_x(x_i)-\psi_x(x_j)\|^2 + \lambda_{\text{alg}}\|\psi_v(v_i)-\psi_v(v_j)\|^2

  $$

  where $\lambda_{\text{alg}}=\lambda_v$ in the canonical configuration. Thus this is exactly the Sasaki distance between squashed features. The physical phase-space metric instead uses unsquashed coordinates; the two metrics have different quantitative bounds.

- **Reward** $R:\mathcal X_{\mathrm{valid}}\times\mathcal V_{\mathrm{alg}}\to\mathbb R$ couples the position potential with a kinetic regularizer:

  $$
  R(x,v):=R_{\mathrm{pos}}(x)-\lambda_{\mathrm{vel}}\|v\|^2,

  $$
  where $R_{\mathrm{pos}}=-U$ for a minimized objective $U$. The canonical preset permits $\lambda_{\mathrm{vel}}=0$; a positive kinetic regularizer is an additional objective choice. The acceleration $F=-\nabla U$ must be defined on all of $\mathbb R^d$, because jitter and BAOAB drift can leave $D$ before the terminal check. For the estimates below assume $F$ is globally Lipschitz with constant $L_F$, so $\|F(x)\|\le B_F+L_F\|x\|$, where $B_F=\|F(0)\|$. Bounded comparison features do not supply physical confinement.

::::{prf:lemma} Properties of smooth radial squashing maps
:label: lem-squashing-properties-generic

For any constant $C>0$ define $\psi_C: \mathbb R^d\to B(0,C)$ by $\psi_C(z):=C\,z/(C+\|z\|)$. The map $\psi_C$ satisfies:

1. $\psi_C$ is $1$-Lipschitz on $\mathbb R^d$.
2. $\psi_C\in C^{\infty}(\mathbb R^d\setminus\{0\})$.
3. $\psi_C(\mathbb R^d)\subset B(0,C)$.

```{dropdown} Proof
:::{prf:proof}
1. *Lipschitz continuity.* The Jacobian at $z\neq 0$ is

  $$
  D\psi_C(z)=\frac{C}{C+\|z\|}I-\frac{C}{(C+\|z\|)^2}\,\frac{z z^{\top}}{\|z\|}.

  $$
  Setting $\alpha := C/(C+\|z\|)$ and $\hat{z} := z/\|z\|$, this becomes $D\psi_C(z) = \alpha I - (\alpha^2\|z\|/C)\hat{z}\hat{z}^\top$. The eigenvalues are $\alpha$ (with multiplicity $d-1$, for directions perpendicular to $z$) and $\alpha - \alpha^2\|z\|/C = \alpha^2 = C^2/(C+\|z\|)^2$ (for the $z$ direction). Since $0 < \alpha < 1$ and $C^2/(C+\|z\|)^2 < \alpha$ for $\|z\| > 0$, the operator norm is $\|D\psi_C(z)\| = \alpha = C/(C+\|z\|) < 1$ for all $z\neq 0$. At $z=0$, $\psi_C$ is differentiable with $D\psi_C(0) = I$, so $\|D\psi_C(0)\| = 1$. The mean-value inequality then implies $\|\psi_C(z)-\psi_C(z')\|\le\|z-z'\|$ for all $z,z'\in\mathbb R^d$.

2. *Smoothness away from the origin.* For $z\neq 0$, $\psi_C$ is a composition of smooth functions: $z\mapsto\|z\|$, inversion on $(0,\infty)$, and scalar-vector multiplication. Hence $\psi_C\in C^{\infty}(\mathbb R^d\setminus\{0\})$.

3. *Image contained in the open ball.* For any $z\in\mathbb R^d$, $\|\psi_C(z)\| = C\,\|z\|/(C+\|z\|) < C$, so $\psi_C(z)$ lies in $B(0,C)$.

All three properties follow immediately.
:::
```
::::

Both the positional squashing map $\psi_x$ and the velocity squashing map $\psi_v$ are obtained by setting $C=R_x$ and $C=V_{\mathrm{alg}}$, respectively, so they inherit the 1-Lipschitz and smoothness properties of Lemma {prf:ref}`lem-squashing-properties-generic`.

::::{prf:lemma} Lipschitz continuity of the projection $\varphi$
:label: lem-projection-lipschitz

For $(x,v),(x',v')\in\mathbb R^d\times\mathbb R^d$ the projection $\varphi(x,v)=(\psi_x(x),\psi_v(v))$ satisfies

$$
d_{\mathcal Y}^{\mathrm{Sasaki}}\bigl(\varphi(x,v),\varphi(x',v')\bigr)\le\sqrt{\|x-x'\|^2+\lambda_v\|v-v'\|^2}.

$$

That is, $\varphi$ is $1$-Lipschitz when both domain and codomain carry the Sasaki metric with the same weight $\lambda_v$.

```{dropdown} Proof
:::{prf:proof}
Because $\psi_x$ and $\psi_v$ are $1$-Lipschitz (Lemma {prf:ref}`lem-squashing-properties-generic`),

$$
\|\psi_x(x)-\psi_x(x')\|\le\|x-x'\|,\qquad \|\psi_v(v)-\psi_v(v')\|\le\|v-v'\|.

$$

Therefore

$$
\begin{aligned}
d_{\mathcal Y}^{\mathrm{Sasaki}}\bigl(\varphi(x,v),\varphi(x',v')\bigr)^2
&=\|\psi_x(x)-\psi_x(x')\|^2+\lambda_v\|\psi_v(v)-\psi_v(v')\|^2\\
&\le\|x-x'\|^2+\lambda_v\|v-v'\|^2.
\end{aligned}

$$

Taking square roots gives the stated bound.
:::
```
::::

This bound compares projected and physical displacements. The physical kinetic moment estimate is proved in {prf:ref}`lem-euclidean-perturb-moment`.

The projection is injective on finite coordinates and gives bounded comparison features. Its inverse is not uniformly Lipschitz near the boundary of the feature image. Consequently physical moment estimates and local changes of metric remain separate parts of the analysis.

### 3.4 Swarm distance and canonical operators

We measure dispersion in the Sasaki metric and retain the canonical aggregation pipeline:

- **Dispersion distance.** For swarms $\mathcal S_1,\mathcal S_2$ write

  $$
  d_{\mathrm{Disp},\mathcal Y}^{\mathrm{Sasaki}}(\mathcal S_1,\mathcal S_2)^2
  := \frac{1}{N}\sum_{i=1}^{N} d_{\mathcal Y}^{\mathrm{Sasaki}}\big(\varphi(x_{1,i},v_{1,i}),\varphi(x_{2,i},v_{2,i})\big)^2
  + \frac{\lambda_{\mathrm{status}}}{N}\sum_{i=1}^{N}(s_{1,i}-s_{2,i})^2,

  $$
  with status penalty $\lambda_{\mathrm{status}}>0$ as in the canonical framework. Because the Sasaki metric adds a velocity term, Section 4.3 re-validates every deterministic Lipschitz bound against $d_{\mathrm{Disp},\mathcal Y}^{\mathrm{Sasaki}}$.

  :::{admonition} Distinction: Algorithmic Distance vs. Sasaki Metric
  :class: note

  It is critical to distinguish two different distance metrics used in this document:

  1. **Algorithmic distance ({prf:ref}`def-alg-distance`)** $d_{\text{alg}}(i,j)$: Used by the *algorithm itself* for intra-swarm companion selection (diversity measurement and cloning). This defines how the algorithm "perceives" proximity between walkers within the same swarm.

  2. **Sasaki metric** $d_{\mathcal Y}^{\mathrm{Sasaki}}$: Used by the *analysis* to measure inter-swarm dispersion and derive continuity bounds. This is an analytical tool for proving convergence properties.

  For the Euclidean Gas, we set $\lambda_{\text{alg}} = \lambda_v$ so that these metrics coincide in their functional form, simplifying the connection between algorithmic behavior and analytical properties. However, they serve conceptually different roles: the algorithmic distance is intrinsic to the algorithm's design, while the Sasaki metric is extrinsic to the convergence analysis.
  :::
- **Walkers:** $N\ge1$, with the explicit singleton convention; the empirical reward and distance aggregators ({prf:ref}`def-swarm-aggregation-operator-axiomatic`) keep their canonical formulas. Lemma {prf:ref}`lem-sasaki-aggregator-lipschitz` supplies Sasaki-specific error moduli, and Lemma {prf:ref}`lem-sasaki-standardization-lipschitz` applies them to the regularized standard deviation and logistic rescale operators.
- **Dynamics weights:** $\alpha,\beta\ge 0$ with $\alpha+\beta>0$. The preset takes both exponents equal to one; any stronger amplification inequality used by a convergence theorem is an additional condition.

### 3.5 Kinetic Langevin perturbations with velocity capping

:::{div} feynman-prose
Keep track of where the force is evaluated. The first half-kick uses the post-cloning position. The second uses the position after the second drift. Then position diffusion moves the walker once more, and the velocity cap acts once. Reordering any of these operations changes the transition law.

The cap is smooth radial compression: even a small nonzero velocity is reduced. It is therefore an order-one operation per update when its radius is held fixed. We study the population limit at fixed step size before asking whether any differential equation describes a further limit.
:::

:::{prf:definition} BAOAB, position diffusion, smooth cap, and terminal classification
:label: def-eg-baoab-canonical

Write $h=\tau>0$, $\gamma\ge0$, $F=-\nabla U$, and
$$
c=e^{-\gamma h},\qquad
q^2=\sigma_v^2\begin{cases}(1-e^{-2\gamma h})/(2\gamma),&\gamma>0,\\ h,&\gamma=0.\end{cases}
$$
From the post-collision row $(x,v)$ draw independent standard Gaussian vectors $\xi_v,\xi_x$ and apply
$$
\begin{aligned}
v_1&=v+\tfrac h2F(x), &x_1&=x+\tfrac h2v_1,\\
v_2&=c v_1+q\xi_v, &x_2&=x_1+\tfrac h2v_2,\\
v_3&=v_2+\tfrac h2F(x_2), &x^+&=x_2+\sigma_x\sqrt h\xi_x,\\
v^+&=\psi_v(v_3)=\frac{V_{\mathrm{alg}}v_3}{V_{\mathrm{alg}}+\|v_3\|},
&a^+&=\mathbf1_D(x^+).
\end{aligned}
$$
The force is an acceleration; a nonunit physical mass is incorporated into $F$. The canonical thermostat is centered at zero. No boundary operation occurs at the clone, B, A, or O stages. The cap changes velocities only, and terminal absorption changes the mark only. In particular, $\|v^+\|<V_{\mathrm{alg}}$ for every finite input.

Clone jitter uses the separate parameter $\sigma_{\mathrm{clone}}$ and independent $\zeta_i$ for every accepted row, including revived rows. These jitters and the kinetic noises are independent across rows and independent of the graph. Rotations are independent across components, with one rotation shared inside each component. Conditional collision outputs are consequently correlated.
:::

:::{prf:remark} Continuous-time interpretation
:label: remark-eg-fixed-step-kinetics

BAOAB discretizes underdamped Langevin dynamics before the final diffusion, cap, and selection operations are composed with it. The complete canonical kernel is the composition just defined. Its stationary law is not asserted to be a Gibbs law. Holding the cap radius and order-one cloning rule fixed while $h\downarrow0$ need not yield a finite continuous-time generator; {doc}`08_mean_field` analyzes this identification using the actual one-step map.
:::


(sec-eg-operator-estimates)=
## 4. Axiom-by-axiom validation (Sasaki formulation)

The estimates in this section state the transition and metric to which they apply. The physical phase-space metric is
$$
d_{\mathrm{phys}}((x,v),(x',v'))^2=\|x-x'\|^2+\lambda_v\|v-v'\|^2.
$$
The squashed comparison metric has the same topology on finite physical states, but a global physical Lipschitz estimate cannot be inferred from bounded feature distance. On each compact physical set the inverse squashing map is Lipschitz, so local constants transfer between these metrics.

### 4.1 Revival and terminal survival

:::{prf:lemma} Scheduled revival
:label: lem-eg-scheduled-revival

If at least one slot is alive, every dead slot draws an eligible donor using its retained position and velocity in the Gaussian weights and is accepted with probability one. Thus the post-cloning population has $N$ alive slots. If no slot is alive, the swarm is absorbing. This conclusion does not require an inequality relating the fitness floor to the clone acceptance denominator.

*Proof.* The Gaussian donor weights are positive for all finite coordinates. Their sum over the nonempty alive pool is positive. The dead-row branch of the cloning rule accepts the selected donor deterministically. The component transform and jitter are then applied before any terminal classification. $\square$
:::


:::{prf:lemma} Lipschitz property of the kinetic position map
:label: lem-sasaki-kinetic-lipschitz

For the canonical kinetic step, let $b=h(1+c)/2$ and
$$
M_h(x,v)=x+b\bigl(v+\tfrac h2F(x)\bigr),\qquad
s_h^2=\tfrac{h^2q^2}{4}+h\sigma_x^2.
$$
Then $x^+$ has law $\mathcal N(M_h(x,v),s_h^2 I_d)$, and under identical innovations
$$
\|x^+-x'^+\|\le L_{\mathrm{flow}}d_{\mathrm{phys}}((x,v),(x',v')),
\qquad L_{\mathrm{flow}}=1+\tfrac{bhL_F}{2}+\frac b{\sqrt{\lambda_v}}.
$$

*Proof.* Substitute $v_2=cv_1+q\xi_v$ into $x_2=x+h(v_1+v_2)/2$. This gives $x^+=M_h+(hq/2)\xi_v+\sigma_x\sqrt h\xi_x$. The noises are independent and their covariance is $s_h^2I_d$. In a synchronous coupling they cancel. The Lipschitz bound on $F$ and the two coordinate bounds supplied by $d_{\mathrm{phys}}$ give the result. The B2 kick and cap do not change $x^+$. $\square$
:::

:::{prf:lemma} Lipschitz continuity of the death probability
:label: lem-euclidean-boundary-holder

For any Borel domain $D$, define $p_{\mathrm{dead}}(x,v)=\mathbb P(x^+\notin D)$. If $s_h>0$, then
$$
|p_{\mathrm{dead}}(x,v)-p_{\mathrm{dead}}(x',v')|
\le \frac{L_{\mathrm{flow}}}{\sqrt{2\pi}s_h}\,
 d_{\mathrm{phys}}((x,v),(x',v')).
$$
For a domain with Lebesgue-null boundary, $\mathbb P(x^+\in\partial D)=0$.

*Proof.* Two Gaussians with covariance $s_h^2I$ and mean separation $r$ have total variation distance $2\Phi(r/(2s_h))-1$. To see this, divide their densities: the region where the first is larger is the half-space through their midpoint perpendicular to the mean difference. Integrating over that half-space reduces the distance to the displayed one-dimensional expression. Since $\Phi'$ is at most $1/\sqrt{2\pi}$, this distance is at most $r/(\sqrt{2\pi}s_h)$. Apply the bound to the exit event and then use {prf:ref}`lem-sasaki-kinetic-lipschitz`. The null-boundary assertion follows from the Gaussian density. $\square$
:::


3. **Finite algorithmic diameter.** Section 3.3 built $(\mathcal Y,d_{\mathcal Y}^{\mathrm{Sasaki}})$ from the capped velocities and showed that the projection $\varphi$ is $1$-Lipschitz. Consequently $\operatorname{diam}_{d_{\mathcal Y}^{\mathrm{Sasaki}}}(\mathcal Y)<\infty$, meeting the Axiom of Bounded Algorithmic Diameter ({prf:ref}`axiom-bounded-algorithmic-diameter`).

### 4.2 Environmental axioms

The physical state remains Euclidean, and the retained dead coordinates need not belong to the valid box. Reward statistics use alive rows. Global force growth, rather than a bound restricted to the alive box, controls intermediate kinetic states.

::::{prf:lemma} Reward regularity in the Sasaki metric
:label: lem-euclidean-reward-regularity

The reward function $R(x,v)=R_{\mathrm{pos}}(x)-\lambda_{\mathrm{vel}}\|v\|^2$ is continuous in physical coordinates and Lipschitz on each compact physical set. Expressed in squashed coordinates it is Lipschitz on the image of each such compact set. This gives the reward-regularity bound on the alive box with capped velocities. It does not assert a bounded continuous extension to the boundary of the full feature-space compactification.


```{dropdown} Proof
:::{prf:proof}
Let $\mathcal Y^{\circ}:=B(0,R_x)\times B(0,V_{\mathrm{alg}})$ be the image of the projection $\varphi:\mathbb R^d\times\mathbb R^d\to\mathcal Y^{\circ}$. For $y=(y_x,y_v)\in\mathcal Y^{\circ}$ the inverse mapping is explicit:

$$
\psi_C^{-1}(y)=\frac{y}{1-\|y\|/C}\qquad(\|y\|<C).

$$

Define $R_{\mathcal Y}:\mathcal Y^{\circ}\to\mathbb R$ by

$$
R_{\mathcal Y}(y):=R_{\mathrm{pos}}\big(\psi_{R_x}^{-1}(y_x)\big)-\lambda_{\mathrm{vel}}\,\big\|\psi_{V_{\mathrm{alg}}}^{-1}(y_v)\big\|^2.

$$

This is well defined because the squashing maps are bijections between $\mathbb R^d$ and the open balls $B(0,R_x)$ and $B(0,V_{\mathrm{alg}})$. The maps $\psi_{R_x}^{-1}$ and $\psi_{V_{\mathrm{alg}}}^{-1}$ are continuous on $\mathcal Y^{\circ}$, and the compositions with $R_{\mathrm{pos}}$ and the quadratic velocity term are continuous. Hence $R_{\mathcal Y}$ is continuous on $\mathcal Y^{\circ}$.

Because $R_{\mathcal Y}$ is continuous on $\mathcal Y^{\circ}$ and $\mathcal Y^{\circ}$ is bounded, the restriction of $R_{\mathcal Y}$ to any compact subset of $\mathcal Y^{\circ}$ is uniformly continuous. In particular, the walker positions belong to the compact valid domain $\mathcal X_{\mathrm{valid}}$, so the image $\varphi(\mathcal X_{\mathrm{valid}}\times\mathcal V_{\mathrm{alg}})$ is compact and $R_{\mathcal Y}$ is uniformly continuous (indeed, Lipschitz) on that set. Consequently the reward evaluated along the Sasaki projection is uniformly continuous, satisfying the reward-regularity axiom without invoking a global Lipschitz bound for $R_{\mathrm{pos}}$ on $\mathbb R^d$.
:::
```
::::

:::{prf:lemma} Reward variation and a quantitative richness condition
:label: lem-euclidean-richness

Let $\pi_B$ be the reference probability on a specified local region $B$ used in a richness assertion. Suppose two measurable subsets $B_1,B_2\subset B$ satisfy $\pi_B(B_i)\ge p_i>0$ and
$$
\inf_{z\in B_1,z'\in B_2}|R(z)-R(z')|\ge\Delta>0.
$$
Then $\operatorname{Var}_{\pi_B}R\ge p_1p_2\Delta^2$.

*Proof.* If $Z,Z'$ are independent with law $\pi_B$, then $\operatorname{Var}R=\frac12\mathbb E[(R(Z)-R(Z'))^2]$. The two ordered events $B_1\times B_2$ and $B_2\times B_1$ contribute at least $2p_1p_2\Delta^2$. $\square$

A nonzero kinetic penalty creates local reward variation, but a pair of distinct reward values alone does not provide the probability factors $p_1,p_2$. A uniform richness axiom requires these factors and the gap to be bounded uniformly over the specified regions and reference law. The canonical fixed-step population proofs use positive regularization floors and do not assume this additional richness statement. For a constant objective with zero kinetic penalty, reward variance is exactly zero while the regularized algorithm remains defined.
:::


### 4.3 Algorithmic & operator axioms

1. **Valid noise measure (kinetic perturbation).** Lemma {prf:ref}`lem-euclidean-perturb-moment` provides a quadratic-growth second-moment bound and the Feller property for the capped kinetic kernel.

:::{admonition} Non-compact moment interpretation
:class: note
On an unbounded domain we cannot demand a uniform moment bound. Instead, the kinetic axiom tracks the squared Sasaki increment through a Lyapunov-style control that grows at most quadratically in $\|x\|$ and $\|v\|$. The following lemma establishes this controlled growth together with the requisite Feller property.
:::

:::{prf:lemma} Perturbation second moment in physical phase space
:label: lem-euclidean-perturb-moment

For the canonical BAOAB step, let $b,s_h$ be as in {prf:ref}`lem-sasaki-kinetic-lipschitz`, and assume $\|F(x)\|\le B_F+L_F\|x\|$. Then
$$
\mathbb E d_{\mathrm{phys}}((x,v),(x^+,v^+))^2
\le C_x\|x\|^2+C_v\|v\|^2+C_0,
$$
where
$$
C_x=\frac{3b^2h^2L_F^2}{4},\qquad
C_v=3b^2+2\lambda_v,\qquad
C_0=\frac{3b^2h^2B_F^2}{4}+d s_h^2+2\lambda_v V_{\mathrm{alg}}^2.
$$
The unmarked kinetic kernel maps bounded continuous functions to bounded continuous functions.

*Proof.* The Gaussian position formula gives
$$
\mathbb E\|x^+-x\|^2=b^2\|v+\tfrac h2F(x)\|^2+d s_h^2
\le3b^2\|v\|^2+\tfrac{3b^2h^2}{4}(L_F^2\|x\|^2+B_F^2)+d s_h^2.
$$
The smooth cap gives $\|v^+-v\|^2\le2V_{\mathrm{alg}}^2+2\|v\|^2$. Sum these inequalities with velocity weight $\lambda_v$. Every BAOAB substep and the cap are continuous in the input for each fixed pair of innovations. Dominated convergence proves the asserted kernel continuity. The bound for projected displacement follows from the squashing maps' 1-Lipschitz property. $\square$
:::

:::{prf:lemma} Kinetic drift, positional covariance, and local phase-space nondegeneracy
:label: lem-euclidean-geometric-consistency

For the same transition,
$$
\mathbb E(x^+-x)=b(v+\tfrac h2F(x)),\qquad
\operatorname{Cov}(x^+)=s_h^2I_d.
$$
Hence the positional covariance condition number is exactly $1$, and the phase-space mean displacement has magnitude at most $\sqrt{C_x\|x\|^2+C_v\|v\|^2+C_0}$. The full position–velocity covariance is not generally isotropic.

If $F$ is $C^1$, $q>0$, $\sigma_x>0$, and $h^2L_F/4<1$, then the full physical phase-space covariance is positive definite. Its condition number is bounded on every compact set of inputs, with constants depending on that set and on $h$.

*Proof.* The first two identities follow directly from the Gaussian position formula; Jensen's inequality and {prf:ref}`lem-euclidean-perturb-moment` give the drift bound. For the final assertion, condition on $x_1$. The map from the O-stage velocity $w$ to the B2 velocity is
$$
T(w)=w+\tfrac h2F(x_1+\tfrac h2w).
$$
Its perturbation of the identity has Lipschitz constant $h^2L_F/4<1$. For every target $y$, the equation $w=y-\tfrac h2F(x_1+hw/2)$ has a unique solution by geometric convergence of this contraction. Its derivative is invertible, so $T$ is a $C^1$ bijection with continuous inverse. Composing with $\psi_v$ maps onto $B(0,V_{\mathrm{alg}})$ with a positive density. Independent final position noise then gives a positive joint density on $\mathbb R^d\times B(0,V_{\mathrm{alg}})$.

A nonzero linear functional cannot be constant on this open set, so its variance is positive. Covariance entries depend continuously on the initial state by Gaussian moment domination and the velocity bound. Compactness therefore supplies a positive minimum eigenvalue and a finite maximum eigenvalue on each compact input set. Their ratio is the stated local condition-number bound. $\square$

For the unit quadratic potential used by the canonical experiments, $F(x)=-x$ and $L_F=1$, so the positive-definiteness condition is verified whenever $0<h<2$, including the experiment default $h=0.04$. There is an exact obstruction at $h=2$: $x_2=x_1+v_2$ and $v_3=v_2-x_2=-x_1$, so the final velocity is deterministic conditional on the kinetic input, despite positive thermostat noise. Its velocity covariance is zero. The final position still has the displayed nondegenerate Gaussian law.

These are fixed-step drift and covariance statements. The mean force drift, the full covariance, and contraction of the selection–collision kernel are different quantities.
:::


3. **Auxiliary distance continuity.** The following inequalities isolate the geometric and scalar-array calculations used when their sampling law and input bounds apply.

#### 2.3.3 Continuity of the Expected Raw Distance Vector ($k \ge 2$ Regime)

:::{prf:remark} Scope of the auxiliary uniform-companion estimates
:label: remark-eg-uniform-estimates

The expected-distance estimates {prf:ref}`lem-sasaki-single-walker-structural-error` and {prf:ref}`thm-sasaki-distance-ms` below are for the explicitly defined extension with uniform independent companions. Their constants do not cover the canonical finite-width Gaussian donor probabilities. The deterministic aggregation and standardization inequalities remain usable for actual input arrays satisfying their stated bounds. Replacing a sampled distance by its expectation before nonlinear standardization or acceptance is not licensed by an expected-distance estimate.

For the canonical weighted law, use the normalized-kernel denominator bounds, marked measurement law, component exploration estimates, and one-step consistency proof in {doc}`08_mean_field` and {doc}`09_propagation_chaos`. Pairing without replacement is another configuration, not the canonical independent single-companion law.
:::


:::{prf:definition} Notation for auxiliary finite-swarm continuity estimates
:label: def-eg-auxiliary-continuity-notation

For two swarms $\mathcal S_r$ let $\mathcal A_r$ be their alive sets, $k_r=|\mathcal A_r|$, and $\mathcal A_{\mathrm{stable}}=\mathcal A_1\cap\mathcal A_2$. Set
$$
n_c(\mathcal S_1,\mathcal S_2)=\sum_{i=1}^N(s_{1,i}-s_{2,i})^2,\qquad
\Delta_{\mathrm{pos,Sasaki}}^2=
\sum_{i=1}^N d_{\mathcal Y}^{\mathrm{Sasaki}}(\varphi(w_{1,i}),\varphi(w_{2,i}))^2,
$$
and $D_{\mathcal Y}=\operatorname{diam}(\mathcal Y)$. Expected-distance statements below use unregularized comparison distance and the uniform-companion extension. The deterministic scalar-array statements assume their displayed uniform bound $V_{\max}$; physical reward comparisons are restricted to compact sets where the specified $L_R^{\mathrm{Sasaki}}$ is finite.

For the canonical global regularizer, the notation $\sigma_{\min,\mathrm{patch}}$ in these scalar-array inequalities means $\sigma_{\min}$: take $\kappa_{\mathrm{var,min}}=0$ and $\varepsilon_{\mathrm{std}}=\sigma_{\min}$. As a function of variance $t\ge0$, the scale $\sqrt{t+\sigma_{\min}^2}$ has derivative at most $1/(2\sigma_{\min})$. Thus its denominator bounds are available directly. Its application to sampled arrays still requires averaging after, rather than before, the nonlinear pipeline.
:::

::::{prf:lemma} Single-walker positional error bound in the Sasaki metric
:label: lem-sasaki-single-walker-positional-error

Let $\mathcal S_1$ and $\mathcal S_2$ be two swarm states. For a given walker $i$ that is alive in swarm $\mathcal S_1$ ($s_{1,i}=1$), let $\mathbb C_i(\mathcal S_1)$ be its companion selection measure.

The absolute error in its expected distance due to the positional displacement of the walkers between the two states, evaluated over the fixed companion set from $\mathcal S_1$, is bounded by the sum of its own displacement and the average displacement of its potential companions.

Referenced by {prf:ref}`lem-sasaki-total-squared-error-stable` and {prf:ref}`thm-euclidean-feller`.

$$
\left| \mathbb{E}_{c \sim \mathbb{C}_i(\mathcal{S}_1)} \left[ d_{\mathcal Y}^{\mathrm{Sasaki}}(\varphi(w_{1,i}), \varphi(w_{1,c})) \right] - \mathbb{E}_{c \sim \mathbb{C}_i(\mathcal{S}_1)} \left[ d_{\mathcal Y}^{\mathrm{Sasaki}}(\varphi(w_{2,i}), \varphi(w_{2,c})) \right] \right| \le d_{\mathcal Y}^{\mathrm{Sasaki}}(\varphi(w_{1,i}), \varphi(w_{2,i})) + \mathbb{E}_{c \sim \mathbb{C}_i(\mathcal{S}_1)} \left[ d_{\mathcal Y}^{\mathrm{Sasaki}}(\varphi(w_{1,c}), \varphi(w_{2,c})) \right]

$$

```{dropdown} Proof
:::{prf:proof}
Let $\Delta_{\mathrm{pos},i}$ denote the absolute error term we wish to bound. The proof proceeds by applying standard metric and probability inequalities.

**Step 1: Apply Linearity of Expectation.**
We combine the two terms into a single expectation over the fixed companion selection measure $\mathbb C_i(\mathcal S_1)$.

$$
\Delta_{\mathrm{pos},i} = \left| \mathbb{E}_{c \sim \mathbb{C}_i(\mathcal{S}_1)} \left[ d_{\mathcal Y}^{\mathrm{Sasaki}}(\varphi(w_{1,i}), \varphi(w_{1,c})) - d_{\mathcal Y}^{\mathrm{Sasaki}}(\varphi(w_{2,i}), \varphi(w_{2,c})) \right] \right|

$$

**Step 2: Apply Jensen's Inequality.**
Using Jensen's inequality for the convex function $f(x)=|x|$, we can move the absolute value inside the expectation, which provides an upper bound:

$$
\Delta_{\mathrm{pos},i} \le \mathbb{E}_{c \sim \mathbb{C}_i(\mathcal{S}_1)} \left[ \left| d_{\mathcal Y}^{\mathrm{Sasaki}}(\varphi(w_{1,i}), \varphi(w_{1,c})) - d_{\mathcal Y}^{\mathrm{Sasaki}}(\varphi(w_{2,i}), \varphi(w_{2,c})) \right| \right]

$$

**Step 3: Apply the Reverse Triangle Inequality.**
The term inside the expectation is the absolute difference between two distance values. For any points $a,b,c,d$ in a metric space $(M,d)$, the reverse triangle inequality states that $|d(a,b) - d(c,d)| \le d(a,c) + d(b,d)$. Applying this to the Sasaki metric $d_{\mathcal Y}^{\mathrm{Sasaki}}$ yields:

$$
\left| d_{\mathcal Y}^{\mathrm{Sasaki}}(\varphi(w_{1,i}), \varphi(w_{1,c})) - d_{\mathcal Y}^{\mathrm{Sasaki}}(\varphi(w_{2,i}), \varphi(w_{2,c})) \right| \le d_{\mathcal Y}^{\mathrm{Sasaki}}(\varphi(w_{1,i}), \varphi(w_{2,i})) + d_{\mathcal Y}^{\mathrm{Sasaki}}(\varphi(w_{1,c}), \varphi(w_{2,c}))

$$

**Step 4: Finalize the Bound.**
We substitute the inequality from Step 3 back into the expression from Step 2.

$$
\Delta_{\mathrm{pos},i} \le \mathbb{E}_{c \sim \mathbb{C}_i(\mathcal{S}_1)} \left[ d_{\mathcal Y}^{\mathrm{Sasaki}}(\varphi(w_{1,i}), \varphi(w_{2,i})) + d_{\mathcal Y}^{\mathrm{Sasaki}}(\varphi(w_{1,c}), \varphi(w_{2,c})) \right]

$$

By linearity of expectation, we can separate the terms. The first term, $d_{\mathcal Y}^{\mathrm{Sasaki}}(\varphi(w_{1,i}), \varphi(w_{2,i}))$, is a constant with respect to the expectation over the companion index $c$. This gives the final bound as stated in the lemma.

**Q.E.D.**
:::
```

::::
::::{prf:lemma} Single-walker structural error bound in the Sasaki metric
:label: lem-sasaki-single-walker-structural-error

Let $i\in\mathcal A_{\mathrm{stable}}$ and keep the second swarm's capped positions fixed. Let the initial swarm have at least two alive walkers, $k_1=|\mathcal A(\mathcal S_1)| \ge 2$. The absolute error in the expected distance for walker $i$ due to the change in the companion selection measure is bounded by:

Referenced by {prf:ref}`thm-sasaki-distance-ms`.

$$
\left| \mathbb{E}_{c \sim \mathbb{C}_i(\mathcal{S}_1)} \left[ d_{\mathcal Y}^{\mathrm{Sasaki}}(\varphi(w_{2,i}), \varphi(w_{2,c})) \right] - \mathbb{E}_{c \sim \mathbb{C}_i(\mathcal{S}_2)} \left[ d_{\mathcal Y}^{\mathrm{Sasaki}}(\varphi(w_{2,i}), \varphi(w_{2,c})) \right] \right| \le \frac{2 D_{\mathcal Y}}{k_1-1} \cdot n_c(\mathcal S_1, \mathcal S_2)

$$

where $D_{\mathcal Y}$ is the diameter of the algorithmic space.

```{dropdown} Proof
:::{prf:proof}
This result is a direct application of the framework's **Total Error Bound in Terms of Status Changes** ({prf:ref}`thm-total-error-status-bound`) to the specific function of interest in the Sasaki geometry.

**Step 1: Identify the Function and its Bound.**
Let the function being evaluated under the expectation be $f(c) := d_{\mathcal Y}^{\mathrm{Sasaki}}(\varphi(w_{2,i}), \varphi(w_{2,c}))$. This function measures the Sasaki distance from the fixed walker $i$ to a potential companion $c$, using the positions from the second swarm. By definition, any distance in the algorithmic space is bounded by the space's diameter, $D_{\mathcal Y}$. Therefore, we have a uniform bound on the function's value: $|f(c)| \le D_{\mathcal Y} =: M_f$.

**Step 2: Identify the Companion Support Sets.**
Let $S_1 = S_i(\mathcal{S}_1)$ and $S_2 = S_i(\mathcal{S}_2)$ be the companion support sets for walker $i$ in the two swarms. Since walker $i$ is alive in $\mathcal S_1$ (i.e., $i \in \mathcal A_{\mathrm{stable}} \subseteq \mathcal A_1$) and the precondition states $k_1 \ge 2$, the initial support set is $S_1 = \mathcal A_1 \setminus \{i\}$. Its size is therefore $|S_1| = k_1 - 1 > 0$.

**Step 3: Apply the General Error Bound.**
The framework theorem {prf:ref}`thm-total-error-status-bound` provides a general bound for the change in expectation of a bounded function due to a change in the underlying support set:

$$
\text{Error} \le \frac{2 M_f}{|S_1|} \cdot n_c(\mathcal S_1, \mathcal S_2)

$$
This bound is algebraic for uniform probability on the specified support sets and any bounded test function. Changing Gaussian weights on a fixed support requires an additional term.

**Step 4: Substitute and Finalize.**
We substitute our specific function bound $M_f = D_{\mathcal Y}$ and the support set size $|S_1| = k_1 - 1$ into the general formula. This immediately yields the stated bound for the structural error component.

**Q.E.D.**
:::
```



::::
::::{prf:lemma} Mean-square error on stable walkers (Sasaki)
:label: lem-sasaki-total-squared-error-stable

Let $\mathcal S_1,\mathcal S_2$ be swarms with alive sets $\mathcal A_r$ and let $\mathbf d^{(r)}$ denote the expected raw distance vector produced by the measurement operator on $\mathcal S_r$. Write $\mathcal A_{\mathrm{stable}}:=\mathcal A_1\cap\mathcal A_2$ and $k_{\mathrm{stable}}:=|\mathcal A_{\mathrm{stable}}|$. Then

$$
\sum_{i\in\mathcal A_{\mathrm{stable}}}\big|d^{(1)}_i-d^{(2)}_i\big|^2\le C_{\mathrm{pos}}^{\mathrm{Sasaki}}(k_1,k_{\mathrm{stable}})\,\Delta_{\mathrm{pos,Sasaki}}^2(\mathcal S_1,\mathcal S_2),

$$

where $C_{\mathrm{pos}}^{\mathrm{Sasaki}}(k_1,k_{\mathrm{stable}}):=2\Big(1+\frac{k_{\mathrm{stable}}}{\max\{1,k_1-1\}}\Big)$.

Referenced by {prf:ref}`thm-sasaki-distance-ms`.

```{dropdown} Proof
:::{prf:proof}
For $i\in\mathcal A_{\mathrm{stable}}$ set $\Delta_i:=|d^{(1)}_i-d^{(2)}_i|$. Lemma {prf:ref}`lem-sasaki-single-walker-positional-error` gives

$$
\Delta_i\le d_{\mathcal Y}^{\mathrm{Sasaki}}(\varphi(w_{1,i}),\varphi(w_{2,i})) + \mathbb E_{c\sim\mathbb C_i(\mathcal S_1)}\big[d_{\mathcal Y}^{\mathrm{Sasaki}}(\varphi(w_{1,c}),\varphi(w_{2,c}))\big].

$$

Apply $(a+b)^2\le 2a^2+2b^2$ and Jensen's inequality to obtain

$$
\Delta_i^2\le 2\,d_{\mathcal Y}^{\mathrm{Sasaki}}(\varphi(w_{1,i}),\varphi(w_{2,i}))^2 + \frac{2}{k_1-1}\sum_{j\in\mathcal A_1} d_{\mathcal Y}^{\mathrm{Sasaki}}(\varphi(w_{1,j}),\varphi(w_{2,j}))^2,

$$
where the averaging denominator $k_1-1$ is interpreted as $1$ when $k_1=1$. Summing over $i\in\mathcal A_{\mathrm{stable}}$ yields

$$
\sum_{i\in\mathcal A_{\mathrm{stable}}}\Delta_i^2\le 2\,\Delta_{\mathrm{pos,Sasaki}}^2(\mathcal S_1,\mathcal S_2)+\frac{2k_{\mathrm{stable}}}{\max\{1,k_1-1\}}\,\Delta_{\mathrm{pos,Sasaki}}^2(\mathcal S_1,\mathcal S_2),

$$
which is the claimed bound.
:::
```

::::
::::{prf:theorem} Mean-square continuity of the distance measurement (Sasaki)
:label: thm-sasaki-distance-ms

Let $\mathbf d^{(r)}$ be the expected raw distance vectors of swarms $\mathcal S_r$. With $k_{\min}:=\max\{1,\min(k_1,k_2)\}$, $k_{\mathrm{stable}}:=|\mathcal A_{\mathrm{stable}}|$, and alive-difference count $n_c:=\sum_{i=1}^N(s_{1,i}-s_{2,i})^2$, define

$$
F_{d,ms}^{\mathrm{Sasaki}}(\Delta_{\mathrm{pos}}^2,n_c):=C_{\mathrm{pos}}^{\mathrm{Sasaki}}(k_1,k_{\mathrm{stable}})\,\Delta_{\mathrm{pos}}^2+4k_{\mathrm{stable}}\frac{D_{\mathcal Y}^2}{\max\{1,k_1-1\}^2}\,n_c^2+D_{\mathcal Y}^2 n_c.

$$

Referenced by {prf:ref}`thm-euclidean-feller`.

Then

$$
\big\|\mathbf d^{(1)}-\mathbf d^{(2)}\big\|_2^2\le F_{d,ms}^{\mathrm{Sasaki}}\big(\Delta_{\mathrm{pos,Sasaki}}^2(\mathcal S_1,\mathcal S_2),n_c(\mathcal S_1,\mathcal S_2)\big).

$$

```{dropdown} Proof
:::{prf:proof}
Decompose the index set into stable walkers $\mathcal A_{\mathrm{stable}}$ and the complement. For stable walkers the bound in Lemma {prf:ref}`lem-sasaki-total-squared-error-stable` applies. For walkers whose status changes between the two swarms we use $|d^{(1)}_i-d^{(2)}_i|\le D_{\mathcal Y}$ because each expected distance is bounded by the diameter of the Sasaki algorithmic space. There are at most $n_c$ such indices (one per status change), contributing at most $D_{\mathcal Y}^2 n_c$ to the squared error.

Finally, the structural perturbation of the companion distribution for stable walkers is controlled by Lemma {prf:ref}`lem-sasaki-single-walker-structural-error`. Squaring its bound and summing over the $k_{\mathrm{stable}}$ indices yields the middle term in $F_{d,ms}^{\mathrm{Sasaki}}$. Adding the three contributions completes the proof.
:::
```
::::

4. **Non-degenerate noise.** The positive canonical velocity diffusion and final position diffusion give non-Dirac kinetic noise. Clone jitter has its own scale $\sigma_{\mathrm{clone}}$. Shared component rotations add collision randomness when relative velocities and restitution are nonzero; this randomness can be degenerate when either vanishes.

5. **Sufficient amplification.** The weights $\alpha,\beta\ge 0$ satisfy $\alpha+\beta>0$ exactly as in the canonical swarm ({prf:ref}`axiom-sufficient-amplification`).

6. **Aggregator axioms.** Let $R_{\max}:=\sup_{x\in\mathcal X}|R_{\mathrm{pos}}(x)|+\lambda_{\mathrm{vel}}V_{\mathrm{alg}}^2$ and recall from Lemma {prf:ref}`lem-euclidean-reward-regularity` that the reward satisfies the Lipschitz bound

$$
|R(x_1,v_1)-R(x_2,v_2)|\le L_R^{\mathrm{Sasaki}}\,d_{\mathcal Y}^{\mathrm{Sasaki}}\big((x_1,v_1),(x_2,v_2)\big),\qquad L_R^{\mathrm{Sasaki}}:=L_{\mathrm{pos}}+\frac{2\lambda_{\mathrm{vel}}V_{\mathrm{alg}}}{\sqrt{\lambda_v}}.

$$
Whenever aggregators act on reward vectors we use the uniform bound $V_{\mathrm{max}}^{(R)}:=\max\{|R_{\min}|,R_{\max}\}$; for distance vectors we use $V_{\mathrm{max}}^{(d)}:=D_{\mathcal Y}$. For swarms $\mathcal S_r$ write $k_r:=|\mathcal A(\mathcal S_r)|$, define $k_{\min}:=\max\{1,\min(k_1,k_2)\}$, and let $n_c:=\sum_{i=1}^N(s_{1,i}-s_{2,i})^2$ count the status changes.

::::{prf:lemma} Value continuity of the empirical moments
:label: lem-sasaki-aggregator-value

Fix a swarm $\mathcal S$ with alive index set $\mathcal A(\mathcal S)$ of size $k\ge 1$. Let $\mathbf v_1,\mathbf v_2\in\mathbb R^k$ be two scalar value vectors whose components satisfy $|v_{j,i}|\le V_{\max}$. Then the empirical mean and second moment obey

Referenced by {prf:ref}`lem-sasaki-aggregator-lipschitz` and {prf:ref}`lem-sasaki-mean-shift-bound-sq`.

$$
|\mu(\mathcal S,\mathbf v_1)-\mu(\mathcal S,\mathbf v_2)|\le \frac{1}{\sqrt{k}}\,\|\mathbf v_1-\mathbf v_2\|_2,\qquad|m_2(\mathcal S,\mathbf v_1)-m_2(\mathcal S,\mathbf v_2)|\le \frac{2V_{\max}}{\sqrt{k}}\,\|\mathbf v_1-\mathbf v_2\|_2.

$$

```{dropdown} Proof
:::{prf:proof}
The identities follow from the gradient calculations $\nabla\mu=(1/k)\mathbf 1$ and $\nabla m_2=(2/k)\mathbf v$ together with Cauchy–Schwarz, as in Lemma 6.2.2.a of the framework.

:::
```
::::

::::{prf:lemma} Structural continuity of the empirical moments
:label: lem-sasaki-aggregator-structural

Let $\mathcal S_r=((x_{r,i},v_{r,i},s_{r,i}))_{i=1}^N$ with alive counts $k_r\ge 1$ and let $\mathbf v$ be a scalar vector on the union of alive indices satisfying $|v_i|\le V_{\max}$. Set $k_{\min}:=\max\{1,\min(k_1,k_2)\}$ and $n_c:=\sum_{i=1}^N(s_{1,i}-s_{2,i})^2$. Then

Referenced by {prf:ref}`lem-sasaki-aggregator-lipschitz` and {prf:ref}`lem-sasaki-indirect-structural-error-sq`.

$$
|\mu(\mathcal S_1,\mathbf v)-\mu(\mathcal S_2,\mathbf v)|\le \frac{3V_{\max}}{k_{\min}}\,n_c,\qquad|m_2(\mathcal S_1,\mathbf v)-m_2(\mathcal S_2,\mathbf v)|\le \frac{3V_{\max}^2}{k_{\min}}\,n_c.

$$

```{dropdown} Proof
:::{prf:proof}
The proof mirrors Lemma 6.2.2.b of the framework. Decompose the difference in means into contributions from walkers that remain alive in both swarms and those that change status. The former vanish, whereas the latter introduce at most $V_{\max}$ per status flip. Accounting for the normalisation factors $1/k_r$ and the difference in alive counts yields the stated bounds. The argument for $m_2$ uses $|a^2-b^2|\le 2V_{\max}|a-b|$.

:::
```
::::

::::{prf:lemma} Lipschitz data for the Sasaki empirical aggregators
:label: lem-sasaki-aggregator-lipschitz

For reward vectors take $V_{\max}=V_{\mathrm{max}}^{(R)}$; for distance vectors take $V_{\max}=V_{\mathrm{max}}^{(d)}$. The empirical mean and second moment satisfy the aggregator axioms with

Referenced by {prf:ref}`lem-sasaki-mean-shift-bound-sq` and {prf:ref}`thm-euclidean-feller`.

$$
L_{\mu,M}^{\mathrm{Sasaki}}(k)=\frac{1}{\sqrt{k}},\qquad L_{m_2,M}^{\mathrm{Sasaki}}(k)=\frac{2V_{\max}}{\sqrt{k}},

$$

$$
L_{\mu,S}^{\mathrm{Sasaki}}(k_{\min})=\frac{3V_{\max}}{k_{\min}},\qquad L_{m_2,S}^{\mathrm{Sasaki}}(k_{\min})=\frac{3V_{\max}^2}{k_{\min}},

$$
and growth exponents $p_{\mu,S}=p_{m_2,S}=p_{\mathrm{worst\text{-}case}}=-1$. Consequently $\kappa_{\mathrm{var}}^{\mathrm{Sasaki}}=\kappa_{\mathrm{range}}^{\mathrm{Sasaki}}=1$ as in the canonical framework.

```{dropdown} Proof
:::{prf:proof}
Combine Lemmas {prf:ref}`lem-sasaki-aggregator-value` and {prf:ref}`lem-sasaki-aggregator-structural` with the dispersion metric identity $n_c\le\frac{N}{\lambda_{\mathrm{status}}}d_{\mathrm{Disp},\mathcal Y}^{\mathrm{Sasaki}}(\mathcal S_1,\mathcal S_2)^2$ to obtain the stated Lipschitz functions and exponents.
:::
```
::::

7. **Standardization & rescale continuity.** Let $\sigma_{\min,\mathrm{patch}}:=\sqrt{\kappa_{\mathrm{var,min}}+\varepsilon_{\mathrm{std}}^2}$ be the lower bound supplied by the regularized standard deviation operator, and denote by $L_{\sigma'_{\mathrm{patch}}}$ the global derivative bound from Lemma {prf:ref}`lem-sigma-patch-derivative-bound`. For notational compactness write

$$
L_{\sigma',M}^{\mathrm{Sasaki}}(k):=L_{\sigma'_{\mathrm{patch}}}\Big(L_{m_2,M}^{\mathrm{Sasaki}}(k)+2V_{\mathrm{max}}^{(R)}L_{\mu,M}^{\mathrm{Sasaki}}(k)\Big).

$$

:::{prf:definition} Standardization constants (Sasaki geometry)
:label: def-sasaki-standardization-constants

Let $\sigma_{\min,\mathrm{patch}}:=\sqrt{\kappa_{\mathrm{var,min}}+\varepsilon_{\mathrm{std}}^2}$ be the uniform lower bound on the regularized standard deviation, and let $L_{\sigma'_{\mathrm{patch}}}$ be its global Lipschitz constant from Lemma {prf:ref}`lem-sigma-patch-derivative-bound`.

#### Value Error Coefficients
The following coefficients bound the error in the standardization operator when the swarm structure is fixed but the raw values change due to positional displacement. They are notably independent of the number of alive walkers, `k`.

-   **Direct Shift Coefficient ($C_{V,\mathrm{direct}}$):** Bounding the error from the direct change in the raw value vector.

    $$
    C_{V,\mathrm{direct}} := \frac{1}{\sigma_{\min,\mathrm{patch}}}

    $$

-   **Mean Shift Coefficient ($C_{V,\mathrm{mean}}$):** Bounding the error from the resulting change in the empirical mean.

    $$
    C_{V,\mathrm{mean}} := \frac{1}{\sigma_{\min,\mathrm{patch}}}

    $$

-   **Denominator Shift Coefficient ($C_{V,\mathrm{denom}}$):** Bounding the error from the resulting change in the regularized standard deviation.

    $$
    C_{V,\mathrm{denom}} := \frac{8\big(V_{\mathrm{max}}^{(R)}\big)^2 L_{\sigma'_{\mathrm{patch}}}}{\sigma_{\min,\mathrm{patch}}^2}

    $$

-   **Total Value Error Coefficient (Linear Form) ($C_{V,\mathrm{total,lin}}^{\mathrm{Sasaki}}$):** The composite coefficient for the full (unsquared) Lipschitz bound on the value error, which aggregates the component-wise effects.

    $$
    C_{V,\mathrm{total,lin}}^{\mathrm{Sasaki}} := L_R^{\mathrm{Sasaki}} \left( C_{V,\mathrm{direct}} + C_{V,\mathrm{mean}} + C_{V,\mathrm{denom}} \right) = L_R^{\mathrm{Sasaki}} \left( \frac{2}{\sigma_{\min,\mathrm{patch}}} + \frac{8\big(V_{\mathrm{max}}^{(R)}\big)^2 L_{\sigma'_{\mathrm{patch}}}}{\sigma_{\min,\mathrm{patch}}^2} \right)

    $$

#### Structural Error Coefficients
The structural error coefficients, which are used in the subsequent theorem for structural continuity, remain as defined:

$$
C_{S,\mathrm{direct}}^{\mathrm{Sasaki}}(k_{\min}):=\frac{V_{\mathrm{max}}^{(R)}}{\sigma_{\min,\mathrm{patch}}}+\frac{2\big(V_{\mathrm{max}}^{(R)}\big)^2}{\sigma_{\min,\mathrm{patch}}^2},
\qquad C_{S,\mathrm{indirect}}^{\mathrm{Sasaki}}(k_{\min}):=\frac{3V_{\mathrm{max}}^{(R)}}{\sigma_{\min,\mathrm{patch}}k_{\min}}+\frac{6\big(V_{\mathrm{max}}^{(R)}\big)^2}{\sigma_{\min,\mathrm{patch}}^2k_{\min}}L_{\sigma',M}^{\mathrm{Sasaki}}(k_{\min}).

$$
:::

The squared coefficients used in the mean-square bounds are defined in {prf:ref}`def-sasaki-standardization-constants-sq`.


Set $C_R:=L_R^{\mathrm{Sasaki}}\sqrt{N}+R_{\max}\sqrt{\tfrac{N}{\lambda_{\mathrm{status}}}}$ for later use.

These constants verify the continuity axioms for the patched standardization and logistic rescale operators in the Sasaki geometry.

#### 2.3.4. Theorem: Value Continuity of Patched Standardization (Sasaki)

:::{prf:theorem} Value continuity of patched standardization (Sasaki)
:label: thm-sasaki-standardization-value-sq

Suppose $\mathcal S_1$ and $\mathcal S_2$ share the same alive set $\mathcal A$ of size $k\ge 1$ (so $n_c(\mathcal S_1,\mathcal S_2)=0$). Let $\mathbf r^{(r)}$ denote the raw reward vectors on $\mathcal A$. The N-dimensional standardization operator is Lipschitz continuous with respect to positional changes in the Sasaki metric. The squared L2-norm of the output error is bounded as follows:

$$
\big\|z(\mathcal S_1)-z(\mathcal S_2)\big\|_2^2 \le C_{V,\mathrm{total}}^{\mathrm{Sasaki}}(\mathcal S_1)\cdot\big\|\mathbf r^{(1)}-\mathbf r^{(2)}\big\|_2^2 \le C_{V,\mathrm{total}}^{\mathrm{Sasaki}}(\mathcal S_1)\cdot\left(L_R^{\mathrm{Sasaki}}\right)^2 \Delta_{\mathrm{pos,Sasaki}}^2(\mathcal S_1,\mathcal S_2).

$$

where $C_{V,\mathrm{total}}^{\mathrm{Sasaki}}$ is the **Total Value Error Coefficient**, a deterministic constant defined in {prf:ref}`def-sasaki-standardization-constants-sq`. The proof is provided in the subsequent sections by decomposing the total error into its constituent parts.
:::

#### 2.3.4.1. Sub-Lemma: Algebraic Decomposition of the Value Error

::::{prf:lemma} Decomposition of the Value Error
:label: lem-sasaki-value-error-decomposition

Let $\mathcal S$ be a fixed swarm state with alive set $\mathcal A$ of size $k$. Let $\mathbf r_1$ and $\mathbf r_2$ be two raw value vectors for the alive set. Let $(\mu_1, \sigma'_1)$ and $(\mu_2, \sigma'_2)$ be the corresponding statistical properties, and let $\mathbf z_1$ and $\mathbf z_2$ be the corresponding standardized vectors.


The total value error vector, $\Delta\mathbf{z} = \mathbf z_1 - \mathbf z_2$, can be expressed as the sum of three components:

$$
\Delta\mathbf{z} = \Delta_{\text{direct}} + \Delta_{\text{mean}} + \Delta_{\text{denom}}

$$

where:
1.  **The Direct Shift ($\Delta_{\text{direct}}$):** The error from the change in the raw value vector itself, scaled by the initial standard deviation.

    $$
    \Delta_{\text{direct}} := \frac{\mathbf r_1 - \mathbf r_2}{\sigma'_1}

    $$

2.  **The Mean Shift ($\Delta_{\text{mean}}$):** The error from the change in the aggregator's computed mean, applied uniformly to all walkers.

    $$
    \Delta_{\text{mean}} := \frac{\mu_2 - \mu_1}{\sigma'_1} \cdot \mathbf{1}

    $$
    where $\mathbf{1}$ is a k-dimensional vector of ones.

3.  **The Denominator Shift ($\Delta_{\text{denom}}$):** The error from the change in the regularized standard deviation, which rescales the second standardized vector.

    $$
    \Delta_{\text{denom}} := \mathbf z_2 \cdot \frac{\sigma'_2 - \sigma'_1}{\sigma'_1}

    $$

Furthermore, the total squared error is bounded by three times the sum of the squared norms of these components:

$$
\|\Delta\mathbf{z}\|_2^2 \le 3\left( \|\Delta_{\text{direct}}\|_2^2 + \|\Delta_{\text{mean}}\|_2^2 + \|\Delta_{\text{denom}}\|_2^2 \right)

$$

```{dropdown} Proof
:::{prf:proof}
**Step 1: Algebraic Decomposition.**
The proof of the decomposition is a direct algebraic manipulation. We start with the definition of the error and add and subtract the intermediate term $(\mathbf r_2 - \mu_2) / \sigma'_1$.

$$
\begin{aligned}
\Delta\mathbf{z} &= \frac{\mathbf r_1 - \mu_1}{\sigma'_1} - \frac{\mathbf r_2 - \mu_2}{\sigma'_2} \\
&= \left( \frac{\mathbf r_1 - \mu_1}{\sigma'_1} - \frac{\mathbf r_2 - \mu_2}{\sigma'_1} \right) + \left( \frac{\mathbf r_2 - \mu_2}{\sigma'_1} - \frac{\mathbf r_2 - \mu_2}{\sigma'_2} \right) \\
&= \frac{(\mathbf r_1 - \mathbf r_2) - (\mu_1 - \mu_2)}{\sigma'_1} + (\mathbf r_2 - \mu_2) \left( \frac{1}{\sigma'_1} - \frac{1}{\sigma'_2} \right) \\
&= \frac{\mathbf r_1 - \mathbf r_2}{\sigma'_1} + \frac{\mu_2 - \mu_1}{\sigma'_1}\mathbf{1} + \frac{\mathbf r_2 - \mu_2}{\sigma'_2} \frac{\sigma'_2 - \sigma'_1}{\sigma'_1} \\
&= \Delta_{\text{direct}} + \Delta_{\text{mean}} + \Delta_{\text{denom}}
\end{aligned}

$$
The final line follows by recognizing the definitions of the three components.

**Step 2: Bound on the Squared Norm.**
The bound on the total squared norm follows from the triangle inequality (`||A+B+C|| <= ||A|| + ||B|| + ||C||`) and the elementary inequality $(a+b+c)^2 \le 3(a^2+b^2+c^2)$ for non-negative reals. For vectors, this becomes:

$$
\|\Delta\mathbf{z}\|_2^2 = \|\Delta_{\text{direct}} + \Delta_{\text{mean}} + \Delta_{\text{denom}}\|_2^2 \le \left( \|\Delta_{\text{direct}}\|_2 + \|\Delta_{\text{mean}}\|_2 + \|\Delta_{\text{denom}}\|_2 \right)^2 \le 3\left( \|\Delta_{\text{direct}}\|_2^2 + \|\Delta_{\text{mean}}\|_2^2 + \|\Delta_{\text{denom}}\|_2^2 \right)

$$

This completes the proof.

**Q.E.D.**
:::
```
::::

#### 2.3.4.2. Sub-Lemma: Bounding the Squared Direct Shift Component

::::{prf:lemma} Bound on the Squared Direct Shift Component
:label: lem-sasaki-direct-shift-bound-sq

Let $\mathcal S$ be a fixed swarm state. Let $\mathbf r_1$ and $\mathbf r_2$ be two raw value vectors for the alive set. The squared Euclidean norm of the direct shift error component, $\Delta_{\text{direct}} = (\mathbf r_1 - \mathbf r_2) / \sigma'_1$, is bounded as follows:


$$
\|\Delta_{\text{direct}}\|_2^2 \le \frac{1}{\sigma_{\min,\mathrm{patch}}^2} \cdot \|\mathbf r_1 - \mathbf r_2\|_2^2

$$

where $\sigma_{\min,\mathrm{patch}} := \sqrt{\kappa_{\mathrm{var,min}}+\varepsilon_{\mathrm{std}}^2}$ is the uniform lower bound from the regularized standard deviation.

```{dropdown} Proof
:::{prf:proof}
The proof is a direct application of the definition of $\Delta_{\text{direct}}$ and the uniform lower bound on the regularized standard deviation.

1.  **Start with the Definition.**
    The squared L2-norm of the direct shift component is:

    $$
    \|\Delta_{\text{direct}}\|_2^2 = \left\| \frac{\mathbf r_1 - \mathbf r_2}{\sigma'_1} \right\|_2^2

    $$

2.  **Factor out the Scalar Term.**
    Since $\sigma'_1$ is a scalar value for the fixed swarm state and value vector $\mathbf r_1$, we can factor it out of the norm:

    $$
    \|\Delta_{\text{direct}}\|_2^2 = \frac{1}{(\sigma'_1)^2} \cdot \|\mathbf r_1 - \mathbf r_2\|_2^2

    $$

3.  **Apply the Uniform Lower Bound.**
    The regularized standard deviation function $\sigma'_{\mathrm{patch}}(V)$ is, by construction in the framework ({prf:ref}`def-statistical-properties-measurement`), strictly positive and uniformly bounded below by the constant $\sigma_{\min,\mathrm{patch}}$. Therefore, $\sigma'_1 \ge \sigma_{\min,\mathrm{patch}} > 0$. This implies:

    $$
    \frac{1}{(\sigma'_1)^2} \le \frac{1}{\sigma_{\min,\mathrm{patch}}^2}

    $$

4.  **Combine to Finalize the Bound.**
    Substituting the inequality from Step 3 into the expression from Step 2 yields the final bound as stated in the lemma.

    $$
    \|\Delta_{\text{direct}}\|_2^2 \le \frac{1}{\sigma_{\min,\mathrm{patch}}^2} \cdot \|\mathbf r_1 - \mathbf r_2\|_2^2

    $$

**Q.E.D.**
:::
```
::::

#### 2.3.4.3. Sub-Lemma: Bounding the Squared Mean Shift Component

::::{prf:lemma} Bound on the Squared Mean Shift Component
:label: lem-sasaki-mean-shift-bound-sq

Let $\mathcal S$ be a fixed swarm state with alive set $\mathcal A$ of size $k \ge 1$. Let $\mathbf r_1$ and $\mathbf r_2$ be two raw value vectors. The squared Euclidean norm of the mean shift error component, $\Delta_{\text{mean}} = ((\mu_2 - \mu_1) / \sigma'_1) \cdot \mathbf{1}$, is bounded as follows:


$$
\|\Delta_{\text{mean}}\|_2^2 \le \frac{k \cdot (L_{\mu,M}^{\mathrm{Sasaki}}(k))^2}{\sigma_{\min,\mathrm{patch}}^2} \cdot \|\mathbf r_1 - \mathbf r_2\|_2^2

$$

where $L_{\mu,M}^{\mathrm{Sasaki}}(k)$ is the axiomatic **Value Lipschitz Function** for the aggregator's mean from {prf:ref}`lem-sasaki-aggregator-lipschitz`.

```{dropdown} Proof
:::{prf:proof}
The proof combines the definition of the mean shift component with the axiomatic continuity of the mean aggregator.

1.  **Start with the Definition.**
    The squared L2-norm of the mean shift component is:

    $$
    \|\Delta_{\text{mean}}\|_2^2 = \left\| \frac{\mu_2 - \mu_1}{\sigma'_1} \cdot \mathbf{1} \right\|_2^2

    $$

2.  **Factor out the Scalar and Evaluate the Norm.**
    The term $(\mu_2 - \mu_1) / \sigma'_1$ is a scalar. The L2-norm of the k-dimensional vector of ones, $\mathbf{1}$, is $\|\mathbf{1}\|_2 = \sqrt{k}$. Therefore, the squared norm is:

    $$
    \|\Delta_{\text{mean}}\|_2^2 = \frac{(\mu_2 - \mu_1)^2}{(\sigma'_1)^2} \cdot \|\mathbf{1}\|_2^2 = \frac{k \cdot (\mu_2 - \mu_1)^2}{(\sigma'_1)^2}

    $$

3.  **Apply Axiomatic Continuity of the Mean.**
    The empirical aggregator is Lipschitz continuous with respect to the raw value vector, as established in {prf:ref}`lem-sasaki-aggregator-value`. This provides the bound:

    $$
    |\mu_2 - \mu_1|^2 \le \left(L_{\mu,M}^{\mathrm{Sasaki}}(k)\right)^2 \cdot \|\mathbf r_1 - \mathbf r_2\|_2^2

    $$

4.  **Apply the Uniform Lower Bound.**
    As in the previous lemma, we use the bound $1/(\sigma'_1)^2 \le 1/\sigma_{\min,\mathrm{patch}}^2$.

5.  **Combine to Finalize the Bound.**
    Substituting the bounds from Step 3 and Step 4 into the expression from Step 2 yields the final result as stated in the lemma.

    $$
    \|\Delta_{\text{mean}}\|_2^2 \le \frac{k \cdot (L_{\mu,M}^{\mathrm{Sasaki}}(k))^2}{\sigma_{\min,\mathrm{patch}}^2} \cdot \|\mathbf r_1 - \mathbf r_2\|_2^2

    $$

**Q.E.D.**
:::
```
::::

#### 2.3.4.4. Sub-Lemma: Bounding the Squared Denominator Shift Component

::::{prf:lemma} Bounding the Squared Denominator Shift Component
:label: lem-sasaki-denom-shift-bound-sq

Let $\mathcal S$ be a fixed swarm state with alive set $\mathcal A$ of size $k \ge 1$. Let $\mathbf r_1$ and $\mathbf r_2$ be two raw value vectors with components bounded by $V_{\max}^{(R)}$. The squared Euclidean norm of the denominator shift error component, $\Delta_{\text{denom}} = \mathbf z_2 \cdot ((\sigma'_2 - \sigma'_1) / \sigma'_1)$, is bounded as follows:


$$
\|\Delta_{\text{denom}}\|_2^2 \le k \left( \frac{2V_{\max}^{(R)}}{\sigma_{\min,\mathrm{patch}}} \right)^2 \left( \frac{L_{\sigma',M}^{\mathrm{Sasaki}}(k)}{\sigma_{\min,\mathrm{patch}}} \right)^2 \cdot \|\mathbf r_1 - \mathbf r_2\|_2^2

$$

where $L_{\sigma',M}^{\mathrm{Sasaki}}(k)$ is the derived Lipschitz constant for the regularized standard deviation.

```{dropdown} Proof
:::{prf:proof}
The proof bounds the squared norm by bounding its three constituent parts: the norm of the standardized vector, the change in the regularized standard deviation, and the inverse of the standard deviation.

1.  **Start with the Definition.**
    The squared L2-norm of the denominator shift component is:

    $$
    \|\Delta_{\text{denom}}\|_2^2 = \left\| \mathbf z_2 \cdot \frac{\sigma'_2 - \sigma'_1}{\sigma'_1} \right\|_2^2

    $$

2.  **Factor out the Scalar Term.**
    The fractional term involving the standard deviations is a scalar. We factor it out of the norm:

    $$
    \|\Delta_{\text{denom}}\|_2^2 = \|\mathbf z_2\|_2^2 \cdot \frac{(\sigma'_2 - \sigma'_1)^2}{(\sigma'_1)^2}

    $$

3.  **Bound Each Factor.**
    We now find a deterministic upper bound for each of the three factors in the expression.
    *   **Bound on `||z2||_2^2`**: The framework provides a universal bound on the squared norm of any standardized vector, proven in {prf:ref}`thm-z-score-norm-bound`. For the k-dimensional vector $\mathbf z_2$, this is:

        $$
        \|\mathbf z_2\|_2^2 \le k \left( \frac{2V_{\max}^{(R)}}{\sigma_{\min,\mathrm{patch}}} \right)^2

        $$

    *   **Bound on `(sigma'_2 - sigma'_1)^2`**: The regularized standard deviation function is Lipschitz continuous with respect to the raw value vector, as established by composing the Lipschitz properties of the aggregator moments and the patching function itself ({prf:ref}`lem-stats-value-continuity` in the framework). This gives:

        $$
        (\sigma'_2 - \sigma'_1)^2 \le \left(L_{\sigma',M}^{\mathrm{Sasaki}}(k)\right)^2 \cdot \|\mathbf r_1 - \mathbf r_2\|_2^2

        $$

    *   **Bound on `1/(sigma'_1)^2`**: As in the preceding lemmas, we use the uniform lower bound:

        $$
        \frac{1}{(\sigma'_1)^2} \le \frac{1}{\sigma_{\min,\mathrm{patch}}^2}

        $$

4.  **Combine to Finalize the Bound.**
    Substituting the bounds for all three factors from Step 3 into the expression from Step 2 yields the final bound as stated in the lemma.

    $$
    \|\Delta_{\text{denom}}\|_2^2 \le \left( k \left( \frac{2V_{\max}^{(R)}}{\sigma_{\min,\mathrm{patch}}} \right)^2 \right) \cdot \left( \left(L_{\sigma',M}^{\mathrm{Sasaki}}(k)\right)^2 \cdot \|\mathbf r_1 - \mathbf r_2\|_2^2 \right) \cdot \left( \frac{1}{\sigma_{\min,\mathrm{patch}}^2} \right)

    $$
    Rearranging the terms gives the stated result.

**Q.E.D.**
:::
```
::::

#### 2.3.4.5. Proof of Theorem 2.3.4

:::{prf:proof} of {prf:ref}`thm-sasaki-standardization-value-sq`

The proof establishes the final bound by assembling the deterministic bounds for each of the three error components derived in the preceding sub-lemmas.

**Step 1: Start with the Decomposed Error Bound.**
From the algebraic decomposition in {prf:ref}`lem-sasaki-value-error-decomposition`, the total squared value error is bounded by:

$$
\|z(\mathcal S_1)-z(\mathcal S_2)\|_2^2 \le 3\left( \|\Delta_{\text{direct}}\|_2^2 + \|\Delta_{\text{mean}}\|_2^2 + \|\Delta_{\text{denom}}\|_2^2 \right)

$$

**Step 2: Substitute the Bounds for Each Component.**
We substitute the deterministic bounds for the squared norm of each component, which all relate the component error to the squared norm of the raw value difference, $\|\mathbf r_1 - \mathbf r_2\|_2^2$.

*   From {prf:ref}`lem-sasaki-direct-shift-bound-sq`:

    $$
    \|\Delta_{\text{direct}}\|_2^2 \le C_{V,\mathrm{direct}}^{\mathrm{sq}}(\mathcal S_1) \cdot \|\mathbf r_1 - \mathbf r_2\|_2^2

    $$
*   From {prf:ref}`lem-sasaki-mean-shift-bound-sq`:

    $$
    \|\Delta_{\text{mean}}\|_2^2 \le C_{V,\mathrm{mean}}^{\mathrm{sq}}(\mathcal S_1) \cdot \|\mathbf r_1 - \mathbf r_2\|_2^2

    $$
*   From {prf:ref}`lem-sasaki-denom-shift-bound-sq`:

    $$
    \|\Delta_{\text{denom}}\|_2^2 \le C_{V,\mathrm{denom}}^{\mathrm{sq}}(\mathcal S_1) \cdot \|\mathbf r_1 - \mathbf r_2\|_2^2

    $$

**Step 3: Combine and Factor.**
Substituting these into the inequality from Step 1 and factoring out the common term $\|\mathbf r_1 - \mathbf r_2\|_2^2$ gives:

$$
\|z_1 - z_2\|_2^2 \le 3 \left( C_{V,\mathrm{direct}}^{\mathrm{sq}}(\mathcal S_1) + C_{V,\mathrm{mean}}^{\mathrm{sq}}(\mathcal S_1) + C_{V,\mathrm{denom}}^{\mathrm{sq}}(\mathcal S_1) \right) \cdot \|\mathbf r_1 - \mathbf r_2\|_2^2

$$

By definition ({prf:ref}`def-sasaki-standardization-constants-sq`), the term in parentheses is the **Total Value Error Coefficient**, $C_{V,\mathrm{total}}^{\mathrm{Sasaki}}(\mathcal S_1)$.

**Step 4: Relate Raw Value Error to Positional Displacement.**
The raw reward vector difference is bounded by the positional displacement via the Lipschitz continuity of the reward function ({prf:ref}`lem-euclidean-reward-regularity`):

$$
\|\mathbf r_1 - \mathbf r_2\|_2^2 = \sum_{i \in \mathcal A} |R(x_{1,i},v_{1,i}) - R(x_{2,i},v_{2,i})|^2 \le \sum_{i \in \mathcal A} \left(L_R^{\mathrm{Sasaki}}\right)^2 d_{\mathcal Y}^{\mathrm{Sasaki}}(\varphi(w_{1,i}), \varphi(w_{2,i}))^2 \le \left(L_R^{\mathrm{Sasaki}}\right)^2 \Delta_{\mathrm{pos,Sasaki}}^2(\mathcal S_1,\mathcal S_2)

$$

**Step 5: Final Assembly.**
Substituting the bound from Step 4 into the inequality from Step 3 gives

$$
\|z_1 - z_2\|_2^2 \le C_{V,\mathrm{total}}^{\mathrm{Sasaki}}(\mathcal S_1)\cdot\left(L_R^{\mathrm{Sasaki}}\right)^2 \Delta_{\mathrm{pos,Sasaki}}^2(\mathcal S_1,\mathcal S_2),
$$

where $C_{V,\mathrm{total}}^{\mathrm{Sasaki}}$ is defined in {prf:ref}`def-sasaki-standardization-constants-sq`.

This completes the proof.

**Q.E.D.**
:::

#### 2.3.5. Definition: Value Error Coefficients (Squared Form)

:::{prf:definition} Value Error Coefficients (Squared Form)
:label: def-sasaki-standardization-constants-sq

Let $\mathcal S$ be a fixed swarm state with alive set $\mathcal A$ of size $k \ge 1$, and let $M$ be the chosen **Swarm Aggregation Operator**. The coefficients for the bounds on the squared value error are defined as follows:

Referenced by {prf:ref}`thm-sasaki-standardization-value-sq`.

1.  **The Squared Direct Shift Coefficient ($C_{V,\mathrm{direct}}^{\mathrm{sq}}(\mathcal S)$):**

    $$
    C_{V,\mathrm{direct}}^{\mathrm{sq}}(\mathcal S) := \frac{1}{\sigma_{\min,\mathrm{patch}}^2}

    $$

2.  **The Squared Mean Shift Coefficient ($C_{V,\mathrm{mean}}^{\mathrm{sq}}(\mathcal S)$):**

    $$
    C_{V,\mathrm{mean}}^{\mathrm{sq}}(\mathcal S) := \frac{k \cdot (L_{\mu,M}^{\mathrm{Sasaki}}(k))^2}{\sigma_{\min,\mathrm{patch}}^2}

    $$

3.  **The Squared Denominator Shift Coefficient ($C_{V,\mathrm{denom}}^{\mathrm{sq}}(\mathcal S)$):**

    $$
    C_{V,\mathrm{denom}}^{\mathrm{sq}}(\mathcal S) := k \left( \frac{2V_{\max}^{(R)}}{\sigma_{\min,\mathrm{patch}}} \right)^2 \left( \frac{L_{\sigma',M}^{\mathrm{Sasaki}}(k)}{\sigma_{\min,\mathrm{patch}}} \right)^2

    $$

4.  **The Total Value Error Coefficient ($C_{V,\mathrm{total}}^{\mathrm{Sasaki}}(\mathcal S)$):** The composite coefficient that bounds the total squared value error, incorporating the factor of 3 from the error decomposition.

    $$
    C_{V,\mathrm{total}}^{\mathrm{Sasaki}}(\mathcal S) := 3 \cdot \left( C_{V,\mathrm{direct}}^{\mathrm{sq}}(\mathcal S) + C_{V,\mathrm{mean}}^{\mathrm{sq}}(\mathcal S) + C_{V,\mathrm{denom}}^{\mathrm{sq}}(\mathcal S) \right)

    $$

    When raw values are induced by positions, the positional coefficient in Theorem {prf:ref}`thm-sasaki-standardization-value-sq` is $\left(L_R^{\mathrm{Sasaki}}\right)^2 C_{V,\mathrm{total}}^{\mathrm{Sasaki}}$.

where $L_{\mu,M}^{\mathrm{Sasaki}}(k)$ and $L_{\sigma',M}^{\mathrm{Sasaki}}(k)$ are the value Lipschitz functions for the aggregator's mean and regularized standard deviation, respectively. For the canonical empirical aggregator, these coefficients simplify, notably making the mean shift coefficient independent of $k$: $C_{V,\mathrm{mean}}^{\mathrm{sq}}(\mathcal S) = 1/\sigma_{\min,\mathrm{patch}}^2$.
:::

#### 2.3.6. Theorem: Structural Continuity of Patched Standardization (Sasaki)

:::{prf:theorem} Structural Continuity of Patched Standardization (Sasaki)
:label: thm-sasaki-standardization-structural-sq

For general swarms $\mathcal S_1,\mathcal S_2$ with alive counts $k_r\ge 1$, the squared L2-norm of the output error of the standardization operator is bounded by a function of the number of status changes, $n_c(\mathcal S_1,\mathcal S_2)$.

Referenced by {prf:ref}`def-sasaki-structural-coeffs-sq` and {prf:ref}`lem-sasaki-standardization-lipschitz`.

$$
\|z(\mathcal S_1)-z(\mathcal S_2)\|_2^2 \le C_{S,\mathrm{direct}}^{\mathrm{sq}}(\mathcal S_1, \mathcal S_2) \cdot n_c(\mathcal S_1, \mathcal S_2) + C_{S,\mathrm{indirect}}^{\mathrm{sq}}(\mathcal S_1, \mathcal S_2) \cdot n_c(\mathcal S_1, \mathcal S_2)^2

$$

where $C_{S,\mathrm{direct}}^{\mathrm{sq}}$ and $C_{S,\mathrm{indirect}}^{\mathrm{sq}}$ are the **Squared Structural Error Coefficients** defined in {prf:ref}`def-sasaki-structural-coeffs-sq`. The proof is provided in the subsequent sections.
:::

#### 2.3.6.1. Sub-Lemma: Decomposition of the Structural Error

::::{prf:lemma} Decomposition of the Structural Error
:label: lem-sasaki-structural-error-decomposition

Let $\mathbf r_2$ be a fixed raw value vector. Let $\mathcal S_1$ and $\mathcal S_2$ be two swarm states. Let $\mathbf z_1 = z(\mathcal S_1, \mathbf r_2)$ and $\mathbf z_2 = z(\mathcal S_2, \mathbf r_2)$ be the corresponding N-dimensional standardized vectors, computed using the fixed raw values from the second swarm but the structure of each respective swarm.

Referenced by {prf:ref}`def-sasaki-structural-coeffs-sq`.

The total structural error vector, $\Delta\mathbf{z} = \mathbf z_1 - \mathbf z_2$, can be expressed as the sum of two **orthogonal** components:

$$
\Delta\mathbf{z} = \Delta_{\text{direct}} + \Delta_{\text{indirect}}

$$

where:
1.  **The Direct Error ($\Delta_{\text{direct}}$):** The error vector whose non-zero components correspond to walkers whose status changes between $\mathcal S_1$ and $\mathcal S_2$.
2.  **The Indirect Error ($\Delta_{\text{indirect}}$):** The error vector whose non-zero components correspond to walkers that are alive in both swarms.

Because these two vectors have disjoint support, the squared L2-norm of the total error is the sum of the squared L2-norms of the components:

$$
\|\Delta\mathbf{z}\|_2^2 = \|\Delta_{\text{direct}}\|_2^2 + \|\Delta_{\text{indirect}}\|_2^2

$$

```{dropdown} Proof
:::{prf:proof}
The proof follows from partitioning the sum of squared errors over the N walker indices. Let the full set of indices be $\{1, ..., N\}$.

1.  **Define Index Partitions.**
    Let $\mathcal{A}_{\text{unstable}} := \mathcal{A}(\mathcal S_1) \triangle \mathcal{A}(\mathcal S_2)$ be the set of indices of walkers whose survival status changes. Let $\mathcal{A}_{\text{stable}} := \mathcal{A}(\mathcal S_1) \cap \mathcal{A}(\mathcal S_2)$ be the set of indices for walkers that remain alive. Let $\mathcal{D}_{\text{stable}}$ be the indices of walkers dead in both swarms. These three sets form a partition of $\{1, ..., N\}$.

2.  **Analyze Error Components on Each Partition.**
    *   For $i \in \mathcal{A}_{\text{unstable}}$, the error component $(z_{1,i} - z_{2,i})$ is generally non-zero.
    *   For $i \in \mathcal{A}_{\text{stable}}$, the error component $(z_{1,i} - z_{2,i})$ is generally non-zero because the statistical moments $(\mu, \sigma')$ change with the swarm structure.
    *   For $i \in \mathcal{D}_{\text{stable}}$, both $z_{1,i}$ and $z_{2,i}$ are deterministically zero by the definition of the standardization operator. The error component is zero.

3.  **Define Orthogonal Error Vectors.**
    We define the vector $\Delta_{\text{direct}}$ such that its components are $(\Delta\mathbf{z})_i$ for $i \in \mathcal{A}_{\text{unstable}}$ and zero otherwise. We define $\Delta_{\text{indirect}}$ such that its components are $(\Delta\mathbf{z})_i$ for $i \in \mathcal{A}_{\text{stable}}$ and zero otherwise. By construction, $\Delta\mathbf{z} = \Delta_{\text{direct}} + \Delta_{\text{indirect}}$.

4.  **Show Orthogonality.**
    The two vectors have disjoint support, meaning for any index $i$, at most one of the vectors can have a non-zero component. Therefore, their dot product is zero: $\Delta_{\text{direct}} \cdot \Delta_{\text{indirect}} = 0$.

5.  **Finalize the Squared Norm Identity.**
    For orthogonal vectors, the squared norm of the sum is the sum of the squared norms:

    $$
    \|\Delta\mathbf{z}\|_2^2 = \|\Delta_{\text{direct}} + \Delta_{\text{indirect}}\|_2^2 = \|\Delta_{\text{direct}}\|_2^2 + \|\Delta_{\text{indirect}}\|_2^2 + 2(\Delta_{\text{direct}} \cdot \Delta_{\text{indirect}}) = \|\Delta_{\text{direct}}\|_2^2 + \|\Delta_{\text{indirect}}\|_2^2

    $$

This completes the proof.

**Q.E.D.**
:::
```
::::

#### 2.3.6.2. Sub-Lemma: Bounding the Squared Direct Structural Error

::::{prf:lemma} Bound on the Squared Direct Structural Error
:label: lem-sasaki-direct-structural-error-sq

Let $\mathbf r_2$ be a fixed raw value vector with components bounded by $V_{\max}^{(R)}$. The squared Euclidean norm of the direct structural error component, $\|\Delta_{\text{direct}}\|_2^2$, is bounded by a term linear in the number of status changes, $n_c(\mathcal S_1, \mathcal S_2)$.

Referenced by {prf:ref}`def-sasaki-structural-coeffs-sq`.

$$
\|\Delta_{\text{direct}}\|_2^2 \le \left( \frac{2V_{\max}^{(R)}}{\sigma_{\min,\mathrm{patch}}} \right)^2 \cdot n_c(\mathcal S_1, \mathcal S_2)

$$

```{dropdown} Proof
:::{prf:proof}
The proof bounds the squared error for each unstable walker and sums the results.

1.  **Isolate the Sum.**
    By definition, the vector $\Delta_{\text{direct}}$ has non-zero components only for walkers in the unstable set $\mathcal{A}_{\text{unstable}} = \mathcal{A}(\mathcal S_1) \triangle \mathcal{A}(\mathcal S_2)$. The number of walkers in this set is exactly $n_c = n_c(\mathcal S_1, \mathcal S_2)$. The squared norm is the sum of the squared errors over this set:

    $$
    \|\Delta_{\text{direct}}\|_2^2 = \sum_{i \in \mathcal{A}_{\text{unstable}}} (z_{1,i} - z_{2,i})^2

    $$

2.  **Bound the Error for a Single Unstable Walker.**
    Consider a single walker $i \in \mathcal{A}_{\text{unstable}}$. Its status changes between $\mathcal S_1$ and $\mathcal S_2$. This means one of two cases:
    *   Case A: Walker $i$ is alive in $\mathcal S_1$ and dead in $\mathcal S_2$. Then $z_{2,i} = 0$. The error is $(z_{1,i})^2$.
    *   Case B: Walker $i$ is dead in $\mathcal S_1$ and alive in $\mathcal S_2$. Then $z_{1,i} = 0$. The error is $(-z_{2,i})^2 = (z_{2,i})^2$.

    In both cases, the squared error for walker $i$ is the square of a single, valid standardized score.

3.  **Apply the Universal Z-Score Bound.**
    The framework provides a universal bound for the magnitude of any single standardized score in {prf:ref}`thm-z-score-norm-bound`, which is $|z_j| \le 2V_{\max}^{(R)} / \sigma_{\min,\mathrm{patch}}$. Squaring this gives a uniform bound for the squared error of any unstable walker:

    $$
    (z_{1,i} - z_{2,i})^2 \le \left( \frac{2V_{\max}^{(R)}}{\sigma_{\min,\mathrm{patch}}} \right)^2

    $$

4.  **Sum Over All Unstable Walkers.**
    We sum this uniform bound over all $n_c$ walkers in the unstable set:

    $$
    \|\Delta_{\text{direct}}\|_2^2 = \sum_{i \in \mathcal{A}_{\text{unstable}}} (z_{1,i} - z_{2,i})^2 \le \sum_{i=1}^{n_c} \left( \frac{2V_{\max}^{(R)}}{\sigma_{\min,\mathrm{patch}}} \right)^2

    $$

    This yields the final result as stated in the lemma.

    $$
    \|\Delta_{\text{direct}}\|_2^2 \le n_c \cdot \left( \frac{2V_{\max}^{(R)}}{\sigma_{\min,\mathrm{patch}}} \right)^2

    $$

**Q.E.D.**
:::
```
::::

#### 2.3.6.3. Sub-Lemma: Bounding the Squared Indirect Structural Error

::::{prf:lemma} Bound on the Squared Indirect Structural Error
:label: lem-sasaki-indirect-structural-error-sq

Let $\mathbf r_2$ be a fixed raw value vector. Let $\mathcal S_1$ and $\mathcal S_2$ be two swarm states. The squared Euclidean norm of the indirect structural error component, $\|\Delta_{\text{indirect}}\|_2^2$, is bounded by a term quadratic in the number of status changes, $n_c(\mathcal S_1, \mathcal S_2)$.

Referenced by {prf:ref}`def-sasaki-structural-coeffs-sq`.

$$
\|\Delta_{\text{indirect}}\|_2^2 \le C_{S,\mathrm{indirect}}^{\mathrm{sq}}(\mathcal S_1, \mathcal S_2) \cdot n_c(\mathcal S_1, \mathcal S_2)^2

$$

where $C_{S,\mathrm{indirect}}^{\mathrm{sq}}$ is the **Squared Indirect Structural Error Coefficient** defined in {prf:ref}`def-sasaki-structural-coeffs-sq`.

```{dropdown} Proof
:::{prf:proof}
The proof decomposes the error for each stable walker into a mean-shift and a denominator-shift component and then bounds the sum of their squares.

**Step 1: Decompose the Error for a Single Stable Walker.**
For any walker $i$ in the stable set $\mathcal A_{\mathrm{stable}} = \mathcal A(\mathcal S_1) \cap \mathcal A(\mathcal S_2)$, the error is:

$$
\begin{aligned}
z_{1,i} - z_{2,i} &= \frac{r_{2,i} - \mu_1}{\sigma'_1} - \frac{r_{2,i} - \mu_2}{\sigma'_2} \\
&= \left(\frac{r_{2,i} - \mu_1}{\sigma'_1} - \frac{r_{2,i} - \mu_2}{\sigma'_1}\right) + \left(\frac{r_{2,i} - \mu_2}{\sigma'_1} - \frac{r_{2,i} - \mu_2}{\sigma'_2}\right) \\
&= \underbrace{\frac{\mu_2 - \mu_1}{\sigma'_1}}_{\text{Mean Shift}} + \underbrace{z_{2,i} \frac{\sigma'_2 - \sigma'_1}{\sigma'_1}}_{\text{Denominator Shift}}
\end{aligned}

$$
The squared error for this single walker is bounded using $(a+b)^2 \le 2(a^2+b^2)$:

$$
(z_{1,i} - z_{2,i})^2 \le 2\left(\frac{\mu_2 - \mu_1}{\sigma'_1}\right)^2 + 2\left(z_{2,i} \frac{\sigma'_2 - \sigma'_1}{\sigma'_1}\right)^2

$$

**Step 2: Sum the Errors Over All Stable Walkers.**
The total squared indirect error is the sum over all $i \in \mathcal A_{\mathrm{stable}}$. Let $k_{\mathrm{stable}} := |\mathcal A_{\mathrm{stable}}|$.

$$
\|\Delta_{\text{indirect}}\|_2^2 = \sum_{i \in \mathcal A_{\mathrm{stable}}} (z_{1,i} - z_{2,i})^2 \le \sum_{i \in \mathcal A_{\mathrm{stable}}} 2\left(\frac{\mu_2 - \mu_1}{\sigma'_1}\right)^2 + \sum_{i \in \mathcal A_{\mathrm{stable}}} 2\left(z_{2,i} \frac{\sigma'_2 - \sigma'_1}{\sigma'_1}\right)^2

$$

This can be simplified:

$$
\|\Delta_{\text{indirect}}\|_2^2 \le 2 k_{\mathrm{stable}} \frac{(\mu_2 - \mu_1)^2}{(\sigma'_1)^2} + 2 \frac{(\sigma'_2 - \sigma'_1)^2}{(\sigma'_1)^2} \sum_{i \in \mathcal A_{\mathrm{stable}}} (z_{2,i})^2

$$

**Step 3: Bound the Components.**
We now bound each term using the axiomatic properties of the aggregator and the uniform bounds from the framework.
*   **Bound on `(mu_2 - mu_1)^2`**: The structural continuity of the empirical mean ({prf:ref}`lem-sasaki-aggregator-structural`) gives:

    $$
    (\mu_2 - \mu_1)^2 \le \left(L_{\mu,S}^{\mathrm{Sasaki}}(k_{\min}) \cdot n_c\right)^2

    $$
*   **Bound on `(sigma'_2 - sigma'_1)^2`**: By composing the structural continuity of the variance with the Lipschitz property of the patching function ({prf:ref}`lem-stats-structural-continuity` in the framework), we get:

    $$
    (\sigma'_2 - \sigma'_1)^2 \le \left(L_{\sigma',S}^{\mathrm{Sasaki}}(\mathcal S_1, \mathcal S_2) \cdot n_c\right)^2

    $$
*   **Bound on `sum(z_2,i^2)`**: The sum is over the stable set, which is a subset of the alive walkers in $\mathcal S_2$. Thus, the sum is bounded by the total squared norm of the z-score vector for $\mathcal S_2$:

    $$
    \sum_{i \in \mathcal A_{\mathrm{stable}}} (z_{2,i})^2 \le \|\mathbf z_2\|_2^2 \le k_2 \left(\frac{2V_{\max}^{(R)}}{\sigma_{\min,\mathrm{patch}}}\right)^2

    $$
*   **Bound on `1/(sigma'_1)^2`**: This is bounded by $1/\sigma_{\min,\mathrm{patch}}^2$.

**Step 4: Assemble the Final Bound.**
Substituting these bounds back into the inequality from Step 2 gives a bound that is quadratic in $n_c$.

$$
\|\Delta_{\text{indirect}}\|_2^2 \le 2 k_{\mathrm{stable}} \frac{(L_{\mu,S})^2 n_c^2}{\sigma_{\min,\mathrm{patch}}^2} + 2 \frac{(L_{\sigma',S})^2 n_c^2}{\sigma_{\min,\mathrm{patch}}^2} k_2 \left(\frac{2V_{\max}^{(R)}}{\sigma_{\min,\mathrm{patch}}}\right)^2

$$

Factoring out $n_c^2$ and combining the coefficients gives:

$$
\|\Delta_{\text{indirect}}\|_2^2 \le \left[ 2 k_{\mathrm{stable}} \frac{(L_{\mu,S})^2}{\sigma_{\min,\mathrm{patch}}^2} + 2 k_2 \left(\frac{2V_{\max}^{(R)}}{\sigma_{\min,\mathrm{patch}}}\right)^2 \frac{(L_{\sigma',S})^2}{\sigma_{\min,\mathrm{patch}}^2} \right] \cdot n_c^2

$$
The term in the brackets is precisely the definition of the **Squared Indirect Structural Error Coefficient**, $C_{S,\mathrm{indirect}}^{\mathrm{sq}}(\mathcal S_1, \mathcal S_2)$. This completes the proof.

**Q.E.D.**
:::
```
::::

#### 2.3.6.4. Proof of Theorem 2.3.6

:::{prf:proof} of {prf:ref}`thm-sasaki-standardization-structural-sq`
The proof establishes the final bound by assembling the deterministic bounds for the two orthogonal error components derived in the preceding sub-lemmas.

**Step 1: Start with the Orthogonal Decomposition.**
From {prf:ref}`lem-sasaki-structural-error-decomposition`, the total squared structural error is the sum of the squared norms of the direct and indirect error components:

$$
\|z(\mathcal S_1)-z(\mathcal S_2)\|_2^2 = \|\Delta_{\text{direct}}\|_2^2 + \|\Delta_{\text{indirect}}\|_2^2

$$

**Step 2: Substitute the Bounds for Each Component.**
We substitute the deterministic bounds for the squared norm of each component.

*   From {prf:ref}`lem-sasaki-direct-structural-error-sq`, the direct error is bounded by a term linear in $n_c$:

    $$
    \|\Delta_{\text{direct}}\|_2^2 \le C_{S,\mathrm{direct}}^{\mathrm{sq}} \cdot n_c(\mathcal S_1, \mathcal S_2)

    $$

*   From {prf:ref}`lem-sasaki-indirect-structural-error-sq`, the indirect error is bounded by a term quadratic in $n_c$:

    $$
    \|\Delta_{\text{indirect}}\|_2^2 \le C_{S,\mathrm{indirect}}^{\mathrm{sq}}(\mathcal S_1, \mathcal S_2) \cdot n_c(\mathcal S_1, \mathcal S_2)^2

    $$

**Step 3: Combine the Bounds.**
Summing the two bounds from Step 2 directly gives the final inequality as stated in Theorem {prf:ref}`thm-sasaki-standardization-structural-sq`.

$$
\|z(\mathcal S_1)-z(\mathcal S_2)\|_2^2 \le C_{S,\mathrm{direct}}^{\mathrm{sq}} \cdot n_c(\mathcal S_1, \mathcal S_2) + C_{S,\mathrm{indirect}}^{\mathrm{sq}}(\mathcal S_1, \mathcal S_2) \cdot n_c(\mathcal S_1, \mathcal S_2)^2

$$

This completes the proof, establishing a deterministic, worst-case bound on the operator's output error due to structural changes.

**Q.E.D.**

:::

#### 2.3.7. Structural Error Coefficients (Squared Form)

:::{prf:definition} Structural Error Coefficients (Squared Form)
:label: def-sasaki-structural-coeffs-sq

Let $\mathcal S_1$ and $\mathcal S_2$ be two swarm states with alive sets $\mathcal A_1$ and $\mathcal A_2$, of sizes $k_1:=|\mathcal A_1|$ and $k_2:=|\mathcal A_2|$. Let $k_{\mathrm{stable}}:=|\mathcal A_1\cap\mathcal A_2|$. The coefficients for the bounds on the squared structural error are defined as follows:

Referenced by {prf:ref}`lem-sasaki-indirect-structural-error-sq` and {prf:ref}`thm-sasaki-standardization-structural-sq`.

1.  **The Squared Direct Structural Error Coefficient ($C_{S,\mathrm{direct}}^{\mathrm{sq}}$):** The coefficient of the term linear in $n_c$.

    $$
    C_{S,\mathrm{direct}}^{\mathrm{sq}} := \left( \frac{2V_{\max}^{(R)}}{\sigma_{\min,\mathrm{patch}}} \right)^2

    $$

2.  **The Squared Indirect Structural Error Coefficient ($C_{S,\mathrm{indirect}}^{\mathrm{sq}}(\mathcal S_1, \mathcal S_2)$):** The coefficient of the term quadratic in $n_c$, which bounds the error for the stable walkers.

    $$
    C_{S,\mathrm{indirect}}^{\mathrm{sq}}(\mathcal S_1, \mathcal S_2) := 2 k_{\mathrm{stable}} \frac{(L_{\mu,S}^{\mathrm{Sasaki}})^2}{\sigma_{\min,\mathrm{patch}}^{2}} + 2 k_2 \left(\frac{2V_{\max}^{(R)}}{\sigma_{\min,\mathrm{patch}}}\right)^2 \frac{(L_{\sigma',S}^{\mathrm{Sasaki}})^2}{\sigma_{\min,\mathrm{patch}}^{2}}

    $$
:::

where $L_{\mu,S}^{\mathrm{Sasaki}}$ and $L_{\sigma',S}^{\mathrm{Sasaki}}$ are the structural continuity functions for the aggregator's mean and regularized standard deviation, respectively, which depend on the swarm states.

#### 2.3.8. Theorem: Composite Continuity of the Patched Standardization Operator

::::{prf:theorem} Composite Continuity of the Patched Standardization Operator (Sasaki)
:label: thm-sasaki-standardization-composite-sq

The N-dimensional standardization operator $z(\mathcal S)$, when applied to the reward vector, is continuous with respect to the dispersion metric. For any two swarms $\mathcal S_1, \mathcal S_2$ with $k_1=|\mathcal A(\mathcal S_1)|\ge 1$, the squared L2-norm of the output error is bounded by a composite function of the squared dispersion distance:

$$
\|z(\mathcal S_1)-z(\mathcal S_2)\|_2^2 \le L_{z,L}^2(\mathcal S_1,\mathcal S_2) \cdot d_{\mathrm{Disp},\mathcal Y}^{\mathrm{Sasaki}}(\mathcal S_1, \mathcal S_2)^2 + L_{z,H}^2(\mathcal S_1,\mathcal S_2) \cdot d_{\mathrm{Disp},\mathcal Y}^{\mathrm{Sasaki}}(\mathcal S_1, \mathcal S_2)^4

$$

where $L_{z,L}^2$ and $L_{z,H}^2$ are state-dependent coefficients representing the Lipschitz and higher-order parts of the bound, respectively. Consequently, the logistic rescale operator $u(\mathcal S) = g_A(z(\mathcal S))$ is also continuous.

```{dropdown} Proof
:::{prf:proof}
The proof establishes a deterministic bound on the total error $\|z(\mathcal S_1, \mathbf r_1) - z(\mathcal S_2, \mathbf r_2)\|_2^2$ by combining the bounds for value-induced error and structure-induced error.

**Step 1: Decomposing the Total Error.**
Let $\mathbf r_1$ and $\mathbf r_2$ be the raw reward vectors for swarms $\mathcal S_1$ and $\mathcal S_2$. We introduce an intermediate vector $z(\mathcal S_1, \mathbf r_2)$ and use the inequality $\|A-C\|_2^2 \le 2(\|A-B\|_2^2 + \|B-C\|_2^2)$. The total squared error is bounded by the sum of a pure value error component and a pure structural error component:

$$
\|z(\mathcal S_1, \mathbf r_1) - z(\mathcal S_2, \mathbf r_2)\|_2^2 \le 2\,\|\underbrace{z(\mathcal S_1, \mathbf r_1) - z(\mathcal S_1, \mathbf r_2)}_{E_V}\|_2^2 + 2\,\|\underbrace{z(\mathcal S_1, \mathbf r_2) - z(\mathcal S_2, \mathbf r_2)}_{E_S}\|_2^2

$$

**Step 2: Bounding the Squared Value Error Term (`||E_V||_2^2`).**
The first term is a pure value error for the fixed swarm structure $\mathcal S_1$. We apply Theorem {prf:ref}`thm-sasaki-standardization-value-sq`:

$$
\|E_V\|_2^2 \le C_{V,\mathrm{total}}^{\mathrm{Sasaki}}(\mathcal S_1) \cdot \|\mathbf r_1 - \mathbf r_2\|_2^2

$$
The squared difference of the raw reward vectors is bounded by the sum of contributions from walkers with stable status and unstable status:

$$
\|\mathbf r_1 - \mathbf r_2\|_2^2 = \sum_{i \in \mathcal A_1 \cap \mathcal A_2} |r_{1,i}-r_{2,i}|^2 + \sum_{i \in \mathcal A_1 \triangle \mathcal A_2} |r_{1,i}-r_{2,i}|^2 \le (L_R^{\mathrm{Sasaki}})^2 \Delta_{\mathrm{pos,Sasaki}}^2 + n_c \big(V_{\max}^{(R)}\big)^2

$$
where we used that for unstable walkers, one reward is zero and the other is bounded by $V_{\max}^{(R)}$.

**Step 3: Bounding the Squared Structural Error Term (`||E_S||_2^2`).**
The second term is a pure structural error for the fixed raw value vector $\mathbf r_2$. We apply Theorem {prf:ref}`thm-sasaki-standardization-structural-sq`:

$$
\|E_S\|_2^2 \le C_{S,\mathrm{direct}}^{\mathrm{sq}} \cdot n_c + C_{S,\mathrm{indirect}}^{\mathrm{sq}}(\mathcal S_1, \mathcal S_2) \cdot n_c^2

$$

**Step 4: Assembling the Composite Bound in Terms of Displacement Components.**
Combining the bounds from Steps 2 and 3 into the decomposition from Step 1 gives a complete bound in terms of $\Delta_{\mathrm{pos,Sasaki}}^2$ and $n_c$:

$$
\|z_1 - z_2\|_2^2 \le 2\,C_{V,\mathrm{total}}^{\mathrm{Sasaki}} \left[ (L_R^{\mathrm{Sasaki}})^2 \Delta_{\mathrm{pos,Sasaki}}^2 + \big(V_{\max}^{(R)}\big)^2 n_c \right] + 2 \left[ C_{S,\mathrm{direct}}^{\mathrm{sq}} n_c + C_{S,\mathrm{indirect}}^{\mathrm{sq}} n_c^2 \right]

$$

**Step 5: Expressing the Bound in Terms of the Dispersion Metric.**
Let $d^2:=d_{\mathrm{Disp},\mathcal Y}^{\mathrm{Sasaki}}(\mathcal S_1,\mathcal S_2)^2$. From the definition of the dispersion metric we obtain

$$
\Delta_{\mathrm{pos,Sasaki}}^2\le N d^2,\qquad n_c\le\frac{N}{\lambda_{\mathrm{status}}}d^2,\qquad n_c^2\le\left(\frac{N}{\lambda_{\mathrm{status}}}\right)^2 d^4.

$$

Substituting these bounds into the expression from Step 4 yields

$$
\|z(\mathcal S_1)-z(\mathcal S_2)\|_2^2\le L_{z,L}^2(\mathcal S_1,\mathcal S_2)d^2+L_{z,H}^2(\mathcal S_1,\mathcal S_2)d^4.

$$

The coefficients are explicit:

$$
\begin{aligned}
L_{z,L}^2(\mathcal S_1,\mathcal S_2)&:=2C_{V,\mathrm{total}}^{\mathrm{Sasaki}}(\mathcal S_1)(L_R^{\mathrm{Sasaki}})^2N\\&\quad{}+\frac{N}{\lambda_{\mathrm{status}}}\Big(2C_{V,\mathrm{total}}^{\mathrm{Sasaki}}(\mathcal S_1)\big(V_{\max}^{(R)}\big)^2+2C_{S,\mathrm{direct}}^{\mathrm{sq}}(\mathcal S_1,\mathcal S_2)\Big),\\[4pt]
L_{z,H}^2(\mathcal S_1,\mathcal S_2)&:=2C_{S,\mathrm{indirect}}^{\mathrm{sq}}(\mathcal S_1,\mathcal S_2)\left(\frac{N}{\lambda_{\mathrm{status}}}\right)^2.
\end{aligned}

$$

All quantities on the right-hand side depend only on the swarm parameters and the bounds established earlier, so the coefficients are finite. Since the rescale function $g_A$ is globally Lipschitz ({prf:ref}`thm-rescale-function-lipschitz`), the continuity of $z$ implies the continuity of the composite operator $u(\mathcal S)=g_A(z(\mathcal S))$.

**Q.E.D.**
```
:::
::::
::::{prf:lemma} Lipschitz continuity of patched standardization (Sasaki)
:label: lem-sasaki-standardization-lipschitz

The bounds in Theorem {prf:ref}`thm-sasaki-standardization-composite-sq` show that the patched standardization operator $z$ is continuous with respect to the dispersion metric. In particular, $z$ admits the composite Lipschitz–Hölder control

$$
\|z(\mathcal S_1)-z(\mathcal S_2)\|_2^2\le L_{z,L}^2(\mathcal S_1,\mathcal S_2)\,d_{\mathrm{Disp},\mathcal Y}^{\mathrm{Sasaki}}(\mathcal S_1,\mathcal S_2)^2+L_{z,H}^2(\mathcal S_1,\mathcal S_2)\,d_{\mathrm{Disp},\mathcal Y}^{\mathrm{Sasaki}}(\mathcal S_1,\mathcal S_2)^4.

$$

```{dropdown} Proof
:::{prf:proof}
The inequality is precisely the statement of Theorem {prf:ref}`thm-sasaki-standardization-composite-sq`; no additional work is required.
:::
```

::::

### 4.4 Swarm-level continuity and population evolution

:::{prf:remark} Constants and their scope
:label: remark-eg-continuity-scope

The kinetic constants $L_{\mathrm{flow}},s_h,C_x,C_v,C_0$ are given explicitly in {prf:ref}`lem-sasaki-kinetic-lipschitz` and {prf:ref}`lem-euclidean-perturb-moment`. The terminal death-probability constant is $L_{\mathrm{flow}}/(\sqrt{2\pi}s_h)$ in physical phase-space distance. On a compact physical set these can be converted to squashed-distance constants using the Lipschitz constant of the inverse projection.

For each fixed $N$, the complete marked kernel is Feller by {prf:ref}`thm-euclidean-feller`. Uniform-in-$N$ continuity and concentration require control of the accepted components and sampled global statistics. Their proofs are given for the same canonical transition in {doc}`08_mean_field` and {doc}`09_propagation_chaos`; they do not follow from independent per-walker collision outputs. The shared rotation is part of the population map.
:::



### 2.6 Axiom for Convergence: Non-Deceptive Landscape

For the geometric ergodicity proof in {doc}`06_convergence` to hold, the Euclidean Gas must satisfy an additional environmental axiom that strengthens the Axiom of Environmental Richness. This axiom prevents pathological scenarios where positional diversity and reward signals become decoupled.

:::{prf:axiom} Axiom of Non-Deceptive Landscapes
:label: axiom-non-deceptive

The environment $(X_{\mathrm{valid}}, R_{\mathrm{pos}})$ is **non-deceptive** if there exist constants $\kappa_{\mathrm{grad}} > 0$ and $L_{\mathrm{grad}} > 0$ such that for any two points $x, y \in X_{\mathrm{valid}}$ with $\|x - y\| \ge L_{\mathrm{grad}}$, the average squared norm of the reward gradient along the line segment connecting them is bounded below:

$$
\frac{1}{\|x-y\|} \int_{0}^{\|x-y\|} \big\|\nabla R_{\mathrm{pos}}\big(x + t\tfrac{y-x}{\|y-x\|}\big)\big\|^2 dt \ge \kappa_{\mathrm{grad}}.

$$

**Applicability:** This inequality is an additional quantitative landscape hypothesis wherever a theorem uses it. Continuity alone does not prove a positive lower bound. It is not required to define the canonical transition or its finite-horizon population limit; any geometric-ergodicity application must verify it for its specified potential and scales.
:::



(sec-eg-verified-kernel)=
## 5. The canonical marked transition

:::{prf:theorem} Well-defined canonical Euclidean Gas
:label: thm-eg-canonical-kernel

For finite $N\ge1$, positive Gaussian donor widths, positive regularization and fitness floors, finite $h>0$, globally Lipschitz force, and the canonical schedule, {prf:ref}`alg-euclidean-gas` defines a time-homogeneous Markov kernel on the full marked state. It preserves all-slot momentum during the collision stage and is permutation equivariant. With $\sigma_x>0$ and Lebesgue-null boundary it is Feller.

*Proof.* With at least one alive row the donor normalizers are positive; self-exclusion and singleton rules specify every draw. Positive regularization denominators make the sampled fitness finite, and all acceptance probabilities lie in $[0,1]$. A finite undirected graph has a unique partition into connected components. The common-rotation formula assigns each row exactly one output. BAOAB, final diffusion, cap, and status classification are measurable, so their composition with the finite draws is a probability kernel. The all-dead branch is absorbing.

For each component, $\sum_{i\in C}(v_i-\bar v_C)=0$, so summing its velocity formula gives $\sum_{i\in C}\widetilde v_i=\sum_{i\in C}v_i$. Relabeling the input, donor indices, innovations, and components relabels the output: weights, empirical statistics, connectedness, and component means are unchanged as unlabeled objects. Haar matrices have the same independent law after this transport. This proves kernel equivariance, without requiring equality of arbitrary fixed-seed trajectories under relabeling. The Feller assertion is proved in {prf:ref}`thm-euclidean-feller`. $\square$
:::

:::{prf:remark} What requires a separate convergence argument
:label: remark-eg-convergence-hypotheses

Kernel existence and symmetry do not imply every axiom of an abstract convergence theorem. In particular, a richness lower bound depends on an explicitly specified reference measure, correlated component outputs cannot satisfy an independent-output assumption, and stationary uniqueness requires an attraction or contraction argument for the actual nonlinear map. The canonical population and finite-horizon chaos statements are established in {doc}`08_mean_field` and {doc}`09_propagation_chaos`; the latter gives the exact stationary identification obligations.
:::


(sec-eg-kernel)=
## 6. Swarm Update Operator Kernel

We define the one-step kernel $\Psi_{\mathcal F_{\mathrm{EG}}}$ on the ordered swarm space $\Sigma_N=(\mathcal X\times\mathbb R^d\times\{0,1\})^N$. Measurement and donor innovations are independent across recipient rows conditional on the input, while rotations are independent across accepted components. Output walkers within a component share one rotation. When no walkers are alive the process becomes absorbing.

(sec-eg-stage1)=
### 6.1 Stage 1 — Cemetery absorption

If the alive index set $\mathcal A(\mathcal S_t)$ is empty, the operator returns $\delta_{\mathcal S_t}$ and the run stops. All subsequent stages are skipped.

(sec-eg-stage2)=
### 6.2 Stage 2 — Single-shot measurement and frozen potentials

:::{prf:definition} Finite sampled measurement and fitness law
:label: def-eg-frozen-measurements

Let $\mathcal A=\{i:a_i=1\}$. For $M\ge2$, the measurement and cloning kernels for an alive recipient exclude its own label. A dead recipient's clone kernel uses all of $\mathcal A$. For either role $b\in\{D,C\}$,
$$
P_b^N(i,j)=\frac{\mathbf1_{j\in\mathcal A\setminus\{i\}}\exp[-d_{\mathrm{alg}}(i,j)^2/(2\epsilon_b^2)]}
{\sum_{k\in\mathcal A\setminus\{i\}}\exp[-d_{\mathrm{alg}}(i,k)^2/(2\epsilon_b^2)]}.
$$
An alive singleton uses the zero-distance exception and cannot clone from itself. Dead rows still select the sole alive donor.

Fix the separation floor $\delta_D>0$. Independently for each alive $i$, sample $J_i^D$ and retain
$$
r_i=R(x_i,v_i),\qquad d_i=\sqrt{d_{\mathrm{alg}}(i,J_i^D)^2+\delta_D^2}.
$$
In the singleton exception the measured raw distance is zero and the separation is $\delta_D$. Set unused dead measurement entries to zero. For $y=r,d$, compute
$$
\bar y=\frac1M\sum_{i\in\mathcal A}y_i,\qquad
\widehat\sigma_y=\sqrt{\frac1M\sum_{i\in\mathcal A}(y_i-\bar y)^2+\sigma_{\min,y}^2},\qquad
z_i^y=(y_i-\bar y)/\widehat\sigma_y.
$$
For an objective minimized by the engine, $R$ denotes its negative, so larger $z_i^r$ is better. Define
$$
V_{\mathrm{fit},i}=
\left(\frac{A_r}{1+e^{-z_i^r}}+\eta_r\right)^\alpha
\left(\frac{A_d}{1+e^{-z_i^d}}+\eta_d\right)^\beta
\quad(i\in\mathcal A),
$$
and set unused dead fitness to zero. Freeze the entire realized fitness vector for acceptance. In particular, its donor entry contains the donor's own sampled measurement, and common empirical means and variances are retained.
:::


(sec-eg-stage3)=
### 6.3 Stage 3 — Accepted edges and component collisions

:::{prf:definition} Frozen component cloning transformation
:label: def-eg-component-collision

Draw cloning donors $J_i^C$ from the kernels in {prf:ref}`def-eg-frozen-measurements` and independent $U_i\sim\operatorname{Unif}[0,1]$. For alive rows with a distinct donor, set
$$
p_i=\min\left\{1,\frac{[V_{\mathrm{fit},J_i^C}-V_{\mathrm{fit},i}]_+}{p_{\max}(V_{\mathrm{fit},i}+\varepsilon_{\mathrm{clone}})}\right\},
\qquad A_i=\mathbf1_{\{U_i<p_i\}}.
$$
For dead rows set $p_i=A_i=1$; for an alive singleton set $p_i=A_i=0$.

Let $G$ have undirected edges $\{i,J_i^C\}$ whenever $A_i=1$, and let $\mathcal C(G)$ be its nontrivial connected components. From frozen input coordinates set
$$
\widetilde x_i=\begin{cases}x_{J_i^C}+\sigma_{\mathrm{clone}}\zeta_i,&A_i=1,\\x_i,&A_i=0,\end{cases}
\qquad \zeta_i\overset{\mathrm{iid}}\sim\mathcal N(0,I_d).
$$
For each $C\in\mathcal C(G)$, draw one independent Haar orthogonal matrix $R_C$ and set
$$
\bar v_C=|C|^{-1}\sum_{i\in C}v_i,\qquad
\widetilde v_i=\bar v_C+\alpha_{\mathrm{restitution}}R_C(v_i-\bar v_C).
$$
For vertices outside these components set $\widetilde v_i=v_i$. All intermediate marks equal one. A persisting donor can therefore change velocity even though it does not copy a position.
:::

:::{prf:theorem} Component conservation, restitution, and shared covariance
:label: thm-eg-component-balances

For each component, with $u_i=v_i-\bar v_C$,
$$
\sum_{i\in C}\widetilde v_i=\sum_{i\in C}v_i,\qquad
\sum_{i\in C}\|\widetilde v_i-\bar v_C\|^2=
\alpha_{\mathrm{restitution}}^2\sum_{i\in C}\|u_i\|^2.
$$
Conditional on the accepted graph and all frozen velocities,
$$
\mathbb E\widetilde v_i=\bar v_C,\qquad
\operatorname{Cov}(\widetilde v_i,\widetilde v_j)=
\frac{\alpha_{\mathrm{restitution}}^2}{d}(u_i\cdot u_j)I_d,
\quad i,j\in C.
$$
At $\alpha_{\mathrm{restitution}}=0$ all component velocities become their mean. At $\alpha_{\mathrm{restitution}}=1$ total component kinetic energy is conserved.

*Proof.* The sum of the $u_i$ is zero, and a common linear map preserves that identity. Orthogonality preserves every squared relative norm. Haar invariance under $R\mapsto-R$ gives $\mathbb E R=0$, and left orthogonal invariance implies $\mathbb E[(Ru_i)(Ru_j)^\top]$ is a scalar multiple of $I_d$. Taking its trace gives that scalar as $(u_i\cdot u_j)/d$. Summing the relative-energy identity and the unchanged center-of-mass energy proves the last assertion. $\square$

These momentum sums include dead input slots. Before kinetics, all slots have become alive, so the change in alive-only momentum relative to the input is exactly the sum of the retained dead input velocities. The full-slot collision momentum change is zero. Terminal killing then removes its own measured momentum from the alive sum.
:::

:::{div} feynman-prose
Take three slots with velocities $-1,0,1$ connected in a chain. Their center-of-mass velocity is zero. In one dimension the shared Haar matrix is a single random sign, so the output is either $(-\alpha,0,\alpha)$ or $(\alpha,0,-\alpha)$. Momentum is zero in both cases. Assigning separate signs to the two nonzero relative velocities would sometimes make them point the same way and would fail the conservation identity.

The graph also tells us which errors can travel together. A measurement affects a gate; a changed gate can merge two components; the merged component uses one mean velocity and one rotation. This is why the population proof explores accepted neighborhoods instead of treating collision outputs as independent row updates.
:::


(sec-eg-stage4)=
### 6.4 Stage 4 — Kinetic perturbation and status update

Apply {prf:ref}`def-eg-baoab-canonical` independently to the rows of $(\widetilde x,\widetilde v)$, conditional on this correlated intermediate population. Use its B1–A1–O–A2–B2 sequence, final position diffusion, smooth cap, and terminal classification in exactly that order. The next swarm is $S^+=((x_i^+,v_i^+,a_i^+))_{i=1}^N$; every coordinate is retained even when $a_i^+=0$.

(sec-eg-kernel-repr)=
### 6.5 Kernel representation

:::{prf:definition} Innovation representation of the kernel
:label: def-eg-complete-kernel

For each input $S$, let $\nu_S$ be the law obtained by: drawing independent measurement and cloning companions with their input-dependent categorical weights; drawing the independent gate uniforms; forming $G$; drawing one Haar matrix per component of $G$; and drawing independent clone and kinetic Gaussian innovations. If $\Phi_h$ executes the specified stages, then
$$
\Psi_{\mathcal F_{\mathrm{EG}}}(S,B)=\int\mathbf1_B(\Phi_h(S;\omega))\nu_S(d\omega).
$$
The equivalent representation with input-independent uniforms realizes each categorical draw by inverse cumulative probabilities. Component matrices can be preassigned independently to every nonempty vertex subset and the matrix indexed by each realized component selected. Only the selected matrices are used. This supplies a fixed product innovation space without assuming independent collision outputs.
:::


(sec-eg-feller-proof)=
## Proof of the Feller Property for the Euclidean Gas Kernel

:::{prf:theorem} Feller continuity of $\Psi_{\mathcal F_{\mathrm{EG}}}$
:label: thm-euclidean-feller

For fixed $N$, continuous reward, globally Lipschitz force, positive donor and standardization denominators, $\sigma_x>0$, and Lebesgue-null $\partial D$, the canonical marked kernel maps bounded continuous functions to bounded continuous functions on
$$
\Sigma_N=(\mathbb R^d\times\mathbb R^d\times\{0,1\})^N,
$$
with the discrete topology on the marks. Restricting to terminally consistent marked states preserves this assertion. The physical and squashed coordinate metrics induce the same topology on finite states.

*Proof.* Suppose $S_k\to S$. The finite vector of discrete marks is eventually constant. If all marks are zero, the transition is the identity and the result follows. Otherwise the alive set and the singleton convention are eventually fixed.

There are finitely many possible measurement donor arrays, clone donor arrays, and acceptance arrays. Their probabilities depend continuously on $S$: Gaussian weights have positive denominators, measured features and rewards are continuous, and regularized scales and positive fitness denominators do not vanish. Fix one such discrete array. Its accepted graph and connected components are fixed. Each component mean is a continuous linear function of the frozen velocities, and the rotation and position-copy maps are continuous for each fixed set of Haar and Gaussian innovations. BAOAB and the cap are likewise continuous.

The terminal status indicator can be discontinuous only when one output position lies on $\partial D$. Conditional on all earlier innovations, independent final Gaussian position noise gives this event probability zero. Hence the marked outputs converge almost surely under the common innovations for the fixed discrete array. A bounded continuous test function then converges in expectation by dominated convergence. Sum over the finitely many discrete arrays with their continuous weights. This proves the assertion. No independence of output rows is used. $\square$
:::


(sec-eg-references)=
## References (selected)

- Federer, H. *Geometric Measure Theory*. Springer, 1969. Standard tubular-neighbourhood volume estimates and Weyl's tube formula.
- Ethier, S. N., and Kurtz, T. G. *Markov Processes: Characterization and Convergence*. Wiley, 1986. Composition properties of Feller kernels.

Bibliographic records for the framework estimates follow.

```bibtex
@book{MeynTweedie2009,
  author    = {Sean P. Meyn and Richard L. Tweedie},
  title     = {Markov Chains and Stochastic Stability},
  edition   = {2},
  publisher = {Cambridge University Press},
  address   = {Cambridge},
  year      = {2009},
  doi       = {10.1017/CBO9780511626630},
  isbn      = {9780521731829}
}
```

([Cambridge University Press & Assessment][1])

```bibtex
@book{BoucheronLugosiMassart2013,
  author    = {St{\'e}phane Boucheron and G{\'a}bor Lugosi and Pascal Massart},
  title     = {Concentration Inequalities: A Nonasymptotic Theory of Independence},
  publisher = {Oxford University Press},
  address   = {Oxford},
  year      = {2013},
  isbn      = {9780198767657}
}
```

([Oxford Academic][2])

```bibtex
@book{Federer1969,
  author    = {Herbert Federer},
  title     = {Geometric Measure Theory},
  series    = {Grundlehren der mathematischen Wissenschaften},
  volume    = {153},
  publisher = {Springer},
  address   = {Berlin Heidelberg},
  year      = {1969},
  isbn      = {9783540045052}
}
```

([SpringerLink][3])

```bibtex
@book{Santambrogio2015,
  author    = {Filippo Santambrogio},
  title     = {Optimal Transport for Applied Mathematicians: Calculus of Variations, PDEs, and Modeling},
  series    = {Progress in Nonlinear Differential Equations and Their Applications},
  volume    = {87},
  publisher = {Birkh{\"a}user},
  address   = {Cham},
  year      = {2015},
  doi       = {10.1007/978-3-319-20828-2},
  isbn      = {9783319208275}
}
```

([SpringerLink][4])

```bibtex
@book{Kechris1995,
  author    = {Alexander S. Kechris},
  title     = {Classical Descriptive Set Theory},
  series    = {Graduate Texts in Mathematics},
  volume    = {156},
  publisher = {Springer},
  address   = {New York},
  year      = {1995},
  doi       = {10.1007/978-1-4612-4190-4},
  isbn      = {9780387943749}
}
```

([SpringerLink][5])

```bibtex
@article{FritschCarlson1980,
  author  = {F. N. Fritsch and R. E. Carlson},
  title   = {Monotone Piecewise Cubic Interpolation},
  journal = {SIAM Journal on Numerical Analysis},
  year    = {1980},
  volume  = {17},
  number  = {2},
  pages   = {238--246},
  doi     = {10.1137/0717021}
}
```

([SIAM E-Books][6])

```bibtex
@article{Hyman1983,
  author  = {James M. Hyman},
  title   = {Accurate Monotonicity Preserving Cubic Interpolation},
  journal = {SIAM Journal on Scientific and Statistical Computing},
  year    = {1983},
  volume  = {4},
  number  = {4},
  pages   = {645--654},
  doi     = {10.1137/0904045}
}
```

([SIAM E-Books][7])

```bibtex
@incollection{McDiarmid1989,
  author    = {Colin McDiarmid},
  title     = {On the Method of Bounded Differences},
  booktitle = {Surveys in Combinatorics, 1989},
  editor    = {J. Siemons},
  series    = {London Mathematical Society Lecture Note Series},
  volume    = {141},
  publisher = {Cambridge University Press},
  address   = {Cambridge},
  year      = {1989},
  pages     = {148--188}
}
```

([Cambridge University Press & Assessment][8])

```bibtex
@book{AmbrosioFuscoPallara2000,
  author    = {Luigi Ambrosio and Nicola Fusco and Diego Pallara},
  title     = {Functions of Bounded Variation and Free Discontinuity Problems},
  series    = {Oxford Mathematical Monographs},
  publisher = {Oxford University Press},
  address   = {Oxford},
  year      = {2000},
  isbn      = {0198502451}
}
```

([Oxford University Press][9])

[1]: https://www.cambridge.org/core/books/markov-chains-and-stochastic-stability/E2B82BFB409CD2F7D67AFC5390C565EC?utm_source=chatgpt.com "Markov Chains and Stochastic Stability"
[2]: https://academic.oup.com/book/26549?utm_source=chatgpt.com "Concentration Inequalities: A Nonasymptotic Theory of ..."
[3]: https://link.springer.com/content/pdf/10.1007/978-3-642-62010-2.pdf?utm_source=chatgpt.com "Download book PDF"
[4]: https://link.springer.com/book/10.1007/978-3-319-20828-2?utm_source=chatgpt.com "Optimal Transport for Applied Mathematicians"
[5]: https://link.springer.com/book/10.1007/978-1-4612-4190-4?utm_source=chatgpt.com "Classical Descriptive Set Theory"
[6]: https://epubs.siam.org/doi/10.1137/0717021?utm_source=chatgpt.com "Monotone Piecewise Cubic Interpolation | SIAM Journal on ..."
[7]: https://epubs.siam.org/doi/abs/10.1137/0904045?utm_source=chatgpt.com "Accurate Monotonicity Preserving Cubic Interpolation"
[8]: https://www.cambridge.org/core/books/surveys-in-combinatorics-1989/BF6F779EA29B29CBB30715E8C406C282?utm_source=chatgpt.com "Surveys in Combinatorics, 1989"
[9]: https://global.oup.com/academic/product/functions-of-bounded-variation-and-free-discontinuity-problems-9780198502456?utm_source=chatgpt.com "Functions of Bounded Variation and Free Discontinuity ..."
