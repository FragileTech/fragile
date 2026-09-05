---
title: "Latent Fractal Gas: Algorithm and Analytic Conditions"
---

# Latent Fractal Gas: Algorithm and Analytic Conditions

(sec-latent-fractal-gas-measurement)=
## 1. State, measurement, and companion selection

:::{div} feynman-prose
Picture a walker carrying two pieces of information: its location in a learned
representation and its direction of travel. It measures reward and separation
from another walker, compares fitness, and may copy a more successful companion.
Then the swarm moves under forces, friction, and noise. Those operations specify
an algorithm. To find its long-time distribution we must examine the transition
law they produce.

There are several different probability laws to keep track of. A finite swarm
has a law on all its positions and velocities. A killed walker has a law
conditioned on its own survival. An infinite-population approximation describes
a representative walker whose dynamics depend on the population. Keeping these
objects separate tells us which question each calculation answers.

The definitions below retain the latent algorithm, including the coordinate
companion distance, inelastic collision rule, curvature-dependent noise, and
Boris-BAOAB sequence. Elementary identities have direct proofs. The later sections
state the additional conditions needed for confinement, convergence, and
population limits, with links to the Volume 2 convergence program.
:::

### 1.1. Phase space and the alive mask

:::{prf:definition} Latent swarm state and coordinate distance
:label: def-latent-fractal-gas-state

Let $N\ge2$, let $\mathcal Z$ have a specified latent chart with coordinates in
$\mathbb R^{d_z}$, and let $G(z)$ be a positive-definite metric. Walker $i$ has
position $z_i$, tangent velocity $v_i$, and covector momentum $p_i=G(z_i)v_i$.
The swarm state is $S=((z_i,v_i))_{i=1}^N\in(T\mathcal Z)^N$. Let $B$ be the
measurable region used by the alive mask and set

$$
\mathcal A(S)=\{i:z_i\in B\},\qquad k=|\mathcal A(S)|.
$$

If termination depends on an environment state, time, or learned parameters,
those variables must be included in $S$ to obtain a Markov state. A stationary
kernel requires a fixed update rule on this enlarged state.

For $\lambda_{\mathrm{alg}}\ge0$, the algorithm uses

$$
d_{\mathrm{alg}}(i,j)^2
=\|z_i-z_j\|^2+\lambda_{\mathrm{alg}}\|v_i-v_j\|^2.
$$

Periodic boundary conditions are disabled. Both differences use the specified
chart coordinates. At $\lambda_{\mathrm{alg}}=0$ this is a pseudometric on phase
space. Coordinate subtraction of tangent velocities uses their coordinate
components; it does not perform parallel transport. The rule therefore depends
on the chart even when the kinetic update uses $G$.
:::

### 1.2. Two companion draws

:::{div} feynman-prose
Each walker makes two draws: one supplies a distance measurement, and the other
supplies a possible cloning source. A Gaussian weight favors nearby companions.
Every eligible companion has positive probability in exact arithmetic, but this
only describes a draw from a finite list. Whether the entire swarm can move
between regions depends on the subsequent collision and kinetic steps.
:::

:::{prf:definition} Soft companion selection and regularized distance
:label: def-latent-fractal-gas-companions

For $k\ge2$, $\epsilon>0$, and $i\in\mathcal A$, define

$$
w_{ij}=\exp\left(-\frac{d_{\mathrm{alg}}(i,j)^2}{2\epsilon^2}\right),
\quad w_{ii}=0,\qquad
P_i(j)=\frac{w_{ij}}{\sum_{\ell\in\mathcal A\setminus\{i\}}w_{i\ell}},
\quad j\in\mathcal A\setminus\{i\}.
$$

Dead walkers draw uniformly from $\mathcal A$. The two channels sample
$c_i^{\mathrm{dist}}$ and $c_i^{\mathrm{clone}}$ from these distributions, using
fresh randomness conditional on the current state. With $\epsilon_{\mathrm{dist}}>0$,

$$
d_i=\sqrt{\|z_i-z_{c_i^{\mathrm{dist}}}\|^2
 +\lambda_{\mathrm{alg}}\|v_i-v_{c_i^{\mathrm{dist}}}\|^2
 +\epsilon_{\mathrm{dist}}^2}.
$$

This distance is smooth conditional on the companion indices and alive mask.
Differentiating the sampled fitness with frozen indices, as in
`FitnessOperator.compute_gradient` and `compute_hessian`, gives a derivative of
that conditional function. Differentiating its expectation also differentiates
the companion probabilities.

For the killed mathematical version used here, $k<2$ sends the swarm to an
absorbing state $\dagger$. The implementation in
`src/fragile/fractalai/core/companion_selection.py` instead allows a lone survivor
to select itself. That extension defines a different kernel at $k=1$; a result
using the present stopping rule must be checked again for that extension.
The common nondegenerate update is the one specified above.
:::

### 1.3. Reward and fitness

:::{prf:definition} Reward, patched standardization, and positive fitness
:label: def-latent-fractal-gas-fitness

Let $\mathcal R$ be the reward 1-form of {prf:ref}`def-reward-1-form`.
The reward channel is

$$
r_i=\langle\mathcal R(z_i),v_i\rangle_G=\mathcal R_{z_i}(v_i).
$$

Here the metric pairing uses the metric-dual vector $\mathcal R^\sharp$.
If $\mathcal R=d\Phi_{\mathrm{rew}}$, then
$r_i=\langle\operatorname{grad}_G\Phi_{\mathrm{rew}},v_i\rangle_G$.
This is directional reward along the velocity.

Using alive-only statistics, optionally localized at scale $\rho$, set

$$
z_r(i)=\frac{r_i-\mu_r}{\sigma_r'},\qquad
z_d(i)=\frac{d_i-\mu_d}{\sigma_d'},\qquad
\sigma_r'=\sqrt{\sigma_r^2+\sigma_{\min}^2},\quad
\sigma_d'=\sqrt{\sigma_d^2+\sigma_{\min}^2}.
$$

For global statistics, $\mu_q=k^{-1}\sum_{j\in\mathcal A}q_j$ and
$\sigma_q^2=k^{-1}\sum_{j\in\mathcal A}(q_j-\mu_q)^2$ for $q=r,d$.
Localized statistics use the configured normalized nonnegative alive weights
of `FitnessOperator`, with the same positive denominator regularization.
A choice of localization is part of the kernel specification.

For $A,\eta,\sigma_{\min}>0$ and
$\alpha_{\mathrm{fit}},\beta_{\mathrm{fit}}\ge0$, set

$$
g_A(u)=\frac{A}{1+e^{-u}},\qquad
r_i'=g_A(z_r(i))+\eta,\quad d_i'=g_A(z_d(i))+\eta,
$$

$$
V_i=(d_i')^{\beta_{\mathrm{fit}}}(r_i')^{\alpha_{\mathrm{fit}}}
\quad(i\in\mathcal A),\qquad V_i=0\quad(i\notin\mathcal A).
$$

The fitness and its derivatives depend on the swarm statistics as well as the
sampled distance companions. Smoothness of the distance alone does not establish
smoothness across alive-mask changes, patch thresholds, or localization cutoffs.
:::

(sec-latent-fractal-gas-cloning)=
## 2. Cloning and inelastic collisions

:::{div} feynman-prose
The fitness comparison answers a narrow question: should a walker copy the
companion whose fitness was just measured? The answer can improve the list of
stored fitness values. After copying, however, the positions, velocities, and
population statistics have changed. Recomputing fitness is a new measurement.
This distinction matters when we ask whether selection supplies a decreasing
energy function for the whole algorithm.

The velocity update has its own elementary mechanics. Subtract a collision
group's mean velocity, shrink the deviations, and put the mean back. The
mean stays fixed and the relative kinetic energy decreases. The qualifications
are about which walkers belong to each group and whether different groups
write to the same velocity.
:::

### 2.1. Decisions and updates

:::{prf:definition} Cloning step
:label: def-latent-fractal-gas-cloning

For $\epsilon_{\mathrm{clone}},p_{\max}>0$, define

$$
S_i=\frac{V_{c_i^{\mathrm{clone}}}-V_i}{V_i+\epsilon_{\mathrm{clone}}},
\qquad p_i=\min\{1,\max\{0,S_i/p_{\max}\}\}.
$$

Draw conditionally independent $B_i\sim\operatorname{Bernoulli}(p_i)$ for alive
walkers; set $B_i=1$ for dead walkers. The symbol $p_i$ in this probability
formula is distinct from the metric momentum used in kinetic formulas.
For each cloner, with independent $\zeta_i\sim\mathcal N(0,I_{d_z})$,

$$
z_i'=z_{c_i^{\mathrm{clone}}}+\sigma_x\zeta_i.
$$

A walker that does not clone keeps its position. Jitter uses the latent chart,
and an output outside $B$ is detected by the next alive-mask evaluation.

For each recipient companion, let $G_c$ contain that recipient and its cloners,
counting each member once. For the velocities read by that group update, set

$$
V_{\mathrm{COM}}=\frac1{|G_c|}\sum_{j\in G_c}v_j,\qquad
u_j=v_j-V_{\mathrm{COM}},\qquad
v_j'=V_{\mathrm{COM}}+\alpha_{\mathrm{rest}}u_j,
\quad 0\le\alpha_{\mathrm{rest}}\le1.
$$

`inelastic_collision_velocity` in `src/fragile/fractalai/core/cloning.py`
forms recipient groups from the input velocities and writes each group's result
to the output array. A recipient may itself be a cloner. Overlapping groups then
have overlapping writes; their output depends on the specified recipient order.
:::

### 2.2. What the collision conserves

:::{prf:lemma} Group momentum and relative energy
:label: lem-latent-fractal-gas-collision-energy

For one collision group with unit coordinate masses,

$$
\sum_{j\in G_c}v_j'=\sum_{j\in G_c}v_j,\qquad
\sum_{j\in G_c}\|v_j'\|^2
=|G_c|\|V_{\mathrm{COM}}\|^2
 +\alpha_{\mathrm{rest}}^2\sum_{j\in G_c}\|u_j\|^2.
$$

Consequently its coordinate kinetic energy decreases by
$\frac12(1-\alpha_{\mathrm{rest}}^2)\sum_j\|u_j\|^2$.
For disjoint groups these identities sum to global momentum conservation and
energy dissipation. Overlapping writes from the original velocities require
separate analysis. On a variable metric these coordinate identities also do
not identify a sum of covector momenta at different positions.
:::

:::{prf:proof}
The definition of the mean gives $\sum_j u_j=0$. Summing
$v_j'=V_{\mathrm{COM}}+\alpha_{\mathrm{rest}}u_j$ proves the momentum identity.
Expanding the squared norms gives a cross term
$2\alpha_{\mathrm{rest}}\langle V_{\mathrm{COM}},\sum_j u_j\rangle=0$.
Subtract the corresponding formula with $\alpha_{\mathrm{rest}}=1$ to obtain
the energy decrease.
:::

### 2.3. Selection-stage alignment

:::{prf:lemma} Frozen-fitness alignment of selection
:label: lem-latent-fractal-gas-selection-alignment

Condition on the values $V$ and clone companions used in a step. Define the
surrogate copied value

$$
V_i^{\mathrm{sel}}=(1-B_i)V_i+B_iV_{c_i^{\mathrm{clone}}}.
$$

Then

$$
\mathbb E[V_i^{\mathrm{sel}}-V_i\mid V,c^{\mathrm{clone}}]
=p_i(V_{c_i^{\mathrm{clone}}}-V_i)\ge0.
$$

Thus the mean of these surrogate values is nondecreasing in expectation.
Equivalently, $\Phi^{\mathrm{sel}}=V_{\max}-N^{-1}\sum_iV_i^{\mathrm{sel}}$
has nonpositive conditional drift relative to the frozen input values.
This statement concerns copied scores before jitter, collision, new measurement,
and kinetics.
:::

:::{prf:proof}
For an alive walker, $B_i$ has conditional mean $p_i$. If the companion's value
is no larger, the clipping rule gives $p_i=0$; otherwise both factors in the
product are nonnegative. For a dead walker $p_i=1$ and
$V_{c_i^{\mathrm{clone}}}\ge0=V_i$. Sum the identities over walkers.
:::

(sec-latent-fractal-gas-kinetics)=
## 3. Viscosity, adaptive noise, and the kinetic step

:::{div} feynman-prose
Friction reduces momentum, the thermostat supplies random momentum, and the
curl force turns it. Each operation has a different mathematical role. A
rotation can preserve kinetic energy without exploring new positions. A
Gaussian kick can fill momentum space without giving a one-step density in
position and momentum together. The complete transition needs its own analysis.

The optional viscosity is an average of neighboring velocities. Since each
walker normalizes by its own total neighbor weight, the interaction matrix is
generally asymmetric. Its conserved average uses the degrees as weights. That
small distinction changes which energy calculation is valid.
:::

### 3.1. Row-normalized viscous coupling

:::{prf:definition} State-dependent viscous force on the latent chart
:label: def-latent-fractal-gas-viscous-force

For $\ell_{\mathrm{visc}}>0$, $\nu_{\mathrm{visc}}\ge0$, and $i,j\in\mathcal A$,
set

$$
K_{ij}=\exp\left(-\frac{\|z_i-z_j\|^2}{2\ell_{\mathrm{visc}}^2}\right),
\quad \deg(i)=\sum_{j\in\mathcal A\setminus\{i\}}K_{ij},\quad
\omega_{ij}=\frac{K_{ij}}{\deg(i)}\quad(j\ne i).
$$

For $k\ge2$,

$$
\mathbf F_{\mathrm{viscous},i}(S)
=\nu_{\mathrm{visc}}\sum_{j\in\mathcal A\setminus\{i\}}
\omega_{ij}(v_j-v_i).
$$

Set this force to zero for dead walkers or when $k<2$.
The norms and velocity differences here are coordinate Euclidean quantities.
:::

:::{prf:lemma} Uniform bound in the maximum velocity norm
:label: lem-latent-fractal-gas-viscous-bounded

On $\max_{i\in\mathcal A}\|v_i\|\le V_{\mathrm{core}}$,

$$
\|\mathbf F_{\mathrm{viscous},i}\|\le2\nu_{\mathrm{visc}}V_{\mathrm{core}}.
$$

For fixed positions the linear operator on velocities has norm at most
$2\nu_{\mathrm{visc}}$ in $\|v\|_{\infty,2}=\max_i\|v_i\|$,
independently of $k$.
:::

:::{prf:proof}
Use $\omega_{ij}\ge0$ and $\sum_{j\ne i}\omega_{ij}=1$ in

$$
\|\mathbf F_{\mathrm{viscous},i}\|
\le\nu_{\mathrm{visc}}\sum_{j\ne i}\omega_{ij}(\|v_j\|+\|v_i\|)
\le2\nu_{\mathrm{visc}}\|v\|_{\infty,2}.
$$
:::

:::{prf:lemma} Dissipation for frozen positions
:label: lem-latent-fractal-gas-viscous-dissipative

Fix the positions and alive mask with $k\ge2$, and solve
$\dot v_i=\mathbf F_{\mathrm{viscous},i}$. Put

$$
D_{\mathrm{tot}}=\sum_i\deg(i),\quad
\bar v_{\deg}=\frac{\sum_i\deg(i)v_i}{D_{\mathrm{tot}}},\quad
V_{\mathrm{Var},v}^{(\deg)}
=\frac{\sum_i\deg(i)\|v_i-\bar v_{\deg}\|^2}{D_{\mathrm{tot}}}.
$$

Then $\bar v_{\deg}$ is constant and

$$
\frac{d}{dt}V_{\mathrm{Var},v}^{(\deg)}
=-\frac{2\nu_{\mathrm{visc}}}{D_{\mathrm{tot}}}
\sum_{i<j}K_{ij}\|v_i-v_j\|^2\le0.
$$

The conserved quantity is degree-weighted momentum. Ordinary total momentum
need not be conserved. The identity applies to this frozen-position ODE;
explicit force kicks, changing positions, and a position-dependent metric
require additional estimates.
:::

:::{prf:proof}
Because $K_{ij}=K_{ji}$,

$$
\sum_i\deg(i)\dot v_i
=\nu_{\mathrm{visc}}\sum_{i\ne j}K_{ij}(v_j-v_i)=0.
$$

Thus $\dot{\bar v}_{\deg}=0$. With $\delta_i=v_i-\bar v_{\deg}$, differentiate
and pair the $(i,j)$ and $(j,i)$ terms:

$$
\begin{aligned}
\frac{d}{dt}V_{\mathrm{Var},v}^{(\deg)}
&=\frac{2\nu_{\mathrm{visc}}}{D_{\mathrm{tot}}}
\sum_{i\ne j}K_{ij}\langle\delta_i,\delta_j-\delta_i\rangle\\
&=-\frac{2\nu_{\mathrm{visc}}}{D_{\mathrm{tot}}}
\sum_{i<j}K_{ij}\|\delta_i-\delta_j\|^2.
\end{aligned}
$$

The degrees are constant because the positions are frozen. Equivalence of two
norms bounds their values but does not transfer the sign of their derivatives;
it therefore cannot extend this identity to a variable $G$.
:::

### 3.2. Curvature-adapted diffusion

:::{prf:definition} Regularized diffusion factor
:label: def-latent-fractal-gas-diffusion

With distance companions, velocities, and other walker coordinates frozen as
in `FitnessOperator.compute_hessian`, define

$$
H_{\mathrm{fit}}(z_i,S)
=\nabla_{z_i}^2 V_{\mathrm{fit}}^{(i)}(S;c^{\mathrm{dist}}),
$$

$$
H_{\mathrm{reg}}
=\operatorname{Clamp}_{\epsilon_\Sigma}
 (H_{\mathrm{fit}}+\epsilon_\Sigma I),\qquad
\Sigma_{\mathrm{reg}}=H_{\mathrm{reg}}^{-1/2},\qquad\epsilon_\Sigma>0.
$$

The clamp raises each eigenvalue, or each diagonal entry in the diagonal
approximation, to at least $\epsilon_\Sigma$, as in
`KineticOperator._compute_diffusion_tensor`. If this option is disabled,
$\Sigma_{\mathrm{reg}}=I$.

For a symmetric finite Hessian this defines a positive-definite matrix and
$\Sigma_{\mathrm{reg}}\Sigma_{\mathrm{reg}}^\top\preceq\epsilon_\Sigma^{-1}I$.
A uniform positive lower diffusion bound also requires an upper bound on
$H_{\mathrm{reg}}$. Eigenvalue clipping is continuous but can fail to be
differentiable at its thresholds. Its regularity must match the theorem being
applied; the update uses the actual clamp.
:::

### 3.3. Boris-BAOAB sequence and speed cap

:::{prf:definition} Radial velocity squashing
:label: def-latent-velocity-squashing

For $V_{\mathrm{alg}}>0$ and the metric at the current position, define

$$
\psi_v(v)=V_{\mathrm{alg}}\frac{v}{V_{\mathrm{alg}}+\|v\|_G},
\qquad \psi_v(0)=0.
$$

It preserves direction and satisfies
$\|\psi_v(v)\|_G=V_{\mathrm{alg}}\|v\|_G/(V_{\mathrm{alg}}+\|v\|_G)
<V_{\mathrm{alg}}$ for finite $v$. It is smooth away from zero and $C^1$ at
zero for fixed positive-definite $G$; in general it is not $C^2$ there. Its
restriction to a line contains $a\mapsto V_{\mathrm{alg}}a/(V_{\mathrm{alg}}+|a|)$,
whose one-sided second derivatives at zero differ. Compare the Euclidean map
in {prf:ref}`lem-squashing-properties-generic`.
:::

:::{prf:definition} Latent kinetic update
:label: def-latent-fractal-gas-kinetic

Let $S$ be the post-cloning state, $p=G(z)v$, $h>0$, and
$\mathcal F=d\mathcal R$. Retain the following Boris-BAOAB sequence from
{prf:ref}`def-baoab-splitting`, with the viscous force and adaptive noise specified
above. The index $i$ is suppressed on $(z,p)$.

1. **B:** Set
   $p\leftarrow p-\frac h2\nabla\Phi_{\mathrm{eff}}(z)
     +\frac h2G(z)\mathbf F_{\mathrm{viscous},i}(S)$;
   apply the Boris rotation associated with $\beta_{\mathrm{curl}}G^{-1}\mathcal F$
   when $\mathcal F\ne0$; then repeat the same force kick.
2. **A:** Set
   $z\leftarrow\operatorname{Exp}_z(\frac h2\psi_v(G^{-1}(z)p))$.
3. **O:** With independent $\xi\sim\mathcal N(0,I_{d_z})$, set
   $p\leftarrow c_1p+c_2G^{1/2}(z)\Sigma_{\mathrm{reg}}(z,S)\xi$, where
   $c_1=e^{-\gamma h}$ and $c_2=\sqrt{(1-c_1^2)T_c}$.
4. **A:** Repeat step 2.
5. **B:** Repeat step 1 at the updated position, using the prescribed evaluation
   of the swarm force.
6. Store $v\leftarrow\psi_v(G^{-1}(z)p)$.

The evaluation policy for swarm-dependent forces and curvature is part of the
step map. Coordinates for $p$ are used at the updated positions according to
the specified chart rule. A global manifold implementation must specify chart
changes and covector transport, and must define all exponential-map evaluations
reached by the noise and jitter.
:::

:::{prf:remark} Force normalization and the Euclidean comparison
:label: rem-latent-fractal-gas-splitting-normalization

The B block displayed above contains two $h/2$ force kicks and is applied twice.
When the rotation is the identity, each end therefore gives a total $h$ kick.
For the same $\Phi_{\mathrm{eff}}$, ordinary BAOAB has one $h/2$ force kick at
each end. The displayed sequence retains the latent specification's coefficients;
its conservative limit has different force normalization from ordinary BAOAB.
The Euclidean implementation in `src/fragile/fractalai/core/kinetic_operator.py`
must be compared with that literal sequence before importing a discretization
result.

Setting $G=I$, $\mathcal F=0$, and replacing $\operatorname{Exp}$ by straight
coordinate drift removes the geometric terms. Claims of second-order weak
accuracy, invariant-measure accuracy, or convergence to a particular continuous
Lorentz-Langevin equation still require consistency of the force coefficients,
regularity of the actual cap and clamp, stability, and a treatment of killing.
:::

### 3.4. Rotation and thermostat identities

:::{prf:lemma} Boris rotation preserves the dual kinetic norm
:label: lem-latent-fractal-gas-boris-energy

Freeze $G$ during a rotation, let $M=G^{-1}$, and suppose its covector generator
$J$ satisfies $J^\top M+MJ=0$. For real $a$, the Cayley rotation
$R_a=(I-aJ)^{-1}(I+aJ)$ satisfies

$$
R_a^\top M R_a=M,\qquad
\frac12p'^\top G^{-1}p'=\frac12p^\top G^{-1}p
\quad(p'=R_ap).
$$

Thus the rotation alone does no kinetic work. The associated tangent generator
$G^{-1}\mathcal F$ and its covector representation
$\mathcal F G^{-1}$ are related by $p=Gv$; their skew-adjoint metrics are
respectively $G$ and $G^{-1}$.
:::

:::{prf:proof}
The real spectrum of a skew-adjoint operator is contained in $\{0\}$, so
$I-aJ$ is invertible. More directly, if $x=aJx$ then
$x^\top Mx=a x^\top MJx=0$, whence $x=0$.
Expand to obtain

$$
(I+aJ)^\top M(I+aJ)=(I-aJ)^\top M(I-aJ).
$$

The factors $I+aJ$ and $(I-aJ)^{-1}$ commute. Multiplying the identity by
$(I-aJ)^{-\top}$ and $(I-aJ)^{-1}$ yields $R_a^\top MR_a=M$.
For an antisymmetric matrix $\mathcal F$, $J=\mathcal F G^{-1}$ satisfies the
stated covector identity.
:::

:::{prf:lemma} Frozen-coefficient thermostat moments
:label: lem-latent-fractal-gas-ou-moments

Condition on all inputs to the O step and freeze $G$ and $\Sigma=\Sigma_{\mathrm{reg}}$.
Then

$$
\mathbb E[p'\mid p]=c_1p,\qquad
\operatorname{Cov}(p'\mid p)=
C=c_2^2G^{1/2}\Sigma\Sigma^\top G^{1/2},
$$

$$
\mathbb E[\|p'\|_{G^{-1}}^2\mid p]
=c_1^2\|p\|_{G^{-1}}^2+c_2^2\operatorname{tr}(\Sigma\Sigma^\top).
$$

If $\gamma,T_c,h>0$, $g_{\min}I\preceq G\preceq g_{\max}I$, and
$\epsilon_\Sigma I\preceq H_{\mathrm{reg}}\preceq H_* I$ on a chosen set, then

$$
\frac{c_2^2g_{\min}}{H_*}I\preceq C
\preceq\frac{c_2^2g_{\max}}{\epsilon_\Sigma}I.
$$

These are conditional momentum estimates before the final cap. A density or
minorization for the full position-momentum transition requires an additional
controllability and change-of-variables argument for the actual splitting map.
:::

:::{prf:proof}
Use $\mathbb E\xi=0$ and $\mathbb E\xi\xi^\top=I$ to compute the mean and
covariance. Expanding the squared dual norm cancels the cross term and gives
$\operatorname{tr}(G^{-1}C)=c_2^2\operatorname{tr}(\Sigma\Sigma^\top)$.
The covariance inequalities follow from
$H_*^{-1}I\preceq\Sigma\Sigma^\top\preceq\epsilon_\Sigma^{-1}I$
and the bounds on $G$.
:::

(sec-latent-fractal-gas-constants)=
## 4. Parameters and elementary bounds

:::{div} feynman-prose
A bound on a finite list of companion probabilities is inexpensive: bound each
numerator from below and the denominator from above. A bound on the swarm's
mixing rate is a different calculation. It concerns all positions, velocities,
cloning decisions, and survival events together.

The constants here record what the formulas themselves provide. A bounded
region gives a companion floor on that region. A confining potential on an
unbounded domain may instead give a moment bound. A moment bound controls the
probability of large excursions; it does not turn the whole domain into a
finite-diameter set.
:::

### 4.1. Configuration and units

:::{prf:definition} Parameters of the latent specification
:label: def-latent-fractal-gas-parameters

The table records the parameter values used in this specification. They are
reference choices rather than a claim that every implementation constructor
has the same defaults. Learned fields and environment inputs remain part of
the application data.

| Category | Symbol / Name | Default / Type | Meaning | Source | Unit |
|----------|---------------|----------------|---------|--------|------|
| Swarm | $N$ | 50 | Number of walkers | algorithm config | [count] |
| Swarm | $d_z$ | model-specific | Latent dimension | latent encoder | [count] |
| Swarm | $G$ | learned / implicit | Latent metric tensor | Metric Law in `docs/source/1_agent/05_geometry/01_metric_law.md` | [dimensionless] |
| Swarm | $B$ | application-defined | Region used by the alive mask | domain membership and environment flags | [dimensionless] |
| Swarm | `enable_cloning` | True (fixed) | Cloning is always enabled | algorithm config | [dimensionless] |
| Swarm | `enable_kinetic` | True (fixed) | Kinetic update is always enabled | algorithm config | [dimensionless] |
| Companion | `method` | softmax (fixed) | Soft companion selection kernel | {prf:ref}`def-latent-fractal-gas-companions` | [dimensionless] |
| Companion | $\epsilon$ | 0.1 | Companion kernel range | `CompanionSelection.epsilon` | [distance] |
| Companion | $\lambda_{\text{alg}}$ | 0.0 | Velocity weight in $d_{\text{alg}}$ | `CompanionSelection.lambda_alg` | [dimensionless] in normalized coordinates |
| Fitness | $\alpha_{\text{fit}}$ | 1.0 | Reward channel exponent | `FitnessOperator.alpha` | [dimensionless] |
| Fitness | $\beta_{\text{fit}}$ | 1.0 | Diversity channel exponent | `FitnessOperator.beta` | [dimensionless] |
| Fitness | $\eta$ | 0.1 | Positivity floor | `FitnessOperator.eta` | [dimensionless] |
| Fitness | $\lambda_{\text{alg}}$ | $\lambda_{\text{alg}}$ | Velocity weight used inside $d_{\text{alg}}$ for fitness distances (tied to companion selection) | `FitnessOperator.lambda_alg` | [dimensionless] |
| Fitness | $\sigma_{\min}$ | 1e-8 | Standardization regularizer | `FitnessOperator.sigma_min` | [dimensionless] |
| Fitness | $\epsilon_{\text{dist}}$ | 1e-8 | Distance smoothness regularizer | `FitnessOperator.epsilon_dist` | [dimensionless] |
| Fitness | $\epsilon_{\Sigma}$ | 1e-4 | Anisotropic regularization | Anisotropic Diffusion | [dimensionless] |
| Fitness | $A$ | 2.0 | Logistic rescale bound | `FitnessOperator.A` | [dimensionless] |
| Fitness | $\rho$ | None | Localization scale (None = global) | `FitnessOperator.rho` | [distance] |
| Cloning | $p_{\max}$ | 1.0 | Max cloning probability scale | `CloneOperator.p_max` | [dimensionless] |
| Cloning | $\epsilon_{\text{clone}}$ | 0.01 | Cloning score regularizer | `CloneOperator.epsilon_clone` | [dimensionless] |
| Cloning | $\sigma_x$ | 0.1 | Position jitter scale | `CloneOperator.sigma_x` | [distance] |
| Cloning | $\alpha_{\text{rest}}$ | 0.5 | Restitution coefficient | `CloneOperator.alpha_restitution` | [dimensionless] |
| Kinetic | $h$ | 0.01 | BAOAB time step | {prf:ref}`def-baoab-splitting` | [time] |
| Kinetic | $\gamma$ | 1.0 | Friction coefficient | {prf:ref}`def-baoab-splitting` | [1/time] |
| Kinetic | $\nu_{\mathrm{visc}}$ | 0.0 | Viscous velocity coupling strength | {prf:ref}`def-latent-fractal-gas-viscous-force` | [1/time] |
| Kinetic | $\ell_{\mathrm{visc}}$ | 1.0 | Viscous coupling length scale | `KineticOperator.viscous_length_scale` | [distance] |
| Kinetic | $T_c$ | $>0$ | Cognitive temperature | {prf:ref}`def-cognitive-temperature` | [dimensionless] |
| Kinetic | $\beta_{\text{curl}}$ | $\ge 0$ | Curl coupling strength | {prf:ref}`def-bulk-drift-continuous-flow` | [dimensionless] |
| Kinetic | $V_{\mathrm{alg}}$ | problem-dependent | Velocity cap for $\psi_v$ | {prf:ref}`def-latent-velocity-squashing` | [distance/time] |
| Kinetic | $\Phi_{\text{eff}}$ | field | Effective potential | {prf:ref}`def-effective-potential` | [dimensionless] |
| Kinetic | $\mathcal{R}$ | field | Reward 1-form | {prf:ref}`def-reward-1-form` | [dimensionless] |
| Kinetic | $u_\pi$ | policy field | Control drift | {prf:ref}`def-bulk-drift-continuous-flow` | [dimensionless] |

Positions, rewards, and the metric are normally expressed in normalized latent
units. If position and time are dimensional, $\lambda_{\mathrm{alg}}$ has units
of time squared, $\nu_{\mathrm{visc}}$ has units of inverse time, and
$p_{\max}$ is a dimensionless score scale. The reward denominator $\sigma_{\min}$
and distance denominator $\sigma_{\min}$ use their respective channel units;
sharing one numerical regularizer presupposes normalization. The temperature
sets momentum covariance in the chosen unit-mass convention. The field $u_\pi$
enters the agent dynamics only through the separately specified controlled
force; the displayed update contains the forces written in Section 3.
:::

### 4.2. Bounds on a specified core

:::{prf:definition} Local geometric bounds
:label: def-latent-fractal-gas-core

For a bounded coordinate region $B_0\subset B$ and a velocity core, suppose

$$
D_z=\sup_{z,z'\in B_0}\|z-z'\|<\infty,\qquad
\max_i\|v_i\|\le V_{\mathrm{core}},\qquad
0<g_{\min}I\preceq G(z)\preceq g_{\max}I\quad(z\in B_0).
$$

Set

$$
D_v=2V_{\mathrm{core}},\quad
D_{\mathrm{alg}}^2=D_z^2+\lambda_{\mathrm{alg}}D_v^2,\quad
m_\epsilon=\exp\left(-\frac{D_{\mathrm{alg}}^2}{2\epsilon^2}\right).
$$

A finite global diameter is available when the actual alive domain is bounded;
otherwise these constants are local to $B_0$. The metric speed cap gives a
coordinate bound $V_{\mathrm{alg}}/\sqrt{g_{\min}}$ wherever the metric lower
bound holds. It does not bound pre-cap Gaussian momenta.
:::

:::{prf:lemma} Companion Doeblin floor on the finite alive set
:label: lem-latent-fractal-gas-companion-doeblin

Suppose $k\ge2$ and every eligible pair satisfies
$d_{\mathrm{alg}}(i,j)^2\le D_{\mathrm{alg}}^2$. Let $U_i$ be uniform on
$\mathcal A\setminus\{i\}$. Then

$$
P_i(j)\ge\frac{m_\epsilon}{k-1}\quad(j\ne i),\qquad
P_i(\cdot)\ge m_\epsilon U_i(\cdot).
$$

This minorizes the companion distribution for a fixed state and walker.
The reference measure $U_i$ depends on the alive set and excludes $i$.
:::

:::{prf:proof}
Each numerator is at least $m_\epsilon$, and the denominator is a sum of
$k-1$ weights each at most one. For any subset $A$ of eligible companions,

$$
P_i(A)\ge\frac{m_\epsilon}{k-1}|A|=m_\epsilon U_i(A).
$$

This proves both the pointwise and measure inequalities. A common minorization
of the full swarm transition is a separate statement on a different state space.
:::

### 4.3. Fitness, scores, and jitter

:::{prf:lemma} Deterministic fitness and score bounds
:label: lem-latent-fractal-gas-fitness-bounds

For alive walkers,

$$
V_{\min}:=\eta^{\alpha_{\mathrm{fit}}+\beta_{\mathrm{fit}}}
\le V_i\le(A+\eta)^{\alpha_{\mathrm{fit}}+\beta_{\mathrm{fit}}}=:V_{\max}.
$$

For an alive walker and an alive clone companion,

$$
|S_i|\le S_{\max}:=\frac{V_{\max}-V_{\min}}
 {V_{\min}+\epsilon_{\mathrm{clone}}}.
$$

With $\alpha_{\mathrm{fit}}=\beta_{\mathrm{fit}}=1$, $\eta=0.1$, $A=2$, and
$\epsilon_{\mathrm{clone}}=0.01$, these bounds give
$V_{\min}=0.01$, $V_{\max}=4.41$, and $S_{\max}=220$.
For Gaussian position jitter,

$$
\operatorname{Cov}(z_i'-z_{c_i^{\mathrm{clone}}})=\sigma_x^2I,\qquad
\mathbb E\|z_i'-z_{c_i^{\mathrm{clone}}}\|^2=d_z\sigma_x^2.
$$
:::

:::{prf:proof}
The logistic rescale lies between zero and $A$. Add $\eta$, raise each positive
channel to its nonnegative exponent, and multiply. Bound the score numerator
by $V_{\max}-V_{\min}$ and its denominator below by
$V_{\min}+\epsilon_{\mathrm{clone}}$. The jitter identities are the coordinate
second moments of $\sigma_x\zeta_i$.
:::

:::{prf:lemma} Reward and standardized-channel envelopes
:label: lem-latent-fractal-gas-channel-bounds

On the core of {prf:ref}`def-latent-fractal-gas-core`, suppose
$R_{\max}=\sup_{z\in B_0}\|\mathcal R_z\|_{G^{-1}}<\infty$ and that every
alive distance companion lies in the same core. Set

$$
R_*=R_{\max}\sqrt{g_{\max}}V_{\mathrm{core}},\qquad
D_{\mathrm{dist}}=\sqrt{D_{\mathrm{alg}}^2+\epsilon_{\mathrm{dist}}^2}.
$$

Then

$$
|r_i|\le R_*,\qquad
\epsilon_{\mathrm{dist}}\le d_i\le D_{\mathrm{dist}},\qquad
|z_r(i)|\le\frac{2R_*}{\sigma_{\min}},\qquad
|z_d(i)|\le\frac{D_{\mathrm{dist}}-\epsilon_{\mathrm{dist}}}{\sigma_{\min}}.
$$
:::

:::{prf:proof}
The dual Cauchy-Schwarz inequality gives
$|\mathcal R(v_i)|\le\|\mathcal R\|_{G^{-1}}\|v_i\|_G
\le R_{\max}\sqrt{g_{\max}}V_{\mathrm{core}}$.
The distance bounds follow from its definition. Global or normalized
nonnegative local means lie in the channel's range, and each patched
standard deviation is at least $\sigma_{\min}$.
:::

:::{prf:remark} Selection intensity and a geometric comparison scale
:label: rem-latent-fractal-gas-derived-scales

Define the conditional expected fraction cloned by

$$
\lambda_{\mathrm{alg}}^{\mathrm{eff}}(S)
=\mathbb E\left[\frac1N\sum_iB_i\,\middle|\,S\right]\in[0,1].
$$

A positive lower fitness bound supplies no positive lower cloning frequency:
when all alive fitnesses agree and no walker is dead, every $p_i$ is zero.
Cloning frequency alone also gives no spatial contraction estimate.

If a bounded regular domain is specified, one may record
$\kappa_{\mathrm{conf}}^{(B)}=\lambda_1(-\Delta_G;B)$ with Dirichlet boundary
conditions. This is the principal eigenvalue of a particular killed elliptic
operator. Relating it to the latent kinetic kernel, a QSD relaxation rate, or
an entropy inequality requires an operator comparison. For even a symmetric
killed diffusion, the survival exponent and the relaxation rate of its
conditioned law involve different spectral quantities.
:::

(sec-latent-fractal-gas-finite-swarm)=
## 5. The finite swarm and its analytic conditions

:::{div} feynman-prose
A Markov kernel tells us how to take the next step. It need not return to a
bounded region, forget its initial condition, or survive for a long time.
Those are additional properties of that kernel.

For an unbounded space, a useful energy must notice a swarm that moves far
away. The bounded fitness height cannot do that: translate all walkers to a
remote location and it still lies in the same fixed numerical interval.
A confining potential can supply a position-sensitive function, but its drift
must be checked through cloning, jitter, the capped kinetic update, and killing.
The composition is where the analytic work sits.
:::

### 5.1. One full step

:::{prf:definition} Full latent update and killed swarm kernel
:label: def-latent-fractal-gas-step

For a state with at least two alive walkers, perform the following operations:

1. Evaluate rewards $r_i=\mathcal R_{z_i}(v_i)$ and the alive mask.
2. Draw $c^{\mathrm{dist}}$ using the companion kernel.
3. Compute the alive-only statistics and $V(S;c^{\mathrm{dist}})$, setting dead
   fitness values to zero.
4. Draw $c^{\mathrm{clone}}$ using the same companion rule with fresh randomness.
5. Apply the Bernoulli cloning decisions, position jitter, and the specified
   recipient-group inelastic collision updates.
6. Apply the latent kinetic sequence and store the squashed velocities.

Diagnostics include the fitness values used, companions, and cloning decisions.
All-degenerate inputs follow the cemetery convention in
{prf:ref}`def-latent-fractal-gas-companions`. Invalid chart or field evaluations
must have a specified measurable failure outcome; when such failure is killing,
it is included in the absorption time.

Let $P_N$ be the resulting kernel on the state space enlarged by $\dagger$.
Write $E_N$ for the set of nonabsorbed states and

$$
Q_N(s,A)=P_N(s,A),\quad s\in E_N,\ A\subset E_N,
\qquad \tau_\dagger=\inf\{n\ge0:S_n=\dagger\}.
$$

Thus $Q_N$ is sub-Markov. Under the step-start stopping convention, an output
with fewer than two survivors is absorbed when the next step is attempted.
This convention fixes the timing used in $Q_N$ and $\tau_\dagger$.
:::

:::{prf:theorem} Well-defined latent swarm transition
:label: thm-latent-fractal-gas-main

Suppose the state space and $B$ are standard Borel, the parameter denominators
are positive, and every deterministic update in
{prf:ref}`def-latent-fractal-gas-step` is measurable and defined at every state
where it is evaluated, with a specified measurable failure rule. Suppose the
recipient order, field evaluation policy, chart rules, and termination variables
are fixed parts of the state update. Then the algorithm defines a Markov kernel
$P_N$ on the enlarged state space.

No smoothness, recurrence, or entropy inequality is needed for this conclusion.
Such properties require the separate conditions below.
:::

:::{prf:proof}
When $k\ge2$, each finite softmax denominator is strictly positive. The
companion probabilities are measurable functions of the state. They define
finite-valued Markov kernels. The patched statistics, clipping probabilities,
Gaussian laws, and conditional Bernoulli decisions are measurable kernels as
well. Composing them with the specified measurable collision and kinetic maps
gives a Markov kernel by successive integration. The measurable failure rule
and the assignment $P_N(\dagger,\{\dagger\})=1$ complete the definition.
If environment or learner variables are present, the same argument applies to
the enlarged state including their specified transition law.
:::

### 5.2. Confinement and the complete transition

:::{prf:remark} Analytic hypotheses to verify for the latent model
:label: rem-latent-fractal-gas-analytic-hypotheses

The following conditions concern the actual kernel and its chosen state space.
They must be established for the instance under study.

1. **Geometry and regularity.** Metric eigenvalues, derivatives of the effective
   force and reward, and the regularized Hessian have the bounds required by
   the chosen argument on each relevant set. A $C^2$ fitness function gives a
   continuous Hessian; differentiating that Hessian requires more regularity.
   The actual eigenvalue clamp, speed cap, and alive-mask thresholds retain
   their stated regularity. Learned fields changing in time require uniform
   bounds and a time-dependent convergence statement, or an enlarged Markov
   model for the learner.
2. **Coercive drift on unbounded spaces.** Construct $W:E_N\to[1,\infty)$ whose
   sublevel sets control position, velocity, and boundary excursions as needed,
   and prove a one-step inequality for the complete kernel. The established
   confining-envelope or Safe Harbor analysis must control revival and jitter;
   see {doc}`../convergence_program/03_cloning` and
   {doc}`../convergence_program/06_convergence`. Bounds proved only on a core
   require a return or excursion estimate outside that core.
3. **Full-transition accessibility.** Prove a small-set or contraction estimate
   for $P_N$ or the conditioned evolution of $Q_N$. This includes transport of
   momentum noise into position, the force and geometric maps, collision-group
   dependencies, and the relevant boundary events. The finite companion floor
   and conditional Gaussian covariance alone provide neither this estimate nor
   a log-Sobolev inequality.
4. **Survival control.** Specify the absorption event and show that conditioning
   is defined for the relevant initial laws and times. A killed-chain theorem
   also needs a comparison of survival probabilities or other explicit control
   of the normalization by $\mu Q_N^n\mathbf1$. Forced revival can still fail
   when too few walkers survive.
5. **Quantitative uniformity.** Each constant must state whether it is local,
   depends on $N$, or is uniform over $N$, the learning parameters, or $h$.
   An $N$-uniform bound on a single viscous force does not supply uniform
   mixing of a $2Nd_z$-dimensional swarm.

The Euclidean operator estimates in
{doc}`../convergence_program/05_kinetic_contraction` and the composition analysis
in {doc}`../convergence_program/06_convergence` provide the corresponding
calculation for their stated Euclidean assumptions. The latent metric,
nonconservative reward, adaptive diffusion, cap, and force normalization each
need to be included when transferring those estimates.
:::

:::{prf:proposition} What a drift inequality implies directly
:label: prop-latent-fractal-gas-drift-iteration

Let $W\ge0$, set $W(\dagger)=0$ for a killed kernel, and suppose

$$
P_NW(s)\le aW(s)+b,\qquad 0\le a<1,\quad b<\infty.
$$

Then for every $n\ge0$,

$$
\mathbb E_s W(S_n)
\le a^nW(s)+b\frac{1-a^n}{1-a}.
$$

For a killed process this is an unconditioned moment bound with zero value
after absorption. Its conditioned version is obtained by dividing by
$\mathbb P_s(\tau_\dagger>n)$, when positive. A bound on moments alone gives
neither uniqueness of a stationary law nor a mixing rate.
:::

:::{prf:proof}
The tower property gives $u_{n+1}\le a u_n+b$, where
$u_n=\mathbb E_sW(S_n)$. Iterating yields
$u_n\le a^nu_0+b\sum_{j=0}^{n-1}a^j$. Summing the geometric series proves the
formula. Since $W=0$ at the cemetery,
$u_n=\mathbb P_s(\tau_\dagger>n)\mathbb E_s[W(S_n)\mid\tau_\dagger>n]$.
:::

:::{prf:remark} Fitness height and a confining Lyapunov function
:label: rem-lyapunov-classical

The diagnostic

$$
\Phi_{\mathrm{sel}}(S)=V_{\max}-\frac1N\sum_iV_i,
\qquad
\mathcal L_0(S)=\Phi_{\mathrm{sel}}(S)
 +\frac{\lambda_{\mathcal L}}{2N}\sum_i\|v_i\|_G^2
$$

is bounded when stored velocities obey the metric cap. Its sublevel sets
therefore need not confine positions in an unbounded latent chart. In addition,
$V_i\ge V_{\min}>0$ for every alive walker, so the specified fitness cannot
tend to zero at spatial infinity. Selection-stage alignment is insufficient
to prove a full-step drift for $\mathcal L_0$.

For a confining application a candidate $W$ must include an appropriate
position-dependent envelope and, where needed, velocity-position coupling and
boundary terms. Their coefficients and one-step estimates are analytic inputs.
A positive value returned by a rate formula cannot establish those estimates.
:::

### 5.3. A precise conditional QSD statement

:::{prf:definition} Finite-swarm quasi-stationary distribution
:label: def-latent-fractal-gas-qsd

A probability measure $\nu_N$ on $E_N$ is quasi-stationary for $Q_N$ if

$$
\nu_NQ_N=\alpha_N\nu_N,\qquad 0<\alpha_N\le1.
$$

Equivalently, whenever survival has positive probability,

$$
\frac{\nu_NQ_N^n}{\nu_NQ_N^n\mathbf1}=\nu_N.
$$

For nontrivial killing under $\nu_N$, $\alpha_N<1$. It follows that
$\mathbb P_{\nu_N}(\tau_\dagger>n)=\alpha_N^n$ and
$\mathbb E_{\nu_N}\tau_\dagger=(1-\alpha_N)^{-1}$ under the convention that
the initial state is not absorbed. An exponential dependence of this mean on
$N$ requires a separate estimate of $1-\alpha_N$.
:::

:::{prf:proof}
Iterate the eigenmeasure identity to obtain
$\nu_NQ_N^n=\alpha_N^n\nu_N$. Integrate the constant function one for survival,
then use $\mathbb E\tau=\sum_{n\ge0}\mathbb P(\tau>n)$ for the mean.
:::

:::{prf:proposition} QSD convergence from a full conditioned-block contraction
:label: prop-latent-fractal-gas-conditional-qsd

Fix $N$ and write $Q=Q_N$. Suppose there is $s>0$ such that
$Q\mathbf1(x)\ge s$ for all $x\in E_N$. Define
$F_n(\mu)=\mu Q^n/(\mu Q^n\mathbf1)$ for probability measures $\mu$.
Suppose that for some integer $m\ge1$ and $r\in[0,1)$,

$$
\|F_m(\mu)-F_m(\eta)\|_{\mathrm{TV}}
\le r\|\mu-\eta\|_{\mathrm{TV}}
\quad\text{for all probability measures }\mu,\eta\text{ on }E_N,
$$

where $\|\mu-\eta\|_{\mathrm{TV}}=\sup_A|\mu(A)-\eta(A)|$.
Then there is a unique QSD $\nu_N$, allowing $\alpha_N=1$ when it never
experiences killing. For $n=km+j$, $0\le j<m$,

$$
\|F_n(\mu)-\nu_N\|_{\mathrm{TV}}
\le2s^{-j}r^k\|\mu-\nu_N\|_{\mathrm{TV}}.
$$

These hypotheses concern the entire conditioned swarm update. Neither is
established here for the general latent model. When uniform survival fails
near a boundary or at infinity, a localized argument needs additional
survival and return estimates of the kind discussed in
{doc}`../convergence_program/06_convergence`.
:::

:::{prf:proof}
The space of probability measures is complete in total variation: a Cauchy
sequence converges in the Banach space of finite signed measures, and positivity
and total mass one are preserved in the limit. Hence the contraction $F_m$
has a unique fixed point $\nu_N$.

Normalized sub-Markov evolution satisfies $F_aF_b=F_{a+b}$. Thus
$F_1\nu_N$ is another fixed point of $F_m$, so $F_1\nu_N=\nu_N$.
Set $\alpha_N=\nu_NQ\mathbf1\in[s,1]$ to obtain the QSD identity.
Any QSD is a fixed point of $F_m$, proving uniqueness.

The survival lower bound gives $Q^j\mathbf1\ge s^j$ by induction. For a
sub-Markov kernel $K$ with $K\mathbf1\ge a>0$, normalize $\mu K$ and $\eta K$.
For $0\le f\le1$, both $Kf$ and $K\mathbf1$ take values in $[0,1]$.
Writing the difference of the two fractions and bounding the numerator and
normalizer differences separately gives

$$
\left\|\frac{\mu K}{\mu K\mathbf1}
 -\frac{\eta K}{\eta K\mathbf1}\right\|_{\mathrm{TV}}
\le\frac2a\|\mu-\eta\|_{\mathrm{TV}}.
$$

Apply the block contraction $k$ times and this estimate to $K=Q^j$ to obtain
the asserted bound.
:::

(sec-latent-fractal-gas-limits)=
## 6. Population limits, killing, and continuous time

:::{div} feynman-prose
Increasing the number of walkers and decreasing the time step ask different
questions. At fixed time step, a large population still makes discrete cloning
decisions. To obtain a differential equation, those decisions must have a
specified small-time scaling. An order-one chance of replacement at every
step does not become a finite-rate jump process when the steps get arbitrarily
short.

There is another source of randomness that survives a large population. Each
walker samples one distance companion. Its measured distance can remain random
even if the population distribution becomes deterministic. Averaging that
distance before applying the nonlinear fitness and clipping functions changes
the transition law. A faithful population limit must retain this conditional
randomness until the corresponding transition has been averaged.
:::

### 6.1. A candidate discrete-time population map

:::{prf:definition} Empirical measure and candidate nonlinear transition
:label: def-latent-fractal-gas-mean-field

Write $X_i^N(n)=(z_i(n),v_i(n))$ and

$$
\mu_n^N=\frac1N\sum_{i=1}^N\delta_{X_i^N(n)}.
$$

For a single-particle law $\mu$ with positive alive mass, a candidate limiting
companion distribution for an alive state $x$ is

$$
\mathsf C_\mu(x,dy)=
\frac{\exp[-d_{\mathrm{alg}}(x,y)^2/(2\epsilon^2)]\,
      \mathbf1_{\{y\text{ alive}\}}\mu(dy)}
 {\int\exp[-d_{\mathrm{alg}}(x,u)^2/(2\epsilon^2)]\,
      \mathbf1_{\{u\text{ alive}\}}\mu(du)}.
$$

The self-index exclusion removes one finite-population index. Its limiting
effect must be controlled as $N\to\infty$; excluding an index does not exclude
all other walkers at the same phase-space point.

If the complete limiting collision, measurement, revival, and kinetic rules
can be represented by a measurable law-dependent kernel $K_\mu$, the proposed
fixed-step evolution is

$$
\mu_{n+1}=\mathcal T_h(\mu_n),\qquad
\mathcal T_h(\mu)=\mu K_\mu.
$$

This notation specifies the object to derive. It does not supply a proof that
the finite algorithm converges to it. Whole-swarm survival conditioning, if
present, must also be accounted for in the construction.
:::

:::{prf:remark} The limiting fitness retains the sampled companion
:label: rem-mean-field-fitness-field-latent

For a limiting law $\mu$ and a sampled distance companion $Y\sim\mathsf C_\mu(x,\cdot)$,
write $V_{\mathrm{fit}}(x,Y;\mu)$ for the fitness obtained using limiting
population statistics. Its average

$$
\bar V_{\mathrm{fit}}(x;\mu)
=\int V_{\mathrm{fit}}(x,y;\mu)\,\mathsf C_\mu(x,dy)
$$

is a useful deterministic field. In the cloning transition the clipping and
ratio act on realized fitnesses. In general, averaging the resulting clone
probability differs from first replacing fitness by $\bar V_{\mathrm{fit}}$.
The same issue applies to a diffusion tensor formed by nonlinear functions of
a sampled Hessian. The limit must use the order of operations in the algorithm.
:::

:::{prf:remark} Substantive conditions for propagation of chaos
:label: rem-latent-fractal-gas-chaos-conditions

A propagation-of-chaos proof at fixed $h$ must address all of the following:

- Initial empirical convergence or chaotic initial data, with the moment bounds
  needed for the chosen observables and metric.
- Exchangeability, or a justified alternative describing label-dependent
  interactions. The implemented overlapping collision-group writes can depend
  on recipient order, so permutation symmetry needs to be checked for that
  rule. It cannot be inferred from the notation $N^{-1}\sum_i\delta_{X_i}$.
- Uniform control of normalized companion and localized-statistic denominators,
  sufficient alive mass, and continuity of the actual conditional fitness and
  noise laws with respect to the population measure.
- Control of shared companions, recipient group sizes, simultaneous collisions,
  and their correlations. These features enter the one-step error estimate.
- Stability and uniqueness of the limiting nonlinear evolution, with tightness
  and integrability on unbounded spaces. If conditioning is used, its survival
  denominators need their own uniform control.

The definitions and forward-equation program in
{doc}`../convergence_program/08_mean_field` identify the corresponding Euclidean
objects. Quantitative estimates for the latent update require the ingredients
above for its exact operations.

A typical proof reduces a specified error $e_n^N$ to an inequality
$e_{n+1}^N\le L_he_n^N+a_N$ with $a_N\to0$ established by a one-step coupling
or martingale calculation. Its elementary consequence is

$$
e_n^N\le L_h^ne_0^N+a_N\sum_{j=0}^{n-1}L_h^j.
$$

If $L_h<1$ uniformly in $N$, the residual contribution is at most
$a_N/(1-L_h)$. The rate $a_N$, its dimension dependence, and the existence of
such a contraction must be proved. An empirical Wasserstein error and the
error for a fixed bounded test function can have different sampling rates.
Positive-temperature finite samples continue to fluctuate at stationarity;
their full error is not generally $e^{-\kappa n}/\sqrt N$.
:::

### 6.2. Which quasi-stationary object is being approximated

:::{prf:remark} Four distinct stationary questions
:label: rem-latent-fractal-gas-stationary-objects

The relevant objects have different state spaces and equations:

| Object | State space and defining equation | Additional identification needed |
|---|---|---|
| Finite swarm QSD | $\nu_NQ_N=\alpha_N\nu_N$ on $E_N$ | Full killed-swarm survival and convergence estimates |
| Frozen single-walker QSD | $\nu Q=\alpha\nu$ on a one-walker space for a specified fixed sub-Markov $Q$ | Freeze population fields and specify the killing rule |
| Stationary nonlinear population | $\mathcal T_h(\mu_*)=\mu_*$ | Derive $\mathcal T_h$ and prove its fixed-point properties |
| Continuous-time eigenmeasure | $(\mathcal L+V)^*\nu=\lambda_0\nu$ | Establish the generator, a linear Feynman-Kac weighting $V$, boundary conditions, and the time scaling |

A tagged walker in the finite swarm is generally not Markov by itself because
its transition depends on the other walkers. A fixed single-walker killed
kernel therefore needs a specified frozen environment or an enlarged state.

The classical Fleming-Viot construction revives a particle from a survivor
when it is killed {cite}`burdzy2000fleming`. Identifying it with a conditioned-law approximation requires
that construction's actual resampling and mutation rules. Fitness-based
replacement of alive walkers, Gaussian jitter, and group changes to recipients'
velocities add further operations in the latent algorithm. Its stationary law
is consequently determined by those operations.

A linear Feynman-Kac eigenmeasure formula applies when the weighted semigroup
has actually been identified {cite}`delmoral2004feynman`. General pairwise clipped replacement with
law-dependent statistics yields a nonlinear population map, whose fixed point
requires its own analysis. These distinctions are developed in
{doc}`../convergence_program/07_discrete_qsd` and
{doc}`../convergence_program/08_mean_field`.
:::

### 6.3. Time scaling and entropy estimates

:::{prf:remark} Conditions for a continuous-time limit
:label: rem-latent-fractal-gas-continuous-time

To obtain a finite jump intensity as $h\downarrow0$, a family of cloning rules
must satisfy $p_i(h)=h\lambda_i+o(h)$ on the relevant states, or another
explicitly justified scaling. With fixed fitness parameters and $p_{\max}$,
the specified clone probabilities generally remain order one. That fixed
algorithm supplies no automatic finite-rate mutation-selection PDE.

The final radial cap is another scaling issue. For fixed $V_{\mathrm{alg}}$,
$\psi_v(v)$ differs from $v$ even as $h\to0$ at a fixed nonzero velocity.
Repeated application is therefore not an infinitesimal identity map. A
continuous-time derivation must account for this behavior. It also must check
the force normalization, consistency of the geometric maps, stochastic
regularity, tightness, and the survival or boundary terms. Changing the scaling
of these parameters defines a family of algorithms and must be stated as such.

At fixed $h$, a proved per-step factor $r_h\in(0,1)$ converts to a physical-time
rate by

$$
r_h^n=e^{-\kappa_{\mathrm{time}}nh},\qquad
\kappa_{\mathrm{time}}=-\frac{\log r_h}{h}.
$$

If $r_h=1-\kappa h$, the limiting physical-time rate is $\kappa$. If a quantity
$\delta$ is a dimensionless decrement per step, the rate is
$-\log(1-\delta)/h$. Multiplying a per-step decrement by $h$ is not this
conversion.
:::

:::{prf:proposition} Conditional entropy decay
:label: prop-latent-fractal-gas-entropy

For a specified continuous-time probability evolution with reference law
$\pi$, suppose $D(t)=D_{\mathrm{KL}}(\mu_t\|\pi)$ is finite and absolutely
continuous, and that almost everywhere

$$
\frac{d}{dt}D(t)\le-\mathcal I(t),\qquad
D(t)\le C\mathcal I(t),\qquad C>0.
$$

Then $D(t)\le e^{-t/C}D(0)$.
:::

:::{prf:proof}
Combine the inequalities to get $D'(t)\le-D(t)/C$. The function
$e^{t/C}D(t)$ is nonincreasing by absolute continuity.
:::

:::{prf:remark} Entropy hypotheses for this algorithm
:label: rem-latent-fractal-gas-entropy-conditions

Applying {prf:ref}`prop-latent-fractal-gas-entropy` requires a reference law,
a proved entropy dissipation identity for the complete evolution, and a
functional inequality with the same dissipation. Kinetic noise acts in momentum;
its conditional Gaussian law alone cannot control arbitrary position-dependent
entropy. Hypocoercive estimates may use a modified functional whose comparison
with relative entropy must also be shown.

For a killed evolution, normalization changes the evolution equation. For the
finite-step algorithm one instead needs a discrete entropy contraction or a
valid comparison with a continuous semigroup. A Gaussian covariance bound,
Dirichlet eigenvalue, or finite companion minorization supplies none of these
identifications by itself. The functional-inequality analysis in
{doc}`../convergence_program/10_kl_hypocoercive` and
{doc}`../convergence_program/15_kl_convergence` must be applied only under
its stated model and hypotheses.
:::

(sec-latent-fractal-gas-diagnostics)=
## 7. Numerical diagnostics and the convergence program

:::{div} feynman-prose
A recorded run can tell us which estimates might be promising. We can measure
the frequency of cloning, the Hessian spectrum on visited states, or how often
walkers approach the boundary. Each is useful evidence about that run. A
uniform mathematical bound must also cover states and events the run did not
visit.

The same care applies to a rate calculator. It evaluates a formula using the
numbers supplied to it. The formula becomes a rate for this swarm only after
the underlying operator inequality has been proved, its constants have been
bounded on the required domain, and its units match the chosen step size.
:::

### 7.1. Implementation correspondence and numerical proxies

:::{prf:remark} Sources and interpretation of calculated quantities
:label: rem-latent-fractal-gas-implementation

The implementations relevant to this specification are:

| Operation | Source |
|---|---|
| Coordinate distances and companion draws | `src/fragile/fractalai/core/companion_selection.py` |
| Fitness and frozen-companion derivatives | `src/fragile/fractalai/core/fitness.py` |
| Bernoulli cloning, jitter, and recipient groups | `src/fragile/fractalai/core/cloning.py` |
| Euclidean kinetics, optional viscosity and adaptive diffusion | `src/fragile/fractalai/core/kinetic_operator.py` |
| Scalar convergence formulas | `src/fragile/fractalai/convergence_bounds.py` |

The Euclidean kinetic implementation does not by itself implement the global
latent chart, covector transport, and Boris geometry specified in Section 3.
Its companion sampler also has numerical underflow fallbacks and the singleton
extension described in Section 1. Quantitative comparisons must fix the actual
implementation and configuration being analyzed.

The functions `kappa_v`, `kappa_x`, `kappa_W_cluster`, `kappa_total`, and `T_mix`
evaluate proposed component-rate and time formulas. `C_LSI_geometric`,
`KL_convergence_rate`, `kappa_QSD`, and `mean_field_error_bound` similarly return
scalar expressions. For the general latent algorithm they are diagnostic
proxies until a theorem identifies their inputs with valid uniform operator
bounds. In particular:

- The observable $\lambda_{\mathrm{alg}}^{\mathrm{eff}}$ measures replacement
  frequency, while positional contraction needs an estimate of displacement
  and fitness alignment.
- The companion floor $m_\epsilon$ bounds a finite draw. It does not determine
  a full-state Wasserstein or entropy contraction constant.
- The code's `kappa_QSD` time conversion must be checked against the exact
  relation $-\log(r_h)/h$ for a proved contraction factor.
- An expression returned by `mean_field_error_bound` must be compared with a
  proved error metric and sampling estimate, including any nonzero finite-$N$
  fluctuation contribution.

Monitoring Hessian eigenvalues, alive counts, and collision-group overlap helps
locate missing hypotheses. Such observations concern the sampled trajectory
and require further analysis to become uniform bounds.
:::

### 7.2. Where each proof belongs

:::{prf:remark} Analytic reading map
:label: rem-latent-fractal-gas-reading-map

| Question | Volume 2 source | What must be matched in the latent setting |
|---|---|---|
| What is the Euclidean transition? | {doc}`../convergence_program/02_euclidean_gas` | Coordinates, boundary convention, parameters, and kinetic normalization |
| What can cloning contract? | {doc}`../convergence_program/03_cloning` | Companion law, nondegeneracy conditions, revival, and collision groups |
| What drift and minorization does kinetics provide? | {doc}`../convergence_program/05_kinetic_contraction` | Forces, variable metric, actual noise, cap, and time step |
| How are component estimates composed? | {doc}`../convergence_program/06_convergence` | Complete-step drift, full-state mixing, and survival normalization |
| What is the discrete QSD problem? | {doc}`../convergence_program/07_discrete_qsd` | Specified killed kernel, state space, and absorption event |
| What is the population evolution? | {doc}`../convergence_program/08_mean_field` | Random measurement marks, group collisions, and parameter scaling |
| Which uniform estimates are needed for chaos? | {doc}`../convergence_program/09_propagation_chaos` | Tightness, uniqueness, and quantitative coupling |
| Which functional inequality controls entropy? | {doc}`../convergence_program/10_kl_hypocoercive`, {doc}`../convergence_program/15_kl_convergence` | The reference law, dissipation, boundary conditions, and model-specific constants |

The agent geometry supplying $G$, $\mathcal R$, and $\Phi_{\mathrm{eff}}$ is
specified in {doc}`../../1_agent/05_geometry/04_equations_motion` and
{doc}`../../1_agent/06_fields/02_reward_field`. Here the directly proved results
are the finite companion floor, conditional selection alignment, group collision
identities, frozen-position viscous dissipation, rotation and thermostat
identities, and the explicitly conditional drift, QSD, and entropy statements.
Their hypotheses and state spaces determine how they may be combined.
:::
