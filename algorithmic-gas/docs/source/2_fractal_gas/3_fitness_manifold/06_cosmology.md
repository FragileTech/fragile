(sec-cosmological-constants)=
# Expansion, Equilibrium, and Macroscopic Closure

:::{div} feynman-prose

Suppose a swarm has settled into a reproducible statistical state. You can
measure its mean energy, follow the growth of a region, and ask how much of its
future can be predicted from a few coarse variables. These are three different
experiments, each with a precise mathematical description.

This chapter connects them through observable bounds, geometric evolution
equations, and coarse-graining criteria. The probability results control
relaxation to a QSD. Raychaudhuri's equation controls the expansion of a specified
congruence. Closure conditions describe when a reduced state carries enough
information to predict its own future. A physical cosmology joins these pieces
through a metric, a stress tensor, a constitutive equation, and calibrated units.

:::

(sec-cosmology-intro)=
(sec-three-lambda-problem)=
(sec-three-vacuum-energies)=
## Boundary response, stationary statistics, and bulk geometry

:::{div} feynman-prose

Start by labeling the measuring instruments. A boundary experiment measures how
energy changes when an area changes. A statistical experiment averages an
observable over surviving runs. A curvature experiment measures a metric. The
first two definitions make the boundary and statistical quantities explicit,
so we can see exactly what a later geometric identification must supply.

:::

:::{prf:definition} Boundary energy response
:label: def-holographic-boundary-vacuum

For a specified boundary energy $E_\partial(A)$, define the surface response
$\Pi_\partial=-dE_\partial/dA$. Converting this quantity to a bulk energy
density requires a length scale and a constitutive prescription. The
Gaussian dilation derivative and correlation stiffness are computed in
{prf:ref}`thm-elastic-pressure`; their sign and units are retained in any
such conversion. A surface response is not itself a cosmological constant.
:::

:::{prf:definition} Stationary bulk reference
:label: def-bulk-qsd-vacuum

Let $\pi_N$ be the finite-particle QSD of the killed update established under
{prf:ref}`thm-main-convergence`. For an integrable energy observable
$E_N$, define its stationary reference $e_N=\pi_N(E_N)$ and centered
observable $\widetilde E_N=E_N-e_N$. This centering specifies the energy
zero for statistical comparisons. It does not specify an Einstein constant.
:::

:::{prf:theorem} Centered observables at the QSD
:label: thm-vanishing-bulk-vacuum

For the preceding reference and an initial law $\pi_N$,

$$
\mathbb E_{\pi_N}[\widetilde E_N(X_t)\mid\tau_\dagger>t]=0
$$

at every time with positive survival probability. Adding a constant to
the energy observable changes $e_N$ by the same constant and leaves this
identity unchanged.
:::

:::{prf:proof}
The defining QSD identity gives
$\mathcal L_{\pi_N}(X_t\mid\tau_\dagger>t)=\pi_N$.
Integrate $\widetilde E_N$ against this law. The result is
$\pi_N(E_N)-e_N=0$. Centering $E_N+C$ gives the same centered observable.
No geometric field equation enters this argument.
:::


:::{div} feynman-prose

Centering an observable is like choosing the zero on a measuring scale. Once
you subtract its stationary mean, its mean at stationarity is zero by
construction. Every individual configuration can still fluctuate, and the
conditioned law can stay fixed while fewer runs survive. The theorem uses
precisely this conditioned-law statement.

The next quantity has a measurable relaxation rate: the difference between the
current mean and its stationary value. A bound on probability laws becomes a
bound on that difference by testing the laws against the chosen observable.

:::

:::{prf:definition} Nonequilibrium observable excess
:label: def-effective-exploration-vacuum

For a conditioned law $\mu_t^N$, define
$\Delta_E(t)=\mu_t^N(E_N)-\pi_N(E_N)$.
For bounded $E_N$,
$|\Delta_E(t)|\le2\|E_N\|_\infty\|\mu_t^N-\pi_N\|_{\rm TV}$.
If $E_N$ is $L$-Lipschitz and the laws have finite first moments,
$|\Delta_E(t)|\le L W_1(\mu_t^N,\pi_N)$.
Thus the convergence estimates in the probability chapters give actual
relaxation estimates for the chosen observable. Its identification with
a gravitational energy density would require a separate calibration.
:::


:::{div} feynman-prose

For a bounded energy observable, total variation gives the error bound directly.
For a Lipschitz observable, transporting probability mass gives the corresponding
Wasserstein bound. These are the routes by which the convergence chapters turn
statistical equilibrium into a quantitative prediction for a measurement.

An Einstein constant enters through an additional field equation. The following
flat-space example is a quick way to check that distinction: a constant vacuum
stress and a constant geometric term can cancel even when both are nonzero.

:::

(sec-why-different)=
:::{prf:proposition} Stationarity does not determine the Einstein constant
:label: prop-geometric-distinction

Conservation of a stress tensor and stationarity of a probability law do
not determine a cosmological constant. Even within an Einstein
constitutive equation, a static example can have a nonzero constant.
:::

:::{prf:proof}
For the flat metric $g$, choose a constant $e$ and
$T_{ab}=-e g_{ab}$. Metric compatibility gives $\nabla_aT^{ab}=0$.
Since $G_{ab}=0$, the equation $G_{ab}+\Lambda g_{ab}=\kappa_G T_{ab}$
holds for $\Lambda=-\kappa_Ge$, which can be nonzero.
A constant shift of a statistical energy observable changes its reference
but no transition probabilities. Neither conservation nor QSD stationarity
selects $e$ or $\Lambda$.
:::

(sec-uv-regime)=
## Short-range kernels and constant-curvature metrics

:::{div} feynman-prose

A small interaction radius tells you which neighborhood a kernel samples. To
calculate curvature, you also need to know the limiting metric. Keep those two
inputs separate as you examine the explicit anti-de Sitter model below: its
negative Einstein constant determines its radius, and substitution verifies the
field equation exactly.

:::

:::{prf:definition} Short-range geometric regime
:label: def-uv-regime

The parameter $\varepsilon_c/L\to0$ denotes a kernel length small relative
to a specified geometric length $L$. Local kernel expansions require the
regularity and normalization conditions in {doc}`05_holography`.
This scale separation alone imposes no sign on a Ricci tensor.
:::

:::{prf:theorem} Negative Einstein constant and the AdS model
:label: thm-ads-boundary-uv

For a declared Einstein constant $\Lambda<0$ in dimension $d+1$, $d>1$,
the metric constructed in {prf:ref}`thm-ads-uv-regime` has radius
$L_{\rm AdS}^2=-d(d-1)/(2\Lambda)$ and satisfies
$G_{ab}+\Lambda g_{ab}=0$. Identifying this metric with a continuum limit
of the swarm requires the metric convergence and constitutive equation.
:::

:::{prf:proof}
The cited construction computes
$R_{ab}=-dL_{\rm AdS}^{-2}g_{ab}$ and
$R=-d(d+1)L_{\rm AdS}^{-2}$. Substitution gives
$G_{ab}=d(d-1)g_{ab}/(2L_{\rm AdS}^2)=-\Lambda g_{ab}$.
The sign condition fixes the radius of this explicit model, independently
of a boundary-pressure interpretation.
:::


:::{div} feynman-prose

The model calculation answers a concrete geometric question: which radius gives
the declared Einstein constant? A swarm realization then asks whether its
reconstructed metrics and stress tensors converge to the quantities in that
calculation. The same care applies to the word “expansion.” A growing cloud
variance and a growing infinitesimal spacetime volume have separate definitions.

:::

(sec-qsd-vs-exploration)=
:::{prf:definition} QSD regime
:label: def-qsd-regime

The QSD regime means that the conditioned configuration law is $\pi_N$,
or is close to it in a stated probability metric. The survival probability
of the unnormalized killed process may continue to decrease. Stationary
one-time statistics permit nonzero stationary probability currents.
:::

:::{prf:definition} Exploration observable
:label: def-exploration-regime

An exploration observable is a specified statistic such as the empirical
spatial variance or mean pairwise distance. Increasing expectation of this
statistic describes spreading in that statistic. A spacetime expansion
scalar instead requires a metric and a congruence and is defined by
$\theta=\nabla_a u^a$. Relating the two requires the volume and flux
identities of {doc}`03_curvature_gravity`.
:::

(sec-raychaudhuri-expansion)=
## Quantitative expansion estimates

:::{div} feynman-prose

Imagine carrying a small bundle of neighboring worldlines. Its volume changes
at the rate $\theta$. Raychaudhuri's equation says how that rate evolves:
curvature, shear, and vorticity all contribute. The comparison theorem keeps
every contribution in one forcing term and asks for a lower bound on that
whole term.

The function on the right of the theorem starts at zero and approaches
$\sqrt{ad}$. It gives a quantitative growth estimate for every congruence
satisfying the forcing bound, throughout the interval on which the congruence
remains smooth.

:::

:::{prf:theorem} A sufficient Raychaudhuri expansion bound
:label: thm-exploration-expansion

For a smooth geodesic timelike congruence in $d+1$ dimensions, define
$F=\omega_{ab}\omega^{ab}-\sigma_{ab}\sigma^{ab}-R_{ab}u^au^b$.
If $F(t)\ge a>0$ and $\theta(0)\ge0$, then, while the congruence is smooth,

$$
\theta(t)\ge\sqrt{ad}\tanh\!\left(\sqrt{a/d}\,t\right).
$$

This criterion retains both shear and vorticity. A spreading statistic
alone does not supply the curvature and shear bound.
:::

:::{prf:proof}
Raychaudhuri gives $\dot\theta=-\theta^2/d+F$.
Let $y=\sqrt{ad}\tanh(\sqrt{a/d}\,t)$, which solves
$\dot y=a-y^2/d$, $y(0)=0$. Then $w=\theta-y$ satisfies
$\dot w+(\theta+y)w/d=F-a\ge0$.
Multiplication by the positive integrating factor and $w(0)\ge0$ give
$w(t)\ge0$.
:::


:::{div} feynman-prose

The integrating-factor proof is the essential step. Once the comparison
solution is subtracted, the difference satisfies a linear differential
inequality with a nonnegative source. This fixes the sign of the difference
without assuming that shear vanishes.

The de Sitter metric provides a second, explicit calculation. Here we can
compute the connection, curvature, and expansion from the same metric and
check that they satisfy both Einstein's and Raychaudhuri's equations.

:::

:::{prf:theorem} An explicit de Sitter expansion model
:label: thm-bulk-can-be-ds

For $H>0$, the metric
$ds^2=-dt^2+e^{2Ht}\sum_{i=1}^d(dx^i)^2$ has

$$
\theta=dH,\quad R_{ab}=dH^2g_{ab},\quad
\Lambda=\frac{d(d-1)}2H^2>0
$$

and satisfies $G_{ab}+\Lambda g_{ab}=0$.
:::

:::{prf:proof}
The nonzero Christoffel symbols are
$\Gamma^0_{ij}=Hg_{ij}$ and $\Gamma^i_{0j}=H\delta^i_j$.
Their Ricci contraction gives $R_{00}=-dH^2$ and
$R_{ij}=dH^2g_{ij}$, hence $R=d(d+1)H^2$ and the claimed Einstein tensor.
For $u=\partial_t$, $\theta=\Gamma^i_{i0}=dH$, with zero shear and
vorticity. Raychaudhuri reads $0=-dH^2+dH^2$.
This construction proves existence of the geometric model; assigning it
to a particular swarm evolution requires a metric identification.
:::

(sec-closure-theory)=
## Predictive and dynamical coarse-graining

:::{div} feynman-prose

A macroscopic description also needs a predictive test. Suppose you replace a
complete swarm configuration by a handful of variables. Two microscopic
histories can give the same coarse history while carrying different information
about what happens next. Closure asks whether that discarded information still
improves predictions of the coarse future.

A causal state groups histories by their conditional future law. You can think
of it as the complete forecast associated with a history. Information closure
asks whether observing the microscopic past improves the forecast of the
macroscopic future. Computational closure asks whether the microscopic causal
state determines the macroscopic causal state. The distinction matters in the
finite-bit example below.

:::

:::{prf:definition} Predictive causal states
:label: def-epsilon-machine

Two micro-pasts are equivalent when their conditional micro-future laws
coincide. The equivalence classes form the predictive causal states
$\Sigma_X$. Define $\Sigma_Y$ similarly for a deterministic coarse-graining
$Y_t=f(X_t)$. Conditional laws are understood almost surely; for finite
alphabets all entropy identities below use finite windows or finite
mutual information. The causal state is sufficient for the future by
its definition: it records precisely the conditional future law.
:::

:::{prf:definition} Information closure
:label: def-information-closure-cosmo

With $X^-$ denoting the micro-past, $Y^-$ its coarse-graining, and $Y^+$
the macro-future, information closure means
$I(Y^+;X^-\mid Y^-)=0$. Equivalently,
$\mathcal L(Y^+\mid X^-)=\mathcal L(Y^+\mid Y^-)$ almost surely.
:::

:::{prf:definition} Computational closure
:label: def-computational-closure-cosmo

Computational closure means that a well-defined map
$\pi:\Sigma_X\to\Sigma_Y$ sends the causal state of a micro-past to the
causal state of its macro-past.
:::

:::{prf:definition} Preservation of causal equivalence
:label: def-causal-closure-cosmo

Causal closure means that equal micro causal states give equal macro
causal states after coarse-graining the past. This is the condition that
the map in {prf:ref}`def-computational-closure-cosmo` be well-defined.
:::

:::{prf:theorem} Closure implications and their converse
:label: thm-closure-equivalence-cosmo

With these definitions, computational and causal closure are equivalent.
Information closure implies both. The converse need not hold, including
for finite-valued stationary processes.
:::

:::{prf:proof}
A map on equivalence classes is well-defined exactly when representatives
from the same class have the same image; this proves the first equivalence.
Under information closure the macro-future law given a macro-past equals
its law given the micro-past. Equal micro causal states give equal
micro-future laws and hence equal macro-future laws. They therefore give
equal macro causal states.

For the converse, let $(B_t)_{t\in\mathbb Z}$ be independent fair bits and
$X_t=(B_t,B_{t+1})$, $Y_t=B_t$. The macro-process is independent, so its
causal-state space has one point and computational closure holds. But the
micro-past through time $t$ contains $B_{t+1}$, while the macro-past does
not. Thus $I(Y_{t+1};X_{\le t}\mid Y_{\le t})=\log2>0$.
:::


:::{div} feynman-prose

In the example, the coarse sequence is a string of independent fair bits, so
all coarse pasts give the same forecast. Its causal-state space therefore has
one point. Yet a microscopic observation includes the next bit, which improves
the prediction of the next coarse observation by one bit. A map between causal
states exists even though the discarded information helps prediction.

For a Markov model, there is a direct operational check. Group microscopic
states into blocks and add the transition rates into each target block. If
those sums agree for all states in a starting block, a macro-observer can use
one transition rule for that block. An approximate equality gives the
accumulated residual bound in the next proposition.

:::

:::{prf:proposition} Exact and approximate Markov closure
:label: prop-cosmo-generator-closure

For a finite Markov chain with rates $q(x,z)$ and projection $f$, the
macro-process is Markov for every initial law if
$\sum_{z:f(z)=b}q(x,z)$ depends on $x$ only through $f(x)$ for every block
$b$. More generally, suppose a generator $L_N$ and proposed macro-generator
$\overline L$ satisfy, for a macro test function $\varphi$,

$$
\sup_x|L_N(\varphi\circ f)(x)-\overline L\varphi(f(x))|\le\epsilon_N.
$$

Then the expected Dynkin residual over an interval of length $t$ is at
most $t\epsilon_N$ in absolute value. The same bound holds after
multiplication by a past-measurable variable of absolute value at most one.
:::

:::{prf:proof}
Under the block-rate condition,
$L(\varphi\circ f)(x)=\sum_b\overline q(f(x),b)
[\varphi(b)-\varphi(f(x))]$. The backward equation preserves block-constant
functions, so its semigroup induces the claimed macro-transition matrix.
For the approximate statement, subtract the two generator integrals in
Dynkin's martingale formula. The remaining integral has absolute value
at most $t\epsilon_N$. The martingale increment has zero expectation,
also against a bounded variable measurable at the starting time.
:::

(sec-cosmological-observations)=
## Relating model parameters to observables

:::{div} feynman-prose

To compare a model with a cosmological measurement, give time and length their
physical units and state the field equation being fitted. The first two
propositions then become algebra: one converts a dimensionless density
parameter into a curvature constant, and the other moves that constant between
the geometric and stress sides of Einstein's equation.

The energy density $e_{\rm vac}$ and mass density $\rho_{\rm vac}$ differ by
$c^2$. Keeping that factor visible prevents a statistical energy scale from
being inserted into a curvature formula with incompatible units.

:::

:::{prf:proposition} Cosmological-constant inference within a flat model
:label: prop-observed-lambda

Within a flat Friedmann model with constant vacuum term, define
$\Omega_\Lambda=\Lambda c^2/(3H_0^2)$. Then
$\Lambda=3H_0^2\Omega_\Lambda/c^2$.
This inference uses physical time and length units and the specified
cosmological model. An algorithmic observable cannot supply $\Lambda$
without a map to those units and to the model observables.
:::

:::{prf:proof}
Rearrange the definition of $\Omega_\Lambda$. The units are
$[H_0^2/c^2]=\mathrm{length}^{-2}$, as required for curvature.
:::

:::{prf:proposition} Vacuum energy and pressure in the Einstein convention
:label: prop-dark-energy-exploration

A cosmological term moved to the stress side has
$T^{\rm vac}_{ab}=-\Lambda c^4g_{ab}/(8\pi G_N)$. Its energy density,
mass density, and pressure are respectively

$$
e_{\rm vac}=\frac{\Lambda c^4}{8\pi G_N},\quad
\rho_{\rm vac}=\frac{\Lambda c^2}{8\pi G_N},\quad
P_{\rm vac}=-e_{\rm vac}.
$$

Thus $w=P_{\rm vac}/e_{\rm vac}=-1$ when $e_{\rm vac}\ne0$.
:::

:::{prf:proof}
Move $\Lambda g_{ab}$ to the right of the Einstein equation and divide
by $8\pi G_N/c^4$. Compare the resulting tensor with
$T_{ab}=(e+P)u_au_b+Pg_{ab}$. The coefficient of $u_au_b$ is zero,
so $P=-e$; the coefficient of $g_{ab}$ determines $e$.
:::


:::{div} feynman-prose

The relation $P=-e$ follows from the form of a vacuum stress tensor in the
stated convention. It does not require a particular microscopic mechanism.
Conversely, observing expansion does not determine that tensor. Milne
coordinates give a particularly sharp example: an expanding family of
observers lives in flat spacetime with zero cosmological constant.

:::

(sec-de-sitter-conjecture)=
:::{prf:theorem} Expansion does not determine the sign of the Einstein constant
:label: thm-de-sitter-resolution

Positive expansion of a geodesic congruence is compatible with both the
positive constant in {prf:ref}`thm-bulk-can-be-ds` and a zero constant.
:::

:::{prf:proof}
The de Sitter example was computed above. In Minkowski space introduce
Milne coordinates $T=\tau\cosh\chi$, $R=\tau\sinh\chi$. The metric is
$-d\tau^2+\tau^2h_{\mathbb H^d}$ and is flat because this is a coordinate
change of the Minkowski metric. The comoving congruence has
$\theta=d/\tau>0$, while $R_{ab}=0$ and $\Lambda=0$ in vacuum.
Consequently expansion alone determines neither a nonzero Einstein
constant nor a distance from a statistical QSD. These examples are
classical geometric constructions, with no implication for a quantum-gravity
existence conjecture.
:::


:::{div} feynman-prose

These examples suggest a practical division of measurements. Estimate a
statistical relaxation error from the conditioned law; estimate geometric
expansion from a metric and a congruence; test predictive closure from the
chosen coarse variables. A physical identification must make their units and
dynamical equations agree. The remarks below summarize the assumptions attached
to each use.

:::

(sec-fg-cosmology-physical-interpretation)=
:::{prf:remark} Operational equilibrium tests
:label: rem-vacuum-algorithmic

A stationary-observable test uses the finite-particle QSD convergence
bounds, while a metric-expansion test uses a specified congruence and the
volume derivative. Their measurement procedures and hypotheses are separate.
:::

:::{prf:remark} Statistical spreading and geometric expansion
:label: rem-exploration-exploitation-cosmic

The quantitative Raychaudhuri criterion retains the full forcing
$\omega^2-\sigma^2-R(u,u)$. Neither a reward gradient nor an empirical
variance increment fixes that forcing without an established metric map.
:::

:::{prf:remark} Physical calibration
:label: rem-cc-problem-reframed

The energy-reference identity does not resolve a physical vacuum-energy
problem. A physical identification requires an effective stress tensor,
a constitutive equation, and independently specified units.
:::

:::{prf:remark} Selection of a stationary state
:label: rem-multiverse-qsd

Under the proved contraction conditions, the mean-field stationary state
is unique; under the finite-particle minorization conditions, the QSD is
unique. Other parameter regimes are not assigned multiple stationary states
without an existence and nonuniqueness argument.
:::

(sec-conclusions-cosmology)=
(sec-symbols-cosmology)=
## Analytical dependencies

:::{div} feynman-prose

The probability results already proved in this volume supply the statistical
part of the argument. Use the theorem for the actual evolution and parameter
regime: a finite-particle QSD, a mean-field stationary law, and a spacetime
metric answer different questions.

:::


:::{div} feynman-added

| Question | Analytical input | Conclusion used here |
|---|---|---|
| How quickly does an observable settle? | [Finite-particle QSD theory](../convergence_program/06_convergence.md) and [entropy convergence](../convergence_program/15_kl_convergence.md), with their stated hypotheses | Bounds on the conditioned observable excess. |
| Is the mean-field stationary law selected uniquely? | [Mean-field well-posedness and contraction](../convergence_program/09_propagation_chaos.md) | Existence, uniqueness, and attraction in the proved contraction regime. |
| How is volume growth related to geometry? | [Curvature and transport identities](03_curvature_gravity.md) | A specified metric, congruence, and flux define the expansion measurement. |
| Which boundary quantities enter a bulk model? | [Boundary response and holography](05_holography.md) | The boundary derivative, normalization, and constitutive assumptions remain explicit. |
| Does a coarse model predict its own future? | {prf:ref}`thm-closure-equivalence-cosmo` and {prf:ref}`prop-cosmo-generator-closure` | Exact closure implications and quantitative generator residuals. |

:::
