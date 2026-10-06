---
title: "Parameter Constraints and Tuning"
---

# Parameter Constraints and Tuning

(sec-parameter-scope)=
## 1. What a parameter constraint controls

:::{div} feynman-prose
Suppose you increase the companion range. Distant walkers become more likely to
be selected. That conclusion follows from the companion formula. Whether the
swarm now reaches its long-time distribution faster is a separate question:
the new companions also change the fitness comparisons, collision groups, and
subsequent motion.

This is how to read a parameter bound. First identify the quantity it controls.
Then identify the states on which the estimate holds. Finally, check which
convergence result uses that quantity. A lower bound on one companion
probability is a local ingredient. A decay estimate for the complete transition
is a conclusion about an entire stochastic process.

The algorithm and elementary identities are specified in
{doc}`02_fractal_gas_latent`. This chapter turns those identities into explicit
constraints, then explains the additional assumptions needed for drift,
quasi-stationarity, entropy, and population-error estimates. Choices based on
an approximate density model or observations from a run remain tuning choices.
:::

### 1.1. Objects, units, and scope

:::{prf:definition} Parameter notation
:label: def-parameter-notation

Write $\alpha=\alpha_{\mathrm{fit}}$ and $\beta=\beta_{\mathrm{fit}}$ for the
reward and diversity exponents. Let $N$ be the population size,
$k=n_{\mathrm{alive}}$ the current alive count, $h$ the kinetic time step, and
$\epsilon$ the soft companion range. The phase-space distance is

$$
d_{\mathrm{alg}}^2(i,j)=\|z_i-z_j\|^2
 +\lambda_{\mathrm{alg}}\|v_i-v_j\|^2.
$$

The coefficient $\lambda_{\mathrm{alg}}$ weights velocity in this distance.
The quantity $\lambda_{\mathrm{alg}}^{\mathrm{eff}}(S)$ is instead the expected
fraction of walkers cloned per step. At fixed $h$ one may report
$\nu_{\mathrm{clone}}(S)=\lambda_{\mathrm{alg}}^{\mathrm{eff}}(S)/h$ as an
observed frequency per unit time. Identifying it with a finite-rate jump
intensity as $h\to0$ requires a scaling argument.

Distances and rewards use normalized latent coordinates unless units are
specified otherwise. With dimensional coordinates,
$[\lambda_{\mathrm{alg}}]=\mathrm{time}^2$,
$[\gamma]=[\nu_{\mathrm{visc}}]=\mathrm{time}^{-1}$, and
$[\epsilon]=[\sigma_x]=\mathrm{distance}$.
The parameter $p_{\max}>0$ is a dimensionless score scale; clipping makes the
resulting clone probability at most one for every positive $p_{\max}$.
:::

:::{prf:remark} Analytic inputs and empirical measurements
:label: rem-parameter-evidence

A useful bound must specify its scope:

| Kind of quantity | Example | What it establishes |
|---|---|---|
| Definitional condition | $\epsilon,\eta,\sigma_{\min}>0$ | Specified denominators are positive |
| Exact local bound | $P_i(j)\ge m_\epsilon/(k-1)$ on a diameter-bounded core | A companion-draw probability |
| Proved operator estimate | $PW\le rW+b$ for the complete kernel | A moment recurrence with stated $W,r,b$ |
| Conditional theorem input | A full conditioned-kernel contraction or entropy inequality | Convergence when that input is established |
| Run statistic | An observed Hessian maximum or cloning frequency | Behavior on the sampled states |
| Tuning model | A temperature ratio from an approximate density ansatz | A proposed way to organize a parameter sweep |

A bound on a Euclidean operator applies to the latent model only after the
metric, forces, sampled fitness, collision update, boundary rule, and discrete
coefficients have been matched. On an unbounded domain, use a proved confining
envelope and return estimates; a bounded observed trajectory does not supply
a global diameter.
:::

(sec-master-constraints)=
## 2. Exact selection bounds and companion range

:::{div} feynman-prose
The fitness pipeline has two bounded positive outputs, so its range can be
computed before running the swarm. This keeps the cloning score finite. It
does not force a walker to clone: if its companion has equal fitness, its
cloning probability is zero.

For companion selection, the calculation is equally concrete. The most distant
eligible companion has a Gaussian weight bounded below by the diameter of the
chosen core. We can invert that bound to choose a range that reaches a desired
probability. The target must be possible: $k-1$ probabilities must sum to one.
:::

### 2.1. Fitness range and clone probabilities

:::{prf:definition} Fitness bounds
:label: def-parameter-fitness-bounds

For $A,\eta>0$ and $\alpha,\beta\ge0$, the alive fitness
$V_i=(d_i')^\beta(r_i')^\alpha$ satisfies

$$
V_{\min}:=\eta^{\alpha+\beta}\le V_i
\le(A+\eta)^{\alpha+\beta}=:V_{\max}.
$$

Dead walkers have $V_i=0$. The values $\alpha=\beta=1$, $\eta=0.1$, and $A=2$
give $V_{\min}=0.01$ and $V_{\max}=4.41$.
These are deterministic bounds from the logistic range and positive floor;
see {prf:ref}`lem-latent-fractal-gas-fitness-bounds`.
:::

:::{prf:definition} Cloning score bound
:label: def-parameter-cloning-score

For an alive walker with an alive companion,

$$
S_i=\frac{V_{c_i}-V_i}{V_i+\epsilon_{\mathrm{clone}}},\qquad
|S_i|\le S_{\max}:=\frac{V_{\max}-V_{\min}}
 {V_{\min}+\epsilon_{\mathrm{clone}}},\qquad
\epsilon_{\mathrm{clone}}>0.
$$

The replacement probability is

$$
p_i=\min\{1,\max\{0,S_i/p_{\max}\}\},\qquad p_{\max}>0.
$$

Dead walkers are forced to clone when a recipient is available. For the
reference fitness values and $\epsilon_{\mathrm{clone}}=0.01$, $S_{\max}=220$.
The bound concerns scores; probabilities still lie in $[0,1]$.
:::

:::{prf:proposition} Conditional bounds on cloning activity
:label: prop-parameter-cloning-activity

For an alive walker,

$$
0\le p_i\le\min\{1,S_{\max}/p_{\max}\}.
$$

If $V_{c_i}-V_i\ge\Delta>0$, then

$$
p_i\ge\min\left\{1,\frac{\Delta}
 {p_{\max}(V_{\max}+\epsilon_{\mathrm{clone}})}\right\}.
$$

Consequently a positive uniform cloning-pressure estimate requires a positive
fitness-gap event and a lower bound on its probability, in addition to the
range bounds.
:::

:::{prf:proof}
The clipping function is increasing. Apply it first to $S_i\le S_{\max}$,
and then to
$S_i\ge\Delta/(V_{\max}+\epsilon_{\mathrm{clone}})$ on the specified event.
When all alive fitnesses are equal, every alive score is zero, which shows why
a positive lower pressure does not follow from $V_{\min}>0$.
:::

### 2.2. Pointwise and measure minorization

:::{prf:proposition} Soft companion probability floor
:label: prop-parameter-doeblin-softmax

Suppose $k\ge2$ and all eligible pairs lie in a specified core with
$d_{\mathrm{alg}}(i,j)\le D_{\mathrm{alg}}$. Set

$$
D_{\mathrm{alg}}^2=D_z^2+\lambda_{\mathrm{alg}}D_v^2,\qquad
m_\epsilon=e^{-D_{\mathrm{alg}}^2/(2\epsilon^2)}.
$$

For the soft companion distribution and the uniform law $U_i$ on the
$k-1$ eligible companions,

$$
P_i(j)\ge\frac{m_\epsilon}{k-1},\qquad
P_i(\cdot)\ge m_\epsilon U_i(\cdot).
$$

The first expression is a bound for one companion; the second is a
minorization coefficient relative to a probability measure.
:::

:::{prf:proof}
Each Gaussian weight is at least $m_\epsilon$, and their sum is at most $k-1$.
Divide, then sum the pointwise inequalities over a subset of eligible indices.
This is the argument of {prf:ref}`lem-latent-fractal-gas-companion-doeblin`.
:::

### 2.3. Inverting a companion target

:::{prf:proposition} Sufficient companion range
:label: prop-parameter-kernel-bound

Fix $D_{\mathrm{alg}}>0$ and a pointwise target
$0<p_*<1/(k-1)$. The condition

$$
\boxed{\epsilon\ge
\frac{D_{\mathrm{alg}}}
 {\sqrt{2\log\!\left(1/((k-1)p_*)\right)}}}
$$

ensures that the floor in {prf:ref}`prop-parameter-doeblin-softmax` is at least
$p_*$. Alternatively, for a measure-minorization target $m_*\in(0,1)$,

$$
\epsilon\ge\frac{D_{\mathrm{alg}}}{\sqrt{2\log(1/m_*)}}
\quad\Longrightarrow\quad P_i\ge m_*U_i.
$$

These are sufficient conditions on the diameter-based estimate, not necessary
conditions for the actual probabilities. When the alive count varies and a
single pointwise target is required for every $2\le k\le N$, use $N-1$ in the
first formula. A measure target $m_*$ avoids this count dependence.
:::

:::{prf:proof}
The desired pointwise floor is equivalent to

$$
e^{-D_{\mathrm{alg}}^2/(2\epsilon^2)}\ge(k-1)p_*.
$$

Since $(k-1)p_*\in(0,1)$, taking logarithms and rearranging gives

$$
\frac{D_{\mathrm{alg}}^2}{2\epsilon^2}
\le\log\!\left(\frac1{(k-1)p_*}\right).
$$

Take positive square roots. The measure target follows by replacing
$(k-1)p_*$ by $m_*$.
A pointwise target above $1/(k-1)$ is impossible because probabilities sum to
one. At equality every eligible probability must be uniform. If
$D_{\mathrm{alg}}=0$, the companion law is already uniform for every
$\epsilon>0$. For $k=2$ there is only one eligible companion and its probability
is one regardless of distance.
:::

:::{admonition} Example: checking a proposed range
:class: feynman-added example

Take $k=50$, $D_{\mathrm{alg}}=1$, and $p_*=10^{-3}$. The sufficient bound is
$\epsilon\ge[2\log(1/0.049)]^{-1/2}\approx0.407$.
At $\epsilon=0.1$, the diameter estimate gives
$e^{-50}/49\approx3.94\times10^{-24}$.
This reports the strength of that worst-case estimate. Actual nearby companions
can have much larger probabilities, and the estimate concerns a finite draw
rather than a full-state mixing rate.
:::

(sec-parameter-kinetic-controls)=
## 3. Friction, step size, and noise

:::{div} feynman-prose
There are two places to inject randomness: the thermostat and cloning jitter.
They act at different points in the update. The thermostat acts every kinetic
step. Jitter acts on a position only when its walker clones. If no walker
clones, increasing jitter has no effect on that step.

Friction also has a precise local role. It reduces the old momentum by
$e^{-\gamma h}$ in the thermostat. This tells us how that substep forgets its
input momentum. How quickly the whole swarm explores position space depends
on the force, transport, boundary, and cloning mechanisms as well.
:::

### 3.1. Thermostat and adaptive covariance

:::{prf:proposition} Thermostat parameter identities
:label: prop-parameter-thermostat

For $\gamma,T_c,h>0$, the frozen-coefficient O step has

$$
c_1=e^{-\gamma h}\in(0,1),\qquad
c_2^2=(1-e^{-2\gamma h})T_c>0.
$$

Under a synchronous coupling of two momenta with identical frozen coefficients,

$$
\|p'-\widetilde p'\|_{G^{-1}}^2
=e^{-2\gamma h}\|p-\widetilde p\|_{G^{-1}}^2.
$$

The conditional covariance is
$C=c_2^2G^{1/2}\Sigma\Sigma^\top G^{1/2}$.
If $g_{\min}I\preceq G\preceq g_{\max}I$ and
$\epsilon_\Sigma I\preceq H_{\mathrm{reg}}\preceq H_*I$, then

$$
\frac{c_2^2g_{\min}}{H_*}I\preceq C
\preceq\frac{c_2^2g_{\max}}{\epsilon_\Sigma}I.
$$

In the isotropic case $\Sigma=I$, repeated O steps with frozen $G$ have
stationary covariance $T_cG$. These statements are before the final speed cap.
:::

:::{prf:proof}
Subtract two O updates driven by the same Gaussian vector to obtain the first
identity. The covariance bounds are
{prf:ref}`lem-latent-fractal-gas-ou-moments`.
In the isotropic case a stationary covariance $S$ solves
$S=c_1^2S+c_2^2G$, giving $S=T_cG$.
:::

:::{prf:remark} Regularity and viscosity constraints
:label: rem-parameter-regularity

A positive Hessian floor $\epsilon_\Sigma$ ensures that the inverse square root
exists and bounds its largest eigenvalue. The lower noise bound also needs
$H_*<\infty$. Smoothness across eigenvalue-clamp thresholds and the origin of
the radial cap must be checked against the regularity assumed by a kinetic or
entropy proof. Finite regularizers do not make all these maps globally smooth.

For $\nu_{\mathrm{visc}}\ge0$ and $\ell_{\mathrm{visc}}>0$, the row-normalized
viscous force satisfies
$\|F_{\mathrm{viscous},i}\|\le2\nu_{\mathrm{visc}}V_{\mathrm{core}}$ on a
coordinate velocity core. Its exact dissipated variance is degree-weighted
with frozen positions; see
{prf:ref}`lem-latent-fractal-gas-viscous-dissipative`.
These identities do not set a universal allowed ratio of viscosity to step
size for the full nonlinear map.
:::

### 3.2. A conditional friction threshold

:::{prf:proposition} Friction margin from an established dissipation estimate
:label: prop-parameter-friction-bound

Suppose a specified continuous-time model has a nonnegative functional
$\mathcal H(t)$ and a proved inequality

$$
\mathcal H'(t)\le
-\left(a\gamma-\frac{bM^2}{\lambda_h}
             -\frac{C_{\mathrm{Dob}}\nu_{\mathrm{clone}}}{\kappa_W}\right)
 \mathcal H(t),
$$

where $a,b,\lambda_h,\kappa_W>0$ and the other displayed constants are
nonnegative. Then the sufficient condition

$$
\gamma>\gamma_*:=\frac{bM^2}{a\lambda_h}
 +\frac{C_{\mathrm{Dob}}\nu_{\mathrm{clone}}}{a\kappa_W}
$$

gives exponential decay of $\mathcal H$ at rate $a(\gamma-\gamma_*)$.
If the available kinetic contribution is instead
$a\min\{\gamma,\chi\}$, positivity requires both
$\gamma>\gamma_*$ and $\chi>\gamma_*$.
:::

:::{prf:proof}
The coefficient in parentheses is $a(\gamma-\gamma_*)$; integrate the scalar
differential inequality. With the saturated kinetic term it is
$a(\min\{\gamma,\chi\}-\gamma_*)$, which is positive precisely when both
entries of the minimum exceed $\gamma_*$.
:::

:::{prf:remark} Applicability of the friction calculation
:label: rem-parameter-friction-scope

The constants $a,b$ above are analytic constants, distinct from the OU
coefficients $c_1,c_2$. The symbol $\lambda_h$ denotes a weight in the
hypocoercive functional, not the companion velocity weight or cloning
frequency. A Hessian bound $M$ alone supplies none of these constants.

The entropy analyses in {doc}`../convergence_program/10_kl_hypocoercive` and
{doc}`../convergence_program/15_kl_convergence` discuss kinetic dissipation and
cloning perturbations. Applying a threshold requires a valid dissipation
inequality for the chosen generator, a positive comparison between its
functional and the claimed error, and an estimate of the actual nonlocal
cloning contribution. For the finite-step latent map, one additionally needs
a discrete analogue or a proved comparison with that generator.

The displayed threshold is sufficient conditional on its stated inequality.
It is not a necessary condition for convergence of every Fractal Gas instance,
and increasing friction alone cannot overcome a saturated kinetic bound.
:::

### 3.3. Step size and deterministic comparison

:::{prf:proposition} Harmonic Verlet stability calculation
:label: prop-parameter-harmonic-step

For the uncapped, deterministic one-dimensional oscillator
$\ddot x=-\omega^2x$ with $\omega>0$, a velocity-Verlet step with one $h/2$
force kick at each end has matrix

$$
A_h=\begin{pmatrix}
1-\tfrac12h^2\omega^2 & h\\
-h\omega^2(1-\tfrac14h^2\omega^2)&1-\tfrac12h^2\omega^2
\end{pmatrix}.
$$

When $0<h\omega<2$, its eigenvalues are distinct and on the unit circle, so
its powers remain bounded. When $h\omega>2$, one eigenvalue has modulus larger
than one. The endpoint $h\omega=2$ is generally not power bounded.
:::

:::{prf:proof}
Direct multiplication of the two half kicks and full drift gives $A_h$.
Its determinant is one and its trace is $2-h^2\omega^2$.
For $0<h\omega<2$ the trace lies strictly between $-2$ and $2$, giving distinct
complex-conjugate roots of modulus one. Above two the roots are real and
reciprocal, with one of modulus greater than one. At equality the matrix has
a nonzero off-diagonal entry and a repeated eigenvalue $-1$, producing a
nontrivial Jordan block.
:::

:::{prf:remark} What a step-size experiment must hold fixed
:label: rem-parameter-step-size

The harmonic calculation is a comparison for ordinary Verlet coefficients.
The latent specification retains two $h/2$ force kicks in each B block at
both ends, giving a different force normalization; see
{prf:ref}`rem-latent-fractal-gas-splitting-normalization`.
Adaptive geometry, capped drift, viscosity, killing, and stochastic cloning
require estimates for their actual maps.

A universal dimensional bound such as $h<0.1$ has no meaning until time units
are fixed. If a proof gives an observable discretization error
$C_{\phi,T}h^q$, then $h\le(\varepsilon_{\mathrm{disc}}/C_{\phi,T})^{1/q}$
is sufficient for that error target. Neither the order $q$ nor its constant
follows from the BAOAB name alone.

When varying $h$, record which parameters remain fixed. Order-one clone
probabilities and the fixed radial velocity cap do not approach identity
updates as $h\to0$. A step-size sweep changes the corresponding per-time
selection and capping effects; a continuous-time interpretation needs the
scaling conditions in {prf:ref}`rem-latent-fractal-gas-continuous-time`.
:::

### 3.4. Jitter and a probabilistic locality target

:::{prf:proposition} Gaussian displacement bound
:label: prop-parameter-jitter-bound

Conditional on a clone with position jitter $\Delta z=\sigma_x\zeta$,
$\zeta\sim\mathcal N(0,I_{d_z})$,

$$
\mathbb E\|\Delta z\|^2=d_z\sigma_x^2,\qquad
\mathbb P(\|\Delta z\|>r)\le\frac{d_z\sigma_x^2}{r^2}\quad(r>0).
$$

Thus for a tolerated probability $\delta\in(0,1)$,
$\sigma_x\le r\sqrt{\delta/d_z}$ is a sufficient, conservative condition
for displacement at most $r$ with probability at least $1-\delta$.
:::

:::{prf:proof}
Sum the $d_z$ coordinate variances, then apply Markov's inequality to
$\|\Delta z\|^2$.
:::

:::{prf:remark} Noise constraints require a specified target
:label: rem-parameter-noise-scope

The often-used comparison $\sigma_x\le\epsilon$ is a locality preference,
not a deterministic displacement bound or an algorithm requirement. Gaussian
jitter has unbounded support for every $\sigma_x>0$, and its root-mean-square
displacement is $\sqrt{d_z}\sigma_x$.

To obtain an entropy rate from jitter, one must establish an inequality for
the full update and its reference measure. If a valid theorem for a specified
model gives
$\lambda_{\mathrm{ent}}\ge
 \gamma\kappa_{\mathrm{conf}}\kappa_W\sigma_x^2/C_0$
with positive, identified constants, then solving for a target rate gives

$$
\sigma_x^2\ge\frac{\lambda_{\mathrm{target}}C_0}
 {\gamma\kappa_{\mathrm{conf}}\kappa_W}.
$$

For the general latent algorithm that entropy estimate is an additional
hypothesis. The Gaussian jitter identity does not establish it. In particular,
changing the jitter affects revival, boundary loss, and the distribution to
which the update might converge. Clipping an analytically required lower bound
to a smaller preferred value leaves that target unproved; it does not establish
an alternative Wasserstein rate.
:::
(sec-convergence-rate)=
## 4. From component bounds to a convergence claim

:::{div} feynman-prose
A minimum of component rates can arise from a straightforward estimate. If
every part of an energy loses at least a certain fraction, their weighted sum
loses at least the smallest fraction. But one component can also increase
another. We need bounds on those transfers before taking the minimum.

Even after that calculation, look at what decreases. A moment bound controls
an energy-like observable. A mixing estimate compares probability laws. A
killed process also loses mass, and normalizing by survival changes the
comparison. These calculations use related ingredients, but each has its own
conclusion.
:::

### 4.1. A conditional bottleneck calculation

:::{prf:theorem} Weighted component drift
:label: thm-parameter-combined-drift

Let $P$ be the complete one-step kernel, and suppose nonnegative functions
$W_1,\ldots,W_m$ satisfy

$$
PW_j\le(1-\delta_j)W_j+\sum_{\ell\ne j}b_{j\ell}W_\ell+c_j,
\qquad 0<\delta_j\le1,\quad b_{j\ell},c_j\ge0.
$$

Choose weights $a_j>0$ and $\varepsilon_c\in[0,1)$ such that for each $\ell$,

$$
\sum_{j\ne\ell}a_jb_{j\ell}
\le\varepsilon_c a_\ell\delta_\ell.
$$

Then, for $W=\sum_ja_jW_j$,

$$
PW\le(1-\delta_{\mathrm{total}})W+C,\qquad
\delta_{\mathrm{total}}=(1-\varepsilon_c)\min_j\delta_j>0,
\quad C=\sum_ja_jc_j.
$$

All component inequalities are for the same complete kernel. Bounds for
separate cloning and kinetic substeps must first be composed using conditional
expectation. The constants are uniform in $N$ only if the full set of
inequalities, weights, and coupling bounds is uniform in $N$.
:::

:::{prf:proof}
Multiply each component inequality by $a_j$ and sum. The coefficient of
$W_\ell$ is at most

$$
a_\ell(1-\delta_\ell)+\sum_{j\ne\ell}a_jb_{j\ell}
\le a_\ell[1-(1-\varepsilon_c)\delta_\ell]
\le a_\ell(1-\delta_{\mathrm{total}}).
$$

Summing the additive terms proves the result. Iteration gives

$$
\mathbb E_sW(S_n)
\le(1-\delta_{\mathrm{total}})^nW(s)
 +\frac C{\delta_{\mathrm{total}}}
  [1-(1-\delta_{\mathrm{total}})^n].
$$
:::

:::{prf:remark} Meaning of the combined drift
:label: rem-parameter-drift-scope

The last formula bounds a moment. The ratio $C/\delta_{\mathrm{total}}$ bounds
its asymptotic value; it is not automatically an optimization error or a
radius around a QSD. A bounded fitness height with capped velocity cannot
confine positions on an unbounded chart. An appropriate $W$ needs a confining
envelope and a full-step estimate, as discussed in
{prf:ref}`rem-lyapunov-classical`.

The component program in {doc}`../convergence_program/03_cloning`,
{doc}`../convergence_program/05_kinetic_contraction`, and
{doc}`../convergence_program/06_convergence` supplies the relevant Euclidean
calculations under their stated assumptions. For killed-chain convergence one
also needs full-state accessibility, survival normalization, and the remaining
hypotheses of the chosen QSD theorem. One explicit sufficient formulation is
{prf:ref}`prop-latent-fractal-gas-conditional-qsd`.

An entropy estimate additionally requires a reference law and a functional
inequality for the actual dissipation. Positive combined drift does not by
itself establish a log-Sobolev inequality.
:::

### 4.2. Wasserstein bounds require a coupling

:::{prf:proposition} Wasserstein estimate from a transition coupling
:label: prop-parameter-wasserstein-bound

Let $K$ be a probability kernel on a metric space $(E,d)$, with finite second
moments for the laws under consideration. Suppose there is a measurable choice
of coupling of $K(x,\cdot)$ and $K(y,\cdot)$ such that

$$
\mathbb E[d(X',Y')^2\mid x,y]
\le(1-\kappa)d(x,y)^2+C_W,
\qquad 0<\kappa\le1,\quad C_W\ge0.
$$

Then

$$
W_2^2(\mu K,\eta K)
\le(1-\kappa)W_2^2(\mu,\eta)+C_W.
$$

For $C_W>0$ this is a bound with an additive residual. A strict contraction
without such a residual requires a stronger coupling estimate.
:::

:::{prf:proof}
Choose any coupling of $\mu$ and $\eta$, then conditionally apply the given
transition coupling. Its marginals are $\mu K$ and $\eta K$. Integrate the
pointwise cost bound and take an infimum over the initial couplings, or use a
sequence whose costs approach the infimum.
:::

:::{prf:remark} What must be supplied for cloning
:label: rem-parameter-cloning-coupling

A contraction constant for cloning depends on companion geometry, fitness-gap
and alignment estimates, and control of collision and mutation errors. The
cluster analysis in {doc}`../convergence_program/04_wasserstein_contraction`
addresses its specified coupling and nondegeneracy conditions. Inserting
profiled positive numbers into a product formula does not prove the required
coupling inequality for a different latent operator.

The full swarm laws live on a different space from one-walker population laws.
Their Wasserstein metrics, dimension dependence, and normalizations must be
specified before comparing rates. A probability-kernel coupling estimate also
needs additional survival control before it can be used for a normalized killed
evolution.
:::

### 4.3. Time to reach a proved error target

:::{prf:corollary} Error-target time from a geometric bound
:label: cor-parameter-mixing-time

Suppose a specified nonnegative error satisfies

$$
e_n\le A r^n+b,\qquad A>0,\quad 0<r<1,\quad b\ge0.
$$

For a target $\varepsilon>b$, the sufficient number of steps is

$$
n\ge\max\left\{0,\left\lceil
\frac{\log(A/(\varepsilon-b))}{-\log r}
\right\rceil\right\}.
$$

In units of physical time, $t=nh$ and the exponential rate is
$\kappa_{\mathrm{time}}=-\log(r)/h$.
When $e_n$ is a proved distance between laws, this gives a mixing-time bound
for that distance. When $e_n$ is a moment, it only gives the time to its stated
moment threshold.
:::

:::{prf:proof}
Solve $Ar^n\le\varepsilon-b$ by taking logarithms and using $\log r<0$.
The rate identity follows from $r^n=e^{-\kappa_{\mathrm{time}}nh}$.
:::

(sec-parameter-approximation)=
## 5. Population error and approximate tuning models

:::{div} feynman-prose
A single run has sampling fluctuations. Its expected result can also be biased
relative to a population model. These are different errors. Averaging more
independent runs can reduce fluctuations without changing the bias of the
finite-population stationary law.

The familiar square-root law comes from a variance calculation. In a swarm,
walkers share ancestors and companions, so the calculation must include
covariances. If their total covariance grows too quickly with population size,
the independent-sampling rate need not survive. The right question is which
observable and which dependence estimate we have established.
:::

### 5.1. A conditional observable bound

:::{prf:proposition} Sampling fluctuation and marginal bias
:label: prop-parameter-observable-error

Let $X_1,\ldots,X_N$ have any joint law, let
$A_N=N^{-1}\sum_i\phi(X_i)$, and suppose

$$
\sum_i\operatorname{Var}(\phi(X_i))\le NC_{\mathrm{var}},\qquad
\sum_{i\ne j}|\operatorname{Cov}(\phi(X_i),\phi(X_j))|
\le NC_{\mathrm{dep}}.
$$

For a target law $\mu$ with integrable $\phi$, define
$b_N=|\mathbb EA_N-\mu\phi|$. Then

$$
\operatorname{Var}(A_N)\le\frac{C_{\mathrm{var}}+C_{\mathrm{dep}}}{N},
\qquad
\mathbb E|A_N-\mu\phi|
\le b_N+\sqrt{\frac{C_{\mathrm{var}}+C_{\mathrm{dep}}}{N}}.
$$

The constants may depend on $\phi$, the model, and the observation time. An
$N^{-1/2}$ fluctuation bound needs them to be uniform in $N$. A bound on $b_N$
is a separate population-approximation estimate.
:::

:::{prf:proof}
Expand

$$
\operatorname{Var}(A_N)
=\frac1{N^2}\left[\sum_i\operatorname{Var}(\phi(X_i))
 +\sum_{i\ne j}\operatorname{Cov}(\phi(X_i),\phi(X_j))\right].
$$

Apply the assumed bounds. The triangle inequality and Cauchy-Schwarz give
$\mathbb E|A_N-\mu\phi|\le
|\mathbb EA_N-\mu\phi|+\sqrt{\operatorname{Var}(A_N)}$.
:::

:::{prf:remark} Choosing a population size
:label: rem-parameter-population-size

Given a proved bias bound and a fluctuation budget $\varepsilon_{\mathrm{fluc}}$,

$$
N\ge\frac{C_{\mathrm{var}}+C_{\mathrm{dep}}}
 {\varepsilon_{\mathrm{fluc}}^2}
$$

suffices for the fluctuation part of
{prf:ref}`prop-parameter-observable-error`. It supplies no lower bound on the
error of every observable. Constant observables, for example, have zero
variance. Time averages or independent repeats have their own covariance
estimates and can reduce sampling error at fixed $N$.

For empirical measures on a continuous space, total variation from a
nonatomic target is one: the finite set of sample points has empirical mass
one and target mass zero. Therefore an $N^{-1/2}$ total-variation statement
cannot describe that empirical measure. Wasserstein empirical errors have
additional dimension and moment dependence. Errors of fixed observables need
their own estimates, such as the proposition above.

The population and quantitative-error programs are in
{doc}`../convergence_program/09_propagation_chaos` and
{doc}`../convergence_program/13_quantitative_error_bounds`. Applying their
proposed estimates requires checking the particular chaos, bias, and covariance
hypotheses for the exact algorithm. Exchangeability alone does not bound the
sum of covariances by order $N$.
:::

### 5.2. A density ansatz and its limitations

:::{prf:proposition} Algebraic consequence of an iso-fitness ansatz
:label: prop-parameter-density-ansatz

Suppose an auxiliary model has a positive reward function $R$, a density
$\rho>0$, an effective dimension $D>0$, and the exact relations

$$
d(x)=c\rho(x)^{-1/D},\qquad
V(x)=d(x)^\beta R(x)^\alpha=C,
\quad c,C>0,\quad\beta>0,\quad\alpha\ge0.
$$

If $0<Z=\int R(x)^{\alpha D/\beta}\,dx<\infty$, normalization forces

$$
\rho(x)=Z^{-1}R(x)^{\gamma_{\mathrm{eff}}},\qquad
\gamma_{\mathrm{eff}}=\frac{\alpha D}{\beta}.
$$
:::

:::{prf:proof}
Substitute the distance relation into the fitness relation:
$c^\beta\rho^{-\beta/D}R^\alpha=C$.
Solving for $\rho$ gives a constant times $R^{\alpha D/\beta}$.
Integration determines the constant as $Z^{-1}$.
:::

:::{prf:remark} Relation to the implemented measurement
:label: rem-parameter-density-model

The ansatz explains the scaling formula discussed in
{doc}`../convergence_program/07_discrete_qsd`. Its two identities are
additional model assumptions. The actual algorithm samples a soft companion
at a fixed range, uses regularized and standardized channels, and applies a
logistic map and positivity floor. Its diversity measurement need not equal
$c\rho^{-1/D}$, and its fitness need not be constant at stationarity. The
latent reward is a velocity-dependent evaluation of a 1-form, which also need
not define a positive position-only function $R$.

Thus the algebraic density is an auxiliary approximation. A finite-swarm QSD,
a single killed-walker QSD, and a stationary nonlinear population require the
separate equations in {prf:ref}`rem-latent-fractal-gas-stationary-objects`.
Taking $N\to\infty$ alone establishes none of the ansatz identities.
:::

### 5.3. Temperature-ratio tuning

:::{prf:definition} Temperature-ratio proxy
:label: def-parameter-phase-ratio

For the isotropic continuous OU equation
$dv=-\gamma v\,dt+\sigma_v\,dB_t$, define
$T_{\mathrm{kin}}=\sigma_v^2/(2\gamma)$.
For the auxiliary density model with $R=e^{-U}$, $\alpha,\beta>0$, and $D>0$,
define

$$
T_{\mathrm{clone}}=\frac{\beta}{\alpha D},\qquad
\Gamma=\frac{T_{\mathrm{kin}}}{T_{\mathrm{clone}}}
=\frac{\sigma_v^2}{2\gamma}\frac{\alpha D}{\beta}.
$$

In the isotropic latent O-step convention, the corresponding continuous noise
amplitude is $\sigma_v=\sqrt{2\gamma T_c}$, so this proxy becomes
$\Gamma=T_c\alpha D/\beta$. Fixing $T_c$ and changing $\gamma$ therefore has a
different effect from fixing $\sigma_v$ and changing $\gamma$.
The proxy is undefined when its temperature parametrization has $\alpha=0$
or $\beta=0$; those exponent choices remain valid algorithm configurations.
:::

:::{admonition} Using a proxy in a tuning experiment
:class: feynman-added note

A sweep around $\Gamma=1$, or a trial interval such as $[0.5,2]$, can organize
an experiment when the auxiliary temperature picture is relevant. There is
no established universal phase transition or optimal interval at those values
for the latent algorithm. Track the actual outcomes: rewards, diversity,
survival, clipping frequency, and cost per update.

If velocity enters the companion distance, the chart dimension is $2d_z$
when both coordinates vary freely; at zero velocity weight the distance only
sees $d_z$ position coordinates. Calling either number an effective density
dimension still requires an approximation appropriate to the sampled population.
:::

(sec-parameter-selection)=
## 6. Choosing and reporting a configuration

:::{div} feynman-prose
Start with quantities whose meaning is fixed by the update. Choose the units,
regularizers, and companion target; compute what they imply. Then examine the
operator estimates available for your particular model. Some will give a
feasible parameter region. Others may need a missing geometric or survival
bound before they can be evaluated.

During tuning, report that distinction alongside the numbers. A recorded
maximum Hessian and a uniform Hessian estimate are different inputs to a rate
formula. Writing down which one was used makes the result useful to the next
person who runs the experiment.
:::

(sec-quantitative-bounds)=
### 6.1. Parameter table

:::{prf:definition} Configuration ranges and reference values
:label: def-parameter-table

Reference values below reproduce the chapter's example configuration. They
are not asserted to satisfy a model-specific convergence theorem or to match
every implementation constructor default.

| Parameter | Definitional range | Reference value | Analytic or practical condition |
|---|---|---|---|
| Population $N$ | Integer $N\ge2$ | $50$ | Companion estimates use current $k\ge2$; a population-error target needs covariance and bias bounds |
| Dimension $d_z$ | Positive integer | Model-specific | Track its appearance in jitter moments and empirical-measure errors |
| Companion range $\epsilon$ | $>0$ | $0.1$ | {prf:ref}`prop-parameter-kernel-bound` for a specified core and probability target |
| Velocity weight $\lambda_{\mathrm{alg}}$ | $\ge0$ | $0$ | Increases $D_{\mathrm{alg}}^2$ by $\lambda_{\mathrm{alg}}D_v^2$ |
| Reward exponent $\alpha$ | $\ge0$ | $1$ | Determines fitness range jointly with $\beta$ |
| Diversity exponent $\beta$ | $\ge0$ | $1$ | Positive $\beta$ is required only for the density/temperature ansatz |
| Positivity floor $\eta$ | $>0$ | $0.1$ | Gives the positive alive fitness floor |
| Logistic amplitude $A$ | $>0$ | $2$ | Bounds each rescaled channel |
| Standardization regularizer $\sigma_{\min}$ | $>0$ | $10^{-8}$ | Bounds denominators; derivative bounds also involve channel and localization data |
| Distance regularizer $\epsilon_{\mathrm{dist}}$ | $>0$ | $10^{-8}$ | Smooths the conditional distance with frozen companions |
| Localization scale $\rho$ | None or $>0$ | None | Specify alive weights and control normalized denominators |
| Clone regularizer $\epsilon_{\mathrm{clone}}$ | $>0$ | $0.01$ | Bounds score denominators |
| Clone score scale $p_{\max}$ | $>0$ | $1$ | No upper bound of one is needed; probabilities are clipped |
| Jitter $\sigma_x$ | $\ge0$ | $0.1$ | Positive for nondegenerate clone-position noise; locality is probabilistic |
| Restitution $\alpha_{\mathrm{rest}}$ | $[0,1]$ | $0.5$ | Relative group energy factor $\alpha_{\mathrm{rest}}^2$; overlapping groups need separate control |
| Friction $\gamma$ | $>0$ for a nondegenerate thermostat | $1$ | Conditional kinetic estimates and, where proved, a dissipation margin |
| Temperature $T_c$ | $>0$ for a nondegenerate thermostat | $1$ | O-step noise scale, not a full-state invariant temperature |
| Step size $h$ | $>0$ | $0.01$ | Actual splitting, force normalization, cap, and error target determine allowable values |
| Diffusion floor $\epsilon_\Sigma$ | $>0$ when adaptation is enabled | $10^{-4}$ | Lower noise bound also requires upper regularized curvature $H_*$ |
| Viscosity $\nu_{\mathrm{visc}}$ | $\ge0$ | $0$ | Force bound on a chosen velocity core |
| Viscous range $\ell_{\mathrm{visc}}$ | $>0$ | $1$ | Determines row-normalized velocity coupling |
| Speed cap $V_{\mathrm{alg}}$ | $>0$ | Model-specific | Metric speed bound and coordinate conversion depend on $G$ |
| Curl strength $\beta_{\mathrm{curl}}$ | $\ge0$ | Model-specific | Rotation geometry and full-step stability need their own estimates |

The metric $G$, alive domain $B$, reward 1-form $\mathcal R$, effective potential
$\Phi_{\mathrm{eff}}$, and any controlled field $u_\pi$ are application data
specified in {prf:ref}`def-latent-fractal-gas-parameters`. Their bounds and
regularity cannot be chosen solely by setting the scalar values in this table.
:::

### 6.2. A parameter-selection procedure

:::{prf:algorithm} Selecting parameters from explicit estimates
:label: alg-parameter-selection

**Inputs:** the exact update and boundary convention; normalized units;
a chosen position-velocity core or a confining-envelope argument; a companion
probability target; observable and survival targets; and any proved operator
estimates with their domains and constants.

1. Choose the scalar parameters in their definitional ranges. Record the field,
   chart, collision-group, singleton-survival, and kinetic evaluation rules.
2. Compute $V_{\min},V_{\max},S_{\max}$ and the regularized denominator bounds.
   A lower cloning-pressure target additionally needs a quantified fitness-gap
   event as in {prf:ref}`prop-parameter-cloning-activity`.
3. Bound $D_{\mathrm{alg}}$ on the specified core. Choose either a pointwise
   target $p_*$ or a measure target $m_*$ and solve
   {prf:ref}`prop-parameter-kernel-bound`. On an unbounded domain retain the
   separate return and excursion estimates.
4. Compute the thermostat covariance bounds from $G$, $H_*$, and
   $\epsilon_\Sigma$. Select jitter using its displacement or exploration
   target. If a valid entropy theorem imposes another noise constraint, solve
   both constraints together; an empty intersection requires a revised target
   or configuration.
5. Select $\gamma$, viscosity, and $h$ using proved estimates for the actual
   kinetic map where available. Record exploratory choices and step-size
   comparisons separately from those estimates. If a continuous-time result
   is used, check the required scaling of cloning and the cap.
6. Compose the full-step component inequalities. If applicable, solve the
   weighted coupling constraints in {prf:ref}`thm-parameter-combined-drift`.
   State the resulting moment bound. Assert a QSD or entropy rate only after
   the additional full-kernel and survival or functional-inequality hypotheses
   have also been established.
7. For an approximation target, specify the observable, population bias,
   covariance contribution, discretization error, and transient error. Choose
   $N$ and runtime using the corresponding proved bounds. Use
   {prf:ref}`cor-parameter-mixing-time` only for the error it actually bounds.

**Output:** the configured values, the exact statements they satisfy, any
conditional statements awaiting analytic inputs, and recorded tuning choices.
:::

### 6.3. Interpreting calculated rates

:::{prf:remark} Rate calculations and model correspondence
:label: rem-parameter-calculated-rates

The functions in `src/fragile/fractalai/convergence_bounds.py` evaluate scalar
formulas. For example, `kappa_v` uses the correction
$2\gamma(1-h)$, whereas the exact frozen-O-step squared-momentum decrement is
$1-e^{-2\gamma h}$. These quantities have different conventions and require
an explicit operator comparison before either becomes a full-step rate.
`kappa_total`, `kappa_W_cluster`, `C_LSI_geometric`, `T_mix`, and
`mean_field_error_bound` likewise require validated inputs and a theorem that
matches the model and error being reported.

The conditional analyses in {doc}`../convergence_program/06_convergence`,
{doc}`../convergence_program/10_kl_hypocoercive`, and
{doc}`../convergence_program/15_kl_convergence` identify the needed drift,
survival, and entropy estimates. The latent chapter records the operations and
local identities to be used when checking that correspondence. Numerical
positivity of a proposed rate is a useful diagnostic after its mathematical
meaning has been specified.
:::
