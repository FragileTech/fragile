# Convergence, Survival, and Parameter Dependence

:::{div} feynman-prose
The contraction calculation becomes more informative when its constants retain their landscape dependence. [Structural landscape convergence](06a_structural_landscape_convergence.md) follows that dependence through regional kinetic and keystone estimates, basin communication and tail control. Its convergence statements specify which additional certificates make the complete-update argument applicable to a given landscape.
:::

:::{div} feynman-prose
A cloning step can pull positions together while adding kinetic energy. A kinetic
step can dissipate that energy while moving positions apart. To understand the
whole algorithm, we must keep an account of what each step transfers to the
other. This chapter makes that account explicit.

There are then two different probability questions. Does a conservative swarm
forget its initial state? Does a swarm that can become extinct forget its initial
state **conditional on survival**? The second question includes a change of
weights: initial states with better survival prospects contribute more to the
conditional law. We carry that normalization through the proofs.

The results below assemble the cloning estimates, kinetic moment calculations,
and comparison arguments already developed in this volume. They also state
exactly which additional kernel or entropy estimates produce mixing,
concentration, and long survival. Parameter tuning uses the constants in those
estimates; measured proxies remain identified as measurements.
:::

(sec-convergence-objects)=
## 1. The Process and the Quantities Being Controlled

:::{div} feynman-prose
First choose what one state means. Here it means the entire finite swarm,
including its alive indicators. A distribution of such states is different from
the distribution of one walker and from a deterministic density obtained as the
population tends to infinity. Keeping these objects separate prevents a result
about one of them from being used as a theorem about another.
:::

:::{prf:definition} Swarm state and cemetery convention
:label: def-cemetery-state

Fix a population size $N\ge2$ and timestep $h>0$. Let $E_N$ be the measurable
space of admissible swarm states with at least two alive walkers. Use the
$k<2$ cemetery convention of
{doc}`../1_the_algorithm/02_fractal_gas_latent`: the process is sent to an
absorbing state $\dagger$ when this condition fails. Write

$$
T_\dagger=\inf\{n\ge0:S_n=\dagger\}.
$$

The killed one-step kernel is

$$
Q(S,A)=\mathbb P_S(S_1\in A,\ T_\dagger>1),\qquad A\subset E_N.
$$

Thus $Q1\le1$. Its missing mass is the probability of absorption. Extend every
nonnegative moment observable by zero at $\dagger$. A conservative kernel,
written $P$, instead satisfies $P1=1$ on its specified state space. A software
convention that continues singleton swarms defines a different kernel and must
be analyzed with its own absorbing set.
:::

:::{prf:definition} Conditional evolution and quasi-stationarity
:label: def-qsd

For a probability measure $\mu$ with $\mu Q^n1>0$, define

$$
\Phi_n(\mu)=\frac{\mu Q^n}{\mu Q^n1}.
$$

A quasi-stationary distribution (QSD) is a probability measure $\nu_N$ on $E_N$
with

$$
\nu_NQ=\alpha_N\nu_N,\qquad
\alpha_N=\nu_NQ1\in(0,1].
$$

It satisfies $\Phi_n(\nu_N)=\nu_N$ and
$\mathbb P_{\nu_N}(T_\dagger>n)=\alpha_N^n$. When absorption has positive
probability under $\nu_N$, $\alpha_N<1$.

Throughout the chapter,
$\|\mu-\eta\|_{\rm TV}=\sup_A|\mu(A)-\eta(A)|$ for probabilities. A finite-swarm
QSD is a measure on $E_N$; its one-particle marginal is not by definition a QSD
of a frozen single-particle kernel.
:::

:::{prf:remark} Completed population-uniform conservative alive-law mixing
:label: rem-convergence-completed-uniform-alive-law

For the current-frame conservative nonviscous canonical gas,
{prf:ref}`thm-slcw-finite-uniform-law` proves exact convergence of the
uniformly alive-sampled law to its finite-swarm stationary law:
$$
 \|\lambda_{N,n}^a-\lambda_N^{a,*}\|_{\rm TV}
 \le q_f^{\lfloor n/2\rfloor},\qquad N\ge2,
$$
with $q_f<1$ independent of $N$. The same theorem proves Wasserstein
relaxation for the law of the random alive empirical measure. Its actual
finite-component coupling and two-update Gaussian bridge supply a primitive
nonempty weak-selection interval in
{prf:ref}`cor-slcw-finite-positive-exponents`. The bounded raw reward,
bounded completed force-center profile, positive kinetic noises, capped
velocity and death-disabled hypotheses remain part of that theorem.
For a same-potential unbounded raw reward,
{prf:ref}`thm-slcw-alive-uniform-law` instead controls convergence toward
the stationary population law with an explicit vanishing particle floor.
The quadratic and Rastrigin regimes are verified in
{prf:ref}`cor-slcw-concrete-alive-profile` and
{prf:ref}`cor-slcw-rastrigin-uniform-law`.

At the original nonresonant harmonic timestep,
{prf:ref}`thm-kuhw-active-exact-uniform-law` proves exact
finite-swarm invariant alive-law mixing at weak positive selection
with no particle floor. Its uniform weighted coupling and moment
drift include the actual current-frame global fitness normalizers
and accepted collision components. The complete bounded-reward
profile is {prf:ref}`cor-kuhw-original-step-active-profile`; viscosity
and death remain disabled in this active theorem.

For the conservative harmonic kernel with cloning disabled,
{prf:ref}`cor-ku-count-kinetic-invariant-law` additionally proves
exact finite-swarm invariant-law and alive-law Wasserstein
relaxation with an $N$-independent rate at strictly positive count
viscosity in {prf:ref}`cor-ku-original-step-positive-count`.
For active cloning at the original count viscosity $\nu=0.3$,
{prf:ref}`lem-ku-count-active-harmonic-drift` supplies an explicit
population-uniform fourth-moment drift at weak positive selection.
That moment result does not close the active-law feedback estimate.

The complete-update native-cap certificate
{prf:ref}`thm-rcap-harmonic-whole-update` now gives a stronger
cloning-disabled baseline. Its enlarged strictly positive count-viscosity
interval is {prf:ref}`cor-rcap-positive-count-interval`, with exact
finite-swarm invariant and alive-law Wasserstein relaxation and no
particle floor. At the original harmonic parameters, its diagnostic
endpoint is approximately $0.000260436$; its exact formula, rather than
this decimal, certifies the interval. It still does not reach $0.3$.

For active selection and strictly positive count viscosity,
{prf:ref}`thm-pvb-active-population-convergence` and
{prf:ref}`thm-vupt-uniform-time` now prove actual alive-observation
relaxation toward the conservative population stationary law.
With the explicit $q'<1$, $\varepsilon_N\to0$ and $C_G,C_*$ of
{prf:ref}`cor-vupt-alive-w2`,
$$
 \mathbb E W_{2,G}^2(\widehat\mu_n^N,\pi_{\nu,\theta}),\qquad
 W_{2,G}^2(\mathbb E\widehat\mu_n^N,\pi_{\nu,\theta})
 \le C_G\sqrt{\min\{1,C_*q'^n+\varepsilon_N\}}.
$$
The random empirical-measure law obeys the same estimate. The actual
sampled preparation and both correlated viscous kicks are retained;
entering swarms need no independence or exchangeability. The sufficient
regime includes $F=-x$, $h=0.04$, bounded configured reward
$-\tanh(|x|^2/2)$, explicit positive selection and viscosity intervals,
death disabled, and a uniform initial averaged eighth-moment budget.
This result has a population target and a particle floor; it does not
assert exact dense finite-array invariant mixing or certify $\nu=0.3$.

For the raw same-potential channels $F=-x$, $R=-|x|^2/2$,
{prf:ref}`cor-rqf-active-population` and
{prf:ref}`thm-rqpt-uniform-time` prove population attraction and actual
finite-swarm alive Wasserstein relaxation with an explicit vanishing
uniform-time particle floor, using the actual logistic derivative decay.
The optimal alive empirical-law and alive-sampled bounds are
{prf:ref}`cor-rqpt-alive-w2`, under the raw primitive positive selection
and count-viscosity intervals and initial averaged eighth-moment budget.
For the terminal-box marked population update,
{prf:ref}`thm-kpf-large-box-population-convergence` additionally proves
attraction to a unique stationary revival population law in a primitive
nonempty regime with a fixed sufficiently large box and positive selection
and viscosity. {prf:ref}`cor-kpf-current-alive-relaxation` gives its current
alive TV and Wasserstein relaxation. Mandatory revival, each environment's
alive normalizer and terminal death are retained. This marked population
law is not identified with a finite-swarm QSD or survivor law.

The actual surviving finite-swarm transfer is now proved in
{prf:ref}`thm-spt-uniform-surviving-law` and {prf:ref}`cor-spt-alive-w2`.
For the fixed large-box, positive weak-selection and count-viscosity regime,
its optimal current-alive empirical-law and swarm-first sampled bounds are
$$
 \mathbb E[W_{2,G}^2(\widehat\alpha_n^N,\pi_L^A)\mid\tau_N>n],\qquad
 W_{2,G}^2(\mathbb E[\widehat\alpha_n^N\mid\tau_N>n],\pi_L^A)
 \le\mathcal D_G^2\min\{1,2u_{N,n}/m_f+c_{\rm s}r_N\},
$$
where $u_{N,n}=\min\{1,C_{\rm pop}r_*^{n-1}+\varepsilon_N^{\rm s}\}$,
$r_*<1$ is independent of $N$, and
$\varepsilon_N^{\rm s},r_N\to0$. The proof retains each swarm's own
survival denominator and the actual alive empirical normalization; it
charges survival only over a recent restart window. Arbitrary retained
entering dead coordinates are allowed. Its target is the alive restriction
of the marked population stationary law, with a vanishing particle floor,
and is not identified with an exact finite-swarm QSD.

The same-potential raw quadratic surviving law is completed in
{prf:ref}`cor-rqk-surviving-alive-law`. Its unchanged channels are
$F=-x$, $R=-|x|^2/2$; all population endpoints are computed before the
fixed sufficiently large box is chosen. Only the finite weak comparison
uses the subsequent alive-box reward bounds. It gives the preceding
survival-conditioned optimal alive-law estimate with its own explicit
raw positive selection/viscosity interval and vanishing particle floor.

The true row-normalized kernel also has a completed population and
actual finite surviving-swarm law in
{prf:ref}`thm-rpf-marked-population`,
{prf:ref}`thm-rft-uniform-surviving-law` and
{prf:ref}`cor-rft-alive-wasserstein`. The rate is independent of $N$;
the explicit particle floor vanishes uniformly over all times. Both
alive sampling orders are covered, with their separate normalization
in {prf:ref}`cor-rft-all-slot-alive-sample`. The proof retains the raw
same-potential reward, true empirical row degrees, self exclusion,
uncapped joint OU providers, original component velocities, both kicks
and each own survival denominator. Its nonempty primitive regime is
{prf:ref}`def-rpf-positive-endpoints`: small positive selection and
row viscosity after one fixed sufficiently large box has been chosen.
Its target is the stationary row population alive law, with a particle
floor, rather than an identified finite-swarm QSD.

The default alive coverage and recent-survival bounds are proved in
{prf:ref}`thm-dsa-default-box-alive-floor` and
{prf:ref}`cor-dsa-recent-window-normalization` for both normalizations.
The original count-viscosity frozen root-kernel mixing theorem is
{prf:ref}`thm-rcb-source-box-frozen-gap`; it compares common providers.
The actual harmonic default moment and finite consistency interfaces
are additionally proved in {prf:ref}`thm-rvb-population-burn`,
{prf:ref}`thm-rfk-two-count-principal`,
{prf:ref}`thm-dmc-default-interface` and
{prf:ref}`thm-dev-current-provider-budget`. At $\nu=.3,L=2$ they
retain the native cap, both count graphs, full Gaussian tails,
mandatory revival and current-survival starting reweighting.
Their finite interface uses an explicit nonempty weak positive
fitness interval and vanishing errors uniformly over current time.
The full own-provider spatial and preparation/marking absorption
remains unproved; these moment and consistency results are not
a default convergence rate or a finite-swarm QSD mixing theorem.
The actual nonlinear default block still requires comparison of each
law's own preparation and both own providers.

The original count profile now has complete physical kinetic transport
on arbitrary-position constant-velocity slices, nonconstant pointwise
velocity bands and a signed inward-velocity family, in
{prf:ref}`thm-dsa53-physical-shapes`,
{prf:ref}`cor-csb-optimal-law` and
{prf:ref}`thm-dsg-inward-family`. The last family also has actual
finite kinetic alive empirical-law and both sampled-law estimates,
retaining each own survival normalization, in
{prf:ref}`thm-siat-signed-alive-laws`. These prepared-input classes
are not asserted invariant or obtained from arbitrary active swarms.
The preserved boundary envelopes and exact delayed source/Haar/cap
moment response are {prf:ref}`thm-dsti-preserved-boundary` and
{prf:ref}`thm-dsti-delayed-moments`. The complete full-dead
marked feedback and chronological law response are
{prf:ref}`thm-dlb-default-feedback` and
{prf:ref}`thm-dlb-delayed-response`, whose loose absolute gain does
not certify default mixing. The remaining task is a closed delayed
signed bound through genuine noisy velocity laws, actual preparation
and each own alive normalizer. The local empirical-law regularity
result {prf:ref}`thm-tqp-empirical-local-regularity` only restricts a
one-update Lipschitz proof; it does not refute this delayed target.

The exact general physical signed account is
{prf:ref}`thm-dbl68-ledger`, with its full cap loss and both providers.
The actual conditional cap bound
{prf:ref}`thm-ccl-conditional-coercivity` preserves local
velocity/displacement correlations. The full own-OU graph trace and
inward mixed flux in {prf:ref}`thm-icb-stein-cap` and
{prf:ref}`cor-icb-cross-flux` also remain raw Gaussian expectations
until each actual next-survival division is charged. Their primitive
remainder is proved, but their general signed absorption and delayed
source/Haar comparison remain open.
The entire original recipient-jitter account, including the finite
coincident-environment force square, is
{prf:ref}`thm-fjt73-first-account`. Its matched-status inward-source
sign covers the linear part only and leaves the complete absorption
and alive-law obligations explicit.
The conditional coefficient $159/200$ of
{prf:ref}`thm-gca74-cap-deficit` applies to pre-OU-fixed vectors
with the actual noisy graph retained. The full correlated force
and cross terms in {prf:ref}`cor-gca74-signed-cap-account` and
the weighted own restrictions in
{prf:ref}`prop-gca74-own-restriction` remain necessary; a scalar
conditional deficit alone does not close default law mixing.
The actual first-provider positional margin now passes in
{prf:ref}`cor-ffo77-position`, from the general centered operator
{prf:ref}`thm-ffo77-operator`. The endpoint positional estimate
retains $b\|\Delta P\|_2$, its declared RMS budgets and each
random-array mixed moment. It precedes terminal restriction and
does not close the phase, preparation or surviving alive-law account.
The full source-plan population capped-force consumers in
{prf:ref}`thm-nca76-first-cap-force` and
{prf:ref}`thm-nca76-second-cap-force` retain both own noisy fields
and their complete noncommuting spatial residual. The signed
second-provider cross products and actual preparation, marks
and own normalization remain unabsorbed. Their population moments
are not substituted into random finite empirical products.
The actual mean-cap matrix sector is proved in
{prf:ref}`thm-mcm79-lower`, with its full-jitter weighted
population Gram sector in {prf:ref}`thm-mcm79-source-sector`.
The signed first response is now absorbed by
{prf:ref}`thm-sfc80-first-consumer`, for its auxiliary
$(R,DZ_b)$ differential under the stated prepared budgets.
The exact complete second response remains explicitly in
{prf:ref}`cor-sfc80-second-interface`; that auxiliary margin
has not been integrated into a transport map or iterated.
The mixed source-weighted extinction and empirical-position
charges are bounded in {prf:ref}`thm-wsd81-surviving-deficit`.
Its own-survivor paired cap deficit applies in a declared
pathwise low-speed class and explicit sufficiently-large-$N$
regime. It does not compare two separately surviving alive
readouts or establish class invariance.

The conservative results use all-alive normalization; the large-box
surviving result uses the separate killed proof and its own normalized
alive law. The original $L=2$, $\nu=0.3$ reference remains outside the
population-uniform survivor-law certificate.
The whole-horizon normalization step is proved in
{prf:ref}`thm-ku-finite-future-survival-ratio` and
{prf:ref}`cor-ku-future-conditioned-alive-law`, conditional on the
stated actual full-state block comparison. It retains each swarm's
own survival denominator.
:::

:::{prf:definition} Single-swarm moment observable
:label: def-tv-lyapunov

Let $A=\mathcal A(S)$ and $k=|A|$. Define the alive barycentres and variances by

$$
\bar x_A=\frac1k\sum_{i\in A}x_i,\qquad
\mu_v=\frac1k\sum_{i\in A}v_i,
$$

$$
V_x=\frac1N\sum_{i\in A}\|x_i-\bar x_A\|^2,\qquad
V_v=\frac1N\sum_{i\in A}\|v_i-\mu_v\|^2,\qquad
E_v=\|\mu_v\|^2.
$$

For a nonnegative barrier $\phi$, put
$W_b=N^{-1}\sum_{i\in A}\phi(x_i)$, whenever finite. Set

$$
\mathbf V=(V_x,V_v,E_v,W_b)^\top,\qquad
V_{\rm TV}=c_V(V_x+V_v)+c_\mu E_v+c_BW_b,
$$

where $c_V,c_\mu,c_B>0$. The subscript records the observable historically used
in the total variation argument; a moment inequality for $V_{\rm TV}$ has a
separate logical role from a total variation contraction.

The alive barycentre uses $1/k$ even though the variances use $1/N$. For $k=N$
these conventions coincide. Bounds for a two-swarm sum, a two-swarm average, or
a different alive normalization must first be converted to this convention.
:::

:::{prf:remark} Analytic inputs and their domains
:label: rem-prerequisite-drifts

The component analysis is in {doc}`03_cloning` and
{doc}`05_kinetic_contraction`. The assembly below uses their inequalities on the
states where their hypotheses hold, with constants uniform over those states.
In particular, a contraction depending on occupancy of an interior reward set
requires a lower bound on that occupancy. Existence of the set alone supplies
no such lower bound.

The Euclidean kinetic bounds below use the specified velocity cap and Euclidean
position increments. The latent algorithm additionally requires the metric,
force, diffusion, and collision estimates in
{doc}`../1_the_algorithm/02_fractal_gas_latent`. An instantaneous norm
comparison between two metrics does not identify their drift inequalities.

For an unbounded position space, centred variance does not control the position
of the swarm barycentre. A tightness argument therefore also needs a confining
position observable with proved drift. On a domain with an absorbing boundary,
$W_b$ must have finite transition expectations. For example, a reciprocal
boundary-distance singularity need not be integrable against a Gaussian
position density; the barrier analysis in {doc}`03_cloning` must be applied
before inserting a finite $C_b$.
:::

(sec-convergence-components)=
## 2. Component Estimates That Can Be Composed

:::{div} feynman-prose
A rate written beside a component is a promise about a particular transition.
For example, friction gives an exact exponential estimate for a continuous
Langevin evolution. A split numerical step needs its own estimate. We first
record bounds that follow directly from capped motion, then show how sharper
kinetic calculations enter when their hypotheses are verified.
:::

:::{prf:proposition} Component drift inputs
:label: prop-complete-drift-summary

Suppose cloning and kinetics have nonnegative, state-independent comparison
matrices and source vectors satisfying

$$
P_C\mathbf V\le A_C\mathbf V+\mathbf b_C,\qquad
P_K\mathbf V\le A_K\mathbf V+\mathbf b_K.
$$

The inequalities are componentwise and apply to the same observables, including
the cemetery extension.
A kinetic coefficient imported from a continuous generator additionally
requires a proved observable weak-error certificate for a generator-consistent
transition family, as specified in
{prf:ref}`thm-discretization` and
{prf:ref}`rem-kinetic-generator-transfer-scope`.
The canonical fixed radial velocity cap is not near the identity as the timestep
tends to zero and does not inherit an uncapped Langevin rate by that argument.
Its coefficients must come from direct native finite-step estimates, such as
{prf:ref}`thm-kinetic-exact-baoab-cap-coupling` for unit-quadratic physical
coordinates. A location or centered-transport coefficient using
{prf:ref}`lem-location-error-drift-kinetic` also requires its actual macroforce
closure, positive matrix certificate and residual/status bounds; confinement
and a force Lipschitz constant alone do not supply those inputs.
 If cloning has a separate intermediate state space,
the observables must be defined there and the kinetic estimate must cover
every intermediate state to which cloning assigns mass. This includes proposed
jittered positions under the specified status-update convention. The diagonal
input frequently used in the component
chapters has the form

$$
A_C=\operatorname{diag}(r_{C,x},1,r_{C,\mu},r_{C,b}),\qquad
A_K=\operatorname{diag}(r_{K,x},r_{K,v},r_{K,\mu},r_{K,b}).
$$

Here $r_{C,x}=1-\kappa_x$ only when the corresponding affine cloning
estimate, including its source term, holds. In particular, (3.AC3) of
{prf:ref}`thm-complete-cloning-drift` obtains such a coefficient from a
uniform positional reset bound, with the full $C_x$ offset. Its chosen
$\kappa_x$ is **not** the $N$-uniform Keystone pressure coefficient and
does not certify attraction to a stationary phase. A bounded expansion
uses coefficient one; a uniform bound uses coefficient zero. Every source
vector must include the jitter, collision, and discretization terms
appropriate to that estimate. The rate-sensitive Keystone calculation
is (SCK.3)--(SCK.6) of {prf:ref}`thm-slc-signed-complete-update`.

The complete cloning statement {prf:ref}`thm-complete-cloning-drift` supplies
one collection of such inputs. Its boundary component uses the favorable
companion and integrable-barrier estimates
{prf:ref}`lem-boundary-enhanced-cloning` and
{prf:ref}`lem-barrier-reduction-cloning`, with revival contributions tracked
in {prf:ref}`thm-complete-boundary-drift`. The following direct kinetic bounds
supply another, without an assumed equilibrium variance.
:::

:::{prf:lemma} Capped displacement and removal of walkers
:label: lem-convergence-capped-displacement

Suppose a kinetic step adds no alive indices, its surviving set is $B\subset A$,
and every surviving position satisfies $x_i'=x_i+\Delta_i$ with
$\|\Delta_i\|\le h v_{\max}$. If its output velocities satisfy
$\|v_i'\|\le v_{\max}$, then, for every $\theta>0$,

$$
V_x(S')\le(1+\theta)V_x(S)
 +(1+\theta^{-1})h^2v_{\max}^2,
\qquad
V_v(S')\le v_{\max}^2,\qquad E_v(S')\le v_{\max}^2.
$$

These inequalities also hold after absorption, with the observables set to
zero. They apply to two capped half-position updates whose durations sum to
$h$. For a latent metric, the displacement bound must first be established in
the norm used to define $V_x$.
:::

:::{prf:proof}
For any nonempty index set $I$,

$$
\sum_{i\in I}\|z_i-\bar z_I\|^2
 =\min_a\sum_{i\in I}\|z_i-a\|^2.
$$

Use $a=\bar x_A$ for the final set $B$. The inequality
$\|u+w\|^2\le(1+\theta)\|u\|^2+(1+\theta^{-1})\|w\|^2$ gives

$$
NV_x(S')\le(1+\theta)\sum_{i\in B}\|x_i-\bar x_A\|^2
 +(1+\theta^{-1})|B|h^2v_{\max}^2.
$$

Enlarge the first sum to $A$ and use $|B|\le N$. For velocities, the minimum
over centres is at most the value at zero, so
$NV_v(S')\le\sum_{i\in B}\|v_i'\|^2\le Nv_{\max}^2$.
The triangle inequality bounds the norm of the alive velocity mean by
$v_{\max}$. Empty and cemetery outputs have zero observable by convention.
:::

:::{prf:corollary} Kinetic position noise and a displacement moment bound
:label: cor-convergence-displacement-moment

Suppose the kinetic step has no new alive indices and its proposed position
increments satisfy

$$
\frac1N\sum_{i\in A}\mathbb E_S\|\Delta_i\|^2\le D_h^2.
$$

Then

$$
P_KV_x\le(1+\theta)V_x+(1+\theta^{-1})D_h^2.
$$

For the position-noisy Euler variant EG-kin$^+$ specified in
{doc}`02_euclidean_gas`, write its kinetic position-noise scale as
$\sigma_{\rm pos}$ to distinguish it from cloning jitter. Its update
$\Delta_i=hv_i^++\sqrt h\sigma_{\rm pos}\xi_i^x$, with capped $v_i^+$ and
independent centred $\xi_i^x\sim N(0,I_d)$, gives

$$
D_h^2=h^2v_{\max}^2+hd\sigma_{\rm pos}^2.
$$

If both BAOAB half-position velocities are capped and there is no separate
position Gaussian, $D_h^2=h^2v_{\max}^2$. The standard native BAOAB cap is
terminal-only and does not supply this pathwise premise. Instead freeze the
complete prepared first-kick velocities $v_{1,i}$ before independent OU and
position Gaussians. Put $a_O=e^{-\gamma h}$,
$c_h=h(1+a_O)/2$, let $\sigma_v$ denote the actual OU innovation amplitude,
and let $\sigma_x$ denote the actual final position Gaussian amplitude. Then

$$
\Delta_i=c_hv_{1,i}+\frac h2\sigma_v\eta_i+\sigma_x\zeta_i,
\qquad
\tau_h^2=(h\sigma_v/2)^2+\sigma_x^2,
$$

with independent standard $d$-dimensional Gaussians $\eta_i,\zeta_i$. For the
prepared sigma-field $\mathscr G_1$, before these innovations,

$$
\widehat D_h^2(\mathscr G_1)
 =\frac{c_h^2}{N}\sum_{i\in A}\|v_{1,i}\|^2
  +\frac{|A|}{N}d\tau_h^2,
\qquad
D_h^2(S)=\mathbb E_S\widehat D_h^2(\mathscr G_1).
$$

The displayed kinetic inequality holds with this state-dependent budget.
For the all-alive prepared population, $|A|=N$. Terminal capping does not
bound $v_1$. The budget retains the actual first force, viscosity, OU noise
and full-support final position noise; predictable innovation shifts or a
different kinetic specification require their own displacement expression.

A sharper centred budget also applies:

$$
\widehat D_{h,\mathrm{cent}}^2(\mathscr G_1)
 =c_h^2 V_v(v_1;A)
  +\frac{(|A|-1)_+}{N}d\tau_h^2,
\qquad
V_v(v_1;A)=\frac1N\sum_{i\in A}\|v_{1,i}-\bar v_{1,A}\|^2.
$$

One may replace $D_h^2(S)$ by
$\mathbb E_S\widehat D_{h,\mathrm{cent}}^2$ in the same kinetic inequality.
This removes common displacement without assigning identities to walkers.
:::

:::{prf:proof}
Apply the minimum-over-centres calculation of the preceding lemma to the
surviving subset, then enlarge the nonnegative squared-increment sum to all
input alive indices before taking expectations. For EG-kin$^+$, independence
and centring remove the mixed term, while
$\mathbb E\|\xi_i^x\|^2=d$. The cap bounds the remaining deterministic
velocity contribution.
For standard native BAOAB, the two half-position updates give the displayed
increment exactly. Conditional centring and independence remove its mixed
terms, and each Gaussian squared norm has expectation $d$. Integrate first
over these innovations and then over the prepared first kick. For the centred
refinement, choose the centre $\bar x_A+\bar\Delta_A$ for the surviving
subset and enlarge its nonnegative sum to all proposed rows in $A$.
Independent identical Gaussian rows have expected centred squared sum
$d(|A|-1)_+\tau_h^2$; the deterministic part contributes
$c_h^2\sum_{i\in A}\|v_{1,i}-\bar v_{1,A}\|^2$. Empty and absorbed
outputs have zero observable. No terminal cap is substituted for a transient
first-kick velocity.
:::

:::{prf:proposition} Positional drift from the cloning estimate and the cap
:label: prop-position-rate-explicit

Assume $P_CV_x\le r_{C,x}V_x+b_{C,x}$ with $0\le r_{C,x}<1$, and the kinetic
step satisfies {prf:ref}`lem-convergence-capped-displacement`. Then

$$
QV_x\le r_xV_x+b_x,
$$

$$
r_x=(1+\theta)r_{C,x},\qquad
b_x=(1+\theta)b_{C,x}+(1+\theta^{-1})h^2v_{\max}^2.
$$

For $r_{C,x}>0$, choose $0<\theta<(1-r_{C,x})/r_{C,x}$ to obtain $r_x<1$.
For $r_{C,x}=0$, every $\theta>0$ works. The positional decrement per full step
is $\kappa_x^{\rm step}=1-r_x$. Under
{prf:ref}`cor-convergence-displacement-moment`, replace
$h^2v_{\max}^2$ in $b_x$ by $D_h^2$.

For the standard native state-dependent displacement budget, instead assume
the additional same-kernel estimate
$P_CD_h^2\le a_DV_x+b_D$, with $a_D,b_D\ge0$. The coefficients then are

$$
r_x=(1+\theta)r_{C,x}+(1+\theta^{-1})a_D,
\qquad
b_x=(1+\theta)b_{C,x}+(1+\theta^{-1})b_D.
$$

Contraction requires this full $r_x<1$. A recorded displacement budget alone
does not supply the additional affine closure, and a transient first-kick
force cannot be bounded using the terminal velocity cap. The centred budget
may be used under its corresponding affine closure.
:::

:::{prf:proof}
Apply $P_C$ to the kinetic inequality. Since $P_C1\le1$, its constant term
increases by at most the displayed amount. Substitute the cloning bound and
collect the coefficient and source. The restriction on $\theta$ is exactly
$(1+\theta)r_{C,x}<1$.
For a state-dependent budget apply $P_C$ to its complete kinetic inequality
and substitute both the cloning drift and $P_CD_h^2\le a_DV_x+b_D$.
Collecting their coefficients gives the second displayed pair; no term is
discarded before testing $r_x<1$.
:::

:::{prf:proposition} Kinetic velocity rates and their time convention
:label: prop-velocity-rate-explicit

For a fixed alive population of $N$ walkers evolving by the continuous
Euclidean Langevin equation with friction $\gamma$, forces bounded by
$F_{\max}$, independent Brownian increments, and diffusion matrices bounded
by $\sigma_{\max}$, the Itô estimates of {doc}`05_kinetic_contraction` take the
form

$$
\frac{d}{dt}\mathbb EV_v\le-a_v\mathbb EV_v+B_v,
\qquad a_v=2\gamma-\epsilon>0,
\qquad B_v=F_{\max}^2/\epsilon+d\sigma_{\max}^2,
$$

$$
\frac{d}{dt}\mathbb EE_v\le-\gamma\mathbb EE_v+B_\mu,
\qquad B_\mu=F_{\max}^2/\gamma+d\sigma_{\max}^2/N.
$$

For the exact continuous transition over time $h$, these imply

$$
P_hV_v\le e^{-a_vh}V_v+\frac{B_v}{a_v}(1-e^{-a_vh}),\qquad
P_hE_v\le e^{-\gamma h}E_v+\frac{B_\mu}{\gamma}(1-e^{-\gamma h}).
$$

A BAOAB or Boris-BAOAB estimate can use these coefficients only after its split
step has been bounded. If the split step instead has an error estimate
$P_KV_v\le(e^{-a_vh}+\eta_v(h))V_v+b_v(h)$, its decrement is
$1-e^{-a_vh}-\eta_v(h)$ when positive. The capped output bound
$P_KV_v,P_KE_v\le v_{\max}^2$ remains available independently of this sharper
friction estimate.
:::

:::{prf:proof}
For the continuous equations, centring the velocity force term gives
$2N^{-1}\sum_i(v_i-\mu_v)\cdot F_i$; the part involving the mean force vanishes.
Young's inequality bounds it by
$\epsilon V_v+F_{\max}^2/\epsilon$. The centred quadratic variation is at most
$d\sigma_{\max}^2$. For the mean velocity, Young's inequality gives
$2\mu_v\cdot\bar F\le\gamma\|\mu_v\|^2+F_{\max}^2/\gamma$,
and independence bounds its quadratic variation by $d\sigma_{\max}^2/N$.
These are the two stated differential inequalities.

Multiplying an inequality $y'\le-ay+B$ by $e^{at}$ and integrating gives
$y(h)\le e^{-ah}y(0)+B(1-e^{-ah})/a$. In particular,
$1-e^{-ah}\le ah$; replacing $e^{-ah}$ by the smaller number $1-ah$ is not an
upper bound. Changes of alive set, conditioning on survival, interactions in
the diffusion noise, and split-step remainders must be treated in the
corresponding transition estimate before using this calculation.
:::

:::{prf:proposition} Boundary coefficients under sequential updates
:label: prop-boundary-rate-explicit

For a nonnegative, transition-integrable barrier observable, suppose

$$
P_CW_b\le r_{C,b}W_b+b_{C,b},\qquad
P_KW_b\le r_{K,b}W_b+b_{K,b},
\qquad r_{C,b},r_{K,b}\ge0.
$$

Then

$$
QW_b\le r_{K,b}r_{C,b}W_b+r_{K,b}b_{C,b}+b_{K,b}.
$$

In particular, when $r_{C,b}=1-\kappa_b$ and
$r_{K,b}=1-\kappa_{\rm pot}h$ are valid nonnegative coefficients, the full-step
decrement is

$$
\kappa_b^{\rm step}
=\kappa_b+\kappa_{\rm pot}h-\kappa_b\kappa_{\rm pot}h.
$$

A uniform expected barrier bound can instead be entered as $r_{K,b}=0$.
The choice of a barrier observable does not modify the algorithm's force,
reward, or absorbing boundary.
:::

:::{prf:proof}
Apply $P_C$ to the kinetic inequality, use $P_C1\le1$, and substitute its
bound on $W_b$. Expanding $1-(1-\kappa_b)(1-\kappa_{\rm pot}h)$ yields the
decrement.
:::

:::{prf:lemma} Final-Gaussian logarithmic barrier with explicit dimension
:label: lem-convergence-dimension-log-barrier

Let $N,d\ge1$, $L>0$, $h>0$, $\sigma_{\rm pos}>0$, and
$s=\sigma_{\rm pos}\sqrt h$. Condition on the complete preparation
$\mathscr G$ before the final independent positional Gaussian of the actual
split update. Suppose

$$
X_{ij}=m_{ij}+sZ_{ij},\qquad 1\le i\le N,\quad 1\le j\le d,
$$

where, conditional on $\mathscr G$, the $Z_{ij}$ in each row are independent
standard Gaussians and the prepared means $m_{ij}\in\mathbb R$ are arbitrary.
Define $D_L=(-L,L)^d$, $\phi_1(x)=-\log(1-(x/L)^2)$ on $(-L,L)$, and

$$
\widetilde W_b=\frac1N\sum_{i=1}^N
 \mathbf1_{D_L}(X_i)\sum_{j=1}^d\phi_1(X_{ij}).
$$

The actual alive barrier $W_b'$ is bounded by $\widetilde W_b$ if the final
alive set is contained in $\{i:X_i\in D_L\}$; assigning zero to a swarm
cemetery state preserves this inequality. Put

$$
p_s(m)=\int_{-L}^L\frac{e^{-(x-m)^2/(2s^2)}}{\sqrt{2\pi}s}\,dx,
\qquad
g_s(m)=\int_{-L}^L\phi_1(x)
 \frac{e^{-(x-m)^2/(2s^2)}}{\sqrt{2\pi}s}\,dx,
\qquad
a=\frac{2L}{\sqrt{2\pi}s},\quad q=\min\{1,a\}.
$$

Then the tensor identity and dimension-explicit bounds are

$$
\mathbb E[\widetilde W_b\mid\mathscr G]
=\frac1N\sum_{i=1}^N\sum_{j=1}^d
 g_s(m_{ij})\prod_{\ell\ne j}p_s(m_{i\ell}),
\qquad
\mathbb E[W_b'\mid\mathscr G]
\le B_d:=d\Lambda(a)q^{d-1}
\le\frac{4dL(1-\log2)}{\sqrt{2\pi}s},
$$

where

$$
\Lambda(a)=
\begin{cases}
2a(1-\log2),&0<a\le1,\\
\log a-\log(2-1/a)+2+2a\log(1-1/(2a)),&a>1.
\end{cases}
$$

Thus $QW_b\le B_d$ for either the native zero-alive killed kernel or the
additional fewer-than-two cemetery convention under these Gaussian and box
hypotheses. The constant is independent of $N$; for $a\ge1$ its dependence
on $d$ is linear and $\Lambda(a)=\log(a/2)+1+O(a^{-1})$ as $a\to\infty$.
The identity concerns the candidate box observable; additional killing can
reduce its expectation. For a survival-conditioned law one must instead use

$$
\mathbb E_\mu[W_b(S_n)\mid\tau>n]
=\frac{\mu Q^nW_b}{\mu Q^n1}
$$

when the denominator is positive. A bound on the unconditioned numerator
is not a bound with the same constant after conditioning.
:::

:::{prf:proof}
Nonnegativity permits Tonelli throughout. Conditional coordinate
independence factors the integral of each summand into $g_s$ in its marked
coordinate and $p_s$ in the others, proving the identity. No independence
between swarm rows is required. The Gaussian density is at most
$D=(\sqrt{2\pi}s)^{-1}$, so $p_s(m)\le q$ for every real mean. Direct
integration gives

$$
\int_{-L}^L\phi_1(x)\,dx=4L(1-\log2).
$$

Consequently $g_s(m)\le4LD(1-\log2)$, and retaining the other-coordinate
probabilities gives $2da(1-\log2)q^{d-1}$. Dropping only those probabilities
gives the displayed linear bound. Bounding the whole $d$-dimensional
Gaussian by $D^d$ and integrating the entire cube would introduce the
unnecessary factor $a^{d-1}$ when $a>1$.

For the sharper coordinate bound, the layer-cake representation yields

$$
g_s(m)=\int_0^\infty
 \mathbb P\bigl(|X|<L,\ \phi_1(X)>t\mid\mathscr G\bigr)\,dt
\le\int_0^\infty
 \min\{1,a(1-\sqrt{1-e^{-t}})\}\,dt.
$$

Indeed the level set consists of two intervals adjacent to the box
endpoints, with total length $2L(1-\sqrt{1-e^{-t}})$. For $a\le1$ the minimum
never changes branch. Substituting $u=\sqrt{1-e^{-t}}$ gives
$2a\int_0^1u/(1+u)\,du=2a(1-\log2)$. For $a>1$ the branch changes at

$$
t_0=\log a-\log(2-1/a),\qquad u_0=1-1/a.
$$

The integral is
$t_0+2a\int_{u_0}^1u/(1+u)\,du
=t_0+2+2a\log(1-1/(2a))=\Lambda(a)$.
Inserting this coordinate bound and $p_s\le q$ into the tensor identity
proves $B_d$. The asymptotic expression follows by expanding
$\log(1-1/(2a))$ and $\log(2-1/a)$. Averaging $N$ identical uniform bounds
cancels $N$ exactly. Conditional means are unrestricted and the Gaussian
is integrated over its full law before applying the box indicator; no
compact-support replacement occurs. Additional swarm killing preserves
the nonnegative upper bound. The survival identity follows from the
killed semigroup definition and requires its own nonzero denominator.
:::

:::{prf:lemma} State-aware analytic barrier envelopes and independent-seed precision
:label: lem-convergence-state-log-barrier

Under {prf:ref}`lem-convergence-dimension-log-barrier`, write
$D=(\sqrt{2\pi}s)^{-1}$ and

$$
q_m=\min\{1,2LD e^{-(|m|-L)_+^2/(2s^2)}\},\qquad
F(r)=L\left[2u-(1+u)\log(1+u)+(1-u)\log(1-u)\right],\quad u=r/L,
$$

with $0\log0=0$. For $|m|<r<L$, set

$$
T_1(r)=4L(1-\log2)-2F(r),\qquad
T_2(r)=2L(1-r/L)\left[\log^2(1-r/L)-2\log(1-r/L)+2\right],
$$

and $D_{m,r}=D e^{-(r-|m|)^2/(2s^2)}$.
Valid first- and second-moment coordinate upper bounds are

$$
\begin{aligned}
G(m)&=\min\left\{\Lambda(a),\ 4LD(1-\log2)e^{-(|m|-L)_+^2/(2s^2)},
\ \inf_{|m|<r<L}\bigl[\phi_1(r)q_m+D_{m,r}T_1(r)\bigr]\right\},\\
H(m)&=\min\left\{4LD e^{-(|m|-L)_+^2/(2s^2)},
\ \inf_{|m|<r<L}\bigl[\phi_1(r)^2q_m+D_{m,r}T_2(r)\bigr]\right\}.
\end{aligned}
$$

An infimum over an empty set is $+\infty$. Any finite set of admissible
radii gives a valid, possibly larger bound; no numerical quadrature or
unproved supremum search is needed. For a prepared row, its candidate
barrier $Y_i=\mathbf1_{D_L}(X_i)\sum_j\phi_1(X_{ij})$ satisfies

$$
\begin{aligned}
\mathbb E[Y_i\mid\mathscr G]
&\le U_i=\sum_jG(m_{ij})\prod_{\ell\ne j}q_{m_{i\ell}},\\
\mathbb E[Y_i^2\mid\mathscr G]
&\le R_i=\sum_jH(m_{ij})\prod_{\ell\ne j}q_{m_{i\ell}}
 +2\sum_{j<k}G(m_{ij})G(m_{ik})
   \prod_{\ell\notin\{j,k\}}q_{m_{i\ell}}.
\end{aligned}
$$

Therefore $\mathbb E[W_b'\mid\mathscr G]\le N^{-1}\sum_iU_i$ and
$\mathbb E[(W_b')^2\mid\mathscr G]\le N^{-1}\sum_iR_i$.
For $M$ independently seeded trajectories with their own preparations,
including absorbed zero observables, let $U^{(r)},R^{(r)}$ denote these
swarm bounds. For any $0<\delta<1$, with conditional probability at least
$1-\delta$,

$$
\frac1M\sum_{r=1}^M W_b'^{(r)}
\le \frac1M\sum_{r=1}^M U^{(r)}
 +\frac1M\sqrt{\frac{1-\delta}{\delta}\sum_{r=1}^M R^{(r)}}.
$$

The independent uncertainty units are trajectories, not walkers or serial
updates. Dividing the empirical unconditional average by its own empirical
survival fraction gives the corresponding observed survivor average only
when that fraction is positive.
:::

:::{prf:proof}
The maximum Gaussian density on $[-L,L]$ is
$D e^{-(|m|-L)_+^2/(2s^2)}$, giving the stated $q_m$ and the first alternatives
for $G,H$. The derivative of $F$ is $\phi_1$ and
$F(L)=2L(1-\log2)$, so $T_1$ is the exact barrier integral on
$\{r<|x|<L\}$. On $|x|\le r$, $\phi_1(x)\le\phi_1(r)$ and its probability
is at most $q_m$. On $r<|x|<L$, the Gaussian density is at most $D_{m,r}$.
Splitting the first-moment integral proves the remaining $G$ bounds.
For the second moment, use
$\phi_1(x)\le-\log(1-|x|/L)$ and integrate its square on the two tails:

$$
2L\int_0^{1-r/L}\log^2 v\,dv=T_2(r).
$$

At $r=0$ this integral is $4L$. The same split proves the $H$ bounds.
Expanding $Y_i^2$ and factoring marked-coordinate moments and unmarked
survival probabilities proves the row formulas. Jensen's inequality gives
$(N^{-1}\sum_iY_i)^2\le N^{-1}\sum_iY_i^2$, so correlations introduced by
cloning do not insert any factor growing with $N$.

Conditional on all preparations, independent trajectory seeds and their
independent current Gaussian innovations give independent observables.
Their average has variance at most $M^{-2}\sum_rR^{(r)}$ and mean at most
$M^{-1}\sum_rU^{(r)}$. Cantelli's one-sided inequality with the displayed
allowance gives failure probability at most $\delta$. Preparations of
already absorbed trajectories contribute zero. A union bound can allocate
a declared family budget over times, initial distributions and cemetery
conventions; serial times do not create additional independent units.
The survivor-average identity is purely its own normalization and does not
identify a quasi-stationary law.
:::

:::{prf:lemma} Gaussian regional landing and weighted tail envelopes
:label: lem-convergence-gaussian-regional-tail

Condition on a preparation for which the actual one-particle positional
update is $X=m+AZ$, with $Z\sim\mathcal N(0,I_d)$, $d\ge1$,
$\sigma_-^2I_d\preceq AA^T\preceq\sigma_+^2I_d$ and
$0<\sigma_-\le\sigma_+$. Suppose the mean satisfies the proved coordinate
enclosure $m_j\in[a_j,b_j]$. For a target analysis box
$B=\prod_j[\ell_j,u_j]$ with $\ell_j<u_j$, define

$$
D_j=\max\{|\ell_j-b_j|,|u_j-a_j|\},\qquad
L_B=\frac{\prod_j(u_j-\ell_j)}{(\sqrt{2\pi}\sigma_+)^d}
      \exp\!\left[-\frac{\sum_jD_j^2}{2\sigma_-^2}\right].
$$

Then $\mathbb P(X\in B\mid\mathscr G)\ge L_B$. When all mean intervals
are contained in the target intervals, the additional escape upper bound is

$$
U_B=\min\!\left\{1,\sum_{j=1}^d
 \left[e^{-(a_j-\ell_j)^2/(2\sigma_+^2)}
       +e^{-(u_j-b_j)^2/(2\sigma_+^2)}\right]\right\}.
$$

Consequently the landing lower bound is $\max\{L_B,1-U_B\}$ and the
escape upper bound is $\min\{1-L_B,U_B\}$; use $U_B=1$ when containment
fails. These are one-particle transition bounds, not coefficients of a
coupled discrepancy contraction.

More generally, suppose only $\|m\|\le c$ and $\|A\|_{\rm op}\le\sigma$,
with $c\ge0$ and $\sigma>0$. For $R>0$ set
$u=((R-c)_+/\sigma)^2$ and, for integers $k\ge0$, define

$$
T_k(d,u)=\left[\prod_{j=0}^{k-1}(d+2j)\right]
\begin{cases}
1,&u\le d+2k,\\
\exp\!\left[\frac{d+2k-u+(d+2k)\log(u/(d+2k))}{2}\right],
 &u>d+2k.
\end{cases}
$$

The empty product is one. For $q>0$, $k=\lceil q\rceil$ and
$T_q=T_k^{q/k}T_0^{1-q/k}$; for $q=k$ take $T_q=T_k$ directly. For
$r>0$,

$$
\mathbb P(\|X\|>R\mid\mathscr G)\le T_0(d,u),\qquad
\mathbb E[\|X\|^r\mathbf1_{\{\|X\|>R\}}\mid\mathscr G]
 \le 2^{(r-1)_+}\{c^rT_0(d,u)+\sigma^rT_{r/2}(d,u)\}.
$$

For $r=0$ the weighted bound is $T_0$. All noise remains Gaussian on its
unbounded support; the analysis boxes and cutoffs do not modify the chain.
For the native isotropic BAOAB positional step, conditional on the
post-cloning $(x,v)$, the force-dependent center and covariance are

$$
b=\frac h2(1+e^{-\gamma h}),\qquad
m=x+bv-\frac{hb}{2}\nabla V(x),\qquad
AA^T=\left[\frac{h^2}{4}\frac{\tau^2(1-e^{-2\gamma h})}{2\gamma}
                       +h\sigma_{\rm pos}^2\right]I_d,
$$

when the O-noise is isotropic with thermostat amplitude $\tau$. For a
frozen O-geometry matrix $\Lambda$ the first covariance summand is instead
multiplied by $\Lambda\Lambda^T$ and its spectral bounds must be supplied.
The last force kick and radial velocity cap do not change the position.
:::

:::{prf:proof}
The Gaussian density determinant is at most $\sigma_+^{2d}$ and its
precision matrix is at most $\sigma_-^{-2}I_d$. Every point of $B$ is at
squared distance at most $\sum_jD_j^2$ from every admitted mean. Integrating
this uniform density lower bound over $B$ gives $L_B$. Each coordinate
marginal has variance at most $\sigma_+^2$. The scalar Gaussian Chernoff
inequality and a union bound over the two faces in each coordinate give
$U_B$, without requiring coordinate independence.

Put $Y=\|Z\|^2$. The event $\|X\|>R$ implies $Y>u$ by the triangle
inequality and the operator-norm bound. For $0\le\lambda<1/2$ the Gaussian
integral gives

$$
\mathbb E[Y^ke^{\lambda Y}]
 =\left[\prod_{j=0}^{k-1}(d+2j)\right](1-2\lambda)^{-d/2-k}.
$$

Multiplication by $e^{-\lambda u}$ bounds the truncated moment. Its
minimum over $\lambda\ge0$ is attained at zero when $u\le d+2k$, and at
$\lambda=(1-(d+2k)/u)/2$ otherwise, giving $T_k$. Hölder's inequality on
the tail event gives the displayed fractional moment. Finally
$(c+\sigma\sqrt Y)^r\le2^{(r-1)_+}(c^r+\sigma^rY^{r/2})$.
The BAOAB formula follows by substituting its first B, first A, O and
second A updates; the independently added final position Gaussian
contributes $h\sigma_{\rm pos}^2I_d$.
:::

:::{prf:lemma} Moment tails for permutation-invariant coupled costs
:label: lem-convergence-moment-tail-transfer

Let a coupling $(X,Y)$ have proved marginal moment bounds
$\mathbb E\|X\|^p\le M_x$ and $\mathbb E\|Y\|^p\le M_y$, with
$p>r\ge0$ and $R>0$. The law may be a uniformly sampled member of an
unlabelled permutation-invariant swarm coupling, provided these moments
use its specified normalization. Each marginal satisfies

$$
\mathbb P(\|X\|>R)\le\min\{1,M_x/R^p\},\qquad
\mathbb E[\|X\|^r\mathbf1_{\{\|X\|>R\}}]\le M_x/R^{p-r}.
$$

Suppose the pair cost is at most
$c_0+c_1(\|X\|^r+\|Y\|^r)$, $c_0,c_1\ge0$, and let
$H=\{\max(\|X\|,\|Y\|)>R\}$. For $r>0$ its tail contribution is at most

$$
c_0\min\{1,(M_x+M_y)/R^p\}
+\frac{c_1}{R^{p-r}}
 \left[M_x+M_y+M_x^{r/p}M_y^{1-r/p}+M_y^{r/p}M_x^{1-r/p}\right].
$$

For $r=0$ the bound is
$(c_0+2c_1)\min\{1,(M_x+M_y)/R^p\}$.
These estimates require proved moment hypotheses for the precise laws
being compared. In particular, an $N^{-1}$-normalized alive moment and
a survival-conditioned uniform-alive marginal moment have different
denominators; one cannot replace the latter by the former.
:::

:::{prf:proof}
On $\{\|X\|>R\}$, $1\le\|X\|^p/R^p$ and
$\|X\|^r\le\|X\|^p/R^{p-r}$, proving the marginal estimates. Split
$\|X\|^r\mathbf1_H$ into its own tail and the event $\|Y\|>R$.
Hölder bounds the cross term by
$M_x^{r/p}\mathbb P(\|Y\|>R)^{1-r/p}
\le M_x^{r/p}M_y^{1-r/p}/R^{p-r}$.
Repeat with $X,Y$ interchanged and use the union bound for the constant
cost. This uses no independence, no persistent walker labels and no
factor of swarm size. The $r=0$ case follows directly from the union bound.
:::

:::{prf:proposition} Wasserstein control requires a coupling observable
:label: prop-wasserstein-rate-explicit

Let $\widehat P$ be a coupling of two conservative full-step swarm kernels.
Suppose a nonnegative observable $D(S,\widetilde S)$ dominates
$W_2^2(\mu_S,\mu_{\widetilde S})$ for the empirical measures under consideration
and satisfies

$$
\widehat P D\le r_DD+b_D,\qquad 0\le r_D<1.
$$

Then

$$
\mathbb E W_2^2(\mu_{S_n},\mu_{\widetilde S_n})
\le r_D^nD(S_0,\widetilde S_0)
 +\frac{b_D(1-r_D^n)}{1-r_D}.
$$

The centred variance identities in {doc}`04_wasserstein_contraction` provide
parts of such an observable. Full phase-space contraction additionally needs
control of the barycentre difference, velocity difference, and their coupling
under the same transition. When $b_D>0$, the displayed result is control to a
nonzero error level. For killed swarms, the coupling must also specify survival
conditioning; a conservative coupling inequality cannot be used unchanged.
:::

:::{prf:proof}
Iterate the affine inequality by conditional expectation and use the domination
of $W_2^2$ by $D$. The geometric sum is
$(1-r_D^n)/(1-r_D)$.
:::

(sec-convergence-composition)=
## 3. A Complete Comparison-Matrix Proof

:::{div} feynman-prose
The matrices act on observables, which reverses the apparent order of the
algorithm. The swarm clones first and moves second. To estimate its final
energy, first bound the energy after motion, and then average that bound over
cloning. This gives the kinetic matrix followed algebraically by the cloning
matrix. The same calculation determines how noise injected by cloning is
subsequently damped.
:::

:::{prf:theorem} Foster-Lyapunov drift for the composed kernel
:label: thm-foster-lyapunov-main

Let $P_C,P_K$ be nonnegative sub-Markov kernels between the input, intermediate,
and output spaces, with the component observables defined on all three. Let the component estimates
of {prf:ref}`prop-complete-drift-summary` hold with constant nonnegative
matrices and finite nonnegative source vectors. For cloning followed by
kinetics, the observable kernel is $Q=P_CP_K$, and

$$
Q\mathbf V\le M\mathbf V+\mathbf b,
\qquad
M=A_KA_C,\qquad \mathbf b=A_K\mathbf b_C+\mathbf b_K.
$$

If $\mathbf w>0$ and $0\le r<1$ satisfy

$$
\mathbf w^\top M\le r\mathbf w^\top,
$$

then $V=\mathbf w^\top\mathbf V$ satisfies

$$
QV\le rV+b,\qquad b=\mathbf w^\top\mathbf b.
$$

For the historical observable $V_{\rm TV}$, take
$\mathbf w=(c_V,c_V,c_\mu,c_B)^\top$ whenever those weights satisfy the matrix
inequality. Constants are uniform in $N$ only if the component bounds and
weights, including their positive lower bounds when comparing individual
components, have that uniformity.
:::

:::{prf:proof}
For component $j$, first use the kinetic estimate:

$$
P_CP_KV_j
\le\sum_\ell(A_K)_{j\ell}P_CV_\ell
 +(\mathbf b_K)_jP_C1.
$$

All coefficients are nonnegative. Substitute
$P_CV_\ell\le\sum_i(A_C)_{\ell i}V_i+(\mathbf b_C)_\ell$ and use $P_C1\le1$.
This proves the vector estimate with the displayed product order and source.
Multiplication by $\mathbf w^\top$ gives

$$
QV\le\mathbf w^\top M\mathbf V+\mathbf w^\top\mathbf b
\le r\mathbf w^\top\mathbf V+b.
$$

Finiteness and uniformity follow exactly from the quantities used in these
inequalities. State-dependent matrices require bounding their expectations
through the intermediate state; their values cannot be multiplied as fixed
matrices.
:::

:::{prf:theorem} Positive weights and the spectral criterion
:label: thm-synergistic-rate-derivation

For a finite nonnegative matrix $M$, the following are equivalent:

1. There are $\mathbf w>0$ and $r<1$ with
   $\mathbf w^\top M\le r\mathbf w^\top$.
2. The spectral radius $\rho(M)$ is less than one.

For the two-component matrix

$$
M=\begin{pmatrix}1-\delta_x&a\\ b&1-\delta_v\end{pmatrix},
\qquad 0<\delta_x,\delta_v\le1,\qquad a,b\ge0,
$$

weights $(1,\omega)$ have positive effective decrements precisely when

$$
\delta_x-\omega b>0,\qquad \delta_v-a/\omega>0.
$$

If $a,b>0$, such weights exist exactly when

$$
ab<\delta_x\delta_v,
\qquad \frac a{\delta_v}<\omega<\frac{\delta_x}b.
$$

The resulting full-step decrement is
$\min(\delta_x-\omega b,\delta_v-a/\omega)$. Zero off-diagonal coefficients
are handled directly by the two strict inequalities.
:::

:::{prf:proof}
For the first implication, the weighted norm
$\|z\|_{1,\mathbf w}=\sum_iw_i|z_i|$ satisfies
$\|Mz\|_{1,\mathbf w}\le r\|z\|_{1,\mathbf w}$, since $M\ge0$.
Hence $\rho(M)\le r<1$.

Conversely, if $\rho(M)<1$, the finite-dimensional Neumann series converges.
Set

$$
\mathbf w^\top=\mathbf1^\top\sum_{n=0}^\infty M^n.
$$

Its entries are finite and at least one, and
$\mathbf w^\top M=\mathbf w^\top-\mathbf1^\top$. Thus
$r=\max_i(1-1/w_i)<1$ gives the required inequality.

For the two-component calculation,

$$
(1,\omega)M=(1-\delta_x+\omega b, a+\omega(1-\delta_v)).
$$

Divide the second coefficient by $\omega$. Positivity of both decrements is
the displayed interval condition, whose endpoints are ordered exactly when
$ab<\delta_x\delta_v$.
:::

:::{prf:theorem} Full-step rate and source
:label: thm-total-rate-explicit

For any admissible positive weights define

$$
r(\mathbf w)=\max_i\frac{(\mathbf w^\top M)_i}{w_i},\qquad
\kappa_{\rm drift}=1-r(\mathbf w),\qquad
b(\mathbf w)=\mathbf w^\top(A_K\mathbf b_C+\mathbf b_K).
$$

If $r(\mathbf w)<1$, these are a proved decrement and source for the weighted
moment observable. In the diagonal case,

$$
r(\mathbf w)=\max_i r_{K,i}r_{C,i},\qquad
\kappa_{\rm drift}=\min_i(1-r_{K,i}r_{C,i}),
$$

and any fixed positive weights work. Off-diagonal terms require the weight
criterion of {prf:ref}`thm-synergistic-rate-derivation`.

For $h>0$ and $0<r<1$, the corresponding physical-time exponential rate of
this moment bound is $-h^{-1}\log r$. It is not, by definition, a total
variation, relative entropy, Wasserstein, or extinction rate.
:::

:::{prf:proof}
Each component coefficient is bounded by $r(\mathbf w)w_i$, so the preceding
theorem applies. In the diagonal case the quotient cancels $w_i$. Finally,
$r^n=\exp[-(-\log r/h)nh]$.
:::

:::{prf:lemma} Iteration for conservative and killed kernels
:label: lem-convergence-drift-iteration

If $QV\le rV+b$, $Q1\le1$, and $0\le r<1$, then

$$
Q^nV\le r^nV+b\sum_{j=0}^{n-1}r^{n-1-j}Q^j1
\le r^nV+\frac{b(1-r^n)}{1-r}.
$$

Consequently, when $\mu Q^n1>0$,

$$
\Phi_n(\mu)V
\le\frac{r^n\mu V+b(1-r^n)/(1-r)}{\mu Q^n1}.
$$
:::

:::{prf:proof}
The first formula follows by induction, applying $Q$ to the preceding
inequality. The second uses $Q^j1\le1$. Integration against $\mu$ and division
by the positive survival probability gives the conditional bound.
:::

:::{div} feynman-prose
The denominator in the last expression is essential. A small unnormalized
moment can mean that most runs have died. To bound the size of the surviving
swarm, we must also control survival. The next section supplies two ways to
complete a convergence argument, according to which kernel is being studied.
:::

(sec-convergence-mixing)=
## 4. From Component Drift to Mixing

:::{div} feynman-prose
Drift brings trajectories back toward a controlled region. Mixing asks whether
two trajectories can lose their memory of where they started. A common part in
their transition laws provides that loss of memory. For a conservative process,
we combine this common part with drift. For a killed process, we compare whole
surviving blocks so that the preference for better-surviving paths is included.
:::

### 4.1. Conservative drift and minorization

:::{prf:theorem} A weighted coupling form of Harris' argument
:label: thm-convergence-conservative-harris

Let $P$ be a probability kernel on a standard Borel space. Let $V$ be finite
and nonnegative, with

$$
PV\le rV+b,\qquad 0\le r<1,\qquad b<\infty.
$$

Suppose $R>2b/(1-r)$ and there are $\epsilon\in(0,1]$ and a probability
measure $\eta$ such that

$$
P(x,\cdot)\ge\epsilon\eta(\cdot)
\quad\text{whenever }V(x)\le R.
$$

Choose $\beta>0$ so that $\beta(rR+2b)\le\epsilon$; if $rR+2b=0$, any
$\beta>0$ works. Define

$$
\|\mu-\zeta\|_\beta
=\int(1+\beta V)\,d|\mu-\zeta|.
$$

There is a $\rho_H<1$ such that

$$
\|\mu P-\zeta P\|_\beta\le\rho_H\|\mu-\zeta\|_\beta
$$

for all probabilities with finite $V$ moment. Hence $P$ has a unique invariant
probability $\pi$ in this class, and
$\|\mu P^n-\pi\|_\beta\le\rho_H^n\|\mu-\pi\|_\beta$.
:::

:::{prf:proof}
Use the cost

$$
d_\beta(x,y)=\begin{cases}0,&x=y,\\2+\beta V(x)+\beta V(y),&x\ne y.\end{cases}
$$

Its optimal transport cost between two probabilities is their displayed
weighted variation norm: couple their common part on the diagonal and couple
the mutually singular residual measures arbitrarily. Every coupling has at
least this cost, since off-diagonal mass must account for both residual
measures.

Write $s=V(x)+V(y)$. If $s\ge R$, any coupling gives

$$
\mathbb E d_\beta(X',Y')\le2+\beta(rs+2b)
\le\rho_1(2+\beta s),
$$

where

$$
\rho_1=\max\left(r,\frac{2+\beta(rR+2b)}{2+\beta R}\right)<1.
$$

The strict inequality follows from $(1-r)R>2b$; the ratio for $s\ge R$ is
bounded by its endpoint value and its limit $r$.

If $s<R$, both states are in the minorized set. With probability $\epsilon$,
draw the same point from $\eta$; couple the remaining transition mass
arbitrarily. Its probability of unequal points is at most $1-\epsilon$, and
its total expected $V$ moment is no greater than the unconditioned moment sum.
Therefore

$$
\mathbb E d_\beta(X',Y')
\le2(1-\epsilon)+\beta(rs+2b)\le2-\epsilon
\le(1-\epsilon/2)d_\beta(x,y)
$$

for $x\ne y$. For equal initial points use identical transitions. Integrating
these couplings gives contraction with
$\rho_H=\max(\rho_1,1-\epsilon/2)<1$.

Probabilities with finite $V$ moment form a complete space under
$\|\cdot\|_\beta$: a Cauchy sequence converges in the Banach space of signed
measures with weight $1+\beta V$, and positivity and total mass one pass to
the limit. The drift bound maps this space into itself. The contraction
mapping theorem now gives the invariant measure and the asserted convergence.
:::

:::{prf:remark} What must be verified for the gas
:label: rem-convergence-harris-inputs

This is the weighted coupling method of
[Hairer and Mattingly, *Yet another look at Harris' ergodic theorem*](https://arxiv.org/abs/0810.2777).
The comparison theorem above supplies the drift input when its component
hypotheses hold. The other input is a minorization of the **same probability
kernel** on the stated level set. On an unbounded domain, the level set needs
the established confining position control as well as centred variance bounds.
The theorem does not replace a killed kernel $Q$ by a probability kernel:
normalizing $Q(x,\cdot)$ separately at every step generally differs from
conditioning a complete trajectory on survival to time $n$.
:::

:::{prf:theorem} Accessibility and irreducibility
:label: thm-phi-irreducibility

Let $Q$ be a sub-Markov kernel. Suppose a measurable set $C$, an integer $m$,
and a nonzero finite measure $\eta$ satisfy
$Q^m(x,\cdot)\ge\epsilon\eta(\cdot)$ for all $x\in C$, with
$\epsilon>0$. Suppose also that, for every $x$, some $n=n(x)\ge0$ has
$Q^n(x,C)>0$. Then $Q$ is $\eta$-irreducible: for every $A$ with $\eta(A)>0$,
some iterate from every $x$ has positive mass on $A$.
:::

:::{prf:proof}
For the indicated $n$,

$$
Q^{n+m}(x,A)\ge\int_CQ^m(y,A)Q^n(x,dy)
\ge\epsilon\eta(A)Q^n(x,C)>0.
$$

This identifies the irreducibility measure as $\eta$. Lebesgue
irreducibility needs the corresponding accessibility of every set of positive
Lebesgue measure; a minorization on one interior ball alone does not identify
that larger irreducibility measure.
:::

:::{prf:theorem} One-step common mass excludes periodic classes
:label: thm-aperiodicity

Let $P$ be a probability kernel with $P(x,\cdot)\ge\epsilon\eta(\cdot)$ for
all $x$ in its state space, where $\epsilon>0$ and $\eta$ is a probability.
Then $P$ has no decomposition into two or more nonempty disjoint cyclic
classes $D_0,\ldots,D_{d-1}$ with $P(x,D_{j+1})=1$ for $x\in D_j$
(indices modulo $d$). It also satisfies

$$
\|\mu P-\zeta P\|_{\rm TV}\le(1-\epsilon)\|\mu-\zeta\|_{\rm TV}.
$$
:::

:::{prf:proof}
If such cyclic classes existed, the minorization and $P(x,D_{j+1})=1$ would
force $\eta(D_{j+1})=1$ for every $j$, contradicting disjointness. For the
contraction, write $P=\epsilon\eta+(1-\epsilon)R$ with $R$ a probability
kernel and use contraction of total variation under $R$. The case
$\epsilon=1$ is immediate.
:::

### 4.2. A complete killed-block criterion

:::{prf:theorem} QSD convergence from a two-sided surviving-block bound
:label: thm-main-convergence

Fix $N$. Suppose that for an integer $m\ge1$, a probability measure $\eta_N$
on $E_N$, and constants $0<c_N\le C_N<\infty$,

$$
c_N\eta_N(A)\le Q^m(S,A)\le C_N\eta_N(A)
\qquad(S\in E_N,\ A\text{ measurable}).
$$

Then $Q$ has a unique QSD $\nu_N$. With
$\rho_N=1-c_N/C_N\in[0,1)$,

$$
\|\Phi_n(\mu)-\nu_N\|_{\rm TV}
\le\rho_N^{\lfloor n/m\rfloor}
$$

for every initial probability $\mu$. At exponent zero the right-hand side is
interpreted as one. All these conditional laws are well-defined.
Its one-step survival eigenvalue satisfies

$$
c_N^{1/m}\le\alpha_N\le\min(1,C_N^{1/m}).
$$

The same existence and uniqueness conclusion follows from the alternative
full-block conditioned contraction criterion in
{prf:ref}`prop-latent-fractal-gas-conditional-qsd`. These are sufficient
criteria for the specified killed finite-swarm kernel, with constants that may
depend on $N$.
:::

:::{prf:proof}
Put $T=Q^m$ and suppress $N$. Its bounds imply
$c\le T1\le C$ and $Q1\ge Q^m1\ge c$, so every finite survival probability
is positive. Let $h_j=T^j1$. For a horizon $k$ define probability kernels

$$
R_{j,k}(x,dy)=\frac{T(x,dy)h_{k-j-1}(y)}{h_{k-j}(x)},
\qquad 0\le j<k.
$$

Since $h_{k-j}(x)\le C\eta(h_{k-j-1})$, these satisfy

$$
R_{j,k}(x,dy)\ge\frac cC
 \frac{h_{k-j-1}(y)\eta(dy)}{\eta(h_{k-j-1})}.
$$

The common probability measure on the right is independent of $x$.
Consequently each $R_{j,k}$ contracts total variation by at most
$\rho=1-c/C$, by the decomposition used in
{prf:ref}`thm-aperiodicity`.

Starting $R_{0,k},\ldots,R_{k-1,k}$ from
$\mu_k(dx)=h_k(x)\mu(dx)/\mu h_k$ gives the final law
$\mu T^k/\mu T^k1$: the factors $h_j$ cancel along a path. Two initial
probabilities therefore give final laws at distance at most $\rho^k$, even
though their initial reweightings differ. Write
$F_k(\mu)=\mu T^k/\mu T^k1$. We have proved

$$
\sup_{\mu,\zeta}\|F_k(\mu)-F_k(\zeta)\|_{\rm TV}\le\rho^k.
$$

Normalization cancels in composition, so $F_kF_l=F_{k+l}$.
For every $l\ge0$,
$\|F_{k+l}(\mu)-F_k(\mu)\|_{\rm TV}\le\rho^k$.
Thus $F_k(\mu)$ converges in the complete total variation space of
probabilities to a limit $\nu$ independent of $\mu$.
The map $F_1$ is continuous because $T1\ge c$ and $T1\le C$; taking limits
in $F_1F_k=F_{k+1}$ gives $F_1(\nu)=\nu$. The diameter bound proves
uniqueness of this fixed point and $\|F_k(\mu)-\nu\|_{\rm TV}\le\rho^k$.

The one-step conditional map $\Phi_1$ commutes with $F_1$, so
$\Phi_1(\nu)$ is another fixed point of $F_1$ and equals $\nu$.
Hence $\nu Q=\alpha\nu$ with $\alpha=\nu Q1>0$.
Every QSD is a fixed point of $F_1$, proving uniqueness. Finally, if
$n=km+j$ with $0\le j<m$, then
$\Phi_n(\mu)=F_k(\Phi_j(\mu))$, giving the displayed bound.
Integrating the two-sided bound on $Q^m1$ against $\nu$ gives
$c\le\alpha^m\le C$; combine this with $\alpha\le1$ for the eigenvalue bounds.
:::

:::{prf:remark} Analytic scope of the block estimate
:label: rem-convergence-block-verification

A proof of the displayed two-sided bound must handle the full swarm,
intermediate killing, all alive-status transitions, and the actual collision
and cap maps. Gaussian density bounds on a bounded interior region are usable
ingredients. Uniform ellipticity and bounded positions alone do not establish
the required bound for a split, interacting kernel.

In particular, positive companion probabilities do not imply positive cloning
probabilities for all alive walkers: competitive cloning can have probability
zero. A one-step velocity Gaussian also has only $Nd$ noise coordinates for a
$2Nd$ phase-space update. A full density claim needs a multi-step
controllability or change-of-variables argument. A hard radial projection can
produce boundary atoms; a smooth cap has a different density transformation.
The reference measure $\eta_N$ must accommodate the actual transition law.

The general theory of conditioned convergence is developed by
[Champagnat and Villemonais](https://arxiv.org/abs/1404.1349).
The theorem above gives an elementary sufficient condition with its complete
proof; it also isolates the kernel estimate needed to use that route here.
For unbounded latent spaces, the volume's confining estimates and a compatible
conditioned mixing argument are required rather than a new compactness
assumption.
:::

(sec-convergence-consequences)=
## 5. Moments, Survival, Entropy, and Concentration

:::{div} feynman-prose
A QSD looks stationary only after the surviving runs have been renormalized.
Before renormalization its mass decreases by a factor $\alpha_N$ each step.
That factor enters the moment balance. Likewise, a large swarm survives for a
long time only if many walkers can survive together without a common failure
mechanism. A count of interior walkers is one part of that argument;
probability estimates for their joint failures are another.
:::

### 5.1. Moment bounds under a QSD

:::{prf:proposition} Eigenmeasure, symmetry, and marginal statements
:label: prop-qsd-properties

For any QSD $\nu_N$ with eigenvalue $\alpha_N$:

1. $\nu_NQ^n=\alpha_N^n\nu_N$ and its survival law is geometric in the step
   index. If $\alpha_N<1$, $\mathbb E_{\nu_N}T_\dagger=1/(1-\alpha_N)$.
2. If $Q$ commutes with a measurable symmetry and its QSD is unique, $\nu_N$
   has that symmetry. Exchangeability follows when the actual swarm kernel
   commutes with every permutation of walker labels.
3. Its marginal densities and correlations are determined by the full
   eigenmeasure equation. A Gibbs spatial density or Gaussian velocity
   marginal requires a separate solution or approximation theorem for that
   equation, including cloning, capping, and killing.
:::

:::{prf:proof}
Iteration proves the eigenmeasure identity. Summing
$\mathbb P_{\nu_N}(T_\dagger>n)=\alpha_N^n$ over $n\ge0$ gives the mean
lifetime. If $g$ is a symmetry, the pushforward $g_\#\nu_N$ satisfies the same
eigenmeasure equation and normalization; uniqueness identifies it with
$\nu_N$. Taking coordinate projections of the equation gives the marginal
relations, with interactions retained inside the projected transition.
:::

:::{prf:theorem} Equilibrium moment bounds with the survival eigenvalue
:label: thm-equilibrium-variance-bounds

Suppose $QV\le rV+b$, $\nu_NQ=\alpha_N\nu_N$, and $\nu_NV<\infty$.
If $\alpha_N>r$, then

$$
\nu_NV\le\frac b{\alpha_N-r}.
$$

More generally, if $Q\mathbf V\le M\mathbf V+\mathbf b$,
$\nu_N\mathbf V$ is finite componentwise, and $\rho(M)<\alpha_N$, then

$$
\nu_N\mathbf V\le(\alpha_N I-M)^{-1}\mathbf b.
$$

For a conservative invariant measure, $\alpha_N=1$. Under the block bounds of
{prf:ref}`thm-main-convergence`, the integrability premise holds for every
nonnegative $V$ with $\eta_NV<\infty$, since
$\nu_N\le(C_N/\alpha_N^m)\eta_N$.
In that case $r<c_N^{1/m}$ is a directly checkable sufficient survival-gap
condition and gives $\nu_NV\le b/(c_N^{1/m}-r)$.
:::

:::{prf:proof}
Integrating the scalar drift gives
$\alpha_N\nu_NV\le r\nu_NV+b$. Subtract the finite term $r\nu_NV$ and
divide by $\alpha_N-r>0$.

For the vector bound, put $u=\nu_N\mathbf V$. Then
$u\le(M/\alpha_N)u+\mathbf b/\alpha_N$. Iterating $l$ times yields

$$
u\le(M/\alpha_N)^l u
 +\sum_{j=0}^{l-1}(M/\alpha_N)^j\mathbf b/\alpha_N.
$$

The first term tends to zero because $u$ is finite and
$\rho(M/\alpha_N)<1$. The nonnegative series converges to the stated inverse.
Finally $\alpha_N^m\nu_N=\nu_NQ^m\le C_N\eta_N$, proving the integrability
criterion. The bounds are inequalities; stationarity does not turn an upper
drift estimate into an exact moment formula.
:::

### 5.2. Direct survival estimates

:::{prf:lemma} Barrier control counts interior walkers
:label: lem-convergence-interior-count

Suppose $W_b=N^{-1}\sum_{i\in A}\phi(x_i)\le L$ and
$\phi\ge H>0$ outside a chosen interior set $K$. Then

$$
\#\{i\in A:x_i\in K\}\ge |A|-NL/H.
$$

In particular, after a cloning step with $N$ alive walkers, at least
$N(1-L/H)$ lie in $K$ when $H>L$.
:::

:::{prf:proof}
Every alive index outside $K$ contributes at least $H$ to
$NW_b\le NL$. Therefore the number of such indices is at most $NL/H$.
Subtract it from $|A|$.
:::

:::{prf:proposition} Joint failure bounds and the extinction scale
:label: prop-convergence-survival-bound

Suppose every state in a class $G_N$ has $m_N\ge2$ identified distinct walker
indices whose final alive indicators are part of the next swarm state.
Assume for every subset $I$ of these indices and every state $S\in G_N$,

$$
\mathbb P_S(\text{every walker in }I\text{ dies during the step})\le p^{|I|},
\qquad 0<p<1.
$$

Then

$$
1-Q1(S)\le m_Np^{m_N-1},\qquad S\in G_N.
$$

If a QSD exists and $\nu_N(G_N^c)\le a_N$, its one-step extinction hazard
satisfies

$$
1-\alpha_N\le a_N+m_Np^{m_N-1}.
$$

Consequently $\mathbb E_{\nu_N}T_\dagger\ge
[a_N+m_Np^{m_N-1}]^{-1}$ whenever the right side is defined and
$\alpha_N<1$. If $m_N\ge qN$ for a fixed $q>0$ and
$a_N\le Ae^{-bN}$, this is an exponential lower bound on the mean lifetime,
up to the polynomial factor $m_N$. An exponential upper bound requires a
separate lower bound on the extinction hazard.
:::

:::{prf:proof}
If fewer than two identified walkers survive, some subset of $m_N-1$ of them
has entirely died. There are $m_N$ such subsets. Apply the union bound and the
joint failure estimate. Integrate $1-Q1$ against $\nu_N$, bounding it by one
on $G_N^c$. The eigenmeasure survival formula gives the lifetime bound. For
$m_N\ge qN$, $p^{m_N-1}$ decays exponentially in $N$.

The joint failure hypothesis follows, for example, from conditional
independence of the identified deaths with individual probabilities at most
$p$, or from sequential conditional failure bounds at most $p$. Shared
cloning randomness and state-dependent interactions must be included in the
conditioning before either property can be invoked.
:::

:::{prf:proposition} Eventual absorption from a uniform block hazard
:label: rem-extinction-inevitable

If $Q^m1(S)\le1-\delta$ for every $S\in E_N$, with $m\ge1$ and
$\delta>0$, then

$$
\mathbb P_\mu(T_\dagger>km)\le(1-\delta)^k,
\qquad \mathbb E_\mu T_\dagger\le m/\delta.
$$

Thus absorption is almost sure. Conversely, a uniform one-step upper hazard
$1-Q1\le\delta$ gives
$\mathbb P_\mu(T_\dagger\le n)\le n\delta$.
:::

:::{prf:proof}
For the first statement, apply the block bound conditionally at times
$0,m,2m,\ldots$. Sum the resulting tail bound in blocks of length $m$ to
bound the expected lifetime. For the second, sum the conditional probability
of first absorption at each of the $n$ steps.

Gaussian velocity noise alone does not establish a uniform block hazard for
capped motion. In particular, a walker farther than $hv_{\max}$ from a
boundary cannot cross it in a single capped position step. The exit estimate
must use the positions, intermediate updates, and the stated block length.
:::

### 5.3. Entropy and concentration for the specified limiting law

:::{prf:theorem} Entropy dissipation implies total variation convergence
:label: thm-convergence-entropy-to-tv

Let $\nu$ be a fixed probability law and let $\mu_t\ll\nu$ be the evolution
being studied, which may be a conditioned evolution when its entropy identity
has been established. Suppose a nonnegative functional $\mathcal E(t)$ obeys

$$
H(\mu_t\mid\nu)\le A\mathcal E(t),\qquad
\mathcal E'(t)\le-\lambda\mathcal E(t),\qquad
A<\infty,\quad\lambda>0.
$$

Then

$$
\|\mu_t-\nu\|_{\rm TV}
\le\sqrt{A\mathcal E(0)/2}\,e^{-\lambda t/2}.
$$

For a discrete bound $\mathcal E_{n+1}\le r_E\mathcal E_n$, replace
$e^{-\lambda t/2}$ by $r_E^{n/2}$. The entropy and hypocoercive estimates in
{doc}`10_kl_hypocoercive` and {doc}`15_kl_convergence` can be used here when
their reference measure and evolution coincide with $\nu$ and $\mu_t$.
In particular, {prf:ref}`thm-kl-convergence-euclidean` gives the entropy route
under its stated modified-Fisher dissipation assumptions. For a QSD this
includes the killing and normalization contribution to the entropy balance,
identified in {prf:ref}`prop-kl-conditioned-entropy`.
:::

:::{prf:proof}
Integration gives $\mathcal E(t)\le e^{-\lambda t}\mathcal E(0)$.
For completeness, let $A_+=\{d\mu/d\nu\ge1\}$ and set
$p=\mu(A_+)$, $q=\nu(A_+)$. Then
$p-q=\|\mu-\nu\|_{\rm TV}$. Jensen's inequality on $A_+$ and its complement
gives

$$
H(\mu\mid\nu)\ge p\log(p/q)+(1-p)\log((1-p)/(1-q)).
$$

As a function of $p$, the right side vanishes with zero derivative at $p=q$,
and its second derivative is $1/[p(1-p)]\ge4$. It is therefore at least
$2(p-q)^2$, with the endpoint cases obtained by continuity or infinite
entropy. Combining this inequality with the dissipation estimate proves the
result.
:::

:::{prf:theorem} Joint-law LSI gives concentration of swarm averages
:label: thm-convergence-lsi-concentration

Suppose a probability law $\nu_N$ on the specified swarm coordinates satisfies

$$
\operatorname{Ent}_{\nu_N}(f^2)
\le\frac2{\rho_N}\int\|\nabla f\|^2\,d\nu_N,
\qquad \rho_N>0.
$$

Let $F$ be a bounded observable in the associated form domain with
$\|\nabla F\|^2\le L^2/N$ almost everywhere, where $L>0$. Then

$$
\nu_N(|F-\nu_NF|\ge t)
\le2\exp\left(-\frac{\rho_NNt^2}{2L^2}\right),\qquad t>0.
$$

For $F=N^{-1}\sum_{i=1}^N\varphi(z_i)$ and
$\|\nabla\varphi\|\le L$, the gradient condition follows directly in the
product Euclidean metric, even when the law is interacting. An $N$-uniform
LSI yields an $N$-uniform concentration coefficient in the exponent.
Different metric or diffusion conventions require the corresponding gradient
bound in that Dirichlet form. A law with alive-status strata also requires
an LSI that controls the stated observable across those strata.
:::

:::{prf:proof}
Put $X=F-\nu_NF$ and $\psi(\theta)=\log\nu_Ne^{\theta X}$.
Apply the LSI to $f=e^{\theta X/2}$ and divide by
$\nu_Ne^{\theta X}$. The result is

$$
\theta\psi'(\theta)-\psi(\theta)
\le\frac{\theta^2L^2}{2\rho_NN}.
$$

For $\theta>0$, integrate the inequality for $(\psi(\theta)/\theta)'$,
using $\psi(0)=\psi'(0)=0$, to obtain
$\psi(\theta)\le\theta^2L^2/(2\rho_NN)$.
Exponential Markov inequality gives
$\nu_N(X\ge t)\le\exp[-\theta t+\theta^2L^2/(2\rho_NN)]$.
Minimize at $\theta=\rho_NNt/L^2$ and repeat with $-X$.
For the empirical average,
$\|\nabla F\|^2=N^{-2}\sum_i\|\nabla\varphi(z_i)\|^2\le L^2/N$.
:::

:::{prf:remark} Concentration inputs and finite-population limits
:label: rem-convergence-concentration-inputs

The LSI in this result is an inequality for the full law $\nu_N$, with its
specified form and constant. It is not supplied by a scalar moment drift or
by applying an independent-sample bounded-differences inequality to correlated
walkers. The uniform-law hypotheses and the resulting LSI are stated in
{prf:ref}`cor-n-uniform-lsi`; the choice of conservative, quasi-stationary,
or conditioned-process reference law is specified in
{prf:ref}`def-kl-finite-particle-laws`. The adaptive metric hypotheses are
treated in {doc}`17_geometric_gas`.

Finite-$N$ conditional mixing does not by itself establish existence or
uniqueness of a nonlinear mean-field stationary law. Passing to that limit
also uses tightness, identification of the limiting equation, control of
survival normalization, and the appropriate propagation-of-chaos estimates.
Those arguments belong to {doc}`08_mean_field`, {doc}`09_propagation_chaos`,
and {doc}`16_continuum_discharge`. Exchanging the limits $N\to\infty$,
$n\to\infty$, and $h\to0$ requires the uniform estimates stated there.
:::

:::{prf:proposition} Mixing time from the proved contraction
:label: prop-mixing-time-explicit

Under {prf:ref}`thm-main-convergence`, for $0<\rho_N<1$ and
$0<\varepsilon<1$, the step count

$$
n\ge m\left\lceil\frac{\log(1/\varepsilon)}{-\log\rho_N}\right\rceil
$$

ensures $\|\Phi_n(\mu)-\nu_N\|_{\rm TV}\le\varepsilon$ for every $\mu$.
When $\rho_N=0$, one block suffices. Under a bound
$\|\mu P^n-\pi\|\le B_\mu\rho_H^n$, use
$\lceil\log(B_\mu/\varepsilon)/(-\log\rho_H)\rceil_+$ instead.
Physical time is the step count multiplied by $h$.
:::

:::{prf:proof}
Solve the corresponding geometric inequality for the integer number of
blocks or steps. Neither formula identifies its contraction factor with
$1-\kappa_{\rm drift}$; the required minorization or entropy constants enter
through the theorem that established convergence.
:::

(sec-convergence-sensitivity)=
## 6. Parameter Dependence and Sensitivity

:::{div} feynman-prose
Once the inequalities are assembled, parameter dependence becomes a concrete
calculation. Changing a parameter can alter a diagonal damping term, an
interaction term, or the injected noise. A table of guessed powers hides these
possibilities. We instead differentiate a specified bound, keeping its domain
of validity in view.
:::

:::{prf:definition} Algorithm parameters and derived analysis quantities
:label: def-complete-parameter-space

A historical analysis vector is

$$
\mathbf P=(\lambda,\sigma_x,\alpha_{\rm rest},\lambda_{\rm alg},
\epsilon_c,\epsilon_d,\gamma,\sigma_v,h,N,\kappa_{\rm wall},d_{\rm safe}).
$$

Here $\sigma_x$ is the position jitter standard deviation;
$\alpha_{\rm rest}\in[0,1]$ is the inelastic restitution coefficient;
$\lambda_{\rm alg}\ge0$ weights velocity in companion distance;
$\epsilon_c,\epsilon_d>0$ are the cloning and diversity companion scales;
$\gamma>0$ is friction; $\sigma_v$ specifies the chosen velocity-noise
convention; $h>0$ is the numerical step; and $N\ge2$ is an integer.
The quantities $\kappa_{\rm wall}$ and $d_{\rm safe}$ parametrize a declared
boundary potential and interior set when that model uses them.

The symbol $\lambda$ is a derived cloning frequency or a parameter of an
explicitly specified rate model. The actual algorithm uses its fitness-based
cloning probabilities; introducing an independent $\lambda$ must not change
those probabilities implicitly. This vector is a selection of coordinates
used for analysis, not an exhaustive list of the latent algorithm's controls.
The full definitions, including fitness and metric regularizers, are in
{doc}`../1_the_algorithm/02_fractal_gas_latent` and
{doc}`../1_the_algorithm/03_parameter_constraints`.

Logarithmic derivatives below use a chosen set of strictly positive,
continuous coordinates. Zero restitution or zero velocity-distance weight
requires ordinary or one-sided derivatives. Treating $N$ as continuous is an
explicit relaxation whose proposed values must later be checked at integers.
:::

:::{prf:proposition} What a parameter can change in a proved bound
:label: prop-parameter-classification

Suppose a scalar moment estimate has differentiable coefficients
$r(\mathbf P)<1$ and $b(\mathbf P)>0$. Its conservative equilibrium bound is
$B=b/(1-r)$, so for any continuous parameter $p$,

$$
\partial_pB=\frac{\partial_pb}{1-r}
 +\frac{b\,\partial_pr}{(1-r)^2}.
$$

For a QSD with a differentiable eigenvalue $\alpha(\mathbf P)>r(\mathbf P)$,
the corresponding bound is $B_Q=b/(\alpha-r)$ and

$$
\partial_pB_Q=\frac{\partial_pb}{\alpha-r}
 -\frac{b(\partial_p\alpha-\partial_pr)}{(\alpha-r)^2}.
$$

Thus a parameter can affect damping, the additive source, and survival at the
same time. Classification as affecting only one of these quantities requires
those other derivatives to vanish in the particular bound being used.
:::

:::{prf:proof}
Differentiate the two quotient formulas on their stated domains.
:::

:::{prf:definition} Rate sensitivity matrix
:label: def-rate-sensitivity-matrix

Choose positive differentiable rate functions
$\kappa_1(\mathbf P),\ldots,\kappa_q(\mathbf P)$ obtained from specified
analytic inequalities, and positive continuous coordinates
$p_1,\ldots,p_s$. Put $x_j=\log p_j$ and

$$
(M_\kappa)_{ij}=\frac{\partial\log\kappa_i}{\partial x_j}
=\frac{p_j}{\kappa_i}\frac{\partial\kappa_i}{\partial p_j}.
$$

Rates may describe component moments, a conditioned block contraction, or
entropy dissipation, but their type and units must be recorded. The minimum
of rates of different quantities has no automatic convergence interpretation.
For empirically fitted rates, the same matrix is a local sensitivity estimate
of the fit.
:::

:::{prf:theorem} Explicit sensitivity from the comparison matrix
:label: thm-explicit-rate-sensitivity

Suppose $M(\mathbf P)$ is differentiable and positive weights $\mathbf w$ are
held fixed. For every component with

$$
\kappa_i=1-\frac{(\mathbf w^\top M)_i}{w_i}>0,
$$

its logarithmic sensitivity is

$$
(M_\kappa)_{ij}
=-\frac{p_j}{\kappa_iw_i}
 \sum_\ell w_\ell\frac{\partial M_{\ell i}}{\partial p_j}.
$$

Since $M=A_KA_C$,

$$
\partial_pM=(\partial_pA_K)A_C+A_K(\partial_pA_C),
$$

$$
\partial_p\mathbf b
=(\partial_pA_K)\mathbf b_C+A_K\partial_p\mathbf b_C
 +\partial_p\mathbf b_K.
$$

If weights are themselves functions of the parameters, their derivatives
must also be included. At a change of the active minimum rate, use the
one-sided directional calculation below.
:::

:::{prf:proof}
Differentiate the component quotient with fixed $w_i$, then multiply by
$p_j/\kappa_i$. The remaining identities are the product rule applied to the
proved composition formulas.
:::

:::{prf:definition} Moment-bound sensitivity matrix
:label: def-equilibrium-sensitivity-matrix

For positive differentiable bounds $B_i(\mathbf P)$, define
$(M_B)_{ij}=\partial\log B_i/\partial\log p_j$.
For a scalar conservative bound $B=b/\kappa$,

$$
\partial\log B=\partial\log b-\partial\log\kappa.
$$

For a QSD bound replace $\kappa$ by $\alpha-r$. These derivatives describe
the upper bound; they equal derivatives of an actual equilibrium moment only
when an additional identity establishes equality with that bound.
:::

:::{prf:theorem} Singular directions of the sensitivity matrix
:label: thm-svd-rate-matrix

For a real $q\times s$ sensitivity matrix $M_\kappa$ of rank $r$, there are
orthonormal right and left singular vectors with singular values
$\sigma_1\ge\cdots\ge\sigma_r>0$. A right singular vector $v_i$ produces
first-order rate change $M_\kappa v_i=\sigma_i u_i$.
Moreover,

$$
\dim\ker M_\kappa=s-r\ge s-q.
$$

A vector in this kernel has zero first-order rate change at the point where
$M_\kappa$ was evaluated. This is a local statement; it does not by itself
produce a family of parameters with exactly equal rates.
:::

:::{prf:proof}
The symmetric positive semidefinite matrix $M_\kappa^\top M_\kappa$ has an
orthonormal eigenbasis. Its positive eigenvalues are $\sigma_i^2$. For their
unit eigenvectors $v_i$, set $u_i=M_\kappa v_i/\sigma_i$. Then
$u_i^\top u_j=\delta_{ij}$. Complete these vectors to orthonormal bases to
obtain a singular value decomposition. The zero eigenspace is precisely
$\ker M_\kappa$, proving the dimension formula. Differentiability gives
$\log\boldsymbol\kappa(x+tv)=\log\boldsymbol\kappa(x)
+tM_\kappa v+o(t)$, which gives the local interpretation.
:::

:::{prf:proposition} Conditioning on the identifiable subspace
:label: prop-condition-number-rate

When $r\ge1$, define the restricted condition number
$\operatorname{cond}_+(M_\kappa)=\sigma_1/\sigma_r$.
For $v\perp\ker M_\kappa$,

$$
\sigma_r\|v\|\le\|M_\kappa v\|\le\sigma_1\|v\|.
$$

The upper factor controls forward sensitivity. The lower factor controls
inversion on the identifiable subspace. If the matrix has a nontrivial
kernel, arbitrary parameter changes cannot be recovered uniquely from rate
changes.
:::

:::{prf:proof}
Expand $v$ in the right singular vectors with positive singular values and
bound $\sum_i\sigma_i^2v_i^2$ between
$\sigma_r^2\sum_iv_i^2$ and $\sigma_1^2\sum_iv_i^2$.
:::

:::{prf:theorem} Finite perturbations and the minimum rate
:label: thm-error-propagation

Let $g(x)=\log\boldsymbol\kappa(e^x)$ be continuously differentiable on a
convex neighbourhood. If $\|Dg(x)\|_2\le L$ there, then

$$
\|g(x+u)-g(x)\|_2\le L\|u\|_2.
$$

If $Dg$ is $L_1$-Lipschitz on that neighbourhood, then

$$
\|g(x+u)-g(x)-Dg(x)u\|_2\le\tfrac12L_1\|u\|_2^2.
$$

For positive component rates with a valid common interpretation,
$\log\kappa_{\min}=\min_i\log\kappa_i$, and

$$
|\Delta\log\kappa_{\min}|
\le\|\Delta\log\boldsymbol\kappa\|_\infty.
$$
:::

:::{prf:proof}
Integrate $Dg(x+tu)u$ over $t\in[0,1]$. For the remainder, subtract
$Dg(x)u$ and use $\|Dg(x+tu)-Dg(x)\|\le L_1t\|u\|$.
Finally, for any two vectors $a,b$,
$\min_i a_i\le\min_i b_i+\|a-b\|_\infty$; exchange $a,b$ to obtain the
absolute-value inequality.
:::

(sec-convergence-tuning)=
## 7. Tuning a Specified Bound

:::{div} feynman-prose
A slow component can limit a weighted moment estimate, so balancing component
rates is often sensible. But the slow component may already be at its own
maximum, or improving it may increase the noise source or reduce survival.
We therefore optimize a stated bound over stated constraints. A numerical
optimizer finds candidates; the inequalities determine what those candidates
mean.
:::

### 7.1. Optimization and the active rates

:::{prf:definition} Parameter optimization problem
:label: def-parameter-optimization

Let $\mathcal F$ be the set on which the chosen component, metric,
integrability, and transition estimates hold. For a proved moment bound,
one possible objective is

$$
\max_{\mathbf P\in\mathcal F,\ \mathbf w>0}
\left[1-\max_i\frac{(\mathbf w^\top M(\mathbf P))_i}{w_i}\right],
$$

with a weight normalization such as $\sum_iw_i=1$ and, when needed, positive
lower bounds on the weights. Source constraints can impose
$b/(1-r)\le B_{\max}$ for conservative moments or
$b/(\alpha-r)\le B_{\max}$ for an integrable QSD with $\alpha>r$.

Optimizing TV mixing instead uses its actual bound, for example
$-m^{-1}\log\rho_N$, together with the cost per block. Entropy optimization
uses its proved dissipation constant. Integer population sizes are compared
as discrete choices.
:::

:::{prf:theorem} Directional derivatives of a minimum
:label: thm-subgradient-min

Let $f(x)=\min_{1\le i\le q}f_i(x)$ with all $f_i$ continuously
differentiable near $x$. Define the active set
$I(x)=\{i:f_i(x)=f(x)\}$. Then

$$
f'(x;d)=\min_{i\in I(x)}\nabla f_i(x)\cdot d.
$$

If every $f_i$ is affine, $f$ is concave. In that case every convex
combination of the active gradients is a supergradient: it satisfies
$f(y)\le f(x)+g\cdot(y-x)$ for all $y$.
:::

:::{prf:proof}
Inactive functions have a strictly positive gap at $x$ and remain inactive
for sufficiently small positive movement along a fixed direction. Taylor
expansion for the finitely many active functions gives the directional
formula. For affine branches and active $i$,
$f(y)\le f_i(y)=f(x)+\nabla f_i(x)\cdot(y-x)$.
Taking a convex combination gives the supergradient inequality. The minimum
of affine functions is concave by this same inequality, or directly from its
hypograph as an intersection of half-spaces.
:::

:::{prf:theorem} A conditional balancing principle
:label: thm-balanced-optimality

At an interior local maximum $x_*$ of $\min_i f_i(x)$, if exactly one branch
$i_*$ is active, then $\nabla f_{i_*}(x_*)=0$. Consequently, if all possible
uniquely active branches have nonzero gradients, every interior local
maximizer has at least two active branches. Boundary constraints can instead
prevent further improvement of a single active branch.
:::

:::{prf:proof}
If a unique branch is active, continuity and the finite positive gap to the
other branches make the same branch active in a neighbourhood. The minimum
therefore equals the differentiable function $f_{i_*}$ locally. The ordinary
first-order condition for an interior local maximum gives zero gradient.
:::

:::{prf:theorem} Exact balancing in a two-rate allocation model
:label: thm-closed-form-optimum

Consider the auxiliary optimization problem

$$
\max_{\lambda,\gamma\ge0,\ \lambda+\gamma=B}
\min(a\lambda,b\gamma),\qquad a,b,B>0.
$$

Its unique optimizer is

$$
\lambda_*=\frac{bB}{a+b},\qquad
\gamma_* =\frac{aB}{a+b},\qquad
\kappa_* =\frac{abB}{a+b}.
$$

This solves the displayed linear resource model. Applying it to algorithmic
parameters requires independently establishing those rate formulas and that
resource constraint.
:::

:::{prf:proof}
Substitute $\gamma=B-\lambda$. The first branch increases strictly and the
second decreases strictly. Their minimum increases up to their unique
intersection and decreases afterwards. Solve
$a\lambda=b(B-\lambda)$ to obtain the formulas.
:::

### 7.2. Parameter couplings with explicit assumptions

:::{prf:proposition} Restitution and friction enter different energy terms
:label: prop-restitution-friction-coupling

For one collision group with conserved mean velocity $\bar v$ and collision
rule $v_i'=\bar v+\alpha_{\rm rest}(v_i-\bar v)$,

$$
\sum_i\|v_i'\|^2
=|G|\|\bar v\|^2
 +\alpha_{\rm rest}^2\sum_i\|v_i-\bar v\|^2.
$$

Thus restitution multiplies the relative energy of that group by
$\alpha_{\rm rest}^2$. If an independently proved complete-step velocity
bound is $QV_v\le r_v(\alpha_{\rm rest},\gamma)V_v+
 b_v(\alpha_{\rm rest},\gamma)$, its conservative moment bound is
$b_v/(1-r_v)$, and its QSD moment bound, under the integrability and survival
gap conditions, is $b_v/(\alpha_N-r_v)$.
:::

:::{prf:proof}
Expand the square in the collision rule. The cross term vanishes because
$\sum_i(v_i-\bar v)=0$. Apply
{prf:ref}`thm-equilibrium-variance-bounds` to the complete-step estimate.
Overlapping collision groups must follow the algorithm's actual update
convention; conservation for each isolated group cannot be summed as a
conservation law for overlapping overwrites.
:::

:::{prf:proposition} Jitter and contraction in a declared scalar model
:label: prop-jitter-cloning-coupling

Suppose a conservative positional estimate has, on its admissible parameter
range, the specific coefficients

$$
QV_x\le(1-c\lambda)V_x+B_0+B_1\sigma_x^2,
\qquad B_0,B_1\ge0,\quad0<c\lambda\le1.
$$

Then its invariant moment bound is
$(B_0+B_1\sigma_x^2)/(c\lambda)$ when the invariant moment is finite.
A sufficient condition for this bound to be at most $V_*$ is

$$
\lambda\ge\frac{B_0+B_1\sigma_x^2}{cV_*}.
$$

For the actual competitive cloning kernel, $c$, $\lambda$, and the source
must be derived from its probabilities and the kinetic composition. The
simple ratio above applies only to the displayed scalar model.
:::

:::{prf:proof}
Integrate the given inequality against the invariant law and rearrange.
The target-bound condition is the resulting scalar inequality solved for
$\lambda$.
:::

:::{prf:proposition} Companion geometry at a fixed swarm state
:label: prop-phase-space-pairing

For the Euclidean softmax companion law with fixed candidate set,

$$
P_i(j)\propto\exp\left[-\frac{\|x_i-x_j\|^2+
\lambda_{\rm alg}\|v_i-v_j\|^2}{2\epsilon_c^2}\right],
$$

define
$\Delta_x=\|x_i-x_j\|^2-\|x_i-x_l\|^2$ and
$\Delta_v=\|v_i-v_j\|^2-\|v_i-v_l\|^2$.
Then

$$
\log\frac{P_i(j)}{P_i(l)}
=-\frac{\Delta_x+\lambda_{\rm alg}\Delta_v}{2\epsilon_c^2},
$$

$$
\partial_{\lambda_{\rm alg}}\log\frac{P_i(j)}{P_i(l)}
=-\frac{\Delta_v}{2\epsilon_c^2},\qquad
\partial_{\log\epsilon_c}\log\frac{P_i(j)}{P_i(l)}
=\frac{\Delta_x+\lambda_{\rm alg}\Delta_v}{\epsilon_c^2}.
$$

These identities quantify companion reweighting at a fixed state. They do
not specify how fitness or the stationary law changes after the swarm evolves.
:::

:::{prf:proof}
The common softmax denominator cancels in the ratio. Differentiate its
logarithm while holding the state and candidate set fixed.
:::

### 7.3. A reviewable tuning procedure

:::{prf:algorithm} Selecting parameters from analytic bounds
:label: alg-param-selection

**Input:** a specified Euclidean or latent transition, a target observable,
and admissible parameter ranges.

1. Choose the object of study: conservative law, killed finite-swarm QSD,
   nonlinear mean-field law, or a stated continuous-time limit.
2. Collect the component inequalities for that transition, retaining their
   alive-normalization, metric, integrability, and occupancy hypotheses.
3. Form $M=A_KA_C$ and $\mathbf b=A_K\mathbf b_C+\mathbf b_K$.
   Find positive weights with $\mathbf w^\top M<\mathbf w^\top$.
4. For a mixing target, add the required minorization, killed-block, or
   entropy-dissipation estimate. For a QSD moment target also check
   integrability and $\alpha_N>r$.
5. Evaluate the resulting rate, source, and computational cost. Compare
   feasible parameter choices and verify the proposed choice in the original
   inequalities.

**Output:** parameters together with the bound they satisfy, or a list of
analytic inputs still requiring verification for that proposed choice.
:::

:::{prf:algorithm} Projected optimization of a declared objective
:label: alg-projected-gradient-ascent

Fix a continuously parametrized feasible region where the selected bound is
valid, and keep integer parameters in an outer search. At each iterate:

1. Evaluate the objective and all active component inequalities.
2. Compute their derivatives, including parameter-dependent weights and
   source terms. For a minimum objective use the active directional
   derivative from {prf:ref}`thm-subgradient-min`.
3. Propose a feasible direction and a step size. A projection requires a
   specified projection rule and a region on which it is defined.
4. Accept the proposed point only after checking feasibility and the intended
   improvement of the actual objective; otherwise shorten the step or stop.
5. Report the final objective, active constraints, and verification residuals.

This procedure defines a numerical search. Global optimality requires the
convexity or other structural hypotheses of a separate optimization theorem.
:::

:::{prf:definition} Pareto comparison
:label: def-pareto-optimality

A feasible parameter choice is Pareto optimal for specified objectives, such
as a proved mixing rate, moment bound, and cost, if no other feasible choice
improves at least one objective without worsening another. All objectives
must have declared directions of improvement and be evaluated for the same
model and parameter convention.
:::

:::{prf:algorithm} Empirical tuning with analytic checks
:label: alg-adaptive-tuning

For a fixed parameter choice, record empirical cloning frequency, component
moment changes, alive counts, and the costs relevant to the chosen objective.
Estimate sensitivities with repeated runs and quantified sampling error.
Use those estimates to propose a new parameter choice, then re-evaluate the
analytic constraints and run the updated configuration.

Diagnostics in `src/fragile/fractalai/convergence_bounds.py` provide computed
bounds or proxies according to their documented inputs. A fitted decay rate,
a positive proxy, or a finite trajectory cannot establish a uniform kernel
inequality. Online adaptation also changes the process into a time-dependent
kernel unless the controller is included in the state; convergence claims
then require estimates uniform along that adaptation or a separate adaptive
process analysis.
:::

:::{div} feynman-prose
The calculations now have distinct jobs. The comparison matrix bounds moments
of the actual composed step. Kernel overlap or an entropy estimate controls
loss of memory. The QSD eigenvalue controls survival. A joint-law LSI controls
fluctuations of observables. Keeping those jobs explicit lets us reuse each
proved estimate without asking it to answer a different probability question.
:::
