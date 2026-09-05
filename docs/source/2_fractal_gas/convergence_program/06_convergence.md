# Convergence, Survival, and Parameter Dependence

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
the cemetery extension. If cloning has a separate intermediate state space,
the observables must be defined there and the kinetic estimate must cover
every intermediate state to which cloning assigns mass. This includes proposed
jittered positions under the specified status-update convention. The diagonal
input frequently used in the component
chapters has the form

$$
A_C=\operatorname{diag}(r_{C,x},1,r_{C,\mu},r_{C,b}),\qquad
A_K=\operatorname{diag}(r_{K,x},r_{K,v},r_{K,\mu},r_{K,b}).
$$

Here $r_{C,x}=1-\kappa_x$ and $r_{C,b}=1-\kappa_b$ when the corresponding
cloning contraction estimates hold. A bounded expansion uses coefficient one;
a uniform bound uses coefficient zero. Every source vector must include the
jitter, collision, and discretization terms appropriate to that estimate.

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

For a capped BAOAB position update without a separate position Gaussian,
$D_h^2=h^2v_{\max}^2$. These are different kinetic specifications.
:::

:::{prf:proof}
Apply the minimum-over-centres calculation of the preceding lemma to the
surviving subset, then enlarge the nonnegative squared-increment sum to all
input alive indices before taking expectations. For EG-kin$^+$, independence
and centring remove the mixed term, while
$\mathbb E\|\xi_i^x\|^2=d$. The cap bounds the remaining deterministic
velocity contribution.
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
:::

:::{prf:proof}
Apply $P_C$ to the kinetic inequality. Since $P_C1\le1$, its constant term
increases by at most the displayed amount. Substitute the cloning bound and
collect the coefficient and source. The restriction on $\theta$ is exactly
$(1+\theta)r_{C,x}<1$.
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
