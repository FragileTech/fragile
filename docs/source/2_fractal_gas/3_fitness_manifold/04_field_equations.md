(sec-field-equations-pressure)=
# Field Equations and Pressure Dynamics

**Prerequisites:** {doc}`01_emergent_geometry`,
{doc}`02_scutoid_spacetime`, and {doc}`03_curvature_gravity`.

(sec-field-equations-tldr)=
## TLDR

The primary field equations in this chapter are exact finite-step laws of
the algorithm. The full transition kernel determines the conditional drift
and fluctuations of an observable, including a specified reconstruction of
the fitness metric. Operator increments telescope to the full change in
energy, momentum, or empirical density. These identities require no Gibbs
law, continuum limit, or gravitational constitutive equation.

The Rust BAOAB thermostat gives, for one unit-mass walker and a frozen noise
factor $B$,

$$
\mathbb E[\Delta K\mid v,B]
=(c^2-1)K+\frac{s^2}{2}\operatorname{tr}(BB^\top),\qquad
c=e^{-\gamma h},\quad
s^2=\begin{cases}(1-e^{-2\gamma h})/(2\gamma),&\gamma>0,\\h,&\gamma=0.
\end{cases}
$$

Geometry enters this measured budget through the actual diffusion factor.
An equation for the metric alone needs an additional closure result.

Later sections retain comparison density and fluctuation models. A Gaussian
pair energy has an explicit dilation derivative. Its quadratic stiffness
and its pressure are different derivatives.

A homogeneous diffusion and Gaussian gain–loss model has exact decay rate

$$
\omega(k)=D_{\mathrm{eff}}|k|^2+
 \lambda_{\mathrm{kill}}(1-e^{-\varepsilon_c^2|k|^2/2}).
$$

Every nonzero mode decays when
$D_{\mathrm{eff}}+\lambda_{\mathrm{kill}}>0$. A periodic domain has
a positive first nonzero frequency; on the whole space frequencies
approach zero and there is no uniform exponential $L^2$ rate.

The finite Gaussian mode ensemble gives pressure by differentiating its
partition function. The Einstein-type Ricci contraction is a comparison
calculation under a specified constitutive equation. The actual swarm's QSD,
mean-field, concentration, and entropy results supply separate,
established analytical estimates with their stated hypotheses.

(sec-field-equations-intro)=
## Introduction

:::{div} feynman-prose
Start with a single completed update. We know the population before it and
after it, and we know which instructions produced the change. Ask how much
the local ruler, the kinetic energy, or the momentum changed at each
instruction. Average over fresh realizations of that same update. That
gives a prediction which another set of realizations can test.

A ruler attached to a moving walker can change because the walker moved,
because the population changed the landscape, or because cloning replaced
its ancestry. We must separate these changes before assigning any of them
a field equation. The identities below do this at the actual time step.

To calculate pressure, imagine expanding the region occupied by a fixed
amount of material. Specify what moves, what stays fixed, and how its
energy changes. Pressure is the negative derivative of that energy
with respect to volume.

A different experiment changes the density profile inside a fixed region.
Its quadratic energy measures stiffness. There is no reason for the
first derivative in the expansion experiment to equal the second
derivative in the density experiment.

We will do both calculations. We will also solve a homogeneous model
exactly in Fourier space and calculate the long-time spreading of an
Ornstein–Uhlenbeck velocity process. Each calculation has a definite
model and normalization. These results become statements about measured
walkers when the corresponding law or evolution has been identified.
:::

(sec-algorithmic-metric-evolution)=
## Exact Metric Evolution from the Transition Law

:::{div} feynman-prose
Imagine saving the complete experiment immediately before an update.
Restart it many times, keeping the physical state fixed and supplying
fresh random draws. You get a distribution of next states. The average
change of any measured quantity is its conditional drift; the spread
around that average is its fluctuation.

The word “complete” matters. A saved cloud of positions is insufficient
when the next donor may come from retained history. A saved velocity is
insufficient when a provider has internal state or a schedule changes the
next update. And replaying the identical random seed gives the identical
future, rather than an independent experiment. We distinguish replay
from resampling the future innovations.
:::

:::{prf:definition} Extended state and transition experiment
:label: def-algorithmic-transition-state

Fix population capacity $N$, step size $h$, the algorithm configuration,
and the specified input protocol. Let $X_n$ contain the population
(including validity and ancestry), retained donor history, provider and
domain state, and the schedule and random-stream addresses needed at step
$n$. Include changing external inputs in the state, or condition on their
specified protocol. A measurable update is

$$
X_{n+1}=F_{N,h}(X_n,\xi_{n+1}),\qquad
\mathsf P_{N,h}A(X)=
\mathbb E[A(F_{N,h}(X,\xi))\mid X].
$$

The conditional law of $\xi$ includes the configured donor, acceptance,
revival, jitter, and kinetic innovations. Independence is used only where
that law supplies it. Write $\mathscr F_n$ for the information available
through $X_n$ before these future innovations are exposed.

The stochastic experiment uses the declared innovation law. A numerical
replay with a fixed complete pseudo-random future is deterministic.
Checkpoint-conditioned replicas hold the algorithmic state fixed and
resample future innovation keys under a recorded ensemble protocol; they
do not condition on an already exposed future key.

An observable $A:\mathcal X\to\mathbb R^m$ includes its reconstruction
convention and reference coordinates. When extinction is possible, extend
the transition to a cemetery outcome and specify $A$ there. If a metric
does not exist at an outcome, report that outcome and its probability;
discarding it changes the conditional experiment. A killed observable
with a specified cemetery value and a survival-conditioned observable are
different quantities.
:::

:::{prf:theorem} Finite-step observable equation and fluctuation covariance
:label: thm-algorithmic-observable-increment

Assume $A(X_n)$ and $A(X_{n+1})$ are square integrable. Define

$$
b_A(X)=\mathsf P_{N,h}A(X)-A(X),\qquad
\Gamma_A(X)=\mathsf P_{N,h}(AA^\top)(X)
 -(\mathsf P_{N,h}A(X))(\mathsf P_{N,h}A(X))^\top.
$$

Then

$$
\boxed{
A(X_{n+1})-A(X_n)=b_A(X_n)+\eta_{n+1},\qquad
\mathbb E[\eta_{n+1}\mid\mathscr F_n]=0,\quad
\mathbb E[\eta_{n+1}\eta_{n+1}^\top\mid\mathscr F_n]
=\Gamma_A(X_n).
}
$$

In particular, for fixed spatial probes $z_1,\ldots,z_q$, take $A(X)$ to
be the vector of independent components of the specified metric
$g_X(z_\ell)=\mathcal G(X;z_\ell)$. This gives an exact stochastic metric
equation at finite $N,h$, whenever that observable is defined and square
integrable. For the implemented global quadratic research fixture,
{prf:ref}`lem-algorithmic-quadratic-finite-step-moments` proves this
integrability directly. It does not assert that $b_A$ is a function of
$g$ alone.
:::

:::{prf:proof}
Set $\eta_{n+1}=A(X_{n+1})-\mathsf P_{N,h}A(X_n)$. The conditional
transition law gives its zero conditional mean. Expanding its conditional
outer product yields $\Gamma_A$. Adding and subtracting
$\mathsf P_{N,h}A(X_n)$ proves the increment identity. Square integrability
makes these expressions finite. The metric application is the same
calculation for the stated vector-valued observable.
:::

:::{prf:proposition} Operator decomposition without a closure assumption
:label: prop-algorithmic-stage-telescoping

Let $X_n^{(0)},\ldots,X_n^{(L)}$ be the actual extended intermediate
states, with $X_n^{(0)}=X_n$ and $X_n^{(L)}=X_{n+1}$. A skipped operation
is an identity stage. Define $\Delta_\ell A=A(X_n^{(\ell)})-
A(X_n^{(\ell-1)})$. Then

$$
\Delta A=\sum_{\ell=1}^L\Delta_\ell A,\qquad
b_A(X_n)=\sum_{\ell=1}^L
 \mathbb E[\Delta_\ell A\mid X_n],
$$

and

$$
\Gamma_A(X_n)=
\sum_{\ell,k=1}^L
\operatorname{Cov}(\Delta_\ell A,\Delta_k A\mid X_n).
$$

If $b_{A,\ell}(X_n^{(\ell-1)})$ is the stage drift conditional on
everything entering stage $\ell$, its contribution to the full-step
drift is $\mathbb E[b_{A,\ell}(X_n^{(\ell-1)})\mid X_n]$.
:::

:::{prf:proof}
Successive differences cancel in the sum. Taking conditional expectation
proves the drift formula; applying the tower property proves its
stage-conditional form. Expanding the covariance of the sum proves the
last identity. Distinct realized stage increments generally have nonzero
cross covariance, even when the newly drawn innovations are independent.
:::

:::{prf:proposition} Fixed-probe and material metric increments
:label: prop-algorithmic-material-metric

Express all tensors in a fixed reference chart. For a tracked old probe
$z$ and its specified new location $z'$, the exact identity is

$$
g_{X'}(z')-g_X(z)
=\underbrace{g_{X'}(z)-g_X(z)}_{\text{field change at a fixed probe}}
 +\underbrace{g_{X'}(z')-g_{X'}(z)}_{\text{spatial sampling change}}.
$$

For a clone replacing a slot at $z_i$ by a source at $y_j$, and ending at
$z_i'$, a useful refinement is

$$
\begin{aligned}
g_{X'}(z_i')-g_X(z_i)
={}&[g_{X'}(y_j)-g_X(y_j)]
  +[g_{X'}(z_i')-g_{X'}(y_j)]\\
 &+[g_X(y_j)-g_X(z_i)].
\end{aligned}
$$

The three terms respectively measure field evolution at the source
probe, subsequent motion, and the slot replacement. A historical source
has both a source frame and a location. Comparing with its historical
metric requires recording that frame and adding the corresponding metric
difference explicitly.

For a reconstructed material map $\Phi$, the geometrically transported
comparison is $\Phi^*g_{X'}-g_X$, where

$$
(\Phi^*g_{X'})(z)=D\Phi(z)^\top
                  g_{X'}(\Phi(z))D\Phi(z).
$$

A discrete clone map does not automatically provide this differentiable
map or its Jacobian.
:::

:::{prf:proof}
The first two identities follow by adding and subtracting the indicated
metric values. The pullback expression is the tensor transformation law.
It includes deformation of the coordinate frame, which simple component
evaluation along a path does not include.
:::

:::{div} feynman-prose
These equations tell us exactly which measurements to compare. They do
not yet tell us that the metric remembers enough to predict its own
future. Two populations can share the same metric and carry different
velocities or different historical donors. Their next metric increments
can therefore differ. A proposed equation involving only the metric must
survive that test.

There is a similar trap in simulation. If we estimate the mean change
from a collection of replicas and subtract that mean from those same
replicas, the residual mean is zero by construction. We have tested an
arithmetic identity. A prediction deserves a fresh collection of
replicas.
:::

:::{prf:proposition} Independent-replica drift test
:label: prop-algorithmic-independent-replicas

Fix $X$ and draw two conditionally independent groups of increments
$Y=A(X')-A(X)$, of sizes $R,S>1$. Let their sample means be
$\widehat b_{\mathrm{pred}}$ and $\widehat b_{\mathrm{test}}$. Then

$$
\mathbb E[\widehat b_{\mathrm{test}}-\widehat b_{\mathrm{pred}}\mid X]=0,
\qquad
\operatorname{Cov}(\widehat b_{\mathrm{test}}-\widehat b_{\mathrm{pred}}
 \mid X)
=\left(\frac1R+\frac1S\right)\Gamma_A(X).
$$

Each group's sample covariance with denominator one less than its size
is unbiased for $\Gamma_A(X)$. For a fixed analytic prediction $b_*(X)$,
the test mean instead has bias $b_A(X)-b_*(X)$ and covariance
$\Gamma_A(X)/S$.
:::

:::{prf:proof}
Conditional independence removes cross covariance between samples and
between groups. Summing the individual means and covariances gives the
displayed formulas. Expanding
$\sum_r(Y_r-\overline Y)(Y_r-\overline Y)^\top$ gives expected value
$(R-1)\Gamma_A$, proving sample covariance unbiasedness.
:::

:::{prf:remark} What a simulation can establish
:label: rem-algorithmic-validation-law

Use replicas or independent trajectories as the sampling units. Walkers
sharing a population are not independent replicas. Finite variance gives
the identities above; it does not make a finite-sample Gaussian confidence
interval exact. Exact reference distributions, proved tail bounds, or a
justified asymptotic approximation determine the uncertainty procedure.
An estimated drift and an independently checked analytic drift must be
reported separately. A reduced predictor using only density, current, or
metric is tested for bias against the full-state conditional law.

The formulas concern the declared innovation law and real arithmetic.
Finite-precision evaluation and pseudo-random simulation are numerical
approximations whose discrepancy is checked against independent analytic
or enumerable reference cases. An unchanged-seed replay separately checks
that instrumentation has not changed the executed dynamics.
:::

(sec-algorithmic-quadratic-moments)=
### Finite-step integrability of the quadratic research fixture

:::{div} feynman-prose
The square-integrability condition in the observable equation can be
checked directly for the quadratic research runs. Their positions are
unbounded, and we leave them that way. The useful bound comes from the
programmed metric floor: however large the fitness Hessian becomes, the
noise factor cannot exceed a fixed amplitude.

Cloning can amplify the population norm by copying a large row many
times, but at fixed population size that amplification is bounded.
The remaining quadratic-force updates are linear. Combining these facts
controls every moment at every finite step, including the moments needed
to measure the metric.
:::

:::{prf:lemma} Finite-step moments for the implemented global quadratic fixture
:label: lem-algorithmic-quadratic-finite-step-moments

Consider the global-normalization research configuration in
`algorithmic-gas/crates/benchmarks/examples/physics_research.rs`:
finite $N\geq2$ and dimension $d$, the minimizing `Quadratic` objective
$U(x)=|x|^2/2$, unbounded boundary policy, current-frame single-source
literal cloning, the identity clone transform (no jitter or restitution),
and the implemented BAOAB integrator. Let its fixed parameters satisfy
$h>0$, $\gamma>0$, $T\geq0$, and $\epsilon_\Sigma>0$. Its metric provider
uses global standardization with positive variance regularizers,
regularized Euclidean diversity with positive distance floor, and logistic
maps with positive amplitude and positive output floor. Fitness powers
are the configured finite constants. The innovations are independent
standard Gaussian vectors, and the initial population is a fixed finite
realization of the configured initialization.

In the simulation's dimensionless units, write
$Y_n=(x_1,\ldots,x_N,v_1,\ldots,v_N)$ and let
$\xi_{n+1}\in\mathbb R^{Nd}$ collect the $O$-stage innovations. With

$$
c=e^{-\gamma h},\qquad
s^2=\frac{1-e^{-2\gamma h}}{2\gamma},\qquad
\beta=\sqrt{\frac{2\gamma T}{\epsilon_\Sigma}},\qquad
a=\sqrt N(1+h/2)^4,\quad b=(1+h/2)^2s\beta,
$$

the executed update obeys the pathwise bound

$$
\|Y_{n+1}\|\leq a\|Y_n\|+b\|\xi_{n+1}\|.
$$

For every finite $n$ and $p\geq1$,

$$
\left(\mathbb E\|Y_n\|^p\right)^{1/p}
\leq a^n\|Y_0\|+
b\left(\mathbb E\|\xi_1\|^p\right)^{1/p}
 \sum_{j=0}^{n-1}a^j<\infty.
$$

All BAOAB intermediate coordinates have finite moments of every order.
For the programmed fitness query field with frozen coordinate context
$Y_{\mathcal C}$, there is a finite constant $C$ such that

$$
\left\|\nabla_z^2F_i^{\mathrm{probe}}(z;\mathcal C)\right\|_F
\leq C(1+\|z\|+\|Y_{\mathcal C}\|)^{12}.
$$

The constant depends on the fixed population size, dimension, objective,
and configured regularizers, maps, and powers, and is uniform over the
finite companion assignments. Consequently the clipped metric at fixed
probes and at the executed $O$ queries has finite moments of every order.
In particular, these metric readouts satisfy the square-integrability
condition of {prf:ref}`thm-algorithmic-observable-increment`.
:::

:::{prf:proof}
**1. The noise bound is a consequence of the configured clipping rule.**
For the symmetric fitness Hessian $H$,

$$
g=\epsilon_\Sigma I+H_+\succeq\epsilon_\Sigma I,\qquad
B=\sqrt{2\gamma T}\,g^{-1/2},\qquad
\|B\|_{\mathrm{op}}\leq\beta.
$$

All query expressions are finite and smooth at finite coordinate inputs:
the distance square root has a positive floor, the standardization
denominator is bounded below by its positive regularizer, and a logistic
map plus its positive floor takes values in a compact positive interval.
Thus finite inputs give a finite Hessian and the displayed factor.
For the block-diagonal population factor
$\mathcal B=\operatorname{diag}(B_1,\ldots,B_N)$ the same operator bound
$\|\mathcal B\|_{\mathrm{op}}\leq\beta$ holds. The factors may depend on
all the recorded population data.

**2. Bound each actual update.**
Let $Y_n^{\mathrm{cl}}$ be the literal-clone output. Each of its $N$ rows
is one row of the frozen current population, whether retained or copied.
Consequently

$$
\|Y_n^{\mathrm{cl}}\|^2
\leq N\max_i(|x_i|^2+|v_i|^2)
\leq N\|Y_n\|^2.
$$

This estimate holds for every realized donor and acceptance outcome.
For the configured minimizing quadratic objective, the gradient provider
returns $x$ exactly. With $q=h/2$, each kick and displacement therefore
has the respective block form

$$
\mathsf K=\begin{pmatrix}I&0\\-qI&I\end{pmatrix},\qquad
\mathsf A=\begin{pmatrix}I&qI\\0&I\end{pmatrix},
\qquad
\|\mathsf K\|_{\mathrm{op}},\|\mathsf A\|_{\mathrm{op}}\leq1+q.
$$

The bound follows by writing each matrix as the identity plus a matrix
of norm $q$. The $O$ stage is

$$
Y^+=\mathsf OY^-+
\begin{pmatrix}0\\s\mathcal B\xi_{n+1}\end{pmatrix},
\qquad
\mathsf O=\operatorname{diag}(I,cI),\quad
\|\mathsf O\|_{\mathrm{op}}\leq1.
$$

Applying the remaining displacement and kick to this noise term gives

$$
Y_{n+1}=\mathsf K\mathsf A\mathsf O\mathsf A\mathsf K
            Y_n^{\mathrm{cl}}
 +\mathsf K\mathsf A
       \begin{pmatrix}0\\s\mathcal B\xi_{n+1}\end{pmatrix}.
$$

The matrix bounds and the clone bound prove the stated inequality.
The unbounded boundary policy and the identity clone transform add no
coordinate change. Finite inputs and Gaussian innovations remain finite
at every finite step; the quadratic rewards and positive fitness maps
therefore keep this mathematical fixture eligible.

**3. Iterate the moment bound.**
Iterating the pathwise inequality gives

$$
\|Y_n\|\leq a^n\|Y_0\|
 +b\sum_{j=1}^n a^{n-j}\|\xi_j\|.
$$

Minkowski's inequality yields the displayed $L^p$ bound. Gaussian
moments are finite, as follows by integrating a polynomial against
$e^{-\|\xi\|^2/2}$. The same stage inequalities give every intermediate
coordinate bound. No independence of the state-dependent factor and the
previous population was used. The constants can grow with $n$; the claim
is at each finite step.

**4. Bound the differentiated fitness formula.**
Put $R=1+\|z\|+\|Y_{\mathcal C}\|\geq1$, and let $D$ denote a query
derivative. The quadratic rewards and their derivatives through order
two are bounded by $CR^2$. Each regularized diversity measurement is

$$
d(z,y)=\sqrt{|z-y|^2+\delta^2},\qquad \delta>0.
$$

It satisfies $d\leq CR$, $\|Dd\|\leq1$, and
$\|D^2d\|_{\mathrm{op}}\leq\delta^{-1}$. Frozen rows have zero query
derivatives. Thus, for either channel, its measurements and derivatives
through order two are bounded by $CR^2$.

For the actual global mean and variance
$\mu=N^{-1}\sum_jm_j$ and $V=N^{-1}\sum_jm_j^2-\mu^2$, the product rule
therefore bounds $\mu,D\mu,D^2\mu$ by $CR^2$, and
$V,DV,D^2V$ by $CR^4$. Write
$t=(V+\sigma_{\min}^2)^{-1/2}$. Since $V\geq0$,

$$
|t|\leq\sigma_{\min}^{-1},\quad
Dt=-\tfrac12(V+\sigma_{\min}^2)^{-3/2}DV,\quad
D^2t=\tfrac34(V+\sigma_{\min}^2)^{-5/2}DV\otimes DV
      -\tfrac12(V+\sigma_{\min}^2)^{-3/2}D^2V.
$$

Hence $\|Dt\|\leq CR^4$ and $\|D^2t\|\leq CR^8$. For the standardized
target $Z=(m_i-\mu)t$, the product rule gives
$\|DZ\|\leq CR^6$ and $\|D^2Z\|\leq CR^{10}$.
Logistic derivatives through order two are bounded on the real line.
The configured powers have bounded derivatives on the positive range
of their logistic maps. Applying the chain and product rules to the
two-channel fitness therefore gives
$\|D^2F_i^{\mathrm{probe}}\|_F\leq CR^{12}$.

Finally, spectral clipping gives

$$
\|g_{\mathcal C}(z)\|_F
\leq\epsilon_\Sigma\sqrt d+
       \|H(z;\mathcal C)\|_F
\leq C'(1+\|z\|+\|Y_{\mathcal C}\|)^{12}.
$$

The context coordinates are retained finite-stage population coordinates.
A fixed probe is deterministic; an executed $O$ probe is an intermediate
coordinate already controlled in step 3. Their $24$th moments give metric
square integrability. Their moments of every higher order give the
remaining asserted metric moments. Finite sums of these readouts, including
the recorded population-average $O$ metric, inherit the same conclusion.
:::

(sec-algorithmic-transition-response)=
### Frozen query inputs and the response of the transition

:::{div} feynman-prose
“Conditional fitness” names the inputs held fixed while the configured
metric provider differentiates a query. Keep the recorded donor coordinates
and the other measured rows fixed, vary the query, and recompute its
fitness through the configured normalization. The resulting Hessian is
the one used by that provider's metric rule.

Now move an actual walker in the input population. It may also be another
walker's donor, so other measured distances change. Local weights and the
probabilities of future donor and clone choices can change too. To
differentiate the expected next measurement, follow both changes: what
each possible update produces and how often the algorithm selects it.
:::

:::{prf:definition} Query derivative and population perturbation
:label: def-algorithmic-query-versus-population

Let $\mathcal C$ be the recorded `FrozenFitnessContext`. The configured
query field $F_i^{\mathrm{probe}}(z;\mathcal C)$ holds its donor source
coordinates and other rows' measurements fixed while replacing target
row $i$ by the query measurements. It differentiates the resulting
standardization, positive maps, and powers. For example, global reward
standardization uses

$$
\bar r(z)=\frac{r_i(z)+\sum_{j\ne i}r_j^0}{k}
$$

over the $k$ eligible rows, with its variance recomputed from these same
values. Local standardization also differentiates the target's configured
localization weights. In the clipped branch the programmed metric is
$g_{\mathcal C}(z)=\epsilon_\Sigma I+
[\nabla_z^2F_i^{\mathrm{probe}}(z;\mathcal C)]_+$.

A population perturbation instead specifies $X_\theta$ and reevaluates
the transition from that extended state. Even with companion identities
fixed, its branch observable $A_\theta(c)$ differentiates every affected
measurement and normalization. For example, if $c_j=i$, moving current
walker $i$ changes the sampled distance in row $j$. This dependence is
absent from the query field whose other measurements remain fixed.
These are two explicitly different arguments of the programmed formulas.
The sampled/expected distinction in {prf:ref}`def-c3-fitness-laws`
additionally distinguishes holding a discrete assignment fixed from
averaging over its law.
:::

:::{prf:proposition} Derivative of the complete finite-choice transition
:label: prop-algorithmic-transition-derivative

Let $\theta$ vary in an open finite-dimensional parameter region. Let
$\mathcal C_{\mathrm{fin}}$ be a fixed finite list of complete discrete
choice histories, including zero-probability histories. Write
$p_\theta(c)$ for the actual joint probability of history $c$ and
$A_\theta(c)$ for its resulting observable after all deterministic
dependencies on $\theta$ have been evaluated. Where these functions are
differentiable,

$$
\partial_a\sum_c p_\theta(c)A_\theta(c)
=\sum_c\left[p_\theta(c)\partial_a A_\theta(c)
             +A_\theta(c)\partial_a p_\theta(c)\right].
$$

If the positive-probability support is constant in a neighborhood and
$\ell_\theta(c)=\log p_\theta(c)$ on that support, this becomes

$$
\boxed{
\partial_a\mathbb E_\theta[A_\theta]
=\mathbb E_\theta[
 \partial_a A_\theta+A_\theta\,\partial_a\ell_\theta].
}
$$

For twice differentiable terms, the Hessian is

$$
\partial_{ab}\mathbb E_\theta[A_\theta]
=\mathbb E_\theta\!\left[
\partial_{ab}A_\theta
+(\partial_a A_\theta)(\partial_b\ell_\theta)
+(\partial_b A_\theta)(\partial_a\ell_\theta)
+A_\theta\bigl(\partial_{ab}\ell_\theta+
        \partial_a\ell_\theta\,\partial_b\ell_\theta\bigr)\right].
$$

Use the actual joint law, including mutual-pair constraints and sequential
choices. Its factorization into conditional choice probabilities, rather
than a product of independent row marginals, follows the operator
composition in {prf:ref}`thm-cloning-operator-composition`. Where those
conditional probabilities are positive, the joint log-probability
derivative is the sum of their log-probability derivatives, each evaluated
along that history with all its state dependencies retained.
:::

:::{prf:proof}
Differentiate each term of the finite sum by the product rule. On the
fixed positive support, substitute
$\partial_a p=p\,\partial_a\ell$. Differentiate once more and use
$\partial_{ab}p=p(\partial_{ab}\ell+
\partial_a\ell\,\partial_b\ell)$ to obtain the Hessian formula.
The chain rule for joint probabilities gives
$p_\theta(c)=\prod_jp_\theta(c_j\mid c_1,\ldots,c_{j-1})$; taking the
logarithm gives the stated sum. For independent companion rows,
{prf:ref}`lem-c3-joint-companion-law` supplies the corresponding
quantitative derivative bounds under its stated hypotheses.
:::

:::{prf:proposition} Including continuous innovations
:label: prop-algorithmic-continuous-response

For continuous innovations $u$, write the full transition expectation as
$J(\theta)=\sum_c\int A_\theta(c,u)r_\theta(c,u)\,\nu(du)$ against a
fixed reference measure. Fix a parameter coordinate $a$ and a neighborhood
of the evaluation point. Suppose $A_\theta,r_\theta$ are continuously
differentiable in that coordinate for $\nu$-almost every $u$, $J$ is
absolutely integrable at the evaluation point, and

$$
\sup_\theta\left[
r_\theta(c,u)\|\partial_a A_\theta(c,u)\|
+\|A_\theta(c,u)\|\,|\partial_a r_\theta(c,u)|
\right]\leq M_c(u),\qquad
\sum_c\int M_c\,d\nu<\infty.
$$

Then

$$
\partial_aJ
=\sum_c\int[
r_\theta\,\partial_a A_\theta+
A_\theta\,\partial_a r_\theta]\,d\nu.
$$

On a common positive support this is the same score formula with
$\partial_a\log r_\theta$. If innovations are represented by a
parameter-independent base law, its density derivative is zero; their
parameter-dependent transformation remains in $\partial_a A_\theta$.

**Proof.** The product rule gives the derivative of the integrand.
The fundamental theorem of calculus bounds its difference quotient by
$M_c$. Dominated convergence therefore passes the derivative through the
integral and the finite sum. Positivity permits division by $r_\theta$.
:::

:::{prf:remark} Applying the response formula to the metric equation
:label: rem-algorithmic-complete-response

For a perturbation of the exact drift in
{prf:ref}`thm-algorithmic-observable-increment`, differentiate the entire
next-state expectation by the preceding propositions and subtract the
derivative of the initial readout $A_\theta(X_\theta)$. For the programmed
metric observable, this differentiates both its query formula and the
changes to its recorded context produced by the transition.

The finite-choice product formula retains probability changes even
when a history has zero probability; the score form uses the stated
support condition. At a clipping, acceptance, eligibility, or candidate-set
change, apply a derivative formula only where its derivatives exist;
the finite-step expectation and finite differences use the executed
transition wherever the chosen observable is defined at the compared
states. A frozen-choice derivative alone
omits the law term whenever those probabilities respond to $\theta$.
:::

(sec-algorithmic-balance-laws)=
## Energy, Momentum, and Density from Actual Updates

:::{div} feynman-prose
For the mechanical budget, begin with the velocity that actually enters
an operation. Squaring the new velocity tells us the exact energy change.
There is no need to guess a pressure or a temperature first.

The order of operations is part of the physics of this algorithm. The
Rust integrator checks boundaries after each kick, each displacement, and
the thermostat. A reflected velocity or a newly killed walker contributes
at that boundary operation. Combining the thermostat and the boundary
into one measurement hides their separate contributions. It is still a
valid combined increment, but it cannot be compared directly with the
unconstrained thermostat formula.
:::

:::{prf:definition} Mechanical observables and stage convention
:label: def-algorithmic-mechanical-observables

For unit particle masses, positions $x_i$, velocities $v_i$, and
eligibility indicators $a_i$, define

$$
K(X)=\frac12\sum_i a_i|v_i|^2,\qquad
p(X)=\sum_i a_i v_i,\qquad
E(X)=K(X)+U(X).
$$

Here $U$ is an explicitly specified mechanical energy observable,
possibly population-dependent. Raw reward is not automatically $-U$.
With position units $L$ and integration-time units $\tau$, $v$ has units
$L/\tau$, $K$ has unit-mass units $L^2/\tau^2$, friction has units
$\tau^{-1}$, and a velocity noise factor $B$ has units
$L/\tau^{3/2}$. The algorithm may choose dimensionless reference units.

Use the executed stage ordering: measurement and donor selection; literal
cloning; clone transforms; reconciliation, reward-validity, and boundary
updates; kinetics; final reconciliation and validity checks. BAOAB
kinetics executes $B_1,A_1,O,A_2,B_2$, with boundary checks between these
operations. Eligibility is recomputed between stages. For a pure
thermostat or kick formula, use its input and output before any subsequent
boundary or domain transformation; those transformations receive separate
increments when separately observed.
:::

:::{prf:theorem} Conditional energy and momentum of the Rust thermostat
:label: thm-algorithmic-thermostat-moments

For one eligible walker, condition on everything entering the $O$ stage,
including $v$ and $B\in\mathbb R^{d\times r}$. Let $h>0$, $\gamma\geq0$,
and suppose its innovation
satisfies $\mathbb E\xi=0$ and $\mathbb E\xi\xi^\top=I_r$. The update in
`algorithmic-gas/crates/algorithmic-gas/src/kinetic.rs` is

$$
v^+=c v+sB\xi,\qquad
c=e^{-\gamma h},\qquad
s^2=\begin{cases}
\dfrac{1-e^{-2\gamma h}}{2\gamma},&\gamma>0,\\[4pt]
h,&\gamma=0.
\end{cases}
$$

Put $Q=BB^\top$ and $K_v=|v|^2/2$. Then, before boundary handling,

$$
\boxed{\mathbb E[\Delta K_v\mid v,B]
=(c^2-1)K_v+\frac{s^2}{2}\operatorname{tr}Q,}
\qquad
\mathbb E[\Delta v\mid v,B]=(c-1)v,\quad
\operatorname{Cov}(\Delta v\mid v,B)=s^2Q.
$$

For Gaussian innovations,

$$
\boxed{\operatorname{Var}(\Delta K_v\mid v,B)
=c^2s^2 v^\top Qv+\frac{s^4}{2}\operatorname{tr}(Q^2).}
$$

For independent, symmetric unit-variance components with common fourth
moment $\mu_4$, the variance instead is

$$
c^2s^2 v^\top Qv+\frac{s^4}{2}\operatorname{tr}(Q^2)
+\frac{s^4}{4}(\mu_4-3)
 \sum_{\alpha=1}^r[(B^\top B)_{\alpha\alpha}]^2.
$$

The built-in standardized uniform law has $\mu_4=9/5$; its correction is
therefore negative. The mean formula does not require Gaussian noise.
:::

:::{prf:proof}
The integrator applies $c=e^{-\gamma h}$ and the time factor
$s^2=\int_0^h e^{-2\gamma t}\,dt$ to an unscaled noise-provider sample.
Evaluating this integral gives both branches and the continuous limit at
$\gamma=0$. Expanding the kinetic energy gives

$$
\Delta K_v=(c^2-1)K_v+
cs\,v^\top B\xi+\frac{s^2}{2}\xi^\top B^\top B\xi.
$$

The linear term has zero mean and the quadratic term has mean
$s^2\operatorname{tr}(B^\top B)/2$. The velocity formulas follow
directly. Set $M=B^\top B$. For independent symmetric standardized
components, terms of odd degree have zero expectation. Thus the linear
term and the centered quadratic term have zero covariance. Expanding
$\mathbb E(\xi^\top M\xi)^2$ by matching indices gives

$$
\operatorname{Var}(\xi^\top M\xi)
=2\operatorname{tr}(M^2)+(\mu_4-3)\sum_\alpha M_{\alpha\alpha}^2.
$$

The linear variance is $c^2s^2v^\top Qv$ and
$\operatorname{tr}(M^2)=\operatorname{tr}(Q^2)$, which proves both
variance formulas. Integrating $x^4$ under the uniform density on
$[-\sqrt3,\sqrt3]$ gives $9/5$. Singular and rectangular $B$ cause no
difficulty because no inverse was used.
:::

:::{prf:corollary} Geometry in the thermostat budget
:label: cor-algorithmic-geometric-heating

If a provider constructs $B=\sigma g^{-1/2}$ with $g$ positive definite,
then the conditional noise-energy injection for that walker is

$$
\frac{s^2\sigma^2}{2}\operatorname{tr}(g^{-1}).
$$

For conditionally independent innovations across walkers, total
conditional energy means and variances sum over the eligible rows.
Without that independence, include the cross covariances. If $B$ or the
eligible set is itself random before the stage, the full-step prediction
uses the tower property and total covariance over those earlier events.

**Proof.** Substitute $Q=\sigma^2g^{-1}$ in
{prf:ref}`thm-algorithmic-thermostat-moments` and apply conditional
independence only for the stated variance sum.
:::

:::{div} feynman-prose
Two thermostats can inject exactly the same mean energy and have different
fluctuations. Gaussian and standardized uniform innovations demonstrate
this directly: they have the same covariance, but their fourth moments
differ. Looking only at the mean would miss a wrong innovation law in a
simulation.

The noise factor also tells us where geometry enters. If a direction has
smaller metric length and the provider uses the inverse metric for its
noise covariance, that direction receives larger velocity kicks. The
trace sums their energy contributions. This is a consequence of squaring
the implemented update, with no temperature assigned. Interpreting it as
heat exchanged with an equilibrium reservoir requires further information
about the complete transition law.
:::

:::{prf:proposition} Exact kicks and displacements
:label: prop-algorithmic-kick-work

For any realized velocity increment $\delta v_i$ on a fixed eligible
set,

$$
\Delta p=\sum_i a_i\delta v_i,\qquad
\Delta K=\sum_i a_i\left(v_i\cdot\delta v_i+
                                      \frac12|\delta v_i|^2\right).
$$

A Rust BAOAB kick uses the executed acceleration
$f_i^{\mathrm{tot}}=-\mathcal D_i(X)+f_i^{\mathrm{visc}}(X)$, where
$\mathcal D_i$ is the vector returned by the configured gradient provider
and $f_i^{\mathrm{visc}}$ is the configured QFT viscosity contribution,
zero when disabled. The recorded `total_force` stores this acceleration.
Thus $\delta v_i=(h/2)f_i^{\mathrm{tot}}$, and its energy contribution per
eligible walker is

$$
\frac h2 v_i\cdot f_i^{\mathrm{tot}}
 +\frac{h^2}{8}|f_i^{\mathrm{tot}}|^2.
$$

When viscosity is disabled, this specializes to
$-(h/2)v_i\cdot\mathcal D_i+(h^2/8)|\mathcal D_i|^2$.
When both accelerations are present, the squared total includes their
cross term; separate squared-force terms alone do not give the kick work.

When the provider is established to be the gradient of a scalar potential,
one may write $\mathcal D_i=\nabla_iU_{\mathrm{pot}}$. The increment
identity does not require that additional identification.
The second kick recomputes the provider at the post-$A_2$ state. A displacement
with unchanged velocities has zero kinetic increment and mechanical
potential increment $U(X^+)-U(X^-)$. Any relation between this $U$ and
$U_{\mathrm{pot}}$ is part of the supplied energy model. Changes in
eligibility and boundary transformations are accounted for by direct
differences of the full observables.

**Proof.** Expand $|v_i+\delta v_i|^2-|v_i|^2$ and substitute the kick.
The displacement statement follows from the definition of $E$. No
continuous-time work approximation was made.
:::

:::{prf:proposition} Literal-clone transfers and restitution
:label: prop-algorithmic-clone-balances

Condition on a frozen source pool and realized clone decisions. Let
$C_i$ indicate that slot $i$ is replaced, and let $(y_{J_i},w_{J_i},
\bar a_{J_i})$ be its source position, velocity, and eligibility.
Literal cloning has the exact increments

$$
\Delta p_{\mathrm{clone}}=
\sum_i C_i(\bar a_{J_i}w_{J_i}-a_i v_i),\qquad
\Delta K_{\mathrm{clone}}=
\frac12\sum_i C_i(\bar a_{J_i}|w_{J_i}|^2-a_i|v_i|^2).
$$

The source may be historical. Simultaneous recipients read the frozen
pool, including repeated donors, rather than one another's updated slots.
The potential contribution is the exact difference of the specified $U$.

For an activated disjoint current mutual pair with pre-clone velocities
$v_i,v_j$, the built-in restitution transform with coefficient
$\alpha\in[0,1]$ sets

$$
\bar v=\frac{v_i+v_j}{2},\qquad
v_i^*=\bar v+\frac\alpha2(v_i-v_j),\quad
v_j^*=\bar v-\frac\alpha2(v_i-v_j).
$$

Relative to that pair's pre-clone state,

$$
\Delta p_{\mathrm{pair}}=0,\qquad
\Delta K_{\mathrm{pair}}=
-\frac{1-\alpha^2}{4}|v_i-v_j|^2.
$$

The code activates this transform when either pair member accepted a
clone. It overwrites both velocities using their pre-clone values.
Consequently its transform-stage increment is

$$
\Delta K_{\mathrm{transform,pair}}
=-\frac{1-\alpha^2}{4}|v_i-v_j|^2
  -\Delta K_{\mathrm{literal,pair}},
$$

with the analogous subtraction for momentum. It is not an additional
pair loss applied to the literal-clone velocities.
:::

:::{prf:proof}
Literal cloning replaces exactly the indicated row observables, proving
the first formulas by subtraction. For restitution, pair momentum equals
$2\bar v$ both before and after the transform, while pair kinetic energy
is $|\bar v|^2+|v_i-v_j|^2/4$ before and
$|\bar v|^2+\alpha^2|v_i-v_j|^2/4$ after. Finally, subtract the already
recorded literal-clone increment to obtain the increment from that
intermediate state.
:::

:::{div} feynman-prose
Follow one cloned slot carefully. First it takes a donor's velocity.
Then restitution may overwrite that velocity using the original pair.
The energy change from the original pair to the final pair is the
dissipative loss we just calculated. The change from the intermediate
literal clone to the final pair can have another sign. These statements
agree because they compare different endpoints.

For a local balance, we must also record where the replaced walker and
its donor were. A clone can move momentum across a large distance in one
update. Replacing this transfer with a local pressure before deriving a
spatial approximation would erase part of the algorithm. Test functions
let us retain the full transfer while choosing the spatial resolution of
the measurement.
:::

:::{prf:proposition} Weak density and local momentum equations
:label: prop-algorithmic-weak-density

For a bounded test function $\varphi$ on the walker state, define the
fixed-capacity empirical measure

$$
\mu_X=\frac1N\sum_i a_i\delta_{z_i},\qquad A_\varphi(X)=\mu_X\varphi.
$$

Then the exact finite-step weak equation is

$$
\mu_{n+1}\varphi-\mu_n\varphi
=(\mathsf P_{N,h}A_\varphi-A_\varphi)(X_n)+\eta_{\varphi,n+1}.
$$

Condition on the source pool and fitness values, but average over clone
source selection and acceptance. If
$q_{ij}=\Pr(C_i=1,J_i=j\mid\text{this context})$, literal cloning gives

$$
\mathbb E[\Delta_{\mathrm{clone}}\mu\varphi\mid\text{context}]
=\frac1N\sum_{i,j}q_{ij}
 [\bar a_j\varphi(y_j,w_j)-a_i\varphi(x_i,v_i)].
$$

For the default decision law on an eligible row, with donor marginal
$\pi_{ij}$ conditional on the same frozen fitness context,

$$
q_{ij}=\pi_{ij}
\min\!\left(1,\max\!\left(0,
\frac{f_j-f_i}{(f_i+\epsilon)\,\zeta}\right)\right),
$$

where $\epsilon$ and $\zeta$ are the configured acceptance regularizer and
saturation. An ineligible row is instead revived uniformly from eligible
current-frame sources. Mutual donor constraints affect joint laws and
covariances; they do not change this marginal expectation formula.

For a spatial test function $\psi$, the local momentum observable is
$p_\psi(X)=\sum_i a_i v_i\psi(x_i)$. Each row satisfies the exact
decomposition

$$
\Delta(a v\psi)
=a'\psi(x')\,\Delta v+
 a'v[\psi(x')-\psi(x)]+(a'-a)v\psi(x).
$$

These are the impulse, transport, and eligibility contributions under
this specified discrete convention. For a pure literal clone, the local
momentum transfer is

$$
\sum_i C_i[
\bar a_{J_i}w_{J_i}\psi(y_{J_i})-a_iv_i\psi(x_i)].
$$

It generally connects distinct locations and need not conserve total
momentum. Historical replacement need not be a current-time pair
interaction.
:::

:::{prf:proof}
Apply {prf:ref}`thm-algorithmic-observable-increment` to $A_\varphi$.
Subtract each replaced atom and average its event indicator to obtain
the clone formula. The default acceptance probability is precisely the
clipped ratio evaluated by the Rust decision operator; multiply it by
the conditional donor marginal. Expand the three local-momentum terms:
the intermediate products cancel, leaving
$a'v'\psi(x')-av\psi(x)$. The literal-clone specialization follows by
replacement.
:::

:::{prf:remark} Stress, stationarity, and closure
:label: rem-algorithmic-stress-closure

The tested weak momentum law determines impulses and transport.
Representing nonlocal transfers by a stress density additionally requires
a specified spatial deposition convention. A smooth segment permits
$\psi(y)-\psi(x)=\int_0^1\nabla\psi(x+t(y-x))\cdot(y-x)\,dt$;
periodic domains and boundaries require the actual chosen path.
This representation cannot silently discard sources or impose an
isotropic perfect-fluid tensor.

If a transition-closed state representation has an invariant probability
law $\nu$ and $A$ is integrable, then $\int b_A\,d\nu=0$. Pure bookkeeping
counters may be removed only when doing so leaves the conditional
transition law determined by the retained state; a changing schedule
cannot be discarded in this way. This says that the
total stationary mean increment vanishes. The separate stage budgets
can remain nonzero: thermostat injection can balance dissipative
restitution or boundary losses. A quasi-stationary law of a killed chain
is instead conditioned on survival and obeys its corresponding
normalized law; it does not automatically satisfy this conservative
stationarity identity.

The displayed empirical measure uses fixed $N$; its mass changes with
eligibility. Renormalizing by the number of survivors introduces a
random denominator and yields another observable, not the same weak
equation. Likewise, dividing a fixed-step clone increment by $h$ does
not establish a finite continuous-time generator when clone
probabilities remain of order one. A macroscopic closure must be
derived from the transition, with its limit and error controlled.

A comparison between an Einstein tensor and stress must compute the
stress from these independent mechanical observables and specify any
spacetime construction separately. Solving the proposed geometric
equation for its own source cannot validate that equation.
:::

(sec-ig-free-energy)=
## Density Fluctuations and a Correlation Energy

:::{div} feynman-prose
A sampling rate measures how unlikely a density fluctuation is. An interaction energy is an additional specification. For independent samples the rate can be calculated from a multinomial probability. For the interacting swarm, the established entropy and LSI estimates control fluctuations directly. We retain those estimates and specify the comparison energy separately.
:::

### Independent sampling and interacting fluctuations

:::{prf:lemma} Independent-sampling rate and interacting concentration
:label: lem-ig-rate-function

For independent samples with law $\pi$, the empirical frequencies on a fixed
finite partition with probabilities $p_1,\ldots,p_m>0$ have rate
$I(q)=\sum_jq_j\log(q_j/p_j)$. More precisely, for any attainable type $q$,

$$
\log\Pr(L_N=q)=-NI(q)+O(m\log(N+1)).
$$

For an interacting law $\mu_N$, the independent-sampling identification
requires a separate comparison. A proved implication sufficient for
bounded-observable estimates is the following: if
$H(\mu_N\mid\pi^{\otimes N})\le H_*$, then
{prf:ref}`thm-mixing-variance-corrected` gives, for $|f|\leq B$,

$$
\mathbb E_{\mu_N}|L_Nf-\pi f|^2
\leq\frac{4B^2}{N}\left(H_*+\frac12\log2\right).
$$

Alternatively a full-gradient joint LSI with constant $C_*$ gives
$\operatorname{Var}_{\mu_N}(L_Nf)\leq C_*L^2/N$ for a fixed
$L$-Lipschitz observable, by
{prf:ref}`cor-quantitative-lsi-final`.

**Proof.** The multinomial formula gives
$\Pr(L_N=q)=N!\prod_jp_j^{Nq_j}/(Nq_j)!$.
Applying $\log n!=n\log n-n+O(\log(n+1))$ and $\sum_jq_j=1$
yields the displayed rate. The interacting implications are precisely the
entropy and Poincaré estimates proved in the cited chapters. Existence
of a joint QSD alone supplies neither the multinomial formula nor an
independent-sampling large-deviation rate.
:::

:::{prf:definition} Fixed-mass Gaussian correlation free energy
:label: def-ig-free-energy

For this comparison model, take a bounded coordinate region
$\Omega\subset\mathbb R^d$ of volume $V$, total mass $N$, and
$\rho_0=N/V$. This finite region defines the auxiliary model; the
confining Fractal Gas state space may remain unbounded.

For nonnegative $\rho$ with $\int_\Omega\rho=N$, set $u=\rho-\rho_0$
and define, in fixed reference energy units,

$$
\mathcal F_{\mathrm{IG}}[\rho]
=\int_\Omega\rho\log(\rho/\rho_0)\,dx
 +\frac12\iint_{\Omega^2}K_\varepsilon(x-y)u(x)u(y)\,dx\,dy,
$$

where

$$
K_\varepsilon(r)=C_0e^{-|r|^2/(2\varepsilon_c^2)},\qquad
C_0,\varepsilon_c>0.
$$

Assume the displayed terms are finite. The first term equals
$N D_{\mathrm{KL}}((\rho/N)\,dx\|V^{-1}\,dx)$.
The pair term is a specified correlation energy. Identifying this sum
with the rate function or free energy of an interacting QSD requires
its own law calculation.
:::

:::{prf:lemma} Positivity of the Gaussian correlation free energy
:label: lem-field-free-energy-positivity

For $u\in L^2(\Omega)$ and the preceding fixed-mass model,

$$
\mathcal F_{\mathrm{IG}}[\rho]\geq0,\qquad
\mathcal F_{\mathrm{IG}}[\rho_0]=0.
$$

Equality requires $\rho=\rho_0$ almost everywhere. The functional is
strictly convex on its finite-energy fixed-mass domain.
:::

:::{prf:proof}
The entropy term is a relative entropy times $N$, hence nonnegative,
with equality only at $\rho_0$. Extend $u$ by zero outside $\Omega$.
With Fourier transform $\widehat f(k)=\int e^{-ik\cdot x}f(x)\,dx$,

$$
\widehat K_\varepsilon(k)
=C_0(2\pi)^{d/2}\varepsilon_c^d
 e^{-\varepsilon_c^2|k|^2/2}>0,
$$

so Plancherel gives

$$
\iint K_\varepsilon(x-y)u(x)u(y)\,dx\,dy
=\frac1{(2\pi)^d}\int
 \widehat K_\varepsilon(k)|\widehat u(k)|^2\,dk\geq0.
$$

The positive quadratic form is convex. The strictly convex function
$r\mapsto r\log(r/\rho_0)$ makes the entropy integral strictly convex
on distinct densities. This proves the assertions.
:::

:::{prf:proposition} Quadratic response and its relation to the jump form
:label: prop-jump-hamiltonian-derivation

Let $\rho=\rho_0(1+\phi)$ in the fixed-mass model, with
$\int_\Omega\phi=0$ and $\|\phi\|_\infty\leq a<1$. Then

$$
\mathcal F_{\mathrm{IG}}[\rho]
=\frac12\langle\phi,\mathcal L_{\mathrm{IG}}\phi\rangle+\mathcal R_3,
\qquad
\mathcal L_{\mathrm{IG}}=\rho_0 I+\rho_0^2\mathcal K,
$$

where $(\mathcal K\phi)(x)=\int_\Omega K_\varepsilon(x-y)\phi(y)\,dy$
and

$$
|\mathcal R_3|
\leq\frac{\rho_0}{6(1-a)^2}\int_\Omega|\phi|^3\,dx.
$$

Define the positive jump operator for this symmetric kernel by

$$
(\mathcal J\phi)(x)
=\int_\Omega K_\varepsilon(x-y)(\phi(x)-\phi(y))\,dy,\qquad
\kappa_\Omega(x)=\int_\Omega K_\varepsilon(x-y)\,dy.
$$

Its exact relation to the free-energy Hessian is

$$
\mathcal L_{\mathrm{IG}}
=(\rho_0+\rho_0^2\kappa_\Omega)I-\rho_0^2\mathcal J,
$$

where multiplication by $\kappa_\Omega$ is understood. Moreover,

$$
\langle\phi,\mathcal J\phi\rangle
=\frac12\iint_{\Omega^2}K_\varepsilon(x-y)
                       (\phi(x)-\phi(y))^2\,dx\,dy\geq0.
$$

Thus the pair-energy Hessian and the jump operator are distinct,
explicitly related operators.
:::

:::{prf:proof}
For $f(r)=(1+r)\log(1+r)$,
$f(0)=0$, $f'(0)=1$, $f''(0)=1$, and
$f'''(r)=-(1+r)^{-2}$. Taylor's theorem on $[-a,a]$ gives

$$
f(\phi)=\phi+\tfrac12\phi^2+r_3,\qquad
|r_3|\leq|\phi|^3/[6(1-a)^2].
$$

The integrated linear term vanishes by mass conservation. The
interaction is exactly quadratic, proving the expansion and its bound.

Finally, $\mathcal J=\kappa_\Omega I-\mathcal K$ gives the operator
identity. Exchange $x,y$ in half of
$\int\phi(x)\int K_\varepsilon(x-y)(\phi(x)-\phi(y))\,dy\,dx$.
Kernel symmetry gives the difference-square formula.
:::

### Density perturbation and spatial dilation

The following definition distinguishes a fixed-volume density perturbation from a mass-preserving spatial dilation.

:::{prf:definition} Mass-preserving affine density perturbation
:label: def-boost-perturbation

On $[0,L]^d$, define $\phi_\kappa(z)=\kappa(z_1/L-1/2)$ and
$\rho_\kappa=\rho_0(1+\phi_\kappa)$ for $|\kappa|<2$.
Then $\rho_\kappa>0$ and $\int\rho_\kappa=\rho_0L^d$.
This is a density perturbation. A spatial dilation is instead the
pushforward $\rho_a(z)=a^{-d}\rho(z/a)$ on $a[0,L]^d$;
{prf:ref}`thm-elastic-pressure` differentiates that explicit deformation.
:::

(sec-elastic-pressure)=
## Dilation Pressure and Correlation Stiffness

:::{div} feynman-prose
Hold the kernel and total transported mass fixed while increasing all distances by a factor $a$. A positive Gaussian pair energy decreases because its kernel becomes smaller at separated points. Its pressure is therefore positive for a nonnegative density. An attractive energy with the opposite sign has the opposite response. The derivative will also show why a signed density fluctuation has no universal pressure sign.
:::

:::{prf:theorem} Dilation response and Gaussian correlation stiffness
:label: thm-elastic-pressure

Let $u$ be an integrable compactly supported density or signed density on
$\mathbb R^d$, and set

$$
E(a)=\frac12\iint K_\varepsilon(a(x-y))u(x)u(y)\,dx\,dy,\quad
K_\varepsilon(r)=C_0e^{-|r|^2/(2\varepsilon_c^2)},\quad V(a)=a^dV.
$$

This describes transport of a fixed amount of $u$ by dilation. Its pressure
at $a=1$ is

$$
-\frac{dE}{dV}=\frac{1}{2dV\varepsilon_c^2}
\iint |x-y|^2K_\varepsilon(x-y)u(x)u(y)\,dx\,dy.
$$

For $u\ge0$ this is nonnegative; reversing the sign of the interaction
energy reverses the pressure. For a signed fluctuation $u=\rho-\rho_0$
the formula has no fixed sign. A quadratic free-energy expansion in a
mass-preserving perturbation has zero first derivative at zero strain.

For the translation-invariant jump form with affine perturbation
$\Phi(x)=b\cdot x$, its quadratic energy per unit volume is

$$
\frac{\rho_0^2}{8}\int K_\varepsilon(r)(b\cdot r)^2\,dr
=\frac{C_0\rho_0^2(2\pi)^{d/2}\varepsilon_c^{d+2}}8|b|^2.
$$

**Proof.** Differentiate the Gaussian under the integral:
$E'(1)=-(2\varepsilon_c^2)^{-1}
\iint|x-y|^2K_\varepsilon(x-y)u(x)u(y)\,dx\,dy$.
Divide by $V'(1)=dV$ and change the sign. For $\rho=\rho_0+\kappa u$
with $\int u=0$, the entropy derivative is $\int u=0$ and the interaction
is quadratic in $\kappa$, proving the zero-strain statement. Finally
$\int r_ir_jK_\varepsilon(r)\,dr=
\delta_{ij}C_0(2\pi)^{d/2}\varepsilon_c^{d+2}$ by Gaussian integration.
Contracting with $b_ib_j$ proves the stiffness formula. Stiffness is a
second variation; identifying it with a first-variation pressure requires
a specified deformation and energy convention.
:::

:::{prf:remark} Sign of the interaction response
:label: rem-elastic-interpretation

The positive Gaussian pair energy, its negative attractive counterpart,
and the signed fluctuation energy have different dilation responses.
The formula in {prf:ref}`thm-elastic-pressure` fixes the sign after the
energy and deformation are specified. The jump-form stiffness is positive
and scales as $\varepsilon_c^{d+2}$ with $C_0$ fixed. If instead the kernel
mass is normalized by choosing $C_0\propto\varepsilon_c^{-d}$, its
stiffness scales as $\varepsilon_c^2$. The selected kernel normalization
must therefore be retained when comparing bandwidths or applying the
formula to a row-normalized companion mechanism. Neither stiffness
formula alone fixes a vacuum pressure.
:::

(sec-linearized-dynamics)=
## A Homogeneous Density Model

:::{div} feynman-prose
A translation-invariant model lets us ask how a single sinusoidal density
wave changes. Diffusion smooths it. A balanced Gaussian redistribution
also smooths it, because averaging over nearby points reduces its amplitude.

This calculation is exact for the specified linear model. The full
Fractal Gas mean-field equation contains its actual state-dependent
coefficients and normalized cloning terms. Those are derived in the
mean-field chapter. We use the homogeneous model here as a comparison
whose Fourier multipliers can be calculated completely.
:::

:::{prf:definition} Auxiliary gain–loss density equation
:label: def-mckean-vlasov

For a specified drift $b$, nonnegative kernel $K_{\mathrm{clone}}$,
loss rate $\lambda_{\mathrm{kill}}$, and constant
$D_{\mathrm{eff}}\geq0$, consider

$$
\partial_t\rho
=D_{\mathrm{eff}}\Delta\rho-\nabla\cdot(\rho b)
 +\int K_{\mathrm{clone}}(x,y)\rho(y)\,dy
 -\lambda_{\mathrm{kill}}(x)\rho(x).
$$

Use periodic boundary conditions, or sufficient whole-space decay for
the integrations under consideration. The gain–loss part conserves
mass for every integrable density precisely when

$$
\int K_{\mathrm{clone}}(x,y)\,dx=\lambda_{\mathrm{kill}}(y)
\quad\text{almost everywhere}.
$$

This is a specified linear density model. The actual nonlinear
mean-field equation, companion law, and survival-normalized evolution
are derived in {doc}`../convergence_program/08_mean_field`.

For comparison, if an unnormalized killed density satisfies
$\partial_tf=L^*f-\kappa f$ for a conservative $L$, then
$\rho=f/\int f$ satisfies

$$
\partial_t\rho=L^*\rho-(\kappa-\langle\kappa\rangle_\rho)\rho.
$$

A QSD is stationary for this normalized evolution and is a left
eigenmeasure for the killed one.
:::

:::{prf:proof}
Integrate the gain–loss term and use Fubini. Its integral is
$\int[\int K_{\mathrm{clone}}(x,y)\,dx-\lambda_{\mathrm{kill}}(y)]
\rho(y)\,dy$, proving sufficiency and necessity by testing all
nonnegative densities. For killing,
$m'=-m\langle\kappa\rangle_\rho$ where $m=\int f$.
Differentiating $f/m$ gives the normalized equation.
:::

:::{prf:definition} Uniform reference and homogeneous Gaussian closure
:label: def-uniform-qsd-linearization

Set $b=0$ and let the redistribution operator be convolution:

$$
K_{\mathrm{clone}}(x,y)=k_\varepsilon(x-y),\qquad
k_\varepsilon(r)=\lambda_{\mathrm{kill}}
 (2\pi\varepsilon_c^2)^{-d/2}
 e^{-|r|^2/(2\varepsilon_c^2)}.
$$

On a periodic box of side $L$, periodize this kernel by summing its
translates in $L\mathbb Z^d$. Its integral over the box is
$\lambda_{\mathrm{kill}}$, and the constant density $\rho_0=N/L^d$
is stationary for the conservative model.

Writing $\rho=\rho_0+\delta\rho$ gives the exact equation

$$
\partial_t\delta\rho
=D_{\mathrm{eff}}\Delta\delta\rho
 +k_\varepsilon*\delta\rho-\lambda_{\mathrm{kill}}\delta\rho.
$$

There is no neglected nonlinear term in this specified closure.
On $\mathbb R^d$ the same equation describes perturbations about a
homogeneous background. A nonzero constant background on that space
has infinite total mass and is not a probability density or a QSD.
:::

### Fourier Analysis and Dispersion Relation

:::{prf:theorem} Exact decay multipliers of the homogeneous closure
:label: thm-dispersion-relation

For a translation-invariant gain kernel $k_{\mathrm{gain}}$ and the Fourier convention
$\widehat f(k)=\int e^{-ik\cdot x}f(x)\,dx$, a mode
$e^{ik\cdot x-\omega(k)t}$ has

$$
\omega(k)=D_{\mathrm{eff}}|k|^2+
                 \lambda_{\mathrm{kill}}-\widehat k_{\mathrm{gain}}(k).
$$

For the Gaussian model of
{prf:ref}`def-uniform-qsd-linearization`,

$$
\widehat k_\varepsilon(k)
=\lambda_{\mathrm{kill}}e^{-\varepsilon_c^2|k|^2/2},\qquad
\omega(k)
=D_{\mathrm{eff}}|k|^2+
 \lambda_{\mathrm{kill}}(1-e^{-\varepsilon_c^2|k|^2/2}).
$$

On the periodic box these identities hold at
$k\in(2\pi/L)\mathbb Z^d$.
:::

:::{prf:proof}
The Laplacian multiplies a Fourier mode by $-|k|^2$.
For the gain term, substitute $r=x-y$:

$$
\int k_{\mathrm{gain}}(x-y)e^{ik\cdot y}\,dy
=e^{ik\cdot x}\int k_{\mathrm{gain}}(r)e^{-ik\cdot r}\,dr
=\widehat k_{\mathrm{gain}}(k)e^{ik\cdot x}.
$$

The loss term multiplies the mode by $-\lambda_{\mathrm{kill}}$.
Thus $-\omega=-D_{\mathrm{eff}}|k|^2+\widehat k_{\mathrm{gain}}(k)
-\lambda_{\mathrm{kill}}$.

The product of the one-dimensional Gaussian Fourier integrals is
$e^{-\varepsilon_c^2|k|^2/2}$ after normalization, giving the formula.
For a periodized kernel, integration over one box and summation over
its translates give the same Fourier integral at the allowed discrete
frequencies.
:::

:::{prf:remark} Self-adjoint homogeneous closure
:label: rem-real-eigenvalues

For an even integrable convolution kernel and periodic or whole-space
Laplacian with its standard self-adjoint domain, convolution is a bounded
self-adjoint operator. The bounded perturbation of the self-adjoint
Laplacian is self-adjoint on the same domain. Thus the Gaussian closure
has real Fourier multipliers. General directed cloning kernels need not
satisfy this symmetry; their evolution is treated by the kinetic and
mean-field estimates in the convergence chapters.
:::

(sec-qsd-stability)=
## Stability and Domain-Dependent Rates

:::{div} feynman-prose
Every nonzero sinusoid decays in this Gaussian model. Whether there is a single exponential rate for all perturbations depends on the domain. A periodic box has a smallest nonzero wave number. The whole space has arbitrarily long waves, and those decay arbitrarily slowly. The conserved constant mode must also be separated from the relaxing modes.
:::

:::{prf:theorem} Stability of the homogeneous Gaussian closure
:label: thm-qsd-stability

For the translation-invariant closure in {prf:ref}`def-uniform-qsd-linearization`,
let $D_{\mathrm{eff}},\lambda_{\mathrm{kill}}\ge0$ and $\varepsilon_c>0$.
Every nonzero Fourier mode decays strictly if and only if
$D_{\mathrm{eff}}+\lambda_{\mathrm{kill}}>0$. The zero mode is conserved.
For $q=\varepsilon_c^2|k|^2/2$,

$$
D_{\mathrm{eff}}|k|^2+\lambda_{\mathrm{kill}}\frac{q}{1+q}
\le\omega(k)
\le D_{\mathrm{eff}}|k|^2+\lambda_{\mathrm{kill}}\min(q,1).
$$

**Proof.** Substitute the Gaussian multiplier from
{prf:ref}`thm-dispersion-relation`. For $q\ge0$, $e^q\ge1+q$ implies
$1-e^{-q}\ge q/(1+q)$, and integration of $e^{-r}\le1$ gives
$1-e^{-q}\le q$. Also $1-e^{-q}\le1$. The multiplier is positive for
$k\ne0$ when at least one coefficient is positive and is identically
zero when both vanish. At $k=0$ it is zero in every case.
:::

:::{prf:corollary} Relaxation on a domain with a nonzero first frequency
:label: cor-exponential-relaxation

Assume $D_{\mathrm{eff}}+\lambda_{\mathrm{kill}}>0$. For the preceding
closure on the periodic box of side $L$, the mean-zero
solution satisfies

$$
\|\delta\rho_t\|_2\le e^{-\omega_*t}\|\delta\rho_0\|_2,\qquad
\omega_*=D_{\mathrm{eff}}(2\pi/L)^2+
\lambda_{\mathrm{kill}}[1-e^{-\varepsilon_c^2(2\pi/L)^2/2}]>0.
$$

On $\mathbb R^d$ the same multiplier has infimum zero over nonzero
frequencies; the closure therefore has no uniform exponential $L^2$ rate.
For $D_{\mathrm{eff}}>0$ and $\delta\rho_0\in L^1\cap L^2$ it has the bound
$\|\delta\rho_t\|_2\le C_d(D_{\mathrm{eff}}t)^{-d/4}\|\delta\rho_0\|_1$.
The confining kinetic model uses the different long-time estimates of
{doc}`../convergence_program/06_convergence` and
{doc}`../convergence_program/10_kl_hypocoercive`.

**Proof.** The multiplier increases with $|k|^2$. On the box, every
nonzero Fourier frequency has length at least $2\pi/L$; Parseval gives
the first bound. On $\mathbb R^d$, $\omega(k)\to0$ as $k\to0$.
Plancherel, $|\widehat{\delta\rho_0}|\le\|\delta\rho_0\|_1$, and
$\omega(k)\ge D_{\mathrm{eff}}|k|^2$ give the Gaussian integral bound.
:::

:::{prf:remark} Long-wavelength diffusion
:label: rem-anti-diffusion

The exact Gaussian multiplier has expansion

$$
\omega(k)
=\left(D_{\mathrm{eff}}+
 \frac{\lambda_{\mathrm{kill}}\varepsilon_c^2}{2}\right)|k|^2
-\frac{\lambda_{\mathrm{kill}}\varepsilon_c^4}{8}|k|^4
+O(|k|^6).
$$

Its long-wavelength coefficient is

$$
D_{\mathrm{long}}
=D_{\mathrm{eff}}+\lambda_{\mathrm{kill}}\varepsilon_c^2/2
\geq D_{\mathrm{eff}},
$$

with strict inequality when $\lambda_{\mathrm{kill}}>0$.
The negative fourth-order term is a correction to a small-frequency
series. Extending that truncated polynomial to arbitrarily large
frequencies would produce a spurious instability; the exact multiplier
remains nonnegative. Nonlinear or directed cloning mechanisms use
their own linearization and stability calculation.
:::



(sec-chapman-enskog)=
## Kinetic Diffusion and the Time Step

:::{div} feynman-prose
A velocity keeps part of its value from one instant to the next. Those correlations determine how far position spreads. For an Ornstein–Uhlenbeck velocity, the covariance is an exponential, so we can integrate it exactly. This gives the continuous-time diffusion coefficient. A finite BAOAB step has a related geometric covariance series, which gives its own coefficient before taking the small-step limit.
:::

:::{prf:definition} Free kinetic reference operator
:label: def-phase-space-kinetic-operator

The constant-coefficient reference process

$$
dX=V\,dt,\qquad dV=-\gamma V\,dt+\sigma_v\,dW,\qquad
\gamma,\sigma_v>0,
$$

has backward generator

$$
\mathcal L_{\mathrm{kin}}f
=v\cdot\nabla_xf-\gamma v\cdot\nabla_vf
                   +\frac{\sigma_v^2}{2}\Delta_vf.
$$

Its stationary velocity covariance is $v_T^2I$, where
$v_T^2=\sigma_v^2/(2\gamma)$. Position on $\mathbb R^d$ spreads
and has no uniform invariant probability law. The adaptive, forced,
aligned, and cloning swarm uses its complete generator rather than this
free reference operator.
:::

:::{prf:theorem} Diffusion coefficient of integrated Ornstein--Uhlenbeck motion
:label: thm-einstein-relation

For $dV_t=-\gamma V_tdt+\sigma_vdW_t$, $dX_t=V_tdt$, with
$V_0$ in its stationary Gaussian law, put $v_T^2=\sigma_v^2/(2\gamma)$.
For each coordinate,

$$
\operatorname{Var}(X_t-X_0)=
2v_T^2\left[\frac t\gamma-\frac{1-e^{-\gamma t}}{\gamma^2}\right].
$$

Thus the long-time diffusion coefficient is
$D_{\mathrm{eff}}=v_T^2/\gamma=\sigma_v^2/(2\gamma^2)$.
This coefficient belongs to the stated kinetic reference process.

**Proof.** The explicit OU solution gives
$\mathbb E[V_sV_r]=v_T^2e^{-\gamma|s-r|}$ coordinatewise.
Integrating this covariance over $[0,t]^2$ gives the displayed variance.
Moreover
$X_t-X_0=(V_0-V_t)/\gamma+(\sigma_v/\gamma)W_t$.
After diffusive rescaling the first term vanishes in mean square at each
fixed rescaled time. The Brownian term has variance
$\sigma_v^2t/\gamma^2=2D_{\mathrm{eff}}t$.
:::


:::{prf:lemma} Diffusion of the force-free BAOAB reference step
:label: lem-field-discrete-ou-diffusion

Fix a constant positive definite covariance shape $D_0$, timestep
$\Delta t>0$, and $0<a<1$. Consider

$$
V_{n+1}=aV_n+c_2D_0^{1/2}\xi_n,\qquad
X_{n+1}=X_n+\frac{\Delta t}{2}(V_n+V_{n+1}),
$$

where the $\xi_n$ are independent standard Gaussians and the velocity
starts in stationarity. Then

$$
C_v=\operatorname{Cov}(V_n)=\frac{c_2^2}{1-a^2}D_0,\qquad
\lim_{m\to\infty}
 \frac{\operatorname{Cov}(X_m-X_0)}{2m\Delta t}
=\frac{\Delta t\,c_2^2}{2(1-a)^2}D_0.
$$

For the thermostat coefficients $a=e^{-\gamma\Delta t}$ and
$c_2^2=v_T^2(1-a^2)$, this is

$$
D_{\Delta t}
=\frac{v_T^2\Delta t}{2}
 \coth(\gamma\Delta t/2)D_0
=\frac{v_T^2}{\gamma}
 \left[1+\frac{(\gamma\Delta t)^2}{12}
             +O((\gamma\Delta t)^4)\right]D_0.
$$

This is the existing BAOAB update specialized to zero force, zero
alignment, no jumps, and constant diffusion shape. The result identifies
that reference update; it does not alter the full algorithm.
:::

:::{prf:proof}
The covariance recursion is $C_v=a^2C_v+c_2^2D_0$, giving the
stationary covariance. Iteration gives
$\operatorname{Cov}(V_{n+r},V_n)=a^rC_v$. Therefore

$$
\frac1m\operatorname{Cov}\left(\sum_{n=0}^{m-1}V_n\right)
=\left[1+2\sum_{r=1}^{m-1}(1-r/m)a^r\right]C_v
\longrightarrow\frac{1+a}{1-a}C_v.
$$

The position increment is

$$
X_m-X_0
=\Delta t\sum_{n=0}^{m-1}V_n
 +\frac{\Delta t}{2}(V_m-V_0).
$$

The second term has bounded second moment as $m\to\infty$; its
covariance and cross terms vanish after division by $m$. This proves
the limiting diffusion tensor. Substitute the thermostat coefficients
and use $(1+e^{-z})/(1-e^{-z})=\coth(z/2)$ and its Taylor expansion.
:::

(sec-radiation-pressure)=
## Pressure of Specified Gaussian Modes

:::{div} feynman-prose
Now specify a finite collection of fluctuating modes and their quadratic energy. A Gaussian integral gives the partition function exactly. Pressure depends on how the energy coefficients change with volume. A relaxation rate by itself does not specify that energy: it also depends on mobility and noise normalization. The following model states all three choices.
:::

:::{prf:assumption} Gaussian fluctuation reference model
:label: ass-thermal-equilibrium

For a finite set of real modes $q_j$, specify
$\Theta=k_BT_{\mathrm{eff}}>0$ and the quadratic energy

$$
E(q)=\frac12\sum_jw_jq_j^2,\qquad w_j>0,
$$

with density $Z^{-1}e^{-E/\Theta}$ relative to a specified Lebesgue
measure on mode coordinates. Then $\mathbb E q_j^2=\Theta/w_j$
and the modes are independent.

For mobility $m_j>0$, the dynamics

$$
dq_j=-m_jw_jq_j\,dt+\sqrt{2m_j\Theta}\,dB_j
$$

have this invariant law and relaxation rate $m_jw_j$.
Identifying $w_j$ with a measured rate therefore requires $m_j=1$
in the declared units, or another measured mobility and its matched
noise amplitude. An arbitrary cloning QSD does not determine this
Gaussian energy model.
:::

:::{prf:proof}
The density factors into normalized one-dimensional Gaussians with
variance $\Theta/w_j$. The stated OU process has stationary variance
$(2m_j\Theta)/(2m_jw_j)=\Theta/w_j$, proving invariance and the
decay rate.
:::

:::{prf:proposition} Pressure of a finite Gaussian mode ensemble
:label: prop-radiation-pressure

For {prf:ref}`ass-thermal-equilibrium`, at fixed temperature, fixed
mode index set, and fixed reference measure on mode coordinates, suppose
$w_j(V)>0$ is differentiable. The mode pressure is

$$
P_{\mathrm{modes}}=-\frac{k_BT_{\mathrm{eff}}}{2}
\sum_j\partial_V\log w_j(V).
$$

If every $w_j(V)=a_jV^{-2/d}$, then
$P_{\mathrm{modes}}=k_BT_{\mathrm{eff}}n/(dV)$ for $n$ real modes.
A prescribed spatial cutoff $|k|\le k_*$ on a periodic box gives
$n\sim V\operatorname{vol}(B_d)k_*^d/(2\pi)^d$ in the large-box limit,
with the zero mode omitted. A classical Gaussian ensemble has no intrinsic
thermal cutoff; an infinite unregularized mode count diverges.

**Proof.** Gaussian integration gives
$Z=\prod_j(2\pi k_BT_{\mathrm{eff}}/w_j)^{1/2}$.
Differentiate $F=-k_BT_{\mathrm{eff}}\log Z$ and use $P=-\partial_VF$.
The power law follows by differentiating $\log w_j$.
For mode counting, place a unit cube at each integer lattice point in the
ball of radius $Lk_*/(2\pi)$. Their union lies between balls with radii
differing by at most $\sqrt d$, giving the stated leading volume.
Changing the cutoff or the mode index set during a volume derivative
requires the corresponding additional terms in $F$.
:::

:::{prf:remark} Pressure and stiffness use different derivatives
:label: rem-pressure-comparison

The pair-energy pressure is computed along a specified spatial dilation.
The mode pressure differentiates the Gaussian partition function at fixed
temperature and fixed mode set. Their sum represents a chosen effective
energy model only when these conventions and the common state law agree.
The quadratic correlation stiffness is a separate response coefficient.
:::

(sec-pressure-regimes)=
## Pressure Regime Analysis

:::{prf:definition} Crossover of a prescribed pressure model
:label: def-thermal-correlation-length

For a phenomenological pressure model
$P(\varepsilon_c)=B-A\varepsilon_c^{d+2}$ with fixed $A,B>0$, define
$\varepsilon_c^{\mathrm{th}}=(B/A)^{1/(d+2)}$.
The coefficient $A$ is an attractive-pressure coefficient of the chosen
model; its value must come from a specified dilation derivative.
It is not obtained by relabeling the positive jump stiffness.
:::

:::{prf:theorem} Sign of the prescribed two-term pressure
:label: thm-pressure-regimes

Under {prf:ref}`def-thermal-correlation-length`,

$$
P(\varepsilon_c)=B\left[1-
\left(\frac{\varepsilon_c}{\varepsilon_c^{\mathrm{th}}}\right)^{d+2}\right].
$$

It is positive below the crossover, zero at the crossover, and negative
above it. These conclusions concern the prescribed pressure model.

**Proof.** Substitute $A=B/(\varepsilon_c^{\mathrm{th}})^{d+2}$ and use
strict monotonicity of $r^{d+2}$ for $r>0$. In particular, at fixed $A,B$
the term proportional to $\varepsilon_c^{d+2}$ becomes smaller, rather
than larger, as the correlation length tends to zero.
:::

:::{prf:remark} Analytical results and effective closures
:label: rem-analysis-limitations

The homogeneous Gaussian closure has the exact Fourier multiplier and
mode estimates above. The kinetic reference has the proved OU diffusion
coefficient. Gaussian pair energies and mode ensembles have explicit
first-variation pressures. Applying these formulas to the full interacting
QSD requires the corresponding closure and law identifications, using
{doc}`../convergence_program/07_discrete_qsd`,
{doc}`../convergence_program/09_propagation_chaos`, and
{doc}`../convergence_program/15_kl_convergence`.
:::

(sec-stress-energy-tensor)=
## Comparison: Stress and an Einstein Constitutive Equation

:::{div} feynman-prose
The transition-derived budgets above give measurable sources. The
calculation in this section asks a separate, conditional question:
what curvature would a specified Einstein equation predict for a chosen
perfect fluid? Keeping the answer is useful for comparison, provided we
do not count the assumed equation as an algorithmic derivation.

An isotropic stress in a local rest frame has one energy density and one
pressure. This determines the algebraic form of a perfect-fluid tensor.

A field equation is an additional relation between that tensor and the
metric. Once the relation is specified, taking its trace and contracting
with an observer's velocity gives a definite curvature formula. Geometry
alone does not determine the energy model or the proportionality
constant.
:::

:::{prf:definition} Perfect-fluid effective stress
:label: def-effective-stress-energy

Let $G_{ab}$ be the specified Lorentzian metric in dimension $d+1$,
and let $u$ be a unit timelike field, $G(u,u)=-1$. A symmetric stress
with zero rest-frame energy flux and isotropic spatial stress has form

$$
T_{ab}^{\mathrm{eff}}
=e\,u_au_b+P\,h_{ab}
=(e+P)u_au_b+P\,G_{ab},\qquad
h_{ab}=G_{ab}+u_au_b.
$$

Here $e=T(u,u)$ is energy density and $P$ is rest-frame pressure.
The field $u$ need not be geodesic. An anisotropic stress or nonzero
energy flux requires the corresponding additional tensor terms.

For the chosen effective energy model one may set
$P=P_{\mathrm{pair}}+P_{\mathrm{modes}}$ after fixing their common
volume, temperature, and state-law conventions. This equation is a
definition of that model, not an identification forced by isotropy of
a sampling density.
:::

:::{prf:proof}
Choose a local orthonormal rest frame with time direction $u$.
The stated conditions give components
$T_{00}=e$, $T_{0i}=0$, and $T_{ij}=P\delta_{ij}$.
The displayed tensor has exactly these components, proving its
coordinate-independent form.
:::

:::{prf:definition} Signed pressure parameter and vacuum convention
:label: def-effective-cosmological-constant

Choose a positive coupling $\kappa_G$, with
$\kappa_G=8\pi G_{\mathrm{eff}}/c^4$ in four-dimensional physical
units, and define

$$
\Lambda_P=\kappa_G P_{\mathrm{vac}}.
$$

This is a signed pressure parameter. Let
$\mathsf E_{ab}=R_{ab}-\tfrac12R G_{ab}$ denote the Einstein tensor.
In the convention

$$
\mathsf E_{ab}+\Lambda G_{ab}=\kappa_G T_{ab},
$$

vacuum stress has
$T_{ab}^{\mathrm{vac}}=-e_{\mathrm{vac}}G_{ab}$ and
$P_{\mathrm{vac}}=-e_{\mathrm{vac}}$. Moving that term to the left
gives

$$
\Delta\Lambda=\kappa_Ge_{\mathrm{vac}}=-\Lambda_P.
$$

Thus the pressure parameter and the vacuum contribution to the
Einstein constant have opposite signs. A physical identification also
requires the vacuum equation of state and the field equation.
Dimensional consistency does not fix the coupling's value.
:::

:::{prf:theorem} Ricci contraction under an Einstein constitutive equation
:label: thm-structural-correspondence

In spacetime dimension $n=d+1>2$, suppose an effective metric and
stress satisfy the specified constitutive equation

$$
\mathsf E_{ab}+\Lambda G_{ab}=\kappa_G T_{ab},
\qquad
\mathsf E_{ab}=R_{ab}-\tfrac12R G_{ab}.
$$

For a perfect fluid with energy density $e$, pressure $P$, and
unit timelike field $u$,

$$
R_{ab}u^au^b
=\frac{\kappa_G}{d-1}\bigl[(d-2)e+dP\bigr]
 -\frac{2\Lambda}{d-1}.
$$

Here $\kappa_G$ has the units of the specified dimension and energy
normalization. In four-dimensional physical units it may be written
$8\pi G_{\mathrm{eff}}/c^4$.
:::

:::{prf:proof}
Taking the trace yields

$$
R=\frac{2(n\Lambda-\kappa_GT)}{n-2}.
$$

Substitution into the constitutive equation gives

$$
R_{ab}
=\kappa_G\left(T_{ab}-\frac{T}{n-2}G_{ab}\right)
 +\frac{2\Lambda}{n-2}G_{ab}.
$$

Now $G(u,u)=-1$, $T(u,u)=e$, and $T=-e+dP$.
Contracting yields the stated formula. Under the congruence hypotheses,
it can be substituted into Raychaudhuri's identity.

Conservation and Raychaudhuri do not imply the constitutive equation.
For example, on Minkowski space take $\Lambda=0$ and a nonzero
constant perfect-fluid stress. Its divergence vanishes and all geometric
identities hold, while $\mathsf E=0\ne\kappa_G T$.
:::

:::{prf:remark} Geometric identity and constitutive equation
:label: rem-correspondence-meaning

This section supplies no derivation of an Einstein equation from the
transition law. A test of such an equation must use independently computed
mechanical observables from
{ref}`sec-algorithmic-balance-laws` and an independently specified
spacetime reconstruction. Its coupling, source, and error cannot be
defined by fitting the source to the same geometric tensor.

Raychaudhuri relates the expansion of a specified congruence to the Ricci
tensor of its metric. The perfect-fluid contraction in
{prf:ref}`thm-structural-correspondence` additionally uses the stated
Einstein constitutive equation. A pressure sign determines that contraction
only after the energy density, cosmological term, dimensional normalization,
and constitutive equation are fixed.
:::

(sec-summary-field-equations)=
## Results and Their Applications

The algorithm's finite-step metric equation is
{prf:ref}`thm-algorithmic-observable-increment` applied to specified fixed
spatial probes. Its drift and covariance follow from the full transition
law. {prf:ref}`prop-algorithmic-stage-telescoping` resolves the drift into
actual operator contributions and retains cross covariances.
{prf:ref}`prop-algorithmic-material-metric` separates field change,
spatial sampling, and clone replacement.

The thermostat has exact conditional energy and momentum moments.
Kicks, clone transfers, restitution, transport, and eligibility changes
have explicit finite-step budgets. Together they determine mechanical
sources without imposing an equilibrium law. Independent-replica tests
check predictions with the appropriate ensemble uncertainty. A closed
macroscopic or gravitational equation remains a further derivation.

The fixed-mass Gaussian correlation energy is nonnegative and has the
proved quadratic expansion. Its Hessian and the jump operator obey the
explicit identity in {prf:ref}`prop-jump-hamiltonian-derivation`.
The dilation pressure follows by differentiating the stated energy;
the sign depends on that energy and the transported density.

The homogeneous closure has the exact Fourier multiplier and stability
bounds above. On a periodic box its mean-zero modes have an exponential
rate; on $\mathbb R^d$ arbitrarily long waves remove a uniform spectral
gap. The continuous OU and constant-coefficient BAOAB references give
their respective diffusion tensors.

For a finite Gaussian mode ensemble,

$$
P_{\mathrm{modes}}
=-\frac{k_BT_{\mathrm{eff}}}{2}
 \sum_j\partial_V\log w_j.
$$

The prescribed two-term pressure
$B-A\varepsilon_c^{d+2}$ is positive below its crossover and negative
above it when $A,B$ remain fixed. That algebra does not identify an
Einstein cosmological constant.

Applications to the actual interacting swarm use its identified law,
normalized mean-field dynamics, and the finite-particle, LSI, and entropy
estimates. The particular stress–Ricci comparison proved in
{prf:ref}`thm-structural-correspondence` assumes its constitutive
equation; it is not the transition-derived metric equation.

(sec-symbols-field-equations)=
## Table of Symbols

| Symbol | Meaning |
|---|---|
| $X_n$, $\mathsf P_{N,h}$ | Extended algorithmic state and fixed-step transition operator |
| $A$, $b_A$, $\Gamma_A$ | Specified observable, conditional increment drift, and covariance |
| $g_X(z)$ | Metric reconstruction at a fixed reference-coordinate probe |
| $\eta_{n+1}$ | Zero-mean conditional observable fluctuation |
| $K$, $p$, $U$, $E$ | Unit-mass kinetic energy, momentum, specified potential energy, and total energy |
| $B$, $Q=BB^\top$ | Actual velocity noise factor and its covariance rate |
| $c$, $s^2$ | BAOAB damping and integrated variance time factor |
| $\mu_4$ | Fourth moment of a standardized scalar innovation |
| $a_i$, $C_i$, $q_{ij}$ | Eligibility, realized clone indicator, and joint source/acceptance probability |
| $\mu_X$, $p_\psi$ | Fixed-capacity empirical measure and tested local momentum |
| $\mathcal F_{\mathrm{IG}}$ | Specified fixed-mass correlation free energy |
| $\mathcal L_{\mathrm{IG}}$, $\mathcal J$ | Free-energy Hessian and positive jump operator |
| $K_\varepsilon$, $C_0$, $\varepsilon_c$ | Gaussian pair kernel, amplitude, and length scale |
| $\phi_\kappa$ | Fixed-volume affine density perturbation |
| $P_{\mathrm{pair}}$, $P_{\mathrm{modes}}$ | Pressures of the specified pair and mode energies |
| $D_{\mathrm{eff}}$, $D_{\Delta t}$ | Continuous reference and finite-step diffusion coefficients |
| $\omega(k)$ | Decay rate of the homogeneous Fourier mode |
| $\lambda_{\mathrm{kill}}$ | Balanced loss coefficient in the auxiliary gain–loss model |
| $T_{\mathrm{eff}}$, $w_j$, $m_j$ | Mode temperature, quadratic energy coefficient, and mobility |
| $\varepsilon_c^{\mathrm{th}}$ | Crossover in the prescribed two-term pressure model |
| $G_{ab}$, $\mathsf E_{ab}$ | Lorentzian metric and its Einstein tensor |
| $e$, $P$, $T_{ab}$ | Rest-frame energy density, pressure, and effective stress |
| $\kappa_G$, $\Lambda$ | Coupling and cosmological term in the specified constitutive equation |
| $\Lambda_P$ | Signed pressure parameter; vacuum contributes $-\Lambda_P$ to $\Lambda$ |
| $\gamma$, $\sigma_v^2$, $v_T^2$ | Reference friction, noise variance rate, and stationary velocity variance |

(sec-references-field-equations)=
## References

### Geometry and Analytical Foundations

- {doc}`01_emergent_geometry` --- Emergent Riemannian geometry from fitness landscape
- {doc}`02_scutoid_spacetime` --- Cell reconstruction, sampling, and volume evolution
- {doc}`03_curvature_gravity` --- Curvature from discrete holonomy

### External References

```{bibliography}
:filter: docname in docnames
```

**Key citations:**

- Large deviations and rate functions: {cite}`dembo1998large`
- Kinetic scaling and Chapman–Enskog context: {cite}`chapman1990mathematical`
- McKean-Vlasov equations: {cite}`sznitman1991topics`
- Einstein relation and fluctuation-dissipation: {cite}`kubo1966fluctuation`
