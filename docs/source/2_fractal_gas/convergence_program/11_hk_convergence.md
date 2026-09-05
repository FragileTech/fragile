# Mass, Hellinger, and Transport Convergence

:::{div} feynman-prose
**TLDR.** An alive measure carries two kinds of information: how much mass
survives and where that mass is distributed. The square-root mass difference
and the normalized distribution separate exactly in Hellinger distance. Combine
that identity with the entropy estimates of {doc}`15_kl_convergence` and the
entropy-to-transport inequality of {doc}`13_quantitative_error_bounds`, and
one obtains a quantitative bound for mass, shape, and transport together.

Think of two clouds of walkers. One cloud may contain fewer alive walkers even
when both clouds have the same spatial profile. Or their alive fractions may
agree while the clouds occupy different regions. A convergence statement should
tell us which difference it controls. The first three sections make those
questions precise and assemble the main estimate. The density and linearization
sections give additional tools for local comparisons and perturbations.

Choose the underlying space before choosing the distance. An alive
one-particle measure can have mass below one; its normalization describes where
an alive particle lies. A QSD is instead a probability law on living swarm
configurations. Its mass is one, while the fraction of alive slots is a random
observable of the swarm. The same measure inequalities apply on either space,
with the mass and reference law appropriate to that space.
:::

(sec-hk-metrics)=
## Metrics for Alive Measures

:::{div} feynman-prose
Hellinger distance compares square roots of mass density. Wasserstein distance
compares the cost of moving a normalized distribution. The additive distance
below keeps both measurements. Canonical Hellinger–Kantorovich distance solves
a different problem: it chooses transport and creation or removal of mass along
a path, charging for both. Its normalization determines the factors in the
reaction and transport bounds.
:::

:::{prf:definition} Hellinger distance and the additive transport distance
:label: def-hk-metric-intro

For finite measures $\mu,\nu$, define
$H^2(\mu,\nu)=\int(\sqrt{d\mu/d\lambda}-\sqrt{d\nu/d\lambda})^2d\lambda$,
where $\lambda$ dominates both. For positive masses $m=\mu(E)$ and
$n=\nu(E)$, put $p=\mu/m$, $q=\nu/n$ and

$$
D^2(\mu,\nu)=H^2(\mu,\nu)+W_2^2(p,q).
$$

This additive distance is distinct from canonical Hellinger–Kantorovich
(HK) distance. In the normalization

$$
\mathrm{HK}^2(\mu,\nu)=\inf\int_0^1\int
\left(|v_t|^2+\tfrac14 r_t^2\right)d\rho_t\,dt,
\quad\partial_t\rho_t+\nabla\!\cdot(\rho_tv_t)=r_t\rho_t,
$$

pure reaction gives $\mathrm{HK}(\mu,\nu)\leq H(\mu,\nu)$, and pure
transport gives $\mathrm{HK}^2(mp,mq)\leq mW_2^2(p,q)$.
The dynamic metric and its measure-valued extension are developed by
[Liero, Mielke, and Savaré](https://arxiv.org/abs/1509.00068).
:::

:::{prf:proof}
For pure reaction use
$\rho_t=((1-t)\sqrt{d\mu/d\lambda}+t\sqrt{d\nu/d\lambda})^2\lambda$,
$v_t=0$, and $r_t=\partial_t\rho_t/\rho_t$. Its action is
$\int(\sqrt{d\nu/d\lambda}-\sqrt{d\mu/d\lambda})^2d\lambda$; approximation
handles zeros. For equal-mass transport, multiply a probability transport
curve by $m$ and take $r_t=0$. Its action is $m$ times the probability
transport action. Infimizing proves the second bound. The Hellinger formula
is independent of $\lambda$ by the Radon–Nikodym chain rule. The additive
$D$ is a metric on positive finite measures with finite normalized second
moments: apply the triangle inequality to its two components and then the
Euclidean norm on $\mathbb R^2$.
:::

:::{div} feynman-prose
First hold the shape fixed. If both clouds have the same normalized profile
$p$, Hellinger distance becomes $|\sqrt m-\sqrt n|$. Now let the profiles
change as well. Expanding the two squares produces exactly one extra term:
the Hellinger distance between the normalized profiles, weighted by
$\sqrt{mn}$. That algebra is the main organizing identity of the chapter.
:::

:::{prf:lemma} Exact separation of mass and shape
:label: lem-hk-mass-shape-identity

For $\mu=mp$, $\nu=nq$ with probability laws $p,q$,

$$
H^2(mp,nq)=(\sqrt m-\sqrt n)^2+\sqrt{mn}\,H^2(p,q).
$$

Moreover $H^2(p,q)\leq D_{\mathrm{KL}}(p\Vert q)$, and if
$m,n\geq m_0>0$, then
$(\sqrt m-\sqrt n)^2\leq(m-n)^2/(4m_0)$.
:::

:::{prf:proof}
Write the affinity $A=\int\sqrt{dp\,dq}$. Expanding gives
$H^2(mp,nq)=m+n-2\sqrt{mn}A$ and $H^2(p,q)=2-2A$.
If the entropy is finite, Jensen applied under $p$ gives
$\log A\geq-\tfrac12D_{\mathrm{KL}}(p\Vert q)$. Thus
$H^2(p,q)\leq2(1-e^{-D_{\mathrm{KL}}/2})\leq D_{\mathrm{KL}}$.
The infinite-entropy case is immediate. Rationalize the difference of square
roots for the last inequality.
:::

:::{div} feynman-prose
There is a useful warning about what constitutes a cloud. The probability law
of a particle can have a smooth density, while a recorded cloud consists of
finitely many points. A continuous density assigns zero mass to that finite
set; the empirical measure puts all its mass there. Hellinger distance detects
this distinction even when the points form an excellent transport or numerical
approximation to the density.
:::

:::{prf:remark} Laws and finite empirical measures
:label: rem-hk-empirical-singularity

A finite atomic empirical probability measure and an absolutely continuous
probability law are mutually singular, so their squared Hellinger distance
is exactly $2$. Entropy and Hellinger convergence below concern evolving
probability laws or explicitly smoothed empirical measures. Empirical
transport and observable convergence use {doc}`13_quantitative_error_bounds`.
This distinction applies even if the individual walkers have smooth marginal
laws: a realization of their empirical measure remains atomic.
:::

(sec-hk-mass)=
## Alive-Mass Balance and Fluctuations

:::{div} feynman-prose
Follow one update in the order the algorithm performs it. Revival fills dead
slots when its required alive companions are available. The next kinetic and
boundary stage determines which slots survive. Once the shared choices are
fixed, independent survival indicators give a variance of order $1/N$ for
their average. This is the source of the sampling term in the first estimate.

The drift of the conditional mean and the fluctuation around that mean are
separate quantities. A contracting mean can coexist with a nonzero variance
floor, and an alive replacement preserves the number of alive slots.
:::

:::{prf:lemma} Revival and death in a fixed-size population
:label: lem-mass-contraction-revival-death

Suppose a successful revival stage leaves $N$ alive walkers. Conditional on
the resulting configuration $S^+$, let their next kinetic survival indicators
be independent with probabilities $p_i(S^+)$. For the next alive fraction
$m'=N^{-1}\sum_iI_i$,

$$
b(S^+):=\mathbb E[m'\mid S^+]=\frac1N\sum_ip_i(S^+),\qquad
\operatorname{Var}(m'\mid S^+)\leq\frac1{4N}.
$$

If $|b(S^+)-m_*|\leq r|m-m_*|+\delta(S^+)$, then for every $\eta>0$,

$$
\mathbb E(m'-m_*)^2\leq(1+\eta)r^2\mathbb E(m-m_*)^2
 +(1+\eta^{-1})\mathbb E\delta^2+\frac1{4N}.
$$

Thus $(1+\eta)r^2<1$ and a uniform defect bound imply a geometric approach
to the corresponding variance floor. Replacing an alive walker by a clone
preserves the alive count; it is not an additional birth.
:::

:::{prf:proof}
Conditional independence gives variance
$N^{-2}\sum_ip_i(1-p_i)\leq1/(4N)$. Apply the conditional second-moment
identity around $m_*$, followed by
$(a+b)^2\leq(1+\eta)a^2+(1+\eta^{-1})b^2$.
Iterating $u_{j+1}\leq Ru_j+B$, $R<1$, gives
$u_j\leq R^ju_0+B(1-R^j)/(1-R)$.
The assumptions specify the update stage and its available alive population;
extinction or an unsuccessful revival stage is included as an extra defect
or treated by the conditioned-kernel estimate below.
:::

:::{prf:lemma} Structural drift retains its fluctuation floor
:label: lem-structural-variance-contraction

If the combined transition satisfies $\mathbb E[V_{j+1}\mid S_j]\leq rV_j+b$
for $V\geq0$, $r<1$, then

$$
\mathbb EV_j\leq r^j\mathbb EV_0+\frac{b(1-r^j)}{1-r}.
$$

The component-matrix estimates in {doc}`06_convergence` establish this form
for their specified weighted structural observables. If $V$ controls a
squared transport distance, the same upper bound controls that distance.
:::

:::{prf:proof}
Take expectations and iterate the affine inequality. A positive $b$ produces
a positive floor; the displayed inequality by itself does not show
$\mathbb EV_j\to0$ or identify a stationary distribution.
:::

:::{div} feynman-prose
In the continuous alive equation, death removes existing mass and revival
supplies mass from the inactive population. Integrating the equation exposes
that balance directly. The cancellation of cloning gain and loss uses a
probability-valued proposal; a proposal that can leave the valid domain has
its own loss term.
:::

:::{prf:proposition} Killing rates and alive mass in the continuous model
:label: prop-killing-rate-continuous

For bounded interior killing $0\leq c\leq C$, probability-valued revival at
rate $\lambda>0$, and a mass-neutral cloning kernel, the continuous alive
fraction obeys

$$
m_a'=-\int cf+\lambda(1-m_a),\qquad
m_a(t)\geq\frac\lambda{C+\lambda}
+\left(m_a(0)-\frac\lambda{C+\lambda}\right)e^{-(C+\lambda)t}.
$$

For boundary absorption or subprobability cloning proposals, the flux and
proposal-loss terms from {doc}`08_mean_field` enter the same balance.
:::

:::{prf:proof}
Integrate the alive equation and cancel the cloning gain and loss. Since
$\int cf\leq Cm_a$, multiply the resulting differential inequality by
$e^{(C+\lambda)t}$ and integrate. The boundary and failed-proposal terms are
exactly those in {prf:ref}`proof-mean-field-equation` and
{prf:ref}`rem-mean-field-cloning-boundary-loss`.
:::

:::{div} feynman-prose
A small chance of extinction in one step gives a useful bound over a specified
number of steps. The bound pays for every opportunity for extinction. It
therefore retains the horizon $T$; it does not create an event of survival
forever with positive probability.
:::

:::{prf:theorem} Finite-horizon survival from safe-walker estimates
:label: thm-exponential-survival

Suppose at each successful step, conditional on its history, at least $aN$
walkers have independent death probabilities at most $q<1$. Then
$\mathbb P(\tau_\dagger\leq T)\leq Tq^{aN}$ for integer $T$.
If the safe-walker condition fails with conditional probability at most
$\delta_N$, replace the right side by $T(q^{aN}+\delta_N)$.
:::

:::{prf:proof}
On the stated stage condition, total extinction requires all $aN$ designated
walkers to die, which has probability at most $q^{aN}$. Add the condition's
failure probability, then sum conditional one-step probabilities over the
first $T$ steps. {prf:ref}`cor-extinction-suppression` supplies the same
counting argument with its explicit Gaussian and barrier inputs.
:::

:::{prf:lemma} Lower alive-mass concentration
:label: lem-mass-lower-bound-high-prob

Conditional on a post-revival configuration, suppose independent indicators
$I_i$ have mean alive fraction $\bar p\geq p_0>0$. For $0<b<p_0$,

$$
\mathbb P\left(N^{-1}\sum_iI_i<b\mid S^+\right)
\leq e^{-2N(p_0-b)^2}.
$$
:::

:::{prf:proof}
For a variable in $[0,1]$, the log moment generating function of its centered
version has second derivative at most $1/4$ (the variance under each
exponential tilt), and vanishing value and derivative at zero. Thus its log
moment generating function is at most $t^2/8$. Multiply the independent
bounds and apply Markov's inequality to
$\exp[-t\sum_i(I_i-\mathbb EI_i)]$. Minimizing at
$t=4(\bar p-b)$ gives the result.
:::

:::{div} feynman-prose
The mass concentration estimate makes a positive lower bound available with
high probability. That event must be carried into subsequent estimates: a
positive mean alive fraction is not a pathwise lower bound. Conditioning on
survival also changes probabilities, through the denominator in the following
elementary calculation.
:::

:::{prf:corollary} Conditioning on finite-time survival
:label: cor-conditional-mass-lower-bound

If $\mathbb P(A^c)\leq\varepsilon$ and $\mathbb P(\tau_\dagger>t)\geq s>0$,
then $\mathbb P(A^c\mid\tau_\dagger>t)\leq\varepsilon/s$.
Likewise, for $Y\geq0$,
$\mathbb E[Y\mid\tau_\dagger>t]\leq\mathbb EY/s$.
:::

:::{prf:proof}
Bound the numerator of each conditional probability or expectation
by its unconditional counterpart and divide by the survival probability.
:::

:::{prf:proposition} Alive-mass variance under stationary concentration
:label: prop-poc-mass

For an exchangeable marked-state law, let $I_i$ be its alive indicators.
If $\operatorname{Var}(I_i)\leq1/4$ and
$|\operatorname{Cov}(I_i,I_j)|\leq C/N$, then
$\operatorname{Var}(N^{-1}\sum_iI_i)\leq(1/4+C)/N$.
The bounded-observable entropy estimate
{prf:ref}`thm-mixing-variance-corrected` also applies directly to these
indicators when its joint relative-entropy hypothesis holds.
:::

:::{prf:proof}
Expand the variance of the sum. There are $N$ diagonal and $N(N-1)$
off-diagonal terms. Divide by $N^2$ and use the stated bounds.
:::

(sec-hk-entropy)=
## Hellinger and Transport Bounds from Entropy

:::{div} feynman-prose
Here the entropy calculation does most of the work. Relative entropy bounds
Hellinger distance directly. A logarithmic Sobolev inequality for the specified
reference law also converts entropy into a squared transport bound. The two
conversions preserve the time dependence and any residual in the original
entropy estimate.

For a kinetic process, that entropy estimate comes from the modified Fisher
information and hypocoercive calculation. For a killed process, it includes
the normalization of the evolving law. Those proofs are in
{doc}`10_kl_hypocoercive` and {doc}`15_kl_convergence`; here we use their
conclusions for the law to which they apply.
:::

:::{prf:lemma} Hellinger convergence from the recovered kinetic estimate
:label: lem-kinetic-hellinger-contraction

Suppose the normalized evolving law $p_t$ and its specified reference $q$
satisfy the recovered entropy estimate

$$
D_{\mathrm{KL}}(p_t\Vert q)\leq Ae^{-\lambda t}+E_t.
$$

Then $H^2(p_t,q)\leq Ae^{-\lambda t}+E_t$. If $q$ has LSI constant $C$
in the convention $\operatorname{Ent}_q(g^2)\leq2C\int|\nabla g|^2dq$,
then also $W_2^2(p_t,q)\leq2C(Ae^{-\lambda t}+E_t)$.
:::

:::{prf:proof}
Use {prf:ref}`lem-hk-mass-shape-identity` for Hellinger distance and
{prf:ref}`lem-wasserstein-entropy` for transport. The complete kinetic and
conditioned-QSD entropy proofs, including their distinct reference laws and
normalization terms, are in {doc}`10_kl_hypocoercive` and
{doc}`15_kl_convergence`. Their population-independent constants pass to
these inequalities without a global upper bound on $dp_t/dq$.
:::

:::{div} feynman-prose
Now assemble the pieces. The lower mass bound controls the difference of
square roots by the difference of masses. The upper mass bound controls the
factor multiplying the shape distance. The transport contribution adds its
LSI constant. Every term has a visible origin, so a fluctuation or approximation
floor in an input remains visible in the result.
:::

:::{prf:theorem} Joint convergence of mass and shape
:label: thm-hk-convergence-main-assembly

Let $\mu_t=m_tp_t$ and $\mu_*=m_*q$, with
$0<m_0\leq m_t,m_*\leq m_1$. Suppose

$$
\mathbb E(m_t-m_*)^2\leq A_me^{-\lambda_mt}+B_m(t),\qquad
\mathbb E D_{\mathrm{KL}}(p_t\Vert q)\leq A_he^{-\lambda_ht}+B_h(t),
$$

and $q$ has LSI constant $C$. Then

$$
\mathbb ED^2(\mu_t,\mu_*)\leq
\frac{A_me^{-\lambda_mt}+B_m(t)}{4m_0}
+(m_1+2C)\left(A_he^{-\lambda_ht}+B_h(t)\right).
$$

The same right side bounds $\mathbb E\mathrm{HK}^2(\mu_t,\mu_*)$.
:::

:::{prf:proof}
The exact mass-shape identity and the lower mass bound control the squared
root-mass difference by $(m_t-m_*)^2/(4m_0)$. Since
$\sqrt{m_tm_*}\leq m_1$, its remaining term is at most $m_1H^2(p_t,q)$.
Apply the preceding entropy-to-Hellinger and transport inequalities, then
take expectations. Finally $\mathrm{HK}\leq H\leq D$.
All defects remain in the estimate; if both vanish the squared distances
decay with rate $\min(\lambda_m,\lambda_h)$.
:::

:::{div} feynman-prose
For two normalized laws the mass term vanishes. In particular, compare the
swarm's law conditioned on survival to its QSD at each finite time. Both are
probabilities on the same configuration space. The survival factor belongs in
the evolution that produced the entropy estimate, not in a fictitious
conditioning event at infinite time.
:::

:::{prf:theorem} QSD convergence at finite conditioning horizons
:label: thm-hk-convergence-conditional

Let $p_t=\mathcal L(S_t\mid\tau_\dagger>t)$ and let $\pi$ be its QSD.
Whenever the normalized QSD entropy estimate of
{prf:ref}`thm-kl-convergence-euclidean` supplies
$D_{\mathrm{KL}}(p_t\Vert\pi)\leq Ae^{-\lambda t}$,

$$
H^2(p_t,\pi)\leq Ae^{-\lambda t},\qquad
\mathrm{HK}^2(p_t,\pi)\leq Ae^{-\lambda t}.
$$

If the specified full reference law also has LSI constant $C$, then
$D^2(p_t,\pi)\leq(1+2C)Ae^{-\lambda t}$.
:::

:::{prf:proof}
Both laws have mass one. Apply the entropy inequalities and the pure-reaction
HK bound. These are estimates between conditional probability laws at time
$t$. They do not condition on survival forever: under a QSD with killing
rate $\lambda_\dagger>0$, survival to $t$ has probability
$e^{-\lambda_\dagger t}$ and infinite survival has probability zero.
:::

(sec-hk-density)=
## Density Estimates and Their Domains

:::{div} feynman-prose
The entropy argument already supplies the preceding convergence bounds.
Density estimates answer additional questions: whether a law has a density,
where it is positive, and how its values compare with a reference density.
Keep track of the set on which each estimate holds.

A positive Gaussian kernel gives a positive contribution everywhere, but its
lower bound becomes small far from its source. On a bounded observation set,
source mass and distance give an explicit lower bound. A global comparison to
a reference measure instead uses an order relation that the transition kernel
can propagate.
:::

:::{prf:definition} Domination by a reference measure
:label: ax-bounded-density-ratio-rigorous

For probability measures $\mu,\pi$, write $\mu\leq M\pi$ if
$\mu(A)\leq M\pi(A)$ for every measurable set $A$. This is equivalent to
$\mu\ll\pi$ and $d\mu/d\pi\leq M$ almost everywhere. A pointwise bound on a
density relative to Lebesgue measure is a different assertion.
:::

:::{prf:theorem} Propagation of density domination
:label: thm-uniform-density-bound-hk

If $P$ is a conservative kernel with $\pi P=\pi$ and $\mu\leq M\pi$, then
$\mu P^j\leq M\pi$ for every $j$. For a killed kernel $Q$ with
$\pi Q=\alpha\pi$ and $\mu Q^j1>0$,

$$
\frac{\mu Q^j}{\mu Q^j1}\leq
\frac{M\alpha^j}{\mu Q^j1}\,\pi.
$$

If additionally $\mu\geq m\pi$, $m>0$, the normalized bound is $M/m$.
:::

:::{prf:proof}
Positive kernels preserve order. Iterate $\mu Q^j\leq M\alpha^j\pi$ and
normalize. The lower order bound gives $\mu Q^j1\geq m\alpha^j$.
For $P$, take $\alpha=1$ and use conservation.
:::

:::{div} feynman-prose
Velocity noise reaches position through transport. Differentiating the
transport coefficient $v$ in a noisy velocity direction produces a position
direction. The bracket calculation records this mechanism exactly. Smoothness
from that local mechanism still has to be distinguished from a quantitative
kernel comparison on a prescribed region.
:::

:::{prf:lemma} Kinetic bracket calculation
:label: lem-hormander-bracket

For constant nondegenerate velocity noise and smooth drift, put
$X_i=\partial_{v_i}$ and
$Y=v\cdot\nabla_x-(\gamma v+\nabla U)\cdot\nabla_v$.
Then $[X_i,Y]=\partial_{x_i}-\gamma\partial_{v_i}$; the noise fields and
these brackets span the phase tangent space.
:::

:::{prf:proof}
Differentiate the coefficients of $Y$ with respect to $v_i$;
$X_i$ has constant coefficients. The resulting $2d$ fields span because the
$x$ components of the brackets form the identity matrix. The local
hypoelliptic consequence under smooth coefficients is stated and applied in
{prf:ref}`thm-uniqueness-hypoelliptic-regularity`.
:::

:::{prf:theorem} A kernel comparison estimate
:label: thm-parabolic-harnack

If transition densities at two specified points and positive times satisfy
$k_s(z_0,y)\leq C k_t(z_1,y)$ for almost every $y$, then for every
nonnegative datum $f$,
$\int k_s(z_0,y)f(y)dy\leq C\int k_t(z_1,y)f(y)dy$.
For a compact set of arguments, such a constant follows from a finite upper
bound on the first kernel and a strictly positive lower bound on the second.
:::

:::{prf:proof}
Multiply the kernel inequality by $f\geq0$ and integrate. The
compact-set statement follows by dividing the two extrema. Local
hypoellipticity alone does not supply a global comparison constant on an
unbounded phase space or across an absorbing boundary.
:::

:::{prf:lemma} Upper density bound from a transition kernel
:label: lem-linfty-full-operator

If a sub-Markov kernel has density $k(z,y)\leq K(y)$ and its survival mass
from $\mu$ is at least $s>0$, then the normalized output has density at most
$K(y)/s$. A mixture containing a singular unsmoothed branch need not have a
Lebesgue density.
:::

:::{prf:proof}
Integrate $k(z,y)$ against the probability measure $\mu(dz)$ and
divide by the output mass. For the second assertion, a positive coefficient
of a point mass remains singular under addition of an absolutely continuous
measure.
:::

:::{div} feynman-prose
To get a numerical lower bound, reserve some source mass in a specified set
$C$. If the observation point remains at a controlled distance from $C$, every
piece of that reserved mass contributes at least the same Gaussian value.
The source mass and distance remain in the bound because either can make the
observed density small.
:::

:::{prf:lemma} Gaussian lower bound on specified bounded sets
:label: lem-gaussian-kernel-lower-bound

For $g_\sigma(y-x)=(2\pi\sigma^2)^{-D/2}e^{-|y-x|^2/(2\sigma^2)}$ and sets
$K,C$ with $\sup_{y\in K,x\in C}|y-x|\leq R$, every positive finite measure
$\mu$ satisfies

$$
(g_\sigma*\mu)(y)\geq
\mu(C)(2\pi\sigma^2)^{-D/2}e^{-R^2/(2\sigma^2)},\qquad y\in K.
$$
:::

:::{prf:proof}
Restrict the convolution integral to $C$ and use the displayed
pointwise Gaussian lower bound. The constant keeps both the source mass
$\mu(C)$ and the distance $R$.
:::

:::{prf:lemma} Positivity of a Gaussian cloning contribution
:label: lem-strict-positivity-cloning

If a transition has a positive Gaussian cloning contribution of total mass
$w>0$, that contribution has a strictly positive smooth density everywhere
in the ambient Euclidean space. Its derivatives are bounded by $w$ times
the corresponding Gaussian derivative suprema. This statement describes
the contribution, not an unsmoothed remainder of the full transition.
:::

:::{prf:proof}
Gaussian values are positive, so their integral against a nonzero
positive measure is positive. Each derivative of a fixed Gaussian is a
polynomial times a Gaussian and is bounded. Differentiate under the integral
using this bound.
:::

:::{prf:lemma} Positive QSD density from a positive transition density
:label: lem-qsd-strict-positivity

If $Q^j$ has positive density $k_j(z,y)$ and $\pi Q^j=\alpha^j\pi$, then
$\pi$ has density $\alpha^{-j}\int k_j(z,y)\pi(dz)>0$ almost everywhere.
A continuous positive version has a positive minimum on each compact set.
:::

:::{prf:proof}
Use the eigenmeasure identity and Tonelli. Continuity plus
positivity on a compact set gives a positive minimum. A probability density
cannot have a strictly positive global lower bound on a set of infinite
Lebesgue volume, since its integral would then be infinite.
:::

:::{prf:theorem} Local density ratios and global reference domination
:label: thm-bounded-density-ratio-main

If $p_t(y)\leq B_K$ and $q(y)\geq b_K>0$ on a specified set $K$, then
$p_t/q\leq B_K/b_K$ there. A global time-uniform ratio follows instead from
the reference-domination hypotheses of
{prf:ref}`thm-uniform-density-bound-hk`, or from separately proved global
weighted density estimates with matching tails.
:::

:::{prf:proof}
Divide the local bounds. For the global order statement apply the
cited kernel proof. Compact-set bounds cannot be extended globally by
replacing their positive minimum with an infimum that may be zero.
The entropy proof of {prf:ref}`thm-hk-convergence-main-assembly` does not
require this ratio estimate.
:::

(sec-hk-linearization)=
## Linearization and Stability Estimates

:::{div} feynman-prose
Near an equilibrium, split the equation into its first-order response and a
quadratic remainder. For a normalized killed equation, normalization itself
contributes to both pieces. Omitting it would mean linearizing a different
evolution.

The remaining estimates separate three tasks. A semigroup bound controls the
linear dynamics, a perturbation bound shows how much of that decay survives,
and a smoothing estimate converts convergence in an integral norm into
pointwise control. Each task uses a different hypothesis.
:::

:::{prf:lemma} Linearization on the correct mass space
:label: lem-linearization-qsd

Let $F$ be twice Fréchet differentiable on a neighborhood of an equilibrium
$u_*$ in a Banach space, with $F(u_*)=0$ and $\|D^2F\|\leq M$ there. Then

$$
F(u_*+h)=DF(u_*)h+R(h),\qquad \|R(h)\|\leq\tfrac M2\|h\|^2.
$$

For a normalized killed equation with bounded killing $c$,
$F(f)=A^*f-cf+(\int cf)f$, its derivative on zero-mass perturbations is

$$
DF(q)h=A^*h-ch+(\int cq)h+(\int ch)q,
$$

and its remainder is $(\int ch)h$.
:::

:::{prf:proof}
Twice integrate the derivative along $u_*+sh$, $0\leq s\leq1$:
$R(h)=\int_0^1(1-s)D^2F(u_*+sh)[h,h]ds$.
For the normalized killed equation, multiply out its quadratic normalization
term. The QSD solves $A^*q-cq=-(\int cq)q$, so $F(q)=0$.
For an alive measure of varying mass, a normalization expansion uses the
signed increment $\int h$, rather than $\|h\|_1$.
:::

:::{prf:lemma} Linearized decay with an explicit bounded perturbation
:label: lem-linearized-spectral-gap

Suppose the zero-mass semigroup of $A$ satisfies
$\|T_t\|\leq K e^{-at}$ and a mass-preserving bounded perturbation $B$ has
norm at most $b$. Then the perturbed semigroup obeys
$\|S_t\|\leq K e^{-(a-Kb)t}$. It decays when $a>Kb$.
:::

:::{prf:proof}
The variation-of-constants formula gives
$\|S_th\|\leq Ke^{-at}\|h\|+Kb\int_0^te^{-a(t-s)}\|S_sh\|ds$.
Multiply by $e^{at}$ and apply Grönwall. For the full nonlinear mean-field
flow the corresponding global Lipschitz estimate and attraction theorem
are proved in {prf:ref}`thm-uniqueness-uniqueness-stationary-solution`.
A finite-particle QSD mixing estimate is not itself this linearized
mean-field operator estimate.
:::

:::{prf:lemma} Boundedness of nonlocal integral terms
:label: lem-relative-boundedness-nonlocal

For $Bh(y)=\int k(x,y)h(x)dx-a(y)h(y)$, if
$\sup_x\int|k(x,y)|dy\leq K_1$ and $\|a\|_\infty\leq K_2$, then
$\|Bh\|_1\leq(K_1+K_2)\|h\|_1$. The complete derivative of a nonlinear
kernel includes its derivative with respect to the law, which requires its
own analogous bound.
:::

:::{prf:proof}
Apply Tonelli and the triangle inequality to the integral term,
then the multiplication-operator bound. The law-dependent derivative is
obtained by the product rule; its bound is not included in $K_1$ unless
explicitly verified.
:::

:::{prf:lemma} Local regularity under smooth lower-order sources
:label: lem-hypoellipticity-full-linearized

Suppose a kinetic operator $P$ has the local regularity implication
$Pu\in H^s_{\mathrm{loc}}\Rightarrow u\in H^{s+\delta}_{\mathrm{loc}}$
for some $\delta>0$, and a lower-order operator $B$ preserves every local
Sobolev class under consideration. A distributional solution of $Pu=Bu$
that starts in one such class is locally smooth.
:::

:::{prf:proof}
If $u\in H^s_{\mathrm{loc}}$, then $Bu$ belongs to that class;
the specified estimate gains $\delta$ derivatives. Iterate to obtain every
finite Sobolev order and use local Sobolev embedding. The kernel and
coefficient regularity needed for $B$ must hold at every iteration; an
arbitrary bounded operator on $L^1$ is not enough for this bootstrap.
:::

:::{div} feynman-prose
The last conversion has a simple time interpretation. Let the dynamics first
reduce the integral error, then reserve a fixed interval $\delta$ to smooth
what remains. The smoothing constant depends on that interval. To turn the
result into a density-ratio estimate on a set, one also needs a positive lower
bound for the reference density on that set.
:::

:::{prf:lemma} Fixed-time smoothing after convergence in an integral norm
:label: lem-l1-to-linfty-near-qsd

Let an evolution $\Phi_t$ fix $q$, satisfy
$\|\Phi_tf-q\|_1\leq Ke^{-at}\|f-q\|_1$, and have a fixed-time estimate
$\|\Phi_\delta f-\Phi_\delta g\|_\infty\leq C_\delta\|f-g\|_1$ on the
invariant class under study. Then for $t\geq\delta$,

$$
\|\Phi_tf-q\|_\infty\leq
C_\delta Ke^{-a(t-\delta)}\|f-q\|_1.
$$
:::

:::{prf:proof}
Apply the smoothing estimate to $\Phi_{t-\delta}f$ and $q$, then
the integral-norm estimate. A corresponding density-ratio estimate follows
on each set where $q$ has a known positive lower bound. Its constants retain
that set dependence.
:::

(sec-hk-conclusion)=
## Consequences for the Volume

:::{div} feynman-prose
The main bound combines the mass balance with entropy convergence for the
same measure and reference. Its uniformity in particle number is inherited
from those inputs. The empirical measure of a simulation uses the separate
transport and observable estimates, retaining the sampling error. This is how
the mass, entropy, and finite-particle analyses fit together.
:::

:::{prf:theorem} Convergence of mass, shape, and transport
:label: thm-hk-summary

The mass estimate {prf:ref}`lem-mass-contraction-revival-death` and the entropy
estimates in {doc}`15_kl_convergence`, whenever they apply to the same alive
law and reference, imply the quantitative bound of
{prf:ref}`thm-hk-convergence-main-assembly`. Its constants are independent of
$N$ when the input mass lower bound, LSI constant, entropy rates, and defect
bounds are independent of $N$.

For normalized QSD laws, {prf:ref}`thm-hk-convergence-conditional` gives
Hellinger and canonical HK convergence directly. For finite empirical
measures, the transport and observable bounds in
{doc}`13_quantitative_error_bounds` retain their sampling error.
:::

:::{prf:proof}
Substitute the proved input inequalities into the mass-shape
identity and the entropy-to-transport bound. No step in that substitution
introduces a new population factor. The empirical statement follows from
{prf:ref}`prop-empirical-wasserstein-concentration` and
{prf:ref}`lem-lipschitz-observable-error`.
:::
