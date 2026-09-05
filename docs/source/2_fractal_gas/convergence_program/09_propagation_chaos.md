# Mean-Field Existence, Uniqueness, and Propagation of Chaos

:::{div} feynman-prose
A large swarm has two kinds of randomness. A walker fluctuates around the
population distribution, and the population distribution itself can fluctuate
from run to run. Exchangeability says that the labels carry no information.
Propagation of chaos says more: the second kind of randomness disappears, and
any fixed number of walkers become independent in the limit.

We develop the analysis in that order. First we control empirical measures and
the nonlinear measurement pipeline. Then we construct the continuous evolution
by a positive, mass-preserving iteration, and prove uniqueness and attraction
when kinetic mixing dominates the interaction bound. Finally we connect
finite-swarm QSDs to that evolution. This last step retains both survival
normalization and the distinction between a fixed numerical timestep and a
continuous-time limit.
:::

(sec-chaos-population-laws)=
## 1. Swarm Laws, Empirical Measures, and Tightness

:::{div} feynman-prose
The correct object to follow is the empirical measure of a whole swarm. Its
expectation is a one-walker marginal, but an expectation can hide a mixture.
For example, half the runs could put every walker near one point and half near
another. Every label then has the same marginal, while the walkers remain
strongly correlated.
:::

:::{prf:definition} Finite-swarm QSDs and marked marginals
:label: def-sequence-of-qsds

Let $\mathsf Z$ be the Polish one-walker state space used by the specified
model. It includes the alive indicator and any retained dead-walker coordinates
that influence revival. For a model observed immediately after a proved
all-alive revival step, that observation space may instead contain only
position and velocity.

For each $N\ge2$, let $Q_N$ be the killed full-swarm kernel on its noncemetery
space, and suppose a QSD $\nu_N$ has been established by an applicable theorem
of {doc}`06_convergence` or the entropy analysis. Write

$$
\nu_NQ_N=\alpha_N\nu_N,\qquad 0<\alpha_N\le1.
$$

For a swarm $S=(z_1,\ldots,z_N)$ define

$$
L_N(S)=\frac1N\sum_{i=1}^N\delta_{z_i},\qquad
\mu_N=(\operatorname{proj}_1)_\#\nu_N,\qquad
\Lambda_N=(L_N)_\#\nu_N.
$$

Here $\mu_N$ is a probability on $\mathsf Z$, while $\Lambda_N$ is a
probability on $\mathcal P(\mathsf Z)$. Propagation of chaos toward $\mu_*$
means that, for every fixed $l$, the first $l$ coordinate marginal converges
weakly to $\mu_*^{\otimes l}$.
:::

:::{prf:definition} Alive mass and normalized alive distribution
:label: def-sequence-of-qsds-summary

For a marked probability $\mu$, let $f=\mu|_{\{s=1\}}$ be its alive
subprobability and $m_a=f(\mathsf Z)$. If $m_a>0$, its normalized alive law is
$\rho=f/m_a$. The finite-swarm counterparts are

$$
f_N=\frac1N\sum_{i:s_i=1}\delta_{(x_i,v_i)},\qquad
m_{a,N}=\frac{k_N}{N},\qquad \rho_N=\frac{f_N}{m_{a,N}}.
$$

The mass-one marked law, alive subprobability, and normalized alive law have
different normalization equations. A scalar dead reservoir is a closed model
only when the revival rule depends on dead walkers through that scalar alone.
Companion selection that depends on their retained coordinates requires their
distribution in the state.
:::

:::{prf:lemma} Exchangeability from kernel symmetry
:label: lem-exchangeability

If $Q_N$ commutes with every permutation of walker labels and has a unique
QSD, then $\nu_N$ is exchangeable. Consequently
$\mathbb E_{\nu_N}L_N\varphi=\mu_N\varphi$ for integrable $\varphi$.
:::

:::{prf:proof}
For a permutation $\sigma$, the commutation identity implies that
$\sigma_\#\nu_N$ satisfies the same QSD equation as $\nu_N$. It is a
probability, so uniqueness gives $\sigma_\#\nu_N=\nu_N$. All coordinate
marginals therefore coincide, and averaging their expectations proves the
last assertion. Symmetry must be checked for the actual collision update,
including overlapping recipient groups and their ordering; it is not supplied
by identical parameter values alone.
:::

:::{prf:theorem} Tightness from a uniform confining moment
:label: thm-qsd-marginals-are-tight

Suppose $\nu_N$ is exchangeable and there is a nonnegative lower
semicontinuous function $W$ on $\mathsf Z$ with compact sublevel sets such that

$$
\sup_N\mathbb E_{\nu_N}L_NW\le C_W<\infty.
$$

Then the sequence $\{\mu_N\}$ is tight. The moment hypothesis follows, for
example, from integrable swarm Lyapunov functions satisfying

$$
L_NW\le aV_N+b,\qquad Q_NV_N\le r_NV_N+B_N,
$$

with $a,b$ fixed, $\sup_NB_N<\infty$, and
$\inf_N(\alpha_N-r_N)>0$.
:::

:::{prf:proof}
Exchangeability gives $\mu_NW=\mathbb E_{\nu_N}L_NW\le C_W$. Therefore

$$
\mu_N(\{W>R\})\le C_W/R.
$$

The compact set $\{W\le R\}$ contains at least $1-C_W/R$ of every marginal.
This is tightness; Prokhorov's theorem then gives weakly convergent
subsequences on the Polish space.

For the stated Lyapunov criterion,
{prf:ref}`thm-equilibrium-variance-bounds` gives
$\nu_NV_N\le B_N/(\alpha_N-r_N)$. Substitute this into the domination of
$L_NW$ to obtain the uniform moment bound.
:::

:::{prf:corollary} Tightness of the random empirical measures
:label: thm-qsd-marginals-are-tight-summary

Under the preceding moment hypothesis, $\{\Lambda_N\}$ is tight in
$\mathcal P(\mathcal P(\mathsf Z))$.
:::

:::{prf:proof}
For $R>0$, the set
$\mathcal K_R=\{\mu\in\mathcal P(\mathsf Z):\mu W\le R\}$ is tight by the
same compact-sublevel estimate. It is closed under weak convergence by lower
semicontinuity, so it is compact by Prokhorov's theorem. Markov's inequality
now gives
$\Lambda_N(\mathcal K_R^c)\le C_W/R$.
:::

:::{prf:remark} The confining observable on an open or unbounded domain
:label: rem-chaos-confining-observable

Centred swarm variance alone does not bound the spatial barycentre. On an
unbounded domain, $W$ must use the established confining position envelope.
On an open bounded domain, a Euclidean ball intersected with that domain need
not be compact in its interior topology; control near the excluded boundary
may also be needed. The integrable barrier analysis in {doc}`03_cloning`
provides the corresponding moment inputs where its hypotheses hold.
Tightness permits atomic limits. Absolute continuity or regularity of a limit
requires a smoothing or equation argument.
:::

(sec-fg-propagation-intro-limit-point)=
## 2. Empirical Convergence and the Measurement Pipeline

:::{div} feynman-prose
Once a population law converges, bounded continuous measurements of it converge
too. The regularizers in the algorithm keep divisions stable. There are two
points to watch: the companion law is weighted by its kernel, and fitness is a
nonlinear function of a sampled distance. Replacing that sampled distance by
its expectation before evaluating fitness changes the quantity being averaged.
:::

:::{prf:lemma} Empirical measures, mixtures, and chaos
:label: lem-empirical-convergence

For exchangeable $\nu_N$, if $\Lambda_N\Rightarrow\Lambda$, then every fixed
$l$-coordinate marginal converges to

$$
\int_{\mathcal P(\mathsf Z)}\rho^{\otimes l}\,\Lambda(d\rho).
$$

In particular, $\Lambda_N\Rightarrow\delta_\rho$ implies chaos toward
$\rho$. Conversely, convergence of the first two coordinate marginals to
$\rho$ and $\rho^{\otimes2}$ implies
$\Lambda_N\Rightarrow\delta_\rho$ whenever the empirical laws are tight.
Removing one tagged walker does not change the empirical limit, since for
$\|\varphi\|_\infty\le1$,

$$
\left|\frac1{N-1}\sum_{j\ne i}\varphi(z_j)-L_N\varphi\right|\le\frac2N.
$$

For a companion population of $k_N$ alive walkers, the same estimate is
$2/k_N$ when the excluded tag is alive. A positive alive-fraction bound makes
this $O(1/N)$; the full-population formula does not replace that normalization.
:::

:::{prf:proof}
For $\|\varphi_j\|_\infty\le1$, the product
$\prod_{j=1}^lL_N\varphi_j$ averages over sampling indices with replacement.
Conditioned on all sampled indices being distinct, exchangeability gives the
expectation of $\prod_{j=1}^l\varphi_j(z_j)$. The probability of a repeated
index is at most $l(l-1)/(2N)$, so the two expectations differ by at most
$l(l-1)/N$. The map
$\eta\mapsto\prod_j\eta\varphi_j$ is bounded continuous for bounded
continuous $\varphi_j$. Pass to the limit in its expectation to obtain the
mixture formula. Product test functions determine probability measures on
finite products; tightness allows the usual approximation on compact sets.

For the converse, exchangeability gives

$$
\mathbb E(L_N\varphi)^2
=\frac1N\mu_N(\varphi^2)
 +\frac{N-1}{N}\nu_N^{(2)}(\varphi\otimes\varphi).
$$

The assumed first and second marginal limits therefore imply
$\mathbb E|L_N\varphi-\rho\varphi|^2\to0$ for every bounded continuous
$\varphi$. Apply this to a countable convergence-determining family and use
tightness to identify every empirical-law limit as $\delta_\rho$.
For the final inequality, subtract the two finite averages and bound the
tagged summand and the mean by one.
The identical calculation with $k_N$ proves the alive-population version.
:::

:::{prf:remark} A symmetric mixture need not become independent
:label: rem-chaos-exchangeability-mixture

For distinct points $a,b$, the law
$\nu_N=\tfrac12\delta_a^{\otimes N}+\tfrac12\delta_b^{\otimes N}$ is
exchangeable for every $N$. Its first marginal is always
$\tfrac12(\delta_a+\delta_b)$, while its empirical law is always
$\tfrac12\delta_{\delta_a}+\tfrac12\delta_{\delta_b}$. The empirical law
therefore remains random. No infinite exchangeable extension or law of large
numbers can turn this mixture into the product of its first marginal.
:::

:::{prf:lemma} Continuity of reward moments
:label: lem-reward-continuity

If $\rho_n\Rightarrow\rho$ and $R$ is bounded continuous, then

$$
\rho_nR\longrightarrow\rho R,\qquad
\rho_nR^2-(\rho_nR)^2\longrightarrow\rho R^2-(\rho R)^2.
$$

For unbounded $R$, the same statement holds when the squared rewards are
uniformly integrable under $\rho_n$ and integrable under $\rho$.
:::

:::{prf:proof}
For bounded $R$, apply weak convergence to $R$ and $R^2$ and then continuity
of multiplication. For unbounded $R$, truncate with continuous cutoffs, pass
to the limit for the bounded functions, and let the cutoff grow. Uniform
integrability controls the discarded tails uniformly in $n$; Cauchy-Schwarz
also controls the first-moment tails.
:::

:::{prf:lemma} Continuity of companion-weighted distance moments
:label: lem-distance-continuity

Let $k(z,z')$ be a bounded continuous companion weight, and define

$$
K_\rho(z,dz')=\frac{k(z,z')\rho(dz')}{Z_\rho(z)},\qquad
Z_\rho(z)=\int k(z,z')\rho(dz'),\qquad
J_\rho(dz,dz')=\rho(dz)K_\rho(z,dz').
$$

Suppose the denominators are positive and bounded away from zero on compact
sets of $z$, uniformly along a weakly convergent sequence
$\rho_n\Rightarrow\rho$. Then $J_{\rho_n}\Rightarrow J_\rho$.
Consequently, for bounded continuous algorithmic distance $d$,

$$
J_{\rho_n}d\to J_\rho d,\qquad
J_{\rho_n}d^2-(J_{\rho_n}d)^2
 \to J_\rho d^2-(J_\rho d)^2.
$$

For unbounded $d$, require uniform integrability of $d^2$ under the joint
companion laws. Uniform companion selection is the special case $k=1$.
:::

:::{prf:proof}
For a bounded continuous test $g(z,z')$, the integrals
$\int k(z,z')g(z,z')\rho_n(dz')$ converge uniformly for $z$ in a fixed
compact set. To see this, restrict $z'$ to a compact set containing all but
arbitrarily small mass of the tight family $\{\rho_n,\rho\}$; uniform
continuity on the product compact gives a finite net in $z$. Weak convergence
handles its finitely many points, and boundedness controls the tail.
The same argument applies to $Z_{\rho_n}$. The denominator bound therefore
gives uniform convergence of their ratios on each compact set. Integrate the
ratio against $\rho_n$, using tightness again, to obtain convergence of
$J_{\rho_n}g$. Apply this to $d$ and $d^2$, or truncate their tails under the
stated integrability hypothesis.
:::

:::{prf:lemma} Explicit Lipschitz bounds for normalized measurements
:label: lem-uniqueness-lipschitz-moments

Use $\|\cdot\|_1$ for the full variation norm of signed measures or the
$L^1$ norm of densities. If $|R|\le M_R$, then

$$
|\rho R-\eta R|\le M_R\|\rho-\eta\|_1,\qquad
|\operatorname{Var}_\rho R-\operatorname{Var}_\eta R|
\le3M_R^2\|\rho-\eta\|_1.
$$

If $0\le k\le1$ and $Z_\rho,Z_\eta\ge a>0$ uniformly, then

$$
\sup_z\|K_\rho(z,\cdot)-K_\eta(z,\cdot)\|_1
\le\frac2a\|\rho-\eta\|_1,
\qquad
\|J_\rho-J_\eta\|_1\le(1+2/a)\|\rho-\eta\|_1.
$$

The corresponding distance mean and variance constants are
$M_D(1+2/a)$ and $3M_D^2(1+2/a)$ when $|d|\le M_D$.
For alive subprobabilities of mass at least $m_*>0$,

$$
\left\|\frac f{\int f}-\frac g{\int g}\right\|_1
\le\frac2{m_*}\|f-g\|_1.
$$
:::

:::{prf:proof}
The mean estimate is the dual bound for integration. The second moment
difference is at most $M_R^2\|\rho-\eta\|_1$, while the difference of the
squared means is at most $2M_R^2\|\rho-\eta\|_1$.
For a fixed $z$, add and subtract $k(z,\cdot)\eta/Z_\rho(z)$. The numerator
difference contributes at most $\|\rho-\eta\|_1/Z_\rho(z)$ and the
normalization difference contributes at most the same quantity. For $J$,
also change its first marginal, which contributes
$\|\rho-\eta\|_1$. The distance estimates follow by the reward calculation
on $J$. The alive normalization bound follows from the same add-and-subtract
argument and $|\int f-\int g|\le\|f-g\|_1$.
:::

:::{prf:lemma} Lipschitz continuity of the actual regularized fitness pipeline
:label: lem-uniqueness-lipschitz-fitness-potential

On a class where the preceding measurement bounds hold, suppose the fixed
regularized standard-deviation map $s$ has lower bound $s_*>0$ and Lipschitz
constant $L_s$ on the attained variance range. Let the rescale map $g$ be
Lipschitz and have range in $[g_*,g^*]\subset(0,\infty)$. For fixed
$\alpha,\beta\ge0$, the sampled fitness

$$
V_\rho(z,z')=
 g\!\left(\frac{R(z)-\mu_R[\rho]}{s(\sigma_R^2[\rho])}\right)^\alpha
 g\!\left(\frac{d(z,z')-\mu_D[\rho]}{s(\sigma_D^2[\rho])}\right)^\beta
$$

is bounded and Lipschitz as a function of $\rho$ in the uniform norm over
$(z,z')$. These are the multiplicative fitness operations of
{prf:ref}`def-latent-fractal-gas-fitness`, with moments taken under the
specified companion law. The expected fitness is obtained by integrating
this sampled quantity against that law.
:::

:::{prf:proof}
For bounded raw measurement $u$ and two means and scales,

$$
\left|\frac{u-m_1}{s_1}-\frac{u-m_2}{s_2}\right|
\le\frac{|m_1-m_2|}{s_*}
 +\frac{|u-m_2|}{s_*^2}|s_1-s_2|.
$$

The preceding lemma bounds the mean and variance differences; Lipschitz
continuity of $s$ bounds the scale difference. Apply the Lipschitz bound for
$g$. The maps $x\mapsto x^\alpha$ and $x\mapsto x^\beta$ have finite
Lipschitz constants on $[g_*,g^*]$, including the constant map when an exponent
is zero. Finally, expand the difference of the two products. Integrating
against $K_\rho$ adds its Lipschitz contribution, bounded by the fitness
supremum times $\|K_\rho-K_\eta\|_1$.

When cloning probability is nonlinear in sampled fitness, its mean must be
computed by integrating that probability over the sampled fitness variables.
Evaluating it at expected fitness is a different operation, as specified in
{prf:ref}`rem-mean-field-fitness-field-latent`.
:::

:::{prf:lemma} Uniform integrability and passage to expectations
:label: lem-uniform-integrability

If random variables $Y_n$ converge in distribution to $Y$, their absolute
values are uniformly integrable, and $Y$ is integrable, then
$\mathbb EY_n\to\mathbb EY$. In particular, this applies to bounded
continuous functions of a convergent random empirical measure. A sufficient
uniform-integrability condition is
$\sup_n\mathbb E|Y_n|^{1+\epsilon}<\infty$ for some $\epsilon>0$.
:::

:::{prf:proof}
Clip $Y_n$ to $[-R,R]$. Expectations of the clipped variables converge by
weak convergence. The difference from their original expectations is bounded
by $\mathbb E[|Y_n|1_{\{|Y_n|>R\}}]$, uniformly small as $R\to\infty$.
The same tail vanishes for $Y$. For the sufficient condition, this tail is at
most $R^{-\epsilon}\mathbb E|Y_n|^{1+\epsilon}$.
:::

(sec-chaos-positive-evolution)=
## 3. Positive Mean-Field Evolution and Function Spaces

:::{div} feynman-prose
The gain part of cloning puts mass at new states; its loss part removes mass
from old states. Their integrals agree. This simple accounting gives a direct
construction of the evolution. It avoids asking a degenerate kinetic operator
for an elliptic estimate that it does not have.
:::

:::{prf:definition} A specified continuous mean-field equation
:label: def-chaos-gain-loss-model

Let $T_t$ be a strongly continuous positive, mass-preserving contraction
semigroup on $X=L^1(\mathsf Z,\mathfrak m)$, with generator $A$. The reference
measure $\mathfrak m$ may combine Lebesgue measure on continuous coordinates
and counting measure on status coordinates. Consider

$$
\partial_tu=Au+\mathcal R(u),\qquad u\ge0,\quad\int u=1.
$$

Assume on the probability densities that

$$
\mathcal R(u)=G(u)-a(u)u,\qquad
G(u)\ge0,\quad0\le a(u)\le\Lambda,
\quad\int\mathcal R(u)=0,
$$

and $\mathcal R$ is Lipschitz in $L^1$ with constant $L_{\mathcal R}$.
Then $\|\mathcal R(u)\|_1\le2\Lambda$.

This is the continuous model selected in {doc}`08_mean_field` when its
transport law, reaction kernels, and dead-mass bookkeeping satisfy these
conditions. Killing of an individual walker can be a transition to its dead
state; whole-swarm absorption is a different event. Failed jitter proposals
must be represented in the dead component, as in
{prf:ref}`rem-mean-field-cloning-boundary-loss`. The definition does not change
a killed transport operator into a reflecting one or identify a fixed BAOAB
step with this generator.
:::

:::{prf:lemma} A complete cloning gain-loss Lipschitz bound
:label: lem-uniqueness-lipschitz-cloning-operator

Suppose the reaction is represented by a finite nonnegative jump kernel
$B_u(z,dy)$ with

$$
B_u(z,\mathsf Z)\le\Lambda,\qquad
\sup_z\|B_u(z,\cdot)-B_v(z,\cdot)\|_1
\le L_B\|u-v\|_1.
$$

Let $J_u(dz,dy)=u(z)\mathfrak m(dz)B_u(z,dy)$. Set
$\mathcal R(u)=\operatorname{proj}_{y\#}J_u-
\operatorname{proj}_{z\#}J_u$, assuming its signed measure has an $L^1$
density. Then $\mathcal R$ has the gain-loss structure above and

$$
\|\mathcal R(u)-\mathcal R(v)\|_1
\le2(\Lambda+L_B)\|u-v\|_1.
$$

Companion normalization, fitness, and probability clipping contribute to
$L_B$ through the preceding lemmas. A retained finite swarm collision group
requires the limiting jump kernel for that group; a one-particle replacement
kernel cannot be substituted without identifying that limit.
:::

:::{prf:proof}
The second projection of $J_u$ is nonnegative; its first projection equals
$u(z)B_u(z,\mathsf Z)\mathfrak m(dz)$. Both have the same total mass, at
most $\Lambda$. For the difference of $J_u$ and $J_v$, first change the
input density and then the jump kernel. This gives

$$
\|J_u-J_v\|_1\le\Lambda\|u-v\|_1+L_B\|u-v\|_1.
$$

Pushforward of a signed measure contracts its full variation norm. Apply
this to both projections and add their bounds.
:::

:::{prf:theorem} Global positive mild solutions
:label: thm-chaos-mild-wellposedness

Under {prf:ref}`def-chaos-gain-loss-model`, every initial probability density
$u_0\in X$ has a unique global mild solution $u_t\in X$. It remains a
probability density, depends continuously on $u_0$, and defines a semigroup
$\mathcal S_t$. In particular,

$$
\|\mathcal S_tu_0-\mathcal S_tv_0\|_1
\le e^{L_{\mathcal R}t}\|u_0-v_0\|_1.
$$
:::

:::{prf:proof}
For $\Lambda>0$, use the equivalent damped integral equation

$$
u_t=e^{-\Lambda t}T_tu_0
 +\int_0^te^{-\Lambda(t-s)}T_{t-s}
 [\mathcal R(u_s)+\Lambda u_s],ds.
$$

The integrand in brackets is
$G(u_s)+(\Lambda-a(u_s))u_s\ge0$ and has mass $\Lambda$. Thus the right
side has mass $e^{-\Lambda t}+\int_0^t\Lambda e^{-\Lambda(t-s)}ds=1$ and
is nonnegative for every continuous probability-valued input path.

The space of continuous probability-density paths on $[0,t_0]$ is closed in
$C([0,t_0],L^1)$ and therefore complete. The displayed map contracts its
supremum norm when

$$
\frac{L_{\mathcal R}+\Lambda}{\Lambda}
 (1-e^{-\Lambda t_0})<1.
$$

Choose such $t_0$ and apply the contraction mapping theorem. The same
$t_0$ works at every restart, so the solution extends for all time and remains
positive and normalized. For $\Lambda=0$, the reaction is zero and
$u_t=T_tu_0$.

Undoing the damping gives the ordinary mild equation
$u_t=T_tu_0+\int_0^tT_{t-s}\mathcal R(u_s)ds$. Subtract two such equations
and apply Gronwall's inequality to obtain the stability bound. Uniqueness
also gives $\mathcal S_{t+s}=\mathcal S_t\mathcal S_s$.
:::

:::{prf:corollary} Extension to singular initial laws through kinetic smoothing
:label: cor-chaos-measure-initial-data

In addition to the global gain-loss assumptions, suppose $T_t$ is a Markov
semigroup on probabilities with $T_t\mu\Rightarrow\mu$ as $t\downarrow0$.
Assume that for every $t>0$ it has densities $p_t(z,\cdot)$ with respect to
$\mathfrak m$, and that $z\mapsto p_t(z,\cdot)$ is continuous in $L^1$.
Then the nonlinear evolution extends uniquely to every initial probability
$\mu_0$. It has an $L^1$ density at each positive time and is weakly continuous
in the initial probability. If the zero-mass mixing condition in
{prf:ref}`thm-uniqueness-uniqueness-stationary-solution` holds, its attraction
conclusion extends to these initial probabilities.
:::

:::{prf:proof}
For $t>0$, replace $T_tu_0$ in the damped iteration by the density of
$T_t\mu_0$. Perform the same contraction on essentially bounded measurable
probability-density paths on $(0,t_0]$, with the supremum $L^1$ metric.
Positivity and mass are unchanged, and the same constant gives contraction.
The integral formula has a representative continuous in $L^1$ away from zero:
split the integral away from its upper endpoint, use strong continuity of
$T_t$ on $L^1$, and bound the remaining interval by its length times the
bounded reaction source. Its weak limit at zero is $\mu_0$, because the
reaction integral is $O(t)$ in $L^1$ and $T_t\mu_0\Rightarrow\mu_0$.
Restarting at positive times gives the unique global evolution.

If $\mu_n\Rightarrow\mu$, then
$a_n(t):=\|T_t\mu_n-T_t\mu\|_1\to0$ for every fixed $t>0$.
Indeed, tightness restricts the initial states to a compact set up to a
uniformly small tail; the continuous $L^1$-valued kernel on that compact set
can be uniformly approximated by finitely many values with continuous
partition weights. Weak convergence applies to those finitely many weights.
For the nonlinear solutions, the mild equation gives

$$
d_n(t)\le a_n(t)+L_{\mathcal R}\int_0^td_n(s)ds,
\qquad d_n(t)=\|\mathcal S_t\mu_n-\mathcal S_t\mu\|_1.
$$

The integral version of Gronwall bounds this by
$a_n(t)+L_{\mathcal R}\int_0^te^{L_{\mathcal R}(t-s)}a_n(s)ds$.
Since $a_n\le2$, dominated convergence proves $d_n(t)\to0$.
For attraction, first evolve an arbitrary initial measure for any
$\epsilon>0$ to obtain a density, then apply the density attraction theorem
from that time. Its initial distance to the stationary density is at most
two.
:::

:::{prf:remark} Localization at positive alive mass
:label: rem-chaos-positive-alive-localization

For kernels using $f/m_a$, the Lipschitz constants are uniform on
$m_a\ge m_*>0$. Local existence follows from the same iteration in a small
$L^1$ neighbourhood of an initial law with positive alive mass. Choose the
time interval so that the image stays in that neighbourhood; the reaction
bound $2\Lambda$ and strong continuity of $T_tu_0$ make this possible.

Continuation is global when an a priori alive-mass lower bound stays positive
on every finite interval and the other Lipschitz and moment constants remain
bounded there. For the mass-neutral cloning, bounded-killing model with
normalized revival in {doc}`08_mean_field`,
{prf:ref}`cor-mean-field-positive-alive-mass` supplies this lower bound.
No arbitrary value of $f/m_a$ at $m_a=0$ is required for those trajectories.
:::

:::{prf:definition} Weighted Sobolev analysis space
:label: def-uniqueness-weighted-sobolev-h1w

On an open continuous phase space $\Omega\subset\mathbb R^D$, let $w>0$ be
locally bounded above and locally bounded away from zero. Define the vector
space

$$
H^1_w(\Omega)=\left\{u:\int_\Omega(|u|^2+|\nabla u|^2)w<\infty\right\},
\qquad
\|u\|_{H^1_w}^2=\int_\Omega(|u|^2+|\nabla u|^2)w.
$$

Probability densities form a subset of this vector space. When normalization
is imposed using this norm, require
$C_w^2=\int_\Omega w^{-1}<\infty$. For example,
$w(z)=(1+|z|^2)^p$ has this property on $\mathbb R^D$ when $p>D/2$.
The lower power $1+|z|^2$ needs a separate integrability check on the chosen
domain. This weight is an analysis choice, not an algorithm parameter.
:::

:::{prf:theorem} Completeness of the weighted Sobolev space
:label: thm-uniqueness-completeness-h1w-omega

The space $H^1_w$ is Hilbert. If $C_w<\infty$, then
$\|u\|_1\le C_w\|u\|_{H^1_w}$.
:::

:::{prf:proof}
A Cauchy sequence has limits $u,g_1,\ldots,g_D$ in $L^2(w)$ for its functions
and first derivatives. Local lower bounds on $w$ imply convergence in
$L^2$ on each compact subset. For a compactly supported smooth test function
$\varphi$, pass to the limit in
$\int u_n\partial_i\varphi=-\int(\partial_i u_n)\varphi$ to obtain
$\partial_i u=g_i$ weakly. Thus the limit belongs to $H^1_w$ and convergence
holds in its norm. The norm comes from the displayed inner product.
Finally, Cauchy-Schwarz gives
$\int|u|\le(\int|u|^2w)^{1/2}(\int w^{-1})^{1/2}$.
:::

:::{prf:remark} Complete probability and bounded-ball constraint sets
:label: rem-uniqueness-completeness-constraint-set

When $C_w<\infty$, the set
$\mathcal P_w=\{u\in H^1_w:u\ge0,\ \int u=1\}$ is closed and complete.
Indeed, norm convergence gives local almost-everywhere convergence along a
subsequence, preserving positivity, and $L^1$ convergence preserves mass.
Its intersection with a closed norm ball is also complete. The analogous
probability set is closed in $L^1$ without a Sobolev weight. An $H^1_w$ bound
on existing fixed points alone does not establish that an entire ball maps
into itself; that mapping estimate is proved below when its hypotheses hold.
:::

(sec-fg-propagation-intro-uniqueness)=
## 4. Resolvents, Stationary Existence, and Uniqueness

:::{div} feynman-prose
A resolvent averages the kinetic evolution over an exponentially distributed
waiting time. It preserves positivity, and its effect on total mass is known
exactly. To obtain contraction, however, we must use the part of kinetic
mixing that removes differences of shape. Such differences have zero mass.
The unchanged mass direction explains why adding a large scalar shift to an
operator is not by itself a uniqueness proof.
:::

### 4.1. Positive resolvent and the stationary equation

:::{prf:lemma} Positive, mass-preserving fixed-point map
:label: lem-uniqueness-self-mapping

Under {prf:ref}`def-chaos-gain-loss-model`, choose $C\ge\Lambda$ with $C>0$.
Define

$$
R_Cg=\int_0^\infty e^{-Ct}T_tg\,dt,\qquad
\mathcal T_C(u)=R_C[Cu+\mathcal R(u)].
$$

Then $R_C=(C-A)^{-1}$ on $L^1$, $R_C$ preserves positivity,
$\|R_C\|_{1\to1}\le1/C$, and $\int R_Cg=(\int g)/C$.
The map $\mathcal T_C$ sends probability densities to probability densities.
Its fixed points are exactly the stationary solutions in the generator domain
of $Au+\mathcal R(u)=0$.
:::

:::{prf:proof}
The Bochner integral converges because $T_t$ is an $L^1$ contraction.
Positivity, the norm bound, and the mass identity follow by integrating the
corresponding properties of $T_t$.
The semigroup property gives

$$
T_hR_Cg=e^{Ch}\left(R_Cg-\int_0^he^{-Ct}T_tg\,dt\right).
$$

Subtract $R_Cg$, divide by $h$, and let $h\downarrow0$. Strong continuity
gives $AR_Cg=CR_Cg-g$, so $R_Cg$ is in the generator domain. Conversely,
for $u$ in that domain, integrate the derivative of $e^{-Ct}T_tu$ to obtain
$R_C(C-A)u=u$.

The source $Cu+\mathcal R(u)=G(u)+(C-a(u))u$ is nonnegative and has mass
$C$. Its resolvent image is therefore a probability density. Applying
$C-A$ to the fixed-point identity proves stationarity, and applying $R_C$ to
the stationary equation proves the converse.
:::

:::{prf:proposition} Normalized alive stationary states and the dead reservoir
:label: prop-chaos-alive-stationary-reconstruction

For the continuous model with normalized revival kernel $G_\rho$,
$\int G_\rho=1$, suppose its cloning operator is homogeneous in alive mass:
$S[m\rho]=mS[\rho]$. Write $\bar c[\rho]=\int c\rho$ and let
$\lambda_{\rm rev}>0$. Then a mass-one alive profile $\rho$ solves

$$
0=A\rho+S[\rho]-c\rho+\bar c[\rho]G_\rho
$$

if and only if the reconstructed pair

$$
m_a=\frac{\lambda_{\rm rev}}{\lambda_{\rm rev}+\bar c[\rho]},\qquad
m_d=\frac{\bar c[\rho]}{\lambda_{\rm rev}+\bar c[\rho]},\qquad
f=m_a\rho
$$

solves the stationary alive/dead equations with $m_a+m_d=1$. The nonlinear
term in the normalized equation is mass-neutral and has a bounded gain-loss
form when $c$ and the cloning loss rate are bounded.
:::

:::{prf:proof}
The stationary dead equation is
$m_a\bar c[\rho]=\lambda_{\rm rev}m_d$. Combine it with $m_a+m_d=1$ to
obtain the mass formulas. Divide the stationary alive equation by $m_a>0$,
using homogeneity of $S$ and
$\lambda_{\rm rev}m_d/m_a=\bar c[\rho]$. This gives the displayed normalized
equation. Conversely, multiply that equation by the reconstructed $m_a$.
The two reaction terms have opposite integrals; their gain and loss parts are
nonnegative. If revival or cloning proposals have subprobability mass in the
alive domain, use the full marked-state balance from {doc}`08_mean_field`
instead of this normalized-kernel specialization.
:::

### 4.2. Contraction on zero-mass differences

:::{prf:lemma} Resolvent constants and the conserved mass direction
:label: lem-uniqueness-scaling-hypoelliptic-constant

Suppose the chosen kinetic semigroup has a zero-mass mixing estimate

$$
\|T_tg\|_1\le K e^{-at}\|g\|_1,
\qquad \int g=0,\qquad K\ge1,\quad a>0.
$$

Then

$$
\|R_Cg\|_1\le\frac K{C+a}\|g\|_1\quad(\int g=0).
$$

On the full $L^1$ space, $\|R_C\|_{1\to1}=1/C$. In particular,
$CR_C$ cannot be a strict contraction on all densities. A decay law
$\|R_C\|\sim1/\sigma_v^2$ at fixed $C$ is incompatible with the mass
identity in that norm.
:::

:::{prf:proof}
Integrate the zero-mass estimate against $e^{-Ct}dt$. For any nonnegative
density of mass one, the positive image $R_Cu$ has $L^1$ norm $1/C$;
together with the upper bound from the preceding lemma this proves equality
of the full operator norm. If an invariant kinetic density $\pi$ exists,
this mass direction is also explicit: $R_C\pi=\pi/C$.
:::

:::{prf:theorem} Contraction of the stationary solution operator
:label: thm-uniqueness-contraction-solution-operator

Under the gain-loss assumptions and the zero-mass kinetic estimate above, if
some $C\ge\Lambda$, $C>0$, satisfies

$$
q_C:=\frac{K(C+L_{\mathcal R})}{C+a}<1,
$$

then $\mathcal T_C$ is a strict contraction on the complete $L^1$ space of
probability densities. It has a unique fixed point $u_*$, and

$$
\|\mathcal T_C^nu-u_*\|_1\le q_C^n\|u-u_*\|_1.
$$
:::

:::{prf:proof}
Both $u-v$ and $\mathcal R(u)-\mathcal R(v)$ have zero integral. Apply the
zero-mass resolvent estimate to

$$
\mathcal T_Cu-\mathcal T_Cv
=R_C[C(u-v)+\mathcal R(u)-\mathcal R(v)].
$$

Its norm is at most $q_C\|u-v\|_1$. The positive self-mapping property and
completeness were proved above, so the contraction mapping theorem gives the
fixed point and its iteration bound. The resolvent identity makes it a
stationary solution.
:::

:::{prf:theorem} Global attraction and a unique stationary mean-field law
:label: thm-uniqueness-uniqueness-stationary-solution

Under the global gain-loss assumptions, suppose
$\|T_tg\|_1\le Ke^{-at}\|g\|_1$ for zero-mass $g$ and

$$
b:=a-KL_{\mathcal R}>0.
$$

Then the nonlinear equation has a unique stationary probability density
$u_*$ and

$$
\|\mathcal S_tu-u_*\|_1\le K e^{-bt}\|u-u_*\|_1
$$

for every initial probability density. This conclusion does not require the
stronger shifted-resolvent condition $q_C<1$.
:::

:::{prf:proof}
Subtract the two ordinary mild equations. Every source difference has zero
mass, so

$$
\|\mathcal S_tu-\mathcal S_tv\|_1
\le Ke^{-at}\|u-v\|_1
 +KL_{\mathcal R}\int_0^te^{-a(t-s)}
 \|\mathcal S_su-\mathcal S_sv\|_1,ds.
$$

Multiply by $e^{at}$ and apply Gronwall's inequality to obtain
$\|\mathcal S_tu-\mathcal S_tv\|_1\le Ke^{-bt}\|u-v\|_1$.
Choose $t_0$ with $Ke^{-bt_0}<1$. The complete probability-density space is
mapped into itself by $\mathcal S_{t_0}$, so it has a unique fixed point
$u_*$. Commutation of the nonlinear semigroup implies that
$\mathcal S_su_*$ is another fixed point of $\mathcal S_{t_0}$, hence equals
$u_*$ for every $s\ge0$. Thus $u_*$ is stationary and the displayed bound
follows. Every stationary probability density is a fixed point of
$\mathcal S_{t_0}$, proving uniqueness.

Finally, its mild stationarity identity gives
$(T_tu_*-u_*)/t=-t^{-1}\int_0^tT_{t-s}\mathcal R(u_*)ds
\to-\mathcal R(u_*)$ in $L^1$. Therefore $u_*\in D(A)$ and the stationary
equation holds in $L^1$ as well.
:::

:::{prf:lemma} A genuine invariant-ball estimate
:label: lem-uniqueness-fixed-point-bounded

Let $X_*$ be a Banach analysis space in which positivity and total mass define
a closed probability set. Suppose $\mathcal T_C$ preserves that set and, for
**every** density in it,

$$
\|\mathcal T_Cu\|_{X_*}\le A_*+b_*\|u\|_{X_*},
\qquad A_*<\infty,\quad0\le b_*<1.
$$

Then every fixed point lies in the ball of radius
$R_*=A_* /(1-b_*)$, and every probability-density ball of radius $R\ge R_*$
is invariant. If such a ball is nonempty and the map contracts on it, it has
a unique fixed point there. If the contraction holds on the ball of radius
$R_*$, that fixed point is unique in the entire probability set.
:::

:::{prf:proof}
For a fixed point, the bound gives
$(1-b_*)\|u\|_{X_*}\le A_*$. For arbitrary $u$ with norm at most $R$,
$\|\mathcal T_Cu\|_{X_*}\le A_*+b_*R\le R$. This proves actual
self-mapping of the ball. Its completeness follows from closedness, so
contraction gives existence and uniqueness there. Every other fixed point
lies in the smaller ball and is therefore the same point.
:::

:::{prf:remark} Which contraction estimate to use
:label: rem-uniqueness-proof-technique

The $L^1$ gain-loss proof yields a complete existence and uniqueness route
when the kinetic zero-mass mixing bound dominates the nonlinear Lipschitz
constant. A weighted or Sobolev proof can instead use
{prf:ref}`lem-uniqueness-fixed-point-bounded`, provided its mapping estimate
and contraction are proved on that entire ball. In a weighted measure norm,
the same Duhamel argument applies once the kinetic estimate, reaction
Lipschitz bound, positivity, and completeness are established in that norm.
The weighted Harris analysis in {doc}`06_convergence` and the entropy
estimates in {doc}`15_kl_convergence` give routes to the needed kinetic or
nonlinear mixing estimates under their stated hypotheses.
:::

:::{prf:remark} Parameter dependence of uniqueness
:label: rem-uniqueness-algorithm-connection

The sufficient comparison is between proved quantities: $a$, $K$, and
$L_{\mathcal R}$, or the constants of a proved invariant-ball estimate.
Changing kinetic noise changes its equilibrium, smoothing constants, and
possibly interaction bounds. No unlimited-noise conclusion follows without
tracking those changes. The conditions above are sufficient conditions for
uniqueness and attraction; they are not necessary parameter thresholds.
For the adaptive latent generator, measure-dependent kinetic terms must also
be controlled in the same Duhamel estimate.
:::

### 4.3. What hypoellipticity contributes

:::{prf:theorem} Classical local hypoellipticity
:label: thm-uniqueness-hormander

Let $L=\sum_{i=1}^mX_i^2+X_0+c$ have smooth coefficients on an open manifold.
If the Lie algebra generated by the indicated vector fields spans every
tangent space, then $L$ is hypoelliptic: distributional solutions of
$Lu\in C^\infty$ are locally smooth. This is the local theorem of
[Hörmander, *Hypoelliptic second order differential equations*](https://doi.org/10.1007/BF02392081).
It gives no unspecified global boundary condition or isotropic
$L^2\to H^1$ resolvent estimate.
:::

:::{prf:lemma} Kinetic bracket computation
:label: lem-uniqueness-hormander-verification

For $D_v>0$, smooth $F(x)$, and

$$
X_i=\sqrt{D_v}\,\partial_{v_i},\qquad
X_0=v\cdot\nabla_x+(F(x)-\gamma v)\cdot\nabla_v,
$$

one has

$$
[X_0,X_i]=\sqrt{D_v}(-\partial_{x_i}+\gamma\partial_{v_i}).
$$

Thus the diffusion fields and these brackets span all position and velocity
directions in the interior.
:::

:::{prf:proof}
Differentiate the coefficients of $X_0$ with respect to $v_i$. The position
coefficient contributes $-\partial_{x_i}$, the friction coefficient
contributes $+\gamma\partial_{v_i}$, and $F(x)$ contributes zero. Multiplying
by $\sqrt{D_v}$ gives the formula. The $X_i$ span all velocity directions;
subtracting their friction multiples from the brackets gives all position
directions.
:::

:::{prf:theorem} Interior regularity and a quantitative resolvent criterion
:label: thm-uniqueness-hypoelliptic-regularity

For a stationary solution satisfying
$(A-a[u_*])u_*=-G[u_*]$ in the interior, if the kinetic coefficients and
$a[u_*]$ are smooth, $G[u_*]$ is smooth, and the kinetic bracket condition
holds, then $u_*$ is smooth in the interior.

A quantitative global resolvent estimate uses an additional semigroup bound:
if $T_t:X\to Y$ satisfies
$\|T_tg\|_Y\le k(t)\|g\|_X$ and
$C_{X,Y}(C)=\int_0^\infty e^{-Ct}k(t)dt<\infty$, then

$$
\|R_Cg\|_Y\le C_{X,Y}(C)\|g\|_X.
$$

Here $Y$ may be a specified weighted or anisotropic Sobolev space. Its boundary
conditions and the integrability of $k(t)$ must be verified for that space.
:::

:::{prf:proof}
The first statement applies the preceding local hypoellipticity theorem to
the operator with the smooth zeroth-order loss term. For the second, estimate
the defining Bochner integral of $R_C$ in $Y$ and integrate the semigroup
bound. Hörmander's bracket computation alone supplies neither $k(t)$ nor its
integrability at zero. In particular, velocity smoothing and position
smoothing can have different short-time powers.
:::

(sec-chaos-identification)=
## 5. Finite-Time Consistency and the Stationary Equation

:::{div} feynman-prose
We now connect the constructed evolution to the swarm. The connection has two
parts: one step of the swarm must approximate one step of the proposed
population evolution, and the fluctuations about that conditional mean must
vanish. A smooth formula for the mean drift supplies the first part. It does
not supply the second when many outputs share the same random inputs.
:::

### 5.1. A full-step fluctuation and consistency estimate

:::{prf:lemma} Independent innovations with controlled influence
:label: lem-chaos-innovation-variance

Conditioned on the input swarm, let a proposed full update be a function of
independent innovations $\xi_1,\ldots,\xi_M$. Write
$F=\langle L_N',\varphi\rangle$ and let $F^{(j)}$ replace innovation $j$ by
an independent copy. Then

$$
\operatorname{Var}(F\mid S)
\le\frac12\sum_{j=1}^M\mathbb E[(F-F^{(j)})^2\mid S].
$$

If the replacement affects at most $D_j$ output walker coordinates, then

$$
\operatorname{Var}(F\mid S)
\le\frac{2\|\varphi\|_\infty^2}{N^2}
 \sum_{j=1}^M\mathbb E[D_j^2\mid S].
$$

Thus $\sum_j\mathbb ED_j^2\le C N$ is a sufficient full-step variance
estimate of order $1/N$, even when the outputs themselves are dependent.
:::

:::{prf:proof}
Reveal the independent innovations in order and form the Doob martingale
of $F$. Its orthogonal differences sum to the conditional variance.
For the $j$th difference, conditional Jensen's inequality bounds its second
moment by the conditional variance produced by varying innovation $j$ while
holding the others fixed, averaged over the remaining innovations.
That variance equals one half of the mean squared difference of two
independent copies, giving the first inequality after summation.
Changing $D_j$ bounded summands of the empirical average changes it by at
most $2D_j\|\varphi\|_\infty/N$. Substitute this bound into the first
inequality. Any intermediate status checks and common collision updates are
part of the function whose influence is being estimated.
:::

:::{prf:theorem} Finite-time propagation from a one-step empirical estimate
:label: thm-chaos-finite-time-consistency

Let $d_*$ be a bounded metric for weak convergence on
$\mathcal P(\mathsf Z)$, and let $\mathcal F_h$ be the specified nonlinear
one-step population map. Suppose it is $L_h$-Lipschitz in $d_*$, and the
finite population update has

$$
\mathbb E[d_*(L_N',\mathcal F_h(L_N))\mid S]\le\varepsilon_N,
\qquad\varepsilon_N\to0,
$$

uniformly on the states under consideration. Then

$$
\mathbb E d_*(L_N(S_n),\mathcal F_h^n(L_N(S_0)))
\le\varepsilon_N\sum_{j=0}^{n-1}L_h^j.
$$

If $L_N(S_0)\to\mu_0$ in probability, the empirical measure at each fixed
$n$ converges to $\mathcal F_h^n(\mu_0)$. With exchangeability, every fixed
number of walkers has the corresponding product-law limit.
:::

:::{prf:proof}
Use the triangle inequality, condition on $S_n$, and apply Lipschitz
continuity of $\mathcal F_h$. The expected error at step $n+1$ is at most
$\varepsilon_N+L_h$ times the error at step $n$. Induction gives the geometric
sum. The initial-condition error is bounded by
$L_h^n d_*(L_N(S_0),\mu_0)$; boundedness of the metric turns convergence in
probability into convergence of this expectation. The empirical-to-chaos
lemma then gives the coordinate marginal conclusion.

A practical sufficient one-step bound uses a convergence-determining family
$\{\varphi_j\}$ and
$d_*(\mu,\eta)=\sum_jw_j|\mu\varphi_j-\eta\varphi_j|$, with
$\|\varphi_j\|_\infty\le1$ and summable positive weights. Sum the absolute
conditional bias plus the square root of the conditional variance for each
test. The innovation lemma controls the latter when its influence moments
are bounded.
:::

:::{prf:remark} Fixed timestep, continuous time, and surviving trajectories
:label: rem-chaos-time-and-conditioning

At fixed $h$, the limit in the preceding theorem is the nonlinear discrete
map $\mathcal F_h$. To identify a continuous equation, one additionally
establishes convergence of $\mathcal F_h^{\lfloor t/h\rfloor}$ to its
semiflow $\mathcal S_t$ and a finite-population error that vanishes on the
same time scale. In a joint limit the accumulated error
$\varepsilon_{N,h}\sum_{j< t/h}L_h^j$ must tend to zero.
Fixed clipping, finite cloning probabilities, and the Boris-BAOAB force
normalization must follow the scaling specified in
{doc}`../1_the_algorithm/02_fractal_gas_latent`.

For killed swarms, the estimate must either describe the law conditioned on
the entire time horizon, or hold for an extended process whose probability
of reaching the cemetery during that horizon tends to zero. Conditioning
separately at each step changes the path law. A finite-horizon conditioning
error can be bounded by its extinction probability; the QSD identity below
makes that probability explicit.
:::

:::{prf:lemma} Boundary and revival limits use the same time convention
:label: lem-boundary-convergence

Suppose the empirical alive and dead components converge jointly, the alive
mass stays bounded below, and their transition integrands converge as bounded
continuous functions, or under the uniform-integrability hypothesis of
{prf:ref}`lem-uniform-integrability`. Then the expectations of their boundary
and revival transition contributions converge to the corresponding integrals
of those limiting kernels.

For the explicitly chosen continuous interior-killing model with bounded
continuous $c$, normalized revival profile $G_\rho$, and a scalar dead
reservoir, these contributions are

$$
-\int c(z)\varphi(z)f(dz)
 +\lambda_{\rm rev}m_d\int\varphi(z)G_\rho(dz).
$$

At fixed $h$, use the actual exit and revival kernels instead. Passing
$N\to\infty$ does not send $h\to0$, and hard boundary absorption is not
identified with a smooth bounded interior killing rate by that population
limit.
:::

:::{prf:proof}
The lower alive-mass bound makes normalization continuous, by
{prf:ref}`lem-uniqueness-lipschitz-moments`. Joint convergence and continuity
of the specified kernels therefore give convergence of their bounded test
integrals. For unbounded integrands use truncation and uniform integrability.
In the displayed continuous model, substitute the killing multiplier and
revival rate into those integrals. A transport boundary-flux limit instead
requires the boundary analysis in {doc}`08_mean_field`; it is a different
limiting term from the bounded multiplier $c$.
:::

### 5.2. Survival terms in stationary identification

:::{prf:remark} Exact stationary balance for a killed kernel
:label: rem-qsd-vs-true-stationarity

For a bounded full-swarm test $F$ extended by zero at the cemetery,

$$
\nu_N(Q_N-I)F=-(1-\alpha_N)\nu_NF.
$$

For the scaled operator $G_{N,h}=(Q_{N,h}-I)/h$, its right side is
$-\lambda_{N,h}^{\rm bal}\nu_NF$, where
$\lambda_{N,h}^{\rm bal}=(1-\alpha_{N,h})/h$.
The physical exponential survival exponent is
$\lambda_{N,h}^{\rm exp}=-\log(\alpha_{N,h})/h$; these agree asymptotically
when $1-\alpha_{N,h}\to0$ at the relevant scale.
For a continuous killed generator with survival law $e^{-\lambda_Nt}$,
the corresponding eigenmeasure identity is
$\nu_N\mathcal L_NF=-\lambda_N\nu_NF$.
:::

:::{prf:theorem} Vanishing extinction contribution
:label: thm-extinction-rate-vanishes

Suppose the QSD one-step hazards obey
$1-\alpha_N\le\delta_N\to0$. Then their contribution to every bounded
unscaled stationary balance tends to zero, and
$\mathbb P_{\nu_N}(T_\dagger\le n)\le n\delta_N$ for fixed $n$.
For a joint limit with timestep $h_N$, the sufficient condition is
$\delta_N/h_N\to0$ for the scaled generator balance and for survival on
fixed physical-time intervals.

One quantitative source is
{prf:ref}`prop-convergence-survival-bound`: an interior-population estimate,
a joint failure bound for distinct identified walkers, and
$\nu_N(G_N^c)\le a_N$ give
$\delta_N=a_N+m_Np^{m_N-1}$ for the $k<2$ cemetery convention.
:::

:::{prf:proof}
The stationary error has absolute value at most
$\delta_N\|F\|_\infty$, or
$\delta_N\|F\|_\infty/h_N$ for the scaled operator. The QSD survival law
gives
$1-\alpha_N^n\le n(1-\alpha_N)\le n\delta_N$.
For $n=\lfloor t/h_N\rfloor$, the same upper bound tends to zero if
$\delta_N/h_N\to0$. The quoted quantitative source was proved by counting
interior walkers and bounding joint failures in {doc}`06_convergence`.
:::

:::{prf:remark} Population stability and the order of limits
:label: rem-extinction-rate-physical-interpretation

An exponential-in-$N$ upper bound on the hazard makes survival likely on
fixed time intervals and on some growing intervals. It does not exclude
extinction over arbitrarily long times at fixed $N$. Moment drift alone
controls neither this hazard nor the dependence between individual failures.
The empirical law's concentration and the simultaneous-survival estimate
enter the limiting argument as distinct analytic inputs.
:::

:::{prf:theorem} Deterministic empirical limits solve the stationary equation
:label: thm-limit-is-weak-solution

Suppose $L_N\to\mu_*$ in probability under $\nu_N$, and for each test
$\varphi$ in a specified determining generator core,

$$
\mathbb E_{\nu_N}
\left|G_{N,h_N}\langle L_N,\varphi\rangle
 -\mathcal B_\varphi(L_N)\right|\longrightarrow0.
$$

Assume $\mathcal B_\varphi$ is continuous along these empirical limits,
its values are uniformly integrable, and
$(1-\alpha_{N,h_N})/h_N\to0$. Then

$$
\mathcal B_\varphi(\mu_*)=0
$$

for every such test. When the limiting generator has been identified with
$Au+\mathcal R(u)$, this is its stationary weak equation. Existence of a
density and its regularity require the corresponding analytic results of
Section 4.
:::

:::{prf:proof}
Apply the exact QSD identity to
$F(S)=\langle L_N(S),\varphi\rangle$, extended by zero at the cemetery.
Its right side tends to zero because $|F|\le\|\varphi\|_\infty$.
The assumed consistency error also tends to zero, so
$\mathbb E\mathcal B_\varphi(L_N)\to0$. Continuity at the deterministic
limit and uniform integrability identify this limit with
$\mathcal B_\varphi(\mu_*)$.
:::

:::{prf:corollary} What a random empirical limit satisfies
:label: thm-limit-is-weak-solution-summary

If the same assumptions give only $\Lambda_N\Rightarrow\Lambda$, the
conclusion is instead

$$
\int\mathcal B_\varphi(\mu)\Lambda(d\mu)=0.
$$

This averaged identity alone does not make the barycentre
$\int\mu\Lambda(d\mu)$ a solution of a nonlinear stationary equation.
Deterministic concentration, or the invariant-flow argument below, completes
that additional identification.
:::

:::{prf:proof}
Repeat the preceding expectation argument with the distributional limit of
the random measure. Nonlinear $\mathcal B_\varphi$ does not in general
commute with integration over $\Lambda$.
:::

(sec-chaos-stationary-limit)=
## 6. Stationary Chaos and Macroscopic Convergence

:::{div} feynman-prose
There are two ways to remove the remaining randomness in the population law.
One uses concentration: empirical measurements fluctuate less and less as the
swarm grows. The other uses dynamics: every possible limiting population law
is carried toward the same attractor. Both arguments are stronger than
uniqueness of a stationary density considered in isolation.
:::

### 6.1. Concentration of empirical measurements

:::{prf:lemma} A variance criterion for deterministic empirical limits
:label: lem-chaos-concentration-criterion

Suppose the empirical laws are tight, $\mu_N\Rightarrow\mu_*$, and a
countable convergence-determining family $\{\varphi_j\}\subset C_b(\mathsf Z)$
satisfies

$$
\operatorname{Var}_{\nu_N}(L_N\varphi_j)\longrightarrow0
\quad\text{for every }j.
$$

Then $\Lambda_N\Rightarrow\delta_{\mu_*}$ and the sequence is
$\mu_*$-chaotic. A sufficient source of the variance estimate is a joint-law
Poincaré inequality

$$
\operatorname{Var}_{\nu_N}(F)\le C_P\mathcal E_N(F,F),\qquad
\mathcal E_N(L_N\varphi_j,L_N\varphi_j)\le C_j/N,
$$

with $C_P$ uniform in $N$. The joint LSI of
{prf:ref}`cor-n-uniform-lsi`, for the law and form to which it applies, implies
such a Poincaré inequality. When status varies, the form must also control
observables that distinguish status strata.

An alternative input is the total relative-entropy estimate
{prf:ref}`thm-mixing-variance-corrected` from
{doc}`12_qsd_exchangeability_theory`. For a product reference $\rho^{\otimes N}$
and $H_N=H(\nu_N\mid\rho^{\otimes N})$, it gives, for bounded $\varphi$,

$$
\mathbb E_{\nu_N}|L_N\varphi-\rho\varphi|^2
\le\frac{4\|\varphi\|_\infty^2}{N}
 \left(H_N+\tfrac12\log2\right).
$$

Thus $H_N=o(N)$ yields a deterministic empirical limit $\rho$; bounded total
entropy gives the displayed $O(1/N)$ estimate. The reference and entropy are
those of the actual joint law, including its status convention.
:::

:::{prf:proof}
Exchangeability gives $\mathbb E L_N\varphi_j=\mu_N\varphi_j$. Therefore

$$
\mathbb E|L_N\varphi_j-\mu_*\varphi_j|^2
=\operatorname{Var}(L_N\varphi_j)
 +|\mu_N\varphi_j-\mu_*\varphi_j|^2\longrightarrow0.
$$

Every subsequential empirical-law limit is thus concentrated on measures
whose integrals of all $\varphi_j$ equal those of $\mu_*$. The determining
property makes that measure unique. Tightness then proves convergence of the
whole empirical-law sequence, and {prf:ref}`lem-empirical-convergence` gives
chaos. The Poincaré conditions bound each displayed variance by $C_PC_j/N$.

For an LSI written as
$\operatorname{Ent}_{\nu_N}(f^2)\le2\rho^{-1}\mathcal E_N(f,f)$,
substitute $f=1+\epsilon F$ for bounded centred $F$ and let
$\epsilon\to0$. Expanding the entropy to second order gives
$2\epsilon^2\operatorname{Var}(F)+o(\epsilon^2)$; the form is
$\epsilon^2\mathcal E_N(F,F)$. Thus $C_P=1/\rho$, with extension to the
form domain by the defining approximation.
:::

:::{prf:corollary} Stationary identification using LSI concentration
:label: cor-chaos-lsi-stationary-limit

Suppose the uniform confining moment, concentration criterion, generator
consistency, and vanishing scaled extinction contribution above hold along
every convergent subsequence. Suppose the limiting stationary equation has
exactly one probability solution $\mu_*$ in the class containing all those
limits. Then $\Lambda_N\Rightarrow\delta_{\mu_*}$, every fixed marginal
converges to $\mu_*^{\otimes l}$, and $\mu_N\Rightarrow\mu_*$.
:::

:::{prf:proof}
Take an arbitrary subsequence. Tightness gives a further subsequence of first
marginals converging to some $\bar\mu$. The variance criterion makes its
empirical-law limit $\delta_{\bar\mu}$. The stationary identification
theorem makes $\bar\mu$ a solution of the limiting stationary equation, so
uniqueness gives $\bar\mu=\mu_*$. Every subsequence has such a further
subsequence with the same limit; this proves convergence of the full sequence
and, by the empirical-to-chaos lemma, of every fixed marginal.
:::

### 6.2. Identification by the full mean-field flow

:::{prf:theorem} Stationary chaos from finite-time consistency and attraction
:label: thm-uniqueness-of-qsd

Suppose $\{\Lambda_N\}$ is tight, the swarm QSDs are exchangeable, and
$\mathcal S_t$ is a continuous semiflow on the relevant closed class of
probabilities on $\mathsf Z$. Assume:

1. For each fixed $t>0$, with observation indices $n_N(t)$, the empirical
   evolution started from $\nu_N$ satisfies

   $$
   \mathbb E d_*(L_N(S_{n_N(t)}),\mathcal S_tL_N(S_0))\longrightarrow0.
   $$

   On absorbed trajectories assign any fixed probability to $L_N$.
2. The extinction probability at that observation time vanishes:
   $1-\alpha_N^{n_N(t)}\to0$.
3. Every measure in the relevant limit class is attracted to one probability
   $\mu_*$: $\mathcal S_t\mu\Rightarrow\mu_*$ as $t\to\infty$.

Then

$$
\Lambda_N\Rightarrow\delta_{\mu_*},\qquad
\nu_N^{(l)}\Rightarrow\mu_*^{\otimes l}\quad\text{for every fixed }l.
$$

The limiting law is stationary. For a continuous model satisfying
{prf:ref}`thm-uniqueness-uniqueness-stationary-solution`, its attraction is
proved by the zero-mass mixing comparison, with the appropriate extension
to initial measures in this limit class. At fixed timestep, the same theorem
holds with iterates of a continuous nonlinear map and its globally attracting
fixed point.
:::

:::{prf:proof}
Choose a subsequence with $\Lambda_N\Rightarrow\Lambda$. For a bounded
Lipschitz function $H$ on $\mathcal P(\mathsf Z)$, the QSD identity gives

$$
\left|\mathbb E H(L_N(S_{n_N(t)}))-\int H\,d\Lambda_N\right|
\le2\|H\|_\infty(1-\alpha_N^{n_N(t)}).
$$

The finite-time consistency assumption changes the expectation on the left
to $\int H(\mathcal S_t\mu)\Lambda_N(d\mu)$ with an error tending to zero.
Continuity of $\mathcal S_t$ lets us pass to the limit. We obtain

$$
\int H(\mathcal S_t\mu)\Lambda(d\mu)=\int H(\mu)\Lambda(d\mu).
$$

Thus $\Lambda$ is an invariant probability for the nonlinear flow on the
space of population laws. By attraction and bounded convergence, the left
side tends as $t\to\infty$ to $H(\mu_*)$. Hence
$\int H\,d\Lambda=H(\mu_*)$ for every such $H$, giving
$\Lambda=\delta_{\mu_*}$. All subsequential limits agree, proving empirical
convergence and then chaos.

For stationarity of $\mu_*$, choose any $\mu$ in the attraction class.
Continuity and the semiflow identity give
$\mathcal S_s\mu_*=\lim_{t\to\infty}\mathcal S_s\mathcal S_t\mu
=\lim_{t\to\infty}\mathcal S_{s+t}\mu=\mu_*$.
:::

:::{prf:remark} Why attraction appears in the theorem
:label: rem-chaos-attraction-versus-uniqueness

An invariant probability on the space of population measures can be supported
on a periodic orbit of a deterministic nonlinear flow. Uniqueness of that
flow's stationary point alone excludes neither such an orbit nor an invariant
mixture. The full-time argument above uses global attraction, while the LSI
route uses vanishing empirical variance. Both supply the additional step
needed to turn a stationary empirical mixture into one stationary law.
The empirical-measure approach to chaos for general Markov transitions is
also developed in
[Gottlieb, *Markov Transitions and the Propagation of Chaos*](https://arxiv.org/abs/math/0001076).
:::

### 6.3. Observables and Wasserstein convergence

:::{prf:theorem} Convergence of macroscopic observables
:label: thm-thermodynamic-limit

Under either stationary-chaos theorem above, for every bounded continuous
$\varphi$,

$$
L_N\varphi\longrightarrow\mu_*\varphi
\quad\text{in probability and in every finite }L^p,
$$

and

$$
\lim_{N\to\infty}\mathbb E_{\nu_N}L_N\varphi=\mu_*\varphi.
$$

If the limiting alive mass is positive, the same convergence holds for
bounded continuous observables averaged over the alive population, with
limit under the normalized alive law.
:::

:::{prf:proof}
Convergence of the empirical law to a deterministic measure gives convergence
in probability of each bounded continuous test integral. Since these
integrals are uniformly bounded, their powers are uniformly integrable, so
convergence holds in every finite $L^p$ and in expectation.
For the alive average, both its numerator and denominator converge; the
limiting denominator is positive. Apply continuity of their ratio, using
boundedness of the normalized observable. On the marked state space the alive
indicator is a bounded continuous status function.
:::

:::{prf:corollary} Wasserstein-2 convergence with control of second-moment tails
:label: cor-w2-convergence-thermodynamic-limit

Let $\mu_N\Rightarrow\mu_*$ be the phase-space marginals obtained above.
Suppose their squared distances from a reference point are uniformly
integrable:

$$
\lim_{R\to\infty}\sup_N\int_{|z|>R}|z|^2\,\mu_N(dz)=0.
$$

Then $W_2(\mu_N,\mu_*)\to0$. A uniform $(2+\epsilon)$-moment bound for some
$\epsilon>0$, or a proved uniform confining exponential moment, is sufficient.
For a bounded phase-space metric the tail condition is automatic.
:::

:::{prf:proof}
The tail condition and weak convergence imply that $\mu_*$ has a finite
second moment. By the representation theorem for weak convergence on a
Polish space, choose a coupling $Z_N\to Z$ almost surely with laws
$\mu_N$ and $\mu_*$. The variables $|Z_N-Z|^2$ are uniformly integrable,
since they are bounded by $2|Z_N|^2+2|Z|^2$ and the first family has the stated
uniform-integrability property. Almost-sure convergence and truncation thus
give $\mathbb E|Z_N-Z|^2\to0$. This coupling bounds $W_2^2$ from above.
A uniform higher moment bounds the squared tail by
$R^{-\epsilon}\sup_N\mu_N|z|^{2+\epsilon}$.

A bounded second moment alone is insufficient: on the line,
$\mu_N=(1-1/N)\delta_0+(1/N)\delta_{\sqrt N}$ converges weakly to
$\delta_0$ and has second moment one, but
$W_2^2(\mu_N,\delta_0)=1$ for every $N$.
:::

:::{prf:remark} Scope of the assembled mean-field results
:label: rem-chaos-model-scope

For the continuous bounded-rate model, the gain-loss decomposition and
normalization bounds give a complete positive mild-solution construction.
A proved kinetic zero-mass mixing estimate yields stationary existence,
uniqueness, and attraction through the explicit comparison with
$L_{\mathcal R}$. A joint-law LSI supplies empirical concentration when its
form controls the observables being used. The finite-population conclusion
then follows from the corresponding full-kernel consistency and survival
estimates, with each time scale fixed as stated.

For the discrete latent algorithm, the companion kernel, sampled fitness,
collision groups, position metric, and capped split update must all appear
in that consistency estimate. For an adaptive continuous generator,
measure-dependent drift and diffusion require their own well-posedness and
stability bounds. These are the analytic connections to
{doc}`14_a_geometric_gas_c3_regularity`,
{doc}`14_b_geometric_gas_cinf_regularity_full`, and
{doc}`17_geometric_gas`. The continuum constructions use the limits and
uniformity conditions developed in {doc}`16_continuum_discharge`.
:::
