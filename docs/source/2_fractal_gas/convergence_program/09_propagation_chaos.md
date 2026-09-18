# The Discrete Population Limit and Propagation of Chaos

:::{div} feynman-prose
A large swarm has two kinds of randomness. A walker fluctuates around the
population distribution, and the population distribution itself can fluctuate
from run to run. Exchangeability says that the labels carry no information.
Propagation of chaos says more: the second kind of randomness disappears, and
any fixed number of walkers become independent in the limit.

We follow one complete programmed update. Measurement companions create
sampled fitness marks; accepted donor edges assemble collision components;
one rotation acts on each component; and BAOAB, position diffusion, the
velocity cap, and boundary classification finish the step. The component can
contain several walkers, so its shared randomness must survive in the
one-walker population map. Independence emerges between separately tagged
components as the swarm grows.

The resulting evolution is $\mu_{n+1}=\mathcal F_h(\mu_n)$ at the actual
fixed timestep. We prove its one-step approximation and then iterate that
proof. Stationary existence, stationary concentration, and global attraction
are different claims; the last section states precisely which of them follow.
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

Let $\mathsf Z=\mathbb R^d\times\overline B(0,V)\times\{0,1\}$
be the complete marked slot state at the end of an update. Its coordinates
are position, capped velocity, and alive indicator. Dead coordinates are
retained: their position enters weighted revival-donor sampling and their
velocity enters the connected-component collision. The status space has its
discrete topology. The canonical process has no donor history; configurations
with history require a correspondingly enlarged state.

For each $N\ge2$, let $Q_N$ be the killed full-swarm kernel on its noncemetery
space. For the canonical quadratic-force terminal-box configuration,
{prf:ref}`thm-chaos-canonical-finite-n-qsd` establishes its unique QSD $\nu_N$.
For other configurations, use an applicable QSD theorem for their complete
transition. Write

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

For a marked probability $\mu$, let $f=\mu|_{\{a=1\}}$ be its alive
subprobability and $m_a=f(\mathsf Z)$. If $m_a>0$, its normalized alive law is
$\rho=f/m_a$. The finite-swarm counterparts are

$$
f_N=\frac1N\sum_{i:a_i=1}\delta_{(x_i,v_i)},\qquad
m_{a,N}=\frac{k_N}{N},\qquad \rho_N=\frac{f_N}{m_{a,N}}.
$$

The mass-one marked law, alive subprobability, and normalized alive law have
different normalization equations. A scalar dead reservoir is a closed model
only when the revival rule depends on dead walkers through that scalar alone.
Companion selection that depends on their retained coordinates requires their
distribution in the state.
:::

:::{div} feynman-prose
For a fixed number of walkers, the final two independent noises provide a
particularly direct route to equilibrium under survival. The OU noise spreads
velocity, and the final position noise spreads position. Revival first brings
every position back to a donor inside the box. Consequently even a walker
whose retained dead position is very far away has a controlled distribution
at the next update.

The far-away coordinate is still part of the state. The useful observation is
that the next donor choice sees it through the bounded distance feature; once
the donor is chosen, revival replaces its position before the kinetic motion.
We can exploit this exact dependence to prove the finite-swarm QSD theorem.
:::

:::{prf:theorem} Unique QSD and conditioned convergence for the canonical box gas
:label: thm-chaos-canonical-finite-n-qsd

Fix $N\geq1$ and the canonical terminal absorbing-box update with
$U(x)=|x|^2/2$, positive OU and final position-noise amplitudes, and smooth
velocity cap $V$. For $h>0$ with $h\ne2$, this actual killed kernel has a
unique QSD $\nu_N$ and a survival eigenvalue $0<\alpha_N<1$. There are
$C_N<\infty$ and $r_N<1$ such that

$$
\sup_\eta
\left\|\frac{\eta Q_N^n}{\eta Q_N^n1}-\nu_N\right\|_{\mathrm{TV}}
\leq C_Nr_N^n.
$$

The supremum is over probability laws on nonextinct, terminally consistent
full marked states with capped velocities. Dead physical positions remain
unbounded and are retained in this statement. In particular the theorem
applies to the canonical $h=0.04$ configuration. Its constants may depend on
$N$; no assumption of positional contraction by cloning is needed.
:::

:::{prf:proof}
**1. Represent exactly the input coordinates used by the transition.**
Let $S_R(x)=Rx/(R+|x|)$ be the position feature in the donor distance.
For an alive slot retain $x\in\overline D$; for a dead slot represent its
position by $u=S_R(x)\in\overline{S_R(D^c)}$. Retain the velocity in
$\overline B_V$ and its discrete alive/dead mark in both cases. The finite
union of the resulting products over all nonempty alive masks is a compact
space $K_N$.

For $|u|<R$, the inverse is $x=Ru/(R-|u|)$, so every finite physical dead
coordinate remains recoverable. Points with $|u|=R$ compactify only the
input representation. The donor weights extend continuously to these points.
Canonical reward statistics use eligible walkers, whose physical positions
are bounded by the box. No other canonical operation requires the raw dead
position: mandatory revival copies a frozen eligible donor position before
jitter, and the collision uses retained capped velocities. Thus the actual
kernel extends to $K_N$. The status components are disjoint, and the exact
single-eligible-donor convention is retained on those components.

Every position before jitter is in $\overline D$. The frozen component rule
bounds each post-collision velocity by $(1+2\alpha)V$, where $\alpha$ is the
restitution coefficient. These are bounds on kinetic input means, not a
claim that physical dead output positions have compact support.

**2. Verify full phase-space smoothing for the declared schedule.**
Condition on the finite measurement-companion, cloning-companion, and gate
pattern and on its component rotations. The collision coordinates are
bounded before independent Gaussian jitter. The remaining quadratic-force
BAOAB stages are affine. With $x_1$ the A1 position, $v_1$ the B1 velocity,
$a=e^{-\gamma h}$, and actual OU innovation amplitude $q>0$, write

$$
\begin{aligned}
v_2&=av_1+q\xi,&x_2&=x_1+\tfrac h2v_2,\\
v_3&=v_2-\tfrac h2x_2,&x_3&=x_2+s\zeta,\qquad s=\sigma_x\sqrt h>0.
\end{aligned}
$$

The independent standard Gaussian innovations $(\xi,\zeta)$ have coefficient
matrix, for the output ordered as $(x_3,v_3)$,

$$
L=\begin{pmatrix}
\tfrac h2qI&sI\\
q(1-h^2/4)I&0
\end{pmatrix}.
$$

It is invertible for $h\ne2$. After integrating Gaussian cloning jitter,
the pre-cap joint phase-space law is therefore a Gaussian with strictly
positive density on $\mathbb R^{2dN}$. Its mean ranges over a compact set as
the effective input and rotations vary. Its covariance belongs to a finite
positive-definite family indexed by the accepted-recipient pattern.

The final cap $v\mapsto Vv/(V+|v|)$ is a $C^1$ diffeomorphism from
$\mathbb R^d$ onto $B_V$, with positive Jacobian. The resulting density is
positive on $\mathbb R^{dN}\times B_V^N$. Terminal classification assigns
its actual marks; removing the all-dead output gives $Q_N$. Every nonempty
relatively open subset of $K_N$ contains a positive-measure set of physical
interior outputs, so $Q_N(k,\cdot)$ has full support on $K_N$. Both survival
and extinction have positive probability at every input.

**3. Prove compactness of the transition operator.**
At fixed $N$ there are finitely many companion and gate patterns. Within each
fixed alive mask their probabilities are continuous in the effective input:
weighted normalizers stay positive, regularized population statistics are
continuous, and the positive-part acceptance formula is continuous at ties.
For a fixed pattern, component membership is fixed and its Haar rotations
range over a compact product of orthogonal groups.

Gaussian laws with fixed positive covariance vary continuously in total
variation with their means, uniformly over these compact parameter sets.
Finite mixing and integration over the rotations preserve this continuity.
The cap, terminal marking, and survival restriction contract total variation.
Consequently $k\mapsto Q_N(k,\cdot)$ is TV-continuous on $K_N$.

The operator $Q_N:C(K_N)\to C(K_N)$ maps the unit ball to a bounded
uniformly equicontinuous family, since

$$
|Q_Nf(k)-Q_Nf(l)|
\leq\|Q_N(k,\cdot)-Q_N(l,\cdot)\|_{\mathrm{TV}}
\quad(\|f\|_\infty\leq1),
$$

where the TV norm in this proof is $\sup_{|f|\leq1}|\mu f|$.
Arzelà–Ascoli makes $Q_N$ compact. Full support gives $Q_Nf>0$ for every
nonzero continuous $f\geq0$, and compactness of $K_N$ gives
$\min Q_Nf>0$. Moreover continuity of $Q_N1$ gives constants
$0<a_N\leq Q_N1\leq b_N<1$.

**4. Obtain an eigenfunction and verify a Markov minorization.**
The positive cone of $C(K_N)$ is total. The compact positive operator has
spectral radius at least $a_N>0$, because $Q_N^n1\geq a_N^n1$.
The compact-operator Krein–Rutman theorem therefore supplies an eigenfunction
$e_N\geq0$, $e_N\not\equiv0$, with
$Q_Ne_N=\alpha_Ne_N$ and $\alpha_N=r(Q_N)$. Strong positivity gives
$0<m_N\leq e_N\leq M_N<\infty$, and
$a_N\leq\alpha_N\leq b_N<1$. These verify the hypotheses of
[Zhang, Theorem 1.1](https://arxiv.org/pdf/1606.04377).

Choose a compact target box strictly inside the all-alive position domain
and the open velocity ball, and let $\theta_N$ be normalized Lebesgue measure
there. The conditional Gaussian means are bounded and the covariance family
is finite and positive definite. Their densities consequently have a common
positive lower bound on the inverse-cap image of this target. The inverse-cap
Jacobian has a positive lower bound there as well. Hence, for some
$\epsilon_N>0$,

$$
Q_N(k,A)\geq\epsilon_N\theta_N(A)\qquad(k\in K_N).
$$

This bound holds for every accepted graph and rotation and therefore also
for their actual mixture. The Doob kernel

$$
P_N(k,dl)=\frac{Q_N(k,dl)e_N(l)}{\alpha_Ne_N(k)}
$$

is Markov and satisfies

$$
P_N(k,\cdot)\geq\delta_N\widehat\theta_N(\cdot),\qquad
\delta_N=\frac{\epsilon_N\theta_N(e_N)}{\alpha_NM_N}>0,\qquad
\widehat\theta_N(dl)=\frac{e_N(l)\theta_N(dl)}{\theta_N(e_N)}.
$$

Splitting this common part from $P_N$ contracts total variation by
$1-\delta_N$. Iteration gives its unique invariant law $\pi_N$ and uniform
geometric mixing.

**5. Recover the QSD and conditioned convergence.**
Define

$$
\nu_N(dl)=\frac{e_N(l)^{-1}\pi_N(dl)}{\pi_N(e_N^{-1})}.
$$

Invariance of $\pi_N$ gives $\nu_NQ_N=\alpha_N\nu_N$. Any other QSD
$\nu$ with eigenvalue $\beta$ satisfies
$\beta\nu(e_N)=\nu Q_Ne_N=\alpha_N\nu(e_N)$, so
$\beta=\alpha_N$. Its normalized $e_N$-weighted law must be invariant for
$P_N$ and hence equals $\pi_N$. This proves QSD uniqueness.

For every bounded measurable $f$,

$$
\eta Q_N^nf=\alpha_N^n\eta\!\left[e_NP_N^n(f/e_N)\right].
$$

Divide by the same formula for $f=1$. Uniform mixing of $P_N$ and the
positive bounds on $e_N$ give the stated estimate with, for example,
$r_N=1-\delta_N$ and $C_N=4(M_N/m_N)^2$.

**6. Lift to the complete physical state.**
Let $\iota:S_{\rm phys}\to K_{\rm real}$ retain alive positions, every
velocity and every mark, and apply $S_R$ only to dead position coordinates.
It is a measurable bijection, with the inverse given in Step 1. Couple the
two coordinate descriptions with identical measurement draws, cloning draws,
gates, component rotations, jitters, OU innovations, and final position noises.
Stage by stage, the complete updates satisfy

$$
T_{\rm eff}(\iota s,\xi)=\iota T_{\rm phys}(s,\xi),
$$

with the same extinction event. Consequently, for all bounded measurable
$f$ on $K_{\rm real}$ and $g$ on the physical marked state,

$$
Q_{\rm phys}(f\circ\iota)(s)=Q_{\rm eff}f(\iota s),\qquad
Q_{\rm phys}g(s)=Q_{\rm eff}(g\circ\iota^{-1})(\iota s).
$$

These are exact transition identities. The coordinate map changes neither
the mechanism nor its survival probabilities.

Every output of $Q_N$ has finite physical positions and velocity strictly
inside the cap. The eigenmeasure identity
$\nu_N=\alpha_N^{-1}\nu_NQ_N$ therefore gives zero QSD mass to artificial
compactification points. Applying the inverse feature coordinate to dead
slots reconstructs their full physical law. This measurable bijection on
actual states preserves the kernel identities and TV bounds, proving the
claim for the retained-coordinate algorithm itself.
:::

:::{prf:remark} Exact velocity collapse at the resonant timestep
:label: rem-chaos-canonical-baoab-resonance

The restriction $h\ne2$ in the preceding theorem corresponds to an actual
change in the canonical dynamics. Take $N=1$, $h=2$, and
$U(x)=|x|^2/2$. The sole live walker has no accepted cloning edge and receives
no cloning jitter. Writing its entering state as $(x,v)$, the actual stages
give

$$
v_1=v-x,\qquad x_1=x+v_1=v,\qquad
v_2=e^{-2\gamma}(v-x)+q\xi,\qquad
x_2=v+v_2,\qquad v_3=v_2-x_2=-v.
$$

Final position noise and terminal classification leave this pre-cap
velocity calculation unchanged. Therefore, on every surviving trajectory,

$$
v_{n+1}=-\frac{Vv_n}{V+|v_n|},\qquad
v_n=(-1)^n\frac{Vv_0}{V+n|v_0|},\qquad
|v_n|\leq\frac{V}{n+1}\quad\text{when }|v_0|\leq V.
$$

Any QSD is supported on $v=0$. Indeed, the $n$-step killed output is
supported on $\{|v|\leq V/(n+1)\}$ for every entering state. Its QSD identity
$\nu Q^n=\alpha^n\nu$ with $\alpha>0$ forces the same support for $\nu$;
intersecting these sets over $n$ gives $v=0$. This argument retains the
survival weights exactly.

On the invariant set $v=0$, the physical position update is

$$
x'=-e^{-2\gamma}x+q\xi+\sigma_x\sqrt2\,\zeta,
\qquad x'\in D\text{ for survival}.
$$

It is the actual restricted Gaussian transition, with strictly positive
variance $q^2+2\sigma_x^2$. On the compact box closure its killed kernel has
a continuous strictly positive density. The compact-positive-operator and
Doob-minorization argument in the preceding proof therefore gives a unique
QSD on this invariant set. Since every QSD must be supported there, it is
also the unique QSD of the complete one-walker killed process at $h=2$.

Nevertheless, starting from any $v_0\ne0$, the velocity at each finite
surviving update is the nonzero deterministic vector displayed above. Its
conditional law and the QSD have disjoint velocity supports, so their TV
norm distance is $2$ at every finite $n$. Each such conditioning event has
positive probability because the final position noise has positive density.
Thus the velocity norm tends to zero, but total-variation convergence to
the QSD fails. The loss of the preceding convergence conclusion at $h=2$
is a property of the programmed BAOAB and cap composition.
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
last assertion. The required kernel symmetry is proved for the component update in
{prf:ref}`lem-chaos-canonical-equivariance`. For the canonical quadratic-force
box configuration, {prf:ref}`thm-chaos-canonical-finite-n-qsd` supplies the
unique QSD, so both hypotheses are verified for this algorithm.
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
## 3. The Actual One-Step Consistency Estimate

:::{div} feynman-prose
Sit on a randomly chosen walker and look at all the accepted cloning edges
attached to it. Some point toward its donor. Others come from walkers that
selected it. Those incoming walkers matter: they change the component's
centre-of-mass velocity, and they share its random rotation.

There is a useful ordering hidden in this graph. An accepted live edge goes
strictly uphill in sampled fitness. A dead walker points to a live donor,
and no walker points to a dead donor. Thus a component has no cycle. The
ordering also bounds long paths, even when acceptance probabilities are
large. This is the estimate that lets a local component describe a walker in
a very large swarm.
:::

:::{prf:definition} Canonical fixed-step regime and convergence class
:label: def-chaos-canonical-regime

Use the full update $\mathcal F_h$ constructed in {doc}`08_mean_field`:
independent current Gaussian-weighted measurement and cloning companions,
global regularized fitness statistics, frozen order-one acceptance,
mandatory revival through the same weighted current-donor law, jitter for
all accepted recipients, and one independent Haar $O(d)$ rotation per accepted
undirected component. Follow this by the specified constant-noise BAOAB
step, independent final position diffusion, radial velocity cap, and terminal
boundary classification. The canonical configuration has no donor history and
no viscosity; the potential has globally Lipschitz gradient with linear
growth. The timestep $h>0$, positive final position-noise amplitude, finite
cap $V$, positive fitness floors, positive standardization regularizers,
and donor widths are fixed independently of $N$.

The squashed algorithmic distances give actual weight bounds
$0<\kappa_D\le w_D\le1$ and $0<\kappa_C\le w_C\le1$. They do not bound
physical positions. Take either the terminal absorbing box, or the unbounded
confining configuration with quadratic reward. In the box, alive positions
are bounded and all retained velocities obey the cap. In the unbounded case,
assume initial empirical convergence and a uniform initial $(4+\delta)$
position moment for some $\delta>0$; independent initialization with that
moment supplies this condition. All slots are alive in the unbounded case.

Write $\mu_N=L_N(S_N)$ for deterministic input arrays and suppose
$\mu_N\Rightarrow\mu$ with $m(\mu)>0$. In the unbounded case use the stated
moment bound as well. Every sufficiently large $N$ then has
$m(\mu_N)\ge m_*>0$. The finite-step moment and terminal-survival estimates
in {doc}`08_mean_field` propagate this class over each fixed finite horizon.
:::

:::{prf:lemma} Permutation equivariance of the complete canonical kernel
:label: lem-chaos-canonical-equivariance

The canonical transition kernel commutes with every permutation of slot
labels. Consequently it preserves exchangeability, including the marked
alive/dead state and the cemetery event.
:::

:::{prf:proof}
Let $\pi$ relabel an input array. Transport a donor index $j$ to $\pi(j)$,
each row innovation to the corresponding relabelled row, and each component
rotation to the relabelled vertex set. Distances, eligible donor sets, and
normalization sums are unchanged by this operation. The conditional
probability of every transported donor choice is therefore unchanged.
The empirical reward and diversity statistics are symmetric sums; thus fitness,
acceptance probabilities, and transported gate outcomes agree.

The accepted undirected graph is relabelled by $\pi$. Its components, their
sizes, their frozen position sources, and their centre-of-mass velocities
are transported exactly. Assigning independent Haar matrices to components
has the same joint law after any such component permutation. With those
matrices transported, the component update commutes pointwise with $\pi$.
Row jitter, the prescribed force evaluations, BAOAB noises, cap, and terminal
boundary tests do likewise. Extinction depends on the number of eligible
slots, which is unchanged. Integrating the transported innovations proves
the kernel identity. This is a statement about the random kernel: keeping
numerical random addresses fixed while relabelling the inputs need not give
the same sample path.
:::

:::{prf:lemma} Concentration of the actual sampled measurement marks
:label: lem-chaos-sampled-marks

Let $\eta_\mu$ be the limiting type law $(z,y_D,F)$ constructed from the
weighted measurement companion and the actual sampled fitness in
{doc}`08_mean_field`. Under {prf:ref}`def-chaos-canonical-regime`, the
finite empirical type law $\eta_N$ converges in probability to $\eta_\mu$.
Every bounded continuous type test converges in $L^2$ as well. In the box,
the random diversity mean and second moment have conditional variances
bounded by constants times $1/N$.
:::

:::{prf:proof}
**Measurement before normalization.** Condition on the deterministic input
array. A live row's companion has probabilities

$$
p^D_{ij}=\frac{1_{\{a_j=1,j\ne i\}}w_D(z_i,z_j)}
 {\sum_{k:a_k=1,k\ne i}w_D(z_i,z_k)}.
$$

The full empirical-kernel denominator is at least $\kappa_Dm_*$. Removing
the self atom changes the normalized companion law by at most
$2/[\kappa_D(m_*N-1)]$ in full variation, for sufficiently large $N$.
The companion draws of distinct measurement rows are conditionally
independent, although the selected indices can coincide. For any bounded
test $g$ of $(z,y_D)$, the conditional variance of its empirical average is
at most $\|g\|_\infty^2/N$. Its conditional expectation tends to the integral
against the weighted joint law, by {prf:ref}`lem-distance-continuity` and the
self-exclusion bound. Assign a dummy measurement mark to dead rows; their
contribution is a deterministic empirical integral.

**The global statistics.** Apply the same calculation to the sampled
bounded diversity and its square, restricted to live rows, and divide by
the deterministic alive fraction. Their conditional variances are bounded
by constants times $1/(m_*^2N)$. Reward statistics are deterministic functions
of the input array. In the box they converge by boundedness; for quadratic
reward in the unbounded case the $(4+\delta)$ position moment gives uniform
integrability of squared reward. {prf:ref}`lem-reward-continuity` applies.

**Fitness remains a mark.** Replace only the empirical normalization
statistics by their limits. Positive regularizers and positive bounded
rescale maps make the fitness map continuous. On bounded reward sets the
replacement error tends uniformly to zero; the unbounded case follows by
restricting the row type to a compact set and then using tightness. The
sampled diversity itself is retained in every row's fitness. It is not
replaced by its mean. Combining this replacement with the preceding
empirical joint-law convergence proves convergence of $\eta_N$. A bounded
continuous type test is uniformly bounded, so convergence in probability
also gives its $L^2$ convergence.
:::

:::{prf:lemma} Uniform collision-component truncation
:label: lem-chaos-component-truncation

Condition on the input swarm and every measurement mark. For sufficiently
large $N$ with at least $m_*N$ live rows, each accepted recipient-to-donor
edge has probability at most $C/N$, where one may take
$C=2/(\kappa_Cm_*)$. Let $\mathcal C_N(i)$ be the component of a tagged
vertex $i$. Then

$$
\mathbb E[|\mathcal C_N(i)|\mid S,\text{measurements}]\le e^{2C},
\qquad
\mathbb P(\operatorname{rad}_i\mathcal C_N(i)\ge r\mid S,\text{measurements})
 \le\frac{(2C)^r}{r!},
$$

and $\mathbb P(|\mathcal C_N(i)|>K\mid S,\text{measurements})\le e^{2C}/K$.
These bounds require no small-acceptance assumption.
:::

:::{prf:proof}
A live row has at most one accepted outgoing edge, and that edge strictly
increases frozen fitness. A dead row has one outgoing revival edge to a live
row and cannot be a target. Put dead vertices below live vertices in a total
ordering compatible with fitness, breaking ties arbitrarily. Every edge
strictly increases this ordering. An undirected cycle would have as many
edges as vertices; because each vertex has at most one outgoing edge, all
cycle vertices would have exactly one outgoing cycle edge, giving a directed
cycle. This contradicts strict ordering. The graph is a forest.

Consider a simple length-$\ell$ path from $i$. Along that path arrows cannot
point outward in both directions from an internal vertex. They therefore
point toward a single sink, with an increasing leg of length $a$ and a
decreasing leg of length $\ell-a$, for some $0\le a\le\ell$.
There are at most $N^a/a!$ possible label sequences on the increasing leg:
choose its labels, whose order is then determined. There are at most
$N^{\ell-a}/(\ell-a)!$ choices for the other leg. Ignoring overlap and
additional constraints at the sink only enlarges this bound.

Each path edge uses a distinct recipient's outgoing draw. Conditional on
all measurement marks these row draws are independent, and each required
edge has probability at most $C/N$. This includes weighted mandatory
revival: its denominator is at least $\kappa_Cm_*N$. Thus the expected
number of such paths is at most

$$
\sum_{a=0}^{\ell}\frac{C^\ell}{a!(\ell-a)!}
 =\frac{(2C)^\ell}{\ell!}.
$$

A vertex at radius $r$ supplies a length-$r$ path. Summing the path bound
over $\ell\ge0$ bounds the number of vertices by $e^{2C}$. Markov's
inequality gives the size-tail estimate. Averaging over measurement marks
preserves all bounds. The constant can be large; the proof asserts finite
component control, not a sharp practical estimate of component size.
:::

:::{prf:lemma} Every collision-component moment is uniformly bounded
:label: lem-chaos-component-moments

Under the conditioning of {prf:ref}`lem-chaos-component-truncation`, for
each integer $p\geq1$ and every fixed root $i$,

$$
\mathbb E[|\mathcal C_N(i)|^p\mid S,\text{measurements}]
\leq M_p(C),\qquad
M_p(C)=e^{(2^p-1)C}
\sum_{k=0}^{\infty}[(k+1)^p-k^p]\frac{C^k}{k!}<\infty.
$$

The estimate also holds when any fixed subset of rows has its outgoing
edge removed. It uses neither weak acceptance nor a subcritical branching
assumption. In particular,
$M_2(C)=(1+2C)e^{4C}$ and
$M_3(C)=(1+6C+3C^2)e^{8C}$ are admissible constants.
:::

:::{prf:proof}
Freeze all measurement marks, including their global normalization
statistics. Order the vertices by fitness, with dead rows below live rows.
Every accepted edge increases this order, and different rows choose their
outgoing edges independently. Starting from $i$, follow its outgoing edges
to the unique terminal vertex, and let $A$ denote the number of vertices
in this ancestor chain. The increasing-path count gives

$$
\mathbb P(A\geq k+1\mid S,\text{measurements})\leq C^k/k!,
\qquad
\mathbb EA^p\leq
\sum_{k\geq0}[(k+1)^p-k^p]C^k/k!.
$$

Condition now on an exact realized ancestor chain, including the terminal
row's absent outgoing edge. This event only specifies draws of chain
rows; all other row draws retain their independent distributions. The
entire component is the descendant closure of that chain. Initialize the
discovered set with its $A$ vertices and scan every other row in decreasing
order. At the time a row is scanned, all its possible parents have already
been scanned or are chain seeds. It joins the component exactly when its
outgoing edge hits the currently discovered set. If that set has size $s$,
the conditional probability of joining is at most $Cs/N$.

For $s\geq1$,
$(s+1)^p-s^p\leq(2^p-1)s^{p-1}$. Hence each scan step satisfies

$$
\mathbb E[S_{\mathrm{new}}^p\mid S_{\mathrm{old}}=s,
 \text{previous scan outcomes},\text{chain}]
\leq[1+(2^p-1)C/N]s^p.
$$

This inequality remains valid when $Cs/N>1$, since it is used only as an
upper bound on a probability. There are at most $N$ scan steps, giving
$\mathbb E[|\mathcal C_N(i)|^p\mid\text{chain}]
\leq e^{(2^p-1)C}A^p$. Average over the chain. Removing outgoing rows
preserves independence, the order, and the edge-probability bound, so the
same proof applies. The factorial series converges for every fixed $p$.
:::

:::{prf:theorem} Rooted collision convergence and two-root independence
:label: thm-chaos-rooted-collision-limit

Under {prf:ref}`def-chaos-canonical-regime`, the component around a uniformly
sampled root converges to the marked rooted component used to define
$\mathcal J(\mu)$ in {doc}`08_mean_field`. Two uniformly sampled distinct
roots converge jointly to independent copies of that rooted law. Consequently,
for every bounded continuous test $\varphi$ of the post-clone state,

$$
\mathbb E L_N^{\rm cl}\varphi\longrightarrow\mathcal J(\mu)\varphi,
\qquad
\mathbb E\big|L_N^{\rm cl}\varphi-\mathcal J(\mu)\varphi\big|^2
 \longrightarrow0.
$$

The collision readout uses one shared rotation per component, frozen
velocities of every component member including revived rows, and independent
jitter for every accepted recipient.
:::

:::{prf:proof}
**1. Freeze the type array.** By {prf:ref}`lem-chaos-sampled-marks`, every
subsequence has a further subsequence on which the empirical types converge
almost surely to $\eta_\mu$. It suffices to prove the assertion for each
such deterministic convergent type sequence. Write $t=(z,y_D,F)$ and

$$
\beta_\mu(t,u)=
\frac{1_{\{a_u=1\}}w_C(z_t,z_u)}{Z_C(\mu;z_t)}
\begin{cases}
 \min\{1,[(F_u-F_t)/(s_c(F_t+\epsilon_c))]_+\},&a_t=1,\\
 1,&a_t=0.
\end{cases}
$$

Here $Z_C(\mu;z)=\int1_{\{a_y=1\}}w_C(z,y)\mu(dy)$.
The finite edge probabilities are $N^{-1}\beta$ evaluated using the
empirical law, with the explicit self-exclusion correction. The kernels
are bounded by $C$, continuous in their type arguments on each status
stratum, and their integrals converge. Fitness ties cause no discontinuity
because acceptance vanishes continuously at a tie.

**2. Explore a bounded number of vertices.** Expose the root's type and its
outgoing edge, if present. A newly exposed target has its own free outgoing
draw. To discover incoming edges to a target $t$, scan the still unexposed
rows. Each row independently hits $t$ with probability at most $C/N$.
For finitely many targets and type-test bins, their incoming counts converge
to independent Poisson counts with intensities
$\eta_\mu(du)\beta_\mu(u,t)$.

Here is the rare-event estimate underlying that assertion. A row that hits
one of $k$ specified targets has a categorical law of total probability
$p_j\le Ck/N$. Couple it to independent Poisson counts with the same
category means. Expanding $e^{-p_j}$ shows that the probability of a mismatch
is at most $2p_j^2$ for $p_j\le1/2$: the errors are absence versus one-hit
probabilities and the Poisson probability of two or more hits. Summing over
rows bounds the mismatch by $2C^2k^2/N$. For bounded nonnegative mark tests
$g$, the same limit is read directly from

$$
\prod_j\left[1+\sum_{r=1}^k p_{jr}
       (e^{-g_r(t_j)}-1)\right].
$$

Taking logarithms changes this product to the exponential of the sum of its
linear terms with an error $O(C^2k^2/N)$. Empirical kernel convergence
identifies the limiting marked Poisson intensities. This establishes the
point-process statement, rather than just unmarked count convergence.

If a row has already been found not to hit an earlier exposed target, its
remaining hit probabilities are divided by $1-p_j^{\rm old}$, with
$p_j^{\rm old}\le Ck/N$. The resulting total change is $O(C^2k^2/N)$.
Removing the finitely many exposed labels contributes another $O(Ck^2/N)$.
For an exploration stopped after $K$ vertices, summing these estimates gives
an error at most $A(C,K)/N$, with a finite constant, in addition to the
converging empirical kernel integrals. One may use a constant of order
$(1+C)^2K^3$; no uniform estimate in unbounded $K$ is needed here.

**3. Respect the direction information.** An incoming child has already
used its outgoing choice to select its parent. It receives no second free
outgoing draw. Its additional incoming children form the corresponding
Poisson process. When a known source already points to a newly exposed
target, it is excluded from that target's additional incoming process.
These rules are exactly the rooted construction in {doc}`08_mean_field`.
They retain shared donors and all component members.

**4. Remove truncation.** The chance that a root requires more than $K$
vertices is at most $e^{2C}/K$ by
{prf:ref}`lem-chaos-component-truncation`. The same bound holds in the limit:
apply the finite-exploration convergence to the event of discovering $K+1$
vertices. Thus the limiting component is finite almost surely. First take
$N\to\infty$ with $K$ fixed, then $K\to\infty$.
On a finite typed graph, the centre-of-mass formula, shared Haar rotation,
copying, and jitter are continuous readouts. Their innovation laws agree
in the finite and limiting constructions. Boundedness of $\varphi$ controls
the discarded events and proves the one-root assertion.

**5. Explore two roots.** Explore the two components together with a
$K$-vertex cutoff for each. An outgoing draw has probability at most
$2CK/N$ to hit an exposed vertex; the joint incoming Poisson approximation
above also controls rows that could connect the two explorations. The chance
of a label collision tends to zero. The limiting incoming processes and
free outgoing draws are independent between the two explorations. Their
component rotations are independent unless the components intersect, an
event whose probability vanishes. More directly, for two uniform distinct
labels $I,J$,

$$
\mathbb P(J\in\mathcal C_N(I))
 =\frac{\mathbb E(|\mathcal C_N(I)|-1)}{N-1}
 \le\frac{e^{2C}-1}{N-1}.
$$

The empirical normalization statistics have a deterministic limit by
Step 1, so they leave no additional common random variable. Remove both
cutoffs to obtain independent rooted limits.

Finally express the second moment of an empirical average as its diagonal
$1/N$ contribution plus the expectation at two uniform distinct roots.
The latter converges to $(\mathcal J(\mu)\varphi)^2$, and the first moment
converges to $\mathcal J(\mu)\varphi$. This proves the displayed $L^2$
consistency without assuming independent collision outputs.
:::

:::{prf:lemma} Continuity of the canonical population map
:label: lem-chaos-canonical-map-continuity

On the convergence class of {prf:ref}`def-chaos-canonical-regime`, the actual
clone/collision map $\mathcal J$ and full step $\mathcal F_h$ are continuous
for weak convergence, with the stated moment control in the unbounded case.
In particular, their restrictions to any compact subset of this class are
uniformly continuous for a metric inducing that convergence.
:::

:::{prf:proof}
For $\mu_j\to\mu$, positive alive mass bounds all donor denominators away
from zero eventually. The companion and reward lemmas give convergence of
the type laws and their normalization statistics. The rooted construction
truncated at $K$ involves only finitely many kernel integrations,
Poisson intensities, and continuous finite-component readouts. Each is
continuous in $\mu$. The component bound is uniform along this sequence;
letting $K\to\infty$ proves continuity of $\mathcal J$.

For the canonical kinetic step, the deterministic BAOAB kicks and drifts
are continuous because the force is globally Lipschitz. Its O step uses
$c=e^{-\gamma h}$ and the exact integrated constant-noise variance
$s_h^2=(1-e^{-2\gamma h})/(2\gamma)$, with $s_h^2=h$ at $\gamma=0$.
Independent row noise acts through continuous Markov kernels. The radial
velocity cap is continuous. Final position noise has a strictly positive
Gaussian variance; its convolution gives zero mass to the box boundary.
Thus terminal status classification is continuous almost everywhere under
the entering limiting noise law, which suffices for convergence of bounded
continuous marked tests. In the unbounded configuration there is no such
boundary discontinuity. Finite-horizon moment bounds give tightness and the
uniform integrability needed in the unbounded reward passage.

Uniform continuity on a compact subset follows by contradiction: two
sequences whose input distance tends to zero have convergent subsequences
with the same limit; continuity then makes their output distance tend to
zero. This argument supplies a continuity modulus. It does not assert a
Lipschitz constant or contraction.
:::

:::{prf:theorem} Full one-step consistency for the canonical algorithm
:label: thm-chaos-canonical-one-step

For the deterministic input arrays of
{prf:ref}`def-chaos-canonical-regime`, let $L_N'$ be the empirical marked
law after the actual full step. On total extinction, assign it any fixed
probability measure. For every bounded continuous $\varphi$,

$$
\mathbb E\left|L_N'\varphi-\mathcal F_h(\mu)\varphi\right|^2\to0.
$$

For a bounded metric $d_*$ inducing weak convergence,

$$
\mathbb E d_*(L_N',\mathcal F_h(\mu_N))\to0.
$$

These are conclusions of the specified donor, fitness, collision, and
kinetic rules; they are not assumed consistency hypotheses.
:::

:::{prf:proof}
The rooted theorem proves empirical concentration immediately after cloning.
A continuous deterministic row map transports this concentration. For an
independent-noise row kernel $P$, condition on its input population. The
empirical test average has conditional variance at most
$\|\varphi\|_\infty^2/N$ and conditional mean $L_NP\varphi$.
The latter converges because $P\varphi$ is bounded continuous. Iterating
this observation over B1, A1, O, A2, B2, final position noise, and cap proves
the full-step statement before classification. For the classification step,
integrating the final Gaussian position noise gives a continuous kernel even
for the alive/dead marked readout: its only spatial discontinuity is a box
boundary of Gaussian measure zero. The exact boundary schedule is essential.

The alive-fraction estimate of {doc}`08_mean_field` makes total-extinction
probability tend to zero, uniformly on admitted box inputs. Consequently
the arbitrary cemetery extension has no effect on any bounded limit.
The moment bounds there provide tightness. Convergence of bounded tests
from a countable determining family implies weak convergence in probability
of the empirical law; boundedness of $d_*$ gives convergence of its expected
distance to $\mathcal F_h(\mu)$. The triangle inequality and
{prf:ref}`lem-chaos-canonical-map-continuity` replace $\mu$ by $\mu_N$.
:::

(sec-chaos-identification)=
## 4. Finite-Horizon Chaos at Fixed Timestep

:::{div} feynman-prose
We now have the crucial comparison: start a large swarm with a known
population distribution, run one complete update, and its empirical
output approaches the distribution prescribed by $\mathcal F_h$.
The same reasoning can be applied to the next update. A fixed number of
repetitions requires continuity and survival over that horizon. It does not
require the evolution to forget its initial state.

That last distinction matters for the long-time problem. A map can be
continuous and accurately approximated by large swarms without carrying all
initial populations toward the same equilibrium. The finite-time proof and
the stationary-attraction problem therefore have different final steps.
:::

:::{prf:theorem} Finite-horizon propagation of chaos for the actual update
:label: thm-chaos-finite-time-consistency

Let $S_0^N$ have exchangeable initial laws with
$L_N(S_0^N)\to\mu_0$ in probability, where $\mu_0$ is deterministic and has
positive alive mass. Assume the canonical regime above, including its
initial moment condition in the unbounded case. Define

$$
\mu_{n+1}=\mathcal F_h(\mu_n).
$$

For every fixed integer $n\ge0$,

$$
L_N(S_n^N)\longrightarrow\mu_n\quad\text{in probability},
\qquad
\mathcal L(z_{1,n}^N,\ldots,z_{\ell,n}^N)
 \Rightarrow\mu_n^{\otimes\ell}
\quad\text{for every fixed }\ell.
$$

The empirical trajectory at any fixed finite list of update indices converges
jointly to the corresponding deterministic trajectory. At every fixed
horizon, extinction probability tends to zero.
:::

:::{prf:proof}
**Pass from deterministic arrays to random inputs.** The one-step theorem
has the sequential property that every deterministic sequence of admissible
arrays with empirical limit $\mu$ has output empirical limit
$\mathcal F_h(\mu)$. Suppose a random sequence converges in probability to
that same $\mu$. From every subsequence choose a further one with almost-sure
empirical convergence. If the conditional expected one-step error did not
tend to zero in probability, there would be input realizations converging to
$\mu$ for which that error remains bounded away from zero. This contradicts
the deterministic sequential property. Boundedness then turns convergence
in probability of the conditional error into convergence of its expectation.
For unbounded rewards, restrict first to a compact set in a Wasserstein
$p$ topology with $4<p<4+\delta$; the uniform $(4+\delta)$ moment bounds
make the complement arbitrarily unlikely. The same contradiction argument
applies on each such compact set.

**Iterate the actual map.** Assume the claim at update $n$. The terminal
alive-fraction bound gives a positive limiting alive mass and makes an
exceptionally small finite alive count negligible. The moment bounds keep
the input sequence in the convergence class. Apply the random-input argument
and continuity of $\mathcal F_h$ to obtain the claim at update $n+1$.
This begins with the stated initialization. The union bound over a fixed
number of updates gives the extinction conclusion and joint empirical
trajectory convergence.

**Convert empirical convergence to particle chaos.** Kernel equivariance
preserves exchangeability at every update. The sampling-without-replacement
argument in {prf:ref}`lem-empirical-convergence` compares the first $\ell$
coordinates to $\ell$ independent samples from the empirical law, with
error at most $\ell(\ell-1)/N$ on bounded product tests. The deterministic
empirical limit then gives $\mu_n^{\otimes\ell}$.
:::

:::{prf:remark} Quantitative errors and the continuity modulus
:label: rem-chaos-discrete-error-modulus

For the canonical kernel, {prf:ref}`thm-chaos-canonical-quantitative-bias`
and {prf:ref}`thm-chaos-canonical-conditional-variance` give an
$O(N^{-1/2})$ conditional bias and an $O(N^{-1})$ conditional variance
for each bounded observable, relative to $\mathcal F_h(\mu_N)$ at the
entering empirical law. These estimates do not prescribe the rate at
which an arbitrary entering sequence $\mu_N$ approaches a specified
law $\mu$, nor a rate in every metric on probability measures. Iterating
an empirical-distance estimate also requires control of that metric and
of the map's continuity modulus. The geometric cluster argument for
contraction remains distinct from these one-step fluctuation estimates.

On a compact invariant class let $\omega$ be the continuity modulus of
$\mathcal F_h$, and let $e_n$ denote the expected bounded empirical distance
to $\mu_n$. If $\delta_N$ bounds the expected one-step error on that class,
then, for every $r>0$,

$$
e_{n+1}\le\delta_N+\omega(r)+D_*e_n/r,
$$

where $D_*$ bounds the metric diameter. Indeed, on the event that the input
distance is at most $r$, the output-map distance is at most $\omega(r)$;
on its complement use $D_*$ and Markov's inequality. Tightness permits a
compact restriction with an additional arbitrarily small exceptional
probability. Choose $r$, then $N$, to iterate convergence over fixed $n$.
This uses the proved continuity, without replacing it by an unproved
Lipschitz or contractive estimate.
:::

:::{prf:remark} Random initial populations
:label: rem-chaos-random-initial-law

If the initial empirical law converges to a random directing measure $M_0$,
the same argument, conditional on that limit, gives
$M_n=\mathcal F_h^n(M_0)$. Fixed particle marginals converge to
$\mathbb E[M_n^{\otimes\ell}]$. This preserves the common macroscopic
randomness. It reduces to deterministic chaos exactly when the directing
measure is almost surely fixed at the observation time.
:::

:::{prf:remark} Fixed timestep, continuous time, and surviving trajectories
:label: rem-chaos-time-and-conditioning

The theorem concerns the actual map at fixed $h$. It does not replace
simultaneous cloning by a finite-rate jump process. A continuous-time
identification requires convergence of the iterates
$\mathcal F_h^{\lfloor t/h\rfloor}$ for the specified parameter family,
and finite-population errors controlled over the same growing number of
updates. The complete update, including collision, immediate revival,
jitter, and repeated cap, must enter that analysis.

The finite-horizon theorem uses an arbitrary extension after total
extinction, with vanishing probability of visiting that extension. The law
conditioned on survival through the entire fixed horizon has the same limit:
conditioning changes any bounded test expectation by at most twice its
supremum times the extinction probability. Conditioning each intermediate
transition separately is a different path-law operation.
:::

:::{prf:lemma} The actual boundary and revival contributions converge
:label: lem-boundary-convergence

In the canonical box regime, every dead recipient takes a current live donor
with the Gaussian weight evaluated from its retained coordinates, receives
the prescribed jitter and component collision, and is marked alive before
kinetics. The empirical post-revival law and the terminal alive and dead
subprobabilities converge to these same stages of $\mathcal F_h$.
:::

:::{prf:proof}
The dead-row density $\beta_\mu(t,u)$ in the rooted theorem integrates to
one over live targets and contains the actual retained-position donor
weight. Thus the rooted output includes every revived recipient and its
shared component rotation. There is no unrevived finite-rate reservoir at
this stage. The kinetic consistency proof supplies the terminal position
law. Its Gaussian final-noise convolution has zero boundary mass, so
restriction to the box and its complement converges by the continuity-set
criterion. These restrictions give the alive and dead subprobabilities;
their sum is the full marked probability.
:::

### 4.1. An additional variance tool

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

:::{div} feynman-prose
Hold the input swarm fixed and repeat one complete update. How much can its
empirical average fluctuate? Replace just one measurement draw. Every fitness
normalizer must be recomputed, so even distant walkers can acquire different
gate probabilities. A changed gate can then join or split collision groups,
changing the shared rotation experienced by several walkers. The following
proof keeps this entire chain of effects. Its key estimate bounds the
expected square of the number of affected output slots independently of
$N$; it does not require a deterministic bound on that number.

There are order $N$ independent innovation blocks, while each affected slot
contributes order $1/N$ to an empirical average. The squared-influence
estimate therefore gives conditional variance of order $1/N$. The separate
bias estimate compares the average output with the actual population map
at the same empirical input. That bias is order $N^{-1/2}$, giving a
mean-square one-step error of order $1/N$ with configuration constants
independent of population size.

Collision components serve a specific purpose here: they measure how far
fresh randomness can spread during one update. The geometric fitness and
error clusters have a different job in the stationary argument: they track
the evolution of error already present in the input swarm. The one-step
noise estimate supplies that argument's fluctuation term. Closing the
stationary estimate still requires controlling the error flux between those
clusters, including its sign for small errors.
:::

:::{prf:lemma} Squared influence of the actual sampled measurement
:label: lem-chaos-canonical-innovation-replacement

In either canonical regime, fix an input with at least $m_*N$ alive slots.
Couple two updates by replacing one row's measurement companion and using
identical remaining row innovations and identical rotations on every
unchanged component. Recompute the global fitness statistics in both
updates. Let $D_r$ be the number of output rows that can differ, before any
collapse of the entire marked output to a cemetery state. There is an
explicit finite configuration constant $A_D$, independent of $N$, such that

$$
\mathbb E[D_r^2\mid S]\leq A_D.
$$

Replacing one row's cloning donor and gate block instead gives the bound
$9M_2(C)$. A local jitter or kinetic block affects at most one final row;
replacing the rotation assigned to a component affects only that component.
:::

:::{prf:proof}
**1. Quantify every changed acceptance probability.** Let $S_*$ bound the
nonnegative squashed separation $s$, let $\sigma_s>0$ be its standardization
regularizer, and use the fitness parameters of
{prf:ref}`def-mean-field-fitness-potential`. Changing one sampled separation
changes the alive empirical mean by at most $S_*/(m_*N)$, its second moment
by at most $S_*^2/(m_*N)$, and its variance by at most
$3S_*^2/(m_*N)$. Consequently, for every unchanged measurement row, its
standardized separation changes by at most $L_q/N$, where

$$
L_q=\frac{S_*}{m_*\sigma_s}
 +\frac{3S_*^3}{2m_*\sigma_s^3}.
$$

Reward statistics are unchanged. Define

$$
H_s=(A_r+\eta_r)^{p_r}\frac{A_s}{4}
 p_s\max\{\eta_s^{p_s-1},(A_s+\eta_s)^{p_s-1}\},
\qquad L_0=H_sL_q,
$$

with $H_s=0$ when $p_s=0$. The logistic derivative is at most $A_s/4$,
so $|F_i-\widetilde F_i|\leq L_0/N$ for $i\ne r$. This uses the bounded
reward *rescaling factor*, and is valid even when physical rewards or
retained dead coordinates are unbounded. Row $r$ may change by order one.
For the actual clipped cloning probability, a Lipschitz constant in the
sum of its two fitness arguments is

$$
L_a=\max\left\{\frac1{s_c(F_*+\epsilon_c)},
 \frac{F^*+\epsilon_c}{s_c(F_*+\epsilon_c)^2}\right\}.
$$

Retain identical cloning donor draws and gate uniforms. Conditional on both
complete measurement arrays, the event that row $i$'s accepted outgoing
edge changes has probability $q_i\leq B/N$ for $i\ne r$, where
$B=C+2L_aL_0$. The $C/N$ term includes choosing donor $r$; otherwise both
fitness arguments change by at most $L_0/N$. Mandatory revival is covered:
its gate is identically one, and its frozen donor law is unchanged.

**2. Expose exceptional rows, not their surrounding components.** Put $r$
in an exceptional set $E$ regardless of its gate outcome, and include every
other row whose accepted outgoing edge changes. These row events are
independent under the preceding conditioning. With $Q=|E|$,

$$
\mathbb EQ^2\leq(1+B)^2+B.
$$

Suppose $N\geq2B$. Condition on $E$ and the two outgoing outcomes of every
exceptional row. Every remaining row has a common outgoing edge in the
two graphs; its conditional probability of a specified edge is at most

$$
\frac{C/N}{1-q_i}\leq\frac{2C}{N}.
$$

These remaining row draws are still independent, since the conditioning
factorizes by row. Remove the exceptional outgoing edges. The resulting
common graph is an ordered forest, using the first measurement array's
fitness order. This remains true if the two arrays order some fitnesses
differently: each common accepted edge respects both orders.

All affected vertices belong to components of this common forest meeting
an exceptional recipient or one of its two donor endpoints. There are at
most $3Q$ such seeds. The seeds are fixed under the conditioning, independently
of the remaining row draws. Restoring exceptional edges can only join
components already meeting these seeds. Thus, by Cauchy--Schwarz and
{prf:ref}`lem-chaos-component-moments`,

$$
\mathbb E[D_r^2\mid E,\text{exceptional outcomes},
 \text{both measurement arrays}]
\leq 9Q^2M_2(2C).
$$

No component-size estimate has been conditioned on the event that a
random component is unusually influential; the order of exposure above
is what permits the uniform moment bound. Averaging proves the claim with

$$
A_D=9M_2(2C)[(1+B)^2+B]+\max\{1,4B^2\}.
$$

For the omitted finite sizes $N<2B$, simply use $D_r\leq N<2B$.

**3. Couple the remaining full mechanism.** Give each possible component
representative an independent Haar rotation, using its smallest slot label
as representative. This realizes exactly one independent Haar matrix per
component. If a component is unchanged, its representative and rotation
are identical in both updates. Component copying, center-of-mass velocities,
and the prescribed correlated rotation can therefore change only rows in
the affected components. Jitter, BAOAB, final position noise, capping, and
terminal status classification are row-local under identical innovations.
They create no additional affected rows.

For replacement of one donor/gate block, remove that row's outgoing edge
from both graphs. The common forest meets at most three seeds (recipient,
old donor, new donor); the other row draws are independent with bound
$C/N$. The same squared-sum estimate gives $9M_2(C)$. The remaining support
claims follow from their actual readouts. $\square$
:::

:::{prf:theorem} Quantitative conditional concentration for the full update
:label: thm-chaos-canonical-conditional-variance

For either canonical regime and every bounded measurable marked-state test
$\varphi$, write $F=L_N'\varphi$ for the complete physical marked output,
retaining its coordinates also on total extinction. Uniformly over fixed
inputs with alive fraction at least $m_*$,

$$
\operatorname{Var}(F\mid S)
\leq\frac{A_\varphi}{N},\qquad
A_\varphi=2\|\varphi\|_\infty^2
 [A_D+10M_2(C)+1].
$$

If the output empirical law is instead assigned a fixed probability law on
extinction, the bound is
$2A_\varphi/N+8\|\varphi\|_\infty^2\delta_N$, where $\delta_N$ is the
uniform terminal-extinction bound of
{prf:ref}`cor-mean-field-positive-alive-mass`. It too is $O(N^{-1})$.
:::

:::{prf:proof}
Represent the full update by four independent innovation blocks per row:
measurement companion; cloning donor with gate; addressed Haar rotation;
and row-local jitter with both kinetic noises. The force evaluations, cap,
and terminal classification are deterministic given these blocks. This is
an exact representation of the probabilistic kernel, irrespective of how
a particular implementation streams its random numbers.

The measurement and donor/gate blocks have squared influence at most
$A_D$ and $9M_2(C)$ by the preceding lemma. Changing row $i$'s addressed
rotation changes no output unless $i$ represents its component; its squared
influence is bounded by $|\mathcal C_N(i)|^2$, hence in expectation by
$M_2(C)$. A row-local block has influence at most one. Therefore

$$
\sum_j\mathbb E[D_j^2\mid S]
\leq N[A_D+10M_2(C)+1].
$$

Substitution into {prf:ref}`lem-chaos-innovation-variance` proves the
claim. This calculation retains the actual random global normalization,
incoming cloners, shared donors, and correlated component rotations.

For the cemetery extension $\widehat F$, its difference from $F$ is at
most $2\|\varphi\|_\infty$ and is supported on extinction. The inequalities
$\operatorname{Var}(X+Y)\leq2\operatorname{Var}X+2\operatorname{Var}Y$
and $\operatorname{Var}Y\leq\mathbb EY^2$ give the stated correction.
Each term of $\delta_N$ decays exponentially in $N$; since
$\sup_{x\geq0}xe^{-cx}=1/(ce)$, that correction is bounded by a
configuration constant times $N^{-1}$. $\square$
:::

:::{prf:theorem} Quantitative one-step bias and mean-square consistency
:label: thm-chaos-canonical-quantitative-bias

In either canonical regime, fix an input array $S$ with alive fraction at
least $m_*$, and let $\mu_N=L_N(S)$. For every bounded measurable marked
observable $\varphi$, there is a finite constant $B_*$ depending only on
the fixed algorithmic parameters and $m_*$ such that

$$
\left|\mathbb E[L_N'\varphi\mid S]
 -\mathcal F_h(\mu_N)\varphi\right|
\leq\frac{2\|\varphi\|_\infty B_*}{\sqrt N}.
$$

Consequently, for the full physical marked output,

$$
\mathbb E\left[|L_N'\varphi-\mathcal F_h(\mu_N)\varphi|^2\mid S\right]
\leq\frac{A_\varphi+4\|\varphi\|_\infty^2B_*^2}{N}.
$$

Assigning a fixed empirical law on extinction adds the exponentially small
correction controlled in {prf:ref}`thm-chaos-canonical-conditional-variance`.
The comparison is to the population map at the actual empirical input;
any error between that input and another prescribed initial law is separate.
:::

:::{prf:proof}
**1. Integrate the normalization fluctuation.** This step uses a coupled
integration device to estimate the actual update; it does not redefine
its fitness. It suffices to treat $N\geq N_0$ as defined in Step 3;
smaller populations use the elementary bound there. In particular there
are at least two alive slots in the calculations below. In that device
only, replace the empirical diversity mean and
variance by the corresponding donor-weighted moments of $\mu_N$. Retain
every row's sampled measurement, all reward statistics, the same donor and
gate draws, and all subsequent collision and kinetic operations. Denote the
device's empirical output by $\widetilde L_N'$. Its measurement marks are
independent conditional on $S$, and its deterministic normalizers are exactly
those of the target population map.

Use the constants $S_*,\sigma_s,H_s,L_a$ from
{prf:ref}`lem-chaos-canonical-innovation-replacement`, and put

$$
D_D=\frac{2}{\kappa_Dm_*},\qquad
A_T=\left(m_*^{-1}+D_D^2\right)
\left(\frac{2S_*^2}{\sigma_s^2}
      +\frac{5S_*^6}{\sigma_s^6}\right),\qquad L_T=2L_aH_s.
$$

The self-exclusion error in each measurement law is at most $D_D/N$ in
full variation. Conditional independence therefore gives, for the errors
$\Delta_1$ and $\Delta_2$ of the sampled first and second separation
moments relative to their population values,

$$
\mathbb E\Delta_1^2\leq
\frac{S_*^2(m_*^{-1}+D_D^2)}N,
\qquad
\mathbb E\Delta_2^2\leq
\frac{S_*^4(m_*^{-1}+D_D^2)}N.
$$

For the variance error $\Delta_v$,
$|\Delta_v|\leq|\Delta_2|+2S_*|\Delta_1|$. Thus

$$
T=\frac{|\Delta_1|}{\sigma_s}
 +\frac{S_*|\Delta_v|}{2\sigma_s^3}
\quad\hbox{satisfies}\quad
\mathbb ET^2\leq A_T/N.
$$

Condition on all realized measurement marks. The two acceptance probabilities
of every row differ by at most $L_TT$. On $L_TT\leq1/2$, expose the set of
changed edges and their endpoint labels. Its expected size is at most
$NL_TT$. The common forest after removing these edges has independent
row probabilities bounded by $2C/N$, exactly as in the squared-influence
proof. At most three seeds per changed row are needed. Its first component
moment gives

$$
\frac{\mathbb E[D\mid\text{measurement marks}]}N
\leq3M_1(2C)L_TT
\qquad(L_TT\leq1/2).
$$

On the complementary event use $D/N\leq1$, and its probability is at most
$4L_T^2A_T/N$. Hence

$$
\left|\mathbb E[(L_N'-\widetilde L_N')\varphi\mid S]\right|
\leq2\|\varphi\|_\infty
\left[\frac{3M_1(2C)L_T\sqrt{A_T}}{\sqrt N}
      +\frac{4L_T^2A_T}{N}\right].
$$

This term explicitly restores the original random normalization.

**2. Compare the marked explorations at the same empirical input.** Set
$A=1+C+D_D$. Explore at most $K$ component vertices in both the integration
device and the rooted construction for $\mu_N$. Also retain the target and
its measurement mark of every rejected free cloning proposal. Such an
auxiliary target can be revisited later; assigning it a fresh fitness mark
would lose the algorithm's shared fitness information. There are at most
$2K$ exposed component or auxiliary labels. An incoming child already has
its outgoing edge fixed to its parent, and receives no free outgoing query.

Here are bounds for this finite exploration, valid when $K\leq N/(8A)$.
All are conditional on matching histories so far.

- For an undiscovered row, avoidance of the previous targets has probability
  at least $1-2CK/N$. Its joint measurement/edge law is tilted by that
  avoidance event. Conditioning changes the intensity of a specified new
  target by at most $4C^2K/N^2$; summing over rows gives $4C^2K/N$.
  The conditioning factorizes over undiscovered rows, preserving their
  independence. This calculation includes the bias in their measurement
  marks, instead of assuming that an unrevealed empirical type law is
  deterministic.
- A fresh donor query meets one of the at most $2K$ exposed labels with
  probability at most $4CK/N$ under the same conditioning. A freshly
  queried mark's avoidance tilt costs at most another $4CK/N$ in
  variation. Removing
  already exposed labels from an incoming scan changes its intensity by
  at most $4CK/N$. Self-exclusion of cloning targets changes the incoming
  intensity by at most $2C^2/N$ and a donor-query law by at most $C/N$.
- Replacing each finite row's self-excluded measurement law by its population
  companion law costs at most $D_D/N$ for a directly queried mark, and at
  most $2CD_D/N$ in an incoming intensity summed over all rows. These are
  integrals against the actual atomic input $\mu_N$; no empirical-law
  approximation term remains.
- In an incoming scan the conditional probability of one row hitting the
  target is at most $2C/N$. Coupling its marked Bernoulli event to a Poisson
  event of the same marked intensity costs at most twice the square of
  that probability. Summing gives at most $8C^2/N$. If several targets
  are scanned together, the categorical bound is at most $8C^2K^2/N$.
  Use the latter, larger bound for every scan.

For clarity, the intensity-conditioning estimate in the first item is
obtained before dividing: if $p$ is the probability of the new-target event
and $u\leq2CK/N$ is the probability of any earlier-target event, those
outgoing events are disjoint. Their conditioned intensity is $p/(1-u)$,
whose excess is at most $2up$. This remains an identity of marked measures
when the measurement mark is retained in the event. Known incoming children
are removed as fixed labels; their outgoing choices are never resampled.
The independent marked Poisson processes can be coupled by retaining their
common intensity and drawing their two excess intensities separately.

Each of the at most $K$ vertex-exposure stages uses one incoming scan and
at most one free donor query, with at most two new measurement queries.
Summing the displayed bounds and enlarging the constant gives total mismatch
probability at most

$$
\frac{64A^2K^3}{N}.
$$

This sum includes failed proposals' auxiliary labels and all exclusion
conditioning. It does not assert independence of two components before
excluding their shared finite labels.

**3. Remove the finite exploration.** Both the integration device and the
rooted construction have the component moment bound $M_3(C)$. For the
rooted law it follows by applying finite exploration to repeated copies
of the atomic input array and then taking its population limit; the alive
fraction and edge bound remain the same throughout. Thus the probability
that either component exceeds $K$ is at most $2M_3(C)/K^3$. On a matching
finite exploration, use identical component rotations, jitter, and row
kinetic noises. Its full marked root output is then identical, including
the terminal boundary decision. It follows that

$$
\left|\mathbb E[\widetilde L_N'\varphi\mid S]
 -\mathcal F_h(\mu_N)\varphi\right|
\leq2\|\varphi\|_\infty
\left[\frac{64A^2K^3}{N}+\frac{2M_3(C)}{K^3}\right].
$$

Choose $K=\lfloor N^{1/6}\rfloor$. When
$N\geq N_0=\lceil(8A)^{6/5}\rceil$, the required exploration condition
holds. Since $K\geq N^{1/6}/2$, the last bracket is at most
$[64A^2+16M_3(C)]/\sqrt N$. For $N<N_0$, the elementary coupling bound
one is at most $\sqrt{N_0}/\sqrt N$. Combining all steps proves the theorem
with the explicit, population-independent choice

$$
B_*=3M_1(2C)L_T\sqrt{A_T}+4L_T^2A_T
 +64A^2+16M_3(C)+\sqrt{N_0}.
$$

The mean-square identity is conditional variance plus squared conditional
bias. Apply {prf:ref}`thm-chaos-canonical-conditional-variance` to its first
term. $\square$
:::

(sec-fg-propagation-intro-uniqueness)=
## 5. Discrete Stationary Identification and Its Remaining Estimate

:::{div} feynman-prose
At stationarity a finite surviving swarm is governed by a QSD identity.
A limiting population is governed by a fixed-point equation. These equations
can be connected, but we must retain the survival factor until we have shown
that its contribution vanishes.

There is also a second possibility: different stationary swarm runs might
converge to different population laws. The object that is then stationary
is a probability distribution over population laws. We will identify that
object exactly, and state the additional concentration or attraction estimate
needed to reduce it to one fixed population.
:::

:::{prf:remark} Exact stationary balance for a killed discrete kernel
:label: rem-qsd-vs-true-stationarity

For a bounded full-swarm test $H$ extended by zero at the cemetery,

$$
\nu_N(Q_N-I)H=-(1-\alpha_N)\nu_NH.
$$

The operator $(Q_N-I)/h$ therefore has balance defect
$-(1-\alpha_N)\nu_NH/h$. This is the exact finite-step difference operator;
identifying it with a differential generator requires a further limit.
The exponential survival exponent is $-\log\alpha_N/h$, while the balance
coefficient is $(1-\alpha_N)/h$.
:::

:::{prf:theorem} Vanishing extinction contribution at fixed timestep
:label: thm-extinction-rate-vanishes

For the canonical terminal-box update, let $\delta_N$ be the exponentially
vanishing uniform probability of failing the positive alive-fraction bound
proved in {doc}`08_mean_field`. For every existing QSD of this same kernel,

$$
1-\alpha_N=\int\mathbb P_S(T_\dagger\le1)\nu_N(dS)\le\delta_N.
$$

Hence its bounded stationary balance defect tends to zero, and
$\mathbb P_{\nu_N}(T_\dagger\le n)\le n\delta_N$ for every fixed $n$.
The unbounded all-alive kernel has no boundary-extinction contribution.
:::

:::{prf:proof}
Total extinction is included in failure of the positive alive-fraction event.
Integrate the uniform one-step bound against the QSD. The QSD survival law
then gives $1-\alpha_N^n\le n(1-\alpha_N)\le n\delta_N$.
For bounded $H$, the exact balance defect is bounded by
$\delta_N\|H\|_\infty$. These estimates do not require independent
unconditional walker deaths; independence of terminal position innovations
is used inside the conditional survival proof in {doc}`08_mean_field`.
:::

:::{prf:remark} Population stability and the order of limits
:label: rem-extinction-rate-physical-interpretation

The estimate is for fixed $h$ and fixed observation horizons. At fixed $N$
it permits eventual extinction. For a joint limit with $h=h_N\to0$, even
survival alone needs $\delta_{N,h_N}/h_N\to0$; the constants in the
terminal-noise bound depend on $h$. No uniform small-timestep conclusion
follows by suppressing that dependence.
:::

:::{prf:theorem} Exact stationary variance budget for the complete update
:label: thm-chaos-qsd-variance-budget

Let $\nu_NQ_N=\alpha_N\nu_N$ be a QSD of the canonical terminal-box
kernel, and let $H(S)=L_N(S)\varphi$ for a bounded measurable marked test
$\varphi$, with $M=\|\varphi\|_\infty$. For a nonextinct input define

$$
\begin{aligned}
q_N(S)&=Q_N1(S),&
r_N(S)&=\frac{Q_NH(S)}{q_N(S)},\\
s_N(S)&=\frac{Q_N(H^2)(S)}{q_N(S)}-r_N(S)^2,&
\widetilde\nu_N(dS)&=\frac{q_N(S)}{\alpha_N}\nu_N(dS).
\end{aligned}
$$

These are respectively the one-step survival probability, the conditional
mean and variance of the empirical output given survival, and the input
law reweighted by that survival probability. Then

$$
\boxed{\operatorname{Var}_{\nu_N}(H)
 =\widetilde\nu_Ns_N+
   \operatorname{Var}_{\widetilde\nu_N}(r_N).}
$$

Let $G_N$ be the positive alive-fraction input set from
{prf:ref}`cor-mean-field-positive-alive-mass`, and use its uniform error
$\delta_N$ also as an upper bound for one-step extinction. With the constant
$A_\varphi$ of {prf:ref}`thm-chaos-canonical-conditional-variance`,

$$
0\leq \widetilde\nu_Ns_N
\leq \frac{A_\varphi}{N\alpha_N}
 +\frac{M^2\delta_N}{\alpha_N^2},
\qquad
\|\widetilde\nu_N-\nu_N\|_1
\leq\frac{2\delta_N}{\alpha_N}.
$$

In particular, the same stationary variance has the quantitative balance

$$
\left|\operatorname{Var}_{\nu_N}(H)
 -\operatorname{Var}_{\nu_N}(r_N)\right|
\leq \frac{A_\varphi}{N\alpha_N}
 +\frac{M^2\delta_N}{\alpha_N^2}
 +\frac{6M^2\delta_N}{\alpha_N}.
$$

For a bounded continuous $\varphi$, put
$g_N(S)=\mathcal F_h(L_N(S))\varphi$ and
$b_N=\nu_N|r_N-g_N|$. The proved stationary moment bounds and actual
one-step consistency give $b_N\to0$. Consequently

$$
\left|\operatorname{Var}_{\nu_N}(L_N\varphi)
 -\operatorname{Var}_{\nu_N}
   \bigl(\mathcal F_h(L_N)\varphi\bigr)\right|
\leq \frac{A_\varphi}{N\alpha_N}
 +\frac{M^2\delta_N}{\alpha_N^2}
 +\frac{6M^2\delta_N}{\alpha_N}
 +4Mb_N.
$$
:::

:::{prf:proof}
The QSD identities for $H$ and $H^2$ imply

$$
\nu_NH=\widetilde\nu_Nr_N,\qquad
\nu_NH^2=\widetilde\nu_N(s_N+r_N^2).
$$

Subtracting the square of the first identity proves the boxed formula.
It conditions on survival of the whole swarm; all incoming cloners,
component rotations, and alive/dead correlations remain inside $Q_N$.

Let $F=L_N'\varphi$ denote the physical marked empirical output, including
its retained coordinates on extinction, and let $E$ be the survival event.
For each input $S$,

$$
q_N(S)s_N(S)
\leq\mathbb E_S\!\left[
 (F-\mathbb E_SF)^2\mathbf1_E\right]
\leq\operatorname{Var}_S(F).
$$

The first inequality follows because the conditional mean on $E$ minimizes
the conditional squared error. On $G_N$ the conditional variance theorem
bounds the right side by $A_\varphi/N$; on its complement it is at most
$M^2$. The QSD equation and the uniform output good-set bound give
$\nu_N(G_N^c)\leq\delta_N/\alpha_N$. Integrating the preceding inequality
and dividing by $\alpha_N$ proves the bound on $\widetilde\nu_Ns_N$.

Since $q_N\in[1-\delta_N,1]$ and
$\alpha_N=\nu_Nq_N$, one has

$$
\int|q_N-\alpha_N|\,d\nu_N
\leq\nu_N(1-q_N)+(1-\alpha_N)
\leq2\delta_N.
$$

This proves the variation bound for the tilted input law. For a function
$|f|\leq M$ and probability laws $\rho,\eta$,

$$
|\operatorname{Var}_\rho(f)-\operatorname{Var}_\eta(f)|
\leq3M^2\|\rho-\eta\|_1.
$$

Indeed, the second moments differ by at most $M^2\|\rho-\eta\|_1$,
and the squared means differ by at most twice that amount. Apply this to
$f=r_N$ and use the exact budget.

For the last estimate, the stationary output moment bounds, the QSD
identity, and the positive alive-fraction estimate place the input family
in the compact localization class of the one-step theorem. Its uniform
localized conditional-mean consistency gives $b_N\to0$; passage from an
unconditioned mean to $r_N$ adds at most $2M\delta_N$.
Finally, for $|f|,|g|\leq M$ under the same probability law,

$$
|\operatorname{Var}(f)-\operatorname{Var}(g)|
\leq4M\,\mathbb E|f-g|.
$$

Use this with $f=r_N$ and $g=g_N$. This completes each asserted estimate.
:::

:::{prf:theorem} Deterministic QSD empirical limits are fixed points
:label: thm-limit-is-weak-solution

Suppose $\nu_N$ are exchangeable QSDs of the canonical kernel and
$L_N\to\mu_*$ in probability under $\nu_N$. Assume the stationary input
family lies in the tightness and moment class needed for the actual
one-step theorem. Then

$$
\mathcal F_h(\mu_*)=\mu_*.
$$

In the terminal-box configuration the required positive alive-fraction,
capped-velocity, and output moment bounds follow from the canonical one-step
estimates and the QSD equation; they do not require a reservoir model.
:::

:::{prf:proof}
Apply the QSD identity to $H(S)=L_N(S)\varphi$ for bounded continuous
$\varphi$. The one-step theorem and its random-input extension give
$\nu_NQ_NH\to\mathcal F_h(\mu_*)\varphi$; the cemetery convention changes
this by at most the vanishing extinction error.
Meanwhile $\alpha_N\nu_NH\to\mu_*\varphi$. Equality for a determining
family gives the fixed-point identity.

For the stated box inputs, if $G_N$ is the good alive-fraction set, then
$\alpha_N\nu_N(G_N^c)=\nu_NQ_N1_{G_N^c}\le\delta_N$.
For an output position moment $W$ uniformly bounded in conditional
expectation by $B_W$, the same identity gives
$\nu_NL_NW\le B_W/\alpha_N$. The cap holds on every admitted output.
Since $\alpha_N\to1$, these estimates supply the required stationary
input control. They use the full marked law, including dead positions.
:::

:::{prf:theorem} A stationary empirical mixture is invariant under the actual map
:label: thm-limit-is-weak-solution-summary

Under the corresponding tightness and moment conditions, if
$\Lambda_N\Rightarrow\Lambda$ for canonical QSDs, then

$$
(\mathcal F_h)_\#\Lambda=\Lambda.
$$

In particular, for every bounded continuous $\varphi$,

$$
\int\big[\mathcal F_h(\mu)\varphi-\mu\varphi\big]\Lambda(d\mu)=0.
$$

The invariant-measure assertion is stronger than this averaged weak balance.
Neither assertion alone makes the barycentre a fixed point of a nonlinear map.
:::

:::{prf:proof}
For a bounded Lipschitz $H$ on the space of population measures, the
one-step theorem and compact localization from the finite-horizon proof
show that replacing $H(L_N')$ by $H(\mathcal F_h(L_N))$ changes its
expectation by $o(1)$. The QSD identity changes the former expectation to
$\int H\,d\Lambda_N$ with error at most
$2\|H\|_\infty(1-\alpha_N)$. Continuity of $\mathcal F_h$ and the
stationary tightness/moment bounds permit passage to the limit, giving
$\int H\circ\mathcal F_h\,d\Lambda=\int H\,d\Lambda$.
This identifies the pushforward measure. Apply it also to
$H(\mu)=\mu\varphi$ to obtain the averaged balance.
:::

:::{prf:remark} Stationary existence and the precise attraction problem
:label: rem-chaos-stationary-obstruction

The compact-convex argument in {doc}`08_mean_field` proves existence of a
fixed point for the canonical terminal-box map. The consistency and
continuity proofs above do not prove uniqueness or global attraction.
To prove attraction it would suffice to establish, for this same map on its
invariant class $\mathcal K$, an integer $r\ge1$, a complete metric $d$,
and $q<1$ such that

$$
d(\mathcal F_h^r\mu,\mathcal F_h^r\eta)\le q\,d(\mu,\eta)
\qquad(\mu,\eta\in\mathcal K).
$$

No such inequality is established here for the canonical parameter values.
Its missing terms are concrete: changing the input law changes sampled
fitness normalization, accepted-edge probabilities, component membership,
and the shared rotation acting on the resulting centre of mass. The
finite-component estimate controls truncation, but its constant is not a
contraction coefficient. Independent final noise proves continuity and
survival; it does not by itself dominate these nonlinear changes.

An alternative completion is a population-uniform concentration estimate
for the actual marked QSDs, together with identification of their possible
fixed-point limits. The next section retains these two exact routes. Their
remaining hypotheses are not certified for the canonical kernel by a
continuous kinetic mixing theorem or by the existence of its fixed point.
:::

(sec-chaos-stationary-limit)=
## 6. Stationary Chaos and Macroscopic Convergence

:::{div} feynman-prose
One way to make the stationary population deterministic is to show that every
empirical measurement has vanishing variance. Another is to show that every
population law is drawn toward the same fixed point under repeated actual
updates. The first is a concentration statement about the stationary swarm;
the second is a dynamical statement about $\mathcal F_h$.

Both arguments are useful, and neither should be confused with the
finite-horizon result already proved. That result starts with a concentrated
population. A stationary swarm still needs a reason to be concentrated.
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

:::{prf:corollary} Stationary identification using actual QSD concentration
:label: cor-chaos-lsi-stationary-limit

Suppose the canonical QSD family has the tightness and moment control above,
the concentration criterion holds along every convergent subsequence, and
$\mathcal F_h$ has exactly one fixed point $\mu_*$ in the class of possible
limits. Then $\Lambda_N\Rightarrow\delta_{\mu_*}$ and
$\nu_N^{(\ell)}\Rightarrow\mu_*^{\otimes\ell}$ for every fixed $\ell$.

The canonical one-step consistency and extinction contributions have been
proved above. Population-uniform QSD concentration and uniqueness of the
actual fixed point remain additional, explicitly unverified applications of
this corollary.
:::

:::{prf:proof}
Take any subsequence. Tightness gives a further subsequence of first marginals
converging to $\bar\mu$. The concentration criterion makes its empirical-law
limit $\delta_{\bar\mu}$. The discrete stationary-identification theorem
then gives $\mathcal F_h(\bar\mu)=\bar\mu$, hence $\bar\mu=\mu_*$ by
the assumed uniqueness. All subsequences have the same limit. Apply
{prf:ref}`lem-empirical-convergence` for fixed marginal convergence.
:::

### 6.2. Identification by the full discrete evolution

:::{prf:lemma} Explicit one-step stability of the full marked population law
:label: lem-chaos-full-map-variation-stability

Consider two input laws of the canonical terminal-box algorithm,

$$
\mu=m\rho_A\otimes\delta_1+(1-m)\rho_D\otimes\delta_0,
\qquad
\eta=n\zeta_A\otimes\delta_1+(1-n)\zeta_D\otimes\delta_0,
$$

with $m,n\geq m_*>0$, alive positions in the declared box, and all
velocities capped. The conditional dead laws retain physical positions.
When a dead mass is zero, choose any conditional dead law for its
zero-weight term. Use full variation norms and put

$$
\Delta=|m-n|+\|\rho_A-\zeta_A\|_1
                   +\|\rho_D-\zeta_D\|_1.
$$

The following constants are determined by the actual parameters. Let
$\kappa_D,\kappa_C$ be the Gaussian measurement and donor weight lower
bounds, $R$ the oscillation of reward on the box, and $D$ a bound on the
oscillation of the raw diversity measurement. For the canonical logistic
maps $g_j(u)=\eta_j+A_j/(1+e^{-u})$, standardization regularizers
$\sigma_j>0$, and fitness exponents $p_j\geq0$, $j\in\{R,D\}$, define

$$
\begin{aligned}
G_j&=\eta_j+A_j,\qquad
T_j=\frac{A_jp_j}{4}
       \max\{\eta_j^{p_j-1},G_j^{p_j-1}\},\\
L_M&=2+\frac2{\kappa_D},\qquad
Q_R=\frac R{\sigma_R}+\frac{3R^3}{2\sigma_R^3},\qquad
Q_D=L_M\left(\frac D{\sigma_D}
                    +\frac{3D^3}{2\sigma_D^3}\right),\\
L_F&=T_RG_D^{p_D}Q_R+T_DG_R^{p_R}Q_D,\qquad
F_*=\eta_R^{p_R}\eta_D^{p_D},\qquad
F^*=G_R^{p_R}G_D^{p_D}.
\end{aligned}
$$

Set $T_j=0$ when $p_j=0$. If $\epsilon_a>0$ and $s_a>0$ are the actual
acceptance denominator and saturation parameters, put

$$
\begin{aligned}
L_a&=\max\left\{\frac1{s_a(F_*+\epsilon_a)},
 \frac{F^*+\epsilon_a}{s_a(F_*+\epsilon_a)^2}\right\},\\
C&=\frac1{\kappa_Cm_*},\qquad
L_\beta=2C^2+2CL_aL_F,\qquad
B=2CL_M+L_\beta,\\
L_{\mathrm{step}}&=2\left[L_M+2e^{2C}B\right].
\end{aligned}
$$

Then the complete actual population map satisfies

$$
\boxed{\|\mathcal F_h(\mu)-\mathcal F_h(\eta)\|_1
       \leq L_{\mathrm{step}}\Delta.}
$$

For the tagged frozen source position $X$ and jitter gate $J$ immediately
before recipient jitter, the stronger bound

$$
\|\operatorname{Law}_\mu(X,J)
 -\operatorname{Law}_\eta(X,J)\|_1
\leq L_{\mathrm{pos}}\Delta,
\qquad
L_{\mathrm{pos}}=2\left[L_M(1+2/\kappa_C)+2L_aL_F\right]
$$

is independent of $m_*$. Both constants are independent of $N$;
$L_{\mathrm{step}}$ is a stability bound and need not be smaller than one.
:::

:::{prf:proof}
First couple physical input types and their actual measurement companions.
Normalization of a weighted donor law with weights in $[\kappa_D,1]$
has variation bound $2/\kappa_D$. Also
$\|\mu-\eta\|_1\leq2\Delta$. Combining the alive and dead branches gives
a coupling whose probability of differing physical or measurement types is
at most $L_M\Delta$. This couples physical samples, rather than treating
fitness values computed with two different normalizers as identical marks.

Raw reward may be shifted to $[0,R]$ without changing standardization.
Its mean and variance differences are bounded by $R\Delta$ and
$3R^2\Delta$. The derivative of
$u\mapsto\sqrt{u+\sigma_R^2}$ is at most $1/(2\sigma_R)$.
On matched physical samples these bounds give a standardized reward
difference at most $Q_R\Delta$. Apply the same calculation to the
measurement-pair law to get the diversity bound $Q_D\Delta$.
The derivative of $g_j^{p_j}$ is bounded by $T_j$. Expanding the two
fitness factors therefore gives a difference at most $L_F\Delta$ on
matched physical and measurement types.

The actual clipped acceptance function is

$$
a(f,g)=\min\left\{1,\max\left\{0,
                   \frac{g-f}{s_a(f+\epsilon_a)}\right\}\right\}.
$$

Clipping is nonexpansive. Its two unclipped derivatives on
$[F_*,F^*]^2$ have absolute value at most $L_a$, so matched live
recipient--donor pairs have acceptance differences at most
$2L_aL_F\Delta$. Matched revival pairs have equal acceptance one.
The donor normalization integral changes by at most $2\Delta$ and is
at least $\kappa_Cm_*$. Thus the accepted edge densities, expressed
against their full marked type laws, are bounded by $C$ and differ by at
most $L_\beta\Delta$ on matched source and target types.

Use the same coupling of base types when exploring the two actual limiting
marked collision components. For a matched vertex, couple the outgoing
accepted subprobability measure, including its no-edge outcome. Couple
incoming Poisson point measures by their common intensity; these point
measures describe incoming labels at a fixed update, not a different time
clock. In either direction the unmatched intensity is at most

$$
2CL_M\Delta+L_\beta\Delta=B\Delta.
$$

The first term accounts for unmatched base types on the two sides, and
the second for the difference of edge densities on matched types. The
probability of an incoming mismatch is at most its unmatched intensity.
The same bound applies to the outgoing subprobability, with its remaining
mass coupled at the no-edge outcome.

Stop at the first mismatch. The expected number of vertices exposed before
stopping is bounded by the expected component size in either complete
first exploration, hence by $e^{2C}$ from the actual fitness-ordered
component estimate. A union bound over the two edge directions gives

$$
\mathbb P(\text{any mismatch})
\leq\left[L_M+2e^{2C}B\right]\Delta.
$$

On the complementary event all physical states, measurement companions,
accepted edges, and component membership agree. Use the same single Haar
matrix for the matching components and the same root jitter, OU innovation,
and position innovation. Frozen-source copying, shared component velocity
update, both force evaluations, velocity cap, and terminal classification
then agree exactly. Full variation is at most twice the coupling failure
probability, proving the first assertion.

The pair $(X,J)$ requires only the root type, its cloning donor and that
donor's sampled measurement, and the root gate. Its donor law is normalized
against the conditional alive law, so the factor $m$ cancels. A root
mismatch contributes at most $L_M\Delta$, an augmented donor mismatch at
most $(2/\kappa_C)L_M\Delta$, and a matched gate disagreement at most
$2L_aL_F\Delta$. Incoming component members do not change a recipient's
frozen copied position. Converting this coupling bound to full variation
gives $L_{\mathrm{pos}}$. Every estimate retains the actual sampled fitness
and the complete component collision transformation.
:::

:::{prf:lemma} An exact kinetic observable whose noise excludes the OU amplitude
:label: lem-chaos-kinetic-memory-observable

For the quadratic-force BAOAB step, let $(X,V)$ be a walker's input after
cloning and its specified recipient jitter. Put $c=h/2$, $k=1-c^2$, and
$s=\sigma_x\sqrt h$. Let $(x^+,v^+)$ be its final physical coordinates
including the smooth cap, before any cemetery replacement. The inverse cap
on $|v|<V_{\max}$ is

$$
C_{V_{\max}}^{-1}(v)=\frac{V_{\max}v}{V_{\max}-|v|}.
$$

The bounded marked-state observable

$$
\varphi_t(x,v,a)=\cos\!\left(
 t\cdot\left[C_{V_{\max}}^{-1}(v)-\frac{k}{c}x\right]\right)
$$

has, for every $t\in\mathbb R^d$, the exact conditional expectation

$$
\boxed{\mathbb E\bigl[\varphi_t(x^+,v^+,a^+)\mid X,V\bigr]
 =\cos\!\left(t\cdot[-(k/c)X-V]\right)
   \exp\!\left[-\tfrac12(ks/c)^2|t|^2\right].}
$$

It is independent of the OU noise amplitude. In particular, increasing
velocity noise alone does not make the complete one-step physical output
independent of its input. This assertion concerns an exact observable of
the declared update and does not assert a failure of multistep attraction.
:::

:::{prf:proof}
Write the BAOAB intermediate variables as

$$
v_1=V-cX,\quad x_1=X+cv_1=kX+cV,\quad
v_2=e^{-\gamma h}v_1+q\xi,\quad
x_2=x_1+cv_2,\quad v_3=v_2-cx_2.
$$

Eliminating $v_2$ gives

$$
v_3=\frac{k}{c}x_2-\frac{k}{c}X-V.
$$

The final operations are $x^+=x_2+s\zeta$ and
$v^+=C_{V_{\max}}(v_3)$, with $\zeta$ a standard Gaussian independent
of the input and OU innovation. Therefore

$$
C_{V_{\max}}^{-1}(v^+)-\frac{k}{c}x^+
=-\frac{k}{c}X-V-\frac{ks}{c}\zeta.
$$

Taking its Gaussian characteristic function and the real part proves the
formula. Terminal alive/dead classification does not change physical
coordinates, and the observable uses both status strata. The inverse cap
is finite on every physical output of the smooth cap; its values outside
that open ball may be assigned arbitrarily. No cancellation of cloning
or collision terms is used: their full outcome is the conditioned pair
$(X,V)$.
:::

:::{prf:theorem} Stationary chaos from actual-map attraction
:label: thm-uniqueness-of-qsd

Suppose the canonical QSD empirical laws are tight with the stationary
moment control above, and every measure in their limiting class is attracted
to the same probability $\mu_*$ under the actual map:

$$
\mathcal F_h^n(\mu)\Rightarrow\mu_*\qquad(n\to\infty).
$$

Then $\mu_*$ is a fixed point and

$$
\Lambda_N\Rightarrow\delta_{\mu_*},\qquad
\nu_N^{(\ell)}\Rightarrow\mu_*^{\otimes\ell}
\quad\text{for each fixed }\ell.
$$

Finite-horizon consistency and vanishing extinction for this kernel are
proved in {prf:ref}`thm-chaos-canonical-one-step` and
{prf:ref}`thm-extinction-rate-vanishes`. The stated global attraction remains the unresolved
hypothesis for the canonical algorithm; no finite-rate substitute is used
to assert it.
:::

:::{prf:proof}
Take a subsequence with $\Lambda_N\Rightarrow\Lambda$. The preceding
invariant-mixture theorem gives $(\mathcal F_h)_\#\Lambda=\Lambda$ and
therefore $(\mathcal F_h^n)_\#\Lambda=\Lambda$ for every integer $n$.
For bounded continuous $H$ on the space of population laws,

$$
\int H(\mathcal F_h^n\mu)\Lambda(d\mu)=\int H(\mu)\Lambda(d\mu).
$$

The assumed attraction and bounded convergence make the left side tend to
$H(\mu_*)$. Thus $\Lambda=\delta_{\mu_*}$. Every subsequential limit has
this value; tightness gives convergence of the whole sequence. Exchangeability
and the empirical-to-chaos lemma give the marginal conclusion.
Finally continuity and the iteration identity imply

$$
\mathcal F_h\mu_*
=\lim_{n\to\infty}\mathcal F_h(\mathcal F_h^n\mu)
=\lim_{n\to\infty}\mathcal F_h^{n+1}\mu=\mu_*.
$$
:::

:::{prf:remark} Why uniqueness of a fixed point is insufficient
:label: rem-chaos-attraction-versus-uniqueness

A nonlinear discrete map may have a unique fixed point and also a periodic
orbit. The uniform measure on a finite periodic orbit is an invariant
probability on the space of population laws. Fixed-point uniqueness alone
does not eliminate that invariant mixture. The attraction theorem excludes
it dynamically; the concentration theorem excludes it by vanishing
empirical variance. This observation identifies a logical requirement, not
an asserted periodic orbit of the canonical gas.
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

:::{prf:remark} What has been established for the actual algorithm
:label: rem-chaos-model-scope

The canonical fixed-step algorithm has an explicit rooted-component
population map, proved normalization and component-size bounds, one-step
empirical consistency, continuity, and finite-horizon propagation of chaos.
The terminal-box configuration also has at least one nonlinear stationary
fixed point. None of these conclusions requires cloning probabilities to
vanish with the timestep.

Stationary QSD chaos additionally requires the concentration or attraction
step stated above. Continuous-time limits require control over a growing
number of the same updates, with the actual collision, cap, revival, and
boundary operations retained. History-dependent donors, adaptive diffusion,
local fitness normalization, and alternative boundary schedules require
analysis of their own specified transition; the canonical theorem does not
silently identify those extensions with its kernel. These distinctions
carry into {doc}`10_kl_hypocoercive`, {doc}`12_qsd_exchangeability_theory`,
and {doc}`16_continuum_discharge`.
:::
