# The Discrete Population Limit and Propagation of Chaos

:::{div} feynman-prose
Stationary chaos needs information about the long-time population dynamics in addition to finite-horizon approximation. [Structural landscape convergence](06a_structural_landscape_convergence.md) makes that extra obligation landscape-dependent: prove attraction where the required estimates hold, or identify the limiting phase distribution when several phases persist. Finite-particle uniqueness alone does not settle this population-level question.
:::

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
For the general canonical terminal-box kernel with its declared continuous
Lipschitz force, {prf:ref}`thm-chaos-general-box-qsd-existence` proves
existence of an exchangeable QSD without asserting uniqueness. Write

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

These one-step theorems concern the connected-component Haar collision
kernel specified in {prf:ref}`def-inelastic-collision-update`. The current
Python `clone_walkers` routine uses sequential donor-star collisions
without Haar rotation. Its separate collision calculation and
priority-decorated one-step bound are
{prf:ref}`thm-slc-ordered-collision-balance` and
{prf:ref}`thm-chaos-ordered-star-quantitative`. The Haar bounds below
retain their original kernel and constants.
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
{prf:ref}`thm-chaos-conditioned-propagation` gives the exact
full-path TV cost and the finite-row bound for the actual
survivor-conditioned law.
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

:::{prf:definition} Priority-decorated population map for ordered donor-star collisions
:label: def-chaos-ordered-star-population-map

For the ordered donor-star collision of
{prf:ref}`thm-slc-ordered-collision-balance`, attach the fixed mathematical
priority $r_i=i/N$ to finite slot $i$ and set
$\nu_N=N^{-1}\sum_i\delta_{(r_i,z_i)}$, where $z_i$ is its complete
physical marked state. The priority is not passed to or updated by the
algorithm. Let $\nu(dr,dz)$ be any probability law on this decorated
state, and let $\mu$ be its physical marginal. The limiting laws of
interest have uniform $[0,1]$ priority marginal; the finite empirical
$\nu_N$ have atomic grid marginals. Apply the measurement and global
fitness construction of {doc}`08_mean_field` to $\mu$, retaining the
priority on every resulting sampled type
$t=(r,z,y_D,F)$. Call this type law $\eta_\nu$. The construction below
also defines $\mathcal F_h^{\rm ord}$ for atomic priority marginals by
choosing the outgoing-donor branch when two distinct sampled vertices
have equal priorities; such ties have probability zero for a uniform
priority marginal. For live $t$, define
$$
\beta_\nu(t,u)=
\frac{a_u w_C(z_t,z_u)\,p(F_t,F_u)}{Z_C(\mu;z_t)},
\qquad
Z_C(\mu;z_t)=\int a_u w_C(z_t,z_u)\,\mu(du),             \tag{C.OS1}
$$
and replace $p(F_t,F_u)$ by one for dead $t$. This is the accepted-edge
density relative to $\eta_\nu(du)$; it uses precisely the companion and
acceptance rules of {prf:ref}`def-mean-field-rooted-collision`.

Draw a root $t$ from $\eta_\nu$. Draw its outgoing accepted donor $U$
with subprobability $\beta_\nu(t,u)\eta_\nu(du)$. Draw an independent
incoming Poisson cloud $\Pi_t$ of types with intensity
$\beta_\nu(u,t)\eta_\nu(du)$. If $U=u$ exists, draw an independent
additional incoming Poisson cloud $\Pi_u^\circ$ of intensity
$\beta_\nu(w,u)\eta_\nu(dw)$, and insert $t$ once in $u$'s star.
Set
$$
m_t=\frac{v_t+\sum_{w\in\Pi_t}v_w}{1+|\Pi_t|},\qquad
m_u=\frac{v_u+v_t+\sum_{w\in\Pi_u^\circ}v_w}
 {2+|\Pi_u^\circ|}.                                      \tag{C.OS2}
$$
The root's collision velocity is
$$
v_t^{\rm ord}=
\begin{cases}
\alpha_{\rm col}v_t+(1-\alpha_{\rm col})m_t,
 &|\Pi_t|>0\text{ and }(U=\bot\text{ or }r_t>r_U),\\
\alpha_{\rm col}v_t+(1-\alpha_{\rm col})m_U,
&U\ne\bot\text{ and }(|\Pi_t|=0\text{ or }r_U\ge r_t),\\
v_t,&U=\bot,\ |\Pi_t|=0.
\end{cases}                                               \tag{C.OS3}
$$
Distinct priorities tie with probability zero under a uniform
priority marginal. Use the same frozen donor position, accepted-row jitter,
BAOAB, cap and terminal classification as in {doc}`08_mean_field`, and
preserve $r_t$. Their root-output law defines
$\mathcal F_h^{\rm ord}(\nu)$. Its physical marginal describes the
population of physical walkers. The original, permutation-equivariant
component-Haar map $\mathcal F_h(\mu)$ remains as separately defined.
This ordered map substitutes the Python donor-star collision into the
declared Volume 2 kinetic and boundary stages. It is not the entire
default Python `EuclideanGas.step` kernel: its optional kinetic terms,
position noise, cap and boundary timing require separate matching.
:::

:::{prf:theorem} Quantitative one-step consistency for the ordered collision mode
:label: thm-chaos-ordered-star-quantitative

Use the full canonical measurement, cloning, kinetic and terminal rules
of {prf:ref}`def-chaos-canonical-regime`, substituting only the ordered
donor-star collision (SCK.O1). Suppose $0\le\alpha_{\rm col}\le1$ and
the entering array has alive fraction at least $m_*>0$. Keep the
algorithmic constants $C,B,A_D,D_D,A_T,L_T,A,M_3(C),N_0$ exactly as
defined in {prf:ref}`lem-chaos-canonical-innovation-replacement` and
{prf:ref}`thm-chaos-canonical-quantitative-bias`; in particular
$A=1+C+D_D$, $N_0=\lceil(8A)^{6/5}\rceil$. For every bounded measurable
test $\varphi(r,z)$, put $b=\|\varphi\|_\infty$ and
$$
K_{\rm ord}(C)=27+30C+6C^2,
\quad A_{\rm ord,\varphi}=2b^2[A_D+K_{\rm ord}(C)+1],
\quad B_{\rm ord}=3M_1(2C)L_T\sqrt{A_T}+4L_T^2A_T
 +128A^2+16M_3(C)+\sqrt{N_0}.                         \tag{C.OS4}
$$
The constants depend on the declared fitness floors, distance weights,
alive fraction and kinetic parameters through the cited expressions,
but none depends on $N$. With $L_N'$ the complete physical marked
output, still carrying fixed priorities, and
$\nu_N=N^{-1}\sum_i\delta_{(i/N,z_i)}$, one has
$$
\operatorname{Var}(L_N'\varphi\mid S)
 \le\frac{A_{\rm ord,\varphi}}N,                         \tag{C.OS5}
$$
$$
\left|\mathbb E[L_N'\varphi\mid S]
 -\mathcal F_h^{\rm ord}(\nu_N)\varphi\right|
 \le\frac{2bB_{\rm ord}}{\sqrt N},
\qquad
\mathbb E\left[|L_N'\varphi-
 \mathcal F_h^{\rm ord}(\nu_N)\varphi|^2\mid S\right]
 \le\frac{A_{\rm ord,\varphi}+4b^2B_{\rm ord}^2}{N}.    \tag{C.OS6}
$$
The fixed cemetery-law convention adds the terminal-extinction correction
from {prf:ref}`thm-chaos-canonical-conditional-variance`, with
$A_{\rm ord,\varphi}$ in place of $A_\varphi$.

If $\nu_N\Rightarrow\nu$ with uniform priority marginal and the same
moment control as the corresponding canonical regime, then
$\mathcal F_h^{\rm ord}(\nu_N)\Rightarrow
\mathcal F_h^{\rm ord}(\nu)$. Consequently the ordered update has a
finite-horizon mean-field limit on priority-decorated laws, obtained by
iterating this single nonlinear map. The physical population law is its
projection. The existing Haar one-step and finite-horizon bounds remain
the unmarked statements of
{prf:ref}`thm-chaos-canonical-quantitative-bias`.
:::

:::{prf:proof}
First fix the complete accepted-edge plan. A row can be written by only
two donor stars: its own incoming star, if nonempty, and the star of its
one accepted outgoing donor. The code processes donor labels in increasing
order, so their priority comparison gives exactly (C.OS3) as the local
limit of (SCK.O1). Its incoming edges have the Poisson intensities of
{prf:ref}`def-mean-field-rooted-collision`; when the root has an outgoing
edge to $u$, that edge is inserted once and excluded from the additional
incoming cloud at $u$. The finite-component path bound
{prf:ref}`lem-mean-field-component-bound` makes this construction finite
almost surely. The last-writer readout uses frozen velocities, and is
therefore determined by this finite marked neighborhood.

For the variance, use the independent innovation blocks of
{prf:ref}`thm-chaos-canonical-conditional-variance`, omitting its Haar
rotation block. Replacing a measurement row's draw can change accepted
edges through the shared normalizers. The exceptional-row and component
exposure proof of {prf:ref}`lem-chaos-canonical-innovation-replacement`
still bounds the squared number of affected rows by $A_D$: with all
unchanged edges fixed, every ordered-star output outside the affected
components uses the same donor stars, priorities and frozen velocities.
For replacement of one donor/gate row $i$, remove that row's edge first.
If $Y_j$ is the number of other recipients at a specified center $j$,
the independent remaining rows each hit $j$ with probability at most
$C/N$. Hence $\mathbb EY_j\le C$ and
$\mathbb EY_j^2\le C+C^2$, also conditional on the old and new targets
of row $i$. Only row $i$ and the two old/new target stars can change.
If $D_i$ counts potentially affected rows, then
$D_i\le1+(2+Y_{j_0})+(2+Y_{j_1})$, with absent targets omitted.
The inequality $(x+y+z)^2\le3(x^2+y^2+z^2)$ gives
$$
\mathbb ED_i^2\le3[1+2\mathbb E(2+Y_j)^2]
\le27+30C+6C^2=K_{\rm ord}(C).
$$
Changing one row's remaining jitter and kinetic innovations affects one
output. {prf:ref}`lem-chaos-innovation-variance` then yields (C.OS5),
including the random normalizers and all correlated collision outputs.

For the bias, the first step of
{prf:ref}`thm-chaos-canonical-quantitative-bias` couples the actual
sampled global normalizers to deterministic population normalizers. Its
exceptional-edge bound uses only accepted-edge patterns and affected
components, so it is unchanged by the star readout and gives the first
two terms of $B_{\rm ord}$. In its finite rooted exploration, attach
the fixed priority to each discovered vertex. The same self-exclusion,
conditional avoidance, rejected-target mark and categorical-to-Poisson
estimates hold, since those are estimates of the edge law. On matching
explorations, (SCK.O1) and (C.OS3) agree exactly: they use the same
priorities, frozen velocities, donor positions, jitter and row-local
kinetic noise. The stated $64A^2K^3/N$ mismatch bound for an exploration
of at most $K$ vertices can be enlarged to $128A^2K^3/N$ to include
duplicate sampled finite labels: at most $2K$ labels are exposed, and
their collision probability is at most $8AK^2/N$ because every queried
label has conditional mass at most $2A/N$; $A\ge1$ absorbs it.
The unchanged two-component tail bound is $2M_3(C)/K^3$. Choosing
$K=\lfloor N^{1/6}\rfloor$ and treating $N<N_0$ by the elementary
bound gives the last three terms of $B_{\rm ord}$. Thus the conditional
bias in (C.OS6) follows; its mean-square estimate is variance plus
squared bias. The cemetery adjustment is the same bounded rare-event
calculation because collision velocities remain capped.

For continuity, first truncate the rooted exploration to at most $K$
vertices. The measurement and companion kernels and their positive
denominators have the continuity proved in
{prf:ref}`lem-chaos-canonical-map-continuity`. The only new comparison
in the finite readout is $r_u>r_t$. Its equality set has zero probability
because the priority marginal is uniform. The finite-tree law therefore
converges under weak convergence of the joint marked inputs. Remove the
truncation by the same component moment bound. The kinetic, cap and
terminal stages then have the existing continuity proof. Combining this
with (C.OS6) and inducting over any fixed number of steps proves the
finite-horizon assertion. The priority is preserved during that
induction; the physical marginal alone need not determine the next
ordered collision law.
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

:::{prf:theorem} Quantitative noncommutation for the unconditioned terminal-box gas
:label: thm-chaos-unconditioned-extinction-obstruction

Use the actual terminal-box kernel of
{prf:ref}`def-mean-field-marked-state`, with its declared positive final
position noise, mandatory revival while a donor exists, and absorbing
all-dead state. No restart or conditioning is applied. Write
$D=\prod_{k=1}^d[\ell_k,u_k]$, $w_k=u_k-\ell_k>0$,
$s=\sigma_x\sqrt h>0$, and let $\Phi$ be the standard normal distribution
function. Define the explicit numbers

$$
 p_D=\prod_{k=1}^d\left[2\Phi\!\left(\frac{w_k}{2s}\right)-1\right],
 \qquad q_D=1-p_D\in(0,1),\qquad b_N=q_D^N.
$$

Let $\tau_N=\inf\{n\ge0:M_n=0\}$ for an entering population with
$M_0>0$, and let $A_{N,n}=M_n/N$, set to zero after extinction. For every
integer $n\ge0$ and every initial law supported on these entering states,

$$
 \Pr(\tau_N>n)\le(1-b_N)^n\le e^{-nb_N},\qquad
 \mathbb E\tau_N\le b_N^{-1},\qquad
 \mathbb EA_{N,n}\le(1-b_N)^n.
 \tag{9.E1}
$$

In particular extinction occurs almost surely at every finite $N$.
Every invariant probability of the unconditioned, absorbing full kernel
is supported on all-dead states. For any existing QSD of the killed
kernel, its eigenvalue instead satisfies the explicit two-sided estimate

$$
 q_D^N\le1-\alpha_N
 \le\min\{1,e^{-pN/8}+e^{-a_0N/16}\},\qquad p=p_Jp_G,
 \tag{9.E1a}
$$

with $a_0,p_J,p_G$ evaluated below. The QSD assertion uses its exact
eigenmeasure equation; it does not assert QSD existence for a new class
of forces.
For the same fixed-step population map, write
$\mu_n=\mathcal F_h^n(\mu_0)$, $m_n=\mu_n(a=1)$, with $m_0>0$.
Let $a_0>0$ be the explicit Gaussian landing bound below, obtained by
the calculation of {prf:ref}`cor-mean-field-positive-alive-mass` on a
ball inside the box. Then $m_n\ge a_0$ for every $n\ge1$. Set
$x_D=((\ell_k+u_k)/2)_{k=1}^d$, $r_0=\min_k w_k/4$ and
$R_D=(\sum_k\max\{|\ell_k|,|u_k|\}^2)^{1/2}$. The constants are

$$
\begin{gathered}
 c_h=e^{-\gamma h},\quad
 s_h^2=\begin{cases}(1-e^{-2\gamma h})/(2\gamma),&\gamma>0,\\h,&\gamma=0,\end{cases}
 \quad
 W=(1+2\alpha)V,\quad F_J=L_U(R_D+J)+B_U,\\
 L=R_D+J+\frac h2(1+c_h)(W+\tfrac h2F_J)
                   +\frac h2s_h\|B\|G,\\
 p_J=\Pr(|\sigma_J\xi|\le J),\quad p_G=\Pr(|\xi|\le G),\\
 a_0=p_Jp_G\,\frac{\pi^{d/2}r_0^d}{\Gamma(1+d/2)}
       (2\pi s^2)^{-d/2}
       \exp\!\left[-\frac{(L+|x_D|+r_0)^2}{2s^2}\right].
\end{gathered}
$$

Here $J,G>0$ are arbitrary evaluation radii and $\xi$ is a standard
$d$-dimensional Gaussian. For $\sigma_J>0$ these probabilities are,
respectively, $\gamma(d/2,J^2/(2\sigma_J^2))/\Gamma(d/2)$ and
$\gamma(d/2,G^2/2)/\Gamma(d/2)$, where $\gamma(\cdot,\cdot)$ is the
lower incomplete gamma function; $p_J=1$ when $\sigma_J=0$.
The ball $B(x_D,r_0)$ lies strictly inside $D$. Here $\alpha$ is the
configured collision restitution, $V$ the completed velocity cap,
$\gamma$ the friction, $B$ the OU noise factor, $\sigma_J$ the cloning
jitter, and $L_U,B_U$ the declared force-growth constants
$|\nabla U(x)|\le L_U|x|+B_U$. The bound $b_N$ is
independent of reward, cloning, collision and force parameters because it
holds uniformly over every possible preterminal position center.

The resulting full-law error has the lower bound

$$
 \mathbb E|A_{N,n}-m_n|\ge a_0-(1-b_N)^n\quad(n\ge1),\qquad
 \boxed{\sup_{n\ge1}\mathbb E|A_{N,n}-m_n|\ge a_0.}
 \tag{9.E2}
$$

In particular, the error is at least $a_0/2$ whenever

$$
 n\ge n_N^{\mathrm{sep}}
 :=\left\lceil\frac{\log(2/a_0)}{-\log(1-q_D^N)}\right\rceil.
 \tag{9.E3}
$$

The simpler sufficient bound is
$n\ge\lceil q_D^{-N}\log(2/a_0)\rceil$.
Physical time is $hn$; the number of row updates through this horizon is
$Nn$. Likewise extinction has probability at least $1-\delta$ by
$n=\lceil q_D^{-N}\log(1/\delta)\rceil$, for $0<\delta<1$, and the
expected physical extinction time is at most $h q_D^{-N}$.

For initial arrays covered by the existing finite-horizon mean-field
theorem, the two orders of limits satisfy

$$
 \lim_{N\to\infty}\lim_{n\to\infty}\mathbb EA_{N,n}=0,
 \qquad
 \liminf_{n\to\infty}\lim_{N\to\infty}\mathbb EA_{N,n}
 =\liminf_{n\to\infty}m_n\ge a_0>0.
 \tag{9.E4}
$$

No existence of a long-time limit of $\mu_n$ is required for this strict
separation. Under $\|\nu-\mu\|_{\mathrm{TV}}=\sup_A|\nu(A)-\mu(A)|$,
both $\mathbb E\|L_N(S_n)-\mu_n\|_{\mathrm{TV}}$ and
$\|\mathbb E L_N(S_n)-\mu_n\|_{\mathrm{TV}}$ are at least the respective
alive-mass discrepancy. The same lower bound holds for a bounded-Lipschitz
metric whose state cost dominates $|a-\widetilde a|$ and in which the
test $z\mapsto a$ is admitted. Thus the obstruction is present even for
this single bounded population observable, independently of the atomic
nature of $L_N$.
:::

:::{prf:proof}
**1. A bound uniform over every preceding stage.** For one interval of
width $w$, translate its center to zero and write
$f(t)=\Pr(t+s\xi\in[-w/2,w/2])$.
It is even and

$$
 f'(t)=s^{-1}\left[\phi((w/2+t)/s)-\phi((w/2-t)/s)\right]\le0
 \quad(t\ge0),
$$

where $\phi$ is the standard normal density: the first argument has at
least the absolute value of the second. Hence
$\sup_t f(t)=2\Phi(w/(2s))-1$. The independent coordinates of the final
position innovation give
$\sup_x\Pr(x+s\xi\in D)=p_D$.

Condition now on the entire actual update before its final position
innovations. In particular all companions, fitnesses, accepted edges,
component rotations, cloning jitters and OU innovations are retained in
this conditioning. The preterminal centers $X_{2,i}$ may be arbitrarily
dependent. The final position innovations are still independent, so

$$
 \Pr(M_{n+1}=0\mid X_{2,1},\ldots,X_{2,N},\text{preceding innovations})
 =\prod_{i=1}^N\Pr(X_{2,i}+s\xi_i\notin D)\ge q_D^N.
$$

This holds on every nonextinct entering state. The second force kick and
the cap do not change the terminal position. Integrating over the
preceding innovations proves the same extinction lower bound for the
full kernel, without independence of the unconditional output rows.

**2. Iterate the actual absorption event.** Conditional on
$\{\tau_N>j\}$, the next survival probability is at most $1-b_N$.
Induction gives (9.E1); summing
$\mathbb E\tau_N=\sum_{n\ge0}\Pr(\tau_N>n)$ gives the mean bound.
Also $0\le A_{N,n}\le\mathbf1_{\{\tau_N>n\}}$ because the algorithm
does not restart after all donors have died. The exponential relaxation
uses $1-b_N\le e^{-b_N}$.
For an invariant law $\Pi_N$, the one-step survival estimate gives
$\Pi_N(M>0)\le(1-b_N)\Pi_N(M>0)$; thus $\Pi_N(M>0)=0$.
Integrating the one-step extinction lower bound against any QSD gives
$1-\alpha_N\ge b_N$. The two-stage Gaussian landing and Chernoff
calculation of {prf:ref}`cor-mean-field-positive-alive-mass`, using the
same translated core ball, gives its upper bound in (9.E1a).

**3. Compare with the unchanged population map.** The positive-alive-mass
corollary's proof applies with the core ball centered at $x_D$.
Copying or persistence places every source in $D$, and the collision
speed is at most $W$. On the independent jitter and OU events of
probabilities $p_J,p_G$, the preterminal center has norm at most $L$.
Every point in $B(x_D,r_0)$ is then at distance at most
$L+|x_D|+r_0$ from that center. Integrating the final Gaussian density
over this ball gives exactly $a_0$, with no population-size factor.
This proves $m_1\ge a_0$ and inductively $m_n\ge a_0$.
Since $m_n$ is deterministic,

$$
 \mathbb E|A_{N,n}-m_n|\ge m_n-\mathbb EA_{N,n}
                       \ge a_0-(1-b_N)^n.
$$

Taking a supremum proves (9.E2). Solving
$(1-b_N)^n\le a_0/2$ proves (9.E3); the exponential upper bound proves
the simpler horizons and their physical-time conversion. The same test
$\{a=1\}$ proves the stated TV and bounded-observable lower bounds.

**4. Take the two orders separately.** At each fixed $N$, (9.E1) gives
$\mathbb EA_{N,n}\to0$. At each fixed $n$, the proved marked
finite-horizon limit gives $A_{N,n}\to m_n$ in probability; boundedness
then gives convergence of its expectation. These are the two claims
in (9.E4). None of these steps uses a stationary-attraction hypothesis.
:::

:::{prf:remark} Scope of the extinction obstruction
:label: rem-chaos-extinction-obstruction-scope

The preceding theorem applies to the unconditioned absorbing configuration
already defined by the algorithm. It does not concern a survival-conditioned
QSD, and it does not replace the separately declared conservative
$D=\mathbb R^d$ problem: there $q_D=0$ and (9.E1)--(9.E3) give no positive
extinction rate. In particular it neither proves nor disproves conservative
nonlinear phase attraction. For the terminal-box configuration, however,
the requested vanishing uniform-time full-law error is false even though
its fixed-horizon mean-field law is valid. Conditioning on survival would
change the mathematical object being compared and must be stated explicitly.
:::

(sec-chaos-survival-conditioned)=
### Survival-conditioned population laws

:::{div} feynman-prose
Imagine collecting runs of the unchanged gas and looking at those still alive
at time $n$. Their law is $\eta_0Q_N^n/(\eta_0Q_N^n\mathbf 1)$: evolve first,
then normalize the surviving mass. Eventual extinction of every finite swarm
does not obstruct studying this conditional law. Nor is this the procedure
that rejects every fatal update and tries again; that procedure changes the
transition rule.

The alive-fraction estimate makes another distinction useful. At each current
time, its uniform conditional bound gives probability at most $\delta_N$ of
an insufficient alive fraction among surviving swarms. There is no factor
$n$ in that statement. Such a factor enters when we instead demand that no
low-fraction episode occurred anywhere in the history. Those are different
events, and the present estimate concerns the first.

Keep the whole marked population while evolving: dead rows still belong to
the state on which revival acts. Extract and normalize the alive distribution
afterward. These operations specify the convergence problem precisely; they
do not by themselves prove attraction to a stationary conditional law.
:::

:::{prf:definition} Survival conditioning and the quantitative alive floor
:label: def-chaos-survival-filter

Retain the actual terminal-box kernel and parameters of
{prf:ref}`def-mean-field-marked-state`. Let $P_N$ retain the complete
physical output, including all-dead outputs, and let $Q_N$ be its
restriction to $E_N=\{M>0\}$. Use the explicit $a_0,p=p_Jp_G$ of
{prf:ref}`thm-chaos-unconditioned-extinction-obstruction`, and put

$$
 m_*=a_0/4,\qquad G_N=\{M/N\ge m_*\},\qquad
 \delta_N=\min\{1,e^{-pN/8}+e^{-a_0N/16}\}.
$$

The proved update estimates are
$P_N(S,G_N^c)\le\delta_N$ and
$q_N(S):=Q_N1(S)\ge a_0$ for every nonextinct input $S$.
The second follows also from
$\mathbb E_S[M'/N]\ge a_0$ and $M'/N\le\mathbf1_{E_N}$.
For an initial law $\eta_0$ on $E_N$, define the actual surviving law

$$
 \eta_n=\frac{\eta_0Q_N^n}{\eta_0Q_N^n1},\qquad
 \eta_{n+1}=\frac{\eta_nQ_N}{\eta_nq_N},\qquad
 e_n^{\dagger}=1-\eta_nq_N.
$$

These denominators are positive for every finite $n$, since they are
at least $a_0^n$. For reference, the entirely explicit threshold

$$
 N_{\mathrm{surv}}=
 \left\lceil\max\{8/p,16/a_0\}\log4\right\rceil
$$

ensures $\delta_N\le1/2$ for $N\ge N_{\mathrm{surv}}$.
All constants except the displayed $N$ dependence are independent of
population size. No conditioning step changes the simulated algorithm.
:::

:::{prf:corollary} Exact extinction hazard and exponential recovery window
:label: cor-chaos-exact-hazard-recovery-window

Retain the actual canonical terminal-box update and the constants of
{prf:ref}`def-chaos-survival-filter`. Put $c_0=h/2$,
$a=e^{-\gamma h}$, $s=\sigma_x\sqrt h$ and

$$
q^2=\sigma_v^2
\begin{cases}(1-e^{-2\gamma h})/(2\gamma),&\gamma>0,\\h,&\gamma=0,
\end{cases}
\qquad \tau^2=c_0^2q^2+s^2>0.
$$

Condition on the complete post-cloning positions $X_i$ and collision
velocities $V_i^C$, before the independent row kinetic Gaussians.
The actual position update has the exact form

$$
x_i^+=\mu_i+\tau Z_i,\qquad
\mu_i=X_i+c_0(1+a)(V_i^C+c_0F(X_i)),
\qquad Z_i\overset{\rm iid}{\sim}N(0,I_d).
                                                               \tag{9.H1}
$$

For the box $D=\prod_{k=1}^d[\ell_k,u_k]$ define

$$
P_{D,\tau}(z)=\prod_{k=1}^d
\left[\Phi\!\left(\frac{u_k-z_k}{\tau}\right)
     -\Phi\!\left(\frac{\ell_k-z_k}{\tau}\right)\right],
\qquad
q_\tau=1-\prod_{k=1}^d
 \left[2\Phi\!\left(\frac{u_k-\ell_k}{2\tau}\right)-1\right].
                                                               \tag{9.H2}
$$

For every nonextinct entering state $S$, the complete-kernel
one-step extinction hazard is the specified finite-plan and Gaussian
expectation

$$
\boxed{\quad
h_N(S):=P_N(S,M^+=0)
=\mathbb E_S^{\rm prep}\prod_{i=1}^N
 [1-P_{D,\tau}(\mu_i)],\qquad
q_\tau^N\le h_N(S)\le\delta_N.
\quad}                                                         \tag{9.H3}
$$

Here $\mathbb E_S^{\rm prep}$ uses the actual measurement,
acceptance, mandatory revival, jitter and component-Haar laws; no
independence of the prepared centers $\mu_i$ is asserted. The upper
bound uses the already proved full-update alive-fraction estimate,
not a separate survival assumption. For $N\ge N_{\rm surv}$,
the absorption time consequently satisfies, at every integer $n\ge0$,

$$
\boxed{\quad
(1-\delta_N)^n\le\Pr_S(\tau_N>n)
 \le(1-q_\tau^N)^n,
\qquad
\delta_N^{-1}\le\mathbb E_S\tau_N\le q_\tau^{-N}.
\quad}                                                         \tag{9.H4}
$$

The chance of *any* alive-fraction failure through step $T$ is at
most $T\delta_N$, while the surviving law at each observation time
obeys the sharper no-$T$ estimate (9.S1). Thus the guaranteed
survival-only horizon at failure tolerance $\varepsilon$ is
$T\le\varepsilon/\delta_N$ steps, or $hT$ physical time; a
mean-field approximation on that entire horizon additionally uses
its own finite-horizon error bound.

The exact formula also resolves the role of landscape geometry.
For $r>0$, let $D_r=\{z\in D:\operatorname{dist}(z,\partial D)
\ge r\}$ and set
$\epsilon_r=\min\{1,2d\Phi(-r/\tau)\}$. For $0<\rho\le1$ let
$\mathcal H_{\rho,r}$ be the event, under the preparation law, that
at least $\lceil\rho N\rceil$ of the centers $\mu_i$ lie in $D_r$.
Then the same actual hazard satisfies

$$
h_N(S)\le\min\left\{\delta_N,
\Pr_S^{\rm prep}(\mathcal H_{\rho,r}^c)
 +\epsilon_r^{\lceil\rho N\rceil}\right\}.
                                                               \tag{9.H5}
$$

The safe-center probability in (9.H5) is an explicit finite-plan
Gaussian/Haar integral through (9.H1); force, reward, clone jitter,
restitution, friction and the entering alive geometry all remain in
that integral. The velocity cap bounds the entering stored velocities
but does not cap the OU position displacement $c_0q\xi_v$ or the
final position noise $s\xi_x$. Hence a one-step exit remains possible;
(9.H3)--(9.H5) quantify how unlikely simultaneous exit is under the
actual population geometry, without positing that ordinary exits must
accumulate over several steps.

*Proof.* Substitute the B1--A1--O--A2 equations of
{prf:ref}`def-eg-baoab-canonical` into $x^+$: the deterministic
center is $\mu_i$, and the independent Gaussian sum
$c_0q\xi_{v,i}+s\xi_{x,i}$ has variance $\tau^2I_d$. The B2
force kick and velocity cap occur after the position used for
classification and cannot change it. Conditional on preparation,
these Gaussian pairs are independent across rows, so their product
gives the equality in (9.H3). The maximum probability that a
$N(z,\tau^2I_d)$ point lies in an interval of width $w_k$ is
$2\Phi(w_k/(2\tau))-1$, attained when $z_k$ is its midpoint;
multiply over coordinates to get the lower bound $q_\tau^N$.
Extinction is contained in $G_N^c$, so the proved
$P_N(S,G_N^c)\le\delta_N$ gives the upper bound. Iterating both
conditional hazard bounds until absorption proves (9.H4), and
summing their survival tails gives the expectation bounds. The
first-hit union bound gives $T\delta_N$. If $\mu_i\in D_r$,
each coordinate's exit probability is at most
$2\Phi(-r/\tau)$; the union bound over coordinates gives
$1-P_{D,\tau}(\mu_i)\le\epsilon_r$. On
$\mathcal H_{\rho,r}$ at least $\lceil\rho N\rceil$ factors in
the product are at most $\epsilon_r$; the others are at most one.
Average over preparation and split by this event to obtain (9.H5).
$\square$
:::

:::{prf:proposition} Safe-center survival with arbitrary position-displacing noise
:label: prop-chaos-safe-center-noise

Use the same complete terminal-box update as in
{prf:ref}`cor-chaos-exact-hazard-recovery-window`, but allow
$\sigma_x\ge0$ and $\sigma_v\ge0$. Set
$\tau^2=(h q/2)^2+h\sigma_x^2\ge0$, with $q$ as in (9.H1).
The post-cloning/collision preparation law and centers $\mu_i$ are
exactly those in (9.H1). At $\tau=0$ define
$P_{D,0}(z)=\mathbf1_D(z)$; at $\tau>0$ use (9.H2). Then, for every
nonextinct entering state $S$, including the zero-noise case,

$$
 P_N(S,M^+=0)
 =\mathbb E_S^{\rm prep}\prod_{i=1}^N
       [1-P_{D,\tau}(\mu_i)].                         \tag{9.H6}
$$

When $\tau>0$, the box-width argument in (9.H2) also gives
$P_N(S,M^+=0)\ge q_\tau^N>0$ for every nonextinct $S$, even
when $\sigma_x=0$ and the position displacement comes entirely
from the OU innovation. At $\tau=0$ there is no such positive
state-uniform lower bound.

Fix $r>0$, $0<\rho\le1$, and integers
$g=\lceil\rho N\rceil$ and $1\le k\le g$. Let
$D_r=\{z\in D:\operatorname{dist}(z,\partial D)\ge r\}$,
and define the *complete-preparation coverage deficit*

$$
 \eta_{N,g,r}(S)=
 \mathbb E_S^{\rm prep}
 \mathbf1\!\left\{\sum_{i=1}^N\mathbf1_{D_r}(\mu_i)<g\right\}.
                                                               \tag{9.H7}
$$

This is an integral against the specified finite measurement,
acceptance, donor, clone-jitter and component-rotation kernel; in
particular it retains dependence among the $\mu_i$. Define

$$
 \epsilon_r(\tau)=
 \begin{cases}
  \min\{1,2d\Phi(-r/\tau)\},&\tau>0,\\
  0,&\tau=0,
 \end{cases}
 \quad p_r(\tau)=1-\epsilon_r(\tau),
 \quad
 B_{g,k}(p)=\sum_{j=0}^{k-1}{g\choose j}p^j(1-p)^{g-j}.
                                                               \tag{9.H8}
$$

With $0^0=1$ in the finite binomial sum, the actual full-update
alive-fraction and extinction bounds are

$$
 \boxed{\quad
 P_N(S,M^+<k)\le
 \eta_{N,g,r}(S)+B_{g,k}(p_r(\tau)),\qquad
 P_N(S,M^+=0)\le
 \eta_{N,g,r}(S)+\epsilon_r(\tau)^g.
 \quad}                                                         \tag{9.H9}
$$

The right sides may be truncated at one. In particular, if
$\tau=0$ and the preparation guarantees $g$ centers in $D_r$
($\eta_{N,g,r}(S)=0$), then $M^+\ge g$ almost surely and the
extinction hazard is zero. This conclusion permits clone jitter:
its effect is already included in the centers and their coverage
deficit. Setting only $\sigma_x=0$ does **not** make $\tau=0$ when
$q>0$; the OU innovation is applied before the second position
drift, and the subsequent velocity cap cannot undo that displacement.

The estimate also has a pathwise form without independence between
updates. Let $\mathcal C\subseteq\{S:M(S)\ge k\}$ be a declared set of entering
states, let $\theta=\inf\{j\ge0:S_j\notin\mathcal C\}$, and let
$\zeta_k=\inf\{j\ge1:M_j<k\}$. Write
$b(S)=\min\{1,\eta_{N,g,r}(S)+B_{g,k}(p_r(\tau))\}$.
For $S_0\in\mathcal C$ and every integer $T\ge1$,

$$
 \Pr_{S_0}(\zeta_k\le T,\ \zeta_k\le\theta)
 \le\sum_{j=0}^{T-1}
 \mathbb E_{S_0}\!left[
 \mathbf1_{\{j<\zeta_k,\,j<\theta\}}b(S_j)\right].
                                                               \tag{9.H10}
$$

If $b(S)\le\bar b<1$ on $\mathcal C$, this is at most
$1-(1-\bar b)^T\le T\bar b$; explicitly,
$\Pr_{S_0}(\zeta_k>T\text{ or }\theta<\zeta_k)
\ge(1-\bar b)^T$. If $\mathcal C$ is invariant up to the
first alive-fraction failure, then
$\mathbb E_{S_0}\zeta_k\ge1/\bar b$ for $\bar b>0$, and
$\zeta_k=\infty$ almost surely when $\bar b=0$.
In this invariant case, for the full path law $\mathsf P_T$ on
$(S_0,\ldots,S_T)$ and its actual conditioning
$\mathsf P_T^{\rm good}=\mathsf P_T(\cdot\mid\zeta_k>T)$,

$$
 \|\mathsf P_T-\mathsf P_T^{\rm good}\|_{\rm TV}
 =\Pr_{S_0}(\zeta_k\le T)
 \le1-(1-\bar b)^T.                                 \tag{9.H11}
$$

The equality uses the convention
$\|P-Q\|_{\rm TV}=\sup_A|P(A)-Q(A)|$.
It is a full-path total-variation estimate for removing rare
low-alive-fraction histories; it is not an attraction estimate
between two different initial populations.
Thus a certified structural upper bound on (9.H7) translates
directly into iteration, physical-time $hT$, and $NT$ row-update
budgets. Neither (9.H9) nor (9.H10) requires a Markov model of the
basin labels.

*Proof.* The algebra giving (9.H1) does not use strict positivity of
either noise parameter. When $\tau=0$, its Gaussian term is the zero
vector almost surely, so conditional survival is
$\mathbf1_D(\mu_i)$; otherwise it is $P_{D,\tau}(\mu_i)$.
The row kinetic innovations are independent conditional on the full
preparation, giving (9.H6), including at $\tau=0$.

On the event complementary to (9.H7), choose measurably the first
$g$ row indices whose centers lie in $D_r$. For each chosen center,
the Gaussian coordinate tail and a union bound give conditional
exit probability at most $\epsilon_r(\tau)$, including zero at
$\tau=0$. The chosen row survival indicators are conditionally
independent. Couple each one to an independent Bernoulli variable
of success probability $p_r(\tau)$ using its independent uniform
quantile; the row indicator dominates that variable. If fewer
than $k$ of all $N$ rows survive, fewer than $k$ chosen rows survive.
The binomial lower tail therefore bounds its conditional
probability. On the coverage-failure event use the bound one and
average over preparation. For $k=1$ the binomial lower tail is
$\epsilon_r(\tau)^g$, proving both bounds in (9.H9).

For (9.H10), partition the first hit of $\{M<k\}$ by its step.
On $\{j<\zeta_k,j<\theta\}$ the entering state lies in
$\mathcal C$ and is nonextinct. The Markov property and (9.H9)
bound the conditional chance of the next-step hit by $b(S_j)$;
summation proves (9.H10). If $b\le\bar b$ on $\mathcal C$,
conditional survival at each eligible step is at least
$1-\bar b$, yielding the geometric bound. The expectation claim
follows by summing these survival probabilities when exit from
$\mathcal C$ cannot occur before the first hit. Finally, for any
probability $P$ and event $A$ of positive probability,
$\|P-P(\cdot\mid A)\|_{\rm TV}=P(A^c)$: the upper bound follows
by writing $P=P(A)P(\cdot\mid A)+P(A^c)P(\cdot\mid A^c)$,
and $A^c$ attains it. Apply this identity to the full path event
$A=\{\zeta_k>T\}$ to obtain (9.H11). $\square$
:::

:::{prf:theorem} Uniform alive-fraction control under survival
:label: thm-chaos-survival-uniform-floor

For every $n\ge1$ and every $N$ in the preceding definition,

$$
 \boxed{\eta_n(G_N^c)\le\delta_N.}
 \tag{9.S1}
$$

Every QSD $\nu_NQ_N=\alpha_N\nu_N$ obeys the same bound
$\nu_N(G_N^c)\le\delta_N$. More precisely, writing
$b_N=q_D^N$ as in the extinction theorem, both bounds can be replaced by
$\overline\delta_N=(\delta_N-b_N)/(1-b_N)\le\delta_N$.
The exact one-step normalization satisfies

$$
 \eta_{n+1}
 =\frac{q_N\eta_n}{\eta_nq_N}\,\overline P_N,
 \qquad \overline P_N(S,\cdot)=Q_N(S,\cdot)/q_N(S),
$$
$$
 \|\eta_{n+1}-\eta_nP_N\|_{\mathrm{TV}}
 =e_n^{\dagger}\le\delta_N,
 \tag{9.S2}
$$

where $\|\cdot\|_{\mathrm{TV}}=\sup_A|\cdot(A)|$ on the full marked
swarm space. In general $\eta_n\ne\eta_0\overline P_N^n$.
Thus (9.S1) is a bound at every current observation time under survival
through that time; it has no factor $n$ and no inverse cumulative-survival
factor. The statement does not exclude a low-fraction event earlier in
the history.
:::

:::{prf:proof}
Let $e=\eta_nP_N(E_N^c)=e_n^{\dagger}$. Extinction is a subset of
$G_N^c$, so

$$
 \eta_{n+1}(G_N^c)
 =\frac{\eta_nP_N(G_N^c\cap E_N)}{1-e}
 =\frac{\eta_nP_N(G_N^c)-e}{1-e}
 \le\frac{\delta_N-e}{1-e}\le\delta_N.
$$

The last inequality is equivalent to $e(1-\delta_N)\ge0$.
The argument applies to every entering law on $E_N$, hence to each
$\eta_n$ and to a QSD. Since $e\ge b_N$ and
$(\delta_N-e)/(1-e)$ is decreasing in $e$, it also proves the sharper
bound. In particular $\delta_N\ge b_N$ follows from the two bounds on
the same actual update; no inconsistent probability range is introduced.

Disintegration by the last input gives the displayed tilted-input
formula. Finally $\eta_nP_N$ is a mixture of its conditional laws on
the disjoint events $E_N$ and $E_N^c$, with respective weights $1-e,e$.
Its first conditional law is $\eta_{n+1}$. The total variation distance
from this mixture to $\eta_{n+1}$ is exactly $e$, proving (9.S2).
:::

:::{prf:theorem} Uniform conditional moments and QSD existence for the box kernel
:label: thm-chaos-general-box-qsd-existence

Use the existing continuous, globally Lipschitz canonical force and its
declared growth constants $|F(x)|\le B_U+L_U|x|$. This theorem does not
require a quadratic objective, phase-space invertibility, or an additional
timestep restriction. With the parameters of
{prf:ref}`def-baoab-update-rule` and the preceding landing bound, set

$$
 c=h/2,\quad b=c(1+c_h),\quad \eta=c^2(1+c_h),\quad
 A_x=1+\eta L_U,\quad W=(1+2\alpha)V,
$$
$$
 \tau_x^2=c^2s_h^2\|B\|^2+\sigma_x^2h,\qquad
 g_{d,r}=\left[2^{r/2}\frac{\Gamma((d+r)/2)}{\Gamma(d/2)}\right]^{1/r},
$$
$$
 K_r=\left[A_x(R_D+\sigma_J g_{d,r})
                    +bW+\eta B_U+\tau_xg_{d,r}\right]^r
 \quad(r\ge1),\qquad H_x=(2\pi\sigma_x^2h)^{-d/2}.
$$

For every $n\ge1$, every admitted initial law on $E_N$, and every row $i$,

$$
 \mathbb E_{\eta_n}\frac1N\sum_i|x_i|^r\le K_r/a_0,
 \qquad (\eta_n)_{x_i}\le(H_x/a_0)\,dx.
 \tag{9.S3}
$$

The actual kernel has at least one exchangeable QSD for every finite $N$.
Every QSD obeys (9.S1), (9.S3), and
$a_0\le\alpha_N\le1-q_D^N$, as well as (9.E1a).
These claims concern existence and explicit moment/coverage bounds;
they do not assert uniqueness or attraction of every initial law.
:::

:::{prf:proof}
**1. Evaluate the actual position update.** Before jitter every source
position $Y$ lies in $D$: a live slot either persists or copies an
eligible donor, and a dead slot is revived from one. The prepared
position is $X_0=Y+A\sigma_J\xi^J$ for $A\in\{0,1\}$.
The exact collision formula gives $|V_0|\le W$. Combining B1, A1,
O and A2, and then the final position diffusion, gives

$$
 X'=X_0+bV_0+\eta F(X_0)+cs_hB\xi^O
                         +\sigma_x\sqrt h\,\xi^x.
$$

The last two independent Gaussian terms have covariance bounded by
$\tau_x^2I_d$. Their $L^r$ norm is at most $\tau_xg_{d,r}$: write their
law as $TZ$ with $\|T\|\le\tau_x$ and a standard Gaussian $Z$.
Using the force-growth bound and Minkowski's inequality proves
$\mathbb E_S|X_i'|^r\le K_r$ for every entering state and row.
This argument does not require independent collision outputs.
Conditional on the preceding innovations, the final position density
is a translated Gaussian bounded by $H_x$. Mixing preserves this bound.

**2. Normalize only the current update.** For any entering probability
$\eta$ on $E_N$, its next survival probability is at least $a_0$.
Restricting a nonnegative moment or a positional event to survival only
reduces its unnormalized expectation. Division by $\eta q_N\ge a_0$
therefore proves both bounds in (9.S3), independently of the history.
Applying this at a QSD proves its same bounds.

**3. Check a compact, convex class for the normalized map.** Work on
the full physical marked space with the closed capped velocity ball;
the actual formulas extend continuously to its boundary. Let
$\mathcal K_N$ be the probability laws supported on nonextinct,
terminally consistent states, with capped velocities, averaged second
position moment at most $K_2/a_0$, and each positional marginal
dominated by $(H_x/a_0)dx$. It is nonempty by Step 2, applied to any
nonextinct input. It is convex. Its moment bound implies

$$
 \Pr_{\lambda}(\max_i|x_i|>R)
 \le NK_2/(a_0R^2)\quad(\lambda\in\mathcal K_N),
$$

so it is tight at each fixed $N$. The moment constraint is weakly
closed by lower semicontinuity. Density domination is weakly closed
by testing nonnegative continuous compactly supported functions.
In particular every limiting positional marginal gives zero mass to
$\partial D$. Terminal mark consistency is then retained in weak
limits, since its only possible discontinuities are at those null
boundaries. Capped velocities and the nonempty alive masks are closed
conditions. Thus $\mathcal K_N$ is weakly compact.

**4. Verify continuity and take a fixed point.** The actual Feller
proof {prf:ref}`thm-euclidean-feller` applies: within each fixed input
alive mask there are finitely many companion/gate patterns with
continuous probabilities. The force, collision, jitter, cap and kinetic
updates are continuous for fixed innovations. Final position noise
makes every terminal boundary a null event. For a bounded continuous
test on output swarm states this gives continuity after survival
restriction as well, since nonextinction is a clopen event of the
discrete output marks. Dominated convergence proves that both $Q_Nf$
and $q_N$ are bounded continuous on the admitted input class. This
also gives continuity under weak limits in $\mathcal K_N$; any spatial
boundary exceptional set has zero mass by Step 3.

Consequently
$T_N(\lambda)=\lambda Q_N/(\lambda q_N)$ is continuous on
$\mathcal K_N$, since its denominator is at least $a_0$. Step 2 shows
$T_N(\mathcal K_N)\subset\mathcal K_N$. The compact-convex fixed-point
theorem in the locally convex space of signed measures with the weak
topology supplies $\nu_N=T_N(\nu_N)$, that is,
$\nu_NQ_N=\alpha_N\nu_N$ with $\alpha_N=\nu_Nq_N\ge a_0$.
The same argument on the nonempty closed convex exchangeable subclass
gives an exchangeable QSD, because the actual kernel is permutation
equivariant. The extinction lower bound supplies
$\alpha_N\le1-q_D^N$. No spectral gap or attraction premise was used.
:::

:::{prf:proposition} A kinetic resonance that prevents conditioned TV attraction
:label: prop-chaos-conditioned-kinetic-resonance

Retain the complete canonical box kernel, including active cloning,
component collisions, positive thermostat and position noises, cloning
jitter, revival, terminal absorption and the programmed radial cap
$C_V(v)=Vv/(V+|v|)$ and restitution $0\le\alpha\le1$. Consider the parameter slice with affine restoring
force $F(x)=-\kappa(x-x_c)$ and

$$
 \kappa>0,\qquad c=h/2,\qquad c^2\kappa=1.
$$

No noise amplitude, acceptance rule or cloning parameter is changed in
the following calculation. If every retained slot initially has the same
velocity $v_0$, with $0<|v_0|<V$, then on every surviving trajectory

$$
 \boxed{v_{i,n}=v_n=(-1)^n\frac{Vv_0}{V+n|v_0|}
       \quad\text{for all }i=1,\ldots,N.}
 \tag{9.R1}
$$

This identity is independent of $N$, initial positions, companion
realizations, fitness values, collision components, restitution,
friction, jitters and noise realizations. It persists under survival
conditioning, current alive-floor conditioning, or any finite-history
alive-floor conditioning of positive probability. Mean-field iterates
from a monokinetic marked law have the same deterministic velocity
marginal, as do their normalized alive laws.

For any two distinct finite update indices, the corresponding
survival-conditioned swarm laws have TV distance one, with the convention
$\|\mu-\nu\|_{\mathrm{TV}}=\sup_A|\mu(A)-\nu(A)|$. Every QSD of
this resonant kernel is supported on $v_1=\cdots=v_N=0$. Therefore,
for every such QSD $\nu_N$ and every finite $n$,

$$
 \|\eta_n-\nu_N\|_{\mathrm{TV}}=1.
 \tag{9.R2}
$$

Nevertheless the velocity magnitude decays quantitatively:

$$
 |v_n|=\frac{V|v_0|}{V+n|v_0|},\qquad
 n\ge\left\lceil V(\epsilon^{-1}-|v_0|^{-1})\right\rceil
 \ \Longrightarrow\ |v_n|\le\epsilon
 \quad(0<\epsilon<|v_0|).
$$

Physical time is $nh$. These statements distinguish failure of full-law
TV attraction from valid weak mean-field evolution. In particular they
do not disprove uniform-time approximation in a weak population metric.
The strict nonresonance/smoothing regimes used elsewhere exclude this
parameter slice; the proposition makes no claim that those stronger
theorem conditions hold here.
:::

:::{prf:proof}
**1. Keep the actual cloning and collision stages.** If all incoming
velocities equal $v$, every component has mean $v$ and every relative
velocity is zero. The prescribed collision is consequently
$v+\alpha R_C0=v$, for every component and every matrix. Dead slots
participate with their retained velocity and are revived without
changing this conclusion. Copying and jitter change positions only.

**2. Evaluate all kinetic substeps.** Write $X$ for an arbitrary actual
prepared position, including its jitter. At resonance, the first kick
and drift give

$$
 v_1=v-(X-x_c)/c,\qquad x_1=X+cv_1=x_c+cv.
$$

The thermostat supplies its actual $v_2=c_hv_1+s_hB\xi^O$. The next
drift and force kick then give, for every value of that innovation,

$$
 x_2=x_c+c(v+v_2),\qquad
 v_3=v_2-(x_2-x_c)/c=-v.
$$

The final position diffusion changes no velocity. Applying the actual
cap gives $v^+=-Vv/(V+|v|)$. Inverting its magnitude yields
$|v^+|^{-1}=|v|^{-1}+V^{-1}$. Induction proves (9.R1).
Because this is a pathwise identity, conditioning cannot change it.
For the rooted population kernel, all velocities in each finite
component are again equal; the same calculation proves the population
claim. Finite-time survival and the normalized alive laws are defined
by the positive landing bound already proved.

**3. Identify the velocity support of every QSD.** This step does not
assume common entering velocities. Let
$E_v(S)=N^{-1}\sum_i|v_i|^2$. Component momentum and energy balance
gives $E_v(S^c)\le E_v(S)$, pathwise, for the configured restitution
$0\le\alpha\le1$. The resonant kinetic calculation applies separately
to each prepared row, so its precap velocity is $-v_i^c$.
The scalar function

$$
 f(t)=\frac{t}{(1+\sqrt t/V)^2},\qquad
 f'(t)=(1+\sqrt t/V)^{-3},\qquad
 f''(t)=-\frac{3}{2V\sqrt t}(1+\sqrt t/V)^{-4}\quad(t>0)
$$

is increasing and concave, with its continuous extension at zero.
Jensen's inequality and component dissipation therefore imply

$$
 E_v(S^+)\le f(E_v(S^c))\le f(E_v(S)).
$$

Every admitted input has $\sqrt{E_v(S)}\le V$. Along every surviving
$n$-step path, iteration gives
$\sqrt{E_v(S_n)}\le V/(n+1)$. If the entering law is a QSD, its
law conditional on survival at time $n$ is that same QSD. It must
therefore satisfy $E_v\le V^2/(n+1)^2$ almost surely for every $n$.
Intersecting these events proves $E_v=0$ under the QSD.

**4. Evaluate TV rather than infer it from weak decay.** For the
monokinetic initial law, $v_n\ne0$ at every finite $n$ and the vectors
$v_n$ at distinct indices are different. The measurable event
$\{v_1=v_n\}$ has probability one under $\eta_n$ and zero under
each other $\eta_m$ and under every QSD. This proves the stated TV
distances. For normalized alive laws the velocity marginal is likewise
$\delta_{v_n}$, so those laws are pairwise at TV distance one and
cannot have a TV limit. Solving the exact magnitude formula gives
the displayed weak velocity-relaxation time. No phase-space smoothing
has been assumed where the actual update cancels the thermostat noise.
:::

:::{prf:theorem} Quantitative mean-field equation under survival and alive normalization
:label: thm-chaos-conditioned-quantitative-map

Let $d$ be the diameter-one countable-test metric of
{prf:ref}`def-slc-empirical-metric`, and let $W_d$ be its transport
metric on laws of population measures. Evaluate the explicit constants
$A,B_*$ in that definition at the derived floor $m_*=a_0/4$, and set

$$
 C_{\mathrm{upd}}=\sqrt{A+4B_*^2},\qquad
 \varepsilon_N=\min\{1,C_{\mathrm{upd}}/(2\sqrt N)\}.
$$

This is the full displayed dependency chain of the existing one-step
proof, including companion kernels, feature radii, diversity floor,
fitness exponents, rescalers, regularization and acceptance parameters.
Its dependence on force, jitter, friction, collision restitution, noise,
domain and cap additionally enters through the derived $a_0$ and $m_*$.
There is no unspecified optimal mixing constant.
Put $\Lambda_{N,n}=(L_N)_\#\eta_n$. Uniformly over every $n\ge1$,

$$
 \boxed{W_d\bigl(\Lambda_{N,n+1},
                 (\mathcal F_h)_\#\Lambda_{N,n}\bigr)
       \le\varepsilon_N+2\delta_N.}
 \tag{9.S4}
$$

For a deterministic population trajectory $\mu_{n+1}=\mathcal F_h\mu_n$
the exact consequence is

$$
 \mathbb E_{\eta_{n+1}}d(L_N,\mu_{n+1})
 \le\varepsilon_N+2\delta_N+
       \mathbb E_{\eta_n}d(\mathcal F_h(L_N),\mathcal F_h(\mu_n)).
 \tag{9.S5}
$$

To express the alive distribution, write
$\mathcal R(\mu)=\mu(a\,\cdot)/\mu(a)$ when $\mu(a)>0$.
Choose bounded continuous determining tests $|\psi_j|\le1$ on
position-velocity space and set
$d_a(\rho,\zeta)=\frac12\sum_{j\ge1}2^{-j}|\rho\psi_j-\zeta\psi_j|$.
Then, with
$\varepsilon_N^a=\min\{1,C_{\mathrm{upd}}/(a_0\sqrt N)\}$,

$$
 W_{d_a}\bigl((\mathcal R\circ L_N)_\#\eta_{n+1},
              (\mathcal R\circ\mathcal F_h\circ L_N)_\#\eta_n\bigr)
 \le\varepsilon_N^a+2\delta_N.
 \tag{9.S6}
$$

Thus the conditioning and one-step population errors vanish with all
their constants independent of time and population size. The response
term in (9.S5) is retained; (9.S4) is a uniform approximate evolution
equation, not by itself a uniform trajectory-attraction estimate.
:::

:::{prf:proof}
For each input $S\in G_N$, the actual one-step consistency estimate
{prf:ref}`lem-slc-empirical-error` bounds
$\mathbb E_S d(L_N',\mathcal F_h(L_N(S)))$ by $\varepsilon_N$.
For other nonextinct inputs use the metric diameter one. Integrating
over $\eta_n$ gives a coupling with cost at most
$\varepsilon_N+\eta_n(G_N^c)\le\varepsilon_N+\delta_N$
between $(L_N)_\#(\eta_nP_N)$ and
$(\mathcal F_h)_\#\Lambda_{N,n}$. By (9.S2), replacing the first law
by $(L_N)_\#\eta_{n+1}$ costs at most $\delta_N$ in any
diameter-one transport metric. The triangle inequality proves (9.S4).
The same joint law followed by the triangle inequality against
$\mu_{n+1}$ proves (9.S5). Neither step divides by the probability of
surviving from time zero.

For (9.S6), fix $|\psi|\le1$ and an input $S\in G_N$. Write
$v=L_N'a$, $U=L_N'(a\psi)$,
$m=\mathcal F_h(L_N(S))a\ge a_0$ and
$u=\mathcal F_h(L_N(S))(a\psi)$. On $v>0$, $|U|\le v$ implies

$$
 \left|\frac Uv-\frac um\right|
 \le\frac{|U-u|+|v-m|}{m}.
$$

Indeed $U/v-u/m=(U/v)(m-v)/m+(U-u)/m$.
On $v=0$ choose any probability as the auxiliary normalized-alive
output; its test value has absolute value at most one, and the same
bound holds since $U=0$ and its right side is $1+|u|/m$.
Apply the actual bounded-test mean-square consistency bound to $a\psi$
and $a$. Each expected absolute error is at most
$C_{\mathrm{upd}}/\sqrt N$. The ratio inequality therefore gives
$2C_{\mathrm{upd}}/(a_0\sqrt N)$ for each test. Multiply by
$2^{-j}/2$ and sum to obtain $\varepsilon_N^a$.
The preceding good-input and survival-conditioning argument adds
$2\delta_N$. On the surviving output the auxiliary definition at
extinction disappears. Dead input coordinates still enter
$\mathcal F_h$; (9.S6) does not replace it by an unproved closed
evolution on the normalized alive law alone.
:::

:::{prf:corollary} Noise-dependent law conditioned on survival
:label: cor-chaos-noise-conditioned-law

Use the actual terminal-box full kernel $P_N$, its killed restriction
$Q_N$, and $\eta_n=\eta_0Q_N^n/(\eta_0Q_N^n1)$.
The formulas also apply when $\sigma_x=0$ whenever their survival
denominators are positive. Fix $r,\rho,g,k$ as in
{prf:ref}`prop-chaos-safe-center-noise`, and write

$$
\begin{aligned}
 u_k(S)&=P_N(S,M^+<k),&h_N(S)&=P_N(S,M^+=0),\\
 b_k(S)&=\min\{1,\eta_{N,g,r}(S)+B_{g,k}(p_r(\tau))\},
 &b_0(S)&=\min\{1,\eta_{N,g,r}(S)+\epsilon_r(\tau)^g\},\\
 U_n&=\eta_nu_k,&e_n&=\eta_nh_N,
 &\beta_n^k&=\eta_nb_k,\quad\beta_n^0=\eta_nb_0.
\end{aligned}                                                    \tag{9.S6a}
$$

By (9.H9), $0\le e_n\le\beta_n^0\le\beta_n^k\le1$.
Whenever $e_n<1$, the next survivor law satisfies

$$
\boxed{\quad
 \eta_{n+1}(M<k)=\frac{U_n-e_n}{1-e_n}
 \le\frac{\beta_n^k-e_n}{1-e_n}\le\beta_n^k,
 \qquad
 \|\eta_{n+1}-\eta_nP_N\|_{\rm TV}=e_n\le\beta_n^0.
\quad}                                                         \tag{9.S6b}
$$

This has no factor $n$ and no inverse probability of survival from
time zero. At $\tau=0$, if the preparation coverage deficit
$\eta_{N,g,r}(S)$ vanishes for $\eta_n$-almost every input, then
$\beta_n^k=\beta_n^0=0$: at least $g$ rows survive, and
conditioning does not change the next law. If the same coverage
identity holds at every subsequently reached conditioned input,
induction gives $\eta_n=\eta_0P_N^n$ and $M_n\ge g$ almost
surely for every $n\ge1$.

To transfer this estimate to the existing one-step mean-field law,
choose $k=\lceil m_*N\rceil\le g$ with fixed $m_*>0$, and evaluate
$C_{\rm upd}(m_*)$ in {prf:ref}`def-slc-empirical-metric`.
Whenever the canonical hypotheses of its empirical-consistency
lemma hold for these parameters, put
$\Lambda_{N,n}=(L_N)_\#\eta_n$. For $n\ge1$,

$$
\boxed{\quad
 W_d\bigl(\Lambda_{N,n+1},
       (\mathcal F_h)_\#\Lambda_{N,n}\bigr)
 \le\min\{1,C_{\rm upd}(m_*)/(2\sqrt N)\}
       +\beta_{n-1}^k+\beta_n^0.
\quad}                                                         \tag{9.S6c}
$$

For $n=0$, replace $\beta_{-1}^k$ by $\eta_0(M<k)$.
If $\sup_n\beta_n^k\le\bar\beta_N\to0$ and the displayed
one-step constant is independent of $N$, (9.S6c) is a uniform-in-time
*one-step* mean-field equation with error at most
$C_{\rm upd}(m_*)/(2\sqrt N)+2\bar\beta_N$. It concerns the
complete marked population law. Extracting its normalized alive
row distribution additionally needs a positive population alive-mass
denominator, as in (9.S6); neither operation closes a law on that
normalized row distribution alone.

*Proof.* Extinction is a subset of $\{M^+<k\}$, so restricting the
next output to survival removes exactly $e_n$ from the numerator
of its low-fraction probability and its total mass. This proves the
equality in (9.S6b). Average (9.H9) against $\eta_n$ to obtain
$U_n\le\beta_n^k$ and $e_n\le\beta_n^0$. Since
$\beta_n^k\le1$, $(\beta_n^k-e_n)/(1-e_n)\le\beta_n^k$.
The TV equality is the mixture identity (9.S2), valid for any
positive survival denominator. At $\tau=0$, both noise terms in
(9.H8) vanish for $k\le g$.

For (9.S6c), on $G_N=\{M\ge k\}$ the one-step empirical lemma
bounds the expected diameter-one metric error by
$\min\{1,C_{\rm upd}(m_*)/(2\sqrt N)\}$; off $G_N$ it is at
most one. Equation (9.S6b) at time $n-1$ gives
$\eta_n(G_N^c)\le\beta_{n-1}^k$. Couple the full next-output
empirical law to $(\mathcal F_h)_\#\Lambda_{N,n}$ by this
conditional input. Replacing that full output by its survivor law
costs at most $e_n\le\beta_n^0$ in the bounded transport metric.
The triangle inequality proves (9.S6c). $\square$
:::

:::{prf:corollary} Conditioning also on a controlled alive fraction
:label: cor-chaos-good-fraction-conditioning

For $N\ge N_{\mathrm{surv}}$, the current-time law
$\zeta_n=\eta_n(\,\cdot\mid G_N)$ is defined for every $n\ge1$ and
$\|\zeta_n-\eta_n\|_{\mathrm{TV}}\le\delta_N$.
Its counterparts of (9.S4) and (9.S6) have respective right sides
$\varepsilon_N+3\delta_N$ and $\varepsilon_N^a+3\delta_N$.

If the chosen event instead excludes every previous alive-floor
failure, define the restricted kernel
$Q_N^G(S,A)=P_N(S,A\cap G_N)$ and

$$
 \xi_n=\frac{\eta_0(Q_N^G)^n}{\eta_0(Q_N^G)^n1}.
$$

This is the original path law conditioned on
$S_1,\ldots,S_n\in G_N$, not a rejection-and-retry algorithm.
Its one-step counterparts of (9.S4) and (9.S6), for $n\ge1$, have
right sides $\varepsilon_N+\delta_N$ and
$\varepsilon_N^a+\delta_N$, respectively. Its conditioning event has
probability at least $(1-\delta_N)^n>0$. No uniform closeness of
$\xi_n$ and $\eta_n$ over all times is asserted.
:::

:::{prf:proof}
The first TV identity is conditioning on an event of complement
probability at most $\delta_N$. On $\zeta_n$ the one-step consistency
bound costs only $\varepsilon_N$, because its inputs all lie in $G_N$.
Compare $\zeta_nP_N$ to $\eta_nP_N$, then to $\eta_{n+1}$, then
to $\zeta_{n+1}$. Each of the three TV changes costs at most
$\delta_N$. Pushforward does not increase TV, giving the stated bounds
also for the alive normalization.

For the history-conditioned law, $Q_N^G1(S)\ge1-\delta_N$ on every
nonextinct input. Iterating conditional probabilities proves the
event-probability lower bound. Its exact normalized recursion is
$\xi_{n+1}=\xi_nQ_N^G/(\xi_nQ_N^G1)$, and all $\xi_n$ for $n\ge1$
are supported on $G_N$. Restricting its physical next-step law to
$G_N$ changes it in TV by at most $\delta_N$. Combining this with
one-step consistency on $G_N$ proves the two bounds. This normalization
is performed on laws of observed paths, not on separately resampled
individual transitions.
:::

:::{prf:theorem} Propagation of chaos under survival conditioning
:label: thm-chaos-conditioned-propagation

Let $\mathsf P_{N,T}$ be the law of the complete marked trajectory
$(S_0,\ldots,S_T)$ under the unchanged terminal-box kernel, with an
exchangeable nonextinct initial law. Let
$\mathsf P_{N,T}^{\rm surv}$ be this path law conditioned on
$\tau_N>T$, whenever the event has positive probability. Keep
$h_N(S)$ from (9.H6) and put

$$
 H_{N,T}=\sum_{j=0}^{T-1}
 \mathbb E_{\mathsf P_{N,T}}
 [\mathbf1_{\{\tau_N>j\}}h_N(S_j)].
                                                               \tag{9.S6d}
$$

The full-path survival transfer is exact:

$$
 \boxed{\quad
 H_{N,T}=\Pr(\tau_N\le T),\qquad
 \|\mathsf P_{N,T}-\mathsf P_{N,T}^{\rm surv}\|_{\rm TV}
 =H_{N,T}.
 \quad}                                                         \tag{9.S6e}
$$

For arbitrary position-displacing noise, (9.H9) gives the
parameterized bound

$$
 H_{N,T}\le\sum_{j<T}\mathbb E
 [\mathbf1_{\{\tau_N>j\}}b_0(S_j)].
                                                               \tag{9.S6f}
$$

If $b_0(S)\le\bar b_N<1$ on the nonextinct states reached through
step $T-1$, then $H_{N,T}\le1-(1-\bar b_N)^T\le T\bar b_N$.
For the canonical $\sigma_x>0$ box regime, the proved global bound
also gives $H_{N,T}\le1-(1-\delta_N)^T\le T\delta_N$.
Writing $c_{\rm surv}=\min\{p/8,a_0/16\}>0$ as in
{prf:ref}`def-chaos-survival-filter`, every $0<c<c_{\rm surv}$
and $T_N=\lfloor e^{cN}\rfloor$ therefore give the explicit
full-path conditioning cost
$H_{N,T_N}\le2e^{-(c_{\rm surv}-c)N}$.
More sharply, if the evaluated preparation integral satisfies
$\eta_{N,g,r}(S)\le C_\eta e^{-\kappa_\eta N}$ on those reached
states, with $C_\eta\ge0$, $\kappa_\eta>0$ and
$\epsilon_r(\tau)<1$, put
$\kappa_*=\min\{\kappa_\eta,
\rho\log(1/\epsilon_r(\tau))\}$, taking the second term as
$+\infty$ when $\epsilon_r=0$. Then every $0<c<\kappa_*$
gives, for $T_N=\lfloor e^{cN}\rfloor$,
$H_{N,T_N}\le(C_\eta+1)e^{-(\kappa_*-c)N}$.
At $\tau=0$, safe-center coverage with
$\eta_{N,g,r}(S)=0$ on the reached states gives $H_{N,T}=0$.

Suppose the initial empirical laws converge in probability to the
deterministic $\mu_0$ of
{prf:ref}`thm-chaos-finite-time-consistency`, and retain that theorem's
actual kernel, moment and continuity hypotheses. Put
$\mu_n=\mathcal F_h^n\mu_0$, let $d$ be the bounded countable-test
metric of {prf:ref}`def-slc-empirical-metric`, and define the
unconditioned finite-horizon error

$$
 E_{N,T}=\max_{0\le n\le T}
 \mathbb E_{\mathsf P_{N,T}}d(L_N(S_n),\mu_n).
$$

For $N$ with $H_{N,T}<1$, the survival-conditioned empirical laws
obey the explicit transfer bound

$$
 \max_{0\le n\le T}
 \mathbb E_{\mathsf P_{N,T}^{\rm surv}}
 d(L_N(S_n),\mu_n)
 \le\min\{1,E_{N,T}+H_{N,T}\}.
                                                               \tag{9.S6g}
$$

In particular $E_{N,T}\to0$ at every fixed $T$ by the cited
finite-horizon theorem, while $H_{N,T}\to0$ by the displayed
exponential survival bound. Thus conditioning on survival through
any fixed $T$ preserves the full empirical trajectory limit.
No uniqueness or attraction of $\mu_n$ is used.

For a single observation $n\le T$, use its actual current-time law
$\eta_n=\mathcal L(S_n\mid\tau_N>n)$ and put
$e_{N,n}^{\rm cond}=\mathbb E_{\eta_n}d(L_N,\mu_n)$.
For any $1\le\ell\le N$ and indices $j_1,\ldots,j_\ell$ from the
defining test family $(\varphi_j)$ of $d$, the finite-row
propagation estimate is

$$
\boxed{\quad
 \left|\mathbb E_{\eta_n}
       \prod_{i=1}^{\ell}\varphi_{j_i}(z_i)
       -\prod_{i=1}^{\ell}\mu_n\varphi_{j_i}\right|
 \le\frac{\ell(\ell-1)}N
   +2\sum_{i=1}^{\ell}2^{j_i}e_{N,n}^{\rm cond},
 \qquad
 e_{N,n}^{\rm cond}\le E_{N,T}+H_{N,n}.
 \quad}                                                         \tag{9.S6h}
$$

For arbitrary bounded continuous row tests $|f_i|\le1$, the same
argument gives the more general bound

$$
 \left|\mathbb E_{\eta_n}\prod_{i=1}^{\ell}f_i(z_i)
       -\prod_{i=1}^{\ell}\mu_nf_i\right|
 \le\frac{\ell(\ell-1)}N+
       \sum_{i=1}^{\ell}
       \mathbb E_{\eta_n}|L_Nf_i-\mu_nf_i|.
                                                               \tag{9.S6i}
$$

In the canonical $\sigma_x>0$ box regime, the population trajectory
has $m_n=\mu_n(a)\ge a_0>0$ for $n\ge1$. Define the normalized alive
laws $\rho_n=\mu_n(a\,\cdot)/m_n$ and
$\rho_N(S)=L_N(S)(a\,\cdot)/L_N(S)a$ on surviving states.
For every bounded continuous physical-row test $|\psi|\le1$,

$$
 \mathbb E_{\eta_n}|\rho_N\psi-\rho_n\psi|
 \le\frac{
 \mathbb E_{\eta_n}|L_N(a\psi)-\mu_n(a\psi)|
 +\mathbb E_{\eta_n}|L_Na-m_n|}{a_0}.
                                                               \tag{9.S6j}
$$

Consequently the normalized alive empirical law converges weakly
in probability to $\rho_n$ for each fixed $n\ge1$.
More concretely, sample $\ell$ *distinct* alive indices uniformly
conditional on a surviving configuration with $M\ge\ell$; on
$M<\ell$ use any fixed fallback value. If
$N\ge\lceil\ell/m_*\rceil$, then for $|\psi_i|\le1$,

$$
\begin{aligned}
&\left|\mathbb E_{\eta_n}
  \prod_{i=1}^{\ell}\psi_i(Z_i^{\rm alive})
  -\prod_{i=1}^{\ell}\rho_n\psi_i\right|\\
&\quad\le2\delta_N+\frac{\ell(\ell-1)}{m_*N}
 +\frac1{a_0}\sum_{i=1}^{\ell}
 \left(\mathbb E_{\eta_n}|L_N(a\psi_i)-\mu_n(a\psi_i)|
       +\mathbb E_{\eta_n}|L_Na-m_n|\right).
\end{aligned}                                                    \tag{9.S6k}
$$

Here $m_*=a_0/4$ and $\delta_N$ are the explicit constants of
{prf:ref}`def-chaos-survival-filter`. Thus distinct alive samples
also converge to $\rho_n^{\otimes\ell}$ at fixed time. The
normalized alive projection is a consequence of the complete
marked-law evolution, rather than a separately closed update.

Hence for each fixed $n$ and $\ell$, the survivor-conditioned
$\ell$-row marginal converges weakly to $\mu_n^{\otimes\ell}$.
The result is phase compatible: another deterministic initial law
produces its own trajectory under the same $\mathcal F_h$.
If an explicit unconditioned trajectory error is available on a
chosen horizon, (9.S6g)--(9.S6h) add the displayed survival cost to
that error for the same kernel and parameter regime.

*Proof.* On $\{\tau_N>j\}$ the conditional probability of extinction
at update $j+1$ is $h_N(S_j)$. The first-extinction events for
$j=0,\ldots,T-1$ are disjoint, proving the first equality in
(9.S6e). Conditioning a probability law on an event of probability
$1-H_{N,T}$ changes it in TV by exactly $H_{N,T}$, as proved in
(9.H11); apply that identity to the full path. Equation (9.H9),
then conditional geometric survival under a uniform hazard upper
bound, proves (9.S6f) and its specializations. For the exponential
windows use $\delta_N\le2e^{-c_{\rm surv}N}$ and
$\epsilon_r^{\lceil\rho N\rceil}
\le e^{-\rho N\log(1/\epsilon_r)}$ when $0<\epsilon_r<1$;
the $\epsilon_r=0$ term vanishes exactly. Multiply each one-step
upper bound by $T_N\le e^{cN}$.

The functional $d(L_N(S_n),\mu_n)$ takes values in $[0,1]$.
Its expectation under two path laws differs by at most their TV
distance, proving (9.S6g). The finite-horizon theorem gives
convergence in probability of each empirical output, and boundedness
of $d$ upgrades it to convergence of $E_{N,T}$.

Survival through time $n$ is invariant under permutation of row
labels, so $\eta_n$ is exchangeable. Conditional on its empirical
multiset, the first $\ell$ labels are sampled without replacement.
Independent sampling from the same multiset repeats a label with
probability at most $\ell(\ell-1)/(2N)$; because every displayed
product test has absolute value at most one, the two expectations
differ by at most $\ell(\ell-1)/N$. The independent empirical
expectation is $\prod_iL_N\varphi_{j_i}$. Telescope this product
against $\prod_i\mu_n\varphi_{j_i}$ and use
$|L_N\varphi_j-\mu_n\varphi_j|\le2^{j+1}d(L_N,\mu_n)$.
Finally, the $n$-step path TV identity bounds
$e_{N,n}^{\rm cond}$ by its unconditioned expectation plus
$H_{N,n}\le H_{N,T}\,$. This proves (9.S6h).
The identical sampling and telescoping calculation for general
$f_i$ proves (9.S6i). Each final expectation there tends to zero:
empirical weak convergence in probability under $\eta_n$ and the
boundedness $|L_Nf_i-\mu_nf_i|\le2$ give convergence in mean.
Finite products of bounded continuous row tests determine weak
convergence on the fixed finite product space.

For (9.S6j), put $v=L_Na>0$, $U=L_N(a\psi)$,
$m=m_n\ge a_0$, and $u=\mu_n(a\psi)$. Since $|U|\le v$,
$|U/v-u/m|\le(|U-u|+|v-m|)/m$, exactly as in the proof of
(9.S6). This proves the displayed expectation bound.
On $G_N=\{M/N\ge m_*\}$, at least $\ell$ alive rows are
available. Sampling them without replacement rather than from
$\rho_N^{\otimes\ell}$ changes a product-test expectation by at
most $\ell(\ell-1)/M\le\ell(\ell-1)/(m_*N)$, by the same
repeat-index coupling. On $G_N^c$ the fallback contributes at most
$2\eta_n(G_N^c)\le2\delta_N$. Telescope the product of
$\rho_N\psi_i$ against that of $\rho_n\psi_i$ and apply
(9.S6j) to every factor, proving (9.S6k). Weak convergence of
$\rho_N$ follows from (9.S6j), the marked empirical limit and a
countable convergence-determining family of physical-row tests.
$\square$
:::

:::{prf:corollary} Quantitative stationary mean-field identification under survival
:label: cor-chaos-conditioned-stationary-defect

For any QSD of the preceding actual kernel let
$\Lambda_N=(L_N)_\#\nu_N$. Then

$$
 \boxed{W_d(\Lambda_N,(\mathcal F_h)_\#\Lambda_N)
             \le\varepsilon_N+2\delta_N\longrightarrow0.}
 \tag{9.S7}
$$

The same actual stationary defect has a landscape- and noise-sensitive
refinement. Choose $k=\lceil m_*N\rceil\le g$ and $r>0$ as in
{prf:ref}`cor-chaos-noise-conditioned-law`, and set
$\beta_{N,\nu}^k=\nu_Nb_k$,
$\beta_{N,\nu}^0=\nu_Nb_0$. Since $\eta_n=\nu_N$ for a QSD,
(9.S6c) gives

$$
 W_d(\Lambda_N,(\mathcal F_h)_\#\Lambda_N)
 \le\varepsilon_N+
 \min\{2\delta_N,\beta_{N,\nu}^k+\beta_{N,\nu}^0\}.
                                                               \tag{9.S7a}
$$

These $\beta$ terms are integrals of the specified preparation
coverage deficit and Gaussian binomial tail under the QSD; no
independence of its walkers is used.

The normalized-alive projection has the corresponding bound
$\varepsilon_N^a+2\delta_N$. The laws $\Lambda_N$ are tight without
an added stationary-moment hypothesis. Every subsequential limit
satisfies $(\mathcal F_h)_\#\Lambda=\Lambda$ and is supported on
marked laws with alive mass at least $a_0$ and capped velocities.
Exchangeable QSDs exist by
{prf:ref}`thm-chaos-general-box-qsd-existence`; for those, along the
same subsequence, each fixed $k$-row law converges weakly to
$\int\mu^{\otimes k}\Lambda(d\mu)$.
For $N\ge\lceil\ell/m_*\rceil$, sample $\ell$ distinct alive
rows uniformly when $M\ge\ell$, with any fixed fallback on the
complement. Writing $\mathcal R(\mu)=\mu(a\,\cdot)/\mu(a)$, the
stationary alive-sample law obeys, for $|\psi_i|\le1$,

$$
 \left|\mathbb E_{\nu_N}\prod_{i=1}^{\ell}
       \psi_i(Z_i^{\rm alive})
 -\int\prod_{i=1}^{\ell}\mathcal R(\mu)\psi_i\,
                         \Lambda_N(d\mu)\right|
 \le2\delta_N+\frac{\ell(\ell-1)}{m_*N}.
                                                               \tag{9.S7b}
$$

Along the same subsequence its limit is therefore the mixture
$\int\mathcal R(\mu)^{\otimes\ell}\Lambda(d\mu)$.

This is a stationary population law invariant under the actual
mean-field evolution. Identifying its support as particular stationary
phases, or proving attraction and selecting their weights, remains a
population-dynamics question; it is not an extinction or alive-floor
obligation.
:::

:::{prf:proof}
Use $\eta_n=\nu_N$ in (9.S4) and (9.S6); a QSD is fixed by the exact
survival-conditioned evolution. For tightness, (9.S3) at $r=2$ gives
$\mathbb E_{\Lambda_N}\mu(|x|^2)\le K_2/a_0$. For any $R>0$,
the probability that this moment exceeds $R$ is at most $K_2/(a_0R)$.
The set of probability measures with second moment at most $R$, capped
velocity, and discrete marks is weakly compact, proving tightness of
$\Lambda_N$. The uniform alive-floor bound and $\delta_N\to0$ imply
that every limit is supported on $\mu(a)\ge m_*>0$.

For completeness, terminal consistency is preserved in these empirical
limits despite their finite-$N$ atoms. For the open boundary layer
$U_t=\{x:\operatorname{dist}(x,\partial D)<t\}$, (9.S3) implies
$\mathbb E_{\Lambda_N}\mu(U_t)\le(H_x/a_0)|U_t|$. The volume tends
to zero as $t\downarrow0$. Weak lower semicontinuity, followed by this
bound for each $t$, shows that $\mu(\partial D)=0$ almost surely under
any limit. Empirical terminal consistency therefore passes to that limit.

The actual map is weakly continuous on the resulting admitted class
with alive mass bounded below; this is
{prf:ref}`lem-mean-field-map-continuity`. Alive reward is bounded on
the box. Retained dead coordinates enter companion probabilities through
the bounded features and are replaced before force evaluation, as in
that proof. Applying a bounded Lipschitz population test $H$ to (9.S7)
and passing to the weak limit gives
$\int H\,d\Lambda=\int H\circ\mathcal F_h\,d\Lambda$.
The usual almost-sure continuity form of weak convergence applies here;
the limit gives full mass to the admitted continuity set just verified.
This proves invariance. Since every output $\mathcal F_h(\mu)$ has
alive mass at least $a_0$, invariance improves the limit's alive-floor
support from $m_*$ to $a_0$.

For exchangeable QSDs, sampling $k$ labels without replacement versus
independently from $L_N$ differs in TV by at most $k(k-1)/(2N)$.
Integrating and then using weak convergence of $\Lambda_N$ proves the
displayed mixture limit. This step does not assume independent walkers
inside a finite collision component. For alive labels use the same
repeat-index coupling on $G_N$ and its uniform bound
$M\ge m_*N$; the complement has $\nu_N$-probability at most
$\delta_N$ by (9.S1), yielding (9.S7b). The map $\mathcal R$ is
weakly continuous on marked laws with positive alive mass because
the mark is discrete. Since $\Lambda$ is supported on
$\mu(a)\ge a_0$, weak convergence of $\Lambda_N$ passes the
bounded product test through this map, proving the final mixture.
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
 +\frac{M^2\delta_N}{\alpha_N},
\qquad
\|\widetilde\nu_N-\nu_N\|_1
\leq\frac{2\delta_N}{\alpha_N}.
$$

In particular, the same stationary variance has the quantitative balance

$$
\left|\operatorname{Var}_{\nu_N}(H)
 -\operatorname{Var}_{\nu_N}(r_N)\right|
\leq \frac{A_\varphi}{N\alpha_N}
 +\frac{M^2\delta_N}{\alpha_N}
 +\frac{6M^2\delta_N}{\alpha_N}.
$$

For a bounded continuous $\varphi$, put
$g_N(S)=\mathcal F_h(L_N(S))\varphi$ and
$b_N=\nu_N|r_N-g_N|$. With the explicit $B_*$ evaluated at
$m_*=a_0/4$, the actual one-step consistency and conditional floor give

$$
 b_N\le\frac{2MB_*}{\sqrt N}+4M\delta_N\longrightarrow0.
$$

Consequently

$$
\left|\operatorname{Var}_{\nu_N}(L_N\varphi)
 -\operatorname{Var}_{\nu_N}
   \bigl(\mathcal F_h(L_N)\varphi\bigr)\right|
\leq \frac{A_\varphi}{N\alpha_N}
 +\frac{M^2\delta_N}{\alpha_N}
 +\frac{6M^2\delta_N}{\alpha_N}
 +4Mb_N.
$$

In particular, eliminating the unknown QSD eigenvalue by its proved
lower bound $\alpha_N\ge a_0$, the right side is at most the fully
parameterized quantity

$$
 \frac{A_\varphi}{Na_0}
 +\frac{8M^2B_*}{\sqrt N}
 +M^2\left(\frac7{a_0}+16\right)\delta_N.
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
$M^2$. The survival-conditioned floor theorem gives the sharper bound
$\nu_N(G_N^c)\leq\delta_N$. Integrating the preceding inequality
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

For the last estimate, on $G_N$ the quantitative bias theorem bounds the
unconditioned mean error by $2MB_*/\sqrt N$; on its complement it is
at most $2M$. The sharper QSD floor bound
{prf:ref}`thm-chaos-survival-uniform-floor` gives
$\nu_N(G_N^c)\le\delta_N$. Passage from each unconditioned mean to
$r_N$ adds at most $2M(1-q_N(S))$, since the two means differ by the
weight of the extinct part times a difference of numbers in $[-M,M]$.
Integrating proves the displayed explicit bound for $b_N$.
Finally, for $|f|,|g|\leq M$ under the same probability law,

$$
|\operatorname{Var}(f)-\operatorname{Var}(g)|
\leq4M\,\mathbb E|f-g|.
$$

Use this with $f=r_N$ and $g=g_N$, and then substitute the explicit
$b_N$ bound and $\alpha_N\ge a_0$. This completes each asserted estimate.
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

:::{prf:remark} Quantitative trajectory, structural confinement and long-time scope
:label: rem-chaos-structural-quantitative-scope

The scalar constants in this chapter enter the matching weak-metric
estimate of {prf:ref}`thm-slct-trajectory`, its active quadratic-growth
reward extension {prf:ref}`thm-slct-unbounded-trajectory`, and the regional
force-profile extension {prf:ref}`cor-slcs-trajectory`.
{prf:ref}`cor-slct-finite-marginals` makes their finite-row chaos error
explicit. The structural confinement estimates use selection flux,
regional adverse transfers and tail defects, rather than imposing
convexity at infinity.

Uniform-time empirical approximation and stationary-limit exchange are
fully evaluated in {prf:ref}`cor-slcp-uniform-iid-mean-field`, whose zero
fitness exponents are essential to its independence proof. For active
cloning, {prf:ref}`thm-slcr-structural-path-rate` gives a finite-particle
TV rate when its full-update tail and communication estimates close.
Its population-size dependence does not supply uniform-time nonlinear
phase attraction. Such attraction still requires a population-level
argument within the declared attraction region, with its recovery and
communication defects controlled. Distinct attracting phases remain
compatible with the unique evolution map defined for each initial law.

The long-time extension in {prf:ref}`thm-slcm-joint-invariant` identifies
stationary and joint occupation limits of this same population map with
explicit error $a_N+1/T$. The actual active-cloning regime in
{prf:ref}`thm-slca-active-stationary` and
{prf:ref}`cor-slca-joint-stationary-time` supplies uniform moments,
finite-population stationarity and an explicit simultaneous large-population,
long-time observation schedule. These results identify invariant population
dynamics. Fixed-phase support, arbitrary instantaneous diagonals and unique
stationary mixture weights follow under the distinct quantified conditions of
{prf:ref}`thm-slcm-fixed-support`, {prf:ref}`cor-slclt-moment-localization`
and {prf:ref}`thm-slcj-phase-weights`, respectively. The order obstruction
{prf:ref}`thm-slcj-order-obstruction` explains why distinct nonlinear phases
need not preserve the same weights in both orders of limits.

The population-independent Keystone bound in
{prf:ref}`thm-slcn-keystone-power` is transferred through the exact signed
balances of {prf:ref}`thm-slkd-signed-cloning` and
{prf:ref}`thm-slkd-full-position`. The complete quadratic drift differs
between the finite and population updates by the explicit $N^{-1/4}$ bound
of {prf:ref}`thm-slqc-quadratic-consistency`. Given the quantified phase
attraction and coverage inputs, {prf:ref}`thm-slcn-uniform-rate` and
{prf:ref}`cor-slcn-general-profile` supply an $N$-independent time-decay
profile and a fully displayed particle-error floor tending to zero.
Quadratic drift alone does not distinguish all population laws; its
full-law consequences require the stated additional dissipation estimate.


A closed active-cloning regime is now proved by
{prf:ref}`thm-slcc-active-contraction`: its population contraction constant
$q_2=1-\epsilon_2+2L_R+L_R^2$ is computed from the actual kinetic
minorization and marked-component perturbation. It retains actual collisions,
and {prf:ref}`cor-slcc-positive-exponents` gives a strictly positive,
explicit selection interval. Under its bounded configured reward and finite
discrete-center hypotheses, {prf:ref}`thm-slcf-nonlinear-restart` proves
uniform-time empirical approximation with an explicit vanishing error floor;
{prf:ref}`cor-slcf-stationary-limits` proves stationary chaos and both orders
of limits. The population law converges in TV; empirical approximation uses
the stated bounded transport metric. No phase-attraction constant is assumed
in this closed regime.


For unbounded quadratic-growth raw reward,
{prf:ref}`thm-slcw-active-contraction` replaces ordinary TV feedback control
by its explicit fourth-moment weighted norm. It derives $q_w<1$ from the
actual marked collision law and kinetic kernel, with a nonempty positive
selection interval. {prf:ref}`thm-slcw-transfer` gives the complete vanishing
particle-error floor; {prf:ref}`cor-slcw-trajectories` proves uniform-time
initialized approximation, stationary chaos and both limit orders. The
same-potential substitution {prf:ref}`cor-slcw-same-potential` preserves
$R=-U$ and $F=-\nabla U$ without clipping. The finite kinetic-center
profile and the computed feedback inequality remain explicit hypotheses.


Reward-driven self-confinement without a bounded kinetic center is supplied by
{prf:ref}`thm-slceg-full-moment` and {prf:ref}`thm-slce-entropy-floor`.
Their selection coefficients come from the actual regional accepted-edge
flux; the zero-trap criterion is explicit. The latter theorem controls the
normalized entropy of the full joint positional law, retaining dependence
between walkers. Its tail and coverage budgets are quantitative and
population-independent. For full-law entropy relaxation, the exact balance
{prf:ref}`prop-slce-exact-balance` retains the transported-reference
production alongside cloning and kinetic information losses.

:::


:::{prf:remark} Full-kernel feedback and phase-compatible time horizons
:label: rem-chaos-long-time-audit

The structural chapter now bounds unbounded-reward normalization uniformly
through the actual sigmoid derivatives in
{prf:ref}`lem-slcef-logistic-normalization`, and controls the full frozen
root kernel's environment dependence in {prf:ref}`thm-slcef-environment`.
The rootwise selection Harris estimate does not complete nonlinear attraction:
{prf:ref}`prop-slcfz-unsigned-empty` proves that this specific unsigned
assembly has no admissible contraction parameters. The finite-particle
class-exit estimate {prf:ref}`thm-slcex-one-step` and the explicit growing
window {prf:ref}`cor-slcex-global-growth-window` retain the associated
probability of departure. Finally {prf:ref}`prop-slcfu-phase-obstruction`
proves why distinct nonlinear stationary phases and finite-particle
ergodicity cannot justify uniform-time approximation to every initial phase.
These results preserve the fixed-step mean-field evolution law while
specifying the extra mathematical content required for a long-time claim.
The positive trajectory conclusion is now quantitative on a diverging
horizon in {prf:ref}`thm-slcgt-growing-trajectory`, with explicit
uniform-ball initialization and fixed-row chaos. The complete infinite
path of empirical population laws converges in the stated product metric
by {prf:ref}`cor-slcgt-infinite-population-path`. These results require
neither stationary attraction nor a bounded kinetic center; they do not
change the topology to uniform convergence over all times.

:::
