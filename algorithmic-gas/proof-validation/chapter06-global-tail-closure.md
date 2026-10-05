# Global selected-source moment and tail interface

This note concerns the unchanged conservative native kernel, with death disabled,
all current and retained historical source frames alive, and no elite or external
source injection. It proves normalized moments and tails. It does not assert a
stationary-law or survivor-law mixing rate. The absorbing-box hazard result at
the end concerns the separate killed kernel.

## Actual source moment

Let $N\ge2$. Write $M_p(S)=N^{-1}\sum_i|x_i|^p$, $p\ge1$, and let
$\mu_i$ be the actual zero-jitter source positions in the frozen literal-copy
plan. The independent one-donor companion kernel uses the actual squashed
phase-space features. With feature radii $R_x,R_v$, velocity weight
$\lambda_{\rm alg}\ge0$, and bandwidth $\epsilon_C>0$, its squared diameter
is bounded by

$$
D^2=4(R_x^2+\lambda_{\rm alg}R_v^2),\qquad
\kappa=\exp[-D^2/(2\epsilon_C^2)]>0.
$$

This is a bound on algorithmic features; physical positions remain unbounded.
Actual Gaussian weights lie in $[\kappa,1]$. For current-frame companions,
$p_{ij}\le[(N-1)\kappa]^{-1}$ for every $i\ne j$.

For each actual positive logistic channel with amplitude $A_c\ge0$, floor
$\eta_c>0$ and exponent $e_c\ge0$, define

$$
F_-=\prod_c\eta_c^{e_c},\qquad
F_+=\prod_c(A_c+\eta_c)^{e_c},\qquad
a_* =\min\left\{1,{F_+-F_-\over s_c(F_-+\epsilon_c)}\right\}.
$$

Here $s_c>0$ and $\epsilon_c\ge0$ are the actual competitive-gate parameters.
The bound holds for every complete sampled measurement and nonlinear global
normalization; no fitness of an averaged measurement is substituted. Let
$a_{ij}$ be the resulting actual acceptance probability. Simultaneous
frozen-source copying gives the exact conditional identity

$$
\mathbb E[M_p(\mu)\mid S]
=M_p(S)+{1\over N}\sum_{i\ne j}p_{ij}a_{ij}
(|x_j|^p-|x_i|^p).
$$

Discard only the negative source-loss terms. Every incoming accepted-source
load obeys
$\sum_{i\ne j}p_{ij}a_{ij}\le a_*/\kappa$. Therefore

$$
\mathbb E[M_p(\mu)\mid S]\le(1+a_*/\kappa)M_p(S).
\tag{GTM.1}
$$

This uses no labeling in its observable or bound; permutations preserve all
finite sums. It also uses no physical support bound and introduces no factor
growing with $N$. Revived dead recipients are excluded from this conservative
hypothesis. Their source gain cannot silently be controlled by the same
current alive $1/N$ moment when the alive fraction can vanish.

## Native kinetic step and unbounded force

Suppose the actual force is $F(x)=-\omega x+r(x)$, with
$|r(x)|\le B\sqrt d$ globally. The harmonic benchmark has $(\omega,B)=(1,0)$;
the actual Rastrigin benchmark has $(\omega,B)=(2,20\pi)$. No sampled force
maximum is used. For BAOAB define

$$
t=h/2,\quad c=e^{-\gamma h},\quad b=t(1+c),\quad
\eta=tb,\quad a=|1-\eta\omega|,\quad
\tau=\sqrt{t^2q^2+s^2}.
$$

Here $q$ is the actual OU standard deviation and $s$ the actual final position
standard deviation. Every frozen entering collision input, including a revived
slot when used in a different killed application, and every copied historical
velocity must obey its stated bound $V$. Native restitution obeys
$\alpha\in[0,1]$; the component mean and relative-velocity triangle bound give
$|v_i^C|\le(1+2\alpha)V$. The actual first viscous matrix is stochastic when
$0\le t\nu\le1$, in both declared count and row normalizations. It may depend
on the full recipient jitter. Its pathwise velocity norm bound remains valid
without assuming that graph is independent of the jitter.

Let $g_{d,p}$ be a certified upper bound on $\|Z\|_{L^p}$ for a standard
$d$-dimensional Gaussian. The implementation uses the next even moment:
$k=\lceil p/2\rceil$,
$g_{d,p}=[\prod_{j=0}^{k-1}(d+2j)]^{1/(2k)}$, which is exact at even $p$ and
valid otherwise by monotonicity of probability-normalized $L^p$ norms.
Condition on the complete frozen source/copy plan and entering inputs BEFORE
recipient jitter and its dependent viscous graph. Component rotations may be
held fixed because their independent draws depend only on the frozen copy
components; alternatively average them jointly. Minkowski over the uniform row
law and the original full-support jitter, Haar, OU and position laws gives

$$
\left(\mathbb E[M_p(X^+)\mid\mu,\text{pre-jitter frozen inputs}]\right)^{1/p}
\le aM_p(\mu)^{1/p}+C_p,
\qquad
C_p=a\sigma_Jg_{d,p}+b(1+2\alpha)V+\eta B\sqrt d+\tau g_{d,p}.
\tag{GTM.2}
$$

Accepted rows use jitter $\sigma_J$ and unaccepted rows use zero; replacing
their amplitudes by the common upper amplitude proves the inequality. The
unrestricted OU and final-position noises combine into the stated isotropic
Gaussian. B2 and the final velocity cap do not alter the completed position.

For $p>1$ and $0<a<1$, let
$\lambda_p=(1+a^p)/2$,
$\varepsilon=(\lambda_p/a^p)^{1/(p-1)}-1>0$, and
$B_p=(1+\varepsilon^{-1})^{p-1}C_p^p$.
Young's inequality proves

$$
\mathbb E M_p(X^+)\le\lambda_p\mathbb E M_p(\mu)+B_p.
\tag{GTM.3}
$$

For $p=1$ use $(\lambda_1,B_1)=(a,C_1)$; for $a=0$ the source term vanishes
exactly. Combining (GTM.1) and (GTM.3),

$$
P M_p\le r_pM_p+B_p,\qquad
r_p=\lambda_p(1+a_*/\kappa).
\tag{GTM.4}
$$

When $r_p<1$, iteration yields
$\mathbb E M_p(S_n)\le r_p^n\mathbb E M_p(S_0)+
B_p(1-r_p^n)/(1-r_p)$.
This is a global unbounded-space moment estimate with an $N$-independent rate
and floor, even when the landscape has slow zones. It does not imply a pure
global contraction of distinct phase laws. Any conservative invariant law with
finite moment inherits the floor; this drift alone does not prove its existence
or unique attraction.

## Finite historical windows

For this native historical case component collision is disabled by
`restitution=None`, and accepted historical donors literally copy their
recorded velocities. `Some(0)` still enables the component path and is not an
equivalent configuration: the native component constructor rejects accepted
historical donors. The kinetic interface therefore uses effective
$\alpha=0$ in this case. With $j\ge1$ past all-alive frames, the pool has $N(j+1)$ rows and at most one
excluded current self. Its normalization gives

$$
\mathbb E M_p(\mu)\le M_p(S_n)
 +{a_*\over\kappa}\,{N(j+1)\over N(j+1)-1}
 {1\over j+1}\sum_{l=0}^j M_p(S_{n-l}).
\tag{GTM.5}
$$

The ratio is at most $4/3$ for all $N\ge2,j\ge1$. Current-only warmup has
the sharper coefficient one. After expectation, (GTM.3) gives
$m_{n+1}\le\lambda_pm_n+(4\lambda_pa_*/(3\kappa))
\max_{0\le l\le H}m_{n-l}+B_p$, where $m_n=\mathbb E M_p(S_n)$.
The maximum is over expected moments, not a sampled maximum inside an
expectation. If
$R_p=\lambda_p(1+4a_*/(3\kappa))<1$ and $D_p=B_p/(1-R_p)$,
the positive excesses obey
$[m_{n+1}-D_p]_+\le R_p\max_{0\le l\le H}[m_{n-l}-D_p]_+$.
After each $H+1$ updates every old excess has left the memory window; induction
gives the block factor $R_p^{\lfloor n/(H+1)\rfloor}$ against the initial
window's maximum excess. This proves finite-history normalized moment and tail
control. It requires all source frames alive and every used historical velocity
bounded; it is not applied to a disappearing survivor pool.

## Tail and escape consumer

For $0\le r<p$ and a proved moment bound $M$ for the precise law of interest,

$$
\mathbb E{1\over N}\#\{i:|x_i|>R\}\le M/R^p,\qquad
\mathbb E{1\over N}\sum_i|x_i|^r1_{|x_i|>R}\le M/R^{p-r}.
\tag{GTM.6}
$$

Use the time-dependent bound from (GTM.4) or its historical analogue; no compact
support replaces Gaussian tails. The Chapter 6 coupled-tail lemma then charges
cross-coupled costs using both laws' own moment bounds. A regional discrepancy
operator still needs proved transfer coefficients and the full slow-zone flux;
moment control does not manufacture those coefficients.

For the explicit primitive probe $R_x=R_v=2$, $\lambda_{\rm alg}=1$,
$\epsilon_C=3$, $A_r=A_s=2$, $\eta_r=\eta_s=0.1$,
$e_r=e_s=10^{-5}$, $s_c=1$, $\epsilon_c=10^{-6}$, one has
$\kappa=e^{-32/18}\simeq0.1690133$ and
$a_*\le\exp(2\cdot10^{-5}\log21)-1\simeq6.089\cdot10^{-5}$.
At $h=.04,\gamma=1$ both harmonic and Rastrigin p4/p8 interfaces absorb the
current and finite-history source gains. Strong default exponents generally
fail this sufficient inequality and are not certified by these formulas.

## Separate global absorbing-box hazard

For the actual independent final positional Gaussian with standard deviation
$s>0$, box landing in $[-L,L]^d$ is maximal at prepared mean zero. This follows
by differentiating each coordinate interval probability, whose derivative is
negative for positive mean and positive for negative mean. Put
$q_1=2\Phi(-L/s)$ and $q_d=1-(1-q_1)^d$. Conditional on every random
preparation, each of the N independent terminal alive indicators has death
probability at least $q_d$. Monotone binomial coupling therefore gives

$$
\delta_{0,N}=q_d^N\le\mathbb P(k'=0\mid\text{preparation}),\qquad
\delta_{2,N}=q_d^N+N(1-q_d)q_d^{N-1}
\le\mathbb P(k'<2\mid\text{preparation}).
\tag{GTM.7}
$$

The full preparatory law integrates without restrictions on its mean. Native
absorption and externally stopped Chapter 6 absorption are distinct kernels.
Each fixed-N chain has survival at most $(1-\delta)^n$ and mean lifetime at
most $1/\delta$. These hazards decay with N and are not claimed population
uniform. The implementation uses the positive Mills lower bound
$2\Phi(-z)\ge2\varphi(z)z/(1+z^2)$ and a lower coordinate-union bound to
store rigorous real formulas in log scale. A floating lower bound may underflow
to zero; no positive floor is inserted into a lower certificate.

## Conditional Gaussian LSI scope

The final candidate position law given the full preparation is a product
Gaussian on its Gaussian-covered coordinates, with covariance $s^2I$ and LSI
$\rho=1/s^2$. Apply Chapter 6 concentration only to a bounded smooth or form-domain
observable of this very conditional law. The retained-data probe uses
$F=\operatorname{clip}(N^{-1}\sum_i(X_{i,1}-m_{i,1}),[-1,1])$;
it is permutation invariant, has conditional mean zero by symmetry, and
$|\nabla F|^2\le N^{-1}$ almost everywhere. Its actual recorded terminal
innovations test the bound at an externally fixed checkpoint, with independent
trajectory seed units and absorbed-zero initial-law normalization.
This does not prove an LSI for the complete nonlinear stationary or killed
alive law across status strata, and it does not supply its entropy dissipation.

## Conservative component-energy and gate-weighted jitter refinement

Keep every actual native parameter in (GTM.2). Suppose the first viscous
matrix is doubly stochastic with nonnegative entries (identity when
$\nu=0$, or the count-normalized symmetric graph under $h\nu/2\le1$).
This is an additional hypothesis; general row normalization does not imply it.

For current-frame component collision, every frozen entering velocity has norm
at most $V$. Component momentum is preserved and relative energy is multiplied
by $\alpha^2\le1$, so the full-slot probability-normalized $L^2$ norm is at most
$V$. The pathwise row bound is $(1+2\alpha)V$. Consequently, probability-space
monotonicity for $1\le p\le2$, and interpolation for $p>2$, give

$$
\|V_{\rm collision}\|_{L^p({\rm row})}
\le V(1+2\alpha)^{(1-2/p)_+}=:V_p.
\tag{GTM.8}
$$

The same bound survives the first doubly stochastic matrix by Jensen and its
column sums. For the native historical `restitution=None` branch the literal
copied source velocities are individually capped by hypothesis, so $V_p=V$
with effective $\alpha=0$. No component operation is invented for that branch.

In the all-alive conservative kernel, only accepted recipients receive fresh
position jitter. If $I_i$ is its accepted indicator and $Z_i$ is its independent
standard Gaussian, then, conditional on frozen measurement and source draws,
$\mathbb E[\|I_i\sigma_J Z_i\|^p]\le a_*\sigma_J^p g_{d,p}^p$.
Averaging over the actual full nonlinear measurement leaves the same inequality.
After probability normalization over rows, the jitter contribution to the
Minkowski budget is therefore $\sigma_J g_{d,p}a_*^{1/p}$, independent of $N$.
Mandatory revival and elite injection are excluded; their always-jittered or
external recipients do not satisfy this accepted-gate argument.

Thus replace only the additive budget in (GTM.2) by

$$
\widetilde C_p
 =a\sigma_Jg_{d,p}a_*^{1/p}
  +bV(1+2\alpha)^{(1-2/p)_+}
  +\eta B\sqrt d+\tau g_{d,p}.
\tag{GTM.9}
$$

All native parameters retain their original values. The same Young
$\lambda_p,\varepsilon_p$ and source coefficient give
$\widetilde B_p=(1+\varepsilon_p^{-1})^{p-1}\widetilde C_p^p$,
$\widetilde R_p=R_p$, and floor
$\widetilde B_p/(1-R_p)\le B_p/(1-R_p)$ when $R_p<1$.
The moment rate is unchanged, while the floor is reduced. This is a complete
conservative p-moment estimate, not a population-law contraction.

## Root-moment closure without Young slack

Apply Minkowski on the joint space of the complete native randomness and a
uniform row before taking any nonlinear power of expectation. With
$S=1+a_*/\kappa$ for current sources, (GTM.1) and (GTM.9) imply

$$
u_n=(\mathbb E M_p(S_n))^{1/p},\qquad
u_{n+1}\le r_pu_n+\widetilde C_p,\qquad
r_p=aS^{1/p}.
\tag{GTM.10}
$$

Here the symbol is $u_n$ (root moment), not the unrooted moment $m_n$.
Whenever $a^pS<1$, the explicit bounds are
$u_n\le r_p^nu_0+\widetilde C_p(1-r_p^n)/(1-r_p)$ and
$\limsup\mathbb E M_p\le[\widetilde C_p/(1-r_p)]^p$.
The closure criterion is weaker than $\lambda_pS<1$ and the floor is no larger
than the Young floor: the fixed point of the root recurrence is a subsolution
of the Young moment recurrence. One must not replace this by an unproved
linear unrooted-moment recurrence or claim rate $r_p^p$ for its excess above
a positive floor.

For finite all-alive historical windows with native collision disabled as
stated above, use $S=1+4a_*/(3\kappa)$ and the maximum of the recent expected
root moments. Subtracting $\widetilde C_p/(1-r_p)$ from this maximum yields a
positive excess contracting by $r_p$ after each $H+1$ steps. The initial
window maximum remains in the explicit bound. All full-support Gaussian
noise, native gate probabilities, true force centers, and probability
normalization are retained.

For $0<a<1$, an explicit sufficient parameter interval follows without a
numerical optimization. Define
$\eta_F=\sum_c\zeta_c\log[(A_c+f_c)/f_c]$.
The native nonnegative gate regularizer gives $a_*\le(e^{\eta_F}-1)/s_c$,
where $s_c>0$ is the actual gate saturation.
Consequently current-frame root closure follows from

$$
\eta_F<\log\{1+s_c\kappa(a^{-p}-1)\}.
\tag{GTM.11}
$$

For the supported finite-history `None` branch replace $s_c\kappa$ in the right
side by $3s_c\kappa/4$. With the actual native equal channel exponents $\zeta$,
amplitude two and floor $0.1$, divide the right side by $2\log21$ to get an
explicit sufficient upper bound on $\zeta$. This inequality keeps the
nonnegative native epsilon and gate clipping; dropping their improvements
only makes the bound conservative. It is a global analytic interval, not an
empirical estimate of a worst-case fitness configuration.
