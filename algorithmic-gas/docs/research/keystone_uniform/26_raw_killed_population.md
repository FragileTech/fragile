# Raw quadratic reward for the large-box marked population law

(sec-rkpf-register)=
## 1. Actual alive-only reward and marked population

:::{prf:definition} Raw quadratic marked population record
:label: def-rkpf-record

Retain the full marked revival population kernel of
{prf:ref}`def-kpf-marked-register`, with both count-viscous kicks,
the actual joint noisy second provider, original frozen-slot component
Haar velocities and terminal mark on $D_L=[-L,L]^d$. Use the unchanged
same-potential channels
$$
F(x)=-x,\qquad R(x)=-|x|^2/2.
$$
Only the normalized alive law
$\alpha_\mu=\mu_A/m_A$ enters reward and diversity normalizers and
donor probabilities. Dead roots accept mandatory revival with probability
one. Their retained positions do not enter reward statistics or copied
sources; their original velocities do enter the component collision.
The stored speed is capped by $V$.

Use the primitive $r_8,B_8,H_8$ and Gaussian-mixture marked class
$\mathfrak G_{8,s,L}$ of that record. Use $a_0,c_0,G_0,J_b$ from
the positive-base logistic linear register of
{prf:ref}`def-rqf-record`, with actual powers
$p_b=\theta\bar p_b$, $\bar p_b>0$.
Choose $\epsilon_0,m_0,\theta_0,\bar a,\bar c,k_0,H_{{\rm p},p},M_2$
exactly by (KPF.16). These constants use harmonic moment bounds and the
logistic fitness range, and do not depend on $L$ or bounded raw reward.
In particular put
$$
H_Q=H_8/m_0,\qquad B_A=1+\sqrt{H_Q}.
\tag{RKPF.1}
$$
For every moment-class law with alive mass at least $m_0$, the exact
conditional alive law has eighth moment at most $H_Q$.
:::

(sec-rkpf-interfaces)=
## 2. Raw reward feedback and boundary variation

:::{prf:lemma} Complete raw reward interfaces for marked preparation
:label: lem-rkpf-raw-interfaces

Assume $0<\theta\le\theta_0$. Let $\mu,\mu'$ belong to
$\mathfrak G_{8,s,L}$ with alive masses at least $m_0$ and dead masses
at most $\epsilon_0$.
Evaluate $T_r^Q,Q_x,C_{F,0}^Q,C_J^Q,L_J^Q$ in
(RQF.3), (RQF.6)--(RQF.8) at $H_8=H_Q$; keep the actual unchanged
diversity primitives. Then the marked preparation feedback proposition
{prf:ref}`prop-kpf-marked-preparation-feedback` holds with
$L_J$ replaced by $L_J^Q$ and
$$
\begin{gathered}
G=e^{1/4},\quad
C_\dagger=64D_JB_A^2G(1+\kappa_D^{-1})(1+\kappa_C^{-2}),\\
h_A=(\kappa_C^2m_0)^{-1},\qquad
h_D=(\kappa_Cm_0)^{-1}+\epsilon_0(\kappa_Cm_0^2)^{-1},\\
C_A^Q=L_J^Q+C_\dagger
 [1/(\kappa_Cm_0)+(1+\epsilon_0)h_A+1+L_J^Q],\\
C_D=C_\dagger(1+\epsilon_0)h_D.
\end{gathered}
\tag{RKPF.2}
$$
For a common consistent root law $\zeta$ with moment at most $H_8$
and dead mass at most $\epsilon_0$, this is the explicit bound
$$
\|\zeta J_\mu-\zeta J_{\mu'}\|_{W_4}
\le C_A^Q(\theta+\epsilon_0)
          \|\alpha_\mu-\alpha_{\mu'}\|_{W_4}
       +C_D\|\mu_D-\mu'_D\|_1.
\tag{RKPF.3}
$$
The marked weighted BV lemma
{prf:ref}`lem-kpf-marked-preparation-bv` also holds with its root
fitness spatial derivative replaced by
$$
L_F^Q=\theta[Q_x(H_Q)+J_s/\sigma_s].
\tag{RKPF.4}
$$
In detail, use the unchanged $\ell_b,L_D,r_J,c_J,L_{G,p},L_{J,p}$
of that lemma, and set
$$
\begin{aligned}
C_{\rm out}^Q&=2a_*\ell_C/\kappa_C+L_{\rm rec}L_F^Q,\\
C_{\rm in}^Q&=[a_*\ell_C+L_{\rm don}L_F^Q
                      +\epsilon_0\ell_C/m_0]/\kappa_C,\\
C_r^Q&=L_D+C_{\rm out}^Q+2C_{\rm in}^Q,\\
B_{p,L}^Q&=d[L_{G,p}+L_{J,p}+C_r^Q(1+H_8^{p/8})]
                                          +\mathcal T_p(L)
\end{aligned}
\tag{RKPF.5}
$$
for $1\le p\le5$. The actual prepared physical phase law satisfies
$\sum_j\int W_p|D_{X_j}(\mu J_\mu)|\le B_{p,L}^Q$, including the
velocity-fibre form required for the joint OU scores.
:::

:::{prf:proof}
The exact alive normalization lemma
{prf:ref}`lem-kpf-provider-normalization` gives eighth moment at most
$H_Q$ for each conditional alive environment. Apply
{prf:ref}`lem-rqf-normalization` to those actual probabilities, in
full $W_4$ variation. The actual raw reward means and variances,
including their denominators, are compared by the explicit Gaussian
logistic tail at that budget. Thus the source-first alive-forest
provider proof gives $L_J^Q$ in place of its bounded-reward constant.
It requires no Gaussian representation of the normalized alive law.

All additional terms of the marked provider proof are unchanged:
incoming dead children are mandatory leaves, with intensities given
by (KPF.9); a common dead root has a mandatory alive donor, whose
weighted normalized-kernel difference is charged before exposing its
remaining component. Neither step evaluates a dead reward. The
averaged alive-edge, extra-dead-leaf and common-dead-root charges are
$$
\theta L_J^Q\delta_A+
C_\dagger[d_0\delta_A+H_D]+\epsilon_0C_\dagger
                         [(1+\theta L_J^Q)\delta_A+H_D],
$$
where $\delta_A=\|\alpha_\mu-\alpha_{\mu'}\|_{W_4}$,
$d_0=\epsilon_0/(\kappa_Cm_0)$ and
$H_D=\epsilon_0h_A\delta_A+h_D\|\mu_D-\mu'_D\|_1$.
The source is exposed first, and the complete first-marginal alive
forest bounds conditional query counts, so these charges retain
all component dependence. Substitution gives exactly (RKPF.2)--(RKPF.3).

For BV, the environment and its normalized alive law are fixed while
differentiating the dummy root position. The alive reward mean and
variance are their exact deterministic population statistics. The
tail lemma at $H_Q$ gives the global root reward-fitness gradient
$\theta Q_x(H_Q)$; the actual bounded sampled-diversity gradient
adds at most $\theta J_s/\sigma_s$. This proves (RKPF.4).
On the persistent branch only an alive root is possible. Its
measurement, no-edge factor and first incoming intensity have the
full derivative bounds in (RKPF.5). Child subtrees, given their
types and used outgoing edges, are unchanged Markov readouts.
The dead-child intensity has no fitness gate and contributes the
unchanged $\epsilon_0\ell_C/(\kappa_Cm_0)$ term.

Translate the root Gaussian on this persistent branch. The input
alive restriction adds its face derivative measures, bounded by
{prf:ref}`lem-kpf-boundary-trace`, with unconditional moment budget
$H_8$. Copied alive roots and revived dead roots instead use their
own independent recipient jitter for the output derivative. Their
combined mass and source-weight bounds are precisely $r_J,c_J$.
Original frozen-slot velocities make each fixed-pattern collision
readout spatially independent. The direct population-row argument
of {prf:ref}`lem-rqf-preparation-bv`, with this one alive boundary
term, consequently yields (RKPF.5). Keeping absolute derivatives
before graph and velocity mixing gives its fibrewise version.
There is no bounded extension of the raw reward in either interface.
:::

(sec-rkpf-endpoints)=
## 3. All parameter constants precede the box choice

:::{prf:definition} Noncircular raw marked parameter interval
:label: def-rkpf-positive-endpoints

First compute the uniform source bound $\overline B_5^Q$ from
(RKPF.5) at $p=5$, using $\bar a,\theta_0,G_0$ for the gate and
derivative envelopes and replacing $\mathcal T_5(L)$ by the
$L$-independent bound $\overline{\mathcal T}_5$ in
{prf:ref}`lem-kpf-boundary-trace`. Use moment $H_8^{5/8}$.
With this source bound and the already fixed $H_{{\rm p},5}$,
compute the uniform fibrewise joint scores $M_*,S_x,S_w$ of
{prf:ref}`lem-pvb-joint-scores` at its positive endpoint $\bar\nu$.
Compute the resulting finite $\overline C_{\rm fb}^Q$ from
{prf:ref}`thm-pvb-viscous-feedback`, using $\bar\nu$ in every
increasing denominator or factor.

Next compute the frozen marked drift, common floor and Harris constants
$B_g,u_0,r_H,C_B,R,R_x,\bar\epsilon_H,\beta,q_H,g_H$ exactly as in
(KPF.17) and its preceding definitions. This uses $r_4,B_4$, the
fixed joint velocity moment $M_2$ and isolated-root factor
$(1-\bar a)\exp[-\bar c-\epsilon_0/(\kappa_Cm_0)]$.
The calculation contains no raw-reward bound or box radius.
Set
$$
\begin{gathered}
C_J=1+D_JB_A/\kappa_C,\quad
C_K=1+\beta(1+u_0)(1+B_4),\\
C_{{\rm fb},Q}^w=[1+\beta(1+u_0)]\overline C_{\rm fb}^Q,\\
\omega=\max\{1,\beta R+1,\beta B_A,8C_KC_D/g_H\}.
\end{gathered}
$$
Finally define the three strictly positive endpoints
$$
\boxed{\begin{aligned}
\theta_*^Q&=\min\{\theta_0/2,
                  g_H\beta m_0/(16C_KC_A^Q)\},\\
\epsilon_*^Q&=\min\{\epsilon_0/2,
                  g_H\beta m_0/(16C_KC_A^Q)\},\\
\nu_*^Q&=\min\left\{\bar\nu,
 \frac{g_H}{8C_{{\rm fb},Q}^w
 [C_J/\beta+2C_A^Q(\theta_0+\epsilon_0)/(\beta m_0)+C_D]}\right\},\\
r_*^Q&=(1+q_H)/2<1.
\end{aligned}}
\tag{RKPF.6}
$$
All constants through (RKPF.6) depend only on the fixed primitive
parameters. After computing them, choose one box radius
$$
\boxed{\quad
L>\max\left\{1,R_x,(H_8/\epsilon_*^Q)^{1/8},
 [\omega/(\beta u_0)]^{1/4},
 \frac{bV_c+\sigma_*\sqrt{2\log(2d/\epsilon_*^Q)}}{1-a_x}
                     \right\},\quad
\sigma_*^2=a_x^2\sigma_J^2+\tau^2.
\quad}
\tag{RKPF.7}
$$
This is a finite, nonempty sufficient regime. Neither the raw reward
bound on $D_L$ nor its spatial derivative bound on that box is used
to compute any endpoint in (RKPF.6).
:::

(sec-rkpf-convergence)=
## 4. Complete marked population contraction with raw reward

:::{prf:theorem} Raw same-potential large-box marked population law
:label: thm-rkpf-large-box-population

Use {prf:ref}`def-rkpf-record` and
{prf:ref}`def-rkpf-positive-endpoints`, with
$0<\theta\le\theta_*^Q$, $0<\nu\le\nu_*^Q$.
For $w=1+\beta W_4+\omega\mathbf1_{\{a=0\}}$,
$$
\|\mathcal F_L^Q\mu-\mathcal F_L^Q\mu'\|_w
\le r_*^Q\|\mu-\mu'\|_w
\qquad(\mu,\mu'\in\mathfrak G_{8,s,L}).
\tag{RKPF.8}
$$
This Gaussian-mixture marked class is complete and invariant. It has
a unique stationary revival population law $\pi_L^Q$ and the corresponding
weighted geometric relaxation. Every consistent capped input law with
positive alive mass enters the class at the finite $n_{\rm box}$ of
{prf:ref}`thm-kpf-large-box-population-convergence`, using the fixed
box $L$ in (RKPF.7). Every stationary population law with positive
alive mass belongs to this class and equals $\pi_L^Q$.
:::

:::{prf:proof}
The moment proof of {prf:ref}`lem-kpf-marked-moments` uses the actual
alive donor column bound and mandatory revival source, never a bounded
raw reward. The positive-base gate range and $\theta_0,\epsilon_0$
verify its source coefficient. Hence the eighth-moment sublevel and
fresh Gaussian representation are invariant. The box inequalities
give dead mass at most $\epsilon_*^Q$ on every class input and after
every output; the latter follows from the raw-source large-box Gaussian
tail (KPF.15), independent of input moments and fitness gaps.

Write $D=\|\mu-\mu'\|_w$. The exact alive conditional normalization
inequality (KPF.8) and $\omega\ge\beta B_A$ give
$$
\|\mu'J_\mu-\mu'J_{\mu'}\|_{W_4}\le A_JD,
\qquad
A_J=\frac{2C_A^Q}{\beta m_0}(\theta+\epsilon_*^Q)+C_D/\omega.
$$
The complete prepared difference is at most $(C_J/\beta+A_J)D$.
The prepared moment and speed bounds are the fixed $H_{{\rm p},p},V_c$;
the raw weighted BV bound (RKPF.5) verifies the joint-score hypotheses
before velocity mixing. It retains the actual second provider and both
its own OU and source correlations.

Freeze the full marked provider at $\mu$. Its root kernel has the
proved frozen marked Harris coefficient $q_H$, because (RKPF.7) gives
$\omega/(\beta L^4)<u_0$ and contains the common final position ball.
The terminal mark is a fixed common map; its pulled-back weight is
at most $(1+\beta+\omega/L^4)W_4$ by (KPF.12).
Split the full population difference into its signed entering-law
term and its common-root provider term. Change preparation first,
holding kinetic fields fixed, and then change the two kinetic fields
holding the own prepared law fixed. The total coefficient is at most
$$
q_H+C_KA_J+\nu C_{{\rm fb},Q}^w(C_J/\beta+A_J).
$$
By (RKPF.6), the active and small-dead-mass pieces of $C_KA_J$
sum to at most $g_H/4$, while its dead-measure piece is at most
$g_H/8$. The remaining kinetic feedback is at most $g_H/8$.
Thus the coefficient is at most $q_H+g_H/2=r_*^Q<1$.
This proves (RKPF.8) without holding reward statistics fixed across
the environment comparison.

The complete-class argument in the marked population theorem uses
only Gaussian latent tightness, cap, moment sublevel and zero Gaussian
mass on box faces. These hypotheses are unchanged, so the class is
closed in the Banach weighted variation space and is nonempty.
Its invariant contraction yields its unique fixed law and rate.
For arbitrary positive-alive input, the first frozen source is in
the fixed box, even when retained dead coordinates have no moment
bound. Its first output has the finite $M_{8,\rm box}$ and fresh
Gaussian representation of (KPF.15). The parameter halves in
(RKPF.6) give the same strict subsequent drift coefficient
$\lambda_{\rm burn}=(1+3r_8)/4$ and stationary moment level
$2H_8/3$. The exact burn-in argument of the marked theorem therefore
gives entry at its $n_{\rm box}$. A stationary positive-alive law is
an output and has that same finite moment bound and representation;
the drift forces it into the class. This proves all asserted uniqueness
and attraction statements for the actual raw-reward map.
:::

:::{prf:corollary} Current-time alive relaxation in the raw marked regime
:label: cor-rkpf-current-alive

For $\mu_0\in\mathfrak G_{8,s,L}$, let
$\mu_n=(\mathcal F_L^Q)^n\mu_0$, $\alpha_n=(\mu_n)_A/\mu_n(A)$ and
$\pi_L^{Q,A}=(\pi_L^Q)_A/\pi_L^Q(A)$. Then
$$
\operatorname{TV}(\alpha_n,\pi_L^{Q,A})
\le\min\{1,(r_*^Q)^n\|\mu_0-\pi_L^Q\|_w/(2m_0)\},
$$
$$
W_{2,G}(\alpha_n,\pi_L^{Q,A})^2
\le4\lambda_{\max}(G)(dL^2+V^2)
 \min\{1,(r_*^Q)^n\|\mu_0-\pi_L^Q\|_w/(2m_0)\}.
\tag{RKPF.9}
$$
The physical-time $W_2$ rate is $-\log r_*^Q/(2h)$.
At the original harmonic step and declared positive primitive profile,
the unchanged raw reward admits this nonempty sufficient interval with
one fixed large box. The default $L=2$ and $\nu=0.3$ are not certified.
:::

:::{prf:proof}
Exact normalized alive restriction has full variation Lipschitz factor
$m_0^{-1}$, by separating its alive and dead mass changes. TV is half
full variation. Apply (RKPF.8). Alive phase points lie in
$D_L\times\overline B_V$, whose squared physical diameter is at most
$4\lambda_{\max}(G)(dL^2+V^2)$. Match common mass and couple the
remaining mass to prove the Wasserstein inequality.
For the original profile, every pre-box Gaussian-polynomial constant,
weighted trace envelope, score, inverse and minorization floor is finite,
with positive required denominators. The endpoints are therefore positive
and the subsequent box bound finite. This verifies nonemptiness without
changing the reward channel or assuming positive realized variance.
:::

:::{prf:remark} Finite particles and future survival are separate interfaces
:label: rem-rkpf-particle-interface

After the fixed $L$ in (RKPF.7) has been chosen, every actual alive
reward evaluation satisfies $|R|\le dL^2/2$ and the alive-region
gradient bound $|\nabla R|\le\sqrt dL$. For finite-particle marked
weak estimates these exact alive-support bounds suffice: both alive
points lie in the convex box, so the local gradient bound gives their
Lipschitz comparison. Dead reward values do not enter fitness normalizers.
The proof of {prf:ref}`lem-spt-marked-consistency` explicitly uses only
these bounds for reward and its normalizers.

A globally bounded $C^1$ comparison extension, if desired, is
$\widetilde R(x)=-\frac12\sum_j\rho_L(x_j)^2$. Set $\rho_L(u)=u$
for $|u|\le L$; for $r=|u|-L\in[0,1]$ set
$\rho_L(u)=\operatorname{sign}(u)[L+r-r^2/2]$; and for $r\ge1$ set
$\rho_L(u)=\operatorname{sign}(u)(L+1/2)$.
Its value bound is $d(L+1/2)^2/2$ and its gradient bound
$\sqrt d(L+1/2)$. It agrees with the exact raw reward on the entire
alive box. This comparison function changes no algorithm evaluation;
it is unnecessary for the sharper inside-box constants used below.

All constants that determine the positive population endpoints and box
were already fixed before this analytic bound, so no circular dependence
on $L$ is introduced. Current-time alive restriction of the stationary
marked revival population is not a finite-swarm QSD. Conditioning on
survival through a separate horizon requires its exact own marginal
normalization and the separate finite-swarm argument.
:::

(sec-rkpf-surviving-particles)=
## 5. Actual surviving finite-swarm alive laws

:::{prf:corollary} Raw reward surviving alive law with a vanishing particle floor
:label: cor-rqk-surviving-alive-law

Fix the primitive endpoints and then the one box $L$ in
{prf:ref}`def-rkpf-positive-endpoints`. Use the actual raw-reward finite
chain with $N\ge2$, stopped at its first all-dead time $\tau_N$.
Its initial law may be arbitrary on consistent positive-alive states
with stored velocities capped by $V$; retained dead coordinates need
no moment bound. The target is the current alive restriction
$\pi_L^{Q,A}$ of the raw marked revival population law.

In all finite comparison formulas (SPT.7)--(SPT.8), use the exact
post-box values
$$
R_b=dL^2/2,\qquad L_R=\sqrt dL,
\tag{RKPF.10}
$$
with the actual raw-reward gate and positive-base fitness derivative
parameters. Use $\epsilon_*^Q,r_*^Q$ and the marked population burn-in
from the raw theorem for $\epsilon_*,r_*,n_{\rm box}$ in the surviving
particle register. Explicitly put
$$
\begin{gathered}
\epsilon_{\rm box}=2d\exp[-((1-a_x)L-bV_c)^2/(2\sigma_*^2)]
                                      <\epsilon_*^Q,\\
m_f=1-\epsilon_*^Q,\quad p=1-\epsilon_{\rm box},\quad
\eta=p-m_f>0,\\
e_N=\epsilon_{\rm box}^N,\quad r_N=e^{-2N\eta^2},\quad
c_s=(1-\epsilon_{\rm box}^2)^{-1},\quad
\overline M=c_sM_{8,\rm box},\\
D_{\rm cl}^Q=2+2\beta(1+\sqrt{H_8})+2\omega\epsilon_*^Q,\quad
C_{\rm pop}^Q=D_{\rm cl}^Q(r_*^Q)^{-n_{\rm box}},\\
L_N=\log(N+e),\quad H_N=1+H_8+\sqrt{L_N},\quad
b_N=1+\left\lfloor\frac{\log L_N}{2\log32}\right\rfloor,\\
\alpha=1/32,\quad a=1/(128d),\quad
A_N=\min\{1,C_{\rm cons}^L(1+H_N)^{9/32}N^{-a}\},\\
D_N=1+C_{\rm mod}^L(1+H_N)^{1/4},\quad
V_N=D_N^{1/(1-\alpha)}A_N^{\alpha^{b_N-1}},\quad
T_N=(1-e_N)^{-b_N},\\
\varepsilon_{N,Q}^{s}
=T_N\left[V_N+(b_N+1)
              (\overline M/H_N+c_sr_N)\right]
                           +C_{\rm pop}^Q(r_*^Q)^{b_N},\\
u_{N,n}^Q=\min\{1,C_{\rm pop}^Q(r_*^Q)^{n-1}
                                      +\varepsilon_{N,Q}^{s}\}.
\end{gathered}
\tag{RKPF.11}
$$
Here $C_{\rm cons}^L,C_{\rm mod}^L$ are the complete explicit formulas
(SPT.7)--(SPT.8), with (RKPF.10). These constants are fixed after $L$
and are independent of $N,n$. Then $\varepsilon_{N,Q}^{s}\to0$ and
for every $n\ge1$,
$$
\mathbb E[\mathsf d_{\rm m}(\widehat\mu_n^N,\pi_L^Q)
                              \mid\tau_N>n]\le u_{N,n}^Q.
\tag{RKPF.12}
$$
For a positive physical phase matrix $G$ and the actual current alive
empirical probability $\widehat\alpha_n^N$,
$$
\boxed{\quad
\mathbb E[W_{2,G}(\widehat\alpha_n^N,\pi_L^{Q,A})^2
                                     \mid\tau_N>n]
\le4\lambda_{\max}(G)(dL^2+V^2)
       \min\{1,2u_{N,n}^Q/m_f+c_sr_N\}.
\quad}
\tag{RKPF.13}
$$
The law of the random alive empirical probability conditional on survival,
with ground distance $W_{2,G}$, has the same squared Wasserstein bound
to $\delta_{\pi_L^{Q,A}}$. Sampling a surviving swarm and then a uniform
current alive slot also has this bound. Sampling a uniform stored slot
and conditioning it to be alive within a surviving swarm gives the
bound with at most an additional factor $m_f^{-1}$.
The rate and vanishing floor are uniform over all observation times.
This does not identify a finite-swarm QSD with the population target.
:::

:::{prf:proof}
The only global bounded-reward hypothesis in the surviving particle
register is replaced here by its two proved interfaces. Population
attraction, moments, boundary variation and positive endpoints come
from {prf:ref}`thm-rkpf-large-box-population`. The finite comparison
and modulus use reward only at actual alive roots and donors inside
the fixed box. As proved in {prf:ref}`lem-spt-marked-consistency`,
the pointwise and local gradient bounds (RKPF.10) verify every reward
mean, variance and good-type comparison in (SPT.5)--(SPT.8).
Dead gates are one and evaluate no dead reward. Neither that proof
nor its measured-normalizer integration device replaces the actual
raw alive reward. Thus its conditional consistency and marked
modulus apply with precisely the constants stated here.

The independent safe-return events in
{prf:ref}`lem-spt-actual-survival` use only the alive frozen source,
bounded count-averaged velocity and own Gaussian innovations.
They are unchanged for the raw reward. They give extinction bound
$e_N$, low-alive probability $r_N$ and last-update survival-conditioned
eighth-moment bound $\overline M$, uniformly over the previous history.
The strict box inequality supplies $p>m_f$ and all displayed survival
constants, irrespective of the entering alive fraction.

For the recent window, begin at the actual law conditioned on survival
to its starting time and use its ordinary stopped continuation. Its
subsequent survival tilt is exactly (SPT.4), including the tilt of
the starting array, and is at most $T_N$ on a window of length $b_N$.
On the single window event of sufficient alive fraction and moment
at most $H_N$, the raw population forecast has the same strict
moment drift and large-box alive floor as the actual marked theorem.
The local consistency and modulus therefore give the identical
subprobability Hölder recurrence and $V_N$ estimate in the proof of
{prf:ref}`thm-spt-uniform-surviving-law`.
Charge the one bad-window probability and then the exact recent
survival tilt. The raw population global burn-in gives the uniform
forecast error $C_{\rm pop}^Q(r_*^Q)^j$ for every consistent
positive-alive starting law. This proves (RKPF.12) with its explicit
floor, without dividing by survival through the entire elapsed past.

After $L$ is fixed, $C_{\rm cons}^L,C_{\rm mod}^L$ are finite constants.
Consequently the same log-log restart proof gives $V_N\to0$,
$(b_N+1)\overline M/H_N\to0$ and $(r_*^Q)^{b_N}\to0$.
The exponentially small $e_N,r_N$ give $T_N\to1$ and
$(b_N+1)r_N\to0$. Hence $\varepsilon_{N,Q}^{s}\to0$.

Finally apply the normalized-alive weak coupling and actual alive
phase diameter argument of {prf:ref}`cor-spt-alive-w2`. The target
alive mass is at least $p>m_f$, because it is an actual stationary
output; the conditional empirical low-alive probability is at most
$c_sr_N$. This proves (RKPF.13). The Dirac empirical-law equality,
integration of optimal physical couplings and the distinct
all-slots-first mixture weight establish the stated observation-law
conclusions. All bounds use the own survival-conditioned marginals.
:::
