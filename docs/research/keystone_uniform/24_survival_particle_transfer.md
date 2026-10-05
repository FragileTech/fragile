# Recent-window transfer for the surviving marked swarm

This note transfers the large-box marked population contraction to the
actual finite gas conditioned on survival through its current observation
time. The conditioning is handled on a recent window only. The proof
retains mandatory revival, stored dead coordinates, original frozen-slot
collision velocities, sampled alive-only normalizers, and both correlated
count kicks. It proves a uniform-time vanishing particle floor for the
current alive empirical and sampled laws. It does not identify a finite-swarm
quasi-stationary distribution.

(sec-spt-register)=
## 1. The marked finite and population register

:::{prf:definition} Actual surviving chain and marked metric
:label: def-spt-register

Use every primitive and positive endpoint of
{prf:ref}`thm-kpf-large-box-population-convergence`. In particular
$F=-x$, $D_L=[-L,L]^d$, the reward is bounded $C^1$, both actual fitness
powers and the count viscosity are positive, $t\nu\le1$, and the box radius
satisfies the strict inequalities (KPF.19).
Let $S_n^N$ be the actual finite chain, $N\ge2$, stopped at the first
all-dead time
$$
\tau_N=\inf\{n\ge0:A_N=0\},\qquad
A_N=\sum_i a_i^n .
$$
An initial law may be arbitrary on consistent states with $A_N>0$ and
all stored velocities capped by $V$. No entering dead-position moment
bound is required. An absorbed array is kept fixed only to define
expectations on paths after extinction; no further gas update is performed
there.

Write $\widehat\mu_n^N=L_N(S_n^N)$ for the full marked empirical law,
$\mathcal F_L$ for the actual marked population map, and $\pi_L$ for
its stationary marked law. Set
$$
c_{\rm m}((z,a),(z',a'))=\min\{1,|z-z'|+\mathbf1_{\{a\ne a'\}}\},
$$
and let $\mathsf d_{\rm m}$ be its optimal transport distance.
Marked empirical laws are compared in this weak metric, not in total
variation. The alive empirical probability on $\{\tau_N>n\}$ is
$$
\widehat\alpha_n^N=
(\widehat\mu_n^N)_A/\widehat\mu_n^N\{a=1\}.
$$
Let $\epsilon_*,\sigma_*,\Delta_L,M_{8,\rm box},n_{\rm box},
r_*,\beta,\omega$ have their exact values in research record 22.
Define
$$
\epsilon_{\rm box}=2d\exp[-\Delta_L^2/(2\sigma_*^2)]<\epsilon_*,
\quad p=1-\epsilon_{\rm box},\quad m_f=1-\epsilon_*,
\quad \eta=p-m_f>0,
$$
$$
e_N=\epsilon_{\rm box}^N,\quad
r_N=e^{-2N\eta^2},\quad
c_{\rm s}=(1-\epsilon_{\rm box}^2)^{-1},\quad
\overline M=c_{\rm s}M_{8,\rm box}.
\tag{SPT.1}
$$
Use the strict population burn coefficient
$\lambda_8=(1+3r_8)/4$ of research record 22.
The constants $a=1/(128d)$, $\alpha=1/32$, $J_d,\kappa,g_8$ are those
in {prf:ref}`def-vupt-register` and {prf:ref}`lem-vupt-cell`.
:::

(sec-spt-survival)=
## 2. Actual survival and conditional moment budgets

:::{prf:lemma} Independent safe-return events under the correlated finite kernel
:label: lem-spt-actual-survival

For every actual surviving entering array, uniformly in its alive
fraction and retained dead positions,
$$
\Pr(A_N^+=0\mid S)\le e_N,\qquad
\Pr(A_N^+/N<m_f\mid S)\le r_N,
$$
$$
\mathbb E[M_8(L_N(S^+))\mid S]\le M_{8,\rm box}.
\tag{SPT.2}
$$
Consequently, for every $n\ge1$,
$$
\Pr(A_N/N<m_f\mid\tau_N>n)\le c_{\rm s}r_N,
\qquad
\mathbb E[M_8(\widehat\mu_n^N)\mid\tau_N>n]\le\overline M,
\tag{SPT.3}
$$
where $A_N$ is the alive count at the stated observation time.
Also $\Pr_S(\tau_N>b)\ge(1-e_N)^b$ for every surviving starting
state and every integer $b\ge0$.
:::

:::{prf:proof}
Condition on all frozen source choices, component Haar marks and original
velocities before recipient jitters and kinetic innovations. Every source
lies in $D_L$, because every dead row revives from an alive donor.
The exact finite landing identity is
$$
x_i^+=a_xx_{{\rm src},i}+bU_i+
                  a_xI_iJ_i+tq\xi_i+s\zeta_i,\qquad |U_i|\le V_c.
$$
The first count averaging is convex, so its bound holds even though $U_i$
depends on the entire jittered array. The second kick does not change
position.
Under this conditioning the vectors
$G_i=a_xI_iJ_i+tq\xi_i+s\zeta_i$ are independent over $i$ and have
centered coordinate variances at most $\sigma_*^2$.
The events $E_i=\{|G_i|_\infty\le\Delta_L\}$ are therefore independent
and have probabilities at least $p$. On $E_i$ the row is alive,
regardless of every dependence in $U_i$.
Thus the number of alive rows stochastically dominates
$\operatorname{Bin}(N,p)$, conditionally on the frozen choices.
This domination remains true after integrating them.
It proves the extinction bound $(1-p)^N=e_N$ and the lower-tail bound
$\exp[-2N(p-m_f)^2]=r_N$ by the bounded-Bernoulli exponential estimate.
One may obtain that estimate directly by applying
$\mathbb E e^{\lambda(B-\mathbb EB)}\le e^{\lambda^2/8}$ independently
to the Bernoulli variables and optimizing $\lambda$.

The same identity gives
$$
M_{8,\rm box}
=2^7[(a_x\sqrt dL+bV_c)^8+\sigma_*^8G_8],
$$
as in (KPF.15), where $G_8=d(d+2)(d+4)(d+6)$.
The bounded term may depend on the Gaussians; the displayed bound needs
only its pointwise norm bound. Thus no hidden independence from $U_i$
is used in the moment estimate.

Condition on survival up to time $n-1$. The resulting entering array law
is arbitrary but supported on surviving states, so (SPT.2) still applies.
Its survival probability for the next update is at least $1-e_N$.
Dividing the moment and low-alive numerators by this probability gives
(SPT.3), since $N\ge2$ implies $(1-e_N)^{-1}\le c_{\rm s}$.
Iterating the same lower survival bound proves the final assertion.
These divisions use one last update only; the bound never divides by
the survival probability of the full elapsed history.
:::

:::{prf:lemma} Recent-window survival change of measure
:label: lem-spt-recent-tilt

Fix $k\ge1$ and $b\ge0$. Start an ordinary stopped continuation from
the actual law $\operatorname{Law}(S_k^N\mid\tau_N>k)$.
Call its path probability $\mathbb P_k$. Its restriction conditioned
on survival through the next $b$ updates is exactly the actual path law
from time $k$ to $k+b$, conditional on $\tau_N>k+b$.
For every nonnegative path observable $Z$,
$$
\mathbb E[Z\mid\tau_N>k+b]
\le(1-e_N)^{-b}\mathbb E_k Z.
\tag{SPT.4}
$$
This includes the tilt of the starting array at time $k$.
:::

:::{prf:proof}
The Markov property expresses the desired path law as
$1_{\{\text{window survives}\}}\,d\mathbb P_k/
 \mathbb P_k(\text{window survives})$.
For each possible starting surviving state the denominator is at least
$(1-e_N)^b$ by the preceding lemma, so its mixture has the same lower
bound. This proves (SPT.4). No finite-future comparison constant for an
unproved full-array kernel has been assumed.
:::

(sec-spt-local)=
## 3. Full marked consistency and the boundary-safe weak modulus

:::{prf:lemma} Quantitative marked dense-update comparison
:label: lem-spt-marked-consistency

There are explicit primitive constants $C_{\rm cons}^L,C_{\rm mod}^L<\infty$
such that, for every surviving input array with alive fraction at least
$m_f$ and positional eighth moment at most $H$,
$$
\mathbb E[\mathsf d_{\rm m}(L_N(S^+),\mathcal F_LL_N(S))\mid S]
\le\min\{1,C_{\rm cons}^L(1+H)^{9/32}N^{-a}\}.
\tag{SPT.5}
$$
For two consistent marked input laws with alive masses at least $m_f$
and eighth moments at most $H$,
$$
\mathsf d_{\rm m}(\mathcal F_L\mu,\mathcal F_L\mu')
\le\min\{1,C_{\rm mod}^L(1+H)^{1/4}
                         \mathsf d_{\rm m}(\mu,\mu')^\alpha\}.
\tag{SPT.6}
$$
The following substitutions define the constants without an assumed
consistency or regularity coefficient.

First use the explicit preparation constants of
{prf:ref}`lem-vupt-preparation` with alive floor $m_*=m_f$:
replace $C$ by $2/(\kappa_Cm_f)$, $D_D$ by $2/(\kappa_Dm_f)$,
$L_q$ by $S_*/(m_f\sigma_s)+3S_*^3/(2m_f\sigma_s^3)$ and $A_T$ by
$$
(m_f^{-1}+D_D^2)(2S_*^2/\sigma_s^2+5S_*^6/\sigma_s^6).
$$
All the other displayed formulas for $A_D,A_{\rm p},B_{\rm p},G_{\rm p}$
are unchanged. Set
$$
k_f=a_*/\kappa_C+\epsilon_*/(\kappa_Cm_f),\quad
p_0=(1+k_f)^{1/8}+\sigma_Jg_8,\quad P_0=p_0+V_c
$$
and recompute $A_{\rm p,w},U_{\rm p},w_{\rm p},C_{K,{\rm p}}$ by the
formulas in research record 21. The purely physical kinetic constants
$A_1,C_y,C_w,w_0,y_0,k_0,o_0,U_{\rm o},B_2$ remain as in that record.
Put
$$
T_s=\max\{1,(s\sqrt{2\pi})^{-1}\},\quad
A_{\rm f}^{\rm m}=2+J_d+2o_0^2,
$$
$$
C_{\rm cons}^L=
[T_sB_2U_{\rm o}+A_{\rm f}^{\rm m}](1+p_0^8)^{9/32}
                              +T_sC_{K,{\rm p}}U_{\rm p}.
\tag{SPT.7}
$$

For the modulus, retain $w_b',\ell_f,R_b,S_*,H_b,L_a$ as in
{prf:ref}`lem-vupt-population-modulus`, and define
$$
\begin{aligned}
b_{\rm m}&=1+(3+4w_D')/(\kappa_Dm_f),\\
K_{\rm a}&=2(1+b_{\rm m})/m_f,\\
m_r^{\rm m}&=L_R+2R_bK_{\rm a},\quad
m_s^{\rm m}=2\ell_f+S_*K_{\rm a},\\
Q_r^{\rm m}&=(L_R+m_r^{\rm m})/\sigma_r
                      +4R_b^2m_r^{\rm m}/\sigma_r^3,\\
Q_s^{\rm m}&=(2\ell_f+m_s^{\rm m})/\sigma_s
                      +2S_*^2m_s^{\rm m}/\sigma_s^3,\\
E_F^{\rm m}&=H_rQ_r^{\rm m}+H_sQ_s^{\rm m},\\
D_\beta^{\rm m}
&=\frac{2w_C'}{\kappa_Cm_f}
 +\frac{2w_C'+1}{(\kappa_Cm_f)^2}
 +\frac{2L_aE_F^{\rm m}}{\kappa_Cm_f},\\
C_{\rm p}^{\rm m}
&=b_{\rm m}+8[D_\beta^{\rm m}+2b_{\rm m}/(\kappa_Cm_f)]
                  +2e^{2C}+1+k_v,\\
C_{\rm mod}^L
&=T_sC_{K,{\rm p}}(1+16P_0^4)^{1/4}(C_{\rm p}^{\rm m})^{1/8}.
\end{aligned}
\tag{SPT.8}
$$
:::

:::{prf:proof}
The full canonical preparation proofs in Chapter 9 already include
mandatory revival and an alive floor $m_*$. Stop them before kinetics as
in {prf:ref}`lem-vupt-preparation`. Their influence and finite-exploration
arguments therefore give the claimed $G_{\rm p}/N$ preparation test
second-moment bound at the actual marked empirical input.
Alive reward statistics are fixed by the entering array; only the sampled
alive diversity statistics fluctuate. The displayed $m_f$ substitutions
are exactly those in their conditional calculations.
No incoming dead edge is assigned a weak acceptance gate.

For the moment bound, an alive donor's expected incoming accepted-alive
column is at most $a_*/\kappa_C$: its $M-1$ potential alive cloners have
individual probability at most $a_*/[\kappa_C(M-1)]$.
Its mandatory-dead column is at most
$(N-M)/(\kappa_CM)\le\epsilon_*/(\kappa_Cm_f)$.
Thus the average source eighth moment is at most
$(1+k_f)M_8$, including the original retained alive positions.
The same bound holds in the population tree. Gaussian jitter and original
frozen-slot velocities give the phase moment bound $P_0^8(1+H)$.
The physical preparation weak-to-$W_4$ estimate of record 21 follows
with the recomputed constants.

The first kinetic kick at the empirical prepared law is exact. Apply the
independent conditional OU cell comparison and the actual joint second
provider force estimate of {prf:ref}`lem-vupt-kinetics`.
Only final terminal marking is new. For two fixed pre-noise landing means
$y,y'$ and capped velocities $v,v'$, maximally couple their final position
Gaussians. Their mismatch probability is at most
$|y-y'|/(s\sqrt{2\pi})$. On a successful match their terminal marks are
identical and their phase cost is at most $|v-v'|$; on failure their marked
bounded cost is at most one. Hence the marked comparison costs at most
$T_s(|y-y'|+|v-v'|)$.
This is valid at a box face as well and does not assume the indicator
of the box is Lipschitz.
The actual final independent-noise empirical comparison uses the two
copies of each phase cell, one for each mark. The cell count is $2J$,
so its coefficient is $A_{\rm f}^{\rm m}$ in place of $A_{\rm f}$.
Its conditional moment is random after OU; apply the conditional cell
estimate and Jensen exactly as in record 21.
The same kinetic stability, preparation comparison, and concave moment
integration then give (SPT.5) with (SPT.7).

For (SPT.6), put $\delta=\mathsf d_{\rm m}(\mu,\mu')$ and
$r=\sqrt\delta$. An optimal marked coupling has bad mass at most
$\delta/r$ when physical distance exceeds $r$ or marks differ.
On its complement both marks agree. Lift the alive-only weighted companion
laws to this coupling; their denominators are at least $\kappa_Dm_f$.
The same numerator subtraction as in record 21 gives full marked-type
bad mass at most $b_{\rm m}\sqrt\delta$.
Restrict its good joint submeasure to alive recipients and divide by the
larger alive mass. Complete its residual marginals to a coupling of the
two conditional alive measured laws. Since
$|m_A-m_A'|\le\delta$, its bad mass is at most
$(1+b_{\rm m})\sqrt\delta/m_f$, hence at most
$K_{\rm a}\sqrt\delta$.
Good recipients and companions are within $r$, so their reward and
diversity mean differences are bounded by
$m_r^{\rm m}\sqrt\delta,m_s^{\rm m}\sqrt\delta$.
The bounded variance subtraction and regularized reciprocal-square-root
derivative give the displayed $Q_r^{\rm m},Q_s^{\rm m}$.

At a good source pair the raw alive cloning denominator is
$\int w_C(z,y)\mu_A(dy)\ge\kappa_Cm_f$.
Subtracting it under the original full marked coupling bounds its
difference by $(2w_C'+1)\sqrt\delta$.
The accepted density relative to the full marked base law therefore
differs at good type pairs by at most $D_\beta^{\rm m}\sqrt\delta$.
For dead recipients the acceptance factor is exactly one in both laws,
so this bound is still valid; incoming edges to dead vertices are zero.
Good alive targets share their actual fitness marks and normalization.
Comparing the common incoming intensities and outgoing subprobabilities
costs at most
$[D_\beta^{\rm m}+2b_{\rm m}/(\kappa_Cm_f)]\sqrt\delta$.
The full marked ordered-forest moment bound has
$C=2/(\kappa_Cm_f)$, as in the canonical alive-floor proof.
Dead vertices can only be initial roots or incoming leaves, because
they consume their mandatory edge to an alive parent.
Stopping the common exploration at
$K=\lceil\delta^{-1/4}\rceil$ gives exactly
$$
\mathsf d(J_\mu\mu,J_{\mu'}\mu')
\le C_{\rm p}^{\rm m}\delta^{1/4}.
$$
Matching complete components uses the same original-slot velocities,
Haar matrix, copied source and jitter.
The prepared laws have phase eighth moments at most $P_0^8(1+H)$.
Apply the weak-to-$W_4$ moment upgrade, then physical kinetic stability,
then the final Gaussian marked coupling just proved.
This gives (SPT.6) and (SPT.8), including its exponent $1/32$.
The case $\delta=0$ follows by identical laws.
:::

The finite comparison and modulus use the reward only on the actual alive
provider support $D_L$. Their constants therefore remain valid with
$R_b=\sup_{D_L}|R|$ and $L_R=\sup_{D_L}|\nabla R|$, even when the
configured reward is unbounded outside the box. Every reward mean and
variance here is alive-only; mandatory dead gates are identically one and
do not use the retained dead reward. Convexity of the box gives the local
Lipschitz estimate between its alive points. For the raw harmonic reward
$R=-|x|^2/2$ the exact substitutions are $R_b=dL^2/2$ and
$L_R=\sqrt dL$.
Applying the transfer theorem to that channel additionally requires its
own proved marked population attraction and moment register. These
$L$-dependent comparison bounds are not used to construct population
endpoints before choosing $L$.

(sec-spt-uniform)=
## 4. Uniform-time transfer under current survival conditioning

:::{prf:theorem} Surviving marked empirical-law convergence with a vanishing floor
:label: thm-spt-uniform-surviving-law

Under {prf:ref}`def-spt-register` define
$$
D_{\rm cl}=2+2\beta(1+\sqrt{H_8})+2\omega\epsilon_*,
\qquad C_{\rm pop}=D_{\rm cl}r_*^{-n_{\rm box}},
$$
$$
L_N=\log(N+e),\quad H_N=1+H_8+\sqrt{L_N},\quad
b_N=1+\left\lfloor\frac{\log L_N}{2\log32}\right\rfloor,
$$
$$
A_N^{\rm loc}=\min\{1,C_{\rm cons}^L(1+H_N)^{9/32}N^{-a}\},\quad
D_N=1+C_{\rm mod}^L(1+H_N)^{1/4},
$$
$$
V_N=D_N^{1/(1-\alpha)}(A_N^{\rm loc})^{\alpha^{b_N-1}},
\qquad T_N=(1-e_N)^{-b_N},
$$
$$
\varepsilon_N^{\rm s}
=T_N\left[V_N+(b_N+1)
       \left(\frac{\overline M}{H_N}+c_{\rm s}r_N\right)\right]
                         +C_{\rm pop}r_*^{b_N}.
\tag{SPT.9}
$$
Then $\varepsilon_N^{\rm s}\to0$, and for every $N\ge2$, $n\ge1$,
$$
\boxed{\quad
\mathbb E[\mathsf d_{\rm m}(\widehat\mu_n^N,\pi_L)
                            \mid\tau_N>n]
\le u_{N,n}:=\min\{1,C_{\rm pop}r_*^{n-1}
                                    +\varepsilon_N^{\rm s}\}.
\quad}
\tag{SPT.10}
$$
The estimate is uniform over all observation times and every consistent
initial law with nonzero alive population and capped velocities.
It requires no empirical-measure total-variation limit and no bound on
retained dead coordinates.
:::

:::{prf:proof}
By the uniform first-output moment and burn-in of record 22, for every
consistent entering population law with positive alive mass,
$$
\mathsf d_{\rm m}(\mathcal F_L^\ell\mu,\pi_L)
\le\min\{1,C_{\rm pop}r_*^\ell\}\quad(\ell\ge0).
\tag{SPT.11}
$$
Indeed after $n_{\rm box}$ updates the law is in the Gaussian moment
class, whose $w$ diameter is at most $D_{\rm cl}$; use the population
contraction there. Before that time the displayed right side is at least
one because $D_{\rm cl}\ge2$.

Fix $n,N$, put $m=\min\{n-1,b_N\}$ and $k=n-m\ge1$.
Take the unconditioned stopped continuation $\mathbb P_k$ of
{prf:ref}`lem-spt-recent-tilt`, starting from the actual conditional
past law at time $k$. Forecast
$\mu_j=\mathcal F_L^j\widehat\mu_k^N$ for $0\le j\le m$.
Let $G$ be the one whole-window event that each array through this window
has alive fraction at least $m_f$ and eighth moment at most $H_N$.
It is contained in the window-survival event.
The initial conditional law has the bounds (SPT.3).
At each later step while its entering state survives, (SPT.2) bounds the
next moment-failure probability by $M_{8,\rm box}/H_N$ and the
next low-alive probability by $r_N$.
Extinction is already a low-alive failure. Taking a union over the initial
array and these updates, stopped at the first extinction, gives
$$
\mathbb P_k(G^c)\le(m+1)
                    [\overline M/H_N+c_{\rm s}r_N].
\tag{SPT.12}
$$
No later hypothetical gas step from an all-dead array is used here.

On $G$, the population forecast begins with alive mass at least $m_f$.
Its subsequent outputs have alive mass at least $p>m_f$ by the uniform
large-box tail bound. The strict marked-source moment coefficient in
record 22 is at most $\lambda_8<1$ when its dead mass is at most
$\epsilon_*$. Since $H_N\ge H_8$, its moments remain at most $H_N$.
Thus both population input laws in every localized modulus comparison
satisfy (SPT.6).

Set
$e_j=\mathbb E_k[1_G\mathsf d_{\rm m}
                     (\widehat\mu_{k+j}^N,\mu_j)]$.
Drop $1_G$ to the past-measurable local-moment/alive event before applying
the actual conditional estimate (SPT.5). The triangle inequality and
concavity therefore give
$$
e_{j+1}\le A_N^{\rm loc}+(D_N-1)e_j^\alpha,\qquad e_0=0.
$$
Exactly the induction in record 21 gives $e_m\le V_N$ (and $e_0=0$
when $m=0$). Add the outside-window probability only once, after that
iteration. Applying (SPT.4) to the resulting endpoint error yields
$$
\mathbb E[\mathsf d_{\rm m}(\widehat\mu_n^N,\mu_m)
                       \mid\tau_N>n]
\le T_N\{V_N+(b_N+1)[\overline M/H_N+c_{\rm s}r_N]\}.
$$
This step includes the survival tilt of the starting array at time $k$.
The global population bound (SPT.11) is uniform in that starting array,
so under the same conditional path law it contributes at most
$C_{\rm pop}r_*^m$ without a moment or tilting correction.
This is at most
$C_{\rm pop}r_*^{n-1}+C_{\rm pop}r_*^{b_N}$.
The transport triangle inequality proves (SPT.10).

The proof in record 21 already shows $V_N\to0$ with the displayed
Hölder and moment exponents. Also
$(b_N+1)\overline M/H_N\to0$,
$(b_N+1)r_N\to0$, and $r_*^{b_N}\to0$.
Finally $e_N=\epsilon_{\rm box}^N$ decays exponentially, so
$b_Ne_N\to0$ and $T_N=(1-e_N)^{-b_N}\to1$.
This proves $\varepsilon_N^{\rm s}\to0$.
Neither a survival probability accumulated over the entire past nor a
union bound over that entire past appears in the floor.
:::

(sec-spt-alive)=
## 5. Optimal current-alive empirical and sampled observations

:::{prf:corollary} Surviving alive Wasserstein relaxation
:label: cor-spt-alive-w2

Let $\pi_L^A=(\pi_L)_A/\pi_L\{a=1\}$ and let $G_{\rm ph}$ be a
positive physical phase matrix. Define the actual alive phase diameter
bound
$$
\mathcal D_G^2=4\lambda_{\max}(G_{\rm ph})(dL^2+V^2).
$$
For every $n\ge1$ and $N\ge2$,
$$
\boxed{\quad
\mathbb E[W_{2,G_{\rm ph}}(\widehat\alpha_n^N,\pi_L^A)^2
                                      \mid\tau_N>n]
\le\mathcal D_G^2
       \min\{1,2u_{N,n}/m_f+c_{\rm s}r_N\}.
\quad}
\tag{SPT.13}
$$
The law of the random alive empirical probability conditional on
$\tau_N>n$, equipped with $W_{2,G_{\rm ph}}$ as its underlying distance,
has the same squared Wasserstein bound to $\delta_{\pi_L^A}$.
If one first samples a surviving swarm and then samples uniformly among
its current alive slots, the averaged sampled phase law has the same
bound to $\pi_L^A$.

If instead one samples uniformly among all stored slots and conditions
that slot to be currently alive in a surviving swarm, its normalized law
also converges, with the right side of (SPT.13) multiplied by at most
$m_f^{-1}$. These two sampling conventions are stated separately because
their swarm weights differ.
:::

:::{prf:proof}
For two marked probabilities with alive masses at least $m_f$, take a
coupling optimal for $\mathsf d_{\rm m}$ and restrict its common
alive--alive part. Divide by the larger alive mass; this is a
subcoupling of the two normalized alive laws. Its physical bounded cost
is at most $\mathsf d_{\rm m}/m_f$.
Its missing mass is at most the cross-mark mass divided by $m_f$:
if those two cross masses are $r_1,r_2$, the missing normalized mass
is $\max(r_1,r_2)/\max(m_A,m_A')$.
Cross-mark mass has cost one, so completing that subcoupling gives
$$
\mathsf d(\alpha_\mu,\alpha_{\mu'})
\le2\mathsf d_{\rm m}(\mu,\mu')/m_f .
$$
The target alive mass is at least $p>m_f$, since $\pi_L$ is an output.
On the event $A_N/N\ge m_f$ apply this inequality with
$\mu=\widehat\mu_n^N$; outside it use $\mathsf d\le1$.
(SPT.3) and (SPT.10) give conditional expected alive bounded distance at
most $\min\{1,2u_{N,n}/m_f+c_{\rm s}r_N\}$.

Every current alive physical point is in
$D_L\times\overline B_V$. Its Euclidean phase diameter squared is at
most $4(dL^2+V^2)>1$. For two points in that actual alive set,
$|z-z'|^2\le4(dL^2+V^2)\min\{1,|z-z'|\}$.
Multiplying by $\lambda_{\max}(G_{\rm ph})$ and optimizing the coupling
proves (SPT.13).
The only coupling to a Dirac law of probability measures pairs each
random alive empirical probability with $\pi_L^A$, so its squared cost
is exactly the conditional expectation in (SPT.13).
Integrating almost-optimal couplings gives the same bound for the
swarm-first uniform-alive sampled law.

For the all-slots-first convention, its swarm mixture has density
$\widehat m_A/\mathbb E[\widehat m_A\mid\tau_N>n]$ relative to the
swarm-first conditional law.
Given any previous surviving array, the expected next alive fraction is
at least $p$ by the Bernoulli domination. Extinction has zero alive
fraction, so division by the next survival probability can only increase
that expectation. Hence
$\mathbb E[\widehat m_A\mid\tau_N>n]\ge p>m_f$.
The mixture density is at most $m_f^{-1}$, proving the last bound.
:::

(sec-spt-scope)=
## 6. The target and its limits

The theorem concerns the actual finite chain conditioned on survival
through its current observation time. Its target is the current-alive
restriction of the proved stationary marked revival population law.
It supplies an exponential population term and an explicit vanishing
particle floor uniformly over that time. The physical alive Wasserstein
distance rate is $-\log r_*/(2h)$, with the particle floor from (SPT.9).

The configured alive domain is a fixed finite box, as in record 22.
Stored dead coordinates remain unbounded Gaussian-tailed coordinates;
they are included in the raw empirical comparison and moment localization.
Their donor replacement, rather than a compact-support approximation,
supplies the uniform first-output moment.

The proof does not establish an exact finite-array QSD or an optimal
rate for that QSD. It does not identify $\pi_L^A$ with such a law.
It does establish the stated survivor-conditioned observable estimates
for the original marked update on the explicitly computed large-box,
small-positive-viscosity and small-positive-fitness-power interval.
