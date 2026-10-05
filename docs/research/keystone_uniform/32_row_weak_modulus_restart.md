# Source-box row continuity and the surviving restart

This proof supplies the population weak-modulus half of the actual
row-normalized finite-particle transfer. Its entering marked laws may be
atomic and may have arbitrary retained dead coordinates. It uses actual
alive donor sources and full Gaussian tails, without a copied-mass
floor for a finite empirical provider.

(sec-rwm-register)=
## 1. Uniform exponential budgets after preparation

:::{prf:definition} Source-box row continuity register
:label: def-rwm-register

Use the actual harmonic preparation, original frozen-slot component
velocities, both row-normalized kicks, OU innovation, stored cap and
terminal box of {prf:ref}`def-rpf-record`. For the local continuity
results only require $0\le t\nu\le1$, positive alive input and
consistent entering marks. Set $R_D=\sqrt dL$, $\sigma=\sigma_J>0$,
$V_c=(1+2|\alpha_{\rm col}|)V$, and retain $a_x,b,c,t,q,s>0$.
The prepared law $\lambda$ and its own joint OU-stage law
$\Lambda_\lambda$ satisfy the exact identities
$$
X=u+I\sigma Z,\quad |u|\le R_D,\quad |v|\le V_c,
\qquad U=(1-t\nu)v+t\nu b_\lambda(X),
$$
$$
Y=a_xu+bU+a_xI\sigma Z+tq\xi,\qquad
W=-ctu+cU-ctI\sigma Z+q\xi,
\tag{RWM.1}
$$
where $Z,\xi$ are independent standard Gaussian vectors conditional
on the frozen source/component plan, and $I\in\{0,1\}$.
$U$ can depend on $X$ but is bounded by $V_c$.
Write
$$
\Sigma=a_x^2\sigma^2+t^2q^2,\quad
\Sigma_w=c^2t^2\sigma^2+q^2,\quad
B_y=a_xR_D+bV_c,\quad B_w=ctR_D+cV_c,
$$
$$
\chi=\min\{1,(8\sigma^2)^{-1},(8\Sigma)^{-1},
                                      (8\Sigma_w)^{-1}\}>0,
$$
$$
M=2^{d/2}\exp\{2\chi\max(R_D^2,V_c^2,B_y^2,B_w^2)\}\ge1,
\qquad H=\frac{\log M}{\chi},\quad E=\max\{1,\sqrt{8H}\}.
\tag{RWM.2}
$$
Every coordinate pair $\lambda=(X,v)$ or
$\Lambda_\lambda=(Y,W)$ consequently has the two separate bounds
$$
\int e^{\chi|x|^2}\,d\eta\le M,
\qquad \int e^{\chi|v|^2}\,d\eta\le M.
\tag{RWM.3}
$$
These statements hold for arbitrary consistent entering marked laws,
including empirical laws, without a moment hypothesis on dead positions.
The entering prepared and stage laws in (RWM.3) are population laws,
not a pathwise assertion about a random finite empirical array.
:::

:::{prf:proof}
Every persistent source is alive, every accepted alive source is an
alive donor, and every dead root mandatorily revives from an alive donor.
Thus all $u$ lie in $D_L$. Actual original-slot collisions give the
stated $V_c$ bound. The Gaussian row mean of bounded prepared velocities
is convex, hence $|U|\le V_c$. Both identities (RWM.1) follow from
the actual first kick, two drifts and OU update. The second kick does
not change $Y$.

For $B+G$ with $|B|\le B_0$ and centered Gaussian $G$ of variance
at most $v_0I_d$, the pointwise inequality
$|B+G|^2\le2B_0^2+2|G|^2$ gives
$$
\mathbb E e^{\chi|B+G|^2}
\le e^{2\chi B_0^2}(1-4\chi v_0)^{-d/2}
\le2^{d/2}e^{2\chi B_0^2}
$$
when $\chi v_0\le1/8$. Dependence of $B$ on the Gaussian does
not affect this bound. Apply it separately to $X,Y,W$ and use the
bounded prepared velocity. This proves (RWM.3). Jensen also gives
the second-moment bound $H$ for each coordinate. The $W_2$ distance
between any two such phase laws is at most $\sqrt{8H}$.
:::

(sec-rwm-row)=
## 2. A global transport modulus for a true normalized Gaussian row

:::{prf:lemma} Exponential-tail row continuity without a global degree floor
:label: lem-rwm-row-continuity

Let two joint providers $\eta,\eta'$ satisfy (RWM.3), and let
$e=W_2(\eta,\eta')$. Couple their coordinates optimally as
$(X,V),(X',V')$. For the actual row means
$$
b_\eta(x)=\frac{\int e^{-|x-y|^2/(2\rho^2)}w\,d\eta(y,w)}
                  {\int e^{-|x-y|^2/(2\rho^2)}\,d\eta(y,w)},
$$
there are the following explicit constants:
$$
R_0=\sqrt{\log(2M)/\chi},\quad
B^2=\chi^{-1}[\log(2M)+R_0^2/\rho^2+\rho^{-2}],
\quad C=4/\rho^2,
$$
$$
K_g=64(1+2\ell_\rho\sqrt H)^2e^{4R_0^2/\rho^2},
\qquad
K_t=2B^2\sqrt{64M[1+4M/(e_{\rm E}^2\chi^2)]},
$$
$$
K=[K_g+K_t+8B^2(1+H)]^{1/2},\qquad
\gamma=\frac{\chi}{4(C+\chi)}\in(0,1/4),
\tag{RWM.4}
$$
where $e_{\rm E}=\exp(1)$ and
$\ell_\rho=e^{-1/2}/\rho$. Then
$$
\left(\mathbb E|b_\eta(X)-b_{\eta'}(X')|^2\right)^{1/2}
\le K\min\{1,e\}^{\gamma}.
\tag{RWM.5}
$$
The joint velocities can be uncapped. No copied Gaussian mass or
pathwise positive empirical degree floor is assumed.
:::

:::{prf:proof}
Exponential Markov gives $\eta\{|X|\le R_0\}\ge1/2$.
Therefore every denominator obeys its actual lower bound
$$
a_\eta(x)\ge\tfrac12 e^{-(|x|+R_0)^2/(2\rho^2)}.
$$
The kernel posterior obeys
$\mathbb E_{\pi_x}e^{\chi|V|^2}\le M/a_\eta(x)$.
Jensen and $(r+R_0)^2\le2r^2+2R_0^2$ give
$|b_\eta(x)|\le B(1+|x|)$, uniformly in the provider.
These estimates require neither independence between $X,V$ nor a
bounded velocity numerator.

On $|X|,|X'|\le R$, both denominators are at least
$a_R=\tfrac12e^{-(R+R_0)^2/(2\rho^2)}$.
The exact normalized-numerator comparison
{prf:ref}`lem-rpf-local-row-transport` gives
$$
|b_\eta(X)-b_{\eta'}(X')|
\le(1+2\ell_\rho\sqrt H)a_R^{-2}(|X-X'|+e).
$$
Its squared expected contribution is at most
$K_g e^{CR^2}e^2$, since
$(R+R_0)^2\le2R^2+2R_0^2$ and
$\mathbb E(|X-X'|+e)^2\le4e^2$.

The complementary event has probability at most
$2M e^{-\chi R^2}$. Also
$\mathbb E|X|^4\le4M/(e_{\rm E}^2\chi^2)$, by maximizing
$u^2e^{-u}$. Cauchy--Schwarz and the linear posterior bound
therefore give a squared tail contribution at most
$K_t e^{-\chi R^2/2}$; the factor $32$ in the squared polynomial
bound uses $(1+r)^4\le8(1+r^4)$ for each query.

For $0<e\le1$ choose
$R^2=\log(1/e)/(C+\chi)$. The two contributions are bounded by
$(K_g+K_t)e^{2\gamma}$, since
$2-C/(C+\chi)\ge1>2\gamma$ and
$\chi/[2(C+\chi)]=2\gamma$.
For $e\ge1$ the linear posterior and second moments instead give
$\mathbb E|b_\eta(X)-b_{\eta'}(X')|^2\le8B^2(1+H)$.
For $e=0$ the coupled providers and queries coincide. Taking square
roots proves (RWM.5).
:::

(sec-rwm-kinetic)=
## 3. Both kicks, cap and terminal mark

:::{prf:theorem} Actual row kinetic and marked population weak moduli
:label: thm-rwm-population-modulus

For two actual prepared source-box laws $\lambda,\lambda'$ set
$$
A_1=(a_x+ct+b+c)E+(b+c)t\nu K,
\quad A_2=(1+t)E+t\nu K,
\quad T_s=\max\{1,(s\sqrt{2\pi})^{-1}\},
$$
$$
C_{\rm kin}=(A_2+T_sE)\max\{1,A_1\}^{\gamma}.
\tag{RWM.6}
$$
Their actual row kinetic maps, including terminal marking, satisfy
$$
\mathsf d_{\rm m}(K^{\rm row}\lambda,K^{\rm row}\lambda')
\le\min\{1,C_{\rm kin}\min\{1,W_2(\lambda,\lambda')\}^{\gamma^2}\}.
\tag{RWM.7}
$$
Let $C_p^{\rm m}$ be the completely proved preparation-prefix
modulus from {prf:ref}`lem-spt-marked-consistency`, evaluated at an
alive floor $m_f>0$ and the fixed box's alive reward bounds. Put
$$
Z_L=R_D+\sigma g_8+V_c,
\quad U_p=(1+16Z_L^4)^{1/4}(C_p^{\rm m})^{1/8},
\quad \alpha_{\rm row}=\gamma^2/32>0,
$$
$$
C_{\rm mod}^{\rm row}=C_{\rm kin}\max\{1,U_p\}^{\gamma^2}.
\tag{RWM.8}
$$
For every two consistent capped marked inputs with alive masses at
least $m_f$, including atomic inputs with unrestricted dead positions,
$$
\mathsf d_{\rm m}(\mathcal F_L^{\rm row}\mu,
                         \mathcal F_L^{\rm row}\mu')
\le\min\{1,C_{\rm mod}^{\rm row}
                   \mathsf d_{\rm m}(\mu,\mu')^{\alpha_{\rm row}}\}.
\tag{RWM.9}
$$
This modulus uses the actual first and correlated second row providers,
and each own normalizer. It requires no population attraction premise.
:::

:::{prf:proof}
Couple the prepared inputs optimally, with $e=W_2(\lambda,\lambda')$,
and use the same independent OU Gaussian in their individual updates.
By (RWM.5) the first convex averages obey
$$
\|U-U'\|_2\le e+t\nu K\min\{1,e\}^{\gamma}.
$$
The actual correlated stage identities give
$\|Y-Y'\|_2\le a_xe+b\|U-U'\|_2$ and
$\|W-W'\|_2\le cte+c\|U-U'\|_2$.
Their joint coupling norm $e_1$ is consequently at most
$A_1\min\{1,e\}^{\gamma}$, using $e\le E$ and
$e\le E\min\{1,e\}^{\gamma}$. Each stage provider satisfies
(RWM.3), so its optimal $W_2$ distance is no larger than this
coupling norm and is also at most $E$. The actual prescribed stage
coupling itself has $e_1\le\sqrt{8H}\le E$, by the same two marginal
second-moment bounds, independently of whether it is optimal.

Apply the proof of (RWM.5) to the actual coupled stage queries.
It only needs a coupling whose joint norm bounds the optimal distance;
the same good-event proof thus gives $K\min\{1,e_1\}^{\gamma}$.
The actual second row kick is
$$
z=(1-t\nu)W-tY+t\nu b_{\Lambda_\lambda}(Y).
$$
Hence $\|z-z'\|_2\le A_2\min\{1,e_1\}^{\gamma}$.
The Euclidean stored cap is a common nonexpansive pushforward.
Conditional on these pre-final states, maximally couple the final
Gaussian positions $Y+s\zeta,Y'+s\zeta'$. Their mismatch probability
is at most $|Y-Y'|/(s\sqrt{2\pi})$. At matched positions their
terminal marks also match, and the marked cost is at most the velocity
difference. At a Gaussian mismatch its cost is at most one.
Therefore the output marked distance is at most
$\|z-z'\|_2+T_s\|Y-Y'\|_2$.
The inequalities
$e_1\le E\min\{1,e_1\}^{\gamma}$ and
$\min\{1,A_1u^\gamma\}^{\gamma}\le
\max\{1,A_1\}^{\gamma}u^{\gamma^2}$ prove (RWM.7).
Every individual Gaussian transition marginal is preserved by these
couplings; the second stage is never replaced by a product law.

For (RWM.9), the unchanged source-first preparation proof gives
$\mathsf d(J\mu,J\mu')\le C_p^{\rm m}
\mathsf d_{\rm m}(\mu,\mu')^{1/4}$. Actual sources have the
uniform prepared phase eighth moment $Z_L^8$ from
{prf:ref}`lem-sbst-source-moments`. The proved deterministic
weak-to-$W_4$ upgrade gives
$W_2(J\mu,J\mu')\le W_4(J\mu,J\mu')\le
U_p\mathsf d_{\rm m}(\mu,\mu')^{1/32}$.
Substitute this in (RWM.7). Reward bounds here are solely the
post-box alive bounds in a finite weak estimate. In particular the
raw same-potential reward remains $-|x|^2/2$.
:::

## 4. Retained transfer endpoint

The conditional finite-row comparison is proved separately in research31.
The positive marked row population attraction theorem used at the temporal
endpoint is {prf:ref}`thm-rpf-marked-population`. Neither the present
continuity modulus nor that small-viscosity population result certifies
the prescribed $\nu=.3,L=2$ nonlinear law.
