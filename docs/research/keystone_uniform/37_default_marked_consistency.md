# Default harmonic count consistency and marked weak continuity

(sec-dmc-register)=
## 1. The fixed reference kinetics and explicit positive fitness interval

:::{prf:definition} Default marked comparison register
:label: def-dmc-register

Use the actual continuous-coordinate harmonic dense count gas at
$$
d=3,\quad h=.04,\quad \nu=.3,\quad \gamma=b_O=\rho=1,
\quad V=2,\quad L=2,\quad
\sigma_x=\sigma_J=.1,\quad \alpha_{\rm col}=.5.
\tag{DMC.1}
$$
Both kinetic kicks use the actual count-normalized Gaussian kernel
$K_\rho(x,y)=\exp[-|x-y|^2/(2\rho^2)]$, with denominator $N$ and
zero self contribution. The force and raw reward remain
$F(x)=-x$, $R(x,v)=-|x|^2/2$. The terminal alive region is
$D=[-2,2]^3$. Retain current-frame measured fitness, its actual sampled
alive-only standardizers, self-excluded eligible companion laws, simultaneous
frozen position sources, mandatory revival, full component Haar collisions
on original frozen slot velocities, copied-recipient Gaussian jitter,
independent original OU and final-position Gaussians, and the native cap
$C_V(z)=Vz/(V+|z|)$. Donor velocities are not copied. No historical,
curl, elite or geometry-feedback branch is enabled in this restriction.

All role widths, positive fitness floors and amplitudes, standardizer
floors, and gate parameters are fixed independently of $N$. Write the
logistic factors as $f_b(z)=\eta_b+A_b/(1+e^{-z})$, $b=r,s$, with
$\eta_b,A_b>0$, and the fitness as $F=f_r^{p_r}f_s^{p_s}$.
Fix $\bar p_r,\bar p_s>0$ and set $p_b=\theta\bar p_b$. Use the
configured gate $[(F_{\rm donor}-F_{\rm root})/
\{s_c(F_{\rm root}+\epsilon_c)\}]_0^1$, with $s_c,\epsilon_c>0$.
Mandatory dead-root gates are identically one.

For the configured bounded comparison features define
$$
\ell_f=\max\{1,\sqrt{\lambda_{\rm alg}}\},\qquad
D_*=2\sqrt{(R_x^{\rm feat})^2+
                         \lambda_{\rm alg}(R_v^{\rm feat})^2},
$$
$$
S_*=\sqrt{D_*^2+\delta_D^2},\qquad
\kappa_b=e^{-D_*^2/(2\epsilon_b^2)},\qquad
w_b'=D_*\ell_f/\epsilon_b^2\quad(b=D,C).
\tag{DMC.2}
$$
For the existing uniform role tag set $\kappa_b=1,w_b'=0$.
The feature radii bound the actual comparison maps, not physical
dead positions. The smoothing $\delta_D$ and standardizer floors
$\sigma_r,\sigma_s$ retain their positive configured values.

The following constants give a nonempty primitive exponent interval:
$$
\begin{split}
M&=\sum_{b=r,s}\bar p_b
              \max\{|\log\eta_b|,|\log(\eta_b+A_b)|\},\\
\Delta&=\sum_{b=r,s}\bar p_b\log[(\eta_b+A_b)/\eta_b],\\
a_0&=\frac{e^M\Delta}{s_c(e^{-M}+\epsilon_c)},\qquad
\theta_f=\min\{1,\kappa_C/(8a_0)\}>0.
\end{split}
\tag{DMC.3}
$$
Require $0<\theta\le\theta_f$. This is a restriction on the
configured positive fitness exponents. It does not assert that the
reference choice $p_r=p_s=1$ satisfies this test.

Set $t=h/2$, $c=e^{-h}$, $b=t(1+c)$, $a_x=1-tb$,
$q^2=(1-c^2)/2$, $s^2=\sigma_x^2h$, $V_c=4$ and
$\ell_\rho=e^{-1/2}/\rho$. Retain the exact positive safe-return
floor $a_{\rm ret}$ of (DSA.3), evaluated at these reference
parameters, and put
$$
m_f=a_{\rm ret}/2,\qquad
e_N=(1-a_{\rm ret})^N,\qquad
r_N=e^{-(1-\log2)a_{\rm ret}N/2},\qquad
c_{\rm ret}=[1-(1-a_{\rm ret})^2]^{-1}.
\tag{DMC.4}
$$
The deterministic population map $\mathcal F$ is the original rooted
preparation followed by both count kicks and terminal classification.
Its input class consists of capped, terminally consistent marked laws
with alive mass at least $m_f$. Retained dead positions may have no
finite moment. No population contraction, invariant phase or small-dead
class is assumed.

For $z=(x,v)$ use the original phase norm and marked ground cost
$$
c_{\rm m}((z,a),(z',a'))=
\min\{1,|z-z'|+\mathbf1_{\{a\ne a'\}}\},
$$
and write $\mathsf d_{\rm m}$ for its transport distance. For
unmarked prepared laws write $\mathsf d$ for transport with
$\min\{1,|z-z'|\}$. Empirical laws are not compared in total
variation to a continuous population law.
:::

:::{prf:lemma} Alive acceptance, mandatory leaves and source-box moments
:label: lem-dmc-source-forest

Let
$$
F_*=\eta_r^{p_r}\eta_s^{p_s},\qquad
F^*=(\eta_r+A_r)^{p_r}(\eta_s+A_s)^{p_s},\qquad
a_* =\min\left\{1,\frac{F^*-F_*}{s_c(F_*+\epsilon_c)}\right\}.
$$
Then every alive-root outgoing acceptance probability is at most
$$
a_*\le\theta a_0,\qquad c_*=a_*/\kappa_C\le1/8<1.
\tag{DMC.5}
$$
For an actual input with $M\ge m_fN$ alive slots, each alive donor's
expected incoming accepted-alive column is at most $c_*$. Its mandatory
dead column is at most
$ (N-M)/(\kappa_CM)\le(1-m_f)/(\kappa_Cm_f)$.
Thus a valid total incoming-column bound is
$$
k_f=c_*+\frac{1-m_f}{\kappa_Cm_f}.
\tag{DMC.6}
$$
The mandatory part need not be small. Dead vertices have no incoming
accepted edges and are leaves of the source-oriented component forest.
The accepted alive-arm subcritical test is discharged by (DMC.5);
the ordered-component moment estimates below retain the actual full
mandatory-dead leaves, not a weak gate on those leaves.

Write
$$
g_8=[d(d+2)(d+4)(d+6)]^{1/8},\qquad
X_L=\sqrt dL+\sigma_Jg_8,\qquad Z_L=X_L+V_c.
$$
For any consistent positive-alive input array $S$, let $\eta_N$ be
its actual prepared empirical phase law and $\eta=J(L_N(S))$ the
actual population preparation at that same atomic input. Then
$$
\mathbb E[M_8(\eta_N)\mid S]\vee M_8(\eta)\le X_L^8,
\qquad
\mathbb E[M_{8,z}(\eta_N)\mid S]\vee M_{8,z}(\eta)\le Z_L^8.
\tag{DMC.7}
$$
The deterministic prepared bounds also hold for every consistent
population input with positive alive mass and unrestricted dead positions.
:::

:::{prf:proof}
For $0<\theta\le1$, all logarithmic factor sums lie in $[-M,M]$.
Integrate the derivative of $e^{\theta u}$ between the sums defining
$F_*$ and $F^*$ to get $F^*-F_*\le\theta e^M\Delta$ and
$F_*\ge e^{-M}$. The original clipped gate gives (DMC.5).
An alive donor has $M-1$ possible alive cloners, each with donor
probability at most $1/[\kappa_C(M-1)]$ and acceptance at most
$a_*$. Summing proves $c_*$. If $M=1$, there is no such cloner;
the actual singleton convention has zero alive cloning. Every dead
root chooses an alive donor with probability at most $1/(\kappa_CM)$
and has mandatory gate one. Summing proves (DMC.6).

Place dead vertices before all alive vertices and order alive vertices
by frozen measured fitness, breaking ties by a deterministic label.
Every accepted alive edge strictly increases fitness; every mandatory
edge goes from a dead vertex to an alive vertex. Thus the entire
component graph is the actual ordered forest. In particular mandatory
dead vertices add leaves rather than repeated dead offspring. At the
population limit their alive incoming intensity is at most $c_*<1$;
even the more conservative $2c_*\le1/4$ test is satisfied. No
Poisson independence of an actual finite graph is inferred from this
intensity bound: its finite component estimates use the ordered-path
argument of {prf:ref}`lem-chaos-component-moments`.

Every prepared row persists at an alive position or copies/revives
from an alive position. Its source therefore lies in $D$, including
when its retained entering dead position is arbitrarily far away.
Condition on all sources and component Haar marks. Its actual position
is $X=x_{\rm src}+I\sigma_JZ$ with $|x_{\rm src}|\le\sqrt dL$,
$I\in\{0,1\}$, and an independent standard Gaussian $Z$.
Minkowski gives its conditional positional eighth norm at most $X_L$.
The original-slot component formula gives $|v^{\rm p}|\le V_c$.
A second Minkowski inequality gives phase eighth norm at most $Z_L$.
Average over rows and the complete source plan. The tagged population
calculation is the same actual rooted source calculation. This proves
(DMC.7) without a retained dead-position moment assumption.
:::

(sec-dmc-preparation)=
## 2. Full alive-floor preparation coefficients

:::{prf:definition} Explicit preparation consistency and continuity constants
:label: def-dmc-preparation-budget

Use the actual factor derivatives
$$
H_b=(\eta_{b'}+A_{b'})^{p_{b'}}\frac{A_bp_b}{4}
\max\{\eta_b^{p_b-1},(\eta_b+A_b)^{p_b-1}\},
\quad \{b,b'\}=\{r,s\},
$$
$$
L_a=\max\left\{\frac1{s_c(F_*+\epsilon_c)},
\frac{F^*+\epsilon_c}{s_c(F_*+\epsilon_c)^2}\right\},\qquad
C=\frac2{\kappa_Cm_f},\qquad D_D=\frac2{\kappa_Dm_f}.
\tag{DMC.8}
$$
For $p=1,2,3$ set the explicit convergent component-moment formula
$$
M_p(C)=e^{(2^p-1)C}
\sum_{k\ge0}[(k+1)^p-k^p]\frac{C^k}{k!}.
$$
Define
$$
\begin{aligned}
L_q&=S_*/(m_f\sigma_s)+3S_*^3/(2m_f\sigma_s^3),\\
L_0&=H_sL_q,\qquad B=C+2L_aL_0,\\
A_D&=9M_2(2C)[(1+B)^2+B]+\max\{1,4B^2\},\\
A_{\rm p}&=2[A_D+10M_2(C)+1],\\
A_T&=(m_f^{-1}+D_D^2)(2S_*^2/\sigma_s^2+5S_*^6/\sigma_s^6),\\
L_T&=2L_aH_s,\qquad A_{\rm e}=1+C+D_D,\\
N_0&=\left\lceil\max\{(8A_{\rm e})^{6/5},2/m_f\}\right\rceil,\\
B_{\rm p}&=3M_1(2C)L_T\sqrt{A_T}+4L_T^2A_T
 +64A_{\rm e}^2+16M_3(C)+\sqrt{N_0},\\
G_{\rm p}&=A_{\rm p}+4B_{\rm p}^2.
\end{aligned}
\tag{DMC.9}
$$
All constants are evaluated at the original configured fitness and
standardizers. In particular the sampled diversity standardizer is
not replaced in the executed update.

Raw reward statistics use alive roots only, so their local physical
bounds are $R_b=dL^2/2=6$ and $L_R=\sqrt dL=\sqrt{12}$.
These are bounds on the unchanged quadratic reward on its actual
provider support. Define
$$
\begin{aligned}
b_{\rm m}&=1+(3+4w_D')/(\kappa_Dm_f),\\
K_{\rm a}&=2(1+b_{\rm m})/m_f,\\
m_r&=L_R+2R_bK_{\rm a},\qquad
m_s=2\ell_f+S_*K_{\rm a},\\
Q_r&=(L_R+m_r)/\sigma_r+4R_b^2m_r/\sigma_r^3,\\
Q_s&=(2\ell_f+m_s)/\sigma_s+2S_*^2m_s/\sigma_s^3,\\
E_F&=H_rQ_r+H_sQ_s,\\
D_\beta&=\frac{2w_C'}{\kappa_Cm_f}
 +\frac{2w_C'+1}{(\kappa_Cm_f)^2}
 +\frac{2L_aE_F}{\kappa_Cm_f},\\
C_{\rm p}^{\rm m}
&=b_{\rm m}+8[D_\beta+2b_{\rm m}/(\kappa_Cm_f)]
 +2e^{2C}+1+k_v,\qquad k_v=1+2|\alpha_{\rm col}|=2.
\end{aligned}
\tag{DMC.10}
$$
:::

:::{prf:lemma} Preparation comparison without a small-dead or input-moment premise
:label: lem-dmc-preparation

For every capped consistent array $S$ with alive fraction at least
$m_f$, and every bounded measurable phase test $|\varphi|\le1$,
$$
\mathbb E[|(\eta_N-J(L_N(S)))\varphi|^2\mid S]\le G_{\rm p}/N.
\tag{DMC.11}
$$
For any two consistent capped marked input laws with alive masses
at least $m_f$,
$$
\mathsf d(J(\mu),J(\mu'))
\le\min\{1,C_{\rm p}^{\rm m}
                          \mathsf d_{\rm m}(\mu,\mu')^{1/4}\}.
\tag{DMC.12}
$$
Both assertions retain mandatory revival and all measured fitness
normalizers. They require neither an input dead-position moment nor
a small actual dead fraction.
:::

:::{prf:proof}
For (DMC.11), freeze the array. Reward values and their alive statistics
are fixed. Replacing one measurement changes alive diversity mean,
second moment and variance by at most
$S_*/(m_fN),S_*^2/(m_fN),3S_*^2/(m_fN)$.
The derivative of its regularized reciprocal standard deviation gives
$L_q/N$ for every unchanged measured row. The actual bounded fitness
factor and clipped gate give $L_0/N$ and the changed-edge coefficient
$B/N$ in (DMC.9). Mandatory dead gates remain one; their unchanged
frozen donor laws are not charged a weak-fitness acceptance.

Expose exceptional row outcomes before the remaining common forest.
For $N\ge2B$ its specified-edge densities after that exposure are
at most $2C/N$. Accepted common alive edges respect the same frozen
fitness order; mandatory edges retain dead vertices as leaves.
The ordered-forest moment theorem, applied before conditioning on
any unusually large component, therefore bounds a measurement
replacement's squared influence by $A_D$. Donor/gate replacement,
an addressed Haar mark and one local jitter contribute
$9M_2(C),M_2(C),1$ respectively. The conditional independent-block
variance argument gives $A_{\rm p}/N$.

For the mean bias use the population normalizers solely as a coupling
device inside the proof of
{prf:ref}`thm-chaos-canonical-quantitative-bias`, stopped before any
kinetic stage. Measurement self exclusion has full-variation error at
most $D_D/N$. The conditional sampled normalization errors have
second-moment coefficient $A_T/N$, so integrating their gate effect
costs at most
$2[3M_1(2C)L_T\sqrt{A_T}/\sqrt N+4L_T^2A_T/N]$ for a unit test.
The finite-label exploration retains failed proposals, their auxiliary
targets and the used outgoing edge of an incoming child. At
$K=\lfloor N^{1/6}\rfloor$ its label, self-exclusion and Poisson
comparison costs at most $64A_{\rm e}^2K^3/N$, and the two full
component tails cost $2M_3(C)/K^3$. With $N\ge N_0$ every used
alive pool has at least two slots. Below that threshold the elementary
unit-test difference bound is covered by the $\sqrt{N_0}$ term in
$B_{\rm p}$. The resulting bias is at most $2B_{\rm p}/\sqrt N$.
Variance plus squared bias proves (DMC.11). This is a preparation-only
use of the cited influence/bias proofs; their later no-viscosity
restriction is not invoked for the present dense kinetic stages.

For (DMC.12), put $\delta=\mathsf d_{\rm m}(\mu,\mu')$ and
$r=\sqrt\delta$. An optimal marked coupling has bad mass at most
$\sqrt\delta$ when marks differ or phase distance exceeds $r$.
Lift the actual weighted measurement laws to this coupling.
Their denominators are at least $\kappa_Dm_f$, so good measured-type
pairs have own and companion phase distances at most $r$, with
bad mass at most $b_{\rm m}\sqrt\delta$.

Restrict this coupling to alive recipients and divide by the larger
alive mass before completing residual marginals. The actual alive
mass difference is at most $\delta$. The conditional alive measured
coupling consequently has bad mass at most
$K_{\rm a}\sqrt\delta$. Alive reward and diversity mean differences
are bounded by $m_r\sqrt\delta,m_s\sqrt\delta$; bounded variance
subtraction and the positive standardizer floors give
$Q_r\sqrt\delta,Q_s\sqrt\delta$. The fitness difference at good
alive types is at most $E_F\sqrt\delta$.

The actual cloning denominator is at least $\kappa_Cm_f$.
Subtract its weighted alive numerator and denominator under the full
marked coupling. Their good-pair derivative and residual-mark costs
give $D_\beta\sqrt\delta$ for accepted density differences. For a
dead root the gate is identically one, so its normalization comparison
uses only these donor numerator/denominator terms. No raw dead reward
is needed. Incoming edges always target alive vertices, and the
conditional-root construction keeps each incoming child's used
outgoing edge. The common incoming/outgoing exploration discrepancy
per exposed vertex is bounded by
$4[D_\beta+2b_{\rm m}/(\kappa_Cm_f)]\sqrt\delta$.

Stop the comparison at $K=\lceil\delta^{-1/4}\rceil$ exposed
vertices. The discrepancy cost is at most the term with coefficient
$8$ in (DMC.10); both complete ordered-component tails cost at most
$2e^{2C}/K$. On matching full components use the same original-slot
velocities and Haar matrix, the coupled source, and the same fresh
jitter. Copied sources differ by at most $r$ and component velocity
readouts by at most $k_vr$. Summing these charges gives (DMC.12).
The case $\delta=0$ uses identical laws.

Every bounded statistic in this proof is evaluated on alive input
roots or on bounded comparison features. Raw harmonic reward is
therefore bounded by $R_b$ exactly where it is used. Neither the
bounded-test comparison nor the component exploration sees a
retained dead-position moment. The possibly large dead column in
(DMC.6) is retained by the alive-floor forest coefficient $C$.
:::

(sec-dmc-kinetic)=
## 3. Default count kinetics and terminal marks

:::{prf:definition} Source-box kinetic coefficients
:label: def-dmc-kinetic-budget

Set $\kappa=1/(16d)$, $a=1/(128d)=1/384$, $\alpha=1/32$,
and $J_d=(2+2\sqrt{2d})^{2d}$. Define
$$
\begin{aligned}
A_{{\rm p},L}&=2+\tfrac12J_d\sqrt{G_{\rm p}}+2Z_L^2,\\
U_{{\rm p},L}&=(1+16Z_L^4)^{1/4}A_{{\rm p},L}^{1/8},\\
A_1&=1+t+t\nu(2+4\ell_\rho V_c),\\
C_y&=1+bA_1,\qquad C_w=cA_1,\\
u_0&=(1+2t\nu)V_c+t,\quad
w_0=cu_0+qg_8,\quad y_0=1+bu_0+tqg_8,\\
k_0&=y_0+w_0,\qquad o_0=y_0+sg_8+V,\\
A_{\rm o}&=2+\tfrac12J_d+2k_0^2,\quad
U_{\rm o}=(1+16k_0^4)^{1/4}A_{\rm o}^{1/8},\\
B_2&=2+t+t\nu(2+4\ell_\rho w_0),\\
T_s&=\max\{1,(s\sqrt{2\pi})^{-1}\},\qquad
A_{\rm f}^{\rm m}=2+J_d+2o_0^2.
\end{aligned}
\tag{DMC.13}
$$
For $M\ge0$, put
$$
w(M)=c[(1+2t\nu)V_c+tM^{1/8}]+qg_8,
$$
$$
C_K(M)=C_y+C_w+tC_y+
       t\nu(2+4\ell_\rho w(M))(C_y+C_w).
\tag{DMC.14}
$$
Finally define the full primitive comparison coefficients
$$
\begin{split}
\mathcal C_{\rm cons}
&=[T_sB_2U_{\rm o}+A_{\rm f}^{\rm m}](1+X_L^8)^{9/32}
                    +T_sC_K(X_L^8)U_{{\rm p},L},\\
\mathcal C_{\rm mod}
&=T_sC_K(X_L^8)(1+16Z_L^4)^{1/4}
                              (C_{\rm p}^{\rm m})^{1/8}.
\end{split}
\tag{DMC.15}
$$
Their dependence on $m_f$ can be very large, but they are finite
explicit functions of fixed consumed parameters, independent of
$N$, elapsed time and entering dead positions.
:::

:::{prf:lemma} Dense default kinetic comparison with the actual joint OU provider
:label: lem-dmc-kinetics

For prepared laws of speed at most $V_c$, if the target positional
eighth moment is at most $M$, their actual population kinetic images
obey the marked transport estimate
$$
\mathsf d_{\rm m}(K\eta,K\eta')
\le T_s C_K(M)W_4(\eta,\eta').
\tag{DMC.16}
$$
For any fixed prepared empirical array $\eta_N$ of positional eighth
moment $M$, its actual finite two-count-kick marked empirical output
obeys
$$
\mathbb E[\mathsf d_{\rm m}(L_N(S^+),K\eta_N)]
\le[T_sB_2U_{\rm o}+A_{\rm f}^{\rm m}]
                         (1+M)^{9/32}N^{-a}.
\tag{DMC.17}
$$
These estimates require neither small $\nu$ nor a population
contraction. At (DMC.1), $t\nu=.006$; all coefficients are evaluated
at that value. Final terminal marks and unbounded innovations are retained.
:::

:::{prf:proof}
Under a product of any phase coupling, subtract the velocity and
Gaussian-kernel factors in the count integral. The first provider
stability is
$\|\nu C_\eta-\nu C_{\eta'}\|_4
\le\nu(2+4\ell_\rho V_c)W_4(\eta,\eta')$.
For arbitrary joint noisy stage laws the corresponding $L^2$ bound is
$\nu(2+4\ell_\rho\|w'\|_4)W_4(\Lambda,\Lambda')$.
The latter uses Hölder on the same joint phase coupling; landing
position and uncapped OU velocity need not be independent.

Share each own independent OU innovation across the entering
coupling. The first force/kick costs at most $A_1W_4$ in $L^4$;
the landing and OU-velocity differences are at most
$C_yW_4,C_wW_4$. The target stage fourth velocity norm is at most
$w(M)$. Apply the joint provider bound at the second count kick,
then the nonexpansive native cap. Position plus velocity cost is
at most $C_K(M)W_4$. The argument multiplies finite factors by
$\nu$; it never absorbs them into a small-viscosity contraction.

For fixed pre-final-noise means $y,y'$ and capped velocities $v,v'$, a
maximal coupling of the final Gaussians has mismatch probability at
most $|y-y'|/(s\sqrt{2\pi})$. On a match positions and terminal
marks agree; off a match the marked ground cost is at most one.
The resulting marked cost is at most
$T_s(|y-y'|+|v-v'|)$. This proves (DMC.16), including at the
actual box boundary.

For (DMC.17), freeze the prepared array. Its first finite count
field is exactly the population field at its empirical law, since
its added self term is $K(x_i,x_i)(v_i-v_i)/N=0$.
Conditionally independent own OU draws produce a joint empirical
stage $\Lambda_N$ with exact mean law $\Lambda$; each unit test
has variance at most $1/N$. Averaged phase eighth moments are at
most $k_0^8(1+M)$. The cell and moment estimates of
{prf:ref}`lem-vupt-cell` give
$$
\mathbb E W_4(\Lambda_N,\Lambda)
\le U_{\rm o}(1+M)^{5/32}N^{-a}.
$$
The actual second finite count field is again the empirical field
at $\Lambda_N$, with its exact zero self term. The joint-force,
cap and final Gaussian comparison just proved costs at most
$T_sB_2(1+M)^{1/8}W_4$. This gives the first contribution
in (DMC.17), while retaining every field/OU correlation.

Finally condition on the full OU array and second kick. The last
position Gaussians are independent; the capped velocities are fixed.
Use the same phase-cell partition with a separate cell for each
terminal mark, so its cell coefficient is $J_d$ rather than
$J_d/2$. Its conditional phase eighth moment is random, but its
expectation is at most $o_0^8(1+M)$. Conditional cell comparison
and Jensen therefore cost at most
$A_{\rm f}^{\rm m}(1+M)^{1/4}N^{-\kappa}$, which is bounded
by the corresponding term in (DMC.17). No empirical-row independence
before the OU draw, no independent second-stage graph, and no
row-degree floor has been used.
:::

(sec-dmc-interface)=
## 4. The complete default finite interface

:::{prf:theorem} Uniform marked one-update consistency and population weak modulus
:label: thm-dmc-default-interface

Under {prf:ref}`def-dmc-register`, for every consistent capped finite
input $S$ with alive fraction at least $m_f$,
$$
\mathbb E[\mathsf d_{\rm m}(L_N(S^+),\mathcal F(L_N(S)))\mid S]
\le A_N:=\min\{1,\mathcal C_{\rm cons}N^{-1/384}\}.
\tag{DMC.18}
$$
For any two consistent capped marked population inputs with alive
masses at least $m_f$,
$$
\mathsf d_{\rm m}(\mathcal F\mu,\mathcal F\mu')
\le\min\{1,\mathcal C_{\rm mod}
                       \mathsf d_{\rm m}(\mu,\mu')^{1/32}\}.
\tag{DMC.19}
$$
No retained dead-position moment or small-dead hypothesis occurs.
Each actual population output has alive mass at least $a_{\rm ret}$,
so the modulus remains applicable to its subsequent iterates.
:::

:::{prf:proof}
Apply (DMC.11) and (DMC.7) to the actual prepared law at the frozen
empirical input. The cell comparison and weak-to-$W_4$ upgrade give
$$
\mathbb E\mathsf d(\eta_N,\eta)\le A_{{\rm p},L}N^{-\kappa},
\qquad
\mathbb E W_4(\eta_N,\eta)\le U_{{\rm p},L}N^{-a}.
$$
Conditional on the actual prepared array, (DMC.17) bounds the full
finite marked kinetic error. Average its moment factor: concavity
and (DMC.7) replace it by $(1+X_L^8)^{9/32}$.
For the remaining population kinetic comparison between $\eta_N$
and the deterministic target $\eta$, its target moment is at most
$X_L^8$; (DMC.16) bounds it by
$T_sC_K(X_L^8)\mathbb EW_4(\eta_N,\eta)$.
The triangle inequality gives (DMC.18) with (DMC.15).

For two population inputs, (DMC.12) gives their preparation weak
distance. Both prepared phase eighth moments are at most $Z_L^8$,
so the deterministic moment upgrade gives
$$
W_4(J\mu,J\mu')
\le(1+16Z_L^4)^{1/4}(C_{\rm p}^{\rm m})^{1/8}
                             \mathsf d_{\rm m}(\mu,\mu')^{1/32}.
$$
Apply (DMC.16) with target positional moment $X_L^8$ to obtain
(DMC.19). The exact source-box safe-return proof
{prf:ref}`thm-dsa-default-box-alive-floor` supplies the final output
alive-mass assertion at these same default kinetic parameters.
:::

:::{prf:corollary} Actual current-survivor conditional one-update error
:label: cor-dmc-current-survival-interface

Start the actual finite chain from any consistent nonempty-alive
capped input law and stop it at extinction $\tau_N$. For every
$n\ge1$, with each conditional expectation using its own actual
survival event,
$$
\mathbb E[\mathsf d_{\rm m}(L_N(S_{n+1}),
                         \mathcal F(L_N(S_n)))\mid\tau_N>n+1]
\le\min\left\{1,\frac{A_N+c_{\rm ret}r_N}{1-e_N}\right\}.
\tag{DMC.20}
$$
The right-hand side tends to zero with $N$, uniformly in $n$.
It compares one actual survivor update with the population update at
that actual entering empirical law; it does not assert attraction to
a stationary law.
:::

:::{prf:proof}
Condition first on $\tau_N>n$. The entering array has alive fraction
at least $m_f$ except with probability at most $c_{\rm ret}r_N$,
by {prf:ref}`cor-dsa-current-conditional-control`. On that good
event (DMC.18) bounds its raw next-update error by $A_N$; elsewhere
the marked cost is at most one. The actual next survival probability
is at least $1-e_N$ from every nonextinct input. Restricting this
nonnegative error to next survival and dividing by that one-step
probability proves (DMC.20). This includes the change of the entering
law induced by next survival; no denominator for the full elapsed
history appears. Since $A_N,r_N,e_N\to0$, the limit follows.
:::

(sec-dmc-horizon)=
## 5. Finite-horizon survivor comparison without an attraction premise

:::{prf:corollary} A complete recent finite-horizon population comparison
:label: cor-dmc-finite-horizon

Start from a fixed capped consistent input $S$ with alive fraction
at least $m_f$, and set $\mu_0=L_N(S)$,
$\mu_{j+1}=\mathcal F\mu_j$. For any integer $b\ge1$, put
$$
D=1+\mathcal C_{\rm mod},\qquad
V_{N,b}=D^{1/(1-\alpha)}A_N^{\alpha^{b-1}},\qquad
T_{N,b}=(1-e_N)^{-b},\quad \alpha=1/32.
$$
Then the original killed chain, conditioned once on its own survival
through $b$ updates, satisfies
$$
\mathbb E[\mathsf d_{\rm m}(L_N(S_b),\mu_b)\mid\tau_N>b]
\le\min\{1,T_{N,b}[V_{N,b}+(b-1)r_N]\}.
\tag{DMC.21}
$$
For each fixed $b$ this tends to zero. It also tends to zero for
$$
b=b_N=1+\left\lfloor\frac{\log\log(N+e)}{2\log32}\right\rfloor.
\tag{DMC.22}
$$
The same bound can be used on a recent continuation from an actual
current-survivor starting law: add $c_{\rm ret}r_N$ inside the
bracket for its initial low-alive event, and use the random population
trajectory starting at that entering empirical law on its good event.
This does not replace that random starting law by a stationary phase.
:::

:::{prf:proof}
Use the stopped raw continuation, assigning its retained marked array
after extinction only to define bounded observables. An all-dead
array never receives another gas update. Let $G_j$ be the event
that all entering arrays at updates $0,\ldots,j-1$ have alive
fraction at least $m_f$. Define the bounded comparison cost $d_j$
to the deterministic $\mu_j$ even on the stopped continuation and
$q_j=\mathbb E[\mathbf1_{G_j}d_j]$. Thus $q_0=0$.
On $G_{j+1}$ the input is nonextinct and (DMC.18)--(DMC.19)
give, by conditional expectation and subprobability Jensen,
$$
q_{j+1}\le A_N+\mathcal C_{\rm mod}q_j^\alpha.
$$
The population comparator has alive mass at least $a_{\rm ret}$
after its first update, so both inputs meet the modulus premise
wherever it is applied. Since $q_j\le1$ and $A_N\le1$,
induction gives
$q_b\le D^{1+\alpha+\cdots+\alpha^{b-2}}
 A_N^{\alpha^{b-1}}\le V_{N,b}$.

An entering failure at any update $j\ge1$ is charged only on
paths that have survived to its preceding update. Its probability
is at most $r_N$ by the conditional binomial safe-return theorem;
extinction at that output is itself a low-alive failure.
Thus the first-failure union has probability at most $(b-1)r_N$.
The actual continuation survival probability is at least
$(1-e_N)^b$ from its fixed nonextinct input. Restrict its nonnegative
cost to surviving paths and divide by this recent-window denominator
to obtain (DMC.21). For an actual current-survivor starting law,
the initial low-alive probability is at most $c_{\rm ret}r_N$;
the same first-failure argument applies to its ordinary continuation.
The exact recent-window Radon--Nikodym formula
{prf:ref}`cor-dsa-recent-window-normalization` includes the starting-law
tilt in that final division.

For (DMC.22), write $L_N=\log(N+e)$. Then
$\alpha^{b_N-1}\ge L_N^{-1/2}$.
Eventually $\log A_N\le-\frac a2\log N$, hence
$\log V_{N,b_N}\le\log D/(1-\alpha)
 -\frac a2L_N^{-1/2}\log N\to-\infty$.
Also $b_Nr_N\to0$ and $b_Ne_N\to0$, so $T_{N,b_N}\to1$.
This proves the displayed growing-horizon limit without a
population contraction or a mixing rate.
:::

:::{prf:remark} Precisely completed default interface
:label: rem-dmc-scope

The primitive interval (DMC.3) gives a nonempty small positive
fitness-exponent regime at the unchanged default harmonic timestep,
count viscosity, cap, noises, restitution and box. It verifies the
accepted alive-arm subcritical test explicitly. All mandatory revival
and sampled normalizer terms remain in its finite comparison constants;
the possible dead fraction is $1-m_f$, not a presumed small value.

The completed results are (DMC.18)--(DMC.19)'s marked consistency
and weak modulus, (DMC.20)'s current-survival one-update comparison,
and (DMC.21)'s finite/growing-window approximation to the actual
population trajectory. They supply a finite-particle interface for a
future separately proved default population attraction theorem.
They establish no stationary target, exact finite-array mixing,
finite QSD rate, or complete own-law contraction at $\nu=.3$.
The reward stays the raw same-potential quadratic; its alive-support
bound is an analytic estimate and not an altered reward channel.
:::
