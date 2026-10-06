# Complete stationary population linearization of the native count instrument

(sec-npl-register)=
## 1. Existing execution and the accepted component regime

:::{prf:definition} Native linearization register
:label: def-npl-register

Retain the complete execution record of
{prf:ref}`def-native-complete-execution-record` and the active count
restriction of {prf:ref}`def-npf-register`. In particular the existing
real-coordinate dense execution, independently sampled original distance
companions, sampled global fitness standardizers, current clone donors,
original clipped acceptance, simultaneous copying, entire accepted-component
Haar rotations, both dense kicks, positive cap and original unbounded
jitter, OU and terminal position noises are unchanged.
Keep every primitive coefficient of
{prf:ref}`def-native-phase-preparation-coefficients` and
{prf:ref}`def-nqc-preparation-budget`.
The entering high-alive class is $H_N$, its actual alive fraction is
at least $m_0$, and $m_0N\ge2$.
The donor lower profile is $\kappa_C>0$.
Uniform donors mean the existing `Kernel::Uniform` tag.
The present continuous-law assertions do not assign Jacobians or Gaussian
sampling laws to a deterministic fixed-seed finite-arithmetic execution.

For the original fitness coefficients $H_r,H_s$, reward range $R_b$,
diversity range $S_b$, regularizers $\sigma_r,\sigma_s$, positive
fitness floor $F_*$ and gate parameters $s_c,\epsilon_c$, define

$$
\begin{gathered}
G_{\rm live}=
 \min\left\{1,
 \frac{H_rR_b/\sigma_r+H_sS_b/\sigma_s}
                  {s_c(F_*+\epsilon_c)}\right\},\qquad
C_{\rm dead}=\frac1{\kappa_Cm_0},\\
\varrho=2e C_{\rm dead}G_{\rm live}.
\end{gathered}
\tag{NPL.1}
$$
These are derived functions of the original parameters. The additional
positive component regime below is the evaluated inequality
$\varrho<1$, rather than an assumed random graph property.
Skipped existing cloning periods can only reduce $G_{\rm live}$;
mandatory revival retains acceptance one on those periods.
:::

:::{prf:theorem} Primitive exponential tail of every actual accepted component
:label: thm-npl-component-exponential

Suppose $0<\varrho<1$ and set

$$
\begin{gathered}
\theta=\min\left\{1,
       \frac{-\log\varrho}{2(1+2C_{\rm dead})}\right\},\qquad
u_\theta=\theta+C_{\rm dead}(e^\theta-1),\\
M_{\rm comp}=e^{\theta+\nu_\theta}
 \left(1+\frac{e\varrho e^{\nu_\theta}}
                       {1-\varrho e^{\nu_\theta}}\right).
\end{gathered}
\tag{NPL.2}
$$
For every entering $S\in H_N$, every fixed original row $i$ and its
actual accepted connected component $\mathcal C_i$,

$$
E[e^{\theta|\mathcal C_i|}\mid S]\le M_{\rm comp},\qquad
P\left[\max_i|\mathcal C_i|\ge k\mid S\right]
             \le\min\{1,NM_{\rm comp}e^{-\theta k}\}.
\tag{NPL.3}
$$
The bounds remain valid after freezing the complete sampled fitness
array, with every actual mandatory revival leaf retained.
Consequently, for every $0<\varepsilon<1$, all actual components
have size less than
$\lceil\theta^{-1}\log(NM_{\rm comp}/\varepsilon)\rceil$
with probability at least $1-\varepsilon$.
The graph and collision operator have not been truncated.

If $G_{\rm live}=0$, its live components have size exactly one;
(NPL.3) holds for any $\theta>0$ with
$M_{\rm comp}=\exp[2\theta+C_{\rm dead}(e^\theta-1)]$.
:::

:::{prf:proof}
For two live rows the shared alive means and standardization scales
cancel in their score differences. Their reward standardized difference
is at most $R_b/\sigma_r$, and their diversity standardized difference
at most $S_b/\sigma_s$. Integrating the actual positive logistic-power
derivative coefficients $H_r,H_s$ along the joining segment bounds
their fitness difference by the numerator of (NPL.1).
The own fitness is at least $F_*$. The existing clipped ratio therefore
has live acceptance at most $G_{\rm live}$, including every sampled
measurement and every tie.

Freeze $S$ and its full measured fitness array. The original clone donor
and acceptance blocks are independent by recipient. Every specified
accepted live edge $i\to j$ has conditional probability at most
$G_{\rm live}/(\kappa_Cm_0N)$. An accepted live edge strictly increases
fitness, and a row has outdegree at most one. Hence the live graph is
a forest. Each dead row points to a current alive donor and is itself
ineligible as a donor: dead rows are leaves and cannot join two live
components. These facts use the actual accepted graph; rejected
proposals are not collision edges.

For a fixed alive root let $L$ be its number of live component vertices.
If $L\ge k\ge2$, that component contains a connected set of exactly
$k$ live vertices containing the root. There are at most
$\binom{N-1}{k-1}$ choices of the other vertices, $k^{k-2}$ labeled
trees and $2^{k-1}$ orientations. A feasible orientation has distinct
source rows, so its specified edges have joint probability at most
$[G_{\rm live}/(\kappa_Cm_0N)]^{k-1}$.
An orientation violating outdegree one has probability zero.
The union bound and $(k-1)!\ge((k-1)/e)^{k-1}$ give

$$
P[L\ge k\mid S,\text{fitness}]
\le\frac{k^{k-2}}{(k-1)!}
       (2C_{\rm dead}G_{\rm live})^{k-1}
\le\frac e k\varrho^{k-1}\le e\varrho^{k-1}.
\tag{NPL.4}
$$
For a dead root expose its own current donor first. That draw is
independent of the remaining live graph, conditional on the frozen
record. Mixing the uniform bound (NPL.4) over its possible alive donors
gives the same bound for the number $L$ of live vertices in its component.

Conditional on the entire live forest, the remaining dead donor draws
are still independent. A given dead row hits the $L$ live vertices of
this component with probability at most
$\min\{1,C_{\rm dead}L/N\}$. Thus the number $D$ of its other dead
leaves obeys

$$
E[e^{\theta D}\mid\text{live forest},S,\text{fitness}]
\le\exp[C_{\rm dead}L(e^\theta-1)].
$$
An exposed dead root contributes at most one additional leaf. In both
root cases $|\mathcal C_i|\le1+L+D$ is safe.
For $\theta\le1$, $e^\theta-1\le2\theta$; hence
$\nu_\theta\le-\frac12\log\varrho$ and
$\varrho e^{\nu_\theta}<1$.
The integer tail-sum identity and (NPL.4) yield

$$
E[e^{\nu_\theta L}]
\le e^{\nu_\theta}
 \left(1+\frac{e\varrho e^{\nu_\theta}}
                      {1-\varrho e^{\nu_\theta}}\right).
$$
Multiplication by the exposed-root factor $e^\theta$ gives (NPL.2)--(NPL.3).
The union bound covers all original roots simultaneously.
If $G_{\rm live}=0$, $L=1$ deterministically and the same dead-leaf
calculation gives the stated separate formula.
Finally integrate the unchanged original fitness array and entering
state. No independence of completed output rows was used.
:::

:::{prf:corollary} Evaluated active count witness has exponentially small components
:label: cor-npl-component-witness

Keep every original parameter of (PC.37), its exact positive cap
$V=V_{\rm crit}/2$ and every restriction of
{prf:ref}`cor-npf-positive-witness`. The exact primitive formulas give

$$
\begin{gathered}
R_b\simeq17.5454058871,\quad S_b\simeq5.65585433788,\quad
G_{\rm live}\simeq1.16006185119\,10^{-7},\\
\varrho\simeq1.60266471369\,10^{-6},\quad
C_{\rm dead}\simeq2.54118950060,\quad
\theta=1,\quad \nu_\theta\simeq5.36647974155,\quad
M_{\rm comp}\simeq582.548466466.
\end{gathered}
\tag{NPL.5}
$$
In particular $P[\max_i|\mathcal C_i|\ge k\mid S]
\le582.549Ne^{-k}$ throughout $H_N$.
The decimals are diagnostics; (NPL.1)--(NPL.3) define the bound.
This confirms a nonempty original parameter regime for exponentially
tailed components. Failure of $\varrho<1$ does not prove that its
actual algorithm has large components: it marks the limit of this
particular primitive tree bound.
:::

:::{prf:proof}
Insert $H_r=H_s=1/2$, $\sigma_r=\sigma_s=10^6$,
$s_c=100$, $F_*=1$, $\epsilon_c=10^{-6}$,
$\kappa_C=1$, $m_0=a_0/2$, $R_D=2\sqrt3$,
$\lambda=4/(1+e^{-1})$ and the original $S_b$ from (PC.4).
The resulting $\varrho$ is positive and less than one.
Moreover $-\log\varrho>2(1+2C_{\rm dead})$, so (NPL.2)
indeed selects $\theta=1$. The displayed evaluations follow.
:::


(sec-npl-fresh-functional)=
## 2. Exact empirical functional of all original post-collision sources

:::{prf:definition} Complete downstream functional and first variation
:label: def-npl-downstream-functional

The actual compact preparation is
$W_i=(Y_i,I_i,w_i)\in\mathcal W$
with $|Y_i|\le R_D$, $I_i\in\{0,1\}$, $|w_i|\le V_c$.
Write $\eta_N=N^{-1}\sum_i\delta_{W_i}$.
Conditional on this ENTIRE original preparation, its addressed original
jitter and OU blocks $(G_i,\xi_i)$ are independent standard $d$-Gaussians,
independent by row. Put

$$
E_i=(W_i,G_i,\xi_i),\quad
L_N=N^{-1}\sum_i\delta_{E_i},\quad
\Lambda_\eta=\eta\otimes\gamma_d\otimes\gamma_d,
\qquad a_s(y)=P[y+sG^{\rm pos}\in D].
\tag{NPL.6}
$$
For any probability $\Lambda$ on these extended coordinates with the
original finite moments, define

$$
\begin{aligned}
X(e)&=Y+\sigma_J I G,\
U_\Lambda(e)&=w+t\nu\int K(X(e)-X(v))(w(v)-w(e))\,d\Lambda(v),\
z_\Lambda(e)&=c[U_\Lambda(e)-t\lambda X(e)]+q\xi(e),\
y_\Lambda(e)&=a_xX(e)+bU_\Lambda(e)+tq\xi(e),\
F_\Lambda(e)&=\nu\int K(y_\Lambda(e)-y_\Lambda(v))
                       (z_\Lambda(v)-z_\Lambda(e))\,d\Lambda(v),\
h_\Lambda(e)&=(1-I(e))a_s(y_\Lambda(e))
                         \psi(z_\Lambda(e),F_\Lambda(e)),\
\mathcal T(\Lambda)&=\int h_\Lambda(e)\,d\Lambda(e).
\end{aligned}
\tag{NPL.7}
$$
Both count-kick self terms vanish exactly. Consequently the actual
original $U_i,z_i,y_i,F_i$ equal (NPL.7) at $(\Lambda,e)=(L_N,E_i)$,
and its conditional terminal mean is exactly $\mathcal T(L_N)$.
This is an identity for the existing simultaneous finite update, including
its dense correlations and clone mask.

Let $|\psi|\le B$, its first derivative budgets be $L_z,L_F$, and its
joint Hessian norm be bounded by a declared finite $L_2$.
These are budgets of the chosen smooth TEST of the original record.
They do not alter its actual force threshold or any noise.
Set $m(e)=1-I(e)$ and write $y,z,F,h$ for (NPL.7) at a base $\Lambda$.
Define

$$
\begin{aligned}
J_U(e,v)&=t\nu K(X(e)-X(v))(w(v)-w(e)),\\
J_F(e,v)&=\nu K(y(e)-y(v))(z(v)-z(e))\\
&\quad+\nu\int\Big[
 b\nabla K(y(e)-y(e'))\!\cdot\!
       (J_U(e,v)-J_U(e',v))(z(e')-z(e))\\
&\hspace{43mm}+cK(y(e)-y(e'))
       (J_U(e',v)-J_U(e,v))\Big]d\Lambda(e'),\\
\Phi_\Lambda(v)&=h(v)
 +\int m(e)\{b\psi(z(e),F(e))\nabla a_s(y(e))
             +ca_s(y(e))\nabla_z\psi(z(e),F(e))\}
                         \!\cdot\!J_U(e,v)\,d\Lambda(e)\\
&\quad+\int m(e)a_s(y(e))\nabla_F\psi(z(e),F(e))
                         \!\cdot\!J_F(e,v)\,d\Lambda(e).
\end{aligned}
\tag{NPL.8}
$$
The kernels and every sign in this derivative use the stored original
count B1 and B2 stages. In particular $J_F$ contains both arguments of
its kernel, both velocity arguments and the propagated B1 variation.
:::

:::{prf:lemma} Exact first variation and primitive Gaussian envelope
:label: lem-npl-first-variation

For a signed zero-mass perturbation $\dot\Lambda$ with finite second
moments along an admitted probability segment,

$$
D\mathcal T(\Lambda)[\dot\Lambda]
                  =\int\Phi_\Lambda(v)\,d\dot\Lambda(v).
\tag{NPL.9}
$$
Put $U_0=2t\nu V_c$, $A_1=g_{d,1}/s$,
$\ell_\rho=e^{-1/2}/\rho$ and
$Z_p=cV_c+ct\lambda(R_D+\sigma_Jg_{d,p})+qg_{d,p}$.
Then, for $\Lambda=\Lambda_\eta$ and every compact $\eta$,

$$
\begin{aligned}
|\Phi_\Lambda(v)|&\le A_\Phi+B_\Phi|z_\Lambda(v)|,\\
A_\Phi&=B+U_0(bBA_1+cL_z)
 +\nu L_F\{Z_1+2U_0(2b\ell_\rho Z_1+c)\},\
B_\Phi&=\nu L_F,\qquad
\|\Phi_\Lambda\|_{L^p(\Lambda)}\le A_\Phi+B_\Phi Z_p.
\end{aligned}
\tag{NPL.10}
$$
The SAME last bound holds conditional on each fixed compact $W(v)$,
since $|U_\Lambda|\le V_c$, $|Y|\le R_D$ and the two original
Gaussian norms give $\|z_\Lambda(W,G,\xi)\|_{L^p(G,\xi)}\le Z_p$
uniformly in $W$. Here $g_{d,p}=(E|G_d|^p)^{1/p}$ is the explicit
original Gaussian moment. No source draw has been bounded pointwise.
:::

:::{prf:proof}
The first kick is affine in its integrating measure, so
$\dot U(e)=\int J_U(e,v)d\dot\Lambda(v)$,
$\dot z=c\dot U$ and $\dot y=b\dot U$.
Differentiate the original B2 integral. Differentiation of its measure
gives the first term of $J_F$; differentiation of its kernel and
velocity difference gives respectively the two terms in its inner
integral. Finally differentiating $h$ and its outer measure gives
(NPL.8)--(NPL.9). Gaussian moments and bounded kernel derivatives
justify the displayed differentiations by dominated integration.

The actual first kick is a convex combination for $0\le t\nu\le1$,
so $|U_\Lambda|\le V_c$. Minkowski yields $\|z_\Lambda\|_p\le Z_p$.
The Gaussian density score gives $|\nabla a_s|\le g_{d,1}/s$.
Also $|J_U|\le U_0$ and

$$
\int|J_F(e,v)|d\Lambda(e)
\le\nu[|z(v)|+Z_1+2U_0(2b\ell_\rho Z_1+c)].
$$
Insert these in (NPL.8) to prove (NPL.10).
:::

:::{prf:lemma} Explicit Hessian budget for a smooth native projector test
:label: lem-npl-projector-hessian

In (NPF.3) choose the original passive taper
$\chi(r)=10u^3-15u^4+6u^5$ for
$u=(r-\delta_c)/\delta_c\in[0,1]$, extended by zero and one
at the respective ends. For $\delta_c>0$ it is $C^2$ and the
original available projector test has the primitive joint Hessian budget

$$
L_2=128B_A(1+|\kappa_c|+\delta_c^{-1})^2.
\tag{NPL.11}
$$
The existing threshold and matched mask are retained.
:::

:::{prf:proof}
The taper has $|\chi'|\le2/\delta_c$ and
$|\chi''|\le60/\delta_c^2$; its first two derivatives vanish at
both joining endpoints. The normalization map $F\mapsto F/|F|$
has first derivative norm at most $1/|F|$ and second derivative norm
at most $3/|F|^2$.
Differentiating $cc^\dagger$ gives the bounds
$4B_A\kappa_c^2$, $8B_A|\kappa_c|/\delta_c$ and
$100B_A/\delta_c^2$ for its two pure and one mixed blocks
including the taper. Twice the mixed block bounds the joint norm.
Their sum is at most (NPL.11), including the $C^2$ zero extension.
:::

(sec-npl-quantitative-remainder)=
## 3. Quantitative complete downstream empirical linearization

:::{prf:definition} Explicit sampling and derivative budgets
:label: def-npl-remainder-budget

All constants below are functions of the original register and declared
smooth test. Put $m=4d$, $a=1/(m+2)$ and, for $N\ge2$,

$$
\begin{gathered}
r_N=\sqrt{16d\log(N+1)},\quad
T_N=\max\{1,R_D,V_c,r_N\},\qquad
D_N=2\sqrt m T_N+1,\\
S=2+d+R_D+V_c+\sigma_J+t+\nu+c+\lambda
 +|a_x|+b+q+s^{-1}+\rho^{-1}+B+L_z+L_F+L_2,\\
\mathcal C_N=2^{50}(d+1)^{12}S^{100}(1+r_N)^{16},\qquad
R_N^{\rm query}=S^6(1+r_N)^2,\
Q_N=(3+2\sqrt d R_N^{\rm query}N^2)^d,\qquad
J_d=2(d+1)^2,\\
\varepsilon_N=
 \sqrt{\frac{2\log[2J_dQ_N(N+1)^4]}N}
                +4S^4N^{-2},\qquad
C_{W,N}=\sqrt m T_N+D_N\sqrt2\,3^{m/2},\\
\mathcal E_N=\mathcal C_N\sqrt N
 [\varepsilon_N^2+
       \varepsilon_N C_{W,N}N^{-a}+(N+1)^{-4}]
       +\mathcal C_N\sqrt{8dN}\,(N+1)^{-4}.
\end{gathered}
\tag{NPL.12}
$$
Thus $\mathcal E_N\to0$. Its logarithmic factors are deliberately
retained; none of these finite budgets is an assumption on an unknown
law. The clipping in the proof below is an analytic comparison only.
The theorem concerns the original unbounded source draws.
:::

:::{prf:theorem} Complete post-collision linearization with a vanishing population remainder
:label: thm-npl-downstream-linearization

For every deterministic compact preparation array $W_1,\ldots,W_N$,
with $\Lambda_N=\Lambda_{\eta_N}$,

$$
\mathcal T(L_N)-\mathcal T(\Lambda_N)
 =\frac1N\sum_{i=1}^N
 [\Phi_{\Lambda_N}(E_i)
       -E(\Phi_{\Lambda_N}(E_i)\mid W_i)]+R_N,
\qquad E[\sqrt N|R_N|\mid W]\le\mathcal E_N.
\tag{NPL.13}
$$
The inequality is uniform over the entire compact preparation class.
It includes both dense kicks and all original correlated first-kick
source/velocity inputs. In particular
$\sqrt N[E(\mathcal T(L_N)\mid W)-\mathcal T(\Lambda_N)]$
tends uniformly to zero.
:::

:::{prf:proof}
First clip only the two original Gaussian arguments to the radius-$r_N$
ball, retaining their independent row laws and the exact fixed array $W$.
A componentwise Gaussian union bound gives
$P[|G_d|>r_N]\le2d(N+1)^{-8}$.
Consequently the probability that any of the $2N$ original blocks differs
from its clipped comparison is at most $4dN(N+1)^{-8}$.
For the reference law, Cauchy--Schwarz and its fourth Gaussian moment
bound the first three absolute tail moments by a primitive polynomial
in the register times $(N+1)^{-4}$.
These source tails, and the linear envelope (NPL.10), are dominated by
the last term of (NPL.12). It suffices to prove the asserted expansion
for the clipped comparison and restore these tails at the end.

Write $\Lambda$ for the clipped mean law, $L$ for its empirical law and
$\delta=L-\Lambda$. The $m=4d$ continuous coordinates consist of
$(Y,w,G,\xi)$; $I$ contributes two discrete strata.
Partition each coordinate into cells of width $T_NN^{-a}$.
Each cell empirical mass has variance at most its mean divided by $N$:
rows are independent, with their actual nonidentical fixed-$W_i$ laws.
There are at most $2\,3^mN^{ma}$ cells.
Couple within cells, and pay the entire diameter for excess cell masses.
Cauchy--Schwarz for the summed cell errors gives

$$
E W_1(L,\Lambda)\le C_{W,N}N^{-a}.
\tag{NPL.14}
$$
This uses no independent completed output rows and no randomization of
its fixed preparation coordinates.

The linear first-kick error is

$$
\dot U(X,w)=t\nu\left[
 \int K(X-X')w'\,d\delta- w\int K(X-X')\,d\delta\right].
$$
Apply the elementary exponential bound for the mean of independent
bounded variables to the scalar fields $K$, $\nabla K$ and those
fields multiplied by each normalized $w'$ coordinate.
Do the same for the base B2 fields $K(y-y')$, $\nabla K(y-y')$
and their products with each normalized $z'$ coordinate.
Their number is at most $J_d$; both query boxes have radius at most
$R_N^{\rm query}$.
A mesh of Euclidean accuracy $N^{-2}$ has at most $Q_N$ points.
Since $|\nabla K|\le e^{-1/2}/\rho$ and
$\|D^2K\|\le2/\rho^2$, interpolation between grid points costs
at most $4S^4N^{-2}$ after normalization.
The union bound therefore gives an event $\mathcal A_N$, of complement
probability at most $(N+1)^{-4}$, on which all these field errors and
all their required first query derivatives are bounded by their
primitive scales times $\varepsilon_N$.
In particular $\dot U$, its first source derivative and the direct
base B2 empirical field have this bound.

For completeness, all subsequent composition budgets can be checked
without an unknown modulus. On the clipped box, the value and first
two continuous derivatives of the following original expressions are
bounded respectively by

$$
\begin{array}{c|c}
X&4S^2(1+r_N)\\
U_\Lambda&2^6S^8(1+r_N)^2\\
(y_\Lambda,z_\Lambda)&2^9S^{12}(1+r_N)^2\\
K(y_\Lambda(e)-y_\Lambda(v))(z_\Lambda(v)-z_\Lambda(e))
   &2^{25}S^{30}(1+r_N)^6\\
h_\Lambda&2^{40}(d+1)^3S^{70}(1+r_N)^{12}.
\end{array}
\tag{NPL.15}
$$
Use the displayed kernel bounds, $|U_\Lambda|\le V_c$,
$|\nabla a_s|\le g_{d,1}/s$ and
$\|D^2a_s\|\le(d+1)/s^2$ in the ordinary product and chain rules.
Each discrete-$I$ difference costs at most twice the corresponding
value bound; each such term is absorbed by $\mathcal C_N$.
These deliberately larger bounds also absorb the finite number of
products of linear variation kernels in the following Taylor expansion.

The first-kick variation is exact. Its original $y,z$ variations are
$b\dot U,c\dot U$. Expand the B2 kernel in these two arguments and its
velocity difference to first order. The term with its integrating
measure fixed at $\Lambda$ is exactly (NPL.8). Its quadratic source
Taylor remainder is bounded by $\mathcal C_N\varepsilon_N^2$.
Its mixed integrating-measure term has the form
$\int f_{e,\delta}(v)d\delta(v)$, where the first derivative budget of
$f_{e,\delta}$ is at most
$\mathcal C_N\varepsilon_N$ on $\mathcal A_N$:
$\dot U$ and its derivative are small there, whereas the other factors
are the fixed base fields bounded in (NPL.15).
Thus its absolute value is at most
$\mathcal C_N\varepsilon_N W_1(L,\Lambda)$.
This bounds the full B2 remainder, with its actual dense common-field
correlations retained.

Now expand $h_L$ at the base $(y_\Lambda,z_\Lambda,F_\Lambda)$.
Its integrated linear term is $\int\Phi_\Lambda d\delta$ by
(NPL.9). The quadratic pointwise terms cost
$\mathcal C_N\varepsilon_N^2$; the remaining linear term integrated
against $\delta$ has first derivative budget at most
$\mathcal C_N\varepsilon_N$, because the direct B2 field and its first
query derivative were included in $\mathcal A_N$.
The same transport bound applies. Enlarging the already displayed
budget to include these finitely many terms, on $\mathcal A_N$ we have

$$
|\mathcal T(L)-\mathcal T(\Lambda)-\int\Phi_\Lambda d\delta|
\le\mathcal C_N[\varepsilon_N^2+
                   \varepsilon_N W_1(L,\Lambda)].
$$
If $\varepsilon_N>1$, the uniform clipped value budgets already give
the same inequality after the stated enlargement of $\mathcal C_N$.
On its complement, $|\mathcal T|\le B$ and the clipped influence
envelope bounds this difference by $\mathcal C_N$.
Take expectations, use (NPL.14), and restore the original Gaussian
tails as above. This proves (NPL.12)--(NPL.13).
Finally $\Lambda_N=N^{-1}\sum_i\delta_{W_i}\otimes\gamma_d\otimes\gamma_d$;
therefore $\int\Phi_{\Lambda_N}d(L_N-\Lambda_N)$ is exactly the
conditionally centered, independent row sum in (NPL.13).
The last assertion follows by taking its conditional expectation.
:::


(sec-npl-quenched-gaussian)=
## 4. Full post-collision instantaneous Gaussian law and its actual covariance

:::{prf:definition} Native downstream covariance
:label: def-npl-downstream-covariance

For $\Lambda=\Lambda_\eta$, define its conditional source-centered influence
and two covariance contributions by

$$
\begin{aligned}
\widetilde\Phi_\eta(W,G,\xi)
 &=\Phi_{\Lambda_\eta}(W,G,\xi)
       -E_{G',\xi'}\Phi_{\Lambda_\eta}(W,G',\xi'),\\
V_{\rm src}(\eta)&=\int|\widetilde\Phi_\eta|^2d\Lambda_\eta,\\
V_{\rm mark}(\eta)&=\int (1-I)
  \psi(z_{\Lambda_\eta},F_{\Lambda_\eta})^2
           a_s(y_{\Lambda_\eta})(1-a_s(y_{\Lambda_\eta}))
                                                    \,d\Lambda_\eta,\\
V_{\rm down}(\eta)&=V_{\rm src}(\eta)+V_{\rm mark}(\eta).
\end{aligned}
\tag{NPL.16}
$$
The rowwise conditional centering is necessary: the compact source
$W$ is fixed in the fresh-noise theorem, and its population fluctuation
is not counted a second time in $V_{\rm src}$.
For finitely many tests use their polarized products in these formulas.
Set $M_p=2(A_\Phi+B_\Phi Z_p)$ and

$$
\begin{aligned}
\mathcal J_N&=\mathcal C_N
 [\varepsilon_N+C_{W,N}N^{-a}+(N+1)^{-4}]
          +\mathcal C_N\sqrt{8d}(N+1)^{-4},\\
\mathcal A_N(\vartheta)&=2|\vartheta|\mathcal E_N
 +\tfrac12\vartheta^2\mathcal J_N
 +\frac{|\vartheta|^3(M_3^3+B^3)}{6\sqrt N}
 +\frac{\vartheta^4(M_2^4+B^4)}{8N}.
\end{aligned}
\tag{NPL.17}
$$
Every term tends to zero at each fixed real $\vartheta$.
:::

:::{prf:theorem} Complete downstream conditional population central limit theorem
:label: thm-npl-downstream-clt

Let $H_N$ be the actual complete color/source observation (NPF.2),
with the original terminal status, and let
$\mathcal G_N(W)=E[H_N\mid W]$ under its original raw update.
Uniformly over all deterministic compact arrays,

$$
\left|E\left[e^{i\vartheta\sqrt N(H_N-\mathcal G_N(W))}\mid W\right]
       -e^{-\vartheta^2V_{\rm down}(\eta_N)/2}\right|
                           \le\mathcal A_N(\vartheta).
\tag{NPL.18}
$$
Consequently, if $\eta_N\to\eta$ weakly on $\mathcal W$, its conditional
population fluctuation converges to the centered Gaussian with variance
$V_{\rm down}(\eta)$. For a random actual preparation with
$\eta_N\to\eta$ in probability and deterministic $\eta$, this convergence
is stable relative to its complete original compact preparation:
for every bounded $W$-measurable $Z_N$,

$$
E[Z_N e^{i\vartheta\sqrt N(H_N-\mathcal G_N(W))}]
 -E[Z_N]e^{-\vartheta^2V_{\rm down}(\eta)/2}\longrightarrow0.
\tag{NPL.19}
$$
The limit includes the original jitter, both dense kicks, OU and terminal
alive mark. The global normalizers, donor forest and shared Haar
collisions remain in the exact conditioning variable $W$.
They have not been replaced by independent output rows.
:::

:::{prf:proof}
Conditional on all original upstream $E_i$, write

$$
\sqrt N(H_N-\mathcal T(L_N))
 =N^{-1/2}\sum_i(1-I_i)\psi(z_i,F_i)
                 [a_i^+-a_s(y_i)].
\tag{NPL.20}
$$
The terms on this one final fiber are independent, centered, bounded by
$B$, and have conditional variance average

$$
V_{{\rm mark},N}=N^{-1}\sum_i(1-I_i)\psi(z_i,F_i)^2
                        a_s(y_i)(1-a_s(y_i)).
$$
The same field expansion and empirical transport estimate used in
(NPL.13), now applied to this bounded continuously differentiable
integrand, give

$$
E[|V_{{\rm mark},N}-V_{\rm mark}(\eta_N)|\mid W]\le\mathcal J_N.
\tag{NPL.21}
$$
The original tails are retained by the last term of (NPL.17).

For independent centered variables $X_i$, the elementary third-order
Taylor bound and $|e^{-x}-(1-x)|\le x^2/2$ give

$$
\left|E e^{i\vartheta N^{-1/2}\sum_iX_i}
 -\exp[-\vartheta^2N^{-1}\sum_iEX_i^2/2]\right|
\le\frac{|\vartheta|^3}{6N^{3/2}}\sum_iE|X_i|^3
 +\frac{\vartheta^4}{8N^2}\sum_i(EX_i^2)^2.
\tag{NPL.22}
$$
Indeed compare each characteristic factor to its Gaussian factor and
sum their differences; all factors have modulus at most one.
Apply this on the final fiber to (NPL.20) and then use (NPL.21).
Multiplication by any bounded upstream characteristic factor preserves
these error bounds. Thus this final noise is asymptotically independent
of the upstream linearized sum, with its full actual mark variance.

By (NPL.13), that upstream sum consists of the independent centered
$X_i=\widetilde\Phi_{\eta_N}(W_i,G_i,\xi_i)$.
Their averaged second moment is exactly $V_{\rm src}(\eta_N)$;
the uniform ROWWISE conditional form of (NPL.10) bounds each
$p$th norm by $M_p$, including $\sum_i(EX_i^2)^2\le NM_2^4$.
Apply (NPL.22) once more. The replacement of the original functional
by its linearization costs at most $|\vartheta|\mathcal E_N$.
Its exact conditional center differs from $\mathcal T(\Lambda_N)$
by at most $\mathcal E_N/\sqrt N$, and costs the other identical
term. Combining these estimates proves (NPL.17)--(NPL.18).

It remains to verify that this covariance has the required native
continuity, rather than assume it. For $\eta_n\Rightarrow\eta$ on the
compact source space, couple converging source variables with the same
original independent $G,\xi$. In each binary-$I$ stratum they converge
almost surely; the different stratum probability tends to zero.
Bounded kernel convergence gives $U_{\Lambda_{\eta_n}}\to U_{\Lambda_\eta}$,
hence $y,z$ converge. The uniform original Gaussian moments dominate the
B2 integral, then both integrals in (NPL.8). Consequently $F$ and $\Phi$
converge in $L^2$ under this coupling. Their rowwise conditional means
converge in $L^2$ by conditional Jensen and dominated integration.
The same argument applies to the bounded mark integrand.
Thus $V_{\rm down}(\eta_n)\to V_{\rm down}(\eta)$.
This proves the deterministic-array assertion by (NPL.18).
For random preparation, its right side is uniform and its covariance
converges in probability. Multiply the conditional formula by bounded
$Z_N$ and use bounded convergence to obtain (NPL.19).
:::

:::{prf:corollary} Actual stationary instantaneous law is preparation plus an independent complete downstream Gaussian
:label: cor-npl-stationary-convolution

In the existing positive stationary count phase, the actual compact
preparation has the proved deterministic limit $\Theta_*$ of
{prf:ref}`lem-nqg-compact-preparation`.
Set $V_*=V_{\rm down}(\Theta_*)$ and define its centered preparation
fluctuation under the original stationary raw input by

$$
A_N=\sqrt N\left\{\mathcal T(\Lambda_{\eta_N})
                         -E\mathcal T(\Lambda_{\eta_N})\right\}.
\tag{NPL.23}
$$
It is tight. The actual complete raw stationary population fluctuation
satisfies the exact asymptotic characteristic factorization

$$
E e^{i\vartheta\sqrt N(H_N-EH_N)}
       -E e^{i\vartheta A_N}e^{-\vartheta^2V_*/2}\longrightarrow0.
\tag{NPL.24}
$$
On every subsequence with $A_N\Rightarrow A$, the complete fluctuation
converges to $A+G$, where $G\sim\mathcal N(0,V_*)$ is independent of $A$.
The same assertions hold for the actual stationary full Doob instrument,
with its own exact stationary means and its original preparation.
All finite collections of smooth tests obey the polarized vector
factorization. Thus only the compact preparation bracket remains for
this smooth instantaneous population law; no missing B1/B2 force
influence is hidden in a terminal-only covariance.
:::

:::{prf:proof}
The inherited actual compact preparation transport bound implies
$\eta_N\to\Theta_*$ in probability, including its shared original Haar
components and global measurement normalization.
By (NPL.13),
$\sqrt N[\mathcal G_N(W)-\mathcal T(\Lambda_{\eta_N})]$
tends to zero in $L^1$. Its centered version does also.
Moreover $\operatorname{Var}(\mathcal G_N(W))\le\operatorname{Var}(H_N)$
by the conditional variance decomposition.
The already derived stationary bound of
{prf:ref}`thm-npf-stationary-tightness` therefore makes the centered
$\sqrt N\mathcal G_N(W)$ tight. This proves the tightness of (NPL.23)
without postulating a Gaussian preparation or its variance limit.

Write $\sqrt N(H_N-EH_N)$ as this centered conditional mean plus its
conditionally centered downstream fluctuation. Apply (NPL.19) with
$Z_N=e^{i\vartheta A_N}$, and absorb their vanishing $L^1$ center
difference. This gives (NPL.24). A centered Gaussian of variance $V_*$
has the displayed characteristic factor; multiplication by the limiting
characteristic function of $A$ identifies precisely the independent
sum law. The same proof for each real linear combination gives the
finite-dimensional polarized assertion.

Finally {prf:ref}`thm-nue-doob-comparison` bounds the TV discrepancy
of the entire raw and actual stationary Doob one-update records by its
explicit eigenfunction/survival defect $\tau_N$, with
$\sqrt N\tau_N\to0$. All characteristic observables are bounded by one,
and their mean center discrepancy is at most $2B\tau_N$.
The preparation functional is bounded by $B$, so it obeys the same
center and characteristic comparison. Consequently the identical
factorization transfers to the actual full Doob instrument.
This bounded full-record comparison preserves the conditional source
correlations rather than declaring its Doob noises independent.
:::
