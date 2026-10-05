# Exact original-jitter consumer for the signed first count provider

(sec-fjt73-register)=
## 1. Frozen actual plans and the original recipient Gaussians

:::{prf:definition} Coupled source-plan register
:label: def-fjt73-register

Retain the default harmonic count register and signed coefficients of
{prf:ref}`def-dbl68-register`. In particular $d=3$, $h=.04$,
$a=.006$, $\sigma_J=.1$, $V_c=4$, and
$$
A_r=ba_x+gf,\qquad A_p=b^2+g^2,
\qquad .0399074<A_r<.0399395,\quad
.92464<A_p<.926183.
$$
The own count denominator, raw reward, accepted source/component plans,
original-slot Haar collision velocities and every donor/fitness normalizer
are unchanged.

Couple two complete actual preparation plans before drawing their recipient
jitters. For root $i$, the plan fixes
$$
S_{0,i},S_{1,i}\in[-2,2]^3,\qquad
I_{0,i},I_{1,i}\in\{0,1\},\qquad P_{0,i},P_{1,i}.
$$
Each $P_{j,i}$ is the actual original-slot component/Haar velocity, with
$|P_{j,i}|\le4$. It is not the velocity at the selected position donor.
A copied or revived root has its original fresh recipient jitter. Share
that jitter only between the two coupled swarms:
$$
X_{j,i}=S_{j,i}+I_{j,i}J_i,\qquad
J_i\sim N(0,\sigma_J^2I_d).
\tag{FJT.1}
$$
Different recipients have independent $J_i$ conditional on the entire
coupled plan, even when their position donors coincide. Each original
transition marginal is preserved by this common-noise coupling.

For $\theta\in[0,1]$ put
$$
S_i=(1-\theta)S_{0,i}+\theta S_{1,i},\quad
I_i=(1-\theta)I_{0,i}+\theta I_{1,i},\quad
P_i=(1-\theta)P_{0,i}+\theta P_{1,i},
$$
$$
e_i=S_{1,i}-S_{0,i},\quad
j_i=I_{1,i}-I_{0,i},\quad p_i=P_{1,i}-P_{0,i},
$$
$$
X_i=S_i+I_iJ_i,\qquad r_i=e_i+j_iJ_i.
\tag{FJT.2}
$$
The interpolation need not itself be an algorithm preparation: its two
endpoints are the two actual prepared marginals. Its own intermediate
count field is exactly the field used in the differentiation argument
of research68.

For population laws use independent copies of the coupled root plan and
of its jitter. For finite arrays condition on the entire coupled plans,
use their actual recipient jitters, and retain denominators $N$, $N^2$
and $N^3$ in root, pair and triple sums. A self pair has zero velocity
and differential differences and contributes zero. It is never assigned
an independent second jitter.

This conditioning precedes fresh recipient jitter and future terminal
marking. Conditioning an output on its own survival would reweight these
jitters; that reweighting is not omitted or identified with (FJT.1).
:::

(sec-fjt73-tilt)=
## 2. Gaussian tilting including singular copy-status covariances

:::{prf:lemma} Semidefinite joint Gaussian tilt
:label: lem-fjt73-general-tilt

Let $(D,\eta)$ be any jointly Gaussian vector with means $(\Delta,e)$ and
possibly singular covariance
$$
\Sigma=
\begin{pmatrix}
 V&T\\ T^{\mathsf T}&W
\end{pmatrix}\ge0.
$$
Here each block has dimension $k$, and $D,\eta$ have dimension $k$.
Put $G=I_k+V$. Multiplication of the original joint law by
$K_k(D)=e^{-|D|^2/2}$ has mass
$$
\kappa=\det(G)^{-1/2}
           \exp[-\tfrac12\Delta^{\mathsf T}G^{-1}\Delta].
\tag{FJT.3}
$$
After division by this mass its joint law is Gaussian, with means
$$
\mu=G^{-1}\Delta,\qquad n=e-T^{\mathsf T}G^{-1}\Delta
$$
and covariance blocks
$$
A=V-VG^{-1}V,\qquad
C=T-VG^{-1}T,\qquad
B=W-T^{\mathsf T}G^{-1}T.
\tag{FJT.4}
$$
The blocks are respectively the tilted covariances of $D$, of
$(D,\eta)$ and of $\eta$. Only $G$ is inverted; it is positive
definite even when every original jitter variance is zero.
:::

:::{prf:proof}
Add an independent $E\sim N(0,I_k)$ and let $O=D+E$.
Then $O$ has mean $\Delta$ and nonsingular covariance $G$. Its density
at zero is $(2\pi)^{-k/2}\kappa$. Given $D$, the density of $O$
at zero is $(2\pi)^{-k/2}e^{-|D|^2/2}$. Bayes' density formula
therefore identifies the Gaussian tilt with the law of $(D,\eta)$
conditional on $O=0$.

For completeness its conditional mean and covariance follow without
inverting $\Sigma$. Subtract from $(D,\eta)$ its centered linear
regression
$$
\begin{pmatrix}V\\T^{\mathsf T}\end{pmatrix}
G^{-1}(O-\Delta).
$$
The remainder is jointly Gaussian and has zero covariance with $O$;
its joint Gaussian characteristic function then factors, proving
independence, also for singular remainder covariance. Setting $O=0$
gives the means and covariance in (FJT.4). Finally
$I-VG^{-1}=G^{-1}$ because $G=I+V$. This proves the displayed
mean and cross-covariance formulas. The density calculation proves
(FJT.3). No Gaussian outcome is truncated.
:::

:::{prf:corollary} Actual pair tilt for all combinations of copy statuses
:label: cor-fjt73-pair-tilt

For distinct roots in (FJT.2), define the plan-fixed differences
$$
\Delta=S_i-S_j,\qquad e=e_i-e_j,\qquad
H=P_i-P_j,\qquad \pi=p_i-p_j,
$$
and the actual random differences
$$
D=X_i-X_j,\qquad \eta=r_i-r_j.
$$
Their original joint covariance blocks are scalar multiples of $I_d$:
$$
v=\sigma_J^2(I_i^2+I_j^2),\quad
w=\sigma_J^2(j_i^2+j_j^2),\quad
k=\sigma_J^2(I_ij_i+I_jj_j).
\tag{FJT.5}
$$
In particular $v,w\ge0$, $k^2\le vw$, and
$0\le v,w\le .02$, $|k|\le .02$.
With $\Gamma=1+v$ the exact pair tilt is
$$
\kappa=\Gamma^{-d/2}e^{-|\Delta|^2/(2\Gamma)},\qquad
\mu=\Delta/\Gamma,\qquad n=e-k\Delta/\Gamma,
$$
$$
A=\alpha I_d,\quad C=\chi I_d,\quad B=\omega I_d,
\qquad
\alpha=v/\Gamma,\quad\chi=k/\Gamma,\quad
\omega=w-k^2/\Gamma\ge0.
\tag{FJT.6}
$$
For zero jitter $v=0$, the covariance inequality forces $k=0$;
the formulas still apply. Different copy statuses can give $v=0$
and $w>0$ at an endpoint, or $k$ of either sign. These cases are
included without a division by $v$ or a copy-mass lower bound.
:::

:::{prf:proof}
Conditional on the entire coupled plans, $D$ and $\eta$ are the
affine functions
$$
D=\Delta+I_iJ_i-I_jJ_j,\qquad
\eta=e+j_iJ_i-j_jJ_j.
$$
The two recipient Gaussians are independent and have covariance
$\sigma_J^2I_d$. This gives (FJT.5); covariance positivity gives
$k^2\le vw$. The coefficient bounds use $I_i\in[0,1]$ and
$j_i\in\{-1,0,1\}$. Apply (FJT.3)--(FJT.4). The scalar
simplifications give (FJT.6), including all degenerate cases.
:::

(sec-fjt73-linear)=
## 3. Complete conditional first-bilinear response

:::{prf:lemma} First-jitter response tensor and its mixed signed consumer
:label: lem-fjt73-linear-consumer

Under the actual pair tilt (FJT.6), put
$$
s=\mu\cdot n+d\chi,\qquad
h=n(s+\chi)+\omega\mu.
\tag{FJT.7}
$$
Here $d=3$ is the spatial dimension. The exact original-jitter moments are
$$
\mathbb E_J[K(D)(D\cdot\eta)H]=\kappa sH,
$$
$$
\mathbb E_J[K(D)D\eta^{\mathsf T}]
       =\kappa(\mu n^{\mathsf T}+\chi I_d),\qquad
\mathbb E_J[K(D)\eta(D\cdot\eta)]=\kappa h.
\tag{FJT.8}
$$
Consequently, for the complete coefficient
$\Psi_1=A_r\eta+A_p\pi$ in research68,
$$
\begin{split}
&\mathbb E_J\{K(D)\Psi_1\cdot[(D\cdot\eta)H-\pi]\}\\
&\qquad=\kappa\mathcal L,\qquad
\mathcal L=A_rH\cdot h+A_p(\pi\cdot H)s
                    -A_r n\cdot\pi-A_p|\pi|^2.
\end{split}
\tag{FJT.9}
$$
There is also an exact plan-level square completion:
$$
\begin{split}
\mathcal L={}&
-A_p\left|\pi-\tfrac12
                   \left(sH-\frac{A_r}{A_p}n\right)\right|^2\\
&+\frac{A_p}{4}s^2|H|^2
 +A_rH\cdot\left[\left(\frac s2+\chi\right)n+\omega\mu\right]
 +\frac{A_r^2}{4A_p}|n|^2.
\end{split}
\tag{FJT.10}
$$
Every factor on the right is fixed by the complete coupled pair plans.
Their expectation over plans retains all source/velocity/component
correlations. The vector term involving $\chi$ and $\omega$ is
the actual mixed copy-status response, not a term assigned a favorable sign.
:::

:::{prf:proof}
Under the tilted joint Gaussian, write $D=\mu+\xi$ and
$\eta=n+\zeta$, with centered blocks
$\mathbb E\xi\xi^{\mathsf T}=\alpha I_d$,
$\mathbb E\xi\zeta^{\mathsf T}=\chi I_d$ and
$\mathbb E\zeta\zeta^{\mathsf T}=\omega I_d$.
Then $\mathbb E(D\cdot\eta)=\mu\cdot n+d\chi=s$,
and $\mathbb E(D\eta^{\mathsf T})=\mu n^{\mathsf T}+\chi I_d$.
Odd centered Gaussian moments vanish. Expanding
$\eta(D\cdot\eta)$ leaves
$$
n(\mu\cdot n+d\chi)+\chi n+\omega\mu=h.
$$
Multiplication by the tilt mass proves (FJT.8). The velocity
differences $H,\pi$ are fixed before the recipient Gaussians,
by the original-slot component construction. Distributing
$\Psi_1$ and using $\mathbb E_J[K(D)\eta]=\kappa n$
proves (FJT.9).

Complete the square in the deterministic vector $\pi$:
$$
-A_p|\pi|^2+\pi\cdot(A_psH-A_rn)
=-A_p|\pi-(sH-A_rn/A_p)/2|^2
 +\tfrac{A_p}{4}|sH-A_rn/A_p|^2.
$$
Expand the remaining square, use $h=n(s+\chi)+\omega\mu$,
and collect its $H\cdot n$ term. This gives (FJT.10).
The conditional integration does not factor any random source
displacement from its component velocity.
:::

:::{prf:lemma} Exact conditional spatial-defect moment
:label: lem-fjt73-defect-moment

In (FJT.6)--(FJT.7), the complete spatial-defect moment is
$$
\begin{split}
\mathbb E_J[K(D)(D\cdot\eta)^2|H|^2]
=\kappa|H|^2\big\{&
s^2+\omega|\mu|^2+\alpha|n|^2
 +2\chi\mu\cdot n\\
&+d(\alpha\omega+\chi^2)\big\}.
\end{split}
\tag{FJT.11}
$$
Thus the exact $S_\theta$ of (RFP.10) is half the plan-pair
expectation of (FJT.11). In particular this is a full Gaussian
consumer of that weighted local product, including mismatched copy
statuses. It does not replace the remaining plan expectation by a
product of averaged moments.
:::

:::{prf:proof}
Write $D\cdot\eta=\mu\cdot n+\mu\cdot\zeta+
n\cdot\xi+\xi\cdot\zeta$. Its tilted mean is $s$.
The centered linear part has variance
$\omega|\mu|^2+\alpha|n|^2+2\chi\mu\cdot n$.
Its covariance with $\xi\cdot\zeta-d\chi$ is zero by the
vanishing of odd centered Gaussian moments. The latter part has
variance $d(\alpha\omega+\chi^2)$: expanding its fourth
moment gives one pairing $\alpha\omega$ and one additional
pairing $\chi^2$ per coordinate, after subtracting its squared
mean. Adding these variances to $s^2$ proves (FJT.11).
The definition of $S_\theta$ uses
$\dot k=-K(D)(D\cdot\eta)$, so
$\dot k^2/k=K(D)(D\cdot\eta)^2$ exactly.
:::

(sec-fjt73-square)=
## 4. Exact force square with the common recipient retained

:::{prf:lemma} Two-edge jitter tensor with coincident environments
:label: lem-fjt73-two-edge

Take a query root $i$ and environment roots $j,k$. Population
environment roots are independent copies, while finite roots may have
$j=k$. The same actual recipient Gaussian is used wherever the same
root index occurs. Exclude zero self edges $i=j$ or $i=k$, whose force
vectors vanish identically.

For edges $\ell=1,2$ define
$$
D_1=X_i-X_j,\quad D_2=X_i-X_k,\qquad
\eta_1=r_i-r_j,\quad\eta_2=r_i-r_k,
$$
$$
H_1=P_i-P_j,\quad H_2=P_i-P_k,\qquad
\pi_1=p_i-p_j,\quad\pi_2=p_i-p_k.
$$
Let $V,T,W$ be the original joint covariance blocks of
$D=(D_1,D_2)$ and $\eta=(\eta_1,\eta_2)$. They are computed
using the same original recipient Gaussians in all four vectors.
Apply (FJT.3)--(FJT.4) with $k=2d$ and partition its tilted
means as $\mu_1,\mu_2,n_1,n_2$ and its tilted covariance blocks
as $A_{\ell m},C_{\ell m},B_{\ell m}$.

Put
$$
s_\ell=\mu_\ell\cdot n_\ell+
                            \operatorname{tr}C_{\ell\ell},
$$
$$
\begin{split}
q_{12}={}&s_1s_2+\mu_1^{\mathsf T}B_{12}\mu_2
 +n_1^{\mathsf T}A_{12}n_2
 +\mu_1^{\mathsf T}C_{21}^{\mathsf T}n_2
 +n_1^{\mathsf T}C_{12}\mu_2\\
&+\operatorname{tr}(A_{12}B_{12}^{\mathsf T})
 +\operatorname{tr}(C_{12}C_{21}).
\end{split}
\tag{FJT.12}
$$
The exact two-force contraction is
$$
\begin{split}
&\mathbb E_J\big\{
 K(D_1)K(D_2)
 [(D_1\cdot\eta_1)H_1-\pi_1]
       \cdot[(D_2\cdot\eta_2)H_2-\pi_2]\big\}\\
&\quad=\kappa_{12}\mathcal T_{12},\\
&\mathcal T_{12}=(H_1\cdot H_2)q_{12}
 -(H_1\cdot\pi_2)s_1-(\pi_1\cdot H_2)s_2
 +\pi_1\cdot\pi_2.
\end{split}
\tag{FJT.13}
$$
Here $\kappa_{12}$ is the mass (FJT.3) for the joint two-edge
position vector. When $j=k$, all the original and tilted blocks
are allowed to be singular; the same formula applies.
:::

:::{prf:proof}
Condition on the complete triple of coupled plans. Let the distinct root
indices among $i,j,k$ index the independent standard $d$-dimensional
recipient Gaussians. For each edge use the row with coefficient $I_i$
at index $i$ and coefficient $-I_j$ or $-I_k$ at its environment.
Use instead $j_i,-j_j,-j_k$ for the differential rows. If $E_D$
and $E_\eta$ denote these two-row coefficient matrices, then
$$
V=\sigma_J^2(E_DE_D^{\mathsf T})\otimes I_d,\quad
T=\sigma_J^2(E_DE_\eta^{\mathsf T})\otimes I_d,\quad
W=\sigma_J^2(E_\eta E_\eta^{\mathsf T})\otimes I_d.
\tag{FJT.14}
$$
In particular this construction uses two distinct jitter variables,
rather than three, when $j=k$. Their mean vectors are the actual
source and source-differential edge differences. Gaussian tilting
with $K(D_1)K(D_2)$ is precisely the $2d$-dimensional kernel
in (FJT.3).

Under the tilted Gaussian expand
$(D_1\cdot\eta_1)(D_2\cdot\eta_2)$ about its means.
The product of the two means is $s_1s_2$. The covariance of
the centered linear parts gives the four vector contractions in
(FJT.12). The covariance of the two centered bilinear parts
gives the two trace contractions there. The linear/bilinear
cross terms vanish by odd centered Gaussian moments.
For the fourth centered moment the three Gaussian pairings can
be verified by differentiating its Gaussian moment generating
function $\exp(u^{\mathsf T}\Sigma u/2)$ four times.
Subtracting the product of the two bilinear means removes the
within-edge pairing, leaving precisely the two displayed traces.
This proves $q_{12}$.

Now expand the two vector force factors. Their scalar product
is $(H_1\cdot H_2)(D_1\cdot\eta_1)(D_2\cdot\eta_2)$,
minus the two displayed linear products and plus
$\pi_1\cdot\pi_2$. The fixed plan velocities can be left outside
their jitter expectations. Formula (FJT.13) follows from
$q_{12}$ and $s_\ell$.
:::

:::{prf:theorem} Entire first-provider contribution with no remaining jitter expectation
:label: thm-fjt73-first-account

In the valid source-plan coupling (FJT.1)--(FJT.2), the complete
first-provider term $\mathfrak J_1$ of (DBL.4) is exactly
$$
\mathfrak J_1
 =a\,\mathbb E_{\mathrm{pair\ plans}}[\kappa\mathcal L]
  +a^2A_p\,
     \mathbb E_{\mathrm{triple\ plans}}
                          [\kappa_{12}\mathcal T_{12}].
\tag{FJT.15}
$$
For finite arrays the expectations include the exact normalized
$N^{-2}$ pair sum and $N^{-3}$ triple sum, with zero self edges
removed and coincident environments retained. For population
laws the two environment roots in the triple are conditionally
independent copies given the query root. Formula (FJT.15) has no
first-jitter Gaussian tail remainder or factorized moment assumption.
It remains an exact signed physical account rather than a claimed
uniform negative bound.
:::

:::{prf:proof}
The first linear term is exactly (FJT.9) averaged over coupled
plans. For the force square use the actual root force
$$
F_{1,i}=\mathbb E_j\left\{
 K(X_i-X_j)[((X_i-X_j)\cdot(r_i-r_j))(P_i-P_j)
                                      -(p_i-p_j)]\right\}.
\tag{FJT.16}
$$
In the finite array $\mathbb E_j$ is the actual normalized
uniform-index sum. In the population field it is the independent
environment integral, with the query root fixed. Squaring its
norm introduces two such environment integrals sharing that query.
The product is exactly the integrand of (FJT.13).
Consequently its expectation over the actual jitters is the
triple term in (FJT.15). A finite squared sum includes $j=k$,
so deleting that diagonal or assigning it independent jitters
would change the force square. Formula (FJT.14) retains it.

All changes of conditioning and integration are legitimate.
The gradient maximum $K(D)|D|\le e^{-1/2}$, the velocity
bound $|H|\le8$ and the square-integrable differentials imply
that each force in (FJT.16) is in $L^2$. In the present
source-plan register, the source means are bounded and the
remaining displacements are affine Gaussian with bounded
coefficients, so the conditional products also have integrable
polynomial dominators. Cauchy--Schwarz bounds the absolute
triple products by those $L^2$ norms. Thus Fubini applies to
the exact square before and after conditioning. This proves
(FJT.15) for actual finite sums and population integrals.
:::

(sec-fjt73-sign)=
## 5. A nonconstant-velocity signed condition and the remaining full block

:::{prf:corollary} Inward source velocities with matched copy statuses
:label: cor-fjt73-inward-source

Suppose a comparison path satisfies, for almost every coupled pair of
complete plans,
$$
j_i=j_j=0,\qquad
H=-\lambda\Delta,\qquad \pi=-\lambda e,
\qquad |\Delta|^2\le 1+\sigma_J^2(I_i^2+I_j^2),
\tag{FJT.17}
$$
with a fixed $\lambda\in[1/20,1/2]$. Then its entire
linear first-provider pair contribution has the proved sign
$$
\kappa\mathcal L
 =\kappa\,\lambda(A_r-A_p\lambda)
       \left[|e|^2-\frac{(\Delta\cdot e)^2}{\Gamma}\right]\le0.
\tag{FJT.18}
$$
Matched copy statuses can be either zero or one at each root;
all present recipient Gaussians are still integrated in (FJT.18).
For example source means in the ball of radius $1/2$, with
$P_\theta=-\lambda S_\theta$ and $p=-\lambda e$, satisfy
the pair-size condition for every pair. The velocities may vary
within the source law. This is a sign criterion on the actual
plan-fixed quantities, not a conclusion that arbitrary accepted
component/revival plans satisfy it or remain in this class.
:::

:::{prf:proof}
Matched statuses give $w=k=0$, hence
$n=e$, $\chi=\omega=0$, $s=(\Delta\cdot e)/\Gamma$
and $h=es$. Substitute $H=-\lambda\Delta$ and
$\pi=-\lambda e$ into (FJT.9) to obtain (FJT.18).
Cauchy--Schwarz and (FJT.17) make its bracket nonnegative.
The certified coefficient bounds give
$$
A_r-A_p/20<.0399395-.92464/20
                   =-.0062925<0.
$$
Thus $A_r-A_p\lambda<0$ throughout the stated interval.
The ball example has $|\Delta|\le1$ and $\Gamma\ge1$.
No individual jitter realization is required to lie in that ball:
only the source means obey this condition.

The signed condition has actual nonconstant-velocity examples
at the preparation interface. In particular the two-slot all-alive
symmetric inward inputs with source means $\pm ze_1$,
$z\in[1/5,3/10]$, and original velocities
$\mp\lambda ze_1$, have tied raw reward and tied nonself
diversity measurements. Their accepted gate is zero and their
components are singletons, so their prepared velocities are
the unchanged original velocities and $I_i=0$.
Interpolation within this family obeys (FJT.17).
This verifies nonemptiness using the actual source law; it
does not claim a general copied/revived preparation has the same
velocity relation. The present corollary proves the linear sign
only. The force square in (FJT.15) still has its own positive
contribution.
:::

:::{prf:remark} Exact remaining absorption and alive-law interface
:label: rem-fjt73-remaining

The exact general signed identity of research68 now has the entirely
consumed first-jitter form
$$
\begin{split}
\mathbb E Q_\beta(\dot x^+,\dot v^+)-\mathbb E Q_\beta(r,p)
={}&-\mathfrak D_H
 +a\mathbb E_{\mathrm{pair\ plans}}[\kappa\mathcal L]\\
&+a^2A_p\mathbb E_{\mathrm{triple\ plans}}
                           [\kappa_{12}\mathcal T_{12}]
 +\mathfrak J_2-\mathfrak C.
\end{split}
\tag{FJT.19}
$$
Here $\mathfrak J_2$ is the actual second-provider tensor and
force square in (DBL.5), and $\mathfrak C$ is the actual
correlated cap loss, with the conditional coercivity proved in
research69. The new formula does not separate the noisy first
graph from its resulting $U$ inside these second-stage terms.

A general physical differential rate $\varepsilon>0$ requires
the independently verified inequality
$$
\begin{split}
&a\mathbb E_{\mathrm{pair\ plans}}[\kappa\mathcal L]
 +a^2A_p\mathbb E_{\mathrm{triple\ plans}}
                         [\kappa_{12}\mathcal T_{12}]
 +\mathfrak J_2-\mathfrak C\\
&\qquad\le
\mathfrak D_H-\varepsilon\mathbb E Q_\beta(r,p)
\end{split}
\tag{FJT.20}
$$
uniformly over a proved comparison class. Formula (FJT.20)
is not asserted by this record. The exact tensor calculations
remove recipient-jitter uncertainty from the first-provider
term; they retain the plan-local covariance and mixed velocities
that must actually pass this inequality.

Even a proved kinetic inequality would additionally require
the actual preparation/source/component response, terminal
mark feedback, each own survival denominator, and a proved
invariant class or delayed block before implying default alive-law
mixing. The current result supplies neither a default convergence
rate nor a finite quasi-stationary mixing theorem.
:::

(sec-fjt73-checks)=
## 6. Retained inputs and verification boundary

:::{prf:remark} Frozen proof inputs and algebra checks
:label: rem-fjt73-checks

The input records were read and left unchanged at these SHA-256 values:

| Record | SHA-256 |
|---|---|
| 36 | `be58078b8f130f635658d5660c3edb9e826617f4eaedd77cdf8c3929b71136fe` |
| 61 | `bb2a271f9905c42a3edf9ac3881a7990bf77c8e264fdb3f37d01785670776425` |
| 68 | `ac2799245012b378edb22326fb4b9035b6dfa9aab47de5dba8fe66022e40af58` |
| 69 | `012f6f1f2671ddac98e10b4d63abca7662a5759deeb8267c475a399425a01a10` |

Equations (FJT.3)--(FJT.4) follow from nonsingular Gaussian
observation noise, even for singular original covariance.
Equations (FJT.8)--(FJT.13) follow from explicit second,
third and fourth Gaussian moment expansions in their proofs.
The verification must retain mismatched copy statuses,
zero-jitter endpoints and the $j=k$ force-square terms.
Numerical integration is only a diagnostic for these identities;
the proofs use their exact matrix and polynomial formulas.
No Monte Carlo value certifies (FJT.20).

An 18-node one-dimensional Gauss--Hermite diagnostic checked all
16 initial/final pair copy-status configurations at $\theta=0,.31,1$.
The 48 cases included zero-variance endpoints and both signs of the
cross covariance. The maximum absolute residual across the scalar
versions of (FJT.8), (FJT.9) and (FJT.11) was less than
$6\cdot10^{-17}$. Six two-edge cases, with distinct and coincident
environments at $\theta=0,.37,1$, checked (FJT.12)--(FJT.13);
their maximum absolute residual was less than $3\cdot10^{-17}$.
These floating diagnostics supplement the dimension-independent
moment proofs; they are not an interval or absorption certificate.
:::
