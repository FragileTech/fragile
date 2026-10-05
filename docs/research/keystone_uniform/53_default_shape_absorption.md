# A complete default kinetic estimate for arbitrary position shapes

(sec-dsa53-retained)=
## 1. Retained kernel and the larger physical comparison class

:::{prf:definition} Constant-velocity slices and physical transport
:label: def-dsa53-slices

Retain every default harmonic count primitive of
{prf:ref}`def-rfk-retained`, including $h=.04$, $\nu=.3$,
$d=3$, $L=2$, $V=2$, $V_c=4$, $\rho=\gamma=b_O=1$,
$\sigma_J=\sigma_x=.1$ and both full kinetic Gaussian innovations.
In particular
$$
t=.02,\quad c=e^{-.04},\quad b=t(1+c),\quad
a_x=1-tb,\quad a=.006,\quad q^2=(1-c^2)/2,\quad \beta=.04.
$$
Use the positive phase quadratic
$$
Q_\beta(r,p)=|r|^2+2\beta r\cdot p+|p|^2,
\qquad
G=\begin{pmatrix}1&\beta\\\beta&1\end{pmatrix}\otimes I_3.
$$
For population laws use its Wasserstein distance $W_{2,G}$.
For finite array laws use the squared ground cost
$$
Q_{\beta,N}(S,T)=\frac1N\sum_{i=1}^N
                 Q_\beta(x_i-x_i',v_i-v_i'),
$$
and denote the corresponding law distance by $W_{2,G,N}$.

The comparison class consists of two prepared physical laws with
arbitrary square-integrable position distributions and deterministic
within-law velocities $P_j=u_j$. Here $|u_j|\le V$ and $u_0,u_1$
may differ. In a finite array every row has its own law's same
deterministic $u_j$. Neither law is required to have a narrow position
cloud, a small centered displacement, or a small velocity moment.

Write $K_{\rm ph}$ and $\mathscr K_{N,\rm ph}$ for the physical
projections of the actual population and finite kinetic kernels.
Both own count fields, their actual noisy joint second-stage laws,
the native cap and terminal classification are retained. The discrete
terminal mark is excluded from these physical ground costs.
No preparation cost or current-alive normalization is included.
:::

(sec-dsa53-pair)=
## 2. A uniform conditional Gaussian bound for the actual second force

:::{prf:lemma} A pair bound before the fresh OU draw
:label: lem-dsa53-gaussian-pair

For every fixed $D\in\mathbb R^3$ and
$\Xi\sim N(0,2I_3)$, set
$$
Y=a_xD+tq\Xi,\qquad W=-ctD+q\Xi.
$$
Then
$$
\mathbb E\big[|Y|^2e^{-|Y|^2}|W|^2\big]
\le C_{\rm pair}<\frac7{40},
\tag{DSA53.1}
$$
where the explicit uniform coefficient is
$$
C_{\rm pair}=
\frac{16c^2t^2}{a_x^2e^2}
+\left(\frac{4c^2t^4q^2}{a_x^2}+2q^2\right)\frac6e.
\tag{DSA53.2}
$$
The estimate integrates every Gaussian outcome and is uniform in the
possibly unbounded fixed pair displacement $D$.
:::

:::{prf:proof}
The exact identity $D=(Y-tq\Xi)/a_x$ gives
$$
|W|^2\le2c^2t^2|D|^2+2q^2|\Xi|^2
\le\frac{4c^2t^2}{a_x^2}|Y|^2
+\left(\frac{4c^2t^4q^2}{a_x^2}+2q^2\right)|\Xi|^2.
$$
Use $\sup_{r\ge0}r^4e^{-r^2}=4/e^2$,
$\sup_{r\ge0}r^2e^{-r^2}=1/e$ and
$\mathbb E|\Xi|^2=6$. This proves (DSA53.1)--(DSA53.2)
without assuming that $Y$ and $W$ are independent.

For an exact rational upper calculation, the retained exponential
interval gives $a_x^2>.998$, $c^2<1$ and $q^2<.0392$.
The elementary bounds $e^{-1}<.368$ and $e^{-2}<.136$ give
$$
C_{\rm pair}<
\frac{16(.0004)(.136)}{.998}
+\left[\frac{4(.0004)^2(.0392)}{.998}+2(.0392)\right]6(.368)
<.175=\frac7{40}.
$$
All numbers in the last display are terminating rationals.
:::

:::{prf:lemma} The full own second spatial force on these slices
:label: lem-dsa53-second-force

Couple arbitrary inputs from {prf:ref}`def-dsa53-slices` and
share fresh original OU innovations, independent of the entire
prepared-input coupling. Put $r=X_1-X_0$, $p=u_1-u_0$,
$d_X=\|r\|_2$, and interpolate
$X_\theta=X_0+\theta r$, $u_\theta=u_0+\theta p$.
In a finite array these norms use normalized counting and the outer
expectation. The actual forces and position differential satisfy
$$
B_1=0,\qquad R=\dot y_\theta=a_xr+bp,\qquad
\|B_2\|_2^2\le2a_x^2 C_{\rm pair}d_X^2.
\tag{DSA53.3}
$$
In particular, at every interpolation point,
$$
a\|B_2\|_2\le B_*d_X,\qquad B_*=.003552.
\tag{DSA53.4}
$$
These are population and exact finite-array statements, uniformly in
$N$. They do not substitute a population RMS for a correlated random
empirical velocity moment.
:::

:::{prf:proof}
Every own first count Laplacian annihilates $u_\theta$.
Consequently $U_\theta=u_\theta$ and the first spatial force
$-L_{\dot k_1}u_\theta$ is identically zero. The actual stages are
$$
y_\theta=a_xX_\theta+bu_\theta+tq\xi,\qquad
w_\theta=c(u_\theta-tX_\theta)+q\xi.
$$
After conditioning on the entire prepared pair of roots, including
any source choices, copied jitter or component randomness, the fresh
$\xi,\xi'$ remain independent standard Gaussians in each own
marginal. Their differences give exactly the pair $(Y,W)$ in
(DSA53.1), with $D=X_\theta-X_\theta'$ and
$\Xi=\xi-\xi'$. No product assumption on the joint $(y,w)$ law
is made.

The common velocity displacement cancels from pair differences:
$R-R'=a_x(r-r')$. For the actual Gaussian count kernel,
$$
|\dot k_2|^2|w_\theta-w_\theta'|^2
\le a_x^2|r-r'|^2
       |y_\theta-y_\theta'|^2e^{-|y_\theta-y_\theta'|^2}
       |w_\theta-w_\theta'|^2.
$$
Apply Jensen to the actual environment integral defining $B_2$,
then condition on the complete prepared pair before integrating both
fresh OU noises. The uniform pair bound gives
$$
\|B_2\|_2^2
\le a_x^2C_{\rm pair}\mathbb E|r-r'|^2
\le2a_x^2C_{\rm pair}\mathbb E|r|^2.
$$
Here the independent copy is a copy of the entire coupled input,
so correlations between $r$, its source and either actual provider
are retained.

For a finite array, rowwise Jensen uses the actual denominator $N$.
The self term is zero. For every $i\ne j$ the conditional noise
difference is $N(0,2I_3)$, and
$$
\frac1{N^2}\sum_{i,j}|r_i-r_j|^2
=2\left[\frac1N\sum_i|r_i|^2
                         -\left|\frac1N\sum_i r_i\right|^2\right].
$$
The same conditional bound therefore proves (DSA53.3) after outer
expectation, with no independent-row assertion about the second
graph. Finally $a_x<1$, $2C_{\rm pair}<.35<.592^2$ and
$a(.592)=.003552$ prove the non-strict bound (DSA53.4), including
zero displacement.
:::

(sec-dsa53-absorption)=
## 3. Complete signed absorption with arbitrary shape change

:::{prf:lemma} Complete cap majorant on the constant-velocity slices
:label: lem-dsa53-majorant

At every interpolation point in
{prf:ref}`lem-dsa53-second-force`, let $A_2=I-aL_2$ be the
actual noisy second operator and let $D_C=DC_V(z_\theta)$.
With
$$
R=a_xr+bp,\qquad W=c(p-tr),\qquad
Z_0=A_2W-tR,\qquad T=Z_0+\beta R,
$$
the complete actual velocity differential is
$C=D_C(Z_0+aB_2)$, and
$$
\mathbb E Q_\beta(R,C)
\le Q-.001d_X^2-.04d_P^2,
\qquad Q=\mathbb E Q_\beta(r,p),\quad d_P=|p|.
\tag{DSA53.5}
$$
The same assertion holds on the normalized finite-array Hilbert
space and after expectation over its complete actual noise.
:::

:::{prf:proof}
The positive contraction sector $0\le D_C\le I$ of the actual
radial cap gives the pointwise Hilbert-space inequality
$$
Q_\beta(R,C)\le\|R\|^2+\|Z_0+aB_2+\beta R\|^2.
\tag{DSA53.6}
$$
This is the complete cap-square identity used in
{prf:ref}`thm-rfk-two-count-principal`; it does not factor any
averaged cap Jacobian from a displacement.

First bound its principal part. Set $e_2=(I-A_2)W$ and
$\delta=a/(2-a)$. Since $A_2$ is self-adjoint with
$(1-a)I\le A_2\le I$, the exact second-alignment spectral
inequality gives
$$
\|R\|^2+\|T\|^2
\le\|R\|^2+\|W+(\beta-t)R\|^2
                       +\delta(\beta-t)^2\|R\|^2.
$$
Use {prf:ref}`lem-rfk-anisotropic-majorant` and
$\|R\|^2\le\|r\|^2+\|p\|^2$ to obtain
$$
\|R\|^2+\|T\|^2
\le Q-.00149d_X^2-.0721d_P^2.
\tag{DSA53.7}
$$
The negative second-alignment form has been absorbed using its
actual conductances; no favorable sign of $B_2$ is presumed.

Because $p$ is constant across roots, $A_2p=p$.
Writing $f=(\beta-t)a_x-ct$ and $g=c+(\beta-t)b$ gives
$$
T=(fI+ctaL_2)r+gp.
$$
The retained rational intervals give
$0<f+cta<.0009$ and $0<g<.962$. Thus
$\|T\|_2\le.0009d_X+.962d_P$. Restore the exact full force
in (DSA53.6), and use (DSA53.4): its additional cost is at most
$$
[2(.0009)B_*+B_*^2]d_X^2+2(.962)B_*d_Xd_P.
$$
Young's inequality bounds the last term by
$$
.0004d_X^2+\frac{(.962B_*)^2}{.0004}d_P^2.
$$
The two remaining strict rational coefficient margins are
$$
.00149-2(.0009)B_*-B_*^2-.0004
=.001070989696>.001,
$$
$$
.0721-\frac{(.962B_*)^2}{.0004}
=.04290986745856>.04.
$$
This proves (DSA53.5), including both actual count stages and the
actual uncapped joint second force. The full OU integration occurred
before this coefficient comparison.
:::

:::{prf:theorem} Optimal physical law contraction for arbitrary position shapes
:label: thm-dsa53-physical-shapes

For every pair in {prf:ref}`def-dsa53-slices`,
$$
W_{2,G}(K_{\rm ph}\lambda_0,K_{\rm ph}\lambda_1)^2
\le\frac{1039}{1040}W_{2,G}(\lambda_0,\lambda_1)^2.
\tag{DSA53.8}
$$
For every $N\ge1$, the actual finite physical kernels satisfy
$$
W_{2,G,N}(\mathscr K_{N,\rm ph}\Lambda_0,
                         \mathscr K_{N,\rm ph}\Lambda_1)^2
\le\frac{1039}{1040}W_{2,G,N}(\Lambda_0,\Lambda_1)^2.
\tag{DSA53.9}
$$
Their position laws may have unrelated shapes and identical means.
There is no centered-displacement restriction and no finite-$N$
error. These are one-update kinetic physical law estimates.
:::

:::{prf:proof}
Start with any input coupling, not necessarily a source-plan coupling,
and adjoin shared fresh actual OU and final Gaussian innovations,
independent of the input. Each own marginal retains its original
innovation laws and its own count fields. Since
$Q\le(1+\beta)(d_X^2+d_P^2)$, (DSA53.5) gives
$$
\mathbb E Q_\beta(R,C)
\le\left(1-\frac{.001}{1.04}\right)Q
=\frac{1039}{1040}Q.
$$
The physical endpoint difference is the integral of these actual
interpolation derivatives. The same final Gaussian cancels from its
position difference. Jensen for $G$ and integration in $\theta$
therefore give the displayed bound on this valid output coupling.

The conditional Gaussian bound controls every force in $L^2$
uniformly along the interpolation. It justifies the differentiated
provider integrals and their $L^2$ integration even for unbounded
square-integrable positions. Finite arrays have the same bound after
conditioning on the input and taking outer expectation. The cap is
$C^1$ with its stated positive contraction Jacobian; all original
noise outcomes are included.

The input coupling was arbitrary. Taking its infimum proves
(DSA53.8)--(DSA53.9) in the correct transport direction. No optimal
input plan is required to share the earlier source jitter. It only
needs the original kinetic innovations to be fresh, which is an
actual kernel property. Both input and output physical laws have
finite second moments, as required for these distances.
:::

:::{prf:corollary} Stronger same-velocity shape estimate
:label: cor-dsa53-same-velocity

If $u_0=u_1$, either squared transport factor in
(DSA53.8)--(DSA53.9) may be replaced by $1997/2000$.
:::

:::{prf:proof}
Here $p=0$, $R=a_xr$ and
$\|T\|_2\le.0009d_X$. Directly in the complete cap majorant,
$$
\mathbb E Q_\beta(R,C)
\le\big[a_x^2+(.0009+B_*)^2\big]d_X^2.
$$
The exact rational upper calculation is
$$
.999216^2+(.0009+.003552)^2
=.99845243496<.9985=\frac{1997}{2000}.
\tag{DSA53.10}
$$
The input phase cost is exactly $d_X^2$ for every coupling.
Integration and the unrestricted input infimum give the claim.
:::

(sec-dsa53-actual)=
## 4. Actual preparations and a centered shape example

:::{prf:corollary} A nonempty actual class beyond common-translation comparisons
:label: cor-dsa53-actual-preparations

Any well-defined actual preparation whose original frozen slot
velocities are all a deterministic $u$ belongs to the slice class,
including preparations with active copying, mandatory revival and
uncut copied-recipient jitter. The original configured fitness and
normalizers are unchanged. Two such preparations may have different
original common velocities.

The class contains distinct zero-mean prepared position laws, so it
is not contained in the centered-displacement class of
{prf:ref}`def-rsa-class`.
:::

:::{prf:proof}
Cloning only changes the frozen position source. Each original-slot
component velocity equals $u$, its mean equals $u$ and every
deviation from that mean is zero. Its actual full Haar readout is
therefore $u$ at every row, irrespective of the source, component or
jitter outcome. The source-box Gaussian envelope gives its prepared
position law a finite second moment. The source plan may be sampled
using every original reward/diversity measurement and gate; none is
altered in applying the theorem. In particular the genuine active
preparations constructed in
{prf:ref}`cor-rsa-narrow-cloud-class` are included, but narrowness
is not a hypothesis of the kinetic estimate.

For an exact centered shape example, take a population with entering
alive fraction $\alpha\in(0,1)$, every alive position equal to
zero, every dead retained position outside the box, and every original
velocity zero. All alive reward and measured-diversity values tie,
so every alive accepted-copy gate is zero. Every dead root is revived
from position zero and receives its full original Gaussian jitter.
The prepared physical law is exactly
$$
\lambda_\alpha=
\left[\alpha\delta_0+(1-\alpha)N(0,\sigma_J^2I_3)\right]
\otimes\delta_0^{\rm velocity}.
\tag{DSA53.11}
$$
The mandatory components are finite stars: only dead roots have an
accepted outgoing edge, and their eligible donors are alive. Their
Poisson incoming parameter is finite for every fixed $\alpha>0$;
the constant original velocities also make their readout identically
zero. Thus this is an actual prepared population law.

For distinct $\alpha_0,\alpha_1$, these laws have the same phase
mean and different position variances
$3(1-\alpha_j)\sigma_J^2$. They are distinct and their optimal
transport distance is positive. Every coupling has mean displacement
zero, so the condition in {prf:ref}`def-rsa-class` would force
the laws to coincide. Nevertheless (DSA53.8) and the stronger
same-velocity factor apply to this exact shape change.
Finite arrays with a positive number of alive consensus roots and
zero original velocities give the corresponding exact Gaussian
revival preparations; their estimate is uniform in their size.
:::

(sec-dsa53-scope)=
## 5. Precise remaining full-law obligations

:::{prf:remark} One complete shape consumer and its endpoint
:label: rem-dsa53-scope

This note closes both spatial count forces on deterministic
within-law velocity slices: the first is identically zero, and the
actual second force is absorbed after its full conditional Gaussian
integration. It proves optimal population and finite-array physical
kinetic transport contraction for arbitrary position shapes at the
default viscosity. The second-force estimate retains the exact joint
$(y,w)$ law and every OU realization. It does not replace a local
displacement/velocity product by an unconditional RMS product or
factor an averaged cap Jacobian from a correlated displacement.

General nonconstant within-law prepared velocities still introduce
the original first spatial force and additional pair terms in the
second force. The present conditional identity does not bound those
terms by the same constants. It supplies no signed absorption for
that remaining general class. Nor is the constant-velocity class
preserved by a kinetic step with full independent OU noise.

The physical ground cost omits terminal marks. Conditioning on each
own current-survival event, sampling only alive rows, comparing a
preparation to its incoming swarm, and iterating the complete active
update are separate obligations. No default global rate, population
attraction or finite survivor/QSD mixing is asserted by this
one-update result. The exact new endpoint is a nonempty actual,
arbitrary-position-shape kinetic law class with a size-independent
rate and no additive floor.
:::
