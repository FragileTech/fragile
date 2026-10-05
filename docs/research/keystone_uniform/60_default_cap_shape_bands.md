# Full cap-square absorption for nonconstant velocity bands

(sec-csb-register)=
## 1. Retained default law and the new comparison class

:::{prf:definition} Prepared velocity bands and physical law transport
:label: def-csb-register

Retain the actual harmonic count stages and all primitives of
{prf:ref}`def-rfk-retained`. In particular $d=3$, $h=.04$,
$\nu=.3$, $t=.02$, $c=e^{-.04}$, $b=t(1+c)$,
$a_x=1-tb$, $m=1-t^2$, $a=t\nu=.006$,
$q^2=(1-c^2)/2$ and $\beta=.04$.
Both original kinetic Gaussians, the own joint second provider,
the native cap and terminal classification remain in the algorithm.

Use the physical population and finite-array transport conventions
of {prf:ref}`def-dsa53-slices`. For each of two prepared laws,
suppose there is a deterministic $u_j$ with $|u_j|\le V=2$ and
$$
|P_j-u_j|\le\delta_v=\frac1{200}
\quad\text{almost surely}.
\tag{CSB.1}
$$
For a finite array this bound holds at every row in every input
realization. Its velocities may be nonconstant and depend on its
positions and full source/component plan. The positions have an
arbitrary square-integrable law. The centers $u_0,u_1$ may differ.

This is a pointwise prepared-velocity hypothesis, not an averaged
post-burn moment. The physical metric omits terminal marks and any
alive normalization. No invariance of (CSB.1) is assumed.
:::

(sec-csb-pair)=
## 2. Exact fresh-noise pair control with nonconstant first velocities

:::{prf:lemma} Conditional pair coefficient for both actual providers
:label: lem-csb-gaussian-pair

Let $D,H\in\mathbb R^3$ be fixed, $|H|\le2\delta_v=.01$,
and $\Xi\sim N(0,2I_3)$. Set
$$
Y=a_xD+bH+tq\Xi,\qquad W=c(H-tD)+q\Xi.
$$
Then
$$
\mathbb E[|Y|^2e^{-|Y|^2}|W|^2]
\le C_v<.18,
\tag{CSB.2}
$$
where the complete explicit coefficient is
$$
C_{\rm noise}
=\frac{8c^2t^2}{a_x^2e^2}
                         +\frac{4dq^2m^2}{a_x^2e},
\qquad
C_v=\frac{51}{50}C_{\rm noise}
              +\frac{51c^2(2\delta_v)^2}{a_x^2e}.
\tag{CSB.3}
$$
The estimate is uniform in the unbounded $D$ and includes every
original Gaussian outcome.
:::

:::{prf:proof}
The exact stage identity is
$$
W=\frac c{a_x}H-\frac{ct}{a_x}Y+\frac{qm}{a_x}\Xi.
$$
Apply $|A+B|^2\le(1+\eta)|A|^2+(1+\eta^{-1})|B|^2$
with $\eta=1/50$, $A=-(ct/a_x)Y+(qm/a_x)\Xi$ and
$B=(c/a_x)H$. Bound the two terms of $A$ by twice their
squares, then use the maxima $\sup r^4e^{-r^2}=4/e^2$,
$\sup r^2e^{-r^2}=1/e$ and $\mathbb E|\Xi|^2=2d=6$.
This gives (CSB.2)--(CSB.3) without treating $Y,W$ as independent.

The retained intervals $a_x^2>.998$, $c^2,m^2<1$,
$q^2<.0392$, $e^{-1}<.368$ and $e^{-2}<.136$ give the
entirely rational upper bound
$$
C_v<\frac{51}{50}
 \left[\frac{8(.0004)(.136)}{.998}
                         +\frac{12(.0392)(.368)}{.998}\right]
+\frac{51(.0001)(.368)}{.998}<.18.
$$
Every local velocity term is bounded by the stated pointwise $H$
bound before Gaussian integration. No unconditional velocity moment
has been factored from a correlated displacement.
:::

:::{prf:lemma} Full first and second spatial-force budgets
:label: lem-csb-force-budgets

Start with any coupling of the prepared inputs in (CSB.1), set
$r=X_1-X_0$, $p=P_1-P_0$, $d_X=\|r\|_2$ and
$d_P=\|p\|_2$, and interpolate both inputs. Adjoin shared fresh
original kinetic innovations, independent of the input coupling.
Use the actual operators and complete forces of
{prf:ref}`lem-rfk-own-differential`. At every interpolation point,
$$
\|B_1\|_2\le.009d_X,\qquad
\|B_2\|_2\le.6\|R\|_2,
\qquad R=a_xr+b(A_1p+aB_1).
\tag{CSB.4}
$$
These are population and exact finite-array bounds, uniformly in
size. Their norms include all outer input and original-noise
expectations.
:::

:::{prf:proof}
Each interpolated own law has its velocities in
$B(u_\theta,\delta_v)$. Therefore every pair velocity
difference is at most $2\delta_v$ pointwise. The actual first
Gaussian kernel has $|\nabla K|\le\ell=e^{-1/2}$.
Jensen in its environment integral gives
$$
\|B_1\|_2^2
\le(2\delta_v\ell)^2\mathbb E|r-r'|^2
\le8\delta_v^2\ell^2d_X^2.
$$
The coefficient $2\sqrt2\delta_v\ell<.009$ follows from
$\sqrt2<1.415$ and $\ell<.607$. This is a pointwise
velocity-spread bound, not an averaged source-product estimate.

The first count step is a convex average because $a=.006<1$.
Consequently its actual $U_\theta$ also belongs to
$B(u_\theta,\delta_v)$. Conditional on the whole prepared
pair, its two values $U_\theta,U_\theta'$ and their difference
are fixed before OU noise. The actual second-stage pair is exactly
the $(Y,W)$ of (CSB.2), with
$D=X_\theta-X_\theta'$ and
$H=U_\theta-U_\theta'$. It satisfies $|H|\le2\delta_v$.
The complete position differential $R$ includes $aB_1$ but
is still fixed before that fresh noise.

The actual second kernel derivative obeys
$$
|\dot k_2|^2|w-w'|^2
\le|R-R'|^2|y-y'|^2e^{-|y-y'|^2}|w-w'|^2.
$$
Jensen first, followed by conditioning on the whole prepared pair
and then integrating both OU draws, yields
$$
\|B_2\|_2^2\le C_v\mathbb E|R-R'|^2
                         \le2C_v\|R\|_2^2\le.36\|R\|_2^2.
$$
This retains the original source, velocity and graph correlations,
including the dependence of $U,R$ on the first provider.

For finite arrays, Jensen uses denominator $N$, the self term is
zero, and every distinct pair of OU draws has difference
$N(0,2I_3)$. The exact normalized pair identity is
$$
\frac1{N^2}\sum_{i,j}|R_i-R_j|^2
=2\left[\frac1N\sum_i|R_i|^2
                         -\left|\frac1N\sum_iR_i\right|^2\right].
$$
The same identity with $r$ proves its first bound. Conditional
pair integration and outer expectation therefore prove (CSB.4)
without independent second-graph rows or a population-moment
replacement for a random empirical quantity.
:::

(sec-csb-cap)=
## 3. Complete oriented cap-square account

:::{prf:theorem} Both spatial forces absorbed on nonconstant velocity bands
:label: thm-csb-cap-absorption

For the complete actual physical derivative at every interpolation
point in {prf:ref}`lem-csb-force-budgets`,
$$
\mathbb E Q_\beta(\dot x^+,\dot v^+)
\le Q-.001d_X^2-.04d_P^2,
\qquad Q=\mathbb E Q_\beta(r,p).
\tag{CSB.5}
$$
The assertion is population and exact finite-array, with normalized
array cost. It includes the negative count forms, both full spatial
forces and the actual native cap.
:::

:::{prf:proof}
Set $p_1=A_1p$ and define the full principal variables
$$
R_0=a_xr+bp_1,\quad W_0=c(p_1-tr),\quad
Z_0=A_2W_0-tR_0,\quad T_0=Z_0+\beta R_0.
$$
The exact complete differential is
$$
R=R_0+E_R,\qquad
Z+\beta R=T_0+E_T,
$$
$$
E_R=baB_1,\qquad
E_T=a[cA_2+(\beta-t)bI]B_1+aB_2.
$$
Here $B_2$ uses the complete $R$, including $E_R$.
It is never evaluated on $R_0$ alone.

Both actual count operators are self-adjoint with
$(1-a)I\le A_j\le I$, without a commutation requirement.
The proof of {prf:ref}`thm-rfk-two-count-principal` gives
the principal cap-square majorant
$$
\|R_0\|_2^2+\|T_0\|_2^2
\le Q-.00149d_X^2-.0721d_P^2.
\tag{CSB.6}
$$
For clarity, apply the second alignment spectral inequality with
$\delta=a/(2-a)$ to obtain its correction
$\delta(\beta-t)^2\|R_0\|_2^2$, and then
{prf:ref}`lem-rfk-anisotropic-majorant` with $p_1$.
The first alignment spectral inequality adds at most
$\delta\beta^2\|r\|_2^2$ to the input phase cost.
Use $\|R_0\|_2^2\le\|r\|_2^2+\|p_1\|_2^2$ and
$\|p_1\|_2\ge(1-a)\|p\|_2$.
The two remaining coefficients strictly exceed $.00149$ and
$.0721$. Thus (CSB.6) bounds the majorant itself, rather than
inferring its bound from a smaller capped cost.

Writing $f=(\beta-t)a_x-ct$ and
$g=c+(\beta-t)b$ gives the oriented identity
$$
T_0=(fI+ctaL_2)r+[cA_2+(\beta-t)bI]A_1p.
$$
The retained intervals imply $0<f+cta<.0009$ and $g<.962$.
Therefore
$$
\|R_0\|_2\le.999216d_X+.039216d_P,
\qquad \|T_0\|_2\le.0009d_X+.962d_P.
$$
Keeping this phase orientation, rather than replacing the principal
phase norm by a scalar force norm, is the needed cancellation.

By (CSB.4), the complete force budgets are
$$
\|E_R\|_2\le\alpha d_X,\qquad
\|E_T\|_2\le e_Xd_X+e_Pd_P,
$$
$$
\alpha=.00000212,\qquad e_X=.00365,\qquad e_P=.000142.
\tag{CSB.7}
$$
Indeed $ba(.009)<\alpha$, while
$a\|B_2\|_2\le.0036[(a_x+\alpha)d_X+bd_P]$.
The first term of $E_T$ contributes at most
$a(.962)(.009)d_X$. Their sum has position coefficient below
$.00365$ and velocity coefficient below $.000142$.

For the actual cap Jacobian $D_C=DC_V(z_\theta)$,
$0\le D_C\le I$ gives the complete pointwise identity bound
$$
Q_\beta(R,D_CZ)\le\|R\|^2+\|Z+\beta R\|^2.
$$
This remains valid with the actual correlated root $z_\theta$.
It neither averages $D_C$ nor presumes its independence from
the force or displacement. Restore both full forces in its two
squares. In addition to (CSB.6), the cost is at most
$$
A_Xd_X^2+A_Pd_P^2+2A_{XP}d_Xd_P,
$$
where the completely rational coefficients are
$$
\begin{aligned}
A_X&=2(.999216)\alpha+\alpha^2+2(.0009)e_X+e_X^2,\\
A_P&=2(.962)e_P+e_P^2,\\
A_{XP}&=.039216\alpha+.0009e_P+.962e_X+e_Xe_P.
\end{aligned}
$$
Young's inequality gives
$2A_{XP}d_Xd_P\le.0004d_X^2+A_{XP}^2d_P^2/.0004$.
Direct exact rational calculation yields
$$
.00149-A_X-.0004=.0010658708196656>.001,
$$
$$
.0721-A_P-A_{XP}^2/.0004>.0409>.04.
$$
This proves (CSB.5) with both complete actual forces restored.
No favorable force sign or averaged-Jacobian factorization was used.
:::

(sec-csb-transport)=
## 4. Optimal physical law consequence and actual nonempty preparations

:::{prf:corollary} Size-independent optimal physical law estimate
:label: cor-csb-optimal-law

Every pair of prepared population laws satisfying (CSB.1) obeys
$$
W_{2,G}(K_{\rm ph}\lambda_0,K_{\rm ph}\lambda_1)^2
\le\frac{1039}{1040}W_{2,G}(\lambda_0,\lambda_1)^2.
\tag{CSB.8}
$$
For every $N\ge1$, the exact physical finite-array kernels obey
the analogous estimate with $W_{2,G,N}$ and
$\mathscr K_{N,\rm ph}$. There is no additive particle floor.
:::

:::{prf:proof}
The band hypothesis holds pointwise under every input coupling.
Adjoin shared fresh kinetic innovations to any such coupling.
Since $Q\le1.04(d_X^2+d_P^2)$, (CSB.5) gives derivative
cost at most $(1039/1040)Q$. Integration of the complete physical
derivative and Jensen in $G$ give this bound on a valid output
coupling. The conditional Gaussian bounds control the force
derivatives in $L^2$ along the interpolation. They justify both
the population differentiated provider integrals and their
integration for square-integrable positions. The finite conditional
pair proof has the same justification after outer expectation.

Finally take the infimum over all input couplings. The prepared
source jitter need not be shared by those plans; only subsequent
kinetic innovations are fresh. Consequently this is an optimal
physical law estimate, with the same correct infimum direction as
{prf:ref}`thm-dsa53-physical-shapes`.
:::

:::{prf:corollary} Original component readout supplies nonconstant examples
:label: cor-csb-actual-bands

Suppose all original frozen velocities, alive and dead, belong to
$B(u,1/400)$, with their original stored cap respected. Every
well-defined actual preparation at $\alpha_{\rm col}=.5$
then satisfies (CSB.1). This includes actual nonconstant prepared
velocities, active cloning, mandatory revival and the full
copied-recipient jitter, at unchanged configured fitness parameters.
:::

:::{prf:proof}
On each actual component the original readout is
$$
P_i=\bar v_C+\alpha_{\rm col}O_C(v_i-\bar v_C).
$$
If every $|v_i-u|\le\varepsilon_v$, then
$|\bar v_C-u|\le\varepsilon_v$ and
$|v_i-\bar v_C|\le2\varepsilon_v$. Orthogonality gives
$$
|P_i-u|\le(1+2|\alpha_{\rm col}|)\varepsilon_v
=2\varepsilon_v\le1/200.
$$
This holds for every component size and every Haar realization.
Cloning changes only positions; original velocities are not copied.
Jitter is also positional. Thus both retain this bound exactly.

For explicit nonconstant population examples, use two positive-mass
types inside a small interior phase ball about $(x,u)$, with
unequal radial positions and distinct original velocities.
Choose the ball radius below $1/400$, below the positional boundary
margin and small enough that its actual alive accepted column is
at most $1/8$. Such a positive radius exists at any fixed positive
configured fitness powers: positive standardizer floors and bounded
logistic-power derivatives bound acceptance by a finite constant
times the phase-ball radius. This is the same local range proof as
{prf:ref}`cor-rsa-narrow-cloud-class`, now using the phase-ball
diameter in its diversity estimate.

The two cross-type measurement choices have the same symmetric
diversity distance. Their reward factors differ strictly, so a
positive-probability donor draw and gate cause genuine active
copying at the original positive powers. Each original type also
has positive probability of an isolated component: its outgoing
acceptance is below one and its incoming intensity is finite.
Its readout then equals its distinct original velocity. Therefore
the prepared velocity law is nonconstant, while the pathwise band
bound remains valid. Full Gaussian position tails are still present.
Finite arrays require no component-size restriction for the same
pathwise readout bound.
:::

(sec-csb-scope)=
## 5. Scope and the remaining general signed inference

:::{prf:remark} The new class does not close the delayed default law
:label: rem-csb-scope

This is a complete one-update physical kinetic estimate for
nonconstant prepared velocities and arbitrary position shapes.
It retains negative count-alignment forms and the exact cap-square
orientation. The uniform pair bounds hold after conditioning before
OU noise and integrate every outcome; they do not turn a post-burn
RMS into a bound on a correlated local product.

The pointwise radius $1/200$ is additional. The proved post-burn
RMS $.55$ does not imply it. An original noisy output is not
asserted to remain in this class. Already for $N=1$ at collapsed
zero input the actual precap velocity is $mq\xi$, so its capped
output has the full open velocity ball as support. A radius-$1/200$
band cannot contain that law. Thus the theorem cannot be iterated
by presuming class invariance or a burn-in entrance.

For general prepared velocity laws the actual pair
$H=U-U'$ is not bounded by $.01$. Its contribution to the
conditional pair coefficient is a local term proportional to
$|R-R'|^2|H|^2$, and the first spatial force likewise contains
$|r-r'|^2|P-P'|^2$. No averaged moment or present class estimate
discharges those general terms. Signed tensor feedback, an actual
previous-noise coupling, or a proved different transport drift is
still required to close them.

Preparation-to-input cost, terminal marks, each own conditional
alive readout, multistep signed provider response and nonlinear
attraction are also separate. No default global rate, stationary
population, QSD mixing or finite survivor convergence follows
from this class theorem. Its endpoint is a proved nonempty,
nonconstant-velocity physical kinetic law class with the original
default viscosity and a size-independent optimal transport bound.
:::
