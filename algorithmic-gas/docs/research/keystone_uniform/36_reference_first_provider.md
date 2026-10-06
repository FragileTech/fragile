# First count-provider feedback at the harmonic reference

(sec-rfp-record)=
## 1. Actual population carrier and conditional source law

:::{prf:definition} Reference first-provider comparison
:label: def-rfp-record

Retain the actual harmonic count-population record of
{prf:ref}`def-rvb-record`: $d=3$, $L=2$, $t=.02$, $\nu=.3$,
$\rho=\gamma=b_O=1$, $\sigma_J=\sigma_x=.1$, $V=2$, and
$\alpha_{\rm col}=.5$. Put $a=t\nu=.006$, $\ell=e^{-1/2}$,
$c=e^{-.04}$, $b=t(1+c)$, $m=1-t^2$, $a_x=1-tb$, and
$V_c=4$. The native smooth cap is $C_V(z)=Vz/(V+|z|)$.
The reference phase quadratic is
$Q_\beta(r,s)=|r|^2+2\beta r\cdot s+|s|^2$, $\beta=.04$.

A prepared root is $(X,P)$, where
$$
X=S+I J,\qquad S\in[-L,L]^d,\quad I\in\{0,1\},\quad
J\sim N(0,\sigma_J^2I_d),\quad |P|\le V_c.
\tag{RFP.1}
$$
Conditional on the complete discrete measurement/source/component plan
and its Haar marks, $S,I,P$ are fixed and $J$ is independent of that
plan. The velocity $P$ comes from the component map applied to the
original frozen slot velocities; donor velocities are not copied.
The preparation retains all current fitness statistics, eligible donor
probabilities, accepted edges and mandatory revival. In particular this
definition does not require the discrete plan to have independent root
and source velocities.

For its own prepared probability $\lambda$, use the actual count field
$$
K(r)=e^{-|r|^2/2},\qquad
a_\lambda(x)=\int K(x-y)\,\lambda(dy,dw),\qquad
M_\lambda(x)=\int K(x-y)w\,\lambda(dy,dw),
$$
$$
U_\lambda(X,P)=(1-a a_\lambda(X))P+aM_\lambda(X).
\tag{RFP.2}
$$
There is no row-degree denominator or degree-floor substitution.
Since $0\le a_\lambda\le1$ and $|M_\lambda|\le V_ca_\lambda$,
$|U_\lambda|\le V_c$. The formulas apply to the actual deterministic
population provider, not to a noisy finite empirical provider.

The post-burn comparisons below require the proved all-slot preparation
bound $\int |P|^2d\lambda\le r^2$, with $r=.55$, inherited from
{prf:ref}`thm-rvb-population-burn` and
{prf:ref}`lem-rvb-preparation-energy`. It is an averaged bound and is
not an individual-speed bound or an alive-normalized bound.
:::

(sec-rfp-source-products)=
## 2. Conditional source products without displacement independence

:::{prf:lemma} Fresh-jitter velocity products
:label: lem-rfp-source-products

Let a common actual prepared root law $\zeta$ satisfy (RFP.1) and
$\|P\|_2\le r$. Put
$$
X_2=\sqrt{d(L^2+\sigma_J^2)}=\sqrt{12.03},\qquad
X_{1,2}=mX_2+tV_c.
$$
For any convex interpolation of two deterministic first providers
supported in speed $V_c$, let $U_\theta$ be (RFP.2) and
$x_{1,\theta}=mX+tU_\theta$. Then
$$
\|X\|_2\le X_2,\qquad
\big\||X||P|\big\|_2\le X_2r,\qquad
\|x_{1,\theta}\|_2\le X_{1,2},\qquad
\big\||x_{1,\theta}||P|\big\|_2\le X_{1,2}r.
\tag{RFP.3}
$$
In particular, if deterministic field differences satisfy
$\|\Delta M_0\|_\infty\le A_0$ and
$\|\Delta a_0\|_\infty\le B_0$, the actual random perturbation
$H(X,P)=\Delta M_0(X)-P\Delta a_0(X)$ obeys
$$
\|H\|_2\le A_0+rB_0,\qquad
\big\||x_{1,\theta}||H|\big\|_2
\le X_{1,2}(A_0+rB_0).
\tag{RFP.4}
$$
:::

:::{prf:proof}
Condition on the complete plan in (RFP.1). Fresh recipient jitter
has zero mean and is independent of $P$, so
$$
\mathbb E(|X|^2|P|^2\mid\mathrm{plan})
=(|S|^2+I d\sigma_J^2)|P|^2
\le X_2^2|P|^2.
$$
Integration proves the second inequality in (RFP.3); the same
calculation without $|P|^2$ proves the first. Convex interpolations of
the provider fields retain $|U_\theta|\le V_c$, even though that
common root need not be an own-provider root at intermediate values.
Minkowski gives
$$
\big\||x_{1,\theta}||P|\big\|_2
\le m\big\||X||P|\big\|_2+tV_c\|P\|_2
\le X_{1,2}r.
$$
The unweighted version proves the third bound in (RFP.3). Finally
$|H|\le A_0+B_0|P|$. Apply Minkowski to this pointwise bound and
to its product with $|x_{1,\theta}|$. This proves (RFP.4) despite
the possible correlation of $H$ with $X$ and $x_{1,\theta}$.
:::

(sec-rfp-global-provider)=
## 3. Actual own-provider fields in a transport coupling

:::{prf:lemma} Count-field comparison with the actual donor velocity moment
:label: lem-rfp-provider-fields

Let $\pi$ couple two prepared probabilities $\lambda,\widetilde\lambda$
and write
$$
d_X^2=\int |X-\widetilde X|^2d\pi,\qquad
d_P^2=\int |P-\widetilde P|^2d\pi.
$$
If $\|\widetilde P\|_2\le r$, their actual fields satisfy
$$
\|\Delta a_0\|_\infty\le\ell d_X,\qquad
\|\Delta M_0\|_\infty\le d_P+\ell r d_X.
\tag{RFP.5}
$$
For two actual joint second-stage providers coupled with RMS differences
$d_y,d_w$ and second velocity moment at most $M_w^2$, the same argument
gives
$$
\|\Delta a_2\|_\infty\le\ell d_y,\qquad
\|\Delta M_2\|_\infty\le d_w+\ell M_wd_y.
\tag{RFP.6}
$$
These are joint-provider estimates. No independence between a provider's
landing position and its uncapped OU velocity is required.
:::

:::{prf:proof}
The Gaussian count kernel has $\|\nabla K\|_\infty=\ell$.
At each fixed query $x$, integrate
$$
|K(x-X)-K(x-\widetilde X)|\le\ell|X-\widetilde X|.
$$
Cauchy--Schwarz proves the first bound. Split the vector numerator
exactly as
$$
K(x-X)P-K(x-\widetilde X)\widetilde P
=K(x-X)(P-\widetilde P)
+[K(x-X)-K(x-\widetilde X)]\widetilde P.
$$
Use $K\le1$ on the first term and Cauchy--Schwarz on the second.
This gives $d_P+\ell r d_X$ uniformly in $x$. The proof for a
joint second-stage coupling is identical with $(X,P)$ replaced by
$(y,w)$. Its moment product is bounded by Cauchy--Schwarz on that
joint coupling, rather than an assumption about its factors.
:::

:::{prf:theorem} Burn-aware full-jitter first-provider feedback through the native cap
:label: thm-rfp-first-provider-cap

Let the common actual prepared root $\zeta$ satisfy
{prf:ref}`lem-rfp-source-products`. Compare its two complete Gaussian
kinetic kernels with deterministic first providers and a common
deterministic joint second provider of velocity moment at most
$M_w=.70$. Retain the actual OU noise, both count kicks, native cap
and final position Gaussian. Use the constants of
{prf:ref}`def-rvb-common-root`:
$$
A=m-a>0,\quad \alpha=t^2\nu\ell,\quad
\overline S_w=\frac{V/4+tX_{1,2}+aM_w}{A},
$$
$$
\overline K_w=m+\alpha(M_w+\overline S_w),\qquad
\overline K_x=t+a\ell(M_w+\overline S_w),
$$
$$
C_0=\sqrt{(ba)^2+(ca\overline K_w+ta\overline K_x)^2}<.00578.
$$
With each own innovation shared between the compared kernels,
$$
\big(\mathbb E[|\Delta x^+|^2+|\Delta v^+|^2]\big)^{1/2}
\le C_0(A_0+rB_0).
\tag{RFP.7}
$$
At $r=.55$, for first providers coupled as in (RFP.5), the right
side is at most
$$
C_0[d_P+2r\ell d_X]
\le.00578[d_P+1.1\ell d_X].
\tag{RFP.8}
$$
If the joint second providers also differ, the complete fixed-root
comparison becomes
$$
\big(\mathbb E Q_\beta(\Delta x^+,\Delta v^+)\big)^{1/2}
\le\sqrt{1+\beta}\left\{
C_0[d_P+2r\ell d_X]
+a[d_w+\ell(M_w+\overline S_w)d_y]\right\}.
\tag{RFP.9}
$$
The root distribution is held fixed in this theorem. Its preparation-law
change is not included in the displayed provider contribution.
:::

:::{prf:proof}
Interpolate the first provider and keep the actual independent root OU
and final Gaussian innovations shared. Its perturbation is $H(X,P)$
from (RFP.4), and its exact first-drift/OU/position derivatives are
$$
\dot x_1=taH,\qquad \dot w=caH,\qquad \dot y=baH.
$$
The uncut pointwise second-kick/cap derivative bounds of
{prf:ref}`lem-rvb-uncut-cap-jacobians` are
$$
J_w(x_1)=m+\alpha\left[
M_w+\frac{V/4+aM_w+t|x_1|}{A}\right],
$$
$$
J_x(x_1)=t+a\ell\left[
M_w+\frac{V/4+aM_w+t|x_1|}{A}\right].
$$
They apply to the actual map with $y=x_1+tw$ and retain its
force/OU correlations. Both are affine nonnegative functions of
$|x_1|$. Substitute (RFP.4) into these pointwise products to obtain
$$
\|J_w(x_1)H\|_2\le\overline K_w(A_0+rB_0),\qquad
\|J_x(x_1)H\|_2\le\overline K_x(A_0+rB_0).
$$
This is the needed weighted-product estimate; it is not inferred by
factoring an averaged Jacobian from an arbitrarily correlated
displacement. Integrate in the provider interpolation parameter.
The cap velocity difference is at most
$(ca\overline K_w+ta\overline K_x)(A_0+rB_0)$ in RMS,
and the position difference is at most $ba(A_0+rB_0)$.
The final shared position Gaussian cancels. Squaring the two
components proves (RFP.7), and (RFP.5) proves (RFP.8).

For the second-provider change with the root law and first provider
fixed, {prf:ref}`lem-rvb-uncut-provider-cap` supplies
$a[\|\Delta M_2\|_\infty+\overline S_w\|\Delta a_2\|_\infty]$.
Use (RFP.6), Minkowski, and
$Q_\beta(r,s)\le(1+\beta)(|r|^2+|s|^2)$ to obtain (RFP.9).
The full own marginals have their original independent Gaussian
innovations throughout the coupling. The numerical $C_0$ bound is
the same rational coefficient calculation as in
{prf:ref}`thm-rvb-global-own-provider-feedback`.
:::

(sec-rfp-signed-first-kick)=
## 4. Exact signed own-law first-kick balance

:::{prf:lemma} Symmetric first-kick derivative and retained alignment form
:label: lem-rfp-signed-first-kick

Let $\pi$ be any coupling of two prepared laws with finite second
position moments and velocities bounded by $V_c$. On its probability
space write $\delta X=\widetilde X-X$, $\delta P=\widetilde P-P$,
and interpolate $X_\theta=X+\theta\delta X$,
$P_\theta=P+\theta\delta P$. The actual endpoint first-count
outputs are $U_0$ and $U_1$ of (RFP.2); intermediate laws are used
only for comparison. For an independent copy denoted by primes put
$$
k_\theta=K(X_\theta-X_\theta'),\qquad
\dot k_\theta=\nabla K(X_\theta-X_\theta')\cdot
                       (\delta X-\delta X'),
$$
$$
D_\theta=\frac12\mathbb E[k_\theta|\delta P-\delta P'|^2],
\qquad
S_\theta=\frac12\mathbb E\left[
\frac{\dot k_\theta^2}{k_\theta}|P_\theta-P_\theta'|^2\right].
\tag{RFP.10}
$$
The lifted count Laplacian $L_\theta$ is self-adjoint, nonnegative
and bounded by $I$, and
$$
\dot U_\theta=(I-aL_\theta)\delta P+aB_\theta,\qquad
B_\theta=-L_{\dot k_\theta}P_\theta.
\tag{RFP.11}
$$
For every $\eta\in(0,2-a)$,
$$
\|U_1-U_0\|_2^2
\le\|\delta P\|_2^2
-a(2-a-\eta)\int_0^1D_\theta\,d\theta
+\left[\frac{a(1+a\sqrt2)^2}{\eta}+2a^2\right]
                  \int_0^1S_\theta\,d\theta.
\tag{RFP.12}
$$
The spatial forcing is thus charged against an explicit weighted
pair displacement; its correlation with velocities is retained.
:::

:::{prf:proof}
On the coupling probability define
$(L_\theta f)(\omega)=\mathbb E'[k_\theta(f-f')]$.
Symmetry gives
$\langle f,L_\theta f\rangle=\frac12\mathbb E[k_\theta|f-f'|^2]$.
Because $0\le k_\theta\le1$, this form is at most
$\mathbb E|f-\mathbb Ef|^2\le\|f\|_2^2$. The asserted operator
properties follow. Differentiating $U_\theta=P_\theta-aL_\theta P_\theta$
gives (RFP.11). Kernel and velocity derivatives are dominated by
integrable bounded multiples of $|\delta X|+|\delta X'|$ and
$|\delta P|+|\delta P'|$, so this differentiation is legitimate
in $L^2$.

Conditional Cauchy--Schwarz with weight $k_\theta$, using
$\mathbb E'k_\theta\le1$, gives
$\|B_\theta\|_2^2\le2S_\theta$. Pair symmetrization also gives
$$
|\langle\delta P,B_\theta\rangle|
\le\sqrt{D_\theta S_\theta},\qquad
\|L_\theta\delta P\|_2^2\le D_\theta.
$$
Consequently
$$
|\langle(I-aL_\theta)\delta P,B_\theta\rangle|
\le(1+a\sqrt2)\sqrt{D_\theta S_\theta}.
$$
The exact first-kick self-damping obeys
$\|(I-aL_\theta)\delta P\|_2^2
\le\|\delta P\|_2^2-a(2-a)D_\theta$.
Expand (RFP.11), bound the remaining square by $2a^2S_\theta$,
and use
$2a(1+a\sqrt2)\sqrt{D_\theta S_\theta}
\le a\eta D_\theta+a(1+a\sqrt2)^2S_\theta/\eta$.
Jensen on $U_1-U_0=\int_0^1\dot U_\theta d\theta$ proves (RFP.12).
:::

:::{prf:lemma} Conditional source-aware spatial defect
:label: lem-rfp-source-defect

Suppose the coupling in (RFP.10) is realized by two coupled complete
plans followed by a shared fresh recipient jitter $J$:
$$
X=S+IJ,\qquad \widetilde X=\widetilde S+\widetilde I J,
\qquad \delta S=\widetilde S-S,\quad\delta I=\widetilde I-I.
$$
The joint plan, including both prepared velocities, is independent of
$J$. An independent copy uses its own independent $J'$. Then
$$
S_\theta\le\frac1e\mathbb E_{\rm plan}\left[
\left\{|\delta S-\delta S'|^2+
d\sigma_J^2[(\delta I)^2+(\delta I')^2]\right\}
|P_\theta-P_\theta'|^2\right].
\tag{RFP.13}
$$
Writing $r_\theta^2=\mathbb E|P_\theta|^2$, this implies
$$
\begin{split}
S_\theta\le{}&\frac8e\mathbb E[
|\delta S|^2(|P_\theta|^2+r_\theta^2)]\\
&+\frac{4d\sigma_J^2}{e}\mathbb E[
(\delta I)^2(|P_\theta|^2+r_\theta^2)].
\end{split}
\tag{RFP.14}
$$
After both populations have burned, $r_\theta\le.55$ by Minkowski.
The local factors $|P_\theta|^2$ remain inside the expectations.
:::

:::{prf:proof}
The exact Gaussian identity is
$|\nabla K(r)|^2/K(r)=|r|^2e^{-|r|^2/2}\le2/e$.
It bounds the integrand in (RFP.10) by
$(2/e)|\delta X-\delta X'|^2|P_\theta-P_\theta'|^2$.
Condition on the two independent coupled plans. The prepared
velocities are fixed before fresh jitters. The jitters have zero
mean, independent cross terms and covariance $\sigma_J^2I_d$,
so
$$
\mathbb E(|\delta X-\delta X'|^2\mid\mathrm{plans})
=|\delta S-\delta S'|^2+
d\sigma_J^2[(\delta I)^2+(\delta I')^2].
$$
This proves (RFP.13). Use
$|\delta S-\delta S'|^2\le2(|\delta S|^2+|\delta S'|^2)$ and
$|P_\theta-P_\theta'|^2\le2(|P_\theta|^2+|P_\theta'|^2)$.
The independent-copy cross terms factor at the plan level. Expanding
them yields the coefficients $8/e$ and $4d\sigma_J^2/e$ in
(RFP.14). This factorization involves independent root plans, not
the possibly correlated displacement and velocity of one root.
Finally $\|P_\theta\|_2\le(1-\theta)\|P\|_2+
\theta\|\widetilde P\|_2\le.55$.
:::

:::{prf:corollary} The first-kick account in the complete-update phase quadratic
:label: cor-rfp-first-kick-quadratic

In {prf:ref}`lem-rfp-signed-first-kick`, write $R=\delta X$ and
$$
E_\theta=\langle R,L_\theta\delta P\rangle
=\frac12\mathbb E[k_\theta(R-R')\cdot
                                   (\delta P-\delta P')],\qquad
D_{X,\theta}=\frac12\mathbb E[k_\theta|R-R'|^2].
$$
Then the actual endpoint first kick satisfies
$$
\begin{split}
\mathbb E Q_\beta(R,U_1-U_0)
-\mathbb E Q_\beta(R,\delta P)
\le\int_0^1\big\{&-2aD_\theta
 +a^2\|L_\theta\delta P\|_2^2-2\beta aE_\theta\\
 &+2a\langle(I-aL_\theta)\delta P+\beta R,B_\theta\rangle
 +a^2\|B_\theta\|_2^2\big\}\,d\theta.
\end{split}
\tag{RFP.16}
$$
The remaining spatial mixed term has the correlation-safe estimate
$$
|\langle R,B_\theta\rangle|
\le\sqrt{D_{X,\theta}S_\theta},\qquad
D_{X,\theta}\le\mathbb E|R|^2.
\tag{RFP.17}
$$
In particular the signed $E_\theta$ term, the negative velocity
alignment and the source-weighted forcing can be used directly in
the complete harmonic/cap quadratic; they have not been replaced
by a claim that the first kick separately contracts $Q_\beta$.
:::

:::{prf:proof}
For each $\theta$, expand the quadratic after substituting
$\dot U_\theta=(I-aL_\theta)\delta P+aB_\theta$.
The velocity-square difference is
$-2aD_\theta+a^2\|L_\theta\delta P\|_2^2
+2a\langle(I-aL_\theta)\delta P,B_\theta\rangle
+a^2\|B_\theta\|_2^2$. The mixed phase difference adds
$-2\beta aE_\theta+2\beta a\langle R,B_\theta\rangle$.
Jensen in the second argument of $Q_\beta(R,\cdot)$, with
$U_1-U_0=\int_0^1\dot U_\theta d\theta$, gives (RFP.16).
The same pair symmetrization and weighted Cauchy--Schwarz used for
$\langle\delta P,B_\theta\rangle$ give the first inequality in
(RFP.17). The second follows from $k_\theta\le1$ and the
independent-copy variance identity. These operations retain every
root displacement/velocity correlation.
:::

(sec-rfp-stage-scope)=
## 5. Joint-stage identities and the remaining consumer

:::{prf:lemma} Both actual joint-stage differences under shared OU noise
:label: lem-rfp-joint-stage

In any valid prepared-root coupling with its actual own first
providers, share the original OU noise. Set
$R=X-\widetilde X$, $E=U-\widetilde U$,
$d_X^2=\mathbb E|R|^2$, $d_U^2=\mathbb E|E|^2$ and
$H=\mathbb E[R\cdot E]$. The actual joint second-stage coupling
satisfies exactly
$$
d_y^2=a_x^2d_X^2+b^2d_U^2+2a_xbH,\qquad
d_w^2=c^2(t^2d_X^2+d_U^2-2tH).
\tag{RFP.15}
$$
Its own stage marginals retain their independent root OU innovation.
The source and count fields are not conditioned on that innovation.
:::

:::{prf:proof}
The exact stages are $y=a_xX+bU+tq\xi$ and
$w=c(U-tX)+q\xi$. Sharing the same independent $\xi$ between
the two own root kernels cancels the innovation differences and
gives $\Delta y=a_xR+bE$, $\Delta w=c(E-tR)$.
Expand the two squared norms and integrate. The construction preserves
the original own Gaussian marginal independently of its preparation.
:::

:::{prf:remark} Completed first-provider interfaces and unclosed own-law block
:label: rem-rfp-scope

(RFP.7)--(RFP.9) improve the full-jitter fixed-root first-provider
comparison using the proved post-burn moment and the actual source
construction. They do not change the algorithm, require a reduced
individual cap, or introduce a tail floor. (RFP.12)--(RFP.15) give
an exact signed first-kick account and conditional source forcing
for a complete actual prepared-law coupling. (RFP.16)--(RFP.17)
also retain its complete-update quadratic cross term. The negative
alignment form and the shared joint-stage covariance remain visible.

The all-slot velocity moment alone does not permit replacing
$\mathbb E[|\delta S|^2|P_\theta|^2]$ by
$r_\theta^2\mathbb E|\delta S|^2$: the source displacement and
component velocity can depend on the same measured/Haar plan.
The theorem that uses an averaged cap derivative similarly cannot
factor it from an arbitrary correlated phase displacement. (RFP.4)
proves the particular weighted product needed for deterministic
provider changes; it does not prove that assertion for other changes.

To close the full reference own-law block, its consumer must add the
actual rooted preparation-law change (including dead-root revival and
normalizer feedback), the source-local cost in (RFP.14), and the
actual B2/cap balance for the joint-stage variables in (RFP.15) to
the complete-update $Q_\beta$ margin. No coefficient below one for
that combined account is claimed here. Finite empirical-provider
consistency and current-survival normalization remain separate from
the deterministic population provider formulas proved in this note.
:::
