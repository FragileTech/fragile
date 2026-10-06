# Same-record marked spatial geometry and the actual B2 color field

(sec-nmg-ledger)=
## 1. Full preparation, terminal geometry and posterior innovations

:::{prf:definition} Complete marked-geometry record
:label: def-nmg-complete-register

Retain every execution, algorithm, landscape, initialization, donor,
cloning, component-collision, force, boundary, arithmetic, observation
and calibration parameter in {prf:ref}`def-nsg-complete-register`.
The positive results use its existing fixed-step quadratic count/row
gas, independent O and final-position Gaussians, terminal recording
and matched B2 viscous color. Geometry remains the actual Euclidean
open CSR/Delaunay geometry with its covariance ridge, clamp,
pseudo-inverse, weight, volume, availability and error marks.
The color parameters are its actual force threshold $\delta_c$,
phase coefficient $\kappa_c=m\ell_0/\hbar_{\rm eff}$, stage,
same-stage alignment, finite-value mask and any consumed preceding
calibration. An extra clone-deletion mask is a different channel.
Keep all source-slot/generation/frame labels at finite $N$.
The executed common $SU(3)$ color/projector statements have
$d=3$: their recorded viscous force and force-input velocity
are aligned three-component vectors. The Gaussian posterior,
point-process and force-sampling statements are valid in any
configured dimension; they do not assert this three-component
color instrument in other dimensions.
The fixed-reference-length instrument has its configured finite
$\kappa_c$. For a consumed preceding calibration varying with
$N$, the joint color-limit statements require retaining that
calibration in the same joint subsequence and its actual finite
limit. Neither state convergence nor the conditional posterior
proves calibration convergence on its own.

Freeze the complete post-collision/jitter preparation $\mathcal F$.
Put $p_j=X_j^J+tv_{1j}$ and retain the preparation mark
$A_j=(p_j,v_{1j},m_j,\mathfrak s_j)$, where $\mathfrak s_j$
is the actual frozen source/fitness/gate/component/record metadata.
It is allowed to depend on the entire original swarm. The actual
stages are

$$
z_j=cv_{1j}+q\xi_j,\quad
X_j=p_j+tz_j,\quad Y_j=X_j+s\zeta_j=m_j+tq\xi_j+s\zeta_j,
\quad m_j=p_j+tcv_{1j}.
\tag{NMG.1}
$$

Here $X_j$ is the B2 force-input position, $z_j$ its uncapped
force-input velocity, and $Y_j$ the later terminal position.
Their distinction is preserved. Define
$\tau^2=t^2q^2+s^2$, $\chi=s^2/\tau^2>0$ and
$a_Y=tq^2/\tau^2$.
:::

:::{prf:lemma} Exact native posterior and residual-source marks
:label: lem-nmg-exact-posterior

Conditionally on $\mathcal F$ and the complete terminal array
$Y=y$, the original O rows remain independent with laws

$$
\xi_j\mid(\mathcal F,Y=y)
 \sim N\left(\frac{tq}{\tau^2}(y_j-m_j),\chi I\right).
\tag{NMG.2}
$$

Equivalently, on the original probability space introduce the
computed residual coordinates

$$
E_j=\chi^{-1/2}\left[\xi_j-
                 \frac{tq}{\tau^2}(Y_j-m_j)\right].
$$

Conditionally on $\mathcal F$, these are independent standard
Gaussian rows, independent of the entire $Y$ array. The native
force-input coordinates are exactly

$$
z(A_j,y_j,E_j)=cv_{1j}+a_Y(y_j-m_j)+q\sqrt\chi E_j,
\quad X(A_j,y_j,E_j)=p_j+t z(A_j,y_j,E_j).
\tag{NMG.3}
$$

No new noise has been added. In particular the conditional
residual standard deviation is $q\sqrt\chi$; the addressed
conditional mean-source coefficient $q\chi$ is a different
quantity.
:::

:::{prf:proof}
For each row, $(\xi_j,Y_j-m_j)$ is jointly normal with covariance
blocks $I$, $tqI$ and $\tau^2I$. Completing its square gives
(NMG.2) and the residual variance $\chi I$.
The displayed residual has zero covariance with $Y_j-m_j$.
Its covariance is $I$ and the joint law is Gaussian; the Gaussian
characteristic function therefore factors. Different rows use
different original $(\xi_j,\zeta_j)$ and remain independent on
the frozen preparation. This proves the full-array statement.
Substitution in (NMG.1) gives (NMG.3).
:::

(sec-nmg-marked-poisson)=
## 2. Source-marked local point processes from the same record

:::{prf:theorem} Conditional source-marked Poisson comparison
:label: thm-nmg-marked-poisson

Set $r_N=N^{-1/d}$. Condition on $\mathcal F,Y_i=x$.
In a bounded window $W$, retain the actual marked process

$$
\sum_{j\ne i}\delta_{((Y_j-x)/r_N,A_j,E_j)}|_W.
$$

Its total-variation distance from a Poisson process on the
spatial/mark product with intensity

$$
du\ \mathfrak M_{N,x}(dA,dE),\qquad
\mathfrak M_{N,x}=
 \frac1N\sum_{j\ne i}\varphi_\tau(x-m_j)
                         \delta_{A_j}\otimes\gamma_d
\tag{NMG.4}
$$

is at most

$$
\frac{L_{\tau,0}^2|W|^2}{N}
+r_NL_{\tau,1}\int_W|u|\,du.
\tag{NMG.5}
$$

Every finite-$N$ source label and full preparation record can
remain in this comparison measure, even if their mark spaces
are $N$ dependent. No tight limit of absolute slot labels is
asserted. On any fixed coordinate/source-mark space where the
empirical preparation measures $\eta_N=N^{-1}\sum_j\delta_{A_j}$
have a joint subsequential weak limit $\eta$ with the tag,
the limiting intensity is

$$
\mathfrak M_{\eta,x}(dA,dE)
 =\varphi_\tau(x-m(A))\eta(dA)\gamma_d(dE).
\tag{NMG.6}
$$

The tag's own preparation and residual marks remain in their
same joint law; its residual is an independent $\gamma_d$ row
conditionally on $\mathcal F,Y_i=x$. Alive-only geometry has
the identical comparison on windows inside $D$.
:::

:::{prf:proof}
The posterior lemma makes $(Y_j,E_j)$ independent across rows,
with fixed preparation mark $A_j$. Each row hits $W$ with
probability at most $L_{\tau,0}|W|/N$. Couple its Bernoulli point
to a Poisson point process, retaining the same source mark and
residual Gaussian. The sum of its count errors is at most
$L_{\tau,0}^2|W|^2/N$, exactly as in
{prf:ref}`lem-nsg-local-poisson-tv`.
The resulting marked intensity at $u$ is
$N^{-1}\sum_{j\ne i}\varphi_\tau(x+r_Nu-m_j)
\delta_{A_j}\otimes\gamma_d$.
Its total-variation difference from (NMG.4), integrated over
$W$, is at most the Gaussian gradient term in (NMG.5).
The common-minimum intensity coupling proves the result.

For the subsequential statement the bounded Gaussian weight
is continuous in the coordinate marks and in $x$; the residual
law is fixed. On bounded windows this gives weak intensity
convergence. Poisson count probabilities and conditional
independent mark integrals then converge by their exponential
and finite product formulas; approximate bounded continuous
marked tests and truncate the Poisson counts to verify the
point-process convergence. No independent output population
has replaced the actual preparation. A full label sequence
without tightness is instead retained by (NMG.4).
:::

(sec-nmg-bulk)=
## 3. Native dense B2 forces at locally selected geometry sites

:::{prf:corollary} Actual root-source bias for a uniformly sampled terminal tag
:label: cor-nmg-root-mark-law

Select a slot $I$ uniformly, independently after the native
preparation. Conditional on its empirical preparation record
and terminal position $Y_I=x$, its exact preparation/residual
mark law is

$$
\pi_{N,x}(dA,dE)=
 \frac{\varphi_\tau(x-m(A))}{\rho_N(x)}
                 \eta_N(dA)\gamma_d(dE).
\tag{NMG.13}
$$

The omitted-root intensity in (NMG.4) differs from the full
intensity $\rho_N(x)\pi_{N,x}$ by a measure of mass at most
$L_{\tau,0}/N$. Thus the same local comparison can retain
the root drawn from (NMG.13) and a Poisson configuration of
other sites with mark law $\pi_{N,x}$ and spatial intensity
$\rho_N(x)$, conditionally independent of that root on the
comparison space, at additional error at most
$L_{\tau,0}|W|/N$. This conditional independence is a derived
finite-window comparison, not an assumption about finite
swarm rows. For a fixed labelled tag of an exchangeable QSD,
the same statement applies to its coordinate/equivariant
source-mark projection. Absolute fixed slot names keep their
finite-$N$ identity rather than inheriting that projection.

Along the joint coordinate-preparation subsequence, the root
law is the explicit tilted mark law
$\pi_{\eta,x}=\varphi_\tau(x-m)\eta\otimes\gamma_d/
\rho_\eta(x)$. It is the same mark distribution as every
local Poisson neighbor, conditionally on the common
environment and query. All its B2 force/color functions
still use that common environment.
:::

:::{prf:proof}
Uniform sampling makes the conditional entering preparation
mark law exactly $\eta_N$. Its terminal position density
given that mark is $\varphi_\tau(x-m)$, and its residual
is independent $\gamma_d$ by the posterior lemma. Bayes'
formula gives (NMG.13). Excluding the sampled root removes
one atom of mass at most $L_{\tau,0}/N$. Poisson processes
with and without that intensity atom can be coupled by
adding its independent Poisson points, whose appearance
probability is bounded by its mass times $|W|$.
The complete-intensity process depends on the empirical
preparation record but no longer on which root was selected.
This yields the stated conditional comparison. QSD
exchangeability and uniform-label sampling of the empirical
multiset were proved in
{prf:ref}`thm-native-stationary-closure-population-invariance`.
The bounded continuous Gaussian weight and its strictly
positive integral pass the same root bias to the subsequence.
:::

:::{prf:definition} Common quenched B2 stage law
:label: def-nmg-common-stage-law

For the coordinate projection of $\eta_N$, let

$$
\Lambda_{\eta_N}=\int\operatorname{Law}
 \bigl(p+t(cv_1+qG),cv_1+qG\bigr)\,\eta_N(dA),
\qquad G\sim\gamma_d.
$$

This is the conditional mean law of the actual B2 input array.
For its common environment define

$$
\alpha_N(X)=\int K_\rho(X,X')\,d\Lambda_{\eta_N},\qquad
\beta_N(X)=\int K_\rho(X,X')z'\,d\Lambda_{\eta_N},
$$
$$
F_N^{\rm pop,count}(X,z)=\nu[\beta_N(X)-\alpha_N(X)z],
\quad
F_N^{\rm pop,row}(X,z)=
 \nu[\beta_N(X)/\alpha_N(X)-z].
\tag{NMG.7}
$$

$\alpha_N(X)>0$ at every finite $X$. These functions use the
same $\eta_N$ for every local neighbor. They are not recomputed
from independent local copies or from terminal positions in
place of $X$.
:::

:::{prf:lemma} Locally selected count-force error from actual fresh rows
:label: lem-nmg-local-count-force

Conditional on $\mathcal F$ and on row $j$'s own B2 inputs,
the actual count-normalized viscous force satisfies

$$
\mathbb E\bigl[
|F_j^{N,\rm count}-F_N^{\rm pop,count}(X_j,z_j)|^2
\mid\mathcal F,X_j,z_j\bigr]
\le\frac{C\nu^2}{N}
 [q^2+\eta_N|cv_1|^2+|z_j|^2]
+\frac{C\nu^2}{N^2}|cv_{1j}|^2,
\tag{NMG.8}
$$

with a dimension-dependent finite $C$. If one further conditions
on the terminal tag $Y_i=x$, $i\ne j$, the same bound holds
with the additional correction
$C\nu^2N^{-2}(|z_i|^2+|cv_{1i}|^2+q^2)$ conditional on its
own actual posterior row. For $x$ in a fixed compact set and
fixed bounded local window $W$, the expected number of local
sites with count-force error exceeding any fixed $a>0$ tends
to zero. The expectation is over the original incoming law
and update, retaining the event $Y_i$ in that compact set.
Every preparation/velocity moment used is the primitive
Gaussian-jitter/OU moment of the quadratic capped register.
For the root row itself, condition on $\mathcal F,Y_i=x$
and its residual $E_i$. Its own $(X_i,z_i)$ are then fixed
by (NMG.3), while all other O rows remain independent.
Thus (NMG.8) applies unchanged to its root force. Integrating
the root posterior and original tag law gives a vanishing
root-force error on compact terminal-tag events as well.
:::

:::{prf:proof}
Conditioning on the own input leaves all other original O rows
independent on $\mathcal F$. Their summands
$K_\rho(X_j,X_l)(z_l-z_j)$ have second moments at most
$2\mathbb E|z_l|^2+2|z_j|^2$.
Their variance sum divided by $N^2$ is therefore bounded by
the first term of (NMG.8). The conditional mean differs from
the mixture in (NMG.7) only by its independently averaged
own row: the actual self-summand is zero. Its squared error
is at most $C\nu^2N^{-2}(q^2+|cv_{1j}|^2+|z_j|^2)$,
absorbed in the displayed bound. A conditioned tag removes
one more independent summand; its actual/averaged difference
has the stated $N^{-2}$ second-moment correction.

For local selection integrate $Y_j=y$ over $x+r_NW$ using
its exact density and posterior (NMG.3). For each such row,
$\mathbb E[|z_j|^2\mid\mathcal F,Y_j=y]
\le C(q^2+|cv_{1j}|^2+a_Y^2|y-m_j|^2)$.
The density weight $\varphi_\tau(y-m_j)$ is bounded by
$L_{\tau,0}$, and its product with $|y-m_j|^2$ has a
finite Gaussian supremum. The window has volume $|W|/N$.
Summing the Markov second-moment bound over $j$ consequently
gives a bound $C_{W,a}/N$ times finite empirical preparation
second moments, plus $C_{W,a}/N^2$ times the conditioned tag
moment. Integrating the original tag/preparation law bounds
these by the uniform jitter/collision/B1 Gaussian moments.
One may integrate the tag density first; its product with
its posterior moments has the same Gaussian bounds. Thus the
expected number tends to zero. This conditions on the actual
selected positions and never assumes iid incoming rows.
The root case uses no further removed row: fix its posterior
residual and apply the initial independent-summand calculation.
Its posterior second moment, integrated against the actual tag
density, is bounded by the same preparation/Gaussian moments.
:::

:::{prf:lemma} Locally selected nonself row-force error
:label: lem-nmg-local-row-force

The same conclusion holds for the existing nonself row
normalization, by localization to its primitive moment core.
For $|X_j|\le R$, if
$\Lambda_{\eta_N}|X'|^p\le H$, put $S=(2H)^{1/p}$ and

$$
a_{R,H}=\tfrac12 e^{-(R+S)^2/(2\rho^2)}>0.
$$

Then $\alpha_N(X_j)\ge a_{R,H}$. The empirical nonself
denominator divided by $N$ has conditional variance at most
$1/N$ and differs in mean from $\alpha_N(X_j)$ by at most
$1/N$ (or $2/N$ when retaining the conditioned tag).
On its event of deviation at most $a_{R,H}/2$, subtracting
the two literal force fractions bounds the error by a fixed
multiple of the numerator and denominator sampling errors.
Its complementary probability is $O(1/(Na_{R,H}^2))$.
The exact $1/N$ self-weight is retained, rather than removed
from this estimate. First $N\to\infty$ on bounded moment
cores, then $R,H\to\infty$ proves the local-window result.
:::

:::{prf:proof}
At least half the mean stage position law lies in $B(0,S)$
by Markov. On that ball the Gaussian kernel is bounded below
by the displayed exponential, proving the degree bound.
The other fresh O rows give the stated variance and missing
row corrections exactly as in the count proof. Chebyshev
controls the degree event. On it use
$|a/b-c/d|\le2|a-c|/d+2|c||b-d|/d^2$ for $b\ge d/2$.
The numerator moments and local selection are those just
proved. Stage position/velocity moments are uniform because
$p,v_1$ have the primitive Gaussian-jitter bounds and O
adds its original Gaussian. The spatially selected moment
tails have the same bounded density weights used above;
Markov removes their cores. Thus there is no assumed degree
floor or omitted Gaussian tail.
:::

(sec-nmg-joint-limit)=
## 4. Actual color marks and covariance geometry in one common environment

:::{prf:lemma} Derived count-color threshold boundary has zero native mass
:label: lem-nmg-count-threshold-boundary

For every common stage law $\Lambda_\eta$ above with finite second
position and first velocity moments, $t,q,s>0$, and configured
$\delta_c>0$, the count-color threshold boundary has zero conditional
posterior mass at every finite preparation mark and terminal query.
No regularity or nondegeneracy of an unknown stationary law is needed.
The available region may be empty; this conclusion retains that case.
:::

:::{prf:proof}
Gaussian convolution makes $\alpha(X)$ and every component of
$\beta(X)$ real analytic: for complex $X$ in a bounded imaginary
strip the kernel modulus is bounded by
$e^{|\Im X|^2/(2\rho^2)}$, times its real Gaussian modulus.
The finite velocity first moment dominates the latter integral.
The locally uniformly convergent complex power series, or its
dominated derivatives, therefore gives analyticity.

As $|X|\to\infty$, $\beta(X)\to0$ by dominated convergence.
Moreover $|X|\alpha(X)\to0$: split the position integral at
$|X'|=|X|/2$. Its inner part is at most
$|X|e^{-|X|^2/(8\rho^2)}$, and its outer part is bounded by
$4\mathbb E|X'|^2/|X|$. Thus, along $X=p+tz$,
$F_\eta^{\rm pop,count}(p+tz,z)\to0$ as $|z|\to\infty$.
The real analytic function
$|F_\eta^{\rm pop,count}(p+tz,z)|^2-\delta_c^2$
is consequently nonzero, tending to $-\delta_c^2$.

A nonzero real analytic function on connected Euclidean space
has a Lebesgue-null zero set. In an analytic coordinate box
expand in one coordinate; at least one coefficient is a nonzero
analytic function of the other coordinates. By dimension
induction its vanishing set is null. On the remaining slices
the one-variable function has discrete zeros, so Fubini gives
zero measure. Countably many coordinate boxes cover the space;
analytic continuation excludes an identically zero open box.
The posterior $z$ has nondegenerate Gaussian density by (NMG.3),
so it gives this zero set zero mass. Mixtures over preparation
marks and over the common environment preserve that assertion.
:::

:::{prf:lemma} Both native normalization thresholds have zero posterior boundary mass
:label: lem-nmg-full-threshold-boundary

In the same complete quadratic capped register, retain
$\lambda\ge0$, $t,q,s,\rho>0$, $\nu>0$ and

$$
a_x=1-t^2(1+c)\lambda>0,\qquad
U_*=\kappa_\nu V_c<\infty,
\quad \kappa_\nu=\max\{1,2t\nu-1\}.
$$

For every actual empirical preparation and every admitted
limiting coordinate preparation law, both count and nonself row
population forces have Lebesgue-null sets
$\{|F_\eta^{\rm pop,\mathfrak n}(p+tz,z)|=\delta_c\}$
for every finite preparation coordinate $p$ and every finite
configured threshold $\delta_c\ge0$. Their conditional posterior
boundary mass is zero. This includes the zero-threshold branch.
It uses no independence between jitter and the first viscous kick.

The unchanged reference has
$a_x=.999215684224339\ldots>0$, $t\nu=.006\le1$ and
$U_*=V_c=4$; hence this discharges its hard color masks for
BOTH configured normalizations. More generally, the proof
retains $U_*=\kappa_\nu V_c$ outside that convex first-kick
subregime. The reset $a_x=0$ is handled separately below;
the negative-$a_x$ branches retain their evaluated boundary
test. At $\nu=0$ both exact and limiting
viscous forces are zero, so their strict availability channel
is identically unavailable, including $\delta_c=0$.
:::

:::{prf:proof}
The original first kick has
$v_1=U-t\lambda X^J$, where $U=W_{X^J}v^c$ and the
configured count or row matrix has absolute row sum at most
$\kappa_\nu$. The original collision bound is
$|v_i^c|\le V_c$. Thus $|U_i|\le U_*$, even though this
first viscous kick can depend on the entire jitter array.
Its terminal preparation center is exactly
$m=a_xX^J+bU$, $b=t(1+c)$.
Since $a_x>0$, the two exact identities give

$$
v_1=\frac{U-t\lambda m}{a_x},\qquad
U=a_xv_1+t\lambda m.
\tag{NMG.21}
$$

This closed coordinate relation and its uniform $U$ bound
pass to every preparation limit. No factorization of its
joint $(m,U)$ law is imposed.

Set $B=\rho^2+t^2q^2$ and
$C_B=(\rho^2/B)^{d/2}$. Completing the original Gaussian
square in $z'=cv_1+qG$ and $X'=m+tqG$ yields

$$
\alpha(X)=C_B\int e^{-|X-m|^2/(2B)}\,\eta(dA),
$$
$$
\frac{\beta(X)}{\alpha(X)}=
\frac{tq^2}{B}X-a_m E_Xm+\frac c{a_x}E_XU,
\qquad a_m=\frac{ct\lambda}{a_x}+\frac{tq^2}{B}\ge0,
\tag{NMG.22}
$$

where $E_X$ denotes expectation under the actual preparation
law tilted by $e^{-|X-m|^2/(2B)}$. Its denominator is strictly
positive. Its tilted moments of $m$ are finite, since
each polynomial times this Gaussian weight is bounded at
fixed $X$; $U$ stays bounded by $U_*$.

Fix $p$ and a unit vector $u$, and put $X_r=p+tr u$.
Differentiating the normalized tilted integral gives exactly

$$
\frac d{dr}E_{X_r}(u\cdot m)
=\frac tB\operatorname{Var}_{X_r}(u\cdot m)\ge0.
\tag{NMG.23}
$$

The Gaussian-polynomial bounds justify this derivative
locally uniformly in $r$. Therefore (NMG.22) gives

$$
u\cdot\left[\frac{\beta(p+tr u)}{\alpha(p+tr u)}-ru\right]
\le-\frac{\rho^2}{B}r+C_{p,u},
$$
$$
C_{p,u}=\frac{tq^2}{B}|u\cdot p|
             +a_m|E_p(u\cdot m)|+\frac c{a_x}U_*<\infty.
\tag{NMG.24}
$$

Here the monotonic lower bound on the tilted mean is used
with the nonnegative coefficient $a_m$; no upper bound on
the unbounded jitter centers is inserted. The row force
norm consequently grows at least as
$\nu[(\rho^2/B)r-C_{p,u}]$ for sufficiently large $r$.
Its analytic squared norm minus any finite $\delta_c^2$
is not identically zero. Analyticity follows from the
Gaussian-convolution argument in the preceding lemma
and the strictly positive analytic denominator $\alpha$.

The count force is exactly $\alpha$ times the row force.
It is therefore nonzero at all sufficiently large points
on this ray, even though its magnitude tends to zero.
For $\delta_c=0$ its squared norm is a nonzero analytic
function. For $\delta_c>0$ the decay argument of the
preceding lemma makes its squared norm minus $\delta_c^2$
nonzero as well. The analytic-zero-set proof there gives
Lebesgue-null boundaries for both normalizations and
every threshold. The original residual posterior has
covariance $q^2\chi I>0$, so it assigns them zero mass.
The reference inequalities follow by substituting its
actual recorded coefficients. If $\nu=0$, the original
force formula proves the stated exact unavailable channel.
:::

:::{prf:lemma} The exact preparation reset also has null hard-mask boundaries
:label: lem-nmg-reset-threshold-boundary

Keep the same complete quadratic capped register and
$t,q,s,\rho,\nu>0$, but now let $a_x=0$ exactly.
For actual empirical preparations, and almost surely for
their admitted limiting coordinate environments, the conclusion
of {prf:ref}`lem-nmg-full-threshold-boundary` still holds for
both normalizations and every finite $\delta_c\ge0$.
The original jitter is unbounded and remains Gaussian.
:::

:::{prf:proof}
The original identities are now
$m=bU$ and $v_1=U-t\lambda X^J$.
Thus $|m|\le M_m=b\kappa_\nu V_c$ and

$$
|v_1|\le A_v+B_v|G^J|,\qquad
A_v=\kappa_\nu V_c+t\lambda R_D,\quad
B_v=t\lambda\sigma_J.
$$

The unused jitter when its indicator is zero may be
retained as the same independent latent original Gaussian.
Choose a proof coefficient $a_v>0$ with $4a_vB_v^2<1$.
The Gaussian square and $|v_1|^2\le2A_v^2+2B_v^2|G^J|^2$
give the primitive bound

$$
\sup_N E\,\eta_N e^{a_v|v_1|^2}
\le e^{2a_vA_v^2}(1-4a_vB_v^2)^{-d/2}<\infty.
\tag{NMG.25}
$$

It is valid for every entering law in the complete register
and every actual first-kick dependence on jitter. Bounded
continuous approximations and monotone convergence pass
this bound to a random limiting environment. Consequently
$M_\eta=\eta e^{a_v|v_1|^2}<\infty$ almost surely.
Each finite empirical preparation has finite $M_{\eta_N}$
without taking an expectation.

Let $\eta_X$ be the tilt by
$w_X=e^{-|X-m|^2/(2B)}$, $B=\rho^2+t^2q^2$.
The oscillation of its logarithm on $|m|\le M_m$ gives

$$
D(\eta_X\Vert\eta)
\le\frac{2M_m|X|}{B}+\frac{M_m^2}{2B}=D_X.
$$

This bound requires no functional inequality:
$E_{\eta_X}\log w_X-\log\eta w_X$ is at most the
supremum minus infimum of $\log w_X$.
Jensen applied under $\eta_X$ to
$\exp[f-\log(d\eta_X/d\eta)]$ proves
$E_{\eta_X}f\le D(\eta_X\Vert\eta)+\log\eta e^f$.
Use truncated $f=a_v|v_1|^2$ and then monotone convergence
to obtain

$$
E_X|v_1|\le
\sqrt{\frac{D_X+\log M_\eta}{a_v}}.
\tag{NMG.26}
$$

Completing the same OU/kernel Gaussian square, without
dividing by $a_x$, gives exactly

$$
\frac{\beta(X)}{\alpha(X)}
=\frac{tq^2}{B}X+E_X\left[cv_1-\frac{tq^2}{B}m\right].
$$

Along $z=ru$, $X=p+tr u$, equations (NMG.26) and
$|m|\le M_m$ yield

$$
u\cdot\left[\frac{\beta(p+tr u)}{\alpha(p+tr u)}-ru\right]
\le-\frac{\rho^2}{B}r+\frac{tq^2}{B}(|p|+M_m)
 +c\sqrt{\frac{D_{p+tr u}+\log M_\eta}{a_v}}
\longrightarrow-\infty.
$$

The last term grows at most as $\sqrt r$; its coefficient
is a proved primitive/moment expression. The row force
therefore has unbounded norm along this ray. The count
force is nonzero there because $\alpha>0$.
Their analyticity, the count decay for positive threshold,
the analytic-zero-set argument and nondegenerate posterior
passage are exactly those proved in the two preceding lemmas.
This proves every claimed hard boundary without replacing
the Gaussian tails by a support restriction.
:::

:::{prf:theorem} Joint finite-window geometry/B2-color correspondence
:label: thm-nmg-joint-color-geometry

Let the preparation coordinate laws and terminal tag have
any joint subsequential limit $(\eta,x)$, using the $W_2$
topology for the coordinate preparation law. These subsequences
exist from the primitive moments: for every fixed $p>2$,
$\sup_N\mathbb E\eta_N|A^{\rm coord}|^p<\infty$, since
the post-copy position has a bounded source plus its original
Gaussian jitter, the collision velocity is bounded, and the
first force/kick has linear growth. Higher-moment cores give
tightness in $W_2$, not only weak tightness. Use (NMG.6) as the local
source/residual-marked Poisson process, and evaluate each
limiting B2 input at the same terminal query $x$:

$$
z(A,x,E)=cv_1+a_Y(x-m)+q\sqrt\chi E,\qquad
X(A,x,E)=p+t z(A,x,E).
\tag{NMG.9}
$$

For either literal normalization, evaluate its force
$F_\eta^{\rm pop,\mathfrak n}(X,z)$ from the single
common stage law $\Lambda_\eta$, and then its native color
on the actually available branch

$$
c_\eta(A,x,E)=
 \frac{F_\eta^{\rm pop,\mathfrak n}(X,z)}
 {|F_\eta^{\rm pop,\mathfrak n}(X,z)|}
              \odot e^{i\kappa_c z}.
\tag{NMG.10}
$$

Every bounded continuous finite-window test of the native
positions, preparation marks, B2 inputs, viscous forces and
smooth color/projector cutoff cylinders converges to this
joint common-environment law. A cutoff may be identically
one on any actually tested force-threshold margin, and its
derivatives remain its own. Hard color availability and
zero-extension also converge whenever the computed limiting
threshold-boundary mass is zero. If that mass is positive,
the conclusion concerns the stated smooth cylinders and the
actual boundary mark is retained as an unresolved hard-mask
comparison, rather than assigned an arbitrary value.
For the count branch with its actual positive threshold,
the preceding lemma proves this null-boundary property for
every admitted limiting preparation law. In the primitive
$\lambda\ge0$, $a_x>0$, $\nu>0$ capped regime,
{prf:ref}`lem-nmg-full-threshold-boundary` proves it for
both normalizations and every $\delta_c\ge0$.
This includes the unchanged reference. Outside those
evaluated regimes the hard branch retains its boundary test;
the explicit consensus regime below also evaluates it directly.
The exact $a_x=0$ reset is covered by
{prf:ref}`lem-nmg-reset-threshold-boundary` with its derived
Gaussian preparation moments.

Protected CSR stars and their relative/absolute covariance
metric functions in Chapter NSG extend this same finite-window
correspondence to their local marked readouts by stabilization.
The metric is a function of these Poisson positions; it is
not an independent tensor. The force/color marks use the
same $\Lambda_\eta$ and the original conditional residual
noise. Unconditionally, their shared environment can be
random and they are not independent populations.

For QSD inputs, the complete rooted preparation consistency
in {prf:ref}`thm-native-stationary-closure-population-invariance`
identifies the coordinate projection of $\eta$ as the original population preparation and
B1 pushforward of its random entering law $\mu$.
In the derived count regime
{prf:ref}`thm-native-phase-stationary-chaos`, $\mu=\mu_*$
and this environment is deterministic. Neither statement
assumes spatial regularity of the limiting color law.
:::

:::{prf:proof}
The marked Poisson theorem retains the full quenched
preparation. Replacing $Y_j=x+r_Nu$ by $x$ in (NMG.3)
changes $z_j$ by exactly $a_Yr_Nu$ and $X_j$ by exactly
$ta_Yr_Nu$. On bounded windows those changes vanish,
independently of the preparation magnitudes. Truncate mark
tails using their Gaussian-weighted moment bounds.
The two force-sampling lemmas show that the expected number
of local points with a fixed positive force error tends to
zero, so with probability tending to one no such point is
present. They compare the actual global dense B2 sum, not
only the finitely many local points.

On compact mark sets, the common stage law converges in
$W_2$ under the same empirical preparation convergence:
couple convergent preparation marks and use the same
independent original Gaussian. Explicitly, weak preparation
convergence upgrades to $W_2$ on each uniform $p>2$ moment
core because the second-moment tail is at most
$H/R^{p-2}$; the primitive higher moments remove those
cores in probability. The Gaussian kernel is
bounded and Lipschitz and has linear velocity integrands;
their uniform second moments give uniform convergence of
$\alpha_N,\beta_N$ on compact $X$ sets. For row normalization
their limiting positive degree is bounded below on each
compact set; the proved moment-core argument removes
that localization. Thus the common force functions converge
where evaluated. Native phases and smooth cutoff colors
are continuous functions of those same inputs. Point-process
count tightness and bounded-test approximation prove the
finite-window convergence. A null hard-threshold boundary
permits the same approximation by cutoffs on shrinking
boundary neighborhoods; a positive atom does not.

The protected-star coupling in Chapter NSG, extended to
its actual source/residual marks, localizes every requested
cell, covariance metric and neighboring metric to a finite
random window. Rooted mark moment tails and protection
remove that window cutoff. Their configured continuous
spectral regimes and actual availability masks retain the
scopes already proved there. All colors remain evaluated
with the same bulk law, not with independent resampled
point clouds. The QSD preparation identification and its
deterministic phase specialization are exactly the proved
full rooted-component one-step consistency and phase
concentration, whose donor/collision hypotheses were
already verified in those theorems. Survival selection
changes the full marked output law by at most
$(1-a_*)^N$, so it changes none of these bounded-test limits.
:::

(sec-nmg-assembled)=
## 5. Assembled joint color/metric laws and the actual environment covariance

:::{prf:lemma} Two-root marked comparison with the same B2 environment
:label: lem-nmg-two-root-marked

Use the primitive center core, compact query set, radii and
one-neighborhood protection of {prf:ref}`lem-nga-two-root`.
Condition on $\mathcal F,Y_i=x,Y_k=y$ for two different slots,
with $|x-y|>2r_NR$; retain their independent own posterior
residuals and their actual preparation marks. The two local
source/residual-marked configurations couple to independent
Poisson configurations with intensities

$$
du\ \rho_N(x)\pi_{N,x}(dA,dE),\qquad
du\ \rho_N(y)\pi_{N,y}(dA,dE)
\tag{NMG.14}
$$

at error at most $\epsilon_N(R)+2\eta_M(R)$ from (NGA.2).
The root residuals remain conditionally independent, and
the two comparison configurations use the SAME bulk force
functions $F_N^{\rm pop,\mathfrak n}$ and the same frozen
calibration. Root/neighbor colors are computed from their
own retained marks by (NMG.3), using their actual spatial
position arguments. Every covariance metric, volume and
geodesic length is computed from its own Poisson configuration
and its actual adjacent stars.

Consequently bounded tests of those common-environment
color/metric observations factor conditionally on the
preparation and the two root marks, up to the stated error.
This is a comparison for the population-force color
observations. For the original global finite-$N$ B2 colors,
the own-root and locally selected force estimates above add
an error tending to zero for smooth color cutoff tests,
or for hard branches covered by the two threshold lemmas.
No finite-$N$ color independence is asserted.
:::

:::{prf:proof}
Append each independent posterior residual and its frozen
source mark to the Bernoulli/Poisson coupling on the union
of the two disjoint windows in {prf:ref}`lem-nga-two-root`.
This does not change the count probabilities or their
Gaussian derivative bounds. Deleting the two addressed
root source labels has intensity error at most
$2L_{\tau,0}/N$; it is already included in (NGA.2).
The homogeneous marked Poisson restrictions on the two
windows are independent by their Poisson generating
functions. The original root residuals use different
Gaussian pairs and are independent on the conditioned
preparation. Coupling their same residuals in the
comparison adds no error.

On successful coupling and protection all point coordinates
and appended marks agree inside the two determining
windows. Evaluate the same common force functions and
same geometric estimator on each; the readouts then agree,
including their correlations inside a neighborhood.
For original B2 colors, truncate only the proof to bounded
mark counts and input magnitudes. The count/row force
lemmas control every root and local mark by union bound.
Native Gaussian moment tails remove the input cutoff;
Poisson count tightness and protection remove the count
and window cutoffs. Smooth color cylinders are continuous
on those bounds. For the covered hard color branches, the derived null
threshold boundary makes its shrinking boundary-band
probability tend to zero. This proves the additional
vanishing bounded-test error. No test substitutes a force
calculated from only its local window.
:::

:::{prf:theorem} Empirical joint native color/metric law
:label: thm-nmg-assembled-law

Let $H_N$ be a bounded ($|H_N|\le B$) one-neighborhood
observation of the original local terminal geometry,
covariance metric, weights, root/neighbor B2 inputs and
color/projector marks. It is zero outside a fixed compact
tag set. Use smooth color cutoff cylinders, or a hard
branch covered by the threshold lemmas and bounded continuous
tests of its actual availability-marked outputs.
For other hard branches retain their evaluated null
boundary tests. Its actual numerical scale, relative
ridge and floors may depend on $r_N$ through the existing
formulas, as in Chapter NSG.
Here the tests are fixed bounded continuous functions of the
declared scaled native readouts; their color cutoff profiles
are fixed. Arbitrarily sharpening a test with $N$ is not part
of this assertion. For the limiting display below, $H$ is the
test of the jointly limiting estimator/color variables in
Chapter NSG and the joint color theorem. Configured scales
without a limit retain their actual conditional comparator
rather than an assigned $H$.

Define its complete-population conditional comparator

$$
\Phi_{H,N}(\mathcal F)=\int\rho_N(x)
 E_{\pi_{N,x},\Pi_{\rho_N(x)\pi_{N,x}}}
           H_N^{\rm pop}(x,\text{marked neighborhood})\,dx,
\tag{NMG.15}
$$

where its bulk force is (NMG.7), its root uses
$\pi_{N,x}$, and all local metric/color data are evaluated
on that same marked neighborhood. Then the actual assembled
observation satisfies

$$
\frac1N\sum_iH_N(Y_i,\text{native neighborhood}_i)
          -\Phi_{H,N}(\mathcal F)\longrightarrow0
                    \quad\hbox{in }L^2.
\tag{NMG.16}
$$

Along the joint $W_2$ preparation/consumed-calibration
subsequence, its limit is the parameter-determined functional

$$
\Phi_H(\eta)=\int\rho_\eta(x)
 E_{\pi_{\eta,x},\Pi_{\rho_\eta(x)\pi_{\eta,x}}}
             H(x,\text{marked neighborhood})\,dx.
\tag{NMG.17}
$$

The environment $\eta$ remains random unless its deterministic
phase has been proved. QSD selection and numerical graph/
payload errors have exactly the transfer scopes in
{prf:ref}`thm-nsg-qsd-transfer`. The original alive average,
when it is the configured normalization, has the corresponding
limit $\Phi_H^{\rm alive}(\eta)=
\Phi_{\mathbf1_DH}(\eta)/\int_D\rho_\eta$.
Its denominator is bounded below in probability by the
proved actual alive landing certificate; no deterministic
finite-swarm alive floor is inserted.
:::

:::{prf:proof}
First replace each original local B2 color by its
common-stage-force counterpart, on the same original
positions/residual marks. The root and local-neighbor
force approximations show that the fraction of observations
affected by any fixed force error tends to zero in mean.
Stabilize to a fixed determining window, truncate large
marks/counts in the proof, and use uniform continuity of
the bounded color cylinders. Then remove those cutoffs.
Every covered hard-mask case uses its proved null boundary.
The difference of empirical means is bounded by $2B$;
convergence in mean therefore also gives $L^2$ convergence.

For the common-force observations the proof of
{prf:ref}`thm-nga-empirical-law` applies with the appended
source/residual marks and the frozen $\eta_N$ force
functions. One-root bias averages its true root densities
and root marks to exactly (NMG.15), by
{prf:ref}`cor-nmg-root-mark-law`. Split the second moment
into diagonal pairs, close pairs and separated pairs.
The first two have the same $B^2/N$ and
$2B^2L_{\tau,0}v_d(2R)/N$ bounds. The marked two-root
lemma factors separated comparison observations on the
conditioned preparation. Their averaged products equal
$\Phi_{H,N}^2$, with the same missing diagonal correction.
Thus (NGA.4) holds for these marked common-force
observations, with the same core/protection errors.
Take $N$, then $R$, then $M$ to infinity and combine
with the original-color replacement to obtain (NMG.16).

The coordinate preparation $W_2$ convergence, Gaussian
posterior formulas and common-stage force convergence
were proved in the joint color theorem. The tilted
mark density passes to the limit because its denominator
is positive. Poisson windows, conditional mark integrals
and relative/absolute metric formulas therefore converge
on each protected bounded configuration. Bounded tests,
mark/count tightness, protection and spatial moment
tails justify (NMG.17), retaining any actual consumed
calibration in that same joint limit.

For alive normalization first use compact interior tag
sets, then remove the box boundary layer using its
Gaussian density upper bound. The actual alive fraction
concentrates conditionally about $\int_D\rho_N$ with
variance at most $1/(4N)$. The full original binomial
landing estimate puts this conditional denominator above
$a_*/4$ in probability, as proved in
{prf:ref}`thm-nga-native-curvature`. Dividing the bounded
numerator gives the asserted alive average. Every ratio
is the actual existing normalization. The QSD and
arithmetic comparison bounds pass these bounded tests
with the errors already stated in Chapter NSG.
:::

:::{prf:corollary} Long-range classical covariance is the common-environment covariance
:label: cor-nmg-environment-covariance

For two bounded observations $H,G$ in the preceding
theorem, write their assembled empirical averages as
$A_{H,N},A_{G,N}$. Along the same joint subsequence,

$$
\operatorname{Cov}(A_{H,N},A_{G,N})\longrightarrow
 \operatorname{Cov}(\Phi_H(\eta),\Phi_G(\eta)).
\tag{NMG.18}
$$

This includes disjoint spatial support sets, and preserves
their random-phase covariance. For two uniformly sampled
distinct roots with those bounded spatially localized
observations, their product expectation tends to
$E[\Phi_H(\eta)\Phi_G(\eta)]$; their limiting covariance
has the same environment term. Under the actual deterministic
count phase in {prf:ref}`thm-native-phase-stationary-chaos`,
and fixed consumed calibration/current marks covered by
that preparation consistency, $\eta$ is deterministic;
these environment covariance terms vanish.

At distinct fixed queries $x,y$, the corresponding
conditional two-root environment is weighted by
$\rho_\eta(x)\rho_\eta(y)$ under its original limiting
law. In that conditional ensemble the covariance is the
covariance of the two conditional local means under that
weighted environment law. It is not a covariance under
the unweighted phase law. These are classical probability
statements; they assert neither a $\sqrt N$ fluctuation
limit nor physical spacelike commutation.
:::

:::{prf:proof}
Equation (NMG.16) gives bounded $L^2$ errors for both
observations. The covariance changes by a quantity tending
to zero when each is replaced by its $\Phi_{H,N}$ or
$\Phi_{G,N}$. These bounded conditional means converge
jointly in law to (NMG.17), so their first and product
moments converge, proving (NMG.18). Equivalently the
conditional covariance term tends to zero and the law
of total covariance retains the covariance of means.

For uniform distinct roots, conditional sampling of the
empirical list gives
$E[H_IG_J\mid\text{record}]=(NA_{H,N}A_{G,N}
-N^{-1}\sum_iH_iG_i)/(N-1)$.
The diagonal correction is $O(B_HB_G/N)$; the preceding
product convergence proves the claim. At conditioned
positions the root-mark likelihoods are exactly the two
Gaussian density factors in (NMG.13). Their joint
environment Bayes weight is therefore
$\rho_\eta(x)\rho_\eta(y)$, and the separated two-root
marked comparison gives independent local residual
configurations conditional on that environment.
This proves its stated conditional covariance decomposition.
The proved deterministic phase makes the functions of
that environment constants. It does not change the
underlying operator algebra or supply a central limit
normalization.
:::

(sec-nmg-actual-regime)=
## 6. Terminal color-density regularity and an explicit fluctuating regime

:::{prf:lemma} Exact native terminal color-density regularity
:label: lem-nmg-terminal-color-density

Retain the matched B2 projector instrument with its native
pre-terminal force/finite-value/source masks and fixed or
preceding calibrated phase, independent of the current final
position innovation. Let $P_i^{\rm B2}$ be its projector on
an available row and its declared zero extension otherwise.
All original B2 force correlations remain in this bounded
matrix, with $\|P_i^{\rm B2}\|_{\rm op}\le1$.
Conditional on the complete preparation, the exact raw
terminal color-density profile is

$$
\Psi_{N,\mathcal F}(y)=
\mathbb E_O\left[\frac1N\sum_iP_i^{\rm B2}
                     \varphi_s(y-X_i)\ \middle|\ \mathcal F\right].
\tag{NMG.19}
$$

For every bounded spatial test $f$ it satisfies
$E[N^{-1}\sum_i f(Y_i)P_i^{\rm B2}\mid\mathcal F]
=\int f(y)\Psi_{N,\mathcal F}(y)dy$.
Because $s>0$, it is $C^\infty$ with the primitive bounds
$\sup_y\|D^k\Psi_{N,\mathcal F}(y)\|_{\rm op}
\le L_{s,k}$ for all $k\ge0$, uniformly in $N$ and
every actual preparation. Final alive-only restriction
multiplies this profile by $\mathbf1_D(y)$; smoothness
holds on the actual interior, with its literal boundary
mask retained. QSD transfer concerns bounded tests and
does not remove that boundary.

Along the joint preparation/common-force limit for the
hard branches covered by the threshold lemmas, or smooth
cutoff/evaluated other mask branches, the profile has the common-environment
limit

$$
\Psi_\eta(y)=\int\eta(dA)\mathbb E_G\left[
 P_\eta(p+tz,z)\varphi_s(y-p-tz)\right],
\quad z=cv_1+qG.
\tag{NMG.20}
$$

It has the same spatial derivative bounds.
This is a matrix-valued law density and need not be
rank one or idempotent. It is not substituted for a
native individual projector or used to invent an
eigenline transport.
:::

:::{prf:proof}
Condition on the complete O/B2 record. Every final
position noise remains independent of that record,
and $Y_i=X_i+s\zeta_i$ has density
$\varphi_s(y-X_i)$. Integrate its bounded projector
and average to obtain (NMG.19). Gaussian derivatives
are bounded by $L_{s,k}$ and the projectors by one;
dominated differentiation of every order proves
the regularity estimates despite all B2 coupling
and force-threshold discontinuities. The final alive
mask is exactly the stated multiplication.

For the limiting display, use the count/row force
closure and native null-boundary or cutoff passage
already proved for empirical color tests. The bounded
Gaussian kernels and each of their derivatives give
the common-stage integral (NMG.20), by localization
and the same preparation moment tightness.
Their uniform next-derivative bounds make these
profiles equicontinuous on every compact set;
therefore the convergence of the corresponding
bounded smeared tests identifies the profile limit
there. The displayed integral itself is smooth
by the same dominated Gaussian differentiation.
No sample-field regularity or idempotency is inferred.
:::

:::{prf:theorem} Fixed-step consensus preparation gives nonvanishing microscopic colors
:label: thm-nmg-consensus-color-regime

For an actual allowed nonempty entering state with every
position and velocity zero, all rows eligible, quadratic
force and the configured positive fitness maps, both
measurement channels agree across rows. Every living
acceptance is exactly zero; source sampling is retained but
there is no accepted clone. B1 and A1 have $p=v_1=m=0$.
This is a one-step native kernel regime, rather than an
assertion that a fixed-reference QSD is concentrated at
consensus. Keep $h,q,s,\nu,\rho>0$ and the literal B2 color.
Put

$$
B=\rho^2+t^2q^2,\quad a_0=(\rho^2/B)^{d/2},
\quad r_0=\rho^2/B.
$$

The same common population B2 force is explicitly

$$
F^{\rm pop,count}(tz,z)=
-\nu a_0r_0e^{-t^2|z|^2/(2B)}z,
\qquad F^{\rm pop,row}(tz,z)=-\nu r_0z.
\tag{NMG.11}
$$

At any finite terminal query $x$, the local posterior is
$z=a_Yx+q\sqrt\chi E$, with full Gaussian support.
For an alive-restricted channel use $x\in D^\circ$; all its
other declared masks remain in the record.
In the count branch the available-color region is the
literal radial set
$\{r>0:\nu a_0r_0r e^{-t^2r^2/(2B)}>\delta_c\}$.
It is nonempty precisely when
$\delta_c<\nu a_0r_0\sqrt B/(t\sqrt e)$.
At equality it has zero Gaussian mass; above it the color
is unavailable almost surely. In the row branch it is
$\{|z|>\delta_c/(\nu r_0)\}$ for any finite threshold.
All its threshold spheres have posterior probability zero.
On the available region, the actual color is

$$
c(z)=-\frac z{|z|}\odot e^{i\kappa_c z}.
\tag{NMG.12}
$$

If
$\delta_c<\nu a_0r_0e^{-t^2/(2B)}$ for count, or
$\delta_c<\nu r_0$ for row, neighborhoods of
$z=e_1$ and $z=(e_1+e_2)/\sqrt2$ are available.
Their projector distance is bounded away from zero and
their mutual overlap is bounded away from zero.
The local marked Poisson field consequently has positive
probability of adjacent available projectors with an
order-one difference at a physical CSR separation of
order $r_N$. The canonical native projector transport
of {prf:ref}`thm-npc-native-direct-rotation` does not have
$O(r_N)$ increments on this event. Its difference from
identity divided by $r_N$ is not tight in this specified
unchanged-step kernel family.

For the unchanged numerical force/noise reference values,
the posterior standard deviation is
$q\sqrt\chi=.19240259384554195\ldots$;
the count force magnitude at unit $z$ is
$.29992847699047004\ldots$, and the row magnitude is
$.29999538705171525\ldots$.
Thus the actual threshold $10^{-12}$ passes these strict
tests. This proves a native fluctuating microscopic regime,
not a failure of the algorithm or a prohibition of a weak,
distributional, averaged or differently calibrated field limit.
:::

:::{prf:proof}
At consensus the standardized reward and diversity
numerators vanish and all positive fitnesses agree.
Every literal living clone gate is zero. The potential
and viscous forces at B1 vanish. The stage population is
$(X,z)=(tqG,qG)$ with its original Gaussian law.
Completing the Gaussian square in its dense kernel gives

$$
\alpha(X)=a_0e^{-|X|^2/(2B)},\qquad
\beta(X)/\alpha(X)=\frac{tq^2}{B}X.
$$

Substitute $X=tz$ into (NMG.7) to obtain (NMG.11).
The count radial force has its unique maximum at
$r=\sqrt B/t$; differentiation proves the exact
threshold classification. The row expression is linear.
Their boundary sets are finitely many spheres or a
singleton and have Gaussian measure zero. Directional
normalization gives (NMG.12) with its configured phase.

At the two chosen unit directions the projectors have
distance $1/\sqrt2$ in operator norm and overlap modulus
$1/\sqrt2$, independently of $\kappa_c$.
Continuity preserves strict force, overlap and distance
margins in small neighborhoods. Their posterior Gaussian
probabilities are positive at every fixed terminal $x$.
The marked Poisson comparison has independent residual
marks conditionally on this deterministic preparation.
Take a protected simplex star event from Chapter NSG;
assign the tag and one of its actual neighbors to those
two mark neighborhoods. That joint event has positive
probability, with a finite star edge length bounded
above and below. The force-sampling and marked-star
correspondence transfer its strict margins to the original
finite-population native kernel with positive limiting
probability. The direct-rotation identity
$\|U-I\|=2\sin[\tfrac12\arcsin\|P_j-P_i\|]$
then bounds its increment below by a positive constant.
Dividing by $r_N\to0$ proves nontightness. The reference
numbers follow by substituting the actual $q,s,t,\nu,\rho$.
All O and final-position draws retain their unbounded law.
:::

(sec-nmg-stationary-microscopic-field)=
## 7. Stationary native projector variation and neighboring energy

:::{prf:lemma} A primitive positive-threshold availability certificate
:label: lem-nmg-primitive-availability

Retain the complete capped quadratic preparation, including
its original standard jitter latent $G^J$ and its literal
jitter indicator. Every admitted limiting preparation law
has a joint extension with

$$
X^J=S+I\sigma_JG^J,\quad |S|\le R_D,\quad |U|\le U_*=
\kappa_\nu V_c,\quad G^J\sim\gamma_d.
$$

This extension retains all correlations with $U$, source,
component and current history. It does not assume their
independence. Unused jitter is only an unconsumed Gaussian
latent in the same spatial-kernel representation.

For $j>0$ put $p_j=\Pr(|G^J|\le j)$, and define the
following primitive profiles for $R\ge0$:

$$
\begin{gathered}
B=\rho^2+t^2q^2,\quad C_B=(\rho^2/B)^{d/2},\quad
M_j=|a_x|(R_D+\sigma_Jj)+bU_*,\\
D_j(R)=-\log p_j+\frac{(R+M_j)^2}{2B},\qquad
g_j(R)=2\sqrt{D_j(R)+\frac d2\log2},\\
C_j(R)=c[U_*+t\lambda(R_D+\sigma_Jg_j(R))]
 +\frac{tq^2}{B}
 [R+|a_x|(R_D+\sigma_Jg_j(R))+bU_*],\\
L_j(r)=\nu C_Bp_j e^{-(tr+M_j)^2/(2B)}
                      [r-C_j(tr)]_+\quad(r>0).
\end{gathered}
\tag{NMG.27}
$$

For every finite own preparation $p$, the exact count
population-force profile satisfies

$$
\max_z|F_\eta^{\rm pop,count}(p+tz,z)|\ge L_j(r).
\tag{NMG.28}
$$

Thus any computed $\delta_c<L_j(r)$ gives a nonempty
open available set at every own preparation and every
finite terminal query. The original residual posterior
assigns that set positive probability. Failure of this
sufficient test is not assigned unavailability: the
exact profile remains its configured Gaussian integral
and count availability occurs precisely when its maximum
exceeds $\delta_c$.

At the unchanged reference, $j=3$ and $r=10$ give

$$
p_j=.9707091134651118\ldots,\quad
M_j=3.9180125259919967\ldots,\quad
C_j(tr)=3.9250090336473327\ldots,
$$
$$
L_j(r)=.00036763241127807725\ldots>10^{-12}.
\tag{NMG.29}
$$

This proves availability in every admitted stationary
preparation environment of that original reference.
:::

:::{prf:proof}
The original standard jitter array has independent Gaussian
rows regardless of the pre-jitter history; unused rows can
be retained as independent latent rows. For every bounded
test its empirical variance is $O(N^{-1})$. Consequently
its empirical marginal converges to $\gamma_d$ in
probability. Uniform Gaussian and preparation moments give
tightness of the joint coordinate/jitter empirical laws;
every subsequential joint extension has that Gaussian
marginal. The closed source, jitter-indicator, first-kick
and collision bounds pass to this limit. None of these
steps factorizes the joint preparation.

For $|X|\le R$ let $w_X=e^{-|X-m|^2/(2B)}$.
On $|G^J|\le j$, the actual $m=a_xX^J+bU$ satisfies
$|m|\le M_j$, proving

$$
\eta w_X\ge p_j e^{-(R+M_j)^2/(2B)},\qquad
\alpha(X)=C_B\eta w_X.
$$

The tilted law has relative entropy at most
$-\log\eta w_X\le D_j(R)$, since $\log w_X\le0$.
The Gaussian marginal gives exactly
$\eta e^{|G^J|^2/4}=2^{d/2}$.
The elementary entropy/Jensen argument in the reset
lemma therefore yields
$E_X|G^J|\le g_j(R)$.
Using $v_1=U-t\lambda X^J$ and $m=a_xX^J+bU$,

$$
E_X|v_1|\le U_*+t\lambda(R_D+\sigma_Jg_j(R)),
$$
$$
E_X|m|\le |a_x|(R_D+\sigma_Jg_j(R))+bU_*.
$$

The exact Gaussian square
$\beta/\alpha=(tq^2/B)X+E_X[cv_1-(tq^2/B)m]$
now proves $|\beta(X)/\alpha(X)|\le C_j(R)$.
Given $p\ne0$, take $u=-p/|p|$ and
$z=-p/t+ru$; for $p=0$ take any unit $u$.
Then $X=p+tz=tr u$ and $|z|\ge r$.
The reverse triangle inequality and the proved degree
lower bound give (NMG.28), with its literal count
coefficient. A strict positive gap gives an open
available neighborhood; the actual posterior is
nondegenerate Gaussian. The maximum exists because
the continuous count force tends to zero at infinity,
as proved above. Substituting the recorded reference
coefficients into these formulas gives (NMG.29).
:::

:::{prf:theorem} Native stationary posterior projector variation
:label: thm-nmg-stationary-projector-variation

Use the complete real-coordinate QSD/preparation family
of this chapter and its admitted common environment $\eta$.
Let $d=3$, $\lambda\ge0$, $a_x\ge0$, and
$t,q,s,\rho,\nu>0$, with the original finite cap and
finite consumed $\kappa_c$. Use the matched B2
force/velocity channel with its literal force threshold
and finite-value masks. An extra source-deletion or
clone-deletion mask retains its own selected integrals;
it is not assigned the positivity conclusion below.

For row normalization allow every finite $\delta_c\ge0$.
For count normalization allow $\delta_c=0$, or a positive
threshold passing (NMG.28). These include both unchanged
reference normalizations at $\delta_c=10^{-12}$.
For each finite $x$, let $a(A,x,E)$ be its original
available mark and $P(A,x,E)$ its available rank-one
projector. With its actual posterior tilt
$\pi_{\eta,x}$ from (NMG.13), define

$$
a_\eta(x)=\int a\,d\pi_{\eta,x}>0,\qquad
M_\eta(x)=\int aP\,d\pi_{\eta,x},
$$
$$
\mathcal V_\eta(x)=
2\left[a_\eta(x)^2-\operatorname{tr}M_\eta(x)^2\right]>0.
\tag{NMG.30}
$$

These are exact parameter-computed posterior Gaussian
integrals of the actual same-environment force law.
The available projector distribution has no atoms and
is not a fixed line. In particular the positivity in
(NMG.30) is proved for $\eta$, rather than assumed of an
unknown stationary color law. At $\delta_c=0$ its full
channel has $a_\eta(x)=1$. The deterministic phase
specialization uses the original $\eta_*$ furnished by
{prf:ref}`thm-native-phase-stationary-chaos`; no consensus
input or consensus stationary premise is introduced.
For alive-restricted colors use $x\in D^\circ$ and retain
the literal final alive mask.
:::

:::{prf:proof}
For each fixed own $p$, the row radial projection is
eventually negative with linear growth along every ray,
by the positive-$a_x$ and reset proofs. Thus its force
cannot lie in a fixed real line globally: choosing a
unit ray direction perpendicular to that line would
contradict its strictly negative unbounded projection.
The count force is $\alpha>0$ times this row vector
and likewise cannot lie in a fixed real line.
Row availability is a nonempty open set for any finite
threshold. Count availability is nonempty at zero
threshold by analytic nontriviality, or at a certified
positive threshold by the preceding lemma. Every own
posterior assigns each such open set positive mass.

Suppose its projector is constant on a nonempty open
available set. At $\kappa_c=0$ the real force there lies
in the fixed real line of that projector. Real analytic
continuation of its orthogonal components puts the
entire force in that line, contradicting the ray result.
At $\kappa_c\ne0$, if two components of the constant
projector are nonzero, their off-diagonal entry is a
nonzero real amplitude times
$e^{i\kappa_c(z_a-z_b)}$. Its argument cannot be constant
on an open $z$ set, even allowing the two possible signs
of the real amplitude. A constant projector must then
be a single coordinate line. All other force components
vanish on that open set; analytic continuation and the
same ray argument again contradict this.

More precisely, for any fixed candidate projector $P_0$
at least one real or imaginary component of

$$
F_a(z)F_b(z)e^{i\kappa_c(z_a-z_b)}
                    -(P_0)_{ab}|F(z)|^2
$$

is a nonzero global real analytic function. Otherwise
the projector would equal $P_0$ on every nonzero-force
open set, which was just excluded. Its zero set is
Lebesgue null by the already proved analytic-zero-set
argument. The set of available $z$ giving $P=P_0$ is
contained in it. The full Gaussian residual posterior
therefore has no projector atoms. Integrating over the
actual preparation marks preserves this no-atom property.
The null threshold lemmas justify the declared hard masks.

Conditioned on availability, the mean projector is
$\overline M=M_\eta/a_\eta$ with trace one, and

$$
E[\|P-\overline M\|_F^2\mid a=1]
=1-\operatorname{tr}\overline M^2>0.
$$

Equality would make $P$ constant almost surely, contrary
to the no-atom proof. For two independent marks from
the same posterior, their available neighboring energy
is therefore exactly
$2[a_\eta^2-\operatorname{tr}M_\eta^2]$ and is positive.
All masks, phases and forces inside this integral are
the original channel. The QSD preparation identification
and deterministic phase are the previously proved
ones; only their newly derived posterior variation is used.
:::

:::{prf:theorem} Stationary native neighboring energy and microscopic transport scale
:label: thm-nmg-stationary-neighbor-energy

In the preceding primitive regimes let $f\ge0$ be a
fixed nonzero bounded continuous spatial test with
compact support in $D^\circ$. Let $w_{ij}^N$ be the
configured native geometric CSR weights with the existing
`WeightSpec.normalize=true`, retaining their
actual floors and covariance metrics, in one of the
convergent regimes of Chapter NSG. Write $a_i^N,P_i^N$
for the actual matched B2 availability and projector.
The bounded existing-descriptor neighboring statistic

$$
\mathcal E_N(f)=\frac1N\sum_i f(Y_i)
\sum_{j\sim i}w_{ij}^N a_i^Na_j^N
                         \|P_i^N-P_j^N\|_F^2
\tag{NMG.31}
$$

has the same-environment limit

$$
\mathcal E_\eta(f)=\int f(x)\rho_\eta(x)
            T_\eta(x)\mathcal V_\eta(x)\,dx,
\qquad T_\eta(x)=E_{\Pi_{\rho_\eta(x)}}\sum_{j\sim0}w_{0j}^*.
\tag{NMG.32}
$$

For uniform CSR weights or the actual Euclidean Gaussian
kernel with $\ell_N/r_N\to\infty$, $T_\eta(x)=1$.
For $\ell_N/r_N\to L\in(0,\infty)$ it is strictly
positive with its literal $10^{-12}$ floor.
The fixed-positive-length relative-metric weights in
{prf:ref}`cor-nsg-relative-metric-weights` likewise have
strictly positive row mass in their successful estimator
regimes. Thus (NMG.32) is strictly positive in these
existing readout regimes. If the actual kernel/floor
regime instead makes row mass vanish, (NMG.32) is zero;
the proof does not renormalize those weights.
With `normalize=false` the local conditional mark identity
still holds for the actual raw row mass, but its unbounded
assembled moment passage is outside this bounded theorem.
Under the proved deterministic stationary phase,
$\mathcal E_N(f)\to\mathcal E_{\eta_*}(f)>0$ in $L^2$.
Under a random stationary environment it retains the
random functional (NMG.32). QSD and numerical agreement
transfer retain exactly their preceding scopes.

Moreover, a uniformly rooted actual CSR star has, with positive
limiting probability, an available edge with
$|Y_j-Y_i|\le Cr_N$ and

$$
\|P_j-P_i\|_{\rm op}\ge\epsilon,\qquad
|c_i^\dagger c_j|\ge\epsilon
\tag{NMG.33}
$$

for some $C<\infty$, $\epsilon>0$. Its canonical
composite transport from Chapter NPC consequently
satisfies $\|U_{ji}-I\|\ge2\sin[\tfrac12\arcsin\epsilon]>0$.
The native neighboring projector increments and this
specific canonical transport, divided by their physical
edge lengths, are not tight at the $r_N$ gradient scale
in the actual stationary reference. This is a proved
microscopic field statement; it neither substitutes
that composite for literal edge transport nor excludes
the already derived smooth law density, smeared fields
or a separately identified physical action/field limit.
:::

:::{prf:proof}
Every configured row has $\sum_jw_{ij}^N\le1$, so
$0\le\mathcal E_N(f)\le2\|f\|_\infty$.
The assembled law applies to this bounded joint
color/geometry observation, using its proved hard-mask
boundaries and protected stars. In its homogeneous
marked Poisson comparison the root and neighbor marks
have the same $\pi_{\eta,x}$ law and are independent
of the unmarked positions conditionally on $\eta,x$.
Their color force is still the SAME dense bulk force;
independence is only this derived conditional marked-law
factorization. The metrics and weights remain functions
of the same unmarked Poisson configuration and its
adjacent stars. Integrating its independent color marks
therefore gives exactly $\mathcal V_\eta(x)$ on each
edge and proves (NMG.32), including the actual row mass.
The NSG weight formulas give the stated positive or
vanishing $T_\eta$ regimes. Since $\rho_\eta(x)>0$,
$\mathcal V_\eta(x)>0$ and $f$ is positive on a
nonempty open set, their integral is strictly positive
whenever $T_\eta>0$ there. The deterministic-phase
$L^2$ assertion follows from the bounded assembled
law and its bounded constant limiting functional.

For the edge statement, condition on $\eta,x$.
Two independently available projectors are distinct
almost surely by their no-atom law. Also
$E\operatorname{tr}(P_0P_1)=\operatorname{tr}\overline M^2
\ge1/3$; hence a positive-probability subset has
nonzero overlap as well as positive distance.
The countable union of positive distance/overlap
margin events contains this subset, so some strict
pair of margins has positive probability.
There is a protected finite Poisson star with a
neighbor and bounded scaled edge lengths with positive
probability. Its color marks have precisely these
independent posterior laws, so the joint edge event
has positive probability. Integrate over the original
environment and tag density and choose common finite
$C$ and positive $\epsilon$ from the countable union.
Choosing strict margins off their possible scalar
boundary atoms lets the same-record marked-star law
transfer a positive lower probability to the original
QSD update; its final survival selection has vanishing
total-variation error. The exact direct-rotation norm
formula then proves the transport bound. Dividing by
an edge length at most $Cr_N\to0$ gives a fixed positive
probability beyond every proposed finite gradient bound,
which is the claimed nontightness.
:::

:::{prf:remark} Scope of the joint marked discharge
:label: rem-nmg-remaining-correspondence

The exact posterior, source-marked local Poisson law,
actual dense B2 force approximation at spatially selected
sites, and common-environment color/metric correspondence
are now derived from the executed record. Neither the
geometry nor the color is replaced by an independent
comparison population. Subsequence environments and the
proved deterministic phase regime have their respective
scopes. The explicit consensus transition additionally
shows that shrinking terminal graph edges need not give
shrinking native color-projector increments.

Physical spatial scaling therefore requires its actual
readout and normalization: smooth cutoff marked fields,
assembled/averaged observables, calibrated transport,
native action and first variations, and physical-time
reconstruction cannot be inferred from the unmarked star
law. Hard-mask boundary mass outside the derived nonnegative-$a_x$
quadratic regime and the other evaluated cases,
global source-label tightness, numerical payload agreement,
other feedback/landscape variants and a full physical
Yang--Mills field limit retain their explicit obligations.

The excluded degenerate noise parameters have exact posterior
interpretations: if $s=0<q$, then $\tau=tq$ and O is determined
by terminal $Y$; the residual mark is absent. If $q=0<s$,
then $z=cv_1$ is deterministic on preparation and the spatial
Gaussian is the final-position draw. If both vanish, the fresh
spatial Poisson approximation here is unavailable; included
consensus positions can remain coincident. At $\nu=0$ a
positive force threshold rejects every viscous color. At
$\kappa_c=0$ the consensus color is real but its orientation
projector still fluctuates. The proof never declares that
real field to be a non-Abelian physical gauge limit.
:::
