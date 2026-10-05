# Harmonic reference velocity burn and uncut cap feedback

(sec-rvb-record)=
## 1. Actual harmonic count record

:::{prf:definition} The velocity-moment arm and its population carrier
:label: def-rvb-record

Use the harmonic reference arm of research records 27, 29 and 33:
$F(x)=-x$, $D=[-L,L]^d$, $d=3$, $L=2$, $h=.04$,
$\nu=.3$, $\gamma=b_O=\rho=1$, $\sigma_J=\sigma_x=.1$,
$V=2$ and $\alpha_{\rm col}=.5$. Set
$$
t=h/2,\quad c=e^{-h},\quad b=t(1+c),\quad
m=1-t^2,\quad a_x=1-tb,\quad
q^2=(1-c^2)/2,\quad s^2=\sigma_x^2h,\quad
\ell=e^{-1/2}.
$$
The configured reward, current-frame measured fitness normalizers,
eligible donor probabilities, accepted edges, mandatory revival,
component Haar marks and Gaussian recipient jitters are retained.
Every stored original velocity, alive or dead, is capped by $V$.
Donor velocities are not copied. On a component $C$ the canonical
prepared velocity is
$$
v_i^{\rm p}=\bar v_C+
 \alpha_{\rm col}O_C(v_i-\bar v_C).
$$
All Gaussian innovations remain uncut, and dead coordinates outside
$D$ remain unrestricted.

For finite $N$, start each proposed update from a consistent
nonempty-alive array. Let $X_i=x_{{\rm src},i}+I_iJ_i$ be its
actual prepared positions; every selected source is in $D$ and the
$J_i$ are independent $N(0,\sigma_J^2I_d)$ innovations sampled
after the discrete source/component plan. The first count average
and the complete kinetic stages are
$$
U=A_Xv^{\rm p},\quad u=U-tX,\quad
w=cu+q\xi,\quad y=X+bu+tq\xi,\quad
z=(I-t\nu L_y)w-ty,\quad
x^+=y+s\chi,\quad v^+=C_V(z).
$$
Here $L$ is the actual symmetric count operator with width one and
$C_V(z)=Vz/(V+|z|)$ is the native smooth cap. A singleton has zero
count operator. Its looser bounds below remain valid.

For the population statements use the actual marked mean-field
recursion of {prf:ref}`cor-cg-mf-full-update`, in its declared
one-step consistency regime, with the canonical rooted preparation
law. In particular its rooted component law is the uniform-root
finite-swarm limit and is finite almost surely. These are the
existing population hypotheses, rather than independent collision
outputs. The finite-array results below do not require a population
limit or a small acceptance probability.

This is the harmonic force arm. A configured Rastrigin force has
additional nonlinear kick terms and cannot import the harmonic
velocity threshold below.
:::

(sec-rvb-energy)=
## 2. Energy through the actual source, collision and two count kicks

:::{prf:lemma} Original-slot collision energy and exact source moment
:label: lem-rvb-preparation-energy

For every actual source/component plan, the prepared velocities obey
$$
\frac1N\sum_i|v_i^{\rm p}|^2\le\frac1N\sum_i|v_i|^2,\qquad
|v_i^{\rm p}|\le V_c=(1+2|\alpha_{\rm col}|)V=4.
\tag{RVB.1}
$$
For every nonempty-alive entering array, without a retained-position
moment hypothesis,
$$
\mathbb E\frac1N\sum_i|X_i|^2
\le X_2^2:=d(L^2+\sigma_J^2)=12.03.
\tag{RVB.2}
$$
The same two averaged bounds hold for the actual population
preparation with its own entering law.
:::

:::{prf:proof}
On any fixed actual component,
$$
\sum_{i\in C}|v_i^{\rm p}|^2
=|C||\bar v_C|^2+
 \alpha_{\rm col}^2\sum_{i\in C}|v_i-\bar v_C|^2
\le\sum_{i\in C}|v_i|^2.
$$
The identity uses $\sum_{i\in C}(v_i-\bar v_C)=0$ and the
orthogonality of its common Haar matrix. It holds before expectation,
for every active/revival forest. The triangle inequality gives
$|v_i^{\rm p}|\le V+2|\alpha_{\rm col}|V=V_c$.
The incoming velocities in this calculation are the original frozen
slot velocities; there is no donor-velocity mixture.

Conditional on the complete discrete plan, each source has norm
at most $\sqrt dL$, $I_i$ is fixed, and the independent recipient
jitter has mean zero. Hence
$\mathbb E(|X_i|^2\mid{\rm plan})
=|x_{{\rm src},i}|^2+I_id\sigma_J^2
\le d(L^2+\sigma_J^2)$.
This holds for a mandatory revival regardless of its retained dead
position.

For the population preparation the source calculation is identical
at its tagged root. The collision energy inequality is inherited
from the canonical uniform-root finite-swarm limit. To see the
boundedness needed for this passage, both original and prepared
velocity-squared observables are bounded by $V^2$ and $V_c^2$.
On the event that the root component has at most $K$ vertices,
the velocity output is its actual finite-component continuous
formula. The existing rooted consistency passes that bounded
observable to the limit. The probability of a larger root component
tends to zero as $K\to\infty$ because the declared component is
finite almost surely. Passing the finite uniform-root averaged
inequality then gives the population energy inequality.
No pointwise population speed smaller than $V_c$ is asserted.
:::

:::{prf:lemma} A precap velocity-energy inequality retaining the noisy second graph
:label: lem-rvb-precap-energy

For a random entering array, let
$r^2=\mathbb E N^{-1}\sum_i|v_i|^2$.
Define
$$
z_v=c-tb>0,\qquad z_r=-t(c+a_x),\qquad
F_{\rm box}=|z_r|X_2,\qquad
B_{\rm vel}=m^2q^2d+\frac{2t^3\nu}{(2-t\nu)e}.
$$
Then the actual proposed full update satisfies
$$
\mathbb E\frac1N\sum_i|z_i|^2
\le(z_vr+F_{\rm box})^2+B_{\rm vel}.
\tag{RVB.3}
$$
The same inequality holds for the actual own-provider population
update, with $r^2=\mu|v|^2$.
:::

:::{prf:proof}
The actual first count Laplacian is symmetric, positive semidefinite,
and has norm at most one. Since $0\le t\nu\le1$,
$A_X=I-t\nu L_X$ contracts the averaged Euclidean second moment
for every realized jitter array. Together with (RVB.1) this gives
$\mathbb E\|U\|_{2,N}^2\le r^2$.

Let $z_0=w-ty$. The exact identities give
$$
z_0=z_vU+z_rX+mq\xi.
$$
The OU array is sampled independently after the preparation and
first count kick. Its cross terms with $U,X$ therefore vanish, so
$$
\mathbb E\|z_0\|_{2,N}^2
\le(z_vr+|z_r|X_2)^2+m^2q^2d.
$$
The actual second count kick has the pointwise energy bound from
{prf:ref}`thm-rcb-finite-dissipation`:
$$
\|z\|_{2,N}^2-\|z_0\|_{2,N}^2
\le-\frac{t\nu(2-t\nu)}2\langle w,L_yw\rangle_N
 +\frac{2t^3\nu}{2-t\nu}\langle y,L_yy\rangle_N.
$$
The first term is nonpositive and
$\langle y,L_yy\rangle_N\le1/e$ for every noisy $y$.
Dropping only that nonpositive term proves (RVB.3).
No kernel/OU independence at this second kick is used.

For the population update, on the actual prepared law $\lambda$
the first count operator is self-adjoint on $L^2(\lambda)$:
its bilinear form is the symmetric pair integral with kernel
$K(X-X')$. Its quadratic form is nonnegative and bounded above
by that of the complete unit kernel, hence its norm is at most one.
The first contraction is therefore valid on that law.
The same reasoning applies to $L_y$ on the actual joint stage law
$\Lambda_2$; its $y$ quadratic form is at most $1/e$ by the same
pair bound. The affine OU identity and independent OU noise hold
on the prepared root law. This proves (RVB.3) for the own-provider
population, without replacing its joint stage law by marginals.
:::

:::{prf:lemma} Concavity of the actual squared cap
:label: lem-rvb-cap-concavity

For $u\ge0$ put
$f_V(u)=u/(1+\sqrt u/V)^2$.
This is increasing and concave, and
$$
|C_V(z)|^2=f_V(|z|^2).
$$
Consequently, averaging over rows and all actual innovations,
$$
r^+:=\left(\mathbb E\frac1N\sum_i|v_i^+|^2\right)^{1/2}
\le T(r):=
\frac{\sqrt{(z_vr+F_{\rm box})^2+B_{\rm vel}}}
 {1+\sqrt{(z_vr+F_{\rm box})^2+B_{\rm vel}}/V}.
\tag{RVB.4}
$$
The same inequality holds for the actual population output.
:::

:::{prf:proof}
For $u>0$, differentiation gives
$f_V'(u)=(1+\sqrt u/V)^{-3}>0$ and
$f_V''(u)=-3(1+\sqrt u/V)^{-4}/(2V\sqrt u)<0$.
The function is continuous at zero, so its concavity extends there.
Its expression is the native cap's squared radius.
Apply Jensen to the joint probability obtained by drawing a uniform
slot and then the actual array innovations, and use (RVB.3).
For the population use its own root probability in the same Jensen
calculation.
:::

(sec-rvb-burn)=
## 3. Uniform velocity burn and stationary thresholds

:::{prf:theorem} Six-step harmonic population velocity burn
:label: thm-rvb-population-burn

Along the actual marked harmonic population recursion in its declared
regime, every initial law with positive alive mass and stored speed
cap $V=2$ obeys
$$
\mu_n|v|^2\le0.55^2\qquad(n\ge6).
\tag{RVB.5}
$$
No initial retained-dead-position moment is required by this estimate.
The bound is preserved at all later updates. Every positive-alive
fixed point of this actual population map, if one exists, satisfies
$\mu|v|^2<0.545^2$.
No fixed-point existence or law convergence is inferred here.
:::

:::{prf:proof}
The source-box survival floor of research record 29 keeps the
population alive mass positive, so the actual updates remain defined.
The preceding lemmas give $r_{n+1}\le T(r_n)$ and $r_0\le2$.
The exact primitives satisfy
$z_v<0.961$, $F_{\rm box}<0.137$ and
$B_{\rm vel}<0.118$.
Indeed $c<0.9608$, $b>0.0392$ and $X_2<3.47$ give
the first two bounds. Also $c>0.96$ gives $q^2<0.0392$;
$m<1$, $2-t\nu>1$ and $e>2$ give
$B_{\rm vel}<0.1176+0.0000024<0.118$.
Thus $T(r)$ is bounded by the increasing rationally parametrized
envelope
$$
\overline T(r)=
\frac{\sqrt{(0.961r+0.137)^2+0.118}}
 {1+\sqrt{(0.961r+0.137)^2+0.118}/2}.
$$
For $a,b_*>0$ with $b_*<2$, the inequality
$\overline T(a)<b_*$ is equivalent to
$$
(0.961a+0.137)^2+0.118
 <\left(\frac{2b_*}{2-b_*}\right)^2.
\tag{RVB.6}
$$
Exact rational substitution verifies (RVB.6) successively for
$$
(a,b_*)=(2,1.022),\ (1.022,.739),\ (.739,.628),\
(.628,.580),\ (.580,.559),\ (.559,.55).
$$
It also verifies $\overline T(.55)<.545<.55$.
This proves the six-step threshold and its preservation.

The actual $T$ is a strict contraction as a scalar function:
its derivative is bounded by $z_v<1$, since the two maps
$r\mapsto\sqrt{(z_vr+F_{\rm box})^2+B_{\rm vel}}$ and
$z\mapsto z/(1+z/V)$ have derivative at most $z_v$ and one.
It has a unique fixed point $r_*$ on $[0,V]$.
Moreover $T(.545)<.545$ by another substitution in (RVB.6),
so $r_*<.545$. At a positive-alive population fixed point,
$r\le T(r)$. Since $T(r)-r$ is strictly decreasing, this forces
$r\le r_*<.545$. This argument supplies a necessary moment bound
for such a fixed point, rather than its existence.
:::

:::{prf:theorem} Six-step velocity burn under actual finite current survival
:label: thm-rvb-finite-survival-burn

Let $\tau_\dagger$ be the first completed state with no alive
slot. Use the actual killed finite chain, without a restart.
Let $\eta_n=\operatorname{Law}(S_n\mid\tau_\dagger>n)$ and
$$
r_n^2=\mathbb E_{\eta_n}\frac1N\sum_i|v_i|^2.
$$
Use the explicit source-box extinction bound
$e_N=(1-a_{\rm ret})^N$ of
{prf:ref}`thm-dsa-default-box-alive-floor`. For every $N$ with
$e_N\le.01$,
$$
r_n\le.56\qquad(n\ge6).
\tag{RVB.7}
$$
This is an average over all retained slots under current whole-swarm
survival. Every quasi-stationary law of this actual killed chain,
if one exists, obeys the same bound.
Neither existence nor attraction of a quasi-stationary law is asserted.
:::

:::{prf:proof}
From its actual current law $\eta_n$, the unconditioned proposed
update obeys (RVB.4). The conditional probability of survival through
the next completed update is at least $1-e_N$, uniformly over
the entering arrays. Hence the exact survival normalization gives
$$
r_{n+1}\le\frac{T(r_n)}{\sqrt{1-e_N}}
\le1.01\,\overline T(r_n).
$$
The final inequality follows from
$1/\sqrt{.99}<1.01$. This division is part of the actual conditional
law; it does not condition each intermediate innovation separately.
The rational square certificate (RVB.6), with the right-hand
threshold replaced by $b_*/1.01$, verifies the six transitions
$$
2\longmapsto1.032\longmapsto.750\longmapsto.639
\longmapsto.591\longmapsto.570\longmapsto.560.
$$
It also verifies $1.01\,\overline T(.56)<.56$.
This proves (RVB.7) and its preservation.
The explicit population-size threshold is
$$
N\ge\max\left\{1,
\left\lceil\frac{\log(.01)}{\log(1-a_{\rm ret})}\right\rceil\right\}.
$$
At a quasi-stationary law the same conditional recursion repeats
with the same initial law; iterating its six bounds yields the
necessary stationary moment inequality. No existence result is used.
The population and finite thresholds concern averaged moments,
rather than a reduced individual cap or an alive-row conditional
moment. Alive-row normalization still divides by the actual alive
fraction.
:::

:::{prf:corollary} Smaller actual joint OU providers after the velocity burn
:label: cor-rvb-joint-provider-moment

For the actual own-provider population after burn, its joint
second-stage law obeys
$$
\Lambda_2|w|
\le(\Lambda_2|w|^2)^{1/2}
\le\sqrt{c^2(.55+tX_2)^2+q^2d}<.69.
\tag{RVB.8}
$$
For an actual finite update drawn from the current-survival input
law in (RVB.7), the unconditioned next-stage average satisfies
$$
\left(\mathbb E\frac1N\sum_i|w_i|^2\right)^{1/2}<.70.
\tag{RVB.9}
$$
The latter is not a claim about each empirical provider conditioned
on its future survival or about a realized rowwise maximum.
:::

:::{prf:proof}
The first count/true collision energy inequality gives
$\|U\|_{L^2}\le r$. The exact OU identity and its independent
innovation give
$\|w\|_{L^2}^2=c^2\|U-tX\|_{L^2}^2+q^2d
\le c^2(r+tX_2)^2+q^2d$.
For the population use $r\le.55$ and for the finite current
input use $r\le.56$.
The rational upper bounds $c<.9608$, $X_2<3.47$ and
$q^2<.0392$ verify
$$
.9608^2(.55+.02(3.47))^2+.1176<.69^2,
$$
$$
.9608^2(.56+.02(3.47))^2+.1176<.70^2.
$$
This retains the OU covariance rather than replacing its norm by
a sum of absolute first moments.
:::

(sec-rvb-uncut-radial)=
## 4. Uncut averaged radial cap sensitivity

:::{prf:definition} Common root and two actual deterministic providers
:label: def-rvb-common-root

Let a common prepared root law $\zeta$ be produced by an actual
source-box preparation. Its sources lie in $D$, its own recipient
Gaussian jitter has variance $\sigma_J^2I_d$, and its prepared
speed is bounded by $V_c$. It may be the frozen preparation under
a different entering law; no own collision-energy contraction for
this common root law is assumed.

Compare two deterministic count first providers supported in speed
$V_c$ and two actual joint second-stage laws
$\Lambda_2,\widetilde\Lambda_2$, each with absolute velocity
moment at most $M_w=.70$. Their fields are
$(a_0,M_0)$, $(\widetilde a_0,\widetilde M_0)$ and
$(a_2,M_2)$, $(\widetilde a_2,\widetilde M_2)$ as in research
record 33. Convex provider interpolations retain these bounds.
The root kernels use independent own OU/final innovations, shared
between the compared kernels. All rooted measurements and
conditional-alive normalizers remain inside their actual prepared
laws and providers.

For the actual root's first drift define
$$
X_{1,2}=mX_2+tV_c,\quad A=m-t\nu,\quad
\alpha=t^2\nu\ell,\quad
C_2=tX_{1,2}+t\nu M_w,\quad
\overline S_w=(V/4+C_2)/A.
\tag{RVB.10}
$$
For every first-provider interpolation,
$\|x_1\|_{L^2}\le X_{1,2}<3.55$ because its $U$ is a
convex average bounded by $V_c$. Thus $C_2<.076$ and
$\overline S_w<.58$. These are full-jitter moment bounds.
:::

:::{prf:lemma} Full-jitter radial provider sensitivity
:label: lem-rvb-uncut-provider-cap

For the actual second-kick/cap map
$$
T_\Lambda(x_1,w)
=C_V\big([m-t\nu a_2(x_1+tw)]w
              -tx_1+t\nu M_2(x_1+tw)\big),
$$
compare the two providers while holding the actual root
$(x_1,w)$ law fixed. Then
$$
\left(\mathbb E|T_{\Lambda_2}(x_1,w)
                    -T_{\widetilde\Lambda_2}(x_1,w)|^2\right)^{1/2}
\le t\nu[
 \|\Delta M_2\|_\infty+\overline S_w\|\Delta a_2\|_\infty].
\tag{RVB.11}
$$
No good-jitter event is used and no additive tail floor occurs.
The coefficient is less than
$.006[\|\Delta M_2\|_\infty+.58\|\Delta a_2\|_\infty]$
at the declared reference parameters.
:::

:::{prf:proof}
At every provider interpolation write its actual pre-cap velocity
as $Z=l(y)w+B(y)$, where $l=m-t\nu a_2(y)\ge A$ and
$|B|\le t|x_1|+t\nu M_w$.
The native radial Jacobian identity of
{prf:ref}`lem-rcap-radial-self-coefficient` gives pointwise
$$
\|DC_V(Z)w\|
\le\frac{V/4+t|x_1|+t\nu M_w}{A}.
$$
Consequently its $L^2$ norm is at most $\overline S_w$ by
Minkowski and $\|x_1\|_2\le X_{1,2}$.
The variation of the pre-cap velocity under provider interpolation
is $t\nu[\Delta M_2(y)-w\Delta a_2(y)]$.
The first cap multiplier has norm at most one, while the second
has exactly the radial bound just proved. Minkowski in the root
probability and then over the interpolation parameter gives
(RVB.11). The providers are deterministic population fields;
their differences are deterministic supremum norms. No independence
of $x_1,w$ or $y=x_1+tw$ has been imposed.
:::

:::{prf:lemma} Uncut averaged second-kick Jacobians
:label: lem-rvb-uncut-cap-jacobians

On the same actual root laws, uniformly over their first and second
provider interpolations,
$$
\|D_wT_\Lambda\|_{L^2}
\le\overline K_w:=m+\alpha(M_w+\overline S_w)<.9997,
$$
$$
\|D_{x_1}T_\Lambda\|_{L^2}
\le\overline K_x:=t+t\nu\ell(M_w+\overline S_w)<.0247.
\tag{RVB.12}
$$
For arbitrary random tangent perturbations $h_w,h_x$ this yields
only the pointwise differential bound followed by
$$
\|DT_\Lambda(x_1,w)[h_x,h_w]\|_{L^2}
\le\left\|J_w(x_1)|h_w|+J_x(x_1)|h_x|\right\|_{L^2},
\tag{RVB.13}
$$
with
$J_w=m+\alpha[M_w+(V/4+t|x_1|+t\nu M_w)/A]$
and the analogous $J_x$.
For a finite perturbation the same differential estimate must be
integrated along its actual phase segment, with $J_w,J_x$
evaluated on that segment. Neither form justifies replacing the
right side by
$\overline K_w\|h_w\|_2+\overline K_x\|h_x\|_2$
when those random perturbations are correlated with $x_1$.
For uniformly bounded perturbations it does give
$\overline K_w\|h_w\|_\infty+
\overline K_x\|h_x\|_\infty$.
:::

:::{prf:proof}
Differentiate the actual map, keeping $y=x_1+tw$:
$$
D_wZ=(m-t\nu a_2)I+
 t^2\nu[DM_2-w\otimes Da_2],\quad
D_{x_1}Z=-tI+t\nu[DM_2-w\otimes Da_2].
$$
The field bounds are $\|DM_2\|\le\ell M_w$ and
$|Da_2|\le\ell$.
After multiplication by the native cap Jacobian, the constant
terms are bounded by $m$ and $t$, the $DM_2$ terms by
$\alpha M_w$ and $t\nu\ell M_w$, and the $w\otimes Da_2$
terms by the pointwise radial coefficient from the previous proof.
Taking its $L^2$ norm gives (RVB.12).
The numerical inequalities follow from
$\alpha<.00007284$, $M_w=.70$ and
$\overline S_w<.58$:
$$
.9996+.00007284(1.28)<.9997,\qquad
.02+.003642(1.28)<.0247.
$$
The derivative estimates give the differential bound (RVB.13).
For a finite phase perturbation integrate that bound along its
segment, retaining the Jacobian bounds at each interpolated phase.
The final distinction follows because an $L^2$ bound on a
Jacobian alone does not control its product with an arbitrarily
correlated $L^2$ displacement. With bounded displacements their
supremum norms can be pulled outside the expectations.
:::

(sec-rvb-own-feedback)=
## 5. Complete direct own-provider feedback with the root law fixed

:::{prf:theorem} Global uncut kinetic feedback from both actual own providers
:label: thm-rvb-global-own-provider-feedback

Retain the common source-box root law of
{prf:ref}`def-rvb-common-root` and let its two kinetic kernels
use their respective first and joint second providers. Put
$$
d_0=\|\Delta M_0\|_\infty+V_c\|\Delta a_0\|_\infty,\qquad
d_2=\|\Delta M_2\|_\infty+\overline S_w\|\Delta a_2\|_\infty,
$$
$$
C_0=\sqrt{(bt\nu)^2+
        (ct\nu\overline K_w+t^2\nu\overline K_x)^2}<.00578.
$$
There is a valid coupling of their full Gaussian kinetic kernels
such that
$$
\left(\mathbb E[
 |\Delta x^+|^2+|\Delta v^+|^2]\right)^{1/2}
\le C_0d_0+t\nu d_2.
\tag{RVB.14}
$$
Each kernel retains both of its own providers. The root law is held
fixed here; its preparation-law difference is not included in this
estimate.
:::

:::{prf:proof}
First hold the second provider fixed and interpolate the first
provider. The difference of its actual count field at the prepared
root is
$\Delta C_0(X,v)=\Delta M_0(X)-v\Delta a_0(X)$, whose norm is
at most $d_0$, because $|v|\le V_c$.
Along that interpolation the actual first drift, OU mean and
final position mean have derivatives
$$
\dot x_1=t^2\nu\Delta C_0,\qquad
\dot{\overline w}=ct\nu\Delta C_0,\qquad
\dot y=bt\nu\Delta C_0.
$$
The first count average remains convex and bounded by $V_c$.
The full root first-drift second moment is therefore uniformly
bounded by $X_{1,2}$ along this interpolation. Its OU innovation
is independent and shared; it is never conditioned on $y$.
The bounded-displacement conclusion of (RVB.12) gives velocity
RMS difference at most
$(ct\nu\overline K_w+t^2\nu\overline K_x)d_0$.
The shared final position noise cancels, leaving positional
difference at most $bt\nu d_0$.
Combining the two components gives $C_0d_0$.

Then change the actual second provider while keeping the chosen
actual first provider and root law fixed. This leaves position
unchanged, and (RVB.11) bounds velocity RMS difference by
$t\nu d_2$. Minkowski gives (RVB.14).
The two common Gaussian arrays have independent standard entries
within each own marginal law, so this is an actual kinetic coupling.
The bound $C_0<.00578$ follows from
$b<.039216$, $c<.9608$, $\overline K_w<.9997$ and
$\overline K_x<.0247$, by exact rational squaring.
:::

:::{prf:corollary} Exact terminal marking and actual alive normalization
:label: cor-rvb-marked-alive-feedback

In the coupling of (RVB.14), with the actual terminal mark
$a^+=1_D(x^+)$,
$$
\Pr(a^+\ne\widetilde a^+)
\le C_{\rm mark}d_0,\qquad
C_{\rm mark}=\frac{2\sqrt d}{s\sqrt{2\pi}}bt\nu.
\tag{RVB.15}
$$
Let $d_{\rm ph}(z,z')=\min\{1,|z-z'|\}$.
The two output laws have actual alive masses at least
$a_{\rm ret}>0$ from the source-box safe-return lemma.
Writing their own normalized alive phase laws as
$\lambda_A,\widetilde\lambda_A$, one has
$$
W_{d_{\rm ph}}(\lambda_A,\widetilde\lambda_A)
\le\frac2{a_{\rm ret}}
 [(C_0+C_{\rm mark})d_0+t\nu d_2].
\tag{RVB.16}
$$
The masses in these two alive normalizations are their actual output
masses; they are not set equal.
:::

:::{prf:proof}
The second kick does not change position. The first-provider
position shift is bounded by $bt\nu d_0$. Conditional on the
actual preparation and OU innovations, the independent final
position Gaussian has variance $s^2I_d$.
For each coordinate a box-mark change requires that coordinate to
cross one of the two faces; the two intervals have total length
at most twice that coordinate's mean displacement.
Its one-dimensional Gaussian density is at most
$1/(s\sqrt{2\pi})$. Sum over the faces and use
$|\Delta y|_1\le\sqrt d|\Delta y|$ to get (RVB.15).
Both frozen first-provider kernels retain the pathwise $V_c$
bound and the same source-box Gaussian form. Hence the actual
safe-return event of research record 29 applies and gives their
positive alive masses.

For any coupling of two marked probabilities, match its common-alive
part after dividing by the larger alive mass. The mass left over in
either normalized alive law is at most the mark-disagreement
probability divided by the smaller mass. Since $d_{\rm ph}\le1$,
couple those residual masses arbitrarily. This gives the
conservative estimate
$W_{d_{\rm ph}}(\lambda_A,\widetilde\lambda_A)
\le2[\mathbb E d_{\rm ph}(z,z')+\Pr(a\ne a')]/a_{\rm ret}$.
Use Cauchy--Schwarz and (RVB.14)--(RVB.15) to prove
(RVB.16). This is the actual normalization comparison, rather
than a statement about current whole-swarm survival conditioning.
:::

(sec-rvb-residual)=
## 6. The remaining signed and marked block accounts

:::{prf:remark} Precise global progress and surviving obligations
:label: rem-rvb-own-block-residual

The actual harmonic active/revival velocity-moment calculation is
closed, including both count kicks, all uncut innovations and finite
current-survival normalization. It lowers the actual population
second-provider absolute velocity envelope after burn from the
pointwise $V_c$ scale to $M_w<.69$.
The global uncut radial sensitivity (RVB.11)--(RVB.16) closes the
good-jitter/bad-jitter split for direct fixed-root provider changes.
No positive additive tail floor is introduced.

These statements still do not prove the full default own-provider
law block. Three distinct terms survive:

1. Changing an entering population changes its actual rooted
   preparation law, not only its two kinetic providers. The current
   measured reward/diversity normalizers, mandatory dead-root donor
   choice and component covariance must be retained in that term.
2. A globally averaged Jacobian below one does not certify phase
   transport for arbitrary coupled displacements correlated with
   the root position. Its signed cross terms and the graph's
   negative alignment form still need a complete discrete law
   estimate, such as the cap-aware balance in research record 33.
3. A positive alive-mass floor does not imply a small-dead
   invariant class. Mandatory-revival feedback remains independent
   of small positive fitness exponents unless a sharper default
   dead-tail bound or a delayed block estimate absorbs it.

The velocity thresholds are averages of all original slots.
They do not reduce the individual speed cap, nor do they become
alive-normalized thresholds without division by the actual alive
mass. The finite estimates are bounds for the actual killed chain's
current-survivor laws; they neither create a restart nor identify a
quasi-stationary limit. The population bound concerns its own
marked recursion, not the finite noisy empirical-provider kernel.

Result kind: complete moment closure and complete uncut direct
provider-feedback interface, with an explicit remaining signed and
marked own-law block. No impossibility theorem or full default
convergence theorem is claimed. The force is harmonic throughout;
the Rastrigin arm retains its additional force-account obligation.
:::
