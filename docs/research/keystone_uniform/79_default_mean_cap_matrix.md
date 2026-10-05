# Source-dependent conditional mean of the actual noisy native cap

(sec-mcm79-register)=
## 1. Actual count carrier and the conditional information

:::{prf:definition} Mean-cap matrix register
:label: def-mcm79-register

Retain the actual harmonic count kinetic map of research67,68,74 and76:

$$
d=3,\quad t=.02,\quad a=.006,\quad c=e^{-.04},\quad
m=1-t^2,\quad b=t(1+c),\quad a_x=1-tb,
$$

$$
q^2=(1-c^2)/2,\quad z_v=c-tb,\quad r_H=mb,
\quad V=2,\quad V_c=4.
\tag{MCM79.1}
$$

For an actual prepared root and its actual first count field use

$$
U=(I-aL_X)P,\quad x_1=mX+tU,\quad
w=c(U-tX)+q\xi,\quad y=x_1+tw,
$$

$$
z=w-ty-aL_yw,\qquad D=DC_V(z),\qquad
C_V(z)=\frac{Vz}{V+|z|}.
\tag{MCM79.2}
$$

The second graph and its velocity numerator use the same actual joint
$(y,w)$ law or actual finite array. The new original OU rows are independent
standard Gaussian rows before this update. They are not conditioned on an
empirical provider event, terminal marking or future survival.

For a population fix its prepared law and assume

$$
|P|\le4,\qquad \int |w'|\,d\Lambda(y',w')\le .70.
\tag{MCM79.3}
$$

For a finite array condition on the complete prepared array and assume

$$
|P_i|\le4,\quad
\langle|P|^2\rangle_N\le .56^2,\quad
\langle|X|^2\rangle_N\le12.25.
\tag{MCM79.4}
$$

Write $\mathscr P$ for the full preparation before the new OU array. The
prepared root position, first count output and physical test vector belong
to $\mathscr P$. Define the actual conditional matrices

$$
A=\mathbb E_\xi[D\mid\mathscr P],\qquad
B=\mathbb E_\xi[D^2\mid\mathscr P].
\tag{MCM79.5}
$$

For finite arrays these are row matrices, each using its own graph under
the complete original Gaussian array. They are not derivatives evaluated
at an averaged velocity or graph.
:::

(sec-mcm79-lower)=
## 2. An exact source-dependent lower matrix

:::{prf:lemma} The root's exact affine OU representation
:label: lem-mcm79-affine

Put

$$
\mu=z_vU-r_HX,\qquad
\eta=\frac{ct}{r_H}=\frac{c}{c+a_x},\qquad g_3=\mathbb E|\xi|.
$$

Then the actual root satisfies

$$
w=\eta(U+\mu)+q\xi,\qquad
z_0:=w-ty=\mu+mq\xi.
\tag{MCM79.6}
$$

Under (MCM79.3), or for a finite array under (MCM79.4),

$$
\mathbb E_\xi[|z|\mid\mathscr P]
\le (1+a\eta)|\mu|+a\eta|U|+(m+a)qg_3+a(.70)
<1.003|\mu|+.331.
\tag{MCM79.7}
$$

All large Gaussian outcomes and all unbounded root positions are retained.
:::

:::{prf:proof}

Since $a_x+tb=1$ and $r_H=t(c+a_x)$,
$r_H-tz_v=t$. Substituting $X=(z_vU-\mu)/r_H$ into
$w=c(U-tX)+q\xi$ therefore gives
$w=(ct/r_H)(U+\mu)+q\xi$ exactly. Direct substitution into
$w-ty=mw-tx_1$ gives the second identity.

The actual count formula has degree at most one. In the population case,

$$
|L_yw|\le |w|+\int |w'|\,d\Lambda(y',w')\le |w|+.70.
$$

In a finite array the exact bound is instead
$|(L_yw)_i|\le |w_i|+\langle|w|\rangle_N$. The self term cancels;
including it in these nonnegative upper sums does not change a denominator.
Moreover first count contraction and the original OU centering give

$$
\mathbb E_\xi\langle|w|^2\rangle_N
=c^2\langle|U-tX|^2\rangle_N+3q^2
\le c^2(.56+.02\sqrt{12.25})^2+3q^2<.70^2.
$$

Thus $\mathbb E_\xi\langle|w|\rangle_N<.70$ by Cauchy--Schwarz.
This averages the entire original OU array; it does not restrict the array
to a provider-good event. Its graph remains correlated with every root row.

Use $z=z_0-aL_yw$, (MCM79.6), and the two actual environment bounds.
The triangle inequality before expectation yields the non-strict estimate
in (MCM79.7). Every first count output is a convex combination of prepared
velocities, so $|U|\le4$.

For completeness, elementary Gaussian radial integration gives
$g_3=2\sqrt{2/\pi}<1.596$. The latter decimal bound is certified in
Section 6 together with $q<.196066$, $\eta<.491$, and
$1+a\eta<1.003$. These rational bounds give

$$
a\eta V_c+(m+a)qg_3+a(.70)
<(.006)(.491)(4)+(1.0056)(.196066)(1.596)+.0042
<.331.
$$

This proves the strict scalar coefficient bound, including $\mu=0$.
:::

:::{prf:theorem} Actual conditional mean-cap lower matrix and joint sector
:label: thm-mcm79-lower

Under {prf:ref}`def-mcm79-register`, put

$$
\ell_c(\mu)=\left[\frac2{2+1.003|\mu|+.331}\right]^2.
$$

The actual conditional matrices satisfy

$$
\ell_c(\mu)I\preceq A,\qquad
A^2\preceq B\preceq A\preceq I,\qquad
B\preceq\frac{159}{200}I.
\tag{MCM79.8}
$$

In particular $\|A\|_{\rm op}\le\sqrt{159/200}$. For each
pre-OU-fixed vector $T$ the full actual law obeys

$$
\mathbb E_\xi\langle T,DT\rangle\ge\ell_c(\mu)|T|^2,
\qquad
|AT|^2\le\mathbb E_\xi|DT|^2\le\frac{159}{200}|T|^2.
\tag{MCM79.9}
$$

The lower coefficient depends on the actual unbounded source and root;
the matrices retain their orientation. No cap/count commutation is used.
:::

:::{prf:proof}

The native cap Jacobian is self-adjoint. Its tangential and radial
eigenvalues are $2/(2+|z|)$ and $[2/(2+|z|)]^2$. Thus, pointwise,

$$
\left[\frac2{2+|z|}\right]^2I\preceq D,\qquad
0\preceq D^2\preceq D\preceq I.
$$

The scalar function $f(s)=[2/(2+s)]^2$ is decreasing and convex on
$[0,\infty)$: $f'(s)=-8/(2+s)^3$ and
$f''(s)=24/(2+s)^4$. Conditional Jensen and (MCM79.7) give

$$
A\succeq\mathbb E_\xi f(|z|)I
\succeq f(\mathbb E_\xi|z|)I\succeq\ell_c(\mu)I.
$$

Integrating $D^2\preceq D$ gives $B\preceq A$. Conditional variance
gives $|AT|^2\le\mathbb E|DT|^2=\langle T,BT\rangle$ for every
fixed $T$, hence $A^2\preceq B$. The exact twelve-threshold full-Gaussian
calculation of {prf:ref}`thm-gca74-cap-deficit` gives
$B\preceq159I/200$ under precisely the hypotheses declared here.
It retains the finite empirical-provider exceptional event as its actual
unconditional charge; it never treats a restricted noise as Gaussian.
The stated norm and test-vector conclusions follow. None of these
Loewner inequalities asserts that $A$ and $B$ commute.
:::

(sec-mcm79-source)=
## 3. A lower quadratic through every original recipient-jitter outcome

:::{prf:theorem} Source-weighted population mean-cap sector
:label: thm-mcm79-source-sector

Use the actual population source-plan comparison of
{prf:ref}`def-nca76-register`, including its own joint OU provider:

$$
X=S+IJ,\qquad r=\delta S+\delta I J,\qquad
J\sim N(0,.01I_3),\quad S\in[-2,2]^3,\quad0\le I\le1.
$$

The complete coupled source/component/Haar plans, $P$ and $p$ are fixed
before this original jitter. Assume $|P|\le4$ and $\|P\|_2\le.55$.
These supply the deterministic actual OU provider bound (MCM79.3).

Let $T=T_0+\tau J$, where $T_0\in\mathbb R^3$ and $\tau\in\mathbb R$
are fixed by those plans, and suppose $T$ is square-integrable. Then

$$
\mathbb E\langle T,DT\rangle
\ge k_r\mathbb E|T|^2,\qquad
k_r:=\left[\frac2{2+1.003(4z_v+r_H\sqrt{20.05})+.331}\right]^2
>.0989.
\tag{MCM79.10}
$$

For a vector $p$ fixed before jitter the sharper conclusion is

$$
\mathbb E\langle p,Dp\rangle\ge k_p\mathbb E|p|^2,\qquad
k_p:=\left[\frac2{2+1.003(4z_v+r_H\sqrt{12.03})+.331}\right]^2
>.1001.
\tag{MCM79.11}
$$

In particular, for every real $\alpha,\gamma$ the vector
$T=\alpha r+\gamma p$ satisfies (MCM79.10). Consequently the exact
two-vector mean-cap Gram matrix obeys

$$
\mathbb E
\begin{pmatrix}
\langle r,Dr\rangle&\langle r,Dp\rangle\\
\langle p,Dr\rangle&\langle p,Dp\rangle
\end{pmatrix}
\succeq .0989\,
\mathbb E
\begin{pmatrix}
|r|^2&\langle r,p\rangle\\
\langle p,r\rangle&|p|^2
\end{pmatrix}.
\tag{MCM79.12}
$$

This is an oriented quadratic inequality with its actual correlations.
It is not the substitution $A=.0989I$ in a signed cross term.
:::

:::{prf:proof}

Fix the complete coupled plans before $J$. Centering and the exact
Gaussian fourth moments give

$$
\begin{split}
\mathbb E_J[|X|^2|T|^2]={}&|S|^2|T_0|^2
 +.01[I^2(3)|T_0|^2+\tau^2(3)|S|^2+4I\tau S\cdot T_0]\\
&+.0001 I^2\tau^2(3)(5).
\end{split}
$$

This is the same complete source product as (RFK.12), with $T_0$ in
place of $\delta S$ and $\tau$ in place of $\delta I$; its homogeneous
proof does not require a bound on either replacement. Since
$4|I\tau S\cdot T_0|\le2|T_0|^2+2|S|^2\tau^2$, it follows that

$$
\mathbb E_J[|X|^2|T|^2]\le12.05|T_0|^2+.6015\tau^2
\le20.05\mathbb E_J|T|^2.
\tag{MCM79.13}
$$

All original jitter outcomes are integrated. For a pre-jitter-fixed $p$,
the corresponding product is at most $12.03\mathbb E_J|p|^2$.
Outer integration preserves these inequalities even when the coupled
plans correlate $S,T_0,\tau,P,p$ arbitrarily.

If $\mathbb E|T|^2=0$ the assertion is immediate. Otherwise use the
probability measure with density $|T|^2/\mathbb E|T|^2$ on the complete
pre-OU preparation. Denote its expectation by $\mathbb E_T$. The actual
conditional matrix bound (MCM79.9) and convexity of $f$ yield

$$
\frac{\mathbb E\langle T,DT\rangle}{\mathbb E|T|^2}
\ge\mathbb E_T f(1.003|\mu|+.331)
\ge f(1.003\mathbb E_T|\mu|+.331).
$$

The exact root $\mu=z_vU-r_HX$ and pointwise $|U|\le4$ give

$$
\mathbb E_T|\mu|\le4z_v+r_H\mathbb E_T|X|
\le4z_v+r_H\sqrt{\mathbb E_T|X|^2}
\le4z_v+r_H\sqrt{20.05}.
$$

This is Cauchy--Schwarz under the actual displacement-weighted measure,
not the factorization of a source or velocity moment from $|T|^2$.
Monotonicity proves (MCM79.10). The pre-jitter-fixed $p$ proof uses
$12.03$ instead. Section 6 gives exact rational certificates for both
strict scalar endpoints. Finally $\alpha r+\gamma p$ has precisely
the scalar-affine jitter form just proved. Testing every pair
$(\alpha,\gamma)$ proves (MCM79.12), including zero displacements.
:::

:::{prf:remark} Actual finite source products remain a separate interface
:label: rem-mcm79-finite-source

(MCM79.8)--(MCM79.9) are exact finite row statements under the entire
fixed prepared array budget (MCM79.4). The population weighted constants
in (MCM79.10)--(MCM79.12) have not been assigned to finite arrays by
conditioning on a random prepared-budget event: that event can depend
on recipient jitters and change their Gaussian law. A finite extension
must retain its actual mixed empirical-provider and displacement moments,
or charge its removed weighted moment. No population deterministic
provider has been inserted into such a random product here.
:::

(sec-mcm79-no-uniform)=
## 4. Why a positive conditional lower constant cannot be uniform in the root

:::{prf:proposition} Exact obstruction to root-uniform conditional ellipticity
:label: prop-mcm79-no-uniform-root

There is no $\varepsilon>0$ such that $A\succeq\varepsilon I$
for every population preparation allowed by (MCM79.3), including the
actual source-box Gaussian preparations of Section 3. There is also no
such constant uniform in $N$ for all fixed finite preparations satisfying
(MCM79.4). This statement concerns a conditional matrix estimate, not
delayed alive-law mixing or contraction of optimal transport.
:::

:::{prf:proof}

For a population take an entering marked law with half its mass alive
at $(0,0)$ and half dead at $(3e_1,0)$. Equal alive fitness makes the active
gate zero. Mandatory revival copies the alive zero source for the dead
half. All original component velocities are zero, so their actual Haar
readout is zero as well. Its actual prepared law therefore has $P=0$,
$S=0$ and $I$ Bernoulli with parameter $1/2$, independent of the new
recipient jitter. Thus $X=IJ$ and $U=0$. Its actual joint OU provider has

$$
\mathbb E|w|^2=c^2t^2(.015)+3q^2<.70^2,
$$

and hence satisfies (MCM79.3). Conditional on the copied prepared root
$I=1$, $X=ne_1$, retain this fixed actual provider and the original fresh root
Gaussian. Since its degree is at most one and its numerator is bounded
by $.70$,

$$
|z|\ge |z_0|-a|w|-a(.70)
\ge(r_H-act)n-(m+a)q|\xi|-a(.70).
\tag{MCM79.14}
$$

The coefficient $r_H-act$ is positive. Thus $|z|\to\infty$ for
every finite $\xi$ as $n\to\infty$, while
$\|D\|_{\rm op}=2/(2+|z|)\le1$. Dominated convergence gives
$\|A\|_{\rm op}\le\mathbb E\|D\|_{\rm op}\to0$.
Conditional root values in every neighborhood of these points have
positive probability under the original jitter. The same lower bound
holds uniformly over such neighborhoods, so the failure cannot be
removed by choosing an almost-everywhere version of the conditional law.

For finite arrays let $P_i=0$, $X_1=3.4\sqrt N e_1$, and $X_i=0$
for $i>1$. Then (MCM79.4) holds, with positional moment $11.56<12.25$.
These preparations lie in the support of the actual mandatory-revival
preparation with one entering dead slot at $(3e_1,0)$ and $N-1$ alive
slots at $(0,0)$: every velocity is zero, the active gate is zero, and
the single revived position is its original Gaussian recipient jitter.
Couple the original independent
OU rows for all $N$ on one infinite Gaussian product space. The first
row obeys

$$
|z_1|\ge (r_H-act)3.4\sqrt N
 -(m+a)q|\xi_1|-a\langle|w|\rangle_N.
$$

Its actual empirical velocity mean is bounded by
$ct(3.4)/\sqrt N+q\langle|\xi|\rangle_N$. The Gaussian average
converges almost surely to the finite first moment $g_3$; this last
statement follows directly, for example, from the usual independent
finite-variance average law, or from its elementary fourth-moment proof.
That proof gives
$\mathbb E|(N^{-1}\sum_i(|\xi_i|-g_3))|^4\le C/N^2$ by expanding
the centered fourth power, so Markov and summability imply convergence
almost surely. Consequently $|z_1|\to\infty$ almost surely.
Dominated convergence again gives $\|A_1\|_{\rm op}\to0$.
Every actual finite graph and every original Gaussian row was retained.
:::

(sec-mcm79-survival)=
## 5. Each own restriction preserves the weighted missing moment

:::{prf:proposition} Exact lower-sector transfer to an own event
:label: prop-mcm79-own-restriction

Let $T$ be fixed before the original OU, and let $\mathcal S$ be any
own output event of probability $p_{\mathcal S}>0$. Suppose a raw
source class supplies $\mathbb E\langle T,DT\rangle\ge k\mathbb E|T|^2$.
Then

$$
\mathbb E[\langle T,DT\rangle\mid\mathcal S]
\ge k\mathbb E[|T|^2\mid\mathcal S]
 -\frac{(1-k)\mathbb E[|T|^2\mathbf1_{\mathcal S^c}]}{p_{\mathcal S}}.
\tag{MCM79.15}
$$

If the raw lower estimate is known only on a pre-OU prepared event
$\mathcal G$, its right side requires the additional negative charge
$k\mathbb E[|T|^2\mathbf1_{\mathcal G^c}]/p_{\mathcal S}$.
Without a uniform $k$, the general conditional form is

$$
\mathbb E[\langle T,DT\rangle\mid\mathcal S]
\ge\frac{\mathbb E[\ell_c(\mu)|T|^2\mathbf1_{\mathcal G}]
 -\mathbb E[|T|^2\mathbf1_{\mathcal S^c}]}{p_{\mathcal S}}.
\tag{MCM79.16}
$$

The event can be that swarm's next nonextinction or a population root's
terminal-alive event. A second swarm has its own numerator and denominator.
:::

:::{prf:proof}

Pointwise $0\le\langle T,DT\rangle\le|T|^2$. Hence its own restricted
numerator is at least its raw expectation minus
$\mathbb E[|T|^2\mathbf1_{\mathcal S^c}]$. Insert the assumed raw
lower estimate, split $\mathbb E|T|^2$ over the two own events, and
divide by the actual $p_{\mathcal S}$. This proves (MCM79.15).
For a prepared-budget event integrate the conditional lower bound only
on $\mathcal G$ and use nonnegativity on its complement. The same
calculation gives the additional charge, or (MCM79.16) directly.
An extinction probability alone does not factor from its displacement
weight. No restricted Gaussian law has been asserted fresh.
:::

:::{prf:remark} Exact signed phase interface and present scope
:label: rem-mcm79-phase-interface

For fixed pre-OU $R,Z_b$ the actual bare capped phase expression is

$$
\mathbb E_\xi Q_\beta(R,DZ_b)
=|R|^2+\langle Z_b,BZ_b\rangle
 +2\beta\langle R,AZ_b\rangle.
\tag{MCM79.17}
$$

Equation (MCM79.8) now supplies a source-dependent lower $A$ together
with the joint sector $A^2\preceq B\preceq A$ and the strict upper
$B\preceq159I/200$. The full-jitter source sector also gives an oriented
two-vector lower Gram matrix. These are completed estimates for the actual
own graph, not a claim that its matrix is scalar. If $R$ and $Z_b$ have
opposite or mixed directions, a lower bound on $A$ cannot be substituted
into their signed bilinear term without controlling that orientation.

The actual complete velocity differential contains $aDF_2$, with its
graph/cap correlations and first-force contribution retained in
research68,74 and76. Formula (MCM79.17) and a lower cap matrix alone
do not absorb those signed cross terms or compare accepted preparation,
Haar/revival laws and terminal-alive normalization. No general phase
gap, iterated default alive-law rate, or exact finite-swarm QSD rate is
concluded here. The obstruction in Section 4 only excludes a uniform
root-conditional positive lower matrix; it leaves the full weighted
source law and delayed law block as the correct remaining targets.
:::

(sec-mcm79-certificate)=
## 6. Exact rational certificates

The exponential bounds below are alternating Taylor bounds with signed
remainders. The Gaussian radial constant uses the elementary identity
$\pi/4=4\arctan(1/5)-\arctan(1/239)$. If $\theta=\arctan(1/5)$,
tangent doubling gives $\tan(2\theta)=5/12$ and
$\tan(4\theta)=120/119$. Subtracting the angle with tangent $1/239$
then gives tangent one. The resulting angle is positive since
$\theta\ge(1/5)/(1+1/25)=5/26$, and it is less than $.8<1<\pi/2$;
the last bound follows from $\pi/4=\int_0^1(1+x^2)^{-1}dx>1/2$.
Thus it equals $\pi/4$ by the injectivity of tangent on that interval.
The alternating four-term lower series for $\arctan(1/5)$ and the
upper bound $\arctan(1/239)<1/239$ give the certified lower value of
$\pi$. Thus no numerical quadrature enters these comparisons.

```python
from fractions import Fraction as F
from math import factorial

x = F('.04')
c_lo = sum((-x)**k / factorial(k) for k in range(6))
c_hi = sum((-x)**k / factorial(k) for k in range(7))
t, a, m = F('.02'), F('.006'), F('.9996')
ax_hi = 1 - t*t*(1+c_lo)
ax_lo = 1 - t*t*(1+c_hi)
eta_hi = c_hi / (c_hi+ax_lo)
q2_hi = (1-c_lo*c_lo)/2
assert F('.196066')**2 > q2_hi
assert eta_hi < F('.491')
assert 1+a*eta_hi < F('1.003')

atan_lo = sum(F((-1)**k, 2*k+1)*F(1, 5)**(2*k+1) for k in range(4))
pi_lo = 16*atan_lo - F(4, 239)
assert pi_lo > F('3.141')
assert F(8)/pi_lo < F('1.596')**2
constant = a*F('.491')*4+(m+a)*F('.196066')*F('1.596')+a*F('.70')
assert constant < F('.331')
assert c_hi*c_hi*F('.63')**2+3*q2_hi < F('.70')**2

zv_hi = c_hi - t*t*(1+c_lo)
rh_hi = m*t*(1+c_hi)
assert zv_hi < F('.960016')
assert rh_hi < F('.039216')
assert F('4.478')**2 > F('20.05')
assert F('3.47')**2 > F('12.03')

def lower_sector(source_product):
    upper = F('1.003')*(4*F('.960016')+F('.039216')*source_product)+F('.331')
    return (F(2)/(2+upper))**2

assert lower_sector(F('4.478')) > F('.0989')
assert lower_sector(F('3.47')) > F('.1001')
assert m*t*(1+c_lo)-a*c_hi*t > 0
print('All actual mean-cap matrix rational endpoints pass.')
```
