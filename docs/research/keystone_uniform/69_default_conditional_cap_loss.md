# Conditional native-cap coercivity with the full noisy provider

The result below supplements the complete signed identity of research68.
It proves a conditional lower bound for its actual cap loss without
factoring a cap Jacobian from a correlated displacement. It does not close
the signed multi-step source/provider account at the default parameters.

:::{div} feynman-prose

A large physical velocity makes the native cap less sensitive to a velocity
perturbation. The perturbation and the noisy interaction graph may be
correlated, so an average cap derivative cannot be applied to their product.
Instead, complete the actual cap-loss square first. One term still measures
the full correlated velocity perturbation; another depends only on the
position perturbation, which is fixed before the fresh OU kick. We can
average this latter term over every Gaussian outcome. The resulting weight
keeps the prepared position dependence rather than concealing it in an
unproved uniform constant.

:::

(sec-ccl-register)=
## 1. Actual cap and conditional count-stage register

:::{prf:definition} Native-cap loss and pre-OU conditioning
:label: def-ccl-register

Use {prf:ref}`def-dbl68-register`, with $d=3$, $t=.02$,
$c=e^{-.04}$, $b=t(1+c)$, $m=1-t^2$, $a_x=1-tb$,
$a=t\nu=.006$, $q^2=(1-c^2)/2$, $\beta=.04$ and $V=2$.
At any prepared interpolation point keep its actual own first count output
$U=P-aL_1P$, with $|P|,|U|\le V_c=4$. Put
$$
x_1=mX+tU,\quad w=c(U-tX)+q\xi,\quad y=x_1+tw,
\quad z=w-ty-aL_yw,\quad \xi\sim N(0,I_d).
\tag{CCL.1}
$$
The second graph is built from this same $y$ and acts on this same $w$.
The native cap is $C_V(z)=Vz/(V+|z|)$, including its continuous derivative
$DC_V(0)=I$. For the complete physical differential let
$$
R=\dot y=a_xr+b\dot U,\qquad Z=\dot z,
\qquad D_C=DC_V(z),
$$
$$
\mathcal L_C=|(I-D_C)Z+\beta R|^2
          +2\langle(I-D_C)Z,D_CZ\rangle.
\tag{CCL.2}
$$
$Z$ includes both own count-field derivatives, including their actual
spatial forces. No independence of $Z,z$ or either graph is assumed.

For a population law, condition on its prepared root, its prepared
physical differential and its deterministic own prepared provider. For a
finite array, condition on the entire prepared array and its entire
physical differential. Denote this pre-OU information by $\mathscr P$.
It fixes $R$ but not the own second graph or $Z$. Prepared inputs may
already include the full source, component-Haar and recipient-jitter law.
The new OU and final position Gaussians remain independent of
$\mathscr P$; the final position Gaussian has zero shared differential.
:::

(sec-ccl-completion)=
## 2. Complete the actual cap loss before taking expectations

:::{prf:lemma} Exact cap-loss completion and pointwise coercivity
:label: lem-ccl-completion

At every actual outcome,
$$
\begin{split}
\mathcal L_C={}&
\left\langle Z+\beta(I+D_C)^{-1}R,
 (I-D_C^2)[Z+\beta(I+D_C)^{-1}R]\right\rangle\\
&+2\beta^2\langle R,D_C(I+D_C)^{-1}R\rangle.
\end{split}
\tag{CCL.3}
$$
In particular
$$
\mathcal L_C\ge f(|z|)|R|^2,
\qquad f(u)=\frac{2\beta^2}{1+(1+u/V)^2},\quad u\ge0.
\tag{CCL.4}
$$
The full correlated remainder in the first line of (CCL.3) also obeys
$$
\mathcal L_C\ge
\left[1-\left(\frac V{V+|z|}\right)^2\right]
    |Z+\beta(I+D_C)^{-1}R|^2+f(|z|)|R|^2.
\tag{CCL.5}
$$
Thus the actual velocity-force differential has not been omitted from
the loss or replaced by an independent Gaussian marginal.
:::

:::{prf:proof}
The radial and tangential eigenvalues of the native derivative are
$$
d_r=\left(\frac V{V+|z|}\right)^2,
\qquad d_t=\frac V{V+|z|},
$$
with both equal to one at $z=0$. Hence $D_C$ is self-adjoint,
$0<D_C\le I$, and $I+D_C$ is invertible. Expansion of (CCL.2) gives
$$
\mathcal L_C=\langle Z,(I-D_C^2)Z\rangle
 +2\beta\langle R,(I-D_C)Z\rangle+\beta^2|R|^2.
$$
All functions of $D_C$ commute. Completing this square uses
$(I-D_C^2)(I+D_C)^{-1}=I-D_C$ and leaves
$$
\beta^2[I-(I-D_C)(I+D_C)^{-1}]
       =2\beta^2D_C(I+D_C)^{-1},
$$
proving (CCL.3), even when an eigenvalue equals one. The first
term is nonnegative. The increasing scalar function $d/(1+d)$
and $D_C\ge d_r I$ prove (CCL.4). Finally
$I-D_C^2\ge(1-d_t^2)I$, giving (CCL.5).
All steps are pointwise under the actual graph and physical cap input.
:::

(sec-ccl-gaussian)=
## 3. Conditional coercivity integrating all fresh OU outcomes

:::{prf:theorem} Actual population and finite-array conditional cap bound
:label: thm-ccl-conditional-coercivity

Let $g_d=\mathbb E|\xi|=\sqrt2\,\Gamma((d+1)/2)/\Gamma(d/2)$,
and $r_H=mb$. For the prepared population provider put
$\bar X_1=\mathbb E_{\rm prep}|X|$ and
$\bar U_1=\mathbb E_{\rm prep}|U|$. For finite arrays put
$\bar X_1=N^{-1}\sum_j|X_j|$ and
$\bar U_1=N^{-1}\sum_j|U_j|$, using the actual complete
prepared array. Define the nonnegative $\mathscr P$-measurable root bound
$$
C_*(X,U)=(mc+t^2)|U|+r_H|X|+mqg_d
       +a[c(\bar U_1+t\bar X_1)+qg_d].
\tag{CCL.6}
$$
Then every actual prepared population root satisfies
$$
\mathbb E[\mathcal L_C\mid\mathscr P]
             \ge f(C_*(X,U))|R|^2.
\tag{CCL.7}
$$
For a finite array this holds at each row conditional on the entire
prepared array, and therefore
$$
\mathbb E\langle\mathcal L_C\rangle_N
\ge\mathbb E\frac1N\sum_i f(C_*(X_i,U_i))|R_i|^2.
\tag{CCL.8}
$$
The expectations retain all fresh row Gaussians and the resulting
correlated graph. No empirical moment is replaced by a population moment.
For a random prepared population provider the same result holds
conditionally on that provider and then under its actual outer law.
:::

:::{prf:proof}
For the own count provider define
$$
d_y=\int K(y-y')\,d\Lambda(y',w'),\qquad
M_y=\int K(y-y')w'\,d\Lambda(y',w'),
$$
where $\Lambda$ is the actual joint second-stage provider. For finite
arrays the integrals are the actual normalized sums, including the
self numerator and denominator terms that cancel in $L_yw$.
Then
$$
z=(m-ad_y)w-tx_1+aM_y.
$$
Since $0\le d_y\le1$ and $m-a=.9936>0$,
$$
|z|\le m|w|+t|x_1|+a\int|w'|\,d\Lambda(y',w').
\tag{CCL.9}
$$
This is pointwise, even though $d_y,M_y,w$ and $y$ are correlated.
The exact affine stage gives
$$
|w|\le c|U|+ct|X|+q|\xi|,
\qquad |x_1|\le m|X|+t|U|.
$$
In a population provider its first velocity moment is at most
$c(\bar U_1+t\bar X_1)+qg_d$ by the actual Gaussian marginal.
In a finite array the empirical first velocity moment is random;
its expectation conditional on the whole preparation has the same upper
bound by linearity over its actual row Gaussians. Substitution in (CCL.9)
therefore proves $\mathbb E[|z|\mid\mathscr P]\le C_*(X,U)$.
No graph/noise independence was used.

The function $f$ in (CCL.4) is decreasing and convex on $[0,\infty)$:
with $v=1+u/V\ge1$, its second derivative has positive numerator
$4\beta^2(3v^2-1)/V^2$. Since $R$ is fixed before the fresh OU,
(CCL.4) and conditional Jensen give
$$
\mathbb E[\mathcal L_C\mid\mathscr P]
\ge|R|^2\mathbb E[f(|z|)\mid\mathscr P]
\ge|R|^2 f(\mathbb E[|z|\mid\mathscr P])
\ge|R|^2 f(C_*(X,U)).
$$
This proves (CCL.7). Average rows and their actual preparation for
(CCL.8). The domination uses the entire Gaussian expectation, including
outcomes that cancel the deterministic physical velocity. It does not
assert that a large prepared speed gives a deterministic lower bound
on $|z|$.
:::

(sec-ccl-source)=
## 4. Source-box form without compactifying the prepared positions

:::{prf:corollary} Full-jitter source-dependent coercivity register
:label: cor-ccl-source-box

For an actual prepared population with $X=S+IJ$,
$S\in[-L,L]^d$, $I\in\{0,1\}$ and
$J=\sigma_JG$, $G\sim N(0,I_d)$ independent of its frozen source plan,
put
$$
X_1^{\rm box}=\sqrt d L+\sigma_Jg_d,
$$
$$
C_0=(mc+t^2)V_c+r_H\sqrt d L+mqg_d
       +a[c(V_c+tX_1^{\rm box})+qg_d],\qquad
C_J=r_H\sigma_J.
\tag{CCL.10}
$$
Under the actual full preparation law,
$$
\mathbb E\mathcal L_C
\ge\mathbb E\big[f(C_0+C_J|G|)|R|^2\big].
\tag{CCL.11}
$$
Here $R$ may depend on $G$, the full accepted component law and the
actual first graph. No independence of those factors is assumed.
For finite arrays replace $X_1^{\rm box}$ inside the provider term
by the actual bound
$\sqrt dL+\sigma_J N^{-1}\sum_j|G_j|$ before taking outer expectation.
All Gaussian draws remain in the finite-array weighted register.
:::

:::{prf:proof}
Every persistent, copied or revived source is in the alive box before
its own recipient jitter. Thus $|X|\le\sqrt dL+\sigma_J|G|$ and
$\bar X_1\le\sqrt dL+\sigma_Jg_d$ in the population law.
The original component formula gives the stated $V_c$ bound; the first
count convexity gives $|U|,\bar U_1\le V_c$.
Substitution bounds (CCL.6) by $C_0+C_J|G|$. Since $f$ is decreasing,
(CCL.7) and outer averaging prove (CCL.11). For a finite array the
prepared jitter average stays inside its own provider moment. Its
conditional expectation cannot be substituted in a product containing
an arbitrary correlated $R_i$, so the displayed random bound is retained.
:::

(sec-ccl-signed-interface)=
## 5. What this contributes to the general signed block

:::{prf:remark} Closed cap coercivity and the remaining absorption
:label: rem-ccl-signed-interface

Theorem {prf:ref}`thm-dbl68-ledger` has the exact physical change
$-\mathfrak D_H+\mathfrak J_1+\mathfrak J_2-\mathfrak C$.
The result here gives a proved lower bound for its actual
$\mathfrak C$ and preserves the additional nonnegative force-centered
square in (CCL.3)--(CCL.5). This bound is valid for genuinely noisy
velocities, arbitrary square-integrable prepared positions and every
finite array, with no pointwise velocity-band hypothesis.

The bound is weighted by actual prepared coordinates and derivatives.
It does not replace $\mathbb E[f(C_*)|R|^2]$ by
$\mathbb E f(C_*)\mathbb E|R|^2$, remove its full-jitter tails or
assert that high prepared velocity implies a high physical $|z|$.
Absorbing the remaining first/second signed pair forms must use these
weights and the correlated square together. No positive general margin
for their completed sum has been proved here.

Actual source/Haar preparation differences and each terminal-alive or
whole-swarm survival normalization also remain outside this physical
conditional calculation. Even if a physical block gap is later obtained,
those additional sums must be discharged in the exact chronological
marked-law response before claiming the full active default convergence
rate. This note proves conditional cap coercivity, not that rate.
:::

(sec-ccl-first-provider)=
## 6. A sharper general first-force bound with the local product retained

:::{prf:lemma} Independent-environment Cauchy bound for the actual first force
:label: lem-ccl-first-force

For an actual prepared population and its physical interpolation
let $r=\dot X$, $d_X=\|r\|_2$, $|P|\le V_c$ and
$\|P\|_2\le r_0$. With $\ell=\sup|\nabla K|=e^{-1/2}$,
the complete first spatial force obeys
$$
\|B_1\|_2\le\ell(V_c+3r_0)d_X.
\tag{CCL.12}
$$
The same estimate holds for each fixed finite prepared array with
empirical velocity RMS at most $r_0$ and normalized array cost.
For a random finite prepared array, define its actual
$r_P=(N^{-1}\sum_i|P_i|^2)^{1/2}$ and
$d_X^{\rm arr}=(N^{-1}\sum_i|r_i|^2)^{1/2}$. Without a
per-realization RMS bound the complete conclusion is
$$
\mathbb E\langle|B_1|^2\rangle_N
\le\ell^2\mathbb E[(V_c+3r_P)^2(d_X^{\rm arr})^2].
\tag{CCL.13}
$$
In particular population $V_c=4,r_0=.55$ gives a coefficient
less than $3.43$ in (CCL.12), while a fixed or moment-good finite
array with $r_P\le.56$ gives a coefficient less than $3.448$.
The physical cap lemma does not assert that these RMS bounds hold
on every comparison path.
:::

:::{prf:proof}
For an independent copy of the complete prepared root and its differential,
write $D=X-X'$. The own field derivative is exactly
$$
B_1=\mathbb E'[K(D)(D\cdot(r-r'))(P-P')].
$$
Since $K(D)|D|\le\ell$, expansion of the product of the two
triangle bounds gives the pointwise independent-environment inequality
$$
|B_1|\le\ell\big[
 |r||P|+|r|\mathbb E'|P'|
 +|P|\mathbb E'|r'|+\mathbb E'(|r'||P'|)\big].
\tag{CCL.14}
$$
The local factor $|r||P|$ is bounded by $V_c|r|$,
without factoring its random coordinates. The primed factors obey
$\mathbb E'|P'|\le r_0$, $\mathbb E'|r'|\le d_X$ and
$\mathbb E'(|r'||P'|)\le r_0d_X$ by Cauchy--Schwarz inside
the independent environment. Thus the root $L^2$ norm is at most
$$
\ell[(V_c+r_0)d_X+d_X\|P\|_2+r_0d_X]
\le\ell(V_c+3r_0)d_X.
$$
This proves (CCL.12). For a fixed finite array use the actual
uniform-index averages in exactly the same argument. The self
force is zero and its inclusion in the triangle sums only increases
the upper bound; no denominator is changed. Conditional on a random
prepared array it gives the pathwise estimate with $r_P$ and
$d_X^{\rm arr}$. Square and then average for (CCL.13).

The certified elementary bound $\ell<.607$ gives
$.607(4+3(.55))=3.42955<3.43$ and
$.607(4+3(.56))=3.44776<3.448$ exactly.
An averaged finite velocity budget alone does not replace the mixed
expectation in (CCL.13) by a product. A per-realization moment-good
event permits (CCL.12) on that event; its complement, and any subsequent
own-survival conditioning, require their own actual charges. Neither
charge is omitted or supplied by this first-force estimate.
:::
