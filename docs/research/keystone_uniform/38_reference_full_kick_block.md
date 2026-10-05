# Full-kick cap balance at the harmonic reference viscosity

(sec-rfk-retained)=
## 1. Retained state and exact carrier

:::{prf:definition} The complete-kick comparison arm
:label: def-rfk-retained

Retain the harmonic record of research notes 27, 33, 34 and 36:
$F(x)=-x$, $d=3$, $L=2$, $h=1/25$, $t=1/50$, $\nu=3/10$,
$\gamma=b_O=\rho=1$, $V=2$, $V_c=4$ and
$\sigma_J=\sigma_x=1/10$. Define
$$
c=e^{-1/25},\quad b=t(1+c),\quad a_x=1-tb,\quad
m=1-t^2,\quad a=t\nu=3/500,\quad \beta=1/25,
$$
$$
q^2=(1-c^2)/2,\qquad s^2=1/2500,\qquad \ell=e^{-1/2},
\qquad Q_\beta(r,p)=|r|^2+2\beta r\cdot p+|p|^2.
$$
The actual preparation uses the original frozen slot velocities for
its component collision. The source is alive and lies in the box;
Gaussian recipient jitter is sampled after the source/component plan.
The current-frame measured fitness and its conditional-alive
normalizers, active acceptance and mandatory revival remain unchanged.

The kinetic stages, with both independent full Gaussian innovations,
are
$$
U=(I-aL_X)P,\quad w=c(U-tX)+q\xi,\quad
y=a_xX+bU+tq\xi,
$$
$$
z=(I-aL_y)w-ty,\quad x^+=y+s\chi,\quad
v^+=C_V(z)=Vz/(V+|z|).
\tag{RFK.1}
$$
For finite arrays, $L_y$ is the actual noisy count operator. For the
population carrier, it is the operator on the lifted root probability
space of the actual joint $(y,w)$ law. In particular that joint law is
not a product of its position and velocity marginals.

Section 2 certifies the complete principal part of (RFK.1), at the
default viscosity, for arbitrary noncommuting count operators. Section
3 restores the spatial derivatives of both actual operators. Sections
4--5 prove correlation-preserving consumers for the second derivative.
These are complete intermediate estimates. Their remaining signed
terms are not assumed to satisfy a positive full-law margin.
:::

(sec-rfk-principal)=
## 2. An anisotropic whole-map certificate for both count kicks

:::{prf:lemma} Strong harmonic majorant before the two alignment defects
:label: lem-rfk-anisotropic-majorant

For every pair of vectors, or elements of a real Hilbert space,
$r,p$, put
$$
R=a_xr+bp,\qquad W=c(p-tr),\qquad Z_0=W-tR.
$$
Then
$$
\|R\|^2+\|Z_0+\beta R\|^2
\le Q_\beta(r,p)-\kappa_x\|r\|^2-\kappa_v\|p\|^2,
\quad \kappa_x=3/2000,\quad \kappa_v=73/1000.
\tag{RFK.2}
$$
Also $a_x^2+b^2<1$.
:::

:::{prf:proof}
Write $r_H=t(c+a_x)$ and $v_H=c-tb$. The matrix of the
majorant deficit is
$$
M=\begin{pmatrix}1&\beta\\\beta&1\end{pmatrix}
-\binom{a_x}{b}(a_x,b)
-\binom{\beta a_x-r_H}{v_H+\beta b}
             (\beta a_x-r_H,v_H+\beta b).
$$
Here is an entirely rational certificate. The elementary exponential
bounds $0.96<c<0.9608$ imply
$$
0.0392<b<0.039216,\quad
0.99921568<a_x<0.999216,
$$
$$
0.0391843136<r_H<0.03920032,\quad
0.95921568<v_H<0.960016.
$$
Consequently, with $f=\beta a_x-r_H$ and $g=v_H+\beta b$,
$$
0.0007683072<f<0.0007843264,\qquad
0.96078368<g<0.96158464.
$$
The two diagonal entries of
$M-\operatorname{diag}(\kappa_x,\kappa_v)$ are strictly greater
than $0.00006$ and $0.0008$. Its off-diagonal absolute value
is less than $0.0001$: insert the displayed intervals into
$\beta-a_xb-fg$. Its determinant therefore exceeds
$$
0.00006(0.0008)-0.0001^2=0.000000038>0.
$$
This proves the positive definiteness needed for (RFK.2).
Finally the rational upper bound
$0.999216^2+0.039216^2=0.999970509312<1$ proves the last assertion.
The Hilbert-space statement follows by applying the scalar block
quadratic to each coordinate, or its tensor product with the identity.
:::

:::{prf:theorem} Both count-alignment principal parts and the native cap
:label: thm-rfk-two-count-principal

Let $\mathcal H$ be a real Hilbert space of vector fields. Let
$A_1,A_2$ be any self-adjoint operators satisfying
$(1-a)I\le A_j\le I$. No commutation between them is required.
Let $D$ be any self-adjoint operator satisfying $0\le D\le I$.
For
$$
p_1=A_1p,\quad R=a_xr+bp_1,\quad
W=c(p_1-tr),\quad Z=A_2W-tR,\quad C=DZ,
$$
one has
$$
Q_\beta(R,C)
\le Q_\beta(r,p)-\frac{149}{100000}\|r\|^2
                     -\frac{721}{10000}\|p\|^2
\le \left(1-\frac{149}{104000}\right)Q_\beta(r,p).
\tag{RFK.3}
$$
The result applies pointwise to the principal differential of the
actual finite or population update. There $D$ is the multiplication
operator of actual cap Jacobians and $A_j=I-aL_j$ at the actual
two graphs. Spatial graph derivatives are additional forces; they
are included explicitly in Section 3.
:::

:::{prf:proof}
For $E=Z-C$, $0\le D\le I$ gives
$\|Z\|^2-\|C\|^2\ge\|E\|^2$. Thus
$$
Q_\beta(R,C)\le\|R\|^2+\|Z+\beta R\|^2,
\tag{RFK.4}
$$
by $-\|E\|^2-2\beta\langle R,E\rangle
\le\beta^2\|R\|^2$.

Put $e_2=(I-A_2)W$. Spectral calculus gives
$\langle e_2,W\rangle\ge\|e_2\|^2/a$. With
$k=(2-a)/a$ and $\delta=1/k=a/(2-a)$, the difference between
the right side of (RFK.4) and
$\|R\|^2+\|W+(\beta-t)R\|^2$ is at most
$$
-k\|e_2\|^2-2(\beta-t)\langle R,e_2\rangle
\le\delta(\beta-t)^2\|R\|^2.
$$
Apply (RFK.2) with $p_1$ in place of $p$ and use
$\|R\|^2\le\|r\|^2+\|p_1\|^2$.

For $e_1=(I-A_1)p$, the same spectral calculation yields
$$
Q_\beta(r,p_1)-Q_\beta(r,p)
\le-k\|e_1\|^2-2\beta\langle r,e_1\rangle
\le\delta\beta^2\|r\|^2.
$$
Moreover $\|p_1\|\ge(1-a)\|p\|$. Therefore the two
respective loss coefficients are at least
$$
\kappa_x-\delta[\beta^2+(\beta-t)^2]
>149/100000,
$$
$$
[\kappa_v-\delta(\beta-t)^2](1-a)^2
>721/10000.
$$
Both are direct rational inequalities at $a=3/500$.
Finally $Q_\beta(r,p)\le(1+\beta)(\|r\|^2+\|p\|^2)$.
The smaller loss coefficient divided by $1+\beta=26/25$ is
$149/104000$, proving (RFK.3).

For the actual cap the radial and tangential eigenvalues of its
Jacobian are $V^2/(V+|z|)^2$ and $V/(V+|z|)$, both in $[0,1]$.
For actual count operators, symmetry gives
$0\le L_j\le I$ on the normalized array Hilbert space or lifted
population space. These verify every operator hypothesis without
requiring the noisy second graph to be independent of its OU noise.
:::

(sec-rfk-own-signed)=
## 3. Full own-graph differential and signed forcing account

:::{prf:lemma} Actual complete differential with both spatial graph forces
:label: lem-rfk-own-differential

Couple two prepared laws or arrays and interpolate their phase values
$X_\theta=X+\theta r$, $P_\theta=P+\theta p$.
Keep each own OU and final innovation shared across this interpolation;
within each marginal the two full innovation arrays remain independent.
Let a prime denote an independent copy for the population carrier,
or the count sum over the second index for finite arrays. Set
$$
k_1=K(X_\theta-X_\theta'),\qquad
\dot k_1=\nabla K(X_\theta-X_\theta')\cdot(r-r'),
$$
$$
B_1=-L_{\dot k_1}P_\theta,\quad A_1=I-aL_1,\quad
p_1=A_1p,\quad e_1=aB_1.
$$
At the actual second stage put
$$
k_2=K(y_\theta-y_\theta'),\qquad
\dot k_2=\nabla K(y_\theta-y_\theta')\cdot(R-R'),
$$
$$
B_2=-L_{\dot k_2}w_\theta,\quad A_2=I-aL_2,\quad e_2=aB_2.
$$
The exact phase differential, including the cap, is
$$
R=a_xr+b(p_1+e_1),\qquad W=c(p_1+e_1-tr),
$$
$$
Z=A_2W-tR+e_2,\qquad C=DC_V(z_\theta)Z.
\tag{RFK.5}
$$
Define
$$
D_1=\langle p,L_1p\rangle,\quad E_1=\langle r,L_1p\rangle,
\quad D_2=\langle W,L_2W\rangle,
$$
$$
E_2=\langle R,L_2W\rangle,\qquad
T=A_2W+(\beta-t)R.
$$
Then the complete differential obeys the signed account
$$
\begin{split}
Q_\beta(R,C)-Q_\beta(r,p)\le{}&
-2aD_1+a^2\|L_1p\|^2-2\beta aE_1\\
&+2a\langle p_1+\beta r,B_1\rangle+a^2\|B_1\|^2\\
&-\kappa_x\|r\|^2-\kappa_v\|p_1+aB_1\|^2\\
&-2aD_2+a^2\|L_2W\|^2-2a(\beta-t)E_2\\
&+2a\langle T,B_2\rangle+a^2\|B_2\|^2.
\end{split}
\tag{RFK.6}
$$
The norms and inner products are the actual normalized array values
or population expectations at $\theta$. All moments needed for the
differentiation are finite for source-box preparation with uncut
Gaussian jitter. The interpolated providers are comparison objects;
the endpoints are exactly the actual own providers.
:::

:::{prf:proof}
Differentiate each count operator before its kick. Kernel differentiation
gives $\dot U=A_1p+aB_1$ and
$\dot z=A_2\dot w-t\dot y+aB_2$. The affine harmonic and OU
stages give (RFK.5); the shared final Gaussian has derivative zero.
The width-one Gaussian kernel and its derivative are bounded, while
prepared velocities are bounded and prepared positions have all
Gaussian moments. Their lifted derivatives consequently exist in
$L^2$. The same finite-array formulas are ordinary derivatives.

Write $F=W+(\beta-t)R$. Apply (RFK.4) to the actual differential,
and expand the majorant as
$$
\begin{split}
\|R\|^2+\|F+e_2-aL_2W\|^2
={}&\|R\|^2+\|F\|^2-2aD_2+a^2\|L_2W\|^2\\
&-2a(\beta-t)E_2+2a\langle T,B_2\rangle
   +a^2\|B_2\|^2.
\end{split}
$$
Apply (RFK.2) to $(r,p_1+e_1)$. Expand the remaining
$Q_\beta(r,p_1+e_1)-Q_\beta(r,p)$ exactly. This is the first
two lines of (RFK.6), proving the claim. The argument has not
replaced the signed forms $E_1,E_2$ by absolute values.
:::

:::{prf:lemma} Pair forms that consume the actual spatial forces
:label: lem-rfk-pair-consumer

For $j=1,2$, define
$$
S_j=\frac12\mathbb E\mathbb E'
       \left[\frac{\dot k_j^2}{k_j}|V_j-V_j'|^2\right],
\qquad V_1=P_\theta,\quad V_2=w_\theta.
\tag{RFK.7}
$$
For finite arrays the expectations are normalized count sums.
For every test vector field $f$,
$$
\|B_j\|^2\le2S_j,\qquad
|\langle f,B_j\rangle|
\le\sqrt{\langle f,L_jf\rangle S_j}.
\tag{RFK.8}
$$
Consequently
$$
|\langle p_1+\beta r,B_1\rangle|
\le[(1+a\sqrt2)\sqrt{D_1}
      +\beta\sqrt{\langle r,L_1r\rangle}]\sqrt{S_1},
$$
$$
|\langle T,B_2\rangle|
\le[\sqrt{D_2}+|\beta-t|\sqrt{\langle R,L_2R\rangle}]
                                                      \sqrt{S_2}.
\tag{RFK.9}
$$
These estimates retain the same actual pair kernels as the negative
alignment forms in (RFK.6).
:::

:::{prf:proof}
Conditional weighted Cauchy--Schwarz and
$\mathbb E'k_j\le1$ give $\|B_j\|^2\le2S_j$.
Pair symmetrization gives
$\langle f,B_j\rangle
=-\frac12\mathbb E\mathbb E'[\dot k_j(f-f')\cdot(V_j-V_j')]$,
so a second weighted Cauchy--Schwarz gives (RFK.8).
For the first assertion use
$|\langle p,B_1\rangle|\le\sqrt{D_1S_1}$,
$\|L_1p\|\le\sqrt{D_1}$ and $\|B_1\|\le\sqrt{2S_1}$.
For the second, $A_2$ commutes with its own $L_2$ and
$\langle A_2W,L_2A_2W\rangle\le D_2$.
The triangle inequality in the seminorm induced by $L_2$ gives
the displayed bound. There is no assertion that $A_1$ and $A_2$
commute with one another.
:::

(sec-rfk-fresh-ou)=
## 4. Second-graph defect with the complete fresh OU noise

:::{prf:lemma} A conditional Gaussian consumer for the noisy second graph
:label: lem-rfk-fresh-ou-defect

Conditional on the complete coupled preparation and its jitters, put
$\bar w=c(U_\theta-tX_\theta)$, so
$w_\theta=\bar w+q\xi$. In this conditioning $R=\dot y_\theta$
is fixed and is independent of the fresh OU noises. Then
$$
S_2\le\frac1e\mathbb E_{\rm prep}\mathbb E_{\rm prep}'
 \left[|R-R'|^2\{ |\bar w-\bar w'|^2+2dq^2\}\right].
\tag{RFK.10}
$$
Writing $M_{\bar w}^2=\mathbb E|\bar w|^2$, this implies
$$
S_2\le\frac8e\left\{
\mathbb E[|R|^2|\bar w|^2]+M_{\bar w}^2\mathbb E|R|^2
\right\}+\frac{4dq^2}{e}\mathbb E|R|^2.
\tag{RFK.11}
$$
For finite arrays, condition on the realized complete preparation
and write $\langle\cdot\rangle_N$ for its empirical row average.
The precise finite counterpart is
$$
\mathbb E_\xi S_2\le\frac8e\left\{
\langle|R|^2|\bar w|^2\rangle_N+
\langle|R|^2\rangle_N\langle|\bar w|^2\rangle_N
\right\}+\frac{4dq^2}{e}\langle|R|^2\rangle_N.
\tag{RFK.11f}
$$
The counterpart of (RFK.10) uses the same conditional empirical
pair average. If preparation is subsequently random, the product
$\mathbb E_{\rm prep}[\langle|R|^2\rangle_N
\langle|\bar w|^2\rangle_N]$ remains a mixed expectation.
It is not replaced by a product of unconditional moments.
No independence between $K(y-y')$ and the full OU array is assumed.
:::

:::{prf:proof}
The pointwise identity
$|\nabla K(z)|^2/K(z)=|z|^2e^{-|z|^2/2}\le2/e$
bounds (RFK.7) before any conditioning by
$e^{-1}\mathbb E\mathbb E'[|R-R'|^2|w-w'|^2]$.
For two distinct roots the independent fresh noises have difference
covariance $2q^2I_d$. Their cross term with $\bar w-\bar w'$
has zero mean. This proves (RFK.10). Finite diagonal pairs have
$R-R'=0$ and cause no exception.
For the first term use
$|R-R'|^2\le2(|R|^2+|R'|^2)$ and
$|\bar w-\bar w'|^2\le2(|\bar w|^2+|\bar w'|^2)$.
Expanding the independent-copy terms gives
$8\{\mathbb E|R|^2|\bar w|^2+
\mathbb E|R|^2\mathbb E|\bar w|^2\}$.
Also $\mathbb E\mathbb E'|R-R'|^2\le2\mathbb E|R|^2$.
These prove (RFK.11); the same normalized deterministic-sum
expansion proves the finite-array version conditional on preparation.
The local mixed product has remained inside its expectation.
:::

(sec-rfk-source-cap)=
## 5. An uncut source-plan and radial-cap consumer

:::{prf:lemma} Exact fresh-jitter source product
:label: lem-rfk-source-product

Suppose the two complete plans are coupled before a shared fresh
$J\sim N(0,\sigma_J^2I_d)$, and
$X_\theta=S_\theta+I_\theta J$,
$r=\delta S+\delta I J$, with $S_\theta\in[-L,L]^d$,
$0\le I_\theta\le1$ and $|\delta I|\le1$.
For each fixed coupled plan,
$$
\begin{split}
\mathbb E_J[|X_\theta|^2|r|^2]={}&
|S_\theta|^2|\delta S|^2
 +\sigma_J^2\{I_\theta^2d|\delta S|^2
 +\delta I^2d|S_\theta|^2
 +4I_\theta\delta I S_\theta\cdot\delta S\}\\
&+I_\theta^2\delta I^2d(d+2)\sigma_J^4.
\end{split}
\tag{RFK.12}
$$
At the reference parameters, with $d_X=\|r\|_2$,
$d_P=\|p\|_2$ and $X_2=\sqrt{12.03}$,
$$
\||X_\theta|\,|r|\|_2^2
\le12.05\mathbb E|\delta S|^2
       +.6015\mathbb E\delta I^2\le20.05d_X^2.
\tag{RFK.13}
$$
If $p$ is fixed before recipient jitter, also
$\||X_\theta|\,|p|\|_2\le X_2d_P$.
:::

:::{prf:proof}
Expand the two squared norms and use centered Gaussian second
and fourth moments. The product of the two linear terms is
$4I_\theta\delta I\sigma_J^2S_\theta\cdot\delta S$;
the fourth moment is $d(d+2)\sigma_J^4$. This proves (RFK.12).
Apply $|S_\theta|^2\le dL^2=12$ and
$4|I_\theta\delta I S_\theta\cdot\delta S|
\le2|\delta S|^2+2dL^2\delta I^2$.
The resulting coefficients are $12.05$ and $.6015$.
Since $d_X^2=\mathbb E|\delta S|^2+.03\mathbb E\delta I^2$
and $.6015/.03=20.05$, (RFK.13) follows.
Conditional on the plan, $\mathbb E_J|X_\theta|^2\le12.03$.
Multiply by its fixed $|p|^2$ and average to obtain the last bound.
:::

:::{prf:theorem} Correlation-safe radial consumer for the own second spatial force
:label: thm-rfk-own-second-cap-consumer

Use the population carrier and the source-plan coupling of the
preceding lemma. Suppose each entering prepared population has
$\|P\|_2\le r_0=.55$; thus every interpolated prepared law has
the same bound and $|P_\theta|\le V_c=4$.
The velocity burn in {prf:ref}`thm-rvb-population-burn` supplies this
hypothesis after its declared six population updates.
Let $M_w=.70$ be an upper bound for every actual intermediate
$\|w_\theta\|_2$, and put
$$
A=m-a=.9936,\quad
D_y=ba\ell(V_c+r_0),\quad A_y=a_x+D_y,
$$
$$
d_Y=(a_x+2D_y)d_X+bd_P,
\qquad H_Y=A_y\sqrt{20.05}\,d_X
                 +b(1+a)X_2d_P+D_yX_2d_X,
$$
$$
S_0=\frac{V/4+aM_w+t^2V_c}{A},\quad
S_x=\frac{tm}{A},\quad
\overline S=\frac{V/4+aM_w+t(mX_2+tr_0)}A<.58.
$$
For the actual $B_2$ of (RFK.5), including its root/environment
dependence and the complete OU noise,
$$
a\|DC_V(z_\theta)B_2\|_2
\le a\ell\{(2M_w+S_0+\overline S)d_Y+S_xH_Y\}
\le.0094d_X+.000366d_P.
\tag{RFK.14}
$$
Also the local product in (RFK.11) satisfies the explicit bound
$$
\||R|\,|\bar w|\|_2
\le c(V_cd_Y+tH_Y),\qquad
M_{\bar w}\le c(r_0+tX_2).
\tag{RFK.15}
$$
This theorem is a population statement. Replacing a random finite
empirical $M_w$ by its averaged RMS in (RFK.14) would require an
additional mixed-product estimate and is not asserted here.
:::

:::{prf:proof}
The first spatial force has the pointwise bound
$$
|B_1|\le\ell(V_c+r_0)(|r|+d_X).
$$
Indeed differentiate the kernel under its actual lifted joint law,
use $|\nabla K|\le\ell$, $|P_\theta|\le V_c$, and apply
Cauchy--Schwarz to its environment term
$\mathbb E'|r'||P_\theta'|\le d_Xr_0$.
Moreover
$|(I-aL_1)p|\le|p|+ad_P$ pointwise: its nonlocal incoming
$a\mathbb E'[k_1p']$ term must be retained. Therefore
$$
|R|\le A_y|r|+b|p|+bad_P+D_yd_X.
$$
Multiplication by $|X_\theta|$, followed by (RFK.13), gives
$\||X_\theta||R|\|_2\le H_Y$.
For the unweighted bound use instead the exact $L^2$ contraction
of $I-aL_1$ and the displayed $B_1$ estimate. This gives
$\|R\|_2\le d_Y$.

Put $x_1=mX_\theta+tU_\theta$, so $y=x_1+tw$.
The actual precap velocity can be rewritten
$$
z=(m-aa_2(y))w-tx_1+aM_2(y).
$$
The provider is the actual joint second law, hence
$0\le a_2\le1$ and $|M_2(y)|\le M_w$.
The native radial Jacobian satisfies $\|DC_V(z)z\|\le V/4$.
Rearrange the preceding identity and use $\|DC_V(z)\|\le1$
to obtain the pointwise cancellation
$$
\|DC_V(z)w\|\le
S_w(x_1):=\frac{V/4+aM_w+t|x_1|}{A}
\le S_0+S_x|X_\theta|.
$$
Its $L^2$ norm is at most $\overline S$: count alignment gives
$\|U_\theta\|_2\le r_0$, while
$\|X_\theta\|_2\le X_2$. No large-jitter event is removed.

In $B_2=-\mathbb E'[\dot k_2(w-w')]$, bound $|\nabla K|$
pointwise before taking expectations. Acting with the root's actual
cap Jacobian and using the preceding cancellation gives
$$
\|DC_V(z)B_2\|
\le\ell\{(M_w+S_w)|R|+(M_w+S_w)\|R\|_2\}.
$$
The environment product is bounded by
$\mathbb E'|R'||w'|\le\|R\|_2M_w$; the root's random
$S_w|R|$ remains inside its norm.
Now apply $\|S_w|R|\|_2\le S_0d_Y+S_xH_Y$ and
$\|S_w\|_2\le\overline S$. This proves the first inequality
in (RFK.14).

Here is a rational upper certificate for its final numerical bound:
use $c<.9608$, $b<.039216$, $a_x<.999216$, $\ell<.607$,
$X_2<3.47$, $\sqrt{20.05}<4.478$ and
$\overline S<.58$. Inserting these bounds gives respective
coefficients below $.0093981$ and $.000365531$, strictly below
the coefficients in (RFK.14). The claimed $M_w$ bound follows
from
$\|w_\theta\|_2^2\le c^2(r_0+tX_2)^2+dq^2<.70^2$.
Finally $|\bar w|\le c(V_c+t|X_\theta|)$ gives the first part
of (RFK.15); count contraction and Minkowski give its second part.
All products involving random source displacement have been proved
with their actual conditional Gaussian moments.
:::

(sec-rfk-endpoint)=
## 6. Last justified balance and the remaining law interfaces

:::{prf:remark} What has and has not closed
:label: rem-rfk-last-interface

The complete principal balance (RFK.3) holds at the actual reference
$\nu=.3$, not just in an infinitesimal-viscosity interval. It uses
both count self-alignment kicks, the native cap and an anisotropic
complete-update quadratic. It remains valid when the two count
operators fail to commute and when the second operator is correlated
with the OU innovations.

For the actual own-provider comparison the spatial terms are exactly
$B_1,B_2$ in (RFK.6). The first stage has the signed source-aware
consumers (RFP.10--17); the second has (RFK.7--15). The radial
consumer removes the former uncut-jitter product obstruction:
all source/jitter outcomes have finite explicit charges and no fixed
bad-event floor occurs. The current estimates do not prove that
these charges fit inside the retained losses in (RFK.6). For example,
the absolute second-force position coefficient $.0094$ alone is
larger than the principal norm-contraction margin; an absolute
triangle inequality cannot complete the claimed default gap.
This is a limitation of that inequality, not a counterexample to
actual law contraction.

The first remaining inference is a complete signed absorption of
the local source-weighted spatial forcing against alignment,
harmonic restoring force and the actual cap loss. Any invocation
of the postburn RMS $.55$ in place of a local factor inside
$\mathbb E[|\delta S|^2|P_\theta|^2]$ still needs a proved
conditional product estimate. The preparations of different
entering laws also require their actual fitness/donor/component
coupling; a fixed-root provider comparison does not include this
change.

For the default killed $L=2$ arm, terminal marking, mandatory-dead
feedback and conditional-alive normalizations must subsequently be
controlled in the same block. A positive alive-mass floor is not
an assertion that the dead mass is small. This note proves no
default own-provider invariant, quasi-stationary law or uniform
full active convergence rate. It also supplies no Rastrigin force
certificate: the harmonic linear kick identities are essential
to (RFK.2)--(RFK.6).
:::
