# Bilinear provider balance for arbitrary prepared count laws

(sec-dbl68-register)=
## 1. Actual carrier and general comparison paths

:::{prf:definition} General prepared-law signed register
:label: def-dbl68-register

Retain the exact default harmonic count stages of
{prf:ref}`def-rfk-retained` and {prf:ref}`def-dsg-register`:
$d=3$, $h=.04$, $t=.02$, $\nu=.3$, $a=t\nu=.006$,
$c=e^{-.04}$, $b=t(1+c)$, $a_x=1-tb$ and $\beta=.04$.
The width-one Gaussian interaction kernel, both own count providers,
the complete OU and final position innovations and the native cap remain
unchanged. Put
$$
f=(\beta-t)a_x-ct,\qquad
g=c+(\beta-t)b,
$$
$$
A_r=ba_x+gf,\qquad A_p=b^2+g^2.
\tag{DBL.1}
$$

Couple any two prepared phase laws with square-integrable positions and
$|P_j|\le V_c=4$, and interpolate $X_\theta=X_0+\theta r$,
$P_\theta=P_0+\theta p$. The source/component plan and any previously
sampled jitters belong to this input coupling. Only the new OU and final
innovations are shared after it, independently of that coupling.
All quantities below are evaluated at an arbitrary interpolation point.
For finite arrays the same definition applies to the full normalized
array Hilbert space, with outer expectation over a random input plan
when present. Every count operator uses its actual denominator $N$.

Let $L_1$ be the own first count Laplacian and set
$$
B_1=-L_{\dot k_1}P,\qquad
F_1=B_1-L_1p,\qquad U=P-aL_1P.
$$
Then the complete first velocity differential is $E=p+aF_1$.
At the own second stage define
$$
R=a_xr+bE,\qquad \mathcal W=c(E-tr),\qquad
T_0=\mathcal W+(\beta-t)R,
$$
$$
F_2=B_2-L_2\mathcal W,
\qquad Z=\mathcal W-tR+aF_2,
\qquad C=D_CZ,\quad D_C=DC_V(z).
$$
Here $B_2$ uses the complete $R$, and $L_2$ and $z$ use the actual
joint OU stage. The physical quadratic omits terminal marks and any
survival normalization. No pointwise narrow-velocity class, independence
of displacements from velocities, or sign of a provider response is
assumed.
:::

(sec-dbl68-coefficients)=
## 2. Oriented first-provider coefficients

:::{prf:lemma} Small position coefficient in the first bilinear response
:label: lem-dbl68-coefficients

The exact harmonic coefficients satisfy
$$
.0399074<A_r<.0399395,\qquad
.92464<A_p<.926183,
$$
$$
0<aA_r<.000240,\qquad
0<af<.000004706.
\tag{DBL.2}
$$
These inequalities concern the coefficients of signed inner products.
They do not bound their correlated random factors separately.
:::

:::{prf:proof}
Use the analytic interval $.96<c<.9608$ already certified in
{prf:ref}`lem-rfk-anisotropic-majorant`. It gives
$$
.0392<b<.039216,\qquad
.99921568<a_x<.999216,
$$
$$
.0007683072<f<.0007843264,
\qquad .96078368<g<.96158464.
$$
All factors are positive. Inserting their endpoints in (DBL.1) gives
the exact rational comparisons
$$
.0399074<\frac{2435756327819}{61035156250000}<A_r
 <\frac{1218855312347}{30517578125000}<.0399395,
$$
$$
.92464<.9246419197543424<A_p
 <.9261829145399296<.926183.
$$
The two fractions are the exact lower and upper endpoint products.
All displayed decimals are exact terminating rational values. Finally
$a(.0399395)<.000240$ and
$a(.0007843264)=.0000047059584<.000004706$.
:::

(sec-dbl68-identity)=
## 3. Both providers in one exact signed account

:::{prf:theorem} Complete first-bilinear and second-Gaussian identity
:label: thm-dbl68-ledger

Use {prf:ref}`def-dbl68-register`, and define the bare harmonic
differentials and deficit
$$
R_H=a_xr+bp,\qquad T_H=fr+gp,
$$
$$
\mathfrak D_H=
\mathbb E Q_\beta(r,p)-\mathbb E(|R_H|^2+|T_H|^2).
\tag{DBL.3}
$$
For a first-stage prepared pair put
$$
D=X-X',\quad H=P-P',\quad
\eta_1=r-r',\quad \pi_1=p-p',\quad
\Psi_1=A_r\eta_1+A_p\pi_1.
$$
Its exact first-provider contribution is
$$
\mathfrak J_1=
a\mathbb E_{\rm prep,pair}
 K(D)\Psi_1\cdot[(D\cdot\eta_1)H-\pi_1]
+a^2A_p\mathbb E\|F_1\|^2.
\tag{DBL.4}
$$
For the actual second-stage prepared pair, before its fresh OU
difference, put
$$
\eta_2=R-R',\quad \psi_2=\mathcal W-\mathcal W',\quad
\zeta_2=T_0-T_0'.
$$
Evaluate $\kappa_G,\mathcal M_G$ from
{prf:ref}`lem-dsg-gaussian-response-tensor` at this pair's actual
$X-X'$ and $U-U'$. Define
$$
\mathfrak J_2=
a\mathbb E_{\rm prep,pair}
 \zeta_2\cdot(\mathcal M_G\eta_2-\kappa_G\psi_2)
+a^2\mathbb E\|F_2\|^2,
$$
$$
\mathfrak C=
\mathbb E\left[
 |(I-D_C)Z+\beta R|^2
 +2\langle(I-D_C)Z,D_CZ\rangle\right]\ge0.
\tag{DBL.5}
$$
Then the actual complete physical differential obeys the equality
$$
\mathbb E Q_\beta(\dot x^+,\dot v^+)
-\mathbb E Q_\beta(r,p)
=-\mathfrak D_H+\mathfrak J_1+\mathfrak J_2-\mathfrak C.
\tag{DBL.6}
$$
Also
$$
\mathfrak D_H\ge.0015\|r\|_2^2+.073\|p\|_2^2.
$$
The assertions hold for population laws and each actual finite array,
with normalized finite pair sums and no finite-size residual.
:::

:::{prf:proof}
The complete first-field derivative is $E=p+aF_1$, so
$$
R=R_H+baF_1,\qquad T_0=T_H+gaF_1.
$$
Expanding these two squares gives the exact first response
$$
\mathbb E(|R|^2+|T_0|^2)
=\mathbb E(|R_H|^2+|T_H|^2)
+2a\langle A_rr+A_pp,F_1\rangle
+a^2A_p\|F_1\|^2.
$$
For the own Gaussian kernel, pair symmetrization gives
$$
2\langle A_rr+A_pp,B_1-L_1p\rangle
=\mathbb E_{\rm prep,pair}
 K(D)\Psi_1\cdot[(D\cdot\eta_1)H-\pi_1].
$$
The derivative of the kernel is $-K(D)D\cdot\eta_1$;
the definition $B_1=-L_{\dot k_1}P$ therefore gives the plus
sign on its velocity product. This proves (DBL.4).

Now apply the exact signed identity
{prf:ref}`thm-dsg-signed-second-response` to the actual $E$.
It gives the second response and the complete cap loss in (DBL.5).
Substitution proves (DBL.6). The bare harmonic deficit follows from
{prf:ref}`lem-rfk-anisotropic-majorant` applied to $(r,p)$.

All first-stage velocities and kernel gradients are bounded, so
$F_1,R,\mathcal W$ have finite $L^2$ norm for the declared input
coupling. The conditional Gaussian estimate (DSG.4) bounds
$B_2$ in $L^2$ even with unbounded positional differences.
Thus every displayed inner product is integrable, with no fourth-moment
displacement hypothesis. The final position innovation has zero
differential under the shared-noise coupling.

For finite arrays, each symmetrization is the exact normalized
$N^{-2}$ pair sum. First-stage and second-stage self differences are
zero. Only distinct OU rows use the independent Gaussian difference;
the zero self contribution is not assigned an independent innovation.
Outer expectation, when the prepared array is random, preserves the
equalities. Neither graph is replaced by its expectation.
:::

(sec-dbl68-local)=
## 4. The remaining local displacement--velocity product

:::{prf:lemma} Pointwise completion of the first signed pair form
:label: lem-dbl68-completion

For a first pair in (DBL.4), set
$u=(D\cdot\eta_1)H$. Its complete linear integrand has the identity
$$
\begin{split}
\Psi_1\cdot(u-\pi_1)={}&
-A_p\left|\pi_1-\tfrac12u+
                    \frac{A_r}{2A_p}\eta_1\right|^2\\
&+\frac{A_p}{4}|u|^2
 +\frac{A_r}{2}\eta_1\cdot u
 +\frac{A_r^2}{4A_p}|\eta_1|^2.
\end{split}
\tag{DBL.7}
$$
In particular its positive remainder includes the local quantity
$K(D)(D\cdot\eta_1)^2|H|^2$. No averaged velocity bound
alone controls this quantity by its product with an averaged
displacement bound.
:::

:::{prf:proof}
Expand the square in (DBL.7); the terms involving $\pi_1$ are
$-A_p|\pi_1|^2+A_p\pi_1\cdot u-A_r\eta_1\cdot\pi_1$.
The remaining two cross contributions add to
$A_r\eta_1\cdot u$, proving the identity.

For the asserted limitation on averaged moments, let $I$ be Bernoulli
with parameter $e\in(0,1)$, and use an independent copy $I'$.
Choose bounded velocities $P=2Ie_1$ and a displacement
$r=\epsilon Ie_1$, with $0<\epsilon\le.1$. They are admissible
bounded prepared coordinates; positions can be fixed at two points
inside the source box. Then
$$
\mathbb E|P|^2=4e,\qquad
\mathbb E|H|^2=8e(1-e),\qquad
\mathbb E|\eta_1|^2=2\epsilon^2e(1-e),
$$
$$
\mathbb E[|H|^2|\eta_1|^2]
=4\mathbb E|\eta_1|^2.
\tag{DBL.8}
$$
Thus the last ratio stays equal to four while the average velocity
and pair-velocity moments tend to zero. Taking, for example,
$X=Ie_1/2$ makes $K(D)(D\cdot\eta_1)^2$ a fixed positive
multiple of $|\eta_1|^2$ on the nonzero pairs. This is a
counterexample to factoring these correlated moments, not a
counterexample to contraction of the actual full kinetic map.
It does not assert that this comparison input is an invariant or
reachable class of the complete active process.
:::

(sec-dbl68-endpoint)=
## 5. Exact absorption interface and present scope

:::{prf:remark} Unclosed inference required for a general kinetic gap
:label: rem-dbl68-scope

For a given comparison path, a physical differential rate $\kappa>0$
follows from this ledger exactly when the completed signed account
satisfies
$$
\mathfrak J_1+\mathfrak J_2-\mathfrak C
\le\mathfrak D_H-\kappa\mathbb E Q_\beta(r,p)
$$
uniformly along the path. This inequality is an explicitly remaining
interface here, not an additional assertion discharged by an average
post-burn velocity moment. In particular the latter cannot replace
the local products in (DBL.7) or justify factoring an averaged cap
Jacobian from a correlated displacement.

The position coefficient $aA_r<.000240$ and the second bare position
coefficient $af<.000004706$ expose cancellations hidden by a scalar
principal-norm/force-norm bound. However the actual second tensor
contains the mixed $H\mu_G^{\mathsf T}$ term, its displacement
arguments already contain $F_1$, and its quadratic force square remains
correlated with the native cap loss. The present identity retains all
these terms. It does not replace them by a favorable sign or combine
different root marginals.

The completed result is a general exact bilinear physical account,
not another pointwise velocity-band contraction class. It supplies
the missing terms of a potential signed consumer without a claimed
general margin. Even a later physical kinetic gap would still need
the actual component/source preparation response and separately
normalized terminal-alive law account before yielding a default
delayed nonlinear or finite survivor mixing rate.
:::
