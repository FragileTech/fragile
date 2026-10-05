# Full-tail first-cap and second-provider spatial consumers

(sec-nca76-register)=
## 1. Actual population source-plan comparison

:::{prf:definition} Source-aware noisy cap comparison
:label: def-nca76-register

Retain the actual default harmonic count kinetic map and constants of
{prf:ref}`def-gca74-register`. In particular $d=3$, $V=2$,
$V_c=4$, $L=2$, $\sigma_J=.1$, $a=.006$ and $\beta=.04$.
Use the actual prepared population source-plan coupling of
{prf:ref}`lem-rfk-source-product`:
$$
X_\theta=S_\theta+I_\theta J,\qquad
r=\dot X_\theta=\delta S+\delta I J,
\quad J\sim N(0,.01I_3),
$$
$$
S_\theta\in[-2,2]^3,\quad 0\le I_\theta\le1,\quad
|\delta I|\le1.
\tag{NCA76.1}
$$
The complete plans are coupled before this shared original recipient
jitter. $P_\theta$ and $p=\dot P_\theta$ are fixed before that jitter,
but may depend on the entire source/component/Haar plans. Assume
$$
|P_\theta|\le4,\qquad \|P_\theta\|_2\le r_0=.55
\tag{NCA76.2}
$$
at every comparison point. Write $d_X=\|r\|_2$, $d_P=\|p\|_2$.

The new OU and final Gaussians are shared only after the preparation;
each individual transition marginal keeps its original independent
Gaussian laws. Both own count graphs and their full spatial responses
are retained. These are physical kinetic comparisons before terminal
alive normalization. The cost of comparing the two actual accepted
preparations is not asserted here.

Put $D=DC_V(z)$ at the actual own noisy precap velocity,
$A_1=I-aL_1$, $A_2=I-aL_2$ and
$$
B_1=-L_{\dot k_1}P_\theta,\qquad B_2=-L_{\dot k_2}w_\theta,
\qquad R=a_xr+b(A_1p+aB_1).
\tag{NCA76.3}
$$
$B_2$ uses this full $R$. The population norms below include the
entire source, jitter and fresh Gaussian expectations. No finite
empirical provider is replaced by these population moments.
:::

:::{prf:lemma} Fixed-vector cap and source products
:label: lem-nca76-fixed-products

With $\kappa=\sqrt{159/200}<.892$, $X_2=\sqrt{12.03}<3.47$
and $X_r=\sqrt{20.05}<4.478$, the actual source and fresh OU laws obey
$$
\||X_\theta|\,|r|\|_2\le X_rd_X,\qquad
\|X_\theta\|_2\le X_2,\qquad \|U_\theta\|_2\le r_0,
$$
$$
\||D X_\theta|\,|r|\|_2\le\kappa X_rd_X,\qquad
\||D|_{\rm op}|r|\|_2\le\kappa d_X,\qquad
\||D|_{\rm op}\|_2\le\kappa,\qquad
\|DP_\theta\|_2\le\kappa r_0.
\tag{NCA76.4}
$$
Also $\||\xi|\,|r|\|_2=\sqrt3\,d_X$ for the root's original
fresh OU vector. No inequality in (NCA76.4) factors a correlated
prepared displacement from a prepared velocity moment.
:::

:::{prf:proof}

The full-jitter source calculation (RFK.12)--(RFK.13) gives the
first product. Direct Gaussian centering before jitter gives
$\mathbb E|X_\theta|^2\le12.03$; no large-jitter set is removed.
First count contraction gives the displayed $U$ moment.
Thus the actual joint OU provider obeys
$$
\mathbb E|w|^2\le c^2(r_0+tX_2)^2+3q^2<.70^2.
$$
Consequently $M_w=.70$ bounds both its velocity RMS and its
first velocity moment. The later environment product Cauchy
estimate uses the RMS bound.

All prepared root values, including $r,X_\theta,P_\theta$, are fixed
before the new OU innovation. Apply the conditional scalar and
fixed-vector estimates (GCA74.12)--(GCA74.13) at each such preparation.
Multiplication by the actual pre-OU $|r|^2$, followed by outer
integration, gives the product bounds involving $D$. The first
source product then gives the stated coefficient $X_r$.
The same reasoning with the fixed vector $P_\theta$ gives
$\|DP_\theta\|_2\le\kappa r_0$.
The root's original $\xi$ is independent of its complete preparation,
so the final identity follows from $\mathbb E|\xi|^2=3$.
None of these uses takes place after a survival restriction.
:::

(sec-nca76-first-cap)=
## 2. Radial cancellation of the first spatial velocity factor

:::{prf:lemma} The actual local capped prepared velocity
:label: lem-nca76-local-first-cap

Define $A=m-a=.9936$ and
$$
S_0=\frac{V/4+aM_w+t^2V_c}{A},\qquad
S_x=\frac{tm}{A},\qquad M_w=.70.
$$
Then the complete actual correlated cap and providers satisfy
$$
\begin{split}
z_v(1-a)|DP_\theta|\le{}&
V/4+r_H|DX_\theta|+mq|D\xi|\\
&+a(z_vr_0+M_w)|D|_{\rm op}
+a(S_0+S_x|X_\theta|).
\end{split}
\tag{NCA76.5}
$$
In particular
$$
\||r|\,|DP_\theta|\|_2\le C_Pd_X,
$$
$$
C_P=
\frac{V/4+r_H\kappa X_r+mq\sqrt3
+a\kappa(z_vr_0+M_w)+a(S_0+S_xX_r)}
     {z_v(1-a)}
<1.06.
\tag{NCA76.6}
$$
This keeps the full Gaussian outcomes that cancel a large prepared
velocity. It is not obtained by replacing $P_\theta$ with its RMS
inside a local product.
:::

:::{prf:proof}

The exact own stages give
$$
z=z_vP_\theta-r_HX_\theta+mq\xi
        -a[z_vL_1P_\theta+L_2w_\theta].
\tag{NCA76.7}
$$
For the native radial derivative, $|Dz|\le V/4$ pointwise.
Apply $D$ to (NCA76.7). The actual first provider has
$\mathbb E|P_\theta'|\le r_0$, hence
$$
|DL_1P_\theta|\le |DP_\theta|+r_0|D|_{\rm op}.
$$
This bounds only the independent environment mean; the root
$DP_\theta$ remains local.

The actual second provider satisfies its own first moment bound
$M_w$. Its exact affine representation is
$$
z=(m-ad_y)w-tx_1+aM_y,\qquad
0\le d_y\le1,\quad |M_y|\le M_w.
$$
Radial cancellation gives
$$
|Dw|\le\frac{V/4+t|x_1|+aM_w}{A}
\le S_0+S_x|X_\theta|,
$$
since $x_1=mX_\theta+tU_\theta$ and $|U_\theta|\le V_c$.
Therefore
$$
|DL_2w|\le S_0+S_x|X_\theta|+M_w|D|_{\rm op}.
$$
Substitution into the differentiated first expression and moving the
$a z_v|DP_\theta|$ term to the left proves (NCA76.5).
Neither $D$ nor the second graph is presumed independent of $w$.

Multiply (NCA76.5) by $|r|$ and take the complete $L^2$ norm.
Use the fixed-vector products (NCA76.4), $|D\xi|\le|\xi|$ and
the original fresh-noise identity there. This gives (NCA76.6).
In particular the local product involving $X_\theta$ is charged
by its exact full-jitter coefficient $X_r$, not by a compact support
or an averaged position factorization.
Section 5 provides the exact numerical upper certificate.
:::

:::{prf:theorem} A full noisy cap consumer for the own first spatial force
:label: thm-nca76-first-cap-force

Under {prf:ref}`def-nca76-register`,
$$
\|DB_1\|_2
\le\ell(C_P+3\kappa r_0)d_X\le1.54d_X,
\qquad \ell=e^{-1/2}.
\tag{NCA76.8}
$$
Also the entire first-spatial-force contribution to the final capped
velocity, with the actual noncommuting second count operator, obeys
$$
a\|D(cA_2-tbI)B_1\|_2
\le a[(z_v+ac)\ell(C_P+3\kappa r_0)
               +ac\kappa K_1]d_X
\le .00902d_X,
\tag{NCA76.9}
$$
where $K_1=2.76185$ is the certified bound of research77.
No $A_2,D$ commutation is assumed.
:::

:::{prf:proof}

For the independent prepared environment the exact first force is
$$
B_1=\mathbb E'[K(X-X')( (X-X')\cdot(r-r'))(P-P')].
$$
Apply the root's actual $D$ before estimating its norm. Since
$K(D_0)|D_0|\le\ell$, expanding only the two triangle products
gives
$$
|DB_1|\le\ell[
 |r||DP|+|r||D|_{\rm op}\mathbb E'|P'|
 +|DP|\mathbb E'|r'|
 +|D|_{\rm op}\mathbb E'(|r'||P'|)].
\tag{NCA76.10}
$$
Here the independent environment is the entire prepared root and its
differential, not a resampled velocity marginal. Population providers
are deterministic conditional on the comparison law, so the root's
$D$ can be applied outside that first-provider integral even though
$D$ depends on its own fresh OU and own second provider.

The environment quantities obey
$\mathbb E'|P'|\le r_0$,
$\mathbb E'|r'|\le d_X$ and
$\mathbb E'(|r'||P'|)\le r_0d_X$.
The latter is Cauchy--Schwarz inside the independent environment.
Taking the norm of (NCA76.10), (NCA76.6) controls its first local
term and (NCA76.4) controls each of the other three terms by
$\kappa r_0d_X$. This proves (NCA76.8).

For the complete first-force coefficient use
$cA_2-tbI=z_vI-acL_2$. The actual count formula gives
$$
|DL_2B_1|\le |DB_1|+|D|_{\rm op}\,\mathbb E|B_1|.
$$
The environment integral remains bounded by $\|B_1\|_2$ even
with its noisy conductances, because $0\le K\le1$ and $B_1$ was
fixed before the new OU innovations. Research77 gives
$\|B_1\|_2\le K_1d_X$. The root's conditional cap defect gives
$\||D|_{\rm op}\|_2\le\kappa$. Triangle substitution proves
(NCA76.9) with the noncommuting $L_2$ term retained. Every Gaussian
outcome appears in the norm.
:::

(sec-nca76-second-cap)=
## 3. A smaller full-tail second-force budget

:::{prf:theorem} Conditional cap defect inside the actual second consumer
:label: thm-nca76-second-cap-force

Use the exact quantities $D_y,A_y,d_Y,H_Y,\overline S$ of
{prf:ref}`thm-rfk-own-second-cap-consumer`:
$$
D_y=ba\ell(V_c+r_0),\quad A_y=a_x+D_y,\quad
d_Y=(a_x+2D_y)d_X+bd_P,
$$
$$
H_Y=A_yX_rd_X+b(1+a)X_2d_P+D_yX_2d_X,\qquad
\overline S<.58.
$$
The complete own second spatial force now satisfies
$$
a\|DB_2\|_2
\le a\ell[(2\kappa M_w+S_0+\overline S)d_Y+S_xH_Y]
\le .00885d_X+.000344d_P.
\tag{NCA76.11}
$$
This is the population source-plan statement. It does not substitute
$M_w=.70$ in an actual random finite provider product.
:::

:::{prf:proof}

The full position differential $R$ in (NCA76.3) is fixed before
the new OU innovations, even though the second graph is not.
The actual independent-environment count formula, acted on by $D$,
gives the same pointwise bound as (RFK.14) while retaining its
operator factors:
$$
|DB_2|\le\ell[
 (M_w|D|_{\rm op}+S_w)|R|
 +(M_w|D|_{\rm op}+S_w)d_Y],
\qquad S_w\le S_0+S_x|X_\theta|.
\tag{NCA76.12}
$$
For clarity its four terms arise from the exact product
$(R-R')(w-w')$:
the local $Dw$ is bounded by $S_w$;
the independent environment first moment is at most $M_w$;
$\mathbb E'|R'|\le d_Y$;
and $\mathbb E'(|R'||w'|)\le d_YM_w$.
The last bound uses $\|w'\|_2\le M_w$ and
$\|R'\|_2\le d_Y$ by Cauchy--Schwarz inside the actual
environment, with the complete noisy joint stage law retained.

The prior full-jitter source proof gives
$\|R\|_2\le d_Y$ and $\||X_\theta||R|\|_2\le H_Y$.
Consequently
$$
\|S_w|R|\|_2\le S_0d_Y+S_xH_Y,\qquad
\|S_w\|_2d_Y\le\overline S\,d_Y.
$$
For each preparation $R$ is pre-OU fixed, so (GCA74.12) gives
$$
\||D|_{\rm op}|R|\|_2\le\kappa d_Y,\qquad
\||D|_{\rm op}\|_2d_Y\le\kappa d_Y.
$$
These are conditional fixed-vector integrations, not products of
unconditional averages. Applying them to the two $M_w$ terms of
(NCA76.12) gives (NCA76.11).
The remaining source products and every OU outcome are unchanged.
Section 5 proves the displayed rational coefficients.
:::

(sec-nca76-complete-forcing)=
## 4. Complete spatial residual and the unclosed signed absorption

:::{prf:corollary} Both own spatial forces with the principal kept intact
:label: cor-nca76-complete-spatial-residual

At the actual comparison point define the principal physical
differentials using the same realized own count operators and cap:
$$
R_{\rm pr}=a_xr+bA_1p,\qquad
W_{\rm pr}=c(A_1p-tr),\qquad
C_{\rm pr}=D(A_2W_{\rm pr}-tR_{\rm pr}).
$$
Then the complete actual physical differentials satisfy
$$
R=R_{\rm pr}+baB_1,\qquad
C=C_{\rm pr}+aD(cA_2-tbI)B_1+aDB_2,
$$
$$
\|R-R_{\rm pr}\|_2\le .000650d_X,\qquad
\|C-C_{\rm pr}\|_2\le .01787d_X+.000344d_P.
\tag{NCA76.13}
$$
The entire $B_2$ in this formula uses the full $R$, not its
principal replacement.
:::

:::{prf:proof}

Insert $E=A_1p+aB_1$ into the exact harmonic and both-count
differentials (RFK.5). The first spatial increment to $R$ is
$baB_1$. Its increment to the uncapped velocity is
$a(cA_2-tbI)B_1$ before the actual $D$, while the complete
second spatial increment is $aB_2$.
This proves the exact decomposition without commuting any
operators or freezing either own provider.

Research77 and $b<.039216$ give
$baK_1<.000650$.
The two proved capped consumers (NCA76.9) and (NCA76.11)
then give the second bound. This is a force residual relative to
the actual realized principal; it is not an independent-noise
principal kernel.
:::

:::{prf:remark} Remaining signed terms after the improved consumers
:label: rem-nca76-remaining-signed

The complete general phase identity of research68 remains
$$
\mathbb E Q_\beta(R,C)-\mathbb E Q_\beta(r,p)
=-\mathfrak D_H+\mathfrak J_1+\mathfrak J_2-\mathfrak C.
$$
The new consumers (NCA76.8)--(NCA76.13) preserve the radial
velocity-force cancellation that was lost by applying only the
uncapped first-force norm. They are valid for genuinely noisy
prepared velocity laws satisfying the post-burn moment hypothesis,
without a narrow velocity band, favorable signed feedback, or
discarded jitter/OU outcomes.

These improved absolute residuals do not alone prove that the
complete signed sum is negative. In particular the physical
principal's anisotropic losses must consume its actual signed
cross products with the two residuals, and the correlated
$(I-D^2)Z_b$/$F_2$ term in (GCA74.15) remains part of the
same account. Replacing that term by a product with an average
cap derivative would again be invalid. No passing general
phase margin is asserted in this record.

For pre-OU conditioning, the exact native-cap mean matrices also obey
$$
A_c=\mathbb E_\xi D,\qquad B_c=\mathbb E_\xi D^2,\qquad
A_c^2\le B_c\le A_c,\qquad B_c\le\frac{159}{200}I.
\tag{NCA76.14}
$$
The middle inequality follows from the pointwise sector $D^2\le D$;
the other two are the conditional variance and deficit statements
of research74. These are joint matrix constraints for fixed test
vectors. They do not permit scalarization of a noisy force or
establish a lower mean-matrix bound on arbitrary jitter tails.

The original source-plan coupling supplies the input to these
kinetic estimates. Actual changes in accepted components,
sampled normalizers, mandatory revival and the terminal-alive
normalizations are still required for the full default law.
For a single law, a nonnegative raw squared-force observable
restricted to an own event of probability $p>0$ is bounded
above by its raw expectation divided by $p$. For a deficit
use instead the exact weighted loss formula (GCA74.17).
These elementary one-law statements do not couple the two
separately normalized alive laws or remove their own terminal
mismatch charges.

Finite versions require actual conditional provider bounds and
the mixed source/array products already stated in research38,41.
The population constants here have not been inserted into those
random products. No default nonlinear mixing rate, source-boundary
invariance or $N$-uniform exact QSD rate is concluded.
:::

(sec-nca76-certificate)=
## 5. Exact rational certificate

The following checks every numerical coefficient in the preceding
population consumers. All floats used in the formal upper estimates
are terminating rational constants.

```python
from fractions import Fraction as F

a, t = F(".006"), F(".02")
c, b = F(".9608"), F(".039216")
ell, kappa = F(".607"), F(".892")
r0, Mw = F(".55"), F(".70")
Xr, X2 = F("4.478"), F("3.47")
V, Vc = F(2), F(4)
A = F(".9936")
zv_lower = F(".96") - t * b
denominator = zv_lower * (1 - a)
S0 = (V / 4 + a * Mw + t * t * Vc) / A
Sx = t / A

Cp = (
    V / 4 + b * kappa * Xr + F(".198") * F("1.733")
    + a * kappa * (c * r0 + Mw)
    + a * (S0 + Sx * Xr)
) / denominator
first_capped = ell * (Cp + 3 * kappa * r0)
K1 = F("2.76185")
first_complete = a * (
    (c + a * c) * first_capped + a * c * kappa * K1
)
assert Cp < F("1.06")
assert first_capped < F("1.54")
assert first_complete < F(".00902")
assert b * a * K1 < F(".000650")

Dy = b * a * ell * (Vc + r0)
Ay = F(".999216") + Dy
dYx, dYp = F(".999216") + 2 * Dy, b
HYx = Ay * Xr + Dy * X2
HYp = b * (1 + a) * X2
consumer = 2 * kappa * Mw + S0 + F(".58")
second_x = a * ell * (consumer * dYx + Sx * HYx)
second_p = a * ell * (consumer * dYp + Sx * HYp)
assert second_x < F(".00885")
assert second_p < F(".000344")
assert first_complete + second_x < F(".01787")

print("All full-tail first/second cap consumer comparisons pass.")
```
