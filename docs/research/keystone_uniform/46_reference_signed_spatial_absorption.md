# A proved source-plan class for reference spatial-force absorption

(sec-rsa-record)=
## 1. Retained default kernel and the restricted comparison class

:::{prf:definition} Physical kinetic transport and centered source displacements
:label: def-rsa-class

Retain every default harmonic count-kinetic primitive of
{prf:ref}`def-rfk-retained`: $h=.04$, $\nu=.3$, $d=3$,
$L=2$, $V=2$, $V_c=4$, $\rho=\gamma=b_O=1$,
$\sigma_J=\sigma_x=.1$, $t=.02$, $c=e^{-.04}$,
$b=t(1+c)$, $a_x=1-tb$, $a=t\nu=.006$, and $\beta=.04$.
Write $K_{\rm ph}\lambda$ for the physical phase projection of
the actual complete population kinetic output of a prepared law
$\lambda$. Both own count providers, the actual joint second
provider, full independent OU/final Gaussians, cap and terminal
classification are retained. The projection records $(x,v)$;
the discrete terminal mark is not part of the phase ground quadratic
$$
Q_\beta(r,p)=|r|^2+2\beta r\cdot p+|p|^2.
$$
Let $W_{2,G}$ be Wasserstein transport for this positive-definite
quadratic, $G=\left(\begin{smallmatrix}1&\beta\\\beta&1\end{smallmatrix}\right)
\otimes I_d$.

Two actual prepared population laws admit a source-plan coupling
if their complete plans, including the original-slot Haar readout,
are coupled before a shared fresh $J\sim N(0,\sigma_J^2I_d)$,
independent of both plans, and
$$
X_j=S_j+I_jJ,\qquad S_j\in[-2,2]^3,\quad I_j\in\{0,1\},
\quad |P_j|\le V_c,\quad \|P_j\|_2\le r_0=.55.
\tag{RSA.1}
$$
Set $r=X_1-X_0$, $p=P_1-P_0$ and
$$
d_X=\|r\|_2,\quad d_P=\|p\|_2,\qquad
r_c=r-\mathbb Er,\quad p_c=p-\mathbb Ep,\qquad
\sigma_X=\|r_c\|_2,\quad\sigma_P=\|p_c\|_2.
$$
The proved class in this note consists of pairs admitting such a
coupling with
$$
\sigma_X\le\varepsilon_0d_X,\qquad
\sigma_P\le\varepsilon_0d_P,\qquad \varepsilon_0=1/128.
\tag{RSA.2}
$$
Thus the comparison displacements are mostly their common means.
This is a condition on a particular valid coupling, not an assertion
about every optimal transport or every swarm. The post-burn velocity
bound can be inherited from {prf:ref}`thm-rvb-population-burn`, or
verified directly on the two actual preparations. It is not used
as an individual-speed bound. No invariance of (RSA.2) under subsequent
preparation or kinetic updates is assumed.
:::

(sec-rsa-centered)=
## 2. Spatial forces depend on centered displacements

:::{prf:lemma} Centered full-jitter spatial-force bounds
:label: lem-rsa-centered-forces

Interpolate the coupled preparations by $X_\theta=X_0+\theta r$,
$P_\theta=P_0+\theta p$ and retain the same full OU/final
innovations across the interpolation. Use the actual lifted
operators and spatial forces $A_1,A_2,B_1,B_2,D$ of
{prf:ref}`lem-rfk-own-differential`, with $D=DC_V(z_\theta)$.
At every $\theta\in[0,1]$,
$$
\|B_1\|_2\le2\ell(V_c+r_0)\sigma_X,\qquad
a\|DB_2\|_2\le.0094\sigma_X+.000366\sigma_P,
\quad \ell=e^{-1/2}.
\tag{RSA.3}
$$
The proof retains all source, velocity and graph/noise correlations.
There is no tail event or additive bad-jitter floor.
:::

:::{prf:proof}
The derivative of the first pair kernel uses $r-r'=r_c-r_c'$.
Hence its force is unchanged when $r$ is replaced by $r_c$.
The pointwise estimate of the actual pair integral gives
$$
|B_1|\le\ell(V_c+r_0)(|r_c|+\sigma_X).
$$
Only its independent environment product is bounded by
$\mathbb E'|r_c'||P_\theta'|\le\sigma_Xr_0$.
Taking the root $L^2$ norm proves the first part of (RSA.3).
The local source/velocity product has not been replaced by an
unconditional RMS times a source-displacement moment.

Both lifted count Laplacians annihilate constants. Their alignment
operators $A_j=I-aL_j$ preserve constants and the mean. Pair
antisymmetry gives $\mathbb EB_1=0$. Consequently, if
$R=\dot y_\theta$, then
$$
\mathbb ER=a_x\mathbb Er+b\mathbb Ep,\qquad
R_c=R-\mathbb ER=a_xr_c+b(A_1p_c+aB_1).
\tag{RSA.4}
$$
The second pair-kernel derivative likewise uses
$R-R'=R_c-R_c'$, so $B_2$ is unchanged by centering $R$.

The source-plan structure persists under centering:
$$
r_c=(S_1-S_0-\mathbb E[S_1-S_0])+(I_1-I_0)J.
$$
The centered plan displacement and $p_c$ are fixed before fresh
jitter. The exact Gaussian polynomial of
{prf:ref}`lem-rfk-source-product` therefore gives
$\||X_\theta||r_c|\|_2\le\sqrt{20.05}\sigma_X$ and
$\||X_\theta||p_c|\|_2\le X_2\sigma_P$,
$X_2=\sqrt{12.03}$. The nonlocal pointwise term of $A_1p_c$
is bounded by $a\sigma_P$ and is retained.

Apply the full proof of
{prf:ref}`thm-rfk-own-second-cap-consumer` with
$(r_c,p_c,R_c,\sigma_X,\sigma_P)$ in its displacement terms.
Its actual root variables, own providers, cap and source positions
are unchanged. In particular $\|P_\theta\|_2\le r_0$ by
Minkowski, $\|U_\theta\|_2\le r_0$ by its own count operator,
and the actual joint OU moment remains at most $M_w=.70$.
Its radial root factor retains the product $S_w|R_c|$ until the
conditional source polynomial bounds it. The independent environment
term alone uses $\mathbb E'|R_c'||w'|\le\|R_c\|_2M_w$.
Thus the same rational coefficients in (RFK.14) give the second
part of (RSA.3). Every original Gaussian draw has been integrated.
:::

:::{prf:lemma} Principal plus centered-force decomposition
:label: lem-rsa-force-decomposition

Put
$$
\kappa_X=.00149,\qquad\kappa_P=.0721,\qquad
\Delta_Q=\kappa_Xd_X^2+\kappa_Pd_P^2,\qquad
Q=\mathbb E Q_\beta(r,p).
$$
The derivative of the complete actual physical output admits a
principal part $H_\theta$ and a force part $F_\theta$ in the
$L^2(G)$ Hilbert space, with
$$
\|H_\theta\|_G^2\le Q-\Delta_Q,\qquad
\|F_\theta\|_G\le.0414\sigma_X+.000366\sigma_P.
\tag{RSA.5}
$$
:::

:::{prf:proof}
With the actual operators at $\theta$, define
$$
R_0=a_xr+bA_1p,\qquad W_0=c(A_1p-tr),\qquad
Z_0=A_2W_0-tR_0,\qquad H_\theta=(R_0,DZ_0).
$$
The anisotropic complete-principal theorem
{prf:ref}`thm-rfk-two-count-principal` gives the first bound
without commutation or graph/OU independence. The exact full
differential of {prf:ref}`lem-rfk-own-differential` then gives
$$
F_\theta=(baB_1,\ aD(cA_2-tbI)B_1+aDB_2).
\tag{RSA.6}
$$
Here $B_2$ is the actual force computed using the complete $R$,
including the first spatial force; it has not been evaluated at
the principal $R_0$.

The self-adjoint operator $cA_2-tbI$ has eigenvalues between
$c(1-a)-tb>0$ and $v_H=c-tb$. Thus its norm is at most $v_H$.
Since $\|D\|\le1$, the first-force contribution in the phase
quadratic has norm at most
$$
aC_1\|B_1\|_2,\qquad
C_1=\sqrt{b^2+2\beta bv_H+v_H^2}<.963.
$$
This uses Cauchy--Schwarz on the cross term of $Q_\beta$ and
does not require $D$ to commute with $A_2$. The second-force
contribution has zero position component, so its $G$ norm is
exactly $a\|DB_2\|_2$.
By (RSA.3), the total is at most
$[2aC_1\ell(V_c+r_0)+.0094]\sigma_X+.000366\sigma_P$.
The first bracket is strictly below $.0414$: the entirely rational
upper calculation
$$
2(.006)(.963)(.607)(4.55)=.0319159386<.032
$$
uses $\ell<.607$; $C_1<.963$ follows by squaring with
$b<.039216$ and $v_H<.960016$. This proves (RSA.5).
:::

(sec-rsa-absorption)=
## 3. Complete absorption on the stated class

:::{prf:theorem} A complete default kinetic coupling estimate with nonzero spatial forces
:label: thm-rsa-class-absorption

For every source-plan coupling satisfying (RSA.1)--(RSA.2), the
actual complete population physical kinetic outputs have a valid
coupling such that
$$
\mathbb E Q_\beta(x_1^+-x_0^+,v_1^+-v_0^+)
\le\left(1-\frac{149}{208000}\right)
                       \mathbb E Q_\beta(X_1-X_0,P_1-P_0).
\tag{RSA.7}
$$
This estimate includes both spatial graph forces, both count
alignment operators and the native cap at $\nu=.3$. It holds
for uncut recipient jitter and both full kinetic Gaussians.
:::

:::{prf:proof}
Let
$$
\delta_0=\frac{\kappa_X}{1+\beta}=\frac{149}{104000},\qquad
\Lambda^2=\frac{.0414^2}{\kappa_X}
                         +\frac{.000366^2}{\kappa_P}<1.074^2.
$$
Since $Q\le(1+\beta)(d_X^2+d_P^2)$,
$\Delta_Q\ge\delta_0Q$. Weighted Cauchy--Schwarz, (RSA.2)
and (RSA.5) give
$\|F_\theta\|_G\le\varepsilon_0\Lambda\sqrt{\Delta_Q}$.
The principal part has norm at most $\sqrt{Q-\Delta_Q}$.
Consequently
$$
\begin{split}
\|H_\theta+F_\theta\|_G^2
\le Q-\Delta_Q+
\left[\frac{2\varepsilon_0\Lambda}{\sqrt{\delta_0}}
                  +\varepsilon_0^2\Lambda^2\right]\Delta_Q.
\end{split}
$$
The bracket is strictly below $1/2$. An exact rational check uses
$\sqrt{\delta_0}>.03785$ and $\Lambda<1.074$:
$$
\frac{2(1.074)}{128(.03785)}+\frac{1.074^2}{128^2}<\frac12.
\tag{RSA.8}
$$
Thus every interpolation derivative has phase-square norm at most
$Q-\Delta_Q/2\le(1-\delta_0/2)Q$.

Share the actual own OU and final Gaussian innovations between
the endpoint kernels, independently of the source plans. The
preceding derivative bounds integrate in $L^2$: the source-box
Gaussian moments and bounded kernel derivatives dominate all
appearing products, as in the full signed differential proof.
The endpoint output difference is the integral of these complete
physical derivatives. Jensen for the positive phase quadratic
gives (RSA.7). Each own marginal retains its original independent
innovations and own actual providers. This is an upper bound from
a valid actual coupling; no positive signed margin has been presumed.
:::

:::{prf:corollary} Optimal physical-phase population transport on this class
:label: cor-rsa-optimal-class-transport

For any pair of actual prepared laws admitting the coupling in
{prf:ref}`thm-rsa-class-absorption`,
$$
W_{2,G}(K_{\rm ph}\lambda_0,K_{\rm ph}\lambda_1)^2
\le\left(1-\frac{149}{312000}\right)
                            W_{2,G}(\lambda_0,\lambda_1)^2.
\tag{RSA.9}
$$
The source-plan coupling need not be an optimal input transport.
The conversion uses its centered-variance condition and the correct
mean lower bound on the optimal input cost.
:::

:::{prf:proof}
Put $\bar r=\mathbb Er$, $\bar p=\mathbb Ep$ and
$Q_{\rm mean}=Q_\beta(\bar r,\bar p)$.
The centered orthogonal decomposition gives
$Q=Q_{\rm mean}+\mathbb EQ_\beta(r_c,p_c)$.
From (RSA.2),
$$
\sigma_X^2+\sigma_P^2
\le\frac{\varepsilon_0^2}{1-\varepsilon_0^2}
                         (|\bar r|^2+|\bar p|^2).
$$
Thus
$$
Q\le(1+\chi)Q_{\rm mean},\qquad
\chi=\frac{1+\beta}{1-\beta}
                        \frac{\varepsilon_0^2}{1-\varepsilon_0^2}.
$$
For every input coupling, Jensen in the positive phase quadratic
gives $W_{2,G}(\lambda_0,\lambda_1)^2\ge Q_{\rm mean}$;
the mean difference is determined by its two marginals. The valid
output coupling in (RSA.7) therefore gives
$$
W_{2,G}(K_{\rm ph}\lambda_0,K_{\rm ph}\lambda_1)^2
\le(1-\delta_0/2)(1+\chi)
                            W_{2,G}(\lambda_0,\lambda_1)^2.
$$
At $\beta=1/25$, $\varepsilon_0=1/128$ and
$\delta_0=149/104000$, the exact rational inequality
$(1-\delta_0/2)(1+\chi)<1-\delta_0/3$ proves (RSA.9).
If the mean difference vanishes, (RSA.2) forces both centered
differences to vanish as well, so the two laws are identical and
the assertion remains valid. The argument does not infer an
optimal-transport lower bound from a chosen-coupling cost.
:::

(sec-rsa-nonempty)=
## 4. A nonempty class of actual active preparations

:::{prf:corollary} Narrow alive clouds at fixed configured fitness parameters
:label: cor-rsa-narrow-cloud-class

Keep any fixed positive original reward/diversity fitness exponents,
positive logistic floors/amplitudes, standardizer floors and gate
parameters. Let $x_0,x_1$ be distinct points in the interior of
$[-2,2]^3$, and let $u_0,u_1$ be constant original velocities
with $|u_j|\le.55$. Put $s_0=|x_1-x_0|>0$ and
$\eta_D=\min_j(2-|x_j|_\infty)>0$.
Let $L_R=\sqrt{12}$ and $\ell_f$ be the actual feature-map
Lipschitz bound. With
$$
F_*=\eta_r^{p_r}\eta_s^{p_s},\qquad
H_b=(\eta_{b'}+A_{b'})^{p_{b'}}\frac{A_bp_b}{4}
 \max\{\eta_b^{p_b-1},(\eta_b+A_b)^{p_b-1}\},
$$
$$
C_{\rm ball}=\frac{2}{s_c(F_*+\epsilon_c)}
             \left(\frac{H_rL_R}{\sigma_r}
                         +\frac{H_s\ell_f}{\sigma_s}\right)>0,
\tag{RSA.10}
$$
choose any positive radius
$$
\epsilon\le\epsilon_*:=\frac12\min\left\{
\eta_D,\frac{s_0}{4},\frac{\varepsilon_0s_0}{8},
\frac{\varepsilon_0^2s_0^2}{16d\sigma_J^2C_{\rm ball}},
\frac{\kappa_C}{8C_{\rm ball}}\right\}>0.
\tag{RSA.11}
$$
For any all-alive incoming population laws supported on
$B(x_j,\epsilon)\times\{u_j\}$, their actual complete prepared
laws admit (RSA.1)--(RSA.2). Hence their physical kinetic images
satisfy (RSA.7) and the optimal prepared-to-output transport
contraction (RSA.9). The configured algorithm, exponents and
normalizers have not been changed. The class includes nontrivial
profiles with positive accepted cloning, not only point consensus.
:::

:::{prf:proof}
All incoming roots are alive. Every actual frozen source, persistent
or copied, belongs to its own ball. Constant original velocity
$u_j$ makes the component mean $u_j$ and every original deviation
zero; the actual complete Haar readout therefore gives $P_j=u_j$
for every component. Both prepared moment hypotheses in (RSA.1)
hold directly.

On one such cloud, the raw harmonic reward range is at most
$2L_R\epsilon$. The actual nonnegative measured diversity range
is at most $2\ell_f\epsilon$, since every own and eligible
companion phase lies in that cloud with the same velocity. For
any realization of all measurement marks, their shared regularized
standardizers are at least $\sigma_r,\sigma_s$. Thus the
within-cloud standardized ranges are at most
$2L_R\epsilon/\sigma_r$ and $2\ell_f\epsilon/\sigma_s$.
The actual logistic/power derivative bounds are exactly $H_r,H_s$.
The original clipped gate consequently gives acceptance at most
$C_{\rm ball}\epsilon$ for every alive root. The same range
calculation holds in the actual rooted population law. In
particular its accepted incoming column is at most
$C_{\rm ball}\epsilon/\kappa_C\le1/8$, so its required
accepted-arm finite-component condition is satisfied. No actual
fitness statistic is replaced by a target statistic.

Couple the two complete source/component plans by any coupling,
and then use a common fresh $J$ independent of both. There is no
mandatory revival at these all-alive inputs. Each copy indicator
has probability at most $C_{\rm ball}\epsilon$, so
$\mathbb E(I_1-I_0)^2\le2C_{\rm ball}\epsilon$.
The source displacement differs from $x_1-x_0$ by at most
$2\epsilon$, hence
$$
\sigma_X^2\le4\epsilon^2+
                         2d\sigma_J^2C_{\rm ball}\epsilon,
\qquad d_X\ge s_0-2\epsilon\ge s_0/2.
$$
The radius bounds in (RSA.11), even without their prefactor $1/2$,
make the first right-hand term at most
$\varepsilon_0^2s_0^2/16$ and the second at most
$\varepsilon_0^2s_0^2/8$. Their sum is at most
$\varepsilon_0^2s_0^2/4\le\varepsilon_0^2d_X^2$.
The velocity difference is constant, so $\sigma_P=0$.
This proves (RSA.2), with the full Gaussian jitter uncut.

For a concrete nonzero-cloning profile take each cloud to contain
two distinct radial positions with unequal harmonic rewards and
equal positive masses. The two cross-point measurement choices
have the same symmetric diversity distance and positive probability
under the actual positive role weights. Two such measured types with
unequal rewards have equal diversity factors and strictly unequal
reward factors because $p_r>0$. A positive-probability donor draw
and gate then gives positive accepted cloning. These same two-type
profiles can be placed inside the stated positive-radius balls,
for example about $x_0=-e_1/2$, $x_1=e_1/2$. Their source laws
are not point masses and their copy jitters have positive mass.
The proof already accounts for those source and indicator differences.
:::

(sec-rsa-scope)=
## 5. Last justified endpoint

:::{prf:remark} Restricted kinetic closure and remaining full-law obligations
:label: rem-rsa-scope

The reference default viscosity admits the proved one-update kinetic
absorption (RSA.7) and optimal physical-phase transport estimate
(RSA.9) on the explicit nonempty source-plan class (RSA.2).
Both spatial graph forces are included. All local source/velocity
products remain controlled by the actual source polynomial and
individual $V_c$ bound where needed; the proof never substitutes
$r_0^2\mathbb E|\delta S|^2$ for
$\mathbb E[|\delta S|^2|P_\theta|^2]$.

The class condition is substantially stronger than the post-burn
velocity moment. A coupling with a small common mean difference
and substantial centered shape change need not satisfy it. This
note proves no absorption for that remaining part of (RFK.6),
and no invariance of the proved class over repeated updates.
It gives no comparison of the preparation cost to the incoming
swarm cost and no contraction in a metric containing terminal
dead marks or a conditional-alive normalization.

Thus no global default active convergence, finite QSD mixing or
default marked population attraction is inferred. The exact new
endpoint is a proved nonempty one-step physical population kinetic
law class. The root-conditioned population estimates used here
do not become empirical finite-array estimates by replacing a
random empirical velocity moment with its averaged RMS.
:::
