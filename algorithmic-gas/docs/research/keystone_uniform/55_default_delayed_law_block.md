# Default marked-law feedback and an exact delayed response formula

This note proves a quantitative response formula for the actual harmonic
count-population map at the default step, viscosity and terminal box.
It includes the complete dead population in the preparation and keeps
the actual alive-only reward and diversity normalizers. Its final
absolute feedback bound does not establish a nonlinear contraction
coefficient at these defaults. The signed delayed estimate needed to
improve that bound is identified at the end.

(sec-dlb-register)=
## 1. Retained kernel, norm and a closed output class

:::{prf:definition} Default delayed law register
:label: def-dlb-register

Retain {prf:ref}`def-dmc-register`: $d=3$, $h=.04$, $\nu=.3$,
$L=2$, $V=2$, $V_c=4$, $\rho=\gamma=b_O=1$,
$\sigma_J=\sigma_x=.1$ and $\alpha_{\rm col}=.5$.
The force and raw reward are exactly $F(x)=-x$ and $R(x)=-|x|^2/2$.
Keep the actual current-frame measured fitness, global alive-only
normalizers, eligible donor rows, frozen positional sources, mandatory
revival, full component-Haar collisions on original velocities, recipient
jitters, both count kicks, actual joint OU provider, native cap and
terminal classification. Historical, curl, elite and geometry branches
are disabled in this restriction.

Use the positive exponent interval $0<\theta\le\theta_f$ of
(DMC.3). In particular the actual alive gate and accepted incoming
column obey $a_*\le\theta a_0$ and $c_*\le1/8$.
No assertion that the configured unit powers satisfy this interval is
made. Mandatory dead-root gates remain one.

Set
$$
t=.02,\quad c=e^{-.04},\quad b=t(1+c),\quad m=1-t^2,
\quad a_x=1-tb,\quad q^2=(1-c^2)/2,\quad s^2=.0004,
$$
$$
\tau^2=t^2q^2+s^2,\qquad
a_{\rm ret}>0\ \text{as in (DSA.3)},\qquad
m_0=a_{\rm ret}/2,\quad \bar e=1-m_0.
\tag{DLB.1}
$$
For a finite signed measure use full variation
$\|\xi\|_1=\int d|\xi|$ and weighted full variation
$\|\xi\|_{W_p}=\int(1+|x|^p)d|\xi|$.
Thus $\operatorname{TV}(\mu,\mu')=\|\mu-\mu'\|_1/2$ for
probabilities. For an alive measure write
$\alpha_\mu=\mu_A/\mu_A1$.

The actual population map is $\mathcal F\mu=\mu P_\mu$, where
$P_\mu=J_\mu K[\lambda_\mu,\Lambda_{2,\mu}]$ is the full frozen
root kernel, $\lambda_\mu=\mu J_\mu$ is its actual preparation and
$\Lambda_{2,\mu}$ its actual joint noisy stage. This is a frozen
kernel representation of the nonlinear map; providers at different
population laws remain different.
:::

:::{prf:lemma} Uniform source-box moment and Gaussian closure
:label: lem-dlb-output-class

For a standard $d$-Gaussian put $G_p=\mathbb E|Z|^p$ and define
$$
H_{\rm box}=
\left[a_x\sqrt dL+bV_c+
 \sqrt{a_x^2\sigma_J^2+\tau^2}\,G_8^{1/8}\right]^8,
\qquad
H_{{\rm p},5}=\left[\sqrt dL+\sigma_JG_5^{1/5}\right]^5.
\tag{DLB.2}
$$
Every actual complete population output from a consistent capped
positive-alive input belongs to
$$
\begin{split}
\mathfrak C=\{\mu:\ &(x,v,a)=(Y+sZ,v,\mathbf1_D(Y+sZ)),\\
&Z\text{ independent of }(Y,v),\quad |v|\le V,\quad
\mu|x|^8\le H_{\rm box},\quad \mu_A1\ge m_0\}.
\end{split}
\tag{DLB.3}
$$
Every actual prepared law, including a common-root preparation under
a different consistent environment, has speed at most $V_c$,
fifth position moment at most $H_{{\rm p},5}$, and conditional
fourth-weight readout at most
$$
C_{\rm src}=D_JB_L,\qquad
D_J=1+(d+2)\sigma_J^2+d(d+2)\sigma_J^4,\quad
B_L=1+d^2L^4=145.
\tag{DLB.4}
$$
In particular $\mathfrak C$ is preserved, and entry occurs after
one actual update without any moment condition on entering dead
coordinates. No small-dead bound is included in this class.
:::

:::{prf:proof}
Every actual persistent, copied or revived source belongs to $D$.
The component collision gives $|P|\le V_c$, and the first count
average $U$ is convex because $t\nu<1$. The exact landing position is
$$
x^+=a_xS+bU+a_xIJ+tq\xi+s\chi.
$$
Conditional on its source/component plan, $I$ is fixed before its
fresh jitter; $a_xIJ+tq\xi+s\chi$ is a centered Gaussian with
variance at most $a_x^2\sigma_J^2+\tau^2$ in each coordinate.
The bounded term $a_xS+bU$ may depend on the jitter. Its pointwise
bound and Minkowski still give (DLB.2). The last $\chi$ is independent
of everything used to compute the stored capped velocity and the
pre-final position $Y$. Hence the representation in (DLB.3) is exact.
The source-box safe-return lemma
{prf:ref}`thm-dsa-default-box-alive-floor` gives output alive mass
at least $a_{\rm ret}>m_0$.

For a prepared source $X=S+IJ$, Minkowski gives its fifth-moment
bound independently of all original retained dead positions.
For fixed $S,I$, the Gaussian fourth-moment identity and
$|S|^2\le(1+|S|^4)/2$ give
$\mathbb E(1+|S+IJ|^4)\le D_J(1+|S|^4)\le D_JB_L$.
This applies conditionally on every actual source/component outcome.
Only velocity uses the remaining component; its original-slot cap
bound is independent of that fourth-weight calculation. This proves
(DLB.4), including cross-environment roots.
:::

(sec-dlb-preparation)=
## 2. Full mandatory-dead feedback without a small-dead premise

:::{prf:definition} Raw preparation constants at the actual alive support
:label: def-dlb-preparation-constants

Set $H_A=(\sqrt dL)^8$, $B_A=1+\sqrt{H_A}=B_L$.
Evaluate $Q_x$, $C_{F,0}^Q$, $C_J^Q$ and $L_J^Q=B_AC_J^Q$
from (RQF.1)--(RQF.3), (RQF.6)--(RQF.8) at $H_8=H_A$.
Those are explicit raw logistic-tail and ordered-forest constants;
no bounded replacement reward is used. Retain their positive
role floors and exact standardizer parameters. In particular
$$
G=e^{1/4},\qquad
C_*=64D_JB_A^2G(1+\kappa_D^{-1})(1+\kappa_C^{-2}),
$$
$$
h_A=(\kappa_C^2m_0)^{-1},\qquad
h_D=(\kappa_Cm_0)^{-1}+\bar e(\kappa_Cm_0^2)^{-1},
$$
$$
\begin{aligned}
C_A&=L_J^Q+C_*[1/(\kappa_Cm_0)
                         +(1+\bar e)h_A+1+L_J^Q],\\
C_D&=C_*(1+\bar e)h_D,\\
A_J&=\frac{2B_L}{m_0}C_A(\theta+\bar e)+C_D.
\end{aligned}
\tag{DLB.5}
$$
All moments of a conditional alive input are bounded by $H_A$,
since its actual coordinates lie in $D$. Dead positions remain
unrestricted before the first update.
:::

:::{prf:lemma} Complete common-root preparation and own-preparation comparison
:label: lem-dlb-full-dead-preparation

Let $\mu,\mu'\in\mathfrak C$ and put $\delta=\|\mu-\mu'\|_1$.
For any common consistent root law $\zeta\in\mathfrak C$,
$$
\|\zeta J_\mu-\zeta J_{\mu'}\|_{W_4}\le A_J\delta.
\tag{DLB.6}
$$
For the actual own preparations,
$$
\|\lambda_\mu-\lambda_{\mu'}\|_{W_4}
\le(C_{\rm src}+A_J)\delta.
\tag{DLB.7}
$$
These estimates retain the full mandatory-dead population, original
dead velocities and changes in the actual alive-only reward means
and variances. The floor $m_0$ supplies a finite denominator; it
does not assert that $\bar e$ is small.
:::

:::{prf:proof}
First subtract the two normalized alive probabilities. Since
$W_4\le B_L$ on the alive support,
$$
\|\alpha_\mu-\alpha_{\mu'}\|_{W_4}
\le\frac{B_L}{m_0}
 [\|\mu_A-\mu'_A\|_1+|\mu_A1-\mu'_A1|]
\le\frac{2B_L}{m_0}\delta.
$$
Also $\|\mu_D-\mu'_D\|_1\le\delta$.
Every conditional alive law has eighth moment at most $H_A$,
so the raw normalization and tagged-provider proof
{prf:ref}`thm-rqf-preparation-feedback` applies to its actual reward,
not to a substituted statistic.

For completeness, the marked extension of that provider proof has
no need for a small-dead hypothesis in this preparation comparison.
Expose the accepted alive forest first and attach the actual mandatory
dead children afterward. The alive column ceiling $c_*\le1/8$
bounds its expected alive vertex count by $G$, or $2G$ after one
known edge. Dead vertices have consumed their outgoing edge and
cannot be donors; they add leaves, with mean at most
$\bar e/(\kappa_Cm_0)$ per alive vertex. That finite intensity
may exceed one and is retained in the constants.

The exact dead-leaf intensity at an alive target has denominator
$m_AZ_C(\alpha_\mu;z)$, with $Z_C\ge\kappa_C$.
Subtracting this denominator and the dead measure gives the failure
hazard
$$
H_D\le\bar e h_A\delta_A+h_D\delta_D,
\quad \delta_A=\|\alpha_\mu-\alpha_{\mu'}\|_{W_4},\quad
\delta_D=\|\mu_D-\mu'_D\|_1.
$$
Expose each root source before its remaining component. For an alive
root, alive-forest differences are bounded by $\theta L_J^Q\delta_A$.
On the isolated-alive-forest branch, a root-mark discrepancy has an
additional effect only when the root receives a dead leaf. The
nontrivial alive-forest branch is already included in that first charge.
Source-first dead-leaf and root-mark comparisons therefore add at most
$C_*[\bar e\delta_A/(\kappa_Cm_0)+H_D]$.
For a common dead root, the mandatory outgoing alive donor is exposed
first. Its normalized donor law and remaining component contribute
at most $C_*[(1+\theta L_J^Q)\delta_A+H_D]$ per unit dead-root
mass; this is averaged against a mass at most $\bar e$.

These are precisely the complete source-first charges in
{prf:ref}`lem-rkpf-raw-interfaces`, with $\epsilon_0$ replaced by
$\bar e$ and $m_0$ retained. Only the alive forest uses the
subcritical estimate $c_*\le1/8$. The displayed dead-leaf hazards
are upper bounds even when larger than one. No other step in those
charges uses $\epsilon_0\le1/4$; that restriction belongs to the
later small-dead drift and endpoint argument. Summing gives
$C_A(\theta+\bar e)\delta_A+C_D\delta_D$ and then (DLB.6).
The actual readout keeps original-slot collision velocities and its
own conditional source law throughout.

Finally write
$$
\mu J_\mu-\mu'J_{\mu'}
=(\mu-\mu')J_\mu+\mu'(J_\mu-J_{\mu'}).
$$
The first term is at most $C_{\rm src}\delta$ in fourth-weight
variation by the conditional readout bound (DLB.4); the second is
bounded by (DLB.6). This proves (DLB.7).
:::

(sec-dlb-gaussian)=
## 3. Complete Gaussian score feedback at the default viscosity

:::{prf:definition} Explicit BV and joint-score constants
:label: def-dlb-score-constants

Use the exact weighted trace $\mathcal T_5(L)$ of (KPF.5) with
moment $H_{\rm box}$ and final noise $s$. Put
$$
\ell_b=(\epsilon_b\sqrt e)^{-1},\quad
L_F=\theta[Q_x(H_A)+J_s/\sigma_s],\quad
L_D=2\ell_D/\kappa_D,
$$
$$
C_{\rm out}^{J}=2a_*\ell_C/\kappa_C+L_{\rm rec}L_F,
\quad
C_{\rm in}^{J}=
[a_*\ell_C+L_{\rm don}L_F+\bar e\ell_C/m_0]/\kappa_C,
$$
$$
C_r=L_D+C_{\rm out}^{J}+2C_{\rm in}^{J},\quad
r_J=a_*+\bar e,\quad
c_J=(a_*+\bar e)/(\kappa_Cm_0),\quad M_5=H_{\rm box}^{5/8},
$$
$$
\begin{aligned}
L_{G,5}&=\frac{\sqrt{2/\pi}}s(1+16M_5)+16s^4G_6,\\
L_{J,5}&=\frac{\sqrt{2/\pi}}{\sigma_J}(r_J+16c_JM_5)
                                      +16r_J\sigma_J^4G_6,\\
B_5^{\rm src}&=d[L_{G,5}+L_{J,5}+C_r(1+M_5)]
                                          +\mathcal T_5(L).
\end{aligned}
\tag{DLB.8}
$$
At $\nu=.3$ evaluate $M_*,S_x,S_w,L_I$ in (PVB.3) with
this $B_5^{\rm src}$ and prepared moment $H_{{\rm p},5}$.
These are explicit Gaussian moment, inverse-coordinate and score
constants. Evaluate $C_T,C_{\rm stage},C_{\rm out},C_{B2},C_{\rm fb}$
in (PVB.6)--(PVB.8) at the same viscosity. Their formulas are
$$
\begin{aligned}
C_T&=2V_c[t^2S_x+ctS_w+dt^2\ell_\rho L_IM_*],\\
C_{\rm stage}&=1+27[a_x^4+(ct)^4+(bV_c)^4+(cV_c)^4
                                      +((tq)^4+q^4)G_4],\\
C_{\rm out}&=1+8(1+s^4G_4),\\
C_{B2}&=\frac{C_{\rm out}}{1-t\nu}[dM_*+2(S_w+tS_x)],\\
C_{\rm fb}&=C_{\rm out}C_T+
                         C_{B2}(C_{\rm stage}+\nu C_T),\\
L_{\rm fb}&=A_J+\nu C_{\rm fb}(C_{\rm src}+A_J).
\end{aligned}
\tag{DLB.9}
$$
Every quantity here is a finite primitive expression. None is an
assumed Sobolev, conditional empirical-provider or signed-margin
constant.
:::

:::{prf:theorem} Actual full marked common-root defect bound
:label: thm-dlb-default-feedback

For $\mu,\mu'\in\mathfrak C$,
$$
\|\mu'(P_\mu-P_{\mu'})\|_1
\le L_{\rm fb}\|\mu-\mu'\|_1.
\tag{DLB.10}
$$
The left side includes the preparation change, both actual deterministic
count providers, full correlated OU stage, native cap, final Gaussian
and terminal mark. All mandatory revivals and full Gaussian tails are
included.
:::

:::{prf:proof}
The proof of the raw marked BV interface (RKPF.5) remains valid
with $\epsilon_0=\bar e$: its persistent-root incoming dead
intensity has the finite derivative displayed in (DLB.8), while copied
and revived roots use their independent recipient jitter. Its root
measurement, no-edge and incoming-intensity derivatives require only
the positive alive floor. The alive restriction contributes the full
boundary trace $\mathcal T_5(L)$, even if it is large. Thus the
actual own preparation $\lambda_{\mu'}$ satisfies the weighted,
velocity-fibre BV bound $B_5^{\rm src}$.

At the default parameters the inverse-score endpoint of
{prf:ref}`lem-pvb-joint-scores` is admissible. Indeed
$L_0=4dV_c\ell_\rho<48$ and
$$
\bar\nu=\min\{1,m/(2t),m/(2t^2L_0)\}=1>.3.
$$
The denominators $m-t^2\nu L_0$ and $1-t\nu$ are strictly
positive. The source-box fifth moment is (DLB.2), so (PVB.3)
supplies the actual joint density and its weighted scores. Keeping
absolute derivatives in each prepared-velocity fibre permits the
original component mixing law to remain singular in velocity.

Write $K_\mu=K[\lambda_\mu,\Lambda_{2,\mu}]$.
The exact difference is
$$
\mu'P_\mu-\mu'P_{\mu'}
=(\mu'J_\mu-\mu'J_{\mu'})K_\mu
             +\lambda_{\mu'}(K_\mu-K_{\mu'}).
$$
The first term contracts full variation under its fixed kinetic
Markov map, and is bounded by $A_J\delta$ using (DLB.6).
The complete two-provider density theorem
{prf:ref}`thm-pvb-viscous-feedback` bounds the second term by
$\nu C_{\rm fb}\|\lambda_\mu-\lambda_{\mu'}\|_{W_4}$.
Its first-provider proof differentiates the actual drift coordinates
and Gaussian OU density in each velocity fibre. Its second-provider
proof holds the correlated density $\rho(y,w)$ fixed and uses the
affine velocity pullback at each fixed $y$. In particular its score
is $\nabla_w\rho$, including the shear contribution $tS_x$;
it is not the score of an independently resampled velocity marginal.
All derivative products are integrated before the provider interpolation.

The cap and final Gaussian are common Markov pushforwards in each
part of that comparison. So is the terminal map
$(x,v)\mapsto(x,v,\mathbf1_D(x))$. These pushforwards preserve the
full variation bound; no differentiation of a discontinuous terminal
indicator or discarded tail event is needed. Use (DLB.7) and
$\|\cdot\|_1\le\|\cdot\|_{W_4}$ to obtain (DLB.10).
:::

(sec-dlb-history)=
## 4. Exact delayed response of the nonlinear law

:::{prf:theorem} A full-law Duhamel formula through actual provider histories
:label: thm-dlb-delayed-response

Let $\mu_{j+1}=\mathcal F\mu_j$ and
$\mu'_{j+1}=\mathcal F\mu'_j$ start in $\mathfrak C$.
Evaluate $\epsilon_D$ in (RCB.14) at the actual default primitives
and set $\epsilon=\min\{\epsilon_D,1/2\}$, $q_D=1-\epsilon<1$.
Put
$$
P_j=P_{\mu_j},\qquad
\Delta_j=\mu_j-\mu'_j,\qquad
\mathcal E_j=\mu'_j(P_{\mu_j}-P_{\mu'_j}).
$$
For every $n\ge0$ and $B\ge1$, the following is an exact signed
measure identity:
$$
\Delta_{n+B}=
\Delta_nP_n\cdots P_{n+B-1}
+\sum_{j=0}^{B-1}
 \mathcal E_{n+j}P_{n+j+1}\cdots P_{n+B-1}.
\tag{DLB.11}
$$
An empty product is the identity. Consequently
$$
\|\Delta_{n+B}\|_1
\le q_D^B\|\Delta_n\|_1
+\sum_{j=0}^{B-1}q_D^{B-1-j}\|\mathcal E_{n+j}\|_1,
\tag{DLB.12}
$$
$$
\|\mathcal E_j\|_1\le L_{\rm fb}\|\Delta_j\|_1.
\tag{DLB.13}
$$
Both histories are their actual nonlinear population histories.
No common-provider identification, alive-kernel replacement or
unconditioned averaged-Jacobian product occurs.
:::

:::{prf:proof}
The common frozen source-box Doeblin theorem
{prf:ref}`thm-rcb-source-box-frozen-gap` applies at these exact
primitives. Every source lies in $D$, the prepared velocity is
bounded by $V_c$, and the actual joint stage first moment is bounded
by its explicit $M_w^D$. Its proper-degree and Gaussian area argument
minorizes every frozen root kernel by a common alive-output measure,
uniformly in the consistent root type, its mandatory revival and its
own frozen providers. Reducing the common floor to $\epsilon$ gives
$\|\xi P_j\|_1\le q_D\|\xi\|_1$ for every zero-mass signed
measure. The kernel still computes its actual full component law.

Subtract the two actual recursions:
$\Delta_{j+1}=\Delta_jP_j+\mathcal E_j$.
Successive substitution proves (DLB.11) in the stated order;
the kernels need not commute. Each $\mathcal E_j$ has zero mass
because both of its root kernels are probabilities. Therefore every
remaining product contracts its full variation by its number of
steps. The triangle gives (DLB.12), and the complete default feedback
theorem gives (DLB.13). The closed output class ensures all hypotheses
at every subsequent time. No stationary law or attractor was assumed.
:::

:::{prf:corollary} Exact current-alive transport readout
:label: cor-dlb-current-alive-response

For actual outputs at times $n\ge1$, put
$\alpha_n=(\mu_n)_A/(\mu_n)_A1$ and analogously $\alpha'_n$.
Their Euclidean physical phase laws satisfy
$$
\|\alpha_n-\alpha'_n\|_1
\le\frac2{a_{\rm ret}}\|\Delta_n\|_1,
\qquad
W_2(\alpha_n,\alpha'_n)^2
\le\frac{64}{a_{\rm ret}}\|\Delta_n\|_1.
\tag{DLB.14}
$$
The right side can be bounded by (DLB.12), capped by the physical
diameter bound. This is the actual normalized current-alive
population law; it is distinct from a finite-swarm QSD or a swarm
conditioned on future nonextinction.
:::

:::{prf:proof}
Each actual output alive mass is at least $a_{\rm ret}$.
Subtract the normalized alive measures and bound the numerator and
denominator changes separately. This gives the first inequality.
Alive positions lie in $[-2,2]^3$ and stored speed is at most two,
so their Euclidean phase squared diameter is at most $48+16=64$.
Maximal coupling matches their common measure and charges the
remaining probability at most this diameter. Its remaining mass is
half the full variation. This proves (DLB.14), with each law's
own actual alive denominator.
:::

(sec-dlb-endpoint)=
## 5. Last proved endpoint and the missing delayed absorption

:::{prf:remark} What the complete absolute register certifies
:label: rem-dlb-last-endpoint

(DLB.10) is a complete primitive default feedback estimate, and
(DLB.11) is an exact full marked-law identity. They extend the frozen
mixing result to a quantitative response to changing actual providers
with the full mandatory-dead population retained. The Gaussian
source and score class, raw alive normalizers, both viscous kicks,
cap and terminal marking have all been discharged in that response.

Using only the absolute estimate (DLB.13) gives
$$
\|\Delta_{n+B}\|_1
\le\min\{2,(q_D+L_{\rm fb})^B\|\Delta_n\|_1\}.
\tag{DLB.15}
$$
Indeed the one-step inequality follows from (DLB.12), and iteration
proves the displayed power. This register does not yield a strict
default nonlinear gap: $C_*\ge256$, $h_D\ge1$ and hence
$L_{\rm fb}\ge C_D\ge256$ for the declared positive role floors
$\kappa_b\le1$. These are properties of the displayed upper-bound
constants, not lower bounds on the actual feedback or an obstruction
to a sharper law estimate. Delaying the same absolute estimates does
not improve their proportional gain.

A nonlinear block rate requires a proved sharper estimate of the
signed sum in (DLB.11), or of its action on a specified transport
metric, that uses the alignment, cap and source correlations across
the block. The one-step signed interfaces in research 34--38 and
the restricted class in research 46 retain quantities that may
supply such a bound. An all-slot velocity burn, frozen-provider gap
or positive alive floor alone supplies none of that signed sum.
The source-interior condition of research 47 also requires an
entrance or preservation proof before it can support a delayed
small-dead argument.

No full default nonlinear attraction, invariant marked law, finite
QSD mixing or population-uniform finite alive convergence is inferred
from this response estimate. A finite survivor transfer would additionally
use the actual recent-window likelihood ratio of research 37, rather
than conditioning each intermediate Gaussian or multiplying future
survival probabilities into the population recursion. The proved
endpoint here is the complete default full-dead Gaussian feedback and
exact delayed marked-law response, with explicit current-alive readout.
:::
