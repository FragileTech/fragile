# Raw quadratic reward in the active population feedback proof

(sec-rqf-register)=
## 1. Exact reward and retained population regime

:::{prf:definition} Raw quadratic population record
:label: def-rqf-record

Retain the conservative population record of {prf:ref}`def-pvb-record`,
including $F(x)=-x$, both actual count-viscous kicks, the native smooth cap,
current-frame companions, recipient jitters and original frozen-slot component
velocities. Replace its bounded reward hypothesis by the actual raw reward
$$
R(x)=-|x|^2/2.
$$
Fix the eighth-moment budget $H_8$ of {prf:ref}`lem-pvb-eighth-moment`.
Use the positive logistic floors $f_b$, amplitudes $A_b$, reference powers
$\bar p_b>0$ and actual powers $p_b=\theta\bar p_b$, $0<\theta\le1$.
Retain the primitive constants $M,a_0,c_0,G_0,J_b$ of
{prf:ref}`lem-kuhw-positive-preparation-register`, through
$J_b=e^M\bar p_bA_b/(4f_b)$. Their bounds on the fitness range, gate
ceiling, gate derivatives and logistic-power derivatives hold for every raw
reward because the logistic bases lie in $[f_b,f_b+A_b]$.

Write $\sigma_r,\sigma_s>0$ for the actual standardizer floors and
$$
A_Q=H_8^{1/4}/2,\qquad
S_Q=(H_8^{1/2}/4+\sigma_r^2)^{1/2},\qquad
k_Q=(2S_Q)^{-1},\qquad B_Q=A_Q/\sigma_r.
$$
Define
$$
E_0=1,\qquad
E_j=\left(\frac{j}{2e k_Q}\right)^{j/2}\quad(j>0).
\tag{RQF.1}
$$
Here all physical coordinates and standardizer parameters use the declared
units of the population record. No physical truncation is made.
:::

(sec-rqf-logistic)=
## 2. The actual logistic derivative and moment changes

:::{prf:lemma} Uniform quadratic logistic tail
:label: lem-rqf-logistic-tail

For any environment with positional eighth moment at most $H_8$, let
$m=\mu R$, $v=\operatorname{Var}_\mu R$ and
$Z_r(x)=(R(x)-m)/(v+\sigma_r^2)^{1/2}$. The same bounds hold along
linear interpolation of the means and variances of two such environments.
For the actual full two-channel fitness $F$, at every fixed diversity mark,
$$
\begin{aligned}
Z_r(x)&\le B_Q-k_Q|x|^2,\\
|\partial_{Z_r}F|&\le4\theta J_r e^{B_Q-k_Q|x|^2},\\
|\partial_mF|&\le\theta C_m e^{-k_Q|x|^2},\\
|\partial_vF|&\le
 \frac{2\theta J_r e^{B_Q}}{\sigma_r^3}
                (A_Q+|x|^2/2)e^{-k_Q|x|^2},
\end{aligned}
\tag{RQF.2}
$$
where $C_m=4J_re^{B_Q}/\sigma_r$. At fixed environment and measurement
companion, the reward contribution to the physical fitness gradient obeys
$$
|\nabla_xF|_{\rm reward}\le\theta Q_x,
\qquad Q_x=4J_re^{B_Q}E_1/\sigma_r.
\tag{RQF.3}
$$
For $W_4(x)=1+|x|^4$, stronger weighted bounds are
$$
\begin{aligned}
\sup_x W_4|\partial_mF|
 &\le\frac{4\theta J_re^{B_Q}}{\sigma_r}(1+E_4),\\
\sup_x W_4|\partial_vF|
 &\le\frac{2\theta J_re^{B_Q}}{\sigma_r^3}
    [A_Q(1+E_4)+(E_2+E_6)/2],\\
\sup_x W_4|\nabla_xF|_{\rm reward}
 &\le\frac{4\theta J_re^{B_Q}}{\sigma_r}(E_1+E_5).
\end{aligned}
\tag{RQF.4}
$$
These weighted bounds concern the reward contribution. The diversity
channel is retained and is integrated with the moment weight below.
:::

:::{prf:proof}
Hölder gives $-A_Q\le m\le0$ and
$0\le v\le\mu R^2\le H_8^{1/2}/4$. Therefore the actual reward
denominator lies in $[\sigma_r,S_Q]$ and
$$
\frac{-|x|^2/2-m}{(v+\sigma_r^2)^{1/2}}
\le-\frac{|x|^2}{2S_Q}+\frac{A_Q}{\sigma_r}.
$$
The same interval bounds hold for interpolated moments. For
$\ell(z)=(1+e^{-z})^{-1}$,
$\ell'(z)=e^z/(1+e^z)^2\le e^z$ for every real $z$.
The positive-base formula for fitness gives
$|\partial_{Z_r}F|\le4\theta J_r\ell'(Z_r)$, including both channels.
Differentiate the actual standardized reward:
$|\partial_mZ_r|\le\sigma_r^{-1}$,
$|\partial_vZ_r|\le(A_Q+|x|^2/2)/(2\sigma_r^3)$, and
$|\nabla_xZ_r|\le|x|/\sigma_r$ at fixed moments.
This proves (RQF.2)--(RQF.3), since
$\sup_{r\ge0}r^je^{-k_Qr^2}=E_j$. That identity follows by
differentiating $r^je^{-k_Qr^2}$; the $j=0$ case has maximum one.
Multiply by $1+|x|^4$ and bound each nonnegative polynomial term
by its separate supremum to obtain (RQF.4).
:::

:::{prf:lemma} Raw reward normalization in full weighted variation
:label: lem-rqf-normalization

Let $\mu,\mu'$ have positional eighth moments at most $H_8$ and put
$D=\|\mu-\mu'\|_{W_4}$, $\delta=D/2$. Then
$$
|\mu R-\mu'R|\le D/4,\qquad
|\operatorname{Var}_\mu R-\operatorname{Var}_{\mu'}R|
                           \le(1+2A_Q)D/4.
\tag{RQF.5}
$$
Define
$$
T_r^Q=e^{B_Q}\left[
 \frac2{\sigma_r}
 +\frac{(1+2A_Q)(A_Q+E_2/2)}{\sigma_r^3}\right].
\tag{RQF.6}
$$
On identical physical and measurement types, the reward-normalizer
contribution to the full fitness difference is at most
$\theta J_rT_r^Q\delta$. Including the actual sampled diversity
normalizers gives
$$
|F_\mu-F_{\mu'}|\le\theta C_{F,0}^Q\delta,
\qquad
C_{F,0}^Q=J_rT_r^Q+J_sK_DT_s^f,
\quad K_D=1+2/\kappa_D.
\tag{RQF.7}
$$
Here $T_s^f=S_b/\sigma_s+S_b^3/(2\sigma_s^3)$ is the unchanged
bounded-diversity coefficient. The bound includes changes of the actual
reward mean and variance; those statistics are not kept fixed across
the two environments.
:::

:::{prf:proof}
The inequalities $|R(x)|\le W_4(x)/4$ and
$R(x)^2\le W_4(x)/4$ give the mean and second-moment differences.
Since both means have absolute value at most $A_Q$, the difference
of their squares is at most $2A_QD/4$. This proves (RQF.5).
Interpolate the means and variances. Integrating (RQF.2) along this
interpolation, with the changes in (RQF.5) and
$\sup |x|^2e^{-k_Q|x|^2}=E_2$, gives exactly
$\theta J_rT_r^Q\delta$.

Couple physical laws maximally and then their actual measurement
companions at matching physical states. Their bad marked-type mass
is at most $K_D\delta$. The unchanged bounded-diversity mean and
variance calculation therefore contributes
$\theta J_sK_DT_s^f\delta$. Add the reward and diversity changes.
:::

(sec-rqf-provider)=
## 3. Complete common-root preparation feedback

:::{prf:theorem} Raw quadratic common-root provider bound
:label: thm-rqf-preparation-feedback

Use the actual population tree $J_\mu(z,\cdot)$ of
{prf:ref}`thm-kuhw-frozen-provider-fourth-feedback`, with the reward
and moment class of {prf:ref}`def-rqf-record`. Retain its constants
$B_w,K_D,K_{D,w},D_J$ and define
$$
\begin{aligned}
D_{\beta,0}^Q&=a_0/\kappa_C^2+2G_0C_{F,0}^Q/\kappa_C,\\
L_0^Q&=2c_0K_D+D_{\beta,0}^Q,\\
C_J^Q&=D_J\left[
\frac{4(a_0+2c_0+c_0B_w)}{\kappa_D}
 +4(c_0K_{D,w}+D_{\beta,0}^QB_w+L_0^Q)
 +24L_0^Q(1+c_0B_w)\right],\\
L_J^Q&=B_wC_J^Q.
\end{aligned}
\tag{RQF.8}
$$
For $0<\theta\le\min\{1,\kappa_C/(8a_0)\}$,
$$
\|J_\mu(z,\cdot)-J_{\mu'}(z,\cdot)\|_{W_4}
 \le\theta C_J^QW_4(z)\|\mu-\mu'\|_{W_4}.
\tag{RQF.9}
$$
For a common root law $\zeta$ with positional fourth moment at most
$\sqrt{H_8}$, integration gives the same inequality with coefficient
$\theta L_J^Q$ and with $J_\mu(z),J_{\mu'}(z)$ replaced by
$\zeta J_\mu,\zeta J_{\mu'}$.
:::

:::{prf:proof}
Set $\delta=\|\mu-\mu'\|_{W_4}/2$. The marked-type coupling in
the preceding lemma has bad probability at most $K_D\delta$ and
weighted bad mass at most $2K_{D,w}\delta$. These estimates use
only the positive companion floor and the moment bound $B_w$.
On good types (RQF.7) gives the complete numeric fitness change.
Subtracting the donor normalizers and the clipped gates therefore
gives accepted-density change at most
$\theta D_{\beta,0}^Q\delta$. Each own density is at most
$c_*\le\theta c_0$. The complete outgoing token and incoming
Poisson intensities have coupling failure hazard at most
$2\theta L_0^Q\delta$ per queried matching vertex.

Expose the root source before its remaining component. A failed
outgoing match has weighted readout cost bounded by
$$
4D_J\theta(c_0K_{D,w}+D_{\beta,0}^QB_w+L_0^QW_4(z))\delta.
$$
On a matched source, complete the first-marginal tree with its own
unconditioned remaining primitives. After one exposed edge its
conditional expected size is at most $2e^{2c_*}$ by
{prf:ref}`lem-slcef-ordered-component`. The queried vertices before
the first failure are a subset of that completed component.
The conditional hazard sum, uniformly in the exposed source type,
is at most $4\theta L_0^Qe^{2c_*}\delta$. Its source weight has
mean at most $W_4(z)+c_*B_w$. The recipient jitter readout bound
is $D_JW_4$ and collision changes velocity only.

If the root measurement marks fail to match, subtract the common
identity-preparation baseline. An isolated root has zero increment;
its outgoing and incoming probabilities are bounded by $a_*,c_*$,
and its outgoing source weight by $c_*B_w$. Combining both own
increments with root-mark failure probability at most
$2\delta/\kappa_D$ gives the first term of (RQF.8), with its
displayed conservative factor four. Since $c_*\le1/8$,
$e^{2c_*}\le3$, and the matched-source term is covered by the
factor $24L_0^Q(1+c_0B_w)$. The failed-outgoing term is covered
by the middle term. These are exactly the three readout estimates
in the source-first provider proof, with (RQF.7) supplying its
formerly bounded-reward normalization step. No other step uses a
bound on raw reward. Passing from half to full weighted variation
cancels the common factors two, proving (RQF.9).
Integrate $W_4(z)$ against $\zeta$, whose mean weight is at most $B_w$.
:::

(sec-rqf-bv)=
## 4. Population spatial variation with raw reward

:::{prf:lemma} Weighted spatial variation of the raw quadratic preparation
:label: lem-rqf-preparation-bv

Let $\mu$ have $|v|\le V$, positional eighth moment at most $H_8$,
and representation $(x,v)=(Y+sZ,v)$ with $Z$ independent standard
Gaussian. Fix $1\le p\le5$ and put $M_p=\mu|x|^p$. Use
$L_{G,p},L_{J,p}$ of {prf:ref}`lem-pvb-weighted-preparation-bv` and
$B_D,B_C$ of (KVS.2). Define
$$
G_Q=\theta(Q_x+J_s/\sigma_s),\qquad
B_{\rm pat}^Q=B_D+2a_*B_C+
             2(L_{\rm rec}+\kappa_C^{-1}L_{\rm don})G_Q,
$$
$$
B_p^Q=d[L_{G,p}+L_{J,p}+(1+M_p)B_{\rm pat}^Q].
\tag{RQF.10}
$$
For the actual population preparation $\lambda=\mu J_\mu$,
$$
\sum_a\int(1+|X|^p)|D_{X_a}\lambda|\le B_p^Q.
\tag{RQF.11}
$$
The same bound is valid with absolute derivatives taken in prepared
velocity fibres before mixing. This is a population-row result;
it does not assert a finite-array joint weighted derivative bound.
:::

:::{prf:proof}
In this proof $\mu$ is fixed and $x$ is the root integration variable.
The actual population normalizers are deterministic functions of
that fixed environment. Keeping them fixed while differentiating the
dummy variable $x$ is their exact definition. Their changes when the
environment changes were separately charged in (RQF.5)--(RQF.9).
For a fixed root measurement companion the full root fitness spatial
gradient is at most $G_Q$: use (RQF.3) for reward and
$|\nabla_xs(x,y_D)|\le1$ for diversity. Its denominator is at least
$\sigma_s$, and its logistic-power derivative at most $\theta J_s$.

Split the root law by its accepted outgoing edge. On the copied
branch the recipient jitter is independent of every graph primitive
and the collision velocities. Translate that jitter to differentiate
the prepared root position while leaving the stored preparation
velocity fixed. Accepted mass is at most $a_*$ and accepted source
density is at most $c_*$ times $\mu$. The weighted Gaussian score
calculation gives $dL_{J,p}$ exactly.

On the persistent branch the prepared root position equals its own
entering $x$. Conditional on its mark, the no-edge factor is
$1-\int\beta_\mu(t,u)\eta_\mu(du)$ and the incoming children are
the actual marked Poisson process of intensity
$\beta_\mu(u,t)\eta_\mu(du)$. Each child's outgoing edge is used;
its further incoming subtree, given its type, is a Markov readout
independent of the dummy root position. Original slot velocities
and a fixed component Haar matrix determine the root collision
velocity. They contain no further spatial dependence in this term.

For one spatial coordinate write $d_C=D_*/\epsilon_C^2$.
The no-edge factor has derivative at most
$2a_*d_C+L_{\rm rec}G_Q$. The incoming intensity has full derivative
mass at most
$a_*d_C/\kappa_C+L_{\rm don}G_Q/\kappa_C$.
For finite intensity measures $I,I'$, couple their common Poisson
process and their residual processes; the full variation between
the process laws is at most $2\|I-I'\|$. Consequently its derivative
has full variation at most twice that intensity derivative mass.
Subtrees and the Haar readout decrease variation. No conditioning
on subsequent exploration successes is used.
The normalized root measurement law has full derivative variation
at most $2D_*/\epsilon_D^2\le B_D$. Combining these three derivatives
is bounded by $B_{\rm pat}^Q$; the constants $B_D,B_C$ deliberately
overcharge the displayed population-row quantities. Product
differentiation does not divide by a no-edge or acceptance probability.

The persistent root Gaussian score, weighted by $1+|x|^p$, costs
$dL_{G,p}$, using Jensen $\mathbb E|Y|^p\le M_p$. Its pattern
derivative costs at most $d(1+M_p)B_{\rm pat}^Q$. There are no
other prepared coordinates to hold fixed for this sampled-row
derivative, so no compensation jitter is required. In particular
large copied recipient positions do not introduce a velocity term:
the component velocities are the original frozen slots.
Sum the persistent and copied terms to get (RQF.11).

Perform the calculation first against smooth compactly supported
joint phase tests, retaining absolute values before latent, mark,
graph and velocity mixing. Spatial mollification and increasing
weight truncations yield the finite derivative measures by lower
semicontinuity. The same estimates integrated over prepared-velocity
fibres follow by this pre-mixing absolute bound, as required by the
joint OU score calculation. Since $p\le5$, the moment $M_p$ is at
most $H_8^{p/8}$ and every displayed term is finite.
:::

(sec-rqf-population)=
## 5. Completed raw-reward population parameter interval

:::{prf:corollary} Active positive-viscosity convergence with the raw harmonic reward
:label: cor-rqf-active-population

In {prf:ref}`def-rqf-record`, construct the uniform envelopes and
positive endpoints of {prf:ref}`def-pvb-positive-envelopes` with the
following replacements only:

1. Use $L_J^Q$ of (RQF.8) for the complete common-root preparation
   feedback constant.
2. Use $B_5^Q$ of (RQF.10), with $M_5=H_8^{5/8}$ and upper endpoint
   values of $a_*,\theta,L_{\rm rec},L_{\rm don}$, for the uniform
   source spatial score. The first-moment and fifth-moment envelopes
   after preparation remain unchanged.

All such constants are finite. The resulting $\theta_*^Q,\nu_*^Q$
are strictly positive. For every
$0<\theta\le\theta_*^Q$, $0<\nu\le\nu_*^Q$, the conclusions
of {prf:ref}`thm-pvb-active-population-convergence` and
{prf:ref}`cor-pvb-alive-w2-relaxation` hold for the unchanged channels
$$
F(x)=-x,\qquad R(x)=-|x|^2/2.
$$
This includes the original harmonic step parameters in
{prf:ref}`cor-pvb-original-step-active-viscosity`, with their declared
positive floors, widths and noises, in this sufficient small-power and
small-viscosity interval. No fitness-gap or variance lower bound is needed.
:::

:::{prf:proof}
The eighth-moment drift and the gate-range primitive envelopes depend
on the logistic fitness range and accepted donor column bound, not on
bounded raw reward. They therefore hold unchanged. Equations
(RQF.8)--(RQF.11) supply exactly the two population interfaces where
bounded raw reward entered the completed proof: common-root provider
feedback and weighted source BV. The fresh Gaussian representation,
speed bounds and eighth-moment class verify both replacements.

All later inverse, fibrewise joint OU score, actual second-provider
feedback and local minorization arguments use these interfaces and
the same harmonic moment class. Their constants remain finite with
the two declared replacements. The frozen Harris margin is strictly
positive, and each feedback term spends at most one quarter of it
at the new explicit endpoints. Thus the complete weighted contraction
coefficient is at most $(1+q_H)/2<1$. Gaussian-mixture completeness,
the strict eighth-moment burn-in, Banach fixed point, uniqueness in
the finite-eighth-moment class and the physical Wasserstein bound
are consequently the same proved implications, with their actual
raw-reward provider. This proves the corollary.
:::

:::{prf:remark} Finite-particle scope
:label: rem-rqf-particle-scope

The preparation-only conditional scalar mean-square estimate in
{prf:ref}`lem-vupt-preparation` uses bounded logistic reward factors,
not bounded raw reward: the reward statistics at a fixed empirical
input are unchanged by measurement resampling. Its proof remains
available. The bounded-reward weak modulus in
{prf:ref}`lem-vupt-population-modulus` compares distinct physical
input laws and uses bounded raw reward in that comparison. A raw-reward
version needs its own moment-dependent mean/variance calculation and
restart analysis. Those calculations are supplied separately by
{prf:ref}`lem-rqpt-population-modulus` and
{prf:ref}`thm-rqpt-uniform-time`, yielding the raw-reward conservative
particle transfer. Killed-law conditioning remains a separate result;
it does not follow from this population corollary alone.
:::
