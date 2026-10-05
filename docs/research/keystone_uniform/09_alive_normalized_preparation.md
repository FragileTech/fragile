# Alive-normalized Keystone preparation and exact well flux

(sec-kuan-target)=
## The original normalization and the fixed preparation law

The pressure coordinate is exactly Chapter 3's
$W=N^{-1}\sum_{i\in I_{11}}|\Delta\delta_{x,i}|^2$, with each entering
swarm centered at its own alive positional mean. The alive-normalized
donor transport below is an auxiliary quantity for bounding complete
preparation; it does not replace $W$. The complete proposal still has
$N$ rows, including every mandatory revival. Its copying map removes a
common translation exactly after centering. The negative recipient pressure, signed
incoming donors and barycenter corrections are those of Chapter 6a,
Section 14.2; none is replaced by a contraction assumption. Well flux
is retained so that different attainable orbit populations are not
required to coalesce.

:::{prf:definition} Actual alive inputs and donor-centered comparison
:label: def-kuan-input

Use the unchanged complete parameter record of
`def-cgd-parameter-register` and the actual preparation arrays of
`def-ku-preparation-record`. Let $A,\widetilde A$ be the two nonempty
alive pools, $M=|A|$, $\widetilde M=|\widetilde A|$,
$I=A\cap\widetilde A$, $K=|I|$, and $M_\vee=\max(M,\widetilde M)$.
The uniform probability measures on the two alive label pools have
the maximal common-label coupling

$$
\Gamma(i,j)=\frac{\mathbf1_{i=j\in I}}{M_\vee}
+\frac{r_i\widetilde r_j}{\tau},\quad
r_i=\frac{\mathbf1_{i\in A}}M-\frac{\mathbf1_{i\in I}}{M_\vee},\quad
\widetilde r_j=\frac{\mathbf1_{j\in\widetilde A}}{\widetilde M}
-\frac{\mathbf1_{j\in I}}{M_\vee},\quad
\tau=1-\frac K{M_\vee},
\tag{KUAN.1}
$$

where the residual fraction is zero when $\tau=0$. Thus $\Gamma$
has the exact uniform alive marginals and residual mass $\tau$.
Define the auxiliary actual entering discrepancies

$$
P_A=\sum_{i,j}\Gamma(i,j)|x_i-\widetilde x_j|^2,\qquad
U_A=\sum_{i,j}\Gamma(i,j)|v_i-\widetilde v_j|^2,
$$

$$
Z_A=\sum_{i,j}\Gamma(i,j)|z_i-\widetilde z_j|^2,\quad
Z_N=\frac1N\sum_i|z_i-\widetilde z_i|^2,\quad
U_N=\frac1N\sum_i|v_i-\widetilde v_i|^2.
\tag{KUAN.2}
$$

Here $z_i$ is the actual squashed phase-space feature. All slot
velocities, including retained dead velocities, enter $U_N$; retained
dead positions enter their actual bounded comparison features in
$Z_N$. Alive physical positions lie in $D\subset B(0,B_x)$ and
stored velocities satisfy $|v_i|\le V$. No raw dead position bound
is imposed. The feature-map Lipschitz property gives
$Z_A\le P_A+\lambda_{\rm alg}U_A$.

Writing $m_A=M^{-1}\sum_Ax_i$ and $\widetilde m_A$ similarly,
the exact alive normalization also gives

$$
P_A=|m_A-\widetilde m_A|^2+
\sum_{i,j}\Gamma(i,j)
|(x_i-m_A)-(\widetilde x_j-\widetilde m_A)|^2.
\tag{KUAN.3}
$$

Put $P_A^\circ=\sum_{i,j}\Gamma(i,j)
|(x_i-m_A)-(\widetilde x_j-\widetilde m_A)|^2$. The actual Chapter 3
pressure coordinate and the residual unmatched-label cost are

$$
W=\frac1N\sum_{i\in I}|(x_i-m_A)
 -(\widetilde x_i-\widetilde m_A)|^2,\qquad
P_A^\circ=\frac N{M_\vee}W+
\sum_{i,j}\frac{r_i\widetilde r_j}{\tau}
|(x_i-m_A)-(\widetilde x_j-\widetilde m_A)|^2.
\tag{KUAN.3a}
$$

Consequently $W$ is zero for two alive arrays differing only by a
translation. Their raw donor transport $P_A$ can nevertheless be
positive. Neither a raw fixed-slot amplification nor a translation
counterexample disproves the source centered Keystone estimate.
The centered error, alive barycenter error and residual unmatched-label
cost remain separate quantities. Actual reward and squashed-feature
sensitivities below still use $P_A$, since their configured landscape
and feature map need not be translation invariant. For a declared phase
comparison, label pairing and phase mass discrepancies must be those
of that comparison; this record does not require convergence between
distinct limiting phase populations.
:::

(sec-kuan-measurement)=
## Actual measurement coupling and different alive normalizations

:::{prf:lemma} The original global standardizer on a common alive transport space
:label: lem-kuan-weighted-normalization

For every frozen pair of raw arrays $b_i$ on $A$ and
$\widetilde b_j$ on $\widetilde A$, their actual globally standardized
values, using their respective alive-population means and variances,
satisfy

$$
\sum_{i,j}\Gamma(i,j)|Z_{b,i}-\widetilde Z_{b,j}|^2
\le\sigma_{b,\min}^{-2}
\sum_{i,j}\Gamma(i,j)|b_i-\widetilde b_j|^2.
\tag{KUAN.4}
$$

The actual logistic-product fitness arrays therefore obey

$$
\|F-\widetilde F\|_{2,\Gamma}
\le\sum_{b=r,s}\frac{H_b}{\sigma_{b,\min}}
\|b-\widetilde b\|_{2,\Gamma},
\tag{KUAN.5}
$$

where $H_b$ is exactly the primitive derivative coefficient in
`lem-ku-standardization`. The canonical coefficient is $10.5$ for
each raw channel. No equal alive counts, positive variance or fitness
gap is required.

*Proof.* Work in the finite Hilbert space $L^2(\Gamma)$ and regard
$b(i,j)=b_i$, $\widetilde b(i,j)=\widetilde b_j$. The two means
and variances in this space are exactly the actual means and
variances on the two alive pools, because $\Gamma$ has their
uniform probability marginals. For the centering projection $P$,
$y=Pb$ and $s=(\|y\|^2+\sigma^2)^{1/2}$, differentiation gives
$D(y/s)=P/s-y\otimes y/s^3$. Its operator norm is at most
$1/\sigma$, by the same projection/rank-one calculation as
`lem-ku-standardization`, which does not depend on the number or
weights of atoms. Integrate along the line joining the two functions
in $L^2(\Gamma)$. This proves (KUAN.4). The actual logistic-power
product has coordinate slopes $H_b$; Minkowski then gives (KUAN.5).
$\square$
:::

:::{prf:lemma} A whole-array measurement coupling that preserves independent rows
:label: lem-kuan-measurement-coupling

For each common alive label $i\in I$, let $P_i,\widetilde P_i$ be
its two actual normalized Gaussian measurement companion laws.
For singleton measurement use the declared self companion. Put

$$
\lambda_i(j,k)=\Gamma(j,k)
\min\{MP_i(j),\widetilde M\widetilde P_i(k)\},\qquad
T_i=1-\sum_{j,k}\lambda_i(j,k).
$$

Complete this subcoupling by the product of its nonnegative residual
row marginals divided by $T_i$, with zero residual when $T_i=0$.
Use these couplings independently for the common measurement rows
$i\in I$. Draw each remaining alive measurement row independently
with its actual law. This constructs a joint law of the two complete
measurement vectors with their correct independent one-swarm row
marginals. It does not draw multiple measurement companions for a
single row when $\Gamma$ has split mass.

With $\ell_D=1/(\epsilon_D\sqrt e)$ and $\kappa_D$ the actual
Gaussian weight floor, define

$$
\mathcal E_D=\sum_{i\in I}\frac1{M_\vee}
\left[\frac{d_M}{M}+\frac{d_{\widetilde M}}{\widetilde M}
-2d_Md_{\widetilde M}\Gamma(i,i)\right],\qquad
d_M=\mathbf1_{M>1}.
$$

Then

$$
\overline T_D:=\sum_{i\in I}T_i/M_\vee
\le\min\left\{1,\frac{4\ell_D}{\kappa_D}\sqrt{Z_A}
+\frac{2\mathcal E_D}{\kappa_D}\right\}.
\tag{KUAN.6}
$$

For the actual raw diversity range
$R_s=\sqrt{D_*^2+\delta_D^2}-\delta_D$,

$$
\mathbb E_m\|s-\widetilde s\|_{2,\Gamma}^2
\le2(1+2/\kappa_D)Z_A+R_s^2(\tau+\overline T_D).
\tag{KUAN.7}
$$

If $R(x,v)=-U(x)-\lambda_{\rm vel}|v|^2$ and
$L_U=\sup_D|\nabla U|$ is the actual computed landscape profile,

$$
\|r-\widetilde r\|_{2,\Gamma}
\le\sqrt{L_U^2+(2\lambda_{\rm vel}V)^2/\lambda_{\rm alg}}
\sqrt{P_A+\lambda_{\rm alg}U_A}.
\tag{KUAN.8}
$$

For the quadratic reference $L_U=B_x$ and
$\lambda_{\rm vel}=0$. For native Rastrigin $U=10d+
\sum_r[x_r^2-10\cos(2\pi x_r)]$, the unchanged formula gives
$L_U\le2B_x+20\pi\sqrt d$. Combining (KUAN.5), (KUAN.7) and
(KUAN.8), with Jensen on the outer measurement expectation, yields
the completely specified fitness envelope

$$
\begin{aligned}
\mathcal F_A:={}&\frac{H_r}{\sigma_r}
\sqrt{L_U^2+(2\lambda_{\rm vel}V)^2/\lambda_{\rm alg}}
\sqrt{P_A+\lambda_{\rm alg}U_A}\\
&+\frac{H_s}{\sigma_s}
\sqrt{2(1+2/\kappa_D)Z_A+R_s^2(\tau+\overline T_D)},\qquad
\mathbb E_m\|F-\widetilde F\|_{2,\Gamma}\le\mathcal F_A.
\end{aligned}
\tag{KUAN.9}
$$

*Proof.* The first marginal of $\lambda_i$ is at most $P_i$,
because the first marginal of $\Gamma$ is $1/M$ on each alive
label; its second marginal is at most $\widetilde P_i$ similarly.
The two residual marginal masses equal $T_i$, so their product
divided by $T_i$ completes a valid coupling. Each row is coupled
at most once. Independence of these pairs and of the unpaired
rows establishes the whole-vector marginal statement.

Under $\Gamma$, write the unnormalized weights as
$a(j,k)=w_D(z_i,z_j)I_i(j)$ and
$\widetilde a(j,k)=w_D(\widetilde z_i,\widetilde z_k)
\widetilde I_i(k)$, where the eligibility indicator excludes the
alive root when its pool has at least two rows and includes the
singleton self convention. Their means are $Z_i/M$ and
$\widetilde Z_i/\widetilde M$, each at least $\kappa_D/2$.
For two nonnegative arrays with positive means $a_0,b_0$, the
total variation of their normalized densities is at most
$\mathbb E|a-b|/\max(a_0,b_0)$: split the normalized difference,
use $|a_0-b_0|\le\mathbb E|a-b|$, and choose the larger mean.
Here

$$
\mathbb E_\Gamma|a-\widetilde a|
\le\ell_D\left[|z_i-\widetilde z_i|
+\mathbb E_\Gamma|z_j-\widetilde z_k|\right]
+\mathbb E_\Gamma|I_i-\widetilde I_i|.
$$

The eligibility expectation is exactly
$d_M/M+d_{\widetilde M}/\widetilde M
-2d_Md_{\widetilde M}\Gamma(i,i)$. Cauchy--Schwarz and the
diagonal mass $1/M_\vee$ give (KUAN.6). This keeps the self
exclusion correction, which is zero for equal identical pools.

On its $\lambda_i$ subcoupling, the two smoothed diversity norms
differ by at most
$|z_i-\widetilde z_i|+|z_j-\widetilde z_k|$.
Furthermore $MP_i(j)\le2/\kappa_D$, including singleton
measurement, so $\lambda_i\le(2/\kappa_D)\Gamma$.
Use $(u+v)^2\le2(u^2+v^2)$, average the diagonal root mass,
and bound each residual companion difference by $R_s$.
The residual root mass in (KUAN.1) also has raw diversity difference
at most $R_s$, irrespective of the compatibility of root pairings.
This proves (KUAN.7). The potential gradient bound and
$||v|^2-|w|^2|\le2V|v-w|$ give (KUAN.8). Finally apply the
weighted standardization lemma and Jensen, proving (KUAN.9).
No expectation has been taken before evaluating a gate or a
fitness normalizer. $\square$
:::

(sec-kuan-source)=
## Incoming sources, jitter and the complete paired preparation

:::{prf:theorem} Alive-normalized source and collision preparation bound
:label: thm-kuan-preparation

Use the actual marked token laws (KU.15)--(KU.17), conditional on
the complete coupled sampled fitness vectors. Let $\bar T$ be their
average row-token mismatch probability. Let $\theta_N$ be the exact
paired accepted-component mismatch probability (KU.26). Then

$$
\mathbb E_{\rm prep}\frac1N\sum_i|X_i-\widetilde X_i|^2
\le\left(\frac{M_\vee}{N}+\frac1{\kappa_C}\right)P_A
+(4B_x^2+d\sigma_J^2)\mathbb E_m\bar T.
\tag{KUAN.10}
$$

With $A_c=\max(1,\alpha_c^2)$,
$V_c=(1+2|\alpha_c|)V$,

$$
\mathbb E_{\rm prep}\frac1N\sum_i|V_i^C-\widetilde V_i^C|^2
\le A_cU_N+4V_c^2\mathbb E_m\theta_N.
\tag{KUAN.11}
$$

The positions are frozen donor copies, while $V_i^C$ uses every
component member's own frozen velocity, including revived dead
members and donors that persist in position. The exact mixed term
for any chosen phase-space quadratic form is

$$
\mathbb E_R[(X_i-\widetilde X_i)\cdot(V_i^C-\widetilde V_i^C)]
=(X_i-\widetilde X_i)\cdot(m_i-\widetilde m_i),
\tag{KUAN.12}
$$

before averaging its actual paired plans and jitters. If $P_C,U_C$
denote the right sides of (KUAN.10)--(KUAN.11), a fully explicit
upper envelope is
$\alpha P_C+2|\beta|\sqrt{P_CU_C}+\gamma_PU_C$.
The signed mixed moment (KUAN.12) is preferable when its sign is useful.

For arbitrary different alive pools, the actual candidate clone law
obeys

$$
\overline{\operatorname{TV}}_C
\le\min\left\{1,
\frac{2\ell_C}{\kappa_C}(\sqrt{Z_N}+\sqrt{Z_A})
+\frac{4\tau}{\kappa_C}\right\},\qquad
\ell_C=1/(\epsilon_C\sqrt e).
\tag{KUAN.13}
$$

Writing $s_A=|A\triangle\widetilde A|/N$, the accepted token
mismatch is bounded by

$$
\mathbb E_m\bar T\le\min\left\{1,
2\overline{\operatorname{TV}}_C+s_A+
\frac{M_\vee}{N}
(L_{\rm rec}+\kappa_C^{-1/2}L_{\rm don})\mathcal F_A\right\},
\tag{KUAN.14}
$$

where the gate slopes are the original saturation-aware primitive
values in (KU.8). For equal pools, use the sharper normalized
Gaussian bound

$$
\overline{\operatorname{TV}}_C
\le\min\left\{1,
\frac{R_*\sqrt{Z_N}+D_*\kappa_C^{-1/2}\sqrt{Z_A}}
 {2\epsilon_C^2}\right\},
\tag{KUAN.15}
$$

and (KU.9)--(KU.11) for the alive measurement array. An alive
singleton has identical point-mass candidate laws when the pool
agrees, so its candidate mismatch is exactly zero.

*Proof.* Let $Q_{ij}$ be a marginal source matrix, summing both
token flags. For each alive source label $j$,

$$
\sum_iQ_{ij}\le1+\frac1{\kappa_C}
+\frac{N-M}{\kappa_CM}=1+\frac N{\kappa_CM}.
$$

The three terms are persistence, at most $M-1$ distinct alive
recipients, and $N-M$ mandatory revivals. Their denominators are
exactly $M-1$ and $M$, respectively. At $M=1$ the exact column
sum is $N$, which satisfies the displayed upper bound.
Common-token mass at source label $j$ is bounded by this column
sum in both marginals, hence by $1+N/(\kappa_CM_\vee)$.
Consequently its normalized positional cost is at most

$$
\left(1+\frac N{\kappa_CM_\vee}\right)
\frac1N\sum_{j\in I}|x_j-\widetilde x_j|^2
\le\left(\frac{M_\vee}N+\frac1{\kappa_C}\right)P_A.
$$

On a residual token pair both source positions are alive and lie
in $B(0,B_x)$, giving positional square at most $4B_x^2$.
Only residual tokens can have different jitter flags; their exact
Gaussian difference variance is at most $d\sigma_J^2$.
This proves (KUAN.10). The matched-component orthogonal
mean/deviation decomposition and the exact Haar moments in
`thm-ku-marked-collision` give (KUAN.11)--(KUAN.12), keeping
the full frozen dead-slot velocity data.

For (KUAN.13), if $\tau\ge1/2$ the pool term already exceeds one.
Otherwise the alive pools both have at least two members, except
for the equal singleton case which has zero candidate mismatch.
Their maximum normalizer is at least
$\kappa_C(M_\vee-1)\ge\kappa_CM_\vee/2$.
The unnormalized common-weight difference is at most
$\ell_C[K|z_i-\widetilde z_i|+
\sum_{j\in I}|z_j-\widetilde z_j|]$; omitted self terms only
decrease this bound. The two unmatched-pool contributions total
at most $|A\triangle\widetilde A|\le2M_\vee\tau$.
Apply the normalized-density inequality from the preceding proof,
average rows and use Cauchy--Schwarz and
$M_\vee^{-1}\sum_{j\in I}|z_j-\widetilde z_j|^2\le Z_A$.

On each common alive recipient, the accepted-measure difference
is at most twice its candidate TV, plus the two actual gate slopes
times its recipient and donor fitness differences. Other rows
with different alive marks cost at most one; dead rows in both
states have exactly the candidate TV because revival is unconditional.
The common recipient average is bounded by
$(M_\vee/N)\|F-\widetilde F\|_{2,\Gamma}$.
For the donor average, the $\widetilde M-1$ alive-recipient column
load is at most $1/\kappa_C$. Cauchy--Schwarz with total joint
mass at most $K/N$ bounds it by
$(M_\vee/N)\kappa_C^{-1/2}\|F-\widetilde F\|_{2,\Gamma}$.
Thus (KUAN.9) yields (KUAN.14). A common singleton has no live
gate; different singleton pools are covered by the pool/mark terms.

For equal pools, interpolate all frozen features and differentiate
their actual normalized Gaussian laws as in `lem-ku-companion`.
The full-row candidate column load is at most $N/(\kappa_CM)$;
its $1/N$ average is therefore controlled by the alive-normalized
$Z_A/\kappa_C$. This gives (KUAN.15), with no alive-fraction floor.
$\square$
:::

:::{prf:theorem} Exact paired alive-centered copying and separate centroid update
:label: thm-kuan-centered-preparation

Condition on the two complete sampled fitness vectors. Write
$H_i(e,j;f,k)$ for the actual paired token law on row $i$, including
all live acceptance, persistence and mandatory revival probabilities.
The token flags are $e,f\in\{0,1\}$ and the source labels are alive
in their respective entering swarms. Share the recipient Gaussian
jitter between these two tokens and draw paired tokens and jitters
independently across rows. Define

$$
d_{jk}=(x_j-m_A)-(\widetilde x_k-\widetilde m_A),\quad
u_i=\sum_{e,j,f,k}H_i(e,j;f,k)d_{jk},
$$

$$
c_i=\sum_{e,j,f,k}H_i(e,j;f,k)
 [|d_{jk}|^2+d\sigma_J^2(e-f)^2]-|u_i|^2,\qquad
\bar u=\frac1N\sum_i u_i.
$$

Here $c_i\ge0$ is the exact conditional variance trace of the paired
row difference. Put $\bar X=N^{-1}\sum_iX_i$ and
$\overline{\widetilde X}=N^{-1}\sum_i\widetilde X_i$. Then

$$
\begin{aligned}
\mathbb E\frac1N\sum_i
 |(X_i-\bar X)-(\widetilde X_i-\overline{\widetilde X})|^2
={}&\frac1N\sum_{i,e,j,f,k}H_i(e,j;f,k)
 [|d_{jk}|^2+d\sigma_J^2(e-f)^2]\\
&-|\bar u|^2-\frac1{N^2}\sum_i c_i,
\end{aligned}
\tag{KUAN.15a}
$$

$$
\mathbb E|\bar X-\overline{\widetilde X}|^2
=|m_A-\widetilde m_A+\bar u|^2+
\frac1{N^2}\sum_i c_i.
\tag{KUAN.15b}
$$

In the actual common-token coupling, (KUAN.15a) has the bound

$$
\mathbb E\frac1N\sum_i
 |(X_i-\bar X)-(\widetilde X_i-\overline{\widetilde X})|^2
\le\left(\frac{M_\vee}N+\frac1{\kappa_C}\right)P_A^\circ
+(16B_x^2+d\sigma_J^2)\bar T
-|\bar u|^2-\frac1{N^2}\sum_i c_i.
\tag{KUAN.15c}
$$

Thus copying has no direct positive term in
$|m_A-\widetilde m_A|^2$ in its centered output. A translation can
still change the actual sampled gates and Gaussian source laws through
the unchanged environment and feature map; those effects remain in
$H_i$, $\bar T$, $\bar u$ and the computed measurement coefficients.
The centers $\bar X,\overline{\widetilde X}$ here are those of the
complete preterminal proposals. Current alive centers after the
terminal test use its actual survivor indicators and must be evaluated
at that stage; they are not identified with these proposal centers.

*Proof.* Every raw source difference equals
$m_A-\widetilde m_A+d_{jk}$. The paired row difference is therefore
$m_A-\widetilde m_A+d_{jk}+\sigma_J(e-f)\zeta_i$.
Its conditional mean and variance are the displayed $u_i$ and $c_i$.
The complete conditional position pairs are independent across rows,
so their empirical mean has variance trace $N^{-2}\sum_i c_i$.
Subtracting its exact square proves (KUAN.15a)--(KUAN.15b).
Common tokens have the incoming-column bound in the preceding theorem,
applied to the centered alive donor discrepancies. On every residual
token $|d_{jk}|\le4B_x$, since each alive center lies in $B(0,B_x)$.
The same shared-jitter calculation gives at most $d\sigma_J^2$ per
residual token. This proves (KUAN.15c), retaining both negative
barycenter terms. Outer measurement averaging is taken only after
these conditional quantities have been evaluated. $\square$
:::

:::{prf:corollary} The original two-swarm Keystone pressure is unchanged
:label: cor-kuan-original-pressure

Under exactly the source hypotheses of
{prf:ref}`thm-keystone-discharged-averaged-pressure`, use its centered
$W$ from (KUAN.3a) and its already computed constants
$\chi_*,B_*,W_0$. Then the present whole-array measurement coupling
satisfies the same original estimate

$$
\mathcal A_{11}:=\mathbb E_m\frac1N\sum_{i\in I}
 (p_i+\widetilde p_i)
 |(x_i-m_A)-(\widetilde x_i-\widetilde m_A)|^2
\ge\chi_*(W-W_0)-\frac{B_*}{N^2}.
\tag{KUAN.15d}
$$

Neither $P_A$, $P_A^\circ$ nor the one-population variance $W_A$
replaces $W$ in this statement. Formula (KUAN.3a) instead supplies
the exact alive-mass and residual-label relation needed when a
complete preparation bound is composed with (KUAN.15d).
The marked port is exactly
{prf:ref}`thm-slc-marked-keystone-port`; its signed remainder retains
the complete source, Haar, kinetic, cap and terminal terms.

*Proof.* The measurement coupling proved above has precisely the two
one-swarm measurement laws required by the source theorem. Its proof
uses only their summed marginal expectations, hence is unchanged.
The centered coordinates and common-label normalization are identical.
Apply (3.CC11a) and then its marked port, without altering any source
constant. $\square$
:::

(sec-kuan-current-law)=
## The current survivor law controls the remaining inverse alive mass

:::{prf:theorem} Complete Haar preparation under current survivor laws
:label: thm-kuan-current-preparation

Let both entering marginals be current-time survivor laws at $n\ge1$
or QSDs of the unchanged kernel. Use any valid explicitly computed
landing floor $a>0$ from `lem-ku-coupled-binomial-survival` or
`thm-ku-quadratic-binomial-survival`. Put

$$
c_*=(1-\log2)/2,\qquad
C_{{\rm inv},r}=(2/a)^r+[r/(e c_*a)]^r.
$$

The already proved current-time law gives
$\mathbb E(N/M)^r,\mathbb E(N/\widetilde M)^r
\le C_{{\rm inv},r}$ without a cumulative survival denominator.
Let $\varepsilon=\mathbb E_{S,\widetilde S,m}\bar T$. For every
$r>1$, the complete-component mismatch satisfies

$$
\mathbb E\theta_N\le\min\left\{1,
6e^{2/\kappa_C}\left[
2\varepsilon+\frac{2C_{{\rm inv},r}^{1/r}}{\kappa_C}
\varepsilon^{1-1/r}\right]\right\}.
\tag{KUAN.16}
$$

Consequently (KUAN.10)--(KUAN.12), averaged over these actual
entering laws, have coefficients independent of $N$, while retaining
all mandatory revival, frozen dead velocities, measurement correlations
and shared-component Haar rotations. No independence between the
alive mass and the mismatch has been assumed.

For native Rastrigin the landing input can be computed directly from
its unchanged force:

$$
\mathcal F_J=2(\sqrt d L_D+J)+20\pi\sqrt d,\quad
C_J=L_D+J+\eta\mathcal F_J+b_h\kappa_\nu V_c,
$$

$$
a_J=G_d(J/\sigma_J)
[\Phi((L_D-C_J)/\sigma_h)-\Phi((-L_D-C_J)/\sigma_h)]^d,
\quad \sigma_h^2=t^2q^2+s^2.
\tag{KUAN.17}
$$

This is a tagged-jitter probability integrated over the true
unbounded law, not a noise clipping rule or an $N$-row bounded-noise
event. The quadratic reference's sharper exact floor has
$\log a=-41.2528745903$ as evaluated in the conditioning record.

*Proof.* The actual fitness-ordered alive forest and mandatory dead
leaves give, conditionally on each entering pair and its frozen
fitnesses, the proved component estimate

$$
\theta_N\le6e^{2/\kappa_C}
\left[2+\frac{N/M+N/\widetilde M}{\kappa_C}\right]\bar T.
$$

This is a conservative version of (KU.27); its exact finite-plan
value can always replace it. For $r>1$, Holder gives

$$
\mathbb E[(N/M)\bar T]\le
[\mathbb E(N/M)^r]^{1/r}
[\mathbb E\bar T^{r/(r-1)}]^{1-1/r}
\le C_{{\rm inv},r}^{1/r}\varepsilon^{1-1/r},
$$

because $0\le\bar T\le1$. The same applies to the other
marginal, proving (KUAN.16). All joint correlations survive this
Holder bound. The inverse moment is that of
`thm-ku-uniform-alive-inverse-moments`; it is valid at each current
surviving time, not conditioned on survival of an entire future path.
Finally the recorded Rastrigin force is
$F_r(x)=-2x_r-20\pi\sin(2\pi x_r)$. Its norm is at most
$2|x|+20\pi\sqrt d$, giving (KUAN.17) from the explicit landing
lemma. $\square$
:::

(sec-kuan-signed-variance)=
## Original Keystone pressure in the alive-centered signed donor balance

:::{prf:theorem} Exact alive-centered preparation variance and the unchanged global pressure
:label: thm-kuan-signed-variance

For one nonextinct entering swarm, set $m=M/N$,
$m_A=M^{-1}\sum_Ax_i$, $W_A=M^{-1}\sum_A|x_i-m_A|^2$.
Condition on its actual sampled fitnesses. Use the actual accepted
array $b_{ij}$, with $b_{ij}=P_C(j\mid i)$ on dead rows, and define

$$
A_{\rm rec}^A=\frac1M\sum_{i\in A}p_i|x_i-m_A|^2,\quad
D_{\rm live}^A=\frac1M\sum_{i\in A,j\in A}b_{ij}|x_j-m_A|^2,
$$

$$
D_{\rm rev}^N=\frac1N\sum_{i\notin A,j\in A}P_C(j\mid i)|x_j-m_A|^2,
$$

$$
\bar t=\frac1N\left[
\sum_{i\in A,j\in A}b_{ij}(x_j-x_i)
+\sum_{i\notin A,j\in A}P_C(j\mid i)(x_j-m_A)\right].
$$

Let $\sigma_i^2$ be each actual proposal row's conditional variance
trace and $\bar p=N^{-1}\sum_i p_i$, including $p_i=1$ on dead rows.
The output full-row empirical variance satisfies exactly

$$
\mathbb E W_N(S^C)=m[W_A-A_{\rm rec}^A+D_{\rm live}^A]
+D_{\rm rev}^N+d\sigma_J^2\bar p
-|\bar t|^2-\frac1{N^2}\sum_i\sigma_i^2.
\tag{KUAN.18}
$$

The source Chapter 3/Chapter 6a one-population Keystone constants
$k_{\rm key},p,E_{\max}$ are unchanged. Its actual alive subpopulation
has the same measurement, normalizer and live donor law as an
$M$-row all-alive preparation, hence

$$
\mathbb E_m A_{\rm rec}^A\ge k_{\rm key}W_A^p-E_{\max}/M^2,
\qquad p=5+4d.
\tag{KUAN.19}
$$

Thus its exact negative full-proposal contribution is bounded by

$$
-\mathbb E_m[mA_{\rm rec}^A]
\le-mk_{\rm key}W_A^p+\frac{E_{\max}}{NM}.
\tag{KUAN.20}
$$

Under a current survivor law or QSD with valid floor $a$, this retains
the population-independent original pressure with the explicit survival
factor and finite-particle corrections

$$
\mathbb E[mA_{\rm rec}^A]\ge
\frac a2 k_{\rm key}\mathbb E W_A^p
-\frac a2k_{\rm key}B_x^{2p}e^{-c_*aN}
-\frac{E_{\max}C_{{\rm inv},1}}{N^2}.
\tag{KUAN.21}
$$

The signed incoming donor terms in (KUAN.18) remain present. For
example their explicit coarse bounds are
$D_{\rm live}^A\le W_A/\kappa_C$ and
$D_{\rm rev}^N\le(1-m)W_A/\kappa_C$; signs and actual block
fluxes are preferable to these coarse envelopes.

*Proof.* Compute the output second moment about the actual entering
alive center $m_A$. Alive rows contribute exactly
$m(W_A-A_{\rm rec}^A+D_{\rm live}^A)$; mandatory revivals contribute
$D_{\rm rev}^N$; all accepted-row Gaussian jitters contribute
$d\sigma_J^2\bar p$. The mean output displacement from $m_A$ is
$\bar t$. Conditional row independence gives the empirical
barycenter variance $N^{-2}\sum_i\sigma_i^2$. Subtracting both
mean-square terms proves (KUAN.18).

For alive rows the actual fitness statistics and both companion
roles use only the frozen alive pool. Adding dead revival rows
does not change that live measurement/acceptance law. Apply
`cor-slkd-one-population-pressure` and the discharged source theorem
with population size $M$, retaining the original constants, to obtain
(KUAN.19). At $M=1$, $W_A=0$ and the same lower bound is valid.
Multiplication by $m=M/N$ proves (KUAN.20); its correction is
$E_{\max}(N/M)/N^2$. The current inverse alive-mass bound controls
its expectation. On $M/N\ge a/2$, $mW_A^p\ge(a/2)W_A^p$.
On its complement $mW_A^p\ge0$ and $W_A\le B_x^2$, while its
probability is at most $e^{-c_*aN}$. This proves (KUAN.21) without
replacing a correlated expectation by a product. The incoming-column
bounds used earlier prove the two coarse donor bounds.
$\square$
:::

(sec-kuan-phase-flux)=
## Frozen well labels, signed between-well transfer and Gaussian crossings

:::{prf:definition} Phase partition and actual accepted block moments
:label: def-kuan-phase-blocks

Fix the actual measurable well/basin partition $(B_a)_{a\in\mathcal B}$
of $D$ and the exterior location cell $B_\dagger=\mathbb R^d\setminus D$.
For native Rastrigin the cells may be the Cartesian products of the
actual barrier-root intervals truncated by $D$, as computed in the
landscape phase record. A location label $\dagger$ at preparation
does not mark a slot dead: terminal classification still occurs only
after kinetics. Frozen alive labels determine $A_a=\{i\in A:x_i\in B_a\}$,
$M_a=|A_a|$, $\pi_a=M_a/M$, means $m_a$ and variances $W_a$.
Zero-mass phase terms are defined as zero. Put $d_a=m_a-m_A$.
Then the exact alive law of total variance is

$$
W_A=\sum_a\pi_aW_a+\sum_a\pi_a|d_a|^2
=W_{\rm within}+W_{\rm between}.
\tag{KUAN.22}
$$

For every frozen accepted live edge block, define

$$
B_{ab}=\frac1M\sum_{i\in A_a,j\in A_b}b_{ij},\quad
R_{ab}=\frac1M\sum_{i\in A_a,j\in A_b}b_{ij}|x_i-m_a|^2,
$$

$$
D_{ab}=\frac1M\sum_{i\in A_a,j\in A_b}b_{ij}|x_j-m_b|^2,
\quad L^R_{ab}=\frac1M\sum_{i\in A_a,j\in A_b}b_{ij}(x_i-m_a),
\quad L^D_{ab}=\frac1M\sum_{i\in A_a,j\in A_b}b_{ij}(x_j-m_b).
$$

The actual signed donor-minus-recipient expression is exactly

$$
D_{\rm live}^A-A_{\rm rec}^A
=\sum_{a,b}\left[
D_{ab}-R_{ab}+2d_b\cdot L^D_{ab}-2d_a\cdot L^R_{ab}
+B_{ab}(|d_b|^2-|d_a|^2)\right].
\tag{KUAN.23}
$$

The within-well first moments cannot be dropped: acceptance weights
are not uniform within a well, so the weighted centered sums need
not vanish. For revival, use the same donor block second/first
moments with normalization $1/N$ and incoming mass
$B_{{\rm rev},b}=N^{-1}\sum_{i\notin A,j\in A_b}P_C(j\mid i)$.
This gives

$$
D_{\rm rev}^N=\sum_b
[D_{{\rm rev},b}+2d_b\cdot L_{{\rm rev},b}
+B_{{\rm rev},b}|d_b|^2].
\tag{KUAN.24}
$$

For the paired Chapter 3 coordinate, put
$I_{ab}=\{i\in I:x_i\in B_a,\widetilde x_i\in B_b\}$,
$\zeta_i=(x_i-m_a)-(\widetilde x_i-\widetilde m_b)$,
$w_{ab}=|I_{ab}|/N$ and
$l_{ab}=N^{-1}\sum_{i\in I_{ab}}\zeta_i$. Then

$$
W=\sum_{a,b}\left[
\frac1N\sum_{i\in I_{ab}}|\zeta_i|^2
+w_{ab}|d_a-\widetilde d_b|^2
+2(d_a-\widetilde d_b)\cdot l_{ab}\right].
\tag{KUAN.24a}
$$

The original two-swarm pressure is still
$\chi_*(W-W_0)-B_*/N^2$ from (KUAN.15d), with this exact within-well,
between-well and weighted cross decomposition of its centered $W$.
The $l_{ab}$ term need not vanish on a common-label or phase block.
No term in the raw centroid difference has been added to $W$.

The one-population Keystone pressure in (KUAN.19)--(KUAN.21) remains
$k_{\rm key}(W_{\rm within}+W_{\rm between})^p$.
Equations (KUAN.23)--(KUAN.24) determine its net effect with the
actual between-well incoming mass. They do not assert collapse of
different attainable phase populations.
:::

:::{prf:lemma} Explicit accepted flux and mandatory revival bounds
:label: lem-kuan-phase-mass

For actual retained phase fitness bands
$F_a^-\le F_i\le F_a^+$, computed as the phase minima/maxima of
the original globally standardized sampled array, set

$$
\alpha_{ab}^-=\min\left\{1,
\frac{(F_b^--F_a^+)_+}{s_c(F_a^++\epsilon_c)}\right\},\quad
\alpha_{ab}^+=\min\left\{1,
\frac{(F_b^+-F_a^-)_+}{s_c(F_a^-+\epsilon_c)}\right\}.
$$

For $M\ge2$, the actual accepted blocks obey

$$
\frac{\kappa_C\alpha_{ab}^-M_a(M_b-\mathbf1_{a=b})}{M(M-1)}
\le B_{ab}\le
\min\left\{\pi_a,
\frac{\alpha_{ab}^+M_a(M_b-\mathbf1_{a=b})}{\kappa_CM(M-1)}\right\}.
\tag{KUAN.25}
$$

For every $M\ge1$ mandatory revival satisfies

$$
(1-m)\kappa_C\pi_b\le B_{{\rm rev},b}
\le(1-m)\min\{1,\pi_b/\kappa_C\},\qquad
\sum_bB_{{\rm rev},b}=1-m.
\tag{KUAN.26}
$$

All donor probabilities here use the actual retained dead features
and actual Gaussian normalizers, not uniform revival. These bands
are evaluated after sampling; outer averaging uses the complete
measurement-vector law.

*Proof.* The exact accepted gate lies between its two band values.
Each distinct alive donor probability lies between
$\kappa_C/(M-1)$ and $1/[\kappa_C(M-1)]$. Count the eligible
ordered pairs, retain their self-exclusion factor, and use total
recipient mass at most one to obtain (KUAN.25). Each dead donor
law has denominator between $\kappa_CM$ and $M$; summing its
$M_b$ eligible phase donors gives (KUAN.26). Revival is mandatory,
so its total incoming mass is exactly $(N-M)/N$ regardless of the
actual frozen fitnesses. $\square$
:::

:::{prf:theorem} Exact clone-jitter well crossing and proposal phase flux
:label: thm-kuan-jitter-phase-flux

For $\sigma_J>0$ define

$$
J_c(x)=\Pr(x+\sigma_J\zeta\in B_c),\qquad
Q_c(x)=\mathbb E[(x+\sigma_J\zeta)\mathbf1_{B_c}],\quad
S_c(x)=\mathbb E[|x+\sigma_J\zeta|^2\mathbf1_{B_c}].
$$

For a box $B_c=\prod_r[\ell_{c,r},u_{c,r}]$,

$$
J_c(x)=\prod_r[\Phi((u_{c,r}-x_r)/\sigma_J)
-\Phi((\ell_{c,r}-x_r)/\sigma_J)],
\tag{KUAN.27}
$$

and exterior moments equal the unrestricted Gaussian moments minus
the finite sum over well boxes. The donor's exact jitter crossing
probability is $1-J_{c(x)}(x)$, with $c(x)$ its frozen well label.
No donor position is replaced by its stable root or orbit center.
If all coordinate margins to its well boundary are at least $u>0$,
the explicit upper bound is $\min\{1,2d\Phi(-u/\sigma_J)\}$.

The actual three-label live flux (recipient well, donor well,
jittered output well) is

$$
F^{\rm live}_{a,b,c}=\frac1N
\sum_{i\in A_a,j\in A_b}b_{ij}J_c(x_j),\qquad
F^{\rm rev}_{b,c}=\frac1N
\sum_{i\notin A,j\in A_b}P_C(j\mid i)J_c(x_j).
\tag{KUAN.28}
$$

The exact expected proposal phase mass is

$$
\omega_c=\frac1N\sum_{i\in A}(1-p_i)\mathbf1_{x_i\in B_c}
+\sum_{a,b}F^{\rm live}_{a,b,c}+\sum_bF^{\rm rev}_{b,c},
\qquad\sum_c\omega_c=1.
\tag{KUAN.29}
$$

Its truncated first and second moments are given by replacing
$J_c(x_j)$ in (KUAN.28)--(KUAN.29) by $Q_c(x_j)$ and $S_c(x_j)$,
and the persistent indicators by their $x_i$ or $|x_i|^2$ multiple.
Write these totals as $T_c,H_c$. For $\omega_c>0$ set
$\mu_c=T_c/\omega_c$ and
$V_c^{\rm well}=H_c/\omega_c-|\mu_c|^2$.
The proposal's conditional mixture variance decomposes exactly as

$$
W_{\rm mix}=\sum_c\omega_c V_c^{\rm well}
+\sum_c\omega_c|\mu_c-\sum_bT_b|^2.
\tag{KUAN.30}
$$

The actual expected full empirical variance is

$$
\mathbb E W_N(S^C)=W_{\rm mix}-\frac1{N^2}\sum_i\sigma_i^2,
\tag{KUAN.31}
$$

agreeing exactly with the signed alive-centered balance (KUAN.18).
Mixture well variances in (KUAN.30) must not be identified with
expected empirical within-well variances, whose random counts and
centers produce extra corrections. For any specified time-dependent
orbit centers $r_c(n+1)$, the exact wellwise error is
$\omega_c[V_c^{\rm well}+|\mu_c-r_c(n+1)|^2]$;
the actual orbit displacement remains present, and no force-zero
condition is assumed.

*Proof.* Conditional on the actual accepted donor, the recipient
position is its frozen donor position plus its own independent
Gaussian jitter. Integrate its well indicator to obtain (KUAN.27)
and (KUAN.28), then add persistence to obtain (KUAN.29). Gaussian
independence across coordinates gives the box product. Exiting a
box requires at least one coordinate tail event; union bounding
the two tails in each coordinate gives the margin bound.

The same conditional integration gives the first and second
moments. Applying the ordinary probability law of total variance
to the normalized mixture $N^{-1}\sum_i\mathcal L(X_i^C)$ gives
(KUAN.30). Its mean is $N^{-1}\sum_i\mathbb EX_i^C$; the empirical
mean fluctuates with variance trace $N^{-2}\sum_i\sigma_i^2$,
because the proposal position rows are independent conditional on
the complete sampled fitness vector. Subtract that variance to
obtain (KUAN.31). Collision velocities may be correlated through
Haar rotations, but do not enter this positional independence
statement. Expansion around $r_c(n+1)$ gives the final identity.
$\square$
:::

:::{prf:lemma} Exact empirical within-well and between-well corrections
:label: lem-kuan-empirical-phase-variance

Continue to condition on the complete sampled fitness vector and
consider the complete proposal positions before kinetics. Let
$p_{ic}=\Pr(X_i\in B_c)$, $t_{ic}=\mathbb E[X_i\mathbf1_{B_c}]$ and
$h_{ic}=\mathbb E[|X_i|^2\mathbf1_{B_c}]$, computed from the actual
persistence and donor arrays using (KUAN.27)--(KUAN.29), without
the factor $1/N$ on an individual row. For $p_{ic}>0$ set
$\mu_{ic}=t_{ic}/p_{ic}$ and
$v_{ic}=h_{ic}/p_{ic}-|\mu_{ic}|^2$. The proposal phase assignment
$\mathbf c=(c_1,\ldots,c_N)$ has exact probability

$$
q(\mathbf c)=\prod_{i=1}^Np_{i,c_i}.
\tag{KUAN.31a}
$$

For each positive-probability assignment put
$I_c=\{i:c_i=c\}$, $n_c=|I_c|$,
$\bar\mu_c=n_c^{-1}\sum_{i\in I_c}\mu_{i,c}$ on nonempty groups,
and $\bar\mu=N^{-1}\sum_i\mu_{i,c_i}$. The exact expected empirical
within-well variance, conditional on this assignment, is

$$
\mathbb E[W_{\rm within}^N\mid\mathbf c]
=\frac1N\sum_{c:n_c>0}\left[
\left(1-\frac1{n_c}\right)\sum_{i\in I_c}v_{i,c}
+\sum_{i\in I_c}|\mu_{i,c}-\bar\mu_c|^2\right].
\tag{KUAN.31b}
$$

The exact expected empirical between-well variance is

$$
\mathbb E[W_{\rm between}^N\mid\mathbf c]
=\sum_{c:n_c>0}\frac{n_c}N|\bar\mu_c-\bar\mu|^2
+\sum_{c:n_c>0}\left(\frac1{Nn_c}-\frac1{N^2}\right)
\sum_{i\in I_c}v_{i,c}.
\tag{KUAN.31c}
$$

Their unconditional values are the finite sums of these two
expressions weighted by $q(\mathbf c)$. Empty groups contribute
zero. These expressions include every random-count and barycenter
correction. Their sum is exactly (KUAN.31); all count factors are
at most one, without a minimum phase mass or inverse alive fraction.
No independent-phase assumption is made for the later full kinetic
positions, which share their actual component Haar rotations.

*Proof.* Proposal positions are independent conditional on the full
sampled fitness vector. Conditioning independently on the individual
phase events preserves this independence and gives row means
$\mu_{i,c_i}$ and variance traces $v_{i,c_i}$. On a group of size
$n_c$, subtract its empirical mean square from its total second
moment. The variance trace of that empirical mean is
$n_c^{-2}\sum_{i\in I_c}v_{i,c}$; this gives (KUAN.31b).
For (KUAN.31c), the between-well expression is
$\sum_c(n_c/N)|\bar X_c|^2-|\bar X|^2$.
Its first term has variance contribution
$\sum_c(Nn_c)^{-1}\sum_{i\in I_c}v_{i,c}$, while the second
has variance contribution $N^{-2}\sum_i v_{i,c_i}$.
Subtracting proves (KUAN.31c). Average the independent phase
assignment law (KUAN.31a). The ordinary sample law of total
variance gives equality of their sum to (KUAN.31). $\square$
:::

(sec-kuan-quantitative-scope)=
## Quantitative scope and explicit reference coefficients

:::{prf:remark} Population-independent coefficients and phase-specific interpretation
:label: rem-kuan-scope

For the unchanged $d=3$ reference, direct evaluation gives

$$
1+\kappa_C^{-1}=55.59815003314424,\quad
4B_x^2+d\sigma_J^2=48.03,\quad
2\ell_C/\kappa_C=33.11545195869232,\quad
4/\kappa_C=218.39260013257697,
$$

$$
2(1+2/\kappa_D)=220.39260013257697,\quad
R_s^2=31.98868829132423,\quad
L_{\rm rec}+\kappa_C^{-1/2}L_{\rm don}=938.8117287201932.
$$

The original exact source plans, signed donor block moments and
Gaussian well probabilities can be much sharper than these upper
envelopes. Every displayed multiplier is independent of population
size. Small alive pools no longer create a positional $N/M$
amplification because their donor discrepancy has its correct alive
normalization. The remaining component sensitivity uses the actual
current-time inverse alive mass, with correlations controlled by
Holder rather than discarded.

At the reference global landing floor, the $r=2$ inverse-moment
component coefficient in (KUAN.16) has logarithm about
$158.581943786$, so that particular coarse graph estimate is not
a practical rate certificate. It can be replaced by the exact
finite-plan component response or by a better landing floor that
has actually been computed from the declared phase and its excursion
law. No minimum alive fraction, spectral gap, fitness separation,
zero between-well flux or eventual contraction has been postulated.

This record supplies population-independent complete preparation
and current-survivor bounds while preserving the original global
Keystone pressure and its $N^{-2}$ correction after inverse-mass
averaging. Full kinetic, cap, terminal and orbit residual balances
must be composed with the same phase fluxes. These estimates do
not force different attainable phases to synchronize and do not,
by themselves, identify a full-swarm QSD eigenfunction ratio or a
global entropy constant.

For the initialized moving population target, Chapter 19's
{prf:ref}`thm-cg-mf-kinetic-limit`,
{prf:ref}`thm-cg-mf-row-kinetic-limit` and
{prf:ref}`cor-cg-mf-full-update` already give the fixed-horizon
consistency port for both configured viscous normalizations on their
stated quadratic reference. They assume the prescribed selection
consistency at the initial law, positive initial alive mass,
post-collision $W_4$ convergence, bounded collision velocities and
a uniform expected position moment of order $p>4$. The bounded-donor
Gaussian preparation above supplies the latter moment and velocity
inputs. Their target is the actual initialized recursion
$\mu_{n+1}=\mathcal F_h^\nu(\mu_n)$; uniqueness of this recursion
does not require different population phases to converge together.

This qualitative port must be distinguished from the quantitative
bounded-test consistency required by Chapter 6a's growing-horizon
estimate. Chapter 19's
{prf:ref}`thm-cg-mf-kinetic-variance` proves the explicit count-mode
$N^{-1}$ kinetic variance for bounded tests that are Lipschitz in
velocity, conditional on the entire prepared array. It retains the
dense B2 influence through an independent-innovation resampling
calculation and leaves the separate preparation variance present.
It does not state an arbitrary bounded-measurable-test $G/N$ estimate
for both modes. The row consistency proof retains the exact degree
$a_{L_N}(x_i)-1/N$ and derives its local lower bound from target
moments; it does not assume independent coupled B2 outputs. Thus
the old zero-viscosity $G/N$ constant cannot be imported into the
viscous growing-horizon recursion solely from (KUAN.10)--(KUAN.16).
An explicitly proved full-update quantitative consistency estimate
is still needed for that particular rate conclusion.
:::
