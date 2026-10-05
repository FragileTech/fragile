# Original preparation brackets and stationary population nondegeneracy

(sec-npb-register)=
## 1. Primitive strict covariance inside the complete downstream bracket

:::{prf:definition} Original preparation-bracket register
:label: def-npb-register

Retain every complete execution parameter and original restriction of
{prf:ref}`def-npl-register`, including its global sampled standardizers,
current donor laws, whole accepted components and shared Haar variables.
The executed color covariance statements use $d=3$.
The sampling formulas remain valid in general configured dimension.
The positive covariance conclusions below use the proved stationary
count phase and its actual compact limit $\Theta_*$ from
{prf:ref}`lem-nqg-compact-preparation`. Additional primitive component
conclusions use the evaluated $\varrho<1$ of (NPL.1).
No stationary Gaussian preparation, independent complete rows,
unknown bracket limit or linear hard-boundary profile is inserted.

For the vector of ORIGINAL diagonal projector tests use
$A_a=e_ae_a^\dagger$, $1\le a\le d$, with the same $C^2$ passive
force taper of {prf:ref}`lem-npl-projector-hessian`.
Thus $\psi_a(z,F)=\chi(|F|)P_{aa}(z,F)$, retaining the matched
clone-deletion and terminal alive masks. Its global derivative budgets
remain those already derived. Define the explicit availability
profiles $L_j(r)$ of (NMG.27) in the same mean-field preparation
$\Lambda_{\Theta_*}$, with its original standard jitter marginal.
Suppose the evaluated primitive test is

$$
L:=L_j(r)>2\delta_c\quad\text{for some }j,r>0,
\qquad G_{\rm live}<1.
\tag{NPB.1}
$$
Use the exact first-kick convex bound $U_*\le V_c$ in (NMG.27).
Set

$$
\begin{gathered}
P_0=\alpha R_D+tV_c,\qquad
R_z=P_0/t+r,\qquad M_z=c(V_c+t\lambda R_D),\\
Z_1=cV_c+ct\lambda(R_D+\sigma_Jg_{d,1})+qg_{d,1},\qquad
L_{Fz}=\nu[1+t\ell_\rho(Z_1+R_z+1)],\\
\varepsilon_z=\min\{1,(L-2\delta_c)/(2L_{Fz})\},\qquad
R_y=P_0+t(R_z+1),\qquad r_D=\min\{1,L_D/2\},\\
p_z=v_d\varepsilon_z^d(2\pi q^2)^{-d/2}
 \exp[-(R_z+\varepsilon_z+M_z)^2/(2q^2)],\\
p_a=v_dr_D^d(2\pi s^2)^{-d/2}
 \exp[-(R_y+r_D)^2/(2s^2)],\qquad
p_d=v_d(2\pi s^2)^{-d/2}
 \exp[-(R_y+L_D+3)^2/(2s^2)],\\
c_{\rm down}=m_0(1-G_{\rm live})p_zp_ap_d/d>0.
\end{gathered}
\tag{NPB.2}
$$
Every term uses an original configured parameter or a proved primitive
profile, and every probability lower bound is for the actual unbounded
Gaussian sources.
:::

:::{prf:theorem} Primitive nonzero full stationary population color covariance
:label: thm-npb-positive-downstream

Under (NPB.1), the exact complete downstream covariance matrix of
(NPL.16) for these $d$ projector tests has

$$
\operatorname{tr}V_{\rm down}(\Theta_*)
 \ge\operatorname{tr}V_{\rm mark}(\Theta_*)\ge c_{\rm down}>0.
\tag{NPB.3}
$$
For the original actual full stationary raw and Doob updates,

$$
\liminf_{N\to\infty}
 N\sum_{a=1}^d\operatorname{Var}(H_{a,N})\ge c_{\rm down}.
\tag{NPB.4}
$$
Together with the already proved complete upper covariance budget this
gives a nonzero, finite population fluctuation scale for an actual
vector of matched color projector observations.
Every subsequential preparation-plus-Gaussian population limit from
{prf:ref}`cor-npl-stationary-convolution` consequently has a nonzero
Gaussian contribution. A Gaussian law for its whole preparation is
not required for this conclusion.
:::

:::{prf:proof}
An entering alive row remains uncloned with conditional probability at
least $1-G_{\rm live}$. Dead rows revive and have $I=1$.
The actual admitted stationary population alive fraction is at least
$m_0$. Integrating its original sampled fitness and donor variables
therefore gives
$\Theta_*(I=0)\ge m_0(1-G_{\rm live})$.
This uses neither an independence of row statuses and fitness nor an
independence of accepted components.

For every such uncloned own row, $X=Y$, $|Y|\le R_D$ and
$|U|\le V_c$. Its original A1 position $p=\alpha Y+tU$ has
$|p|\le P_0$; its force-input velocity is exactly
$z=c(U-t\lambda Y)+q\xi$ with Gaussian mean norm at most $M_z$.
The count availability proof of
{prf:ref}`lem-nmg-primitive-availability` chooses an explicit point
$z_0=-p/t+ru$ with $|z_0|\le R_z$ and
$|F^{\rm pop}(p+tz_0,z_0)|\ge L$.
This point can depend on the own preparation; every following bound
is uniform over that dependence.

Differentiation of the original population count force gives on
$|z-z_0|\le1$ the Lipschitz budget $L_{Fz}$ in (NPB.2): its direct
velocity term costs $\nu$, and its spatial kernel derivative costs
$t\nu\ell_\rho(Z_1+|z|)$.
Thus the radius-$\varepsilon_z$ ball about $z_0$ has force magnitude
strictly greater than $2\delta_c$, and the declared taper is one.
Its actual conditional Gaussian $z$-probability is at least $p_z$,
by the density lower bound in (NPB.2).

Throughout that ball $|y|=|p+tz|\le R_y$.
The terminal Gaussian can land in the interior ball $B(0,r_D)\subset D$
with probability at least $p_a$.
It can land in $B((L_D+2)e_1,1)\subset D^c$ with probability at least
$p_d$. The same actual independent final source consequently has
$a_s(y)(1-a_s(y))\ge p_ap_d$ on this event.
For every rank-one unit projector,
$\sum_a P_{aa}^2\ge(\sum_aP_{aa})^2/d=1/d$.
Insert these facts and the uncloned mass into the exact mark covariance
(NPL.16) to prove (NPB.3).

For finite $N$, condition on the entire original preterminal array.
The conditional variance matrix of the actual empirical color vector
is exactly its original independent-terminal Bernoulli matrix divided
by $N$, including every matched clone mask.
The remaining covariance matrix in the conditional variance
decomposition is positive semidefinite.
Its expected mark matrix converges to (NPL.16) by (NPL.21) and the
proved compact preparation limit. Hence (NPB.4) holds for the raw law.
The full stationary Doob record differs by exponentially small TV;
the observations are bounded and its scaled covariance discrepancy is
$O(N\tau_N)\to0$. The same lower bound holds there.
The other covariance contribution in (NPL.16) is positive semidefinite,
so (NPB.3) concerns the COMPLETE downstream bracket as stated.
:::


(sec-npb-normalizers)=
## 2. Exact sampled-normalizer and measured-fitness preparation brackets

:::{prf:definition} Original independent measurement block and its redundant moments
:label: def-npb-measurement

Freeze an original entering state $S\in H_N$.
Let $A_N=\{i:a_i=1\}$, $M_N=|A_N|$ and $a_N=M_N/N\ge m_0$.
Write $Q_i^D$ for the actual current distance-donor law at row $i$,
including its original feature map, width/tag, alive eligibility and
self exclusion. Its independently sampled original donor is $D_i$.
The shifted measured diversity is exactly
$S_i=\mathfrak s(S_i^{\rm state},S_{D_i}^{\rm state})\in[0,S_b]$,
where $\mathfrak s$ is the configured regularized distance followed by
its constant shift. The shift does not change its standardized score.
Define

$$
q_i=(S_i,S_i^2),\quad \bar q_i=E_{Q_i^D}q_i,\qquad
\widehat m=N_A^{-1}\sum_{i\in A_N}q_i,\quad
\bar m=N_A^{-1}\sum_{i\in A_N}\bar q_i,
\qquad N_A=M_N.
\tag{NPB.5}
$$
The two moment coordinates come from the SAME original sampled donor;
they are not two independent sources.
The actual global diversity mean, variance and scale are

$$
\widehat m_1,\qquad \widehat m_2-\widehat m_1^2,\qquad
\widehat\sigma=(\widehat m_2-\widehat m_1^2+\sigma_s^2)^{1/2}.
\tag{NPB.6}
$$
The original alive reward statistics are deterministic once $S$ is
frozen. Their actual values and $\sigma_r$ remain in the reward factor.
For each live row define its actual fitness, with the original positive
logistic-power maps, as the function

$$
f_i(s;m)=\mathcal R_r(z_{r,i})
 \mathcal R_s\left(\frac{s-m_1}
                {\sqrt{m_2-m_1^2+\sigma_s^2}}\right).
\tag{NPB.7}
$$
Thus its original measured fitness is $f_i(S_i;\widehat m)$.
All moment segments below remain in the convex admissible moment set
$0\le m_1\le S_b$, $m_1^2\le m_2\le S_b^2$.
:::

:::{prf:theorem} Derived complete measurement-moment Gaussian bracket
:label: thm-npb-normalizer-clt

Conditional on the original entering $S$,
$\sqrt N(\widehat m-\bar m)$ is a centered independent-row sum with
EXACT covariance

$$
\mathsf Q_N(S)=\frac N{M_N^2}\sum_{i\in A_N}
        \operatorname{Cov}_{Q_i^D}(S_i,S_i^2),\qquad
\operatorname{tr}\mathsf Q_N\le (S_b^2+S_b^4)/m_0.
\tag{NPB.8}
$$
For every real two-vector $\vartheta$ its characteristic function differs
from $e^{-\vartheta^T\mathsf Q_N\vartheta/2}$ by at most

$$
\frac{|\vartheta|^3K_q^3}{6\sqrt N}
       +\frac{|\vartheta|^4K_q^4}{8N},\qquad
K_q=2\sqrt{S_b^2+S_b^4}/m_0.
\tag{NPB.9}
$$
No distributional regularity of the original measured distance is
required; its actual discrete donor distribution is used.
For any sequence of admitted input empirical laws $\mu_N\to\mu$
in the bounded marked metric, the existing continuous bounded feature
and positive current donor kernels give

$$
\mathsf Q_N(S)\longrightarrow
\mathsf Q(\mu)=\frac1{\mu(a=1)^2}
  \int a(z)\operatorname{Cov}_{Q^D_\mu(z)}
        (\mathfrak s(z,Z),\mathfrak s(z,Z)^2)\,d\mu(z).
\tag{NPB.10}
$$
In particular the actually proved stationary chaos yields the
DETERMINISTIC $\mathsf Q(\mu_*)$. This is the derived chronological
bracket of the original measurement-moment block.
:::

:::{prf:proof}
Given $S$, the addressed original distance donor draws are independent
by row; their exact distributions are $Q_i^D$.
Consequently (NPB.5) is its centered independent-row sum and its
covariance is (NPB.8). Each $q_i$ has squared norm at most
$S_b^2+S_b^4$; summing its variance gives the displayed trace bound.
Apply the elementary factor comparison (NPL.22) to
$X_i=(N/M_N)\vartheta\cdot(q_i-\bar q_i)$, which has
$|X_i|\le K_q|\vartheta|$, to obtain (NPB.9).

Its donor kernel is the actual positive weight divided by the current
alive weight integral. The latter is at least $\kappa_Dm_0$.
The bounded feature restriction and the regularized distance make its
numerator and each of the four moments in its covariance continuous,
bounded functions of the source and donor coordinates, with the alive
mark retained as a discrete coordinate.
Couple converging source laws and use dominated integration in these
actual kernels. Self exclusion changes these normalized moments by
at most their explicit bounded range times $2/(\kappa_Dm_0N)$.
Therefore the empirical covariance sum converges to (NPB.10).
The inherited stationary marked chaos proves the asserted specialization,
including the high-alive exception; no deterministic finite-swarm floor
outside $H_N$ is imposed.
:::

:::{prf:lemma} Primitive first two normalizer and fitness derivatives
:label: lem-npb-normalizer-derivatives

Put

$$
\begin{gathered}
D_z=\sigma_s^{-1}+S_b^2\sigma_s^{-3}
                        +S_b/(2\sigma_s^3),\\
E_z=(1+3S_b)\sigma_s^{-3}
 +(3S_b^3+3S_b^2+3S_b/4)\sigma_s^{-5},\\
T_s=(\eta_r+A_r)^{p_r}\Bigg[
 \frac{p_sA_s}4\max\{\eta_s^{p_s-1},(\eta_s+A_s)^{p_s-1}\}
 +\frac{|p_s(p_s-1)|A_s^2}{16}
       \max\{\eta_s^{p_s-2},(\eta_s+A_s)^{p_s-2}\}\Bigg],\\
L_f=H_sD_z,\qquad E_f=T_sD_z^2+H_sE_z,\\
E_\sigma=\sigma_s^{-1}
 +(S_b^2+S_b+1/4)\sigma_s^{-3}.
\end{gathered}
\tag{NPB.11}
$$
Set $T_s=0$ when $p_s=0$.
The actual $m$-derivatives of $f_i$ obey
$|\nabla_m f_i|\le L_f$, $\|D_m^2f_i\|\le E_f$,
uniformly in the entire measured $s$ range and entering state.
The variance map has Hessian norm two, and the scale map in (NPB.6)
has Hessian norm at most $E_\sigma$.
:::

:::{prf:proof}
For $D=m_2-m_1^2+\sigma_s^2$, the standardized score has derivatives

$$
\partial_1 z=-D^{-1/2}+(s-m_1)m_1D^{-3/2},\qquad
\partial_2 z=-(s-m_1)D^{-3/2}/2,
$$
$$
\partial_{11}z=(s-3m_1)D^{-3/2}
                +3(s-m_1)m_1^2D^{-5/2},\quad
\partial_{12}z=D^{-3/2}/2
             -3(s-m_1)m_1D^{-5/2}/2,\quad
\partial_{22}z=3(s-m_1)D^{-5/2}/4.
$$
Their absolute sums are bounded by $D_z,E_z$ as displayed.
The positive logistic-power factor has first derivative bounded by
$H_s$ and second derivative bounded by $T_s$, because
$|\ell'|,|\ell''|\le1/4$ for the actual logistic map.
The product/chain rule gives $L_f,E_f$.
Differentiating $\sqrt D$ gives Hessian entries
$-D^{-1/2}-m_1^2D^{-3/2}$,
$m_1D^{-3/2}/2$ and $-D^{-3/2}/4$.
Their absolute sum is bounded by $E_\sigma$.
Every segment retains $D\ge\sigma_s^2$, proving the uniform claims.
:::

:::{prf:theorem} Original empirical measured fitness includes its shared normalizer influence
:label: thm-npb-fitness-clt

Let $\widehat F=M_N^{-1}\sum_{i\in A_N}f_i(S_i;\widehat m)$.
Define

$$
\begin{aligned}
\bar D_N&=M_N^{-1}\sum_{i\in A_N}
                    E\nabla_mf_i(S_i;\bar m),\\
B_i&=f_i(S_i;\bar m)+\bar D_N\cdot q_i,\qquad
\mathsf Q_{F,N}=\frac N{M_N^2}\sum_{i\in A_N}
                              \operatorname{Var}(B_i\mid S),\\
C_f&=L_f\sqrt{S_b^2+S_b^4}/m_0
                   +E_f(S_b^2+S_b^4)/(2m_0).
\end{aligned}
\tag{NPB.12}
$$
There is an exact conditional centered expansion

$$
\sqrt N[\widehat F-E(\widehat F\mid S)]
 =\frac{\sqrt N}{M_N}\sum_{i\in A_N}[B_i-E(B_i\mid S)]
                  +R_{F,N},\qquad
E[|R_{F,N}|\mid S]\le2C_f/\sqrt N.
\tag{NPB.13}
$$
Its Gaussian characteristic approximation therefore has variance
$\mathsf Q_{F,N}$ and explicit vanishing error: apply (NPL.22) with
$|X_i|\le2[F^*+L_f\sqrt{S_b^2+S_b^4}]/m_0$, and add
$2|\vartheta|C_f/\sqrt N$.
Along the actually proved stationary input-law limit its covariance
converges to the same native donor integral as (NPB.10), with $B_i$
replaced by its displayed full own-fitness plus global-statistic
influence. Its covariance with (NPB.8) is obtained by their SAME row
products, rather than independent copies.
:::

:::{prf:proof}
Taylor expand each actual $f_i(S_i;\widehat m)$ at $\bar m$.
The averaged Hessian remainder has expectation at most
$E_f(S_b^2+S_b^4)/(2m_0N)$.
Its random averaged first derivative minus $\bar D_N$ has second
moment at most $L_f^2/M_N$. The original moment deviation has second
moment at most $(S_b^2+S_b^4)/M_N$.
Cauchy--Schwarz bounds their correlated product by
$L_f\sqrt{S_b^2+S_b^4}/(m_0N)$.
The retained first-order statistic is precisely the independent row
sum of $B_i$: its global normalizer contribution uses that SAME $q_i$.
Thus the uncentered remainder has $\sqrt N$-scaled $L^1$ bound
$C_f/\sqrt N$; centering costs at most twice this bound.
This proves (NPB.12)--(NPB.13).
The characteristic bound follows directly from (NPL.22).
Continuity of each bounded donor moment, including $\bar D_N$,
is the same verified positive-kernel argument as (NPB.10).
:::


:::{prf:corollary} Evaluated original positive-bracket witness
:label: cor-npb-positive-witness

At the ORIGINAL active count witness (PC.37), keep the exact
$V=V_{\rm crit}/2$, $\nu=.01$, $\delta_c=10^{-12}$ and all original
parameters of {prf:ref}`cor-npf-positive-witness`.
Taking $j=3$, $r=10$ and the convex $U_*=V_c$ gives

$$
\begin{gathered}
L_j(r)\simeq7.12040619750\,10^{-7}>2\delta_c,\qquad
\varepsilon_z\simeq6.12739621573\,10^{-6},\\
\log p_z\simeq-253.9842925381,\quad
\log p_a\simeq-36.2966367143,\quad
\log p_d\simeq-77.7497600114,\\
\log c_{\rm down}\simeq-370.0619339472,\qquad
c_{\rm down}\simeq1.92373002898\,10^{-161}>0,\\
L_f\simeq5.00000000017\,10^{-7},\quad
E_f\simeq5.00008983816\,10^{-13},\quad
C_f\simeq4.12757383137\,10^{-5},\quad
K_q\simeq165.100271573.
\end{gathered}
\tag{NPB.14}
$$
The exact formulas define these bounds; their small or large diagnostic
magnitudes do not change their strict parameter regimes.
This is a derived positive COMPLETE population color covariance lower
bound, rather than a presumed nonzero finite color or a discarded
terminal status contribution.
:::

:::{prf:proof}
Use $q^2=(1-e^{-2})/2$, $s=\rho=1$, $t=1/2$,
$\lambda=4/(1+e^{-1})$, $R_D=2\sqrt3$, $L_D=2$,
$\sigma_J=.1$, $V_c=2V=V_{\rm crit}$,
$m_0=a_0/2$ and the $G_{\rm live}$ of (NPL.5).
For a three-Gaussian,
$p_j=\operatorname{erf}(j/\sqrt2)-\sqrt{2/\pi}j e^{-j^2/2}$.
Insert these in (NMG.27), (NPB.2) and (NPB.11)--(NPB.12), with
$H_s=T_s=1/2$ and $\sigma_s=10^6$.
The displayed logarithms retain all Gaussian ball and terminal
alive/dead factors. In particular the positive lower bound has not
been rounded to zero.
:::


(sec-npb-haar)=
## 3. Entire accepted-component Haar covariance and its Gaussian limit

:::{prf:definition} Native whole-component Haar block
:label: def-npb-haar-block

Freeze the entering state and complete measured fitness array.
Let $\Gamma_N$ be its original accepted forest, including mandatory
revival leaves. Each of its components $C$ uses its single original
independent Haar variable $O_C$, with the configured group and dimension.
Conditional on $\Gamma_N$, its frozen copied source $Y_i$, original
recipient mark $I_i$ and collision map are the actual algorithmic ones:

$$
W_i(O_C)=(Y_i,I_i,w_i(O_C)),\qquad
w_i(O_C)=\bar v_C+\alpha_{\rm col}O_C(v_i-\bar v_C),\quad i\in C.
\tag{NPB.15}
$$
The actual configured component collision tag is retained.
For any fixed bounded continuous compact preparation test $f$, $|f|\le B$,
put

$$
\begin{aligned}
Z_C(O)&=\sum_{i\in C}f(W_i(O)),\qquad
b_C=E_O Z_C,\qquad v_C=\operatorname{Var}_O Z_C,\\
Q_{{\rm Haar},N}&=N^{-1}\sum_Cv_C,\qquad
\widehat f_N=N^{-1}\sum_C Z_C(O_C).
\end{aligned}
\tag{NPB.16}
$$
Each $v_C$ includes all within-component pair covariances.
The test can be vector-valued by polarization.
:::

:::{prf:theorem} Complete Haar bracket concentrates and gives an instantaneous conditional CLT
:label: thm-npb-haar-clt

Use the derived exponential-component parameters $\theta,M_{\rm comp}$
from (NPL.2), in the original $\varrho<1$ regime.
For each frozen original entering state and full measured array,

$$
\operatorname{Var}(Q_{{\rm Haar},N}\mid S,\text{measurements})
\le\frac{3888B^4M_{\rm comp}}{\theta^4N}.
\tag{NPB.17}
$$
Moreover its exact conditional Haar fluctuation satisfies

$$
\begin{aligned}
E_{\Gamma_N}\Bigg|
 E_Oe^{i\vartheta\sqrt N(\widehat f_N-N^{-1}\sum_Cb_C)}
          -e^{-\vartheta^2Q_{{\rm Haar},N}/2}\Bigg|
\le\frac{8|\vartheta|^3B^3M_{\rm comp}}
                 {3\theta^2\sqrt N}
       +\frac{12\vartheta^4B^4M_{\rm comp}}{\theta^3N}.
\end{aligned}
\tag{NPB.18}
$$
When the admitted original input empirical law converges to $\mu$,
the actual sampled-normalizer and marked-component limits give

$$
Q_{{\rm Haar},N}\longrightarrow
Q_{\rm Haar}(\mu)
 =E_{\mathcal C_\mu}\left[
   \frac1{|\mathcal C_\mu|}
     \operatorname{Var}_{O}\!\sum_{i\in\mathcal C_\mu}
                                   f(W_i(O))\right]
                           \quad\text{in probability}.
\tag{NPB.19}
$$
Here $\mathcal C_\mu$ is precisely the original rooted accepted-component
law of {prf:ref}`thm-chaos-rooted-collision-limit`, with its sampled
fitness marks, current weighted donors, complete copying and entire
collision retained.
Consequently the Haar fluctuation converges stably relative to the
original entering state, complete measurement array and accepted forest
to its centered Gaussian with variance $Q_{\rm Haar}(\mu)$.
At the proved stationary phase this covariance is the deterministic
$Q_{\rm Haar}(\mu_*)$.
:::

:::{prf:proof}
Conditional on the entire accepted forest, $Z_C(O_C)-b_C$ are independent
centered blocks, bounded in absolute value by $2B|C|$.
Apply (NPL.22), now with component blocks in place of rows.
Its third-order sum is at most
$8B^3\sum_C|C|^3$, and its squared-variance sum is at most
$16B^4\sum_C|C|^4$.
For every integer $k\ge0$, (NPL.3) gives
$E|\mathcal C_i|^k\le k!M_{\rm comp}/\theta^k$.
The exact counting identities
$\sum_C|C|^3=\sum_i|\mathcal C_i|^2$ and
$\sum_C|C|^4=\sum_i|\mathcal C_i|^3$ give (NPB.18).
The same calculation holds after freezing the complete sampled fitness
because (NPL.3) holds under that exact conditioning.

For the bracket variance, resample one original recipient donor/gate
outcome, conditional on the frozen measured array. Delete its outgoing
edge first. The remaining row choices retain independence.
After exposing its old and new outcomes, every changed component lies
in the union of the remaining components of at most three fixed seeds:
that recipient and the two possible donor endpoints.
The deleted forest is a subgraph of an original forest obtained by
restoring that one independent row choice, so its fixed-seed exponential
moment is still at most $M_{\rm comp}$.
The exposed endpoints are independent of the other row choices.
If $D$ is the number of affected rows, convexity therefore gives

$$
ED^4\le3^3\sum_{\text{three seeds}}E|\mathcal C_{\rm seed}|^4
       \le81\cdot24M_{\rm comp}/\theta^4.
$$
For a partition of these affected rows,
$\sum_Cv_C\le B^2\sum_C|C|^2\le B^2D^2$.
The old and new bracket consequently differ by at most $2B^2D^2/N$.
The independent-outcome resampling inequality over the $N$ original
recipient blocks gives

$$
\operatorname{Var}Q_{{\rm Haar},N}
\le\tfrac12N(4B^4/N^2)ED^4,
$$
which is exactly (NPB.17).
This step retains common Haar covariances through the already integrated
$v_C$; it does not resample its rows separately.

For its mean note the rooted identity
$N^{-1}\sum_Cv_C=N^{-1}\sum_i v_{\mathcal C_i}/|\mathcal C_i|$.
On components of size at most $K$ this is a bounded continuous function
of the entire marked original finite component and its original Haar
integral. The actual sampled marks converge, with their global
standardizers, by (NPB.5)--(NPB.10) and
{prf:ref}`lem-chaos-sampled-marks`.
The complete rooted-exploration theorem
{prf:ref}`thm-chaos-rooted-collision-limit` therefore identifies its
conditional expected empirical mean at this cutoff.
Its omitted rooted tail is at most
$B^2E[|\mathcal C_i|\mathbf1_{|\mathcal C_i|>K}]$, which tends to
zero uniformly by the proved exponential moment.
Thus the mean tends to (NPB.19). Combine with (NPB.17) to get convergence
in probability of its ACTUAL complete Haar bracket.
Finally (NPB.18) permits multiplication by any bounded observable of
its original entering/measurement/forest record, while its bracket has
the deterministic limit just proved. This is the stated stable Gaussian
limit. The inherited stationary law convergence and high-alive defect
supply its actual stationary specialization.
:::

:::{prf:remark} Exact decomposition preserves the still unclosed donor-graph center
:label: rem-npb-preparation-bracket-scope

The covariance in (NPB.19) closes the original entire-component Haar
innovation bracket, and (NPB.8)--(NPB.13) close the original sampled
normalizer/measured-fitness brackets. The centered forest conditional
mean $N^{-1}\sum_Cb_C$, together with its feedback through the entering
stationary empirical state, remains in the complete preparation
fluctuation. It is not assigned a Gaussian law by either conditional
limit. In particular a momentum-preserving linear collision test may
have zero Haar bracket, even with nontrivial Haar component rotations.
Different tests keep their explicit native integrals (NPB.19).
:::


(sec-npb-two-copy)=
## 4. Actual two-copy donor components and the chronological clone bracket

:::{prf:definition} Primitive two-copy component regime
:label: def-npb-two-copy-regime

Choose a proof fraction $a_f\in(m_0,a_0)$ and restrict the PRESENT
conditional calculation to entering states with at least $a_fN$ alive
rows. No deterministic floor is imposed on the chain.
Put $g=G_{\rm live}$ and

$$
\begin{gathered}
C_f^{\rm donor}=1/(\kappa_Ca_f),\qquad
\beta_0=4gC_f^{\rm donor}+2C_f^{\rm donor}(1-a_f)<1,\\
\delta_0=(1-\beta_0)/(4C_f^{\rm donor}),\qquad
D_0^{\rm avoid}=1-C_f^{\rm donor}\delta_0=(3+\beta_0)/4,\qquad
\beta=\beta_0/D_0^{\rm avoid}<1,\\
u_0=(1-\beta)/2,\qquad
\mathfrak a=u_0-\beta(e^{u_0}-1)>0,\qquad
\theta_U=\mathfrak a\delta_0/8,\qquad A_U=e^{2u_0},\\
M_U=e^{\theta_U}
 +(e^{\theta_U}-1)A_U
 \left\{\frac1{1-e^{\theta_U-\mathfrak a/2}}
                  +\frac8{3\mathfrak a\delta_0e}\right\},\\
N_U=\left\lceil\max\{4/\delta_0,
             2C_f^{\rm donor}/D_0^{\rm avoid},2/a_f\}\right\rceil.
\end{gathered}
\tag{NPB.20}
$$
The notation $u_0$ is an auxiliary Chernoff exponent; it does not
change the original viscosity $\nu$.
For the stationary transfer of this higher-alive class below use the
existing RESET $a_x=0$ branch, whose original conditional independent
terminal Gaussian landing probabilities have the primitive lower bound
$a_0$. Its entering raw QSD exception is at most
$\alpha_N^{-1}e^{-2(a_0-a_f)^2N}$ by the original landing variables.
The full Doob exception adds its already proved exponentially small
record-comparison error.
:::

:::{prf:lemma} Exponential component bound for the actual union of two outcome copies
:label: lem-npb-two-copy-component

Given the same original entering state and full sampled fitness array,
draw two independent copies $\Omega,\Omega'$ of ONLY the actual
recipient clone donor/gate outcomes. Keep their common sampled fitness
and input state. Let $\mathcal U$ be the undirected union of their
accepted edges, with no edges for rejected proposals.
For $N\ge N_U$ and every fixed root,

$$
E[e^{\theta_U|\mathcal U(i)|}\mid S,\text{measurements}]\le M_U.
\tag{NPB.21}
$$
The same bound holds after deleting the two outgoing choices of one
specified row. Unlike one original forest, its dead vertices can have
two live donors and can join live components. Those bridges are
explicitly included in $\beta_0$.
:::

:::{prf:proof}
Every live copy has accepted-target subprobability at most $g$ and
every specified live target has subprobability at most
$gC_f^{\rm donor}/N$. Every dead copy always has one live target,
with specified probability at most $C_f^{\rm donor}/N$.
Both independent copies and all recipient rows are independent after
the stated exact conditioning.

Explore its live vertices by testing incoming accepted choices at each
processed live target, and expose the remaining outgoing choices of
newly discovered vertices when needed. Before $\delta_0N$ live targets
have been processed, an unexposed choice has only been conditioned to
avoid a set of at most $\delta_0N$ previous targets. Its conditioning
probability is at least $D_0^{\rm avoid}$.
Thus an incoming live choice hits the current target with probability
at most $gC_f^{\rm donor}/(ND_0^{\rm avoid})$, an incoming dead choice
with probability at most $C_f^{\rm donor}/(ND_0^{\rm avoid})$, and a
remaining live outgoing choice is accepted with probability at most
$g/D_0^{\rm avoid}$.
If the current live vertex was discovered as an incoming source, that
particular outgoing copy is already exposed and consumed; only its
other independent original copy remains. If it was discovered as a
target, both unexposed copies retain only their previous avoidance
restrictions. Already exposed addresses are removed from the candidate
list. These conditional bounds use only avoidance restrictions on ORIGINAL
independent recipient choices. The target set can be exposed adaptively;
given its realized exploration record, the remaining choices retain
these product restrictions. Rejected auxiliary targets can be retained
unexposed in their original conditional law; they are not component edges.

A newly encountered dead vertex has only two live outgoing choices.
One brought it to the current target; its other choice can create at
most one further live vertex. Duplicate targets reduce this count.
For domination count one fictitious live child also when that other
target was already discovered. This explicitly counts every dead bridge
and every dead leaf. Direct live incoming or outgoing hits each add at
most one child. At a processed live vertex the total dominating child
count $Z$ therefore has conditional exponential moment

$$
E[e^{uZ}\mid\text{exploration prefix}]
\le\exp\left[
 \frac{2g+2gC_f^{\rm donor}
                 +2C_f^{\rm donor}(1-a_f)}{D_0^{\rm avoid}}
                       (e^u-1)\right]
\le e^{\beta(e^u-1)}\quad(u\ge0).
\tag{NPB.22}
$$
Indeed there are at most two remaining live outgoing choices,
$2N$ live incoming candidates and $2N(1-a_f)$ dead incoming candidates;
first reveal the incoming-hit INDICATORS for the current target as a
batch, then their target/source details. Conditional on the preceding
exploration prefix these indicators are independent across their
remaining original addresses, and disjoint from the current vertex's
own remaining outgoing copies. Revealing the other copy of a discovered
dead source can only overcount a repeated hit or target. The independent
Bernoulli exponential bounds therefore give the first expression.
Already exposed choices and repeated/newly discovered source labels
only reduce these counts. Incoming choices and that vertex's own
remaining outgoing choices are different original addresses.
Since $C_f^{\rm donor}\ge1$, the last inequality follows from (NPB.20).

An alive root starts one live queue. A dead root has at most two alive
donors and starts at most two queues, plus itself.
Include the fictitious children as extra queue vertices and extend their
offspring by the displayed dominating exponential law.
If its total processed progeny $T$ is at least $k$, the first $k-1$
child counts sum to at least $k-2$. Applying conditional exponential
bounds successively at $u=u_0$ gives, until the exploration cutoff,

$$
P(T\ge k)\le A_Ue^{-\mathfrak a k}.
\tag{NPB.23}
$$
For $0\le u\le1$, $e^u-1\le u+u^2$.
The choice $u_0=(1-\beta)/2$ consequently makes
$\mathfrak a>0$. The actual union component has size at most $1+2T$:
each direct child contributes one live vertex and each dead-mediated
child contributes at most that dead vertex and one live vertex.
The fictitious children dominate dead leaves as well.

Let $J=\lfloor\delta_0N\rfloor\ge\delta_0N/2$.
For $k\le2J$ the component tail is bounded by
$A_Ue^{-\mathfrak a(k-1)/2}$; beyond this range its probability is at
most $A_Ue^{-\mathfrak a J}$.
The integer exponential tail-sum formula at $\theta_U$ bounds its
first part by
$e^{\theta_U}+(e^{\theta_U}-1)A_U/
 (1-e^{\theta_U-\mathfrak a/2})$.
The remaining at most $N$ terms are bounded by
$(e^{\theta_U}-1)A_UNe^{\theta_UN-\mathfrak a\delta_0N/2}$.
Use $\theta_U=\mathfrak a\delta_0/8$ and
$\sup_{x\ge0}xe^{-3\mathfrak a\delta_0x/8}
 =8/(3\mathfrak a\delta_0e)$ to obtain (NPB.21).
Deleting any outgoing choices decreases the union component, so its
stated deleted-row version obeys the same fixed-root bound.
:::

:::{prf:definition} Original clone component-additive conditional mean
:label: def-npb-clone-sum

For the fixed bounded compact test of (NPB.16), integrate only its
original component Haar variable and let

$$
\mathcal B_N(\Omega)=\sum_{C\in\Gamma(\Omega)}b_C,\qquad
F_N(\Omega)=N^{-1/2}
 [\mathcal B_N(\Omega)-E_\Omega\mathcal B_N(\Omega)].
\tag{NPB.24}
$$
The whole measured array and entering state remain frozen.
This is the actual clone/donor preparation fluctuation of the original
Haar conditional mean. It contains the copied source, recipient masks
and complete component collision mean; it is not a sum of independent
rooted output rows.
For $A\subset[N]$ let $\Omega^A$ use $\Omega'$ at rows in $A$ and
$\Omega$ otherwise. Set $\Delta_iF=F(\Omega)-F(\Omega^{\{i\}})$
and $\Delta_iF^A=F(\Omega^A)-F(\Omega^{A\cup\{i\}})$ for $i\notin A$.
Define its EXACT resampling covariance

$$
\mathcal T_N=\frac12\sum_i
 \sum_{A\subset[N]\setminus\{i\}}
 \frac{\Delta_iF_N\,\Delta_iF_N^A}
           {\binom N{|A|}(N-|A|)}.
\tag{NPB.25}
$$
The inner weights sum to one at every $i$.
This analytical resampling is not an additional executed stage.
:::

:::{prf:theorem} Complete conditional clone/donor Gaussian law from its actual local covariance
:label: thm-npb-clone-clt

In the primitive two-copy regime, condition on the original entering
state and full measured array. Let
$\sigma_{{\rm clone},N}^2=\operatorname{Var}_\Omega F_N$.
Then $E\mathcal T_N=\sigma_{{\rm clone},N}^2$ and

$$
\operatorname{Var}_{\Omega,\Omega'}\mathcal T_N
\le\frac{16\cdot4^6\cdot6!\,B^4M_U}{\theta_U^6N}
                    =:\frac{C_T}N.
\tag{NPB.26}
$$
For every real $\vartheta$,

$$
\left|E_\Omega e^{i\vartheta F_N}
       -e^{-\vartheta^2\sigma_{{\rm clone},N}^2/2}\right|
\le\frac{\vartheta^2\sqrt{C_T}}{2\sqrt N}
       +\frac{4|\vartheta|^3B^3M_U}{\theta_U^3\sqrt N}.
\tag{NPB.27}
$$
Thus the original conditional donor-graph fluctuation has a Gaussian
population approximation with its exact derived covariance, uniformly
in every frozen full measured array. Degenerate zero covariances retain
their exact Gaussian limit; nondegeneracy has not been imposed.
:::

:::{prf:proof}
We first record the elementary independent-coordinate covariance identity.
For any square-integrable $F$ and $g(F)$ on a product law,

$$
\operatorname{Cov}(g(F),F)
 =\frac12\sum_i\sum_{A\not\ni i}
 \frac{E[(g(F(\Omega))-g(F(\Omega^{\{i\}})))\Delta_iF^A]}
           {\binom N{|A|}(N-|A|)}.
\tag{NPB.28}
$$
One proof exposes an independent uniform permutation of the coordinates,
telescopes $F(\Omega)-F(\Omega')$ along its successive replacement sets,
and pairs each summand with its version exchanging the original and
resampled $i$th coordinates. Terms measurable without coordinate $i$
then cancel by independence; symmetrization contributes the factor
$1/2$. A given preceding set $A$ occurs with probability
$|A|!(N-|A|-1)!/N!$, which is exactly its displayed weight.
This proves (NPB.28), including the exact identity for its finite original
product law. Taking $g(F)=F$ gives $E\mathcal T_N=\operatorname{Var}F_N$.

A changed row choice affects only original components within its
TWO-COPY union component. Hence
$|\Delta_iF_N|,|\Delta_iF_N^A|
 \le2B|\mathcal U(i)|/\sqrt N$, simultaneously for every $A$.
This retains the entire original Haar conditional mean on each changed
component. No independent-component assertion is made for overlapping
outcome copies.

To resample one primitive coordinate of $(\Omega,\Omega')$, delete both
row-$j$ outgoing choices. Expose its other copy, old choice and new
choice. The affected union components lie in those of at most four fixed
seeds: $j$ and those three possible donor endpoints.
The exposed endpoints are independent of the remaining deleted union.
By (NPB.21), their total size $D$ has
$ED^6\le4^6\cdot6!M_U/\theta_U^6$.
Every summand of (NPB.25) rooted outside this union is unchanged.
For a root in it, each old or new product is bounded by
$4B^2D^2/N$. The factor one-half and unit summed weights therefore
bound the total covariance change by $4B^2D^3/N$.
Apply independent-coordinate resampling to these $2N$ original primitive
variables to obtain (NPB.26).

Finally take $g(F)=e^{i\vartheta F}$ in (NPB.28).
Taylor expansion around $F(\Omega)$ gives

$$
E[F_Ne^{i\vartheta F_N}]
 =i\vartheta E[e^{i\vartheta F_N}\mathcal T_N]+R(\vartheta),\qquad
|R(\vartheta)|\le\frac{\vartheta^2}4
 \sum_i\sum_{A\not\ni i}w_A
                E[|\Delta_iF_N|^2|\Delta_iF_N^A|].
$$
The union-component third moment and its unit weights bound the sum by
$8B^3N^{-1/2}E|\mathcal U(i)|^3
 \le48B^3M_U/(\theta_U^3\sqrt N)$.
For its characteristic function $\phi$, therefore,
$|\phi'(\vartheta)+\vartheta\sigma_{{\rm clone},N}^2\phi(\vartheta)|$
is at most
$|\vartheta|\sqrt{C_T/N}
 +12\vartheta^2B^3M_U/(\theta_U^3\sqrt N)$.
Solve this first-order equation with $\phi(0)=1$.
The Gaussian integrating factor has contraction modulus on the integral
between zero and $\vartheta$, yielding exactly (NPB.27).
This proof uses the actual complete graph conditional covariance;
there is no presumed conditional or stationary Gaussian preparation.
:::


(sec-npb-clone-limit)=
## 5. Derived joint clone/Haar covariance with all sampled type marks retained

:::{prf:definition} Native two-copy rooted covariance integral
:label: def-npb-clone-limit

Let $\mu$ be an admitted input probability with alive mass at least
$a_f$. Its ORIGINAL sampled row type contains the entire entering row,
its original distance-companion measurement, its actual reward statistics
and the population values of ONLY the global normalization moments.
The sampled diversity itself remains its original random type coordinate,
with donor law $Q_\mu^D$.
Use this type law and the original current clone-donor/acceptance kernels
to define the two-copy rooted outcome component $\mathcal U_\mu$:
each source has two independent original outcome copies, sharing that
same sampled row type. Incoming copy addresses have their native
Poisson exploration limit; each incoming child retains its other full
original outgoing copy and all required parent/source type marks.
The limit is finite in (NPB.20), as proved below. This is a limit law of
the existing two-copy resampling calculation, not a replacement collision
algorithm.

Give each NONROOT recipient an independent Bernoulli-$p$ copy label.
The root uses copy zero. The selected actual forest is $\Gamma_p$;
$\Gamma_0$ always selects copy zero.
Switch only the root choice to copy one and define $\Psi_p$ as the
change of the summed original component Haar means $b_C$.
Unchanged components cancel, so this finite functional depends only on
$\mathcal U_\mu$ and its retained marks. Set

$$
Q_{\rm clone}(\mu)=\frac12\int_0^1
                   E[\Psi_0\Psi_p]\,dp,
\qquad 0\le Q_{\rm clone}(\mu)
                    \le4B^2M_U/\theta_U^2 .
\tag{NPB.29}
$$
The integrand may have either sign; its nonnegative total is proved
as the limit of the exact original conditional variance below.
Every donor/gate/collision parameter is consumed by the displayed
original type and outcome laws.
:::

:::{prf:theorem} Actual clone bracket limit and joint full collision-preparation CLT
:label: thm-npb-joint-clone-haar

For any admitted input empirical-law sequence $\mu_N\to\mu$ and its
ORIGINAL sampled full fitness array, the exact conditional clone
variance in (NPB.27) satisfies

$$
\sigma_{{\rm clone},N}^2\longrightarrow Q_{\rm clone}(\mu)
                                      \quad\text{in probability}.
\tag{NPB.30}
$$
For its ACTUAL compact preparation empirical average $\widehat f_N$,

$$
\sqrt N\left\{\widehat f_N
             -E[\widehat f_N\mid S,\text{measurements}]\right\}
\ \Longrightarrow\
\mathcal N(0,Q_{\rm clone}(\mu)+Q_{\rm Haar}(\mu)).
\tag{NPB.31}
$$
Convergence is stable relative to the complete original entering state
and sampled measurement record. Finite vectors use their original
polarized covariances. In the actual stationary reset phase, the higher
alive exception in (NPB.20) tends to zero exponentially and the proved
stationary input law is $\mu_*$; this yields the deterministic displayed
collision-preparation bracket there. No Gaussian stationary entering
state or Gaussian sampled-measurement CENTER has been assumed.
:::

:::{prf:proof}
First note the exact beta-integral identity

$$
\frac1{\binom N{|A|}(N-|A|)}
 =\int_0^1p^{|A|}(1-p)^{N-1-|A|}\,dp\qquad(i\notin A).
$$
Thus the conditional expectation of (NPB.25) is exactly
one-half the uniform-root average of
$\int_0^1E[\Psi_{0,N}\Psi_{p,N}]dp$ for the original two-copy
component calculation. Its selected NONROOT labels are genuinely
independent Bernoulli-$p$ inside this identity; no asymptotic replacement
of its random subsets is needed.

Here are the hypotheses for its full marked exploration limit.
Every incoming specified address has probability at most
$C_f^{\rm donor}/N$, and its two original copies are independent
conditional on that source's frozen full type.
At any fixed finite exploration, simultaneous hits by one primitive
address are impossible; hits by its TWO copies on two specified targets
have probability at most $(C_f^{\rm donor}/N)^2$.
Their sum over source indices is $O(K^2(C_f^{\rm donor})^2/N)$
for a size-$K$ exploration. Label coincidences have the same vanishing
bound. Conditioning an unexposed address to avoid its finitely many
previous targets changes its actual law by at most
$KC_f^{\rm donor}/N$.
The elementary product generating function of the remaining rare
incoming events therefore converges to the claimed marked Poisson
incoming law, with its ACTUAL type-dependent target/gate subprobability.
The other copy of a newly found child retains its full original outcome
law under this exploration. Mandatory dead second donors, including
their resulting bridges, are consequently retained in the limit.

The row-type empirical law converges by the original independent
measurement-row law, the actual bounded global moment convergence
(NPB.8), positive standardization scales and
{prf:ref}`lem-chaos-sampled-marks`.
The current clone kernel has positive denominator at least
$\kappa_Ca_f$; its accepted/rejected SUBPROBABILITIES use the actual
continuous clipped gate. Integrating these subprobabilities avoids any
division by a zero acceptance probability at tied fitness.
Every fixed finite marked exploration thus converges with its original
copying and full Haar conditional means, exactly by the preceding
rare-event products and bounded continuous kernel integrations.
This verifies the two-copy version of the rooted exploration directly;
it has not applied a single-forest theorem to dead bridges.

The uniform exponential union bound (NPB.21) removes the exploration
cutoff. Indeed $|\Psi_{0,N}|,|\Psi_{p,N}|
\le2B|\mathcal U_N(\text{root})|$, and the omitted product tail tends
to zero uniformly in $p$ by that proved exponential moment.
The same moment passes to the limiting rooted law, which is therefore
finite almost surely. Its product bound also gives the upper bound in
(NPB.29), using $E|\mathcal U_\mu|^2\le2M_U/\theta_U^2$.
Dominated integration in $p$ proves (NPB.30) and (NPB.29)'s
nonnegativity: its left side is the EXACT original variance.

For the joint law decompose its centered preparation fluctuation as
$F_N$ of (NPB.24) plus the conditional Haar-centered sum of (NPB.18).
The latter has deterministic limiting covariance by (NPB.19) and its
conditional characteristic error tends to zero in mean. Multiply this
conditional expression by the former's characteristic factor.
Now use (NPB.27) and (NPB.30) for the original donor fluctuation.
This proves (NPB.31), including asymptotic independence of these TWO
innovation blocks derived from their actual chronology.
Multiplication by bounded entering/measurement-record tests gives its
stated stability. Both blocks keep their complete native covariances.

At the actual reset stationary phase, the original terminal landing
variables yield the exception in (NPB.20), and the already proved
stationary input and sampled-mark laws identify $\mu_*$.
Their actual Doob full-record comparison is exponentially small; its
$\sqrt N$ center corrections also vanish because $|\widehat f_N|\le B$.
The same stable conclusion therefore holds for its full stationary
Doob instrument without assigning that instrument independent primitive
noise blocks.
:::


(sec-npb-nonlinear-preparation)=
## 6. Nonlinear original color preparation has the same derived bracket

:::{prf:lemma} Original compact collision preparation has uniform high moments
:label: lem-npb-compact-high-moment

Freeze the original entering state and full measured array.
Let $\Theta_N=N^{-1}\sum_i\delta_{W_i}$ be its actual compact preparation,
$\bar\Theta_N=E[\Theta_N\mid S,\text{measurements}]$.
For every bounded measurable $g$, $|g|\le B_g$, and real $p\ge2$,

$$
\|\Theta_Ng-\bar\Theta_Ng\|_p
\le A_HB_gp^{3/2}/\sqrt N,\qquad
A_H=7\sqrt{M_{\rm comp}}/\theta .
\tag{NPB.32}
$$
For any finite measurable cell partition its SUMMED empirical cell-mass
variance is at most $A_{{\rm cell},H}/N$, where

$$
A_{{\rm cell},H}=40M_{\rm comp}/\theta^2.
\tag{NPB.33}
$$
These are bounds for the whole original donor/Haar preparation; correlated
rows inside accepted components have not been resampled independently.
:::

:::{prf:proof}
Under resampling one donor/gate block, the changed preparation lies in
at most three fixed-seed components of the deleted original forest.
Their sizes $D$ satisfy
$\|D\|_p\le3(M_{\rm comp}\Gamma(p+1))^{1/p}/\theta$.
This follows from (NPL.3)'s tail bound and Minkowski, and it also holds
for real $p\ge2$. Resampling one addressed Haar block affects its single
component, with the same bound without the factor three.
The empirical test change is at most $2B_gD/N$.
Thus its resampling $L^p$ budgets are at most
$6B_gpM_{\rm comp}^{1/p}/(\theta N)$ and
$2B_gpM_{\rm comp}^{1/p}/(\theta N)$, using
$\Gamma(p+1)^{1/p}\le p$ for $p\ge2$.

For a product-law function with chronological differences $D_j$,
conditional Jensen bounds $\|D_j\|_p$ by its original independent
resampling difference norm.
The scalar $L^p$ second derivative inequality gives

$$
\left\|\sum_jD_j\right\|_p^2
                      \le(p-1)\sum_j\|D_j\|_p^2.
$$
Indeed differentiate $\|M+tD\|_p^2$ twice, use Hölder to bound its
second derivative by $2(p-1)\|D\|_p^2$, and use
$E[D\mid\text{prefix}]=0$ to cancel the first derivative at zero.
Iterate over the martingale prefixes.
There are $N$ donor blocks and $N$ addressed Haar blocks. The preceding
budgets give $\sqrt{40}B_gp\sqrt{p-1}M_{\rm comp}^{1/p}/(\theta\sqrt N)$,
which is bounded by (NPB.32).
Unused nonrepresentative Haar addresses remain original independent
latent variables and do not add executed collision noise.

For a cell-mass vector one replacement has summed squared changes at
most $4D^2/N^2$, since its summed absolute change is at most $2D/N$.
The donor affected-count second moment is at most
$18M_{\rm comp}/\theta^2$; the Haar count moment is at most
$2M_{\rm comp}/\theta^2$.
Apply independent-block resampling to each cell, sum first and then
bound. Its total is at most
$2(18+2)M_{\rm comp}/(\theta^2N)$, proving (NPB.33).
:::

:::{prf:definition} Primitive compact nonlinear remainder register
:label: def-npb-nonlinear-budget

Use $\mathcal C_N,S,r_N,Q_N,J_d$ of (NPL.12),
$b_H=(2d+2)^{-1}$, $T_H=\max\{1,R_D,V_c\}$ and
$D_H=2\sqrt{2d}T_H+1$. Set

$$
\begin{gathered}
p_N=\max\{2,\lceil\log[2J_dQ_N(N+1)^4]\rceil\},\qquad
\varepsilon_{H,N}=eA_Hp_N^{3/2}/\sqrt N+4S^4N^{-2},\\
C_{W,H}=\sqrt{2d}T_H
       +D_H\sqrt{2\,3^{2d}A_{{\rm cell},H}},\\
\mathcal E_{H,N}=\mathcal C_N\sqrt N[
 \varepsilon_{H,N}^2+\varepsilon_{H,N}C_{W,H}N^{-b_H}
                                      +(N+1)^{-4}]
         +\mathcal C_N\sqrt{8dN}(N+1)^{-4}\longrightarrow0,\\
\varphi_\Theta(W)=E_{G,\xi}\Phi_{\Lambda_\Theta}(W,G,\xi),\qquad
B_\varphi=A_\Phi+B_\Phi Z_1.
\end{gathered}
\tag{NPB.34}
$$
The original global normalizers and measurement array remain frozen;
$\bar\Theta_N$ is their full original donor/Haar mean measure.
:::

:::{prf:theorem} Complete nonlinear color preparation linearizes without independent components
:label: thm-npb-nonlinear-preparation

For the original downstream functional (NPL.7),

$$
\begin{aligned}
\mathcal T(\Lambda_{\Theta_N})-
                 \mathcal T(\Lambda_{\bar\Theta_N})
 &=\int\varphi_{\bar\Theta_N}(W)
                         d(\Theta_N-\bar\Theta_N)(W)+R_{H,N},\\
E[\sqrt N|R_{H,N}|\mid S,\text{measurements}]&\le\mathcal E_{H,N}.
\end{aligned}
\tag{NPB.35}
$$
The inequality is uniform over the original entering and measured arrays
in the proved component regime. It is a complete primitive comparison
for the actual nonlinear preparation dependence of the color readout.
:::

:::{prf:proof}
The first variation with $\dot\Lambda=\dot\Theta\otimes\gamma_d\otimes\gamma_d$
is exactly (NPL.9) integrated over the original future sources;
its compact influence is $\varphi_\Theta$, with
$|\varphi_\Theta|\le B_\varphi$ by the ROWWISE form of (NPL.10).

For the remainder apply the same first-kick and B2 field expansion as
(NPL.13), now to $\delta\Theta=\Theta_N-\bar\Theta_N$ and its product
with the unchanged future Gaussian laws.
Each fixed base field and first query derivative is a bounded test of
$W$ after integrating its original Gaussian source. On the same finite
query grid, normalize its displayed primitive scale and apply (NPB.32).
Markov at $e$ times that $L^{p_N}$ bound gives probability at most
$e^{-p_N}$; the grid union has failure at most $(N+1)^{-4}$.
Consequently its small-field budget is precisely $\varepsilon_{H,N}$.
The original compact cell SUM variance bound (NPB.33), using width
$T_HN^{-b_H}$, gives

$$
E W_1(\Theta_N,\bar\Theta_N)\le C_{W,H}N^{-b_H}.
$$
The product coupling with the SAME future $G,\xi$ gives this identical
transport bound for their extended measures.
Every mixed variation integral is therefore bounded by its small first
field derivative times this transport discrepancy, exactly as in the
proved full B2 expansion (NPL.13). The quadratic field terms and finite
primitive chain-rule budgets remain (NPL.15).
The future-source radius-$r_N$ comparison and its original tails have
already been bounded uniformly over compact source measures there.
Thus its scaled expected remainder is (NPB.34), proving (NPB.35).
All original donor/component correlations enter (NPB.32)--(NPB.33);
no empirical preparation field was assigned an independent-row law.
:::

:::{prf:corollary} Complete original update after the actual measured array has a derived Gaussian population law
:label: cor-npb-postmeasurement-clt

In the existing stationary reset count phase and the positive primitive
single/two-copy tests, let
$\varphi_* =\varphi_{\Theta_*}$ be the exact bounded compact influence
(NPB.34). Let $\mathcal M_N$ contain the complete original entering state
and all original current distance measurements and fitness standardizers.
For the ORIGINAL full nonlinear color observation (NPF.2),

$$
\sqrt N\{H_N-E[H_N\mid\mathcal M_N]\}
\ \Longrightarrow\
\mathcal N\left(0,
 V_{\rm down}(\Theta_*)+
 Q_{\rm clone}(\mu_*;\varphi_*)+
 Q_{\rm Haar}(\mu_*;\varphi_*)\right).
\tag{NPB.36}
$$
Convergence is stable relative to the entire original $\mathcal M_N$.
This covariance includes actual donor/gate/collision sources, both dense
kicks, unbounded original jitter and OU, matched clone deletion and
terminal alive marks. The full vector of diagonal projector tests has
strict positive covariance trace at least (NPB.2).

The REMAINING instantaneous fluctuation is exactly
$\sqrt N\{E[H_N\mid\mathcal M_N]-EH_N\}$.
It contains the original measured-array influence and stationary entering
population feedback; it is not silently replaced by the measured-fitness
mean CLT (NPB.13).
:::

:::{prf:proof}
The exact conditional donor/Haar mean preparation converges to $\Theta_*$:
convexity of transport and conditional Jensen give
$E W_1(\bar\Theta_N,\Theta_*)\le E W_1(\Theta_N,\Theta_*)\to0$
from the already proved actual compact preparation transport. Thus
$\bar\Theta_N$ converges in probability without an assumed conditional
measurement law.
The original Gaussian integrals in (NPL.8) then give
$\varphi_{\bar\Theta_N}\to\varphi_*$ uniformly on $\mathcal W$.
To verify uniformity, take any converging sequence of compact source
points, couple their identical original Gaussian draws, and use bounded
kernel derivatives plus the original uniform Gaussian moments in both
integrals. Sequential compactness makes this dominated convergence
uniform. Its $B_\varphi$ bound is independent of the measured array.

The clone/Haar CLT proof (NPB.27)--(NPB.31) applies to these bounded
converging tests. Their covariance differences are dominated by the
uniform original component second moment times
$\|\varphi_{\bar\Theta_N}-\varphi_*\|_\infty$, and tend to zero.
Thus (NPB.35), centered by its exact conditional expectation, gives the
original nonlinear preparation Gaussian with the displayed two brackets.
The full downstream CLT (NPL.18)--(NPL.19) is stable relative to the
complete preparation, so it supplies the independent additional bracket
$V_{\rm down}(\Theta_*)$ under the same chronology.
Its conditional center differs from
$\mathcal T(\Lambda_{\Theta_N})$ by a uniformly vanishing scaled $L^1$
error; both centering errors from (NPB.35) also vanish.
This proves (NPB.36), including its full conditioning and native masks.
The higher-alive exception and actual full Doob comparison vanish with
the already proved exponential bounds. Disintegration of that bounded
full-record TV comparison also bounds the averaged conditional-kernel
TV by twice its joint TV; hence the conditional mean-center difference
has scaled $L^1$ norm at most $4B\sqrt N\tau_N\to0$. This justifies
the displayed Doob conditional center as well as its bounded cylinder
transfer.
The positive trace is (NPB.3), a positive semidefinite part of this entire
sum. Finally the remaining conditional mean is the exact variance/martingale
chronological complement written in the statement, which establishes its
scope without declaring that remaining source Gaussian.
:::


(sec-npb-two-copy-witness)=
## 7. Evaluated original active-count two-copy regime

:::{prf:corollary} The original positive stationary witness satisfies the derived two-copy tests
:label: cor-npb-two-copy-witness

Retain the complete active-count parameters of (NPB.14), including
its existing Uniform donor tag, finite $V=V_{\rm crit}/2$ and the
ORIGINAL positive terminal spatial noise. Choose the proof fraction
$a_f=3/4$. The constants of (NPB.20) satisfy

$$
\begin{gathered}
G_{\rm live}\simeq1.160061851188191\,10^{-7},\qquad
C_f^{\rm donor}=4/3,\qquad
\beta_0\simeq .6666672853663206<1,\\
\delta_0\simeq .06249988399381489,\qquad
\beta\simeq .7272732795004243<1,\qquad
u_0\simeq .13636336024978785,\\
\mathfrak a\simeq .03010999630072919,\qquad
\theta_U\simeq .00023523390948122124,\qquad
M_U\simeq1.1823363148479027,\qquad N_U=65,\\
2(a_0-a_f)^2\simeq .002742886398094533,\qquad
\log(C_T/B^4)\simeq67.96667958601432.
\end{gathered}
\tag{NPB.37}
$$
Here the Latin $u_0$ is the auxiliary exponent in (NPB.20).
Every displayed constant is finite and strictly positive. Thus this
ORIGINAL active phase has both the deterministic native clone/Haar
bracket of (NPB.31) and the full nonlinear postmeasurement Gaussian
law (NPB.36), with the primitive positive trace (NPB.14).
The sampled-measurement and entering-state conditional center remains
exactly the separate fluctuation stated there.
:::

:::{prf:proof}
The ORIGINAL $a_0$ is the minimum of the proved no-copy/copy
floors (KU.S8)--(KU.S9), evaluated at the register cap $V_0=.1$.
It retains the OU contribution to $\tau$, the possible kinetic mean
shift $R=bV_c(V_0)$ and the eroded-box copy floor. In particular

$$
\begin{gathered}
\tau^2=1+t^2q^2,\qquad R=2bV_0,\qquad
P_{\rm base}=\operatorname{erf}(L_D/(\sqrt2\tau))^3,\\
a_0=\min\left\{
 \Phi\bigl(\Phi^{-1}(P_{\rm base})-R/\tau\bigr),
 \operatorname{erf}((L_D-R)/(\sqrt2\tau))^3\right\}
 \simeq .7870330015>3/4.
\end{gathered}
$$
The stationary high-alive parameter remains $m_0=a_0/2<3/4$.
Its existing Uniform clone-donor tag has $\kappa_C=1$.
The finite primitive fitness range bound is the exact $G_{\rm live}$
of (NPL.5), evaluated with the ORIGINAL shifted logistic powers and
positive normalizer floors. Insert these quantities into (NPB.20).
The covariance constant is the exact
$C_T=16\cdot4^6\cdot6!B^4M_U/\theta_U^6$ of (NPB.26).
These substitutions give (NPB.37), including the unconditional
stationary high-alive exception; no new floor is imposed on any
realized population. The preceding theorems apply with their same
original tests and conditional centers.
:::
