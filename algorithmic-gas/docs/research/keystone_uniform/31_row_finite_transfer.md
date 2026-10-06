# Conditional finite-particle transfer for the actual row-normalized gas

(sec-rft-register)=
## 1. Actual marked law and source-box constants

:::{prf:definition} Row particle register
:label: def-rft-register

Use the actual row-normalized marked kernel and the positive parameter
interval of {prf:ref}`thm-rpf-marked-population`, with the fixed chosen
box $D_L=[-L,L]^d$ and raw same-potential reward $R(x)=-|x|^2/2$.
Stop the actual finite-array chain at its first all-dead time $\tau_N$.
For the one-step estimates below it suffices that $0\le t\nu\le1$,
the stored speed is capped by $V$, and the entering alive fraction is
at least a fixed $m_f>0$. Population attraction is used only in the
subsequent uniform-time theorem, not in the conditional comparison.

Let $J(\mu)=\mu J_\mu$ be its actual population preparation. Keep
the exact component-Haar original-slot velocity law, alive-only sampled
fitness normalizers, donor/self-exclusion rules, both row viscous kicks,
own Gaussian innovations and the declared canonical nonexpansive cap
$C_V$ of {prf:ref}`def-pvb-record`. Put
$V_c=(1+2|\alpha_{\rm col}|)V$ and $R_D=\sqrt dL$.

Use the marked bounded-transport metric
$$
\mathsf d_m(\mu,\mu')=\inf_{\pi\in\Pi(\mu,\mu')}
\int\min\{1,|z-z'|+\mathbf1_{a\ne a'}\}\,d\pi,
\qquad z=(x,v).
$$
Prepared and noisy-stage laws are unmarked physical phase laws;
$W_4$ uses their Euclidean phase norm. Retain
$\kappa=1/(16d)$, $a=\kappa/8=1/(128d)$,
$J_d=(2+2\sqrt{2d})^{2d}$ and the Gaussian radial moments $g_p$.
All row fields use the original Gaussian kernel
$K(x,X)=e^{-|x-X|^2/(2\rho^2)}$, with
$\ell=1/(\rho\sqrt e)$; no degree floor is inserted.

The conditional preparation-only test MSE constant $G_p$ is the full
alive-floor constant in {prf:ref}`lem-spt-marked-consistency`, with
$m_*=m_f$. In its finite normalizer formulas use the exact alive-support
bounds $R_b=dL^2/2$, $L_R=\sqrt dL$. Dead rewards enter neither a
normalizer nor the mandatory revival gate. Its proof stops before
kinetics and therefore holds in row mode, as in
{prf:ref}`lem-sbst-local-comparison`.

Define
$$
X_L=R_D+\sigma_Jg_8,\quad Z_P=X_L+V_c,\quad
A_P=2+\tfrac12J_d\sqrt{G_p}+2Z_P^2,\quad
D_P=(1+16Z_P^4)\sqrt{A_P}.
\tag{RFT.1}
$$
Let $\sigma=\sigma_J$, and put
$$
\Sigma=a_x^2\sigma^2+t^2q^2,\quad
\Sigma_w=c^2t^2\sigma^2+q^2,\quad
B_y=a_xR_D+bV_c,\quad B_w=ctR_D+cV_c,
$$
$$
Z_O=B_y+\sqrt\Sigma g_8+B_w+\sqrt{\Sigma_w}g_8,
\quad W_L=B_w+\sqrt{\Sigma_w}g_4,
$$
$$
Z_F=B_y+\sqrt{\Sigma+s^2}g_8+V,\quad
A_O=2+\tfrac12J_d+2Z_O^2,\quad
D_O=(1+16Z_O^4)\sqrt{A_O},\quad
A_F=2+J_d+2Z_F^2,
$$
$$
B_1=R_D+\sigma\sqrt{2d},\quad
B_2=B_y+\sqrt\Sigma\sqrt{2d},\quad
b_j(R)=\tfrac12e^{-(R+B_j)^2/(2\rho^2)}\quad(j=1,2),
$$
$$
T_X(R)=\min\{1,2d e^{-((R-R_D)_+)^2/(2d\sigma^2)}\},
$$
$$
T_Y(R)=\min\{1,2d e^{-((R-B_y)_+)^2/(2d\Sigma)}\},\quad
T_W(R)=\min\{1,2d e^{-((R-B_w)_+)^2/(2d\Sigma_w)}\}.
\tag{RFT.2}
$$
Every quantity is an explicit fixed-parameter constant or function. The
source box bounds donor centers only; jitter, OU innovations, stored dead
positions and uncapped $w$ remain unbounded.
:::

:::{prf:lemma} Uniform tails and target bulk degrees
:label: lem-rft-source-tails

For every consistent positive-alive entering array or population law,
its actual prepared position obeys the averaged tail bound $T_X$ and
prepared speed obeys $V_c$. Its joint row stage obeys the averaged tails
$T_Y,T_W$, phase eighth moment at most $Z_O^8$, and uncapped velocity
fourth norm at most $W_L$. Its final marked phase eighth moment is at
most $Z_F^8$. These assertions are conditional on an entering array,
then averaged over its full actual preparation and kinetic innovations;
no independence among prepared rows is required.

Every deterministic target preparation $\eta=J(\mu)$ has
$\eta(|X|\le B_1)\ge1/2$, so its original Gaussian denominator
is at least $b_1(R)$ on $|x|\le R$. Its actual joint stage law
$\Lambda_\eta$ has $\Lambda_\eta(|Y|\le B_2)\ge1/2$, and its
second denominator is at least $b_2(R)$ on $|y|\le R$.
No positive empirical copied fraction is assumed.
:::

:::{prf:proof}
Every prepared source is an alive position in $D_L$, whether it
persists, is copied, or supplies a mandatory revival. Conditional on
the frozen choices,
$X=u+I\sigma Z$ with $|u|\le R_D$, $I\in\{0,1\}$ and its original
independent jitter. The component collision has speed at most $V_c$.
Both original row averages preserve that bound in the first kick when
$0\le t\nu\le1$, giving
$$
Y=a_xu+bU+a_xI\sigma Z+tq\xi,\qquad
W=-ctu+cU-ctI\sigma Z+q\xi,\qquad |U|\le V_c.
$$
The bounded terms have norms at most $B_y,B_w$. The displayed Gaussian
sums have conditional variances at most $\Sigma,\Sigma_w$.
Their dependence on the bounded $U$ is immaterial for the pointwise
triangle inequality. A union bound over Gaussian coordinates proves
(RFT.2)'s tails, and Minkowski proves its moments.
The final independent position Gaussian adds variance $s^2$ while the
stored velocity is capped by $V$, giving the final moment bound.

For the bulk assertions, Markov's inequality for the squared centered
Gaussian norm gives probability at least $1/2$ of norm at most
$\sigma\sqrt{2d}$ or $\sqrt\Sigma\sqrt{2d}$ respectively.
The pointwise bounded shifts give the stated bulk radii even when $U$
depends on those Gaussians. Integrate the original Gaussian kernel over
that half mass to obtain $b_j(R)$ on the indicated query ball. These
are target-law degree bounds, not pathwise bounds on an empirical degree.
:::

(sec-rft-first)=
## 2. The first empirical row field and its exact self exclusion

:::{prf:lemma} First row field in a fourth-moment coupling
:label: lem-rft-first-row

Let $\eta_N=N^{-1}\sum_i\delta_{(X_i,v_i)}$ have speeds bounded by
$V_c$, and let its deterministic target $\eta$ have the source-box
tails and bulk degree just proved. For a $W_4$ coupling of these laws
with cost $e$, use the original nonself row force at each empirical
query and the population row force at the target query. Then for any
$R>0$ their coupling $L^4$ difference is at most
$$
F_1(e,R)=K_1(R)(e+N^{-1})+4V_cT_X(R)^{1/4},
$$
$$
K_1(R)=1+\frac{2(2+4\ell V_c)}{b_1(R)}
       +\frac{4V_c(2\ell+1)}{b_1(R)^2}
       +\frac{8V_c(2\ell+1)}{b_1(R)}.
\tag{RFT.3}
$$
Its two coupled first stages, with shared own OU Gaussian, have joint
$L^4$ phase difference at most
$$
H_1(R)(e+N^{-1})+H_TT_X(R)^{1/4},
$$
$$
H_1(R)=a_x+ct+b+c+(b+c)t\nu K_1(R),\quad
H_T=4V_c(b+c)t\nu.
\tag{RFT.4}
$$
:::

:::{prf:proof}
Write the count numerator $C_\eta(x,v)=\int K(x,X)(v'-v)d\eta$.
Its empirical self contribution is exactly zero. The actual nonself
row force is therefore $C_{\eta_N}(X_i,v_i)/
[a_{\eta_N}(X_i)-1/N]$: the two factors $N/(N-1)$ cancel.
The Gaussian Lipschitz bound under a product of the same coupling gives
$$
\|a_{\eta_N}(X)-1/N-a_\eta(X')\|_4\le2\ell e+1/N=:d_4,
$$
$$
\|C_{\eta_N}(X,v)-C_\eta(X',v')\|_4
\le(2+4\ell V_c)e=:B_4.
$$
For the second inequality subtract velocity and kernel factors separately;
the two velocity differences cost $2e$ and the bounded target velocity
factor costs $4\ell V_ce$.

On $|X'|\le R$ the target degree is at least $b_1(R)$.
Exclude additionally the event that degree difference exceeds $b_1/2$.
There both denominators are at least $b_1/2$ and numerator subtraction
bounds the good force $L^4$ difference by
$2B_4/b_1+4V_cd_4/b_1^2$.
The excluded mass is at most
$T_X(R)+16d_4^4/b_1^4$.
Every first row force has norm at most $2V_c$, so its difference on
this mass is at most $4V_c$. Taking the fourth root costs at most
$4V_cT_X(R)^{1/4}+8V_cd_4/b_1$.
Collect these terms; (RFT.3) enlarges the coefficients of $e$ and $1/N$
to one common bound. In particular it controls the exceptional empirical
degree event rather than assuming it does not occur.

The first averaged velocity is $U=v+t\nu C^{\rm row}$, so its
coupled $L^4$ difference is at most $e+t\nu F_1$.
The own shared OU innovation cancels, and
$\Delta Y=a_x\Delta X+b\Delta U$,
$\Delta W=-ct\Delta X+c\Delta U$.
Their norm sum gives (RFT.4).
:::

(sec-rft-joint)=
## 3. The correlated empirical OU law

:::{prf:lemma} Conditional joint-stage comparison without independent prepared rows
:label: lem-rft-joint-stage

Freeze a consistent input array $S$ with alive fraction at least $m_f$.
Put $\mu_N=L_N(S)$, $\eta=J(\mu_N)$ and let $\eta_N$ be the actual
random prepared empirical phase law. Its actual noisy joint empirical
law is $\Lambda_N$, and the deterministic target is the population
stage law $\Lambda_\eta$ using $\eta$'s own first row field. For any
$R>0$,
$$
\mathbb E[W_4(\Lambda_N,\Lambda_\eta)^4\mid S]
\le D_S(R)N^{-\kappa/2}+D_TT_X(R),
$$
$$
D_S(R)=512H_1(R)^4(D_P+1)+8D_O,\qquad D_T=64H_T^4.
\tag{RFT.5}
$$
Consequently define
$$
E_S(N,R)=[D_S(R)N^{-\kappa/2}+D_TT_X(R)]^{1/4}.
\tag{RFT.6}
$$
Its conditional first and second transport moments are at most
$E_S(N,R)$ and $E_S(N,R)^2$, respectively.
:::

:::{prf:proof}
The actual preparation test-MSE $G_p/N$, source-box moment bound
$Z_P^8$ and the conditional cell estimate of
{prf:ref}`lem-vupt-cell` give
$$
\mathbb E[\mathsf d(\eta_N,\eta)\mid S]\le A_PN^{-\kappa},\quad
\mathbb E[W_4(\eta_N,\eta)^4\mid S]\le D_PN^{-\kappa/2}.
$$
The second assertion is the fourth-power form of its proof:
$\mathbb E W_4^4\le(1+16Z_P^4)\sqrt{\mathbb E\mathsf d}$.
It does not require independent prepared rows.

Condition on the entire actual prepared array and choose a measurable
almost-optimal coupling $\pi_N$ of $\eta_N$ to deterministic $\eta$.
At each actual labeled point sample a target prepared point from the
corresponding conditional measure of this coupling, independently over
labels. If there are repeated empirical atoms, disintegrate using their
individual labels with equal mass $1/N$. These conditional target laws
need not be equal, but their average is exactly $\eta$ for each realized
array. Use the same independent own OU Gaussian in an actual and target
row. The target first field is the fixed population field of $\eta$;
the actual first field remains its nonself empirical row field.

Conditional on preparation, the auxiliary target stage rows are independent
and their average conditional law is exactly $\Lambda_\eta$.
This follows by integrating their individual target prepared laws and
the own Gaussian through that same deterministic target stage map.
Hence each bounded target-empirical test has conditional variance at
most $1/N$, and the averaged conditional target phase eighth moment
is at most $Z_O^8$. The cell estimate and its fourth-power upgrade give
$\mathbb E[W_4(\Lambda_N',\Lambda_\eta)^4\mid\eta_N]
\le D_ON^{-\kappa/2}$ for this auxiliary empirical $\Lambda_N'$.
Its independent rows are a comparison device; the actual prepared or
actual stage rows are not asserted to be independent.

The labelwise coupled actual/target stages obey (RFT.4). Therefore
$$
\mathbb E[W_4(\Lambda_N,\Lambda_N')^4\mid S]
\le64H_1(R)^4(D_P+1)N^{-\kappa/2}+8H_T^4T_X(R),
$$
using $(A+B)^4\le8(A^4+B^4)$ twice and
$N^{-4}\le N^{-\kappa/2}$. The transport triangle inequality with
another fourth-power factor eight proves (RFT.5).
Almost-optimal measurable choices followed by a zero-error limit yield
the same bounds if a disintegration is not fixed uniquely.
The actual joint provider is $\Lambda_N$ in every ensuing actual second
row field; no independent position/velocity product law has replaced it.
:::

(sec-rft-second)=
## 4. Uncapped second velocities and terminal marks

:::{prf:lemma} Local second row comparison in bounded marked transport
:label: lem-rft-second-row

Let $e_S=W_4(\Lambda_N,\Lambda_\eta)$ for the actual stage array and
the deterministic target stage above. For cutoffs $R,H>0$ put
$$
K_2(R,H)=T_s\left[2+t+
 \frac{2t\nu(2+4\ell W_L)}{b_2(R)}
 +\frac{4t\nu\ell(W_L+H)}{b_2(R)^2}\right],
$$
$$
J_2(R,H)=\frac{2T_st\nu(W_L+H)}{b_2(R)^2},\qquad
T_s=\max\{1,(s\sqrt{2\pi})^{-1}\}.
\tag{RFT.7}
$$
The marked distance of the conditional final-noise mean output law of
the actual array to the full population output $\mathcal F_L^{\rm row}\mu_N$
is at most
$$
K_2(R,H)e_S+J_2(R,H)/N+T_Y(R)+T_W(H)
 +32\ell^2e_S^2/b_2(R)^2+8/[N^2b_2(R)^2].
\tag{RFT.8}
$$
Its expectation is bounded by the same expression with $e_S,e_S^2$
replaced by $E_S(N,R_1),E_S(N,R_1)^2$ for any first cutoff $R_1$.
:::

:::{prf:proof}
Couple the joint phase laws by their actual $W_4$ cost. The numerator
subtraction in {prf:ref}`lem-cg-viscous-force-stability`, before any
count normalization, gives
$$
\|\Delta C\|_2\le(2+4\ell W_L)e_S,
\qquad
\|a_{\Lambda_N}(Y)-1/N-a_{\Lambda_\eta}(Y')\|_2
\le2\ell e_S+1/N=:d_2.
$$
The numerator estimate uses the target's fourth uncapped velocity
moment by Hölder; it does not use an actual or target velocity maximum.
The self numerator contribution is zero, so its exact original nonself
force has denominator $a_{\Lambda_N}-1/N$ as in the first kick.

Exclude $|Y'|>R$, $|W'|>H$, and a degree difference exceeding
$b_2(R)/2$. Their total coupling mass is at most
$T_Y(R)+T_W(H)+4d_2^2/b_2(R)^2$.
On the remaining set the target count numerator has norm at most
$W_L+H$, and both degrees have the stated positive local bounds.
Ratio subtraction gives good-force $L^2$ difference at most
$$
\frac{2(2+4\ell W_L)e_S}{b_2(R)}
 +\frac{2(W_L+H)d_2}{b_2(R)^2}.
$$
The actual pre-cap velocity is $w-ty+t\nu C^{\rm row}$.
The 1-Lipschitz cap charges the good pairs by their phase differences
and the displayed force difference. Maximally couple the final position
Gaussians for each pre-noise pair. Their mismatch probability is at most
$|y-y'|/(s\sqrt{2\pi})$, so the good marked bounded cost is at most
$T_s(|\Delta y|+|\Delta v^{\rm cap}|)$.
This couples the original box mark even at a face; it does not use a
Lipschitz bound on the terminal indicator.
Every bad pair has bounded marked cost at most one, regardless of its
uncapped velocities. Finally
$4d_2^2/b_2^2\le32\ell^2e_S^2/b_2^2+8/(N^2b_2^2)$.
Collecting the good coefficients proves (RFT.8). The conditional mean
output law on the actual side averages its original labeled self-excluded
second kicks and caps; no independent-field approximation is made.
:::

(sec-rft-consistency)=
## 5. Explicit full marked conditional consistency

:::{prf:definition} A vanishing row consistency function
:label: def-rft-consistency-floor

Let
$$
k_X=(8d\sigma^2)^{-1},\quad
k_Y=(8d\Sigma)^{-1},\quad k_W=(8d\Sigma_w)^{-1},\quad
c_2=1+2/\rho^2,
$$
$$
A=\max\{1,\sqrt{8(c_2+1)/k_X}\},\quad
R_*=1+2\max\{R_D,B_y,B_w\},\quad
R_N=R_*+[\log(N+e)]^{1/4},\quad R_{1,N}=AR_N.
\tag{RFT.9}
$$
Define the explicit fixed-parameter function
$$
\begin{aligned}
\mathcal A_N=\min\{1,\;&A_FN^{-\kappa}
 +K_2(R_N,R_N)E_S(N,R_{1,N})+J_2(R_N,R_N)/N\\
 &+T_Y(R_N)+T_W(R_N)
 +32\ell^2E_S(N,R_{1,N})^2/b_2(R_N)^2
 +8/[N^2b_2(R_N)^2]\}.
\end{aligned}
\tag{RFT.10}
$$
The possibly very large first radius is an analysis cutoff; it is not
applied to a gas source, companion law, or Gaussian draw.
:::

:::{prf:theorem} Conditional consistency of the actual full marked row update
:label: thm-rft-conditional-consistency

For every consistent capped input array $S$ with alive fraction at least
$m_f$, including unbounded retained dead positions, the actual full
one-step row update satisfies
$$
\mathbb E[\mathsf d_m(L_N(S^+),\mathcal F_L^{\rm row}L_N(S))\mid S]
\le\mathcal A_N\longrightarrow0.
\tag{RFT.11}
$$
The bound is independent of its observation time, stored dead-coordinate
distribution, sampled component realization and particle number except
through the explicit function $\mathcal A_N$.
More precisely, for fixed primitives there is a finite explicit constant
$C_L$ such that
$$
\mathcal A_N\le\min\{1,C_L[
e^{C_AR_N^2}N^{-a}+e^{-k_FR_N^2}]\},\quad
C_A=2+4(1+A^2)/\rho^2,\quad k_F=\min\{1,k_Y,k_W\}>0.
\tag{RFT.12}
$$
An explicit choice is obtained by defining
$$
F_j=2e^{B_j^2/\rho^2},\quad
\overline K_1=1+2(2+4\ell V_c)F_1
 +4V_c(2\ell+1)F_1^2+8V_c(2\ell+1)F_1,
$$
$$
\overline H_1=a_x+ct+b+c+(b+c)t\nu\overline K_1,\quad
C_S=[512\overline H_1^4(D_P+1)+8D_O]^{1/4}
                                         +(2dD_T)^{1/4},
$$
$$
\overline K_2=T_s[2+t+2t\nu(2+4\ell W_L)F_2
                       +8t\nu\ell(W_L+1)F_2^2],\quad
\overline J_2=4T_st\nu(W_L+1)F_2^2,
$$
$$
C_L=1+A_F+2\overline K_2C_S+\overline J_2
                    +128\ell^2F_2^2C_S^2+8F_2^2+4d.
$$
Thus (RFT.10) and (RFT.12) both use fully specified constants.
For a closed eventual threshold, put
$$
K_0=\log C_L+2C_AR_*^2+ae_{\rm E}/2,\quad
U_0=\max\left\{1,\frac{2(2C_A+k_F/2)}a,
 \sqrt{\frac{2(K_0+\log2)}a},\frac{2\log(2C_L)}{k_F}\right\},
$$
$$
N_L=\left\lceil e^{U_0^2}\right\rceil,\qquad c_L=k_F/2,
\quad e_{\rm E}=\exp(1).
$$
Then
$\log\mathcal A_N\le-c_L\sqrt{\log(N+e)}$ for $N\ge N_L$.
:::

:::{prf:proof}
Apply {prf:ref}`lem-rft-joint-stage` with $R_{1,N}$, then
{prf:ref}`lem-rft-second-row` with $R=H=R_N$. These compare the
conditional final-noise mean output to its deterministic population
target and retain all original dense second-stage correlations.

Conditional on the entire actual noisy array and all second kicks/caps,
the only remaining innovations are the original independent final
position Gaussians. The conditional empirical law has that mean output
and each bounded-test variance at most $1/N$. Apply the conditional
cell comparison with two copies of each phase cell, one per mark.
The moment conditional on this noisy array is random; its averaged
phase eighth moment is at most $Z_F^8$ by
{prf:ref}`lem-rft-source-tails`. The same concave moment averaging as in
{prf:ref}`lem-spt-marked-consistency` therefore costs at most
$A_FN^{-\kappa}$. This gives exactly (RFT.10)--(RFT.11).
No prepared independence, empirical copied-mass floor or all-row Gaussian
maximum was introduced.

We verify vanishing with the displayed cutoffs. For any $R\ge R_*$,
the Gaussian tail functions satisfy
$T_X(AR)\le2de^{-k_XA^2R^2}$,
$T_Y(R)\le2de^{-k_YR^2}$ and
$T_W(R)\le2de^{-k_WR^2}$.
Also
$b_j(r)^{-1}\le2e^{B_j^2/\rho^2}e^{r^2/\rho^2}$.
Thus $H_1(AR)$ is bounded by its fixed coefficient times
$e^{2A^2R^2/\rho^2}$, and
$$
E_S(N,AR)\le C_S[
 e^{2A^2R^2/\rho^2}N^{-a}
 +e^{-k_XA^2R^2/4}]
$$
with exactly the $C_S$ displayed in the theorem.
The coefficients $K_2(R,R),J_2(R,R)$ are at most their fixed
coefficients times $e^{c_2R^2}$, using $1+R\le2e^{R^2}$.
The squared transport term uses at most
$e^{2R^2/\rho^2}E_S(N,AR)^2$ times its fixed coefficient.
The chosen $A$ has $k_XA^2/4\ge2(c_2+1)$, so every amplified
first-query tail is bounded by a fixed coefficient times $e^{-R^2}$.
The remaining consistency terms are bounded by
$e^{C_AR^2}N^{-a}$, enlarging $N^{-2a},N^{-1},N^{-2}$ and
$N^{-\kappa}$ to $N^{-a}$ when necessary.
These inequalities prove (RFT.12) with the displayed coefficient sum.
For example the squared transport contribution is bounded by
$64\ell^2F_2^2C_S^2$ times each of its consistent and tail terms;
the other coefficients are $A_F$, $\overline K_2C_S$,
$\overline J_2$, $8F_2^2$ and the two Gaussian tail coefficients $2d$.
Since $R_N^2=O(\sqrt{\log N})$, its first term has logarithm
$-a\log N+O(\sqrt{\log N})$ and its second has logarithm
$-k_F\sqrt{\log N}+O((\log N)^{1/4})$.
For the closed threshold, use
$R_N^2\le2R_*^2+2\sqrt{\log(N+e)}$ and
$\log N\ge\log(N+e)-e_{\rm E}/2$ for $N\ge2$.
Writing $u=\sqrt{\log(N+e)}\ge U_0$, the first term's logarithm
is at most $K_0+2C_Au-au^2\le-(k_F/2)u-\log2$.
The second term's logarithm is at most
$\log C_L-k_Fu\le-(k_F/2)u-\log2$.
Their sum is at most $e^{-c_Lu}$, proving the stated bound.
Consequently both terms vanish.
This establishes a full conditional consistency rate for the actual
finite row kernel, rather than invoking qualitative population continuity.
:::

(sec-rft-transfer)=
## 6. Uniform-time transfer under the actual survival conditioning

:::{prf:definition} Row survivor and recent-restart constants
:label: def-rft-restart-register

Use the complete raw marked row population theorem
{prf:ref}`thm-rpf-marked-population`, its fixed point
$\pi_L^{\rm row}$, weight $w=1+\beta W_4+\omega\mathbf1_{a=0}$,
rate $r_*=r_*^{\rm row}<1$, and positive dead-mass endpoint
$\epsilon_*^{\rm row}$. Put
$$
\sigma_*^2=a_x^2\sigma^2+t^2q^2+s^2,\quad
\Delta_L=(1-a_x)L-bV_c,\quad
\epsilon_{\rm box}=2d e^{-\Delta_L^2/(2\sigma_*^2)}
                         <\epsilon_*^{\rm row},
$$
$$
p=1-\epsilon_{\rm box},\quad m_f=1-\epsilon_*^{\rm row},\quad
\eta_f=p-m_f>0,\quad
e_N=\epsilon_{\rm box}^N,\quad r_N=e^{-2N\eta_f^2},\quad
c_s=(1-\epsilon_{\rm box}^2)^{-1}.
\tag{RFT.13}
$$
Use this $m_f$ in the preparation constants of (RFT.1).
The source-box first-output bound and the uniform population burn-in
give the explicit constants
$$
M_{\rm box}=2^7[(a_xR_D+bV_c)^8+\sigma_*^8g_8^8],\quad
\lambda_{\rm burn}=(1+3r_8)/4<1,
$$
$$
n_{\rm box}=1+\left\lceil
 \frac{\log\max\{1,3M_{\rm box}/H_8\}}{-\log\lambda_{\rm burn}}
                                      \right\rceil,
$$
$$
D_{\rm cl}=2+2\beta(1+\sqrt{H_8})+2\omega\epsilon_*^{\rm row},
\qquad C_{\rm pop}=D_{\rm cl}r_*^{-n_{\rm box}}.
\tag{RFT.14}
$$
Use the independently proved modulus
{prf:ref}`thm-rwm-population-modulus` on the entire alive-floor class:
$$
\mathsf d_m(\mathcal F_L^{\rm row}\mu,
                   \mathcal F_L^{\rm row}\mu')
\le\min\{1,C_{\rm mod}^{\rm row}
                   \mathsf d_m(\mu,\mu')^\alpha\},
\quad \alpha=\alpha_{\rm row}=\gamma^2/32\in(0,1).
\tag{RFT.15}
$$
Its completely specified constants are (RWM.2)--(RWM.8), with the same
fixed source box and $m_f$. This modulus applies to atomic empirical
forecast inputs and does not use a copied-mass floor.

Finally let $L_N=\log(N+e)$, $D=1+C_{\rm mod}^{\rm row}$, and set
$$
b_N=1+\left\lfloor\frac{\log L_N}{4\log(1/\alpha)}\right\rfloor,
\quad V_N=D^{1/(1-\alpha)}\mathcal A_N^{\alpha^{b_N-1}},
\quad T_N=(1-e_N)^{-b_N},
$$
$$
\epsilon_N^{\rm row}
 =T_N[V_N+(b_N+1)c_sr_N]+C_{\rm pop}r_*^{b_N}.
\tag{RFT.16}
$$
All constants and cutoffs are deterministic functions of the fixed
primitive row regime. Retained dead-position moments are absent.
:::

:::{prf:theorem} Uniform-time surviving marked row law with a vanishing floor
:label: thm-rft-uniform-surviving-law

For every $N\ge2$, every consistent initial array law with nonzero
alive count and capped velocities, and every $n\ge1$, the actual
survival-conditioned row chain obeys
$$
\mathbb E[\mathsf d_m(\widehat\mu_n^N,\pi_L^{\rm row})
                                         \mid\tau_N>n]
\le u_{N,n}^{\rm row}:=
\min\{1,C_{\rm pop}r_*^{n-1}+\epsilon_N^{\rm row}\},
\qquad \epsilon_N^{\rm row}\longrightarrow0.
\tag{RFT.17}
$$
This is a rate uniform over all observation times with an explicit
vanishing finite-particle error, for the unchanged raw reward and true
row denominator in the completed small-positive-viscosity, large-box
regime. It does not claim exact mixing of the full finite-array law.
:::

:::{prf:proof}
First verify the needed finite survival interfaces for row mode. Condition
on the actual frozen source, component and original-velocity choices
before jitters and kinetic Gaussians. Each source is in $D_L$ and the
landing identity is
$$
x_i^+=a_xu_i+bU_i+a_xI_i\sigma Z_i+tq\xi_i+s\zeta_i,
\qquad |U_i|\le V_c.
$$
The Gaussian sums displayed after $bU_i$ are independent over labels
under this conditioning and have variances at most $\sigma_*^2$.
The inward events with coordinate norm at most $\Delta_L$ therefore
have independent probabilities at least $p$. They imply alive output
even though $U_i$ may depend on the full jitter array.
Thus the actual alive count dominates their Bernoulli sum; extinction
probability is at most $e_N$ and the alive-fraction lower tail is at
most $r_N$ by the elementary Hoeffding bound. The original second
row kick is retained and does not move these positions.
After conditioning on survival at any one observation time, its
low-alive probability is at most $c_sr_N$: divide that last-step
bound by the last-step survival probability at least $1-e_N$,
then average the actual preceding conditional law. No historical
survival denominator appears.

For the population forecast, these same identities imply alive mass
at least $p>m_f$ after every update and first-output eighth moment
at most $M_{\rm box}$. The row population proof gives the recurrence
$M_8^+\le\lambda_{\rm burn}M_8+B_8$ after that output, with limiting
level $B_8/(1-\lambda_{\rm burn})=2H_8/3$.
Consequently $n_{\rm box}$ updates place every positive-alive input
in the invariant Gaussian moment class. Its $w$ diameter is at most
$D_{\rm cl}$. Combining burn-in with (RPF.16) proves the uniform
forecast bound
$$
\mathsf d_m((\mathcal F_L^{\rm row})^j\mu,\pi_L^{\rm row})
\le\min\{1,C_{\rm pop}r_*^j\}\qquad(j\ge0)
$$
for every positive-alive consistent capped input, including arbitrary
retained dead coordinates. Before burn-in the right side equals one.

Fix $n,N$, put $m=\min\{n-1,b_N\}$ and $k=n-m\ge1$.
Start the ordinary stopped continuation $\mathbb P_k$ from the actual
law of $S_k^N$ conditioned on $\tau_N>k$. The desired law conditioned
through the later endpoint $n$ is its exact own-window reweighting.
Each starting surviving array has probability at least $(1-e_N)^m$
of surviving that window. Hence its Radon--Nikodym derivative is at
most $T_N$, including the change of the starting-array law. This is
the proof of {prf:ref}`lem-spt-recent-tilt` with the just-verified
row extinction bound, not conditioning two coupled chains on joint
survival.

Let $G$ be the single event that the actual alive fraction is at least
$m_f$ at every time $k,\ldots,k+m$. Extinction is a failure of this
event. Its initial conditional-past failure probability is at most
$c_sr_N$, and every later ordinary update while alive has failure
probability at most $r_N$. A window union bound, stopped on the first
extinction, gives $\mathbb P_k(G^c)\le(m+1)c_sr_N$.

Forecast $\mu_j=(\mathcal F_L^{\rm row})^j\widehat\mu_k^N$.
On $G$ its initial alive mass is at least $m_f$; each later population
output has alive mass at least $p$. Thus (RFT.11) and (RFT.15) apply
at every compared step. Write
$q_j=\mathbb E_k[1_G\mathsf d_m(\widehat\mu_{k+j}^N,\mu_j)]$.
Enlarge $1_G$ to the past-measurable alive-floor event before invoking
conditional one-step consistency. The transport triangle and Jensen
on the subprobability $1_Gd\mathbb P_k$ then give
$$
q_{j+1}\le\mathcal A_N+(D-1)q_j^\alpha,\qquad q_0=0.
$$
The induction
$q_j\le D^{1/(1-\alpha)}\mathcal A_N^{\alpha^{j-1}}$ for $j\ge1$
follows from $\mathcal A_N\le1$ and
$D^{1/(1-\alpha)}\ge1+(D-1)D^{\alpha/(1-\alpha)}$.
Since $m\le b_N$ and the base is at most one, $q_m\le V_N$;
for $m=0$ the error is zero. Add the bad-window probability only once
after this recursion. The exact own-window survival reweighting costs
at most $T_N$. Therefore the conditional endpoint forecast error is
at most $T_N[V_N+(b_N+1)c_sr_N]$.

The population forecast error is uniformly bounded by
$C_{\rm pop}r_*^m\le C_{\rm pop}r_*^{n-1}
+C_{\rm pop}r_*^{b_N}$ under the same conditional path law.
This bound uses neither a starting moment nor an additional tilt.
The triangle inequality proves (RFT.17).

Finally $\alpha^{b_N-1}\ge L_N^{-1/4}$, while (RFT.12) gives
$\log\mathcal A_N\le-c_LL_N^{1/2}$ for $N\ge N_L$.
Thus
$$
\log V_N\le\frac{\log D}{1-\alpha}-c_LL_N^{1/4}
\longrightarrow-\infty.
$$
The window grows as $O(\log\log N)$, $e_N,r_N$ decay exponentially
in $N$, and $r_*^{b_N}\to0$. Hence $T_N\to1$, each term of
(RFT.16) vanishes, and the floor tends to zero uniformly in $n$.
:::

:::{prf:corollary} Surviving physical alive row laws
:label: cor-rft-alive-wasserstein

Let $\widehat\mu_n^{N,A}$ be the actual alive empirical phase law at
time $n$, normalized by its own nonzero alive count under $\tau_N>n$.
Let $\pi_L^{\rm row,A}$ be the normalized current-alive stationary
population law of {prf:ref}`thm-rpf-marked-population` and define
the physical squared transport cost using one fixed symmetric positive
definite $2d\times2d$ phase matrix $G$. Put
$$
D_G^2=4\lambda_{\max}(G)(dL^2+V^2),\quad
\zeta_{N,n}=\min\{1,2u_{N,n}^{\rm row}/m_f+c_sr_N\}.
$$
Then
$$
\mathbb E[W_{2,G}(\widehat\mu_n^{N,A},\pi_L^{\rm row,A})^2
                                           \mid\tau_N>n]
\le D_G^2\zeta_{N,n}.
\tag{RFT.18}
$$
The squared Wasserstein distance between the conditional law of the
random alive empirical measure and the Dirac law at
$\pi_L^{\rm row,A}$ has the same bound. First sampling a surviving
swarm and then sampling one of its alive slots uniformly gives a phase
law whose squared $W_{2,G}$ distance to $\pi_L^{\rm row,A}$ is also
at most $D_G^2\zeta_{N,n}$.

In particular the bounds have a geometric population relaxation term
with rate $r_*$ independent of $N$ and an explicit error vanishing
as $N\to\infty$, uniformly over all observation times.
:::

:::{prf:proof}
The complete alive normalization argument of
{prf:ref}`cor-spt-alive-w2` depends only on the conditional marked
distance, the own alive-fraction floor event, positive target alive
mass and the physical alive phase-space diameter. All four inputs
have now been proved for the actual row kernel. The target alive mass
is at least $p>m_f$; the finite conditional lower-tail probability is
at most $c_sr_N$; and (RFT.17) supplies its marked distance.
Restrict an optimal marked coupling to its alive parts and complete
the residual after dividing each part by its own alive mass. On the
good event, the expected normalized bounded alive transport cost,
after averaging over that event, is at most
$2u_{N,n}^{\rm row}/m_f$. The bad event costs at most one.
The squared physical $G$ diameter bounds the physical cost by $D_G^2$
times that bounded normalized cost, giving (RFT.18).

With a Dirac target in the probability-measure space, its outer
transport cost is exactly the expected inner squared Wasserstein
cost. For the sampled law, integrate measurable almost-optimal
alive phase couplings under the actual conditional surviving-swarm
law. Their target marginal is always $\pi_L^{\rm row,A}$ and their
first marginal is the stated swarm-first alive sample law; taking an
infimum gives the same upper bound. Neither step replaces a selected
coupling's lower cost bound by an optimal-transport lower bound.
:::

:::{prf:corollary} All-slot sampling followed by current-alive conditioning
:label: cor-rft-all-slot-alive-sample

Alternatively sample $S_n^N$ under $\tau_N>n$, independently select
one of all $N$ slots uniformly, and then condition that slot on its
current alive mark. Write $\rho_{N,n}^{\rm slot,A}$ for this phase
law, which weights surviving swarms by their actual alive fraction
$a_N=m_A(\widehat\mu_n^N)$. Then for every $n\ge1$,
$$
\mathbb E[a_N\mid\tau_N>n]\ge p>m_f,
$$
$$
W_{2,G}(\rho_{N,n}^{\rm slot,A},\pi_L^{\rm row,A})^2
\le D_G^2\min\{1,\zeta_{N,n}/m_f\}.
\tag{RFT.19}
$$
The additional factor $m_f^{-1}$ refers to this sampling order;
the swarm-first uniform-alive-slot law in the preceding corollary
has the bound (RFT.18).
:::

:::{prf:proof}
Given each surviving input at time $n-1$, the raw expected output
alive fraction is at least $p$ by the independent inward indicators.
It is zero on extinction. Averaging under the actual conditional past
and then dividing by its one-step survival probability at most one
gives $\mathbb E[a_N\mid\tau_N>n]\ge p$. This calculation uses a
last-step marginal; it does not divide by whole-history survival.

For each surviving swarm choose a measurable almost-optimal coupling
of its normalized alive empirical law to $\pi_L^{\rm row,A}$.
Integrate these couplings with normalized weight
$a_N/\mathbb E[a_N\mid\tau_N>n]$ under the actual conditional
surviving-swarm law. The first marginal is exactly
$\rho_{N,n}^{\rm slot,A}$ and the second remains the fixed target.
Its cost is at most
$m_f^{-1}\mathbb E[W_{2,G}(\widehat\mu_n^{N,A},
\pi_L^{\rm row,A})^2\mid\tau_N>n]$, since $a_N\le1$.
Apply (RFT.18), then also the diameter bound $D_G^2$ to obtain
(RFT.19). This normalization differs from choosing an alive slot
uniformly after first fixing a surviving swarm.
:::

:::{prf:corollary} Two separately surviving swarm laws
:label: cor-rft-two-surviving-swarms

Consider two actual row-normalized swarms in the same primitive regime
and fixed box, of sizes $N,M\ge2$, with possibly different consistent
initial laws and nonzero alive counts. Each chain is conditioned only
on its own survival through an observation time $n\ge1$. Put
$$
\mathcal Q_{N,n}^A=
\operatorname{Law}(\widehat\mu_n^{N,A}\mid\tau_N>n),\qquad
\mathcal Q_{M,n}^A=
\operatorname{Law}(\widehat\mu_n^{M,A}\mid\tau_M>n).
$$
Let $\mathscr W_2$ denote Wasserstein distance between these laws of
probability measures, with ground metric $W_{2,G}$ on alive phase
probabilities. Then
$$
\mathscr W_2(\mathcal Q_{N,n}^A,\mathcal Q_{M,n}^A)
\le D_G(\sqrt{\zeta_{N,n}}+\sqrt{\zeta_{M,n}}).
\tag{RFT.20}
$$
Their swarm-first uniform-alive-slot phase laws $\rho_{N,n}^A$ and
$\rho_{M,n}^A$ satisfy the same bound:
$$
W_{2,G}(\rho_{N,n}^A,\rho_{M,n}^A)
\le D_G(\sqrt{\zeta_{N,n}}+\sqrt{\zeta_{M,n}}).
\tag{RFT.21}
$$
For the distinct all-slot sampling order followed by current-alive
conditioning, the phase laws of
{prf:ref}`cor-rft-all-slot-alive-sample` satisfy
$$
W_{2,G}(\rho_{N,n}^{\rm slot,A},\rho_{M,n}^{\rm slot,A})
\le D_G\left[
\sqrt{\min\{1,\zeta_{N,n}/m_f\}}
+\sqrt{\min\{1,\zeta_{M,n}/m_f\}}\right].
\tag{RFT.22}
$$
Here every $\zeta$ is the separately proved uniform-time bound for
that swarm's own survival-conditioned marginal.
:::

:::{prf:proof}
The common target is the same deterministic current-alive stationary
population law $\pi_L^{\rm row,A}$ for both swarms.
Equation (RFT.18) bounds each outer distance to its Dirac law by
$D_G\sqrt{\zeta_{N,n}}$ or $D_G\sqrt{\zeta_{M,n}}$.
The triangle inequality for $\mathscr W_2$ through that Dirac target
proves (RFT.20). Equivalently, couple the two separately conditioned
empirical-measure laws by their product, use the pointwise metric
triangle through $\pi_L^{\rm row,A}$ and then Minkowski; this has
exactly their own two conditional marginals.

Apply the ordinary $W_{2,G}$ triangle through the same phase target
to the two swarm-first sample bounds of (RFT.18), yielding (RFT.21).
Applying it to the two distinct sample bounds (RFT.19) gives (RFT.22).
Each input bound was normalized by its own survival probability and,
for the all-slot convention, its own expected alive fraction. No law
conditioned on simultaneous survival of a prescribed coupled pair is
substituted into these comparisons.

These inequalities describe delayed relaxation of the two laws toward
their common stationary population target, with the explicit finite-
particle floors already stated. They impose no common initial orbit
or equilibrium configuration and make no assertion that a one-update
pathwise discrepancy decreases monotonically.
:::

:::{prf:remark} Completed row transfer and its retained scope
:label: rem-rft-scope

The conditional row consistency, whole alive-floor population modulus,
and short own-survival restart now compose to the full finite-particle
alive-law estimate (RFT.18). All Gaussian tails, both original row
denominators, self exclusion, sampled fitness normalizers, rooted
component collisions and original frozen-slot velocities are retained.
The primitive sufficient regime remains that of
{prf:ref}`def-rpf-positive-endpoints`: a fixed sufficiently large box,
small positive fitness powers and small positive viscosity. This does
not certify the prescribed default $\nu=.3,L=2$.
The target is the marked stationary population law and its current-alive
restriction. A finite-particle QSD, the full finite-array invariant law,
and an additional future-horizon survival tilt are distinct objects
and are not identified by this transfer.
:::
