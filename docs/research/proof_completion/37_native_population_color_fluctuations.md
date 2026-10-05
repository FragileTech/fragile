# Population fluctuations of the actual matched count-color instrument

(sec-npf-register)=
## 1. Complete existing algorithm and consumed color tests

:::{prf:definition} Native population color fluctuation register
:label: def-npf-register

Retain every parameter and original stage of
{prf:ref}`def-nfs-register`, including its real-coordinate continuous
Gaussian dense count execution, complete sampled global normalization,
both donor roles, current revival, simultaneous copying, entire accepted
component Haar variable, both viscous kicks, cap and terminal alive marks.
Keep the original positive contraction tests and every primitive constant
of {prf:ref}`def-native-concentration-register` and
{prf:ref}`def-nqc-preparation-budget`. In particular
$0\le t\nu\le1/2$, $q,s,\sigma_J>0$, and $0<V\le\min\{V_0,1\}$.
The actual entering high-alive class is $H_N$ with floor $m_0$;
its QSD complement has the already proved bound
$\zeta_N=e^{-uN}$, $u=a_0/8$, for
$N\ge\max\{\widehat N_*,4/a_0,2/m_0\}$.
Use $A_{\rm prep},D_J,R_D,V_c,L_{Xx},L_{Xv},L_{Vx},L_{Vv}$
from the existing complete preparation proof.

The actual stopped record is $W_i=(Y_i,I_i,w_i)$, with
$|Y_i|\le R_D$, $I_i\in\{0,1\}$ and $|w_i|\le V_c$.
Its original recipient jitter, first kick, OU and B2 inputs are

$$
\begin{gathered}
X_i=Y_i+\sigma_J I_iG_i^J,\qquad
U_i=w_i+\frac{t\nu}{N}\sum_{j\ne i}
                 K(X_i-X_j)(w_j-w_i),\\
p_i=\alpha X_i+tU_i,\quad v_{1i}=U_i-t\lambda X_i,
\quad z_i=cv_{1i}+q\xi_i,\quad
y_i=p_i+tz_i=a_xX_i+bU_i+tq\xi_i,\\
F_i=\frac\nu N\sum_{j\ne i}K(y_i-y_j)(z_j-z_i),\quad
a_i^+=\mathbf1_D(y_i+sG_i^{\rm pos}),
\quad K(x)=e^{-|x|^2/(2\rho^2)}.
\end{gathered}
\tag{NPF.1}
$$

These are the existing equations with their exact count denominator
and self exclusion. Every row is eligible after original revival.
The cap follows B2 and does not change these stored force inputs.

Let $\psi(z,F)$ be an explicit bounded real test, $|\psi|\le B$,
with global Euclidean Lipschitz budgets $L_z,L_F$ in its two
variables. The complete actual observation studied first is

$$
H_N=\frac1N\sum_{i=1}^N
                   (1-I_i)a_i^+\psi(z_i,F_i).
\tag{NPF.2}
$$

It retains the literal matched clone-deletion and final-alive masks.
The version omitting only the passive final-alive factor has the
same formulas with $\zeta_s=0$ and its terminal variance term zero.
The algorithm and its hard color are unchanged; $\psi$ is a declared
test of the stored original source/force record.
Use a fixed originally supplied finite color phase $\kappa_c$.
Shared random calibrations retain their actual comparison errors
of {prf:ref}`cor-nsc-calibration-error` rather than being assigned
these fixed-budget conclusions by name.
:::

:::{prf:lemma} A smooth bounded test of the original available projector
:label: lem-npf-projector-test

Let $\delta_c>0$ be the literal original force threshold.
Choose a fixed $C^1$ scalar weight $\chi$ with
$0\le\chi\le1$, $\chi(r)=0$ for $r\le\delta_c$,
$\chi(r)=1$ for $r\ge2\delta_c$, and
$|\chi'|\le2/\delta_c$.
For a fixed Hermitian matrix $A$, $\|A\|_F\le B_A$, set

$$
\psi(z,F)=\chi(|F|)\operatorname{tr}(AP(z,F)),\qquad
P=cc^\dagger,\quad
c=(F/|F|)\odot e^{i\kappa_c z},
\tag{NPF.3}
$$

with its continuous zero extension at $|F|\le\delta_c$.
This is a smooth weighted test of the literal available color,
with explicit safe budgets

$$
B=B_A,\qquad L_z=2B_A|\kappa_c|,\qquad
L_F=6B_A/\delta_c.
\tag{NPF.4}
$$

Its original unavailable rows remain unavailable.
This does not give the unweighted hard projector these global
Lipschitz budgets.
:::

:::{prf:proof}
For a unit color, $\|P\|_F=1$.
The derivative of $F/|F|$ has operator norm at most $1/|F|$,
and that of its phase in $z$ has norm at most $|\kappa_c|$.
Differentiating $cc^\dagger$ costs twice these bounds.
The product with $\chi$ costs its displayed derivative plus
$2/\delta_c$ on the available set. The larger safe $6/\delta_c$
therefore proves (NPF.4), including the zero extension.
All masks in (NPF.2) retain their original values.
:::

(sec-npf-primitive-influence)=
## 2. Primitive full-force Gaussian and compact-record influences

:::{prf:definition} Explicit original B2 observation constants
:label: def-npf-influence

Put $\ell_\rho=e^{-1/2}/\rho$,
$\zeta_s=2/(\sqrt{2\pi}s)$, and

$$
\begin{gathered}
\theta_U=4t\nu V_c\ell_\rho,\qquad
Z_2=cV_c+ct\lambda(R_D+\sigma_J\sqrt d)+q\sqrt d,\\
K_{Fw}=2\nu c(1+t\nu)
                 +4\nu\ell_\rho Z_2b(1+t\nu),\\
K_{FX}=2\nu c(\theta_U+t\lambda)
           +4\nu\ell_\rho Z_2(|a_x|+b\theta_U),\\
K_w=(L_zc+B\zeta_sb)(1+t\nu)+L_FK_{Fw},\\
K_X=L_zc(\theta_U+t\lambda)+L_FK_{FX}
                         +B\zeta_s(|a_x|+b\theta_U),\\
K_J=\max\{K_X,K_w,B+\sigma_J\sqrt d K_X\},\\
K_O=L_z+tB\zeta_s+2\nu L_F
                           +4t\nu\ell_\rho L_FZ_2,\\
C_{\rm inst}=B^2+q^2K_O^2+\sigma_J^2K_X^2
                         +\frac{A_{\rm prep}}4(K_JD_J)^2.
\end{gathered}
\tag{NPF.5}
$$

All these constants are functions of the complete original algorithm
and declared readout. In particular the primitive component, fitness,
gate, donor and standardizer dependence remains in $A_{\rm prep}$.
The positive cap enters $V_c$ and the existing contraction register;
unbounded jitter and OU are retained in $Z_2$.
:::

:::{prf:lemma} Averaged actual color record is Lipschitz in its compact source
:label: lem-npf-compact-prediction

Given a deterministic compact array $W$, let
$\mathcal G_N(W)=E[H_N\mid W]$, integrating exactly its original
jitter, OU and final-position noises.
For any two such arrays coupled by the same row labels,

$$
|\mathcal G_N(W)-\mathcal G_N(W')|
\le K_X\overline{|Y-Y'|}+K_w\overline{|w-w'|}
 +(B+\sigma_J\sqrt d K_X)\overline{|I-I'|}
\le K_J\overline d_J(W,W').
\tag{NPF.6}
$$

This includes the same-record kernel and force variation at both kicks.
The constants are uniform over compact PRE-JITTER inputs, not over
arbitrary realized unbounded post-jitter arrays.
:::

:::{prf:proof}
Join the two compact records by their straight coordinate paths;
$I_\theta\in[0,1]$ is used only in this analytic comparison.
Use the same original Gaussian $G^J,\xi,G^{\rm pos}$ at its endpoints.
Write $A_i=|\dot Y_i|$, $C_i=|\dot w_i|$,
$P_i=A_i+\sigma_J|\dot I_i||G_i^J|$.
The first count kick is convex since $t\nu\le1$.
Differentiating its actual Gaussian weights gives

$$
|\dot U_i|\le C_i+t\nu\overline C
          +2t\nu V_c\ell_\rho(P_i+\overline P).
\tag{NPF.7}
$$

Set $p_{i,2}=A_i+\sigma_J\sqrt d|\dot I_i|$.
Its averaged $L^2$ derivative norms are consequently at most
$(1+t\nu)\overline C+\theta_U\overline p_2$.
Equation (NPF.1) gives
$\dot z=c\dot U-ct\lambda\dot X$ and
$\dot y=a_x\dot X+b\dot U$.
Along the entire compact path every original $z_i$ has
$L^2$ norm at most $Z_2$. Symmetric count summation gives

$$
\overline{|\dot F|}
\le2\nu\overline{|\dot z|}
 +2\nu\ell_\rho
 [\overline{|\dot y||z|}
             +\overline{|\dot y|}\,\overline{|z|}].
\tag{NPF.8}
$$

Cauchy--Schwarz with those actual correlated row derivatives
and $Z_2$ gives
$E\overline{|\dot F|}\le
K_{Fw}\overline C+K_{FX}\overline p_2$.
No jitter/force product is factorized.
Integrating the final-alive mark gives its exact original box
probability $p_D(y)=P(y+sG\in D)$, with Lipschitz budget
$\zeta_s$. The row readout therefore costs
$L_zE\overline{|\dot z|}+L_FE\overline{|\dot F|}
 +B\zeta_sE\overline{|\dot y|}+B\overline{|\dot I|}$.
Substitution yields (NPF.6). Gaussian domination justifies
integration of the original derivatives; integrating the
coordinate path preserves its exact original endpoint laws.
:::

:::{prf:theorem} Population-uniform conditional variance of the complete native color test
:label: thm-npf-one-step-variance

For every deterministic entering $S\in H_N$ with $m_0N\ge2$,
the complete original raw record satisfies

$$
\operatorname{Var}(H_N\mid S)\le C_{\rm inst}/N.
\tag{NPF.9}
$$

There is no independence assertion for its prepared rows or
its complete B2 outputs.
:::

:::{prf:proof}
Apply total variance in the actual compact-preparation,
jitter, OU and terminal order.
Conditional on preparation, jitter and OU, the final-position
Gaussian rows are independent and all force/color factors are fixed.
Their bounded marked-row average has variance at most $B^2/N$.
For its final-noise conditional mean, the derivative in an original
OU row satisfies

$$
|\nabla_{\xi_k}EH_N^{\rm terminal}|
\le\frac qN\left[L_z+tB\zeta_s+2\nu L_F
            +2t\nu\ell_\rho L_F(|z_k|+\overline{|z|})\right].
$$

This follows from (NPF.8) with only row $k$ changing directly.
Its $L^2$ norm over the original jitter and OU is at most
$qK_O/N$. The elementary independent standard-Gaussian Poincaré
inequality, summed over its original OU rows, gives $q^2K_O^2/N$.
For the subsequent OU conditional mean its jitter-row derivative
has $L^2$ norm at most $\sigma_JK_X/N$.
Indeed $\dot X_i=\sigma_JI_k\mathbf1_{i=k}$ in a unit direction;
(NPF.7) now has deterministic upper bounds for every
$\dot U_i$, while the products in (NPF.8) keep the actual
$L^2$ norm $Z_2$. Conditional Jensen over OU can only decrease
this derivative norm. Gaussian Poincaré over the original
independent recipient jitters gives $\sigma_J^2K_X^2/N$.

For the compact prefix use its exact chronological original
measurement, donor/gate and addressed component-Haar blocks.
Their expected squared affected-row counts are respectively
$A_D$, $9M_2(C)$ and $M_2(C)$ from the complete component proof.
An affected set of size $D_i$ changes the compact conditional
mean by at most $K_JD_JD_i/N$, by (NPF.6).
The original-block resampling inequality consequently gives
variance at most
$(K_JD_J)^2[A_D+10M_2(C)]/(2N)
 =A_{\rm prep}(K_JD_J)^2/(4N)$.
This bound retains every global measured normalizer change,
every incoming child and its entire correlated Haar component.
Adding the four chronological bounds proves (NPF.9).
:::

(sec-npf-stationary-tightness)=
## 3. Actual stationary color fluctuations have their population scale

:::{prf:definition} Complete conditional-prediction and stationary budgets
:label: def-npf-stationary-budget

Let $g_N(S)=E_{\rm raw}[H_N\mid S]$, and put
$g_d=E|G_d|$ for its original standard Gaussian norm.
Define the primitive coefficients

$$
\begin{gathered}
J_x=K_J[(1+(\sigma_Jg_d)^{-1})L_{Xx}+L_{Vx}],\qquad
J_v=K_J[(1+(\sigma_Jg_d)^{-1})L_{Xv}+L_{Vv}],\\
L_H=\max\{J_x,J_v/\omega\},\qquad
C_H=C_{\rm inst}+2C_{\rm stat}L_H^2,\\
C_{H,\infty}=\frac8{7a_0}
       \left[C_H+\frac{9B^2}{eu}\right],\qquad u=a_0/8.
\end{gathered}
\tag{NPF.10}
$$

$C_{\rm stat}$ is the already derived full marked QSD Lipschitz
variance constant, not a presumed law curvature coefficient.
The original component and global-normalizer constants remain
in $J_x,J_v$ through (PC.8) and $K_J$.
:::

:::{prf:lemma} The actual complete color prediction is Lipschitz on high-alive inputs
:label: lem-npf-input-prediction

For the two original finite entering swarms in $H_N$,

$$
|g_N(S)-g_N(S')|
\le J_x\overline d_x(S,S')+J_v\overline d_v(S,S')
\le L_H\overline d_\omega(S,S').
\tag{NPF.11}
$$

This concerns the complete raw transition's conditional color mean,
including the actual random measured normalizers.
:::

:::{prf:proof}
Use the finite original preparation coupling underlying
{prf:ref}`thm-native-concentration-finite-contraction`.
Before applying the downstream kinetic or cap estimates,
its original compact-pair/jitter bounds are exactly (PC.9).
The coupled jitter is independent of the full compact pair.
For each such pair the symmetric Gaussian norm inequalities give
$E|Y-Y'+\sigma_J(I-I')G|\ge|Y-Y'|$
and $E|Y-Y'+\sigma_J(I-I')G|\ge\sigma_Jg_d|I-I'|$.
The latter follows by averaging a shift and its reflection and
using the norm triangle inequality. Consequently

$$
E\overline d_J(W,W')
\le(1+(\sigma_Jg_d)^{-1})D_X+D_V.
$$

Integrate (NPF.6) against this genuine full component coupling
and insert (PC.9). This proves (NPF.11).
Only the actual entering alive floor is used, not a deterministic
floor imposed on the following original output.
:::

:::{prf:theorem} Complete stationary color variance and population-fluctuation tightness
:label: thm-npf-stationary-tightness

For $N$ in the admitted threshold range, start the original raw
complete update from its actual QSD $\nu_N$.
The native empirical test satisfies

$$
\operatorname{Var}_{\nu_N,\rm raw}(H_N)
                       \le C_H/N+9B^2\zeta_N.
\tag{NPF.12}
$$

Under the stationary complete Doob record law its exact mean
$\theta_N=E_{\pi_N}^eH_N$ has

$$
N\operatorname{Var}_{\pi_N}^e(H_N)
\le\frac{C_H+9NB^2\zeta_N}{\alpha_N\min e_N}
\le C_{H,\infty}.
\tag{NPF.13}
$$

Thus $\sqrt N(H_N-\theta_N)$ is tight, with the explicit bound
$P(|\sqrt N(H_N-\theta_N)|>R)\le C_{H,\infty}/R^2$.
Any fixed finite list of these actual source/projector tests
has the corresponding joint tightness and bounded covariance
matrix. The finitely many smaller admitted populations retain
their primitive bound $NB^2/[\alpha_N\min e_N]$.
:::

:::{prf:proof}
The bounded function $g_N$ is Lipschitz on $H_N$ by (NPF.11).
Its infimum Lipschitz extension followed by clipping to $[-B,B]$
gives $g_N^\sharp$ with the same global coefficient $L_H$
and exact agreement on $H_N$.
The proved stationary marked inequality gives
$\operatorname{Var}_{\nu_N}g_N^\sharp
 \le C_{\rm stat}L_H^2/N$.
Its $L^2$ remainder is at most $2B\sqrt{\zeta_N}$.
Therefore

$$
\operatorname{Var}_{\nu_N}g_N
\le2C_{\rm stat}L_H^2/N+8B^2\zeta_N.
$$

Total variance and (NPF.9) add at most
$C_{\rm inst}/N+B^2\zeta_N$.
This proves (NPF.12), with the actual high-alive exception retained.

The exact stationary Doob one-update density relative to the
original $\nu_N$-started raw record is
$e_N(S_1)\mathbf1_{\rm survive}/[\alpha_N\nu_N(e_N)]$.
It is at most $1/[\alpha_N\min e_N]$.
Center first by the raw mean in this density comparison; the
Doob variance is the infimum over all such centers.
This proves the first inequality of (NPF.13).
Use $\alpha_N\ge a_0$, $\min e_N\ge7/8$,
and $\sup_{x\ge0}xe^{-ux}=1/(eu)$ for its second inequality.
Chebyshev and the finite coordinate covariance inequality prove
the asserted joint tightness. These are fluctuations of the
actual complete recorded color test and its own stationary mean.
They do not assume a stationary Gaussian fluctuation field.
:::

(sec-npf-complete-green-kubo)=
## 4. Population-uniform complete color time covariance

:::{prf:definition} Full instrument covariance constants
:label: def-npf-green-kubo-budget

Retain the actual full-state block length
$K_N=\widehat T_N\le\widehat C_TN$, its Doob TV factor
$\Delta_N\le2/7$, $\bar q=(1+r)/2<1$, and

$$
\begin{gathered}
\varepsilon_N=(1-a_0)^N,\quad
d_N=\widehat b_N/(1-\widehat b_N),\quad
C_R=C_H+9B^2/(eu),\quad
C_D=8C_R/(7a_0),\quad
A_H=L_H\sqrt{C_RC_{\rm stat}},\quad
D_{H,N}=\frac{C_H+9NB^2\zeta_N}
                         {\alpha_N\min e_N}\le C_D.
\end{gathered}
\tag{NPF.14}
$$

All block and eigenfunction constants are the already proved
original ones, with
$d_N\le(16/7)\widehat C_\theta Ne^{-uN}$ and
$\Delta_N\le(32/7)\widehat C_\theta Ne^{-uN}$.
:::

:::{prf:theorem} Uniform full Green--Kubo covariance of the native matched color test
:label: thm-npf-complete-green-kubo

Let $h_{N,n}=H_{N,n}-\theta_N$ under the stationary
complete Doob instrument. Its exact full covariance is

$$
\sigma_{H,N}^2=N E h_{N,1}^2
        +2N\sum_{k\ge1}E[h_{N,1}h_{N,k+1}]
       \le\mathcal C_{H,N},
$$
$$
\begin{aligned}
\mathcal C_{H,N}={}&D_{H,N}+\frac{2A_H}{1-\bar q}\\
&+4B\sqrt{NC_R}
 [K_N(\sqrt{\varepsilon_N}+\sqrt{\zeta_N})
                       +K_N(K_N-1)\zeta_N]\\
&+16NB^2[K_Nd_N+
                        \varepsilon_NK_N(K_N+3)/2]\\
&+\frac{2C_DK_N\sqrt{\Delta_N}}{1-\sqrt{\Delta_N}}.
\end{aligned}
\tag{NPF.15}
$$

The entire covariance series is absolutely convergent.
A primitive upper bound uniform over all threshold-range populations is

$$
\begin{aligned}
\mathcal C_{H,\infty}={}&C_D+\frac{2A_H}{1-\bar q}\\
&+4B\sqrt{C_R}\widehat C_T
            [M_{3/2}(a_0/2)+M_{3/2}(u/2)]\\
&+4B\sqrt{C_R}\widehat C_T^2M_{5/2}(u)\\
&+\frac{256}7B^2\widehat C_T\widehat C_\theta M_3(u)
 +8B^2(\widehat C_T^2+3\widehat C_T)M_3(a_0)\\
&+\frac{2C_D\widehat C_T\sqrt{32\widehat C_\theta/7}}
                         {1-\sqrt{2/7}}M_{3/2}(u/2),
\qquad M_s(v)=(s/(ev))^s.
\end{aligned}
\tag{NPF.16}
$$

For the finitely many smaller admitted populations include
their primitive maximum
$NB^2[1+2/(1-\sqrt{1-a_{F,N}})]$.
Thus the complete temporal covariance of the actual
$\sqrt N$-scaled matched B2 color/source test is population-uniform,
including its preparation, hard native masks and both force stages.
:::

:::{prf:proof}
Extend the actual raw conditional mean $g_N$ by zero at
the cemetery, where no future update observation is emitted.
For $k\ge1$ the raw later prediction from the first terminal
state is $Q_N^{k-1}g_N$.
The stopped original $q_N$ marked coupling, followed by
(NPF.11) on the final good states, gives on $H_N$

$$
|Q_N^{k-1}g_N(S)-Q_N^{k-1}g_N(S')|
\le L_Hq_N^{k-1}\overline d_\omega(S,S')
                                  +4B(k-1)\zeta_N.
$$

Each failure is paid by its original binomial floor exception;
no new restart transition is introduced. The infimum extension
and clipping used in {prf:ref}`lem-nfs-semigroup-extension`
therefore give a globally Lipschitz physical-state prediction
with coefficient $L_Hq_N^{k-1}$ and $L^2(\nu_N)$ remainder
at most $4B(k-1)\zeta_N+2B\sqrt{\zeta_N}$.
Both predictions are set to zero at the cemetery.
The first raw terminal-state law is exactly
$\alpha_N\nu_N+(1-\alpha_N)\delta_\dagger$.
Its extended-prediction variance is at most
$C_{\rm stat}L_H^2q_N^{2(k-1)}/N+4B^2\varepsilon_N$.
The first actual record has variance at most $C_R/N$
by (NPF.12). Conditional prediction and Cauchy--Schwarz hence give

$$
\begin{aligned}
N|\operatorname{Cov}_{\nu_N,\rm raw}(H_{N,1},H_{N,k+1})|
\le{}&A_Hq_N^{k-1}\\
&+2B\sqrt{NC_R}
 [\sqrt{\varepsilon_N}+2(k-1)\zeta_N+\sqrt{\zeta_N}].
\end{aligned}
\tag{NPF.17}
$$

The exact native Doob/raw history comparison through $k+1$
updates has TV at most $d_N+(k+1)\varepsilon_N$.
For two $[-B,B]$ tests their covariance changes by at most
$8B^2$ times this TV. Add this difference to (NPF.17)
and sum only over $1\le k\le K_N$.
Its geometric leading term sums to $A_H/(1-\bar q)$;
the displayed polynomial and exponential terms give the
second and third lines of (NPF.15), with their factor two.

For all remaining lags use the PROVED full physical-state
$L^2$ block estimate and its full-record conditional Markov
prediction in {prf:ref}`thm-nsc-recorded-time-covariance`.
Both complete records have variance at most $C_D/N$.
Their exact physical gap is $k-1$, so

$$
N|E[h_{N,1}h_{N,k+1}]|
\le C_D\Delta_N^{\lfloor(k-1)/K_N\rfloor/2}.
$$

For $k>K_N$ the geometric block sum is at most
$C_DK_N\sqrt{\Delta_N}/(1-\sqrt{\Delta_N})$.
This proves absolute convergence and the last term of
(NPF.15); no small constant error is summed over infinite time.
The exact complete Green--Kubo identity
{prf:ref}`thm-nfs-green-kubo` identifies its covariance with
the full Poisson-corrected bracket, rather than with only
the original terminal-noise component.

Use $K_N\le\widehat C_TN$, the two primitive exponential
eigenfunction/block bounds in (NPF.14),
$\varepsilon_N\le e^{-a_0N}$ and
$1-\sqrt{\Delta_N}\ge1-\sqrt{2/7}$.
The short-window error terms are bounded by the displayed
coefficients of $N^{3/2}e^{-a_0N/2}$,
$N^{3/2}e^{-uN/2}$, $N^{5/2}e^{-uN}$,
$N^3e^{-uN}$ and $N^3e^{-a_0N}$.
The final block tail is bounded by its displayed coefficient
of $N^{3/2}e^{-uN/2}$.
Maximizing $x^se^{-vx}$ gives $M_s(v)$ and proves (NPF.16).
For smaller populations the original finite primitive
$L^2$ factor is $\sqrt{1-a_{F,N}}$ per step; its full-record
geometric covariance sum gives the stated finite maximum.
:::

:::{prf:corollary} Actual joint population/time color path limits
:label: cor-npf-joint-time-limit

Fix a finite list of actual tests (NPF.2) with their original
fixed observation budgets. Let $\Sigma_{H,N}$ be the exact
complete stationary covariance of their $\sqrt N$ versions.
It has uniformly bounded trace by (NPF.16), so its actual
covariance matrices have convergent subsequences.
For every such subsequence $\Sigma_{H,N}\to\Sigma_H$ and every
original update-count schedule with

$$
n_N/N^9\longrightarrow\infty,
\tag{NPF.18}
$$

the linearly interpolated full additive color path
$n_N^{-1/2}\sum_{j\le n_Nt}\sqrt N(H_{N,j}-\theta_N)$
converges on every bounded time interval to Brownian motion
with its limiting complete covariance $\Sigma_H$.
The same limit holds for the original QSD-started history
conditioned once on survival through the entire selected horizon.
The declared finite-$N$ drift is retained exactly.
:::

:::{prf:proof}
Apply the proved complete triangular path theorem
{prf:ref}`thm-njt-joint-functional` to these actual instruments.
Their full observation bound is $B_N=\sqrt N B$ for a
fixed finite vector norm. The original $K_N=O(N)$ and
$1-\Delta_N\ge5/7$ give $U_N,L_N=O(N^{3/2})$;
the bracket-ergodic coefficient is $O(N)$.
Thus its three schedule tests are bounded respectively
by primitive constants times
$N^{3/2}/\sqrt{n_N}$, $N^{9/2}/\sqrt{n_N}$ and
$N^{7/2}/\sqrt{n_N}$. They vanish under (NPF.18).
The actual full covariance bound required there is now
(NPF.16), proved for this complete color instrument.
The exact uniform full-history eigenfunction telescope
transfers its original once-conditioned path as in that theorem.
No Gaussian stationary preparation or independent complete rows
are substituted, and no instantaneous $N$-CLT follows just
from this joint long-time schedule.
:::

(sec-npf-hard-masks)=
## 5. Quantitative variance of the literal unweighted hard color

:::{prf:theorem} Original hard-mask variance retains its derived boundary modulus
:label: thm-npf-hard-variance

For a Hermitian test $\|A\|_F\le B$, let $H_N^{\rm hard}$
be (NPF.2) with its literal available projector and no additional
force taper. For $0<\epsilon\le1/4$, choose its smooth comparison
$H_N^{[\epsilon]}$ as follows. At $\delta_c>0$ the force weight
is zero at $|F|\le\delta_c$ and one at
$|F|\ge\delta_c+\epsilon$; at $\delta_c=0$ it is zero at
$|F|\le\epsilon$ and one at $|F|\ge2\epsilon$.
Its safe budgets are $L_z=2B|\kappa_c|$ and

$$
L_F^{[\epsilon]}=
\begin{cases}
6B(\delta_c^{-1}+\epsilon^{-1}),&\delta_c>0,\\
6B/\epsilon,&\delta_c=0.
\end{cases}
\tag{NPF.19}
$$

Let $D_{H,N}^{[\epsilon]}$ be (NPF.14) with these exact
observation budgets. Then its original stationary Doob variance obeys

$$
\operatorname{Var}(H_N^{\rm hard})
\le\frac{2D_{H,N}^{[\epsilon]}}N
                   +2B^2\beta_N(\epsilon),\qquad
\beta_N(\epsilon)=P_{\pi_N}^e
 [a_{I_{\rm root}}^+=1,
       ||F_{I_{\rm root}}|-\delta_c|\le2\epsilon].
\tag{NPF.20}
$$

Here $I_{\rm root}$ is an independent uniform original row.
$\beta_N$ is a native boundary probability, not an assumed
anti-concentration hypothesis. In the existing reset phase
$a_x=0$, it has the following entirely primitive upper bound.
Use query radius $R_x=R_D$ and empty spatial window in
(NQG.28)--(NQG.30), and let $\mathfrak B$ be its already
derived entire-force boundary profile (NQG.27).
For $R\ge1$, set $R_F=R+1$, $L_F=1/\epsilon$,
$L_0=2(c+a_Y)$ and $B_H=1$ in that coefficient:

$$
C_{\rm band}(\epsilon,R)
=2L_\psi+\frac{h_\tau}{d_x}
 \left\{2(c+a_Y)+\frac\nu\epsilon
 [(c+a_Y)+\ell_\rho(Z_2+R+1)(1+t(c+a_Y))]\right\}
 +\frac\nu\epsilon[1+\ell_\rho(Z_2+R+1)].
$$

With $C_F$ the original selected-root variance (NQG.30),
every declared $p>2$ has

$$
\begin{aligned}
\beta_N(\epsilon)\le\mathcal B_N(\epsilon,R):=
\frac1{\alpha_N\min e_N}\Bigg\{
&\frac{Z_p^p}{R^p}+h_\tau|D|
 \bigg[\mathfrak B(4\epsilon)\\
&+\frac{h_\tau}{d_x}
 \left(C_{\rm band}(\epsilon,R)\sqrt{\mathcal R_{A,N}}
                  +\frac1\epsilon\sqrt{C_F/N}\right)\bigg]\Bigg\},\\
Z_p={}&cV_c+ct\lambda(R_D+\sigma_Jg_{d,p})+qg_{d,p}.
\end{aligned}
\tag{NPF.21}
$$

Take $\epsilon\le1/16$ when the inherited profile uses its
declared $4\epsilon\le1/4$ domain. Consequently

$$
\mathcal V_{H,N}^{\rm hard}=
\min\left\{B^2,\inf_{\substack{0<\epsilon\le1/16\\R\ge1}}
 \left[\frac{2D_{H,N}^{[\epsilon]}}N
                         +2B^2\mathcal B_N(\epsilon,R)\right]\right\}
\longrightarrow0.
\tag{NPF.22}
$$

The literal hard empirical color concentrates about its OWN
stationary mean at this explicit variance scale.
This formula does not assert $\mathcal V_{H,N}^{\rm hard}=O(N^{-1})$.
:::

:::{prf:proof}
The two original readouts differ only when the original row
force is in the displayed threshold band. Pointwise,
$|H_N^{\rm hard}-H_N^{[\epsilon]}|
 \le B N^{-1}\sum_i a_i^+
                   \mathbf1_{||F_i|-\delta_c|\le2\epsilon}$.
The squared averaged indicator is at most that averaged indicator,
so its squared expectation is at most $B^2\beta_N(\epsilon)$.
The variance triangle inequality in squared form and (NPF.13)
give (NPF.20), retaining the actual correlated color array.

For the primitive bound dominate the band indicator on
$|z|\le R$ by a smooth scalar root test which is one on
$||F|-\delta_c|\le2\epsilon$ and zero outside the
$4\epsilon$ band, with velocity support $|z|\le R+1$.
It has $L_F\le1/\epsilon$, velocity coefficient at most two,
and bound one. Its conditional original terminal-root error
is (NQG.29) with the EMPTY window. In that expression the
same-record force field and posterior coordinate costs give
exactly $C_{\rm band}(\epsilon,R)$.
The whole stationary raw root conditional on terminal $x$
has the actual preparation Bayes factor at most $h_\tau/d_x$.
Its limiting reference band probability is at most
$\mathfrak B(4\epsilon)$, uniformly for $x\in D$,
by the previously proved primitive entire-force profile.
The original terminal density is at most $h_\tau$;
integrating over the actual box therefore costs $h_\tau|D|$.
The force-input velocity has original $p$th moment at most
$Z_p^p$, from the unchanged jitter, convex first kick and OU.
Markov bounds its omitted $|z|>R$ event by $Z_p^p/R^p$.
Finally the exact full Doob/raw one-update density comparison
costs $1/(\alpha_N\min e_N)$. This proves (NPF.21).
No survival-selection correction is needed on a root $x\in D$:
the actual surviving event is then certain.

For fixed $\epsilon,R$, the preparation and force sampling
errors tend to zero and $D_{H,N}^{[\epsilon]}$ is bounded.
The remaining primitive profile tends to zero as
$\epsilon\downarrow0$, and $Z_p^p/R^p$ tends to zero as
$R\to\infty$. The positive eigenfunction/survival denominators
are uniformly bounded below. Taking these successive limits
proves that the finite infimum in (NPF.22) tends to zero.
Its concentration assertion is Chebyshev at that exact
finite-$N$ variance scale. The source noise and hard threshold
are unchanged throughout.
:::

(sec-npf-witness)=
## 6. Evaluated active count phase and the remaining Gaussian population law

:::{prf:corollary} Positive complete-instrument fluctuation witness
:label: cor-npf-positive-witness

Keep every original parameter of (PC.37), its exact
$V=V_{\rm crit}/2$, $\omega$ from (PC.39), certified
$\bar q=3/4$, original $\delta_c=10^{-12}$,
fixed supplied $\kappa_c=1$, and a Hermitian test
$\|A\|_F\le1$ with the taper (NPF.3).
Diagnostic evaluations of the exact primitive formulas are

$$
\begin{gathered}
Z_2\simeq3.095302660,\quad
K_X\simeq6.454594113\,10^{10},\quad
K_w=K_J\simeq3.540732525\,10^{11},\\
C_{\rm inst}\simeq1.448020645\,10^{46},\quad
L_H\simeq3.120254855\,10^{14},\\
C_H\simeq5.890418643\,10^{59},\quad
C_D\simeq8.553525719\,10^{59},\quad
\log C_D\simeq137.9988640,\\
\mathcal C_{H,\infty}\simeq5.122349944\,10^{64},\qquad
\log\mathcal C_{H,\infty}\simeq148.9990593.
\end{gathered}
\tag{NPF.23}
$$

The exact constants define the bounds; these large values
diagnose a conservative positive certificate.
The actual update schedule $n_N=\lceil N^{10}\rceil$
satisfies (NPF.18), giving full native color Brownian path
limits along its actual covariance subsequences.
The unchanged larger-cap reference fails the inherited
contraction test and is not assigned these population bounds.
:::

:::{prf:proof}
Substitute the original complete witness into (NPF.4)--(NPF.16).
The inherited $A_{\rm prep}=2[A_D+10M_2(C)]$ is finite and
positive; $A_D\simeq3.675098301\,10^{21}$ and
$C_{\rm stat}\simeq3.025074197\,10^{30}$ are its already
evaluated original constants. All denominator tests are
the same strict positive ones, and its fixed phase lies
inside the original supplied finite range.
The displayed schedule has $n_N/N^9\to\infty$.
Apply the preceding actual-instrument theorems.
:::

:::{prf:remark} Exact missing dependency for an instantaneous full population CLT
:label: rem-npf-population-clt-dependency

The complete smooth native tests now have $N^{-1}$ stationary
variance, population-scale tightness, a population-uniform
complete time covariance and actual joint long-time/population
path limits. Their instantaneous centered subsequential laws
are not thereby identified as Gaussian.
For that identification a limit of the COMPLETE population
chronological bracket or a proved linearization of its empirical
preparation and both dense-force stages is still required.
Global sampled normalization, accepted forest components and
their common Haar variables contribute to that bracket.
They cannot be discarded because the terminal Gaussians are independent.

For the literal unweighted hard projector, the derived entire-force
profile proves the variance scale (NPF.22). It does not provide
an $O(\epsilon)$ boundary density or a uniform second moment
of the number of native threshold crossings under one source-block
replacement. Those are precise additional deductions needed
to obtain its $N^{-1}$ variance by this route; they are not
assumed properties of the unknown stationary law.
Likewise the strictly positive finite-population complete color
fiber of {prf:ref}`thm-nfs-complete-color-positive` does not give
a positive limiting covariance for the unweighted empirical mean.
The current full bracket, its null coboundary directions and all
actual parameters remain specified by the complete instrument.
:::
