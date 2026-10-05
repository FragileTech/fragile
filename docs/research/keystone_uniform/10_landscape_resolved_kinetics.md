# Landscape evaluation of the existing regional Keystone kinetic ledger

(sec-kul-regional-record)=
## 1. Fixed force, regional profiles, and the observable

This record evaluates the method of
{prf:ref}`thm-slkd-full-position`,
{prf:ref}`thm-slkd-structural-variance-threshold`,
{prf:ref}`thm-slc-evaluated-basins`, and
{prf:ref}`thm-slc-local-attraction` on the unchanged viscous reference.
Its signed full-state terms use the algebra of (SCK.2)--(SCK.3), and
its source terms use the same retained measurements, accepted plans and
component-Haar law as (SCK.P1)--(SCK.P4). It does not introduce a
contraction comparison between populations in different attainable phases.

The force follows `Benchmark::Rastrigin::expression` in
`algorithmic-gas/crates/benchmarks/src/lib.rs`, the identical objective
in `physics/fitness.rs`, and `rastrigin` in
`src/fragile/fractalai/core/benchmarks.py`:

$$
U(x)=\sum_k[x_k^2+10(1-\cos(2\pi x_k))],\quad
F_k(x)=-2x_k-20\pi\sin(2\pi x_k),\quad
U''(u)=2+40\pi^2\cos(2\pi u).                         \tag{KUL.1}
$$

Write $t=h/2$, $c=e^{-\gamma h}$, $b=t(1+c)$,
$\eta=tb$, $\tau^2=t^2q^2+s^2$. All other reference parameters
are those in {prf:ref}`def-ku-preparation-record`, including
$h=.04$, $\gamma=b_O=\rho=1$, $\nu=.3$,
$\sigma_J=\sigma_x=.1$, $V=2$, $\alpha_{\rm col}=.5$ and
$D=[-2,2]^3$. In either normalization $0\le t\nu=.006\le1$.

For $k=-2,\ldots,2$, the force-root $\zeta_k$ lies in
$[k-1/8,k+1/8]$, with

$$
m=2+20\sqrt2\pi^2\le U''\le M=2+40\pi^2,
\qquad |\zeta_k-k|\le2|k|/m.                            \tag{KUL.2}
$$

The nonzero roots point inward. The basin boundaries are the actual
unstable roots, not midpoint cuts. The phase cells and exterior label
$\dagger$ are those of the landscape partition; their transition masses
are retained below. A force-root is an anchor for the existing regional
calculation. It is not identified with the noisy full-state stationary law.

Indeed, on the displayed interval the cosine is at least
$\sqrt2/2$, proving (KUL.2). For $k=1,2$,
$U'(k-1/8)=2k-1/4-10\sqrt2\pi<0$ and $U'(k)=2k>0$.
Strict increase proves the unique inward root; oddness gives the
negative roots. The mean-value theorem between $k$ and its root gives
the displayed displacement bound. Likewise on
$[k+3/8,k+5/8]$ the curvature is at most
$2-20\sqrt2\pi^2<0$, and the endpoint derivatives have opposite
signs for the internal barriers $k=-2,-1,0,1$.

For a single phase with center $z\in\{\zeta_k\}_{k=-2}^2{}^d$,
write $Q_z=z+[-R,R]^d$, $R=1/16$ and $J=1/64$.
Then $R+J+4/m<1/8$, so the enlarged core has the same $m,M$.
The actual landing map $g_h(x)=x+\eta F(x)$ has core Lipschitz
constant

$$
\rho_z=\max\{|1-\eta m|,|1-\eta M|\}
=0.7794860369297958,
\qquad \rho_z^2=0.6075984817685189.                     \tag{KUL.3}
$$

The centered variance is
$W_N(X)=N^{-1}\sum_i|X_i-\bar X|^2$.
The centroid error $|\bar X-z|^2$ is a separate term:
$N^{-1}\sum_i|X_i-z|^2=W_N(X)+|\bar X-z|^2$.
A common translation has zero centered discrepancy. No root or orbit
separation is therefore inserted as a negative centered-variance term.

(sec-kul-centered-ledger)=
## 2. The existing centered positional bridge with both normalizations

:::{prf:theorem} Exact viscous extension of the centered positional bridge
:label: thm-kul-viscous-centered-bridge

After the complete actual preparation, let $X,V^C$ denote its position
and collision-velocity arrays, and put

$$
\widehat V=W_XV^C,\qquad
W_X=I-t\nu L_X\quad\hbox{(count)},\qquad
W_X=(1-t\nu)I+t\nu P_X\quad\hbox{(row)}.
$$

Both matrices are the actual first-kick matrices. The completed positions
have exactly the conditional law

$$
X_i^+=g_h(X_i)+b\widehat V_i+\tau Z_i,
\qquad Z_i\stackrel{\rm iid}{\sim}N(0,I_d),             \tag{KUL.4}
$$

given the entire prepared swarm. Consequently

$$
\begin{aligned}
\mathbb E[W_N(X^+)\mid\mathrm{prep}]={}&
W_N(g_h(X))+2b\operatorname{Cov}_N(g_h(X),\widehat V)\\
&+b^2W_N(\widehat V)+(1-N^{-1})d\tau^2,\\
\mathbb E[\bar X^+\mid\mathrm{prep}]={}&
\overline{g_h(X)}+b\overline{\widehat V},\\
\mathbb E[|\bar X^+-z|^2\mid\mathrm{prep}]={}&
|\overline{g_h(X)}+b\overline{\widehat V}-z|^2+d\tau^2/N.
\end{aligned}                                                   \tag{KUL.5}
$$

In count mode $\overline{\widehat V}=\bar V^C$. In row mode
$\overline{\widehat V}=\bar V^C+t\nu(\overline{P_XV^C}-\bar V^C)$;
this normalization-dependent centroid motion is retained.
The second force evaluation and the original cap affect the joint
position/velocity law and its next update; neither changes (KUL.4).

Equivalently, the five terms of
{prf:ref}`thm-slkd-full-position` remain exactly its signed polynomial,
with $V$ there replaced by $\widehat V$ here. The preparation increment
$H_x$ is the unchanged signed source/centering/jitter ledger. Thus the
Keystone contribution, its incoming donor excess, and the two negative
centering corrections are composed in their original order.

*Proof.* The actual B1 velocity is
$u=W_XV^C+tF(X)$. Its OU velocity is $w=cu+q\xi$ and
its A2 position is $y=X+bu+tq\xi$.
Final position diffusion gives (KUL.4), independently of B2 and the
velocity cap. Expand the centered empirical square. Its Gaussian
average square contributes $d\tau^2$ and its empirical-mean square
contributes $d\tau^2/N$, proving (KUL.5).
The count matrix preserves summed velocity; the row formula follows
directly from its actual normalized matrix. This proof uses conditional
independence only after the full prepared swarm has been fixed.
$\square$
:::

:::{prf:proposition} Same-phase and between-phase force profiles
:label: prop-kul-force-phase-profiles

If $x,y$ lie in one enlarged core, then the pair profiles in (SLKD.T1)
have $m$ from (KUL.2), $b=0$, $L=M$, $J=0$.
The sharper map profile is
$|g_h(x)-g_h(y)|\le\rho_z|x-y|$.
For $x$ in the core of $z_a$ and $y$ in the core of $z_b$, write
$x=z_a+r_a$, $y=z_b+r_b$. The exact external-force increment is

$$
F(x)-F(y)=-K_a(x)r_a+K_b(y)r_b,
\qquad mI\preceq K_a,K_b\preceq MI,                     \tag{KUL.6}
$$

where each diagonal secant is the integral of $U''$ from its own root.
For a declared reference $k\in[m,M]$, its regional excess obeys

$$
|F(x)-F(y)+k(x-y)|
\le k|z_a-z_b|+\max\{k-m,M-k\}(|r_a|+|r_b|).            \tag{KUL.7}
$$

This is a basin/passage entry in the existing regional force table.
The root separation is present in the cross-phase defect and is not
removed by applying a within-core curvature to the whole population.
Outside the cores, the actual global profiles are
$|F(x)|\le2|x|+20\pi\sqrt d$ and
$\|DF(x)\|\le M$; their signed curvature can be negative.

*Proof.* Integrate (KUL.2) along the appropriate core segments.
The derivative of $g_h$ lies in $[1-\eta M,1-\eta m]$.
For different roots add and subtract $k(x-y)$ in (KUL.6).
The global bounds follow directly from the sine and cosine formulas.
$\square$
:::

(sec-kul-unrestricted-jitter)=
## 3. Exact Gaussian evaluation of the existing within-well force profile

:::{prf:lemma} Unrestricted Rastrigin jitter moments at a declared source
:label: lem-kul-jitter-periodic-profile

Put $\omega=2\pi$, $\ell=1-2\eta$, $A=20\pi\eta$.
For a frozen scalar source $\mu$ and the actual jitter standard
deviation $\sigma\in\{0,\sigma_J\}$, define
$a_\sigma=e^{-\omega^2\sigma^2/2}$,
$b_\sigma=e^{-2\omega^2\sigma^2}$, and

$$
\begin{aligned}
m_\sigma(\mu)&=\ell\mu-Aa_\sigma\sin(\omega\mu),\\
k_\sigma(\mu)&=\ell^2\sigma^2
-2\ell A\omega\sigma^2a_\sigma\cos(\omega\mu)\\
&\quad+A^2\left\{\tfrac12[1-b_\sigma\cos(2\omega\mu)]
-a_\sigma^2\sin^2(\omega\mu)\right\}.
\end{aligned}                                                   \tag{KUL.8}
$$

These are exactly the mean and variance of
$g_h(\mu+\sigma Z)$, $Z\sim N(0,1)$.
Let $\mu_i$ be the actual frozen sources under one accepted plan,
$A_i$ its actual copy indicators, and let $m_i$ and $k_i$ be the
coordinatewise mean and summed variance in (KUL.8) with
$\sigma=A_i\sigma_J$. Conditional on this plan and its component
rotations, the exact source-force contribution to centered variance is

$$
\mathbb E_J W_N(g_h(X))
=W_N(m)+(1-N^{-1})\frac1N\sum_i k_i.                    \tag{KUL.9}
$$

Neither Gaussian jitter nor its periodic-force value is truncated.
All dependence of the sampled global fitness on its complete measurement
vector remains in the actual plan probabilities.

*Proof.* Integrate $\cos(\omega\sigma Z)$ and
$\sin(\omega\mu+\omega\sigma Z)$ against the standard Gaussian.
Differentiating its characteristic function gives
$\mathbb E[Z\sin(\omega\mu+\omega\sigma Z)]
=\omega\sigma a_\sigma\cos(\omega\mu)$.
Use $\sin^2 u=(1-\cos2u)/2$ to expand the variance, obtaining
(KUL.8). Given the frozen plan, jitters are independent across rows.
For independent, possibly nonidentically distributed rows, the average
variance minus the variance of their average is precisely
$(1-N^{-1})N^{-1}\sum_i k_i$. This proves (KUL.9).
$\square$
:::

:::{prf:theorem} Evaluated centered within-core bridge with signed gate feedback
:label: thm-kul-centered-regional-attraction

Suppose all current frozen sources in a plan lie in one $Q_z$.
Set

$$
r_*=R+4/m,\quad c_* =\cos(\omega r_*),\quad
c_{2,*}=\cos(2\omega r_*),\quad a_J=e^{-\omega^2\sigma_J^2/2},
$$
$$
\rho_J=\ell-A\omega a_J c_*,\qquad
K_J=\ell^2\sigma_J^2-2\ell A\omega\sigma_J^2a_Jc_*
+\tfrac12A^2[1-e^{-2\omega^2\sigma_J^2}c_{2,*}].          \tag{KUL.10}
$$

For the fixed reference,
$r_*<1/8$, $0<\rho_J<1$ and $K_J>0$.
The primitive expressions evaluate to
$\rho_J=.773229643494586$, $\rho_J^2=.5978840815787646$,
and $K_J=.006371638355086486$ per coordinate.
Define the actual gate-refresh array

$$
d_i=-A(1-a_J)(1-A_i)\sin(\omega\mu_i),\qquad
\mathcal R_{\rm gate}=2\operatorname{Cov}_N(m_{\sigma_J}(\mu),d)
+W_N(d).                                                   \tag{KUL.11}
$$

Then, after averaging the same full source law and Haar law as the
Keystone ledger, the original complete position update obeys

$$
\begin{aligned}
\mathbb E W_N(X^+)\le{}&
\rho_J^2\mathbb E W_N(\mu)+\mathbb E\mathcal R_{\rm gate}
+(1-N^{-1})dK_J\mathbb E\bar A\\
&+2b\mathbb E\operatorname{Cov}_N(g_h(X),W_XV^C)
+b^2\mathbb E W_N(W_XV^C)+(1-N^{-1})d\tau^2.             \tag{KUL.12}
\end{aligned}
$$

The source variance is exactly the zero-jitter part of
{prf:ref}`thm-slkd-signed-cloning`. Consequently it may be replaced,
under that theorem's actual Keystone hypotheses, by

$$
W_N(S)-\theta k_{\rm key}W_N(S)^p
+\theta E_{\max}/N^2+\mathbb E_{\mathbf F}\Gamma_\theta
-\mathbb E_{\mathbf F}|\bar t|^2
-N^{-2}\mathbb E_{\mathbf F}\sum_i\sigma_{i,\rm source}^2,
\quad p=5+4d.                                               \tag{KUL.13}
$$

Here $\sigma_{i,\rm source}^2$ excludes recipient jitter, which has
already been integrated exactly in (KUL.8). Incoming donor flux,
shared normalization, gate-refresh feedback, signed kinetic covariance,
and centroid correction all retain their signs. There is no uncomputed
population contraction coefficient in (KUL.12).

*Proof.* Throughout $Q_z$, $\cos(\omega\mu_k)\ge c_*>0$.
The derivative of $m_{\sigma_J}$ is
$\ell-A\omega a_J\cos(\omega\mu)$, positive and at most $\rho_J$.
Thus its centered variance is at most $\rho_J^2W_N(\mu)$ by the
pairwise variance identity. Exactly
$m_i=m_{\sigma_J}(\mu_i)+d_i$; expanding its centered variance gives
the signed refresh term (KUL.11). In (KUL.8), use
$\cos(\omega\mu)\ge c_*$,
$\cos(2\omega\mu)\ge c_{2,*}$ and discard only the nonpositive
$-A^2a_J^2\sin^2(\omega\mu)$. This bounds each accepted coordinate
variance by $K_J$, while the persisting coordinate variance is zero.
Insert (KUL.9) in (KUL.5), proving (KUL.12).
Apply the already proved zero-jitter signed cloning algebra to the
same frozen plan to obtain (KUL.13). No auxiliary donor model has been
used. $\square$
:::

(sec-kul-joint-signed)=
## 4. Both kicks, the actual cap, and noise-correlated signed terms

:::{prf:proposition} Root-anchored evaluation of the signed two-force polynomial
:label: prop-kul-root-signed-two-kick

For a fixed root anchor array $O_i$, put $r=X-O$, $v=V^C$ and

$$
f=F(X)-\nu L_Xv,\quad u=v+tf,\quad w=cu+q\xi,\quad
y=X+bu+tq\xi,\quad g=F(y)-\nu L_yw,\quad z=w+tg.
$$

Use $L=(I-P)$ in row mode and the actual count Laplacian in count
mode. At a singleton the viscous term is zero. Let
$Q_N(r,v)=\alpha\|r\|_{2,N}^2+2\beta\langle r,v\rangle_N+
\gamma_P\|v\|_{2,N}^2$, $\alpha\gamma_P>\beta^2$,
and write
$R_0=r+bv+\eta f$, $Z_0=cv+ctf+tg$.
Let $\mathscr K(r,v,f,g)$ be the exact polynomial (SCK.2), with
its kinetic symbols translated to $t,c,b,\eta$ here. The actual cap
correction is

$$
\mathscr C=2\beta\langle y-O,C_V(z)-z\rangle_N
+\gamma_P(\|C_V(z)\|_{2,N}^2-\|z\|_{2,N}^2).
$$

The complete single-population kinetic increment is exactly

$$
\begin{aligned}
\mathbb E\Delta Q_N={}&\mathbb E[\mathscr K+\mathscr C]
+dq^2(\alpha t^2+2\beta t+\gamma_P)+\alpha ds^2\\
&+2qt(\beta t+\gamma_P)\mathbb E\langle\xi,g\rangle_N.
\end{aligned}                                                   \tag{KUL.14}
$$

The last signed term cannot be set to zero: B2 depends on the OU
draw. For the external Rastrigin force it is explicitly

$$
\mathbb E[\langle\xi,F(y)\rangle_N\mid\mathrm{prep}]
=-tq\left[2d+\frac{40\pi^2}{N}
 e^{-2\pi^2t^2q^2}\sum_{i,k}\cos(2\pi[X_i+bu_i]_k)\right].
                                                               \tag{KUL.15}
$$

For count viscosity put $M_i=X_i+bu_i$,
$B_\rho=\rho^2+2t^2q^2$, and
$H_{ij}=(\rho^2/B_\rho)^{d/2}
\exp[-|M_i-M_j|^2/(2B_\rho)]$. Its exact contribution is

$$
\begin{aligned}
\mathbb E[\langle\xi,-\nu L_yw\rangle_N\mid\mathrm{prep}]
=-\frac{\nu}{2N^2}\sum_{i\ne j}H_{ij}\Big[&
-\frac{2ctq}{B_\rho}(M_i-M_j)\cdot(u_i-u_j)\\
&+q\left(\frac{2d\rho^2}{B_\rho}
+\frac{4t^2q^2|M_i-M_j|^2}{B_\rho^2}\right)\Big].        \tag{KUL.16}
\end{aligned}
$$

For row viscosity and $N\ge2$, the exact contribution is

$$
-\nu qd+\frac{\nu tq}{\rho^2}\mathbb E\left[
\overline{P_y(y\cdot w)}-\overline{(P_yy)\cdot(P_yw)}
\,\middle|\,\mathrm{prep}\right].                          \tag{KUL.17}
$$

It is a specified Gaussian integral with the original positive row
degrees. The absolute value of its second term is at most
$2\nu tq\overline C_d\mathcal Y_2\mathcal W_2/\rho^2$,
where
$\mathcal Y_2^2=\|X+bu\|_{2,N}^2+dt^2q^2$,
$\mathcal W_2^2=c^2\|u\|_{2,N}^2+dq^2$.
This evaluates a finite $N$-independent envelope without an assumed
degree floor. The signed integral should be retained when it is sharper.

*Proof.* The actual differences from the root with zero velocity are
$(y-O,z)=(R_0+tq\xi,Z_0+q\xi)$.
Expand its quadratic, subtract the input quadratic, and add the
original cap correction. The deterministic expansion is precisely
(SCK.2). In the linear Gaussian term only $tg$ in $Z_0$ has OU
dependence, giving (KUL.14). The independent final position innovation
contributes exactly $\alpha ds^2$.
Gaussian integration by parts and the cosine characteristic function
give (KUL.15).

For (KUL.16), symmetrize the count pair force. For $i\ne j$ set
$Z=\xi_i-\xi_j\sim N(0,2I_d)$. Complete the square in
$\exp[-|M_i-M_j+tqZ|^2/(2\rho^2)]$.
Its weighted first moment is
$-2tq(M_i-M_j)H_{ij}/B_\rho$ and its weighted squared moment is
$[2d\rho^2/B_\rho+4t^2q^2|M_i-M_j|^2/B_\rho^2]H_{ij}$.
Dot with $c(u_i-u_j)+qZ$ to obtain the displayed formula.
For (KUL.17), differentiate the actual self-excluded row weights:
$\partial_{\xi_i}p_{ij}=(tq/\rho^2)p_{ij}(y_j-(P_yy)_i)$.
The direct derivative of $-\nu w_i$ contributes $-\nu qd$;
the normalized derivative gives exactly the covariance in (KUL.17).
The Gaussian column lemma bounds each of its two averaged products
by $\overline C_d\|y\|_2\|w\|_2$. Conditional Cauchy--Schwarz
gives the stated envelope. All differentiations are justified by the
linear-growth force and finite Gaussian moments.
$\square$
:::

(sec-kul-orbit-flux)=
## 5. Target changes, phase transfers, and terminal normalization

:::{prf:proposition} Exact phase-target changes preserve centroid and orbit motion
:label: prop-kul-target-switch

For an output row $(x^+,v^+)$, let its old declared orbit anchor be
$(o,u)$ and its new anchor be $(o+\Delta o,u+\Delta u)$.
With $r=x^+-o$, $z=v^+-u$, changing the anchor changes the same
quadratic by exactly

$$
\begin{aligned}
\Delta_{\rm target}Q={}&-2\alpha r\cdot\Delta o+\alpha|\Delta o|^2
-2\beta[r\cdot\Delta u+z\cdot\Delta o]+2\beta\Delta o\cdot\Delta u\\
&-2\gamma_Pz\cdot\Delta u+\gamma_P|\Delta u|^2.          \tag{KUL.18}
\end{aligned}
$$

For roots with zero target velocity, this is
$-2\alpha(x^+-z_a)\cdot(z_b-z_a)
+\alpha|z_b-z_a|^2-2\beta v^+\cdot(z_b-z_a)$.
The root or orbit move has no assumed restoring sign.
If an anchor path is declared, its residual is obtained by evaluating
the same noiseless B1/A1/O/A2/B2/cap stages on its anchor array, with
the actual two matrices $W_X,W_y$, and subtracting the declared next
anchor. Explicitly, for the declared arrays $(O,U)$ and $(O',U')$,

$$
u^\circ=W_XU+tF(O),\quad y^\circ=O+bu^\circ,\quad
z^\circ=cW_yu^\circ+tF(y^\circ),\quad
e_o=y^\circ-O',\quad e_u=C_V(z^\circ)-U'.              \tag{KUL.18a}
$$

These are algebraic anchor stages using the actual graph matrices;
they do not define a substitute population kernel.
Applying (KUL.18) to this residual retains its position,
velocity, cap, and force-path costs exactly; it does not presume that
the declared path is an actual stationary population phase.

For rectangular landscape phase cells $C_b\subset D$, the actual
one-row kinetic phase transfer, conditional on preparation, is

$$
p_{ib}=\prod_k\left[
\Phi((r_{b,k}-M_{i,k})/\tau)
-\Phi((l_{b,k}-M_{i,k})/\tau)\right],\quad M_i=X_i+bu_i.
                                                               \tag{KUL.19}
$$

The exterior probability is $1-\sum_b p_{ib}$.
Source-phase transfers are the same actual source probabilities of
(KU.2)--(KU.3), integrated against the recipient's Gaussian jitter
with the same interval formula at standard deviation $A_i\sigma_J$.
Integrate (KUL.18) on each actual transfer event. This is the signed
between-phase flux of the existing regional ledger, not an autonomous
Markov approximation for phase labels.

For any realized nonempty alive set, the exact variance decomposition is

$$
W_A=\sum_b\frac{M_b}{M_A}W_b+
\sum_b\frac{M_b}{M_A}|\bar x_b-\bar x_A|^2,\qquad
\frac1{M_A}\sum_{i\in A}|x_i-z_{b(i)}|^2
=\sum_b\frac{M_b}{M_A}[W_b+|\bar x_b-z_b|^2].             \tag{KUL.20}
$$

Thus within-phase dispersion, centroid/orbit deviation, and separation
between attainable phases remain distinct actual quantities.

*Proof.* Expand the completed quadratic after subtracting
$(\Delta o,\Delta u)$, proving (KUL.18).
The conditional position Gaussian (KUL.4) gives (KUL.19).
The source-stage formula follows from the actual recipient jitter law.
Finally decompose each $x_i-\bar x_A$ through its phase centroid.
The mixed term vanishes within every phase, giving (KUL.20).
$\square$
:::

:::{prf:proposition} Exact alive-centered variance after terminal classification
:label: prop-kul-alive-centered-terminal

Fix the full preparation and put
$p_i=P_{D,\tau}(M_i)$, $\mu_i^D=N(M_i,\tau^2I_d)(\cdot\mid D)$.
Let $m_i^D$ and $v_i^D$ be its mean and variance trace, computed
from the usual one-dimensional Gaussian interval integrals.
Define the alive variance to be zero for zero or one survivor.
The exact unconditioned, survival-weighted alive-centered variance is

$$
\mathbb E[W_A;M_A>0\mid\mathrm{prep}]
=\frac12\sum_{i\ne j}p_ip_j
[v_i^D+v_j^D+|m_i^D-m_j^D|^2]
\int_0^1 t(-\log t)\prod_{k\ne i,j}(1-p_k+p_kt)\,dt.
                                                               \tag{KUL.21}
$$

Average this numerator over the same complete preparation law and
divide by $1-\mathbb E_{\rm prep}\prod_i(1-p_i)$ for the
whole-update survivor-conditioned variance. Phase-specific versions
replace $D$ by its actual cell. The inverse-count moment bounds of
{prf:ref}`thm-ku-quadratic-binomial-survival` apply to this original
normalization. No fixed-slot quadratic substitutes for $W_A$.

*Proof.* Conditional on preparation the position rows are independent.
Use $W_A=(2M_A^2)^{-1}\sum_{i,j}A_iA_j|X_i-X_j|^2$.
Given $A_i=A_j=1$, the other survivor count is the sum of its
independent Bernoulli variables, and the two positions have laws
$\mu_i^D,\mu_j^D$. The identity
$1/(n+2)^2=\int_0^1t^{n+1}(-\log t)\,dt$ evaluates the exact
normalizing factor. Averaging and conditioning prove the result.
$\square$
:::

(sec-kul-excursions)=
## 6. Residence, excursions, and evaluated original-parameter bounds

:::{prf:lemma} Existing regional landing constants with both actual viscous matrices
:label: lem-kul-viscous-regional-landing

Suppose every eligible frozen copied source lies in $Q_z$, and retain
the original collision bound $V_c$, jitters and both kinetic kicks.
Put $H=\rho_z(R+J)+bV_c$ and
$p_J=2\Phi(J/\sigma_J)-1$.
The constants in {prf:ref}`thm-slc-evaluated-basins` remain valid for
both positive-viscosity normalizations:

$$
s_z=p_J^d\left[\Phi((R-H)/\tau)-\Phi((-R-H)/\tau)\right]^d,
$$
$$
p_{zj}=p_J^d(2R_j)^d(2\pi\tau^2)^{-d/2}
\exp\left[-\frac{\sum_k(|z_{j,k}-z_k|+R_j+H)^2}{2\tau^2}\right].
                                                               \tag{KUL.23}
$$

The first floor is positive even for $H\ge R$, although then the
small-exit conclusion is uninformative. The counts dominate
$\operatorname{Bin}(N,s_z)$ and $\operatorname{Bin}(N,p_{zj})$,
respectively, for the completed raw position counts. When the source
and target cores lie inside $D$, these are also alive landing counts.
This applies directly to $z\in\{\zeta_{-1},\zeta_0,\zeta_1\}^d$.
For the boundary wells $\zeta_{\pm2}$, retain $Q_z\cap D$ as the
alive target: its coordinate interval relative to $z$ is
$[\max\{-R,-2-z_k\},\min\{R,2-z_k\}]$.
Replace each centered interval factor by the minimum of its Gaussian
interval probability at means $-H$ and $H$; the target-volume and
density-infimum calculation uses this actual truncated rectangle.
The raw core probability must not be substituted for that alive
probability. For cores inside $D$, the all-in-core residence factors are the existing
$s_z^{Nn}$ and $\min\{1,nN(1-s_z)\}$.

*Proof.* Fix the actual frozen source/gate pattern and component
rotations. The jittered matrix $W_X$ can depend on every jitter;
it must not be held fixed before those jitters are sampled.
Its convex kick nevertheless obeys
$|(W_XV^C)_i|\le V_c$ for every array.
On row $i$'s own coordinatewise good-jitter event, its conditional
mean has coordinate distance at most $H$ from $z$.
Conditional on the entire preparation, (KUL.4) gives independent
row Gaussian landing events with their exact probabilities. Realize
each by an independent uniform rank variable. On the own-jitter good
event the landing rank threshold is at least the geometric floor in
(KUL.23); for the target cube use the same density-infimum proof as
{prf:ref}`thm-slc-evaluated-basins`.
The own-jitter good events are independent conditional on the frozen
pattern, with probability $p_J^d$ on accepted rows and one on
persisting rows. They are independent of the rank variables. Their
products give independent lower Bernoulli indicators; persisting rows
can be thinned to the same parameter. This proves the binomial
dominations despite the jitter-dependent viscous graph. Average the
frozen law and iterate the all-in-core bound as in the existing
residence proof. No simultaneous bounded-noise event is imposed on
the algorithm.
$\square$
:::

The coefficients in {prf:ref}`ex-slc-rastrigin-residence` and
{prf:ref}`ex-slc-rastrigin-attraction` retain their precise dependence
on $h$, noise, cap and curvature. At the unchanged reference,

$$
m=281.15456798555516,\quad M=396.78417604357435,\quad
\eta=.0007843157756609293,\quad \tau^2=.00041537673072267287.
$$

With the original speed envelope $V_c=4$,
$H=\rho_z(R+J)+bV_c=.21776050176732614$.
The sufficient $H<R$ residence test of that example is therefore
uninformative at this different parameter choice. Its speed-profile
threshold is explicitly
$V_*<(R-\rho_z(R+J))/b=.04086755397745709$.
Such a profile is a measured phase condition, not a reduction of the
configured cap. Noise can leave it and its departure must be charged.
For an interior core and the original bound $V_c=4$, the unrestricted positive floor in
(KUL.23) evaluates to $s_z\simeq4.09361595\,10^{-45}$.
This is a valid landing/establishment coefficient; it does not certify
long residence at the reference noise and speed scale.

The original jitter-cutoff attraction bound at $d=3$ evaluates to
$p_{\rm bad}=\min\{1,2d e^{-J^2/(2\sigma_J^2)}\}=1$,
$E_{\rm bad}=.5478224805405415$ and
$\rho_z^2r_{C,\rm core}=5.123561097182721$.
This uses the **phase-specific all-alive core** profile of
{prf:ref}`thm-slc-local-attraction`:
$\kappa_{C,\rm core}=\exp[-(4dR^2+4\lambda_{\rm alg}V^2)/(2\epsilon_C^2)]
=.13454462171467518$ and $r_{C,\rm core}=1+\kappa_{C,\rm core}^{-1}$.
Its proof uses the squashes' Lipschitz constant one, core positional
diameter $2\sqrt dR$, and entering velocity diameter $2V$.
It is distinct from the global feature floor $\kappa_C=e^{-4}$.
Likewise $p_{\rm bad}$ is a union-bound probability of failed
coordinatewise jitter control; the coordinatewise good-jitter
probability in (KUL.23) is instead
$p_J=2\Phi(J/\sigma_J)-1=.1241640336165897$.
These are conservative upper coefficients, not a dynamical failure.
Equations (KUL.8)--(KUL.13) evaluate the same unrestricted force
profile exactly and retain the actual signed donor flux instead of
that coarse copying factor.

Conditional on preparation, any enlarged core $z+[-R_2,R_2]^d$
has its exact B2 excursion probability from (KUL.19), with $\tau$
replaced by $tq$. If its conditional mean has coordinate distance
at most $H_2<R_2$, the existing Gaussian bound gives
$p_2\le\min\{1,2d e^{-(R_2-H_2)^2/(2t^2q^2)}\}$.
Use the actual mean when the coarse $H$ makes this bound one.
For an averaged regional observable of order $r<p$, the excursion
charge is its computed joint $p$th moment to the power $r/p$ times
the averaged excursion probability to the power $1-r/p$, exactly
as in {prf:ref}`lem-slc-excursion`.

All such moments are primitive evaluations. With
$g_{d,p}=[2^{p/2}\Gamma((d+p)/2)/\Gamma(d/2)]^{1/p}$,
the unchanged Rastrigin stages satisfy

$$
\begin{aligned}
X_p&=R_D+\sigma_Jg_{d,p},\\
U_p&=V_c+t(2X_p+20\pi\sqrt d),\\
W_p&=cU_p+qg_{d,p},\\
Y_p&=X_p+bU_p+tqg_{d,p},\\
Z_p&=H_pW_p+t(2Y_p+20\pi\sqrt d),\\
X_p^+&=Y_p+sg_{d,p},
\end{aligned}                                                   \tag{KUL.22}
$$

where $H_p=1$ in count mode and
$H_p=[1+t\nu(\overline C_d-1)]^{1/p}$ in row mode.
These are the existing Gaussian-column and linear-growth moment proofs
with the actual Rastrigin coefficients $2,20\pi\sqrt d$ inserted.
They cover both kicks, the unbounded OU velocity in B2, final position
diffusion, and every recipient jitter. The cap still bounds the final
velocity by its original radius $2$.
For example, (KUL.22) gives $Z_2=8.745996525698438$ in count
mode and $Z_2=68.23482085970537$ in row mode; the corresponding
fourth-moment norm budgets are $8.794145078916031$ and
$23.04367190689138$. These are uncapped budgets, while the actual
output still has its radius-$2$ cap.

For a phase-averaged law, use the exact regional source masses and
feedback test of {prf:ref}`def-slceg-regional-profiles` and
{prf:ref}`prop-slcs-regional-test`. A stationary phase $\pi$ must
satisfy its actual full-kernel stationary identity before invoking
{prf:ref}`prop-slcs-signed-bregman`; a force-root does not supply it.
The signed phase-production and refresh terms there are not discarded
when (KUL.12) gives a restoring within-core coefficient.

Every coefficient above is independent of $N$. The source-plan law,
the actual normalized phase masses, the displayed $N^{-1}$ variance
correction and $N^{-2}$ Keystone correction retain their population
dependence. A residence estimate for the event that **all** $N$ rows
stay in a core still has the original $1-s^N$ and $nN(1-s)$ factors;
it is not population-uniform whole-swarm trapping. Full-support Gaussian
innovations give positive transfer to every nonempty phase cell and to
the exterior. These phases are metastable, not absorbing.
