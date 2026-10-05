# A derived stationary-chaos regime for the complete marked count gas

(sec-native-phase-register)=
## 1. Complete parameters and the marked law metric

:::{prf:definition} Native phase-concentration register
:label: def-native-phase-register

Retain the complete execution record of
{prf:ref}`def-native-complete-execution-record` and the canonical dense
count restriction of {prf:ref}`def-native-stationary-closure-register`.
The force and reward providers are the configured quadratic providers
$F(x)=-\lambda x$, $R(x,v)=-\lambda|x|^2/2$ on
$D=[-L_D,L_D]^d$. Both kicks use the configured count-normalized Gaussian
viscosity. Every current-step measurement, actual sampled fitness, positive
standardizer and rescale, gate, donor width, self exclusion, mandatory
revival, frozen simultaneous copy, connected-component Haar collision,
recipient Gaussian jitter, OU innovation, smooth radial velocity cap,
final position noise and terminal mark is retained. The two donor roles
may have their existing uniform-width convention. No historical donor,
geometry feedback, elite or curl channel is enabled in this restriction.
Passive observations and physical calibration remain in the complete
record but do not enter this state transition. Other variant tags keep
their separate kernels.

Use $t,c,b,a_x,q,s,\tau,R_D$ from (SC.1), and put

$$
\alpha=1-t^2\lambda,\quad
D_c=c-t\lambda b,\quad k_c=1+2|\alpha_{\rm col}|,\quad
V_c=k_cV,\quad\ell_\rho=e^{-1/2}/\rho.
\tag{PC.1}
$$

Fix a proof upper bound $V_0>0$ for the configured cap and consider
$0<V\le V_0$. It is fixed independently of $N$. Suppose
$h,q,s,L_D,\rho>0$, $\alpha>0$, and $0\le t\nu\le1/2$.
All fitness floors, regularizers and finite donor widths are positive.
Let $a_0>0$ be any of the explicitly derived quadratic binomial row
floors of {prf:ref}`thm-ku-quadratic-binomial-survival`, evaluated with
$V_c=k_cV_0$. The same floor is valid for every smaller cap. Set
$m_0=a_0/2$ and $C=2/(\kappa_Cm_0)$, where

$$
D_0=2\sqrt{(R_x^{\rm feat})^2+
                \lambda_{\rm alg}(R_v^{\rm feat})^2},\qquad
\kappa_b=\exp[-D_0^2/(2\epsilon_b^2)],\quad b\in\{D,C\}.
\tag{PC.2}
$$

At a uniform-width convention set $\kappa_b=1$ and all weight coordinate
derivatives below to zero. Proof radii used in the inherited floor do
not truncate any innovation. The analytic innovation convention is
independent continuous Gaussian draws in real coordinates; a fixed-seed
finite arithmetic execution has its separately recorded law.

For a marked row $z=(x,v,a)$ define

$$
d_x(z,z')=|a-a'|+\min\{1,|x-x'|\},\qquad
d_v(z,z')=|v-v'|,\qquad d_\omega=d_x+\omega d_v,
\tag{PC.3}
$$

with $\omega>0$. Let $W_\omega$ be the transport distance for this cost.
On the capped marked row space it is bounded by $2+2\omega V$ and
metrizes weak convergence. In particular the cost retains the actual
dead coordinates and the discrete alive mark. The admitted input class
has terminally consistent marks, velocities bounded by $V$, and alive
mass at least $m_0$. The population map is the complete map
$\mathcal F=\mathcal F_h^{\nu,\mathrm{count}}$ of
{prf:ref}`def-cg-mf-kinetic-map`; its collision components are not cut off.
:::

(sec-native-phase-preparation)=
## 2. Linear preparation bounds with all collision components retained

:::{prf:definition} Explicit companion, fitness and gate coefficients
:label: def-native-phase-preparation-coefficients

Every constant in this definition is a function of (PC.1)--(PC.2) and
the configured canonical parameters. Put

$$
\begin{gathered}
L_D^{\rm pos}=\max\{1,2R_D\},\quad
S_b=\sqrt{D_0^2+\delta_D^2}-\delta_D,\quad
R_b=\lambda R_D^2/2,\quad L_R=\lambda R_DL_D^{\rm pos},\\
\ell_{xb}=\max\{1,e^{-1/2}/\epsilon_b\},\quad
\ell_{vb}=e^{-1/2}\sqrt{\lambda_{\rm alg}}/\epsilon_b,\\
B_{bx}=\frac{2(1+\ell_{xb})}{\kappa_bm_0},\qquad
B_{bv}=\frac{2\ell_{vb}}{\kappa_bm_0},\quad b\in\{D,C\}.
\end{gathered}
\tag{PC.4}
$$

For uniform donor weights take $\ell_{xb}=\ell_{vb}=0$; the eligibility
term $1$ in $B_{bx}$ remains. Define

$$
\begin{aligned}
A_{sx}&=L_D^{\rm pos}[1+2/(\kappa_Dm_0)]+2S_bB_{Dx},&
A_{sv}&=\sqrt{\lambda_{\rm alg}}[1+2/(\kappa_Dm_0)]+2S_bB_{Dv},\\
M_{sx}&=(A_{sx}+2S_b)/m_0,&M_{sv}&=A_{sv}/m_0,\\
T_{sx}&=(4S_bA_{sx}+6S_b^2)/m_0,&T_{sv}&=4S_bA_{sv}/m_0,\\
M_{rx}&=(L_R+2R_b)/m_0,&T_{rx}&=(4R_bL_R+6R_b^2)/m_0,\\
Q_{rx}&=(L_R+M_{rx})/\sigma_r+R_bT_{rx}/(2\sigma_r^3),\\
Q_{sx}&=(A_{sx}+M_{sx})/\sigma_s+S_bT_{sx}/(2\sigma_s^3),&
Q_{sv}&=(A_{sv}+M_{sv})/\sigma_s+S_bT_{sv}/(2\sigma_s^3).
\end{aligned}
\tag{PC.5}
$$

For $b=r,s$, let

$$
H_b=\frac{A_bp_b}{4}
 \max\{\eta_b^{p_b-1},(\eta_b+A_b)^{p_b-1}\}
 (\eta_{b'}+A_{b'})^{p_{b'}},\quad b'\ne b,
\tag{PC.6}
$$

with $H_b=0$ if $p_b=0$. These are the actual positive logistic-power
derivative coefficients. Put

$$
\begin{gathered}
F_* =\eta_r^{p_r}\eta_s^{p_s},\qquad
F^*=(\eta_r+A_r)^{p_r}(\eta_s+A_s)^{p_s},\\
L_g=\max\left\{\frac1{s_c(F_*+\epsilon_c)},
 \frac{F^*+\epsilon_c}{s_c(F_*+\epsilon_c)^2}\right\},\\
C_{Fx}=H_rQ_{rx}+H_sQ_{sx},\qquad C_{Fv}=H_sQ_{sv},\\
R_x=1+2B_{Cx}+L_g[1+2/(\kappa_Cm_0)]C_{Fx},\qquad
R_v=2B_{Cv}+L_g[1+2/(\kappa_Cm_0)]C_{Fv},\qquad
G=8e^{2C}.
\end{gathered}
\tag{PC.7}
$$

Let $g_1=\mathbb E|Z_d|$ for a standard $d$-Gaussian. The preparation
coefficients are

$$
\begin{aligned}
L_{Xx}&=L_D^{\rm pos}[1+2/(\kappa_Cm_0)]
                  +(2R_D+\sigma_Jg_1)R_x,&
L_{Xv}&=(2R_D+\sigma_Jg_1)R_v,\\
L_{Vx}&=2k_cVG R_x,&L_{Vv}&=k_c+2k_cVG R_v.
\end{aligned}
\tag{PC.8}
$$
:::

:::{prf:lemma} Complete marked preparation coupling
:label: lem-native-phase-preparation-coupling

For any coupling of two admitted entering laws let
$\delta_x=\mathbb E d_x(z,z')$ and
$\delta_v=\mathbb E|v-v'|$. Their actual population preparations have
a coupling retaining pre-jitter variables $(Y,I,V^c)$ and
$(Y',I',V^{c\prime})$, with $Y,Y'\in\overline D$, $I,I'\in\{0,1\}$,
and the unchanged recipient Gaussian $\zeta$ independent of those
variables. In this coupling
$X=Y+\sigma_JI\zeta$, $X'=Y'+\sigma_JI'\zeta$, and

$$
D_X:=\mathbb E|X-X'|\le L_{Xx}\delta_x+L_{Xv}\delta_v,\qquad
D_V:=\mathbb E|V^c-V^{c\prime}|
                  \le L_{Vx}\delta_x+L_{Vv}\delta_v.
\tag{PC.9}
$$

These coefficients are independent of population size. Fitness and
mandatory revival are retained, including every tie and every accepted
connected component.
:::

:::{prf:proof}
First use two paired deterministic arrays with at least $m_0N$ alive
rows each and $m_0N\ge2$. Write their average costs as $\delta_x,\delta_v$.
Both companion roles have denominators at least $\kappa_bm_0N/2$.
The Gaussian weight derivative in one feature argument is at most
$e^{-1/2}/\epsilon_b$. Squashing is nonexpansive. For physical position
differences less than one this proves the position bound in (PC.4);
for larger differences a weight changes by at most one. The velocity
bound is the second derivative coefficient in (PC.4).

Couple each pair of companion-index distributions by its common minimum
mass. Subtracting their unnormalized weights and denominators shows
that the failure probability for recipient $i$ is at most

$$
B_{bx}(d_{x,i}+\delta_x)+B_{bv}(d_{v,i}+\delta_v).
\tag{PC.10}
$$

The terms containing $1$ in $B_{bx}$ pay for changes in eligibility.
For a successfully paired index, its paired subprobability is bounded
above by $2/(\kappa_bm_0N)$ on every index. A common alive pair of
positions is separated by at most $L_D^{\rm pos}d_{x,i}$, and the
regularized diversity distance is nonexpansive in each feature argument.
Shift this distance by its constant minimum $\delta_D$, so its range is
$[0,S_b]$; the shift changes neither its variance nor its standardized
value. Summing successful-pair increments and paying $S_b$ on a failed
index proves that the expected average diversity difference over common
alive recipients is at most $A_{sx}\delta_x+A_{sv}\delta_v$.
The corresponding reward difference is at most $L_R\delta_x$.

For variables in $[0,B]$, over two alive pools of mass at least $m_0$,
an unnormalized common-pool difference $A$ and an eligibility discrepancy
$\varepsilon$ give mean difference at most $(A+2B\varepsilon)/m_0$.
Subtract the two restricted sums and then their reciprocal pool masses
to obtain this inequality. The second-moment difference is at most
$(2BA+2B^2\varepsilon)/m_0$. Adding at most $2B$ times the mean
difference gives the variance bound
$(4BA+6B^2\varepsilon)/m_0$. Here $\varepsilon\le\delta_x$.
This proves the coefficients $M,T$ in (PC.5), also for the random
sampled diversity statistics after expectation.

The derivative of $(u+\sigma^2)^{-1/2}$ on $u\ge0$ has absolute
value at most $1/(2\sigma^3)$. Subtract numerator and denominator in
each actual standardized score and use its bounded numerator range.
This proves the $Q$ coefficients. The logistic derivative is at most
$A_b/4$ and the positive power derivative gives (PC.6). Consequently
the expected average fitness difference on common alive rows is bounded
by $C_{Fx}\delta_x+C_{Fv}\delta_v$. This compares the actual sampled
fitness; diversity has not been replaced by its mean.

Freeze both complete measured fitness arrays. Couple clone donors by
(PC.10), and use one common uniform for the two actual gates. The
clipped gate is $L_g$-Lipschitz in the sum of its two fitness arguments;
the displayed coefficient follows by differentiating the unclipped ratio,
and clipping is nonexpansive. A successfully matched donor index has the
same upper subprobability bound as before. A common dead recipient has
gate one in both updates. A changed recipient mark costs at most its
mark discrepancy. Thus the average probability $\bar q$ that a
recipient's accepted outgoing outcome differs is at most

$$
\mathbb E\bar q\le R_x\delta_x+R_v\delta_v.
\tag{PC.11}
$$

These paired outcome draws can be independent by recipient after freezing
both measurement arrays. Each marginal graph is the actual ordered
forest: a live accepted edge strictly increases fitness, a dead vertex
points to a live donor, and outdegree is at most one. Its specified-edge
probabilities are at most $C/N$.

Let $E_i$ be the event that recipient $i$'s accepted outcome differs.
Condition on its paired outcomes alone, keeping the frozen measurements.
Delete row $i$'s outgoing edge from each graph. All remaining recipient
draws retain independence and their $C/N$ bound. The path-count proof
of {prf:ref}`lem-chaos-component-truncation` applies after this deletion
and gives expected component size at most $e^{2C}$ at any fixed seed.
Restoring the exposed edge joins at most its two endpoint components.
Therefore the sum of the expected sizes of the two components incident
to this changed outcome is at most $4e^{2C}$. The donor endpoints may
depend on the exposed row draw; once it is exposed they are fixed and
independent of the remaining draws. Every vertex whose full component
differs is in one of these incident components for some $E_i$.
The union bound and summation give expected affected fraction at most
$4e^{2C}\mathbb E\bar q\le G\mathbb E\bar q$.
This argument never conditions a component-size estimate on its being
an influential random component.

Use the same independent Haar matrix on every identical component.
On a common component the collision formula is
$\bar v+\alpha_{\rm col}O(v_i-\bar v)$. Summing its difference
over that component bounds the velocity discrepancy by
$k_c$ times the entering velocity discrepancy. On affected components
each collision velocity has norm at most $V_c$, so their discrepancy
is at most $2V_c$. This proves the $D_V$ bound in (PC.9).

For matched copying outcomes, the own or common donor position costs
at most $L_D^{\rm pos}[1+2/(\kappa_Cm_0)]\delta_x$ on average.
For a failed outcome both sources remain in $\overline D$, so pay
$2R_D$. Give each recipient its same original Gaussian jitter in
both updates. Changing its jitter indicator costs $\sigma_Jg_1$ and
can occur only on a failed copying outcome. Equations (PC.11) and
(PC.8) now give the $D_X$ bound.

Finally approximate any given marked-law coupling by paired deterministic
empirical arrays with alive fractions at least $m_0$; a strict floor
can first be used and then decreased to $m_0$. The bounded pre-jitter
variables and all Gaussian jitter moments make these extended preparation
couplings tight. The already proved one-step rooted-component consistency,
with the uniform truncation bound just verified, identifies both limiting
marginals as their complete population preparations. Lower semicontinuity
passes (PC.9) to the limit. The common independent recipient jitter remains
independent under this passage. This proves the lemma for the actual
population map without changing its component or innovation law.
:::

(sec-native-phase-local-density)=
## 3. A target-local B2 density bound and the configured cap

:::{prf:definition} Primitive weak-viscosity interval
:label: def-native-phase-viscosity-interval

Let $g_1=\mathbb E|Z_d|$ and retain the fixed cap upper bound $V_0$.
Define

$$
\begin{gathered}
M_e=|D_c|k_cV_0+t\lambda|c+a_x|(R_D+\sigma_Jg_1)+\alpha qg_1,\\
\overline E=2+t\lambda\rho e^{-1/2}+M_e,\qquad
\overline C=1+2t^2\lambda/e+t\ell_\rho(M_e+\overline E),\\
\nu_0=\min\{1/(2t),\alpha/(2t\overline C)\},\qquad
H_v=(2\pi q^2)^{-d/2}(\alpha/2)^{-d}.
\end{gathered}
\tag{PC.12}
$$

In the following results $0\le\nu\le\nu_0$. This interval contains
positive viscosities. All constants remain functions of the actual force,
noise, box, jitter and collision parameters. In particular $q$ is the
configured OU amplitude, rather than an added noise.
:::

:::{prf:lemma} Population B2 anti-concentration uniform over unbounded jitters
:label: lem-native-phase-b2-local-density

Consider a prepared population with $X=Y+\sigma_JI\zeta$,
$|Y|\le R_D$, $0\le I\le1$, $|V^c|\le k_cV_0$ and independent
standard Gaussian $\zeta$. Set

$$
U=V^c+t\nu\int K_\rho(X,X')(V^{c\prime}-V^c)\,d\pi(X',V^{c\prime}),
\quad y=a_xX+bU+tq\xi,\quad z=cU-ct\lambda X+q\xi.
\tag{PC.13}
$$

Here $\pi$ is that complete prepared population law and $\xi$ is its
original independent OU Gaussian. Let $\Lambda$ be the joint law of
$(y,z)$ and let

$$
u=z-t\lambda y+t\nu\int K_\rho(y,y')(z'-z)\,d\Lambda(y',z')
\tag{PC.14}
$$

be its actual uncapped B2 velocity. Conditional on every preparation
value, the density of $u$ on $B(0,1)$ is at most $H_v$. This includes
every unbounded realized clone jitter. The conclusion also applies to
interpolated preparations with $Y,I,V^c$ as above; no assertion of a
globally bounded B2 inverse derivative is made.
:::

:::{prf:proof}
Since $t\nu\le1/2$, the count kick is a convex average and $|U|\le k_cV_0$.
The exact identity

$$
z-t\lambda y=D_cU-t\lambda(c+a_x)X+\alpha q\xi
\tag{PC.15}
$$

gives $\int|z'-t\lambda y'|\,d\Lambda\le M_e$.
Fix a preparation value and put $p=\alpha X+tU$, so $y=p+tz$.
The conditional B2 map of the Gaussian variable
$z\sim N(cU-ct\lambda X,q^2I)$ is

$$
T_p(z)=\alpha z-t\lambda p
       +t\nu\int K_\rho(p+tz,y')(z'-z)\,d\Lambda.
\tag{PC.16}
$$

Write $e=z-t\lambda y$, $e'=z'-t\lambda y'$ and
$a_\Lambda(y)=\int K_\rho(y,y')d\Lambda$. At a target $T_p(z)=w$,

$$
(1-t\nu a_\Lambda(y))e
 =w-t\nu\int K_\rho(y,y')e'\,d\Lambda
        -t^2\nu\lambda\int K_\rho(y,y')(y'-y)\,d\Lambda.
\tag{PC.17}
$$

For $|w|\le W$, the Gaussian kernel bound
$\sup_r r e^{-r^2/(2\rho^2)}=\rho e^{-1/2}$ gives

$$
|e|\le E_W:=\frac{W+t\nu M_e+t^2\nu\lambda\rho e^{-1/2}}{1-t\nu}.
\tag{PC.18}
$$

Differentiate (PC.16). Substitute
$z'-z=e'-e+t\lambda(y'-y)$ in its kernel derivative.
Using $\sup_r(r^2/\rho^2)e^{-r^2/(2\rho^2)}=2/e$ gives, at such a
preimage,

$$
\|DT_p-\alpha I\|
 \le t\nu\{1+2t^2\lambda/e+t\ell_\rho(M_e+E_W)\}.
\tag{PC.19}
$$

At $W=1$, $t\nu\le1/2$ implies $E_1\le\overline E$.
The interval (PC.12) therefore bounds (PC.19) by $\alpha/2$.
Each preimage of a target in $B(0,1)$ is regular, has positive Jacobian,
and has every singular value at least $\alpha/2$.

There is exactly one such preimage. Indeed,

$$
|T_p(z)|\ge(\alpha-t\nu)|z|-t\lambda|p|
                                  -t\nu\int|z'|d\Lambda,
$$

so this map and the homotopy replacing $\nu$ by $\theta\nu$,
$0\le\theta\le1$, are proper uniformly along the homotopy for the fixed
$p,\Lambda$. The same regularity bound holds at every target preimage
along it. At $\theta=0$ the map is the affine map
$\alpha z-t\lambda p$, of degree one. Proper homotopy keeps that
degree one; the degree at a regular target is the sum of the positive
Jacobian signs of its preimages. Compactness and regularity make that
set finite, so it consists of exactly one point. This is the elementary
degree argument already used for the actual B2 map; all its regularity
and properness conditions have been checked here.

The Gaussian input density is at most $(2\pi q^2)^{-d/2}$.
Change variables at this unique preimage, whose determinant is at least
$(\alpha/2)^d$, to obtain $H_v$. Neither $p$ nor the conditional
Gaussian mean enters that bound. This proves the required uniformity
over all jitters.
:::

:::{prf:corollary} Uniform small-cap derivative budget
:label: cor-native-phase-small-cap

Under {prf:ref}`lem-native-phase-b2-local-density`, for
$0<V\le\min\{V_0,1\}$ let $v_d(r)$ denote the volume of a $d$-ball and put

$$
\chi(V)=\sqrt{\min\left\{1,
 H_vv_d(\sqrt V)+\left(\frac{V}{V+\sqrt V}\right)^2\right\}},\qquad
\beta=\min\{d/4,1/2\},\qquad K_\chi=\sqrt{H_vv_d(1)+1}.
\tag{PC.20}
$$

Conditional on any preparation value,
$\mathbb E_\xi\|DC_V(u)\|^2\le\chi(V)^2$ and
$\chi(V)\le K_\chi V^\beta\to0$. Every interpolation in the preceding
lemma has the same estimate.
:::

:::{prf:proof}
The tangential and radial derivative eigenvalues of the configured cap
$C_V(u)=Vu/(V+|u|)$ are $V/(V+|u|)$ and $V^2/(V+|u|)^2$.
On $|u|>\sqrt V$ its derivative norm is at most
$V/(V+\sqrt V)$. The conditional probability of the complementary
ball is at most $H_vv_d(\sqrt V)$ by the preceding lemma; this ball
lies in $B(0,1)$. Split the expectation at that radius. Finally
$v_d(\sqrt V)=v_d(1)V^{d/2}$ and
$[V/(V+\sqrt V)]^2\le V$. This proves (PC.20). The radius is used only
to estimate a Gaussian pushforward; every original innovation is kept.
:::

(sec-native-phase-two-kick)=
## 4. The complete two-kick marked contraction matrix

:::{prf:definition} Native two-channel matrix
:label: def-native-phase-matrix

Use the coefficients above and define

$$
\begin{gathered}
R_J=R_D+\sigma_J(g_1+d/g_1),\qquad
Z_1=1+c k_cV_0+2ct\lambda R_J+q\sqrt d,\\
A_U=|D_c|+2t\nu c+4t\nu\ell_\rho bZ_1,\qquad
A_X=t\lambda|c+a_x|+2t\nu ct\lambda+4t\nu\ell_\rho|a_x|Z_1,\\
\vartheta=4t\nu k_cV\ell_\rho,\quad
U_x=L_{Vx}+\vartheta L_{Xx},\quad
U_v=L_{Vv}+\vartheta L_{Xv},\quad
\zeta_s=2/(\sqrt{2\pi}s),\\
M(V)=\begin{pmatrix}
 \zeta_s(|a_x|L_{Xx}+bU_x)&\zeta_s(|a_x|L_{Xv}+bU_v)\\
 \chi(V)(A_UU_x+A_XL_{Xx})&\chi(V)(A_UU_v+A_XL_{Xv})
\end{pmatrix}.
\end{gathered}
\tag{PC.21}
$$

The use of $V_0$ in $Z_1$ is an upper estimate; the actual cap in the
transition and all $L_V,U,\chi$ coefficients is $V$. Write $m_{ij}$
for the matrix entries. Retain $0<V\le\min(V_0,1)$, the proved
domain of the cap derivative estimate (PC.20). Its derived strict contraction test is

$$
m_{11}<1,\qquad m_{22}<1,\qquad
m_{12}m_{21}<(1-m_{11})(1-m_{22}).
\tag{PC.22}
$$
:::

:::{prf:theorem} Contraction of the actual full marked population map
:label: thm-native-phase-contraction

For the complete records in {prf:ref}`def-native-phase-register` and
{prf:ref}`def-native-phase-viscosity-interval`, with
$0<V\le\min(V_0,1)$, the actual output laws
admit a coupling with costs

$$
\begin{pmatrix}\delta_x^+\\\delta_v^+\end{pmatrix}
\le M(V)\begin{pmatrix}\delta_x\\\delta_v\end{pmatrix}
\tag{PC.23}
$$

for every entering coupling. If (PC.22) holds, choose

$$
\frac{m_{12}}{1-m_{22}}<\omega<\frac{1-m_{11}}{m_{21}},\qquad
r=\max\{m_{11}+\omega m_{21},m_{22}+m_{12}/\omega\}<1,
\tag{PC.24}
$$

with the evident zero-entry conventions. Then
$W_\omega(\mathcal F\mu,\mathcal F\mu')\le rW_\omega(\mu,\mu')$.
The force, both viscous kicks, OU-force correlations, complete collision
law and terminal marks are the actual configured ones.
:::

:::{prf:proof}
Use the preparation coupling of (PC.9). Interpolate its pre-jitter
variables linearly, retaining the same independent recipient Gaussian:
$X_\theta=Y_\theta+\sigma_JI_\theta\zeta$,
$V^c_\theta=(1-\theta)V^c+\theta V^{c\prime}$.
This is a comparison path, not an alteration of either endpoint kernel.
Its source norm is at most $R_D$, $I_\theta\in[0,1]$, and collision
velocity norm is at most $V_c$. Let $\pi_\theta$ be its law and apply
the actual count first kick to that law. A dot denotes differentiation
in $\theta$. By symmetry of the count kernel and positivity of its
convex averaging weights,

$$
\mathbb E|\dot U_\theta|
\le D_V+4t\nu V_c\ell_\rho D_X=:T_U.
\tag{PC.25}
$$

Indeed the terms differentiating velocities have average at most $D_V$:
the losses $t\nu a(X)|\dot V^c|$ cancel the corresponding symmetric
integrated gains. A differentiated kernel contributes at most
$2t\nu V_c\ell_\rho\mathbb E(|\dot X|+|\dot X'|)$.

One weighted bound will retain the unbounded jitter-force correlations.
For deterministic $d_0$ and scalar $a_0'$,

$$
\mathbb E[|\zeta|\,|d_0+a_0'\zeta|]
 \le(g_1+d/g_1)\mathbb E|d_0+a_0'\zeta|.
\tag{PC.26}
$$

The numerator is at most $g_1|d_0|+d|a_0'|$.
The denominator is at least $|d_0|$ by Jensen and at least
$g_1|a_0'|$: pairing $\zeta,-\zeta$ and using convexity shows that
the norm expectation is minimized by a zero shift. Applying (PC.26)
conditional on the compact pre-jitter pair proves
$\mathbb E|X_\theta||\dot X_\theta|\le R_JD_X$.
The direct velocity derivative is independent of recipient jitter.
The same differentiated first-kick formula, now multiplied by
$|X_\theta|$, consequently gives

$$
\mathbb E|X_\theta||\dot U_\theta|\le2R_JT_U.
\tag{PC.27}
$$

For its velocity terms use $1+t\nu\le2$ and the source first moment;
for its kernel terms use (PC.26) and
$\mathbb E|X_\theta|\le R_J$. These are actual Gaussian moment
estimates, without truncating any jitter.

Give every interpolation value the same OU Gaussian $\xi$. Its A2 and
OU stages have
$y_\theta=a_xX_\theta+bU_\theta+tq\xi$ and
$z_\theta=cU_\theta-ct\lambda X_\theta+q\xi$. Thus

$$
D_y:=|a_x|D_X+bT_U,\qquad
D_z:=cT_U+ct\lambda D_X
\tag{PC.28}
$$

bound the average norms of $\dot y_\theta,\dot z_\theta$.
They do not depend on $\xi$. Each interpolation satisfies the hypotheses
of {prf:ref}`lem-native-phase-b2-local-density`: its complete stage law
$\Lambda_\theta$ obeys (PC.15), with the fixed upper moment $M_e$.
Conditional on its preparation value, Cauchy--Schwarz therefore gives

$$
\mathbb E_\xi\|DC_V(u_\theta)\|\le\chi(V),\qquad
\mathbb E_\xi[\|DC_V(u_\theta)\||\xi|]\le\chi(V)\sqrt d.
\tag{PC.29}
$$

In particular no independence between the second force and the OU
innovation has been presumed.

Differentiate the actual B2 expression (PC.14), including its changing
population law by using an independent copy of the coupled stage pair.
Its linear force part is
$D_c\dot U_\theta-t\lambda(c+a_x)\dot X_\theta$.
The derivative of its velocity difference contributes, after multiplication
by the cap derivative and integration, at most
$2t\nu\chi(V)D_z$. For its kernel derivative use

$$
|\dot K_\rho(y,y')|\le\ell_\rho(|\dot y|+|\dot y'|).
$$

There are four resulting products: a root or a donor $\dot y$, each
multiplied by a root or donor $z$. Each has expectation, with the root
cap derivative, at most $\chi(V)Z_1D_y$.
For a root product use $|z|\le cV_c+ct\lambda|X|+q|\xi|$,
(PC.27), and (PC.29). For the corresponding donor product integrate
the independent donor first, use the same bound with
$\mathbb E|\xi'|\le\sqrt d$, and then (PC.29) for the root.
Mixed root/donor products use their separate first moments, bounded
by the same $Z_1$. This explains the factor four and keeps every
actual force-noise correlation. Combining the pieces yields

$$
\mathbb E\left|\frac d{d\theta}C_V(u_\theta)\right|
 \le\chi(V)[A_UT_U+A_XD_X].
\tag{PC.30}
$$

All differentiations are justified first on bounded latent subsets.
The bounded Gaussian-kernel derivatives, bounded source and collision
variables, and finite Gaussian moments dominate the displayed first
derivatives by integrable polynomials. Dominated convergence removes
those subsets and integrates the path. It gives the second row of
(PC.23), after substituting (PC.9) into (PC.25).

For positions condition on the coupled preparation and OU variables.
The last position innovation has its actual independent Gaussian law
$N(y,s^2I)$. Couple these two final Gaussians by their common minimum
density. Their total variation is at most
$|y-y'|/(\sqrt{2\pi}s)$, from the one-dimensional halfspace separating
equal-covariance Gaussian means. On the common outcome both physical
positions and terminal marks agree. Off it the cost $d_x$ is at most
two. This gives $\delta_x^+\le\zeta_sD_y$, proving the first row.
The conditional maximal coupling preserves each marginal last innovation
and its independence within the respective update; it is not an added
algorithmic transition.

Finally multiply the two rows of (PC.23) by $1,\omega$ and use (PC.24).
Take the infimum over entering couplings to obtain the stated transport
contraction. Condition (PC.22) is exactly the nonempty-interval condition
for (PC.24). Alive marks have remained in its true bounded law cost
throughout the proof.
:::

(sec-native-phase-positive-regime)=
## 5. A nonempty positive regime and stationary chaos

:::{prf:theorem} Explicit active-cloning reset regime
:label: thm-native-phase-reset-regime

Choose existing quadratic and kinetic parameters satisfying

$$
\lambda=\frac1{t^2(1+c)},\qquad a_x=0,\qquad
\alpha=\frac c{1+c}>0,
\tag{PC.31}
$$

and retain any of their positive standardization, fitness, gate and donor
parameters, including $p_r,p_s>0$. Choose any
$0<\nu\le\nu_0$ from (PC.12). There is an explicitly positive interval
of configured caps on which (PC.22) holds. In particular the following
formula defines such an interval without a stationary-law hypothesis.

Set $\theta_0=4t\nu k_c\ell_\rho$ and

$$
u_x=2k_cGR_x+\theta_0L_{Xx},\quad
u_v=k_c+V_0(2k_cGR_v+\theta_0L_{Xv}),
\tag{PC.32}
$$

where $L_{Xx},L_{Xv},G,R_x,R_v$ do not depend on $V$.
With $A_U,A_X$ evaluated as in (PC.21), define

$$
\begin{gathered}
A=\zeta_sbu_x,\quad B=\zeta_sbu_v,\\
C_1=K_\chi(A_UV_0u_x+A_XL_{Xx}),\quad
D_1=K_\chi(A_Uu_v+A_XL_{Xv}),\\
V_{\rm crit}=\min\left\{V_0,1,\frac1{4A},
 (4D_1)^{-1/\beta},(16BC_1)^{-1/\beta}\right\}>0.
\end{gathered}
\tag{PC.33}
$$

A vanishing coefficient contributes an infinite bound in the corresponding
term. For $0<V\le V_{\rm crit}$ there is a weight with contraction
coefficient at most $1/2$. Both endpoint algorithms retain positive
viscosity, active fitness, mandatory revival, unbounded original noises
and actual terminal deaths. The reset is an equality between existing
configured quadratic and timestep parameters; the force is unchanged
inside each theorem instance.
:::

:::{prf:proof}
At (PC.31), (PC.21) and (PC.32) give

$$
m_{11}\le AV,\quad m_{12}\le B,\quad
m_{21}\le C_1V^\beta,\quad m_{22}\le D_1V^\beta.
$$

For (PC.33), $m_{11},m_{22}\le1/4$ and
$m_{12}m_{21}\le1/16$. If both off-diagonal entries are positive choose
$\omega=\sqrt{m_{12}/m_{21}}$. Then both terms in (PC.24) are at most
$1/4+1/4=1/2$. The zero-entry cases follow by choosing a finite weight
making the sole off-diagonal contribution at most $1/4$.
Every constant in (PC.33) is finite: $m_0,\kappa_b,q,s,\alpha$ and
the standardizers are positive, while $e^{2C}$ is finite. Hence the
interval is nonempty even when conservative. No parameter scales with
$N$. The strict inequalities at a smaller positive cap persist in a
neighborhood of the reset equality by continuity of these primitive
constants, giving also an open near-reset parameter regime. For the alive
floor retain its fixed proof radii and, if necessary, decrease its value
by a factor two; continuity of the underlying Gaussian lower bound makes
that same positive floor valid in a sufficiently small neighborhood.
:::

:::{prf:theorem} Unique population stationary phase and full marked QSD chaos
:label: thm-native-phase-stationary-chaos

Assume (PC.22), or its sufficient regime (PC.31)--(PC.33), and retain the
complete finite-population QSD smoothing condition
$\alpha-t\nu>0$. It is already implied by (PC.12).
The complete population map has a unique stationary probability $\mu_*$
within the admitted class, and every admitted population law satisfies

$$
W_\omega(\mathcal F^n\mu,\mu_*)\le r^nW_\omega(\mu,\mu_*).
\tag{PC.34}
$$

For the actual full marked finite-population QSD $\nu_N$, with all
consumed parameters fixed and increasing permitted populations,

$$
(L_N)_\#\nu_N\Longrightarrow\delta_{\mu_*},\qquad
\nu_N^{(k)}\Longrightarrow\mu_*^{\otimes k}\quad(k\text{ fixed}).
\tag{PC.35}
$$

The corresponding $k$ distinct uniformly selected alive rows converge
to $\mathcal R(\mu_*)^{\otimes k}$. For each bounded continuous full
marked row test $f$,
$\operatorname{Var}_{\nu_N}(L_Nf)\to0$ and
$\mathbb E_{\nu_N}L_Nf\to\mu_*f$.
This is stationary empirical concentration and chaos for the actual
phase, without assuming an LSI, stationary curvature or chaos itself.
:::

:::{prf:proof}
The full invariant-law theorem
{prf:ref}`thm-native-stationary-closure-population-invariance` supplies at
least one stationary population law with alive mass at least $a_0$,
terminally consistent marks, capped velocities and all Gaussian output
moments. Every output of an admitted law has alive mass at least $a_0$,
by the actual quadratic binomial row floor, so all subsequent iterations
are admitted. Apply {prf:ref}`thm-native-phase-contraction` to a stationary
law and to each iterate to obtain (PC.34). Comparing two stationary laws
and iterating forces their distance to zero, proving uniqueness.
Every well-defined fixed point with positive alive mass is an output law
and hence has alive mass at least $a_0$. Thus it automatically belongs
to the admitted class. More generally any entering capped law with
positive alive mass enters that class after one full update, so its
subsequent attraction follows from (PC.34) with this one-update delay.

Write $P_N^{\rm raw}$ for the complete raw kernel called $P_N$ in
(SC.13)--(SC.15), including its all-dead output. It is distinct from
the stationary Doob kernel used in the physical-transfer chapters.
The same invariant-law theorem gives tightness of
$\Lambda_N=(L_N)_\#\nu_N$ in $W_4$ and invariance of every subsequential
limit $\Lambda$ under $\mathcal F$. Its support consists of admitted
laws, in fact output laws of alive mass at least $a_0$. For this passage
the actual QSD alive-floor exception and raw-survival discrepancy are
explicitly

$$
\nu_N\{L_Na<m_0\}\le e^{-c_*a_0N},\qquad
\|\nu_NP_N^{\rm raw}-\nu_N\|_{\rm TV}=1-\alpha_N\le(1-a_0)^N,
\quad c_*=(1-\log2)/2.
\tag{PC.36}
$$

They are retained in compact localization and go to zero, rather than
being replaced by a deterministic finite-$N$ alive floor. One-step
consistency is the already proved full rooted preparation and both-kick
consistency on that high-alive class; all its donor and component
hypotheses were quantitatively verified in Section 2 above.

Boundedness of $W_\omega$ and (PC.34), together with invariance, give

$$
\int W_\omega(\mu,\mu_*)\,d\Lambda
 =\int W_\omega(\mathcal F^n\mu,\mu_*)\,d\Lambda
 \le r^n(2+2\omega V).
$$

Letting $n\to\infty$ proves $\Lambda=\delta_{\mu_*}$.
All subsequences have this limit, which proves the first assertion of
(PC.35). The existing exchangeable full-marked subsequence theorem now
identifies the fixed-$k$ marginals with $\mu_*^{\otimes k}$; its alive
sampling counterpart gives the asserted normalized product. Finally
$L_Nf$ converges in probability to the constant $\mu_*f$ and is uniformly
bounded. Its first and second moments therefore converge, proving the
variance statement. No population-uniform joint stationary LSI is used.
:::

(sec-native-phase-reference)=
## 6. Reference evaluation and remaining parameter regimes

:::{prf:corollary} An evaluated existing-parameter witness
:label: cor-native-phase-positive-witness

The following values specify a family of the canonical complete records
of Section 1, using the original uniform companion convention in both
roles:

$$
\begin{gathered}
d=3,\ h=1,\ \gamma=1,\ b_O=1,\ \sigma_x=1,\ \sigma_J=0.1,
\ L_D=2,\ \rho=1,\ \nu=0.01,\ \alpha_{\rm col}=0.5,\\
\lambda=4/(1+e^{-1}),\quad V_0=0.1,\quad V=V_{\rm crit}/2,\\
R_x^{\rm feat}=R_v^{\rm feat}=2,\quad\lambda_{\rm alg}=1,
\quad\epsilon_D=\epsilon_C=\infty,\quad\delta_D=10^{-3},\\
A_r=A_s=\eta_r=\eta_s=p_r=p_s=1,\quad
\sigma_r=\sigma_s=10^6,\quad s_c=100,\quad\epsilon_c=10^{-6}.
\end{gathered}
\tag{PC.37}
$$

Here $\epsilon_D=\epsilon_C=\infty$ denotes the existing
`Kernel::Uniform` tag, rather than an infinite numeric width passed
to `Kernel::Gaussian`. All other fields have the canonical restriction
in Section 1. The cap
and force are the fixed configured values of this record at every
population. Use the minimum of (KU.S8)--(KU.S9) at cap $V_0$ for $a_0$.
The exact formulas define its certificate. Diagnostic decimal evaluations
are

$$
\begin{gathered}
a_x=0,\quad \alpha\simeq0.2689414214,\quad
q\simeq0.6575198540,\quad\tau\simeq1.052655257,\\
a_0\simeq0.7870330015,\quad m_0\simeq0.3935165007,\quad
C\simeq5.082379001,\quad\log G\simeq12.24419954,\\
\overline C\simeq3.843387650,\quad
\nu_0\simeq0.06997509642>\nu,\quad H_v\simeq91.85874000,\\
A\simeq5.063595812\,10^6,\quad B\simeq1.210241400,\quad
C_1\simeq1.254333518\,10^7,\quad D_1\simeq29.97671683,\\
V_{\rm crit}\simeq1.695079100\,10^{-17},\qquad
\log V_{\rm crit}\simeq-38.61621717.
\end{gathered}
\tag{PC.38}
$$

This is a proved positive regime, even though its cap bound is
conservative. The statements use the exact $V_{\rm crit}$ of (PC.33),
not rounded decimals. For orientation, the matrix at its exact
half-cap is diagnostically

$$
M(V)\simeq\begin{pmatrix}
4.29160\,10^{-11}&1.091409887\\
1.91804\,10^{-7}&4.00694\,10^{-9}
\end{pmatrix},\qquad
\omega\simeq2385.42394,\qquad r\simeq4.57537\,10^{-4}.
\tag{PC.39}
$$

At the unchanged reference, using the explicitly evaluated inherited
floor of (KU.S11) in the same coefficient register gives the rigorous
failure test

$$
m_{11}\ge\zeta_s|a_x|L_D^{\rm pos}[1+2/(\kappa_Cm_0)]>1.
\tag{PC.40}
$$

Its diagnostics are $a_x\simeq0.9992156842$,
$\log$ of the displayed lower bound $\simeq52.26021649$
and $\log G\simeq3.598849720\,10^{20}$.
This evaluates the sufficient certificate honestly; it does not show
failure of the reference algorithm's stationary chaos.
:::

:::{prf:proof}
Substitution in (PC.31) gives the reset and its positive $\alpha$.
The force, noise and donor parameters satisfy the stated canonical
restriction. The binomial floor formulas are positive, and (PC.12)
gives the stated weak-viscosity interval, containing the chosen positive
viscosity. The cap is half the exactly defined positive value (PC.33),
so the proved contraction coefficient is at most $1/2$, independently
of any numerical approximation in (PC.38)--(PC.39).
All entries of (PC.21) are nonnegative. Keeping only its displayed
position-copy term proves (PC.40). Substituting the reference values
in that term proves it exceeds one; its primitive formula is also
directly bounded below using $a_x>0.99$, $s=0.02$,
$\kappa_C\le1$, $m_0\le1/2$ and $L_D^{\rm pos}=4\sqrt3$.
These yield a lower bound greater than $10^3$, without relying on
the tiny floor's decimal approximation.
The remaining decimal evaluations report the formulas'
scale and do not carry the analytic proof.
:::

:::{prf:remark} Regime interpretation
:label: rem-native-phase-regime-interpretation

All equalities and inequalities above are tests of the complete original
configuration. A failed test is not a counterexample to chaos. In
particular the unchanged reference of
{prf:ref}`def-cgd-existing-reference` has $a_x\ne0$, $V=2$, and is
outside the explicitly small-cap reset certificate. Its invariant-law
and spatial inequalities remain those of Chapter 26; its deterministic
stationary-chaos conclusion is not supplied by this theorem.

This result discharges a full marked-law stationary phase and
population-uniform attraction for a positive active-cloning count regime.
The row-normalized kernel, the unchanged reference outside this regime,
other configured landscapes, and a population-uniform unconditional
joint stationary functional inequality retain their respective separate
obligations. The complete conditional marked entropy inequality
{prf:ref}`thm-native-stationary-closure-marked-entropy` is available, but
is not relabelled as that unconditional stationary inequality. Graph
viscosity, historical or ordered-star variants and fixed-seed finite
executions receive no transfer based only on a shared variant name.
:::
