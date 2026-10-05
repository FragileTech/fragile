# Quantitative QSD certificates beyond the existing derivative margins

(sec-native-qsd-extension-ledger)=
## Parameter ledger and credited discharge

:::{prf:definition} Complete-record restriction used by this draft
:label: def-native-qsd-extension-record

Every statement is parameterized by the full execution record
$\mathfrak P$ of {prf:ref}`def-native-complete-execution-record`. Its dynamical
restriction is the terminal-box, radial-cap, mandatory-revival canonical
algorithm of {prf:ref}`def-slc-parameter-register`, extended, when indicated,
by the dense Gaussian viscosity of {prf:ref}`def-cgd-parameter-register`.
Thus retain all entries

$$
\begin{aligned}
\theta={}&(d,N,h,\gamma,b_O,\sigma_x,\sigma_J,V,
\alpha_{\rm col},R_x^{\rm feat},R_v^{\rm feat},\lambda_{\rm alg},
\epsilon_D,\epsilon_C,\delta_D,A_r,A_s,\eta_r,\eta_s,p_r,p_s,
\sigma_r,\sigma_s,s_c,\epsilon_c;U,R,D),\\
\Theta={}&(\theta;\nu,\rho,\mathfrak n,\Gamma,\vartheta_\Gamma),
\qquad D=[-L_D,L_D]^d,\quad L_D>0.
\end{aligned}
$$

The complete record additionally retains the actual implementation tag,
all nested configuration payloads, initial law, arithmetic/innovation
convention, providers and passive recording/calibration fields. The results
below use the existing real-coordinate, independent standard Gaussian
innovation convention. They do not assert a density for a fixed-seed,
finite-precision trajectory.

The restriction has independent current-population donor draws, the declared
singleton rule, one donor per role, no self donor except that rule, global
regularized fitness standardization, gates every complete update, simultaneous
cloning, one Haar rotation per accepted connected component, retained dead
coordinates and terminal classification. All force evaluations use the
revived population and the actual new positions at B2. No historical donors,
elite restoration, geometry feedback, graph viscosity, Boris/curl rotation,
innovation shifts, fitness-gradient force, adaptive metric noise or extra
kinetic substeps are enabled. These excluded fields retain their specified
disabled values in $\mathfrak P$; they are not silently omitted.

For Rust these are restrictions of `GasConfig::euclidean`, or its one-field
`viscous_euclidean` extension, with the complete nested configuration preserved.
For a Python tag the statements apply only when its actual stage maps and
innovation law equal this restriction; the default Python configuration is
not identified with it. In particular the Python kinetic integrator,
thermostat, gradient and diffusion flags, neighbor policies, periods and cap
must be checked through {prf:ref}`prop-native-complete-parameter-law`.

Let

$$
t=h/2,\quad c=e^{-\gamma h},\quad
q^2=b_O^2\begin{cases}(1-e^{-2\gamma h})/(2\gamma),&\gamma>0,\\
h,&\gamma=0,\end{cases}\quad s^2=\sigma_x^2h,
\quad V_c=(1+2|\alpha_{\rm col}|)V,
\quad R_D=\sqrt dL_D,\quad m=dN.
$$

Assume $h,V>0$, $\gamma\ge0$, $b_O,\sigma_x>0$, $\sigma_J\ge0$, and
the positive donor widths and fitness/standardization regularizers of the
canonical register. These are explicit parameter regimes. The given initial
law is any probability on nonextinct capped states; no moment bound on raw
dead coordinates is imposed. Passive readout/calibration parameters do not
enter the gas kernel, so the constants are independent of them. Resource-error
termination and numerical underflow branches are not part of this analytic
kernel; an actual execution taking such a branch needs its own comparison.
:::

:::{prf:remark} Existing results credited before extension
:label: rem-native-qsd-existing-discharge

The current Chapter 18 already proves
{prf:ref}`thm-cgd-analytic-force-qsd`,
{prf:ref}`lem-cgd-two-update-density` and
{prf:ref}`thm-cgd-primitive-eigenfunction`. In their evaluated regime they
bound both $\min e/\max e$ and the weighted minorizing mass from primitive
parameters, for both existing dense normalizations. Those results are not
new contributions of this draft. The extensions below cover exact affine
regimes outside their sufficient positive margins, including zero clone
jitter, and the existing Styblinski–Tang force whose global derivative
profile is infinite.
:::

(sec-native-qsd-affine-extension)=
## Exact affine viscosity regimes

:::{prf:lemma} Exact B2 matrix in the included affine regimes
:label: lem-native-qsd-affine-matrix

Restrict {prf:ref}`def-native-qsd-extension-record` to the actual configured
force $F(x)=-Ax+b$, $A=A^{\mathsf T}\in\mathbb R^{d\times d}$, and one of:
(i) $N=1$; (ii) $\nu=0$ with any $N$ and either normalization; or
(iii) $N=2$ with row normalization. Let $\nu_N=0$ in (i)–(ii), and
$\nu_N=\nu$ in (iii). Put $\mathcal L=0$ in (i)–(ii), and
$\mathcal L=\left(\begin{smallmatrix}1&-1\\-1&1\end{smallmatrix}\right)
\otimes I_d$ in (iii), and $\mathcal A=I_N\otimes A$.
For the actual A1 positions $x_1$ and OU velocities $z$, the B2 output is

$$
y=Tz-t\mathcal A x_1+tb_N,
\qquad T=I-t\nu_N\mathcal L-t^2\mathcal A,
\qquad b_N=(b,\ldots,b).
$$

If $\lambda_1,\ldots,\lambda_d$ are the eigenvalues of $A$, then

$$
\begin{array}{ll}
\Delta=|\det T|=\prod_{\ell=1}^d|1-t^2\lambda_\ell|^N,
&\nu_N=0,\\[1mm]
\Delta=\prod_{\ell=1}^d
|1-t^2\lambda_\ell|\,|1-t^2\lambda_\ell-2t\nu|,
&N=2,\ \mathfrak n=\mathrm{row}.
\end{array}
$$

The exact sufficient smoothing regime is $\Delta>0$. Its minimum singular
value $r_T$ is the minimum absolute value of the displayed factors, with
$r_T>0$ in that regime. No positivity of the factors is required.
:::

:::{prf:proof}
Mandatory revival makes both rows eligible at both B kicks. For a row-normalized
pair its sole off-diagonal Gaussian weight cancels with its strictly positive
normalizer in real arithmetic. The two viscous forces are consequently
$\nu(v_2-v_1)$ and $\nu(v_1-v_2)$, independent of positions and $\rho$.
For a singleton the declared force is zero, and $\nu=0$ removes it for any
population. Since A2 gives $x_2=x_1+tz$, substitution into the actual B2
formula gives the displayed $T$. Its consensus and difference subspaces
have the respective matrices $I_d-t^2A$ and
$I_d-t^2A-2t\nu I_d$. Diagonalizing symmetric $A$ proves the determinant
and singular-value formulas. $\square$
:::

:::{prf:theorem} Primitive affine QSD bounds without positive-kick margins
:label: thm-native-qsd-affine-extension

Under {prf:ref}`lem-native-qsd-affine-matrix`, suppose $\Delta>0$ and the
configured reward/provider and donor maps have the canonical continuity
already required in {prf:ref}`thm-cgd-finite-n-qsd`; in particular the actual
quadratic and Styblinski–Tang reward providers meet this requirement. Define
$B_F=|b|$, $L_F=\|A\|_{2\to2}$ and, for an analysis radius $J>0$,

$$
B_0(J)=R_D+J,\quad
B_1(J)=(1+2t\nu_N)V_c+t[B_F+L_FB_0(J)],\quad
X(J)=B_0(J)+tB_1(J).
$$

Choose any $J_0>0$, and put $p_0=G_d(J_0/\sigma_J)$ for positive jitter,
and $p_0=1$ for zero jitter. Set

$$
\begin{aligned}
\sigma_h&=\sqrt{t^2q^2+s^2},\quad
A_0=L_D+J_0+t(1+c)B_1(J_0),\\
a&=p_0\left[\Phi((L_D-A_0)/\sigma_h)
-\Phi((-L_D-A_0)/\sigma_h)\right]^d>0,\\
Z(H)&=\frac{\sqrt N[H+t(B_F+L_FX(J_0))]}{r_T},\\
\ell(R,H)&=p_0^N(2\pi qs)^{-m}\Delta^{-1}
\exp\!\left[-\frac{N[Z(H)+cB_1(J_0)]^2}{2q^2}
-\frac{N[\sqrt dR+X(J_0)+tZ(H)]^2}{2s^2}\right],\\
M&=(2\pi qs)^{-m}\Delta^{-1}.
\end{aligned}
$$

For $L_0=L_D/2$ and $r_v=V/2$, let
$\epsilon=\min\{1/2,\ell(L_0,V)(2L_0)^m v_d(r_v)^N\}$ and let
$\theta_0$ be normalized physical Lebesgue measure on
$[-L_0,L_0]^{dN}\times B(0,r_v)^N$, with every mark alive.
Choose

$$
\begin{aligned}
r&=\sqrt{2d\log(12dN/a)},\quad G=r,\quad
J=\begin{cases}\sigma_Jr,&\sigma_J>0,\\J_0,&\sigma_J=0,\end{cases}\\
Z_0&=cB_1(J)+qG,\quad X_2=X(J)+tZ_0,\\
R&=X_2+sG,\quad
H=(1+2t\nu_N)Z_0+t(B_F+L_FX_2),\\
\underline m&=\min\{1,\ell(R,H)a/(2M)\},\quad
\underline\delta=\epsilon\underline m.
\end{aligned}
$$

The actual full killed marked kernel has a unique QSD $\nu_Q$, an eigenvalue
$\alpha_Q\in[a,1)$, and its positive continuous eigenfunction normalized
by $\max e=1$ satisfies

$$
\min e\ge\underline m,\qquad
\theta_0(e)\ge\underline m,\qquad
\frac{\max e}{\min e}\le\underline m^{-1},\qquad
\delta_Q=\frac{\epsilon\theta_0(e)}{\alpha_Q}
\ge\underline\delta>0.
$$

The conditioned TV and relative-entropy bounds (CGD.16)–(CGD.17) apply
with $\underline m,\underline\delta$. Positive clone jitter, a
positive-kick coercivity margin, and a first-drift margin are unnecessary in
these exact affine regimes. Constants are finite-$N$ constants.
:::

:::{prf:proof}
**1. Bound preparation and retain the actual joint law.** Every post-revival
position comes from an alive input position followed, only when copied, by
its own jitter. Preassigning unused latent jitters changes no output.
On the all-jitter event of radius $J_0$, the collision bound $V_c$ and the
actual B1 formula give $B_0,B_1,X$. Conditional on the entire preparation,
including its shared component rotations, $z=cv_1+q\xi$ and
$u=x_1+tz+s\zeta$ retain their actual independent fresh innovations.
The map $(z,\zeta)\mapsto(y,u)$ is affine, invertible and has absolute
Jacobian $s^m\Delta$. Its joint density is everywhere bounded above by
$M$. On target row velocities $|y_i|\le H$, its unique preimage has
$|z_i|\le Z(H)$ by the Euclidean inverse bound $r_T^{-1}$. The two
Gaussian density lower bounds therefore give $\ell(R,H)$ on the raw
target set $[-R,R]^{dN}\times B(0,H)^N$. Mixtures over the actual
preparation retain the upper bound, and integration of the all-jitter event
retains the lower bound. No independence of cloning outcomes is used.
The cap inverse on $B(0,V/2)$ has preimage radius $V$ and determinant
at least one; this proves $Q\ge\epsilon\theta_0$.

**2. Verify survival and the spectral construction.** For a tagged row,
its own latent jitter event suffices for its B1 bound: the viscous force
uses velocities bounded by $V_c$ and has row norm at most $2\nu_NV_c$.
Its final position is exactly
$x+t(1+c)v_1+tq\xi+s\zeta$. The Gaussian probability of the box is
bounded below by $a$ as in the elementary interval calculation in
{prf:ref}`thm-cgd-analytic-force-qsd`. Hence $Q1\ge a$. A positive
extinction event follows by restricting all jitters and OU draws to finite
balls and taking every final position innovation sufficiently large in its
first coordinate; its uniform probability is positive, so $Q1\le1-\zeta_*$
for some calculated positive Gaussian product $\zeta_*$.

The effective compactification is unchanged: alive positions are in
$\overline D$, dead donor features and capped velocities are bounded, and
there are finitely many nonempty masks. Mandatory revival makes raw dead
coordinates irrelevant to subsequent force evaluation. Conditional affine
Gaussian densities vary continuously in TV; integration over the bounded
rotations and full jitter laws preserves this continuity. The finitely many
canonical pattern probabilities are continuous on each mask component.
The operator on $C(K_N)$ is therefore compact by Arzelà–Ascoli and strongly
positive by the full support just proved. Its spectral radius is at least
$a$ by iteration of $Q1\ge a$. The total positive cone has nonempty
interior. These verify every hypothesis of
[Zhang, Theorem 1.1](https://arxiv.org/pdf/1606.04377), which yields
$\alpha_Q$ and $e>0$; their extrema give
$a\le\alpha_Q\le1-\zeta_*$. The Doob common part and contraction
argument of {prf:ref}`thm-cgd-finite-n-qsd` supplies the unique QSD and
the full conditioned TV/entropy conclusions. The eigenmeasure identity
puts zero mass at the artificial compactification boundary because every
output physical coordinate is finite, so these describe the retained
physical marked law.

**3. Replace its spectral constants.** The elementary Gaussian union bound
$1-G_d(r)\le2d\exp[-r^2/(2d)]$ bounds the probability that any latent
jitter or either kinetic innovation exceeds the chosen radius by $a/2$.
On the complementary event the stage budgets give the displayed raw
target box $\mathcal B$. Extend $E(u,y)=e(u,C_V(y),\mathbf1_D(u))$
by zero at extinct targets. At an effective input with $e=1$,

$$
a\le\alpha_Q=Qe\le M\int_{\mathcal B}E(u,y)\,du\,dy+a/2.
$$

Thus the integral is at least $a/(2M)$. For every input the raw lower
density gives $\alpha_Qe\ge\ell(R,H)a/(2M)$, and
$\alpha_Q\le1$ yields $e\ge\underline m$. The common part then gives
$\delta_Q\ge\epsilon\underline m$. Apply the already proved Doob
TV/entropy estimates with these lower bounds. $\square$
:::

:::{prf:corollary} Evaluated pair regimes and actual singular limitation
:label: cor-native-qsd-affine-regime

For the actual scalar quadratic force $F=-\lambda x$ with a row-normalized
pair, all conclusions of {prf:ref}`thm-native-qsd-affine-extension` hold
whenever

$$
1-\lambda h^2/4\ne0,
\qquad 1-\lambda h^2/4-h\nu\ne0.
$$

For example $\lambda=1$, $h=1$, $\nu=1$ gives factors $3/4$ and $-1/4$,
so the actual unchanged-force pair has the claimed QSD and primitive rates
although the old row coercivity margin is negative. Bandwidth $\rho>0$
does not affect this real-coordinate pair kernel because its weight cancels.
If a displayed factor vanishes, this affine density proof does not apply.
For the actual $N=1$, $\lambda=1$, $h=2$ configuration,
{prf:ref}`rem-chaos-canonical-baoab-resonance` already proves the unique
velocity-zero QSD and failure of TV convergence from every $v_0\ne0$.
Thus excluded hypersurfaces include actual failures of this TV conclusion,
not merely failures of a proof device; no failure claim is made for every
singular pair parameter.
:::

:::{prf:proof}
Substitute the scalar eigenvalue into the two factors of
{prf:ref}`lem-native-qsd-affine-matrix`. The numerical example and the
negative previous margin are direct substitutions. The singleton statement
is precisely the established native resonance calculation, which retains
the actual cap, survival conditioning and both Gaussian innovations.
$\square$
:::

(sec-native-qsd-cubic-extension)=
## The existing Styblinski–Tang force with its unbounded derivative

:::{prf:definition} Separable cubic force and computed profiles
:label: def-native-qsd-cubic-profile

Restrict {prf:ref}`def-native-qsd-extension-record` to $\nu=0$ and the
actual configured polynomial force

$$
F_\ell(x)=-g_\ell x_\ell^3-\lambda_\ell x_\ell+b_\ell,
\quad g_\ell>0,\ \lambda_\ell,b_\ell\in\mathbb R,
\quad \ell=1,\ldots,d.
$$

Its potential is
$U(x)=\sum_\ell[g_\ell x_\ell^4/4+\lambda_\ell x_\ell^2/2-b_\ell x_\ell]$
up to its actual additive constant; its reward remains separately configured.
For the included Styblinski–Tang benchmark these are exactly
$g_\ell=2$, $\lambda_\ell=-16$, $b_\ell=-5/2$ and
$U=\frac12\sum_\ell(x_\ell^4-16x_\ell^2+5x_\ell)$.
No configured run is changed by this definition. A run belongs to this regime
only when its actual gradient provider supplies this force at both B stages.

Define its finite regional bound

$$
M_F(r)=\left[\sum_\ell(g_\ell r^3+|\lambda_\ell|r+|b_\ell|)^2\right]^{1/2},
\quad B_0(J)=R_D+J,
\quad B_1(J)=V_c+tM_F(B_0(J)),
\quad X(J)=B_0(J)+tB_1(J).
$$

Its global derivative profile is infinite. The following proof does not
replace that profile by a finite assumed constant.
:::

:::{prf:lemma} Cubic preimages, raw density lower bound and critical mass
:label: lem-native-qsd-cubic-density

Under {prf:ref}`def-native-qsd-cubic-profile`, choose $J_0>0$ and define
$p_0$ as in {prf:ref}`thm-native-qsd-affine-extension`. Write
$X=X(J_0)$, $\alpha_\ell=1-\lambda_\ell t^2$ and, for $H>0$,

$$
\begin{aligned}
C_\ell(H)&=tH+X+t^2|b_\ell|,\\
W_\ell(H)&=\max\left\{
\sqrt{\frac{2|\alpha_\ell|}{g_\ell t^2}},
\left[\frac{2C_\ell(H)}{g_\ell t^2}\right]^{1/3}\right\},\\
Z(H)&=\left[\sum_\ell\left(\frac{W_\ell(H)+X}{t}\right)^2\right]^{1/2},
\qquad W(H)=\left[\sum_\ell W_\ell(H)^2\right]^{1/2},\\
D_\ell(H)&=|\alpha_\ell|+3g_\ell t^2W_\ell(H)^2,\\
\ell_3(R,H)&=\frac{p_0^N(2\pi qs)^{-m}}{\prod_\ell D_\ell(H)^N}
\exp\!\left[-\frac{N[Z(H)+cB_1(J_0)]^2}{2q^2}
-\frac{N[\sqrt dR+W(H)]^2}{2s^2}\right],\\
C_{\rm crit}&=N\sum_\ell\frac{2}{q t^2\sqrt{3\pi g_\ell}}.
\end{aligned}
$$

The actual full raw-output law has density at least $\ell_3(R,H)$ almost
everywhere on $[-R,R]^{dN}\times B(0,H)^N$. For every preparation and
$\varepsilon>0$, the source event that any scalar B2 derivative has
absolute value below $\varepsilon$ has probability at most
$C_{\rm crit}\sqrt\varepsilon$. Its complement pushes forward to a
submeasure with raw joint density at most

$$
M_\varepsilon=3^m(2\pi qs)^{-m}\varepsilon^{-m}.
$$

These statements retain all critical points of the original cubic map; only
their explicit probability is charged in an inequality.
:::

:::{prf:proof}
**1. Solve the actual B2 polynomial.** Given preparation, a scalar B2 map is
$T(z)=z+t[-g(x_1+tz)^3-\lambda(x_1+tz)+b]$. Put $w=x_1+tz$.
The equation $T(z)=y$ becomes

$$
ty+x_1-t^2b=\alpha w-gt^2w^3.
$$

The right side is a cubic with nonzero leading coefficient, so it reaches
every real target. If $|x_1|\le X$ and $|y|\le H$, a root with
$|w|>W_\ell(H)$ would have
$gt^2|w|^3>2|\alpha||w|$ and
$gt^2|w|^3>2C_\ell(H)$. Their combination gives
$|\alpha w-gt^2w^3|>C_\ell(H)$, a contradiction. Thus every root has
the displayed bound. Differentiation gives
$T'(z)=\alpha-3gt^2w^2$, bounded in absolute value by $D_\ell(H)$
on those roots. Its zero set is finite. Each regular target has at least
one regular root and at most three roots. Gaussian change of variables on
the finitely many monotonicity intervals, followed by independent final
position convolution, now proves the raw density lower bound exactly using
the displayed preimage, OU and position budgets. The all-jitter event gives
$p_0^N$; all actual cloning/rotation mixtures preserve the lower bound by
Fubini. Critical target values form a finite scalar set and a null joint
set, so they do not affect the almost-everywhere assertion.

**2. Bound the critical source event uniformly.** In the original $z$
coordinate the derivative is
$\alpha-3gt^4(z+x_1/t)^2$. For $B>0$ and any real $\alpha$ the set
$\{|\alpha-Bu^2|<\varepsilon\}$ has length at most
$2\sqrt{2\varepsilon/B}$. Indeed, for $\alpha+\varepsilon>0$
its length is
$2[\sqrt{(\alpha+\varepsilon)/B}
-\sqrt{\max(0,\alpha-\varepsilon)/B}]$; this is maximized at
$\alpha=\varepsilon$. For $\alpha+\varepsilon\le0$ the set is empty.
The scalar OU density is at most $(\sqrt{2\pi}q)^{-1}$, uniformly in
its entering mean. Taking $B=3gt^4$, multiplying length by this density,
and summing over all $N d$ scalar coordinates proves
$C_{\rm crit}\sqrt\varepsilon$. It remains valid for every unbounded
preparation coordinate and every shared collision realization.

**3. Upper-bound the complementary submeasure.** Every regular joint target
has at most $3^m$ preimages because the scalar B2 maps separate. At each
retained preimage the velocity Jacobian is at least $\varepsilon^m$;
the final independent position stage contributes $s^m$. The maximum OU
and standard-position Gaussian density, times the inverse Jacobian and
preimage count, gives $M_\varepsilon$. This bound is uniform in all
preparation values, so integrating the actual preparation law incurs no
donor-pattern factor. $\square$
:::

:::{prf:theorem} Primitive QSD and mixing certificate for the native cubic force
:label: thm-native-qsd-cubic-extension

Under {prf:ref}`def-native-qsd-cubic-profile`, retain the canonical continuous
reward/donor regime stated in {prf:ref}`thm-native-qsd-affine-extension`.
Define

$$
\begin{aligned}
\sigma_h&=\sqrt{t^2q^2+s^2},\quad
A_0=L_D+J_0+t(1+c)B_1(J_0),\\
a&=p_0[\Phi((L_D-A_0)/\sigma_h)
-\Phi((-L_D-A_0)/\sigma_h)]^d,\\
r&=\sqrt{2d\log(24dN/a)},\quad G=r,\quad
J=\begin{cases}\sigma_Jr,&\sigma_J>0,\\J_0,&\sigma_J=0,\end{cases}\\
Z_0&=cB_1(J)+qG,\quad X_2=X(J)+tZ_0,\quad
R=X_2+sG,\quad H=Z_0+tM_F(X_2),\\
\varepsilon_*&=(a/(4C_{\rm crit}))^2,\quad
M_*=M_{\varepsilon_*},\\
\epsilon_3&=\min\{1/2,\ell_3(L_D/2,V)L_D^m v_d(V/2)^N\},\\
\underline m_3&=\min\{1,\ell_3(R,H)a/(2M_*)\},\quad
\underline\delta_3=\epsilon_3\underline m_3.
\end{aligned}
$$

All these constants are strictly positive and finite for every $h>0$ in the
declared regime. The actual killed marked kernel has a unique QSD and its
normalized principal eigenfunction satisfies

$$
\alpha_Q\ge a,\qquad
\min e\ge\underline m_3,\qquad
\theta_0(e)\ge\underline m_3,\qquad
\delta_Q\ge\underline\delta_3.
$$

The conditioned TV and relative-entropy inequalities (CGD.16)–(CGD.17)
hold with these primitive constants. This includes the existing
Styblinski–Tang benchmark, at every $h>0$, every finite $N,d$, all
$\gamma\ge0$, all positive OU and final-position amplitudes, any
$\sigma_J\ge0$, and every other declared canonical parameter value.
Neither global convexity nor a finite global force derivative is required.
:::

:::{prf:proof}
**1. Establish the complete finite-QSD construction.** The polynomial force
is finite and analytic at every reachable real coordinate, including the
unbounded intermediate A2 positions. Its scalar B2 maps are proper and
surjective cubics, with finite critical sets. Thus their product and the
independent final-position innovation give absolute continuity and full
support. For converging preparations their maps converge in $C^1$ on every
compact set, and their critical source set has zero Gaussian mass.
{prf:ref}`lem-cgd-pushforward-tv` proves TV continuity; full Gaussian
jitter laws are integrated without clipping. The compact effective input,
finite continuous donor/gate probabilities, strong positivity and
Arzelà–Ascoli compactness checks are the same verified checks in Step 2 of
{prf:ref}`thm-native-qsd-affine-extension`. Here the spectral-radius floor
is the tagged-row Gaussian survival floor $a>0$, obtained from its exact
final-position identity and regional B1 bound. A uniform positive extinction
event is obtained by bounded latent jitters and OU draws followed by large
final-position draws. Therefore Zhang's Theorem 1.1 applies with exactly
those verified hypotheses. The Doob argument gives uniqueness and the
conditioned estimates, and the eigenmeasure identity recovers every physical
dead coordinate. The minorization weight $\epsilon_3$ follows from
{prf:ref}`lem-native-qsd-cubic-density` and the existing cap inverse,
on the same physical all-alive target as before.

**2. Calculate the exceptional probability.** The Gaussian-coordinate union
bound gives probability at most $a/12$ for each of the three all-row failures
(latent jitter, OU noise, final-position noise) at the chosen $r$. Their
union has probability at most $a/4$; with zero jitter there are only two
terms. The critical-source bound is at most
$C_{\rm crit}\sqrt{\varepsilon_*}=a/4$. The total discarded mass is
therefore at most $a/2$. Every original draw remains in the kernel.
On the complementary event the two actual force evaluations and the stage
budgets give the raw target box $\mathcal B=[-R,R]^{dN}\times B(0,H)^N$.
Its restricted output density is at most $M_*$ by the cubic lemma.

**3. Obtain the primitive eigenfunction floor.** Extend
$E(u,y)=e(u,C_V(y),\mathbf1_D(u))$ by zero on extinct targets. At an
effective input attaining $\max e=1$,
$a\le Qe\le M_*\int_{\mathcal B}E+a/2$. Hence that same raw integral
is at least $a/(2M_*)$. The original full-kernel raw density lower bound
on $\mathcal B$ gives $\alpha_Qe(k)\ge\ell_3(R,H)a/(2M_*)$ for every
input. Since $\alpha_Q\le1$, the displayed $\underline m_3$ follows.
The common part gives
$\delta_Q=\epsilon_3\theta_0(e)/\alpha_Q\ge\underline\delta_3$.
Substitution into the established full-law conditioned estimates proves
the assertion. All computed quantities are finite because $t,q,s,g_\ell$
are strictly positive and every analysis radius is finite. $\square$
:::

(sec-native-qsd-extension-register)=
## Exact regime and remaining-obligation register

:::{prf:remark} What is and is not discharged
:label: rem-native-qsd-extension-regimes

| Complete-record restriction | Derived conclusion | Status outside it |
|---|---|---|
| Existing analytic finite-$L_F$ force, either dense normalization, evaluated (CGD.8), (CGD.12), (CGD.18) | Existing finite-QSD and primitive eigenfunction/mixing certificate | Already proved in Chapter 18; not a new draft result |
| Affine force, $\nu=0$ at any $N$, or singleton, all factors $1-t^2\lambda_\ell\ne0$ | New one-update primitive bound, including zero clone jitter | Singular factors are outside this route; existing singleton resonance proves an actual TV failure at one such regime |
| Affine force, row-normalized $N=2$, both factor families nonzero | New QSD/primitive bounds with either sign of each factor | No general conclusion asserted at singular factors; count-normalized pairs have position-dependent weights and cannot use this matrix |
| Actual Styblinski–Tang force, canonical $\nu=0$, every $h>0$, positive $q,s$, any $\sigma_J\ge0$ | New full finite-QSD, eigenfunction and weighted-minorizing-mass bounds despite $L_F=\infty$ | Positive viscosity is outside the separable-cubic quantitative proof; it is not replaced by zero in a viscous run |
| Other actual separable cubic force with $g_\ell>0$ and the same stage contract | Same computed profiles and proof | Cross-coordinate quartics, Rosenbrock, radial Mexican Hat, nonsmooth objectives and numerical-gradient providers require their own map/preimage estimates |
| History donors, elites, metric or geometry feedback, Boris/curl, other normalization or noise schedules | No transfer from these restrictions | Their consumed stage maps remain explicit in $\mathfrak P$; they require a separate kernel proof |
| Finite-precision, fixed seed, or a floating-point zero-normalizer branch | Actual numerical transition remains its recorded transition | No Lebesgue-density conclusion is inferred from the real-coordinate pair weight cancellation |

The new bounds concern the full killed marked gas and its Doob transform.
They do not prove a population-uniform mixing rate, joint LSI, stationary
chaos, optimization success, unbounded-domain invariant/QSD laws, graph
consistency or a physical Yang–Mills identification. Large timesteps can
make every lower constant extremely small even when it remains positive.
The polynomial-force proof establishes a QSD of the actual explicit scheme,
not numerical stability of an uncapped Hamiltonian integrator.

Source checks used the actual constructors
`algorithmic-gas/crates/algorithmic-gas/src/variants/{euclidean,viscous_euclidean}.rs`,
the nested configuration and force/stage code in `engine.rs` and `kinetic.rs`,
the `RunConfig`/`BenchmarkModel` reward and gradient providers in
`algorithmic-gas/crates/benchmarks/src/lib.rs`, and the native
`Benchmark::StyblinskiTang` gradient in that file. Its identical potential is
also present in `src/fragile/fractalai/core/benchmarks.py`; that agreement
does not identify the different Python kinetic kernel with Rust semantics.
The canonical parameter/compactification proofs in Chapters 06a, 09 and 18,
the current variants chapter, and the existing
`src/fragile/fractalai/theory/qsd_certificate.py` were inspected. The existing
evaluator was not edited and does not implement these new polynomial bounds.
The primary [Krein–Rutman source](https://arxiv.org/pdf/1606.04377),
Theorem 1.1, was checked against the compactness, cone, positivity and
spectral-radius conditions proved above.
:::
