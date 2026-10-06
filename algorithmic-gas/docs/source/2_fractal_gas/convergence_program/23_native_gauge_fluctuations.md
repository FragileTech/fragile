# Native determinant fluctuations at the actual second kick

(sec-ngf-complete-record)=
## 1. Execution data, stages, and observable

:::{prf:definition} Complete parameter ledger for the native determinant calculation
:label: def-ngf-parameter-ledger

Retain the complete execution record $\mathfrak P$ of
{prf:ref}`def-native-complete-execution-record`, including its initial law,
arithmetic and innovation convention, every recursive configuration field,
landscape/provider, boundary, stage schedule, mask, history convention,
geometry/graph readout, and physical calibration. The positive results below
apply to its restriction to the unchanged real-coordinate Viscous Euclidean
Gas of {prf:ref}`def-cgd-parameter-register` and
{prf:ref}`def-variant-recorded-color-geometry`. In particular retain

$$
\begin{aligned}
\theta={}&(d=3,N,h,\gamma,b_O,\sigma_x,\sigma_J,V_{\max},\alpha_{\rm col},
 R_x^{\rm feat},R_v^{\rm feat},\lambda_{\rm alg},\epsilon_D,\epsilon_C,
 \delta_D,A_r,A_s,\eta_r,\eta_s,p_r,p_s,\sigma_r,\sigma_s,s_c,\epsilon_c;
 U,R,D),\\
\Theta={}&(\theta;\nu,\rho,\mathsf n;\vartheta_G,\vartheta_O),\qquad
\mathsf n\in\{\mathrm{count},\mathrm{row}\}.
\end{aligned}
$$

The fields of $\vartheta_G,\vartheta_O$ are the complete fields listed in
{prf:ref}`def-variant-recorded-color-geometry`; none is suppressed here.
The narrower coupled-gas register excludes donor history, geometry feedback,
curl, noncanonical collision/update schedules, and anisotropic thermostats;
their actual excluded values are retained in $\mathfrak P$. A Python graph
viscosity execution is not identified with the Rust complete Gaussian graph
by its variant name. Its actual graph and kinetic flags remain in
$\mathfrak P$, and the complete-graph results below do not transfer to it
without equality of the consumed maps. Fixed-seed finite-arithmetic records
retain their actual law; the Gaussian calculations below concern the existing
independent real Gaussian convention.

Condition on the full actual post-collision state. Run its first force
evaluation B1 and first drift A1 without alteration and denote their outputs
by $(x_1,v_1)$. Put

$$
t=h/2,\qquad c=e^{-\gamma h},\qquad
q^2=b_O^2
\begin{cases}(1-e^{-2\gamma h})/(2\gamma),&\gamma>0,\\ h,&\gamma=0.
\end{cases}
$$

The actual O and A2 stages give

$$
z_i=cv_{1,i}+q\xi_i,\quad x_{2,i}=x_{1,i}+tz_i,
\quad \xi_i\stackrel{\rm independent}{\sim}N(0,I_3).
$$

At B2 retain its newly evaluated force, not B1's force:

$$
F_i(z)=\nu\frac{\sum_{j\ne i}K_{ij}(z)(z_j-z_i)}{n_i(z)},\quad
K_{ij}(z)=\exp\!\left[-\frac{|x_{1,j}-x_{1,i}+t(z_j-z_i)|^2}{2\rho^2}\right],
\quad
n_i=\begin{cases}N,&\mathsf n=\mathrm{count},\\
\sum_{j\ne i}K_{ij},&\mathsf n=\mathrm{row}.
\end{cases}
$$

This uses the actual all-eligible kick following revival. Set
$\kappa=m\ell_0/\hbar_{\rm eff}$ and retain the actual color threshold
$\delta_c\ge0$. The matched B2 zero-extended colors are

$$
C_i(z)=\mathbf1_{\{|F_i(z)|>\delta_c\}}
 \frac{F_i(z)}{|F_i(z)|}\odot e^{i\kappa z_i},
\qquad b_{ijk}(z)=\det[C_i(z),C_j(z),C_k(z)].
$$

The value is zero on a failed color mask. Additional terminal-alive,
clone-identity, coverage, graph/face, companion, localization, and weighting
masks remain separate parts of the declared readout. They are not silently
deleted from a configured channel average. The exact B2 result applies to
$\mathsf A_{\rm MK}(\mathrm{B2})$, or to $\mathsf A_{\rm PK}$ carrying the
same preceding B2 force-input velocity. The executable Rust `colors` reader
in `physics/qft.rs` uses matched **B1**. The Python batch readers of
`force_viscous[t-1]` use B1 with their declared RO/MK alignment. No B2 theorem
below is asserted for these B1 observables.
:::

(sec-ngf-centroid)=
## 2. Exact centroid coefficient in the full complex determinant

:::{prf:theorem} Native Gaussian centroid phase and exact determinant covariance
:label: thm-ngf-centroid-determinant-covariance

Fix any finite $(x_1,v_1)$ of the preceding definition, $N\ge3$, $q>0$,
$\nu>0$, $\rho>0$, and either of the existing normalizations. Define

$$
G=N^{-1}\sum_i z_i,\quad r_i=z_i-G,\quad
g_0=cN^{-1}\sum_i v_{1,i},\quad
\sigma_N^2=3\kappa^2q^2/N.
$$

Conditional on $(x_1,v_1)$, $G\sim N(g_0,q^2I_3/N)$ is independent of
the complete relative array $r$. Every B2 force and its threshold mask
depends only on $r$. With

$$
D(G)=\operatorname{diag}(e^{i\kappa G^1},e^{i\kappa G^2},e^{i\kappa G^3}),
\quad B_{ijk}(r)=\det[C_i(r),C_j(r),C_k(r)],
\quad A=B_{ijk}(r)e^{i\kappa\mathbf1\cdot g_0},
$$

the actual invariant coordinates satisfy exactly

$$
C_i(r+G)=D(G)C_i(r),\quad
q_{ij}(r+G)=q_{ij}(r),\quad
\Pi_{ijk}(r+G)=\Pi_{ijk}(r),\quad
b_{ijk}(r+G)=B_{ijk}(r)e^{i\kappa\mathbf1\cdot G}.
\tag{NGF.1}
$$

Consequently, conditional on $(x_1,v_1,r)$,

$$
\begin{aligned}
\mathbb E b_{ijk}&=Ae^{-\sigma_N^2/2},\\
\mathbb E|b_{ijk}-\mathbb E b_{ijk}|^2
 &=|B_{ijk}(r)|^2(1-e^{-\sigma_N^2}),\\
\mathbb E(b_{ijk}-\mathbb E b_{ijk})^2
 &=A^2(e^{-2\sigma_N^2}-e^{-\sigma_N^2}).
\end{aligned}
\tag{NGF.2}
$$

The real two-coordinate covariance of $b_{ijk}$ has eigenvectors, when $A\ne0$, in the
radial and tangent directions of $A$ and corresponding eigenvalues

$$
\lambda_{\rm radial}=\tfrac12|A|^2(1-e^{-\sigma_N^2})^2,
\qquad
\lambda_{\rm tangent}=\tfrac12|A|^2(1-e^{-2\sigma_N^2}).
\tag{NGF.3}
$$

For $\kappa q\ne0$ and $B_{ijk}(r)\ne0$ both eigenvalues are strictly
positive. This is a calculation for the full complex determinant, an
existing common-$SU(3)$ invariant. The same centroid contributes zero to
the Gram and triangle coordinates; it is not a curvature-variance proof.
:::

:::{prf:proof}
The orthogonal decomposition of the Gaussian vector
$(\xi_1,\ldots,\xi_N)$ into its constant-row subspace and its orthogonal
complement has block diagonal covariance. Its density in orthogonal
coordinates factors into the two Gaussian densities, proving independence
and the displayed centroid law. Deterministic nonconstant means change only
the relative mean and leave this factorization unchanged.

Adding $G$ to every velocity translates every B2 position by $tG$.
All pair differences, kernel values, normalizers, and force differences are
unchanged. Thus $F_i(r+G)=F_i(r)$, including its configured threshold mask.
The remaining color phase is precisely $D(G)$. A diagonal unitary matrix
preserves every Gram entry and hence every triangle product; its determinant
is $e^{i\kappa\mathbf1\cdot G}$. This proves (NGF.1). No thermostat
innovations or later force evaluations have been capped in this argument.

The scalar $\eta=\kappa\mathbf1\cdot(G-g_0)$ is normal with variance
$\sigma_N^2$. Completing the square in its Gaussian integral gives
$\mathbb E e^{iu\eta}=e^{-u^2\sigma_N^2/2}$. Substitute $u=1,2$ in
$b=Ae^{i\eta}$ and expand the centered products to obtain (NGF.2).
After rotating $A$ onto the positive real axis, the two centered coordinates
are $|A|(\cos\eta-e^{-\sigma_N^2/2})$ and $|A|\sin\eta$.
Oddness makes their covariance zero. The identities
$\cos^2\eta=(1+\cos2\eta)/2$ and
$\sin^2\eta=(1-\cos2\eta)/2$ give (NGF.3). Rotation back proves the
asserted eigenvectors and strictness. When the determinant or the phase
coefficient vanishes, the formulas retain the resulting zero variances.
:::

:::{prf:corollary} Exact population scale and an identified phase innovation
:label: cor-ngf-scaled-centroid-innovation

For any incoming post-collision law, including the incoming-record law from
the proved reference QSD, consider the raw matched B2 record before terminal
selection. With its actual threshold masks,

$$
\operatorname{Var}_{\mathbb C}(\sqrt N\,b_{ijk})
 \ge N(1-e^{-3\kappa^2q^2/N})\,\mathbb E|B_{ijk}(r)|^2,
\tag{NGF.4}
$$

where $\operatorname{Var}_{\mathbb C}X=\mathbb E|X-\mathbb EX|^2$.
The expectation includes the actual selection, collision, B1/A1 input law.
For fixed configured $q,\kappa$, the prefactor tends to
$3\kappa^2q^2$ and is bounded below by
$3\kappa^2q^2e^{-3\kappa^2q^2/N}$.

On an auxiliary product representation of the *same* Gaussian centroid law,
write $G=g_0+(q/\sqrt N)Z$, $Z\sim N(0,I_3)$, independent of $r$ and the
incoming state. Put $a=\kappa q\mathbf1\cdot Z$. The centered phase part has
the exact expression

$$
\sqrt N[b_{ijk}-\mathbb E(b_{ijk}\mid x_1,v_1,r)]
=A\sqrt N(e^{ia/\sqrt N}-e^{-3\kappa^2q^2/(2N)}).
\tag{NGF.5}
$$

Its distance in $L^2$ from $iAa$ is at most

$$
\frac{\sqrt{27}\,\kappa^2q^2+3\kappa^2q^2}{2\sqrt N}.
\tag{NGF.6}
$$

Thus the identified conditional tangent covariance tends, with this explicit
error, to
$3\kappa^2q^2\,(-\operatorname{Im}A,\operatorname{Re}A)
(-\operatorname{Im}A,\operatorname{Re}A)^{\mathsf T}$.
Dividing its per-step coefficient by the actual recorded step length gives
$3\kappa^2q^2/h$; no diffusion limit in algorithmic time is claimed.
For the unchanged reference $h=0.04$, $\gamma=b_O=1$,
$q^2=(1-e^{-0.08})/2$, so this coefficient is exactly
$3\kappa^2(1-e^{-0.08})/0.08$.
:::

:::{prf:proof}
Total variance conditioned first on the incoming state and then on the full
relative array bounds total variance below the expectation of (NGF.2).
Multiplication by $N$ proves (NGF.4). For $u\ge0$,
$1-e^{-u}=\int_0^u e^{-s}ds\ge ue^{-u}$, giving its lower estimate.

The Gaussian factorization in the theorem realizes every centroid using the
displayed independent $Z$. It changes no kernel or law. The elementary
Taylor remainder $|e^{ix}-1-ix|\le x^2/2$ and
$0\le1-e^{-u}\le u$ give

$$
\left|\sqrt N(e^{ia/\sqrt N}-e^{-3\kappa^2q^2/(2N)})-ia\right|
\le\frac{a^2+3\kappa^2q^2}{2\sqrt N}.
$$

The normal variable $a$ has variance $3\kappa^2q^2$ and fourth moment
$27\kappa^4q^4$. Taking $L^2$ norms and using $|A|\le1$ proves
(NGF.6). This is conditional centering of one explicitly identified
innovation. It does not replace the separate fluctuation of
$\mathbb E(b_{ijk}\mid x_1,v_1,r)$ by its full-law mean.
:::

(sec-ngf-amplitude)=
## 3. Native determinant amplitude without a variance assumption

:::{prf:theorem} Almost-sure nonzero complex B2 determinant
:label: thm-ngf-analytic-determinant-amplitude

Fix any finite actual A1 arrays, $q>0$, $\nu>0$, and $0<\rho<\infty$.
Use the exact zero-force mask $\delta_c=0$ and either existing complete
Gaussian normalization. For every fixed triple of distinct rows:

1. If $N\ge4$, its native complex determinant is nonzero almost surely,
   for every real $\kappa$, including $\kappa=0$.
2. If $N=3$, it is nonzero almost surely when $\kappa\ne0$.
   When $N=3$ and $\kappa=0$, it is identically zero.

The nonzero conclusions remain true under every positive-probability
survival conditioning. They concern the matched B2 determinant before
additional terminal/clone/localization masks. For $N<3$ no distinct triple
exists. At $\nu=0$ the zero-force extension is zero. Positive numerical
thresholds retain the explicit regime calculation of the next lemma.
:::

:::{prf:proof}
The unnormalized determinant

$$
\mathcal D(z)=\det[F_i(z)\odot e^{i\kappa z_i},
 F_j(z)\odot e^{i\kappa z_j},F_k(z)\odot e^{i\kappa z_k}]
$$

has real analytic real and imaginary parts. All row denominators are
strictly positive. It suffices to show that one part is not identically zero;
then the elementary analytic zero-set proof in
{prf:ref}`thm-variant-b2-color-nondegeneracy` makes its zero set null under
the full-support OU Gaussian law. Dividing by nonzero force norms does not
change its zero set.

For $N\ge4$, relabel the triple $1,2,3$ and choose a distinct row 4.
Let $K^0_{ab}=\exp[-|x_{1,a}-x_{1,b}|^2/(2\rho^2)]$ and
$L^0=\operatorname{diag}(d_a)-K^0$ with zero diagonal kernel and
$d_a=\sum_{b\ne a}K^0_{ab}$. On the zero-sum subspace $L^0$ is strictly
positive: its quadratic form is
$\frac12\sum_{a\ne b}K^0_{ab}|w_a-w_b|^2$, whose only null vectors are
constant. Thus its inverse on that subspace is an explicitly defined finite
matrix.

Prescribe $g_1=e_1,g_2=e_2,g_3=e_3$ and $g_a=0$ for $a>4$.
For count normalization put $g_4=-(e_1+e_2+e_3)$ and solve
$L^0w=-Ng/\nu$ in each coordinate with zero row sum.
For row normalization put
$g_4=-(d_1e_1+d_2e_2+d_3e_3)/d_4$ and solve
$L^0w=-\operatorname{diag}(d_a)g/\nu$ with the same convention.
Both right sides have zero row sum, so these solutions exist. Expanding the
actual B2 force at $z=\epsilon w$ gives
$F_a(\epsilon w)=\epsilon g_a+O(\epsilon^2)$.
The phase tends to one, hence
$\mathcal D(\epsilon w)=\epsilon^3+O(\epsilon^4)$.
Its real part is nontrivial.

For $N=3$, put $z_a=u e_a$. Write
$M(u)=\operatorname{diag}(n_a(u))^{-1}L(u)$ for the three-row B2 Laplacian.
The row-column matrix of unnormalized colors is

$$
-\nu u\{M(u)+(e^{i\kappa u}-1)\operatorname{diag}(M_{aa}(u))\}.
$$

$\det M(u)=0$ for every $u$. Its diagonal entries are positive. Each
principal two-by-two minor of $L(0)$ is
$K_{12}^0K_{13}^0+K_{12}^0K_{23}^0+K_{13}^0K_{23}^0>0$;
division by positive row normalizers makes each principal cofactor of
$M(0)$ positive as well. The determinant expansion in its diagonal
perturbation therefore has first nonzero term

$$
i\kappa u\sum_a M_{aa}(0)\operatorname{cof}_{aa}M(0)+O(u^2).
$$

It is nonzero for $\kappa\ne0$, so $\operatorname{Im}\mathcal D$ is
nontrivial. For $\kappa=0$ the three force columns are linearly dependent:
their unnormalized count sum is zero, or their row-normalized
degree-weighted sum is zero. This proves the stated failure regime.
Finally a null event remains null after positive-probability conditioning.
:::

:::{prf:lemma} Explicit positive-threshold determinant certificate
:label: lem-ngf-threshold-amplitude-certificate

For $N\ge4$ use the actual finite matrix $w$ constructed in the preceding
proof. Set

$$
\begin{gathered}
W=\max_a|w_a|>0,\quad k_0=\min_{a\ne b}K^0_{ab}>0,
\quad L_\rho=e^{-1/2}/\rho,\\
C_F=\begin{cases}4\nu tL_\rho W^2,&\mathsf n=\mathrm{count},\\
16\nu tL_\rho W^2/k_0,&\mathsf n=\mathrm{row},\end{cases}
\quad Q=2C_F+|\kappa|W,\\
\epsilon_* =\min\left\{1,\frac{k_0}{4tL_\rho W},
 \frac1{4C_F},\frac1{12Q}\right\}>0.
\end{gathered}
$$

An infinite bound is omitted when its denominator is zero. In the explicit
configured regime $\delta_c<\epsilon_*/2$, put $z^*=\epsilon_*w$,
$H=\epsilon_*W+1$, $D_1=\max_{a,b}|x_{1,a}-x_{1,b}|$,

$$
\begin{gathered}
k_R=e^{-(D_1+2tH)^2/(2\rho^2)},\quad
L=\nu(2+8tL_\rho H/k_R),\quad f_*=3\epsilon_*/4,\quad
Q_R=2L/f_*+|\kappa|,\\
R_* =\min\{1,(f_*-\delta_c)/(2L),1/(12Q_R)\}>0,\\
p_* =\left[\frac{4\pi R_*^3/3}{(2\pi q^2)^{3/2}}\right]^N
 \exp\!\left[-\frac1{2q^2}\sum_a(|z_a^*-cv_{1,a}|+R_*)^2\right]>0.
\end{gathered}
$$

Then the actual thresholded B2 determinant satisfies

$$
\mathbb E(|b_{123}|^2\mid x_1,v_1)\ge p_*/4>0.
\tag{NGF.7}
$$

Every profile here is computed from the original force, actual A1 arrays,
normalization, and configured threshold. When the threshold test fails this
certificate supplies no positive bound; it does not change that threshold.
The finite-population product probability $p_*$ is retained explicitly and
is not asserted to have a population-uniform lower bound.
:::

:::{prf:proof}
At $z=\epsilon w$, each kernel differs from $K^0$ by at most
$2tL_\rho\epsilon W$. If $\epsilon\le k_0/(4tL_\rho W)$ every kernel
is at least $k_0/2$. Count normalization directly bounds the error from
the linear term by $4\nu tL_\rho\epsilon^2W^2$.
For row normalization, with $s=\sum K$, $s_0=\sum K^0$,

$$
\sum_b|K_b/s-K_b^0/s_0|
\le2\sum_b|K_b-K_b^0|/s
\le8tL_\rho\epsilon W/k_0.
$$

Multiply by the maximum velocity difference $2\epsilon W$ and by $\nu$
to obtain its stated $C_F\epsilon^2$ bound. For the three prescribed
unit forces their magnitudes are at least $3\epsilon_*/4$.
The elementary inequality
$|F/|F|-F_0/|F_0||\le2|F-F_0|/|F_0|$ and the component phase bound
give $|C_a(z^*)-e_a|\le Q\epsilon_*\le1/12$.
Multilinearity and the unit column bounds give
$|b_{123}(z^*)-1|\le3Q\epsilon_*\le1/4$.

In the product ball $\max_a|z_a-z_a^*|<R_*\le1$ all velocity norms
are at most $H$, so all kernels are at least $k_R$. Kernel changes are
at most $2tL_\rho R_*$. Splitting force differences into velocity
differences and normalized weight differences gives a bound $LR_*$
for either normalization. In the row case the normalized weight
variation is at most $4tL_\rho R_*/k_R$; multiplying by $2H\nu$
gives exactly the displayed term in $L$. The count case has no denominator
error and is smaller. Thus the three force norms remain above the actual
threshold, and each color changes by at most $Q_RR_*$. The determinant
changes by at most $3Q_RR_*\le1/4$, giving $|b_{123}|\ge1/2$ throughout
this product ball. Lower-bound the full OU Gaussian density there by its
displayed worst-distance envelope and multiply by the product of its
three-dimensional ball volumes. This is $p_*$ and proves (NGF.7).
:::

(sec-ngf-selected-law)=
## 4. Exact survival, masking, and later-stage corrections

:::{prf:theorem} Native phase covariance conditional on actual terminal geometry
:label: thm-ngf-terminal-geometry-centroid

Use the unchanged canonical terminal boundary schedule and actual final
position noise $s=\sigma_x\sqrt h>0$. Retain the full post-collision
preparation, hence $(x_1,v_1)$, and condition on the **actual entire terminal
position array** $Y$. Put

$$
\alpha=tq,\quad \tau^2=\alpha^2+s^2,\quad
\chi=\frac{s^2}{\tau^2}>0,\quad m_i^x=x_{1,i}+tcv_{1,i},\quad
z_i^0=cv_{1,i}+\frac{tq^2}{\tau^2}(Y_i-m_i^x).
$$

The conditional law of the actual OU velocity array is exactly

$$
z_i=z_i^0+q\sqrt\chi\,\eta_i,
\qquad \eta_i\stackrel{\rm independent}{\sim}N(0,I_3).
\tag{NGF.13}
$$

Consequently, after additionally retaining $r_i=z_i-G$, the centroid
is independent of $r$ with conditional mean $g_Y=N^{-1}\sum_i z_i^0$
and covariance $\chi q^2I_3/N$. The matched B2 determinant formulas
(NGF.1)--(NGF.3) therefore hold with

$$
A_Y=B(r)e^{i\kappa\mathbf1\cdot g_Y},\qquad
\sigma_{N,Y}^2=3\kappa^2q^2\chi/N.
\tag{NGF.14}
$$

The same statement holds for a configured finite sum or normalized average
of matched B2 determinants whenever its retained weights, denominator and
additional masks are functions of the preparation, $Y$ and $r$. Replace
$B(r)$ by the actual weighted relative determinant sum $B_M(Y,r)$,
including its zero-denominator convention. This includes terminal-alive
masks, geometry and full-face localization computed from $Y$, the pre-O
companion indices, and B2 force-threshold masks. It does not include a
phase-velocity change or a subsequent velocity-dependent companion draw.

For the actual one-step survival-selected record, whose nonextinction event
$E$ is determined by $Y$, total variance gives

$$
\operatorname{Var}_{\mathbb C,E}(\sqrt N\,b^M)
\ge N(1-e^{-3\kappa^2q^2\chi/N})
\mathbb E_E|B_M(Y,r)|^2.
\tag{NGF.15}
$$

This retains the actual terminal geometry and survival law. Its per-step
conditional tangent coefficient is $3\kappa^2q^2\chi/h$.
For the unchanged reference, $s^2=0.0004$ and
$\chi=0.0004/[0.0004+0.0004(1-e^{-0.08})/2]$.
The fixed positive $\chi$ is derived from the configured noises, with no
change of their amplitudes or distributions.
:::

:::{prf:proof}
Before terminal classification the actual terminal positions are
$Y_i=m_i^x+tq\xi_i+s\zeta_i$, with independent standard normal
$\xi_i,\zeta_i$. For each coordinate the covariance of $\xi$ with
$Y-m^x$ is $tq$, and the latter has variance $\tau^2$. Subtracting
$(tq/\tau^2)(Y-m^x)$ from $\xi$ gives a Gaussian residual of variance
$1-t^2q^2/\tau^2=\chi$ and zero covariance with $Y-m^x$.
Factoring its joint Gaussian density proves independence. This calculation
holds independently in all rows and coordinates, proving (NGF.13) and then
the centroid/relative factorization exactly as in
{prf:ref}`thm-ngf-centroid-determinant-covariance`.

All B2 pair differences are unchanged under a common residual velocity
translation, even though its corresponding A2 positions change. Thus the
force and its threshold mask depend on $r$, and all color columns acquire
the common diagonal phase. Once $Y$ is conditioned upon, its geometry,
alive statuses, full-face supports and their configured weights are fixed.
The declared remaining preparation/relative masks also stay fixed. Every
determinant in the retained sum has the same scalar phase; extracting it
gives $b^M=B_M(Y,r)e^{i\kappa\mathbf1\cdot G}$. This proves the
conditional covariance formulas and the weighted extension, without
independence of completed walkers or an iid graph claim.

The canonical terminal boundary only marks the retained coordinates; its
one-step survival event is $\{\sum_i\mathbf1_D(Y_i)>0\}$. Conditioning
further on this event changes the posterior distribution of the preparation
and $Y$, but does not change their already computed conditional residual
law. Total variance under that selected probability gives (NGF.15).
The final velocity cap does not alter $Y$ or an earlier force-input color.
Other boundary schedules or phase velocities use their own maps and do not
inherit this argument. Substituting the reference noise values gives its
displayed $\chi$.
:::

:::{prf:theorem} Selection correction for the native centroid coefficient
:label: thm-ngf-selected-centroid-correction

Retain the preceding input and relative array. Let $E$ be the actual
positive-probability terminal or finite-window survival event. Include every
later update in its conditional probability

$$
w_E(r,g)=\mathbb P(E\mid x_1,v_1,r,G=g),\qquad
s_E(r)=\int w_E(r,g)\phi_N(g)\,dg,
$$

where $\phi_N$ is the actual $N(g_0,q^2I_3/N)$ density. For
$s_E(r)>0$ the selected centroid law is precisely
$w_E(r,g)\phi_N(g)dg/s_E(r)$. Define

$$
a_j(r)=\int w_E(r,g)e^{ij\kappa\mathbf1\cdot g}\phi_N(g)dg
\quad(j=0,1,2).
$$

For the matched B2 determinant, with its B2 threshold mask and no additional
terminal/graph/face mask, its selected moments are exactly

$$
\begin{aligned}
\mathbb E_E b&=B(r)a_1/a_0,\\
\operatorname{Var}_{\mathbb C,E}b&=|B(r)|^2(1-|a_1/a_0|^2),\\
\mathbb E_E(b-\mathbb E_Eb)^2&=B(r)^2\{a_2/a_0-(a_1/a_0)^2\}.
\end{aligned}
\tag{NGF.8}
$$

In the unchanged absorbing-box reference with $s=\sigma_x\sqrt h>0$,
one-step nonextinction has $w_E(r,g)>0$ for every finite $g$.
If $B(r)\ne0$ and $\kappa\ne0$, its selected complex variance is
therefore strictly positive. The same positivity applies to survival
through every fixed finite further window of this unchanged kernel.

For any actual additional mask or bounded recorded weight $M$, the
corresponding observable $b^M=B(r)e^{i\kappa\mathbf1\cdot G}M$ instead
uses the exact moments

$$
\begin{aligned}
\mathbb E_E b^M&=\frac{B(r)}{s_E(r)}
 \int e^{i\kappa\mathbf1\cdot g}
       \mathbb E[M\mathbf1_E\mid r,g]\phi_N(g)dg,\\
\mathbb E_E|b^M|^2&=\frac{|B(r)|^2}{s_E(r)}
 \int \mathbb E[|M|^2\mathbf1_E\mid r,g]\phi_N(g)dg.
\end{aligned}
\tag{NGF.9}
$$

The real/imaginary product moments follow with $M^2$ and the second
harmonic. Configured normalized averages retain their own actual
denominator inside $M$ or inside the complete observable. The complete
incoming-state posterior remains in the outer expectation.

The final velocity cap does not change a recorded B2 force-input color.
For a PF alignment using the completed capped velocity $u_i^+=\psi_V(v_i^+)$,
the exact correction is instead

$$
C_i^{\rm PF}=\operatorname{diag}
 (e^{i\kappa(u_i^+-z_i)})C_i^{\rm B2}.
\tag{NGF.10}
$$

Here $v_i^+$ uses the actual newly evaluated B2 potential and viscous
forces. Its dependence on $g$, the landscape and the cap is retained;
there is no generally common determinant phase after this rephasing.
:::

:::{prf:proof}
Conditioning the original joint Gaussian/future-innovation law on $E$
multiplies its centroid density by its exact future conditional survival
probability and divides by $s_E$. Insert (NGF.1), its modulus, and its
square into this density. Their three integrals prove (NGF.8).

At every finite B2 input, independent final position noise has a strictly
positive density on all of $(\mathbb R^3)^N$. Landing all rows in any
ball inside the configured nonempty box has positive probability. It
implies nonextinction, regardless of the uncapped B2 velocities, proving
$w_E>0$. The same fact can be iterated through every finite surviving
future window; finite stage states and their actual fresh inputs remain
inside its positive kernel probabilities. A random variable of modulus one
has an expectation of modulus one only when it is constant almost surely:
expand its mean-square distance from that expectation. The selected centroid
density is positive everywhere and $g\mapsto e^{i\kappa\mathbf1\cdot g}$
is nonconstant for $\kappa\ne0$. Hence $|a_1/a_0|<1$.

Retaining $M$ in the same conditional integrals proves (NGF.9); products
and variances use exactly those retained moments. These formulas allow
terminal-alive and graph masks to depend on the centroid rather than treating
them as relative-array functions. Finally (NGF.10) is the componentwise
phase identity for the same force and two different phase velocities.
Potential forces and the final cap affect $u_i^+$, whereas they occur after
the recorded matched force input and cannot change (NGF.1) for that record.
:::

(sec-ngf-discharge)=
## 5. Amplitude discharge and remaining limiting fluctuation estimates

:::{prf:proposition} Discharge and residual gauge-fluctuation estimate
:label: prop-ngf-discharge-register

The preceding results discharge the exact full-complex determinant centroid
coefficient, its conditional centering, its finite real covariance, its
$\sqrt N$ phase expansion, and nonzero native determinant amplitude for
the stated exact-mask regimes. They retain both Gaussian normalizations
and the actual complete upstream update. They add no full-law LSI,
stationary-chaos, iid-graph, gauge-Gaussianity, or variance premise.
The terminal-geometry calculation additionally discharges its actual
one-step survival and terminal-position mask correction, with the strictly
positive configured coefficient in (NGF.15).

For a raw B2 family with fixed reference kinetic parameters, (NGF.4)
reduces a sufficient amplitude estimate to

$$
\inf_N\mathbb E|B_{ijk,N}|^2>0
\tag{NGF.11}
$$

for its actual threshold and incoming law. This estimate is now proved in
{prf:ref}`thm-uda-uniform-selected-amplitude` for the unchanged reference,
its actual $10^{-12}$ threshold, both normalizations, and incoming QSD.
The proof retains the complete tagged intermediate history: three mandatory
revivals and a population-average Gaussian tail budget give the large-$N$
bound; complex diagonal dominance covers the remaining finite populations.
Almost-sure finite nonzero amplitude and (NGF.7) alone would not prove
(NGF.11), since their Gaussian product bound can tend to zero with $N$.

For the actual survival-selected or further masked family, the coefficient
conditional on preparation, terminal geometry and relative array is (NGF.15)
whenever that theorem's consumed maps agree. The amplitude
$\inf_N\mathbb E_E|B_{M,N}(Y,r)|^2$ is also proved for the one-step
survival-selected, terminal-alive determinant of the same reference in
{prf:ref}`thm-uda-uniform-selected-amplitude`. Other masks, localization
weights, growing-graph averages and B1/default readers need their own
amplitude estimate. The construction uses revived tagged rows, so a mask
deleting accepted clones does not inherit it. For future-window selection or masks
outside that theorem, the coefficient and centering requiring uniform control
are instead the exact quantities in (NGF.8)--(NGF.9).
In particular, the raw-coordinate conditional mean correction is

$$
B(r)\left\{\frac{a_1(r)}{a_0(r)}-
 e^{i\kappa\mathbf1\cdot g_0-3\kappa^2q^2/(2N)}\right\}.
\tag{NGF.12}
$$

{prf:ref}`cor-uda-uniform-scaled-determinant` consequently gives a positive
population-uniform complex variance lower bound for the reference's
$\sqrt N$ determinant coordinate. No vanishing bound at the $N^{-1/2}$
scale is established here for this
selection correction or for the fluctuation of the relative-array
conditional mean. Thus these results identify a native fluctuation component
and its coefficient, while the full limiting gauge evolution and nonzero
curvature covariance remain undischarged.

The reference's actual conditionally centered centroid innovation now has
nonzero subsequential Gaussian-mixture limits, with every fixed mixed
moment converging, in {prf:ref}`thm-uda-native-centroid-innovation-limit`.
The proof uses the bounded coefficient, the derived uniform amplitude
and the exact remaining independent Gaussian, without a joint-law LSI.
For the full fixed tagged determinant, exchangeability gives zero mean
and the same positive-probability amplitude event proves that its
$\sqrt N$ scaling is non-tight in
{prf:ref}`prop-uda-fixed-tag-centering-nontightness`. An empirical or
growing-graph channel average is a different readout; its centering,
weighting and scaling must remain in the required channel-limit proof.
:::

:::{prf:proof}
Each discharged assertion is the conclusion of its respective complete
calculation above. The inequality (NGF.4) has a prefactor tending to the
positive reference value $3\kappa^2q^2$, so a uniform positive amplitude
estimate would give a uniform complex variance lower bound for the raw
scaled determinant. The proved local-ball probability in (NGF.7) contains
$N$ Gaussian factors, with no uniform bound on those factors. It consequently
does not provide the asserted infimum. The completed-state convergence
theorems retain the distinction between completed and intermediate tagged
histories in their statements. Formula (NGF.12) follows by subtracting
(NGF.2) from (NGF.8). None of their exact finite integrals supplies a
population rate for that difference. The terminal-geometry factor $\chi$
is fixed and positive in the reference, so (NGF.15) similarly reduces its
selected variance lower bound to the actual selected masked amplitude,
now supplied by {prf:ref}`thm-uda-uniform-selected-amplitude` for the
stated reference readout. Total variance gives
{prf:ref}`cor-uda-uniform-scaled-determinant`; a lower bound alone does
not establish tightness or convergence of the full scaled coordinate.
The Gram/triangle centroid cancellation
in (NGF.1) also prevents inferring their positive limiting variance from this
determinant coefficient. These are the exact logical limits of the proofs.
:::

:::{prf:remark} The native innovation process now has a proved drift and bracket
:label: rem-ngf-native-process-advance

{prf:ref}`lem-ngp-causal-reordering` retains the whole actual update
while revealing its terminal positions, relative innovations and final
independent centroid in causal order. In this filtration,
{prf:ref}`thm-ngp-exact-martingale-bracket` identifies the centered
determinant innovation's exact zero drift, covariance and pseudo-covariance.
{prf:ref}`thm-ngp-native-process-limit` proves a nonzero subsequential
process limit at every finite list of native grid times, with its complete
martingale bracket and all fixed mixed moments.
{prf:ref}`cor-ngp-selected-process-limit` proves the same limit for
the actual whole-window survival selection. Future selection is not
asserted to preserve the finite-population centroid independence.

This discharges the time-dependent centroid innovation component.
The full native empirical or graph gauge channels retain their other
increments, amplitudes, weights and correlations. Their full evolution
and curvature identification are the remaining field-level tasks.
:::

:::{prf:remark} A complete native gauge-dynamics regime beyond the centroid component
:label: rem-ngf-complete-scaled-regime

{prf:ref}`thm-ntg-full-process-limit` now proves the full
finite-population coordinate process limit in a specified family of
existing timestep, acceptance-saturation and cap configurations.
{prf:ref}`thm-ntg-native-projector-process` transports the actual B2
projector cylinders to that process and derives their complete drift and
bracket, including every accepted donor jump and shared collision.
Its evaluated chart has a strictly positive rank-four diffusion covariance.

These conclusions retain the existing unbounded-boundary tag and the
declared scaling of the existing fields. They do not rename this family
as the unchanged terminal-box reference. The literal fixed-cap and
fixed-acceptance small-step families are characterized in
{prf:ref}`thm-ntg-fixed-reference-noninfinitesimal`.
The population/graph continuum field remains a separate limit of the
same declared readouts.
:::
