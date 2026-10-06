# Native metric variations and mechanical likelihood stress

(sec-native-gravity-parameter-ledger)=
## Complete parameters and actual geometric branches

:::{prf:definition} Native geometric variation record
:label: def-ng-complete-geometric-record

Retain the complete execution record $\mathfrak P$ of
{prf:ref}`def-native-complete-execution-record`. A variation below is a
differentiable curve $\theta\mapsto\mathfrak P_\theta$ in the numerical fields
or supplied maps of an existing configuration, with its discrete tags fixed.
Every field not varied retains its original value. All initial, landscape,
boundary, donor, history, eligibility, clone, kinetic, arithmetic, recording and
calibration data remain arguments of the results. Classical matrix derivatives
refer to real-coordinate evaluation of these existing formulas with their
configured branch thresholds. Returned finite-arithmetic payloads retain their
actual numerical comparison errors; no derivative of rounding is replaced by
a real-coordinate derivative.

The Rust geometric record includes `GeometryStageConfig.pipeline`, `schedule`
and `write_diffusion`; every field of `GeometryPipelineConfig`; the complete
payloads of its metric, volume, weights and curvature estimators; and any
`RunConfig.geometry_reward`, `potential`, `physics_metric` and reward shift.
It includes the `QftExecutionConfig.graph_viscosity` coefficient and weight key,
`curl.beta_curl`, the separate dense-viscosity branch, innovation shifts and
the actual Gaussian, uniform or other configured innovation law. The geometry
cache, its stale-step counter, previous-cell-volume data and immutable
conditional-fitness context are part of the consumed state. Geometry
coordinates have dimension $d$; kinetic coordinates can have a larger ambient
dimension $a$.

Two existing geometric branches used below are distinct:

1. `RunConfig::einstein_hilbert()` and its parameterized
   `GasConfig::einstein_hilbert(T,h)` use a `DropLast` projection, relative-trace
   neighbor-covariance metric, `SqrtDetMetric` volume, the named conformal
   Laplacian curvature, and Einstein–Hilbert-density reward. The reward enters
   fitness and cloning. Graph viscosity uses normalized
   `RiemannianKernelVolume` weights, with the configured Boris rotation. The
   preset's O innovation is **isotropic Gaussian**, with variance
   $T(1-e^{-2\gamma h})I$, $\gamma=1$. Its default potential gradient is zero;
   a supplied `RunConfig.potential` is evaluated separately. The actual preset
   has clone period $20$, elastic restitution, uniform mutual companion
   pairings, sample-standard-deviation normalization, logistic amplitude $2$
   and zero additive floor. These fields are retained when applying the
   statements to that preset, rather than replaced by smooth-global
   normalization or metric noise.
2. The conditional-fitness metric provider in
   {prf:ref}`thm-algorithmic-conditional-metric-law` supplies
   $g=\epsilon_\Sigma I+[H]_+$ and noise factor
   $B=\sqrt{2\gamma T}\,g^{-1/2}$ in its supported smooth global/local
   normalization branch. Its frozen-query Hessian is differentiated with all
   of its actual normalization dependencies. This is another existing
   configured branch, with its own support restrictions. It is not the
   neighbor-covariance metric of item 1.

A third configured choice, consuming the recorded `geometry.diffusion`
observation as a noise factor, uses the covariance of the **executed** factor.
The geometry code floors each metric eigenvalue at
$f_D=\max(\texttt{min_eig},10^{-6})$ when forming this factor. Its effective
precision is therefore $\widetilde g=Q\operatorname{diag}(\max(g_k,f_D))Q^T$,
not automatically the recorded tensor $g$.

The real Gaussian statements refer to independent Gaussian innovations in
$\mathcal E$. Fixed-seed numerical execution retains its actual deterministic
stream and does not inherit a sampling likelihood assertion without a
comparison result.
:::

:::{prf:definition} Evaluated differentiable geometry branch
:label: def-ng-evaluated-geometry-branch

At a recorded evaluation, a differentiable branch is specified by the actual
strict tests: eligibility and source identities remain fixed; the site
partition, affine rank, projection, minimum-image choices and tessellation
remain fixed; all encountered determinant, distance, row-sum, positive-part,
pseudo-inverse and eigenvalue-clamp thresholds have a strict margin; and all
local matrix solves are nonsingular. A spectrum may have repeated eigenvalues
inside one smooth spectral branch. Repeated eigenvalues are not themselves a
failure of differentiability.

For a covariance metric with $C\succeq0$, ridge $\rho>0$, and
$\tau=1$ in the absolute branch or $\tau=\operatorname{tr}C/d>0$ in the
relative branch, put $M=C+\rho\tau I$. The especially simple inverse branch is
the directly evaluated regime

$$
\lambda_{\min}(M)>d\epsilon_{\mathcal E}\lambda_{\max}(M),\qquad
\frac{b_-}{\tau}<\lambda_k(M)^{-1}<\frac{b_+}{\tau}
\quad\hbox{for every configured bound},
$$

where absent bounds impose no test and $\epsilon_{\mathcal E}$ is the run
precision's epsilon. In the relative branch $\tau=0$ invokes the code's
$\tau=1$ fallback; it is not replaced by a division by zero. `Strict` rejects
an estimate whose actual repair flag is set. Other successful repaired
branches remain defined, with their piecewise spectral derivative.

These are calculated tests on the existing maps. They specify where the
classical first variations below hold; no global regularity of the random
record is assumed. At a threshold the original readout remains the configured
piecewise readout, and the result below does not claim a classical derivative.
:::

(sec-native-gravity-metric-tangent)=
## The metric, curvature reward and graph-force first variations

:::{prf:theorem} Actual covariance-metric tangent and explicit ridge response
:label: thm-ng-covariance-metric-tangent

On a branch in {prf:ref}`def-ng-evaluated-geometry-branch`, write the recorded
neighbor displacements as $z_{ij}$, with $k_i$ outgoing neighbors. The code's
covariance is

$$
C_i=\frac1{\max(k_i,1)}\sum_{j\sim i}z_{ij}z_{ij}^T.
$$

Its variation and the regularized matrix variation are

$$
\dot C_i=\frac1{\max(k_i,1)}\sum_{j\sim i}
(\dot z_{ij}z_{ij}^T+z_{ij}\dot z_{ij}^T),\qquad
\dot M_i=\dot C_i+(\dot\rho\tau_i+\rho\dot\tau_i)I.
$$

Here $\dot\tau_i=\operatorname{tr}(\dot C_i)/d$ on the positive relative
branch, and $\dot\tau_i=0$ on the absolute branch. In the inverse branch,

$$
\boxed{\dot g_i=-g_i\dot M_i g_i},\qquad
\|\dot g_i\|_F\leq\lambda_{\min}(M_i)^{-2}\|\dot M_i\|_F.
$$

For all smooth pseudo-inverse/clamp branches, diagonalize
$M_i=Q_i\operatorname{diag}(\lambda_k)Q_i^T$. Let $f_{i,\theta}$ be the
code's scalar pseudo-inverse, followed by its bounds $b_-/\tau_i,b_+/\tau_i$.
It equals $1/t$ on a retained, unclamped branch, a configured bound on a
clamped branch, and the clamped value of zero on an excluded
pseudo-inverse branch. In the same spectral coordinates,

$$
(Q_i^T\dot g_iQ_i)_{kl}
=f_i[\lambda_k,\lambda_l](Q_i^T\dot M_iQ_i)_{kl}
 +\mathbf1_{k=l}\,\partial_\theta f_{i,\theta}(\lambda_k),
$$

where the divided difference uses $f_i'(\lambda_k)$ when the two eigenvalues
coincide. The parameter derivative holds the scalar argument fixed and
retains the moving bounds. The cutoff is used to choose the locally constant
retained/excluded branches, not differentiated as an inverse eigenvalue.

In particular, varying only $\rho$ at a fixed nonzero covariance in the
relative branch keeps $Q_i$ and $\tau_i$ fixed. On retained eigenvalues,

$$
\partial_\rho g_{i,k}=
\begin{cases}
-\tau_i/(\lambda_k(C_i)+\rho\tau_i)^2,
  &\text{inverse value strictly inside the bounds},\\
0,&\text{strictly clamped to a fixed }b_\pm/\tau_i.
\end{cases}
$$

An excluded eigenvalue also has derivative zero while its exclusion and clamp
branches stay fixed. Thus the preset's ridge response is explicit at every
record satisfying its branch tests, including repaired metric strata.
:::

:::{prf:proof}
The graph is fixed on the evaluated branch. Differentiate each outer product
in its arithmetic mean. Differentiate the definition of $\tau_i$ and the
ridge to obtain $\dot M_i$. Differentiating $M_i g_i=I$ gives the inverse
formula and its norm estimate.

For the spectral formula, first suppose the eigenvalues are simple.
Differentiate $M_iQ_i=Q_i\operatorname{diag}(\lambda_k)$ and
$Q_i^TQ_i=I$. The off-diagonal rotation term for $g_i$ is
$(f_i(\lambda_k)-f_i(\lambda_l))
(Q_i^T\dot M_iQ_i)_{kl}/(\lambda_k-\lambda_l)$. The diagonal term is the
ordinary chain rule. Inside a repeated eigenspace the scalar function is
smooth on one branch; taking the divided-difference limit gives $f_i'$ and
the same formula independently of the chosen basis. This can also be
verified by expanding a polynomial in $M_i$ and then the locally analytic
inverse or constant spectral branch. The strict cutoff and clamp margins
prevent switching the relevant branch in the differentiation.

For a ridge-only variation, $M_i$ is $C_i$ plus a scalar multiple of the
identity. Its eigenvectors are unchanged. Differentiating each retained
inverse eigenvalue gives the displayed negative derivative. Its configured
bounds, excluded zero and $\tau_i$ are constant in this variation, giving
zero on their fixed branches. Ineligible rows have the code's identity
spectrum and zero metric variation while eligibility remains fixed.
:::

:::{prf:theorem} Native conformal reward and normalized-weight variation
:label: thm-ng-native-reward-weight-variation

Retain the configured `ConformalLaplacian`, `SqrtDetMetric` and both weight
keys of the Einstein–Hilbert preset, or the same independently selected
components in another existing geometric configuration. Let their
determinant floors be $f_R$ and $f_V$, their reward scale be $\lambda_R$, and
their weight length scales be $\ell_w$. At a differentiable branch define

$$
u_i=\frac{\log\max(\det g_i,f_R)}{2d},\qquad
v_i=\sqrt{\max(\det g_i,f_V)},\qquad
D_{ij}=z_{ij}^T\frac{g_i+g_j}{2}z_{ij}.
$$

Their exact first variations are

$$
\begin{aligned}
\dot u_i&=\begin{cases}
\operatorname{tr}(g_i^{-1}\dot g_i)/(2d),&\det g_i>f_R,\\
0,&\det g_i<f_R,
\end{cases}\\
\dot v_i&=\begin{cases}
(v_i/2)\operatorname{tr}(g_i^{-1}\dot g_i),&\det g_i>f_V,\\
0,&\det g_i<f_V,
\end{cases}\\
\dot D_{ij}&=2\dot z_{ij}^T\frac{g_i+g_j}{2}z_{ij}
 +z_{ij}^T\frac{\dot g_i+\dot g_j}{2}z_{ij}.
\end{aligned}
$$

If a determinant floor itself is varied, its active floored branch instead
has $\dot u_i=\dot f_R/(2df_R)$ or $\dot v_i=\dot f_V/(2\sqrt{f_V})$.
The inverse is evaluated only on the displayed positive-determinant branch.
On a repaired singular metric the strictly active positive floor gives the
zero derivative above, without evaluating $g_i^{-1}$.
The graph kernel-volume raw weight is
$k_{ij}=e^{-D_{ij}/(2\ell_w^2)}v_j$, with

$$
\dot k_{ij}=k_{ij}\left[
-\frac{\dot D_{ij}}{2\ell_w^2}
+\frac{D_{ij}\dot\ell_w}{\ell_w^3}
+\frac{\dot v_j}{v_j}\right].
$$

For inverse-Riemannian-distance raw weights, the code's constants are
$s=10^{-8}$ and

$$
k_{ij}=(\sqrt{\max(D_{ij},s)}+s)^{-1},\qquad
\dot k_{ij}=-\frac{k_{ij}^2\mathbf1_{D_{ij}>s}}
                       {2\sqrt{D_{ij}}}\dot D_{ij}.
$$

Each normalized row has $w_{ij}=k_{ij}/\max(S_i,t)$,
$S_i=\sum_{j\sim i}k_{ij}$, $t=10^{-12}$. Hence

$$
\boxed{\dot w_{ij}=
\frac{\dot k_{ij}}{\max(S_i,t)}
-\frac{k_{ij}\mathbf1_{S_i>t}}{\max(S_i,t)^2}
          \sum_{l\sim i}\dot k_{il}}.
$$

For an unnormalized configured row, $\dot w_{ij}=\dot k_{ij}$. The actual
curvature, allocated reward and their first variations are

$$
\begin{aligned}
R_i&=-2(d-1)\sum_{j\sim i}w_{ij}(u_j-u_i),
&r_i&=\lambda_R R_iv_i,\\
\dot R_i&=-2(d-1)\sum_{j\sim i}
 [\dot w_{ij}(u_j-u_i)+w_{ij}(\dot u_j-\dot u_i)],
&\dot r_i&=\dot\lambda_R R_iv_i
       +\lambda_R(\dot R_iv_i+R_i\dot v_i).
\end{aligned}
$$

For `CurvatureOnly`, remove $v_i$ from the allocation. For `Unit` volume,
$v_i=1$ and $\dot v_i=0$. With Voronoi volume the actual cell derivative and
its fallback derivative enter by the product rule; this theorem makes no
replacement of that branch by the determinant density.
:::

:::{prf:proof}
Jacobi's identity, obtained by differentiating the determinant expansion,
gives $\partial_\theta\log\det g_i=\operatorname{tr}(g_i^{-1}\dot g_i)$.
The strict floor tests choose the logarithm or constant branch. Differentiate
the edge quadratic form, exponential kernel and inverse-distance expression.
Differentiating the configured row denominator gives the quotient formula,
including its floor. Finally differentiate the exact conformal-Laplacian
sum and the actual allocation. These are finite sums; no continuum curvature
formula, shape regularity, independence or expectation interchange is used.
:::

:::{prf:theorem} Native graph viscosity, curl and Boris tangent
:label: thm-ng-native-graph-kick-tangent

Use the actual graph snapshot consumed by a B stage. In every force and curl
sum below, retain only neighbors eligible under that stage's actual mask;
ineligible source rows are unchanged. Its dimension is the
kinetic ambient dimension $a$, including any coordinate omitted by the
geometry projection. On its differentiable branch the force is

$$
F_i=\nu\sum_{j\sim i}w_{ij}(v_j-v_i),\qquad
\dot F_i=\dot\nu\sum_jw_{ij}(v_j-v_i)
+\nu\sum_j[\dot w_{ij}(v_j-v_i)+w_{ij}(\dot v_j-\dot v_i)].
$$

The curl fit uses $z_{ij}^{\rm amb}$ and

$$
B_i^x=\sum_jw_{ij}z_{ij}^{\rm amb}(z_{ij}^{\rm amb})^T,\qquad
A_i^x=\sum_jw_{ij}(F_j-F_i)(z_{ij}^{\rm amb})^T,
$$

$$
\eta_i=\max(\sqrt{\epsilon_{\mathcal E}}
                    \operatorname{tr}(B_i^x)/a,
                    m_{\mathcal E}),\qquad
J_i=A_i^x(B_i^x+\eta_iI)^{-1},\qquad
\Omega_i=(J_i-J_i^T)/2,
$$

where $m_{\mathcal E}$ is the precision's minimum positive number. Its
derivative is computed by differentiating the displayed sums, with

$$
\dot J_i=\dot A_i^x(B_i^x+\eta_iI)^{-1}
-J_i(\dot B_i^x+\dot\eta_iI)(B_i^x+\eta_iI)^{-1},
\qquad \dot\Omega_i=(\dot J_i-\dot J_i^T)/2.
$$

The max branch specifies $\dot\eta_i$. Let the B duration be $h/2$, set
$q=h/4$, and retain the separately supplied potential gradient $D_i$. The
actual graph kick is

$$
\begin{aligned}
u&=v+q(F(v)-D),\\
C_i&=(I-Z_i)^{-1}(I+Z_i),\qquad
Z_i=\beta_{\rm curl}h\Omega_i/4,\\
v^r_i&=C_i u_i,\\
v^+&=v^r+q(F(v^r)-D).
\end{aligned}
$$

The second $F$ uses the rotated velocities. Its derivative is the above
force formula evaluated at those velocities. The remaining derivatives are

$$
\begin{aligned}
\dot u&=\dot v+\dot q(F-D)+q(\dot F-\dot D),\\
\dot C_i&=(I-Z_i)^{-1}\dot Z_i(C_i+I),\\
\dot v_i^r&=\dot C_iu_i+C_i\dot u_i,\\
\dot v^+&=\dot v^r+\dot q(F(v^r)-D)
                         +q(\dot F(v^r)-\dot D).
\end{aligned}
$$

In particular $\|C_i\|_{\rm op}=1$ and
$\|\dot C_i\|_{\rm op}\leq2\|\dot Z_i\|_{\rm op}$. These formulas apply
separately at B1 and B2, with the actual provider reevaluated at B2 and with
the actual cached graph, including its staleness. For disabled curl the
rotation is the identity. The configured dense-viscosity branch retains its
separate force derivative and is not identified with the graph force.
:::

:::{prf:proof}
Differentiate the force sum. The implemented least-squares solve is
$J_i(B_i^x+\eta_iI)=A_i^x$. Its derivative gives the displayed $\dot J_i$;
antisymmetrization gives $\dot\Omega_i$. No reciprocal-weight assumption was
used, so row-normalized forces retain their actual momentum sources.

The code's quarter kick, Cayley rotation, and recomputed quarter kick give
the displayed composition. Differentiating
$(I-Z_i)C_i=I+Z_i$ proves the Cayley tangent. A real skew $Z_i$ satisfies
$\|(I-Z_i)x\|^2=\|x\|^2+\|Z_ix\|^2$, so
$\|(I-Z_i)^{-1}\|_{\rm op}\leq1$. Its Cayley transform is orthogonal by
multiplying its transpose, giving $\|C_i+I\|\leq2$ and the tangent bound.
Differentiate the two kicks in their executed order. Ineligible rows retain
the code's unchanged velocities, and their derivative is unchanged while
the mask remains fixed. The result includes the potential provider's own
derivative; a geometric reward does not supply that derivative.
:::

(sec-native-gravity-selection-response)=
## The geometric reward's actual selection response

:::{prf:theorem} Einstein–Hilbert reward contribution to the native gate score
:label: thm-ng-geometric-selection-score

Fix a finite pre-selection population and the actual companion assignment,
with $m\geq2$ eligible rows. For the preset's `LegacySample` reward channel
let

$$
\bar r=m^{-1}\sum_ir_i,\quad
\sigma_r^2=(m-1)^{-1}\sum_i(r_i-\bar r)^2,\quad
s_r=\sigma_r+\varepsilon_r,\quad
z_i^r=(r_i-\bar r)/s_r.
$$

On the branch $\sigma_r>0$ its derivatives are

$$
\dot{\bar r}=m^{-1}\sum_i\dot r_i,\quad
\dot\sigma_r=\frac{\sum_i(r_i-\bar r)\dot r_i}
                    {(m-1)\sigma_r},\quad
\dot z_i^r=\frac{\dot r_i-\dot{\bar r}}{s_r}
       -\frac{(r_i-\bar r)(\dot\sigma_r+\dot\varepsilon_r)}{s_r^2}.
$$

The diversity channel uses its actual measurements and the same derivative
form. For a constant channel the code returns zero standardized values;
classical derivatives through a varying sample standard deviation are not
claimed at that threshold. For smooth-global normalization replace $s_r$ by
$(\sigma_{r,\rm pop}^2+\varepsilon_r^2)^{1/2}$ and differentiate this actual
expression; local normalizers retain the differentiated neighborhood weights
in {prf:ref}`thm-algorithmic-conditional-metric-law`.

Write the two positive maps as $M_r,M_s$, and retain the actual fitness powers
$\alpha,\beta$. With $f_i=M_r(z_i^r)^\alpha M_s(z_i^s)^\beta$,

$$
\frac{\dot f_i}{f_i}
=\dot\alpha\log M_r+\dot\beta\log M_s
+\alpha\frac{\dot M_r(z_i^r)}{M_r(z_i^r)}
+\beta\frac{\dot M_s(z_i^s)}{M_s(z_i^s)}.
$$

For a fixed living-row donor $j$ on an enabled cloning step put
$a_{ij}=(f_j-f_i)/[(f_i+\varepsilon_c)\zeta]$ and
$p_{ij}=[a_{ij}]_0^1$. On the strict interior branch $0<a_{ij}<1$,

$$
\dot p_{ij}=
\frac{\dot f_j-\dot f_i}{(f_i+\varepsilon_c)\zeta}
-p_{ij}\left[
\frac{\dot f_i+\dot\varepsilon_c}{f_i+\varepsilon_c}
+\frac{\dot\zeta}{\zeta}\right].
$$

On a strictly zero/saturated branch, or a disabled scheduled cloning step,
$\dot p_{ij}=0$. Conditional on the complete donor choices, the actual
independent acceptance uniforms give the gate log-likelihood score

$$
L_C=\sum_{i:0<p_i<1}
\frac{(C_i-p_i)\dot p_i}{p_i(1-p_i)},\qquad
\mathbb E[L_C\mid\text{donor context}]=0,
$$

$$
\mathbb E[L_C^2\mid\text{donor context}]
=\sum_{i:0<p_i<1}\frac{\dot p_i^2}{p_i(1-p_i)}.
$$

The donor pairing itself has zero parameter score for ridge, curvature and
reward-scale variations of the preset, since its Fisher–Yates permutation law
is uniform and independent of those values. If its existing donor-law fields
are varied, retain the actual joint matching score. Revival, if present,
uses its actual configured donor law. Mutual pairing and restitution still
retain their deterministic joint state map after these gates.

For any fixed bounded branch observable $A(C)$ of the gate outcome,

$$
\partial_\theta\mathbb E[A(C)\mid\text{donor context}]
=\mathbb E[A(C)L_C\mid\text{donor context}],
$$

with the usual direct $\mathbb E[\dot A]$ term when $A$ also depends on
$\theta$. This gives an explicit contribution of the actual metric and
curvature variation to the native selection response.
:::

:::{prf:proof}
Differentiate the finite sample mean and variance. The mean derivative drops
out of the variance derivative because the centered measurements sum to
zero. The positive-map and power formulas follow by differentiating
$\log f_i$. The acceptance expression is precisely the configured clipped
ratio; differentiation on its strict branches gives the displayed formulas.

Conditional on the donor context, the named acceptance streams are
independent in the analytic innovation convention. Differentiating their
Bernoulli factors gives $L_C$. Each centered factor has mean zero and
variance $\dot p_i^2/[p_i(1-p_i)]$; independence makes cross terms zero.
Uniform matching probabilities do not depend on the geometric reward.
Differentiate the finite gate sum to prove the observable response. This
does not factor the mutual matching law into independent donor rows, or
replace restitution by independent copy kernels.
:::

(sec-native-gravity-likelihood-stress)=
## Metric stress from the actual Gaussian likelihood

:::{prf:theorem} Gaussian kinetic metric score and Fisher coercivity
:label: thm-ng-native-thermal-metric-score

Condition on an actual O input, including its frozen metric context, masks,
prepared velocities, any addressed innovation shift and all preceding
choices. Let $Y_i$ be each eligible velocity immediately after the O update
and before boundary reconciliation. For the existing independent Gaussian
branch its conditional law is

$$
Y_i\sim\mathcal N(m_i,\Sigma_i),\qquad
m_i=c v_i+s_hB_i b_i,\qquad
\Sigma_i=s_h^2B_iB_i^T,
$$

where $b_i$ is the configured innovation shift, $c=e^{-\gamma h}$ and
$s_h^2=(1-e^{-2\gamma h})/(2\gamma)$ with value $h$ at $\gamma=0$.
For positive-definite executed factors and $h>0$, the negative log density is

$$
\mathcal A_O(Y)=\frac12\sum_i\left[
\log\det\Sigma_i+(Y_i-m_i)^T\Sigma_i^{-1}(Y_i-m_i)
+a\log(2\pi)\right].
$$

For any differentiable existing parameter variation of these conditional
coefficients, write $e_i=Y_i-m_i$. Its conditional log-likelihood score is

$$
\boxed{L_O=-\dot{\mathcal A}_O
=\sum_i\left[
\dot m_i^T\Sigma_i^{-1}e_i
+\frac12\left(
e_i^T\Sigma_i^{-1}\dot\Sigma_i\Sigma_i^{-1}e_i
-\operatorname{tr}(\Sigma_i^{-1}\dot\Sigma_i)\right)\right].}
$$

It has mean zero and variance

$$
\boxed{\mathcal I_O
=\sum_i\left[
\dot m_i^T\Sigma_i^{-1}\dot m_i
+\frac12\operatorname{tr}
\left((\Sigma_i^{-1/2}\dot\Sigma_i\Sigma_i^{-1/2})^2\right)
\right].}
$$

For the conditional-fitness metric branch with $\gamma T>0$, no addressed
shift and $\Sigma_i=q_Tg_i^{-1}$,
$q_T=T(1-e^{-2\gamma h})>0$, a metric-only variation has

$$
\dot{\mathcal A}_O
=\frac12\sum_i\operatorname{tr}
\left[\left(q_T^{-1}e_ie_i^T-g_i^{-1}\right)\dot g_i\right],
\qquad
\mathcal I_O^{g}
=\frac12\sum_i\|g_i^{-1/2}\dot g_i g_i^{-1/2}\|_F^2.
$$

Consequently the realized native metric likelihood stress is
$\mathcal T_i^{\rm noise}=\tfrac12(q_T^{-1}e_ie_i^T-g_i^{-1})$.
Its conditional mean is zero, and its Fisher form is strictly positive for
every nonzero actual metric tangent. If $\epsilon_\Sigma>0$ and the evaluated
metrics satisfy $\epsilon_\Sigma I\preceq g_i\preceq G I$, then

$$
\frac1{2G^2}\sum_i\|\dot g_i\|_F^2
\leq\mathcal I_O^g
\leq\frac1{2\epsilon_\Sigma^2}\sum_i\|\dot g_i\|_F^2.
$$

For the Hessian metric the lower bound is programmed, and the upper profile
is calculated as
$G=\epsilon_\Sigma+\max_i\|H_i\|_{\rm op}$ on the actual O input.
For the quadratic finite-step fixture this profile has all finite moments
by {prf:ref}`lem-algorithmic-quadratic-finite-step-moments`; it is not asserted
to be uniform in population or time.

If the configured noise consumes `geometry.diffusion`, replace $g_i$ in
these covariance formulas by its effective precision $\widetilde g_i$ and
its actual scalar amplitude. A variation hidden by the diffusion floor has
zero covariance Fisher response. For the Einstein–Hilbert preset,
$\Sigma_i=q_T I$: a pure geometric-ridge variation has **zero direct O
covariance score**. Its geometric response remains in preparation, selection,
graph forces and readouts, as computed in the preceding theorems.
:::

:::{prf:proof}
The O map is affine in the freshly drawn Gaussian innovations. Its actual
factor, shift and mask are measurable at the conditioned input, giving the
product Gaussian law. Differentiate its determinant and quadratic form,
using $\partial\Sigma^{-1}=-\Sigma^{-1}\dot\Sigma\Sigma^{-1}$ and
$\dot e=-\dot m$, to obtain $L_O$.

Set $Z_i=\Sigma_i^{-1/2}e_i$. These vectors are independent standard
Gaussians conditional on the input. For symmetric $M$, expansion using
$\mathbb E Z_kZ_l=\delta_{kl}$ and
$\mathbb E Z_kZ_lZ_pZ_q=
\delta_{kl}\delta_{pq}+\delta_{kp}\delta_{lq}+\delta_{kq}\delta_{lp}$ gives
$\operatorname{Var}(Z^TMZ)=2\operatorname{tr}(M^2)$. Odd centered Gaussian
moments vanish, so the linear and centered quadratic scores have zero
cross covariance. Independence across rows proves the variance formula.

For $\Sigma_i=q_Tg_i^{-1}$, substitute
$\dot\Sigma_i=-q_Tg_i^{-1}\dot g_i g_i^{-1}$ into the score. The variance
reduces to the stated Fisher form. The lower and upper eigenvalue bounds
give the two Frobenius estimates by applying them on both sides of the
symmetric tangent. The Hessian upper profile follows from
$\|H_+\|\leq\|H\|$, and its moments from the cited actual-fixture proof.
Finally use the code's eigenvalue floor for the observed diffusion factor,
or the preset's isotropic factor. Writing a geometry diffusion observation
does not make it the executed factor when the preset selects isotropic
noise.
:::

:::{prf:corollary} Conditional native geometry response without a test-function derivative
:label: cor-ng-native-noise-response

At the fixed O input of {prf:ref}`thm-ng-native-thermal-metric-score`, let
$\Phi(Y)$ be any bounded measurable readout of that conditional output,
with its map fixed under the variation. It can include the fixed downstream
pushforward of the output. Then

$$
\partial_\theta\mathbb E_\theta\Phi
=\mathbb E_\theta[\Phi L_O],\qquad
|\partial_\theta\mathbb E_\theta\Phi|
\leq\sqrt{\operatorname{Var}_\theta\Phi}\sqrt{\mathcal I_O}.
$$

For a fixed measurable conditioning event $E$ with positive probability,

$$
\partial_\theta\mathbb E_\theta[\Phi\mid E]
=\operatorname{Cov}_\theta(\Phi,L_O\mid E),\qquad
|\partial_\theta\mathbb E_\theta[\Phi\mid E]|
\leq\sqrt{\operatorname{Var}_\theta(\Phi\mid E)}
       \sqrt{\mathcal I_O/\Pr_\theta(E)}.
$$

If a density $p_\theta(z)$ of a fixed descriptor pushforward exists, its
native action $-\log p_\theta(z)$ has weak score
$-\mathbb E[L_O\mid Z=z]$. The native action conditional on a fixed geometry
descriptor $G$ has weak metric score

$$
-\mathbb E[L_O\mid Z,G]+\mathbb E[L_O\mid G].
$$

Thus the source action on a geometry fiber includes the actual conditional
thermal stress, with its own geometry-conditioning subtraction. This result
applies to variations of the conditional factor. If a global parameter also
changes the preceding preparation or the downstream map, their actual
derivative terms must additionally be retained.
:::

:::{prf:proof}
In a sufficiently small neighborhood of positive-definite conditional
covariances, the differentiated Gaussian density is bounded by a constant
times a polynomial times $\exp(-c|Y|^2)$ after accounting for its bounded
mean. This function is integrable. Differentiation under the integral is
therefore justified for every bounded measurable $\Phi$, without assuming
smoothness of the readout or its tessellation. The centered score and
Cauchy–Schwarz give the first bound. Differentiate the quotient with the
fixed event indicator to obtain the conditional covariance. Its score second
moment is at most $\mathbb E L_O^2/\Pr(E)$, proving the bound.

The weak derivative of the descriptor law is the pushforward of the signed
measure $L_O\,d\Pr$. Conditional expectation gives its Radon–Nikodym
derivative with respect to the descriptor law. Apply this to $(Z,G)$ and
subtract the $G$ marginal log-density score for the conditional native
action. The same proof works directly with conditional densities on their
positive support; it does not require a Lebesgue density for the complete
staged state, whose deterministic stages can have singular support.
:::

(sec-native-gravity-first-variation-scope)=
## Native first-variation budget and remaining identification

:::{prf:proposition} Stage-complete geometric first-variation budget
:label: prop-ng-native-first-variation-budget

Fix a fully recorded finite history of an existing configuration, and an
actual parameter curve for which its evaluated branches remain
differentiable. The derivative of its metric, curvature reward and graph B
updates is the executed composition of
{prf:ref}`thm-ng-covariance-metric-tangent`,
{prf:ref}`thm-ng-native-reward-weight-variation` and
{prf:ref}`thm-ng-native-graph-kick-tangent`, with every affected stage's
actual inputs. Its discrete-choice likelihood contribution contains
{prf:ref}`thm-ng-geometric-selection-score`; its Gaussian kinetic likelihood
contribution contains {prf:ref}`thm-ng-native-thermal-metric-score`.
Potential forces, diversity, boundary maps, donor-history rescoring,
geometry cache, determinant floors and changed readout maps retain their
separate executed derivatives. These are explicit terms computed from the
same history. They determine the finite native metric first variation on
that branch.

For the unmodified Einstein–Hilbert preset this budget contains the actual
conformal-Laplacian reward and metric-dependent graph mechanics. It contains
no direct metric-dependent O covariance term. For the existing
fitness-Hessian-noise branch the covariance contribution is the proved
thermal stress tensor and Fisher form. These statements hold for the
complete parameter record in the first definition and do not equate the
two variants.
:::

:::{prf:proof}
At a fixed discrete history every deterministic stage is the actual
configured finite-dimensional map. Differentiate their composition in its
executed order. The metric, reward and graph submaps have the derivatives
proved above; all other configured submaps retain their own derivatives.
The conditional probability of the discrete choices factorizes in stage
order, so its log derivative is the sum of its actual conditional scores.
The Gaussian innovation factor has the conditional density and score proved
above. This composition is a pathwise finite-record statement. Integrating
the branch derivatives through a random changing tessellation, boundary or
gate threshold additionally requires the corresponding boundary terms or a
proved differentiation-under-the-law estimate; no such estimate follows
merely from the pathwise chain rule.
:::

:::{prf:remark} Remaining native gravitational correspondence
:label: rem-ng-native-gravity-residual

The new results evaluate the native metric tangent, geometric selection
score, graph mechanical response, Gaussian thermal likelihood stress and
geometry-fiber score from the implemented parameters. They are not a
variational equation obtained by substituting an Einstein action for the
native probability law.

To obtain a continuum dynamical metric equation from this budget, the
remaining estimates are the integrated contribution of retessellation and
other branch interfaces, uniform control of the score and material metric
increments on the actual coupled law, and joint convergence of these terms
with the existing mechanical source/flux equations. The absolute-ridge
passive geometry estimates in {prf:ref}`thm-native-jg-absolute-metric` do not
transfer to the relative-ridge feedback preset without proving its own
regime tests and estimates. A conditional noise response with preparation
frozen does not eliminate the geometric selection response or memory of
unobserved population variables.

An Einstein correspondence additionally requires identifying the limit of
this **native** score budget, its stress, constraints and coefficients with
the target spacetime equation. No Newton constant, cosmological constant,
Lorentzian metric or conservation constraint is supplied by these finite
calculations. The preset's conformal-Laplacian reward is its actual recorded
scalar recipe; its name does not replace that recipe by the first variation
of a continuum Einstein–Hilbert functional. Existing curvature and
transport/diffusion theorems remain available with their stated hypotheses.
:::
