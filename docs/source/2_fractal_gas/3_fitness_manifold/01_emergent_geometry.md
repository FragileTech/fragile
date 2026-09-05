(sec-emergent-geometry)=
# Emergent Geometry from Adaptive Diffusion

**Prerequisites:** {doc}`/source/2_fractal_gas/1_the_algorithm/02_fractal_gas_latent`,
{doc}`/source/2_fractal_gas/convergence_program/14_a_geometric_gas_c3_regularity`,
and {doc}`/source/2_fractal_gas/convergence_program/17_geometric_gas`.

(sec-tldr-emergent-geometry)=
## TLDR

A positive definite diffusion shape $D$ defines a metric $g=D^{-1}$.
For the smooth Hessian construction, $g=H+\epsilon_\Sigma I$ requires a
positive spectral margin. The implementation also clips eigenvalues; its
recorded metric is identified separately below. Both constructions admit
explicit ellipticity and Lipschitz bounds.

The metric determines lengths and the volume density $J=\sqrt{\det g}$.
A stationary sampling law depends on the complete generator. We derive the
overdamped and kinetic generators that have spatial density
$Z^{-1}J e^{-U/T}$, and distinguish those comparison models from the
implemented kinetic, cloning, and killing dynamics.

Gram determinants give cell areas and volumes. Their quadrature errors,
importance-sampling errors, and convergence rates follow from the regularity
and probability estimates established in the analytical chapters, with the
hypotheses of each estimate retained.

(sec-emergent-geometry-intro)=
## Introduction

:::{div} feynman-prose
Imagine measuring a cloud of small random displacements at one point.
In one direction the cloud is wide; in another it is narrow. Use that cloud
as a ruler: a displacement is large when it is large compared with the noise
in its direction. The inverse covariance gives precisely this ruler.

There are three calculations to keep apart. The first builds a ruler from a
covariance matrix. The second asks how rulers at neighboring points fit
together. That is where derivatives, connections, and curvature enter. The
third asks where moving particles spend their time. That depends on forces,
friction, noise, cloning, and survival.

Knowing the ruler solves the first problem. It supplies coefficients for the
other two. We will carry those coefficients through the calculations, so that
a volume formula, a stationary density, and a mixing estimate each have a
specific reason to hold.
:::

(sec-adaptive-diffusion-tensor)=
## The Adaptive Diffusion Tensor

:::{div} feynman-prose
Start in the coordinates used by the algorithm. A symmetric Hessian has
orthogonal eigenvectors, and the adaptive noise assigns a separate scale to
each one. Positive curvature of the fitness potential makes the corresponding
noise smaller. Negative curvature needs a separate rule: adding a small
positive number does not necessarily make a negative eigenvalue positive.

The analytical construction uses a shift larger than the negative part of
the spectrum. The code also has a clipping rule. Both are simple to inspect
one eigenvalue at a time, but their derivatives behave differently at the
clipping threshold.
:::

:::{prf:definition} Adaptive diffusion shape and metric
:label: def-adaptive-diffusion-tensor-latent

Work in the declared Euclidean latent coordinates. On a fixed alive-set and
companion stratum, let $V_{\mathrm{fit}}^{(i)}(z;S,c)$ be the per-walker
fitness with the indicated other coordinates and sampled companion
assignment frozen. When this function is $C^2$, set

$$
H(z;S,c)=\nabla_z^2V_{\mathrm{fit}}^{(i)}(z;S,c).
$$

The smooth shifted-Hessian construction is

$$
g=H+\epsilon_\Sigma I,\qquad
\Sigma_{\mathrm{reg}}=g^{-1/2},\qquad
D_{\mathrm{reg}}=\Sigma_{\mathrm{reg}}\Sigma_{\mathrm{reg}}^T=g^{-1},
$$

on the set where $g$ is positive definite. These are a metric, a noise
square root, and a normalized covariance shape, respectively.

If a finite update has noise amplitude
$\Sigma_{\mathrm{step}}=c_2\Sigma_{\mathrm{reg}}$, then its conditional
velocity-increment covariance is $c_2^2D_{\mathrm{reg}}$. If a continuous
SDE has amplitude $\sqrt{2T}\Sigma_{\mathrm{reg}}$, its second-order
generator is $T D_{\mathrm{reg}}^{ij}\partial_i\partial_j$ in the
coordinates on which the noise acts. The prefactor and those coordinates
must be specified before identifying a generator.
:::

:::{prf:definition} Expected fitness and a limiting fitness field
:label: def-mean-field-fitness-field

Let $c$ denote the complete companion assignment used to compute the sampled
fitness, and let $P_S(dc)$ be its actual joint law. Define

$$
\overline V_N^{(i)}(z;S)
  =\int V_{\mathrm{fit},N}^{(i)}(z;S,c)\,P_S(dc).
$$

When the state varies with the query $z$, differentiation includes the
dependence of both the sampled fitness and $P_S$. In particular, a derivative
of this expectation need not equal the expectation of a frozen-companion
derivative. The distinction and the joint-assignment formula are specified in
{prf:ref}`def-c3-fitness-laws` and {prf:ref}`lem-c3-joint-companion-law`.

A mean-field fitness field $V_{\mathrm{fit}}(z;\mu)$ is an identified limit
of these finite-particle expected fields, with their prescribed
normalizations, when such a limit exists. Defining its Hessian requires
$C^2$ regularity. Using it as the limit of finite-particle Hessians requires
convergence of those derivatives. The geometry statements below apply to the
particular sampled or expected field that has been specified.
:::

:::{prf:proposition} Metric represented by the implemented Hessian branch
:label: prop-geometry-clipped-metric

For a successful eigendecomposition without an added retry jitter, the full
Hessian branch of `KineticOperator._compute_diffusion_tensor` uses

$$
H_{\mathrm{sym}}=\tfrac12(H+H^T),\qquad
g_{\mathrm{clip}}=\epsilon_\Sigma I+(H_{\mathrm{sym}})_+,
\qquad
\Sigma_{\mathrm{step}}=c_2g_{\mathrm{clip}}^{-1/2},
$$

where $A_+$ replaces the negative eigenvalues of a symmetric matrix by zero.
The diagonal branch uses
$(g_{\mathrm{clip}})_{ii}=\epsilon_\Sigma+\max(H_{ii},0)$ and zero
off-diagonal entries. Thus, for $c_2>0$,

$$
g_{\mathrm{clip}}
=\left(c_2^{-2}\Sigma_{\mathrm{step}}
                   \Sigma_{\mathrm{step}}^T\right)^{-1}.
$$

The un-clipped formula $g=H+\epsilon_\Sigma I$ agrees with the full branch
when $H$ is symmetric positive semidefinite. An eigensolver retry adds a
further scalar shift before clipping; the fallback uses the clipped diagonal.
The `grad_proxy`, `voronoi_proxy`, and supplied-tensor branches require
their own covariance identification.

These statements describe the existing implementation, without changing its
update.
:::

:::{prf:proof}
For $H_{\mathrm{sym}}=Q\operatorname{diag}(\lambda_j)Q^T$, the code shifts
by $\epsilon_\Sigma$, clips each shifted eigenvalue below at
$\epsilon_\Sigma$, and takes its inverse square root. The resulting metric
eigenvalue is

$$
\max(\lambda_j+\epsilon_\Sigma,\epsilon_\Sigma)
=\epsilon_\Sigma+\max(\lambda_j,0).
$$

Squaring the symmetric noise matrix gives the stated covariance and inverse.
The diagonal computation is the same scalar identity.
:::

### Coordinates and the meaning of the metric

:::{prf:remark} Coordinate convention
:label: rem-geometry-coordinate-hessian

The inverse covariance transforms as a metric when the covariance transforms
as a contravariant tensor. If $y=\psi(x)$ and $A=D\psi(x)$, then
$D_y=A D_x A^T$ and $g_y=A^{-T}g_xA^{-1}$.

An ordinary coordinate Hessian does not obey this law under an arbitrary
nonlinear coordinate change: its transformation includes first derivatives
of the potential and second derivatives of the coordinate map. The
construction here therefore uses the algorithm's fixed Euclidean
coordinates. An intrinsic alternative would require a specified background
metric $G$, its covariant Hessian $\operatorname{Hess}_G V$, and correctly
raised indices in $g^{-1}$. This alternative is a different mathematical
specification.

A Hessian eigenvalue measures curvature of a scalar potential in these
coordinates. Riemannian curvature depends on derivatives of the metric.
A constant anisotropic $g$ is flat, regardless of its determinant.
:::

(sec-regularity-convergence)=
## Regularity and Convergence

:::{div} feynman-prose
A matrix can be positive at every point and still become arbitrarily close
to singular far away. Bounds need to hold on the whole region visited in
the argument. On an unbounded state space, a local calculation is combined
with the confining estimates from the convergence chapters.

The regularity proofs already give the right inputs. We retain their
distinction between fixed sampled assignments, averaged assignments, and
changes of alive set. This matters because a smooth function inside each
region can jump when the rule selecting the region changes.
:::

(sec-uniform-ellipticity)=
### Two-sided spectral control

:::{prf:assumption} Spectral bounds for the smooth construction
:label: assump-spectral-floor-latent

On the specified region and collection of states, suppose

$$
-\Lambda_- I\preceq H(z;S)\preceq\Lambda_+ I,\qquad
\epsilon_\Sigma>\Lambda_-.
$$

Write $m=\epsilon_\Sigma-\Lambda_->0$ and
$M=\epsilon_\Sigma+\Lambda_+$. A conclusion uniform in $N$ requires
$m$ bounded away from zero and $M$ bounded above uniformly in $N$.

The second-derivative bounds in {prf:ref}`thm-c2-regularity` provide such
spectral estimates where their measurement, normalization, companion-law,
and stratum hypotheses hold. The third-derivative bounds in
{prf:ref}`thm-c3-regularity` provide the spatial Lipschitz estimates used
below. Global or uniform bounds require the corresponding uniform input
bounds from those theorems.
:::

:::{prf:theorem} Uniform ellipticity from metric bounds
:label: thm-uniform-ellipticity-latent

Under {prf:ref}`assump-spectral-floor-latent`,

$$
mI\preceq g\preceq MI,\qquad
\frac1M I\preceq D_{\mathrm{reg}}\preceq\frac1m I.
$$

For the clipped construction, an upper bound
$H_{\mathrm{sym}}\preceq\Lambda_+I$ instead gives

$$
\epsilon_\Sigma I\preceq g_{\mathrm{clip}}
 \preceq\bigl(\epsilon_\Sigma+\max(\Lambda_+,0)\bigr)I,
$$

and the inverse bounds follow by reciprocation. Its positive lower metric
bound holds without a lower bound on the Hessian. A positive lower
diffusion bound still requires the upper Hessian bound.

For kinetic noise these inequalities concern the velocity block; the full
phase-space diffusion matrix has a zero position block.
:::

:::{prf:proof}
Orthogonally diagonalize $H$. The shifted metric eigenvalues are
$\epsilon_\Sigma+\lambda_j(H)\in[m,M]$. Inversion reciprocates each
eigenvalue, proving the first assertion. The clipped eigenvalues lie
between $\epsilon_\Sigma$ and
$\epsilon_\Sigma+\max(\Lambda_+,0)$ by
{prf:ref}`prop-geometry-clipped-metric`. Finally, the vector fields of
velocity noise have zero position component, so their phase-space
covariance has the asserted zero block.
:::

(sec-lipschitz-continuity)=
### Lipschitz bounds and smoothness thresholds

:::{prf:lemma} Operator-Lipschitz bound for inverse square root
:label: lem-operator-lipschitz-inv-sqrt-latent

For symmetric positive definite matrices $A,B\succeq mI$,

$$
\|A^{-1/2}-B^{-1/2}\|_F
 \leq\frac{\|A-B\|_F}{2m^{3/2}},\qquad
\|A^{-1}-B^{-1}\|_F\leq\frac{\|A-B\|_F}{m^2}.
$$
:::

:::{prf:proof}
Along $A_t=A+t(B-A)$, let $X_t=A_t^{-1/2}$. Differentiating
$X_tA_tX_t=I$ gives

$$
L_t(\dot X_t)=-X_t(B-A)X_t,\qquad
L_t(K)=KA_tX_t+X_tA_tK.
$$

In an orthonormal eigenbasis of $A_t$,

$$
(L_tK)_{ij}=(\sqrt{\lambda_i}+\sqrt{\lambda_j})K_{ij},
$$

so $\|L_t^{-1}\|_{F\to F}\leq(2\sqrt m)^{-1}$. Since
$\|X_t\|_{\mathrm{op}}\leq m^{-1/2}$,

$$
\|\dot X_t\|_F\leq\frac{\|B-A\|_F}{2m^{3/2}}.
$$

Integrating in $t$ proves the first bound. The resolvent identity
$A^{-1}-B^{-1}=A^{-1}(B-A)B^{-1}$ proves the second.
:::

:::{prf:proposition} Lipschitz continuity of adaptive diffusion
:label: prop-lipschitz-diffusion-latent

Let $\mathfrak d$ be a specified distance on the field's inputs $u=(z,S)$.
If $g(u)\succeq mI$ and

$$
\|g(u)-g(u')\|_F\leq L_g\mathfrak d(u,u'),
$$

then

$$
\|\Sigma_{\mathrm{reg}}(u)-\Sigma_{\mathrm{reg}}(u')\|_F
 \leq\frac{L_g}{2m^{3/2}}\mathfrak d(u,u'),\qquad
\|D_{\mathrm{reg}}(u)-D_{\mathrm{reg}}(u')\|_F
 \leq\frac{L_g}{m^2}\mathfrak d(u,u').
$$

For a frozen state on a convex coordinate region,
$\sup_z\|\nabla_zH\|_{\mathbb R^d\to F}\leq K_3$ gives
$L_g=K_3$ for $\mathfrak d(z,z')=|z-z'|$. A bound for
$\mathfrak d=|z-z'|+W_1(\mu_S,\mu_{S'})$ requires, in addition, a
Hessian dependence estimate in that Wasserstein distance. Spatial $C^3$
regularity alone provides only the frozen-state bound.

For the clipped construction,
$\|H_{\mathrm{sym}}(u)-H_{\mathrm{sym}}(u')\|_F
\leq L_H\mathfrak d(u,u')$ implies the same conclusions with
$m=\epsilon_\Sigma$ and $L_g=L_H$.
:::

:::{prf:proof}
The smooth assertions follow by the preceding lemma and integration of
$\nabla_zH$ along the segment.

For the clipped assertion, the map $A\mapsto A_+$ is the orthogonal
projection, in the Frobenius inner product, onto the closed convex cone
of positive semidefinite matrices. To verify this directly, write
$A=A_+-A_-$ with $A_\pm\succeq0$ and $A_+A_-=0$. For any $P\succeq0$,

$$
\|A-P\|_F^2
=\|A_+-P\|_F^2+\|A_-\|_F^2+2\operatorname{tr}(PA_-)
\geq\|A_-\|_F^2.
$$

Thus $A_+$ minimizes the distance. The two projection variational
inequalities, added together, give

$$
\|A_+-B_+\|_F^2
 \leq\langle A_+-B_+,A-B\rangle_F
 \leq\|A_+-B_+\|_F\|A-B\|_F.
$$

Clipping is therefore nonexpansive. Apply the inverse-square-root lemma
after adding $\epsilon_\Sigma I$.
:::

:::{prf:remark} Derivatives needed by geometric constructions
:label: rem-geometry-regularity-order

A $C^3$ fitness field gives a $C^1$ smooth shifted metric and locally
Lipschitz diffusion on a positive-margin region. Classical pointwise
Riemann curvature requires a $C^2$ metric, hence a $C^4$ fitness field
for this Hessian construction. Higher-order estimates follow from
{prf:ref}`thm-main-cinf-regularity-fitness-potential-full` under that
theorem's full hypotheses.

Clipping is Lipschitz, but is generally nondifferentiable when a Hessian
eigenvalue crosses zero. Curvature formulas for the recorded clipped field
require a smooth spectral region, a justified smooth reconstruction, or an
explicit weak interpretation. Lipschitz bounds themselves remain valid
across the clipping threshold.

For continuous dynamics, local Lipschitz coefficients give local
well-posedness. Nonexplosion additionally follows from a coercive
Lyapunov bound for the full drift and jump mechanism, as in
{prf:ref}`cor-gg-well-posedness`. A finite-step update has its own
transition kernel and uses the discrete convergence results.
:::


(sec-equivalence-principle)=
## Covariance and Geometric Descriptions

:::{div} feynman-prose
At a single point, stretch each eigenvector by the square root of its metric
eigenvalue. The noise cloud becomes round. Doing this point by point is a
choice of orthonormal frame.

Now let the frame vary as a particle moves. Its variation contributes to the
equations. This is why inspecting the covariance is insufficient to identify
a process as Brownian motion on a manifold. The drift must be carried through
the same calculation. For a kinetic process, one must also remember that the
random kicks act on velocity while position is transported by velocity.
:::

:::{prf:observation} Two descriptions of the same covariance
:label: obs-two-perspectives-latent

For $g=D^{-1}$, the columns of $g^{-1/2}$ are an orthonormal frame for
the metric:

$$
(g^{-1/2})^Tg\,g^{-1/2}=I.
$$

Consequently, a noise increment with coordinate covariance $2TD\,dt$ has
isotropic covariance $2TI\,dt$ in that frame.

For a diffusion acting on position, write $J=\sqrt{\det g}$ and

$$
\Delta_g f=J^{-1}\partial_i(JD^{ij}\partial_jf).
$$

Its coordinate generator admits the exact identity

$$
T D^{ij}\partial_i\partial_jf+b^j\partial_jf
=T\Delta_g f+
 \left[b^j-TJ^{-1}\partial_i(JD^{ij})\right]\partial_jf.
$$

This is a description of the same generator with a specified additional
vector field. For kinetic noise, the second derivatives are instead
$\partial_{v_i}\partial_{v_j}$; the position-space Laplace–Beltrami
identity does not replace those velocity derivatives.
:::

:::{prf:proof}
The first assertion follows by multiplying the three matrices.
Expanding the divergence in $\Delta_g$ gives the second identity,
including its first-order term.
:::

:::{prf:remark} Coordinate changes and preservation of the process
:label: rem-equivalence-reinterpretation

A coordinate change preserves the law of a process when its full generator
is transformed, including the Itô second-derivative term or the equivalent
Stratonovich vector-field transformation. A position-dependent frame
is not generally the Jacobian of a coordinate change. An isometry to
Euclidean space would require a flat metric locally, with additional global
conditions for a global isometry.

The frame identity above therefore establishes a covariance interpretation.
Identical transition laws, invariant measures, and rates require the full
generator identity.
:::

(sec-kinetic-evolution)=
## Kinetic Evolution and Stationary Measures

:::{div} feynman-prose
A velocity kick does not immediately move position. In the equations
$dx=v\,dt$, the position path has finite variation between jumps. This small
observation settles the Stratonovich correction when the noise coefficient
depends only on position: the noise never differentiates that coefficient
in a noisy direction.

There is a separate question about equilibrium. Suppose we want particles
to sample volume measured by the new ruler, weighted by a potential. We can
construct dynamics with exactly that stationary density and verify it by a
flux calculation. Comparing those dynamics with the algorithm then tells us
which terms would need to agree.
:::

(sec-stratonovich-formulation)=
### Stratonovich and Itô forms of velocity noise

:::{prf:proposition} Conversion for the kinetic equations
:label: prop-geometry-kinetic-conversion

Between jumps, write the specified continuous model as

$$
dx_i=v_i\,dt,\qquad
dv_i=a_i(S)\,dt+\sum_\alpha\sigma_{i\alpha}(S)\circ dW_i^\alpha.
$$

Let $B_\ell(S)$ be the full phase-space noise vector fields. The equivalent
Itô drift is

$$
b_{\mathrm I}=b_{\mathrm S}
 +\frac12\sum_\ell(DB_\ell)B_\ell.
$$

If all noise coefficients depend only on positions, this correction is zero.
For noise field $\ell=(i,\alpha)$ acting only on velocity $v_i$, its
velocity component is

$$
\frac12\sum_{\alpha,b}
 \sigma_{i,b\alpha}(S)\,\partial_{v_i^b}\sigma_{i,a\alpha}(S).
$$

Dependence on the complete swarm is included by the full vector-field
formula when the noise fields have more general support.
:::

:::{prf:proof}
For a smooth test function $f$, the Stratonovich generator is

$$
b_{\mathrm S}\cdot\nabla f+
 \frac12\sum_\ell B_\ell(B_\ell f)
=\left(b_{\mathrm S}+\frac12\sum_\ell(DB_\ell)B_\ell\right)
 \cdot\nabla f+
 \frac12\sum_\ell(B_\ell B_\ell^T):\nabla^2f.
$$

Every $B_\ell$ has zero position component. It therefore differentiates a
position-only coefficient to zero. Taking the velocity components gives the
displayed formula. This is the calculation used in
{prf:ref}`lem-gg-geometric-drift`.
:::

The implemented BAOAB step freezes coefficients at its prescribed substeps.
Its finite-step noise covariance and its transition law are the operational
objects. A continuous SDE approximation requires the consistency conditions
in {prf:ref}`rem-c3-baoab-inputs` and
{doc}`/source/2_fractal_gas/convergence_program/13_quantitative_error_bounds`.

(sec-geometric-drift)=
### Geometric drift and exact Gibbs constructions

:::{prf:lemma} Geometric drift for an overdamped comparison model
:label: lem-geometric-drift-latent

Let $g$ be a $C^1$ positive definite spatial metric, $D=g^{-1}$,
$J=\sqrt{\det g}$, and $T>0$. The Itô position SDE

$$
dX^i=\left[-D^{ij}\partial_jU+
 T J^{-1}\partial_j(JD^{ij})\right]dt
 +\sqrt{2T}(g^{-1/2})^i_{\ \alpha}\,dW^\alpha
$$

has generator

$$
Lf=T\Delta_g f-\langle\operatorname{grad}_gU,
                            \operatorname{grad}_gf\rangle_g
 =\frac{T}{J e^{-U/T}}
   \partial_i\left(J e^{-U/T}D^{ij}\partial_jf\right).
$$

If $Z=\int J e^{-U/T}\,dx<\infty$, and the conservative realization
has a generator core with boundary conditions and decay that justify
the corresponding integration by parts, then

$$
\mu(dx)=Z^{-1}e^{-U/T}\,dV_g(x)
$$

is invariant and the generator is symmetric in $L^2(\mu)$.

If $g\succeq mI$ and
$\sup_{|e|=1}\|\partial_e g\|_{\mathrm{op}}\leq K_3$, the geometric
part $b_{\mathrm{geo}}^i=TJ^{-1}\partial_j(JD^{ij})$ satisfies

$$
|\nabla\log J|\leq\frac{dK_3}{2m},\qquad
|b_{\mathrm{geo}}|\leq\frac{3dTK_3}{2m^2}.
$$

This lemma specifies an overdamped comparison model. Its drift is not
automatically a term of the Fractal Gas velocity equation.
:::

:::{prf:proof}
Expand the divergence in the displayed generator. The coefficient of
$\partial_i f$ is the stated drift. For smooth compactly supported
$f,h$, or functions satisfying the stated boundary conditions,

$$
\int hLf\,d\mu
=-T\int(\nabla h)^TD\nabla f\,d\mu.
$$

Symmetry follows, and $h=1$ gives stationarity for the conservative
realization.

Jacobi's determinant formula gives

$$
\partial_e\log J=\frac12\operatorname{tr}(g^{-1}\partial_e g),
\qquad
\partial_eD=-D(\partial_e g)D.
$$

Thus $|\nabla\log J|\leq dK_3/(2m)$ and
$\|\partial_eD\|_{\mathrm{op}}\leq K_3/m^2$. Since
$\operatorname{div}D=\sum_j(\partial_jD)e_j$,

$$
|b_{\mathrm{geo}}|
\leq T\left(|\operatorname{div}D|+
                 \|D\|_{\mathrm{op}}|\nabla\log J|\right)
\leq\frac{3dTK_3}{2m^2}.
$$
:::

:::{prf:proposition} A kinetic model with the Riemannian Gibbs marginal
:label: prop-geometry-kinetic-gibbs

Let $g$ be a $C^2$ spatial metric and define the Hamiltonian on position
and canonical momentum by

$$
E(x,p)=U(x)+\frac12p^Tg^{-1}(x)p.
$$

Consider the specified comparison model

$$
dx=g^{-1}p\,dt,\qquad
dp=-\nabla_xE\,dt-\gamma p\,dt+
                       \sqrt{2\gamma T}\,g^{1/2}(x)\,dW,
\qquad \gamma,T>0.
$$

When its Gibbs density is normalizable and the dynamics are conservative
with a generator core and boundary behavior allowing the integrations below,

$$
\mu(dx\,dp)=Z_{\mathrm{phase}}^{-1}e^{-E(x,p)/T}\,dx\,dp
$$

is invariant. Its position marginal is

$$
\mu_x(dx)=Z_x^{-1}\sqrt{\det g(x)}e^{-U(x)/T}\,dx.
$$

In velocity coordinates $v=g^{-1}p$, the same model has

$$
dx^k=v^k\,dt,\qquad
dv^k=\left[-\Gamma^k_{ij}v^iv^j
 -(g^{-1}\nabla U)^k-\gamma v^k\right]dt
 +\sqrt{2\gamma T}(g^{-1/2})^k_{\ \alpha}\,dW^\alpha.
$$

The agreement of the noise shape with $g^{-1}$ is one part of this
specification. The metric force, geodesic drift, and thermostat scaling are
the other parts.
:::

:::{prf:proof}
The Hamiltonian vector field $(\nabla_pE,-\nabla_xE)$ has zero divergence
and annihilates $E$. Hence its transport preserves $e^{-E/T}\,dx\,dp$.
At fixed $x$, the momentum Ornstein–Uhlenbeck part has probability flux

$$
-\gamma\left[p\,e^{-E/T}
        +Tg\,\nabla_p(e^{-E/T})\right]=0,
$$

because $\nabla_pE=g^{-1}p$. The sum therefore preserves the displayed
density, subject to the conservative realization in the statement.

The Gaussian integral is

$$
\int_{\mathbb R^d}
 e^{-p^Tg^{-1}p/(2T)}\,dp
=(2\pi T)^{d/2}\sqrt{\det g},
$$

which proves the marginal formula.

To transform the equation, write $p_k=g_{kj}v^j$. Since $x$ has finite
variation,

$$
dp_k=g_{kj}\,dv^j+(\partial_\ell g_{kj})v^\ell v^j\,dt.
$$

Also
$\partial_kE=\partial_kU-\tfrac12(\partial_kg_{ij})v^iv^j$,
where the derivative on the left holds $p$ fixed. Multiplication by
$g^{ak}$ and symmetrization in $i,j$ gives the Christoffel term

$$
\Gamma^a_{ij}
=\frac12g^{ak}(\partial_i g_{kj}+\partial_jg_{ki}-\partial_kg_{ij}).
$$

The noise becomes $g^{-1}g^{1/2}=g^{-1/2}$ and has zero Stratonovich
correction by {prf:ref}`prop-geometry-kinetic-conversion`. Finally,
$dx\,dp=(\det g)\,dx\,dv$, so the velocity-coordinate density is

$$
Z_{\mathrm{phase}}^{-1}\det g(x)\,
 e^{-[U(x)+v^Tg(x)v/2]/T}.
$$

Its velocity integral gives the same spatial marginal.
:::

:::{div} feynman-prose
The determinant in the last calculation has a concrete origin. At a fixed
position, we integrate a Gaussian over momenta. The ellipsoid of allowed
momenta has a volume proportional to $\sqrt{\det g}$. That is why this
particular kinetic model produces Riemannian volume in its spatial
marginal.

Changing only the noise ellipsoid leaves the force and transport terms to
be checked. The calculation above makes that check possible: compare each
coefficient with the stated model, then include the cloning and killing
terms.
:::

(sec-riemannian-volume)=
## Riemannian Volume and Integration

:::{prf:remark} Stationary density, survival conditioning, and volume
:label: rem-riemannian-volume-drift

The exact Gibbs identities above apply to their specified conservative
generators. The Fractal Gas generator includes its actual force,
velocity diffusion, alignment, cloning, and killing terms. A proposed
density for that system must solve the resulting stationary equation or
the killed left-eigenmeasure equation.

For example, if $\mu$ is invariant for $L$ and soft killing has rate
$\kappa(x)$, then for $A=L-\kappa$,

$$
\int Af\,d\mu=-\int\kappa f\,d\mu.
$$

Thus the same $\mu$ solves $\mu A=-\alpha\mu$ precisely when
$\kappa=\alpha$ almost everywhere under $\mu$. Boundary killing requires
its own generator domain. For a general killed process a QSD $\nu$
satisfies $\nu Q_t=e^{-\alpha t}\nu$.
The invariant law of its Doob transform is the eigenfunction-weighted law
$\eta\nu$, normalized by $\nu(\eta)=1$, as proved in
{prf:ref}`prop-kl-doob-transform`.

The local QSD density estimates in
{doc}`/source/2_fractal_gas/convergence_program/11_hk_convergence`
and the confining estimates do not identify that density with an exact
Gibbs expression.
:::

(sec-volume-element)=
### Volume density and the reconstructed field

:::{prf:definition} Riemannian volume element
:label: def-riemannian-volume-element-latent

For a positive definite spatial metric $g(x)$ on a coordinate domain,

$$
dV_g(x)=J(x)\,dx,\qquad J(x)=\sqrt{\det g(x)}.
$$

For an oriented $k$-dimensional surface parametrized by
$\Phi:U\subset\mathbb R^k\to\mathbb R^d$, its induced volume density is

$$
\sqrt{\det\left[(D\Phi)^Tg(\Phi)D\Phi\right]}\,du.
$$

In particular, a constant $g$ changes the coordinate volume by a constant
factor while its Riemannian curvature remains zero.
:::

:::{prf:remark} The metric field used in a volume integral
:label: rem-volume-element-regime

A volume integral uses a spatial field, not only a list of matrices at
walker locations. One may specify a frozen-state query field, an identified
limiting field from {prf:ref}`def-mean-field-fitness-field`, or a
reconstruction from the data. Its regularity and positive margin are
part of the specification.

If reconstructed Hessians converge uniformly to $H$ and share a uniform
positive margin, the inverse and inverse-square-root bounds above give
uniform convergence of their covariance shapes. For the clipped branch,
uniform Hessian convergence gives uniform metric convergence by
nonexpansiveness. Propagation of chaos controls particle laws; identifying
these derivative fields requires the regularity and consistency
conditions in {prf:ref}`cor-continuum-consistency-conditional`.

When the metric and the samples come from the same swarm, their dependence
must be retained. The same-sample coverage estimate
{prf:ref}`lem-scutoid-adaptive-coverage` uses a pathwise upper metric
bound. An importance-sampling proof instead needs the joint conditions
stated below; conditioning on a fitted metric does not make its training
samples independent.
:::

:::{prf:lemma} Volume stability under a metric perturbation
:label: lem-geometry-volume-stability

Suppose $g\succeq mI$ and
$\|\widetilde g-g\|_{\mathrm{op}}\leq\eta<m$. For any rank-$k$
matrix $B$,

$$
(1-\eta/m)^{k/2}
 \leq
 \frac{\sqrt{\det(B^T\widetilde gB)}}
      {\sqrt{\det(B^TgB)}}
 \leq(1+\eta/m)^{k/2}.
$$

The same inequalities hold for integrated volumes of a fixed parametrized
surface when the bounds hold along that surface.
:::

:::{prf:proof}
The metric bound gives

$$
(1-\eta/m)g\preceq\widetilde g\preceq(1+\eta/m)g.
$$

Congruence by $B$ preserves these inequalities. Conjugate further by
$(B^TgB)^{-1/2}$; the resulting $k$ eigenvalues lie in
$[1-\eta/m,1+\eta/m]$. Multiply them, take square roots, and integrate
the pointwise inequalities for the final assertion.
:::


(sec-fan-triangulation)=
### Fan triangulation for areas

:::{div} feynman-prose
A loop of points does not select a unique spanning surface. We can select one
by joining every edge to a chosen center. The resulting collection of flat
triangles is a concrete object whose area we can measure.

For a planar polygon that is star-shaped about the center, the triangles
fill the polygon once. For a nonplanar loop, the fan is a chosen spanning
surface. If triangles overlap, summing their unsigned areas counts the
overlap with multiplicity. These are geometric choices made before the
metric enters the calculation.
:::

:::{prf:algorithm} Fan quadrature for Riemannian area
:label: alg-fan-triangulation-latent

**Input:** A cycle $z_0,\ldots,z_{n-1},z_n=z_0$ in a coordinate domain,
a metric field $g$, and the fan surface formed from
$z_c=n^{-1}\sum_{i=0}^{n-1}z_i$. Assume the triangles lie in the domain.
For the area of a planar enclosed region, require that this fan triangulates
that region without overlapping interiors.

**Construction:** Evaluate $g_c=g(z_c)$. For
$B_i=[z_i-z_c,\ z_{i+1}-z_c]$, set

$$
A_i^{(c)}=\frac12\sqrt{\det(B_i^Tg_cB_i)}
=\frac12\sqrt{
 ((z_i-z_c)^Tg_c(z_i-z_c))
 ((z_{i+1}-z_c)^Tg_c(z_{i+1}-z_c))
 -((z_i-z_c)^Tg_c(z_{i+1}-z_c))^2}.
$$

Return $A_g^{(c)}=\sum_i A_i^{(c)}$.

**Interpretation:** This is the exact area of the selected fan for the
constant metric $g_c$. For a variable metric it is a quadrature
approximation to the sum of the triangle areas.

**Cost:** One metric evaluation and $O(nd^2)$ arithmetic operations.
If $g\succeq mI$ and
$\|g(x)-g(z_c)\|_{\mathrm{op}}\leq K_1h$ on the fan, with
$h=\max_i|z_i-z_c|$ and $K_1h<m$, then

$$
|A_g-A_g^{(c)}|\leq\frac{K_1h}{m}\,A_g^{(c)}.
$$

Thus a family with $A_g^{(c)}=O(h^2)$ has absolute error $O(h^3)$.
The constants retain the metric bounds and the total fan area.
:::

:::{prf:proof}
The linear parametrization of each triangle has constant derivative $B_i$.
Its induced constant-metric area is the square root of its Gram determinant
times the area $1/2$ of the standard two-simplex. The perturbation estimate
is {prf:ref}`lem-geometry-volume-stability` with $k=2$; its two factors
are exactly $1\pm K_1h/m$. Sum over triangles.
:::

The following implementation evaluates that specified constant-metric fan
quadrature. Coordinates and the metric must use compatible units.

```python
from collections.abc import Callable

import numpy as np


def compute_riemannian_area_fan(
    vertices: np.ndarray,  # [n, d], without a repeated endpoint
    metric: Callable[[np.ndarray], np.ndarray],
) -> float:
    """Return the unsigned fan area with the metric frozen at its center.

    The supplied cycle specifies a chosen fan; overlapping triangles are
    counted with multiplicity. The metric must be symmetric positive definite.
    """
    if vertices.ndim != 2 or vertices.shape[0] < 3 or vertices.shape[1] < 2:
        raise ValueError("Expected at least three vertices in dimension >= 2.")
    center = vertices.mean(axis=0)
    g_center = np.asarray(metric(center))
    if not np.allclose(g_center, g_center.T):
        raise ValueError("The metric must be symmetric.")
    np.linalg.cholesky(g_center)

    edges = vertices - center
    next_edges = np.roll(edges, -1, axis=0)
    g11 = np.einsum("ni,ij,nj->n", edges, g_center, edges)
    g22 = np.einsum("ni,ij,nj->n", next_edges, g_center, next_edges)
    g12 = np.einsum("ni,ij,nj->n", edges, g_center, next_edges)
    # Exact Gram determinants are nonnegative; clipping handles roundoff.
    return float(0.5 * np.sqrt(np.maximum(g11 * g22 - g12**2, 0.0)).sum())
```

(sec-tetrahedral-decomposition)=
### Tetrahedral volumes and quadrature order

:::{prf:definition} Frozen-metric tetrahedron volume
:label: def-tetrahedron-volume-latent

For vertices $z_0,z_1,z_2,z_3\in\mathbb R^d$, $d\geq3$, let

$$
B=[z_1-z_0,\ z_2-z_0,\ z_3-z_0],\qquad
z_c=\tfrac14\sum_{j=0}^3z_j.
$$

The frozen-metric volume is

$$
V_g^{(c)}(T)=\frac16\sqrt{\det(B^Tg(z_c)B)}.
$$

The exact volume for a variable metric on the affine tetrahedron is

$$
V_g(T)=\int_{\Delta_3}
 \sqrt{\det\left[B^Tg(z_0+B\xi)B\right]}\,d\xi,
\qquad
\Delta_3=\{\xi_j\geq0:\ \xi_1+\xi_2+\xi_3\leq1\}.
$$

The first formula is exact when $g$ is constant along the tetrahedron.
:::

:::{prf:lemma} Barycentric metric quadrature
:label: lem-geometry-simplex-quadrature

Let $T$ be a nondegenerate affine $k$-simplex of diameter $h$, barycenter
$x_c$, and Euclidean $k$-volume $V_E$. Suppose on $T$

$$
mI\preceq g\preceq MI,\qquad
\|\partial_e g\|_{\mathrm{op}}\leq K_1,\qquad
\|\partial_e^2g\|_{\mathrm{op}}\leq K_2\quad(|e|=1),
$$

with $g$ of class $C^2$ on a neighborhood of $T$. Then

$$
|V_g(T)-V_g^{(c)}(T)|
\leq\frac12V_Eh^2 M^{k/2}
 \left[\frac{kK_2}{2m}
       +\left(\frac{k}{2}+\frac{k^2}{4}\right)\frac{K_1^2}{m^2}\right].
$$

In particular, if $V_E=O(h^k)$ with uniform coefficient bounds, the
absolute error is $O(h^{k+2})$. For tetrahedra this is $O(h^5)$.
Freezing every triangle at a common fan center instead of its own
barycenter uses the first-order estimate above.
:::

:::{prf:proof}
Choose an orthonormal matrix $Q\in\mathbb R^{d\times k}$ spanning the
simplex plane, and put $G(x)=Q^Tg(x)Q$,
$w(x)=\sqrt{\det G(x)}$. Along a unit direction in that plane,
Jacobi's formula gives

$$
\frac{w''}{w}
=\frac12\operatorname{tr}(G^{-1}G'')
 -\frac12\operatorname{tr}(G^{-1}G'G^{-1}G')
 +\frac14\left[\operatorname{tr}(G^{-1}G')\right]^2.
$$

Using $G\succeq mI$ and $w\leq M^{k/2}$ bounds $|w''|$ by the
coefficient in brackets times $M^{k/2}$. Taylor expansion about $x_c$
has a linear term whose integral over the simplex vanishes, since

$$
\int_T(x-x_c)\,d\mathcal H^k(x)=0.
$$

The remaining term is at most one half the second-derivative bound times
$|x-x_c|^2$. Integrating and using $|x-x_c|\leq h$ proves the result.
:::

The tetrahedron implementation uses the same frozen-metric convention.

```python
def compute_riemannian_volume_tetrahedron(
    vertices: np.ndarray,  # [4, d]
    metric: Callable[[np.ndarray], np.ndarray],
) -> float:
    """Return barycentric metric quadrature for an affine tetrahedron."""
    if vertices.ndim != 2 or vertices.shape[0] != 4 or vertices.shape[1] < 3:
        raise ValueError("Expected four vertices in dimension >= 3.")
    g_center = np.asarray(metric(vertices.mean(axis=0)))
    if not np.allclose(g_center, g_center.T):
        raise ValueError("The metric must be symmetric.")
    np.linalg.cholesky(g_center)
    edges = vertices[1:] - vertices[0]  # [3, d]
    gram = edges @ g_center @ edges.T
    return float(np.sqrt(max(float(np.linalg.det(gram)), 0.0)) / 6.0)
```

### Operational volume weights

:::{prf:proposition} Volume weights and a frozen neighbor kernel
:label: prop-geometry-volume-weights

For a symmetric positive definite recorded
$\Sigma_{\mathrm{step}}=c_2g^{-1/2}$ with $c_2>0$,

$$
J=\sqrt{\det g}=\frac{c_2^d}{\det\Sigma_{\mathrm{step}}}.
$$

The function `compute_riemannian_volume_weights` applies this determinant
factor to recorded Euclidean Voronoi volumes, with its numerical clamps and
interior-cell mask. Without active clamps, a retained cell has weight

$$
a_i=V_i^E J(x_i).
$$

This is frozen-site quadrature. Its approximation error is controlled by
variation of $J$ across that cell.

For a frozen symmetric nonnegative neighbor matrix $w_{ij}=w_{ji}$,
positive retained weights $a_i$, and positive row sums
$Z_i=\sum_jw_{ij}a_j$, define

$$
P_{ij}=\frac{w_{ij}a_j}{Z_i}.
$$

Then

$$
q_i=\frac{a_iZ_i}{\sum_\ell a_\ell Z_\ell}
$$

is invariant and satisfies detailed balance for this neighbor chain.
The identity applies before any additional degree, threshold, or
state-dependent modifications to the weights.
:::

:::{prf:proof}
Take determinants in $\Sigma_{\mathrm{step}}=c_2g^{-1/2}$ to obtain
the first identity. For the chain,

$$
q_iP_{ij}
=\frac{a_iw_{ij}a_j}{\sum_\ell a_\ell Z_\ell}
=q_jP_{ji}.
$$

Summation over $i$ yields invariance. Thus even this frozen chain has
stationary weights $a_iZ_i$, which reduce to weights proportional to
$a_i$ when $Z_i$ is constant. Its detailed-balance identity concerns the
specified neighbor chain; the full swarm has additional dynamics.
:::

(sec-monte-carlo-integration)=
### Monte Carlo integration with an identified sampling law

:::{div} feynman-prose
Suppose you sample twice as often in one region as in another. An integral
must compensate for that sampling frequency. The compensation is the
reciprocal of the actual density.

Riemannian volume changes the quantity being integrated: its density is
$J$. It does not eliminate the need to know how the samples were drawn.
If a Gibbs formula has been proved, the determinants cancel in the
importance weight. If only bounds on the density are available, retain
the density in the formula and use those bounds to control the error.
:::

:::{prf:proposition} Normalized importance sampling and variance
:label: prop-monte-carlo-riemannian-latent

Fix a spatial metric $g$ and a normalized coordinate sampling density $p$.
For $p>0$ almost everywhere on the support of $fJ$, define

$$
I[f]=\int fJ\,dx,\qquad
w(x)=\frac{f(x)J(x)}{p(x)},\qquad
\widehat I_N=\frac1N\sum_{i=1}^Nw(X_i).
$$

Assume $\int|f|J\,dx<\infty$.

1. If each $X_i$ has marginal density $p$, then
   $\mathbb E\widehat I_N=I[f]$. If the samples are independent and
   $\int f^2J^2/p\,dx<\infty$, then

   $$
   \mathbb E|\widehat I_N-I[f]|^2
   =\frac1N\left(\int\frac{f^2J^2}{p}\,dx-I[f]^2\right).
   $$

2. If the actual joint law $\pi_N$ has these marginals and satisfies the
   full-gradient LSI

   $$
   \operatorname{Ent}_{\pi_N}(F^2)
   \leq2C_*\int\sum_i
       (|\nabla_{x_i}F|^2+|\nabla_{v_i}F|^2)\,d\pi_N,
   $$

   then for a fixed $L$-Lipschitz position weight $w$,

   $$
   \operatorname{Var}_{\pi_N}(\widehat I_N)\leq C_*L^2/N.
   $$

   The sufficient joint-law criteria for a constant $C_*$ uniform in $N$
   are proved in {prf:ref}`cor-n-uniform-lsi`. For a law mixing
   discrete alive-status strata, the status entropy and variance terms
   must also be included.

3. More generally, let $\pi_N^x$ be the joint position law. For bounded
   $|w|\leq B$ and
   $H_N=D_{\mathrm{KL}}(\pi_N^x\|p^{\otimes N})<\infty$,

   $$
   \operatorname{Var}_{\pi_N}(\widehat I_N)
   \leq\mathbb E_{\pi_N}|\widehat I_N-I[f]|^2
   \leq \frac{4B^2}{N}\left(H_N+\frac12\log2\right).
   $$

   This is the established bound in
   {prf:ref}`thm-mixing-variance-corrected`. Its second-moment estimate
   also controls bias when the actual one-particle marginals differ from $p$.
   A joint phase-space entropy bound supplies this positional bound by
   relative-entropy contraction under marginalization, when its reference
   has position marginal $p^{\otimes N}$.

If an exact identity $p=Z^{-1}J e^{-U/T}$ has been proved, then
$w=Z f e^{U/T}$. Its use for an absolute integral requires the
normalizing constant $Z$ and the stated integrability conditions.
A QSD can be used when its actual normalized marginal density supplies
$p$.
:::

:::{prf:proof}
Marginal integration gives

$$
\mathbb E\,w(X_i)=\int \frac{fJ}{p}\,p\,dx=I[f].
$$

For independent samples, cross-covariances vanish, leaving
$\operatorname{Var}(w)/N$, with the displayed second moment.

The LSI implies the Poincaré inequality with constant $C_*$:
substitute $F=1+\varepsilon h$, expand both sides through
$\varepsilon^2$, and divide by $2\varepsilon^2$. For the empirical
weight,

$$
\sum_i|\nabla_{x_i}\widehat I_N|^2
=\frac1{N^2}\sum_i|\nabla w(X_i)|^2\leq\frac{L^2}{N},
\qquad
\nabla_{v_i}\widehat I_N=0.
$$

This proves the second assertion. The third is the bounded-observable
entropy estimate cited in its statement. The Gibbs simplification is
substitution into the exact normalized weight.
:::

:::{prf:remark} Dependence, bias, and fitted geometry
:label: rem-geometry-sampling-dependence

For any square-integrable sequence, including correlated episodes,

$$
\operatorname{Var}\!\left(\frac1N\sum_iw(X_i)\right)
=\frac1{N^2}\sum_{i,j}\operatorname{Cov}(w(X_i),w(X_j)).
$$

A temporal rate therefore needs a covariance or mixing estimate for that
sequence. If $p_i$ is the actual $i$th marginal, the bias of the fixed
reference weight is $N^{-1}\sum_i\int w(p_i-p)\,dx$.
For Lipschitz $w$, it is at most
$L N^{-1}\sum_iW_1(p_i,p)$, using the finite-particle estimates of
{doc}`/source/2_fractal_gas/convergence_program/13_quantitative_error_bounds`.

If a fitted metric $\widehat g_N$ is computed from the same sample,
$\widehat I_N$ becomes a function of all coordinates through the metric.
The Poincaré bound uses the complete gradient of that function, including
those additional derivatives. Alternatively, a deterministic reference
metric can be compared pathwise: if
$|\widehat J_N/J-1|\leq a_N$ on the sample locations, then

$$
\left|
 \frac1N\sum_i\frac{f(X_i)\widehat J_N(X_i)}{p(X_i)}
 -\frac1N\sum_iw(X_i)
\right|
\leq\frac{a_N}{N}\sum_i|w(X_i)|.
$$

Metric error bounds supply $a_N$ through
{prf:ref}`lem-geometry-volume-stability`. This comparison retains
the dependence between fitted geometry and samples.

The exponent $N^{-1/2}$ in a root-mean-square bound has no explicit
dimension dependence. Its second moment, Lipschitz constant, and LSI
constant can depend on dimension; each must be bounded for a uniform
claim.
:::


(sec-hypocoercivity)=
## Hypocoercivity in Anisotropic Geometry

:::{div} feynman-prose
Noise acts directly on velocity. Position feels its effect through
$dx=v\,dt$. A useful distance between two trajectories therefore includes
a cross term between their position and velocity differences. That term
records the transfer between the two parts of the motion.

The covariance bounds tell us how large the noise terms can be. A decay
rate also needs restoring forces and control of the other dynamics. We can
see every contribution by coupling two copies with the same noise. For a
quadratic confining potential, the calculation can be completed explicitly.
For nonconvex confinement, the established entropy proof supplies another
complete route.
:::

:::{prf:definition} Hypocoercive quadratic norm
:label: def-hypocoercive-norm-latent

Use nondimensional position and velocity coordinates. For $\lambda_v>0$
and $|b|<2\sqrt{\lambda_v}$, define

$$
\|(\delta x,\delta v)\|_{\mathrm{hyp}}^2
=|\delta x|^2+\lambda_v|\delta v|^2
 +b\,\delta x\cdot\delta v.
$$

Its matrix and eigenvalues are

$$
Q_0=
\begin{pmatrix}1&b/2\\b/2&\lambda_v\end{pmatrix},\qquad
q_\pm=\frac{1+\lambda_v
 \pm\sqrt{(1-\lambda_v)^2+b^2}}2.
$$

Thus $0<q_-\leq q_+$ and

$$
q_-(|\delta x|^2+|\delta v|^2)
\leq\|(\delta x,\delta v)\|_{\mathrm{hyp}}^2
\leq q_+(|\delta x|^2+|\delta v|^2).
$$

For swarms with matched labels on a common state space, use
$N^{-1}\sum_i\|(\delta x_i,\delta v_i)\|_{\mathrm{hyp}}^2$.
A permutation-invariant transportation distance additionally minimizes
over the declared matching or coupling.
:::

:::{prf:theorem} Quantified contraction for an anisotropic coupling
:label: thm-hypocoercive-anisotropic

Let two copies of a conservative continuous diffusion-jump process admit
a coupling on a common Euclidean state space. Between coupled jumps, use
the same Brownian motion:

$$
dS=b(S)\,dt+B(S)\,dW,\qquad
dS'=b(S')\,dt+B(S')\,dW.
$$

For a constant positive definite matrix $Q$, put
$\Delta=S-S'$ and $\mathcal H(S,S')=\Delta^TQ\Delta$.
Let $\mathcal J_c$ denote the jump part of the coupled generator.
Suppose the following bounds hold:

$$
2\Delta^TQ[b(S)-b(S')]\leq-r_b\mathcal H,
$$

$$
\operatorname{tr}\left((B(S)-B(S'))^TQ(B(S)-B(S'))\right)
\leq r_\sigma\mathcal H,
\qquad
\mathcal J_c\mathcal H\leq r_J\mathcal H.
$$

If $r=r_b-r_\sigma-r_J>0$, and localization is justified by
nonexplosion and the required moment bounds, then

$$
\mathbb E\mathcal H(S_t,S'_t)
\leq e^{-rt}\mathbb E\mathcal H(S_0,S'_0).
$$

Consequently the squared transportation cost induced by $Q$ contracts
by $e^{-rt}$. The conclusion is uniform in $N$ when the displayed
constants and the norm comparisons to the normalized swarm distance
are uniform.

Application to an actual swarm requires the actual coupled force,
alignment, noise, and jump terms in these inequalities. Survival
conditioning has its own normalized evolution and must be included
before applying a contraction estimate.
:::

:::{prf:proof}
Itô's formula gives the continuous part of the coupled generator:

$$
\mathcal L_c\mathcal H
=2\Delta^TQ[b(S)-b(S')]
+\operatorname{tr}\left((B(S)-B(S'))^TQ(B(S)-B(S'))\right).
$$

There is no state-independent noise trace in this difference equation:
the coupled covariance is the square of the difference of the noise
matrices. Add $\mathcal J_c\mathcal H$ to obtain

$$
(\mathcal L_c+\mathcal J_c)\mathcal H\leq-r\mathcal H.
$$

Apply Dynkin's formula to $e^{rt}\mathcal H$ up to localizing stopping
times. The assumed moment bounds justify removal of the stopping times,
yielding the expectation estimate. Start from any coupling of the initial
laws and minimize its initial cost to obtain the transportation bound.
:::

:::{prf:proposition} Explicit confining example and a multiplicative-noise margin
:label: prop-geometry-harmonic-contraction

For the kinetic model in nondimensional coordinates

$$
dx=v\,dt,\qquad
dv=(-a x-\gamma v)\,dt+\Sigma(x,v)\,dW,
\qquad a,\gamma>0,
$$

define

$$
Q=
\begin{pmatrix}p&q\\q&r_0\end{pmatrix}\otimes I_d,\quad
q=\frac1{2a},\quad
r_0=\frac{a+1}{2a\gamma},\quad
p=\frac{\gamma}{2a}+\frac{a+1}{2\gamma}.
$$

For constant $\Sigma$, synchronous coupling gives

$$
\mathbb E\mathcal H_t
\leq e^{-t/\lambda_{\max}(Q)}\mathbb E\mathcal H_0.
$$

If instead
$\|\Sigma(z)-\Sigma(z')\|_F\leq L_\Sigma|z-z'|$
and $r_0L_\Sigma^2<1$, the rate is at least

$$
\frac{1-r_0L_\Sigma^2}{\lambda_{\max}(Q)}.
$$

The same rate holds for the normalized sum of independent copies. The
general coupling theorem quantifies additional interaction and jump
contributions.
:::

:::{prf:proof}
For
$A=\begin{pmatrix}0&1\\-a&-\gamma\end{pmatrix}$,
direct multiplication gives

$$
A^TQ_0+Q_0A=-I_2,\qquad
\det Q_0=\frac{\gamma^2+(a+1)^2}{4a\gamma^2}>0,
$$

where $Q_0$ is the displayed $2\times2$ block. Since $p>0$,
$Q_0$ is positive definite. The deterministic difference therefore
satisfies

$$
\frac{d}{dt}\mathcal H
=-|\delta x|^2-|\delta v|^2
\leq-\frac{\mathcal H}{\lambda_{\max}(Q)}.
$$

For constant noise its difference under synchronous coupling vanishes.
For variable noise, its contribution is
$r_0\|\Sigma(z)-\Sigma(z')\|_F^2$, at most
$r_0L_\Sigma^2(|\delta x|^2+|\delta v|^2)$.
This gives the second rate. Dividing $Q_0$ by $p$ produces the
normalization of {prf:ref}`def-hypocoercive-norm-latent`, with
$b=2q/p$ and $\lambda_v=r_0/p$.
:::

### Entropy convergence under the established joint-law criteria

:::{prf:proposition} The analytical entropy route
:label: prop-geometry-entropy-route

Let the particular continuous joint law $\pi_N$ satisfy the full-gradient
LSI of {prf:ref}`cor-n-uniform-lsi` with constant $C_*$. For a
density $h$ relative to $\pi_N$, write

$$
H_{\pi_N}(h)=\int h\log h\,d\pi_N,\qquad
I(h)=\int h|\nabla\log h|^2\,d\pi_N,
$$

and for a positive definite matrix $G_N\preceq g_+I$ let

$$
I_{G_N}(h)=\int h(\nabla\log h)^TG_N\nabla\log h\,d\pi_N,\qquad
\Phi_{G_N}=H_{\pi_N}+I_{G_N}.
$$

If the complete normalized evolution satisfies
$\dot\Phi_{G_N}(h_t)\leq-\delta I(h_t)$ with $\delta>0$, then

$$
\Phi_{G_N}(h_t)
\leq
 \exp\left[-\frac{\delta t}{C_*/2+g_+}\right]\Phi_{G_N}(h_0).
$$

For the constant-diffusion kinetic Gibbs model with bounded potential
Hessian, {prf:ref}`thm-villani-hypocoercivity` proves the derivative
bound, with its explicit matrix $G$ and constant $\eta$, giving

$$
r_{\mathrm{kin}}=\frac{\eta}{C_*/2+3\eta}>0.
$$

In particular, {prf:ref}`cor-n-particle-hypocoercive` supplies a
uniform rate for its nonconvex product potentials. For the complete
geometric evolution, the derivative and perturbation margins are those
of {prf:ref}`prop-gg-entropy-fisher-gap` and
{prf:ref}`cor-gg-joint-thresholds`.
:::

:::{prf:proof}
Apply the LSI to $\sqrt h$ to obtain
$H_{\pi_N}(h)\leq(C_*/2)I(h)$. Since
$I_{G_N}(h)\leq g_+I(h)$,

$$
\Phi_{G_N}(h)\leq(C_*/2+g_+)I(h).
$$

Substitute this inequality into the assumed derivative bound and apply
Grönwall. The referenced kinetic theorem establishes the derivative
estimate by its full commutator calculation; the nonconvex product
corollary supplies its uniform LSI and Hessian bounds. The geometric
version retains all coefficient, alignment, cloning, and survival terms.
:::

:::{div} feynman-prose
There are now two complete calculations to use. A coupling calculation
compares trajectories. An entropy calculation compares a probability
density with a specified target law. Each gives an explicit decay rate
when its own terms have been bounded.

The regularization parameter enters those bounds in competing ways.
Increasing $\epsilon_\Sigma$ decreases the noise covariance, and it
also decreases the inverse-square-root Lipschitz bound. One change may
help an error term while another slows exploration. The displayed
contraction or entropy margin decides the outcome for the model being
studied.
:::

(sec-emergent-geometry-summary)=
## Geometry, Sampling, and Convergence

The inverse diffusion shape supplies the metric used by the subsequent
spatial constructions. Matrix bounds give lengths and volumes with explicit
constants. Generator calculations identify stationary laws; joint-law
inequalities give sampling errors; dissipative estimates give dynamical
rates.

:::{prf:remark} Hypotheses carried into continuum and dynamical applications
:label: rem-emergent-geometry-convergence-hypotheses

Applications of this chapter retain the following data.

- **Metric identification.** Specify the smooth shifted field or the
  recorded clipped field, the finite-step noise prefactor, and the
  reconstruction between observed points. The covariance identity is
  {prf:ref}`prop-geometry-clipped-metric` for the corresponding
  implementation branch.
- **Regularity and geometry.** Use two-sided metric bounds and the
  derivatives required by the construction. Spatial $C^1$ regularity
  suffices for the stated first-order bounds; classical curvature uses
  $C^2$. A continuum manifold and faithful causal sampling use the
  independent geometric and sampling hypotheses of
  {prf:ref}`cor-continuum-consistency-conditional`.
- **Sampling law.** Exact Gibbs weights use a proved generator identity.
  A QSD uses its killed eigenmeasure equation and normalized marginal.
  Empirical variance uses independence, the stated joint LSI, or the
  bounded-observable relative-entropy estimate. Fitted geometry retains
  its dependence on the same samples.
- **Dynamics.** Covariance ellipticity, moment confinement, QSD
  convergence, and entropy dissipation are separate estimates.
  {prf:ref}`thm-hypocoercive-anisotropic` applies to the stated
  conservative coupling. Killed-kernel convergence is supplied by
  {doc}`/source/2_fractal_gas/convergence_program/06_convergence`;
  actual-law LSI and entropy rates use the precise criteria of
  Chapters 15 and 17.

The positive metric here is spatial. The Lorentzian slab construction,
causal reachability, and proper time are specified in
{doc}`/source/2_fractal_gas/3_fitness_manifold/02_scutoid_spacetime`.
They are not inferred from a covariance determinant.
:::

(sec-emergent-geometry-references)=
## References

The principal proofs used in this chapter are:

- {doc}`/source/2_fractal_gas/convergence_program/14_a_geometric_gas_c3_regularity`:
  sampled and expected fitness fields, derivative bounds, and implementation
  branches.
- {doc}`/source/2_fractal_gas/convergence_program/14_b_geometric_gas_cinf_regularity_full`:
  higher derivatives under its complete regularity hypotheses.
- {doc}`/source/2_fractal_gas/convergence_program/12_qsd_exchangeability_theory` and
  {doc}`/source/2_fractal_gas/convergence_program/13_quantitative_error_bounds`:
  joint-law fluctuations and finite-particle observable errors.
- {doc}`/source/2_fractal_gas/convergence_program/15_kl_convergence`:
  proved LSI criteria, nonconvex kinetic hypocoercivity, and the Doob
  invariant measure.
- {doc}`/source/2_fractal_gas/convergence_program/16_continuum_discharge` and
  {doc}`/source/2_fractal_gas/convergence_program/17_geometric_gas`:
  conditional continuum consistency and the complete geometric evolution.

The determinant, Gaussian marginal, matrix perturbation, and simplex
quadrature arguments are proved directly above.
