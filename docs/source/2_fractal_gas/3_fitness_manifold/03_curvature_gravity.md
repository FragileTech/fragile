(sec-curvature-from-holonomy)=
# Curvature of the Algorithmic Fitness Metric

**Prerequisites:** {doc}`01_emergent_geometry` and
{doc}`02_scutoid_spacetime`.

(sec-tldr-curvature)=
## TLDR

Start with the algorithm's recorded population, eligibility mask,
companion sources, and fitness configuration. Move one query through
that fixed conditioning data, recomputing the affected measurements,
statistics, and local weights. This defines the conditional fitness
field $V_i(z\mid\Xi)$ and its spatial derivatives.

The configured diffusion provider converts its Hessian $H$ into
$g=H+\epsilon_\Sigma I$ on the strict positive branch, or
$g=H_++\epsilon_\Sigma I$ on the clipped branch. Its normalized
covariance shape is $g^{-1}$. These are identified algorithmic
constructions; their curvature follows from differentiating them.

A smooth Hessian metric needs six Hessian and ten third-derivative
components in three dimensions. Direct contractions give Ricci, scalar,
and sectional curvature without evaluated fourth derivatives or a
materialized Riemann tensor. Mixed-sign clipping generally needs fourth
fitness derivatives through the first two derivatives of its metric.
Clipping thresholds require separate smoothness analysis.

Transport supplies a second measurement: consistent small-loop holonomy
recovers the same curvature with a quantified error. Spacetime transport,
Raychaudhuri, and geodesic focusing apply only after the additional
Lorentzian construction and their stated hypotheses. They impose no
evolution equation on the gas. The finite-step metric evolution and
mechanical balances of the executed updates are developed in
{doc}`04_field_equations`.

Polyhedral Gauss–Bonnet constrains curvature on a fixed surface;
subdividing a cell preserves the Euler characteristic.

(sec-introduction-curvature)=
## Introduction

:::{div} feynman-prose
Choose a walker at the stage where the algorithm constructs its adaptive
noise. Keep the recorded population and donor sources in view, and move
the walker's query position a little. Its reward and companion distance
change. The statistics used to normalize those measurements can change
too. Calculating all of those changes gives the derivatives of the
conditional fitness actually used by this provider.

The next step is a concrete matrix calculation. Apply the configured
Hessian shift or spectral clipping, and obtain the positive metric that
sets the shape of the noise. Its inverse tells us which directions the
walker explores easily. Spatial derivatives of that same metric give
the connection and curvature. We can evaluate these quantities on a
snapshot and compare two independent formulas before asking what they
do over time.

Only after this calculation do we carry an arrow around a loop. That
experiment measures the already identified curvature when the transport
rule is sufficiently accurate. Introducing a time coordinate and a
Lorentzian metric lets us study additional geometric identities, but
their hypotheses must be stated separately. To find how the gas changes
its own geometry, we return to its transition law and calculate the
metric increment produced by the full update.
:::

(sec-algorithmic-curvature-computation)=
## From Conditional Fitness to the Metric

:::{div} feynman-prose
Suppose every walker carries a little ruler, with squared length
$g_{ab}\,dx^a dx^b$. Its principal axes tell us which directions the
adaptive noise explores more easily. Now put identical rulers everywhere,
all stretched by the same amount along the same axes. There is anisotropy,
but there is no geometric curvature: a fixed change of coordinates makes
every ruler Euclidean.

This distinguishes two uses of the word curvature. The Hessian measures
how a scalar fitness changes near a point. Riemann curvature measures how
the rulers fit together across neighboring points. Even a positive,
large fitness Hessian gives a flat metric when it is constant. Nor does
every varying Hessian metric have nonzero Riemann curvature; a separable
potential provides another useful flat example.

Three dimensions need not make the local calculation expensive. For a
smooth Hessian metric, the cancellation in
{prf:ref}`lem-curvature-hessian-cancellation` leaves only six
independent Hessian entries and ten independent third derivatives. We can
compute the scalar curvature from those sixteen numbers without building
an 81-entry Riemann array. The important preliminary work is to specify
which fitness function those derivatives belong to. Moving a query while
keeping its sampled donor fixed and moving the whole population are
different experiments, and their derivatives answer different questions.
:::

:::{prf:definition} Conditional fitness field and derivative convention
:label: def-curvature-conditional-fitness-field

Fix a recorded population, its eligibility mask, the sampled companion
identities, their immutable source coordinates, the fitness parameters,
and a target row $i$. Denote these conditioning data by $\Xi$.
An identified conditional fitness provider specifies a scalar field
$V_i(z\mid\Xi)$ by replacing the target query by $z$ and executing its
stated measurement and fitness calculation. Its derivatives are spatial
derivatives with respect to $z$ on one smooth stratum of that calculation.
The stratum excludes changes of eligibility, donor identity, periodic
image, or an active nonsmooth floor.

For the conditional global-statistics construction, the target reward
and separation vary with $z$, and the population means and variances
containing those measurements are recomputed and differentiated. Other
rows and donor source coordinates remain fixed. Freezing the numerical
means and variances instead defines a different comparison field; it
must be identified as such. For a neighborhood-dependent construction,
the dependence of its weights and affected measurements is also part of
the derivative.

Differentiating the complete measurement stage with respect to a source
population coordinate follows all uses of that coordinate. Even with
companion indices fixed, moving row $i$ changes the separation of each
row that selects $i$ as its source. For example, on a noncoincident
Euclidean stratum, a row $j$ selecting $i$ has

$$
\nabla_{x_i}|x_j-x_i|=\frac{x_i-x_j}{|x_i-x_j|}.
$$

Those affected measurements also enter the differentiated population
statistics. This population derivative and the provider's query
derivative have different data dependencies: the former moves the
source coordinate wherever it is used; the latter evaluates a query
against the immutable source snapshot supplied to that operator.
Holding companion indices fixed specifies the source identities, while
holding source coordinates fixed additionally specifies which inputs
are constant. Two such calculations can agree in value at a recorded
point and have different gradients and Hessians.

In the fixed affine coordinates of the algorithm, define the smooth
shifted construction by

$$
\Phi_i(z\mid\Xi)=V_i(z\mid\Xi)+\frac{\epsilon_\Sigma}{2}|z|^2,
\qquad g_{ab}=\partial_a\partial_b\Phi_i.
$$

Its domain of classical Hessian curvature is an open region where
$\Phi_i\in C^4$ and $g\succ0$. The configured shift or clipping is the
algorithm's noise rule; the derivative convention follows the inputs
of the operator that executes it. If coordinates have units $L$ and fitness
has units $F$, then $[g]=F/L^2$, $[\epsilon_\Sigma]=F/L^2$,
$[C]=F/L^3$, and scalar and sectional curvature have units $F^{-1}$.
All are dimensionless for dimensionless algorithm coordinates and
fitness. A nonlinear coordinate change transports $g$ as a tensor;
taking an ordinary Hessian again in the new coordinates generally
constructs a different metric.
:::

(sec-tessellation-to-curvature)=
## From a Metric to a Connection

:::{div} feynman-prose
Once the provider has supplied a smooth positive metric field, we can
compare arrows at neighboring query positions. Requiring that comparison
to preserve the metric and have zero torsion determines the Levi-Civita
connection. Its coefficients follow from the spatial derivatives of the
same field; a collection of metric matrices without a spatial
reconstruction does not supply those derivatives.

The first calculation uses only the algorithm's spatial coordinates.
A spacetime comparison introduces additional coefficients under
{prf:ref}`assump-curvature-geometric-setting`.
:::

:::{prf:definition} Levi-Civita connection
:label: def-affine-connection

For a $C^1$ nondegenerate metric $q$, the Levi-Civita connection has
coefficients

$$
\Gamma^a_{bc}[q]
=\frac12q^{ae}
 (\partial_bq_{ec}+\partial_cq_{eb}-\partial_eq_{bc}).
$$

It is metric compatible and torsion free. Here $q=g_t$ gives the
intrinsic spatial connection. A spacetime application uses the separately
specified $q=G$ in {prf:ref}`assump-curvature-geometric-setting`.
For the smooth shifted Hessian field, spatial coefficients use
third derivatives of the fitness potential.

Metric compatibility and zero torsion determine these coefficients
uniquely: add
$\partial_bq_{ac}=\Gamma_{abc}+\Gamma_{cba}$ and
$\partial_cq_{ab}=\Gamma_{acb}+\Gamma_{bca}$, subtract
$\partial_aq_{bc}$, and use symmetry in the two lower connection
indices to solve for $\Gamma$.
:::

(sec-riemann-tensor)=
## Curvature Tensors and Hessian Metrics

:::{div} feynman-prose
A varying metric produces a connection, and a varying connection produces
curvature. For a general metric this uses second derivatives of the metric.

A Hessian metric has extra symmetry. All its entries are second derivatives
of the same scalar function. When we antisymmetrize to form curvature, the
fourth derivatives of that scalar cancel. What remains is a product of
third derivatives. This gives a useful direct bound from the fitness
regularity estimates.
:::

:::{prf:definition} Riemann curvature tensor
:label: def-riemann-tensor

For a $C^2$ metric and its Levi-Civita connection, use the convention

$$
R^a{}_{bcd}
=\partial_c\Gamma^a_{bd}-\partial_d\Gamma^a_{bc}
 +\Gamma^a_{ce}\Gamma^e_{bd}
 -\Gamma^a_{de}\Gamma^e_{bc}.
$$

Thus $R(X,Y)V$ has components $R^a{}_{bcd}V^bX^cY^d$.
The metric and connection are both spatial or both spacetime according
to the application.
:::

:::{prf:lemma} Cancellation of fourth derivatives in a Hessian metric
:label: lem-curvature-hessian-cancellation

Let $\Phi\in C^4$ in the fixed Euclidean coordinates and let
$g_{ab}=\partial_a\partial_b\Phi$ be positive definite. Write
$C_{abc}=\partial_a\partial_b\partial_c\Phi$. Then

$$
\Gamma^a_{bc}=\frac12g^{ae}C_{ebc},\qquad
R_{abcd}
=\frac14g^{pq}
 \left(C_{adp}C_{bcq}-C_{acp}C_{bdq}\right).
$$

In particular, the smooth construction
$\Phi=V_{\mathrm{fit}}+\epsilon_\Sigma|x|^2/2$ has this formula.

If $g\succeq mI$ and
$|C(X,Y,\cdot)|\leq K_3|X|\,|Y|$ in Euclidean norm, then the
absolute value of every sectional curvature is at most
$K_3^2/(2m^3)$. Classical use of this identity is justified by the
stated $C^4$ regularity. A clipped field has this Hessian identity only
where it actually agrees with the smooth Hessian metric.
:::

:::{prf:proof}
Symmetry of $C$ reduces the connection formula to
$\Gamma^a_{bc}=g^{ae}C_{ebc}/2$. Differentiating the inverse gives

$$
\partial_cg^{ae}=-g^{ap}C_{pqc}g^{qe}.
$$

Substitute into the curvature definition. The two terms
$\tfrac12g^{ae}\partial_c C_{ebd}$ and
$-\tfrac12g^{ae}\partial_d C_{ebc}$ cancel because the fourth
derivatives are symmetric. The remaining derivative-of-inverse terms
combine with the two connection products. Lowering the first index gives

$$
R_{abcd}
=\tfrac14g^{pq}C_{adp}C_{bcq}
 -\tfrac14g^{pq}C_{acp}C_{bdq}.
$$

For $g$-orthonormal $X,Y$, Euclidean lengths satisfy
$|X|,|Y|\leq m^{-1/2}$. Each cubic tensor contracted with two of
these vectors has Euclidean norm at most $K_3/m$. Each of the two
inverse-metric products is therefore at most $K_3^2/m^3$.
Their coefficients sum to $1/2$, proving the sectional bound.
:::

:::{prf:definition} Ricci tensor and scalar curvature
:label: def-ricci-tensor-scalar

For the specified metric $q$,

$$
\operatorname{Ric}_{bd}=R^a{}_{bad},\qquad
R=q^{bd}\operatorname{Ric}_{bd}.
$$

Geodesic focusing in the timelike direction $u$ uses
$\operatorname{Ric}(u,u)$. Scalar curvature is a trace over directions;
its sign alone does not determine that contraction or the sign of every
sectional curvature.
:::

(sec-curvature-direct-contractions)=
### Direct contractions and three-dimensional computation

:::{prf:lemma} Direct Hessian Ricci and scalar contractions
:label: lem-curvature-whitened-contractions

Under {prf:ref}`lem-curvature-hessian-cancellation`, fix a point and let
$E$ be an invertible matrix satisfying $E^{\mathsf T}gE=I$. Its columns
form a $g$-orthonormal frame. Set

$$
\widehat C_{ijk}=C_{abc}E^a{}_iE^b{}_jE^c{}_k,
\qquad (S_i)_{jk}=\widehat C_{ijk},
\qquad t_k=\sum_i\widehat C_{iik}.
$$

Then the Ricci tensor in that frame and its scalar trace are

$$
\boxed{\widehat{\operatorname{Ric}}_{ij}
=\frac14\left(\langle S_i,S_j\rangle_F
-\sum_k t_k\widehat C_{ijk}\right)},
\qquad
\boxed{R=\frac14\left(\|\widehat C\|_F^2-|t|^2\right)}.
$$

The coordinate Ricci tensor is
$\operatorname{Ric}=E^{-\mathsf T}\widehat{\operatorname{Ric}}E^{-1}$.
For linearly independent coordinate vectors $u,v$, sectional curvature is

$$
K(u,v)=
\frac{
 C(u,v,\cdot)^{\mathsf T}g^{-1}C(u,v,\cdot)
-C(u,u,\cdot)^{\mathsf T}g^{-1}C(v,v,\cdot)}
{4\bigl(g(u,u)g(v,v)-g(u,v)^2\bigr)}.
$$

These expressions require no evaluated fourth derivatives and no
materialized fourth-order curvature tensor. They retain the $C^4$
hypothesis used to justify the classical cancellation identity.
:::

:::{prf:proof}
Because $EE^{\mathsf T}=g^{-1}$, changing all four free indices in
{prf:ref}`lem-curvature-hessian-cancellation` to the orthonormal frame
gives

$$
\widehat R_{abcd}
=\frac14\sum_p
 (\widehat C_{adp}\widehat C_{bcp}
 -\widehat C_{acp}\widehat C_{bdp}).
$$

The Ricci convention of {prf:ref}`def-ricci-tensor-scalar` contracts the
first and third indices. Thus

$$
\widehat{\operatorname{Ric}}_{bd}
=\frac14\sum_{a,p}
 (\widehat C_{adp}\widehat C_{bap}
 -\widehat C_{aap}\widehat C_{bdp}).
$$

Symmetry of the cubic tensor identifies the first term as
$\langle S_d,S_b\rangle_F$ and the second as
$\sum_p t_p\widehat C_{bdp}$. Taking the trace sums the first term to
$\|\widehat C\|_F^2$ and the second to $\sum_p t_p^2$.
Since $\widehat{\operatorname{Ric}}=E^{\mathsf T}\operatorname{Ric}E$,
inverting that change of basis gives the coordinate formula. Finally,
contract $R_{abcd}$ with $u^av^bu^cv^d$ and divide by the squared
$g$-area of the parallelogram. Positivity of $g$ and independence of
$u,v$ make this denominator positive.
:::

:::{prf:corollary} Packed three-dimensional calculation
:label: cor-curvature-packed-three-dimensional

In $d$ dimensions, fully symmetric derivatives of order $k$ have
$\binom{d+k-1}{k}$ independent components. For $d=3$, store the Hessian
at indices $a\leq b$ and the cubic tensor at indices $a\leq b\leq c$.
They require six and ten components respectively. The complete scalar
jet through order three contains $1+3+6+10=20$ coefficients when stored
as derivatives rather than Taylor coefficients.

Given an orthonormal frame, scalar contraction costs $O(d^3)$ arithmetic
and cubic storage; the direct Ricci contraction costs $O(d^4)$.
Whitening the dense cubic by three successive index contractions costs
$O(d^4)$, and a dense metric factorization costs $O(d^3)$. These counts
exclude evaluation of the fitness derivatives. A full Riemann output
additionally has $d^4$ entries and is unnecessary for scalar or Ricci
queries.
:::

:::{prf:proof}
A symmetric component corresponds to a multiset of $k$ indices selected
from $d$ possibilities, giving the stated binomial count. In three
dimensions the cubic entries are
$111,112,113,122,123,133,222,223,233,333$.
For each transformed index, multiplication by $E$ sums $d$ terms for
each of $d^3$ output entries. The scalar expression then sums cubic
entries and the $d$ traces. Each of $d^2$ Ricci entries uses an inner
product with $d^2$ terms. Packed storage must account for permutation
multiplicities in full contractions: $\|\widehat C\|_F^2$ counts an
all-equal entry once, a two-equal entry three times, and an all-distinct
entry six times. These are the numbers of distinct permutations of
their indices.
:::

:::{div} feynman-prose
Whitening means choosing units and axes that make the ruler the identity
at the point where we are calculating. It does not flatten a neighborhood.
The third derivatives still describe how the neighboring rulers change,
and their contractions retain the curvature.

Two checks are particularly revealing. If the fitness is quadratic, the
cubic tensor vanishes and all geometric curvature vanishes. If the
potential is separable, $\Phi(x)=\sum_a\phi_a(x^a)$ with positive
$\phi_a''$, the metric varies but is still flat: replace each coordinate
by $y^a=\int^{x^a}\sqrt{\phi_a''(s)}\,ds$. The line element becomes
$\sum_a(dy^a)^2$. Both the direct contraction and an independent general
metric calculation must reproduce these answers. A nonzero scalar
curvature cannot be manufactured merely by measuring a large Hessian.
:::

:::{prf:lemma} Constant-work global-statistics update at a fixed query
:label: lem-curvature-query-moment-cache

Suppose precisely one of $n\geq2$ eligible scalar measurements is a
variable $x(z)$. Let $\bar x_-$ and $M_{2,-}$ be the mean and sum of
squared centered deviations of the other $n-1$ measurements. The
population mean and population variance, with denominator $n$, are

$$
\bar x(z)=\bar x_-+\frac{x(z)-\bar x_-}{n},
\qquad
\sigma^2(z)=\frac{M_{2,-}}{n}
+\frac{n-1}{n^2}\bigl(x(z)-\bar x_-\bigr)^2.
$$

After preparing these summaries, every derivative of these statistics
through the requested order follows by differentiating these formulas;
no new population scan is needed per query. For $n=1$, the mean is
$x(z)$ and the population variance is zero.
:::

:::{prf:proof}
Write $\delta=x(z)-\bar x_-$. The other measurements have deviations
from the new mean equal to their old centered deviations minus
$\delta/n$. Their centered deviations sum to zero, so their new
squared deviations sum to $M_{2,-}+(n-1)\delta^2/n^2$. The target
deviation is $(n-1)\delta/n$. Add its square and divide by $n$ to obtain
the variance formula. The mean follows by adding the target to the
other measurements' sum. Differentiation is exact on the stated smooth
stratum. The formulas do not apply when changing the query also changes
other measurements or query-dependent neighborhood weights.
:::

:::{prf:remark} Clipped metrics require a different derivative calculation
:label: rem-curvature-clipped-computation

For the actual clipped construction, let
$H=\nabla^2 V_i$ and $g_+=\epsilon_\Sigma I+f(H)$, where
$f(\lambda)=\max(\lambda,0)$. Assume $V_i\in C^4$ and the spectrum of
$H$ remains separated from zero on an open neighborhood. Matrix spectral
calculus then gives

$$
\partial_a g_+=Df(H)[\partial_aH],
\qquad
\partial_a\partial_b g_+
=Df(H)[\partial_a\partial_bH]
+D^2f(H)[\partial_aH,\partial_bH].
$$

In an orthonormal eigenbasis of $H$, the first derivative has entries
$f[\lambda_i,\lambda_j](\partial_aH)_{ij}$. The symmetric second
derivative uses

$$
\bigl(D^2f(H)[A,B]\bigr)_{ij}
=\sum_k f[\lambda_i,\lambda_k,\lambda_j]
 (A_{ik}B_{kj}+B_{ik}A_{kj}).
$$

Here brackets denote divided differences, with their continuous
derivative limits at repeated eigenvalues of the same sign. Repeated
nonzero eigenvalues therefore introduce no mathematical singularity.
To verify the formulas, take disjoint complex contours around the
positive and negative spectra, and extend $f$ analytically as $z$ and
$0$ on their respective interiors. In the contour representation
$f(H)=(2\pi\mathrm i)^{-1}\oint f(z)(zI-H)^{-1}\,dz$, write
$A_z=(zI-H)^{-1}$. Its derivatives are
$DA_z[B]=A_zBA_z$ and
$D^2A_z[B,D]=A_zBA_zDA_z+A_zDA_zBA_z$.
Diagonalizing $H$ inside these integrals gives residues with two and
three scalar resolvents, namely the first and second divided
differences displayed above. The spectral gap lets one use the same
contours throughout a neighborhood. The spatial formulas then follow
by the chain rule with $H(z)=\nabla^2V_i(z)$.

If every eigenvalue is positive, $g_+$ agrees with the smooth shifted
Hessian metric and {prf:ref}`lem-curvature-whitened-contractions`
applies. If every eigenvalue is negative, $g_+=\epsilon_\Sigma I$ on
the neighborhood and its curvature is zero. In a mixed-sign region,
$g_+$ is generally not a Hessian metric. Compute its Levi-Civita
curvature from $g_+,\partial g_+,\partial^2 g_+$; the latter generally
needs fourth fitness derivatives. Substituting clipped eigenvalues in
the Hessian curvature identity does not perform this calculation.

At a zero eigenvalue, spectral clipping does not guarantee a $C^2$
metric. Classical curvature is unavailable without a separate proof of
smoothness at that point. A finite-difference estimate across the
threshold is a resolution-dependent diagnostic, not a proof of a
classical curvature value. A numerical tolerance used to exclude such
points must be reported with the result.
:::

:::{prf:remark} Curvature of a sampled field and of an averaged covariance
:label: rem-curvature-sampling-convention

The conditional field above is indexed by its sampled sources and
population snapshot. Its curvature is not automatically the curvature
of a population-averaged metric. In general,

$$
\mathbb E\bigl[(\epsilon_\Sigma I+H_+)^{-1}\bigr]
\ne\bigl(\epsilon_\Sigma I+(\mathbb EH)_+\bigr)^{-1},
\qquad
\mathbb E[R(g)]\ne R(\mathbb E[g]).
$$

If a provider instead constructs an averaged covariance, its own spatial
derivatives define its metric and curvature. Query-dependent sampling
probabilities must also be differentiated when differentiating that
expectation. Likewise the covariance shape $g^{-1}$, its noise factor,
and a physical increment covariance $\alpha g^{-1}$ are separate
quantities. For constant $\alpha>0$, interpreting the latter's inverse
as the metric gives $q=g/\alpha$ and $R(q)=\alpha R(g)$; a varying
$\alpha$ also contributes metric derivatives. A reported curvature must
therefore identify its field, derivative convention, and normalization.
:::

(sec-algorithmic-curvature-rust)=
### Rust representation and execution

:::{prf:definition} Computational representation of fitness and metric jets
:label: def-curvature-rust-representation

The Rust module `algorithmic_gas::physics::jet` represents multivariate
Taylor polynomials in a shared `JetSpace::new(dimension, order)`.
An internal coefficient indexed by the multi-index $\alpha$ equals
$\partial^\alpha V/\alpha!$. Conversion through
`physics::geometry::FitnessJet::from_jet` produces **ordinary
derivatives**: its `hessian`, `third`, and optional `fourth` arrays use
lexicographic order over nondecreasing axis tuples. `value` and
`gradient` complete the scalar jet. Thus a repeated-index derivative
must not be copied directly from a normalized Taylor coefficient.

The geometry entry points are:

| Function | Mathematical input and result |
|---|---|
| `hessian_curvature(&jet, epsilon, full_riemann)` | Strict positive shifted Hessian; uses third derivatives and the direct contractions above |
| `fitness_curvature(&jet, epsilon, policy, threshold, full_riemann)` | Strict or clipped construction; the absolute Hessian spectral threshold determines where clipped classical curvature is unavailable |
| `metric_curvature(&metric_jet, full_riemann)` | An independently supplied positive metric and its first two spatial derivatives; evaluates the general Levi-Civita formula |
| `ExecutionContext::smooth_curvature_batch(&jets, epsilon, policy, threshold)` | Batched smooth Hessian or constant clipped metrics; whitening and curvature contractions use the explicitly selected compute backend |

`MetricJet` uses row-major arrays with layouts $g_{ij}$,
$\partial_k g_{ij}$, and $\partial_k\partial_l g_{ij}$: derivative
indices precede matrix indices. These entries must be derivatives of
the same $C^2$ metric field. A `CurvatureBatch` returns its `spectrum`,
coordinate `ricci`, `scalar`, `einstein`, and eigenframe `sectional`
curvatures, with optional covariant row-major `riemann`. The `einstein`
field is the geometric contraction $\operatorname{Ric}-Rg/2$; computing
it imposes no field equation or identification with stress.

The sectional array enumerates pairs $i<j$ of orthonormal metric
eigenvectors. Within a repeated-eigenvalue eigenspace the eigenbasis is
not unique, so these particular sectional planes are frame-dependent.
The scalar and the coordinate Ricci tensor do not depend on that choice.
The `full_riemann=false` route avoids allocating a Riemann output.
The third-derivative Hessian route additionally avoids fourth fitness
derivatives; that saving does not apply to general mixed-sign clipping.

The batch method performs the small eigendecompositions on the host,
then uses backend tensor products for whitening, Ricci, scalar, and
sectional contractions, with one combined result readback. It retains
$O(Nd^3)$ working storage and $O(Nd^4)$ contraction work, subject to
the execution context's allocation limits. All queries in a batch
must share a dimension and precision. Mixed-sign clipping and clipping
thresholds are rejected by this method; there is no automatic backend
fallback. Such mixed-sign queries require the separately invoked
general metric calculation.
:::

:::{div} feynman-prose
Here is a small calculation whose answer can be checked on paper. Take
$V(x,y,z)=(x^2+y^2+z^2)/2+0.1xyz$ and evaluate at the origin, with
$\epsilon_\Sigma=0.2$. The metric there is $1.2I$. The only nonzero
cubic derivatives are the six permutations of $V_{xyz}=0.1$.
Whitening divides each by $1.2^{3/2}$, while the trace vector $t$ is
zero. Consequently $R=6(0.1)^2/(4(1.2)^3)$. This example tests a
nonzero curvature, all three coordinates, and the multiplicity of a
packed mixed derivative at once.
:::

:::{div} feynman-added
```rust
use algorithmic_gas::physics::{
    geometry::{FitnessJet, hessian_curvature},
    jet::JetSpace,
};

fn check_three_dimensional_curvature() -> algorithmic_gas::Result<()> {
    let space = JetSpace::new(3, 3)?;
    let x = space.variable(0.0_f64, 0)?;
    let y = space.variable(0.0_f64, 1)?;
    let z = space.variable(0.0_f64, 2)?;
    let quadratic = x.pow(2.0).add(&y.pow(2.0)).add(&z.pow(2.0));
    let potential = quadratic.scale(0.5).add(&x.mul(&y).mul(&z).scale(0.1));
    let jet = FitnessJet::from_jet(&potential)?;
    let curvature = hessian_curvature(&jet, 0.2, false)?;
    let expected = 6.0 * 0.1_f64.powi(2) / (4.0 * 1.2_f64.powi(3));
    assert!((curvature.scalar - expected).abs() < 1e-12);
    Ok(())
}
```
:::

:::{note}
:class: feynman-added

The benchmark runner selects the fitness metric provider through
`RunConfig.physics_metric`. The provider supports smooth global and
local standardizers with logistic positive maps. Its global cache uses
{prf:ref}`lem-curvature-query-moment-cache`. For local normalization,
`pipeline_from_measurement_jets` and `local_log_weights` retain the
derivatives of both measurements and neighborhood weights, using an
$O(N)$ calculation per query. Supported smooth Euclidean and phase-space
kernels retain the recorded eligibility and donor sources; nonsmooth
coincident-neighbor distances or periodic seams require their own
identified treatment.

When a population-coordinate calculation changes several measurement
rows, supply every affected row as a jet to
`pipeline_from_measurement_jets`. The test
`coupled_population_derivative_matches_executed_fitness_with_fixed_companions`
checks the resulting gradient and Hessian against finite differences of
the production `FitnessPipeline::evaluate`, with two other rows using
the moved walker as their companion. It also checks a matching
fitness value and a different gradient for the immutable-source query
calculation. This tests the dependency distinction directly.

The provider evaluates conditional jets and its curvature diagnostics
on the host in `f32` or `f64`; applying the resulting diffusion factors
uses the selected noise backend. The separate
`smooth_curvature_batch` entry point runs its tensor contractions on
the selected CPU, WebGPU, or CUDA backend, retaining host
eigendecompositions. CPU supports `f32` and `f64`; WebGPU uses `f32`,
and CUDA precision requires the corresponding device capability. A
backend build does not establish numerical agreement or a speedup on
GPU hardware; those require an actual device run.

At the thermostat stage, the provider records `fitness_hessian`,
`fitness_metric`, `metric_inverse_sqrt`, and, when requested,
`fitness_ricci` and `fitness_scalar_curvature`. Third and, where needed,
fourth derivatives are evaluated only when curvature recording is
enabled; unrecorded steps retain the second-order metric calculation.
Coverage masks distinguish
unavailable curvature from a computed zero. In particular, a clipping
threshold can make classical curvature unavailable while the positive
clipped diffusion metric remains usable. Conditional target-slot and
revival-field records identify which reconstructed field was evaluated.
:::

(sec-curvature-spacetime-setting)=
## Additional Spacetime Geometry

:::{div} feynman-prose
The conditional fitness calculation supplies a spatial metric. To use
that family of metrics in spacetime transport, we must also specify how
time enters the line element. The following construction states that
extra choice explicitly. Its mixed connection coefficients then follow
by substitution; they are not inferred from a spatial curvature map.
:::

:::{prf:assumption} Spatial and spacetime geometry used in this chapter
:label: assump-curvature-geometric-setting

Let $g_t$ be an identified positive definite spatial metric on a coordinate
region $\mathcal Z$, with the regularity needed by each result below.
For spacetime statements choose

$$
M=(t_0,t_1)\times\mathcal Z,\qquad
G=-c^2dt^2+g_t,\qquad c>0.
$$

This is the specified Lorentzian model, of dimension $d+1$ when
$\dim\mathcal Z=d$. The metric and its reconstruction are those in
{prf:ref}`rem-volume-element-regime`. For a clipped recorded metric,
classical curvature requires the additional smoothness specified in
{prf:ref}`rem-geometry-regularity-order`.

The continuum cone and normalized-volume hypotheses of
{prf:ref}`lem-causal-order-conformal-class` and
{prf:ref}`cor-order-volume-fix-conformal` can identify an already
recovered Lorentzian metric. Applying that identification to the recorded
CST requires the corresponding continuum reconstruction and sampling
conditions; a finite order alone supplies no smooth metric.

The unit timelike parameter $\tau$ below satisfies
$G(u,u)=-1$ for $u=dz/d\tau$. With dimensional $G=-c^2dt^2+g_t$,
$\tau$ is proper length, equal to $c$ times physical proper time.
Equivalently one may use geometric units $c=1$.
:::

:::{prf:lemma} Connection of the time-dependent slab metric
:label: lem-curvature-slab-connection

For $G=-c^2dt^2+g_t$ with constant $c$,

$$
\Gamma^k_{ij}[G]=\Gamma^k_{ij}[g_t],\qquad
\Gamma^t_{ij}[G]=\frac1{2c^2}\partial_tg_{ij},\qquad
\Gamma^i_{tj}[G]=\frac12g^{ik}\partial_tg_{kj},
$$

and $\Gamma^a_{tt}=\Gamma^t_{ti}=0$.
Thus the purely spatial coefficients agree, but ambient parallel
transport along a spatial curve need not preserve its tangent space.
:::

:::{prf:proof}
Insert $G_{tt}=-c^2$, $G_{ti}=0$, and $G_{ij}=g_{ij}$ into the
connection formula. For a spatial curve and an initially spatial vector,

$$
\frac{dV^t}{ds}
=-\frac1{2c^2}\partial_tg_{ij}\,V^i\dot\gamma^j.
$$

This component can be nonzero. When $\partial_tg=0$, the spatial
subbundle is preserved and its connection is the intrinsic one.
:::

(sec-parallel-transport-holonomy)=
## Parallel Transport and Holonomy

:::{prf:definition} Parallel transport
:label: def-parallel-transport

For a piecewise smooth curve $\gamma$ and a specified connection, parallel
transport solves

$$
\frac{dV^a}{ds}
+\Gamma^a_{bc}(\gamma(s))V^b\dot\gamma^c=0.
$$

The linear map from the initial tangent space to the final one is denoted
$P_\gamma$. Intrinsic slice transport uses $\Gamma[g_t]$ throughout.
Spacetime transport uses $\Gamma[G]$, including its mixed components.
:::

:::{prf:definition} Holonomy
:label: def-holonomy

For a closed loop $\gamma$ based at $p$,
$\operatorname{Hol}_\gamma=P_\gamma:T_pM\to T_pM$.

Metric compatibility makes spacetime holonomy an element of $O(1,d)$.
On an oriented time-oriented spacetime, transport continuously connected
to the identity lies in $SO^+(1,d)$. Intrinsic holonomy of a
$d$-dimensional Riemannian slice lies in $O(d)$.

The restricted holonomy uses loops homotopic to the constant loop.
Vanishing curvature is equivalent to trivial restricted holonomy on
each connected component. Global flat connections may still have
holonomy around noncontractible loops.
:::

:::{prf:theorem} Ambrose–Singer theorem
:label: thm-ambrose-singer

For a connected smooth manifold with the specified metric connection,
the Lie algebra of restricted holonomy at $p$ is

$$
\mathfrak{hol}_p
=\operatorname{span}
 \{P_\gamma^{-1}R_q(X,Y)P_\gamma:
       \gamma:p\to q,\ X,Y\in T_qM\}.
$$

The transported curvature endomorphisms in this formula refer to that
same connection.
:::

:::{prf:proof}
This is the classical holonomy theorem, stated with its bundle argument in
{prf:ref}`appx-ambrose-singer`. Curvature is the vertical component of
commutators of horizontal lifts. Infinitesimal horizontal loops therefore
generate the displayed transported curvature directions. Conversely, their
span is preserved by horizontal transport and contains the infinitesimal
vertical displacement of such loops. These two inclusions identify the
holonomy algebra. The full bundle theorem is given in
{cite}`ambrose1953theorem,kobayashi1963foundations`.
:::

:::{prf:lemma} Holonomy of shape-controlled small loops
:label: lem-holonomy-small-loops

On a fixed normal neighborhood, let the metric be $C^3$. Consider an
oriented coordinate rectangle based at $p$, with orthonormal initial
directions $X,Y$, side lengths $r,s$, and
$\ell=\max(r,s)$, where $\min(r,s)\geq a\ell$ for a fixed $a>0$.
For a Lorentzian plane use a nondegenerate pseudo-orthonormal pair and
norms in a fixed auxiliary positive definite frame norm.

With $A=rs\asymp\ell^2$, $K=\sup|R|$, and
$K_1=\sup|\nabla R|$, choose the loop orientation so that

$$
\operatorname{Hol}_\gamma V
=V+A R_p(X,Y)V+E,\qquad
|E|\leq C(K_1A^{3/2}+K^2A^2)|V|.
$$

The constants include the neighborhood and frame bounds.
:::

:::{prf:proof}
Use the radial frame and transport integral equation in
{prf:ref}`appx-holonomy-small-loops`. In that frame the connection
one-form is $O(K\ell)$. Its curvature differs from transported curvature
at $p$ by $O(K_1\ell+K^2\ell^2)$. The leading curvature integral is
$rsR_p(X,Y)$; its variation contributes
$O(K_1\ell^3+K^2\ell^4)$. Iterated connection integrals contribute
$O(K^2\ell^4)$. The bounded aspect ratio converts these powers to
$A^{3/2}$ and $A^2$. The same local matrix estimates apply in the
fixed auxiliary norm for the Lorentzian connection.
:::

(sec-riemann-scutoid-dictionary)=
## Curvature from Consistent Plaquette Transport

:::{div} feynman-prose
The small-loop formula measures curvature per unit area. This sets the
accuracy required of a discrete rule. If the loop has diameter $h$ and
area comparable to $h^2$, an $O(h^3)$ transport error becomes an
$O(h)$ curvature error after division.

A well-shaped mesh supplies useful lengths and areas. The transport error
still has to be checked for the face maps or edge maps that are actually
used.
:::

:::{prf:definition} Scutoid plaquette
:label: def-scutoid-plaquette

A plaquette is a specified closed quadrilateral: a bottom edge from
$z_i(t)$ to $z_j(t)$, the forward trajectory of $j$, a top edge from
$z_j(t+\Delta t)$ to $z_i(t+\Delta t)$, and the reversed trajectory
of $i$. Its spanning surface and connection are specified separately.

For a nondegenerate tangent two-plane with spanning vectors $X,Y$,
the unsigned metric area factor is

$$
\sqrt{|G(X,X)G(Y,Y)-G(X,Y)^2|}.
$$

The estimate $A_\Pi\asymp\ell\,c\Delta t$ requires uniform control of
this factor and of the surface shape. A null or nearly null plaquette
needs separate treatment.
:::

:::{prf:definition} Piecewise-flat comparison holonomy
:label: def-scutoid-regge-holonomy

For a nondegenerate simplicial metric, choose orthonormal frames in adjacent
simplices and their metric-preserving face-identification maps $T_e$.
The discrete holonomy around a closed oriented dual path is their ordered
product. In spacetime dimension $d+1$ and signature $(-,+,\ldots,+)$,
orientation-preserving maps belong to $SO(1,d)$.

For a Riemannian hinge, rotation is described by the deficit
$2\pi-\sum_j\theta_j$. Around a Lorentzian spacelike hinge, the normal
plane has Lorentzian signature; its parameter is the oriented boost
rapidity determined by the face maps. The ordered product defines
comparison transport in both cases.
:::

:::{prf:definition} Shape-controlled scutoid refinement
:label: def-scutoid-refinement-regime

Let $\ell$ be a representative spatial edge length and
$h=\max(\ell,c\Delta t)\to0$, with

$$
0<\kappa_{\min}\leq\frac{c\Delta t}{\ell}\leq\kappa_{\max}<\infty.
$$

In addition, require nondegenerate tangent planes and uniformly controlled
plaquette shapes as in {prf:ref}`def-scutoid-plaquette`. Then
$A_\Pi\asymp h^2$. These geometric conditions specify the refining
family to which a small-loop estimate is applied.
:::

:::{prf:lemma} Error accumulation for comparison holonomy
:label: lem-regge-holonomy-approx

Suppose a plaquette has $m\leq m_0$ comparison transports $T_e$ and
smooth transports $P_e$, expressed in compatible frames, with

$$
\|T_e\|,\|P_e\|\leq M,\qquad
\|T_e-P_e\|\leq C_{\mathrm{conn}}h^3,\qquad M\geq1.
$$

Then

$$
\|T_m\cdots T_1-P_m\cdots P_1\|
\leq m_0M^{m_0-1}C_{\mathrm{conn}}h^3.
$$

When $A_\Pi\asymp h^2$, this is $O(A_\Pi^{3/2})$.
The connection consistency estimate is an input for the chosen maps.
:::

:::{prf:proof}
The exact telescoping identity is

$$
T_m\cdots T_1-P_m\cdots P_1
=\sum_{k=1}^m
 T_m\cdots T_{k+1}(T_k-P_k)P_{k-1}\cdots P_1.
$$

Bound the one error factor and the at most $m_0-1$ other factors in
each summand. Point-set shape regularity and weak convergence of a
curvature measure do not by themselves supply this edgewise estimate.
:::

:::{prf:theorem} Curvature recovery from consistent plaquette transport
:label: thm-riemann-scutoid

Assume {prf:ref}`lem-holonomy-small-loops` and
{prf:ref}`lem-regge-holonomy-approx` for a shape-controlled family
based at $z$, with limiting oriented orthonormal or pseudo-orthonormal
directions $X,Y$. Then

$$
\lim_{A_\Pi\to0}\frac{(\mathcal H[\Pi]-I)V}{A_\Pi}
=R_z(X,Y)V.
$$

For fixed directions the error is bounded by

$$
C\bigl((C_{\mathrm{conn}}+K_1)A_\Pi^{1/2}
                              +K^2A_\Pi\bigr)|V|.
$$

The coordinate contraction is $R^a{}_{bcd}V^bX^cY^d$.
Using the same vector in both antisymmetric slots gives zero.
:::

:::{prf:proof}
Insert smooth holonomy between $\mathcal H[\Pi]$ and $I$. Divide
the two remainder estimates by $A_\Pi$ and let the area tend to zero.
Continuity supplies the additional error when directions converge rather
than remain fixed.

For intrinsic slice transport, the result is intrinsic spatial curvature.
Spacetime curvature uses the full connection of $G$; the relation between
the two also involves the slice's second fundamental form.
:::


(sec-discrete-connection)=
## Discrete Connection Recovery

:::{div} feynman-prose
Watch an edge between two moving points. It changes because the points
have different velocities, and because comparing vectors at different
base points requires transport. These effects occur in the same
measurement.

There is also a counting issue. If all transport experiments follow one
velocity direction, they measure the connection contracted with that
direction. Recovering every connection coefficient requires enough
independent directions. The singular values of the measurement matrix
make this requirement quantitative.
:::

:::{prf:definition} Edge deformation in a declared chart
:label: def-edge-deformation

For neighboring sites, represent the coordinate displacement by a vector
$\ell_{ij}(t)$ at the base point $z_i(t)$. Alternatively, use a specified
logarithmic displacement in a normal neighborhood. For a specified
transport $P_{\gamma_i}$, define

$$
\Delta\ell_{ij}
=\ell_{ij}(t+\Delta t)-P_{\gamma_i}\ell_{ij}(t).
$$

For coordinate displacements and smooth trajectories,

$$
\Delta\ell_{ij}
=\left[v_j-v_i+\Gamma(v_i)\ell_{ij}\right]\Delta t
 +O(\Delta t^2),
$$

with the remainder retaining the corresponding trajectory and edge
derivative bounds. Thus the deformation includes the relative velocity.
It also uses the prescribed transport, which must be supplied independently
if the expression is used as measurement data.
:::

:::{prf:proof}
Taylor expand the edge and solve the transport equation through first order:

$$
\ell_{ij}(t+\Delta t)
=\ell_{ij}(t)+(v_j-v_i)\Delta t+O(\Delta t^2),\qquad
P_{\gamma_i}=I-\Gamma(v_i)\Delta t+O(\Delta t^2).
$$

Subtract the two expansions.
:::

:::{prf:proposition} Connection recovery with an identified least-squares design
:label: prop-connection-least-squares

Fix a chart and a point. For each output component $a$, parameterize the
torsion-free coefficients $\Gamma^a_{bc}$ in an orthonormal basis
$E_\alpha$ of symmetric $d\times d$ matrices, with coefficient vector
$\gamma^a$. Suppose $m$ transport measurements satisfy

$$
y_r^a=\Delta t_r\,\Gamma^a_{bc}w_r^bv_r^c+\rho_r^a,\qquad
X_{r\alpha}=\Delta t_r\,w_r^TE_\alpha v_r,
$$

where $|w_r|=O(h)$, $\Delta t_r=O(h)$, $|v_r|=O(1)$,
and $|\rho_r^a|\leq C h^3$. If

$$
\sigma_{\min}(X)\geq\kappa\sqrt m\,h^2,\qquad \kappa>0,
$$

then the least-squares estimator obeys

$$
|\widehat\gamma^a-\gamma^a|\leq(C/\kappa)h.
$$

One valid transport datum is
$y_r=w_r-P_rw_r$, for an independently specified transport with the
stated expansion and error. Raw edge deformation must first account for
the relative-velocity term in {prf:ref}`def-edge-deformation`.

For $d>1$, measurements with a single fixed $v_r=v$ cannot recover
all symmetric coefficients: they identify at most the matrix contraction
$\Gamma(v)$ and do not satisfy the stated full-rank condition.
:::

:::{prf:proof}
The data equation is $y^a=X\gamma^a+\rho^a$. Full column rank gives

$$
\widehat\gamma^a-\gamma^a=X^\dagger\rho^a,\qquad
|\widehat\gamma^a-\gamma^a|
\leq\frac{|\rho^a|}{\sigma_{\min}(X)}
\leq\frac{\sqrt m\,Ch^3}{\kappa\sqrt m\,h^2}.
$$

For a smooth connection and uniformly controlled transport paths,
$P_r=I-\Gamma(v_r)\Delta t_r+O(h^2)$; multiplication by
$w_r=O(h)$ gives the stipulated $O(h^3)$ measurement remainder.
Any discrete transport error must be included in that remainder.

With fixed $v$, the map from a symmetric coefficient matrix $A$ to
these measurements factors through $Av\in\mathbb R^d$.
Its rank is at most $d$, whereas the coefficient space has dimension
$d(d+1)/2>d$. This proves the final assertion.
:::

(sec-raychaudhuri)=
## The Raychaudhuri Equation

:::{div} feynman-prose
Take a small ball of neighboring observers moving along timelike curves.
It can grow, stretch into an ellipsoid, or rotate. Expansion, shear, and
vorticity measure these three changes.

If the observers fall freely, their acceleration vanishes and the
curvature term completes the evolution equation. Driven trajectories have
an extra acceleration term. The distinction matters for walkers subject
to forces, friction, and random velocity kicks.
:::

:::{prf:definition} Kinematic decomposition
:label: def-kinematic-decomposition

Let $u$ be a smooth unit timelike field, $G(u,u)=-1$. Set

$$
h_{ab}=G_{ab}+u_au_b,\qquad a_a=u^b\nabla_bu_a,\qquad
B_{ab}=h_a{}^ch_b{}^d\nabla_du_c.
$$

Define $\theta=\nabla_au^a$, the symmetric trace-free part
$\sigma_{ab}$, and the antisymmetric part $\omega_{ab}$ by

$$
B_{ab}=\frac{\theta}{d}h_{ab}+\sigma_{ab}+\omega_{ab}.
$$

The complete decomposition is

$$
\nabla_bu_a=B_{ab}-a_a u_b.
$$

For a geodesic congruence $a=0$. The contractions
$\sigma^2=\sigma_{ab}\sigma^{ab}$ and
$\omega^2=\omega_{ab}\omega^{ab}$ are nonnegative, because these
tensors act on the positive definite rest space orthogonal to $u$.
:::

:::{prf:theorem} Raychaudhuri equation
:label: thm-raychaudhuri

For a smooth unit timelike geodesic congruence in dimension $d+1$,

$$
D_u\theta
=-\frac{\theta^2}{d}-\sigma^2+\omega^2-\operatorname{Ric}(u,u).
$$

For a smooth accelerated congruence, the right-hand side additionally
contains $\nabla_a a^a$.
:::

:::{prf:proof}
Commute derivatives in $D_u(\nabla_au^a)$, using the curvature
convention above and $a=\nabla_u u$. The result is

$$
D_u\theta
=\nabla_aa^a-(\nabla_au^b)(\nabla_bu^a)
                  -\operatorname{Ric}(u,u).
$$

Orthogonality of $B$ and $a$ to $u$ makes the quadratic contraction
equal to $B_{ab}B^{ba}$. Trace-freeness and symmetry remove its cross
terms, leaving

$$
B_{ab}B^{ba}
=\frac{\theta^2}{d}+\sigma^2-\omega^2.
$$

Substitute and set $a=0$ for the geodesic case. This is the direct
calculation of {prf:ref}`appx-raychaudhuri`.
:::

(sec-discrete-raychaudhuri)=
## Expansion from Cell Volumes

:::{div} feynman-prose
A material cell follows the same particles at all times. A Voronoi cell is
redrawn as its neighboring sites move. Their boundaries need not have the
same normal velocity.

To compare their expansions, divide the difference in boundary flux by
cell volume. Since a small cell has much more boundary area per unit
volume, a small error on each face can still produce a large expansion
error. This is why the normalized flux and its derivative appear
explicitly below.
:::

:::{prf:definition} Regularity and cell scales
:label: def-regularity-conditions

On the fixed spacetime window, assume a sufficiently smooth metric and
unit timelike congruence for the displayed differentiated identities,
with uniform bounds on the geometric and flow derivatives they use.
For transverse spatial cells let $\epsilon_N\to0$ be a chosen
resolution scale, with

$$
\operatorname{diam}(C_i)\leq C_1\epsilon_N,\qquad
\operatorname{Vol}(C_i)\geq C_2\epsilon_N^d,
$$

and bounded shape ratios in the specified normal neighborhoods.
Cell injectivity and regularity concern the transverse Riemannian
geometry. A Lorentzian metric does not supply a positive distance
for these diameter bounds.

The scale $\epsilon_N$ is determined by the actual reconstruction and
coverage estimate. The independent, joint-entropy, LSI, and same-sample
coverage bounds in {doc}`02_scutoid_spacetime` give distinct sufficient
sampling estimates. A uniform maximal cell diameter $O(N^{-1/d})$
is an additional geometric property, not a consequence of particle
count alone.
:::

:::{prf:theorem} Discrete Raychaudhuri correspondence
:label: thm-discrete-raychaudhuri

Let the sites follow a smooth unit timelike geodesic congruence in dimension
$d+1$. In addition to {prf:ref}`def-regularity-conditions`, suppose
the transverse cell volumes $V_i>0$ satisfy

$$
\theta_i=\frac{\dot V_i}{V_i}=\theta(z_i)+r_i,\qquad
|r_i|+|\dot r_i|\leq C\epsilon_N.
$$

On a window with bounded expansion, shear, vorticity, and curvature,

$$
\dot\theta_i
=-\frac{\theta_i^2}{d}-\sigma^2(z_i)+\omega^2(z_i)
-\operatorname{Ric}(u,u)(z_i)+O(\epsilon_N).
$$

Material cells satisfy the consistency condition under the spatial and
material derivative bounds of {prf:ref}`appx-discrete-raychaudhuri`.
Reconstructed Voronoi cells additionally require their normalized
boundary-flux defect and its time derivative to be $O(\epsilon_N)$.
The statement applies on intervals of differentiable positive cell
volume; topology events require the corresponding jump terms.
:::

:::{prf:proof}
Differentiate $\theta_i=\theta(z_i)+r_i$. The Raychaudhuri identity
gives

$$
\dot\theta_i
=-\frac{(\theta_i-r_i)^2}{d}-\sigma^2(z_i)+\omega^2(z_i)
-\operatorname{Ric}(u,u)(z_i)+\dot r_i.
$$

Bounded expansion and the two bounds on $r_i$ give the result.

For material cells use the transverse flux volume form
$\iota_u d\operatorname{vol}_G$, pulled back to a transported transverse
section, and write $\langle f\rangle_i=V_i^{-1}\int_{C_i}f$.
This agrees with induced rest-space volume for sections orthogonal to $u$.
Since $\mathcal L_u(\iota_u d\operatorname{vol}_G)
=\theta\,\iota_u d\operatorname{vol}_G$, the material Jacobian
satisfies $\dot J=\theta J$, so

$$
\frac{\dot V_i}{V_i}=\langle\theta\rangle_i,\qquad
\frac d{d\tau}\langle\theta\rangle_i
=\langle D_u\theta\rangle_i
 +\langle\theta^2\rangle_i-\langle\theta\rangle_i^2.
$$

Spatial Lipschitz bounds on $\theta$ and $D_u\theta$, together with
cell diameter $O(\epsilon_N)$, give the differentiated consistency
estimate. In particular the variance term is $O(\epsilon_N^2)$.

For a reconstructed boundary with normal velocity $w\cdot n$, the
additional normalized flux is

$$
b_i=\frac1{V_i}\int_{\partial C_i}(w-u)\cdot n\,dA.
$$

Reynolds transport gives this term exactly, with the appropriate
time-dependent volume derivative for the chosen transverse chart.
The conditions $|b_i|+|\dot b_i|=O(\epsilon_N)$ retain it and its
derivative. These are the flux conditions proved and specified in
{prf:ref}`appx-discrete-raychaudhuri`. A face error is multiplied
by the surface-to-volume ratio when estimating $b_i$.
:::

:::{prf:corollary} Relaxation under positive expansion damping
:label: cor-discrete-relaxation

Suppose $\dot\theta_i=-\theta_i^2/d+F_i$ and
$\dot{\bar\theta}=-\bar\theta^2/d+\bar F$, with

$$
\theta_i+\bar\theta\geq d\lambda>0,\qquad
|F_i-\bar F|\leq C\epsilon_N.
$$

Then

$$
|\theta_i(t)-\bar\theta(t)|
\leq e^{-\lambda t}|\theta_i(0)-\bar\theta(0)|
 +\frac{C\epsilon_N}{\lambda}(1-e^{-\lambda t}).
$$
:::

:::{prf:proof}
The difference $\delta=\theta_i-\bar\theta$ satisfies

$$
\dot\delta
=-\frac{\theta_i+\bar\theta}{d}\delta+(F_i-\bar F).
$$

Use its integrating factor and the lower damping bound to estimate the
initial term and the forcing integral. Curvature boundedness alone does
not provide this positive damping coefficient.
:::

(sec-focusing-theorem)=
## Geodesic Focusing

:::{prf:definition} Timelike convergence condition
:label: def-sec

The geometric timelike convergence condition is

$$
\operatorname{Ric}(u,u)\geq0
$$

for every timelike $u$. In relativity it is related to a strong energy
condition through specified field equations and their cosmological term.
Here it is a condition on the chosen metric. The historical label
`def-sec` refers to this geometric condition.
:::

:::{prf:theorem} Focusing of an initially contracting congruence
:label: thm-focusing

Let a unit timelike geodesic congruence be vorticity free, satisfy the
timelike convergence condition, and have $\theta(0)=\theta_0<0$
on the geodesic under consideration. If that geodesic extends to

$$
\tau_*=\frac d{|\theta_0|},
$$

then the congruence cannot remain regular with nonzero transverse
Jacobian throughout $[0,\tau_*]$. Focusing occurs no later than
$\tau_*$; along a regular branch approaching the focusing time,
the expansion becomes unbounded below.
:::

:::{prf:proof}
Raychaudhuri and $\sigma^2\geq0$ give
$\dot\theta\leq-\theta^2/d$. The initially negative expansion stays
negative and decreases. Therefore

$$
\frac d{d\tau}\frac1\theta
=-\frac{\dot\theta}{\theta^2}\geq\frac1d,
\qquad
\theta(\tau)\leq\frac{\theta_0}{1+\theta_0\tau/d}
\quad(0\leq\tau<\tau_*).
$$

If $J$ is the positive transverse Jacobian, then
$\dot J/J=\theta$, so integration yields

$$
\frac{J(\tau)}{J(0)}
\leq(1+\theta_0\tau/d)^d.
$$

The right-hand side tends to zero at $\tau_*$. Thus regular
nonvanishing transverse volume cannot persist to that time. Since
$\theta$ is decreasing, a finite endpoint for a regular branch
leading to focusing has unbounded negative expansion.
:::


(sec-curvature-topology)=
## Curvature and Changes of Tessellation

:::{div} feynman-prose
Draw another point inside a triangulated disk and redraw the triangles.
The numbers of vertices, edges, and faces change, but the disk is still a
disk. Euler's alternating sum keeps track of that fact.

Curvature has a related accounting identity in two dimensions. On a closed
surface, the total angle deficit is fixed by topology. On a surface with
boundary, turning at the boundary shares that total. Inserting a site can
redistribute the bookkeeping; it does not create a new handle or hole by
itself.
:::

:::{prf:definition} Riemannian Regge curvature
:label: def-regge-curvature

For a nondegenerate piecewise Euclidean manifold of spatial dimension $d$,
a hinge $h$ has dimension $d-2$. At an interior hinge define the
Riemannian rotation deficit

$$
\varepsilon_h=2\pi-\sum_{\text{incident simplices}}\theta_h.
$$

The associated scalar-curvature measure assigns integrated weight

$$
\mathcal R_{\mathrm{Regge}}
=2\sum_h\varepsilon_h\,\operatorname{Vol}_{d-2}(h).
$$

For $d=2$, hinges are vertices and the Gaussian-curvature weight is
$\varepsilon_h$, half the scalar-curvature weight. Boundary terms are
specified separately.

A smooth-limit comparison uses a chosen metric approximation and its
curvature-measure consistency theorem. Lorentzian spacetime hinges have
dimension $(d+1)-2=d-1$ and use the face-map transport of
{prf:ref}`def-scutoid-regge-holonomy`, with the appropriate rotation
or boost parameter.
:::

:::{prf:theorem} Euler constraints for cell refinements
:label: thm-curvature-topology

Let a finite regular cell decomposition of a fixed space have $F_k$
cells of dimension $k$. Then

$$
\chi=\sum_{k=0}^d(-1)^kF_k.
$$

Two such decompositions of the same underlying space have the same
Euler characteristic. Consequently a tessellation change on that space
satisfies

$$
\sum_{k=0}^d(-1)^k\Delta F_k=0.
$$

In two dimensions, $\Delta V-\Delta E+\Delta F=0$.
When site insertion or removal only changes the decomposition, this
relation applies to all affected face counts together.
:::

:::{prf:proof}
For the finite cellular chain complex, write $Z_k$ and $B_k$ for
cycles and boundaries. Rank-nullity gives

$$
\dim C_k=\dim Z_k+\dim B_{k-1},\qquad
\dim Z_k=\dim H_k+\dim B_k.
$$

Alternating sums cancel the boundary dimensions, yielding
$\sum_k(-1)^kF_k=\sum_k(-1)^k\dim H_k$. The latter depends only on
the underlying space. Subtract the identity for the two decompositions.
:::

:::{prf:proposition} Curvature balance under a two-dimensional cell change
:label: prop-curvature-change-2d

For a finite piecewise Euclidean surface with polygonal boundary, let

$$
\varepsilon_v=2\pi-\sum_{T\ni v}\alpha_{T,v}
\quad\text{at interior vertices},\qquad
\beta_v=\pi-\sum_{T\ni v}\alpha_{T,v}
\quad\text{at boundary vertices}.
$$

Then

$$
\sum_{v\ \mathrm{interior}}\varepsilon_v
+\sum_{v\ \mathrm{boundary}}\beta_v=2\pi\chi.
$$

A change of tessellation on a fixed closed surface therefore has
$\Delta\sum_v\varepsilon_v=0$. For a fixed surface with boundary,

$$
\Delta\sum_{\mathrm{interior}}\varepsilon_v
=-\Delta\sum_{\mathrm{boundary}}\beta_v.
$$

More generally, a true change of topology contributes $2\pi\Delta\chi$.
Adding a site while retaining the same underlying domain has
$\Delta\chi=0$.

For a generic planar Voronoi modification away from an unchanged boundary,
if all affected vertices remain trivalent, then

$$
\Delta V=2\Delta F,\qquad \Delta E=3\Delta F.
$$

These counting identities require no curvature creation. For an ordinary
Euclidean tessellation every interior deficit is zero because its incident
angles fill the full angle $2\pi$.

If at most $m_0$ vertices change and each has at most $q_0$ incident
triangles, their local absolute deficit variation is bounded by

$$
\sum_{\mathrm{changed}\ v}
 |\varepsilon_v^{\mathrm{new}}-\varepsilon_v^{\mathrm{old}}|
\leq2m_0(2\pi+q_0\pi),
$$

where missing vertices have zero weight. This is an explicit sufficient
condition for a local $O(1)$ bound.
:::

:::{prf:proof}
Triangulate each polygon without changing the metric. Let $V_i,V_b$
be the numbers of interior and boundary vertices, and let $E_i,E_b,F$
count interior edges, boundary edges, and triangles. The sum of all
triangle angles is $\pi F$, so the left side of the first identity is

$$
2\pi V_i+\pi V_b-\pi F.
$$

For a surface with closed polygonal boundary components,
$E_b=V_b$ and $3F=2E_i+E_b$. Substitution gives

$$
2\pi V_i+\pi V_b-\pi F
=2\pi(V_i+V_b-E_i-E_b+F)=2\pi\chi.
$$

Subtract the identities before and after the change. On a closed
surface the boundary sum is absent.

For the trivalent planar modification, unchanged boundary contributions
cancel in degree counting, leaving $3\Delta V=2\Delta E$.
Combine this with $\Delta V-\Delta E+\Delta F=0$ to obtain the
displayed face-count changes.

Each triangle angle lies in $[0,\pi]$, so
$|\varepsilon_v|\leq2\pi+q_0\pi$. Applying this bound before and
after the modification proves the local variation estimate. In a
Euclidean realization the exact full-angle sum proves zero interior
deficit, including at irregular vertices.
:::

:::{prf:remark} Higher-dimensional curvature and scaling
:label: rem-higher-dim-curvature

In spatial dimension $d=3$, Riemannian hinges are edges; in $d=4$
they are two-dimensional faces. Their deficits and hinge volumes both
enter the Regge curvature measure. Face counts alone do not determine
these weights.

For a direct obstruction to a universal curvature-per-cell constant,
scale all metric lengths by $s>0$ in a piecewise Euclidean
$d$-manifold. Its combinatorics and dihedral angles stay fixed,
while each hinge volume scales by $s^{d-2}$. Hence

$$
\mathcal R_{\mathrm{Regge}}(s)
=s^{d-2}\mathcal R_{\mathrm{Regge}}(1).
$$

For $d>2$ and nonzero initial curvature measure, identical cell counts
can therefore have different total scalar curvature. A statistical
relation for a specified random metric would require its metric law and
normalization in addition to an incidence model. The smooth convergence
results of {cite}`cheeger1984curvature` apply with their metric and
approximation hypotheses.
:::

(sec-fg-gravity-physical-interpretation)=
## Geometric and Dynamical Interpretations

:::{div} feynman-prose
The equations now tell us exactly what a curvature measurement means.
The configured diffusion rule identifies a metric field. Differentiating
that field computes its curvature, and consistent discrete transport
provides an independent recovery. A specified smooth flow additionally
allows us to compare geometric expansion with measured volume changes.

Optimization asks an additional question: does the actual stochastic
process approach its target law or improve its objective? The force,
noise, and cloning analyses answer that question. A geometric caustic is
a loss of regularity of a family of geodesics, so it cannot be substituted
for a mixing or optimization theorem.
:::

:::{prf:remark} The geometric correspondence and its hypotheses
:label: rem-gravity-optimization

| Geometric quantity | Construction used here | Required identification |
|---|---|---|
| Fitness derivatives | Differentiated conditional fitness pipeline | Recorded conditioning data, active smooth stratum, and stated derivative convention |
| Spatial metric | Inverse adaptive covariance shape | Correct diffusion branch, prefactor, and spatial reconstruction |
| Spacetime metric | $G=-c^2dt^2+g_t$ | The specified Lorentzian model and causal consistency |
| Parallel transport | Levi-Civita or stated face maps | Same connection and compatible frames |
| Curvature | Metric derivative formula or consistent small-loop holonomy | A $C^2$ metric; transport recovery also needs nondegenerate loops and $o(A)$ transport error |
| Hessian-metric curvature | Quadratic expression in third derivatives | A smooth positive Hessian metric |
| Expansion | $\dot V/V$ | Transverse volume and differentiated consistency |
| Geodesic focusing | Raychaudhuri inequality | Geodesic flow, convergence condition, and zero vorticity |

The next dynamical calculation uses the full transition law to derive
metric increments and mechanical balances, as in
{doc}`04_field_equations`. An Einstein-type equation would require
identifying a controlled limit of those independently derived laws,
including its stress tensor and any unresolved terms. Computing
curvature alone does not determine that evolution or its coefficients.
:::

:::{prf:remark} Focusing and optimization use different evolution estimates
:label: rem-focusing-optimization

The focusing theorem controls a smooth timelike geodesic congruence.
Its conclusion is a caustic or vanishing transverse Jacobian. It does
not identify an objective minimizer or a stationary probability law.

The actual kinetic walkers have forces, friction, velocity noise, and
cloning events. A smooth accelerated comparison has the additional
$\nabla_a a^a$ term in Raychaudhuri; stochastic trajectories require
their own stochastic analysis. Approximation by a geodesic congruence
requires a stated approximation estimate.

The established analytical routes are
{doc}`../convergence_program/06_convergence` for the specified
finite-particle transition kernel,
{doc}`../convergence_program/12_qsd_exchangeability_theory` for
joint-law fluctuations, and
{doc}`../convergence_program/15_kl_convergence` together with
{doc}`../convergence_program/17_geometric_gas` for the proved
joint-law LSI criteria and complete entropy evolution. Their hypotheses
determine the dynamical convergence statement.
:::

(sec-symbols-curvature)=
## Table of Symbols

| Symbol | Meaning |
|---|---|
| $d$ | Spatial dimension; spacetime dimension is $d+1$ |
| $g_t$, $G$ | Intrinsic spatial and Lorentzian spacetime metrics |
| $\Gamma[q]$ | Levi-Civita connection of the specified metric $q$ |
| $R^a{}_{bcd}$ | Curvature convention in {prf:ref}`def-riemann-tensor` |
| $\operatorname{Ric}$, $R$ | Ricci contraction and scalar trace |
| $\theta$, $\sigma$, $\omega$ | Expansion, shear, and vorticity |
| $a=\nabla_u u$ | Acceleration of the chosen smooth flow |
| $\mathcal H[\Pi]$ | Ordered discrete transport around the plaquette |
| $A_\Pi$, $h$ | Oriented-plane area magnitude and mesh resolution |
| $\epsilon_N$ | Cell scale established by the chosen geometric estimate |
| $\varepsilon_h$ | Riemannian hinge deficit |
| $\chi$ | Euler characteristic of the underlying space |

(sec-summary-curvature)=
## Results Carried Forward

The conditioned algorithm data and differentiated fitness pipeline
specify the metric-producing calculation. Smooth Hessian metrics admit
direct third-derivative curvature contractions, including the packed
three-dimensional calculation and its explicit curvature bound.
Mixed-sign clipping uses the full metric derivative rule under a
spectral gap; the required smoothness remains part of each result.

The small-loop and product-error proofs give independent curvature
recovery for a consistent discrete connection. Least-squares recovery
has the stated design-rank and measurement-error requirements.

Raychaudhuri's identity and the material-volume calculation transfer to
reconstructed cells when their normalized flux and differentiated errors
are controlled. Positive expansion damping yields the quantified
relaxation estimate. Timelike convergence with zero vorticity yields
geodesic focusing.

Euler and Gauss–Bonnet constrain changes of a tessellation on its
underlying space. In two dimensions, total curvature on a fixed closed
surface is preserved. These facts specify the geometric content used
by the following field-equation chapter.

(sec-curvature-references)=
## References

The local transport and volume arguments also appear in
{doc}`../convergence_program/17_geometric_gas`. The classical
sources used here are the holonomy theorem
{cite}`ambrose1953theorem,kobayashi1963foundations`,
Raychaudhuri's identity and focusing theory
{cite}`raychaudhuri1955relativistic,hawking1973large,wald1984general`,
and piecewise-flat curvature
{cite}`regge1961general,cheeger1984curvature`.

```{bibliography}
:filter: docname in docnames
```
