(sec-scutoid-spacetime)=
# Scutoid Spacetime: Moving Cells and Neighbor Changes

**Prerequisites:** {doc}`01_emergent_geometry`,
{doc}`../2_fractal_set/02_causal_set_theory`, and the metric and sampling
conditions of {doc}`../convergence_program/16_continuum_discharge`.

(sec-scutoid-tldr)=
## TLDR

:::{div} feynman-prose
At each recorded time, give every point of space to its nearest walker.
Now stack these Voronoi partitions in time. A cell sweeps out a region of
spacetime. When neighbors change, the labeled boundaries have to reorganize.
Scutoids provide a useful language for describing that reorganization.

The construction has three inputs: a distance on each spatial slice, a rule
for filling the intervals between recorded frames, and the recorded cloning
and motion events. Cloning can change neighbors, but continuous motion can do
so too. A cell can also keep the same neighbors while its size and shape
change substantially. We keep those possibilities visible in the definitions.
:::

(sec-scutoid-introduction)=
## 1. From recorded sites to moving cells

Write $N$ for the population and $n\le N$ for the sites included in a
particular alive-status stratum. The geometric reconstruction uses those
sites, their labels, and a specified spatial metric $g_t$. It preserves the
algorithm's recorded states. Interpolation between frames is reconstruction
data, rather than an additional kinetic or cloning update.

The term *scutoid* comes from the description of neighbor exchange in epithelial
packing by [Gómez-Gálvez and colleagues](https://www.nature.com/articles/s41467-018-05376-1).
Here it describes cells with changes in their labeled boundary structure along
the time direction. Curved Voronoi boundaries generally give curved cells;
a polytope requires an additional piecewise-linear realization. In arbitrary
dimension, a transition can occupy a higher-dimensional stratum rather than
a single vertex.

The analytic inputs already established in this volume remain available:

- {prf:ref}`thm-c3-regularity` controls sampled and expected fitness on its
  specified configuration strata. A Hessian metric built from a $C^3$
  potential is $C^1$. The higher derivatives required for classical curvature
  can be supplied by {prf:ref}`thm-main-cinf-regularity-fitness-potential-full`
  for the same chosen field under that theorem's hypotheses.
- {prf:ref}`lem-qsd-strict-positivity`,
  {prf:ref}`lem-linfty-full-operator`, and
  {prf:ref}`lem-gaussian-kernel-lower-bound` provide density bounds for their
  specified kernels and domains, retaining survival denominators and source
  masses.
- {prf:ref}`thm-mixing-variance-corrected` gives concentration of bounded
  empirical observables from total relative entropy.
  {prf:ref}`cor-n-uniform-lsi` gives full-gradient concentration for the actual
  joint laws meeting its four sufficient criteria. Discrete statuses retain
  the entropy term in {prf:ref}`prop-kl-status-entropy`.
- {prf:ref}`appx-voronoi-boundary-velocity` and
  {prf:ref}`appx-reynolds-transport` describe moving interfaces and their
  volumes. Their differentiated consistency conditions, rather than a cell
  count alone, enter {prf:ref}`appx-discrete-raychaudhuri`.

(sec-time-varying-voronoi)=
## 2. Spatial metrics and Voronoi partitions

### 2.1. Regularity inherited from the analytic fitness estimates

:::{prf:lemma} Metric bounds from the specified fitness field
:label: lem-scutoid-metric-bounds

On a fixed coordinate domain let
$g=D_x^2V+\epsilon_\Sigma I$, where $V$ is the identified spatial fitness
field. Suppose

$$
 \lambda I\preceq g\preceq\Lambda I,\qquad\lambda>0,
 \qquad |D_x^3V|\le K_3.
\tag{SC.1}
$$

Then $g$ is $C^1$ when $V$ is $C^3$, its connection is continuous, and
$|\Gamma|\le C_dK_3/\lambda$. If $V\in C^4$ with
$|D_x^4V|\le K_4$, its curvature components satisfy

$$
 |R^a{}_{bcd}|\le
 C_d\left(K_4/\lambda+K_3^2/\lambda^2\right).
\tag{SC.2}
$$

Here derivative bounds are in fixed coordinate tensor norms and $C_d$
absorbs dimension-dependent norm comparisons. For example,
$\|D_x^2V\|_{\rm op}\le\eta\epsilon_\Sigma$ with $\eta<1$ gives
$\lambda=(1-\eta)\epsilon_\Sigma$ and
$\Lambda=(1+\eta)\epsilon_\Sigma$.
:::

:::{prf:proof}
The last inequalities follow by adding $\epsilon_\Sigma I$ to the Hessian
bounds. In general the positive spectral margin in (SC.1) is the required
hypothesis; a positive shift alone does not control a more negative Hessian.
The connection formula in {prf:ref}`def-gg-metric-connection` contains one
factor of $g^{-1}$ and one derivative of $g$, giving its stated bound.
Differentiating $g^{-1}$ gives
$Dg^{-1}=-g^{-1}(Dg)g^{-1}$. Thus $D\Gamma$ is bounded by a constant times
$K_4/\lambda+K_3^2/\lambda^2$. The quadratic $\Gamma\Gamma$ terms have
the same bound. This proves (SC.2). For $g$-orthonormal vectors, a sectional
curvature bound follows, for example, by multiplying its right-hand side by
$C_d\Lambda/\lambda^2$.
:::

These bounds reuse the actual derivative estimates, with their dependence on
localization scales, regularizers, and the selected companion law. A nodal
fitness statistic becomes a common spatial field only after that field has
been defined. Switching sampled companion assignments or alive strata can
produce jumps between otherwise smooth fields.

For every curve in the coordinate domain, (SC.1) gives
$\sqrt\lambda L_E\le L_g\le\sqrt\Lambda L_E$. Infimizing over the same
admissible curves gives the same bounds for the intrinsic distances. When the
domain is Euclidean convex, its Euclidean intrinsic distance is $|x-y|$.
These estimates do not impose a sign on sectional curvature.

### 2.2. Cells and their regular interfaces

:::{prf:definition} Voronoi partition at a fixed time
:label: def-voronoi-tessellation-time-t

Let $(M,d_{g_t})$ be a specified length space from the spatial metric, and let
$z_1(t),\ldots,z_n(t)$ be distinct included sites. Define

$$
 \operatorname{Vor}_i(t)
 =\{x\in M:d_{g_t}(x,z_i(t))\le d_{g_t}(x,z_j(t))\ \forall j\}.
\tag{SC.3}
$$

These are closed nearest-site cells whose union is $M$. Ties belong to more
than one closed cell; a fixed tie rule gives disjoint measurable ownership.
For computations on a specified observation region $K$, use the clipped cells
$K\cap\operatorname{Vor}_i(t)$ and retain the clipping boundary. Clipping is
an observation convention, not a change to the swarm state space.
:::

:::{prf:lemma} Cell coverage and radial connectivity
:label: lem-scutoid-voronoi-basic

The cells in (SC.3) cover $M$ and are closed. If minimizing geodesics from
$z_i$ to points of its cell exist, every point on each such geodesic remains
in the cell. In particular the cell is path connected. If these geodesics
are unique and depend continuously on their endpoints throughout the cell,
radial contraction makes the cell contractible. In Euclidean space every
cell is an intersection of affine half-spaces and is convex.
:::

:::{prf:proof}
The minimum of finitely many distances is attained at some site, proving
coverage. Distance functions are continuous, so their non-strict comparisons
define closed sets. Let $x\in\operatorname{Vor}_i$ and let $y$ lie on a
minimizing geodesic from $z_i$ to $x$. For every $j$,

$$
 d(y,z_i)=d(x,z_i)-d(x,y)
 \le d(x,z_j)-d(x,y)\le d(y,z_j),
$$

where the last step is the triangle inequality. This proves the radial
statement. The stated uniqueness and continuous-dependence hypothesis makes
the radial interpolation a continuous deformation to $z_i$. In Euclidean
coordinates, squaring the distance inequality cancels $|x|^2$ and gives
$2(z_j-z_i)\cdot x\le|z_j|^2-|z_i|^2$, an affine half-space.
:::

On a complete connected Riemannian manifold, minimizing geodesics exist.
The radial argument therefore gives connectivity without a curvature sign.
Geodesic convexity of all cells is a stronger property and is not assumed
for a general variable-curvature metric. Clipping by an arbitrary region can
also destroy connectivity.

:::{prf:definition} Facet neighbors and interfaces
:label: def-neighbor-set

On a regular cell family define

$$
 \Gamma_{ij}(t)=\partial\operatorname{Vor}_i(t)
                    \cap\partial\operatorname{Vor}_j(t),\qquad
 \mathcal N_i(t)=
 \{j\ne i:\mathcal H^{d-1}_{g_t}(\Gamma_{ij}(t))>0\}.
\tag{SC.4}
$$

Regular portions of an interface avoid the sites' cut loci and have
nonvanishing spatial gradient of
$\psi_{ij}(x,t)=\frac12d_{g_t}^2(x,z_i(t))-\frac12d_{g_t}^2(x,z_j(t))$.
There the implicit function theorem gives a hypersurface. A lower-dimensional
contact does not make two cells facet neighbors. If clipped cells are used,
apply (SC.4) to their retained interfaces and record the boundary faces
separately.
:::

(sec-dual-delaunay-triangulation)=
### 2.3. The nerve and the Euclidean triangulation

:::{prf:definition} Delaunay nerve and a triangulation realization
:label: def-delaunay-triangulation

The geodesic Delaunay nerve is the abstract simplicial complex

$$
 \mathcal D(t)=\{I\subseteq\{1,\ldots,n\}:I\ne\varnothing,
                   \ \bigcap_{i\in I}\operatorname{Vor}_i(t)\ne\varnothing\}.
\tag{SC.5}
$$

Every subset of an admitted simplex is admitted. In degenerate configurations
its edges can include contacts that are not facet neighbors in (SC.4).
A triangulation realization additionally requires that its simplices embed
with disjoint interiors and cover the region being triangulated. A
contractible-intersection condition in a suitable good cover provides a nerve
homotopy equivalence; it does not by itself provide this geometric realization.
:::

:::{prf:lemma} Euclidean Delaunay realization by lifting
:label: lem-scutoid-euclidean-delaunay

Let finitely many sites in $\mathbb R^d$ affinely span $\mathbb R^d$, with
no $d+2$ cospherical sites and with the usual simplicial general-position
conditions. Their Euclidean Delaunay simplices triangulate their convex hull.
Their full-dimensional simplices are exactly the empty-circumsphere simplices.
:::

:::{prf:proof}
For $n=d+1$ the only full simplex is the required triangulation. Otherwise
lift each site $z$ to $(z,|z|^2)$. A nonvertical lower supporting hyperplane
has equation $u=2c\cdot x+b$. Every lifted site is on or above it precisely
when $|z-c|^2\ge b+|c|^2$. Its contact sites therefore lie on a sphere
with no site in its interior. General position makes its full lower facets
simplicial. The lower boundary of the convex hull of the lifted points is
the graph of a continuous piecewise-affine function over the original convex
hull. Its projected facets cover that hull and meet only in common faces.
Thus they form a triangulation with exactly the claimed empty spheres.
:::

For a nonconstant Riemannian metric, density and avoidance of cospherical
degeneracy alone do not imply this realization; see the explicit
[obstructions of Boissonnat, Dyer, Ghosh, and Martynchuk](https://arxiv.org/abs/1612.02905).
The moving-cell constructions below can use the Voronoi partition and its
nerve without asserting a triangulation. Euclidean algorithms are applied
only in their Euclidean or fixed flat-metric regime.

### 2.4. What density and entropy prove about sampling

:::{prf:lemma} Counts at fixed and sampled centers
:label: lem-scutoid-neighbor-counts

Fix the metric and the tested balls. Let $Z_1,\ldots,Z_n$ have marginal densities bounded by $q^*$ with respect
to $dV_g$ on the balls considered. For a deterministic center $x$,

$$
 \mathbb E\#\{i:Z_i\in B_g(x,r)\}
 \le nq^*\operatorname{Vol}_g(B_g(x,r)).
\tag{SC.6}
$$

For a ball centered at $Z_i$, the corresponding bound with $n-1$ requires
bounds on the conditional densities of $Z_j$ given $Z_i$; then add the
center itself. If these conditional bounds hold and ball volumes are at most
$v^*r^d$, the probability that any two sites are within $r$ is at most
$\binom n2q^*v^*r^d$.
:::

:::{prf:proof}
Write the count as a sum of indicators and integrate each marginal density.
For a sampled center condition on $Z_i$ before integrating. A union bound
over unordered pairs gives the last result. This is the spatial counterpart
of {prf:ref}`lem-c3-count-density-tail`, with the appropriate spatial volume
in place of its phase-space volume. Identical correlated sites can have a
bounded common marginal density and still all lie at the sampled center;
this explains the conditional-density requirement.
:::

:::{prf:lemma} Empty-region and fill-distance bounds
:label: lem-scutoid-fill-distance

Fix a deterministic metric $g$, observation region $K$, and probability reference
$\rho=q\,dV_g$. Suppose $K$ has an $r$-net of $M_r$ centers in $K$, and
every radius-$r$ ball at those centers has $\rho$ mass at least $m_r>0$.
Let $h_K=\sup_{x\in K}\min_i d_g(x,Z_i)$. For independent sites of law
$\rho$,

$$
 \mathbb P(h_K>2r)\le M_re^{-nm_r}.
\tag{SC.7}
$$

For an arbitrary joint law $\pi_n$ with total relative entropy
$H_n=D_{\rm KL}(\pi_n\Vert\rho^{\otimes n})$, if
$nm_r>\log M_r$, then

$$
 \mathbb P_{\pi_n}(h_K>2r)
 \le\frac{H_n+\log2}{nm_r-\log M_r}.
\tag{SC.8}
$$

An alternative uses the actual continuous joint law's full-gradient LSI with
constant $C_*$, in coordinates satisfying $g\preceq\Lambda I$. Suppose its
site marginals give every radius-$r/2$ ball mass at least $m_{r/2}$.
Then

$$
 \mathbb P_{\pi_n}(h_K>2r)
 \le M_r\exp\!\left[-\frac{n r^2m_{r/2}^2}{8C_*\Lambda}\right].
\tag{SC.9}
$$

The LSI is for this law or the specified continuous status stratum. Mixed
status laws require their additional discrete control.
:::

:::{prf:proof}
If every net-center ball contains a site, the triangle inequality places a
site within $2r$ of every point of $K$. In the independent law, a ball of
mass at least $m_r$ is empty with probability at most
$(1-m_r)^n\le e^{-nm_r}$. A union bound gives (SC.7).

For any event $A$ with reference probability $p\in(0,1)$ and actual
probability $a$, the relative entropy chain rule for its indicator gives

$$
 H_n\ge a\log(a/p)+(1-a)\log((1-a)/(1-p))
 \ge a\log(1/p)-\log2.
$$

Here binary entropy is at most $\log2$, and the omitted
$-(1-a)\log(1-p)$ is nonnegative. Apply this to $A=\{h_K>2r\}$ and
$p\le M_re^{-nm_r}$ to obtain (SC.8). Reference-null events also have
actual probability zero when $H_n<\infty$. This is the same relative-entropy
comparison underlying {prf:ref}`thm-mixing-variance-corrected`.

For (SC.9), at each net center choose
$\psi(x)=\max(0,1-d_g(x,x_0)/r)$ and set
$F=n^{-1}\sum_i\psi(Z_i)$. Its mean is at least $m_{r/2}/2$, and
its coordinate Lipschitz constant is at most
$\sqrt\Lambda/(r\sqrt n)$, by the metric length bound. The full-gradient
LSI and its exponential-moment proof in {prf:ref}`lem-ym-lsi-moments` give

$$
 \mathbb P(F=0)
 \le\exp[-n r^2m_{r/2}^2/(8C_*\Lambda)].
$$

An empty ball forces $F=0$. Sum this bound over the net centers. Distance
functions and the cutoff are Lipschitz; the LSI extends to them by its usual
Sobolev approximation.
:::

For example, suppose $q\ge q_*>0$ on the tested balls,
$\operatorname{Vol}_g B_g(x,r)\ge v_*r^d$, and
$M_r\le C_Kr^{-d}$. These are local hypotheses on a fixed observation region.
The independent estimate gives fill distance of order
$(\log n/n)^{1/d}$ with a sufficiently large constant. Estimate (SC.8)
also tends to zero whenever
$(H_n+1)/(nr^d)\to0$ and $\log M_r=o(nr^d)$.
The LSI estimate (SC.9) gives convergence, for instance, for
$r=n^{-b}$ with $0<b<1/(2d+2)$ and uniform $C_*,\Lambda,q_*,v_*$.
These are quantitative geometric sampling consequences of the available
analytic inequalities. Conditional versions require the metric, net, and
reference law to be fixed by the conditioning data and the joint-law
hypotheses to hold after that conditioning.

:::{prf:lemma} Coverage for a metric computed from the same sample
:label: lem-scutoid-adaptive-coverage

Let $g_n$ be an adaptive metric depending on the sites. Suppose every realized
metric satisfies $g_n\preceq\Lambda I$ on the Euclidean $2r$ neighborhood
of an observation region $K$, contained in its coordinate domain. If the
Euclidean fill distance on $K$ is at most $2r$, its $g_n$ fill distance is
at most $2\sqrt\Lambda r$. Consequently any of (SC.7)--(SC.9), applied
to fixed Euclidean balls and the actual coordinate law, supplies the
corresponding coverage probability for this adaptive metric.
:::

:::{prf:proof}
For each $x\in K$, select a site at Euclidean distance at most $2r$.
The straight segment to that site lies in the tested neighborhood and has
$g_n$ length at most $\sqrt\Lambda$ times its Euclidean length. The
intrinsic $g_n$ distance is at most this admissible length. Take the
supremum over $x$. This is a pathwise implication using the same realized
metric and sites, so it transfers the probability bound without conditioning
on the random metric or asserting independent sites after that conditioning.
:::

Density lower bounds are obtained on their stated sets. For example,
{prf:ref}`lem-gaussian-kernel-lower-bound` keeps the mass of a bounded source
set and its distance to the tested region. A positive continuous QSD density
from {prf:ref}`lem-qsd-strict-positivity` has a local positive minimum;
controlling it uniformly along $n$ or a bandwidth sequence requires uniform
kernel and survival bounds. A probability density cannot be bounded below
by a positive constant on a space of infinite volume.

Fill distance controls how closely the sites cover $K$. Separation, simplex
shape, and protection from Delaunay degeneracy are additional quantities.
For a law with a positive continuous joint density, arbitrarily close sites
have positive probability. Thus such density estimates do not give a
deterministic separation or a uniformly bounded aspect ratio. Nor do
finite-dimensional propagation of chaos or the estimates above identify the
local rescaled point process as Poisson. The latter identification requires
its own limit theorem.

(sec-scutoid-cell-definition)=
## 3. Slabs, lengths, and moving-cell realizations

:::{div} feynman-prose
There are two clocks to keep straight. Recorded time orders the frames.
Proper time measures a timelike path after a Lorentzian metric has been
specified. A positive spatial metric tells us how far apart sites are on one
frame; it does not identify the two clocks. This matters most when a cloning
update moves a slot across a large spatial distance in one recorded step.
:::

:::{prf:definition} Frozen slab metrics
:label: def-scutoid-slab-metric

On a slab $[t_k,t_{k+1}]$, choose a spatial reconstruction $g_{h,k}$ from a
specified recorded state. A state after cloning and before kinetic evolution
can be used when recorded; otherwise state the chosen available-frame rule.
It is a representative state, not necessarily a temporal midpoint. Define

$$
 G_h=-c^2dt^2+g_{h,k},\qquad
 \overline G_h=c^2dt^2+g_{h,k},\qquad c>0.
\tag{SC.10}
$$

The first is a supplied Lorentzian slab metric and the second a positive
product metric. Both use the same spatial tensor. Choosing $c$ to be an
algorithmic speed parameter identifies its units; causal admissibility of the
actual recorded displacements must still be checked. Convergence to
$G=-c^2dt^2+g_t$ requires consistency of $g_{h,k}$ with that specified $g_t$.
:::

:::{prf:definition} Spatial travel cost, positive length, and causal reachability
:label: def-scutoid-path-length

For an absolutely continuous path $\gamma:[t_a,t_b]\to M$, set

$$
 L_{\rm sc}(\gamma)=\int_{t_a}^{t_b}|\dot\gamma|_{g_h}\,dt,
 \qquad d_{\rm sc}(e_a,e_b)=\inf_{\gamma:x_a\to x_b}L_{\rm sc}(\gamma).
\tag{SC.11}
$$

This is an endpoint spatial travel cost with assigned start and end times.
The positive spacetime length is instead
$\int_{t_a}^{t_b}\sqrt{c^2+|\dot\gamma|_{g_h}^2}\,dt$.
Define $e_a\prec_{\rm sc}e_b$ by $t_a<t_b$ and the existence of such a
path satisfying $|\dot\gamma(t)|_{g_h}\le c$ almost everywhere.
For a timelike path its proper-time duration is

$$
 \tau_{\rm prop}(\gamma)
 =\int_{t_a}^{t_b}\sqrt{1-|\dot\gamma|_{g_h}^2/c^2}\,dt.
\tag{SC.12}
$$

A jump is not an absolutely continuous causal path. A causal interpolation
between its endpoints is a separate geometric choice.
:::

On a complete static spatial metric, the causal criterion reduces to
$d_g(x_a,x_b)\le c(t_b-t_a)$: necessity follows by integration, and
sufficiency by constant-speed traversal of a minimizing geodesic.
For a varying metric, (SC.11) alone cannot replace the pointwise speed bound.
For example, in one dimension with $c=1$ and $t\in[0,1]$, take
$g_t=dx^2$ for the first half and $g_t=4dx^2$ for the second. Causal motion
covers at most $1/2+1/4=3/4$ in coordinate distance. Yet a displacement
$9/10$ has spatial travel cost $9/10\le1$ by moving entirely during the
first half at speed $9/5$. Thus the integrated cost test would accept an
endpoint pair with no causal path.

:::{prf:proposition} Slab consistency and the scope of causal convergence
:label: prop-scutoid-cst-compatibility

On the common domain of admissible paths suppose
$g_t\succeq\lambda I$ and
$\sup_{t,x}\|g_h(t,x)-g_t(x)\|_{\rm op}\le\eta_h\to0$.
Put $\delta_h=\eta_h/\lambda<1$. Then every such path satisfies

$$
 \sqrt{1-\delta_h}\,L_g(\gamma)
 \le L_{\rm sc}(\gamma)
 \le\sqrt{1+\delta_h}\,L_g(\gamma).
\tag{SC.13}
$$

The same comparison holds for the infima and for the positive spacetime
lengths. For the minimal required speed
$R_g(e_a,e_b)=\inf_\gamma\operatorname*{ess\,sup}_t
|\dot\gamma|_{g_t}$, the analogous comparison also holds. Hence endpoint
pairs with $R_g<c$ are eventually causally reachable in the slabs, and pairs
with $R_g>c$ are eventually unreachable. At $R_g=c$, boundary membership
requires additional attainment and boundary control.

For a fixed path uniformly timelike for both metrics with
$|\dot\gamma|_g^2,|\dot\gamma|_{g_h}^2\le c^2(1-\zeta)$,
$\zeta>0$, its two proper-time integrals differ by at most
$\eta_h(2c^2\sqrt\zeta)^{-1}\int|\dot\gamma|^2dt$.
:::

:::{prf:proof}
The operator bound gives
$(1-\delta_h)g\preceq g_h\preceq(1+\delta_h)g$.
Take square roots of its quadratic-form bounds and integrate to prove
(SC.13). For the positive product metrics the unchanged $c^2dt^2$ term
obeys the same comparison. Infimizing preserves both inequalities, without
requiring minimizing curves to exist. Taking essential suprema before
infimizing proves the required-speed comparison. If $R_g<c$, choose a path
with speed norm strictly below $c$ and then take $h$ small; if $R_g>c$, the
lower bound excludes every speed-capped slab path. Finally, rationalizing the
difference of the square roots in (SC.12) bounds its absolute value by
$\eta_h|\dot\gamma|^2/(2c^2\sqrt\zeta)$ and proves the proper-time bound.
:::

For instance, if $g_t$ is uniformly Lipschitz in time and the sampled tensor
approximates $g_{s_k}$ with error $e_h$, where $s_k\in[t_k,t_{k+1}]$, then
$\eta_h\le e_h+L_t\max_k(t_{k+1}-t_k)$. A finite set of genuine metric
jumps can be aligned with slab boundaries and treated one-sidedly. A sequence
of unaligned cloning jumps need not satisfy a uniform time-continuity bound.
Mean-field convergence alone supplies neither this tensor estimate nor causal
reachability. Faithful embedding additionally retains the order-reflection
and sampling hypotheses of {prf:ref}`thm-fractal-faithful-embedding`.

### 3.1. Boundary correspondences and their geometric requirements

:::{prf:definition} Boundary correspondence
:label: def-boundary-correspondence-map

For shared neighbors $j\in\mathcal N_i(t_a)\cap\mathcal N_i(t_b)$, let
$\mu_0,\mu_1$ be the finite positive interface area measures at the two
endpoints, with masses $A_0,A_1$. A normalized transport correspondence is a
coupling of $\mu_0/A_0$ and $\mu_1/A_1$. Such a coupling always exists:
the product of these two probabilities is one.

A geometric boundary correspondence additionally specifies a continuous map
$\phi_j:\Gamma_{ij}(t_a)\to\Gamma_{ij}(t_b)$, or a homeomorphism when
needed, compatible with neighboring face maps on their common boundaries.
It preserves unnormalized area only if
$(\phi_j)_*\mu_0=\mu_1$, which requires $A_0=A_1$.
A measurable coupling alone is not an embedded lateral face.
:::

:::{prf:lemma} Area normalization and an explicit interval correspondence
:label: lem-scutoid-interface-correspondence

A map preserving finite measures must preserve their total masses. For two
line segments of lengths $A_0,A_1>0$, the affine arclength map
$s\mapsto(A_1/A_0)s$ preserves their normalized arclength probabilities.
It preserves their unnormalized arclength measures exactly when $A_0=A_1$.
:::

:::{prf:proof}
Evaluate the pushforward identity on the whole target space to obtain the
necessary equality of masses. On segments, substitution
$u=(A_1/A_0)s$ gives
$\int_0^{A_0}f((A_1/A_0)s)ds/A_0=\int_0^{A_1}f(u)du/A_1$.
Without the normalizing denominators, the pushforward density is $A_0/A_1$.
:::

Optimal transport between interface measures, when used, needs hypotheses
appropriate to these lower-dimensional measures. Absolute continuity with
respect to interface area is not absolute continuity in the ambient volume,
and does not by itself imply a unique smooth transport map. Even a chosen
smooth map requires nonintersection and compatibility of all its rulings to
bound a cell.

:::{prf:definition} Swept Voronoi cells and scutoid transitions
:label: def-scutoid-cell

Choose a time interpolation of the sites and metric, with an explicit policy
at birth, death, and cloning jumps. The swept region of label $i$ is

$$
 \mathcal S_i=
 \overline{\{(t,x):t_a<t<t_b,\ x\in\operatorname{Vor}_i(t)\}}.
\tag{SC.14}
$$

If clipping is used, intersect with $[t_a,t_b]\times K$.
Assume a regular stratified cell realization when speaking of faces, edges,
and vertices. A *labeled prismatic interval* has a face-preserving product
trivialization over time that preserves every neighbor label. A *scutoid
transition* is a change in that labeled cell structure. A bounded
piecewise-linear realization can be called a scutoid polytope when it is
indeed a polytope; a general swept cell can be curved, nonconvex, or fail to
be a topological ball.

The cell label follows the recorded walker slot. The parent identifier at a
clone event supplies a different genealogical relation. An instantaneous
replacement can be represented by two one-sided swept regions and horizontal
caps at the jump. Interpolating the replaced slot continuously through space
is an additional visualization rule.
:::

For a globally compatible isotopy $\Phi_t$ of the spatial region with
$\Phi_t(\operatorname{Vor}_i(t_a))=\operatorname{Vor}_i(t)$ for all $i$,
the map $(t,x)\mapsto(t,\Phi_t(x))$ constructs the swept partition and its
lateral faces. It is a homeomorphism with inverse
$(t,y)\mapsto(t,\Phi_t^{-1}(y))$, so no overlap is introduced into cell
interiors. Geodesic rulings in the positive slab metric can instead be used
when they define such a compatible embedding. For a fixed product slab,
its geodesics have affine time coordinate; existence, uniqueness, and
noncrossing of the spatial rulings remain the relevant conditions.

:::{div} feynman-prose
Picture two cells that both have five faces, but one neighbor changes from
label E to label F. The polygons can still have exactly the same unlabeled
shape. What fails is the demand that each labeled wall continue unchanged
through the whole interval. A transition must occur somewhere in that labeled
structure. Its location and geometry depend on the interpolation; the endpoint
lists do not put a special vertex at the temporal midpoint.
:::

(sec-cloning-topological-transitions)=
## 4. Neighbor changes and their probability

:::{prf:theorem} Neighbor changes obstruct a labeled product cell
:label: thm-cloning-implies-scutoid

The following statements hold for the specified cell reconstruction.

1. In a continuous, nondegenerate family whose face incidences are locally
   constant throughout the closed interval, the neighbor lists are constant.
   A supplied compatible face isotopy then gives labeled prismatic cells.
2. If $\mathcal N_i(t_a)\ne\mathcal N_i(t_b)$, there is no
   neighbor-label-preserving product trivialization of $\mathcal S_i$ over
   that interval. Any stratified interpolation realizing both endpoint
   incidences has a transition. Under an isolated generic planar Delaunay
   flip, this is the familiar intermediate scutoid vertex. General jumps or
   higher-dimensional events need not have that particular realization.
3. For a clone event, condition on specified endpoints and other data.
   Suppose unchanged neighbors imply that a specified measurable region $A$
   contains no other site. If the remaining sites form a Poisson process of
   intensity measure $\Lambda$ and $A$ is fixed under that conditional law,
   then
   $\mathbb P(\mathcal N_i(t_a)=\mathcal N_i(t_b))\le e^{-\Lambda(A)}$.
   In particular, $\Lambda(A)\ge c_0\varrho r^d$ gives an exponential
   upper bound $e^{-c_0\varrho r^d}$, where $r$ is the specified clone
   displacement. The implication defining $A$ and the conditional Poisson
   law are additional geometric and stochastic hypotheses.
:::

:::{prf:proof}
A locally constant finite incidence pattern is constant on a connected time
interval. In Euclidean general position this can be checked through strict
orientation and in-sphere predicates: continuous motion keeps their signs
until a zero is reached. A supplied isotopy gives the product realization
as explained after (SC.14).

A label-preserving product sends each lateral face labeled $j$ at the bottom
to a face with the same label at the top, and conversely. It therefore forces
equality of the neighbor sets, contradicting the second hypothesis. Equality
of the number of neighbors would only allow an unlabeled bijection and
does not repair that contradiction. In a stratified realization the change
must be located on a transition stratum. This proves the obstruction, without
asserting a unique filling or a fixed event time.

For the last statement the equality event is contained in the void event
$\{\#A=0\}$. A Poisson count with mean $\Lambda(A)$ is zero with
probability $e^{-\Lambda(A)}$. Conditioning first makes clear which
information fixes $A$ and which randomness defines that count.
:::

For independent sites of law $\rho$, the last void probability is instead
$(1-\rho(A))^m\le e^{-m\rho(A)}$, where $m$ is the number of remaining
sites. For a dependent conditional law with total entropy $H$ relative to
that product, the binary-entropy argument of (SC.8) gives the upper bound
$(H+\log2)/(m\rho(A))$ when $\rho(A)>0$. Thus an appropriate geometric
void implication can use the existing entropy route without positing an
exact Poisson law. If the region is selected after inspecting the same
points, a fixed-region void formula is insufficient; one needs conditional
control or a union bound over a specified candidate family.

Neither cloning nor a large displacement alone proves the void implication.
A finite nondegenerate Euclidean configuration has an open neighborhood with
the same strict predicate signs. Moving a site inside that neighborhood,
including a sufficiently small random cloning displacement, can preserve
its neighbors with positive probability. Conversely a continuous trajectory
can cross a cocircular configuration without cloning and change the graph.
Density and mean-field estimates from {doc}`../convergence_program/09_propagation_chaos`
provide the stated laws and errors under their hypotheses; they do not assert
that these geometric events occur with probability one.

(sec-cell-type-classification)=
### 4.1. Classifying recorded changes

:::{prf:definition} Endpoint change types
:label: def-scutoid-type-classification

Set $\mathcal N_{\rm lost}=\mathcal N_i(t_a)\setminus\mathcal N_i(t_b)$,
$\mathcal N_{\rm gained}=\mathcal N_i(t_b)\setminus\mathcal N_i(t_a)$,
and

$$
 \chi_{\rm scutoid}(i;[t_a,t_b])
 =|\mathcal N_{\rm lost}|+|\mathcal N_{\rm gained}|.
\tag{SC.15}
$$

Type 0 has unchanged endpoint neighbors, $\chi_{\rm scutoid}=0$. It is
prismatic only when there are no intervening incidence transitions and the
product realization conditions hold. Type 1 has one lost and one gained
neighbor. Type 2 collects the other nonzero endpoint changes, including a
single lost or gained neighbor. These are descriptive categories, not an
ordering by the integer $\chi_{\rm scutoid}$.

A transition-resolved activity also sums changes across every intermediate
event time. Its value can exceed the endpoint index when a neighbor is lost
and later regained. Neither index is an Euler characteristic.
:::

:::{prf:remark} Metric consistency and cell data
:label: rem-scutoid-metric-recovery

The cells are constructed using a supplied metric. Their lengths approximate
that metric under {prf:ref}`prop-scutoid-cst-compatibility`. Neighbor lists
alone do not determine its scale: multiplying a static spatial metric by a
positive constant leaves nearest-site comparisons unchanged while changing
all spatial lengths. Recovering a metric from cells therefore requires
additional length, volume, or transport data. The Lorentzian construction and
its faithful-embedding hypotheses remain those of
{doc}`../2_fractal_set/02_causal_set_theory`.
:::

(sec-euler-characteristic)=
### 4.2. Counting changes and preserving topology

:::{prf:proposition} Incidence-change count and Euler characteristic
:label: prop-euler-characteristic-scutoid

Let $E_k$ be the undirected facet-neighbor edges on a fixed label set at
recorded time $t_k$. Give absent labels empty neighbor lists. Then

$$
 \sum_i\chi_{\rm scutoid}(i;[t_k,t_{k+1}])
 =2|E_{k+1}\mathbin\triangle E_k|.
\tag{SC.16}
$$

For $K$ slabs define the cumulative endpoint activity
$\mathcal K_{\rm total}=\sum_{k=0}^{K-1}\sum_i\chi_{\rm scutoid}(i;k)$.
The number of cells with nonzero endpoint index is
$\sum_{k,i}\mathbf1_{\{\chi(i;k)>0\}}$, whose expectation is the sum
of the corresponding probabilities. It is not the number of cloning events.

An interior planar Delaunay flip replaces one diagonal by the other. It
contributes four to the sum in (SC.16), and preserves the Euler
characteristic of the triangulated domain.
:::

:::{prf:proof}
Each changed undirected edge appears in exactly the two endpoint neighbor
lists, proving (SC.16). Linearity of expectation proves the cell-count
formula without independence. A planar flip removes one edge and adds one
edge; its four endpoints each lose or gain one neighbor. It keeps the vertex
count, edge count, and face count fixed, so $V-E+F$ is unchanged. More
generally a subdivision of a fixed underlying region preserves its Euler
characteristic, although its neighbor graph can change. A family with an
actual topological change or a changing observation boundary requires its
own Euler calculation.
:::

Thus a single planar flip does not give a symmetric one-loss/one-gain event
for each affected cell. Endpoint categories are also sensitive to recording
frequency. No count of lateral connected faces follows solely from the
endpoint neighbor union: one label can have several interface components,
and a face can disappear and reappear between frames.

:::{prf:proposition} Planar mean degree and clone-conditioned endpoint activity
:label: prop-scutoid-planar-mean-incidence

For a planar Delaunay graph on $n\ge3$ distinct sites in general position,
$\sum_i|\mathcal N_i|\le6n-12$. If its law is label exchangeable, then
$\mathbb E|\mathcal N_i|\le6-12/n$ for each label. For two such frames,

$$
 \sum_i\chi_{\rm scutoid}(i)\le12n-24.
\tag{SC.16a}
$$

If both marginal frame laws are exchangeable and a fixed label clones with
probability $p_i>0$, its clone-conditioned endpoint index satisfies
$\mathbb E[\chi(i)\mid i\text{ clones}]\le12/p_i$.
Thus a uniform positive lower bound on these cloning probabilities gives
a population-independent bound on this particular conditional mean.
Transition-resolved activity and geometric repair costs are different
quantities.
:::

:::{prf:proof}
A simple planar graph with $n\ge3$ has at most $3n-6$ edges. For a connected
plane graph this follows from Euler's identity $n-E+F=2$ and the bound
$3F\le2E$ counting face incidences; adding edges handles disconnected
graphs. Summing vertex degrees counts every edge twice. Under
exchangeability all expected degrees coincide, so divide the deterministic
bound by $n$. Each endpoint index is at most the sum of its degrees in the
two frames, proving (SC.16a) and $\mathbb E\chi(i)\le12$.
Finally
$p_i\mathbb E[\chi(i)\mid i\text{ clones}]
\le\mathbb E\chi(i)$ because the index is nonnegative.
Exchangeability follows from the complete-kernel conditions of
{prf:ref}`thm-qsd-exchangeability` when they apply to the recorded law.
:::

This direct planar estimate requires no Poisson model or deterministic
separation of the sites. It controls first moments of degrees and endpoint
changes. It does not bound higher degree moments, repeated transitions between
frames, or the number of flips performed by a particular repair procedure.

### 4.3. Moving interfaces and volume change

:::{prf:lemma} Cell-volume balance with a moving metric
:label: lem-scutoid-volume-balance

On a regular time interval let $\Omega_i(t)$ be a cell with positive finite
volume $V_i(t)$, piecewise smooth boundary, and boundary normal velocity
$w\cdot n$. Then

$$
 \dot V_i=
 \int_{\partial\Omega_i}w\cdot n\,dA_{g_t}
 +\frac12\int_{\Omega_i}\operatorname{tr}_{g_t}(\partial_tg_t)\,dV_{g_t}.
\tag{SC.17}
$$

For a regular Voronoi interface $\psi_{ij}(x,t)=0$,
$w\cdot n=-\partial_t\psi_{ij}/|\nabla\psi_{ij}|_{g_t}$ with
$n=\nabla\psi_{ij}/|\nabla\psi_{ij}|_{g_t}$.
Here $\partial_t\psi_{ij}$ includes both site motion and metric motion.
:::

:::{prf:proof}
Pull back the cell integral by a flow extending its boundary velocity.
Differentiation of the flow Jacobian gives its divergence; integration by
parts gives the boundary flux. Differentiating the metric volume factor gives
$\partial_t\sqrt{\det g}=\frac12\sqrt{\det g}\operatorname{tr}_g\partial_tg$.
This proves (SC.17), recovering {prf:ref}`appx-reynolds-transport` with its
metric term. Differentiating $\psi_{ij}(x(t),t)=0$ gives
$\partial_t\psi_{ij}+\langle\nabla\psi_{ij},w\rangle_g=0$, proving the
normal-velocity formula of {prf:ref}`appx-voronoi-boundary-velocity`.
:::

For a fixed metric and a comparison velocity field $u$, set
$E_i=\int_{\partial\Omega_i}(w-u)\cdot n\,dA$. The divergence theorem
then gives the exact normalized relation

$$
 \frac{\dot V_i}{V_i}
 =\frac1{V_i}\int_{\Omega_i}\operatorname{div}_gu\,dV_g+\frac{E_i}{V_i}.
\tag{SC.18}
$$

The cell-average term approximates the divergence at its site with error at
most $\operatorname{diam}(\Omega_i)\|\nabla\operatorname{div}_gu\|_\infty$,
as in {prf:ref}`appx-divergence-remainder`. Transferring a differentiated
expansion equation requires bounds on both $E_i/V_i$ and its time derivative,
along with the material-transport comparison in
{prf:ref}`appx-discrete-raychaudhuri`. An area estimate of order
$\epsilon^{d-1}$ must be divided by a volume of order $\epsilon^d$; this
normalization can remove a power of smallness. At cloning jumps the
piecewise volume balance also contains its jump increments. These statements
supply the inputs used in {doc}`03_curvature_gravity`; they do not assert
that arbitrary reconstructed cells follow a geodesic congruence.

(sec-dynamic-delaunay-algorithm)=
## 5. Maintaining the geometric reconstruction

:::{div} feynman-prose
A dynamic triangulation is useful because much of yesterday's geometry may
still be valid today. But “small movement” needs a scale. A tiny displacement
can flip a nearly cocircular quadrilateral. We therefore count the operations
that actually occur and state the conditions under which their average cost
stays small.
:::

The following algorithms describe geometric postprocessing of recorded states.
They do not change companion selection, cloning, or kinetic evolution.
Their Euclidean form also applies to one fixed positive constant metric after
a linear change of coordinates. A varying Riemannian metric requires a
geometric triangulation method with its own distance predicates and validity
proof, or an explicitly controlled flat approximation.

:::{prf:algorithm} Online triangulation from recorded changes
:label: alg-online-triangulation-update

**Input:** A valid planar Euclidean triangulation, current coordinates and
label-to-vertex map, and the next recorded coordinates, alive statuses,
cloning indicators, and parent identifiers. Sites are distinct and the
triangulation predicates satisfy the chosen general-position convention.

**Initialization:** Build the Delaunay triangulation of the initial included
sites and its Voronoi dual. Account for its construction cost separately.
The standard planar batch construction takes $O(n\log n)$.

**Update procedure:**

1. Read the recorded changes. Classify slot replacements using the cloning
   indicator and parent record, and retain every positional and status change.
   Metric changes are also inputs to the reconstruction.
2. For each removed or replaced site, delete its vertex and retriangulate the
   resulting cavity by a valid planar triangulation routine. Include the
   deletion, cavity triangulation, and predicate costs in its repair cost.
3. Locate each inserted position, using a surviving parent location or a
   neighboring old cell as a hint when available. Insert it using a valid
   cavity construction or a containing-simplex split, and restore the
   Delaunay property. Handle convex-hull changes explicitly.
4. A moved vertex may use a cheaper local update only when that update
   preserves an embedded triangulation and checks every potentially changed
   predicate. Otherwise process its movement by deletion and insertion.
   Assigning new coordinates to an old vertex without this check can invert
   triangles or make edges cross.
5. Refresh all affected dual cells, their interfaces, and the label map.
   Store the changes in the incidence graph. A full export of every simplex
   or cell is charged separately from an incremental update.

Degenerate or coincident sites require an explicit tie or multiplicity
convention. A changed metric can change predicates even when no site moves;
its distance evaluations and global or local repair work must be included.
:::

Under a kinetic model with $\dot x=v$, spatial paths between cloning events
are absolutely continuous. For example,

$$
 \mathbb E|x_{t+h}-x_t|^2
 \le h\int_t^{t+h}\mathbb E|v_s|^2ds
 \le h^2V_*\quad\hbox{if }\sup_s\mathbb E|v_s|^2\le V_*.
\tag{SC.19}
$$

This is Cauchy–Schwarz followed by integration. It gives a typical spatial
scale of order $h$ in that kinetic model; velocity diffusion itself has a
noise increment of order $\sqrt h$. A model with direct position diffusion
has a different displacement estimate. The implemented split transition
uses its recorded positions and the appropriate estimates of
{doc}`../convergence_program/13_quantitative_error_bounds`.
Even (SC.19) does not bound a Delaunay repair count without protection from
near-degenerate predicates.

(sec-lawson-flip)=
### 5.1. Planar Lawson restoration

:::{prf:algorithm} Lawson flips on a valid planar triangulation
:label: alg-lawson-flip

**Input:** A valid planar triangulation of fixed distinct sites in general
position. All possibly illegal interior edges are placed in a queue; scanning
all interior edges is a valid initialization.

**Procedure:** Remove an edge from the queue and check its current incident
triangles. If the union is a convex quadrilateral and the opposite point is
strictly inside the other triangle's circumcircle, replace the diagonal by
the other diagonal. Requeue every edge whose adjacent triangles changed.
Stale queue entries are checked against the current incidence records.
Continue until the queue is empty.

**Output:** The Euclidean Delaunay triangulation. In particular, a permanently
marked “already visited” triangle must not suppress a later check after its
neighbors have changed.
:::

:::{prf:lemma} Termination and the cost of actual flips
:label: lem-scutoid-lawson-termination

Under the preceding planar hypotheses, Lawson flips terminate. If $Q_0$
queue entries are initially inserted and $F$ flips occur, an implementation
with constant-time incidence updates and predicates costs $O(Q_0+F)$.
This statement gives no bound $F=O(1)$ or $F=O(k)$ solely from the size of
an initially inspected star.
:::

:::{prf:proof}
For a planar triangulation $\mathcal T$, let $u_\mathcal T$ be the
piecewise-affine interpolation of the lifted heights $|z_i|^2$ over its
triangles. An illegal diagonal is locally nonconvex in this lifting. The
other diagonal replaces its lifted patch by the lower convex patch of the
same four vertices. Hence $u_\mathcal T$ decreases pointwise on that
quadrilateral and decreases strictly on an open subset. Consequently
$\int_{\operatorname{conv}Z}u_\mathcal T(x)dx$ strictly decreases.
There are only finitely many triangulations of the fixed finite point set,
so the process terminates.

At termination every interior edge is locally Delaunay. The lifted
piecewise-affine surface is then locally convex across every edge and hence
convex on the convex hull: along a line crossing edges transversely, its
piecewise-constant slope never decreases; continuity extends this to all
lines. Its facets are therefore lower supporting facets. The lifting proof
of {prf:ref}`lem-scutoid-euclidean-delaunay` identifies the result as
Delaunay. Each flip changes a bounded number of planar adjacencies and
inserts a bounded number of queue entries, giving $O(Q_0+F)$ work.
:::

This lifting proof is the classical one described in the
[ETH computational geometry notes](https://geometry.inf.ethz.ch/gca18-5.pdf).
Its planar termination argument is not a proof that arbitrary higher-dimensional
flips, geodesic flips, or an unchecked star retriangulation will succeed.
Their algorithms and costs must be supplied separately.

(sec-jump-and-walk)=
### 5.2. Point location from a geometric hint

:::{prf:algorithm} Straight-segment walk with a hint
:label: alg-jump-and-walk

**Input:** A valid simplicial triangulation, a query point $z$, and an interior
point $q$ in a known hint simplex. Assume the segment from $q$ to $z$ crosses
faces transversely and avoids vertices and lower-dimensional faces, or use an
explicit rule for such simultaneous crossings.

**Procedure:** Parameterize $q+s(z-q)$, $0\le s\le1$. In the current
simplex find the first exit face with parameter strictly larger than the
previous crossing parameter. Cross to its neighbor and continue. Stop when
$z$ lies in the current simplex or the segment leaves the triangulated
region. The jump stage is the selection of the initial hint; its search and
maintenance costs are counted.

The cost is the hint cost plus the number of crossed simplices. An arbitrary
“face beyond the query” rule is replaced here by the increasing segment
parameter, which gives a termination criterion.
:::

:::{prf:lemma} A walk bound under actual shape control
:label: lem-scutoid-walk-bound

Suppose simplices along the query segment have diameter at most $C\ell$
and volume at least $c_0\ell^d$, with disjoint interiors in Euclidean
coordinates. A segment of length $R$ crosses at most
$C'(1+R/\ell)$ simplices, where $C'$ depends on $d,C,c_0$.
For a hint-to-query displacement with
$\mathbb E R\le\sigma\sqrt d$, the expected number is at most
$C'(1+\sigma\sqrt d/\ell)$ under the same uniform shape control.
:::

:::{prf:proof}
Every crossed simplex lies in the $C\ell$ neighborhood of the segment.
This tube has volume at most
$C_d(R(C\ell)^{d-1}+(C\ell)^d)$. Since simplex interiors are disjoint
and each has volume at least $c_0\ell^d$, divide these volumes to obtain
the first bound. Take expectations for the second. A centered Gaussian
jitter with covariance $\sigma^2I$ satisfies
$\mathbb E|\xi|\le\sqrt{\mathbb E|\xi|^2}=\sigma\sqrt d$.
The shape hypotheses concern the triangulation actually traversed; Gaussian
jitter alone does not supply them.
:::

For arbitrary configurations, the number of crossed simplices can be large.
A logarithmic point-location claim requires an identified data structure
with that bound and a budget for maintaining it. A spatial index or a parent
hint by itself is not a complexity theorem.

(sec-complexity-analysis)=
## 6. Conditional complexity and output costs

:::{prf:theorem} Operation-count and expected update bounds
:label: thm-amortized-complexity

Let $C_k$ be the number of cloned slots on a step. Let $R_i$ denote the
actual geometric repair cost of update $i$, and $L_i$ its point-location
cost when required. Let $M_k$ include metric evaluations, index maintenance,
and additional global repairs. For the stated reconstruction procedure,

$$
 T_k=O\!\left(N+M_k+\sum_i R_i+\sum_{i:\,\mathrm{located}}L_i\right).
\tag{SC.20}
$$

Suppose the continuous-move repairs and their location costs have total
expected cost $O(N)$, cloned-slot repair costs have total expected cost
$O(\mathbb EC_k)$, each clone's location cost is at most $O(\log N)$ in
conditional expectation, and $\mathbb EM_k=O(N)$. Then, writing
$p_k=\mathbb EC_k/N$,

$$
 \mathbb ET_k=O(N+p_kN\log N).
\tag{SC.21}
$$

In particular $p_k=O(1/\log N)$ gives an expected linear update cost.
If the hypotheses are bounds on totals over many steps, the same calculation
gives an amortized bound over those steps. A full export adds its output cost.
:::

:::{prf:proof}
The initial record scan costs $O(N)$ in fixed dimension. Add the actual
repair, search, metric, and maintenance costs to get (SC.20). For a clone
indicator $I_i$, conditioning on $I_i=1$ bounds
$\mathbb E[I_iL_i]\le C\log N\,\mathbb P(I_i=1)$.
Sum over $i$ and use $\sum_i\mathbb P(I_i=1)=\mathbb EC_k=Np_k$.
Apply the other assumed total-cost bounds to obtain (SC.21). No independence
of cloning indicators is required. Summing the same identities over time
proves the amortized version.
:::

The sampling bounds (SC.7)--(SC.9) establish coverage under their stated
laws. They do not establish bounded repair costs near degeneracies, a
linear-size intrinsic Delaunay complex in every dimension, or cheap updates
of an adaptive metric. Those hypotheses in (SC.21) must be proved for the
chosen reconstruction or measured as operation counts. A fixed positive
cloning fraction leaves an $N\log N$ term as $N\to\infty$ in this bound;
a small numerical coefficient is not an asymptotic linear-time guarantee.

(sec-lower-bound-proof)=
### 6.1. What the output lower bound actually measures

:::{prf:theorem} Explicit-output and dynamic-update lower bounds
:label: thm-omega-n-lower-bound

In a sequential word-RAM model with fixed dimension, writing a fresh complete
triangulation representation with $n$ vertices and $s$ simplex records costs
$\Omega(n+s)$. Thus an $O(N)$ update followed by a complete export is
output-optimal when $n+s=\Theta(N)$ and the hypotheses of (SC.21) hold.
For an incremental representation, the corresponding output lower bound is
only the number of records explicitly changed or emitted; an unchanged
triangulation need not be rewritten.
:::

:::{prf:proof}
Each vertex or simplex record requires at least one word write, and a step
writes only a bounded number of words. This proves the complete-output
bound. An incremental procedure can retain unchanged records, so that
argument counts only its changed or emitted records. For example, storing
a global rigid transformation lazily can describe rotated coordinates while
retaining the same combinatorial triangulation. It gives no reason to visit
every simplex. A procedure that reads all $N$ new coordinates still incurs
its explicit $\Omega(N)$ input cost.
:::

(sec-topological-information-rate)=
## 7. Incidence activity and information in the record

:::{div} feynman-prose
Counting changed neighbors tells us how much the geometric graph reorganized.
It is a useful observable, but its units are changes per unit time. To count
bits, we also need to know how surprising those changes were. And a stable
neighbor graph can accompany substantial motion or improvement in fitness.
Those are different measurements of the computation.
:::

:::{prf:definition} Recorded topological activity rate
:label: def-topological-information-rate

For a recorded interval of length $T$, define

$$
 \dot I_{\rm topo}^{\rm record}
 =\frac1T\sum_{k,i}\chi_{\rm scutoid}(i;k).
\tag{SC.22}
$$

This retains the name *topological information rate* as an incidence-count
observable, with units of oriented neighbor changes per unit time. If every
resolved transition is counted, use its transition-resolved sum instead.
A clone-only version sums only the explicitly designated clone-associated
changes. For $C>0$ recorded clone events, it factors exactly as
$(C/T)(C^{-1}\sum_{\rm clones}\chi)$. For $C=0$ its value is zero.
A homogeneous expected clone frequency is $Np_{\rm clone}/\Delta t$;
the corresponding expected activity uses the conditional mean index per
clone, rather than the unconditional per-cell mean.
:::

:::{prf:proposition} Activity bound from event incidence
:label: prop-scutoid-event-incidence

Suppose an observed spacetime region of volume $V_{\rm st}$ and duration
$T$ contains $J$ resolved transition events, each contributing at most
$m_*$ oriented neighbor changes. Put $\rho_{\rm event}=J/V_{\rm st}$.
Then

$$
 \dot I_{\rm topo}^{\rm record}
 \le\dot I_{\rm topo}^{\rm resolved}
 \le m_*\frac{V_{\rm st}}T\rho_{\rm event}.
\tag{SC.23}
$$

For a fixed spatial observation region of volume $V$ and lapse $c$,
$V_{\rm st}=cTV$, so the last bound is $m_*cV\rho_{\rm event}$.
If all events are isolated interior planar Delaunay flips on a fixed label
set, the resolved activity equals $4J/T$.
:::

:::{prf:proof}
For finite sets the symmetric-difference distance satisfies the triangle
inequality. Apply it to every neighbor list between recorded endpoints to
bound its endpoint change by the sum over intermediate events. Summing over
sites gives the first inequality. The incidence bound gives a resolved total
at most $m_*J$, proving the second. The volume formula follows from
$dV_G=c\,dt\,dV_g$ for this fixed product geometry. The planar flip count
is exactly four by (SC.16).
:::

:::{prf:theorem} Uniform incidence under protected local events
:label: thm-uniform-event-incidence

Consider a family of reconstructed configurations indexed by population
$N$, with local mesh scale $\varepsilon_N>0$. Assume that every resolved
event $e$ satisfies the following conditions, with constants independent of
$N$ and $e$.

1. **Metric regularity and anti-degeneracy.** On the event neighborhood, the
   reconstruction metric has a uniform doubling bound with constant
   $C_{\mathrm{dbl}}$.
   Distinct active sites satisfy the anti-degeneracy (hard-core) estimate

   $$
   d_g(z_i,z_j)\ge a\,\varepsilon_N,\qquad i\ne j,
   \tag{SC.23a}
   $$

   for some $a>0$.
2. **Locality.** All endpoint labels of changed edges lie in a metric ball
   $B_g(x_e,R\varepsilon_N)$. This holds, for example, when at most $q_*$
   labels move by at most $r_*\varepsilon_N$, the adaptive metric update is
   supported in that ball, and every predicate outside the ball has a strict
   margin larger than the corresponding perturbation.
3. **Resolved-event budget.** The event is decomposed into at most $F_*$
   local predicate transitions. A transition is counted once, even if the
   endpoint graph is recorded only at the beginning and end of the event.

Then there is a constant

$$
 M_* \;=\; C_{\mathrm{dbl}}\left(1+\frac{2R}{a}\right)^{d_*},\qquad
 m_* \;=\; F_*\,M_*^2,
 \tag{SC.23b}
$$

where $d_*$ is the doubling dimension, such that the event incidence index
$\chi_e$ obeys

$$
 \mathbb E[\chi_e\mid e\text{ is resolved},\mathcal F_e]\le m_*
 \quad\text{almost surely}.
 \tag{SC.23c}
$$

Consequently, if $J_N$ events occur in a region of spacetime volume
$V_{\rm st}$ and duration $T$, then

$$
 \mathbb E\dot I_{\rm topo}^{\rm resolved}
 \le m_*\,\frac{\mathbb E J_N}{T}
 =m_*\,\frac{V_{\rm st}}T\,\rho_{\rm event}^{(N)},\qquad
 \rho_{\rm event}^{(N)}:=\frac{\mathbb E J_N}{V_{\rm st}}.
 \tag{SC.23d}
$$

The constant is independent of population and resolution. In the planar
fixed-label case, the existing exchangeable estimate
{prf:ref}`prop-scutoid-planar-mean-incidence` supplies the corresponding
endpoint bound without the hard-core assumption. The assumptions above are
the additional protection needed for transition-resolved activity and for
adaptive metrics.
:::

:::{prf:proof}
Let $A_e$ be the set of labels appearing at an endpoint of a changed edge.
Locality places all sites in $A_e$ inside $B_g(x_e,R\varepsilon_N)$.
The disjoint balls of radius $a\varepsilon_N/2$ centered at these sites fit
inside a ball of radius $(R+a/2)\varepsilon_N$. The doubling estimate
therefore gives

$$
 |A_e|\le C_{\mathrm{dbl}}\left(1+\frac{2R}{a}\right)^{d_*}=M_*.
 \tag{SC.23e}
$$

A single transition can change at most $|A_e|^2$ oriented incidences. The
resolved-event budget then gives $\chi_e\le F_*M_*^2=m_*$ pathwise. Taking
the conditional expectation proves (SC.23c). Summing this bound over the
$J_N$ events and dividing by $T$ gives (SC.23d), followed by the definition
of $\rho_{\rm event}^{(N)}$.

The hard-core and predicate-margin hypotheses are genuine geometric inputs:
moment or entropy bounds alone do not exclude nearly coincident sites or
nearly degenerate Delaunay predicates. Once they are verified for a scaling
family, the result is a theorem rather than an assumed uniform incidence
property.
:::

This formulation keeps the spatial-volume factor required by dimensions.
A density of events per spacetime volume multiplied only by a speed does not
have units of total events per time. It also distinguishes a resolved event
from a hypothetical vertex placed at the midpoint of a slab.

:::{prf:lemma} Bits required to specify an edge-change set
:label: lem-scutoid-change-entropy

On $n$ fixed labels there are $M=\binom n2$ possible undirected edges.
If a random change set $A$ has cardinality $C$, then its base-two Shannon
entropy satisfies

$$
 H_2(A)\le H_2(C)+\mathbb E\log_2\binom MC.
\tag{SC.24}
$$

For a fixed $1\le C=m\le M$, the conditional term is at most
$m\log_2(eM/m)$. An incidence count alone does not determine this entropy.
:::

:::{prf:proof}
Since $C$ is a function of $A$, the entropy chain rule gives
$H_2(A)=H_2(C)+H_2(A\mid C)$. Conditional on $C=m$, at most
$\binom Mm$ change sets are possible, so the conditional entropy is bounded
by the logarithm of that number. The estimate
$\binom Mm\le M^m/m!\le(eM/m)^m$ gives the last bound.
For the factorial inequality, sum $\log j$ and compare with
$\int_1^m\log x\,dx=m\log m-m+1$.
:::

For example, $z_i(t)=e^{-t}z_i(0)$ preserves a Euclidean Delaunay graph and
all its neighbor lists because it is a similarity. Nevertheless
$n^{-1}\sum_i|z_i(t)|^2=e^{-2t}n^{-1}\sum_i|z_i(0)|^2$ decreases whenever
the sites are not all zero. This demonstrates that geometric incidence
activity is not a necessary measure of improvement in a quadratic objective.
Interpreting it as exploration or learning requires comparison with the
recorded fitness and other observables.

(sec-scutoid-conclusions)=
## 8. Using the construction in the next chapter

The spatial cells are well-defined nearest-site regions for the supplied
metric. Their labeled changes are quantified by (SC.15)--(SC.16), and their
regular motion obeys the volume balance (SC.17). Slab lengths and causal
reachability have the quantitative comparison in (SC.13); proper time uses
(SC.12). These constructions preserve the distinction between a recorded
clone jump and a chosen continuous cell interpolation.

The density and concentration estimates give quantitative coverage, including
the entropy-based bound (SC.8) and the LSI bound (SC.9). Local shape and
predicate protection remain the conditions needed for a regular triangulation
and the cost estimates (SC.20)--(SC.21). The curvature and expansion
calculations in {doc}`03_curvature_gravity` use the same metric, actual
connection consistency, and the normalized, differentiated volume conditions
of {prf:ref}`appx-discrete-raychaudhuri`.

(sec-scutoid-references)=
## References

The biological geometry motivating the term is described by
[Gómez-Gálvez et al., *Scutoids are a geometrical solution to three-dimensional packing of epithelia*](https://www.nature.com/articles/s41467-018-05376-1).
For planar lifting and Lawson flips, see the
[ETH computational geometry notes](https://geometry.inf.ethz.ch/gca18-5.pdf).
The extra conditions needed for intrinsic Delaunay realizations are discussed
in [Boissonnat et al., *An obstruction to Delaunay triangulations in Riemannian manifolds*](https://arxiv.org/abs/1612.02905).

The analytic results used here are proved in
{doc}`../convergence_program/09_propagation_chaos`,
{doc}`../convergence_program/11_hk_convergence`,
{doc}`../convergence_program/12_qsd_exchangeability_theory`,
{doc}`../convergence_program/14_a_geometric_gas_c3_regularity`,
{doc}`../convergence_program/15_kl_convergence`,
{doc}`../convergence_program/16_continuum_discharge`, and
{doc}`../convergence_program/17_geometric_gas`.
