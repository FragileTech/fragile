# Fractal Set as Causal Set

(sec-cst-intro)=
## 1. Finite Orders and Continuum Questions

:::{div} feynman-prose
Follow the arrows in a recorded run. Every CST arrow moves to a later
iteration. Following several arrows can never bring you back to the event
where you started. That observation gives a causal set in the mathematical
sense: a locally finite set with a strict partial order.

Now draw light cones on a smooth spacetime. The order they describe answers a
different question: which events can be joined by a causal curve? Two walkers
can lie inside one another's light cones even when their recorded worldlines
have no CST connection. The graph order and the geometric order must therefore
be kept distinct.

The finite order is established below by a direct proof. The later volume,
wave-operator, dimension, and curvature calculations use a specified continuum
geometry and sampling hypotheses. Their conclusions remain conditional when
applied to the interacting episode process.
:::

:::{prf:definition} Scope and notation
:label: def-cst-scope

Let $E$ be a finite collection of recorded episodes from the Fractal Set in
{doc}`01_fractal_set`, and let $E_{\mathrm{CST}}$ be its recorded temporal
edge set. The iteration index is $k(e)\in\mathbb Z$; a physical or algorithmic
time coordinate $t(e)$, when supplied, increases with this index. We write
$N=|E|$ for the number of sampled episodes, which need not equal the number
of walker slots.

The finite-order results use only that CST edges strictly increase $k$ and
that each iteration contains finitely many episodes. Continuum statements
add a spatial dimension $d$, a spacetime dimension $D=d+1$, an episode map
into a specified manifold, and the hypotheses explicitly named in those
statements. All tensors and densities below refer to that specified geometry.
:::

(sec-cst-axioms)=
### Causal sets and a reference sampling model

:::{prf:definition} Causal set
:label: def-causal-set-blms

A causal set is a pair $(C,\prec)$ with a strict partial order satisfying
local finiteness:

1. **Irreflexivity:** $x\not\prec x$ for every $x\in C$.
2. **Transitivity:** $x\prec y$ and $y\prec z$ imply $x\prec z$.
3. **Local finiteness:** $\{y:x\prec y\prec z\}$ is finite for every $x,z$.

These are the order conventions of {prf:ref}`def-fractal-set-cst-axioms` and
{cite}`BombelliLeeEtAl87,Sorkin05`. They impose no manifold dimension, metric,
or continuum approximation.
:::

:::{prf:definition} Poisson sprinkling into a specified region
:label: def-poisson-sprinkling-cst

Let $(M,g)$ be a time-oriented Lorentzian manifold and let $W\subset M$ have
finite geometric volume $V_W>0$. A Poisson sprinkling of intensity $\lambda_0$
per unit geometric volume is obtained by drawing
$N\sim\operatorname{Poisson}(\lambda_0V_W)$ and, conditional on $N$, drawing
independent points with probability law $V_W^{-1}d\mathrm{vol}_g|_W$.
The geometric causal relation on $M$ induces the order of the sampled points.

For measurable $A\subset W$, the expected count is
$\lambda_0\mathrm{vol}_g(A)$. This model uses geometric volume, which has
coordinate density $\sqrt{|\det g|}$; it is generally not uniform in coordinate
volume. A prescribed nonconstant intensity gives an inhomogeneous Poisson
model. Specifying only the one-point density of a different process does not
identify that process as Poisson.
:::

(sec-fractal-causal-set)=
### The recorded order

:::{prf:definition} CST order and weighted path length
:label: def-fractal-causal-order

The recorded CST order is the transitive closure

$$
e\prec_{\mathrm{CST}}e'
\quad\Longleftrightarrow\quad
\text{a directed path in }E_{\mathrm{CST}}\text{ joins }e\text{ to }e'.
$$

The edge set is specified in {prf:ref}`def-fractal-set-cst-edges`. Clone
ancestry is a separate relation
({prf:ref}`def-fractal-set-clone-ancestry`); it is not added to the CST order.

If nonnegative edge lengths $\ell_a$ are supplied, define

$$
L_g(\gamma)=\sum_{a\in\gamma}\ell_a,
\qquad d_g(e,e')=\inf_{\gamma:e\to e'}L_g(\gamma),
$$

with $d_g(e,e')=\infty$ when there is no directed path. This is a directed
path length on the graph. For a path sampled in a declared coordinate chart,
$\ell_a=\|x_{k+1}-x_k\|_{g_R(t_k,x_k)}$ is a frozen-metric quadrature rule.
Interpreting it as a continuum length requires control of that quadrature.

For a finite layer of edges with $\Delta t_a>0$, the empirical quantity

$$
c_{\mathrm{eff}}(k)=\max_{a\text{ in layer }k}\frac{\ell_a}{\Delta t_a}
$$

is finite whenever the edge lengths are finite; set it to zero for an empty
layer. Its definition gives a bound on those recorded ratios. A deterministic
bound uniform over runs, and a causal interpolation in a varying metric,
require additional conditions.
:::

:::{figure} figures/cst-growth-tree.svg
:alt: Recorded temporal arrows between finite layers of episodes.
:width: 90%
:align: center

Layered temporal arrows illustrate the strict increase of iteration labels.
Only recorded CST edges enter the order; clone genealogy is stored separately.
:::

(sec-axiom-verification)=
### Direct verification of the finite axioms

:::{prf:theorem} The recorded Fractal Set is a causal set
:label: thm-fractal-is-causal-set

Assume every CST edge strictly increases the integer iteration index and each
iteration contains finitely many episodes. Then $(E,\prec_{\mathrm{CST}})$ is
a causal set. The conclusion also holds for a history with infinitely many
integer-indexed layers, provided every layer is finite.
:::

:::{prf:proof}
A directed path strictly increases the iteration index at every step, so it
cannot return to its initial node. This proves irreflexivity. Concatenating
a path from $e$ to $e'$ with one from $e'$ to $e''$ proves transitivity.

If $k(e)=k_0<k_1=k(e')$, every episode strictly between $e$ and $e'$ lies in
one of the finitely many layers $k_0+1,\ldots,k_1-1$. If layer $k$ has $n_k$
episodes, the interval contains at most
$\sum_{k=k_0+1}^{k_1-1}n_k<\infty$ elements. This proves local finiteness.
$\square$
:::

(sec-cst-geometric-comparison)=
## 2. Comparison with a Specified Lorentzian Geometry

:::{div} feynman-prose
A speed limit is a statement about a curve at every instant. In a changing
metric, a short total path length alone does not establish that limit: a path
might travel too fast during a short favorable interval. We will therefore
define geometric causality by the existence of a curve satisfying the
pointwise speed bound. The familiar distance divided by elapsed time test is
recovered for a static product metric.
:::

:::{prf:assumption} Continuum geometry for geometric reconstructions
:label: assm-cst-continuum-geometry

For geometric statements, specify a connected spatial manifold $\mathcal X$,
an open time interval $I$, and a $C^4$ Lorentzian product metric

$$
M=I\times\mathcal X,\qquad g=-c^2dt^2+g_R(t),\qquad c>0.
$$

Assume this spacetime is globally hyperbolic. A sufficient condition is a
complete proper background Riemannian metric $h$ and constants
$0<a\le b<\infty$ with $ah\preceq g_R(t)\preceq bh$, as proved directly in
{prf:ref}`lem-continuum-a1-geometry`. This allows unbounded spaces such as
$\mathbb R^d$; no compact spatial support is inferred from a confining law.
Let $W\subset M$ be an observation region with time closure inside $I$.

Supply an episode map $\iota:E\to W$, including the time and position of each
episode. Its injectivity, any reconstruction of $g_R$, and convergence of graph
distances are separate conditions when needed. One may choose a metric field
from an inverse diffusion convention only when its positivity, two-sided
spectral bounds, and regularity have been checked; see
{prf:ref}`lem-continuum-a2-smooth-fields`.

The constant $c$ is part of this comparison geometry. Identifying it with
$V_{\mathrm{alg}}$ requires a speed estimate in this same metric for the
actual interpolation of the recorded update.
:::

:::{prf:definition} Geometric causal order and path length
:label: def-cst-geometric-order

Under {prf:ref}`assm-cst-continuum-geometry`, write $p\prec_{\mathrm{LC}}q$
when $t(p)<t(q)$ and there is an absolutely continuous spatial curve $x(t)$
joining them such that

$$
g_R(t,x(t))(\dot x(t),\dot x(t))\le c^2
\quad\text{for almost every }t\in[t(p),t(q)].
$$

The chronological relation $p\ll_gq$ is existence of a future-directed
timelike curve. These are relations of the supplied continuum metric.
They may be restricted to episode images using $\iota$.

For distinct endpoint times, retain the spatial path functional

$$
d_{\mathrm{geo}}(p,q)=\inf_x
\int_{t_-}^{t_+}\|\dot x(t)\|_{g_R(t,x(t))}\,dt,
\qquad t_-=\min(t(p),t(q)),\quad t_+=\max(t(p),t(q)),
$$

where the infimum is over absolutely continuous curves with the specified
endpoints. At equal times use the slice distance. Every causal curve gives
$d_{\mathrm{geo}}(p,q)\le c|t(q)-t(p)|$. For time-dependent $g_R$, this
integrated inequality is only a necessary test for causality.

If $g_R$ is time-independent and complete, then
$d_{\mathrm{geo}}(p,q)=d_{g_R}(x(p),x(q))$, and

$$
p\prec_{\mathrm{LC}}q
\quad\Longleftrightarrow\quad
 t(p)<t(q),\quad d_{g_R}(x(p),x(q))\le c(t(q)-t(p)).
$$

Strict inequality characterizes $p\ll_gq$ in this static case. To prove the
converse causal implication, traverse a minimizing spatial geodesic at
constant speed over the available time. The forward implication follows by
integrating the speed bound.
:::

:::{prf:proposition} Conditional compatibility of CST paths and geometric causality
:label: prop-fractal-causal-order-equivalence

Assume {prf:ref}`assm-cst-continuum-geometry`. Suppose each recorded CST edge
has an interpolation between its episode images satisfying the geometric
causal speed bound in {prf:ref}`def-cst-geometric-order`. Then

$$
e\prec_{\mathrm{CST}}e'
\quad\Longrightarrow\quad
\iota(e)\prec_{\mathrm{LC}}\iota(e').
$$

This is preservation of the order. An order embedding would also require
injectivity and reflection of the order, namely the converse implication;
that converse is an additional condition and generally fails for distinct
recorded lineages.

For a fixed $C^1$ spatial trajectory $x:[s,t]\to\mathcal X$ contained in a
coordinate chart and a continuous metric along it, the frozen-metric path
sums satisfy

$$
\sum_k\|x(t_{k+1})-x(t_k)\|_{g_R(t_k,x(t_k))}
\longrightarrow
\int_s^t\|\dot x(u)\|_{g_R(u,x(u))}\,du
$$

as the maximum partition step tends to zero. The limit is the length of
that trajectory, and is at least $d_{\mathrm{geo}}$ between its endpoints.
For a unique recorded CST path, its graph path length is that path sum.
:::

:::{prf:proof}
Concatenate the causal edge interpolations. The concatenated curve is
absolutely continuous on each finite time interval and obeys the speed bound
almost everywhere, which proves order preservation.

For the path sums, set $h_*=\max_k(t_{k+1}-t_k)$ and let
$\omega_{\dot x}$ be a modulus of continuity of $\dot x$. Then

$$
\left|\frac{x(t_{k+1})-x(t_k)}{t_{k+1}-t_k}-\dot x(t_k)\right|
\le\omega_{\dot x}(h_*).
$$

On the compact trajectory, let $b_*$ bound the largest coordinate eigenvalue
of $g_R$. Replacing each difference quotient by $\dot x(t_k)$ changes the
sum by at most $(t-s)\sqrt{b_*}\,\omega_{\dot x}(h_*)$. The resulting sum is a
Riemann sum for the continuous function
$u\mapsto\|\dot x(u)\|_{g_R(u,x(u))}$ and converges to its integral. The
infimum defining $d_{\mathrm{geo}}$ is at most this realized length.
$\square$
:::

:::{prf:remark} Finite displacement, graph distances, and interpolation
:label: rem-cst-geometry-reconstruction

The radial squashing map in {prf:ref}`def-latent-velocity-squashing` bounds a
specified velocity in its current metric. Transferring that bound to all
position increments and then to a causal curve in a time-dependent metric
requires verification of the actual update and interpolation. A curve with
speed exactly $c$ is null; strict inequality makes it timelike.

The Riemann-sum proof concerns a specified trajectory. A shortest-path
approximation on the IG graph requires sampling, connectivity, and edge-length
error estimates of its own. Graph-Laplacian convergence to an operator on a
pre-existing manifold gives a different conclusion; see
{prf:ref}`rem-continuum-graph-identification`.
:::

:::{prf:lemma} Edge accuracy and path coverage imply distance accuracy
:label: lem-cst-graph-distance-comparison

Let graph vertices lie in a specified metric space $(X,d)$, and assume every
edge weight satisfies

$$
(1-\delta)d(u,v)\le\ell_{uv}\le(1+\delta)d(u,v),\qquad 0\le\delta<1.
$$

For target vertices $p,q$, suppose a graph path joining them has the sum of
its exact metric edge distances at most $d(p,q)+\eta$. Then its graph
shortest-path distance $d_G$ satisfies

$$
(1-\delta)d(p,q)\le d_G(p,q)
\le(1+\delta)(d(p,q)+\eta),
\qquad
|d_G(p,q)-d(p,q)|\le\delta d(p,q)+(1+\delta)\eta.
$$

Thus uniform edge accuracy and path coverage with $\delta,\eta\to0$ give
distance convergence on pairs with bounded $d(p,q)$.
:::

:::{prf:proof}
On every graph path the sum of exact edge distances is at least $d(p,q)$ by
the triangle inequality. The edge lower bound therefore proves the lower
bound for the graph infimum. Apply the edge upper bound to the path in the
coverage hypothesis to obtain the upper bound. Subtract $d(p,q)$ on each
side to conclude. $\square$
:::

:::{prf:remark} Scope of the earlier shortest-path calculation
:label: rem-cst-prior-geodesic-analysis

The edge-length analysis in {doc}`../3_fitness_manifold/07_computational_proxies`
separates chord error, metric interpolation error, and path non-optimality.
The preceding lemma supplies the explicit role of its last term: accurate
local edge lengths must be accompanied by paths whose excess length tends
to zero. This condition is stronger than quasi-uniform placement of vertices.

For example, the Euclidean square grid of spacing $1/n$, with edges between
cells sharing a Voronoi side, is quasi-uniform with regular cells. All graph
edges have exact Euclidean lengths. Nevertheless its shortest-path distance
from $(0,0)$ to $(1,1)$ is $2$ for every $n$, while the Euclidean distance is
$\sqrt2$. This is the path non-optimality term that the earlier error
calculation needs to control. The comparison lemma uses the valid edge/path
decomposition without asserting coverage for an unspecified IG graph.
:::

:::{figure} figures/cst-light-cone.svg
:alt: Geometric light cones in a specified product spacetime.
:width: 85%
:align: center

Light cones refer to the supplied metric. Geometric causal relations can
connect episode images that have no recorded CST path between them.
:::

### Conformal geometry and available data

:::{prf:lemma} Continuum causal cones determine the conformal class
:label: lem-causal-order-conformal-class

Let $G$ and $G'$ be smooth Lorentzian metrics of signature $(-,+,\ldots,+)$
on the same smooth manifold of dimension $D\ge2$. Assume they have the same
time-oriented timelike tangent cones at every point. Then
$G'=\Omega^2G$ for a positive smooth function $\Omega$.

The same conclusion applies when their full continuum chronological
relations are known to recover these tangent cones and agree. For example,
on a common smooth strongly causal spacetime the cones can be read locally
from the chronological relation in arbitrarily small causally convex normal
neighborhoods. Knowledge of the finite CST order does not supply this
continuum cone information.
:::

:::{prf:proof}
Fix a point and choose a $G$-orthonormal basis $e_0,e_1,\ldots,e_{D-1}$.
The boundaries of the common timelike cones are the same null cone.
Thus $G'(e_0+e_i,e_0+e_i)=G'(e_0-e_i,e_0-e_i)=0$, giving
$G'_{0i}=0$ and $G'_{ii}=-G'_{00}$. When $i\ne j$, the null vector
$e_0+(e_i+e_j)/\sqrt2$ similarly gives $G'_{ij}=0$. Therefore
$G'=(-G'_{00})G$ in this basis. The common timelike direction has
$G'_{00}<0$, so the factor is positive. Taking this factor as the ratio of
the two metrics on any local timelike vector field proves it is smooth.

For the stated local recovery condition, in a convex normal neighborhood
the chronological future of a point is the exponential image of its future
timelike cone. The derivative of that exponential map at the origin is the
identity. Hence limiting displacement directions of this future recover the
closed future causal cone, whose interior is the timelike cone. Causal
convexity makes the local chronological relation the restriction of the
full one. This proves the claimed sufficient use of continuum order.
$\square$
:::

:::{prf:corollary} A specified geometric volume fixes the conformal factor
:label: cor-order-volume-fix-conformal

Under {prf:ref}`lem-causal-order-conformal-class`, if $G'=\Omega^2G$, then

$$
d\mathrm{vol}_{G'}=\Omega^D\,d\mathrm{vol}_G,
\qquad
\Omega=\left(\frac{d\mathrm{vol}_{G'}}{d\mathrm{vol}_G}\right)^{1/D}.
$$

Consequently equal geometric volume forms and equal continuum cones imply
$G'=G$. Applying this identification to sampled data requires recovery of
the continuum cones and the normalized geometric volume measure; raw counts
from an unknown nonuniform law supply neither condition.
:::

:::{prf:proof}
In dimension $D$, $\det(\Omega^2G)=\Omega^{2D}\det G$. Take absolute square
roots to obtain the volume relation, and then its positive $D$th root.
$\square$
:::

:::{prf:proposition} Finite inputs needed for CST reconstruction operators
:label: prop-fractal-cst-framework-lift

On a finite recorded graph, the CST relation, graph path lengths with given
finite weights, and counts of comparable pairs are determined by the recorded
edges and their stated attributes. If geometric neighborhood membership,
positive sampling weights, kernel values, and field values are additionally
supplied at the episode pairs used by an estimator, its finite sums are
well-defined and can be evaluated from these inputs.

A finite list of pointwise metric values does not determine the metric
between those points, its continuum geodesics, or a continuous sampling
density. These reconstructions require a specified interpolation or model
and approximation estimates when a continuum limit is claimed.
:::

:::{prf:proof}
Reachability is determined by the finite adjacency relation. Acyclicity
allows dynamic programming for path lengths, and a finite pair enumeration
gives order counts. With the listed numerical inputs every estimator below
is a finite sum with nonzero denominators.

For the final assertion, choose a smooth function supported in a coordinate
ball disjoint from the finitely many observed points and perturb the metric
there by a sufficiently small multiple of that function times a positive
definite spatial tensor. The perturbed metric agrees with all observed values
but differs on the ball. Point observations therefore do not specify a unique
continuum metric. $\square$
:::

(sec-faithful-discretization)=
## 3. Sampling, Counts, and Geometric Volume

:::{div} feynman-prose
Counting answers a question about the distribution that generated the points.
If twice as many points are placed in one region, raw counting assigns that
region twice as much weight. To estimate geometric volume, each point must
instead carry the reciprocal of its sampling density relative to geometric
volume. This is ordinary importance sampling, with the reference measure
made explicit.

A quasi-stationary law describes a particle conditional on survival. It is
not automatically the stationary law of a process conditioned to survive
forever, nor does it determine the joint law of an entire recorded history.
Those distinctions matter both for the weights and for their variance.
:::

:::{prf:assumption} Exact episode sampling law when used
:label: assm-cst-episode-sampling

On the specified observation region $W$, assume the observations used in a
sampling statement have a common normalized law

$$
\pi(dy)=q(y)\,d\mathrm{vol}_g(y),\qquad
\int_W q\,d\mathrm{vol}_g=1.
$$

For an integrand supported on $A$, require $q>0$ on $A$. Independence or a
specific covariance bound is imposed separately when convergence or a rate
is claimed.

An optional exact Gibbs model on a product observation window
$W=[t_0,t_1)\times\mathcal X$ is

$$
p(t,x)=\frac{r(t)}{RZ(t)}\sqrt{\det g_R(t,x)}
 e^{-U_{\mathrm{eff}}(t,x)/T},\qquad
Z(t)=\int_{\mathcal X}\sqrt{\det g_R(t,x)}
 e^{-U_{\mathrm{eff}}(t,x)/T}\,dx,\qquad R=\int r(t)\,dt,
$$

where $p$ is the density with respect to $dt\,dx$, $T>0$,
$0<R<\infty$, and $r(t),Z(t)$ are positive and finite. With
$d\mathrm{vol}_g=c\sqrt{\det g_R}\,dt\,dx$, this gives

$$
q(t,x)=\frac{r(t)}{cRZ(t)}e^{-U_{\mathrm{eff}}(t,x)/T},
\qquad W_{\mathrm{geo}}(t,x):=q(t,x)^{-1}
=\frac{cRZ(t)}{r(t)}e^{U_{\mathrm{eff}}(t,x)/T}.
$$

This density is an assumption unless verified for the actual episode law.
If a QSD has only upper and lower Gibbs envelopes, or an additional
multiplicative decoration, those facts do not establish the displayed
identity. The reciprocal of the actual normalized density is required.
:::

:::{prf:theorem} Adaptive counts under the specified sampling law
:label: thm-fractal-adaptive-sprinkling

Under {prf:ref}`assm-cst-episode-sampling`, for every measurable $A\subset W$
and deterministic sample count $N$,

$$
\mathbb E\left[\frac1N\sum_{i=1}^N\mathbf1_A(Y_i)\right]
=\pi(A)=\int_A q\,d\mathrm{vol}_g.
$$

In the optional exact Gibbs model this is

$$
\pi(A)=\frac1R\int_A\frac{r(t)}{Z(t)}
 \sqrt{\det g_R(t,x)}e^{-U_{\mathrm{eff}}(t,x)/T}\,dt\,dx.
$$

The expected counting measure is $N\pi$. If instead a point process is
specified to have intensity $r(t)\rho_t(x)\,dt\,dx$, with
$\rho_t=Z(t)^{-1}\sqrt{\det g_R}\,e^{-U_{\mathrm{eff}}/T}$, its expected
counting measure is that intensity by definition and its expected total
count is $R$. These descriptions agree when the sampling model relates
$N\pi$ to that intensity. Neither statement establishes Poisson statistics,
time stationarity, or an exact Gibbs law for algorithmic episodes.
:::

:::{prf:proof}
The expectation of each indicator is $\pi(A)$. Sum and divide by $N$.
The coordinate expression follows by substituting the assumed $q$ and the
geometric volume element. The intensity assertion is the defining first
moment identity of a point-process intensity. $\square$
:::

:::{prf:remark} Time grids, rates, and quasi-stationarity
:label: rem-cst-time-grid-sampling

For selected recorded layers with $n_k$ sampled episodes and time widths
$\Delta t_k$, one can define an empirical rate
$r_{\mathrm{hist}}(t)=n_k/\Delta t_k$ on each corresponding time cell. Its
integral is $\sum_kn_k=N$. Replacing $n_k$ by an outgoing CST-edge count is
valid only when the chosen episode set is in bijection with those edges,
including the window and cloning conventions.

This piecewise-constant function is a count summary. Recorded times on a
fixed grid have an atomic time law. Treating them as samples from a smooth
spacetime density requires a time-quadrature or time-randomization argument,
with its error controlled in any limiting estimator. Smoothness of an
empirical rate does not follow from the count identity.

The QSD/invariant-law distinction and the exact importance-sampling
normalization are proved in {prf:ref}`lem-continuum-a3-qsd-sampling`.
Exchangeability does not imply independent episodes or temporal mixing.
:::

:::{figure} figures/adaptive-sprinkling.svg
:alt: Illustrative comparison of uniform and nonuniform point densities.
:width: 90%
:align: center

A specified nonuniform density changes expected counts. Its identification
with the episode law is a sampling hypothesis.
:::

:::{prf:definition} Adaptive counts and geometric volume estimators
:label: def-cst-volume

Under the specified geometry and sampling law, for a deterministic measurable
region $A\subset W$ define

$$
\widehat\pi_N(A)=\frac1N\sum_{i=1}^N\mathbf1_A(Y_i),
\qquad
\widehat V_{g,N}(A)=\frac1N\sum_{i=1}^N
 W_{\mathrm{geo}}(Y_i)\mathbf1_A(Y_i),\qquad W_{\mathrm{geo}}=q^{-1}.
$$

For a fixed geometric event $p$, a past-volume statistic uses
$A=W\cap J_g^-(p)$. These regions are defined by the geometric relation,
not by CST lineage reachability. If $p$ is an observation in the same sample,
expectations conditional on $p$ require the corresponding conditional
sampling law.

Whenever $\mathrm{vol}_g(A)<\infty$,

$$
\mathbb E\widehat\pi_N(A)=\pi(A),\qquad
\mathbb E\widehat V_{g,N}(A)=\mathrm{vol}_g(A).
$$

The second statistic is an absolute geometric volume. If a volume fraction
is desired, divide by a specified $V_W$ or use
$\widehat V_{g,N}(A)/\widehat V_{g,N}(W)$ with a separate ratio convergence
argument. In the exact Gibbs model, retaining the historical shorthand
$w_{\mathrm{geo}}=Z(t)e^{U_{\mathrm{eff}}/T}/r(t)$ gives
$W_{\mathrm{geo}}=cR\,w_{\mathrm{geo}}$. The factor $c$ is absent if the
target measure is $\sqrt{\det g_R}\,dt\,dx$ instead of Lorentzian volume.
:::

:::{prf:theorem} Conditional volume matching and trajectory clocks
:label: thm-fractal-faithful-embedding

Assume the geometry and exact sampling law above. If

$$
\sum_{i,j=1}^N\left|\operatorname{Cov}
 (W_{\mathrm{geo}}(Y_i)\mathbf1_A(Y_i),
  W_{\mathrm{geo}}(Y_j)\mathbf1_A(Y_j))\right|\le C_A N,
$$

then $\widehat V_{g,N}(A)$ converges in mean square to
$\mathrm{vol}_g(A)$ with variance at most $C_A/N$. An analogous covariance
condition gives convergence of normalized counts to $\pi(A)$.
These are measure-approximation conclusions. Faithful geometric embedding
would additionally require the specified geometric order, distance, and
sampling identifications.

For a specified timelike trajectory $\gamma(t)=(t,x(t))$, its proper time is

$$
\tau_{\mathrm{prop}}(\gamma)=
\int_s^t\sqrt{1-c^{-2}\|\dot x(u)\|_{g_R(u,x(u))}^2}\,du.
$$

A CST path's recorded increments sum to coordinate time
$\tau_{\mathrm{alg}}=\sum_k\Delta t_k=t-s$. A constant calibration
$\tau_{\mathrm{prop}}=\kappa_\tau\tau_{\mathrm{alg}}$ is valid for a family
of trajectories only when this ratio is known to be the same for that
family. In particular, a path of constant metric speed $v_0<c$ has
$\kappa_\tau=\sqrt{1-v_0^2/c^2}$. A CST chain length alone supplies no
universal proper-time calibration or spatial distance estimate.
:::

:::{prf:proof}
The volume expectation follows by integrating $\mathbf1_A/q$ against
$q\,d\mathrm{vol}_g$. Divide the covariance sum by $N^2$ to obtain the
variance bound. The count statement has the same proof.

Along the specified trajectory,
$-g(\dot\gamma,\dot\gamma)=c^2-\|\dot x\|_{g_R}^2$. Proper time is
$c^{-1}$ times the integral of its square root. Substitution of a constant
speed gives the displayed calibration; summing the coordinate-time
increments telescopes to $t-s$. $\square$
:::

:::{prf:proposition} The variance criterion for adaptive sampling
:label: prop-adaptive-vs-poisson

For independent samples from a normalized density $q$ relative to
a specified volume measure, assume $q>0$ wherever $F\ne0$ and let
$F\in L^1(d\mathrm{vol}_g)$ and
$F\in L^2(q^{-1}d\mathrm{vol}_g)$. Then the importance estimator of
$I_F=\int F\,d\mathrm{vol}_g$ has variance

$$
\frac1N\left(\int\frac{F^2}{q}\,d\mathrm{vol}_g-I_F^2\right).
$$

Consequently one density improves upon another for this integrand precisely
when its integral of $F^2/q$ is smaller. Among normalized densities the
minimum of that second moment is $(\int|F|\,d\mathrm{vol}_g)^2$, attained
by $q\propto|F|$ on its support when $F$ is nonzero. There is no general
variance ordering between an adaptive episode density and uniform sampling
without specifying the observable and dependence structure.
:::

:::{prf:proof}
The single-sample mean is $I_F$ and its second moment is $\int F^2/q$.
Independence divides the variance by $N$. Cauchy–Schwarz gives

$$
\left(\int |F|\,d\mathrm{vol}_g\right)^2
\le\left(\int F^2/q\,d\mathrm{vol}_g\right)
   \left(\int q\,d\mathrm{vol}_g\right),
$$

with equality for the stated density on the support of $F$. $\square$
:::

(sec-cst-machinery)=
## 4. A Conditional Nonlocal Wave Operator

:::{div} feynman-prose
A finite sum can always be evaluated once its inputs are supplied. Convergence
to a differential operator asks for more. The neighborhood must shrink in the
right geometry, signed kernel moments must reproduce the wave operator, and
the sample average must still control its fluctuations at that shrinking
scale. We will state those conditions before using the sum.

The time coordinate and finite cutoff are part of the reconstruction. A
coordinate change can carry these data along consistently; that does not
make a finite cutoff invariant under boosts or remove the preferred time
foliation.
:::

:::{prf:assumption} Fractal Gas continuum hypotheses
:label: assm-fractal-gas-nonlocal

The following conditions concern a sequence of reconstructions on a specified
observation region $W$ and compact evaluation region $K\subset\operatorname{int}W$.
They are hypotheses for the episode application.

**A1 (Geometry).** Assume {prf:ref}`assm-cst-continuum-geometry`. The geometric
causal relation is {prf:ref}`def-cst-geometric-order`. Any graph reconstruction
of these data has separate approximation bounds; finite ancestry order,
propagation of chaos, and convergence of graph Laplacians alone do not identify
this geometry.

**A2 (Local regularity).** The fields used by the reconstruction have the
bounded local derivatives required in
{prf:ref}`lem-continuum-local-bias` on a neighborhood of $K$: in its specified
normal coordinates, $f\circ\Psi_p$ is $C^4$ and the volume density $j_p$ is
$C^3$, with the derivative bounds there uniform for $p\in K$. Positive
denominators have uniform lower bounds on this region. If derivatives of
$Z(t)$ or $r(t)$ are used, assume the integrable domination and positivity
needed for their differentiation. No global bound on a confining potential
or compact support of the sampling law is inferred.

**A3 (Sampling).** Assume {prf:ref}`assm-cst-episode-sampling` for the samples
used by the estimator, with exact weights $W_{\mathrm{geo}}=1/q$. There is a
constant $q_*>0$ on $K$ and the union of the local neighborhoods considered. Grid-time
quadrature, conditional sampling at a random query, and errors from estimated
densities require additional estimates when present.

**A4 (Covariance).** For the actual summands
$H_{\varepsilon,p}(Y_i)$ of the operator defined below, assume

$$
\sum_{i,j=1}^N|\operatorname{Cov}(H_{\varepsilon,p}(Y_i),
 H_{\varepsilon,p}(Y_j))|
\le C_{\mathrm{mix}}N\,\mathbb EH_{\varepsilon,p}(Y_1)^2,
$$

uniformly in $p\in K$, $N$, and the bandwidth. Independence is one sufficient
case. An LSI for a one-time law or a fixed-observable LLN supplies this
condition only through a verified dynamical estimate for these summands.
An alternative sufficient variance estimate, with power $D+4$, is available
from a Poincaré inequality for the same joint observation law under the
gradient conditions in {prf:ref}`lem-cst-poincare-variance`.

**A5 (Kernel, cutoff, and local bias).** Choose a signed
$K_0\in C_c^2([0,1])$ and a normalization $m_2>0$. Let the reconstruction
cutoffs obey $R_{\mathrm{loc}}\le A_R\varepsilon$ and
$cT_{\mathrm{loc}}\le A_T\varepsilon$ with fixed constants. They may be
chosen within the available algorithmic localization scales. Identifying
$R_{\mathrm{loc}}$ exactly with $\min(\rho,\varepsilon_c)$ requires a family
of runs with these same bounds.

Let $\kappa_{\varepsilon,p}(y)$ be the kernel including the actual geometric
neighborhood indicator defined below. Assume its support has volume at most
$C_J\varepsilon^D$, it is bounded by $K_*$, and
$|f(y)-f(p)|\le C_f\varepsilon$ on that support. Require the deterministic
consistency bound

$$
\left|\frac1{m_2\varepsilon^{D+2}}
\int\kappa_{\varepsilon,p}(y)(f(y)-f(p))\,d\mathrm{vol}_g(y)
-\Box_gf(p)\right|\le C_{\mathrm b}\varepsilon^2.
$$

One sufficient way to verify this bound is to use
{prf:ref}`lem-continuum-a5-kernel` and
{prf:ref}`lem-continuum-local-bias` for a normal-coordinate kernel
$\kappa^0_{\varepsilon,p}$ and prove

$$
\int|\kappa_{\varepsilon,p}-\kappa^0_{\varepsilon,p}|
 |f(y)-f(p)|\,d\mathrm{vol}_g(y)\le C_{\mathrm{geom}}\varepsilon^{D+4}.
$$

The required second moments are
$\int_Jk(\zeta)\zeta^\mu\zeta^\nu\,d\zeta=2m_2\eta^{\mu\nu}$ in
Lorentz-orthonormal coordinates. Evenness removes odd moments. A finite
proper-time cutoff does not impose the Lorentzian second-moment identity;
its temporal and spatial moments must be checked separately. The zero-mass
condition can be imposed, but the subtraction $f(y)-f(p)$ already annihilates
constants. The deterministic comparison above is an additional requirement
for the proper-time proxy in a varying metric.

**A6 (Scaling).** Let $N\to\infty$, $\varepsilon\to0$, and
$N\varepsilon^{D+4}\to\infty$, with all preceding constants controlled along
the sequence. For example,
$\varepsilon_N=\ell_0N^{-\alpha}$, $0<\alpha<1/(D+4)$, is an analytic
bandwidth schedule. It does not guarantee these hypotheses for an existing
run or for changing algorithmic localization parameters.
:::

:::{prf:remark} Relation to the continuum estimates
:label: rem-cst-continuum-conditions

{doc}`../convergence_program/16_continuum_discharge` proves sufficient
regularity conditions, normalized importance identities, a tangent-kernel
construction, a local bias bound, and covariance-based variance estimates.
Its {prf:ref}`cor-continuum-consistency-conditional` retains the geometry,
sampling, and reconstruction hypotheses. The bias comparison and covariance
bound above must still be verified for this chapter's proper-time operator.
The auxiliary tangent kernel is a mathematical example, not a change to the
particle update or an identification of the existing reconstruction kernel.
:::

:::{prf:lemma} Spatial metric regularity from the existing fitness estimates
:label: lem-cst-existing-spatial-regularity

Use the selected spatial fitness field and regularity regime of
{prf:ref}`thm-main-complete-cinf-geometric-gas-full`. For the expected field,
use its actual companion-law majorant from
{prf:ref}`thm-unified-cinf-regularity-both-mechanisms`; for mean-field integrals,
use the normalized-integral transfer in
{prf:ref}`thm-cinf-mean-field-integrals`. Keep algorithmic smoothing and
localization scales fixed. Write $\mathcal A$ for the corresponding proved
majorant. Its spatial bounds are

$$
\|\nabla_x^m V_{\mathrm{fit}}\|_\infty
\le B_m:=m![t^m]\mathcal A(t)
\le C_V B_V^m m!,\qquad
C_V=\mathcal A(t_*),\quad B_V=t_*^{-1},
$$

for a common $t_*>0$ in its established convergence region. For smooth inputs
without the factorial regime, use the finite-order bounds through $m=6$ from
the same derivative recursions. The derivative coordinate and uniformity region
are those of the selected field; a one-walker block estimate is used for that
spatial derivative, rather than as a full configuration-space Hessian bound.

For the Hessian metric $g_R=\nabla_x^2V_{\mathrm{fit}}+\epsilon_\Sigma I$,
import its existing two-sided spectral bounds $aI\preceq g_R\preceq bI$ from
{prf:ref}`thm-uniform-ellipticity-latent` with the constants of
{prf:ref}`cor-fractal-set-inherited-ellipticity`.
Then

$$
\|\nabla_x^jg_R\|_\infty\le B_{j+2},\qquad 1\le j\le4,
$$

and the inverse and inverse-square-root regularity follow from
{prf:ref}`lem-continuum-a2-smooth-fields`. This supplies the spatial part of
A2 for that field under the cited analytical hypotheses. The same conclusion
holds for a velocity marginal when the derivatives admit integrable
domination before marginalization.
:::

:::{prf:proof}
For the analytic regime, nonnegative majorant coefficients give
$[t^m]\mathcal A(t)\,t_*^m\le\mathcal A(t_*)$, which proves the displayed
factorial estimate with all scale dependence retained in $C_V,B_V$.
Differentiate the Hessian $j$ times and apply the corresponding existing
finite-order estimate at order $j+2\le6$. The spectral hypotheses justify the inverse recursions and
resolvent integral proved in the cited continuum lemma. For a marginal,
dominated differentiation commutes the derivatives with its integral.
$\square$
:::

:::{prf:remark} Parameters and fields covered by the regularity transfer
:label: rem-cst-spatial-regularity-transfer

The underlying fitness proof estimates derivatives of normalized Gaussian
weights, localized means and variances, and the positive regularized standard
deviation before composing the rescaling function. Those analytical
estimates are retained in the preceding lemma. Their constants depend on
$\rho$, $\varepsilon_c$, $\varepsilon_d$, and the denominator bounds.
Uniformity in the number of walkers therefore does not mean uniformity as
these scales vanish.
On an unbounded spatial domain the required normalizer lower bounds must
hold there directly; a bounded-diameter argument from the bounded-state
analysis cannot be imported globally.

The theorem concerns expected fitness as a function of phase-space variables.
It supplies neither time derivatives of a changing empirical history nor
smoothness across spectral-clipping thresholds. A different effective
potential, a clamped metric, or time-dependent normalization requires its
own matching regularity argument. The decorated Gibbs statement
{prf:ref}`thm-decorated-gibbs` explicitly uses an approximate envelope and a
decoration; its stated conclusion does not identify the exact density in A3.
:::

:::{prf:lemma} A variance estimate from an established joint Poincaré inequality
:label: lem-cst-poincare-variance

Let the joint observation law $\Pi_N$ have the common marginal
$\pi=q\,d\mathrm{vol}_g$ of A3 and satisfy

$$
\operatorname{Var}_{\Pi_N}(\Phi)
\le C_P\,\mathbb E_{\Pi_N}\sum_{i=1}^N|\nabla_i\Phi|^2
$$

for the stated positive Riemannian gradient norm on the observation space,
with $C_P$ independent of $N$. Suppose the actual summand
$H_{\varepsilon,p}$ belongs to its Sobolev test domain and

$$
\int|\nabla H_{\varepsilon,p}|^2\,d\pi
\le C_{\nabla}\varepsilon^{-D-4}.
$$

Then

$$
\operatorname{Var}\left(\frac1N\sum_iH_{\varepsilon,p}(Y_i)\right)
\le\frac{C_PC_{\nabla}}{N\varepsilon^{D+4}}.
$$

For a differentiable compactly supported kernel, sufficient local gradient
bounds are $|\kappa|\le K_*$, $|\nabla\kappa|\le K_1/\varepsilon$,
$|f(y)-f(p)|\le C_f\varepsilon$, $|\nabla f|\le F_1$,
$|\nabla\log q|\le L_q$, $q\ge q_*$, and support volume at most
$C_J\varepsilon^D$. For $\varepsilon\le\varepsilon_0$ these give

$$
C_{\nabla}=\frac{C_J}{m_2^2q_*}
\left(K_1C_f+K_*F_1+K_*C_f\varepsilon_0L_q\right)^2.
$$

An LSI $D_{\mathrm{KL}}(\nu\|\Pi_N)\le C_L I(\nu\|\Pi_N)$ implies
the displayed Poincaré inequality with $C_P=2C_L$ under the same gradient
convention. Thus the established joint-law LSI in
{prf:ref}`cor-n-uniform-lsi`, with its Poincaré consequence
{prf:ref}`cor-quantitative-lsi-final`, supplies this input for its specified law
when the observation variables and gradient form match the sampling statement.
The inherited marginal and weak-limit inequalities are
{prf:ref}`cor-kl-lsi-mean-field-limit`. A one-time particle law must not be replaced
by a spacetime-history law without this identification.
:::

:::{prf:proof}
For $\Phi=N^{-1}\sum_iH(Y_i)$, its $i$th gradient is
$N^{-1}\nabla H(Y_i)$. Insert this identity into the joint Poincaré
inequality and use the common marginal, giving
$\operatorname{Var}\Phi\le C_PN^{-1}\int|\nabla H|^2d\pi$.

Differentiate
$H=\kappa(f-f(p))/(m_2\varepsilon^{D+2}q)$. The stated bounds give

$$
|\nabla H|\le
\frac{K_1C_f+K_*F_1+K_*C_f\varepsilon_0L_q}
{m_2\varepsilon^{D+2}q}.
$$

Square, integrate against $q\,d\mathrm{vol}_g$, and apply the support and
density bounds to obtain $C_{\nabla}$. Finally, apply the LSI to the density
$1+s\Phi$ for bounded centered $\Phi$. Its entropy is
$\tfrac12s^2\operatorname{Var}\Phi+o(s^2)$ and its Fisher information is
$s^2\int|\nabla\Phi|^2d\Pi_N+o(s^2)$. Divide by $s^2$ and let $s\to0$;
the usual closure in the stated Sobolev domain extends the inequality.
$\square$
:::

:::{prf:corollary} Continuum estimator error from the established joint LSI
:label: cor-cst-inherited-lsi-consistency

For the observation law and smooth summand in
{prf:ref}`lem-cst-poincare-variance`, take the existing joint LSI convention

$$
\operatorname{Ent}_{\Pi_N}(f^2)
\le2C_*\mathbb E_{\Pi_N}\sum_i|\nabla_i f|^2.
$$

Use the exact normalized sampling law of A3 and the local consistency
calculation of {prf:ref}`lem-continuum-local-bias`. For
$A_{N,\varepsilon}=N^{-1}\sum_iH_{\varepsilon,p}(Y_i)$ these give

$$
\mathbb E|A_{N,\varepsilon}-\Box_gf(p)|^2
\le C_b^2\varepsilon^4+
\frac{C_*C_\nabla}{N\varepsilon^{D+4}}.
$$

With fixed algorithmic regularizers and uniform constants on the observation
region, $\varepsilon\to0$ and $N\varepsilon^{D+4}\to\infty$ imply mean-square
consistency. Choosing $\varepsilon=N^{-1/(D+8)}$ gives
$O(N^{-4/(D+8)})$. Here $\varepsilon$ is the estimator bandwidth; it is
separate from the algorithmic scales entering the fitness majorants.

*Proof.* The entropy convention above gives Poincaré constant $C_*$ by
{prf:ref}`cor-quantitative-lsi-final`. Equivalently its density convention is
$D_{\mathrm{KL}}\le(C_*/2)I$, so the $C_P=2C_L$ convention of the preceding
lemma gives the same constant. Since
$\nabla_i A_{N,\varepsilon}=N^{-1}\nabla H_{\varepsilon,p}(Y_i)$,

$$
\operatorname{Var}(A_{N,\varepsilon})
\le\frac{C_*}{N^2}\sum_i\mathbb E|\nabla H_{\varepsilon,p}(Y_i)|^2
\le\frac{C_*C_\nabla}{N\varepsilon^{D+4}}.
$$

The common marginal and inverse-density normalization give
$\mathbb E A_{N,\varepsilon}=L_\varepsilon f(p)$.
The existing Taylor and kernel-moment calculation bounds
$|L_\varepsilon f(p)-\Box_gf(p)|\le C_b\varepsilon^2$.
Apply the exact bias-variance decomposition and substitute the stated
bandwidth. This calculation uses the joint inequality directly, so it introduces
no independence requirement on the walkers. $\square$
:::

:::{prf:remark} Order of the analytical and geometric dependencies
:label: rem-cst-proof-dependency-order

The fitness derivative recursions precede metric inversion. The joint-law LSI
proof precedes the empirical-gradient estimate, which precedes the continuum
bias-variance calculation. These proofs use the algorithmic fields and their
specified laws, independently of the lossless record theorem.

For the continuum metric built from those fields, spatial regularity is
{prf:ref}`lem-cst-existing-spatial-regularity`; time regularity is treated in
{prf:ref}`lem-continuum-a2-smooth-fields`. The established metric comparison
and complete background geometry then give global hyperbolicity by
{prf:ref}`lem-continuum-a1-geometry`. Comparison with the recorded graph uses
{prf:ref}`prop-fractal-causal-order-equivalence` for order preservation and
{prf:ref}`lem-cst-graph-distance-comparison` for distance accuracy.
Those comparisons identify the graph quantities to which the analytic estimates
are applied. Operator consistency on a specified metric is a conclusion of
this chain, not a premise proving the graph-metric identification.

The LSI is an inequality for its specified joint observation law. Coordinate
projection to a recorded single-time marginal inherits that inequality by
{prf:ref}`cor-kl-lsi-mean-field-limit`. Pooling episodes across time uses the
actual episode law and the estimates in {prf:ref}`lem-continuum-a4-mixing`
or the same joint Poincaré route on that law. For a normalized reconstructed
operator, {prf:ref}`cor-continuum-consistency-conditional` supplies the final
comparison with the local estimator.
:::

:::{prf:remark} Which analytical error estimates transfer
:label: rem-cst-analytic-error-transfer

The preceding lemma gives a direct use of an established N-particle
functional inequality, without assuming temporal decorrelation. It requires
the same joint law and a Sobolev summand. Sharp indicator cutoffs can fail
the latter condition; it must be checked for the actual reconstruction.
The smooth auxiliary tangent kernel satisfies this aspect when the other
local fields are smooth.

The Kantorovich–Rubinstein step in the proof of
{prf:ref}`thm-quantitative-propagation-chaos` gives
$|\mathbb E\mu_N(H)-\pi(H)|\le\operatorname{Lip}(H)
\mathbb E W_1(\mu_N,\pi)$. It is available whenever the stated Wasserstein
bound has been established for the same law. In a shrinking-neighborhood
application the Lipschitz constant depends on the bandwidth, and a hard
cutoff may make it infinite. This controls a bias comparison; it does not
replace either variance estimate above. A dimension-independent empirical
Wasserstein rate is not used in the present proofs.
:::

:::{prf:definition} Interior evaluation region
:label: def-fractal-gas-interior-episodes

Choose a compact set $K\subset\operatorname{int}W$ for evaluation, and let
$E_{\mathrm{int}}$ be episodes with images in $K$ whose complete geometric
kernel neighborhoods lie in $W$. For a fixed compact support of a test field,
this may be arranged by taking sufficiently small reconstruction bandwidths
inside an open observation neighborhood.

This is an analysis region, not an assertion that the alive process has
compact support. A contribution from omitted boundary or tail regions
requires an explicit estimate with respect to the sampling and geometric
measures; no universal fraction of omitted episodes follows from a finite
time window.
:::

:::{prf:definition} Proper-time neighborhoods and trajectory diagnostics
:label: def-fractal-gas-proper-time-neighborhoods

Under the specified geometry, for a geometrically causal pair $p,q$ define

$$
\tau_g(p,q)=\sqrt{c^2(t(q)-t(p))^2-d_{\mathrm{geo}}(p,q)^2}.
$$

This is a proper-length proxy. In a static complete product metric it equals
the maximal Lorentzian proper length between the endpoints; division by $c$
gives proper time. For a time-dependent metric it is a proxy whose local
error must be estimated when used in a consistency proof. The true proper
length of a causal curve is
$\int\sqrt{c^2-\|\dot x(t)\|_{g_R(t)}^2}\,dt$.

For an interior event $p$, define the localized two-sided causal neighborhood

$$
\begin{aligned}
J_{g,\mathrm{loc}}^\pm(p;\varepsilon)=\{y\in W:\;&
 p\prec_{\mathrm{LC}}y\ \text{or}\ y\prec_{\mathrm{LC}}p,\quad
 0<\tau_g(p,y)\le\varepsilon,\\
&|t(y)-t(p)|\le T_{\mathrm{loc}},\quad
 d_{\mathrm{geo}}(p,y)\le R_{\mathrm{loc}}\}.
\end{aligned}
$$

For the recorded graph set $d_g^\pm(e,e')=\min(d_g(e,e'),d_g(e',e))$ and,
when $d_g^\pm\le c|t(e')-t(e)|$, define

$$
\tau_{\mathrm{traj}}(e,e')=
\sqrt{c^2(t(e')-t(e))^2-d_g^\pm(e,e')^2}.
$$

The trajectory neighborhood $J_{\mathrm{traj,loc}}^\pm(e;\varepsilon)$ uses
CST-comparable episodes, $0<\tau_{\mathrm{traj}}\le\varepsilon$, and the
same time and graph-length cutoffs. It samples recorded lineages, which can
omit geometric neighbors from other lineages.
:::

:::{prf:proof} Static proper-length formula
Let $L$ be a causal curve's spatial length and $\Delta t$ its elapsed time.
Concavity of $v\mapsto\sqrt{c^2-v^2}$ on $[0,c]$ gives

$$
\int\sqrt{c^2-\|\dot x\|_{g_R}^2}\,dt
\le\Delta t\sqrt{c^2-(L/\Delta t)^2}
\le\sqrt{c^2\Delta t^2-d_{g_R}(x(p),x(q))^2}.
$$

A minimizing spatial geodesic traversed at constant speed attains the last
bound. Completeness supplies that geodesic. This proves the static assertion
and explains why it concerns the maximal endpoint proper length, rather
than the proper length of every recorded trajectory. $\square$
:::

:::{prf:definition} Weighted nonlocal Fractal Gas d'Alembertian
:label: def-cst-fractal-dalembertian

Use the specified geometry, normalized weights, kernel $K_0$, and cutoffs in
{prf:ref}`assm-fractal-gas-nonlocal`. For an interior evaluation point $p$ put

$$
\kappa_{\varepsilon,p}(y)=
\mathbf1_{J_{g,\mathrm{loc}}^\pm(p;\varepsilon)}(y)
K_0\!\left(\frac{\tau_g(p,y)}{\varepsilon}\right).
$$

For $N$ observations define

$$
(\Box_{\mathrm{FG}}f)(p)
=\frac1{m_2\varepsilon^{D+2}N}
\sum_{i=1}^N W_{\mathrm{geo}}(Y_i)
\kappa_{\varepsilon,p}(Y_i)\bigl(f(Y_i)-f(p)\bigr).
$$

Thus $H_{\varepsilon,p}(y)=
W_{\mathrm{geo}}(y)\kappa_{\varepsilon,p}(y)(f(y)-f(p))
/(m_2\varepsilon^{D+2})$ is the summand used in A4.
Under the exact Gibbs model, the prefactor and weight can equivalently be
written as $cR/(m_2\varepsilon^{D+2}N)$ times
$w_{\mathrm{geo}}=Z e^{U_{\mathrm{eff}}/T}/r$.
The operator vanishes on constants. With length coordinates and consistently
transformed sampling density, its units are those of $f$ divided by length
squared, as for $\Box_g$.

The finite sum uses the supplied reconstruction inputs. Estimated distances,
weights, or order tests define an approximate operator and need their own
comparison bound after this normalization.
:::

:::{prf:definition} Weighted scalar action
:label: def-cst-fractal-action

Let $\mu_{\mathrm{geo},N}(Y_i)=W_{\mathrm{geo}}(Y_i)/N$ and let $f$ have
compact support inside the evaluation region. The same-sample scalar action
is the finite statistic

$$
S_{\mathrm{FG}}[f]=\frac12\sum_{Y_i\in E_{\mathrm{int}}}
\mu_{\mathrm{geo},N}(Y_i)f(Y_i)(\Box_{\mathrm{FG}}f)(Y_i).
$$

Its continuum target is $\tfrac12\int f\Box_gf\,d\mathrm{vol}_g$ when the
joint convergence conditions below hold. Pointwise consistency at a fixed
query does not itself establish convergence of this same-sample action.
:::

:::{prf:theorem} Conditional continuum consistency of the nonlocal operator
:label: thm-cst-fractal-dalembertian-consistency

Assume A1–A3, A5, and A6 of {prf:ref}`assm-fractal-gas-nonlocal`. For each
fixed interior point $p$ and the stated test field $f$, the bias satisfies

$$
\left|\mathbb E(\Box_{\mathrm{FG}}f)(p)-\Box_gf(p)\right|
\le C_{\mathrm b}\varepsilon^2.
$$

For fluctuations, assume either the covariance condition A4, which gives

$$
\operatorname{Var}((\Box_{\mathrm{FG}}f)(p))
\le\frac{C_{\mathrm{mix}}K_*^2C_f^2C_J}
 {m_2^2q_*N\varepsilon^{D+2}}.
$$

or all the joint-law and gradient conditions of
{prf:ref}`lem-cst-poincare-variance`, which give instead
$\operatorname{Var}((\Box_{\mathrm{FG}}f)(p))
\le C_PC_{\nabla}/(N\varepsilon^{D+4})$.
In either case $(\Box_{\mathrm{FG}}f)(p)\to\Box_gf(p)$ in mean square and
probability. For a query chosen from the observations, these conclusions
require the same hypotheses for its conditional sampling law and the
appropriate remaining sample count.

An approximate operator using graph reconstructions has the same limit in
probability if its difference from this exact-input operator tends to zero
in probability. The displayed bias and variance rates require corresponding
moment bounds on that additional error.

For the same-sample action, assume additionally a quadrature LLN for
$f\Box_gf$ with the weights $\mu_{\mathrm{geo},N}$ and

$$
\sup_{p\in\operatorname{supp}f}
|(\Box_{\mathrm{FG}}f)(p)-\Box_gf(p)|\xrightarrow{\mathbb P}0.
$$

Then $S_{\mathrm{FG}}[f]\to\tfrac12\int f\Box_gf\,d\mathrm{vol}_g$ in
probability. The uniform statement is an additional hypothesis, not a
consequence of the pointwise estimate. An alternative sufficient construction
with independent evaluation samples is proved in
{prf:ref}`cor-continuum-consistency-conditional`.
:::

:::{prf:proof}
Cancellation of $q$ in the sampling expectation gives exactly the integral
in A5, so that assumption proves the bias bound. For the second moment,

$$
\begin{aligned}
\mathbb EH_{\varepsilon,p}(Y_1)^2
&=\frac1{m_2^2\varepsilon^{2D+4}}
\int\frac{\kappa_{\varepsilon,p}(y)^2(f(y)-f(p))^2}{q(y)}
\,d\mathrm{vol}_g(y)\\
&\le\frac{K_*^2C_f^2C_J}{m_2^2q_*\varepsilon^{D+2}}.
\end{aligned}
$$

Divide the covariance bound A4 by $N^2$. Variance plus squared bias bounds
the mean square error by $C/(N\varepsilon^{D+2})+C_{\mathrm b}^2\varepsilon^4$,
which tends to zero under A6. With the alternative Poincaré input, the same
argument uses $C_PC_{\nabla}/(N\varepsilon^{D+4})$ for the variance, which
also vanishes under A6. An additional error tending to zero in
probability preserves convergence in probability.

For the action subtract the weighted quadrature of $f\Box_gf$. The absolute
difference is at most

$$
\frac12\sup_{\operatorname{supp}f}
|\Box_{\mathrm{FG}}f-\Box_gf|
\left(\frac1N\sum_i\frac{|f(Y_i)|}{q(Y_i)}\right)
\le\frac{\|f\|_\infty}{2q_*}
\sup_{\operatorname{supp}f}|\Box_{\mathrm{FG}}f-\Box_gf|.
$$

The last bound tends to zero in probability. The stated quadrature LLN
handles the remaining term. $\square$
:::

:::{prf:remark} Comparison with the Benincasa–Dowker operator
:label: rem-cst-bd-comparison

The construction here is a two-sided localized difference operator.
[Benincasa and Dowker](https://arxiv.org/abs/1001.2725) construct retarded
operators from causal-set layers; their curved-spacetime continuum expression
contains $\Box-\tfrac12R$ under their stated approximation conditions.
The two operators have different domains and weights. A uniform sampling law
alone does not identify them. Any proposed equality or limiting comparison
must match their kernels, causal orientation, normalizations, and curvature
terms explicitly.
:::

(sec-cst-dimension-curvature)=
## 5. Dimension and Curvature Statistics

:::{div} feynman-prose
A counting formula needs a calibration experiment. For the Myrheim–Meyer
statistic that experiment uses points uniformly distributed inside a
Minkowski causal diamond. Changing the shape of the observation region changes
the fraction of comparable pairs, even in perfectly flat spacetime.

Curvature is more demanding. The leading volume term must be subtracted, and
a smaller second-order term is then divided by the square of the neighborhood
size. Both sampling noise and geometric calibration errors are magnified by
that division. The curvature hypotheses below keep those requirements visible.
:::

:::{prf:definition} Myrheim–Meyer ordering statistic and its calibration
:label: def-myrheim-meyer

For $N\ge2$ events with a specified geometric order, define the fraction of
unordered comparable pairs

$$
r_N=\frac{\sum_{i<j}\mathbf1_{Y_i\prec_{\mathrm{LC}}Y_j\,
\mathrm{or}\,Y_j\prec_{\mathrm{LC}}Y_i}}{\binom N2}.
$$

For independent uniform samples in a finite Alexandrov interval of
$D$-dimensional Minkowski spacetime, the continuum calibration is

$$
r_D=\frac{\Gamma(D+1)\Gamma(D/2)}{2\Gamma(3D/2)}.
$$

An Alexandrov interval is the intersection of the chronological future of
one event with the chronological past of another. This calibration is the
Myrheim–Meyer formula {cite}`Myrheim1978,Meyer1988`; the relation-count
normalization is also displayed in
[Abajian and Carlip, equation (2.2)](https://arxiv.org/pdf/1710.00938).
Their ratio of expected relation count to squared expected point count is
half the unordered-pair fraction used here. For example $r_2=1/2$ and
$r_4=1/10$.

The estimator $d_{\mathrm{MM}}$ inverts this calibration when the measured
fraction lies in its range; the estimated spatial dimension is
$d_{\mathrm{MM}}-1$. For nonuniform samples with specified density $q_A$
relative to volume on the same interval $A$, put $w_i=1/q_A(Y_i)$ and define

$$
r_{w,N}=\frac{\sum_{i<j}w_iw_j
\mathbf1_{Y_i\prec_{\mathrm{LC}}Y_j\,\mathrm{or}\,Y_j\prec_{\mathrm{LC}}Y_i}}
{\sum_{i<j}w_iw_j}.
$$

Multiplying all weights by one constant leaves this ratio unchanged.
Its geometric calibration requires a pair-sampling LLN such as the one below.
Neither the CST lineage order nor an arbitrary coordinate cylinder has the
same calibration as an Alexandrov interval. A local curved-space application
requires small neighborhoods whose rescaled geometry and shape approach that
calibration, together with enough samples and uniform pair-error estimates.
:::

:::{prf:lemma} A sufficient weighted pair law of large numbers
:label: lem-cst-weighted-pairs

Let $A$ have finite positive geometric volume. Suppose $Y_1,\ldots,Y_N$ are
independent with density $q_A$ on $A$, where $q_A\ge q_*>0$, and use the
exact geometric comparability indicator. Then

$$
r_{w,N}\xrightarrow{\mathbb P}
\frac{\int_{A\times A}
\mathbf1_{x\prec_{\mathrm{LC}}y\,\mathrm{or}\,y\prec_{\mathrm{LC}}x}
\,d\mathrm{vol}_g(x)d\mathrm{vol}_g(y)}{\mathrm{vol}_g(A)^2}.
$$

For the specified Minkowski Alexandrov interval this is $r_D$. For
interacting episodes the corresponding pair law is an additional hypothesis.
:::

:::{prf:proof}
Divide numerator and denominator by $\binom N2$. Their symmetric pair
kernels are bounded by $B=q_*^{-2}$, and their expectations are respectively
the double integral in the numerator and $\mathrm{vol}_g(A)^2$.

Terms from disjoint index pairs are independent. Each index pair overlaps
at most $2N-3$ pairs, including itself, and every covariance has absolute
value at most $B^2$. Thus each normalized pair average has variance at most

$$
\frac{(2N-3)B^2}{\binom N2}\le\frac{4B^2}{N}.
$$

Both averages converge in mean square to their expectations. The denominator
limit is positive, so their ratio converges in probability. The flat-diamond
calibration then gives the last assertion. $\square$
:::

:::{prf:assumption} Additional curvature calibration
:label: assm-cst-curvature-calibration

Use the geometric neighborhoods of
{prf:ref}`def-fractal-gas-proper-time-neighborhoods`, with cutoffs of order
$\varepsilon$. Fix a bounded $K_R\in C_c^2([0,1])$ and write

$$
\kappa^R_{\varepsilon,p}(y)=
\mathbf1_{J_{g,\mathrm{loc}}^\pm(p;\varepsilon)}(y)
K_R(\tau_g(p,y)/\varepsilon).
$$

In addition to the specified geometry and sampling law, require constants
$M_0^{(R)}$, $M_R\ne0$, and $C_R$ such that

$$
\left|\frac1{\varepsilon^D}
\int\kappa^R_{\varepsilon,p}\,d\mathrm{vol}_g
-M_0^{(R)}-M_RR_g(p)\varepsilon^2\right|
\le C_R\varepsilon^3.
$$

The constants are fixed along the chosen scaled-domain construction.
$M_0^{(R)}$ is a flat-volume moment. Identifying $M_R$ and ensuring that the
second-order term is only the scalar curvature requires a curved geometric
expansion and any necessary tensor-moment cancellations. A finite cutoff
can also introduce directional Ricci terms. Flat-space calibration, where
$R_g=0$, cannot determine this curvature coefficient by itself.

For the stochastic limit assume $q\ge q_*$ on the neighborhoods,
$\mathrm{vol}_g(\operatorname{supp}\kappa^R_{\varepsilon,p})
\le C_J\varepsilon^D$, and a covariance bound of the A4 form for
$G_{\varepsilon,p}(y)=\kappa^R_{\varepsilon,p}(y)/q(y)$, with a constant
uniform in the bandwidth. These are hypotheses for the curvature estimator;
the scalar-operator covariance bound does not automatically cover this
different observable.
:::

:::{prf:definition} Geometric curvature estimator and lineage diagnostic
:label: def-fractal-gas-curvature

Under the exact sampling weights and stated nonzero curvature calibration,
define

$$
\widehat R_{\mathrm{FG}}^{(g)}(p)=
\frac1{M_R\varepsilon^2}\left[
\frac1{N\varepsilon^D}\sum_{i=1}^N
W_{\mathrm{geo}}(Y_i)\kappa^R_{\varepsilon,p}(Y_i)-M_0^{(R)}\right].
$$

Replacing the geometric neighborhood and proxy by
$J_{\mathrm{traj,loc}}^\pm(e;\varepsilon)$ and
$\tau_{\mathrm{traj}}$ gives a recorded-lineage statistic
$\widehat R_{\mathrm{FG}}^{(\mathrm{traj})}(e)$ with the same algebraic form.
Its sampling support differs from the geometric neighborhood, so the
curvature calibration does not transfer to it. Interpreting it as scalar
curvature requires a separate geometric and statistical theorem.
:::

:::{prf:theorem} Conditional scalar-curvature consistency
:label: thm-fractal-gas-ricci

Assume {prf:ref}`assm-cst-continuum-geometry`,
{prf:ref}`assm-cst-episode-sampling`, and every condition in
{prf:ref}`assm-cst-curvature-calibration`. For a fixed interior point $p$,

$$
\left|\mathbb E\widehat R_{\mathrm{FG}}^{(g)}(p)-R_g(p)\right|
\le\frac{C_R}{|M_R|}\varepsilon,
\qquad
\operatorname{Var}(\widehat R_{\mathrm{FG}}^{(g)}(p))
\le\frac{C_{\mathrm{mix}}\|K_R\|_\infty^2C_J}
{M_R^2q_*N\varepsilon^{D+4}}.
$$

Thus $\widehat R_{\mathrm{FG}}^{(g)}(p)\to R_g(p)$ in mean square and
probability if $\varepsilon\to0$ and $N\varepsilon^{D+4}\to\infty$ with
these constants controlled. Approximate graph inputs require an additional
error bound after normalization by $\varepsilon^{-D-2}$.

For a compactly supported bounded test function $\chi$ in the interior and
$\kappa_D\ne0$, define

$$
S_{\mathrm{FG},\chi}^{(g)}=
\frac1{2\kappa_DN}\sum_i
W_{\mathrm{geo}}(Y_i)\chi(Y_i)\widehat R_{\mathrm{FG}}^{(g)}(Y_i).
$$

Assume additionally uniform curvature convergence in probability on
$\operatorname{supp}\chi$, a positive density lower bound there, and a
weighted quadrature LLN for $\chi R_g$. Then this statistic converges to
$(2\kappa_D)^{-1}\int\chi R_g\,d\mathrm{vol}_g$. Taking $\chi=1$ over an
unbounded observation region further requires integrability and uniform tail
control. This is a conditional geometric action approximation, not a
selection principle for solutions of Einstein's equations.
:::

:::{prf:proof}
The sampling expectation is the calibrated volume integral, so subtracting
$M_0^{(R)}$ and dividing by $M_R\varepsilon^2$ gives the stated bias.
Furthermore,

$$
\mathbb EG_{\varepsilon,p}(Y_1)^2
=\int\frac{(\kappa^R_{\varepsilon,p})^2}{q}\,d\mathrm{vol}_g
\le q_*^{-1}\|K_R\|_\infty^2C_J\varepsilon^D.
$$

Apply the curvature covariance condition, divide by $N^2$, and multiply by
$M_R^{-2}\varepsilon^{-2D-4}$. This yields the variance bound. Squared bias
plus variance tends to zero under the stated scaling.

For the action, its difference from weighted quadrature of $\chi R_g$ is
bounded by

$$
\frac{\|\chi\|_\infty}{2|\kappa_D|q_*}
\sup_{\operatorname{supp}\chi}|\widehat R_{\mathrm{FG}}^{(g)}-R_g|.
$$

Use uniform convergence for this term and the additional quadrature LLN for
the other one. Integrating over a noncompact region requires a further bound
on the omitted tails, as stated. $\square$
:::

(sec-physical-consequences)=
## 6. Interpretation and Scope

:::{div} feynman-prose
The calculations give conditional ways to compare recorded data with a
specified geometric model. They do not choose that model's dynamics. An
estimate of curvature is a measurement of a metric; Einstein's equations
would additionally relate that curvature to matter. Likewise, a probability
law on stochastic histories is not yet a quantum amplitude.

These distinctions suggest concrete checks: compare weighted volumes on
known domains, test the wave operator on smooth fields, and calibrate
curvature on metrics with known curvature. The questions become measurable
approximation errors once the reference geometry and sampling experiment are
specified.
:::

:::{prf:proposition} Resolution scales from a specified counting intensity
:label: prop-cst-predictions

If an episode process has expected counting measure
$\lambda_g(y)\,d\mathrm{vol}_g(y)$ with $\lambda_g>0$, the quantity

$$
\ell_{\mathrm{count}}(y)=\lambda_g(y)^{-1/D}
$$

is a length scale obtained from volume per expected point. For a normalized
law with deterministic count $N$, $\lambda_g=Nq$. On a region of finite
volume $V_W$, $(V_W/N)^{1/D}$ is the corresponding scale from average density.
These expressions use density per geometric volume, rather than coordinate
density. They are dimensional count diagnostics, not formulas for Lorentzian
nearest-neighbor proper distance or a physical Planck scale.

Their values alone imply no modified dispersion relation or numerical
coefficient for Lorentz violation. Such a prediction requires specified
field dynamics, a continuum approximation, and a physical calibration of
units and observables.
:::

:::{prf:proof}
Geometric $D$-volume has length units $D$, so counting intensity has units
$\mathrm{length}^{-D}$. Taking its inverse $D$th root gives the displayed
length unit. The average intensity over a finite region is $N/V_W$.
No field evolution equation enters this calculation. $\square$
:::

:::{prf:definition} An optional finite ensemble average
:label: def-cst-ensemble-average

Let $\mathcal C_N$ be the finite set of labeled strict partial orders on $N$
labels, let $P_N$ be a specified probability law on that set, and let $S(C)$
be a real-valued action. For a specified action scale $\hbar>0$, the complex
ensemble average

$$
Z_N=\sum_{C\in\mathcal C_N}P_N(C)e^{iS(C)/\hbar}
$$

is well-defined and satisfies $|Z_N|\le1$ by the triangle inequality. For
continuously marked histories the corresponding expression is an integral
against a specified history law. A one-time QSD alone does not specify that
history law. Interpreting this average as a quantum gravitational path
integral requires additional dynamical and physical assumptions.
:::

:::{prf:remark} Status of the results
:label: rem-cst-result-status

The finite CST relation is a causal-set order by
{prf:ref}`thm-fractal-is-causal-set`. Its construction establishes no
manifold approximation by itself. The geometric order, exact sampling law,
distance reconstruction, kernel calibration, and dependence estimates enter
as explicit conditions for the continuum statements.

Under those conditions, the chapter proves normalized count and volume
identities, conditional wave-operator consistency, a sufficient weighted pair
LLN, and conditional curvature consistency. The same-sample actions require
the additional uniform and quadrature hypotheses in their statements.
General relativity, a quantum theory, and a physical identification of the
sampling scale require further arguments beyond these reconstruction results.
:::

(sec-cst-references)=
## 7. References and Dependencies

:::{div} feynman-added
- {doc}`01_fractal_set`: recorded episodes, temporal edges, and clone ancestry.
- {doc}`../convergence_program/16_continuum_discharge`: direct continuum
  estimates and the conditions needed in an episode application.
- {doc}`../3_fitness_manifold/01_emergent_geometry`: the proposed metric and
  diffusion convention, with its required regularity and spectral bounds.
- Bombelli, Lee, Meyer, and Sorkin, *Space-Time as a Causal Set*
  {cite}`BombelliLeeEtAl87`; Sorkin, *Causal Sets: Discrete Gravity*
  {cite}`Sorkin05`.
- Myrheim, *Statistical Geometry* {cite}`Myrheim1978`; Meyer,
  *The Dimension of Causal Sets* {cite}`Meyer1988`.
- Benincasa and Dowker, *The Scalar Curvature of a Causal Set*
  {cite}`BenincasaDowker2010`, for its distinct retarded construction and
  approximation conditions.
:::
