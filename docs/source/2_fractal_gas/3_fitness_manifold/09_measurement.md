(sec-measurement)=
# Measurement Operators on Scutoid Spacetime

:::{div} feynman-prose
A measurement needs three decisions: which cells to inspect, what number to
attach to each cell, and how to combine those numbers. Geometry answers the
first question. A distance and an amplitude rule answer the second. If the
cell values have local internal frames, parallel transport answers the third.

The cell reconstruction is developed in {doc}`02_scutoid_spacetime`.
The comparison links used here follow
{doc}`../2_fractal_set/03_lattice_qft` and
{doc}`../2_fractal_set/05_yang_mills_noether`.
:::

(sec-measurement-tldr)=
:::{div} feynman-prose
A line or hyperplane selects the Voronoi cells it intersects. Each selected
swept cell supplies an amplitude and a geometric phase. Transport brings its
field value to a common reference before addition. The norm of this sum is
independent of local frame choices for any fixed set of paths. Independence
from the paths themselves requires a further condition: the relevant loop
transports must act trivially. A spanning tree makes that condition finite
and directly testable.
:::

(sec-measurement-setup)=
## 1. Probes and the cells they select

:::{div} feynman-prose
Imagine passing a sheet of paper through a collection of cells. A cell is
selected when some part of it touches the sheet; its center need not lie on
the sheet. The same rule works for a thin straight probe. This distinction
matters computationally: a line can pass through the middle of a square
while missing all four vertices.

Hyperplanes and straight lines refer to a specified Euclidean coordinate
chart. The Voronoi cells may use either Euclidean distance or the declared
spatial metric. Those choices are separate parts of the measurement.
:::

:::{prf:definition} Measurement schedule and observation geometry
:label: def-measurement-schedule

Fix $T>0$ and measurement times

$$
 t_k=t_0+kT,\qquad k\in\mathbb Z_{\ge0}.
$$

At each time, specify the included sites, spatial domain $M$, metric used
for the Voronoi cells, and any observation clipping region. The cells
$\operatorname{Vor}_i(t_k)$ are those of
{prf:ref}`def-voronoi-tessellation-time-t`, with the clipping convention
applied when present. In coordinate formulas below, the observation region
lies in a fixed chart $x\in\mathbb R^d$.

When the recorded history is discrete, select available frames or declare
the interpolation used to evaluate an intervening measurement time. The
measurement interval $T$ need not equal the algorithm's time step.
:::

:::{prf:definition} Hyperplane probe
:label: def-hyperplane-probe

For a Euclidean unit normal $n\in\mathbb R^d$ and offset $b$, define the
ambient hyperplane and its signed coordinate distance by

$$
 H(n,b)=\{x:n\cdot x=b\},\qquad
 \operatorname{dist}_E^{\mathrm{sgn}}(x,H)=n\cdot x-b.
$$

The probe inside the observation domain is $H\cap M$. The displayed
formula measures signed distance to the ambient hyperplane; distance to a
clipped portion of that hyperplane can differ.
:::

:::{prf:definition} Line probe
:label: def-line-probe

For an anchor $p\in\mathbb R^d$ and Euclidean unit direction
$u\in\mathbb R^d$, define

$$
 L(p,u)=\{p+\lambda u:\lambda\in\mathbb R\},\qquad
 \operatorname{dist}_E(x,L)=\|(I-uu^T)(x-p)\|.
$$

This is the unsigned distance to the ambient line. A segment probe is
obtained by specifying a closed parameter interval for $\lambda$; its
distance and intersection tests use that interval.
:::

(sec-measurement-piercing)=
### Cell selection and adjacency

:::{prf:definition} Pierced Voronoi cell: hyperplane
:label: def-pierced-cell-hyperplane

A cell is pierced when its intersection with the probe is nonempty. For a
hyperplane, the selected indices are

$$
 P^H_{t_k}=\{i:H(n,b)\cap\operatorname{Vor}_i(t_k)\ne\varnothing\}.
$$

This convention includes tangencies and contact along a cell boundary.
:::

:::{prf:definition} Pierced Voronoi cell: line
:label: def-pierced-cell-line

For a line probe, the selected indices are

$$
 P^L_{t_k}=\{i:L(p,u)\cap\operatorname{Vor}_i(t_k)\ne\varnothing\}.
$$

For a segment, replace $L$ by the specified segment.
:::

:::{prf:definition} Pierced neighbor graph
:label: def-pierced-neighbor-graph

Let $E_{\mathrm{DT}}(t_k)$ be the edges of the specified Delaunay or
cell-adjacency construction on the slice. For either selected set $P_{t_k}$,
define the induced graph

$$
 G^{\mathrm{pierced}}_{t_k}
  =(P_{t_k},E^{\mathrm{pierced}}_{t_k}),\qquad
 E^{\mathrm{pierced}}_{t_k}
  =\{\{i,j\}\in E_{\mathrm{DT}}(t_k):i,j\in P_{t_k}\}.
$$

Adjacency uses the geometric convention in
{prf:ref}`def-neighbor-set` and {prf:ref}`def-delaunay-triangulation`.
For a general metric it is supplied by that construction; the Euclidean
Delaunay predicates apply under {prf:ref}`lem-scutoid-euclidean-delaunay`.
The induced graph can have several connected components.
:::

:::{prf:lemma} Exact piercing tests for convex polyhedral cells
:label: lem-measurement-polytope-piercing

Let $C\ne\varnothing$ be a convex polyhedron. If $C$ is bounded with
vertex set $V$, then

$$
 C\cap H(n,b)\ne\varnothing
 \quad\Longleftrightarrow\quad
 \min_{v\in V}(n\cdot v-b)\le0\le
 \max_{v\in V}(n\cdot v-b).
$$

For a half-space description $C=\{x:a_\ell\cdot x\le c_\ell\}$,
line piercing is equivalent to feasibility of

$$
 (a_\ell\cdot u)\lambda\le c_\ell-a_\ell\cdot p
 \quad\text{for every }\ell.
$$

Thus positive coefficients give upper bounds on $\lambda$, negative
coefficients give lower bounds, and zero coefficients require a nonnegative
right-hand side. The intersection is nonempty precisely when all zero
constraints hold and the largest lower bound is at most the smallest upper
bound. Segment bounds are included in this interval intersection.
:::

:::{prf:proof}
A linear functional on a bounded polyhedron takes its minimum and maximum
at vertices. Its image on a convex set is an interval, so that image contains
zero exactly under the stated inequalities. For a line, substitute
$x=p+\lambda u$ into every defining half-space. Dividing each nonzero
coefficient with its corresponding inequality direction gives exactly the
listed bounds. These substitutions are reversible, proving sufficiency
as well as necessity. For an unbounded cell, the hyperplane test can instead
be performed as affine feasibility with the same half-space constraints.
:::

:::{div} feynman-prose
The vertex sign test succeeds because a linear function takes every value
between its extrema on a convex cell. Distance to a line has a different
minimum: that minimum may occur inside the cell. For example, the horizontal
axis pierces $[-1,1]^2$, although every vertex has distance one from it. The
interval test keeps the geometry of the whole cell.
:::

(sec-measurement-scutoids)=
## 2. Swept cells, representatives, and Euclidean phases

:::{div} feynman-prose
The slice gives us a selected label. To attach a quantity to its evolution,
we need the cell swept out during a time interval. The endpoint positions
alone do not determine that region. In particular, a cloning replacement
can jump: the recorded slot continues, while its parent identifier records
a separate genealogical relation.

A midpoint is often enough for a phase feature. Its appeal is that it is
cheap and reproducible. Its meaning remains an average of the chosen
points; the center of mass of a curved swept region generally requires an
integral over the region.
:::

:::{prf:definition} Scutoid indexing on a measurement slice
:label: def-scutoid-indexing-measurement

Use the interpolation and event policy of {prf:ref}`def-scutoid-cell`.
Write $S_{i,k}$ for the swept cell of recorded slot $i$ on
$[t_k,t_{k+1}]$, with explicit one-sided pieces at replacement jumps when
required. Bottom association selects the cells whose bottom index lies in
$P_{t_k}$; top association uses the corresponding top cells on the terminal
slice. The set selected by bottom association is denoted
$\mathcal S_{t_k}$.

An association across a birth, death, or missing endpoint must be supplied
by the reconstruction. A parent identifier can define a separate lineage
observable, but does not by itself identify that parent's swept cell with
the replaced slot's cell. A measurement interval containing several events
retains those events or records an explicit coarsening rule.
:::

:::{prf:definition} Center-only representative
:label: def-scutoid-barycenter-center

For supplied bottom and top positions $x^-_{i,k},x^+_{i,k}$, define

$$
 \bar x_{\mathrm{ctr}}(S_{i,k})
 =\tfrac12(x^-_{i,k}+x^+_{i,k}),\qquad
 \bar X_{\mathrm{ctr}}(S_{i,k})
 =\left(\bar x_{\mathrm{ctr}}(S_{i,k}),
         \tfrac12(t_k+t_{k+1})\right).
$$

These are barycenters of the two specified endpoint points. They need not
be volume barycenters of $S_{i,k}$ or lie inside a nonconvex swept cell.
Coordinates use one chart and, on a periodic domain, a declared consistent
lift before averaging.
:::

:::{prf:definition} Vertex-augmented representative
:label: def-scutoid-barycenter-vertex

If finite bottom and top vertex sets $V^-_{i,k},V^+_{i,k}$ are supplied in
the same coordinates, define

$$
 \bar x_{\mathrm{vx}}(S_{i,k})
 =\frac{x^-_{i,k}+x^+_{i,k}
        +\sum_{v\in V^-_{i,k}}v+\sum_{v\in V^+_{i,k}}v}
        {2+|V^-_{i,k}|+|V^+_{i,k}|}.
$$

This averages endpoint generators and endpoint vertices with equal weights.
For a cell of finite positive measure $\nu$, its spatial volume barycenter
is instead $\nu(S)^{-1}\int_S x\,d\nu(t,x)$, when that integral exists.
The vertex formula requires a finite vertex representation; it does not
supply one for a curved or unbounded cell.
:::

:::{prf:definition} Euclidean hyperplane phase
:label: def-phase-hyperplane-euclidean

For either chosen representative $\bar x(S)$, define

$$
 \phi_E^H(S)=k_H\big(n\cdot\bar x(S)-b\big).
$$

The scale $k_H$ has inverse-length units in the specified coordinates, so
the phase is dimensionless. Reversing the probe orientation reverses this
phase.
:::

:::{prf:definition} Euclidean line phase
:label: def-phase-line-euclidean

Define the unsigned line-distance phase

$$
 \phi_E^L(S)=k_L\|(I-uu^T)(\bar x(S)-p)\|,
$$

where $k_L$ has inverse-length units. Reversing $u$ leaves this phase
unchanged.
:::

(sec-measurement-walker-relation)=
## 3. Intrinsic distances and graph measurements

:::{div} feynman-prose
A path along graph edges is a permitted route between sites. Its length can
exceed the shortest route through the surrounding space. If the metric also
changes the cost of moving in different directions, Euclidean edge lengths
introduce another difference. Both effects can be bounded, but the bounds
need geometric information.

There is also a change of endpoint to watch. Distance from a cell
representative to its walker measures displacement within the reconstructed
cell. Distance from a walker to the set of pierced walkers measures access
to the probe. A walker already in that set has access distance zero even
when its cell representative has moved.
:::

:::{prf:definition} Intrinsic probe phases and their graph alternatives
:label: def-phase-geodesic-proxy

On a specified measurement slice, let $g$ be a positive-definite spatial
metric and $d_g$ its length distance. If
$g=D_x^2V+\epsilon_\Sigma I$, require the spectral bounds of
{prf:ref}`lem-scutoid-metric-bounds`; positivity of
$\epsilon_\Sigma$ alone is insufficient. For a nonempty probe portion
$B\subset M$, set

$$
 \operatorname{dist}_g(x,B)=\inf_{y\in B}d_g(x,y),\qquad
 \phi_g^H(S)=k_H\operatorname{dist}_g(\bar x(S),H\cap M),\qquad
 \phi_g^L(S)=k_L\operatorname{dist}_g(\bar x(S),L\cap M).
$$

These intrinsic phases are unsigned. An oriented hyperplane variant is
$\operatorname{sgn}(n\cdot\bar x-b)\phi_g^H(S)$.
Intrinsic evaluation requires $\bar x(S)\in M$, with the slice and metric
specified even when $\bar x$ comes from a time interval.

For a spatial graph with sites $x_i$, define $d_{\mathrm{DT}}$ using
Euclidean edge lengths. If each edge has a specified admissible spatial
curve, define $d_{G,g}$ using its $g$-length instead. A shortest-path
calculation approximates an intrinsic phase only with a distance comparison
and a rule attaching the representative and the probe to the graph.
:::

:::{prf:lemma} Metric comparison and endpoint errors for graph distances
:label: lem-measurement-distance-comparison

For a graph whose edges are admissible rectifiable curves in $M$,

$$
 d_g(x_i,x_j)\le d_{G,g}(i,j).
$$

If the edges are straight segments in the chart and
$\lambda I\preceq g\preceq\Lambda I$ along them, then

$$
 \sqrt\lambda\,d_{\mathrm{DT}}(i,j)
 \le d_{G,g}(i,j)
 \le\sqrt\Lambda\,d_{\mathrm{DT}}(i,j).
$$

These comparisons do not assert that the graph contains routes close to
minimizing geodesics. Suppose, in addition, that for the required vertices

$$
 0\le d_{G,g}(i,j)-d_g(x_i,x_j)\le\eta.
$$

For a nonempty set of attached probe sites $X_P=\{x_j:j\in P\}$,
assume their $d_g$-Hausdorff distance from the chosen probe portion $B$ is
at most $h$, and $d_g(\bar x,x_i)\le a$. Then

$$
 \left|\min_{j\in P}d_{G,g}(i,j)
             -\operatorname{dist}_g(\bar x,B)\right|
 \le a+h+\eta.
$$
:::

:::{prf:proof}
Every graph path concatenates admissible curves. Its $g$-length is at least
the infimum over all such curves, proving the first assertion. The spectral
bounds give $\sqrt\lambda|\dot\gamma|\le
|\dot\gamma|_g\le\sqrt\Lambda|\dot\gamma|$ on each edge.
Integrating, summing, and taking infima over graph paths proves the second.

The additional graph estimate survives taking the minimum over $j\in P$.
Distance to a set is a 1-Lipschitz function of its starting point. Replacing
one target set by another at Hausdorff distance at most $h$ changes its
distance function by at most $h$, as follows by the triangle inequality
and an arbitrarily close point to each infimum. Applying these two facts
and the graph estimate gives the last bound.
:::

:::{prf:definition} Euclidean walker-relative phases
:label: def-walker-relative-phase-euclidean

For the specified endpoint positions and representative, define

$$
 \phi_E^-(S_{i,k})=k_W\|\bar x(S_{i,k})-x^-_{i,k}\|,
 \qquad
 \phi_E^+(S_{i,k})=k_W\|\bar x(S_{i,k})-x^+_{i,k}\|.
$$

Here $k_W$ has inverse-length units. For the center-only representative,
both phases equal $k_W\|x^+_{i,k}-x^-_{i,k}\|/2$.
:::

:::{prf:definition} Intrinsic endpoint phases and graph probe-access phases
:label: def-walker-relative-phase-geo

With a specified spatial metric $g_-$ or $g_+$ for each endpoint comparison,
the intrinsic endpoint phases are

$$
 \phi_g^-(S_{i,k})=k_W d_{g_-}(\bar x(S_{i,k}),x^-_{i,k}),\qquad
 \phi_g^+(S_{i,k})=k_W d_{g_+}(\bar x(S_{i,k}),x^+_{i,k}).
$$

All compared points must lie in the domain of the respective distance.
Separately, the computable graph probe-access phases are

$$
 \phi_{\mathrm{access}}^-(S_{i,k})
   =k_W d_{\mathrm{DT}(t_k)}(i,P_{t_k}),\qquad
 \phi_{\mathrm{access}}^+(S_{i,k})
   =k_W d_{\mathrm{DT}(t_{k+1})}(i,P_{t_{k+1}}),
$$

where $d_G(i,P)=\min_{j\in P}d_G(i,j)$. A missing endpoint or unreachable
set gives an undefined measurement, or an explicitly flagged infinite
distance, rather than a finite phase. For a reachable pierced vertex these
access phases vanish. They describe access to selected sites; approximating
the intrinsic endpoint phases instead requires graph attachments to
$\bar x$ and the endpoints, with errors controlled as above.
:::

(sec-measurement-amplitude)=
## 4. Amplitudes and local complex fields

:::{div} feynman-prose
An amplitude determines how strongly each selected cell contributes. Volume
weighting asks how much reconstructed space-time the cell occupies.
Probability weighting asks how much mass a specified sampling law assigns
to a region. Inverse-density weighting has a third role: it compensates
for uneven sampling when estimating volume.

After choosing an amplitude, multiplying it by a complex phase gives a
signal feature. Interference in the resulting sum is an algebraic property
of complex numbers. A probabilistic or physical interpretation of its
squared magnitude requires a separately specified measurement model.
:::

:::{prf:definition} Scutoid amplitude and its reference measure
:label: def-scutoid-amplitude

Choose a nonnegative finite amplitude for each included swept cell. Two
geometric choices are

$$
 A_V(S)=\operatorname{Vol}_{d+1}(S),\qquad
 A_Q(S)=w_{\mathrm{geo}}(S).
$$

The first uses a declared volume measure $\nu$ on the reconstruction,
such as coordinate measure $dt\,dx$ or $dt\,d\operatorname{vol}_{g_t}$.
Its units follow that measure. The notation $w_{\mathrm{geo}}(S)$ in the
second formula denotes a specified nonnegative volume quadrature. For
example, under the sampling law $q\,d\nu$ with $q>0$, the inverse-density
estimator from {prf:ref}`def-cst-volume` is

$$
 w_{\mathrm{geo}}(S)=\frac1m\sum_{\ell=1}^m
          \frac{\mathbf1_S(Y_\ell)}{q(Y_\ell)}.
$$

For a deterministic finite-volume $S$ and samples with that marginal law,
its expectation is $\nu(S)$, since
$\int\mathbf1_S q^{-1}q\,d\nu=\nu(S)$.
If $S$ is selected from the same samples, this expectation needs the
corresponding conditional sampling argument.

Here $A_Q$ estimates geometric volume by inverse-density quadrature.
A probability amplitude such as
$A_\mu(S)=\mu(B_S)$ instead requires a specified probability law $\mu$
on the slice and a measurable footprint $B_S$. Using a QSD for $\mu$
requires the appropriate marginal of the identified QSD; its mass is
dimensionless and differs from inverse-density weighting.
:::

:::{prf:definition} Measurement functional in a common frame
:label: def-measurement-functional

For a declared common trivialization of the field values, define the raw
complex signal

$$
 \mathcal M(t_k)=\sum_{S\in\mathcal S_{t_k}}\psi(S).
$$

In the scalar case, $\psi(S)=A(S)e^{i\phi(S)}$ in that trivialization.
In a $q$-dimensional unitary representation, the field lies in
$\mathbb C^q$. Choose one of the probe phases above, optionally adding
a specified endpoint or access phase. Every summand uses the same units
and normalization convention.

When each cell has its own local frame, this raw sum depends on those
frames. The intrinsic comparison uses the reference-transported sum of
{prf:ref}`def-reference-transported-measurement`.
:::

(sec-measurement-gauge-compatibility)=
## 5. Comparing fields through a connection

:::{div} feynman-prose
Suppose two observers express vectors using different axes. Adding their
coordinate lists before aligning the axes gives an answer that changes
when either observer rotates a notebook. A connection records how to make
that alignment along an edge. A chain of edges transports all vectors into
one observer's frame, where their sum has a definite meaning.

The geometric phase remains a scalar under this change of internal axes.
The internal direction of the field changes with the frame. Keeping these
two transformations separate fixes the orientation signs in the formulas.
:::

:::{prf:definition} Edge parallel transport on the measurement graph
:label: def-pierced-edge-transport

Use a unitary representation of a group $G\subset U(q)$ and a fiber
$V_i\simeq\mathbb C^q$ at each measurement vertex. The comparison link

$$
 U_{ij}:V_j\longrightarrow V_i,\qquad
 U_{ji}=U_{ij}^{-1}=U_{ij}^{\dagger}
$$

maps coordinates at $j$ into the frame at $i$, following
{prf:ref}`def-lqft-link-convention` and
{prf:ref}`def-link-variable-ym`. Under local frame changes,

$$
 \psi_i\mapsto G_i\psi_i,\qquad
 U_{ij}\mapsto G_iU_{ij}G_j^{-1}.
$$

The scalar case uses $G=U(1)$; $SU(q)$ gives a nonabelian example.
For a directed path $\gamma=(i_0,i_1,\ldots,i_m)$ traversed from
$i_0$ to $i_m$, the forward transport is

$$
 T_\gamma=U_{i_mi_{m-1}}\cdots U_{i_2i_1}U_{i_1i_0}.
$$

The rightmost factor acts first. Reverse paths have inverse transport.
:::

:::{prf:definition} Edge-type transport assignment
:label: def-edge-type-transport-assignment

Specify the graph on which the measurement transports its fields and
assign a comparison link to every edge used. A recorded Fractal Set edge
can inherit a supplied link of the same representation from
{prf:ref}`def-fractal-set-gauge-connection`:

1. An IG comparison uses its spatial link.
2. A CST edge included across slices uses its temporal link, with the
   inverse for reverse traversal.
3. An IA edge uses its attribution link when that link is part of the
   selected field construction.

A Delaunay neighbor pair need not be a recorded IG pair. Such an edge
requires a declared connection reconstruction or another path through
available links. Geometry alone does not assign its gauge matrix.
All links in a product must have matching source and target fibers.
Interaction-pair doublets use a graph of the corresponding pair states,
as in {prf:ref}`def-su2-clone-transport`; passing from vertex fields to
pair fields requires that explicit representation choice.
:::

:::{prf:definition} Scutoid field and transport to a reference
:label: def-scutoid-field-transport

Attach the bottom-associated cell $S_{i,k}$ to the fiber at its recorded
vertex $i$ on slice $t_k$. Choose a unit vector $\chi_i\in V_i$ and set

$$
 \psi_i=\psi(S_{i,k})=A(S_{i,k})e^{i\phi(S_{i,k})}\chi_i.
$$

The amplitude and geometric phase are gauge scalars; $\chi_i$ transforms
as $G_i\chi_i$, so $\psi_i$ transforms as $G_i\psi_i$.
No fiber at the geometric representative is required for this definition.

Within a connected measurement component choose a reference vertex $r$
and a path $\gamma_i:i\to r$ for each included field. The paths are fixed
as part of the measurement, independently of local frame coordinates.
Define

$$
 \psi_r(S_{i,k})=T_{\gamma_i}\psi_i\in V_r.
$$

For disconnected components use one reference per component, or supply
additional comparison paths in a specified larger graph.
:::

:::{prf:definition} Reference-transported measurement
:label: def-reference-transported-measurement

For the cells attached to one reference component, define

$$
 \mathcal M_r(t_k)=\sum_{S_{i,k}}T_{\gamma_i}\psi_i,
 \qquad
 \mathcal J_r(t_k)=\|\mathcal M_r(t_k)\|^2.
$$

The scalar amplitude statistic and nonabelian intensity statistic retain
the conventions

$$
 \mathcal I_r=|\mathcal M_r|\quad(G=U(1)),\qquad
 \mathcal I_r=\operatorname{tr}(\mathcal M_r\mathcal M_r^\dagger)
             =\mathcal J_r\quad(G=SU(q)).
$$

Thus $\mathcal J_r=\mathcal I_r^2$ in the first convention. Use one
normalization consistently when comparing measurements. For disconnected
components, $\sum_c\mathcal J_{r_c}$ is a scalar statistic without
requiring an identification of their fibers.
:::

:::{prf:lemma} Reference-transport covariance
:label: lem-reference-transport-covariance

For any fixed paths and any unitary connection,

$$
 T_{\gamma_i}\mapsto G_rT_{\gamma_i}G_i^{-1},\qquad
 \psi_r(S_{i,k})\mapsto G_r\psi_r(S_{i,k}),\qquad
 \mathcal M_r\mapsto G_r\mathcal M_r.
$$

Consequently $\mathcal I_r$ and $\mathcal J_r$ are gauge invariant.
Moreover,

$$
 \|\mathcal M_r\|\le\sum_{S_{i,k}}A(S_{i,k}),\qquad
 \mathcal J_r\le\left(\sum_{S_{i,k}}A(S_{i,k})\right)^2.
$$

These conclusions require no flatness assumption.
:::

:::{prf:proof}
Under the frame change, the factors in
$T_\gamma=U_{i_mi_{m-1}}\cdots U_{i_1i_0}$ become

$$
 (G_{i_m}U_{i_mi_{m-1}}G_{i_{m-1}}^{-1})\cdots
 (G_{i_1}U_{i_1i_0}G_{i_0}^{-1}).
$$

Each adjacent pair $G_j^{-1}G_j$ cancels. The surviving factors are
$G_{i_m}T_\gamma G_{i_0}^{-1}$. Multiplication by
$G_{i_0}\psi_{i_0}$ proves field covariance, and summation gives the same
transformation for $\mathcal M_r$. Unitarity preserves its norm; equivalently,
$\mathcal M_r\mathcal M_r^\dagger$ transforms by conjugation and its
trace is unchanged. Finally every $T_{\gamma_i}$ is unitary and
$\|\psi_i\|=A(S_{i,k})$. The triangle inequality proves both bounds.
:::

:::{div} feynman-prose
Gauge invariance has now been proved, even if different routes rotate a
vector differently. The routes are part of this measurement, just as the
position of the probe is part of it. Changing a route is a change of the
measurement protocol. The next result identifies precisely when that change
has no effect.
:::

(sec-measurement-path-independence)=
## 6. Path independence and loop observables

:::{div} feynman-prose
Start with a spanning tree: it reaches every vertex using one route, with
no loops. Every extra edge closes a loop against that tree. These extra
edges provide a finite list of tests for whether all possible routes agree.
Checking small triangles suffices only when those triangles also account
for the larger loops. A hole in the selected graph can leave an additional
loop to check.
:::

:::{prf:definition} Gauge-compatible probe field
:label: def-gauge-compatible-probe-phase

A unit section $s_i\in V_i$ is parallel on the measurement graph when

$$
 U_{ij}s_j=s_i\qquad\text{on every edge}.
$$

One may impose this as an additional compatibility requirement on
$s_i=e^{i\phi(S_{i,k})}\chi_i$. In the scalar case, writing
$s_i=e^{i\theta_i}$ in a chosen trivialization gives

$$
 U_{ij}=e^{i(\theta_i-\theta_j)}.
$$

The total phase $\theta_i$ includes the frame-dependent phase of $\chi_i$;
the geometric probe phase $\phi$ by itself remains gauge invariant.
The same equation for full fields, $U_{ij}\psi_j=\psi_i$, also requires
equal amplitudes along each edge, by unitarity. Parallel-section
compatibility is optional for the measurement functional.
:::

:::{prf:theorem} Path-independent reference transport on the pierced graph
:label: thm-path-independent-reference-transport

Let $G_0=(V,E)$ be a finite connected component of the measurement graph,
with unitary links as above. Fix a root $r$, a spanning tree, and its forward
transports $B_i=T_{r\to i}^{\mathrm{tree}}$, with $B_r=I$.
The following conditions are equivalent:

1. Transport between any two vertices is independent of the chosen path.
2. Every closed path has identity transport.
3. For each of the $|E|-|V|+1$ edges outside the tree, in either chosen
   orientation,
   $B_i^{-1}U_{ij}B_j=I$.
4. Every link has the form $U_{ij}=B_iB_j^{-1}$.

Under these conditions the reference-transported fields and their sum are
path independent for every field assignment, and their norm statistics
are gauge invariant. No compatibility between the geometric phases and
the connection is needed.
:::

:::{prf:proof}
If all paths with the same endpoints agree, a loop agrees with the constant
path, proving condition 2. For an edge carrying $U_{ij}:V_j\to V_i$,
start at $r$, follow the tree to $j$, traverse that edge to $i$, and return
along the tree to $r$. Its transport is $B_i^{-1}U_{ij}B_j$.
Thus condition 2 implies condition 3.

For a tree edge, the defining tree paths already give
$U_{ij}=B_iB_j^{-1}$: one endpoint extends the path to the other, and
reverse traversal uses the inverse. For every edge outside the tree, the
same identity follows by multiplying condition 3 on the left by $B_i$
and on the right by $B_j^{-1}$. Hence condition 3 implies condition 4.

Finally, on a path $i_0\to\cdots\to i_m$, condition 4 gives

$$
 T_\gamma=(B_{i_m}B_{i_{m-1}}^{-1})\cdots
          (B_{i_1}B_{i_0}^{-1})=B_{i_m}B_{i_0}^{-1}.
$$

This depends only on the endpoints and proves condition 1. The non-tree
edges are exactly $|E|-(|V|-1)$ in number. Path independence of each
transport gives path independence of the sum; gauge invariance is
{prf:ref}`lem-reference-transport-covariance`.
:::

:::{prf:corollary} Reference changes and field-specific path independence
:label: cor-measurement-reference-and-section

For any connection, changing reference from $r$ to $r'$ by appending the
same fixed path $\delta:r\to r'$ to every transport gives

$$
 \mathcal M_{r'}=T_\delta\mathcal M_r,
 \qquad \mathcal J_{r'}=\mathcal J_r.
$$

For a particular vector $v\in V_i$, transport from $i$ is independent of
paths precisely when every loop based at $i$ fixes $v$. This is weaker
than identity holonomy on the whole fiber. For example,

$$
 \operatorname{diag}(1,e^{i\alpha},e^{-i\alpha})\in SU(3)
$$

fixes $(1,0,0)^T$ while being nonidentity when
$\alpha\notin2\pi\mathbb Z$.
:::

:::{prf:proof}
Appending a common path multiplies every summand by the same unitary
$T_\delta$, proving the first assertion. For two paths $\gamma,\gamma'$
from $i$ to the same endpoint,

$$
 T_\gamma v=T_{\gamma'}v
 \quad\Longleftrightarrow\quad
 T_{\gamma'}^{-1}T_\gamma v=v.
$$

The matrix on the right is the transport around the loop obtained by
following $\gamma$ and returning along $\gamma'$. Conversely, compare
a fixed outgoing path with that path preceded by any loop at $i$.
This proves necessity for every loop as well as sufficiency. The displayed
matrix verifies the example directly.
:::

:::{prf:lemma} When triangle tests establish global flatness
:label: lem-measurement-triangle-flatness

Suppose the measurement graph carries specified triangular faces, and
assume every closed edge path can be reduced to the constant path by a
finite sequence of backtrack cancellations and replacements of two sides
of a face by its third side, or the reverse replacements. If every face
has identity holonomy, all loop transports are identity.

Without this reduction property, checking the triangles alone is
insufficient. A square cycle with no triangular faces has no triangle
tests, but assigning $U(1)$ phase $e^{i\alpha}\ne1$ to one forward edge
and phase one to the other three gives nonidentity loop transport.
:::

:::{prf:proof}
A backtrack contributes $U_{ij}U_{ji}=I$. Identity transport around a
triangle gives $U_{cb}U_{ba}=U_{ca}$ for its two-side path $a\to b\to c$.
Thus each permitted replacement preserves the full ordered path product.
After the finite reduction, the product is that of the constant path,
namely $I$. The square example has forward loop product $e^{i\alpha}$,
which proves the second assertion.
:::

:::{div} feynman-prose
For nonabelian matrices, even the order of loop comparisons matters. A
collection of triangles that spans cycles only after treating their edges
as commuting symbols does not supply the reduction used in the proof.
The spanning-tree test avoids that issue: it checks an explicit generating
set of based loops in the graph. Taking an induced subgraph of a spatial
tessellation can create holes, so the selected graph needs its own test.
:::

:::{prf:definition} Gauge-invariant measurement through holonomy
:label: def-gauge-invariant-measurement

For a specified oriented closed path $\mathcal C$ based at $i$, define
its Wilson statistic

$$
 W(\mathcal C)=\operatorname{tr}T_{\mathcal C}.
$$

This is {prf:ref}`def-wilson-loop-lqft` with the path orientation and
comparison-link convention made explicit. Reversing a path conjugates its
Wilson statistic: $W(\mathcal C^{-1})=\overline{W(\mathcal C)}$.
Real statistics may use its real part or modulus. Loop statistics and the
norms of reference-transported fields may be combined using a declared
scalar function to define a gauge-invariant probe observable.
:::

:::{prf:proof}
The path transformation proved in
{prf:ref}`lem-reference-transport-covariance` gives
$T_{\mathcal C}\mapsto G_iT_{\mathcal C}G_i^{-1}$.
Cyclicity of the trace proves invariance. Unitarity gives
$T_{\mathcal C^{-1}}=T_{\mathcal C}^\dagger$, proving the conjugation
identity. A scalar function of invariant arguments remains invariant.
:::

(sec-measurement-implementation-notes)=
## 7. Recording a reproducible measurement

:::{div} feynman-prose
The resulting observable is reproducible when its geometric and transport
choices travel with the output. Two signals may disagree because one uses
a different clipping window, a different distance, or a different path
through a curved connection. Recording those choices makes the difference
interpretable.

A neighbor-radius heuristic can help select candidate cells, but an exact
piercing result requires a cell intersection test. Likewise, a graph
distance is a well-defined graph observable even before a geodesic
approximation theorem applies. Naming the quantity computed is enough to
keep these measurements usable without assigning them an unsupported
interpretation.
:::

:::{prf:algorithm} Measurement specification
:label: alg-measurement-record

For each requested measurement time:

1. Record the frame, included sites, observation window, coordinate chart,
   metric, probe parameters, and cell reconstruction. Use the corresponding
   cell-intersection rule; polyhedral Euclidean cells admit
   {prf:ref}`lem-measurement-polytope-piercing`.
2. Select bottom or top associations, retaining slot identifiers and event
   policies. Record the representative rule, including periodic lifts and
   any interpolation across the measurement interval.
3. Record the phase formula and its scale. Distinguish ambient Euclidean
   distance, intrinsic distance, endpoint displacement, and graph access
   distance. For a claimed intrinsic approximation, retain its attachment
   and distance error bounds.
4. Record the amplitude measure or quadrature, its units, and any
   normalization. Identify the sampling law used for probabilistic
   weighting or inverse-density estimates.
5. Specify the representation and every link used in transport. Choose
   references and paths separately for connected components. A deterministic
   spanning forest supplies such paths. If path independence is claimed,
   check the non-tree loop products in
   {prf:ref}`thm-path-independent-reference-transport`.
6. Return the selected raw signal or transported statistic, with the
   amplitude-versus-intensity convention and any loop observables.
   Empty selections give zero sums; missing associations and inaccessible
   required paths receive an explicit undefined status.

This specifies an observable of the recorded history and the supplied
geometric reconstruction. It leaves the Fractal Gas transition rule
unchanged.
:::
