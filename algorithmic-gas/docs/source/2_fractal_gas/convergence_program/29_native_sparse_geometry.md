# Native shrinking-scale Delaunay geometry and covariance-ridge limits

(sec-nsg-execution)=
## 1. Complete execution and the configured estimators

:::{prf:definition} Native local geometry register
:label: def-nsg-complete-register

Retain the complete execution record $\mathfrak P$ of
{prf:ref}`def-native-complete-execution-record`, the geometry record in
{prf:ref}`def-native-jg-ledger`, and the quadratic capped terminal-box
register in {prf:ref}`def-native-stationary-closure-register`.
All donor widths, feature clamps, reward and diversity maps and exponents,
global standardization constants, acceptance saturation and regularizer,
current-source exclusions, mandatory revival, recipient jitter,
simultaneous frozen copying, shared component Haar collision, both
viscous kicks, force coefficient, OU friction and amplitude, final
position noise, velocity cap, boundary and recording choices retain
their configured values. The positive calculation has independent
real Gaussian innovations, passive geometry, `Projection::Full` and
`TessellationDomain::Open`. It has no consumed geometry, graph-viscosity,
curl, metric-noise or reward feedback. Those tags have distinct laws.
Every numerical policy, graph budget, precision, validity flag and
physical calibration remains in $\mathfrak P$.

The primary increasing-population regime keeps $h>0$ and every consumed
primitive parameter fixed, with quadratic $F(x)=-\lambda x$, either
existing count or nonself row Gaussian viscosity, $q>0$, $s>0$, bounded
eligible donor domain $D$, and finite cap $V$. All permitted populations
must be executable under their recorded allocation limits. Let
$\mathcal F$ contain the complete post-jitter/collision preparation and
the preceding record. With the actual coefficients in
{prf:ref}`def-native-stationary-closure-register`, the terminal positions
have exactly

$$
Y_j=m_j+\tau Z_j,\qquad
m_j=a_xX_j^J+bU_j,\qquad \tau^2=t^2q^2+s^2>0,
$$

conditionally on $\mathcal F$, where the $Z_j$ are independent standard
normal rows. This identity includes the O/final-noise correlation with
completed velocities. The $m_j$ retain all interacting preparation
dependence. Their primitive moment bound is

$$
\mathbb E\frac1N\sum_j|m_j|^p\le H_p,
\quad H_p=[|a_x|(R_D+\sigma_Jg_{d,p})+b\kappa_\nu V_c]^p.
\tag{NSG.1}
$$

In this quadratic register the same bound holds for each fixed row,
because $|U_i|\le\kappa_\nu V_c$ and its own source/jitter bound
is uniform. Thus fixed-tag tails do not need exchangeability.

The full conditional spatial density and its derivative bounds are

$$
\rho_N(y)=\frac1N\sum_j\varphi_\tau(y-m_j),\qquad
L_{\tau,k}=\sup_y\|D^k\varphi_\tau(y)\|<\infty,
\quad \varphi_\tau(y)=(2\pi\tau^2)^{-d/2}e^{-|y|^2/(2\tau^2)}.
$$

Thus $L_{\tau,k}$ is a directly evaluated Gaussian derivative supremum,
scaling as $\tau^{-d-k}$; it is independent of the realized centers.
For example $L_{\tau,0}=(2\pi\tau^2)^{-d/2}$ and
$L_{\tau,1}=L_{\tau,0}/(\tau\sqrt e)$.
The final alive mark is $\mathbf1_D(Y_j)$.

The sparse results concern the actual Euclidean Delaunay CSR rows and
their dual Voronoi cells at these sites. A returned full-dimensional
native graph agreeing with that Euclidean construction has these rows.
The code's duplicate lifting, rank projection, tessellator failure,
edge budget and rounding branches are retained as separate marked
outcomes; no probability of numerical agreement is assumed below.
In the real-coordinate Euclidean construction, distinctness and
general position hold almost surely by the conditional Gaussian density.

For `MetricKind::NeighborCovariance`, the actual matrix is
$C_i=k_i^{-1}\sum_{j\sim i}(Y_j-Y_i)(Y_j-Y_i)^T$ for $k_i>0$.
It is an unweighted mean over the CSR row. It is not a Gaussian-weighted
covariance or a kernel-ball estimator. The ridge $\eta>0$, absolute or
relative scale, configured bounds $b_-,b_+$ (each possibly absent),
`Strict`/`Clipped` policy, pseudo-inverse threshold
$\theta_T=d\epsilon_T$, determinant floor and diffusion floor keep
their literal definitions in `metric.rs` and `linalg.rs`.
`WeightMode::Kernel` has raw weights
$\exp[-|Y_j-Y_i|^2/(2\ell_N^2)]$ only on those CSR edges;
normalized rows divide by $\max(\sum_j\mathrm{raw}_{ij},10^{-12})$.
The existing dense viscous count/row weights are a different included
map, with all nonself pairs. Their short estimate below supplies no
missing edge to the CSR estimator.
:::

(sec-nsg-dense-comparison)=
## 2. The existing full Gaussian weights and their actual fluctuation scale

:::{prf:lemma} Conditional degree and first/second Gaussian row estimates
:label: lem-nsg-dense-row-estimates

Condition on $\mathcal F$ and on a tag $Y_i=y$. Put
$K_\varepsilon(z)=e^{-|z|^2/(2\varepsilon^2)}$ and

$$
D_i(y)=\frac1N\sum_{j\ne i}K_\varepsilon(Y_j-y),\qquad
A_i(y)=\frac1N\sum_{j\ne i}K_\varepsilon(Y_j-y)(Y_j-y),
$$
$$
B_i(y)=\frac1N\sum_{j\ne i}K_\varepsilon(Y_j-y)
                         (Y_j-y)(Y_j-y)^T.
$$

These are the existing nonself full Gaussian weights with their
numerators retained. For each component of $D_i$, $A_i/\varepsilon$
and $B_i/\varepsilon^2$, let $P_k(u)$ be respectively $1$, $u_a$ or
$u_au_b$, and set

$$
M_k=\sup_u e^{-|u|^2/2}|P_k(u)|,\qquad
V_k=L_{\tau,0}\int e^{-|u|^2}|P_k(u)|^2\,du.
$$

For every $z>0$, its conditional deviation from its conditional mean
is at most

$$
e_k(N,\varepsilon,z)=
\sqrt{\frac{2V_k\varepsilon^d z}{N}}
                     +\frac{4M_kz}{3N}
\tag{NSG.2}
$$

except on an event of conditional probability at most $2e^{-z}$.
For all tags simultaneously a union bound multiplies that probability
by $N(1+d+d^2)$. No independence between different rows is used.

Let $\bar D_i=\mathbb E[D_i\mid\mathcal F,Y_i=y]$. Then exactly

$$
\bar D_i=(2\pi)^{d/2}\varepsilon^d q_i(y),\qquad
q_i(y)=\frac1N\sum_{j\ne i}\varphi_{\sqrt{\tau^2+\varepsilon^2}}(y-m_j),
$$
$$
\mathbb E A_i=\varepsilon^2\nabla\bar D_i,\qquad
\mathbb E B_i=\varepsilon^2\bar D_i I+
                         \varepsilon^4D^2\bar D_i.
\tag{NSG.3}
$$

At a compact query set where $q_i\ge\beta_N>0$, the normalized first
row moment has deterministic comparison
$\varepsilon^2\nabla\log q_i$ and error bounded by the numerator
errors in (NSG.2) divided by $\bar D_i-e_0$; the second has comparison
$\varepsilon^2I+\varepsilon^4D^2q_i/q_i$ with the analogous bound.
In particular, for fixed positive $\beta_N=\beta$ and fixed $\tau$,
$N\varepsilon^d/\log N\to\infty$ controls degrees and the first and
second moments divided by $\varepsilon$ and $\varepsilon^2$.
Convergence of the differential first moment
$2\varepsilon^{-2}A_i/D_i$ requires the stronger sampling condition
$N\varepsilon^{d+2}/\log N\to\infty$ by this estimate. Its native
deterministic target is $2\nabla\log q_i$, rather than zero.
For the literally executed full Gaussian B2 force, use its actual
position stage $Y=X^{\rm B2}$ and $\tau=tq$ in this lemma; the
same independence proof applies. The terminal-stage formulas describe
that kernel map on its retained terminal positions and do not relocate
the executed B2 force after final position diffusion.
:::

:::{prf:proof}
Conditional on the tag, every other row still has its independent
Gaussian density. Each displayed component is bounded by $M_k$ and
its second moment is at most
$L_{\tau,0}\varepsilon^d\int e^{-|u|^2}|P_k(u)|^2du$.
For a centered independent bounded summand $W$, the power-series
estimate $\mathbb E|W|^r\le(2M_k)^{r-2}\mathbb EW^2$ for $r\ge2$
gives, for $0<t<3/(2M_k)$,
$\log\mathbb Ee^{tW}\le t^2\mathbb EW^2/[2(1-2M_kt/3)]$.
Multiplication of the moment-generating functions, exponential Markov
and the same estimate for $-W$ give (NSG.2), with its conservative
linear constant. Integrate the conditional tag law and union bound.

Gaussian convolution gives $\bar D_i$.
Differentiating $K_\varepsilon(z-y)$ in $y$ gives
$\nabla_yK=\varepsilon^{-2}(z-y)K$ and
$D_y^2K=[\varepsilon^{-4}(z-y)(z-y)^T-
\varepsilon^{-2}I]K$. Dominated differentiation proves (NSG.3).
For any normalized numerator, subtract the two fractions using
$D_i\ge\bar D_i-e_0>0$. Their leading errors after normalization
are respectively $\varepsilon\sqrt{\log N/(N\varepsilon^d)}$
and $\varepsilon^2\sqrt{\log N/(N\varepsilon^d)}$, times the
explicit lower-density and derivative constants. Multiplying the
first by $2/\varepsilon^2$ gives the stronger condition.
Self-exclusion remains in $q_i$ throughout.
:::

(sec-nsg-native-poisson)=
## 3. Native conditional local Poisson law and protected Delaunay stars

:::{prf:lemma} Total-variation local point-process approximation
:label: lem-nsg-local-poisson-tv

Set $r_N=N^{-1/d}$. Conditional on $\mathcal F,Y_i=x$, retain the
point process of the other actual sites in a bounded Borel set
$W\subset\mathbb R^d$ after rescaling:

$$
\Xi_{N,i,x}|_W=\sum_{j\ne i}
                 \delta_{(Y_j-x)/r_N}|_W.
$$

Let $\Pi_{N,x}$ be a comparison homogeneous Poisson point process
with intensity $\rho_N(x)$. Then

$$
\|\mathcal L(\Xi_{N,i,x}|_W\mid\mathcal F,Y_i=x)
                   -\mathcal L(\Pi_{N,x}|_W)\|_{\rm TV}
\le \frac{L_{\tau,0}^2|W|^2+L_{\tau,0}|W|}{N}
       +r_NL_{\tau,1}\int_W|u|\,du.
\tag{NSG.4}
$$

This bound is uniform in all realized centers and tags. It requires
no independent incoming swarm or deterministic population density.
For final alive-only geometry it is identical whenever
$x+r_NW\subset D$. The intensity is the native raw density
$\rho_N(x)$, not a normalized selected marginal.
:::

:::{prf:proof}
Each other row contributes either no point to $W$ or one point, with
probability $p_j\le L_{\tau,0}|W|/N$ and its actual conditional mark
density. Couple this Bernoulli count to a Poisson count of mean $p_j$:
the total-variation error is at most $p_j^2$, as follows directly by
comparing the probabilities at zero and one and bounding the
Poisson probability of two or more by $p_j^2/2$. When both counts
are one, use the same mark. The independent Poisson processes sum
to a Poisson process whose density on $W$ is

$$
\lambda_N(u)=\frac1N\sum_{j\ne i}
                        \varphi_\tau(x+r_Nu-m_j).
$$

The total Bernoulli/Poisson error is at most
$\sum_jp_j^2\le L_{\tau,0}^2|W|^2/N$.
The omitted self-density contributes at most $L_{\tau,0}/N$;
the gradient bound contributes at most $r_NL_{\tau,1}|u|$.
Couple the process with intensity $\lambda_N$ and the homogeneous
process by their common intensity $\min(\lambda_N,\rho_N(x))$.
The probability of any remaining point is at most the integral of
the absolute density difference. This proves (NSG.4).
If the rescaled window is inside $D$, every point in it is alive,
so the actual terminal mask changes no local point.
:::

:::{prf:lemma} Primitive density core and exponential star protection
:label: lem-nsg-protected-star

For $N\ge4$ define the actual preparation event
$\mathcal C_M=\{N^{-1}\sum_j|m_j|^p\le M^p/2\}$.
Then $P(\mathcal C_M^c)\le2H_p/M^p$.
For tags in $B(0,R_0)$ and rescaled windows inside $B(0,R_0+1)$,
both the full and the omitted-tag spatial sums have the lower bound

$$
\beta(M,R_0,\tau)=\tfrac14L_{\tau,0}
              e^{-(R_0+1+M)^2/(2\tau^2)}>0
\tag{NSG.5}
$$

on $\mathcal C_M$.
There is a fixed set of $L_d\le9^d$ directions such that, if each
ball $B(Rv_\ell/2,R/8)$ contains a rescaled non-tag site, the tag's
Voronoi cell is contained in $B(0,R)$ and its entire Delaunay star
is determined by the sites in $B(0,2R)$.
For $r_NR\le1$, conditional on $\mathcal F,Y_i=x$ and $\mathcal C_M$,
failure of that protection has probability at most

$$
L_d\exp[-\beta(M,R_0,\tau)v_d(R/8)].
\tag{NSG.6}
$$

The homogeneous comparison Poisson process satisfies the same
bound with its actual intensity $\rho_N(x)\ge\beta$.
For alive-only geometry require also $2r_NR<\operatorname{dist}(x,D^c)$.
:::

:::{prf:proof}
The moment estimate and Markov give the first assertion. At least
$N/2$ centers have norm at most $M$ on $\mathcal C_M$; after deleting
the tag at least $N/4$ do. Their Gaussian densities on the stated
unscaled window are at least
$L_{\tau,0}e^{-(R_0+1+M)^2/(2\tau^2)}$. This proves (NSG.5).

Choose a maximal $1/4$-separated subset of the unit sphere. Its
$1/8$ balls are disjoint inside $B(0,9/8)$, so its cardinality is
at most $9^d$ and it is a $1/4$-net. At any $y=Ru$ on the sphere,
choose $|u-v_\ell|\le1/4$. A site $z$ in the designated ball gives
$|y-z|\le R/2+R/8+R/8=3R/4<R=|y|$.
Consequently no sphere point belongs to the tag cell. Convexity
then puts its entire cell in $B(0,R)$, exactly as in
{prf:ref}`lem-native-jg-protected-cell`. Sites outside $B(0,2R)$
cannot affect this cell: for $|y|\le R$ they have distance greater
than $R$, whereas the tag has distance at most $R$. Every incident
facet neighbor is inside $B(0,2R)$.

For each designated ball the sum of the other rows' landing
probabilities is at least $\beta v_d(R/8)$, since $Nr_N^d=1$.
Conditional independence bounds its empty probability by the
exponential of the negative sum. Union bound gives (NSG.6).
The Poisson empty probability has the same expression. The interior
condition makes every required site alive and excludes any influence
of a deleted exterior site on a protected cell.
:::

:::{prf:theorem} Native random Delaunay-star limit without stationary chaos
:label: thm-nsg-native-star-limit

Let $\mathcal S_{N,i}$ be the exact rescaled Euclidean Delaunay star
at the actual terminal tag, including its neighbor positions,
Voronoi cell and every specified measurable local functional.
Let $\mathcal S(\Pi_c)$ denote the same construction with a point at
the origin and homogeneous Poisson sites of intensity $c>0$.
Conditional on $\mathcal F,Y_i=x\in B(0,R_0),\mathcal C_M$,

$$
\|\mathcal L(\mathcal S_{N,i}\mid\mathcal F,Y_i=x)
-\mathcal L(\mathcal S(\Pi_{\rho_N(x)}))\|_{\rm TV}
\le E_N(R)+2L_d e^{-\beta v_d(R/8)},
\tag{NSG.7}
$$

where $E_N(R)$ is the right side of (NSG.4) for $W=B(0,2R)$,
provided $r_NR\le1$. For alive-only stars impose the interior
condition above. First $N\to\infty$, then $R\to\infty$ proves
uniform convergence on this primitive center core. Then
$M\to\infty$ removes that core by (NSG.1). Native tag tails are
removed by the same primitive Gaussian moment bound.

This is a comparison kernel with random intensity determined by
the actual preparation and tag. If along a subsequence
$(\mathcal F\text{-center empirical law},Y_i)$ converges to
$(\mathsf M,X)$, its limit star conditional on those variables is
$\mathcal S(\Pi_{\rho_{\mathsf M}(X)})$, where
$\rho_{\mathsf M}=\varphi_\tau*\mathsf M$.
The spatially rescaled covariance and first row mean satisfy
along that same jointly convergent subsequence

$$
r_N^{-2}C_i\ \Rightarrow\ C_*=
 \frac1{K_*}\sum_{z\sim0}zz^T,\qquad
r_N^{-1}\frac1{k_i}\sum_{j\sim i}(Y_j-Y_i)
\ \Rightarrow\ A_*=
 \frac1{K_*}\sum_{z\sim0}z.
\tag{NSG.8}
$$

$K_*$ is finite, $C_*$ is positive definite almost surely, and
$A_*$ is nonzero with positive probability. In particular the
native local covariance does not concentrate to a deterministic
matrix solely from $N\to\infty$. This result retains local
retessellation, without a deterministic simplex-angle assumption.
:::

:::{prf:proof}
Use the point-process coupling in the two preceding lemmas. If the
two configurations agree inside $B(0,2R)$ and both are protected,
their cells, stars and all their measurable functions are identical.
The coupling failure bound is (NSG.7). A homogeneous Poisson process
has finitely many sites in every bounded ball, by its Poisson count;
its star is bounded almost surely by (NSG.6) and $R\to\infty$.
It therefore has finite degree. Conditional point densities are
continuous, so every finite collection including the origin is in
general position almost surely: each relevant affine-dependence or
cosphere determinant is a nonzero polynomial, whose zero set has
Lebesgue measure zero by induction and Fubini. The conditional
Gaussian sample has the same property.

If all neighbor vectors belonged to a proper linear subspace, the
Voronoi inequalities would put no bound on a perpendicular ray.
The cell would be unbounded. Thus the neighbor vectors span
$\mathbb R^d$ and $C_*$ is positive definite.

There is a positive-probability asymmetric star. Place one site near
each vector $e_1,\ldots,e_d,-2\sum_ae_a$. These $d+1$ vectors
positively span the space and their bisector inequalities define a
bounded simplex with all its facets present. Choose finite $R$
containing that simplex and its small perturbations, with all those
vectors inside $B(0,2R)$. The Poisson event of one point in each
small disjoint neighborhood and no other point in $B(0,2R)$ has
strictly positive probability. Outside sites cannot change the
cell. Its neighbor mean is near
$-(d+1)^{-1}\sum_ae_a$, hence bounded away from zero.
This proves the stated first-moment assertion. Scaling all these
neighborhoods by two also gives a positive-probability different
covariance neighborhood; $C_*$ is consequently nonconstant.

Equations (NSG.8) are literal functions of the rescaled star.
The subsequential statement follows directly from (NSG.7): on
compact sets Gaussian convolution is continuous under weak center
convergence, by the bounded continuous integrands and their uniform
derivative bounds. Poisson stars at intensities converging to
$c>0$ can be coupled to agree on each bounded window by their
common minimum intensity; protection removes the window cutoff.
No uniqueness or deterministic stationary phase is used.
:::

:::{prf:corollary} An evaluated fixed-reference local geometry schedule
:label: cor-nsg-reference-schedule

Keep every primitive parameter of the unchanged count or declared row
reference fixed: $d=3$, $h=.04$, $\gamma=b_O=1$,
$\sigma_x=\sigma_J=.1$, $V=2$, $\alpha_{\rm col}=.5$,
$\lambda=1$, $\nu=.3$, $\rho=1$, $D=[-2,2]^3$.
Its actual values are

$$
\tau^2=.00041537673072267287\ldots,\quad
a_x=.999215684224339\ldots,\quad
b=.03921578878304646\ldots,\quad H_4\le211.801576517.
$$

For $N\ge3$ choose proof radii
$M_N=(\log N)^{1/8}$ and
$R_N=N^{1/[4d(d+1)]}$. On any fixed compact tag set the star
comparison error, after integrating the original preparations,
is at most

$$
\frac{2H_4}{(\log N)^{1/2}}
+O\bigl(N^{-1+1/[2(d+1)]}+N^{-3/(4d)}\bigr)
+2L_d\exp[-N^{1/[4(d+1)]-o(1)}],
\tag{NSG.15}
$$

and tends to zero. Constants are the explicit ball-volume and
Gaussian derivative constants above, with the compact tag radius.
For interior alive-only geometry $2r_NR_N$ eventually lies within
its fixed boundary margin. This is an evaluated regime of the
unchanged algorithm; the proof radii restrict estimates only.
The local degree remains a finite random $K_*$ in this regime.
:::

:::{prf:proof}
The displayed numbers follow by substitution in the actual OU,
two-drift and collision formulas. On $\mathcal C_{M_N}$,
$-\log\beta(M_N,R_0,\tau)=O((\log N)^{1/4})=o(\log N)$,
with its fixed, possibly large reference coefficient.
For $W=B(0,2R_N)$ the first term of (NSG.4) is
$O(R_N^{2d}/N+R_N^d/N)$ and the derivative term is
$O(N^{-1/d}R_N^{d+1})$. These give the two powers in (NSG.15).
The protection exponent is
$\beta v_d(R_N/8)=N^{1/[4(d+1)]-o(1)}$.
Markov gives the first term. Finally $r_NR_N\to0$.
No innovation is bounded or removed; no Gaussian lower-density
constant is asserted to be numerically large.
:::

(sec-nsg-local-likelihood)=
## 4. The actual local spatial likelihood and its source action

:::{prf:theorem} Native local point-pattern action and O-source first variation
:label: thm-nsg-local-spatial-action

Fix $\mathcal F$, the tag $Y_i=x$ and a bounded Borel window $W$.
Use the existing addressed O-source probe $\xi_j^O\mapsto
\xi_j^O+\theta f_j$, with its actual mean/variance laws,
$f_i=0$ and fixed $\max_j|f_j|\le F_f<\infty$ on this preparation
fiber. The unchanged executed kernel is $\theta=0$; this probe
is a derivative of its likelihood, not a changed gas configuration.
The terminal spatial mean shift is exactly $tq\theta f_j$.
Let

$$
a_j^\theta(u)=\frac1N\varphi_\tau
 (x+r_Nu-m_j-tq\theta f_j),\qquad
p_j^\theta=\int_Wa_j^\theta(u)\,du.
$$

Relative to the point-configuration reference measure
$du_1\cdots du_k/k!$ in the sector of $k$ points, the native
conditional Janossy density is exactly

$$
J_{N,k}^\theta(u_1,\ldots,u_k)=
\sum_{j_1,\ldots,j_k\ne i\ {\rm distinct}}
  \prod_{a=1}^k a_{j_a}^\theta(u_a)
  \prod_{l\notin\{i,j_1,\ldots,j_k\}}(1-p_l^\theta).
\tag{NSG.16}
$$

For every fixed $k$, bounded window $W$, and compact source
interval, it obeys, together with its first $\theta$ derivative,
the uniformly evaluated approximation

$$
J_{N,k}^\theta=
e^{-c_N(\theta)|W|}c_N(\theta)^k+O(r_N+N^{-1}),
\quad
c_N(\theta)=\frac1N\sum_{j\ne i}
       \varphi_\tau(x-m_j-tq\theta f_j).
\tag{NSG.17}
$$

The constants depend on $k,W,\tau,tq,F_f$ and the source
interval, through the Gaussian derivative bounds through order two;
they are uniform in every realized center. On the primitive center
core and compact tag set, $c_N(\theta)$ has the explicit positive
lower bound obtained from (NSG.5) by replacing $M$ by
$M+tqF_f\sup|\theta|$. Therefore the conditional native local
negative log-density and its first variation satisfy

$$
-\log J_{N,k}^\theta
=|W|c_N(\theta)-k\log c_N(\theta)+O(r_N+N^{-1}),
$$
$$
-\partial_\theta\log J_{N,k}^\theta
=c_N'(\theta)[|W|-k/c_N(\theta)]+O(r_N+N^{-1}),
$$
$$
c_N'(\theta)=-\frac{tq}{N}\sum_{j\ne i}
f_j\cdot\nabla\varphi_\tau(x-m_j-tq\theta f_j).
\tag{NSG.18}
$$

For any bounded measurable function $G$ of this local point
configuration, including its truncated tessellation, the exact
native source response at zero is

$$
\left.\partial_\theta\mathbb E_\theta G\right|_0
=\mathbb E_0\!\left[G\sum_{j\ne i}f_j\cdot\xi_j^O\right]
=\mathbb E_0\!\left[G\frac{tq}{\tau^2}
                   \sum_{j\ne i}f_j\cdot(Y_j-m_j)\right].
\tag{NSG.19}
$$

Thus the local Poisson action and score above are derived from
the existing Gaussian spatial likelihood. They identify neither
a Yang--Mills nor an Einstein--Hilbert probability action.
No convergence of rescaled unbounded moments, global-star
derivatives or native color marks is asserted by this local result.
:::

:::{prf:proof}
Each independent row either contributes its one point with density
$a_j^\theta$ or misses $W$ with probability $1-p_j^\theta$.
Sum over all injective assignments of source labels to the $k$
unordered points. Dividing the reference measure by $k!$ makes
the ordered-assignment sum exactly (NSG.16). It integrates to
the original point-count probabilities in every sector.

Uniformly on the stated source interval,
$a_j^\theta\le L_{\tau,0}/N$ and
$|\partial_\theta a_j^\theta|\le tqF_fL_{\tau,1}/N$.
Consequently $p_j=O(N^{-1})$, $|p_j'|=O(N^{-1})$ and their
sums are bounded, with constants independent of the centers.
For sufficiently large $N$ every $p_j<1/2$. Factor (NSG.16) as

$$
\left[\prod_{j\ne i}(1-p_j)\right]
\sum_{j_1,\ldots,j_k\ne i\ {\rm distinct}}
     \prod_a\frac{a_{j_a}(u_a)}{1-p_{j_a}}.
$$

The first factor differs, in value and first source derivative,
from $\exp(-\sum_jp_j)$ by $O(N^{-1})$, since
$\sum_jp_j^2+\sum_j|p_jp_j'|=O(N^{-1})$.
Replace each $1/(1-p_{j_a})$ by one at error $O(N^{-1})$.
Allow all labels instead of distinct labels at the same order:
for any specified colliding pair,
$\sum_ja_j(u_a)a_j(u_b)\le L_{\tau,0}^2/N$;
the remaining $k-2$ sums are bounded. For the derivative,
differentiating one factor preserves that order using the uniform
$a_j'$ bound. There are at most $k(k-1)/2$ colliding pairs.
The unrestricted sum is exactly
$\prod_a\lambda_N^\theta(u_a)$, with
$\lambda_N^\theta=\sum_{j\ne i}a_j^\theta$.

Gaussian derivative bounds now give uniformly for $u\in W$

$$
|\lambda_N^\theta(u)-c_N(\theta)|
 \le r_NL_{\tau,1}|u|,
\quad
|\partial_\theta\lambda_N^\theta(u)-c_N'(\theta)|
 \le r_NtqF_fL_{\tau,2}|u|.
$$

Their integrals replace $\sum_jp_j$ by $|W|c_N$ at the same
orders. This proves the $C^1$ approximation (NSG.17).
On the center core, at least $N/4$ omitted-tag centers have norm
at most $M$, and their source shifts have norm at most
$tqF_f\sup|\theta|$. Their densities give the stated positive
floor. The limiting sector density is consequently bounded away
from zero for fixed $k$; taking its logarithm and derivative proves
(NSG.18). This uses all points, with no independence assumption
about the original preparations.

For (NSG.19), the independent addressed O normals have the exact
likelihood derivative $\sum_{j\ne i}f_j\cdot\xi_j^O$.
The tag is unaffected because $f_i=0$; its conditioning leaves
the other O/final draws independent. Dominated differentiation
for bounded $G$ follows from finite Gaussian exponential moments.
Conditional Gaussian regression for
$Y_j-m_j=tq\xi_j^O+s\xi_j^x$ gives
$\mathbb E[\xi_j^O\mid\mathcal F,Y_j]
=tq(Y_j-m_j)/\tau^2$ and proves the second equality.
This does not make the completed velocities or colors independent
of the O draws; it is a likelihood projection for spatial tests.
:::

(sec-nsg-ridge)=
## 5. Exact native ridge regimes and their random metric limit

:::{prf:theorem} Absolute and relative covariance-ridge geometry
:label: thm-nsg-covariance-ridge-limits

Use the native star regime above with fixed ridge $\eta>0$.
The distributional displays use the same jointly convergent
center/tag subsequence; no deterministic unique stationary density
is implicit in $C_*$. The absolute convergence in probability
does not require choosing such a subsequence.
In the absolute branch, if $\theta_T<1$, the native pseudo-inverse
eventually retains all local directions with probability tending
to one. Put $g_*=\operatorname{clip}_{[b_-,b_+]}(\eta^{-1})$,
interpreting absent bounds as absent tests. For `Clipped` policy,

$$
g_i\longrightarrow g_*I \quad\hbox{in probability}.
\tag{NSG.9}
$$

If $\eta^{-1}$ is strictly inside all present bounds, the first
nonconstant correction has the native random limit

$$
r_N^{-2}(g_i-\eta^{-1}I)\Rightarrow-\eta^{-2}C_*.
\tag{NSG.10}
$$

A strict upper clamp $b_+<\eta^{-1}$ instead makes $g_i=b_+I$
with probability tending to one. A lower clamp
$b_-\ge\eta^{-1}$ makes $g_i=b_-I$ whenever all inverse directions
are retained. Equal bounds give that exact identity including
repaired directions. `Strict` accepts precisely its existing
no-repair event; a saturated clamped limit is not a successful
strict metric.

In the relative-trace branch put $\tau_*^C=\operatorname{tr}C_*/d$.
If the primitive numerical test

$$
\eta>\theta_T(d+\eta)
\tag{NSG.11}
$$

holds, no positive-trace covariance direction is discarded by the
actual pseudo-inverse rule. The complete relative `Clipped` limit is

$$
r_N^2g_i\Rightarrow G_*=
Q\operatorname{diag}\left[
\operatorname{clip}_{[b_-/\tau_*^C,b_+/\tau_*^C]}
       \frac1{\lambda_k(C_*+\eta\tau_*^C I)}\right]Q^T.
\tag{NSG.12}
$$

This is a positive native random metric when the lower clamp is
positive or absent; the unclamped inverse is already positive.
The actual zero-trace fallback is absent from the limiting star,
because $C_*$ is positive definite. If
$b_-\le1/(d+\eta)$ and $b_+\ge1/\eta$, with absent bounds ignored,
no covariance spectrum ever needs a clamp on a positive-trace row.
Then `Strict` and `Clipped` have the same successful spectrum.
In particular $d=3$, $\eta=10^{-5}$, default $b_-=10^{-6}$,
no upper bound and binary64 $\theta_T=3\epsilon_T$ satisfy these
tests; the relative metric is the literal unmodified inverse.

With the fixed determinant floor $\delta_{\det}>0$ and executed
diffusion floor $f_D=\max(b_-,10^{-6})$, where an absent lower
bound contributes $10^{-6}$, the same relative regime has

$$
r_N^d\sqrt{\max(\det g_i,\delta_{\det})}
       \Rightarrow\sqrt{\det G_*},\qquad
r_N^{-1}\widetilde g_i^{-1/2}\Rightarrow G_*^{-1/2}.
\tag{NSG.13}
$$

For `VolumeKind::RiemannianCell`, the native cell volume times this
determinant density instead converges to
$\operatorname{vol}(\mathcal V_*)\sqrt{\det G_*}$.
`SqrtDetMetric` keeps the distinct $r_N^{-d}$ density scaling.
Here $f_D$ is the fixed code floor formed from the unscaled
configured `min_eig`, even in the relative branch. It is not the
metric's scaled clamp $b_-/\tau_i^C$.
:::

:::{prf:proof}
Write the exact rescaled covariance $\widehat C_i=r_N^{-2}C_i$.
Its law is tight by the star theorem, and it converges to $C_*$.
For absolute ridge the eigenvalues of $C_i+\eta I$ tend to $\eta$.
Their ratios tend to one; $\theta_T<1$ therefore retains them.
Scalar reciprocal and clipping give (NSG.9). On an unclamped
neighborhood,
$(\eta I+C_i)^{-1}=\eta^{-1}I-\eta^{-2}C_i+O(\|C_i\|^2)$,
which gives (NSG.10) by covariance tightness. Strict clamp gaps and
the inequality $(\eta I+C_i)^{-1}\preceq\eta^{-1}I$ give the
exact saturation assertions. The code reports a moved clamp as a
repair and `Strict` rejects it.

For relative ridge with $\tau_i^C>0$, exactly
$\tau_i^C=r_N^2\operatorname{tr}\widehat C_i/d$ and
$C_i+\eta\tau_i^C I=r_N^2[\widehat C_i+
\eta(\operatorname{tr}\widehat C_i/d)I]$.
Since $0\preceq C_i\preceq d\tau_i^C I$, every regularized
eigenvalue lies in $[\eta\tau_i^C,(d+\eta)\tau_i^C]$.
Test (NSG.11) excludes the pseudo-inverse cutoff for every such row.
Inversion, the actual scaled bounds and this exact homogeneity
give (NSG.12). The scalar clipping map is continuous, including
its boundary, so covariance convergence suffices. The same spectral
interval proves the sufficient no-clamp tests and the stated
default arithmetic evaluation.

The limit is positive definite almost surely. Every fixed floor
in its determinant and eigenvalues is negligible after the
respective exact scaling: $r_N^{2d}\delta_{\det}\to0$ and
$r_N^2 f_D\to0$. Continuous matrix square root/inverse on the
positive-definite cone proves (NSG.13), by localization to compact
spectral subsets. The rescaled Voronoi cell agrees under the
protected-star coupling; its volume scales exactly as $r_N^d$.
Multiplying the two factors proves the cell-volume assertion.
These are distributional limits, not assertions of unproved
inverse-covariance moments.
:::

(sec-nsg-actual-weights)=
## 6. Configured Gaussian CSR weights and differential moments

:::{prf:theorem} Actual sparse Gaussian rows retain a random finite star
:label: thm-nsg-sparse-kernel-moments

Use `WeightMode::Kernel` with normalization, its literal
$10^{-12}$ row floor, and configured lengths $\ell_N>0$.
Use the same jointly convergent center/tag subsequence for the
distributional displays.
If $\ell_N/r_N\to L\in(0,\infty]$, the actual rescaled row
moments have the conditional native limits

$$
r_N^{-1}\sum_{j\sim i}w_{ij}(Y_j-Y_i)\Rightarrow
A_{*,L}=\sum_{z\sim0}\omega_L(z)z,
$$
$$
r_N^{-2}\sum_{j\sim i}w_{ij}(Y_j-Y_i)(Y_j-Y_i)^T
\Rightarrow C_{*,L}=\sum_{z\sim0}\omega_L(z)zz^T,
\tag{NSG.14}
$$

where

$$
\omega_L(z)=
\frac{e^{-|z|^2/(2L^2)}}{
 \max(\sum_{u\sim0}e^{-|u|^2/(2L^2)},10^{-12})},
\qquad \omega_\infty(z)=1/K_*.
$$

For every such $L$, $A_{*,L}$ is nonzero with positive probability.
In particular the schedule $N\ell_N^d/\log N\to\infty$ has
$L=\infty$ and uniformly weighted local stars; it does not produce
the full-kernel degree $N\ell_N^d$.
If $\ell_N/r_N\to0$, all raw edge weights tend to zero and the
fixed row floor instead gives
$\sum_jw_{ij}\to0$ in probability, with both rescaled moments
tending to zero.

At differential normalization $b_N=r_N^{-2}$ in the $L>0$ regimes,
the actual first row moment is not tight: it is $r_N^{-1}$ times
a vector with a positive-probability nonzero limit. Its second
row moment has the nonconstant limit $C_{*,L}/2$ under the
Taylor convention in
{prf:ref}`prop-native-jg-operator-remainder`.
At normalization $b_N=2/\ell_N^2$ with $L=\infty$, the second
row moment instead tends to zero. Thus the claimed deterministic
elliptic row-moment identification is not supplied by increasing
the number of sites in a kernel ball absent from this CSR graph.
:::

:::{prf:proof}
Under the protected-star coupling, write each edge exactly as
$Y_j-Y_i=r_Nz_j$. Its raw Gaussian factor is
$\exp[-|z_j|^2/(2(\ell_N/r_N)^2)]$.
The star is finite, all its edge lengths are strictly positive,
and its law is tight. For $0<L\le\infty$, continuity and the
literal denominator give (NSG.14). For $L=\infty$ its raw sum
tends to the integer $K_*\ge d+1$, so the floor is inactive.
For $L=0$ every raw factor tends to zero and the finite sum
eventually uses the floor; all normalized factors tend to zero.

For the nonzero-mean event use the asymmetric simplex star in the
previous proof. The weighted sum is proportional to
$[e^{-1/(2L^2)}-2e^{-2d/L^2}]\sum_ae_a$ at that prototype.
If this coefficient vanishes at a particular finite $L$, replace
the last vector by $-a\sum_ae_a$ with $a>0$ different from one;
the coefficient $e^{-1/(2L^2)}-a e^{-a^2d/(2L^2)}$ is not
identically zero in $a$ (it tends to a positive number as
$a\to\infty$). The positively spanning simplex and its
positive-probability neighborhoods give a strict nonzero mean.
The denominator, including its floor, remains positive. For
$L=\infty$ the earlier asymmetric star already suffices.

Choose a closed neighborhood on which the limiting mean norm is
greater than $c>0$ and whose probability is positive. For any
fixed $M$, sufficiently small $r_N$ makes $c/(2r_N)>M$.
The coupling bound then gives a fixed positive lower bound on
the probability that the differential first moment exceeds $M$.
This contradicts tightness. The second-moment statements follow
from (NSG.14) and $r_N^2/\ell_N^2\to0$ for $L=\infty$.
The second-moment limit is also nonconstant: at a scaled simplex
configuration its weighted covariance is positive at any fixed
positive scale, whereas it tends to zero as that scale tends to
zero. Both strict neighborhoods have positive Poisson probability.
All fluctuations arise from actual finite local rows, rather than
from an invented estimator or omitted Gaussian tails.
:::

:::{prf:corollary} Same-record relative metric edge weights and floor regimes
:label: cor-nsg-relative-metric-weights

In the successful relative-ridge regime (NSG.11), jointly construct
the Poisson star, the covariance metrics of its neighbors and their
Voronoi cells from the same point configuration. They all have
finite determining neighborhoods almost surely. Let $G_0,G_z$
be their relative limiting metrics and define the actual limiting
geodesic squared edge length

$$
q_z=z^T(G_0+G_z)z/2>0.
$$

For a fixed configured Riemannian kernel length $\ell>0$,
`RiemannianKernel` has the native limiting weights
$e^{-q_z/(2\ell^2)}/\max(\sum_u e^{-q_u/(2\ell^2)},10^{-12})$.
`RiemannianKernelVolume` with `SqrtDetMetric` has limiting
normalized weights proportional to
$e^{-q_z/(2\ell^2)}\sqrt{\det G_z}$; its common factor
$r_N^{-d}=N$ makes the floor inactive.
With `RiemannianCell` the weights instead use the same-record
factor $\operatorname{vol}(\mathcal V_z)\sqrt{\det G_z}$ and
retain the fixed floor.

If the configured Riemannian length tends to zero, the
`RiemannianKernel` row mass tends to zero by its literal floor.
For the `SqrtDetMetric` volume-weighted arm the same conclusion
holds under $\ell_N^2\log N\to0$, since its raw logarithm is
$\log N-q_z/(2\ell_N^2)+O_P(1)$ on localized stars.
These conclusions retain every correlation between sites,
metrics, volumes and weights.
:::

:::{prf:proof}
Every point of a homogeneous Poisson configuration has a bounded
Voronoi cell almost surely. One may verify this directly for each
bounded-window point: conditional on its location, the remaining
Poisson points outside that window are still Poisson, and the
finite directional empty-ball estimate makes an unbounded cell
probability zero. There are countably many points; union bound
over them gives the simultaneous assertion. The root has finitely
many neighbors, so their stars and cells are determined inside
some finite random ball. Truncate this ball and use (NSG.4);
then remove the truncation. This proves joint same-configuration
convergence for every stated functional. It assigns no independent
copy to a neighboring metric.

The actual symmetric edge metric is $(g_i+g_j)/2$. Its squared
length equals
$(r_Nz)^T[r_N^{-2}(G_0+G_z)/2](r_Nz)$ in the limit, giving $q_z$.
The two volume scalings were proved in (NSG.13). Substitute these
identities into the literal weighting formulas and row floor.
For shrinking length, $q_z>0$ on each finite localized star;
the exponential tends to zero. In the determinant-density arm,
$\ell_N^2\log N\to0$ makes its negative exponential dominate
the density factor $N$. Spectral and star localization remove
the boundedness restrictions in probability. No inverse-moment
or independence assertion is needed.
:::

(sec-nsg-selection)=
## 7. Selected laws, finite arithmetic and the remaining joint obligations

:::{prf:theorem} QSD and survival transfer for the same geometric record
:label: thm-nsg-qsd-transfer

Retain the evaluated finite-$N$ QSD regime and the actual row
landing floor $a_*>0$ in
{prf:ref}`thm-native-stationary-closure-qsd-poincare`.
The geometric assertions above apply to the raw output from
an entering QSD; their transfer to its current survivor-selected
geometry changes any bounded-event probability by at most
$e_N=(1-a_*)^N$. For a path of $T_N$ actual updates started from
that QSD and conditioned on survival through $T_N$, the full
record-law difference is at most $T_Ne_N$. Consequently every
proved local geometric convergence above persists under that
selection whenever $T_Ne_N\to0$.

Alive-only and all-slot stars agree on the interior protected
events in the local theorem. The current-time selection does
not replace $\rho_N$ by an iid density. Centers can remain
random, and the Poisson comparison is conditional on them.
If the returned numerical graph has disagreement probability
$\delta_N^{\rm graph}$, all corresponding returned-payload
error estimates acquire that additional probability, and any
strict-metric rejection remains its actual recorded error mark.
No vanishing $\delta_N^{\rm graph}$ is asserted from absolute
continuity alone.

Metric eigensolves, sums, determinants, cell volumes and weights
have their separate arithmetic comparison. For any chosen scaled
payload $A_N$ above and tolerance $u>0$, define its actual error
probability
$\delta_N^{\rm payload}(u)=P\{\text{graph agrees, and }
|\widehat A_N-A_N|>u\text{ or validity marks differ}\}$.
For bounded Lipschitz payload tests the returned/exact expectation
difference is at most
$2\|f\|_\infty(\delta_N^{\rm graph}+
\delta_N^{\rm payload}(u))+\operatorname{Lip}(f)u$.
These are distinct consumed numerical errors. Their vanishing is
not a consequence of graph agreement or real-coordinate density.
:::

:::{prf:proof}
The raw law, including all terminal positions and its retained
history, restricts to the current survivor law by the actual
terminal event. Its QSD identity and landing floor give
$1-\alpha_N\le e_N$. The exact full-law defect and path-law
selection calculation in
{prf:ref}`thm-native-stationary-closure-full-law-defect` give
the two total-variation bounds. Projection or any bounded
measurable geometric construction cannot increase them.
Apply those bounds to the coupled star failures, covariance and
weight tests above. The equality of local alive/all-slot geometry
is already deterministic on the protected interior event.

For a numerical graph, on the event of agreement its local
functional is the one proved above; off it use its actual
failure/repair/alternative payload. Union bound adds
$\delta_N^{\rm graph}$. On that graph-agreement event compare
the returned arithmetic payload to the exact configured estimator
separately. Off its tolerance event the test difference is bounded
by $2\|f\|_\infty$; on it the Lipschitz estimate is at most
$\operatorname{Lip}(f)u$. This proves the displayed arithmetic
bound and distinguishes analytical native geometry from unproved
numerical comparison. It makes no unconditional successful
tessellation or exact eigensolve claim.
:::

:::{prf:remark} Discharged native geometry and remaining scale identifications
:label: rem-nsg-obligation-register

The actual conditional Gaussian output now gives a
total-variation local Poisson approximation, a protected random
Delaunay star, its full unweighted covariance law, exact native
absolute/relative ridge scalings, and correlated metric/volume
Gaussian CSR weights. These hold for the unchanged fixed-step
quadratic count/row reference, with its actual cloning, revival,
cap and unbounded noises; QSD selection has an explicitly
vanishing defect. The relative metric is random at its native
spacing scale. The full-kernel estimate uses the different
existing dense weight map and supplies no CSR concentration.
The local point-pattern likelihood and its O-source score are
also derived explicitly, retaining their actual random density
and finite-window reference measure.

For a deterministic continuum differential operator, the literal
CSR row has a proved positive-probability uncancelled first
moment. A weak assembled finite-volume operator, the actual
facet-area weights, multi-star averaging, a declared physical
calibration or another already configured readout therefore needs
its own calculation; none has been substituted here. A field
limit must also retain the joint native color/velocity marks,
whose O/B2 correlations are not removed by this spatial
marginal argument. Numerical graph agreement, feedback
variants, shrinking-time families without an evaluated
Gaussian/protection budget, global inverse-metric moments and
physical action/reconstruction retain their explicit obligations.

Sources inspected: the complete geometry and stationary chapters;
Rust `kinetic.rs`, tessellation `pipeline.rs`, `frame.rs`,
`metric.rs`, `weights.rs`, `linalg.rs`, `degenerate.rs` and
`curvature.rs`; and Python `scutoid/weights.py`. The proofs here
use exact native spatial marginals, elementary independent
Bernoulli/Poisson couplings, protected Voronoi cells and the
configured spectral formulas. No joint LSI, stationary chaos,
local gauge invariance or unknown smooth metric is assumed.
:::
