# Same-record stationary color, geometry and phase scaling

(sec-nhs-register)=
## 1. Original stationary algorithm and composite spatial observation

:::{prf:definition} Complete harmonic stationary spatial register
:label: def-nhs-register

Retain every algorithm, landscape, boundary, arithmetic, noise, donor,
fitness, mask, recording and calibration parameter of
{prf:ref}`def-nhr-register`. Use its proved recording resonance and
$\sigma_x=0$. In particular the actual all-alive constant-fitness
harmonic gas has $\nu=0$, cap `None`, its original unbounded OU noise,
and the configured B1 `RecordedField` potential-force/input-velocity
color. Its derived stationary rows obey
$$
x_i\sim N(0,XI_3),\qquad v_i\sim N(0,VI_3),\qquad
X=\frac{q^2}{\lambda(1-c^2)},\quad
V=\frac{(1-t^2\lambda)q^2}{1-c^2}>0,
\tag{NHS.1}
$$
independently across rows, with positions independent of velocities.
The unused parameters have no effect because their actual gates and
viscous forces vanish. This is a derived full invariant law.

At an original recorded entering position the actual projector is
$$
P_i^\kappa=c_i^\kappa(c_i^\kappa)^\dagger,\qquad
c_i^\kappa=-\frac{x_i}{|x_i|}\odot e^{i\kappa v_i},
\quad \lambda|x_i|>\delta .
\tag{NHS.2}
$$
The original clone mask deletes no row. The separate default viscous
color at $\nu=0$ is unavailable.

Consume the existing book's passive exact full-dimensional open
Euclidean Delaunay/CSR real-geometry tag on these
same recorded positions, its rooted ordered three-cycles
$\mathcal T_i^N$, canonical projector-product transports of
{prf:ref}`thm-npc-native-direct-rotation`, and unit composite Wilson
coefficients of {prf:ref}`def-nswt-register`. Retain their overlap and
finite-value tests, with zero on unavailable composite faces:
$$
H_{ijk}=U_{i\leftarrow k}U_{k\leftarrow j}U_{j\leftarrow i},
\quad W_{ijk}=1-\Re\operatorname{tr}H_{ijk}/3,\qquad
A_N(f)=\frac1N\sum_i f(x_i)
                  \sum_{(j,k)\in\mathcal T_i^N}r_N^{-4}W_{ijk},
\quad r_N=N^{-1/3}.
\tag{NHS.3}
$$
These are the existing composite projector observations. The literal
archive `x+iv` interaction-ray faces retain their different events
and action. No feedback metric, inverse-volume weight or additional
face coefficient is inserted here.
This is the same geometric tag as the exact shielding/local-Poisson
results of Chapters NSPC/NSWT. Rust's rank-tolerant
`GeometryPipeline` is a different original geometry tag: its
positive affine-rank cutoff can project an open Gaussian region.
Gaussian general position alone does not remove that branch.

Let $f$ be continuous with compact support $K$ in the strict chart
$\lambda|x|>\delta$. Choose $r_0>0$ so the closed $4r_0$
neighborhood $K^+$ stays inside that chart, and set
$R_-=\inf_{K^+}|x|>0$, $R_+=\sup_{K^+}|x|$.
These are observation parameters; the Gaussian population is unbounded.
Its original position density is $p=\varphi_{\sqrt X}$.

For a population family each original phase scale is finite and
positive, $\kappa_N=m_{c,N}\ell_{0,N}/\hbar_{c,N}>0$.
Every theorem below states its actual parameter path. It does not
replace a fixed configured phase by a shrinking phase without saying so.
The zero-phase projector used for comparison is the deterministic
limit of (NHS.2); it need not be an allowed finite-$N$ spectroscopy
configuration. Overlap thresholds in the positive theorems are
the original fixed finite values strictly below one.
:::

(sec-nhs-local-loop)=
## 2. Native loop expansion with the correlated velocity marks

:::{prf:lemma} Three-endpoint expansion on the original projector manifold
:label: lem-nhs-endpoint-expansion

Fix a unit real $n$, $Q=I-nn^T$. Suppose three rank-one projectors
have line representatives $n+\epsilon h_a+O(\epsilon^2)$,
$n^\dagger h_a=0$, with locally bounded remainders. Put
$$
B_h=hn^\dagger+nh^\dagger,\qquad
D_1=B_{h_1-h_0},\quad D_2=B_{h_2-h_0}.
$$
Their actual available canonical loop obeys
$$
H-I=-\frac{\epsilon^2}{2}[D_1,D_2]+O(\epsilon^3),\qquad
W=\frac{\epsilon^4}{24}\|[D_1,D_2]\|_F^2+O(\epsilon^5).
\tag{NHS.4}
$$
The bounds are uniform on every bounded endpoint core.

For a root at $x=Rn$ and neighbors $x+r_Nu,x+r_Nw$,
if $\kappa_N/r_N\to\eta<\infty$, the ORIGINAL marks give
$$
h_u=\frac{Qu}{R}+i\eta Q\operatorname{diag}(n)(v_j-v_i),
\qquad
h_w=\frac{Qw}{R}+i\eta Q\operatorname{diag}(n)(v_k-v_i).
\tag{NHS.5}
$$
The two velocity differences have covariance $VI_3$
and individual covariance $2VI_3$. They are not independent.
:::

:::{prf:proof}
Use the same unitary phase frame at all three endpoints, removing
the root's diagonal $e^{i\kappa_Nv_i}$. Overall line phases disappear
from each projector. Taylor expansion of $x/|x|$ and the original
componentwise phase gives (NHS.5), including $Q$.
The joint velocity covariance follows from the three independent
original Gaussian rows, with the common root retained.

For the endpoint calculation write
$P_1=P_0+\epsilon D_1+\epsilon^2E_1+O(\epsilon^3)$,
and similarly $D_2,E_2$.
The exact polar formula gives
$U_{b\leftarrow a}=I-[P_a,P_b-P_a]
-\tfrac12(P_b-P_a)^2+O(\|P_b-P_a\|^3)$.
Multiply its three factors in the actual order. First-order terms
and terms $[P_0,E_a]$ telescope. For tangent projector differences,
$[-[P_0,D_2]][-[P_0,D_1]]=-D_2D_1$.
The remaining quadratic coefficient is
$$
-[D_1,D_2]+\tfrac12(D_1D_2+D_2D_1)-D_2D_1
                 =-\tfrac12[D_1,D_2].
$$
Convergent polar series on a fixed projector half-norm ball bound
the cubic remainder. Since the actual loop is unitary,
$W=\|H-I\|_F^2/6$, proving (NHS.4).
:::

:::{prf:lemma} Original uncut root moments at every bounded critical phase ratio
:label: lem-nhs-root-moments

If $\sup_N\kappa_N/r_N<\infty$, every original root summand
in (NHS.3) has a population-uniform moment of every finite order.
All constants depend only on the full register, the declared
phase-ratio bound and $f,K^+,r_0$, including the original
overlap margin. No original Gaussian source is clipped.
:::

:::{prf:proof}
The position-star bounds of {prf:ref}`lem-nswt-star-moments`
apply with $s_0=\sqrt X$: they use only the derived independent
Gaussian position law. Let $D_i$ be the degree,
$R_i^*=\max\{1,\max_{j\sim i}|x_j-x_i|/(2r_N)\}$ and
$M_i=1+|v_i|+\sum_{j\sim i}|v_j|$.
Conditional on the entire position graph, the velocities are
the original independent $N(0,VI_3)$ variables. Jensen gives
$$
E[M_i^b\mid x]\le
 (D_i+2)^b\,2^{b-1}
 \left(1+V^{b/2}2^{b/2}
             \frac{\Gamma((3+b)/2)}{\Gamma(3/2)}\right)
\tag{NHS.6}
$$
for integer $b\ge1$. Thus every needed joint moment of
$D_i,R_i^*,M_i$ is bounded by the explicit star series
and Gaussian gamma moments.

Choose $\ell>0$ below $r_0,R_-/128$ and small enough that
real radial projectors along edges of length $4\ell$
pass the original fixed overlap margin.
On $2r_NR_i^*\le\ell$, all endpoints stay in $K^+$.
Their differences from the root's real projector are bounded
by $8r_NR_i^*/R_-+2\kappa_NM_i$.
Below a positive constant determined by the overlap margin,
the exact polar series and (NHS.4) cancellation bound
$$
\|H-I\|_F\le C
     (r_NR_i^*/R_-+\kappa_NM_i)^2 .
$$
One can take $C=2^{60}$ on a projector $1/8$ ball:
in line coordinates $(n+\alpha)/\sqrt{1+|\alpha|^2}$ on
$|\alpha|\le1/4$, projector derivatives through order two
are bounded by $100$. Polar-series derivatives through
order two are bounded by the geometric series and its
first two derivatives at $1/4$; product differentiation
of the three factors fits this larger constant.
The first loop derivative vanishes at a diagonal triple.
Enlarge $C$ by the reciprocal fourth power of the
original smaller overlap-smallness margin as necessary.
Above that margin the native bound $W\le2$ gives the
same quartic bound. Unavailable faces contribute zero.
Consequently on this position event
$$
|Z_i^N|\le C'\|f\|_\infty
D_i^2\left(R_i^*/R_-+(\kappa_N/r_N)M_i\right)^4 .
\tag{NHS.7}
$$
Here $C'$ is the explicit square of the loop bound
plus the failed-smallness factor.

The complementary long-position-edge event has exponentially
small probability in $N$ by the original shielding-ball estimate
of Chapter NSWT. Always
$|Z_i^N|\le2\|f\|_\infty N^{10/3}$.
Every power of this polynomial times the exponential tail
is uniformly bounded, with its finite supremum obtained
at the endpoint or its derivative-zero point.
Combine this with (NHS.6)--(NHS.7).
The constants use only the explicit NSWT series, Gaussian
moments, $X,V,R_\pm,r_0$, original overlap margin and
actual phase-ratio bound.
:::

(sec-nhs-critical-action)=
## 3. Complete critical phase/geometry action with original correlations

:::{prf:theorem} Same-record stationary marked spatial action
:label: thm-nhs-critical-action

If the original positive phase family satisfies
$\kappa_N/r_N\to\eta\in[0,\infty)$, then (NHS.3) converges in
$L^2$ to
$$
\mathcal A_\eta(f)=
\frac1{24}\int f(x)p(x)
 E_{\Pi_{p(x)},\,v}
 \sum_{(u,w)\in\mathcal T_*}
     \|[B_{h_u},B_{h_w}]\|_F^2\,dx .
\tag{NHS.8}
$$
Equation (NHS.5) consumes the SAME root velocity in every
face and the original shared neighbor marks. The local
geometry, all faces and all their correlations are retained.

Define three finite positive constants of the existing
unit-intensity rooted CSR star:
$$
c_0=E_{\Pi_1}|\mathcal T_*|,\quad
c_L=\frac13E_{\Pi_1}\sum_{(u,w)\in\mathcal T_*}
                     (|u|^2+|w|^2-u\cdot w),\quad
c_{\rm Del}=\frac13E_{\Pi_1}\sum_{(u,w)\in\mathcal T_*}
                                      |u\times w|^2 .
\tag{NHS.9}
$$
At $x=Rn$ put
$S=Q\operatorname{diag}(n_1^2,n_2^2,n_3^2)Q$,
$T=\operatorname{tr}S=1-\sum_a n_a^4$ and
$T_2=\operatorname{tr}S^2$.
The COMPLETE original Gaussian mark integral is
$$
\mathcal A_\eta(f)=\frac1{24}\int f(x)
\left[
\frac{2c_{\rm Del}}{R^4}p(x)^{-1/3}
+\frac{20\eta^2Vc_LT}{R^2}p(x)^{1/3}
+6\eta^4V^2c_0(T^2-T_2)p(x)
\right]dx,\quad T^2-T_2=6n_1^2n_2^2n_3^2 .
\tag{NHS.10}
$$
:::

:::{prf:proof}
On a protected bounded determining neighborhood, the original
Gaussian position rows have their established rooted local
Poisson limit. Adjoin their independent original Gaussian
velocity marks. Finite-dimensional position/source densities
give the marked limit directly; degree, radius, protection
and mark truncations serve only as proof cores.
The endpoint expansion gives (NHS.8) there.
Two separated protected neighborhoods use disjoint original
rows and marks in the limit. Their binomial joint local
densities converge to the product Poisson densities, so the
bounded-core empirical variance tends to zero. Intersecting
neighborhood probability tends to zero by the original
Gaussian density bound. This is the determining-neighborhood
argument of Chapter NSPC with independent marks appended.

The root moments remove every core in $L^2$ by Jensen and
Cauchy–Schwarz. Bounded truncation followed by monotone
convergence gives the same limiting Poisson star moments.
The protected tetrahedron event of Chapter NSWT proves
positivity of all three constants in (NHS.9); the original
star series proves finiteness.

For the full mark calculation write
$a=Qu/R$, $b=Qw/R$, $m=Q\operatorname{diag}(n)(v_j-v_i)$,
$l=Q\operatorname{diag}(n)(v_k-v_i)$.
Then $E[mm^T]=E[ll^T]=2VS$ and $E[ml^T]=VS$.
The commutator is $F_0+\eta F_1+\eta^2F_2$.
Its real complement blocks are $F_0=ab^T-ba^T$ and
$F_2=ml^T-lm^T$; their line blocks vanish.
The imaginary complement block of $F_1$ is
$i(mb^T+bm^T-al^T-la^T)$, and its line block is
$2i(a\cdot l-b\cdot m)$.
Gaussian second and fourth moment expansion gives
$$
\begin{aligned}
\|F_0\|_F^2&=2R^{-4}[n\cdot(u\times w)]^2,\\
E\|F_1\|_F^2
 &=4V\{T(|a|^2+|b|^2-a\cdot b)
             +3(a^TSa+b^TSb-a^TSb)\},\\
E\|F_2\|_F^2&=6V^2(T^2-T_2),\qquad E F_2=0 .
\end{aligned}
\tag{NHS.11}
$$
For example
$E[|m|^2|l|^2-(m\cdot l)^2]=3V^2(T^2-T_2)$:
the original shared-root covariance $VS$ is retained.
Real/imaginary orthogonality removes the $F_1$ cross terms,
and $EF_2=0$ removes the remaining cross term.

Orthogonal equivariance makes the original star's summed
area tensor and summed quadratic edge tensor scalar,
with scalars $c_{\rm Del},c_L$.
Its $F_0$ sum is $2c_{\rm Del}/R^4$ and its
$F_1$ sum is $20Vc_LT/R^2$ at intensity one.
Intensity $p$ scales these by $p^{-4/3}$ and $p^{-2/3}$;
the face count remains $c_0$.
The original root density and coefficient $1/24$ give
(NHS.10). Finally
$\det(S|_{n^\perp})=n^T\operatorname{adj}
(\operatorname{diag}(n^2))n=3n_1^2n_2^2n_3^2$,
so $T^2-T_2=2\det(S|_{n^\perp})$.
:::

(sec-nhs-subcritical-geometry)=
## 4. Smooth spatial and critical marked sectors

:::{prf:corollary} A positive finite-phase family reaches the nonzero radial field
:label: cor-nhs-radial-field

If $\kappa_N=o(N^{-1/3})$, the original uncut action converges to
$$
\mathcal A_0(f)=\frac{c_{\rm Del}}{24}
      \int f(x)p(x)^{-1/3}\frac2{|x|^4}\,dx .
\tag{NHS.12}
$$
Its local projector is $P_0(x)=nn^T$, connection
$\omega_0=[P_0,dP_0]$ and curvature
$\mathcal F_0=dP_0\wedge dP_0$, with energy $2/|x|^4$.
It solves the original radial weighted vacuum equation
$D_a(p^{-1/3}\mathcal F_{0,ab})=0$ away from the origin.
Its nonzero curvature is reducible: components at a fixed
point are proportional to the tangent-plane $SO(2)$ generator.
It is an included native endpoint, rather than an unrestricted
local $SU(3)$ Yang–Mills ensemble.

For $\eta>0$, (NHS.10) has two positive additional mark terms
on every nonempty open chart. Independent original velocities
and their common-root differences produce those terms;
a smooth interpolation of only $P_0(x)$ does not reproduce them.
:::

:::{prf:proof}
Set $\eta=0$ above. Direct differentiation gives
$\mathcal F_{0,ab}=-\epsilon_{abc}n_cJ_n/R^2$,
$D_aJ_n=0$ and $D_a\mathcal F_{0,ab}=0$, as in Chapter NASI.
The original Gaussian weight has gradient parallel to $n$,
whose contraction with this curvature vanishes.
This proves the weighted equation.
For $\eta>0$, $T>0$ off the coordinate axes and
$T^2-T_2>0$ off the coordinate planes. Their zero sets
have empty interior. The positive $c_L,c_0,V$ give
the stated positive integrated contribution.
:::

(sec-nhs-phase-divergence)=
## 5. Actual supercritical and fixed-phase normalization

:::{prf:theorem} Original phase above the shrinking graph scale
:label: thm-nhs-supercritical-action

For continuous nonnegative $f$ positive on an open subset
of the strict chart:

1. If $\kappa_N\to0$ and $\kappa_N/r_N\to\infty$, then
$$
\left(\frac{r_N}{\kappa_N}\right)^4A_N(f)
\longrightarrow
\frac{V^2c_0}{4}\int f(x)p(x)(T^2-T_2)\,dx>0
\quad\text{in }L^2 .
\tag{NHS.13}
$$
Thus $A_N(f)\to+\infty$ in probability and in expectation.
2. If $\kappa_N\to\kappa\in(0,\infty)$, then
$$
r_N^4A_N(f)\longrightarrow
L_\kappa(f)=
\int f(x)p(x)E_{\Pi_{p(x)},v}
        \sum_{(u,w)\in\mathcal T_*}
               W(n;v_0,v_u,v_w;\kappa)\,dx>0
\quad\text{in }L^2 .
\tag{NHS.14}
$$
This face uses three original phase-marked projectors of common
direction $n$, original overlap masks and original velocity
marks. The normalized $A_N$ again diverges.
These are the composite readout's actual $r_N^{-4}$
normalizations; the literal archive-ray action is separate.
:::

:::{prf:proof}
In the first case take $\epsilon=\kappa_N$ in (NHS.4).
The position tangent has prefactor $r_N/\kappa_N\to0$;
its limiting commutator is $F_2$ of (NHS.11).
The scaled root moment bound is (NHS.7) with $R_i^*/R_-$
multiplied by $r_N/\kappa_N$.
The long-edge term also vanishes:
$\kappa_N\ge r_N$ eventually, so its trivial
$\kappa_N^{-4}$ bound is at most the previous $r_N^{-4}$
bound. The same marked local theorem proves (NHS.13).
Its integral is positive because $f$ is positive on an
open set and the coordinate planes have zero volume.

In the second case no scaled-loop expansion is used.
On a bounded protected core, directions tend to the common
$n$ and original phase marks converge.
Nonzero-overlap masks converge almost surely: their
nonconstant analytic overlap functions have null level
sets under the positive Gaussian density, apart from
identically satisfied tests. The native bound $W\le2$
and original degree moments remove the cores; the
separated-root argument gives (NHS.14).

For strict positivity take $n$ with all components nonzero.
The vectors $Q\operatorname{diag}(n)e_1$ and
$Q\operatorname{diag}(n)e_2$ are independent.
Choose three small original phase changes in these
directions. The endpoint commutator is nonzero, so the
Wilson defect is positive, and every overlap can be
arbitrarily close to one. An open Gaussian velocity
neighborhood preserves the defect and passes every
original fixed overlap threshold below one. Its
probability is positive for every finite $\kappa>0$.
The protected tetrahedron event gives positive geometry
probability. Their actual densities yield $L_\kappa(f)>0$.
The two positive $L^2$ limits prove both divergences.
:::

(sec-nhs-phase-process)=
## 6. Nonzero original color dynamics along the same phase family

:::{prf:theorem} Actual phase-calibrated projector process and positive gap
:label: thm-nhs-calibrated-process

Retain any original positive phase family $\kappa_N\to0$ above.
For distinct components $a,b$, the following bounded passive
function uses only the EXISTING projector entry, availability
mark and its supplied original calibration:
$$
U_\kappa(P)=
A\,\tanh\left[
\frac1{\kappa\sqrt{2V}}
\arctan\frac{\Im P_{ab}}{\Re P_{ab}}\right],\qquad
A=\mathbf1_{\{\lambda|x|>\delta\}} .
\tag{NHS.15}
$$
It is zero when unavailable; other null entry cases can also
be assigned zero. The arctangent has its principal
$(-\pi/2,\pi/2)$ value. This is a tested function of the
original record, rather than a new algorithmic color.

On the actual stationary row history it converges in $L^2$
at every finite recorded time to
$U_0=A\tanh Y$, $Y=(v_a-v_b)/\sqrt{2V}$.
Set
$$
p_A=\frac{\Gamma(3/2,\delta^2/(2\lambda^2X))}{\Gamma(3/2)},
\quad b_0=\tanh(1)\sqrt{2/\pi}(e^{-1/2}-e^{-2})>0,
\quad
\kappa_*=\frac{\pi}
 {\sqrt{16V\log(4\sqrt{2/\pi}/b_0)}} .
\tag{NHS.16}
$$
Every $0<\kappa\le\kappa_*$ has an original first-Hermite
coefficient at least $p_Ab_0/2$. Its complete Green–Kubo
variance therefore satisfies the primitive positive bound
$$
\Sigma_\kappa\ge
\frac{1+r}{1-r}(p_Ab_0/2)^2>0,\qquad
r=c^{m/2}.
\tag{NHS.17}
$$
The covariance and Green–Kubo series converge to those of
$U_0$. For ANY $N\to\infty$, $n_N\to\infty$, the actual
empirical phase-calibrated color sums have
$$
\frac1{\sqrt{Nn_N}}\sum_{i=1}^N
       \sum_{k=1}^{\lfloor n_N\tau\rfloor}
               U_{\kappa_N}(P_{i,k})
\Longrightarrow \sqrt{\Sigma_0}\,B_\tau
\tag{NHS.18}
$$
as functional path laws on compact native-time intervals.
Their finite-history instantaneous population Gaussian
laws have the corresponding nonzero limiting covariance.
The native transfer-cyclic sector generated by these
observations, including its $\kappa=0$ observation limit,
has exact physical gap $\gamma/(2t_*)$.
:::

:::{prf:proof}
On available rows the real force product in $P_{ab}$
cancels from its imaginary/real ratio. Thus the original
arctangent returns $\kappa(v_a-v_b)$ whenever
$|\kappa(v_a-v_b)|<\pi/2$.
The Gaussian difference is finite almost surely; availability
depends only on the independent original position.
Dominated convergence proves the $L^2$ limit.
Every observation is odd in $Y$ and hence has exact zero
stationary mean. Write $d_\kappa=E[U_\kappa Y]$.
Since $|U_\kappa|\le1$, the exact original Gaussian tail gives
$$
|d_\kappa-d_0|\le
2p_A\sqrt{2/\pi}
          e^{-\pi^2/(16\kappa^2V)},\qquad
d_0=p_AE[Y\tanh Y]\ge p_Ab_0 .
$$
The interval (NHS.16) yields $d_\kappa\ge p_Ab_0/2$.
$Y$ is a normalized first Hermite mode of the original
Mehler operator with eigenvalue $r$. The complete
Green–Kubo identity of
{prf:ref}`thm-nhf-color-green-kubo` proves (NHS.17).
The same spectral mode belongs to the cyclic sector
by its nonzero projection, so its physical gap is exact.

For each lag the $L^2$ convergence and contraction of
the original row transfer give covariance convergence.
Its centered $L^2$ norm is at most $r^k$ at lag $k$;
$|U_\kappa|\le1$ dominates the full covariance series
uniformly by a geometric series.
Finally repeat the independent-row characteristic and
native ordered-fourth-moment proof of
{prf:ref}`thm-nhf-joint-time-limit` for this triangular
observation family. Those error budgets use only its
uniform bound and the SAME fixed $r$, so remain uniform
in $\kappa_N$. The covariance limit is $\Sigma_0>0$;
this proves (NHS.18) on every diverging joint schedule
and the finite-history instantaneous assertion.
:::

:::{prf:remark} Complete regime and physical-sector scope
:label: rem-nhs-scope

These are laws of the existing full stationary algorithm
with same-record marked geometry. Every finite spectroscopy
phase remains positive. The actual phase/spacing ratio
distinguishes the smooth radial, critical marked,
vanishing-supercritical and fixed-positive-phase regimes.

Other initial laws, positive terminal position noise,
nonzero selection/viscosity, caps, graph feedback,
current calibration, inverse-metric weights, finite
rounding/streams and origin passage retain their original
proved laws or remaining estimates. They are not assigned
(NHS.1) or the uncut chart limit.
An overlap threshold at least one has no available faces.
The NHR positive full-history transfer and uniform finite-$N$
gap hold along the positive phase family, but its identification
with the target physical spacetime algebra still requires
that correspondence.
The bounded phase-calibrated observation retains its explicit
$\kappa$ dependence. Fixed continuous functions of projector
entries instead converge to functions of the radial $P_0$;
they need not retain this odd-velocity mode. Thus a vanished
uncalibrated color entry is not an assumed positive physical
field, and a changed observation topology is not hidden.

For completeness a graph-only numerical comparison on the SAME
position/color payload has an exact explicit discrepancy budget.
Let $E_i$ be the event that its actual rooted cycle collection
differs from the exact one, and let $D_i^{\rm num},D_i^{\rm ex}$
be their actual degrees. The native unitary bound gives
$$
E|A_N^{\rm num}-A_N^{\rm ex}|^2
\le4\|f\|_\infty^2r_N^{-8}
 E_{\rm root}\!\left[
  ((D_i^{\rm num})^2+(D_i^{\rm ex})^2)^2
                  \mathbf1_{E_i}\right].
\tag{NHS.19}
$$
It is Jensen applied to the actual root differences, with no
independent graph substitution. A bounded determining-core
comparison instead costs its bounded observation budget times
the actual core discrepancy probability. Neither numerical
test is assumed to vanish here. Rank-projected payloads,
rounded kinetics/colors and fixed resources retain their
additional discrepancies; they do not inherit (NHS.8)
from an exact-geometry Gaussian argument.
:::
