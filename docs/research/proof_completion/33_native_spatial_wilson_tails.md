# Native spatial Wilson moments and uncut compact-chart face limits

(sec-nswt-register)=
## 1. Original execution and the consumed composite readout

:::{prf:definition} Complete spatial face and moment register
:label: def-nswt-register

Retain every field of {prf:ref}`def-nspc-complete-register`:
the quadratic landscape, all preparation parameters, both kicks,
cap, terminal box, arithmetic, color threshold, calibration and
passive open Euclidean Delaunay/CSR instrument. Its included
consensus, $\sigma_x=0$ transition has original terminal positions
$Y_i=tqG_i$, with independent real standard three-dimensional
Gaussian $G_i$ and $t,q,\rho,\nu>0$.
Put $s_0=tq$, $r_N=N^{-1/3}$ and $p=\varphi_{s_0}$.
No original draw or recorded coordinate is truncated.
These analytic results use the existing canonical real-coordinate
dense transition, also bound to Rust `QftExecutionConfig.viscosity`:
its eligible-count denominator or actual positive nonself row mass
is the one used below. The separate Python graph-backed force,
its row floor, and floating underflow/rounded source execution
keep their own tags and native comparison errors.

Use the full matched recorded B2 projector channel and its
actual canonical projector transports and rooted CSR
three-cycles of {prf:ref}`thm-nspc-native-triangle-curvature`.
Retain its original color/overlap availability masks and the
explicit zero-on-unavailable composite observation.
On a defined available face put
$W_{ijk}=1-\Re\operatorname{tr}H_{ijk}/3$, zero otherwise.
This is the existing composite Wilson readout; the literal
archive ray instrument omits its unavailable faces instead.
Any additionally consumed source/clone deletion mask retains
its own channel. The result with that mask includes its actually
proved limiting factor; no force margin silently removes it.
The full matched B2 result here consumes no such extra deletion.
It does not
replace a literal ray link or source interaction triangle.
Let its consumed real coefficients have the primitive
pointwise bound $|\beta_{ijk}^N|\le B_\beta<\infty$.
Existing unit coefficients give $B_\beta=1$; bounded
Gaussian/normalized coefficients retain their actual bound.
Convergence in probability of an inverse-metric coefficient
does not establish this test.

Fix a continuous $f$ with compact support $K$ in a strict
available spatial chart. Choose $r_0\in(0,1/4]$ such that
its closed $4r_0$ neighborhood $K^+$ is compact in that
chart, and put $R_K=\sup_{K^+}|y|$. These are observation
parameters, not support restrictions on the law.
With $B=\rho^2+s_0^2$, $a_*=(\rho^2/B)^{3/2}$ and
$b_*=\rho^2/B$, evaluate the chart test from the proved fields:

$$
F_{\rm c}(y)=\frac{\nu a_*b_*}{t}|y|e^{-|y|^2/(2B)},
\qquad F_{\rm r}(y)=\frac{\nu b_*}{t}|y|,
\qquad
\eta_F=\inf_{K^+}F_\star(y)-\delta_c>0 .
\tag{NSWT.1}
$$

A count annulus can pass when the original EFFECTIVE threshold is below
$\nu a_*b_*\sqrt B/(t\sqrt e)$, subject to the actual box/chart.
Any readout clamp is evaluated first; its input threshold
is not substituted for that effective value.
The zero-threshold result below applies to the canonical
full-field color instrument whose threshold parameter allows zero.
A literal archive instrument clamping that threshold to a
positive floor cannot satisfy its zero-threshold regime.
The row chart is above its explicit radial threshold, with
the same box restriction. At zero threshold any compact
chart avoiding the origin passes.
Retain fixed finite original calibration $\kappa$, every
native face multiplicity and the actually proved local
coefficient/metric regime. Unbounded coefficients and
diverging calibrations retain their separate obligations.
:::

(sec-nswt-finite-star-moments)=
## 2. Primitive finite-binomial star moments

:::{prf:lemma} Uniform moments of original Gaussian Delaunay stars
:label: lem-nswt-star-moments

Condition an original uniform root on $Y_i=x$, $|x|\le R_K$.
The remaining original positions are independent Gaussians.
Let $K_N$ be its Delaunay degree and

$$
R_N^*=\max\left\{1,\frac1{2r_N}
                        \max_{j\sim i}|Y_j-x|\right\}.
$$

The empty maximum is zero. For every nonnegative integers
$a,b$ there is a finite primitive $C_{a,b}$ such that

$$
\sup_{N\ge2,\,|x|\le R_K}
 E_x[(1+K_N)^a(R_N^*)^b]\le C_{a,b}.
\tag{NSWT.2}
$$

Its only inputs are $a,b,R_K,s_0$.
No inverse-volume or inverse-covariance moment is asserted.
:::

:::{prf:proof}
Take a maximal Euclidean $1/8$ separated set of sphere
directions $\theta_l$. It covers the sphere; disjoint
radius-$1/16$ balls inside $B(0,17/16)$ give $M\le17^3$.
For physical $R\le1$ use shielding balls
$C_l(R)=B(x+3R\theta_l/4,R/16)$.
If each contains an original other position, any direction
$\theta$ has one $x+w$ with
$|w|\le13R/16$ and
$\theta\cdot w\ge(3/4)(1-1/128)R-R/16>R/2$.
Its Voronoi inequality along the ray is
$2r\,\theta\cdot w\le|w|^2$, forcing $r<R$.
Thus the root cell lies in $B(x,R)$.
Every original Delaunay neighbor has a bisector meeting
that cell, so is within $2R$ of $x$.
Original Gaussian general position handles all tie/null
cases without changing the geometry.

The original density on $B(0,R_K+1)$ is at least

$$
\beta=(2\pi s_0^2)^{-3/2}e^{-(R_K+1)^2/(2s_0^2)}.
$$

Each shielding ball has mass at least
$\beta v_3R^3/16^3$, $v_3=4\pi/3$.
For $1\le u\le N^{1/3}$ independence and $N-1\ge N/2$
give, with $c=\beta v_3/(2\cdot16^3)$,

$$
P_x(R_N^*>u)\le\min\{1,M e^{-cu^3}\}.
\tag{NSWT.3}
$$

On $R_N^*\le u$ its degree is at most the binomial
count $B_N(u)$ in $B(x,2r_Nu)$, whose mean is at most
$\Lambda(u)=h v_3(2u)^3$, $h=(2\pi s_0^2)^{-3/2}$.
For integer $m$ the factorial-moment expansion proves

$$
E_x(1+B_N(u))^m\le T_m(\Lambda(u)),\qquad
T_m(\Lambda)=\sum_{j=0}^m\binom mj
 \sum_{l=0}^j
 \left\{\begin{matrix}j\\l\end{matrix}\right\}\Lambda^l .
\tag{NSWT.4}
$$

The braces are finite Stirling numbers, including
their order-zero value.
Split $1<R_N^*\le N^{1/3}$ into dyadic intervals.
Cauchy--Schwarz with the binomial $2a$ moment and
(NSWT.3) bounds its contribution by the convergent
primitive series

$$
C_{\rm dyad}=T_a(\Lambda(1))+
\sum_{k=1}^{\infty}2^{kb}\sqrt{T_{2a}(\Lambda(2^k))}
       \sqrt{\min\{1,M e^{-c2^{3(k-1)}}\}} .
\tag{NSWT.5}
$$

Even the last partial dyadic interval has lower endpoint
at most $N^{1/3}$, where the shielding bound applies.

Failure of the physical $R=1$ shield has probability
at most $Me^{-cN}$. Always $K_N\le N-1$ and
$R_N^*\le1+N^{1/3}(R_K+\max_j|Y_j|)/2$.
The original Gaussian moment is

$$
m_\ell=s_0^\ell2^{\ell/2}
                    \frac{\Gamma((3+\ell)/2)}{\Gamma(3/2)}.
$$

Use $E\max_j|Y_j|^{2b}\le(N-1)m_{2b}$, the elementary
power inequality, and Cauchy--Schwarz. A larger bound
for this remaining contribution is

$$
C_{\rm far}=\sqrt M\,2^b(1+R_K^b+\sqrt{m_{2b}})
 \left[1+\sup_{n\ge2}n^{a+b/3+1/2}e^{-cn/2}\right].
\tag{NSWT.6}
$$

The real-axis maximum of $n^\zeta e^{-cn/2}$ occurs
at $2\zeta/c$; its value there and at $2$ bound this
last supremum. This also covers $b=0$.
Thus $C_{a,b}=C_{\rm dyad}+C_{\rm far}$ is finite
and primitive, including all hull-root configurations.
:::

(sec-nswt-force-control)=
## 3. Exponentially probable smooth native fields

:::{prf:lemma} Primitive compact native projector derivative event
:label: lem-nswt-smooth-event

Under (NSWT.1) there are positive primitive constants
$A,c_F,L_1,L_2,L_3,N_0$ such that, for $N\ge N_0$,
with probability at least $1-Ae^{-c_FN}$, every native
projector in $K^+$ is the value of a common smooth
$P_N(y)$ with derivative bounds $L_1,L_2,L_3$.
Its actual force is above threshold by $\eta_F/2$.
The row case also has its actual nonself degree bounded
below by a fixed positive primitive constant.
The same exponential estimate holds uniformly when
one original root is conditioned at $x\in K$.
:::

:::{prf:proof}
Use the original empirical physical-coordinate functions

$$
a_N(y)=\frac1N\sum_l e^{-|y-Y_l|^2/(2\rho^2)},\qquad
b_N(y)=\frac1{tN}\sum_l e^{-|y-Y_l|^2/(2\rho^2)}(Y_l-y).
\tag{NSWT.7}
$$

At an original node its self numerator is exactly zero.
Count force is $\nu b_N(Y_i)$; nonself row force is
$\nu b_N(Y_i)/(a_N(Y_i)-1/N)$.
These common fields therefore equal the ORIGINAL force
at every native endpoint. Their expectations are the
explicit convolutions in (NSWT.1).

For every multiindex through order four the Gaussian
summand derivative is a polynomial times its Gaussian.
It is globally bounded independently of original $Y_l$.
Its primitive bound $M_\alpha^a$ or $M_\alpha^b$ is
computed by expanding its Hermite polynomial, summing
absolute coefficients, and using
$\sup_{r\ge0}r^m e^{-r^2/2}=m^{m/2}e^{-m/2}$, with
value one for $m=0$. Retain every power of $\rho$ and
$t^{-1}$ and the vector component sum.

For fixed tolerance $\epsilon_*>0$, a Euclidean mesh of $K^+$
with covering radius at most
$\epsilon_*/[8(1+\sum_{|\alpha|\le4}(M_\alpha^a+M_\alpha^b))]$
reduces derivative discrepancies through order three
to finitely many bounded independent averages.
A variable bounded by $M$ has centered log moment-
generating function at most $2M^2\theta^2$; bound
its second derivative by $4M^2$ and integrate twice.
Exponential Markov therefore gives the larger bound
$2e^{-N\epsilon_*^2/(128M^2)}$ for a discrepancy
$\epsilon_*/2$. The derivative/mesh union bound is
$Ae^{-c_FN}$ with explicit finite cardinality and
the largest displayed summand bound.
Conditioning one root changes one bounded summand
and its mean by at most $2M/N$. Increasing $N_0$
absorbs this into the tolerance and keeps the same
exponential form uniformly at that fixed root.

In count normalization choose
$\epsilon_*\le\eta_F/(4\nu)$.
For row normalization put
$d_*=\inf_{K^+}Ea_N=a_*e^{-R_K^2/(2B)}>0$.
Require $1/N\le d_*/4$, $\epsilon_*\le d_*/4$.
The quotient estimate

$$
|\Delta(b/a)|\le2|\Delta b|/d_*+
    4(\sup_{K^+}|Eb_N|)|\Delta a-1/N|/d_*^2
$$

gives an explicit smaller tolerance and larger $N_0$
making force discrepancy at most $\eta_F/2$.
Its quotient-rule derivatives through order three
are bounded by finite polynomials in $d_*^{-1}$
and the displayed summand bounds.
The count case needs no degree inverse.
The force floor $\delta_c+\eta_F/2$ bounds all
unit-direction derivatives. Differentiate
$c_N=(F_N/|F_N|)\odot e^{i\kappa y/t}$ and
$P_N=c_Nc_N^\dagger$ through order three.
Finite product-rule bounds give $L_j$, retaining
every force floor and power of $\kappa/t$.
This derives the event from the original law.
:::

(sec-nswt-uncut-limit)=
## 4. Removing the bounded Wilson and geometry tests

:::{prf:theorem} Uncut native compact-chart composite face limit
:label: thm-nswt-uncut-face-limit

Use only the complete register and primitive tests
above, including an actually proved local coefficient
regime. For ALL original rooted CSR three-cycles,

$$
\frac1N\sum_i f(Y_i)
 \sum_{(j,k)\in\mathcal T_i^N}
                \beta_{ijk}^N r_N^{-4}W_{ijk}^N
\longrightarrow
\int f(x)p(x)E_{\Pi_{p(x)}}\left[
 \sum_{(u,v)\in\mathcal T_*}
       \frac{\beta_{uv}^*}{24}
               \|\mathcal F_\kappa(x)(u,v)\|_F^2
 \right]dx
\tag{NSWT.8}
$$

in $L^2$. The limiting sum is absolutely integrable.
The original uniform-root summand has uniform moments
of every finite order. All same-configuration neighbors,
coefficients, metrics and face geometry are retained.
There is no bounded Wilson-input test or protection/
degree cutoff in this conclusion.
:::

:::{prf:proof}
Write $Z_i^N$ for its original root summand.
There are at most $K_N^2$ ordered rooted cycles.
Every defined native transport is unitary, so
$0\le W\le2$, including the original zero mask, and

$$
|Z_i^N|\le2\|f\|_\infty B_\beta
                    N^{4/3}K_N^2
              \le2\|f\|_\infty B_\beta N^{10/3}.
\tag{NSWT.9}
$$

On the smooth event put
$\ell_0=\min\{r_0,[8(1+L_1)]^{-1}\}$.
If $2r_NR_N^*\le\ell_0$, all endpoints and
straight edges lie in $K^+$, their projector
differences have norm below $1/2$, and their
actual direct rotations are well defined.
The exact native second-order transport comparison
gives a finite primitive constant $C_{\rm loop}$ with

$$
\|H_{ijk}^N-I\|_F
              \le C_{\rm loop}r_N^2(R_N^*)^2.
\tag{NSWT.10}
$$

For an explicit larger choice take
$C_{\rm loop}=10^6e^{16L_1\ell_0}
(1+L_1+L_2+L_3)^6$.
To verify this bound, differentiate
$\omega=[P_N,dP_N]$ twice, bound its products
by $L_1,L_2,L_3$, and bound the polar factors
on $\|\Delta P_N\|\le1/2$ by their convergent
binomial series. Sum those absolute second- and
third-order coefficients with their product
coefficients. For instance the exact polar factor
is $(I+\Delta P_N(2P_N-I))(I-(\Delta P_N)^2)^{-1/2}$;
its remainder after order two is at most
$2\sqrt3\|\Delta P_N\|^3$ in Frobenius norm on this
half-norm ball. The projected-transport third-order
remainder is bounded by
$e^{2L_1\ell}(8L_1^3+8L_1L_2+2L_3)\ell^3$
on an edge of length $\ell$. Multiplying its three
edge expansions bounds all quadratic coefficients
by $64(1+L_1+L_2)^4$ and all cubics by
$10^4e^{16L_1\ell_0}(1+L_1+L_2+L_3)^6$,
using total edge length at most $8r_NR_N^*$.
The first-order loop terms cancel;
the third-order spatial remainder is absorbed
using $2r_NR_N^*\le\ell_0$. This is a finite
polynomial/series bound in the displayed primitive
constants, not an independent regularity hypothesis.
The exact identity $W=\|H-I\|_F^2/6$ now yields

$$
|Z_i^N|\le
\frac{\|f\|_\infty B_\beta C_{\rm loop}^2}{6}
                    K_N^2(R_N^*)^4 .
\tag{NSWT.11}
$$

Section 2 bounds every power of this expression.
On the exceptional smooth event (NSWT.9)
contributes only a polynomial times $Ae^{-c_FN}$
to any moment. The long-edge event has probability
at most $M e^{-c(\ell_0/2)^3N}$ for all sufficiently
large $N$ by (NSWT.3). The same polynomial bound
handles it, and (NSWT.9) handles the finitely many
remaining populations. All asserted original
moments are consequently uniform.

Apply the bounded protected determining-neighborhood
comparison of Chapter NSPC on increasing degree,
radius and protection cores, with a bounded continuous
truncation of the original scaled Wilson input.
The native triangle expansion and actual coefficient
limit give (NSWT.8) on each core.
The finite moments just proved remove degree/radius
and Wilson truncations in $L^2$.
The limiting Poisson process has the same moment
bounds: pass finite-root inequalities first for
bounded local truncations, then increase them by
monotone convergence. Its field curvature is
bounded on $K^+$, and its general-position stars
are finite with positive protection almost surely.
Thus protection failures tend to zero as the
core increases. Uniform fourth moments and
Cauchy--Schwarz remove them in $L^2$, on both sides.
For the empirical average, Jensen bounds squared
removal error by the uniform-root squared error.
No independent-root substitution occurs.
The bounded-core empirical theorem then proves
the displayed uncut $L^2$ limit.
:::

:::{prf:remark} Literal action and remaining regimes
:label: rem-nswt-scope

This discharges an unbounded scaled composite Wilson
readout in an included native spatial regime, with
all original faces and bounded configured coefficients.
It does not identify it with executed ray interaction
action. Unbounded inverse-metric coefficients,
full-space passage across the zero-threshold origin,
stationary/multi-time spacetime action and diverging
calibration retain their explicit obligations.
Original final-position positive-noise stationary
projectors retain the different microscopic regime
already proved in Chapter NMG.
:::

(sec-nswt-origin)=
## 5. The zero-threshold full-chart normalization at the actual origin

:::{prf:theorem} Divergent original composite action when the chart includes the origin
:label: thm-nswt-zero-threshold-origin

In the same included consensus transition, use the original
zero threshold $\delta_c=0$, finite fixed $\kappa$, unit face
coefficients and a continuous nonnegative compactly supported
test $f$ that is strictly positive at the origin.
Use all original available composite CSR faces, with their
original zero masks. Then their actual normalized sum

$$
A_N=\frac1N\sum_i f(Y_i)
             \sum_{(j,k)\in\mathcal T_i^N}r_N^{-4}W_{ijk}^N
$$

tends to $+\infty$ in probability, and $EA_N\to+\infty$.
This concerns the original composite readout in this included
parameter regime. It is not a statement about the literal
ray interaction action or the target physical Hamiltonian.
:::

:::{prf:proof}
Both exact count and row limiting unit forces are
$-n$, $n=y/|y|$, away from zero. The proved native
color field is consequently
$c(y)=-n\odot e^{i\beta y}$, $\beta=\kappa/t$.
At a fixed $y=Rn$, remove its unitary diagonal phase.
The horizontal derivative in spatial direction $u$ is

$$
h_u=R^{-1}(I-nn^T)u+
                i\beta(I-nn^T)\operatorname{diag}(n)u .
$$

Thus at $\beta=0$ the complement curvature has
Frobenius square

$$
\|\mathcal F_0(Rn)(u,v)\|_F^2
                     =2R^{-4}[n\cdot(u\times v)]^2 .
\tag{NSWT.12}
$$

For finite $\beta$, its remaining terms in curvature
have norm at most
$8(|\beta|/R+\beta^2)|u||v|$ by these horizontal
derivatives; the line block has the same larger bound.
On any bounded face-vector event with
$|n\cdot(u\times v)|\ge a>0$, the leading term
therefore gives $\|\mathcal F_\kappa(Rn)(u,v)\|_F^2
\ge a^2R^{-4}$ for all sufficiently small
primitive $R>0$. Its permitted radius depends
only on $a,\beta$ and that vector bound.

The homogeneous Poisson Delaunay star has a positive
probability of such a bounded protected face.
For a direct construction rotate coordinates to put
$n=e_3$ and use small disjoint balls around
$e_1,e_2,e_3$. The tetrahedron with vertices
$0,e_1,e_2,e_3$ is nondegenerate. For sufficiently
small ball radius, each perturbed tetrahedron has
circumsphere contained in one fixed bounded ball
and has $|e_3\cdot(u\times v)|\ge1/2$ for its first
two vertices. Exactly one Poisson point in each
small ball and no other point in the containing
ball is a positive finite-volume probability event.
Its empty circumsphere supplies the actual Delaunay
face and corresponding CSR three-cycle. Additional
points outside that ball cannot remove its empty
circumsphere. Rotational invariance gives the same
event probability for every $n$.
For intensities in the compact positive interval
occupied by $p(Rn)$ near zero, this probability has
one primitive positive lower bound: use the smaller
intensity in the three occupied-ball factors and
the larger intensity in the empty-ball exponential.

The local expected curvature sum is therefore
bounded below by $C R^{-4}$, $C>0$, for all small
$R$ and directions $n$. The original Gaussian
density and $f$ are bounded below by positive
constants there. The limiting integral over
$\epsilon<|y|<R_1$ consequently grows at least
like $C'\int_\epsilon^{R_1}R^{-2}\,dR$.

For each fixed $\epsilon>0$ choose a continuous
nonnegative annular test $f_\epsilon\le f$,
supported away from zero and equal to $f$ on
$2\epsilon\le|y|\le R_1/2$.
Section 4 gives its actual finite sum converging
in $L^2$ to this finite annular integral.
All original summands are nonnegative and
$A_N\ge A_{N,\epsilon}$.
For any prescribed $L$, choose $\epsilon$
making the annular limit exceed $2L$.
Its convergence implies $P(A_N>L)\to1$.
The expectation comparison gives
$\liminf_N EA_N$ at least every annular limit,
hence infinite. No exchange of an unbounded
full-chart limit was assumed.
:::

(sec-nswt-isotropic-action)=
## 6. The native isotropic curvature energy and its exact translation source

:::{prf:theorem} Original unit-face curvature energy with its derived density coefficient
:label: thm-nswt-isotropic-curvature-action

Use the original unit composite-face coefficients and full-slot
matched B2 channel of Section 1 on a strict compact chart.
Let $\mathcal T_1$ be its unit-intensity homogeneous Poisson
rooted CSR cycles, with the original orientation/multiplicity,
and define the finite positive geometry constant

$$
c_{\rm Del}=\frac13E_{\Pi_1}
                       \sum_{(u,v)\in\mathcal T_1}|u\times v|^2 .
\tag{NSWT.13}
$$

The original uncut limit (NSWT.8) is exactly

$$
\mathcal A(f)=\frac{c_{\rm Del}}{24}
 \int f(x)p(x)^{-1/3}
             \sum_{1\le a<b\le3}
                     \|\mathcal F_{\kappa,ab}(x)\|_F^2\,dx .
\tag{NSWT.14}
$$

The coefficient $p^{-1/3}$ is derived from the original
unit weights and $r_N=N^{-1/3}$. It is not set to a
constant or absorbed into a newly chosen face coefficient.
This identifies a local curvature energy on the actual
native line/complement connection support. It is distinct
from both the executed scalar ray interaction action and
an unrestricted independent $SU(3)$ field ensemble.
:::

:::{prf:proof}
Write the area bivector as
$L_{ab}(u,v)=u_av_b-u_bv_a$, $a<b$.
Then $\mathcal F(u,v)=\sum_{a<b}\mathcal F_{ab}L_{ab}$.
The original Poisson Delaunay/CSR cycle law is equivariant
under every orthogonal spatial map. Its area vectors
$u\times v$ therefore have an invariant second-moment
matrix after summing the rooted cycles.
Coordinate sign reflections make off-diagonal entries
zero; coordinate permutations make all diagonal entries
equal. Their trace is the sum of squared areas.
Consequently
$E\sum L_{ab}L_{cd}=c_{\rm Del}\mathbf1_{\{(a,b)=(c,d)\}}$
at intensity one. No independence of different faces
is used. The star moments in Section 2 show finiteness;
the explicit positive tetrahedron event in Section 5
shows $c_{\rm Del}>0$.

At intensity $\zeta>0$, scaling all spatial vectors
by $\zeta^{-1/3}$ turns the original unit-intensity
Poisson star into its intensity-$\zeta$ star.
The area products therefore scale by $\zeta^{-4/3}$,
with cycle multiplicities unchanged. Expansion of
the Frobenius square gives
$E_{\Pi_\zeta}\sum\|\mathcal F(u,v)\|_F^2
=c_{\rm Del}\zeta^{-4/3}\sum_{a<b}\|\mathcal F_{ab}\|_F^2$.
Multiply by the original root density $p(x)$ and
the proved coefficient $1/24$ in (NSWT.8).
This gives exactly (NSWT.14).
:::

:::{prf:theorem} Exact original common-source action variation and its limit
:label: thm-nswt-common-source-variation

Use the same full-slot composite channel with unit
coefficients and fixed calibration, and let $f$ be
continuously differentiable with compact support in
a strict chart. Its support enlargement must pass
(NSWT.1). For any fixed real vector $a$, make the
original simultaneous Gaussian source shift
$G_i\mapsto G_i+\theta a$ in every original row.
Retain every resulting force, cap, final state and
terminal status in the complete executed record.
Let $A_N^f$ denote the original action observable
of Section 4. Then the exact finite-$N$ identities are

$$
A_N^f(G+\theta a)=A_N^{f(\,\cdot+\theta s_0a)}(G),
$$

$$
\left.\partial_\theta EA_N^f(G+\theta a)\right|_0
=s_0 EA_N^{a\cdot\nabla f}(G)
=E\left[A_N^f(G)\sum_i a\cdot G_i\right].
\tag{NSWT.15}
$$

The same native source first variation has the limit

$$
\lim_{N\to\infty}E\left[A_N^f\sum_i a\cdot G_i\right]
=
\frac{s_0c_{\rm Del}}{24}
 \int (a\cdot\nabla f)(x)p(x)^{-1/3}
                    \sum_{b<c}\|\mathcal F_{\kappa,bc}(x)\|_F^2\,dx .
\tag{NSWT.16}
$$

This is the original collective Gaussian source score
and its local curvature-energy response, with all
same-record faces retained. The source changes the
complete original transition, rather than rephasing
an independently supplied edge.
An additional final-alive or source deletion readout
retains its own boundary/mask variation; it is not
assigned this full-slot identity.
:::

:::{prf:proof}
In this original consensus transition the shifted
pre-B2 coordinates are
$z_i^\theta=z_i+q\theta a$ and
$Y_i^\theta=Y_i+s_0\theta a$.
Every original spatial difference, viscous velocity
difference and Gaussian weight is identical.
Thus both the count and the nonself row B2 force
are unchanged exactly. The recorded B2 colors change
by the common unitary diagonal matrix
$D_\theta=\operatorname{diag}(e^{i\kappa q\theta a})$,
so $P_i^\theta=D_\theta P_iD_\theta^\dagger$.
Their exact canonical endpoint transports transform
by that same conjugation. Every composite loop and
its unitary Wilson defect is therefore unchanged.
The availability norm/overlap mask is also unchanged.
The full-slot Delaunay geometry is merely translated;
its incidences, orientations and unit coefficients
are identical. The actually executed final potential
kick, cap and terminal status may change; those
retained outputs do not enter this particular
matched-B2 full-slot composite observable.
This proves the first equality in (NSWT.15).

Differentiate only the translated $f$.
For each fixed $N$ the deterministic polynomial
bound (NSWT.9) gives integrable domination.
Gaussian change of variables in all original sources
proves its equal score expectation; its linear score
is unbounded and retains its full Gaussian law.
Section 4 and (NSWT.14), applied to the continuous
compact test $a\cdot\nabla f$, now give (NSWT.16)
without taking a limit of a growing-$N$ score by
an unjustified uniform-integrability claim.

Equivalently the shifted limiting connection is
$D_\theta P_\kappa(x-s_0\theta a)D_\theta^\dagger$,
and its curvature is conjugated and translated.
The native density becomes $p(x-s_0\theta a)$.
Changing variables in its original curvature
energy leaves precisely the translated $f$
whose derivative was just computed. Thus the
finite source calculation and the local energy
variation refer to the same native supported law.
:::
