# Same-record marked spatial action fluctuations and native time covariance

(sec-nmaf-register)=
## 1. Original stationary history and the actual conditional center

:::{prf:definition} Complete marked-action fluctuation register
:label: def-nmaf-register

Retain every parameter and restriction of
{prf:ref}`def-nhs-register`: the existing all-alive, zero-selection,
zero-viscosity, uncapped harmonic gas, its original positive OU source,
zero terminal spatial noise, and the declared resonant recording stride.
The source is the derived full stationary law. Use the same matched B1
recorded-potential projector, strict available compact chart, fixed
overlap tests, canonical composite Wilson loops and exact book
open Euclidean Delaunay/CSR geometry. Rust's rank-tolerant geometry,
the default unavailable viscous color at zero viscosity, and archive
interaction-ray faces retain their different original tags.

Write $r_N=N^{-1/3}$ and
$\eta_N=\kappa_N/r_N\to\eta\in[0,\infty)$ for the explicitly configured
positive phase family. No finite-population phase is set to zero.
For any real continuous compactly supported chart test $f$, let
$A_{N,k}(f)$ be the ORIGINAL action (NHS.3) on recorded time $k$.
Let $\mathcal X_N$ be the sigma-field of the entire recorded position
history and define the exact conditional center

$$
Z_{N,k}(f)=\sqrt N\{A_{N,k}(f)
                  -E[A_{N,k}(f)\mid\mathcal X_N]\}.
\tag{NMAF.1}
$$

This centers the executed Wilson action, including its original
availability tests. It does not replace the full spatial action
fluctuation by its velocity innovation or discard the position center.

Put $r=c^{m/2}\in(0,1)$ from (NHR.5). At the recorded clock the ORIGINAL
position and velocity histories are independent, and across row addresses
their laws are independent copies of

$$
x_{k+1}=rx_k+\sqrt{X(1-r^2)}\,g_k,\qquad
v_{k+1}=rv_k+\sqrt{V(1-r^2)}\,e_k,
\tag{NMAF.2}
$$

where all $g_k,e_k$ are their original independent standard
three-Gaussians. This is a consequence of the executed stride:
$A^m=rI$ and the stationary covariance in (NHS.1) is
$\operatorname{diag}(XI_3,VI_3)$. The full stride innovation covariance
is therefore $(1-r^2)\operatorname{diag}(XI_3,VI_3)$.
It is not a new time evolution or an assumed independent position law.
The physical interval remains $mt_*h$.
:::

(sec-nmaf-local-scores)=
## 2. Exact marked Poisson scores and their finite primitive integrals

:::{prf:definition} Original leading face and one-address response
:label: def-nmaf-scores

Fix $x=Rn$ in the strict chart, $Q=I-nn^T$, and the rooted
Poisson Delaunay graph of intensity $p(x)=\varphi_{\sqrt X}(x)$.
Every site, including its root, has its independent original
$N(0,VI_3)$ velocity mark. At each vertex $z$ of this common graph put

$$
\begin{gathered}
h_{z,u}=Qu/R+i\eta Q\operatorname{diag}(n)(v_{z+u}-v_z),\\
\xi_z=\frac1{24}\sum_{(u,w)\in\mathcal T_z}
          \|[B_{h_{z,u}},B_{h_{z,w}}]\|_F^2,
\qquad B_h=hn^\dagger+nh^\dagger .
\end{gathered}
\tag{NMAF.3}
$$

All rooted cycles and repeated vertices use the SAME graph and marks.
The physical chart coefficients $n,R$ are held at $x$ in this local
rescaling. For a two-root Poisson configuration with forced roots
$0,z$, use $\xi_0,\xi_z$ on that common graph, including both forced
root marks. Write $\operatorname{Cov}_v$ for covariance over marks
conditional on the whole local geometry.

A root velocity can enter finitely many of these scores. Define its
complete one-address response by

$$
E_v\!\left[\sum_{j:\,0\ {\rm enters}\ \xi_j}\xi_j
             \,\middle|\,v_0=w,\Pi\right]
-E_v\sum_{j:\,0\ {\rm enters}\ \xi_j}\xi_j
 =w^TB_\eta(x,\Pi)w-V\operatorname{tr}B_\eta(x,\Pi).
\tag{NMAF.4}
$$

The sum includes scores rooted at neighbors, not just the root's own
score. Define $\overline B_\eta(x)=E_\Pi B_\eta(x,\Pi)$.
The matrix exists uniquely, is symmetric and positive semidefinite.
It has finite moments of every order, with bounds from the original
Gaussian star series, $X,V,R_\pm$, the compact chart and $\eta$.
At $\eta=0$ it is zero.

Define the native same-time variance

$$
\begin{split}
\mathcal V_\eta(f)=\int f(x)^2\bigg[&
p(x)E_{\Pi_{p(x)}}\operatorname{Var}_v\xi_0\\
&+p(x)^2\int_{\mathbb R^3}
 E_{\Pi_{p(x)}^{0,z}}\operatorname{Cov}_v(\xi_0,\xi_z)\,dz
                          \bigg]dx .
\end{split}
\tag{NMAF.5}
$$

The pair covariance keeps every overlapping face and mark. This
absolutely convergent integral is a derived conditional variance
limit; its pair part need not be nonnegative term by term.
:::

:::{prf:lemma} Primitive localization and exact one-address polynomial
:label: lem-nmaf-localization

Every integral in (NMAF.3)--(NMAF.5) is finite.
The original finite-binomial leading-score versions have uniform
moments of all orders on the chart. Their covariance and one-address
responses are determined by a finite protected four-ring neighborhood,
with failure probability bounded by
$C(1+L^3)e^{-cL^3}$ at rescaled radius $L$.
For distinct forced roots distance $|z|$, the absolute expected
mark covariance is at most $C'e^{-c'|z|^3}$ after enlarging the
constants on $|z|\le1$.
All constants are primitive star/Gaussian bounds; no moment or
stabilization hypothesis is added.
:::

:::{prf:proof}
Use the shielding balls of {prf:ref}`lem-nswt-star-moments`
and the additional neighbor shields of
{prf:ref}`lem-nga-two-root`. Their proof for independent Gaussian
positions applies with the derived $X$ and density $p$. Its finite
binomial count bounds and Poisson finite-window series give
$C(1+L^3)e^{-cL^3}$ for the determining neighborhood. On a shielded
annulus the number of vertices is dominated by its original binomial
or Poisson count, and the scores are polynomials of degree four in
the Gaussian marks, polynomial in the radius and count. The
convergent dyadic series (NSWT.5), applied to larger integer powers,
bounds all their moments. The same argument bounds the sum of
scores containing a fixed velocity address: all their roots are
neighbors of that address. For the covariance sum, its other roots
are within two graph edges, and their cycle tests require at most
four rings. Protect the original root and every candidate site in
the enclosing fixed multiple of the shielding ball. Conditioning
on one candidate leaves the other original rows independent; the
finite count union bound gives the same polynomial prefactor and
exponential empty-ball bound. This supplies every additional
neighbor shield without an independence assumption on graph rings. A forced second root adds one site
to these count bounds and preserves them.

For separated roots take determining radii below $|z|/8$.
If both neighborhoods are protected, they use disjoint velocity
addresses; hence their CONDITIONAL mark covariance is zero.
On the complement Cauchy--Schwarz, the just proved fourth moments
and the exponential shield probability give the stated bound.
Its integral in three dimensions is finite. This proves absolute
integrability in (NMAF.5), including the close-root part.

For the polynomial response, write the face commutator as
$C_0+i\eta C_1-\eta^2C_2$, with real matrices, where $C_0$
is geometric, $C_1$ is linear in the three velocities and $C_2$
is bilinear in them. The real and imaginary parts are orthogonal
in Frobenius norm, so its square is

$$
\|C_0-\eta^2C_2\|_F^2+\eta^2\|C_1\|_F^2.
\tag{NMAF.6}
$$

For a fixed vertex velocity $w$, $C_2$ is linear in $w$:
the two occurrences of the common root phase cancel their
self-commutator. Its remaining terms have either one other centered
Gaussian mark or two distinct other marks. Their conditional mean
is zero. The cross term with $C_0$ therefore vanishes after integrating
the other marks. In $E\|C_2\|^2$, the cross between its linear-in-$w$
part and its remaining two-mark part vanishes by Gaussian oddness.
Likewise $E\|C_1\|^2$ is a constant plus a nonnegative quadratic
form in $w$. Thus the conditional face response has exactly a
positive semidefinite quadratic coefficient; no linear or quartic
single-address term remains. Sum over all faces containing the
address to obtain (NMAF.4). The proved moment bounds justify that
finite sum and its average.
:::

(sec-nmaf-variance)=
## 3. Complete conditional variance and strict nondegeneracy

:::{prf:theorem} Derived same-record mark variance with a primitive positive lower bound
:label: thm-nmaf-variance

At every $\eta\in[0,\infty)$,

$$
N\operatorname{Var}(A_{N,0}(f)\mid\mathcal X_N)
          \longrightarrow\mathcal V_\eta(f)
\quad\hbox{in probability and }L^1.
\tag{NMAF.7}
$$

If $f\ge0$ is nonzero and $\eta>0$, the limit is strictly positive.
More precisely, with $S=Q\operatorname{diag}(n_a^2)Q$ and
$G(n)=(\operatorname{tr}S)^2-\operatorname{tr}S^2
     =6n_1^2n_2^2n_3^2$,

$$
\mathcal V_\eta(f)\ge
 \frac{\eta^8V^4}{54}
 \int f(x)^2p(x)G(n_x)^2
        E_{\Pi_{p(x)}}|\mathcal T_0|^2\,dx>0 .
\tag{NMAF.8}
$$

This is nondegeneracy of the ORIGINAL spatial composite action
fluctuation at the critical mark scale, rather than a presumed
nonzero continuum field or independent-face variance.
For $\eta=0$ the conditional mark variance limit is zero.
:::

:::{prf:proof}
First use the leading finite-binomial score (NMAF.3) with the actual
$\eta_N$ and root $n_i,R_i$. Denote it by $\xi_{N,i}$ and set
$F_N=\sum_i f(x_i)\xi_{N,i}$.
Conditional on positions, its mark covariance is exactly

$$
N^{-1}\operatorname{Var}_vF_N
 =N^{-1}\sum_i\left[f_i^2\operatorname{Var}_v\xi_{N,i}
              +\sum_{j\ne i}f_if_j
                         \operatorname{Cov}_v(\xi_{N,i},\xi_{N,j})\right].
\tag{NMAF.9}
$$

The quantity in brackets is an original protected local graph
function with uniform moments by the preceding lemma.
At a fixed determining radius its single-root binomial density
converges to the forced-root Poisson density. Two separated roots
have independent limiting local configurations by the finite
Bernoulli/Poisson window argument already proved in Chapter NGA.
Consequently the empirical average of the brackets converges in
$L^2$ on every fixed radius/count core. The moment and shield series
remove those cores in $L^1$. This proves a deterministic limit,
without assuming convergence of a random bracket.

In the limiting root expression, the diagonal term is the first
term of (NMAF.5). For the sum over other local roots, expand the
finite Poisson count series, distinguish one of its sites and
integrate its position. This changes its count coefficient
$e^{-\alpha}\alpha^\ell/\ell!$ into the same series with one forced
root and a factor $p(x)\,dz$. Thus the off-diagonal term is exactly
the second term of (NMAF.5). Absolute covariance integrability
justifies passage to the whole determining neighborhood.
This is a direct counting identity on the common graph.

For positivity let $q_i=|v_i|^2-3V$, with
$E q_i^2=6V^2$. The independent $q_i$ span mutually orthogonal
single-address Gaussian-chaos subspaces. Their projections give

$$
N^{-1}\operatorname{Var}_vF_N
 \ge\frac{6V^2}{N}\sum_i b_{N,i}^2,\qquad
b_{N,i}=\frac{E_v[F_Nq_i]}{6V^2}.
\tag{NMAF.10}
$$

Every face contributes nonnegatively to $b_{N,i}$ when $f\ge0$
by (NMAF.6). The pure quartic phase energy of one rooted face
has mean $\eta_N^4V^2G(n_i)/4$ by the actual shared-root
Gaussian calculation (NHS.10). Its polynomial is quadratic
in each of its three marks and its norm is symmetric under
permutation of those three labels. Gaussian scale differentiation
therefore gives equal radial projections at the three labels.
Indeed its mean is proportional to $V^2$, so

$$
\sum_{\ell\in\{i,j,k\}}E[\xi_{\rm phase}q_\ell]
        =2V^2\partial_V E\xi_{\rm phase}
        =4V E\xi_{\rm phase}.
$$

Each label consequently has coefficient
$\eta_N^4VG(n_i)/18$ in (NMAF.10). The geometric/phase
quadratic coefficient is nonnegative. Keeping just the faces
rooted at $i$ gives

$$
b_{N,i}\ge
 f_i|\mathcal T_i^N|\eta_N^4VG(n_i)/18 .
$$

The original star law and uniform moments now give (NMAF.8).
A nonzero continuous nonnegative $f$ is positive on an open set.
The coordinate planes have zero Lebesgue measure, $p>0$, and
a shielded simplex configuration has a rooted three-cycle on
an open positive-probability Poisson event. Hence the lower
integral is strictly positive. At $\eta=0$ every leading score
is independent of marks.

Finally the exact Wilson action and its leading polynomial
have the same CENTERED conditional fluctuation. This quantitative
statement is proved in the next lemma and transfers (NMAF.9)
to (NMAF.7). Its $L^2$ error, and the uniform variance moments
from the same shield series, also give the asserted $L^1$
variance convergence.
:::

(sec-nmaf-gaussian)=
## 4. Conditional Gaussian law for the original uncut Wilson action

:::{prf:lemma} Vanishing centered Wilson remainder and connected mark cumulants
:label: lem-nmaf-cumulants

For any fixed finite set of recorded times and chart tests, replace
the original Wilson action by $N^{-1}F_N$ at each time.
The difference of their $\sqrt N$ conditional-centered vectors
tends to zero in $L^2$.

For every fixed integer $k\ge3$, the conditional cumulant of
order $k$ of any fixed real linear combination of the leading
centered vectors tends to zero in probability. The same conclusion
holds for the original uncut conditional characteristic functions.
Every Gaussian draw and literal face remains in the original quantity.
:::

:::{prf:proof}
Here are explicit proof cores, followed by their removal. Let
$h_X=(2\pi X)^{-3/2}$ and let $\beta_X>0$ be its minimum
on the one-unit enlargement of the compact chart neighborhood.
The star shields give $Me^{-c_XL^3}$ with
$c_X=\beta_Xv_3/(2\cdot16^3)$, $M\le17^3$.
For any prescribed power $a>10$, choose
$L_N=(A\log N)^{1/3}$ with $A>(a+5)/c_X$.
For sufficiently large $N$ its physical radius $r_NL_N$
is below the fixed chart buffer. A union bound over every original
root in that buffer protects all needed stars and neighbor stars,
except with probability $O(N^{-a})$.

The number of sites in any ball of radius $16r_NL_N$ about
an original site has binomial mean at most
$h_Xv_3\,16^3A\log N$, plus the conditioned root.
Its original factorial-moment/Chernoff bound gives a uniform
$D\log N$ upper bound outside $O(N^{-a})$ after taking

$$
D>\max\{2e h_Xv_3\,16^3A,\ (a+5)/\log2\}+2.
$$

There are only $N$ addressed centers and a fixed number of times.
Thus this is an original spatial event, not a deterministic
degree hypothesis. On it each score uses $O(\log N)$ mark
addresses and intersects at most $O(\log N)$ other scores.
An intersection must have its root within the above enlarged
ball. The score dependency graph conditional on positions
therefore has maximal degree $O(\log N)$, also after taking
the union of a fixed number of recorded graphs.

For proof comparison only replace each velocity-history block
by a bounded version on $|v_{i,k}|\le H\sqrt{\log N}$.
Choose $H^2>8V(a+5)$.
A Gaussian exponential moment, followed by the finite time/root
union bound, makes the changed-source event $O(N^{-a})$.
Blocks at different row addresses remain independent.
All original root scores and the leading polynomials have
uniform fourth and higher moments by (NHS.7) and the shield
series. Jensen and Cauchy--Schwarz show that this comparison
changes a $\sqrt N$ centered action vector by $o(1)$ in $L^2$:
for example $N E[(A_N-\widetilde A_N)^2]
\le C N P(\text{changed event})^{1/2}$.
The same estimate handles the exceptional spatial event.
The comparison is removed; it is not a modified noise law.

On these proof cores every endpoint is available and its
canonical transport passes the fixed overlap test for large $N$.
The exact polar Taylor formula (NHS.4) has a remainder bounded,
per ROOT SUM, by
$C r_N(\log N)^{10}$ after multiplication by $r_N^{-4}$.
Here its argument bounds use $r_NL_N$,
$\kappa_N H\sqrt{\log N}$, the fixed $R_-^{-1}$, and the
actual bounded $\eta_N$. Derivatives through order three
of the normalized radial line and polar transport are bounded
on that fixed overlap ball; multiplying the three transports
gives the stated safe polynomial exponent.
No rate of $\eta_N\to\eta$ is needed because the polynomial
uses $\eta_N$ itself.

A centered sum of variables with dependency degree $\Delta$
and individual magnitude at most $b$ has variance at most
$4N(\Delta+1)b^2$, by its exact covariance sum.
For the normalized centered remainder this is at most
$C r_N^2(\log N)^{21}\to0$.
This proves the claimed remainder after removal of both cores.

For the bounded leading scores, a joint cumulant is zero if
its addressed score vertices split into two groups with no
dependency edge: their underlying original mark blocks are
independent, and the logarithm of their generating function
splits. The finite partition formula bounds any order-$k$
joint cumulant by $C_kb_N^k$, with
$b_N\le C(\log N)^{10}$.
The number of connected ordered $k$-tuples in the dependency
graph is at most $N C'_k(\Delta+1)^{k-1}$, allowing repeated
vertices. Therefore the normalized sum has cumulant bound

$$
C''_kN^{1-k/2}(\log N)^{11k}\longrightarrow0,
\qquad k\ge3.                                      \tag{NMAF.11}
$$

This elementary connected-tuple calculation uses the whole
overlapping graph, not independent face marks.
The conditional second cumulants have their derived limits
and bounded moments. The partition formula for moments
therefore gives Gaussian moments of every fixed order along
any spatial subsequence on which the variance and good events
converge. For a direct characteristic-function conclusion, expand through
order $2b-1$ and bound its Taylor remainder by
$|u|^{2b}E|Z|^{2b}/(2b)!$. First send $N$ to infinity at fixed
$b$; the Gaussian even-moment bound then sends this remainder
to zero as $b$ tends to infinity. Conditional tightness follows
from the second moment. Every further subsequential limit is
consequently the same Gaussian. This proves convergence of conditional
characteristic functions in probability; boundedness upgrades
it to $L^1$. The removed $L^2$ comparisons transfer that
conclusion to the original action.
:::

(sec-nmaf-time-covariance)=
## 5. Full native recorded-time covariance with repeated row addresses

:::{prf:theorem} Exact finite-history Gaussian mark law
:label: thm-nmaf-history

For every fixed finite collection of recorded times and chart tests,
the vector (NMAF.1) converges to a centered Gaussian vector, stably
relative to the complete original position history $\mathcal X_N$.
For one test $f$ the limit is stationary with covariance

$$
\begin{gathered}
C_\eta(0;f)=\mathcal V_\eta(f),\\
C_\eta(k;f)=
2V^2r^{2|k|}
 \int f(x)f(y)p_k(x,y)
  \operatorname{tr}\{\overline B_\eta(x)\overline B_\eta(y)\}\,dx\,dy,
\qquad k\ne0 ,
\end{gathered}
\tag{NMAF.12}
$$

where $p_k$ is the ORIGINAL joint Gaussian position density with
marginal covariance $XI_3$ and cross covariance $Xr^{|k|}I_3$.
For distinct tests $f,g$ replace $f(x)f(y)$ by $f(x)g(y)$
and use their same-time bilinear covariance. No temporal
independence of the graph or repeated velocity address is assumed.
:::

:::{prf:proof}
The conditional Gaussian calculation is the connected-tuple
argument above with one independent velocity-HISTORY block per
original row. The remaining task is its cross-time bracket.

Expand the centered leading action in the orthogonal product
Hermite basis of its row velocities. Its total degree is at most
four. The single-address term is precisely

$$
v_i^TB_{N,i}v_i-V\operatorname{tr}B_{N,i},
$$

where $B_{N,i}$ includes ALL scores whose face uses row $i$,
with their actual root weights $f(x_j)$.
At recorded lag $k$, (NMAF.2) gives the exact quadratic covariance
$2V^2r^{2|k|}\operatorname{tr}(B_{N,i,0}B_{N,i,k})$.
Products with different Hermite address supports are orthogonal
under this original joint Gaussian law.

A support with at least two distinct row addresses can contribute
at both times only if some same pair of addresses is geometrically
close at both times. On the proof shields this means distance at
most $C r_NL_N$ at each time. For $k\ne0$ the ORIGINAL joint
position density is nonsingular and bounded. The pair of position
differences has a bounded joint six-dimensional Gaussian density.
Thus the probability for a specified pair is at most
$C_k L_N^6/N^2$. The expected number of such pairs among all
$N^2$ pairs is $O(L_N^6)$, not order $N$. All involved coefficients,
local counts and Gaussian covariances on the proof cores have
polynomial logarithmic bounds. After division by $N$, their
entire multiple-address cross-covariance tends to zero in $L^1$.
The removed cores have the vanishing moment errors already proved.
This retains every possible shared face/mark, including differently
rooted faces, rather than asserting independent time layers.

For the single-address coefficients, condition a uniform row on
its two original positions $x,y$. Each other row has probability
$O(N^{-1})$ of entering either determining window and
$O(N^{-2})$ of entering both. The latter estimate again follows
from its nonsingular original two-time Gaussian density.
A Bernoulli/Poisson point coupling on these two time-tagged
windows consequently gives independent limiting Poisson
neighborhoods conditional on $x,y$. This is a derived local
limit; the original position histories themselves remain
correlated through $p_k$. In each neighborhood the actual root
weights converge to $f(x)$ or $f(y)$.
Their mean one-address coefficients are therefore
$f(x)\overline B_\eta(x)$ and
$f(y)\overline B_\eta(y)$.

The same two-root counting argument, now with each original row
carrying its two-time position mark, proves concentration of
the average coefficient product. At fixed determining windows,
different tagged roots have independent limiting neighborhoods;
coincidences have vanishing probability. The uniform shield/count
moments remove the windows. Hence

$$
N^{-1}\sum_i\operatorname{tr}
             (B_{N,i,0}B_{N,i,k})
\ \longrightarrow\
\int f(x)f(y)p_k(x,y)
       \operatorname{tr}(\overline B_\eta(x)\overline B_\eta(y))\,dx\,dy.
$$

This proves (NMAF.12), including the actual repeated-row covariance.
The same-time term was (NMAF.7). The mixed cumulants already vanish,
so these brackets determine the joint Gaussian law.
Conditional characteristic convergence in $L^1$ permits
multiplication by any bounded $\mathcal X_N$-measurable variable.
This is the asserted full position-history stable convergence.
:::

(sec-nmaf-green-kubo)=
## 6. Complete covariance spectrum and original long-time scaling

:::{prf:theorem} Native covariance spectrum and its strictly positive complete Green–Kubo limit
:label: thm-nmaf-green-kubo

Let $M(x)=f(x)\overline B_\eta(x)$ and decompose its matrix
entries in the original position-Gaussian Hermite spaces:
$M=\sum_{\ell\ge0}M_\ell$ in
$L^2(p;\mathbb R^{3\times3})$. Set

$$
a_\ell=2V^2\|M_\ell\|_{L^2(p;F)}^2,\qquad
b=\mathcal V_\eta(f)-\sum_{\ell\ge0}a_\ell\ge0 .
$$

The COMPLETE derived covariance is

$$
C_\eta(k;f)=
 b\,\mathbf1_{\{k=0\}}+
 \sum_{\ell\ge0}a_\ell r^{(\ell+2)|k|}.
\tag{NMAF.13}
$$

Consequently its complete Green–Kubo coefficient is

$$
\Sigma_\eta(f)=
 b+\sum_{\ell\ge0}
       {1+r^{\ell+2}\over1-r^{\ell+2}}a_\ell ,
\qquad
\mathcal V_\eta(f)\le\Sigma_\eta(f)
 \le {1+r^2\over1-r^2}\mathcal V_\eta(f).
\tag{NMAF.14}
$$

For nonzero $f\ge0$ and $\eta>0$, both $\mathcal V_\eta(f)$
and $\Sigma_\eta(f)$ are positive, and
$a_0=2V^2\|\int M(x)p(x)\,dx\|_F^2>0$.
The exact slowest nonzero covariance decay is $r^{2|k|}$,
or rate $\gamma/t_*$ per recorded physical time.
This is an identified covariance rate; it is not substituted
for the Hamiltonian gap of a target local Yang–Mills sector.
:::

:::{prf:proof}
The original position Mehler transfer acts as $r^\ell$ on
its degree-$\ell$ Hermite space, by (NHR.6). Apply that
orthogonal expansion to (NMAF.12). Every matrix entry gives
its squared coefficient and the additional actual velocity
factor $r^{2|k|}$. This proves (NMAF.13) at $k\ne0$.

At the same time the complete single-address Gaussian projection
of the finite leading action has variance
$2V^2N^{-1}\sum_i\|B_{N,i}\|_F^2$.
Orthogonality makes this at most its full conditional variance.
The rooted law and Jensen give in the limit

$$
\mathcal V_\eta(f)\ge
2V^2\int f(x)^2p(x)E_\Pi\|B_\eta(x,\Pi)\|_F^2\,dx
\ge2V^2\int\|M(x)\|_F^2p(x)\,dx.
$$

The last quantity is $\sum_\ell a_\ell$; hence $b\ge0$
and (NMAF.13) also holds at zero. Its zero-lag excess is
derived from the original graph and multiple-address marks,
not an independently added white-noise field.

The nonnegative coefficient series is summable. Sum its
geometric time series to obtain (NMAF.14), including all lags.
The bounds follow from $0<r^{\ell+2}\le r^2$.
For positivity, the own-root quartic radial projection used
in (NMAF.8) shows that $\operatorname{tr}B_\eta(x,\Pi)$
has a positive mean at almost every chart point.
It is positive semidefinite. Thus a nonzero nonnegative $f$
has a nonzero positive semidefinite mean matrix, so $a_0>0$.
The remaining terms in (NMAF.13) decay at least as $r^{3|k|}$;
dominated convergence after division by $r^{2|k|}$ gives
the stated exact leading decay. Finally
$-2\log r/(mt_*h)=\gamma/t_*$.
:::

:::{prf:lemma} Uniform original centered-action moments and temporal Poisson budgets
:label: lem-nmaf-poisson

For every fixed even integer $p\ge2$, the ORIGINAL state observable

$$
F_N(S)=\sqrt N\{A_N(f;S)-E_v[A_N(f;S)\mid x]\}
$$

has $\sup_N\|F_N\|_{L^p(\mu_N)}<\infty$ at bounded
$\eta_N$. It has no zero-velocity Hermite component and no
odd total-velocity component. Thus

$$
\|\mathcal P_m^kF_N\|_2\le r^{2k}\|F_N\|_2.
\tag{NMAF.15}
$$

Its original Poisson series $u_N=\sum_{k\ge0}\mathcal P_m^kF_N$
has uniformly bounded $L^4$ norm, as do its full-state martingale
increments and the $L^2$ norm of their exact conditional bracket.
These constants depend on every consumed parameter, the chart
and the phase-ratio bound, and are independent of $N$.
:::

:::{prf:proof}
For the leading action polynomial, conditional on positions
the centered sum has Gaussian degree at most four. The Gaussian
Wick expansion gives a dimension-independent $p$th-moment bound
$E_v|F|^p\le C_{p,4}(E_vF^2)^{p/2}$.
To see that its constant is finite without inserting a moment
hypothesis, expand $F$ into its degree-zero through degree-four
symmetric normal-ordered Gaussian tensors. In a $p$-fold product
there are at most $(4p)!$ cross-pairing patterns.
Each tensor contraction is bounded by the product of tensor
norms, by repeated Cauchy--Schwarz; summing the five possible
degrees costs at most $5^p$. One safe constant is
$C_{p,4}=5^p(4p)!$ after absorbing the finite Hermite
normalization factors. The squared tensor norms sum to
$E_vF^2$. The local bracket in (NMAF.9) has uniform moments
of all orders by the shield series and Jensen, so this proves
the original leading-sum bound.

The centered Taylor remainder on proof cores is a sum of bounded
variables with dependency degree $O(\log N)$ and bound
$Cr_N(\log N)^{10}$. The same connected-tuple cumulant estimate
through order $p$ and the moment partition formula bound its
normalized $p$th moment by $Cr_N^p(\log N)^{C_p}\to0$.
Choose the shield/source exception power larger than $2p+10$.
Its removed contribution is bounded by
$C N^{p/2}P(\text{exception})^{1/2}$ using the ORIGINAL
uniform $2p$ root moments. This proves the asserted bound
for the exact Wilson action as well.

Complex conjugation of every original projector is induced
by $v\mapsto-v$. The canonical transports conjugate with it,
and the real Wilson trace and all position/overlap masks are
unchanged. The exact centered action is therefore even in
the full velocity array. Its original conditional velocity
mean is zero by definition. The product Gaussian Hermite
expansion consequently has velocity degree at least two.
The independent position/velocity Mehler transfers preserve
that subspace and multiply its terms by at most $r^{2k}$.
This proves (NMAF.15), for the exact original observation.

Use its uniform $L^8$ bound and Markov $L^8$ contraction.
Interpolation gives

$$
\|\mathcal P_m^kF_N\|_4
 \le\|\mathcal P_m^kF_N\|_2^{1/3}
      \|\mathcal P_m^kF_N\|_8^{2/3}
 \le C r^{2k/3}.
$$

The series is summable, uniformly in population.
The actual martingale difference
$D_{N,k}=u_N(S_k)-\mathcal P_m u_N(S_{k-1})$
has $L^4$ norm at most $2\|u_N\|_4$.
Its exact bracket
$\mathcal B_N=\mathcal P_mu_N^2-(\mathcal P_mu_N)^2$
has $L^2$ norm at most $2\|u_N\|_4^2$ by conditional Jensen.
These are budgets for the original FULL stationary transfer,
not a newly assigned graph Markov kernel.
:::

(sec-nmaf-joint-time)=
## 7. Joint population and time Brownian limit of the original graph action innovation

:::{prf:theorem} Every diverging population/time schedule and commuting ordered limits
:label: thm-nmaf-joint-time

Let $N\to\infty$ and let the ORIGINAL integer recording horizon
$n_N\to\infty$ along any schedule. Under the full stationary law,

$$
\mathcal Z_{N,n_N}(t)
 ={1\over\sqrt{n_N}}\sum_{k=1}^{\lfloor n_Nt\rfloor}
                         F_N(S_k)
 \Longrightarrow \sqrt{\Sigma_\eta(f)}\,B_t
\quad\hbox{in }D([0,1]).
\tag{NMAF.16}
$$

The finite-test vector version has its polarized complete covariance.
The ordered population/time limits agree with this joint limit.
For nonzero $f\ge0,\eta>0$ it is nondegenerate by (NMAF.8).
The centering is the actual conditional position center in (NMAF.1).
No stationary Gaussian law for its separate position-center
fluctuation is asserted.
:::

:::{prf:proof}
At each original $N$, the positive self-adjoint full-state
Mehler transfer and (NMAF.15) give the complete covariance sum

$$
\Sigma_N=E F_N^2+2\sum_{k\ge1}E[F_N\mathcal P_m^kF_N].
$$

Its absolute tail is uniformly bounded by
$2C\sum_{k>K}r^{2k}$.
The finite-lag bracket limits (NMAF.7), (NMAF.12) therefore give
$\Sigma_N\to\Sigma_\eta(f)$, without exchanging an uncontrolled
infinite time sum. The exact martingale variance is $\Sigma_N$
by the Poisson identity. Indeed the Poisson equation
$F_N=u_N-\mathcal P_mu_N$ gives the original additive decomposition

$$
\sum_{k=1}^nF_N(S_k)
 =\sum_{k=1}^nD_{N,k}
   +\mathcal P_m u_N(S_0)-\mathcal P_m u_N(S_n).
\tag{NMAF.17}
$$

The whole-state spectral contraction on centered $L^2$ functions
is at most $r^k$, independent of dimension. Its application to
the derived exact bracket and the lemma's uniform $L^2$ budget
bounds the variance of every $n$-term bracket average by
$C/[n(1-r)]$. Thus those original bracket averages converge
to $\Sigma_N$ in $L^2$, uniformly in $N$.
The uniform fourth-moment martingale budget gives Lindeberg
error at most $C/(\epsilon^2n)$.

For completeness the native martingale proof is the same
conditional Taylor calculation as
{prf:ref}`thm-njt-characteristic`: expand each martingale
characteristic increment through its conditional second moment,
bound its large increments by the just proved Lindeberg budget,
and replace block bracket averages by their $L^2$ limit.
It gives all finite-dimensional Brownian characteristic functions
along every $N,n_N$ schedule. For tightness use the elementary
fourth-moment martingale inequality
$E|\sum_{k=l+1}^{l+b}D_{N,k}|^4
 \le C E(\sum_{k=l+1}^{l+b}D_{N,k}^2)^2
 \le C'b^2$.
The first inequality follows by expanding the squared martingale
and applying the maximal $L^2$ inequality and Cauchy--Schwarz;
its constant is absolute. The second uses the uniform fourth
moments. Linear interpolation has the corresponding quadratic
increment bound, and the maximal jump tends to zero by its
fourth-moment union bound.
The original Poisson endpoints vanish uniformly in the
interpolation: the union bound over $n+1$ endpoints is at most
$C/(\epsilon^4n)$, by the uniform $L^4$ norm of $\mathcal P_mu_N$.
This proves (NMAF.16).

At fixed $N$ the same proof gives the stationary Brownian law
with coefficient $\Sigma_N$, and its population limit is
(NMAF.14). In the opposite order, (NMAF.12)--(NMAF.13) give
the stationary Gaussian population process. Its summable
covariance yields the same long-time Brownian coefficient:
the covariance of two partial sums converges to
$\Sigma_\eta\min(s,t)$ by the summable series, and Gaussian
fourth moments give the tightness bound.
Both ordered limits and the proved joint limit coincide.
:::

(sec-nmaf-scope)=
## 8. The remaining full spatial center and parameter regimes

The original critical marked Wilson action now has its positive
same-record velocity fluctuation scale, identified finite-history
Gaussian law, complete native-time covariance and every-schedule
population/time Brownian limit. Every same-face, shared-root,
neighbor and repeated-time velocity correlation has been retained.

Its full spatial fluctuation still decomposes exactly as

$$
\sqrt N(A_N-EA_N)
 =F_N+\sqrt N\{E_v[A_N\mid x]-EA_N\}.
$$

The second term is the original position/retessellation center.
The preceding proof does not assign it a Gaussian law. A complete
physical graph-field limit still requires that term and the specified
spacetime correspondence.

At $\eta=0$ this mark-centered scale is zero; the nonzero radial
spatial action in Chapter NHS remains. At $\eta>0$ the primitive
positive result holds for every nonzero nonnegative continuous chart
test. Signed tests retain (NMAF.5), rather than a positivity assumption.
Unbounded readout charts, a fixed nonvanishing phase, other phase/graph
ratios, historical or current random calibration, actual metric
feedback, positive viscosity/selection, terminal noise, numerical
geometry and finite arithmetic retain their original distinct
parameter laws. The theorem proves the displayed existing branch
and its declared positive phase family. It changes no force,
algorithmic source, threshold or available finite-$N$ calibration.

