# Full stationary spatial action fluctuations with native retessellation

(sec-nspf-register)=
## 1. Original complete spatial action and its position center

:::{prf:definition} Complete position-center fluctuation register
:label: def-nspf-register

Retain EVERY original parameter and restriction of
{prf:ref}`def-nmaf-register` and {prf:ref}`def-nhs-register`.
In particular the existing all-alive, constant-fitness, uncapped
harmonic gas uses its original OU innovations, zero terminal spatial
noise and resonant recording stride. Its matched recorded-potential
B1 colors and exact book open Euclidean Delaunay/CSR composite loops
are the declared instruments. No Rust rank-tolerance branch, feedback
metric, archive ray loop or default zero-viscosity viscous color
is assigned this geometric law. The full original stationary rows
and recorded history are those already derived in (NMAF.2).

For the same real continuous compact chart test $f$ and positive
configured phase family $\eta_N=\kappa_N/r_N\to\eta<\infty$, define

$$
J_N(x)=E_v A_N(f;x,v),\quad
P_N(x)=\sqrt N\{J_N(x)-EJ_N\},\quad
T_N(S)=\sqrt N\{A_N(f;S)-EA_N\}.
\tag{NSPF.1}
$$

The exact decomposition is $T_N=F_N+P_N$ with $F_N$ of
(NMAF.15). The conditional velocity integration retains the same
velocity in every face using its address. It is not a new geometric
algorithm or a factorization of face expectations.

The original finite configuration has its fixed $r_N=N^{-1/3}$
even in deletion/insertion comparisons. A proof comparison deleting
one row does not replace it by $r_{N-1}$. Expectations center the
EXECUTED finite-$N$ action; no $o(N^{-1/2})$ deterministic mean-bias
assumption is introduced.
:::

(sec-nspf-add-one)=
## 2. Native insertion and deletion with derived locality

:::{prf:definition} Original limiting add-one action response
:label: def-nspf-add-one

Fix $x=Rn$ and its Poisson position process $\Pi$ of intensity
$p(x)>0$. Let $\bar\xi_z=E_v\xi_z$ for the complete leading
score (NMAF.3) on its common graph. Add a site at the origin,
keeping the SAME other sites, original geometry and mark law.
Define

$$
D_\eta(x,\Pi)=f(x)\sum_z
 \{\bar\xi_z(\Pi\cup\{0\})-\bar\xi_z(\Pi)\},
\tag{NSPF.2}
$$

where the added root contributes its score and an absent root has
zero contribution. Every root in this difference belongs to the
native insertion cavity or to the bounded neighboring stars needed
by its rooted cycles. The sum is finite almost surely, as proved
below. This is the response of the actual local geometric action,
not a separately assigned point-process action.
Put $d_\eta(x)=E_\Pi D_\eta(x,\Pi)$ and
$m_\eta=\int p(x)d_\eta(x)\,dx$.

For $a\in[0,1]$, couple two original local Poisson configurations as

$$
\Pi^0=\Pi^{\rm common}\cup\Pi^{0,\rm private},\qquad
\Pi^a=\Pi^{\rm common}\cup\Pi^{a,\rm private},
$$

with independent component intensities
$(1-a)p(x),ap(x),ap(x)$. The two configurations share their
actual common sites. Define

$$
\mathcal Q_\eta(f)=\int_0^1\left\{
 \int p(x)E[D_\eta(x,\Pi^0)D_\eta(x,\Pi^a)]\,dx
                        -m_\eta^2\right\}da .
\tag{NSPF.3}
$$

The ONE mixing value $a$ is shared across the entire resampling
comparison. No independently chosen mixture per local site is
inserted. This integral is identified below as an original variance limit.
Its integrand is at least $\|d_\eta-m_\eta\|_{L^2(p)}^2$,
by conditioning on the shared Poisson sites; no geometric
nondegeneracy assumption is required.
:::

:::{prf:lemma} Primitive native cavity locality and add-one moments
:label: lem-nspf-cavity

For every finite integer $b\ge1$, $D_\eta$ has a finite $b$th
moment, locally uniformly over the original chart and bounded
phase ratios. Its finite-binomial insertion/deletion versions
have the same population-uniform moments. The required cavity
and its additional cycle stars are protected within a fixed
multiple of a rescaled radius $L$, with failure probability
$C(1+L^3)e^{-cL^3}$.

For a replacement of one ORIGINAL position row $X_i$ by its
independent original copy $X_i'$, let $\Delta_i$ act on $P_N$.
Then

$$
\|\Delta_iP_N\|_b\le C_b/\sqrt N ,
\tag{NSPF.4}
$$

and $\sup_N\|P_N\|_b<\infty$ for every fixed even $b$.
All constants use only the complete register, the chart, the
phase-ratio bound and the proved Gaussian/star series.
:::

:::{prf:proof}
A removed site's Delaunay simplices have their other vertices
among its Delaunay neighbors. A shielded cell puts those neighbors
within twice the shielding radius. Deletion retriangulates only
their cavity: the new simplices lie inside the convex hull of
these neighbors. Thus every changed edge has both endpoints
in that bounded neighborhood. For insertion, the conflict
simplices form the cavity incident to the added site's new star;
its vertices are precisely vertices of that new star. A shield
from the unchanged surrounding sites bounds them in the same
way. The exact full-dimensional Delaunay predicates are in
general position almost surely under these Gaussian/Poisson
laws. No positive numerical-rank test is being removed.

A changed rooted CSR three-cycle must contain a changed edge,
a changed root, or a neighbor of a changed endpoint. Shield
each candidate site using the UNCHANGED surrounding sites, excluding
the deleted/replaced address and every finitely forced root. At large
$N$ these omissions leave the same fixed positive fraction of independent
original rows, so the finite Gaussian empty-ball and count bounds retain
their primitive constants after enlargement. Apply this shielding in
a fixed enclosing ball as in
{prf:ref}`lem-nmaf-localization`. Its finite-window count
union and empty-ball bounds protect every needed additional
star, without assuming independent cavities or neighboring
cells. The same $C(1+L^3)e^{-cL^3}$ envelope results.
On the protected annuli the number of changed root cycles is
polynomial in the original local count, and all leading
Gaussian-averaged scores are polynomials in their edge lengths
with bounded $R_-^{-1}$ and phase ratio.
The original binomial factorial moments and dyadic shield
series therefore prove every stated add-one moment.

The exact Wilson root action has the same moment envelope
(NHS.7) after Gaussian velocity integration. Its long-edge
exception has the original exponentially small physical
shield probability, while its unconditional bound is only
polynomial in $N$. That exponential bound removes this
exception in every fixed moment. Consequently the TOTAL
sum of exact changed root scores from one inserted or
deleted point has a uniform moment bound. Outside the
fixed chart enlargement it contributes zero whenever the
chart roots are shielded. No lower density is required in
the remote Gaussian tail: a remote insertion affecting a
chart root would break that root's own local shield.
Combining removal and insertion proves (NSPF.4).

Here is the last claim directly from the independent original
row inputs. In their chronological coordinate Doob martingale,
each difference is a conditional expectation of one independent
replacement difference, so its $L^b$ norm is at most
$C_b/\sqrt N$. The finite martingale moment inequality gives

$$
\left\|\sum_{i=1}^ND_i^{\rm pos}\right\|_b
 \le C_b'\left(\sum_i\|D_i^{\rm pos}\|_b^2\right)^{1/2}
 \le C_b'C_b .
$$

For even $b$ this inequality follows by expanding the even
power, conditioning away isolated final martingale factors,
and repeated Cauchy--Schwarz; its constant depends only on
$b$. Alternatively the squared-martingale maximal estimate
iterated through these even powers gives the same finite
constant. The centered sum is exactly $P_N$. Hence its
uniform moments are derived from original insertion/deletion
moments, not postulated.
:::

(sec-nspf-resampling)=
## 3. Complete product-resampling bracket and its native limit

:::{prf:lemma} Original coordinate covariance identity and concentrated bracket
:label: lem-nspf-bracket

Let $X,X'$ be the two independent ORIGINAL Gaussian position
arrays, and for a subset $A$ let $X^A$ replace its addressed rows
by $X'$. For $i\notin A$ define
$\Delta_iP_N=P_N(X^{\{i\}})-P_N(X)$ and its corresponding
difference $\Delta_iP_N^A$ at $X^A$. The exact covariance
proxy is

$$
\mathfrak T_N=\frac12\sum_i\sum_{A\subset[N]\setminus\{i\}}
 \frac{\Delta_iP_N\,\Delta_iP_N^A}
        {\binom N{|A|}(N-|A|)} .
\tag{NSPF.5}
$$

It satisfies $E\mathfrak T_N=\operatorname{Var}P_N$ and

$$
\operatorname{Var}\mathfrak T_N\longrightarrow0,\qquad
E\mathfrak T_N\longrightarrow\mathcal Q_\eta(f).
\tag{NSPF.6}
$$

No variance-convergence hypothesis is added.
:::

:::{prf:proof}
The finite independent-coordinate covariance identity is the
same product-space identity proved for (NPB.26). To verify it,
randomly order the $N$ addresses and telescope the conditional
coordinate expectations along that order. Orthogonality of
the resulting differences gives their covariance sum.
At each address the independent two-copy difference supplies
the factor $1/2$. The probability of its predecessor set $A$
is $[\binom N{|A|}(N-|A|)]^{-1}$.
This proves the exact identity for any square-integrable
function of the original independent positions.

For clarity the required concentration is established on a
finite number of RANDOM resampling configurations, never
on all $2^N$ arrays simultaneously. Squaring a conditional
average in (NSPF.5) introduces two independently chosen
subset contexts. Resampling one of its $2N$ original source
coordinates introduces only finitely many additional contexts.
Each context has the ORIGINAL iid Gaussian position marginal,
since its addressed copy selection is independent of those
position values. The logarithmic shield/count events of
{prf:ref}`lem-nmaf-cumulants` apply to this fixed finite
collection, with any prescribed polynomial exception power.

On those events a one-point total-action difference is at
most $(\log N)^C/\sqrt N$. Resampling source $i$ changes
a difference with source $j\ne i$ only when one of its old/new
positions lies in a protected cavity neighborhood of an
old/new position of source $i$, in one of those finite contexts.
Otherwise the two insertion/deletion comparisons commute
and their mixed difference is exactly zero. The radius is
$C r_N(\log N)^{1/3}$ by the preceding cavity lemma.
The number of such addressed $j$ is bounded by the local
Gaussian counts about these finitely many $i$ positions.
Its first and second moments are $O((\log N)^C)$;
they follow by conditioning on those positions and using
the bounded original Gaussian density and binomial factorial
moments. Coinciding addressed sources contribute only the
single $j=i$ term.

The weight sum over $A$ is one for each $j$. Cauchy--Schwarz
in those conditional weights, followed by the just described
finite-context square calculation, therefore gives

$$
E|\Delta_i\mathfrak T_N|^2
 \le C(\log N)^C/N^2+o(N^{-2}).
$$

The exception terms are removed with the original higher
add-one moments and an arbitrarily large shield/source
exception power. The $2N$-coordinate Efron--Stein identity
now gives $\operatorname{Var}\mathfrak T_N
\le C(\log N)^C/N+o(1)\to0$.
This calculation retains all correlated retessellations.

For the mean, the exact beta identity is

$$
[\binom N{|A|}(N-|A|)]^{-1}
 =\int_0^1a^{|A|}(1-a)^{N-1-|A|}\,da .
\tag{NSPF.7}
$$

Conditional on this ONE $a$, give the nonroot addresses their
independent original Bernoulli-$a$ copy choices. On every
fixed protected window the two backgrounds then converge
to Poisson processes with common intensity $(1-a)p(x)$
and private intensities $ap(x)$.
This follows from the finite two-point-per-address generating
series: a given address enters the window with probability
$O(N^{-1})$, while both independent copies enter with
probability $O(N^{-2})$. The error summed over addresses
vanishes. Thus the common/private source allocation is
derived from the actual subset comparison.

Conditional on the original source pair $X_i=x,X_i'=y$,
the scaled replacement difference is the add-one difference
$D_\eta(y)-D_\eta(x)$ in each background. For $x\ne y$
their protected neighborhoods are disjoint in the limit,
and the common/private backgrounds at these TWO spatial
locations have independent Poisson limits. The diagonal
$x=y$ is null under the original two Gaussian densities.
Consequently

$$
\begin{split}
\frac12\lim E[(\sqrt N\Delta_iP_N)
                    (\sqrt N\Delta_iP_N^A)]
 ={}&\int p(x)E[D_\eta(x,\Pi^0)D_\eta(x,\Pi^a)]\,dx\\
 &-\left(\int p(x)d_\eta(x)\,dx\right)^2 .
\end{split}
$$

The exact action and its leading integrated score have the
same add-one limit by (NHS.4) on each protected core.
Original add-one moments remove the cores and dominate
the full $a$ integration. Summing the $N$ identical
addressed terms in (NSPF.5) gives precisely (NSPF.3).
Together with concentration and the covariance identity,
this proves (NSPF.6) and its nonnegative variance limit.
More sharply, conditional on $\Pi^{\rm common}$ the two private
processes are independent with the same law. Hence
$E[D_\eta(x,\Pi^0)D_\eta(x,\Pi^a)]
=E[(E[D_\eta(x,\Pi^0)\mid\Pi^{\rm common}])^2]
\ge d_\eta(x)^2$. Integrating and subtracting $m_\eta^2$
shows that every integrand in (NSPF.3) is at least
$\|h_\eta\|_{L^2(p)}^2$.
:::

(sec-nspf-position-clt)=
## 4. Gaussian retessellation center and its original time covariance

:::{prf:theorem} Complete position-center Gaussian history
:label: thm-nspf-position-history

For every fixed collection of original recorded times,
$P_N(x_k)$ converges to a centered stationary Gaussian
process with covariance

$$
C_{\rm pos}(0)=\mathcal Q_\eta(f),\qquad
C_{\rm pos}(k)=
 \int p_k(x,y)h_\eta(x)h_\eta(y)\,dx\,dy\quad(k\ne0),
\quad h_\eta=d_\eta-m_\eta .
\tag{NSPF.8}
$$

Its variance may be zero; its actual add-one integral decides
that regime. The assertion introduces no positive geometric
variance premise.
:::

:::{prf:proof}
The original product covariance identity also gives Gaussian
characteristic approximation. Use its symmetric covariance identity with $e^{iuP_N}$
in the FIRST replacement factor and $P_N$ in the second.
Taylor-expand that first exponential at the unreplaced $P_N(X)$
under each independent replacement. Its leading term is
$iu E[e^{iuP_N}\mathfrak T_N]$ in the differential equation
for the characteristic function. The remainder is bounded
by $C_u\sum_i E|\Delta_iP_N|^3$ after Cauchy--Schwarz
over its two replacement factors. Equation (NSPF.4)
bounds that sum by $C_u/\sqrt N$.
Replacing $\mathfrak T_N$ by its derived mean costs
$C_u\sqrt{\operatorname{Var}\mathfrak T_N}\to0$.
Solving the resulting scalar characteristic equation with
value one at zero gives
$\exp[-u^2\mathcal Q_\eta(f)/2]$.
This is the same elementary product-space normal
calculation as (NPB.27), with cavity moments replacing
component moments.

For a finite recorded-time linear combination, use one
independent original position-HISTORY block per row address.
Each fixed time marginal has the cavity moments above,
so their finite sum has the same replacement bounds and
bracket concentration. For distinct times $k,l$, the
original paired position law is nonsingular Gaussian.
A row can enter both rescaled protected windows with
probability only $O(N^{-2})$, by its bounded joint
six-dimensional density. The common/private neighborhood
backgrounds at these two time layers therefore have
independent Poisson limits conditional on the root's
original two positions. This is true in both resampling
contexts, with the shared global mixing value retained.

At such distinct times the expected add-one response at
the old/new root is $d_\eta$ independently of the local
background context. The limiting half product of the
two replacement differences is

$$
\frac12E[(d_\eta(X_k')-d_\eta(X_k))
         (d_\eta(X_l')-d_\eta(X_l))]
 =E[h_\eta(X_k)h_\eta(X_l)].
$$

Here the primed history is an independent ORIGINAL copy
and the unprimed pair has exactly density $p_{k-l}$.
No independence of those root positions was used.
The bracket identity and its concentrated finite-time
version consequently give (NSPF.8) and, by arbitrary
real linear combinations, the stated Gaussian history.
:::

(sec-nspf-full-action)=
## 5. Complete original action law and its positive covariance

:::{prf:theorem} Full same-record stationary spatial-action Gaussian fluctuation
:label: thm-nspf-full-action

The complete executed centered action $T_N$ in (NSPF.1),
at every finite set of recorded times, converges to a
stationary Gaussian process with COMPLETE covariance

$$
C_{\rm full}(k)=C_\eta(k;f)+C_{\rm pos}(k).
\tag{NSPF.9}
$$

For a nonzero nonnegative chart test and $\eta>0$ its
variance is at least the strict primitive lower bound
(NMAF.8). Every face, position cavity, shared Gaussian
mark and repeated temporal row is retained. There is no
unresolved position conditional center in this assertion.
:::

:::{prf:proof}
The original mark-centered action $F_N$ has its derived
conditional Gaussian characteristic limit, stable relative
to the ENTIRE original position history, by
{prf:ref}`thm-nmaf-history`. The position-center vector
is measurable in that history. Multiplying its bounded
characteristic test by the conditional characteristic
function and using the latter's $L^1$ convergence shows
that the two limiting vectors are independent.
Their actual finite identity is $T_N=F_N+P_N$.
The original position-center law is the Gaussian law just
proved. The product of their two limiting characteristic
functions gives (NSPF.9), without postulating independent
source blocks at finite population. Its same-time position
variance is nonnegative by (NSPF.6); the mark contribution
has (NMAF.8). This proves the complete positive fluctuation.
:::

(sec-nspf-joint-time)=
## 6. Every-schedule native-time Brownian limit of the full action

:::{prf:theorem} Full-action covariance, joint time limit and commuting orders
:label: thm-nspf-full-joint-time

The COMPLETE Green--Kubo coefficient of the original
$T_N$ converges to

$$
\Sigma_{\rm full}(f)=
 \Sigma_\eta(f)+
 \mathcal Q_\eta(f)+2\sum_{k\ge1}
       \int p_k(x,y)h_\eta(x)h_\eta(y)\,dx\,dy.
\tag{NSPF.10}
$$

All time sums converge absolutely. For every
$N\to\infty,n_N\to\infty$ schedule,

$$
n_N^{-1/2}\sum_{k=1}^{\lfloor n_Nt\rfloor}T_N(S_k)
 \Longrightarrow\sqrt{\Sigma_{\rm full}(f)}\,B_t
 \quad\hbox{in }D([0,1]).
\tag{NSPF.11}
$$

The vector version retains polarized complete covariance.
Both ordered population/time limits agree with this
joint limit. If $f\ge0$ is nonzero and $\eta>0$, its
coefficient is at least $\Sigma_\eta(f)>0$.
At $\eta=0$ the position covariance (NSPF.3) is the
actual remaining fluctuation mechanism, with no assigned
strict positivity if its native coefficient vanishes.
:::

:::{prf:proof}
The original full action has uniform moments of every
fixed even order: (NSPF.4) proves them for its position
center, and {prf:ref}`lem-nmaf-poisson` for its mark
center. Hence $T_N$ has uniform $L^8$ and $L^2$ bounds.
The original full-state Mehler operator is positive and
selfadjoint with centered $L^2$ contraction $r^k$,
independent of $N$. The exact full autocovariance therefore
has uniformly summable tail $C\sum_{k>K}r^k$.
Its finite-lag limits (NSPF.9) imply the complete sum
limit (NSPF.10).

For the position contribution, expand $h_\eta$ in the
original position-Gaussian Hermite basis. Its degree zero
is zero by construction and its covariance terms are
nonnegative sums of squared coefficients times $r^{jk}$,
$j\ge1$. The same-time variance is at least
$\|h_\eta\|_{L^2(p)}^2$ by the singleton input projection
(or by the concentrated resampling bracket). Thus its
complete Green--Kubo contribution is nonnegative.
In particular (NSPF.10) retains the positive mark lower
bound when (NMAF.8) applies.

Its ORIGINAL Poisson series now has uniformly bounded
$L^4$ norm because interpolation of its $L^2$ contraction
with its uniform $L^8$ bound gives
$\|\mathcal P_m^kT_N\|_4\le C r^{k/3}$.
The actual full-state martingale increments have bounded
fourth moments and their exact brackets have bounded
second moments. Full-state Mehler mixing makes the
variance of their $n$-term bracket average $O(n^{-1})$,
uniformly in population. The martingale Lindeberg error
is $O(n^{-1})$ by the fourth moments. The native Poisson
endpoint and fourth-moment tightness argument in
{prf:ref}`thm-nmaf-joint-time` now applies verbatim to
this complete action observable and its identified
covariance. It proves (NSPF.11) on every schedule.
At fixed $N$ it gives Brownian covariance $\Sigma_N$;
at fixed finite times the population limit is (NSPF.9),
whose summable Gaussian covariance gives Brownian
coefficient (NSPF.10). Both ordered limits coincide.
:::

(sec-nspf-scope)=
## 7. Parameter and physical identification scope

This completes the full population and native-time
fluctuation law of the original spatial composite action
in the declared harmonic stationary critical-phase family.
Both velocity marks and the actual position retessellation
center have been proved, and their dependence at finite
population is retained through exact conditioning.

The theorem is parameterized by the complete register,
chart, original phase family and actual positive recording
resonance. Its primitive positive regime is $\eta>0$
and a nonzero nonnegative chart test. The radial
$\eta=0$ action and its separate position variance keep
their exact native integrals. Unbounded charts, fixed
positive phase, positive selection/viscosity, metric
feedback, historical calibration, other graph tags and
finite arithmetic retain their actual separate laws.
No drift, source noise or spatial cutoff has been changed.

This stochastic graph-action identification is not an
identification of the complete local spacetime SU(3)
Yang--Mills probability action. That target still
consumes the original local field and physical
reconstruction correspondence, beyond the scalar
composite spatial-action fluctuations proved here.

