# Nondegenerate population and time fluctuations of an original color history

(sec-nhf-register)=
## 1. The achieved full stationary native color regime

:::{prf:definition} Complete harmonic color fluctuation register
:label: def-nhf-register

Retain every original algorithm, landscape, recording, source, arithmetic,
threshold, cap, donor, calibration and clock parameter of
{prf:ref}`def-nhr-register`.
This is its actual all-alive harmonic, zero-viscosity, constant-fitness,
cap-`None` branch with the configured B1
`ColorSource::RecordedField` potential-force/input-velocity source.
The original recording resonance satisfies $A^m=rI$,
$r=c^{m/2}\in(0,1)$, and has the derived full invariant law.
The default zero-viscosity viscous-color source remains unavailable.
There is no new force, noise, particle floor, action weight or color field.

Let $\mu_1=N(0,C\otimes I_3)$ be its DERIVED single-row state law.
Use the complete recorded stationary row process $Z_{i,k}=(x_{i,k},v_{i,k})$
at its original stride $m$. Its configured raw or projector color
dictionary $D_i$ includes the literal availability and alignment.
A declared real finite list $h(D_i)\in\mathbb R^p$ has
$|h|\le B$, $\bar h=\mu_1h$, and $f=h-\bar h$.
The actual empirical recorded color is
$$
 H_{N,k}=\frac1N\sum_{i=1}^N h(D_{i,k}).
 \tag{NHF.1}
$$
The original threshold is not smoothed. A shared random/current
calibration, additional graph-dependent observation or different
color source retains its separately derived law.

All formulas below are known Gaussian integrals in the complete
original parameters $C,\lambda,\kappa,\delta,r,m,t_*,h$, whose finite
primitive definitions are (NHR.2)--(NHR.5).
Every unused algorithm parameter stays in the register; its lack of
effect follows from the zero actual gates and viscosity, as proved
in Chapter NHR.
:::

(sec-nhf-population-law)=
## 2. The complete actual population color Gaussian law

:::{prf:theorem} Native finite-history population fluctuation field
:label: thm-nhf-population-color

For every finite set of original recorded times $k_1,\ldots,k_l$,
$$
 \left(\sqrt N(H_{N,k_b}-\bar h)\right)_{b=1}^l
       \Longrightarrow
       \left(\mathcal Z_{k_b}\right)_{b=1}^l .
 \tag{NHF.2}
$$
The limiting sequence is centered Gaussian and its COMPLETE
stationary covariance is
$$
 E[\mathcal Z_k\mathcal Z_l^\top]
   =C_h(l-k),\qquad
 C_h(j)=\mu_1[f(Z_0)f(Z_j)^\top]\quad(j\ge0),
 \qquad C_h(-j)=C_h(j)^\top .
 \tag{NHF.3}
$$
Its drift, covariance and all finite joint distributions are identified
by the original row transfer $\mathcal P_m$; there is no independent
incoming-population approximation.
Its actual full-history laws converge in the countable product
topology on $(\mathbb R^p)^{\mathbb Z}$.
Every fixed polynomial moment of the finite list converges.
The two-sided history is the canonical stationary extension of the
DERIVED original kernel. Every finite nonnegative-time list is
the literal recorded history started from its proved invariant law.
:::

:::{prf:proof}
The unchanged algorithm has independent row state/source chains:
each actual gate is zero, each component is a singleton, and both
viscous forces vanish. Reserved donor and source marks do not affect
the consumed configured color. The full invariant physical law is
the proved product $\mu_1^{\otimes N}$.
Consequently the entire stationary row trajectories are independent
identically distributed copies of the actual Gaussian Markov history,
not just independent at one time.

For a fixed real coefficient list $u_b$, put
$Y_i=\sum_bu_b\cdot f(D_{i,k_b})$.
These are bounded independent identically distributed centered variables
with exact variance given by (NHF.3).
The original exponential Taylor formula gives
$$
 E e^{iY_i/\sqrt N}
   =1-\frac{EY_i^2}{2N}+R_N,\qquad
 |R_N|\le E|Y_i|^3/(6N^{3/2}).
$$
Raising this exact characteristic function to its $N$th power
gives the Gaussian limit. Finite-coordinate polarization identifies
the asserted full joint covariance, including its possible null
directions. The common covariance formula makes all finite laws
consistent, and constructs the centered Gaussian sequence.
A finite-window maximum has uniformly bounded second moment.
Choosing coordinate radius budgets with summable error probabilities
gives tightness in the countable product space; a diagonal finite-law
argument identifies every cluster law with this same sequence.

For every integer $q\ge1$, expansion of the $2q$th moment of
$N^{-1/2}\sum_iY_i$ leaves only index patterns in which each
index occurs at least twice, because the rows are independent and
centered. Their number is bounded by a constant depending only on
$q$ times $N^q$, while every factor is bounded.
Thus all even moments are uniformly bounded; using a larger even
moment proves uniform integrability of any fixed polynomial.
The Gaussian finite-law convergence therefore gives the moment claim.
All hard native masks remain inside the original bounded row function.
:::

(sec-nhf-full-temporal-covariance)=
## 3. Exact complete time covariance and a primitive positive color mode

:::{prf:theorem} Full native color Green--Kubo form at every population
:label: thm-nhf-color-green-kubo

Let $f_q$ be the degree-$q$ Hermite projection of $f$ under the
DERIVED $\mu_1$, and set
$V_q=\mu_1[f_qf_q^\top]\succeq0$.
The complete color time covariance is the explicit convergent form
$$
 \Sigma_h=C_h(0)+\sum_{k\ge1}[C_h(k)+C_h(k)^\top]
       =\sum_{q\ge1}\frac{1+r^q}{1-r^q}V_q .
 \tag{NHF.4}
$$
For EVERY admitted population, this is exactly the covariance of
the $\sqrt N$-scaled empirical native color time sum.
It obeys the primitive matrix bounds
$$
 \operatorname{Cov}_{\mu_1}(h)
       \preceq\Sigma_h
       \preceq\frac{1+r}{1-r}\operatorname{Cov}_{\mu_1}(h).
 \tag{NHF.5}
$$
It vanishes in a declared direction exactly when that actual row
color test is constant almost surely.
There is no cancellation by an unobserved kinetic or preparation
component.

In the existing terminal-position-noise-zero branch,
choose the ACTUAL bounded projector readout
$h=\Im(P_{ab}^2)$, $a\ne b$, with the executed zero extension.
Write $V=C_{22}>0$ and
$$
 p_A=\frac{\Gamma(3/2,\delta^2/(2\lambda^2C_{11}))}
                  {\Gamma(3/2)}>0,\qquad
 d_1=\frac{2\kappa Vp_A}{15}e^{-4\kappa^2V}\ne0 .
$$
For every finite configured positive phase and finite threshold,
$$
 \operatorname{Var}_{\mu_1}(h)\ge\frac{d_1^2}{V}>0,\qquad
 \Sigma_h\ge
       \frac{1+r}{1-r}\frac{d_1^2}{V}>0 .
 \tag{NHF.6}
$$
These are population-independent lower bounds for the FULL actual
empirical color covariance and complete long-time covariance.
With positive terminal position noise, the original primitive phase
interval (NHR.12) gives the same conclusion with
$d_1^2$ replaced by $\kappa^2C_0^2$ and the DERIVED $V$.
:::

:::{prf:proof}
The original positive Mehler transfer satisfies
$\mathcal P_m f_q=r^q f_q$.
Orthogonality of its known Gaussian Hermite spaces gives
$C_h(k)=\sum_{q\ge1}r^{qk}V_q$.
It has norm at most
$r^k\mu_1|f|^2$, so the complete covariance series is absolutely
convergent. Summing each nonnegative Hermite block gives (NHF.4).
Its factors lie between one and $(1+r)/(1-r)$; this proves
(NHF.5) and its exact zero-direction characterization.
Independence of the ORIGINAL row histories cancels every cross-row
centered covariance. Their empirical factor $N^{-1}$ cancels
the $\sqrt N$ scaling at every lag and in the whole convergent series.

The original threshold Gaussian integral and all-phase first-Hermite
coefficient were proved in (NHR.11)/(NHR.14):
$E[v_a h]=d_1$. The centered coordinate $v_a$ belongs to the
actual degree-one Hermite space and has square norm $V$.
Cauchy--Schwarz against that projection therefore gives
$E|f_1|^2\ge d_1^2/V$.
The $q=1$ term of (NHF.4) proves its stronger time lower bound.
The primitive positive-phase coefficient (NHR.13) yields the stated
positive-terminal-noise version. No new conditional variance hypothesis
or uniform lower claim for the separate viscous color was used.
:::

(sec-nhf-joint-time-limit)=
## 4. Joint native population/time scaling with no artificial duration ratio

:::{prf:theorem} Nondegenerate full color Brownian limits on every diverging schedule
:label: thm-nhf-joint-time-limit

Let $N\to\infty$ and let the ORIGINAL recorded update count
$n_N\to\infty$ be ANY diverging integer sequence.
Interpolate the exact native color sum
$$
 X_N(t)=\frac1{\sqrt{n_N}}\left[
 \sum_{k\le\lfloor n_Nt\rfloor}
             \sqrt N(H_{N,k}-\bar h)
 +(n_Nt-\lfloor n_Nt\rfloor)
             \sqrt N(H_{N,\lfloor n_Nt\rfloor+1}-\bar h)\right].
 \tag{NHF.7}
$$
On every bounded time interval it satisfies
$$
 X_N\Longrightarrow B_{\Sigma_h}
       \quad\hbox{in }C([0,T],\mathbb R^p).
 \tag{NHF.8}
$$
The covariance is uniquely (NHF.4).
For the original color in (NHF.6) it is strictly positive with
the explicit population-independent lower bound just proved.
Taking the population and long-time limits in either order gives
this same law. The original physical duration is $n_Nmt_*h$
and its complete covariance rate is $\Sigma_h/(mt_*h)$.

No relation such as $n_N\gg N^9$ is required here.
The generic count-phase bound retains that separate sufficient
schedule. This theorem uses the proved independence and positive
transfer of THIS included configured color family.
:::

:::{prf:proof}
For one original row define the centered interpolated path
$Y_{i,n}(t)=n^{-1/2}\sum_{k\le nt}f(D_{i,k})$ with its stated
fractional update. Then $X_N=N^{-1/2}\sum_iY_{i,n_N}$.
These are independent identically distributed ACTUAL paths.

The scalar/vector Poisson series for the row color exists in $L^2$:
$$
 u=\sum_{k\ge0}\mathcal P_m^k f,\qquad
 \|u\|_2\le\|f\|_2/(1-r).
$$
For a convenient fourth-moment proof it also exists in $L^4$.
To see this without a new mixing assumption, apply the original
Gaussian Mehler formula to a bounded $f$. Its correlation with
a point initially at $z$ can be coupled to a stationary starting
Gaussian by the same future noises. For a merely measurable bounded
test, the original normal TV shift and covariance comparison give
$$
 |\mathcal P_m^k f(z)|
 \le C B\,r^k(1+|C_1^{-1/2}z|),\qquad k\ge1,
 \tag{NHF.9}
$$
with the explicit safe $C=8/(1-r^2)$ in the fixed single-row
dimension six. For example comparing
$N(r^kz,(1-r^{2k})C_1)$ with $N(0,C_1)$
uses its mean Gaussian score and covariance score; its variance is
at least $(1-r^2)C_1$. The resulting bound has one power of the
standardized starting norm plus a constant.
Thus $|u(z)|\le 2B+CB r(1-r)^{-1}
(1+|C_1^{-1/2}z|)$, proving primitive fourth moments by original
six-dimensional Gaussian moments.
Here $C_1=C\otimes I_3$; no original noise is truncated.

The corresponding exact row martingale difference
$D_k=u(Z_k)-\mathcal P_m u(Z_{k-1})$ has bounded fourth moment,
but not necessarily a pointwise bound.
For a uniform interval moment one can instead prove directly
from the row's geometric Gaussian mixing that
$$
 E\left|\sum_{k=b+1}^{b+a}f(D_{i,k})\right|^4\le C_4a^2,
 \tag{NHF.10}
$$
with a primitive $C_4$ independent of $a,b,n,N$.
Here is a direct verification.
For scalar centered bounded coordinates, order four indices
$i\le j\le k\le l$.
Reversibility and conditional expectation at $j,k$ give
$$
 |E[f_i f_j f_k f_l]|
 \le 4B^4\min\{r^{j-i},r^{l-k}\}
 \le4B^4 r^{((j-i)+(l-k))/2}.
$$
The first bound conditions the last centered factor on time $k$
and uses the $L^2_0$ row operator norm $r^{l-k}$;
the other conditions the first on time $j$ using the proved
stationary reversed kernel. Each centered coordinate has
$L^2$ norm at most $B$, since the uncentered vector has norm at
most $B$. The product of the other three centered coordinates
has $L^2$ norm at most $4B^3$: use the $L^2$ norm for one and
the bound $2B$ for the other two. This proves the factor $4B^4$.
Summing their two outer gaps gives a geometric constant
$(1-\sqrt r)^{-2}$, while the middle gap and initial index
have at most $a^2$ choices. Repeated indices are covered by
the same bound or bounded coincident terms. The at most $24$
orderings and finitely many vector-coordinate terms give, for example,
a safe $C_4=384p^2B^4/(1-\sqrt r)^2$.
This proves (NHF.10) directly.

For $|t-s|\ge1/n$, the complete block and two fractional
endpoints of $Y_{i,n}$ therefore have fourth moment at most
$C'_4|t-s|^2$, with one primitive larger constant.
For $|t-s|<1/n$, its increment is at most
$2B\sqrt n|t-s|$, so the same bound follows.
The covariance also obeys
$E|Y_{i,n}(t)-Y_{i,n}(s)|^2\le C'_2|t-s|$,
by its summable original $r^k$ covariance and the same mesh argument.
Independence and centering across rows give
$$
 E|X_N(t)-X_N(s)|^4
 \le C''_4|t-s|^2
$$
uniformly in $N,n$, by the expansion of the fourth moment of
their normalized independent sum.
The dyadic-grid argument from the complete native functional CLT
therefore proves tightness of $X_N$ in the continuous path space.

For finite-dimensional limits, each coefficient list gives
a one-row variable $V_{i,n}$ with uniformly bounded fourth moment.
Its exact variance converges, as $n\to\infty$, to that of
the Brownian covariance (NHF.4). Indeed
$$
 \left|\frac1n\operatorname{Cov}(S_{\lfloor nt\rfloor},
                              S_{\lfloor ns\rfloor})
            -\min(s,t)\Sigma_h\right|
 \le C_{\rm cov}/n
$$
on a fixed time interval, where summing the at most $k$
boundary pairs gives a primitive multiple of
$\sum_{k\ge1}k r^k=r/(1-r)^2$.
Fractional endpoints add the same order by the summable covariance.
The row-averaging characteristic proof of Section 2 now has
a uniformly $O(N^{-1/2})$ third-moment error.
It therefore gives the unique Brownian finite laws for every
simultaneously diverging $n_N,N$; tightness proves (NHF.8).

For fixed $n$, Section 2 gives the Gaussian finite-history
population limit. Its covariance converges to Brownian covariance
as $n\to\infty$, with the uniform path moments above, proving
that order. For fixed $N$, the exact row martingale displayed above
has an $L^2$ bracket function because its differences are in $L^4$.
The DERIVED centered row operator norm $r$ gives convergence of its
bracket average by the same covariance-sum argument used for the
complete native functional CLT. Bounded radial truncation of a
martingale difference, subtracting its actual entering conditional
mean, gives a bounded martingale approximation. Its bracket proof,
conditional Taylor argument and fourth-moment tightness give the
functional martingale limit exactly as in that native proof;
no uniform pointwise TV mixing is required here.
The approximation error is controlled uniformly in duration by
the $L^2$ martingale maximum inequality.
The Poisson endpoint is $\mathcal P_m u(Z_0)-\mathcal P_m u(Z_n)$;
its normalized path maximum vanishes by the stationary union-tail
argument for an $L^2$ endpoint used in
{prf:ref}`thm-nfs-l2-full-record`.
The same $L^2$ second-moment identity identifies its bracket with
the absolutely convergent (NHF.4).
Thus each original row has that native functional CLT. The finite
normalized sum of the independent row limits has the same Brownian
covariance. Taking $N\to\infty$ thereafter leaves its law unchanged.
This proves both ordered limits.
The physical clock is the originally recorded stride, giving the
stated duration and covariance rate.
:::

(sec-nhf-regional-covariance)=
## 5. Exact actual regional covariance and the recorded CAR assignment

:::{prf:theorem} Native regional color counts retain a nonzero normalized CAR coefficient
:label: thm-nhf-native-regional-car

Use the SAME full stationary harmonic branch, and its actual B1
position and available projector records.
For two bounded disjoint native spatial regions $O,V$, put

$$
p_O=\int_O\mathbf1_{\{\lambda|x|>\delta\}}
        (2\pi C_{11})^{-3/2}e^{-|x|^2/(2C_{11})}\,dx ,
$$
$$
B_{O,N}=\frac1N\sum_i
        \mathbf1_O(x_i)\operatorname{tr}P_i .
\tag{NHF.11}
$$

Choose $p_O,p_V>0$, which is an evaluated primitive region test.
For example the two original coordinate balls centered at
$\pm2(\delta/\lambda+\sqrt{C_{11}})e_1$ with radius
$\sqrt{C_{11}}/4$ pass at every finite original threshold.
Then $p_O+p_V<1$ and the SAME executed record has

$$
\operatorname{Cov}(B_{O,N},B_{V,N})=-\frac{p_Op_V}{N},
\qquad
\operatorname{Var}(B_{O,N})=\frac{p_O(1-p_O)}N .
\tag{NHF.12}
$$

For the existing normalized centered mode assignment
$f_O=(B_{O,N}-p_O)/\sqrt{\operatorname{Var}B_{O,N}}$,
the actual cross coefficient is

$$
s_{OV}=\langle f_O,f_V\rangle
   =-\sqrt{\frac{p_Op_V}{(1-p_O)(1-p_V)}}\in(-1,0),
\tag{NHF.13}
$$

independently of population.
When the existing recorded regional CAR construction assigns these
actual descriptor modes to its region, its bounded even number
operators have the nonzero commutator norm

$$
\|[n_{f_O},n_{f_V}]\|
       =|s_{OV}|\sqrt{1-|s_{OV}|^2}>0 .
\tag{NHF.14}
$$

This tests the literal covariance-based CAR assignment of the
book on an actual native color/position regime.
It does not identify disjoint native coordinate regions with
physical spacelike regions without the separate causal correspondence.
It does not obstruct the already derived positive color-history transfer.
:::

:::{prf:proof}
The actual available projector has trace one and the unavailable
extension trace zero. Its single-row regional trace is the indicator
of $\{x\in O,\lambda|x|>\delta\}$.
Different ORIGINAL row histories are independent, and disjoint regions
make the two same-row indicators have product zero.
Expansion of their native empirical sum therefore gives (NHF.12)
and its means $p_O,p_V$. The known positive Gaussian density makes
the named balls' probabilities positive; their distance from the
origin exceeds the literal force threshold radius.
Boundedness of $O\cup V$ leaves positive original Gaussian mass outside,
so $p_O+p_V<1$. Normalization gives exactly (NHF.13).

These are functions of the actual regional B1 position/color descriptor,
so they belong to the declared centered regional mode spaces of
{prf:ref}`thm-ym-hk-record-instantiation`.
The original regional union has dimension at least three:
take a third bounded available region disjoint from the first two.
Their three native count modes have Gram matrix proportional to
$\operatorname{diag}(p)-pp^\top$, which is positive definite because
all three probabilities are positive and their sum is less than one.
Indeed Cauchy--Schwarz bounds $(\sum p_jt_j)^2$ by
$(\sum p_j)(\sum p_jt_j^2)$, strictly below the latter for a nonzero
real vector. Thus three independent actual modes exist.
The exact even-CAR number-operator formula in that
same theorem gives (NHF.14); its even vacuum restriction retains
that norm by wedging the tested two-mode plane with an orthogonal
third mode. Every field, inner product and probability belongs to
the original record and its proved invariant law.
The final physical qualification distinguishes the region assignment
tested here from a still unidentified spacetime local algebra.
:::

(sec-nhf-complete-scope)=
## 6. Evaluated positivity and exact remaining scope

:::{prf:corollary} Original all-population nondegenerate color witness
:label: cor-nhf-positive-witness

Retain every original parameter of
{prf:ref}`cor-nhr-positive-family`:
$h=b_O=1,\ c=(3-\sqrt5)/2,\ \gamma=-\log c,\
\lambda=4/(1+c),\ \sigma_x=\nu=0,\ m=3$,
its full stationary law, cap `None`, original positive fixed phase
calibration, actual recorded-potential color and any finite threshold.
Then (NHF.2), (NHF.4), (NHF.6) and (NHF.8) hold for every admitted
population and every diverging native time schedule.
The nonzero lower bound is an exact finite Gaussian expression in
$\kappa,\delta,\lambda,C_{11},C_{22},r$, given in (NHF.6).
The positive physical gap remains $\gamma/(2t_*)$ from the
SAME actual complete color-history transfer.
:::

:::{prf:proof}
The primitive resonance and all-phase original coefficient were
derived in Chapter NHR. Its $r=c^{3/2}$ is strictly between zero
and one, and every finite threshold has $p_A>0$.
The original phase tag gives finite $\kappa>0$.
Substitution proves the strictly positive coefficient and gap.
Apply the original independent row-history and complete covariance
proofs above; all other original unused parameters retain their
already proved zero-gate/source scope.
:::

:::{prf:remark} Achieved color fluctuations and retained interacting endpoint
:label: rem-nhf-interacting-scope

This proves an instantaneous full finite-history population Gaussian
color law, a unique nonzero complete temporal covariance, joint
population/time Brownian dynamics on every diverging schedule,
and a population-uniform physical gap for an existing FULL stationary
configured color family. Its original hard threshold is included.

The positive-viscosity active-cloning gas has its different full
instrument variance, covariance-cluster and joint-time limits in
{prf:ref}`thm-npf-complete-green-kubo` and
{prf:ref}`cor-npf-joint-time-limit`.
Its instantaneous population Gaussian identification and uniform
nondegeneracy require their own chronological bracket/linearization.
This harmonic recorded-potential color is not the default viscous
source, a non-Abelian spacetime vacuum ensemble, or its physical
local quantum algebra. Finite arithmetic and source-stream comparisons
remain explicit. No result for those different regimes is inferred
by changing the name of this positive native color channel.
:::
