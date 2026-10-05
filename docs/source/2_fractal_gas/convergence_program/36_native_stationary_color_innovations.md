# Stationary native color innovations and their discrete drift and bracket

(sec-nsi-register)=
## 1. The selected phase and the actual terminal record

:::{prf:definition} Stationary native innovation register
:label: def-nsi-register

Retain every algorithm, landscape, sampling, collision, boundary,
arithmetic, recording and physical-calibration parameter of
{prf:ref}`def-nqc-register`. Use the existing quadratic count gas in
its proved positive-viscosity contraction regime, with the configured
cap, current-frame donors, sampled normalization, mandatory revival,
simultaneous copying, shared component Haar variables and both force
kicks. The initial law is its actual full marked QSD $\nu_N$.
The uniquely proved population fixed point is $\mu_*$.

Use the matched B2 projector of {prf:ref}`def-ngf-parameter-ledger`,
with its actual phase $\kappa_c$, threshold $\delta_c$, finite-value
and force-source conventions. Its zero extension is $P_i$, with
$\|P_i\|_F\le1$. Positive results use the all-slot matched B2 channel.
Additional pre-terminal masks remain factors of $P_i$ in a channel
that consumes them. Terminal alive-only observations retain their
literal indicator in the spatial test. This is a covariant instrument
in the recorded ambient frame; no orbit-quotient identification is made.

Let $\mathcal B_n$ be the complete record through update $n$'s B2
and velocity cap, before its final position innovation. It contains
the entire preceding history, current preparation, OU innovations,
forces and consumed calibration. On a nonextinct entering path,

$$
Y_{i,n}=X_{i,n}+sZ_{i,n},\qquad s=\sigma_x\sqrt h>0,\qquad
Z_{i,n}\stackrel{\rm independent}{\sim}N(0,I_3).
\tag{NSI.1}
$$

Conditional on $\mathcal B_n$, these are the original independent
innovations and all $P_{i,n},X_{i,n}$ are fixed. Their dense-force
correlations remain in that record. The phase/source calibration is
fixed or preceding-record measurable as specified by the complete
register. A deterministic limiting bracket uses its fixed consumed
values; a fluctuating current calibration needs its own joint limit.

Choose bounded measurable spatial tests $\varphi_a$ and fixed Hermitian
matrices $T_a$, $a=1,\ldots,m$. Put $G_{i,n}^a=\operatorname{tr}(T_aP_{i,n})$
and

$$
\mathcal G_sf(x)=Ef(x+sZ),\qquad
C_s^{ab}(x)=\mathcal G_s(\varphi_a\varphi_b)(x)
                -\mathcal G_s\varphi_a(x)\mathcal G_s\varphi_b(x).
\tag{NSI.2}
$$

The exact terminal innovation of their empirical averages is

$$
\Delta M_{N,n}^a=\frac1{\sqrt N}\sum_iG_{i,n}^a
 [\varphi_a(Y_{i,n})-\mathcal G_s\varphi_a(X_{i,n})].
\tag{NSI.3}
$$

Stop this observation after actual extinction, setting subsequent
increments to zero. This observes the killed gas without adding a
restart transition. Its native clock is $nt_*h$.
:::

(sec-nsi-finite)=
## 2. Exact complete-record drift and bracket

:::{prf:theorem} Exact conditional terminal color drift and bracket
:label: thm-nsi-exact-bracket

The conditional drift of (NSI.3) is zero. Its exact conditional
covariance on a nonextinct entering path is

$$
\Sigma_{N,n}^{ab}=\frac1N\sum_iG_{i,n}^aG_{i,n}^bC_s^{ab}(X_{i,n}).
\tag{NSI.4}
$$

After extinction this matrix is zero. Thus
$M_{N,k}=\sum_{n=1}^k\Delta M_{N,n}$ is a discrete martingale for
the terminal-record filtration, with predictable bracket
$\sum_{n=1}^kE[\Sigma_{N,n}\mid\mathcal T_{n-1}]$.
In the finer stage filtration, $\mathcal B_n$ immediately precedes
the increment and its bracket contribution is (NSI.4).

For $u\in\mathbb R^m$ set
$L_u=2\sum_a|u_a|\|T_a\|_F\|\varphi_a\|_\infty$.
For $N\ge\max\{1,2L_u^2\}$,

$$
\left|E[e^{iu\cdot\Delta M_{N,n}}\mid\mathcal B_n]
                      -e^{-u^\top\Sigma_{N,n}u/2}\right|
\le e^{E_N(u)}E_N(u),\qquad
E_N(u)=\frac{L_u^3}{6\sqrt N}+\frac{L_u^4}{N}.
\tag{NSI.5}
$$

These identities retain the actual coupled record.
:::

:::{prf:proof}
Conditional on $\mathcal B_n$, the summands are independent and
centered, by the original final Gaussians. Their second moments give
(NSI.4). The tower property gives the terminal-filtration drift and
bracket; stopping at the observed extinction preserves them.

Let $W_i=\sum_a u_aG_{i,n}^a
[\varphi_a(Y_{i,n})-\mathcal G_s\varphi_a(X_{i,n})]$.
Its conditional mean is zero and $|W_i|\le L_u$. Taylor's formula gives

$$
E[e^{iW_i/\sqrt N}\mid\mathcal B_n]
=1-\frac{E(W_i^2\mid\mathcal B_n)}{2N}+r_i,\qquad
|r_i|\le\frac{L_u^3}{6N^{3/2}}.
$$

For the stated $N$, each factor's deviation from one is at most
$L_u^2/N\le1/2$. The logarithm series satisfies
$|\log(1+z)-z|\le|z|^2$ for $|z|\le1/2$.
The sum of the $N$ logarithms consequently differs from
$-u^\top\Sigma_{N,n}u/2$ by at most $E_N(u)$.
Use $|e^w-1|\le e^{|w|}|w|$ to obtain (NSI.5).
On an extinct path both characteristic functions are one.
:::

(sec-nsi-phase)=
## 3. The bracket determined by the proved native phase

:::{prf:definition} Native stationary pre-terminal color law
:label: def-nsi-phase-law

Let $\eta_*$ be the actual population preparation/B1 coordinate law
from $\mu_*$, identified by {prf:ref}`thm-nmg-joint-color-geometry`.
For $A=(p,v_1,m,\ldots)\sim\eta_*$ and an independent original
$Z\sim N(0,I_3)$, put

$$
z=cv_1+qZ,\qquad X=p+tz,\qquad
F=F_{\eta_*}^{\rm pop,count}(X,z).
$$

Evaluate the original thresholded B2 projector $P(A,Z)$ with its
recorded phase. Write $\Lambda_*^{\rm col}$ for the law of $(X,P)$.
Its preparation dependence and complete population force are retained;
no force computed solely from a local star is substituted. Define

$$
\Sigma_*^{ab}=\int\operatorname{tr}(T_aP)\operatorname{tr}(T_bP)
                 C_s^{ab}(X)\,\Lambda_*^{\rm col}(dX,dP).
\tag{NSI.6}
$$

This finite positive semidefinite matrix is determined by the complete
native parameters and their uniquely proved phase.
:::

:::{prf:lemma} Actual stationary pre-terminal empirical consistency
:label: lem-nsi-preterminal-consistency

For every bounded continuous $f(X,P)$ of the availability-marked
projector,

$$
\frac1N\sum_i f(X_i,P_i)\longrightarrow\int f\,d\Lambda_*^{\rm col}
\quad\hbox{in probability and in }L^1
\tag{NSI.7}
$$

under the actual incoming QSD and raw update. Use $\delta_c>0$, or
$\delta_c=0$ in the derived $a_x\ge0$ regimes of the marked-geometry
threshold lemmas. In particular (NSI.4) converges to (NSI.6) in $L^1$.
:::

:::{prf:proof}
The entering empirical law converges to $\mu_*$ by
{prf:ref}`thm-nqc-stationary-rate`, with the coordinate quadratic
transport of {prf:ref}`cor-nqc-quadratic-transport`.
The stopped preparation bias and variance of
{prf:ref}`lem-nqc-preparation`, combined with
{prf:ref}`lem-native-phase-preparation-coupling`, identify its compact
empirical preparation with $\Pi(\mu_*)$. The original independent
jitters, their higher moments and the actual first count kick give
its B1 coordinate projection $\eta_*$. This is precisely the full
preparation consistency proved in
{prf:ref}`thm-nmg-joint-color-geometry`.

Apply {prf:ref}`thm-nmg-assembled-law` to $f(X,P)$ multiplied by a
continuous compact terminal-position localization. This is a permitted
observation of the retained B2 inputs and color marks; it need not
consume a star. The comparator integrates the actual posterior mark
law against its terminal density. The Gaussian Bayes identity of
{prf:ref}`lem-nmg-exact-posterior` makes this exactly
$E[f(X,P)\psi(X+sZ)]$ in the population record.
Its environment is deterministic in this phase. The derived null
threshold boundary permits the original hard mask.

Remove the terminal localization using the primitive position moments,
for both the raw empirical and population outputs. This proves (NSI.7).
Boundedness upgrades probability convergence to $L^1$.
For bounded measurable $\varphi_a$, translation continuity in $L^1$
of the Gaussian density makes $\mathcal G_s\varphi_a$ and
$\mathcal G_s(\varphi_a\varphi_b)$ continuous.
Thus the bounded test in (NSI.6) is permitted by (NSI.7), proving the
bracket convergence.
:::

(sec-nsi-process)=
## 4. Several-update stationary innovations and survival transfer

:::{prf:theorem} Native stationary terminal innovation process
:label: thm-nsi-stationary-process

For every fixed number $K$ of existing updates,

$$
(\Delta M_{N,1},\ldots,\Delta M_{N,K})
\Longrightarrow(\mathcal Z_1,\ldots,\mathcal Z_K),\qquad
\mathcal Z_n\stackrel{\rm independent}{\sim}N(0,\Sigma_*).
\tag{NSI.8}
$$

Hence the cumulative process on the native clock $nt_*h$, through that
fixed window, converges to a Gaussian random walk with zero drift and
bracket $k\Sigma_*$. This same limit holds for the actual killed path
conditioned on survival through $K$, with its attached pre-terminal
records. Its whole-window selection error is at most $K(1-a_0)^N$.
This is a stationary selected-phase color innovation limit.
:::

:::{prf:proof}
Start from $\nu_N$. Conditional on survival to the beginning of
update $n$, the actual current law is
$\nu_NQ_N^{n-1}/\alpha_N^{n-1}=\nu_N$.
The landing certificate bounds extinction at each nonextinct step
by $(1-a_0)^N$, hence extinction through $K$ by $K(1-a_0)^N$.
Apply (NSI.7) to each of these finitely many conditional current laws.
The stopped covariance is zero on the remaining event, whose
probability tends to zero. Thus $\Sigma_{N,n}\to\Sigma_*$ in $L^1$.

For fixed $u_1,\ldots,u_K$, condition the joint characteristic function
on $\mathcal B_K$, where all earlier increments are measurable.
Equations (NSI.5)--(NSI.7) replace the final factor by
$e^{-u_K^\top\Sigma_*u_K/2}$ with an $L^1$ error tending to zero.
Repeat backward through the updates. The result is
$\prod_n e^{-u_n^\top\Sigma_*u_n/2}$.
Bounded second moments from (NSI.4) make these finite vectors tight.
Convolution with an arbitrarily small independent Gaussian identifies
each subsequential law from this characteristic function by Fourier
inversion of its integrable Gaussian-smoothed characteristic function.
Letting the auxiliary smoothing vanish identifies the unsmoothed law.
Thus the limiting independence is derived; native updates were not
assumed independent.

Conditioning a record on an event of probability $1-\epsilon$ changes
its full law by TV at most $\epsilon$. Apply this to actual survival
through $K$. The bound is preserved by the $\sqrt N$ measurable
observation and by attaching pre-terminal records.
Finite sums give the cumulative process and its limiting drift/bracket.
:::

(sec-nsi-positive)=
## 5. A derived nonzero stationary color covariance

:::{prf:theorem} Positive stationary traceless color bracket
:label: thm-nsi-positive-bracket

Take the all-slot matched B2 instrument, $\delta_c=0$, and the proved
count phase with $a_x\ge0$, $q,s,\nu>0$.
Let $T_1,\ldots,T_8$ be an orthonormal real basis of traceless Hermitian
$3$-by-$3$ matrices for the Frobenius inner product.
Use the same bounded spatial test $\varphi$ in all eight components.
If $\varphi$ is nonconstant modulo Lebesgue-null sets, then

$$
\operatorname{tr}\Sigma_*=\frac23
 \int\operatorname{Var}[\varphi(X+sZ)]\,
                         \Lambda_*^{\rm col}(dX,dP)>0.
\tag{NSI.9}
$$

At least one actual traceless projector component therefore has
nonzero stationary innovation variance. This includes the literal
final alive-only test $\varphi=\mathbf1_D$ and holds for every finite
recorded phase $\kappa_c$.

For a rectangular box $D=\prod_{\ell=1}^3[-R_\ell,R_\ell]$ with
$R_\ell>0$, define the following primitive constants:

$$
H_{X,2}=[|a_x|(R_D+\sigma_J\sqrt3)+bV_c+tq\sqrt3]^2,\qquad
M=\sqrt{2H_{X,2}},
$$
$$
a_M=\prod_{\ell=1}^3
 [\Phi((R_\ell-M)/s)-\Phi((-R_\ell-M)/s)]>0,
$$
$$
b_D=1-\prod_{\ell=1}^3[2\Phi(R_\ell/s)-1]>0.
\tag{NSI.10}
$$

For the actual alive-only cylinder,

$$
\operatorname{tr}\Sigma_*\ge a_Mb_D/3>0.
\tag{NSI.11}
$$

All other active parameters remain in the complete phase register;
the displayed moment bound is valid independently of their realized
source, component and force correlations.
:::

:::{prf:proof}
The derived threshold lemmas in Chapter NMG put zero posterior mass
on $F=0$ throughout the stated regime. Integrating their posterior
over terminal positions shows that the population B2 projector is
available almost surely. For each such rank-one projector,

$$
\sum_{a=1}^8[\operatorname{tr}(T_aP)]^2
=\|P-I_3/3\|_F^2=2/3.
$$

Sum the eight diagonal entries of (NSI.6) to obtain (NSI.9).
The Gaussian centered at any finite $X$ has a strictly positive
density everywhere. Zero variance would make $\varphi$ constant
modulo Lebesgue-null sets, so the integrand is positive under the
stated condition.

The actual preparation has $|U|\le V_c$, and
$X=a_xX^J+bU+tqZ$. Minkowski and the original jitter give
$E|X|^2\le H_{X,2}$ in the population law.
Markov gives probability at least $1/2$ to $|X|\le M$.
On that set, Gaussian landing in the rectangular box is at least
$a_M$. Its maximum over all centers is the product of centered
interval probabilities in (NSI.10), so its complement is at least
$b_D$ everywhere. Therefore
$\operatorname{Var}(\mathbf1_D(X+sZ))\ge a_Mb_D$ on this half-mass
set. Equation (NSI.9) proves (NSI.11). Neither a deterministic
finite-population alive floor nor a noise truncation is used.
:::

:::{prf:corollary} Positive stationary innovations at the configured positive threshold
:label: cor-nsi-positive-threshold

In the same count phase, if the original threshold satisfies
$\delta_c<L_j(r)$ for a primitive certificate of
{prf:ref}`lem-nmg-primitive-availability`, then the eight-component
stationary process remains nonzero. Its exact trace bracket is

$$
\operatorname{tr}\Sigma_*=\frac23
 \int \mathbf1_{\{|F|>\delta_c\}}
       \operatorname{Var}[\varphi(X+sZ)]\,d\Lambda_*^{\rm full}>0,
\tag{NSI.12}
$$

Here $\Lambda_*^{\rm full}$ is the same complete pre-terminal
population input law before its declared availability mask.
Every bounded $\varphi$ nonconstant modulo Lebesgue-null sets is allowed.
The positive witness (PC.37), with its existing threshold
$\delta_c=10^{-12}$, satisfies this certificate using $j=3,r=10$.
:::

:::{prf:proof}
The primitive certificate gives a nonempty open available set for every
own preparation $p$. The original $z=cv_1+qZ$ has a full Gaussian
density there, so the full pre-terminal population law assigns positive
mass to availability. For each available projector the same traceless
identity gives $2/3$; it gives zero for the declared zero extension.
Equation (NSI.6) gives (NSI.12). Its integrand is strictly positive on
that positive-mass available set.

Here is a conservative certificate for the actual positive witness.
It has $a_x=0$, $t=1/2$, $\nu=.01$, $\rho=1$,
$q^2=(1-e^{-2})/2<1/2$, $b<.685$, $c<.37$,
$\lambda<3$, $\sigma_J=.1$, $R_D=2\sqrt3<3.5$ and
$U_*=V_c\le2V_0=.2$.
The standard three-dimensional Gaussian gives
$p_3=\operatorname{erf}(3/\sqrt2)
 -3\sqrt{2/\pi}e^{-9/2}>.95$.
Thus $1<B<9/8$, $C_B>.8$ and $M_3<.14$.
The exact profiles of (NMG.27) obey

$$
D_3(5)<13.27,\qquad g_3(5)<7.6,\qquad
C_3(5)<.37[.2+1.5(3.5+.76)]+.25(5+.14)<4.
$$

Consequently

$$
L_3(10)>.01(.8)(.95)e^{-13.3}(10-4)>10^{-12}.
\tag{NSI.13}
$$

These bounds use the actual cap's upper bound $V_0$; the configured
exact $V_{\rm crit}/2$ satisfies it.
Thus the positive threshold, unbounded original noises, all active
gates, source correlations and force evaluations remain unchanged.
:::

:::{prf:lemma} Literal B1 reader availability remains a separate stage test
:label: lem-nsi-b1-availability

For the same capped gas, its matched B1 viscous force satisfies

$$
|F_i^{\rm B1,count}|\le2\nu V_c(1-N^{-1}),\qquad
|F_i^{\rm B1,row}|\le2\nu V_c
\tag{NSI.14}
$$

on every actual preparation and every realization. The singleton
force is zero by its original convention.
Thus a strict color availability test with
$\delta_c\ge2\nu V_c$ deletes every B1 viscous color for both
normalizations. At a zero-velocity consensus input the B1 channel
is identically unavailable for every $\delta_c\ge0$, while the
original subsequent OU/B2 stages retain their own force law.
:::

:::{prf:proof}
The original copied/collided velocities have norm at most $V_c$.
Every B1 difference therefore has norm at most $2V_c$.
Count normalization sums at most $N-1$ kernel weights, each at most
one, divided by $N$. Row normalization is a convex average of
the same differences whenever its degree is positive.
These are (NSI.14); the strict threshold proves unavailability.
At zero consensus every difference is zero before the OU stage.
:::

:::{prf:remark} Parameter regimes and remaining identification
:label: rem-nsi-scope

The explicit positive witness of
{prf:ref}`cor-native-phase-positive-witness` has $a_x=0$ and
$q,s,\nu>0$. Both its configured positive threshold $10^{-12}$
and the included zero-threshold instrument realize the nonzero
stationary process, by {prf:ref}`cor-nsi-positive-threshold`
and {prf:ref}`thm-nsi-positive-bracket` respectively.
For $\delta_c>0$, (NSI.4)--(NSI.8) retain the actual
availability-weighted bracket. A derived positive-threshold
availability certificate supplies positivity when applicable;
an unavailable channel has zero bracket.
At $s=0$ the terminal innovation is identically zero; at $\nu=0$
the declared viscous color channel is unavailable.
Alternate transition/readout tags retain their distinct stage laws.
In particular the executable Rust `colors` reader in
`algorithmic-gas/crates/algorithmic-gas/src/physics/qft.rs` consumes
matched B1 force/input-velocity records. Its literal stage is governed
by (NSI.14), and is not assigned the positive B2 result.
The declared matched B2 instrument observes the original later force;
it adds no kinetic update or noise. Each recording tag retains this
distinction.

The centering is the original conditional terminal expectation.
This theorem identifies a stationary several-update component of
native color fluctuations; earlier normalization, component, jitter
and OU innovations also contribute to the complete empirical
fluctuation around its stationary mean.
A full stationary fluctuation decomposition, a joint small-step/graph
limit and physical local Yang--Mills reconstruction retain their
separate obligations.
:::
