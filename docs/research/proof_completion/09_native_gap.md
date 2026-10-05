# Native spectral gap: exact fitness reduction and a derived dispersion certificate

(sec-native-gap-parameters)=
## 1. Executed spectrum and complete parameters

:::{prf:definition} Complete parameters of the native Dirac readout
:label: def-native-gap-parameters

Fix the complete execution record $\mathfrak P$ of
{prf:ref}`def-native-complete-execution-record` and its realized history $Y$.
The algorithm, all nested configuration fields, providers and landscape
parameters, initial law, innovation and arithmetic convention, stage and mask
data, error policies and physical units remain part of $\mathfrak P$.
The statements about a matrix below are evaluated functions of these data;
they do not replace their history law by an independent sample model.

Retain every field of the existing Python `DiracSpectrumConfig`:
`mc_time_index`, `epsilon_clone`, `kernel_mode`, `epsilon_c`, `lambda_alg`,
`h_eff`, `include_phase`, `n_generations`, `color_threshold`, `min_sector_size`,
`svd_top_k`, `time_average`, `time_range`, `warmup_fraction`, and
`max_avg_frames`. Retain the actual `RunHistory`, its bounds, periodic flag,
parameter header, recorded fitness, alive mask, cloning mask and viscous force.
Missing viscous-force recording has the implemented zero-array fallback.
The selected frame is the implemented clipping to the available frame range.
The phase-space branch consumes its recorded `x_before_clone` and
`v_before_clone`; no stage realignment is inserted.

Let $I$ be the selected alive indices, $m=|I|$, and $V_i$ their stored fitnesses.
Write $\varepsilon$ for the *readout* `epsilon_clone` and $\tau_0=10^{-30}$
for its denominator threshold. This $\varepsilon$ need not equal the gas's
cloning regularizer. The default `fitness_ratio` recipe is

$$
b_i=V_i+\varepsilon,\qquad
d_{ij}=\begin{cases}b_i b_j,&|b_i b_j|\ge\tau_0,\\
\tau_0,&|b_i b_j|<\tau_0,\end{cases}\qquad
T_{ij}=\frac{b_j^2-b_i^2}{d_{ij}}.
$$

The analytic formulas concern exact evaluation on the represented input
values. A rounded executed matrix retains its actual arithmetic error, as
specified in {prf:ref}`prop-native-gap-arithmetic` below.

The implemented spectral helper evaluates

$$
\mathfrak h(T)=\tfrac12\bigl(iT+(iT)^*\bigr)
$$

and sorts the absolute values of its Hermitian eigenvalues. For real $T$,
this is $iT$. For comparison with
{prf:ref}`def-lqft-edge-spectral-parameters`, choose the already directed
coefficient recipe $K=T/2$. Then $i(K-K^{\mathsf T})=iT$.
Using $T$ itself as the directed recipe instead gives $2iT$ and doubles every
rate below. The declared recipe and physical-time conversion decide which
convention is consumed. A positive rate conversion $c_t$ multiplies the rates;
the action calibration $\hbar_{\rm eff}$ multiplies the resulting energies.
Neither conversion changes the underlying transition.

The code's four classification subsets are exactly the alive rows with the
two marks `will_clone` and $\|F^{\rm visc}\|>$ its selected threshold. A numeric
threshold is used literally; otherwise the alive median is used. Each sector
matrix is the corresponding principal submatrix of $T$, and is reported only
when its size reaches `min_sector_size`. These index subsets are not, by that
classification alone, Haar projections for a unitary gauge representation.
The latter representation, when declared, remains the additional data of
{prf:ref}`thm-ym-edge-gauge-spectral-certificate`.

The top-$k$ truncation, paired-value deduplication and generation clustering
operate on the computed list; they do not change the underlying matrix.
`time_average` averages sorted per-frame spectral lists after truncation to
their shortest length. It does not define the spectrum of an averaged
Hamiltonian. All the theorems below apply frame by frame before those reporting
operations.
:::

(sec-native-gap-exact-spectrum)=
## 2. Exact reduction of the configured fitness-ratio operator

:::{prf:theorem} Exact fitness-ratio spectrum and full ground multiplicity
:label: thm-native-gap-fitness-spectrum

Use {prf:ref}`def-native-gap-parameters`. In the unfloored branch
$b_i\ne0$ and $|b_i b_j|\ge\tau_0$ for every $i\ne j$, put

$$
a_i=b_i^{-1},\qquad
\delta^2=\left(\sum_i b_i^2\right)\left(\sum_i b_i^{-2}\right)-m^2.
$$

Then the *implemented recipe* has the exact factorization

$$
T=ab^{\mathsf T}-ba^{\mathsf T},\qquad
\delta^2=\sum_{i<j}(V_j-V_i)^2
       \left(\frac1{b_i}+\frac1{b_j}\right)^2.
$$

If $\delta>0$, $iT$ has eigenvalues $-\delta,+\delta$ and $m-2$ zeros.
Its singular values are $\delta,\delta$ and $m-2$ zeros. If $\delta=0$,
$T=0$. In the positive-denominator branch $b_i>0$, this last case is exactly
equal alive fitness. With denominators of either sign it is instead the
condition that all $b_i^2$ are equal.

For the existing Fock operator $H=d\Gamma(iT)$, $\delta>0$ gives

$$
E_{\rm sea}=-\delta,\qquad
\dim\ker(H+\delta I)=2^{m-2},\qquad
\operatorname{Spec}(H+\delta I)=\{0,\delta,2\delta\}.
$$

The multiplicities of these three energies on the full Fock space are
$2^{m-2},2^{m-1},2^{m-2}$. For $m\ge3$, each parity has ground multiplicity
$2^{m-3}$ and gap $\delta$. For $m=2$, the filled ground is the unique
odd-parity vector at energy zero and its parity gap is $2\delta$; the even
parity block consists of two vectors of energy $\delta$. For $m=0$ or $1$
the matrix vanishes and there is no excited sector. Every principal
classification submatrix satisfies the same formulas with its own size and
fitness list.
:::

:::{prf:proof}
The denominator is $b_i b_j$ off diagonal and the diagonal numerator is zero.
Thus

$$
T_{ij}=\frac{b_j}{b_i}-\frac{b_i}{b_j}=a_i b_j-b_i a_j.
$$

Every vector perpendicular to $a$ and $b$ is in the kernel. Moreover
$a\cdot b=m$. When $a,b$ are independent, set

$$
p=\frac b{\|b\|},\qquad
w=a-\frac m{\|b\|}p,\qquad q=\frac w{\|w\|}.
$$

These vectors are orthonormal, and
$T=\|b\|\|w\|(qp^{\mathsf T}-pq^{\mathsf T})$.
The coefficient squared is
$\|b\|^2\|a\|^2-m^2=\delta^2$.
On their plane, $Tp=\delta q$ and $Tq=-\delta p$, giving the stated
eigenvalues of $iT$. When $a,b$ are dependent this coefficient is zero and
the matrix is zero. Their dependence is equivalent to $b_i^2$ being constant;
with positive $b_i$ it is equivalent to equal $b_i$.

Expanding the product of squared norms and pairing the terms $i,j$ gives

$$
\delta^2=\sum_{i<j}\left(\frac{b_j}{b_i}-\frac{b_i}{b_j}\right)^2,
$$

which is the displayed fitness-difference formula. Applying
{prf:ref}`thm-lqft-edge-filled-ground-gap` fills the single negative mode,
leaves the positive mode empty, and leaves all $m-2$ zero modes arbitrary.
Removing the negative occupation or adding the positive occupation costs
$\delta$; doing both costs $2\delta$. The multiplicities follow by choosing
zero-mode occupations. For $m\ge3$, half the zero-mode choices have each
parity, and a zero-mode change compensates one nonzero change. When $m=2$
there is no zero mode; listing the four occupations gives the asserted two
parity blocks. Restriction to a principal submatrix simply restricts $a,b$
and repeats the same proof. $\square$
:::

:::{prf:theorem} Primitive fitness ranges and exact dispersion bounds
:label: thm-native-gap-fitness-bounds

In the positive unfloored branch, let $0<l\le b_i\le B<\infty$ and let

$$
\sigma_V^2=\frac1m\sum_i(V_i-\overline V)^2.
$$

For $m\ge2$ the normalized native nonzero rate $\gamma=\delta/m$ satisfies

$$
\boxed{\quad \frac{2\sigma_V}{B}\le\gamma\le\frac{2\sigma_V}{l}.\quad}
$$

These constants come from actual fitness-map parameters. For the existing
Python logistic fitness branch, with its implemented replacement of nonfinite
standardized inputs and clamp to $[-50,50]$, put

$$
u_- =\eta+A\operatorname{sigmoid}(-50),\qquad
u_+ =\eta+A\operatorname{sigmoid}(50).
$$

For its configured positive exponents $\alpha,\beta$,
$u_-^{\alpha+\beta}\le V_i\le u_+^{\alpha+\beta}$ on alive rows.
Consequently $l=u_-^{\alpha+\beta}+\varepsilon$ and
$B=u_+^{\alpha+\beta}+\varepsilon$ give the above certificate whenever
$l>0$ and $l^2\ge\tau_0$. All local/global standardization, donor, landscape,
periodic and detach parameters remain in the actual $V_i$; the range estimate
is independent of them.

For a Rust logistic fitness pipeline with its actual reward/diversity
amplitudes $A_r,A_s$, floors $\eta_r,\eta_s$, and exponents $p_r,p_s\ge0$,
use

$$
v_- =\eta_r^{p_r}\eta_s^{p_s},\qquad
v_+ =(A_r+\eta_r)^{p_r}(A_s+\eta_s)^{p_s},\qquad
l=v_-+\varepsilon,\quad B=v_++\varepsilon.
$$

A disabled channel contributes one, as it does in `FitnessPipeline::combine`;
thus its zero exponent is interpreted without evaluating its map. A zero
positive-map floor can give $v_-=0$. The criterion remains $l>0$ and
$l^2\ge\tau_0$. The unchanged reference has $v_-=0.01$, $v_+=4.41$;
the default Dirac-readout $\varepsilon=0.01$ gives $l=0.02$, $B=4.42$.
The denominator-floor branch is therefore absent for that reference readout,
without a fitness-separation assumption.

For an enabled Rust `LegacyAsymmetric` channel whose actual standardizer
gives $|z|\le Z$, its map is bounded between
$e^{-Z}+\eta$ and $1+\log(1+Z)+\eta$. A global standardizer gives
$Z\le\sqrt{m-1}$ for $m\ge2$ (the singleton gives $z=0$), including its
legacy sample-standard-deviation branch. A local standardizer instead retains
its actual derived $Z$, which can be infinite. Substituting the bounds of each
consumed map and its exponent supplies $v_-,v_+$ when they are finite; no
bounded support is imposed on the physical coordinates or innovations.
:::

:::{prf:proof}
For positive $b_i$, the factor $b_i^{-1}+b_j^{-1}$ lies in $[2/B,2/l]$.
The elementary identity
$\sum_{i<j}(V_i-V_j)^2=m\sum_i(V_i-\overline V)^2=m^2\sigma_V^2$
and {prf:ref}`thm-native-gap-fitness-spectrum` prove both bounds, including
the zero-dispersion case. The Python map evaluates a sigmoid only after the
actual clamp, so monotonicity gives its two endpoints. The two channel powers
are increasing on positive inputs. The Rust logistic map has its stated floor
and amplitude range, and its channel powers give the displayed bounds.
The inequality $b_i b_j\ge l^2\ge\tau_0$ verifies the absence of flooring.

For a global population standardization write $y_i=r_i-\overline r$.
Since $\sum_i y_i=0$, Cauchy--Schwarz gives
$y_i^2\le(m-1)\sum_{j\ne i}y_j^2$ and hence
$y_i^2\le(m-1)m^{-1}\sum_j y_j^2$.
Dividing by the population standard deviation with its positive regularizer
gives $|z_i|\le\sqrt{m-1}$. The sample standard deviation has a larger
unregularized denominator and its additive regularizer only enlarges it.
Finally the legacy map is increasing on each side of zero, continuous at
zero, and has the stated endpoint values. These arguments use the actual
standardization branch and never bound an innovation by a prescribed cutoff.
:::

:::{prf:proposition} Floored denominators and numerical arithmetic
:label: prop-native-gap-arithmetic

For every finite actual fitness list and readout regularizer, the exact
floored matrix of {prf:ref}`def-native-gap-parameters` is real skew. If every
off-diagonal denominator is floored, then

$$
T=\mathbf1 u^{\mathsf T}-u\mathbf1^{\mathsf T},\qquad
u_i=b_i^2/\tau_0,\qquad
\delta=m\operatorname{std}(u).
$$

Thus this arm also has rank at most two and the preceding occupation
formulas apply when $\delta>0$. For $m=2$ in any branch the nonzero pair,
if present, is exactly $\pm|T_{12}|$. For a mixed-floor branch retain its
actual real skew matrix; a rank-two formula is not asserted. Every rate is
bounded above by $\max_i\sum_j|T_{ij}|$.

Let $T_{\rm num}$ be the actually executed finite matrix, and let
$E=\mathfrak h(T_{\rm num})-\mathfrak h(T)$ be its retained arithmetic
residual. If $\epsilon_{\rm op}=\|E\|$ is finite, the ordered eigenvalues
of these Hermitian matrices differ by at most $\epsilon_{\rm op}$.
For an exact rank-two branch, its two nonzero eigenvalues therefore lie
within $\epsilon_{\rm op}$ of $\pm\delta$, and every remaining eigenvalue
lies in $[-\epsilon_{\rm op},\epsilon_{\rm op}]$. This does not give a
positive lower bound on additional small numerical eigenvalues. A numerical
policy for treating them as zero is additional readout data; the current
positive-value generation clustering supplies no such exact-zero policy.
:::

:::{prf:proof}
The denominator is symmetric and the numerator changes sign under exchange.
In the fully floored arm it is $u_j-u_i$. The rank-two proof with
$\mathbf1,u$ gives
$\delta^2=m\sum_i u_i^2-(\sum_i u_i)^2=m^2\operatorname{Var}(u)$.
The $2\times2$ formula follows by diagonalizing its two entries. For a real
skew matrix $\|iT\|\le\sqrt{\|T\|_1\|T\|_\infty}
=\|T\|_\infty$, proving the row-sum bound. The Hermitian variational
formula for its $j$th ordered eigenvalue bounds a perturbation between
$-\|E\|I$ and $\|E\|I$ by those same scalar shifts. Applying this to
the computed Hermitian matrix proves the error bounds. A small eigenvalue
created in an exact zero eigenspace can be arbitrarily close to zero, so this
comparison supplies no positive minimum for that additional pair.
:::

(sec-native-gap-native-dispersion)=
## 3. Dispersion from the actual spatial noise and diversity sampling

:::{prf:definition} Primitive two-ball diversity certificate
:label: def-native-gap-two-ball

Use the real-coordinate viscous Euclidean subfamily in
{prf:ref}`def-native-jg-ledger`, retaining both count and row normalization,
all its configured cloning/collision stages and uncapped innovations. At the
next pre-clone fitness stage retain its actual independent nonself
distance-companion draws, one companion per row, no history pool, and the
global logistic reward/diversity pipeline. This is the unchanged reference
pipeline, with its positive $\eta_r,\eta_s,p_r,p_s,\sigma_r,\sigma_s$.
Different existing tags retain their complete parameters; the certificate
does not transfer to mutual companions, a local standardizer, an enabled
historical pool or a different distance reducer by their names.

Write the actual squashed feature map as
$\chi_R(x)=x/(1+|x|/R)$. Choose two disjoint balls
$A=B(z_A,r_A)$ and $B=B(z_B,r_B)$ whose closures are inside the configured
alive domain $D$, and a velocity-cell diameter $r_v>0$. These balls and cells
restrict events in the proof, not the innovations or state space. A partition
of the configured capped velocity cube $[-V,V]^d$ has

$$
K=\left\lceil\frac{2V\sqrt d}{r_v}\right\rceil^d
$$

cells of diameter at most $r_v$. Set

$$
\begin{aligned}
L_x&=|\chi_{R_x^{\rm feat}}(z_A)-\chi_{R_x^{\rm feat}}(z_B)|-r_A-r_B,\\
D_0^2&=4(R_x^{\rm feat})^2+4\lambda_{\rm alg}(R_v^{\rm feat})^2,\\
d_{\max}&=\sqrt{D_0^2+\delta_D^2},\qquad
d_L=\sqrt{4r_A^2+\lambda_{\rm alg}r_v^2+\delta_D^2},\\
d_H&=\sqrt{L_x^2+\delta_D^2},\qquad
S_s=\sqrt{(d_{\max}-\delta_D)^2/4+\sigma_s^2},\\
M_s&=(d_{\max}-\delta_D)/\sigma_s,\qquad
c_s=A_s\frac{e^{-M_s}}{(1+e^{-M_s})^2},\\
\kappa_D&=\exp[-D_0^2/(2\epsilon_D^2)].
\end{aligned}
$$

Require the evaluated geometric inequality $L_x>0$ and $d_H>d_L$.
At fixed finite $N$ one can replace $M_s$ by
$\min\{M_s,\sqrt{N-1}\}$, retaining its additional population dependence.
Put $r_-=\eta_r,r_+=\eta_r+A_r,s_-=\eta_s,s_+=\eta_s+A_s$ and
evaluate the actual reward oscillation

$$
\Omega_A=\sup_{x,x'\in A,\ |v|,|v'|\le V}
|R(x,v)-R(x',v')|.
$$

An infinite value gives no positive certificate. Define

$$
\begin{aligned}
C_D&=r_-^{p_r}\,p_s\min(s_-^{p_s-1},s_+^{p_s-1})
           \frac{c_s(d_H-d_L)}{S_s},\\
C_R&=s_+^{p_s}\,p_r\max(r_-^{p_r-1},r_+^{p_r-1})
                   \frac{A_r}{4\sigma_r},\\
\Delta_V&=C_D-C_R\Omega_A.
\end{aligned}
$$

The positive regime is the explicit inequality $\Delta_V>0$, together with
the native readout margin $l>0$, $l^2\ge\tau_0$ from
{prf:ref}`thm-native-gap-fitness-bounds`. All quantities are calculated from
existing fields and the configured reward; no lower bound on an unknown
stationary variance is assumed.
:::

:::{prf:lemma} Diversity separation inside one reward ball
:label: lem-native-gap-diversity-separation

In {prf:ref}`def-native-gap-two-ball`, consider alive focal rows in $A$ whose
capped velocities lie in one velocity cell. A companion from that same set
has regularized separation at most $d_L$; a companion in $B$ has separation
at least $d_H$. For any two such focal rows, one with a low and one with a
high companion, their *completed globally standardized fitnesses* satisfy

$$
V_{\rm high}-V_{\rm low}\ge\Delta_V.
$$

This comparison includes the dependence of the diversity mean and standard
deviation on every companion choice in the same record.
:::

:::{prf:proof}
The derivative of $\chi_R$ has radial eigenvalue
$(1+|x|/R)^{-2}$ and transverse eigenvalue $(1+|x|/R)^{-1}$, both at most
one; at zero it is the identity. Thus it is 1-Lipschitz. Same-ball position
differences have feature norm at most $2r_A$, and same-cell velocity
differences have feature norm at most $r_v$. Across the two spatial balls
the position feature distance is at least $L_x$, independent of velocities.
The implemented square root with its $\delta_D$ gives $d_L,d_H$.

Every actual separation belongs to $[\delta_D,d_{\max}]$. Its global mean
belongs to that interval, its regularized standard deviation is at least
$\sigma_s$ and at most $S_s$. The latter uses
$\operatorname{Var}(Y)\le(\sup Y-\inf Y)^2/4$, obtained by taking
expectations of $(Y-\inf Y)(\sup Y-Y)\ge0$ and completing the square in its
mean. Hence all standardized diversity values lie in $[-M_s,M_s]$, and the
two focal standardized values differ by at least $(d_H-d_L)/S_s$.
The logistic diversity map has derivative at least $c_s$ there. This proves
that their mapped diversity values differ by at least
$c_s(d_H-d_L)/S_s$, with their *common, random completed* mean and scale.

The positive power $u\mapsto u^{p_s}$ has derivative at least
$p_s\min(s_-^{p_s-1},s_+^{p_s-1})$ on the map range. The two mapped reward
values, although they also use a random common mean and scale, differ by at
most $A_r\Omega_A/(4\sigma_r)$. Their reward powers therefore differ in
absolute value by at most
$p_r\max(r_-^{p_r-1},r_+^{p_r-1})A_r\Omega_A/(4\sigma_r)$.
Decompose the product difference as the high reward factor times the
diversity difference plus the reward difference times the low diversity
factor. Bounding these two terms by their native ranges proves $\Delta_V$.
No companion was redrawn after the standardization, and no factorization of
the completed fitnesses was used.
:::

:::{prf:theorem} Population-independent positive-probability native gap
:label: thm-native-gap-noise-diversity

Use the positive regime of {prf:ref}`def-native-gap-two-ball` at a fixed
configured $h>0$. At an executed terminal update let $m_i$ be the actual
random preparation means of {prf:ref}`lem-native-jg-spatial-gaussian` and let

$$
\tau^2=(h/2)^2q^2+\sigma_x^2h>0,
\qquad \mathbb E\frac1N\sum_i|m_i|^4\le H_4,
$$

where $q$ and the finite force-derived $H_4$ are exactly those of that
lemma. In particular the configured force is retained in $H_4$.
For a proof radius $M>0$ define

$$
\begin{aligned}
p_A&=|A|(2\pi\tau^2)^{-d/2}
\exp[-(M+|z_A|+r_A)^2/(2\tau^2)],\\
p_B&=|B|(2\pi\tau^2)^{-d/2}
\exp[-(M+|z_B|+r_B)^2/(2\tau^2)],\\
f&=p_A/(4K),\qquad g=p_B/4,\qquad
\ell=\kappa_D f^2/4,\qquad u=\kappa_D fg/2,\\
\Gamma_*&=\frac{2\Delta_V\sqrt{\ell u}}{v_++\varepsilon},\\
\mathcal E_N&=e^{-Np_A/16}+e^{-Np_B/16}
             +e^{-\kappa_D f^2N/16}+e^{-\kappa_D fgN/8}.
\end{aligned}
$$

Here $|A|,|B|$ are their ordinary Euclidean volumes. For every *permitted*
population $N\ge2/f$, the alive fitness-ratio readout at the next actual
fitness measurement satisfies

$$
\mathbb P\!\left(m\ge2,\ \frac{\delta}{m}\ge\Gamma_*\right)
\ge\left[1-\frac{2H_4}{M^4}-\mathcal E_N\right]_+,
$$

and

$$
\mathbb E\left[\left(\frac{\delta}{m}\right)^2
                \mathbf1_{\{m\ge2\}}\right]
\ge \Gamma_*^2\left[1-\frac{2H_4}{M^4}-\mathcal E_N\right]_+.
$$

The constants are independent of $N$ when all displayed parameters, balls
and $M$ are held fixed, and are generally extremely small. For example,
$M^4\ge8H_4$ and $\mathcal E_N\le1/4$ give probability at least $1/2$.
The size test must respect the full execution's allocation, run and termination
limits; if no allowed $N$ passes it, this large-population certificate makes
no claim for that execution family.

The same lower bound holds after conditioning this one transition on
nonextinction, because the proved event implies nonextinction. If a QSD of
the original marked state is already established and used as the input law,
the next state has that QSD after this one-step conditioning; its fresh
fitness measurement therefore has the same certificate. A terminal-horizon
conditioning or another stationary law retains its separate normalization.
The theorem is a positive-probability and moment lower bound for a specified
native spectral readout. It is not an almost-sure lower bound on all histories
or the uniform physical transfer estimate (YM.33).
:::

:::{prf:proof}
Markov's inequality gives probability at least $1-2H_4/M^4$ that the average
fourth moment of the preparation means is at most $M^4/2$. On this event at
least $N/2$ means have norm at most $M$. Conditional on the full preparation,
the actual terminal positions are independent Gaussians with covariance
$\tau^2I$. Each of those core rows hits $A$ with probability at least $p_A$
and hits $B$ with probability at least $p_B$, by integrating the Gaussian
density lower bound over the balls.

For independent Bernoulli indicators with sum $S$ and mean $\mu$,

$$
\mathbb P(S\le\mu/2)
\le e^{\mu/2}\mathbb E e^{-S}
\le\exp\bigl[(1/2-(1-e^{-1}))\mu\bigr]\le e^{-\mu/8}.
$$

Apply this separately to the two core hit counts. Except for probabilities
$e^{-Np_A/16},e^{-Np_B/16}$, the full population has at least $Np_A/4$
rows in $A$ and $Np_B/4$ in $B$. All these rows are alive since the balls
lie inside $D$. The velocity cap partitions the rows in $A$ into the $K$
cells; one cell has a set $A_0$ of size at least $fN$. Choose that cell
measurably from the terminal record, for example by its first maximizing
index. This choice precedes the next independent companion draws.

Every unnormalized soft distance weight is between $\kappa_D$ and one.
The actual nonself probability, for a row in $A_0$, of a donor in
$A_0\setminus\{i\}$ is at least
$\kappa_D(fN-1)/N\ge\kappa_D f/2$; the probability of a donor in $B$
is at least $\kappa_D g$. The independent distance-companion rows therefore
give at least $\ell N$ low-companion focal rows and at least $uN$
high-companion focal rows, except for probabilities
$e^{-\kappa_D f^2N/16}$ and $e^{-\kappa_D fgN/8}$. The two outcome counts
within a row are not independent; applying the Bernoulli bound separately
and a union bound requires no such independence.

For every low/high pair, {prf:ref}`lem-native-gap-diversity-separation` gives
fitness difference at least $\Delta_V$, after all global statistics are
computed. Thus

$$
\sigma_V^2=\frac1{m^2}\sum_{i<j}(V_i-V_j)^2
\ge\frac{\ell uN^2}{m^2}\Delta_V^2\ge\ell u\Delta_V^2,
$$

since $m\le N$. The primitive fitness upper range and
{prf:ref}`thm-native-gap-fitness-bounds` prove $\delta/m\ge\Gamma_*$.
Combining the four failure probabilities with the preparation event proves
the probability bound, and restricting the expectation to that event gives
the moment bound. The conditional-on-survival probability is its unconditioned
probability divided by a number at most one, since the event is a subset of
survival. The QSD assertion uses only its proved defining one-step conditional
invariance; it does not infer independence under that selected law.
:::

:::{prf:corollary} A finite-population certificate at the unchanged reference
:label: cor-native-gap-reference-finite

The unchanged real-coordinate reference
{prf:ref}`def-cgd-existing-reference` with its default fitness-ratio readout
passes a finite-population diversity certificate at $N=200$.
Take $z_A=-1.5e_1$, $z_B=1.5e_1$, $r_A=r_B=10^{-11}$, and $r_v=1.6$.
Then $K=125$, both balls lie in $[-2,2]^3$, and

$$
L_x=12/7-2\cdot10^{-11},\qquad
d_H-d_L>0.114,\qquad
M_s\le\sqrt{199},\qquad
\Delta_V>5\cdot10^{-9}.
$$

Let $M_N^4=4NH_4$ and use the two Gaussian ball lower bounds $p_A,p_B$
with this $M_N$ and these balls. For its next pre-clone measurement,

$$
\mathbb P\!\left(m=N,\frac{\delta}{N}
                         \ge\frac{2\Delta_V}{NB}\right)
\ge\frac34\,p_A^{N-1}p_B\frac{\kappa_D^2}{N^2}>0,
\qquad B=4.42.
$$

For the same native parameter family with arbitrary permitted $N$, taking
the smaller fixed radii $r_A=r_B=10^{-30}$ and the population-independent
$M_s$ instead gives $\Delta_V>10^{-27}$ and the explicit large-population
certificate of {prf:ref}`thm-native-gap-noise-diversity` whenever its size
tests pass. These are statements about the unchanged force, both configured
fitness channels and their actual donor draws; neither channel is removed.
:::

:::{prf:proof}
The reference feature radii are two, so the feature centers are
$\mp(6/7)e_1$ and their distance is $12/7$. The velocity-cube partition has
$\lceil4\sqrt3/1.6\rceil^3=125$ cells. Its diversity upper range is
$d_{\max}=\sqrt{32+10^{-6}}$, and the regularized scale upper bound is
$S_s<2.831$. With $M_s=\sqrt{199}<14.107$,
$c_s>1.49\cdot10^{-6}$. Since $p_r=p_s=1$, $\eta_r=\eta_s=0.1$,
$A_r=A_s=2$, the positive term satisfies
$C_D>0.1(1.49\cdot10^{-6})(0.114)/2.831>5.99\cdot10^{-9}$.
For $R=-|x|^2/2$, $\Omega_A=3r_A=3\cdot10^{-11}$ and $C_R=10.5$.
Consequently $\Delta_V>5\cdot10^{-9}$, as asserted. With the larger
population-independent $M_s<56.56$, one has $c_s>5\cdot10^{-25}$;
the same calculation at radius $10^{-30}$ gives $\Delta_V>10^{-27}$.

Markov's inequality and $M_N^4=4NH_4$ give probability at least $3/4$ that
*every* preparation mean has norm at most $M_N$:
$\mathbb P(\max_i|m_i|>M_N)\le NH_4/M_N^4=1/4$.
Conditional on such a preparation, place the first $N-1$ terminal positions
in $A$ and the last one in $B$. Their conditional independence gives
probability at least $p_A^{N-1}p_B$. Every row is then alive. Since
$N-1=199>K$, two distinct rows in $A$ share a velocity cell. Choose them
measurably before measuring companions. Require the first to choose the
second as its low donor and the second to choose the row in $B$ as its high
donor. Their independent soft donor probabilities give at least
$\kappa_D^2/N^2$. The completed fitnesses have difference at least
$\Delta_V$. One such pair gives $\sigma_V\ge\Delta_V/N$, and the exact
spectral bound proves the displayed event. No Gaussian innovations have
been truncated; only a positive-probability event was selected in the proof.
This real-coordinate certificate has no unproved numerical-law transfer.
:::

(sec-native-gap-phase-space)=
## 4. The actual phase-space spectral branch

:::{prf:theorem} Hermitian spectrum actually computed in the phase-space arm
:label: thm-native-gap-phase-spectrum

Use the complete parameters and stages of
{prf:ref}`def-native-gap-parameters`. Resolve $\epsilon_c$ exactly as the code
does: explicit readout value, otherwise the header's
`companion_selection_clone.epsilon`, otherwise
`companion_selection.epsilon` with its default one, and then floor at
$10^{-12}$. Set $h_S=\max\{\texttt{h_eff},10^{-12}\}$.
The actual periodic minimum-image rule, position and velocity arrays, and
configured $\lambda_{\rm alg}$ give the symmetric squared distances
$d_{ij}^2$ and amplitudes

$$
W_{ij}=e^{-d_{ij}^2/(2\epsilon_c^2)},\qquad W_{ii}=0.
$$

For `include_phase=True`, set

$$
\widehat b_i=\begin{cases}b_i,&|b_i|\ge\tau_0,\\\tau_0,&|b_i|<\tau_0,
\end{cases}\qquad
\theta_{ij}=\frac{V_j-V_i}{\widehat b_i h_S},\qquad
T_{ij}=W_{ij}(e^{i\theta_{ij}}-e^{i\theta_{ji}}).
$$

The helper returns the absolute eigenvalues of the Hermitian matrix

$$
\boxed{\quad\mathfrak h(T)=iR,\qquad
R_{ij}=W_{ij}(\cos\theta_{ij}-\cos\theta_{ji}).\quad}
$$

It thus evaluates the real skew part $R$ and does not, in general, return
the singular values of the complex skew matrix $T$. If
`include_phase=False`, the symmetric amplitude is its entire directed
kernel, so $T=0$ and its entire computed spectrum vanishes.

In the positive unfloored branch write $\Delta=b_j-b_i$. Then

$$
R_{ij}=-2W_{ij}\sin\!\left[\frac{\Delta}{2h_S}
                    \left(\frac1{b_i}+\frac1{b_j}\right)\right]
             \sin\!\left[\frac{\Delta}{2h_S}
                    \left(\frac1{b_i}-\frac1{b_j}\right)\right].
$$

For $b_i,b_j\ge l>0$ this gives the concrete entry bound

$$
|R_{ij}|\le\min\left\{2W_{ij},
               \frac{W_{ij}|V_j-V_i|^3}{l^3h_S^2}\right\}.
$$

For a two-row matrix its nonzero rate is $|R_{12}|$. If a pair has
$|V_j-V_i|\ge\Delta_0>0$, $b_i,b_j\le B$, $d_{ij}^2\le D_*^2$ and

$$
0<\frac{|\Delta|}{2h_S}\left(\frac1{b_i}+\frac1{b_j}\right)\le\pi/2,
\qquad
0<\frac{\Delta^2}{2h_S b_i b_j}\le\pi/2,
$$

then the evaluated pair coefficient satisfies

$$
|R_{ij}|\ge\frac4{\pi^2}
 e^{-D_*^2/(2\epsilon_c^2)}\frac{\Delta_0^3}{h_S^2B^3}.
$$

For a larger matrix this proves a lower bound on its *largest* singular
value, since $\|R\|\ge|R_{ij}|$; it does not bound the smallest nonzero
one when additional pairs are present. The operator $d\Gamma(iR)$ has its
existing exact occupation spectrum. A full physical gap for this branch
retains its actual smallest nonzero spectral pair and its ground sector.
:::

:::{prf:proof}
Write $T=R+iS$ with $R,S$ real skew. Then $iT=iR-S$ and
$(iT)^*=iR+S$, so the implemented Hermitian cleanup is $iR$ exactly.
Taking the real part of its entries gives the cosine formula. With phase
disabled, the amplitude is symmetric even with the actual minimum-image
rule, and subtracting its transpose gives zero.

In the unfloored branch $\theta_{ij}=\Delta/(b_i h_S)$ and
$\theta_{ji}=-\Delta/(b_j h_S)$. The difference-of-cosines identity gives
the two-sine formula. The first sine argument has absolute value at most
$|\Delta|/(l h_S)$, while the second has absolute value
$\Delta^2/(2h_S b_i b_j)\le\Delta^2/(2h_S l^2)$.
Using $|\sin x|\le\min\{1,|x|\}$ proves the entry upper bound.
On $[0,\pi/2]$, concavity gives $\sin x\ge2x/\pi$.
The first positive argument is at least $|\Delta|/(h_SB)$ and the second
at least $\Delta^2/(2h_SB^2)$. Their product, the factor $2W_{ij}$ and
the amplitude lower bound give the displayed lower bound. A matrix norm
dominates the norm of each matrix entry. None of these arguments replaces
the executed complex arm by the fitness-ratio recipe.
:::

(sec-native-gap-convergence)=
## 5. Native spectral convergence and the remaining physical identification

:::{prf:theorem} Quantitative convergence of the native two-mode block
:label: thm-native-gap-active-block

For the positive unfloored fitness branch define its actual empirical
denominator law $\mu_m=m^{-1}\sum_i\delta_{b_i}$ on $[l,B]$, and

$$
\gamma(\mu)=\sqrt{\left(\int b^2\,d\mu\right)
                          \left(\int b^{-2}\,d\mu\right)-1}.
$$

Then $\delta/m=\gamma(\mu_m)$. For any two probability laws on that same
interval,

$$
|\gamma(\mu)^2-\gamma(\nu)^2|
\le C_{l,B}W_1(\mu,\nu),\qquad
C_{l,B}=2B l^{-2}+2B^2l^{-3},
$$

and consequently
$|\gamma(\mu)-\gamma(\nu)|\le\sqrt{C_{l,B}W_1(\mu,\nu)}$.
If both rates are at least $c>0$, the stronger bound is
$C_{l,B}W_1(\mu,\nu)/(2c)$.

For $\delta>0$, the existing finite Fock space factorizes along the
spectral plane and its exact zero space as

$$
\mathcal F_-(E)\simeq\mathcal F_-(E_-\oplus E_+)
                           \otimes\mathcal F_-(E_0).
$$

Under the isometry sending its actual occupation eigenmodes to a fixed
four-vector occupation basis, the ground-shifted, $m$-normalized operator is

$$
\frac{H+\delta I}{m}
\simeq\operatorname{diag}(\gamma,0,2\gamma,\gamma)\otimes I_{\mathcal F_-(E_0)}.
$$

Thus its active-plane semigroup is explicitly continuous in the two native
empirical moments. At fixed $t\ge0$, the difference of its four-dimensional
semigroups for rates $\gamma,\gamma'$ is at most $2t|\gamma-\gamma'|$.
The corresponding active-plane ground is unique when $\gamma>0$; the
original full operator still has its proved zero-mode ground multiplicity.
The plane and this factorization are invariant spectral subspaces of the
existing matrix. No assertion selects that plane as the physical Hilbert
space or removes the zero-mode observables from the native representation.
:::

:::{prf:proof}
The exact spectrum theorem gives the empirical-moment formula.
The functions $b^2,b^{-2}$ have Lipschitz constants $2B,2l^{-3}$ and are
bounded by $B^2,l^{-2}$. For any coupling of $\mu,\nu$, apply these
Lipschitz bounds to the difference of their integrals, and expand the
difference of the two products. Infimizing the coupling cost gives
$C_{l,B}W_1$. The inequalities
$|\sqrt x-\sqrt y|\le\sqrt{|x-y|}$ and
$|\sqrt x-\sqrt y|=|x-y|/(\sqrt x+\sqrt y)$ give the rate bounds.

Order the negative, positive and zero eigenmodes. Mapping an occupation
wedge to the tensor of its first-two and zero-mode occupations is unitary
by the orthonormal occupation basis. Its energies are additive and the
zero modes have zero energy. The four first-two occupations (empty,
negative only, positive only, both) have shifted energies
$\delta,0,2\delta,\delta$. This proves the diagonal factorization.
For $x,y\ge0$, $|e^{-tx}-e^{-ty}|\le t|x-y|$ by the derivative bound,
so the diagonal semigroup comparison is $2t|\gamma-\gamma'|$.
The zero-energy first-two occupation is unique when $\gamma>0$, while
tensoring any zero-mode occupation retains zero energy. This completes
the spectral convergence calculation without a claim of native empirical
law convergence or physical transfer identification.
:::

:::{prf:proposition} Arbitrarily small native fitness rates are actually accessible
:label: prop-native-gap-accessible-small

For the unchanged $N=200$ real-coordinate reference, both its count and row
normalizations admit the following exact conclusion. At its next pre-clone
fitness measurement, after one executed transition or with its already proved
marked QSD as input, for every $\epsilon>0$,

$$
\mathbb P\left(m=N,\quad 0<\frac{\delta}{N}<\epsilon\right)>0.
$$

Thus its actual normalized fitness-ratio nonzero rates have essential
infimum zero, despite the positive-probability and moment certificate above.
This statement concerns the declared fitness-ratio readout on that existing
history law. It does not exclude a physical gap of a separately identified
transfer Hamiltonian or physical sector.
:::

:::{prf:proof}
On a consensus configuration $x_i=x_0\in\operatorname{int}D$,
$v_i=v_0\in B_V$, every nonself diversity companion has the same regularized
distance $\delta_D$, and every reward is equal. The two actual global
standardizations are zero, so every alive fitness is
$(\eta_r+A_r/2)^{p_r}(\eta_s+A_s/2)^{p_s}$ and the matrix is zero.

The finite reference certificate
{prf:ref}`cor-native-gap-reference-finite` supplies a configuration with all
positions in the interior of $D$, all capped velocities in $B_V$, and a
particular admissible nonself companion list for which $\delta/N>0$.
There are finitely many such lists, so at least one list gives that positive
configuration. Fix it. Join this configuration to a consensus configuration
by a straight path in the convex sets $D^N$ and $B_V^N$. The same list is
admissible at every point. Positive statistical regularizers, positive map
floors and the continuous squashed features make its actual fitness vector
and exact rate continuous on that path. Its rate starts at zero and ends
positive. The intermediate value theorem therefore supplies a path point
with rate in $(0,\epsilon)$; continuity supplies a nonempty open coordinate
neighborhood with rates still in that interval.

The configured $q,s>0$ and both positive B2 margins of the unchanged reference
were verified in {prf:ref}`cor-cgd-reference-qsd`. Therefore
{prf:ref}`thm-cgd-phase-smoothing` gives positive probability for that
all-alive open neighborhood from every surviving input. The fixed companion
list also has strictly positive probability under the actual Gaussian soft
donor rows. Their joint probability is positive after an executed transition.
For the marked QSD $\pi_N$, its eigenmeasure identity
$\pi_NQ_N=\rho_N\pi_N$ and $\rho_N>0$ imply positive $\pi_N$ mass of the
same neighborhood: integrate its strictly positive transition probability
against $\pi_N$. The fresh companion measurement again has positive
conditional probability there. This proves the stated access to arbitrarily
small *nonzero* native rates, without a hypothetical matrix or a change to
the algorithm.
:::

:::{prf:proposition} Actual scope of a uniform physical-gap claim
:label: prop-native-gap-physical-register

For the unchanged reference fitness-ratio readout, the following obligations
have been discharged here: its exact rank and spectrum, primitive
denominator margins and range bounds, full and parity ground multiplicities,
finite-population strict dispersion events, a population-independent
positive-probability dispersion certificate in its displayed size regime,
and quantitative spectral/semigroup transfer of actual empirical fitness
errors. These are properties of the existing recorded operator.

The zero modes of this particular operator are actual computed spectral
subspaces. For $m\ge4$, the full filled ground and each parity ground are
degenerate. In the already proved invariant-mode closure $u_g=I$,
{prf:ref}`cor-ym-edge-filled-gauge-closure` preserves that parity multiplicity;
it does not make it a unique vacuum. The code's default reported sectors
have `min_sector_size=10`; whenever such a sector is present and its
fitness-ratio matrix is nonzero, its unrestricted Fock ground has at least
$2^8$ vectors and its parity ground has at least $2^7$ vectors. This is a
calculation for that recorded representation, not a failure theorem about
an independently identified physical sector.

Application of {prf:ref}`thm-mass-gap-rg-fixed-point` still requires the
physical sector and clock of this same operator to be identified, its
unique-vacuum projection to be established in that sector, and an actual
uniform spectral estimate with the required semigroup/Hilbert-space limit.
The probabilistic lower bound above does not supply its all-vector
uniform estimate. A change to the phase-space readout also retains its
distinct actual Hermitian-cleanup spectrum and cannot inherit the rank-two
fitness gap. No missing property has been imposed as a new assumption on
the native algorithm.
:::

:::{prf:proof}
The discharged statements are the preceding formulas and proofs. The ground
multiplicities follow by substituting $z=m-2$ into the exact occupation
counts, or the actual reported sector size into those same counts.
For invariant modes the already proved group implementation is the identity,
so Haar averaging removes none of these vectors. The existing gap-passage
theorem uses a rank-one ground projection and an operator norm estimate
uniform over the declared family. An event on which a randomly evaluated
matrix has a positive rate does not give such an estimate on the complement,
and a degenerate full ground does not yield that rank-one projection.
The required sector, transfer and limit therefore remain exactly the stated
identification and estimate, beyond the native calculations proved here.
:::
