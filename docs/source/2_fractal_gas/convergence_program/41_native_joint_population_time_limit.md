# Joint population and native-time fluctuation limits from the original kernel

(sec-njt-register)=
## 1. Complete algorithm and observation parameters

:::{prf:definition} Original joint-limit register
:label: def-njt-register

Retain all execution, landscape, fitness, donor, cloning, Haar, noise,
cap, status, arithmetic, mask and calibration parameters of
{prf:ref}`def-nfs-register`. The real-coordinate independent continuous
Gaussian count phase is unchanged. Fix those parameters independently
of population $N$, and use its primitive $T_N,\Delta_N$.
Finite arithmetic and different noise streams retain their actual
comparison errors.

A bounded time-homogeneous complete observation
$H_N(S,R,T)\in\mathbb R^m$ has actual bound $B_N$.
Use its EXACT $\theta_N,u_N,U_N,L_N,\Sigma_N$ from
(NFS.2)--(NFS.6), and define
$$
 C_{E,N}=1+\frac{8T_N}{1-\Delta_N}.
 \tag{NJT.1}
$$
A triangular observation schedule chooses only the original update
count $n_N$. It does not change its force, transitions, source law,
recording stride or physical clock $t_*h$.
:::

(sec-njt-polynomial-characteristic)=
## 2. A characteristic bound without an exponential population cost

:::{prf:theorem} Stopped complete-bracket characteristic estimate
:label: thm-njt-characteristic

For every admitted finite $N,n$ and $v\in\mathbb R^m$, put
$\ell=|v|L_N$, $\epsilon=\ell/\sqrt n$, and
$$
 \delta=\ell^2\sqrt{C_{E,N}/n}.
$$
If $\epsilon\le1$, the stationary native complete-record error satisfies
$$
\begin{split}
\left|E\exp\!\left\{\frac{i}{\sqrt n}
 v\cdot\sum_{j=1}^n(H_{N,j}-\theta_N)\right\}
       -e^{-v^\top\Sigma_Nv/2}\right|
\le\min\left\{2,\frac{2|v|U_N}{\sqrt n}
 +9\delta+5\left[
       \frac{\ell^3}{6\sqrt n}+\frac{\ell^4}{8n}\right]\right\}.
\end{split}
\tag{NJT.2}
$$
Every parameter appears through the already derived original budgets.
In particular the estimate has no $e^{|v|^2L_N^2}$ factor.
The full complete covariance, including cross-stage terms, is unchanged.
:::

:::{prf:proof}
Write $Y_j=v\cdot D_{N,j}/\sqrt n$,
$a_j=E[Y_j^2\mid\mathcal T_{j-1}]$,
$A_j=\sum_{l\le j}a_l$, and $b=v^\top\Sigma_Nv$.
Then $|Y_j|\le\epsilon$, $a_j\le\epsilon^2$,
$EA_n=b$, and (NFS.11) applied to
$v^\top A_N(S)v\le\ell^2$ gives
$$
 E|A_n-b|\le\delta.
 \tag{NJT.3}
$$
This uses the actual full-state block inequality.

Stop the martingale after its predictable bracket first exceeds
$R=b+1$. More explicitly retain update $j$ when
$A_{j-1}\le R$, and retain no later update after crossing.
The indicators are predictable. Denote the stopped increments by
$Y'_j$, their bracket by $A'_n$, and their sum by $Z'_n$.
Monotonicity gives $A'_n\le R+\epsilon^2$.
On $A_n\le R$ the original and stopped paths coincide.
By (NJT.3) the crossing probability is at most $\delta$.

The conditional Taylor estimate used in (NFS.10) gives
$$
 \left|e^{a'_j/2}E[e^{iY'_j}\mid\mathcal T_{j-1}]-1\right|
 \le e^{\epsilon^2/2}
 \left[\frac{\ell^3}{6n^{3/2}}+\frac{\ell^4}{8n^2}\right].
$$
The compensated product
$\exp\{i\sum_{l\le j}Y'_l+\frac12\sum_{l\le j}a'_l\}$
has modulus at most $e^{(R+\epsilon^2)/2}$.
Telescoping its conditional expectation therefore proves
$$
 |E e^{iZ'_n+A'_n/2}-1|
 \le e^{R/2+\epsilon^2}
 \left[\frac{\ell^3}{6\sqrt n}+\frac{\ell^4}{8n}\right].
 \tag{NJT.4}
$$
On the noncrossing event, the mean-value bound for
$e^{x/2}$ costs
$e^{(R+\epsilon^2)/2}|A_n-b|/2$.
On the crossing event its difference from $e^{b/2}$ is at most
$2e^{(R+\epsilon^2)/2}$.
Consequently
$$
 |e^{b/2}E e^{iZ'_n}-1|
 \le e^{R/2+\epsilon^2}
 \left[\frac{\ell^3}{6\sqrt n}+\frac{\ell^4}{8n}\right]
 +\frac52 e^{(R+\epsilon^2)/2}\delta .
$$
Divide by $e^{b/2}$. Since $R-b=1$, the factors become
$e^{1/2+\epsilon^2}$ and $e^{(1+\epsilon^2)/2}$,
independently of $b,N$. Removing the stop costs $2\delta$.
For $\epsilon\le1$,
$e^{3/2}<5$ and $2+\frac52e<9$.
Finally the original Poisson endpoint costs
$2|v|U_N/\sqrt n$. This proves (NJT.2).
No independence of stages or assumed limiting covariance was used.
:::

(sec-njt-joint-functional)=
## 3. Complete triangular path limits

:::{prf:theorem} Parameterized joint population/time Brownian cluster limits
:label: thm-njt-joint-functional

Suppose the chosen actual complete observation family has the PROVED
uniform covariance bound
$$
 \sup_N\operatorname{tr}\Sigma_N\le C_\Sigma<\infty,
 \tag{NJT.5}
$$
with $C_\Sigma$ supplied by its native observation certificate.
It is not an extra hypothesis on an unknown stationary law:
Section 4 discharges it for an existing empirical state channel.
For other readouts the theorem applies only after their certificate
has been derived. Choose original time counts $n_N\to\infty$ so that
$$
 \frac{U_N}{\sqrt{n_N}}\to0,\qquad
 \frac{L_N^3}{\sqrt{n_N}}\to0,\qquad
 L_N^2\sqrt{\frac{C_{E,N}}{n_N}}\to0.
 \tag{NJT.6}
$$
For every finite $T>0$, interpolate the original centered record sum
$$
 X_N(t)=\frac1{\sqrt{n_N}}\left[
 \sum_{j\le\lfloor n_Nt\rfloor}(H_{N,j}-\theta_N)
 +(n_Nt-\lfloor n_Nt\rfloor)
       (H_{N,\lfloor n_Nt\rfloor+1}-\theta_N)\right].
 \tag{NJT.7}
$$
Its laws are tight in $C([0,T],\mathbb R^m)$.
Along EVERY subsequence for which the bounded deterministic matrices
$\Sigma_N$ converge to $\Sigma$,
$$
 X_N\Longrightarrow B_\Sigma,\qquad
 E[B_\Sigma(s)B_\Sigma(t)^\top]=\min(s,t)\Sigma.
 \tag{NJT.8}
$$
Every possible path cluster limit is therefore the Brownian motion
of an actual complete covariance cluster matrix. Singular and zero
matrices remain allowed. Existence of covariance cluster subsequences
follows from (NJT.5); uniqueness or a positive uniform lower bound is
not inserted.

The same result holds for the ORIGINAL $\nu_N$-started history
conditioned once on survival through $\lceil n_NT\rceil$.
Its path error from the stationary Doob law is at most the
already proved $d_N\to0$, independently of this horizon.
The physical duration is $n_Nt_*h$, with original covariance rate
$\Sigma_N/(t_*h)$.
:::

:::{prf:proof}
Use the original complete martingale interpolation. Its Poisson endpoint
remainder is bounded by $2U_N/\sqrt{n_N}$, uniformly on the interval.
It vanishes by (NJT.6).
The bracket quadratic form at a fixed $t$, divided by $n_N$,
has centered $L^1$ error at most
$$
 |v|^2L_N^2\sqrt{(T+1)C_{E,N}/n_N}.
$$
Its mean differs from $t\,v^\top\Sigma_Nv$ only by a mesh
error at most $C_\Sigma|v|^2/n_N$.
Thus bracket errors vanish at each finite grid point.
Quadratic-form brackets are increasing. Between grid points their
comparison function $t\,v^\top\Sigma_Nv$ has uniform slope
at most $C_\Sigma|v|^2$. Squeezing between the two endpoints
and then shrinking the grid proves bracket convergence uniformly
in probability along a covariance cluster subsequence.
Polarization gives the full matrix statement.

For finite-dimensional laws apply the stopped argument of Section 2
to deterministic vector coefficients on finitely many time intervals.
The summed Taylor costs are bounded by a fixed coefficient constant
times $L_N^3/\sqrt{n_N}+L_N^4/n_N$.
The variance of each interval bracket average is bounded by the same
block covariance estimate, so its total centered error is at most
a fixed coefficient constant times
$L_N^2\sqrt{C_{E,N}/n_N}$.
The mean quadratic form tends to
$\sum_l(t_l-t_{l-1})v_l^\top\Sigma v_l$.
The same stop at that mean plus one proves convergence of their
joint characteristic function to independent Brownian increments.
This argument also covers an interval endpoint's fractional update,
whose norm is at most $L_N/\sqrt{n_N}\to0$.

Here is an explicit tightness argument, avoiding a fourth-moment
constant that grows with $N$.
For a scalar martingale increment $Y$ with $|Y|\le d$ and
conditional mean zero, Taylor's exponential remainder gives
$$
 E[e^{\vartheta Y}\mid\mathcal T]
 \le\exp\left\{\frac{\vartheta^2e^{|\vartheta|d}}2
                       E[Y^2\mid\mathcal T]\right\}.
 \tag{NJT.9}
$$
Products of these compensated exponentials are nonnegative
supermartingales. Stopping at an excursion and a bracket budget $b$
therefore gives, for every $\vartheta>0$,
$$
 P\left(\max_{j\le k}|M_j-M_0|>a,\
       \sum_{j\le k}E[Y_j^2\mid\mathcal T_{j-1}]\le b\right)
 \le2\exp\{-\vartheta a+
                 \vartheta^2e^{\vartheta d}(b+d^2)/2\}.
 \tag{NJT.10}
$$
The extra $d^2$ pays for the last crossing increment.
The proof is conditional at the interval's initial time, so it
also applies to each deterministic time grid interval.

For our normalized martingale,
$d=L_N/\sqrt{n_N}\to0$.
On the event that its bracket is uniformly within $\delta$ of
$t\Sigma_N$, a grid interval of length $\epsilon$ has scalar
bracket increment at most
$b=C_\Sigma\epsilon+2\delta+o(1)$.
Fix $a,\epsilon>0$, let $N\to\infty$ and choose
$\vartheta=a/[2(C_\Sigma\epsilon+2\delta)]$ when the denominator
is positive. The eventual exponent is at most
$-a^2/[4(C_\Sigma\epsilon+2\delta)]$.
If the denominator is zero, any increasing fixed $\vartheta$
gives a vanishing limit.
Uniform bracket convergence lets $\delta$ tend to zero after
$N\to\infty$. Union over the at most $1+T/\epsilon$
grid intervals and $m$ coordinates bounds the modulus probability
by a constant times
$$
 (1+T/\epsilon)
   \exp\{-a^2/[4C_\Sigma\epsilon]\},
$$
with a harmless change of $a$ for vector coordinates and adjacent
intervals. It tends to zero as $\epsilon\downarrow0$.
Their starting values are zero and their endpoint second moments
are bounded by $(T+1)C_\Sigma$.
This proves tightness for the original martingale interpolation.
The vanished Poisson remainder transfers tightness and the identified
finite-dimensional limit to (NJT.7).

Finally the primitive uniform eigenfunction telescope
{prf:ref}`thm-nue-doob-comparison` compares the complete
Doob and original once-conditioned paths by $d_N\to0$ at EVERY
horizon, with all attached draws retained. It transfers both tightness
and the path limit. No horizon-dependent killing union bound is used.
:::

(sec-njt-evaluated-schedule)=
## 4. An evaluated schedule in the existing active count phase

:::{prf:corollary} Actual empirical state channels have joint native limits
:label: cor-njt-state-schedule

Let $f_N$ be an existing bounded physical-state empirical readout
satisfying the explicit budget of {prf:ref}`def-nfs-lipschitz-channel`,
with its fixed original $B,L$ and no added density hypothesis.
Use $H_N(S,R,T)=\sqrt N f_N(T)$, whose time sums retain the entire
original full marked state process.
The scalar complete covariance has the proved uniform upper
certificate (NFS.16). A finite list of such coordinates has
$C_\Sigma$ equal to the sum of its derived scalar certificates.

For EVERY choice of original run durations satisfying
$$
 n_N/N^9\longrightarrow\infty,
 \tag{NJT.11}
$$
the triangular functional limit (NJT.8) holds along each actual
complete covariance cluster subsequence.
The simple native update schedule $n_N=\lceil N^{10}\rceil$ passes.
This is an evaluated sufficient schedule, not an optimized exponent
or a population Gaussian assumption.

In particular the existing recorded mean capped velocity
$f_N(S)=N^{-1}\sum_i e_a\cdot v_i$ consumes exactly the original stored
velocity array and has $B=V$, $L=1/\omega$.
The finite vector of its $d$ recorded Cartesian components therefore
has this full joint path limit with the sum of the native scalar
covariance certificates. Alive-restricted or hard color averages retain
their different normalization and continuity tests.
:::

:::{prf:proof}
The explicit full-state certificate gives, for all sufficiently
large admitted $N$,
$T_N\le\widehat C_TN$,
$1-\Delta_N\ge5/7$, with every primitive parameter in
$\widehat C_T$.
Hence for $B_N=\sqrt N B$ (or its finite vector norm)
$$
 U_N\le\frac{14}{5}B\widehat C_TN^{3/2},\qquad
 L_N\le\left(2B+\frac{28}{5}B\widehat C_T\right)N^{3/2},
 \quad C_{E,N}\le1+\frac{56}{5}\widehat C_TN.
 \tag{NJT.12}
$$
The finitely many smaller admitted populations keep their own positive
finite-QSD constants and do not affect this joint limit.
Substitution in (NJT.6) bounds its three tests by constants times
$N^{3/2}/\sqrt{n_N}$,
$N^{9/2}/\sqrt{n_N}$, and
$N^{7/2}/\sqrt{n_N}$, respectively.
All vanish under (NJT.11).
The covariance bound is exactly the already proved
{prf:ref}`thm-nfs-uniform-green-kubo`, not a hypothesis to be checked
later for this channel. Apply the full triangular theorem.
For the named native velocity readout, the configured cap gives its
actual bound $V$. Its row-average difference is at most
$\overline{|v-v'|}\le\overline d_\omega(S,S')/\omega$,
so the displayed observation budgets are proved directly.
:::

:::{prf:remark} Complete centering, covariance and remaining gauge scope
:label: rem-njt-centering-scope

The drift in this theorem is the EXACT finite-$N$ stationary
complete-record drift $\theta_N$.
Replacing it by a limiting population mean $\theta_*$ changes
the path by $t\sqrt{n_N}(\theta_N-\theta_*)$ up to one mesh term.
That substitution is valid only when this derived centering error
vanishes. The existing quantitative law-of-large-numbers transport
rate alone does not certify it on the schedule (NJT.11).

For the bounded observation the single raw QSD update mean
$\theta_N^Q$, conditioned on that update's own survival, differs
from the stationary complete Doob mean by at most $2B_Nd_N$.
Thus it can replace $\theta_N$ when
$\sqrt{n_N}B_Nd_N\to0$.
Every fixed polynomial run-duration schedule, including $N^{10}$,
passes this test because $d_N$ decays exponentially in $N$
times a primitive polynomial. This is the actual native QSD
mean of the same declared record, not the limiting mean field.

The polynomial characteristic bound applies to EVERY bounded
original complete-record channel. The uniform covariance premise
for the path result is discharged here for the specified physical
state observations; it is used for matched B2, hard graph or
shared-calibration fields only after their own complete temporal
covariance certificate is derived. Their finite-$N$ functional
limits and the positive actual color-fiber result remain proved.
A positive finite-$N$ color covariance is not assigned a positive
cluster matrix without a primitive uniform lower estimate.
:::
