# Native same-record spatial geometry: parameter-derived finite-horizon estimates

(sec-native-jg-ledger)=
## 1. Execution and observation ledger

:::{prf:definition} Complete data consumed by the spatial estimates
:label: def-native-jg-ledger

Every statement below is indexed by the complete execution record
{prf:ref}`def-native-complete-execution-record`, including the initial law,
arithmetic and innovation convention, every landscape/provider parameter,
recording stage and mask, all nested configuration fields, and physical units.
The positive probabilistic estimates first concern the real-coordinate,
independent Gaussian innovation convention of the existing transition. A fixed
seed or rounded random stream is a different execution convention.

The named positive subfamily is the recorded viscous Euclidean instance
{prf:ref}`def-variant-recorded-color-geometry`, with the parameter tuple

$$
\begin{aligned}
\theta={}&(d=3,N,h,\gamma,b_O,\sigma_x,\sigma_J,V,
 \alpha_{\rm col},R_x^{\rm feat},R_v^{\rm feat},\lambda_{\rm alg},
 \epsilon_D,\epsilon_C,\delta_D,A_r,A_s,\eta_r,\eta_s,p_r,p_s,
 \sigma_r,\sigma_s,s_c,\epsilon_c;U,R,D),\\
\Theta_{\rm CG}={}&(\theta;\nu,\rho,\mathsf n;\vartheta_G,\vartheta_O).
\end{aligned}
$$

The meanings and ranges are those of
{prf:ref}`def-slc-parameter-register` and
{prf:ref}`def-variant-recorded-color-geometry`. In particular the force is
the configured $F=-\nabla U$, while $R$ is its separately configured reward.
Both count and nonself row normalization are included. The reward maps,
global standardizers, donor widths, squashed donor features, acceptance,
simultaneous copying and one rotation per collision component are retained.
No result below requires a reward gap or changes the configured force.

The additional restriction of the full record is explicit: BAOAB; one full
update with revival before kinetics; terminal absorption in the configured
bounded donor domain $D$; no donor-history pool, elite/frozen row, curl,
innovation shift, graph-viscosity stage or consumed geometry feedback;
isotropic independent OU and final position innovations; all-slot passive
geometry at the specified stage. All other fields, including allocation,
backend, precision, invalid-reward, error and termination policies, retain
their values in the full record. A statement about an executed update is
restricted to the event on which those policies permit that update. The initial
law for the first such update has eligible positions in $D$ and entering
velocities bounded by $V$; subsequent completed updates supply the same
velocity bound through the actual cap. The all-dead state contributes no
executed update. These are restrictions of an existing algorithm, not new
premises about an unknown stationary law.

Write

$$
a=h/2,\quad c=e^{-\gamma h},\quad
q^2=b_O^2\begin{cases}
(1-e^{-2\gamma h})/(2\gamma),&\gamma>0,\\ h,&\gamma=0,
\end{cases}\quad s^2=\sigma_x^2h,
\quad R_c=(1+2|\alpha_{\rm col}|)V.
$$

The geometry ledger retains the full `GeometryPipelineConfig`: position-field
key, projection, domain/image-site rule, tessellator, duplicate/rank/failure
policy, metric enum and all its ridge/clamp/policy fields, volume enum and its
determinant floor, every named weight and curvature specification, cell
configuration and parallelism; the actual edge budget and previous-cell data;
and the native success/error, repair and validity marks. The local metric
theorem uses `Projection::Full`, `TessellationDomain::Open`,
`MetricKind::NeighborCovariance` with `RidgeScale::Absolute`, positive ridge
$\varepsilon_G$, `MetricPolicy::Clipped` and configured
$0<g_-\le g_+$. Its pseudo-inverse threshold is $3\epsilon_T$, where
$\epsilon_T$ is the scalar type's machine epsilon; it is not set to zero.
The native determinant density has floor $\delta_{\det}>0$. The row-weight
denominator floor is $10^{-12}$; the inverse-distance squared and additive
floors are $10^{-8}$. The kernel length $\ell_G$ is distinct from the dynamic
viscosity bandwidth $\rho$.

`VolumeKind::SqrtDetMetric` is a determinant density for a unit coordinate
cell. It is distinct from the existing `VolumeKind::RiemannianCell`, which
multiplies that density by the native Voronoi-cell volume. The quadrature
result below applies to the latter enum arm. It does not change an execution
configured with the former arm into a cell quadrature.

The observation tuple additionally retains its color stage and alignment,
color threshold $\delta_c$, mass $m$, reference length $\ell_0$, action unit
$\hbar_{\rm eff}$, speed calibration $c_{\rm phys}$, channel/test functions,
normalizations, codec and precision. Their omission from an upper bound means
that the bound is independent of them, not that they have been erased. Lengths
below are in the recorded geometry coordinates; $h$ uses its recorded time
unit. Conversion by the declared $(\ell_*,t_*)$ of
{prf:ref}`def-cg-mf-parameter-register` changes units only.

Python `EuclideanGas` is a separate tag. Its graph period, `freeze_best`,
cloning/kinetic enable flags and periods, periodic policies, derivative detach
rules, thermostat, substeps, Boris, anisotropic/proxy/supplied-tensor branches,
viscous weights, penalties/caps and volume feedback all remain in the full
record. In `KineticOperator.apply` its A2 position is $x_1+a(c v_1+\Sigma_i
\xi_i)$, and B2 changes velocity only. There is no Rust final independent
position-noise stage in that method. The Python volume readout has its actual
alive/boundary mask, determinant clamps, solver retries and fallback volumes.
No Rust theorem below is transferred to that implementation merely by name.
:::

(sec-native-jg-fresh-positions)=
## 2. Fresh spatial noise retains the actual preparation dependence

:::{prf:lemma} Exact conditional Gaussian spatial law and primitive center moments
:label: lem-native-jg-spatial-gaussian

Fix an executed update of the subfamily in
{prf:ref}`def-native-jg-ledger`, and let $\mathcal F$ contain its entire
post-collision array $(x_i^J,v_i^J)_{i=1}^N$ and preceding record. Compute the
first kick using the configured normalization and force:

$$
v_{1i}=v_i^J+a\left[F(x_i^J)+F_i^{\rm visc}(x^J,v^J)\right],
\qquad m_i=x_i^J+a(1+c)v_{1i}.
$$

At the native B2 input the position is $X_i^{\rm B2}=m_i+a q\xi_i$.
At the final position stage, before terminal classification, it is

$$
X_i^+=m_i+a q\xi_i+s\zeta_i.
$$

Thus, conditional on $\mathcal F$, the corresponding positions are independent
Gaussian vectors with means $m_i$ and scalar covariance $\tau^2I$, where
$\tau^2=a^2q^2$ or $a^2q^2+s^2$, respectively. Their unconditional law is the
mixture over the actual interacting preparations. The terminal alive mark is
$\mathbf 1_D(X_i^+)$; it is retained rather than removed from the law.

For $p\ge1$ set

$$
G_{d,p}=\left[2^{p/2}\frac{\Gamma((d+p)/2)}{\Gamma(d/2)}\right]^{1/p},
\quad B_D=\sup_{y\in D}|y|,
\quad X_{J,p}=B_D+\sigma_JG_{d,p},
$$

and evaluate the configured force profile

$$
\mathcal F_p=
\max\left\{\sup_{y\in D}|F(y)|,
 \sup_{y\in D}\left(\mathbb E|F(y+\sigma_J Z)|^p\right)^{1/p}\right\},
\qquad Z\sim N(0,I_d).
$$

An infinite profile gives an infinite certificate. Otherwise at every executed
update, including conditioning on the complete history before that update,

$$
\mathbb E\frac1N\sum_i|m_i|^p\le H_p,
\qquad
H_p=\left\{X_{J,p}+a(1+c)
 \left[R_c+a(\mathcal F_p+2\nu R_c)\right]\right\}^p.
$$

For a globally Lipschitz configured force with $L_F=\operatorname{Lip}(F)$
and $f_0=|F(0)|$, one may replace $\mathcal F_p$ by
$f_0+L_F X_{J,p}$. For the existing quadratic landscape this is
$\lambda_U X_{J,p}$, without altering $\lambda_U$.
The bound is independent of donor weights, reward and diversity maps, and
$\rho$, because it holds for every possible accepted donor/component outcome
of those actual stages. It retains their effect in the random means $m_i$.
:::

:::{prf:proof}
After the frozen copying/revival stage every retained position is an eligible
input position in $D$, possibly plus its actual fresh Gaussian clone jitter.
The gate and donor choices precede that jitter. Minkowski therefore gives
$\|x^J\|_{L^p(\text{row average})}\le X_{J,p}$ and
$\|F(x^J)\|_{L^p(\text{row average})}\le\mathcal F_p$, independently of the
chosen donor. The component collision is a single shared orthogonal rotation
of centered velocities. The bound on the entering cap gives
$|v_i^J|\le R_c$ for every component member, including a revived row.

The complete Gaussian viscous weights are nonnegative. Count normalization
has row sum at most one; nonself row normalization has row sum one for
$N>1$ and force zero for $N=1$. Consequently
$|F_i^{\rm visc}(x^J,v^J)|\le2\nu R_c$ in both cases, without a lower
degree assumption. Minkowski now gives the displayed bound on $v_1$ and then
on $m$. The same calculation is valid after any past-survival conditioning:
eligible donor sites still belong to $D$, and the current jitter and kinetic
innovations are fresh. Future-survival conditioning is treated separately
below.

The first B and A maps are measurable functions of $\mathcal F$. The O step
gives $v_{2i}=c v_{1i}+q\xi_i$ and the second A step gives
$x_{2i}=x_i^J+a v_{1i}+a v_{2i}=m_i+a q\xi_i$. The complete coupled B2
kick changes $v_2$ and does not change $x_2$. The final independent position
stage adds $s\zeta_i$ and the terminal cap acts only on velocity. Gaussian
addition proves the conditional laws. No independence of the random means,
the collision components, the colors or the completed velocities was used.
:::

:::{prf:theorem} Conditional kernel sampling and local coverage from the native noise
:label: thm-native-jg-spatial-sampling

At either stage of {prf:ref}`lem-native-jg-spatial-gaussian` with $\tau>0$,
define its actual conditional mixture density

$$
\rho_{\mathcal F}(y)=\frac1N\sum_i
(2\pi\tau^2)^{-d/2}e^{-|y-m_i|^2/(2\tau^2)}.
$$

For any deterministic bounded spatial test $\psi$,

$$
\mathbb E\left[\frac1N\sum_i\psi(X_i)\mid\mathcal F\right]
=\int\psi\rho_{\mathcal F},\qquad
\operatorname{Var}\left(\frac1N\sum_i\psi(X_i)\mid\mathcal F\right)
\le\frac{\|\psi\|_\infty^2}{N}.
$$

The same statement holds for $\psi(X_i)\mathbf1_D(X_i)$ at the final
stage. It samples the subprobability density $\mathbf1_D\rho_{\mathcal F}$,
not the normalized selected marginal. In particular, for a bounded kernel
$\kappa$ supported in the unit ball and a Lipschitz test $f$, at a fixed query $x$,

$$
\begin{aligned}
\operatorname{Var}\left[
 \frac1{Nr^{d+2}}\sum_i\kappa((X_i-x)/r)(f(X_i)-f(x))
 \,\middle|\,\mathcal F\right]
&\le\frac{\|\kappa\|_\infty^2\operatorname{Lip}(f)^2}
 {Nr^{2d+2}},\\
\operatorname{Var}\left[
 \frac1{Nr^d}\sum_i\kappa((X_i-x)/r)
 \,\middle|\,\mathcal F\right]
&\le\frac{\|\kappa\|_\infty^2}{Nr^{2d}}.
\end{aligned}
$$

Let $C\subset B(0,R_0)$ be compact and
$\operatorname{fill}_C(X)=\sup_{y\in C}\min_i|y-X_i|$. For any $M,r>0$,

$$
\begin{aligned}
P\{\operatorname{fill}_C(X)>r\}
&\le \frac{2H_p}{M^p}
+\left(1+\frac{4R_0}{r}\right)^d
 \exp\left[-\frac N2v_d(r/2)\,b(M,r,R_0,\tau)\right],\\
b(M,r,R_0,\tau)
&=(2\pi\tau^2)^{-d/2}
 \exp\left[-\frac{(R_0+M+r/2)^2}{2\tau^2}\right],\\
v_d(t)&=\frac{\pi^{d/2}t^d}{\Gamma(1+d/2)}.
\end{aligned}
$$

If the observer uses only final alive rows, the same coverage bound holds
when the covering balls used in the proof belong to the interior of $D$.
For all-slot geometry there is no such restriction. The bound can be summed
over every executed step in a finite observation schedule. Under conditioning
on a future survival event $E$ of positive probability, its unconditioned
error probability is divided by the actual $P(E)$.
:::

:::{prf:proof}
Conditional independence gives the expectation and variance by addition of
the individual Gaussian integrals and variances. On the kernel support,
$|f(X_i)-f(x)|\le r\operatorname{Lip}(f)$, giving the two displayed
variance bounds. These deliberately retain conservative bounds that do not
require an $N$-uniform joint LSI, a density bound, or cloning concentration.

On the event $N^{-1}\sum_i|m_i|^p\le M^p/2$, at least $N/2$ means have
norm at most $M$. Markov bounds its complement by $2H_p/M^p$.
Choose a maximal $r/2$-separated subset of $C$. Its $r/4$ balls are disjoint
and contained in $B(0,R_0+r/4)$, so its cardinality is at most
$(1+4R_0/r)^d$, and maximality gives an $r/2$-net. For each net point $y$,
each core-mean Gaussian has density at least $b$ on $B(y,r/2)$.
Conditional independence gives

$$
P\{X_i\notin B(y,r/2)\text{ for every }i\mid\mathcal F\}
\le(1-v_d(r/2)b)^{N/2}
\le e^{-Nv_d(r/2)b/2}.
$$

The Gaussian lower density integral is a lower probability, hence its product
$v_d(r/2)b$ is at most one. A union bound over the net proves coverage:
every point of $C$ is within $r/2$ of a net point with a sample within
$r/2$. A covering ball inside $D$ automatically supplies an alive sample.
Apply the same estimate conditional on each preceding executed history and
sum its error probabilities to obtain the schedule bound. Finally
$P(A\mid E)\le P(A)/P(E)$ is the exact survival transfer.
:::

:::{prf:corollary} Fixed-reference and joint spatial/time schedules with survival closure
:label: cor-native-jg-coverage-schedule

Fix the complete bounded-donor subfamily of
{prf:ref}`def-native-jg-ledger`, its landscape, a nonextinct initial law
and a finite algorithmic time $T$. Suppose its evaluated $\mathcal F_4$
is finite. The primary reference schedule keeps $h=h_0$ and every other
gas/landscape parameter fixed, with actual terminal $\tau_0>0$, and uses

$$
M_N=(\log N)^{1/8},\qquad r_N=N^{-\beta},\quad 0<\beta<1/d.
$$

The probability that any executed terminal observation before $T$ fails
$r_N$-coverage on a fixed compact set tends to zero. In particular this
applies to the unchanged reference $h_0=0.04$, $d=3$,
$\gamma=b_O=1$, $\sigma_x=\sigma_J=0.1$, $V=2$,
$\alpha_{\rm col}=0.5$, $\nu=0.3$, $\rho=1$, the configured box
$[-2,2]^3$ and $U(x)=|x|^2/2$, with either specified normalization.
At that reference,

$$
q_0^2=(1-e^{-0.08})/2,\qquad
\tau_0^2=0.02^2q_0^2+0.02^2,
$$

and the primitive center budget is the displayed $H_4$ with
$X_{J,4}=2\sqrt3+0.1\,15^{1/4}$ and $R_c=4$.

An optional joint spatial/time schedule uses the same existing parameter
fields, takes $\sigma_x>0$, and sets

$$
h_N=(\log N)^{-1/4},\qquad
M_N=(\log N)^{1/8},\qquad
r_N=N^{-\beta},\qquad 0<\beta<1/d.
$$

Only $N$ sufficiently large that $h_N\le h_0$ is used. The primitive budget
$\sup_{h\le h_0}H_4(h)$ is finite. For every fixed compact $C$, the
unconditioned probability that any executed terminal observation before $T$
fails $r_N$-coverage tends to zero. This is a statement about the actual
spatial outputs, including all unbounded innovations and all retained dead
slots. It is not a statement that the numerical tessellator succeeds in a
full-dimensional branch over this schedule.

If $D$ has nonempty interior, include a fixed nonempty compact ball inside
$D$ in the coverage set. Coverage at every executed terminal stage forces
at least one alive row. Therefore for either displayed terminal schedule
the actual probability $P(E_T)$ of survival through $T$ tends to one.
Survival-conditioned coverage and bounded geometric readout error
probabilities have the same vanishing bounds divided by $P(E_T)=1-o(1)$.

At B2 input, $b_O>0$ gives the coverage conclusion with
$h_N=(\log N)^{-1/8}$ and the same $M_N,r_N$. These stage-specific
bandwidths reflect the actual $h^3$ OU-induced spatial covariance.
:::

:::{prf:proof}
At fixed $h_0$, $H_4$ and $\tau_0$ are fixed. The lower-density penalty
is $O((\log N)^{1/4})=o(\log N)$ and there are finitely many terminal
observations before $T$. Its coverage exponent is
$N^{1-\beta d-o(1)}$, whereas the summed center-moment error is
$O((\log N)^{-1/2})$. Substitution of the actual reference coefficients
gives the displayed formulas.

For the optional shrinking-$h$ schedule, the center moment formula is a
continuous bounded function of $h\le h_0$;
it does not require a stationary or large-time theorem because revival resets
the donor centers into the same bounded $D$ at every executed update.
At the terminal stage,
$\sigma_x^2h\le\tau^2\le C h$ for $h\le h_0$. Thus
$-\log b(M_N,r_N,R_0,\tau_N)=O((\log N)^{1/2}+\log\log N)=o(\log N)$.
The exponent $Nv_d(r_N/2)b/2$ is $N^{1-\beta d-o(1)}$.
The net factor and the number $O(T/h_N)$ of observation times are dominated
by that exponential. The summed moment error is
$O(h_N^{-1}M_N^{-4})=O((\log N)^{-1/4})$.

For B2 input, the actual covariance is $a^2q^2$. For fixed positive $b_O$
and finite $\gamma$, $c_1h^3\le a^2q^2\le c_2h^3$ for sufficiently small
$h$. The logarithmic density penalty is now
$O((\log N)^{5/8})=o(\log N)$ and the summed moment error is
$O((\log N)^{-3/8})$. Nothing in the proof modifies either kinetic kick,
clone frequency, jitter amplitude or landscape. The bound concerns executed
stages; extinct paths do not acquire invented later stages.

Extinction before $T$ has a first executed terminal stage at which all
positions lie outside $D$. That stage cannot cover the chosen ball inside
$D$ once $r_N$ is smaller than its distance to $D^c$. Thus extinction is
contained in the already bounded union of terminal coverage failures and
$P(E_T)\to1$. Divide the failure probabilities by $P(E_T)$ for the
survival-selected statements. This proves survival from the actual output
positions and terminal mark, without assuming a long-horizon survival rate.
:::

(sec-native-jg-density-postprocessing)=
## 3. The existing same-sample density-corrected post-processing choice

:::{prf:theorem} Explicit conditional-density correction without an independent sample or joint LSI
:label: thm-native-jg-density-correction

Use the real-coordinate scalar-noise subfamily of
{prf:ref}`def-native-jg-ledger`, with the actual terminal position stage,
$\sigma_x>0$, finite evaluated $\mathcal F_4$ and a finite time $T$.
Select the existing full kernel-neighborhood post-processing operator
{prf:ref}`def-density-corrected-laplacian` on the same all-slot positions.
This choice consumes the complete kernel neighborhood, rather than only the
recorded Delaunay/Voronoi edges. It is a declared existing readout choice,
not a changed gas transition or a theorem for an execution using only those
sparse edges.

Let $\kappa$ be the nonnegative smooth radial supported kernel of that
definition with $m_0,m_2>0$, and let $f\in C^4$ on a fixed compact
neighborhood. At each actual terminal stage define the same-sample degree
density and its native conditional target by

$$
\widehat\rho_N(y)=\frac1{Nm_0\varepsilon_N^d}
 \sum_j\kappa((X_j-y)/\varepsilon_N),\qquad
\rho_N(y)=\rho_{\mathcal F}(y).
$$

Use the actual diagonal degree, including its self-term, at every site
$X_i$, as in the existing degree reconstruction. Set

$$
h_N=(\log N)^{-1/4},\quad
\varepsilon_N=N^{-1/20},\quad
M_N=(\log N)^{1/8}.
$$

Alternatively keep $h=h_0=0.04$ and every reference gas/landscape
parameter fixed, using the same $\varepsilon_N,M_N$. All conclusions
hold on that primary reference schedule as well. Both schedules have
survival probability through $T$ tending to one by
{prf:ref}`cor-native-jg-coverage-schedule`, so the operator and degree
conclusions hold for the actual survival-selected history law too.

For every fixed compact set $C$, simultaneously over all executed terminal
stages before $T$,

$$
\sup_{y\in C}\left|\frac{\widehat\rho_N(y)}{\rho_N(y)}-1\right|
=o_P(\varepsilon_N).
$$

At any fixed query $x$ whose kernel ball belongs to that neighborhood, define
the existing row-normalized corrected readout

$$
\widehat L_N f(x)=\frac{2m_0}{m_2\varepsilon_N^2}
\frac{\sum_j\kappa((X_j-x)/\varepsilon_N)
 [f(X_j)-f(x)]/\widehat\rho_N(X_j)}
{\sum_j\kappa((X_j-x)/\varepsilon_N)/\widehat\rho_N(X_j)}.
$$

Its denominator is positive with probability tending to one and
$\widehat L_N f(x)\to\Delta f(x)$ in probability, simultaneously at
every finite collection of declared queries and executed times. For an
alive-only terminal instrument the same assertion holds on compact
neighborhoods strictly inside $D$, with the existing alive mask included
in every sum and still using the recorded $N$ normalization. The common
row-normalization cancels any common alive-fraction factor.

The conditional density $\rho_N$ is random. This theorem proves the
density-corrected operator result by cancelling that actual random density;
it does not assert a deterministic limit for the preparation centers,
stationarity, or spatial regularity of the actual color channels.
:::

:::{prf:proof}
On the event $N^{-1}\sum_i|m_i|^4\le M_N^4/2$, at least half the
centers have norm at most $M_N$. The Gaussian lower bound in
{prf:ref}`thm-native-jg-spatial-sampling` therefore gives, on every fixed
compact neighborhood,

$$
\inf\rho_N\ge N^{-o(1)},\qquad
\sup|\partial^\alpha\rho_N|
\le C_{\alpha,d}\tau_N^{-d-|\alpha|},\quad |\alpha|\le4.
$$

For the derivative inequality differentiate a single Gaussian: its derivative
is $\tau^{-d-|\alpha|}$ times a fixed Hermite polynomial times a
Gaussian. The supremum of that polynomial times the Gaussian is finite;
averaging the rows preserves the same explicit constant. These smoothness
and positivity bounds are derived from the native innovations, not posited
for an unknown interacting law. Since $\tau_N^2\asymp h_N$, their upper
bounds grow only as powers of $\log N$.

The conditional expectation of the degree density is convolution with
$\rho_N$. Radial symmetry cancels its first-order Taylor term and gives
$|\mathbb E(\widehat\rho_N\mid\mathcal F)-\rho_N|
\le C\varepsilon_N^2\tau_N^{-d-2}$, uniformly on that compact
neighborhood. Both the empirical density and its expectation have spatial
Lipschitz constants at most $C\varepsilon_N^{-d-1}$.

For completeness, a bounded independent scalar sum has
$P(|N^{-1}\sum(Y_i-\mathbb EY_i)|>u)
\le2\exp(-2Nu^2/B^2)$ when every $Y_i$ has range length at most $B$.
Indeed the second derivative of its log moment-generating function is the
variance under a tilted law, at most $B^2/4$: the variance is bounded by
the mean square distance to the midpoint of its containing interval.
Integrating twice bounds the centered log moment-generating function by
$t^2B^2/8$; independence and exponential Markov, optimized in $t$,
give the inequality. Thus it applies to the nonidentical Gaussian rows
conditional on their complete preparation.

Cover the compact neighborhood by a grid of mesh $N^{-1/3}$ and cardinality
$O(N^{d/3})$. In this application $B=C\varepsilon_N^{-d}$.
Take $u_N=N^{-1/10}$. For $d=3$ the grid deviation probability is bounded
by a polynomial times $\exp(-cN^{1/2})$. Its interpolation error is
$O(N^{-1/3}\varepsilon_N^{-4})=O(N^{-2/15})=o(u_N)$.
Therefore

$$
\sup_C|\widehat\rho_N-\rho_N|
\le O_P(N^{-1/10})+O(\varepsilon_N^2\tau_N^{-d-2}).
$$

Divide by the derived lower density $N^{-o(1)}$; both terms are
$o_P(\varepsilon_N)$. The center-moment exceptional probabilities summed
over $O(T/h_N)$ stages are $O((\log N)^{-1/4})$. The exponentially
small grid errors remain summable over that schedule.

First insert exact $\rho_N$ in the corrected row. Conditional on
$\mathcal F$, it is a known function, so the native independent Gaussian
row integrals cancel that density exactly. The scaled numerator and
denominator expectations are

$$
\frac1{\varepsilon_N^{d+2}}
\int\kappa((y-x)/\varepsilon_N)[f(y)-f(x)]\,dy
=\frac{m_2}{2}\Delta f(x)+O(\varepsilon_N^2),
$$

and $m_0$, respectively. Their conditional variances are at most
$C/(N\varepsilon_N^{2d+2}\inf_C\rho_N^2)$ and
$C/(N\varepsilon_N^{2d}\inf_C\rho_N^2)$.
For $d=3$ these decay as $N^{-3/5+o(1)}$ and
$N^{-7/10+o(1)}$, remaining summable over the stated time schedule.
Consequently the ratio tends to $\Delta f$ and its denominator stays positive.

Finally, a uniform relative density error $\delta_N<1/2$ changes each
reciprocal weight multiplicatively by at most $2\delta_N$. Subtraction
of two normalized nonnegative rows bounds the row-weight difference by
$8\delta_N$. Since the field differences in the kernel ball are at most
$\varepsilon_N\operatorname{Lip}(f)$, the scaled corrected-operator
error is at most $C\delta_N/\varepsilon_N=o_P(1)$. This comparison is
pathwise and permits the degree estimator to use the same points at which
it is evaluated, including self terms. For alive-only sampling, every
kernel ball used in the argument is strictly inside $D$, so inserting its
mask changes none of the local integrals. No new independent evaluation
sample, full-law LSI or center-chaos assumption was used.
:::

(sec-native-jg-geometry)=
## 4. Native protected cells, covariance metrics and retessellation

:::{prf:lemma} Coverage protects Voronoi cells and their incident Delaunay edges
:label: lem-native-jg-protected-cell

Let $K\subset\mathbb R^d$ be compact and let
$K^\delta=\{x:\operatorname{dist}(x,K)\le\delta\}$. A finite site set
has fill distance at most $r$ on $K^\delta$, where $6r<\delta$.
Use its full Euclidean open Voronoi cells and the dual Delaunay graph, including
their actual deterministic tie convention. Every site in $K^{4r}$ has its
Voronoi cell contained in $B(x_i,2r)$. In particular the cell is bounded,
and every incident graph edge has length at most $4r$. Every cell intersecting
$K$ has its site in $K^r$ and has the same protection.

The statement is deterministic. It uses no independence, separation estimate,
simplex angle bound or shape regularity. When a returned native graph is full
rank and is the Euclidean Delaunay graph of those sites, it applies to that
graph directly. A rank-projected, failure or graph-budget branch is retained
as its actual distinct outcome.
:::

:::{prf:proof}
For $x_i\in K^{4r}$ and $|y-x_i|=2r$, the point $y$ belongs to
$K^\delta$. Some site lies within $r$ of $y$ and is strictly closer to $y$
than $x_i$. Thus no such $y$ belongs to the Voronoi cell of $x_i$.
The cell is convex and contains $x_i$; a point farther than $2r$ in that
cell would put the intervening radius-$2r$ point in the cell, a contradiction.
At a shared facet a point $z$ has
$|x_i-z|=|x_j-z|\le2r$, so $|x_j-x_i|\le4r$.
If a cell intersects $K$ at $y$, its site is no farther from $y$ than the
nearest sample, hence within $r$ of $y$. Apply the preceding argument.
:::

:::{prf:theorem} Actual absolute-ridge metric and curvature on a protected neighborhood
:label: thm-native-jg-absolute-metric

Use the native absolute-ridge clipped covariance metric in
{prf:ref}`def-native-jg-ledger` on a full-dimensional successful graph branch.
On the coverage event of {prf:ref}`lem-native-jg-protected-cell`, set

$$
g_* =\operatorname{clip}_{[g_-,g_+]}(\varepsilon_G^{-1}),\qquad
E_g(r)=\frac{16r^2}{\varepsilon_G^2}.
$$

If the primitive numerical threshold test

$$
3\epsilon_T(\varepsilon_G+16r^2)<\varepsilon_G
$$

holds, then for every site in $K^{4r}$,

$$
\|g_i-g_*I\|_{\rm op}\le E_g(r).
$$

For the native determinant density
$d_i=\sqrt{\max\{\det g_i,\delta_{\det}\}}$ put
$d_* =\sqrt{\max\{g_*^d,\delta_{\det}\}}$ and

$$
C_v=\frac{d\,g_+^{d-1}}{2\sqrt{\delta_{\det}}}.
$$

Then $|d_i-d_*|\le C_vE_g(r)$. For $d=3$ and the actual conformal
Laplacian readout with nonnegative subunit-normalized row weights,

$$
\sup_{x_i\in K}|R_i^{\rm obs}|\le\frac{4E_g(r)}{g_-}.
$$

There are also exact, finite-parameter flat branches. If
$\varepsilon_G+16r^2\le1/g_+$, then the protected metrics equal $g_+I$;
if $\varepsilon_G\ge1/g_-$ they equal $g_-I$. If $g_-=g_+$, that
equality holds for every successful metric evaluation, including repaired
pseudo-inverse directions. The displayed curvature is then exactly zero
when the row and its neighbors belong to that same saturated region.

Consequently the fixed absolute-ridge observation has a locally constant
metric regime. It does not recover an arbitrary nonconstant metric by
increasing sample density at fixed ridge. `RidgeScale::RelativeToTrace`,
`HessianFd`, `ObservationField`, `Strict` policy and metric feedback are
different configured regimes and are not conclusions of this theorem.
:::

:::{prf:proof}
Every incident displacement has norm at most $4r$, so its average covariance
$C_i$ satisfies $0\preceq C_i\preceq16r^2I$, independently of its degree.
Every eigenvalue of $C_i+\varepsilon_GI$ lies in
$[\varepsilon_G,\varepsilon_G+16r^2]$. The displayed threshold test
therefore proves that none is replaced by zero by the actual $d\epsilon_T$
pseudo-inverse rule ($d=3$). Its reciprocal lies between
$(\varepsilon_G+16r^2)^{-1}$ and $\varepsilon_G^{-1}$.
Scalar clipping is 1-Lipschitz. Comparing those eigenvalues to
$\operatorname{clip}(\varepsilon_G^{-1})$ proves

$$
\|g_i-g_*I\|_{\rm op}
\le\frac{16r^2}{\varepsilon_G(\varepsilon_G+16r^2)}
\le E_g(r).
$$

Products of the $d$ eigenvalues give
$|\det g_i-g_*^d|\le d g_+^{d-1}E_g(r)$. The floor is 1-Lipschitz
and square root is $1/(2\sqrt{\delta_{\det}})$-Lipschitz after that
floor, proving the density bound. The function
$u_i=(2d)^{-1}\log\max(\det g_i,\delta_{\det})$ differs from its
constant value by at most $E_g(r)/(2g_-)$: interpolate the eigenvalues
between $g_*I$ and $g_i$, bound the logarithmic derivative by $1/g_-$,
and use that the logarithmic floor is a contraction on logarithmic values.
All neighbors of a site in $K$ belong to $K^{4r}$. Hence
$|u_j-u_i|\le E_g(r)/g_-$ and
$|-4\sum_jw^R_{ij}(u_j-u_i)|\le4E_g(r)/g_-$.

For upper saturation, every reciprocal eigenvalue is at least $g_+$.
For lower saturation, every reciprocal is at most $g_-$. Clipping makes
the metric exactly the corresponding scalar identity. Equal clamps give
the asserted identity even if the pseudo-inverse repaired a direction.
The native conformal differences then vanish. These are deductions from
the existing positive ridge and clamp values; no Gaussian sample is capped.
:::

:::{prf:lemma} Same-graph native weight errors retain all metric-field correlations
:label: lem-native-jg-row-weights

On a protected graph compare two metrics bounded below by $g_-I$ with
$\max_i\|\widehat g_i-g_i\|_{\rm op}\le e_g$, retaining the same sites,
edges, determinant floor and row-normalization floor. For native inverse
Riemannian distance weights,

$$
\sum_j|\widehat w^R_{ij}-w^R_{ij}|
\le\min\{2,e_g/g_-\}.
$$

For native Riemannian kernel determinant-density weights of length $\ell_G$,

$$
\sum_j|\widehat w^V_{ij}-w^V_{ij}|
\le\min\left\{2,\exp\left[
2e_g\left(\frac{8r^2}{\ell_G^2}+\frac{d}{2g_-}\right)\right]-1\right\}.
$$

For cell-volume kernel weights these bounds hold when the cell volumes are
retained. Any change of cells is charged separately. Thus an existing row
readout $\sum_jw_{ij}B_{ij}$, $|B_{ij}|\le M$, changes by at most $M$
times the appropriate row bound when only its metric is reconstructed.
For changed fields add $\max_j|\widehat B_{ij}-B_{ij}|$.
No independence of $B$, metric or sites is involved.
:::

:::{prf:proof}
The metric comparison gives
$(1+e_g/g_-)^{-1}d_{g,ij}^2\le d_{\widehat g,ij}^2
\le(1+e_g/g_-)d_{g,ij}^2$. The positive squared floor and additive
distance floor preserve the corresponding square-root ratio bound.
The inverse raw weights therefore have ratios in $[q^{-1},q]$ with
$q=(1+e_g/g_-)^{1/2}$. Their row sums have the same ratio bounds,
as do their positively floored row denominators. The normalized weight
ratios lie in $[q^{-2},q^2]$. Summing against the original subunit row
gives $q^2-1=e_g/g_-$; both rows have mass at most one, giving also 2.

For kernel weights, $|\widehat d_{ij}^2-d_{ij}^2|\le16r^2e_g$.
The logarithmic determinant density changes by at most $d e_g/(2g_-)$,
also after its floor. Thus the logarithm of each raw-weight ratio has
absolute value at most $e_g(8r^2/\ell_G^2+d/(2g_-))$. Row normalization
with its actual floor doubles this bound and gives the exponential estimate.
Finally expand the readout difference into a field difference and a weight
difference, using subunit row masses. The calculation is pathwise.
:::

:::{prf:theorem} Native local cell quadrature and a topology-independent retessellation bound
:label: thm-native-jg-cell-quadrature

Use the existing `VolumeKind::RiemannianCell` with its stated determinant
floor and exact native cell-volume formulas, on the successful full-dimensional
branch of {prf:ref}`thm-native-jg-absolute-metric`. Let $f$ be Lipschitz with
compact support $K$, and let $V_i$ be its Euclidean Voronoi-cell volume.
The actual filled-volume recipe replaces unbounded/empty volumes by their
bounded mean. Define the recorded sum

$$
Q_X(f)=\sum_i\operatorname{vol}_i f(X_i),
\qquad Q_*(f)=d_*\int_{\mathbb R^d}f(x)\,dx.
$$

On $r$-coverage of $K^\delta$, $6r<\delta$, with the preceding metric
threshold test,

$$
|Q_X(f)-Q_*(f)|
\le |K^{3r}|\left[
 2d_*r\operatorname{Lip}(f)+C_vE_g(r)\|f\|_\infty\right].
$$

The filled fallback cells outside the protected region do not contribute:
their site values are zero. When $\delta_{\det}\le g_*^d$, the target
$Q_*$ is precisely the volume integral of the constant native limiting
metric $g_*I$. When $\delta_{\det}>g_*^d$, it is the algorithm's floored
density integral, which has a different constant normalization.

For two configurations $X,\widehat X$ evaluated with the same absolute ridge,
clamps and determinant floor, each satisfying its own protected-cell test,
possibly with entirely different Delaunay graphs and no site matching,

$$
|Q_X(f)-Q_{\widehat X}(f)|\le E_Q(r)+E_Q(\widehat r),
$$

where $E_Q$ is the preceding explicit bound. This is an actual
retessellation estimate for the local cell-weighted readout. It requires
neither a cell-shape bound nor stability of each individual cell volume.

For finite-arithmetic payloads retain, in addition, the exact discrepancy

$$
E_{\rm payload}(f)
=\sum_i|\operatorname{vol}^{\rm returned}_i-V_i d_i|\,|f(X_i)|.
$$

The finite-arithmetic inequality adds $E_{\rm payload}$, including all
native unbounded/failed-cell substitutions and arithmetic errors. A native
error carrying no payload is an error outcome, not a zero quadrature.
:::

:::{prf:proof}
Cells whose sites belong to $K$ are protected and have their exact positive
bounded cell volumes. Thus their filled-volume recipe uses $V_i$, not its
fallback. The cells partition Euclidean space up to bisectors of zero
Lebesgue measure. Every cell meeting $K$, and every cell whose site has
nonzero $f$, has its site in $K^r$ and belongs to $K^{3r}$ by
{prf:ref}`lem-native-jg-protected-cell`. For every point of these cells,
$|f(X_i)-f(y)|\le2r\operatorname{Lip}(f)$. Integrating gives

$$
\left|\sum_i V_i f(X_i)-\int f\right|
\le2r\operatorname{Lip}(f)|K^{3r}|.
$$

The determinant-density error is bounded by
$C_vE_g(r)\|f\|_\infty$ times the sum of the protected contributing cell
volumes; their disjoint union is contained in $K^{3r}$. Combining these
two inequalities proves the bound. Apply it twice to the same $Q_*(f)$
to obtain the retessellation result, regardless of changes in the graph.
Direct subtraction of the returned payload from the exact same-site
cell-density sum gives $E_{\rm payload}$. Nothing in this argument asserts
that a statistical curvature estimator or a local gauge readout is smooth.
:::

(sec-native-jg-numeric-branches)=
## 5. Primitive rank tests and the actual numerical branches

:::{prf:lemma} Finite-population sufficient test for the native affine-rank threshold
:label: lem-native-jg-rank-certificate

Use $N\ge d+1$ Gaussian slots of
{prf:ref}`lem-native-jg-spatial-gaussian`, with distinct sites under the
real-coordinate innovation convention. The native Rust affine-rank rule uses
$\varepsilon_{\rm rank}=\texttt{f64::EPSILON}=2^{-52}$ and threshold
$\sigma_1\max(N,d)\varepsilon_{\rm rank}$ on the centered $N\times d$ site
matrix, independently of the covariance metric's separate $3\epsilon_T$ rule.
Here $\sigma_1$ is its largest singular value. Let $M>0$ and
$B=M+2\tau\sqrt d$. In exact evaluation of that retained threshold rule,
if

$$
\max(N,d)\varepsilon_{\rm rank}<\frac{\tau}{4B},
$$

the probability that the rule does not return full rank is bounded by

$$
\frac{H_p}{M^p}
+\left(1+\frac{8B}{\tau}\right)^d
 e^{N/8}2^{-(N-1)/2}
+e^{-Nd}2^{Nd/2},\qquad p\ge2.
$$

The bound is primitive and finite-population. It does not assume full rank
of the interacting preparation. The separate actual numerical SVD error,
tessellator/solver outcome and edge budget remain part of the execution
record; the lemma does not certify those errors as zero. The edge budget
$B_{\rm edge}\ge N(N-1)$ is a sufficient existing-parameter bound on the
number of directed open all-slot edges when no duplicate image sites are used.
:::

:::{prf:proof}
The event $N^{-1}\sum|m_i|^p\le M^p$ has complement probability at most
$H_p/M^p$. On that event the operator norm of the centered mean matrix
is at most $M\sqrt N$ by Jensen. If $\Xi$ is the $N\times d$ standard
Gaussian matrix, exponential Markov with exponent $1/4$ gives

$$
P\{\|\Xi\|_F^2>4Nd\}
\le e^{-Nd}\mathbb E e^{\|\Xi\|_F^2/4}
=e^{-Nd}2^{Nd/2}.
$$

Consequently the centered site matrix $A$ has
$\|A\|_{\rm op}\le B\sqrt N$. For any fixed unit vector $u$,
$Au$ is a deterministic shifted Gaussian in the $(N-1)$-dimensional
mean-zero subspace with scalar covariance $\tau^2$. Completing the square
in its Gaussian integral gives

$$
\mathbb E e^{-\|Au\|^2/(2\tau^2)}
\le2^{-(N-1)/2}.
$$

Therefore $P\{\|Au\|\le\tau\sqrt N/2\}\le
e^{N/8}2^{-(N-1)/2}$, uniformly in the deterministic centers.
A $\tau/(4B)$-net of the unit sphere has at most
$(1+8B/\tau)^d$ elements, by the disjoint-ball packing argument.
If every net vector has norm above $\tau\sqrt N/2$, the operator bound
extends this to $\|Au\|\ge\tau\sqrt N/4$ for every unit $u$.
Thus $\sigma_d/\sigma_1\ge\tau/(4B)$ and the primitive threshold
inequality retains all $d$ singular directions. Union bounds give the
displayed probability. Distinctness and exact affine general position of
finite Gaussian coordinates hold because their joint density is continuous
and the corresponding nonzero polynomial equalities are Lebesgue-null;
the native numerical threshold still requires the inequality just proved.
The budget bound follows because there are at most $N(N-1)$ distinct
directed pairs. Actual solver and predicate errors require their own
comparison and are not covered by Gaussian absolute continuity.
:::

:::{prf:proposition} Exact branch distinctions and finite-precision scope
:label: prop-native-jg-branch-scope

The spatial and moment estimates above have the following parameter regimes.

1. If $q=s=0$, the conditional position law is a point mass at its actual
   preparation, and the Gaussian coverage certificate is zero. Included
   resting configurations with coincident initial sites, zero clone jitter
   and zero force remain coincident; a full-dimensional geometric claim fails
   for those configurations.
2. A passive all-slot observer and an alive-only observer are distinct. Even
   with nondegenerate Gaussian spatial noise, a finite terminal alive-only
   record can contain fewer than $d+1$ sites with positive probability when
   $D$ is bounded. General position of all slots does not remove this event.
3. A configured `FailurePolicy::EmptyGraph` gives its empty graph and failure
   flag. With absolute ridge its isolated-site covariance is zero, its
   clipped metric is $g_*I$, and its conformal Laplacian is zero. Those
   identities do not certify correct cells or a differential graph operator.
4. The affine-rank rule's exact threshold requires
   $\sigma_d/\sigma_1>\max(n_{\rm sites},d)2^{-52}$ for full rank.
   If $n_{\rm sites}2^{-52}\ge1$, it cannot retain any singular direction.
   This is an algebraic scope statement about that rule, not an executed
   counterexample outside the validated population/resource range. In
   particular `RunConfig` limits the configured walker count; finite-
   precision/resource execution is not identified with an unbounded
   population limit by this draft.
5. The real-coordinate coverage schedule proves coverage of the actual
   analytic transition. Convergence of returned full-rank geometric payloads
   on it additionally requires a numerical implementation comparison. The
   finite rank certificate supplies a concrete preasymptotic test; it does
   not discard or replace the native threshold, nor infer its success at
   all $N$ from continuous Gaussian draws.
:::

:::{prf:proof}
For Item 1 substitute the zero amplitudes in the exact spatial formula.
At coincident resting sites with constant zero force the two kicks and drifts
are zero, copying identical sites leaves them identical, and zero jitter/noise
changes nothing. For Item 2 condition on any finite preparation. Each row's
nondegenerate Gaussian has positive mass outside bounded $D$, and the product
law gives positive probability that every row, or all but at most $d$, lands
outside. For Item 3 insert its actual empty row in the configured covariance
and curvature formulas. For Item 4 all singular values are at most $\sigma_1$
and the test is strict. Item 5 is the distinction between the proved law of
spatial coordinates, the preceding threshold estimate, and actual returned
numerical payloads. No result changes a numerical implementation to force
the desired geometric branch.
:::

(sec-native-jg-action)=
## 6. Native source action stability and differential-operator limits

:::{prf:theorem} Joint native source actions and all finite source derivatives obey the same-record error
:label: thm-native-jg-source-action

Fix the actual full recorded history law $P$ from the complete parameter
record. Let $\Phi=(\Phi_1,\ldots,\Phi_m)$ be its selected bounded native
field/geometry readouts and $\widehat\Phi$ their same-history reconstructions,
with $|\Phi_i|,|\widehat\Phi_i|\le M_i$ and
$|\widehat\Phi_i-\Phi_i|\le e_i$. The errors may be the metric/weight
budgets above or those of {prf:ref}`thm-ym-same-record-metric-field`, including
all reconstruction, mask, retessellation and branch errors. No independent
sample is introduced. Define

$$
Z(J)=\mathbb E_P e^{J\cdot\Phi},\qquad
W(J)=-\log Z(J),
$$

and their hatted versions. For $S(J)=\sum_i|J_i|M_i$,

$$
|\widehat W(J)-W(J)|
\le e^{2S(J)}\sum_i|J_i|\mathbb E_Pe_i.
$$

For a multi-index $\alpha$, let
$M^\alpha=\prod_iM_i^{\alpha_i}$, and interpret products with a zero
exponent as one. Its unnormalized source derivatives obey

$$
\begin{aligned}
|\partial^\alpha\widehat Z-\partial^\alpha Z|
\le e^{S(J)}\bigg[
M^\alpha\sum_i|J_i|\mathbb E_Pe_i
+\sum_{i:\alpha_i>0}\alpha_i
 M_i^{\alpha_i-1}\prod_{j\ne i}M_j^{\alpha_j}\mathbb E_Pe_i
\bigg].
\end{aligned}
$$

All derivatives of the source action consequently converge locally uniformly
when $\mathbb E_Pe_i\to0$, retaining the actual geometry/field/time
correlations. If centered fields are multiplied by $a_N$, their reconstruction
error is bounded by $2a_N\|e_i\|_{L^p(P)}$, exactly as in the existing
same-record theorem. A bound $e_i=O(r_N)$ therefore cannot be used at
$\sqrt N$ scale without its additional rate.

This controls the native source-generating action. It does not identify the
descriptor density action $-\log\mathbb E_R[\mathcal L\mid Y=y]$ with a
local Yang--Mills functional or prove convergence of its geometric first
variation.
:::

:::{prf:proof}
The exponential mean-value theorem and the bounded range give
$|e^{J\cdot\widehat\Phi}-e^{J\cdot\Phi}|
\le e^{S(J)}\sum_i|J_i|e_i$. Also
$e^{-S(J)}\le Z,\widehat Z\le e^{S(J)}$.
The logarithmic mean-value theorem gives the first bound.
Differentiation under the expectation is permitted by boundedness and yields
$\partial^\alpha Z=\mathbb E_P[\Phi^\alpha e^{J\cdot\Phi}]$.
Replace one factor in the monomial at a time and then replace the exponential;
the monomial telescoping bounds give exactly the second display, including
zero $M_i$ by interpreting the individual products directly. Repeated
differentiation of $\log Z$ expresses its derivatives as finite polynomials
of the displayed derivatives divided by positive powers of $Z$. On every
compact source set, $Z\ge e^{-S}$ is uniform, so the same error controls
all source derivatives with constants depending only on that set, $\alpha$
and $M$. Centering uses
$\|\Delta-\mathbb E\Delta\|_p\le2\|\Delta\|_p$ and retains $a_N$.
Every expectation used the same $P$; no factorization was used.
:::

:::{prf:proposition} What the actual sparse graph operator still requires
:label: prop-native-jg-operator-remainder

On a protected row with the native nonnegative subunit weights, define its
actual graph difference

$$
(D_N f)_i=\sum_jw_{ij}[f(X_j)-f(X_i)].
$$

For Lipschitz $f$, $|(D_Nf)_i|\le4r\operatorname{Lip}(f)$.
For $f\in C^3$ on the protected region and any declared differential
normalization $b_N\ge0$, define the actual row moments

$$
T_i=b_N\sum_jw_{ij}(X_j-X_i),\qquad
Q_i=\frac{b_N}{2}\sum_jw_{ij}(X_j-X_i)(X_j-X_i)^{\mathsf T}.
$$

Then, with $M_3=\sup\|D^3f\|$ on the row's segments,

$$
b_N(D_Nf)_i
=\nabla f(X_i)\cdot T_i+D^2f(X_i):Q_i+\mathcal R_i,
\qquad |\mathcal R_i|\le\frac{64}{6}b_NM_3r^3.
$$

The native row-normalized conformal Laplacian has no inserted $r^{-2}$
factor. Its flat limit in the absolute-ridge regime is proved above. A
nontrivial second-order graph limit instead requires control of the actual
first and second row moments at its declared differential normalization.
Coverage alone supplies neither cancellation of $T_i$ nor identification
of $Q_i$. The density-corrected full kernel-neighborhood operator of
{prf:ref}`prop-density-corrected-limit` is an explicitly different
post-processing operator. The conditional kernel-sampling estimate above
does not silently add its omitted kernel-ball edges or density corrections
to the recorded Delaunay/Voronoi graph.
:::

:::{prf:proof}
Use the actual edge bound $|X_j-X_i|\le4r$ and row mass at most one
for the Lipschitz inequality. Taylor's formula on each edge with its
third-derivative integral remainder gives
$f(X_j)-f(X_i)=\nabla f(X_i)\cdot\Delta x+
\tfrac12D^2f(X_i):\Delta x\Delta x^{\mathsf T}+R_{ij}$,
where $|R_{ij}|\le M_3|\Delta x|^3/6$. Sum with the existing weights
and multiply by $b_N$. There is no identity forcing the weighted first
moment to vanish on an arbitrary Delaunay row. The theorem neither changes
these weights nor infers the differential normalization from coverage.
:::

(sec-native-jg-residual)=
## 7. Exact remaining obligations and source audit

:::{prf:remark} Discharge and remaining regime register
:label: rem-native-jg-residual

The newly proved positive results are the stage-native conditional spatial
law, kernel conditional variance, finite-horizon local coverage with a joint
$N,h,r$ schedule, protected-cell/edge estimates, native absolute-ridge flat
metric and exact saturation regimes, same-graph native weight perturbations,
local existing-cell-volume quadrature and retessellation, a finite primitive
rank certificate, same-sample conditional-density-corrected post-processing,
and complete bounded source-action derivative stability.
They preserve the actual unbounded noise and every preparation correlation.

The following points remain distinct and are not declared discharged.

| Remaining target | Exact missing estimate or identification |
|---|---|
| Returned geometry along an unbounded analytic population schedule | Comparison of actual rank/SVD/predicate/solver/failure marks and arithmetic with the protected exact cell/graph computation, inside the execution's validated resource and parameter family |
| Nonconstant emergent metric | Derivation for the configured relative-ridge, Hessian, supplied-field or feedback branch; fixed absolute ridge has the proved scalar-identity regime |
| Density-corrected native sparse graph differential operator | Actual first/second row moment limits, omitted-edge error and density-estimator rate after its declared singular normalization; no complete-graph replacement |
| Gauge-weighted cell integrals | Spatial regularity or a directly derived weak integration estimate for the actual stage-native color/loop readouts; the Lipschitz scalar-test proof does not assert that those colors are smooth |
| Global normalized volume readout and unbounded tails | Accumulated mass of actual filled unbounded/error cells and normalization on the same law; local test support alone does not certify the global denominator |
| Full stationary concentration | Variance of the random preparation density and its long-time limit; the conditional kinetic variance does not erase cloning fluctuations |
| Fluctuation-scale geometry and joint action | $a_N\|e_i\|_p\to0$ with actual normalization and error law, and the native density/score first variation rather than only source-parameter derivatives |
| Stationary/QSD observation schedules beyond the finite-horizon result | The actual selected-law normalization and long-time estimates; the proved finite-horizon survival closure is not an unbounded-horizon QSD history theorem |
| Physical-time continuum dynamics | The complete update's generator/drift/bracket limit, including unchanged cloning frequency/jitter; shrinking $h$ in the spatial coverage estimate does not supply it |

The source audit consumed Chapters 04 (variants), 18 and 19 (coupled kinetic
and population estimates), the same-record metric/covariance theorems in
Chapter 05 (Yang--Mills), and the conditional kernel/density-corrected graph
statements in Chapter 03 (lattice QFT). Executable sources inspected are the
Rust Euclidean/viscous constructors and observer, `kinetic.rs`, and native
tessellation `pipeline.rs`, `metric.rs`, `volume.rs`, `voronoi.rs`,
`degenerate.rs`, `linalg.rs`, `weights.rs`, `curvature.rs`; and Python
`euclidean_gas.py`, `kinetic_operator.py`, `voronoi_observables.py`,
`geometry/weights.py`, `geometry/hessian_estimation.py` and neighbor helpers.
All arguments here are elementary finite-dimensional inequalities, Gaussian
integrals, packing, Taylor expansion and conditional expectation. No external
regularity, concentration, shape-regularity or independence theorem was
invoked with unverified hypotheses.
:::
