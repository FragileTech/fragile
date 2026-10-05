# Default-box survival normalization and the remaining law estimate

This record audits the original harmonic reference regime after the completed
small-viscosity population and particle proofs. It proves the missing positive
alive-count floor at the default box radius without importing the large-box
margin. The result concerns the actual finite update, including both viscous
kicks, mandatory revival, recipient jitter, and the terminal mark. It does not
complete the nonlinear population-law estimate at viscosity $0.3$.

The review follows the structural-exhaustion-audit protocol: mathematical
correctness, applicability to the original transition, and closure of the
remaining population-law claim are recorded separately. The completed proofs
in research records 19, 21--26 retain their stated scopes.

(sec-dsa-register)=
## 1. The actual bounded first velocity average

:::{div} feynman-prose
The large-box proof made every sufficiently small noise draw safe. That
particular event fails in the default box: a particle at the boundary can
move outward before the terminal decision. We can choose a different event.
Ask the Gaussian innovation to move the particle far enough inward that
every allowed first velocity average still lands inside the box.

That event may be rare, but its probability is a fixed positive number. It
gives an alive-count floor independent of the number of slots. Its role is
to control normalization and recent survival conditioning; it supplies no
population mixing rate by itself.
:::

:::{prf:definition} Harmonic default-box survival record
:label: def-dsa-survival-record

Use the actual canonical current-frame simultaneous preparation with Gaussian
recipient jitter of amplitude $\sigma_J>0$ and original frozen-slot component
Haar collision. Let the stored speed cap be $V>0$, restitution be
$\alpha_{\rm col}$, and terminal box be $D_L=[-L,L]^d$, $L>0$.
An entering finite array has $N\ge2$, a nonempty alive pool, and consistent
marks $a_i=\mathbf1_{D_L}(x_i)$. Retained dead coordinates are unrestricted.
All original slot velocities, alive and dead, obey $|v_i|\le V$.
The actual measurement/fitness/donor primitives are retained. The argument
requires no weak-fitness-power or realized-variance hypothesis.

Use $F(x)=-x$, $t=h/2>0$, $c=e^{-\gamma h}$, and
$$
b=t(1+c),\qquad a_x=1-tb\in(0,1),\qquad
\tau^2=t^2q^2+s^2>0,\qquad
V_c=(1+2|\alpha_{\rm col}|)V.
\tag{DSA.1}
$$
Both original viscous normalizations are allowed: eligible-count or
row-normalized Gaussian weights, with the exact self-exclusion convention
in the latter. Assume $0\le t\nu\le1$ and
$$
r_L=L-bV_c>0.
\tag{DSA.2}
$$
The row denominator is its actual sum over the other prepared slots. The
preparation has revived every row before this sum is used. The second kick
uses its actual joint OU-position/velocity provider and then caps the stored
velocity. It does not change the landing position.

Let $\Phi(u)=(2\pi)^{-1/2}\int_{-\infty}^u e^{-z^2/2}\,dz$, and set
$$
\sigma_0=\tau,\qquad
\sigma_1=(\tau^2+a_x^2\sigma_J^2)^{1/2},\qquad
\ell_*=\min_{j=0,1}
\left[\Phi\left(\frac{r_L-a_xL}{\sigma_j}\right)
      -\Phi\left(\frac{-r_L-a_xL}{\sigma_j}\right)\right],
\quad a_{\rm ret}=\ell_*^d\in(0,1).
\tag{DSA.3}
$$
The symbol $a_{\rm ret}$ denotes the safe-return probability. The cloning
acceptance ceiling remains a separate primitive.
:::

:::{prf:lemma} The first velocity average stays bounded in both actual normalizations
:label: lem-dsa-first-average

For either bit in {prf:ref}`def-dsa-survival-record`, write the first
pre-force velocity as $U_i=v_i^{\rm p}+t\mathcal V_i$, where
$v_i^{\rm p}$ is its actual component-collision velocity. Then
$|U_i|\le V_c$ pathwise. The exact terminal-position identity is
$$
x_i^+=a_xx_{{\rm src},i}+bU_i+G_i,\qquad
G_i=a_xI_iJ_i+tq\xi_i+s\zeta_i.
\tag{DSA.4}
$$
The frozen source lies in $D_L$; $I_i=1$ for a copied or revived row and
$I_i=0$ for a persistent row. Conditional on the entire source/acceptance
plan and Haar collision marks before recipient jitters, the $G_i$ are
independent over rows and have covariance $\sigma_{I_i}^2I_d$.
The first average $U_i$ may depend on all recipient jitters. Its bound is
pathwise and is not an independence statement.
:::

:::{prf:proof}
The component mean has norm at most $V$, and each original velocity differs
from that mean by at most $2V$. The actual Haar formula therefore gives
$|v_i^{\rm p}|\le V_c$, including all retained dead velocities.
For count weights $K_{ij}\in[0,1]$ put
$a_i=N^{-1}\sum_{j\ne i}K_{ij}\le1$ and
$m_i=N^{-1}\sum_{j\ne i}K_{ij}v_j^{\rm p}$. Then
$$
U_i=(1-t\nu a_i)v_i^{\rm p}+t\nu m_i.
$$
This is a convex combination of the prepared velocities because
$t\nu a_i\le1$. Its self term is the exact zero numerator contribution.
For the actual row bit, $\bar v_i=\sum_{j\ne i}K_{ij}v_j^{\rm p}/
\sum_{j\ne i}K_{ij}$ has norm at most $V_c$, and
$U_i=(1-t\nu)v_i^{\rm p}+t\nu\bar v_i$ is again convex.
The Gaussian weights are positive, so the row denominator is nonzero.

Write the prepared position as $X_i=x_{{\rm src},i}+I_iJ_i$.
The harmonic first kick is $U_i-tX_i$. The two position drifts give
$y_i=X_i+b(U_i-tX_i)+tq\xi_i=a_xX_i+bU_i+tq\xi_i$.
The second kick changes velocity only; final position noise adds
$s\zeta_i$. This proves (DSA.4). Every persistent source is alive and
every copying/revival donor is alive in the frozen entering pool, so its
source is in the box. The source plan, indicators and component Haar marks
are sampled before the independent recipient jitters and kinetic innovations.
Their conditional independent Gaussian sums have the stated two covariances.
:::

(sec-dsa-safe-events)=
## 2. Independent inward-return events

:::{prf:theorem} Positive alive-count floor at a fixed box
:label: thm-dsa-default-box-alive-floor

Use {prf:ref}`def-dsa-survival-record` and $a_{\rm ret}$ from (DSA.3).
For every actual surviving entering array,
the terminal alive count $A_N^+=\sum_i a_i^+$ stochastically dominates
$\operatorname{Bin}(N,a_{\rm ret})$. In particular
$$
\Pr(A_N^+=0\mid S)\le e_N=(1-a_{\rm ret})^N,
$$
$$
\Pr(A_N^+/N<a_{\rm ret}/2\mid S)
\le\delta_N=
 \exp[-(1-\log2)a_{\rm ret}N/2].
\tag{DSA.5}
$$
These statements are uniform over entering alive fractions and retained
dead coordinates. The exact marked population map has output alive mass
at least $a_{\rm ret}$ under the same primitives and either normalization.
:::

:::{prf:proof}
Condition as in {prf:ref}`lem-dsa-first-average` and define independent
row events
$$
E_i=\{|a_xx_{{\rm src},i}+G_i|_\infty\le r_L\}.
\tag{DSA.6}
$$
On $E_i$, for every allowed value of the globally dependent $U_i$,
$|x_i^+|_\infty\le r_L+bV_c=L$. Thus $E_i$ implies the actual terminal
alive mark. No conditioning on the realized viscous fields is introduced.

For a centered one-dimensional Gaussian of variance $\sigma^2$, the
probability $f_\sigma(u)=\Pr(|u+\sigma Z|\le r_L)$ is even in $u$ and
nonincreasing for $u\ge0$. Indeed
$$
f'_\sigma(u)=\sigma^{-1}
 [\varphi((r_L+u)/\sigma)-\varphi((r_L-u)/\sigma)]\le0,
$$
where $\varphi$ is the standard Gaussian density and
$|r_L-u|\le r_L+u$. Each frozen source coordinate has
$|a_xx_{{\rm src},i,j}|\le a_xL$. Coordinate independence and the two
possible variances therefore give $\Pr(E_i\mid\text{frozen plan})
\ge\ell_*^d=a_{\rm ret}$. This is positive because each interval in
(DSA.3) has length $2r_L/\sigma_j>0$ and positive Gaussian density.
It is less than one because its interval is finite.

Independent indicators with success probabilities at least $a_{\rm ret}$
dominate independent Bernoulli indicators with that common parameter: use
independent uniforms for the elementary threshold coupling. Since the
actual alive count is at least $\sum_i\mathbf1_{E_i}$ pointwise, the
domination follows conditionally and then unconditionally. Extinction has
the binomial bound in (DSA.5). For the second bound, use the exponential
Markov inequality at $u=\log2$:
$$
\Pr(B<a_{\rm ret}N/2)
\le e^{u a_{\rm ret}N/2}
    (1-a_{\rm ret}+a_{\rm ret}e^{-u})^N
\le e^{-(1-\log2)a_{\rm ret}N/2}.
$$
The population-root version uses the same frozen source and independent
own noises; its probability of $E$ is at least $a_{\rm ret}$. Integrating
its actual rooted preparation law proves the population alive-mass floor.
:::

:::{prf:corollary} Exact certification at the original harmonic preset
:label: cor-dsa-original-reference

In the original harmonic reference profile,
$$
d=3,\quad h=1/25,\quad\gamma=b_O=1,\quad
\sigma_x=\sigma_J=1/10,\quad V=2,\quad
\alpha_{\rm col}=1/2,\quad L=2,\quad\nu=3/10,
$$
both count and row bits satisfy the preceding alive-count theorem with the
strictly positive exact Gaussian-integral floor (DSA.3). Specifically
$$
c=e^{-1/25},\quad t\nu=3/500<1,\quad
b=(1+c)/50,\quad a_x=1-(1+c)/2500,\quad V_c=4,
$$
$$
q^2=(1-e^{-2/25})/2,\quad s^2=1/2500,\quad
r_L=2-2(1+c)/25>46/25>0,
$$
$$
\sigma_0^2=q^2/2500+1/2500,\qquad
\sigma_1^2=\sigma_0^2+a_x^2/100.
\tag{DSA.7}
$$
The large-box small-noise margin is instead exactly
$$
\Delta_L=(1-a_x)L-bV_c
                 =-99(1+c)/1250<0.
\tag{DSA.8}
$$
Thus the positive floor here is a different, explicitly proved inward-return
event. No decimal tail evaluation or numerical rounding is a proof input.
:::

:::{prf:proof}
Here $0<c<1$, so $0<a_x<1$ and $r_L>2-4/25=46/25$.
All three Gaussian amplitudes are positive. The exact product
$t\nu=3/500$ gives the convexity required by both first averages.
Substitution into (DSA.3) therefore gives a positive finite interval integral
for each of the two variances. Direct substitution also gives (DSA.8).
The acceptance, donor and standardizer data remain the configured original
ones; none enters the source-box and velocity-convexity calculation.
:::

(sec-dsa-current-conditioning)=
## 3. Current survival, moments, and recent-window tilt

:::{prf:corollary} Uniform current conditional moments and alive coverage
:label: cor-dsa-current-conditional-control

Let $S_n^N$ be the actual chain stopped at
$\tau_N=\inf\{n:A_N=0\}$, with an arbitrary initial law on consistent
positive-alive capped states. Put
$$
G_8=d(d+2)(d+4)(d+6),\quad
M_{8,\rm box}=2^7[(a_x\sqrt dL+bV_c)^8+\sigma_1^8G_8],
$$
$$
c_{\rm ret}=[1-(1-a_{\rm ret})^2]^{-1}<\infty.
\tag{DSA.9}
$$
For every $N\ge2$ and observation time $n\ge1$,
$$
\mathbb E[M_8(L_N(S_n^N))\mid\tau_N>n]
\le\frac{M_{8,\rm box}}{1-e_N}
\le c_{\rm ret}M_{8,\rm box},
$$
$$
\Pr(A_N/N<a_{\rm ret}/2\mid\tau_N>n)
\le\min\{1,\delta_N/(1-e_N)\}
\le\min\{1,c_{\rm ret}\delta_N\},
$$
$$
\mathbb E[A_N/N\mid\tau_N>n]\ge a_{\rm ret}.
\tag{DSA.10}
$$
No entering dead-position moment, independence of the current array, or
bound on the survival probability over the whole history is assumed.
:::

:::{prf:proof}
From (DSA.4), $|a_xx_{{\rm src},i}+bU_i|\le a_x\sqrt dL+bV_c$
pathwise. Each conditional Gaussian $G_i$ has eighth moment at most
$\sigma_1^8G_8$. Apply $(A+B)^8\le2^7(A^8+B^8)$ and average rows
to obtain $\mathbb E[M_8(L_N(S^+))\mid S]\le M_{8,\rm box}$ for
every surviving array. This does not require independence from $U_i$.
The unconditioned next alive fraction has expectation at least
$a_{\rm ret}$ by the binomial domination.

Condition on survival up to the previous update. Its entering array law
is arbitrary and supported on surviving arrays, so each bound still
holds. The next survival probability is at least $1-e_N$. Divide the
moment and low-alive numerators by this one-step denominator. Since
$N\ge2$, its reciprocal is at most $c_{\rm ret}$. The extinct event has
alive count zero; removing that event and renormalizing can only increase
the conditional expected alive fraction. This proves all of (DSA.10).
The argument also covers $n=1$ and arbitrarily large original retained
dead coordinates, which are replaced before the first force kick.
:::

:::{prf:corollary} Exact recent-window survival normalization at the default box
:label: cor-dsa-recent-window-normalization

Fix $k\ge0$ and $m\ge0$. Start an ordinary stopped continuation from
$\eta_k=\operatorname{Law}(S_k^N\mid\tau_N>k)$ and call its path
probability $\mathbb P_k$. The actual continuation conditional on
$\tau_N>k+m$ has the exact Radon--Nikodym derivative
$$
\frac{\mathbf1_{\{\text{window survives}\}}}
           {\mathbb P_k(\text{window survives})},\qquad
\mathbb P_k(\text{window survives})\ge(1-e_N)^m.
\tag{DSA.11}
$$
Consequently any nonnegative path observable $Z$ satisfies
$$
\mathbb E[Z\mid\tau_N>k+m]
\le(1-e_N)^{-m}\mathbb E_k Z.
\tag{DSA.12}
$$
With TV defined as $\sup_B$, the two path laws have TV distance
$1-\mathbb P_k(\text{window survives})\le1-(1-e_N)^m\le me_N$.
The same upper bound holds for their endpoint laws and their starting-array
laws. The latter comparison explicitly includes the future-survival tilt
$$
\eta_k^{\rm future}(dS)=
 \frac{q_m(S)}{\eta_k q_m}\eta_k(dS),\qquad
q_m(S)=\Pr_S(\tau_N>m)\ge(1-e_N)^m.
\tag{DSA.13}
$$
For any recent lengths $m_N$ with $m_Ne_N\to0$,
$(1-e_N)^{-m_N}\to1$. This permits a growing recent window at the
default radius; it introduces no whole-history extinction charge.
:::

:::{prf:proof}
By the Markov property, the path law conditional on the longer surviving
history is the restriction of $\mathbb P_k$ to the window-survival event,
divided by that event's mixture probability. Each surviving starting array
has survival probability at least $(1-e_N)^m$ by iterating (DSA.5).
This proves (DSA.11)--(DSA.12). Restricting a probability to an event of
probability $p$ and renormalizing gives path TV exactly $1-p$; projection
cannot increase it. Projection to the starting coordinate gives (DSA.13).
The elementary Bernoulli inequality gives $1-(1-e_N)^m\le me_N$.
Finally $-\log(1-e_N)\le e_N/(1-e_N)$ and $e_N\to0$ exponentially
in $N$, so $m_Ne_N\to0$ proves the final statement.
:::

(sec-dsa-residual)=
## 4. What this closes and what still needs proof

The original count-viscous transition at $L=2,\nu=0.3$ now has proved
current alive coverage, uniform conditional moment budgets, and an exact
recent-window survival comparison. A failed large-box margin is therefore
not a survival-normalization obstruction. The same three conclusions hold
for the actual row bit under $t\nu\le1$.

| Interface | Status in this record | Remaining requirement |
|---|---|---|
| Source position and first average | Proved for the original count and row bits | $F=-x$, capped original slots, $t\nu\le1$, nonempty alive input |
| Default-box alive fraction and extinction | Explicit Gaussian-integral and binomial bounds | None beyond the displayed register |
| Current conditional eighth moment | Uniform in $N,n$ after one actual update | No initial dead-position moment needed |
| Recent survival tilt | Exact own marginal/path change of measure | Choose a recent window; retain its factor |
| Default count population-law attraction | Not supplied by the alive floor | A proved complete marked population block estimate at $\nu=0.3$ |
| Default row population-law attraction | Not supplied by the alive floor | Actual first and second normalized-provider feedback, including uncapped OU velocities |
| Exact finite-$N$ QSD | Separate finite-chain theorem | Its own primitive minorization/eigenfunction hypotheses and finite-$N$ rate |
| Population-uniform alive particle attraction | Completed for count and row in their proved small-positive-viscosity, fixed-large-box regimes | At the default parameters, a complete population-law estimate is still required; the recent survival tilt is already proved |

For the count bit, the finite local weak estimates in
{prf:ref}`lem-spt-marked-consistency` can use the newly proved alive
floor $m_*=a_{\rm ret}/2$, the actual bound on reward over the fixed box,
and its actual positive standardizer/gate parameters. Their preparation
and forward kinetic arguments need no small viscosity: the first and
second count force comparisons have finite coefficients at $\nu=0.3$.
One must replace the weak-selection source moment envelope there by the
general finite coefficient
$a_{\rm clone}/\kappa_C+(1-m_*)/(\kappa_Cm_*)$.
Population forecasts instead obtain their post-first-update eighth-moment
bound directly from the source box, rather than from a small-dead-fraction
drift coefficient. These observations explain why a proved default count
population block would be usable in a recent-window transfer. They do not
assume or establish that block.

The row bit has the same survival and moment normalization. The actual
joint-provider denominator and uncapped posterior velocities are now
controlled by {prf:ref}`thm-rwm-population-modulus` and
{prf:ref}`thm-rft-conditional-consistency`. Their recent-window assembly
proves {prf:ref}`thm-rft-uniform-surviving-law` in the independently proved
small-positive-viscosity, fixed-large-box population regime. These row
estimates use Gaussian tails and local denominators; positive alive mass
alone does not supply a global denominator floor. The default nonlinear
population block remains the missing input at $\nu=.3,L=2$.

### Independent review records

The following independent reviews bind their conclusions to the saved
source revisions. A local analytic estimate, its applicability to the actual
kernel, and the population or particle endpoint are separate audit questions.

#### Row-normalized marked population result

The review of `28_row_population_feedback.md` passes at SHA-256
`fb04549e1cf762109942a6b2b7f6a503674970cb662ec83cc20ea1c69fc26c3e`.
Its exact target is the full marked population map at one fixed sufficiently
large box and its displayed strictly positive, sufficiently small fitness
powers and row viscosity. It is not the default $L=2$, $\nu=0.3$ transition.

The review checks the following complete chain of deductions:

- Fresh final Gaussian noise gives a positive population dead mass
  $p_0(L)$, and mandatory revival supplies an independent recipient-Gaussian
  component of at least that mass. The claim holds for the cross laws
  $\zeta J_\mu$ needed when a common root law is frozen.
- The compact/Gaussian posterior comparison bounds the first row field's
  derivatives globally. Its copied-mass premise prevents the distant
  mixture crossover from invalidating the first global inverse.
- Integrating the own OU Gaussian first yields the actual second-provider
  denominator with broadened variance $v_2>\Sigma$. Posterior velocity
  moments remain polynomial in the query. No uncapped speed bound or product
  replacement for the correlated stage law enters the proof.
- The velocity-fibre BV and OU scores are integrated against explicit
  radial Gaussian envelopes. Their quadratic exponential coefficients are
  below $1/2$, so both provider-feedback constants are finite. The root
  Gaussian budget and fixed environment derivative budgets also cover the
  cross laws in the nonlinear decomposition.
- The second row map has proper degree one under its displayed post-box
  viscosity restrictions. Its bounded preimage and area-formula density
  floor give the preliminary minorization independently of the eventual
  box radius. Choosing the Harris margin first and the row viscosity after
  fixing the box avoids a circular endpoint choice.

Thus {prf:ref}`thm-rpf-marked-population` closes the marked population and
current-alive law endpoint it states. The local denominator, transport and
self-exclusion lemmas (RPF.18)--(RPF.22) also pass. They retain uncapped
second velocity moments and are deterministic on their localization event.
They do not themselves provide the finite empirical comparison or the
uniform-time finite-particle restart. In particular the population copied
mass is not a pathwise empirical mass or denominator floor.

#### Source-box particle-transfer sharpening

The independent review of `30_sharp_survivor_transfer.md` passes at SHA-256
`1bde8000faa0607827f5ea03291cb9a8f61d3fb202cf03f225f6b718da10c4e5`.
Its source bound covers persistence, accepted copying and mandatory revival,
so every prepared source is in $D_L$. Full Gaussian jitter still remains
unbounded. The resulting positional and phase eighth-moment constants are
$X_L^8$ and $Z_L^8$, deterministic for the population preparation and
conditional expected bounds for the empirical preparation. No entering
retained-dead moment is needed.

The bounded-test preparation variance and weak preparation modulus use
bounded configured features and alive-only box reward bounds. Their prefix
proofs therefore retain their constants without an input moment condition.
The kinetic comparison uses the deterministic target moment $X_L^8$; the
random prepared moment occurs only in a concave $9/32$ factor, so conditional
integration supplies exactly the displayed consistency constant.

The single recent-window event then requires only the alive-count floor.
Dropping its future part before conditional consistency, followed by
subprobability Jensen, gives the stated recurrence. The exact recent
survival tilt includes the starting-array bias. Its accumulated error,
the exponential alive-count tail and the proved population attraction all
vanish under the displayed restart sequence. This proves
{prf:ref}`thm-sbst-sharp-survivor-transfer` and its surviving alive-law
consequences in the already completed large-box count regimes. Its separate
population theorem remains an explicit input; the sharpening supplies no
default population contraction.

#### Reference count viscosity

The independent review of `27_reference_count_block.md` passes for its
complete source-box draft, SHA-256
`66469cc6600f481df6b0d402bf3860aa3c572f2568f49b32594ba6af602a8e32`.
The subsequent terminology and evidence-record revision, SHA-256
`1a474c8fd079a65bacc02093272a1784e2035ffb3804f8159cecefceac1b1711`,
preserves those mathematical statements and proofs.

The deterministic Laplacian trace calculation and the pointwise Gaussian
Hessian remainder retain the dependence of the noisy second graph on its
own OU array. Eliminating the entering position gives the exact stage
velocity decomposition. Its bounded, position and Gaussian terms produce
the proved defect coefficient below $0.805$. The terminating rational
inequalities and both signed comparison-field coercivity constants check.
The artificial comparison drift is explicitly separate from the actual
second kick.

The full kinetic energy calculation also checks: the modified harmonic
energy has the exact OU loss, both actual count alignments remain on the
dissipative side, and the cap loss is subtracted exactly. The spatial
Gaussian pair bound supplies its finite remainder without factoring the
noisy graph and velocities. Its moment drift and Feller occupation argument
give finite-swarm invariant existence, with the stated individual moment
budgets after permutation symmetrization. They assert no uniqueness or
mixing rate for that interacting finite kernel.

The conservative frozen population kernel has the proved weighted-Harris
gap at $\nu=0.3$. Its actual second map is proper of degree one because
$m-t\nu>0$, and its absolute provider velocity moment supplies the preimage
and Jacobian bounds. The default-box frozen theorem goes further: every
source outcome begins in the alive box, and a positive good-jitter event
supplies uniform root minorization over both alive and dead entering roots.
It retains both count kicks, the native cap and terminal mark. Thus fixed
frozen providers, or a common prescribed sequence of them, have the exact
Doeblin contraction and invariant conclusion stated in (RCB.14)--(RCB.15).

Each passing endpoint compares root laws with the same providers. Neither
it nor the signed-field constraints completes an active nonlinear comparison
in which each law recomputes its own preparation and both viscous providers.
The remaining cap-compatible complete discrete block is correctly named as
open. The cap counterexample invalidates only the proposed signed-metric
nonexpansion step; it makes no claim against optimal law transport. No
unproved population gap is consumed by the survival results in Sections
1--3 of this record.

#### Row-normalized weak population modulus

The independent review of Sections 1--3 of
`32_row_weak_modulus_restart.md` passes at SHA-256
`4d997d9e23083a33f497595075a64e35fc4fe52856384ade4855a54567134841`.
The explicit prescribed-coupling moment clarification is also checked at
SHA-256
`3139eb8a4c7681c51a423e388be5bdcb7fea0eb3f4621122487a401e26045681`.
Every persistence, copying and revival source lies in the actual alive box.
Its original component velocity is bounded, while recipient jitter and OU
innovations retain their full Gaussian laws. The resulting exponential
position and velocity budgets therefore apply even when entering dead
coordinates have infinite moments and the marked input is atomic.

The proof uses the provider's own exponential positional tail to lower-bound
its actual Gaussian denominator at each query. The velocity exponential
budget gives a posterior mean growing at most linearly in query size. On a
bounded query event, exact numerator subtraction supplies the local transport
bound. Outside it, the two query tails and fourth moments give the displayed
$K_t$ coefficient. Balancing these contributions proves the global row
Hölder exponent $\gamma$ with the stated constants; uncapped stage velocities
require no uniform maximum.

The row proof also applies to the prescribed, generally nonoptimal,
common-OU coupling of the two joint stage laws. Its query discrepancy is
bounded by that coupling's phase cost, while a separate optimal provider
coupling has no larger cost in the pointwise row comparison. The marginal
second-moment bounds give the prescribed coupling cost at most
$\sqrt{8H}\le E$, which justifies the second-kick constant. Both Gaussian
transition marginals remain exact. Nonexpansiveness of the native cap and
the conditional maximal coupling of final positions then control terminal
mark disagreements, including box faces.

The unchanged marked preparation prefix compares bounded features and
alive-only box rewards. Its source-box eighth moments give the deterministic
weak-to-$W_4$ upgrade without an entering dead-tail assumption. Composition
with both row kicks proves {prf:ref}`thm-rwm-population-modulus` with
$\alpha_{\rm row}=\gamma^2/32>0$. This closes the population continuity
interface, including its arbitrary stage-coupling application. A conditional
finite empirical comparison and a temporal restart remain separate proof
inputs; no default population attraction follows from this modulus.

#### Conditional finite row comparison

The independent review of Sections 1--5 of `31_row_finite_transfer.md`
passes at SHA-256
`e77c67f85428c4bb12982b610f31b78a6ba0035bf335cd4eadf356cdd783bddd`.
The claimed endpoint is the explicit one-step consistency function
$(\mathrm{RFT}.10)$ for the actual full finite row update at its own
marked empirical input. It is uniform over its declared alive-floor class,
including unrestricted retained dead positions.

The first row proof subtracts the exact empirical self term: its numerator
is zero and its denominator is $a_{\rm emp}-1/N$. A target bulk-degree
bound and a transport-controlled exceptional degree event replace a
pathwise empirical degree premise. The ratio coefficients and fourth-root
cost of that event give $(\mathrm{RFT}.3)$--$(\mathrm{RFT}.4)$.

Conditional on the complete actual preparation, its empirical transport
plan to the deterministic target preparation defines independent auxiliary
target draws by label. Repeated empirical atoms can be disintegrated with
equal label weights. The auxiliary rows need not have identical laws; their
average conditional law is exactly the target preparation. Shared own OU
innovations preserve each transition marginal, and the fixed target first
field gives their exact average stage law. This justifies the conditional
variance and cell comparison, without replacing the actual prepared or noisy
empirical provider by those auxiliary rows. Both fourth-power triangle
charges and $(\mathrm{RFT}.5)$'s coefficients check.

The second row comparison retains the full joint provider. Its numerator
subtraction uses the deterministic target's uncapped fourth velocity moment
and Hölder, while the local target query cutoff bounds its numerator for
ratio subtraction. Bad pairs have bounded marked cost after the cap and
terminal decision. The actual self-excluded denominator is again unchanged.
Final independent Gaussian positions give the marked cell comparison with
the averaged source-box phase moment; conditional moment randomness is
integrated, not replaced by a pathwise moment assumption.

Finally the first query radius is a fixed multiple of the second radius.
Its displayed coefficient makes the amplified first-stage Gaussian tail
decay, while all denominator amplification of sampling errors is
$\exp(O(\sqrt{\log N}))$. The complete explicit
$\mathcal A_N$ therefore tends to zero. This proves
{prf:ref}`thm-rft-conditional-consistency`; it supplies the finite comparison
interface needed with the separately reviewed row population modulus and
recent survival tilt.

The complete temporal assembly and physical observations are independently
checked at final SHA-256
`11e4d6375dab604800a8847db1e141668b620542c25067ac9de3a216771083a1`.
The added closed consistency coefficients and eventual threshold dominate
the exact one-step function. The population burn-in follows from the uniform
first-output source-box moment and its strict marked moment recurrence.
The full row population contraction supplies the separate stationarity
input.

The recent continuation starts from the actual conditional past and retains
its own endpoint-survival Radon--Nikodym derivative, including the starting
tilt. A single whole-window alive-floor event is charged once. Conditional
consistency and subprobability Jensen give the displayed recurrence on that
event. The explicit window length has
$\alpha^{b_N-1}\ge\log(N+e)^{-1/4}$, while the local error's negative
logarithm grows at least as $\sqrt{\log(N+e)}$. Thus the iterated local
error, alive-floor failure, recent survival reweighting and population
relaxation term all have the stated vanishing limit. This proves
{prf:ref}`thm-rft-uniform-surviving-law`.

The normalized alive coupling is averaged over its good event before the
conditional marked expectation bound is used. Its physical diameter gives
the expected alive squared Wasserstein estimate, which also controls the
law of the random alive empirical measure against its Dirac target and the
swarm-first uniform-alive-slot sample. The separate all-slots-first sample
weights each surviving swarm by its actual alive fraction. That fraction
has conditional expectation at least $p$ by the raw last-step lower bound
and zero fraction on extinction; the resulting coupling therefore pays
the stated additional factor $m_f^{-1}$. Both sampling orders and their
own current-survival marginals check.

These deductions complete the small-positive-row-viscosity, fixed-large-box
surviving alive-law transfer. Its target is the normalized current-alive
stationary population law. Exact full finite-array mixing, a finite-$N$ QSD,
and the prescribed $\nu=0.3$, $L=2$ nonlinear population gap remain separate
claims.

#### Two separately surviving row swarm laws

The added {prf:ref}`cor-rft-two-surviving-swarms` in research record 31
passes at SHA-256
`627efdfe7eed14f1ed5334f89c471babbda6b9be61aa56bf1dd2d8cd5ca467e4`.
Each random alive empirical measure is distributed under its own chain's
current survival event. The outer Wasserstein triangle through the Dirac
law at their common current-alive stationary population target gives
(RFT.20). Equivalently, their two conditional laws can be coupled by their
product and the inner metric triangle integrated with Minkowski. Both
constructions preserve the two required marginals.

The ordinary phase-law triangle gives (RFT.21) for swarm-first uniform
alive sampling. Applying it to the separately normalized all-slot sample
bounds gives (RFT.22), with each additional alive-fraction factor retained.
No simultaneous-survival law of a prescribed paired trajectory enters
these deductions. They prove delayed law relaxation with the displayed
particle floors, without requiring monotone one-update pathwise distance.

#### Complete cap baseline and reference second-provider sensitivity

Research record 33 passes at SHA-256
`e66444a25d63ebc35a6fed13331d69f3b0f8c0e537b1a381f5b6c5eba2ff9aa8`.
The native-cap Jacobian is symmetric with eigenvalues in $[0,1]$;
integrating it gives an exact symmetric secant. Orthogonal diagonalization
therefore reduces the complete harmonic update to the scalar cap-sector
endpoints. The displayed rational interval certificates prove both
endpoint matrix deficits, yielding the strict full-update factor
$1-1/1040$ in $Q_{1/25}$. This argument checks the whole update and does
not import the invalid standalone cross-metric cap inference.

The complete innovation coupling maps the exact nonviscous finite kernel
into its own marginal laws. Contraction on the complete quadratic
Wasserstein space and the finite zero-anchor moment give the unique
finite-array invariant and the stated population-independent law rate.
The empirical-measure law and uniform-slot law are Lipschitz pushforwards
of the array law; their respective invariant targets remain distinct
from a deterministic population measure.

The enlarged positive-count interval uses research record 15's complete
paired-output perturbation before its old endpoint is imposed. That
derivation retains the actual correlated noisy second graph and the cap
Jacobian perturbation. Its two viscosity restrictions are exactly
$\nu\le1$ and $t\nu\le1$. The displayed $Q_m$--$Q_{1/25}$ norm
comparisons thus give the claimed new primitive endpoint and exact finite
law contraction, without a population floor. It still disables cloning
and death and does not include $\nu=.3$.

For the actual reference population second provider, differentiation
retains $y=x_1+tw$. Coercivity of the pre-cap map and the cap derivative
give the global radial Jacobian envelope. Shifted Gaussian ball
comparison and layer-cake integration prove the uniform Gaussian-averaged
velocity-mean bound. The own-provider interpolation retains both actual
joint stage laws. The radial identity
$|DC_V(z)z|\le V/4$ sharpens its scalar-field coefficient to the stated
$S_w<.622$; the remaining rigorous numeric margins follow from the given
rational bounds and a positive Gaussian tail mass.

The two-update cap-residual identity telescopes exactly, including any
intervening preparation's exact cost change. Its good-jitter estimates
cover persistent, cloned and revived source outcomes. The position
bad-event bound uses the stipulated shared OU and final innovations;
the jitter tail is kept and is not proportional to the input discrepancy.
The resulting reference own-provider signed block, terminal marking and
default dead-mass feedback remain explicit open estimates. These
intermediate bounds do not claim full active default convergence.
