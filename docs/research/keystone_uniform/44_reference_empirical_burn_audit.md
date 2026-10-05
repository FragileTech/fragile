# Independent audit of the default empirical-provider velocity burn

(sec-deva-retained)=
## 1. Exact source and consumed regime

The audited source is
`41_reference_empirical_velocity_burn.md` at SHA-256
`930ea99b9ffc53b0d2cc455c9e03dc8c2d78db09ad56cd4ab1bad30fe3522588`.
The recent-window input is
`37_default_marked_consistency.md` at SHA-256
`7cf47e4409933501bf1a926e4600ca80ca6ebe6d5dc536c20137205da8b9e58e`.
The six-step population energy burn and actual source/collision
identity are those of research 34; the safe-return/current-survival
and recent-window normalization are those of research 29.
The review did not alter any of these sources.

The executed arm is the harmonic current-frame count gas:
$F=-x$, $R=-|x|^2/2$, $d=3$, $h=.04$, $\nu=.3$, $L=2$,
$V=2$, $\sigma_J=\sigma_x=.1$ and component restitution $.5$.
Both count kicks, native smooth cap, original-slot collision,
mandatory revival, sampled alive normalizers and all Gaussian
innovations remain present. The source requires the explicit weak
positive exponent interval (DMC.3), rather than the unrestricted
reference choice $p_r=p_s=1$.

The audit result is **pass** for (DEV.4), (DEV.6), (DEV.9)--(DEV.11).
These are full-slot empirical moment events under the specified
current-survival law. They do not assert a stationary target or
default full-law contraction.

(sec-deva-population)=
## 2. Applicability and measurability of the population comparator

The interval in (DMC.3) is nonempty. Its primitive logarithmic range
$\Delta$ is positive because both amplitudes and reference exponents
are positive. Its denominator is positive and finite. Thus
$a_0>0$ and $\theta_f=\min\{1,\kappa_C/(8a_0)\}>0$.
For $0<\theta\le\theta_f$, the accepted alive incoming ceiling
$c_*\le1/8$ discharges the population finite-component premise.
Mandatory dead vertices remain leaves; the alive-floor class gives
them a finite incoming intensity, without declaring that intensity
small. The source-box first-moment argument of research 34 then
applies to the actual rooted preparation, including the empirical
atomic input law.

For each good entering array, the comparator is exactly
$\mathcal F^6(L_N(S))$. On the initial low-alive event one may,
as the source states, choose a fixed capped consistent positive-alive
law, for example the point mass at $(x,v,a)=(0,0,1)$, and use
its sixth image. These comparison objects do not change the
finite transition.

The random comparator is measurable. Empirical-law formation and the
good alive-fraction event are Borel. For the rooted population map,
all measured factors, normalized proposal probabilities with positive
denominators, gates, deterministic fitness tie rules, finite-component
readouts and Gaussian/cap stages are Borel. Its finite exploration
truncations therefore define Borel probability kernels. Their
almost-sure finite-component limits, consumed from the established
rooted construction, give a Borel population map when tested against
bounded Borel functions. Its sixfold composition at the entering
empirical law, together with the fixed choice on the bad event,
is consequently a valid random probability law.

Every possible comparator in this construction obeys the pointwise
population statement
$$
\mu_6|v|^2\le.55^2.
$$
That inequality holds separately for every permitted entering law;
it is unaffected by subsequent reweighting of the random entering
law by its recent future-survival probability.

(sec-deva-recent)=
## 3. The exact recent-survival carrier

:::{prf:lemma} The six-update random comparison retains its starting tilt
:label: lem-deva-recent-tilt

Let $k=n-6\ge1$, and start the ordinary stopped six-update
continuation from
$\eta_k=\operatorname{Law}(S_k\mid\tau_N>k)$.
Its conditional law given survival through the window has terminal
marginal $\eta_n$, and its random comparator satisfies
$$
\mathbb E\!\left[
\mathsf d_{\rm m}(L_N(S_n),\mu_6)
\mid\tau_N>n\right]
\le\min\{1,T_{N,6}[V_{N,6}+(5+c_{\rm ret})r_N]\}.
\tag{DEVA.1}
$$
The reweighted distribution of $S_k$ is part of this expectation.
:::

:::{prf:proof}
The Markov property and the definition of $\eta_k$ give the joint
ordinary-continuation law
$\eta_k(dS_k)P(dS_{k+1}\cdots dS_n)$, restricted once to its
window-survival event. Its terminal marginal is precisely
$\operatorname{Law}(S_n\mid\tau_N>n)$.
Its entering marginal is tilted by the actual probability of
surviving that window. There is no claim that its entering law
remains $\eta_k$ after this restriction.

Before that restriction, the initial low-alive event costs at most
$c_{\rm ret}r_N$, by the current-survivor alive-count estimate.
On its complement, the actual marked consistency and weak modulus
give the six-step bound $V_{N,6}$ on the good entering path.
The five subsequent entering failures cost at most $5r_N$.
The bounded marked distance is charged by one off this full good
event, after the comparison recurrence. Therefore its raw
window-survival numerator is at most
$V_{N,6}+(5+c_{\rm ret})r_N$.
Every nonextinct input has survival probability at least
$1-e_N$ at its next update, so the recent six-step denominator
is at least $(1-e_N)^6$. Dividing the same nonnegative numerator
by that actual denominator proves (DEVA.1).
This argument already includes the random comparator's starting
tilt. It does not introduce a denominator for the preceding $k$
updates, nor recondition any innovation to be a fresh Gaussian.
:::

On capped phase space $f(x,v,a)=|v|^2$ is $4$-Lipschitz for
$c_{\rm m}=\min\{1,|z-z'|+\mathbf1_{\{a\ne a'\}}\}$.
For cost below one, marks agree and
$|f-f'|\le(|v|+|v'|)|v-v'|\le4|z-z'|$.
For cost one, $f\in[0,4]$ proves the same bound.
Thus the threshold event $H_n>.56^2$ forces
$$
\mathsf d_{\rm m}(L_N(S_n),\mu_6)>
\frac{.56^2-.55^2}{4}=\frac{.0111}{4}.
$$
Markov's inequality applied to (DEVA.1) gives exactly the declared
$B_N^v$ and (DEV.4).

For fixed six, $T_{N,6}\to1$, $r_N\to0$ and $V_{N,6}\to0$.
In the eventual branch $A_N=\mathcal C_{\rm cons}N^{-1/384}$,
the power in $V_{N,6}$ is explicitly
$$
\frac{\alpha^5}{384}=\frac1{12884901888}>0.
$$
Hence the claimed vanishing floor is rigorous, though this
particular estimate gives an extremely slow particle-size rate.
The source does not claim an efficient numerical concentration
threshold from this rate.

(sec-deva-jitter)=
## 4. Actual source jitter and the first empirical field

Conditional on the complete actual source/component/Haar plan,
the canonical original-slot collision gives, for every plan,
$$
\frac1N\sum_i|P_i|^2\le H\le.56^2.
$$
It contracts the original velocity energy; no donor-velocity copy
has entered this calculation. Every positional source is in $D$,
including for a revived row with an arbitrarily distant retained
dead coordinate. The independent jitters are sampled after that
plan.

For $I_i\in\{0,1\}$ and $X_i=S_i+I_iJ_i$, the exact conditional
Gaussian quadratic formulas are
$$
\mathbb E_J|X_i|^2=|S_i|^2+I_id\sigma_J^2,\qquad
\operatorname{Var}_J(|X_i|^2)
=4I_i\sigma_J^2|S_i|^2+2I_id\sigma_J^4.
$$
Their conditional independence gives
$$
\mathbb E_JX_N\le12.03,\qquad
\operatorname{Var}_J(X_N)\le.4806/N.
$$
The exact rational comparison
$$
\frac{.4806}{(.22)^2}=9.929752\ldots<10
$$
proves (DEV.7) without a truncation of any recipient jitter.

After the complete prepared array is frozen, its count operator is
self-adjoint with $0\le L_X\le I$, and
$t\nu=.006<1$. Consequently
$\|(I-t\nu L_X)P\|_N\le\|P\|_N\le.56$ pathwise.
The dependence of $L_X$ on all preceding jitters does not alter
this deterministic operator inequality.

(sec-deva-ou)=
## 5. Full actual OU quadratic variance

Conditional on the prepared array with $X_N\le12.25$,
the original OU innovations are fresh independent standard Gaussians.
They have not yet been conditioned on terminal survival.
For $\bar w=c(U-tX)$,
$$
\|\bar w\|_N^2\le c^2(.56+.02\sqrt{12.25})^2
<.9608^2(.63)^2=.366392932416<.367.
$$
The bound $c>.96$ gives $q^2<.0392$.
The exact Gaussian square formulas therefore yield
$$
\mathbb E_\xi W_N
=\|\bar w\|_N^2+3q^2
<.367+3(.0392)=.4846<.485,
$$
$$
\operatorname{Var}_\xi(W_N)
=\frac{4q^2\|\bar w\|_N^2+6q^4}{N}
<\frac{4(.0392)(.367)+6(.0392)^2}{N}
=\frac{.06676544}{N}<\frac{.067}{N}.
$$
Because $.70^2=.49$, Chebyshev at the conservative gap $.005$
gives a conditional failure bound $2680/N$.
Adding the preceding positional failure $10/N$ proves
$2690/N$, and taking the minimum with one is valid for all $N$.
The computation uses the actual unbounded OU array. The second
count graph is not invoked in this stage-moment calculation and
has not been made independent of that array.

(sec-deva-survival-providers)=
## 6. Own next survival and actual empirical queries

Split the proposal from $\eta_n$ according to the entering event
$H_n\le.56^2$. Its complement has probability at most $B_N^v$,
and the actual conditional proposal on that event has stage failure
probability at most $2690/N$. This proves (DEV.9).

For the union of bad entering energy and bad proposed OU energy,
restrict its nonnegative indicator to the same actual next-survival
event and divide by that event's probability. The latter is at
least $1-e_N$ from every entering state. Hence
$$
\Pr(H_n>.56^2\ \text{or}\ W_N>.70^2\mid\tau_N>n+1)
\le\min\left\{1,\frac{B_N^v+2690/N}{1-e_N}\right\}.
$$
This retains the survival reweighting of the entering array,
discrete plan, jitters and OU draws. The calculation does not
preserve a conditional Gaussian law on the surviving good event.

For each realized empirical provider and every spatial query $x$,
$0\le K\le1$ gives
$$
\left|\frac1N\sum_iK(x-X_i)P_i\right|\le\|P\|_N,\qquad
\left|\frac1N\sum_iK(x-y_i)w_i\right|\le\|w\|_N.
$$
The count masses are at most one. On the displayed good event
these are exactly $.56$ and $.70$, respectively. At an own-row
query the added self term cancels in the count force:
$K(0)(v_i-v_i)/N=0$. No row-normalized or deterministic population
field has replaced the executed count field.

(sec-deva-endpoint)=
## 7. Audit endpoint and consumer restrictions

All requested interfaces pass at the stated weak positive
fitness-exponent interval:

- The random six-step comparator is measurable and retains the
  recent future-survival starting tilt.
- Population burn applies to each comparator, including atomic
  initial empirical laws, through the declared rooted component
  regime.
- Every scalar threshold and variance coefficient has an exact
  rational certificate.
- The source plan uses original velocities, and both empirical
  count fields retain their actual normalization and correlations.
- Next survival is charged by its own one-step denominator.

The good-provider event is a moment event involving the actual
Gaussians. A downstream derivative or coupling consumer must
retain its conditional dependence, use the provider inequalities
pointwise where appropriate, and charge its exceptional probability
with a proved moment inequality. It may not treat the restricted
jitters or OU draws as fresh Gaussian variables. The results concern
all slots, rather than an empirical average renormalized by the
random alive count. No population attraction, finite QSD law
convergence, or complete signed active feedback gap is supplied
by this audit.
