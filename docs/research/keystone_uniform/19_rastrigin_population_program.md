# The existing physical population programme on the unchanged Rastrigin force

:::{prf:proposition} Explicit native Rastrigin inputs for both dense normalizations
:label: prop-kurp-rastrigin-inputs

Retain the complete canonical real-coordinate marked update and the
declared $d=3$, $h=.04$, $D=[-2,2]^3$ coupled reference, changing
neither its force provider nor any source/noise parameter.
Here the declared provider is native Rastrigin,
$$
U(x)=\sum_r[x_r^2+10(1-\cos(2\pi x_r))],\quad
R(x)=-U(x),\quad F(x)=-2x-20\pi\sin(2\pi x).
$$
It is globally real analytic and has the calculated profiles
$$
F(0)=0,\quad L_F=2+40\pi^2,\quad
|F(x)|\le2|x|+20\pi\sqrt d,\quad
|R(x)|\le d(L_D^2+20)\quad(x\in D),
$$
$$
\operatorname{Lip}(R|_D)\le\sqrt d(2L_D+20\pi).
                                                               \tag{KURP.1}
$$
The two actual finite-QSD coercivity margins are
$$
\kappa_{\rm count}=1-t^2(2+40\pi^2)-t\nu
=.8352863295825703,\quad
\kappa_{\rm row}=1-t^2(2+40\pi^2)-2t\nu
=.8292863295825703.                                           \tag{KURP.2}
$$
Thus {prf:ref}`thm-cgd-analytic-force-qsd` supplies the unique
finite-$N$ QSD, including the actual unbounded Gaussian innovations.
This import gives finite-population spectral quantities; no uniform
right-eigenfunction conclusion is inferred from it.

For every $p\ge1$, the complete source and kinetic moment inputs are
the explicit native budgets
$$
\begin{aligned}
X_p&=\sqrt dL_D+\sigma_Jg_{d,p},\\
U_p&=V_c+t(2X_p+20\pi\sqrt d),\\
W_p&=cU_p+qg_{d,p},\\
Y_p&=X_p+bU_p+tqg_{d,p},\\
Z_p&=H_pW_p+t(2Y_p+20\pi\sqrt d),\\
X_p^+&=Y_p+sg_{d,p},\qquad |v_i^+|\le V,
\end{aligned}                                                 \tag{KURP.3}
$$
where $g_{d,p}$ is the proved Gaussian radial moment, $H_p=1$
in count mode and
$$
H_p=\min\{[1+t\nu(C_d-1)]^{1/p},
                         1-t\nu+t\nu C_d^{1/p}\}
$$
in row mode. Use the exact dimension-only column envelope of
(KUK.7), or any proved finite-series upper bound for it.
The formulas apply to the actual correlated joint laws; they
are normalized row moments, not bounds on a maximum Gaussian draw.

*Proof.* Differentiate the configured cosine objective.
The diagonal derivative of $F$ has absolute value at most
$2+40\pi^2$, and $|\sin|\le1$ gives its stronger linear-growth
profile. On the alive box each coordinate objective lies in
$[0,L_D^2+20]$ and its gradient is bounded by $2L_D+20\pi$.
The actual $t=.02$, $\nu=.3$ give (KURP.2), verifying the
analytic-force QSD hypotheses. Positive normalizers, isotropic
noises, clone jitter, cap and terminal rules are precisely the
unchanged canonical hypotheses of that theorem.
Minkowski applied to the complete stage identities proves
(KURP.3), just as in the original kinetic budget.
The stochastic first-kick matrices preserve $V_c$ rowwise.
Count contraction or the proved Gaussian column inequality
bounds the unbounded B2 input in $L^p$. No curvature sign is used.
$\square$
:::

:::{prf:theorem} Physical Rastrigin mean-field trajectory and survivor transfer
:label: thm-kurp-rastrigin-physical-mean-field

For either dense normalization, require the entering swarms to be
nonextinct almost surely. Let their marked empirical
laws converge in probability in $W_4$ to a specified deterministic
$\mu_0$ with positive alive mass, terminally consistent marks and
capped velocities. Assume a uniform expected position moment of
some order $p>4$. Let $\mathcal F_h^R$ be the actual rooted-component
population map with this same native reward and force, both actual
B kicks and the complete original preparation. Put
$\mu_{n+1}=\mathcal F_h^R\mu_n$.

For every fixed $T$, the complete empirical trajectory satisfies
$$
(L_N(S_0),\ldots,L_N(S_T))
\longrightarrow(\mu_0,\ldots,\mu_T)
\quad\text{in probability in marked }W_4.                    \tag{KURP.4}
$$
The same convergence holds after conditioning once on survival
through $T$. If the actual global extinction envelope is
$\delta_N\le e^{-Na_*}$, its entire-path conditioning cost is
$$
\|\mathsf P_{N,T}^{\rm surv}-\mathsf P_{N,T}\|_{\rm TV}
=\Pr(\tau_N\le T)\le T e^{-Na_*}.                            \tag{KURP.5}
$$
A cemetery observation on extinct paths makes the statement
well defined. Initial data may follow different population
trajectories; no attraction to one universal phase is assumed.

*Proof.* The actual alive sources lie in $D$, so their raw reward
is bounded and Lipschitz by (KURP.1). Squashed comparison features
are bounded and Lipschitz; their Gaussian companion weights have
the same positive primitive lower bounds as (KU.1).
Positive alive mass, normalization floors and the complete sampled
measurement law therefore verify the original preparation
consistency hypotheses of
{prf:ref}`thm-mean-field-one-step-consistency`.
The strict actual fitness ordering of an accepted live edge,
outdegree at most one, and conditional independent recipient
choices verify the component truncation hypotheses.
The expected component size bound is finite on every positive
alive-mass class. Its tail is removed only in the proof.
It includes mandatory dead leaves and the shared component Haar
matrices, rather than replacing them by independent rotations.

Post-collision velocities are bounded by $V_c$.
All prepared position moments are given by (KURP.3); hence
preparation consistency upgrades to $W_4$.
The two count kinetic consistency estimates used in
{prf:ref}`thm-cg-mf-kinetic-limit` require the force to be
globally Lipschitz with linear growth at the two actual inputs.
(KURP.1) verifies those properties directly.
The first count kick has bounded collision velocities; the
second uses its uncapped OU velocities and the proved fourth-moment
coupling estimate. The additional periodic part is Lipschitz,
so it introduces at most $20\pi(2\pi)|x-\widetilde x|$
into that same force difference. Every input moment is (KURP.3).

For row normalization, apply the exact local-degree argument of
{prf:ref}`lem-cg-mf-row-local-normalization`.
On a fixed analysis ball it bounds the true denominator
$a_{L_N}(x_i)-1/N$ using a positive mass of the target law
in a finite ball and its Gaussian kernel value there.
No global degree floor is postulated.
The bounded first-kick velocities give $L^4$ convergence of
its force; its nonlinear Rastrigin term uses $L_F$ in (KURP.1).
After OU, the entire intermediate array is the conditioning
variable for its independent innovations.
At B2 the local-degree lemma uses the finite fourth moments
of (KURP.3). The actual uncapped force difference converges
in coupling probability. Continuity and boundedness of the
original radial cap upgrade its velocity output to $W_4$.
This repeats the row proof without invoking count-only
conditional variance or independent prepared rows.

Both modes retain final position noise $s>0$. The limiting
position marginal is therefore absolutely continuous and has
zero mass on $\partial D$; terminal marks pass to the limit.
These steps prove complete one-step marked $W_4$ consistency.
The actual global landing floor supplies positive output alive
mass, so the same argument can be iterated at every fixed time.
The uniform moment budgets permit induction over the fixed
$T$ steps, proving (KURP.4).
First-extinction events and the actual one-step bound give
(KURP.5); conditioning any path law on an event of mass
$1-\Pr(\tau_N\le T)$ changes it by exactly that TV amount.
Thus the survivor trajectory has the same limit. $\square$
:::

:::{prf:theorem} QSD population distributions and attainable Rastrigin orbits
:label: thm-kurp-rastrigin-qsd-population-invariance

Let $\nu_N$ be the actual finite-$N$ Rastrigin QSD from (KURP.2).
For each dense normalization separately:

1. The actual population map $\mathcal F_h^R$ has at least one
   stationary population law with positive alive mass.
2. The distributions $\Lambda_N=(L_N)_\#\nu_N$ are tight on
   marked population laws in $W_4$. Every subsequential limit
   satisfies exactly
   $$
   (\mathcal F_h^R)_\#\Lambda=\Lambda.                       \tag{KURP.6}
   $$
3. Along this same subsequence, for every fixed $k$ the labelled
   QSD rows converge weakly to
   $$
   \int\mu^{\otimes k}\Lambda(d\mu).
                                                               \tag{KURP.7}
   $$
   Uniformly selected distinct alive rows converge instead to
   $\int[\mu(a\,\cdot)/\mu(a)]^{\otimes k}\Lambda(d\mu)$.
4. Starting the physical gas from $\nu_N$ and conditioning through
   any fixed $T$ gives the limiting empirical trajectory
   $\mu_0\sim\Lambda$, $\mu_{n+1}=\mathcal F_h^R\mu_n$.
   This trajectory is stationary under integer shifts.

These are stationary distributions of population trajectories.
They retain cycles, distinct well orbits and nontrivial invariant
distributions; they do not identify every force root with a
stationary swarm law or every invariant distribution with a
mixture of fixed points. No right-eigenfunction ratio is an input.

*Proof.* Let $K_p=(X_p^+)^p$ from (KURP.3), $p>4$.
The global extinction envelope gives
$\alpha_N\ge1-e^{-Na_*}>0$, and the QSD equation implies
$$
\nu_N\left(N^{-1}\sum_i|x_i|^p\right)
\le\frac{K_p}{1-e^{-Na_*}}\le
                    \frac{K_p}{1-e^{-a_*}}.                 \tag{KURP.8}
$$
The cap bounds velocities. Markov and the $p>4$ tail give
$W_4$ tightness of $\Lambda_N$, exactly as in the existing
stationary empirical proof.

The actual preparation-exact count Laplace bound, or the stronger
independent lower-trial version, gives
$$
\nu_N\{M/N<a_*/2\}
\le\frac{\exp[-c_*a_*N]}{1-e^{-Na_*}},\qquad
c_*=(1-\log2)/2>0.                                          \tag{KURP.9}
$$
Indeed use $\theta=\log2$ in the count transform and the QSD
equation; no stochastic binomial domination is needed for this
Chernoff implication. Its right side tends to zero, so limit
laws have positive alive fraction almost surely.

The raw one-row completed position is a Gaussian mixture with
variance $\tau^2I$, $\tau^2=t^2q^2+s^2$, and hence density at most
$(2\pi\tau^2)^{-d/2}$. Its QSD marginal is bounded above by this
value divided by $\alpha_N$. This excludes boundary mass in
the empirical limit: the expected mass in shrinking boundary
neighborhoods tends to zero, and nonnegative disintegration
makes each limiting population law boundary-null almost surely.
Terminal marks therefore retain their declared meaning.

For population fixed points consider the weakly compact convex
class with moment at most $K_p$, capped velocities, alive mass
at least $a_*$, terminal consistency and position density bounded
above by $(2\pi\tau^2)^{-d/2}$. Every actual population output
lies in this class. The moment and density bounds make it closed
and tight and exclude boundary atoms.
The preparation consistency proof used in (KURP.4), its finite
component truncation, and the two kinetic continuity arguments
prove that $\mathcal F_h^R$ is continuous on it.
The compact-convex fixed-point argument of
{prf:ref}`thm-mean-field-stationary-existence` applies,
proving existence. Output alive mass at least $a_*$ follows
from the mean of the same count Laplace bound, or by the
proved deterministic empirical approximation of the rooted law.

One-step consistency is uniform on compact input classes with
the above moments and positive alive mass: otherwise a sequence
of deterministic counterexample inputs has a convergent
subsequence, contradicting (KURP.4).
Localize the QSD inputs by (KURP.8)--(KURP.9).
For a bounded population metric $d_*$, this gives
$$
\mathbb E_{\nu_N}d_*(L_N(S^+),
                         \mathcal F_h^R(L_N(S)))\to0.        \tag{KURP.10}
$$
Meanwhile the full raw QSD output differs from $\nu_N$ in TV
by exactly $1-\alpha_N\le e^{-Na_*}$.
Pass a bounded Lipschitz population test through (KURP.10)
and the proven continuity. This proves (KURP.6).
Invariance also upgrades the limiting support to the population
output class, including alive mass at least $a_*$.

The entire configured kernel is equivariant under row relabeling,
so uniqueness makes $\nu_N$ exchangeable.
Given its empirical multiset, fixed labelled rows are sampled
without replacement. Their difference from independent empirical
sampling is at most $k(k-1)/(2N)$ in TV. Passing the same
subsequence proves (KURP.7).
For alive labels, use $M/N\ge a_*/2$; its sampling difference
is at most $k(k-1)/(a_*N)$ and its complement vanishes by
(KURP.9). The normalized alive map is continuous at positive
alive mass in the marked space, proving the stated mixture.
Finally iterate compact-localized consistency at the fixed
number of times. From a QSD the full-horizon conditioning cost
is at most $T e^{-Na_*}$, so it preserves the trajectory limit.
(KURP.6) gives shift stationarity. $\square$
:::
