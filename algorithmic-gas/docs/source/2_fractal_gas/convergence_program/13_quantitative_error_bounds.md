# Quantitative Error Bounds

(sec-quantitative-entropy)=
## Relative Entropy and Particle Marginals

:::{div} feynman-prose
A numerical estimate can be inaccurate for several reasons. The finite population can have a different equilibrium from the limiting equation. Its empirical average fluctuates. A finite time step changes the transition law. And a run may still remember its initial condition. The final estimate in this chapter keeps these four contributions separate so that each can be traced to its own proof.

We begin with population error. Relative entropy compares the full joint law with a product reference. When that reference satisfies a logarithmic Sobolev inequality, entropy also controls a transport distance. Exchangeability then divides the joint squared transport cost equally among coordinates. This is the source of the factor $1/N$ in the one-particle bound below.
:::

:::{prf:lemma} Transport bound with the reference law in the correct direction
:label: lem-wasserstein-entropy

Suppose a probability law $\rho$ on Euclidean phase space satisfies
$\operatorname{Ent}_{\rho}(f^2)\leq2C\int|\nabla f|^2d\rho$.
For a law $\pi_N$ on the $N$-particle product space, put
$H_N=D_{\mathrm{KL}}(\pi_N\Vert\rho^{\otimes N})$. Then

$$
W_2^2(\pi_N,\rho^{\otimes N})\leq2CH_N.
$$

If $\pi_N$ is exchangeable, its one-particle marginal satisfies
$W_2^2(\pi_{N,1},\rho)\leq2CH_N/N$.
:::

:::{prf:proof}
Tensorization preserves the LSI constant, as proved in
{prf:ref}`thm-tensorization`. We recall the transport implication to specify
its reference law. For a bounded Lipschitz function $f$ let
$Q_tf(x)=\inf_y\{f(y)+|x-y|^2/(2t)\}$ be its Hopf–Lax infimum convolution.
It satisfies $\partial_tQ_tf=-|\nabla Q_tf|^2/2$ almost everywhere. Define
$Z(t)=\int e^{tQ_tf/C}d\rho$ and $F(t)=C\log Z(t)/t$.
Differentiation and LSI applied to $e^{tQ_tf/(2C)}$ give

$$
F'(t)=\frac{C}{t^2}\frac{\operatorname{Ent}_{\rho}(e^{tQ_tf/C})}{Z(t)}
+\mathbb E_t\partial_tQ_tf
\leq\tfrac12\mathbb E_t|\nabla Q_tf|^2
-\tfrac12\mathbb E_t|\nabla Q_tf|^2=0.
$$

Thus $C\log\int e^{Q_1f/C}d\rho\leq\int f\,d\rho$.
The entropy inequality yields
$\int Q_1f\,d\nu-\int f\,d\rho\leq C D_{\mathrm{KL}}(\nu\Vert\rho)$.
The quadratic Kantorovich dual formula, followed by truncation when needed,
gives $W_2^2(\nu,\rho)\leq2CD_{\mathrm{KL}}(\nu\Vert\rho)$.
Apply this to the product reference.

Finally symmetrize any nearly optimal coupling under simultaneous particle
permutations. Both marginals are preserved, and each coordinate pair has
expected squared cost equal to the total cost divided by $N$. Its first
coordinate is a coupling of $\pi_{N,1}$ and $\rho$. Taking the infimum proves
the marginal bound.
:::

:::{prf:lemma} Quantitative entropy bound from a closed dissipation estimate
:label: lem-quantitative-kl-bound

For a specified evolving joint law and reference law, suppose its relative
entropy is absolutely continuous and satisfies

$$
H_N'(t)\leq-\lambda H_N(t)+D_N(t),\qquad \lambda>0.
$$

Then

$$
H_N(t)\leq e^{-\lambda t}H_N(0)
+\int_0^te^{-\lambda(t-s)}D_N(s)\,ds.
$$

If $D_N(t)\leq D_*$ uniformly in $N,t$, then
$\limsup_{t\to\infty}H_N(t)\leq D_*/\lambda$ uniformly in $N$.
The exact kinetic, jump and conditioned-QSD entropy identities used to verify
the dissipation estimate are in {doc}`15_kl_convergence`.
:::

:::{prf:proof}
Multiply the differential inequality by $e^{\lambda t}$ and integrate.
The uniform bound follows from integrating the resulting exponential kernel.
The entropy, reference law and all normalization terms must be the same in
the identity and in its estimates.
:::

:::{div} feynman-prose
Imagine preparing the comparison population by drawing each marked slot from
$\rho$, then discarding preparations in which every slot is dead. That is
$R_N$. Now advance this preparation through the complete gas update. Walkers
can share donors and component rotations, so the resulting $R_N^+$ contains
the correlations produced by the algorithm. The term $\mathcal I_n$ measures
the change of reference caused by this step.

Survival also changes which inputs we see. An input with larger survival
probability contributes more often to the surviving output ensemble. The
covariance and logarithm in $\mathcal S_n$ account for that reweighting and
its normalization. Keeping these two effects visible lets us locate the
estimates needed for concentration: entropy lost through the transition,
entropy contributed by the evolving comparison law, and the selection of
surviving runs.
:::

:::{prf:theorem} Exact discrete population-entropy accounting
:label: thm-quantitative-full-step-entropy

Let $Q_N$ be the complete marked Euclidean Gas kernel, $k_N=Q_N1$, and
$E_N$ its nonextinct input state space. Let $\rho$ be a terminally
consistent one-slot law with alive mass $m>0$. Define

$$
q_N=\rho^{\otimes N}(E_N)=1-(1-m)^N,\qquad
R_N=\rho^{\otimes N}(\,\cdot\mid E_N),\qquad
R_N^+=\frac{R_NQ_N}{R_Nk_N},\quad
b_N=R_Nk_N.
$$

For the actual survival-conditioned iterates $P_{n+1}=P_nQ_N/a_n$, where
$a_n=P_nk_N$, set
$f_n=dP_n/dR_N$ and $H_n=D(P_n\Vert R_N)$.
Whenever the displayed terms are finite, their exact per-step balance is

$$
H_{n+1}-H_n
=-\mathcal B_N(P_n,R_N)+\mathcal S_n+\mathcal I_n,
$$

where

$$
\mathcal S_n
=\frac{\operatorname{Cov}_{P_n}(k_N,\log f_n)}{a_n}
+\log\frac{b_N}{a_n},
\qquad
\mathcal I_n=P_{n+1}\log\frac{dR_N^+}{dR_N},
$$

and $\mathcal B_N$ is the nonnegative backward relative entropy in
{prf:ref}`thm-kl-full-step-survivor-chain-rule`. Thus for every $t\geq1$,

$$
H_t+\sum_{n=0}^{t-1}\mathcal B_N(P_n,R_N)
=H_0+\sum_{n=0}^{t-1}(\mathcal S_n+\mathcal I_n).
$$

For an actual QSD $P_n=\nu_N$, the left entropy increment is zero, giving
the stationary identity
$\mathcal B_N(\nu_N,R_N)=\mathcal S_\nu+\mathcal I_\nu$.
The unconditioned product-reference entropy and the entropy per slot are
respectively

$$
D(P_n\Vert\rho^{\otimes N})=H_n-\log q_N,
\qquad
\frac1N D(P_n\Vert\rho^{\otimes N})
=\frac{H_n-\log q_N}{N}.
$$

These identities also hold when $\rho$ is a fixed point of the actual
population map $\mathcal F_h$. The finite-population output $R_N^+$ in
$\mathcal I_n$ retains its component dependence and need not be a product.
:::

:::{prf:proof}
The event that every independently sampled slot is dead has probability
$(1-m)^N$, which proves $q_N$. On $E_N$,
$dR_N/d\rho^{\otimes N}=1/q_N$. The logarithmic likelihood-ratio identity
therefore gives $D(P_n\Vert\rho^{\otimes N})=H_n-\log q_N$.
Apply {prf:ref}`thm-kl-full-step-survivor-chain-rule` with $P=P_n$,
$R=T=R_N$; its covariance form gives the claimed increment exactly.
Summing telescopes. For a QSD the normalized full-step output equals the
input, so its entropy increment is zero. No exchangeability cancellation,
time interpolation, or differentiation of a replacement generator is used.
:::

:::{prf:corollary} Bounded-observable concentration with the survival cost
:label: cor-quantitative-marked-product-entropy

Under the notation of {prf:ref}`thm-quantitative-full-step-entropy`, let
$\varphi$ be any bounded measurable one-slot observable, including the
alive indicator, whose range lies in an interval of length $\ell>0$.
For any input law $P$ on $E_N$,

$$
\mathbb E_P\left|L_N\varphi-\rho\varphi\right|^2
\leq\frac{\ell^2}{N}
\left[D(P\Vert R_N)-\log q_N+\frac12\log2\right].
$$

The same right-hand side bounds $\operatorname{Var}_P(L_N\varphi)$.
The conditioning contribution satisfies
$-\log q_N\leq(1-m)^N/[1-(1-m)^N]$.
:::

:::{prf:proof}
Under $Q=\rho^{\otimes N}$, Hoeffding's lemma and independence give,
for $F=L_N\varphi-\rho\varphi$,
$\mathbb E_Qe^{sF}\leq e^{s^2\ell^2/(8N)}$.
For an independent standard Gaussian $G$, Tonelli's theorem yields

$$
\mathbb E_Qe^{NF^2/\ell^2}
=\mathbb E_G\mathbb E_Qe^{\sqrt{2N}\,GF/\ell}
\leq\mathbb E_Ge^{G^2/4}=\sqrt2.
$$

The entropy variational inequality bounds
$\mathbb E_P[NF^2/\ell^2]$ by $D(P\Vert Q)+\tfrac12\log2$.
Substitute the exact conditioning identity. Variance is no larger than
squared error about a specified constant. Finally
$-\log(1-x)=\int_0^x(1-u)^{-1}du\leq x/(1-x)$ for $0\leq x<1$.
No joint QSD LSI or independence of the walkers under $P$ is needed.
:::

:::{div} feynman-prose
The differential inequality has a useful interpretation: dissipation removes entropy, while the residual term can replenish it. A bounded residual leaves a bounded long-time entropy level. To make that bound uniform in the number of particles, the residual must itself have a uniform bound. The covariance calculation below shows one way an interaction error can meet that requirement.
:::

:::{prf:proposition} Interaction fluctuations with covariance control
:label: prop-interaction-complexity-bound

For centered scalar random variables $Y_1,\ldots,Y_N$, if
$\operatorname{Var}(Y_i)\leq v$ and
$|\operatorname{Cov}(Y_i,Y_j)|\leq c/N$ for $i\ne j$, then

$$
\mathbb E\left|\frac1N\sum_iY_i\right|^2\leq\frac{v+c}{N}.
$$

Consequently a defect bounded by
$D_N\leq A N\mathbb E|N^{-1}\sum_iY_i|^2+B$ is at most $A(v+c)+B$,
independently of $N$. The covariance hypotheses can be supplied by
{prf:ref}`thm-correlation-decay` under its stated analytic inputs.
:::

:::{prf:proof}
Expand the squared average. Its $N$ diagonal terms contribute at most $v/N$,
and its $N(N-1)$ off-diagonal terms contribute at most $c(N-1)/N^2$.
Substitution proves the defect bound.
:::

:::{prf:lemma} Observable bias and sampling fluctuations
:label: lem-lipschitz-observable-error

Let $\pi_N$ be exchangeable and $\phi$ be $L$-Lipschitz. Put
$\widehat\phi_N=N^{-1}\sum_i\phi(Z_i)$.
Then

$$
\mathbb E|\widehat\phi_N-\rho\phi|
\leq L W_2(\pi_{N,1},\rho)
+\sqrt{\operatorname{Var}_{\pi_N}(\widehat\phi_N)}.
$$

If $\pi_N$ has full-gradient LSI constant $C_*$ in the convention
$\operatorname{Ent}(f^2)\leq2C_*\int|\nabla f|^2$, then
$\operatorname{Var}(\widehat\phi_N)\leq C_*L^2/N$.
:::

:::{prf:proof}
Split the absolute error at its expectation and use Cauchy–Schwarz for the
centered term. Exchangeability identifies the expectation with
$\pi_{N,1}\phi$, whose bias is bounded by the Kantorovich–Rubinstein inequality
and $W_1\leq W_2$. Linearizing LSI at the constant function gives Poincaré
with constant $C_*$. The empirical average has squared joint gradient norm
$N^{-2}\sum_i|\nabla\phi(Z_i)|^2\leq L^2/N$, proving the variance estimate.
:::

:::{div} feynman-prose
The empirical average has two errors. Its center can be displaced from the desired answer, and individual samples fluctuate around that center. A marginal transport estimate controls the displacement. A joint variance estimate controls the fluctuations, including dependence among walkers.

Estimating an entire distribution is harder than estimating one fixed observable. Even independent samples leave empty regions and uneven coverage. The next bound retains that ordinary sampling error instead of hiding it inside the interaction estimate.
:::

:::{prf:proposition} Empirical Wasserstein error retains its sampling term
:label: prop-empirical-wasserstein-concentration

Under any coupling to independent particles $Y_i\sim\rho$,

$$
\mathbb E W_2\!\left(\frac1N\sum_i\delta_{Z_i},\rho\right)
\leq\left(\frac1N\sum_i\mathbb E|Z_i-Y_i|^2\right)^{1/2}
+\mathbb E W_2\!\left(\frac1N\sum_i\delta_{Y_i},\rho\right).
$$

The last term is the empirical sampling error for $\rho$; it depends on the
dimension and moment class. An $N^{-1/2}$ rate for each fixed observable is
not a dimension-independent $W_2$ rate for the entire empirical measure.
:::

:::{prf:proof}
This is the empirical-coupling construction proved in
{prf:ref}`thm-mean-field-limit-informal`: pair coordinates, apply the triangle
inequality, and use Jensen's inequality on the mean squared coupling cost.
:::

:::{prf:theorem} Physical quantization error for the actual population law
:label: thm-quantitative-algorithm-quantization-lower-bound

Let $\mu=\mathcal F_h(\lambda)$ be the output of the actual marked
Euclidean Gas population update, with final independent position Gaussian
noise of standard deviation $s=\sigma_x\sqrt h>0$ in each coordinate.
Let $\mu_x$ be its full-slot physical position marginal in $\mathbb R^d$,
including the retained positions of dead slots. Write
$\omega_d$ for the volume of the unit Euclidean ball and
$M_s=(2\pi s^2)^{-d/2}$.
For every probability measure $\eta_N$ supported on at most $N$ physical
positions,

$$
W_2^2(\eta_N,\mu_x)
\geq\frac12(2M_s\omega_d)^{-2/d}N^{-2/d}.
$$

In particular the bound holds for every realized empirical position measure
of an actual Rust gas run, without any independence assumption on its slots.
It also holds with $\mu=\mu_*$ for every fixed point
$\mu_*=\mathcal F_h(\mu_*)$ of this same algorithm.

When $d>2$, no constant $C$ independent of $N$ can satisfy either
$\mathbb EW_2^2(\eta_N,\mu_x)\leq C/N$ or
$\mathbb EW_2(\eta_N,\mu_x)\leq C/\sqrt N$ for all $N$.
The same obstruction holds for physical phase-space Wasserstein distance
whose ground metric dominates position distance. It concerns distributional
quantization and does not contradict the bounds for each fixed observable
in {prf:ref}`cor-quantitative-marked-product-entropy`.
:::

:::{prf:proof}
Condition on all stages before the final position noise, including the
sampled-fitness graph, the shared component rotations, revival, jitter,
and BAOAB. If $\Lambda_x$ is the resulting pre-noise position law, the
full-slot output position marginal is exactly

$$
\mu_x=\Lambda_x*\mathcal N(0,s^2I_d).
$$

The velocity cap does not change positions. The terminal boundary operation
assigns alive/dead marks and retains both sets of position coordinates, so
summing over those marks preserves this convolution identity. Consequently
$\mu_x$ has a density bounded above by $M_s$ everywhere.

Let $A$ be the support of $\eta_N$ and put
$r=(2NM_s\omega_d)^{-1/d}$. The union $U$ of radius-$r$ balls about the
at most $N$ points of $A$ satisfies

$$
\mu_x(U)\leq M_sN\omega_dr^d=\frac12.
$$

In every coupling of $X\sim\mu_x$ and $Y\sim\eta_N$, the event
$X\notin U$ therefore has probability at least $1/2$, and on that event
$|X-Y|\geq r$. Thus every coupling costs at least $r^2/2$.
Taking the infimum proves the first bound, including the case of infinite
transport cost. Projection from physical phase space to position cannot
increase Wasserstein distance, so its cost has the same lower bound.

The inequality holds for each realization and hence after taking expectation.
Its square root gives the deterministic lower bound
$W_2\geq(2M_s\omega_d)^{-1/d}N^{-1/d}/\sqrt2$.
For $d>2$, the ratios of these lower bounds to $N^{-1}$ and $N^{-1/2}$,
respectively, grow without bound. The fixed-point substitution uses its
actual one-step identity and requires no replacement particle model.
:::

:::{div} feynman-prose
There are two different measurement tasks here. For a fixed bounded
observable, such as the alive fraction, each run supplies one average.
The entropy bound above gives a standard error of order $N^{-1/2}$ when
the joint product-reference entropy stays bounded uniformly in $N$.
Establishing that entropy bound is the analytic task; the independence
used in its comparison calculation belongs to the reference law.

To approximate the entire position distribution in $W_2$, the same run
must represent a smooth cloud with only $N$ points. Place a small ball
around each point. Because the final Gaussian noise bounds the target
density, these balls cover at most half its probability when their radius
is of order $N^{-1/d}$. The remaining probability must travel at least
that distance. This geometric cost persists even for perfectly arranged
points. It concerns spatial resolution; the cluster analysis controls
the dependence among walkers. A sharp cluster estimate can therefore
coexist with a dimension-dependent empirical transport rate.
:::

:::{prf:proposition} Second moments of a mean-field limit
:label: prop-finite-second-moment-meanfield

If $\pi_{N,1}\Rightarrow\rho$ and
$\sup_N\int|z|^2d\pi_{N,1}\leq M_2$, then $\int|z|^2d\rho\leq M_2$.
Convergence in $W_2$ additionally follows when these second moments are
uniformly integrable and the weak limit is identified.
:::

:::{prf:proof}
Apply weak convergence to the bounded continuous functions
$\min(|z|^2,R)$ and let $R\to\infty$. Uniform integrability also permits
passage of the untruncated moments; weak convergence together with convergence
of second moments is the $W_2$ convergence criterion.
:::

:::{prf:theorem} Quantitative propagation for fixed Lipschitz observables
:label: thm-quantitative-propagation-chaos

Under {prf:ref}`lem-wasserstein-entropy`, if the joint empirical variance bound
is $\operatorname{Var}(\widehat\phi_N)\leq A_\phi/N$, then

$$
\mathbb E_{\pi_N}|\widehat\phi_N-\rho\phi|
\leq L\sqrt{\frac{2CH_N}{N}}+\sqrt{\frac{A_\phi}{N}}.
$$

In particular $H_N=O(1)$ and $A_\phi=O(1)$ give an $O(N^{-1/2})$ error for
that observable. The bounded-observable alternative, including indicators,
is {prf:ref}`thm-mixing-variance-corrected`.
:::

:::{prf:proof}
Insert the marginal transport bound into
{prf:ref}`lem-lipschitz-observable-error`. The two terms respectively control
the one-particle bias and the variance of the empirical estimator.
:::

(sec-quantitative-moments)=
## Moment Estimates for Time Discretization

:::{div} feynman-prose
A Taylor remainder involves powers of the state and of the random increment. Smooth coefficients are therefore only part of a time-discretization proof: we must also control the averages of those powers. The Gaussian fourth-moment calculation gives an exact input for a kinetic substep. A velocity cap gives a pointwise bound. On an unbounded state space, a Lyapunov calculation must control the required higher moment explicitly.

This is why a bound on the mean of a quadratic observable cannot simply be reused as a bound on its square. The product rule introduces the additional noise term displayed below.
:::

:::{prf:proposition} Fourth-moment inputs for Langevin and capped updates
:label: prop-fourth-moment-baoab

For a Gaussian substep $V'=a v+b+s\xi$, with $\xi\sim N(0,I_d)$,

$$
\mathbb E|V'|^4=|av+b|^4+2(d+2)s^2|av+b|^2+d(d+2)s^4.
$$

A cap $|\psi_v(V')|\leq v_{\max}$ gives
$\mathbb E|\psi_v(V')|^4\leq v_{\max}^4$.
For an unbounded diffusion, if an auxiliary $V\geq1$ satisfies
$LV\leq-aV+b$ and its carré du champ satisfies $\Gamma(V)\leq cV$, then

$$
L(V^2)\leq-aV^2+\frac{(b+c)^2}{a}.
$$

Under the usual stopping/localization justification for Dynkin's formula,
this yields a uniform fourth-moment bound when $V$ dominates $|z|^2$.
For a discrete chain the corresponding input is a bound on $P(V^2)$ itself.
:::

:::{prf:proof}
Expand $|m+s\xi|^4$ with $m=av+b$, discard odd Gaussian terms, and use
$\mathbb E|\xi|^2=d$, $\mathbb E|\xi|^4=d(d+2)$ and
$\mathbb E(m\cdot\xi)^2=|m|^2$. The cap estimate is pointwise.
The diffusion product rule gives
$L(V^2)=2VLV+2\Gamma(V)\leq-2aV^2+2(b+c)V$.
Since $2(b+c)V\leq aV^2+(b+c)^2/a$, the claimed drift follows.
Integrating its scalar differential inequality proves the moment bound.
:::

(sec-quantitative-local-errors)=
## Local and Finite-Time Weak Errors

:::{div} feynman-prose
“Weak error” measures the error in an expected observable. To understand its time-step order, first compare one numerical step with one exact step. Then add those local discrepancies over a fixed physical duration. There are roughly $T/h$ steps, so a local error of order $h^{p+1}$ gives a global error of order $h^p$ when propagation remains stable.

The symmetric BAOAB arrangement cancels the second-order local discrepancy on its common regularity domain. The operator calculation below makes that cancellation explicit. Boundary rules, caps, and status changes still have to satisfy the hypotheses used in the expansion.
:::

:::{prf:lemma} BAOAB weak consistency on a common regularity domain
:label: lem-baoab-weak-error

Let $A,B,O$ be the transport, force and Ornstein–Uhlenbeck backward generators,
$L=A+B+O$, and

$$
P_h=e^{hB/2}e^{hA/2}e^{hO}e^{hA/2}e^{hB/2}.
$$

On a common test domain suppose all ordered generator products through degree
three needed in the semigroup Taylor expansions are bounded in a weighted
norm, with their integral remainders bounded uniformly for $0<h\leq h_0$.
Then, for each test function in that domain,

$$
P_h\phi=\phi+hL\phi+\tfrac12h^2L^2\phi+O(h^3),\qquad
\|(P_h-e^{hL})\phi\|_V\leq K_\phi h^3.
$$

These are test-domain conditions on the specified splitting, including its
boundary and cap conventions. Smoothness of a fitness potential alone does
not verify them for a hybrid update with status changes.
:::

:::{prf:proof}
For each factor, twice integrate its semigroup derivative to obtain
$I+hG+h^2G^2/2$ with the third-order integral remainder. Multiply the five
expansions in their stated order. The linear coefficient is $A+B+O$.
For every pair of distinct generators, the symmetric half steps give the
coefficient $1/2$ for each ordered product; the squared-generator terms also
have coefficient $1/2$. The quadratic coefficient is therefore $L^2/2$.
The common-domain bounds control all terms of degree at least three and the
integral remainders. The same expansion for $e^{hL}$ proves the difference.
:::

:::{prf:lemma} Lie splitting and its commutator term
:label: lem-lie-splitting-weak-error

For generators $A,C$ on an analogous common domain,

$$
e^{hC}e^{hA}-e^{h(A+C)}
=\tfrac12h^2(CA-AC)+O(h^3).
$$

In particular the local weak defect is $O(h^2)$ when the displayed operators
and remainders have the required uniform bounds.
:::

:::{prf:proof}
Multiply
$(I+hC+h^2C^2/2)(I+hA+h^2A^2/2)$ and subtract
$I+h(A+C)+h^2(A+C)^2/2$. The surviving second-order term is
$h^2(CA-AC)/2$. Bound the higher terms on the common domain.
:::

:::{div} feynman-prose
The commutator records the difference between doing two operations in opposite orders. If one operation changes the state on which the next acts, that difference usually survives. A symmetric arrangement can cancel its leading contribution. Using a second-order kinetic method therefore does not automatically make a first-order composition with cloning second order; the complete sequence determines the error.
:::

:::{prf:theorem} Finite-time weak error from a local consistency bound
:label: thm-langevin-baoab-discretization-error

Let $T_h=e^{hL}$ and suppose the relevant propagated test functions satisfy
$\|(P_h-T_h)T_s\phi\|_V\leq K_T h^{p+1}$ for $0\leq s\leq T$.
If $\sup_{jh\leq T}\|P_h^j\|_{V\to V}\leq M_T$, then

$$
\|P_h^n\phi-T_{nh}\phi\|_V\leq M_TK_T T h^p,\qquad nh\leq T.
$$

The preceding BAOAB estimate gives $p=2$ on its regularity domain.
:::

:::{prf:proof}
Use the exact telescoping identity

$$
P_h^n-T_h^n=\sum_{j=0}^{n-1}P_h^j(P_h-T_h)T_h^{n-1-j}.
$$

There are $n$ terms, each bounded by $M_TK_T h^{p+1}$, and $nh\leq T$.
:::

:::{prf:theorem} Full split-system consistency
:label: thm-full-system-discretization-error

For a continuous-time cloning model with finite-rate generator $C$, a Lie
composition of its semigroup with the kinetic semigroup has finite-time weak
order one under {prf:ref}`lem-lie-splitting-weak-error` and the stability
hypotheses of {prf:ref}`thm-langevin-baoab-discretization-error`.
A symmetric composition has order two if its third-order common-domain
bounds hold.
:::

:::{prf:proof}
The Lie local defect is $O(h^2)$, so apply the telescoping theorem with $p=1$.
For symmetric splitting the ordered second-order products agree with the
expansion of the full semigroup, as in {prf:ref}`lem-baoab-weak-error`, and
$p=2$ applies. A fixed per-step cloning probability is not an $O(h)$ jump
rate; its transition family requires its own consistency estimate before
this continuous-generator argument applies.
:::

(sec-quantitative-stationary-errors)=
## Stationary and Quasi-Stationary Perturbations

:::{div} feynman-prose
Finite-time accuracy and equilibrium accuracy use different stability arguments. For equilibrium, a local perturbation is repeatedly carried forward by the dynamics. Mixing makes its influence decay, and summing that memory produces the factor $1/(1-r)$ below.

This factor also explains why physical time matters. As the step size decreases, one step represents less elapsed time, so its mixing factor approaches one. A claim of uniform contraction must specify a common physical duration; comparing one step at every step size can conceal this dependence.
:::

:::{prf:theorem} Drift and minorization for conservative mixing
:label: thm-meyn-tweedie-drift-minor

For a conservative kernel, the Lyapunov and common-small-set hypotheses of
{prf:ref}`thm-convergence-conservative-harris` imply a unique invariant law and
weighted geometric convergence. These hypotheses are imposed on the complete
transition kernel under study.
:::

:::{prf:proof}
Apply the weighted coupling proof of
{prf:ref}`thm-convergence-conservative-harris`. Its distance contracts outside
the small set by the drift inequality and inside by the common minorization.
The resulting strict contraction proves existence, uniqueness and geometric
convergence in the stated weighted space.
:::

:::{prf:lemma} Uniformity for a family of discretizations
:label: lem-uniform-geometric-ergodicity

If a family of conservative kernels has common constants in the preceding
Harris hypotheses for blocks of a fixed physical duration, its block mixing
constants are uniform over the family. Uniformity in $N$ requires the same
constants to be uniform in $N$ as well.
:::

:::{prf:proof}
The weighted coupling constants in the cited proof are functions only of the
drift and minorization constants. Uniform inputs give uniform contraction
and norm-equivalence constants. No uniformity follows from irreducibility
alone or from an $N$-dependent minorization.
:::

:::{prf:proposition} Relation between step and physical-time mixing rates
:label: prop-mixing-rate-relationship

A bound $\|P_h^n-\Pi_h\|\leq M e^{-\lambda nh}$ has per-step factor
$r_h=e^{-\lambda h}$. For $0<\lambda h\leq1$,

$$
\tfrac12\lambda h\leq1-r_h\leq\lambda h.
$$
:::

:::{prf:proof}
Use $1-e^{-x}\leq x$ and $e^{-x}\leq1-x+x^2/2\leq1-x/2$ for $0\leq x\leq1$.
:::

:::{prf:theorem} Propagation of a local operator error to invariant laws
:label: thm-quantitative-error-propagation

Let $P$ be a conservative kernel with invariant law $\pi$ and
$\|P^n\phi-\pi\phi\|_V\leq M r^n\|\phi\|_{\mathcal B}$ for $r<1$.
Let $Q$ have invariant law $\widetilde\pi$ with
$\widetilde\pi V<\infty$. Suppose the Poisson series
$u=\sum_{n\geq0}(P^n\phi-\pi\phi)$ is in the domain of $Q-P$ and
$|(Q-P)u|\leq\delta M\|\phi\|_{\mathcal B}V/(1-r)$. Then

$$
|\widetilde\pi\phi-\pi\phi|
\leq\frac{\delta M}{1-r}\|\phi\|_{\mathcal B}\widetilde\pi V.
$$
:::

:::{prf:proof}
The series converges in the weighted norm and solves
$(I-P)u=\phi-\pi\phi$. Invariance under $Q$ gives

$$
\widetilde\pi\phi-\pi\phi
=\widetilde\pi(I-P)u=\widetilde\pi(Q-P)u.
$$

Integrate the assumed local bound. Weighted integrability justifies the
identities by truncating the convergent series and taking the limit.
:::

:::{prf:lemma} Invariant-measure order from local weak order
:label: lem-baoab-invariant-measure-error

Under the preceding theorem, if $\delta_h\leq Kh^{p+1}$,
$r_h=e^{-\lambda h}$, and all other inputs are uniform, then
$|\pi_h\phi-\pi\phi|\leq C_\phi h^p$ for small $h$.
:::

:::{prf:proof}
Use $1-r_h\geq\lambda h/2$ in the perturbation bound. The local consistency
bound must hold for the Poisson solution; a finite-time weak estimate on a
different test class alone does not establish this hypothesis.
:::

:::{div} feynman-prose
For a killed process, we must also divide by the surviving mass. If survival is rare, a small error in the unnormalized transition can produce a much larger error after conditioning. The survival lower bound in the next lemma measures that amplification. The contraction argument then applies to the normalized evolution itself.
:::

:::{prf:lemma} Stability of QSDs under conditioned-block errors
:label: lem-quantitative-qsd-perturbation

Let $\Phi_P(\mu)=\mu P/\mu P1$ contract a metric $d$ with factor $r<1$.
Suppose $\pi=\Phi_P(\pi)$, $\widetilde\pi=\Phi_Q(\widetilde\pi)$, and
$\sup_\mu d(\Phi_Q\mu,\Phi_P\mu)\leq\delta$. Then

$$
d(\widetilde\pi,\pi)\leq\frac\delta{1-r}.
$$

For the total-variation norm, an unnormalized block error at most $\varepsilon$
and survival probabilities at least $s>0$ give $\delta\leq2\varepsilon/s$.
:::

:::{prf:proof}
Insert $\Phi_P(\widetilde\pi)$ between the two fixed points. The triangle
inequality gives $d(\widetilde\pi,\pi)\leq\delta+r d(\widetilde\pi,\pi)$.
For the normalization estimate write the difference of two positive measures
$a/A-b/B$ as $(a-b)/A+b(B-A)/(AB)$. Since $A\geq s$ and
$|A-B|\leq\|a-b\|_{\mathrm{TV}}$, its norm is at most $2\varepsilon/s$.
The estimate uses the actual conditioned maps, rather than treating a QSD as
an invariant measure of the unnormalized killed kernel.
:::

(sec-quantitative-total-error)=
## Combined Error for an Observable

:::{div} feynman-prose
We can now assemble the estimate without assigning one mechanism's rate to another. The target law supplies a population bias. The simulated stationary law supplies a sampling variance. A perturbation estimate compares those two stationary laws, and a mixing estimate compares the finite run with its own stationary law.

Each term suggests a different adjustment. More particles can reduce population and sampling errors under the uniform estimates. A smaller step can reduce discretization error. A longer run can reduce the remaining memory of initialization. The theorem specifies the hypotheses needed for each improvement.
:::

:::{prf:theorem} Total observable error
:label: thm-total-error-bound

Let $\pi_N$ be the target stationary or quasi-stationary joint law, $\pi_{N,h}$
the corresponding discretized law, and $\mu_{N,h,n}$ the law after $n$ steps
(with conditioning on survival when required). For the empirical observable
$\widehat\phi_N$, suppose:

- its bias under $\pi_N$ is at most $b_N$;
- its variance under $\pi_{N,h}$ is at most $v_{N,h}$;
- $|\pi_{N,h}\widehat\phi_N-\pi_N\widehat\phi_N|\leq d_h$;
- the difference between expectations under $\mu_{N,h,n}$ and $\pi_{N,h}$
  of $|\widehat\phi_N-\rho\phi|$ is at most $m_{N,h,n}$.

Then

$$
\mathbb E_{\mu_{N,h,n}}|\widehat\phi_N-\rho\phi|
\leq b_N+\sqrt{v_{N,h}}+d_h+m_{N,h,n}.
$$

For example the preceding results supply
$b_N=L\sqrt{2CH_N/N}$, $v_{N,h}=C_*L^2/N$ when the relevant law has that LSI,
and $d_h=O(h^p)$ under the stationary perturbation conditions.
:::

:::{prf:proof}
First transfer the expectation of the absolute-error observable to
$\pi_{N,h}$. Split its error at its expectation and apply Cauchy–Schwarz to
the centered term. Bound its expectation's difference from $\rho\phi$ by
$d_h+b_N$. Add the four terms.
:::

:::{prf:remark} Rate interpretation
:label: rem-rate-interpretation

The population, sampling, time-step and mixing terms refer to different
estimates. Their constants are uniform only over the regimes in which their
input bounds are uniform. A bounded observable admits the usual TV mixing
bound; an unbounded observable requires a weighted estimate and moments.
:::

:::{prf:remark} Higher-order splitting
:label: rem-higher-order-splitting

Symmetric splitting cancels the second-order local commutator term. Retaining
order two for an invariant law requires the common test domain, Poisson-solution
regularity, moment control and mixing estimates used above. Killing, projection
and fixed-probability cloning must be included in that verification.
:::

:::{prf:remark} Observable rates and empirical geometry
:label: rem-optimality-mean-field-rate

The $N^{-1/2}$ scale arises for a fixed observable with variance proportional
to $N^{-1}$. The empirical Wasserstein distance takes a supremum over spatial
tests and has dimension-dependent sampling behavior. A finite empirical
measure and a nonatomic density are mutually singular in total variation.
:::

:::{prf:proposition} Explicit dependence of the assembled constants
:label: prop-quantitative-explicit-constants

Under $H_N\leq D_*/\lambda_H$, full-gradient empirical LSI constant $C_*$,
local stationary defect $Kh^{p+1}$, physical mixing rate $\lambda_P$ and
uniform weighted moment $\pi_{N,h}V\leq M_V$, the non-transient part of the
preceding bound is at most

$$
\frac{L}{\sqrt N}\left(\sqrt{\frac{2CD_*}{\lambda_H}}+\sqrt{C_*}\right)
+\frac{2KM M_V}{\lambda_P}\|\widehat\phi_N\|_{\mathcal B}h^p.
$$

For a QSD discretization, replace the final term by the conditioned-block
perturbation estimate when that is the applicable stability theorem.
:::

:::{prf:proof}
Substitute the entropy and variance estimates into
{prf:ref}`thm-total-error-bound`, then use
{prf:ref}`prop-mixing-rate-relationship` and
{prf:ref}`thm-quantitative-error-propagation` for the stationary discretization
term. The constants are the constants of those explicit input estimates.
:::

:::{div} feynman-prose
The explicit constants make the estimate usable as a diagnostic. If one contribution dominates, reducing another may have little effect on the total error. The relevant constants also identify what must be measured or bounded before comparing regimes: the entropy residual, joint concentration constant, local operator defect, physical mixing rate, and required moments.
:::
