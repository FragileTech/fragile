# Exchangeability, Empirical Laws, and Functional Inequalities

:::{div} feynman-prose
**TLDR.** A permutation-symmetric transition with a unique QSD has an
exchangeable QSD: relabeling walkers leaves its law unchanged. At finite $N$,
a few walkers can be approximated by independent samples from the random
empirical measure, with an explicit sampling error. When that empirical
measure concentrates on a deterministic law, the approximation becomes
propagation of chaos. Entropy and logarithmic Sobolev estimates provide
quantitative control when their stated joint-law hypotheses hold.

Keep three questions separate. Does changing the labels change the law? Does
the empirical distribution vary substantially between runs? Does the joint
law satisfy a functional inequality with a constant independent of $N$?
Symmetry answers the first question. Concentration answers the second. The
analytical criteria in {doc}`15_kl_convergence` answer the third for their
specified reference and interacting laws.

The finite-particle QSD construction is developed in {doc}`06_convergence`.
The mean-field existence, uniqueness, and convergence arguments are in
{doc}`09_propagation_chaos`. This chapter proves the probabilistic steps that
connect those results to marginals, empirical observables, and limiting
functional inequalities.
:::

(sec-exchangeability-qsd)=
## Symmetry of the Killed Kernel

:::{div} feynman-prose
Imagine exchanging the labels of two walkers before an update and exchanging
them back afterward. Equivariance says the resulting probability law is the
same. If there is only one QSD, its relabeled version must be that same QSD.
The proof is short because uniqueness does the final step; the survival
factor remains in the eigenmeasure equation throughout.
:::

:::{prf:theorem} Exchangeability of a unique QSD
:label: thm-qsd-exchangeability

Let $E$ be a standard Borel one-particle state space, including any status
variables. Let $Q_N$ be a sub-Markov kernel on the surviving configurations in
$E^N$. Suppose it is permutation-equivariant:

$$
Q_N(\sigma S,\sigma A)=Q_N(S,A),\qquad \sigma\in\mathfrak S_N.
$$

If it has a unique QSD $\pi_N$, then $\sigma_\#\pi_N=\pi_N$ for every
permutation $\sigma$.
:::

:::{prf:proof}
Write $\pi_NQ_N=\alpha_N\pi_N$, with $\alpha_N=\pi_NQ_N1>0$.
For every measurable $A$, equivariance gives

$$
((\sigma_\#\pi_N)Q_N)(A)
=\int Q_N(\sigma S,A)\,\pi_N(dS)
=(\pi_NQ_N)(\sigma^{-1}A)
=\alpha_N(\sigma_\#\pi_N)(A).
$$

Thus the permuted law is another QSD, and uniqueness gives equality. This
argument uses the QSD eigenmeasure equation, including its survival factor.
:::

:::{div} feynman-prose
The labels must be irrelevant to the complete experiment. This includes the
order used to resolve collisions or ties, as well as the one-particle motion.
When a rule processes walkers sequentially, its ordering convention belongs
in the symmetry check.
:::

:::{prf:remark} Equivariance is a property of the complete update
:label: rem-exchangeability-complete-kernel

The symmetry condition must include companion selection, collisions, status
updates and all tie-breaking rules. Identical single-particle equations alone
are insufficient. In particular, sequential processing of overlapping
collision groups can introduce a label-order dependence; the operational
conventions are described in {doc}`../1_the_algorithm/02_fractal_gas_latent`.
:::

(sec-exchangeability-finite-mixtures)=
## Finite Exchangeable Laws and Empirical Mixtures

:::{div} feynman-prose
An exchangeable population can be strongly dependent. For example, set every
walker equal to one shared random state $Y$. Relabeling changes nothing, but
observing one walker determines all the others. Its empirical measure is the
random point mass $\delta_Y$.

The useful finite-population statement is therefore a mixture statement.
First draw the entire cloud. Then sample from its empirical measure. These
samples are independent conditional on the cloud, while the cloud itself
remains random. The next proof compares sampling labels with and without
replacement.
:::

:::{prf:theorem} Finite empirical-mixture approximation
:label: thm-hewitt-savage-representation

For an exchangeable probability law $\pi_N$ on $E^N$, put
$L_N=N^{-1}\sum_i\delta_{Z_i}$ and $\mathcal Q_N=\operatorname{Law}_{\pi_N}(L_N)$.
For $1\leq k\leq N$,

$$
d_{\mathrm{TV}}\!\left(\pi_{N,k},\int\mu^{\otimes k}\mathcal Q_N(d\mu)\right)
\leq1-\frac{(N)_k}{N^k}
\leq\min\!\left(1,\frac{k(k-1)}{2N}\right),
$$

where $(N)_k=N(N-1)\cdots(N-k+1)$ and
$d_{\mathrm{TV}}(P,Q)=\sup_A|P(A)-Q(A)|$.
:::

:::{prf:proof}
Conditional on the configuration, sampling $k$ independent uniform indices
with replacement gives law $L_N^{\otimes k}$. Sampling ordered distinct
indices uniformly gives the conditional sampling-without-replacement law.
After averaging the latter over an exchangeable configuration, the law is
exactly $\pi_{N,k}$.

Couple the two index lists to agree whenever the with-replacement list has no
repeated indices; on the collision event sample an independent uniform
ordered list of distinct indices for the second list. Its distribution is
uniform in both cases, so this is a valid coupling. The disagreement
probability is at most the collision probability
$1-(N)_k/N^k$. A union bound over index pairs gives $k(k-1)/(2N)$.
Neither compactness nor projective consistency is required.
:::

:::{div} feynman-prose
The error counts repeated labels in the with-replacement sample. Two different
walkers may occupy the same state, which causes no problem: the coupling is
about their indices. For one selected walker there is no repeated-index event,
so the mixture identity is exact. For several selected walkers, the finite
sampling correction remains.
:::

:::{prf:definition} Single-particle marginal
:label: def-single-particle-marginal

The one-particle marginal is
$\pi_{N,1}(A)=\pi_N\{Z_1\in A\}$ for measurable $A\subset E$.
:::

:::{prf:proposition} Exact barycenter of the empirical mixing law
:label: prop-marginal-mixture

For every bounded measurable $g$,

$$
\pi_{N,1}g=\int(\mu g)\,\mathcal Q_N(d\mu).
$$
:::

:::{prf:proof}
The right-hand side is $\mathbb E_{\pi_N}L_Ng
=N^{-1}\sum_i\mathbb E g(Z_i)=\mathbb E g(Z_1)$ by exchangeability.
For $k>1$, the empirical product mixture is an approximation with the error
in {prf:ref}`thm-hewitt-savage-representation`.
:::

(sec-exchangeability-chaos)=
## From Empirical Concentration to Propagation of Chaos

:::{div} feynman-prose
Now suppose the cloud's empirical distribution settles near one deterministic
law $\rho_0$. The randomness in the mixing measure then disappears. A fixed
number of sampled walkers behaves in the limit like independent draws from
$\rho_0$. This is the content of propagation of chaos here; it concerns fixed
marginals as $N$ grows.

The shared-state example shows why deterministic concentration matters. Its
empirical measure remains $\delta_Y$, so the limiting mixture retains the
common random state. Exchangeability alone has no mechanism for removing that
source of dependence.
:::

:::{prf:theorem} Deterministic empirical limits imply chaos
:label: thm-propagation-chaos-qsd

Let $E$ be Polish and let $\pi_N$ be exchangeable. If $L_N$ converges weakly
in probability to a deterministic law $\rho_0$, then, for every fixed $k$,

$$
\pi_{N,k}\Longrightarrow\rho_0^{\otimes k}.
$$

For QSDs, the analytic hypotheses yielding this empirical concentration and
identifying $\rho_0$ are developed in {doc}`09_propagation_chaos`.
:::

:::{prf:proof}
For bounded continuous $F:E^k\to\mathbb R$, the map
$\mu\mapsto\int F\,d\mu^{\otimes k}$ is bounded and continuous for weak
convergence on a Polish space. Convergence in probability to a constant and
boundedness imply convergence of its expectations. The finite-mixture
approximation changes this expectation by at most
$2\|F\|_\infty k(k-1)/(2N)$, which tends to zero. This proves the result.
:::

:::{div} feynman-prose
To see the finite-$N$ correction quantitatively, expand the covariance of two
empirical averages. Some terms use the same walker twice; the others use two
distinct walkers. Symmetry makes each class uniform, but their coefficients
are different. Keeping the diagonal terms is what produces the correction in
the following identity.
:::

:::{prf:theorem} Exact covariance identity for finite populations
:label: thm-correlation-decay

For an exchangeable law and square-integrable $g,h$, write
$\overline g=L_Ng$ and $\overline h=L_Nh$. For $N\geq2$,

$$
\operatorname{Cov}(g(Z_1),h(Z_2))
=\frac{N\operatorname{Cov}(\overline g,\overline h)
-\operatorname{Cov}(g(Z_1),h(Z_1))}{N-1}.
$$

Consequently, if
$\operatorname{Var}(\overline g)\leq A_g/N$ and
$\operatorname{Var}(\overline h)\leq A_h/N$, then

$$
|\operatorname{Cov}(g(Z_1),h(Z_2))|
\leq\frac{\sqrt{A_gA_h}
+\sqrt{\operatorname{Var}(g(Z_1))\operatorname{Var}(h(Z_1))}}{N-1}.
$$
:::

:::{prf:proof}
Expand the covariance of the two empirical averages into $N^2$ terms.
There are $N$ diagonal terms and $N(N-1)$ off-diagonal terms. Exchangeability
makes the terms within each class equal. Rearrangement gives the identity,
and Cauchy–Schwarz gives the bound. In particular
$\operatorname{Var}_{\mathcal Q_N}(\mu g)=\operatorname{Var}(L_Ng)$;
it is not equal to the distinct-particle covariance in a finite population.
:::

(sec-exchangeability-entropy-concentration)=
## Bounded Observables from Relative Entropy

:::{div} feynman-prose
Under a product reference law, an empirical average has the familiar small
fluctuations of independent sampling. Relative entropy measures how much a
joint law can change the expectation of an exponentially controlled observable.
The proof applies that principle to the square of the empirical error.

The entropy budget here is the total joint entropy $H_N$. A bound independent
of $N$ gives the displayed $1/N$ rate. If only $H_N/N$ tends to zero, the same
formula gives vanishing fluctuations with the rate it actually states. This
estimate applies to bounded measurable observables, including status
indicators when the joint entropy hypothesis holds.
:::

:::{prf:theorem} Empirical variance from a product-reference entropy bound
:label: thm-mixing-variance-corrected

Let $H_N=D_{\mathrm{KL}}(\pi_N\Vert\rho_0^{\otimes N})<\infty$ and
$|g|\leq B$. Then

$$
\operatorname{Var}_{\mathcal Q_N}(\mu g)
\leq\mathbb E_{\pi_N}|L_Ng-\rho_0g|^2
\leq\frac{4B^2}{N}\left(H_N+\frac12\log2\right).
$$

For exchangeable $\pi_N$ this implies, for $i\ne j$,

$$
|\operatorname{Cov}_{\pi_N}(g(Z_i),g(Z_j))|
\leq\frac{B^2}{N-1}
\left[4\left(H_N+\frac12\log2\right)+1\right].
$$

In particular, a uniform bound on the total relative entropy gives an
$O(N^{-1})$ covariance estimate for all bounded measurable observables.
:::

:::{prf:proof}
Let $Q=\rho_0^{\otimes N}$ and $F=L_Ng-\rho_0g$.
Independence and Hoeffding's lemma give
$\mathbb E_Qe^{tF}\leq e^{t^2B^2/(2N)}$.
Introduce an independent standard Gaussian $G$. Its moment generating function
and Tonelli's theorem yield

$$
\mathbb E_Qe^{NF^2/(4B^2)}
=\mathbb E_G\mathbb E_Qe^{\sqrt{N/(2B^2)}GF}
\leq\mathbb E_Ge^{G^2/4}=\sqrt2.
$$

For completeness, the entropy inequality
$\mathbb E_P A\leq D_{\mathrm{KL}}(P\Vert Q)+\log\mathbb E_Qe^A$
follows from nonnegativity of
$D_{\mathrm{KL}}(P\Vert e^AQ/\mathbb E_Qe^A)$.
Apply it with $A=NF^2/(4B^2)$ to obtain the second-moment bound. Variance is
bounded by squared error about any constant, giving the first inequality.
The covariance bound follows from
{prf:ref}`thm-correlation-decay` and $\operatorname{Var}(g(Z_1))\leq B^2$.
For $B=0$ every assertion is immediate.
:::

(sec-exchangeability-lsi)=
## Joint Logarithmic Sobolev Inequalities

:::{div} feynman-prose
A logarithmic Sobolev inequality controls entropy through a gradient cost.
That cost and the reference law must belong to the same problem. The criteria
below identify joint laws for which the constant remains controlled as more
particles are added. Symmetry then helps interpret their marginals and
covariances.

The gradient is the full continuous phase-space gradient. Discrete alive or
dead statuses require their own entropy control: a function depending only on
status has zero continuous gradient while its values can still fluctuate.
:::

:::{prf:theorem} N-uniform LSI under the analytical joint-law criteria
:label: thm-n-uniform-lsi-exchangeable

Let $\pi_N$ be continuous joint laws satisfying one of the analytical
criteria in {prf:ref}`cor-n-uniform-lsi`: the product kinetic reference,
a uniformly bounded joint density tilt of that reference, or an
$N$-uniform positive joint curvature bound. Then

$$
D_{\mathrm{KL}}(\nu\Vert\pi_N)
\leq\frac{C_*}{2}I(\nu\Vert\pi_N),\qquad
I(\nu\Vert\pi_N)=\int|\nabla\log(d\nu/d\pi_N)|^2\,d\nu,
$$

with $C_*$ independent of $N$. The same theorem applies to a QSD when the
criterion holds for that QSD. Exchangeability supplies the marginal and
covariance identities above; it is not itself an LSI criterion. For a law
with discrete status variables, the status entropy must also be controlled
as specified in {doc}`15_kl_convergence`.
:::

:::{prf:proof}
The complete proofs of the joint-law criteria are in
{doc}`15_kl_convergence`. Their conclusion is
$\operatorname{Ent}_{\pi_N}(f^2)\leq2C_*\int|\nabla f|^2d\pi_N$.
Set $f=\sqrt{d\nu/d\pi_N}$, initially smooth and positive. Then
$4\int|\nabla f|^2d\pi_N=I(\nu\Vert\pi_N)$, giving the displayed convention.
Truncation and approximation extend it to its functional domain.
:::

:::{div} feynman-prose
The Gaussian calculation identifies a concrete reference law. Friction
attenuates the initial velocity, and accumulated independent noise produces a
Gaussian covariance approaching $\sigma^2/(2\gamma)$. The factorized kinetic
reference inherits that velocity law directly. Identifying an interacting
stationary law or QSD requires checking its complete stationary or eigenmeasure
equation.
:::

:::{prf:lemma} Gaussian velocity reference and frozen OU covariance
:label: lem-conditional-gaussian-qsd-euclidean

For the conservative kinetic reference
$m_U(dx,dv)\propto e^{-[U(x)+|v|^2/2]/\theta}\,dx\,dv$,
$\theta=\sigma^2/(2\gamma)$, the velocity conditional law is
$\mathcal N(0,\theta I_d)$, independently of $x$.
The product reference $m_U^{\otimes N}$ has conditionally independent
velocities with the same covariance bound for every $N$.
:::

:::{prf:proof}
Factor the Gibbs density into its spatial and velocity factors. Equivalently,
for the centered frozen OU process $dV=-\gamma V\,dt+\sigma dW$,

$$
V_t=e^{-\gamma t}V_0+\sigma\int_0^te^{-\gamma(t-s)}dW_s.
$$

The stochastic integral is Gaussian with covariance
$\sigma^2(1-e^{-2\gamma t})I_d/(2\gamma)$, which converges to
$\theta I_d$. Independent noises yield the product frozen reference.
Identification with an interacting invariant law or QSD requires its own
stationary or eigenmeasure equation; conditioning an evolving kinetic system
on its current positions does not freeze the spatial dynamics.
:::

:::{div} feynman-prose
The last step uses a simple test function: let a function of the whole swarm
look at just its first walker. Every other coordinate gradient vanishes. The
joint inequality therefore gives the same inequality for that marginal.
Smooth compactly supported tests then allow every integral to pass to the weak
limit with the same constant.
:::

:::{prf:corollary} LSI passes to a weak marginal limit
:label: cor-mean-field-lsi

Suppose $\pi_N$ satisfies the full-gradient LSI with constant $C_*$ above,
and its one-particle marginals converge weakly to $\rho_0$ on Euclidean phase
space. Then, for every smooth compactly supported $g$,

$$
\operatorname{Ent}_{\rho_0}(g^2)
\leq2C_*\int|\nabla g|^2\,d\rho_0.
$$
:::

:::{prf:proof}
Test the joint inequality with $f(z_1,\ldots,z_N)=g(z_1)$.
Only the first-coordinate gradient survives, so the same inequality holds
for $\pi_{N,1}$. The functions $g^2$, $g^2\log g^2$ (with $0\log0=0$)
and $|\nabla g|^2$ are bounded and continuous. Weak convergence passes each
integral to the limit. The normalization term
$(\int g^2)\log(\int g^2)$ is continuous as well. This proves the inequality
on the stated core; closure extends it whenever that core is dense in the
corresponding Dirichlet domain.
:::

:::{div} feynman-prose
The resulting chain has separate, checkable links: symmetry of the complete
kernel gives exchangeability of the unique QSD; concentration of the empirical
law gives fixed-marginal independence in the limit; and a uniform joint
functional inequality passes to the limiting marginal. The sampling,
covariance, and entropy formulas retain the finite-$N$ errors between these
steps.
:::

