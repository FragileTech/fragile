---
title: "FractalAI: Research Lineage and Empirical Comparisons"
subtitle: "Optimization, planning, coordination, and analytical questions"
author: "Guillem Duran-Ballester"
---

(sec-fractalai-references)=
# FractalAI: Research Lineage and Empirical Comparisons

(sec-fractalai-tldr)=
(sec-fractalai-introduction)=
## 1. The questions connecting the papers

:::{div} feynman-prose
A population of simulated walkers can serve two purposes: search for a good point in a landscape, or explore what could happen after an action. Cloning reallocates that search effort. The literature below develops these ideas through optimization, planning, and a comparison with human coordination.

The comparisons and the convergence theorems answer different questions. A benchmark measures performance in its specified environment. A theorem explains the behavior of a specified transition law under explicit hypotheses. Connecting them requires identifying the algorithm, observable, and probability law used in both. This chapter follows that connection without turning a shared motivation into an identity of models.
:::

(sec-fractalai-timeline)=
:::{div} feynman-prose
The primary sources give the following sequence. Dates in the first three rows are the initial preprint years.

| Year | Source | Subject |
|:--|:--|:--|
| 2017 | [General Algorithmic Search](https://arxiv.org/abs/1705.08691) | Swarm optimization |
| 2018 | [Fractal AI: A Fragile Theory of Intelligence](https://arxiv.org/abs/1803.05049) | Simulated futures and decision policies |
| 2018 | [Solving Atari Games Using Fractals and Entropy](https://arxiv.org/abs/1807.01081) | Fractal Monte Carlo experiments |
| 2022 | [Modeling of Human Group Coordination](https://doi.org/10.1103/PhysRevResearch.4.023037) | Option-maximizing and learned-agent comparisons |

The algorithm definitions in {doc}`../1_the_algorithm/01_algorithm_intuition` and {doc}`../1_the_algorithm/02_fractal_gas_latent` specify the Fractal Gas studied in this book. Their parameters and update rules must be used when applying the convergence results.
:::

(sec-fractalai-gas-foundation)=
## 2. General Algorithmic Search

:::{div} feynman-prose
A swarm explores several candidate solutions at once. Replacing one walker by another redistributes computational effort toward selected regions; motion supplies further exploration. The useful mathematical questions are then about that combination of movement and replacement, not only the objective function being evaluated.

Hernández, Duran, and Amigó's 2017 preprint reports tests on 31 objective functions, comparing GAS with Basin Hopping, Cuckoo Search, and Differential Evolution. It also studies concurrent runs, where the first successful run determines completion. These are numerical results for the reported benchmark and protocol. [Primary paper](https://arxiv.org/pdf/1705.08691). {cite}`hernandez2017gas`
:::

:::{prf:definition} GAS population state and search objective
:label: def-gas-algorithm

In the GAS formulation, a walker carries a position $x_i$ and a positive internal flow $F_i$. The population searches for a global extremum of a scalar objective $f$. Its update combines stochastic motion, objective-dependent flow, cloning, and a tabu mechanism for previously found minima. This description identifies the historical method; its complete rules are given in Section 2 of the [GAS paper](https://arxiv.org/pdf/1705.08691). It does not define the later Boris–BAOAB Fractal Gas kernel.
:::

:::{prf:definition} Examples of benchmark geometry
:label: def-gas-benchmarks

The following standard objectives illustrate distinct features relevant to optimization benchmarks:

$$
f_{\mathrm{Sphere}}(x)=\sum_{i=1}^d x_i^2,
$$

$$
f_{\mathrm{Rosenbrock}}(x)=\sum_{i=1}^{d-1}
\left[100(x_{i+1}-x_i^2)^2+(x_i-1)^2\right],
$$

$$
f_{\mathrm{Rastrigin}}(x)=10d+\sum_{i=1}^d
\left[x_i^2-10\cos(2\pi x_i)\right],
$$

$$
f_{\mathrm{Ackley}}(x)=
-20\exp\!\left[-0.2\sqrt{d^{-1}\sum_i x_i^2}\right]
-\exp\!\left[d^{-1}\sum_i\cos(2\pi x_i)\right]+20+e.
$$

A numerical comparison also specifies the search domain, initialization, stopping tolerance, evaluation budget, and aggregation across runs. These formulas alone give neither a convergence rate nor a ranking of algorithms.
:::

(rb-gas-to-fractalai)=
:::{div} feynman-prose
Moving from static optimization to planning changes what a walker represents. A position in the optimizer becomes a simulated state reached through an action history. To choose a real action, the planner must retain how that history began. Cloning then changes both the explored states and the population representing each initial choice.
:::

(sec-fractalai-theory-framework)=
## 3. Simulated futures and Fractal Monte Carlo

:::{div} feynman-prose
The Fractal AI preprint distinguishes a policy that scans possible futures from a policy that chooses an action using that exploration. It includes reward-weighted future distributions and a walker implementation. This is a planning proposal with explicit objectives; it should not be reduced to an unqualified claim that all intelligence maximizes one entropy. [Primary preprint](https://arxiv.org/abs/1803.05049). {cite}`hernandez2018fractal`
:::

:::{prf:definition} A precise future-state entropy objective
:label: def-fsx-principle

Fix a state representation with finitely many cells $\mathcal S$, a horizon $\tau$, and a rollout policy after the first action. Let $p_a(s)$ be the resulting distribution over future cells when that first action is $a$. One possible future-state-maximization objective is

$$
a_*\in\operatorname*{arg\,max}_{a\in\mathcal A}
H(p_a),\qquad H(p)=-\sum_{s\in\mathcal S}p(s)\log p(s),
$$

with $0\log0=0$. The representation, horizon, and rollout policy are part of this definition. Reward weighting, path entropy, endpoint entropy, and counts of visited states specify different objectives unless an equivalence is proved.
:::

(rb-fsx-empowerment)=
:::{prf:remark} Which random quantity carries the entropy?

For finite random variables, endpoint entropy is $H(S_\tau)$, path entropy is $H(S_1,\ldots,S_\tau)$, and action–endpoint mutual information is

$$
I(A;S_\tau)=H(S_\tau)-H(S_\tau\mid A).
$$

The last quantity removes uncertainty remaining after the action is known. Maximizing it over an action distribution is a channel-capacity objective. The displayed identity shows why these objectives need not rank actions in the same way. No priority claim or equivalence of algorithms follows from their common use of information measures.
:::

:::{prf:definition} Historical FMC search loop
:label: def-fmc-algorithm

The 2018 Atari paper's FMC initializes walkers at the current state, advances them through simulated actions, evaluates relative rewards and companion distances, and recycles selected or dead walkers by cloning. After the computation budget is exhausted, the distribution of root actions among the remaining walkers determines action utilities. The paper's discrete and continuous action-selection conventions are specified in its algorithm. This is a population search with reward and diversity, not an algorithm defined solely by maximizing endpoint variance. [Algorithm, Section 4](https://arxiv.org/pdf/1807.01081).
:::

:::{div} feynman-prose
The Atari study reports 55 environments and comparisons with several learning and planning methods. Its efficiency claims concern the reported emulator-access and simulation protocols. Those comparisons motivate careful measurement of planning cost; they do not give a universal advantage over reinforcement learning or a proof of the Fractal Gas convergence hypotheses. [Atari experiments](https://arxiv.org/pdf/1807.01081). {cite}`hernandez2018atari`
:::

(rb-fractalai-vs-rl)=
:::{prf:remark} Planning and learning costs

A planner needs a way to simulate the proposed futures. A learned policy has a training cost and an evaluation cost; a planner has a simulation cost each time it replans. A comparison therefore specifies model access, training data, simulator calls, computational budget, and the evaluation distribution. Combining a learned model or policy with population planning is a separate algorithm whose errors include those of the learned component.
:::

:::{prf:proposition} Variance does not determine entropy
:label: prop-literature-variance-entropy

Two distributions can have equal variance and different discrete entropies. Hence endpoint variance is not an exact substitute for future-state entropy without further distributional restrictions.
:::

:::{prf:proof}

On the common state set $\{-\sqrt2,-1,0,1,\sqrt2\}$, let $p$ place probability $1/2$ at each of $-1,1$. Let $q$ place probabilities $1/4,1/2,1/4$ at $-\sqrt2,0,\sqrt2$. Both have mean zero and variance one, while $H(p)=\log2$ and $H(q)=\tfrac32\log2$.
:::

(sec-fractalai-hornischer-validation)=
## 4. The human coordination comparison

:::{div} feynman-prose
A behavioral comparison asks whether a specified decision rule reproduces measured group patterns. Similar outcomes do not identify the participants' internal computation.
:::

:::{prf:definition} Human coordination protocol
:label: def-hornischer-experiment

Hornischer and colleagues analyze 400 participants in 40 distinct groups of ten on a hexagonal game board. Players have 15 moves; reward fields favor joint arrival, and two players receive privileged reward information. The cognitive-force simulations use 25 moves. [Experimental design, Sections II–III](https://journals.aps.org/prresearch/pdf/10.1103/PhysRevResearch.4.023037). {cite}`hornischer2022modeling`
:::

:::{prf:definition} Cognitive-force decision rule in the comparison
:label: def-cognitive-force

For each neighboring field, the model simulates 5,000 hypothetical random walks. It selects the neighbor associated with the single walk visiting the most distinct fields. Reward fields extend hypothetical walks. Other agents remain fixed during this forecast. [Section III](https://journals.aps.org/prresearch/pdf/10.1103/PhysRevResearch.4.023037).
:::

:::{prf:definition} Reinforcement-learning comparison
:label: def-marl-baseline

The comparison uses an actor–critic method with a centralized critic and individual policies, trained for up to two million episodes. [Section IV](https://journals.aps.org/prresearch/pdf/10.1103/PhysRevResearch.4.023037).
:::

:::{div} feynman-prose
The reported group-size and destination distributions are reproduced more closely by the cognitive-force model than by that RL implementation. The paper also describes differences in movement dynamics and difficulties in learning cooperative behavior. This supports the tested option-maximization model in that environment. [Results and discussion](https://journals.aps.org/prresearch/pdf/10.1103/PhysRevResearch.4.023037).
:::

(sec-literature-observables)=
## 5. Defining a reproducible comparison

:::{div} feynman-prose
Before comparing two curves, decide what counts as a discrepancy. A success fraction ignores the route taken. A trajectory comparison depends on whether timing matters. An entropy measures the spread of a specified distribution, while a model-evidence ratio also depends on its likelihood and parameter priors.

The definitions below are available for future comparisons. They do not attribute unreported measurements or numerical results to the human study.
:::

:::{prf:definition} Trajectory dissimilarities
:label: def-trajectory-distance

For finite trajectories $x_1,\ldots,x_m$ and $y_1,\ldots,y_n$ in a metric space with distance $d$, let $\mathcal W$ be the paths from $(1,1)$ to $(m,n)$ with allowed steps $(1,0),(0,1),(1,1)$. One dynamic-time-warping convention is

$$
\operatorname{DTW}(x,y)=\min_{W\in\mathcal W}\sum_{(i,j)\in W}d(x_i,y_j).
$$

It is a dissimilarity; the formula is not asserted to satisfy every metric axiom. The discrete Fréchet distance replaces the sum by the maximum:

$$
d_{\mathrm{dF}}(x,y)=\min_{W\in\mathcal W}\max_{(i,j)\in W}d(x_i,y_j).
$$

Sampling, allowed alignments, time normalization, and any path-length normalization must be fixed before reporting either quantity.
:::

:::{prf:definition} Action entropy
:label: def-action-entropy

For a specified finite action alphabet $\mathcal A$ and probability vector $p$, define

$$
H(p)=-\sum_{a\in\mathcal A}p(a)\log p(a),\qquad 0\log0=0.
$$

An empirical estimate replaces $p$ by observed frequencies over a stated sampling unit and time window. The resulting action entropy is distinct from the entropy of a stationary particle law or of a future-state distribution.
:::

:::{prf:proposition} Entropy range for a finite action alphabet
:label: prop-literature-action-entropy-range

If $m=|\mathcal A|$, then $0\leq H(p)\leq\log m$.
:::

:::{prf:proof}

Each summand is nonnegative. For the uniform law $u(a)=1/m$, relative entropy gives $D(p\Vert u)=\log m-H(p)\geq0$. To verify the last inequality directly, use $\log t\leq t-1$ with $t=u(a)/p(a)$ on the support of $p$ and sum: $\sum p(a)\log[u(a)/p(a)]\leq\sum_{p(a)>0}u(a)-1\leq0$.
:::

:::{prf:definition} Bayes factor with specified likelihoods and priors
:label: def-bayes-factor

For data $D$ and models $M_1,M_2$, define the marginal likelihoods

$$
m_j(D)=\int p(D\mid\theta_j,M_j)\Pi_j(d\theta_j),
\qquad BF_{12}=\frac{m_1(D)}{m_2(D)},
$$

when both integrals are finite and the denominator is positive. The likelihood includes the dependence structure of the observations, and $\Pi_j$ is the parameter prior. Posterior model odds equal $BF_{12}$ times prior model odds. A numerical Bayes factor requires these inputs and an evidence calculation; similarity of plotted distributions does not determine one.
:::

(sec-fractalai-volume3-integration)=
## 6. Connecting measurements to the convergence theory

:::{div} feynman-prose
The Fractal Gas analysis supplies precise statements about survival-conditioned laws, stationary population equations, and empirical observables. To use one in an experiment, first identify which of these objects is being measured. A finite game ending at a goal is not automatically a sample from a QSD, and a change in a task's success rate with group size is not automatically a mean-field sampling error.

The two results below state concrete connections that can be checked. The first concerns a normalized killed kernel. The second compares a finite-particle marginal with a specified limiting law on the same state space.
:::

:::{prf:theorem} QSD convergence under conditioned-block contraction
:label: thm-qsd-existence

Let $Q_N$ be a killed finite-particle kernel. For a fixed integer $m\geq1$, suppose $Q_N^m1(x)>0$ and the normalized block map

$$
\Phi_m(\mu)=\frac{\mu Q_N^m}{\mu Q_N^m1}
$$

contracts total-variation distance on probability measures with factor $r_N<1$. Then $Q_N$ has a unique QSD $\pi_N$, and

$$
\left\|\frac{\mu Q_N^{mj}}{\mu Q_N^{mj}1}-\pi_N\right\|_{\mathrm{TV}}
\leq r_N^j\|\mu-\pi_N\|_{\mathrm{TV}}.
$$

If one algorithm step represents physical duration $h$, the block rate is $-\log r_N/(mh)$. Proved sufficient full-kernel hypotheses for such normalized contraction are given in {doc}`../convergence_program/06_convergence` and {prf:ref}`prop-latent-fractal-gas-conditional-qsd`.
:::

:::{prf:proof}

Probability measures form a complete space in total variation. Banach's fixed-point theorem gives a unique fixed point $\pi_N$ of $\Phi_m$ and the displayed iterative bound. Positivity of $Q_N^m1$ implies positivity of $Q_N1$, so the normalized one-step map $\Phi_1$ is defined. Kernel powers commute, hence $\Phi_m\Phi_1=\Phi_1\Phi_m$. It follows that $\Phi_1(\pi_N)$ is also a fixed point of $\Phi_m$, and uniqueness gives $\Phi_1(\pi_N)=\pi_N$. Thus $\pi_NQ_N=\alpha_N\pi_N$ with $\alpha_N=\pi_NQ_N1>0$. Conversely every QSD is fixed by $\Phi_m$, proving uniqueness for the one-step kernel. Taking logarithms of $r_N^j$ gives the physical rate.
:::

:::{prf:theorem} Marginal and observable mean-field errors
:label: thm-mean-field-error

Let $\rho$ be the specified limiting probability law on Euclidean phase space, satisfying full-gradient LSI with constant $C$. Let $\pi_N$ be exchangeable, with

$$
H_N=D_{\mathrm{KL}}(\pi_N\Vert\rho^{\otimes N})<\infty.
$$

Then its one-particle marginal obeys

$$
W_2(\pi_{N,1},\rho)\leq\sqrt{\frac{2CH_N}{N}}.
$$

For an $L$-Lipschitz observable $\varphi$, if
$\operatorname{Var}_{\pi_N}(N^{-1}\sum_i\varphi(Z_i))\leq A_\varphi/N$, then

$$
\mathbb E_{\pi_N}\left|\frac1N\sum_i\varphi(Z_i)-\rho\varphi\right|
\leq L\sqrt{\frac{2CH_N}{N}}+\sqrt{\frac{A_\varphi}{N}}.
$$

Uniform bounds on $H_N$ and $A_\varphi$ give the stated $N^{-1/2}$ scale for that observable. Identifying $\rho$ with a stationary mean-field law uses the consistency, tightness, and uniqueness or attraction arguments in {doc}`../convergence_program/09_propagation_chaos`.
:::

:::{prf:proof}

The product-reference transport argument and symmetric coupling of coordinates in {prf:ref}`lem-wasserstein-entropy` give the marginal inequality. Split the empirical observable at its expectation. Its bias is at most $L W_2(\pi_{N,1},\rho)$, and Cauchy–Schwarz bounds its centered fluctuation by $\sqrt{A_\varphi/N}$. This is {prf:ref}`thm-quantitative-propagation-chaos`. For bounded observables, including indicators, the direct relative-entropy concentration proof is {prf:ref}`thm-mixing-variance-corrected`. An empirical Wasserstein estimate additionally retains the independent-sampling term of {prf:ref}`prop-empirical-wasserstein-concentration`.
:::

(sec-fractalai-conclusions)=
### Measurements that test a stated model

:::{prf:definition} A comparison protocol tied to analytical hypotheses
:label: def-predictions

For a specified algorithm and parameter family, record:

1. The complete transition law, including companion selection, cloning, kinetics, killing, and any conditioning.
2. The observable, its target law, the time convention, initialization, and the population sizes being compared.
3. The constants and uniformity hypotheses in the selected convergence or concentration theorem.
4. Sampling uncertainty, discretization error, and the transient mixing contribution, as separated in {prf:ref}`thm-total-error-bound`.

A contraction theorem provides a sufficient decay bound. It does not assert that every measured observable attains that exponent. An entropy-convergence comparison uses the actual-law LSI and full modified dissipation estimate of {prf:ref}`thm-kl-convergence-euclidean`; changing friction alone does not specify those constants.
:::

:::{div} feynman-prose
The research lineage supplies algorithms to compare and behavioral patterns to explain. The analytical chapters supply proofs for identified models. A useful next experiment makes that identification explicit, then tests the resulting observable bound or equilibrium prediction. This leaves a direct route from an implemented update to a measurable claim, without inferring a universal law of human intelligence from one coordination task.
:::

```{bibliography}
:filter: docname in docnames
:style: unsrt
```
