# Continuum Hypotheses Discharge (A1-A6)

## 0. TLDR

This appendix replaces the Fractal Gas Continuum Hypotheses A1-A6 from
{prf:ref}`assm-fractal-gas-nonlocal` with internal lemmas, each pinned to
Volume 3 certificates and appendices. The result is a self-contained discharge
chain for the nonlocal d'Alembertian and action limit in
{doc}`../2_fractal_set/02_causal_set_theory`.

**Dependencies**: {doc}`../2_fractal_set/02_causal_set_theory`, {doc}`../1_the_algorithm/02_fractal_gas_latent`, {doc}`../3_fitness_manifold/01_emergent_geometry`, {doc}`07_discrete_qsd`, {doc}`08_mean_field`, {doc}`09_propagation_chaos`, {doc}`10_kl_hypocoercive`, {doc}`12_qsd_exchangeability_theory`, {doc}`13_quantitative_error_bounds`, {doc}`15_kl_convergence`, {doc}`03_cloning`, {doc}`06_convergence`, {doc}`01_fragile_gas_framework`

## 1. Target Hypotheses (from {prf:ref}`assm-fractal-gas-nonlocal`)

We discharge the following:

- **A1 (Geometry)**: globally hyperbolic Lorentzian manifold with $g=-c^2dt^2+g_R$,
  $g_R$ $C^4$ and uniformly elliptic.
- **A2 (Smooth fields)**: $U_{\mathrm{eff}}(x,t)$, $r(t)$, $Z(t)$, $g_R(x,t)$ are
  $C^4$ with bounded derivatives on the window.
- **A3 (QSD sampling)**: stationarity and ergodicity with QSD density
  $\rho_{\mathrm{adaptive}} \propto \sqrt{\det g_R}\,e^{-U_{\mathrm{eff}}/T}$.
- **A4 (Mixing)**: LSI with constant $\kappa>0$ on the window; LLN for bounded
  Lipschitz functionals.
- **A5 (Kernel)**: $K \in C^2_c([0,1])$ with moment conditions $M_0=0$ and
  $M_2^{\mu\nu}=2m_2 g^{\mu\nu}$.
- **A6 (Scaling)**: $\varepsilon\to 0$, $N\to\infty$, and $N\varepsilon^{D+4}\to\infty$.



## 2. A2: Smooth Fields from QSD + Algorithmic Smoothing

:::{prf:lemma} Discharge of A2 (Smooth fields)
:label: lem-continuum-a2-smooth-fields

On any finite window $[t_0,t_1]$, the fields $U_{\mathrm{eff}}(x,t)$, $r(t)$,
$Z(t)$, and the emergent metric $g_R(x,t)$ are $C^4$ with bounded derivatives.
:::

:::{prf:proof}
**Step 1: Smoothness of the QSD density.**
By {doc}`06_convergence` and {doc}`09_propagation_chaos`, the mean-field QSD
$\rho_0$ exists and is unique. The kinetic operator satisfies H\"ormander's
condition (Theorem {prf:ref}`thm-uniqueness-hormander` and Lemma
{prf:ref}`lem-uniqueness-hormander-verification` in {doc}`09_propagation_chaos`),
so the stationary equation has hypoelliptic regularity. Thus $\rho_0$ is smooth
on the alive core; smoothness can be bootstrapped because the coefficients are
smooth and the cloning term is an integral operator with smooth kernel
(Gaussian jitter from {doc}`03_cloning`, {doc}`02_euclidean_gas`).

**Step 2: $U_{\mathrm{eff}}$ from smoothed measures.**
The effective potential is defined by the decorated Gibbs envelope
({prf:ref}`thm-decorated-gibbs` in {doc}`07_discrete_qsd`) and the QSD density
used in {prf:ref}`def-cst-volume`. The measurement/standardization pipeline is
built from $C^\infty$ primitives with Gaussian smoothing (see
{prf:ref}`def-smoothed-gaussian-measure` in {doc}`01_fragile_gas_framework` and
the execution certificate in {doc}`../1_the_algorithm/02_fractal_gas_latent`).
Therefore $U_{\mathrm{eff}}$ inherits $C^\infty$ regularity on the alive core;
in particular, $C^4$.

**Step 3: $g_R$ from the adaptive diffusion tensor.**
The emergent metric is defined by the adaptive diffusion tensor
({prf:ref}`def-adaptive-diffusion-tensor-latent`). Lipschitz continuity of
$\Sigma_{\mathrm{reg}}$ follows from
{prf:ref}`prop-lipschitz-diffusion-latent` in
{doc}`../3_fitness_manifold/01_emergent_geometry`. Because $V_{\mathrm{fit}}$ is
$C^\infty$ on the alive core (Step 2), the Hessian map is $C^2$ and the inverse
square root is smooth on uniformly elliptic matrices, so $g_R$ is $C^4$.

**Step 4: $r(t)$ and $Z(t)$ as smooth marginals.**
The episode rate $r(t)$ and the normalizer $Z(t)$ are time marginals of the QSD
density (Section 2.2 of {doc}`../2_fractal_set/02_causal_set_theory`). With
smooth $\rho_0$ and mass conservation
({prf:ref}`thm-mass-conservation` in {doc}`08_mean_field`), differentiation
under the integral sign gives $r, Z \in C^4([t_0,t_1])$.

All four fields are therefore $C^4$ with bounded derivatives on the window.
:::



## 3. A1: Geometry and Global Hyperbolicity

:::{prf:lemma} Discharge of A1 (Emergent Lorentzian geometry)
:label: lem-continuum-a1-geometry

The continuum lift of the Fractal Set yields a globally hyperbolic Lorentzian
manifold $M=[t_0,t_1]\times\mathcal{X}$ with metric
$g=-c^2dt^2+g_R$ where $c=V_{\mathrm{alg}}$ and $g_R$ is $C^4$ and uniformly
elliptic.
:::

:::{prf:proof}
**Step 1: Continuum lift from QSD.**
Propagation of chaos ({prf:ref}`thm-propagation-chaos-qsd`) gives convergence of
single-particle marginals to the mean-field limit of
{prf:ref}`thm-mean-field-equation`, furnishing a deterministic continuum
distribution on each time slice.

**Step 2: Spatial metric from the algorithmic diffusion tensor.**
Define $g_R$ via {prf:ref}`def-adaptive-diffusion-tensor-latent` and the induced
Riemannian structure certificate ({prf:ref}`thm:induced-riemannian-structure`)
from {doc}`../1_the_algorithm/02_fractal_gas_latent`. Uniform ellipticity is
guaranteed by the diffusion floor $\epsilon_\Sigma$.

**Step 3: Continuum injection.**
Apply {prf:ref}`mt:continuum-injection`, {prf:ref}`mt:emergent-continuum`, and
{prf:ref}`mt:cheeger-gradient` to identify the IG distance with the geodesic
distance of $(\mathcal{X}, g_R)$ in the limit.

**Step 4: Lorentzian structure from CST order.**
Use the causal order equivalence
({prf:ref}`prop-fractal-causal-order-equivalence`) and the speed limit
construction in {prf:ref}`def-fractal-causal-order` to define
$g=-c^2dt^2+g_R$, with CST edges timelike and IG edges spacelike by construction.

**Step 5: Global hyperbolicity.**
The window $[t_0,t_1]$ supplies a Cauchy foliation by constant-$t$ slices. The
confining envelope from the decorated Gibbs structure
({prf:ref}`thm-decorated-gibbs`) and Safe Harbor barriers
({doc}`03_cloning`, {doc}`07_discrete_qsd`) make causal diamonds compact on the
window, yielding global hyperbolicity.
:::



## 4. A3: QSD Sampling and Stationarity

:::{prf:lemma} Discharge of A3 (QSD sampling)
:label: lem-continuum-a3-qsd-sampling

The episode process on the window is stationary and ergodic with density
$\rho_{\mathrm{adaptive}}(x,t)\propto\sqrt{\det g_R(x,t)}\,e^{-U_{\mathrm{eff}}(x,t)/T}$.
:::

:::{prf:proof}
**Step 1: QSD existence and uniqueness.**
The discrete chain admits a unique QSD ({doc}`06_convergence`), and the mean-field
limit admits a unique stationary density $\rho_0$ ({doc}`09_propagation_chaos`).

**Step 2: Gibbs envelope form.**
The spatial profile is a decorated Gibbs measure
({prf:ref}`thm-decorated-gibbs`), which matches the reweighted density used in
{prf:ref}`def-cst-volume`.

**Step 3: Stationarity and ergodicity.**
Exchangeability of the QSD ({prf:ref}`thm-qsd-exchangeability`) and uniqueness of
the mean-field stationary solution imply stationarity; ergodicity follows from
LSI-based mixing (Lemma {prf:ref}`lem-continuum-a4-mixing`).
:::



## 5. A4: LSI Mixing and LLN

:::{prf:lemma} Discharge of A4 (LSI mixing)
:label: lem-continuum-a4-mixing

The episode process satisfies a log-Sobolev inequality on the window with
constant $\kappa>0$, implying exponential mixing and a law of large numbers for
bounded Lipschitz functionals.
:::

:::{prf:proof}
**Step 1: N-uniform LSI.**
The exchangeable QSD satisfies an N-uniform LSI
({prf:ref}`thm-n-uniform-lsi-exchangeable` in {doc}`12_qsd_exchangeability_theory`).
The discrete KL convergence result
({prf:ref}`thm-kl-convergence-euclidean` in {doc}`15_kl_convergence`) provides
explicit constants independent of $N$.

**Step 2: Continuous-time mixing.**
Hypocoercive transfer to the continuous-time generator is provided by
{doc}`10_kl_hypocoercive`, yielding exponential KL decay on the window.

**Step 3: LLN for bounded Lipschitz observables.**
With LSI and exchangeability, concentration and propagation-of-chaos bounds
({prf:ref}`thm-quantitative-propagation-chaos` in {doc}`13_quantitative_error_bounds`)
imply the LLN used in the d'Alembertian estimator.
:::



## 6. A5: Kernel Moment Conditions

:::{prf:lemma} Discharge of A5 (Kernel construction)
:label: lem-continuum-a5-kernel

There exists a compactly supported $K\in C^2_c([0,1])$ satisfying
$M_0=0$ and $M_2^{\mu\nu}=2m_2 g^{\mu\nu}$ with $m_2>0$.
:::

:::{prf:proof}
**Step 1: Choose a bump.**
Fix any non-negative $\phi\in C^2_c([0,1])$ with $\int_J \phi(\tau(\xi))\,d\xi\neq 0$
and $\int_J \phi(\tau(\xi))\,\|\xi\|^2\,d\xi\neq 0$.

**Step 2: Solve the two moment equations.**
Set $K(s)=a\,\phi(s)+b\,s^2\phi(s)$. The constraints
$M_0=0$ and $M_2^{\mu\nu}=2m_2 g^{\mu\nu}$ are two linear equations in $(a,b)$.
Choose $(a,b)$ so that $M_0=0$ and $m_2>0$; this is always possible because the
two moments of $\phi$ are nonzero.

**Step 3: Isotropy.**
Because $\tau(\xi)$ depends only on the Minkowski norm, odd moments vanish and
the second moment is proportional to $g^{\mu\nu}$, giving the required tensor
form.
:::



## 7. A6: Scaling Regime

:::{prf:lemma} Discharge of A6 (Scaling)
:label: lem-continuum-a6-scaling

There exists a sequence $\varepsilon_N\to 0$ such that
$N\varepsilon_N^{D+4}\to\infty$ and the d'Alembertian estimator converges in
probability.
:::

:::{prf:proof}
**Step 1: Bias and variance.**
The estimator expansion in {doc}`../2_fractal_set/02_causal_set_theory` yields
bias $O(\varepsilon^2)$ and variance
$O((N\varepsilon^{D+2})^{-1})$ (see the proof of
{prf:ref}`thm-cst-fractal-dalembertian-consistency`).

**Step 2: Choose an explicit schedule.**
Define $\varepsilon_N := N^{-\alpha}$ with any $\alpha\in(0,1/(D+4))$. Then
$\varepsilon_N\to 0$ and $N\varepsilon_N^{D+4}\to\infty$.

**Step 3: Match to sampling density.**
The quantitative propagation-of-chaos rate
({prf:ref}`thm-quantitative-propagation-chaos`) gives $O(N^{-1/2})$ empirical
error, allowing $\varepsilon_N$ to be interpreted as the QSD-driven sampling
radius. This uses only the mean-field certificate and does not alter the
algorithm.
:::



## 8. Corollary: Assumption-Free Continuum Consistency

:::{prf:corollary} Continuum Consistency without A1-A6
:label: cor-continuum-consistency-no-assumptions

Replacing A1-A6 in {prf:ref}`assm-fractal-gas-nonlocal` with Lemmas
{prf:ref}`lem-continuum-a2-smooth-fields`–{prf:ref}`lem-continuum-a6-scaling`
makes {prf:ref}`thm-cst-fractal-dalembertian-consistency` unconditional within
Volume 3.
:::

:::{prf:proof}
Each hypothesis is discharged by the corresponding lemma above, and the proof
of {prf:ref}`thm-cst-fractal-dalembertian-consistency` uses only A1-A6 to invoke
the bias/variance expansion, LLN, and geometric identification steps. Substituting
the lemmas therefore removes all external assumptions. $\square$
:::
