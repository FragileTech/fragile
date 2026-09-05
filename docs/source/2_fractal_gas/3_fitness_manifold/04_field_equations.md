(sec-field-equations-pressure)=
# Field Equations and Pressure Dynamics

**Prerequisites:** {doc}`01_emergent_geometry`,
{doc}`02_scutoid_spacetime`, and {doc}`03_curvature_gravity`.

(sec-field-equations-tldr)=
## TLDR

This chapter computes responses for specified density and fluctuation
models, then states the conditions for connecting them to the Fractal Gas.
A Gaussian pair energy has an explicit dilation derivative. Its quadratic
stiffness and its pressure are different derivatives.

A homogeneous diffusion and Gaussian gain–loss model has exact decay rate

$$
\omega(k)=D_{\mathrm{eff}}|k|^2+
 \lambda_{\mathrm{kill}}(1-e^{-\varepsilon_c^2|k|^2/2}).
$$

Every nonzero mode decays when
$D_{\mathrm{eff}}+\lambda_{\mathrm{kill}}>0$. A periodic domain has
a positive first nonzero frequency; on the whole space frequencies
approach zero and there is no uniform exponential $L^2$ rate.

The finite Gaussian mode ensemble gives pressure by differentiating its
partition function. The Einstein-type Ricci contraction additionally
uses a specified constitutive equation. The actual swarm's QSD,
mean-field, concentration, and entropy results supply separate,
established analytical estimates with their stated hypotheses.

(sec-field-equations-intro)=
## Introduction

:::{div} feynman-prose
To calculate pressure, imagine expanding the region occupied by a fixed
amount of material. Specify what moves, what stays fixed, and how its
energy changes. Pressure is the negative derivative of that energy
with respect to volume.

A different experiment changes the density profile inside a fixed region.
Its quadratic energy measures stiffness. There is no reason for the
first derivative in the expansion experiment to equal the second
derivative in the density experiment.

We will do both calculations. We will also solve a homogeneous model
exactly in Fourier space and calculate the long-time spreading of an
Ornstein–Uhlenbeck velocity process. Each calculation has a definite
model and normalization. These results become statements about measured
walkers when the corresponding law or evolution has been identified.
:::

(sec-ig-free-energy)=
## Density Fluctuations and a Correlation Energy

:::{div} feynman-prose
A sampling rate measures how unlikely a density fluctuation is. An interaction energy is an additional specification. For independent samples the rate can be calculated from a multinomial probability. For the interacting swarm, the established entropy and LSI estimates control fluctuations directly. We retain those estimates and specify the comparison energy separately.
:::

### Independent sampling and interacting fluctuations

:::{prf:lemma} Independent-sampling rate and interacting concentration
:label: lem-ig-rate-function

For independent samples with law $\pi$, the empirical frequencies on a fixed
finite partition with probabilities $p_1,\ldots,p_m>0$ have rate
$I(q)=\sum_jq_j\log(q_j/p_j)$. More precisely, for any attainable type $q$,

$$
\log\Pr(L_N=q)=-NI(q)+O(m\log(N+1)).
$$

For an interacting law $\mu_N$, the independent-sampling identification
requires a separate comparison. A proved implication sufficient for
bounded-observable estimates is the following: if
$H(\mu_N\mid\pi^{\otimes N})\le H_*$, then
{prf:ref}`thm-mixing-variance-corrected` gives, for $|f|\leq B$,

$$
\mathbb E_{\mu_N}|L_Nf-\pi f|^2
\leq\frac{4B^2}{N}\left(H_*+\frac12\log2\right).
$$

Alternatively a full-gradient joint LSI with constant $C_*$ gives
$\operatorname{Var}_{\mu_N}(L_Nf)\leq C_*L^2/N$ for a fixed
$L$-Lipschitz observable, by
{prf:ref}`cor-quantitative-lsi-final`.

**Proof.** The multinomial formula gives
$\Pr(L_N=q)=N!\prod_jp_j^{Nq_j}/(Nq_j)!$.
Applying $\log n!=n\log n-n+O(\log(n+1))$ and $\sum_jq_j=1$
yields the displayed rate. The interacting implications are precisely the
entropy and Poincaré estimates proved in the cited chapters. Existence
of a joint QSD alone supplies neither the multinomial formula nor an
independent-sampling large-deviation rate.
:::

:::{prf:definition} Fixed-mass Gaussian correlation free energy
:label: def-ig-free-energy

For this comparison model, take a bounded coordinate region
$\Omega\subset\mathbb R^d$ of volume $V$, total mass $N$, and
$\rho_0=N/V$. This finite region defines the auxiliary model; the
confining Fractal Gas state space may remain unbounded.

For nonnegative $\rho$ with $\int_\Omega\rho=N$, set $u=\rho-\rho_0$
and define, in fixed reference energy units,

$$
\mathcal F_{\mathrm{IG}}[\rho]
=\int_\Omega\rho\log(\rho/\rho_0)\,dx
 +\frac12\iint_{\Omega^2}K_\varepsilon(x-y)u(x)u(y)\,dx\,dy,
$$

where

$$
K_\varepsilon(r)=C_0e^{-|r|^2/(2\varepsilon_c^2)},\qquad
C_0,\varepsilon_c>0.
$$

Assume the displayed terms are finite. The first term equals
$N D_{\mathrm{KL}}((\rho/N)\,dx\|V^{-1}\,dx)$.
The pair term is a specified correlation energy. Identifying this sum
with the rate function or free energy of an interacting QSD requires
its own law calculation.
:::

:::{prf:lemma} Positivity of the Gaussian correlation free energy
:label: lem-field-free-energy-positivity

For $u\in L^2(\Omega)$ and the preceding fixed-mass model,

$$
\mathcal F_{\mathrm{IG}}[\rho]\geq0,\qquad
\mathcal F_{\mathrm{IG}}[\rho_0]=0.
$$

Equality requires $\rho=\rho_0$ almost everywhere. The functional is
strictly convex on its finite-energy fixed-mass domain.
:::

:::{prf:proof}
The entropy term is a relative entropy times $N$, hence nonnegative,
with equality only at $\rho_0$. Extend $u$ by zero outside $\Omega$.
With Fourier transform $\widehat f(k)=\int e^{-ik\cdot x}f(x)\,dx$,

$$
\widehat K_\varepsilon(k)
=C_0(2\pi)^{d/2}\varepsilon_c^d
 e^{-\varepsilon_c^2|k|^2/2}>0,
$$

so Plancherel gives

$$
\iint K_\varepsilon(x-y)u(x)u(y)\,dx\,dy
=\frac1{(2\pi)^d}\int
 \widehat K_\varepsilon(k)|\widehat u(k)|^2\,dk\geq0.
$$

The positive quadratic form is convex. The strictly convex function
$r\mapsto r\log(r/\rho_0)$ makes the entropy integral strictly convex
on distinct densities. This proves the assertions.
:::

:::{prf:proposition} Quadratic response and its relation to the jump form
:label: prop-jump-hamiltonian-derivation

Let $\rho=\rho_0(1+\phi)$ in the fixed-mass model, with
$\int_\Omega\phi=0$ and $\|\phi\|_\infty\leq a<1$. Then

$$
\mathcal F_{\mathrm{IG}}[\rho]
=\frac12\langle\phi,\mathcal L_{\mathrm{IG}}\phi\rangle+\mathcal R_3,
\qquad
\mathcal L_{\mathrm{IG}}=\rho_0 I+\rho_0^2\mathcal K,
$$

where $(\mathcal K\phi)(x)=\int_\Omega K_\varepsilon(x-y)\phi(y)\,dy$
and

$$
|\mathcal R_3|
\leq\frac{\rho_0}{6(1-a)^2}\int_\Omega|\phi|^3\,dx.
$$

Define the positive jump operator for this symmetric kernel by

$$
(\mathcal J\phi)(x)
=\int_\Omega K_\varepsilon(x-y)(\phi(x)-\phi(y))\,dy,\qquad
\kappa_\Omega(x)=\int_\Omega K_\varepsilon(x-y)\,dy.
$$

Its exact relation to the free-energy Hessian is

$$
\mathcal L_{\mathrm{IG}}
=(\rho_0+\rho_0^2\kappa_\Omega)I-\rho_0^2\mathcal J,
$$

where multiplication by $\kappa_\Omega$ is understood. Moreover,

$$
\langle\phi,\mathcal J\phi\rangle
=\frac12\iint_{\Omega^2}K_\varepsilon(x-y)
                       (\phi(x)-\phi(y))^2\,dx\,dy\geq0.
$$

Thus the pair-energy Hessian and the jump operator are distinct,
explicitly related operators.
:::

:::{prf:proof}
For $f(r)=(1+r)\log(1+r)$,
$f(0)=0$, $f'(0)=1$, $f''(0)=1$, and
$f'''(r)=-(1+r)^{-2}$. Taylor's theorem on $[-a,a]$ gives

$$
f(\phi)=\phi+\tfrac12\phi^2+r_3,\qquad
|r_3|\leq|\phi|^3/[6(1-a)^2].
$$

The integrated linear term vanishes by mass conservation. The
interaction is exactly quadratic, proving the expansion and its bound.

Finally, $\mathcal J=\kappa_\Omega I-\mathcal K$ gives the operator
identity. Exchange $x,y$ in half of
$\int\phi(x)\int K_\varepsilon(x-y)(\phi(x)-\phi(y))\,dy\,dx$.
Kernel symmetry gives the difference-square formula.
:::

### Density perturbation and spatial dilation

The following definition distinguishes a fixed-volume density perturbation from a mass-preserving spatial dilation.

:::{prf:definition} Mass-preserving affine density perturbation
:label: def-boost-perturbation

On $[0,L]^d$, define $\phi_\kappa(z)=\kappa(z_1/L-1/2)$ and
$\rho_\kappa=\rho_0(1+\phi_\kappa)$ for $|\kappa|<2$.
Then $\rho_\kappa>0$ and $\int\rho_\kappa=\rho_0L^d$.
This is a density perturbation. A spatial dilation is instead the
pushforward $\rho_a(z)=a^{-d}\rho(z/a)$ on $a[0,L]^d$;
{prf:ref}`thm-elastic-pressure` differentiates that explicit deformation.
:::

(sec-elastic-pressure)=
## Dilation Pressure and Correlation Stiffness

:::{div} feynman-prose
Hold the kernel and total transported mass fixed while increasing all distances by a factor $a$. A positive Gaussian pair energy decreases because its kernel becomes smaller at separated points. Its pressure is therefore positive for a nonnegative density. An attractive energy with the opposite sign has the opposite response. The derivative will also show why a signed density fluctuation has no universal pressure sign.
:::

:::{prf:theorem} Dilation response and Gaussian correlation stiffness
:label: thm-elastic-pressure

Let $u$ be an integrable compactly supported density or signed density on
$\mathbb R^d$, and set

$$
E(a)=\frac12\iint K_\varepsilon(a(x-y))u(x)u(y)\,dx\,dy,\quad
K_\varepsilon(r)=C_0e^{-|r|^2/(2\varepsilon_c^2)},\quad V(a)=a^dV.
$$

This describes transport of a fixed amount of $u$ by dilation. Its pressure
at $a=1$ is

$$
-\frac{dE}{dV}=\frac{1}{2dV\varepsilon_c^2}
\iint |x-y|^2K_\varepsilon(x-y)u(x)u(y)\,dx\,dy.
$$

For $u\ge0$ this is nonnegative; reversing the sign of the interaction
energy reverses the pressure. For a signed fluctuation $u=\rho-\rho_0$
the formula has no fixed sign. A quadratic free-energy expansion in a
mass-preserving perturbation has zero first derivative at zero strain.

For the translation-invariant jump form with affine perturbation
$\Phi(x)=b\cdot x$, its quadratic energy per unit volume is

$$
\frac{\rho_0^2}{8}\int K_\varepsilon(r)(b\cdot r)^2\,dr
=\frac{C_0\rho_0^2(2\pi)^{d/2}\varepsilon_c^{d+2}}8|b|^2.
$$

**Proof.** Differentiate the Gaussian under the integral:
$E'(1)=-(2\varepsilon_c^2)^{-1}
\iint|x-y|^2K_\varepsilon(x-y)u(x)u(y)\,dx\,dy$.
Divide by $V'(1)=dV$ and change the sign. For $\rho=\rho_0+\kappa u$
with $\int u=0$, the entropy derivative is $\int u=0$ and the interaction
is quadratic in $\kappa$, proving the zero-strain statement. Finally
$\int r_ir_jK_\varepsilon(r)\,dr=
\delta_{ij}C_0(2\pi)^{d/2}\varepsilon_c^{d+2}$ by Gaussian integration.
Contracting with $b_ib_j$ proves the stiffness formula. Stiffness is a
second variation; identifying it with a first-variation pressure requires
a specified deformation and energy convention.
:::

:::{prf:remark} Sign of the interaction response
:label: rem-elastic-interpretation

The positive Gaussian pair energy, its negative attractive counterpart,
and the signed fluctuation energy have different dilation responses.
The formula in {prf:ref}`thm-elastic-pressure` fixes the sign after the
energy and deformation are specified. The jump-form stiffness is positive
and scales as $\varepsilon_c^{d+2}$ with $C_0$ fixed. If instead the kernel
mass is normalized by choosing $C_0\propto\varepsilon_c^{-d}$, its
stiffness scales as $\varepsilon_c^2$. The selected kernel normalization
must therefore be retained when comparing bandwidths or applying the
formula to a row-normalized companion mechanism. Neither stiffness
formula alone fixes a vacuum pressure.
:::

(sec-linearized-dynamics)=
## A Homogeneous Density Model

:::{div} feynman-prose
A translation-invariant model lets us ask how a single sinusoidal density
wave changes. Diffusion smooths it. A balanced Gaussian redistribution
also smooths it, because averaging over nearby points reduces its amplitude.

This calculation is exact for the specified linear model. The full
Fractal Gas mean-field equation contains its actual state-dependent
coefficients and normalized cloning terms. Those are derived in the
mean-field chapter. We use the homogeneous model here as a comparison
whose Fourier multipliers can be calculated completely.
:::

:::{prf:definition} Auxiliary gain–loss density equation
:label: def-mckean-vlasov

For a specified drift $b$, nonnegative kernel $K_{\mathrm{clone}}$,
loss rate $\lambda_{\mathrm{kill}}$, and constant
$D_{\mathrm{eff}}\geq0$, consider

$$
\partial_t\rho
=D_{\mathrm{eff}}\Delta\rho-\nabla\cdot(\rho b)
 +\int K_{\mathrm{clone}}(x,y)\rho(y)\,dy
 -\lambda_{\mathrm{kill}}(x)\rho(x).
$$

Use periodic boundary conditions, or sufficient whole-space decay for
the integrations under consideration. The gain–loss part conserves
mass for every integrable density precisely when

$$
\int K_{\mathrm{clone}}(x,y)\,dx=\lambda_{\mathrm{kill}}(y)
\quad\text{almost everywhere}.
$$

This is a specified linear density model. The actual nonlinear
mean-field equation, companion law, and survival-normalized evolution
are derived in {doc}`../convergence_program/08_mean_field`.

For comparison, if an unnormalized killed density satisfies
$\partial_tf=L^*f-\kappa f$ for a conservative $L$, then
$\rho=f/\int f$ satisfies

$$
\partial_t\rho=L^*\rho-(\kappa-\langle\kappa\rangle_\rho)\rho.
$$

A QSD is stationary for this normalized evolution and is a left
eigenmeasure for the killed one.
:::

:::{prf:proof}
Integrate the gain–loss term and use Fubini. Its integral is
$\int[\int K_{\mathrm{clone}}(x,y)\,dx-\lambda_{\mathrm{kill}}(y)]
\rho(y)\,dy$, proving sufficiency and necessity by testing all
nonnegative densities. For killing,
$m'=-m\langle\kappa\rangle_\rho$ where $m=\int f$.
Differentiating $f/m$ gives the normalized equation.
:::

:::{prf:definition} Uniform reference and homogeneous Gaussian closure
:label: def-uniform-qsd-linearization

Set $b=0$ and let the redistribution operator be convolution:

$$
K_{\mathrm{clone}}(x,y)=k_\varepsilon(x-y),\qquad
k_\varepsilon(r)=\lambda_{\mathrm{kill}}
 (2\pi\varepsilon_c^2)^{-d/2}
 e^{-|r|^2/(2\varepsilon_c^2)}.
$$

On a periodic box of side $L$, periodize this kernel by summing its
translates in $L\mathbb Z^d$. Its integral over the box is
$\lambda_{\mathrm{kill}}$, and the constant density $\rho_0=N/L^d$
is stationary for the conservative model.

Writing $\rho=\rho_0+\delta\rho$ gives the exact equation

$$
\partial_t\delta\rho
=D_{\mathrm{eff}}\Delta\delta\rho
 +k_\varepsilon*\delta\rho-\lambda_{\mathrm{kill}}\delta\rho.
$$

There is no neglected nonlinear term in this specified closure.
On $\mathbb R^d$ the same equation describes perturbations about a
homogeneous background. A nonzero constant background on that space
has infinite total mass and is not a probability density or a QSD.
:::

### Fourier Analysis and Dispersion Relation

:::{prf:theorem} Exact decay multipliers of the homogeneous closure
:label: thm-dispersion-relation

For a translation-invariant gain kernel $k_{\mathrm{gain}}$ and the Fourier convention
$\widehat f(k)=\int e^{-ik\cdot x}f(x)\,dx$, a mode
$e^{ik\cdot x-\omega(k)t}$ has

$$
\omega(k)=D_{\mathrm{eff}}|k|^2+
                 \lambda_{\mathrm{kill}}-\widehat k_{\mathrm{gain}}(k).
$$

For the Gaussian model of
{prf:ref}`def-uniform-qsd-linearization`,

$$
\widehat k_\varepsilon(k)
=\lambda_{\mathrm{kill}}e^{-\varepsilon_c^2|k|^2/2},\qquad
\omega(k)
=D_{\mathrm{eff}}|k|^2+
 \lambda_{\mathrm{kill}}(1-e^{-\varepsilon_c^2|k|^2/2}).
$$

On the periodic box these identities hold at
$k\in(2\pi/L)\mathbb Z^d$.
:::

:::{prf:proof}
The Laplacian multiplies a Fourier mode by $-|k|^2$.
For the gain term, substitute $r=x-y$:

$$
\int k_{\mathrm{gain}}(x-y)e^{ik\cdot y}\,dy
=e^{ik\cdot x}\int k_{\mathrm{gain}}(r)e^{-ik\cdot r}\,dr
=\widehat k_{\mathrm{gain}}(k)e^{ik\cdot x}.
$$

The loss term multiplies the mode by $-\lambda_{\mathrm{kill}}$.
Thus $-\omega=-D_{\mathrm{eff}}|k|^2+\widehat k_{\mathrm{gain}}(k)
-\lambda_{\mathrm{kill}}$.

The product of the one-dimensional Gaussian Fourier integrals is
$e^{-\varepsilon_c^2|k|^2/2}$ after normalization, giving the formula.
For a periodized kernel, integration over one box and summation over
its translates give the same Fourier integral at the allowed discrete
frequencies.
:::

:::{prf:remark} Self-adjoint homogeneous closure
:label: rem-real-eigenvalues

For an even integrable convolution kernel and periodic or whole-space
Laplacian with its standard self-adjoint domain, convolution is a bounded
self-adjoint operator. The bounded perturbation of the self-adjoint
Laplacian is self-adjoint on the same domain. Thus the Gaussian closure
has real Fourier multipliers. General directed cloning kernels need not
satisfy this symmetry; their evolution is treated by the kinetic and
mean-field estimates in the convergence chapters.
:::

(sec-qsd-stability)=
## Stability and Domain-Dependent Rates

:::{div} feynman-prose
Every nonzero sinusoid decays in this Gaussian model. Whether there is a single exponential rate for all perturbations depends on the domain. A periodic box has a smallest nonzero wave number. The whole space has arbitrarily long waves, and those decay arbitrarily slowly. The conserved constant mode must also be separated from the relaxing modes.
:::

:::{prf:theorem} Stability of the homogeneous Gaussian closure
:label: thm-qsd-stability

For the translation-invariant closure in {prf:ref}`def-uniform-qsd-linearization`,
let $D_{\mathrm{eff}},\lambda_{\mathrm{kill}}\ge0$ and $\varepsilon_c>0$.
Every nonzero Fourier mode decays strictly if and only if
$D_{\mathrm{eff}}+\lambda_{\mathrm{kill}}>0$. The zero mode is conserved.
For $q=\varepsilon_c^2|k|^2/2$,

$$
D_{\mathrm{eff}}|k|^2+\lambda_{\mathrm{kill}}\frac{q}{1+q}
\le\omega(k)
\le D_{\mathrm{eff}}|k|^2+\lambda_{\mathrm{kill}}\min(q,1).
$$

**Proof.** Substitute the Gaussian multiplier from
{prf:ref}`thm-dispersion-relation`. For $q\ge0$, $e^q\ge1+q$ implies
$1-e^{-q}\ge q/(1+q)$, and integration of $e^{-r}\le1$ gives
$1-e^{-q}\le q$. Also $1-e^{-q}\le1$. The multiplier is positive for
$k\ne0$ when at least one coefficient is positive and is identically
zero when both vanish. At $k=0$ it is zero in every case.
:::

:::{prf:corollary} Relaxation on a domain with a nonzero first frequency
:label: cor-exponential-relaxation

Assume $D_{\mathrm{eff}}+\lambda_{\mathrm{kill}}>0$. For the preceding
closure on the periodic box of side $L$, the mean-zero
solution satisfies

$$
\|\delta\rho_t\|_2\le e^{-\omega_*t}\|\delta\rho_0\|_2,\qquad
\omega_*=D_{\mathrm{eff}}(2\pi/L)^2+
\lambda_{\mathrm{kill}}[1-e^{-\varepsilon_c^2(2\pi/L)^2/2}]>0.
$$

On $\mathbb R^d$ the same multiplier has infimum zero over nonzero
frequencies; the closure therefore has no uniform exponential $L^2$ rate.
For $D_{\mathrm{eff}}>0$ and $\delta\rho_0\in L^1\cap L^2$ it has the bound
$\|\delta\rho_t\|_2\le C_d(D_{\mathrm{eff}}t)^{-d/4}\|\delta\rho_0\|_1$.
The confining kinetic model uses the different long-time estimates of
{doc}`../convergence_program/06_convergence` and
{doc}`../convergence_program/10_kl_hypocoercive`.

**Proof.** The multiplier increases with $|k|^2$. On the box, every
nonzero Fourier frequency has length at least $2\pi/L$; Parseval gives
the first bound. On $\mathbb R^d$, $\omega(k)\to0$ as $k\to0$.
Plancherel, $|\widehat{\delta\rho_0}|\le\|\delta\rho_0\|_1$, and
$\omega(k)\ge D_{\mathrm{eff}}|k|^2$ give the Gaussian integral bound.
:::

:::{prf:remark} Long-wavelength diffusion
:label: rem-anti-diffusion

The exact Gaussian multiplier has expansion

$$
\omega(k)
=\left(D_{\mathrm{eff}}+
 \frac{\lambda_{\mathrm{kill}}\varepsilon_c^2}{2}\right)|k|^2
-\frac{\lambda_{\mathrm{kill}}\varepsilon_c^4}{8}|k|^4
+O(|k|^6).
$$

Its long-wavelength coefficient is

$$
D_{\mathrm{long}}
=D_{\mathrm{eff}}+\lambda_{\mathrm{kill}}\varepsilon_c^2/2
\geq D_{\mathrm{eff}},
$$

with strict inequality when $\lambda_{\mathrm{kill}}>0$.
The negative fourth-order term is a correction to a small-frequency
series. Extending that truncated polynomial to arbitrarily large
frequencies would produce a spurious instability; the exact multiplier
remains nonnegative. Nonlinear or directed cloning mechanisms use
their own linearization and stability calculation.
:::



(sec-chapman-enskog)=
## Kinetic Diffusion and the Time Step

:::{div} feynman-prose
A velocity keeps part of its value from one instant to the next. Those correlations determine how far position spreads. For an Ornstein–Uhlenbeck velocity, the covariance is an exponential, so we can integrate it exactly. This gives the continuous-time diffusion coefficient. A finite BAOAB step has a related geometric covariance series, which gives its own coefficient before taking the small-step limit.
:::

:::{prf:definition} Free kinetic reference operator
:label: def-phase-space-kinetic-operator

The constant-coefficient reference process

$$
dX=V\,dt,\qquad dV=-\gamma V\,dt+\sigma_v\,dW,\qquad
\gamma,\sigma_v>0,
$$

has backward generator

$$
\mathcal L_{\mathrm{kin}}f
=v\cdot\nabla_xf-\gamma v\cdot\nabla_vf
                   +\frac{\sigma_v^2}{2}\Delta_vf.
$$

Its stationary velocity covariance is $v_T^2I$, where
$v_T^2=\sigma_v^2/(2\gamma)$. Position on $\mathbb R^d$ spreads
and has no uniform invariant probability law. The adaptive, forced,
aligned, and cloning swarm uses its complete generator rather than this
free reference operator.
:::

:::{prf:theorem} Diffusion coefficient of integrated Ornstein--Uhlenbeck motion
:label: thm-einstein-relation

For $dV_t=-\gamma V_tdt+\sigma_vdW_t$, $dX_t=V_tdt$, with
$V_0$ in its stationary Gaussian law, put $v_T^2=\sigma_v^2/(2\gamma)$.
For each coordinate,

$$
\operatorname{Var}(X_t-X_0)=
2v_T^2\left[\frac t\gamma-\frac{1-e^{-\gamma t}}{\gamma^2}\right].
$$

Thus the long-time diffusion coefficient is
$D_{\mathrm{eff}}=v_T^2/\gamma=\sigma_v^2/(2\gamma^2)$.
This coefficient belongs to the stated kinetic reference process.

**Proof.** The explicit OU solution gives
$\mathbb E[V_sV_r]=v_T^2e^{-\gamma|s-r|}$ coordinatewise.
Integrating this covariance over $[0,t]^2$ gives the displayed variance.
Moreover
$X_t-X_0=(V_0-V_t)/\gamma+(\sigma_v/\gamma)W_t$.
After diffusive rescaling the first term vanishes in mean square at each
fixed rescaled time. The Brownian term has variance
$\sigma_v^2t/\gamma^2=2D_{\mathrm{eff}}t$.
:::


:::{prf:lemma} Diffusion of the force-free BAOAB reference step
:label: lem-field-discrete-ou-diffusion

Fix a constant positive definite covariance shape $D_0$, timestep
$\Delta t>0$, and $0<a<1$. Consider

$$
V_{n+1}=aV_n+c_2D_0^{1/2}\xi_n,\qquad
X_{n+1}=X_n+\frac{\Delta t}{2}(V_n+V_{n+1}),
$$

where the $\xi_n$ are independent standard Gaussians and the velocity
starts in stationarity. Then

$$
C_v=\operatorname{Cov}(V_n)=\frac{c_2^2}{1-a^2}D_0,\qquad
\lim_{m\to\infty}
 \frac{\operatorname{Cov}(X_m-X_0)}{2m\Delta t}
=\frac{\Delta t\,c_2^2}{2(1-a)^2}D_0.
$$

For the thermostat coefficients $a=e^{-\gamma\Delta t}$ and
$c_2^2=v_T^2(1-a^2)$, this is

$$
D_{\Delta t}
=\frac{v_T^2\Delta t}{2}
 \coth(\gamma\Delta t/2)D_0
=\frac{v_T^2}{\gamma}
 \left[1+\frac{(\gamma\Delta t)^2}{12}
             +O((\gamma\Delta t)^4)\right]D_0.
$$

This is the existing BAOAB update specialized to zero force, zero
alignment, no jumps, and constant diffusion shape. The result identifies
that reference update; it does not alter the full algorithm.
:::

:::{prf:proof}
The covariance recursion is $C_v=a^2C_v+c_2^2D_0$, giving the
stationary covariance. Iteration gives
$\operatorname{Cov}(V_{n+r},V_n)=a^rC_v$. Therefore

$$
\frac1m\operatorname{Cov}\left(\sum_{n=0}^{m-1}V_n\right)
=\left[1+2\sum_{r=1}^{m-1}(1-r/m)a^r\right]C_v
\longrightarrow\frac{1+a}{1-a}C_v.
$$

The position increment is

$$
X_m-X_0
=\Delta t\sum_{n=0}^{m-1}V_n
 +\frac{\Delta t}{2}(V_m-V_0).
$$

The second term has bounded second moment as $m\to\infty$; its
covariance and cross terms vanish after division by $m$. This proves
the limiting diffusion tensor. Substitute the thermostat coefficients
and use $(1+e^{-z})/(1-e^{-z})=\coth(z/2)$ and its Taylor expansion.
:::

(sec-radiation-pressure)=
## Pressure of Specified Gaussian Modes

:::{div} feynman-prose
Now specify a finite collection of fluctuating modes and their quadratic energy. A Gaussian integral gives the partition function exactly. Pressure depends on how the energy coefficients change with volume. A relaxation rate by itself does not specify that energy: it also depends on mobility and noise normalization. The following model states all three choices.
:::

:::{prf:assumption} Gaussian fluctuation reference model
:label: ass-thermal-equilibrium

For a finite set of real modes $q_j$, specify
$\Theta=k_BT_{\mathrm{eff}}>0$ and the quadratic energy

$$
E(q)=\frac12\sum_jw_jq_j^2,\qquad w_j>0,
$$

with density $Z^{-1}e^{-E/\Theta}$ relative to a specified Lebesgue
measure on mode coordinates. Then $\mathbb E q_j^2=\Theta/w_j$
and the modes are independent.

For mobility $m_j>0$, the dynamics

$$
dq_j=-m_jw_jq_j\,dt+\sqrt{2m_j\Theta}\,dB_j
$$

have this invariant law and relaxation rate $m_jw_j$.
Identifying $w_j$ with a measured rate therefore requires $m_j=1$
in the declared units, or another measured mobility and its matched
noise amplitude. An arbitrary cloning QSD does not determine this
Gaussian energy model.
:::

:::{prf:proof}
The density factors into normalized one-dimensional Gaussians with
variance $\Theta/w_j$. The stated OU process has stationary variance
$(2m_j\Theta)/(2m_jw_j)=\Theta/w_j$, proving invariance and the
decay rate.
:::

:::{prf:proposition} Pressure of a finite Gaussian mode ensemble
:label: prop-radiation-pressure

For {prf:ref}`ass-thermal-equilibrium`, at fixed temperature, fixed
mode index set, and fixed reference measure on mode coordinates, suppose
$w_j(V)>0$ is differentiable. The mode pressure is

$$
P_{\mathrm{modes}}=-\frac{k_BT_{\mathrm{eff}}}{2}
\sum_j\partial_V\log w_j(V).
$$

If every $w_j(V)=a_jV^{-2/d}$, then
$P_{\mathrm{modes}}=k_BT_{\mathrm{eff}}n/(dV)$ for $n$ real modes.
A prescribed spatial cutoff $|k|\le k_*$ on a periodic box gives
$n\sim V\operatorname{vol}(B_d)k_*^d/(2\pi)^d$ in the large-box limit,
with the zero mode omitted. A classical Gaussian ensemble has no intrinsic
thermal cutoff; an infinite unregularized mode count diverges.

**Proof.** Gaussian integration gives
$Z=\prod_j(2\pi k_BT_{\mathrm{eff}}/w_j)^{1/2}$.
Differentiate $F=-k_BT_{\mathrm{eff}}\log Z$ and use $P=-\partial_VF$.
The power law follows by differentiating $\log w_j$.
For mode counting, place a unit cube at each integer lattice point in the
ball of radius $Lk_*/(2\pi)$. Their union lies between balls with radii
differing by at most $\sqrt d$, giving the stated leading volume.
Changing the cutoff or the mode index set during a volume derivative
requires the corresponding additional terms in $F$.
:::

:::{prf:remark} Pressure and stiffness use different derivatives
:label: rem-pressure-comparison

The pair-energy pressure is computed along a specified spatial dilation.
The mode pressure differentiates the Gaussian partition function at fixed
temperature and fixed mode set. Their sum represents a chosen effective
energy model only when these conventions and the common state law agree.
The quadratic correlation stiffness is a separate response coefficient.
:::

(sec-pressure-regimes)=
## Pressure Regime Analysis

:::{prf:definition} Crossover of a prescribed pressure model
:label: def-thermal-correlation-length

For a phenomenological pressure model
$P(\varepsilon_c)=B-A\varepsilon_c^{d+2}$ with fixed $A,B>0$, define
$\varepsilon_c^{\mathrm{th}}=(B/A)^{1/(d+2)}$.
The coefficient $A$ is an attractive-pressure coefficient of the chosen
model; its value must come from a specified dilation derivative.
It is not obtained by relabeling the positive jump stiffness.
:::

:::{prf:theorem} Sign of the prescribed two-term pressure
:label: thm-pressure-regimes

Under {prf:ref}`def-thermal-correlation-length`,

$$
P(\varepsilon_c)=B\left[1-
\left(\frac{\varepsilon_c}{\varepsilon_c^{\mathrm{th}}}\right)^{d+2}\right].
$$

It is positive below the crossover, zero at the crossover, and negative
above it. These conclusions concern the prescribed pressure model.

**Proof.** Substitute $A=B/(\varepsilon_c^{\mathrm{th}})^{d+2}$ and use
strict monotonicity of $r^{d+2}$ for $r>0$. In particular, at fixed $A,B$
the term proportional to $\varepsilon_c^{d+2}$ becomes smaller, rather
than larger, as the correlation length tends to zero.
:::

:::{prf:remark} Analytical results and effective closures
:label: rem-analysis-limitations

The homogeneous Gaussian closure has the exact Fourier multiplier and
mode estimates above. The kinetic reference has the proved OU diffusion
coefficient. Gaussian pair energies and mode ensembles have explicit
first-variation pressures. Applying these formulas to the full interacting
QSD requires the corresponding closure and law identifications, using
{doc}`../convergence_program/07_discrete_qsd`,
{doc}`../convergence_program/09_propagation_chaos`, and
{doc}`../convergence_program/15_kl_convergence`.
:::

(sec-stress-energy-tensor)=
## Stress and a Specified Field Equation

:::{div} feynman-prose
An isotropic stress in a local rest frame has one energy density and one
pressure. This determines the algebraic form of a perfect-fluid tensor.

A field equation is an additional relation between that tensor and the
metric. Once the relation is specified, taking its trace and contracting
with an observer's velocity gives a definite curvature formula. Geometry
alone does not determine the energy model or the proportionality
constant.
:::

:::{prf:definition} Perfect-fluid effective stress
:label: def-effective-stress-energy

Let $G_{ab}$ be the specified Lorentzian metric in dimension $d+1$,
and let $u$ be a unit timelike field, $G(u,u)=-1$. A symmetric stress
with zero rest-frame energy flux and isotropic spatial stress has form

$$
T_{ab}^{\mathrm{eff}}
=e\,u_au_b+P\,h_{ab}
=(e+P)u_au_b+P\,G_{ab},\qquad
h_{ab}=G_{ab}+u_au_b.
$$

Here $e=T(u,u)$ is energy density and $P$ is rest-frame pressure.
The field $u$ need not be geodesic. An anisotropic stress or nonzero
energy flux requires the corresponding additional tensor terms.

For the chosen effective energy model one may set
$P=P_{\mathrm{pair}}+P_{\mathrm{modes}}$ after fixing their common
volume, temperature, and state-law conventions. This equation is a
definition of that model, not an identification forced by isotropy of
a sampling density.
:::

:::{prf:proof}
Choose a local orthonormal rest frame with time direction $u$.
The stated conditions give components
$T_{00}=e$, $T_{0i}=0$, and $T_{ij}=P\delta_{ij}$.
The displayed tensor has exactly these components, proving its
coordinate-independent form.
:::

:::{prf:definition} Signed pressure parameter and vacuum convention
:label: def-effective-cosmological-constant

Choose a positive coupling $\kappa_G$, with
$\kappa_G=8\pi G_{\mathrm{eff}}/c^4$ in four-dimensional physical
units, and define

$$
\Lambda_P=\kappa_G P_{\mathrm{vac}}.
$$

This is a signed pressure parameter. Let
$\mathsf E_{ab}=R_{ab}-\tfrac12R G_{ab}$ denote the Einstein tensor.
In the convention

$$
\mathsf E_{ab}+\Lambda G_{ab}=\kappa_G T_{ab},
$$

vacuum stress has
$T_{ab}^{\mathrm{vac}}=-e_{\mathrm{vac}}G_{ab}$ and
$P_{\mathrm{vac}}=-e_{\mathrm{vac}}$. Moving that term to the left
gives

$$
\Delta\Lambda=\kappa_Ge_{\mathrm{vac}}=-\Lambda_P.
$$

Thus the pressure parameter and the vacuum contribution to the
Einstein constant have opposite signs. A physical identification also
requires the vacuum equation of state and the field equation.
Dimensional consistency does not fix the coupling's value.
:::

:::{prf:theorem} Ricci contraction under an Einstein constitutive equation
:label: thm-structural-correspondence

In spacetime dimension $n=d+1>2$, suppose an effective metric and
stress satisfy the specified constitutive equation

$$
\mathsf E_{ab}+\Lambda G_{ab}=\kappa_G T_{ab},
\qquad
\mathsf E_{ab}=R_{ab}-\tfrac12R G_{ab}.
$$

For a perfect fluid with energy density $e$, pressure $P$, and
unit timelike field $u$,

$$
R_{ab}u^au^b
=\frac{\kappa_G}{d-1}\bigl[(d-2)e+dP\bigr]
 -\frac{2\Lambda}{d-1}.
$$

Here $\kappa_G$ has the units of the specified dimension and energy
normalization. In four-dimensional physical units it may be written
$8\pi G_{\mathrm{eff}}/c^4$.
:::

:::{prf:proof}
Taking the trace yields

$$
R=\frac{2(n\Lambda-\kappa_GT)}{n-2}.
$$

Substitution into the constitutive equation gives

$$
R_{ab}
=\kappa_G\left(T_{ab}-\frac{T}{n-2}G_{ab}\right)
 +\frac{2\Lambda}{n-2}G_{ab}.
$$

Now $G(u,u)=-1$, $T(u,u)=e$, and $T=-e+dP$.
Contracting yields the stated formula. Under the congruence hypotheses,
it can be substituted into Raychaudhuri's identity.

Conservation and Raychaudhuri do not imply the constitutive equation.
For example, on Minkowski space take $\Lambda=0$ and a nonzero
constant perfect-fluid stress. Its divergence vanishes and all geometric
identities hold, while $\mathsf E=0\ne\kappa_G T$.
:::

:::{prf:remark} Geometric identity and constitutive equation
:label: rem-correspondence-meaning

Raychaudhuri relates the expansion of a specified congruence to the Ricci
tensor of its metric. The perfect-fluid contraction in
{prf:ref}`thm-structural-correspondence` additionally uses the stated
Einstein constitutive equation. A pressure sign determines that contraction
only after the energy density, cosmological term, dimensional normalization,
and constitutive equation are fixed.
:::

(sec-summary-field-equations)=
## Results and Their Applications

The fixed-mass Gaussian correlation energy is nonnegative and has the
proved quadratic expansion. Its Hessian and the jump operator obey the
explicit identity in {prf:ref}`prop-jump-hamiltonian-derivation`.
The dilation pressure follows by differentiating the stated energy;
the sign depends on that energy and the transported density.

The homogeneous closure has the exact Fourier multiplier and stability
bounds above. On a periodic box its mean-zero modes have an exponential
rate; on $\mathbb R^d$ arbitrarily long waves remove a uniform spectral
gap. The continuous OU and constant-coefficient BAOAB references give
their respective diffusion tensors.

For a finite Gaussian mode ensemble,

$$
P_{\mathrm{modes}}
=-\frac{k_BT_{\mathrm{eff}}}{2}
 \sum_j\partial_V\log w_j.
$$

The prescribed two-term pressure
$B-A\varepsilon_c^{d+2}$ is positive below its crossover and negative
above it when $A,B$ remain fixed. That algebra does not identify an
Einstein cosmological constant.

Applications to the actual interacting swarm use its identified law,
normalized mean-field dynamics, and the finite-particle, LSI, and entropy
estimates. A relation between stress and Ricci curvature additionally
requires the constitutive equation of
{prf:ref}`thm-structural-correspondence`.

(sec-symbols-field-equations)=
## Table of Symbols

| Symbol | Meaning |
|---|---|
| $\mathcal F_{\mathrm{IG}}$ | Specified fixed-mass correlation free energy |
| $\mathcal L_{\mathrm{IG}}$, $\mathcal J$ | Free-energy Hessian and positive jump operator |
| $K_\varepsilon$, $C_0$, $\varepsilon_c$ | Gaussian pair kernel, amplitude, and length scale |
| $\phi_\kappa$ | Fixed-volume affine density perturbation |
| $P_{\mathrm{pair}}$, $P_{\mathrm{modes}}$ | Pressures of the specified pair and mode energies |
| $D_{\mathrm{eff}}$, $D_{\Delta t}$ | Continuous reference and finite-step diffusion coefficients |
| $\omega(k)$ | Decay rate of the homogeneous Fourier mode |
| $\lambda_{\mathrm{kill}}$ | Balanced loss coefficient in the auxiliary gain–loss model |
| $T_{\mathrm{eff}}$, $w_j$, $m_j$ | Mode temperature, quadratic energy coefficient, and mobility |
| $\varepsilon_c^{\mathrm{th}}$ | Crossover in the prescribed two-term pressure model |
| $G_{ab}$, $\mathsf E_{ab}$ | Lorentzian metric and its Einstein tensor |
| $e$, $P$, $T_{ab}$ | Rest-frame energy density, pressure, and effective stress |
| $\kappa_G$, $\Lambda$ | Coupling and cosmological term in the specified constitutive equation |
| $\Lambda_P$ | Signed pressure parameter; vacuum contributes $-\Lambda_P$ to $\Lambda$ |
| $\gamma$, $\sigma_v^2$, $v_T^2$ | Reference friction, noise variance rate, and stationary velocity variance |

(sec-references-field-equations)=
## References

### Geometry and Analytical Foundations

- {doc}`01_emergent_geometry` --- Emergent Riemannian geometry from fitness landscape
- {doc}`02_scutoid_spacetime` --- Cell reconstruction, sampling, and volume evolution
- {doc}`03_curvature_gravity` --- Curvature from discrete holonomy

### External References

```{bibliography}
:filter: docname in docnames
```

**Key citations:**

- Large deviations and rate functions: {cite}`dembo1998large`
- Kinetic scaling and Chapman–Enskog context: {cite}`chapman1990mathematical`
- McKean-Vlasov equations: {cite}`sznitman1991topics`
- Einstein relation and fluctuation-dissipation: {cite}`kubo1966fluctuation`
