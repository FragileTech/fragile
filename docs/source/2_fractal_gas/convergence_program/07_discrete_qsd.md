# Equilibrium Profiles and Their Analytical Characterization

(sec-equilibrium-stationary)=
## Stationary Equations and Conditional Laws

:::{div} feynman-prose
An equilibrium profile is a distribution whose shape survives the evolution. We must specify which evolution, and what happens to lost mass. A conservative process preserves probability directly. A killed process loses probability, and its quasi-stationary distribution preserves its shape only after we divide by the surviving mass. A mean-field equation describes a third object: a population law evolving under its own interaction field.

These distinctions matter when we draw a proposed equilibrium curve. Its shape may solve a kinetic equation while failing the cloning balance, or it may describe the positions of alive walkers without accounting for the fraction that are alive. This chapter derives several useful reference profiles and then gives a quantitative test for a proposed profile of the full stationary model.
:::

:::{prf:definition} Three equilibrium objects
:label: def-equilibrium-objects

For a killed finite-swarm kernel $Q_N$, a QSD is a probability law satisfying
$\pi_NQ_N=\alpha_N\pi_N$, $0<\alpha_N\leq1$. Its one-walker marginal is a
probability measure on the walker's marked state space. A stationary solution
of the continuous mean-field equation satisfies $Au+\mathcal R(u)=0$.
A kinetic reference law instead satisfies $A\rho=0$.

The QSD convergence theorem in {doc}`06_convergence`, the mean-field
existence, uniqueness, and attraction theorems in {doc}`09_propagation_chaos`,
and the kinetic estimates in {doc}`15_kl_convergence` refer to these specified
objects. Identification of a QSD marginal with a stationary mean-field law
uses {prf:ref}`thm-uniqueness-of-qsd` and its consistency and concentration
hypotheses.
:::

:::{prf:proposition} Alive equilibrium profile and its mass
:label: prop-equilibrium-alive-balance

For the normalized-kernel continuous model of
{prf:ref}`prop-chaos-alive-stationary-reconstruction`, let $\rho$ be a
probability density solving

$$
A\rho+S[\rho]-c\rho+\bar c[\rho]G_\rho=0,
\qquad \bar c[\rho]=\int c\rho.
$$

Then its stationary alive measure is $m_a\rho$, where
$m_a=\lambda_{\rm rev}/(\lambda_{\rm rev}+\bar c[\rho])$.
The remaining mass is $m_d=1-m_a$. Boundary-loss proposals are treated with
the full marked-state equations of {doc}`08_mean_field`.

*Proof.* Integrating the alive equation gives the stationary balance
$m_a\bar c[\rho]=\lambda_{\rm rev}m_d$. Solve this equation with
$m_a+m_d=1$, and substitute into the normalized stationary equation.
:::

:::{div} feynman-prose
The alive fraction has a simple flow balance. The rate at which alive mass is lost equals the rate at which dead mass returns. The normalized density $\rho$ tells us where the alive walkers are; multiplying by $m_a$ tells us how much of the whole population they represent. A plot normalized to unit area can hide this distinction.
:::

(sec-equilibrium-distance)=
## Companion Distances and Local Density

:::{div} feynman-prose
In a dense cloud, the nearest particle is usually closer. To estimate the distance, expand a ball until it contains roughly one particle. Density times ball volume is then of order one, giving the familiar inverse-density scale.

Kernel selection asks a different question. It assigns a positive weight to every eligible companion and samples from those weights. A nearby candidate is favored, but it need not be chosen. The expected distance must therefore use the actual sampling law. The two calculations below show exactly where their different scales enter.
:::

:::{prf:proposition} Nearest-neighbor scaling and companion selection
:label: prop-continuum-distance

For a homogeneous Poisson point process of intensity $\lambda>0$ in
$\mathbb R^D$, the distance $R$ from a fixed test point to the nearest point
satisfies, with $\omega_D=\pi^{D/2}/\Gamma(1+D/2)$,

$$
\mathbb P(R>r)=e^{-\lambda\omega_Dr^D},\qquad
\mathbb ER=\Gamma(1+1/D)(\lambda\omega_D)^{-1/D}.
$$

The local Poisson approximation with intensity $N\rho(z)$ therefore has
scale $[N\rho(z)]^{-1/D}$. Its use requires a local Poisson approximation;
exchangeability alone does not supply it.

For companion selection with a fixed kernel $K_\epsilon(z,y)$, the
continuum expected distance is instead

$$
d_\epsilon[\rho](z)=
\frac{\int d(z,y)K_\epsilon(z,y)\rho(y)\,dy}
{\int K_\epsilon(z,y)\rho(y)\,dy},
$$

provided the denominator is positive and the numerator finite. This is the
conditional law of one kernel-selected companion. Its regularity follows
from the normalized-moment estimates in the regularity chapters.
:::

:::{prf:proof}
The nearest-neighbor event is precisely that the ball of radius $r$ contains
zero Poisson points. Its count has mean $\lambda\omega_Dr^D$, giving the
survival function. Integrating it over $r\geq0$ and substituting
$s=\lambda\omega_Dr^D$ gives the gamma factor. For kernel selection,
normalize the measure $K_\epsilon(z,y)\rho(y)dy$ and integrate $d(z,y)$.
A nearest-neighbor rule and a fixed-width kernel rule define different
sampling laws, so their expectations require their respective formulas.
:::

(sec-equilibrium-selection)=
## The Iso-Fitness Reference Model

:::{div} feynman-prose
Consider a local model in which a region gains population whenever its fitness exceeds the population average. At a positive stationary density, every occupied region must have the same fitness; otherwise some region would still gain or lose mass.

If crowding reduces the diversity contribution, that equality determines how much density a high-reward region can support. The power law below follows directly from this balance. It is a useful reference calculation because every step is explicit. Applying its profile to the Fractal Gas requires checking the nonlocal cloning and kinetic terms as well.
:::

:::{prf:theorem} Power-law equilibrium for the local replicator model
:label: thm-cloning-equilibrium

Consider the specified local model

$$
\partial_t\rho=(V[\rho]-\overline V[\rho])\rho,\qquad
V[\rho](z)=a\rho(z)^{-\beta/D}R(z)^\alpha,
$$

where $a,\beta>0$, $R>0$, and the displayed quantities are integrable.
Among strictly positive probability densities, its stationary density is

$$
\rho_*(z)=Z^{-1}R(z)^{\alpha D/\beta},\qquad
Z=\int R(z)^{\alpha D/\beta}\,dz<\infty.
$$

For $R=e^{-U}$, this is the reference Gibbs law with inverse temperature
$\alpha D/\beta$. The theorem concerns this local replicator equation;
the nonlocal sampled-fitness cloning operator is the operator in
{doc}`08_mean_field`.
:::

:::{prf:proof}
Stationarity and strict positivity give $V[\rho]=\overline V[\rho]$ almost
everywhere. Solving the resulting algebraic equation gives the displayed
power law; normalization fixes its constant. Conversely, substitution makes
$V[\rho_*]$ constant, so the right side vanishes. This also proves uniqueness
in the stated class. If $Z=\infty$, the displayed profile is not a probability
density. On an unbounded space the limit $\alpha/\beta\to0$ cannot be
identified with a uniform probability distribution.
:::

:::{div} feynman-prose
Writing the reward as an exponential turns the power law into a Gibbs-shaped curve. This is an algebraic consequence of the chosen reward and local fitness formula. The name “Gibbs” does not by itself identify the stationary law of a different dynamics. The residual estimate at the end of the chapter provides a way to test that identification quantitatively.
:::

(sec-equilibrium-kinetic)=
## Kinetic Equilibria and Absorption

:::{div} feynman-prose
A boundary can change an equilibrium profile even when the motion inside is simple diffusion. With absorption and no source, mass continually disappears; a quasi-stationary profile preserves its shape while its amplitude decays. With a source, the incoming mass can instead balance the boundary loss. The sine and parabola below are the two solutions of these two balance equations.
:::

:::{prf:proposition} Absorbing diffusion profile on an interval
:label: prop-halo-density

For $\partial_t f=D_0\partial_{xx}f$ on $(0,L)$ with $D_0>0$ and absorbing
Dirichlet endpoints, the normalized positive eigenfunction

$$
\phi(x)=\frac\pi{2L}\sin(\pi x/L)
$$

is a QSD density with decay rate $\lambda_0=D_0\pi^2/L^2$.
For a prescribed source $s$, a stationary forced density instead solves
$D_0f''+s=0$. In particular, a constant source gives
$f(x)=s x(L-x)/(2D_0)$.
:::

:::{prf:proof}
The sine has integral $2L/\pi$, vanishes at both endpoints, and satisfies
$D_0\phi''=-\lambda_0\phi$. Thus the killed evolution starting from it is
$e^{-\lambda_0t}\phi$ and normalization leaves $\phi$. Twice integrating
the constant-source equation and imposing the endpoints gives the parabola.
For a density with killing and revival, use its full source balance rather
than the source-free eigenvalue equation. The kinetic transport boundary
condition also distinguishes incoming and outgoing velocities; the scalar
Dirichlet calculation applies to the specified spatial diffusion model.
:::

:::{div} feynman-prose
Velocity has another useful reference calculation. Friction damps the initial velocity, while noise continually adds fresh fluctuations. In the Ornstein–Uhlenbeck equation these effects can be solved exactly. Their balance gives a Gaussian velocity law and an explicit rate at which the initial velocity is forgotten.

That calculation isolates the thermal part of the dynamics. A collision, reset, or selection event can change the velocity statistics, so it must also appear in the balance equation before we claim that the full velocity marginal is Maxwellian.
:::

:::{prf:theorem} Ornstein–Uhlenbeck thermalization and its reference law
:label: thm-velocity-thermalization

For $dV_t=-\gamma V_tdt+\sigma_vdB_t$, $\gamma>0$, set
$T=\sigma_v^2/(2\gamma)$. The invariant law is
$M_T=\mathcal N(0,TI_d)$ and

$$
W_2(\mathcal L(V_t),M_T)\leq
 e^{-\gamma t}W_2(\mathcal L(V_0),M_T).
$$

For conservative underdamped Langevin dynamics with force $-\nabla U$,
the normalized density
$\rho_U(x,v)=Z^{-1}\exp[-(U(x)+|v|^2/2)/T]$ is invariant when $Z<\infty$
and the generator domain has zero boundary flux. Cloning and killing require
checking their contributions to the stationary equation.
:::

:::{prf:proof}
Variation of constants gives
$V_t=e^{-\gamma t}V_0+\sigma_v\int_0^te^{-\gamma(t-s)}dB_s$.
The Gaussian integral has covariance $T(1-e^{-2\gamma t})I_d$, proving
invariance. Couple two solutions with the same Brownian motion; their
difference is $e^{-\gamma t}(V_0-W_0)$. Optimize the initial coupling to
obtain the Wasserstein estimate. For $\rho_U$, the transport term
$-v\cdot\nabla_x\rho_U$ cancels $\nabla U\cdot\nabla_v\rho_U$;
friction cancels velocity diffusion by $\sigma_v^2=2\gamma T$.
:::

:::{prf:remark} A finite jump rate retains a velocity correction
:label: rem-equilibrium-jump-temperature

For the preceding OU velocity with independent resets to zero at rate
$r>0$, its generator applied to $|v|^2$ gives

$$
\frac{d}{dt}\mathbb E|V_t|^2=-(2\gamma+r)\mathbb E|V_t|^2+d\sigma_v^2.
$$

Its stationary second moment is $d\sigma_v^2/(2\gamma+r)$, rather than
$dT$. Thus a finite jump rate alone does not imply an exactly Maxwellian
stationary velocity marginal. In the Fractal Gas the actual collision,
selection, killing, and revival terms determine the corresponding correction.
:::

(sec-equilibrium-profile-validation)=
## Quantitative Validation of Equilibrium Profiles

:::{div} feynman-prose
Suppose we have drawn a plausible equilibrium curve. How can we tell whether it is close to the true one? Insert it into the stationary equation and measure the leftover term. A zero residual gives an exact stationary solution. A small residual gives a useful error bound when the stationary solution map contracts.

The factor below explains the role of contraction. If the map contracts strongly, it cannot move a point only slightly while that point remains far from its fixed point. If the contraction factor is close to one, a small residual is less informative. This turns a proposed decorated Gibbs profile into a testable approximation.
:::

:::{prf:theorem} Error bound for a proposed decorated Gibbs profile
:label: thm-decorated-gibbs

Under {prf:ref}`thm-uniqueness-contraction-solution-operator`, let
$\mathcal T_C$ have contraction factor $q_C<1$ and stationary density $u_*$.
For any proposed normalized density

$$
g=Z^{-1}e^{-V_{\rm eff}}\Xi\geq0,
$$

in the same complete space,

$$
\|g-u_*\|_1\leq\frac{\|\mathcal T_Cg-g\|_1}{1-q_C}.
$$

If $g$ belongs to the generator domain, its stationary residual
$r=Ag+\mathcal R(g)$ has zero mass. Consequently

$$
\|g-u_*\|_1\leq
\frac{K}{(C+a)(1-q_C)}\|r\|_1,
$$

with the zero-mass semigroup constants $K,a$ from
{prf:ref}`lem-uniqueness-scaling-hypoelliptic-constant`.
:::

:::{prf:proof}
Insert $\mathcal T_Cg$ between $g$ and $u_*=\mathcal T_Cu_*$, then use
contraction and move $q_C\|g-u_*\|_1$ to the left. The resolvent identity
$R_C(C-A)g=g$ gives
$\mathcal T_Cg-g=R_C[Ag+\mathcal R(g)]$. Conservation makes the residual
zero-mass, so the sharper resolvent bound $K/(C+a)$ applies.
:::

:::{prf:corollary} Equilibrium observables and particle fluctuations
:label: cor-equilibrium-observable-control

For a bounded observable $\psi$, the preceding profile error implies
$|g\psi-u_*\psi|\leq\|\psi\|_\infty\|g-u_*\|_1$.
For an exchangeable particle law, its empirical-observable error also
contains marginal bias and sampling variance, as quantified in
{prf:ref}`thm-total-error-bound`. The uniform-in-$N$ LSI and total-relative-
entropy routes in {doc}`12_qsd_exchangeability_theory` and
{doc}`15_kl_convergence` supply the corresponding $O(N^{-1})$ variance bounds
under their analytical hypotheses.

*Proof.* Integrate $\psi(g-u_*)$ and apply the $L^\infty$–$L^1$ inequality.
For an empirical average split the error into its centered fluctuation and
its expectation; Cauchy–Schwarz bounds the former by its standard deviation.
:::

:::{div} feynman-prose
A simulation adds fluctuations to the profile comparison. Averaging more walkers can reduce the random part of an observable's error under the stated concentration hypotheses. It does not automatically reduce an error in the proposed equilibrium shape. The stationary residual controls that shape error; the finite-particle estimates control the remaining fluctuations and marginal bias.
:::
