# The Mean-Field Model and Its Forward Equation

(sec-mean-field-foundations)=
## 1. Alive mass, probability laws, and measurements

:::{div} feynman-prose
There are two limits to distinguish. Increasing the number of walkers turns empirical measurements into functionals of a probability law. Decreasing the integration step replaces a discrete transition by a continuous evolution only when its transition rates and boundary rules have the required limit. This chapter derives the forward equation for a specified continuous-time model and states the estimates that connect it to particles.

The model tracks an alive density and a dead reservoir. Transport moves the alive density, killing transfers mass to the reservoir, revival returns it, and cloning redistributes it according to a companion rule. A probability-preserving offspring kernel makes cloning mass-neutral. An offspring kernel that can leave the alive domain adds another loss term.

The generator assembly, population balance, positive alive-mass bound, boundary-flux limit, and particle-coupling transfer are proved below. Existence, uniqueness, and stationary mean-field analysis use these operators in {doc}`09_propagation_chaos`. The underlying algorithm and its discrete updates are specified in {doc}`02_euclidean_gas`, {doc}`03_cloning`, and {doc}`04_single_particle`.
:::

### 1.1. State space and normalization

:::{prf:definition} Phase space
:label: def-mean-field-phase-space

Let $X_{\mathrm{valid}}\subset\mathbb R^d$ be the valid position domain and let $V$ be the velocity state space. Write $\Omega=X_{\mathrm{valid}}\times V$ and $z=(x,v)$. For the uncapped kinetic diffusion, $V=\mathbb R^d$. A bounded velocity ball requires a separately specified boundary law. The boundary-flux calculation in Section 5 uses a bounded $C^2$ position domain and bounded velocities; the conservative kinetic model below can instead use a confining unbounded domain with justified decay at infinity.
:::

(remark-mean-field-cloud)=
:::{admonition} From particles to a law
:class: feynman-added note

An $N$-particle configuration is a point in a large state space. Its empirical measure places mass $1/N$ at each particle state. A mean-field law describes a representative particle. A density is one representation of that law when it is absolutely continuous; an empirical measure itself is atomic.
:::

:::{prf:definition} Alive density and dead mass
:label: def-phase-space-density

Let $f(t,z)\geq0$ be an alive sub-probability density with

$$
m_a(t)=\int_\Omega f(t,z)dz,\qquad m_d(t)=1-m_a(t).
$$

On the region $m_a(t)>0$, define the conditional alive probability density

$$
\rho_t(z)=\frac{f(t,z)}{m_a(t)}.
$$

The weak forward equation is interpreted in time-integrated form. Whenever a density description is used, $f\in C([0,T];L^1(\Omega))$ is a natural solution class, supplemented by the test-function, boundary-trace, and reaction-integrability requirements of the particular result.
:::

:::{prf:remark} What density continuity provides
:label: remark-mean-field-regularity

Continuity into $L^1$ makes the alive mass continuous and permits a mild-solution formulation. It does not by itself give pointwise derivatives or boundary traces. The forward identities below specify how those stronger operations are justified.
:::

:::{prf:remark} Averaging over the alive population
:label: remark-mean-field-sum-to-integral

The finite average $k^{-1}\sum_{i\in A}Q(z_i)$ corresponds to $\int Q(z)\rho_t(z)dz$, where $\rho_t=f/m_a$. Integrating against $f$ instead gives the unnormalized alive contribution. Products $\rho_t(dz)\rho_t(dz')$ below describe independent uniform alive companion sampling; a nonuniform or correlated companion mechanism uses its actual conditional or joint kernel.
:::

:::{prf:definition} Mean-Field Statistical Moments
:label: def-mean-field-moments

Let $f(t, \cdot)$ be the phase-space density (see {prf:ref}`def-phase-space-density`) at time $t$, with total alive mass $m_a(t) = \int_\Omega f(t,z)\,\mathrm{d}z$. The statistical moments required for the standardization pipeline are defined as the following **functionals** of $f$. The notation $\mu[f]$ emphasizes that these are numbers that depend on the entire *shape* of the function $f$.

The moments are computed with respect to the **normalized density of the alive population**, which is $f(t,z) / m_a(t)$. This normalization is critical for ensuring the mean-field model is a faithful limit of the N-particle system, where statistics are computed by averaging over the $k$ alive walkers.

*   **Reward Moments:** The mean reward, $\mu_R[f]$, is computed as the expected value over the normalized alive population:

    $$
    \mu_R[f](t) := \int_{\Omega} R(z) \frac{f(t,z)}{m_a(t)}\,\mathrm dz

    $$

    $$
    \sigma_R^2[f](t) := \int_{\Omega} \bigl(R(z) - \mu_R[f](t)\bigr)^2 \frac{f(t,z)}{m_a(t)}\,\mathrm dz

    $$

*   **Distance Moments:** The mean distance is the expectation of the distance between two particles drawn independently from the normalized alive population:

    $$
    \mu_D[f](t) := \iint_{\Omega \times \Omega} d_{\mathcal{Y}}(\varphi(z), \varphi(z')) \frac{f(t,z)}{m_a(t)} \frac{f(t,z')}{m_a(t)}\,\mathrm dz\,\mathrm dz'

    $$

    $$
    \sigma_D^2[f](t) := \iint_{\Omega \times \Omega} \bigl(d_{\mathcal{Y}}(\varphi(z), \varphi(z')) - \mu_D[f](t)\bigr)^2 \frac{f(t,z)}{m_a(t)} \frac{f(t,z')}{m_a(t)}\,\mathrm dz\,\mathrm dz'

    $$
:::

:::{prf:remark} Positive mass and the cemetery boundary
:label: remark-cemetery-state

The normalized moments are defined while $m_a>0$. For bounded killing and positive revival, {prf:ref}`cor-mean-field-positive-alive-mass` proves that positive initial alive mass stays bounded away from zero on every finite interval. No reference distribution at extinction is needed for those solutions.

At exactly $m_a=0$, the ratio $f/m_a$ has no canonical value. An extension that keeps extinction absorbing must switch off companion-based revival there; an extension using a prescribed restart law is a different model. Neither extension follows by assigning a limit to arbitrary paths $f\to0$. The results here concern the positive-mass model and its stated boundary law.
:::

### 1.2. Standardization and fitness

:::{prf:definition} Regularized standard deviations
:label: def-mean-field-patched-std

Apply the specified variance regularization $\sigma'_{\mathrm{reg}}$ to the reward and distance variances:

$$
\widehat\sigma_R[f]=\sigma'_{\mathrm{reg}}(\sigma_R^2[f]),\qquad
\widehat\sigma_D[f]=\sigma'_{\mathrm{reg}}(\sigma_D^2[f]).
$$

The regularization is chosen so both denominators are at least a fixed $s_*>0$. A smooth floor and a piecewise patch have their respective derivative domains; those domains remain part of subsequent regularity estimates.
:::

:::{prf:definition} Mean-Field Z-Scores
:label: def-mean-field-z-scores

For a particle at state $z$ and a potential companion at state $z_c$, the mean-field Z-scores at time $t$ are defined using the density-dependent functionals derived in Section 1.2. The means $\mu_R[f]$ and $\mu_D[f]$ are from {prf:ref}`def-mean-field-moments`, and the regularized standard deviations $\widehat{\sigma}_R[f]$ and $\widehat{\sigma}_D[f]$ are from {prf:ref}`def-mean-field-patched-std`:

$$
\widetilde{r}[f](z,t) := \frac{R(z) - \mu_R[f](t)}{\widehat{\sigma}_R[f](t)}, \qquad \widetilde{d}[f](z,z_c,t) := \frac{d_{\mathcal{Y}}(\varphi(z),\varphi(z_c)) - \mu_D[f](t)}{\widehat{\sigma}_D[f](t)}

$$
These Z-scores measure how many "global standard deviations" a particle's raw reward or its distance to a companion is from the swarm's current average. A positive Z-score indicates an above-average measurement.
:::

:::{prf:definition} Mean-Field Fitness Potential
:label: def-mean-field-fitness-potential

The **Mean-Field Fitness Potential**, denoted $V[f](z, z_c, t)$, is a functional of the density $f$ that determines the fitness of a particle at state $z$ relative to a companion at $z_c$. It is constructed using the specified nonnegative rescaling map $g_A$, floor $\eta>0$, and nonnegative exponents $\alpha,\beta$ to the mean-field Z-scores (see {prf:ref}`def-mean-field-z-scores`):

$$
V[f](z,z_c,t) := \left(g_A(\widetilde{d}[f](z,z_c,t)) + \eta\right)^{\beta} \cdot \left(g_A(\widetilde{r}[f](z,t)) + \eta\right)^{\alpha}

$$
The positive floor makes this specified fitness strictly positive.
:::

:::{div} feynman-prose
The order of the operations matters. Compute the moments, standardize each sampled measurement, apply the rescaling map, form fitness, and then evaluate acceptance. Averaging an acceptance probability over sampled companions generally differs from applying acceptance to averaged fitness. The mean-field kernel must retain that order.

The equation is nonlocal because its coefficients contain integrals over the alive law. It is nonlinear because normalization, variance, and fitness depend on that law. These are concrete dependencies that can be estimated, rather than reasons to assume the equation has a particular stationary density.
:::

:::{prf:remark} The nonlinear acceptance kernel
:label: remark-important-nonlocal-nonlinear

Write $P_\rho(z_d,z_c)\in[0,1]$ for acceptance after averaging any additional sampled measurement companions according to their actual joint law. The displayed $V[f](z,z_c)$ is the specified fitness field before this acceptance average. The canonical sampled-versus-expected distinction is {prf:ref}`rem-mean-field-fitness-field-latent`.
:::

(sec-mean-field-kinetic)=
## 2. Kinetic transport and the continuous-time scale

:::{div} feynman-prose
The kinetic update is built from force kicks, position drifts, and an exactly solved Ornstein–Uhlenbeck velocity step. Its interior infinitesimal generator is straightforward to calculate. Boundary reflection, clipping, and rejection are additional operations; their limiting behavior must be checked separately.

A fixed-step cloning probability also needs a time interpretation. Below, a finite attempt intensity defines a continuous jump model. If the algorithm's step is subsequently sent to zero, a finite jump generator requires the accepted jump probability to be of order the step.
:::

### 2.1. BAOAB and its interior generator

:::{prf:definition} The BAOAB Update Rule
:label: def-baoab-update-rule

For a single particle with state $(x_n, v_n)$ at time $t_n$, the state $(x_{n+1}, v_{n+1})$ at time $t_{n+1} = t_n + h$ is computed via the following five steps:

1.  **B-Step (Force Kick):** The velocity is updated with a half-step kick from the conservative force $F(x)$.

    $$
    v_{n+1/2}^{(1)} = v_n + \frac{h}{2m} F(x_n)

    $$

2.  **A-Step (Position Drift):** The position is updated with a half-step drift using the new velocity.

    $$
    x_{n+1/2} = x_n + \frac{h}{2} v_{n+1/2}^{(1)}

    $$

3.  **O-Step (Ornstein-Uhlenbeck):** The velocity is updated for a full timestep by exactly solving the Ornstein-Uhlenbeck process that combines friction and thermal noise. Let $u_{n+1/2} = u(x_{n+1/2})$ be the flow field evaluated at the midpoint.

    $$
    v_{n+1/2}^{(2)} = u_{n+1/2} + e^{-\gamma_{\mathrm{fric}}h}\left(v_{n+1/2}^{(1)} - u_{n+1/2}\right) + \sqrt{\frac{\Theta}{m}(1 - e^{-2\gamma_{\mathrm{fric}}h})} \cdot \xi

    $$
    where $\xi \sim \mathcal{N}(0, I_d)$ is a standard Gaussian random vector.

4.  **A-Step (Position Drift):** The position is updated with a final half-step drift.

    $$
    x_{n+1} = x_{n+1/2} + \frac{h}{2} v_{n+1/2}^{(2)}

    $$

5.  **B-Step (Force Kick):** The velocity is updated with a final half-step kick using the force evaluated at the new position, $F(x_{n+1})$.

    $$
    v_{n+1} = v_{n+1/2}^{(2)} + \frac{h}{2m} F(x_{n+1})

    $$

An optional finite-step velocity cap $\psi_v$ can be applied after the final B-step. Its limiting boundary law must be identified separately; the uncapped interior formula above defines the kinetic approximation used here.
:::

:::{prf:remark} What the splitting identifies
:label: remark-fidelity-generator

For the uncapped interior BAOAB update, the force and drift increments are $O(h)$ and the OU covariance is $\sigma_v^2hI+O(h^2)$. Taylor expansion against a smooth test function yields the kinetic generator below. Applying a velocity squash or cap can change the limiting boundary behavior; it is not automatically a reflecting diffusion. The five-stage discrete kernel is also not the exact finite-time kinetic semigroup.
:::

:::{prf:definition} Backward kinetic generator
:label: def-kinetic-generator

The continuous interior dynamics is

$$
dX_t=V_tdt,\qquad
dV_t=A_v(X_t,V_t)dt+\sigma_vdW_t,
\qquad
A_v(x,v)=m^{-1}F(x)-\gamma_{\mathrm{fric}}(v-u(x)),
$$

with $\sigma_v^2=2\gamma_{\mathrm{fric}}\Theta/m$ for the stated BAOAB temperature convention. Its backward generator acts on an observable $\psi$ as

$$
L\psi=v\cdot\nabla_x\psi+A_v\cdot\nabla_v\psi
+\frac{\sigma_v^2}{2}\Delta_v\psi.
$$

For independent kinetic particle updates, sum these terms over the alive coordinates. A conservative position boundary can use specular reflection with matching incoming and outgoing traces. A bounded velocity domain can use a declared no-flux reflecting law. Alternatively, absorb spatial exits and record their outgoing flux. These are distinct operator domains.
:::

:::{prf:remark} Transport and death
:label: remark-separation-kinetic-death

The population equation in Section 4 first uses a conservative transport semigroup and a prescribed interior killing rate. A spatially absorbing kinetic model replaces that conservative boundary law and adds its outgoing flux to the dead reservoir. Section 5 computes the corresponding discrete exit limit.
:::

### 2.2. Forward flux and mass conservation

:::{prf:definition} Forward transport and probability flux
:label: def-transport-operator

For the constant velocity diffusion above, the forward adjoint acts on densities by

$$
L^\dagger f=-\nabla_x\cdot(vf)-\nabla_v\cdot(A_vf)
+\frac{\sigma_v^2}{2}\Delta_vf=-\nabla\cdot J[f],
$$

where $J_x=vf$ and $J_v=A_vf-(\sigma_v^2/2)\nabla_vf$. An independently specified position diffusion adds its own second-order term and flux; it is absent from this kinetic model.
:::

:::{prf:lemma} Mass conservation of conservative transport
:label: lem-mass-conservation-transport

For the stated conservative domain, with integrable flux and justified boundary traces or cutoff limits,

$$
\int_\Omega L^\dagger f\,dz=0.
$$
:::

:::{prf:proof}
Integrate the divergence to obtain the negative total boundary flux. At a specular spatial boundary, pair $v$ with $R_nv=v-2(v\cdot n)n$. The reflection preserves velocity volume and reverses $v\cdot n$; equality of the incoming and outgoing traces cancels the integrated spatial flux. At a reflecting velocity boundary, $J_v\cdot n_v=0$. On an unbounded domain use the stated vanishing-flux cutoff limit. These conditions eliminate the total flux; kinetic specular reflection need not set $vf\cdot n$ to zero pointwise for each velocity.
:::

(sec-mean-field-reactions)=
## 3. Killing, revival, and the cloning kernel

:::{div} feynman-prose
Track one transition at a time. Killing removes an alive particle. Revival draws a replacement from a specified alive-companion law. An internal cloning attempt removes the donor's old state and inserts its offspring state. Integrating that last difference gives zero exactly when the offspring stays in the alive state space with probability one.
:::

:::{prf:remark} Three population operations
:label: remark-separation-death-revival-cloning

Interior killing, reservoir revival, and alive-to-alive cloning have separate rates. A finite revival rate gives a waiting-time model for dead mass; it is not instantaneous resurrection at every numerical step. Matching an algorithm requires its actual scheduling and transition probabilities.
:::

:::{prf:definition} Interior killing
:label: def-killing-operator

Let $c:\Omega\to[0,\infty)$ be a prescribed rate. Its density contribution is $-cf$ and its total alive loss is

$$
k_{\mathrm{killed}}[f]=\int_\Omega c(z)f(z)dz.
$$

A smooth bounded rate supported in a boundary layer is one possible reaction model. The integrated kinetic exit limit in Section 5 does not define such a rate by a pointwise limit.
:::

:::{prf:definition} Finite-rate reservoir revival
:label: def-revival-operator

For $m_a>0$, copying a uniformly sampled alive companion without a further state change gives

$$
B[f,m_d](z)=\lambda_{\mathrm{revive}}m_d\frac{f(z)}{m_a},
\qquad\lambda_{\mathrm{revive}}>0.
$$

Its integral is $\lambda_{\mathrm{revive}}m_d$. If revival also applies a state-transition kernel, replace $f/m_a$ by the pushforward under that kernel and retain any probability lost from the alive domain.
:::

:::{prf:definition} Continuous-time cloning attempts
:label: def-cloning-generator

Fix a finite attempt rate $\omega_{\mathrm{cl}}\geq0$. Let $\rho=f/m_a$, let $P_\rho(z_d,z_c)$ be the acceptance kernel, and let $Q_\rho(dz\mid z_d,z_c)$ be the offspring law after acceptance, including the specified velocity update. For independent uniform alive donor and companion sampling, the weak cloning operator is

$$
\int\psi\,S[f]
=\omega_{\mathrm{cl}}m_a\iint\rho(dz_d)\rho(dz_c)P_\rho(z_d,z_c)
\left[\int\psi(z)Q_\rho(dz\mid z_d,z_c)-\psi(z_d)\right].
$$

When an offspring density exists,

$$
S_{\mathrm{src}}[f](z)=\frac{\omega_{\mathrm{cl}}}{m_a}
\iint f(z_d)f(z_c)P_\rho(z_d,z_c)Q_\rho(z\mid z_d,z_c)dz_d\,dz_c,
$$

$$
S_{\mathrm{sink}}[f](z)=\omega_{\mathrm{cl}}f(z)
\int P_\rho(z,z_c)\rho(z_c)dz_c,\qquad S=S_{\mathrm{src}}-S_{\mathrm{sink}}.
$$

A donor-independent jitter model is the special case $Q_\rho(dz\mid z_d,z_c)=Q_\delta(dz\mid z_c)$. The formulas below allow either case; $Q$ denotes the selected offspring kernel. The operator is mass-neutral when $Q(\Omega\mid z_d,z_c)=1$.
:::

:::{prf:proof}
In an interval of length $h$, an alive donor attempts a jump with probability $\omega_{\mathrm{cl}}h+o(h)$. Condition on its state, companion, acceptance, and offspring. The observable increment is its offspring value minus its donor value. Averaging gives the weak formula. Fubini identifies the source and sink when densities exist. With $\psi=1$, the bracket is zero for a probability kernel on $\Omega$, proving mass neutrality.
:::

:::{prf:remark} Fixed-step probabilities and Poissonization
:label: rem-mean-field-attempt-scaling

Choosing $\omega_{\mathrm{cl}}=1/\tau$ for a fixed reference step $\tau$ defines a Poissonized model with the same attempt frequency. Its finite-time transition is not the original simultaneous cloning step. A finite continuous-time limit as $h\downarrow0$ requires an accepted probability $P_h=h\,a+o(h)$, or an equivalent finite-attempt-rate construction. Keeping order-one acceptance at every shrinking step does not yield the finite generator above.
:::

(sec-mean-field-population)=
## 4. Generator assembly and the population balance

:::{div} feynman-prose
We can now assemble the equation without treating an unbounded transport operator as a bounded matrix. Strong continuity gives the first-order transport increment; the reaction increment adds to it on the generator domain. Testing the resulting weak equation against one then gives the population balance.

That constant test is useful because it sees exactly what a transition does to alive mass. A clone with a valid offspring changes no total mass. Killing and revival contribute equal and opposite terms to the alive and dead equations.
:::

:::{prf:lemma} First-order assembly of transport and reaction
:label: lem-generator-additivity-mean-field

Let $T_h$ be a strongly continuous semigroup with generator $A$ on a Banach
space $X$. Let a reaction map $R:X\to X$ be continuous at $u\in D(A)$, and
suppose its local update satisfies $S_hu=u+hR(u)+o_X(h)$. Then

$$
T_hS_hu=u+h(Au+R(u))+o_X(h).
$$

For finitely many locally differentiable reaction updates, their first-order
contributions add in the same way.
:::

:::{prf:proof}
Write

$$
T_hS_hu-u=(T_hu-u)+hT_hR(u)+T_ho_X(h).
$$

The first term is $hAu+o_X(h)$ by the generator definition. Strong continuity
gives $T_hR(u)\to R(u)$, and the uniform boundedness principle bounds $T_h$ on
a sufficiently short time interval. Thus the final term is $o_X(h)$.
For multiple reaction maps, telescope their compositions; continuity at $u$
replaces each intermediate value by $u$ in its first-order coefficient.
The argument does not require the differential operator $A$ to be bounded.
:::

:::{prf:theorem} Coupled continuous-time forward equation
:label: thm-mean-field-equation

For the transport and reaction model defined above, let $f\geq0$, $m_d\geq0$ be a weak solution on $[0,T]$ with $m_a=\int f>0$. Suppose transport is conservative on its stated domain, $Q$ is a probability kernel on $\Omega$, and the reaction terms are integrable in time and space. Then

$$
\partial_tf=L^\dagger f-cf+B[f,m_d]+S[f]
$$ (eq-mean-field-pde-main)

and

$$
\frac{d}{dt}m_d=\int_\Omega cf-\lambda_{\mathrm{revive}}m_d.
$$ (eq-dead-mass-ode)

With $f(0)=f_0$ and $m_d(0)=1-\int f_0$, the total population remains one. Identification with a discrete algorithm requires the matching transition-operator and boundary limits.
:::

:::{prf:proof}
:label: proof-mean-field-equation

Apply {prf:ref}`lem-generator-additivity-mean-field` with transport generator
$A=L^\dagger$ and reaction
$R(f,m_d)=-cf+B[f,m_d]+S[f]$, on its domain of differentiability.
For weak solutions the resulting identity is interpreted against a smooth
admissible test function $\psi$:

$$
\frac{d}{dt}\int_\Omega\psi f
=\int_\Omega(L\psi)f-\int_\Omega\psi cf
+\int_\Omega\psi B[f,m_d]+\int_\Omega\psi S[f].
$$

Equivalently, this is an equality integrated over every time interval
$[s,t]\subset[0,T]$. For interior compactly supported test functions,
$L\psi=A\cdot\nabla\psi+\mathsf D:D^2\psi$, which identifies the
transport distribution as
$L^\dagger f=-\nabla\cdot(Af)+\nabla\cdot(\mathsf D\nabla f)$ for constant
$\mathsf D$. A flux representation with boundary integration may also be used
when $J[f]\in H(\operatorname{div},\Omega)$ and its stated normal trace exists.

The transport conservation law supplies the admissible constant test function
$1$ (or its justified cutoff limit). The cloning source and sink cancel exactly:
by Tonelli's theorem and $\int Q(dz\mid z_d,z_c)=1$,

$$
\int S_{\mathrm{src}}[f]
=\frac{\omega_{\mathrm{cl}}}{m_a}\iint f(z_d)f(z_c)P_\rho(z_d,z_c)\,dz_d\,dz_c
=\int S_{\mathrm{sink}}[f].
$$

The revival term integrates to $\lambda_{\mathrm{revive}}m_d$ since
$\int f/m_a=1$. Thus the integrated weak identity gives

$$
m_a(t)-m_a(s)=\int_s^t\left[-\int_\Omega cf
+\lambda_{\mathrm{revive}}m_d\right]du.
$$

The integrand is integrable by hypothesis, proving absolute continuity of
$m_a$ without assuming its differentiability in advance. The dead-reservoir
balance is
$m_d'=\int cf-\lambda_{\mathrm{revive}}m_d$. Adding the two derivatives
shows that $m_a+m_d$ is constant, hence equals one for the stated initial data.
This proves the forward system and its mass balance.
:::

:::{prf:corollary} Positive alive mass in the continuous-time population model
:label: cor-mean-field-positive-alive-mass

If $0\leq c\leq C$ and $\lambda=\lambda_{\mathrm{revive}}>0$, every
nonnegative unit-mass solution satisfies

$$
m_a(t)\geq\frac{\lambda}{C+\lambda}
+\left(m_a(0)-\frac{\lambda}{C+\lambda}\right)e^{-(C+\lambda)t}.
$$

In particular $m_a(0)>0$ gives a positive lower bound on every finite time
interval, and a positive limiting lower bound as $t\to\infty$.
:::

:::{prf:proof}
The population equation gives
$m_a'\geq-Cm_a+\lambda(1-m_a)$ almost everywhere. Multiply by
$e^{(C+\lambda)t}$ and integrate. All terms on the right combine into the
stated solution of the scalar comparison equation. No inequality between
$\lambda$ and $C$ is required.
:::

:::{div} feynman-prose
The positive-mass estimate closes a potential circularity. The measurement formulas divide by alive mass, while the population equation itself contains those measurements. Bounded killing and revival keep the solution in a region where that normalization remains defined. The bound requires no assumption that revival is faster than killing.
:::

:::{prf:remark} Boundary loss in a sub-probability cloning kernel
:label: rem-mean-field-cloning-boundary-loss

If the actual offspring kernel has $Q(\Omega\mid z_d,z_c)<1$, the source
and sink do not cancel on the alive domain. Their integral is instead
$-\ell_Q[f]$, where

$$
\ell_Q[f]=\frac{\omega_{\mathrm{cl}}}{m_a}\iint f(z_d)f(z_c)P_\rho(z_d,z_c)
[1-Q(\Omega\mid z_d,z_c)]\,dz_d\,dz_c\geq0.
$$

This term is added to the dead-reservoir equation, preserving total mass. An
untruncated Gaussian on a bounded alive domain is such a sub-probability
kernel. The probability-kernel formulation above and the boundary-loss
formulation must be matched to the update being modeled. This identity follows
from the same Tonelli calculation, retaining the integral of $Q$. Since $0\leq P_\rho\leq1$, one also has $\ell_Q[f]\leq\omega_{\mathrm{cl}}m_a$. Consequently the positive alive-mass comparison still holds with $C+\omega_{\mathrm{cl}}$ in place of $C$, provided revival remains a probability-preserving injection.
:::

:::{prf:theorem} Total Mass Conservation and Population Dynamics
:label: thm-mass-conservation

Any sufficiently regular solution $(f(t,z), m_d(t))$ to the Mean-Field Equations (see {prf:ref}`thm-mean-field-equation`) satisfies the following properties:

**1. Total Mass Conservation:** The total population is conserved for all time $t>0$:

$$
\frac{\mathrm{d}}{\mathrm{d}t}\left[m_a(t) + m_d(t)\right] = 0

$$

where $m_a(t) = \int_\Omega f(t,z)\,\mathrm{d}z$. This implies that $m_a(t) + m_d(t) = 1$ for all $t$ if this holds initially.

**2. Alive Population Dynamics:** The alive mass evolves according to the balance between killing and revival:

$$
\frac{\mathrm{d}}{\mathrm{d}t}m_a(t) = \lambda_{\mathrm{revive}} m_d(t) - k_{\text{killed}}[f](t)

$$

where $k_{\text{killed}}[f] = \int_\Omega c(z)f(z)\,\mathrm{d}z$ is the instantaneous killing rate. At a stationary state the alive-mass equation requires $k_{\text{killed}}[f_\infty] = \lambda_{\mathrm{revive}} m_{d,\infty}$.
:::

:::{prf:proof}
We compute the time derivatives of both components and show they sum to zero.

**For the alive mass:** Integrate the equation for $\partial_t f$ over $\Omega$:

$$
\frac{\mathrm{d}}{\mathrm{d}t}m_a(t) = \frac{\mathrm{d}}{\mathrm{d}t}\int_\Omega f(t,z)\,\mathrm{d}z = \int_\Omega L^\dagger f\,\mathrm{d}z - \int_\Omega c(z)f\,\mathrm{d}z + \int_\Omega B[f, m_d]\,\mathrm{d}z + \int_\Omega S[f]\,\mathrm{d}z

$$

Evaluating each term using the properties established in previous sections:

1.  **Transport**: From {prf:ref}`lem-mass-conservation-transport`, $\int_\Omega L^\dagger f\,\mathrm{d}z = 0$ (the stated conservative boundary law)
2.  **Killing**: By definition, $\int_\Omega c(z)f\,\mathrm{d}z = k_{\text{killed}}[f]$
3.  **Revival**: From {prf:ref}`def-revival-operator`, $\int_\Omega B[f, m_d]\,\mathrm{d}z = \lambda_{\text{revive}} m_d(t)$
4.  **Internal cloning**: From {prf:ref}`def-cloning-generator`, $\int_\Omega S[f]\,\mathrm{d}z = 0$

Therefore:

$$
\frac{\mathrm{d}}{\mathrm{d}t}m_a(t) = 0 - k_{\text{killed}}[f] + \lambda_{\mathrm{revive}} m_d(t) + 0 = -k_{\text{killed}}[f] + \lambda_{\mathrm{revive}} m_d(t)

$$

**For the dead mass:** From the second equation:

$$
\frac{\mathrm{d}}{\mathrm{d}t}m_d(t) = k_{\text{killed}}[f] - \lambda_{\mathrm{revive}} m_d(t)

$$

**Sum:** Adding these two equations:

$$
\frac{\mathrm{d}}{\mathrm{d}t}\left[m_a(t) + m_d(t)\right] = \left[-k_{\text{killed}}[f] + \lambda_{\mathrm{revive}} m_d(t)\right] + \left[k_{\text{killed}}[f] - \lambda_{\mathrm{revive}} m_d(t)\right] = 0

$$

This demonstrates that the total mass is conserved for all time, completing the proof.

:::

(sec-mean-field-boundary)=
## 5. Boundary exits and the continuous-time identification

:::{div} feynman-prose
At any fixed interior point, a sufficiently short kinetic step almost never exits the domain. Yet a thin layer of points next to the boundary has enough mass to produce a finite outgoing flux. Taking the limit pointwise therefore misses the boundary contribution. The correct calculation integrates over that shrinking layer before dividing by the step.

The next theorem preserves the kinetic position-noise scale: velocity noise contributes a position displacement of order $h^{3/2}$. Independent position diffusion would have a different scale and a different boundary calculation.
:::

:::{prf:assumption} Domain regularity for boundary flux
:label: assumption-domain-regularity

Let $D\subset\mathbb R^d$ be the bounded spatial domain with $C^2$ boundary
and a tubular neighborhood of positive width. Velocities range over a bounded
set $V$. Write $n(y)$ for the outward unit normal at $y\in\partial D$.
:::

:::{prf:assumption} Gaussian position update
:label: assumption-integrator-regularity

Write the position update as
$Y_h=x+hv+r_h(x,v)+B_h(x,v)\xi$, with $\xi\sim N(0,I_d)$,
$\sup\|r_h\|\leq C h^2$ and $\sup\|B_h\|\leq C h^{3/2}$.
This is the kinetic-noise position scaling for the stated splitting update.
An independent position diffusion of order $h^{1/2}$ is a different boundary
scaling and is treated through its diffusive flux.
:::

:::{prf:assumption} Density regularity for the boundary flux limit
:label: assumption-density-regularity-killing

Let $f(x,v)$ be bounded, nonnegative and continuous up to the spatial boundary,
with bounded spatial derivative, and integrable on $D\times V$.
For an approximating family $f_h$, any additional density error is retained
explicitly as $h^{-1}\|f_h-f\|_{L^1}$ unless a stronger trace estimate is supplied.
:::

:::{prf:theorem} Pointwise exit probabilities and integrated kinetic boundary flux
:label: thm-killing-rate-consistency

Under the preceding assumptions, each fixed interior state satisfies

$$
\lim_{h\downarrow0}\frac{\mathbb P(Y_h\notin D)}{h}=0.
$$

Nevertheless the integrated exit fraction has the nonzero boundary limit

$$
\frac1h\int_{D\times V}f(x,v)\mathbb P(Y_h\notin D)\,dx\,dv
=\int_{\partial D\times V}(v\cdot n(y))_+f(y,v)\,dS(y)\,dv+O(\sqrt h).
$$

For $f_h$ in place of $f$, the absolute error increases by at most
$h^{-1}\|f_h-f\|_{L^1}$. The exit mechanism is a boundary flux; it is not a
nonzero smooth interior killing density obtained from the pointwise limit.
:::

:::{prf:proof}
Fix $x\in D$ at distance $r>0$ from the boundary. For small $h$,
$\|hv+r_h\|\leq r/2$. Exiting then requires
$\|\xi\|\geq r/(2Ch^{3/2})$. Exponential Markov inequality gives

$$
\mathbb P(Y_h\notin D)
\leq2^{d/2}\exp\!\left[-\frac{r^2}{16C^2h^3}\right]=o(h).
$$

This proves the pointwise claim, including points in any fixed boundary collar.

For the integrated claim, first take the ballistic update $x\mapsto x+hv$.
Its exiting initial positions lie in a collar of width $h\sup_V\|v\|$.
Write them as $x=y-rn(y)$. The tubular-coordinate Jacobian is $1+O(r)$,
and Taylor expansion of the signed distance shows that the exiting interval
in $r$ differs from $[0,h(v\cdot n(y))_+]$ by a set of length $O(h^2)$,
uniformly in $y,v$. The density satisfies
$f(y-rn(y),v)=f(y,v)+O(r)$. Integration gives

$$
\int f(x,v)\mathbf1_{x+hv\notin D}\,dx\,dv
=h\int_{\partial D\times V}(v\cdot n)_+f\,dS\,dv+O(h^2).
$$

For the Gaussian update, the two exit indicators can differ only when
$x+hv$ is within distance
$C(h^2+h^{3/2}\|\xi\|)$ of $\partial D$. A bounded smooth domain has collar
volume at most a constant times its width for small widths. For larger widths,
the finite volume of $D$ gives the same bound after increasing the constant.
Boundedness of $f$ and $V$ therefore bounds the difference of the integrated
exit fractions by

$$
C'\mathbb E(h^2+h^{3/2}\|\xi\|)=O(h^{3/2}).
$$

Divide by $h$ to obtain the result. Finally the exit probability is at most
one, so replacing $f$ by $f_h$ contributes at most
$h^{-1}\|f_h-f\|_{L^1}$.
:::

:::{prf:remark} Matching the boundary law
:label: remark-killing-rate-interpretation

The continuous reaction model with reflecting transport and prescribed
interior rate $c$ has the population equations of
{prf:ref}`thm-mean-field-equation`. A boundary-killed kinetic model instead
uses its outward flux in the dead-reservoir balance. Equality of these models
requires an approximation theorem for the chosen killing layer; the pointwise
exit limit above does not supply a nonzero interior rate.
:::

:::{prf:remark} Regularity of the prescribed reaction model
:label: remark-important-killing-rate-well-posedness

A bounded prescribed killing rate permits the positive alive-mass estimate
{prf:ref}`cor-mean-field-positive-alive-mass`. The existence and uniqueness
arguments in {doc}`09_propagation_chaos` specify the transport semigroup and
reaction Lipschitz bounds for the model to which they apply.
:::

:::{prf:remark} Numerical comparison of boundary losses
:label: remark-numerical-validation-killing-rate

For the kinetic position-noise scaling, the integrated quantity to compare
with measured exits per unit time is the boundary integral in
{prf:ref}`thm-killing-rate-consistency`. A position-diffusive model requires
its diffusive normal flux as well. The time-step scale, kernel and boundary
convention must agree in the simulation and the analytical comparison.
:::

(sec-mean-field-analysis)=
## 6. Well-posedness, stationary laws, and particle limits

:::{div} feynman-prose
The equation now has a specified transport semigroup, reaction kernel, mass normalization, and boundary law. These are the inputs for existence and uniqueness. The same identification is needed before transferring a particle estimate to the PDE.

There are two errors in that transfer: interacting particles differ from independent copies of the limiting process, and a finite sample of those independent copies differs from their common law. The final theorem keeps both errors visible. For a single Lipschitz observable, the second error has the familiar variance-over-population form; empirical Wasserstein distance has additional dimension dependence.
:::

:::{prf:assumption} Analytic setting for the specified reaction model
:label: assumption-regularity-summary

Use a positive conservative strongly continuous transport semigroup on the chosen function or measure space. Require the normalized moment, acceptance, and offspring maps to satisfy the local Lipschitz and integrability bounds used by the solution theorem, on sets with $m_a\geq a_*>0$. The prescribed killing rate is bounded, and the finite revival and attempt rates are fixed. Boundary domains and any velocity cap have the meanings stated in Section 2.

The concrete moment and cloning estimates in {prf:ref}`lem-uniqueness-lipschitz-moments` and {prf:ref}`lem-uniqueness-lipschitz-cloning-operator` provide these inputs for their specified coefficient class. Smoothness or measurability alone does not establish all of them.
:::

:::{prf:remark} Existing existence and stationary results
:label: rem-mean-field-analytic-results

The mild-solution construction and uniqueness proof are {prf:ref}`thm-chaos-mild-wellposedness`; measure-valued initial data are treated in {prf:ref}`cor-chaos-measure-initial-data`. The positive alive-mass estimate supplies the finite-time normalization bound when its hypotheses hold. Stationary solutions and their uniqueness use the resolvent and contraction argument in {prf:ref}`thm-uniqueness-contraction-solution-operator` and {prf:ref}`thm-uniqueness-uniqueness-stationary-solution`.

A stationary alive/dead population model is a conservative law on the extended state space. A killed process has a QSD after conditioning on survival. Their relation must use their defining equations; existence of one is not a density formula for the other. Full-gradient LSI and hypocoercive entropy convergence are established in {doc}`15_kl_convergence` for its identified laws and generators.
:::

:::{prf:theorem} Transfer of particle-coupling estimates
:label: thm-mean-field-limit-informal

Let $X_1,\ldots,X_N$ be interacting particle states coupled to independent
$Y_1,\ldots,Y_N$ with common probability law $\mu$, all with finite second moments. Put
$\varepsilon_N=N^{-1}\sum_i\mathbb E\|X_i-Y_i\|^2$ and
$\mu_N^X=N^{-1}\sum_i\delta_{X_i}$, $\mu_N^Y=N^{-1}\sum_i\delta_{Y_i}$.
Then

$$
\mathbb E W_2(\mu_N^X,\mu)
\leq\sqrt{\varepsilon_N}+\mathbb E W_2(\mu_N^Y,\mu).
$$

For any $L$-Lipschitz observable $\phi$ with finite variance under $\mu$,

$$
\mathbb E|\mu_N^X\phi-\mu\phi|
\leq L\sqrt{\varepsilon_N}+\sqrt{\operatorname{Var}_{\mu}(\phi)/N}.
$$

If the coupling discrepancy satisfies
$\varepsilon_N'(t)\leq C\varepsilon_N(t)+b_N(t)$ almost everywhere, then

$$
\varepsilon_N(t)\leq e^{Ct}\varepsilon_N(0)
+\int_0^t e^{C(t-s)}b_N(s)\,ds.
$$

The analytical coupling and mean-field identification results are developed
in {doc}`09_propagation_chaos`. The empirical Wasserstein term has its own
moment- and dimension-dependent sampling rate.
:::

:::{prf:proof}
The empirical pairing
$N^{-1}\sum_i\delta_{(X_i,Y_i)}$ is an admissible coupling of the empirical
measures. Thus
$W_2^2(\mu_N^X,\mu_N^Y)\leq N^{-1}\sum_i\|X_i-Y_i\|^2$.
Use the triangle inequality and Jensen's inequality to obtain the first bound.
For the observable, split the error through $\mu_N^Y\phi$. Lipschitz
continuity and Cauchy–Schwarz bound the paired error by
$L\sqrt{\varepsilon_N}$. Independence gives
$\mathbb E|\mu_N^Y\phi-\mu\phi|^2=\operatorname{Var}_{\mu}(\phi)/N$.
Cauchy–Schwarz gives its first-moment bound. The final inequality follows by
multiplying the differential inequality by $e^{-Ct}$ and integrating.
:::

:::{div} feynman-prose
The forward equation gives the macroscopic balance of specified particle transitions. Its analysis then proceeds through the proved normalization, continuity, coupling, and stationary-law estimates. Keeping the finite-step kernel, the infinitesimal generator, and the conditioned law distinct makes those estimates fit together without changing the algorithm along the way.
:::
