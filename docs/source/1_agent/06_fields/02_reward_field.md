(sec-the-reward-field-value-forms-and-hodge-geometry)=
# The Reward Field: Value Forms and Hodge Geometry

## TLDR

- Treat reward as a **field / differential form**, not just a scalar: direction matters when moving through state space.
- The critic/value can be treated as a geometric object (a potential solving a Helmholtz/Poisson-style equation on the
  latent manifold) in the conservative diffusion regime.
- Use Hodge-style decomposition to separate conservative (gradient) structure from cyclic/value-curl structure (games,
  non-equilibrium tasks).
- This chapter gives a principled way to detect when the conservative scalar-reward model fails (curl ≠ 0) and what to
  do about it.
- Outputs are implementable diagnostics (value curl, conformal backreaction, mass/value correlation) that tie value to
  geometry.

## Roadmap

1. Reward as a 1-form; scalar reward as the conservative special case.
2. Value/critic as a PDE object and its geometric interpretation.
3. Hodge decomposition + diagnostics for cyclic/non-conservative structure.

{cite}`evans2010pde,sutton2018rl`

:::{div} feynman-prose
Now we come to one useful representation of reward, and I have to tell you something that might seem heretical at first: in this model, reward is not only a number.

In every RL textbook, you see $r_t$---a scalar. The agent does something, the environment gives back a number, and the goal is to maximize the sum of these numbers over time. That scalar description remains valid for ordinary discrete rewards; the 1-form below resolves the additional dependence on direction.

Here's the extra structure: when a reward component is attached to motion through the world, it depends not just on *where* you are, but on *which direction you're moving*. Walk toward the refrigerator, and you might get closer to food (good). Walk away from it, and you don't. Same position, different directional contribution.

This means reward isn't a scalar field (a number at each point). It's a *1-form*---a mathematical object that eats a direction and spits out a number. The reward you get is the inner product of the reward 1-form with your velocity: $r_t = \langle \mathcal{R}, v \rangle$.

Why does this matter? Because it opens up a whole world of structure that a scalar potential cannot represent. The 1-form can have a "curl"---non-zero circulation around closed loops. When that circulation is present, an optimal strategy may involve cycling rather than converging to a fixed point, but that conclusion requires an objective and dynamical assumptions beyond the decomposition itself.

Think of Rock-Paper-Scissors. There is no dominant action in the cyclic game. The preference loop is a useful analogy for non-exact reward structure; identifying a particular RL optimum still requires specifying the game and policy dynamics.
:::

We have defined Observations as **Configuration Constraints** (manifold position, {ref}`sec-the-symplectic-interface-position-momentum-duality`) and Actions as **Momentum Constraints** (tangent vectors, {ref}`sec-the-symplectic-interface-position-momentum-duality`). We now define the third component of the interface: **Reward**.

We rigorously frame Reward not as a scalar signal, but as a **Differential 1-Form** on the latent manifold. This generalization is fundamental: the agent harvests reward by moving through the field, and the reward it collects depends on both position and direction of motion. The standard scalar value function $V(z)$ emerges as the special case where the reward field is **conservative** (curl-free).

(rb-non-conservative-value)=
:::{admonition} Researcher Bridge: Beyond Conservative Value Functions
:class: tip
Standard RL assumes a scalar Value function $V(z)$ exists such that the reward 1-form is exact: $\mathcal{R} = dV$ (equivalently $A=0$ in the decomposition $\mathcal{R}=dV+A$). This implies that the total reward around any closed loop is zero—no cyclic preference structures exist. But many real-world scenarios violate this:
- **Rock-Paper-Scissors**: Cyclic dominance creates non-zero reward loops
- **Exploration-Exploitation Orbits**: Optimal behavior may involve sustained cycling
- **Paradoxical Preferences**: Humans exhibit intransitive preferences

We generalize by treating reward as a **1-form field** $\mathcal{R}$, with scalar value as the special case where $\mathcal{R}$ is exact (curl vanishes and the harmonic component is fixed to zero by boundary conditions). The **Hodge Decomposition** separates the optimizable (gradient) component from the cyclic (solenoidal) component.
:::

(sec-the-reward-1-form)=
## The Reward 1-Form

:::{div} feynman-prose
Let me make the mathematical setup precise. A 1-form is a linear map from tangent vectors to numbers. At each point $z$ on the manifold, you have a 1-form $\mathcal{R}(z)$ that takes any velocity vector $v$ and returns a real number: the instantaneous directional reward rate. This is a model for that component of the signal; a separate scalar reward density can still be supplied as boundary or bulk data for a field equation.

The beautiful thing about 1-forms is that they integrate naturally along paths. If you want to know the directional reward collected along a trajectory, you just integrate: $R_{\text{cumulative}} = \int_\gamma \mathcal{R}$. This is a *line integral*, exactly like the work done by a force field in physics.

Notice the remark: this 1-form component gives zero instantaneous reward to a stationary agent ($v = 0$). A scalar source or terminal payoff is a different object and can represent reward that is not a directional line integral.

This is different from the textbook picture, where you might imagine sitting in a "good state" and accumulating reward by existing. In the 1-form formulation, the directional contribution is a rate along motion; do not confuse it with a scalar source used by the value PDE.
:::

We begin with the most general formulation: reward is a **differential 1-form** on the latent manifold.

:::{prf:definition} The Reward 1-Form
:label: def-reward-1-form

Let $\mathcal{R}$ be a differential 1-form on the latent manifold $(\mathcal{Z}, G)$. The **instantaneous reward rate** received by the agent moving with velocity $v \in T_z\mathcal{Z}$ is:

$$
r_t = \mathcal{R}(z)[v] = \mathcal{R}_i(z) \dot{z}^i.

$$

*Units:* $[\mathcal{R}] = \mathrm{nat}/[\text{length}]$ and $[r_t]=\mathrm{nat}/\text{time}$ for the
continuous-time reward rate. A discrete sample is $r_t^{\text{step}}=r_t\,\Delta t$ and has units nat.

The cumulative reward along a trajectory $\gamma: [0,T] \to \mathcal{Z}$ is the **line integral**:

$$
R_{\text{cumulative}} = \int_\gamma \mathcal{R} = \int_0^T \mathcal{R}_i(\gamma(t)) \dot{\gamma}^i(t) \, dt.

$$

*Remark.* Instantaneous reward depends on both position $z$ and velocity $\dot{z}$. A stationary agent ($\dot{z} = 0$) receives zero instantaneous reward.

:::

:::{prf:definition} The Reward Flux (Boundary Form)
:label: def-the-reward-flux

The environment provides reward via a boundary 1-form $J_r$ on $\partial\Omega$. Let
$\iota:\partial\Omega\hookrightarrow\Omega$ be the inclusion. The boundary condition is the pullback
$\iota^*\mathcal{R} = J_r$, and the cumulative boundary reward along a boundary trajectory
$\gamma_\partial$ is:

$$
\int_{\gamma_\partial} J_r = \text{Cumulative Boundary Reward}.

$$
In the discrete limit, this manifests as samples $r_t^{\text{step}} = J_r(\partial_t)\,\Delta t$ deposited at the boundary
coordinates $(t, z_{\text{boundary}})$.

*Units:* $[J_r] = \mathrm{nat}/[\text{length}]$ and $[r_t^{\text{step}}] = \mathrm{nat}$.

*Relation to 1-form:* For any surface $\Sigma$ with boundary $\partial\Sigma$, Stokes' theorem gives
$\oint_{\partial\Sigma}\mathcal{R}=\int_\Sigma d\mathcal{R}$. For a loop $\gamma=\partial\Sigma$, Stokes' theorem relates its circulation to $d\mathcal R$; a general closed loop need not bound a surface. The pullback $\iota^*\mathcal R$ is tangential boundary data;
it is not by itself a Neumann flux or a source density for a value PDE.

:::

:::{prf:definition} Terminal Boundary (End/Death Flags)
:label: def-terminal-boundary

When termination is $\sigma(Z_t)$-measurable, let $\Gamma_{\text{term}} \subset \mathcal{Z}$ denote
the terminal subset representing end/death flags. Under partial observability use the conditional
killing rate instead of identifying the flag with a latent subset.
Define the stopping time $\tau_{\text{term}} := \inf\{t \ge 0 : z_t \in \Gamma_{\text{term}}\}$ and
kill the process upon hitting $\Gamma_{\text{term}}$. For the conservative value PDE, impose a
Dirichlet condition $V|_{\Gamma_{\text{term}}} = V_{\text{term}}$ (often $0$ or a terminal payoff).
In WFR form, include a killing rate $\kappa_{\text{term}}(z) \ge 0$ or a reaction term $r<0$
concentrated on $\Gamma_{\text{term}}$.

*Terminal vs holographic boundary.* $\Gamma_{\text{term}}$ is a task boundary and is separate from the
holographic boundary of the hyperbolic space.

*Computational cutoff.* For numerics we truncate the hyperbolic disk at $\lvert z\rvert = 1-\varepsilon$,
with $\varepsilon$ tied to the Levin length/resolution. This is a computational boundary for stability,
not a physical terminal set.

:::

(pi-electromagnetism-reward)=
::::{admonition} Physics Isomorphism: Electromagnetism
:class: note

**In Physics:** A charged particle moving through an electromagnetic field experiences a force that depends on both position and velocity. The electric field $\mathbf{E}$ creates conservative (gradient) forces, while the magnetic field $\mathbf{B}$ creates velocity-dependent (curl) forces via the Lorentz force law: $\mathbf{F} = q(\mathbf{E} + \mathbf{v} \times \mathbf{B})$.

**In Implementation:** An agent moving through a reward field experiences:
- **Gradient force:** $-\nabla\Phi$ (climb toward value peaks)
- **Lorentz force:** $\mathcal{F} \cdot \dot{z}$ (orbit around value cycles)

**Correspondence Table:**

| Electromagnetism | Agent (Reward Field) |
|:-----------------|:---------------------|
| 4-potential $A_\mu$ | Reward 1-form $\mathcal{R}$ |
| Electric potential $\phi$ | Scalar potential $\Phi$ |
| Magnetic field $\mathbf{B} = \nabla \times \mathbf{A}$ | Value Curl $\mathcal{F} = d\mathcal{R}$ |
| Lorentz force $\mathbf{v} \times \mathbf{B}$ | Orbiting strategy |
| Cyclotron orbit | Value harvesting cycle |

::::
(sec-hodge-decomposition-of-value)=
## The Hodge Decomposition of Value

:::{div} feynman-prose
Now we come to one of the most useful theorems in differential geometry, applied to the reward landscape: the Hodge--Morrey--Friedrichs decomposition.

The idea is this: after choosing the boundary realization and the function space, a sufficiently regular 1-form can be split into three orthogonal pieces:

1. **Gradient part** ($d\Phi$): This is the exact, path-independent component. Its line integral depends only on the endpoints. In the reward convention used here $V_{\mathrm{rew}}=\Phi$; the corresponding cost-to-go is $V_{\mathrm{cost}}=-\Phi$, so the sign must be handled consistently in the dynamics.

2. **Solenoidal part** ($\delta\Psi$): This is the "swirl" component. It can support circulation and persistent currents, although whether trajectories orbit depends on the chosen dynamics.

3. **Harmonic part** ($\eta$): This is the boundary-conditioned harmonic component. Only the appropriate absolute or relative harmonic subspace is identified with de Rham cohomology and hence with topological cycles. On the truncated disk with the stated relative condition, $H^1$ is trivial, so a nonzero harmonic field would be boundary-driven rather than hole-driven.

Why does this matter? Because each component has different implications for analysis:
- The exact part can be represented by a scalar potential and integrated by endpoint differences.
- The solenoidal part is where circulation and non-equilibrium currents can enter.
- The harmonic part records the selected boundary/cohomological sector.

Standard scalar-value RL assumes that the reward field is exact. Vanishing curl is not enough on a domain with nontrivial periods; one must also remove the harmonic/period contribution. When that assumption fails, a scalar potential alone does not describe the full reward field.
:::

The central theorem of this section decomposes a sufficiently regular reward 1-form into three orthogonal components:
gradient, solenoidal, and harmonic. On a manifold with boundary the boundary condition is part of the statement.

:::{prf:theorem} Hodge Decomposition of the Reward Field
:label: thm-hodge-decomposition

On a compact latent Riemannian manifold $(\mathcal{Z}, G)$, after choosing the absolute or relative
Hodge--Morrey--Friedrichs boundary condition, the Reward 1-form $\mathcal{R}$ decomposes into:

$$
\mathcal{R} = \underbrace{d\Phi}_{\text{Gradient}} + \underbrace{\delta \Psi}_{\text{Solenoidal}} + \underbrace{\eta}_{\text{Harmonic}}

$$
where:
1. **$\Phi \in \Omega^0(\mathcal{Z})$** (Scalar Potential): The conservative/optimizable component. $d\Phi$ is an exact form.
2. **$\Psi \in \Omega^2(\mathcal{Z})$** (Vector Potential): The rotational/cyclic component. $\delta\Psi$ is a coexact form (divergence-free).
3. **$\eta \in \mathcal{H}^1_{\mathrm{bc}}(\mathcal{Z})$** (Harmonic Flux): a boundary-conditioned harmonic
field satisfying $d\eta = 0$ and $\delta\eta = 0$. Only the corresponding absolute/relative harmonic subspace
is identified with de Rham cohomology and hence with topological cycles; on the truncated disk with the standard
relative boundary condition, $H^1$ is trivial.

We use $\Phi$ for the reward-side potential and $V:=-\Phi$ for the cost-to-go convention used
by the control-loop chapter. Thus $d\Phi=-dV$; the conservative case corresponds to $A=0$.
Define the non-exact component $A := \delta\Psi + \eta$, so $\mathcal{R} = d\Phi + A$ and $\mathcal{F} = dA$.

*Units:* $[\Phi] = \mathrm{nat}$, $[\Psi] = \mathrm{nat}$, $[\eta] = \mathrm{nat}/[\text{length}]$.

*Proof sketch.* The Hodge--Morrey--Friedrichs decomposition follows from the orthogonal decomposition of $L^2(\Omega^1)$
into exact, coexact, and harmonic forms (with absolute/relative boundary conditions fixed when
$\partial\mathcal{Z}\neq\varnothing$). The Hodge Laplacian $\Delta_H = d\delta + \delta d$ has kernel
equal to the harmonic forms. The explicit solution uses the Green's operator
$G = (\Delta_H)^{-1}$ on the orthogonal complement of harmonic forms:
$\Phi = \delta G \mathcal{R}$, $\Psi = d G \mathcal{R}$,
$\eta = \mathcal{R} - d\Phi - \delta\Psi$. $\square$

:::

:::{prf:definition} The Value Curl (Vorticity Tensor)
:label: def-value-curl

The **Value Curl** is the exterior derivative of the reward form:

$$
\mathcal{F} := d\mathcal{R} = dA = d\delta\Psi.

$$
In coordinates: $\mathcal{F}_{ij} = \partial_i \mathcal{R}_j - \partial_j \mathcal{R}_i$.

*Units:* $[\mathcal{F}] = \mathrm{nat}/[\text{length}]^2$ (curvature of value).

**Properties:**
1. $\mathcal{F}$ is antisymmetric: $\mathcal{F}_{ij} = -\mathcal{F}_{ji}$
2. $\mathcal{F}$ satisfies the Bianchi identity: $d\mathcal{F} = 0$
3. $\mathcal{F}$ is gauge-invariant: if $\mathcal{R} \to \mathcal{R} + d\chi$, then $\mathcal{F} \to \mathcal{F}$

:::

:::{prf:definition} Conservative Reward Field
:label: def-conservative-reward-field

The reward field $\mathcal{R}$ is **conservative** when it is exact, i.e. when there is a scalar $\Phi$ with
$\mathcal{R}=d\Phi$. Equivalently, $d\mathcal R=0$ and all periods $\oint_\gamma\mathcal R$ vanish for closed
loops. On a simply connected domain with the boundary condition above, vanishing curl is sufficient:

$$
\mathcal{R}=d\Phi.

$$
**Conservative Special Case:** In the reward convention $\mathcal{R}=d\Phi$; the cost-to-go used elsewhere is
$V=-\Phi$. For a loop that bounds a surface, $\gamma=\partial\Sigma$, Stokes' theorem gives:

$$
\oint_\gamma \mathcal{R} = \int_\Sigma d\mathcal{R} = 0.

$$

*Remark.* Standard RL assumes conservative reward fields. The scalar value function $V(s)$ exists precisely because path-independence holds.

:::

:::{prf:proposition} Value Cycle Detection
:label: prop-value-cycle-detection

The Value Curl $\mathcal{F}$ can be estimated from trajectory data. For a closed loop $\gamma$ in latent space:

$$
\oint_{\partial\Sigma} \mathcal{R} = \int_\Sigma \mathcal{F} \, d\Sigma \neq 0
\implies \text{the reward field is not exact.}

$$
**Diagnostic:** Non-zero circulation $\oint_\gamma \mathcal{R}$ indicates non-conservative structure.
Using accumulated TD-error around closed loops is a heuristic that requires approximate loop closure
and a consistent reward estimator.

:::

:::{admonition} Connection to RL #31: Path-Independence as Degenerate Value Curl
:class: note
:name: conn-rl-31
**The General Law (Fragile Agent):**
The Reward 1-form $\mathcal{R}$ decomposes via Hodge theory into gradient, solenoidal, and harmonic components. The **Value Curl** $\mathcal{F} = d\mathcal{R}$ measures non-conservative structure:

$$
\oint_\gamma \mathcal{R} = \int_\Sigma \mathcal{F} \, d\Sigma

$$
Non-zero Value Curl implies optimal strategies may involve sustained orbiting rather than converging to fixed points.

**The Degenerate Limit:**
Assume $\mathcal{F} = 0$ everywhere (curl-free reward field).

**The Special Case (Standard RL):**

$$
V(s) = \mathbb{E}\left[\sum_{t=0}^\infty \gamma^t r_t \mid s_0 = s\right]

$$
This recovers the **scalar Value function**, which exists precisely because rewards are path-independent (conservative). The Value at state $s$ is well-defined regardless of how the agent arrived there.

**What the generalization offers:**
- **Non-transitive games:** Rock-Paper-Scissors has $\mathcal{F} \neq 0$; no scalar $V$ exists
- **Cyclic exploration:** Optimal agents may orbit through value cycles indefinitely
- **Richer equilibria:** NESS (Non-Equilibrium Steady States) with persistent probability currents
:::

(sec-the-bulk-potential-screened-poisson-equation)=
## The Conservative Case: Scalar Potential and Screened Poisson Equation

:::{div} feynman-prose
Now let's focus on the special case that standard RL assumes: the reward field is exact. Vanishing curl is part of that condition, but on a domain with nontrivial periods it is not sufficient by itself. When the curl and all periods vanish, the reward 1-form is the gradient of a scalar function $\Phi$, and the scalar description becomes available.

In this regime, the reward-side scalar $\Phi=V_{\mathrm{rew}}$ satisfies a Screened Poisson (or Helmholtz) equation in the particular conservative diffusion sector stated below. This is the continuum limit of a Bellman generator under those hypotheses, and the geometry determines how the field propagates. If you instead write the equation for the cost-to-go $V_{\mathrm{cost}}=-\Phi$, the source and score signs must be changed together.

The equation looks like this in the reward convention:

$$
-\Delta_G \Phi + \kappa^2 \Phi = \rho_r
$$

Let me parse that for you:
- $\Delta_G$ is the Laplace-Beltrami operator---the generalization of the Laplacian to curved manifolds. It measures how $\Phi$ differs from its local average.
- $\kappa^2$ is the screening coefficient. For the stationary diffusion convention used below,
  $\lambda:=-\ln\gamma/\Delta t$ gives $\kappa^2=\lambda/T_c$ (and hence
  $\kappa=\sqrt{\lambda/T_c}$). This is the coefficient produced by the Bellman generator.
- $\rho_r$ is the reward density---where rewards are being deposited.

What does this equation mean? Value propagates from the declared bulk sources with the Green kernel of the
chosen geometry. In the stationary diffusion convention the characteristic length is
$\ell=1/\kappa=\sqrt{T_c/\lambda}$. For $\gamma=0.99$, $T_c=\Delta t=1$, this is about
$9.97$ in the corresponding latent-length units.

This gives the discount factor a *spatial* meaning in this selected diffusion model, not a universal conversion for every latent geometry. On a curved or bounded domain, the decay law and effective range must be recomputed.
:::

When the Value Curl vanishes ($\mathcal{F} = 0$) and the periods vanish, the reward field is exact and
we recover the scalar value function framework. In this regime, the reward potential is $\Phi(z)$ and
the cost-to-go used by the control loop is $V(z)=-\Phi(z)$. The Bellman equation, in the stationary
diffusion sector, becomes the **Screened Poisson (Helmholtz) Equation**.
In the general case, this PDE governs only the gradient component $d\Phi$ of the reward 1-form; the
solenoidal/harmonic parts appear as circulation and are not captured by a scalar potential.

:::{prf:theorem} Bellman generator and stationary screening
:label: thm-the-hjb-helmholtz-correspondence

For the diffusion and discount already used in
the Bellman diffusion defined here, write
$\mathcal L=b\cdot\nabla+T_c\Delta_G$ and $\gamma_h=e^{-\lambda h}$.
In this theorem $V$ denotes the reward-side score $V_{\mathrm{rew}}=\Phi$;
the control-loop cost critic is $V_{\mathrm{cost}}=-\Phi$ and has source
$\rho_c=-\rho_r$.
The smooth Bellman equation has continuous-time form

$$
\partial_tV+\mathcal LV-\lambda V+r=0.
$$
In its stationary zero-drift sector,
$(-\Delta_G+\lambda/T_c)V=r/T_c$; denote this screening coefficient by
$\kappa_B^2=\lambda/T_c$. The scalar field action used in this chapter
instead defines the wave operator

$$
\Box_g=-|g|^{-1/2}\partial_\mu(|g|^{1/2}g^{\mu\nu}\partial_\nu),
\qquad(\Box_g+\kappa^2)V=\rho_r.
$$
The stationary operators coincide under the coefficient identification
$\kappa^2=\kappa_B^2$ and the same sources and boundary realization.

*Proof.* Generator consistency gives
$\mathbb E[V(Z_h,t+h)]=V+h(\partial_t+\mathcal L)V+o(h)$.
Insert this and $e^{-\lambda h}=1-\lambda h+o(h)$ into
$V=rh+e^{-\lambda h}\mathbb E[V(Z_h,t+h)]$, cancel $V$, and divide by $h$.
The $\partial_t^2V$ Taylor term has coefficient $h/2$ after division and
vanishes. Finite signal speed does not change that coefficient.
For the wave model vary

$$
S[V]=\int\left[-\tfrac12g^{\mu\nu}\partial_\mu V\partial_\nu V
-\tfrac12\kappa^2V^2+\rho_rV\right]\sqrt{|g|}\,dx.
$$
Integration by parts against a compactly supported variation $\eta$ gives
$\delta S=\int\eta[-\Box_gV-\kappa^2V+\rho_r]\sqrt{|g|}\,dx$.
Stationarity proves the field equation. For a fixed product metric
$g=\operatorname{diag}(-c^2,G)$, $\Box_g=c^{-2}\partial_t^2-\Delta_G$.
These are explicit equations for two defined evolutions; equality of their
stationary operators is the comparison established here. $\square$
:::
:::{prf:remark} Dimensional Consistency of the Helmholtz Equation
:label: rem-helmholtz-dimensions

The screened Poisson equation $-\Delta_G V + \kappa^2 V = \rho_r$ requires careful dimensional analysis. The naive expression $\kappa = -\ln\gamma$ appears dimensionless, which would be inconsistent with $[\Delta_G] = [\text{length}]^{-2}$.

The stationary Bellman generator fixes the coefficient by
$\kappa^2=\lambda/T_c$. If a separate propagation speed or diffusion coefficient is introduced,
it must be included in the generator before converting to a spatial coefficient; it is not an
independent identification. In normalized units $T_c=\Delta t=1$, this gives
$\kappa=\sqrt{-\ln\gamma}$, not $-\ln\gamma$.

If a separate physical propagation speed is used, its coefficient must be derived from the
corresponding dimensional generator. It cannot be identified with $\lambda/c_{\text{info}}$ without
adding that model explicitly. Under the normalized stationary convention,
$\ell_{\text{screen}}=1/\kappa$ is not determined by $\gamma$ alone.

:::

(pi-yukawa-potential)=
::::{admonition} Physics Isomorphism: Yukawa Potential
:class: note

**In Physics:** The Yukawa (screened Coulomb) potential satisfies $(-\nabla^2 + m^2)\phi = \rho$ where $m$ is the mediating boson mass. The screening length $\ell = 1/m$ determines the range of the force {cite}`yukawa1935interaction`.

**In Implementation:** The value function satisfies $(-\Delta_G + \kappa^2)V = \rho_r$ where (in natural units with $\Delta t = c_{\text{info}} = 1$):

$$
\kappa = \sqrt{-\ln\gamma/T_c}, \quad \ell_\gamma = 1/\kappa

$$
**Correspondence Table:**

| Physics (Yukawa) | Agent (Bellman-Helmholtz) |
|:-----------------|:--------------------------|
| Scalar field $\phi$ | Value function $V(z)$ |
| Mass $m$ | Screening coefficient $\kappa=\sqrt{\lambda/T_c}$ (normalized units: $\sqrt{-\ln\gamma}$) |
| Screening length $1/m$ | Reward horizon $\ell_\gamma = 1/\kappa$ |
| Charge density $\rho$ | Reward density $\rho_r$ |
| Laplacian $\nabla^2$ | Laplace-Beltrami $\Delta_G$ |

**Loss Function:** PINN regularizer enforcing $\|(-\Delta_G + \kappa^2)V - \rho_r\|^2$.
::::

:::{admonition} Connection to RL #4: Bellman Equation as Degenerate Helmholtz PDE
:class: note
:name: conn-rl-4
**The General Law (Fragile Agent):**
The Value Function $V(z)$ satisfies the **Screened Poisson (Helmholtz) Equation** on $(\mathcal{Z}, G)$:

$$
(-\Delta_G + \kappa^2) V(z) = \rho_r(z)

$$
where $\Delta_G$ is the Laplace-Beltrami operator on the Riemannian manifold. In the stationary diffusion convention used
here, $\lambda=-\ln\gamma/\Delta t$ and $\kappa^2=\lambda/T_c$ (see Remark
{prf:ref}`rem-helmholtz-dimensions`). A $c_{\text{info}}$-based propagation length belongs to a separate model and is
not substituted into this generator identity.

**The Degenerate Limit:**
Discretize space on a lattice. Replace $\Delta_G$ with the graph Laplacian $\mathcal{L}_{\text{graph}}$.

**The Special Case (Standard RL):**
The Green's function solution on a discrete graph is the **Neumann series** expansion:

$$
V(s) = \sum_{t=0}^\infty \gamma^t \mathbb{E}[r_t | s_0 = s] = (I - \gamma P)^{-1} r

$$
This recovers the **Bellman equation** $V = r + \gamma P V$.

**Result:** Under the stationary diffusion convention, the screening coefficient $\kappa^2=\lambda/T_c$ encodes
the discount factor $\gamma$ (in normalized units $\Delta t=T_c=1$; see Remark {prf:ref}`rem-helmholtz-dimensions`).
The lattice and learned-manifold descriptions are useful mathematical analogies; the sampled critic below remains a
TD proxy unless a spatial residual and boundary-value solver are supplied.

**What the generalization offers:**
- Geometric propagation: rewards propagate as sources in a scalar field, respecting manifold curvature
- Conformal coupling: high-value-curvature regions modulate the metric ({ref}`sec-geometric-back-reaction-the-conformal-coupling`)
- Continuous limit: natural extension to continuous state spaces without discretization artifacts
- Conditional interpretation: under this stationary diffusion model, $\gamma$ also fixes a spatial screening scale; a
  propagation-based interpretation requires its own speed and unit matching.
:::

:::{prf:proposition} Green's Function Interpretation
:label: prop-green-s-function-interpretation

The Critic computes the **Green's function** of the screened Laplacian on the latent geometry:

$$
V(z) = \int_{\Omega} G_\kappa(z, z') \rho_r(z') \, d\mu_G(z') + \mathcal{B}_{\partial\Omega}[G_\kappa, V],

$$
where $G_\kappa(z, z')$ is the Green's function satisfying
$(-\Delta_G + \kappa^2) G_\kappa(z, \cdot) = \delta_z$, and $\mathcal{B}_{\partial\Omega}$ encodes the
chosen boundary condition. A boundary source density $\sigma_r$ must be specified independently as a
single-layer/Neumann datum; it is not the pullback $\iota^*\mathcal R$. Under that separate choice:

$$
V(z) = \int_{\partial\Omega} G_\kappa(z, z') \sigma_r(z') \, d\Sigma(z').

$$

*Remark.* The value at $z$ is a weighted integral of bulk sources plus boundary flux, with weights
given by the Green's function. This is a superposition principle: the Helmholtz equation is linear.

:::

(pi-green-function)=
::::{admonition} Physics Isomorphism: Green's Function
:class: note

**In Physics:** The Green's function $G(x, x')$ is the fundamental solution satisfying
$\mathcal{L}G(x, \cdot) = \delta(x - \cdot)$ for a linear operator $\mathcal{L}$. In electrostatics,
$G$ is the potential at $x$ due to a unit charge at $x'$. For the screened Laplacian,
$G_\kappa(r)$ decays as $r^{-(d-1)/2} e^{-\kappa r}$ at large $r$ (Bessel $K$ form)
{cite}`jackson1999classical`.

**In Implementation:** The Critic computes the Green's function of the screened Laplacian
(Proposition {prf:ref}`prop-green-s-function-interpretation`):

$$
V(z) = \int_{\Omega} G_\kappa(z, z') \rho_r(z') \, d\mu_G(z') + \mathcal{B}_{\partial\Omega}[G_\kappa, V]

$$
where $(-\Delta_G + \kappa^2) G_\kappa(z, \cdot) = \delta_z$.

**Correspondence Table:**
| Electrostatics | Agent (Critic) |
|:---------------|:---------------|
| Green's function $G(x, x')$ | Value kernel $G_\kappa(z, z')$ |
| Charge density $\rho$ | Reward density $\rho_r$ (bulk) / $\sigma_r$ (boundary) |
| Electrostatic potential $\phi$ | Value function $V$ |
| Screening length $1/m$ | Reward horizon $\ell_\gamma = 1/\kappa$ |
| Superposition principle | Linearity of Helmholtz equation |

**Loss Function:** TD-error $\|(-\Delta_G + \kappa^2)V - \rho_r\|^2$ trains the critic as an implicit
Green's function solver (for the conservative component).
::::

:::{prf:remark} Green's Function Decay Scope
:label: prop-green-s-function-decay

For the Euclidean screened operator (and for geometries with the corresponding asymptotic
analysis), the Green's function has the familiar large-distance form:

$$
G_\kappa(z, z') \sim \frac{1}{d_G(z, z')^{(d-1)/2}} \exp\left(-\kappa \cdot d_G(z, z')\right),

$$
where $d_G$ is the geodesic distance and $d$ is the dimension.

:::
:::{prf:corollary} Discount as Screening Length
:label: cor-discount-as-screening-length

The discount factor $\gamma$ determines a characteristic **screening length**:

$$
\ell_{\text{screen}} = \frac{1}{\kappa} = \sqrt{\frac{T_c}{\lambda}}.

$$
where $\lambda := -\ln\gamma / \Delta t$.
For $\gamma=0.99$, $T_c=\Delta t=1$: $\ell_{\text{screen}}\approx 9.97$ in the normalized latent-length units.

*Interpretation:* Under the displayed Green-kernel hypotheses, rewards at geodesic distance
$>\ell_{\text{screen}}$ are suppressed. On a hyperbolic or bounded domain, the decay rate and
boundary terms must be recomputed rather than inferred from the flat-space asymptotic.

*Note:* Numerical values below assume the normalized stationary convention ($T_c=\Delta t=1$).

**Table 24.2.5 (Discount-Screening Correspondence).**

| Discount $\gamma$ | Screening Mass $\kappa$ | Screening Length $\ell$ | Interpretation                    |
|-------------------|-------------------------|-------------------------|-----------------------------------|
| $\gamma \to 1$    | $\kappa \to 0$          | $\ell \to \infty$       | Infinite horizon (massless field) |
| $\gamma = 0.99$   | $\kappa \approx 0.100$   | $\ell \approx 9.97$      | Standard RL (normalized) |
| $\gamma = 0.9$    | $\kappa \approx 0.325$    | $\ell \approx 3.08$       | Short horizon (normalized)                     |
| $\gamma \to 0$    | $\kappa \to \infty$     | $\ell \to 0$            | Myopic (infinitely massive)       |

**Cross-references:** {ref}`sec-the-hjb-correspondence` (HJB Equation), Theorem {prf:ref}`thm-capacity-constrained-metric-law`.

:::

::::{admonition} Connection to RL #30: Temporal Horizon as Degenerate Screening Length
:class: note
:name: conn-rl-30
**The General Law (Fragile Agent):**
The discount factor $\gamma$ defines a **Screening Length** with geometric meaning:

$$
\kappa^2 = \lambda/T_c, \quad \ell_{\text{screen}} = \frac{1}{\kappa}

$$
Value correlations decay exponentially with **geodesic distance**:

$$
G_\kappa(z, z') \sim \frac{1}{d_G(z,z')^{(d-1)/2}} \exp\!\left(-\kappa \cdot d_G(z, z')\right)

$$
The Critic computes the Green's function of the screened Laplacian $(-\Delta_G + \kappa^2)^{-1}$.

**The Degenerate Limit:**
Interpret $\gamma$ purely temporally. Ignore spatial/geometric structure of the latent space.

**The Special Case (Standard RL):**

$$
V(s) = \sum_{t=0}^\infty \gamma^t r_t

$$
This recovers the standard **temporal horizon interpretation** where $\gamma$ controls how far into the future rewards are considered.

**What the generalization offers:**
- **Spatial credit assignment:** Rewards propagate via geodesics in latent space, not just time steps
- **Physical units:** $\kappa$ has units of inverse length; $\ell_{\text{screen}}$ is a correlation length
- **Green's function decay:** The Critic is a PDE solver; value correlations decay geometrically
- **Unified view:** Temporal and spatial horizons are the same phenomenon in different coordinates
::::

(sec-thermodynamic-interpretation-energy-vs-probability)=
## Thermodynamic Interpretation: Energy vs Probability

:::{div} feynman-prose
Now we have to deal with an old confusion: what *is* the value function? Is it an energy? A probability? A utility?

Here the formal convention keeps the Hodge scalar $\Phi$ on the reward side. Write
$V_{\mathrm{rew}}=\Phi$ and $V_{\mathrm{cost}}=-\Phi$ for the corresponding reward score and
cost-to-go. The free-energy score is $F:=V_{\mathrm{cost}}$; it balances energetic cost against
entropic disorder.

For the agent, the analog is:

$$
F(z):=V_{\mathrm{cost}}(z)=E(z) - T_c S(z), \qquad \Phi(z)=-F(z).
$$

Here $E(z)$ is the task cost (low is good), $S(z)$ is the exploration entropy (high means lots of options), and $T_c$ is the cognitive temperature (how much the agent values exploration). The sign convention matters: low $F$ means low free energy, while high $\Phi=V_{\mathrm{rew}}$ means high reward.

Under the invariant-measure and detailed-balance hypotheses stated below, entropy-regularized RL has a free-energy
interpretation. The Boltzmann distribution $P(z) \propto \exp(\Phi(z)/T_c)=\exp(-F(z)/T_c)$ then describes the corresponding equilibrium;
without those hypotheses it is a modeling convention rather than a consequence of the critic update.

Under the stated equilibrium hypotheses, high temperature ($T_c$ large) broadens the law and low temperature concentrates it near high-$\Phi$ (low-$F$) regions. Without those hypotheses, this is an intended modeling picture rather than a conclusion of the critic or WFR closure.
:::

In this thermodynamic subsection the Boltzmann score is the reward-side quantity
$V_{\mathrm{rew}}:=\Phi$. The control-loop critic uses the cost convention
$V_{\mathrm{cost}}:=-V_{\mathrm{rew}}=-\Phi$ when it minimizes cost-to-go. We write the free-energy score as
$F:=V_{\mathrm{cost}}=E-T_cS$. The canonical density below is therefore a conditional reward-side modeling convention;
it must not be read as changing the control-loop sign convention.

We explicitly resolve the ambiguity between "Energy" and "Probability" in the value function interpretation. The Hodge
potential $\Phi(z)$ is the reward-side score, while its negative $F(z)=-\Phi(z)$ is the Gibbs free energy in the
conservative case.

:::{prf:axiom} Boltzmann-Value Modeling Convention
:label: ax-the-boltzmann-value-law

Let $F(z):=-\Phi(z)$ be the cost/free-energy scalar associated with the Hodge potential. The thermodynamic modeling
convention is

$$
F(z) = E(z) - T_c S(z),

$$
where:
- $E(z)$ is the **task risk/cost** at state $z$
- $S(z)$ is the **exploration entropy** (measure of uncertainty/optionality)
- $T_c$ is the **cognitive temperature** ({prf:ref}`def-cognitive-temperature`, {ref}`sec-hyperbolic-volume-and-entropic-drift`)

*Units:* $[F]=[\Phi]=[E]=[T_c S]=\mathrm{nat}$.

**Conservative Case ($\mathcal{F} = 0$):** When the Value Curl vanishes and periods vanish, $\mathcal{R}=d\Phi$.
The cost convention used by the control loop is $V(z)=-\Phi(z)$.

**Non-Conservative Case ($\mathcal{F} \neq 0$):** The scalar potential $\Phi$ captures only the optimizable component of the reward field. The solenoidal component $\delta\Psi$ creates additional cyclic dynamics.

:::
:::{prf:definition} Canonical Ensemble {cite}`sutton2018rl`
:label: def-canonical-ensemble

When an invariant measure exists and the drift and boundary conditions satisfy detailed balance,
this potential induces a probability measure on the manifold via the **Canonical Ensemble**:

$$
P_{\text{stationary}}(z) = \frac{1}{Z} \exp\left(\frac{V_{\mathrm{rew}}(z)}{T_c}\right)
 = \frac{1}{Z} \exp\left(\frac{\Phi(z)}{T_c}\right),

$$
where $Z = \int_{\mathcal{Z}} \exp(V_{\mathrm{rew}}(z)/T_c) \, d\mu_G(z)$ is the partition function.

*Sign Convention:* $V_{\mathrm{rew}}=\Phi$ is the reward-like quantity (higher is better) and
$F=V_{\mathrm{cost}}=-\Phi$ is the cost/free-energy potential. The displayed ensemble therefore uses
$+V_{\mathrm{rew}}/T_c=\Phi/T_c=-F/T_c$. This is a conditional modeling convention,
not a consequence of the WFR reaction closure alone.

:::

(pi-canonical-ensemble)=
::::{admonition} Physics Isomorphism: Canonical Ensemble
:class: note

**In Physics:** The canonical ensemble describes a system in thermal equilibrium with a heat bath at temperature $T$. The probability of microstate $i$ is $P_i = Z^{-1}\exp(-E_i/k_B T)$ where $Z = \sum_i \exp(-E_i/k_B T)$ is the partition function {cite}`landau1980statistical`.

**In Implementation:** The stationary policy distribution (Definition {prf:ref}`def-canonical-ensemble`):

$$
P_{\text{stationary}}(z) = \frac{1}{Z} \exp\left(\frac{V_{\mathrm{rew}}(z)}{T_c}\right)

$$
where $Z = \int_{\mathcal{Z}} \exp(V_{\mathrm{rew}}(z)/T_c) \, d\mu_G(z)$ is the partition function.

**Correspondence Table:**
| Statistical Mechanics | Agent (MaxEnt RL) |
|:----------------------|:------------------|
| Energy $E$ | Negative reward $-r$ |
| Temperature $k_B T$ | Cognitive temperature $T_c$ |
| Partition function $Z$ | Soft value normalization |
| Free energy $F = -k_BT\log Z$ | Soft value function |
| Boltzmann distribution | MaxEnt optimal policy $\pi^* \propto \exp(Q/T_c)$ |
| Entropy $S = -k_B\sum p\log p$ | Policy entropy $H(\pi)$ |

**Scope:** Under the standard finite-action, fixed-temperature entropy-regularized control
hypotheses, the Boltzmann policy is the unique pointwise maximizer. The state-space ensemble
requires the additional invariant-measure and boundary hypotheses stated above.
::::

:::{prf:definition} WFR Reaction Closure: Value Creates Mass
:label: thm-wfr-consistency-value-creates-mass

In the WFR dynamics ({prf:ref}`def-the-wfr-action`, {ref}`sec-wasserstein-fisher-rao-geometry-unified-transport-on-hybrid-state-spaces`), the reaction rate $r(z)$ in the unbalanced continuity equation is determined by the value function:

$$
r(z) = \frac{1}{s_r} \left( V(z) - \bar{V} \right),

$$
where $\bar{V} = \mathbb{E}_\rho[V]$ is the mean value and $s_r$ is the reaction time scale (computation time).

*Consequence:* The total mass satisfies:

$$
\frac{d}{ds}\int_{\mathcal Z}\rho\,d\mu_G
 = \int_{\mathcal Z}\rho(z,s)r(z,s)\,d\mu_G
 = \frac{1}{s_r}\int_{\mathcal Z}\rho\,(V-\bar V)\,d\mu_G.

$$

Under this closure, the local reaction contribution is positive where $V>\bar V$ and negative
where $V<\bar V$; transport and boundary flux can change the total mass separately.

*Scope.* This is a chosen reaction closure. It becomes a consequence only after a WFR variational
problem, endpoint constraints, and a target density have been specified; the WFR action alone does
not determine $r$ from $V$.

:::
:::{prf:remark} Conditional Conservative Equilibrium
:label: cor-equilibrium-distribution

**Conditional statement:** If the transport drift, reaction law, boundary conditions, and invariant
measure are chosen to satisfy detailed balance with the reward convention, then the stationary
density is the Boltzmann form:

$$
\rho_\infty(z) \propto \exp\left(-\frac{V_{\mathrm{cost}}(z)}{T_c}\right)
 = \exp\left(\frac{\Phi(z)}{T_c}\right),

$$
which is exactly the canonical ensemble (Definition {prf:ref}`def-canonical-ensemble`).

Without those detailed-balance hypotheses, the displayed reaction closure does not by itself imply
stationarity or zero current.

:::

:::{prf:proposition} Conditional Non-Equilibrium Steady State (NESS)
:label: thm-ness-existence

If a stationary solution exists and detailed balance is broken (for example by a nonzero curl
term or boundary drive), it is a **Non-Equilibrium Steady State** satisfying:

1. **Stationarity:** $\partial_s \rho_\infty = 0$
2. **Persistent Current:** The probability current $J = \rho v - D\nabla\rho$ is non-zero and divergence-free: $\nabla \cdot J = 0$ but $J \neq 0$
3. **Entropy Production:** The system continually produces entropy at rate:

$$
\dot{S}_i = \int_{\mathcal{Z}} \frac{\|J\|_G^2}{\rho D} \, d\mu_G > 0

$$

*Remark.* The probability density $\rho_\infty$ is time-independent, but individual trajectories circulate indefinitely. This distinguishes NESS from true equilibrium (where $J = 0$).

:::

:::{prf:proposition} NESS Decomposition
:label: prop-ness-decomposition

The probability current in a NESS decomposes into:

$$
J = J_{\text{gradient}} + J_{\text{cyclic}}

$$
where:
- $J_{\text{gradient}} = -D\rho\nabla\ln\rho + \rho\nabla\Phi$ derives from the scalar potential
- $J_{\text{cyclic}} = \rho \cdot v_{\text{curl}}$ derives from the solenoidal component

At stationarity, $\nabla \cdot J = 0$, but only $J_{\text{gradient}} = 0$ at true equilibrium. NESS has $J_{\text{cyclic}} \neq 0$.

:::

**Table 24.4.5 (Thermodynamic-RL Dictionary).**

| Thermodynamics         | RL / Control                               | Mathematical Object |
|------------------------|--------------------------------------------|---------------------|
| Energy $E$             | Negative reward $-r$                       | Instantaneous cost  |
| Free Energy $F$        | Cost potential $V_{\mathrm{cost}}=-\Phi$   | Gibbs free energy   |
| Temperature $T$        | Cognitive temperature $T_c$                | Entropy weighting   |
| Entropy $S$            | Policy entropy $H(\pi)$                    | Exploration measure |
| Partition function $Z$ | Soft value $\log \sum_a \exp(Q/T_c)$       | Normalization       |
| Boltzmann distribution | MaxEnt policy $\pi^* \propto \exp(Q/T_c)$ | Conservative solution |
| **Probability current $J$** | **Value harvesting flow** | **NESS circulation** |
| **Entropy production $\dot{S}_i$** | **Irreversibility diagnostic** | **Nonequilibrium dissipation** |

**Cross-references:** {ref}`sec-the-wfr-metric` (WFR dynamics), {ref}`sec-the-belief-evolution-cycle-perception-dreaming-action` (Thermodynamic Cycle), {ref}`sec-the-equivalence-theorem` (MaxEnt control), Theorem {prf:ref}`thm-hodge-decomposition` (Hodge Decomposition).

:::{prf:remark} Varentropy as a Temperature-Sensitivity Diagnostic
:label: cor-varentropy-stability

Let $\mathcal{I}(a|z) = -\ln \pi(a|z)$ be the surprisal of an action. Define the **Policy Varentropy** $V_H(z)$ as the variance of the surprisal under the Boltzmann policy:

$$
V_H(z) := \mathrm{Var}_{a \sim \pi}[\mathcal{I}(a|z)] = \mathbb{E}_{\pi}\left[ \left( \ln \pi(a|z) + H(\pi) \right)^2 \right].

$$
*Units:* $\mathrm{nat}^2$.

Under the Boltzmann-Value Law (Axiom {prf:ref}`ax-the-boltzmann-value-law`), the Varentropy equals the **Heat Capacity** $C_v$ of the decision state:

$$
V_H(z) = \beta_{\text{ent}}^2 \mathrm{Var}_\pi[Q] = C_v,

$$
where $\beta_{\text{ent}} = 1/T_c$ is the inverse cognitive temperature. Equivalently:

$$
V_H(z) = T_c \frac{\partial H(\pi)}{\partial T_c}.

$$
**Operational Consequence:**
1. **Thermal Stability:** $V_H$ measures the sensitivity of the agent's exploration strategy to changes in the cognitive temperature $T_c$.
2. A spike can flag sharp temperature sensitivity, but it is not by itself a phase-transition
   theorem.
3. Any annealing-rate condition requires a separate mixing-time or spectral-gap estimate; none is
   implied by the varentropy identity.

:::
(sec-geometric-back-reaction-the-conformal-coupling)=
## Geometric Back-Reaction: The Conformal Coupling

:::{div} feynman-prose
Now I want to tell you about a modeling choice that closes the loop: the geometry affects the value field (through the Laplace-Beltrami operator), and *the value field can be fed back into the geometry*.

This is analogous to back-reaction. In general relativity, matter curves spacetime, and curved spacetime tells matter how to move. Here, a value-dependent conformal rule changes the latent metric, and the changed metric enters the agent's dynamics. The analogy does not make the rule an Einstein equation.

Specifically, in regions where the value function has high metric Hessian norm---sharp ridges, steep valleys, critical decision points---the chosen model rescales the metric. The conformal factor $\Omega(z) = 1 + \alpha_{\text{conf}} \|\nabla^2_G V\|_{\text{op}}$ is computed in the stated metric convention.

What does this mean practically? A larger conformal factor changes distances and can change the speed of a chosen integrator near important decisions. Whether the agent actually slows down, and by how much, depends on the equations, discretization, and boundary conditions. The coupling is a tunable modeling rule, not an automatic theorem of caution.

Think about it: in a region where the value landscape is flat, $\Omega$ is near one. Near a high-curvature region, the metric may stretch distances and increase the modeled effort of motion. The resulting caution is a hypothesis to measure, not something the conformal formula proves on its own.
:::

Does the Reward field change the Geometry? **Yes.** From Theorem {prf:ref}`thm-capacity-constrained-metric-law`, the curvature is driven by the Risk Tensor. Both the scalar potential $\Phi$ and the Value Curl $\mathcal{F}$ contribute to risk, and therefore modify the metric.

:::{prf:definition} Value-Metric Conformal Coupling
:label: def-value-metric-conformal-coupling

We model the effect of Value on the Metric $G$ as a **Conformal Transformation**:

$$
\tilde{G}_{ij}(z) = \Omega^2(z) \cdot G_{ij}(z),

$$
where the conformal factor $\Omega(z)$ depends on the **Hessian of the Value**:

$$
\Omega(z) = 1 + \alpha_{\text{conf}} \cdot \|\nabla^2_G V(z)\|_{\text{op}},

$$
with $\alpha_{\text{conf}} \ge 0$ the conformal coupling strength and $\|\cdot\|_{\text{op}}$ the operator norm.

Units: $[\Omega] = 1$ (dimensionless), $[\alpha_{\text{conf}}] = \text{length}^2/\mathrm{nat}$.

:::

(pi-conformal-coupling)=
::::{admonition} Physics Isomorphism: Conformal Transformation
:class: note

**In Physics:** A conformal transformation rescales the metric by a position-dependent factor: $\tilde{g}_{\mu\nu} = \Omega^2(x) g_{\mu\nu}$. In scalar field theory, conformal coupling $\xi R\phi^2$ couples the field to spacetime curvature. Weyl transformations preserve angles but not distances {cite}`wald1984general`.

**In Implementation:** The value-metric conformal coupling (Definition {prf:ref}`def-value-metric-conformal-coupling`):

$$
\tilde{G}_{ij}(z) = \Omega(z)^2 \cdot G_{ij}(z), \quad \Omega(z) = 1 + \alpha_{\text{conf}} \|\nabla^2_G V(z)\|_{\text{op}}

$$
**Correspondence Table:**
| Conformal Field Theory | Agent (Value-Metric Coupling) |
|:-----------------------|:------------------------------|
| Conformal factor $\Omega^2$ | $\left(1 + \alpha\|\nabla^2 V\|\right)^2$ |
| Weyl rescaling | Value-dependent metric inflation |
| Conformal anomaly | Curvature-dependent deliberation cost |
| Preserved angles | Preserved local policy directions |
| Dilated distances | Increased caution in high-curvature regions |

**Effect:** High-curvature value regions acquire increased effective mass, automatically slowing the agent near critical decisions.
::::

:::{prf:proposition} Risk-Curvature Mechanism
:label: prop-risk-curvature-mechanism

The conformal factor encodes the local "importance" of the value landscape:

| Value Landscape           | $\lVert\nabla^2 V\rVert$ | $\Omega$    | Effect                           |
|---------------------------|--------------------------|-------------|----------------------------------|
| **Flat** (low importance) | $\approx 0$              | $\approx 1$ | Default hyperbolic bulk geometry |
| **Curved** (ridge/valley) | $\gg 0$                  | $\gg 1$     | Distances expand, mass increases |
| **Saddle** (transition)   | moderate                 | $> 1$       | Intermediate slowdown            |

:::
:::{prf:corollary} Inertia at Critical Regions
:label: cor-inertia-at-critical-regions

Near sharp ridges or valleys of $V$ (where $\|\nabla^2 V\|$ is large), the conformal factor causes:

1. **Inertia Increase:** The effective mass $\tilde{G}(z) = \Omega^2(z) G(z)$ increases, so the agent slows down near critical decision boundaries ({ref}`sec-the-coupled-jump-diffusion-sde` mass scaling).

2. **Resolution Increase:** The capacity-constrained metric allocates more volume to high-curvature regions (Theorem {prf:ref}`thm-capacity-constrained-metric-law`), allowing higher-fidelity representation of value gradients.

3. **Stability:** The agent cannot "rush through" regions of high value curvature—it is forced to carefully navigate decision boundaries.

*Remark (Physical analogy).* The conformal scaling of effective velocity is mathematically analogous to gravitational time dilation in general relativity, where proper time dilates in regions of high gravitational potential.

:::
:::{prf:remark} Conformal Laplacian Transformation
:label: prop-conformal-laplacian-transformation

Under the conformal transformation $G \to \tilde{G} = \Omega^2 G$, the Laplace-Beltrami operator acting on a scalar function $f$ transforms as:

$$
\Delta_{\tilde{G}} f = \Omega^{-2} \left( \Delta_G f + (d-2) \frac{G^{ij} \partial_i \Omega}{\Omega} \partial_j f \right),

$$
where $d$ is the dimension. For the Value function $V$ itself (which determines $\Omega$), this creates a **nonlinear coupling**. The screened Poisson equation (Theorem {prf:ref}`thm-the-hjb-helmholtz-correspondence`) in the conformally modified metric becomes:

$$
-\Delta_{\tilde{G}} V + \tilde{\kappa}^2 V = \tilde{\rho}_r,

$$
with effective screening mass $\tilde{\kappa}^2 = \Omega^{-2} \kappa^2$.

*Remark (Self-Consistency).* Since $\Omega$ depends on $\nabla^2 V$, the equation becomes nonlinear: the geometry adapts to the value landscape which in turn affects the geometry. In practice, we solve this iteratively or treat $\Omega$ as slowly-varying.

*Interpretation:* The displayed coefficient is a coordinate rewriting of the conformally transformed
operator. A self-focusing or longer-range-correlation conclusion requires solving the transformed
boundary-value problem; it does not follow from the rescaling alone.

**Cross-references:** Theorem {prf:ref}`thm-capacity-constrained-metric-law`, {ref}`sec-the-stochastic-action-principle` (Mass=Metric), Proposition {prf:ref}`prop-mass-scaling-near-boundary`.

:::

::::{admonition} Connection to RL #27: Auxiliary Tasks as Degenerate Conformal Back-Reaction
:class: note
:name: conn-rl-27
**The General Law (Fragile Agent):**
The value function modulates the metric via **Conformal Coupling**:

$$
\tilde{G}_{ij} = \Omega^2(z)\, G_{ij}, \quad \Omega(z) = 1 + \alpha_{\text{conf}} \|\nabla^2_G V(z)\|_{\text{op}}

$$
High-curvature value regions acquire inertia, slowing agent dynamics near critical decision boundaries.

**The Degenerate Limit:**
Remove metric coupling ($\alpha_{\text{conf}} \to 0$). Treat auxiliary losses as independent add-ons.

**The Special Case (Standard RL):**

$$
\mathcal{L} = \mathcal{L}_{\text{RL}} + \sum_k \lambda_k \mathcal{L}_{\text{aux},k}

$$
This recovers **Auxiliary Tasks** (reward prediction, inverse dynamics, world models) {cite}`jaderberg2017unreal`.

**What the generalization offers:**
- **Explicit geometry feedback:** Value landscape modifies the metric, not just the loss
- **Inertia in risky regions:** High value curvature physically slows exploration
- **Self-consistency:** $\Omega$ depends on $V$, creating a nonlinear feedback loop
- **Resolution allocation:** Capacity-constrained metric assigns more volume to high-curvature regions
::::

(sec-implementation-the-holographiccritic-module)=
## Implementation: The HolographicCritic Module

:::{div} feynman-prose
Time to build the Critic. And I want you to think about it differently than you might be used to.

In standard RL, the critic is a "value predictor"---a function approximator that learns to output $V(s)$ for each state $s$. You train it with TD-learning, bootstrap from targets, and try to minimize prediction error.

Here, the critic is intended to play the role of a *field solver*. In the conservative continuum model it represents a solution of the Screened Poisson equation, with a declared reward density as source. The reference implementation below is a sampled consistency proxy, so the role is conditional.

This isn't just a reframing. It changes how you think about training. A TD error can be used as a residual-like diagnostic for the Bellman generator, but zero TD error alone does not establish a PDE solution unless the generator, source, boundary data, and sampling scheme match.

The conformal coupling adds another layer: the critic can compute a metric Hessian proxy and feed it into the selected metric rule. High-curvature regions may then be flagged or rescaled; the dynamics need separate validation before calling that a slowdown or increased caution.

Notice how the implementation computes a TD consistency proxy and a geometric regularization term. These can encourage a well-behaved approximation, but they do not by themselves solve the full boundary-value PDE or guarantee the stated smoothness.
:::

We update the architecture to include the Critic as the third pillar of the Holographic Interface. In the conservative
continuum model it can be interpreted as a field solver; the reference implementation below computes a sampled TD
consistency proxy and a metric-weighted smoothness penalty, so it does not by itself solve a boundary-value PDE.

```python
import torch
import torch.nn as nn
from torch import Tensor
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class CriticConfig:
    """Configuration for the HolographicCritic ({ref}`sec-the-reward-field-value-forms-and-hodge-geometry`)."""
    latent_dim: int = 32          # Dimension of latent space Z
    hidden_dim: int = 256         # Hidden layer dimension
    gamma: float = 0.99           # Discount factor
    alpha_conf: float = 0.1       # Conformal coupling strength (Definition 24.4.1)
    grad_reg_weight: float = 0.01 # Geometric gradient regularization
    T_c: float = 1.0          # Cognitive temperature in the diffusion convention
    delta_t: float = 1.0      # Interaction-time step

    @property
    def screening_mass(self) -> float:
        """Screening mass for the stationary Bellman coefficient kappa=sqrt(lambda/T_c)."""
        return torch.sqrt(-torch.log(torch.tensor(self.gamma)) / self.T_c / self.delta_t).item()


class HolographicCritic(nn.Module):
    """
    {ref}`sec-the-reward-field-value-forms-and-hodge-geometry`: The Reward Encoder / Field Solver.

    Maps Boundary Charges (rewards r) to Bulk Potential (value V).
    Provides a TD consistency proxy for the screened Bellman equation on the latent manifold.

    The Critic does not "predict" reward—it PROPAGATES boundary conditions
    into the bulk to compute the resulting potential field.
    """

    def __init__(self, config: CriticConfig):
        super().__init__()
        self.config = config
        self.kappa = config.screening_mass  # Screening mass (Corollary 24.2.4)
        self.alpha_conf = config.alpha_conf

        # Geometry-aware network for V(z)
        # Note: SiLU activation preserves smoothness needed for Hessian computation
        self.net = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, 1)  # Scalar potential V(z)
        )

        # Initialize to near-zero to start with flat potential
        self._init_weights()

    def _init_weights(self):
        """Initialize to small weights for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z: Tensor) -> Tensor:
        """
        Compute V(z): the scalar potential at bulk location z.

        Args:
            z: Latent positions [B, D]

        Returns:
            V: Scalar potential [B, 1]
        """
        return self.net(z)

    def compute_helmholtz_loss(
        self,
        z: Tensor,
        z_next: Tensor,
        r: Tensor,
        metric: 'PoincareDiskMetric'
    ) -> Tuple[Tensor, dict]:
        """
        Compute a TD consistency proxy for the screened Poisson/Bellman equation.

        The loss has two components:
        1. TD Error: Enforces Bellman consistency (the PDE source term)
        2. Geometric Regularization: Ensures V respects manifold structure

        Args:
            z: Current latent positions [B, D]
            z_next: Next latent positions [B, D]
            r: Rewards [B, 1]
            metric: The Riemannian metric on Z

        Returns:
            loss: Total loss scalar
            info: Dictionary with diagnostic values
        """
        gamma = self.config.gamma

        # Compute V(z) and V(z')
        V = self(z)
        with torch.no_grad():
            V_next = self(z_next)

        # 1. TD consistency proxy (the sampled Bellman relation)
        # V(z) = r + gamma * V(z') is not, by itself, a spatial PDE residual.
        td_error = V - (r + gamma * V_next)
        loss_pde = td_error.pow(2).mean()

        # 2. Geometric Regularization
        # Penalize large metric-weighted gradient norm ||grad V||_G^2
        # This is a smoothness prior that respects the manifold geometry
        z.requires_grad_(True)
        V_for_grad = self(z)
        grad_V = torch.autograd.grad(
            V_for_grad.sum(), z, create_graph=True
        )[0]  # [B, D]

        # Compute ||grad V||_G^2 = G^{ij} (d_i V)(d_j V)
        G_inv = metric.inverse(z)  # [B, D, D]
        grad_norm_sq = torch.einsum(
            'bi,bij,bj->b', grad_V, G_inv, grad_V
        )  # [B]
        loss_smoothness = grad_norm_sq.mean()

        # Total loss
        loss = loss_pde + self.config.grad_reg_weight * loss_smoothness

        info = {
            'td_error': td_error.abs().mean().item(),
            'grad_norm': grad_norm_sq.sqrt().mean().item(),
            'V_mean': V.mean().item(),
            'V_std': V.std().item(),
        }

        return loss, info

    def compute_hessian_norm(self, z: Tensor) -> Tensor:
        """
        Compute a coordinate-Hessian norm proxy for conformal coupling.

        Args:
            z: Latent positions [B, D]

        Returns:
            hess_norm: Operator norm of Hessian [B]
        """
        B, D = z.shape
        z = z.requires_grad_(True)

        # Compute gradient
        V = self(z)  # [B, 1]
        grad_V = torch.autograd.grad(
            V.sum(), z, create_graph=True
        )[0]  # [B, D]

        # Compute Hessian row by row
        hessian = []
        for i in range(D):
            grad_i = torch.autograd.grad(
                grad_V[:, i].sum(), z, retain_graph=True
            )[0]  # [B, D]
            hessian.append(grad_i)

        H = torch.stack(hessian, dim=1)  # [B, D, D]

        # This is a coordinate Frobenius proxy; it is not the covariant
        # operator norm of the Hessian on a general Riemannian manifold.
        hess_norm = torch.linalg.matrix_norm(H, ord='fro')  # [B]

        return hess_norm

    def conformal_factor(self, z: Tensor) -> Tensor:
        """
        Definition 24.4.1: Compute Omega(z) from Value Hessian.

        Omega(z) = 1 + alpha_conf * ||nabla^2 V(z)||

        Args:
            z: Latent positions [B, D]

        Returns:
            Omega: Conformal factor [B]
        """
        hess_norm = self.compute_hessian_norm(z)
        Omega = 1.0 + self.alpha_conf * hess_norm
        return Omega

    def conformally_scaled_metric(
        self,
        z: Tensor,
        base_metric: 'PoincareDiskMetric'
    ) -> Tensor:
        """
        Compute the conformally scaled metric G_tilde = Omega^2 * G.

        Args:
            z: Latent positions [B, D]
            base_metric: The base Poincare disk metric

        Returns:
            G_tilde: Conformally scaled metric [B, D, D]
        """
        Omega = self.conformal_factor(z)  # [B]
        G = base_metric(z)  # [B, D, D]
        G_tilde = Omega.unsqueeze(-1).unsqueeze(-1).pow(2) * G
        return G_tilde


def compute_wfr_reaction_rate(
    V: Tensor,
    s_r: float = 1.0
) -> Tensor:
    """
    Chosen WFR reaction closure: compute a reaction rate from the value function.

    r(z) = (V(z) - mean(V)) / s_r

    Positive values increase the local reaction rate under this convention; this is not a
    consequence of the WFR action alone.

    Args:
        V: Value function evaluations [B, 1]
        s_r: Reaction time scale

    Returns:
        r: Reaction rates [B, 1]
    """
    V_mean = V.mean()
    r = (V - V_mean) / s_r
    return r
```

**Algorithm 24.5.1 (Critic Training Loop).**

```python
def train_critic_step(
    critic: HolographicCritic,
    batch: dict,
    metric: 'PoincareDiskMetric',
    optimizer: torch.optim.Optimizer
) -> dict:
    """
    Single training step for the HolographicCritic.
    Optimizes the sampled TD consistency proxy used by ``compute_helmholtz_loss``.
    """
    z = batch['z']           # Current latent [B, D]
    z_next = batch['z_next'] # Next latent [B, D]
    r = batch['reward']      # Rewards [B, 1]

    optimizer.zero_grad()
    loss, info = critic.compute_helmholtz_loss(z, z_next, r, metric)
    loss.backward()
    optimizer.step()

    return info
```

**Cross-references:** {ref}`sec-the-geodesic-baoab-integrator` (BAOAB integrator uses $\nabla\Phi_{\text{eff}}$), Section 23.7 (HolographicInterface).

(sec-the-unified-holographic-dictionary)=
## The Unified Holographic Dictionary

:::{div} feynman-prose
Let's step back and admire the dictionary we have built. It is a set of useful correspondences between RL objects and field-theory objects, with a choice of boundary data and dynamics attached to each one.

Here is the careful version:
- An observation trace can be treated as Dirichlet-like data when the model clamps a boundary value.
- An action or WFR transport flux can be treated as Neumann-like data when a normal flux is actually prescribed.
- A reward 1-form supplies directional line-integral data; a bulk reward density or an independently specified boundary source supplies the source term for a scalar PDE.
- The value function is a potential field only in the conservative scalar sector.
- The discount factor gives a screening coefficient in the stated diffusion convention; curved-space decay needs its own analysis.
- A policy can be represented as an external force in the selected SDE, while temperature is a thermostat parameter only when the friction and noise satisfy the chosen fluctuation-dissipation convention.

The agent's motion may then be written as a geodesic SDE on a curved manifold, but that equation is an operational model with explicit friction, curl, metric, and boundary hypotheses. The electrodynamics language is an isomorphism of roles and intuition, not an identification theorem.

If you understand which object supplies each datum, the dictionary becomes a reliable guide rather than a source of accidental boundary conditions.
:::

This completes the **Holographic Dictionary** for the Fragile Agent. We now have a complete mapping between boundary data (observations, actions, rewards) and bulk objects (position, momentum, potential).

**Table 24.6.1 (Complete Holographic Dictionary).**

| Phenomenon     | Boundary (Data)  | Bulk (Latent)                           | Mathematical Object | Neural Component | Boundary Condition         | Section |
|----------------|------------------|-----------------------------------------|---------------------|------------------|----------------------------|---------|
| **Perception** | Pixels $\phi(x)$ | Position $q \in \mathcal{Z}$            | Manifold Point      | Visual Encoder   | Dirichlet (clamp position) | [23.1](#sec-the-symplectic-interface-position-momentum-duality) |
| **Action**     | Torques $A(x)$   | Momentum $p \in T\mathcal{Z}$           | Tangent Vector      | Action Encoder   | Neumann (clamp flux)       | [23.1](#sec-the-symplectic-interface-position-momentum-duality) |
| **Reward**     | Charge $r(x)$    | Potential $V \in C^\infty(\mathcal{Z})$ | Scalar Field        | Critic           | Source (Poisson)           | [24.1](#sec-the-reward-1-form) |
| **State**      | —                | $(q, p)$                                | Phase space point   | Full state       | Combined BCs               | [23.1](#sec-the-symplectic-interface-position-momentum-duality) |
| **Dynamics**   | —                | Geodesic flow                           | Hamiltonian flow    | BAOAB integrator | —                          | [22.4](#sec-the-geodesic-baoab-integrator) |

:::{prf:remark} RL--Electrodynamics Correspondence (Formal Analogy)
:label: thm-rl-as-electrodynamics-on-a-curved-manifold

Under the conservative geodesic Langevin convention of the equations-of-motion chapter, the analogy can be written as:

The agent is a **particle** with:
- **Position** $q \in \mathcal{Z}$ (from Perception / Dirichlet BC)
- **Momentum** $p \in T_q\mathcal{Z}$ (from Action / Neumann BC)
- **Mass** $G(q)$ (the Riemannian metric = information geometry)
- **Potential Energy** $\Phi_{\mathrm{eff}}(q)$ (from the conservative reward/cost potential)
- **External Forces** $u_\pi(q)$ (from Policy / symmetry-breaking kick)

moving according to the **geodesic SDE** (Definition {prf:ref}`def-bulk-drift-continuous-flow`):

$$
dq^k = G^{kj}(q) p_j \, ds, \qquad
dp_k = \left[-\partial_k\Phi_{\mathrm{eff}} - \gamma p_k
+\beta_{\text{curl}}\mathcal{F}_{kj}G^{j\ell}p_\ell
+\Gamma^m_{k\ell}G^{\ell j}p_jp_m + \gamma G_{kj}u_\pi^j\right]ds
+\sqrt{2\gamma T_c}\,(G^{1/2})_{kj}dW^j_s,

$$
on a **curved manifold** with metric $G$ satisfying the **capacity constraint** (Theorem {prf:ref}`thm-capacity-constrained-metric-law`). The Christoffel term is the geodesic correction in covector momentum coordinates. Treating the conservative component as a screened Helmholtz solution requires the conservative diffusion hypotheses of {prf:ref}`thm-the-hjb-helmholtz-correspondence`.

This is a physics-inspired correspondence, not an identification theorem. The standard RL components can be
organized using the following field-theory roles:

| RL Component      | Field Theory Role                                         |
|-------------------|-----------------------------------------------------------|
| Encoder           | **Coordinate Chart** (embedding from boundary to bulk)    |
| Critic            | **Field Solver** (Green's function of screened Laplacian) |
| Policy            | **External Force** (symmetry-breaking current)            |
| Discount $\gamma$ | **Screening Mass** (controls correlation length)          |
| Temperature $T_c$ | **Thermal Bath** (fluctuation-dissipation source)         |

:::
:::{prf:remark} Three Interface Roles (Operational)
:label: cor-the-three-boundary-conditions

The agent-environment interface decomposes into exactly three types of boundary conditions:

1. **Dirichlet** (Sensors): Clamp position $q = q_{\text{obs}}$. Information flows **in**.
2. **Neumann** (Motors): Clamp flux $\nabla_n \cdot p = j_{\text{motor}}$. Information flows **out**.
3. **Reward trace/source**: choose either a boundary trace for $\Phi$ or an independently specified Neumann/source
   datum $\sigma_r$. The pullback $\iota^*\mathcal R$ supplies tangential (Dirichlet-type) data and is not itself
   a charge density.

These three conditions fully specify the agent's interaction with its environment.

**Cross-references:** {ref}`sec-the-boundary-interface-symplectic-structure` (Holographic Interface), {ref}`sec-the-equations-of-motion-geodesic-jump-diffusion` (Equations of Motion), {ref}`sec-capacity-constrained-metric-law-geometry-from-interface-limits` (Capacity-Constrained Geometry).

:::
(sec-diagnostic-nodes-for-the-scalar-field)=
## Diagnostic Nodes for the Reward Field

:::{div} feynman-prose
Finally, we need to know when things are going wrong. The Critic is a complex system---a sampled Bellman consistency objective coupled to optional geometric back-reaction. There are many ways it can fail, and the diagnostics tell us which declared model assumption is under strain.

Node 35 measures a residual for the selected Helmholtz/Bellman model. A large value says that the sampled critic is inconsistent with that residual; it does not by itself prove non-convergence of a full boundary-value solver.

Node 36 fits the decay of a measured Green response over a declared asymptotic window. The flat-space exponential and $\ell=1/\kappa$ are reference formulas for the selected operator; on a hyperbolic or bounded domain, the geometry and boundary terms change the fitted rate.

Node 37 compares empirical sampling with the declared Boltzmann model. Agreement is meaningful only when the invariant measure, drift, boundary conditions, and detailed balance hypotheses hold. Disagreement is not automatically an exploration-exploitation diagnosis.

Node 38 monitors variation in the conformal factor. It can reveal weak coupling or excessive distortion; a claim that the
agent is stuck requires trajectory or integrator evidence in addition to this diagnostic.

Node 39 checks the correlation predicted by the chosen value-based reaction closure. The WFR action alone does not say that high value creates mass; the correlation is a model diagnostic.

Node 61 measures circulation on approximately closed sampled loops. Near-zero circulation on the tested family is evidence compatible with an exact reward field, while nonzero circulation detects a path-dependence signal. A NESS interpretation requires additional stationarity and detailed-balance assumptions, and a curl estimate requires local plaquette or differential data.
:::

We define six diagnostic nodes (35-39, 61) to monitor the health of the Critic/Value system, including the new ValueCurlCheck for non-conservative reward fields.

(node-35)=
**Node 35: HelmholtzResidualCheck**

| **#** | **Name** | **Component** | **Type** | **Interpretation** | **Proxy** | **Cost** |
|-------|----------|---------------|----------|-------------------|-----------|----------|
| **35** | **HelmholtzResidualCheck** | **Critic** | **PDE Consistency** | Is the Helmholtz equation satisfied? | $\lVert-\Delta_G V + \kappa^2 V - \rho_r\rVert$ | $O(B \cdot D^2)$ |

**Trigger conditions:**
- High HelmholtzResidualCheck: Bellman equation not converged; Critic training unstable.
- Remedy: Reduce learning rate; increase batch size; check reward normalization.

(node-36)=
**Node 36: GreensFunctionDecayCheck**

| **#** | **Name** | **Component** | **Type** | **Interpretation** | **Proxy** | **Cost** |
|-------|----------|---------------|----------|-------------------|-----------|----------|
| **36** | **GreensFunctionDecayCheck** | **Critic** | **Screening Length** | Does a fitted decay agree with the selected geometry? | fit of $\log|G_\kappa|$ versus $d_G$ over a declared asymptotic window | $O(B^2)$ |

**Trigger conditions:**
- High GreensFunctionDecayCheck: the fitted decay is inconsistent with the chosen screened-operator model; this is not a universal flat-space test.
- Remedy: Check discount factor; verify metric computation; inspect reward structure.

(node-37)=
**Node 37: BoltzmannConsistencyCheck**

| **#** | **Name** | **Component** | **Type** | **Interpretation** | **Proxy** | **Cost** |
|-------|----------|---------------|----------|-------------------|-----------|----------|
| **37** | **BoltzmannConsistencyCheck** | **Critic + Policy** | **Conditional Equilibrium** | Does empirical sampling match the declared Boltzmann model? | $D_{\mathrm{KL}}(P_{\text{empirical}} \lVert P_{\text{Boltzmann}})$ | $O(B \cdot D)$ |

**Trigger conditions:**
- High BoltzmannConsistencyCheck: sampling disagrees with the Boltzmann model under its detailed-balance and invariant-measure hypotheses.
- Remedy: Adjust cognitive temperature $T_c$; check policy entropy; verify WFR reaction rate.

(node-38)=
**Node 38: ConformalBackReactionCheck**

| **#** | **Name** | **Component** | **Type** | **Interpretation** | **Proxy** | **Cost** |
|-------|----------|---------------|----------|-------------------|-----------|----------|
| **38** | **ConformalBackReactionCheck** | **Critic** | **Geometry Coupling** | Is value curvature affecting metric appropriately? | $\text{Var}(\Omega(z))$ in high-$\lVert\nabla^2 V\rVert$ regions | $O(B \cdot D^2)$ |

**Trigger conditions:**
- Low ConformalBackReactionCheck: Value landscape is flat; agent not distinguishing important regions.
- High ConformalBackReactionCheck: Excessive metric distortion; agent "stuck" at decision boundaries.
- Remedy: Adjust conformal coupling $\alpha_{\text{conf}}$; verify Hessian computation.

(node-39)=
**Node 39: ValueMassCorrelationCheck**

| **#** | **Name** | **Component** | **Type** | **Interpretation** | **Proxy** | **Cost** |
|-------|----------|---------------|----------|-------------------|-----------|----------|
| **39** | **ValueMassCorrelationCheck** | **WFR + Critic** | **Chosen Reaction Closure** | Does the selected reaction law correlate mass change with value? | $\text{corr}(\dot m_t, V(z_t)-\bar V)$ | $O(B)$ |

**Trigger conditions:**
- Low ValueMassCorrelationCheck: the selected value-based reaction closure is not visible in the measured mass change; the WFR action alone makes no such prediction.
- Remedy: Check reaction rate computation; verify WFR dynamics; inspect value function gradients.

(node-61)=
**Node 61: ValueCurlCheck**

| **#** | **Name** | **Component** | **Type** | **Interpretation** | **Proxy** | **Cost** |
|-------|----------|---------------|----------|-------------------|-----------|----------|
| **61** | **ValueCurlCheck** | **Critic** | **Topology** | Is the reward 1-form exact on sampled loops? | $\left|\oint_{\partial\Sigma}\mathcal{R}\right|$ on approximately closed loops | $O(T)$ |

**Trigger conditions:**
- Near-zero ValueCurlCheck on a family of contractible loops is consistent with a conservative reward model, subject to loop coverage and estimator error.
- Non-zero ValueCurlCheck detects circulation. A NESS interpretation additionally requires a stationary solution and broken detailed balance. Consider:
  - **Productive curl:** Value cycles that harvest reward continuously (e.g., exploration-exploitation orbits)
  - **Pathological curl:** Indicates preference intransitivity or reward misspecification
- Remedy: If unexpected non-conservative structure, verify reward function consistency; check for cyclic dependencies in multi-objective rewards.

**Diagnostic Implementation:**
```python
def value_curl_check(
    z_trajectory: Tensor,  # [T, D] closed loop trajectory
    rewards: Tensor,       # [T] rewards along trajectory
) -> float:
    """
    Estimate Value Curl via loop integral of TD-errors.
    Non-zero return indicates non-conservative reward field.
    """
    # A loop integral is a circulation proxy; rewards must be interpreted as
    # samples of a consistently estimated reward 1-form.
    loop_integral = rewards.sum().item()
    return abs(loop_integral)
```

**Table 24.8.1 ({ref}`sec-the-reward-field-value-forms-and-hodge-geometry` Diagnostic Summary).**

| # | Name | Monitors | Healthy Range |
|---|------|----------|---------------|
| 35 | HelmholtzResidualCheck | PDE consistency | $< 0.1$ |
| 36 | GreensFunctionDecayCheck | Screening behavior | $\approx 1.0$ (constant after scaling) |
| 37 | BoltzmannConsistencyCheck | Equilibrium sampling | $< 0.5$ nats |
| 38 | ConformalBackReactionCheck | Geometry coupling | $0.1 < \text{Var}(\Omega) < 2.0$ |
| 39 | ValueMassCorrelationCheck | WFR-Value alignment | $> 0.5$ |
| 61 | ValueCurlCheck | Non-conservative structure | Context-dependent (see above) |

**Cross-references:** {ref}`sec-diagnostics-stability-checks` (Sieve Diagnostic Nodes), {ref}`sec-summary-tables-and-diagnostic-nodes-a` (Interface Diagnostics Nodes 30-34), Theorem {prf:ref}`thm-hodge-decomposition` (Hodge Decomposition), Definition {prf:ref}`def-value-curl` (Value Curl).
