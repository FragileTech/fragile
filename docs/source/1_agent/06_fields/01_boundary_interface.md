(sec-the-boundary-interface-symplectic-structure)=
# The Boundary Interface: Symplectic Structure

## TLDR

- A phase-space lift of the agent–world interface can carry a **symplectic structure** where observations and actions
  are represented by conjugate coordinates.
- Treat sensors/actions as boundary constraints on latent dynamics (Dirichlet/Neumann-style), making coupling auditable.
- This chapter defines the “contact surface” that later field equations (reward/value forms, information bounds) depend
  on.
- Operationally: it tells you what information must be exposed at the boundary and how to test coupling/grounding.
- The main failure modes are decoupling (ungrounded inference) and instability (feedback amplification); both are
  monitorable.

## Roadmap

1. Symplectic interface picture and why it is the right mathematical type.
2. How observations/actions appear as boundary conditions on internal dynamics.
3. Diagnostics for coupling and grounding at the interface.

{cite}`arnold1989mathematical`

:::{div} feynman-prose
All right, now we come to something that I find absolutely fascinating---the place where the agent meets the world. You see, so far we've been talking about what happens *inside* the agent, all this beautiful geometry and dynamics on the latent manifold. But an agent that doesn't touch reality isn't much of an agent, is it?

Here's the profound question: How does information get *in* and *out*? Not just "sensors provide data and motors send commands"---that's the boring answer. The interesting answer is that the interface has a useful mathematical structure with two layers. The geometric boundary carries observation traces and fluxes. If we add an even-dimensional phase-space lift with positions and momenta, that lift can carry the canonical symplectic form.

The symplectic statement is exact on that lift when the canonical hypotheses hold. On the base interface, observations and actions are a modeling correspondence, much like position and momentum in physics. This organization clarifies how sensors and motors are coupled without pretending that the analogy alone supplies every PDE boundary condition.
:::

(rb-boundary-conditions)=
:::{admonition} Researcher Bridge: Observations and Actions as Boundary Conditions
:class: info
In standard RL, observations and actions are inputs and outputs. Here they are represented by operational boundary
prescriptions on the latent dynamics: sensor traces are Dirichlet-like and motor fluxes are Neumann-like. The labels do
not assert that an arbitrary environment automatically supplies those PDE boundary conditions.
:::

:::{div} feynman-prose
We have defined the internal dynamics of the agent (the interior) as a Jump-Diffusion process on a Riemannian fiber bundle ({ref}`sec-wasserstein-fisher-rao-geometry-unified-transport-on-hybrid-state-spaces`). We now describe its coupling to the external world.

The interface uses two operational boundary prescriptions: a **Dirichlet-like observation trace** for sensors and a **Neumann-like WFR transport flux** for motors. The word "like" is doing real work here.

Now, what do these terms mean? If you've taken a course in partial differential equations, you know that to solve an equation on a region, you need to specify what happens at the boundary. There are two classic choices:

- **Dirichlet**: You fix the *value* of the solution at the boundary. "The temperature at the wall is 100 degrees."
- **Neumann**: You fix the *flux* (the rate of flow) at the boundary. "Heat flows out through the wall at 50 watts per square meter."

The useful correspondence is that sensors provide a target or trace for *where* the encoded state should be, while motors prescribe a transport flux for *how* probability is pushed. An exact PDE Dirichlet or Neumann statement would require a specified trace, flux, domain, and limiting procedure. Here the labels organize the interface model; they do not turn every encoder target into a literal boundary clamp or every motor command into a PDE normal derivative.
:::

(sec-the-symplectic-interface-position-momentum-duality)=
## The Symplectic Interface: Position-Momentum Duality

:::{div} feynman-prose
Now we get to the heart of the matter. The boundary between agent and environment isn't just a wall or a membrane---it has *structure*. The boundary itself need not be symplectic; the canonical symplectic object appears when we choose an even-dimensional phase-space lift.

What's a symplectic manifold? Let me give you the picture first, then the mathematics. Imagine a dance floor. At each point on the floor, you could be standing still, or moving in some direction, or spinning. Now, the symplectic structure is like a rule that says: if you know your position *and* your momentum, you know everything there is to know about your motion. Position and momentum together form a complete description.

But here's the beautiful part: they're not independent. They're *conjugate*. If you change your position, it affects how your momentum evolves, and vice versa. In classical mechanics, Hamilton's equations tell you exactly how: $\dot{q} = \partial H/\partial p$ and $\dot{p} = -\partial H/\partial q$. Position comes from momentum; momentum comes from position. They're locked in an eternal dance.

At the agent's interface, observations can be represented by position-like coordinates and actions by momentum- or flux-like coordinates on that lift. Hamilton's equations then describe the chosen phase-space model. That is an exact statement about the lift, while the sensor/action interpretation remains conditional on how the interface maps are defined.
:::

The boundary $\partial\mathcal{Z}$ between agent and environment is an interface on the latent
state space. When the interface is lifted to a cotangent phase space with explicit position and
momentum variables, that lift carries the canonical symplectic form. The boundary sphere by itself
need not be even-dimensional or symplectic.

:::{prf:definition} Symplectic Interface Lift
:label: def-symplectic-boundary-manifold

For an interface whose lifted phase space is $T^*\mathcal{Q}$, use canonical coordinates $(q,p)$ and
the symplectic form $(T^*\mathcal{Q},\omega)$ where:
- $q \in \mathcal{Q}$ is the **position bundle** (sensory configuration)
- $p \in T^*_q\mathcal{Q}$ is the **momentum bundle** (motor flux)

The symplectic form is:

$$
\omega = \sum_{i=1}^n dq^i \wedge dp_i.

$$
The units of $\omega$ are inherited from the chosen coordinate and momentum units; no information
unit is implied without an explicit normalization.

*Remark (Causal Structure).* The symplectic structure encodes causality: observations fix "where" the belief state is (position), while actions fix "how" it flows outward (momentum/flux). These cannot be treated symmetrically as static fields.

:::

:::{div} feynman-prose
Let me unpack that definition a bit. The symplectic form $\omega = \sum_i dq^i \wedge dp_i$ might look like abstract nonsense, but it records the oriented area pairing on the lifted phase space. Canonical maps, including Hamiltonian flows under the usual hypotheses, preserve that pairing.

The units come from whatever coordinate and momentum units the model declares; $\omega$ is not an information measure in nat units unless an additional normalization says so. Fixing $q$ and prescribing $p$ are useful ways to describe the two interface channels, but the distinction does not by itself prove a Dirichlet-to-Neumann swap. One may exchange coordinate roles on the lift; the PDE boundary semantics still have to be specified separately.
:::

:::{prf:definition} Sensory Assimilation Target (Dirichlet-like limit)
:label: def-dirichlet-boundary-condition-sensors

The sensory input stream $\phi(x)$ supplies an observation posterior or
target density in the latent domain:

$$
\rho_{\partial}^{\text{sense}}(q, t) = \delta(q - q_{\text{obs}}(t)),

$$
where $q_{\text{obs}}(t) = E_\phi(x_t)$ is the encoded observation. A literal
delta is the zero-noise limit. In the WFR continuity equation this target is
normally coupled through an assimilation source

$$
S_{\mathrm{obs}}(z,t)=\kappa_{\mathrm{obs}}
\bigl(\rho_{\mathrm{obs}}(z,t)-\rho_{\mathrm{bulk}}(z,t)\bigr),
$$

with a finite-width likelihood for a noisy sensor. A literal Dirichlet trace
is recovered only as a strong-assimilation limit on a genuine geometric
boundary.

*Interpretation:* Information flow from environment to agent (observation).

:::

:::{div} feynman-prose
Think about what that delta function means. It is an idealized observation target: in the zero-noise or strong-assimilation limit, the trace is concentrated at $q_{\text{obs}}$. A finite-noise encoder generally supplies a likelihood or finite-width posterior, and the encoded point may be an interior state rather than a literal geometric boundary trace.

So "clamping" is shorthand for a limiting observation model. A camera or accelerometer can localize one aspect of the representation while leaving uncertainty in the others; the mathematics has to retain that uncertainty when the sensor model is not noiseless.
:::

:::{prf:definition} Neumann Boundary Condition --- Motors
:label: def-neumann-boundary-condition-motors

The motor output stream imposes a **Neumann-like** condition on the WFR transport flux:

$$
j_{\rho}\!\cdot\mathbf{n}\big|_{\partial\mathcal{Z}_{\text{motor}}}
= j_{\text{motor}}(z,u_\pi,t),

$$
where $j_{\rho}=\rho v$ for the WFR transport equation. If a Fokker--Planck
diffusion is included, use the total probability flux
$j_{\rho}=\rho v-T_cG^{-1}\nabla\rho$. The motor current is an
information-flux functional of the texture-free interface state:

$$
j_{\text{motor}} = J_A(z,u_\pi,t),

$$
*Interpretation:* Information flow from agent to environment (action). Its
units are the density-flux units induced by the chosen WFR time coordinate.
The raw decoder output (torques or voltages) is a separate physical quantity;
an interface map is required before it can be converted into $J_A$.

:::

:::{div} feynman-prose
Motors work differently from sensors. Instead of specifying *where* the encoded state is, they can specify *how much transport* crosses the interface. Think of it like this: a sensor is a window you look through; a motor is a faucet you turn on.

For the WFR transport part, the relevant normal flux is $j_\rho\!\cdot\mathbf{n}=(\rho v)\!\cdot\mathbf{n}$. If the chosen Fokker--Planck model has diffusion, its diffusive contribution belongs in the total probability flux as well. The decoder's motor command is a separate object and acquires the same flux meaning only after the interface map and units have been declared.

That is why "Neumann-like" is a useful operational label: it says that a flux functional is prescribed, while the state trace is allowed to vary. It is not a claim that the motor condition is the raw normal derivative $\nabla_n\rho$.
:::

(pi-hamiltonian-bc)=
::::{admonition} Physics Isomorphism: Hamiltonian Boundary Conditions
:class: note

**In Physics:** In Hamiltonian mechanics, canonical coordinates $(q, p)$ satisfy $\dot{q} = \partial H/\partial p$ and $\dot{p} = -\partial H/\partial q$. A phase-space model may prescribe $q$ or $p$ at an interface; identifying $p$ with a PDE normal derivative requires a separate field-theoretic boundary model {cite}`arnold1989mathematical`.

**In Implementation:** The agent's interface imposes dual boundary conditions:
- **Perception:** Observations supply a trace or assimilation target $\rho_{\mathrm{obs}}$.
- **Action:** Motors prescribe the WFR flux $j_{\rho}\!\cdot n=J_A$ after an interface map.
- **Reward (Source):** Boundary reward flux $J_r|_{\partial\mathcal{Z}}$ (scalar charge density
  $\sigma_r$ in the conservative case)

**Correspondence Table:**

| Hamiltonian Mechanics | Agent (Symplectic Interface) |
|:----------------------|:-----------------------------|
| Position $q$ | Latent state $z$ |
| Momentum $p$ | Interface momentum/flux coordinate |
| Base constraint $q\vert_{\Gamma}=q_0$ | Observation trace (when a trace model is declared) |
| Fibre constraint $p\vert_{\Gamma}=p_0$ | Action or motor constraint on the lift |
| Hamiltonian $H(q,p)=\tfrac12\|p\|_{G^{-1}}^2+\Phi_{\mathrm{eff}}(q)$ | Phase-space energy model |
::::

:::{prf:remark} Conditional Symplectic Duality
:label: prop-symplectic-duality-principle

On a genuine even-dimensional phase-space lift, the canonical transformation
$(q,p)\mapsto(p,-q)$ exchanges coordinate roles. This gives a useful analogy between sensing and
actuation. It does not, by itself, map a PDE Dirichlet trace into a Neumann flux condition; that
requires a specified Hamiltonian boundary-value problem and a Legendre transform on the same
configuration manifold.

**Cross-references:** {ref}`sec-the-interface-and-observation-inflow` (Observation inflow), Definition {prf:ref}`def-dirichlet-boundary-condition-sensors`.

:::

:::{div} feynman-prose
Now here's the subtle point. On a genuine phase-space lift, the map $(q,p)\mapsto(p,-q)$ preserves the canonical structure and exchanges coordinate roles. That gives a useful analogy between sensing and actuation, but it does not turn a PDE Dirichlet trace into a Neumann flux by itself. For that conclusion one needs a specified Hamiltonian boundary-value problem and a Legendre transform on the same configuration manifold.

The practical lesson is more modest and more useful: perception and action can share design patterns because their lifted coordinates have a canonical relation. A Visual Atlas and an Action Atlas may exploit that relation, but their equivalence is conditional on the lift and regularity assumptions.
:::

(sec-the-dual-atlas-architecture)=
## The Dual Atlas Architecture

:::{div} feynman-prose
Now that we understand the symplectic lift, we need to actually build something. How do you implement an interface that respects the parts of the structure we have chosen?

The answer is to use *atlases*---collections of charts that together cover the whole space. If you've studied differential geometry, you know that a manifold is defined by its atlas: a set of overlapping patches, each with its own coordinate system, with smooth transitions between them.

Here's the key insight: perception and action may need separate atlases because they live on different domains. A Visual Atlas tells you "given what I see, where am I?" An Action Atlas tells you "given what I want to do, how do I push?" A Legendre transform can connect lifted configuration-velocity and configuration-momentum descriptions, just as in mechanics, when a regular Lagrangian and the required lift have been specified.

Why do we need separate atlases? Because the same physical situation might look very different from the perception side versus the action side. When you're looking at a cup, the visual representation involves shape, color, distance. When you're reaching for that cup, the motor representation involves joint angles, velocities, forces. These can be different coordinate systems on related spaces. A Legendre transform translates the lifted mechanical variables when its hypotheses hold; it does not automatically identify independently learned visual and action charts.
:::

To implement the symplectic interface, we require two symmetric topological structures: a **Visual Atlas** for perception and an **Action Atlas** for actuation. This symmetrizes the architecture from {ref}`sec-the-shutter-as-a-vq-vae`.

:::{prf:definition} Visual Atlas — Perception
:label: def-visual-atlas-perception

The Visual Atlas $\mathcal{A}_{\text{vis}} = \{(U_\alpha, \phi_\alpha, e_\alpha^{\text{vis}})\}_{\alpha \in \mathcal{K}_{\text{vis}}}$ is a chart atlas on the sensory manifold $\mathcal{Q}$ with:
- **Charts** $U_\alpha \subset \mathcal{Q}$: Objects, Scenes, Viewpoints
- **Chart maps** $\phi_\alpha: U_\alpha \to \mathbb{R}^{d_{\text{vis}}}$: Local coordinates
- **Codebook embeddings** $e_\alpha^{\text{vis}} \in \mathbb{R}^{d_m}$: Discrete macro codes

*Input:* Raw observations $\phi_{\text{raw}}$ (pixels, sensors).
*Output:* Latent state $z \in \mathcal{Z}$ (configuration).

:::

:::{div} feynman-prose
Notice what the Visual Atlas does. It takes the raw visual chaos---pixels, shapes, colors---and organizes it into a structured representation. The charts ($U_\alpha$) are like different "ways of seeing": one chart might specialize in recognizing faces, another in outdoor scenes, another in small objects. The codebook embeddings ($e_\alpha^{\text{vis}}$) are the discrete labels: "this is a face," "this is a tree."

The output is a position in the latent space $\mathcal{Z}$. Every time you see something, the Visual Atlas tells you where you've landed in this internal coordinate system.
:::

:::{prf:definition} Action Atlas --- Actuation
:label: def-action-atlas-actuation

The Action Atlas $\mathcal{A}_{\text{act}} = \{(V_\beta, \psi_\beta, e_\beta^{\text{act}})\}_{\beta \in \mathcal{K}_{\text{act}}}$ is a chart atlas on the motor manifold $T^*\mathcal{Q}$ with:
- **Charts** $V_\beta \subset T^*\mathcal{Q}$: Gaits, Grasps, Tool Affordances (topologically distinct control regimes)
- **Chart maps** $\psi_\beta: V_\beta \to \mathbb{R}^{d_{\text{act}}}$: Local motor coordinates
- **Codebook embeddings** $e_\beta^{\text{act}} \in \mathbb{R}^{d_m}$: Action primitive codes

*Input:* Intention $u_{\text{intent}} \in T_z\mathcal{Z}$ (from Policy, {ref}`sec-policy-control-field`).
*Output:* Actuation $a_{\text{raw}}$ (torques, voltages).

*Remark (Jump Operator in Action Atlas).* The **Jump Operator** $L_{\beta \to \beta'}$ in the Action Atlas represents **Task Switching**: transitioning from one control primitive to another (e.g., "Walk" $\to$ "Jump", "Grasp" $\to$ "Release"). This mirrors the chart transition operator in the Visual Atlas ({ref}`sec-the-unified-world-model`).

:::

:::{div} feynman-prose
The Action Atlas can mirror the Visual Atlas, but on the motor side. Instead of "ways of seeing," you have "ways of doing." One chart might be for walking, another for grasping, another for using a tool. Some control regimes are separated by genuine mode switches; whether two regimes can be smoothly interpolated is a property of the chosen motor manifold, not of the word "atlas."

The Jump Operator is how you switch. It's the motor equivalent of a saccade in vision: a discrete transition from one mode of operation to another. When you stop walking and start reaching for something, you've jumped between charts in the Action Atlas.

The Legendre transform can connect the two descriptions after the visual representation has been lifted to $T\mathcal Q$ and a regular, strictly convex Lagrangian has been supplied. That conditional correspondence explains why perception and control may share an architecture. It does not force two independently learned atlases to have the same charts, counts, or transitions.
:::

:::{prf:proposition} Conditional Atlas Legendre Correspondence
:label: thm-atlas-duality-via-legendre-transform

Let $\mathcal A_{\mathrm{vis}}^T$ be the tangent lift of the visual atlas to
$T\mathcal Q$, and let $G_{\mathcal Q}=E_\phi^*G$ be a positive metric on
$\mathcal Q$ pulled back from the latent metric. If
$L(q,\dot q)$ is $C^2$ and strictly convex in $\dot q$, its Legendre map

$$
\mathcal L_L:T\mathcal Q\longrightarrow T^*\mathcal Q,
\qquad (q,\dot q)\longmapsto
\left(q,\frac{\partial L}{\partial\dot q}\right)
$$

is a fibrewise diffeomorphism. For
$L=\tfrac12\|\dot q\|_{G_{\mathcal Q}}^2-V(q)$, the induced momentum is
$p=G_{\mathcal Q}(q)\dot q$. On a lifted visual chart $(U_\alpha,
T\phi_\alpha)$, an induced action chart can therefore use

$$
\psi_\beta\circ\mathcal L_L\circ(T\phi_\alpha)^{-1}(q,\dot q)
 =\bigl(q,\,G_{\mathcal Q}(q)\dot q\bigr).
$$

This construction defines the compatible lifted atlas
$\mathcal L_L(\mathcal A_{\mathrm{vis}}^T)$. An independently learned action
atlas may have a different number of charts; compatibility is a design
constraint or alignment loss, not the equality
$\mathcal A_{\mathrm{act}}=\mathcal L_L(\mathcal A_{\mathrm{vis}})$.

*Proof sketch.* Strict fibre convexity makes the fibre derivative invertible,
and the Legendre map pulls the canonical symplectic form back to the
Poincare--Cartan form. The construction applies to the tangent lift, so the
visual atlas on $\mathcal Q$ is not silently treated as an atlas on
$T\mathcal Q$. No PDE Dirichlet-to-Neumann statement follows without an
additional boundary-value model. $\square$

*Remark (Why Legendre?).* Under these hypotheses the map relates velocity to
momentum. It does not identify independently learned visual and action charts
or make their index sets equal.

*Cross-reference:* The metric $G_{\mathcal Q}$ is the pullback metric used in
the lift. It need not be the latent metric itself until the encoder pullback
has been declared.

:::


(pi-legendre-transform)=
::::{admonition} Physics Isomorphism: Legendre Transform
:class: note

**In Physics:** The Legendre transform maps between Lagrangian and Hamiltonian formulations: $H(q,p) = p\dot{q} - L(q,\dot{q})$ where $p = \partial L/\partial \dot{q}$. It exchanges velocity for momentum as the independent variable {cite}`arnold1989mathematical`.

**In Implementation:** A lifted visual atlas induces compatible momentum
coordinates through the conditional Legendre construction (Theorem
{prf:ref}`thm-atlas-duality-via-legendre-transform`):

$$
\mathcal{L}: T\mathcal{Q} \to T^*\mathcal{Q}, \quad (z, \dot{z}) \mapsto (z, p = G\dot{z})

$$
**Correspondence Table:**
| Analytical Mechanics | Agent (Symplectic Interface) |
|:---------------------|:-----------------------------|
| Configuration space $\mathcal{Q}$ | Latent state space $\mathcal{Z}$ |
| Tangent bundle $T\mathcal{Q}$ | Velocity representation |
| Cotangent bundle $T^*\mathcal{Q}$ | Momentum representation |
| Lagrangian $L(q,\dot{q})$ | Kinetic action |
| Hamiltonian $H(q,p)$ | $\tfrac12\|p\|_{G^{-1}}^2+\Phi_{\text{eff}}(q)$ |
| Velocity $\dot{q}$ | Policy output $u_\pi$ |
| Momentum $p$ | Value gradient $\nabla_A V$ |

**Interface roles:** Perception supplies a base-coordinate trace or
assimilation target; action supplies a momentum/flux datum after the interface
map has been specified.
::::

:::{prf:definition} The Holographic Shutter — Unified Interface
:label: def-the-holographic-shutter-unified-interface

The Shutter is extended from {ref}`sec-the-shutter-as-a-vq-vae` to a symmetric tuple:

$$
\mathbb{S} = (\mathcal{A}_{\text{vis}}, \mathcal{A}_{\text{act}}),

$$
where:
- **Ingress (Perception):** $E_\phi: \mathcal{Q} \to \mathcal{Z}$ via Visual Atlas
- **Egress (Actuation):** $D_A: T_z\mathcal{Z} \times \mathcal{Z} \to T^*\mathcal{Q}$ via Action Atlas
- **Proprioception (Inverse Model):** $E_A: T^*\mathcal{Q} \to T_z\mathcal{Z}$ maps realized actions back to intentions

**Cross-references:** {ref}`sec-the-shutter-as-a-vq-vae` (VQ-VAE Shutter), {ref}`sec-tier-the-attentive-atlas` (AttentiveAtlasEncoder), {ref}`sec-decoder-architecture-overview-topological-decoder` (TopologicalDecoder).

:::
(sec-motor-texture-the-action-residual)=
## Motor Texture: The Action Residual

:::{div} feynman-prose
Now we come to something subtle but important. When you reach for a cup, your brain doesn't specify the exact position of every muscle fiber at every millisecond. It specifies something more abstract: "reach toward that location with this general trajectory." The fine details---the slight tremor in your fingers, the micro-adjustments for balance, the precise timing of individual motor units---those emerge from lower-level systems.

This is *motor texture*. It's the high-frequency, fine-grained detail of motor execution that doesn't matter for planning. Just like visual texture (the exact pixel values in an image) doesn't matter for recognizing what object you're looking at, motor texture doesn't matter for deciding what action to take.

The reason this matters is the sim-to-real gap. In simulation, your motors are perfect: no tremor, no noise, no friction. In reality, all of that exists. If your policy depends on motor texture, it will fail catastrophically in the real world. So we build a *firewall*: the policy never sees motor texture, and therefore can't depend on it. The texture is only used for low-level execution, not for decision-making.
:::

Just as visual texture captures reconstruction-only detail ({ref}`sec-the-retrieval-texture-firewall`), **motor texture** captures actuation-only detail that is excluded from planning.

:::{prf:definition} Motor Texture Decomposition
:label: def-motor-texture-decomposition

The motor output decomposes as:

$$
a_t = (K^{\text{act}}_t, z_{n,\text{motor}}, z_{\text{tex,motor}}),

$$
where:
- $K^{\text{act}}_t \in \mathcal{K}_{\text{act}}$ is the **discrete motor macro** (action primitive/chart index)
- $z_{n,\text{motor}} \in \mathbb{R}^{d_{\text{motor},n}}$ is **motor nuisance** (impedance, compliance, force distribution)
- $z_{\text{tex,motor}} \in \mathbb{R}^{d_{\text{motor,tex}}}$ is **motor texture** (tremor, fine-grained noise, micro-corrections)

*Remark (Parallel to Visual Decomposition).* This mirrors the visual decomposition $(K_t, z_{n,t}, z_{\text{tex},t})$ from {ref}`sec-the-shutter-as-a-vq-vae`:

| Component                 | Visual Domain                 | Motor Domain                              |
|---------------------------|-------------------------------|-------------------------------------------|
| **Macro (discrete)**      | Object/Scene chart $K$        | Action primitive $K^{\text{act}}$         |
| **Nuisance (continuous)** | Pose/viewpoint $z_n$          | Compliance/impedance $z_{n,\text{motor}}$ |
| **Texture (residual)**    | Pixel detail $z_{\text{tex}}$ | Tremor/noise $z_{\text{tex,motor}}$       |

:::
:::{prf:definition} Compliance Tensor
:label: def-compliance-tensor

The motor nuisance encodes the **compliance tensor**:

$$
C_{ij}(z_{n,\text{motor}}) = \frac{\partial a^i}{\partial f^j},

$$
where $f$ is the external force/feedback. This determines how the motor output responds to perturbations:
- **High compliance** ($C$ large): Soft, yielding response (safe interaction)
- **Low compliance** ($C$ small): Stiff, precise response (accurate positioning)

Units: $[C_{ij}] = [a]/[f]$.

:::
:::{prf:definition} Motor Texture Distribution
:label: def-motor-texture-distribution

At the motor boundary, texture is sampled from a geometry-dependent Gaussian:

$$
z_{\text{tex,motor}} \sim \mathcal{N}(0, \Sigma_{\text{motor}}(z)),

$$
where:

$$
\Sigma_{\text{motor}}(z) = \sigma_{\text{motor}}^2 \cdot G_{\text{motor}}^{-1}(z) = \sigma_{\text{motor}}^2 \cdot \frac{(1-|z|^2)^2}{4} I_{d_{\text{motor,tex}}}.

$$
This follows the same conformal scaling as visual texture (Definition {prf:ref}`def-boundary-texture-distribution`), ensuring consistent thermodynamic behavior.

:::
:::{prf:axiom} Motor Texture Firewall
:label: ax-motor-texture-firewall

Motor texture is decoupled from the Bulk dynamics:

$$
\partial_{z_{\text{tex,motor}}} \dot{z} = 0, \qquad \partial_{z_{\text{tex,motor}}} u_\pi = 0.

$$
The policy $\pi_\theta$ operates on $(K, z_n, A, z_{n,\text{motor}})$ but **never** on $(z_{\text{tex}}, z_{\text{tex,motor}})$.

*Remark (Sim-to-Real Gap).* The **motor texture variance** $\sigma_{\text{motor}}^2$ is the mathematical definition of the "Sim-to-Real gap":
- **Simulation:** $\sigma_{\text{motor}} \approx 0$ (deterministic, no tremor)
- **Reality:** $\sigma_{\text{motor}} > 0$ (friction, sensor noise, motor tremor)
- **Robustness:** The Bulk policy $u_\pi$ is invariant; only the Action Decoder learns to manage domain-specific noise.

**Cross-references:** {ref}`sec-the-retrieval-texture-firewall` (Texture Firewall), Axiom {prf:ref}`ax-bulk-boundary-decoupling`.

:::
(sec-the-belief-evolution-cycle-perception-dreaming-action)=
## The Belief Evolution Cycle: Perception--Dreaming--Action

:::{div} feynman-prose
All right, now we're going to tie everything together with a useful picture: a three-stage cycle of cognition.

Think of a heat engine as an analogy. It compresses gas, exchanges heat, expands, and repeats. Our cycle uses similar words for information operations:

1. **Perception (Compression)**: sensory data is encoded into a smaller internal description. Calling this "compression" is operational; the sign of an entropy change depends on which entropy and reference measure have been chosen.

2. **Dreaming (Internal evolution)**: with the sensory channel closed, the model evolves from its current state. A Hamiltonian, isentropic picture is valid only in an isolated zero-friction limit. The default WFR/Langevin dynamics can remain thermal and need not conserve entropy.

3. **Action (Expansion)**: an intention is decoded into a motor output, often with additional stochastic texture. The word "expansion" describes that map; it is not by itself a thermodynamic entropy theorem.

The Carnot comparison is therefore an analogy. The mutual-information ratio below is an operational diagnostic. A Carnot bound would require specified reservoirs and temperatures together with a proved entropy-production inequality; none follows from the cycle labels alone.
:::

The agent's interaction loop is a **belief density evolution cycle** on the information manifold.

:::{prf:definition} Cycle Phases
:label: def-cycle-phases


| Phase             | Process            | Information Flow                      | Entropy Change               |
|-------------------|--------------------|---------------------------------------|------------------------------|
| **I. Perception** | Compression        | Mutual information $I(X;K)$ extracted | $\Delta S_{\text{bulk}} < 0$ |
| **II. Dreaming**  | Internal evolution | No external exchange                  | Model-dependent; $\Delta S=0$ only in the isolated reversible limit  |
| **III. Action**   | Expansion          | Mutual information $I(A;K)$ injected  | $\Delta S_{\text{bulk}} > 0$ |

*Remark (Statistical mechanics analogy).* This cycle is structurally analogous to a Stirling cycle in thermodynamics. The analogy does not supply an entropy balance or efficiency bound without specified reservoirs and an entropy-production estimate.

:::
:::{prf:remark} Perception as Compression (Operational Identity)
:label: thm-perception-as-compression

During perception, the agent compresses external entropy into internal free energy:

$$
W_{\text{compress}} = T_c \cdot I(X_t; K_t) \geq 0,

$$
where $T_c$ is the cognitive temperature ({prf:ref}`def-cognitive-temperature`) and $I(X_t; K_t)$ is the mutual information extracted from the observation $X_t$ into the macro-state $K_t$.

*Mechanism:* The Visual Encoder $E_\phi$ compresses high-entropy raw data $\phi_{\text{raw}}$ into a low-entropy macro-state $z$. The "heat" absorbed is the raw sensory stream.

*Information-theoretic interpretation:* Entropy decreases ($\Delta S < 0$). The Information Bottleneck cost bounds the compression.

:::
:::{prf:remark} Action as Expansion (Operational Identity)
:label: thm-action-as-expansion

During action, the agent expands internal free energy into external control:

$$
W_{\text{expand}} = T_c \cdot I(K^{\text{act}}_t; K_t) \geq 0,

$$
where $I(K^{\text{act}}_t; K_t)$ is the mutual information injected from the intention into the motor output.

*Mechanism:* The Action Decoder $D_A$ "expands" the low-entropy Intention $u_\pi$ into high-dimensional motor commands $a_{\text{raw}}$, injecting motor texture.

*Information-theoretic interpretation:* Entropy increases ($\Delta S > 0$). The agent injects stochastic texture into motor outputs.

:::
:::{prf:remark} Dreaming as an Isolated-Limit Approximation
:label: def-dreaming-as-unitary-evolution

In the ideal isolated, zero-friction limit, one may model the dreaming phase by a Hamiltonian flow. This is an approximation; the WFR/Langevin dreaming model remains thermal unless those terms are explicitly removed:


$$
\partial_s \rho + [H_{\text{internal}}, \rho]_{\text{Poisson}} = 0,

$$
where $H_{\text{internal}}$ is the effective Hamiltonian:

$$
H_{\text{internal}}(z, p) = \frac{1}{2}\|p\|_{G^{-1}}^2 + V_{\text{critic}}(z).

$$
*Mechanism:* The agent is decoupled from the boundary (adiabatic/isolated). The Bulk evolves under Hamiltonian dynamics (BAOAB integrator with $\gamma \to 0$).

*Information-theoretic interpretation:* Isentropic ($\Delta S = 0$). Internal planning proceeds without information exchange with the environment.

:::
:::{prf:remark} Information-Cycle Efficiency Analogy
:label: prop-carnot-efficiency-bound

A Carnot-style bound does not follow from the definitions in this volume. The ratio below is an operational efficiency diagnostic; interpreting it as a thermodynamic bound requires an explicit two-reservoir model, temperatures, and an entropy-production inequality:

$$
\eta = \frac{I(K^{\text{act}}_t; K_t)}{I(X_t; K_t)} \leq 1 - \frac{T_{\text{motor}}}{T_{\text{sensor}}},

$$
where $T_{\text{sensor}}$ and $T_{\text{motor}}$ are the effective temperatures at the sensory and motor boundaries.

*Interpretation:* Perfect efficiency ($\eta = 1$) requires $T_{\text{motor}} = 0$ (deterministic motors) or $T_{\text{sensor}} \to \infty$ (infinite sensory entropy). Real systems operate at $\eta < 1$.

**Cross-references:** {ref}`sec-adaptive-thermodynamics` (Adaptive Thermodynamics), {ref}`sec-the-equivalence-theorem` (MaxEnt Control).

*Forward reference (Reward as Heat).* {ref}`sec-the-bulk-potential-screened-poisson-equation` establishes that Reward is the thermodynamic **heat input** that drives the cycle: the Boltzmann-Value Law (Axiom {prf:ref}`ax-the-boltzmann-value-law`) identifies $V(z) = E(z) - T_c S(z)$ as Gibbs Free Energy, and Theorem {prf:ref}`thm-wfr-consistency-value-creates-mass` proves that WFR dynamics materialize the agent in high-value regions ("Value Creates Mass").

:::
(sec-wfr-boundary-conditions-waking-vs-dreaming)=
## WFR Boundary Conditions: Waking vs Dreaming

:::{div} feynman-prose
Now we get to something philosophically deep: what's the difference between being awake and dreaming? In ordinary language, we might say "when you're awake, your senses are active; when you're dreaming, they're not." The model makes this distinction through declared boundary policies.

In the waking schedule, an observation trace or assimilation target is supplied, and the motor channel can prescribe a WFR flux. Calling the observation "Dirichlet" is a useful idealization; a finite-noise encoder is not necessarily an exact boundary clamp.

In the dreaming schedule, the sensory channel is cut and a reflective, zero-net-flux condition can be imposed. The motor boundary is a separate choice, and $u_\pi=0$ is a common closed-loop setting rather than the definition of dreaming. Thus the switch is an operational change from an open trace to a reflective trace, not a theorem that simply swaps PDE Dirichlet and Neumann data.
:::

The **Wasserstein-Fisher-Rao** (WFR, {prf:ref}`def-the-wfr-action`) equation from {ref}`sec-wasserstein-fisher-rao-geometry-unified-transport-on-hybrid-state-spaces` governs the belief density $\rho$. The distinction between Waking and Dreaming is rigorously defined by the **boundary condition** on $\rho$. Boundary conditions update at interaction time $t$, while internal flow evolves in computation time $s$ ({ref}`sec-the-chronology-temporal-distinctions`).

:::{prf:definition} Waking: Observation Assimilation and Motor Flux
:label: def-waking-boundary-clamping

During waking, the sensory stream supplies an observation target (or posterior) to the
bulk.  Write this target as $\rho_{\mathrm{obs}}(z,t)$.  A finite-noise encoder
does not impose an exact Dirichlet trace; its coupling is represented by the
assimilation source introduced in
{prf:ref}`def-dirichlet-boundary-condition-sensors`:

$$
S_{\mathrm{obs}}(z,t)=\kappa_{\mathrm{obs}}(z,t)\bigl(\rho_{\mathrm{obs}}(z,t)-\rho_{\mathrm{bulk}}(z,t)\bigr).

$$
In the same schedule the motor interface may prescribe a WFR flux,

$$
j_{\rho}\!\cdot\mathbf{n}=J_A(z,u_\pi,t),
$$

with $J_A$ carrying the flux units specified in
{prf:ref}`def-neumann-boundary-condition-motors`.  An exact Dirichlet clamp is recovered
only as a declared strong-assimilation limit on a genuine boundary.  The
transport/reaction balance is selected by the stated WFR action and its
parameters; there is no universal threshold at which one term must dominate.

:::
:::{prf:definition} Dreaming: Reflective Boundary
:label: def-dreaming-reflective-boundary

During dreaming, the sensory stream is cut and the sensory boundary is **Reflective**. The motor boundary may still be specified separately; $u_\pi=0$ is a common closed-loop choice, not the definition of the mode:

$$
j_{\rho}\!\cdot\mathbf{n} = 0 \quad \text{(Reflective/Neumann-zero)}.

$$
The system is closed with respect to the sensory boundary.  Total mass is
conserved only when the reaction term also integrates to zero,
$\int_{\mathcal{Z}}\rho r\,d\mu_G=0$; the motor boundary and any internal
source or sink must be specified separately.  In the closed-loop choice
$u_\pi=0$, the remaining drift is driven by the internal potential
$V_{\text{critic}}(z)$, but this is a modeling choice rather than the
definition of dreaming.

:::
:::{prf:remark} WFR Mode Switching
:label: thm-wfr-mode-switching

Changing the sensory boundary from an open observation trace to a reflective trace defines the operational waking/dreaming switch. This is a boundary-condition change, not a thermodynamic phase-transition theorem:

| Mode         | Sensory coupling                         | Motor coupling                    | Internal Flow | Information Balance       |
|--------------|-------------------------------------------|-----------------------------------|---------------|---------------------------|
| **Waking**   | Assimilation source $S_{\mathrm{obs}}$     | Prescribed flux $J_A$             | Source-driven | Depends on supplied trace  |
| **Dreaming** | Reflective $j_\rho\!\cdot n=0$           | Reflective or prescribed         | Recirculating | Zero sensory flux          |
| **Pure actuation** | No sensory source                    | Prescribed flux $J_A$             | Motor-driven  | Net outward flux allowed   |

:::
:::{prf:proposition} Net Interface Flux and Grounding Rate
:label: prop-grounding-rate-via-boundary-flux

The signed net interface flux can be recorded as the operational diagnostic
$\Phi_{\mathrm{net}}(t)$:

$$
\Phi_{\mathrm{net}}(t)=
\oint_{\partial\mathcal{Z}_{\mathrm{sense}}}j_{\mathrm{obs}}\!\cdot dA
-\oint_{\partial\mathcal{Z}_{\mathrm{motor}}}j_{\mathrm{motor}}\!\cdot dA.

$$

This signed flux is distinct from the upstream grounding rate
$G_t:=I(X_t;K_t)$ and its realised rate
$\lambda_{\mathrm{in}}=\mathbb{E}[G_t]$ in
{prf:ref}`def-grounding-rate`.  A relation between expected sensory flux and
$\lambda_{\mathrm{in}}$ is an additional interface calibration assumption.
The sign of $\Phi_{\mathrm{net}}$ records the chosen boundary convention: it
can be positive in an observation-driven schedule, zero for a closed sensory
boundary, or negative when a prescribed motor flux dominates.

**Cross-references:** {ref}`sec-the-wfr-metric` (WFR Action), {ref}`sec-the-unified-world-model` (WFR World Model), {ref}`sec-the-interface-and-observation-inflow` (Observation Inflow).

:::
(sec-the-context-space-unified-definition)=
## The Context Space: Unified Definition

:::{div} feynman-prose
Now I want to show you something that I find really beautiful---a unification that becomes useful once the interface maps are made explicit.

What do these three things have in common?
- A robot deciding which direction to push a lever
- A classifier deciding whether an image shows a cat or a dog
- A language model deciding which word comes next given a prompt

On the surface, they seem totally different. Actions, labels, tokens---different domains, different vocabularies, different applications. The useful common pattern is that each supplies a context or conditioning signal at an interface.

The robot's action, classifier's label, and language prompt generally live in different spaces and use different maps into the model. They can be represented as context-dependent boundary data only after those maps, output spaces, and costs are specified. A prompt is not automatically a motor flux, and a label is not automatically a physical clamp.

We call the space of such conditioning signals the *Context Space* $\mathcal{C}$. The same bulk equations can be reused when the task-specific encoders, action-dependent costs, and boundary lift are compatible. The claim of a shared architecture is therefore an operational analogy, not a blanket equivalence of robotics, classification, and language.
:::

The Action Atlas admits a deeper structure: the **Context Space** $\mathcal{C}$ is the abstract space of boundary conditions that unifies RL actions, classification labels, and LLM prompts.

:::{prf:definition} Context Space
:label: def-context-space

The **Context Space** $\mathcal{C}$ is a manifold parameterizing the control/conditioning signal for the agent:

$$
\mathcal{C} := \{c : c \text{ specifies a boundary condition on } \partial\mathcal{Z}\}.

$$
The context determines the target distribution at the motor boundary via an action-dependent effective cost:

$$
\pi(a | z, c) \propto \exp\left(-\frac{1}{T_c} \mathcal{C}_{\text{eff}}(z,a,c)\right),

$$
Units: $[\mathcal{C}]$ inherits from the task domain.

:::
:::{prf:definition} Context Instantiation Maps
:label: def-context-instantiation-functor

For each task domain, a typed instantiation map sends a task-specific
conditioning signal into $\mathcal C$.  Calling these maps a functor would
require category structures that are not part of this volume.  The three
canonical instantiations are:

| Task Domain        | Context $c \in \mathcal{C}$ | Motor Output $a$           | Action-dependent cost $\mathcal{C}_{\text{eff}}$      |
|--------------------|-----------------------------|----------------------------|----------------------------------------------|
| **RL**             | Task/context space $\mathcal{C}_{\mathrm{RL}}$ | Motor command $a$ | $Q_{\mathrm{cost}}(z,a,c)$ |
| **Classification** | Label space $\mathcal{Y}$   | Class prediction $a$ | $-\log p(a\mid z,c)$ (cross-entropy) |
| **LLM**            | Prompt space $\mathcal{P}$  | Token $a$ | $-\log p(a\mid z,c)$ (next-token loss) |

*Key Insight:* In all cases, the context $c$ functions as the **symmetry-breaking boundary condition** that determines which direction the holographic expansion takes at the origin.

:::
:::{prf:remark} Conditional Context Structure
:label: thm-universal-context-structure

When the task-specific encoder, action-dependent cost, and boundary lift are explicitly identified, the context instantiations share the following schematic structure:

1. **Embedding:** $c \mapsto e_c \in \mathbb{R}^{d_c}$ maps the context to a latent vector.
2. **Typed lift:** a declared map $\iota_c:\mathbb{R}^{d_c}\to T_0\mathcal Z$
   sends that embedding into the tangent space at the reference state.
3. **Symmetry-Breaking Kick:** $\iota_c(e_c)$ determines the initial control field:

   $$
   u_\pi(0) = G^{-1}(0)\,\iota_c(e_c) = \frac{1}{4}\,\iota_c(e_c)

   $$
   (at the Poincare disk origin where $G(0) = 4I$)
4. **Motor Distribution:** The output distribution is:

   $$
   \pi(a | z, c) = \text{softmax}_a\left(-\frac{\mathcal{C}_{\text{eff}}(z,a,c)}{T_c}\right)

   $$
*Scope.* The shared wiring is an architectural analogy. Equivalence of the three task domains requires separate task-specific encoders, action spaces, and costs; it is not implied by the geometric notation alone.

:::
:::{prf:definition} Context-Conditioned WFR
:label: def-context-conditioned-wfr

The WFR dynamics ({ref}`sec-the-wfr-metric`) generalize to context-conditioned form:

$$
\partial_s \rho + \nabla \cdot (\rho\, v_c) = \rho\, r_c,

$$
where:
- $v_c(z) = -G^{-1}(z) \nabla_z \mathcal{C}_{\text{eff}}(z,a,c) + u_\pi(z, c)$ is the context-conditioned velocity, after the action has been selected
- $r_c(z)$ is the context-conditioned reaction rate (chart jumps influenced by context)

:::
:::{prf:remark} Shared Context Interface (Conditional)
:label: cor-prompt-action-label

RL actions, classification labels, and LLM prompts can share an interface
factorization when their task-specific maps are supplied:

$$
 c \longmapsto e_c \longmapsto \iota_c(e_c) \longmapsto \pi(\cdot\mid z,c).

$$
The domains are not isomorphic by notation alone.  Each requires its own
context encoder, action/output space, cost, and boundary lift.  A chart route,
continuous target, or texture channel can be shared only when those maps are
explicitly identified and dimensionally compatible.

**Cross-references:** {ref}`sec-policy-control-field` (Control Field), Theorem {prf:ref}`thm-unified-control-interpretation`, Definition {prf:ref}`def-effective-potential`.

*Forward reference (Effective Potential Resolution).* {ref}`sec-the-bulk-potential-screened-poisson-equation`
resolves the scalar state-cost part of $\Phi_{\text{eff}}$: in the conservative
subcase the critic solves the **Screened Poisson Equation** with the cost
source convention used by the control loop.  An action-dependent context cost
$Q_{\mathrm{cost}}(z,a,c)$ is a separate policy object and is not identified
with the state potential for arbitrary tasks.  The discount factor $\gamma$
 determines the stationary-diffusion screening length
$\ell_{\mathrm{diff}}=\sqrt{T_c\Delta t/(-\ln\gamma)}$ (natural units: $\sqrt{T_c/(-\ln\gamma)}$)
(Corollary {prf:ref}`cor-discount-as-screening-length`),
explaining why distant rewards are exponentially suppressed in policy.

:::
(sec-implementation-the-holographicinterface-module)=
## Implementation: The HolographicInterface Module

:::{div} feynman-prose
All right, enough theory. Let's inspect the implementation.

The code below is a reference architecture for the atlas interfaces, motor texture decomposition, and context-conditioned policy. Read the tensor shapes and actual code paths as part of the specification; a parallel API is not a proof of Legendre duality.

A few things to notice as you read through:
1. The Visual and Action modules have parallel structure---a design symmetry that can support the conditional lifted correspondence
2. Motor texture uses the declared geometry-dependent variance schedule, whose interpretation depends on the chosen metric and sampler
3. The policy accepts context through the task-specific maps; handling RL actions, labels, or prompts requires compatible output heads and costs

The implementation makes the interface idea testable. It does not, by itself, establish the analytic boundary or equivalence claims above.
:::

We provide the Python implementation of the Holographic Interface, combining the Dual Atlas, Motor Texture, and Context Space.

**Algorithm 23.7.1 (HolographicInterface Module).**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Literal, Optional, Dict, Tuple
from enum import Enum


class BoundaryConditionType(Enum):
    """Operational boundary-coupling types used by the interface."""
    DIRICHLET = "dirichlet"    # Position clamping (sensors)
    NEUMANN = "neumann"        # Flux clamping (motors)
    REFLECTIVE = "reflective"  # Dreaming mode (zero flux)


class ContextType(Enum):
    """Task-specific context instantiation types."""
    RL = "rl"                        # Action space
    CLASSIFICATION = "classification"  # Label space
    LLM = "llm"                      # Prompt space


@dataclass
class InterfaceConfig:
    """Configuration for HolographicInterface."""
    obs_dim: int = 64
    action_dim: int = 8
    latent_dim: int = 32
    hidden_dim: int = 256
    num_visual_charts: int = 8
    num_action_charts: int = 4
    codes_per_chart: int = 64
    context_dim: int = 64
    sigma_motor: float = 0.1
    T_c: float = 1.0


class DualAtlasEncoder(nn.Module):
    """
    Symmetric encoder for the visual and action atlases.

    Extends AttentiveAtlasEncoder ({ref}`sec-tier-the-attentive-atlas`) with unified interface.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        num_charts: int,
        codes_per_chart: int,
        atlas_type: Literal["visual", "action"],
    ):
        super().__init__()
        self.atlas_type = atlas_type
        self.num_charts = num_charts
        self.latent_dim = latent_dim

        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # Cross-attention routing ({ref}`sec-tier-the-attentive-atlas`)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.chart_queries = nn.Parameter(torch.randn(num_charts, hidden_dim) * 0.02)
        self.scale = hidden_dim ** 0.5

        # Per-chart codebooks
        self.codebooks = nn.ModuleList([
            nn.Embedding(codes_per_chart, latent_dim)
            for _ in range(num_charts)
        ])

        # Residual decomposition
        self.nuisance_head = nn.Linear(hidden_dim, latent_dim)
        self.texture_head = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Encode input to (macro, nuisance, texture) triple."""
        B = x.shape[0]

        # Feature extraction
        h = self.feature_extractor(x)

        # Cross-attention routing
        k = self.key_proj(h)  # [B, H]
        attn = torch.einsum('bh,ch->bc', k, self.chart_queries) / self.scale
        chart_probs = F.softmax(attn, dim=-1)  # [B, C]
        chart_idx = chart_probs.argmax(dim=-1)  # [B]

        # Residual decomposition
        z_nuisance = self.nuisance_head(h)
        z_texture = self.texture_head(h)

        # Select the nearest code in the routed chart.  The old implementation
        # always selected code zero, so the macro state carried no information
        # about the input.  The straight-through form keeps the codebook value
        # in the forward pass while allowing the pre-quantized representation
        # to receive gradients.
        codebook_weights = torch.stack(
            [codebook.weight for codebook in self.codebooks], dim=0
        )  # [C, codes_per_chart, D]
        selected_weights = codebook_weights[chart_idx]  # [B, codes_per_chart, D]
        distances = (selected_weights - z_nuisance[:, None, :]).square().sum(dim=-1)
        code_idx = distances.argmin(dim=-1)
        batch_idx = torch.arange(B, device=x.device)
        z_quantized = selected_weights[batch_idx, code_idx]
        z_macro = z_nuisance + (z_quantized - z_nuisance).detach()

        return {
            'chart_idx': chart_idx,
            'code_idx': code_idx,
            'chart_probs': chart_probs,
            'z_macro': z_macro,
            'z_nuisance': z_nuisance,
            'z_texture': z_texture,
        }


def sample_motor_texture(
    z: torch.Tensor,
    d_motor_tex: int,
    sigma_motor: float,
) -> torch.Tensor:
    """
    Sample motor texture with conformal scaling.

    Sigma_motor(z) = sigma^2 * G^{-1}(z) = sigma^2 * (1-|z|^2)^2 / 4
    """
    B = z.shape[0]
    device = z.device

    # Conformal factor at z (Poincare disk)
    r_sq = (z ** 2).sum(dim=-1, keepdim=True)
    G_inv_scale = (1.0 - r_sq.clamp(max=0.99)) ** 2 / 4.0

    # Sample with geometry-dependent variance
    xi = torch.randn(B, d_motor_tex, device=device)
    z_tex_motor = sigma_motor * torch.sqrt(G_inv_scale) * xi

    return z_tex_motor


class ContextConditionedPolicy(nn.Module):
    """
    Context-conditioned policy for unified task handling.

    Unifies RL actions, classification labels, and LLM tokens.
    """

    def __init__(
        self,
        latent_dim: int,
        context_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        state_dim: Optional[int] = None,
    ):
        super().__init__()
        self.state_dim = state_dim if state_dim is not None else latent_dim

        # Context embedding
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(self.state_dim + latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(
        self,
        z: torch.Tensor,
        context: torch.Tensor,
        T_c: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute context-conditioned action distribution.

        pi(a|z, c) = softmax_a(-C_eff(z, a, c) / T_c)
        """
        # Embed context
        c_embed = self.context_encoder(context)

        if z.shape[-1] != self.state_dim:
            raise ValueError(
                f"Expected state dimension {self.state_dim}, got {z.shape[-1]}"
            )

        # Concatenate state and context
        z_c = torch.cat([z, c_embed], dim=-1)

        # Compute logits (negative effective potential)
        logits = self.policy_net(z_c)

        # Softmax with temperature
        probs = F.softmax(logits / T_c, dim=-1)

        return {
            'logits': logits,
            'probs': probs,
            'context_embedding': c_embed,
        }


class HolographicInterface(nn.Module):
    """
    {ref}`sec-the-boundary-interface-symplectic-structure`: The Holographic Interface.

    Provides an optional phase-space lift of the agent/environment interface.
    Combines:
    - Dual Atlas (Visual + Action)
    - Motor Texture sampling
    - Context-conditioned policy
    - Thermodynamic cycle tracking

    Cross-references:
    - {ref}`sec-wasserstein-fisher-rao-geometry-unified-transport-on-hybrid-state-spaces` (WFR Geometry)
    - {ref}`sec-radial-generation-entropic-drift-and-policy-control` (Holographic Generation {cite}`thooft1993holographic,susskind1995world`)
    - {ref}`sec-the-equations-of-motion-geodesic-jump-diffusion` (Geodesic SDE)
    """

    def __init__(self, config: InterfaceConfig):
        super().__init__()
        self.config = config

        # Visual atlas
        self.visual_atlas = DualAtlasEncoder(
            config.obs_dim, config.hidden_dim, config.latent_dim,
            config.num_visual_charts, config.codes_per_chart, "visual"
        )

        # Action atlas
        self.action_atlas = DualAtlasEncoder(
            config.action_dim, config.hidden_dim, config.latent_dim,
            config.num_action_charts, config.codes_per_chart, "action"
        )

        # Context-conditioned policy
        self.policy = ContextConditionedPolicy(
            config.latent_dim, config.context_dim,
            config.action_dim, config.hidden_dim,
            state_dim=2 * config.latent_dim,
        )
        self.intent_head = nn.Linear(config.action_dim, config.latent_dim)

        # Action decoder (tangent bundle decoder)
        self.action_decoder = nn.Sequential(
            nn.Linear(config.latent_dim * 2, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.action_dim),
        )

    def forward_perception(
        self,
        x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Phase I: Compression (Environment -> Bulk).
        Information interpretation: an operational compression step; no entropy balance is implied.
        Applies the declared observation assimilation coupling.
        """
        return self.visual_atlas(x)

    def forward_actuation(
        self,
        z: torch.Tensor,
        u_intent: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Phase III: Expansion (Bulk -> Environment).
        Information interpretation: an operational expansion step; no entropy balance is implied.
        Applies the declared motor-flux coupling.
        """
        B = z.shape[0]

        # Sample motor texture with the declared conformal scaling.
        z_tex_motor = sample_motor_texture(
            z, self.config.action_dim, self.config.sigma_motor
        )

        # Decode intention to action
        z_u = torch.cat([z, u_intent], dim=-1)
        a_base = self.action_decoder(z_u)

        # Add motor texture
        a_raw = a_base + z_tex_motor

        return {
            'action': a_raw,
            'action_base': a_base,
            'motor_texture': z_tex_motor,
        }

    def forward_proprioception(
        self,
        a_realized: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Inverse model: Map realized actions back to latent intentions.
        Used to calculate execution error.
        """
        return self.action_atlas(a_realized)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        mode: Literal["waking", "dreaming"] = "waking",
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass through holographic interface.

        Args:
            x: Observation [B, obs_dim]
            context: Context conditioning [B, context_dim]
            mode: "waking" (boundary clamped) or "dreaming" (reflective)
        """
        # Phase I: Perception (compression)
        vis_out = self.forward_perception(x)
        z_state = torch.cat([vis_out['z_macro'], vis_out['z_nuisance']], dim=-1)

        # Get context-conditioned policy
        policy_out = self.policy(z_state, context, self.config.T_c)

        if mode == "dreaming":
            # Reflective boundary: no actuation
            return {
                'visual': vis_out,
                'policy': policy_out,
                'mode': mode,
            }

        # Phase III: Action (expansion)
        u_intent = self.intent_head(policy_out['probs'])
        act_out = self.forward_actuation(
            vis_out['z_nuisance'], u_intent, context
        )

        return {
            'visual': vis_out,
            'policy': policy_out,
            'action': act_out,
            'mode': mode,
        }
```

**Cross-references:** {ref}`sec-tier-the-attentive-atlas` (AttentiveAtlasEncoder), {ref}`sec-decoder-architecture-overview-topological-decoder` (TopologicalDecoder), {prf:ref}`def-baoab-splitting` (BAOAB).

::::{admonition} Connection to RL #7: Dreamer/World Models as Generic RNN Dynamics
:class: note
:name: conn-rl-7
**The General Law (Fragile Agent):**
The HolographicInterface can use BAOAB as a splitting on the lifted phase
space $(T^*\mathcal{Z},\omega)$.  Its deterministic Hamiltonian B-A-B part
is symplectic under the stated smoothness and exact-flow hypotheses; the O
step is an Ornstein--Uhlenbeck thermostat and is stochastic when $\gamma>0$.
Thus the full thermostatted update is not a symplectic map.

$$
\hat{y}_{t+1}=\Phi_{\mathrm{BAOAB}}(y_t),\qquad y=(z,p),
\qquad \Phi_{\mathrm{det}}^*\omega=\omega
\quad (\gamma=0).

$$
The dual atlas structure ({prf:ref}`def-visual-atlas-perception` and
{prf:ref}`def-action-atlas-actuation`) decomposes latent space into Visual and
Action atlases.  Their boundary couplings are matched only when the declared
lift and interface maps are compatible.

**The Degenerate Limit:**
Remove symplectic structure: replace $\Phi_{\text{BAOAB}}$ with generic RNN/GRU. Ignore boundary condition matching.

**The Special Case (Standard RL - Dreamer, MuZero):**
World-model RL uses generic neural network dynamics:

$$
z_{t+1} = f_\theta(z_t, a_t), \quad \hat{r}_t = r_\theta(z_t), \quad \hat{\gamma}_t = \gamma_\theta(z_t).

$$
The RNN/GRU/Transformer architecture has no geometric constraints—it's a universal function approximator.

**Result:** Dreamer/MuZero/PlaNet use unconstrained learned dynamics unless a
separate geometric integrator is imposed.  They are not a literal $\omega\to0$
limit of the stochastic BAOAB scheme.

**What the generalization offers:**
- **Deterministic structure**: The Hamiltonian substep preserves phase-space volume in the exact deterministic limit.
- **Long-horizon stability**: Backward-error bounds apply only under the regularity and step-size hypotheses of the chosen symplectic splitting.
- **Interpretable rollouts**: Latent trajectories follow geodesics modified by potential forces
- **Boundary semantics**: assimilation and flux couplings distinguish observation from action interfaces ({prf:ref}`def-dirichlet-boundary-condition-sensors`, {prf:ref}`def-neumann-boundary-condition-motors`)
::::

(sec-summary-tables-and-diagnostic-nodes-a)=
## Summary Tables and Diagnostic Nodes

**Summary of Holographic Interface:**

| Component              | Visual (Perception)           | Motor (Action)                  |
|------------------------|-------------------------------|---------------------------------|
| **Boundary Coupling**  | Observation assimilation target | Prescribed WFR motor flux $J_A$ |
| **Atlas**              | $\mathcal{A}_{\text{vis}}$    | $\mathcal{A}_{\text{act}}$      |
| **Macro**              | Chart index $K$               | Action primitive $K^{\text{act}}$ |
| **Nuisance**           | Pose/viewpoint $z_n$          | Compliance $z_{n,\text{motor}}$ |
| **Texture**            | Pixel detail $z_{\text{tex}}$ | Tremor $z_{\text{tex,motor}}$   |
| **Thermodynamics**     | Compression is an operational information reduction | Expansion is an operational information release |

**Context Space Instantiation:**

| Task           | Context $c$  | Output          | Action/output cost                          |
|----------------|--------------|-----------------|---------------------------------------------|
| RL             | Task context | Motor command $a$ | $Q_{\mathrm{cost}}(z,a,c)$                |
| Classification | Label space  | Class $a$       | $-\log p(a\mid z,c)$                       |
| LLM            | Prompt space | Token $a$       | $-\log p(a\mid z,c)$                       |

(node-30)=
**Node 30: SymplecticBoundaryCheck**

| **#**  | **Name**                    | **Component** | **Type**           | **Interpretation**               | **Proxy**                                                | **Cost** |
|--------|-----------------------------|---------------|--------------------|----------------------------------|----------------------------------------------------------|----------|
| **30** | **SymplecticBoundaryCheck** | **Interface** | **Lift consistency** | Is the paired phase-space update compatible with the declared symplectic form? | $\lVert J^{\mathsf T}\Omega J-\Omega\rVert_F$ for the Jacobian $J$ of $(z,p)\mapsto(z',p')$ | $O(Bd^2)$  |

**Trigger conditions:**
- High SymplecticBoundaryCheck: the lifted update fails the chosen symplectic-area test.
- Remedy: inspect the phase-space lift and integrator substeps; boundary flux compatibility alone does not imply symplecticity.

(node-31)=
**Node 31: DualAtlasConsistencyCheck**

| **#**  | **Name**                      | **Component** | **Type**          | **Interpretation**                     | **Proxy**                                                                  | **Cost**  |
|--------|-------------------------------|---------------|-------------------|----------------------------------------|----------------------------------------------------------------------------|-----------|
| **31** | **DualAtlasConsistencyCheck** | **Encoder**   | **Atlas alignment** | Are paired Visual and Action charts aligned under the learned lift? | $\lVert L_\theta(e_\alpha^{\mathrm{vis}})-e_{\beta(\alpha)}^{\mathrm{act}}\rVert^2$ | $O(BK^2)$ |

**Trigger conditions:**
- High DualAtlasConsistencyCheck: Visual and Action atlases have drifted apart.
- Remedy: Increase Legendre alignment loss; verify codebook coupling; check chart transition consistency.

(node-32)=
**Node 32: MotorTextureCheck**

| **#**  | **Name**              | **Component** | **Type**           | **Interpretation**                       | **Proxy**                                                  | **Cost**               |
|--------|-----------------------|---------------|--------------------|------------------------------------------|------------------------------------------------------------|------------------------|
| **32** | **MotorTextureCheck** | **Policy**    | **Motor Firewall** | Is motor texture decoupled from control? | $\lVert\partial_{z_{\text{tex,motor}}} \pi(a\mid z)\rVert$ | $O(Bd_{\text{motor}})$ |

**Trigger conditions:**
- High MotorTextureCheck: Motor texture is leaking into control decisions (firewall violated).
- Remedy: Increase motor texture firewall penalty; verify motor residual decomposition; check Axiom {prf:ref}`ax-motor-texture-firewall`.

(node-33)=
**Node 33: ThermoCycleCheck**

| **#**  | **Name**             | **Component**   | **Type**           | **Interpretation**                               | **Proxy**                                              | **Cost** |
|--------|----------------------|-----------------|--------------------|--------------------------------------------------|--------------------------------------------------------|----------|
| **33** | **ThermoCycleCheck** | **World Model** | **Efficiency target** | Does the measured information-cycle efficiency stay within its declared target band? | $\lvert\eta_{\mathrm{info}}-\eta_{\mathrm{target}}\rvert$ | $O(B)$   |

**Trigger conditions:**
- High ThermoCycleCheck: the measured ratio has left the declared diagnostic band.
- Remedy: recalibrate the information-flow measurements and boundary coupling; report the operational efficiency together with its target band.

(node-34)=
**Node 34: ContextGroundingCheck**

| **#**  | **Name**                  | **Component** | **Type**             | **Interpretation**                          | **Proxy**                 | **Cost** |
|--------|---------------------------|---------------|----------------------|---------------------------------------------|---------------------------|----------|
| **34** | **ContextGroundingCheck** | **Policy**    | **Context Validity** | Is context properly grounding motor output? | $I(K^{\text{act}}_t; c) / I(X_t; K_t)$ | $O(B)$   |

**Trigger conditions:**
- Low ContextGroundingCheck: Context is not influencing motor output (ungrounded generation).
- Remedy: Increase context embedding strength; verify context-conditioned potential; check symmetry-breaking kick.
