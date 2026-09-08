(sec-the-equations-of-motion-geodesic-jump-diffusion)=
# The Equations of Motion: Geodesic Jump-Diffusion

## TLDR

- Once geometry is fixed, dynamics “almost write themselves”: the agent follows **controlled geodesic motion** with
  stochastic exploration noise.
- Continuous motion within charts + discrete chart transitions yields a **jump-diffusion** equation of motion.
- The metric plays the role of **mass/preconditioner**: curvature slows or speeds updates depending on capacity and risk.
- This chapter is the bridge from geometric laws (metric/WFR) to implementable update rules and diagnostics.
- Use the diagnostic suite (geodesic consistency, jump consistency, boundary-reached checks) to validate dynamics.

## Roadmap

1. Continuous-time limit: geodesic/Langevin-style motion on $(\mathcal{Z},G)$.
2. Discrete events: jump rates and chart transitions.
3. Summary tables and diagnostics for implementation validation.

{cite}`oksendal2003sde,risken1996fokkerplanck`

(rb-continuous-actor-critic)=
:::{admonition} Researcher Bridge: Continuous-Time Actor-Critic
:class: info
The equations of motion are the continuous-time limit of policy updates with stochastic exploration noise. Think of it as a Langevinized actor-critic where the metric defines the preconditioner.
:::

:::{div} feynman-prose
Now, here is where everything comes together. We have built up all this machinery in the previous sections---the geometry, the metric law, the WFR transport---and you might be wondering: what does the agent actually *do*? How does it move?

This is the chapter where we write down the answer. Once the geometry, the potential, the control convention, and the noise law have been specified, the equations of motion can be written down and checked against those choices. The geometry constrains the form; it does not choose every modeling term for us.

The useful picture is **controlled geodesic Langevin motion on a curved manifold, with jumps**. Between jumps, the agent carries a position and a covector momentum. The connection term accounts for the changing coordinates of a geodesic, while friction, control, curl, and noise alter that free motion. Occasionally---this is the jump part---it can hop from one chart to another when the declared rate law proposes a new representation.

If you have ever studied classical mechanics, this should feel familiar. We use Hamiltonian and Langevin ideas in a stochastic setting on a curved space. The curvature comes from the metric law of Section 18. The noise comes from the chosen exploration-exploitation convention. And the jumps come from the discrete structure of the chart atlas.

Let me emphasize a convention that will matter in the signs below: **the inertial mass tensor is chosen to be the metric**. In ordinary mechanics, mass tells you how momentum is converted into velocity. Here, $G$ does that job through $v^i=G^{ij}p_j$. A large metric can suppress coordinate response to a fixed covector force, but the resulting caution is a consequence of this chosen update law, not a universal statement about every discretization.

When the Metric Law makes $G$ large in a high-risk region, this convention can therefore reduce coordinate step sizes. Keep the qualifier in mind: the geometry supplies the metric, while the mass identification and the control/noise schedules specify how the agent uses it.
:::

We derive the rigorous equation of motion (EoM) for the agent. This equation unifies the WFR geometry ({ref}`Section 20 <sec-wasserstein-fisher-rao-geometry-unified-transport-on-hybrid-state-spaces>`, {prf:ref}`def-the-wfr-action`), the Metric Law ({ref}`Section 18 <sec-capacity-constrained-metric-law-geometry-from-interface-limits>`, Theorem {prf:ref}`thm-capacity-constrained-metric-law`), and the Policy-driven Expansion ({ref}`Section 21 <sec-radial-generation-entropic-drift-and-policy-control>`).

(sec-the-stochastic-action-principle)=
## The Stochastic Action Principle (Mass = Metric)

:::{div} feynman-prose
Before we get into the formalism, let me tell you what we are really doing here. In classical mechanics, an action can generate equations through a variational principle. Here we introduce a related path functional as an operational modeling objective. It is useful for organizing kinetic, potential, curvature, and entropy terms, but its status has to be stated precisely.

It is tempting to call this functional Onsager--Machlup and read its minimizer as a most-probable path. That inference is not available here. A genuine Onsager--Machlup functional depends on the particular drift, diffusion, reference measure, and path-tube convention. Those ingredients are not supplied by merely writing the operational functional below.

Here is the picture I want you to have in your mind. Imagine a particle moving in a potential while a controller and a thermostat act on it. The displayed integral assigns a score to a proposed trajectory: fast motion costs kinetic energy, unfavorable regions cost potential energy, and the selected curvature and policy-entropy terms modify that score. It is a design objective, not by itself a probability density over paths.

The analogy with stochastic mechanics is still helpful. On a curved space, a path measure can acquire geometry-dependent terms, but their coefficient depends on how the diffusion and path measure are defined. The $T_cR/12$ term here is the declared modeling correction; it should not be presented as a universal path-probability formula.

Now, the key decision we have to make: what plays the role of mass? In ordinary mechanics, mass appears in the kinetic energy term, $\frac{1}{2}mv^2$. On a Riemannian manifold, the natural generalization is $\frac{1}{2}G_{ij}\dot{z}^i\dot{z}^j$---the metric-weighted norm of the velocity.

So here is our modeling choice: **Mass = Metric**. The metric already measures tangent vectors, and we use the same tensor to define kinetic cost and the momentum-to-velocity map. Geometry motivates this identification; it does not force it without that convention.
:::

Stochastic systems also admit **Onsager--Machlup functionals**, but their form depends on the drift, diffusion, reference measure, and path-tube convention. The operational path objective below is kept separate from that probabilistic construction.

:::{prf:definition} Mass Tensor
:label: def-mass-tensor

We define the **inertial mass tensor** $\mathbf{M}(z)$ as the capacity-constrained metric:

$$
\mathbf{M}(z) := G(z).

$$
This definition has the following operational consequences:
- **High curvature regions** (large $G$) have larger effective mass, yielding smaller velocity updates per unit force
- **Low curvature regions** (small $G$) have smaller effective mass, yielding larger velocity updates per unit force

Units: $[\mathbf{M}_{ij}] = [z]^{-2}$ (same as metric).

*Remark (Risk-Metric Coupling).* Combined with the Metric Law (Theorem {prf:ref}`thm-capacity-constrained-metric-law`), this yields a causal chain:

$$
\text{High risk } T_{ij} \;\Rightarrow\; \text{Large } G_{ij} \;\Rightarrow\; \text{Large } \mathbf{M}_{ij} \;\Rightarrow\; \text{Reduced step size}

$$
The metric-weighted step size decreases in high-curvature (high-risk) regions without explicit penalty terms.

:::

:::{admonition} The Causal Chain of Caution
:class: feynman-added tip

Under the stated Metric Law and the chosen Mass = Metric update convention, there is a useful feedback loop:

1. **High risk** in a region $\Rightarrow$ Metric Law says curvature increases $\Rightarrow$ metric $G$ grows
2. **Large metric** $\Rightarrow$ Mass = Metric says effective mass increases
3. **Large metric** $\Rightarrow$ the same covector force can produce a smaller coordinate velocity or step

This gives a geometric caution signal. It does not replace an explicit safety constraint when one is required, and the conclusion depends on the integrator and control law using the metric as specified.
:::

:::{prf:definition} Free-energy Path Action (Operational)
:label: def-extended-onsager-machlup-action

Let $(\mathcal{Z}, G)$ be the latent Riemannian manifold with the capacity-constrained metric ({ref}`Section 18 <sec-capacity-constrained-metric-law-geometry-from-interface-limits>`). For a path $z: [0, T] \to \mathcal{Z}$, define the following free-energy path functional:

$$
S_{\mathrm{path}}[z] = \int_0^T \left( \frac{1}{2}\mathbf{M}(z)\|\dot{z}\|^2 + \Phi_{\text{eff}}(z) + \frac{T_c}{12}\,R(z) + T_c \cdot H_{\pi}(z) \right) ds,

$$
where:
- $\mathbf{M}(z)\|\dot{z}\|^2 = G_{ij}(z)\,\dot{z}^i\,\dot{z}^j$ is the kinetic energy (mass = metric)
- $\Phi_{\text{eff}}(z)$ is the effective potential (Definition {prf:ref}`def-effective-potential`)
- $R(z)$ is the scalar curvature of the metric $G$
- $H_{\pi}(z) = -\mathbb{E}_{a \sim \pi}[\log \pi(a|z)]$ is the policy entropy
- $T_c > 0$ is the {prf:ref}`def-cognitive-temperature` (cf. {ref}`Section 21.2 <sec-policy-control-field>`)

This is an operational modeling objective, not the Onsager--Machlup functional of the diffusion below; no most-probable-path claim follows from this definition. A genuine Onsager--Machlup functional also requires the drift, reference measure, and path-tube convention.

*Units.* The displayed expression uses dimensionless computational time and normalized latent coordinates. If a physical time scale is restored, all coefficients are rescaled together; from $dz^k=G^{kj}p_j\,ds$ and $[G]=[z]^{-2}$, the covector momentum has units $[p]=[z]^{-1}\,\mathrm{time}^{-1}$.

*Remark (Curvature Correction).* The term $\frac{T_c}{12}R(z)$ is a stochastic correction that accounts for the path-measure distortion on curved spaces. In flat space ($R = 0$), this term vanishes. The entropy term $T_c H_{\pi}$ ensures the agent prefers stochastic policies in uncertain regions.

:::

:::{div} feynman-prose
Let me decode this action functional term by term, because each piece is doing something important:

**The kinetic term** $\frac{1}{2}G_{ij}\dot{z}^i\dot{z}^j$: This is "how fast am I moving?" in the curved geometry. Not Euclidean speed, but speed measured with the metric. If the metric is large, the same coordinate velocity costs more action.

**The potential term** $\Phi_{\text{eff}}$: This is the scalar landscape that supplies the conservative force. We will unpack it later: it combines the hyperbolic expansion drive with the critic cost and a risk penalty. With the convention used here, $V_{\text{critic}}$ is a cost-to-go, so lower values are better. Its positive contribution to $\Phi_{\text{eff}}$ therefore gives the descent $-G^{-1}dV_{\text{critic}}$ in the conservative control limit.

**The curvature correction** $\frac{T_c}{12}R$: This is a declared geometry-dependent correction in the operational objective. Curved-space path measures can produce curvature terms, but the coefficient is tied to the diffusion and measure convention. Here it vanishes in flat space; it is not, by itself, a derivation of a path probability.

**The entropy term** $T_c H_\pi$: This is intended to encode an exploration trade-off. The sign matters: in a path score that is minimized, the displayed positive term raises the score for high-entropy policies. Calling it a bonus requires the corresponding maximization convention or a sign change in the objective. The temperature $T_c$ controls the size of that declared term.

Now notice what this functional does and does not say. Every proposed path receives a score: the kinetic term penalizes going fast, the potential term penalizes high cost or potential, and the entropy term contributes with the displayed sign. One may optimize this score as part of the model. To say that it is a most-probable path or that paths are sampled with an exponential weight, however, would require a separate Onsager--Machlup derivation for the specified diffusion.
:::

(pi-onsager-machlup)=
::::{admonition} Physics Analogy: Operational Path Objective
:class: note

**In Physics:** An Onsager--Machlup functional can assign relative weight to paths after a drift, diffusion, reference measure, and tube convention have been fixed {cite}`onsager1953fluctuations`. Those choices are not made by the operational objective in this chapter.

**In Implementation:** The free-energy path objective (Definition {prf:ref}`def-extended-onsager-machlup-action`) is:

$$
S_{\text{OM}}[z] = \int_0^T \left(\frac{1}{2}G_{ij}\dot{z}^i\dot{z}^j + \Phi_{\text{eff}} + \frac{T_c}{12}R + T_c H_\pi\right)ds

$$
It is a declared score for comparing candidate paths. It is not a path probability and its minimizer is not identified with a most-probable diffusion path.

**Correspondence Table:**

| Statistical Mechanics | Agent (Path Integral) |
|:----------------------|:----------------------|
| Temperature $k_B T$ | Cognitive temperature $T_c$ |
| Kinetic energy $\frac{1}{2}m\lvert\dot{x}\rvert^2$ | $\frac{1}{2}G_{ij}\dot{z}^i\dot{z}^j$ (mass = metric) |
| Potential $U(x)$ | Effective potential $\Phi_{\text{eff}}$ |
| Curvature correction $\frac{k_BT}{12}R$ | $\frac{T_c}{12}R$ |
| Boltzmann weight $e^{-S/k_BT}$ | Optional path weighting only after an OM derivation |
::::

:::{div} feynman-prose
Now here is a wonderful consistency check. What happens near the boundary of the Poincare disk? Remember, the boundary represents the "edge" of the representable world---where the agent runs out of capacity.
:::

:::{prf:proposition} Mass Scaling Near Boundary
:label: prop-mass-scaling-near-boundary

For the Poincare disk, the mass tensor scales as:

$$
\mathbf{M}(z) = \frac{4}{(1-|z|^2)^2} I_d \quad \xrightarrow{|z| \to 1} \quad +\infty.

$$
The metric diverges as $|z| \to 1$, which bounds all finite-action trajectories to the interior of the disk.

*Proof.* Direct evaluation of the Poincare metric. The factor $(1-|z|^2)^{-2}$ diverges as $|z| \to 1$. $\square$

:::

:::{div} feynman-prose
The divergence has a precise consequence: finite-action trajectories are confined to the interior. In the chosen momentum convention, the same covector force also produces less coordinate motion as the boundary is approached. That is the useful geometric picture; it is stronger and safer than treating an infinite coefficient as an ordinary numerical mass.

Think of it as an interior barrier in the finite-action model. The coordinate motion can slow smoothly as the metric grows, while a numerical implementation still needs an explicit cutoff, projection, or boundary rule. The limiting statement does not by itself specify how a discretized trajectory behaves exactly at the cutoff.
:::

:::{prf:remark} Onsager--Machlup Scope
:label: prop-most-probable-path

For the controlled diffusion

$$
dz^k = b^k(z)\,ds + \sqrt{2T_c}\,\sigma^{kj}(z)\,dW^j_s,

$$
where $\sigma \sigma^T = G^{-1}$. The path functional defined above does not identify the most probable path of this diffusion. That identification requires the drift $b$, the reference measure, and a specified path-tube convention.

The singular-perturbation calculation in {ref}`Appendix A.4 <sec-appendix-a-full-derivations>` establishes the overdamped reduction only; it is not an Onsager--Machlup derivation.

:::

(sec-the-coupled-jump-diffusion-sde)=
## The Coupled Jump-Diffusion SDE

:::{div} feynman-prose
Alright, now we get to the actual equation of motion. This is where the rubber meets the road.

The agent is not just a point in space. Its continuous state has a position $z$ and a momentum $p$; the separate scalar $m$ is an importance weight used by the jump/WFR part of the model. Calling both of them "mass" is convenient shorthand, but they do different jobs: $G$ controls the momentum-to-velocity map, while $m$ records particle weight.

The dynamics have two parts:
1. **Continuous motion**: The particle slides around on the manifold, pulled by gradients and pushed by noise
2. **Discrete jumps**: Occasionally, the particle teleports from one chart to another

Why jumps? Because the latent space is not a single connected manifold. It is an atlas of overlapping charts, like the pages of a flip-book. Sometimes the best move is not to take a small step, but to flip to a completely different page---a different representation, a different conceptual framework.

Think of it like this: you are solving a problem, and you have been thinking about it one way, taking small incremental steps. Then suddenly you realize there is a completely different way to look at it. That is a jump. The continuous dynamics handle the incremental thinking; the jump process handles the conceptual leaps.
:::

The agent's state is not merely a point $z$ but a **particle with position and weight** $(z, m)$, where $m$ is the importance weight (belief probability). The dynamics couple continuous transport with discrete topological jumps.

*Cross-reference (WFR Boundary Conditions).* The SDE below uses the **Waking mode** interface policy (Definition {prf:ref}`def-waking-boundary-clamping`): observations enter through an assimilation source and motors may prescribe a WFR flux. These are Dirichlet-like and Neumann-like labels; an exact PDE trace or normal derivative requires the additional domain and limiting hypotheses stated in that definition. In **Dreaming mode** (Definition {prf:ref}`def-dreaming-reflective-boundary`), the sensory channel is reflective and the motor boundary is chosen separately. See {ref}`Section 23.5 <sec-wfr-boundary-conditions-waking-vs-dreaming>` for the mode-switching table and the thermodynamic interpretation ({ref}`Section 23.4 <sec-the-belief-evolution-cycle-perception-dreaming-action>`).

:::{prf:definition} Second-Order Geodesic Langevin Equation
:label: def-bulk-drift-continuous-flow

The agent's state evolves as a **particle with position $z$ and momentum $p$** on the Riemannian manifold $(\mathcal{Z}, G)$. The dynamics are given by the coupled **second-order Langevin SDE**:

$$
\begin{cases}
dz^k = G^{kj}(z)\, p_j\, ds \\[8pt]
dp_k = \left[ -\partial_k \Phi_{\text{eff}} - \gamma\, p_k + \beta_{\text{curl}}\, \mathcal{F}_{kj}\, G^{j\ell}\, p_\ell + \Gamma^m_{k\ell}\, G^{\ell j}\, p_j\, p_m + \gamma G_{kj}u_\pi^j \right] ds + \sqrt{2\gamma T_c}\, (G^{1/2})_{kj}\, dW^j_s
\end{cases}

$$
where:
- $\Phi_{\text{eff}}$ is the **effective potential** (Definition {prf:ref}`def-effective-potential`)
- $\gamma > 0$ is the **friction coefficient** (damping rate)
- $\mathcal{F}_{ij} = \partial_i \mathcal{R}_j - \partial_j \mathcal{R}_i$ is the **Value Curl** tensor (Definition {prf:ref}`def-value-curl`)
- $\beta_{\text{curl}} \ge 0$ is the **curl coupling strength** (dimensionless)
- $\Gamma^m_{k\ell}$ are the **Christoffel symbols** of the Levi-Civita connection (Proposition {prf:ref}`prop-explicit-christoffel-symbols-for-poincare-disk`)
- $u_\pi^j$ is the contravariant **policy velocity** from Definition {prf:ref}`def-the-control-field`; the momentum equation uses the covector force $\gamma G_{kj}u_\pi^j$
- $T_c$ is the **cognitive temperature** (Definition {prf:ref}`def-cognitive-temperature`)
- $W_s$ is a standard Wiener process

*Units.* In normalized computational units $s$, $\Phi_{\text{eff}}$, and $T_c$ are dimensionless. If a physical time scale $\tau$ is restored, $[p]=[z]^{-1}\tau^{-1}$ follows from the kinematic equation and the force coefficients are rescaled accordingly.

**Interpretation:** The position evolves via the momentum (kinematic relation), while the momentum evolves under:

1. **Gradient force**: $-\nabla\Phi_{\text{eff}}$ — force from effective potential
2. **Friction**: $-\gamma p$ — damping toward equilibrium
3. **Lorentz force**: $\beta_{\text{curl}} \mathcal{F} G^{-1} p$ — velocity-dependent force from Value Curl (perpendicular to velocity)
4. **Geodesic correction**: $+\Gamma^m_{k\ell}G^{\ell j}p_jp_m$ in covector momentum coordinates, which yields $-\Gamma(\dot z,\dot z)$ in the second-order position equation
5. **Control field**: $\gamma G_{kj}u_\pi^j$ — the policy velocity converted to a covector force
6. **Thermal noise**: $\sqrt{2\gamma T_c} G^{1/2} dW$ — fluctuation-dissipation balanced noise

**Hamiltonian Structure:** In the conservative, uncontrolled subcase ($T_c=0$, $\gamma=0$, $\beta_{\text{curl}}=0$, and $u_\pi=0$), the deterministic part derives from the Hamiltonian:

$$
H(z, p) = \frac{1}{2} G^{ij}(z)\, p_i\, p_j + \Phi_{\text{eff}}(z).

$$
With $\beta_{\text{curl}}=0$ and any policy force absorbed into a scalar potential, the friction and noise terms form the usual **Ornstein--Uhlenbeck thermostat** and the Boltzmann statement applies under the stated boundary and regularity conditions.

**Conservative Limit:** When $\mathcal{F} = 0$ (Definition {prf:ref}`def-conservative-reward-field`), the Lorentz term vanishes and we recover the standard geodesic Langevin equation.

**Non-Conservative Dynamics:** When $\mathcal{F} \neq 0$, the Lorentz force induces rotational dynamics. Trajectories may converge to limit cycles rather than fixed points (Theorem {prf:ref}`thm-ness-existence`).

*Remark (BAOAB Integration).* This second-order system is integrated using the Boris-BAOAB scheme (Definition {prf:ref}`def-baoab-splitting`), which preserves the Boltzmann distribution to $O(h^2)$ and handles the velocity-dependent Lorentz force via the Boris rotation.

:::

:::{div} feynman-prose
Let me break down this second-order system, because it looks intimidating but each piece has a clear physical meaning.

**Why two equations?** The agent has both *position* $z$ (where it is) and *momentum* $p$ (how fast it is moving and in what direction). This is like a ball rolling on a curved surface: you need to track both where it is and how fast it is going.

**The position equation** $dz = G^{-1}p\, ds$: This just says "position changes according to velocity." The $G^{-1}$ converts momentum (a covector) to velocity (a vector) using the metric.

**The momentum equation** has several covector forces. This word matters: $p_k$ and every term on its right-hand side carry a lower index. A policy is specified upstream as a contravariant velocity, so it must be lowered with the metric before it can be added to the momentum equation.

**The gradient term** $-\nabla\Phi_{\text{eff}}$: This is "roll downhill." The agent feels a force pushing it toward lower potential. In the pure conservative control limit, the $V_{\text{critic}}$ part becomes $-G^{-1}dV_{\text{critic}}$ at the velocity level, so the motion descends cost-to-go rather than seeking a larger reward-labelled value.

**The friction term** $-\gamma p$: This is damping. Without it, the agent would coast forever. Friction brings it toward equilibrium, balancing against the thermal noise.

**The control term** $\gamma G_{kj}u_\pi^j$: This is the policy. The agent can choose to go somewhere that is not just downhill. The policy provides a contravariant velocity $u_\pi^j$; multiplying by $G_{kj}$ converts it to the covector force that belongs in the momentum equation.

**The Lorentz force** $\beta_{\text{curl}} \mathcal{F} G^{-1} p$: When the reward field has curl, the agent feels a sideways force perpendicular to its velocity---exactly like a charged particle in a magnetic field. It makes the agent spiral or orbit rather than just fall to the bottom.

**The geodesic correction** $+\Gamma^m_{k\ell}G^{\ell j}p_jp_m$ in the covector momentum equation: On a curved manifold, the components of a covector change as the base point moves. After converting back to the second-order position equation, the same connection contribution appears as $-\Gamma(v,v)$ with $v=G^{-1}p$. The signs differ because these are two coordinate descriptions of the same geodesic motion.

**The thermal noise** $\sqrt{2\gamma T_c} G^{1/2} dW$: Exploration via random thermal fluctuations. The $\sqrt{2\gamma T_c}$ factor is the fluctuation-dissipation normalization in the stated momentum convention. A Boltzmann conclusion still requires the conservative, uncontrolled hypotheses and the specified boundary and regularity conditions.

The Hamiltonian picture applies to the conservative, uncontrolled subcase. Once curl or policy forcing is present, the dynamics are controlled and generally non-reversible; the OU thermostat and Boltzmann interpretation must be read with those restrictions in view.
:::

:::{admonition} The Six Terms in the Momentum Equation
:class: feynman-added note

| Term | Expression | Physical Analogy | Effect |
|------|------------|------------------|--------|
| **Gradient** | $-\nabla\Phi_{\text{eff}}$ | Gravity | Pulls toward low potential |
| **Friction** | $-\gamma p$ | Viscous drag | Damps toward equilibrium |
| **Control** | $\gamma G_{kj}u_\pi^j$ | Rocket thrust | Policy velocity converted to a covector force |
| **Lorentz** | $\beta_{\text{curl}} \mathcal{F} G^{-1} p$ | Magnetic force | Induces rotation/orbiting |
| **Geodesic** | $+\Gamma^m_{k\ell}G^{\ell j}p_jp_m$ in $dp_k$; $-\Gamma(v,v)$ in $\ddot z$ | Connection correction | Keeps the covector and velocity descriptions consistent |
| **Noise** | $\sqrt{2\gamma T_c} G^{1/2} dW$ | Thermal fluctuation | Exploration + equilibrium sampling |

The friction-noise pair is an **Ornstein--Uhlenbeck thermostat**. Its Boltzmann sampling claim is conditional on the conservative, uncontrolled, constant-temperature setting and an exact compatible splitting.
:::

:::{prf:proposition} Explicit Christoffel Symbols for Poincaré Disk
:label: prop-explicit-christoffel-symbols-for-poincare-disk

For the Poincare disk model with metric $G_{ij} = \frac{4\delta_{ij}}{(1-|z|^2)^2}$, the Christoffel symbols in Cartesian coordinates are:

$$
\Gamma^k_{ij}(z) = \frac{2}{1-|z|^2}\left(\delta^k_i z_j + \delta^k_j z_i - \delta_{ij} z^k\right).

$$
The geodesic correction term $\Gamma^k_{ij}\dot{z}^i\dot{z}^j$ contracts to:

$$
\Gamma^k_{ij}\dot{z}^i\dot{z}^j = \frac{4(z \cdot \dot{z})}{1-|z|^2}\dot{z}^k - \frac{2|\dot{z}|^2}{1-|z|^2}z^k.

$$
*Proof.* Direct computation from $\Gamma^k_{ij} = \frac{1}{2}G^{k\ell}(\partial_i G_{j\ell} + \partial_j G_{i\ell} - \partial_\ell G_{ij})$ using $\partial_m[(1-|z|^2)^{-2}] = 4z_m(1-|z|^2)^{-3}$. $\square$

*Geometric interpretation.* The first term corrects the radial component of an outward coordinate velocity; the second supplies the complementary centripetal correction. Together they ensure geodesics are circular arcs perpendicular to the boundary.

:::

:::{div} feynman-prose
Why should you care about the explicit Christoffel symbols? Because they tell you something beautiful about the geometry.

On the Poincare disk, the geodesics---the "straight lines"---are not straight at all in Euclidean terms. They are arcs of circles that hit the boundary at right angles. The Christoffel symbols encode this. For an outward coordinate velocity, the first contraction term enters the geodesic equation with a minus sign and decelerates the coordinate motion; the hyperbolic speed can still remain constant. A tangential velocity receives the complementary centripetal correction.

The formula might look complicated, but it is just saying: "adjust the coordinate components so that the velocity is transported in the right way for this particular geometry." A connection correction is not an additional physical force, and its sign depends on whether you are reading the momentum equation or the second-order position equation.
:::

:::{prf:definition} Mass Evolution - Jump Process
:label: def-mass-evolution-jump-process

The importance weight $m(s)$ evolves according to a coupled jump-diffusion:

$$
dm = m \cdot r(z, a)\,ds + m \cdot (\eta - 1)\,dN_s,

$$
where:
- $r(z, a)$ is the **reaction rate** from the WFR dynamics ({ref}`Section 20.2 <sec-the-wfr-metric>`)
- $N_s$ is a Poisson process with target-dependent intensity
  $\lambda_{\text{jump}}(z\to z')$
- $\eta$ is the multiplicative jump factor; its value is a declared
  resampling convention rather than an implication of the critic cost

*Interpretation:* Between jumps, mass evolves smoothly via the reaction term $r$. At jump times, the mass is rescaled by factor $\eta$, and the position is teleported via the chart transition operator $L_{i \to j}$.

:::

:::{div} feynman-prose
Now here is the discrete part of the dynamics. The mass $m$ is like a betting stake---it tells you how much probability weight this particular trajectory carries.

Between jumps, the mass can grow or shrink according to the reaction rate $r$. Positive $r$ increases the particle weight under the displayed law and negative $r$ decreases it. Whether that weight represents better evidence is determined by how $r$ is calibrated to the target density; it cannot be inferred from the sign of the critic cost alone.

At jump times, something more dramatic happens. You teleport to a different chart, and your mass gets rescaled by the model's factor $\eta$. A choice such as $\eta>1$ can encode increased weight after a selected transition, but it is a convention of the resampling model rather than a consequence of the rate formula. The jump is like a sudden insight: "Wait, I should be thinking about this completely differently!"

This is analogous to resampling in particle filtering. To make it literally equivalent to a sequential Monte Carlo step, one must also specify the target-selection law, normalization, and any killing or cloning rule. The rate here is target-dependent, so the total jump intensity and the selected chart have to be tracked separately.
:::

:::{prf:definition} Target-dependent Jump Intensity
:label: prop-jump-intensity-from-value-discontinuity

For a proposed transition $z\mapsto L(z)$, one admissible target-dependent jump intensity is:

$$
\lambda_{\text{jump}}(z\to L(z)) = \lambda_0 \cdot \exp\left(\beta_{\text{ent}} \cdot \left( V_{\text{source}}(z) - V_{\text{target}}(L(z)) - c_{\text{transport}} \right) \right),

$$
where:
- $\lambda_0 > 0$ is a base jump rate
- $\beta_{\text{ent}} > 0$ is the inverse temperature (sharpness)
- $V_{\text{target}}$ and $V_{\text{source}}$ are cost-to-go functions on the target and source charts; the displayed sign favors lower target cost
- $L: \mathcal{Z}_{\text{source}} \to \mathcal{Z}_{\text{target}}$ is the chart transition operator
- $c_{\text{transport}} \ge 0$ is the transport cost (WFR term)

*Remark (SMC Interpretation).* The mass $m(s)$ is precisely the **importance weight** in Sequential Monte Carlo (SMC) / particle filtering. The agent is a single-particle realization of the WFR flow from {ref}`Section 20 <sec-wasserstein-fisher-rao-geometry-unified-transport-on-hybrid-state-spaces>`. Multiple particles can be used for ensemble-based generation.

**Cross-references:** {ref}`Section 20.2 <sec-the-wfr-metric>` ({prf:ref}`def-the-wfr-action`), {ref}`Section 20.6 <sec-the-unified-world-model>` (WFR world model), {ref}`Section 11 <sec-intrinsic-motivation-maximum-entropy-exploration>` (Filtering and projection).

:::

:::{div} feynman-prose
The displayed jump intensity is one admissible score-to-rate rule. It is exponential in the source-cost minus target-cost improvement, reduced by a transport cost. It is a rate, not itself a probability; over a small interval $h$, the corresponding event probability is $1-e^{-\lambda h}$ once a target has been selected.

- With the displayed sign, a target with lower $V_{\text{target}}$ has a larger rate, all else being equal.
- This is consistent with the cost-to-go convention: a positive source-minus-target difference records a reduction in predicted future cost, while a higher-cost target makes the rate smaller, though still positive for finite scores.
- The transport cost $c_{\text{transport}}$ provides a barrier: even if the grass looks greener, there is a cost to jumping the fence

$\beta_{\text{ent}}$ controls how sharply the rate responds to the cost improvement. At high $\beta_{\text{ent}}$, positive cost reductions are amplified and negative ones are suppressed; at low $\beta_{\text{ent}}$, the rates are less selective. The phrase "Boltzmann factor" is an analogy for this chosen rate law, not a claim that every chart transition obeys detailed balance.
:::

(sec-the-unified-effective-potential)=
## The Unified Effective Potential

:::{div} feynman-prose
We have been talking about the potential $\Phi$ that the agent rolls down. But where does this potential come from? It is not given to us by Nature; we have to construct it from the quantities we actually care about.

It turns out there are three natural contributions:

1. **The hyperbolic potential** $U$: This drives expansion from the origin toward the boundary. It is the "generation drive"---the urge to create, to produce output, to sample from the model.

2. **The critic cost-to-go** $V_{\text{critic}}$: This is the cost convention used in this volume. Lower values mean lower predicted future cost. Its positive contribution to $\Phi_{\text{eff}}$ supplies the conservative descent $-G^{-1}dV_{\text{critic}}$ in the overdamped control limit; it is not a high-reward score.

3. **The risk penalty** $\Psi_{\text{risk}}$: This is caution. Some regions are dangerous---high variance, unstable representations. You want to stay away from them.

The effective potential is a weighted combination of these three. The weight $\alpha$ controls the balance between generation and cost descent. When $\alpha = 1$, and when the conservative overdamped hypotheses also hold, the generation calculation gives the prescribed outward flow. When $\alpha = 0$, the scalar term supplies the cost-gradient contribution $-G^{-1}dV_{\text{critic}}$; calling that optimal control requires a control objective and admissible-control class. In between, the declared forces are blended.

This is the key to understanding the agent as a generative model: it is not *either* a generator *or* a controller. It is both, blended together through this unified potential.
:::

The effective potential unifies three terms: the hyperbolic information potential $U$ from holographic generation ({ref}`Section 21.1 <sec-hyperbolic-volume-and-entropic-drift>`), the learned value function $V$ from control ({ref}`Section 2.7 <sec-the-hjb-correspondence>`), and the risk-stress contribution $\Psi_{risk}$ from the stress-energy tensor ({ref}`Section 18 <sec-capacity-constrained-metric-law-geometry-from-interface-limits>`).

:::{prf:definition} Effective Potential
:label: def-effective-potential

The unified effective potential is:

$$
\Phi_{\text{eff}}(z, K) = \alpha\, U(z) + (1 - \alpha)\, V_{\text{critic}}(z, K) + \gamma_{risk}\, \Psi_{\text{risk}}(z),

$$
where:
- $U(z) = -d_{\mathbb{D}}(0, z) = -2\operatorname{artanh}(|z|)$ is the **hyperbolic information potential** (Definition {prf:ref}`def-hyperbolic-information-potential`)
- $V_{\text{critic}}(z, K)$ is the **learned cost-to-go/critic** on chart $K$ ({ref}`Section 2.7 <sec-the-hjb-correspondence>`; lower is better)
- $\Psi_{\text{risk}}(z) = \frac{1}{2}\operatorname{tr}(T_{ij} G^{ij})$ is the **risk-stress contribution** (Theorem {prf:ref}`thm-capacity-constrained-metric-law`)
- $\alpha \in [0, 1]$ is the generation-vs-control hyperparameter
- $\gamma_{risk} \ge 0$ is the risk aversion coefficient

Units: $[\Phi_{\text{eff}}] = \mathrm{nat}$.

:::

:::{admonition} Example: What the Agent "Feels"
:class: feynman-added example

Imagine the agent at position $z$ in the latent space:

- The **hyperbolic term** pulls it outward toward the boundary (generation drive)
- The **value term** pulls it toward high-reward regions (control drive)
- The **risk term** pushes it away from uncertain regions (caution)

The effective potential is a scalar landscape, and its conservative contribution to the momentum equation is the covector force $-\partial_k\Phi_{\text{eff}}$. The agent rolls downhill in this combined landscape; an independently supplied policy velocity or curl field adds the corresponding non-gradient contribution.

At $\alpha = 0.5$ (balanced): the scalar potential combines generation, critic cost, and risk. The resulting trajectory generates while descending the selected cost and avoiding risky regions only to the extent that the specified fields and policy produce those forces.
:::

:::{prf:proposition} Mode Interpretation
:label: prop-mode-interpretation

The parameter $\alpha$ interpolates between pure generation and pure control:

| Regime              | $\alpha$ Value      | Behavior                                                       |
|---------------------|---------------------|----------------------------------------------------------------|
| **Pure Generation** | $\alpha = 1$        | Flow follows $-\nabla_G U$ (holographic expansion, {ref}`Section 21 <sec-radial-generation-entropic-drift-and-policy-control>`) |
| **Pure Control**    | $\alpha = 0$        | Flow follows $-\nabla_G V_{\text{critic}}$ (policy gradient)   |
| **Hybrid**          | $\alpha \in (0, 1)$ | Balanced generation and control                                |

*Remark (Risk Modulation).* The $\gamma_{risk}$ term provides an additional penalty in high-stress regions (large $T_{ij}$), which further discourages risky trajectories beyond the geometric slowdown from Mass=Metric.

:::
:::{prf:corollary} Gradient Decomposition
:label: cor-gradient-decomposition

The gradient of the effective potential decomposes as:

$$
\nabla_G \Phi_{\text{eff}} = \alpha\, \nabla_G U + (1 - \alpha)\, \nabla_G V_{\text{critic}} + \gamma_{risk}\, \nabla_G \Psi_{\text{risk}}.

$$
For the Poincare disk model, the first term simplifies to:

$$
\nabla_G U = -\frac{(1-|z|^2)}{2}\, \hat{z}, \qquad \hat{z} = \frac{z}{|z|}.

$$
**Cross-references:** Definition {prf:ref}`def-hyperbolic-information-potential`, {ref}`Section 2.7 <sec-the-hjb-correspondence>` (Critic $V$), Section 14.2 (MaxEnt control), Theorem {prf:ref}`thm-capacity-constrained-metric-law`.

*Forward reference (Scalar Field Interpretation).* {ref}`Section 24 <sec-the-reward-field-value-forms-and-hodge-geometry>`
provides the complete field-theoretic interpretation of $V_{\text{critic}}$: the Critic solves the **Screened Poisson
Equation** (Theorem {prf:ref}`thm-the-hjb-helmholtz-correspondence`) with rewards as boundary flux (scalar charges in the
conservative case; Definition {prf:ref}`def-the-reward-flux`), the Value represents **Gibbs Free Energy** (Axiom
{prf:ref}`ax-the-boltzmann-value-law`), and the Value Hessian induces a **Conformal Coupling** to the metric (Definition
{prf:ref}`def-value-metric-conformal-coupling`).

:::

:::{div} feynman-prose
Let me say something about the hyperbolic gradient $\nabla_G U$ on the Poincare disk. The formula is:

$$
\nabla_G U = -\frac{(1-|z|^2)}{2}\, \hat{z}

$$

What does this mean? The gradient itself points inward because of the minus sign. The generation force is $-\nabla_G U$, so it points *outward*, toward the boundary, in the radial direction $\hat{z} = z/|z|$. Its coordinate magnitude is $(1-|z|^2)/2$, which is largest near the center and goes to zero at the boundary.

Under the declared pure-generation flow, this gives a strong coordinate drive near the center and a weaker one near the boundary. That is a modeling law for the radial generation schedule; it is not a statement that every policy on the disk must move radially.

The factor of $(1-|z|^2)$ is the inverse of the conformal factor. It compensates for the metric blowup at the boundary. In terms of *coordinate* velocity, the force looks like it is weakening. But in terms of *proper* velocity (measured with the metric), the expansion drive stays roughly constant until you get very close to the boundary.
:::

:::{prf:definition} Cognitive Temperature
:label: def-cognitive-temperature

The **cognitive temperature** $T_c > 0$ is the exploration-exploitation tradeoff parameter that controls:

1. **Diffusion magnitude:** The thermal noise term in the geodesic SDE scales as $\sqrt{2T_c}\,dW$
2. **Boltzmann policy:** The softmax temperature in $\pi(a|z) \propto \exp(Q(z,a)/T_c)$
3. **Free energy tradeoff:** The entropy-energy balance $\Phi = E - T_c S$

*Units:* nat (dimensionless in natural units where $k_B = 1$).

*Correspondence:* $T_c$ is the agent-theoretic analogue of thermodynamic temperature $k_B T$ in statistical mechanics.
:::

:::{div} feynman-prose
The cognitive temperature $T_c$ deserves special attention, because it is one of those parameters that looks simple but controls a lot.

At high $T_c$: The agent explores aggressively. The noise term dominates, the policy becomes diffuse, the free energy prefers entropy over energy. The agent is "hot"---it moves around a lot, tries many things, does not commit.

At low $T_c$: The agent exploits what it knows. The noise term is small, the policy becomes sharp, the free energy prefers energy over entropy. The agent is "cold"---it moves decisively toward the best known option.

This gives the exploration-exploitation tradeoff a thermodynamic analogy. The fluctuation-dissipation relation is a precise covariance constraint, while a Boltzmann distribution is available only in the conservative, reversible setting spelled out below.

The model may also choose to *cool down* as it learns: start with high $T_c$ to explore, then lower it to commit. That is a simulated-annealing schedule supplied by the model; it does not follow automatically from the constant-temperature equations.
:::

(sec-the-geodesic-baoab-integrator)=
## The Geodesic Boris-BAOAB Integrator

:::{div} feynman-prose
Now we get to the practical question: how do you actually simulate this? You have a stochastic differential equation on a curved manifold with multiple force terms. You cannot just use Euler's method.

The key insight is **operator splitting**. Instead of trying to handle everything at once, you split the dynamics into pieces and handle each piece separately. This is the BAOAB scheme:

- **B** for "kick" (the potential gradient)
- **A** for "drift" (moving along the manifold)
- **O** for "Ornstein-Uhlenbeck" (the thermostat, handling the noise)

The name BAOAB tells you the order: half-kick, half-drift, full thermostat, half-drift, half-kick. This symmetric structure can give second-order invariant-measure accuracy when the subflows are implemented as stated. That guarantee belongs to the conservative, constant-temperature, reversible setting; it does not automatically survive control, curl, variable coefficients, or an approximate drift. The exact Hamiltonian B/A pieces can be symplectic on the phase-space lift, but the dissipative O-step makes the full thermostatted update stochastic rather than symplectic.

Why add a Boris-type step to BAOAB? Because the Lorentz force depends on velocity. A standard kick does not handle that velocity-dependent term symmetrically. In three dimensions one can use a cross-product rotation; in general dimension the analogous Cayley transform acts on the matrix $\beta_{\text{curl}}\mathcal{F}G^{-1}$ and preserves the kinetic norm $p^TG^{-1}p$ under the skew condition. It is this metric kinetic norm, rather than the Euclidean $|p|$, that the force leaves unchanged.

The combination is useful only when each piece matches the SDE. The A-step must move $(z,p)$ by the cotangent geodesic flow, including parallel transport of the covector momentum; an exponential map with frozen $p$ is an approximation. The reference code is therefore an implementation approximation, not evidence that all these subflows are exact.
:::

We provide the numerical integrator for the controlled geodesic SDE (Definition {prf:ref}`def-bulk-drift-continuous-flow`). The **Boris-BAOAB** scheme extends the standard BAOAB {cite}`leimkuhler2016computation` to handle the velocity-dependent Lorentz force from non-conservative reward fields.

:::{prf:definition} Boris-BAOAB Splitting
:label: def-baoab-splitting

The Boris-BAOAB integrator splits the Lorentz-Langevin dynamics into five substeps per time step $h$:

1. **B** (half kick + optional curl rotation):
   - Apply the covector half-kick
     $p^- \leftarrow p - \frac{h}{2}(d\Phi-\gamma G u_\pi)$.
   - If $\mathcal{F}\neq0$, apply the Cayley update for the velocity-dependent term. With
     $A=\beta_{\text{curl}}\mathcal{F}G^{-1}$,

     $$
     p^+ \leftarrow \left(I-\frac{h}{2}A\right)^{-1}
       \left(I+\frac{h}{2}A\right)p^-.
     $$
     In three dimensions this has the usual Boris cross-product form; the matrix form is the
     definition in arbitrary dimension.
   - The second half-kick is the B substep at the end of the symmetric composition.

2. **A** (half drift): $z \leftarrow \operatorname{Exp}_z\left(\frac{h}{2} G^{-1}(z)\, p\right)$

3. **O** (thermostat): $p \leftarrow c_1 p + c_2\, G^{1/2}(z)\, \xi$, where $\xi \sim \mathcal{N}(0, I)$

4. **A** (half drift): $z \leftarrow \operatorname{Exp}_z\left(\frac{h}{2} G^{-1}(z)\, p\right)$

5. **B** (half kick + Boris rotation): Same as step 1

where $c_1 = e^{-\gamma h}$ and $c_2 = \sqrt{(1 - c_1^2) T_c}$.

**Conservative Limit:** When $\mathcal{F} = 0$, the Boris rotation is identity and the idealized composition reduces to standard BAOAB.

The reference code below is an implementation approximation: it uses an exponential-map drift and a local Christoffel correction, and therefore should not be read as an exact geodesic substep or as a proof of the invariant measure.

*Remark (Curl Rotation).* Under the skew condition $A^{\mathsf T}G^{-1}+G^{-1}A=0$, the Cayley update is a metric-orthogonal, volume-preserving rotation and preserves the kinetic norm $p^{\mathsf T}G^{-1}p$. It need not preserve the Euclidean norm $|p|$. The Lorentz force therefore does no work in the metric kinetic energy.

*Remark (O-step).* The O-step implements the **Ornstein-Uhlenbeck thermostat**, which exactly preserves the Maxwell-Boltzmann momentum distribution $p \sim \mathcal{N}(0, T_c G)$.

:::

:::{admonition} Why Splitting Works
:class: feynman-added tip

The idealized subflows are the pieces for which exactness can be claimed:
- **B-step** (kick): in a frozen position, the covector momentum receives the appropriate half-kick; with a Boris rotation, the force is applied using the corresponding symmetric substep
- **A-step** (drift): $(z,p)$ follows the cotangent geodesic flow, so $z$ moves by the exponential map and $p$ is parallel-transported along that geodesic
- **O-step** (thermostat): $p \to c_1 p + c_2 G^{1/2}\xi$ is the exact Ornstein--Uhlenbeck update for the stated local momentum law

For exact subflows, symmetric splitting (B-A-O-A-B) cancels the leading odd error and yields the stated $O(h^2)$ invariant-measure result under the proposition's conservative hypotheses. If the code freezes the momentum during A, adds an extra Christoffel correction, omits the rotation, or reuses a stale gradient, that code is not the exact composition and inherits no such guarantee.
:::

:::{prf:definition} Möbius Translation on the Poincaré Disk
:label: def-mobius-translation

For $c,z\in\mathbb D^d$, the Möbius translation that sends $c$ to the
origin is

$$
\phi_c(z):=(-c)\oplus z,
$$

where $\oplus$ is the Poincaré-disk Möbius addition used by the exponential
map below.  In particular $\phi_c(c)=0$ and $\phi_c$ is an isometry of the
Poincaré metric.  The map changes coordinates only; it does not alter a
potential, a router, or a probability law.

:::

**Algorithm 22.4.2 (Full Geodesic BAOAB with Jump Step).**

```python
import torch
import math
from dataclasses import dataclass
from typing import Tuple, Optional, Callable


@dataclass
class GeodesicState:
    """State of the geodesic integrator."""
    z: torch.Tensor          # [B, d] latent position
    p: torch.Tensor          # [B, d] momentum (covariant)
    K: torch.Tensor          # [B] chart index (integer)
    m: torch.Tensor          # [B] importance weight (mass)
    s: float                 # computation time


def poincare_metric(z: torch.Tensor) -> torch.Tensor:
    """
    Poincare disk metric tensor G_{ij}(z).

    G_{ij} = 4 delta_{ij} / (1 - |z|^2)^2

    Returns: [B, d, d] metric tensor
    """
    B, d = z.shape
    r_sq = (z ** 2).sum(dim=-1, keepdim=True)  # [B, 1]
    conformal_factor = 4.0 / (1.0 - r_sq + 1e-8) ** 2  # [B, 1]
    return conformal_factor.unsqueeze(-1) * torch.eye(d, device=z.device).expand(B, d, d)


def poincare_metric_inv(z: torch.Tensor) -> torch.Tensor:
    """
    Inverse Poincare metric G^{ij}(z).

    G^{ij} = (1 - |z|^2)^2 / 4 * delta^{ij}
    """
    B, d = z.shape
    r_sq = (z ** 2).sum(dim=-1, keepdim=True)
    inv_conformal = (1.0 - r_sq + 1e-8) ** 2 / 4.0
    return inv_conformal.unsqueeze(-1) * torch.eye(d, device=z.device).expand(B, d, d)


def mobius_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Möbius addition in the Poincare disk: a ⊕ b.

    (a + b) / (1 + <a, b>)  [simplified for small b]

    Full formula:
    a ⊕ b = ((1 + 2<a,b> + |b|^2) a + (1 - |a|^2) b) / (1 + 2<a,b> + |a|^2|b|^2)
    """
    a_sq = (a ** 2).sum(dim=-1, keepdim=True)
    b_sq = (b ** 2).sum(dim=-1, keepdim=True)
    a_dot_b = (a * b).sum(dim=-1, keepdim=True)

    num = (1 + 2*a_dot_b + b_sq) * a + (1 - a_sq) * b
    denom = 1 + 2*a_dot_b + a_sq * b_sq + 1e-8

    return num / denom


def poincare_exp_map(z: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Exponential map on Poincare disk: Exp_z(v).

    Exp_z(v) = φ_{-z}(tanh(||v||_z / 2) * v / ||v||)

    where φ_{-z} is Möbius translation and ||v||_z is the hyperbolic norm.
    """
    # Compute hyperbolic norm of v at z
    r_sq = (z ** 2).sum(dim=-1, keepdim=True)
    lambda_z = 2.0 / (1.0 - r_sq + 1e-8)  # conformal factor
    v_norm = torch.sqrt((v ** 2).sum(dim=-1, keepdim=True) + 1e-8) * lambda_z

    # Direction (normalized in Euclidean sense)
    v_dir = v / (torch.sqrt((v ** 2).sum(dim=-1, keepdim=True)) + 1e-8)

    # Magnitude in disk: tanh(||v||_z / 2)
    magnitude = torch.tanh(v_norm / 2.0)

    # Point at origin in direction v_dir with magnitude
    w = magnitude * v_dir

    # Translate by -z (Möbius addition)
    return mobius_add(z, w)


def christoffel_contraction(z: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Compute Γ^k_{ij} v^i v^j for Poincare disk.

    For conformal metric G = λ²I with λ = 2/(1-|z|²), the Christoffel contraction is:
    Γ^k_{ij} v^i v^j = 4(z·v)v^k/(1-|z|²) - 2|v|²z^k/(1-|z|²)

    Derivation: For G_{ij} = e^{2φ}δ_{ij}, Γ^k_{ij}v^iv^j = 2(v·∇φ)v^k - |v|²∂^kφ
    With φ = log(2/(1-|z|²)), we have ∂^kφ = 2z^k/(1-|z|²).
    """
    r_sq = (z ** 2).sum(dim=-1, keepdim=True)
    v_sq = (v ** 2).sum(dim=-1, keepdim=True)
    z_dot_v = (z * v).sum(dim=-1, keepdim=True)
    one_minus_r_sq = 1.0 - r_sq + 1e-8

    # Γ^k v^i v^j = 4(z·v)v^k/(1-|z|²) - 2|v|²z^k/(1-|z|²)
    term1 = (4.0 / one_minus_r_sq) * z_dot_v * v
    term2 = -(2.0 / one_minus_r_sq) * v_sq * z

    return term1 + term2


def geodesic_baoab_step(
    state: GeodesicState,
    grad_Phi: torch.Tensor,           # [B, d] gradient of effective potential
    u_pi: torch.Tensor,               # [B, d] control field from policy
    T_c: float,                       # cognitive temperature
    gamma: float,                     # friction coefficient
    h: float,                         # time step
    jump_rate_fn: Optional[Callable] = None,  # λ(z, K) -> [B]
    chart_transition_fn: Optional[Callable] = None,  # L(z, K_src, K_tgt) -> z'
    grad_Phi_fn: Optional[Callable] = None,  # (z, K) -> [B, d], optional endpoint gradient
    target_chart_fn: Optional[Callable] = None,  # (z, K) -> target chart indices
    num_charts: Optional[int] = None,
    mass_jump_factor: float = 1.0,
) -> GeodesicState:
    """
    Full Geodesic BAOAB integrator with Poisson jump process.

    Implements Algorithm 22.4.2:
    1. B-step: half kick from potential + control
    2. A-step: half drift via exponential map
    3. O-step: Ornstein-Uhlenbeck thermostat
    4. A-step: half drift
    5. B-step: half kick
    6. Jump-step: Poisson process for chart transitions

    Cross-references:
        - {prf:ref}`def-bulk-drift-continuous-flow` (bulk drift SDE)
        - {prf:ref}`prop-jump-intensity-from-value-discontinuity` (target-dependent jump rate)
        - {ref}`Section 2.5.1 <sec-levi-civita-connection-and-parallel-transport>` (Christoffel symbols)
    """
    z, p, K, m = state.z, state.p, state.K, state.m
    B, d = z.shape
    device = z.device

    # BAOAB coefficients
    c1 = math.exp(-gamma * h)
    c2 = math.sqrt((1 - c1**2) * T_c) if T_c > 0 else 0.0

    # ===== B-step: half kick =====
    # u_pi is a contravariant velocity; lower it before adding the policy force.
    G = poincare_metric(z)
    control_covector = gamma * torch.einsum("bij,bj->bi", G, u_pi)
    total_force = grad_Phi - control_covector
    p = p - (h / 2) * total_force

    # ===== A-step: half drift =====
    # z ← Exp_z((h/2) G^{-1} p)
    G_inv = poincare_metric_inv(z)
    velocity = torch.einsum('bij,bj->bi', G_inv, p)  # contravariant velocity

    # The exponential-map drift is a local approximation here. An exact A-step
    # would also parallel-transport p along the geodesic; the reference code
    # leaves p in its local frame and therefore carries no exact symplectic claim.
    z = poincare_exp_map(z, (h / 2) * velocity)

    # ===== O-step: thermostat =====
    # p ← c₁ p + c₂ G^{1/2} ξ
    G = poincare_metric(z)
    # G^{1/2} via Cholesky (for diagonal, just sqrt of diagonal)
    r_sq = (z ** 2).sum(dim=-1, keepdim=True)
    G_sqrt_factor = 2.0 / (1.0 - r_sq + 1e-8)  # sqrt of conformal factor

    xi = torch.randn_like(p)
    p = c1 * p + c2 * G_sqrt_factor * xi

    # ===== A-step: half drift =====
    G_inv = poincare_metric_inv(z)
    velocity = torch.einsum('bij,bj->bi', G_inv, p)
    z = poincare_exp_map(z, (h / 2) * velocity)

    # ===== B-step: half kick =====
    # A symmetric composition evaluates the endpoint force at the endpoint.
    grad_end = grad_Phi_fn(z, K) if grad_Phi_fn is not None else grad_Phi
    G_end = poincare_metric(z)
    control_end = gamma * torch.einsum("bij,bj->bi", G_end, u_pi)
    total_force_end = grad_end - control_end
    p = p - (h / 2) * total_force_end

    # ===== Jump-step: Poisson process =====
    if jump_rate_fn is not None and chart_transition_fn is not None:
        # Compute jump probability
        lambda_jump = jump_rate_fn(z, K)  # [B]
        prob_jump = 1 - torch.exp(-lambda_jump * h)

        # Sample jumps
        u = torch.rand(B, device=device)
        jumps = u < prob_jump  # [B] boolean

        if jumps.any():
            # The rate function supplies the total intensity. A target selector
            # supplies the sampled target; the fallback is only a finite-chart
            # example and must be configured with ``num_charts``.
            if target_chart_fn is not None:
                K_target = target_chart_fn(z, K)
            else:
                if num_charts is None or num_charts < 2:
                    raise ValueError("target_chart_fn or num_charts >= 2 is required for jumps")
                K_target = (K + 1) % num_charts

            # Apply chart transition for jumping particles
            z_new = chart_transition_fn(z, K, K_target)
            z = torch.where(jumps.unsqueeze(-1), z_new, z)
            K = torch.where(jumps, K_target, K)

            # Update mass (importance weight) using the declared resampling factor.
            m = torch.where(jumps, m * mass_jump_factor, m)

    # Project to ensure we stay in disk
    z_norm = torch.sqrt((z ** 2).sum(dim=-1, keepdim=True))
    z = torch.where(z_norm > 0.999, z * 0.999 / z_norm, z)

    return GeodesicState(z=z, p=p, K=K, m=m, s=state.s + h)
```
:::{prf:proposition} BAOAB Preserves Boltzmann
:label: prop-baoab-preserves-boltzmann

Under the conservative hypotheses $\beta_{\text{curl}}=0$, $u_\pi=0$, constant $T_c$, reversible boundary conditions, and an exact implementation of the stated symmetric splitting, the BAOAB integrator preserves the Boltzmann distribution $\rho(z, p) \propto \exp(-H(z,p)/T_c)$ to second order in $h$.

*Proof sketch.* The symmetric splitting B-A-O-A-B ensures time-reversibility of the deterministic steps. The O-step exactly samples the Maxwell-Boltzmann momentum distribution. Together, these guarantee that $\rho$ is a fixed point of the numerical flow up to $O(h^3)$ errors. See {cite}`leimkuhler2016computation`. $\square$

*Remark (Comparison to Euler-Maruyama).* Euler-Maruyama has $O(h)$ bias in the stationary distribution, whereas BAOAB achieves $O(h^2)$. For long trajectories, this difference is critical.

:::

:::{div} feynman-prose
This result about preserving the Boltzmann distribution is crucial. When you run a simulation, you want it to sample from the correct distribution. If your integrator has a bias, your samples will be wrong---you will be over-sampling some regions and under-sampling others.

The comparison is conditional. For a fixed conservative Langevin problem, Euler-Maruyama generally has an $O(h)$ stationary bias, while an exact compatible BAOAB composition has the $O(h^2)$ bias stated in the proposition. An order symbol is an asymptotic scaling, not a promise that $h=0.01$ produces exactly a one-percent error.

With the same small step, the second-order method can therefore be much less biased, but the constant depends on the potential, metric, boundary, and observable. Control, curl, state-dependent temperature, and approximate geodesic steps change the problem and require their own analysis.

The price is the extra substeps and the need to implement their geometry correctly. That price is justified when the conservative sampling guarantee is the quantity being tested; outside that regime, the method remains a numerical integrator whose error must be measured directly.
:::

(pi-langevin-thermostat)=
::::{admonition} Physics Isomorphism: Langevin Thermostat
:class: note

**In Physics:** The Langevin equation $m\ddot{x} = -\nabla U - \gamma\dot{x} + \sqrt{2\gamma k_B T}\,\xi(t)$ describes Brownian motion in a potential with friction $\gamma$ and thermal noise. The Ornstein-Uhlenbeck thermostat samples the Maxwell-Boltzmann distribution $p \propto \exp(-mv^2/2k_BT)$ {cite}`leimkuhler2016computation`.

**In Implementation:** The BAOAB integrator (Definition {prf:ref}`def-baoab-splitting`) splits the dynamics:
- **B:** $p \gets p - \frac{h}{2}\nabla_z\Phi_{\text{eff}}$ (kick)
- **A:** $z \gets z + \frac{h}{2}G^{-1}p$ (drift)
- **O:** $p \gets c_1 p + c_2 G^{1/2}\xi$ (thermostat)
- **A, B:** repeat

**Correspondence Table:**
| Molecular Dynamics | Agent (BAOAB) |
|:-------------------|:--------------|
| Position $x$ | Latent state $z$ |
| Momentum $p$ | Auxiliary variable $p$ |
| Potential $U(x)$ | Effective potential $\Phi_{\text{eff}}(z)$ |
| Friction $\gamma$ | Damping coefficient |
| Temperature $k_B T$ | Cognitive temperature $T_c$ |
| Maxwell-Boltzmann | Stationary policy distribution |

**Advantage:** BAOAB preserves the Boltzmann distribution to $O(h^2)$ (Proposition {prf:ref}`prop-baoab-preserves-boltzmann`), avoiding the $O(h)$ bias of Euler-Maruyama.
::::

(pi-detailed-balance)=
::::{admonition} Physics Isomorphism: Detailed Balance
:class: note

**In Physics:** A stochastic process satisfies detailed balance if transition rates satisfy $\pi(x)W(x \to y) = \pi(y)W(y \to x)$ for all states $x, y$. This implies the stationary distribution $\pi$ and time-reversibility {cite}`vanKampen1992stochastic`.

**In Implementation:** The conservative, reversible WFR subcase can satisfy detailed balance at equilibrium:

$$
\rho_*(z) \cdot J(z \to z') = \rho_*(z') \cdot J(z' \to z)

$$
where $\rho_* \propto \exp(-\Phi_{\text{eff}}/T_c)\sqrt{|G|}$ is the Boltzmann distribution.

**Correspondence Table:**
| Statistical Mechanics | Agent (Equilibrium) |
|:----------------------|:--------------------|
| Transition rate $W(x \to y)$ | Jump rate $\lambda_{KK'}$ |
| Stationary distribution $\pi$ | Equilibrium belief $\rho_*$ |
| Detailed balance | Reversibility at Nash |
| Entropy production $\dot{S}$ | Zero at equilibrium |
| Fluctuation-dissipation | Einstein relation for $T_c$ |

**Consequence (conservative case):** Under the compatible boundary, jump, and thermostat hypotheses, detailed balance ensures the BAOAB thermostat samples the correct distribution. Policy forcing, non-reversible jumps, or $\mathcal{F} \neq 0$ generally produce a NESS instead.
::::

(sec-the-overdamped-limit)=
## The Overdamped Limit

:::{div} feynman-prose
So far we have been working with the full second-order dynamics: position *and* momentum. But in many applications, you do not need all that machinery. When friction is strong enough, the momentum equilibrates almost instantly to the force, and you can forget about it.

This is the **overdamped limit**, and it is important for two reasons:

1. **Simplicity**: First-order dynamics are easier to simulate and analyze than second-order
2. **Relevance**: Many real systems operate in this regime---diffusion models, Brownian motion in viscous fluids, biological processes

The mathematical statement is a singular limit, not merely the instruction "set momentum to zero." When $m/\gamma\to0$ with the appropriate computation-time scaling, and in the conservative, uncontrolled, constant-temperature setting used by the theorem, the velocity quickly relaxes to

$$
\dot{z} \approx -G^{-1}\nabla\Phi.
$$

In physical time before this rescaling, the leading drift carries the expected $1/\gamma$ factor.
You can then eliminate momentum and compute the leading position drift from the force. A curl-corrected mobility is a useful formal extension, but it is outside the conservative limit proved here unless an additional scaling and proof are supplied.

The resulting proved equation is a first-order Ito diffusion with the metric gradient drift, the geometry-induced Ito correction, and thermal noise. In a broader controlled or non-conservative model one may obtain extra mobility terms, but those terms are not covered by this theorem. "Instantaneous" means only that momentum relaxation is asymptotically faster than position evolution.

When is this a good approximation? When the timescale of momentum relaxation ($\sim 1/\gamma$) is much shorter than the timescale of position changes. In that case, the momentum "slaves" to the position, and you can eliminate it.
:::

In many applications (diffusion models, biological control), the system operates in the **overdamped regime** where friction dominates inertia. We derive this limit rigorously.

:::{prf:theorem} Overdamped Limit
:label: thm-overdamped-limit

Consider the conservative second-order SDE obtained from Definition {prf:ref}`def-bulk-drift-continuous-flow` by setting
$\beta_{\text{curl}}=0$ and $u_\pi=0$, with inertial scale $m$ and friction $\gamma$:

$$
m\,\ddot{z}^k + \gamma\,\dot{z}^k + G^{kj}\partial_j\Phi_{\text{eff}} + \Gamma^k_{ij}\dot{z}^i\dot{z}^j = \sqrt{2\gamma T_c}\,\left(G^{-1/2}\right)^{kj}\,\xi^j,

$$
where $\xi$ is white noise. In physical time, the Smoluchowski--Kramers limit gives
$dz=-(1/\gamma)G^{-1}d\Phi_{\text{eff}}\,dt+\sqrt{2T_c/\gamma}\,G^{-1/2}dW_t$.
After the declared computation-time rescaling $s=t/\gamma$, the formal limit $m/\gamma\to0$ is the Ito equation:

$$
dz^k = \left[-G^{k\ell}(z)\,\partial_\ell\Phi_{\text{eff}}(z) - T_c G^{ij}(z)\Gamma^k_{ij}(z)\right] ds + \sqrt{2T_c}\,\left(G^{-1/2}(z)\right)^{kj}\,dW^j_s.

$$
*Proof sketch.* In the high-friction limit, velocity equilibrates instantaneously to
$\dot{z} \approx -(1/\gamma)G^{-1}\nabla\Phi_{\text{eff}}$ in physical time. The geodesic term
$\Gamma(\dot{z},\dot{z}) \sim O(|\dot{z}|^2) = O(\gamma^{-2})$ is negligible. What remains is the conservative
gradient flow with the fluctuation--dissipation noise. See {ref}`Appendix A.4 <sec-appendix-a-full-derivations>` for the
singular perturbation analysis. $\square$

:::

:::{div} feynman-prose
In the deterministic part of the overdamped limit, the inertial geodesic term $\Gamma(\dot{z},\dot{z})$ drops out because the velocity relaxation is fast. But noise on a curved manifold leaves a geometric trace: in Ito coordinates the limiting equation contains the drift $-T_cG^{ij}\Gamma^k_{ij}$. Dropping the inertial term does not mean that all Christoffel symbols disappear.

Under the theorem's conservative hypotheses, the coordinate form is
$dz^k = [-G^{k\ell}\partial_\ell\Phi - T_cG^{ij}\Gamma^k_{ij}]ds + \sqrt{2T_c}(G^{-1/2})^{kj}dW^j$.
The metric still controls the drift and diffusion, while the displayed Christoffel contraction supplies the Ito correction needed for the chosen reference measure. A curl mobility can be studied separately, but it is not silently part of this proved reduction.
:::

:::{prf:corollary} Recovery of Holographic Flow
:label: cor-recovery-of-holographic-flow

Setting $\alpha = 1$ (pure generation), $T_c \to 0$, $\mathcal{F}=0$, $u_\pi=0$, and using the computation-time unit in the overdamped equation recovers the prescribed holographic gradient flow from {ref}`Section 21.2 <sec-policy-control-field>`:

$$
\dot{z} = -G^{-1}(z)\,\nabla U(z).

$$
For the Poincare disk and $z\neq0$, this gives $\dot{z} = \frac{(1-|z|^2)}{2}\,\frac{z}{|z|}$, which integrates to $|z(\tau)| = \tanh(\tau/2+\operatorname{artanh}r_0)$.

*Proof.* Direct substitution of $\Phi_{\mathrm{eff}}=U$ in the pure-generation
limit. The explicit solution for the radial coordinate $r(\tau)=|z(\tau)|$
satisfies $\dot r=(1-r^2)/2$, which integrates to
$r(\tau)=\tanh(\tau/2+\operatorname{artanh}(r_0))$. For $r_0=0$, we get
$r(\tau)=\tanh(\tau/2)$. $\square$

*Remark.* This identifies the radial solution of the declared generation flow. It does not establish an optimal-control claim without a specified control objective and admissible-control class.

:::

:::{div} feynman-prose
This is one of those beautiful moments where everything fits together. In Section 21, we introduced the holographic flow $|z(\tau)| = \tanh(\tau/2)$ as a kind of "natural" radial expansion. It looked like an ad-hoc choice.

Under the explicit choices $\alpha=1$, $T_c\to0$, $\mathcal{F}=0$, $u_\pi=0$, and the declared computation-time unit, direct substitution recovers that radial generation field. The radial solution is unique for the stated radial ODE and initial condition; this does not make it the unique geodesic or an optimal-control solution on the whole disk.

This is a useful consistency check: the holographic prescription agrees with the conservative overdamped calculation in precisely that regime. It remains a selected modeling law, and other potentials or controls give other flows.
:::

:::{prf:corollary} Fokker-Planck Duality {cite}`risken1996fokkerplanck`
:label: cor-fokker-planck-duality

In the conservative overdamped subcase above, let $q(z,s)$ denote density
with respect to the Riemannian volume $d\mu_G=\sqrt{|G|}\,dz$.  Its stationary
density is

$$
q_*(z) \propto \exp\left(-\frac{\Phi_{\text{eff}}(z)}{T_c}\right).

$$
The corresponding density with respect to coordinate Lebesgue volume is
$p_*(z)=q_*(z)\sqrt{|G(z)|}$.  The two densities describe the same
Boltzmann law; the factor $\sqrt{|G|}$ is a change of reference measure.

*Proof.* The Fokker--Planck equation for the Riemannian density $q$ is:

$$
\partial_s q = \frac{1}{\sqrt{|G|}}\partial_i\!\left(\sqrt{|G|}G^{ij}\left( q\,\partial_j\Phi_{\text{eff}} + T_c\,\partial_j q \right)\right).

$$
Setting $\partial_s q=0$ and using detailed balance gives
$q\propto e^{-\Phi_{\mathrm{eff}}/T_c}$.  Multiplying by $\sqrt{|G|}$
gives the coordinate density stated above. $\square$

**Cross-references:** {ref}`Section 21.2 <sec-policy-control-field>` (Langevin dynamics), Theorem {prf:ref}`thm-equivalence-of-entropy-regularized-control-forms-discrete-macro`, {ref}`Section 2.11 <sec-variance-value-duality-and-information-conservation>` (Belief density evolution).

:::

:::{div} feynman-prose
The Fokker-Planck equation tells you how probability density evolves under the SDE. But we care most about the *stationary* distribution---the long-time limit. And look at that beautiful formula:

$$
p_*(z) \propto \exp\left(-\frac{\Phi}{T_c}\right)\,\sqrt{|G|}

$$

Under the conservative overdamped hypotheses, and with the Ito drift and reference measure used in the corollary, this is the Boltzmann density with respect to coordinate volume. The $\exp(-\Phi/T_c)$ part suppresses high-potential regions; when the potential contains the positive $V_{\text{critic}}$ term, that means high cost-to-go is suppressed and lower-cost states receive more weight. This is a cost preference, not a high-reward interpretation unless one changes the convention to $V=-\text{reward}$. The $\sqrt{|G|}$ factor converts between coordinate volume and Riemannian volume; it is a measure factor, not an extra reward for large metric.

Together they give the stated equilibrium only for that reversible subcase, with suitable boundary and regularity conditions. Curl, policy forcing, jumps, variable temperature, or a finite-step approximate integrator can produce a non-equilibrium law or sampling bias. The formula is therefore a target for a compatible MCMC implementation, not a guarantee for every version of the agent.
:::

(pi-fokker-planck)=
::::{admonition} Physics Isomorphism: Fokker-Planck Equation
:class: note

**In Physics:** The Fokker-Planck equation describes the time evolution of probability density under drift and diffusion: $\partial_t p = -\nabla \cdot (p\,\mathbf{F}) + D\nabla^2 p$. On a Riemannian manifold with metric $g$, the diffusion term becomes the Laplace-Beltrami operator {cite}`risken1996fokkerplanck`.

**In Implementation:** The belief density $\rho(z,s)$ evolves via (Corollary {prf:ref}`cor-fokker-planck-duality`):

$$
\partial_s p = \nabla_i\left( G^{ij}\left( p\,\partial_j\Phi_{\text{eff}} + T_c\,\partial_j p \right) \right)

$$
with stationary distribution $p_*(z) \propto \exp(-\Phi_{\text{eff}}(z)/T_c)\sqrt{|G(z)|}$.

**Correspondence Table:**
| Statistical Physics | Agent (Belief Dynamics) |
|:--------------------|:------------------------|
| Probability density $p(x,t)$ | Belief density $\rho(z,s)$ |
| Drift force $\mathbf{F}$ | Effective potential gradient $-\nabla\Phi_{\text{eff}}$ |
| Diffusion constant $D$ | Cognitive temperature $T_c$ |
| Laplacian $\nabla^2$ | Laplace-Beltrami $\Delta_G$ |
| Boltzmann equilibrium | WFR stationary distribution |

**Loss Function:** Stein discrepancy $\mathbb{E}[\|\nabla \log p - \nabla \log p_*\|^2_G]$.
::::

(sec-agent-lifecycle-summary)=
## Agent Lifecycle Summary

:::{div} feynman-prose
Now let me put all the pieces together. We have equations for continuous motion, for discrete jumps, for the potential, for the temperature. How do these combine into a coherent picture of what the agent does?

The agent lifecycle has five operational phases. They can be pictured using the language of phase transitions, but that language is an interpretation of the schedule, not a thermodynamic theorem:

1. **Init**: Start at the origin. This is the "gas phase"---maximum entropy, no commitment, all possibilities open.

2. **Kick**: Apply symmetry-breaking control. This is "nucleation"---you have to pick a direction, break the perfect symmetry of the origin.

3. **Bulk**: Geodesic flow with jumps. This is the "liquid phase"---flowing, exploring, but with structure. The trajectory can switch between charts, try different representations.

4. **Boundary**: Reach the cutoff radius. This is "crystallization"---committing to a specific output, sampling the texture.

5. **Decode**: Map to the output space. The latent trajectory becomes an actual observable.

The phases are selected by the lifecycle rules and boundary policy. The symmetry at the origin motivates a kick, the expansion potential supplies a bulk tendency, and the cutoff supplies a stopping criterion. Geometry helps organize the schedule; it does not by itself prove a phase transition or determine the policy.
:::

The complete agent lifecycle integrates the components from Sections 21-22 into a coherent execution flow.

:::{prf:definition} Agent Lifecycle Phases
:label: def-agent-lifecycle-phases


| Phase           | Time Interval                | Dynamics                         | Texture      | Key Operations                                                                         |
|-----------------|------------------------------|----------------------------------|--------------|----------------------------------------------------------------------------------------|
| **1. Init**     | $\tau = 0$                   | $z(0) = 0$                       | None         | Initialize at origin; $p(0) \sim \mathcal{N}(0, T_c G(0))$                             |
| **2. Kick**     | $[0, \tau_{kick}]$           | Langevin at origin               | None         | Apply symmetry-breaking control $u_\pi$ (Def. {prf:ref}`def-the-control-field`)        |
| **3. Bulk**     | $[\tau_{kick}, \tau_{stop}]$ | BAOAB + Jumps                    | **Firewall** | Geodesic flow with chart transitions                                                   |
| **4. Boundary** | $\tau = \tau_{stop}$         | $\lVert z\rVert \geq R_{cutoff}$ | Sampled      | Sample texture $z_{tex} \sim \mathcal{N}(0, \Sigma(z))$                                |
| **5. Decode**   | Post-$\tau_{stop}$           | —                                | Used         | $x = \text{Decoder}(e_K, z_n, z_{tex})$                                                |

*Remark.* The **Texture Firewall** (Axiom {prf:ref}`ax-bulk-boundary-decoupling`) ensures that $\partial_{z_{tex}} \dot{z} = 0$ throughout the bulk phase—texture is completely invisible to the dynamics.

**Algorithm 22.6.2 (Full Agent Loop).**

```python
def run_agent_loop(
    policy: Policy,
    decoder: Decoder,
    T_c: float,
    gamma: float,
    h: float,
    R_cutoff: float = 0.95,
    max_steps: int = 1000,
) -> torch.Tensor:
    """
    Execute the full agent lifecycle from init to decode.

    Returns: Generated output x
    """
    B, d = 1, policy.latent_dim
    device = policy.device

    # ===== Phase 1: Init =====
    z = torch.zeros(B, d, device=device)
    p = torch.randn(B, d, device=device) * math.sqrt(T_c * 4.0)  # G(0) = 4I
    K = torch.zeros(B, dtype=torch.long, device=device)
    m = torch.ones(B, device=device)
    state = GeodesicState(z=z, p=p, K=K, m=m, s=0.0)

    # ===== Phase 2: Kick =====
    # Keep the initial symmetry-breaking velocity and include it in the first
    # bulk control update instead of overwriting it before the first step.
    u_kick = policy.symmetry_breaking_kick(z, mode='generation')

    # ===== Phase 3: Bulk (with Texture Firewall) =====
    for step in range(max_steps):
        # Compute effective potential gradient.  The implementation must return
        # the declared zero subgradient for the radial U term at z=0.
        grad_Phi = compute_effective_potential_gradient(
            state.z, state.K, policy.value_fn, alpha=0.5
        )

        # Update control field.  The kick is applied on the first bulk step.
        u_pi = policy.control_field(state.z, state.K)
        if step == 0:
            u_pi = u_pi + u_kick

        # BAOAB step (texture is invisible here)
        state = geodesic_baoab_step(
            state, grad_Phi, u_pi, T_c, gamma, h,
            jump_rate_fn=policy.jump_rate,
            chart_transition_fn=policy.chart_transition,
            grad_Phi_fn=lambda z_, K_: compute_effective_potential_gradient(
                z_, K_, policy.value_fn, alpha=0.5
            ),
            num_charts=getattr(policy, "num_charts", None),
            mass_jump_factor=getattr(policy, "mass_jump_factor", 1.0),
        )

        # Check boundary condition
        z_norm = torch.sqrt((state.z ** 2).sum(dim=-1))
        if (z_norm >= R_cutoff).all():
            break

    # ===== Phase 4: Boundary - Sample texture =====
    z_tex = sample_holographic_texture(state.z, sigma_tex=0.1)

    # ===== Phase 5: Decode =====
    embedding = policy.chart_embedding(state.K)  # e_K
    x = decoder(embedding, state.z, z_tex)

    return x
```

:::

:::{admonition} The Texture Firewall
:class: feynman-added warning

Notice that during the Bulk phase, the texture is completely invisible. The dynamics depend only on $(z, K, m)$, not on any fine-grained texture information.

This is the **Texture Firewall**: texture is sampled only at the boundary, not during the bulk flow. Why? Because if texture influenced the dynamics, the bulk would become infinitely complex---you would need to track an infinite number of degrees of freedom.

The firewall ensures the bulk dynamics remain finite-dimensional, with all the high-dimensional structure appearing only at the final step.
:::

:::{prf:proposition} Lifecycle Schedule Interpretation
:label: prop-phase-transition-interpretation

The agent lifecycle admits a thermodynamic phase-transition analogy for its schedule:

| Phase | Thermodynamic Analogy | Order Parameter |
|-------|----------------------|-----------------|
| Init (gas) | High entropy, symmetric | $\lVert z\rVert = 0$ |
| Kick (nucleation) | Symmetry breaking | $u_\pi \neq 0$ |
| Bulk (liquid) | Directed flow | $0 < \lVert z\rVert < R_{cutoff}$ |
| Boundary (solid) | Crystallization | $\lVert z\rVert \geq R_{cutoff}$ |

:::

(sec-adaptive-thermodynamics)=
## Adaptive Thermodynamics (Fluctuation-Dissipation)

:::{div} feynman-prose
Up to now, we have been treating the temperature $T_c$ as a constant. But there is something unsatisfying about that. The metric $G$ varies across the manifold---should not the temperature vary too?

You may choose to vary the temperature with position, but that is a modeling decision. The **fluctuation-dissipation relation** then tells you how to match the noise covariance to the chosen temperature and friction. It does not choose the schedule, and matching the covariance alone does not guarantee equilibrium.

In the scalar shorthand, the relation reads $\sigma^2 = 2\gamma T_c / G$; in coordinates it is a covariance relation involving $G^{-1}$. The noise variance, friction, temperature, and metric are tied together after the schedule has been declared.

For the explicit schedule below, effective coordinate noise is larger near the origin and smaller near the boundary. That creates an exploration-to-exploitation operating schedule. It is a useful design, but the geometry does not supply it automatically, and no thermodynamic phase-transition claim follows without additional analysis.
:::

The temperature $T_c$ and friction $\gamma$ may be chosen as state-dependent coefficients. The Einstein relation constrains the noise covariance once those coefficients are chosen; it does not select a temperature schedule.

:::{prf:definition} Einstein Relation on Manifolds
:label: def-einstein-relation-on-manifolds

The fluctuation-dissipation relation requires:

$$
\sigma^2(z) = \frac{2\gamma(z)\, T_c}{G(z)},

$$
where $\sigma^2$ is the noise variance. This fixes the fluctuation--dissipation covariance for a chosen $T_c(z)$ and $\gamma(z)$; a Boltzmann equilibrium additionally requires constant coefficients (or the corresponding variable-coefficient correction terms).

:::
:::{prf:proposition} Geometry-scaled Temperature Schedule
:label: prop-automatic-phase-transitions

For the modeling choice $T_c(z)=T_0(1-|z|^2)^2/4$ on the Poincare disk, the effective coordinate noise decreases toward the boundary:

| Regime                      | Metric $G(z)$ | Effective Noise | Phase Behavior                |
|-----------------------------|---------------|-----------------|-------------------------------|
| **Uncertain** (near origin) | Small         | Large           | Gas phase (exploration)       |
| **Certain** (near boundary) | Large         | Small           | Solid phase (crystallization) |

*Remark.* This is an explicit geometry-scaled schedule, not a phase-transition theorem. The Einstein relation alone does not imply a thermodynamic phase transition.

:::

:::{div} feynman-prose
This is worth pausing on. In standard machine learning, an exploration-to-exploitation transition is implemented by choosing a temperature schedule. High temperature at the start and lower temperature later is simulated annealing, with parameters that must be selected.

Here we make one such choice explicitly, $T_c(z)=T_0(1-|z|^2)^2/4$. Given the metric and the Einstein relation, this makes the effective coordinate noise high near the center and low near the boundary. The schedule still has a scale $T_0$ and other implementation choices to set; it is not forced by the Einstein relation.

The geometry supplies the scale conversion, while the schedule supplies the desired behavior. Keeping those roles separate prevents a useful modeling interpretation from being mistaken for a theorem about automatic phase transitions.
:::

:::{prf:definition} Fisher-Covariance Duality
:label: def-fisher-covariance-duality

The inverse relationship between uncertainty and metric:

$$
G(z) \approx \Sigma^{-1}(z),

$$
where $\Sigma(z)$ is the posterior covariance of the belief at $z$. This duality underlies the Mass=Metric principle (Definition {prf:ref}`def-mass-tensor`).

**Algorithm 22.7.4 (Adaptive Temperature).**

```python
def adaptive_temperature(
    z: torch.Tensor,
    base_T: float,
    certainty_scale: float = 1.0,
) -> torch.Tensor:
    """
    Compute adaptive temperature based on local geometry.

    T_c(z) = base_T * (1 - |z|^2)^2 / 4

    This is a modeling choice that decreases T_c toward the boundary;
    Boltzmann-equilibrium guarantees for constant temperature do not apply.
    """
    r_sq = (z ** 2).sum(dim=-1, keepdim=True)
    # Conformal factor inverse: G^{-1} = (1-|z|^2)^2 / 4
    inv_conformal = (1.0 - r_sq + 1e-8) ** 2 / 4.0
    return base_T * inv_conformal * certainty_scale
```

:::

:::{div} feynman-prose
The Fisher-Covariance duality is a useful local modeling relation. Here is the intuition:

- The **Fisher information** tells you how much information the data provides about the parameters. High Fisher information means the data is very informative.
- The **posterior covariance** tells you how uncertain you are about the parameters after seeing the data. High covariance means high uncertainty.

In the regime where the local metric is identified with information, these are approximately related by $G \approx \Sigma^{-1}$. If the data is very informative (high $G$), the posterior covariance is small; if it is not informative (low $G$), the covariance is larger. The approximation depends on the statistical model and local coordinates.

This explains why Mass = Metric can be a sensible convention. Where the local metric is high, the update law can make coordinate motion more cautious; where it is low, motion can be more exploratory. The metric need not equal a posterior Fisher matrix in every model, and this relation is separate from the chosen temperature schedule.
:::

:::{prf:corollary} Deterministic Boundary
:label: cor-deterministic-boundary

As $|z| \to 1$:

$$
T_c(z) \to 0, \qquad \text{noise} \to 0.

$$
The coordinate noise in the bulk position tends to zero under this schedule. This does not make separately sampled boundary texture deterministic.

:::

:::{div} feynman-prose
This corollary is the final piece of the bulk picture. Under the explicit geometry-scaled schedule, the coordinate noise tends to zero as the boundary is approached. That makes the bulk position update increasingly deterministic.

Why is this useful? A fixed latent path can then be easier to reproduce near the cutoff. But the decoder may still receive separately sampled boundary texture, so the final output is reproducible only after the texture sampling rule and random seed are fixed.

The model therefore has a stochastic-to-low-noise transition in the bulk coordinate. Calling the boundary output deterministic requires the additional texture and decoding conditions; it does not follow from the vanishing coordinate noise alone.
:::

(sec-summary-tables-and-diagnostic-nodes)=
## Summary Tables and Diagnostic Nodes

:::{div} feynman-prose
Let me summarize what we have built. The equations of motion combine three ingredients:

1. **Geodesic dynamics** on a curved manifold, with the metric providing natural inertia
2. **Stochastic fluctuations** controlled by the cognitive temperature
3. **Discrete jumps** between charts, enabling topological transitions

These ingredients are related, but they do not all follow from one Onsager--Machlup principle. The displayed path functional is an operational objective. The metric and connection come from the geometric model; the noise, policy, jump rates, and temperature schedule are additional declared choices, with rigorous limits and invariant laws available only under their stated hypotheses.

The diagnostic nodes below help you check the implementation against the selected regime. A high GeodesicCheck can indicate an incorrect connection term, a force convention mismatch, or an integration error. A high JumpConsistencyCheck flags imbalance in the measured chart rates; it is evidence for investigating the rate and target-selection laws, not by itself proof that detailed balance should hold in a controlled or non-conservative model.

These are the kinds of things you want to monitor in a running system. Not just "is the loss going down," but "are the geometric invariants being preserved."
:::

**Summary of Equations of Motion:**

| Equation                 | Expression                                                                                                                  | Regime       | Units                |
|--------------------------|-----------------------------------------------------------------------------------------------------------------------------|--------------|----------------------|
| Free-energy path action | $S_{\mathrm{path}} = \int (\frac{1}{2}\mathbf{M}\lVert\dot{z}\rVert^2 + \Phi_{\text{eff}} + \frac{T_c}{12}R + T_c H_\pi)\,ds$ | Modeling objective | normalized units |
| Full Geodesic SDE        | $dz=G^{-1}p\,ds,\;dp=[-\nabla\Phi_{\text{eff}}-\gamma p+\beta_{\text{curl}}\mathcal{F}G^{-1}p+\Gamma(G^{-1}p,G^{-1}p)+\gamma Gu_\pi]ds+\sqrt{2\gamma T_c}\,G^{1/2}dW_s$ | Second-order | normalized units |
| Overdamped (conservative) | $dz^k=[-G^{k\ell}\partial_\ell\Phi_{\text{eff}}-T_cG^{ij}\Gamma^k_{ij}]ds+\sqrt{2T_c}G^{-1/2}dW_s$ | First-order | normalized units |
| Jump Intensity           | $\lambda_{K\to j}(z)=\lambda_0\exp\{\beta_{\mathrm{ent}}[V_K(z)-V_j(L_{K\to j}z)-c_{Kj}]\}$ | Discrete     | step$^{-1}$          |
| Mass = Metric            | $\mathbf{M}(z) \equiv G(z)$                                                                                                 | Kinematic    | $[z]^{-2}$           |
| Texture Covariance       | $\Sigma_{\text{tex}}(z) = \sigma_{\text{tex}}^2\, G^{-1}(z)$                                                                | Boundary     | $[z_{\text{tex}}]^2$ |

**Effective Potential Decomposition:**

$$
\Phi_{\text{eff}}(z, K) = \alpha\,U(z) + (1-\alpha)\,V_{\text{critic}}(z, K) + \gamma_{\text{risk}}\,\Psi_{\text{risk}}(z)

$$
where $\alpha \in [0,1]$ interpolates generation/control and $\gamma_{\text{risk}} \ge 0$ is risk aversion.

**BAOAB Coefficients:**

$$
c_1 = e^{-\gamma h}, \qquad c_2 = \sqrt{(1 - c_1^2)\,T_c}

$$
*Cross-reference:* The boundary-reached condition is monitored by **[Node 25 (HoloGenCheck)](#node-25)** defined in {ref}`Section 21.4 <sec-summary-and-diagnostic-node>`.

(node-26)=
**Node 26: GeodesicCheck**

| **#**  | **Name**          | **Component**            | **Type**                   | **Interpretation**                    | **Proxy**                                                                                  | **Cost**  |
|--------|-------------------|--------------------------|----------------------------|---------------------------------------|--------------------------------------------------------------------------------------------|-----------|
| **26** | **GeodesicCheck** | **World Model / Policy** | **Trajectory Consistency** | Is trajectory approximately geodesic? | $\lVert\ddot{z} + \Gamma(\dot{z},\dot{z}) + \gamma\dot z + G^{-1}\nabla\Phi_{\text{eff}} - \beta_{\text{curl}}G^{-1}\mathcal{F}\dot z - \gamma u_\pi\rVert_G$ | $O(BZ^2)$ |

**Trigger conditions:**
- High GeodesicCheck: Trajectory deviates from controlled geodesic (unexpected forces or integration errors).
- Remedy: Reduce time step $h$; verify Christoffel computation; check metric consistency.

(node-27)=
**Node 27: OverdampedCheck**

| **#**  | **Name**            | **Component** | **Type**            | **Interpretation**              | **Proxy**                                             | **Cost** |
|--------|---------------------|---------------|---------------------|---------------------------------|-------------------------------------------------------|----------|
| **27** | **OverdampedCheck** | **Policy**    | **Regime Validity** | Is inertia small relative to friction? | $\chi_{\mathrm{in}}:=\dfrac{m\lVert\dot v\rVert_G}{\gamma\lVert v\rVert_G+ m\lVert\dot v\rVert_G+\varepsilon}$ | $O(BZ)$  |

Here $v := \dot{z}$, $m$ is the inertial scale in the overdamped limit, and
$\chi_{\mathrm{in}}$ is small in the overdamped regime. The proxy is interpreted
only when the force and acceleration are measured in compatible metric units. The curl mobility
$\mathcal{M}_\gamma^{-1} = \gamma I - \beta_{\text{curl}}G^{-1}\mathcal{F}$
is a separate diagnostic quantity.

**Trigger conditions:**
- High OverdampedCheck: Operating in an inertial regime; use the full BAOAB integrator.
- Remedy: Increase friction $\gamma$ if overdamped limit desired; otherwise switch to second-order integrator.

(node-28)=
**Node 28: JumpConsistencyCheck**

| **#**  | **Name**                 | **Component**   | **Type**        | **Interpretation**                  | **Proxy**                                                       | **Cost**  |
|--------|--------------------------|-----------------|-----------------|-------------------------------------|-----------------------------------------------------------------|-----------|
| **28** | **JumpConsistencyCheck** | **World Model** | **Detailed-balance residual** | Are reversible jump rates balanced for the measured chart masses? | $\left(\sum_{K<j}\left\lvert\rho_K\lambda_{K\to j}-\rho_j\lambda_{j\to K}\right\rvert^2\right)^{1/2}$ | $O(BK^2)$ |

**Trigger conditions:**
- High JumpConsistencyCheck: the reversible detailed-balance residual is large under the declared equilibrium model.
- Remedy: verify the target-selection law and chart costs; use a separate mass-balance check when the process is intentionally non-reversible.

(node-29)=
**Node 29: TextureFirewallCheck**

| **#**  | **Name**                 | **Component**  | **Type**                     | **Interpretation**              | **Proxy**                                       | **Cost**             |
|--------|--------------------------|----------------|------------------------------|---------------------------------|-------------------------------------------------|----------------------|
| **29** | **TextureFirewallCheck** | **Generation** | **Bulk-Boundary Separation** | Is texture decoupled from bulk? | $\lVert\partial_{z_{\text{tex}}} \dot{z}\rVert$ | $O(BZ_{\text{tex}})$ |

**Trigger conditions:**
- High TextureFirewallCheck: Texture is leaking into dynamics (firewall violated).
- Remedy: Review implementation; ensure texture sampled only at boundary; verify Axiom {prf:ref}`ax-bulk-boundary-decoupling`.

:::{div} feynman-prose
We have now assembled the complete dynamical picture. The agent has a position and covector momentum on a curved manifold, while the separate scalar $m$ carries an importance weight. It feels covector forces from the effective potential, policy, and possibly curl field, is jostled by noise, and occasionally jumps between charts.

The geometry explains part of this picture. Mass = Metric is the selected momentum convention, and metric divergence gives the finite-action interior barrier. The Einstein relation constrains noise after a temperature and friction schedule have been chosen. The exploration-to-exploitation behavior comes from the explicit geometry-scaled schedule, while the phase-transition language remains an interpretation of the lifecycle.

In the next sections, we will see how this dynamical picture connects to the boundary structure and the decoder. The core message is: **once the geometry and the remaining modeling choices are stated with their hypotheses, the dynamics can be checked rather than guessed**.
:::
