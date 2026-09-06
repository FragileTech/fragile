(sec-radial-generation-entropic-drift-and-policy-control)=
# Radial Generation: Entropic Drift and Policy Control

## TLDR

- Model “generation” as **radial flow**: start near a symmetric, high-entropy center and move outward toward committed,
  specific states.
- Two forces drive the flow: **entropic drift** (push outward) and **policy/value control** (choose direction).
- This unifies diffusion-style generation and RL control as the same geometric picture on the latent manifold.
- The practical outputs are monitorable diagnostics (RadialGenCheck / HoloGenCheck) and knob interpretations (horizon,
  cutoff radius, temperature).
- This chapter links WFR belief geometry to concrete generative/control behavior at the boundary.

## Roadmap

1. Motivation and the radial-flow picture.
2. The drift/control decomposition and its relation to diffusion models.
3. Diagnostics and practical failure modes (early stop vs. runaway drift).

:::{div} feynman-prose
Let me tell you what generation really is. Not the mathematical abstraction---that's coming---but the *idea* first.

Imagine you're standing at the center of a strange kind of universe. At the center, every heading is equivalent, and the output is only weakly committed. As you move outward, the log-volume entropy of the hyperbolic space increases: there are more available microstates near the ideal boundary. At the same time, the particular trajectory can become more specific, so its residual output uncertainty can decrease. Those are two different uses of the word “uncertainty,” and keeping them separate prevents a sign error.

The radial part of the dynamics supplies an outward drift. A policy, a conditioning signal, or noise selects and continually adjusts the angular direction. In the two-dimensional Poincaré-disk calculation below, the entropic field has no preferred heading at the exact center, although its radial magnitude approaches $1/2$ there. The direction is therefore selected by the other terms, while the radial drift supplies the initial speed.

That's the useful picture of generation: expand through the bulk toward a declared stopping cutoff, then synthesize the output at the boundary interface. The same drift/control decomposition can describe RL and conditioned generation, but the control law and its interpretation must be stated separately in each setting.
:::

{cite}`ho2020ddpm,sohldickstein2015deep,nickel2017poincare`

(rb-diffusion-generation)=
:::{admonition} Researcher Bridge: Diffusion-Style Generation with Policy Drift
:class: info
If you know diffusion or score-based models, the radial expansion here is the generative flow. The policy is the controllable drift term that steers generation toward high-value regions.
:::

Data generation is defined as **radial expansion** of the latent state from the low-entropy origin ($z=0$) toward the high-entropy boundary ($|z| \to 1$). This boundary corresponds to the agent's {prf:ref}`def-boundary-markov-blanket`. The expansion is driven by the **entropic drift** (the natural tendency of {prf:ref}`def-hyperbolic-volume-growth` to increase) and steered by the **policy control field** $u_\pi$.

This section establishes the following unification: by identifying the **policy** as the source of initial direction selection, we merge Generative Modeling and Reinforcement Learning into a single variational operation:
- **RL:** The policy chooses a direction to maximize value $V(z)$.
- **Generation:** The policy (or context) chooses a direction to maximize semantic alignment with conditioning.
- Both contribute to the drift term in the latent SDE.

(sec-hyperbolic-volume-and-entropic-drift)=
## Hyperbolic Volume and Entropic Drift

:::{div} feynman-prose
Now I need to tell you something surprising about the geometry we're working in. You're probably used to Euclidean space---flat, ordinary space where a circle of radius $r$ has circumference $2\pi r$ and area $\pi r^2$. In Euclidean space, volume grows polynomially. Double the radius, quadruple the area.

But we're not in Euclidean space. For the explicit polar calculation here we fix $D=2$, the Poincaré disk. In that case, and more generally in hyperbolic $D$-space, volume grows *exponentially* with geodesic radius (with exponent $D-1$ in the general case).

Think about what this means. Near the center of the disk, there is relatively little room---few distinguishable states, few microstates, and small log-volume entropy. As you move toward the ideal boundary, the amount of room explodes exponentially. There is vastly more geometric volume near the boundary than near the center.

This exponential growth is why the log-volume gradient produces an outward entropic drift. A geometric volume calculation by itself does not say that every stochastic trajectory reaches the ideal boundary; it says why the radial component of the model points outward. The “entropic drift” we're about to define is this geometric tendency made precise.
:::

Consider the latent agent in the real Poincaré ball $\mathbb B^D=\{z\in\mathbb R^D:|z|<1\}$. The explicit polar calculations below use $D=2$ (the disk $\mathbb D\subset\mathbb C$); the ambient boundary is $S^{D-1}$.

:::{prf:definition} Manifold Boundary and Interior
:label: def-manifold-boundary-and-interior

Let $\mathcal{Z}\cong\mathbb B^D$ be the latent manifold. The **ideal boundary** is the $(D-1)$-dimensional limit set:

$$
\partial\mathcal{Z} := \{z \in \mathbb{R}^D : |z| = 1\}=S^{D-1}.

$$
The **interior** (or bulk) is the open disk:

$$
\text{int}(\mathcal{Z}) := \{z \in \mathbb{R}^D : |z| < 1\}.

$$
These are standard differential geometry terms; the boundary is the ideal boundary at infinity in the hyperbolic metric.

:::

:::{div} feynman-prose
Notice something subtle here: the boundary is at $|z| = 1$, but in hyperbolic geometry, that boundary is infinitely far away! If you try to walk to it in the hyperbolic metric, you'll never get there---distances diverge as you approach. This is why it's called the "ideal boundary" or "boundary at infinity."

This isn't a bug; it's a feature. It means there's infinite capacity near the boundary to encode fine-grained distinctions, while the interior remains finite and manageable for planning.
:::

:::{prf:definition} Hyperbolic Volume Growth
:label: def-hyperbolic-volume-growth

For the two-dimensional Poincaré disk ($D=2$) with metric $G_{ij} = \frac{4\delta_{ij}}{(1-|z|^2)^2}$, the volume of a hyperbolic ball $B_r(0)$ grows exponentially:

$$
\mathrm{Vol}(B_r(0)) = 4\pi \sinh^2\!\left(\frac{r}{2}\right) \;\approx\; \pi e^r \quad \text{as } r \to \infty.

$$
For general $D$, $\mathrm{Vol}(B_r)=\mathrm{Vol}(S^{D-1})\int_0^r\sinh^{D-1}(s)\,ds\sim e^{(D-1)r}$; the displayed formula and the later $r(\tau)$ calculation are the $D=2$ case.

:::

:::{admonition} Why Exponential Growth Matters
:class: feynman-added note

Let me make the exponential growth concrete. The numerical comparison below is specifically the $D=2$ Poincaré disk. In ordinary (Euclidean) 2D space:
- Circle of radius 1: area $\pi$
- Circle of radius 2: area $4\pi$
- Circle of radius 10: area $100\pi$

In hyperbolic space:
- Ball of radius 1: volume $\approx 4\pi \cdot 0.27 \approx 3.4$
- Ball of radius 2: volume $\approx 4\pi \cdot 1.38 \approx 17$
- Ball of radius 10: volume $\approx \pi e^{10} \approx 70,000$

See the difference? In the $D=2$ hyperbolic calculation, going from radius 2 to radius 10 increases volume by a factor of about 4,000. In Euclidean space, it's only a factor of 25. In $D$ dimensions the exponential rate is $D-1$, but the same geometric lesson remains: outer shells contain many more microstates.
:::

:::{prf:definition} The Entropic Force
:label: def-the-entropic-force

The "Free Energy" of a state at radius $r$ is dominated by the entropic volume term $S(r) = 2 \operatorname{artanh}(r)$. To maximize entropy (fill the capacity), the agent experiences a radial force:

$$
F_{\text{entropy}}(z) = \nabla_G S(z) = \frac{(1-|z|^2)}{2} \cdot \frac{z}{|z|}

$$
This accounts for the Poincaré metric conformal factor. The drift magnitude decreases near the boundary ($|z| \to 1$), ensuring the agent asymptotically approaches but never reaches it.

Units: $[F_{\text{entropy}}] = [z]/\tau$.

:::

:::{div} feynman-prose
Here's the physical intuition. Imagine a gas molecule in a container shaped like the hyperbolic disk. Near the center there are relatively few available microstates; near the ideal boundary there are exponentially more. The log-volume gradient therefore points outward. That is a statement about the drift term of this model, not a claim that an arbitrary stochastic process spends almost all of its time at the boundary.

The “entropic force” $F_{\text{entropy}}$ is the Riemannian gradient of that log-volume quantity. It is not a Newtonian force, and the word “force” should not make us forget the stochastic and policy terms. It supplies one deterministic component of the expected motion.
:::

:::{prf:proposition} Isotropic Radial Expansion
:label: prop-isotropic-radial-expansion

If acting alone (no policy steering), the entropic drift produces the isotropic expansion:

$$
r(\tau) = \tanh(\tau/2)

$$
This represents isotropic diffusion---expanding uniformly in all directions.

*Proof.* The overdamped equation $\dot{r} = (1-r^2)/2$ (from the Riemannian gradient of $U(z) = -2\operatorname{artanh}(|z|)$) integrates to $r(\tau) = \tanh(\tau/2 + \operatorname{artanh}(r_0))$. For $r_0 = 0$, we get $r(\tau) = \tanh(\tau/2)$. $\square$

:::

:::{admonition} What Does $r(\tau) = \tanh(\tau/2)$ Look Like?
:class: feynman-added example

Let's trace out this deterministic, control-free trajectory in the $D=2$ specialization:

| Time $\tau$ | Radius $r = \tanh(\tau/2)$ | Interpretation |
|-------------|---------------------------|----------------|
| 0 | 0 | Starting at origin (vacuum) |
| 1 | 0.46 | Nearly halfway to boundary |
| 2 | 0.76 | Three-quarters out |
| 4 | 0.96 | Very close to boundary |
| $\infty$ | 1 | At boundary (never quite reached) |

Notice how the particle starts fast and slows down asymptotically. This makes sense: the $(1-r^2)/2$ term in the dynamics goes to zero as $r \to 1$. You never quite reach the boundary in finite time---which is appropriate, since the boundary is "at infinity" in hyperbolic terms.
:::

:::{prf:definition} Hyperbolic Information Potential
:label: def-hyperbolic-information-potential

The **information potential** $U: \mathbb{D} \to \mathbb{R}$ is the negative hyperbolic distance from the origin:

$$
U(z) := -d_{\mathbb{D}}(0, z) = -2 \operatorname{artanh}(|z|) = -\log\!\left(\frac{1+|z|}{1-|z|}\right).

$$
Units: $[U] = \mathrm{nat}$.

*Remark (Thermodynamic Interpretation).* The log-volume $S=-U$ is minimal at the origin and increases toward the ideal boundary. Thus at $z=0$, $U=0$ is the maximum potential and the state has minimal committed information; as $|z|\to1$, $U\to-\infty$ and the information depth $-U(z)$ diverges.

:::

:::{div} feynman-prose
Now this is elegant. The “information potential” $U(z)$ is the negative hyperbolic distance from the origin, so it decreases as the radial coordinate grows. Think about what this means:

- At the origin, $U=0$ and the log-volume entropy $S=-U$ is minimal. The output may still be maximally unresolved, but that is a different quantity from geometric log-volume.
- As you move outward, $U$ becomes more negative, while the log-volume entropy increases. The trajectory is accumulating committed information even as the available geometric volume grows.
- At the ideal boundary, $U\to-\infty$ and the information depth $-U$ diverges. A finite stopping cutoff gives a finite operational commitment.

The sign is useful in the control equations: descending a cost potential uses a negative metric gradient. Here the entropic outward drift is $-\nabla_G U=\nabla_G S$; a separate policy control can be chosen independently.
:::

:::{prf:proposition} Riemannian Gradient of $U$
:label: prop-riemannian-gradient-of

The gradient in the Poincaré metric is:

$$
\nabla_G U(z) = G^{-1} \nabla U = -\frac{(1-|z|^2)}{2} \hat{z}, \quad \text{where } \hat{z} = \frac{z}{|z|}.

$$
The **entropic drift** (negative gradient) pushes radially outward:

$$
-\nabla_G U(z) = \frac{(1-|z|^2)}{2} \hat{z}.

$$
*Remark (Connection to {ref}`Section 7.11 <sec-the-geometry-of-the-latent-space-a-hyperbolic-hierarchy>`).* The Poincare coordinate $z$ relates to depth via $\rho = d_{\mathbb{D}}(0, z) = 2\operatorname{artanh}(|z|)$. Chart transitions are handled by the WFR jump process ({ref}`Section 22.2 <sec-the-coupled-jump-diffusion-sde>`), governed by the {prf:ref}`def-the-wfr-action`.

**Cross-references:** Definition {prf:ref}`def-information-density-and-bulk-information-volume`, Theorem {prf:ref}`thm-capacity-constrained-metric-law`.

:::

:::{div} feynman-prose
Notice the $(1-|z|^2)/2$ factor in the entropic drift. Near the origin ($|z| \approx 0$), its radial magnitude is about $1/2$; the field has no unique direction at the cone point, because every heading is equivalent. Near the boundary ($|z| \to 1$), the magnitude goes to zero. In the deterministic $D=2$ radial calculation this gives asymptotic approach without finite-time arrival.

This is the geometry telling you something important: the early stages of the declared radial flow have a strong geometric component, while the final refinement slows down. Noise and policy control still act throughout the path, so “coarse structure first” is an interpretation of this component, not a universal theorem about every generator.
:::

(sec-policy-control-field)=
## Policy Control Field

:::{div} feynman-prose
So far we've talked about the entropic drift---the field that pushes radially outward from the origin. It does not select a preferred heading. At the origin every direction looks the same, and the radial magnitude has a nonzero limiting value even though the vector direction is undefined.

This is where the policy comes in.

At the origin, you have perfect rotational symmetry---$SO(D)$ symmetry in $D$ dimensions. Any direction is as good as any other. The policy or the noise selects a heading, and the entropic drift then contributes outward radial motion along the selected ray. The policy may continue to steer as the state evolves; it is not only an initial kick.

Think of it like a compass placed at the center of a perfectly symmetric disk. The compass has no preferred bearing until a policy or a fluctuation supplies one. Once a bearing is present, the radial field supplies an outward component, while the control law can turn the path.
:::

At the origin ($z=0$), the system has full rotational symmetry $SO(D)$. To generate specific content (or solve a task), this symmetry must be broken. The **policy** provides the initial direction via the control field $u_\pi$.

:::{prf:proposition} SO(D) Symmetry at Origin
:label: prop-so-d-symmetry-at-origin

At $z = 0$:
1. The metric is isotropic: $G(0) = 4I$
2. The entropic field has no preferred direction at the origin; its radial magnitude has the limit $\lim_{r\downarrow0}|F_{\text{entropy}}|=1/2$.
3. The system has full rotational symmetry $SO(D)$

*Orbit calculation:* Every rotation fixes the zero vector, so its stabilizer is $SO(D)$ and its orbit is a point. The separately defined scalar vacuum has the representation-dependent mass matrix of {prf:ref}`thm-higgs-mechanism`.

:::

:::{admonition} What Happens to the Entropic Field at the Origin?
:class: feynman-added note

Look back at the formula: $F_{\text{entropy}}(z) = \frac{1-|z|^2}{2}\hat{z}$. As $r=|z|\downarrow0$, the prefactor tends to $1/2$, not to zero. The issue at $z=0$ is that $\hat z$ has no unique value: the radial field has no preferred direction at the cone point, but its radial magnitude has limit $1/2$.

Physically, this is exactly what rotational symmetry says. No direction can be singled out by an $SO(D)$-invariant radial rule at the center. That symmetry does not force the radial speed to vanish; it only prevents us from assigning one particular outward vector there.

This is crucial for the policy's role: at the origin, the policy or the noise supplies a direction, while the entropic term already supplies an outward radial component. The initial direction and the initial speed come from different parts of the model.
:::

:::{prf:definition} The Control Field
:label: def-the-control-field

The Policy $\pi_\theta(a|z)$ outputs actions in $\mathcal A$. After a declared smooth action-to-tangent map $B(z):\mathcal A\to T_z\mathcal Z$, its mean control is a vector field

$$
u_\pi(z) := B(z)\,\mathbb{E}_{a\sim\pi_\theta(\cdot|z)}[a].

$$
This vector field represents the **Information Preference** of the agent (or the User).

Units: $[u_\pi] = [z]/\tau$.

*Remark (Context-Conditioning).* {ref}`sec-the-context-space-unified-definition` generalizes this to **context-conditioned policies** $\pi(a|z,c)$ where the context $c \in \mathcal{C}$ unifies RL action spaces, classification label spaces, and LLM prompt spaces. For a cost potential, a compatible deterministic drift is $u_\pi(z,c)=-G^{-1}(z)\nabla_z\Phi_{\text{eff}}(z,K,c)$; a learned action policy need not be a gradient field.

:::

:::{div} feynman-prose
The action-to-tangent map $B(z)$ is important---it converts the policy's action into a proper tangent vector in the curved geometry. If the policy is represented by a covector force, one common choice is the metric raising map $B(z)=G^{-1}(z)$; the definition does not require every action space to be a covector space.

Under that particular metric-raised choice, $G^{-1}(0)=I/4$ and the control is a scaled action near the origin; near the boundary the metric factor suppresses that component. For a general smooth map $B$, its scaling must be checked separately.
:::

:::{prf:definition} Control Field at Origin
:label: def-control-field-at-origin

At $\tau=0$, the total drift is:

$$
F_{\text{total}} = F_{\text{entropy}} + u_\pi(0)

$$
The entropic field selects no direction at $z=0$, so the initial *direction* is determined by the policy (or by noise); its radial magnitude is not zero.

:::

:::{div} feynman-prose
This is the key insight: at the origin, the policy or noise must choose a direction because the radial field has no preferred heading. The entropic term already supplies outward speed, and the policy can keep adjusting the direction along the way. There is no mathematical reason to treat the first instant as the only meaningful control decision.

This explains why prompts and initial conditions matter in generation and RL, while leaving room for control throughout the trajectory. The model describes directional bias and radial expansion, rather than a one-time symmetry-breaking event.
:::

:::{prf:theorem} Unified Control Interpretation
:label: thm-unified-control-interpretation

The control field $u_\pi$ has three distinct operating modes:

| **Mode**                     | **Control Field $u_\pi$**                          | **Interpretation**                         |
|------------------------------|----------------------------------------------------|--------------------------------------------|
| **RL**                       | $u_\pi = -G^{-1} \nabla_z V_{\text{critic}}$        | Descends the cost-to-go                   |
| **Conditioned Generation**   | $u_\pi = G^{-1} \cdot \text{embed}(\text{prompt})$ | Clamped to user's prompt embedding         |
| **Unconditional (Dreaming)** | $u_\pi = 0$                                        | Pure thermal fluctuation selects direction |

*Scope.* Each row is a separate parameterization of a tangent drift. The policy-gradient theorem optimizes action parameters; it does not by itself identify $\nabla_zV$ with the policy output.

:::

:::{admonition} The Three Modes in Plain English
:class: feynman-added example

**RL mode:** "I want to reduce future cost." With the book's cost-to-go convention, the compatible gradient control is $u_\pi=-G^{-1}\nabla_z V$; it descends $V$. A learned policy may implement a different tangent field, but it should not be described as climbing the cost.

**Generation mode:** "I want to match this prompt." The prompt gets embedded as a direction in latent space. The policy points toward that direction, regardless of reward.

**Dreaming mode:** "I have no goal, just let me wander." The policy contributes nothing; only thermal noise picks a direction. This is like unguided imagination or free association.

The useful point is that all three are different parameterizations of a tangent control field. The geometry supplies the drift terms, while the interpretation of the control depends on whether it is minimizing cost, following conditioning, or set to zero.
:::

:::{prf:proposition} Radial-angular SDE under the $D=2$ overdamped specialization {cite}`strogatz2015nonlinear`
:label: thm-angular-symmetry-breaking

Assume $D=2$, $\alpha=1$, $\gamma_{\mathrm{risk}}=0$, $\mathcal F=0$, and absorb the constant friction into the time unit. In the **overdamped limit** of the second-order geodesic Langevin equation (Definition {prf:ref}`def-bulk-drift-continuous-flow`, Theorem {prf:ref}`thm-overdamped-limit`), the generation dynamics decompose into radial expansion and angular diffusion.

**Radial dynamics (monotonic expansion):** The radial coordinate $r = |z|$ satisfies:

$$
dr = \left[\frac{1-r^2}{2}+u_\pi^r+\frac{T_c(1-r^2)^2}{4r}\right]d\tau + \frac{1-r^2}{2}\sqrt{2T_c}\,dW_r

$$
with drift $\frac{1-r^2}{2} > 0$ for all $r \in [0,1)$. The origin is not a fixed point; the drift pushes trajectories outward (though stochastic fluctuations can temporarily reverse this at small $r$).

**Angular dynamics (symmetry breaking):** In polar coordinates $z = re^{i\theta}$, the angular evolution satisfies:

$$
d\theta = \frac{u_\pi^\theta}{r}\,d\tau + \frac{1-r^2}{2r}\sqrt{2T_c}\,dW_\theta

$$
where $u_\pi^\theta = u_\pi \cdot \hat{\theta}$ is the tangential component of the control field.

**Local direction-to-noise ratio:** A dimensionless local measure of angular drift relative to diffusion is

$$
\mathrm{Pe}_\theta^2(r) := \frac{2r^2|u_\pi^\theta|^2}{T_c(1-r^2)^2}.

$$
- If $\mathrm{Pe}_\theta\ll1$, angular diffusion dominates over the chosen time interval.
- If $\mathrm{Pe}_\theta\gg1$, the policy drift dominates locally.

*Proof.* Starting from the second-order geodesic Langevin equation (Definition {prf:ref}`def-bulk-drift-continuous-flow`) with the Poincaré metric $G_{ij} = \frac{4\delta_{ij}}{(1-r^2)^2}$, we take the stated overdamped specialization (Theorem {prf:ref}`thm-overdamped-limit`). The overdamped position SDE in Cartesian coordinates is:

$$
dz^k = -G^{kj}\partial_j U\, d\tau + u_\pi^k\, d\tau + \sqrt{2T_c}(G^{-1/2})^{kj}\,dW^j_\tau

$$
where $G^{-1/2} = \frac{1-r^2}{2}I$. Converting to polar coordinates via Itô's lemma:
- Radial: $dr = \langle dz, \hat{r}\rangle + \frac{1}{2}\text{tr}(\text{Hess}_r \cdot \Sigma)$ where $\Sigma = 2T_c G^{-1}$
- Angular: $d\theta = \langle dz, \hat{\theta}/r\rangle + \frac{1}{2}\text{tr}(\text{Hess}_\theta \cdot \Sigma)$

The Itô correction for the radial coordinate is nonzero: $\operatorname{tr}(\operatorname{Hess}r)=1/r$ in two dimensions, giving the displayed Bessel drift. The angular coordinate has the displayed diffusion coefficient and the tangential control component; no general cancellation justifies dropping these terms.

The ratio $\mathrm{Pe}_\theta$ is a local diagnostic. A finite-temperature diffusion on a compact angular fibre has a smooth stationary law; no phase transition or almost-sure freeze-out follows without an additional limiting argument. $\square$

:::

:::{div} feynman-prose
Let me explain what the radial and angular equations say in the $D=2$ overdamped specialization.

The deterministic entropic component is radial and points outward because the $D=2$ log-volume grows with radius. In the full radial SDE, policy and thermal terms are present as well; the deterministic entropic drift is positive on $0\le r<1$, but noise can temporarily move a sample inward near the origin.

The angular direction is where the control and noise compete. At the origin, all directions are equivalent by $SO(2)$ symmetry. Away from the origin, the local comparison is:
1. The **policy** $u_\pi^\theta$, which nudges the trajectory toward a preferred direction
2. **Thermal noise**, which randomizes the direction

Near the origin (small $r$), the polar angular noise coefficient scales as $1/r$. At larger radii, the relative strength still has to be evaluated using the local Péclet diagnostic $\mathrm{Pe}_\theta(r)$; it does not follow universally that the noise weakens or that a direction freezes in.

The scale $\mathrm{Pe}_\theta\approx1$ is a finite-time crossover convention. The finite-temperature angular diffusion has a smooth law on the compact angular fibre; the displayed equations do not establish a phase transition, a critical temperature, or almost-sure freeze-out.
:::

:::{admonition} Temperature and Generation Quality
:class: feynman-added warning

Temperature changes the relative size of the angular diffusion over the chosen horizon:
- **Lower temperature:** the local Péclet number can be larger, so a given policy bias may be easier to observe over a finite interval.
- **Higher temperature:** the same policy bias can be more strongly blurred by angular fluctuations.

These are qualitative finite-time comparisons. The proposition {prf:ref}`thm-angular-symmetry-breaking` supplies the radial and angular SDEs and the local diagnostic; it does not prove a phase transition, a critical temperature, or deterministic directional freeze-out.
:::

(pi-symmetry-breaking)=
::::{admonition} Physics Analogy: Direction Selection under Noise
:class: note

**In Physics:** Spontaneous symmetry breaking occurs when a system's ground state has lower symmetry than its Hamiltonian. The classic example is the Mexican hat potential $V(\phi) = -\mu^2|\phi|^2 + \lambda|\phi|^4$: for $\mu^2 > 0$, the $U(1)$-symmetric origin becomes unstable and the system selects a direction {cite}`goldstone1961field,weinberg1996qft`.

**In Implementation:** The angular dynamics are described by the proposition {prf:ref}`thm-angular-symmetry-breaking`:

$$
d\theta = \frac{u_\pi^\theta}{r}\,d\tau + \frac{1-r^2}{2r}\sqrt{2T_c}\,dW_\theta

$$
The local drift-to-noise diagnostic is $\mathrm{Pe}_\theta^2(r)=2r^2|u_\pi^\theta|^2/[T_c(1-r^2)^2]$; it compares two regimes without asserting a thermodynamic phase transition.

**Correspondence Table:**
| Phase Transition Theory | Agent (Policy Emergence) |
|:------------------------|:-------------------------|
| Order parameter $\phi$ | Angular direction $\theta$ |
| Control parameter | Policy strength $|u_\pi^\theta|$ |
| Crossover scale | $\mathrm{Pe}_\theta(r_*)\approx 1$ |
| Symmetric phase | Isotropic angular distribution |
| Broken phase | Policy-selected direction $\theta_\pi$ |
| Goldstone modes | Angular fluctuations in $\theta$ |

**Significance:** Policy selection is a finite-time drift-versus-noise effect in this model; calling it spontaneous symmetry breaking would require an additional angular potential and a limiting argument.
::::

:::{div} feynman-prose
The angular fluctuations are worth a moment's thought. A policy bias can make nearby headings more likely, while thermal noise keeps producing variations around them. Those variations are “different versions of the same idea,” but the present SDE does not derive Goldstone modes: there is no added angular potential and no proved spontaneous-symmetry-breaking limit.

This is why repeated samples can be related without being identical. They are trajectories of a noisy angular diffusion with a policy-dependent drift. The physics analogy is useful for naming the picture, but the rigorous statement here is the local drift-versus-noise comparison.
:::

**Algorithm 21.2.6 (Control Field Computation).**

```python
import torch
from typing import Literal, Optional


def poincare_metric_inv(z: torch.Tensor) -> torch.Tensor:
    """Compute inverse Poincare metric G^{-1}(z) = (1 - |z|^2)^2 / 4."""
    r_sq = (z ** 2).sum(dim=-1, keepdim=True)
    one_minus_r_sq = torch.clamp(1.0 - r_sq, min=1e-8)
    return (one_minus_r_sq ** 2) / 4.0


def compute_control_field(
    z: torch.Tensor,                    # [B, D] current position (near origin)
    mode: Literal["rl", "generation", "dreaming"],
    prompt_embed: Optional[torch.Tensor] = None,  # [B, D] for generation mode
    grad_V: Optional[torch.Tensor] = None,        # [B, D] critic gradient for RL
    T_c: float = 1.0,                   # Temperature (for dreaming mode)
) -> torch.Tensor:
    """
    Compute the control field u_pi that selects initial direction.

    Breaks SO(D) symmetry at the origin, unifying RL, generation,
    and dreaming into a single operation.

    Cross-ref: Theorem {prf:ref}`thm-unified-control-interpretation`
    """
    B, D = z.shape
    G_inv = poincare_metric_inv(z)  # [B, 1]

    if mode == "rl":
        # RL cost convention: descend V with the metric-preconditioned gradient
        if grad_V is None:
            raise ValueError("grad_V required for RL mode")
        u_pi = -G_inv * grad_V

    elif mode == "generation":
        # Generation: u_pi = G^{-1} * prompt_embed
        if prompt_embed is None:
            raise ValueError("prompt_embed required for generation mode")
        # Normalize prompt to unit vector, then scale by metric
        prompt_norm = prompt_embed / (torch.norm(prompt_embed, dim=-1, keepdim=True) + 1e-8)
        u_pi = G_inv * prompt_norm

    elif mode == "dreaming":
        # Dreaming: pure thermal fluctuation (no deterministic kick)
        # The noise in the Langevin dynamics will break symmetry
        u_pi = torch.zeros(B, D, device=z.device, dtype=z.dtype)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return u_pi
```

**Cross-references:** {ref}`Section 2.7 <sec-the-hjb-correspondence>` (HJB Correspondence), Section 14.2 (MaxEnt control equivalence).

::::{admonition} Connection to RL #24: Diffusion Policies as Degenerate Radial Generation
:class: note
:name: conn-rl-24
**The General Law (Fragile Agent):**
Data generation is **radial expansion** from the vacuum (origin) to the boundary:

$$
F_{\text{total}} = \underbrace{F_{\text{entropy}}}_{\text{Hyperbolic drift}} + \underbrace{u_\pi}_{\text{Policy kick}}

$$
where $F_{\text{entropy}} = \nabla_G S(z)$ is entropic drift from hyperbolic volume growth and, in the cost convention, $u_\pi = -G^{-1} \nabla_z V$ is one possible policy control field.

**The Degenerate Limit:**
Replace hyperbolic geometry with Euclidean ($G \to I$). Reverse the direction (boundary to origin). Remove value-based steering.

**The Special Case (Standard diffusion model / diffusion policy):**

$$
dz_t = s_\theta(z_t, t)\, dt + \sigma(t)\, dW_t, \quad z_T \sim \mathcal{N}(0, I)

$$
This recovers **Diffusion Models** {cite}`ho2020ddpm` and **Diffusion Policies** for robotic control.

**What the generalization offers:**
- **Hyperbolic structure**: Exponential volume growth provides natural hierarchy (Definition {prf:ref}`def-hyperbolic-volume-growth`)
- **Euclidean limit**: Removing the hyperbolic metric gives a time-reversed noise-to-data diffusion, not a boundary-to-origin statement in Euclidean space
- **Policy unification**: RL control and conditional generation share the same drift term $u_\pi$
- **Symmetry breaking**: Policy kicks at origin select generation mode (Theorem {prf:ref}`thm-angular-symmetry-breaking`)
::::

(sec-bulk-boundary-independence)=
## Bulk-Boundary Independence

:::{div} feynman-prose
Now I want to tell you about a separation that's absolutely crucial to how this whole thing works: the separation between the **bulk** (the interior of the disk, where planning happens) and the **boundary interface** (where the internal state is handed to observation or action).

Here's the key idea: when you're planning a trajectory---thinking about what to do, imagining futures, computing values---you're operating on the texture-free bulk projection $(K,z_n)$. You're moving around the interior of the latent space, figuring out which direction to go. During this planning phase, texture is excluded from the bulk equations, so the computation stays at the level of structure, causality, and value.

At the declared stopping interface---which is represented by a finite cutoff before the ideal boundary at infinite hyperbolic distance---the fine-grained details matter. That's where “texture” comes in: high-frequency variation is sampled for the pixels, tokens, or motor commands without feeding back into the bulk trajectory.

This separation has a useful consequence: the planning process can focus on the variables that matter for decision-making, while the interface handles a declared texture distribution. The partition condition is a statement about dependence of the dynamics; it is not a claim that the ideal boundary is reached in finite time.
:::

We strictly enforce the separation of **Planning** (interior $\text{int}(\mathcal{Z})$) and **Observation** (boundary $\partial\mathcal{Z}$). This is formalized as a partition condition.

*Remark (Motor Extension).* The independence constraint applies equally to the **motor/action boundary**: motor texture $z_{\text{tex,motor}}$ (tremor, fine motor noise) is sampled at the output interface and does not participate in planning. {ref}`Section 23.3 <sec-motor-texture-the-action-residual>` formalizes the motor texture distribution $z_{\text{tex,motor}} \sim \mathcal{N}(0, \sigma_{\text{motor}}^2 G^{-1}(z))$ with the same conformal scaling as visual texture. The firewall axiom supplies the partition condition; it does not assert a conjugation duality between the two covariance spaces.

:::{prf:axiom} Bulk-Boundary Decoupling
:label: ax-bulk-boundary-decoupling

The state decomposition $Z = (K, z_n, z_{\text{tex}})$ satisfies a **partition condition**. Write $\mathcal Z_{\mathrm{bulk}}:=\mathcal K\times\mathcal Z_n\subset\mathcal Z$ for the texture-free projection:

1. **Interior (Planning Domain):** The bulk projection evolves on $\mathcal Z_{\mathrm{bulk}}$ and contains no texture component. Planning depends only on geometry and topology:

$$
\Pi_{\mathrm{bulk}}\dot Z = f(\Pi_{\mathrm{bulk}}Z,u_\pi),\qquad \partial_{z_{\mathrm{tex}}}\Pi_{\mathrm{bulk}}\dot Z=0.

$$
2. **Boundary Interface:** Texture $z_{\text{tex}}$ is a stochastic component that exists **only** at the interface where the internal state meets the external observation:

$$
z_{\text{tex}} \sim \mathcal{N}(0, \Sigma(z_{\text{final}}))

$$
Formally, the partition condition is:

$$
\frac{\partial}{\partial z_{\text{tex}}} \left[ \dot{z}^k, \lambda_{\text{jump}}, u_\pi \right] = 0 \quad \forall \tau \in [0, \tau_{\text{stop}})

$$
:::

:::{admonition} The Partition in Plain Language
:class: feynman-added note

Think of it this way: there are two roles in one state decomposition, separated by a one-way interface.

**The Bulk (Planning):**
- Works with coarse-grained structure: "Is this a cat? Where is it going? Should I chase it?"
- Operates on $(K, z_n)$---discrete categories and continuous nuisance variables
- Uses the projection of the full state onto $(K,z_n)$; the global state still has a texture coordinate, but the bulk dynamics do not depend on it.

**The Boundary (Interface):**
- Deals with fine-grained details: "What exact shade of orange is the fur? What's the precise pixel pattern?"
- Samples $z_{\text{tex}}$ when the stopped internal state is handed to the decoder or action interface
- Stochastic: samples from a distribution, doesn't compute deterministically

The partition condition says that texture has no derivative in the bulk drift, jump rate, or policy field during the interior evolution. Planning doesn't see texture, and the sampled interface texture does not affect the already-computed planning path. That is the firewall being asserted.

Why does this matter? Because you can do complex planning without simulating every pixel to figure out whether to turn left or right. You simulate the structure, then apply the declared interface law when an output is requested.
:::

:::{prf:definition} Boundary Texture Distribution
:label: def-boundary-texture-distribution

At the terminal position $z_{\text{final}}$, texture is sampled from a **geometry-dependent** Gaussian:

$$
z_{\text{tex}} \sim \mathcal{N}\big(0,\, \Sigma(z_{\text{final}})\big),

$$
where the covariance matrix is:

$$
\Sigma(z) = \sigma_{\text{tex}}^2 \cdot G^{-1}(z) = \sigma_{\text{tex}}^2 \cdot \frac{(1-|z|^2)^2}{4} I.

$$
Units: $[\Sigma] = [z_{\text{tex}}]^2$.

:::

:::{div} feynman-prose
This is a lovely formula. The texture variance $\Sigma(z)$ scales with the conformal scalar inherited from $G^{-1}(z)$. What does that mean?

Near the origin ($|z| \approx 0$), $G^{-1} \approx 1/4$, so texture has moderate variance. The output is coarse-grained, uncertain, blurry.

Near the boundary ($|z| \to 1$), $G^{-1} \to 0$, so texture variance goes to zero. The output is fine-grained, precise, sharp.

This gives a smaller interface variance near the cutoff when the conformal factor is small. It is a prescribed sampling law for a separate texture space; it does not say that the texture coordinate is itself a tangent vector in the latent manifold.

The geometry supplies the position-dependent scale, while the base variance and the decoder still determine the actual output statistics.
:::

:::{prf:proposition} Conformal Texture Scaling
:label: prop-conformal-texture-scaling

The texture variance scales with the inverse metric:

| **Region** | **$\lvert z\rvert$** | **$\Sigma(z)$**                            | **Interpretation**        |
|------------|----------------------|--------------------------------------------|---------------------------|
| Origin     | $\approx 0$          | $\sigma_{\text{tex}}^2/4 \cdot I$          | Moderate texture (coarse) |
| Mid-disk   | $\approx 0.5$        | $\sigma_{\text{tex}}^2 \cdot 9/64 \cdot I$ | Reduced texture           |
| Boundary   | $\to 1$              | $\to 0$                                    | Deterministic texture     |

*Remark (Conformal suppression).* Near the boundary (high resolution/specificity), the metric $G$ diverges, so $G^{-1} \to 0$ and texture fluctuations are suppressed.

:::

:::{admonition} Why Conformal Scaling is the Right Choice
:class: feynman-added tip

You might wonder: why specifically $\Sigma \propto G^{-1}$? Why not some other scaling?

The answer in this construction is that the texture law borrows the conformal scalar from the inverse metric. If one were sampling a tangent vector at $z$, a Gaussian with covariance proportional to $G^{-1}(z)$ would be natural under a fixed expected squared $G(z)$-norm. Here $z_{\text{tex}}$ lives in a separate texture space, so only that scalar scaling is being transferred.

That observation motivates the scaling; it does not make it unique. The model declares the covariance in the definition, and the implementation must use that declared law.
:::

:::{prf:definition} Boundary Decoder
:label: def-boundary-decoder

The Decoder $\mathcal{D}$ is the **only** component that sees texture. It performs the **boundary synthesis**:

$$
x = \mathcal{D}(z_{\text{final}}, z_{\text{tex}})

$$
where:
- $z_{\text{final}} = (e_K, z_n)$: Determines the shape, physics, and causal structure
- $z_{\text{tex}}$: "Paints" the high-frequency details onto that structure

:::

:::{div} feynman-prose
I like to think of the decoder as a two-stage artist:

**Stage 1:** Draw the structure. Given $z_{\text{final}}$, the decoder knows *what* to draw---the bones, the shapes, the logical relationships. This is the "skeleton" of the output.

**Stage 2:** Paint the texture. Given $z_{\text{tex}}$, the decoder fills in the details---the colors, the patterns, the fine variations. This is the "skin" of the output.

Neither stage can function without the other. Structure without texture gives you a wireframe. Texture without structure gives you noise. The decoder combines them into a coherent output.

Notice that the planning system (which lives in the bulk) only affects Stage 1. It controls *what* gets generated but not the fine details of *how* it looks. This is the bulk-boundary decoupling in action.
:::

:::{admonition} Relation to BarrierEpi
:class: note

The partition condition keeps texture prediction out of the bulk, so it can reduce the capacity spent modelling $z_{\mathrm{tex}}$. This supports, but does not enforce, staying below **BarrierEpi**, whose formal failure condition is information overload in the VQ-VAE/world-model pair ({ref}`Section 3.2 <sec-limits-barriers>`).
:::

:::{admonition} The Wisdom of Not Trying Too Hard
:class: feynman-added note

This separation suggests an engineering choice: the bulk model need not predict a particular texture sample. The interface law specifies the statistics of texture, including how its variance changes with position; a decoder can then sample or model those statistics without feeding texture back into planning.

That does not make fine-grained variation universally unpredictable, and it does not enforce the separate information-overload barrier. It only says where the declared partition places the texture computation.

You see this pattern in biological systems too. Your visual system doesn't try to predict individual photon arrivals; it learns the statistics of images. Your motor system doesn't try to plan individual muscle fiber twitches; it plans movements and lets noise handle the details.

The partition condition formalizes this dependency boundary: structure enters the bulk dynamics, while texture enters through the output interface.
:::

:::{prf:definition} Stopping Criterion
:label: def-stopping-criterion

Given a declared generation horizon $\tau_{\max}$, the flow terminates when the radial coordinate exceeds a cutoff or the horizon is reached:

$$
\tau_{\text{stop}} := \min\!\left\{\tau_{\max},\ \inf\{\tau \ge 0 : |z(\tau)| \ge R_{\text{cutoff}}\}\right\}.

$$
Equivalently, in terms of the information depth $-U(z)=2\operatorname{artanh}|z|$ (Definition {prf:ref}`def-hyperbolic-information-potential`), stop when $-U(z)\ge C_{\mathrm{stop}}$ and choose $R_{\text{cutoff}}=\tanh(C_{\mathrm{stop}}/2)$. Taking $C_{\mathrm{stop}}\le C_\partial$ keeps the generated state within the declared boundary budget; this is an operational choice, not a consequence of the metric-law field equation.
In practice, choose $R_{\text{cutoff}} = 1 - \varepsilon$ with $\varepsilon$ tied to the Levin length/resolution ({ref}`Appendix A <sec-appendix-a-full-derivations>`). This is a
computational cutoff, not a terminal task boundary.

**Algorithm 21.3.7 (Boundary Texture Sampling).**

```python
import torch

def sample_boundary_texture(
    z_final: torch.Tensor,        # [B, D] final semantic position
    texture_dim: int,             # Dimension of texture space
    sigma_tex: float = 1.0,       # Base texture std dev
) -> torch.Tensor:
    """
    Sample texture with geometry-dependent variance.

    Implements Definition 21.3.2:
        z_tex ~ N(0, Sigma(z_final))
        Sigma(z) = sigma_tex^2 * G^{-1}(z)

    The partition condition (Axiom 21.3.1) ensures this is called
    ONLY at terminal time, not during interior dynamics.

    Cross-ref: Proposition 21.3.3 (Conformal Scaling)
    """
    B = z_final.shape[0]

    # Compute G^{-1}(z) = (1 - |z|^2)^2 / 4
    r_sq = (z_final ** 2).sum(dim=-1, keepdim=True)  # [B, 1]
    one_minus_r_sq = torch.clamp(1.0 - r_sq, min=1e-6)
    G_inv_scale = (one_minus_r_sq ** 2) / 4.0  # [B, 1]

    # Texture std = sigma_tex * sqrt(G^{-1})
    texture_std = sigma_tex * torch.sqrt(G_inv_scale)  # [B, 1]

    # Sample isotropic Gaussian, then scale
    z_tex = torch.randn(B, texture_dim, device=z_final.device)
    z_tex = z_tex * texture_std  # broadcast scaling

    return z_tex
```

:::

:::{div} feynman-prose
The stopping criterion $|z| \ge R_{\text{cutoff}}$ is your operational "resolution dial." Set $R_{\text{cutoff}}$ close to 1, and you allow a more committed state before invoking the boundary interface. Set it smaller, and you stop earlier, producing a coarser or more abstract output according to the decoder.

There's a tradeoff here: higher $R_{\text{cutoff}}$ means more computation (the trajectory takes longer to reach the boundary) but finer outputs. Lower $R_{\text{cutoff}}$ is faster but coarser. You tune this based on your application's needs.

For most generation tasks, you want $R_{\text{cutoff}}$ fairly high---maybe 0.95 or 0.99. For quick sketching or brainstorming, a lower value might suffice.
:::

(sec-summary-and-diagnostic-node)=
## Summary and Diagnostic Node

:::{div} feynman-prose
Let me step back and summarize what we've built in this section.

We have a picture of generation as **radial expansion in hyperbolic space**. In the $D=2$ specialization, the log-volume entropy $S=-U$ starts at its minimum at the origin and increases outward. A separate notion---residual output uncertainty---may decrease as a trajectory becomes more committed. The flow is driven by:

1. **Entropic drift:** The radial gradient of geometric log-volume, whose deterministic component points outward and has limiting magnitude $1/2$ at the origin, with no preferred direction there.

2. **Policy and noise:** Tangential and radial terms that select and adjust a direction. Their relative angular influence is measured locally by $\mathrm{Pe}_\theta$, not by a proved critical temperature.

The radial and angular equations displayed above are the $D=2$ overdamped specialization. At finite temperature they describe a smooth diffusion with a finite-time drift-versus-noise crossover; they do not establish a phase transition, pitchfork bifurcation, or almost-sure freeze-out.

The trajectory evolves on the **bulk projection** $(K,z_n)$, while texture is sampled at the **boundary interface** after the declared stopping rule. The partition condition prevents texture from entering the interior drift, jump rate, or policy field. RL and conditioned generation can use the same geometric decomposition, but their control parameterizations and cost interpretations remain distinct.
:::

**Summary of Radial Generation:**

| **Aspect**          | **Formula**                                   | **Units**            | **Reference**                                    |
|---------------------|-----------------------------------------------|----------------------|--------------------------------------------------|
| Entropic Drift      | $F_{\text{entropy}} = \frac{1-\lvert z\rvert^2}{2}\hat{z}$ | $[z]/\tau$           | Def {prf:ref}`def-the-entropic-force`            |
| Radial Expansion    | $r(\tau) = \tanh(\tau/2)$                     | dimensionless        | Prop {prf:ref}`prop-isotropic-radial-expansion`  |
| Control Field       | $u_\pi = G^{-1} \mathbb{E}[a]$                | $[z]/\tau$           | Def {prf:ref}`def-the-control-field`             |
| Partition Condition | $\partial_{z_{\text{tex}}} \dot{z} = 0$       | -                    | Axiom {prf:ref}`ax-bulk-boundary-decoupling`     |
| Texture Covariance  | $\Sigma(z) = \sigma_{\text{tex}}^2 G^{-1}(z)$ | $[z_{\text{tex}}]^2$ | Def {prf:ref}`def-boundary-texture-distribution` |
| Stopping            | $\lvert z\rvert \ge R_{\text{cutoff}}$        | dimensionless        | Def {prf:ref}`def-stopping-criterion`            |

(node-25)=
**Node 25: HoloGenCheck (radial generation check)**

| **#**  | **Name**           | **Component** | **Type**                | **Interpretation**       | **Proxy**                                                         | **Cost** |
|--------|--------------------|---------------|-------------------------|--------------------------|-------------------------------------------------------------------|----------|
| **25** | **HoloGenCheck** | **Generator** | **Generation Validity** | Did flow reach boundary? | $\mathbb{I}(\lvert z_{\text{final}}\rvert \ge R_{\text{cutoff}})$ | $O(B)$   |

**Trigger conditions:**
- Low RadialGenCheck: Generation terminated too early (insufficient specificity).
- Remedy: Increase the declared $\tau_{\text{max}}$ (preferred) or lower $R_{\text{cutoff}}$ at the cost of specificity.

**Cross-references:** {ref}`Section 2.2b <sec-the-shutter-as-a-vq-vae>` (VQ-VAE texture channel), {ref}`Section 7.10 <sec-decoder-architecture-overview-topological-decoder>` (TopologicalDecoder), {ref}`Section 18 <sec-capacity-constrained-metric-law-geometry-from-interface-limits>` (Capacity constraints).
