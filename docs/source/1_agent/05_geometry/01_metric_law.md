(sec-capacity-constrained-metric-law-geometry-from-interface-limits)=
# Capacity-Constrained Metric Law: Geometry from Interface Limits

## TLDR

- Finite boundary bandwidth implies a hard **capacity constraint**: the agent cannot stably maintain bulk information it
  cannot ground at the interface.
- Enforcing this constraint induces a **metric law**: curvature adapts as a regulator when representation approaches
  capacity saturation (a geometric form of “information bottleneck”).
- The law is not analogy-first: it is derived from information/variational structure and yields testable diagnostics.
- The practical output is a **consistency defect** and a runtime diagnostic for when the agent is claiming more
  information than its boundary can support.
- This chapter is the geometric backbone for later results: WFR belief geometry, holographic generation, and the causal
  information bound.

## Roadmap

1. State the capacity constraint and why it must hold for bounded agents.
2. Derive the metric/curvature response as a variational law.
3. Define diagnostics and implementation-facing consistency checks.

:::{div} feynman-prose
Here's a question that sounds almost silly at first: *Why should space be curved?*

In physics, Einstein gave us an answer: space curves because stuff is in it. Mass and energy bend spacetime, and that bending is what we call gravity. The geometry isn't handed down from on high---it emerges from the presence of matter.

Now, what's remarkable is that a related construction can be useful inside an agent. The agent has an internal
"space"---its latent representation, the manifold $\mathcal{Z}$ where it organizes its beliefs about the world.
Under the declared capacity postulate, regularity assumptions, and the variational boundary hypotheses, stationarity
of the chosen action gives a curvature equation. That is a conditional law; the data-processing inequality by itself
does not force every bounded representation to be curved or select a unique metric.

Think about it. The agent only sees the world through a finite-bandwidth channel---its sensors. It can only act through
finite-capacity motors. Everything it can ground about the world has to pass through this narrow boundary. If a chosen
model tries to maintain more grounded structure than the interface supports, a capacity diagnostic should fire; a
separate modelling choice may then adapt the metric or the representation.

This is the capacity-constrained metric law. Its resemblance to Einstein's equations is useful, but its conclusion is
the stationarity statement under the hypotheses above, not a universal physical necessity for every bounded agent.
:::

(rb-info-bottleneck-geometry)=
:::{admonition} Researcher Bridge: Information Bottleneck Becomes Geometry
:class: info
When you push a model to the edge of representational capacity, the geometry must adapt. This is the rigorous version of information bottleneck regularization: capacity limits induce curvature that slows updates in overloaded regions.
:::

{ref}`Section 9.10 <sec-differential-geometry-view-curvature-as-conditioning>` used a "gravity" analogy to motivate curvature as a regulator. This section removes the analogy: the curvature law is derived as a structural response to **information-theoretic constraints** induced by the agent's finite-bandwidth boundary (Markov blanket).

The key idea is operational: **the representational complexity of the internal state is bounded by the capacity of the interface channel.** When the agent operates near this bound (at its {prf:ref}`def-boundary-markov-blanket`), curvature appears as the geometric mechanism that prevents internal information volume from exceeding what can be grounded at the interface.

(sec-the-boundary-bulk-information-inequality)=
## The Boundary--Bulk Information Inequality

:::{div} feynman-prose
Let's start with a very simple observation that turns out to be profound.

Imagine you're an agent. You're sitting there in the world, taking in observations, making decisions. Inside your head (or your neural network, or whatever), you're building up a representation of what's going on. This representation is your internal state $Z$.

Now here's the thing: *everything you know has to come through your sensors*. There's no other way for information to get in. You can't just magically know things about the world---you have to see them, hear them, feel them. Every bit of information in your internal representation had to squeeze through your sensory boundary at some point.

This seems obvious, but it has a sharp mathematical consequence once we say exactly what information is being counted. If
your boundary channel has capacity $C_\partial$ (measured in nats per unit time, say), then mutual information between
the world and a state produced by that channel cannot exceed the corresponding channel budget. You might have more
*stuff* in the state---random noise, hallucinations, or prior structure---but that is not automatically grounded
information about the current world.

This is the data-processing inequality, and it's one of the deepest results in information theory. You cannot create
mutual information by applying a Markov map. The map, the source variable, and the observation window matter: the
inequality does not turn an arbitrary KL-to-reference score into a channel capacity automatically.

For our purposes, $I_{\text{bulk}} \le C_\partial$ is therefore a declared capacity constraint. It is a direct DPI
consequence only when $I_{\text{bulk}}$ has been defined as the relevant mutual information (or a justified rate
derived from it); the relative-information proxy used later must be calibrated to that constraint.
:::

:::{prf:definition} DPI / boundary-capacity constraint
:label: def-dpi-boundary-capacity-constraint

Consider the boundary stream $(X_t)_{t\ge 0}$ and the induced internal state process $(Z_t)_{t\ge 0}$ produced by the shutter (Definition {prf:ref}`def-bounded-rationality-controller`). Because all internal state is computed from boundary influx and internal memory, any information in the bulk must be mediated by a finite-capacity channel. Operationally, the data-processing constraint is:

$$
I_{\text{bulk}} \;\le\; C_{\partial},

$$
where $C_{\partial}$ is the effective information capacity of the boundary channel and $I_{\text{bulk}}$ is a declared grounded-information proxy for the internal state. The inequality is an operational capacity postulate; the data-processing inequality supplies it only after $I_{\text{bulk}}$ has been defined as mutual information with a boundary-history variable. Units are nats for a fixed observation window and nat/step for the corresponding rate.

:::

:::{div} feynman-prose
Now, what do we mean by "information in the bulk"? This is where things get interesting, because we need to be careful about what we're measuring.

When you have a probability distribution $\rho(z)$ over your latent space, there's a natural measure of how much information it carries: the differential entropy. But here's the subtlety---the entropy depends on what volume element you use. If you change your coordinates, the entropy changes.

This is where the metric $G$ comes in. The metric tells you how to measure volume in your latent space. And once you have a proper volume element, you can define information density in a coordinate-invariant way.

The formula might look a bit intimidating, but the idea is simple: we're counting how many nats of information the agent is carrying at each location, and we're doing it in a way that respects the geometry.
:::

:::{prf:definition} Information density and bulk information volume
:label: def-information-density-and-bulk-information-volume

Let $\rho(z,s)$ denote the probability density of the agent's belief state at position $z \in \mathcal{Z}$ and computation time $s$, **defined with respect to the Riemannian volume measure** $d\mu_G = \sqrt{|G|}\,dz^n$. Let $\rho_{\mathrm{ref}}(z,s)>0$ be a reference belief density with respect to the same measure. The **relative-information density** is

$$
\iota_{\mathrm{bulk}}(z,s) := \rho(z,s)\log\frac{\rho(z,s)}{\rho_{\mathrm{ref}}(z,s)},

$$
with units of nats per unit Riemannian volume ($n=\dim\mathcal{Z}$). The local integrand need not be non-negative, but its integral is a KL divergence.

*Remark.* The differential entropy $h_G[\rho]:=-\int \rho\log\rho\,d\mu_G$ is a separate coordinate-invariant quantity. The relative-information integral below is the non-negative proxy used for the capacity diagnostic.

:::

:::{prf:definition} Bulk Information Volume
:label: def-a-bulk-information-volume

Define the bulk information volume over a region $\Omega\subseteq\mathcal{Z}$ by

$$
I_{\text{bulk}}(\Omega) := \int_{\Omega} \iota_{\mathrm{bulk}}(z,s)\, d\mu_G
 = D_{\mathrm{KL}}\!\left(\rho\,\middle\|\,\rho_{\mathrm{ref}}\right)_{\Omega}.

$$
When $\Omega=\mathcal{Z}$ we write $I_{\text{bulk}}:=I_{\text{bulk}}(\mathcal{Z})$. This is conceptually distinct from the probability-mass balance in {ref}`Section 2.11 <sec-variance-value-duality-and-information-conservation>`; it measures deviation from the declared reference belief in nats.

:::

:::{div} feynman-prose
Now here's the really interesting part: the boundary. The agent's interface with the world has a certain "area"---not physical area necessarily, but informational area. Think of it as the total number of independent channels through which information can flow.

In many physical systems, there's a remarkable phenomenon called an **area law**: under the right state space,
cutoff, and observable, the amount of information can scale with boundary area rather than volume. This shows up in
black hole thermodynamics and in entanglement calculations. Here we use the same picture as one operational capacity
model; it is not a theorem that every agent's information obeys a physical area law.

Why can this be a useful model? If the interface is resolved at scale $\ell$, has area $A$, and each unit of area
represents $1/\eta_\ell$ nats, then the model assigns capacity $A/\eta_\ell$. The answer depends on that cutoff and
resolution; when they are not declared, use the channel-capacity definition instead of an area integral.

This is why capacity constraints can be coupled to geometry. The boundary area, measured in $G$, supplies one
capacity schedule, and a separate variational or control rule can adapt $G$ when the schedule is exceeded. The area
model alone does not prove that the metric must adjust.
:::

:::{prf:definition} Boundary capacity: area law at finite resolution
:label: def-boundary-capacity-area-law-at-finite-resolution

Let $\partial_{\varepsilon}\mathcal{Z}$ be an explicitly chosen cutoff hypersurface (for example $|z|=1-\varepsilon$ in the Poincaré chart), and let $dA_G$ be its induced $(n-1)$-dimensional area form. If the boundary interface has a minimal resolvable scale $\ell>0$ (pixel/token floor), then an operational capacity model is

$$
C_{\partial}(\partial_{\varepsilon}\mathcal{Z})
:=
\frac{1}{\eta_\ell}\oint_{\partial_{\varepsilon}\mathcal{Z}} dA_G,

$$
where $\eta_\ell$ is the effective boundary area-per-nat at resolution $\ell$ (a resolution-dependent constant set by the interface). The capacity is therefore cutoff-dependent; if no geometric cutoff is declared, use the interface-channel definition instead of an area integral.
Units: $[\eta_\ell]=[dA_G]/\mathrm{nat}$ and $[\ell]$ is the chosen boundary resolution length scale.

*Remark (discrete macro specialization).* For the split shutter, use distinct rate proxies over a declared observation window:

$$
C_{\partial}^{\mathrm{rate}}\ :=\ \sup_{p(x)} I(X;K)\ \le\ \log|\mathcal{K}|,

$$
while the realised inflow is $\lambda_{\mathrm{in}}=\mathbb{E}[I(X_t;K_t)]$ in nat/step (Definition {prf:ref}`def-grounding-rate`). Multiplying the rate by a window length converts it to a stock in nats; Node 13 monitors the realised inflow rather than identifying it with channel capacity.

:::

:::{admonition} Example: The Pixel Budget
:class: feynman-added example

Let's make this concrete. Suppose your agent sees the world through a 64x64 grayscale camera, and each pixel can take 256 values. What's the maximum information that can flow through this interface in one frame?

Naively, you might say $64 \times 64 \times \log(256) = 64 \times 64 \times 8 \approx 32,000$ bits. But that's almost never achieved in practice. Real images have strong correlations---neighboring pixels are usually similar. The *effective* information rate is much lower.

This is the $\eta_\ell$ factor: it accounts for the redundancy in your sensory channel. A highly compressed representation might achieve near the theoretical limit; a raw pixel stream wastes most of its bandwidth on predictable correlations.

The boundary capacity $C_\partial$ is what survives after all this redundancy is squeezed out. It's the *useful* information that actually constrains your internal representation.
:::

(sec-main-result)=
## Main Result (Capacity-Saturated Metric Law)

:::{div} feynman-prose
Alright, now we come to the main event. We've established a capacity inequality, and we can ask what a particular
variational model does as that inequality becomes tight.

Under the regularity, boundary, and on-shell hypotheses stated with the theorem, stationarity relates the curvature of
$G$ to a risk tensor. The metric need not change merely because a capacity number is large; adapting $G$ is the
response encoded by this chosen model and its training or control schedule.

Let me be clear about what's happening here. We're not doing physics. We're not saying the agent's latent space is
"actually" curved spacetime. What we have is a shared equation shape---a tensor built from curvature on one side and a
source on the other---obtained from different variational problems. The capacity postulate and the curvature equation
are related by the stated hypotheses; neither one is a substitute for the other.

The source term in our equation isn't the stress-energy tensor of matter. It's the "Risk Tensor"---a measure of how
much the agent cares about different regions of its state space. High-value regions can contribute more strongly when
the risk functional is the one used in the theorem, but the equation does not by itself prove a unique learned metric
or a capacity guarantee.
:::

The detailed variational construction is recorded in {ref}`Appendix A <sec-appendix-a-full-derivations>`. The main consequence is an Euler--Lagrange identity that ties curvature of the latent geometry to a risk-induced tensor under a finite-capacity boundary.

:::{prf:theorem} Capacity-constrained metric law
:label: thm-capacity-constrained-metric-law

Under the regularity and boundary-clamping hypotheses stated in {ref}`Appendix A <sec-appendix-a-full-derivations>`, stationarity of the curvature-plus-risk functional implies

$$
R_{ij} - \frac{1}{2}R\,G_{ij} + \Lambda G_{ij} = \kappa\, T_{ij},

$$
where $\Lambda$ and $\kappa$ are constants and $T_{ij}$ is the **total Risk Tensor** induced by the reward field. The equation is consistent with the value equation only when the value and (if present) curl fields are on shell for the same action; this is an explicit hypothesis, not a consequence of the boundary capacity. *Units:* $\Lambda$ has the same units as curvature ($[R]\sim [z]^{-2}$), and $\kappa$ is chosen so that $\kappa\,T_{ij}$ matches those curvature units.

*Operational reading.* The equality supplies a geometric consistency residual. Interpreting it as a mechanism that enforces $I_{\text{bulk}}\le C_{\partial}$ requires the separate capacity postulate (Definition {prf:ref}`def-dpi-boundary-capacity-constraint`) and is not derived by this variation.

**Implementation hook.** The squared residual of this identity defines the metric-law regularizer $\mathcal{L}_{\text{EFE}}$; use the metric-contracted norm specified in {ref}`Appendix F <sec-appendix-f-loss-terms-reference>`.

:::

:::{div} feynman-prose
Let me unpack this equation piece by piece, because it's the heart of this section.

On the left side, we have:
- $R_{ij}$: the Ricci tensor. It measures how the volumes of small geodesic balls deviate from the Euclidean value: positive Ricci curvature makes nearby geodesics converge and the balls smaller, while negative Ricci curvature makes them larger.
- $R$: the scalar curvature (the trace of $R_{ij}$). A single number summarizing the overall curvature at a point.
- $G_{ij}$: the metric tensor itself.
- $\Lambda$: a constant, analogous to the cosmological constant in physics. It sets a baseline curvature even when there's no "stuff" around.

On the right side:
- $\kappa$: a coupling constant. It controls how strongly the risk tensor sources curvature.
- $T_{ij}$: the Risk Tensor. This is where the agent's objectives enter the picture.

The equation says that, at a stationary point of the declared action, curvature and risk are tied together. Regions where
the agent is making high-stakes decisions can contribute strongly through the risk tensor, but the theorem does not
say that every such region must have a particular sign or amount of curvature.

There is an operational interpretation in which the fitted metric helps regulate representation near a capacity limit.
That interpretation needs the separate capacity postulate, a specified information proxy, and a coupling or update rule.
The curvature equation alone does not guarantee that information stays within budget, nor does it say that important
regions must be inflated while unimportant ones are deflated.
:::

:::{warning}
:class: feynman-added
A common mistake is to treat the displayed equation as a universal command that every bounded agent must obey. The
curvature is a stationary response of the declared variational model, after the capacity proxy, boundary data, and
regularity hypotheses have been fixed. A flat or off-shell representation can occur outside that regime; the right
response is to report the residual and check the hypotheses rather than infer ungrounded beliefs from curvature alone.
:::

:::{prf:definition} Extended Risk Tensor with Maxwell Stress
:label: def-extended-risk-tensor

The total Risk Tensor $T_{ij}$ decomposes into gradient and curl contributions:

$$
T_{ij} = T_{ij}^{\text{gradient}} + T_{ij}^{\text{Maxwell}},

$$
where:

1. **Gradient Stress** (from scalar potential $\Phi$):

$$
T_{ij}^{\text{gradient}} = \partial_i \Phi \, \partial_j \Phi - G_{ij}\left(\frac{1}{2}\|\nabla\Phi\|_G^2+U(\Phi)\right)

$$
2. **Maxwell Stress** (from {prf:ref}`def-value-curl` $\mathcal{F}$):

$$
T_{ij}^{\text{Maxwell}} = \alpha_{\mathcal F}\left(\mathcal{F}_{ik}\mathcal{F}_j^{\;k} - \frac{1}{4}G_{ij}\mathcal{F}^{kl}\mathcal{F}_{kl}\right)

$$
*Units:* $[T_{ij}^{\text{gradient}}] = \mathrm{nat}^2/[z]^2$ when $U$ has the same units as the gradient term. Under the volume's convention $[\mathcal F]=\mathrm{nat}/[z]^2$, choose $[\alpha_{\mathcal F}]=[z]^2$ so the Maxwell contribution has the same units. This is an optional action extension; the scalar derivation in Appendix A does not include it.

**Conservative Limit:** When $\mathcal{F} = 0$ (Definition {prf:ref}`def-conservative-reward-field`), the Maxwell term vanishes and we recover the standard gradient-only risk tensor.

**Non-Conservative Case:** When $\mathcal{F} \neq 0$, the Maxwell stress contributes additional terms to the curvature equation.

:::

:::{div} feynman-prose
The Risk Tensor has two pieces, and they have very different characters.

The first piece, $T_{ij}^{\text{gradient}}$, comes from the gradient of the value function. Where value changes rapidly---where you're on a steep slope in reward landscape---you get a large contribution. This makes intuitive sense: regions where small movements lead to big changes in value are "risky" and deserve extra geometric attention.

The second piece, $T_{ij}^{\text{Maxwell}}$, is more subtle. It comes from the *curl* of the reward field. In standard RL, we assume this is zero---rewards are conservative, and there's a well-defined scalar value function. But as we discussed in {ref}`Section 24 <sec-the-reward-field-value-forms-and-hodge-geometry>`, that's not always true. When the agent faces cyclic preferences (like Rock-Paper-Scissors) or when exploration-exploitation creates sustained orbits, the reward field has non-zero curl.

The Maxwell stress tells you how this cyclic structure contributes to curvature. It's called "Maxwell" because the formula is mathematically identical to the electromagnetic stress tensor in physics. In electromagnetism, this term describes how the presence of electromagnetic fields creates pressure and tension in spacetime. Here, it describes how cyclic value structures create geometric stress in the latent space.

For most practical purposes, you can ignore the Maxwell term---most reward functions are conservative. But when they're not, this term explains the geometric consequences.
:::

:::{admonition} Example: The Cliff Walk
:class: feynman-added example

Consider a classic "cliff walking" problem. The agent must traverse a path where one side is safe (low reward) and the other side is a cliff (large negative reward for falling off).

Near the cliff edge, the gradient of value may be large, so the Risk Tensor can be large when the stated
normalization and risk functional apply. If a fitted metric satisfies the metric-law residual and the controller uses
its inverse metric for descent, the resulting local distance or step can make the controller more cautious.

That is an operational scenario to test: compare geodesic and coordinate step lengths, then monitor capacity and
residual diagnostics. It is not an automatic consequence of a large value gradient, and the metric law alone does not
promise that every implementation will inflate the danger zone.
:::

(pi-einstein-equations)=
::::{admonition} Physics Isomorphism: Einstein Field Equations
:class: note

**In Physics:** Einstein's field equations relate spacetime curvature to stress-energy: $R_{\mu\nu} - \frac{1}{2}Rg_{\mu\nu} + \Lambda g_{\mu\nu} = 8\pi G T_{\mu\nu}$ {cite}`einstein1915field,wald1984general`.

**In Implementation:** The capacity-constrained metric law (Theorem {prf:ref}`thm-capacity-constrained-metric-law`) relates latent geometry to risk:

$$
R_{ij} - \frac{1}{2}R\,G_{ij} + \Lambda G_{ij} = \kappa T_{ij}

$$
**Correspondence Table:**

| General Relativity | Agent (Metric Law) |
|:-------------------|:-------------------|
| Spacetime metric $g_{\mu\nu}$ | Latent metric $G_{ij}$ |
| Ricci tensor $R_{\mu\nu}$ | Ricci tensor $R_{ij}$ (of $G$) |
| Cosmological constant $\Lambda$ | Baseline curvature $\Lambda$ |
| Stress-energy $T_{\mu\nu}$ | Risk tensor $T_{ij}$ |
| Gravitational coupling $8\pi G$ | Capacity coupling $\kappa$ |
| Schwarzschild horizon | Saturation horizon (Lemma {prf:ref}`lem-metric-divergence-at-saturation`) |

**Loss Function:** with $E_{ij}:=R_{ij} - \frac{1}{2}R\,G_{ij} + \Lambda G_{ij} - \kappa T_{ij}$, use the invariant residual
$\mathcal{L}_{\text{EFE}} := \int G^{ik}G^{jl}E_{ij}E_{kl}\,d\mu_G$ (or its minibatch approximation).
::::

:::{div} feynman-prose
I want to be careful about what this isomorphism means and what it doesn't mean.

It means that both settings contain a curvature-versus-source equation produced by a specified variational construction. In
general relativity, the Einstein tensor comes from the Einstein-Hilbert action together with the relevant dimension, matter, and
diffeomorphism assumptions. Optimizing an arbitrary metric-dependent functional does not automatically produce that tensor. Here
the risk source and metric equation come from the declared agent action and its own hypotheses.

It *doesn't* mean that the agent's latent space is "really" a spacetime, or that there's actual gravity involved. The physics is completely different. In GR, the metric is a property of the arena in which events occur. Here, the metric is a property of the agent's *representation*---it's a learned structure that organizes information efficiently.

But the structural similarity is useful. We can borrow intuitions: "mass curves spacetime" becomes "risk curves latent space."
We can borrow numerical techniques, while checking their assumptions for the learned metric. We can also use the singularities of
general relativity as a comparison for capacity-saturation signals; those signals are not general-relativistic singularities and
need the separate radial and coupling hypotheses stated in this volume.
:::

(sec-diagnostic-node-capacity-saturation)=
## Diagnostic Node: Capacity Saturation

:::{div} feynman-prose
How do you know when you're approaching the declared capacity? That's what this diagnostic monitors.

The idea is simple: compute a ratio of a chosen bulk-information proxy to a boundary-capacity estimate. A ratio near
one is a useful warning that the calibration is tight. A ratio above one flags a violation of the declared postulate
or a mismatch of units and definitions; it is not, by itself, proof that the agent knows something its sensors could
not have supplied.

When the ratio is small, the chosen proxy leaves headroom. When it approaches one, a metric or representation update
may be scheduled if the model specifies one. Think of the ratio as a gauge on an instrument panel: it tells you when to
inspect the channel, reference measure, and geometry, while the curvature residual supplies a separate check.

The balloon picture is helpful only at that level. A value above one means the declared gauge is out of range; whether
that reflects ungrounded beliefs, a KL proxy that is not a mutual information, or a capacity estimate with the wrong
window must be diagnosed.
:::

| ID | Name                    | Measures                        | Trigger                                         |
|----|-------------------------|---------------------------------|-------------------------------------------------|
| M-cap | CapacitySaturationCheck | Bulk-boundary information ratio | $I_{\text{bulk}} / C_{\partial} > 1 - \epsilon_H$ |

:::{prf:definition} Capacity saturation diagnostic
:label: def-capacity-saturation-diagnostic

Compute the capacity saturation ratio:

$$
\nu_{\text{cap}}(s) := \frac{I_{\text{bulk}}(s)}{C_{\partial}},

$$
where $I_{\text{bulk}}(s) = \int_{\mathcal{Z}} \iota_{\mathrm{bulk}}(z,s)\, d\mu_G$ per Definition {prf:ref}`def-a-bulk-information-volume`.

*Interpretation:*
- $\nu_{\text{cap}} \ll 1$: Under-utilized capacity; the agent may be compressing excessively (lossy representation).
- $\nu_{\text{cap}} \approx 1$: Operating at capacity limit; geometry must regulate to prevent overflow.
- $\nu_{\text{cap}} > 1$: **Violation** of the declared capacity postulate (Definition {prf:ref}`def-dpi-boundary-capacity-constraint`); this is a diagnostic flag, not a direct consequence of the DPI unless $I_{\text{bulk}}$ is a mutual information.

*Cross-reference:* When $\nu_{\text{cap}} > 1$, the curvature residual (Theorem {prf:ref}`thm-capacity-constrained-metric-law`) is insufficient. This triggers a representation reflow---reduce the KL-to-reference proxy through coarsening or stronger regularization, then re-evaluate the metric residual.

:::

:::{admonition} What "Geometric Reflow" Looks Like
:class: feynman-added tip

When the capacity diagnostic triggers, what can the agent do? "Reflow" is a name for a response schedule, not a
new theorem. One implementation might damp inverse-metric steps, another might coarsen the code, increase a declared
noise level, or change the reference model used for the information proxy.

Increasing $|G|$ can change Riemannian volumes and therefore the density convention, but it does not automatically
reduce a KL divergence: the reference density, coordinates, normalization, and dynamics must be transformed
consistently. After any such update, recompute the capacity ratio and curvature residual.

The useful picture is a controller reacting to an instrument reading. Each listed response is conditional on the
training objective and sampler, and none lets the agent exceed the channel. The point of reflow is to return to a
calibrated regime, not to assert that geometry alone has expanded the available information.
:::

::::{admonition} Connection to RL #25: Information Bottleneck as Degenerate Capacity-Constrained Metric
:class: note
:name: conn-rl-25
**The General Law (Fragile Agent):**
The latent metric obeys a **Capacity-Constrained Consistency Law** (Theorem {prf:ref}`thm-capacity-constrained-metric-law`):

$$
R_{ij} - \frac{1}{2}R\, G_{ij} + \Lambda G_{ij} = \kappa\, T_{ij}

$$
where $R_{ij}$ is Ricci curvature and $T_{ij}$ is the Risk Tensor. The operational constraint is the declared capacity inequality $I_{\text{bulk}} \le C_\partial$; the discrete shutter supplies the separate per-step bound $I(X;K)\le\log|\mathcal{K}|$.

**The Degenerate Limit:**
Remove geometric structure ($G \to I$, $R_{ij} \to 0$). Replace the area law with a scalar rate constraint $\beta$.

**The Special Case (Standard RL):**

$$
\max_\theta I(Z; Y) - \beta I(Z; X)

$$
This recovers the **Information Bottleneck** {cite}`tishby2015ib` and **Variational Information Bottleneck (VIB)** {cite}`alemi2016vib`.

**What the generalization offers:**
- **Geometric response**: Curvature *emerges* from capacity constraints---it's not imposed by hand
- **Area law**: Boundary capacity scales with interface area $C_\partial \sim \text{Area}(\partial\mathcal{Z})$, not arbitrary $\beta$
- **Grounded structure**: Bulk information must be mediated by finite-bandwidth boundary (DPI)
- **Diagnostic saturation**: the auxiliary CapacitySaturationCheck monitors $\nu_{\text{cap}} = I_{\text{bulk}}/C_\partial$ at runtime as global Node 40.
::::

:::{div} feynman-prose
Let me close this section by emphasizing what we've actually established.

We have a data-processing statement for a specified channel and mutual-information variable, plus an operational
capacity postulate for the relative-information proxy. We then varied a declared curvature-plus-risk action and obtained
a metric equation under explicit regularity, boundary, and on-shell hypotheses.

That equation resembles Einstein's field equation, but the resemblance is a mathematical correspondence. It does not
identify latent geometry with spacetime, and it does not by itself make curvature the mechanism that enforces capacity.

Finally, CapacitySaturationCheck is an instrument reading. A value above one asks us to check the proxy, units, window,
reference measure, and boundary model before interpreting it as a grounding failure. Reflow is then a conditional
control response, followed by recomputation of the diagnostics.
:::
