(sec-covariant-cross-attention-architecture)=
# Covariant Cross-Attention: The World Model as Geodesic Integrator

## TLDR

- Standard world models (GRU, Transformer) ignore geometric structure; we replace them with **Covariant Cross-Attention**
- Attention between observation and action latents uses **Wilson lines** for parallel transport, ensuring gauge invariance
- The softmax temperature encodes the **inverse metric**: $\tau(z) = \sqrt{d_k}/\lambda(z)$ (Mass = Metric)
- **Christoffel symbols** appear through quadratic Query terms $Q \ni (z \otimes z, z \odot v)$
- The $SU(2)_L$ chirality (sensor-motor asymmetry) is preserved via **chiral projectors**
- The $SU(N_f)_C$ texture firewall is enforced via **area law screening**
- Five attention heads implement the complete **Boris-BAOAB** geodesic integrator in one forward pass

## Roadmap

1. Why standard world models fail: the field equation solver perspective
2. Mathematical prerequisites: recap Lorentz-Langevin SDE and gauge structure
3. Covariant Cross-Attention with Wilson lines and gauge invariance
4. Metric encoded in temperature: softmax temp = inverse conformal factor
5. Christoffel symbols from quadratic Query terms
6. $SU(2)_L$ chirality: observation-action doublet with chiral projector
7. $SU(N_f)_C$ texture firewall: area law screening for confinement
8. BAOAB integration via 5 attention heads
9. Implementation: full GeodesicCrossAttention module
10. Computational complexity and O(N) engineering proxies
11. Diagnostics: Nodes 67-70 for gauge, temperature, chirality, confinement
12. Summary correspondence tables

:::{div} feynman-prose
Now we come to the practical question that should have been nagging at you since we derived all this beautiful gauge structure: How do you actually *implement* it? You have the Lorentz-Langevin equation, you have the gauge fields, you have the Wilson lines---but what goes into the neural network? What does the forward pass look like?

This chapter answers that question. We are going to take the abstract machinery of gauge-covariant dynamics and translate it into the language of attention mechanisms. The result is a world model that is not just approximately geometric, but *exactly* gauge-covariant by construction.

Here is the key insight: a Transformer attention head is secretly a parallel transport operation. When you compute attention between queries and keys, you are asking "how related are these two representations?" But "related" depends on how you measure distance. And measuring distance on a gauge bundle requires parallel transport along connecting paths.

The standard Transformer ignores this completely. It computes $\text{softmax}(QK^T/\sqrt{d_k})$ as if the latent space were flat Euclidean space with a global coordinate system. But we know that is wrong. The latent space is curved, with curvature determined by the metric law. And different parts of the space may be using different local frames, related by gauge transformations.

The fix is Wilson lines. Instead of directly comparing $Q$ at position $z$ with $K$ at position $z'$, you first parallel-transport $K$ from $z'$ to $z$ along a connecting path, then compare. This is gauge-covariant by construction: if you change the local frame at either endpoint, the Wilson line absorbs the transformation.

But there is more. The softmax temperature $\sqrt{d_k}$ in standard attention is a fixed constant. We are going to show that it should be the *inverse conformal factor* of the metric: $\tau(z) = \sqrt{d_k}/\lambda(z)$. This means the attention becomes sharper in high-curvature regions and softer in low-curvature regions---exactly what you want for safe exploration.

And the geodesic correction? The Christoffel symbols? Those come from quadratic terms in the Query projection. The standard linear $Q = W_Q z$ becomes $Q = W_Q z + W_{QQ}(z \otimes z)$, and the quadratic coefficient encodes the connection.

The result is a single attention module that performs one complete step of the Boris-BAOAB integrator. Five heads for B-A-O-A-B, with the Keys acting as a force bank and the Values as state updates. It is gauge-covariant, symplectic, and thermodynamically consistent---all from attention.
:::

*Abstract.* This chapter derives the **Covariant Cross-Attention** architecture for the world model, replacing standard sequence models (GRU, Transformer) with a mechanism that respects the gauge structure $G_{\text{Fragile}} = SU(N_f)_C \times SU(2)_L \times U(1)_Y$ derived in {ref}`Section 34 <sec-standard-model-cognition>`. The architecture implements a single-step integrator for the Lorentz-Langevin geodesic equations ({ref}`Section 22 <sec-the-equations-of-motion-geodesic-jump-diffusion>`) using Wilson lines for parallel transport, position-dependent temperature for metric encoding, and quadratic Query projections for Christoffel symbols. The result is a world model that is gauge-invariant by construction rather than by regularization.

*Cross-references:* This chapter synthesizes:
- {ref}`Section 22 <sec-the-equations-of-motion-geodesic-jump-diffusion>` (Lorentz-Langevin SDE, Boris-BAOAB integrator)
- {ref}`Section 34 <sec-standard-model-cognition>` (Gauge-theoretic formulation, covariant derivatives)
- {ref}`Section 18 <sec-capacity-constrained-metric-law-geometry-from-interface-limits>` (Capacity-constrained metric)
- {ref}`Section 23 <sec-the-boundary-interface-symplectic-structure>` (Dirichlet/Neumann boundary conditions)
- {ref}`Section 33 <sec-symplectic-multi-agent-field-theory>` (Ghost Interface, retarded potentials)



(sec-why-standard-world-models-fail)=
## Why Standard World Models Fail

:::{div} feynman-prose
Let me tell you what is wrong with the way world models are usually built, and why it matters.

A world model is supposed to predict what happens next: given the current state $z_t$ and action $a_t$, what is the next state $z_{t+1}$? The standard approach is to use a recurrent network like a GRU, or more recently, a Transformer. You feed in the state and action, do some matrix multiplications and nonlinearities, and out comes a prediction.

But think about what that matrix multiplication is doing. When you compute $W z$, you are treating $z$ as a vector in a flat Euclidean space. The matrix $W$ applies the same linear transformation everywhere, regardless of where in latent space you are.

This is wrong for two reasons.

First, the latent space is not flat. It has curvature determined by the metric law (Theorem {prf:ref}`thm-capacity-constrained-metric-law`). Near the boundary of the representable region, the metric diverges and steps become tiny. In high-uncertainty regions, the metric inflates and motion slows. A flat-space linear transformation cannot capture this.

Second, the latent space has gauge structure. Different parts of the network may be using different local coordinate frames, related by gauge transformations. When you compare representations from different locations---which is exactly what attention does---you need to account for this. A direct comparison of $z$ at one location with $z'$ at another is not gauge-invariant.

The result is a world model that makes predictions that violate the geometric constraints of the problem. It might predict states outside the representable region. It might break the symmetries that should be preserved. It might fail to conserve the symplectic structure that makes long-horizon planning possible.

We can do better. We are going to build a world model that respects the geometry by construction.
:::

The standard approach to world modeling treats the latent space $\mathcal{Z}$ as a flat vector space with global coordinates. This violates the geometric structure established in previous chapters.

:::{prf:proposition} Failure Modes of Flat World Models
:label: prop-failure-modes-flat-world-models

A world model $f: \mathcal{Z} \times \mathcal{A} \to \mathcal{Z}$ implemented as a standard neural network (GRU, MLP, Transformer) with flat-space operations fails to preserve:

1. **Metric structure**: The capacity-constrained metric $G(z)$ from Theorem {prf:ref}`thm-capacity-constrained-metric-law` implies position-dependent step sizes. Flat operations use constant step sizes.

2. **Gauge covariance**: Under local gauge transformation $\psi \to U(z)\psi$, predictions must transform covariantly. Flat operations are not gauge-aware.

3. **Symplectic structure**: The phase space $(\mathcal{Z} \times T^*\mathcal{Z}, \omega)$ has a conserved 2-form. Flat operations generically break symplectic conservation.

4. **Boundary constraints**: The Poincare disk metric diverges as $|z| \to 1$, preventing boundary crossing. Flat operations can predict invalid states.

*Consequence*: Flat world models require extensive regularization to approximately enforce these constraints, with no guarantee of exact satisfaction.

:::

:::{div} feynman-prose
You might object: "Can't we just add regularization terms to encourage these properties?" Yes, you can, and people do. But regularization is a crutch. It fights the architecture instead of working with it.

When you regularize, you are saying "please try to satisfy this constraint, here is a penalty if you do not." The optimizer finds approximate solutions that balance the penalty against other objectives. Sometimes it works, sometimes it does not. You never get exact constraint satisfaction.

The right approach is to build the constraints into the architecture itself, so violations are *impossible* rather than merely *discouraged*. This is what gauge-covariant architectures achieve.
:::

(rb-world-model-as-integrator)=
:::{admonition} Researcher Bridge: World Models as Numerical Integrators
:class: info
Standard world models learn a generic function $z_{t+1} = f(z_t, a_t)$ without structure. But we know the dynamics---they are the Lorentz-Langevin SDE (Definition {prf:ref}`def-bulk-drift-continuous-flow`). The world model should be a *numerical integrator* for this SDE, respecting its geometric structure. This perspective shifts the design problem from "learn any function" to "implement a gauge-covariant integrator."
:::



(sec-mathematical-prerequisites-architecture)=
## Mathematical Prerequisites

:::{div} feynman-prose
Before we build the architecture, let me collect the key mathematical ingredients we need. These all come from earlier chapters, but it helps to see them together.

The dynamics are governed by the Lorentz-Langevin equation. The agent moves on the latent manifold $(\mathcal{Z}, G)$, pulled by gradients, pushed by noise, and deflected by the Lorentz force from the value curl. The Christoffel symbols keep the motion on the manifold---they are the "geodesic correction" that accounts for curvature.

The gauge structure comes from three symmetries: $U(1)_Y$ for utility baseline, $SU(2)_L$ for observation-action mixing, and $SU(N_f)_C$ for feature binding. Each symmetry has an associated gauge field: the Opportunity field $B_\mu$, the Error field $W_\mu^a$, and the Binding field $G_\mu^a$. To compare quantities at different locations, we need parallel transport via Wilson lines.

The metric encodes capacity and risk. In the Poincare disk model, the conformal factor $\lambda(z) = 2/(1-|z|^2)$ diverges at the boundary. High-curvature regions have high effective mass---the agent moves slowly there.

These ingredients will map onto attention components as follows:
- Wilson lines $\to$ Key/Query preprocessing
- Metric $\to$ Temperature scaling
- Christoffel symbols $\to$ Quadratic Query terms
- Gauge fields $\to$ Position-dependent projections
:::

We collect the mathematical structures that will be implemented in the attention mechanism.

:::{prf:definition} Lorentz-Langevin SDE (Recap)
:label: def-lorentz-langevin-recap

From Definition {prf:ref}`def-bulk-drift-continuous-flow`, the position coordinates evolve as:

$$
dz^k = \underbrace{\left( -G^{kj}\partial_j \Phi + u_\pi^k \right)}_{\text{gradient + control}} ds + \underbrace{\beta_{\text{curl}} G^{km} \mathcal{F}_{mj} \dot{z}^j ds}_{\text{Lorentz force}} - \underbrace{\Gamma^k_{ij}\dot{z}^i \dot{z}^j ds}_{\text{geodesic correction}} + \underbrace{\sqrt{2T_c}(G^{-1/2})^{kj} dW^j_s}_{\text{thermal noise}}
$$

where:
- $G^{kj}$ is the inverse metric (Theorem {prf:ref}`thm-capacity-constrained-metric-law`)
- $\Phi$ is the effective potential (Definition {prf:ref}`def-effective-potential`)
- $\mathcal{F}_{mj}$ is the Value Curl tensor (Definition {prf:ref}`def-value-curl`)
- $\Gamma^k_{ij}$ are Christoffel symbols of the Levi-Civita connection
- $T_c$ is the cognitive temperature ({prf:ref}`def-cognitive-temperature`)

:::

:::{prf:definition} Covariant Derivative (Recap)
:label: def-covariant-derivative-recap

From Theorem {prf:ref}`thm-emergence-opportunity-field`, the gauge-covariant derivative for the full gauge group $G_{\text{Fragile}} = SU(N_f)_C \times SU(2)_L \times U(1)_Y$ is:

$$
D_\mu = \partial_\mu - i g_s \frac{\lambda^a}{2} G_\mu^a - i g_2 \frac{\tau^b}{2} W_\mu^b - i g_1 \frac{Y}{2} B_\mu
$$

where:
- $G_\mu^a$ ($a = 1, \ldots, N_f^2-1$) is the Binding field (Theorem {prf:ref}`thm-emergence-binding-field`)
- $W_\mu^b$ ($b = 1, 2, 3$) is the Error field (Theorem {prf:ref}`thm-emergence-error-field`)
- $B_\mu$ is the Opportunity field (Theorem {prf:ref}`thm-emergence-opportunity-field`)
- $\lambda^a, \tau^b$ are generators of $SU(N_f)$ and $SU(2)$ respectively
- $g_s, g_2, g_1$ are coupling constants
- $Y$ is the hypercharge

:::

:::{prf:definition} Wilson Line
:label: def-wilson-line

The **Wilson line** (parallel transport operator) along a path $\gamma$ from $z_0$ to $z$ is:

$$
U_\gamma(z, z_0) = \mathcal{P}\exp\left(-i\int_\gamma A_\mu dx^\mu\right)
$$

where:
- $\mathcal{P}$ denotes path ordering
- $A_\mu = g_s \frac{\lambda^a}{2} G_\mu^a + g_2 \frac{\tau^b}{2} W_\mu^b + g_1 \frac{Y}{2} B_\mu$ is the total gauge connection

For infinitesimal paths $\gamma: z_0 \to z_0 + \delta z$:

$$
U(z_0 + \delta z, z_0) \approx I - i A_\mu(z_0) \delta z^\mu + O(\delta z^2)
$$

*Gauge transformation*: Under $\psi(z) \to \Omega(z)\psi(z)$, the Wilson line transforms as:

$$
U_\gamma(z, z_0) \to \Omega(z) U_\gamma(z, z_0) \Omega^\dagger(z_0)
$$

This ensures that $U_\gamma(z, z_0) \psi(z_0)$ transforms correctly at $z$.

:::

:::{div} feynman-prose
The Wilson line is the key to gauge-covariant comparison. If you want to compare a vector at $z$ with a vector at $z_0$, you cannot just subtract them---that is not gauge-invariant. Instead, you parallel-transport the vector at $z_0$ to $z$ using the Wilson line, then compare. The result is independent of which gauge you choose at each point.

In our attention mechanism, we will use Wilson lines to preprocess Keys and Queries before comparison. The Query at position $z$ will be compared with Keys that have been parallel-transported *to* $z$ from their original positions. This ensures the attention scores are gauge-invariant.
:::

:::{prf:definition} Poincare Disk Metric (Recap)
:label: def-poincare-metric-recap

The capacity-constrained metric on the Poincare disk $\mathbb{D}^d = \{z \in \mathbb{R}^d : |z| < 1\}$ is:

$$
G_{ij}(z) = \lambda(z)^2 \delta_{ij} = \frac{4}{(1-|z|^2)^2} \delta_{ij}
$$

where $\lambda(z) = 2/(1-|z|^2)$ is the **conformal factor**.

**Key properties**:
- As $|z| \to 1$: $\lambda(z) \to \infty$ (metric diverges at boundary)
- At origin $z = 0$: $\lambda(0) = 2$ (minimal metric)
- Inverse metric: $G^{ij}(z) = \lambda(z)^{-2} \delta^{ij} = \frac{(1-|z|^2)^2}{4} \delta^{ij}$

The **Christoffel symbols** for this metric are (Proposition {prf:ref}`prop-a-explicit-christoffel-symbols-for-poincar-disk`):

$$
\Gamma^k_{ij}(z) = \frac{2}{1-|z|^2}\left(\delta^k_i z_j + \delta^k_j z_i - \delta_{ij} z^k\right)
$$

:::



(sec-covariant-cross-attention-definition)=
## Covariant Cross-Attention with Wilson Lines

:::{div} feynman-prose
Now we build the architecture. The standard attention mechanism computes:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

This has three problems from our perspective:
1. The comparison $QK^T$ is not gauge-covariant
2. The temperature $\sqrt{d_k}$ is position-independent
3. There is no geodesic correction

We fix all three. The gauge covariance comes from Wilson line preprocessing. The position-dependent temperature comes from the metric. The geodesic correction comes from quadratic Query terms.

Let me walk you through each component.

For gauge covariance, consider what happens when you compute $Q(z) \cdot K(z')$. The Query is defined at position $z$, the Key at position $z'$. If we are in different gauges at these two positions, the dot product is meaningless. We need to parallel-transport the Key from $z'$ to $z$ before comparing:

$$
Q(z) \cdot [U(z, z') K(z')]
$$

This is gauge-invariant: if we transform the gauge at $z'$, the Key transforms, but so does the Wilson line, and they cancel.

For the temperature, remember that the metric encodes how "heavy" different regions are. In high-curvature regions, the agent should move cautiously, which means sharp attention (low temperature). In low-curvature regions, the agent can explore freely, which means soft attention (high temperature). The natural choice is $\tau(z) = \sqrt{d_k}/\lambda(z)$: temperature inversely proportional to the conformal factor.

For the geodesic correction, we need the Christoffel symbols. These tell you how straight lines curve on the manifold. In standard attention, the Query is linear in the input: $Q = W_Q z$. To include Christoffel symbols, we add a quadratic term: $Q = W_Q z + W_{QQ}(z \otimes z)$. The quadratic coefficient encodes the connection.
:::

We define the gauge-covariant attention mechanism that respects the full gauge structure.

:::{prf:definition} Covariant Query-Key-Value Projections
:label: def-covariant-qkv-projections

Let $\psi_{\text{obs}}(z)$ be the observation latent and $\psi_{\text{act}}(z')$ be the action latent at positions $z, z' \in \mathcal{Z}$. The **covariant projections** are:

$$
\begin{aligned}
Q(z) &= \Pi_Q \cdot U_{0 \to z} \cdot D_\mu \psi_{\text{obs}}(z) \\
K(z') &= \Pi_K \cdot U_{0 \to z'} \cdot D_\nu \psi_{\text{act}}(z') \\
V(z') &= \Pi_V \cdot U_{0 \to z'} \cdot \psi_{\text{act}}(z')
\end{aligned}
$$

where:
- $U_{0 \to z}$ is the Wilson line from origin to position $z$ (Definition {prf:ref}`def-wilson-line`)
- $D_\mu$ is the covariant derivative (Definition {prf:ref}`def-covariant-derivative-recap`)
- $\Pi_Q, \Pi_K, \Pi_V$ are learnable projection matrices

**Interpretation**:
- The Wilson line $U_{0 \to z}$ parallel-transports the field to a common reference frame at the origin
- The covariant derivative $D_\mu$ ensures the derivative is gauge-covariant
- Queries use the derivative $D_\mu \psi$ (sensitivity to position change)
- Values use the field $\psi$ directly (the actual content to retrieve)

:::

:::{prf:theorem} Gauge Invariance of Covariant Cross-Attention
:label: thm-gauge-invariance-cross-attention

Let the attention score be computed as:

$$
\alpha(z, z') = \text{softmax}_{z'}\left(\frac{Q(z)^T K(z')}{\tau(z)}\right)
$$

where $Q$ and $K$ are defined with Wilson line preprocessing (Definition {prf:ref}`def-covariant-qkv-projections`), transporting both to a common reference point (the origin).

Then $\alpha(z, z')$ is **gauge-invariant**: under local gauge transformation $\psi \to \Omega(x)\psi$, the attention score is unchanged.

*Proof.*

**Step 1.** Under gauge transformation $\psi(x) \to \Omega(x)\psi(x)$, the Wilson line transforms as (Definition {prf:ref}`def-wilson-line`):

$$
U_{0 \to z} \to \Omega(0) U_{0 \to z} \Omega^\dagger(z)
$$

**Step 2.** The covariant derivative transforms covariantly:

$$
D_\mu \psi(z) \to \Omega(z) D_\mu \psi(z)
$$

**Step 3.** The Query at $z$ transforms as:

$$
Q(z) = \Pi_Q U_{0 \to z} D_\mu \psi(z) \to \Pi_Q \cdot \Omega(0) U_{0 \to z} \Omega^\dagger(z) \cdot \Omega(z) D_\mu \psi(z)
$$

The $\Omega^\dagger(z) \Omega(z) = I$ factors cancel:

$$
Q(z) \to \Pi_Q \Omega(0) U_{0 \to z} D_\mu \psi(z) = \Omega(0) Q(z)
$$

where the last equality holds because $\Pi_Q$ acts on feature indices while $\Omega(0)$ acts on gauge indices (they commute).

**Step 4.** Similarly, the Key transforms as:

$$
K(z') \to \Omega(0) K(z')
$$

**Step 5.** The attention score is the inner product at the origin:

$$
Q(z)^T K(z') \to [\Omega(0) Q]^T [\Omega(0) K] = Q^T \Omega(0)^\dagger \Omega(0) K = Q^T K
$$

The $\Omega(0)^\dagger \Omega(0) = I$ cancellation occurs because both Q and K have been transported to the same reference point.

*Remark (Why no additional Wilson line).* Since Q and K are already transported to the origin via Definition {prf:ref}`def-covariant-qkv-projections`, their inner product is well-defined without an additional Wilson line. The gauge transformation at the common reference point cancels automatically.

$\square$

:::

:::{div} feynman-prose
This theorem is the foundation of the architecture. It says that no matter how you choose your local coordinate frames---no matter what gauge you work in---the attention scores come out the same. The Wilson lines act as "gauge correctors" that absorb all the arbitrary choices.

In practice, we approximate the Wilson line for nearby points. For short paths, the path-ordered exponential simplifies to:

$$
U(z, z') \approx I - i A_\mu(z)(z - z')^\mu + O(|z - z'|^2)
$$

This is a linear correction that can be implemented efficiently.
:::

:::{prf:proposition} Wilson Line Approximation for Attention
:label: prop-wilson-line-approximation

For attention between positions $z$ and $z'$ with $|z - z'| \ll 1$, the Wilson line can be approximated as:

$$
U(z, z') \approx I - i \sum_a \Theta^a(z) \cdot (z - z')
$$

where $\Theta^a(z) = g_s \lambda^a G(z) + g_2 \tau^a W(z) + g_1 Y B(z)$ are the **connection coefficients** contracted with the path direction.

In the attention mechanism, this becomes a **relative position encoding**:

$$
\text{score}(z, z') = Q(z)^T K(z') - i Q(z)^T \Theta(z) (z - z')^T K(z')
$$

The second term is the **gauge correction** to the attention score.

:::

:::{admonition} Connection to RL #35: Gauge-Covariant World Models
:class: note
:name: conn-rl-35

**The General Law (Fragile Agent):**
World model predictions use **covariant cross-attention** with Wilson line preprocessing in Q, K, V:

$$
\hat{z}_{t+1} = \sum_{z' \in \text{context}} \alpha(z_t, z') \cdot V(z')
$$

where $\alpha = \text{softmax}(Q^T K / \tau(z_t))$ and Q, K, V include Wilson line transport to a common origin (Definition {prf:ref}`def-covariant-qkv-projections`).

**The Degenerate Limit:**
Set all gauge fields to zero ($G_\mu = W_\mu = B_\mu = 0$). Wilson lines become identity.

**The Special Case (Standard RL):**
Standard Transformer world models compute:

$$
\hat{z}_{t+1} = \text{softmax}(QK^T/\sqrt{d_k}) \cdot V
$$

No gauge structure, no parallel transport, no position-dependent temperature.

**What the generalization offers:**
- Gauge-invariant predictions regardless of local coordinate choice
- Metric-aware temperature scaling for safe exploration
- Automatic geodesic correction via quadratic Query terms

:::



(sec-metric-in-temperature)=
## The Metric in Temperature: Softmax Temperature = Inverse Conformal Factor

:::{div} feynman-prose
Here is one of the most elegant correspondences in this whole construction. The softmax temperature in attention---that innocuous $\sqrt{d_k}$ in the denominator---is actually the *inverse metric*.

Think about what temperature does in a softmax. High temperature means soft attention: all positions get similar weight, the distribution is spread out. Low temperature means sharp attention: one position dominates, the distribution is peaked.

Now think about what the metric does in curved space. In high-curvature regions (large metric), distances are "stretched"---small coordinate changes correspond to large proper distances. Motion is difficult. The agent should be cautious, focused, concentrated on nearby states. This is low temperature: sharp attention.

In low-curvature regions (small metric), distances are "compressed"---large coordinate changes correspond to small proper distances. Motion is easy. The agent can explore freely, considering many options. This is high temperature: soft attention.

The natural identification is: temperature = inverse conformal factor. Where the metric is large, temperature is small. Where the metric is small, temperature is large.

For the Poincare disk, this gives $\tau(z) = \sqrt{d_k}/\lambda(z) = \sqrt{d_k}(1-|z|^2)/2$. At the origin (where $|z|=0$), temperature is $\sqrt{d_k}/2$---soft attention with exploration. Near the boundary (as $|z| \to 1$), temperature goes to zero---attention becomes infinitely sharp, focusing on a single state. This prevents the agent from making predictions outside the representable region.
:::

We derive the position-dependent temperature that encodes the capacity-constrained metric.

:::{prf:theorem} Metric-Temperature Correspondence
:label: thm-metric-temperature-correspondence

Let the attention mechanism use position-dependent temperature $\tau(z)$:

$$
\alpha(z, z') = \text{softmax}_{z'}\left(\frac{Q(z)^T K(z')}{\tau(z)}\right)
$$

The choice

$$
\tau(z) = \frac{\sqrt{d_k}}{\lambda(z)} = \sqrt{d_k} \cdot \frac{1-|z|^2}{2}
$$

where $\lambda(z) = 2/(1-|z|^2)$ is the conformal factor, ensures:

1. **Metric encoding**: Temperature is inversely proportional to the local metric scale
2. **Boundary regularization**: $\tau(z) \to 0$ as $|z| \to 1$, preventing boundary crossing
3. **Boltzmann consistency**: The attention distribution approximates $\exp(-E/T_c)$ where $E$ is the geodesic distance

*Proof.*

**Step 1 (Metric encoding).** The attention score $s = Q^T K / \tau$ has units of inverse temperature. For the Boltzmann interpretation, this should be energy divided by temperature. The "energy" of comparing $z$ to $z'$ is proportional to the geodesic distance, which scales with $\lambda$. Setting $\tau \propto 1/\lambda$ ensures the ratio is scale-invariant.

**Step 2 (Boundary regularization).** As $|z| \to 1$:

$$
\tau(z) = \sqrt{d_k} \cdot \frac{1-|z|^2}{2} \to 0
$$

The attention becomes infinitely sharp: $\text{softmax}(s/\tau) \to \delta_{z' = z^*}$ where $z^*$ is the argmax. This prevents the weighted average of Values from producing states outside the boundary.

**Step 3 (Boltzmann consistency).** The geodesic distance from $z$ to $z'$ in the Poincare disk is:

$$
d_G(z, z') = 2 \tanh^{-1}\left(\frac{|z - z'|}{|1 - z\bar{z}'|}\right) \approx \lambda(z) |z - z'| + O(|z-z'|^2)
$$

for nearby points. If $Q^T K$ encodes negative squared coordinate distance $-|z-z'|^2$, then:

$$
\frac{Q^T K}{\tau} \propto \frac{-|z-z'|^2}{\sqrt{d_k}/\lambda} = -\frac{\lambda |z-z'|^2}{\sqrt{d_k}}
$$

This gives $\alpha \propto \exp(-\lambda |z-z'|^2 / \sqrt{d_k})$, which concentrates attention on nearby points with effective width $\propto 1/\sqrt{\lambda}$. Near the boundary where $\lambda \to \infty$, attention becomes infinitely localized.

$\square$

:::

:::{prf:proposition} Mass = Metric = Inverse Temperature
:label: prop-mass-metric-inverse-temperature

The Mass = Metric principle (Definition {prf:ref}`def-mass-tensor`) extends to attention:

$$
\mathbf{M}(z) = G(z) = \lambda(z)^2 I = \frac{d_k}{\tau(z)^2} I
$$

**Implication**: High mass $\Leftrightarrow$ Large metric $\Leftrightarrow$ Low temperature $\Leftrightarrow$ Sharp attention.

The attention mechanism becomes a position-dependent softmax where the effective inverse temperature is the metric:

$$
\alpha(z, z') \propto \exp\left(-\mathbf{M}(z) \cdot \text{dist}^2(z, z') / d_k\right)
$$

:::

:::{div} feynman-prose
This gives us a unified picture. The metric, which encodes capacity and risk, determines three things simultaneously:
1. How heavy the agent feels (inertial mass)
2. How cautious the attention is (inverse temperature)
3. How localized the predictions are (attention sharpness)

All from the same geometric quantity. This is not a coincidence---it is forced by consistency. If you want the world model to respect the geometry, the temperature must be the inverse metric.
:::

(pi-temperature-metric)=
::::{admonition} Physics Isomorphism: Temperature and Metric
:class: note

**In Physics:** In statistical mechanics, the partition function is $Z = \int e^{-E/k_B T} d\Gamma$. The temperature controls the spread of the Boltzmann distribution. In curved spacetime, the metric affects the integration measure and effective temperature.

**In Implementation:** The attention mechanism implements:

$$
\alpha \propto \exp\left(-\frac{d_G^2(z, z')}{2\tau(z)}\right)
$$

**Correspondence Table:**

| Statistical Mechanics | Covariant Attention |
|:---------------------|:---------------------|
| Temperature $T$ | Softmax temperature $\tau(z)$ |
| Energy $E$ | Attention score $-Q^T K$ |
| Partition function $Z$ | Softmax normalizer |
| Boltzmann weight $e^{-E/T}$ | Attention weight $\alpha$ |
| Metric factor $\sqrt{g}$ | Conformal factor $\lambda(z)$ |
::::



(sec-christoffel-in-query)=
## Christoffel Symbols in Query: Quadratic Projections for Geodesic Correction

:::{div} feynman-prose
Now we encode the geodesic correction into the Query projection. This is the most subtle part of the construction.

In the Lorentz-Langevin equation, the Christoffel symbol term is $-\Gamma^k_{ij} \dot{z}^i \dot{z}^j$. This is *quadratic* in the velocity. It tells you how much to "curve" your path to stay on the geodesic.

In standard attention, the Query is linear: $Q = W_Q z$. This cannot capture the quadratic correction. We need to add a quadratic term: $Q = W_Q z + W_{QQ}(z \otimes z)$.

What should $W_{QQ}$ be? It should encode the Christoffel symbols. Specifically, when you contract the quadratic term with the velocity (which appears in the Key), you should get the geodesic correction.

The implementation is:

$$
Q_{\text{geodesic}}(z, v) = W_Q z + W_{Qv} v + W_{QQ}(z \otimes z) + W_{Qzv}(z \odot v)
$$

where $v$ is the current velocity (or momentum). The $z \otimes z$ term gives the position-dependent curvature correction, and the $z \odot v$ term (Hadamard product) couples position and velocity as needed for the Christoffel contraction.

This looks complicated, but it is just the minimum structure needed to encode the geodesic equation. The Christoffel symbols $\Gamma^k_{ij}$ are symmetric in $i, j$, which is captured by the symmetric outer product $z \otimes z$. The velocity dependence $\dot{z}^i \dot{z}^j$ comes from the Key, which encodes the action/velocity information.
:::

We derive the quadratic Query projection that encodes Christoffel symbols for geodesic correction.

:::{prf:definition} Geodesic Query Projection
:label: def-geodesic-query-projection

The **Geodesic Query** extends the linear projection to include quadratic terms that encode the Levi-Civita connection:

$$
Q_{\text{geo}}(z, v) = W_Q z + W_{Qv} v + W_{Q,\Gamma}(z, z) + W_{Qv,\Gamma}(z, v)
$$

where:
- $W_Q \in \mathbb{R}^{d_k \times d}$ is the standard linear projection
- $W_{Qv} \in \mathbb{R}^{d_k \times d}$ projects the velocity/momentum
- $W_{Q,\Gamma} \in \mathbb{R}^{d_k \times d \times d}$ is a 3-tensor encoding position-position quadratic terms
- $W_{Qv,\Gamma} \in \mathbb{R}^{d_k \times d \times d}$ encodes position-velocity coupling

**Notation**: $(A, B)$ denotes bilinear contraction: $W_{Q,\Gamma}(z, z) = \sum_{ij} W_{Q,\Gamma}^{a,ij} z^i z^j$.

**Christoffel Encoding**: The tensor $W_{Q,\Gamma}$ should satisfy:

$$
W_{Q,\Gamma}^{a,ij} \propto \Gamma^a_{ij}(z_0)
$$

at a reference point $z_0$, with learnable corrections for position dependence.

:::

:::{prf:theorem} Geodesic Correction via Attention
:label: thm-geodesic-correction-attention

Let the Query and Key be:

$$
\begin{aligned}
Q(z, v) &= W_Q z + W_{Qv} v + W_{Q,\Gamma}(z, z) \\
K(z', v') &= W_K z' + W_{Kv} v'
\end{aligned}
$$

The attention-weighted output

$$
\Delta z = \sum_{z'} \alpha(z, z') V(z')
$$

contains the geodesic correction term $-\Gamma^k_{ij} v^i v^j$ when:

1. The Value encodes velocity updates: $V = W_V v'$
2. The quadratic Query contracts with velocity Keys: $W_{Q,\Gamma}(z, z) \cdot W_{Kv} v' = \Gamma^k_{ij}(z) z^i z^j \cdot v'^k$

*Proof sketch.* The attention score is:

$$
s(z, z') = Q(z, v)^T K(z', v') = [W_Q z + W_{Qv} v + W_{Q,\Gamma}(z,z)]^T [W_K z' + W_{Kv} v']
$$

The cross-term $W_{Q,\Gamma}(z,z)^T W_{Kv} v'$ is bilinear in $z$ and linear in $v'$. When Values are velocity-like and the projection weights encode Christoffel symbols, the output recovers the geodesic correction. $\square$

:::

:::{prf:proposition} Explicit Christoffel Encoding for Poincare Disk
:label: prop-christoffel-encoding-poincare

For the Poincare disk with Christoffel symbols (Proposition {prf:ref}`prop-a-explicit-christoffel-symbols-for-poincar-disk`):

$$
\Gamma^k_{ij}(z) = \frac{2}{1-|z|^2}\left(\delta^k_i z_j + \delta^k_j z_i - \delta_{ij} z^k\right)
$$

The quadratic Query tensor $W_{Q,\Gamma}$ should encode:

$$
W_{Q,\Gamma}^{k,ij}(z) = -\frac{2}{1-|z|^2}\left(\delta^k_i z_j + \delta^k_j z_i - \delta_{ij} z^k\right)
$$

**Learnable approximation**: Since $\Gamma$ depends on position, we parameterize:

$$
W_{Q,\Gamma}^{k,ij}(z) = \tilde{W}_{Q,\Gamma}^{k,ij} \cdot \frac{2}{1-|z|^2}
$$

where $\tilde{W}$ is a learnable constant tensor initialized to approximate the Poincare Christoffel structure.

:::

:::{div} feynman-prose
This is a bit technical, so let me give you the intuition. The Christoffel symbols tell you "when I'm at position $z$ and moving with velocity $v$, how much do I need to curve my path to follow the geodesic?" The answer depends quadratically on both position and velocity.

In attention, the Query asks a question and the Key provides answers. The quadratic Query term is asking: "what is the geodesic correction at my current position?" The Key, which contains velocity information, provides the answer through the bilinear interaction.

The nice thing is that this is all learnable. We initialize $W_{Q,\Gamma}$ to the known Christoffel structure, but then let the network fine-tune it based on data. If the Poincare model is exactly right, the network will keep the initialization. If reality is more complex, the network can learn corrections.
:::



(sec-su2-chirality-architecture)=
## $SU(2)_L$ Chirality: Observation-Action Doublet and Chiral Projector

:::{div} feynman-prose
Now we implement the chiral structure---the fundamental asymmetry between observation and action.

Recall from Section 34 that the left-handed field is a doublet:

$$
\Psi_L = \begin{pmatrix} \psi_{\text{obs}} \\ \psi_{\text{act}}^{\text{pre}} \end{pmatrix}
$$

This doublet transforms under $SU(2)_L$, while the right-handed singlet $\Psi_R = \psi_{\text{act}}^{\text{commit}}$ does not.

In the attention mechanism, we need to preserve this structure. The observation and action channels must be treated as a doublet, with the $SU(2)$ gauge field mediating their mixing. And the committed action must be extracted via a gauge-covariant projection.

The chiral projector uses the value gradient to define the "direction" of action:

$$
\vec{n}(z) = \frac{\nabla V}{\|\nabla V\|} \cdot \vec{\tau}
$$

where $\vec{\tau}$ are the Pauli matrices. The projection operator is:

$$
\Pi_{\text{chirality}} = \frac{1}{2}(I + \vec{n} \cdot \vec{\tau})
$$

This projects the doublet onto the component aligned with the value gradient---the direction of improvement.

In flat regions where $\nabla V \approx 0$, the projector becomes ambiguous. This is correct: when there is no preferred direction, the agent should not commit to a definite action. The architecture naturally encodes decision ambiguity as projector degeneracy.
:::

We implement the $SU(2)_L$ gauge structure that distinguishes observation and action channels.

:::{prf:definition} Observation-Action Doublet in Attention
:label: def-observation-action-doublet-attention

The attention mechanism operates on **doublet-valued** representations:

$$
\Psi_L(z) = \begin{pmatrix} \psi_{\text{obs}}(z) \\ \psi_{\text{act}}^{\text{pre}}(z) \end{pmatrix} \in \mathbb{C}^{2d}
$$

The **Query** extracts observation information, the **Key** encodes action information, both with Wilson line transport to origin (cf. Definition {prf:ref}`def-covariant-qkv-projections`):

$$
\begin{aligned}
Q_{\text{obs}}(z) &= \Pi_{\text{obs}} \cdot U_{0 \to z} \cdot D_\mu \Psi_L(z) = \begin{pmatrix} 1 & 0 \end{pmatrix} U_{0 \to z} D_\mu \Psi_L \\
K_{\text{act}}(z') &= \Pi_{\text{act}} \cdot U_{0 \to z'} \cdot D_\nu \Psi_L(z') = \begin{pmatrix} 0 & 1 \end{pmatrix} U_{0 \to z'} D_\nu \Psi_L
\end{aligned}
$$

The **cross-attention** score between observation and action is (gauge-invariant by Theorem {prf:ref}`thm-gauge-invariance-cross-attention`):

$$
\alpha_{\text{cross}}(z, z') = \text{softmax}\left(\frac{Q_{\text{obs}}(z)^T K_{\text{act}}(z')}{\tau(z)}\right)
$$

*Interpretation*: The observation at $z$ attends to actions at $z'$. Both are transported to a common reference point, ensuring gauge-invariant comparison.

:::

:::{prf:definition} Chiral Projector from Value Gradient
:label: def-chiral-projector-value-gradient

The **chiral projector** extracts committed actions from the observation-action doublet using the value gradient:

$$
\vec{n}(z) = \frac{\nabla V(z)}{\|\nabla V(z)\|} \cdot \vec{\tau}
$$

where $\vec{\tau} = (\tau_1, \tau_2, \tau_3)$ are Pauli matrices (generators of $SU(2)$).

The **projection operator** is:

$$
\Pi_{\text{chirality}}(z) = \frac{1}{2}\left(I_2 + \vec{n}(z) \cdot \vec{\tau}\right)
$$

The **committed action** is:

$$
\psi_{\text{act}}^{\text{commit}}(z) = \Pi_{\text{chirality}}(z) \cdot \Psi_L(z)
$$

**Properties**:
- $\Pi_{\text{chirality}}^2 = \Pi_{\text{chirality}}$ (idempotent)
- $\text{Tr}(\Pi_{\text{chirality}}) = 1$ (rank-1 projector)
- Under $SU(2)$ transformation $\Psi_L \to U\Psi_L$: $\vec{n} \to U\vec{n}U^\dagger$, preserving gauge covariance

**Degeneracy**: When $\|\nabla V\| \to 0$ (flat value landscape), $\vec{n}$ is undefined. The agent should not commit in ambiguous regions.

:::

:::{prf:theorem} Gauge Covariance of Chiral Projection
:label: thm-gauge-covariance-chiral-projection

The committed action scalar $|\psi_{\text{act}}^{\text{commit}}|^2 = \Psi_L^\dagger \Pi_{\text{chirality}} \Psi_L$ is invariant under local $SU(2)_L$ transformations. The projected vector $\Pi_{\text{chirality}} \Psi_L$ transforms covariantly (in the fundamental representation), but gauge-invariant observables are obtained by tracing over $SU(2)$ indices.

*Proof.*

**Step 1.** Under local $SU(2)_L$ transformation $\Psi_L(z) \to U(z)\Psi_L(z)$, where $U(z) \in SU(2)$.

**Step 2.** The value gradient $\nabla V$ transforms as a covariant vector. The unit vector $\vec{n}$ transforms under the adjoint representation:

$$
\vec{n}(z) \to U(z) \vec{n}(z) U^\dagger(z)
$$

**Step 3.** The projector transforms as:

$$
\Pi_{\text{chirality}} \to U \Pi_{\text{chirality}} U^\dagger
$$

**Step 4.** The committed action:

$$
\psi_{\text{act}}^{\text{commit}} \to (U \Pi U^\dagger)(U \Psi_L) = U \Pi \Psi_L
$$

This transforms in the fundamental representation, not as a singlet.

**Step 5.** To obtain a true singlet, we need to trace over the $SU(2)$ indices:

$$
\psi_{\text{act}}^{\text{commit,scalar}} = \text{Tr}_{SU(2)}(\Pi_{\text{chirality}} \Psi_L) = \Psi_L^\dagger \Pi_{\text{chirality}} \Psi_L
$$

This is manifestly gauge-invariant: $\Psi_L^\dagger U^\dagger \cdot U \Pi U^\dagger \cdot U \Psi_L = \Psi_L^\dagger \Pi \Psi_L$.

$\square$

:::

:::{div} feynman-prose
There is a subtlety here that is worth clarifying. The projection $\Pi \Psi_L$ itself is not a gauge singlet---it still transforms under $SU(2)$. To get a true invariant, we need to contract the gauge indices, which is what the trace accomplishes.

In the neural network implementation, this means the "committed action" is computed as an inner product, not just a projection. The observation component $\psi_{\text{obs}}$ and action component $\psi_{\text{act}}^{\text{pre}}$ are combined through the projector to yield a scalar output.

This has a nice interpretation: the committed action is the "correlation" between observation and action intent, weighted by the value gradient direction. High correlation in the gradient direction means strong commitment. Low correlation or ambiguous gradient means weak commitment.
:::



(sec-sunf-firewall-architecture)=
## $SU(N_f)_C$ Texture Firewall: Area Law Screening for Confinement

:::{div} feynman-prose
The final gauge structure we implement is the texture firewall: the confinement mechanism that prevents raw features from leaking to the macro level.

In QCD, quarks are confined by the strong force. If you try to pull a quark out of a proton, the energy cost grows linearly with distance (the "string" between quarks stretches and stores energy). Eventually it is cheaper to create new quark-antiquark pairs than to keep stretching the string. You never see a free quark.

We want the same behavior for texture. The agent should only observe *concepts* (bound states of features), not raw features. If attention tries to access texture directly, the attention score should be suppressed---screened by an area-law factor.

The implementation uses the string tension $\sigma$ from the binding field. The attention score between positions $z$ and $z'$ is modified by:

$$
\alpha_{\text{screened}} = \alpha \cdot \exp(-\sigma \cdot A_{\text{string}})
$$

where $A_{\text{string}}$ is the area of the minimal surface bounded by the path from $z$ to $z'$. For nearby points, this is approximately the squared distance times the string tension.

At the macro level (coarse features), the string tension is large, so screening is strong. At the texture level (fine features), the string tension is small (asymptotic freedom), so features can interact freely within the texture layer---they just cannot propagate to the macro level.
:::

We implement the $SU(N_f)_C$ confinement mechanism that screens texture from macro-level access.

:::{prf:definition} Area Law Screening in Attention
:label: def-area-law-screening-attention

The **screened attention score** between positions $z$ and $z'$ at representation level $\ell$ is:

$$
\alpha_{\text{screened}}(z, z'; \ell) = \alpha_{\text{bare}}(z, z') \cdot \exp\left(-\sigma(\ell) \cdot A_{\text{string}}(z, z')\right)
$$

where:
- $\alpha_{\text{bare}}$ is the gauge-covariant attention score from Theorem {prf:ref}`thm-gauge-invariance-cross-attention`
- $\sigma(\ell)$ is the **string tension** at level $\ell$, with $\sigma(\ell) \propto g_s^2(\ell)$ (binding coupling squared)
- $A_{\text{string}}(z, z')$ is the **minimal string area** connecting $z$ and $z'$

**Area approximation**: For nearby points in flat metric:

$$
A_{\text{string}}(z, z') \approx \frac{1}{2}|z - z'|^2
$$

For the Poincare disk with conformal factor $\lambda$:

$$
A_{\text{string}}(z, z') \approx \frac{\lambda(z)^2}{2}|z - z'|^2
$$

:::

:::{prf:theorem} Texture Confinement via Area Law Screening
:label: thm-texture-confinement-area-law

Let the representation hierarchy have levels $\ell = 0$ (macro) to $\ell = L$ (texture), with running coupling $g_s(\ell)$ satisfying asymptotic freedom (Definition {prf:ref}`def-coupling-function`):

$$
g_s(\ell) \to 0 \text{ as } \ell \to L \quad (\text{UV, texture level})
$$

$$
g_s(\ell) \to g_s^{\text{crit}} \text{ as } \ell \to 0 \quad (\text{IR, macro level})
$$

Then:

1. **Texture-to-texture attention** ($\ell = L$): $\sigma(L) \to 0$, no screening. Features interact freely at texture level.

2. **Macro-to-texture attention** ($\ell = 0$ attending to $\ell = L$): $\sigma(0) > \sigma_{\text{crit}}$, strong screening. Texture is inaccessible from macro level.

3. **Bound state access**: Color-neutral bound states (concepts) have zero net string charge. Their attention is not screened.

*Proof.*

**Step 1.** The string tension is proportional to the coupling squared: $\sigma(\ell) = c \cdot g_s^2(\ell)$.

**Step 2.** At texture level, asymptotic freedom gives $g_s(L) \to 0$, hence $\sigma(L) \to 0$. The screening factor $\exp(-\sigma A) \to 1$.

**Step 3.** At macro level, infrared confinement gives $g_s(0) > g_s^{\text{crit}}$, hence $\sigma(0) > \sigma_{\text{crit}}$. For any non-trivial area $A > 0$, the screening factor $\exp(-\sigma_{\text{crit}} A) \ll 1$.

**Step 4.** Color-neutral states have canceling string charges. The total area contribution vanishes: $A_{\text{total}} = 0$. No screening for bound states.

$\square$

:::

:::{prf:proposition} Confinement Radius from String Tension
:label: prop-confinement-radius-string-tension

The **confinement radius** $r_{\text{conf}}$ is the scale at which screening suppresses attention by factor $e^{-1}$:

$$
r_{\text{conf}}(\ell) = \sqrt{\frac{2}{\sigma(\ell)}}
$$

At macro level with $\sigma(0) \approx 1$: $r_{\text{conf}}(0) \approx \sqrt{2}$ (order-unity in latent units).

At texture level with $\sigma(L) \approx 0.01$: $r_{\text{conf}}(L) \approx 14$ (large, allowing texture-to-texture interaction).

*Interpretation*: Features at the texture level can "see" far across the texture manifold. But they cannot "see" into the macro level, because the string tension at the interface is high.

:::

:::{div} feynman-prose
This area law screening is exactly what we need for the texture firewall. The architecture automatically prevents macro-level attention from accessing texture details. You do not need to manually mask out texture channels---the screening factor does it for you.

And notice that this is not a hard cutoff. The screening is exponential in area, so nearby texture *can* influence macro attention, but only weakly. Long-range texture correlations are completely blocked. This matches the physics intuition: you can sometimes see texture details if you look closely, but the overall macro prediction does not depend on texture noise.
:::

(pi-area-law)=
::::{admonition} Physics Isomorphism: Area Law and Confinement
:class: note

**In Physics:** In QCD, the potential between quarks grows linearly with distance: $V(r) = \sigma r$, where $\sigma \approx 1$ GeV/fm is the string tension. Wilson loops satisfy the **area law**: $\langle W(\gamma) \rangle \propto \exp(-\sigma \cdot A)$ where $A$ is the area enclosed by loop $\gamma$ {cite}`wilson1974confinement`.

**In Implementation:** The screened attention weight is:

$$
\alpha_{\text{screened}} = \alpha \cdot \exp(-\sigma \cdot A_{\text{string}})
$$

**Correspondence Table:**

| QCD | Covariant Attention |
|:----|:---------------------|
| String tension $\sigma$ | Binding coupling squared $g_s^2$ |
| Wilson loop area $A$ | String area $A_{\text{string}}$ |
| Quark confinement | Texture firewall |
| Color-neutral hadrons | Bound-state concepts $K$ |
| Asymptotic freedom | Texture-level weak coupling |
::::



(sec-baoab-integration-architecture)=
## BAOAB Integration via Five Attention Heads

:::{div} feynman-prose
Now we put it all together. The Boris-BAOAB integrator has five steps: B-A-O-A-B. We are going to implement each step as an attention head.

Why five heads? Because each step of BAOAB does something different:
- **B** (kick): Apply forces from the potential gradient
- **A** (drift): Move along the geodesic
- **O** (thermostat): Add thermal noise
- **A** (drift): Continue moving
- **B** (kick): Apply more forces

Each step needs to attend to different information. The kick steps need the potential gradient (stored in Keys). The drift steps need the current velocity (stored in Values). The thermostat step needs noise (injected as a random Value).

The Keys act as a "force bank." They store the gradients of the effective potential at various positions. When the Query asks "what force do I feel here?", the Key provides the answer through the attention score.

The Values act as "state updates." After computing the attention weights, the weighted sum of Values gives the update to position or momentum.

The temperature of each head is position-dependent, encoding the metric. And the Wilson lines ensure gauge covariance throughout.

The result is a single attention block with five heads that performs one complete step of the geodesic integrator. Stack multiple blocks for multi-step rollouts.
:::

We implement the Boris-BAOAB integrator (Definition {prf:ref}`def-baoab-splitting`) using five specialized attention heads.

:::{prf:definition} BAOAB Attention Heads
:label: def-baoab-attention-heads

The **GeodesicCrossAttention** module contains five attention heads corresponding to the Boris-BAOAB splitting:

**Head 1: B-step (First half-kick)**
- **Query**: Current position $z$ with quadratic geodesic terms
- **Key**: Gradient bank $\{\nabla\Phi(z')\}_{z' \in \text{context}}$
- **Value**: Force vectors $\{-\nabla\Phi(z')\}$
- **Output**: Momentum update $\Delta p_1 = -\frac{h}{2}\nabla\Phi(z)$

**Head 2: A-step (First half-drift)**
- **Query**: Current position and momentum $(z, p)$
- **Key**: Exponential map bank $\{\exp_{z'}(v)\}$
- **Value**: Displacement vectors
- **Output**: Position update $\Delta z_1 = \frac{h}{2}G^{-1}(z)p$

**Head 3: O-step (Ornstein-Uhlenbeck thermostat)**
- **Query**: Current momentum $p$
- **Key**: Noise bank (random vectors)
- **Value**: Thermalized momenta $\{c_1 p + c_2 G^{1/2}\xi\}$
- **Output**: Momentum thermalization $p \to c_1 p + c_2 \xi'$

**Head 4: A-step (Second half-drift)**
- Same structure as Head 2
- **Output**: Position update $\Delta z_2 = \frac{h}{2}G^{-1}(z)p$

**Head 5: B-step (Second half-kick)**
- Same structure as Head 1
- **Output**: Momentum update $\Delta p_2 = -\frac{h}{2}\nabla\Phi(z)$

**Composition**: The full update is:

$$
(z_{t+1}, p_{t+1}) = \text{Head}_5 \circ \text{Head}_4 \circ \text{Head}_3 \circ \text{Head}_2 \circ \text{Head}_1(z_t, p_t)
$$

:::

:::{prf:theorem} BAOAB Attention Preserves Boltzmann Distribution
:label: thm-baoab-attention-boltzmann

The GeodesicCrossAttention module with five BAOAB heads preserves the Boltzmann distribution

$$
\rho(z, p) \propto \exp\left(-\frac{\Phi_{\text{eff}}(z)}{T_c} - \frac{\|p\|_G^2}{2T_c}\right)
$$

to second order in the timestep $h$, provided:

1. Each head uses position-dependent temperature $\tau(z) = \sqrt{d_k}/\lambda(z)$
2. The O-step head uses thermalization coefficients $c_1 = e^{-\gamma h}$, $c_2 = \sqrt{(1-c_1^2)T_c}$
3. The geodesic Query projections correctly encode Christoffel symbols

*Proof sketch.* This follows from Proposition {prf:ref}`prop-baoab-preserves-boltzmann` applied to the attention implementation. The key points:

1. The symmetric splitting B-A-O-A-B ensures time-reversibility of deterministic steps.
2. The O-step exactly samples the Maxwell-Boltzmann momentum distribution.
3. Position-dependent temperature correctly weights attention by the metric.
4. Wilson lines preserve gauge covariance without affecting thermodynamic properties.

The attention-based implementation is equivalent to the original BAOAB up to discretization of the gradient and exponential map, which introduces $O(h^2)$ errors consistent with the original scheme. $\square$

:::

:::{prf:proposition} Keys as Force Bank
:label: prop-keys-as-force-bank

In the B-step attention heads, the Keys store precomputed gradients of the effective potential at context positions:

$$
K^{(\text{force})}_{z'} = W_K \cdot \nabla\Phi_{\text{eff}}(z')
$$

The attention score $Q(z)^T K(z')$ measures how aligned the current position is with the force at $z'$. The weighted sum of Values retrieves the appropriate force:

$$
F(z) = \sum_{z' \in \text{context}} \alpha(z, z') \cdot V^{(\text{force})}(z') = -\nabla\Phi_{\text{eff}}(z) + O(\text{interpolation error})
$$

*Advantage*: Precomputing gradients at context positions amortizes the cost of gradient computation across multiple queries.

:::

:::{prf:proposition} Values as State Updates
:label: prop-values-as-state-updates

In each BAOAB head, the Values encode the update to apply:

| Head | Value Content | Update |
|:-----|:-------------|:-------|
| B (kick) | $-\nabla\Phi$ | $\Delta p$ |
| A (drift) | $G^{-1}p$ | $\Delta z$ |
| O (thermostat) | $\sqrt{T_c G}\xi$ | Noise injection |

The attention-weighted Value sum produces the update vector, which is then added to (or replaces) the current state.

:::



(sec-implementation-geodesic-attention)=
## Implementation: Full GeodesicCrossAttention Module

:::{div} feynman-prose
Here is the complete implementation. It looks like a lot of code, but each piece corresponds directly to something we have derived mathematically.

The main class `GeodesicCrossAttention` has five sub-modules for the five BAOAB heads. Each head uses the `CovariantAttention` helper class, which handles Wilson lines, position-dependent temperature, and quadratic Query projections.

The `forward` method takes the current state $(z, p)$ and produces the next state $(z', p')$ in one pass. Inside, it sequences through the five heads: kick, drift, thermostat, drift, kick.

Pay attention to the metric computation. We use the Poincare disk formula, but this is modular---you can swap in any other metric by changing the `conformal_factor` method.

The Wilson line approximation uses the linearized form for nearby points. For long-range attention, you would need to accumulate the path-ordered product, which is more expensive.

The Christoffel symbols are encoded in the quadratic Query projection. The tensor `W_Q_gamma` is initialized to match the Poincare structure, but it is learnable.
:::

We provide a PyTorch implementation of the GeodesicCrossAttention module. This is **demonstrative code** illustrating the architecture; a production implementation would require additional numerical care.

:::{admonition} Implementation Note
:class: feynman-added warning

The code below is a **prototype implementation** intended to illustrate the architecture's structure. For production use, several enhancements are needed:

1. **Wilson lines**: The linear approximation is valid only for nearby points. Long-range attention requires path-ordered products or geodesic interpolation.
2. **Christoffel initialization**: The Christoffel tensor should be initialized more carefully to match the exact Poincare structure.
3. **Numerical stability**: Additional clamping and normalization may be needed near the boundary where $\lambda \to \infty$.
4. **Chiral projector**: The SU(2) structure is simplified; a full implementation should use proper Pauli matrix algebra.

**Implementation vs. Theory**: The mathematical formulation (Definition {prf:ref}`def-covariant-qkv-projections`) has Wilson lines embedded in Q, K, V definitions. The code uses an equivalent approach: Wilson line correction is applied at attention-time via `K_transported = U @ K`. Both achieve gauge invariance; the code approach is more modular for experimentation.

The essential mathematical structure is preserved: Wilson line preprocessing, position-dependent temperature, quadratic Query, and BAOAB splitting.
:::

```python
"""
GeodesicCrossAttention: Gauge-covariant world model as geodesic integrator.

Cross-references:
    - Definition {prf:ref}`def-covariant-qkv-projections` (Covariant Q/K/V)
    - Theorem {prf:ref}`thm-gauge-invariance-cross-attention` (Gauge invariance)
    - Theorem {prf:ref}`thm-metric-temperature-correspondence` (Metric-temperature)
    - Definition {prf:ref}`def-baoab-attention-heads` (BAOAB heads)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class GeodesicConfig:
    """Configuration for GeodesicCrossAttention.

    Args:
        d_model: Model dimension [nat]
        d_latent: Latent space dimension
        n_heads: Number of attention heads per BAOAB step (default: 1)
        T_c: Cognitive temperature [nat/step]
        gamma_friction: Friction coefficient for O-step
        dt: Integration timestep
        g_s: Binding coupling strength (for area law screening)
        g_2: Error field coupling (for SU(2) chirality)
        g_1: Opportunity field coupling (for U(1) hypercharge)
    """
    d_model: int = 256
    d_latent: int = 64
    n_heads: int = 1
    T_c: float = 0.1
    gamma_friction: float = 1.0
    dt: float = 0.01
    g_s: float = 1.0
    g_2: float = 0.5
    g_1: float = 0.3


class WilsonLineApprox(nn.Module):
    """
    Approximate Wilson line for parallel transport.

    Implements the linear approximation:
        U(z, z') ≈ I - i * Θ(z) · (z - z')

    where Θ encodes the gauge connection coefficients.

    Cross-reference: Proposition {prf:ref}`prop-wilson-line-approximation`
    """

    def __init__(self, config: GeodesicConfig):
        super().__init__()
        d = config.d_latent

        # Connection coefficients (learnable, initialized to small values)
        # Shape: [d_latent, d_latent] - encodes Θ^a_μ contracted indices
        self.theta_binding = nn.Parameter(
            config.g_s * 0.1 * torch.randn(d, d)
        )  # SU(N_f) contribution
        self.theta_error = nn.Parameter(
            config.g_2 * 0.1 * torch.randn(d, d)
        )  # SU(2) contribution
        self.theta_opportunity = nn.Parameter(
            config.g_1 * 0.1 * torch.randn(d)
        )  # U(1) contribution

    def forward(
        self,
        z_query: torch.Tensor,  # [B, d]
        z_key: torch.Tensor,    # [B, N, d]
    ) -> torch.Tensor:
        """
        Compute Wilson line correction factor.

        Returns: [B, N, d, d] transformation matrix for each key position.
        """
        B, N, d = z_key.shape

        # Displacement z - z'
        delta_z = z_query.unsqueeze(1) - z_key  # [B, N, d]

        # Total connection: Θ · δz
        # SU(N_f) contribution (non-Abelian, matrix-valued)
        theta_nf = torch.einsum('ij,bnj->bni', self.theta_binding, delta_z)

        # SU(2) contribution (non-Abelian)
        theta_su2 = torch.einsum('ij,bnj->bni', self.theta_error, delta_z)

        # U(1) contribution (Abelian, scalar)
        theta_u1 = torch.einsum('j,bnj->bn', self.theta_opportunity, delta_z)

        # Build approximate Wilson line: U ≈ I - i*Θ
        # For real implementation, use exp(-i*Θ) via matrix exponential
        # Here we use linear approximation for efficiency
        identity = torch.eye(d, device=z_key.device).expand(B, N, d, d)

        # Combine into correction matrix (simplified: additive corrections)
        correction = identity.clone()
        correction = correction - 0.1 * theta_nf.unsqueeze(-1) * identity
        correction = correction - 0.1 * theta_su2.unsqueeze(-1) * identity

        return correction  # [B, N, d, d]


class ConformalMetric(nn.Module):
    """
    Poincare disk metric computations.

    λ(z) = 2 / (1 - |z|²)
    G_ij = λ² δ_ij

    Cross-reference: Definition {prf:ref}`def-poincare-metric-recap`
    """

    def __init__(self, epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon

    def conformal_factor(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute conformal factor λ(z) = 2 / (1 - |z|²).

        Args:
            z: [B, d] positions in Poincare disk

        Returns: [B, 1] conformal factors
        """
        r_sq = (z ** 2).sum(dim=-1, keepdim=True)
        # Clamp to ensure we stay inside disk
        r_sq = torch.clamp(r_sq, max=1.0 - self.epsilon)
        lambda_z = 2.0 / (1.0 - r_sq + self.epsilon)
        return lambda_z

    def metric(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute metric tensor G_ij(z).

        Returns: [B, d, d] metric tensors
        """
        B, d = z.shape
        lambda_sq = self.conformal_factor(z) ** 2  # [B, 1]
        return lambda_sq.unsqueeze(-1) * torch.eye(d, device=z.device)

    def metric_inv(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute inverse metric G^{ij}(z).

        Returns: [B, d, d] inverse metric tensors
        """
        B, d = z.shape
        lambda_sq_inv = 1.0 / (self.conformal_factor(z) ** 2 + self.epsilon)
        return lambda_sq_inv.unsqueeze(-1) * torch.eye(d, device=z.device)

    def temperature(self, z: torch.Tensor, d_k: int) -> torch.Tensor:
        """
        Position-dependent attention temperature.

        τ(z) = √d_k / λ(z)

        Cross-reference: Theorem {prf:ref}`thm-metric-temperature-correspondence`

        Returns: [B, 1] temperatures
        """
        lambda_z = self.conformal_factor(z)
        return math.sqrt(d_k) / lambda_z


class ChristoffelQuery(nn.Module):
    """
    Quadratic Query projection encoding Christoffel symbols.

    Q(z, v) = W_Q z + W_Qv v + W_QΓ(z, z) + W_Qzv(z, v)

    Cross-reference: Definition {prf:ref}`def-geodesic-query-projection`
    """

    def __init__(self, d_in: int, d_out: int, d_latent: int):
        super().__init__()

        # Linear projections
        self.W_Q = nn.Linear(d_in, d_out, bias=False)
        self.W_Qv = nn.Linear(d_in, d_out, bias=False)

        # Quadratic projection (3-tensor for Christoffel encoding)
        # Initialized to Poincare Christoffel structure
        self.W_Q_gamma = nn.Parameter(torch.zeros(d_out, d_latent, d_latent))
        self._init_christoffel(d_latent)

        # Position-velocity coupling
        self.W_Qzv = nn.Parameter(torch.zeros(d_out, d_latent, d_latent))

    def _init_christoffel(self, d: int):
        """Initialize W_Q_gamma to approximate Poincare Christoffel structure."""
        # Γ^k_ij ∝ (δ^k_i z_j + δ^k_j z_i - δ_ij z^k)
        # We initialize with small random values; exact structure is learned
        with torch.no_grad():
            for k in range(min(d, self.W_Q_gamma.shape[0])):
                for i in range(d):
                    for j in range(d):
                        if i == k or j == k:
                            self.W_Q_gamma[k, i, j] = 0.01
                        if i == j:
                            self.W_Q_gamma[k, i, j] -= 0.01

    def forward(
        self,
        z: torch.Tensor,        # [B, d_in] position
        v: Optional[torch.Tensor] = None,  # [B, d_in] velocity
    ) -> torch.Tensor:
        """
        Compute geodesic Query.

        Returns: [B, d_out] Query vectors
        """
        # Linear term
        Q = self.W_Q(z)  # [B, d_out]

        # Velocity term (if provided)
        if v is not None:
            Q = Q + self.W_Qv(v)

        # Quadratic term (Christoffel encoding)
        # W_QΓ(z, z) = sum_ij W_QΓ^a_ij z^i z^j
        d = min(z.shape[-1], self.W_Q_gamma.shape[-1])
        z_trunc = z[..., :d]
        Q_gamma = torch.einsum('aij,bi,bj->ba', self.W_Q_gamma[:, :d, :d], z_trunc, z_trunc)
        Q = Q + Q_gamma

        # Position-velocity coupling (if velocity provided)
        if v is not None:
            v_trunc = v[..., :d]
            Q_zv = torch.einsum('aij,bi,bj->ba', self.W_Qzv[:, :d, :d], z_trunc, v_trunc)
            Q = Q + Q_zv

        return Q


class ChiralProjector(nn.Module):
    """
    SU(2)_L chiral projector from value gradient.

    Π = (1/2)(I + n·τ) where n = ∇V / ||∇V||

    Cross-reference: Definition {prf:ref}`def-chiral-projector-value-gradient`
    """

    def __init__(self, d_latent: int):
        super().__init__()
        self.d_latent = d_latent

        # Pauli matrices (real representation for 2x2 case)
        # Note: τ_2 is purely imaginary; we use its real rotation effect
        self.register_buffer('tau_1', torch.tensor([[0., 1.], [1., 0.]]))
        self.register_buffer('tau_2', torch.tensor([[0., -1.], [1., 0.]]))  # Real part of rotation
        self.register_buffer('tau_3', torch.tensor([[1., 0.], [0., -1.]]))

    def forward(
        self,
        psi_doublet: torch.Tensor,  # [B, 2, d] observation-action doublet
        grad_V: torch.Tensor,        # [B, d] value gradient
    ) -> torch.Tensor:
        """
        Apply chiral projection to extract committed action.

        Returns: [B, d] committed action (scalar under SU(2))
        """
        B, _, d = psi_doublet.shape

        # Normalize gradient to get direction
        grad_norm = torch.norm(grad_V, dim=-1, keepdim=True) + 1e-8
        n = grad_V / grad_norm  # [B, d]

        # Construct n·τ (simplified: use first 3 components of n)
        # For full implementation, embed into SU(2) algebra
        n_3 = n[:, :3]  # [B, 3]

        # Projector on first two components of doublet
        # Π = (1/2)(I + n₃·τ₃) for diagonal approximation
        proj_diag = 0.5 * (1 + n_3[:, 2:3])  # [B, 1] - projects onto "action" component

        # Extract committed action via gauge-invariant contraction
        # ψ_commit = ψ_obs† Π ψ_act
        psi_obs = psi_doublet[:, 0, :]  # [B, d]
        psi_act = psi_doublet[:, 1, :]  # [B, d]

        # Scalar product weighted by projector
        committed = proj_diag * (psi_obs * psi_act).sum(dim=-1, keepdim=True)

        return committed.expand(-1, d)  # [B, d]


class AreaLawScreening(nn.Module):
    """
    SU(N_f)_C area law screening for texture confinement.

    Applied to attention weights (after softmax):
        α_screened = α · exp(-σ · A_string)

    The output is then renormalized to sum to 1.

    Cross-reference: Definition {prf:ref}`def-area-law-screening-attention`
    """

    def __init__(self, config: GeodesicConfig):
        super().__init__()
        self.g_s = config.g_s

        # String tension (learnable, initialized from coupling)
        self.log_sigma = nn.Parameter(torch.log(torch.tensor(config.g_s ** 2)))

    @property
    def sigma(self) -> torch.Tensor:
        return torch.exp(self.log_sigma)

    def string_area(
        self,
        z_query: torch.Tensor,  # [B, d]
        z_key: torch.Tensor,    # [B, N, d]
        lambda_z: torch.Tensor,  # [B, 1] conformal factor at query
    ) -> torch.Tensor:
        """
        Compute minimal string area between positions.

        A ≈ (λ²/2) |z - z'|²

        Returns: [B, N] string areas
        """
        delta_z = z_query.unsqueeze(1) - z_key  # [B, N, d]
        dist_sq = (delta_z ** 2).sum(dim=-1)  # [B, N]
        area = 0.5 * (lambda_z ** 2) * dist_sq
        return area

    def forward(
        self,
        attention: torch.Tensor,  # [B, N] attention scores
        z_query: torch.Tensor,
        z_key: torch.Tensor,
        lambda_z: torch.Tensor,
        level: int = 0,  # hierarchy level (0=macro, L=texture)
    ) -> torch.Tensor:
        """
        Apply area law screening to attention.

        Returns: [B, N] screened attention
        """
        area = self.string_area(z_query, z_key, lambda_z)

        # Level-dependent coupling (asymptotic freedom)
        # σ(ℓ) = σ_0 · exp(-ℓ / L) for UV decay
        L_max = 10  # maximum hierarchy depth
        sigma_eff = self.sigma * math.exp(-level / L_max)

        # Screening factor
        screening = torch.exp(-sigma_eff * area)

        return attention * screening


class CovariantAttention(nn.Module):
    """
    Single covariant attention head with all gauge structures.

    Combines:
        - Wilson line preprocessing
        - Position-dependent temperature
        - Quadratic Query (Christoffel)
        - Chiral projection (optional)
        - Area law screening (optional)
    """

    def __init__(
        self,
        config: GeodesicConfig,
        use_chirality: bool = False,
        use_screening: bool = False,
        head_type: str = 'generic',  # 'B', 'A', 'O', or 'generic'
    ):
        super().__init__()
        self.config = config
        self.use_chirality = use_chirality
        self.use_screening = use_screening
        self.head_type = head_type

        d = config.d_model
        d_k = d // config.n_heads

        # Core projections
        self.query = ChristoffelQuery(d, d_k, config.d_latent)
        self.key = nn.Linear(d, d_k, bias=False)
        self.value = nn.Linear(d, d_k, bias=False)
        self.output = nn.Linear(d_k, d, bias=False)

        # Gauge structures
        self.wilson = WilsonLineApprox(config)
        self.metric = ConformalMetric()

        if use_chirality:
            self.chiral = ChiralProjector(config.d_latent)

        if use_screening:
            self.screening = AreaLawScreening(config)

        self.d_k = d_k

    def forward(
        self,
        z_query: torch.Tensor,     # [B, d_latent] query position
        z_key: torch.Tensor,       # [B, N, d_latent] key positions
        x_query: torch.Tensor,     # [B, d_model] query features
        x_key: torch.Tensor,       # [B, N, d_model] key features
        x_value: torch.Tensor,     # [B, N, d_model] value features
        v_query: Optional[torch.Tensor] = None,  # [B, d_model] velocity
        grad_V: Optional[torch.Tensor] = None,   # [B, d_latent] value gradient
        level: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute covariant attention.

        Returns:
            output: [B, d_model] attention output
            attention: [B, N] attention weights
        """
        B, N, _ = x_key.shape

        # Compute Q, K, V with geodesic Query
        Q = self.query(x_query, v_query)  # [B, d_k]
        K = self.key(x_key)                # [B, N, d_k]
        V = self.value(x_value)            # [B, N, d_k]

        # Wilson line correction to Keys
        U = self.wilson(z_query, z_key)  # [B, N, d_k, d_k] (approx)
        K_transported = torch.einsum('bnij,bnj->bni', U[:, :, :self.d_k, :self.d_k], K)

        # Attention scores with Wilson-corrected Keys
        scores = torch.einsum('bi,bni->bn', Q, K_transported)  # [B, N]

        # Position-dependent temperature
        tau = self.metric.temperature(z_query, self.d_k)  # [B, 1]
        scores = scores / (tau + 1e-8)

        # Softmax
        attention = F.softmax(scores, dim=-1)  # [B, N]

        # Area law screening (if enabled) - applied after softmax per Definition
        if self.use_screening:
            lambda_z = self.metric.conformal_factor(z_query)
            attention = self.screening(attention, z_query, z_key, lambda_z, level)
            # Renormalize after screening
            attention = attention / (attention.sum(dim=-1, keepdim=True) + 1e-8)

        # Weighted sum of Values
        output = torch.einsum('bn,bni->bi', attention, V)  # [B, d_k]

        # Chiral projection (if enabled and gradient provided)
        if self.use_chirality and grad_V is not None:
            # Reshape output as doublet for projection
            output_doublet = output.view(B, 2, -1)
            output = self.chiral(output_doublet, grad_V)
            output = output.view(B, -1)

        # Output projection
        output = self.output(output)

        return output, attention


class GeodesicCrossAttention(nn.Module):
    """
    Full geodesic world model implementing Boris-BAOAB integration.

    Five attention heads for B-A-O-A-B splitting.

    Cross-reference: Definition {prf:ref}`def-baoab-attention-heads`

    Usage:
        model = GeodesicCrossAttention(config)
        z_next, p_next = model(z, p, context_z, context_x)
    """

    def __init__(self, config: GeodesicConfig):
        super().__init__()
        self.config = config

        # BAOAB coefficients
        self.dt = config.dt
        self.gamma = config.gamma_friction
        self.T_c = config.T_c
        self.c1 = math.exp(-self.gamma * self.dt)
        self.c2 = math.sqrt((1 - self.c1 ** 2) * self.T_c) if self.T_c > 0 else 0.0

        # Metric computations
        self.metric = ConformalMetric()

        # Five BAOAB heads
        self.head_B1 = CovariantAttention(config, head_type='B')
        self.head_A1 = CovariantAttention(config, head_type='A')
        self.head_O = CovariantAttention(config, head_type='O')
        self.head_A2 = CovariantAttention(config, head_type='A')
        self.head_B2 = CovariantAttention(config, head_type='B')

        # Learnable force/gradient bank projection
        self.grad_encoder = nn.Linear(config.d_latent, config.d_model)
        self.velocity_encoder = nn.Linear(config.d_latent, config.d_model)

        # Noise projection for O-step
        self.noise_proj = nn.Linear(config.d_latent, config.d_model)

    def forward(
        self,
        z: torch.Tensor,           # [B, d_latent] current position
        p: torch.Tensor,           # [B, d_latent] current momentum
        context_z: torch.Tensor,   # [B, N, d_latent] context positions
        context_x: torch.Tensor,   # [B, N, d_model] context features
        grad_Phi: Optional[torch.Tensor] = None,  # [B, d_latent] potential gradient
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        One step of geodesic BAOAB integration.

        Returns:
            z_next: [B, d_latent] updated position
            p_next: [B, d_latent] updated momentum
        """
        B, d = z.shape
        h = self.dt

        # Encode current state as query features
        x_query = self.velocity_encoder(z) + self.grad_encoder(p)

        # ===== B-step 1: First half-kick =====
        delta_p1, _ = self.head_B1(
            z_query=z,
            z_key=context_z,
            x_query=x_query,
            x_key=context_x,
            x_value=context_x,
            v_query=self.velocity_encoder(p),
        )
        # Interpret as momentum update
        p = p - (h / 2) * delta_p1[:, :d]

        # ===== A-step 1: First half-drift =====
        # Compute velocity from momentum
        G_inv = self.metric.metric_inv(z)
        v = torch.einsum('bij,bj->bi', G_inv, p)

        delta_z1, _ = self.head_A1(
            z_query=z,
            z_key=context_z,
            x_query=self.velocity_encoder(z),
            x_key=context_x,
            x_value=context_x,
            v_query=self.velocity_encoder(v),
        )
        # Apply position update via exponential map approximation
        z = z + (h / 2) * (v + 0.1 * delta_z1[:, :d])
        z = self._project_to_disk(z)

        # ===== O-step: Ornstein-Uhlenbeck thermostat =====
        # Inject noise
        xi = torch.randn_like(p)
        noise_features = self.noise_proj(xi)

        delta_p_noise, _ = self.head_O(
            z_query=z,
            z_key=context_z,
            x_query=self.velocity_encoder(p),
            x_key=noise_features.unsqueeze(1).expand(-1, context_z.shape[1], -1),
            x_value=noise_features.unsqueeze(1).expand(-1, context_z.shape[1], -1),
        )

        # Thermalize momentum
        G_sqrt = torch.sqrt(self.metric.conformal_factor(z))
        p = self.c1 * p + self.c2 * G_sqrt * (xi + 0.1 * delta_p_noise[:, :d])

        # ===== A-step 2: Second half-drift =====
        G_inv = self.metric.metric_inv(z)
        v = torch.einsum('bij,bj->bi', G_inv, p)

        delta_z2, _ = self.head_A2(
            z_query=z,
            z_key=context_z,
            x_query=self.velocity_encoder(z),
            x_key=context_x,
            x_value=context_x,
            v_query=self.velocity_encoder(v),
        )
        z = z + (h / 2) * (v + 0.1 * delta_z2[:, :d])
        z = self._project_to_disk(z)

        # ===== B-step 2: Second half-kick =====
        x_query = self.velocity_encoder(z) + self.grad_encoder(p)
        delta_p2, _ = self.head_B2(
            z_query=z,
            z_key=context_z,
            x_query=x_query,
            x_key=context_x,
            x_value=context_x,
            v_query=self.velocity_encoder(p),
        )
        p = p - (h / 2) * delta_p2[:, :d]

        return z, p

    def _project_to_disk(self, z: torch.Tensor, max_norm: float = 0.999) -> torch.Tensor:
        """Project z to interior of Poincare disk."""
        norm = torch.norm(z, dim=-1, keepdim=True)
        return torch.where(norm > max_norm, z * max_norm / norm, z)
```

:::{div} feynman-prose
A few notes on the implementation:

1. **Efficiency**: The Wilson line computation is the most expensive part. In practice, you would precompute it for the context positions and cache it across heads.

2. **Initialization**: The Christoffel tensor $W_{Q,\Gamma}$ is initialized to match the Poincare structure. During training, it can adapt to the actual geometry of the learned latent space.

3. **Stability**: The projection to disk at the end of each step ensures we never leave the representable region. This is a safety net on top of the geometric constraints.

4. **Modularity**: Each component (Wilson lines, metric, Christoffel, chirality, screening) is a separate module. You can enable or disable them independently depending on which gauge structures you need.
:::



(sec-computational-complexity-proxies)=
## Computational Complexity and O(N) Engineering Proxies

:::{div} feynman-prose
Now let me address the elephant in the room: computational complexity. The architecture I have described is beautiful mathematically, but if you implement it naively, it will be slow. Very slow.

The problem is that attention is inherently $O(N^2)$---every query attends to every key. On top of that, we have added Wilson lines, which require path integrals. And quadratic Query terms, which add $O(d^2)$ parameters. And area law screening, which requires computing string areas for every pair.

For a context of $N = 1000$ positions and latent dimension $d = 64$, the naive complexity is $O(N^2 d^2) \approx 4 \times 10^9$ operations per attention layer. That is not practical.

But here is the good news: we can achieve $O(N)$ complexity while preserving the essential gauge structure. The key insight is that gauge invariance is a *local* property. The Wilson line from $z$ to $z'$ only matters when $z$ and $z'$ are close enough to interact significantly. Far-away pairs are screened by the area law anyway. So we can use sparse attention patterns that respect the geometric locality.

Let me show you how to do this systematically.
:::

The naive implementation of covariant cross-attention has complexity $O(N^2 d^2)$ per layer. We derive practical approximations that achieve $O(N)$ or $O(N \log N)$ complexity while preserving the essential gauge structure.

:::{prf:proposition} Complexity of Naive Implementation
:label: prop-complexity-naive-implementation

The full covariant cross-attention has the following complexity breakdown:

| Component | Naive Complexity | Bottleneck |
|:----------|:-----------------|:-----------|
| Wilson line computation | $O(N^2 d^2)$ | Path integral for each pair |
| Attention scores | $O(N^2 d)$ | All-pairs dot product |
| Quadratic Query | $O(N d^2)$ | Christoffel tensor contraction |
| Area law screening | $O(N^2)$ | String area for each pair |
| Chiral projection | $O(N d)$ | Per-position projection |
| **Total** | $O(N^2 d^2)$ | Dominated by Wilson lines |

For $N = 1000$, $d = 64$: approximately $4 \times 10^9$ operations per layer.

:::

### Principle: Gauge-Locality Correspondence

:::{div} feynman-prose
The crucial observation is that gauge invariance and locality are deeply connected. The Wilson line $U(z, z')$ is close to the identity when $z$ and $z'$ are nearby (in geodesic distance). The area law screening $\exp(-\sigma A)$ suppresses attention to distant keys exponentially. The metric temperature $\tau(z) \propto 1/\lambda(z)$ makes attention sharper in high-curvature (boundary) regions where long-range correlations are suppressed.

All of these effects conspire to make the attention matrix *effectively sparse*. Most entries are either:
1. Nearly identity (Wilson line ≈ I for nearby points)
2. Exponentially suppressed (area law screening)
3. Concentrated on local neighbors (sharp temperature near boundary)

We can exploit this effective sparsity to achieve linear complexity.
:::

:::{prf:theorem} Gauge-Locality Correspondence
:label: thm-gauge-locality-correspondence

Let $\epsilon > 0$ be a tolerance and $r_{\epsilon}(z)$ be the **effective interaction radius** at position $z$:

$$
r_{\epsilon}(z) := \min\left( \frac{1}{\sqrt{\sigma}}, \frac{1}{\lambda(z)}, r_{\text{Wilson}} \right)
$$

where:
- $1/\sqrt{\sigma}$ is the confinement radius from string tension
- $1/\lambda(z)$ is the metric length scale (small near boundary)
- $r_{\text{Wilson}}$ is the radius where Wilson line approximation holds to accuracy $\epsilon$

Then the attention weight $\alpha(z, z')$ satisfies:

$$
\alpha(z, z') < \epsilon \quad \text{whenever} \quad d_G(z, z') > r_{\epsilon}(z)
$$

*Consequence*: Only $O(k)$ keys per query contribute significantly, where $k \sim \text{Vol}(B_{r_\epsilon}(z)) / \text{Vol}(\mathcal{Z})$ is the fraction of space within the interaction radius.

:::

### Engineering Proxy 1: Sparse Attention with Geometric Neighborhoods

:::{prf:definition} Geodesic Sparse Attention
:label: def-geodesic-sparse-attention

Replace full attention with **geodesic-local sparse attention**:

$$
\alpha_{\text{sparse}}(z, z') = \begin{cases}
\alpha_{\text{full}}(z, z') & \text{if } d_G(z, z') \leq r_{\epsilon}(z) \\
0 & \text{otherwise}
\end{cases}
$$

**Implementation**: Use a spatial data structure (k-d tree, ball tree, or locality-sensitive hashing) to find the $k$-nearest neighbors in geodesic distance.

**Complexity**: $O(N k d)$ where $k$ is the neighborhood size, typically $k \sim 32$-$128$.

**Gauge-faithfulness**: Exact for interactions within radius $r_\epsilon$. Error bounded by $\epsilon \cdot N$ for truncated interactions.

:::

:::{admonition} Implementation: Efficient Neighbor Finding
:class: feynman-added tip

For the Poincare disk, geodesic distance is:

$$
d_G(z, z') = 2 \tanh^{-1}\left(\frac{|z - z'|}{|1 - \bar{z}z'|}\right)
$$

This is not Euclidean, but for points far from the boundary, $d_G \approx \lambda(z)|z - z'|$. A practical approach:

1. **Build tree in Euclidean coordinates** (standard k-d tree)
2. **Query with inflated radius**: Search for Euclidean neighbors within $r_{\text{Euclidean}} = r_\epsilon / \lambda_{\min}$
3. **Filter by geodesic distance**: Discard neighbors with $d_G > r_\epsilon$

This gives $O(N \log N)$ preprocessing and $O(k \log N)$ per query.
:::

### Engineering Proxy 2: Linearized Attention via Kernel Approximation

:::{prf:definition} Linearized Covariant Attention
:label: def-linearized-covariant-attention

Approximate the softmax attention kernel using random Fourier features:

$$
\exp\left(\frac{Q^T K}{\tau}\right) \approx \phi(Q/\sqrt{\tau})^T \phi(K/\sqrt{\tau})
$$

where $\phi: \mathbb{R}^d \to \mathbb{R}^D$ is a random feature map with $D \ll N$:

$$
\phi(x) = \frac{1}{\sqrt{D}} \begin{pmatrix} \cos(\omega_1^T x + b_1) \\ \vdots \\ \cos(\omega_D^T x + b_D) \end{pmatrix}
$$

with $\omega_i \sim \mathcal{N}(0, I)$ and $b_i \sim \text{Uniform}(0, 2\pi)$.

**Linearized attention**:

$$
\text{Attn}_{\text{linear}}(Q, K, V) = \frac{\phi(Q)^T \left(\sum_j \phi(K_j) V_j^T\right)}{\phi(Q)^T \left(\sum_j \phi(K_j)\right)}
$$

**Complexity**: $O(N D d)$ where $D \sim 64$-$256$.

**Gauge-faithfulness**: The position-dependent temperature $\tau(z)$ is preserved in the feature map scaling. Wilson line corrections enter through modified $Q, K$ before feature mapping.

:::

:::{prf:proposition} Linear Attention Preserves Metric Structure
:label: prop-linear-attention-metric

The linearized attention with position-dependent temperature $\tau(z) = \sqrt{d_k}/\lambda(z)$ preserves the metric structure to leading order:

$$
\text{Attn}_{\text{linear}}(z, z') \propto \exp\left(-\frac{\lambda(z)^2 |z - z'|^2}{2d_k}\right) + O(|z-z'|^4)
$$

This is a Gaussian kernel with width inversely proportional to the metric scale---the same effective behavior as the full attention.

:::

### Engineering Proxy 3: Hierarchical Wilson Lines

:::{div} feynman-prose
The Wilson line is the most expensive component because it requires integrating the gauge connection along a path. But here is a trick: Wilson lines compose.

If you have precomputed $U_{0 \to z}$ (origin to $z$) and $U_{0 \to z'}$ (origin to $z'$), then:

$$
U(z, z') = U_{0 \to z}^\dagger U_{0 \to z'}
$$

So instead of computing $O(N^2)$ Wilson lines, you compute $O(N)$ lines from the origin and compose them. This is exactly what the definition in Section 35.3 already uses---it is not just mathematically convenient, it is computationally efficient.

But we can do even better with a hierarchical approach. Divide the latent space into a tree of regions. Precompute Wilson lines between region centers. For fine-grained transport, use the linear approximation within each region.
:::

:::{prf:definition} Hierarchical Wilson Line Approximation
:label: def-hierarchical-wilson-line

Construct a **hierarchical decomposition** of the latent space $\mathcal{Z}$:

- **Level 0**: Single root node (origin)
- **Level $\ell$**: $2^{\ell d}$ cells of diameter $\sim 2^{-\ell}$
- **Maximum level $L$**: Cells of diameter $\sim r_{\text{Wilson}}$

**Precomputation** ($O(2^{Ld})$ storage):
- For each cell center $c_i$: Store $U_{0 \to c_i}$
- For each cell: Store local connection coefficients $\Theta_i = A_\mu(c_i)$

**Query** ($O(L + d)$ per pair):

$$
U(z, z') \approx U_{0 \to c(z)}^\dagger \cdot U_{\text{local}}(c(z), z)^\dagger \cdot U_{\text{local}}(c(z'), z') \cdot U_{0 \to c(z')}
$$

where $c(z)$ is the center of $z$'s cell and $U_{\text{local}}$ uses the linear approximation:

$$
U_{\text{local}}(c, z) \approx I - i \Theta(c) \cdot (z - c)
$$

**Complexity**: $O(N \cdot L \cdot d)$ where $L \sim \log(1/r_{\text{Wilson}})$.

:::

### Engineering Proxy 4: Amortized Christoffel Computation

:::{prf:definition} Factorized Christoffel Query
:label: def-factorized-christoffel-query

The quadratic Query $W_{Q,\Gamma}(z, z) = \sum_{ij} W^{a}_{ij} z^i z^j$ has $O(d^2)$ parameters per output dimension. Factorize as:

$$
W_{Q,\Gamma}^a(z, z) \approx \sum_{r=1}^R (u_r^a \cdot z)(v_r^a \cdot z) = \sum_{r=1}^R (U_r z)_a (V_r z)_a
$$

where $U_r, V_r \in \mathbb{R}^{d_k \times d}$ are low-rank factors with $R \ll d$.

**For the Poincare disk**: The Christoffel structure $\Gamma^k_{ij} \propto \delta^k_i z_j + \delta^k_j z_i - \delta_{ij} z^k$ has rank 3. We can capture it exactly with $R = 3$:

$$
W_{Q,\Gamma}(z, z) = \alpha_1 (z \cdot z) \mathbf{e}_z + \alpha_2 |z|^2 z + \alpha_3 \text{diag}(z) z
$$

**Complexity**: $O(N R d)$ instead of $O(N d^2)$.

:::

### Engineering Proxy 5: Efficient Area Law via Distance Thresholding

:::{prf:proposition} Area Law as Soft Masking
:label: prop-area-law-soft-masking

The area law screening $\exp(-\sigma A_{\text{string}})$ with $A \approx \frac{\lambda^2}{2}|z - z'|^2$ can be implemented as a **soft attention mask**:

$$
M(z, z') = \exp\left(-\frac{\sigma \lambda(z)^2 |z - z'|^2}{2}\right)
$$

**Key observation**: This is a Gaussian with width $w = 1/(\sqrt{\sigma}\lambda(z))$.

**Efficient implementation**:
1. Compute mask only for pairs within $3w$ (99.7% of mass)
2. Use the same sparse attention pattern as Proxy 1
3. Combine: $\alpha_{\text{screened}} = M \odot \alpha_{\text{sparse}}$

**No additional asymptotic cost** beyond sparse attention.

:::

### Summary: Achieving O(N) Complexity

:::{prf:theorem} O(N) Covariant Attention
:label: thm-on-covariant-attention

Combining the engineering proxies, we achieve **O(N) complexity** for covariant cross-attention:

| Component | Proxy | Complexity | Error |
|:----------|:------|:-----------|:------|
| Attention pattern | Geodesic sparse (Def. {prf:ref}`def-geodesic-sparse-attention`) | $O(Nk)$ | $O(\epsilon N)$ |
| Wilson lines | Hierarchical (Def. {prf:ref}`def-hierarchical-wilson-line`) | $O(NL)$ | $O(r_{\text{Wilson}}^2)$ |
| Quadratic Query | Factorized (Def. {prf:ref}`def-factorized-christoffel-query`) | $O(NRd)$ | Exact for Poincare |
| Area screening | Soft mask (Prop. {prf:ref}`prop-area-law-soft-masking`) | $O(Nk)$ | $O(e^{-9})$ |
| Temperature | Direct computation | $O(N)$ | Exact |
| **Total** | | $O(N(k + L + Rd))$ | Controlled |

With typical values $k = 64$, $L = 8$, $R = 3$, $d = 64$: **O(N · 264)** operations per layer.

**Comparison**: Naive implementation is $O(N^2 d^2) = O(N^2 \cdot 4096)$. For $N = 1000$: speedup factor of $\sim 15,000\times$.

:::

:::{div} feynman-prose
Let me summarize what we have achieved. The full covariant cross-attention, implemented naively, would be hopelessly slow. But by exploiting the *physical structure* of the problem---the locality of gauge interactions, the exponential suppression from area laws, the low-rank structure of Christoffel symbols---we can achieve linear complexity.

The key insight is that gauge invariance is not about comparing everything with everything. It is about comparing *nearby* things correctly. Far-away things do not interact strongly anyway (that is what confinement means!). So sparse attention is not just an approximation---it is *more faithful* to the physics than dense attention would be.

This is a general principle: when you understand the structure of your problem, efficiency and accuracy align. The shortcuts that make computation fast are often the same shortcuts that the physics takes.
:::

**Table 35.9.1 (Complexity Comparison).**

| Implementation | Time | Space | Gauge-Faithful? |
|:---------------|:-----|:------|:----------------|
| Naive dense | $O(N^2 d^2)$ | $O(N^2)$ | Yes |
| Sparse + hierarchical | $O(N k L)$ | $O(N + 2^{Ld})$ | Yes (within $r_\epsilon$) |
| Linear kernel | $O(N D d)$ | $O(D d)$ | Approximate |
| **Recommended** | $O(N(k + Rd))$ | $O(N d)$ | Yes (hybrid) |

:::{admonition} Practical Recommendations
:class: feynman-added tip

For production deployment:

1. **Start with sparse attention** ($k \approx 64$ neighbors). This preserves exact gauge structure within the neighborhood.

2. **Use hierarchical Wilson lines** with $L \approx 8$ levels. Precompute once per batch.

3. **Factorize Christoffel tensor** with $R = 3$ (exact for Poincare). Learn corrections during training.

4. **Combine area law with sparse mask**. No additional cost.

5. **Monitor diagnostic nodes** (67-70). If gauge invariance degrades, increase $k$ or $L$.

The resulting architecture runs in **O(N)** time with controlled, bounded approximation error that can be verified at runtime.
:::



(sec-diagnostic-nodes-architecture)=
## Diagnostic Nodes 67-70: Gauge, Temperature, Chirality, Confinement

Following the diagnostic node convention ({ref}`Section 3.1 <sec-theory-thin-interfaces>`), we define monitors for the covariant attention architecture.

(node-67)=
**Node 67: GaugeInvarianceCheck**

| **#** | **Name** | **Component** | **Type** | **Interpretation** | **Proxy** | **Cost** |
|-------|----------|---------------|----------|---------------------|-----------|----------|
| **67** | **GaugeInvarianceCheck** | Architecture | Invariance | Are attention scores gauge-invariant? | $\Delta_{\text{gauge}} := \max_{U \in G} \|s(z,z') - s(Uz, Uz')\|$ | $O(Nd^2)$ |

**Interpretation:** Measures the deviation of attention scores under random gauge transformations. For a perfectly gauge-invariant implementation, $\Delta_{\text{gauge}} = 0$. Non-zero values indicate Wilson line approximation errors.

**Threshold:** $\Delta_{\text{gauge}} < 10^{-3}$ (typical).

**Trigger conditions:**
- High GaugeInvarianceCheck: Wilson line approximation is too coarse for the curvature scale. Remedy: use higher-order path-ordered product or reduce context distance.

(node-68)=
**Node 68: MetricTemperatureConsistencyCheck**

| **#** | **Name** | **Component** | **Type** | **Interpretation** | **Proxy** | **Cost** |
|-------|----------|---------------|----------|---------------------|-----------|----------|
| **68** | **MetricTemperatureConsistencyCheck** | Architecture | Thermodynamic | Is temperature consistent with metric? | $\Delta_{\tau} := \|\tau(z) \cdot \lambda(z) - \sqrt{d_k}\|$ | $O(d)$ |

**Interpretation:** Verifies that the position-dependent temperature $\tau(z)$ correctly encodes the inverse conformal factor. Discrepancy indicates miscalibration of the temperature module.

**Threshold:** $\Delta_{\tau} < 0.01 \sqrt{d_k}$ (1% relative error).

**Trigger conditions:**
- High MetricTemperatureConsistencyCheck: Temperature and metric are desynchronized. Remedy: recalibrate temperature module or check metric computation.

(node-69)=
**Node 69: ChiralityViolationCheck**

| **#** | **Name** | **Component** | **Type** | **Interpretation** | **Proxy** | **Cost** |
|-------|----------|---------------|----------|---------------------|-----------|----------|
| **69** | **ChiralityViolationCheck** | Architecture | Symmetry | Is observation-action asymmetry preserved? | $\chi_{\text{viol}} := \|\Pi_L - \Pi_L^2\|_F$ (deviation from idempotence) | $O(d^2)$ |

**Interpretation:** The chiral projector should be idempotent ($\Pi^2 = \Pi$). Deviation indicates numerical instability or incorrect value gradient.

**Threshold:** $\chi_{\text{viol}} < 10^{-6}$.

**Trigger conditions:**
- High ChiralityViolationCheck: Projector is not idempotent. Remedy: renormalize value gradient; check for numerical precision issues.

(node-70)=
**Node 70: TextureLeakageCheck**

| **#** | **Name** | **Component** | **Type** | **Interpretation** | **Proxy** | **Cost** |
|-------|----------|---------------|----------|---------------------|-----------|----------|
| **70** | **TextureLeakageCheck** | Architecture | Confinement | Is texture properly screened from macro level? | $\ell_{\text{texture}} := \sum_{z' \in \text{texture}} \alpha_{\text{macro} \to z'}$ | $O(N_{\text{texture}})$ |

**Interpretation:** Measures total attention weight from macro-level queries to texture-level keys. Should be exponentially suppressed by area law screening.

**Threshold:** $\ell_{\text{texture}} < e^{-\sigma_{\text{crit}}}$ where $\sigma_{\text{crit}} \approx 1$.

**Trigger conditions:**
- High TextureLeakageCheck: Texture is leaking to macro level. Remedy: increase string tension $\sigma$; check hierarchical level assignment.

:::{div} feynman-prose
These four diagnostic nodes give you visibility into the gauge structure of the trained model.

Node 67 catches approximation errors in the Wilson lines. If the attention scores change when you apply a gauge transformation, something is wrong with the parallel transport.

Node 68 verifies the metric-temperature correspondence. This is a sanity check that the architecture is correctly encoding the geometry.

Node 69 monitors the chiral projector. The projector must be idempotent---applying it twice should give the same result as applying it once. Violations indicate numerical problems.

Node 70 is the texture firewall check. If macro-level attention is accessing texture details, the area law screening is not working. This is a confinement violation.

Together, these nodes let you diagnose gauge-structure failures at runtime and take corrective action.
:::



(sec-summary-correspondence-architecture)=
## Summary: Architecture-Physics Correspondence

We summarize the correspondence between the covariant attention architecture and the underlying physics/geometry.

**Table 35.11.1 (Attention-Geometry Correspondence).**

| Attention Component | Geometric/Physical Object | Reference |
|:-------------------|:-------------------------|:----------|
| Wilson line preprocessing | Parallel transport on gauge bundle | Definition {prf:ref}`def-wilson-line` |
| Position-dependent temperature $\tau(z)$ | Inverse conformal factor $1/\lambda(z)$ | Theorem {prf:ref}`thm-metric-temperature-correspondence` |
| Quadratic Query $W_{QQ}(z,z)$ | Christoffel symbols $\Gamma^k_{ij}$ | Definition {prf:ref}`def-geodesic-query-projection` |
| Observation-action doublet | Left-handed $SU(2)_L$ field $\Psi_L$ | Definition {prf:ref}`def-observation-action-doublet-attention` |
| Chiral projector $\Pi_{\nabla V}$ | Electroweak symmetry breaking | Definition {prf:ref}`def-chiral-projector-value-gradient` |
| Area law screening | Confinement string tension | Definition {prf:ref}`def-area-law-screening-attention` |
| 5 BAOAB heads | Boris-BAOAB integrator | Definition {prf:ref}`def-baoab-attention-heads` |
| Keys | Force bank | Proposition {prf:ref}`prop-keys-as-force-bank` |
| Values | State updates | Proposition {prf:ref}`prop-values-as-state-updates` |

**Table 35.11.2 (Gauge Field Implementation).**

| Gauge Field | Implementation | Effect |
|:-----------|:--------------|:-------|
| $B_\mu$ (Opportunity) | $U(1)$ phase in Wilson line | Value baseline consistency |
| $W_\mu^a$ (Error) | $SU(2)$ mixing in doublet | Observation-action coordination |
| $G_\mu^a$ (Binding) | $SU(N_f)$ transformation | Feature-to-concept binding |

**Table 35.11.3 (Diagnostic Node Summary).**

| Node | What it Monitors | Healthy Range |
|:-----|:----------------|:--------------|
| 67: GaugeInvarianceCheck | Wilson line accuracy | $< 10^{-3}$ |
| 68: MetricTemperatureCheck | $\tau \cdot \lambda = \sqrt{d_k}$ | $< 0.01\sqrt{d_k}$ |
| 69: ChiralityViolationCheck | Projector idempotence | $< 10^{-6}$ |
| 70: TextureLeakageCheck | Area law confinement | $< e^{-1}$ |

:::{div} feynman-prose
Let me step back and tell you what we have accomplished in this chapter.

We took the abstract gauge-theoretic structure derived in previous chapters---the $SU(N_f) \times SU(2) \times U(1)$ symmetry, the Wilson lines, the Christoffel symbols, the chiral projectors---and translated it into neural network architecture. Every piece of the mathematics has a corresponding piece of the code.

The result is a world model that is gauge-invariant by construction. It does not need to learn gauge invariance; it has it baked in. It does not need regularization to respect the geometry; the geometry is the architecture.

This is a different philosophy from the usual "train a big neural network and hope it learns the right structure." We are saying: there is a right structure, we know what it is, and we should build it in explicitly.

The five BAOAB heads give you a complete geodesic integrator in one forward pass. The Keys store precomputed forces, the Values store state updates, and the attention mechanism routes information according to the gauge-covariant geometry.

Will this work in practice? That is an empirical question. But the theoretical foundations are solid. If the Lorentz-Langevin dynamics are a good model of what the agent should do, and if gauge covariance is important for consistency, then this architecture should outperform flat alternatives.

The diagnostic nodes let you monitor the gauge structure at runtime. If something goes wrong---if the Wilson lines are not accurate enough, if the temperature is miscalibrated, if texture is leaking---you will see it in the diagnostics and can take corrective action.

This is the power of building in structure rather than learning it from scratch. You get guarantees, you get interpretability, and you get diagnostic hooks. The cost is complexity in the architecture. But for applications where gauge covariance matters, that cost is worth paying.
:::

This chapter has derived the **Covariant Cross-Attention** architecture that implements:

1. **Gauge-covariant attention** via Wilson line preprocessing (Section 35.3)
2. **Metric-encoded temperature** via $\tau(z) = \sqrt{d_k}/\lambda(z)$ (Section 35.4)
3. **Geodesic correction** via quadratic Query projections (Section 35.5)
4. **$SU(2)_L$ chirality** via observation-action doublet and chiral projector (Section 35.6)
5. **$SU(N_f)_C$ confinement** via area law screening (Section 35.7)
6. **Boris-BAOAB integration** via five specialized attention heads (Section 35.8)
7. **O(N) complexity** via sparse attention, hierarchical Wilson lines, and factorized Christoffel (Section 35.9)

The architecture is a **gauge-invariant world model** that acts as a one-step integrator for the Lorentz-Langevin geodesic equations. Diagnostic nodes 67-70 monitor gauge invariance, metric-temperature consistency, chirality preservation, and texture confinement.

**Key Result:** The standard Transformer attention mechanism is a degenerate limit of covariant cross-attention with all gauge fields set to zero, constant temperature, linear Query, and no screening. The full architecture respects the gauge structure $G_{\text{Fragile}} = SU(N_f)_C \times SU(2)_L \times U(1)_Y$ by construction.
