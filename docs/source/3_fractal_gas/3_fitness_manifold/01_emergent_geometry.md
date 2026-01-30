(sec-emergent-geometry)=
# Emergent Geometry from Adaptive Diffusion

**Prerequisites**: {doc}`/source/3_fractal_gas/1_the_algorithm/02_fractal_gas_latent` (Latent Fractal Gas), {doc}`/source/3_fractal_gas/2_fractal_set/01_fractal_set` (Fractal Set)



(sec-tldr-emergent-geometry)=
## TLDR

*Notation: $g(z, S) = H(z, S) + \epsilon_\Sigma I$ (emergent Riemannian metric); $D_{\mathrm{reg}} = g^{-1}$ (diffusion tensor); $\Sigma_{\mathrm{reg}} = D_{\mathrm{reg}}^{1/2} = g^{-1/2}$ (diffusion matrix square root); $H = \nabla^2 V_{\mathrm{fit}}$ (fitness Hessian); $\epsilon_\Sigma$ (spectral floor/regularization); $c_{\min}, c_{\max}$ (ellipticity bounds).*

**Geometry from Optimization**: The Latent Fractal Gas does not assume a geometry on its state space—it *creates* one. The adaptive diffusion tensor, which modulates exploration noise based on local fitness curvature, defines an emergent Riemannian metric where the fitness Hessian plays the role of the metric tensor. Curved fitness landscapes produce curved geometries; flat regions produce flat space.

**The Diffusion-Metric Duality**: The relationship between diffusion and metric is not an analogy but a mathematical identity: $D_{\mathrm{reg}} = g^{-1}$. Large diffusion in a direction means noise spreads easily; large metric means that direction is "stretched" and noise should not smear the fine structure. The Latent Fractal Gas realizes this by setting the metric $g = H + \epsilon_\Sigma I$, where $H$ is the fitness Hessian and $\epsilon_\Sigma$ provides a spectral floor.

**Rigorous Convergence via Uniform Ellipticity**: Despite the anisotropic, state-dependent diffusion, the system converges provably. Uniform ellipticity bounds ($c_{\min} I \preceq D_{\mathrm{reg}} \preceq c_{\max} I$) that are independent of swarm size $N$ guarantee hypocoercive convergence to the quasi-stationary distribution. The geometry helps rather than hinders.

**Two Equivalent Descriptions**: The same physics admits two mathematically equivalent formulations: (1) flat space with anisotropic diffusion, or (2) curved Riemannian manifold with isotropic diffusion. All convergence rates, mixing times, and observables are identical in both views—choose whichever makes your calculation simpler.



(sec-emergent-geometry-intro)=
## Introduction

:::{div} feynman-prose
Let me tell you what this section is really about. We are going to ask one of the deepest questions in physics: *where does geometry come from?*

Now, you might think geometry is just "given"—the stage on which physics happens. Space has three dimensions, distances are measured with rulers, and that is that. But this view has been under attack since Einstein showed us that mass curves spacetime, and the attack has only intensified with quantum gravity, where space itself might be emergent from something more fundamental.

Here is the beautiful thing about the Latent Fractal Gas: it gives us a concrete, computable answer to this question. The walkers in the swarm do not know anything about "geometry" at the start. They just diffuse around, following fitness gradients, cloning when they find good spots. But the way they diffuse—anisotropically, adapting to the local curvature of the fitness landscape—*creates* a geometry. The metric tensor does not descend from Mount Sinai; it emerges from the algorithm itself.

The key insight is that diffusion and metric are two sides of the same coin. If you tell me how noise spreads at every point in space, I can compute a metric. If you give me a metric, I can compute how noise should spread. The Latent Fractal Gas ties these together by making the diffusion tensor depend on the Hessian of the fitness function. The result is that the swarm effectively performs a diffusion process *analogous to* Riemannian Brownian motion on an emergent manifold whose geometry is determined by the fitness landscape. (The precise relationship to Laplace-Beltrami Brownian motion involves the geometric drift term; see {ref}`sec-geometric-drift`.)

This is not just mathematically pretty—it is algorithmically powerful. The swarm automatically concentrates exploration where it matters (flat regions with low curvature) and exploits efficiently where the landscape is already well-characterized (peaked regions with high curvature). Geometry and optimization become the same thing.
:::



(sec-adaptive-diffusion-tensor)=
## The Adaptive Diffusion Tensor

The core mechanism of the Latent Fractal Gas is the modulation of exploration noise based on local fitness information. Rather than diffusing isotropically like a drunk stumbling home, each walker adjusts its random steps to match the local terrain.

:::{div} feynman-prose
Here is the key picture to keep in mind. Imagine you are exploring a mountain range blindfolded. In flat meadows, you can take big random steps without worrying about falling off cliffs. But near a sharp ridge, you had better take tiny, careful steps, mostly along the ridge rather than perpendicular to it. That is exactly what the adaptive diffusion tensor does: it reads the local curvature from the Hessian and adjusts the noise accordingly.

The mathematical machinery looks intimidating—inverse matrix square roots and all that—but the physical content is simple: *explore widely where the landscape is flat, tread carefully where it is curved*.
:::

:::{prf:definition} Adaptive Diffusion Tensor
:label: def-adaptive-diffusion-tensor-latent

For a walker at position $z \in \mathcal{Z}$ in the latent space, within a swarm state $S$, the **adaptive diffusion tensor** is defined as:

$$
\Sigma_{\mathrm{reg}}(z, S) = \left( H(z, S) + \epsilon_\Sigma I \right)^{-1/2}
$$

where:

- $H(z, S) = \nabla_z^2 V_{\mathrm{fit}}^{(i)}(z; S)$ is the **local Hessian** of the per-walker fitness potential evaluated at position $z$ (companions and other walkers treated as frozen). Since $V_{\mathrm{fit}} \in C^\infty$ (see {doc}`/source/3_fractal_gas/convergence_program/14_b_geometric_gas_cinf_regularity_full`), the Hessian is symmetric by Schwarz's theorem. In the mean-field limit $N \to \infty$, we may instead use the effective fitness field $V_{\mathrm{fit}}(z; \mu)$ from Definition {prf:ref}`def-mean-field-fitness-field` and set $H(z; \mu) = \nabla_z^2 V_{\mathrm{fit}}(z; \mu)$
- $\epsilon_\Sigma > 0$ is the **regularization parameter** (spectral floor)
- $I$ is the identity matrix in the coordinate basis (we work in coordinates where the latent space is locally Euclidean; for curved ambient spaces, replace $I$ with the ambient metric $G$)
- The matrix square root is the unique symmetric positive definite square root

The induced **diffusion matrix** (covariance of the noise) is:

$$
D_{\mathrm{reg}}(z, S) = \Sigma_{\mathrm{reg}} \Sigma_{\mathrm{reg}}^T = \left( H(z, S) + \epsilon_\Sigma I \right)^{-1}
$$
:::

:::{prf:definition} Mean-Field Fitness Field
:label: def-mean-field-fitness-field

Let $\mu_t$ be the deterministic limiting empirical measure as $N \to \infty$. Define the **effective fitness field**
$V_{\mathrm{fit}}(z; \mu_t)$ by averaging the per-walker fitness over companion selection:

$$
V_{\mathrm{fit}}(z; \mu_t) := \mathbb{E}_{c \sim P_{\mu_t}(z, \cdot)}\!\left[ V_{\mathrm{fit}}^{(i)}(z, c; \mu_t) \right].
$$

Here $P_{\mu_t}(z,\cdot)$ is the companion-selection kernel induced by $\mu_t$ (softmax of algorithmic distance), and
$V_{\mathrm{fit}}^{(i)}(z, c; \mu_t)$ is the same per-walker fitness functional used in the algorithm, with statistics
computed from $\mu_t$ (global if $\rho=\varnothing$, localized if $\rho$ is finite). For finite $N$, the algorithm
samples this field only at the walker locations; in the mean-field limit it becomes a deterministic field on
$\mathcal{Z}$.
:::

### Geometric Interpretation: Diffusion Inverse = Metric Tensor

:::{div} feynman-prose
Now we come to what I think is the most beautiful part of this whole framework. The relationship between diffusion and metric is not an analogy—it is an identity.

Think about what a metric does: it tells you how to measure distances. A large metric component in some direction means that direction is "stretched"—small coordinate changes correspond to large physical distances. Now think about what diffusion does: it tells you how noise spreads. Large diffusion in some direction means noise spreads easily that way.

Here is the key: these are inverses of each other! If a direction is stretched (large metric), noise should *not* spread easily in that direction—the stretched direction represents fine structure that the diffusion should not smear out. Conversely, if a direction is compressed (small metric), noise can spread freely because there is no fine structure to preserve.

So the metric tensor $g$ and the diffusion tensor $D$ satisfy $D = g^{-1}$. This is not a choice we make—it is forced on us by consistency. And the Latent Fractal Gas realizes this by setting:

$$
g(z, S) = H(z, S) + \epsilon_\Sigma I
$$

The fitness Hessian *is* the metric. Geometry emerges from optimization.
:::

We define the **emergent Riemannian metric** on the latent space as:

$$
g(z, S) = H(z, S) + \epsilon_\Sigma I
$$

This identification reveals the geometric logic of the algorithm:

| Fitness Landscape | Hessian Eigenvalues | Metric | Diffusion | Behavior |
|-------------------|---------------------|--------|-----------|----------|
| Sharp peak / valley | Large $\lambda_i$ | Large $g_{ii}$ | Small $D_{ii}$ | **Exploitation** |
| Flat plateau | Small $\lambda_i$ | Small $g_{ii}$ | Large $D_{ii}$ | **Exploration** |
| Saddle point | Mixed signs | After regularization: moderate | Moderate | **Balanced** |

### The Regularization Scale

:::{div} feynman-prose
You might wonder: why do we need this $\epsilon_\Sigma$ term? Can we not just use $g = H$ directly?

The answer is no, and the reason is deeply physical. The Hessian $H$ can have zero eigenvalues (in perfectly flat regions) or even negative eigenvalues (at saddle points or local maxima). A metric with zero or negative eigenvalues is not a metric at all—distances would become zero or imaginary, and the whole geometric picture would collapse.

The regularization $\epsilon_\Sigma I$ acts as a "spectral floor." It guarantees that even in the flattest regions, there is still some minimum stiffness to the geometry. Physically, you can think of it as a thermal energy scale: even with zero fitness curvature, the system still has thermal fluctuations at temperature $T \propto 1/\epsilon_\Sigma$.

This is not a hack to make the math work. It is a statement about the physics: below some scale, the fitness landscape cannot be resolved, and the system defaults to isotropic exploration. The regularization encodes this fundamental limitation.
:::

The parameter $\epsilon_\Sigma > 0$ plays three essential roles:

1. **Mathematical**: Ensures $g(z, S)$ is symmetric positive definite everywhere
2. **Algorithmic**: Acts as a trust-region radius, preventing infinite step sizes in flat regions
3. **Physical**: Sets a thermal/quantum cutoff—the minimum geometric stiffness of spacetime



(sec-regularity-convergence)=
## Regularity and Convergence

A major theoretical challenge with anisotropic, state-dependent diffusion is ensuring the stochastic process remains well-behaved. The specific structure of the Latent Fractal Gas guarantees this through two key properties: **uniform ellipticity** and **Lipschitz continuity**.

(sec-uniform-ellipticity)=
### Uniform Ellipticity

:::{div} feynman-prose
Here is why uniform ellipticity matters. Imagine the diffusion tensor could become arbitrarily small in some direction. Then noise would not spread at all in that direction, and the stochastic process would effectively become deterministic there. The Markov chain would lose its mixing property, and convergence theorems would fail.

Conversely, if diffusion could become arbitrarily large, the process would explode—walkers would jump to infinity in finite time. Neither scenario is acceptable.

Uniform ellipticity says: there exist constants $c_{\min} > 0$ and $c_{\max} < \infty$ such that the diffusion tensor is always sandwiched between $c_{\min} I$ and $c_{\max} I$. This is not an assumption we hope holds—it is a theorem we prove, and the proof is almost trivial given our regularization.
:::

:::{prf:assumption} Spectral Bounds
:label: assump-spectral-floor-latent

There exist constants $\Lambda_- \geq 0$ and $\Lambda_+ < \infty$ such that for all swarm states $S$ and all walker positions $z$ in the accessible region $\mathcal{Z}$:

$$
-\Lambda_- \leq \lambda_{\min}(H(z, S)) \leq \lambda_{\max}(H(z, S)) \leq \Lambda_+
$$

We fix $\epsilon_\Sigma > \Lambda_-$, which ensures that $g(z, S) = H(z, S) + \epsilon_\Sigma I$ is symmetric positive definite for all states. The upper bound $\Lambda_+$ ensures uniform ellipticity from below.

**Verification:** These bounds follow from the Gevrey-1 derivative estimates in {doc}`/source/3_fractal_gas/convergence_program/14_b_geometric_gas_cinf_regularity_full`, which establish k-uniform bounds on all derivatives of $V_{\mathrm{fit}}$. The spectral bounds $\Lambda_\pm$ depend on the regularization parameters $(\rho, \varepsilon_d, \eta_{\min})$ and are independent of swarm size.
:::

:::{prf:theorem} Uniform Ellipticity by Construction
:label: thm-uniform-ellipticity-latent

Under Assumption {prf:ref}`assump-spectral-floor-latent`, the diffusion matrix $D_{\mathrm{reg}}$ is **uniformly elliptic**:

$$
c_{\min} I \preceq D_{\mathrm{reg}}(z, S) \preceq c_{\max} I
$$

where the bounds depend only on $\epsilon_\Sigma$ and the spectral bounds from Assumption {prf:ref}`assump-spectral-floor-latent`, **independent of the number of walkers $N$**:

$$
c_{\min} = \frac{1}{\Lambda_+ + \epsilon_\Sigma}, \quad c_{\max} = \frac{1}{\epsilon_\Sigma - \Lambda_-}
$$
:::

:::{prf:proof}
Let $\{\lambda_k(H)\}$ be the eigenvalues of the Hessian matrix $H(z, S)$. Since $H$ is symmetric, the eigenvalues of the regularized metric $g = H + \epsilon_\Sigma I$ are:

$$
\mu_k = \lambda_k(H) + \epsilon_\Sigma
$$

The diffusion matrix $D_{\mathrm{reg}} = g^{-1}$ has eigenvalues $1/\mu_k$.

**Deriving $c_{\min}$:** The smallest eigenvalue of $D_{\mathrm{reg}}$ corresponds to the largest eigenvalue of $g$:

$$
\lambda_{\min}(D_{\mathrm{reg}}) = \frac{1}{\lambda_{\max}(g)} = \frac{1}{\lambda_{\max}(H) + \epsilon_\Sigma} \geq \frac{1}{\Lambda_+ + \epsilon_\Sigma} =: c_{\min}
$$

**Deriving $c_{\max}$:** The largest eigenvalue of $D_{\mathrm{reg}}$ corresponds to the smallest eigenvalue of $g$. By the spectral floor assumption, $\lambda_k(H) \geq -\Lambda_-$, so:

$$
\lambda_{\min}(g) = \min_k (\lambda_k(H) + \epsilon_\Sigma) \geq \epsilon_\Sigma - \Lambda_- > 0
$$

Therefore:

$$
\lambda_{\max}(D_{\mathrm{reg}}) = \frac{1}{\lambda_{\min}(g)} \leq \frac{1}{\epsilon_\Sigma - \Lambda_-} =: c_{\max}
$$

The matrix inequality $c_{\min} I \preceq D_{\mathrm{reg}} \preceq c_{\max} I$ follows from the eigenvalue bounds.

$\square$
:::

:::{admonition} Why This Makes Everything Work
:class: tip

Uniform ellipticity is the **critical property** enabling convergence:

1. **Non-degeneracy** ($c_{\min} > 0$): Noise is non-degenerate in all directions, essential for hypocoercivity
2. **Boundedness** ($c_{\max} < \infty$): Noise cannot explode, bounding Lyapunov drift expansion terms
3. **N-uniformity**: The bounds depend only on $\epsilon_\Sigma$ and geometry, not on swarm size $N$
:::

(sec-lipschitz-continuity)=
### Lipschitz Continuity

:::{div} feynman-prose
Uniform ellipticity tells us the diffusion tensor is always well-behaved at each point. But for the SDE to have unique solutions, we also need the diffusion to vary *smoothly* as we move through the space. This is Lipschitz continuity: the diffusion tensor cannot change too fast as a function of position.

The proof goes in three steps. First, the Hessian of the fitness function is Lipschitz because the fitness is C∞ with Gevrey-1 bounds (proven in the appendices). Second, the matrix inverse and square root operations are operator-Lipschitz on the space of positive definite matrices. Third, composing Lipschitz functions gives a Lipschitz function. Nothing deep, but the details matter.
:::

:::{prf:proposition} Lipschitz Continuity of Adaptive Diffusion
:label: prop-lipschitz-diffusion-latent

The fitness potential $V_{\mathrm{fit}}$ is $C^\infty$ with Gevrey-1 bounds on all derivatives (see {doc}`/source/3_fractal_gas/convergence_program/14_b_geometric_gas_cinf_regularity_full`); in particular, it is $C^3$ with bounded third derivatives. Therefore, the adaptive diffusion tensor $\Sigma_{\mathrm{reg}}(z, S)$ is Lipschitz continuous:

$$
\|\Sigma_{\mathrm{reg}}(z_1, S_1) - \Sigma_{\mathrm{reg}}(z_2, S_2)\|_F \leq L_\Sigma \cdot d_{\mathrm{alg}}((z_1, S_1), (z_2, S_2))
$$

where $L_\Sigma$ is independent of $N$, and $d_{\mathrm{alg}}$ is the **algorithmic distance** on configuration space defined as:

$$
d_{\mathrm{alg}}((z_1, S_1), (z_2, S_2)) = \|z_1 - z_2\|_2 + W_1(S_1, S_2)
$$

with $W_1$ the 1-Wasserstein distance between swarm empirical measures (see {doc}`/source/3_fractal_gas/1_the_algorithm/02_fractal_gas_latent`).
:::

:::{prf:proof}
We establish Lipschitz continuity in three steps.

**Step 1: Lipschitz continuity of the Hessian**

The fitness potential has mean-field structure:

$$
V_{\mathrm{fit}}(S) = \frac{1}{N} \sum_{i,j} \phi(z_i, z_j)
$$

where $\phi$ is a smooth interaction kernel. The Hessian with respect to walker $i$'s position is:

$$
H_i(S) = \nabla_{z_i}^2 V_{\mathrm{fit}}(S) = \frac{1}{N} \sum_{j=1}^N \nabla_{z_i}^2 \phi(z_i, z_j)
$$

Since $\phi \in C^3$, its Hessian has Lipschitz constant $L_\phi^{(3)}$. The average of Lipschitz functions is Lipschitz with the same constant:

$$
\|H(z_1, S_1) - H(z_2, S_2)\|_F \leq L_H \cdot d_{\mathrm{alg}}((z_1, S_1), (z_2, S_2))
$$

where $L_H = L_\phi^{(3)}$ is **independent of $N$** due to the $1/N$ normalization.

:::{prf:lemma} Operator-Lipschitz bound for inverse square root
:label: lem-operator-lipschitz-inv-sqrt-latent

Let $A, B$ be symmetric positive definite matrices with $A \succeq m I$ and $B \succeq m I$ for some $m > 0$.
Then:

$$
\|A^{-1/2} - B^{-1/2}\|_F \leq \frac{1}{2 m^{3/2}} \|A - B\|_F
$$

In particular, for $A = H_i(S_1) + \epsilon_\Sigma I$ and $B = H_i(S_2) + \epsilon_\Sigma I$ with
$\epsilon_\Sigma > \Lambda_-$, we may take $m = \epsilon_\Sigma - \Lambda_-$.
:::

:::{prf:proof}
Define $F(X, A) = X A X - I$. For each $A \succ 0$ there is a unique SPD solution $X = A^{-1/2}$.
Along the segment $A_t = A + t(B - A)$ let $X_t = A_t^{-1/2}$. Differentiating $F(X_t, A_t) = 0$ gives

$$
L_t(\dot{X}_t) = -X_t (B - A) X_t, \quad L_t(H) = H A_t X_t + X_t A_t H.
$$

Since $A_t$ and $X_t$ commute, in a basis where $A_t$ is diagonal we have
$(L_t(H))_{ij} = (\lambda_i^{1/2} + \lambda_j^{1/2}) H_{ij}$, so
$\|L_t^{-1}\|_{F \to F} \leq (2 m^{1/2})^{-1}$. Also $\|X_t\|_{\mathrm{op}} \leq m^{-1/2}$, hence

$$
\|\dot{X}_t\|_F \leq \frac{1}{2 m^{3/2}} \|B - A\|_F.
$$

Integrating from $t=0$ to $t=1$ yields the bound.
$\square$
:::

**Step 2: Operator-Lipschitz property of matrix inverse square root**

Set $A = H_i(S_1) + \epsilon_\Sigma I$ and $B = H_i(S_2) + \epsilon_\Sigma I$. By the spectral floor
$\lambda_{\min}(H_i) \geq -\Lambda_-$, we have $A, B \succeq (\epsilon_\Sigma - \Lambda_-) I$.
Lemma {prf:ref}`lem-operator-lipschitz-inv-sqrt-latent` then gives:

$$
\|(H_i(S_1) + \epsilon_\Sigma I)^{-1/2} - (H_i(S_2) + \epsilon_\Sigma I)^{-1/2}\|_F
\leq K_{\mathrm{sqrt}} \|H_i(S_1) - H_i(S_2)\|_F
$$

with:

$$
K_{\mathrm{sqrt}} = \frac{1}{2(\epsilon_\Sigma - \Lambda_-)^{3/2}}.
$$

**Step 3: Composition**

Composing the two Lipschitz maps:

$$
\|\Sigma_{\mathrm{reg}}(z_1, S_1) - \Sigma_{\mathrm{reg}}(z_2, S_2)\|_F \leq K_{\mathrm{sqrt}} \cdot L_H \cdot d_{\mathrm{alg}}((z_1, S_1), (z_2, S_2))
$$

Setting $L_\Sigma = K_{\mathrm{sqrt}} \cdot L_H$ completes the proof.

$\square$
:::



(sec-equivalence-principle)=
## The Equivalence Principle: Flat vs Curved

:::{div} feynman-prose
Now we arrive at something that confused me for a long time, until I realized it is actually simple. You can describe the same physics in two completely different ways:

**Way 1 (The Algorithmic View):** Space is flat. Distances are Euclidean. But the diffusion is weird—it is anisotropic and state-dependent, spreading faster in some directions than others.

**Way 2 (The Geometric View):** Diffusion is nice and isotropic—the same in all directions. But space itself is curved. The metric varies from point to point, stretching and compressing distances.

These are not two different theories. They are the same theory in different clothes. It is like choosing coordinates: you can describe the surface of a sphere using latitude and longitude (curved, but the metric is complicated) or embed it in 3D Euclidean space (flat ambient space, but the surface is curved). Same sphere, different descriptions.

For proofs, the flat-space view is simpler because you can use standard Stratonovich calculus without worrying about Christoffel symbols. For intuition, the curved-space view is often better because it makes the geometry explicit. The beautiful thing is that convergence rates, mixing times, and all physical observables are the same in both views.
:::

:::{prf:observation} Two Equivalent Perspectives
:label: obs-two-perspectives-latent

The dynamics of the Latent Fractal Gas admit two mathematically equivalent formulations:

**1. Algorithmic View (Flat Space with Anisotropic Diffusion)**
- State space: Latent space $\mathcal{Z}$ with ambient metric $G$
- Diffusion: Anisotropic tensor $\Sigma_{\mathrm{reg}}(z, S)$
- SDE (Stratonovich): $dv = [F(z) - \gamma v] dt + \Sigma_{\mathrm{reg}}(z, S) \circ dW$
- *Intuition:* Walking through a room where air viscosity varies with location

**2. Geometric View (Curved Space with Isotropic Diffusion)**
- State space: Riemannian manifold $(\mathcal{M}, g)$ with $g = H + \epsilon_\Sigma I$
- Diffusion: Isotropic (constant coefficient relative to $g$)
- SDE (Manifold): $dv = [\tilde{F}_g(z) - \gamma v] dt + \sigma \sqrt{g^{-1}} \circ dW_{\mathcal{M}}$
- *Intuition:* Walking through a warped room where distances themselves change

**Generator-level equivalence (precise):** The two descriptions are the same Markov generator on $\mathcal{Z}$ once the
geometric drift $b_{\mathrm{geo}}$ is included (Lemma {prf:ref}`lem-geometric-drift-latent`). In this sense, anisotropic
diffusion in flat coordinates is equivalent to isotropic diffusion on $(\mathcal{Z}, g)$—no global diffeomorphism is
assumed or required. If a coordinate change $\Psi$ is used, it must satisfy $g=\Psi^*G$ in the chart, and the drift must
transform accordingly.
:::

:::{prf:remark} Equivalence as a Reinterpretation (Not a Global Diffeomorphism)
:label: rem-equivalence-reinterpretation

The equivalence invoked here is **generator-level**: the same stochastic process on $\mathcal{Z}$ can be written either
with anisotropic diffusion $D_{\mathrm{reg}}$ in Euclidean coordinates or with isotropic diffusion relative to the
metric $g$ and the corresponding geometric drift. This does **not** claim a global diffeomorphism that isotropizes an
arbitrary diffusion field; any coordinate-change statement is local and requires $g$ to be a pullback metric.
:::



(sec-kinetic-evolution)=
## Kinetic Evolution in Emergent Geometry

The walkers evolve according to underdamped Langevin dynamics on the emergent Riemannian manifold. The key subtlety is the choice between Ito and Stratonovich interpretations.

(sec-stratonovich-formulation)=
### The Stratonovich Formulation

:::{div} feynman-prose
Here is something that trips up many people. When you have a stochastic differential equation with state-dependent noise, the answer depends on how you interpret the product of noise and diffusion tensor. The two standard choices are Ito (evaluate the diffusion at the start of the step) and Stratonovich (evaluate it at the midpoint).

For most applications, Ito is more convenient because martingale calculus works nicely. But for geometric problems, Stratonovich is essential. The reason is that Stratonovich respects the chain rule: if you change coordinates, the SDE transforms the way you expect from calculus. Ito does not—it picks up extra "spurious drift" terms that depend on your coordinate choice.

Since our whole point is that the geometry is emergent and coordinate-independent, we must use Stratonovich. The physics should not depend on how we label points in space.
:::

The kinetic operator evolves the walker state $w_i = (z_i, v_i, s_i)$ according to:

$$
\begin{aligned}
dz_i &= v_i \, dt \\
dv_i &= \left[ F(z_i) - \gamma v_i \right] dt + \Sigma_{\mathrm{reg}}(z_i, S) \circ dW_i
\end{aligned}
$$

where:
- $F(z) = -\nabla \Phi_{\mathrm{eff}}(z)$ is the force from the effective potential
- $\gamma > 0$ is the friction coefficient
- $\circ$ denotes the **Stratonovich product**
- $W_i$ are independent standard Brownian motions

**Why Stratonovich is essential:**
1. **Chain rule works:** Stratonovich SDEs transform correctly under coordinate changes
2. **Geometric invariance:** Results do not depend on coordinate choice
3. **Physical consistency:** No spurious drift terms that obscure the physics

(sec-geometric-drift)=
### The Geometric Drift Term

For the system to converge to the correct Riemannian equilibrium measure $\rho(z) \propto \sqrt{\det g(z)} e^{-\Phi_{\mathrm{eff}}(z)/T}$, we must include a **geometric drift** term in the dynamics.

:::{admonition} Important Clarification
:class: warning

In underdamped Langevin dynamics where noise acts only on velocity ($dz = v\,dt$ with no stochastic term), the Stratonovich and Ito interpretations **coincide** for the position variable. The diffusion tensor $\Sigma(z)$ depends only on $z$, and since $dz$ has bounded variation ($dz \cdot dz = 0$), there is no Stratonovich-to-Ito correction arising from stochastic calculus.

However, to ensure the system relaxes to the correct **Riemannian measure** (with the $\sqrt{\det g}$ weighting), we must *add* a geometric drift term. This is not a calculus artifact—it is a physical requirement for sampling from the target distribution.
:::

:::{prf:lemma} Geometric Drift for Riemannian Measure
:label: lem-geometric-drift-latent

To ensure convergence to the Riemannian equilibrium $\rho(z) \propto \sqrt{\det g(z)} e^{-\Phi_{\mathrm{eff}}(z)/T}$, we augment the velocity dynamics with a **geometric drift**:

$$
dv_i = \left[ F(z_i) - \gamma v_i + b_{\mathrm{geo}}(z_i) \right] dt + \Sigma_{\mathrm{reg}}(z_i, S) \circ dW_i
$$

where the geometric drift is:

$$
b_{\mathrm{geo}}^k(z) = \frac{T}{2}\,\frac{1}{\sqrt{\det g(z)}}\partial_{z_l}\!\left(\sqrt{\det g(z)}\,g^{kl}(z)\right)
= \frac{T}{2}\Big[(\nabla_z \cdot D_{\mathrm{reg}}(z))^k + (D_{\mathrm{reg}}(z)\,\nabla_z \log \sqrt{\det g(z)})^k\Big]
$$

**Physical interpretation:** This drift compensates for the spatially-varying diffusion **and** the Riemannian volume element, ensuring that the equilibrium density includes the $\sqrt{\det g}$ Jacobian factor. Without it, the stationary distribution would be $\rho(z) \propto e^{-\Phi_{\mathrm{eff}}(z)/T}$ without geometric weighting.

**Bound:** Under uniform ellipticity ($\|\Sigma_{\mathrm{reg}}\|_{\mathrm{op}} \leq c_{\max}^{1/2}$), Lipschitz continuity
($\|\nabla \Sigma_{\mathrm{reg}}\|_{\mathrm{op}} \leq L_\Sigma$), and bounded log-volume gradient
($\|\nabla \log \sqrt{\det g}\|_2 \leq L_{\log\det}$):

$$
\|b_{\mathrm{geo}}(z)\|_2 \leq \frac{T}{2}\left(d^{3/2}\sqrt{c_{\max}}\,L_\Sigma + c_{\max}\,L_{\log\det}\right)
$$
:::

:::{div} feynman-prose
Let me explain why this drift term is necessary. Imagine you have a region where the diffusion tensor changes—say, noise is stronger on the left than on the right. Walkers diffusing from the left bring more randomness; walkers from the right bring less. This asymmetry creates a net drift toward the quieter side.

If we want the equilibrium to reflect the *intrinsic* Riemannian geometry (where the $\sqrt{\det g}$ factor appears), we must counteract this effect. The geometric drift does exactly that: it pushes walkers *toward* regions of higher diffusion, balancing the natural accumulation in quiet regions.

The key insight is that this is not a bug in our formalism—it is a feature. By choosing whether to include this drift, we can select which measure we want to sample: coordinate measure (no drift) or Riemannian measure (with drift). For geometric applications, we want the Riemannian measure.
:::



(sec-riemannian-volume)=
## Riemannian Volume Elements and Integration

:::{prf:remark} Drift convention for Riemannian weighting
:label: rem-riemannian-volume-drift

Throughout this section we assume the geometric drift $b_{\mathrm{geo}}$ from {ref}`sec-geometric-drift` is included in
the velocity dynamics (Lemma {prf:ref}`lem-geometric-drift-latent`). This is the regime in which the QSD density carries
the $\sqrt{\det g}$ factor.
:::

The emergent metric defines a natural notion of volume that differs from the coordinate (Lebesgue) volume. This is crucial for computing probabilities, entropies, and expectation values.

(sec-volume-element)=
### Volume Element Definition

:::{prf:definition} Riemannian Volume Element
:label: def-riemannian-volume-element-latent

Let $(\mathcal{Z}, g)$ be the emergent Riemannian manifold with metric $g(z, S) = H(z, S) + \epsilon_\Sigma I$. The **Riemannian volume element** at point $z \in \mathcal{Z}$ is:

$$
dV_g(z) = \sqrt{\det g(z, S)} \, dz
$$

where $dz = dz_1 \wedge \cdots \wedge dz_d$ is the coordinate (Lebesgue) volume element.

**Physical interpretation:**
- $\sqrt{\det g(z, S)}$: Jacobian factor relating coordinate volume to intrinsic volume
- Large $\sqrt{\det g}$: "Stretched" region (high curvature), hard to explore
- Small $\sqrt{\det g}$: "Compressed" region (low curvature), easy to explore
:::

:::{prf:remark} Regime for the Volume Element
:label: rem-volume-element-regime

In the analytic statements below, we take the **mean-field regime** and interpret $g(z,S)$ as the deterministic field
$g(z;\mu)$ induced by Definition {prf:ref}`def-mean-field-fitness-field`. For finite $N$, the algorithm evaluates
$g(z,S)$ only at walker locations and treats $S$ as frozen within a step (see {ref}`sec-eg-stage2`); the mean-field
expressions describe the limiting field that these samples approximate.
:::

:::{admonition} Two Proof Routes for the Regime Choice
:class: dropdown note

**Route A — Mean-field/propagation-of-chaos (global field):**
- Propagation of chaos gives convergence of single-particle marginals to a deterministic $\mu$ and justifies $g(z;\mu)$:
  {prf:ref}`thm-propagation-chaos-qsd` in {doc}`/source/3_fractal_gas/convergence_program/12_qsd_exchangeability_theory`.
- The hypostructure proof object records the mean-field limit and error bound in
  {doc}`/source/3_fractal_gas/1_the_algorithm/02_fractal_gas_latent` (Part III-B: Mean-Field Limit).

**Route B — Frozen-swarm within a step (local field):**
- The Euclidean Gas kernel freezes $V_{\mathrm{fit}}$ during each step; this provides a fixed-$S$ geometry for the
  kinetic update ({ref}`sec-eg-stage2` in {doc}`/source/3_fractal_gas/convergence_program/02_euclidean_gas`).
:::

:::{div} feynman-prose
Here is something that should make you sit up. The quasi-stationary distribution (QSD) of the swarm is not uniform in coordinate space—it is weighted by $\sqrt{\det g}$. This means episodes naturally sample from the Riemannian volume measure, not the coordinate measure.

Why does this happen? There are two complementary perspectives:

**From diffusion:** In regions of high metric (large $\det g$), the diffusion is small, so walkers tend to accumulate. In regions of low metric, diffusion is large, so walkers spread out.

**From geometric drift:** The geometric drift term $b_{\mathrm{geo}}$ (see {ref}`sec-geometric-drift`) actively compensates for diffusion gradients, ensuring the equilibrium includes the $\sqrt{\det g}$ Jacobian factor. Without this drift, the stationary distribution would lack the geometric weighting.

The interplay of these effects produces sampling proportional to $\sqrt{\det g} \cdot e^{-\Phi_{\mathrm{eff}}/T}$—exactly the Riemannian measure we want.

This is extremely useful for Monte Carlo integration: if you want to compute Riemannian integrals, just average over episodes. The $\sqrt{\det g}$ weighting is automatic (provided the geometric drift is included in the dynamics).
:::

(sec-fan-triangulation)=
### Fan Triangulation for Riemannian Areas

For computing areas of 2D surfaces (plaquettes) in the emergent geometry:

:::{prf:algorithm} Fan Triangulation for Riemannian Area
:label: alg-fan-triangulation-latent

**Input:**
- Ordered cycle of walkers $C = (e_0, e_1, \ldots, e_{n-1}, e_0)$ with positions $z_i \in \mathcal{Z}$
- Metric function $g: \mathcal{Z} \to \mathbb{R}^{d \times d}$

**Output:** Riemannian area $A_g(C)$ of the enclosed surface

**Procedure:**

**Step 1.** Compute centroid: $z_c = \frac{1}{n} \sum_{i=0}^{n-1} z_i$

**Step 2.** Evaluate metric at centroid: $g_c = g(z_c)$

**Step 3.** For each triangle $T_i = (z_c, z_i, z_{i+1})$, compute edge vectors:

$$
v_1^{(i)} = z_i - z_c, \quad v_2^{(i)} = z_{i+1} - z_c
$$

**Step 4.** Compute Riemannian inner products:

$$
\langle v_1, v_1 \rangle_{g_c} = (v_1^{(i)})^T g_c v_1^{(i)}, \quad \langle v_2, v_2 \rangle_{g_c} = (v_2^{(i)})^T g_c v_2^{(i)}, \quad \langle v_1, v_2 \rangle_{g_c} = (v_1^{(i)})^T g_c v_2^{(i)}
$$

**Step 5.** Triangle area via Gram determinant:

$$
A_i = \frac{1}{2} \sqrt{\langle v_1, v_1 \rangle_{g_c} \cdot \langle v_2, v_2 \rangle_{g_c} - \langle v_1, v_2 \rangle_{g_c}^2}
$$

**Step 6.** Sum: $A_g(C) = \sum_{i=0}^{n-1} A_i$

**Complexity:** $O(n \cdot d^2)$

**Error:** $O(\mathrm{diam}(C)^2)$ for smooth $g$
:::

```python
import numpy as np
from typing import Callable

def compute_riemannian_area_fan(
    vertices: np.ndarray,  # Shape: (n, d) - cycle vertices
    metric: Callable[[np.ndarray], np.ndarray],  # z -> g(z)
) -> float:
    """
    Compute Riemannian area of cycle using fan triangulation.

    Parameters
    ----------
    vertices : np.ndarray, shape (n, d)
        Ordered cycle vertices [z_0, ..., z_{n-1}]
    metric : Callable
        Function z -> g(z) returning metric tensor at z

    Returns
    -------
    area : float
        Total Riemannian area
    """
    n, d = vertices.shape

    # Centroid
    z_c = np.mean(vertices, axis=0)
    g_c = metric(z_c)

    # Sum triangle areas
    total_area = 0.0
    for i in range(n):
        v1 = vertices[i] - z_c
        v2 = vertices[(i + 1) % n] - z_c

        # Gram matrix entries
        g11 = v1 @ g_c @ v1
        g22 = v2 @ g_c @ v2
        g12 = v1 @ g_c @ v2

        # Area from Gram determinant
        discriminant = g11 * g22 - g12**2
        if discriminant > 0:
            total_area += 0.5 * np.sqrt(discriminant)

    return total_area
```

(sec-tetrahedral-decomposition)=
### Tetrahedral Decomposition for Volumes

:::{prf:definition} Riemannian Volume of Tetrahedron
:label: def-tetrahedron-volume-latent

Let $T = (z_0, z_1, z_2, z_3)$ be a tetrahedron with vertices $z_i \in \mathcal{Z}$.

**Metric at centroid:**

$$
g_c = g\left(\frac{z_0 + z_1 + z_2 + z_3}{4}\right)
$$

**Edge vectors from base vertex:**

$$
v_1 = z_1 - z_0, \quad v_2 = z_2 - z_0, \quad v_3 = z_3 - z_0
$$

**Gram matrix:** $3 \times 3$ matrix of Riemannian inner products:

$$
G_{ij} = \langle v_i, v_j \rangle_g = v_i^T g_c v_j
$$

**Riemannian volume:**

$$
V_g(T) = \frac{1}{6} \sqrt{\det G}
$$
:::

```python
def compute_riemannian_volume_tetrahedron(
    vertices: np.ndarray,  # Shape: (4, d)
    metric: Callable[[np.ndarray], np.ndarray],
) -> float:
    """
    Compute Riemannian volume of tetrahedron.

    Parameters
    ----------
    vertices : np.ndarray, shape (4, d)
        Four vertices [z_0, z_1, z_2, z_3]
    metric : Callable
        z -> g(z)

    Returns
    -------
    volume : float
        Riemannian 3-volume
    """
    assert vertices.shape[0] == 4

    # Centroid and metric
    z_c = np.mean(vertices, axis=0)
    g_c = metric(z_c)

    # Edge vectors from base
    v1 = vertices[1] - vertices[0]
    v2 = vertices[2] - vertices[0]
    v3 = vertices[3] - vertices[0]

    # Gram matrix
    V = np.array([v1, v2, v3])  # (3, d)
    G = V @ g_c @ V.T  # (3, 3)

    # Volume
    det_G = np.linalg.det(G)
    return (1.0 / 6.0) * np.sqrt(max(det_G, 0.0))
```

(sec-monte-carlo-integration)=
### Monte Carlo Integration with QSD Sampling

We continue under the drift convention of Remark {prf:ref}`rem-riemannian-volume-drift`.

:::{div} feynman-prose
Here is a beautiful fact that makes practical computation much easier. If you have a function $f(z)$ and you want to compute its integral over the Riemannian manifold:

$$
I[f] = \int_{\mathcal{Z}} f(z) \, dV_g(z) = \int_{\mathcal{Z}} f(z) \sqrt{\det g(z)} \, dz
$$

you could try to compute the $\sqrt{\det g}$ factor explicitly everywhere. But that is expensive. Instead, just notice that episodes from the QSD already sample from a density proportional to $\sqrt{\det g(z)} e^{-\Phi_{\mathrm{eff}}(z)/T}$. So:

$$
I[f] \approx \frac{Z}{N_{\mathrm{episodes}}} \sum_{i=1}^{N_{\mathrm{episodes}}} f(z_i) \cdot e^{\Phi_{\mathrm{eff}}(z_i)/T}
$$

The geometry is baked into the sampling—you do not need to compute determinants explicitly.
:::

:::{prf:proposition} Monte Carlo Integration with Riemannian Measure
:label: prop-monte-carlo-riemannian-latent

Let $\{z_i\}_{i=1}^N$ be positions sampled from the QSD with density
$\rho(z) \propto \sqrt{\det g(z)} e^{-\Phi_{\mathrm{eff}}(z)/T}$ for the drift-augmented dynamics
(Remark {prf:ref}`rem-riemannian-volume-drift`). Existence and uniqueness of the QSD is established in
{doc}`/source/3_fractal_gas/convergence_program/07_discrete_qsd`; convergence to the QSD is proven in
{doc}`/source/3_fractal_gas/convergence_program/06_convergence`.

**Method 1 (QSD sampling):** If episodes sample from QSD:

$$
\int_{\mathcal{Z}} f(z) \, dV_g(z) \approx Z \cdot \frac{1}{N} \sum_{i=1}^N f(z_i) \cdot e^{\Phi_{\mathrm{eff}}(z_i)/T}
$$

**Method 2 (Importance sampling):** For arbitrary sampling density $\rho(z)$:

$$
\int_{\mathcal{Z}} f(z) \, dV_g(z) \approx \frac{1}{N} \sum_{i=1}^N f(z_i) \cdot \frac{\sqrt{\det g(z_i)}}{\rho(z_i)}
$$

**Convergence rate:** $O(N^{-1/2})$ regardless of dimension (see {doc}`/source/3_fractal_gas/convergence_program/13_quantitative_error_bounds` for explicit error bounds).
:::



(sec-hypocoercivity)=
## Hypocoercivity in Anisotropic Geometry

:::{div} feynman-prose
Now we come to the main technical result: proving that the Latent Fractal Gas actually converges, despite all this anisotropic complexity.

The standard tool is hypocoercivity, which handles situations where the noise acts directly only on some variables (velocities) but not others (positions). The position variables feel the noise indirectly, through the coupling $\dot{z} = v$. The question is: does anisotropic, state-dependent noise break this indirect transmission?

The answer is no, and the reason is uniform ellipticity. As long as the diffusion tensor is bounded away from zero in all directions, the coupling still works. The noise might spread preferentially in some directions, but it spreads *enough* in every direction to drive the system toward equilibrium.

The convergence rate does depend on the geometry: $\kappa \propto \min(\gamma, c_{\min})$. The system converges as fast as its "slowest" direction allows. But it does converge.
:::

The central convergence proof relies on **hypocoercivity**—transferring dissipation from velocity to position variables through the kinematic coupling.

:::{prf:definition} Hypocoercive Norm
:label: def-hypocoercive-norm-latent

For the coupled error $(\Delta z, \Delta v)$ between two swarm copies, define the **hypocoercive norm**:

$$
\|(\Delta z, \Delta v)\|_h^2 = \|\Delta z\|^2 + \lambda_v \|\Delta v\|^2 + b \langle \Delta z, \Delta v \rangle
$$

where $\lambda_v > 0$ and $b \in \mathbb{R}$ satisfy the **coercivity condition** $|b| < 2\sqrt{\lambda_v}$, ensuring the quadratic form is positive definite. This condition follows from requiring the matrix

$$
\begin{pmatrix} 1 & b/2 \\ b/2 & \lambda_v \end{pmatrix}
$$
to be positive definite, i.e., $\lambda_v - b^2/4 > 0$.
:::

:::{prf:theorem} Hypocoercive Contraction with Anisotropic Diffusion
:label: thm-hypocoercive-anisotropic

Assume uniform ellipticity ($D_{\mathrm{reg}} \succeq c_{\min} I$), Lipschitz continuity
($\|\nabla \Sigma_{\mathrm{reg}}\| \leq L_\Sigma$), and the Foster-Lyapunov/minorization hypotheses of
{doc}`/source/3_fractal_gas/convergence_program/06_convergence` (built from
{doc}`/source/3_fractal_gas/convergence_program/03_cloning` and
{doc}`/source/3_fractal_gas/convergence_program/05_kinetic_contraction`). Then the Latent Fractal Gas exhibits geometric
ergodicity with rate:

$$
\kappa_{\mathrm{total}} = O\left(\min\left\{\gamma \tau, \kappa_z^{\mathrm{clone}}, c_{\min} \underline{\lambda} - C_1 L_\Sigma\right\}\right) > 0
$$

where:
- $\gamma$ is friction coefficient
- $\tau$ is kinetic time step
- $\kappa_z^{\mathrm{clone}}$ is cloning contraction rate (see {doc}`/source/3_fractal_gas/convergence_program/03_cloning`)
- $c_{\min}$ is ellipticity lower bound (Theorem {prf:ref}`thm-uniform-ellipticity-latent`)
- $\underline{\lambda}$ is the coercivity constant of the hypocoercive quadratic form
- $C_1$ is a geometry-dependent constant

**Condition for hypocoercive contraction:** $c_{\min} \underline{\lambda} > C_1 L_\Sigma$ (in addition to the QSD
hypotheses above).

**Full proof:** See {doc}`/source/3_fractal_gas/convergence_program/05_kinetic_contraction` for kinetic drift analysis and {doc}`/source/3_fractal_gas/convergence_program/06_convergence` for the complete convergence theorem.
:::

:::{prf:proof}
*Sketch.* We follow the standard hypocoercive framework adapted for state-dependent diffusion.

**Step 1. Lyapunov functional:** Define

$$
\mathcal{H}(\Delta z, \Delta v) = \|\Delta z\|^2 + \lambda_v \|\Delta v\|^2 + b\langle \Delta z, \Delta v \rangle
$$

where $\lambda_v > 0$ and $|b| < 2\sqrt{\lambda_v}$ ensure $\mathcal{H}$ is equivalent to $\|\Delta z\|^2 + \|\Delta v\|^2$.

**Step 2. Time derivative:** Applying Ito's lemma to coupled trajectories:

$$
\frac{d}{dt}\mathbb{E}[\mathcal{H}] = -2\gamma \lambda_v \|\Delta v\|^2 + b\|\Delta v\|^2 - b\gamma\langle \Delta z, \Delta v\rangle + \text{diffusion terms}
$$

**Step 3. Diffusion contribution:** The diffusion terms involve $\mathrm{tr}[D_{\mathrm{reg}}]$ bounded by $d \cdot c_{\max}$, plus cross-terms from spatial variation bounded by $L_\Sigma$.

**Step 4. Gronwall closure:** Choosing $b = \gamma/2$ and $\lambda_v = 1 + \gamma^2/4$ (standard hypocoercive tuning), and using uniform ellipticity to bound the diffusion contributions:

$$
\frac{d}{dt}\mathbb{E}[\mathcal{H}] \leq -\kappa_{\mathrm{hypo}} \mathbb{E}[\mathcal{H}] + C_1 L_\Sigma \mathbb{E}[\|\Delta z\|^2]
$$

where $\kappa_{\mathrm{hypo}} = O(\min\{\gamma, c_{\min}\})$.

**Step 5. Combined contraction:** Including cloning-induced contraction $\kappa_z^{\mathrm{clone}}$ in position space:

$$
\frac{d}{dt}\mathbb{E}[\mathcal{H}] \leq -(\kappa_{\mathrm{hypo}} + \kappa_z^{\mathrm{clone}} - C_1 L_\Sigma) \mathbb{E}[\mathcal{H}]
$$

The condition $c_{\min}\underline{\lambda} > C_1 L_\Sigma$ ensures the total rate is positive.

$\square$
:::

:::{div} feynman-prose
The condition $c_{\min} \underline{\lambda} > C_1 L_\Sigma$ has a clear physical interpretation. The left side measures how strongly the diffusion drives mixing (proportional to the minimum eigenvalue $c_{\min}$). The right side measures how much the geometry varies spatially (the Lipschitz constant $L_\Sigma$). Convergence requires that mixing dominates spatial variation—the diffusion must be strong enough to overcome the geometric complexity.

For the Latent Fractal Gas with regularization $\epsilon_\Sigma$, this condition is always satisfiable by choosing $\epsilon_\Sigma$ large enough. Larger $\epsilon_\Sigma$ increases $c_{\min}$ (more diffusion everywhere) and decreases $L_\Sigma$ (smoother variation). The price is reduced adaptation to local curvature—you move toward isotropic diffusion. The optimal $\epsilon_\Sigma$ balances convergence speed against geometric adaptation.
:::



(sec-emergent-geometry-summary)=
## Summary

:::{div} feynman-prose
Let me summarize what we have accomplished in this section. We started with a swarm of walkers diffusing according to local fitness curvature, and we showed that this simple algorithmic rule creates a full Riemannian geometry. The metric tensor, the volume element, the geodesic structure—all emerge from the diffusion.

The key results are:

1. **Geometry from optimization:** The emergent metric $g = H + \epsilon_\Sigma I$ ties together the fitness landscape (Hessian $H$) and the exploration dynamics (diffusion $D_{\mathrm{reg}} = g^{-1}$). Curved fitness creates curved space.

2. **Rigorous convergence:** Despite the anisotropy, uniform ellipticity guarantees that the system converges to a unique quasi-stationary distribution. The geometry helps rather than hinders.

3. **Practical computation:** The QSD automatically samples from the Riemannian volume measure, making Monte Carlo integration straightforward.

4. **Two equivalent views:** You can think of the system as flat space with weird diffusion, or curved space with nice diffusion. Same physics, different coordinates.

In the next section, we will see how the discrete events of cloning create a tessellation of spacetime—the scutoid geometry that complements this continuous Riemannian picture.
:::

:::{prf:remark} Convergence hypotheses
:label: rem-emergent-geometry-convergence-hypotheses

The convergence statements in this section use the full Foster-Lyapunov and minorization hypotheses from the convergence
program (see {doc}`/source/3_fractal_gas/convergence_program/06_convergence`). Uniform ellipticity and Lipschitz
continuity are necessary ingredients but not sufficient on their own.
:::

:::{admonition} Key Takeaways
:class: tip

**Emergent Geometry Framework:**

| Concept | Mathematical Object | Physical Meaning |
|---------|---------------------|------------------|
| Emergent metric | $g(z, S) = H(z, S) + \epsilon_\Sigma I$ | Local ruler on fitness landscape |
| Diffusion tensor | $D_{\mathrm{reg}} = g^{-1}$ | Noise spreading rate |
| Volume element | $dV_g = \sqrt{\det g} \, dz$ | Intrinsic "size" of regions |
| Regularization | $\epsilon_\Sigma$ | Thermal/quantum cutoff |

**Convergence Guarantees:**

1. Uniform ellipticity: $c_{\min} I \preceq D_{\mathrm{reg}} \preceq c_{\max} I$ (N-independent)
2. Lipschitz continuity: $\|\nabla \Sigma_{\mathrm{reg}}\| \leq L_\Sigma$ (N-independent)
3. Hypocoercive rate: $\kappa \propto \min(\gamma, c_{\min})$
4. Foster-Lyapunov + minorization hypotheses (see {doc}`/source/3_fractal_gas/convergence_program/06_convergence`)

**Practical Implications:**

- Episodes sample from Riemannian measure automatically
- Fan triangulation computes plaquette areas with $O(h^2)$ error
- Monte Carlo integration works with $O(N^{-1/2})$ convergence regardless of dimension
:::



(sec-emergent-geometry-references)=
## References

### Framework Documents

- {doc}`/source/3_fractal_gas/1_the_algorithm/01_algorithm_intuition` — per-walker fitness definition and mean-field field remark
- {doc}`/source/3_fractal_gas/1_the_algorithm/02_fractal_gas_latent` — Latent Fractal Gas algorithm definition and mean-field limit fitness remark
- {doc}`/source/3_fractal_gas/2_fractal_set/01_fractal_set` — Fractal Set data structure
- {doc}`/source/3_fractal_gas/3_fitness_manifold/02_scutoid_spacetime` — Scutoid tessellation and discrete spacetime geometry

### Regularity and Convergence Proofs

- {doc}`/source/3_fractal_gas/convergence_program/14_b_geometric_gas_cinf_regularity_full` — C∞ regularity of fitness potential with Gevrey-1 bounds
- {doc}`/source/3_fractal_gas/convergence_program/14_a_geometric_gas_c3_regularity` — C³ regularity (simplified model)
- {doc}`/source/3_fractal_gas/convergence_program/03_cloning` — Cloning operator contraction analysis
- {doc}`/source/3_fractal_gas/convergence_program/05_kinetic_contraction` — Hypocoercivity and kinetic drift
- {doc}`/source/3_fractal_gas/convergence_program/06_convergence` — Full convergence theorem
- {doc}`/source/3_fractal_gas/convergence_program/07_discrete_qsd` — Quasi-stationary distribution existence and uniqueness
- {doc}`/source/3_fractal_gas/convergence_program/13_quantitative_error_bounds` — Quantitative error bounds for Monte Carlo estimates
